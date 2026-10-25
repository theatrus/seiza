> Implementation status: `balance_narrowband` and `screen_rgb_stars` are
> implemented in `seiza-stacking`, the C ABI, Python and the CLI. Both operate
> on prepared display samples; hosts own alignment, calibration and external
> star separation. Linear continuum fitting, measurement and signed subtraction
> are also implemented across all four surfaces. Combined LRGB+SHO composition
> and applying a fit to an associated starless revision remain host integration work.

# Color composition from mono stacks

Seiza composes calibrated mono stacks into RGB images. The pure Rust and Python
color functions consume already aligned `LinearImage`/array values and return
three-channel `f32`. The CLI first registers every non-reference filter stack
onto L (LRGB), R (RGB), or H-alpha (narrowband) using the same bounded
star/similarity registration and resampling as the stacker. `--no-register`
is the explicit fast path for masters already on one pixel grid. Display
stretching remains outside the stack accumulator.

## Input scaling

The default quick-look path independently maps each filter's robust 0.1% and
99.5% levels to `[0, 1]`. Percentiles are estimated from at most one million
samples per channel, bounding scratch memory for large frames. This is an
affine black/scale match for visualization, not photometric color calibration.
Use `ColorNormalization::None` or CLI `--normalization none` when the masters
are already background-subtracted and intensity-matched. Because the published
Foraxx expressions operate on normalized display values, its inputs must also
already lie in `[0, 1]` when normalization is disabled; Seiza rejects
sensor-unit Foraxx inputs instead of silently clipping them. The default
percentile normalization handles those inputs directly.

Embedding applications may instead apply an independent ordered stretch stack
to each aligned mono channel, then set `ColorOptions::input_transfer` to
`ColorTransfer::DisplayReferred` with normalization disabled. RGB, LRGB, and
static narrowband compositions preserve that transfer in their result. Foraxx
uses the prepared values directly and skips its shared internal Auto-MTF pass,
so different H-alpha, OIII, and SII parameters are neither collapsed nor
applied twice. The caller remains responsible for ensuring finite input samples
are display-scaled consistently.

Composition reads prepared values directly from caller-owned input planes and
allocates one three-plane output; it does not materialize normalized copies of
every channel. Percentile estimation temporarily holds at most `max_samples`
finite values for one channel at a time.

## Cropping to the shared area

Registering one filter stack onto another leaves `NaN` where the source frame
did not reach, so an uncropped composition keeps blank edges. `ColorOptions::crop`
(CLI `--crop`, Python `crop=`) trims the result to the area every channel
covers. A pixel counts as covered when every sample of every required channel
there is finite, so the kept region is the inner area common to all of them:

| Mode | Keeps |
| --- | --- |
| `none` (default) | the whole reference grid, blank edges included |
| `bounds` | the bounding box of the covered pixels |
| `inscribed` | the largest rectangle every channel covers in full |

Dithered or drifting channels overlap in a rectangle, and `bounds` is enough
for them. Rotated overlaps — a meridian flip, a rotator, two optical trains —
leave uncovered corners inside that box, and only `inscribed` removes them.
Use `inscribed` when a later step cannot handle `NaN`. Both modes read edges
and interior gaps alike, so an uncovered pixel in the middle of the frame
bounds `inscribed` just as an edge does.

The crop is chosen before normalization, and percentile levels are then
estimated from the kept region alone. Every channel is scaled from the same
patch of sky, and a bright edge that the crop discards cannot set a white
point. Composition maps output pixels back into the caller's full-size input
planes, so cropping copies no channel.

The written FITS moves `CRPIX` by the crop origin and records that origin in
`SEIZACRX`/`SEIZACRY`. `CRPIX` is the only WCS card a translation changes: the
rotation matrix, the reference world coordinate, and SIP distortion terms are
all expressed relative to it, so the cropped image keeps the reference
solution.

A crop also reports what each channel covered: its own covered bounding box
and pixel count, and how far its coverage center sits from the median center
of every channel. A channel past the larger of 32 pixels and 2% of the frame
is flagged `off_center` — the one that pulled the crop in. The consensus is a
median rather than a mean because one badly placed channel cannot move it; a
mean would follow the stray and carry the well-placed channels past the limit
with it. With fewer than three channels there is no majority to disagree with,
so nothing is flagged. The
Rust result carries this as `ColorComposition::crop`, the CLI prints a warning
per flagged channel, Python exposes the same measurements through
`seiza.crop_report`, and native front-ends read them as JSON from
`seiza_color_crop_report_json`.

`crop_report` takes borrowed `ChannelSamples` rather than owned images, so a
host asking about its own buffers copies nothing. The C ABI hands its channel
pointers straight through.

## RGB and linear LRGB

RGB directly interleaves the prepared red, green, and blue masters. LRGB uses
linear-light luminance

```text
Yrgb = 0.2126 R + 0.7152 G + 0.0722 B
Yout = (1 - weight) Yrgb + weight L
RGBout = RGB * Yout / Yrgb
```

Scaling the triplet retains its linear RGB chromaticity while replacing its
luminance. A zero-chrominance pixel becomes neutral at `Yout`. A weight of one
uses all source L; zero is an RGB no-op.

Super-luminance mode instead adds every prepared channel into the target
luminance:

```text
Ysuper = L + R + G + B
RGBout = RGB * Ysuper / Yrgb
```

The selected normalization is applied before the sum. The RGB triplet is still
scaled together, retaining its chromaticity, but the additive linear `f32`
result can exceed one. This is a quick-visualization and stack-depth option,
not a photometric calibration. Rust callers select it with
`combine_super_lrgb`; the CLI and Python APIs use luminance mode `super`.

The same target works without a luminance stack. Synthetic super-luminance
composes RGB alone and scales the triplet to `R + G + B`, as if one luminance
exposure had collected every channel's light:

```text
Ysuper = R + G + B
RGBout = RGB * Ysuper / Yrgb
```

Rust callers select it with `combine_super_rgb`; the CLI (`color rgb`) and
Python (`combine_rgb`) use luminance mode `super`, and the FITS output is
marked `SEIZACLR='SUPER-RGB'`.

This follows PixInsight's documented model rather than reproducing every
`LRGBCombination` control. PixInsight staff describe CIE XYZ separation for
linear images, on-demand luminance/chrominance rather than a stored L channel,
and a luminance source-to-target ratio. PixInsight also has a configurable RGB
working space, midtones transfer functions, saturation adjustment, and
chrominance noise reduction. Seiza's quick-look path intentionally fixes the
linear-sRGB/Rec.709 Y coefficients above and does not claim pixel identity with
PixInsight. See the [PixInsight staff explanation of linear XYZ and luminance
ratios](https://www.pixinsight.com/forum/index.php?threads/l-l-rgb-combines.1092/).

## Narrowband palettes

The direct palettes are static linear-light assignments. A three-letter name
lists the physical filter mapped to red, green, and blue:

| Palette | Red | Green | Blue |
| --- | --- | --- | --- |
| SHO | SII | H-alpha | OIII |
| SOH | SII | OIII | H-alpha |
| HSO | H-alpha | SII | OIII |
| HOS | H-alpha | OIII | SII |
| OSH | OIII | SII | H-alpha |
| OHS | OIII | H-alpha | SII |
| HOO | H-alpha | OIII | OIII |

`NarrowbandMatrix` is the library extension point for arbitrary static mixes.
Each output channel has explicit SII, H-alpha, and OIII coefficients.

## Foraxx

Foraxx is a dynamic display palette, not a linear-light matrix. Its author
defines per-pixel factors on stretched `[0, 1]` channels. Seiza therefore
normalizes each input, applies its existing PixInsight/N.I.N.A.-family
median/MAD midtones transfer, and then evaluates the published expressions.
The default puts the median at 0.2 and clips shadows 2.8 normal-equivalent MADs
below it. These dynamic-palette controls live in `ForaxxOptions`, separately
from the normalization-only `ColorOptions`, so linear compositions do not
depend on irrelevant display settings:

```text
fO = OIII ^ (1 - OIII)
fG = (OIII * Ha) ^ (1 - OIII * Ha)

Foraxx-SHO:
  R = fO * SII + (1 - fO) * Ha
  G = fG * Ha  + (1 - fG) * OIII
  B = OIII

Foraxx-HOO:
  R = Ha
  G = fG * Ha  + (1 - fG) * OIII
  B = OIII
```

The FITS writer records `SEIZATRF='DISPLAY'` for Foraxx and for any composition
whose inputs were explicitly marked display-referred. Linear inputs remain
`SEIZATRF='LINEAR'` for RGB, LRGB, super-LRGB, super-RGB, direct palettes, and
custom matrices. It records `SEIZACLR='SUPER-LRGB'` or `SEIZACLR='SUPER-RGB'`
for super-luminance output and the composition or palette name otherwise. A display-referred preview is written
directly; a linear-light preview receives the normal display-only asinh stretch.
The original equations and the stretched-input requirement come from
Ludo/ForaxX's [Dynamic narrowband combinations with
PixelMath](https://thecoldestnights.com/2020/06/pixinsight-dynamic-narrowband-combinations-with-pixelmath/).

## Real-data validation

The CLI path was exercised in release mode against Sh2-132 data from
`Askar107PHQ/_Source/2025/Sh2 132`: twelve 300-second light frames for each of
H-alpha, OIII, and SII. Each channel was calibrated with the available
per-filter flats and a 300-second dark, then stacked independently before color
composition.

| Stage | Result |
| --- | --- |
| H-alpha stack | 12 accepted, 0 rejected |
| OIII stack | 12 accepted, 0 rejected; approximately 179.6-degree meridian-flipped frames accepted |
| SII stack | 12 accepted, 0 rejected; approximately 179.6-degree meridian-flipped frames accepted |
| OIII to H-alpha registration | 0.253 px RMS, 11.8 px drift, -0.007 degrees, 150 matched stars |
| SII to H-alpha registration | 0.212 px RMS, 6.0 px drift, -0.023 degrees, 193 matched stars |

Both direct SHO and Foraxx-SHO produced full-resolution 6248 by 4176 FITS and
PNG outputs. The FITS metadata marked direct SHO as `LINEAR` and Foraxx-SHO as
`DISPLAY`. Cropped, downscaled versions of those PNG previews are shown in the
[image-processing guide](../image-processing.md#color-from-mono-stacks); the large source and
intermediate FITS files are intentionally not checked in.

## Boundaries

- Color composition does not alter or feed back into a live stack.
- It does not infer filter identity from filenames or FITS headers.
- It does not perform gradient removal, spectrophotometric calibration, star
  removal, hue curves, or chrominance denoising.
- FITS output preserves WCS cards from the selected reference input, with
  `CRPIX` moved to a crop's own grid. CLI inputs are resampled to the reference
  grid by default; library callers must align arrays.
- Pixels outside any required registered channel are `NaN` in all three FITS
  planes and render black in the quick-look PNG, avoiding colored crop borders.
  `--crop` removes those edges instead of masking them.
- Cropping trims to a rectangle. It does not reproject, rescale, or pad a
  channel to make one fit.
- PNG output is for fast visual assessment; floating-point FITS remains the
  editable result for linear RGB/LRGB and direct palettes.

## Narrowband balance and continuum subtraction

Status: implemented core and bindings, 2026-10-24, in the unreleased
`codex/narrowband-normalization` branch. The paired PSF Guard branch's
`docs/design/color-channel-processing.md` specifies processing order, default
setups and UI. Its combined LRGB+SHO integration remains planned. Keep these
as separate operations: aesthetic channel balance must not alter the linear
data used for continuum fitting. The current callable contract is documented
in [Continuum fitting and signed residuals](../../seiza-stacking/README.md#continuum-fitting-and-signed-residuals).
The details below retain the design rationale and future host requirements.

### Reference workflows

[Cosmic Photons](https://cosmicphotons.com/scripts/) describes
NarrowbandNormalization as channel balancing. The author's
[bundled reference documentation](https://www.cosmicphotons.com/pi-modules/narrowbandnormalization/nbnDocumentation20240816v010001.tar.gz)
specifies H-alpha-referenced, background-protected midtone curves. It accepts
linear and nonlinear data, but forbids Lightness adjustment on linear data.
The documentation is version 1.0, generated 2024-08-16. The
[update manifest](https://www.cosmicphotons.com/pi-modules/narrowbandnormalization/updates.xri)
also advertises module 1.1; its documentation archive has SHA-1
`61c134af5bf9f6bb22d4fe93dc0195b18d8f0ee5`.

[NightPhotons PCS](https://www.nightphotons.com/software/photometric-continuum-subtraction/)
provides the other reference: infer continuum weights from stellar photometry
on linear narrowband/broadband images. It supports multiple broadband inputs
and a separate starless output. The
[source](https://github.com/charleshagen/pixinsight/blob/main/src/scripts/NightPhotons/PhotometricContinuumSubtraction.js)
inspected identifies itself as 1.4.2 (Git blob
`e60befae3d9c9dd6ed7d0152460626592e76ee63`). Its defaults include 400 stars
and a 0.8 peak cutoff on its expected input scale.

These sources establish intent and useful controls. The following contracts
are proposed Seiza behavior, not a pixel-parity claim or a source-code port.
Implement the mathematics independently under this repository's licence.

### Resolve a balance plan, then apply it

Use physical H-alpha/OIII/SII inputs, not a composed RGB image whose mapping
has to be guessed. Keep the new operation independent of `NarrowbandPalette`:
the same balanced planes can feed SHO, SOH or HOO without fitting again.

Proposed surface, with names subject to implementation review:

```text
resolve_narrowband_balance(channels, options, statistics_region, mask)
    -> NarrowbandBalancePlan
apply_narrowband_balance(channels, plan)
    -> balanced channels + report
```

Inputs share dimensions, grid, transfer and a finite display range `[0, 1]`.
Require explicit `ColorTransfer`; this API never stretches sensor ADU silently.
The first host setup uses display-referred inputs. Preserve the input transfer
in the result: operating on linear input makes it display-referred as soon as
a non-identity nonlinear curve is applied. An identity plan preserves transfer.
Do not call the result calibrated linear line flux.

Resolve statistics over the same valid sky area for all participating channels.
Use deterministic, spatially distributed bounded sampling and f64 reductions.
The resolved plan carries the sampling version, region and mask identity.
Applying a plan to a downsampled display preview does not resolve new curves.
An uncovered pixel stays uncovered; HOO's duplicated OIII is counted once.

For an independently specified midtone model, let `s` be shadow protection,
`q` the median, `l` the minimum, and `d` the mean of strictly positive
deviations from `q` among the sampled valid pixels. The conditional definition
of `d` is an explicit proposed Seiza choice: verify it against a licensed
reference implementation or PI fixtures before claiming matching statistics.
With H-alpha as reference `r`, use:

```text
b       = (1 - s) l + s q
u       = (q + d - b) / (1 - b)
u_r     = (q_r + d_r - b_r) / (1 - b_r)
m       = u (1 - u_r) / (u - 2 u u_r + u_r)
m_boost = m / boost
MTF(m,x)= (m - 1) x / ((2m - 1) x - m)

v <= b: v_out = v
v >  b: v_out = b + (1 - b) MTF(m_boost, (v - b)/(1 - b))
```

This maps the positive-deviation landmark relative to its own background;
it does not equate every channel's median or line flux. H-alpha is unchanged.
Start with `s=1`, OIII/SII boosts `1`, and no lightness replacement. Do not add
highlight/brightness controls to this core operation: existing output stretch
stages already serve them. HOO synthetic green and optional green reduction
belong to display composition after physical-channel balance.

Validate all options and every denominator before mutating caller buffers.
Require `0 <= s <= 1`, finite positive boosts, and usable `0 < u,u_r < 1` and
`0 < m_boost < 1`. Expose rejected options as errors, not silent clamps.
An absent positive tail, flat channel, insufficient samples or a background
at one produces a typed skipped-channel report and leaves that channel alone.
If H-alpha is unusable, skip the whole balance. Noise-only data can still have
a positive tail: add a tested signal-reliability check before enabling an
automatic default. Its statistic and threshold remain an open validation item.

Report the per-channel median, background point, positive deviation, sample
count, curve parameter, boost, transfer and outcome. Distinguish disabled,
identity, applied and skipped. A skipped channel is not proof that emission
is absent. Green reduction must not alter these physical-channel reports.

### Continuum fit is a different linear operation

Implemented Rust surface:

```text
fit_continuum(stars, sources, options) -> Result<ContinuumFitOutcome>
measure_and_fit_continuum(narrowband, references, sources, positions,
                         fit_options, photometry_options)
    -> Result<ContinuumFitOutcome>
subtract_continuum(narrowband, references, sources, plan) -> Result<LinearImage>
```

A successful outcome contains the exact-source plan and diagnostics; a skipped
outcome contains a typed reason and no plan. Empty positions request detection.
The CLI requires already aligned inputs and an explicit linear declaration;
it does not reuse the color-composition CLI's registration or normalization.

The fit takes local-background-subtracted aperture fluxes from common,
isolated, unsaturated stars. Begin with R as the candidate continuum for
H-alpha/SII and G/B as candidates for OIII. Hosts supply physical roles; the
core never infers them from a filename. L is not a default reference. No
external star catalogue is required for a relative stellar-flux fit.

For star `i` and candidate broadband `j`, fit a robust through-origin model:

```text
F_N(i) = sum_j a_j F_Bj(i) + residual_i,   a_j >= 0
E(x)   = N(x) - sum_j a_j B_j(x)
```

Use aperture background measurements rather than fitting the nebula itself.
Gradient correction and pixel background offsets are separate recorded
operations; do not hide an unconstrained image pedestal in the stellar fit.
The coefficients include the relative scaling of the actual input images.
Total integration time and nominal filter width do not determine them.

An explicit manual model supplies coefficients on the same scale and records
its origin as manual. Validate its shape, finite nonnegative coefficients and
reference identities. It still requires linear inputs and a shared grid.
If a host rescales any input by `B'_j = g_j B_j` and `N' = g_N N`, use
`a'_j = g_N a_j / g_j`; propagate offsets separately. Reject application to
different prepared revisions unless the host supplies that exact conversion.

Apply the fitted model to the same background-corrected pixel revisions used
for photometry. Star removal, denoising and deconvolution can change aperture
flux; their names alone are not evidence that a previous fit still applies.
Future starless application must explicitly associate the pre-removal and
starless branches and validate equivalent scaling.

Require and report registration and PSF checks. Aperture totals can fit even
when subtraction leaves stellar rings. Start by refusing automatic subtraction
when stellar residuals show an unacceptable PSF mismatch. The optional Gaussian
PSF preparation records its kernels and resulting target resolution;
never sharpen one image implicitly during fitting.

For multiple references, scale design-matrix columns for numerical stability,
then convert coefficients back to the caller's units. Check rank/conditioning,
not only star count. G and B fluxes may be almost collinear. If a composite
cannot be identified, try a validated single candidate and record the fallback;
otherwise return a typed skip. Do not publish arbitrary large cancelling
weights or assume that an unconstrained least-squares solution is usable.

The diagnostic report includes candidate and used stars, rejections by reason,
coefficients, fit error, spatial/brightness residuals, conditioning, PSF outcome,
input scales and fit/manual/fallback origin. Select thresholds using planted
failures and real data. PCS's 400-star cap is a useful starting workload;
its 0.8 peak cutoff cannot be applied to arbitrary percentile-normalized ADU.
Require a known saturation scale or explicit saturated-pixel mask.

Subtraction keeps negative samples and `NaN` coverage. It does not normalize,
clip, stretch, add luminance, or assert statistical independence between
channels. Record noise propagation assumptions when reporting residual noise.
For an artistic host blend, express a single-reference residual in its
broadband reference's units as `E/a`. A multi-reference plan must define a
composite on a common broadband working scale and record its overall fitted
scale separately from its relative weights. Never add arbitrary raw NB ADU
to RGB and imply that an amount of 0.25 is comparable across exposure groups.

### Broadband emission contamination

A continuum fit from stars cannot identify the broadband's line throughput.
For a simple model `N = a C + E`, `B = C + beta E`, subtraction gives
`N - a B = (1 - a beta) E`. The default residual therefore attenuates line
signal when the broadband transmits that line. Stars alone do not determine
`beta`; without transmission information there is no exact recovery of `E`.
Multiple lines in overlapping filters require a coupled response matrix,
not three independent scalar corrections.

Label the default output an emission residual. Reserve pure line-flux claims
for `solve_emission_flux` with calibrated response inputs, an explicit
uncertainty report, and a well-conditioned matrix. Broadband integration for
an artistic LRGB+emission result does not need that scientific claim.

### Automatic narrowband with RGB stars: explicit core branches

PSF Guard's new automatic narrowband setup must support a mapped starless
nebula plus calibrated RGB stars in the first integration. Automatic selection
is host policy; the core receives concrete physical sources, extraction
results and plans. Keep `NarrowbandPalette` concrete. A stars-source selector
does not redefine the palette or silently introduce broadband nebular structure.

The host retains linear, star-containing input revisions for registration and
continuum measurements. Photometric color calibration runs on broadband RGB
before its stars are extracted. Extract stars from both branches as necessary:
the mapped nebula must be starless before RGB stars are recombined. A cached
starless image alone does not establish that the matching stars layer exists.
Extraction artifacts must carry input revision, tool/model, transfer, mask and
enough information to verify reconstruction. Reuse matching channel-card
results and chain prefixes instead of running the same tool twice.

Define and test a core recombination operation rather than implementing star
layer arithmetic in PSF Guard. The first display recipe can use screen
composition on bounded display samples `D` (nebula) and `S` (RGB stars):

```text
out = 1 - (1 - D) (1 - S)
```

Do not substitute an additive linear stars residual directly for `S`. For a
linear RGB original `I` and extracted starless RGB `B`, resolve one shared,
color-preserving display transfer `T` from the original, apply it consistently,
and derive a screen-compatible stars layer:

```text
S = (T(I) - T(B)) / (1 - T(B))
```

This reconstructs `T(I)` when screened onto `T(B)` on valid samples. The ratio
requires an explicit policy for near-white denominators, uncovered pixels and
negative extraction residuals. Preserve the signed linear residual separately;
any suppression of negative noise belongs to the recorded display plan.
Validate halos, star colors and weak stars on real images before choosing this
as a released default. Alternative additive recombination must be a named
operation with compatible transfer semantics, never an implicit fallback.

Apply the final RGB output stages after recombination. Keep the mapped nebula,
RGB star layer and final composition as separate artifacts with provenance.
Include round-trip reconstruction, empty-star identity, coverage, saturated
stars, negative residuals and calibration-color retention in Rust/C/Python
tests and output-version fixtures. Host tests cover Automatic choosing RGB,
explicit RGB failing rather than changing source, and reported automatic
fallback when sources, tools or separation results are unusable.

### Bindings, versions and validation before release

Put behavior in `seiza-stacking`, reusing `seiza-stretch`'s MTF machinery and
shared photometry where their contracts match. Do not add another host copy
of the algorithm. Ship each public Rust operation with C ABI JSON and Python
twins, Python stubs/docs/tests, generated C headers and binding-parity entries.
Expose balance and continuum operations separately in the CLI; machine-readable
reports must distinguish skipped work from a successful application.

Before touching numerical code, run stacking and stretch `output_version`
tests as required by `AGENTS.md`. Add cases for each new enabled mode, verify
that a planted change fails, and preserve existing disabled-path summaries.
Record output versions according to the repository rules. The upcoming
stacking 0.50.0 output version covers these new operations; existing stacking
and stretch summaries remain unchanged. Cover:

- Known midtone landmarks, identity, protected shadows, H-alpha preservation,
  weak/flat channels, NaNs and invalid/degenerate options.
- Balance before all static palette permutations, HOO duplication, preview
  plan reuse, common statistics regions, rotations and cropped overlaps.
- Known continuum coefficients with stellar colors, outliers, negative noise,
  gradients and planted nebular structure excluded from the fit.
- Input rescaling invariance, manual overrides, changed revisions, saturated
  stars, rank-deficient G/B, mismatched PSFs, failed fits and no stars.
- The attenuated residual predicted by the contamination equation above.
- Rust/C/Python agreement, CLI metadata, cancellation and unchanged old output.

Compare actual SHO/HOO masters before selecting automatic balance defaults,
and LRGB+SHO masters before selecting automatic continuum thresholds. Include
emission-dominated nebulae and a continuum-dominated galaxy/reflection field.
Use held-out stars and residual structure to assess a continuum fit; a prettier
JPEG is not evidence that subtraction preserved the line signal.


### Real star-layer integration check

PSF Guard's paired implementation was exercised with RC-Astro 2.6.6 and SXT
ML11 on Supergateway's CPU backend, using 1024×1024 cutouts of real Bubble
Nebula RGB and SHO channel masters. Explicit RGB stars, automatic RGB stars,
and a starless narrowband output completed. The automatic build reused both
linear separation caches. The final display FITS retained common RGB/nebula
coverage, and OIII/SII balance reported applied curves. The isolated fixture
had no fresh plate solve, so this validates star separation and display
recombination without making a Gaia-calibration or full-frame speed claim.

### Current combined-image contract

The paired draft uses Seiza 0.43.0 and stacking/output 0.50.0, unreleased.
See [continuum, matching and calibrated response APIs](../../seiza-stacking/README.md#validate-stellar-profiles-and-match-psfs-explicitly).
PSF Guard retains aligned, background-prepared linear samples, validates and
subtracts before SXT, and keeps signed residuals separate from artistic amounts.
No plan is transferred to changed starless pixels. Optional Gaussian matching
records exact kernels and propagates clipping/coverage support; held-out pixel
profiles still gate acceptance. Balance masks affect statistics only. Green
reduction runs before independently prepared RGB stars are restored.

Calibrated line-flux export is a separate response-matrix operation. It requires
producer-calibrated responses on exact source units and records full provenance
and uncertainty assumptions. Unknown covariance remains unknown. Display
previews continue to use empirical reference-unit emission residuals.

Synthetic regressions cover equal-aperture-flux core/ring failures, matching
replay and mask/NaN support, extended wing mismatch, malformed saved kernels,
unknown original saturation, independent input gains, coupled line leakage,
rank/conditioning, covariance and extreme finite numeric scales. The output
version changes because the new default pixel gate refuses a profile mismatch
that the prior aperture-only fixture accepted. Existing stacking/stretch
summaries remain unchanged.

Real validation uses isolated copies of completed masters and fresh pixel
solves; original files/catalogs stay unchanged. Reports and images distinguish
accepted emission residuals, omitted lines and failed mapped builds. No physical
line-flux calibration is inferred from those examples' filter names. Keep both
PRs draft and defer publication until the complete release is authorized.

Calibrated export also requires explicit `residualZeroLevels` (Python alias
`residual_zero_levels`), one additive zero level per residual in its exact
sample units. Zero is an explicit calibration declaration. The solver uses
`M F = E − residualZeroLevels`; it never guesses these values from a sky median
or the display baseline. Background preparation may retain an additive reference
level, so response calibration alone cannot establish a physical zero point.
Zero-level uncertainty is outside the propagated residual covariance unless
the producer includes it there.

Image-based validation also audits isolated candidates excluded for clipping or
saturation. They never train the fit. Their observed residual profiles must all
be measurable and below the same absolute limit; `clippedStars`,
`clippedProfiles` and `clippedMaxAbsoluteFraction` record this separate check.
It prevents a clean unsaturated sample from hiding clipped-core artifacts and
does not infer lost flux. The held-out report keeps both its 90th-percentile
and maximum profile error.

### Display RGB star reduction

`screen_rgb_stars_reduced` adds an explicit [0,1] reduction amount to screen
recombination. It blends the extracted star intensity toward one-pixel cross
erosion, applies one linked RGB gain and preserves missing coverage. Edge/
invalid-neighbour pixels retain their unreduced stars. Zero keeps the previous
screen path bitwise. C/Python twins and CLI `--reduction` share this operation;
PSF Guard applies it only to restored RGB stars after other nebula display work.
The analytic output-version fixture was recorded through the existing screen
operation before adding reduction. Existing stacking/stretch summaries remain
unchanged; stacking/output stays at the unreleased 0.50.0.

### Real-frame outcomes of the final gates

Isolated paired-host checks on real Gecko/Cocoon LRGB+SHO, Triangulum LRGB+Ha,
NGC7023 reflection LRGB and SNR G132 RGB/SHO used fresh pixel solves and Gaia.
Explicit 8-pixel broadening reduced unsaturated profile p90 errors to about
1–6%, but clipped-core errors remained roughly 30–84% or were unmeasurable.
The automatic recipes therefore omitted the unsafe natural-color contributions
and stopped required mapped residual builds. These are safety checks, not
successful physical line-flux measurements.

A clean SNR G132 star comparison keeps continuum Off and broadband structure
0. Real licensed BXT ML5, Gaia and SXT ML11 prepared the RGB stars. Display
reduction 0.4 then reused the exact BXT/SXT artifacts and completed in 8.11 s;
native-pixel crops show reduced star prominence with the same starless nebula.
Original masters/catalogs remained unchanged. The paired PSF Guard guide and
offline gallery retain exact requests, reports and source provenance.

Calibrated export requires `continuumCoefficients` (Python alias
`continuum_coefficients`): exact per-plan coefficient snapshots in reference
order. Residual zero-level calibration depends on these weights. A changed
fit/manual weight is rejected even when the raw source revisions are unchanged;
update the explicit zero levels and snapshots after recalibration. Reports and
cache identities retain the snapshots.

### Stretch linear RGB stars independently

Run star separation on linear RGB after any linear sharpening and color
calibration. Pass that original/starless pair to `stretch_rgb_stars(original,
starless, amount)` to extract the linear residual and apply the SetiAstro-style
rational tone `3^amount*x / (1+(3^amount-1)*x)`. Amount is 0–8; 5 is the script's
starting tone, lower values make stars fainter, and 0 leaves the normalized
linear residual without nonlinear lift. One shared source range normalizes
physical units; do not normalize each channel or the stars-only layer to its
own peak. The background offset cancels in the residual.

Use `screen_rgb_star_layer(&mut nebula, &stars, reduction)` on the resulting
display stars and display nebula. Reduction is 0–1 and applies the existing
linked cross erosion only to the stars. Negative extraction noise is clipped
only in the display layer; missing coverage masks whole RGB pixels. Linear
inputs stay unchanged. This independently implements the published
[SetiAstro tone](https://www.setiastro.com/pjsr-scripts), without replicating
PixInsight's optional ColorSaturation or SCNR processes.
