# seiza-compose

Color composition for astrophotography: registered mono stacks combined into
a color picture, Gaia color calibration, narrowband balance, and star layers.
The channels are `seiza_stacking::LinearImage`s, as stacking makes them.

These routines were in `seiza-stacking` up to its 0.65 release; their behavior did not
change when they moved. `OUTPUT_VERSION` names the release of this crate
whose output a build reproduces, so a host that caches compositions keys
them on it.

## The compositor

`picture::Picture` describes a color picture as layers, bottom first: an
input through its `Transfer` (Seiza stretch curves, a sample domain,
normalization) in a color, solid colors, groups, adjustments, and LRGB's
luminance layers, limited by painted and computed masks, with heals and
finishing steps (Seiza stretch plans, Tone and color, curves, saturation).
`render::render` renders one in tiles from a `Source` into a `Sink`, in
32-bit floats, in the caller's Rayon pool; `render::render_image` keeps the
result in memory. Each tile reads a margin for mask smoothing and heals,
reflected at the picture's edge, so any tiling gives the same pixels. Every
`combine_*` call below builds a picture and renders it here. See the
[compositor design](../docs/design/compositor.md).

## Composition

After integration, `combine_rgb`, `combine_lrgb`, `combine_super_lrgb`,
`combine_super_rgb`, and `combine_narrowband` compose aligned mono stacks
without coupling color into the live accumulator. The direct SHO permutations
and HOO remain linear-light; LRGB replaces or blends CIE luminance while
preserving RGB chromaticity, super-LRGB uses the additive `L + R + G + B`
target, and super-RGB synthesizes the same target as `R + G + B` with no
luminance stack. `NarrowbandMatrix`
supports arbitrary static SII/H-alpha/OIII mixes. Foraxx-SHO and Foraxx-HOO
are intentionally marked display-referred because their published dynamic
factors operate on stretched channels. That preparation resolves the shared
parameterized Auto-MTF model from `seiza-stretch`. `write_color_fits_f32`
records the palette and transfer semantics alongside preserved reference WCS
cards. See the [color-composition design](../docs/design/color-composition.md).

`calibrate_color` sets the gains of a linear RGB image from Gaia DR3 star
colours. `calibrate_color_image` places the Gaia stars itself, from the
offline photometry catalog or sources the caller has, and applies the gains;
`calibrate_and_stretch` then display-stretches the result. Neither fails when
calibration cannot run: the outcome gives a typed reason and the image stays
uncalibrated. `transform_wcs` carries a reference frame's solution through
registration, a turn and a crop to a composed image. See the [colour
calibration design](../docs/design/color-calibration.md).
The composition functions themselves require aligned inputs. File-oriented
callers can use `seiza-stacking`'s `Registrar` and `resample_to_reference`; the CLI does this
automatically for every non-reference filter stack. `ColorOptions` controls
normalization and declares whether inputs are linear or already display-
referred. This lets an embedding application independently stretch each mono
input before composition; Foraxx then skips its shared preparation pass rather
than stretching those inputs twice. The separate `ForaxxOptions` controls the
default display preparation used only when dynamic Foraxx palettes receive
linear inputs. Its `mask_smoothing` field applies to linear and
display-referred inputs alike: it is the standard deviation, in pixels, of a
Gaussian blur applied to the Foraxx blend factors before they mix the
channels, which removes the speckle that OIII noise gives factors computed
from single pixels. The default, `0.0`, applies no blur.

`ColorOptions::crop` trims a composition to the area every channel covers,
which registration otherwise leaves ringed with `NaN`. `ColorCrop::Bounds`
keeps the bounding box of the covered pixels; `ColorCrop::Inscribed` keeps the
largest rectangle every channel covers in full, removing the corners a rotated
or meridian-flipped channel leaves behind. Percentile levels are then estimated
from the kept region alone, and composition indexes back into the caller's
full-size planes rather than copying cropped channels. The result carries its
`ReferenceRegion` on the input grid, and `write_color_fits_f32` moves `CRPIX`
onto that grid. `ColorComposition::crop` reports each channel's own coverage
and flags one that sits far from where the others agree — the channel that
pulled the crop in. `crop_report` and `covered_region` expose the same search
for callers cropping something other than a composition; `crop_report` takes
borrowed `ChannelSamples`, so a host measuring its own buffers copies nothing.

## Display narrowband balance and RGB stars

`balance_narrowband(&mut image, &options)` takes three aligned interleaved
physical planes in H-alpha/OIII/SII order, already stretched into [0, 1]. For
HOO, duplicate OIII into plane three and set `has_sii=false`. The returned
report lists each measured channel's background, positive deviation, midpoint,
and application or skip reason. H-alpha and protected shadows keep their
samples; flat or noise-dominated channels remain unchanged.

`screen_rgb_stars(&mut nebula, &original_rgb, &starless_rgb)` derives a
screen-compatible stars layer and screens it onto a starless display nebula.
All images must share one RGB grid and [0, 1] display scaling. Apply the same
linked display transfer to original and starless RGB before calling it. Missing
coverage masks complete pixels. Neither function aligns images or runs external
tools. Both produce display-referred data, unsuitable for photometry or linear
continuum subtraction. C ABI, Python and CLI counterparts provide the same
operations; see the [color composition design](../docs/design/color-composition.md).

### Reduce restored RGB stars

`screen_rgb_stars_reduced(nebula, original, starless, reduction)` performs the
same display screen recombination with optional star minimization. An amount
in [0,1] blends the isolated star layer's maximum-channel intensity toward a
one-pixel cross erosion. A shared per-pixel RGB gain preserves layer color
ratios. The starless nebula is untouched. Invalid neighbours/outer edges retain
the unreduced layer; central coverage is unchanged. High values also remove
faint stars. Zero uses the original screen arithmetic bit for bit.

The operation is display-only; it does not sharpen a PSF or alter scientific
linear artifacts. C exports `seiza_color_screen_rgb_stars_reduced`; Python
returns a copy with the same name; CLI `seiza color screen-stars --reduction
0.4` applies it after a consistently stretched original/starless RGB pair.

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
