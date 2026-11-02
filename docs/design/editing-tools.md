# Image editing tools

## Goal

Seiza processes an image from calibration up to a display stretch. An editor
needs more after that. It needs adjustments made to the stretched image, star
layers, masks, a record of the steps taken, and measurements a user reads off
the stack. This note plans those tools as core-crate functions. Seiza supplies
the operations, masks and step records. A host (the Mac and Windows apps, PSF
Guard, Python) supplies the canvas, the sliders and the history view.

The learned operations (denoising, sharpening and star separation) are planned
separately in [learned restoration](ml-restoration-training.md). This note
covers the classical tools around them, including the classical fallback for
each learned one.

## Principles

1. **Every operation declares its domain.** An image is either linear or
   display-referred. Foraxx palettes already mark display-referred FITS
   output, and every image will carry that tag. An operation given the wrong
   domain refuses with an error variant that names the operation, the domain
   it expected and the domain it was given. It never guesses. Linear tools are
   background, colour calibration, deconvolution, denoising, continuum work
   and HDR composition. Display tools are curves, saturation, local contrast
   and the multiscale tools. Star separation, halo removal and SCNR accept
   either domain, and record which one they were given.
2. **The preview and the result share one plan.** Every tool follows the
   stretch crate's pattern: analyse, resolve, then apply. A host resolves a
   step once and applies the same plan to a screen-sized preview and to the
   full image. Radii and wavelet scales are given in full-resolution pixels,
   and the plan converts them to the preview's scale. A test holds the
   downsampled full-resolution result to the preview within a tolerance.
3. **A recipe records the steps.** A step holds the operation, its options,
   its resolved plan and the report of what it measured. `StretchStack`
   already keeps ordered stages, float intermediates and resolved plans. A
   recipe extends that to every tool. A recipe applied to another image
   resolves each step again from that image, unless the step is locked to its
   resolved plan. It is saved as versioned JSON. Going back to a step is the
   host re-running from the intermediate it cached. Seiza guarantees that
   running the same step again gives the same result.
4. **Every step reports what it used:** the stars it measured, how much of the
   image the mask covered, the gains it applied and the samples it clipped.
   A host shows this report rather than computing its own.
5. **Masks are values that tools accept.** A `Mask` is one feathered plane of
   values in `[0, 1]`. Masks come from a luminance range, from a star layer,
   from detected stars (dilated by the measured FWHM), from coverage, or from
   a host's drawing. Masks combine by union, intersection and inversion. Every
   display tool takes an optional mask, and the structure tools leave stars out
   of their masks by default.
6. **Thresholds use one scale.** A threshold of 0% is the sky background (the
   median of the pixels that are neither star nor object). A threshold of 100%
   is the bright end of the object: a high percentile of the pixels that are
   not stars. Every tool's threshold uses this scale, so one slider means the
   same thing in every tool.

## Where it lives

A new crate, `seiza-edit`, holds the display tools, masks and recipes. It
depends on `seiza-imgproc` for wavelets, blurs and morphology, on `seiza-stats`
and on `seiza-stretch`. It has its own `OUTPUT_VERSION` and `output_version`
test from its first PR, and follows the same rules as stacking and stretching,
because hosts will cache its previews. Keeping it out of `seiza-stretch` means
a change to a curve tool does not invalidate cached stretches.

The star-layer tools stay in `seiza-stacking`, next to `star_stretch` and
`narrowband_balance`, which already handle star layers. The measurements go
in `seiza-analysis`, which owns the stack reports. HDR and comet stacking go
in `seiza-stacking`.

Each public function gets a C export and a Python twin in the same PR, as
[AGENTS.md](../../AGENTS.md) requires.

## Tools

### What Seiza already has

| Tool | Domain | Where |
|---|---|---|
| Background gradient removal | linear | `seiza-background` |
| Photometric colour calibration from Gaia | linear | `color_calibration` |
| Damped Richardson–Lucy deconvolution | linear | `seiza-deconvolution` |
| GHS, asinh, MTF, layered stretches | linear to display | `seiza-stretch` |
| SHO/HOO, Foraxx and custom palettes | both | `color`, `narrowband_balance` |
| Continuum fitting and subtraction | linear | `continuum` |
| Independent star stretch, screen recombine, display star reduction | both | `star_stretch`, `narrowband_balance` |
| Star separation through an external tool | both | [`external`](external-tools.md) |
| Tilt and corner HFR from star cells | measure | `seiza-stars::tilt` |
| Depth curve: how noise falls with frame count | measure | `snr` |

### Display adjustments (`seiza-edit`)

All of these run on display-referred RGB or mono and accept a mask. The
multiscale tools share one à trous B3-spline decomposition per image, cached
by the plan. `StructureRemover` already implements this decomposition.

- **Curves.** Monotone cubic interpolation through control points (the
  Fritsch–Carlson method), so a curve never overshoots between points. A curve
  applies to R, G, B or all of them, to luminance alone (with RGB ratios kept),
  or to saturation as a function of luminance. It is applied through a
  4096-entry table. The plan also returns sampled curve points and the
  histogram, so a host can draw the curve over the histogram.
- **Saturation at constant hue.** Scale chroma in OKLCh, keeping lightness
  and hue. Six smoothly weighted hue bands let one hue be raised alone. A
  luminance mask holds back the background, where raising saturation mostly
  raises colour noise.
- **Temperature.** Opposite gains on red and blue, turned into per-channel
  curves fixed at the measured background level. The sky stays neutral while
  the object warms or cools.
- **SCNR.** Average neutral, `G' = min(G, (R + B) / 2)`, and maximum neutral,
  `G' = min(G, max(R, B))`, each with an amount and an optional mask. This is
  valid in either domain.
- **Local contrast per scale.** One gain each for a fine, a medium and a
  large group of wavelet layers. The groups are given in full-resolution
  pixels: about 16–32, 64–128 and 256–512 px. A gain applies only above about
  3σ of that layer's noise, where σ is the layer's MAD. Below that, it leaves
  noise unamplified. Stars are left out of the mask.
- **Large-scale compression.** Keep the wavelet layers up to a chosen scale.
  Compress the residual (large-scale) plane with a soft knee above the
  threshold, then add the layers back. A bright core keeps its structure
  while its overall level comes down. Luminance only, with RGB ratios kept.
- **Dark structures.** Deepen the negative coefficients in the middle wavelet
  layers, where a region is darker than its surroundings. The deepest layer
  sets the widest dust lane affected. Coefficients below the noise threshold
  are ignored. Stars are left out of the mask, because the dark wells around
  stars would otherwise deepen too.
- **Faint signal lift.** Build a local background map from the large-scale
  residual. Apply a lift curve to each pixel's signal above that map. Range
  sets what counts as faint and Lift sets how far it rises. A black level
  places the darkest pixel. The neutral-black option measures the lift from a
  neutral grey instead of the local colour, so a tinted dark nebula does not
  turn into a colour veil.
- **Region statistics.** A host passes a rectangle and gets back its median,
  its percentiles, its hue and its threshold on the shared scale. A host maps
  a click-and-drag on the image onto a tool's parameter with these. This is
  the core half of "drag on the region you want to change".

### Linear tools

- **Colour calibration profiles.** Save the gains fitted for one camera and
  filter set, keyed by camera, filter and gain. Apply them when a field has
  too few matched stars, or no solve. The report says the gains came from a
  profile, and which fit made it. A fresh fit that differs from the profile by
  more than a few percent is reported as a warning.
- **Hα from an RGB image.** For an OSC or RGB broadband image, estimate the
  continuum in R from G with the existing continuum fit, measured on stars.
  The difference is an Hα estimate. Add it back to R with an amount, and a
  smaller share to B for the usual pink. It runs on linear samples, and
  background neutralisation runs again afterwards.
- **Dual-band extraction.** For dual-band filters on an OSC camera, separate
  Hα (mostly R) from OIII (G and B), weighted by the Bayer response. Then
  build the palettes from those two channels, with a synthesised third
  channel. The output is linear and goes through the existing composition
  code.
- **Classical noise reduction.** Wavelet coefficient shrinkage driven by the
  stack's variance map, with the noise of each layer measured rather than
  assumed. This is the deterministic baseline for the learned noise model.
  It is also what runs when no model is installed.
- **Crop, rotate and flip** of any image, with the WCS and the reference
  pixel updated. Crops already update `CRPIX` in colour composition; this
  makes that a general operation.

### Star tools (`seiza-stacking`)

- **Linear star reduction.** The display path already erodes a screened star
  layer. In the linear domain, each detected star can instead be re-rendered
  with a narrower copy of its fitted PSF at the same flux. Its photometry is
  then unchanged, and its halo shrinks with it. This needs a star layer and
  the PSF field from the [restoration plan](ml-restoration-training.md#psf-field).
- **Halo removal.** The host passes a bright star's position. Seiza measures
  its radial profile as azimuthal medians in annuli, which ignore neighbouring
  stars. The halo ends where the profile meets the background plus a few σ.
  The star keeps its glow inside an inner radius. Between the two radii, the
  tool subtracts an amount of the profile above the glow model, with a smooth
  taper. A residue option subtracts the smooth veil left where neighbouring
  halos overlap. A starless option fills a ring, disc or hollow left by a
  removed star from the annulus outside it. Both radii are in the report, so
  a host can draw them.
- **Catalogue star colours.** Dual-band and narrowband data give stars
  unnatural colours. Replace each star's colour in the star layer with one
  derived from its Gaia BP−RP, scaled to the star's measured flux, with a
  saturation control. The image must be solved. Stars with no catalogue match
  keep their own colour.
- **Linear recombination.** `starless + k · stars` with a contribution `k`.
  Display recombination by screening already exists.

### Measurements (`seiza-analysis`)

- **Field diagnosis.** `seiza-stars::tilt` already measures FWHM and
  eccentricity in cells. Add interpolated maps that a host can contour, and a
  fitted model: `FWHM(x, y) = a + b·x + c·y + d·r²`, with orientation
  coherence for elongation. The diagnosis reads that model:
  - a planar gradient is sensor tilt;
  - a radial term is field curvature;
  - a common elongation angle everywhere is tracking or guiding;
  - elongation that points radially or tangentially is coma or astigmatism.

  A diagnosis is given only when the spread across the field exceeds 10% and
  the term is significant. Otherwise the report says the field is uniform.
- **Depth limits.** On a solved image, cross-match the detected stars with
  Gaia and fit a zero point from instrumental flux against G. From that:
  - The limiting magnitude at 5σ follows from the aperture noise.
  - The surface-brightness limit at 3σ over 10″ × 10″ boxes is
    `ZP − 2.5 log10(3 σ_box / Ω)`, where `σ_box` is the scatter of box means
    placed at random on masked sky. Measuring it that way captures the noise
    correlation left by resampling.

  These are estimates, not calibrated photometry, because the filter does not
  match G, and the report says so.
- **S/N map and remaining exposure.** Signal is the local median above the sky.
  Noise comes from the stack's per-pixel variance and coverage, averaged over
  the region, with the correlation correction measured from the noise
  autocorrelation. The extra time needed to reach a target S/N is
  `t · ((S/N_target / S/N_now)^(1/α) − 1)`, where `α` is the slope at which
  noise falls with depth. The `snr` module already measures that slope while
  stacking, so the estimate uses what this stack did rather than assuming
  `α = 0.5`.

### Stacking modes (`seiza-stacking`)

- **HDR from two exposure lengths.** Stack the long and short exposures on one
  grid. The stacker already records which samples came from clipped raw
  pixels. Fit `long = a · short + b` robustly on samples that are unsaturated
  in both stacks. Replace samples near saturation in the long stack with the
  scaled short stack, ramping the weight smoothly over 70–90% of the clip
  level. The output stays linear, extends past the long exposure's range, and
  the report gives the fitted ratio.
- **Comet stacking.** `seiza::minor_bodies` already propagates MPC orbits to
  each frame's time. Use that for the comet's motion, and refine it from the
  comet's centroid in frames with the stars subtracted. Produce three images:
  - a star-aligned stack, from which rejection drops the moving comet;
  - a comet-aligned stack, from which rejection drops the trailing stars;
  - a composite of the comet-aligned stack onto the star-aligned starless
    image.

  When no motion can be measured or predicted, return the star-aligned stack
  and say why.

## Order of work

1. **Foundations:** domain tags and the domain error, `Mask`, the shared
   threshold scale, region statistics, and the recipe format with its schema.
2. **Display adjustments:** SCNR, curves, saturation and temperature, then
   the three multiscale tools on one cached decomposition, and faint signal
   lift. These are cheap, and hosts need them first.
3. **Measurements:** the S/N map, field diagnosis and depth limits. They
   reuse data Seiza already computes.
4. **Star tools:** halo removal, catalogue star colours and linear
   recombination. Linear star reduction follows once the PSF field exists.
5. **Linear extras:** colour calibration profiles, Hα from RGB, dual-band
   extraction and classical noise reduction.
6. **Stacking modes:** HDR, then comet stacking.

Every tool gets a golden test on a fixed crop and a domain-refusal test. The
multiscale tools also get the preview-consistency test. The CPU path comes
first. The decomposition and the curve tables are good candidates for the
existing wgpu infrastructure once a host shows they are the slow part.
