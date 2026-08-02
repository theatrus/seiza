# Automatic background and gradient extraction

Seiza treats background extraction as a linear-image operation, separate from
display stretching. The estimator fits a smooth surface to pixels likely to be
sky background; the correction removes that surface while retaining a robust
background level. It does not clip to `[0, 1]`, neutralize color, or make an
image display-referred.

## Reference methods and design choice

The initial sampling strategy is informed by SetiAstro's
[Automatic DBE (ADBE)](https://www.setiastro.com/pi-scripts) workflow. The
public ADBE script spreads candidate windows across the frame, moves them
toward nearby low-background locations, measures robust window statistics,
rejects unsuitable points, and hands weighted samples to PixInsight's Dynamic
Background Extraction process. Siril independently documents the same useful
family of techniques: random or gridded samples, local-minimum optimization,
robust bright-structure rejection, polynomial or thin-plate RBF surfaces, and
additive or divisive correction in its
[background-extraction guide](https://siril.readthedocs.io/en/latest/processing/background.html).

[Siril](https://siril.readthedocs.io/en/latest/processing/background.html)
supplies a useful direct comparison: its background tool offers global
polynomials and smoothed thin-plate radial-basis functions, rejects bright
sample regions, and warns that flexible models can remove nebulae.
[SExtractor](https://sextractor.readthedocs.io/en/stable/Background.html) and
[Photutils Background2D](https://photutils.readthedocs.io/en/stable/api/photutils.background.Background2D.html)
use another sound family: robust local estimates on a mesh, followed by
filtering and interpolation. Their mesh-size controls expose the same tradeoff:
a small scale can absorb extended objects, while a large scale misses local
gradients.

Seiza uses polynomial and thin-plate radial-basis surfaces because both stay
compact, work with irregular masked samples, and fit the existing model/apply
API. It does not add a second full-frame mesh path. Held-out sample scoring
chooses among the candidates enabled in automatic mode.

ADBE's source is CC BY-NC. `seiza-background` is an independent Apache-2.0
implementation of the general technique, not a translation or incorporation
of that source. Seiza uses a deterministic grid and its own bounded search,
weighting, rejection, fitting, validation, and correction math.

## Surface models

`ModelConfig::Polynomial` is a total-degree polynomial of degree zero through
four. Image coordinates are normalized to `[-1, 1]`, and each channel is fit
with weighted least squares and optional Tikhonov regularization. A plane is
the safest manual choice for a broad linear gradient.

`ModelConfig::RadialBasis` is a thin-plate spline with an affine tail. Its
smoothing term is divided by each control point's fit weight, so noisy windows
pull the surface less. The control-point cap bounds dense solve cost. Larger
smoothing values make the surface stiffer; zero asks it to interpolate the
control points and may produce a singular or overfit model.

`ModelConfig::Automatic` evaluates constant through `max_degree` polynomial
candidates with deterministic four-way held-out sampling. It can also evaluate
one radial-basis candidate when `allow_radial_basis` is true. That opt-in is
deliberate: training and held-out samples can share real extended emission, so
lower validation error alone cannot prove that a flexible surface models sky.
Automatic mode scores the median absolute residual after normalizing each
channel by its robust sample scale. Starting with the constant, it selects a
more flexible candidate only when the error improves by both a small absolute
floor and `minimum_improvement`. Diagnostics retain every successful score and
the selected model. The library default remains the prior quadratic model for
API compatibility; the CLI opts into polynomial-only automatic mode by
default.

The fitting pipeline is:

1. Distribute deterministic seed points across the usable image area. The
   longest axis receives `samples_per_axis` seeds and the other axis is scaled
   by the aspect ratio.
2. Resolve an image-sized sample radius unless the caller supplies one. Move
   each seed through a bounded 3-by-3 neighborhood toward a window with lower
   robust channel median and dispersion. The bounded displacement keeps the
   samples spatially representative instead of letting them collapse into one
   dark corner.
3. Reject unusually noisy windows from their normal-equivalent MAD. Fit all
   remaining channel surfaces using inverse-dispersion weights.
4. In automatic mode, score candidate surfaces on held-out accepted samples and
   choose the least flexible model that clears the configured improvement.
5. Iteratively reject positive or negative sample residuals using a robust
   median/MAD scale and refit. Two-sided rejection protects both bright emission
   and dark nebulosity from being mistaken for the background.
6. Retain the median accepted sample value per channel as the correction's
   reference background.

Mono and RGB images share sample coordinates while retaining independent
channel medians, coefficients, and reference levels. Non-finite pixels and an
optional one-value-per-pixel exclusion mask are omitted from sample windows.
Normalized protected ellipses and polygons provide a cheaper second seam for
solver/catalog context. A caller can project a stored OpenNGC outline through
the solved WCS, call `ProtectedRegion::polygon_from_pixels`, and add one polygon
per connected contour. Catalogs with only center and extent data can use a
rotated ellipse. The pixel mask remains the finer path for source
segmentation, hand-drawn regions, or learned structure models.

The same regions serialize inside `BackgroundConfig`, so native render callers
can pass solved outlines through the existing background JSON:

```json
{
  "model": {
    "kind": "automatic",
    "max_degree": 2,
    "allow_radial_basis": true
  },
  "protected_regions": [
    {
      "kind": "polygon",
      "points": [[0.31, 0.24], [0.58, 0.20], [0.63, 0.52], [0.35, 0.57]]
    }
  ]
}
```

The sampler pads each protected region by the resolved sample-window radius,
so a window centered just outside a catalog contour does not pull target pixels
into its median.

## Correction semantics

`CorrectionMode::Subtract` applies

```
corrected = input - fitted_background + reference_background
```

for additive sky glow and light-pollution gradients. `CorrectionMode::Divide`
applies

```
corrected = input / fitted_background * reference_background
```

for multiplicative illumination or vignetting-like fields. Division rejects a
zero or non-finite reference and a model that reaches or crosses zero instead
of silently producing infinities or reversing the signal. Input NaNs remain
NaN in both modes.

Both correction modes accept a strength in `[0, 1]`. Subtraction scales the
fitted gradient directly. Division blends between an identity factor and the
full reference-to-model ratio. Zero leaves the input unchanged and one keeps
the prior correction math.

The CLI records `SEIZABG`, `BGMODEL`, `BGSTR`, optional `BGDEG`, and `BGSAMP` in
the corrected FITS and preserves a valid input WCS. The optional model FITS is
linear and shares the same pixel grid and WCS. The optional JSON file contains
the compact fitted coefficients, reference levels, automatic candidate scores,
resolved sample radius, accepted and rejected counts, and every sample's status
for inspection or overlays.

## Memory and API shape

`fit_background` returns a compact `BackgroundFit`; it does not allocate a
full-resolution model or corrected copy. `correct_in_place` evaluates the
chosen surface directly into the caller's image buffer. `correct` allocates
only a corrected image. `render_model` is explicit and is the only operation
that allocates a full image-sized background map.

This fit/apply split is also exposed to Python as `BackgroundModel`. It makes
interactive parameter changes cheap to reason about and lets callers inspect
sample diagnostics before choosing to apply the correction.

The C ABI exposes the same lifecycle as an opaque `SeizaBackgroundModel`.
Callers provide interleaved linear floats, an optional byte mask, and optional
configuration JSON; model rendering and correction write into caller-owned
buffers, while diagnostics JSON remains borrowed from the model. This avoids
cross-allocator image ownership and keeps the ABI stable as model variants are
added.

## Intended use and limits

- Crop black registration borders or exclude them before fitting. The default
  three-percent border helps with minor edges but is not a substitute for a
  valid overlap crop.
- Run on calibrated, linear data. For a final stack, background extraction
  normally precedes color calibration and display stretching.
- A low-degree polynomial is deliberately conservative. It cannot reproduce
  small local gradients, reflections, or complicated mosaics. A radial-basis
  surface can reproduce them, but can also absorb real extended signal.
- Large nebulae or dark-cloud fields need an exclusion mask or a stiff model.
  The automatic sampler is not evidence that every accepted window contains
  pure sky.
- Solver-projected catalog bounds help protect known structure, but incomplete
  catalog geometry does not prove that every unprotected sample is sky.
- Pre-stack correction should normally use a plane and identical conservative
  settings across frames. The primary initial use is a registered stack.

Future `ModelConfig` variants can add multiscale models and learned masks or
estimators without changing FITS loading, correction semantics, or the
fit/apply API.

## Initial real-image validation

The automatic/RBF extension was also checked on an existing 6248-by-4176 mono
stack dominated by extended emission. When RBF was allowed, held-out scoring
preferred it (normalized error 0.328 versus 0.494 for the quadratic), but the
rendered model visibly traced the nebula. That test is why automatic mode does
not consider RBF unless the caller opts in. Polynomial-only automatic mode
selected the quadratic, accepted 85 of 96 windows, and rendered a broad smooth
field without target detail.

The release CLI was exercised from local disk on the existing 6248-by-4176
Askar107PHQ Sh2-132 validation products used by the stacking/color work:

- The mono H-alpha stack accepted 87 of 96 candidate windows. Its rendered
  quadratic model contained only a broad left-to-right/vertical field and no
  recognizable Crescent-region emission or dark structure.
- The three-plane SHO stack fitted each channel from the same positions and
  accepted 77 of 96 windows. The rendered RGB model showed a smooth colored
  illumination field without target morphology.
- Both runs wrote a corrected FITS, full model FITS, and JSON diagnostics in
  about 0.7 seconds in a warm local release-mode smoke test. This is a
  developer-machine observation rather than a portable benchmark.

Synthetic regression tests separately recover known mono and independent RGB
planes in the presence of bright sources, verify structure-mask exclusion,
exercise subtractive and divisive correction, preserve NaNs, and round-trip
the serialized configuration and fitted model.
