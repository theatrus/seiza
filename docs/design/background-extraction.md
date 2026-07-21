# Automatic background and gradient extraction

Seiza treats background extraction as a linear-image operation, separate from
display stretching. The estimator fits a smooth surface to pixels likely to be
sky background; the correction removes that surface while retaining a robust
background level. It does not clip to `[0, 1]`, neutralize color, or make an
image display-referred.

## Reference and independent baseline

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

ADBE's source is CC BY-NC. `seiza-background` is an independent Apache-2.0
implementation of the general technique, not a translation or incorporation
of that source. In particular, Seiza uses a deterministic grid and its own
bounded search, weighting, rejection, fitting, and correction math.

## Current model

The first `ModelConfig` variant is a total-degree polynomial of degree zero
through four. Image coordinates are normalized to `[-1, 1]`, and each channel
is fitted independently with weighted least squares and optional Tikhonov
regularization. A quadratic is the default; a plane is safer when only a broad
linear gradient should be removed.

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
4. Iteratively reject positive or negative sample residuals using a robust
   median/MAD scale and refit. Two-sided rejection protects both bright emission
   and dark nebulosity from being mistaken for the background.
5. Retain the median accepted sample value per channel as the correction's
   reference background.

Mono and RGB images share sample coordinates while retaining independent
channel medians, coefficients, and reference levels. Non-finite pixels and an
optional one-value-per-pixel exclusion mask are omitted from sample windows.
The mask is the future integration seam for source segmentation, hand-drawn
regions, catalog-informed protection, or learned structure models.

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

The CLI records `SEIZABG`, `BGMODEL`, `BGDEG`, and `BGSAMP` in the corrected
FITS and preserves a valid input WCS. The optional model FITS is linear and
shares the same pixel grid and WCS. The optional JSON file contains the compact
fitted coefficients, reference levels, resolved sample radius, accepted and
rejected counts, and every sample's status for inspection or overlays.

## Memory and API shape

`fit_background` returns a compact `BackgroundFit`; it does not allocate a
full-resolution model or corrected copy. `correct_in_place` evaluates the
polynomial directly into the caller's image buffer. `correct` allocates only a
corrected image. `render_model` is explicit and is the only operation that
allocates a full image-sized background map.

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
  small local gradients, reflections, or complicated mosaics; raising the
  degree can also absorb real extended signal.
- Large nebulae or dark-cloud fields need an exclusion mask or a stiff model.
  The automatic sampler is not evidence that every accepted window contains
  pure sky.
- Pre-stack correction should normally use a plane and identical conservative
  settings across frames. The primary initial use is a registered stack.

Future `ModelConfig` variants can add thin-plate/RBF surfaces, multiscale
models, and learned masks or estimators without changing FITS loading,
correction semantics, or the fit/apply API.

## Initial real-image validation

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
