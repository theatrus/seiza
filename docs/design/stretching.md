# Parameterized display stretching

Seiza keeps calibration, registration, integration, and composed linear FITS
data unstretched. Display stretching is an explicit pipeline stage represented
by `seiza-stretch`; it is no longer hidden separately inside FITS loading,
stack previews, color previews, and Foraxx preparation.

## Model

A `StretchConfig` contains a requested `StretchModel`, an RGB
`ColorStrategy`, and a bounded analysis sample limit. Execution has three
separate phases:

```rust
let analysis = config.analyze(samples, channels)?;
let plan = config.resolve(&analysis)?;
let display_f32 = plan.apply_f32(samples, channels)?;
```

Explicit identity, linear, asinh, MTF, and GHS requests can skip the first
phase with `config.resolve_explicit(channels)`. Percentile-asinh and Auto-MTF
require analysis. This keeps later informed auto modes separate from their
deterministic application curves.

One-shot consumers can use `config.resolve_for(samples, channels)`, which
performs analysis only for models that require it. Interactive consumers keep
the explicit phases so they can reuse one analysis across parameter changes.

`StretchAnalysis` retains bounded, sorted samples and robust statistics so an
interactive caller can resolve several parameter choices without rescanning
the source image. `StretchPlan` contains only explicit curves. It is
serializable and can be applied to a downsampled preview, the full-resolution
image, or another image that must share exactly the same display transform.

The crate has no FITS, stacking, image-codec, or Python dependency. Inputs are
mono or interleaved RGB `f32` slices. Callers own decoding and output encoding.

## Included models

- `Identity` clamps already display-referred data only when an output encoder
  requests `[0, 1]` samples.
- `Linear` maps explicit black and white points.
- `Asinh` applies a parameterized asinh curve after explicit range mapping.
- `PercentileAsinh` resolves the black and white points from bounded robust
  percentiles before applying asinh. This is the former stack/color preview.
- `Mtf` applies explicit shadows, midtone, and highlights parameters.
- `Ghs` applies the documented Generalized Hyperbolic Stretch with explicit
  stretch factor, local intensity, symmetry point, shadow/highlight protection,
  and input black/white points. The equations and parameter ranges follow the
  [GHS process documentation](https://www.ghsastro.co.uk/doc/tools/GeneralizedHyperbolicStretch/GeneralizedHyperbolicStretch.html).
- `AutoMtf` is the existing N.I.N.A./PixInsight-family median/MAD autostretch:
  a shadow clipping distance and target median resolve an explicit MTF curve.

The exact-histogram `statistics_u16` and 65,536-entry `stretch_u16_to_u8` LUT
also live in this crate. `seiza-fits` re-exports them and keeps
`FitsImage::stretch_to_u8`, so existing Rust callers retain their API and
bit-depth-specific fast path.

## Compatibility precision

The shared implementation preserves the arithmetic of the paths it replaces:
stack/color preview asinh and Foraxx statistics/application remain `f32`, while
the established `u16` FITS LUT retains its `f64` transfer and quantization.
Regression tests compare preview and FITS bytes and Foraxx `f32` bit patterns
against the previous implementations. Changing those precisions is therefore
an explicit output-format decision, not an incidental refactor.

## Color strategies

- `Linked` analyzes pooled channel samples and applies one curve to all
  channels.
- `Unlinked` resolves one distribution and curve per channel. It can alter
  color balance and is therefore explicit.
- `LuminancePreserving` resolves a curve from Rec.709 luminance, maps `Y`, and
  scales non-negative RGB triplets by one common factor. If the requested
  result is out of gamut, the common factor is reduced rather than clipping
  individual channels, retaining chromaticity.

## Pipeline consumers

- `seiza-fits` delegates its public MTF/statistics compatibility APIs.
- The solver's compact FITS path continues to use the existing default
  Auto-MTF parameters.
- Stack and color PNG previews resolve the former 1%/99.5%, strength-10 asinh
  request through the shared model.
- Foraxx uses the shared Auto-MTF resolver for its required stretched working
  channels; the resulting composition remains explicitly display-referred.
- `seiza stretch` exposes every current model as an explicit subcommand.
- Python `seiza.stretch` accepts the same models and returns display-referred
  `float32` arrays without quantizing them to eight bits.

## Future automatic modes

Automatic selection is policy layered above deterministic models. A future
statistical or Auto-GHS selector may inspect the reusable analysis and produce
an ordinary, inspectable model or resolved plan. It must report its selected
parameters and must not introduce an opaque transfer variant. The manual GHS
model is already available; an informed selector can resolve its parameters
later without changing the application pipeline.
