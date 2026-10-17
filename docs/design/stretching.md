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
phase with `config.resolve_explicit(channels)`. Percentile-asinh, Auto-MTF,
median anchor and layered require analysis. This keeps later informed auto modes separate from their
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

## Ordered stretch stacks

`StretchStack` is the reusable ordered pipeline in the core API. It holds a
list of `StretchConfig` stages and applies them in order: each stage resolves
and applies against the previous stage's output, so a data-driven model such as
Percentile-asinh or Auto-MTF re-analyzes what the earlier stages produced
rather than the original image. Samples stay `f32` between stages; a `u8`
display result is quantized only after the last stage.

A stack always holds at least one stage. `StretchStack::new(stages)` rejects an
empty list and `StretchStack::single(config)` wraps one config. A stack
serializes as a plain array of its stages, so a saved stack and a saved single
config share one JSON shape; the C ABI's JSON array form runs this same code.

`apply_f32` and `apply_u8` return a `StretchStackOutput` holding the final
pixels and every stage's resolved `StretchPlan`, kept for provenance and reuse.
The `*_with_progress` variants report a resolving, applying, or completion
boundary for each stage, so a UI host can show progress across the whole
pipeline instead of one indeterminate spinner.

The layered stretch looks at neighbouring pixels, so it needs the image's
width and height. `apply_f32_image`, `apply_u8_image` and `apply_u16_image`
(with `_with_progress` twins on the stack) take an `ImageShape` and run every
stage; they give other models the same samples as the shape-less methods.
Resolving needs no shape, but the shape-less methods refuse to apply a
layered stage rather than guess the image's size.

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
- `MedianAnchor` places the background median exactly on a target with one
  closed-form midtones-family curve. The black point sits a number of robust
  sigmas below the median, or at the image minimum. The noise is 1.4826 times
  the median distance below the median of the samples at or below it, so stars
  cannot inflate it; for Gaussian noise that is the true sigma, the same scale
  as Auto-MTF's shadow clip. A background with no measurable noise uses the
  minimum. Samples are rescaled so the black point maps to 0 and 1 stays at
  1, and with the rescaled median `M` and target `T`,
  `S(r) = (M - 1)·T·r / (M·(T + r - 1) - T·r)`, so `S(0) = 0`, `S(1) = 1` and
  `S(M) = T` exactly. Optional steps follow in order: a curves boost that lifts
  the quarter tones above the target (pinned at the target and 1), a
  soft-knee highlight compression (a cubic Hermite from slope 1 at the knee to
  at most 2.8 at 1, which stays monotonic), and a rescale so the brightest
  analysed sample reaches 1. With `LuminancePreserving` it stretches Rec.709
  luminance and scales each pixel's RGB by one factor.
- `Layered` applies the median anchor curve to luminance and treats large and
  fine structure apart. It is described below.

## Layered

Input is unit-linear mono or interleaved RGB. Non-finite samples come out 0
and carry no weight in the statistics or the blurs.

1. Luminance `L` is Rec.709 for RGB and the channel itself for mono.
2. From the bounded analysis sample of `L`: median `m`, lower-half noise `σ`
   (as for the median anchor) and minimum.
3. The global curve `S` is the median anchor curve with target
   `target_background`, black point `max(min, m − k·σ)` with
   `k = 6 − 4.5·aggressiveness`, and no boost, compression or normalizing.
   Its slope is exact: with the rescaled median `M`, target `T`, black `b`
   and `r = (x − b)/(1 − b)` held to `[0, 1]`,
   `S'(x) = (M − 1)(T − 1)·M·T / ((M(T + r − 1) − T·r)²·(1 − b))`.
4. The large-scale layer `B` is `L` blurred with `σ = scale_separation / 4`
   pixels: three box passes along rows, then columns, each a running sum, so
   the cost does not depend on the radius. Space past the edges and
   non-finite samples carry no weight, and the result is divided by the
   weight that remains.
5. Contrast recovery `c`:
   `Y = S(B) + (S(L) − S(B))·g`, `g = min(4, (S'(m) / S'(min(L, B)))^(c/2))`.
   To first order `S(L) − S(B)` is `(L − B)·S'(B)`, so for detail brighter
   than its surroundings this gives it the slope
   `S'(B)^(1 − c/2)·S'(m)^(c/2)`: on bright structure, where the curve
   flattens, detail keeps more of the background's slope. At `c = 0` it is
   exactly `S(L)`, and the blur is skipped. Two choices keep it sound. The
   exact difference `S(L) − S(B)` rather than the linear term keeps stars
   from saturating, since the linear term grows without bound across a star.
   Detail darker than its surroundings takes the slope at its own level,
   `min(L, B)`, so the background beside bright structure, where `B` is
   raised but `L` is not, is not pushed toward black.
6. Dynamic range compression `d`: the fine layer is
   `F = L − blur(L, σ = 2 px)`, the mask
   `M = smoothstep(clamp((F − 3σ)/(12σ), 0, 1))` finds star cores, and
   `Y ← Y + d·M·(H(Y) − Y)`, where `H` is the median anchor's soft-knee
   highlight compression with knee 0.5 and end slope 2.8. Broad structure has
   no fine detail and is left alone. Skipped at `d = 0`.
7. `Y` is held to `[0, 1]`.
8. Colour, for RGB: each channel and `L` are rescaled as
   `max(0, (v − b)/(1 − b))`, the channels scaled by `Y / L'`, and the
   triplet divided by its largest channel if that passes 1. A pixel with
   `L' ≤ 10⁻⁶` becomes grey `Y`.
9. Saturation, for RGB when `saturation_amount > 0`: in HSV,
   `s' = s + A·(1 − s)·s^e` with `e = 2·(1 − saturation_boost)` and
   `A = saturation_amount`, times `V` when `lightness_mask` is on. Its slope
   is at least `1 − A`, so it stays monotonic. Each channel's distance below
   `V` scales by `s'/s`, which keeps hue and `V`.

| Control | Range | Default |
|---|---|---|
| `target_background` | `(0, 1)` | 0.15 |
| `aggressiveness` | `[0, 1]` | 0.5 |
| `dynamic_range_compression` | `[0, 1]` | 0 |
| `scale_separation` | 8 to 1024 pixels | 256 |
| `contrast_recovery` | `[0, 1]` | 0 |
| `saturation_amount` | `[0, 1]` | 0 |
| `saturation_boost` | `[0, 1]` | 0.5 |
| `lightness_mask` | on or off | on |

The layered stretch carries colour by luminance, so it takes the `Linked` or
`LuminancePreserving` strategy and refuses `Unlinked`. Like the
luminance-preserving strategy it keeps the input's colour balance, so an
uncalibrated one-shot-colour stack keeps its cast. `scale_separation` counts
the pixels of the image it is applied to, so a downsampled preview separates
larger structure than the full image does.

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
- `seiza stretch` exposes every current model as an explicit subcommand,
  through the shape-aware methods.
- The C ABI renders its stretch configuration through the shape-aware
  methods, so `{"type": "layered", …}` works there.
- Python `seiza.stretch` accepts the same models and returns display-referred
  `float32` arrays without quantizing them to eight bits.

## Future automatic modes

Automatic selection is policy layered above deterministic models. A future
Auto-GHS selector may inspect the reusable analysis and produce
an ordinary, inspectable model or resolved plan. It must report its selected
parameters and must not introduce an opaque transfer variant. The manual GHS
model is already available; an informed selector can resolve its parameters
later without changing the application pipeline.
