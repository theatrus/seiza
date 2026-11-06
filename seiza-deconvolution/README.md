# seiza-deconvolution

Experimental, conservative deconvolution for linear astrophotography images.
The current operation is damped Richardson-Lucy restoration with a circular
Gaussian PSF measured from stars in the image.

```rust
let config = seiza_deconvolution::DeconvolutionConfig::conservative(3.1);
let restored = seiza_deconvolution::deconvolve(
    &linear_pixels,
    width,
    height,
    channel_count,
    &config,
)?;
```

The FWHM is in pixels and should come from unsaturated stars. The conservative
defaults use four iterations, blend only 35% of the estimate into the original,
damp corrections near the configured channel-relative noise floor, limit each
iteration's correction, and renormalize flux per channel.

Use `deconvolve_masked` for registered images whose missing border samples are
`NaN`. It keeps the mask in the output and scales each convolution by its valid
PSF support, so missing pixels do not darken nearby data. `deconvolve` keeps its
strict finite-input check for callers that want to reject gaps.

The Python wheel exposes `seiza.deconvolve(image, psf_fwhm=3.1)` for
C-contiguous NumPy arrays; pass `masked=True` to keep `NaN` registration
borders masked. Native applications can use `seiza_deconvolve_in_place` from
the generated `seiza-cabi` header.

## Towards a target PSF across the field

`deconvolve_field` runs the same damped update with a PSF that varies across
the image, measured from its stars as a `seiza_stars::PsfField`, towards a
round Gaussian target no narrower than 1.5 px:

```rust
let field = seiza_stars::PsfField::measure(&pixels, width, height, 1, &Default::default())?;
let kernel = seiza_deconvolution::RemainingKernel::new(
    &field,
    seiza_deconvolution::TargetPsf::BestRound,
    seiza_deconvolution::DEFAULT_NODE_SPACING,
)?;
let restored = seiza_deconvolution::deconvolve_field(
    &pixels,
    &kernel,
    &seiza_deconvolution::FieldDeconvolutionConfig::conservative(),
)?;
let check = seiza_deconvolution::reblur_consistency(&restored.data, &pixels, &sigma, &kernel)?;
```

`RemainingKernel` is the blur `R` between the PSF and the target, with
`P = T ∗ R`, on a grid of nodes. `reblur_consistency` blurs an estimate by
`R` and reports the reduced χ² of what is left of the input, against its
noise, and how flat that residual's spectrum is: a check that needs no
truth. Python has `seiza.RemainingKernel`, `seiza.deconvolve_field` and
`seiza.reblur_consistency`; the C ABI has `seiza_remaining_kernel_new`,
`seiza_deconvolve_field_in_place` and `seiza_reblur_consistency_json`.

Use it after calibration, stacking, and background correction, but before any
display stretch. Raw Bayer mosaics should be debayered first. The algorithm
supports mono and interleaved RGB `f32` samples in arbitrary linear units,
including negative background-corrected samples.

This is not blind sharpening and does not infer missing detail. A symmetric,
spatially invariant Gaussian cannot model field-dependent aberrations, tracking
motion, saturated stars, or complex optical wings; the field-varying call
models the first and the last, and neither models saturated stars. See the
[design note](https://github.com/theatrus/seiza/blob/main/docs/design/deconvolution.md)
for limits and next steps.
The repository also includes
[real-corpus classical/external-reference comparisons](https://github.com/theatrus/seiza/blob/main/docs/benchmarks/2026-07-deconvolution-corpus.md)
that contrast the conservative defaults with a stronger classical sweep and
retain its ringing failure rather than presenting FWHM alone,
and a
[model-based restoration training plan](https://github.com/theatrus/seiza/blob/main/docs/design/ml-restoration-training.md).
