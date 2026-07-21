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

The Python wheel exposes `seiza.deconvolve(image, psf_fwhm=3.1)` for
C-contiguous NumPy arrays. Native applications can use
`seiza_deconvolve_in_place` from the generated `seiza-cabi` header.

Use it after calibration, stacking, and background correction, but before any
display stretch. Raw Bayer mosaics should be debayered first. The algorithm
supports finite mono and interleaved RGB `f32` samples in arbitrary linear
units, including negative background-corrected samples.

This is not blind sharpening and does not infer missing detail. A symmetric,
spatially invariant Gaussian cannot model field-dependent aberrations, tracking
motion, saturated stars, or complex optical wings. See the
[design note](https://github.com/theatrus/seiza/blob/main/docs/design/deconvolution.md)
for limits and next steps.
The repository also includes
[real-corpus classical/external-reference comparisons](https://github.com/theatrus/seiza/blob/main/docs/benchmarks/2026-07-deconvolution-corpus.md)
that contrast the conservative defaults with a stronger classical sweep and
retain its ringing failure rather than presenting FWHM alone,
and a
[model-based restoration training plan](https://github.com/theatrus/seiza/blob/main/docs/design/ml-restoration-training.md).
