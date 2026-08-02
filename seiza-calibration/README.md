# seiza-calibration

`seiza-calibration` provides host-neutral pixel kernels for calibration
responses. It accepts caller-owned linear `f32` buffers and does not read FITS
files, manage stacks, or decide what caused an image defect.

The first API builds and applies bounded residual-response patches from
detector-aligned crops. A host must choose the source frames and show that a
feature stays fixed on the detector while the sky moves. The result does not
identify dust or another cause.

```rust
use seiza_calibration::{
    LinearImageMut, LinearImageRef, ResidualFlatOptions,
    build_residual_flat_patch,
};

# fn example(crops: &[Vec<f32>], light: &mut [f32]) -> seiza_calibration::Result<()> {
let samples = crops
    .iter()
    .map(|data| LinearImageRef::new(data, 32, 32, 1))
    .collect::<Result<Vec<_>, _>>()?;
let built = build_residual_flat_patch(&samples, &ResidualFlatOptions::default())?;
let light = LinearImageMut::new(light, 64, 64, 1)?;
built.patch.apply_at(light, 16, 16)?;
# Ok(())
# }
```

Residual responses supplement a missing or stale measured flat. They do not
replace source files or prove why a response changed.
