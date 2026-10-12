# seiza-calibration

`seiza-calibration` provides host-neutral pixel kernels for calibration
responses, rules for matching calibration frames to lights, and grades for the
bias, dark, and flat frames that go into a master. It accepts caller-owned
linear `f32` buffers and does not read FITS files, manage stacks, or decide
what caused an image defect.

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

## Grading calibration frames

Measuring and judging are separate. `measure_flat` and `measure_dark` read one
raw frame and return a small, serializable measurement, so a host can measure
each frame once and judge it again whenever its set changes. `screen_flats`,
`screen_darks`, and `screen_bias` judge one set — frames that should match —
and give every frame a verdict of ok, suspect, or poor, with a stable code and
the numbers behind each finding. Every limit has a default the caller can
override.

```rust
use seiza_calibration::{
    FlatMeasureOptions, FlatScreening, Grade, LinearImageRef, measure_flat,
    screen_flats,
};

# fn example(frames: &[Vec<f32>]) -> seiza_calibration::Result<()> {
let measurements = frames
    .iter()
    .map(|data| {
        let image = LinearImageRef::new(data, 9576, 6388, 1)?;
        measure_flat(image, false, &FlatMeasureOptions::default())
    })
    .collect::<Result<Vec<_>, _>>()?;
let report = screen_flats(&measurements, &FlatScreening::default())?;
for verdict in &report.frames {
    if verdict.grade == Grade::Poor {
        // Leave the frame out of the master, or show the findings.
    }
}
# Ok(())
# }
```

What each screen looks for:

- **Flats:** clipped samples, a level too near full scale or too dim, an
  illumination pattern that differs from the set's, and compact dips. Dips are
  matched across the set: steady ones, such as dust shadows, are counted and
  left alone; dips that come and go or change depth mark the frames that hold
  them, and those in more than half the frames mark the master, because a
  median keeps them. A flat's temperature is never judged.
- **Darks:** a level above the quietest frame's, the mark of stray light, and
  noise or hot pixels out of line with the frames whose level passed.
- **Bias:** a level or noise out of line with the set, and row or column
  banding.

`FRAME_QUALITY_ALGORITHM_VERSION` changes whenever equal frames may measure
differently, so stored measurements can be checked for staleness.
`seiza-stacking`'s `calibration_frame_quality` example runs the measures and
screens on real files.
