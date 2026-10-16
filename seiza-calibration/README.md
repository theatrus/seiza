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

`FRAME_QUALITY_ALGORITHM_VERSION` changes whenever equal darks or bias
frames may measure differently, and `FLAT_QUALITY_ALGORITHM_VERSION`
whenever equal flats may, so stored measurements can be checked for
staleness without re-measuring frames whose measure did not change.
`seiza-stacking`'s `calibration_frame_quality` example runs the measures and
screens on real files.

## Matching a light by its measured angle

`optics_match` compares a flat's rotator angle with the light's within
`MatchTolerances::rotation_deg`. The light's recorded angle is the default;
`MatchTolerances::light_rotation` replaces it with an angle the host
measured (`LightRotation::Measured { degrees }`) or compares the light at no
angle (`LightRotation::Unknown`), as when the host has shown from the light's
own pixels which flats it follows. Flats judged against each other
(`optics_consistent`, `coherent_subset`) keep their recorded angles.

## Dust in flats

`dust` is the one definition of dust that `measure_flat`'s features and
`flat_response`'s comparisons share, so a flat's spots and a session
comparison always agree. A shadow is the light cone where it meets the
dust, so its size says how far in front of the sensor the dust sits
(`Optics::distance_mm`), and it comes in five shapes:

- **compact** and **irregular** dips: dust on or near the sensor window;
- **disks**: dust far up an unobstructed train, such as a refractor's
  filter, often under 0.5% deep and hundreds of pixels across;
- **donuts**: the same through a central obstruction (SCT, RASA,
  Newtonian), whose lit hole belongs to the donut and is not a spot of its
  own;
- **patches**: overlapping disks or donuts the rim vote could not tell
  apart, reported as one large shadow.

Large, smooth shadows hanging from the border are **vignetting**, reported
apart and never as dust. Disks and donuts are found by their rim — sharp,
and darker inside most of the way round — which large-scale mottle and
gradients do not have; each candidate is then checked against the rim
contrast that circles of its size show anywhere on the plane, with the
clear shadows already found taken out. Small dips must clear a threshold
raised for the number of pixels tried, so a large frame does not report its
own noise, and are looked for with each disk's own profile taken out, so
window dust in front of a filter's disk is still seen.

- `dust::find_dust` finds the dust in one plane (a binned flat or a
  session's response).
- `dust::find_session_dust` finds it on a session's stack, where noise is
  about √n lower, and says in how many of the session's flats each feature
  shows (`seen_in`): dust holds still across flats shot a minute apart,
  noise does not. `dust::frame_dust` splits one flat's view into the
  session's features it shows and transients (dew drying, a passing
  particle) that are not the light path's.
- `dust::session_dust_from_lists` does the same without the flats: from a
  stack the host kept and each flat's own features (as `measure_flat`
  stored them), paired with `dust::match_features`. It keeps features by
  the same rule as `find_session_dust`, and gives each flat's view, so a
  host that keeps those reads no flat again for its session's dust.
- `dust::find_changes` compares two sessions: new, gone and moved dust,
  with dust that moved no further than focus travel explains
  (`DustOptions::move_tolerance`) marked as within tolerance rather than
  new, stars a tracked sky flat kept left out, and vignetting changes
  apart. A feature of the ratio counts as held dust's only where that dust
  explains it: a crescent on its rim no deeper than the dust, or a change of
  its own depth; window dust that arrives in front of a held disk is new.
  `dust::match_features` pairs two feature lists the same way, taking the
  most pairs and then the least total distance. `DUST_ALGORITHM_VERSION`
  versions all of these.

## Flat responses: same light path, and which one a light follows

`flat_response` measures flat sets rather than single frames:

- `metric::build_response` reduces a session's flats (two or more, binned to
  at most 1,024 px) to one normalized response. `metric::compare` divides one
  response by another and reports the smooth difference (`broad`), the
  deepest change of dust (`dust`) and the spots behind it, and any change of
  vignetting, in fractions of the sky background. Its spots are
  `dust::find_changes`'s, above, and its `sigma` is the noise of their
  ratio's residual. `metric::change_map` is the part of the difference those
  changes cover (`CHANGE_MAP_ALGORITHM_VERSION`), and
  `metric::apply_change_map` carries a flat across it.
- `evidence::fit` asks whether a few lights' background
  (`evidence::background`) carries a change, weighing each shadow of the
  change against placebo positions where nothing changed, so nebular
  structure leaves the answer undecided rather than wrong.
- `choice::choose_response` asks which of several candidate responses a
  light's background follows: every pair is told apart by its change map and
  fitted the same way, and a candidate wins when it beats every candidate it
  can be told apart from and loses to none.

```rust
use seiza_calibration::flat_response::{choose_response, evidence, metric::Plane};

# fn example(lights: &[Plane], candidates: &[Plane]) {
let Some(background) = evidence::background(lights, 0.0, 6) else { return };
let choice = choose_response(&background, candidates, |_, _| false);
match choice.chosen() {
    Some(index) => { /* the lights follow candidates[index] */ }
    None => { /* nothing stands clear: fall back, and say so */ }
}
# }
```
