# seiza-stacking

`seiza-stacking` provides linear, incremental image stacking for
astrophotography applications. It keeps plate solving and catalog access out of
the stacking path while reusing Seiza's star detector for local registration.

The first release supports:

- mono, planar RGB, and Bayer FITS or XISF inputs in linear sensor units;
- optional master bias, dark, and flat calibration;
- bounded-memory, two-pass construction of bias, dark, and flat masters;
- bounded-drift star registration with translation/rotation/scale refinement,
  and optional affine or quadratic polynomial fits for lens distortion;
- robust global or tiled local normalization, or a global gain with local
  background offsets that removes frame-edge seams;
- online residual (delta-sigma) rejection with coverage and rejection maps;
- three-pass completed-stack rejection which revisits early transient samples;
- non-mutating frame admission gates for additive live stacks;
- versioned, checksummed, atomic live-stack checkpoints that can be reopened;
- compact immutable export snapshots that clone only the integrated mean;
- floating-point FITS output on the reference frame's pixel grid.

## Completed-stack rejection

Registration admission uses an absolute `maximum_registration_rms_pixels`
limit in original sensor pixels, independent of earlier frames' residuals.
There is no running-median RMS gate. Plate-solve catalog residuals measure a
different fit; a similarity registration cannot model the curvature a quadratic
mapping can address. Fraction admission thresholds accept equality.

For a camera whose normalized flat response transfers across gain/offset
settings, `StackOptions::match_tolerances.allow_flat_gain_offset_mismatch`
opts into using a different-gain flat. The loaded master must record
`DARKSUB=T`; normalization alone does not qualify. Gain/offset matching for
light darks and flat-darks, sensor/readout-mode matching, and coherence within
each master remain strict. The policy is retained in contexts and replay.

`LiveStacker::with_reference_image` starts an empty stack using a separate
FITS/XISF image for alignment only. It also takes a channel reference light,
which it calibrates and registers onto the alignment grid to supply the
normalization target. Noise weighting uses that light's calibrated noise,
measured before resampling. The alignment image's brightness and noise never
set the channel's photometric scale. Push every light explicitly, including
the channel reference; the external image contributes no observations unless
also pushed. Online integration, retained-frame replay, and drizzle use its
pixel grid and output WCS. Prepared channel counts must match.
Saved contexts retain both references; older readers reject these contexts,
while ordinary stack contexts retain their existing format.
`LiveStacker::open_fits_with_reference_image` does the same from paths, with
`open_fits`'s duplicate-path checks. The reference image may also be one of
the lights, and `protected_paths` keeps outputs and contexts from replacing
it. `LiveStacker::with_reference_image_candidates` takes several channel
references, best first, prepares the alignment image once, and uses the first
that aligns to it; a channel count that differs from the alignment image's
ends the search at once. `LiveStacker::open_fits_with_reference_image_candidates`
does the same from paths.

`LiveStacker::with_expanding_reference` plans a canvas for a connected set of
overlapping lights, such as mosaic panels or a dithered stack whose edges
should be kept, and starts an empty stack on it. The first path anchors the WCS
and photometric scale. Every other frame registers against the footprints of
the panels already placed, not the whole canvas, and a frame that starts a new
panel is refit against every star it covers. Frames may differ in rotation by
any amount. A frame that cannot join is rejected with the reason when pushed;
planning fails when nothing joins the anchor, an input changes or cannot be read,
or the canvas limit is exceeded. The canvas supplies alignment and
normalization only, so push every light, including the anchor: planned paths
reuse their planned transforms, and fail if the file has changed since, while
other frames register through the panels. `maximum_samples` bounds the
canvas's pixels times channels. A similarity places every frame; a frame
lying mostly within one panel also gets the registration model's warp
against it, which follows lens distortion across a meridian flip, while a
panel's first frame keeps its similarity and comes from the anchor's side of
the pier where one connects. The lens bends neighboring panels' shared edges
opposite ways, which those similarities cannot follow, so once every frame
has joined, one radial distortion (two coefficients about an axis that may lie
off the sensor's center) is fitted where the panels' first frames overlap,
with each panel's camera rotation onto the anchor's tangent plane when the
anchor's `FOCALLEN` and `XPIXSZ` give its focal length. When it explains what
the similarities leave, the canvas is planned again on undistorted grids, and
every frame's warp leads through the lens onto its own pixels;
`LiveStacker::expanded_lens` reports what was measured. Panel starts approximate
the fitted sky projection with a quadratic, or a cubic when needed, retaining
a 0.02-pixel error limit checked on both fitting and independent grid points.
A failed rebuild keeps the original plan and reports its reason in the planning
notes. Later planned frames match stars through that geometric prediction before
refining their warp; a similarity alone can miss stars near the edges of a wide
tangent plane even when its central matches have small residuals. Before fitting, known
camera, dimension, binning, pixel-size or focal-length conflicts among joined
frames disable the shared lens fit; pixel size and focal length allow 1% rounding
tolerance. Missing metadata allows fitting with an unverified-compatibility
note. Filter and date changes are ignored. `expansion_planning_notes()` exposes
these notes for newly planned stacks, first among them how many lights did not
join the canvas and, for those that matched a panel but overlapped it too
little, the largest overlap they reached against `minimum_overlap_fraction`;
reopening a context does not rerun planning.
`FrameAcceptanceCriteria::EXPANDED_MINIMUM_OVERLAP_FRACTION` (0.05) is the
minimum overlap the CLI, the C ABI and Python give an expanded canvas when the
caller sets none: panels planned for 10% overlap land nearer 9%. A lens-corrected
canvas drops the original WCS, including SIP: plate-solve its output to obtain
astrometry for the undistorted grid. Without lens correction, the anchor WCS
is retained with its reference pixel shifted. Bayer drizzle and
drizzled reintegration are refused. Each frame is resampled, normalized and
integrated over the bounding box of its own footprint on the canvas, not the
whole canvas, which gives the same stack to the bit, so a frame costs about
the same however many panels the canvas holds. (Only a lens distortion
strong enough to fold back onto the frame far beyond it, where it was never
measured, could have sampled the frame a second time there.) On a canvas of nine
2048x1536 mono panels a frame took 23ms rather than 51, or 34 rather than 78
normalized by local background; of nine 1536x1024 colour panels, 30 rather
than 79, or 67 rather than 141. Reintegration likewise keeps, reads and
integrates each frame over its footprint: on the nine mono panels, 14ms a
frame from the frames kept rather than 31, and 25 rather than 79 from the
sources (23 rather than 39, and 42 rather than 119, normalized by local
background). Saved contexts keep the panels, each
with the stars planning matched it by, and the lens, so a frame added after
reopening registers exactly as it would have before saving; older readers
reject them. A context saved before panels kept their stars still opens,
and its panels then match the stars of the canvas beneath them.

Live delta-sigma rejection cannot remove trails admitted during its warm-up.
`LiveStacker::reintegrate` fixes that for a live stack. The stacker keeps a
ledger of every admitted frame: its source file, the calibration masters that
applied, and the registration and normalization mapping the live pass chose.
Reintegration prepares each frame exactly as the live pass did, keeps it in a
scratch file (or reads the one `LiveStacker::retain_frames_for_reintegration`
kept during the live pass), and integrates it with leave-one-out rejection,
leaving the live stack unchanged. The three passes read the scratch files a
band of rows at a time and run on one band of every frame before the next, so
each file is read once for them and each pixel's statistics stay in the
processor's caches; the result is bit-for-bit that of reading every frame whole
on each pass. Star detection and registration do not run again. A source
file that changed since it was stacked is refused, and
`LiveStacker::reintegration_unavailable` says why a stack cannot be replayed:
frames pushed as pixels, frames calibrated with in-memory masters that were
since replaced, or a context saved before the ledger existed. Saved contexts
carry the ledger from format version 3; versions 1 and 2 still open.

Hosts that keep their own records can call
`integrate_registered_frames_banded` directly to reread admitted frames with
their saved calibration and `RegisteredFrameMapping`. Its loader runs once per
frame; each frame is kept in a scratch file inside
`BatchStackOptions::scratch_directory` (the system temporary directory by
default, four bytes per sample, removed before it returns), and the three
passes read the files back a band of rows at a time, as
`LiveStacker::reintegrate` does. A progress callback hears
`BatchStackPass::Estimate` before each load and `BatchStackPass::Integrate`
as the bands are integrated, and `BatchStackOptions::cancel` stops the work.
`integrate_registered_frames` is the same estimator with no scratch files: its
loader receives `BatchStackPass` and the frame index and runs on each pass.
Both give the same result to the bit. `BatchStackResult` includes the final mean, variance,
coverage, rejection maps, and per-frame finite/integrated sample counts.

The estimator uses leave-one-out sigma clipping rather than comparing a
transient to statistics that include its own pixels. It removes sufficiently
strong isolated early or late trails at three or more finite observations;
shallower pixels are averaged without rejection. Student-t predictive limits
preserve the configured Gaussian tail probabilities when only a few other
frames estimate noise, protecting ordinary low-depth noise samples from
over-rejection.

One large outlier inflates the dispersion its pixel's other samples are judged
by. In a 98-frame M45 stack a satellite trail 50 000 ADU above the sky raised
that pixel's sigma from about 300 to 5 000, so hazy frames 2 000 to 9 000 above
the sky survived along the trail and nowhere else, drawing the trail's line in
the result. So the first pass estimates each pixel's moments without its
largest and smallest sample once it has ten, the second estimates them again
from only the samples the first keeps, and the third rejects against the
second. Both estimates are corrected for the variance that trimming and
clipping take from Gaussian noise, so plain noise is still rejected at the
configured rate. Three or more large outliers at one pixel can still mask one
another. Normalized frames have equal weight unless
`BatchStackOptions::frame_weights` supplies weights (see below), and
`BatchStackOptions::minimum_sigma` is in their physical sample units.
Through `integrate_registered_frames` memory stays proportional to the output
image (about 40 bytes per sample, 44 weighted, plus one loaded input and one
bit per sample per frame), and each frame is loaded three times. Changed shapes
or sample checksums abort the result. Reintegration from scratch files, by
`integrate_registered_frames_banded` or `LiveStacker::reintegrate`, needs 16
bytes per sample (20 weighted) plus `BatchStackOptions::band_memory_bytes` and
one loaded input, and loads each frame once; on 45 frames of a 26 MP mono
sensor replayed from their sources, that took 8.6 s and 5.7 GB at peak against
23.2 s and 9.9 GB. When the system temporary directory cannot hold the files,
`integrate_registered_frames_banded` falls back to the three loads.

## Frame weighting

By default every admitted frame counts equally. Set
`StackOptions::weighting` to `FrameWeighting::inverse_noise_variance()` to
weight each frame by the inverse of its noise variance instead. The stacker
measures each channel's pixel noise on the calibrated frame before
resampling, using the same second-difference estimator as `measure_depth`
(public as `frame_noise`), and scales it by that channel's normalization
gain. A frame's weight is
`(reference noise / frame noise)^2`, clamped to 0.05..=20 by default, so the
reference frame has weight 1 and a frame twice as noisy has weight 0.25.
Delta-sigma rejection widens its limits for low-weight frames to match their
noise.

`FrameDiagnostics::noise` and `FrameDiagnostics::weight` report the values
for each accepted frame, and `LiveStacker::reference_noise` reports the
reference's. They are empty in equal mode, which measures nothing. The
admitted-frame ledger keeps each frame's weight, so `LiveStacker::reintegrate`
replays a weighted stack with the live weights and does not measure noise
again. To replay frames yourself with `integrate_registered_frames_banded`, pass the
recorded weights, with 1 for the reference, as
`BatchStackOptions::frame_weights`.

Coverage still counts frames. In weighted mode the variance output is the
variance of a weight-1 frame, not the plain sample variance. Weighted live
contexts use checkpoint format 4: format 3 plus per-sample weight sums and
the reference noise after the rejection counts, and each admitted frame's
noise and weight after the ledger. Formats 1 to 3 open only as equal-weight
stacks, and format 4 only as a weighted one. Equal-weight stacks keep writing
format 3, byte for byte as before, and their options and configuration
fingerprint do not change.

## GPU resampling and integration

With the default `gpu` feature, defect learning's first screen, VNG
debayering, star detection, registration resampling (clamped Lanczos-3 and
bilinear), the live stack's accumulator and the final integration's three
rejection passes run on the GPU when one is present, through wgpu: Metal, Vulkan or
Direct3D 12. Frames under a million samples stay on the CPU, which is as
quick for them. The GPU works in single precision and stays within a few
thousandths of the frame's noise of the CPU, and the defect map comes out
bit for bit as the CPU's (see `docs/design/gpu-stacking.md`); on a 45-frame,
26 MP stack with reintegration it halves the wall time and quarters the CPU
time.

Each request says which passes may run there, in its `GpuOptions`:
`StackOptions::gpu` for a live stack and its reintegration,
`BatchStackOptions::gpu` for `integrate_registered_frames_banded`, and
`DefectDetectionOptions::gpu` for defect learning. `GpuOptions::cpu()` keeps
every pass on the CPU, and `defects`, `debayer`, `detection`, `resampling`,
`accumulator` and `integration` keep one each; `concurrency` (3) is how many
frame calls the GPU takes at once, and `trace` prints each call's timings to
stderr. A stack's context keeps its GPU options, and its configuration
fingerprint leaves them out. Calls that take no options, such as
`resample_to_reference` and `FitsFrame::into_prepared`, use the defaults.
Build with `default-features = false` for a crate without wgpu. No adapter,
or any GPU error, falls back to the CPU; an error is logged once and turns
the GPU off for the rest of the process.

## Pipelined preparation with a shared pool

`LiveStacker::push_fits_pipelined_with_pool` takes a host-owned Rayon pool and
coordinates outside it. Scoped reader workers overlap storage reads and
decoding with other frames' preparation and ordered integration. Calibration,
cosmetic correction, debayering, registration, normalization and integration
run in the supplied pool, not the global one. Callbacks stay on the coordinator
thread and results retain sequential order. Calling from any Rayon worker
instead selects a safe sequential fallback, including one-thread pools.

`PoolPipelineReport` contains frame counts, execution mode, actual worker count,
a frame-memory estimate and aggregate read/decode, preparation, integration,
coordinator-wait and whole-batch elapsed times. Preparation and integration
timers exclude waits to enter the compute pool. Worker sums include discarded
in-flight work and may exceed elapsed batch time; they are not CPU timings.

This API caps explicit worker requests by `max_in_flight_bytes`. Its estimate
reserves 80 bytes per reference pixel for each mono worker, 112 for RGB, and
one extra integrating frame. A budget below one worker plus that frame fails
before reading. These estimates include preparation scratch and buffered
frames, but assume source images no larger than the reference and are not hard
RSS limits. Hosts must also reserve their reference, accumulator, calibration
masters, larger inputs and codec/allocator overhead. Use
`LiveStacker::pipeline_memory` to share the same worker/integration estimate
when planning the host's budget, and
`CalibrationMasters::image_buffer_bytes` to count resident master samples;
session masters and their active deep clones both consume storage.

When even one queued worker would exceed the host's memory policy, use
`push_fits_sequential_with_pool(paths, normalized_full_scale, pool, callback)`.
It reads, prepares and integrates one frame at a time without reader threads
or a queue budget. It reports `SequentialRequested`, zero coordinator wait,
and `PoolPipelineMemory::sequential_bytes`: `64 * pixels + 8 * samples` of
reference-sized scratch.

On an expanded canvas a frame is resampled onto only the part of the canvas
its footprint covers, so `LiveStacker::pipeline_memory` costs each frame at
the largest frame planning read, landing at any rotation and at the scale
`FrameAcceptanceCriteria::maximum_scale_deviation` admits, rather than at the
canvas's size: `64 * frame pixels` and `16 * footprint samples` for a worker.
Pipelined pushes, and the derived worker count of `push_fits_pipelined`,
follow it. On a four-panel mosaic of 88 megapixels from 26-megapixel frames,
a worker is costed at 2.4 GiB rather than 6.5 GiB. A reopened context, which
does not plan again, costs its frames by its largest panel. The host must reserve its persistent stack and master
buffers separately. Do not inflate a pipeline budget to request this mode.
Cancellation stops before the next frame is read, and callbacks remain on the
caller even when called from a one-thread Rayon pool.

The original `push_fits_pipelined` API and its explicit-worker override remain
unchanged. Its existing Rayon fallback is intentional: wrapping the entire
batch in `pool.install` does not enable overlapped preparation.

## Canonical sky orientation

`SkyOrientationPlan` reprojects an integrated mono or RGB image onto a
north-up, east-left TAN grid. It keeps the solved field center and geometric-
mean pixel scale, expands the output enough to retain the four source corners,
and returns the exact source-to-output `AffineTransform`. The transform handles
camera parity as well as rotation, so two stacks from different optical paths
share one display convention.

The affine path accepts an undistorted TAN solution. It rejects missing,
singular, or SIP WCS rather than labeling an unknown view as sky-up. Its FITS
cards replace the source matrix as one unit and record `SKYORIEN =
'N-UP E-LEFT'`. Use the plan's `fits_header_cards` with
`write_processed_image_fits_f32` when publishing the reprojected image.

`RegisteredFrameMapping::extract_region_after_affine` maps a crop on that
oriented output back through the original registration and normalization. This
keeps source-frame inspection aligned with the displayed stack without
building every full registered frame. The existing similarity-only method now
uses the same affine path.

After integration, `combine_rgb`, `combine_lrgb`, `combine_super_lrgb`,
`combine_super_rgb`, and `combine_narrowband` compose aligned mono stacks
without coupling color into the live accumulator. The direct SHO permutations
and HOO remain linear-light; LRGB replaces or blends CIE luminance while
preserving RGB chromaticity, super-LRGB uses the additive `L + R + G + B`
target, and super-RGB synthesizes the same target as `R + G + B` with no
luminance stack. `NarrowbandMatrix`
supports arbitrary static SII/H-alpha/OIII mixes. Foraxx-SHO and Foraxx-HOO
are intentionally marked display-referred because their published dynamic
factors operate on stretched channels. That preparation resolves the shared
parameterized Auto-MTF model from `seiza-stretch`. `write_color_fits_f32`
records the palette and transfer semantics alongside preserved reference WCS
cards. See the [color-composition design](../docs/design/color-composition.md).

`calibrate_color` sets the gains of a linear RGB image from Gaia DR3 star
colours. `calibrate_color_image` places the Gaia stars itself, from the
offline photometry catalog or sources the caller has, and applies the gains;
`calibrate_and_stretch` then display-stretches the result. Neither fails when
calibration cannot run: the outcome gives a typed reason and the image stays
uncalibrated. `transform_wcs` carries a reference frame's solution through
registration, a turn and a crop to a composed image. See the [colour
calibration design](../docs/design/color-calibration.md).
The composition functions themselves require aligned inputs. File-oriented
callers can use `Registrar` plus `resample_to_reference`; the CLI does this
automatically for every non-reference filter stack. `ColorOptions` controls
normalization and declares whether inputs are linear or already display-
referred. This lets an embedding application independently stretch each mono
input before composition; Foraxx then skips its shared preparation pass rather
than stretching those inputs twice. The separate `ForaxxOptions` controls the
default display preparation used only when dynamic Foraxx palettes receive
linear inputs.

`ColorOptions::crop` trims a composition to the area every channel covers,
which registration otherwise leaves ringed with `NaN`. `ColorCrop::Bounds`
keeps the bounding box of the covered pixels; `ColorCrop::Inscribed` keeps the
largest rectangle every channel covers in full, removing the corners a rotated
or meridian-flipped channel leaves behind. Percentile levels are then estimated
from the kept region alone, and composition indexes back into the caller's
full-size planes rather than copying cropped channels. The result carries its
`ReferenceRegion` on the input grid, and `write_color_fits_f32` moves `CRPIX`
onto that grid. `ColorComposition::crop` reports each channel's own coverage
and flags one that sits far from where the others agree — the channel that
pulled the crop in. `crop_report` and `covered_region` expose the same search
for callers cropping something other than a composition; `crop_report` takes
borrowed `ChannelSamples`, so a host measuring its own buffers copies nothing.

Callers that inspect a small area can use `resample_region_to_reference` with
a `ReferenceRegion`. It applies the same transform and interpolation as the
full-frame path while allocating only the requested crop. The returned crop's
origin is `(0, 0)`; its pixels still come from the region's absolute reference
coordinates. `FitsFrame::into_prepared` exposes the same CFA-to-RGB preparation
used by `LiveStacker` for callers that need to inspect those registered crops.
Accepted-frame diagnostics retain the serializable `NormalizationMap` used by
the stacker inside a versioned `RegisteredFrameMapping`. The mapping validates
persisted coefficients and owns the order of registration and normalization.
Its `extract_region` method reproduces a bounded part of the registered frame;
`extract_region_after` also handles a second registration stage, such as a
channel-to-color mapping. Global normalization keeps that path bounded, while
local normalization preserves the exact two-stage processing order.

`build_residual_flat_patch` remains as a compatibility adapter for one release.
The pixel kernel now lives in
[`seiza-calibration`](https://crates.io/crates/seiza-calibration), whose borrowed-buffer API
does not require `LinearImage` or the stacker. Both paths estimate a small
multiplicative response patch
from at least five calibrated light-frame crops taken at the same detector
coordinates. It fits and removes each crop's local background plane, smooths
pixel noise and moving stars, and keeps only repeated dark response. The patch
retains only its largest connected correction region, blends to a neutral
edge, and caps its correction gain. This rejects scattered low-level noise
even when many individual pixels cross the depth threshold. The function does
not identify dust: the host must first show that the feature stays fixed on
the detector while sky content moves, then ask the user before applying it.
Diagnostics include `RESIDUAL_FLAT_ALGORITHM_VERSION` for cache keys and
provenance.

Residual patches run after ordinary bias, dark, and flat calibration but
before registration. `LiveStacker::from_prepared_frame` lets a host retain the
reference FITS headers after it performs that extra step. Later inputs use
`push_linear` after the host applies the same calibration, patch, and CFA
preparation order. The stacker rejects `push` and `push_fits` in this prepared
input mode, and a saved context retains that rule. A residual patch supplements
a missing or stale flat; it is not a new master flat and does not change the
saved source files.

`LiveStacker::push` is the embedding API intended for acquisition tools and
PSF Guard. The CLI's `seiza stack` command feeds files through the same state
machine. Frame-quality scoring remains the host application's responsibility;
the crate's admission gates cover only compatibility and numeric/geometric
safety. Live renderers can borrow `LiveStacker::view` without copying the
full-resolution accumulator; any display stretch remains a caller-only visual
operation.

When masters are active, every decoded reference and pushed path is checked
before pixel calibration. Known calibration-frame roles, integrated masters,
and inputs already marked bias/dark/flat-corrected are refused. Shared
`seiza-calibration` matching verifies sensor/readout compatibility for every
master, dark temperature, and flat optics. Two recorded readings must agree; a
reading the master never recorded, such as a dark without `CCD-TEMP`, does not
refuse the light. `CalibrationMasters::from_fits_frames` and `from_fits_paths`
keep each master's header signature; `CalibrationMasters::new` sees only
pixels, so it can check little more than geometry. Bias-isolated dark current
may be scaled by exposure; a dark that still contains its pedestal must have
the same known positive exposure as the light. With no masters, preprocessed
lights remain valid inputs.

The gates are `seiza_calibration::MatchTolerances`, and a live stack keeps
its own in `StackOptions::match_tolerances`, the defaults unless set. A host
that chose its masters under its own tolerances — a rotator that scatters more
than two degrees, or a flat borrowed on purpose from the nearest other angle —
gives the stack the same gate, and the reference, every push, pipelined push
and channel reference are judged under it. `LiveStacker::set_match_tolerances`
changes the gate for the lights pushed from then on, and a master swap leaves
it as it was. Each admitted frame keeps the gate it was admitted under, and a
reintegration judges it by that gate, so a host can widen the gate for one
light and narrow it after. Options, a saved context and the configuration
fingerprint record only the tolerances a light is admitted by;
`master_temperature_c` and `flat_session_seconds` read back as their defaults.

Tolerances that are NaN, infinite, negative, or a rotation above 180 degrees
are refused with an error, never clamped: `StackOptions::validate`,
`set_match_tolerances`, `compatible_for_light_with` and the `_with` validators
return it, and `save_context` will not write such a context.
`MatchTolerances::validate` is the same check for a host's own input. A
rotation of 180 already accepts any angle.

A light's recorded rotator angle can be wrong for the question a flat asks:
a rotator that re-homed reads the same angle for another position, and two
angles far apart can share one light path. A host that measured which flats a
light follows — from a plate solve, or by fitting its sky against each
candidate flat set (`seiza_calibration::flat_response`) — says so in
`MatchTolerances::light_rotation`: `LightRotation::Measured { degrees }`
compares the master flat with that angle in place of the light's recorded
one, and `LightRotation::Unknown` compares it at no angle, as for a rig
without a rotator. The default, `Recorded`, is unwritten, so existing
options, contexts and fingerprints keep their bytes; a measured angle is part
of the gate each frame is admitted and reintegrated under.

Masters given to `set_calibration` live only in memory, so once they are
replaced, frames admitted under them can no longer be reintegrated
(`reintegration_unavailable` says so). `set_calibration_from_fits_paths` keeps
reintegration.

Outside a stack, `validate_light_frame_with` and `validate_light_signature_with`
judge one light under the tolerances you pass, for a host that runs its own
pass over the lights. `compatible_for_light_with` asks the same questions and
names what it would set aside, so validation passes exactly when it sets aside
nothing. A refusal names the reading that failed and the tolerance in force, as
in `rotation light=113.8deg master=3.2deg (110.60 deg apart, tolerance 2.00)`.

Darks that caught stray light, such as dawn through an open roof, look like
ordinary darks in their headers. `MasterBuildOptions::dark_level_screening`
measures each dark's robust level and sets aside any frame more than a set
number of pixel-noise sigmas above the quietest input, naming it in
`MasterFrame::skipped_inputs`. It is off by default. `DarkLevel::measure` and
`screen_dark_levels` apply the same rule to a catalog without building a
master.

For a full-resolution output that must not stop a live session, use
`LiveStacker::export_snapshot`. It freezes the finalized mean and scalar frame
counts without cloning variance, coverage, or rejection maps. The independent
owner can move to an output worker and be written with
`write_stack_export_fits_f32` while the accumulator continues. The older
`snapshot` API is unchanged for callers that need all diagnostic maps.

Long-running acquisition tools can checkpoint without consuming the live
handle and reopen the exact online estimator later:

```rust
stacker.save_context("m31.seiza-stack")?;

let mut stacker = LiveStacker::open_context("m31.seiza-stack")?;
stacker.push_fits("light-042.fits")?;
```

The context retains the original prepared registration reference, calibration
masters, stack options, Welford mean and second moment, coverage and rejection
maps, frame counters, compatible FITS headers, and the source-path ledger.
Version 2 additionally retains the normalized reference/master signatures and
whether a dark is safely exposure-scalable. The reader migrates version-1
contexts; if an old context contains masters, later file pushes fail closed
until the caller reloads those masters because v1 cannot prove their metadata.
No-master v1 contexts continue normally.

Writes use an adjacent temporary file and publish by atomic rename only after a
checksummed compressed payload is complete. A context is mutable processing
state, not the final interoperable image product; use `write_fits_f32` for the
finished FITS/XISF artifact.

`StackOptions` and its nested registration, normalization, rejection, and
acceptance types serialize through Serde. Omitted object fields use the same
Rust defaults, while unknown fields are rejected. This is the configuration
contract used by `seiza-cabi`; normalization and rejection enums use adjacent
`mode` / `options` objects so additional algorithms do not change the native
function signatures.

Frame admission remains ordered because online rejection depends on prior
observations. Independent work within each frame—calibration, registration
detection, resampling, normalization, classification, and integration—uses the
shared Rayon worker pool. Applications may set `RAYON_NUM_THREADS` or install
stacking work in a configured Rayon pool when they need to reserve CPU for
acquisition and display work.

## Overlapping frames

Pushing frames one at a time leaves the machine half idle: while a frame is
read and decoded the cores wait, and while it is registered the disk or network
waits. Only integration depends on the frames before it — everything up to and
including normalization reads immutable state, so it can run for several frames
at once.

`push_fits_pipelined` does that, handing results back in the order given:

```rust
let report = stacker.push_fits_pipelined(&paths, &PipelineOptions::default(), |path, outcome| {
    record(path, outcome);
    if cancelled() { Continue::No } else { Continue::Yes }
})?;
println!("{} integrated, {} rejected, {} failed", report.integrated, report.rejected, report.failed);
```

The callback keeps the caller in charge of cancellation, checkpointing, and
per-frame decisions, which is why this is not a batch call that swallows the
loop. The accumulator is still fed strictly in submission order, so a
pipelined run is bit-identical to a sequential one; tests assert that against
the same frames both when every frame is accepted and when the order-dependent
`minimum_integrated_fraction` gate turns frames away.

The callback sees everything a `push_fits` loop would, errors included: a path
that cannot be opened, or that repeats one already stacked, arrives as `Err`
for that path alone and the run carries on. One repeated path in a directory
listing costs that path, not the batch — and costs no read either, because
repeats are settled before anything is opened, which is also what keeps the
concurrent and sequential paths reporting the same error. Since a run can fail
every path and still answer `Ok`, the returned `PipelineReport` carries the
counts a caller checking only the return value needs.

`Continue::No` stops the run, but does not reach back into reads already begun.
Each worker may hold one prepared frame and be building another, so up to two
frames per worker beyond the cancel point have been opened and are finished and
discarded. A read cannot be interrupted once started: a path on a stalled
mount, a FIFO, or a device node holds the call until that read returns. Cancel
promptness is bounded by the slowest read already in flight.

Called from inside a Rayon pool thread, including under `pool.install(..)`,
frames are prepared sequentially on the calling thread instead. Parking a pool
thread while preparation needs that same pool would deadlock, and a caller who
installed a pool did so to reserve cores, which spawning outside it would
quietly undo.

PixInsight writes floating-point images normalized to `bounds="0:1"`, so such a
frame's samples run 0..1 where a camera frame's run in the thousands and a group
mixing the two normalizes against values four orders of magnitude apart. Set
`PipelineOptions::normalized_full_scale` to the scale the rest of the frames use
— 65535.0 for 16-bit camera data — and such a frame arrives comparable;
`FitsFrame::rescale_declared_unit_bounds` does the same for a frame opened by
hand. Only an exact `0:1` is converted, because that is the one spelling whose
meaning is settled: this crate's own writer reports the observed sample minimum
and maximum, so converting from any other declared range would as easily stretch
an already-physical frame. It is off by default, since only a caller knows what
scale its other frames are on.

`PipelineOptions::max_in_flight_bytes` bounds the memory rather than the frame
count, because a prepared frame is the reference image's size and that differs
by an order of magnitude between a guide camera and a full-frame sensor. The
derived worker count falls out of that budget and the machine's parallelism; a
budget too small for even one frame ahead still runs, one frame at a time. An
explicit `workers` is taken at its word rather than held to the budget, since a
caller who names a number has usually measured something this crate cannot see;
memory then follows the count given, roughly two prepared frames per worker, up
to a hard ceiling of `MAXIMUM_WORKERS` threads.

Each frame is read on the worker that will prepare it, so reads overlap both
with each other and with the integration of earlier frames. Measured on a
16-core machine over twelve 12MP frames, against a sequential loop:

| read latency per frame | sequential | pipelined | |
|---|---|---|---|
| warm local cache | 2.00s | 1.01s | 2.0x |
| 50ms | 2.67s | 1.16s | 2.3x |
| 150ms | 3.85s | 1.44s | 2.7x |
| 300ms | 5.66s | 1.89s | 3.0x |

The latency rows were produced by delaying each read, so they model network
storage rather than measuring a real one.

Preparation is already Rayon-parallel internally, so on local storage the gain
comes from covering each frame's serial gaps and the curve flattens around six
workers — which is what the derived default targets. Remote frames want more:
at 300ms, eleven workers finished in 1.58s against 1.89s for the derived six.
Set `PipelineOptions::workers` when the frames are known to be remote, since
this crate cannot tell a network mount from a local disk.

## Bayer frames

Bayer frames are demosaiced with VNG (variable number of gradients, Chang,
Cheung and Pang 1999), PixInsight's default, after balancing the mosaic's
channels to green by their medians; each photosite keeps its own sample. On a
98-frame M45 stack, bilinear interpolation left red and blue 0.42px apart and
stars 3.1px wide; VNG leaves 0.10px and 2.7px (WBPP: 0.09px and 2.6px), and a
star's red and blue stay within 0.75 to 1.16 of its colour across its profile.
Malvar, He and Cutler's linear interpolation, tried first, gave 2.4px stars
but rang around them: unbalanced it left a cyan halo (red at half the star's
colour three to eight pixels out), and balanced, a red ring two pixels out.
The two-pixel border keeps bilinear estimates.

`StackOptions::demosaic` chooses the method: `Demosaic::Vng` (the default),
`Demosaic::Mhc` for the sharpest stars at the cost of a coloured ring around
small ones, or `Demosaic::Bilinear`, the fastest. The CLI takes
`--demosaic vng|mhc|bilinear`, Python `StackOptions(demosaic=...)`, and the C
API's options JSON `"demosaic": "vng" | "mhc" | "bilinear"`.

## Sensor defects without a master dark

A master dark subtracts a sensor's hot pixels. Without one, every light keeps
them, and they do more harm than a few bright dots. Online rejection never
judges a pixel's first few samples, so the reference's and the first frames'
hot pixels reach the stack at their dithered positions; and demosaicing turns
each hot photosite into a small coloured blob that star detection takes for a
star. On a synthetic set of 20 dithered Bayer lights with 630 defects, those
blobs pulled registration onto the sensor's fixed pattern (frames registered
at 0.0px drift, half of them then refused at normalization).

Dithering tells a defect from the sky: the sky moves across the sensor between
frames, and a defect does not. `detect_defects` reads up to 16 of the lights,
evenly spaced, calibrates them with the masters there are, and flags a pixel
when it stands five sigma from the median of its same-colour neighbours, the
same way, in 60% of them (`DefectDetectionOptions`), standing above all its
neighbours but at most one, a partner in a pair. A star's core lands on any one
pixel only in the frames the dither puts it there. A pixel whose surroundings
rise above their own background in most of the frames that flagged it, as a
star's do, is never flagged, even when the star sat still: on a raw colour
frame, judged by the adjacent photosites of the other colours, which a tight
star still lights where its own colour's neighbours, two pixels away, have
faded. Fewer than three lights give an empty map.

`StackOptions::defects` carries the `DefectMap`, raw-frame pixel indices; every
light the stack calibrates, the reference and replayed frames included, has
them repaired after calibration and before the cosmetic filter, debayering
and registration. A steady hot pixel is repaired as a dark would repair it:
the map holds its excess over its neighbours, the median across the lights
that found it hot, at their median exposure, and each light has that,
scaled to its own exposure, subtracted. A star the dither carries onto the
pixel then keeps its core, where its neighbours' median would cut it (on ten
180s O-III lights of an ASI2600MM, a star that sat on a warm pixel in three
of them lost a fifth of its peak). A dead pixel, one whose excess wanders by
more than its noise and a tenth, a light the subtraction would leave far
below its neighbours, and every pixel of a map given as bare pixels take the
median of their healthy same-colour neighbours instead. A light of another
size is left as it is. The map is serialized in the options,
so contexts keep it and it enters the configuration fingerprint; options
without one keep their exact bytes. The CLI's `--defects auto` (the default)
learns a map whenever no `--dark` is given, Python's `stack_fits` does the
same, and `detect_defects` and the C API's `seiza_stack_detect_defects_json`
give it on its own.

On the 20-light synthetic sets, the pixels a defect left more than 0.02 off
the defect-free stack fell from 807 to 6 (mono) and from 2 356 to 1 (Bayer),
with every Bayer light registered; defect-free lights stacked bit for bit as
before, and so did stars that never moved, down to mono stars 1.4px across and
Bayer stars 2.2px. Only stars 1px across that never moved lost a little (14
pixels moved by more than 0.002 of the sky, none by more than 0.02). On ten
180s H-alpha lights from an ASI1600MM the map held 416 pixels and removed
every isolated spike of 11 000 to 20 000 ADU over a 2 000 ADU sky that both
the uncalibrated stack and the stack calibrated with a 180s master dark still
held, reintegrated or not: the session dithered every third light, so three
samples of each hot pixel coincided, too many for rejection to single out. On
sixteen 300s lights of a 61 MP sensor the map held 86 000 pixels and took 4 s
of the stack's 16 s.

## Registered mosaic backgrounds

Set `BatchStackOptions::mosaic_panels` to one panel ID per input to
`integrate_registered_frames` or `integrate_registered_frames_banded`.
Repeated IDs group observations of the same sky panel. Seiza matches additive
planes through all panel overlaps together, then tapers remaining smooth
differences using distance to each panel's edges. Star detail stays in the
three-pass integration, and missing pixels remain missing.

Inputs must already share one pixel grid and photometric scale. This matches
relative backgrounds; it cannot remove a gradient shared by every panel.
Rust, C (`mosaicPanels`) and Python (`mosaic_panels`) use the same core code.
See [method, costs and limits](../docs/design/mosaic-backgrounds.md).

## Normalization and frame-edge seams

Global normalization gives each frame one gain and one offset per channel. A
frame whose sky gradient differs from the reference's then leaves a step in
the stack wherever the set of frames covering a pixel changes, so a drifting
or flipped session draws its frames' edges as faint straight streaks.
`NormalizationMode::LocalBackground { tile_size }` (the CLI's
`--normalization local-background`, its default) keeps the global gain, which
follows transparency, and matches each tile's background to the reference's
with a smoothed grid of offsets, filling tiles a frame does not cover from
their neighbours. On the M45 stack it removed every edge seam (from six
streaks to none) and raised SNR from 361 to 374. `Local`'s per-tile gains
remove the seams too, but a tile of cloud, nebula or frame edge drives them
far enough to reject the frame: it admitted 88 of 98 frames there.

Under local background normalization the gain comes from star photometry:
the median ratio of background-subtracted aperture fluxes at the reference's
brightest unsaturated stars, per channel. A gain from the frames' dispersion,
which global normalization still uses, also follows gradients and cloud,
which inflate a hazy frame's dispersion; its gain then came out low, its
noise was scaled down, and inverse-noise weighting gave the cloudiest M45
frames the largest weights. Photometric gains rank them last.

Matching every frame's background to one reference frame copies that frame's
gradient, vignetting and banding into the whole stack. So reintegration
refits each frame's background offsets against an integration of the twenty
best frames (by weight, or else by lowest photometric gain), keeping each
frame's gain, as WBPP integrates its best frames for local normalization. On
M45, referenced to the first frame of the night, this cut the background
ripple left after gradient removal from 0.16 to 0.12 of the nebula's
brightness, against 0.11 for WBPP.

`choose_reference` picks a better reference to begin with (the CLI's
`--reference auto`, its default). It scores each frame on a half-resolution
luminance by the median, over its brightest unsaturated stars, of
sqrt(flux x peak) over the sky noise, which poor seeing, trailing, haze and
twilight all lower, and among frames within 70% of the best score takes the
flattest sky. On M45 it chose a frame from high in the sky, and the online
stack's background ripple fell to 0.11 and its column banding to 0.06 (WBPP:
0.11 and 0.07), bringing out the faint dust around the cluster.
`rank_references` orders every scored frame the same way: the frames near the
best score by flattest sky, then the rest by falling score.

Frame weighting (`StackOptions::weighting`, the CLI's `--weighting
inverse-noise`, its default) then counts hazy frames for less. Together with
quadratic registration and Lanczos interpolation, the CLI's defaults stacked
M45 in 94 s at 2.42px FWHM and SNR 388, against WBPP's 2.60px and 368 in
about 28 minutes (measured before Lanczos clamping gained its noise gate,
below). The library's defaults stay global, equal, similarity and
bilinear, so existing hosts and contexts are unchanged.

## Registration

Registration fits a similarity (shift, rotation, uniform scale) from the 200
brightest stars by default. `RegistrationOptions::model` set to
`RegistrationModel::Affine` or `RegistrationModel::Quadratic` (the CLI's
`--registration-model`, whose default is `quadratic`) then pairs up to 2 000
stars through that similarity and fits a polynomial from reference to source
coordinates, clipping outliers. A wide field's lens distortion turns against
the sky after a meridian flip, which no similarity can follow: on the M45
stack, at 173 mm, the median registration residual after the flip fell from
0.56 to 0.18px, central star eccentricity from 0.39 to 0.31 (WBPP: 0.31), and
SNR rose from 345 to 361, for about 4 s more over 98 frames. A frame with too
few paired stars for its model keeps the similarity, which still drives the
drift, scale and rotation gates. Each frame's warp is kept in its
`RegisteredFrameMapping`, serialized as an optional `warp` field in JSON, and
saved in live-stack contexts in a section of its own; reintegration resamples
through it.

Registration resamples each frame bilinearly by default.
`StackOptions::interpolation` set to `Interpolation::Lanczos3` (the CLI's
`--interpolation lanczos3`) uses Lanczos-3 over the 6x6 nearest samples,
dropping the negative lobes where they would ring at a high-contrast edge, as
PixInsight's clamping does. Only taps spanning more than five times the noise
of the frame's photosites can clamp, read from each colour's own photosites so
a demosaic's smoothing does not lower it. Without that gate, on a sky only a
few times its noise above zero, the clamp fired on the noise itself and raised
the background by an amount set by each frame's sub-pixel phase; frames
rotated 0.3 degrees against the reference turned that into bands about 190 px
apart (Sh2 140, PSF Guard). A star too faint to clear the gate rings as
unclamped Lanczos-3 does, by under 1% of its peak at 2 px FWHM, and that
ringing does not average down across frames as noise does. In a crowded field
the stars lift the noise reading by a fifth to a half, so there the gate sits
nearer six or seven times the true noise. The gate costs
noisier backgrounds than the old clamp gave on such skies, 8-11% on Sh2 140,
because the old clamp was smoothing their noise. On the M45 stack, before the
gate, Lanczos-3 took FWHM from 2.63 to 2.47px and SNR from 345 to 354, against
2.60px and 368 for WBPP, and with reintegration reached 376; stacking took 66 s
instead of 39 s. Reintegration resamples with the same choice.

`StackOptions::cfa_integration` set to `CfaIntegration::BayerDrizzle` (the
CLI's `--bayer-drizzle`) integrates photosites instead: each registered pixel
takes only the nearest source photosite, in the one channel it records, and
the stack fills the colors in from frames that land on different photosites.
Registration and normalization still use the demosaiced frame, and
reintegration replays the same sampling. Nothing is interpolated, but each
channel sees a third to a quarter of the samples, so it pays only when frames
are dithered by several pixels: on the M45 stack above, which drifted 10 to 20
pixels in a night, it reached 2.61px FWHM but SNR 302 against 345 for
demosaicing. A channel no frame reached at a pixel is filled from the same
channel's neighbours in snapshots, with its coverage left at zero.

## Calibration

Integrated flats are applied in the raw light frame's sampling before CFA
debayering. Master darks and flats retain their Bayer pattern and origin
offsets, and a known layout must match the light before calibration. A supplied
bias is removed first, and planar RGB flat channels are normalized independently
so calibration does not introduce a color-scale shift. When bias subtraction
makes a master dark exposure-scalable, every light must provide an exposure
duration rather than silently assuming a 1:1 scale.

`build_master_from_fits` retains its compatibility name but accepts FITS and
XISF inputs. It builds reusable calibration masters without retaining the
input sequence in memory, validates available acquisition metadata, and returns
per-input rejection statistics. Bias and dark masters reread each file for a
leave-one-out sigma-clipped second pass.

Flat inputs are bias/dark calibrated and normalized to a common median before
integration. Temporal sigma clipping uses the median and a MAD-derived scale,
then averages the surviving values. This improves rejection of moving stars
when a star covers a sensor pixel in multiple exposures, but does not guarantee
a star-free sky flat. It does not align
stars or smooth the sensor response: dust, vignetting, and persistent pixel
response remain. The existing low/high thresholds default to 3 sigma. This
follows the robust combination approach in the
[Astropy flat-combination guide](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/05-04-Combining-flats.html).

At least three inputs are needed for rejection; two are averaged without
clipping. More frames and sufficient star motion are important: clipping
cannot reliably distinguish stars from the flat response when contamination
covers half or more of the samples at a pixel. Changing gradients, saturation,
and stationary stars are not repaired by temporal rejection. Small or noisy
sets can retain faint halos even when most samples are clean. A contaminated
majority, including saturated star cores, can be retained more strongly than
with an ordinary average. A zero MAD uses
a floating-point tolerance; if custom thresholds remove every sample, the
flat pixel falls back to its temporal median instead of its contaminated mean
when star masking is disabled.

Sky flats can opt into `MasterBuildOptions::flat_star_masking` (CLI:
`seiza master flat --star-mask`). This detects stars on a native 2x2 analysis
proxy after calibration, expands their footprints to cover halos, and excludes
those original samples before normalization and temporal clipping. The proxy
does not alter the output sensor grid, CFA phases, or RGB samples; a footprint
excludes every channel at that sensor pixel. Extended saturated cores are
masked even when their shape fails normal star admission. Saturation is read
before calibration from an explicit raw-unit override, `SATURATE`/`SATLEVEL`,
or the integer FITS encoding ceiling, never from `DATAMAX` or an observed peak.
Floating-point inputs without an explicit ceiling report unknown saturation.
Isolated one/two-pixel saturated impulses are reported, not expanded into star
masks; existing defect suppression remains a separate option.

Masked integration requires at least two retained unmasked samples at every
output sample after clipping, configurable with `minimum_clean_samples` (CLI:
`--minimum-clean-samples`). Missing coverage fails the whole build with an
actionable error; no masked samples are reused, smoothed, or filled. Two-sample
coverage is reported as limited and averaged without rejection. More sky flats
with greater star motion may be necessary. Native detection can miss faint
halos or mistake sensor features for stars; coverage reports retained samples,
not guaranteed artifact-free data. Masking is disabled by default. Stacking
0.14 adds this option and separate masked counts to the public build/statistics
structs; callers using exhaustive literals must add the fields.

FITS `STARMASK`, `MASKSAMP`, `MASKPIX`, `COVMIN`, `COVMAX`, `COVREQ`, and
`COVLOW` record masking and retained coverage. `SATUNMSK` and `SATUNKN` expose
isolated saturation and unknown ceilings. JSON reports include the thresholds,
coverage, and per-input masked counts. Accepted, rejected, and masked samples
partition the input samples; coverage failure never publishes a master.
The fallback count is reported separately.

Flat integration decodes each input once and temporarily spools its calibrated,
normalized f32 pixels. Scratch space is about four bytes per input sample
(about 14.5 GiB for 64 mono 61 MP frames), and tile payload, read buffer, and
per-pixel statistics workspace are bounded together at 64 MiB. Memory also
includes the decoded image or output image, calibration masters, and per-input
metadata/tallies. The scratch file is removed on success, cancellation, or
failure. `build_master_from_fits_with_scratch` lets hosts choose an existing
cache directory on their image volume; the original entry point uses OS temp.

`MasterBuildOptions::progress` reports where a build is, as a
`MasterBuildProgress { stage, done, total }`: each input frame of the `Read`,
`Reread` and `Integrate` stages and each tile while a flat combines. A host
building masters inside a visible job can show the frame being read and a bar
that keeps moving; see `MasterProgress` for when reports arrive.

`write_master_fits_f32` records the master kind, input count, actual rejection
method (`REJMETH`: `MEDIAN_MAD`, `LEAVE_ONE_OUT`, or `NONE` for two inputs),
thresholds, counts, and bias/dark/normalization state in the FITS header. Those
state fields prevent a later `CalibrationMasters` consumer from calibrating a
prepared dark or flat twice.

The format-level float writer lives in `seiza-fits`. This crate only selects
stack- and master-specific typed header cards before passing its interleaved
linear image to that generic atomic writer.

Registration uses every retained detection for a bounded translation seed,
complemented by bright-star triangles for rotation and scale. The expected
center displacement is the larger of
`StackOptions::registration.maximum_drift_pixels` and
`maximum_drift_fraction` times the reference frame's larger dimension. The
defaults are 256 pixels and 15%. Differently sized or cropped light frames are
resampled onto the reference grid; samples outside their valid crop remain
masked and are accounted for by the overlap admission gate.

Meridian-flipped frames are accepted by default. The rotation admission limit
is measured from the nearer of the reference orientation and its 180-degree
counterpart, while diagnostics retain the full fitted rotation (for example,
179.3 degrees). The same similarity transform is then used to rotate the
pixels back onto the reference grid before normalization and integration.


## Display narrowband balance and RGB stars

`balance_narrowband(&mut image, &options)` takes three aligned interleaved
physical planes in H-alpha/OIII/SII order, already stretched into [0, 1]. For
HOO, duplicate OIII into plane three and set `has_sii=false`. The returned
report lists each measured channel's background, positive deviation, midpoint,
and application or skip reason. H-alpha and protected shadows keep their
samples; flat or noise-dominated channels remain unchanged.

`screen_rgb_stars(&mut nebula, &original_rgb, &starless_rgb)` derives a
screen-compatible stars layer and screens it onto a starless display nebula.
All images must share one RGB grid and [0, 1] display scaling. Apply the same
linked display transfer to original and starless RGB before calling it. Missing
coverage masks complete pixels. Neither function aligns images or runs external
tools. Both produce display-referred data, unsuitable for photometry or linear
continuum subtraction. C ABI, Python and CLI counterparts provide the same
operations; see the [color composition design](../docs/design/color-composition.md).

## Continuum fitting and signed residuals

Use continuum fitting **before stretching or removing stars**, on mono linear
inputs already registered to one grid. The host supplies one to three concrete
broadband references; Seiza does not infer filter roles, integration-time
scaling or line throughput. R is a starting candidate for H-alpha/SII; G/B are
candidates for OIII. L is not an automatic reference.

`fit_continuum(stars, sources, options)` consumes common isolated-star aperture
photometry. `measure_and_fit_continuum(narrowband, references, sources, positions,
options, photometry)` measures local-background-subtracted fluxes; empty
positions request Seiza detection on the narrowband plane. All `ContinuumSource`
records declare a distinct name, exact prepared revision and linear transfer.
Automatic fitting also needs a physical saturation ceiling or producer clipping
mask, described below. Change the revision whenever
pixels, scale, registration, backgrounds or masks change. Saturation must come
from the image's physical scale, never its observed maximum.

The core fits `F_N = sum(a_j * F_Bj)` with nonnegative weights and no pedestal,
using column scaling, small deterministic subset starts and iterative MAD
rejection. Every fourth usable star is held out. Defaults require 12 stars,
SNR 20, peaks below 80% saturation, width ratios at most 1.3, centroid offsets
at most one pixel, condition number at most 1000, and fractional RMS at most
0.15 on both training and validation sets. At least 70% of each set must
remain after rejection. Image measurement rejects crowded/edge apertures and
fits a robust local sky plane with default radii 8/12/20 pixels. Supplied
photometry must already be from isolated stars with consistent PSF widths.
A rank-deficient multi-reference model may fall back to the best validated
single reference; otherwise it returns a typed skip. No fixed G/B mixture is
substituted. These are conservative starting defaults, not a calibration claim
for every instrument or target.

`ContinuumFitOutcome` distinguishes a fitted plan from skipped work with a
reason, detail and diagnostics. Diagnostics include used/held-out star counts,
rejection counts, residuals per star, conditioning, RMS, four ordered quadrant
medians (null when fewer than three stars), and faint-versus-bright residuals.
A fit never runs on an image-wide nebular regression. Inspect residuals before
using automatic subtraction as a default in a host.

`subtract_continuum(narrowband, references, sources, plan)` validates exact
source revisions, order, grid, transfer, coefficient shape and provenance.
It returns `N - sum(a_j * B_j)` without changing inputs, clipping negative
samples, shifting black levels or stretching. Missing contributing coverage
remains NaN; a zero-weight reference does not mask pixels. Manual plans use
schema version 1, explicit sources and coefficients, `ContinuumOrigin::Manual`
and no diagnostics. Fitted plans retain their diagnostics.

The coefficients include the actual input scales. A rescale requires a new
fit or an explicit host conversion on new revisions; old plans are refused.
Background offsets remain separate. Applying the same plan to a starless or
otherwise processed revision is refused until the host establishes and records
that revision's equivalent flux scale. No implicit PSF matching is performed.

Broadband references can contain line emission. If `B = C + beta*E` and
`N = a*C + E`, subtraction yields `(1-a*beta)*E`. The result is an **emission
residual**, not calibrated pure line flux. LRGB+SHO blend weights, negative-noise
display suppression and luminance integration are separate host policies.
C exports `seiza_continuum_fit_json`, `seiza_continuum_measure_fit_json` and
`seiza_continuum_subtract_json` carry the same core types. C images are planar
NB followed by references; Python uses a list of mono arrays and returns the
same JSON keys. See [CLI usage](../seiza-cli/README.md#subtract-continuum-from-linear-narrowband-images)
and [Python usage](../seiza-py/README.md#subtract-stellar-continuum-before-color-composition).


### Clipping evidence, sampling and combined images

`measure_and_fit_continuum_masked` accepts one binary clipping plane per input
(NB then references). Declare exact `clipping_revision` values in the sources;
a scalar physical `saturation` ceiling may be omitted. Automatic fitting needs
one of these forms of evidence for every source. Manual signed subtraction
can omit both. Masks reject stars before sampling and never change image pixels.
The usable-star budget defaults to 400. Measurement considers up to 4096 candidates independently of that budget, rejects ineligible stars first, samples
across the remaining flux range, and validates against the complete eligible
pool. A disagreement or failed broader validation produces a skipped outcome.
Diagnostics retain eligible/sample counts and sampling stability.

`blend_emission_rgb` projects exact signed residuals into RGB reference units.
Each `EmissionBlendLine` supplies a plan, destination RGB indices in reference
order, amount [0,5] and explicit background offset. For stellar ratios
r_j=F_Bj/F_B0, k_j=sum(a_i*r_i)/r_j; the contribution is amount*(E-offset)/k_j.
Fitted plans record the ratios. Multi-reference manual plans need measured or
explicit ratios; single-reference plans use 1. Independent input gains commute
with projection. `PositiveContribution` clamps the display contribution only;
the original signed residual stays unchanged. This does not correct broadband
line transmission or produce calibrated pure line flux.

Calibrate broadband stars before emission addition. `add_rgb_contribution`
then adds normalized RGB emission with the measured channel gains; calibration
offsets affect the base only. Add L through the existing luminance composition.
`blend_display_structure` mixes prepared starless [0,1] RGB display images,
with endpoints preserving the selected branch's coverage and intermediate
amounts requiring common coverage. Restore a single RGB star layer afterward.
These operations are also exposed through the C ABI, Python and CLI.

### Validate stellar profiles and match PSFs explicitly

Image-based continuum measurement now defaults to an independent held-out pixel
check, controlled by `ContinuumPhotometryOptions.max_pixel_residual` (0.15).
It compares the local-sky-subtracted NB patch with the fitted reference patches,
sums absolute residual above a three-sigma local noise allowance, and divides
by stellar flux. The 90th percentile must pass. At least three profiles and
70% of eligible held-out profiles must be remeasurable. Reports retain signed
core/halo fractions separately. Premeasured `fit_continuum` has no pixels and
therefore reports no pixel validation; manual plans also make no such claim.
Explicit `None` disables the pixel gate for research callers, not PSF Guard's
automatic mode.

Set `photometry.psf_matching = Some(ContinuumPsfOptions { target_fwhm: None })`
to broaden copies before measurement and subtraction. Common unsaturated stars
supply Gaussian-equivalent widths. The default target is 1.05 times the largest
median width; an explicit target must be at least every measured width and at
most 32 pixels. Kernels conserve DC/flux. Missing coverage and unsupported edges
remain NaN; producer clipping evidence expands over the full kernel support.
Original saturation evidence is required before preparation creates derived
mask revisions.

The plan records exact raw sources, derived prepared sources, measured widths,
target and kernel sigma/radius. `subtract_continuum` replays that preparation
from the original inputs. Saved plans must pass source, kernel and target
consistency validation. No deconvolution occurs. A Gaussian-equivalent width
match does not ensure matched real wings: the held-out pixel gate still applies
and can skip a broadened result.

`NarrowbandBalanceOptions.star_mask_radius` optionally excludes detected-star
circles from balance/noise statistics without changing output coverage. It is
off by default, accepts 1–64 pixels and reports counts. `reduce_green_excess`
reduces display green above both red and blue by an explicit amount [0,1],
preserving neutral pixels. Apply it after palette mapping and before adding
independently prepared RGB stars. It is not a photometric operation or a claim
of SCNR parity.

### Export calibrated line flux

`solve_emission_flux(residuals, plans, calibration)` solves one to three coupled
line responses. `EmissionFluxCalibration` supplies exact plan sources, line
order, common unit, calibration revision, NB responses and each plan's reference
responses. Responses are in the actual input sample units per common flux unit;
the producer accounts for throughput, exposure, gain, extinction and doublets.
No filter-name or bandwidth approximation is inferred.

The effective response is `M[i,k] = N[i,k] − sum_j a[i,j] B[i,j,k]`.
The operation solves `M F = E − residualZeroLevels`, retaining signed flux and common coverage.
It refuses stale source identities, inconsistent PSF targets, nonpositive or
nearly cancelled diagonal response, excessive conditioning and numeric overflow.
Inversion uses a normalized matrix and restores row units analytically.

Optional residual covariance propagates as `M⁻¹ C M⁻ᵀ`. Missing covariance means
unknown uncertainty. The report embeds calibration and plan provenance and
explicitly excludes response/fit-coefficient uncertainty, unmodelled lines and
spatial PSF errors. The result is calibrated only to the supplied response model.
It does not replace an artistic display blend. C exports
`seiza_emission_flux_json`; Python returns `(images, report)`; the CLI exposes
`seiza color blend flux` with planar residuals, a settings JSON, output paths
and a report.

Matching also requires `target_fwhm <= aperture_radius` and
`annulus_inner >= 1.5 * target_fwhm`. The default 8/12/20 photometry therefore
permits targets up to **8 pixels**. Larger targets require larger photometry
radii in the options JSON (with a hard target limit of 32 pixels and outer
sky-annulus radius at most 128 pixels). This keeps the target profile inside
the measurement aperture and away from its sky annulus.

Calibrated export also requires explicit `residualZeroLevels` (Python alias
`residual_zero_levels`), one additive zero level per residual in its exact
sample units. Zero is an explicit calibration declaration. The solver uses
`M F = E − residualZeroLevels`; it never guesses these values from a sky median
or the display baseline. Background preparation may retain an additive reference
level, so response calibration alone cannot establish a physical zero point.
Zero-level uncertainty is outside the propagated residual covariance unless
the producer includes it there.

Image-based validation also audits isolated candidates excluded for clipping or
saturation. They never train the fit. Their observed residual profiles must all
be measurable and below the same absolute limit; `clippedStars`,
`clippedProfiles` and `clippedMaxAbsoluteFraction` record this separate check.
It prevents a clean unsaturated sample from hiding clipped-core artifacts and
does not infer lost flux. The held-out report keeps both its 90th-percentile
and maximum profile error.

### Reduce restored RGB stars

`screen_rgb_stars_reduced(nebula, original, starless, reduction)` performs the
same display screen recombination with optional star minimization. An amount
in [0,1] blends the isolated star layer's maximum-channel intensity toward a
one-pixel cross erosion. A shared per-pixel RGB gain preserves layer color
ratios. The starless nebula is untouched. Invalid neighbours/outer edges retain
the unreduced layer; central coverage is unchanged. High values also remove
faint stars. Zero uses the original screen arithmetic bit for bit.

The operation is display-only; it does not sharpen a PSF or alter scientific
linear artifacts. C exports `seiza_color_screen_rgb_stars_reduced`; Python
returns a copy with the same name; CLI `seiza color screen-stars --reduction
0.4` applies it after a consistently stretched original/starless RGB pair.

Calibrated export requires `continuumCoefficients` (Python alias
`continuum_coefficients`): exact per-plan coefficient snapshots in reference
order. Residual zero-level calibration depends on these weights. A changed
fit/manual weight is rejected even when the raw source revisions are unchanged;
update the explicit zero levels and snapshots after recalibration. Reports and
cache identities retain the snapshots.

### Stretch linear RGB stars independently

Run star separation on linear RGB after any linear sharpening and color
calibration. Pass that original/starless pair to `stretch_rgb_stars(original,
starless, amount)` to extract the linear residual and apply the SetiAstro-style
rational tone `3^amount*x / (1+(3^amount-1)*x)`. Amount is 0–8; 5 is the script's
starting tone, lower values make stars fainter, and 0 leaves the normalized
linear residual without nonlinear lift. One shared source range normalizes
physical units; do not normalize each channel or the stars-only layer to its
own peak. The background offset cancels in the residual.

Use `screen_rgb_star_layer(&mut nebula, &stars, reduction)` on the resulting
display stars and display nebula. Reduction is 0–1 and applies the existing
linked cross erosion only to the stars. Negative extraction noise is clipped
only in the display layer; missing coverage masks whole RGB pixels. Linear
inputs stay unchanged. This independently implements the published
[SetiAstro tone](https://www.setiastro.com/pjsr-scripts), without replicating
PixInsight's optional ColorSaturation or SCNR processes.
