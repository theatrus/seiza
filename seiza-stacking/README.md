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

Live delta-sigma rejection cannot remove trails admitted during its warm-up.
`LiveStacker::reintegrate` fixes that for a live stack. The stacker keeps a
ledger of every admitted frame: its source file, the calibration masters that
applied, and the registration and normalization mapping the live pass chose.
Reintegration reads each frame three more times, prepares it exactly as the live
pass did, and integrates it with leave-one-out rejection, leaving the live
stack unchanged. Star detection and registration do not run again. A source
file that changed since it was stacked is refused, and
`LiveStacker::reintegration_unavailable` says why a stack cannot be replayed:
frames pushed as pixels, frames calibrated with in-memory masters that were
since replaced, or a context saved before the ledger existed. Saved contexts
carry the ledger from format version 3; versions 1 and 2 still open.

Hosts that keep their own records can call `integrate_registered_frames`
directly to reread admitted frames with their saved calibration and
`RegisteredFrameMapping`. The loader
receives `BatchStackPass` and the frame index, so a host can report each pass
and cancel the work. `BatchStackResult` includes the final mean, variance,
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
Memory stays proportional to the output image (about 64 bytes per sample plus
one loaded input), and each frame is loaded three times. Changed shapes or sample
digests abort the result.

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
again. To replay frames yourself with `integrate_registered_frames`, pass the
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
`PoolPipelineMemory::for_reference` to share the same worker/integration
estimate when planning the host's budget, and
`CalibrationMasters::image_buffer_bytes` to count resident master samples;
session masters and their active deep clones both consume storage.

When even one queued worker would exceed the host's memory policy, use
`push_fits_sequential_with_pool(paths, normalized_full_scale, pool, callback)`.
It reads, prepares and integrates one frame at a time without reader threads
or a queue budget. It reports `SequentialRequested`, zero coordinator wait,
and `PoolPipelineMemory::sequential_bytes`: `64 * pixels + 8 * samples` of
reference-sized scratch. The host must reserve its persistent stack and master
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
master, dark temperature, and flat optics. Bias-isolated dark current may be
scaled by exposure; a dark that still contains its pedestal must have the same
known positive exposure as the light. With no masters, preprocessed lights
remain valid inputs.

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

Bayer frames are demosaiced with Malvar, He and Cutler's gradient-corrected
linear interpolation, which keeps each color's centroid of an undersampled
star where it belongs: on a 98-frame M45 stack at 2.6px FWHM, bilinear
interpolation left red and blue 0.42px apart and stars 3.1px wide, against
0.15px and 2.6px. The two-pixel border keeps bilinear estimates.

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

Frame weighting (`StackOptions::weighting`, the CLI's `--weighting
inverse-noise`, its default) then counts hazy frames for less. Together with
quadratic registration and Lanczos interpolation, the CLI's defaults stacked
M45 in 94 s at 2.42px FWHM and SNR 388, against WBPP's 2.60px and 368 in
about 28 minutes. The library's defaults stay global, equal, similarity and
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
PixInsight's clamping does. On the M45 stack it took FWHM from 2.63 to 2.47px
and SNR from 345 to 354, against 2.60px and 368 for WBPP, and with
reintegration reached 376; stacking took 66 s instead of 39 s. Reintegration
resamples with the same choice.

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
