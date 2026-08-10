# seiza-stacking

`seiza-stacking` provides linear, incremental image stacking for
astrophotography applications. It keeps plate solving and catalog access out of
the stacking path while reusing Seiza's star detector for local registration.

The first release supports:

- mono, planar RGB, and Bayer FITS or XISF inputs in linear sensor units;
- optional master bias, dark, and flat calibration;
- bounded-memory, two-pass construction of bias, dark, and flat masters;
- bounded-drift star registration with translation/rotation/scale refinement;
- robust global or tiled local normalization;
- online residual (delta-sigma) rejection with coverage and rejection maps;
- non-mutating frame admission gates for additive live stacks;
- versioned, checksummed, atomic live-stack checkpoints that can be reopened;
- floating-point FITS output on the reference frame's pixel grid.

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
stacker.push_fits_pipelined(&paths, &PipelineOptions::default(), |path, outcome| {
    record(path, outcome);
    if cancelled() { Continue::No } else { Continue::Yes }
})?;
```

The callback keeps the caller in charge of cancellation, checkpointing, and
per-frame decisions, which is why this is not a batch call that swallows the
loop. The accumulator is still fed strictly in submission order, so a
pipelined run is bit-identical to a sequential one; a test asserts exactly that
against the same frames.

`PipelineOptions::max_in_flight_bytes` bounds the memory rather than the frame
count, because a prepared frame is the reference image's size and that differs
by an order of magnitude between a guide camera and a full-frame sensor. The
worker count falls out of that budget and the machine's parallelism, or can be
set outright. A budget too small for even one frame ahead degrades to
sequential rather than failing.

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

Integrated flats are applied in the raw light frame's sampling before CFA
debayering. Master darks and flats retain their Bayer pattern and origin
offsets, and a known layout must match the light before calibration. A supplied
bias is removed first, and planar RGB flat channels are normalized independently
so calibration does not introduce a color-scale shift. When bias subtraction
makes a master dark exposure-scalable, every light must provide an exposure
duration rather than silently assuming a 1:1 scale.

`build_master_from_fits` retains its compatibility name but accepts FITS and
XISF inputs. It builds reusable calibration masters without retaining the
input sequence in memory and rereads each file for a leave-one-out
sigma-clipped second pass, validates available acquisition metadata, calibrates
and normalizes each flat before integration, and returns per-input rejection
statistics. `write_master_fits_f32` records the master kind, input count,
rejection settings, and bias/dark/normalization state in the FITS header. Those
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
