# seiza-cabi

A C ABI over Seiza for native application front-ends. It is the single, shared
successor to the near-identical `seiza-cabi` crates that previously lived in the
[`seiza-win`](https://github.com/theatrus/seiza-win) (.NET) and
[`seiza-mac`](https://github.com/theatrus/seiza-mac) (Swift) repositories, and it
exposes the **superset** of what both apps need.

## What it exposes

- **FITS / XISF / raster rendering** — `seiza_rendered_image_open`,
  `..._open_with_rgb_stretch`, `..._open_with_stretch_config` (see the
  **Parameterized stretch** entry below), `..._width`, `..._height`,
  `..._metadata_json`, `..._free`.
- **Both pixel byte orders** — `seiza_rendered_image_rgba` (macOS / CoreGraphics)
  and `seiza_rendered_image_bgra` (Direct2D / WinUI), each with a `_length`
  companion. RGBA is canonical; the BGRA view is computed on first request and
  cached, so a consumer only ever pays for the order it uses.
- **Native 16-bit export pixels** — the parallel
  `seiza_rendered_image16_open*` API returns a separate
  `SeizaRenderedImage16` with borrowed native-endian RGBA `uint16_t` samples.
  FITS stretch stacks quantize directly from their final `f32` result to `u16`,
  and 16-bit PNG/TIFF raster inputs retain their component precision. The
  `_rgba_length` result counts `uint16_t` elements, not bytes. A separate owner
  keeps routine RGBA8 previews from allocating both formats.
- **Parameterized stretch** — `seiza_rendered_image_open_with_stretch_config`
  takes a serialized `seiza-stretch` `StretchConfig` (JSON) and renders a FITS
  or XISF image
  through the full GHS/MTF/percentile pipeline. It also accepts a non-empty
  config array for an ordered `f32` stack, or an object with `stretch`, optional
  `sample_domain`, optional `background` correction, optional `deconvolution`,
  and optional `interactive_preview` mode. Background fitting and
  subtraction/division run first on linear input samples, followed by
  deconvolution, sample-domain mapping, and then the display stretch.
  Interactive previews bound those samples before expensive processing while
  committed renders remain full resolution. The two most recent prepared
  preview buffers are cached by file identity, maximum dimension, and background
  configuration; stretch and deconvolution edits reuse the same corrected linear
  pixels. New processing capabilities remain in their core crates; this shim
  only marshals JSON in and pixels out.
- **Background extraction** — `seiza_background_fit` creates a compact opaque
  model from interleaved linear mono or RGB `float` samples. Callers can inspect
  its borrowed diagnostics JSON, render it into a caller-owned buffer, or apply
  subtractive/divisive correction in place before freeing the model. A second
  correction entry point accepts a strength from zero to one. Optional settings
  use serialized `seiza-background` `BackgroundConfig` JSON, including
  polynomial, radial-basis, and automatic model selection, so the ABI stays
  stable as model options grow. The JSON config can carry normalized ellipses
  or polygons projected from solved catalog bounds, including stored OpenNGC
  contours.
- **Color crop diagnostics** — `seiza_color_crop_report_json` reports the
  region a crop would keep across aligned channels and what each channel
  covers of the shared grid, including any channel sitting far enough from the
  others to look like a pointing error. It reads the caller's buffers in place
  and returns owned JSON.
- **Light deconvolution** — `seiza_deconvolve_in_place` applies the same
  conservative damped Richardson-Lucy operation as the Rust and Python APIs to
  caller-owned linear mono or interleaved RGB `float` samples. The synchronous
  call retains no pointers and reports validation failures through `error_out`.
- **Live stacking** — `seiza_live_stacker_create` starts from caller-provided
  calibrated linear mono/RGB samples, while `seiza_live_stacker_open_fits`
  retains its ABI name but decodes raw linear FITS or XISF and optionally
  applies integrated bias, dark, and flat masters. Array and path-based pushes
  return owned admission JSON. Borrowed
  mean/coverage/rejection views support copy-free live display, snapshots add
  variance, and `seiza_live_stacker_finish` moves the final accumulator into an
  immutable result without cloning its full-frame buffers.
  `seiza_live_stacker_save_context` atomically checkpoints a still-live handle;
  `seiza_live_stacker_open_context` restores it later with its original
  reference, calibration, online moments, counters, and source ledger intact.
  `seiza_live_stacker_state_json` exposes an authoritative resume identity,
  `..._set_calibration_fits` swaps session masters atomically between batches,
  and `..._render_preview` renders from a bounded sample of the live mean
  without making a full-frame snapshot. `..._export_snapshot` copies only one
  full-frame mean for non-destructive FITS/XISF output on a worker thread.
- **Calibration orchestration** — `seiza_probe_frame_json` reads FITS/XISF
  metadata without decoding pixels, `seiza_calibration_plan_json` applies the
  core sensor/optics/exposure/temperature/proximity/coherence rules, and
  `seiza_calibration_build_master_json` atomically publishes a two-pass
  sigma-clipped bias, dark, or flat master from raw paths while reporting any
  metadata outliers it set aside. Long builds accept the thread-safe
  `SeizaCancelSignal` owner.
- **Stack-depth analysis** — `seiza_live_stacker_measure_depth` measures the
  borrowed live accumulator without a snapshot, and `seiza_checkpoint_depths`
  returns the doubling ladder at which a host should sample a finite batch.
- **Plate solving** — `seiza_solve_image_json`.
- **Catalog setup** — `seiza_catalog_status_json` and `seiza_catalog_setup`
  (with a progress callback). The install path delegates to
  `seiza-download`'s `materialize_with`; the shim carries no download logic.
- **Memory** — `seiza_core_version`, `seiza_string_free`.

Rendered-image metadata includes input and display histograms plus the requested
and resolved sample domains used for that render.

### Render sample domains

`sample_domain` describes the numeric scale presented to the stretch pipeline;
it is render policy, not frame-to-frame stack normalization. Samples already in
the zero-to-one working domain use an identity mapping. The examples below show
the optional field only; it sits beside the required `stretch` field in a
processed render request.

```json
{"sample_domain":{"type":"unit-linear"}}
```

Physical camera, calibrated, or stacked samples can resolve a robust display
range before stretching:

```json
{
  "sample_domain": {
    "type": "physical-linear",
    "normalization": {
      "type": "robust-percentile",
      "black_percentile": 0.001,
      "white_percentile": 0.999,
      "max_analysis_samples": 200000
    }
  }
}
```

Within `physical-linear`, robust-percentile with those three values is used
when `normalization` is omitted; omitted robust fields take those values.
Resolution samples complete pixels and pools their finite values into one
linked range for mono or RGB, so it does not silently perform a per-channel
color balance. Non-finite samples are excluded from analysis and remain
non-finite during affine mapping, preserving live-stack coverage masks. A
caller can lock a known physical range instead:

```json
{
  "sample_domain": {
    "type": "physical-linear",
    "normalization": {
      "type": "explicit-range",
      "black": 512.0,
      "white": 16384.0
    }
  }
}
```

The mapping always runs after optional physical-domain background correction
and deconvolution and immediately before the first stretch stage. Both the
requested policy and its resolved mapping (including the black/white range for
physical samples) are retained in render metadata for provenance or a later
locked-range request.

Omission is backward-compatible. Ordinary file rendering and a live stack in
`PreparedOnly` input mode retain the historical `unit-linear` behavior. A live
stack in `CalibrateAndPrepare` mode defaults to `physical-linear` with robust
percentiles because its native mean remains in physical sample units. An
explicit `sample_domain` always overrides that live default.

Sample-domain mapping modifies only the temporary render buffer. Live means,
variances, SNR measurements, saved contexts, export snapshots, and written
FITS/XISF samples remain physical linear `f32`; neither the requested nor the
resolved display range becomes part of scientific stack state.

Use `seiza_rendered_image16_open_with_stretch_config` for a processed FITS
export, or `seiza_rendered_image16_open` for the default FITS/raster path. The
16-bit handle has its own width, height, RGBA, metadata, and free functions; do
not pass it to the RGBA8 accessors. Its RGBA pointer is aligned for `uint16_t`
and uses host byte order. Image encoders or platform image APIs must be told
that byte order when consuming the borrowed samples.

Background input and output buffers are row-major, pixel-interleaved `float`
samples with one or three channels. The model copies only compact samples and
coefficients, so the input buffer may be released after `seiza_background_fit`
returns. Pass null for both mask/configuration pointers to use automatic
defaults. Correction mode constants and the precise pointer/length contracts
are declared in the generated header.

A crop report takes parallel arrays of channel names and sample pointers on
one shared grid, and the mode `none`, `bounds`, or `inscribed`. It borrows
those samples for the call rather than copying them. The JSON carries the
mode, the grid size, the kept `region`, the retained fraction, the offset at
which a channel is flagged, and one entry per channel with its own covered
box, covered pixel count, center offset from the other channels, and
`off_center` flag:

```json
{"mode":"inscribed","grid":{"width":128,"height":128},
 "region":{"x":0,"y":80,"width":128,"height":48},
 "retained_fraction":0.375,"off_center_limit_pixels":32.0,
 "channels":[{"name":"SII","region":{"x":0,"y":80,"width":128,"height":48},
              "covered_pixels":6144,"center_offset_x":0.0,
              "center_offset_y":40.0,"center_offset_pixels":40.0,
              "off_center":true}]}
```

Deconvolution uses the same buffer layout and modifies the caller-owned input
in place. Supply a measured stellar PSF FWHM in pixels; the conservative values
are four iterations, a `0.35` blend, a `0.001` noise fraction, and a maximum
correction of `2.0`. The output remains linear and may contain samples outside
`[0, 1]`.

Stacking uses the same row-major, pixel-interleaved mono/RGB layout. Array
frames are copied during each synchronous call and may be released when it
returns; they must already be calibrated, debayered, and linear. FITS/XISF
pushes retain and apply the calibration masters loaded by the path
constructor; a non-zero dark exposure override requires a dark master path. A
handle created from an array accepts only linear-array pushes, and a handle
created from a path accepts both the configured FITS path and prepared linear
pushes. A saved context retains that input mode. A rejected frame is
represented by `accepted: false` disposition JSON and is not an ABI error. The
zero-copy live pointers are invalidated by the next push, finish, or free;
immutable snapshot pointers remain valid until snapshot free. Snapshot FITS
output refuses to overwrite any tracked light or calibration input.

When any master is active, the native stacker validates every reference and
pushed FITS/XISF frame before touching its pixels. Declared masters, known raw
`bias`/`dark`/`dark-flat`/`flat` roles, and frames already marked `BIASSUB`,
`DARKSUB`, or `FLATNORM` are rejected; unknown roles remain compatible with
legacy files. Bias/dark/flat masters must sensor-match every light, flats must
also match its optics, and darks must match temperature. A dark whose pedestal
was isolated with a usable bias may scale by exposure. An unscaled dark must
carry the same known positive exposure as each light. With no masters these
guards are a no-op, so deliberately preprocessed lights remain stackable.

The authoritative state response is owned JSON and is sufficient for a host's
resume store:

```json
{
  "schemaVersion": 1,
  "coreVersion": "0.17.0",
  "configurationFingerprint": "64-lowercase-sha256-hex-characters",
  "width": 6248,
  "height": 4176,
  "channels": 1,
  "acceptedFrames": 17,
  "rejectedFrames": 2,
  "inputMode": "calibrate-and-prepare",
  "inputPaths": ["reference.fits", "master-bias.fits", "light-002.fits"],
  "referenceFrame": {
    "role": "light",
    "isMaster": false,
    "signature": {
      "camera": "ASI2600MM", "telescope": null,
      "width": 6248, "height": 4176, "channels": 1,
      "binningX": 1, "binningY": 1, "gain": 100, "offset": 50,
      "readoutMode": null, "bayerPattern": null, "filter": "Ha",
      "focalLengthMm": null, "rotationDeg": null,
      "exposureSeconds": 300.0, "cameraTempC": -10.0,
      "capturedAtUnix": 1785593045
    },
    "calibrationState": {
      "biasSubtracted": false,
      "darkSubtracted": false,
      "flatNormalized": false
    }
  }
}
```

`inputPaths` is the native ordered ledger, not a host reconstruction. The
fingerprint covers validated stack options, current calibration pixel content
and metadata, and input mode. It excludes the accumulator, counters, and paths,
so ordinary pushes do not change it; a calibration swap does. Context save/open
persists those fingerprint inputs and recomputes the same value on restore.
`referenceFrame` is native metadata for the immutable reference source, so a
resume host need not reopen a path that may have moved or disappeared.

Swap masters only between push batches:

```c
bool changed = seiza_live_stacker_set_calibration_fits(
    stacker, "session-2-bias.fits", "session-2-dark.fits",
    "session-2-flat.fits", 0.0, &error);
```

All files are loaded and the set is validated against the immutable reference
before calibration or the ledger changes. Null/empty paths clear that master;
all three absent clears the set. Already integrated frames are untouched. The
operation is unavailable on a stack created from prepared arrays.

For non-destructive full-resolution output, capture one compact immutable
owner while holding the live stacker's lock, then release that lock and write
on a worker:

```c
SeizaStackExportSnapshot *output =
    seiza_live_stacker_export_snapshot(stacker, &error);
/* ingestion may continue here */
bool written = seiza_stack_export_snapshot_write_fits(
    output, "live-stack.fits", &error);
seiza_stack_export_snapshot_free(output);
```

The capture copies only the finalized `f32` mean plus small headers, scalar
frame counts, and the source ledger. It does not clone variance, coverage, or
per-sample rejection arrays. Zero-coverage samples become `NaN`. The owner is
independent of the live stacker and may be transferred to another thread; do
not free it until its write returns.

For frequent display updates, render the live mean through the same stretch,
`sample_domain`, background, and deconvolution JSON as file rendering:

```c
SeizaRenderedImage *preview = seiza_live_stacker_render_preview(
    stacker, stretch_json, 1600, &error);
```

The maximum dimension is required and bounds copied linear samples before any
expensive processing. The implementation samples the borrowed accumulator
directly rather than making a full snapshot. Zero-coverage pixels stay masked
through statistics, background fitting, deconvolution, sample-domain mapping,
and stretching; they are transparent in RGBA. The returned image is independent
of the stack and uses the ordinary `SeizaRenderedImage` accessors/free function.

Stack options are serialized `seiza-stacking` `StackOptions`. Every nested
object accepts omitted fields from its defaults. For example, this disables
normalization/rejection while increasing the registration drift floor:

```json
{
  "registration": { "maximum_drift_pixels": 512.0 },
  "normalization": { "mode": "none" },
  "rejection": { "mode": "none" }
}
```

Local normalization is `{"mode":"local","options":{"tile_size":256}}`;
delta-sigma rejection is
`{"mode":"delta-sigma","options":{"low_sigma":3.0,"high_sigma":3.0}}`.
Unknown fields and invalid bounds are rejected rather than silently ignored.

For resumable app sessions, save and reopen the opaque processing state rather
than treating an integrated FITS result as a new reference:

```c
bool saved = seiza_live_stacker_save_context(stacker,
                                             "m31.seiza-stack", &error);
seiza_live_stacker_free(stacker);

stacker = seiza_live_stacker_open_context("m31.seiza-stack", &error);
char *decision = seiza_live_stacker_push_fits_json(stacker,
                                                   "light-042.fits", &error);
```

Saving does not consume or otherwise mutate the handle. Context publication is
atomic, and reopening validates the format version, dimensions, configuration,
payload checksum, and accumulator invariants before returning a handle.
Current writers use context version 2, which retains normalized reference and
master signatures plus the dark-scaling safety fact. Version-1 contexts still
open. If a migrated v1 context contains masters, pushes fail closed until the
caller reloads them with `seiza_live_stacker_set_calibration_fits`, because the
older checkpoint cannot prove their compatibility; no-master v1 contexts
continue normally.

Offering a whole session at once prepares several frames in parallel while
integrating in the order given, which is identical to offering them one at a
time but overlaps the reads with the registration:

```c
char *outcome = seiza_live_stacker_push_fits_pipelined_json(
    stacker, "[\"light-001.fits\",\"light-002.fits\"]",
    0 /* derive workers */, 0 /* default budget */,
    65535.0f /* scale declared-normalized XISF to 16-bit; 0 to leave as stored */,
    &error);
```

Every frame appears in the returned `frames` array in order, a path that could
not be read among them with `accepted` false and a `reason`; check `failed`
rather than reading an absent error as success. Raise the worker count when the
frames arrive over a network, since the library cannot tell a network mount
from a local disk.

Frame probing is genuinely metadata-only for both containers: it calls
`seiza_fits::read_header` or `seiza_xisf::read_header`, never `FitsFrame::open`.
The owned result has this stable shape (unknown signature values are `null`):

```json
{
  "schemaVersion": 1,
  "path": "dark-001.fits",
  "format": "FITS",
  "role": "dark",
  "rawImageType": "Dark Frame",
  "isMaster": false,
  "signature": {
    "camera": "ASI2600MM", "telescope": null,
    "width": 6248, "height": 4176, "channels": 1,
    "binningX": 1, "binningY": 1, "gain": 100, "offset": 50,
    "readoutMode": null, "bayerPattern": null, "filter": null,
    "focalLengthMm": null, "rotationDeg": null,
    "exposureSeconds": 60.0, "cameraTempC": -10.0,
    "capturedAtUnix": 1785593045
  },
  "calibrationState": {
    "biasSubtracted": false,
    "darkSubtracted": false,
    "flatNormalized": false
  }
}
```

Roles are `bias`, `dark`, `dark-flat`, `flat`, `light`, or `unknown`.
`SEIZAMST` takes precedence over `IMAGETYP`/`OBSTYPE`/`FRAME`; common master,
object, and dark-flat spellings are normalized. The signature maps only fields
used by `seiza-calibration` matching. FITS dimensions and XISF-synthesized
`NAXIS*` cards share the same result.

Pass probe records directly as the reference/candidates of a coherent plan:

```json
{
  "kind": "dark",
  "reference": {"path":"light.fits", "role":"light", "signature":{}},
  "references": [
    {"path":"light.fits", "role":"light", "signature":{}},
    {"path":"light-002.fits", "role":"light", "signature":{}}
  ],
  "dependencies": {"biasAvailable": true},
  "candidates": [
    {"path":"dark-001.fits", "role":"dark", "signature":{}}
  ],
  "minimum": 8,
  "tolerances": {
    "exposureSeconds": 0.05,
    "exposureFraction": 0.001,
    "darkTemperatureC": 3.0,
    "masterTemperatureC": 1.0,
    "rotationDeg": 1.0,
    "focalLengthMm": 1.0,
    "flatSessionSeconds": 86400
  }
}
```

Plan kinds are `bias`, `dark`, `dark-flat`, and `flat`. A `dark-flat` plan
uses the same exposure and temperature matching as a dark, but requires
`dark-flat` candidates; use the selected raw flat as its reference. Build the
chosen dark-flat inputs with master kind `dark`, then pass that master as
`dark` when building the flat. This keeps the master builder ABI compatible
while keeping dark flats distinct during discovery and selection.

`references` is optional for backward compatibility. When present it is the
complete, non-empty, path-distinct target set and must include an identical
copy of the primary `reference`; bias/dark/flat targets have role `light`, and
dark-flat targets have role `flat`. Every selected candidate must match every
target. `dependencies.biasAvailable` defaults false and must mean a usable
bias was actually built: only then may a dark's isolated current scale across
target exposures. Scalable targets and candidates all need known positive
exposures, and selected raw dark inputs must still be mutually exposure-
coherent so master construction can succeed.

Every tolerance is optional and defaults to `seiza-calibration`. The response
is `{schemaVersion,kind,minimum,ready,matchedPaths,selectedPaths,excluded}`.
`matchedPaths` is proximity-sorted; `selectedPaths` is the first coherent set
meeting `minimum`, or the first smaller cluster when none can. Exclusion
reasons are `role-mismatch`, `sensor-mismatch`, `exposure-mismatch`,
`missing-exposure`, `temperature-mismatch`, `optics-mismatch`, or
`outside-coherent-set`.

Raw master construction uses one synchronous JSON call:

```json
{
  "kind": "flat",
  "inputs": ["flat-001.fits", "flat-002.fits"],
  "output": "master-flat.fits",
  "bias": "master-bias.fits",
  "dark": "master-dark-flat.fits",
  "darkExposureSeconds": 2.0,
  "exposureSeconds": null,
  "rejection": {"lowSigma": 3.0, "highSigma": 3.0},
  "defectSuppression": {"lowSigma": 16.0, "highSigma": 16.0}
}
```

`bias` is valid for dark/flat builds; `dark` is valid only for flat builds.
Defect suppression is restricted to flats so a dark retains the hot pixels it
must subtract. Response schema 2 reports dimensions, all integration/rejection
tallies, calibration state, exposure, and this input provenance:

- `requestedFrames` is the number of paths supplied by the caller.
- `inputFrames` and `inputs` cover only the paths actually integrated, in
  request order; each input includes its accepted/rejected sample counts.
- `skippedInputs` names every metadata disagreement set aside by the
  integrator, with its `path` and human-readable `reason`.

The accepted and skipped paths are disjoint and together account for every
requested path. This accounting is validated before the writer publishes, so
no output appears after provenance validation, construction, or cancellation
failure. The writer itself publishes atomically.

```c
SeizaCancelSignal *cancel = seiza_cancel_signal_create();
/* Worker: */ char *report = seiza_calibration_build_master_json(
    request_json, cancel, &error);
/* UI thread, if needed: */ seiza_cancel_signal_cancel(cancel);
/* After the worker returned: */ seiza_cancel_signal_free(cancel);
```

The flag is thread-safe but must stay alive until the build returns.
Cancellation is cooperative between input frames. Progress reporting is not
part of this first ABI; hosts should show an indeterminate cancellable task.

For SNR/depth, `seiza_live_stacker_measure_depth` writes one `SeizaSnrSample`
without copying live buffers and returns `1` measured, `0` insufficient data,
or `-1` error. Test for exactly `1`. `seiza_checkpoint_depths` supplies the
doubling ladder and final depth. Generic curve fitting/projection is not yet in
this ABI; retain measured samples if the host wants to graph them.

The full C declarations, plus the memory-ownership contract (which returns are
owned vs. borrowed, and which `seiza_*_free` to call), live in
[`include/seiza_cabi.h`](include/seiza_cabi.h).

## The C header is generated

`include/seiza_cabi.h` is generated from the Rust source by
[cbindgen](https://github.com/mozilla/cbindgen) via `build.rs`, using
[`cbindgen.toml`](cbindgen.toml). Do not edit it by hand — change the Rust FFI
signatures/docs (or `cbindgen.toml`) and run `cargo build -p seiza-cabi`, which
rewrites the header only when it changes. CI (the `lint` job) fails if a source
change lands without a regenerated header.

## Building

The crate builds three artifacts (`crate-type = ["cdylib", "staticlib",
"rlib"]`):

- `libseiza_cabi.so` / `.dll` / `.dylib` (cdylib) — for the Windows .NET app.
- `libseiza_cabi.a` (staticlib) — for the macOS Swift app.
- rlib — for Rust consumers and the crate's own tests.

```
cargo build -p seiza-cabi --release
```

Consumers link the artifact for their platform and include `seiza_cabi.h`.
