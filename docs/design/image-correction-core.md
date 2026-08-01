# Host-neutral image correction core

Status: plan. This records the ownership and API changes needed before Seiza
image corrections become useful outside the CLI and PSF Guard.

## Goal

Every correction kernel should live in a focused Seiza crate and work on
linear pixel buffers without a database, web server, FITS path, or UI. Seiza's
CLI, Python package, C ABI, stacker, PSF Guard, and later native host modules
should call those same kernels.

"Seiza core" means these host-neutral Rust crates and the stable `seiza`
facade. It does not mean one large crate that owns every image operation.

## Current split

Several operations already have the right ownership:

- `seiza-background` owns background fitting, model rendering, and correction.
- `seiza-deconvolution` owns conservative deconvolution.
- `seiza-stretch` owns display stretch operations.
- `seiza-imgproc` owns low-level operations on row-major slices.
- `seiza-cabi` adapts several of these operations for native callers.

PR [#103](https://github.com/theatrus/seiza/pull/103) puts residual-response
estimation and bounded application in `seiza-stacking`. That is a useful first
shared implementation, but the code is not a stacking algorithm. It depends on
`seiza-stacking::LinearImage`, which makes a native image editor or a
calibration tool depend on the full stacker.

PSF Guard should continue to own the parts that need catalog context:

- finding a suspicious region and stating the strength of its evidence;
- checking that the feature stays fixed on the detector while the sky moves;
- choosing source frames and calibration masters;
- jobs, cache keys, provenance, HTTP endpoints, and user confirmation;
- labels and grading policy.

Seiza should own the pixel operation once the host supplies the samples,
region, and settings. A Seiza result must not claim that a dark ring is dust.

## Target crate ownership

Create `seiza-calibration` as the host-neutral home for calibration response
models. Move these items from `seiza-stacking` without changing their numeric
behavior:

- bias, dark, and flat application;
- flat-response normalization;
- calibration-master construction and diagnostics;
- residual-response options, fitting, diagnostics, model rendering, and
  bounded application.

Keep FITS loading, registration, stack normalization, rejection, accumulation,
and stack checkpoints in `seiza-stacking`. The stacker should depend on
`seiza-calibration` and re-export moved names for one release when that avoids
needless downstream churn. The top-level `seiza` crate should re-export the
stable calibration API.

Do not move background extraction or deconvolution into the new crate. Their
focused crates already give other hosts a clean API. Add a shared image type
only if several independent crates need to own the same buffer. For now, keep
the established slice, dimensions, and channel-count APIs so callers retain
their image storage.

## Core API contract

The residual-response API should follow the background fit/apply split:

1. Fit a compact or cropped response model from detector-aligned linear
   samples.
2. Return serializable settings, an algorithm version, and diagnostics.
3. Render the response map only when the caller asks for it.
4. Apply a fitted response in place to a caller-owned buffer.

The API must:

- accept row-major `f32` slices, dimensions, channel count, row stride, and an
  optional validity mask;
- keep estimation separate from application;
- preserve finite linear values without an implicit stretch, clamp, or color
  neutralization;
- reject invalid dimensions, unsafe divisors, and incompatible channel counts;
- preserve non-finite masked borders instead of treating them as signal;
- bound every multiplicative gain and report the applied maximum;
- expose corrected area, largest connected area, consensus, response range,
  and rejected-sample counts;
- use deterministic sampling and stable serialized settings;
- include an algorithm version suitable for cache keys and provenance.

The estimator should accept already chosen detector-aligned samples. Dither
measurement and any claim that the samples show a stable detector defect stay
with the host until we can define a general evidence type without importing
PSF Guard policy.

## Native ABI

Extend `seiza-cabi` with opaque handles that match the existing background
model lifecycle:

- create or fit a residual-response model from caller-owned buffers;
- return borrowed diagnostics JSON from the model;
- render a response into a caller-owned output buffer;
- apply a response to a caller-owned image buffer;
- destroy the handle with the matching Seiza function.

No Rust allocation should cross the ABI. Catch panics at every exported
boundary and return a structured error through the existing error path. Add
explicit planar and interleaved layouts, row strides, and sample type to the
ABI instead of making every native host repack a full image. Start with `f32`;
add a native `f64` path only after parity and performance tests show that it is
worth the added surface.

Long operations need optional progress and cancellation callbacks. Callbacks
must run on a documented thread and must never retain host-owned pointers after
the call returns.

## PixInsight host

A later PixInsight module should be a thin C++ adapter over `seiza-cabi`, using
PixInsight's native [PCL module API](https://www.pixinsight.com/developer/pcl/).
The module should not link to PSF Guard or copy Seiza algorithms.

The first useful processes are:

- **Seiza Background Correction**: fit, inspect, render, and apply a background
  model to one view;
- **Seiza Deconvolution**: apply the current conservative model to one view;
- **Seiza Apply Response**: inspect and apply an existing flat or residual
  response;
- **Seiza Build Residual Response**: use an explicit list of aligned source
  views or files and write the response plus diagnostics.

Single-image processes can support a real-time preview. Multi-frame response
fitting should run as a global process with progress and cancellation. It must
show the response and diagnostics before application. It must not name the
cause of a feature unless the user or host supplies that label.

## Migration phases

### 1. Lock the contract and extract without changing pixels

- Save regression fixtures for calibration and residual-response output before
  moving code.
- Add `seiza-calibration` to the workspace.
- Give it contiguous slice-based APIs so it never depends on
  `seiza-stacking::LinearImage`.
- Move calibration and residual-response code from `seiza-stacking` and adapt
  the stacker's image buffers at that boundary.
- Keep the same defaults, bounds, diagnostics, and test vectors.
- Re-export moved names from `seiza-stacking` for one release if downstream
  code needs a migration window.
- Add parity tests that compare the saved output fixtures with the new crate.
- Make `seiza-stacking` consume the new crate.

### 2. Add native-host image layouts

- Add borrowed image views with explicit planar or interleaved layout, row and
  channel strides, and output buffers. Keep the contiguous helpers as the
  simple Rust API.
- Add masks, algorithm versions, and serialized model round trips.
- Test mono, RGB, NaN borders, cropped responses, invalid buffers, and bounded
  gains.
- Back-test the existing synthetic cases and the C925 positive and negative
  data sets used for PR #103.

### 3. Expose all correction kernels through the C ABI

- Add residual-response fit, render, and apply handles.
- Add layout, error, progress, cancellation, and ownership tests.
- Compare Rust and C ABI output byte-for-byte for `f32` inputs.
- Publish a small C example that applies a saved response without FITS I/O.

### 4. Simplify PSF Guard

- Replace local crop, response-application, and image-buffer glue when the new
  core API covers them.
- Retain source selection, dither evidence, catalog policy, cache management,
  preview jobs, and UI in PSF Guard.
- Record the Seiza algorithm version and serialized settings in every cached
  correction manifest.
- Prove that the current PSF Guard preview and downloadable FITS remain
  unchanged on the same inputs.

### 5. Build the native module

- Start with background correction and apply-response processes.
- Add deconvolution after the buffer, mask, progress, and cancellation paths
  pass host tests.
- Add multi-frame residual fitting last because it needs file-list, alignment,
  and long-job UI.
- Package and sign the module through a PixInsight update repository only
  after the SDK and release terms have been checked for the target platforms.

## Acceptance criteria

The move is complete when:

- no host-neutral calibration or response estimator depends on
  `seiza-stacking`;
- stacking, CLI, Python, C ABI, and PSF Guard call the same Rust kernels;
- equal `f32` inputs and settings yield equal pixels across Rust and the C ABI;
- cancellation, errors, masks, bounds, and ownership have regression tests;
- real positive and negative data tests keep weak evidence unclassified;
- every correction can render its model and diagnostics before changing an
  image;
- docs state the order of linear operations and the provenance written by each
  host.

## Out of scope

This plan does not add PSF Guard databases, grading, cache jobs, or automatic
dust labels to Seiza. It does not define a general plug-in framework, add GPU
code, or promise that a residual response can replace a measured flat. Those
changes need their own evidence and design work.
