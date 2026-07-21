# seiza-cabi

A C ABI over Seiza for native application front-ends. It is the single, shared
successor to the near-identical `seiza-cabi` crates that previously lived in the
[`seiza-win`](https://github.com/theatrus/seiza-win) (.NET) and
[`seiza-mac`](https://github.com/theatrus/seiza-mac) (Swift) repositories, and it
exposes the **superset** of what both apps need.

## What it exposes

- **FITS / XISF / raster rendering** — `seiza_rendered_image_open`,
  `..._open_with_rgb_stretch`, `..._width`, `..._height`,
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
  `background` correction, optional `deconvolution`, and optional
  `interactive_preview` mode. Background fitting and subtraction/division run
  first on linear FITS samples, followed by deconvolution and then the display
  stretch. Interactive previews bound those samples before expensive processing
  while committed renders remain full resolution. The two most recent prepared
  preview buffers are cached by file identity, maximum dimension, and background
  configuration; stretch and deconvolution edits reuse the same corrected linear
  pixels. New processing capabilities remain in their core crates; this shim
  only marshals JSON in and pixels out.
- **Background extraction** — `seiza_background_fit` creates a compact opaque
  model from interleaved linear mono or RGB `float` samples. Callers can inspect
  its borrowed diagnostics JSON, render it into a caller-owned buffer, or apply
  subtractive/divisive correction in place before freeing the model. Optional
  settings use serialized `seiza-background` `BackgroundConfig` JSON, keeping
  the ABI stable as model options grow.
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
- **Plate solving** — `seiza_solve_image_json`.
- **Catalog setup** — `seiza_catalog_status_json` and `seiza_catalog_setup`
  (with a progress callback). The install path delegates to
  `seiza-download`'s `materialize_with`; the shim carries no download logic.
- **Memory** — `seiza_core_version`, `seiza_string_free`.

Rendered-image metadata includes input and display histograms.

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

Deconvolution uses the same buffer layout and modifies the caller-owned input
in place. Supply a measured stellar PSF FWHM in pixels; the conservative values
are four iterations, a `0.35` blend, a `0.001` noise fraction, and a maximum
correction of `2.0`. The output remains linear and may contain samples outside
`[0, 1]`.

Stacking uses the same row-major, pixel-interleaved mono/RGB layout. Array
frames are copied during each synchronous call and may be released when it
returns; they must already be calibrated, debayered, and linear. FITS/XISF
pushes retain and apply the calibration masters loaded by the path constructor; a
non-zero dark exposure override requires a dark master path. A
rejected frame is represented by `accepted: false` disposition JSON and is not
an ABI error. The zero-copy live pointers are invalidated by the next push,
finish, or free; immutable snapshot pointers remain valid until snapshot free.
Snapshot FITS output refuses to overwrite any tracked light or calibration
input.

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
