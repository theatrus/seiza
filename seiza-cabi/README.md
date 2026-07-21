# seiza-cabi

A C ABI over Seiza for native application front-ends. It is the single, shared
successor to the near-identical `seiza-cabi` crates that previously lived in the
[`seiza-win`](https://github.com/theatrus/seiza-win) (.NET) and
[`seiza-mac`](https://github.com/theatrus/seiza-mac) (Swift) repositories, and it
exposes the **superset** of what both apps need.

## What it exposes

- **FITS / raster rendering** — `seiza_rendered_image_open`,
  `..._open_with_rgb_stretch`, `..._width`, `..._height`,
  `..._metadata_json`, `..._free`.
- **Both pixel byte orders** — `seiza_rendered_image_rgba` (macOS / CoreGraphics)
  and `seiza_rendered_image_bgra` (Direct2D / WinUI), each with a `_length`
  companion. RGBA is canonical; the BGRA view is computed on first request and
  cached, so a consumer only ever pays for the order it uses.
- **Parameterized stretch** — `seiza_rendered_image_open_with_stretch_config`
  takes a serialized `seiza-stretch` `StretchConfig` (JSON) and renders a FITS
  through the full GHS/MTF/percentile pipeline. New stretch capabilities are
  added in `seiza-stretch`; this shim only marshals JSON in and pixels out.
- **Background extraction** — `seiza_background_fit` creates a compact opaque
  model from interleaved linear mono or RGB `float` samples. Callers can inspect
  its borrowed diagnostics JSON, render it into a caller-owned buffer, or apply
  subtractive/divisive correction in place before freeing the model. Optional
  settings use serialized `seiza-background` `BackgroundConfig` JSON, keeping
  the ABI stable as model options grow.
- **Plate solving** — `seiza_solve_image_json`.
- **Catalog setup** — `seiza_catalog_status_json` and `seiza_catalog_setup`
  (with a progress callback). The install path delegates to
  `seiza-download`'s `materialize_with`; the shim carries no download logic.
- **Memory** — `seiza_core_version`, `seiza_string_free`.

Rendered-image metadata includes input and display histograms.

Background input and output buffers are row-major, pixel-interleaved `float`
samples with one or three channels. The model copies only compact samples and
coefficients, so the input buffer may be released after `seiza_background_fit`
returns. Pass null for both mask/configuration pointers to use automatic
defaults. Correction mode constants and the precise pointer/length contracts
are declared in the generated header.

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
