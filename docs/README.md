# Seiza documentation

[Project overview and install](../README.md)

## User guides

| Guide | Contents |
|---|---|
| [Features and limits](features.md) | Supported operations and interfaces |
| [Solving](solving.md) | Hinted/blind solving, SIP, overlays, and workers |
| [Image processing](image-processing.md) | Stacking, GPU controls, stretching, background, color, and deconvolution |
| [Catalogs](catalogs.md) | Setup, discovery, contents, queries, and custom builds |
| [Integrations](integrations.md) | N.I.N.A., Siril, and application entry points |
| [Performance](performance.md) | Recorded timings, memory, comparisons, and reproduction |

## API references

- [CLI commands](../seiza-cli/README.md)
- [Python package and NumPy examples](../seiza-py/README.md), [type stubs](../seiza-py/seiza.pyi)
- [C ABI](../seiza-cabi/README.md), [generated C header](../seiza-cabi/include/seiza_cabi.h)
- [Worker JSON protocol](design/worker-protocol.md)

## Rust crates

Use the focused crate for an operation; depend on `seiza` when you need solving,
detection, or catalogs. `seiza-core` provides shared WCS/cancellation/prior types
without the solver. See [dependency and release rules](../RELEASE.md).

| Crate | Purpose |
|---|---|
| [seiza](../seiza/README.md) | Detection, hinted/blind solving, catalogs, and WCS re-exports |
| [seiza-core](../seiza-core/README.md) | Shared WCS, cancellation, and solve priors |
| [seiza-fits](../seiza-fits/README.md) | FITS reading and atomic linear float writing |
| [seiza-xisf](../seiza-xisf/README.md) | XISF reading and atomic linear float writing |
| [seiza-image-io](../seiza-image-io/README.md) | Extension-based image I/O, EXIF orientation and metadata |
| [seiza-stats](../seiza-stats/README.md) | Shared robust image statistics |
| [seiza-imgproc](../seiza-imgproc/README.md) | Image-processing primitives |
| [seiza-background](../seiza-background/README.md) | Robust background fitting and correction |
| [seiza-calibration](../seiza-calibration/README.md) | Calibration responses, matching, grading, and dust analysis |
| [seiza-stretch](../seiza-stretch/README.md) | Display analysis, transfer plans, and ordered stretches |
| [seiza-stacking](../seiza-stacking/README.md) | Calibration, registration, normalization, rejection, mosaics, and live stacking |
| [seiza-deconvolution](../seiza-deconvolution/README.md) | Experimental linear-image restoration |
| [seiza-stars](../seiza-stars/src/lib.rs) | Peak-star detection and centroid refinement |
| [seiza-draw](../seiza-draw/README.md) | Overlay drawing and labels |
| [seiza-parallax](../seiza-parallax/README.md) | Distance-based scenes, tours, frames, and video |
| [seiza-satellites](../seiza-satellites/README.md) | Time/observer-aware satellite tracks |
| [seiza-download](../seiza-download/README.md) | Verified runtime catalog cache |
| [seiza-sources](../seiza-sources/README.md) | Raw upstream acquisition for catalog builders |
| [seiza-cabi](../seiza-cabi/README.md) | Native application bindings |
| [seiza-cli](../seiza-cli/README.md) | Command-line tool |

## Design and evidence

Start with the [user guides](#user-guides) for instructions. Design notes explain
internals, tradeoffs, and proposals.

- Astrometry: [solver lessons](design/solver-lessons-from-olive-solve.md),
  [ASTAP compatibility](design/astap-mode.md), [solve-field compatibility](design/solve-field-mode.md)
- Processing: [stacking](design/image-stacking.md), [GPU stacking](design/gpu-stacking.md),
  [binning](design/binned-stacking.md), [background](design/background-extraction.md),
  [stretching](design/stretching.md), [composition](design/color-composition.md),
  [color calibration](design/color-calibration.md), [deconvolution](design/deconvolution.md)
- Catalogs: [bundles](design/catalog-bundles.md), [lookup](design/catalog-lookup.md),
  [object format](design/objects-bin-v4.md), [identifiers](design/star-identifiers.md)
- Rendering: [depth planes](design/parallax-depth-planes.md),
  [video overlays](design/parallax-overlay-images.md), [satellite tracks](design/satellite-tracks.md)
- [Benchmark reports](benchmarks/) and [scientific regression corpus](../regression/README.md)
- [Satellite mirror operator runbook](SATELLITE_MIRROR.md)

## Development

Build from the repository with `cargo build --release -p seiza-cli`. The minimum
supported Rust version is **1.89**, declared in `Cargo.toml` and checked in CI;
`rust-toolchain.toml` pins the compiler, formatter, and linter used for local
repository builds. `seiza-py` is outside the workspace and is built separately.

Follow [AGENTS.md](../AGENTS.md) for binding parity, output-version protection,
and required checks, and [RELEASE.md](../RELEASE.md) for versioning and publishing.
[Windows packaging](../packaging/windows/README.md) describes the CLI installer.

See the [Seiza 0.44.0 release notes](releases/v0.44.0.md) for registered mosaic
background matching and independent RGB star stretching, and the
[0.44.1 notes](releases/v0.44.1.md) for its fix to local normalization on an
expanded canvas.
