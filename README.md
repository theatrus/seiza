# seiza (星座)

[![MSRV](https://img.shields.io/badge/MSRV-1.89-blue.svg)](Cargo.toml)

Astrophotography tools in Rust: star detection, hinted and blind plate solving,
WCS transforms, calibrated batch/live stacking, and image processing. Available
as a CLI, Rust crates, Python package, and C ABI for native applications.

<a id="native-apps"></a><a id="apps"></a>

**[Try it in your browser](https://seiza.fyi)** ·
**[Mac app](https://github.com/theatrus/seiza-mac)** ·
**[Windows app](https://github.com/theatrus/seiza-win)** ·
**[Documentation](docs/README.md)**

<a id="feature-matrix"></a><a id="status"></a><a id="performance"></a>

## Features and performance

The performance links describe the inputs, hardware, and measurement limits.
“Unmeasured” means no separate runtime benchmark is documented for that operation.

| Capability | What it does | Performance |
|---|---|---|
| Star detection and measurement | Subpixel centroids, flux, HFR/FWHM, tilt analysis | Compact detection uses 179 MiB on a 61 MP FITS frame; [measurement/tilt latency unmeasured](docs/performance.md#operation-coverage) |
| Hinted plate solving | Solve from approximate position and pixel scale | [0.10 s median](docs/performance.md#plate-solving), 17 FITS frames, i5-1340P |
| Blind plate solving | Solve without a position hint using a prebuilt index | [0.09 s median](docs/performance.md#plate-solving), 16 FITS frames, i5-1340P |
| WCS and SIP | Pixel/sky transforms, FITS WCS output, optional order 2–5 distortion | [Included in solve timings](docs/performance.md#plate-solving); standalone transforms unmeasured |
| Catalogs and overlays | Object/name/cone queries, distances, constellations, sky maps | Memory-mapped catalogs; [query/render latency unmeasured](docs/performance.md#operation-coverage) |
| Image I/O | FITS, XISF, PNG, JPEG, TIFF; linear float output and raster export | [16.5 ms FITS decode](docs/performance.md#image-io-and-display), warm local 61 MP frame; other formats unmeasured |
| Calibration and masters | Bias/dark/flat construction, frame grading, response and dust analysis | [Standalone timings unmeasured](docs/performance.md#operation-coverage) |
| Batch and live stacking | Calibration, registration, normalization, rejection, drizzle, mosaics, resumable contexts | [200 × 61 MP mono frames in 12.4–12.5 s](docs/performance.md#stacking), M4 Max GPU with reintegration |
| Display stretching | Ordered MTF, GHS, asinh, median-anchor, and layered curves | [~25 ms for a 26 MP legacy display stretch](docs/performance.md#image-io-and-display); other models unmeasured |
| Background correction | Robust polynomial/RBF fitting, subtraction or division | [~0.7 s for a 26 MP command](docs/performance.md#background-correction), local smoke test including outputs |
| Color composition and calibration | RGB/LRGB, narrowband palettes, Gaia photometric color calibration | [Standalone timings unmeasured](docs/performance.md#operation-coverage) |
| Deconvolution (experimental) | Damped Richardson–Lucy with a supplied Gaussian PSF | [Runtime unmeasured](docs/performance.md#operation-coverage); four iterations by default |
| Satellite prediction | Current/historical orbital elements projected through one exposure | [Runtime unmeasured](docs/performance.md#operation-coverage); requires time and observer position |
| Parallax videos | Distance-based fly-throughs from starless and star images | [Runtime unmeasured](docs/performance.md#operation-coverage); rendering and encoding depend on output settings |
| Image-processing primitives | Blur, thresholds, contours, morphology, edge-aware filtering | [Runtime unmeasured](docs/performance.md#operation-coverage) |

<a id="gpu-acceleration"></a>

Stacking accelerates supported passes through Metal, Vulkan, or Direct3D 12,
with CPU fallback. Linear stacks remain unstretched; previews are separate.
See [features and limits](docs/features.md) and the
[full performance report](docs/performance.md) for details and solver comparisons.
See [GPU controls](docs/image-processing.md#gpu-controls) to select CPU/GPU passes.

<a id="ways-to-use-it"></a><a id="minimum-supported-rust-version"></a>

## Install

| Interface | Install |
|---|---|
| Mac app | `brew install --cask theatrus/seiza/seiza-mac` or [download a DMG](https://github.com/theatrus/seiza-mac/releases) |
| Windows app | [Seiza for Windows](https://github.com/theatrus/seiza-win) |
| CLI, macOS/Linux | `brew install theatrus/seiza/seiza` |
| CLI, Windows/Fedora/Ubuntu | MSI, portable ZIP, RPM, or deb from [releases](https://github.com/theatrus/seiza/releases) |
| CLI, from Rust | `cargo install seiza-cli` (Rust 1.89+) |
| Python | `pip install seiza` ([API and NumPy examples](seiza-py/README.md)) |
| Rust / C ABI | [Crate guide](docs/README.md#rust-crates) / [native API](seiza-cabi/README.md) |

## Quick start

Install the CLI, download catalogs once, and solve an image:

```sh
seiza setup --preset blind-deep
seiza solve-blind image.fits --min-scale 0.5 --max-scale 15
```

Scale bounds are in arcseconds per pixel. Catalogs installed by setup are
found automatically. Solving works locally after installation.

Hinted solving reads position hints from image metadata; explicit options are
available when those are absent. To solve with a scale hint or stack light frames:

```sh
seiza solve image.fits --scale 1.26
seiza stack lights/*.fits --output stack.fits --reintegrate --preview stack.png
```

See the [solving guide](docs/solving.md) for position and scale options.

## Documentation

- [Features and limits](docs/features.md)
- <a id="commands-and-services"></a>[Solving, annotations, and workers](docs/solving.md)
- <a id="image-stacking"></a><a id="automatic-background-extraction"></a><a id="color-from-mono-stacks"></a><a id="light-deconvolution-experimental"></a>[Stacking, background correction, color, and deconvolution](docs/image-processing.md)
- <a id="catalogs-and-data"></a>[Catalog setup, storage, and custom builds](docs/catalogs.md)
- <a id="use-with-nina-astap-compatible-mode"></a><a id="use-with-siril-solve-field-compatible-mode"></a>[N.I.N.A., Siril, and application integration](docs/integrations.md)
- <a id="compared-with-asterion"></a><a id="compared-with-astap"></a>[Performance by operation and benchmark methodology](docs/performance.md)
- <a id="library-apis"></a>[CLI reference](seiza-cli/README.md), [Python API](seiza-py/README.md), [C ABI](seiza-cabi/README.md)
- <a id="layout"></a>[Crates, design notes, and development](docs/README.md)

Built into [PSF Guard](https://github.com/theatrus/psf-guard),
[Tenrankai](https://github.com/theatrus/tenrankai), and
[seiza-server](https://github.com/theatrus/seiza-server).

## License

Apache-2.0. Embedded constellation figures are CC BY 4.0
([attribution](seiza/data/constellation-lines/README.md)); Inter is SIL OFL 1.1
([font license](seiza-cli/fonts/README.md)).
