# seiza (星座)

[![MSRV](https://img.shields.io/badge/MSRV-1.89-blue.svg)](Cargo.toml)

Astrophotography tools in Rust, available as a CLI, Python package, Rust crates,
and C ABI.

<a id="feature-matrix"></a><a id="status"></a><a id="features-and-performance"></a>

## Features

- **Plate solving:** hinted and blind solving, with WCS and SIP distortion output.
- **Star measurement:** subpixel positions, flux, HFR/FWHM, and sensor tilt analysis.
- <a id="gpu-acceleration"></a>**Stacking:** calibrated batch and live stacking, GPU acceleration, rejection,
  drizzle, mosaics, and resumable sessions.
- **Image processing:** stretching, background correction, RGB/LRGB and narrowband
  composition, photometric color calibration, and experimental deconvolution.
- **Catalogs and overlays:** star and deep-sky lookup, constellations, sky maps,
  and satellite track prediction.
- **Parallax videos:** distance-based fly-throughs from starless and star images.
- **Image formats:** FITS, XISF, PNG, JPEG, and TIFF.

<a id="ways-to-use-it"></a><a id="minimum-supported-rust-version"></a>

<a id="native-apps"></a><a id="apps"></a>

## Install

[Try it in your browser](https://seiza.fyi), or install an app or library:

| Interface | Install |
|---|---|
| Mac app | `brew install --cask theatrus/seiza/seiza-mac` or [download a DMG](https://github.com/theatrus/seiza-mac/releases) |
| Windows app | [Seiza for Windows](https://github.com/theatrus/seiza-win) |
| CLI, macOS/Linux | `brew install theatrus/seiza/seiza` |
| CLI, Windows/Fedora/Ubuntu | MSI, portable ZIP, RPM, or deb from [releases](https://github.com/theatrus/seiza/releases) |
| CLI, from Rust | `cargo install seiza-cli` (Rust 1.89+) |
| Python | `pip install seiza` ([API and NumPy examples](seiza-py/README.md)) |
| Rust / C ABI | [Crate guide](docs/README.md#rust-crates) / [native API](seiza-cabi/README.md) |

<a id="quick-start"></a>

## Use

Install catalogs once, then solve images or stack light frames. Scales are in
arcseconds per pixel; hinted solving uses position metadata from the image.

```sh
seiza setup --preset blind-deep
seiza solve-blind image.fits --min-scale 0.5 --max-scale 15
seiza solve image.fits --scale 1.26
seiza stack lights/*.fits --output stack.fits --reintegrate --preview stack.png
```

Seiza finds installed catalogs automatically and solves locally. See the
[solving guide](docs/solving.md) for position and scale options.

## Documentation

- [Features and limits](docs/features.md)
- <a id="commands-and-services"></a>[Solving, annotations, and workers](docs/solving.md)
- <a id="image-stacking"></a><a id="automatic-background-extraction"></a><a id="color-from-mono-stacks"></a><a id="light-deconvolution-experimental"></a>[Stacking, background correction, color, and deconvolution](docs/image-processing.md)
- <a id="catalogs-and-data"></a>[Catalog setup, storage, and custom builds](docs/catalogs.md)
- <a id="use-with-nina-astap-compatible-mode"></a><a id="use-with-siril-solve-field-compatible-mode"></a>[N.I.N.A., Siril, and application integration](docs/integrations.md)
- <a id="performance"></a><a id="compared-with-asterion"></a><a id="compared-with-astap"></a>[Benchmarks and methodology](docs/performance.md)
- <a id="library-apis"></a>[CLI reference](seiza-cli/README.md), [Python API](seiza-py/README.md), [C ABI](seiza-cabi/README.md)
- <a id="layout"></a>[Crates, design notes, and development](docs/README.md)

Built into [PSF Guard](https://github.com/theatrus/psf-guard),
[Tenrankai](https://github.com/theatrus/tenrankai), and
[seiza-server](https://github.com/theatrus/seiza-server).

## License

Apache-2.0. Embedded constellation figures are CC BY 4.0
([attribution](seiza/data/constellation-lines/README.md)); Inter is SIL OFL 1.1
([font license](seiza-cli/fonts/README.md)).
