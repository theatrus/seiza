# Features and limits

[Documentation](README.md) · [Performance by operation](performance.md)

## Astrometry

- **Detection:** tile-based median/MAD background estimation, sigma thresholding,
  connected components, and flux-weighted subpixel centroids. Compact `u8` and
  linear `f32` detector paths are available. Measured stars add shape/size
  statistics for field and tilt analysis.
- **Hinted solving:** approximate center and pixel scale seed triangle matching,
  candidate voting, and iterative refinement.
- **Blind solving:** a whole-sky four-star pattern index finds fields without a
  position hint. A plausible pixel-scale range bounds the search; the prebuilt
  index avoids rebuilding it for each process.
- **WCS:** TAN projection, CD matrix, pixel/world transforms, scale and footprint
  helpers, and FITS header output. Optional SIP orders 2–5 include forward and
  inverse coefficients. The solver uses the linear fit when the SIP fit does
  not justify the additional coefficients. See
  [real-field validation](benchmarks/2026-07-sip-real-field-validation.md).
- **Confidence and control:** solutions report a false-match probability accounting
  for the candidates tried. Blind solving rejects probabilities above one in a
  billion. Both solvers support time limits and cancellation.

See [solving](solving.md), the [solver crate](../seiza/README.md), and
[performance](performance.md#plate-solving).

## Catalogs and annotations

Memory-mapped star and object catalogs support sky-cone and footprint queries,
name/identifier lookup, and prefix completion without a solve. Object details
include upstream records, aliases, geometry, relations, and provenance.

Solved images can show named stars, deep-sky objects, transients, Solar System
objects, and embedded constellation figures. Optional satellite overlays predict
paths during a single exposure; they do not detect trails in the image.
Distances support catalog queries and parallax rendering.

See [catalog contents and setup](catalogs.md) and [annotation commands](solving.md).
Opening a catalog uses local files. Install or update data with the
[catalog commands](catalogs.md).

## Images and processing

| Operation | Support and limits | Guide |
|---|---|---|
| I/O | FITS integer/float data, planar RGB and Bayer metadata; XISF with compressed attached blocks; PNG/JPEG/TIFF with EXIF orientation. Linear float FITS/XISF output; raster display export. | [Image I/O](../seiza-image-io/README.md) |
| Calibration | Bias/dark/flat masters, matching and grading, residual-response fitting, flat changes and dust features. | [Calibration](../seiza-calibration/README.md) |
| Stacking | Mono/RGB/CFA calibration, automatic reference selection, similarity/affine/quadratic registration, meridian flips, local normalization, noise weighting, bilinear/Lanczos-3 resampling, online rejection and completed-stack reintegration. | [Stacking](image-processing.md#stacking) |
| Stack geometry | Cropped inputs, expanded mosaic canvases, drizzle, binning, and solved north-up/east-left reprojection. Saved `.seiza-stack` contexts resume live integration. | [Stacking API](../seiza-stacking/README.md) |
| Stretching | Identity, linear, asinh, percentile-asinh, MTF, GHS, Auto-MTF, median-anchor and layered models; ordered stages keep float intermediates and resolved plans. | [Stretch API](../seiza-stretch/README.md) |
| Background | Robust polynomial and smoothed thin-plate RBF surfaces, masks/regions, automatic polynomial selection, subtraction and division. RBF is opt-in. | [Background](image-processing.md#background-extraction) |
| Color composition | RGB, LRGB, additive super-luminance, SHO/HOO, direct filter permutations and custom matrices. Foraxx palettes produce display-referred output. | [Color](image-processing.md#color-from-mono-stacks) |
| Photometric color | Gaia BP−RP stellar-color fitting and background neutralization; requires solved linear RGB and offline photometry or a catalog query. | [Color calibration](image-processing.md#photometric-color-calibration) |
| Deconvolution | Experimental damped Richardson–Lucy, supplied circular Gaussian PSF, mono/RGB and masked registration borders. Raw Bayer input is rejected. | [Deconvolution](image-processing.md#light-deconvolution-experimental) |
| Parallax | Distance-based stars, lifted objects/depth planes, dust, tours, labels, and video/PNG output. Requires separate starless and star images. | [Parallax](../seiza-parallax/README.md) |
| Primitives | OpenCV-compatible blur, Canny, Otsu, morphology, contours and edge-aware filters in pure Rust. | [Image processing](../seiza-imgproc/README.md) |

[Performance by operation](performance.md#operation-coverage) lists measurements
and identifies operations that still need benchmarks.

## Interfaces

| Interface | Use it for |
|---|---|
| [CLI](../seiza-cli/README.md) | File workflows, catalog administration, solving, stacking, processing, video, and ASTAP/`solve-field` compatibility |
| [Rust crates](README.md#rust-crates) | Direct access to the core algorithms and focused image-processing libraries |
| [Python](../seiza-py/README.md) | NumPy processing, solving, catalogs, calibration, stacking, and parallax; [type stubs](../seiza-py/seiza.pyi) describe the public API |
| [C ABI](../seiza-cabi/README.md) | Native hosts: rendering, solving, catalogs, calibration/masters, incremental stacking, composition, satellite prediction, and parallax |
| [Worker JSON](design/worker-protocol.md) | Persistent local hinted/blind solves, or forwarding to a server; catalogs stay open between local requests |
| [Server](https://github.com/theatrus/seiza-server) | Hosted/queued solves, web uploads, history, and an astrometry.net-compatible HTTP API |

The worker accepts solving requests. Local solves support SIP; remote worker
results currently use linear TAN. Calling conventions also vary between
libraries: C hosts use the incremental stacker for batch workflows, while Python
also provides `stack_fits`. Consult each app's documentation for the controls
available in its UI.
