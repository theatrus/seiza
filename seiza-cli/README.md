# seiza-cli

The `seiza` command-line tool: star detection, hinted and blind plate
solving, and star/object dataset management for astrophotography. The
library lives in the [seiza](https://crates.io/crates/seiza) crate.

## Install

### Windows

Download the x86-64 MSI from the
[latest GitHub release](https://github.com/theatrus/seiza/releases/latest).
The installer supports all-users and current-user installs, adds `seiza` to
`PATH` by default, and can launch the guided catalog setup when installation
finishes. All-users installs place catalogs in the shared
`%ProgramData%\Seiza\catalogs` directory; current-user installs use the
user's local application-data directory.

A portable x86-64 ZIP is available on the same release page. See the
[Windows installer documentation](https://github.com/theatrus/seiza/blob/main/packaging/windows/README.md)
for feature selection, unattended installation, and catalog-directory details.

### Cargo

```
cargo install seiza-cli
```

## Solving

```
# Hinted: approximate center (or FITS RA/DEC headers) plus pixel scale.
# --data takes a catalog file or a directory of catalogs; after
# `seiza setup` it can be omitted entirely.
seiza solve image.jpg --data data --ra 324.8 --dec 57.5 --scale 2.8
seiza solve light.fits --scale 1.45

# Blind: no position, just a plausible scale range. A directory supplies
# the deepest catalog and the blind index automatically.
seiza solve-blind image.jpg --data data --min-scale 0.5 --max-scale 15

# Annotate detections or list objects in a solved field
seiza detect image.jpg --annotate out.png
seiza solve image.jpg --data data ... --objects data

# Predict tracks for one exposure from a cached current OMM set, or from an
# offline/historical OMM JSON or TLE file. FITS supplies time and OBSGEO when
# present; explicit metadata is also accepted.
seiza solve light.fits --scale 1.45 --satellites-celestrak --annotate tracks.png
seiza solve light.fits --scale 1.45 --satellites elements.json \
  --time 2026-07-18T08:30:00Z --exposure-seconds 30 \
  --observer-lat 37.3 --observer-lon -122.0 --observer-alt-m 50

# Query objects when sky bounds are already known; no image or solve needed
seiza catalog objects --data data --ra 10.6848 --dec 41.2691 --radius 3
seiza catalog objects --data data \
  --corner 8.91,42.14 --corner 12.47,42.02 \
  --corner 12.31,40.35 --corner 9.02,40.46 \
  --sort prominence --format json

# Resolve exact IDs/names or complete names; no image, solve, or network needed
seiza catalog object --data data "M 31"
seiza catalog object --data data "openngc:NGC224"
seiza catalog object --data data "andro" --prefix --limit 10
seiza catalog star --data data "TYC 5949-2777-1" --format json
seiza catalog star --data data "HIP 32349"
seiza catalog star --data data "RR Lyr"
seiza catalog star --data data "STF 2382 AB"
seiza catalog star --data data "RR L" --prefix --limit 10
```

Explicit file paths still work everywhere a directory is shown, for
custom-built catalogs or unusual layouts.

Star detection defaults to `--detection-backend auto`: decoded 8-bit images
(including color JPEGs) and MTF-compressed FITS use the compact u8 pipeline,
while other higher-precision images use f32. Pass `--detection-backend f32` to
retain fractional luma during an 8-bit solve or to detect directly from linear,
native-precision FITS samples; pass `--detection-backend u8` to explicitly
quantize any input. The option is global and applies to `detect`, `solve`,
`solve-blind`, and local `worker` solves.

Auto solves default to `--detection-fallback f32`. After an Auto/u8 solve miss,
converted 8-bit color is redetected as f32 and FITS is reopened so detection can
use its linear high-precision samples. `--detection-fallback none` disables the
retry. Explicitly selected detection backends never fall back.

`catalog objects` accepts a cone or a convex polygon whose vertices are in
boundary order. It can filter by object kind, magnitude, angular size, and
common-name availability; results can be emitted as a table, JSON, or CSV.
JSON and CSV include primary and alternate stable IDs, primary and contributing
source provenance, aliases, and parent IDs when the catalog provides them. The
prominence score is a catalog-based prediction, not proof that the object is
visible in the image pixels.

`catalog object` resolves primary/common names, aliases, and stable or
alternate IDs. Both object viewport queries and name completion use indices
embedded in the memory-mapped `objects.bin`; normal open does not decode every
record or touch every index page. Add `--all-sources` to audit every normalized
upstream row, preferred facet selection, and source-qualified geometry.

## Background extraction

Fit and remove a smooth background from a calibrated linear FITS or XISF image:

```
seiza background stack.fits --output corrected.fits \
  --model-output background.fits --diagnostics background.json
```

The default is a robust, per-channel quadratic surface with shared sample
positions. Use `--degree 1` for a conservative plane, `--mode divide` for a
multiplicative field, and `--border-fraction` to keep sample windows away from
registration edges. `--sample-radius`, `--samples-per-axis`, rejection sigmas,
and refit count are available for controlled tuning.

Corrected and model outputs are linear 32-bit floating-point FITS and preserve
a valid input WCS. The JSON diagnostics include coefficients, reference
levels, sample positions, weights, and rejection reasons. The input, corrected
output, model, and diagnostics must be distinct paths. Raw Bayer mosaics are
rejected because fitting the interleaved CFA colors as one channel would create
a false surface; debayer or stack them first.

## Light deconvolution (experimental)

`seiza deconvolve` applies a conservative damped Richardson-Lucy pass to a
calibrated or stacked linear mono/RGB FITS/XISF image. Measure the FWHM of several
unsaturated stars in pixels and use their median as the explicit Gaussian PSF:

```text
seiza deconvolve stack-bg.fits --output stack-light-dc.fits \
  --psf-fwhm 3.1 --iterations 4 --amount 0.35 --noise-fraction 0.001
```

Run this after calibration, stacking, and background correction and before a
display stretch. Start with 3-5 iterations and a 0.25-0.4 blend. Compare input
and output with the same stretch, paying particular attention to bright-star
rings, background noise, and image borders. Raw Bayer mosaics are rejected;
debayer or stack them first. Missing registration borders marked with `NaN`
stay masked in the output.

This prototype uses one circular Gaussian PSF across the whole field. It does
not infer detail, estimate a spatially varying PSF, or replace a model-based
restoration workflow. See the
[deconvolution design note](../docs/design/deconvolution.md) for the algorithm,
guardrails, and next experiments. Four
[real-corpus comparisons](../docs/benchmarks/2026-07-deconvolution-corpus.md)
record the first measured trial, and the
[model-based restoration plan](../docs/design/ml-restoration-training.md)
defines a safer path from synthetic degradations and expert before/after pairs
to a provenance-bearing learned operation.

## Image stacking

`seiza stack` calibrates, registers, and incrementally integrates FITS or XISF light
frames. The first light is the fixed output/reference grid:

```
seiza stack light-001.fits light-002.fits light-003.fits \
  --output stack.fits --preview stack.png --report stack-report.json

seiza stack lights/*.fits --output stack.fits \
  --bias master-bias.fits --dark master-dark.fits --flat master-flat.fits \
  --normalization local --local-tile-size 256 \
  --max-registration-drift 256 \
  --max-registration-drift-fraction 0.15 --min-overlap 0.60
```

Raw calibration sequences can be integrated into reusable masters first:

```
seiza master bias bias/*.fits --output master-bias.fits --report master-bias.json
seiza master dark dark/*.fits --bias master-bias.fits \
  --output master-dark.fits --report master-dark.json
seiza master flat flats/*.fits --bias master-bias.fits \
  --dark-flat master-dark-flat.fits --output master-flat.fits \
  --report master-flat.json
```

Master construction uses a two-pass, leave-one-out sigma-clipped mean. It
rereads inputs for the second pass, so memory is proportional to image size,
not frame count. Dimensions, channels, CFA layout, and available camera,
binning, gain, offset, temperature, filter, and dark-exposure metadata are
checked before incompatible frames can be mixed. Flat frames are calibrated
and normalized individually before integration.

Calibration, registration, normalization, online delta-sigma rejection, and
integration all operate on linear `f32` samples. `--preview` is an optional
display-only stretch and never feeds pixels back into the stack. Incoming
frames are admitted atomically: incompatible images, weak registrations,
excess transform drift, low overlap, or implausible normalization leave the
existing additive stack unchanged. The optional `--report` JSON records
SHA-256 identities for every source and calibration master, the complete
configuration, and the ordered accepted/rejected disposition ledger. The
reference is the first light and is always integrated, so it counts toward
`accepted_frames` but has no entry in the per-frame `frames` ledger; that
ledger lists only the frames pushed after it. FITS and report outputs are
published atomically after they are complete.

Mono inputs produce a one-plane linear FITS stack. Three-plane FITS/XISF inputs
remain RGB, while raw one-shot-color frames with `BAYERPAT` are calibrated in
their native CFA sampling before debayering. Star detection uses a temporary
luminance view, but registration is applied to all three channels and global or
local normalization is estimated per channel. The output is an unstretched
three-plane `float32` RGB FITS; `--preview` writes an RGB display stretch.

The registration search is explicitly bounded at the center of the reference
frame. Its effective default is whichever is larger: 256 pixels or 15% of the
reference frame's larger dimension. Configure those components with
`--max-registration-drift` and `--max-registration-drift-fraction`. Increase
either for a sequence with larger dithers or crop-origin offsets; lower values
make the expected motion constraint tighter. Set the fractional component to
zero when a strict pixel-only limit is desired. Light frames may have different
dimensions: valid samples are mapped onto the first frame's fixed grid, pixels
outside a source crop remain masked, and `--min-overlap` controls how much
usable overlap is required for admission.

Meridian-flipped frames are handled automatically. The default 10-degree
rotation gate measures deviation from the nearer of the reference orientation
and its 180-degree counterpart, so a measured transform of 179.3 degrees has a
0.7-degree admission deviation. The complete measured transform—not a header
flag—is then used to rotate and resample the incoming pixels onto the reference
grid before normalization and integration. Accepted-frame diagnostics and the
JSON report retain the full measured rotation.

`--flat` accepts an integrated master flat in the light frame's raw sampling.
For legacy masters, `--bias` removes the flat's pedestal before normalization.
Masters built by `seiza master` carry FITS calibration-state headers, so a
bias-subtracted dark or calibrated flat is not subtracted twice. Planar RGB
flats are normalized independently per channel. CFA flats remain one-channel
and are applied before debayering. See the
[stacking design](https://github.com/theatrus/seiza/blob/main/docs/design/image-stacking.md)
for the live API, rejection semantics, and PSF Guard integration boundary.

### Color composition

`seiza color` consumes mono `float32` FITS stacks. It can emit an RGB
FITS, a quick-look PNG, or both:

```
seiza color rgb --red r.fits --green g.fits --blue b.fits \
  --output rgb.fits --preview rgb.png

seiza color lrgb --luminance l.fits --red r.fits --green g.fits --blue b.fits \
  --luminance-weight 1.0 --output lrgb.fits --preview lrgb.png

seiza color lrgb --luminance l.fits --red r.fits --green g.fits --blue b.fits \
  --luminance-mode super --output super-lrgb.fits --preview super-lrgb.png

seiza color narrowband --ha ha.fits --oiii oiii.fits --sii sii.fits \
  --palette sho --output sho.fits --preview sho.png

seiza color narrowband --ha ha.fits --oiii oiii.fits \
  --palette foraxx-hoo --preview foraxx-hoo.png
```

Direct palettes are `sho`, `soh`, `hso`, `hos`, `osh`, `ohs`, and `hoo`.
Dynamic palettes are `foraxx-sho` and `foraxx-hoo`. The default independent
0.1%/99.5% percentile scaling is intended for fast visual matching; use
`--normalization none` for masters whose backgrounds and scales are already
matched. Foraxx additionally requires those unnormalized samples to lie in
`[0, 1]`; keep the default percentile mode for sensor-unit masters. Foraxx
working channels use a median/MAD midtones transfer; tune it with
`--foraxx-target-median` and `--foraxx-shadows-clip`.

LRGB defaults to linear luminance replacement. `--luminance-mode super`
instead sets the target luminance to `L + R + G + B` after normalization while
preserving RGB chromaticity. Its linear output may exceed one.
`--luminance-weight` applies only to replacement mode. `color rgb` accepts the
same flag: `--luminance-mode super` scales the triplet to a synthetic
`R + G + B` luminance with no luminance stack, marked `SEIZACLR='SUPER-RGB'`.

Linear RGB/LRGB, super-LRGB, super-RGB, and direct-palette FITS files carry
`SEIZATRF='LINEAR'`.
Foraxx follows its published stretched-channel formula and carries
`SEIZATRF='DISPLAY'`; previews therefore do not pretend it is a linear stack.
Input dimensions and path roles are validated, and WCS comes from the command's
reference channel. See the [color-composition
design](https://github.com/theatrus/seiza/blob/main/docs/design/color-composition.md).

By default, non-reference filter stacks are star-registered and resampled onto
L for LRGB, R for RGB, or H-alpha for narrowband. The command reports matched
stars, RMS, drift, and rotation, rejects RMS above 2 pixels, and uses the normal
256-pixel-or-15% drift bound. Configure those gates with
`--max-registration-rms`, `--max-registration-drift`, and
`--max-registration-drift-fraction`; use `--no-register` for masters already
registered to the same reference.

## Persistent worker

Applications performing repeated solves can keep a catalog and blind index
open behind a newline-delimited JSON-RPC 2.0 process:

```
seiza worker --data data --index data
```

The same protocol can adapt local image paths to a queued `seiza-server`
(self-hosted, or the hosted instance at [seiza.fyi](https://seiza.fyi)):

```
seiza worker --server http://solver-host:8080
```

Remote mode defaults to a compact grayscale PNG upload. Use
`--server-upload fits` to stream the original FITS file and preserve headers,
or `--server-timeout SECONDS` to change the default five-minute deadline.
Bearer authentication uses `--server-token` or `SEIZA_SERVER_TOKEN`. See the
[versioned wire contract](https://github.com/theatrus/seiza/blob/main/docs/design/worker-protocol.md)
for request and response details.

## Use with N.I.N.A.

seiza speaks ASTAP's CLI contract: set N.I.N.A.'s plate solver to ASTAP
and point the ASTAP path at the seiza binary (a copy named `astap.exe`
also works). On Windows, use the **Seiza Catalog Setup** Start-menu shortcut
installed by the MSI; on every platform, `seiza setup` installs a usable
selection into a directory ASTAP mode discovers automatically. For a manual
layout, run `seiza download-data prebuilt --output <directory>` and point
`SEIZA_CATALOG_DIR` at that directory. Advanced `SEIZA_STAR_DATA` and
`SEIZA_BLIND_INDEX` file-or-directory overrides remain available. Hinted and
blind-slot solving both work; see the
[ASTAP-compatible mode design](https://github.com/theatrus/seiza/blob/main/docs/design/astap-mode.md).

## Use with Siril

seiza also speaks astrometry.net's `solve-field` CLI contract, and answers
Siril's Windows `bin/bash` launch wrapper itself (no cygwin needed). Run
`seiza install-solve-field --dir <dir>`, point Siril's astrometry.net
directory preference at that directory, and Siril's normal astrometry.net
solving works unchanged, including SIP distortion orders. Catalogs resolve
the same way as ASTAP mode. Siril reports PSF amplitudes rather than photometric
flux, so seiza automatically re-measures star flux from the source image
next to the star table when present — see the
[solve-field mode design](https://github.com/theatrus/seiza/blob/main/docs/design/solve-field-mode.md)
for details.

## Datasets

The quickest route is the prebuilt, SHA-256-verified set hosted at
downloads.seiza.fyi (Tycho-2 and Gaia solver tiles, the blind index, the
unified object catalog, minor bodies, the
Tycho/Bright Star/GCVS/WDS/IAU identifier sidecar, and a nightly-refreshed
transient list):

```
# Running this by itself prints the recommended prebuilt and setup routes:
seiza download-data

seiza download-data prebuilt --output data
seiza download-data prebuilt --output data --file objects.bin --file transients.bin
# The optional ~9 GB Gaia G≤20 catalog is explicit; pair it with the blind index
# for deep blind solving.
seiza download-data prebuilt --output data \
  --file stars-deep-gaia20.bin --file blind-gaia16.idx
```

The other `download-data` subcommands acquire upstream source material for
custom catalog builds; they are not required for normal Seiza use.

Historical satellite elements use the same resolver as applications: a nearby
entry in the durable cache first, then the rolling Seiza mirror, with IAU
SatChecker as the on-demand fallback. Operators can prewarm one or more epochs
without teaching an application which provider to call:

```
seiza download-data satellite-history \
  --epoch 2025-10-17T12:00:00Z 2025-10-18T12:00:00Z \
  --cache /var/lib/seiza/satellite-mirror/cache
```

`seiza build-data satellite-manifest` converts that cache into a validated,
content-addressed publication tree. The complete cron, S3 publication order,
backfill, retention, and public verification procedure is in the
[satellite mirror runbook](https://github.com/theatrus/seiza/blob/main/docs/SATELLITE_MIRROR.md).
The mirror publisher uses `--origin` so its scheduled bucket is fetched from
IAU SatChecker rather than resolved from the mirror it is updating.

For an interactive selection, run `seiza setup`. This is also the command
offered by the Windows installer. It presents lightweight, Gaia, deep-blind,
optional G≤20, and every-catalog presets, then delegates to the same verified
prebuilt downloader. Every preset includes object search, Solar System objects,
and active transients.

The downloader reads one standard bundle from `/data/v4/manifest.json` and
caches its immutable, content-addressed files by SHA-256 before copying the
requested selection into the same flat local output directory. The shared
platform cache can be overridden with `SEIZA_CACHE_DIR`. The optional G≤20
catalog is hosted alongside the bundle but is never included in a bare
`download-data prebuilt`; name it with `--file` or choose the relevant setup
preset. It never combines catalogs from different hosted bundle versions.
Previously released `/data/` and `/data/v2/` paths remain frozen for classic
v1 and v0.4.1/v0.5 readers. The historical `/data/v3/` probe used by v0.4.0
remains reserved and may be absent; those readers retain their existing
fallback behavior.

Library integrations can use `seiza-download` directly for async, automatic
cache management. The raw catalog commands below are implemented by the
separate `seiza-sources` crate so applications do not inherit Gaia/VizieR/MPC
source-acquisition behavior.

Building from primary sources stays supported for custom depths, epochs,
or tile granularity — note the Gaia TAP download alone can take many
hours:

```
seiza download-data tycho2 --output raw/tycho2
seiza download-data star-identifiers --output raw/star-identifiers
seiza build-data tycho2 --input raw/tycho2 --output stars-lite.bin \
  --identifier-index stars-lite.ids.bin \
  --identifier-sources raw/star-identifiers

seiza download-data gaia --output raw/gaia        # Gaia DR3 via TAP, resumable
seiza build-data gaia --input raw/gaia --output stars-gaia.bin

seiza download-data objects --output raw/objects
seiza download-data curation --output raw/curation --commit <git-sha>
seiza build-data objects --input raw/objects --output objects.bin \
  --curation-dir raw/curation \
  --source-manifest objects.sources.json
seiza download-data transients --output raw/transients
seiza build-data transients --input raw/transients --output transients.bin
seiza download-data mpc --output raw/minor-bodies
seiza build-data minor-bodies --input raw/minor-bodies --output minor-bodies.bin
```

The optional curation directory is a pinned local checkout; the builder never
fetches it. Its `curation.json` records repository, commit, and schema version.
Each `objects/<id>.toml` file owns the corrections, relations, selections,
exceptional outline remappings, notes, and structured evidence for one
canonical target. Normally named OpenNGC outlines are associated directly
during upstream ingestion and do not require curation documents. The optional
source manifest records the output hash and size, metadata coverage counts,
source URLs, curation revision, and hashes of every raw catalog and curation
file used.
The optional identifier sidecar provides memory-mapped exact TYC/HIP/HR/HD/
SAO/FK5 lookup plus exact and prefix search over IAU proper names,
Bayer/Flamsteed names, GCVS variables, and WDS double-star designations. It
does not change the solver's compact star tile file.

To publish new transient and Solar-system data without rebuilding the object or
star catalogs, put only the replacement `transients.bin` and
`minor-bodies.bin` in a directory and roll forward each complete manifest:

```shell
seiza build-data manifest --dir next-dynamic \
  --base-manifest current-v2.json \
  --version catalog-bundle-v2-YYYY-MM-DD --output next-v2.json
seiza build-data manifest --dir next-dynamic \
  --base-manifest current-v4.json \
  --version catalog-bundle-v4-YYYY-MM-DD --output next-v4.json \
  --artifact-dir next-v4-artifacts
```

The v2 output retains flat keys for released v3-object readers. The v4 output
uses content-addressed artifact keys and stages both uncompressed and
maximum-compression zstd artifacts in one upload-ready tree. The manifest
retains the uncompressed artifact for old v4 readers; new readers
stream-decompress into the normal uncompressed mmap cache. Both commands
require the resulting bundle to contain every required catalog.

Normal catalog opens do not perform exhaustive validation. Validate any seiza
star tile, identifier sidecar, blind index, object catalog, or minor-body
catalog explicitly when required; the file format is auto-detected:

```
seiza catalog validate --data stars-lite.ids.bin
seiza catalog validate --data stars-lite.bin
seiza catalog validate --data objects.bin
```

FITS and XISF files are read natively through
[seiza-fits](https://crates.io/crates/seiza-fits) and
[`seiza-xisf`](../seiza-xisf/README.md). Both provide typed linear pixels, an
automatic MTF stretch for u8 detection and previews, a linear normalized
representation for f32 detection, and RA/DEC hints from FITS-compatible
metadata. PNG, JPEG, and TIFF continue to use the Rust `image` decoders.
Commands that save linear images write FITS by default and monolithic
`Float32` XISF when the output path ends in `.xisf`.

### Parameterized display stretching

`seiza stretch` applies an explicit reusable `seiza-stretch` model to a linear
mono or RGB FITS/XISF image and writes a display-referred PNG, JPEG, or TIFF:

```text
seiza stretch stack.fits --output preview.png percentile-asinh \
  --black-percentile 0.01 --white-percentile 0.995 --strength 10

seiza stretch stack.fits --output preview.png auto-mtf \
  --target-median 0.2 --shadows-clip -2.8

seiza stretch color.fits --output preview.png \
  --color-strategy luminance-preserving asinh \
  --black 0 --white 1 --strength 8

seiza stretch stack.fits --output preview.png ghs \
  --stretch-factor 4 --local-intensity -1 --symmetry-point 0.35 \
  --protect-shadows 0.1 --protect-highlights 0.8
```

Other model subcommands are `identity`, `linear`, and explicit `mtf`. The `ghs`
subcommand exposes the deterministic GHS parameters; informed automatic
selection can be layered over it later. RGB may use `linked`, `unlinked`, or
`luminance-preserving` analysis/application.
Stretching is never applied to linear stack output unless this command or a
library caller explicitly requests it. See the
[stretching design](../docs/design/stretching.md).

## License

Apache-2.0
