# seiza-cli

The `seiza` command-line tool: star detection, hinted and blind plate
solving, and star/object dataset management for astrophotography. The
library lives in the [seiza](https://crates.io/crates/seiza) crate.

```
cargo install seiza-cli
```

## Solving

```
# Hinted: approximate center (or FITS RA/DEC headers) plus pixel scale
seiza solve image.jpg --data stars.bin --ra 324.8 --dec 57.5 --scale 2.8
seiza solve light.fits --data stars.bin --scale 1.45

# Blind: no position, just a plausible scale range
seiza solve-blind image.jpg --data stars.bin --min-scale 0.5 --max-scale 15

# Annotate detections or list objects in a solved field
seiza detect image.jpg --annotate out.png
seiza solve image.jpg --data stars.bin ... --objects objects.bin

# Query objects when sky bounds are already known; no image or solve needed
seiza catalog objects --data objects.bin --ra 10.6848 --dec 41.2691 --radius 3
seiza catalog objects --data objects.bin \
  --corner 8.91,42.14 --corner 12.47,42.02 \
  --corner 12.31,40.35 --corner 9.02,40.46 \
  --sort prominence --format json

# Resolve exact IDs/names or complete names; no image, solve, or network needed
seiza catalog object --data objects.bin "M 31"
seiza catalog object --data objects.bin "openngc:NGC224"
seiza catalog object --data objects.bin "andro" --prefix --limit 10
seiza catalog star --data stars-lite-tycho2.ids.bin "TYC 5949-2777-1" --format json
seiza catalog star --data stars-lite-tycho2.ids.bin "HIP 32349"
seiza catalog star --data stars-lite-tycho2.ids.bin "RR Lyr"
seiza catalog star --data stars-lite-tycho2.ids.bin "STF 2382 AB"
seiza catalog star --data stars-lite-tycho2.ids.bin "RR L" --prefix --limit 10
```

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

## Persistent worker

Applications performing repeated solves can keep a catalog and blind index
open behind a newline-delimited JSON-RPC 2.0 process:

```
seiza worker --data stars-deep-gaia17.bin --index blind-gaia16.idx
```

The same protocol can adapt local image paths to a queued `seiza-server`:

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
also works). Provide a star catalog via the `SEIZA_STAR_DATA`
environment variable or a `stars-*.bin` next to the executable —
`seiza download-data prebuilt` fetches one. Hinted and blind-slot
solving both work; see the repository's `docs/design/astap-mode.md`.

## Datasets

The quickest route is the prebuilt, SHA-256-verified set hosted at
downloads.seiza.fyi (Tycho-2 and Gaia solver tiles, the blind index, the
unified object catalog, minor bodies, the
Tycho/Bright Star/GCVS/WDS/IAU identifier sidecar, and a nightly-refreshed
transient list):

```
seiza download-data prebuilt --output data
seiza download-data prebuilt --output data --file objects.bin --file transients.bin
```

For an interactive selection, run `seiza setup`. This is also the command
offered by the Windows installer. It presents object-only, solver, blind, and
complete presets, then delegates to the same verified prebuilt downloader.

The downloader reads one complete bundle from `/data/v4/manifest.json` and
caches its immutable, content-addressed files by SHA-256 before copying the
requested selection into the same flat local output directory. The shared
platform cache can be overridden with `SEIZA_CACHE_DIR`. It never combines
catalogs from different hosted bundle versions. Previously released `/data/`
and `/data/v2/` paths remain frozen for classic v1 and v0.4.1/v0.5 readers.
The historical `/data/v3/` probe used by v0.4.0 remains reserved and may be
absent; those readers retain their existing fallback behavior.

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
  --version catalog-bundle-v4-YYYY-MM-DD --output next-v4.json
```

The v2 output retains flat keys for released v3-object readers. The v4 output
uses content-addressed artifact keys. Both commands require the resulting
bundle to contain every required catalog.

Normal catalog opens do not perform exhaustive validation. Validate any seiza
star tile, identifier sidecar, blind index, object catalog, or minor-body
catalog explicitly when required; the file format is auto-detected:

```
seiza catalog validate --data stars-lite.ids.bin
seiza catalog validate --data stars-lite.bin
seiza catalog validate --data objects.bin
```

FITS files are read natively (see
[seiza-fits](https://crates.io/crates/seiza-fits)) with bounded-memory
streaming into the final typed pixel buffer, an automatic MTF stretch for u8
detection and previews, a linear normalized representation for f32 detection,
and RA/DEC hints taken from headers.

## License

Apache-2.0
