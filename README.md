# seiza (星座)

Star detection, WCS fitting, and plate solving — hinted and blind — for
astrophotography, in Rust. Built to power object overlays and astrometric
features in [tenrankai](https://github.com/theatrus/tenrankai) and
[PSF Guard](https://github.com/theatrus/psf-guard).

## Quick start

Install from [crates.io](https://crates.io/crates/seiza-cli) (or grab an
RPM/deb from the [releases](https://github.com/theatrus/seiza/releases)),
pull the prebuilt star and object catalogs from our CDN, and solve:

```
cargo install seiza-cli
seiza download-data prebuilt --output data       # SHA-256-verified from downloads.seiza.fyi
seiza solve-blind image.jpg --data data/stars-lite-tycho2.bin --min-scale 0.5 --max-scale 15
seiza solve-blind fine-scale.jpg --data data/stars-deep-gaia17.bin --index data/blind-gaia16.idx --min-scale 0.1 --max-scale 5
seiza solve image.fits --data data/stars-gaia.bin --scale 1.26 --objects data/objects.bin
seiza catalog object --data data/objects.bin "Andromeda Galaxy"
seiza catalog objects --data data/objects.bin --ra 10.6848 --dec 41.2691 --radius 3 --format json
seiza catalog star --data data/stars-lite-tycho2.ids.bin "TYC 5949-2777-1" --format json
seiza catalog star --data data/stars-lite-tycho2.ids.bin "RR Lyr"
```

Applications that perform repeated solves can keep the catalog and blind
index open through the versioned JSON-RPC worker protocol:

```text
seiza worker --data data/stars-deep-gaia17.bin --index data/blind-gaia16.idx
```

The worker reads one JSON request per stdin line, writes one response per
stdout line, accepts FITS and normal raster image paths, and exits cleanly at
EOF. See [the worker protocol](docs/design/worker-protocol.md).

The same worker contract can submit to the existing
[`seiza-server`](https://github.com/theatrus/seiza-server). The local worker
opens and stretches the FITS file, then uploads a lossless 8-bit grayscale PNG
through the server's native queued API instead of sending the original FITS:

```text
seiza worker --server http://solver-host:8080
```

Pass `--server-token` or set `SEIZA_SERVER_TOKEN` when the server requires an
API key. Compact PNG upload is the default; `--server-upload fits` preserves
and streams the original FITS payload when its headers or full bit depth are
desired. Remote solves time out after five minutes by default; use
`--server-timeout SECONDS` to change that deadline.
Local and remote operation use the same JSON-RPC contract.

## Performance

Seiza is built to solve inside an imaging loop. On our Windows 11 test machine
(Intel Core i7-12700H, release builds, no GPU), process startup, image loading,
star detection, catalog access, solving, and result output are all included:

- A real hinted FITS solve usually finishes in **0.2-0.4 seconds**.
- Blind solving with a prebuilt index took **0.53 seconds median** across 13
  real FITS frames ranging from 26 to 61 megapixels.
- Compact u8 detection cut peak detector memory from **1.08 GiB to 543 MiB**
  on a 94 MP JPEG, and from **590 MiB to 179 MiB** on a 61 MP FITS frame.

### Compared with ASTAP

[ASTAP](https://www.hnsky.org/astap.htm) is a mature, highly regarded plate
solver and a serious reference point. There is no universal winner: different
search strategies do better on different fields. In our repeated real-FITS
comparison, using the catalog recommended for each solver:

| Workload | seiza | ASTAP | Result |
|---|---:|---:|---|
| Accurate hint, 26 MP narrow field | 0.25-0.27 s | 0.27-0.29 s | Roughly tied; seiza 9-11% faster |
| Accurate hint, 61 MP wide field | 0.42 s | 0.63 s | seiza 1.5x faster |
| Position blind, 26 MP narrow field | 1.61 s | 31.28 s | seiza 19x faster |
| Position blind, 61 MP wide field | 0.65 s | 1.09 s | seiza 1.7x faster |

The 61 MP position-blind row was rerun after the blind-pipeline improvements
(three runs each: seiza 0.62-0.65 s, ASTAP 1.08-1.16 s).

On a separate set of 25 heavily processed JPEGs, seiza solved all 25 with a
hint and all 25 position-blind. ASTAP solved 13 and 12 respectively. Among
images both programs solved, seiza was **3.5x faster hinted** and **6.5x faster
position-blind** by median wall-time ratio. Seiza also solved all 25 with no
position or scale hint in 0.90 seconds median. This is deliberately unusual
input for a plate solver, so it measures robustness on processed web images
rather than ASTAP's normal FITS workflow.

These are measurements on one system, not universal promises. Runs used the
normal OS file cache and included complete command-line wall time. See the
[FITS comparison](docs/benchmarks/2026-07-astap-comparison.md) and
[blind/detection follow-up](docs/benchmarks/2026-07-blind-pipeline-priorities.md)
for the images, catalogs, repetitions, correctness checks, and caveats.

Current hosted datasets use one complete, versioned
[v2 catalog-bundle manifest](https://downloads.seiza.fyi/data/v2/manifest.json).
The unversioned [legacy manifest](https://downloads.seiza.fyi/data/manifest.json)
remains temporarily for classic v1 clients, but new clients never combine
files from different bundle versions:
`stars-lite-tycho2.bin` (2.5M stars, 25 MB), `stars-gaia.bin` (Gaia DR3
G≤15, 36.7M stars, 367 MB), `stars-deep-gaia17.bin` (Gaia DR3 G≤17,
154.1M stars, 1.54 GB), `blind-gaia16.idx` (the memory-mapped G≤16 blind
pattern index, 1.63 GB), `stars-lite-tycho2.ids.bin` (2.7M numeric identifiers
and 387k names, 100 MB), `objects.bin` (315k objects), `minor-bodies.bin`
(comets and asteroids), and `transients.bin` (active supernovae/novae,
refreshed nightly). `download-data prebuilt` combines the bundle into one
local data directory.
The deep catalog and maintained index
enable blind solving of small, fine-scale fields whose brightest detections
are fainter than the G≤15 catalog's small-field pattern tiers without
rebuilding the whole-sky index for every process.

Applications can install only the catalogs they need without invoking the CLI:

```rust,no_run
// Enable seiza's non-default `downloads` feature first.
let manager = seiza::downloads::CatalogManager::builder().build()?;
let files = manager
    .ensure(&seiza::downloads::CatalogSet::solver_lite()
        .with(seiza::downloads::Dataset::Objects))
    .await?;
let stars = seiza::catalog::TileCatalog::open(
    files.path(seiza::downloads::Dataset::StarsLiteTycho2)?,
)?;
```

[`seiza-download`](seiza-download/README.md) owns the async, verified runtime
bundle cache. [`seiza-sources`](seiza-sources/README.md) separately owns raw
Gaia, VizieR, MPC, OpenNGC, and other catalog-building downloads, keeping those
large and rate-limited workflows out of application integrations.

## Status

Working today:

- **Star detection** — tile-based background/noise estimation (median +
  MAD), sigma thresholding, connected components, flux-weighted sub-pixel
  centroids.
- **WCS** — TAN (gnomonic) projection with a CD matrix: pixel ↔ world
  transforms, scale/footprint helpers.
- **Hinted plate solving** — triangle matching over FOV-sized windows,
  affine candidate voting, iterative least-squares refinement, seeded by an
  approximate center and pixel scale. Solves real telescope images in tens
  of milliseconds with sub-arcsecond RMS.
- **Blind plate solving** — no position hint, only a plausible pixel-scale
  range: a disc-anchored whole-sky 4-star pattern index, hypothesis voting
  with smoothing and non-max suppression, parallel verification through the
  hinted solver. The hosted G≤16 index is versioned, SHA-256 verified, and
  memory-mapped
  (`seiza solve-blind image.jpg --data stars-deep-gaia17.bin --index blind-gaia16.idx --min-scale 0.1 --max-scale 15`).
- **Star catalogs** — memory-mappable tile formats with cone search.
  Use the prebuilt sets from `download-data prebuilt` unless you need a
  custom depth or epoch: building from primary sources stays fully
  supported (Tycho-2; Gaia DR3 via ESA TAP with `--max-mag` and
  `--chunks` for deeper sets; ASTAP `.1476` databases) but the Gaia
  download alone can take many hours against the ESA archive.
  An optional memory-mapped identifier sidecar resolves TYC/HIP/HR/HD/SAO/FK5,
  IAU and Bayer/Flamsteed names, GCVS variables, and WDS double-star
  designations without a network request or plate solve. Its normalized name
  index also supports prefix completion for interactive search.
  Catalog readers keep normal opens non-exhaustive; run
  `seiza catalog validate --data FILE` when a deliberate full integrity scan
  is needed.

```
# custom build from primary sources (the prebuilt sets skip all this)
seiza download-data gaia --output raw/gaia --max-mag 17 --chunks 3072
seiza build-data gaia --input raw/gaia --output stars-deep.bin --max-mag 17
seiza build-blind-index --data stars-deep.bin --output blind-gaia16.idx --index-mag-limit 16
```

- **Object catalogs** — OpenNGC (NGC/IC/Messier), Sharpless, Barnard, UGC,
  LDN, LBN, Cederblad, vdB, PGC, Green's Galactic supernova remnants,
  Wolf-Rayet stars, IAU named and HD stars, and live transient
  (supernova/nova) lists built into a memory-mapped object store. Its embedded
  tile and normalized-name indices page in only relevant records for viewport,
  exact-name/ID, and prefix queries. Query a known sky cone or ordered image
  footprint without plate solving (`seiza catalog objects ...`), resolve a
  name or alias with `seiza catalog object ...`, or query a solved image with
  projected pixel and ellipse geometry
  (`seiza solve ... --objects objects.bin`).
- **FITS** — dependency-free reading with typed headers, exact
  histogram statistics, N.I.N.A.-style MTF autostretch, planar RGB
  (NAXIS3) support, OSC debayering (`BAYERPAT`), and bounded-memory
  streaming into native pixel storage, in the
  [`seiza-fits`](https://crates.io/crates/seiza-fits) crate. FITS files
  plate-solve directly, with RA/DEC hints read from headers.
- **Packages & CI** — crates.io releases, Fedora RPMs and Ubuntu debs on
  GitHub releases, and an integration suite that solves real hosted
  camera frames against known-good solutions on every PR.

Planned (see design notes in the tenrankai repository,
`docs/design/plate-solving.md`): SIP distortion terms, serialized blind
pattern indexes.

## Use with N.I.N.A. (ASTAP-compatible mode)

seiza speaks ASTAP's command-line contract, so N.I.N.A. can use it as
its plate solver with no plugin:

1. Grab the Windows build from the
   [releases](https://github.com/theatrus/seiza/releases) (or
   `cargo install seiza-cli`).
2. Download a star catalog once and tell seiza where it lives — either
   set the `SEIZA_STAR_DATA` environment variable, or simply drop the
   `.bin` next to the executable:

   ```
   seiza download-data prebuilt --output C:\seiza-data --file stars-gaia.bin
   setx SEIZA_STAR_DATA C:\seiza-data\stars-gaia.bin
   ```

3. In N.I.N.A.: **Options → Plate Solving → Plate Solver: ASTAP**, and
   point the ASTAP path at `seiza.exe`. It works in the blind-solver
   slot too.

seiza auto-detects ASTAP-style invocations (`-f image.fits -fov … -ra …
-spd …`), solves hinted or blind accordingly, and writes the `.ini`
result file N.I.N.A. reads — including the full CD matrix, so pixel
scale, rotation, and flip all come through. A copy of the binary
renamed `astap.exe` behaves identically. Details:
[docs/design/astap-mode.md](docs/design/astap-mode.md).

## Layout

- `seiza/` — library crate: `detect`, `wcs`, `catalog`, `objects`, `solve`
- `seiza-fits/` — dependency-free FITS reading, statistics, MTF autostretch
- `seiza-cli/` — the `seiza` command-line tool (FITS files solve directly,
  with RA/DEC hints read from their headers)

## License

Apache-2.0
