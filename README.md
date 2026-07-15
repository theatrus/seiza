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
stdout line, accepts FITS paths, and exits cleanly at EOF. See
[the worker protocol](docs/design/worker-protocol.md).

The same worker contract can submit to the existing
[`seiza-server`](https://github.com/theatrus/seiza-server). The local worker
opens and stretches the FITS file, then uploads a lossless 8-bit grayscale PNG
through the server's native queued API instead of sending the original FITS:

```text
seiza worker --server http://solver-host:8080
```

Pass `--server-token` or set `SEIZA_SERVER_TOKEN` when the server requires an
API key. Compact PNG upload is the default; `--server-upload fits` preserves
the original FITS payload when its headers or full bit depth are desired.
Local and remote operation use the same JSON-RPC contract.

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

## How fast?

Measured on a 16-core desktop, no GPU, everything from a cold start:

- A 24 MP wide field **blind-solves in under 2 seconds** against a
  2.5M-star Tycho-2 catalog — and that includes building the entire
  2M-pattern whole-sky index from scratch (1.2 s) before searching it
  (0.4 s). A 36.7M-star Gaia catalog builds a 6.9M-pattern index in ~5 s
  and blind-solves sub-degree galaxy fields in ~3 s.
- A **61-megapixel** raw FITS frame goes from file open to hinted
  solution in **0.7 s** (load, autostretch, star detection, solve).
- Hinted solving itself runs in tens of milliseconds; typical RMS on real
  frames is 0.2–0.5″ at fine pixel scales.
- A 26 MP camera sub loads in ~75 ms with exact median/MAD statistics in
  8 ms (single histogram pass, no sort) and an MTF autostretch in 25 ms.

The tricks: star unit vectors so every radius test is a dot product
(no per-pair trigonometry), boundary-aware descriptor hashing (1–4 probes
per quad instead of 3⁵), a frozen sorted-array index with branchless
binary search, uniform-grid matching, rayon across cores, and
SIMD kernels with runtime AVX2 dispatch in released binaries.

## Layout

- `seiza/` — library crate: `detect`, `wcs`, `catalog`, `objects`, `solve`
- `seiza-fits/` — dependency-free FITS reading, statistics, MTF autostretch
- `seiza-cli/` — the `seiza` command-line tool (FITS files solve directly,
  with RA/DEC hints read from their headers)

## License

Apache-2.0
