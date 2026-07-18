# seiza (星座)

Star detection, WCS fitting, and plate solving — hinted and blind — for
astrophotography, in Rust. Built to power object overlays and astrometric
features in [tenrankai](https://github.com/theatrus/tenrankai) and
[PSF Guard](https://github.com/theatrus/psf-guard).

It is fast. A typical hinted solve of a real telescope frame finishes in
0.2–0.4 seconds, and a blind solve — no position hint at all — takes about
half a second with the prebuilt index. In side-by-side benchmarks seiza
matched or beat ASTAP on every workload we tested, up to 19x faster on blind
solves. The numbers and caveats are in [Performance](#performance).

**Try it without installing anything:** go to [seiza.fyi](https://seiza.fyi),
upload an image, and get a solution and object overlay in your browser. The
site runs [seiza-server](https://github.com/theatrus/seiza-server); the CLI
can submit to the same server with `seiza worker --server`.

## Install

- **Windows** — download the MSI installer from the
  [releases](https://github.com/theatrus/seiza/releases). It puts `seiza` on
  your `PATH` and offers to download catalogs for you when it finishes.
- **Fedora / Ubuntu** — install the RPM or deb from the same releases page.
- **Anywhere with Rust** — `cargo install seiza-cli`.

## Ways to use it

- **As N.I.N.A.'s plate solver** — seiza answers ASTAP's command line, so
  select ASTAP in N.I.N.A. and point it at `seiza.exe`. No plugin needed.
  [Steps below](#use-with-nina-astap-compatible-mode).
- **From your own application** — run `seiza worker` to keep the catalogs
  and blind index open between solves, and send it one JSON request per
  line. It can also forward solves to a seiza-server, local or hosted.
  [Wire protocol](docs/design/worker-protocol.md).
- **From Python** — `pip install seiza`: detection, hinted and blind
  solving, WCS transforms, FITS WCS keyword output, and verified catalog
  downloads, with binary wheels for the common platforms
  ([seiza-py](seiza-py/README.md)).
- **From Rust** — use the crates directly: [`seiza`](seiza/README.md)
  (detection, WCS, solving, catalogs),
  [`seiza-fits`](seiza-fits/README.md) (FITS reading),
  [`seiza-download`](seiza-download/README.md) (catalog download and
  caching), and [`seiza-sources`](seiza-sources/README.md) (raw upstream
  data for custom catalog builds).

## Quick start

Download the ready-made catalogs once, then solve:

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

Not sure which catalogs you need? Run the guided setup — the same one the
Windows installer offers, available on every platform:

```text
seiza setup
```

It walks you through use-case-based choices: lightweight hinted solving,
denser Gaia solving, deep blind solving, or the complete bundle. Every choice
includes object search, Solar System objects, active transients, and at least
one plate-solving catalog. All downloads are versioned and SHA-256 verified.

Set `SEIZA_CATALOG_DIR` to choose the default setup and ASTAP-compatible
catalog directory. The all-users Windows installer sets it system-wide to the
shared `%ProgramData%\Seiza\catalogs` directory.

Solving many images from your own application? Start a worker so the
catalogs and blind index stay open instead of being reloaded for every solve:

```text
seiza worker --data data/stars-deep-gaia17.bin --index data/blind-gaia16.idx
```

Send it one JSON request per line on stdin; it writes one response per line
on stdout, takes FITS or normal image paths, and exits cleanly at EOF. The
full request and response format is in
[the worker protocol](docs/design/worker-protocol.md).

The same worker can send solves to a
[`seiza-server`](https://github.com/theatrus/seiza-server) instead of solving
locally — your own, or the hosted one at [seiza.fyi](https://seiza.fyi). It
converts each FITS to a lossless 8-bit PNG before upload to keep transfers
small:

```text
seiza worker --server http://solver-host:8080
```

If the server needs an API key, pass `--server-token` or set
`SEIZA_SERVER_TOKEN`. To upload the original FITS instead of the PNG (for
example, to preserve headers or full bit depth), pass `--server-upload fits`.
Remote solves give up after five minutes; change that with
`--server-timeout SECONDS`. Local and remote workers speak the same JSON
protocol, so your application code does not change.

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

## Catalogs and data

Most users only need `seiza download-data prebuilt` or `seiza setup` from the
Quick start; this section is the detail behind them — what is in each hosted
bundle and how the compatibility paths work.

V4-capable clients use one complete, versioned
[v4 catalog-bundle manifest](https://downloads.seiza.fyi/data/v4/manifest.json).
Previously released paths remain frozen: the unversioned
[legacy manifest](https://downloads.seiza.fyi/data/manifest.json) serves
classic v1 clients, `/data/v3/` remains the historical v0.4.0 standalone
object-v3 surface, and the
[v2 bundle](https://downloads.seiza.fyi/data/v2/manifest.json) remains for
v0.4.1/v0.5 clients. New clients never combine files from different bundle
versions:
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
  The current [extensible v4 container](docs/design/objects-bin-v4.md) also
  preserves every contributing upstream record, typed relations, preferred
  facet selections, source-qualified geometry (including hand-drawn OpenNGC
  outlines), pinned build provenance, and externally curated corrections;
  `seiza catalog object --all-sources` audits all of it. Earlier `SEIZAOB1`
  and `SEIZAOB3` files remain readable.
- **FITS** — dependency-free reading with typed headers, exact
  histogram statistics, N.I.N.A.-style MTF autostretch, planar RGB
  (NAXIS3) support, OSC debayering (`BAYERPAT`), and bounded-memory
  streaming into native pixel storage, in the
  [`seiza-fits`](https://crates.io/crates/seiza-fits) crate. FITS files
  plate-solve directly, with RA/DEC hints read from headers.
- **Packages & CI** — crates.io releases, a guided
  [Windows MSI installer](packaging/windows/README.md), Fedora RPMs and
  Ubuntu debs on GitHub releases, and an integration suite that solves real
  hosted camera frames against known-good solutions on every PR.

Both solvers can fit SIP distortion polynomials (orders 2-5, forward and
inverse) on the accepted solution with `--sip-order`; the linear solution is
kept whenever the polynomial does not improve the residual.

## Use with N.I.N.A. (ASTAP-compatible mode)

seiza speaks ASTAP's command-line contract, so N.I.N.A. can use it as
its plate solver with no plugin:

1. Grab the Windows MSI from the
   [releases](https://github.com/theatrus/seiza/releases) (or use the portable
   ZIP or `cargo install seiza-cli`). The installer offers to run
   `seiza setup` after installation.
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
- `seiza-cli/` — the `seiza` command-line tool: solving, ASTAP mode, the
  JSON-RPC worker, guided `seiza setup`, and dataset building
- `seiza-download/` — async, verified runtime catalog-bundle cache
- `seiza-sources/` — raw upstream catalog acquisition for custom builds
- `seiza-py/` — Python bindings (`pip install seiza`), outside the cargo
  workspace so workspace builds never need libpython
- `packaging/windows/` — the WiX MSI installer
- `docs/` — design notes and benchmark reports

## License

Apache-2.0
