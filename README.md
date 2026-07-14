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
seiza solve image.fits --data data/stars-gaia.bin --scale 1.26 --objects data/objects.bin
seiza catalog objects --data data/objects.bin --ra 10.6848 --dec 41.2691 --radius 3 --format json
```

Hosted datasets (manifest at
[downloads.seiza.fyi/data/manifest.json](https://downloads.seiza.fyi/data/manifest.json)):
`stars-lite-tycho2.bin` (2.5M stars, 25 MB), `stars-gaia.bin` (Gaia DR3
G≤15, 36.7M stars, 367 MB), `objects.bin` (314k objects), and
`transients.bin` (active supernovae/novae, refreshed nightly).

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
  hinted solver
  (`seiza solve-blind image.jpg --data stars.bin --min-scale 0.5 --max-scale 15`).
- **Star catalogs** — memory-mappable tile formats with cone search.
  Use the prebuilt sets from `download-data prebuilt` unless you need a
  custom depth or epoch: building from primary sources stays fully
  supported (Tycho-2; Gaia DR3 via ESA TAP with `--max-mag` and
  `--chunks` for deeper sets; ASTAP `.1476` databases) but the Gaia
  download alone can take many hours against the ESA archive.

```
# custom build from primary sources (the prebuilt sets skip all this)
seiza download-data gaia --output raw/gaia --max-mag 17 --chunks 3072
seiza build-data gaia --input raw/gaia --output stars-deep.bin --max-mag 17
```

- **Object catalogs** — OpenNGC (NGC/IC/Messier), Sharpless, Barnard, UGC,
  LDN, vdB, PGC, Green's Galactic supernova remnants, Wolf-Rayet stars,
  IAU named and HD stars, and live transient (supernova/nova) lists built
  into a compact object store. Query a known sky cone or ordered image
  footprint without plate solving (`seiza catalog objects ...`), or query a
  solved image with projected pixel and ellipse geometry
  (`seiza solve ... --objects objects.bin`).
- **FITS** — dependency-free reading with typed headers, exact
  histogram statistics, N.I.N.A.-style MTF autostretch, planar RGB
  (NAXIS3) support, and OSC debayering (`BAYERPAT`), in the
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
