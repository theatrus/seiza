# seiza (星座)

Star detection, WCS fitting, and near-field plate solving for
astrophotography, in Rust. Built to power object overlays and astrometric
features in [tenrankai](https://github.com/theatrus/tenrankai) and
[PSF Guard](https://github.com/theatrus/psf-guard).

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
- **Star catalogs** — memory-mappable tile formats with cone search;
  builders for Tycho-2 (`lite`, ~2.5M stars / 25 MB), Gaia DR3 via TAP
  download, and ASTAP `.1476` databases such as the Gaia D80
  (~241M stars / 2.4 GB).

```
seiza download-data tycho2 --output raw/tycho2
seiza build-data tycho2 --input raw/tycho2 --output stars-lite.bin
seiza solve image.jpg --data stars-lite.bin --ra 324.8 --dec 57.5 --scale 2.8
```

- **Object catalogs** — OpenNGC (NGC/IC/Messier), Sharpless, Barnard, UGC,
  LDN, vdB, PGC, IAU named and HD stars, and live transient
  (supernova/nova) lists built into a compact object store; solved images
  can be queried for the objects in their footprint with full ellipse
  geometry (`seiza solve ... --objects objects.bin`).

Planned (see design notes in the tenrankai repository,
`docs/design/plate-solving.md`): SIP distortion terms, serialized blind
pattern indexes.

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
