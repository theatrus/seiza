# seiza (星座)

Star detection, WCS fitting, and plate solving — hinted and blind — for
astrophotography, in Rust.

- **Star detection** — tile-based background/noise estimation (median +
  MAD), sigma thresholding, connected components, flux-weighted sub-pixel
  centroids.
- **WCS** — TAN (gnomonic) projection with a CD matrix: pixel ↔ world
  transforms, scale/footprint helpers.
- **Hinted plate solving** — triangle matching over FOV-sized windows,
  affine candidate voting, iterative least-squares refinement, seeded by an
  approximate center and pixel scale. Solves real telescope images in tens
  of milliseconds with sub-arcsecond RMS.
- **Blind plate solving** — no position needed, only a plausible
  pixel-scale range. A disc-anchored 4-star pattern index over the whole
  sky is matched against quads of the brightest detections; hypotheses are
  voted on, smoothed, and verified in parallel by the hinted solver.
  Typically 1–2 seconds per image on wide fields.
- **Star catalogs** — compact memory-mappable tile formats with cone
  search; builders for Tycho-2, Gaia DR3 (via TAP), and ASTAP databases.
- **Object catalogs** — NGC/IC/Messier, Sharpless, Barnard, UGC, LDN, vdB,
  PGC, named/HD stars, and live transient (supernova/nova) lists built into
  a compact store; solved images can be queried for the objects in their
  footprint with full ellipse geometry.

See the [`seiza-cli`](https://crates.io/crates/seiza-cli) crate for the
command-line tool, and [`seiza-fits`](https://crates.io/crates/seiza-fits)
for dependency-free FITS reading and autostretch.

## License

Apache-2.0
