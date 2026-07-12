# seiza (星座)

Star detection, WCS fitting, and near-field plate solving for
astrophotography, in Rust. Built to power object overlays and astrometric
features in [tenrankai](https://github.com/theatrus/tenrankai) and
[PSF Guard](https://github.com/theatrus/psf-guard).

## Status

Early scaffold. Working today:

- **Star detection** — tile-based background/noise estimation (median +
  MAD), sigma thresholding, connected components, flux-weighted sub-pixel
  centroids.
- **WCS** — TAN (gnomonic) projection with a CD matrix: pixel ↔ world
  transforms, scale/footprint helpers.
- `seiza detect <image> [--annotate out.png]` — detect stars in a PNG/JPEG/
  TIFF and optionally write an annotated copy.

Planned (see design notes in the tenrankai repository,
`docs/design/plate-solving.md`):

- Near-field solver: triangle/quad matching + RANSAC + least-squares CD fit,
  seeded by an approximate center and pixel scale.
- Catalog data: HEALPix-tiled Gaia DR3 subsets (built from primary sources)
  with `seiza build-data` / `seiza download-data`.
- Object overlay support data: OpenNGC, Sharpless, Barnard, named stars.

## Layout

- `seiza/` — library crate: `detect`, `wcs`, `catalog`, `solve`
- `seiza-cli/` — the `seiza` command-line tool

## License

Apache-2.0
