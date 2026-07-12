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
- **Near-field plate solving** — triangle matching over FOV-sized windows,
  affine candidate voting, iterative least-squares refinement. Seeded by an
  approximate center and pixel scale (blind solving is out of scope for
  now). Solves real telescope images in tens of milliseconds with
  sub-arcsecond RMS.
- **Star catalogs** — the `SEIZAST1` tile format with cone search;
  builders for Tycho-2 (`lite`, ~2.5M stars / 25 MB) and ASTAP `.1476`
  databases such as the Gaia DR3 D80 (~241M stars / 2.4 GB).

```
seiza download-data tycho2 --output raw/tycho2
seiza build-data tycho2 --input raw/tycho2 --output stars-lite.bin
seiza solve image.jpg --data stars-lite.bin --ra 324.8 --dec 57.5 --scale 2.8
```

- **Object catalogs** — OpenNGC (NGC/IC/Messier), Sharpless, Barnard, and
  IAU named stars built into a compact object store; solved images can be
  queried for the objects in their footprint with full ellipse geometry
  (`seiza solve ... --objects objects.bin`).

Planned (see design notes in the tenrankai repository,
`docs/design/plate-solving.md`): Gaia TAP download, SIP distortion terms,
blind solving.

## Layout

- `seiza/` — library crate: `detect`, `wcs`, `catalog`, `solve`
- `seiza-cli/` — the `seiza` command-line tool

## License

Apache-2.0
