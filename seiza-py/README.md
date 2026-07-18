# seiza (Python)

Python bindings for [seiza](https://github.com/theatrus/seiza): star
detection, WCS fitting, and hinted/blind plate solving for astrophotography,
implemented in Rust. Solves typical frames in a fraction of a second.

```
pip install seiza
```

Binary wheels cover Linux (x86_64, aarch64), macOS (universal2), and
Windows (x64); each is a single abi3 wheel for every CPython from 3.9 up.
Type stubs are included, and solving releases the GIL.

## Solve an image

```python
import numpy as np
import seiza

# One-time: download the verified solver catalogs into the shared cache.
paths = seiza.fetch_catalogs()  # lightweight Tycho-2 set
catalog = seiza.StarCatalog.open(paths["stars-lite-tycho2.bin"])

# Detect stars in a 2D float32 (or uint8) luma array.
stars = seiza.detect(image_array)

# Hinted solve: approximate center and pixel scale. sip_order=3 also fits
# SIP distortion polynomials when enough matched stars support them.
solution = seiza.solve(
    stars, catalog, width, height,
    ra=150.1, dec=35.2, scale_arcsec_px=2.5, sip_order=3,
)
print(solution)                 # center, scale, matches, RMS
print(solution.rotation_deg, solution.flipped)
ra, dec = solution.wcs.pixel_to_world(100.0, 200.0)
```

Stars can also be plain `(x, y, flux)` tuples from any other detector — the
solver only needs positions and relative brightness:

```python
solution = seiza.solve([(x1, y1, f1), (x2, y2, f2), ...], catalog, w, h,
                       ra=..., dec=..., scale_arcsec_px=...)
```

## Blind solve

No position hint, only a plausible scale range. Uses the prebuilt whole-sky
pattern index and the deep Gaia catalog:

```python
paths = seiza.fetch_catalogs(["stars-deep-gaia17.bin", "blind-gaia16.idx"])
catalog = seiza.StarCatalog.open(paths["stars-deep-gaia17.bin"])
index = seiza.BlindIndex.open(paths["blind-gaia16.idx"])
solution = seiza.solve_blind(stars, catalog, index, width, height,
                             min_scale_arcsec_px=0.5, max_scale_arcsec_px=15.0)
```

## FITS WCS output

Solutions convert directly to FITS WCS keywords (1-indexed `CRPIX`, TAN or
TAN-SIP projection, CD matrix, and the complete `A_p_q`/`B_p_q`/`AP_p_q`/
`BP_p_q` set when distortion was fitted):

```python
cards = solution.fits_header_cards()   # dict of keyword -> value
text = solution.fits_header_text()     # 80-column cards ending with END
```

The header text form is suitable for header-injection APIs — for example
Siril's `sirilpy` scripting interface (`set_image_header`), which makes a
seiza solve usable from a Siril Python script.

## Notes

- Solving and detection release the GIL; other Python threads keep running.
- Catalog files are memory-mapped and SHA-256 verified at download time;
  `fetch_catalogs` caches under the platform cache directory (override with
  `cache_dir=` or `SEIZA_CACHE_DIR`).
- `seiza.StarCatalog.from_stars([...])` builds a small in-memory catalog for
  tests and synthetic fields.

## License

Apache-2.0
