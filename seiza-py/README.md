# seiza (Python)

Python bindings for [seiza](https://github.com/theatrus/seiza): star detection,
WCS fitting, hinted/blind plate solving, satellite prediction, calibration,
deconvolution, and batch/live image stacking for astrophotography, implemented
in Rust.

```
pip install seiza
```

Binary wheels cover Linux (x86_64, aarch64), macOS (universal2), and
Windows (x64); each is a single abi3 wheel for every CPython from 3.9 up.
Type stubs are included, and computational image operations release the GIL.

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

`open` takes a file, a directory (the right catalog inside is picked — the
deepest star catalog wins), or nothing at all. With no argument the standard
places are searched: `SEIZA_STAR_DATA` / `SEIZA_BLIND_INDEX`, files next to
the program, and the `seiza setup` directories (`SEIZA_CATALOG_DIR`). These
are the same rules as the CLI's `--data`:

```python
catalog = seiza.StarCatalog.open("data")   # directory
catalog = seiza.StarCatalog.open()         # after seiza setup
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

For faint fields, the optional `stars-deep-gaia20.bin` catalog reaches Gaia
G≤20 (about 9 GB). It is intentionally not included in `fetch_catalogs("all")`,
so request it explicitly with the same G≤16 blind index:

```python
paths = seiza.fetch_catalogs(["stars-deep-gaia20.bin", "blind-gaia16.idx"])
catalog = seiza.StarCatalog.open(paths["stars-deep-gaia20.bin"])
index = seiza.BlindIndex.open(paths["blind-gaia16.idx"])
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

## Background extraction

Fit a compact background model to a C-contiguous mono `(H, W)` or RGB
`(H, W, 3)` linear `float32` array, inspect it, and then correct the image:

```python
model = seiza.fit_background(stack, degree=2)
print(model.diagnostics)

corrected = model.correct(stack)                 # additive subtraction
illumination_corrected = model.correct(stack, mode="divide")
background = model.render()                      # explicit full-size model
```

Fitting uses deterministic low-noise sample windows, robust sample rejection,
and independent per-channel polynomial coefficients. `model.correct()`
allocates only the corrected array; a full-size background exists only after
`render()`. Pass a boolean `(H, W)` `mask` to exclude extended objects, dark
clouds, registration borders, or source masks:

```python
model = seiza.fit_background(stack, mask=structure_mask,
                             degree=1, samples_per_axis=12,
                             sample_radius=20)
for x, y, values, dispersion, weight, status in model.samples():
    print(x, y, values, status)
```

The output remains linear and may retain negative or greater-than-one values.
Background extraction is not display stretching or color calibration.

## Light deconvolution

Apply the same conservative linear-image restoration as the Rust crate and
CLI to a C-contiguous mono `(H, W)` or RGB `(H, W, 3)` `float32` array:

```python
restored = seiza.deconvolve(stack, psf_fwhm=3.1)
```

`psf_fwhm` is a measured unsaturated-star FWHM in pixels. The defaults use four
damped Richardson-Lucy iterations and blend 35% of the estimate into the input.
The returned array remains linear `float32`; no clipping or display stretch is
applied. The operation releases the GIL. Inspect identical stretches for noise,
rings, saturated-star failures, and field-dependent PSF mismatch before using a
stronger `iterations` or `amount`.

## Image stacking

The wheel includes the same linear calibration, registration, normalization,
and online rejection engine as the Rust crate and CLI. Batch stacking accepts
FITS paths and writes an unstretched linear `float32` FITS result:

```python
options = seiza.StackOptions(
    normalization="local",
    local_tile_size=256,
    maximum_drift_pixels=256.0,
    maximum_drift_fraction=0.15,
)
result = seiza.stack_fits(
    sorted(light_paths),
    "stack.fits",
    options=options,
    bias="master-bias.fits",
    dark="master-dark.fits",
    flat="master-flat.fits",
)
for frame in result.frames:
    print(frame.source, frame.accepted, frame.reason, frame.registration_rms_pixels)
```

For live integration, construct from a FITS path or a C-contiguous mono/HWC
RGB NumPy `float32` array. `push()` accepts already-linear, calibrated arrays;
`push_fits()` performs the configured FITS calibration path. Both return a
typed admission decision, and a rejected frame never mutates the accumulator:

```python
stacker = seiza.LiveStacker.from_array(reference, options=options)
for frame in incoming_arrays:
    disposition = stacker.push(frame)
    if not disposition.accepted:
        print(disposition.reason)

preview_state = stacker.snapshot()  # immutable copy
linear_mean = preview_state.image
coverage = preview_state.coverage
final = stacker.finish("stack.fits")  # consumes the live accumulator
```

Frames taken after a German-equatorial-mount meridian flip are handled by
default. `maximum_rotation_degrees` limits deviation from either the reference
orientation or its 180-degree counterpart; frame diagnostics still report the
full fitted rotation.

Snapshot array properties are copies, so Python cannot mutate live Rust state.
All expensive FITS, calibration, registration, and integration work releases
the GIL.

### Color from mono stacks

Aligned mono `float32` arrays can be combined without writing intermediate
files. Outputs have shape `(height, width, 3)`:

```python
rgb = seiza.combine_rgb(red, green, blue)
lrgb = seiza.combine_lrgb(luminance, red, green, blue,
                          luminance_weight=1.0)

sho = seiza.combine_narrowband(ha, oiii, sii, palette="sho")
hoo = seiza.combine_narrowband(ha, oiii, palette="hoo")
foraxx = seiza.combine_narrowband(ha, oiii, sii, palette="foraxx-sho")
```

The default percentile normalization is a quick-look channel match. Pass
`normalization="none"` for already matched inputs. Foraxx inputs must also
already lie in `[0, 1]` in that mode; keep percentile normalization for
sensor-unit arrays. RGB, LRGB, the six direct S/H/O permutations, and HOO are
linear-light. Foraxx-SHO/HOO use a stretched working copy as required by the
published dynamic formula, so those returned arrays are display-referred.
Composition releases the GIL.

### Parameterized display stretching

`seiza.stretch` applies the shared `seiza-stretch` model to mono `(H, W)` or
RGB `(H, W, 3)` `float32` arrays and returns display-referred `float32` without
eight-bit quantization:

```python
preview = seiza.stretch(linear, model="percentile-asinh",
                        black_percentile=0.01,
                        white_percentile=0.995, strength=10)
preview = seiza.stretch(linear_rgb, model="auto-mtf",
                        target_median=0.2, shadows_clip=-2.8,
                        color_strategy="luminance-preserving")
preview = seiza.stretch(linear, model="ghs", stretch_factor=4,
                        local_intensity=-1, symmetry_point=0.35,
                        protect_shadows=0.1, protect_highlights=0.8)
```

Available models are `identity`, `linear`, `asinh`, `percentile-asinh`, `mtf`,
manual `ghs`, and `auto-mtf`; color strategies are `linked`, `unlinked`, and
`luminance-preserving`. Analysis and application release the GIL.

Calibration masters use the same bounded-memory two-pass builder:

```python
bias = seiza.build_bias(bias_paths, "master-bias.fits")
dark = seiza.build_dark(dark_paths, "master-dark.fits",
                        bias="master-bias.fits")
flat = seiza.build_flat(flat_paths, "master-flat.fits",
                        bias="master-bias.fits",
                        dark_flat="master-dark-flat.fits")
```

## Predicted satellite tracks

After a solve, predict which satellites crossed the image while the shutter
was open. Predictions come from orbital elements — they are never pixel
detections. The exposure must be one continuous shutter-open interval (not a
stack's total integration) and needs an observer location:

```python
sats = seiza.SatelliteCatalog.fetch_celestrak()   # cached; ~2h refresh floor
# or offline / historical: seiza.SatelliteCatalog.open("elements.json")

result = sats.tracks_in_footprint(
    solution.wcs, width, height,
    start="2026-07-19T06:12:00Z",     # Unix seconds, RFC 3339, or tz-aware datetime
    duration_s=120.0,
    latitude=42.466, longitude=-71.1516, altitude_m=150.0,
)
for track in result.tracks:           # highest elevation first
    print(track.label, track.max_elevation_deg, track.clipped_segments)
```

Element records older than seven days are reported in
`result.stale_elements` and skipped rather than silently extrapolated
(`max_element_age_s=None` overrides). CelesTrak rate-limits repeated
downloads: keep reusing one cache directory, and check `sats.cache_state`
and `sats.warning` after `fetch_celestrak()`.

## Notes

- Solving and detection release the GIL; other Python threads keep running.
- Catalog files are memory-mapped and SHA-256 verified at download time;
  `fetch_catalogs` caches under the platform cache directory (override with
  `cache_dir=` or `SEIZA_CACHE_DIR`).
- `seiza.StarCatalog.from_stars([...])` builds a small in-memory catalog for
  tests and synthetic fields.

## License

Apache-2.0
