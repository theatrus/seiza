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
Input arrays are read in place, without a copy, while the GIL is released:
do not mutate an array from another thread until the call returns.

## Solve an image

```python
import numpy as np
import seiza

# One-time: download the verified solver catalogs into the shared cache.
paths = seiza.fetch_catalogs()  # Tycho-2 solver + objects, Solar System, transients
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

## Measure stars and sensor tilt

The measurement detector is separate from the fast alignment detector above.
It reports HFR, FWHM, SNR, flux, and optional Gaussian/Moffat PSF fits from a
mono `uint16` frame. The returned coordinates and radii are always in input
pixels, including when detection binning is enabled:

```python
measured = seiza.detect_measured_stars(
    image_u16,
    focal_length_mm=550.0,
    pixel_size_um=3.76,
    psf_type="moffat4",
)
cells, tilt = seiza.tilt_analysis(measured)
triangle = seiza.triangle_tilt_analysis(measured, angle_degrees=0)
print(len(measured.stars), measured.average_hfr, measured.average_fwhm)
print(tilt.tilt_percent, tilt.curvature_percent, tilt.worst_corner)
print(triangle.ready, triangle.tilt_percent, triangle.worst_sector)
```

The nine cells cover a 3×3 sensor grid. Fitted star `theta` and cell
`mean_theta` values are ellipse major-axis orientations in radians over
`[0, π)`; `theta_coherence` describes how consistently stars in that cell
share the direction. Tilt and curvature need measurements in all four corner
cells, and curvature also needs the center cell.

`tilt_analysis` supplies the 3×3 measurements for a parallelogram diagram.
`triangle_tilt_analysis` reuses the same detected stars and groups an inscribed
circular annulus around three adjustment-screw axes. Its angle uses image
coordinates: `0` points to the top and positive values turn clockwise. Sector
IDs are 1-based, and their axes are the normalized input angle plus 0°, 120°,
and 240°. Per-sector medians remain available for sparse data, but `ready`,
`tilt_percent`, and best/worst sector enforce the native minimum of three
stars per sector. `overall_median_hfr` is the median of every selected annular
star, not the median of the sector medians.

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
model = seiza.fit_background(stack, model="automatic", degree=2)
print(model.diagnostics)

corrected = model.correct(stack)                 # additive subtraction
illumination_corrected = model.correct(stack, mode="divide")
partial = model.correct(stack, strength=0.6)      # tune without refitting
background = model.render()                      # explicit full-size model
```

Fitting uses deterministic low-noise sample windows, robust sample rejection,
and independent per-channel surfaces. Automatic mode chooses among polynomial
degrees from held-out sample errors. Use `model="radial_basis"` with
`rbf_smoothing`, or set `allow_radial_basis=True` in automatic mode, for an
irregular thin-plate model. RBF is explicit because background samples can
share real extended emission. `model.correct()` allocates only the corrected
array; a full-size background exists only after `render()`. Pass a boolean
`(H, W)` `mask` to exclude extended objects, dark clouds, registration borders,
or source masks:

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
Pass `masked=True` for registered images whose missing border samples are
`NaN`: the mask stays in the output and does not darken nearby data. Without
it, non-finite samples raise `seiza.EngineError`.
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
RGB NumPy `float32` array. An array-based stacker accepts only already-linear,
calibrated arrays through `push()`. A path-based stacker can use `push_fits()`
for its configured calibration path or `push()` for caller-prepared arrays. A
stacker keeps that input mode after checkpointing. Both methods return a typed
admission decision, and a rejected frame never mutates the accumulator:

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

When the paths are all known — a finished session rather than a live one —
`push_fits_pipelined()` prepares several frames at once while integrating in
the order given, so the result is identical to pushing them one at a time. It
measured 2.0x faster on local storage and 3.0x with a 300ms read latency:

```python
dispositions, report = stacker.push_fits_pipelined(paths)
print(report)  # PipelineReport(integrated=..., rejected=..., failed=...)
```

A path that cannot be read, or that repeats one already stacked, comes back in
place with `accepted` false and a reason rather than raising, so one bad path in
a night's listing does not lose the rest — check `report.failed` rather than
reading a clean return as success. Pass `workers=` when the frames arrive over a
network, since the library cannot tell a network mount from a local disk, and
`normalized_full_scale=65535.0` when the set mixes PixInsight XISF frames with
16-bit camera data.


Checkpointing is non-consuming. Reopening preserves the original registration
reference, calibration and options, online rejection statistics, coverage, and
the FITS/XISF source ledger:

```python
stacker.save_context("m31.seiza-stack")

stacker = seiza.LiveStacker.open_context("m31.seiza-stack")
decision = stacker.push(next_array)
stacker.save_context("m31.seiza-stack")
```

The context file is versioned, compressed, checksummed, and atomically
replaced. It is processing state for resumption; `finish("stack.fits")` remains
the interoperable final image output.

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
super_lrgb = seiza.combine_lrgb(luminance, red, green, blue,
                                luminance_mode="super")
super_rgb = seiza.combine_rgb(red, green, blue, luminance_mode="super")

sho = seiza.combine_narrowband(ha, oiii, sii, palette="sho")
hoo = seiza.combine_narrowband(ha, oiii, palette="hoo")
foraxx = seiza.combine_narrowband(ha, oiii, sii, palette="foraxx-sho")
```

Pass `crop="bounds"` or `crop="inscribed"` to trim the blank edges that
registering one channel onto another leaves behind. `bounds` keeps the box
every channel covers; `inscribed` keeps the largest rectangle they all cover in
full, so nothing stays `NaN`. `seiza.crop_report` measures the same thing
without composing, and names any channel whose coverage sits far from the
others:

```python
report = seiza.crop_report({"red": red, "green": green, "blue": blue})
x, y, width, height = report["region"]
stray = [c["name"] for c in report["channels"] if c["off_center"]]
```

The default percentile normalization is a quick-look channel match. Pass
`normalization="none"` for already matched inputs. Foraxx inputs must also
already lie in `[0, 1]` in that mode; keep percentile normalization for
sensor-unit arrays. RGB, LRGB, additive super-LRGB (`L + R + G + B`),
synthetic super-RGB (`R + G + B`), the six direct S/H/O permutations, and HOO
are linear-light. Super-luminance output can exceed one.
Foraxx-SHO/HOO use a stretched working copy as required by the
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

Each builder reads every input twice, so a master over dozens of frames runs
for minutes. Pass `cancel=` a predicate to stop one early — it is called once
per input and raises `StackError` when it returns true. Ctrl-C is honoured at
the same points, without a predicate.

```python
stop = threading.Event()
dark = seiza.build_dark(dark_paths, "master-dark.fits", cancel=stop.is_set)
```

## Image processing primitives

OpenCV-compatible building blocks from `seiza-imgproc`, for detection
pipelines that need OpenCV's exact numerics without the dependency. All
operate on 2D single-channel arrays:

```python
import numpy as np
import seiza

image = np.asarray(..., dtype=np.uint8)      # (height, width)

blurred = seiza.gaussian_blur(image, sigma=1.4)      # uint8 or float32
denoised = seiza.median_blur3(image)
edges = seiza.canny(blurred, low=10, high=80)
binary = seiza.otsu_binary(image)
grown = seiza.dilate(binary, shape="rect", ksize=3)

contours = seiza.find_contours(grown)        # list of (n, 2) int32 arrays
areas = [seiza.contour_area(c) for c in contours]

# Edge-aware smoothing and multi-scale structure removal (float inputs).
flat = seiza.dt_filter(guide, src, sigma_spatial=10.0, sigma_color=30.0)
stars_plus_noise = seiza.remove_structures(image.astype(np.float64), layers=4)
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

## Working on the bindings

This directory is its own cargo workspace, so `cargo fmt --all` and
`cargo clippy --workspace` at the repository root never reach it. Run its
checks from here, which is what CI does:

```bash
cd seiza-py
cargo fmt --check
cargo clippy --all-targets -- -D warnings
maturin develop && python -m pytest tests/ -q
```

## Notes

- Solving and detection release the GIL; other Python threads keep running.
- Catalog files are memory-mapped and SHA-256 verified at download time;
  `fetch_catalogs` caches under the platform cache directory (override with
  `cache_dir=` or `SEIZA_CACHE_DIR`).
- `seiza.StarCatalog.from_stars([...])` builds a small in-memory catalog for
  tests and synthetic fields.

## License

Apache-2.0
