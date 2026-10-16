# seiza (Python)

Python bindings for [seiza](https://github.com/theatrus/seiza): star detection,
WCS fitting, hinted/blind plate solving, satellite prediction, calibration,
deconvolution, and batch/live image stacking for astrophotography, implemented
in Rust.

```
pip install seiza
```

Binary wheels cover Linux (x86_64, aarch64), macOS (universal2), and
Windows (x64); each is a single abi3 wheel for every CPython from 3.11 up.
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

An image that already knows where it points can say so.
`read_solve_prior(path)` reads a FITS or XISF file's own WCS (including a
PixInsight solution), its RA/DEC or OBJCTRA/OBJCTDEC, and its pixel scale
from PIXSCALE or pixel size over focal length, without decoding the pixels.
`solve_with_prior` then checks that WCS, else searches around the position,
else solves blind (given an `index`), and when nothing solves returns the
file's own WCS with `source == "header-unchecked"`:

```python
prior = seiza.read_solve_prior("processed.xisf")
solution = seiza.solve_with_prior(stars, catalog, width, height, prior, index=index)
print(solution.source)   # header-checked, header-hint, blind or header-unchecked
```

A field that will not solve can search for a minute or more. Both `solve` and
`solve_blind` take `time_limit=` in seconds, which raises `TimeoutError`, and
`cancel=`, a callable that returns true to stop (asked about every 50 ms),
which raises `SolveError`. Ctrl-C stops a solve either way, and an exception
from `cancel` reaches you as itself:

```python
solution = seiza.solve_blind(stars, catalog, index, width, height,
                             time_limit=30.0, cancel=stop_requested)
```

`solution.false_match_probability` is the chance that a wrong field would
match as well, counting every candidate the search tried. A true field scores
far below one in a billion; a blind solve above that is refused, and a hinted
one above one in a thousand.

For faint fields, the optional `stars-deep-gaia20.bin` catalog reaches Gaia
G≤20 (about 9 GB). It is intentionally not included in `fetch_catalogs("all")`,
so request it explicitly with the same G≤16 blind index:

```python
paths = seiza.fetch_catalogs(["stars-deep-gaia20.bin", "blind-gaia16.idx"])
catalog = seiza.StarCatalog.open(paths["stars-deep-gaia20.bin"])
index = seiza.BlindIndex.open(paths["blind-gaia16.idx"])
```

For phone and camera photos, `read_photo_metadata` reads the EXIF capture
time, GPS position and focal length. Given the upright image's size (after
`PIL.ImageOps.exif_transpose`), it also returns `scale_ranges` to try in
order, the first from the focal length and the last a wide fallback:

```python
from PIL import Image, ImageOps

image = ImageOps.exif_transpose(Image.open("phone.jpg"))
info = seiza.read_photo_metadata("phone.jpg", *image.size)
for min_scale, max_scale in info["scale_ranges"]:
    try:
        solution = seiza.solve_blind(stars, catalog, index, *image.size,
                                     min_scale_arcsec_px=min_scale,
                                     max_scale_arcsec_px=max_scale)
        break
    except seiza.SolveError:
        continue
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

Fitted one channel at a time, the channels of a colour image can each pick
a different automatic surface, and the mismatch shows as a colour cast toward
the edges. `select_shared_model` chooses one surface from the held-out errors
of every fit, and returns the `fit_background` keyword arguments to fit each
channel again with, or `None` when the fits share no choice:

```python
models = [seiza.fit_background(channel, model="automatic") for channel in channels]
shared = seiza.select_shared_model(models)       # the same automatic settings
if shared is not None:
    models = [seiza.fit_background(channel, **shared) for channel in channels]
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

## Photometric colour calibration

`calibrate_color` fits per-channel gains for a linear RGB image from Gaia DR3
star colours: each star's instrumental colour is fitted against Gaia BP−RP,
and the gains render a star of the white reference's colour (the Sun's by
default) neutral, with the sky neutralized too.

```python
import seiza
from astropy.io import fits

header = fits.getheader("stack.fits")
wcs = seiza.Wcs.from_header(header)
gaia = seiza.gaia_photometry_cone(56.75, 24.10, radius_deg=4.8)  # ESA archive; keep it
# Or offline, from the hosted catalog (about 460 MB, fetched once):
path = seiza.fetch_catalogs(["stars-gaia-photometry.bin"])["stars-gaia-photometry.bin"]
gaia = seiza.gaia_photometry_catalog_cone(path, 56.75, 24.10, radius_deg=4.8)
calibration = seiza.calibrate_color(rgb, wcs, gaia)               # rgb: (h, w, 3) float32
print(calibration.gains, calibration.red_fit, calibration.blue_fit)
calibrated = calibration.apply(rgb)
```

`fit_bp_rp_range` (default `(0.0, 1.8)`) sets the colour range of the stars
the fits use, `rejection_sigma` (default 3) their outlier rejection,
`minimum_stars` (default 20) the fewest stars a fit accepts, and
`calibrator_max_g` the faintest Gaia G a calibrator may have (by default two
magnitudes brighter than the faintest row given). `calibration.measurements`
lists every star measured, with its position, `g`, `bp_rp`, R, G, B `flux`,
and `used`, whether the red and the blue fit kept it.

## Image stacking

The wheel includes the same linear calibration, registration, normalization,
and online rejection engine as the Rust crate and CLI. Batch stacking accepts
FITS paths and writes an unstretched linear `float32` FITS result:

```python
options = seiza.StackOptions(
    normalization="local-background",
    local_tile_size=256,
    demosaic="vng",
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

`normalization` is `"none"`, `"global"` (the default), `"local"`, or
`"local-background"`, which keeps one gain per channel and matches each tile's
background to the reference's so frame edges leave no seams. `demosaic` chooses
how one-shot-colour frames are demosaiced: `"vng"` (the default) keeps each
star's colour across its profile, `"mhc"` gives slightly sharper stars but rings
around small ones, and `"bilinear"` is fastest. `registration_model` is
`"similarity"` (the default here), `"affine"`, or `"quadratic"`, which fits a
polynomial to follow lens distortion, for example across a meridian flip; the
CLI defaults to `"quadratic"`. `weighting` is `"equal"` (the default) or
`"inverse-noise-variance"`, which weights each frame by the reference frame's
noise over its own, squared. The CLI's other stacking options (interpolation,
Bayer drizzle) are not yet exposed here.

Under noise weighting, each accepted frame's `noise` and `weight` list one
value per channel, and `LiveStacker.reference_noise` gives the reference
frame's noise, the unit of the weights; under equal weighting they are empty.
`seiza.frame_noise(image)` measures any linear array the same way. Each
accepted frame's `mapping`, and `LiveStacker.reference_mapping` for the
reference, is a dict of its registration and normalization, to keep beside
the frame.

`LiveStacker.open_prepared_fits(path)` opens a stack from a frame already
calibrated and debayered, keeping its FITS header cards, so you can correct a
frame between calibration and registration; push later frames as arrays.

The first light is the reference unless `reference="auto"`, which scores every
light and takes the best, as the CLI's `--reference auto` does. The result's
`reference_light` names the light used. `seiza.choose_reference(paths)` gives
the same choice on its own: its `reference_path`, a `ranking` of the scored
lights' indices, best first, and each light's `scores`.

To stack each filter onto the same grid, pass every run the same
`reference_image`, as the CLI's `--reference-image` does. It sets only the
alignment grid and WCS; the reference light still sets normalization and noise
weighting, and every light listed is stacked, that one included. With
`reference="auto"`, if the best light does not align to the reference image,
the next-best lights are tried in turn, up to five in all. The reference image
is not calibrated and adds no pixels unless it is also listed as a light. It
must have the lights' channel count:

```python
for name in ["red", "green", "blue"]:
    seiza.stack_fits(lights[name], f"{name}.fits", reference_image="common.fits")
```

`LiveStacker.open_fits_with_reference_image(reference_image, channel_reference)`
does the same for a live stack. It starts with no frames, so push the channel
reference like every other light; `reference_integrated` is then false.
`LiveStacker.open_fits_with_reference_image_candidates(reference_image,
channel_references)` takes several lights, best first, and uses the first that
aligns. It returns the stack and a selection naming the light used and why
each earlier one was skipped:

```python
choice = seiza.choose_reference(red_lights)
best = [red_lights[index] for index in choice.ranking[:5]]
stacker, selection = seiza.LiveStacker.open_fits_with_reference_image_candidates(
    "common.fits", best
)
for index, path, reason in selection.skipped:
    print(f"{path} did not align: {reason}")
stacker.push_fits(selection.path)
```

`LiveStacker.open_fits_expanding(paths)` grows the canvas past the anchor to
hold every connected, overlapping light, as the CLI's `--reference expand`
does: mosaic panels, including ones that connect only through other panels,
or a dithered stack whose edges should be kept. The first path anchors the WCS
and photometric scale, so put `choose_reference`'s choice first to anchor on
the clearest light. Lights may differ in rotation by any amount. A light that
cannot join, such as one of another target, comes back rejected with the
reason; `StackError` is raised only when nothing joins the anchor. The stack starts empty; push every light, the
anchor included:

```python
from pathlib import Path

anchor = seiza.choose_reference(panel_lights).reference_path
order = [anchor] + [path for path in panel_lights if Path(path) != anchor]
options = seiza.StackOptions(
    normalization="local-background",
    minimum_overlap=0.1,
    registration_model="quadratic",  # warps frames across a meridian flip
)
stacker = seiza.LiveStacker.open_fits_expanding(
    order, options=options, dark="master-dark.fits", flat="master-flat.fits"
)
dispositions, report = stacker.push_fits_pipelined(order)
stacker.finish("mosaic.fits")
```

`max_canvas_megasamples` (256 by default) bounds the canvas's pixels times
channels; a run peaks at about 40 bytes per sample. When the panels' overlaps
show one lens distortion bending them all, planning undoes it and
`stacker.expanded_lens` describes it as a dict (its axis, coefficients,
largest shift in pixels, and the overlaps' residual with and without it);
otherwise it is `None`. `stacker.expansion_planning_notes` lists what planning
noted: why it skipped fitting one lens to every panel, or why it kept its first
plan when the lens-corrected one failed. `cancel`, a callable, is
asked before each light is planned, and Ctrl-C also stops planning. Drizzle is
not supported on an expanded canvas. See the CLI documentation for how panels
are matched and what the result looks like on real data.

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

To save the stack so far and keep going, call `stacker.write_fits(path)`,
which copies only the mean, or `write_fits(path)` on any `StackSnapshot`,
including one from `reintegrate()`. Both write linear `float32` FITS (XISF for
a `.xisf` path) with the reference frame's WCS, and refuse a path that is one
of the stack's inputs or masters.

`stacker.state()` describes the stack: its size and counts, a
`configuration_fingerprint` of its options, masters and input mode (to check
that a resumed stack is the one you meant), its `input_mode`, the
`input_paths` it has used, its `reference_frame` (role, master flag, signature
and calibration state), its `reference_image_path`, and
`reintegration_unavailable`, which says why `reintegrate()` would fail, or is
`None`. `stacker.set_calibration_fits(bias=..., dark=..., flat=...)` replaces
the masters for frames pushed from then on; frames already stacked keep the
calibration they had. Every file is read and checked against the reference
before anything changes, and passing none clears the masters:

```python
stacker.set_calibration_fits(dark="master-dark-300s.fits", flat="master-flat.fits")
print(stacker.state().configuration_fingerprint)
stacker.write_fits("so-far.fits")
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

Before pushing a light under masters you chose yourself, ask whether they
accept it. `validate_calibration()` raises `StackError` naming the reading that
failed and the tolerance in force; `compatible_calibration()` answers the same
question as the masters it would keep and why it would drop the rest. Pass the
`MatchTolerances` you matched the masters under, so both judge with your gate:

```python
light = seiza.FrameSignature(width=6248, height=4176, channels=1, rotation_deg=113.8)
wide = seiza.MatchTolerances(rotation_deg=180.0)
stacker.validate_calibration(light, wide)        # None, or raises StackError
kept, dropped = stacker.compatible_calibration(light, wide)
```

Passing `tolerances` replaces the stack's own whole: a field you leave unset
takes the built-in default, not the stack's value. Both calls raise
`StackError` for a tolerance that is NaN, infinite, negative, or a rotation
above 180 degrees, which already accepts any angle.

The stack judges every light it is given under its own gate, so give it the
same one, or the light is refused when you push it.
`StackOptions(match_tolerances=...)` sets the gate a stack opens with, which
also judges the reference. `set_match_tolerances()` changes it for the lights
pushed after, and `match_tolerances()` reads it back. Each admitted frame keeps
the gate it was admitted under, and `reintegrate()` judges it by that gate, so
you can widen the gate for one light and narrow it again after:

```python
stacker.set_match_tolerances(wide)
stacker.push_fits("light-113.8deg.fits")
stacker.set_match_tolerances()                   # the defaults again
```

A rotator's angle can be wrong for the question a flat asks: after a re-home
it can read the same angle for another position. When you have measured which
flats a light's pixels follow, by plate solving it or fitting its sky against
the candidate flats, give that angle as `light_rotation`, and the gate compares
it in place of the light's recorded angle. `"unknown"` compares the light at no
angle, so any flat angle passes; `"recorded"` is the default. The flats keep
their recorded angles, and `plan_calibration` reads it the same way:

```python
stacker.set_match_tolerances(seiza.MatchTolerances(light_rotation=3.5))
stacker.match_tolerances().light_rotation        # 3.5
```

A stacker built from FITS or XISF paths can integrate its admitted frames again
from their files, with leave-one-out rejection that removes trails the online
pass kept, and can drizzle that integration onto a finer grid as WBPP's
DrizzleIntegration does. Bayer frames drizzle each photosite into its own
colour:

```python
replayed = stacker.reintegrate()            # StackSnapshot; the live stack is unchanged
integrated, drizzled = stacker.reintegrate_drizzled(scale=2)
drizzled.write_fits("drizzle-2x.fits")      # reference WCS scaled to the grid
weights = drizzled.weight                   # drop area times frame weight
```

Each of those reads every frame from its file and prepares it again. Calling
`stacker.retain_frames_for_reintegration(scratch_directory=...)` before pushing
frames keeps each admitted frame's registered image on disk as it is stacked,
about four bytes per pixel per channel for each frame, so a reintegration reads
that instead and gives the same result sooner.

Frames calibrated, registered and normalized somewhere else, one size on one
pixel grid with uncovered samples as NaN, take the same rejection through
`integrate_registered_frames`. It reads each file once per pass and registers
nothing again; `frame_weights` gives one weight per channel for each frame:

```python
stack, frames = seiza.integrate_registered_frames(registered_paths, sigma_high=3.0)
stack.write_fits("stack.fits")                   # with the first frame's WCS
for frame in frames:
    print(frame.path, frame.finite_samples, frame.integrated_samples)
```


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
custom = seiza.combine_narrowband_matrix(
    ha, oiii, sii,
    matrix={"red": {"sii": 0.7, "ha": 0.3}, "green": {"ha": 1.0},
            "blue": {"oiii": 1.0}},
)
```

`combine_narrowband_matrix` mixes the filters through your own weights: each
output channel maps `"sii"`, `"ha"` and `"oiii"` to a weight, and a weight
left out is zero. A matrix that weights SII needs `sii`.

Each `combine_*` also writes its result when given `output=`: 32-bit float
FITS, or XISF for a `.xisf` name, labelled in `SEIZACLR` and with `SEIZATRF`
saying whether the samples are linear or display-referred. Pass the first
channel's FITS header (an astropy `Header` or a dict) as `header=` and the
output keeps its WCS, moved by any crop:

```python
from astropy.io import fits

red, red_header = fits.getdata("red.fits", header=True)
seiza.combine_rgb(red.astype(np.float32), green, blue,
                  output="rgb.fits", header=red_header)
```

Pass `crop="bounds"` or `crop="inscribed"` to trim the blank edges that
registering one channel onto another leaves behind. `bounds` keeps the box
every channel covers; `inscribed` keeps the largest rectangle they all cover in
full, so nothing stays `NaN`. `seiza.crop_report` measures the same thing
without composing, and flags any channel whose coverage sits more than
`report["off_center_limit_pixels"]` from the others:

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

`build_flat(..., flat_star_masking=seiza.FlatStarMasking())` leaves stars and
broad saturated patches out of each sky flat before it is normalized, and
`scratch_directory=` keeps the build's temporary pixels in a directory you
choose. `build_dark(..., dark_level_screening=1.0)` sets aside darks whose
level sits more than that many times the quietest dark's pixel noise above
it, the mark of stray light. The result's `masked_samples`,
`input_masked_samples` and `flat_star_masking` say what masking left out, and
`rejection_method` which rejection the build used.

A build sets aside an input whose header contradicts the first one, such as a
flat from another rotator angle, rather than failing.
`result.skipped_inputs` lists each as `(path, reason)`. `result.input_paths`
lists the inputs used, and `result.input_statistics` their `(accepted,
rejected)` sample counts, in the same order. The build checks that these
account for every input before it writes the master.

`build_flat(..., defect_suppression=True)` replaces hot and cold pixels in the
flat master with their neighbours' median. A defective pixel repeats in every
flat, so rejection across frames keeps it. `defect_sigma_low` and
`defect_sigma_high` (16 by default) set how many sigma from its neighbourhood a
pixel must sit; the default leaves star cores alone.
`result.defect_pixels_replaced` counts the pixels it replaced.

### Choosing calibration frames

`seiza.probe_frame(path)` reads a FITS or XISF header, without its pixels, into
the frame's `role` (`"bias"`, `"dark"`, `"dark-flat"`, `"flat"`, `"light"` or
`"unknown"`), whether it `is_master`, its `FrameSignature` and its
`calibration_state`. `seiza.plan_calibration` applies the whole matching
policy to those probes, as the desktop apps do. It keeps the candidates that
suit every light, orders them by capture time, nearest the reference light
first, and takes the first set of `minimum` frames that agree with each other.
It says why it left out each of the rest:

```python
light = seiza.probe_frame("light-001.fits")
darks = [seiza.probe_frame(path) for path in dark_paths]
plan = seiza.plan_calibration("dark", light, darks, minimum=5)
if plan.ready:
    seiza.build_dark(plan.selected_paths, "master-dark.fits")
for path, reason in plan.excluded:
    print(path, reason)  # for example "exposure-mismatch"
```

`kind` is `"bias"`, `"dark"`, `"dark-flat"` or `"flat"`. Dark flats are
planned against the flat they will calibrate rather than a light. Pass
`references=` to choose one set for several lights, `tolerances=` a
`MatchTolerances`, and `bias_available=True` once you have a bias master, so
that darks of another exposure can be scaled. A `seiza.CalibrationRecord(path,
signature, role=...)` stands in for a probe when your records come from
elsewhere. A bad request raises `ValueError`.

### Screening calibration frames

Before a master is built, judge its frames as a set. `measure_dark` and
`measure_flat` read one raw FITS or XISF frame each and return a small
measurement; keep it (`to_json()`, `DarkMeasurement.from_json()`) and judge it
again whenever its set changes. `screen_bias`, `screen_darks` and
`screen_flats` give each frame a `FrameVerdict`: a `grade` (`"ok"`,
`"suspect"` or `"poor"`) and the `findings` behind it, each with its `code`,
`value` and the `limit` it reached. Most checks compare a frame with its peers
and need three or more frames. The verdicts say what the pixels show, never
why:

```python
darks = [seiza.measure_dark(path) for path in dark_paths]
for path, verdict in zip(dark_paths, seiza.screen_darks(darks)):
    print(path, verdict.grade, [finding.code for finding in verdict.findings])

flats = [seiza.measure_flat(path) for path in flat_paths]   # raw, before calibration
report = seiza.screen_flats(flats, clipped=(0.001, 0.01))   # (suspect, poor)
print(report.master.grade, report.changing_features)
```

A flat is measured against `full_scale` (65535 by default), and is read as a
raw colour frame when its header records a Bayer pattern (`cfa=` decides
otherwise). Each limit is a `(suspect, poor)` pair.

### Residual flat responses

A flat cannot correct dust that moved after it was taken. Given crops of
several lights cut at the same place on the detector, `build_residual_flat`
keeps only the light most of them lose, and returns that response with its
evidence. Show first that the dip stays put on the detector while the sky
moves; this does not decide what it is:

```python
response, diagnostics = seiza.build_residual_flat(crops)   # one crop per light
seiza.validate_residual_flat_response(stored_response)      # raises ValueError
corrected = seiza.apply_residual_flat_response(response, light, x, y)
```

The response is 1 where nothing is corrected and never gains more than
`maximum_gain` (1.2). `apply_residual_flat_response` returns a corrected copy
with the response's top-left corner at pixel `(x, y)`.

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

## RC-Astro external tools

Drive RC-Astro's standalone `rc-astro` CLI (BlurXTerminator, StarXTerminator,
NoiseXTerminator) on image files. The CLI publishes its own contract — read
each tool's parameters from `rc_astro_tool_schema` rather than hard-coding
flags, because they change between CLI builds. Runs need an installed,
licensed CLI; the schema's `licensed` flag and `cli_version`/`ml_version`
(cache-key material) come from the same probe.

```python
import seiza

print(seiza.rc_astro_locate())               # None when not installed

schema = seiza.rc_astro_tool_schema("sxt")
for p in schema.parameters:
    print(p.name, p.type, p.default, p.min, p.max)

# Star removal, keeping the stars image as a sidecar so starless and
# stars can be stretched independently.
run = seiza.rc_astro_process_file(
    "sxt", "stack.fits", "starless.fits",
    parameters={"stars": True},
    progress=lambda fraction: print(f"{fraction:.0%}"),
)
print(run.primary, run.sidecars, run.device)
```

Ctrl-C cancels a running tool within half a second, as does an optional
`cancel=lambda: ...` predicate returning true. A `False` switch emits the
CLI's `--no-` negation, overriding a true default. A child completely
silent for ten minutes is killed; a first run downloads ML models, which
`rc-astro download-models` handles ahead of time.

## Parallax fly-through videos

Make the video `seiza parallax-video` makes, from a starless image, its
stars and a plate solution: the stars fly at their Gaia distances as the
camera moves toward a point of the nebula. `rc_astro_split_stars` splits a
stretched image with StarXTerminator, and `solve_blind` gives the WCS.

```python
import seiza

starless, stars = seiza.rc_astro_split_stars(image)   # float32 (h, w, 3), 0 to 1
video = seiza.ParallaxVideo(
    starless, stars, solution.wcs,
    focus=(5997, 2351), start="whole", dolly=0.8, seconds=16, size="4k",
    overlay=True, watermark=True,
    progress=lambda event: print(event.get("message", "")),
)
print(video.summary["background_distance_pc"], video.frame_count)

first = video.frame(0)                  # uint8 (height, width, 3)
for frame in video:                     # every frame in order, e.g. for PyAV
    container_stream_encode(frame)
video.render(lambda frame, index: encoder.write(frame), format="bgra")
video.write("iris.mp4")                 # through ffmpeg; encoder="png" for frames
```

A tour visits several places: `tour=` takes stops, each a `--stop` string
or a dict. `seiza.plan_parallax_tour(width, height, wcs, size=...)` plans
one from the catalogued objects in the field and returns it for editing:

```python
plan = seiza.plan_parallax_tour(width, height, solution.wcs, size="1080p-portrait")
for stop in plan["tour"]:
    print(stop["name"], stop["focus"], stop["dolly"])
stops = [stop for stop in plan["tour"] if stop["name"] != "LDN 1267"]
video = seiza.ParallaxVideo(starless, stars, solution.wcs, size="1080p-portrait",
                            tour=stops, distance_focus=plan["focus"])
```

`auto_tour=True` plans and renders in one go. Each planned stop's
`title` is its target's name; `tour_titles=True` shows it low in the frame
while the camera drifts through the stop, fading in and out, so change or
drop titles as you would stops. A tour ends on a longer final drift that
slows to rest; `tour_loop=True` instead ends where it began, so the last
frame leads into the first and the video loops. Images with transparency
go over the frames too: a stop's `card` (a path) shows in place of its
title, `intro_card` and `outro_card` over the start and the end for
`card_seconds`, and `watermark={"image": "logo.png", "corner":
"top_left"}` puts a logo in a corner, with `text`, `size` and `opacity`
keys as well.

Depth planes and dust work as in the command and the C API. `planes=` takes
`(mask, distance_pc)` or `(mask, distance_pc, name)`, each mask a path or a
2-D array, white where the image belongs to the plane; `auto_planes=True`
finds planes from the catalogued objects and the dust; and
`distance_overrides={"NGC 7822": 900}` gives catalogued objects distances of
your own. `summary["planes"]` lists the planes found. `dust_map=` (a path or
a 2-D array, white where the dust is opaque) replaces the dust counted from
the stars; `video.dust_map()` reads the dust back as an array, so it can be
painted over, `video.with_dust_map(painted)` swaps a map in without
preparing again, and `video.write_dust_map("dust.png")` saves it. Once a
video is made, `video.plan_tour(auto_tour)` plans a tour on its scene, each
stop framed at the depth of the plane its target lies on, as `auto_tour=`
would; `plan_parallax_tour` plans from the catalogue alone:

```python
video = seiza.ParallaxVideo(starless, stars, solution.wcs, auto_planes=True)
plan = video.plan_tour({"targets": 6})
toured = video.reconfigure(tour=plan["tour"], tour_titles=True, tour_loop=True)
```

`distance_focus` is where the nebula's distance is measured; `focus` only
aims the camera, so moving it never changes the scene's depth. To film the
same scene again with another camera, tour, size or quality, call
`reconfigure`. It returns a new video that shares the prepared scene, so
nothing is loaded, solved or looked up again, and leaves the first as it
was; both can draw frames at once. Options that change the scene, such as
the distances, `distance_focus`, star placement or dust, raise `ValueError`.
`summary["fit"]` gives each zoom and pan as asked and as used, and each tour
stop's dolly, after the camera was fitted to stay inside the image and to
show little of the fill behind depth planes, and `inside`, false when some
frames still show past the image's edge:

```python
wide = video.reconfigure(size="4k", quality="high", rotate_deg=(0, -360))
print(wide.summary["fit"]["zoom"])      # (asked, used)
```

An image of another shape than the frame is cut to fill it by default.
`framing="fit"` shows all of it at the widest, with black beside it, and a
size ending `-image` takes the image's own shape: `"4k-image"` films a 4:3
image at 3840x2880, all of it. `crop=(x, y, width, height)` films part of
the image, and every other image point, such as `focus` and tour stops,
counts in the cropped image; pass the same `crop` to `plan_parallax_tour`:

```python
video = seiza.ParallaxVideo(starless, stars, solution.wcs,
                            crop=(800, 400, 6000, 4000), size="4k-image")
square = video.reconfigure(size=(1080, 1080), framing="fit", start="whole")
```

Images may be paths (PNG, JPEG, TIFF, FITS or XISF) or arrays, float32 from
0 to 1 or uint8; 16-bit or float files make the best videos, as 8 bits flatten bright star
cores and band smooth nebula. Options are the command's, in snake case,
with frame sizes such as `"720p"`, `"1080p"`, `"4k"`, `"1080p-portrait"` or
`"4k-image"`.
Distances come from the star distance file when installed, or the archives
unless `online=False`. `render` stops when its callback returns `False`;
Ctrl-C or `cancel` stop preparing, `reconfigure`, `render` and `write`.

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
    duration_s=120.0,                 # or end=, in the same forms as start
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
- Parallel work runs in Rayon's global pool, which by default has a thread
  for every core. To cap it, set `RAYON_NUM_THREADS` before the first call
  into `seiza`, for example in the environment that starts Python.
- Catalog files are memory-mapped and SHA-256 verified at download time;
  `fetch_catalogs` caches under the platform cache directory (override with
  `cache_dir=` or `SEIZA_CACHE_DIR`).
- `seiza.StarCatalog.from_stars([...])` builds a small in-memory catalog for
  tests and synthetic fields.

## License

Apache-2.0
