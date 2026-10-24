# seiza-cli

The `seiza` command-line tool: star detection, hinted and blind plate
solving, and star/object dataset management for astrophotography. The
library lives in the [seiza](https://crates.io/crates/seiza) crate.

## Install

### Windows

Download the x86-64 MSI from the
[latest GitHub release](https://github.com/theatrus/seiza/releases/latest).
The installer supports all-users and current-user installs, adds `seiza` to
`PATH` by default, and can launch the guided catalog setup when installation
finishes. All-users installs place catalogs in the shared
`%ProgramData%\Seiza\catalogs` directory; current-user installs use the
user's local application-data directory.

A portable x86-64 ZIP is available on the same release page. See the
[Windows installer documentation](https://github.com/theatrus/seiza/blob/main/packaging/windows/README.md)
for feature selection, unattended installation, and catalog-directory details.

### Cargo

```
cargo install seiza-cli
```

## Solving

```
# Hinted: approximate center plus pixel scale. A FITS or XISF file's own
# headers supply either when the flags are missing: RA/DEC or
# OBJCTRA/OBJCTDEC, and a scale from its WCS, PIXSCALE, or XPIXSZ over
# FOCALLEN. --data takes a catalog file or a directory of catalogs; after
# `seiza setup` it can be omitted entirely.
seiza solve image.jpg --data data --ra 324.8 --dec 57.5 --scale 2.8
seiza solve light.fits --scale 1.45
seiza solve light.fits

# Blind: no position, just a plausible scale range. A directory supplies
# the deepest catalog and the blind index automatically. A FITS or XISF
# file's own WCS (FITS cards or a PixInsight solution) is checked first,
# then its position and scale searched around; if nothing solves, its own
# WCS is used unchecked, with a warning.
seiza solve-blind image.jpg --data data --min-scale 0.5 --max-scale 15

# Give up after 30 seconds of solving (detection not counted)
seiza solve-blind image.jpg --data data --time-limit 30

# JPEG: inspect EXIF and solve a wide field without a pointing hint.
seiza image-info phone.jpg
seiza solve-blind phone.jpg --data data --sip-order 2 \
  --annotate solved.png --wcs solved.wcs

# A labelled chart: constellation figures, named stars, deep-sky objects,
# with marks on the ground below the horizon left out
seiza solve-blind phone.jpg --data data --sip-order 2 --sky-map sky-map.png \
  --sky-map-foreground

# Annotate detections or list objects in a solved field
seiza detect image.jpg --annotate out.png
seiza solve image.jpg --data data ... --objects data

# Predict tracks for one exposure from a cached current OMM set, or from an
# offline/historical OMM JSON or TLE file. FITS supplies time and OBSGEO when
# present; explicit metadata is also accepted.
seiza solve light.fits --scale 1.45 --satellites-celestrak --annotate tracks.png
seiza solve light.fits --scale 1.45 --satellites elements.json \
  --time 2026-07-18T08:30:00Z --exposure-seconds 30 \
  --observer-lat 37.3 --observer-lon -122.0 --observer-alt-m 50

# Query objects when sky bounds are already known; no image or solve needed
seiza catalog objects --data data --ra 10.6848 --dec 41.2691 --radius 3
seiza catalog objects --data data \
  --corner 8.91,42.14 --corner 12.47,42.02 \
  --corner 12.31,40.35 --corner 9.02,40.46 \
  --sort prominence --format json

# Resolve exact IDs/names or complete names; no image, solve, or network needed
seiza catalog object --data data "M 31"
seiza catalog object --data data "openngc:NGC224"
seiza catalog object --data data "andro" --prefix --limit 10
seiza catalog star --data data "TYC 5949-2777-1" --format json
seiza catalog star --data data "HIP 32349"
seiza catalog star --data data "RR Lyr"
seiza catalog star --data data "STF 2382 AB"
seiza catalog star --data data "RR L" --prefix --limit 10

# Distance to an object, from the optional object-distances.bin
seiza catalog distance --data data "M 42"
seiza catalog distance --data data "Horsehead Nebula" --format json
```

Explicit file paths still work everywhere a directory is shown, for
custom-built catalogs or unusual layouts.

### JPEG metadata and pixel coordinates

Raster inputs are decoded and EXIF Orientation is applied once, including
mirrors and the four transforms that swap width and height. Every command
that reads a JPEG, PNG or TIFF does this, including the ASTAP and
`solve-field` drop-ins and the JSON-RPC worker, and so do the C API and the
desktop apps built on it: detection, solving, annotations, previews and
exported WCS all use the same upright pixels.
WCS pixel coordinates in the library are zero-based; the FITS WCS export
uses FITS' one-based CRPIX convention. Apply the source Orientation before
overlaying an exported WCS on the original JPEG. Annotated PNGs contain
already-oriented pixels and must not be rotated a second time.

`image-info` emits JSON containing original/oriented dimensions, the applied
transform and optional EXIF fields: camera, acquisition timestamp,
subseconds, timezone offset, GPS position/altitude, heading and focal length.
It recognizes the file type from its content, not its name, and declines
FITS and XISF, whose headers carry this instead. Subseconds retain leading
zeros. The UTC acquisition time comes from DateTimeOriginal with
OffsetTimeOriginal, or else from the GPS date and time stamps, which are UTC
by definition (`capture_time_source` says which); the machine's timezone is
never assumed. An explicit `solve --time`
overrides the EXIF acquisition hint used for minor bodies. Missing/malformed
EXIF fields remain optional and do not prevent a stellar solve.

When `solve-blind` scale bounds are omitted and a valid 35mm-equivalent focal
length is present, the CLI estimates central angular scale from the decoded
image diagonal and first searches between half and twice that estimate. The
estimate assumes the image is the camera's whole frame, so a crop or a phone
held to an eyepiece can fall outside it; when that first search finds
nothing, `solve-blind` searches again from 0.1"/px up to the estimate's
coarse end. Each explicit `--min-scale` / `--max-scale` holds in both
searches, and a hint bound that conflicts with an explicit one is dropped.
Without this metadata the previous 0.1–20 arcsec/px defaults apply. SIP
remains opt-in; compare residuals with and without it.

An iPhone main-camera frame (24 mm equivalent, from
[#210](https://github.com/theatrus/seiza/pull/210)) solved blind at
71″/px, with detections in green and Gaia stars in red:

![Phone photo solved blind](../docs/images/phone/phone-solve.jpg)

The same frame cropped to its central 1344×1008 pixels makes the
focal-length estimate three times too coarse; the first search misses and
the wide retry solves it at 72″/px:

```
pixel-scale search: 110.669–442.677"/px (EXIF equivalent focal length; explicit bounds override)
600 stars detected in 1344x1008 image
no solution in the EXIF focal-length range; retrying 0.100–442.677"/px
Blind-solved in 2.02s:
  center     : 03h 19m 42.05s +70° 56′ 13.5″  (49.92521°, 70.93708°)
  pixel scale: 72.0169"/px
  quality    : 106 stars matched, RMS 116.265"
  by chance  : 4.8e-90 (false-match probability)
```

`by chance` is the chance that a wrong field would match as well, counting
every candidate the search tried. A true field scores far below one in a
billion; `solve-blind` refuses a field above that, and `solve` one above one
in a thousand.

![Cropped phone photo solved after the wide retry](../docs/images/phone/phone-crop-solve.jpg)

`--annotate` on `solve` and `solve-blind` circles detected stars (green) and
catalog stars (red). Fields wider than 10° across the diagonal, such as phone
and camera-lens frames, get small markers and up to 600 catalog stars;
narrower telescope fields get larger markers and 300 stars. Annotated
images, like sky maps, are pictures to look at: FITS and XISF input is drawn
from an automatic display stretch, in colour when the file has colour,
whatever `--detection-backend` the solve used.

### Sky map

`--sky-map <out.png>` on `solve` and `solve-blind` writes a chart of the
solved image (EXIF-oriented, as loaded) for sharing. FITS and XISF images
get an automatic display stretch, in colour when the file has colour:

- constellation stick figures (light blue) and constellation names;
- stars with IAU proper names down to magnitude 4.5 (gold), plus figure
  stars of magnitude 3 or brighter without one, by Bayer designation
  ("Gamma Cas", "Eta Cen"); the brightest 28 get labels;
- the 14 most prominent deep-sky objects (cyan), with common names,
  catalog ellipses for large ones, and "(edge)" when only the extent
  reaches the frame;
- a title bar, and a footer with matched stars, SIP order, pixel scale, RMS,
  field centre and the line-data credit.

Labels go where they overlap nothing placed before them, brightest stars
first, and are dropped when nowhere fits. `--sky-map-width` sets the width,
from 640 to 12000 pixels (default 2100).

`--sky-map-foreground` is for a photo with a horizon, trees or buildings in
it: marks below the lowest detected stars that a catalog star confirms are
left out, and figure lines fade there. It looks only at fields over 10°
across, and only where the stars stop well short of the bottom of the frame
across several neighbouring columns, so a frame with stars everywhere loses
nothing. A named star the image shows is kept even below that line. The
footer says when marks were left out.

Star names come from the star-identifier sidecar and objects from
`objects.bin`, found next to `--data` or in the standard places (`solve
--objects` picks the object file); without them the map still has the
figures. Catalog marks show where things are, not that they were detected.

![Sky map of the phone frame](../docs/images/phone/phone-sky-map.jpg)

The figures are the Constellation Lines dataset by Marc van der Sluys
(2005-2023), hemel.waarnemen.com. DOI: 10.5281/zenodo.10397192. Licensed under
CC BY 4.0. Labels use the Inter typeface (SIL Open Font License 1.1, see
[`fonts/`](fonts/README.md)).

EXIF Orientation is distinct from camera pointing. GPSImgDirection and its
magnetic/true-north reference are reported but do not seed the solver; the
stellar WCS determines pointing and rotation. HEIF and RAW/DNG decoding and
multi-frame timing are not supported by this JPEG metadata path.

JPEG DateTimeOriginal and ExposureTime **do not establish a continuous
exposure**. In particular, phone Night Sight/Night mode can combine many
frames. Satellite overlays therefore still require explicit `--time` and
`--exposure-seconds` describing one actual exposure; never pass a stack's
total integration. JPEG GPS latitude/longitude can supply the observer
position when both explicit coordinates are absent. Explicit coordinates
take priority as a pair. EXIF GPS altitude is reported with its original
reference and is not treated as ellipsoid height: supply
`--observer-alt-m` when that height is known (otherwise the existing zero
height default applies).

Star detection defaults to `--detection-backend auto`: decoded 8-bit images
(including color JPEGs) and MTF-compressed FITS use the compact u8 pipeline,
while other higher-precision images use f32. Pass `--detection-backend f32` to
retain fractional luma during an 8-bit solve or to detect directly from linear,
native-precision FITS samples; pass `--detection-backend u8` to explicitly
quantize any input. The option is global and applies to `detect`, `solve`,
`solve-blind`, and local `worker` solves.

Auto solves default to `--detection-fallback f32`. After an Auto/u8 solve miss,
converted 8-bit color is redetected as f32 and FITS is reopened so detection can
use its linear high-precision samples. `--detection-fallback none` disables the
retry. Explicitly selected detection backends never fall back.

`catalog objects` accepts a cone or a convex polygon whose vertices are in
boundary order. It can filter by object kind, magnitude, angular size, and
common-name availability; results can be emitted as a table, JSON, or CSV.
JSON and CSV include primary and alternate stable IDs, primary and contributing
source provenance, aliases, and parent IDs when the catalog provides them. The
prominence score is a catalog-based prediction, not proof that the object is
visible in the image pixels.

`catalog object` resolves primary/common names, aliases, and stable or
alternate IDs. Both object viewport queries and name completion use indices
embedded in the memory-mapped `objects.bin`; normal open does not decode every
record or touch every index page. Add `--all-sources` to audit every normalized
upstream row, preferred facet selection, and source-qualified geometry.

`catalog distance` reads `object-distances.bin`, an optional download that
gives about 180,000 catalog objects a distance in parsecs: clusters, nebulae,
supernova remnants, dark clouds and galaxies. Each comes with its range where
the source gives one, the method, the source and its licence, and the
publication. An object no source measures may borrow a distance: from the
galaxy it lies in (a cluster in the Large Magellanic Cloud), from a nebula or
remnant that contains it (the Veil's filaments), or for a dark cloud from the
nearest molecular-cloud sightline. The output says which. With no entry at
all, the command reports a typical distance for the object's kind, marked
`kind-default`. The file records the fingerprint of the `objects.bin` it was
built from, and the command refuses any other: object IDs change between
catalog builds, so a mismatched pair would answer for the wrong objects. The
file is released under the ODbL 1.0 because it is built from SIMBAD.
Applications find the object at a pixel of a solved image, with its distance,
through `ObjectDistances::object_at_pixel` in the `seiza` crate.

## Background extraction

Fit and remove a smooth background from a calibrated linear FITS or XISF image:

```
seiza background stack.fits --output corrected.fits \
  --model-output background.fits --diagnostics background.json
```

The default is a robust, per-channel quadratic surface with shared sample
positions. Use `--degree 1` for a conservative plane, `--mode divide` for a
multiplicative field, and `--border-fraction` to keep sample windows away from
registration edges. `--sample-radius`, `--samples-per-axis`, rejection sigmas,
and refit count are available for controlled tuning.

Corrected and model outputs are linear 32-bit floating-point FITS and preserve
a valid input WCS. The JSON diagnostics include coefficients, reference
levels, sample positions, weights, and rejection reasons. The input, corrected
output, model, and diagnostics must be distinct paths. Raw Bayer mosaics are
rejected because fitting the interleaved CFA colors as one channel would create
a false surface; debayer or stack them first.

## Photometric colour calibration

Calibrate a linear RGB stack's colour against Gaia DR3 star colours:

```
seiza color-calibrate stack.fits --output stack-cc.fits --report colour.json
```

The image needs an astrometric solution: FITS WCS cards, which `seiza stack`
keeps from the reference frame, or the `AstrometricSolution` properties
PixInsight writes into XISF files. A PixInsight solution's distortion layers
have no TAN-SIP equivalent and are left out, which moves stars near the corners
of a distorted field by a few pixels; the photometry finds them anyway. Seiza reads Gaia DR3 photometry from an offline catalog
when one is installed (`stars-gaia-photometry.bin` in a catalog directory, or
`--gaia-catalog`). Otherwise it fetches the field from ESA's Gaia archive, or
from GAVO's mirror when ESA's fails, and caches it (`--gaia-cache`); a CSV you
supply also works (`--gaia-csv`). It measures every isolated, unsaturated Gaia star in R, G and
B, fits each instrumental colour against Gaia BP−RP, and sets the gains that
render a star of the white reference's colour neutral: the Sun's (BP−RP 0.82)
unless `--white-bp-rp` says otherwise. No filter or sensor curves are needed;
the stars measure the camera's own response. Background neutralization then
gives the sky the same level in every channel (`--no-background-neutralization`
leaves it).

On the 126-frame M45 stack (ASI2600MC, 173 mm), about 1,670 isolated Gaia
stars brighter than G 13 calibrated it, with 0.044 and 0.032 mag of scatter
about the red and blue colour fits. PixInsight's SPCC, given the same image and
a G2V white reference, chose a red gain 4.8% lower and a blue gain 3.2% higher.
The colour model is not the cause: on the stars both tools measured, SPCC's
filter-curve method and Seiza's BP−RP fit agree within 0.7% when given the same
fluxes. The fluxes differ instead. For 60 isolated stars SPCC's PSF photometry
reports R/G 0.712, while apertures from 4 to 36 px all give 0.667–0.697.

To see the result, add a display image. The calibrated colours are
stretched with a linked Auto-MTF unless `--stretch` gives another stretch, as
JSON or a file of it (one stage, or an array of stages applied in order):

```
seiza color-calibrate stack.fits --output stack-cc.fits --display stack-cc.png
seiza color-calibrate stack.fits --output stack-cc.fits --display stack-cc.png \
  --stretch '{"model": {"type": "auto-mtf", "target_median": 0.25, "shadows_clip": -2.8}, "color_strategy": "linked", "max_analysis_samples": 200000}'
```

With `--display`, a calibration that cannot run (too few stars, say) does not
fail the command: it prints the reason and writes both images uncalibrated.
See the [colour calibration design](../docs/design/color-calibration.md).

The offline catalog is built from Gaia DR3 itself (CC BY-SA 3.0 IGO), not from
PixInsight's databases. Seiza hosts a prebuilt copy (about 460 MB, downloaded zstd-compressed) that is
not part of the standard bundle; install it with setup or by name:

```
seiza setup --preset solver-lite --gaia-photometry
seiza download-data prebuilt --output data --file stars-gaia-photometry.bin
```

To build it yourself, `scripts/build-gaia-photometry.sh` downloads G, BP, RP
and RUWE for every source to G 15 in 768 resumable chunks and builds the
catalog, about 470 MB; it takes a few hours and can run unattended. Copy the
result into Seiza's catalog directory (`$SEIZA_CATALOG_DIR`, or
`~/.local/share/seiza/catalogs` on Linux) or pass it with `--gaia-catalog`:

```
ARCHIVE=gavo nohup scripts/build-gaia-photometry.sh ~/gaia-photometry > gaia.log 2>&1 &
cp ~/gaia-photometry/stars-gaia-photometry.bin ~/.local/share/seiza/catalogs/
```

The two steps are also available on their own: `seiza download-data
gaia-photometry --output chunks [--archive esa|gavo]` and `seiza build-data
gaia-photometry --input chunks --output stars-gaia-photometry.bin`.

## Light deconvolution (experimental)

`seiza deconvolve` applies a conservative damped Richardson-Lucy pass to a
calibrated or stacked linear mono/RGB FITS/XISF image. Measure the FWHM of several
unsaturated stars in pixels and use their median as the explicit Gaussian PSF:

```text
seiza deconvolve stack-bg.fits --output stack-light-dc.fits \
  --psf-fwhm 3.1 --iterations 4 --amount 0.35 --noise-fraction 0.001
```

Run this after calibration, stacking, and background correction and before a
display stretch. Start with 3-5 iterations and a 0.25-0.4 blend. Compare input
and output with the same stretch, paying particular attention to bright-star
rings, background noise, and image borders. Raw Bayer mosaics are rejected;
debayer or stack them first. Missing registration borders marked with `NaN`
stay masked in the output.

This prototype uses one circular Gaussian PSF across the whole field. It does
not infer detail, estimate a spatially varying PSF, or replace a model-based
restoration workflow. See the
[deconvolution design note](../docs/design/deconvolution.md) for the algorithm,
guardrails, and next experiments. Four
[real-corpus comparisons](../docs/benchmarks/2026-07-deconvolution-corpus.md)
record the first measured trial, and the
[model-based restoration plan](../docs/design/ml-restoration-training.md)
defines a safer path from synthetic degradations and expert before/after pairs
to a provenance-bearing learned operation.

## Parallax fly-through videos

Fly toward a point of a stretched image, with its stars at their distances:

```
seiza parallax-video m45.tif --starless m45-starless.tif --stars m45-stars.tif \
  --output m45.mp4
```

The starless and stars images are the stretched image split by
StarXTerminator, with "unscreen" so the stars screen back over the starless
image. Given only the stretched image, Seiza runs StarXTerminator's `rc-astro`
CLI itself when it is installed and licensed. Inputs can be PNG, JPEG, TIFF,
FITS or XISF, but 16-bit or float images give the best video: at 8 bits a bright star's core
flattens, which can misplace it so it loses its distance, and smooth nebula
bands as the camera nears it. Seiza warns when an image has only 8 bits a
channel.

The image is plate-solved and its stars matched to Gaia DR3 for their
Bailer-Jones distances; Hipparcos supplies the brightest stars Gaia has no
parallax for. Both come from the optional star distance dataset (`seiza setup
--star-distances`, every Gaia star to G 16), or are queried online in small
cones and cached when it is not installed. To build that dataset, `seiza
download-data star-distances` fetches the sky in small HEALPix tiles and
`seiza build-data star-distances` packs them. The starless image is placed at
the distance of the catalogued object at `--distance-focus x,y` (default
`--focus`, the first tour stop flown in toward, or an automatic tour's best
target), from the optional object distance dataset (`seiza setup --object-distances`), or at `--distance`
parsecs. Stars too faint to match sit at the star field's median
distance. Only the distance focus sets the scene's depth, so moving the
camera's destination never moves the nebula.

The camera flies `--dolly` of the way to the target (default 0.4) toward
`--focus x,y` (default the image centre). It never turns: every change of view
comes from moving it, so near stars slide across far ones as they would.
`--start focus` (the default) opens on the widest view centred on the focus
point and flies straight at it; `--start whole` opens on the whole image and
moves sideways as it flies in, until the focus point is ahead. `--pan F` has
it turn for that fraction of the way instead, which sweeps the far star field
with the nebula. `--truck`
swings the camera sideways and back (`--truck-angle` sets the direction),
reduced if the far stars would slide off the image, and `--zoom-end`
lengthens the lens over the shot. `--rotate` and `--rotate-end` turn the frame
about its centre, in degrees anticlockwise, from the first frame to the last;
every depth turns alike, and the first frame zooms in as far as the turned
frame needs to stay inside the image. A tour visits several places instead: give two or more `--stop "X,Y
dolly=D zoom=Z rotate=DEG pan=P travel=S hold=S"` (or `whole` for the image's
centre), and the camera glides through them in turn, easing to a halt
where it holds, turning and panning as each stop asks. The first stop is
the opening view, and the stops' times set the video's length; if a turn
or pan would show the image's edge, every view zooms in together as far as
it must. The camera never quite stops: it moves on through a stop it holds
at at a fifth of its pace between stops (`--tour-glide`, 0 to come to rest).
A stop's `spin=DEG` turns the frame while it holds there, and `push=P` flies
on in by that share of the remaining way, which with a spin turns the near
stars in a spiral past the far ones. The last stop's hold is a final drift
that slows to rest. `--loop` ends the tour where it began instead, adding a
last stop at the first's view unless the tour ends there, so the last frame
leads into the first and the video loops. A stop on a depth plane (below)
centres its place on that plane, turning to look along the line of sight
through it; `depth=PARSECS` names the plane's distance, by default that of
the plane the image shows there. Behind a lifted plane the image keeps only
a smooth fill, which looks smeared where the photograph had texture: where
more than 15% of a frame would show it, as when the camera flies far in
toward the nebula to frame a plane far behind it, that stop flies less far
and zooms in to keep its target's size, and a note says so. A warning names any stop whose depth the
image does not show at its place, as in a tour planned before the planes
changed: drop its depth, or plan the tour again. A stop's `title="TEXT"` names it,
and `--tour-titles` shows each title low in the frame while the camera
drifts through its stop, fading in and out. `--auto-tour[=N]` plans a tour itself from the catalogued objects
in the field: every one worth a visit (the well-known catalogues, and
broad regions that are large, bright or named), up to twelve, or the N
most worth it, visited in a short round
from the whole image and back to a final drift three times as long, each framed to its size, with turns that
alternate in direction and a little pan; a long way between two targets
pulls back on the way. `--tour-hold` sets the seconds at each target and
`--tour-motion` how much it turns and pans (0 for none, 2 for twice the
default). To edit a planned tour first, `--plan-tour FILE` writes the plan,
one `--stop` a line, each target's titled with its name, and a `focus
X,Y` line for the distance focus, and stops; remove, move or change lines,
then render it with `--tour-file FILE`. `--seconds`, `--fps` and `--size` set the
video (`--size` takes `720p`, `1080p`, the default, `1440p` or `4k`, any of
them with `-portrait` for vertical video or with `-image` for the image's own
shape, so `4k-image` films a 4:3 image at 3840x2880, all of it; or
`WIDTHxHEIGHT`). When the image and the frame differ in shape, the widest
view fills the frame and cuts the ends of the image's longer side;
`--framing fit` shows all of it instead, with black beside it. The picture
ends where the far stars end, and they grow little as the camera flies in,
so the bars stay nearly fixed, as in a letterboxed film. `--crop X,Y,WIDTH,HEIGHT` films part of the
image; every other image point, such as `--focus`, the stops and a planned
tour's, counts in the cropped image. `--quality high` draws each frame at twice the size and averages it
down, blending levels of detail so fine detail neither shimmers nor steps in
sharpness as the camera moves, for about four times the rendering time. A deep image holds so many faint stars that, each moving on its own,
they crowd the view: `--max-stars N` lets only the N brightest fly, and
`--small-stars` drops the rest (`drop`, the default) or keeps them on the
distant star field's plane (`field`). Frames go to ffmpeg (H.264 with libx264 or
libopenh264, or with `--codec hevc` HEVC with libx265, about a fifth smaller at the same quality but slower to encode), in MP4, MOV or Matroska as `-o`'s extension says, to PNG files with `--encoder png`, or, in a build with the
`openh264` feature, to a built-in encoder. Catalogued galaxies, which a star remover leaves in the starless image, are
lifted out of it onto the far field, where they hold still while the nebula
grows past them; the nebula behind is filled in from around each one.
`--keep-galaxies` leaves them in place. Dust hides the stars behind it, so
the star counts map how much light it lets through, and a patch of the
starless image darker than its surroundings and short of stars marks a
globule thick enough to black out everything; whatever lies behind the
nebula dims as the camera's move slides it behind thicker dust than it was
photographed through. `--dust-opacity` (default 3) darkens or lightens the
dust, and `--no-dust` turns this off.

`--dust-map FILE` gives the dust as an image to use in place of the counted
map: white (1) where the dust is opaque, black (0) where the sky is clear,
grays between, at any size with the image's shape, as PNG, JPEG, TIFF, FITS
or XISF by the file's extension. A given map is taken at its word: where it
lets through 5% of the light or less, about 96% white, nothing behind shows,
so a star seen there flies in front of the nebula, among the field's
foreground stars. The counted map never moves a star; its finest cells see
small dark patches that are not dust.

`--export-dust-map FILE` writes the dust the video uses, the size of the
image and in the same convention, to paint over in an image editor and give
back with `--dust-map`: a 16-bit grayscale PNG or TIFF (`.png`, `.tif`) or a
32-bit float FITS or XISF (`.fits`, `.xisf`). Read back unchanged, it dims
the same; as a given map, its opaque places also bring stars forward, so
paint out any specks the counted map has in clear sky. Without `-o`, the
command prepares the scene, writes the map and stops:

```
seiza parallax-video m45.tif --starless m45-starless.tif --stars m45-stars.tif \
  --export-dust-map m45-dust.xisf
seiza parallax-video m45.tif --starless m45-starless.tif --stars m45-stars.tif \
  --dust-map m45-dust-painted.xisf --output m45.mp4
```

`--plane MASK=PARSECS` puts a region of the image at a distance of its own.
The mask is an image of the image's shape, at any size, white where the
image belongs to the region and black elsewhere, with grays for a soft edge
(PNG, JPEG, TIFF, FITS or XISF). The region's light is lifted off the
starless image onto a plane at that distance, and what lay behind it is
filled in from the light around it, so the first frame is still the
photograph. As the camera moves, a cloud in front of the nebula grows and
slides past it, and a cloud behind it barely moves. Where the region is
brighter than what fills in behind it, the plane takes the light above
that; where it is darker, and the plane lies in front of the nebula, a dark
cloud blocks the light behind, and the plane keeps the share it lets
through, each colour's. Everything behind the cloud then dims or brightens
as the cloud slides across it, and a star seen where the cloud is opaque
flies in front of it. A plane behind the nebula cannot darken it, and
leaves the darkness where it is. Within a twentieth of the fill either way,
the light is the background's own. Repeat the option
for more planes; where one lies inside another, the smaller keeps
its own light and the larger takes the rest. A star within 2% of a plane's
distance, inside the plane, joins it, and a label on a plane moves with it.
A plane fades as the camera flies through it, and where the image's edge
cuts a plane off, the camera keeps that edge out of view, as it does the
far star field's. Each plane costs a draw over the share of the frame it
covers:

```
seiza parallax-video field.tif --starless field-starless.tif --stars field-stars.tif \
  --plane near-cloud.png=900 --plane far-shell.png=2400 --output field.mp4
```

`--auto-planes` finds planes in the object catalogue: the nebulae in the
field whose distance was measured, or borrowed from an object around them,
that lie at least 15% nearer or farther than the nebula. A distance typical
of an object's kind never counts, and nor does one whose range takes in the
nebula's own, as it cannot say which lies nearer. Each plane takes the
object's outermost catalogued outline, or else its ellipse, grown to where
the image's light levels off. It takes the nebula's light alone: darkness
about it is dust, which stays with its cloud. A nebula that would cover more
than a third of the image is the scene's backdrop and stays with the
nebula, unless you give its distance; so does one that spans most of the
image or holds the nebula. A nebula inside one of these that the catalogue
places at its distance stays with it, and the notes name them.

A dark nebula is the thickest part of a cloud of dust, so it moves with
that cloud rather than on its own. The star counts split the dust into
clouds, faint dust included, each at one distance. A cloud that holds the
nebula stays with it, its dark nebulae too: the notes say which, and where
the Gaia stars put that dust. A cloud lying at least 15% in front of the
nebula moves as one plane, dark and glowing parts together, at the distance
of its largest dark nebula: measured, borrowed from an object around it, or
that of the nearest cloud measured. A cloud the catalogue gives no distance
lies where the Gaia stars seen through its thickest part run short: in front
of a cloud there are as many stars at each distance as over clear sky, and
behind it fewer, which is how published distances to dark clouds are
measured. It needs a few stars in front, and a clear fall behind. A cloud
that would cover more than a third of the image stays with the nebula,
unless you give a distance for a dark nebula in it. A dark nebula in clear
sky gets a plane of its own. The dust mapped from the star
counts clears where a dark cloud's plane blocks the light itself. The notes
name each plane, its distance and how that was found. `--object-distance NAME=PARSECS` gives an object's
distance in place of the catalogue's, for the nebula's distance and for the
automatic planes; repeat it for more objects. An automatic tour frames each
target at its own plane's depth: a nearer plane grows sooner, so the camera
flies less far, and the camera cannot fly past the nebula to a plane behind
it, so the lens makes up the rest.

`--export-planes DIR` writes the planes the video uses, automatic ones
included, as masks the size of the image (16-bit PNG), with `planes.txt`
listing each as `MASK=PARSECS`. Paint over a mask, change a distance or drop
a line, then give the list back with `--plane-file DIR/planes.txt` in place
of `--auto-planes`. Without `-o`, the command writes them and stops. With
planes, `--plan-tour` prepares the scene first, so the tour it writes frames
each target at its plane's depth:

```
seiza parallax-video bigq.tif --starless bigq-starless.tif --stars bigq-stars.tif \
  --auto-planes --export-planes bigq-planes --plan-tour bigq-tour.txt
seiza parallax-video bigq.tif --starless bigq-starless.tif --stars bigq-stars.tif \
  --plane-file bigq-planes/planes.txt --tour-file bigq-tour.txt --tour-titles \
  --output bigq.mp4
```

`--debug-layers DIR` writes the background, the leftover star light, every
cut-out star tinted by distance, each depth plane, a dark cloud's dust as
`plane-N-dust.png`, and the dust (white where opaque), for checking a
result.

`--overlay` labels the catalogued objects in the field the way Seiza's image
overlays do, with the same colours, names, outlines and ranking. Each label
sits at the depth of the layer that shows its object: the nebula's plane, a
named star's own distance, or the far field for a lifted galaxy, so it moves
with what it marks. Labels fade in and out rather than popping: as they near
the frame's edge, as a star the camera passes fades, and as smaller objects
take or lose their share of the view. `--overlay-density` (default 0.6) sets
that share. Once the camera is inside an object, its name moves to a "Field
within" line in the corner. `--label 'X,Y:TEXT'` adds a label of your own at
image pixel X,Y on the nebula's plane, and `--label 'X,Y,R:TEXT'` circles R
pixels about it as well; repeat it for more, and set their colour with
`--label-color '#RRGGBB'`. `--watermark` writes "Rendered with seiza.fyi" in
the bottom-right corner, or `--watermark='TEXT'` your own line. Labels,
titles and the watermark are drawn in Inter, which holds Latin and Greek
letters; others are left blank, with a warning.

Images go over the video as well, each a PNG or TIFF whose transparency is
kept. `--watermark-image FILE` draws one, such as a logo, in the
watermark's corner of every frame, its height 6% of the frame's shorter
side (`--watermark-size`) and at most a third of its width; with
`--watermark` too, it sits just inside the text. `--watermark-corner` moves
both to `top-left`, `top-right`, `bottom-left` or `bottom-right`, and
`--watermark-opacity` (0.8) sets how strongly both show. A title card is an
image you design yourself, in any typeface, laid over the frame: fitted
inside it and centred, so make it the frame's size (1920×1080, or 1080×1920
for `-portrait`) for sharp lettering. `--intro-card FILE` shows one whole
from the first frame and then fades it out, and `--outro-card FILE` fades
one in for the end, each for `--card-seconds` (3); with `--loop` they show
across the join instead, the closing card giving way to the opening one,
so the video still loops. A tour stop's `card=FILE` shows a card while the
camera drifts through the stop, as its title would, in its place, with or
without `--tour-titles`; in a tour file the path is taken from the file's
folder. A card without transparency, or of another shape than the frame,
is drawn with a warning, and a file that cannot be read stops the video
before it renders.

## Image stacking

`seiza stack` calibrates, registers, and incrementally integrates FITS or XISF light
frames onto the reference frame's fixed output grid:

```
seiza stack light-001.fits light-002.fits light-003.fits \
  --output stack.fits --preview stack.png --report stack-report.json

seiza stack lights/*.fits --output stack.fits \
  --bias master-bias.fits --dark master-dark.fits --flat master-flat.fits \
  --reintegrate --max-registration-drift 256 \
  --max-registration-drift-fraction 0.15 --min-overlap 0.60
```

### Defaults and quality options

The defaults aim for the best result from a night of frames; each step can be
changed:

| Option | Default | What it does |
|---|---|---|
| `--reference auto\|first\|expand` | `auto` | `auto` scores every frame on a half-resolution view (star signal over sky noise, which seeing, trailing, haze and twilight all lower) and, among frames near the best score, takes the flattest sky. `first` keeps the first frame given. `expand` auto-selects an anchor and grows the canvas to include connected overlapping panels (see below). |
| `--reference-image PATH` | none | Use a common FITS/XISF reference for separate channel stacks, including `--reintegrate` and `--drizzle`. Sets only the alignment grid and WCS. `--reference auto\|first` selects a light within this channel for normalization and noise weighting. No external-reference pixels are integrated unless its path is also listed as a light. |
| `--max-canvas-megasamples` | `256` | Largest canvas `expand` may grow, in millions of samples (pixels times color channels). A run peaks at about 40 bytes per sample, so the default allows about 10 GB. |
| `--min-overlap` | `0.60`, or `0.05` with `expand` | Expansion requires this fraction of the smaller finite footprint to overlap the growing template. Fixed canvases measure coverage of the reference grid. |
| `--registration-model similarity\|affine\|quadratic` | `quadratic` | After the similarity match, fit a polynomial to up to 2 000 stars to follow lens distortion. A frame with too few stars keeps its similarity. Under `expand`, each panel's first frame is placed through the lens the panels' overlaps measure (see below). |
| `--normalization none\|global\|local\|local-background` | `local-background` | `local-background` keeps one gain per channel, measured from star photometry, and matches each tile's background to the reference's, so frames whose sky gradients differ leave no seams where coverage changes. `local` also fits a gain per tile. |
| `--local-tile-size` | `256` | Tile size for `local` and `local-background`. |
| `--weighting equal\|inverse-noise` | `inverse-noise` | Weight each frame by the inverse of its noise variance relative to the reference, so hazy frames count for less. |
| `--interpolation bilinear\|lanczos3` | `lanczos3` | Lanczos-3 resampling, with PixInsight-style clamping against ringing; sharper, at about twice the resampling cost. |
| `--demosaic vng\|mhc\|bilinear` | `vng` | Bayer frames: VNG keeps each star's colour across its profile; MHC gives slightly sharper stars but rings around small ones; bilinear is fastest and softest. All three balance the mosaic's channels first. |
| `--bayer-drizzle` | off | Integrate each Bayer frame's photosites without demosaicing, on the reference grid; pays off on well-dithered data. Not the same as `--drizzle`. |
| `--defects auto\|on\|off` | `auto` | Learn the sensor's hot and dead pixels from up to 16 of the lights, and repair each in every light before demosaicing and registration: a steady hot pixel has its learned excess subtracted, as a dark would, so a star dithered onto it keeps its core; a dead or flickering one takes its same-colour neighbours' median. A defect stays on its pixel while dithering or drift moves the sky, so a pixel counts only when it stands out in most of the lights, and never when its neighbours rise as a star's do. `auto` does this only without `--dark`, `on` also with one, for the defects a master leaves. Without it, uncalibrated lights' hot pixels reach the stack through the first frames, which online rejection cannot reject, and demosaicing turns them into coloured dots that registration can take for stars. The report gives `defect_pixels` and `defect_frames`. |
| `--reintegrate` | off | After stacking, integrate every admitted frame again in three passes with leave-one-out rejection. This removes trails in the first frames, which online rejection cannot revisit, and refits each frame's background against an integration of the best twenty. |
| `--scratch-directory <PATH>` | output's directory | Where `--reintegrate` keeps each admitted frame's registered image. Stacking writes it while it has the frame in hand, so reintegration reads it back rather than calibrating, demosaicing and resampling the frame again. It needs 4 bytes per pixel per channel for every admitted frame, about 29 GB for 92 ASI2600MC frames, and deletes them when done; a frame that does not fit is prepared afresh each pass. Ctrl-C or SIGTERM stops the run at the next frame and deletes them too (a second interrupt quits at once, still deleting them), exiting with status 130; files left by a run that was killed outright are deleted by the next run in the same directory. |
| `--drizzle 1\|2\|3\|4` | off | Drizzle the stack onto a grid this many times finer than the reference, as WBPP's DrizzleIntegration does. Turns on `--reintegrate`, whose rejection decides which pixels each frame drops. The output is the drizzled image with the reference WCS scaled to it. Bayer frames drop each photosite into its own colour, with no demosaicing. `2` recovers detail in undersampled, well-dithered data. |
| `--drizzle-drop-shrink` | `0.9` mono, `1.0` Bayer | Each drop's side as a fraction of a source pixel. Smaller drops keep more detail and need more dithered frames to fill the grid. |
| `--workers`, `--pipeline-memory-mib` | derived, `4096` | Frames read and prepared at once, and the memory for them; the memory also bounds the bands of rows of every frame `--reintegrate` reads back at once. |
| `--no-gpu`, `--cpu-pass` | off, none | Keep every pass on the CPU and never open the GPU, or keep only the passes named (`defects`, `debayer`, `detection`, `resampling`, `accumulator`, `integration`, comma-separated); otherwise stacking's heavy passes run on the GPU when there is one. `seiza color` accepts the same GPU flags. |
| `--gpu-concurrency`, `--gpu-trace` | `3`, off | Frame calls the GPU takes at once, and printing each GPU call's timings to stderr. |
| `--gpu-memory-mib` | the device's | Device memory to plan the GPU's work in. The live accumulator stays on the CPU when it, the frames on the GPU for it and the frame calls' buffers would need more than three quarters of it. |

On 126 one-shot-color frames of M45 (ASI2600MC, 173 mm, no calibration) the
defaults stacked in 1 min 17 s at 2.7px FWHM with a 0.10px red-blue offset;
the settings before these options gave 3.1px and 0.42px. Adding
`--reintegrate` took 2 min 2 s, and `--drizzle 1` 2 min 39 s at 2.69px FWHM
and an SNR of 356. After WBPP had registered the same frames, PixInsight's
ImageIntegration and DrizzleIntegration took 6 min 54 s and 2 min 41 s for
2.67px and 357. On 62 undersampled H-alpha frames (ASI2600MM, 300 mm),
`--drizzle 2` took 43 s in all and gave 1.78px FWHM in reference pixels;
WBPP's whole run with 2x drizzle took 7 min 5 s for 1.77px. Seiza's times are
from an i5-1340P laptop (4 performance and 8 efficiency cores, 16 threads,
32 GB) in October 2026.

### Expanding beyond the reference frame

Use `--reference expand` for overlapping mosaic panels or a dithered stack whose
outer edges should be retained:

```sh
seiza stack Ha-panel1/*.fits Ha-panel2/*.fits Ha-panel3/*.fits --reference expand --output Ha-mosaic.fits --reintegrate
```

A planning pass anchors on the clearest frame and connects the others through
shared stars, building a canvas that holds their transformed footprints. Panels can
connect through other panels: the third panel need not overlap the first. Each
frame registers against the footprints of the panels already placed rather than
the whole canvas, so a narrow overlap still finds its stars, and a frame that
starts a new panel is refit against every star it covers so that its far edge
meets its other neighbors. A match must pair more stars than the best of the
transforms searched would pair by chance, which keeps a frame that shares no sky
from joining through a coincidence in a crowded field. Frames are then integrated
on that grid, once each; the planning template contributes no exposure. The
anchor's pixel scale and orientation stay fixed, and its WCS reference pixel moves
when the canvas extends left or upward. Uncovered pixels remain undefined.

A similarity places every frame. With the default `--registration-model
quadratic`, a frame lying mostly within one panel is then warped onto it, which
follows lens distortion across a meridian flip. A panel's first frame cannot be
warped that way, since a quadratic fitted through a narrow overlap would
extrapolate across the rest of the frame; it is taken from the anchor's side of
the pier whenever such a frame connects, and placed through the lens (see
below). Expansion does not feather or blend mosaic seams, or drizzle
(`--drizzle` and `--bayer-drizzle` are refused). Sessions
may differ in rotation, as when the camera comes off between nights, so
expansion admits a frame at any rotation; matching, residual, and scale checks
guard against a false placement, as does requiring a match to pair a share of
the stars it claims to overlap, so the hot pixels that uncalibrated lights of
another target share with the mosaic cannot place them. A frame that cannot join
the connected mosaic, such as one of another target or cloud, is rejected with
the reason, and the
mosaic is built from the rest; expansion fails when nothing joins the
anchor, an input changes or cannot be read, or the canvas would exceed
`--max-canvas-megasamples`. One set of
calibration masters applies to every frame, so lights whose sessions need
different flats are better stacked per session first and their stacks
expanded. The default overlap threshold is 5%; change it with
`--min-overlap`. Panels planned for 10% overlap land short of 10%: on a 2x2
IC 1318 mosaic (6248x4176, 450 mm) planned that way, they overlapped by 0.088
to 0.094, so a 10% threshold kept only the anchor's panel, while 5% joined all
four at a median 0.1 px RMS. The threshold does not guard against chance
matches; the star-pairing, scale and residual checks above do. When lights do
not join, a note on stderr and in the report's
`configuration.expansion_planning_notes` says how many, and how close those
that matched a panel came to the threshold. Expansion bounds each search by
the panels it matches; `--max-registration-drift` and `--max-registration-drift-fraction` are
intended for ordinary fixed-reference stacks.

On a three-panel Rosette mosaic from an ASI6200MM (9576x6388), three
uncalibrated 900 s lights per panel and filter, panels shot months apart with
about 15% overlap, each of H-alpha, O-III and S-II connected all 9 frames into a
canvas of 165 to 167 megapixels in 31 to 44 s on a 16-core desktop, peaking at
6.0 to 6.7 GB. Frames registered to their own panel at 0.05 to 0.09 px RMS,
including those from the other side of the pier, where a similarity alone left
1.2 px, and the panels' first frames met through the lens at 0.08 to 0.15 px.

Lens distortion bends the shared edges of two panels opposite ways, and a
similarity placing a panel's first frame through their overlap fits its middle
and misses its ends: by up to 17 px on that mosaic, so stars there appeared
twice, or, under live rejection, as rings whose cores the first panel's frames
had rejected from the second's. So once every frame has joined, one radial
distortion, two coefficients about an axis that may lie off the sensor's
center, is fitted where the panels' first frames overlap. This shared lens
model assumes one optical train and sensor geometry; combining different
rigs or focal reducers needs separate distortion models. Before fitting, joined
frames are checked for conflicting camera identity, dimensions, binning, pixel
size and focal length. Pixel size and focal length allow 1% rounding tolerance.
A known conflict skips the shared lens fit and keeps the first registration
plan. Missing metadata allows fitting but reports that optical compatibility
could not be verified. Filters and dates do not cause conflicts. Notes appear
on stderr and in `configuration.expansion_planning_notes` in the JSON report.
Matching metadata is a compatibility check, not proof of identical optics.
Where the anchor's
`FOCALLEN` and `XPIXSZ` give the focal length, each panel is placed on the
anchor's tangent plane by a rotation of the camera rather than a similarity,
which a wide mosaic's panels, several degrees apart, need. When the lens
explains what the similarities leave, the canvas is planned again on
undistorted grids: only the anchor and each panel's first frame are read again,
the other frames register by the stars planning kept, and every frame's warp
leads through the lens onto its own pixels. The JSON report's
`configuration.expanded_lens` records the measured axis, coefficients, largest
shift, and the overlaps' residual with and without it; the stack is planned on
similarities alone when no lens is found, as for a single panel. Lens correction
changes the pixel grid, so the original WCS (including SIP distortion) is
omitted; plate-solve the output mosaic to obtain valid astrometry. Without a
lens correction, the anchor WCS is retained with its reference pixel shifted. On the
Rosette mosaic's 215 S-II lights, the lens moved stars up to 63 px about an axis
127 px off center, and the panels' first frames met at 0.12 px RMS through it,
against 11.8 px fitted without it. A night that had sat 2 to 3 px off its
neighbors along both seams then matched the full stack to 0.07 px (95th
percentile, 0.13 px at most), and the rings were gone. A panel's brightness
is measured only on stars that pair star with star across the overlap;
measured where a similarity put them, the O-III panels came out 1.8 and 3.5
times too bright, where pairing star with star shows they differ by under 25%.
When too few stars pair up, a panel keeps its own scale and only its
background is matched, since noise from skies on different nights says nothing
of transparency. On synthetic 3x3 mosaics with 12% overlaps, every corner
landed within 0.11 px of its true position.

`--normalization none` preserves the input intensities; it does not disable
registration or canvas expansion. The default `local-background` matches panel
brightness through their overlaps: panels differing only in gain and offset
match exactly. A gradient within one panel cannot be told apart from the sky,
though, and the canvas does not feather. Uncalibrated lights also carry their
vignetting into every overlap. In a synthetic test, a gradient spanning a third of the sky level across one panel
left steps of about a tenth of it, and shifted the panel beyond by about half of
it. For seamless backgrounds, stack each panel on its own, flatten each panel
stack with `seiza background` (see [Background extraction](#background-extraction)),
and expand the flattened panel stacks into the mosaic.

`--reintegrate` uses the same expanded canvas and recorded transforms.
Planning reads every frame before the integration pass. Each frame being
registered and the reintegration scratch, four bytes per sample per frame, both
span the full canvas. A saved stack context keeps the panels, with the stars planning
matched each by (its first frame's), and the lens planning measured, so a
frame added after reopening registers exactly as it would have before saving.

For S/H/O, run expansion separately for each filter and register the resulting
three mosaics together afterward. Independent expansion runs may choose different
anchors and dimensions. `--reference expand` cannot currently be combined with
`--reference-image`; use the shared-reference option when a fixed common grid is
required.

Raw calibration sequences can be integrated into reusable masters first:

```
seiza master bias bias/*.fits --output master-bias.fits --report master-bias.json
seiza master dark dark/*.fits --bias master-bias.fits \
  --output master-dark.fits --report master-dark.json
seiza master flat flats/*.fits --bias master-bias.fits \
  --dark-flat master-dark-flat.fits --output master-flat.fits \
  --report master-flat.json
```

Bias and dark construction uses a two-pass, leave-one-out sigma-clipped mean.
Flat construction calibrates and normalizes each input once, then uses a
scratch-backed temporal median/MAD sigma-clipped mean. This improves rejection
of moving stars that overlap a sensor pixel in multiple sky flats without aligning the
stars or smoothing the sensor response. Two-frame sets are averaged without
rejection, regardless of master kind. More frames and enough star motion are
needed to distinguish stars from persistent flat response. Faint halos can
remain in small or noisy sets even with a clean majority; saturated or other
contamination present in most inputs can be retained more strongly.

Flat scratch storage uses OS temp and needs about four bytes per input sample;
tile memory is bounded at 64 MiB in addition to the image and master buffers.
The temporary file is removed after completion, cancellation, or failure.
Dimensions, channels, CFA layout, and available camera,
binning, gain, offset, temperature, filter, and dark-exposure metadata are
checked before incompatible frames can be mixed. The completion message and
JSON `configuration` report the actual integration and rejection method. The
`rereads_inputs` field describes source pixel reads during integration (not
the separate SHA-256 provenance pass): false for flats, true for bias/dark.
An all-rejected flat pixel falls back to its temporal median; bias/dark pixels
use the unclipped mean. Both the report and warning identify that fallback.
The report's `inputs` contains only the combined frames with their own sample
counts; `skipped_inputs` retains the identity and reason for every excluded
frame, so a metadata mismatch cannot shift another frame's statistics.

Calibration, registration, normalization, online delta-sigma rejection, and
integration all operate on linear `f32` samples. `--preview` is an optional
display-only stretch and never feeds pixels back into the stack. Incoming
frames are admitted atomically: incompatible images, weak registrations,
excess transform drift, low overlap, or implausible normalization leave the
existing additive stack unchanged. The optional `--report` JSON records
SHA-256 identities for every source and calibration master, the complete
configuration, and the ordered accepted/rejected disposition ledger. Without
`--reference-image`, the reference light is integrated, so it counts toward `accepted_frames` but has
no entry in the per-frame `frames` ledger; that ledger lists only the frames
pushed after it, with each one's weight when frames are weighted. FITS and report outputs are
published atomically after they are complete.

To align separate filter channels to the same image, use the same reference
for each run:

```sh
seiza stack red/*.fits --reference-image common.fits --output red.fits
seiza stack green/*.fits --reference-image common.fits --output green.fits --reintegrate
seiza stack blue/*.fits --reference-image common.fits --output blue.fits --reintegrate
```

The common reference is demosaiced if needed, but the lights' bias, dark, and
flat are not applied to it; supply a calibrated reference when needed. Its
prepared channel count must match the lights (mono for separate mono filters,
RGB for RGB or Bayer lights), so mono filter lights cannot align to an RGB
image, nor one-shot-color lights to a mono luminance master. Its pixel scale
must match the lights' to within 4%, so 2x2-binned lights cannot align to an
unbinned image. Normalization and noise weighting use a
calibrated light from the current channel, selected by `--reference auto`
(the default) or `--reference first`. With `--reference auto`, if the chosen
light does not align to the shared image, the next-best lights are tried in
turn, up to five in all. With `--reference first`, a first light that does
not align stops the run; list another light first to use it instead. The shared image supplies geometry only,
so default normalization remains enabled without matching different filters
to one brightness scale. Only listed lights count toward the stack.
With this option, the report's `reference_integrated` is false
and `frames` contains every listed light, even when the reference is one of
them. The alignment image retains its SHA-256 identity in `reference`, and
`normalization_reference` identifies the selected channel light.

Mono inputs produce a one-plane linear FITS stack. Three-plane FITS/XISF inputs
remain RGB, while raw one-shot-color frames with `BAYERPAT` are calibrated in
their native CFA sampling before demosaicing. Star detection uses a temporary
luminance view, but registration is applied to all three channels and
normalization is estimated per channel. The output is an unstretched
three-plane `float32` RGB FITS; `--preview` writes an RGB display stretch.

The registration search is explicitly bounded at the center of the reference
frame. Its effective default is whichever is larger: 256 pixels or 15% of the
reference frame's larger dimension. Configure those components with
`--max-registration-drift` and `--max-registration-drift-fraction`. Increase
either for a sequence with larger dithers or crop-origin offsets; lower values
make the expected motion constraint tighter. Set the fractional component to
zero when a strict pixel-only limit is desired. Light frames may have different
dimensions: valid samples are mapped onto the reference frame's fixed grid, pixels
outside a source crop remain masked, and `--min-overlap` controls how much
usable overlap is required for admission.

Meridian-flipped frames are handled automatically. The default 10-degree
rotation gate measures deviation from the nearer of the reference orientation
and its 180-degree counterpart, so a measured transform of 179.3 degrees has a
0.7-degree admission deviation. The complete measured transform—not a header
flag—is then used to rotate and resample the incoming pixels onto the reference
grid before normalization and integration. Accepted-frame diagnostics and the
JSON report retain the full measured rotation.

`--flat` accepts an integrated master flat in the light frame's raw sampling.
For legacy masters, `--bias` removes the flat's pedestal before normalization.
Masters built by `seiza master` carry FITS calibration-state headers, so a
bias-subtracted dark or calibrated flat is not subtracted twice. Each light is
checked against what the masters' headers record: gain, offset, binning, the
dark's sensor temperature (`CCD-TEMP` or `SET-TEMP`) and the flat's filter and
optics. A master dark that records no temperature is used with a warning.
Planar RGB flats are normalized independently per channel. CFA flats remain
one-channel and are applied before debayering. See the
[stacking design](https://github.com/theatrus/seiza/blob/main/docs/design/image-stacking.md)
for the live API, rejection semantics, and PSF Guard integration boundary.

### Color composition

`seiza color` consumes mono `float32` FITS stacks. It can emit an RGB
FITS, a quick-look PNG, or both:

```
seiza color rgb --red r.fits --green g.fits --blue b.fits \
  --output rgb.fits --preview rgb.png

seiza color lrgb --luminance l.fits --red r.fits --green g.fits --blue b.fits \
  --luminance-weight 1.0 --output lrgb.fits --preview lrgb.png

seiza color lrgb --luminance l.fits --red r.fits --green g.fits --blue b.fits \
  --luminance-mode super --output super-lrgb.fits --preview super-lrgb.png

seiza color narrowband --ha ha.fits --oiii oiii.fits --sii sii.fits \
  --palette sho --output sho.fits --preview sho.png

seiza color narrowband --ha ha.fits --oiii oiii.fits \
  --palette foraxx-hoo --preview foraxx-hoo.png
```

Direct palettes are `sho`, `soh`, `hso`, `hos`, `osh`, `ohs`, and `hoo`.
Dynamic palettes are `foraxx-sho` and `foraxx-hoo`. The default independent
0.1%/99.5% percentile scaling is intended for fast visual matching; use
`--normalization none` for masters whose backgrounds and scales are already
matched. Foraxx additionally requires those unnormalized samples to lie in
`[0, 1]`; keep the default percentile mode for sensor-unit masters. Foraxx
working channels use a median/MAD midtones transfer; tune it with
`--foraxx-target-median` and `--foraxx-shadows-clip`.

LRGB defaults to linear luminance replacement. `--luminance-mode super`
instead sets the target luminance to `L + R + G + B` after normalization while
preserving RGB chromaticity. Its linear output may exceed one.
`--luminance-weight` applies only to replacement mode. `color rgb` accepts the
same flag: `--luminance-mode super` scales the triplet to a synthetic
`R + G + B` luminance with no luminance stack, marked `SEIZACLR='SUPER-RGB'`.

Linear RGB/LRGB, super-LRGB, super-RGB, and direct-palette FITS files carry
`SEIZATRF='LINEAR'`.
Foraxx follows its published stretched-channel formula and carries
`SEIZATRF='DISPLAY'`; previews therefore do not pretend it is a linear stack.
Input dimensions and path roles are validated, and WCS comes from the command's
reference channel. See the [color-composition
design](https://github.com/theatrus/seiza/blob/main/docs/design/color-composition.md).

By default, non-reference filter stacks are star-registered and resampled onto
L for LRGB, R for RGB, or H-alpha for narrowband. The command reports matched
stars, RMS, drift, and rotation, rejects RMS above 2 pixels, and uses the normal
256-pixel-or-15% drift bound. Configure those gates with
`--max-registration-rms`, `--max-registration-drift`, and
`--max-registration-drift-fraction`; use `--no-register` for masters already
registered to the same reference.

Star detection and resampling run on the GPU when there is one. `seiza color`
accepts the same GPU flags as `seiza stack`: `--no-gpu` keeps every pass on
the CPU and never opens the GPU, and `--cpu-pass detection,resampling`,
`--gpu-concurrency`, `--gpu-memory-mib` and `--gpu-trace` work as they do for
stacking.

## Persistent worker

Applications performing repeated solves can keep a catalog and blind index
open behind a newline-delimited JSON-RPC 2.0 process:

```
seiza worker --data data --index data
```

The same protocol can adapt local image paths to a queued `seiza-server`
(self-hosted, or the hosted instance at [seiza.fyi](https://seiza.fyi)):

```
seiza worker --server http://solver-host:8080
```

Remote mode defaults to a compact grayscale PNG upload. Use
`--server-upload fits` to stream the original FITS file and preserve headers,
or `--server-timeout SECONDS` to change the default five-minute deadline.
Bearer authentication uses `--server-token` or `SEIZA_SERVER_TOKEN`. See the
[versioned wire contract](https://github.com/theatrus/seiza/blob/main/docs/design/worker-protocol.md)
for request and response details.

## Threads

`seiza` uses every core. To leave some for other work, cap it with
`RAYON_NUM_THREADS`:

```
RAYON_NUM_THREADS=4 seiza stack lights/*.fits --output stack.fits --reintegrate
```

The cap sizes the pool that does the parallel part of detection, solving,
stacking, reintegration and master building. Stacking and reintegration
also prepare a few frames at once on threads of their own, whose serial
steps run beside that pool. Unless `--workers` says otherwise, stacking uses
half as many of them as the cap, and at most six.

## Use with N.I.N.A.

seiza speaks ASTAP's CLI contract: set N.I.N.A.'s plate solver to ASTAP
and point the ASTAP path at the seiza binary (a copy named `astap.exe`
also works). On Windows, use the **Seiza Catalog Setup** Start-menu shortcut
installed by the MSI; on every platform, `seiza setup` installs a usable
selection into a directory ASTAP mode discovers automatically. For a manual
layout, run `seiza download-data prebuilt --output <directory>` and point
`SEIZA_CATALOG_DIR` at that directory. Advanced `SEIZA_STAR_DATA` and
`SEIZA_BLIND_INDEX` file-or-directory overrides remain available. Hinted and
blind-slot solving both work; see the
[ASTAP-compatible mode design](https://github.com/theatrus/seiza/blob/main/docs/design/astap-mode.md).

## Use with Siril

seiza also speaks astrometry.net's `solve-field` CLI contract, and answers
Siril's Windows `bin/bash` launch wrapper itself (no cygwin needed). Run
`seiza install-solve-field --dir <dir>`, point Siril's astrometry.net
directory preference at that directory, and Siril's normal astrometry.net
solving works unchanged, including SIP distortion orders. Catalogs resolve
the same way as ASTAP mode. Siril reports PSF amplitudes rather than photometric
flux, so seiza automatically re-measures star flux from the source image
next to the star table when present — see the
[solve-field mode design](https://github.com/theatrus/seiza/blob/main/docs/design/solve-field-mode.md)
for details.

## Datasets

The quickest route is the prebuilt, SHA-256-verified set hosted at
downloads.seiza.fyi (Tycho-2 and Gaia solver tiles, the blind index, the
unified object catalog, minor bodies, the
Tycho/Bright Star/GCVS/WDS/IAU identifier sidecar, and a nightly-refreshed
transient list):

```
# Running this by itself prints the recommended prebuilt and setup routes:
seiza download-data

seiza download-data prebuilt --output data
seiza download-data prebuilt --output data --file objects.bin --file transients.bin
# The optional ~9 GB Gaia G≤20 catalog is explicit; pair it with the blind index
# for deep blind solving.
seiza download-data prebuilt --output data \
  --file stars-deep-gaia20.bin --file blind-gaia16.idx
# The optional Gaia photometry catalog for colour calibration is explicit too.
seiza download-data prebuilt --output data --file stars-gaia-photometry.bin
# So are distances to deep-sky objects (or: seiza setup --object-distances).
seiza download-data prebuilt --output data --file object-distances.bin
```

The other `download-data` subcommands acquire upstream source material for
custom catalog builds; they are not required for normal Seiza use.

Historical satellite elements use the same resolver as applications: a nearby
entry in the durable cache first, then the rolling Seiza mirror, with IAU
SatChecker as the on-demand fallback. Operators can prewarm one or more epochs
without teaching an application which provider to call:

```
seiza download-data satellite-history \
  --epoch 2025-10-17T12:00:00Z 2025-10-18T12:00:00Z \
  --cache /var/lib/seiza/satellite-mirror/cache
```

`seiza build-data satellite-manifest` converts that cache into a validated,
content-addressed publication tree. The complete cron, S3 publication order,
backfill, retention, and public verification procedure is in the
[satellite mirror runbook](https://github.com/theatrus/seiza/blob/main/docs/SATELLITE_MIRROR.md).
The mirror publisher uses `--origin` so its scheduled bucket is fetched from
IAU SatChecker rather than resolved from the mirror it is updating.

For an interactive selection, run `seiza setup`. This is also the command
offered by the Windows installer. It presents lightweight, Gaia, deep-blind,
optional G≤20, and every-catalog presets, then delegates to the same verified
prebuilt downloader. Every preset includes object search, Solar System objects,
and active transients.

The downloader reads one standard bundle from `/data/v4/manifest.json` and
caches its immutable, content-addressed files by SHA-256 before copying the
requested selection into the same flat local output directory. The shared
platform cache can be overridden with `SEIZA_CACHE_DIR`. The optional G≤20
catalog is hosted alongside the bundle but is never included in a bare
`download-data prebuilt`; name it with `--file` or choose the relevant setup
preset. It never combines catalogs from different hosted bundle versions.
Previously released `/data/` and `/data/v2/` paths remain frozen for classic
v1 and v0.4.1/v0.5 readers. The historical `/data/v3/` probe used by v0.4.0
remains reserved and may be absent; those readers retain their existing
fallback behavior.

Library integrations can use `seiza-download` directly for async, automatic
cache management. The raw catalog commands below are implemented by the
separate `seiza-sources` crate so applications do not inherit Gaia/VizieR/MPC
source-acquisition behavior.

Building from primary sources stays supported for custom depths, epochs,
or tile granularity — note the Gaia TAP download alone can take many
hours:

```
seiza download-data tycho2 --output raw/tycho2
seiza download-data star-identifiers --output raw/star-identifiers
seiza build-data tycho2 --input raw/tycho2 --output stars-lite.bin \
  --identifier-index stars-lite.ids.bin \
  --identifier-sources raw/star-identifiers

seiza download-data gaia --output raw/gaia        # Gaia DR3 via TAP, resumable
seiza build-data gaia --input raw/gaia --output stars-gaia.bin

seiza download-data objects --output raw/objects
seiza download-data curation --output raw/curation --commit <git-sha>
seiza build-data objects --input raw/objects --output objects.bin \
  --curation-dir raw/curation \
  --source-manifest objects.sources.json
seiza download-data transients --output raw/transients
seiza build-data transients --input raw/transients --output transients.bin
seiza download-data mpc --output raw/minor-bodies
seiza build-data minor-bodies --input raw/minor-bodies --output minor-bodies.bin
seiza download-data object-distances --output raw/distances   # resumable
seiza build-data object-distances --input raw/distances \
  --objects objects.bin --output object-distances.bin
```

`build-data object-distances` keys distances to the stable IDs in the
`objects.bin` it is given, so rebuild it whenever the object catalog changes.
When several sources measure an object, the first in this order wins: Hunt &
Reffert 2024 (clusters, Gaia DR3), Cosmicflows-4 (galaxies), Chornay & Walton
2021 and Gonzalez-Santamaria et al. 2021 (planetary nebulae), Foster & Brunt
2015 (HII regions), Ranasinghe & Leahy 2023 (supernova remnants), Zucker et
al. 2020 (named molecular clouds), the parallax of the star lighting a van den
Bergh nebula, Harris 1997 (globular clusters), SIMBAD's distance
measurements, Stanghellini & Haywood 2010 (planetary nebulae), Hilton &
Lahulla 1995 (Lynds clouds), the WISE HII region catalogue (kinematic), and
last the redshift of a galaxy (H0 = 75 km/s/Mpc, as Cosmicflows-4 uses).

The optional curation directory is a pinned local checkout; the builder never
fetches it. Its `curation.json` records repository, commit, and schema version.
Each `objects/<id>.toml` file owns the corrections, relations, selections,
exceptional outline remappings, notes, and structured evidence for one
canonical target. Normally named OpenNGC outlines are associated directly
during upstream ingestion and do not require curation documents. The optional
source manifest records the output hash and size, metadata coverage counts,
source URLs, curation revision, and hashes of every raw catalog and curation
file used.
The optional identifier sidecar provides memory-mapped exact TYC/HIP/HR/HD/
SAO/FK5 lookup plus exact and prefix search over IAU proper names,
Bayer/Flamsteed names, GCVS variables, and WDS double-star designations. It
does not change the solver's compact star tile file.

To publish new transient and Solar-system data without rebuilding the object or
star catalogs, put only the replacement `transients.bin` and
`minor-bodies.bin` in a directory and roll forward each complete manifest:

```shell
seiza build-data manifest --dir next-dynamic \
  --base-manifest current-v2.json \
  --version catalog-bundle-v2-YYYY-MM-DD --output next-v2.json
seiza build-data manifest --dir next-dynamic \
  --base-manifest current-v4.json \
  --version catalog-bundle-v4-YYYY-MM-DD --output next-v4.json \
  --artifact-dir next-v4-artifacts
```

The v2 output retains flat keys for released v3-object readers. The v4 output
uses content-addressed artifact keys and stages both uncompressed and
maximum-compression zstd artifacts in one upload-ready tree. The manifest
retains the uncompressed artifact for old v4 readers; new readers
stream-decompress into the normal uncompressed mmap cache. Both commands
require the resulting bundle to contain every required catalog.

Normal catalog opens do not perform exhaustive validation. Validate any seiza
star tile, identifier sidecar, blind index, object catalog, or minor-body
catalog explicitly when required; the file format is auto-detected:

```
seiza catalog validate --data stars-lite.ids.bin
seiza catalog validate --data stars-lite.bin
seiza catalog validate --data objects.bin
```

FITS and XISF files are read natively through
[seiza-fits](https://crates.io/crates/seiza-fits) and
[`seiza-xisf`](../seiza-xisf/README.md). Both provide typed linear pixels, an
automatic MTF stretch for u8 detection and previews, a linear normalized
representation for f32 detection, and RA/DEC hints from FITS-compatible
metadata. PNG, JPEG, and TIFF continue to use the Rust `image` decoders.
Commands that save linear images write FITS by default and monolithic
`Float32` XISF when the output path ends in `.xisf`.

### Parameterized display stretching

`seiza stretch` applies an explicit reusable `seiza-stretch` model to a linear
mono or RGB FITS/XISF image and writes a display-referred PNG, JPEG, or TIFF:

```text
seiza stretch stack.fits --output preview.png percentile-asinh \
  --black-percentile 0.01 --white-percentile 0.995 --strength 10

seiza stretch stack.fits --output preview.png auto-mtf \
  --target-median 0.2 --shadows-clip -2.8

seiza stretch stack.fits --output preview.png median-anchor \
  --target-median 0.25 --black-point-sigma 2.7 --hdr-amount 0.3

seiza stretch color.fits --output preview.png layered \
  --target-background 0.15 --contrast-recovery 0.6 \
  --dynamic-range-compression 0.5 --saturation-amount 0.4

seiza stretch color.fits --output preview.png \
  --color-strategy luminance-preserving asinh \
  --black 0 --white 1 --strength 8

seiza stretch stack.fits --output preview.png ghs \
  --stretch-factor 4 --local-intensity -1 --symmetry-point 0.35 \
  --protect-shadows 0.1 --protect-highlights 0.8
```

`median-anchor` places the background median exactly on `--target-median`,
with the black point `--black-point-sigma` robust sigmas below the median
(`--no-black-clip` uses the image minimum), then optional `--curves-boost`,
`--hdr-amount`/`--hdr-knee` highlight compression, and `--normalize`.
`layered` stretches luminance with the same curve, the background on
`--target-background` (default 0.15) and the black point set by
`--aggressiveness` (0 to 1, default 0.5). `--scale-separation` (8 to 1024
pixels, default 256) splits large structure from detail;
`--contrast-recovery` (0 to 1) keeps detail on bright structure,
`--dynamic-range-compression` (0 to 1) shrinks star cores, and
`--saturation-amount` and `--saturation-boost` (both 0 to 1) lift colour,
scaled by each pixel's value unless `--no-lightness-mask` is given. It carries
colour by luminance, so it takes `linked` or `luminance-preserving`. Other
model subcommands are `identity`, `linear`, and explicit `mtf`. The `ghs`
subcommand exposes the deterministic GHS parameters; informed automatic
selection can be layered over it later. RGB may use `linked`, `unlinked`, or
`luminance-preserving` analysis/application.
Stretching is never applied to linear stack output unless this command or a
library caller explicitly requests it. See the
[stretching design](../docs/design/stretching.md).

## License

Apache-2.0
