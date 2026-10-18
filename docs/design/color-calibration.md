# Photometric colour calibration

`seiza-stacking::color_calibration` sets an RGB image's channel gains from
Gaia DR3 star colours. Each isolated, unsaturated Gaia star is measured in R,
G and B by aperture photometry; its instrumental colours, `−2.5 log10(R/G)`
and `−2.5 log10(B/G)`, are fitted against Gaia BP−RP; and the fits, evaluated
at the white reference's colour (the Sun's, BP−RP 0.82, by default), give the
gains that render such a star neutral. Background neutralization then offsets
each channel so the sky is the same grey in all three. No filter or sensor
curves are needed: the stars measure the camera's own response.

## Calibrate, then stretch

`calibrate_color` fits from stars already placed on the image, and fails when
it cannot. Hosts that compose a picture want something else: a step that
always gives them a picture, calibrated when it can be. Two functions do that.

- `calibrate_color_image(image, wcs, gaia, options)` places the Gaia stars
  through `wcs`, fits, and applies the gains in place. It never fails. When
  calibration cannot run it leaves the image as it was and returns
  `ColorCalibrationOutcome::Skipped` with a typed reason:

  | Reason | When |
  | --- | --- |
  | `no_solution` | no `wcs` was given |
  | `no_catalog` | no Gaia photometry catalog is installed, or it could not be opened |
  | `no_stars` | no Gaia star with a colour falls on the image |
  | `too_few_stars` | fewer than `minimum_stars` survive a fit, or too few clean stars to measure the FWHM |
  | `not_rgb` | the image is not three-channel |
  | `invalid_options` | the options are out of range |
  | `fit_failed` | the fits ran but gave no usable (finite, positive) gains |

- `calibrate_and_stretch(image, wcs, gaia, options, stretch)` runs
  `calibrate_color_image` and then the display `StretchStack`. A
  statistics-driven stage such as Auto-MTF resolves from the calibrated
  image. An unlinked stage would pull each channel to its own target median
  and undo the calibration, so once calibration has run, every stage whose
  colour strategy is `Unlinked` runs linked instead, and
  `CalibratedStretch::linked_stages` lists them. Linked and
  luminance-preserving stages run as asked. When calibration cannot run,
  every stage runs as asked. A host that wants each channel stretched on its
  own when there is no calibration, as an uncalibrated preview usually is,
  passes those stages as `Unlinked` and gets both behaviours from one call.
  The stretch runs whether or not calibration did, so the caller always gets
  an image, with the outcome saying which.

The outcome has two JSON forms. Rust's serde form, which hosts written in
Rust store, is flat and snake_case: `{"status": "calibrated", "gains": [...],
"red_fit": {...}, ...}`, the `ColorCalibration` fields beside `status`, or
`{"status": "skipped", "reason": "too_few_stars", "detail": "..."}`. The C ABI
nests and camelCases it, with a version: `{"schemaVersion": 1, "status":
"calibrated", "calibration": {"gains": [...], "redFit": {...}, ...}}`, or
`"reason"` and `"detail"` in place of `"calibration"`. Python gives the C
form's keys as attributes of `ColorCalibrationOutcome` (`status`, `reason`,
`detail`, `calibration`).

## Where the Gaia colours come from

`GaiaColors` takes the offline catalog (`stars-gaia-photometry.bin`, G ≤ 15)
already open, its path or directory (`None` searches the standard catalog
locations), or sources the caller has, such as an archive query at J2016.0
with proper motions and the observation's epoch. The catalog's positions are
already at its own epoch, so no proper motion is applied to them; sources
flagged unreliable (RUWE ≥ 1.4) keep no colour but still count as neighbours,
so they disqualify the calibrators they crowd. The cone searched covers the
image's corners through its solution, with 2% to spare.

Calibrators default to two magnitudes brighter than the faintest star given,
so with the G 15 catalog they stop at G 13: every neighbour bright enough to
disturb a calibrator's aperture is then in the catalogue. A narrow field can
have fewer than 20 such stars and is reported as `too_few_stars`.

## Solutions for composed images

A host rarely solves the image it calibrates. A stack is registered to a
reference frame, may be turned half round for a north-up layout, drizzled,
and cropped to the area every channel covers. `transform_wcs(wcs,
source_to_image)` carries the reference frame's solution through that pixel
mapping: with `A` the mapping's matrix and `t` its translation, the new
solution keeps `CRVAL` and has `CD' = CD · A⁻¹` and `CRPIX' = A · CRPIX + t`.
That is exact for a linear (TAN) solution.

The mapping counts pixels from 0 with centres on whole numbers, as `crpix`
does inside Seiza (a FITS header's `CRPIX` is 1 more), whatever origin the
caller's WCS came in with. In that convention:

- a crop at `(x, y)` is a translation by `(−x, −y)`;
- a half turn of a `w` × `h` image takes `x` to `w − 1 − x` and `y` to
  `h − 1 − y`;
- drizzle at scale `s` puts a source pixel centre `x` at `s · x + (s − 1) / 2`,
  because the output's pixel edges fall at `s` times the source's
  (`seiza-stacking/src/drizzle.rs`), and binning by `b` is the same mapping
  with `s = 1 / b`. `AffineTransform::scale_about_pixel_centres(s)` builds
  it. A mapping of `s · x` alone is half a pixel off for a 2× drizzle.

Steps chain with `AffineTransform::then`.

SIP distortion is defined on offsets from `CRPIX` in the solved image's own
pixel axes. A translation leaves those offsets unchanged, so SIP terms carry
through a crop; a fitted mapping within `1e-9` of the identity matrix counts
as a translation. A turn or a change of scale would need the polynomials
re-expanded (`f'(d') = A · f(A⁻¹ d')`), which `transform_wcs` does not do: it
refuses a SIP solution with any other mapping rather than return a solution
that is subtly wrong.

## Bindings and the CLI

The C ABI carries both steps as `seiza_color_calibrate_image_json` and
`seiza_color_calibrate_and_stretch_json`, which work on the caller's buffer
in place, and `transform_wcs` as `seiza_wcs_transform_json`. Python has
`calibrate_color_image`, `calibrate_and_stretch` and `Wcs.transformed`.
`seiza color-calibrate --display out.png [--stretch JSON]` writes the
stretched result beside the calibrated linear FITS. With `--display`, a
calibration that cannot run, from a header with no solution, Gaia colours
that cannot be fetched or opened, or too few stars, does not fail the
command: it prints the reason, writes both images uncalibrated, and writes
the skip to `--report`. In the C ABI, `gaia_json` overrides `catalog_path`,
and `observationEpoch` applies to `gaia_json` sources only.
