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

- `calibrate_and_stretch(image, wcs, gaia, options, stretch)` runs
  `calibrate_color_image` and then the display `StretchStack`. A
  statistics-driven stage such as Auto-MTF resolves from the calibrated
  image, so a linked stretch keeps the balance. An unlinked stage would pull
  each channel to its own target median and undo the calibration; that is
  the caller's choice. The stretch runs whether or not calibration did, so
  the caller always gets an image, with the outcome saying which.

The outcome serializes with `status` (`calibrated` or `skipped`) beside the
calibration's fields or the `reason` and `detail`, so hosts can keep it with
the image.

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
That is exact for a linear (TAN) solution. A crop at `(x, y)` is a
translation by `(−x, −y)`, chained with `AffineTransform::then`.

SIP distortion is defined on offsets from `CRPIX` in the solved image's own
pixel axes. A translation leaves those offsets unchanged, so SIP terms carry
through a crop. A turn or a change of scale would need the polynomials
re-expanded, so `transform_wcs` refuses a SIP solution with any mapping that
is more than a translation rather than return a solution that is subtly
wrong.

## Bindings and the CLI

The C ABI carries both steps as `seiza_color_calibrate_image_json` and
`seiza_color_calibrate_and_stretch_json`, which work on the caller's buffer
in place, and `transform_wcs` as `seiza_wcs_transform_json`. Python has
`calibrate_color_image`, `calibrate_and_stretch` and `Wcs.transformed`.
`seiza color-calibrate --display out.png [--stretch JSON]` writes the
stretched result beside the calibrated linear FITS; with `--display` a
calibration that cannot run is reported instead of failing the command.
