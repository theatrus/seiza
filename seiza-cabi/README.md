# seiza-cabi

## Project analysis

`seiza_analyze_project_json(request_json, error_out)` analyzes mono, raw Bayer and RGB
FITS/XISF projects without producing a stack. Example request:

```json
{"inputs":["lights"],"samples":262144,"filter":"Ha","maxFrames":100,"metricsOnly":true}
```

Only `inputs` is required. Inputs may be files, nonrecursive directories or
UTF-8 text lists. The response is the core analysis report with camelCase
keys (`schemaVersion`, `algorithmVersion`, `options`, `frames`, `groups`,
`timings`, `warnings`), with schema/algorithm version 3. `metricsOnly` defaults
to false; true skips registration, normalization, sampling and replay. Frames
have separate `measurementStatus` and `depthStatus`, with `error` and
`depthError` respectively. Groups expose `measuredFrames`, `depthFrames` and
`normalizedNoiseFrames`; native quality medians use all measured frames.
An empty `groups` array means no frames were measured. Use separate invocations
per panel/pointing; processing outcomes are not quality verdicts. Invalid requests return null and set `error_out`.
`seiza_analysis_html(report_json, error_out)` renders that camelCase result as
self-contained HTML. Both return owned strings freed with `seiza_string_free`.
These calls write no source images or report files. The Python counterparts are
`analyze_project` and `analysis_html`.

See [the analysis contract and limitations](../seiza-analysis/README.md).

A C ABI over Seiza for native application front-ends. It is the single, shared
successor to the near-identical `seiza-cabi` crates that previously lived in the
[`seiza-win`](https://github.com/theatrus/seiza-win) (.NET) and
[`seiza-mac`](https://github.com/theatrus/seiza-mac) (Swift) repositories, and it
exposes the **superset** of what both apps need.

## What it exposes

- **FITS / XISF / raster rendering** — `seiza_rendered_image_open`,
  `..._open_with_rgb_stretch`, `..._open_with_stretch_config` (see the
  **Parameterized stretch** entry below), `..._width`, `..._height`,
  `..._metadata_json`, `..._free`.
- **Both pixel byte orders** — `seiza_rendered_image_rgba` (macOS / CoreGraphics)
  and `seiza_rendered_image_bgra` (Direct2D / WinUI), each with a `_length`
  companion. RGBA is canonical; the BGRA view is computed on first request and
  cached, so a consumer only ever pays for the order it uses.
- **Filtered reduced images** — a nonzero maximum dimension uses area averaging,
  with weights for partial edge pixels, instead of selecting single pixels.
  RGBA8 and RGBA16 average display values with alpha-weighted colors. Interactive
  file previews average linear samples before processing; live-stack previews
  also exclude uncovered samples. These are display reductions, not changes to
  stored image or stack data. A zero limit leaves full-size file render buffers
  unchanged. Because stretch and background fitting are nonlinear, a bounded
  interactive preview can still differ from a reduced full-resolution render.
- **Native 16-bit export pixels** — the parallel
  `seiza_rendered_image16_open*` API returns a separate
  `SeizaRenderedImage16` with borrowed native-endian RGBA `uint16_t` samples.
  FITS stretch stacks quantize directly from their final `f32` result to `u16`,
  and 16-bit PNG/TIFF raster inputs retain their component precision. The
  `_rgba_length` result counts `uint16_t` elements, not bytes. A separate owner
  keeps routine RGBA8 previews from allocating both formats.
- **Parameterized stretch** — `seiza_rendered_image_open_with_stretch_config`
  takes a serialized `seiza-stretch` `StretchConfig` (JSON) and renders a FITS
  or XISF image
  through the full GHS/MTF/percentile pipeline. It also accepts a non-empty
  config array for an ordered `f32` stack, or an object with `stretch`, optional
  `sample_domain`, optional `background` correction, optional `deconvolution`,
  and optional `interactive_preview` mode. Background fitting and
  subtraction/division run first on linear input samples, followed by
  deconvolution, sample-domain mapping, and then the display stretch.
  Interactive previews bound those samples before expensive processing while
  committed renders remain full resolution. The two most recent prepared
  preview buffers are cached by file identity, maximum dimension, and background
  configuration; stretch and deconvolution edits reuse the same corrected linear
  pixels. New processing capabilities remain in their core crates; this shim
  only marshals JSON in and pixels out.
- **Background extraction** — `seiza_background_fit` creates a compact opaque
  model from interleaved linear mono or RGB `float` samples. Callers can inspect
  its borrowed diagnostics JSON, render it into a caller-owned buffer, or apply
  subtractive/divisive correction in place before freeing the model. A second
  correction entry point accepts a strength from zero to one. Optional settings
  use serialized `seiza-background` `BackgroundConfig` JSON, including
  polynomial, radial-basis, and automatic model selection, so the ABI stays
  stable as model options grow. The JSON config can carry normalized ellipses
  or polygons projected from solved catalog bounds, including stored OpenNGC
  contours. Channels of a colour image fitted one at a time can each pick a
  different automatic surface, which leaves a colour cast toward the edges;
  `seiza_background_select_shared_model_json` takes their models and returns
  the `BackgroundConfig` to fit every channel again with, or the JSON `null`
  when they share no choice.
- **Color crop diagnostics** — `seiza_color_crop_report_json` reports the
  region a crop would keep across aligned channels and what each channel
  covers of the shared grid, including any channel sitting far enough from the
  others to look like a pointing error. It reads the caller's buffers in place
  and returns owned JSON.
- **Colour from mono stacks** — `seiza_color_combine_rgb_json`,
  `..._lrgb_json`, `..._narrowband_json` and `..._narrowband_matrix_json`
  read aligned FITS or XISF mono stacks from the paths in a JSON request, compose them
  as `seiza color` and Python's `combine_*` do, and write the colour image as
  32-bit float FITS, or XISF for a `.xisf` name. RGB and LRGB take a
  `luminanceMode` (`super` for a luminance of R + G + B, or L + R + G + B);
  narrowband takes one of the nine palettes, Foraxx included, and the matrix
  call takes the caller's own SII, H-alpha and OIII weights for each output
  channel. Each takes the same `normalization` and `crop` settings, and the
  output keeps the first stack's WCS, moved by any crop. They do not register
  the channels to each other. The owned JSON gives the label written to
  `SEIZACLR`, whether the samples are linear or display-referred, the region
  of the grid kept, and the crop report when a crop was asked for. A
  `SeizaCancelSignal` stops the call between reads.
- **Light deconvolution** — `seiza_deconvolve_in_place` applies the same
  conservative damped Richardson-Lucy operation as the Rust and Python APIs to
  caller-owned linear mono or interleaved RGB `float` samples. The synchronous
  call retains no pointers and reports validation failures through `error_out`.
- **Live stacking** — `seiza_live_stacker_create` starts from caller-provided
  calibrated linear mono/RGB samples, while `seiza_live_stacker_open_fits`
  retains its ABI name but decodes raw linear FITS or XISF and optionally
  applies integrated bias, dark, and flat masters.
  `seiza_live_stacker_open_fits_on_canvas` starts another filter's empty stack
  on an existing expanded stack's grid (including a reopened context). It
  reuses dimensions, panel geometry, lens correction and WCS, and builds
  normalization from that filter's own calibrated lights. Push all lights
  explicitly. The source handle is borrowed only during construction; the
  returned handle owns its geometry. Bin factors must match. The canvas never
  grows, so plan the complete desired field first; samples outside are cropped.
  The Python twin is `canvas.open_fits_on_canvas(...)`; CLI users save and reuse
  contexts with `--save-canvas` and `--reference-canvas`.
  `seiza_live_stacker_open_fits_with_reference_image` aligns a stack to an
  image shared by every filter's stack, as the CLI's `--reference-image` does,
  so their outputs share one pixel grid and WCS. One of the filter's lights,
  the channel reference, sets normalization and noise weighting; the stack
  starts empty, so it is pushed like every other light, and the state JSON
  reports `referenceIntegrated` false. Outputs and contexts refuse to replace
  the reference image.
  `seiza_live_stacker_open_fits_with_reference_image_candidates` takes a JSON
  array of channel references, best first, and uses the first that aligns to
  the reference image, as the CLI's `--reference auto` does. Its optional
  selection JSON names the light it used and why each earlier one was
  skipped. `seiza_live_stacker_open_fits_expanding` takes a JSON array of
  overlapping lights, anchor first, and plans a canvas that holds every one
  that connects to it, as the CLI's `--reference expand` does: mosaic panels,
  or a dithered stack whose edges should be kept. It reads every light while
  planning, checking a `SeizaCancelSignal` before each, and starts empty, so
  every light is then pushed, the anchor included.
  `seiza_live_stacker_expanded_lens_json` gives the lens distortion planning
  undid, or `null`, and `..._expansion_planning_notes_json` what planning
  noted along the way, starting with how many lights did not join, if any,
  and how close they came to `minimum_overlap_fraction`. Unless the options
  set that, a light joins sharing 0.05 of its footprint with the canvas, as
  with the CLI's `--reference expand`: panels planned for 10% overlap land
  nearer 9%. `seiza_live_stacker_pipeline_memory_json` gives what preparing
  each frame needs, for planning memory around a pipelined push: on an
  expanded canvas a frame is costed at its own footprint, not the canvas.
  Array and path-based pushes
  return owned admission JSON. Borrowed
  mean/coverage/rejection views support copy-free live display, snapshots add
  variance, and `seiza_live_stacker_finish` moves the final accumulator into an
  immutable result without cloning its full-frame buffers.
  `seiza_live_stacker_save_context` atomically checkpoints a still-live handle;
  `seiza_live_stacker_open_context` restores it later with its original
  reference, calibration, online moments, counters, and source ledger intact.
  `seiza_live_stacker_open_context_with_gpu` restores it under other GPU
  options, such as `{"enabled": false}` when a user turns the GPU off; the
  configuration fingerprint stays the same. With null or empty GPU options,
  it keeps the options the stack was saved with.
  `seiza_live_stacker_state_json` exposes an authoritative resume identity,
  `..._set_calibration_fits` swaps session masters atomically between batches,
  and `..._render_preview` renders from a bounded sample of the live mean
  without making a full-frame snapshot. `..._export_snapshot` copies only one
  full-frame mean for non-destructive FITS/XISF output on a worker thread.
  `seiza_live_stacker_reintegrate` rereads every admitted frame and returns a
  new snapshot integrated with leave-one-out rejection, which removes trails
  from the reference and warm-up frames that online rejection kept. It
  reports all three passes through a progress callback, accepts a
  `SeizaCancelSignal`, and leaves the live stack unchanged;
  `reintegrationUnavailable` in the state JSON says when a stack cannot be
  replayed. Calling `seiza_live_stacker_retain_frames_for_reintegration`
  before pushing frames keeps each admitted frame's registered image in a
  scratch directory as it is stacked, so reintegration reads it back instead
  of preparing the frame again, with the same result.
  `seiza_live_stacker_reintegrate_drizzled` also drizzles the
  frames onto a grid 1 to 4 times finer, as WBPP's DrizzleIntegration does,
  and returns a `SeizaDrizzleResult` (image, weight, scale, and
  `seiza_drizzle_result_write_fits` with the WCS scaled to the grid) along
  with the reintegrated snapshot.
  `seiza_stack_integrate_registered_json` gives the same three-pass
  rejection to FITS or XISF frames calibrated, registered and normalized
  elsewhere, onto one grid: it reads each file once into scratch and takes
  optional per-frame weights, a cancel signal and the same progress callback.
  `mosaicPanels` optionally assigns each input a panel ID, repeated for
  observations of the same panel (at most 256 distinct IDs). It matches
  additive background planes and tapers residual differences across overlaps;
  inputs must already share a photometric scale. The call returns a
  `SeizaStackSnapshot` with an optional JSON report of what each frame kept.
  `seiza_live_stacker_open_prepared_fits` opens a stack from a FITS or XISF
  frame the host has already calibrated and debayered, keeping its header
  cards for the outputs, so a host can correct a frame between calibration and
  registration; later frames go in as arrays. An accepted frame's
  `diagnostics` carry its `mapping` onto the reference grid (the core's own
  versioned JSON, for a host to keep beside the frame), and its per-channel
  `noise` and `weight`, which are empty when frames are weighted equally.
  `seiza_live_stacker_reference_noise_json` and
  `seiza_live_stacker_reference_mapping_json` give the same for the reference
  frame, whose weight is 1; `seiza_frame_noise_json` measures any linear
  buffer's noise with the same estimator.
- **Photometric colour calibration** — `seiza_gaia_photometry_cone_json` fetches
  Gaia DR3 photometry for a field from the ESA archive, and
  `seiza_gaia_photometry_catalog_cone_json` reads it from the local catalog
  `seiza_catalog_setup` installs; `seiza_color_calibrate_json` fits per-channel
  gains and background offsets for an interleaved RGB buffer from its FITS WCS
  keywords and those Gaia rows, and `seiza_color_calibration_apply` applies
  them in place through the core, as Python's `ColorCalibration.apply` does.
  The options also set the colour range of the stars the fits use, their
  outlier rejection, the fewest stars a fit accepts and the faintest
  calibrator, and the result lists every star it measured.
  `seiza_color_calibrate_image_json` places the Gaia stars itself, from rows
  or the local catalog, through the image's WCS JSON, and applies the gains
  in place; `seiza_color_calibrate_and_stretch_json` then display-stretches
  the buffer with a stretch configuration. Neither fails when calibration
  cannot run: the JSON outcome gives `status` `skipped` with a `reason`
  (`no_solution`, `no_catalog`, `no_stars`, `too_few_stars`, `not_rgb`,
  `invalid_options`, `fit_failed`) and leaves the pixels uncalibrated. Once
  calibration has run, an `unlinked` stretch stage is linked instead and
  listed in `linkedStages`. `gaia_json` overrides `catalog_path`.
  `seiza_wcs_transform_json` carries a solved frame's WCS through a pixel
  mapping (registration, a turn, drizzle or binning, a crop) to the image
  made from it; the mapping counts pixels from 0 with centres on whole
  numbers.
- **Sensor defects without a dark** — `seiza_stack_detect_defects_json`
  learns a sensor's hot and dead pixels from a JSON array of dithered light
  paths, calibrated with optional bias, dark and flat masters, as the CLI's
  `--defects` does. Its schema-1 JSON holds `defects` (`width`, `height`,
  `frames`, `pixels`, and for steady hot pixels `excess` at
  `exposure_seconds`) in the form the stack options take: put it under
  `"defects"` in any open's `options_json`, and every light has those pixels
  repaired before debayering and registration, a steady hot pixel by
  subtracting its excess scaled to the light's exposure, as a dark would,
  and the rest from their same-colour neighbours' median. Options JSON
  (`sigma`, `persistence`, `maximum_frames`) is optional.
- **Automatic stack reference** — `seiza_stack_choose_reference_json` scores
  an ordered JSON array of FITS/XISF paths using the native stacking algorithm.
  It chooses the flattest background among frames with nearly the best star
  quality, without calibrating or modifying any source. Its owned schema-1
  JSON returns `referenceIndex`, the unchanged input `referencePath`, a
  `ranking` of the scored inputs' indices, best first, and a parallel
  `scores` array. Pass the first few ranked paths to
  `seiza_live_stacker_open_fits_with_reference_image_candidates` so another
  light can stand in when the best does not align to a shared reference
  image. Unreadable or unscorable inputs have null scores;
  a batch with no scoreable frame fails the call. Each score contains `stars`,
  `medianStarArea`, `background`, `backgroundVariation`, and `score`, measured
  on half-resolution luminance. The concurrency argument bounds simultaneous
  source reads/scoring; zero uses one. This synchronous call has no cancellation
  callback. Free its result or error using `seiza_string_free`.
- **Calibration orchestration** — `seiza_probe_frame_json` reads FITS/XISF
  metadata without decoding pixels, `seiza_calibration_plan_json` applies the
  core sensor/optics/exposure/temperature/proximity/coherence rules, and
  `seiza_calibration_build_master_json` atomically publishes a two-pass
  sigma-clipped bias, dark, or flat master from raw paths while reporting any
  metadata outliers it set aside. It can mask stars out of sky flats, set
  aside darks that caught stray light, and keep a flat build's temporary
  pixels in a directory the host chooses. Long builds accept the thread-safe
  `SeizaCancelSignal` owner. A planner request's `tolerances.lightRotation`
  plans a light's flats by an angle the host measured for it, or at no angle;
  `seiza_live_stacker_set_light_rotation` and `seiza_live_stacker_light_rotation`
  set and read the same for a stack's gate, since `SeizaMatchTolerances` is a
  fixed layout with no room for it.
- **Screening calibration frames** — before a master is built,
  `seiza_calibration_measure_dark_json` and `seiza_calibration_measure_flat_json`
  read one raw FITS or XISF frame each into a small JSON measurement, which a
  host keeps and passes back whenever the set changes.
  `seiza_calibration_screen_bias_json`, `..._screen_darks_json` and
  `..._screen_flats_json` judge one set and return a verdict per frame: a
  `grade` (`ok`, `suspect` or `poor`) and the `findings` behind it, each with
  its `code`, measured `value` and the `limit` it reached. Flats also get a
  verdict on the master they would make. Each limit is a `{suspect, poor}`
  pair the request may replace. The verdicts say what the pixels show, never
  why.
- **Residual flat responses** — `seiza_calibration_build_residual_flat_json`
  takes crops of several lights cut at one place on the detector, writes the
  light most of them lose into the caller's buffer as a response (1 where
  nothing is corrected, never gaining more than `maximumGain`), and returns
  its evidence as JSON. `seiza_calibration_validate_residual_flat_response`
  checks a stored response, and `seiza_calibration_apply_residual_flat_response`
  divides one out of a caller-owned light in place at a pixel offset. The host
  shows that the dip stays fixed on the detector first; these calls do not say
  what it is.
- **Constellation figures** — `seiza_constellation_figures_json` returns every
  stick figure (its star pairs and label positions) with the 697 stars its
  lines use, for a host that draws figures its own way, and
  `seiza_constellations_project_json` projects them into a solved image as
  pixel polylines clipped to it, with a label position for each. The figures
  are the Constellation Lines dataset (Marc van der Sluys, CC BY 4.0):
  anything that shows them must carry the `attribution` both calls return.
- **Flat sessions and their dust** — a flat set reduces to one response:
  `seiza_flat_bin_frame` bins each flat (`seiza_flat_bin_factor` gives the
  factor) and `seiza_flat_build_response_json` combines them into the
  caller's buffer. `seiza_flat_compare_json` compares two sessions'
  responses (the broad change, the dust, and the spots that came, went or
  moved); `seiza_flat_change_map` makes the change between them, which
  `seiza_flat_apply_change_map` divides out of a light shot before it, or
  multiplies into one shot after. `seiza_flat_lights_background` combines a
  few lights' sky, and `seiza_flat_fit_change_json` says which side of a
  change they were shot on, or `seiza_flat_choose_response_json` which of
  several responses they follow; `seiza_flat_combine_verdicts_json` merges
  a session's filters' verdicts. For the dust itself,
  `seiza_dust_find_session_json` stacks a session's flats and counts each
  feature across them, `seiza_dust_frame_json` checks one flat against that
  map, `seiza_dust_find_json` reads one plane, `seiza_dust_find_changes_json`
  compares two sessions' dust, and `seiza_dust_match_features_json` pairs
  two lists of features. Planes are caller-owned `float` buffers (several of
  one size laid end to end); results are the core's own JSON, which it also
  caches, so their keys are snake_case; options are `DustOptions` JSON, any
  field left out taking its default. `seiza_flat_encode_plane` and
  `seiza_flat_decode_plane` keep planes compactly in a host's cache.
- **Stack-depth analysis** — `seiza_live_stacker_measure_depth` measures the
  borrowed live accumulator without a snapshot, and `seiza_checkpoint_depths`
  returns the doubling ladder at which a host should sample a finite batch.
- **Measured stars and sensor tilt** —
  `seiza_stars_detect_luma_u16_json` measures a caller-owned mono u16 frame,
  while `seiza_stars_detect_path_json` opens FITS/XISF directly and measures
  its linear luminance without routing through display RGBA. Both return the
  same schema-1 JSON: source dimensions, stars with HFR/FWHM/PSF measurements,
  a 3×3 cell grid for the parallelogram diagram, and the derived corner
  tilt/curvature verdict. Optional `triangleAngleDegrees` adds a native-owned
  triangle result without repeating detection: a circular center and annulus,
  three complete 120-degree screw sectors, their counts/median HFRs, the
  minimum-star readiness policy, and the differential tilt verdict. Angles
  use image coordinates (`0` top, positive clockwise) and are normalized over
  `[0, 360)`. Path options can infer a detector preset from
  `FOCALLEN`/`FOCALLENGTH`/`FOCAL` and `XPIXSZ`; explicit caller choices win.
  PSF directions are normalized ellipse major-axis orientations over `[0, π)`
  and are advertised by `majorAxisOrientationsNormalized: true`.
  For focusing hosts, `seiza_stars_pixel_scale_arcsec` gives a rig's pixel
  scale from its focal length and pixel size, `seiza_stars_classify_pixel_scale`
  the `preset` for that scale, and `seiza_stars_recommend_detection_binning`
  the `detectionBinning` for a measured in-focus HFR.
- **RC-Astro external tools** — `seiza_rc_astro_locate` finds the CLI on
  `PATH` (null without an error when absent), `seiza_rc_astro_tool_schema_json`
  reads one tool's live contract (`rc-astro <tool> --json`: parameters with
  flags, types, ranges, defaults, plus CLI/model versions and license
  state) as schema-1 JSON, and `seiza_rc_astro_process_file_json` runs BXT,
  SXT, or NXT on an image file with parameter values keyed by schema name —
  flags change between CLI builds, so requests are built from the schema.
  Whole numbers satisfy float parameters and fraction-free floats satisfy
  int parameters, so bounds read from the schema round-trip; a `false`
  switch emits the CLI's `--no-` negation, overriding a true default. The
  run accepts a `SeizaCancelSignal` and a progress callback
  (`SeizaRcAstroProgressCallback`: the fraction complete plus a context
  pointer, on the calling thread), kills a child silent for ten minutes,
  and reports every written file, StarXTerminator's stars sidecar included.
  Include the schema's `cliVersion`/`mlVersion` in any cache key: a tool
  upgrade changes the output for identical inputs.
- **Plate solving** — `seiza_solve_image_json`, and
  `seiza_solve_image_with_limits_json`. Both start from what the file says
  about itself: a FITS or XISF image's own WCS (FITS cards, or a PixInsight
  solution in either `AstrometricSolution` namespace) is checked with a
  quick hinted solve; else a hinted solve searches around its `RA`/`DEC` or
  `OBJCTRA`/`OBJCTDEC` at a scale from its WCS, `PIXSCALE`, or `XPIXSZ`
  over `FOCALLEN`; else a blind solve runs, near that scale first (a
  raster's EXIF focal length gives one too). FITS and XISF stars are
  detected on an MTF stretch first, then on linear samples. When nothing
  solves but the file holds a WCS, that WCS is returned unchecked.
  `solvedFrom` says which: `header-checked`, `header-hint`, `blind`, or
  `header-unchecked`, which comes with `matchedStars` 0 and
  `falseMatchProbability` 1. The limits call gives up after a time limit in
  seconds or when a `SeizaCancelSignal` is cancelled, setting `error_out` to
  `solve stopped at its time limit` or `cancelled by the caller`. The JSON's
  `falseMatchProbability` is the chance that a wrong field would match as
  well, counting every candidate the search tried: a true field scores far
  below one in a billion, and a solve that scores above that is refused.
  The returned `wcs` counts pixels from 0, with each pixel's center on a
  whole number, and says so with `"pixelOrigin": 0`; add 1 to `crpix` for a
  FITS header, whose pixels count from 1. The image's middle is
  `((width - 1) / 2, (height - 1) / 2)`, and `centerRaDegrees` is the sky
  there. A WCS passed back in, as to `seiza_parallax_prepare_json`, may give
  `"pixelOrigin": 1` for FITS-style values.
  `objectPositions` lists the catalogued objects in the field (deep-sky
  objects, named stars, transients, comets and asteroids) through the same
  core code as Python's `field_objects`; each outline carries the `method`
  that made it and the `evidence` it rests on. `captureTime` is the time
  that dates transients and places comets and asteroids.
- **Solving helpers** — `seiza_read_capture_time_json` reads when an image
  was taken (FITS or XISF `DATE-OBS`, `DATE-BEG` or `DATE-AVG`, or a photo's
  EXIF) without decoding its pixels. `seiza_read_photo_metadata_json` gives a
  photo's EXIF capture time, GPS position and focal length, and, given the
  decoded size, the pixel-scale ranges to try. `seiza_wcs_fits_header_json`
  turns a WCS, as the solve returns it, into FITS header `cards` (a JSON
  object, in header order, that `seiza_color_calibrate_json` takes as its WCS
  keywords) and header `text`, as Python's `Solution.fits_header_cards` and
  `fits_header_text` do.
- **Parallax videos** — `seiza_parallax_prepare_json` makes what `seiza
  parallax-video` makes from a JSON request: an image for StarXTerminator to
  split (through `rc-astro`) or its starless and stars files, an optional WCS
  (as `seiza_solve_image_json` returns it; without one the image is
  blind-solved), and the command's options in camelCase, with frame sizes
  such as `"1080p"`, `"4k"`, `"1080p-portrait"` or `"4k-image"` (the
  image's own shape at that longer side). `framing` `"fit"` shows all of an
  image whose shape differs from the frame's, with black beside it, where
  `"fill"`, the default, cuts its longer side; `crop` (`[x, y, width,
  height]`) films part of the image, every other image point counting in
  it. An unknown field is an error. It returns an opaque `SeizaParallax`, released with
  `seiza_parallax_free`, and reports each step as JSON to a
  `SeizaParallaxEventCallback`. `distanceFocus` is where the nebula's
  distance is measured; `focus` only aims the camera, so a new destination
  never changes the scene's depth. `planes` (`[{mask, distanceParsecs,
  name}]`) lifts regions of the image, each marked by a mask image, onto
  planes at their own distances; `autoPlanes` lifts the catalogued nebulae
  at distances of their own, and the clouds of dust in front of the nebula,
  each dark nebula moving with the cloud it lies in, and
  `distanceOverrides` gives objects' distances by name. `seiza_parallax_summary_json` says what it
  found, its `planes` among it, with `fit`: each zoom and pan as asked and as used, after the camera
  was fitted to stay inside the image, so an app can explain a change to the
  framing it asked for, and `inside`, false when some frames still show past
  the image's edge. `seiza_parallax_reconfigure_json` films the same
  scene again from a JSON object of camera and output settings (focus, start,
  dolly, pan, rotation, zoom, tour stops, glide, length, frame rate, size,
  quality, labels), fitting the camera anew. Settings left out take their
  defaults, not the first video's. A stop can set `pullBackToWholeImage: true` to keep its selected
  `focus` while widening toward whole-image framing: dolly 0 and zoom 1
  frame the image centre, and increasing either moves the view back toward
  the target. The default remains point-centred for existing tours. Tour
  fitting adjusts stops independently instead of magnifying the whole
  tour when one stop needs a wider fit. It loads, splits, solves and looks up
  nothing: the new `SeizaParallax` shares the prepared scene and the
  catalogue labels with the first, which stays valid and can draw frames at
  the same time; the scene is freed with the last handle. Fields that change
  the scene (distances and their sources, `distanceFocus`, star placement,
  galaxies, depth planes, dust) are an error that says to prepare again. Its
  `SeizaCancelSignal` stops it. `seiza_parallax_plan_tour_json` plans a
  tour of the catalogued objects in the field (`autoTour`) and returns it
  as JSON to edit, each target's stop titled with its name; its `tour`
  goes back into a request as it is, and its `focus` as `distanceFocus`.
  It plans from the catalogue alone. Once a scene is prepared,
  `seiza_parallax_plan_scene_tour_json(video, settings_json, error_out)`
  plans on it instead, as an automatic tour of that scene would: each stop
  frames its target at the depth of the plane the scene shows it on,
  carrying that plane's `depthParsecs`. It takes `{"size", "autoTour"}`
  (the video's own frame if `size` is absent) and returns the same JSON,
  whose `tour` goes into `seiza_parallax_reconfigure_json` as it is.
  `tourTitles` shows each stop's title low in the frame as the camera
  drifts through it, and `tourLoop` ends the tour where it began so the
  video loops. Frames then come three ways:
  `seiza_parallax_render_frame` draws any frame into the caller's own
  buffer, with a row stride and `SEIZA_PIXEL_FORMAT_RGB8`, `_RGBA8` or
  `_BGRA8`, so a platform encoder's pixel buffer (a `CVPixelBuffer`, a Media
  Foundation sample) can take it directly, from several threads at once;
  `seiza_parallax_render_frames` hands every frame in order to a
  `SeizaParallaxFrameCallback`, the hook for an external encoder, which
  returns nonzero to stop; and `seiza_parallax_write_video_json` writes an
  MP4 through ffmpeg or numbered PNG frames. Both whole-video calls take a
  `SeizaCancelSignal` and return `1` done, `0` stopped, or `-1` failed.
  Labels use the Inter typeface, embedded; an app shipping this library must
  ship its licence, `seiza-draw/fonts/LICENSE-Inter.txt`.
- **Catalog setup** — `seiza_catalog_status_json` and `seiza_catalog_setup`
  (with a progress callback). Preset `3` installs only the Gaia photometry
  catalog that colour calibration reads. The install path delegates to
  `seiza-download`'s `materialize_with`; the shim carries no download logic.
- **Satellite tracks** — predictions, from orbital elements, of the
  satellites that crossed a solved image while the shutter was open; never
  detections in the pixels. A `SeizaSatelliteCatalog` holds the elements:
  `seiza_satellite_catalog_open` reads an OMM JSON or TLE file,
  `..._from_omm_json` and `..._from_tle_text` parse text,
  `..._fetch_celestrak` downloads CelesTrak's current set (or reuses a copy
  under two hours old, falling back to an older good copy when a refresh
  fails), and `..._resolve_json` takes `{"time", "cacheDir"}` and picks the
  provider for that time: CelesTrak when recent, else a cached copy, the
  Seiza satellite mirror or IAU SatChecker. The two network calls block and
  take a `SeizaCancelSignal`, which stops them at once. CelesTrak blocks
  clients that download too often, so keep to one cache directory.
  `..._info_json` reports the record count, source, retrieval time, provider,
  cache state, cache path and any warning. `..._tracks_json` takes the
  image's WCS JSON (as the solve returns it, with `pixelOrigin`), its width
  and height, and a request giving `start` (Unix seconds or RFC 3339),
  `durationSeconds` or `end`, the observer (`latitude`, `longitude`,
  `altitudeM`, or `observerItrf` in metres) and optional sampling settings;
  it returns each track's satellite, samples and path inside the image, with
  counts of the records considered, failed and skipped as stale (over seven
  days from the exposure by default). Release the catalog with
  `seiza_satellite_catalog_free`. Python's `SatelliteCatalog` is the twin.
- **Versions** — `seiza_core_version` gives the library release.
  `seiza_stacking_output_version` and `seiza_stretch_output_version` give the
  `seiza-stacking` and `seiza-stretch` crate versions, not Seiza releases,
  whose stacking and stretch output this library reproduces; key
  cached stacks, masters and stretched images on those, since they move only
  when output changes. Python's `__version__`, `STACKING_OUTPUT_VERSION` and
  `STRETCH_OUTPUT_VERSION` are the twins.
- **Memory** — `seiza_string_free`.

Rendered-image metadata includes input and display histograms plus the requested
and resolved sample domains used for that render.

### Render sample domains

`sample_domain` describes the numeric scale presented to the stretch pipeline;
it is render policy, not frame-to-frame stack normalization. Samples already in
the zero-to-one working domain use an identity mapping. The examples below show
the optional field only; it sits beside the required `stretch` field in a
processed render request.

```json
{"sample_domain":{"type":"unit-linear"}}
```

Physical camera, calibrated, or stacked samples can resolve a robust display
range before stretching:

```json
{
  "sample_domain": {
    "type": "physical-linear",
    "normalization": {
      "type": "robust-percentile",
      "black_percentile": 0.001,
      "white_percentile": 0.999,
      "max_analysis_samples": 200000
    }
  }
}
```

Within `physical-linear`, robust-percentile with those three values is used
when `normalization` is omitted; omitted robust fields take those values.
Resolution samples complete pixels and pools their finite values into one
linked range for mono or RGB, so it does not silently perform a per-channel
color balance. Non-finite samples are excluded from analysis and remain
non-finite during affine mapping, preserving live-stack coverage masks. A
caller can lock a known physical range instead:

```json
{
  "sample_domain": {
    "type": "physical-linear",
    "normalization": {
      "type": "explicit-range",
      "black": 512.0,
      "white": 16384.0
    }
  }
}
```

The mapping always runs after optional physical-domain background correction
and deconvolution and immediately before the first stretch stage. Both the
requested policy and its resolved mapping (including the black/white range for
physical samples) are retained in render metadata for provenance or a later
locked-range request.

Omission is backward-compatible. Ordinary file rendering and a live stack in
`PreparedOnly` input mode retain the historical `unit-linear` behavior. A live
stack in `CalibrateAndPrepare` mode defaults to `physical-linear` with robust
percentiles because its native mean remains in physical sample units. An
explicit `sample_domain` always overrides that live default.

Sample-domain mapping modifies only the temporary render buffer. Live means,
variances, SNR measurements, saved contexts, export snapshots, and written
FITS/XISF samples remain physical linear `f32`; neither the requested nor the
resolved display range becomes part of scientific stack state.

Use `seiza_rendered_image16_open_with_stretch_config` for a processed FITS
export, or `seiza_rendered_image16_open` for the default FITS/raster path. The
16-bit handle has its own width, height, RGBA, metadata, and free functions; do
not pass it to the RGBA8 accessors. Its RGBA pointer is aligned for `uint16_t`
and uses host byte order. Image encoders or platform image APIs must be told
that byte order when consuming the borrowed samples.

Background input and output buffers are row-major, pixel-interleaved `float`
samples with one or three channels. The model copies only compact samples and
coefficients, so the input buffer may be released after `seiza_background_fit`
returns. Pass null for both mask/configuration pointers to use automatic
defaults. Correction mode constants and the precise pointer/length contracts
are declared in the generated header.

A crop report takes parallel arrays of channel names and sample pointers on
one shared grid, and the mode `none`, `bounds`, or `inscribed`. It borrows
those samples for the call rather than copying them. The JSON carries the
mode, the grid size, the kept `region`, the retained fraction, the offset at
which a channel is flagged, and one entry per channel with its own covered
box, covered pixel count, center offset from the other channels, and
`off_center` flag:

```json
{"mode":"inscribed","grid":{"width":128,"height":128},
 "region":{"x":0,"y":80,"width":128,"height":48},
 "retained_fraction":0.375,"off_center_limit_pixels":32.0,
 "channels":[{"name":"SII","region":{"x":0,"y":80,"width":128,"height":48},
              "covered_pixels":6144,"center_offset_x":0.0,
              "center_offset_y":40.0,"center_offset_pixels":40.0,
              "off_center":true}]}
```

Deconvolution uses the same buffer layout and modifies the caller-owned input
in place. Supply a measured stellar PSF FWHM in pixels; the conservative values
are four iterations, a `0.35` blend, a `0.001` noise fraction, and a maximum
correction of `2.0`. The output remains linear and may contain samples outside
`[0, 1]`.

Stacking uses the same row-major, pixel-interleaved mono/RGB layout. Array
frames are copied during each synchronous call and may be released when it
returns; they must already be calibrated, debayered, and linear. FITS/XISF
pushes retain and apply the calibration masters loaded by the path
constructor; a non-zero dark exposure override requires a dark master path. A
handle created from an array accepts only linear-array pushes, and a handle
created from a path accepts both the configured FITS path and prepared linear
pushes. A saved context retains that input mode. A rejected frame is
represented by `accepted: false` disposition JSON and is not an ABI error. The
zero-copy live pointers are invalidated by the next push, finish, or free;
immutable snapshot pointers remain valid until snapshot free. Snapshot FITS
output refuses to overwrite any tracked light or calibration input.

When any master is active, the native stacker validates every reference and
pushed FITS/XISF frame before touching its pixels. Declared masters, known raw
`bias`/`dark`/`dark-flat`/`flat` roles, and frames already marked `BIASSUB`,
`DARKSUB`, or `FLATNORM` are rejected; unknown roles remain compatible with
legacy files. Bias/dark/flat masters must sensor-match every light, flats must
also match its optics, and darks must match temperature. A dark whose pedestal
was isolated with a usable bias may scale by exposure. An unscaled dark must
carry the same known positive exposure as each light. With no masters these
guards are a no-op, so deliberately preprocessed lights remain stackable.

How close a light's readings must be to the masters' is the stack's gate. A
host that widened a gate to choose its masters, for a rotator that scatters
more than two degrees or a flat taken from the nearest other angle on
purpose, gives the stack the same gate, or the light is refused when it is
pushed. Open the stack with `"match_tolerances": {"rotation_deg": 5.0}` in its
options JSON to judge the reference under it too; tolerances left out keep
their defaults. `seiza_live_stacker_set_match_tolerances` changes the gate for
the lights pushed from then on, and `seiza_live_stacker_match_tolerances`
reads it back. Each admitted frame keeps the gate it was admitted under, and
a reintegration judges it by that gate, so a host can widen the gate for one
light and narrow it again after. A saved context keeps every gate.

A host can ask those questions before pushing.
`seiza_live_stacker_compatible_calibration_json` names the masters a light
would keep and why each other one would be set aside;
`seiza_live_stacker_validate_calibration` answers yes (1) or no (0, with the
refusal in `error_out`), and returns -1 for an invalid argument. Both ask the
same questions, so validation passes exactly when nothing would be set aside.
Both take an optional `SeizaMatchTolerances`; null means the stack's own gate.
A non-null struct replaces that gate whole: a field whose
`SEIZA_TOLERANCE_HAS_*` flag is clear takes the built-in default, not the
stack's value. An override that is NaN, infinite, negative, or a rotation above
180 degrees also takes the default, as everywhere `SeizaMatchTolerances` is
accepted; 180 already accepts any angle.

```c
SeizaMatchTolerances wide = {0};
wide.known = SEIZA_TOLERANCE_HAS_ROTATION;
wide.rotation_deg = 180.0;
if (seiza_live_stacker_validate_calibration(stacker, &light, &wide, &error) == 1) {
    seiza_live_stacker_set_match_tolerances(stacker, &wide, &error);
    /* push the light, then narrow the gate again */
    seiza_live_stacker_set_match_tolerances(stacker, NULL, &error);
}
```

The authoritative state response is owned JSON and is sufficient for a host's
resume store:

```json
{
  "schemaVersion": 1,
  "coreVersion": "0.17.0",
  "configurationFingerprint": "64-lowercase-sha256-hex-characters",
  "width": 6248,
  "height": 4176,
  "channels": 1,
  "acceptedFrames": 17,
  "rejectedFrames": 2,
  "inputMode": "calibrate-and-prepare",
  "inputPaths": ["reference.fits", "master-bias.fits", "light-002.fits"],
  "referenceFrame": {
    "role": "light",
    "isMaster": false,
    "signature": {
      "camera": "ASI2600MM", "telescope": null,
      "width": 6248, "height": 4176, "channels": 1,
      "binningX": 1, "binningY": 1, "gain": 100, "offset": 50,
      "readoutMode": null, "bayerPattern": null, "filter": "Ha",
      "focalLengthMm": null, "rotationDeg": null,
      "exposureSeconds": 300.0, "cameraTempC": -10.0,
      "capturedAtUnix": 1785593045
    },
    "calibrationState": {
      "biasSubtracted": false,
      "darkSubtracted": false,
      "flatNormalized": false
    }
  }
}
```

`inputPaths` is the native ordered ledger, not a host reconstruction. The
fingerprint covers validated stack options, current calibration pixel content
and metadata, and input mode. It excludes the accumulator, counters, and paths,
so ordinary pushes do not change it; a calibration swap does. Context save/open
persists those fingerprint inputs and recomputes the same value on restore.
`referenceFrame` is native metadata for the immutable reference source, so a
resume host need not reopen a path that may have moved or disappeared.

Swap masters only between push batches:

```c
bool changed = seiza_live_stacker_set_calibration_fits(
    stacker, "session-2-bias.fits", "session-2-dark.fits",
    "session-2-flat.fits", 0.0, &error);
```

All files are loaded and the set is validated against the immutable reference
before calibration or the ledger changes. Null/empty paths clear that master;
all three absent clears the set. Already integrated frames are untouched. The
operation is unavailable on a stack created from prepared arrays.

For non-destructive full-resolution output, capture one compact immutable
owner while holding the live stacker's lock, then release that lock and write
on a worker:

```c
SeizaStackExportSnapshot *output =
    seiza_live_stacker_export_snapshot(stacker, &error);
/* ingestion may continue here */
bool written = seiza_stack_export_snapshot_write_fits(
    output, "live-stack.fits", &error);
seiza_stack_export_snapshot_free(output);
```

The capture copies only the finalized `f32` mean plus small headers, scalar
frame counts, and the source ledger. It does not clone variance, coverage, or
per-sample rejection arrays. Zero-coverage samples become `NaN`. The owner is
independent of the live stacker and may be transferred to another thread; do
not free it until its write returns.

For frequent display updates, render the live mean through the same stretch,
`sample_domain`, background, and deconvolution JSON as file rendering:

```c
SeizaRenderedImage *preview = seiza_live_stacker_render_preview(
    stacker, stretch_json, 1600, &error);
```

The maximum dimension is required and bounds copied linear samples before any
expensive processing. The implementation samples the borrowed accumulator
directly rather than making a full snapshot. Zero-coverage pixels stay masked
through statistics, background fitting, deconvolution, sample-domain mapping,
and stretching; they are transparent in RGBA. The returned image is independent
of the stack and uses the ordinary `SeizaRenderedImage` accessors/free function.

Stack options are serialized `seiza-stacking` `StackOptions`. Every nested
object accepts omitted fields from its defaults. For example, this disables
normalization/rejection while increasing the registration drift floor:

```json
{
  "registration": { "maximum_drift_pixels": 512.0 },
  "normalization": { "mode": "none" },
  "rejection": { "mode": "none" }
}
```

Local normalization is `{"mode":"local","options":{"tile_size":256}}`;
delta-sigma rejection is
`{"mode":"delta-sigma","options":{"low_sigma":3.0,"high_sigma":3.0}}`.
Unknown fields and invalid bounds are rejected rather than silently ignored.

For resumable app sessions, save and reopen the opaque processing state rather
than treating an integrated FITS result as a new reference:

```c
bool saved = seiza_live_stacker_save_context(stacker,
                                             "m31.seiza-stack", &error);
seiza_live_stacker_free(stacker);

stacker = seiza_live_stacker_open_context("m31.seiza-stack", &error);
char *decision = seiza_live_stacker_push_fits_json(stacker,
                                                   "light-042.fits", &error);
```

Saving does not consume or otherwise mutate the handle. Context publication is
atomic, and reopening validates the format version, dimensions, configuration,
payload checksum, and accumulator invariants before returning a handle.
Current writers use context version 2, which retains normalized reference and
master signatures plus the dark-scaling safety fact. Version-1 contexts still
open. If a migrated v1 context contains masters, pushes fail closed until the
caller reloads them with `seiza_live_stacker_set_calibration_fits`, because the
older checkpoint cannot prove their compatibility; no-master v1 contexts
continue normally.

Offering a whole session at once prepares several frames in parallel while
integrating in the order given, which is identical to offering them one at a
time but overlaps the reads with the registration:

```c
char *outcome = seiza_live_stacker_push_fits_pipelined_json(
    stacker, "[\"light-001.fits\",\"light-002.fits\"]",
    0 /* derive workers */, 0 /* default budget */,
    65535.0f /* scale declared-normalized XISF to 16-bit; 0 to leave as stored */,
    &error);
```

Every frame appears in the returned `frames` array in order, a path that could
not be read among them with `accepted` false and a `reason`; check `failed`
rather than reading an absent error as success. Raise the worker count when the
frames arrive over a network, since the library cannot tell a network mount
from a local disk.

Frame probing is genuinely metadata-only for both containers: it calls
`seiza_fits::read_header` or `seiza_xisf::read_header`, never `FitsFrame::open`.
The owned result has this stable shape (unknown signature values are `null`):

```json
{
  "schemaVersion": 1,
  "path": "dark-001.fits",
  "format": "FITS",
  "role": "dark",
  "rawImageType": "Dark Frame",
  "isMaster": false,
  "signature": {
    "camera": "ASI2600MM", "telescope": null,
    "width": 6248, "height": 4176, "channels": 1,
    "binningX": 1, "binningY": 1, "gain": 100, "offset": 50,
    "readoutMode": null, "bayerPattern": null, "filter": null,
    "focalLengthMm": null, "rotationDeg": null,
    "exposureSeconds": 60.0, "cameraTempC": -10.0,
    "capturedAtUnix": 1785593045
  },
  "calibrationState": {
    "biasSubtracted": false,
    "darkSubtracted": false,
    "flatNormalized": false
  }
}
```

Roles are `bias`, `dark`, `dark-flat`, `flat`, `light`, or `unknown`.
`SEIZAMST` takes precedence over `IMAGETYP`/`OBSTYPE`/`FRAME`; common master,
object, and dark-flat spellings are normalized. The signature maps only fields
used by `seiza-calibration` matching. FITS dimensions and XISF-synthesized
`NAXIS*` cards share the same result.

Pass probe records directly as the reference/candidates of a coherent plan:

```json
{
  "kind": "dark",
  "reference": {"path":"light.fits", "role":"light", "signature":{}},
  "references": [
    {"path":"light.fits", "role":"light", "signature":{}},
    {"path":"light-002.fits", "role":"light", "signature":{}}
  ],
  "dependencies": {"biasAvailable": true},
  "candidates": [
    {"path":"dark-001.fits", "role":"dark", "signature":{}}
  ],
  "minimum": 8,
  "tolerances": {
    "exposureSeconds": 0.05,
    "exposureFraction": 0.001,
    "darkTemperatureC": 3.0,
    "masterTemperatureC": 1.0,
    "rotationDeg": 1.0,
    "focalLengthMm": 1.0,
    "flatSessionSeconds": 86400
  }
}
```

Plan kinds are `bias`, `dark`, `dark-flat`, and `flat`. A `dark-flat` plan
uses the same exposure and temperature matching as a dark, but requires
`dark-flat` candidates; use the selected raw flat as its reference. Build the
chosen dark-flat inputs with master kind `dark`, then pass that master as
`dark` when building the flat. This keeps the master builder ABI compatible
while keeping dark flats distinct during discovery and selection.

`references` is optional for backward compatibility. When present it is the
complete, non-empty, path-distinct target set and must include an identical
copy of the primary `reference`; bias/dark/flat targets have role `light`, and
dark-flat targets have role `flat`. Every selected candidate must match every
target. `dependencies.biasAvailable` defaults false and must mean a usable
bias was actually built: only then may a dark's isolated current scale across
target exposures. Scalable targets and candidates all need known positive
exposures, and selected raw dark inputs must still be mutually exposure-
coherent so master construction can succeed.

Every tolerance is optional and defaults to `seiza-calibration`. The response
is `{schemaVersion,kind,minimum,ready,matchedPaths,selectedPaths,excluded}`.
`matchedPaths` is proximity-sorted; `selectedPaths` is the first coherent set
meeting `minimum`, or the first smaller cluster when none can. Exclusion
reasons are `role-mismatch`, `sensor-mismatch`, `exposure-mismatch`,
`missing-exposure`, `temperature-mismatch`, `optics-mismatch`, or
`outside-coherent-set`.

Raw master construction uses one synchronous JSON call:

```json
{
  "kind": "flat",
  "inputs": ["flat-001.fits", "flat-002.fits"],
  "output": "master-flat.fits",
  "bias": "master-bias.fits",
  "dark": "master-dark-flat.fits",
  "darkExposureSeconds": 2.0,
  "exposureSeconds": null,
  "rejection": {"lowSigma": 3.0, "highSigma": 3.0},
  "defectSuppression": {"lowSigma": 16.0, "highSigma": 16.0},
  "flatStarMasking": {"detectionSigma": 4.0, "minimumCleanSamples": 2},
  "scratchDirectory": "/Volumes/Data/seiza-scratch"
}
```

`bias` is valid for dark/flat builds; `dark` is valid only for flat builds.
Defect suppression is restricted to flats so a dark retains the hot pixels it
must subtract. `flatStarMasking`, for flats only, leaves stars and broad
saturated patches out of each sky flat before it is normalized; its fields
(`detectionSigma`, `haloRadiusFactor`, `minimumRadiusPixels`,
`saturationLevel`, `minimumCleanSamples`) are optional, and `{}` takes every
default. `darkLevelScreening`, for darks only, sets aside darks whose level
sits more than `maxExcessSigma` (default 1) times the quietest dark's pixel
noise above it, the mark of stray light. `scratchDirectory`, an existing
directory, holds a flat build's temporary pixels in place of the system's
temporary directory. Response schema 2 reports dimensions, all
integration/rejection tallies, calibration state, exposure, and this input
provenance:

- `requestedFrames` is the number of paths supplied by the caller.
- `inputFrames` and `inputs` cover only the paths actually integrated, in
  request order; each input includes its accepted, rejected and masked sample
  counts.
- `skippedInputs` names every input the integrator set aside, for a metadata
  disagreement or stray light, with its `path` and human-readable `reason`.
- `maskedSamples` counts the samples star masking left out, and
  `flatStarMasking` (null unless masking ran) holds the options used and what
  they masked.
- `rejectionMethod` is the rejection the build used: `NONE` for two inputs,
  `MEDIAN_MAD` for flats, or `LEAVE_ONE_OUT` for bias and darks.

The accepted and skipped paths are disjoint and together account for every
requested path. This accounting is validated before the writer publishes, so
no output appears after provenance validation, construction, or cancellation
failure. The writer itself publishes atomically.

```c
SeizaCancelSignal *cancel = seiza_cancel_signal_create();
/* Worker: */ char *report = seiza_calibration_build_master_json(
    request_json, cancel, &error);
/* UI thread, if needed: */ seiza_cancel_signal_cancel(cancel);
/* After the worker returned: */ seiza_cancel_signal_free(cancel);
```

The flag is thread-safe but must stay alive until the build returns.
Cancellation is cooperative between input frames. Progress reporting is not
part of this first ABI; hosts should show an indeterminate cancellable task.

For SNR/depth, `seiza_live_stacker_measure_depth` writes one `SeizaSnrSample`
without copying live buffers and returns `1` measured, `0` insufficient data,
or `-1` error. Test for exactly `1`. `seiza_checkpoint_depths` supplies the
doubling ladder and final depth. Generic curve fitting/projection is not yet in
this ABI; retain measured samples if the host wants to graph them.

The full C declarations, plus the memory-ownership contract (which returns are
owned vs. borrowed, and which `seiza_*_free` to call), live in
[`include/seiza_cabi.h`](include/seiza_cabi.h).

## Threads

Calls do their parallel work in Rayon's global pool, which by default has a
thread for every core. To cap it, set `RAYON_NUM_THREADS` in the process
environment before the first call into the library; the pool reads it once,
when it starts. Stacking and reintegration also prepare a few frames at
once on threads of their own, whose serial steps run beside that pool.

There is no C function to set a thread count. A Rust caller bounds Seiza by
running it inside a Rayon pool, which cannot be passed across the C ABI, and
every call from a host thread already shares the one global pool that the
environment variable sizes.

## The C header is generated

`include/seiza_cabi.h` is generated from the Rust source by
[cbindgen](https://github.com/mozilla/cbindgen) via `build.rs`, using
[`cbindgen.toml`](cbindgen.toml). Do not edit it by hand — change the Rust FFI
signatures/docs (or `cbindgen.toml`) and run `cargo build -p seiza-cabi`, which
rewrites the header only when it changes. CI (the `lint` job) fails if a source
change lands without a regenerated header.

## Building

The crate builds three artifacts (`crate-type = ["cdylib", "staticlib",
"rlib"]`):

- `libseiza_cabi.so` / `.dll` / `.dylib` (cdylib) — for the Windows .NET app.
- `libseiza_cabi.a` (staticlib) — for the macOS Swift app.
- rlib — for Rust consumers and the crate's own tests.

```
cargo build -p seiza-cabi --release
```

Consumers link the artifact for their platform and include `seiza_cabi.h`.

`seiza_color_stretch_rgb_stars(original, starless, output, width, height, amount,
error_out)` extracts a linear RGB residual and writes a separately stretched
display stars-only layer. Amount is [0,8], typically 5; lower values subdue stars.
`seiza_color_screen_rgb_star_layer(nebula, stars, width, height, reduction,
error_out)` screens it over a display nebula in place with optional linked
reduction [0,1]. All buffers contain width*height*3 interleaved floats. Shared
source normalization preserves RGB scale; inputs remain linear and unchanged.
Errors leave writable buffers unchanged. Neither call runs an external tool.

### Joint or reference canvas

`seiza_live_stacker_open_fits_joint_canvas(channels_json, channel, options_json,
max_canvas_megasamples, cancel, error_out)` builds the union of connected filter
footprints and returns an empty stack for the selected zero-based `channel`.
Push its lights explicitly. `channels_json` is an array:

```json
[
  {"paths": ["ha-left.fits", "ha-middle.fits"], "flat": "ha-flat.fits"},
  {"paths": ["oiii-middle.fits", "oiii-right.fits"], "flat": "oiii-flat.fits"}
]
```

Groups also accept optional `bias`, `dark`, `darkExposureSeconds`. The first
group's first light anchors geometry. Each filter keeps its own calibration
and normalization. Zero maximum uses 256 megasamples; options and cancellation
match expanding stacks. Invalid groups, duplicate lights or invalid indices
return errors. Disconnected frames may be omitted by registration.

Use `seiza_live_stacker_open_fits_on_canvas` on the returned handle for remaining
filters. Reference mode is faster when an existing grid already covers all
filters. Missing coverage stays absent; normalization needs overlap within
each filter. Joint mode adds a geometry pass across all groups, so reuse it.

### Analysis baselines and stellar SNR

The JSON response uses camelCase (`medianStarSnr`, `medianNormalizedNoise`).
Frames now include median unsaturated-star peak SNR. Filter groups expose
measured-frame medians of star count, HFR, FWHM, eccentricity, noise, background
and stellar SNR, plus gain-normalized noise. Raw noise/background retain native
input units; normalized noise uses reference units. Missing measurements are
null/None. See [measurement definitions](../seiza-analysis/README.md#measurements-and-physical-meaning)
for population, units and limitations. Thresholds and rejection remain caller-owned.

### Completed-stack reports

`seiza_live_stacker_report_context_json` captures the admitted-input ledger,
calibration assignments and admission settings without rereading images.
Capture it with the result, then pass it (or null if unavailable) to
`seiza_stack_snapshot_report_json` or `seiza_drizzle_result_report_json`.
The result uses camelCase keys and includes bounded per-channel support maps.
Batch handles retain final sample counts; drizzle support is drop area times
weight, not frame count. Admission RMS/mappings describe the admission pass,
even after full-resolution refinement.

`seiza_stack_report_html` renders that JSON as a self-contained report with
light/dark mode, expandable inputs and downloadable JSON. Free all returned
strings with `seiza_string_free`. Matching context/result counts are checked,
but callers must ensure they came from the same run.

### Combining completed stacks

Report generation is optional. A collection contains independent stack results; it
never re-registers images or infers shared sky coverage. Labels are nonempty and
unique (ignoring case and surrounding whitespace), and input order determines the
initial selection. Missing filters and different output sizes are supported. RGB
components remain inside their own stack, distinct from collection labels.

The collection JSON is `{ "schema_version": 1, "title": "SHO", "stacks":
[{ "label": "S", "report": <PostStackReport> }, ...] }`.
The HTML embeds the validated collection for download, uses no network resources, and
supports keyboard controls, search, 100-row pagination and printing all stacks.
Rejected attempts appear when supplied in host provenance. Empty context remains
unknown, not zero exposure. Maps are bounded summaries; this is not a crop tool.

Rendering saved JSON does not reread images or compute checksums. Building the
original post-stack report scans existing support maps and copies ledger metadata;
CLI provenance additionally hashes input/output files. Reporting never enables
reintegration by itself.

C: `seiza_stack_collection_html(collection_json, &error)` accepts the same structure with camelCase keys (`schemaVersion`, etc.). Free returned HTML/error strings with `seiza_string_free`. Invalid data returns NULL plus an error.

The HTML download uses canonical snake_case JSON, suitable for `seiza stack-report`; the C input uses camelCase as usual.


### OSC / RGB analysis (schema 3)

Analysis automatically recognizes mono, raw Bayer and three-plane RGB sources.
No extra option is needed, including for rich post-stack reports. The input list
is read once per source; measurement and registration run per component.
`max_frames` limits source paths; `samples` is the budget **per component**.

A schema-3 `frames` record represents one **source/component**, not necessarily
one file. CFA produces `R`, `G_red` (green on a red row), `G_blue` (green on a blue
row), and `B`; RGB produces `R`, `G`, `B`; mono produces `Mono`. Source-level read
failures or filter exclusions produce one record without sampling metadata.
Groups separate `(filter, image_kind, component)`, so CFA/RGB/mono never share
normalization or noise baselines. Do not sum exposure across color components.

Frame `sampling` records `image_kind`, `component`, `source_width`,
`source_height`, `spacing_pixels`, `offset_pixels`, declared `bayer_pattern`,
`storage_bayer_pattern` (row-order adjusted, before offsets), `bayer_offsets`,
and declared `row_order`. C JSON uses camelCase spellings of these same keys.
Native CFA sample `(x,y)` is at original stored-image pixel
`(2*x + offset_pixels[0], 2*y + offset_pixels[1])`. Each lattice retains every
native sample, including odd edges. No demosaicing, green averaging, white
balance or new calibration occurs. Unrecognized Bayer declarations and invalid
origin offsets are reported as failures rather than guessed as mono.

HFR, FWHM and registration RMS are in **original-image pixels** (CFA component
values multiplied by two); this does not recover details lost to sparse CFA
sampling. Undersampled stars can have biased shapes. Noise/background retain
native component units; depth noise uses that component's reference units.
Registration and photometric normalization are independent per component. A
component with too few stars/common patches can keep measurements while its
noise curve is unavailable. RGB retains existing demosaicing/resampling
correlations; it must not be interpreted as independent raw detector samples.

Both HTML renderers select filter/layout/component explicitly. Schema 1/2
reports remain renderable; new reports use schema/algorithm version 3.
Stacking and stretching numerical output versions do not change.
