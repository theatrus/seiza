# seiza-stacking

`seiza-stacking` provides linear, incremental image stacking for
astrophotography applications. It keeps plate solving and catalog access out of
the stacking path while reusing Seiza's star detector for local registration.

The first release supports:

- mono, planar RGB, and Bayer FITS inputs in linear sensor units;
- optional master bias, dark, and flat calibration;
- bounded-memory, two-pass construction of bias, dark, and flat masters;
- bounded-drift star registration with translation/rotation/scale refinement;
- robust global or tiled local normalization;
- online residual (delta-sigma) rejection with coverage and rejection maps;
- non-mutating frame admission gates for additive live stacks;
- floating-point FITS output on the reference frame's pixel grid.

`LiveStacker::push` is the embedding API intended for acquisition tools and
PSF Guard. The CLI's `seiza stack` command feeds files through the same state
machine. Frame-quality scoring remains the host application's responsibility;
the crate's admission gates cover only compatibility and numeric/geometric
safety. Live renderers can borrow `LiveStacker::view` without copying the
full-resolution accumulator; any display stretch remains a caller-only visual
operation.

Frame admission remains ordered because online rejection depends on prior
observations. Independent work within each frame—calibration, registration
detection, resampling, normalization, classification, and integration—uses the
shared Rayon worker pool. Applications may set `RAYON_NUM_THREADS` or install
stacking work in a configured Rayon pool when they need to reserve CPU for
acquisition and display work.

Integrated flats are applied in the raw light frame's sampling before CFA
debayering. Master darks and flats retain their Bayer pattern and origin
offsets, and a known layout must match the light before calibration. A supplied
bias is removed first, and planar RGB flat channels are normalized independently
so calibration does not introduce a color-scale shift. When bias subtraction
makes a master dark exposure-scalable, every light must provide an exposure
duration rather than silently assuming a 1:1 scale.

`build_master_from_fits` builds reusable calibration masters without retaining
the input sequence in memory. It rereads each file for a leave-one-out
sigma-clipped second pass, validates available acquisition metadata, calibrates
and normalizes each flat before integration, and returns per-input rejection
statistics. `write_master_fits_f32` records the master kind, input count,
rejection settings, and bias/dark/normalization state in the FITS header. Those
state fields prevent a later `CalibrationMasters` consumer from calibrating a
prepared dark or flat twice.

The format-level float writer lives in `seiza-fits`. This crate only selects
stack- and master-specific typed header cards before passing its interleaved
linear image to that generic atomic writer.

Registration uses every retained detection for a bounded translation seed,
complemented by bright-star triangles for rotation and scale. The expected
center displacement is the larger of
`StackOptions::registration.maximum_drift_pixels` and
`maximum_drift_fraction` times the reference frame's larger dimension. The
defaults are 256 pixels and 15%. Differently sized or cropped light frames are
resampled onto the reference grid; samples outside their valid crop remain
masked and are accounted for by the overlap admission gate.

Meridian-flipped frames are accepted by default. The rotation admission limit
is measured from the nearer of the reference orientation and its 180-degree
counterpart, while diagnostics retain the full fitted rotation (for example,
179.3 degrees). The same similarity transform is then used to rotate the
pixels back onto the reference grid before normalization and integration.
