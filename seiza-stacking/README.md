# seiza-stacking

`seiza-stacking` provides linear, incremental image stacking for
astrophotography applications. It keeps plate solving and catalog access out of
the stacking path while reusing Seiza's star detector for local registration.

The first release supports:

- mono, planar RGB, and Bayer FITS inputs in linear sensor units;
- optional master bias, dark, and flat calibration;
- star-based translation/rotation/scale registration;
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

Integrated flats are applied in the raw light frame's sampling before CFA
debayering. A supplied bias is removed first, and planar RGB flat channels are
normalized independently so calibration does not introduce a color-scale
shift. When bias subtraction makes a master dark exposure-scalable, every
light must provide an exposure duration rather than silently assuming a 1:1
scale.
