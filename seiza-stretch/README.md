# seiza-stretch

`seiza-stretch` provides format-independent, parameterized display stretching
for astrophotography pipelines. It analyzes mono or interleaved RGB `f32`
samples, resolves image-dependent requests into deterministic transfer plans,
and applies those plans to either `f32` or `u8` output.

The crate consolidates Seiza's existing behavior:

- identity and explicit linear range mapping;
- percentile range selection with an asinh transfer;
- explicit PixInsight/N.I.N.A.-family midtones transfer functions;
- manual Generalized Hyperbolic Stretch with symmetry and protection controls;
- the existing median/MAD Auto-MTF model;
- the existing exact-histogram `u16` Auto-MTF fast path.

Analysis and application are intentionally separate. A caller can analyze an
image once, resolve several parameter choices against the same statistics, and
apply a selected `StretchPlan` to a downsampled interactive preview or the
full-resolution image. Deterministic models also resolve directly without an
analysis pass. Future informed automatic modes can select one of these models
without embedding policy in the transfer implementation.

Stretching never happens implicitly. Linear stacking, calibration, and FITS
writing remain linear unless a caller explicitly resolves and applies a model.
