# seiza-core

The types Seiza's crates share, kept apart from the plate solver:

- `wcs::Wcs` and `wcs::Sip`: a TAN or TAN-SIP plate solution, with pixel
  and sky conversions, read from and written to FITS cards.
  `image_center` gives an image's middle, `((width - 1) / 2, (height - 1) / 2)`
  with pixels counted from 0.
- `CancelSignal`: cooperative cancellation for long work.
- `prior::SolvePrior`: what an image's headers say about where it points (a
  WCS, the mount's RA and Dec, or a scale from pixel size and focal length).

`seiza` re-exports all of them at the same paths, so most code uses them
through `seiza`. Depend on `seiza-core` directly from a crate that needs only
these types: it then need not move with every `seiza` release. This crate
changes rarely, and only by adding.
