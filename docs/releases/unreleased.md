# Unreleased

- Registered-frame mosaic integration accepts panel IDs to match additive
  background planes through overlaps and taper remaining smooth differences
  at panel edges. Bright overlaps match disconnected background groups by a
  constant, without fitting gradients from nebula structure. Rust, C and
  Python share the implementation; source coverage and full-resolution detail
  are retained through the existing rejection pass. Matching compares paired
  sample positions before clipping, accepts partial overlap cells, and keeps
  each RGB channel's valid samples independent. Outlying residual differences
  are clipped before averaging to protect backgrounds from transient hits.

Stacking output is 0.51.0. The release shipping this change must include
`seiza-stacking` 0.51.0. Stretch output and the previous stacking output cases
remain unchanged. Four new mosaic cases record the changed output.
