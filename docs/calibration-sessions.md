# Calibration sessions in expanded mosaics

Canvas planning, registration and integration must see the same calibrated sky.
`LiveStacker::open_fits_expanding_sessions` accepts original light paths,
`FileCalibrationSession` records and one session index per path. The C ABI
exports `seiza_live_stacker_open_fits_expanding_sessions`; Python exposes
`LiveStacker.open_fits_expanding_sessions`. The returned stack is empty.
Push every original light, including the anchor, in its original order.

A session holds optional master bias, dark and flat paths, a dark exposure
override, complete matching tolerances and optional input scaling. An absent
master means none for that session. Every transition replaces the whole gate,
including recorded/measured/ignored light rotation. Manual master or gate
replacement and anonymous prepared-pixel pushes are refused on assigned stacks.

PSF Guard's fitted-pedestal recipe is a normalized flat plus a nonnegative
`pedestal`, without bias or dark. The stacker reconstructs the constant bias
and preserves the flat's sensor and optical metadata. `normalized_full_scale`
rescales only a source declaring normalized `0:1` bounds; 65535 reproduces
PSF Guard's loader. Omission preserves original units. Pipeline scaling cannot
override a session's declaration.

Assignments are validated before planning. Canonical path and physical file
identity checks reject duplicate lights and aliases between lights and masters,
including hard links. Source identity, size and modification time are checked
before use. Sources and masters, including future lights, are protected from
output overwrite. This detects replacement and ordinary edits; it is not a
cryptographic content check against deliberately preserved in-place metadata.

The planner uses bounded read-ahead and a one-entry master cache; workers retain
only masters they are preparing with. Assigned file pipelines integrate in
order, reporting `SequentialCalibrationSessions`, rather than retaining every
session's master pixels in worker caches. Existing sample and pool budgets and
cancellation checks still apply.

The immutable assignment plan participates in the configuration fingerprint.
Checkpoints retain it alongside each historical calibration recipe, so resumed
integration and reintegration use the original gates, pedestal and scaling.
Historical master changes refuse replay. Old contexts without session metadata
retain their existing behavior. An older reader refuses a context carrying the
new metadata rather than silently discarding assignments.
