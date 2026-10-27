# PSF Guard reference snapshots

Reference source: theatrus/psf-guard at
`79f5e31d53b8371e8fcd0b22a9b69692412cd8b4` (Apache-2.0).

`capture.rs` calls the actual PSF Guard library against temporary in-memory
SQLite catalogs. It never opens a user catalog. Compile it as a temporary
example of that pinned PSF Guard checkout, retaining the relative
`../flat-groups/request.json` fixture include, or link it against that checkout's
built `psf_guard`, `serde_json` and `rusqlite` rlibs. Its stdout is
`psf-guard.json`. Do not regenerate it using the extracted Seiza implementation.

The first twelve cases reproduce PSF Guard's cohort, validity, mode, age and
external-master regressions. The last four install the previously captured
flat-group response/comparison records into a synthetic catalog and call the
actual selector before/during/after a change, plus a mismatched-gain light.
They capture candidate UUIDs, coherent counts, full gates and flat basis/family.

`grading.json` adapts the native measurements and expected grades/eligibility
captured for seiza-mac#92, at the same PSF Guard revision. Only schema casing,
synthetic identifiers and timestamps are adapted; measurement numbers, grades,
reasons and selected source names are unchanged. The capture source is in
`theatrus/seiza-mac` branch `codex/stack-calibration-plan`,
`Tests/Fixtures/CalibrationParity/capture.rs`.
