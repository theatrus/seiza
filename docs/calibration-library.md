# Shared calibration library policy

`seiza_calibration::library::evaluate`,
`seiza_calibration_library_plan_json` and Python's
`seiza.calibration_library_plan` evaluate the same immutable library snapshot.
The policy was extracted from PSF Guard revision
`79f5e31d53b8371e8fcd0b22a9b69692412cd8b4`, rather than reconstructed from
metadata matching. The grouping stage is described in [flat-groups.md](flat-groups.md).

The extracted code resolves PSF Guard's defaults and global/rig settings;
groups grading peers; screens bias, dark, dark-flat and flat measurements;
applies manual eligibility choices; selects calibration candidates; chooses
complete dark/bias cohorts, named readout modes and flat families; and produces
master dependencies, session assignments and complete light gates. Session
identity includes resolved external-master adoption or ordered raw inputs and
their fingerprints, build options/dependencies, matching rules, the full light
gate and flat correction. Equivalent recipes reuse a session without reordering
the lights; unused candidates cannot hide a change in adopted master. Policy
`algorithm_version` is 2 for this resolved-recipe identity. Failure and pedestal
feedback retain the base key so hosts can submit execution results against it.

## Inputs

Python/native records use snake_case. C JSON uses camelCase of those fields.
Identifiers, enum strings, setting keys and arbitrary setting values retain their
spelling. Collections of settings are `{key, value}` records, never maps whose
keys could be mistaken for schema fields. Unknown schema fields are refused.

| Request field | Contents |
| --- | --- |
| `frames` | PSF Guard-shaped `CalibrationFrame` rows, with stable `frame_uuid`, host-resolved `rig_uuid`, `kind`, source path/fingerprint, optional metadata, validity boundary, legacy kept-out reason and verification status |
| `lights` | Original ordered `Light` records, with path, rig UUID, metadata and Unix `timestamp` |
| `quality` | One `QualityRecord` per measured/graded/overridden frame: measurement, source fingerprint, algorithm version, stored grade/reason, read error and optional `use`/`exclude` override |
| `global_settings` | Global `{key, value}` records |
| `rigs` | `{rig_uuid, name, settings}` layers; own settings override globals, then defaults |
| `grouping` | Measured grouping request; put lights in the outer `lights` array. Grouping limits come from the same resolved rules used for grading and selection |
| `available_responses` | Response keys the host verified are available for a promised cross-change correction |
| `night_shifts` | `{rig_uuid, frame_uuid, seconds}` host-resolved local-noon shifts; settings override site-derived shifts |
| `regrade` | Run current measurements through the extracted peer grading policy; default false preserves stored eligibility |
| `master_failures` | `{session_key, kind, reason}` failed builds; native planning blocks their declared dependents |
| `pedestal_evidence` | `{session_key, estimates, pinned}` successful native pedestal fits for the returned sample lights, or a previous stack's pinned fit |
| `force_flat_without_pedestal` | Explicitly allow PSF Guard's forced-on flat-only behavior; default false |
| `flat_star_masking` | Enable native master flat star masking; PSF Guard's default is false |

Metadata retains PSF Guard's row names: `exposure_s`, `camera_temp`,
`focal_length_mm`, `rotation` on a calibration row, `rotator_position` on a
light, and `readout_mode_name` alongside numeric `readout_mode`. Dimensions,
binning, gain and offset are optional integers. Times are Unix seconds;
rotations are degrees. Night shifts are unwrapped `12h - UTC offset`, default
43200; UTC+13 therefore uses -3600, preserving the night's evening date.

Measurements use PSF Guard's tagged shape: `{"type":"dark", ...DarkMeasurement}`
or `{"type":"flat", ...FlatMeasurement, "optics": ...}`. Bias/dark/dark-flat
measurements use `FRAME_QUALITY_ALGORITHM_VERSION`; flats use
`FLAT_QUALITY_ALGORITHM_VERSION`. Only current, fingerprint-matching raw
measurements are regraded. Stale, failed and absent measurements remain
separate states and never become an invented OK grade. Stored grades and
legacy stray-light eligibility retain PSF Guard's precedence. A `use` override
restores eligibility but cannot bypass sensor, optics or validity gates.

### Reusing a host catalog and caches

PSF Guard can populate these records directly from its existing database. File
paths are opaque identifiers during planning: the planner never opens them,
scans directories, measures pixels or requires a Seiza-owned database. Supply
cached metadata in `frames`/`lights`, cached measurements and stored decisions
in `quality`, and cached response/comparison records and overrides in `grouping`.
Use `regrade=false` to retain stored grading, or `true` to regrade the supplied
current measurements without measuring the files again. `available_responses`
declares which correction responses the host can retrieve from its cache.

The host remains responsible for cache freshness and can reuse prior source
verification while its fingerprint and metadata assertions remain valid. It
can reuse compatible cached master pixels and execute only missing recipes.
A CLI without a catalog gathers metadata and measurements from disk to fill
the same request; persistence and scanning are host choices. Neither host needs
to duplicate the matching policy.

## Results and master execution

`grades` retains input frame order. `selections` and `assignments` retain input
light order. `sessions` contains a stable key, complete `tolerances`, ordered
master recipes and an application policy. Selections also report kept-out
frames, available library kinds, flat basis/family, held-back flats, personal
choices and nondefault rule notes. `settings_schema` exposes the original
settings table, limits and defaults; apps do not need their own threshold table.
Native Rust consumers can name these result models through `library` reexports,
including `FlatBasis`, `FlatFamily`, `RuleNote`, `FieldSpec` and `FieldKind`, and
pattern-match them directly without serializing through JSON.

A recipe's action is `external`, `integrate`, `unavailable`, `failed` or
`blocked`. It names input UUIDs, declared dependencies, coherence exclusions,
exposure and native builder options. Exposure comes from the adopted external
master or first selected coherent raw input, and remains absent if that input
does not record it. Build order is bias, dark-flat, dark, flat.
An external master is adopted as-is, never integrated with raw frames. The
prefer/fallback/ignore decision follows PSF Guard's coherent raw-frame count.
Flat defect suppression is enabled; dark hot pixels remain intact. Legacy dark
level screening runs only when a dark/dark-flat input lacks a current grade
or personal override.

Planning opens no files and does not claim a master has been built. Hosts:

1. Resolve source files and verify current fingerprints and hard metadata,
   deduplicate physical files, then resubmit the rows with `source_verified`.
   A recipe with `requires_source_verification` must be replanned before use.
   Once a kind's candidates are verified, the original nearest-first 64-frame
   master-input ceiling applies before coherence and adoption. There is no
   ceiling on the number of lights.
2. Execute dependencies through `build_master_from_fits` (FITS or XISF through
   image I/O), using the returned options. Retain actual accepted/skipped input
   provenance. Feed build failures back into the planner; never silently omit
   a failed declared bias/dark-flat while building a dependent flat.
3. Preserve PSF Guard's post-build flat stability check and retry policy where
   the host implements it. This value planner describes initial master inputs;
   it does not replace that pixel check or its cache/staging orchestration.
4. Apply any `flat_basis.correction()` through the shared change-map kernels.
   A promised correction that fails is an error, rather than a stack silently
   keyed as corrected. `available_responses` is a host availability assertion,
   not evidence synthesized from metadata.
5. Follow `application.flat_use`. With no usable bias/dark, automatic mode
   returns `needs_pedestal_fit`; its sorted/deduplicated first, middle and last
   paths match PSF Guard's sample selection. Read mono/non-CFA samples, call
   `fit_flat_pedestal`, and submit at most three successful estimates. Native
   policy takes the upper median, checks the 64 ADU/25% spread, 0..13107 ADU
   range, and known ZWO/ASI offset corroboration. It returns `fitted_pedestal`
   or `withheld`. A pinned previous fit wins over fresh estimates. Forced
   calibration returns `forced_without_pedestal` explicitly.

Persist the source snapshots, actual builder outcomes, complete gates,
correction and fitted pedestal with the final recipe. Cache identity must
include these plus the policy `algorithm_version`, settings, build options and
stacking `OUTPUT_VERSION`. A matching session key alone does not validate a
cached master's pixels or dependencies.

For expanded mosaics, execute each unique session once, translate its final
masters/pedestal/gate into `FileCalibrationSession`, then pass the original
paths and returned assignments to the [session-aware opener](calibration-sessions.md).
Canvas planning, integration, resume and reintegration then share calibration.

## Reference coverage

`seiza-calibration/tests/fixtures/library/psf-guard.json` captures 16 selections
and coherent input counts by calling the actual pinned PSF Guard library on
temporary synthetic SQLite catalogs. It covers complete/nearest cohorts,
validity, age, external masters, named readout modes, measured flat families,
change placement and hard-gate rejection. `grading.json` carries the previously
captured PSF Guard measurements/grades for gain separation and a leaking dark,
including its exact reason and use-anyway behavior. Rust, C and Python consume
these shared fixtures. Tests also exercise recipe failures, fitted/pinned
pedestals, input verification and stale/failed/unmeasured distinctions.
