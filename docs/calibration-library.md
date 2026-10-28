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
master dependencies, session assignments and complete light gates.

The code is PSF Guard's, moved: a Rust host calls it as its own (see
[Native hosts](#native-hosts)), and the C ABI and Python reach the same
functions through the JSON calls below, which are built from them.

Session identity covers what each master is made from and how: resolved
external-master adoption, or the raw inputs as a set with their
fingerprints, the build options and dependencies that apply, the matching
rules, the full light gate and flat correction. Lights whose masters take the
same frames share a session whichever light's nearest frame anchors them, as
PSF Guard partitioned them; the first light stands for the rest. Unused
candidates cannot hide a change in adopted master. Policy `algorithm_version`
is 5: inputs key as a set, and star masking, dark screening, the flat retry
policy and held-back flats key only recipes they apply to. Failure and pedestal
feedback retain the base key so hosts can submit execution results against it.

## Inputs

Python/native records use snake_case. C JSON uses camelCase of those fields.
Identifiers, enum strings, setting keys and arbitrary setting values retain their
spelling. Collections of settings are `{key, value}` records, never maps whose
keys could be mistaken for schema fields. Unknown schema fields, unknown setting
keys, invalid setting values and duplicate setting keys are refused at every
planning/cohort/retry boundary. Errors identify the setting and its global/rig
layer. The 43-field `settings_schema` describes accepted values and ranges; C
`seiza_calibration_settings_schema_json` and Python `calibration_settings_schema`
list it on their own, and `seiza_calibration_setting_check_json` /
`calibration_setting_check` judge one value as a settings panel saves it.
The internal stored-settings resolver still tolerates old values when PSF Guard
loads its catalog; hosts submit its resolved, valid records to these APIs.

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
| `flat_stability` | `{enabled, trim_unstable, try_next_session, retain_unstable}`; all default true, preserving PSF Guard's decisions |

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

For repeated selection, Rust hosts pass their existing `Arc<groups::Book>` to
`library::evaluate_with_book`. C hosts call `seiza_flat_groups_prepare_json` once
and reuse its immutable handle with `seiza_calibration_library_plan_with_groups_json`,
then call `seiza_flat_groups_free`. Python uses `FlatGroups(grouping_request)` and
`calibration_library_plan(..., prepared_grouping=book)`. Preparation takes the
same grouping records as `flat_group_plan`, including explicit rig `limits`.
Selection reuses the book and accepts no additional `grouping` rows; its
resolved limits must agree with the library settings. Hosts invalidate cached
books when source rows, night boundaries or settings change. C handles may be
shared for concurrent reads but must outlive all calls borrowing them.

### Native hosts

A Rust host holds the model and calls the functions PSF Guard's code called,
in its own flow, with its own catalog, caches, files and threads:

- `library::rules`: the settings table (`fields`, `field`, `validate`,
  `describe`, `check_setting`), `RuleBook` layers and their resolution into
  `Rules`, with each value's source, check modes and grouping limits. A
  stored value this build cannot read falls through to the next layer.
- `Library::new(frames, book, available_responses)` holds a catalog once;
  `Library::select(light, rules)` matches one light by borrowing it,
  `light_choices` and `flat_family_compatible` answer a choice panel, and
  `library::matching` holds the tests a refusal explains.
- `grade(measured, rules, nights)` judges current measurements against their
  peers; the host's nights come in through `NightSplit`, and `night_of` names
  them. `kept_out_reason`, `describe_finding` and the stored measurement
  records are the ones hosts persist.
- `coherent_cohort`, `adopted_master`, `master_dependencies`,
  `dark_level_screening`, `pedestal_sample_lights`, `pedestal_hint`,
  `accept_pedestal`, `trims_to_majority`, `same_session_as`, `flat_set_label`
  and the retry notes are the decisions a master build makes.
  `resolve_session` resolves a verified selection into its session, and
  `Session::record_failure`, `plan_flat_use` and `accept_pedestal` take
  execution results.
- The flat grouping job's planning: `plan_sessions`, `SessionPlan` (its
  sample, response key and response row), `PedestalTable`, `pedestal_moved`,
  `family_questions` and `FamilyQuestion`, `change_question` and
  `ChangeFilter`. A host gives each imaging session's lights and reads the
  headers and pixels they ask for.

`evaluate` is these calls over one request, so a native host and a host that
submits records reach the same answers.

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
3. Measure post-build flat stability with `seiza_stacking::measure_flat_stability`,
   C's `seiza_master_flat_stability_json`, or Python's `master_flat_stability`.
   The native call accepts the loaded master and optional calibration; bindings
   load their `master`/`bias`/`dark` paths. Source records are
   `{path, exposure_seconds, skipped}`. Set `normalized_full_scale=65535` for
   PSF Guard's declared-unit XISF scale. This is the original mono/RGB/CFA,
   dip-only feature comparison, opening one source at a time. Unreadable or
   incompatible sources count as drifting and return indexed diagnostics when
   the master has enough features to judge; otherwise all disagreements are zero;
   deliberately skipped sources have zero disagreement. The measurement
   algorithm remains version 2.
   Submit accepted UUIDs and aligned disagreement to the shared retry policy
   described below. The host stages/publishes files only after its decision.

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

## Resolving selected candidates and executing retries

A catalog host can call `library::plan_masters(MasterPlanRequest)` after it
verifies/remaps selected rows, without repeating grouping or selection. C uses
`seiza_calibration_master_plan_json`; Python uses `calibration_master_plan(request)`.
The request contains the already selected `frames` in candidate order, the
session's `lights` (used for pedestal sampling), `quality`, global/rig settings,
`flat_basis`, `flat_family`, `flat_held_back`, star masking and retry options.
It also accepts the same failure/pedestal feedback as library planning. It
returns one resolved session, with exactly the same recipes and identity as
library planning for that selection. Different selections still need separate
calls; this API deliberately performs no candidate matching.

Forward current grade fingerprints/algorithm versions and manual overrides,
even when pixel measurements are omitted. A current grade or personal choice
prevents the legacy dark screen from rejudging that input. Forward star masking
explicitly. Building directly from an unverified preview is refused by the host;
replan after verification or changed source metadata. Recipe exposure belongs
to the adopted master/first actual coherent input, including unknown exposure.
`coherent_inputs` reports the raw cohort even if it has only one frame or an
external master is preferred. `master_cohort` (C `seiza_calibration_master_cohort_json`,
Python `calibration_master_cohort`) exposes the original metadata-tolerant
coherence helper for previews/retries using effective settings. It validates
settings just like executable planning; full planning also validates metadata,
source-verification records and identities. Rust now returns `Result<MasterCohort>`.
`FlatRetryRequest.policy` is `Option<FlatRetryPolicy>` so an omitted policy can
be distinguished from an explicit override; existing JSON policy objects retain
their shape.

`library::flat_retry(FlatRetryRequest)` / `seiza_calibration_flat_retry_json` /
`calibration_flat_retry(request)` owns the post-build decision sequence, built
from the decision functions a native host calls. An alternate set is the next
session out, clustered as one master takes it from the remaining raw
candidates nearest first, never every flat outside the first window. Supply
the returned `session` object and all selected raw `candidates`, then the first attempt's accepted `used` UUIDs, `disagreement`, `drift_threshold`,
`feature_pixels` and `memo` flag. Optional host `label` strings affect only
explanations. Empty disagreement means no stability measurement was available.
The session carries `flat_stability`, `flat_held_back` and the resolved
`flat_session_seconds` tolerance into execution. Conflicting explicit policy,
held-back or time-window overrides are rejected. Without a session, the existing
`policy`, `held_back` and effective `settings` fields remain supported; omitted
policy means all four defaults are true. Prefer passing the session to preserve
rig-specific settings and explicit disabling decisions.

The default policy trims a stable majority (at least two sources), then tries
another session excluding the whole initial capture window and held-back
sources. Supply `trimmed`/`alternate` feedback when a retry succeeds; its
presence implies attempted. When a retry fails/yields no master, set
`trimmed_attempted` or `alternate_attempted` instead so it is not repeated.
Actions are `try_trimmed`, `try_alternate`, `publish_first`, `publish_trimmed`,
`publish_alternate` or `withhold`. Retry actions return their candidate UUIDs.
If no stable retry succeeds, the original is retained with its warning. A memo
must be rebuilt when `rebuild_first` is true; it is never a usable file. Hosts
own cancellation, staging cleanup, cache freshness and atomic publication.

Hosts can disable retry decisions, skip trimming or alternate sessions, or set
`retain_unstable=false` to withhold a failed set. These are explicit execution
options, separate from the unchanged 43 user settings. PSF Guard uses the full
default policy. Changing the retry policy changes session identity. Keep final
accepted inputs and dependencies in master provenance; a provisional recipe
key alone cannot validate an executed master cache entry.


### Python records and execution example

`calibration_master_plan`, `calibration_master_cohort`, `calibration_flat_retry`
and `master_flat_stability` accept either a request mapping or keyword fields;
mixing the two raises `ValueError`. Python exports `TypedDict` records and
`Literal` actions for completion and static checking, including
`CalibrationMasterPlanRequest`, `CalibrationSession`, `FlatRetryPolicy`,
`FlatAttempt`, `CalibrationFlatRetryDecision` and `FlatSetStability`. They remain
ordinary dictionaries; core validation applies at runtime.

```python
session = seiza.calibration_master_plan(
    frames=selected_frames, lights=session_lights, quality=cached_quality,
    flat_held_back=held_back,
    flat_stability=seiza.FlatRetryPolicy(retain_unstable=False),
)
decision = seiza.calibration_flat_retry(
    session=session, candidates=selected_raw_flats, first=first_attempt,
)
```

The [complete host loop](../seiza-py/examples/calibration_retry.py) dispatches
build/retry/publication actions, carries the same session through feedback,
handles failed retries and memo rebuilds, and discards unpublished attempts on
exit or cancellation. It contains no independent flat-selection decisions.

For C, insert the returned session object unchanged into the camelCase retry
request's `session` field. Its `flatStability`, `flatHeldBack` and
`tolerances.flatSessionSeconds` are consumed together. Result/error strings are
owned by the caller and freed with `seiza_string_free`; the planner retains no
JSON buffers between calls.

### Measuring cached pixels

File measurement remains available through `measure_flat_stability_files`,
`seiza_master_flat_stability_json` and `master_flat_stability`. To reuse cached
pixels, call `measure_flat_stability_images`,
`seiza_master_flat_stability_images_json`, or Python:

```python
measurement = seiza.master_flat_stability_images(
    master_array, source_arrays, bias=bias_array, dark=dark_array,
    dark_exposure_seconds=2.0,
    input_metadata=[{'exposure_seconds': 2.0} for _ in source_arrays],
    bayer={'pattern': 'RGGB', 'x_offset': 0, 'y_offset': 0},  # raw CFA only
)
```

Python accepts C-contiguous float32 arrays shaped `(height, width)` or
`(height, width, 3)`. C accepts `SeizaFlatStabilityImage` descriptors with
row-major, interleaved samples and a checked length. Both bindings copy buffers
for the duration of the call and retain no caller memory. The
[standalone C example](../seiza-cabi/examples/master_flat_stability.c) shows the
buffer layout, call and error/result ownership. Native Rust borrows
input images and copies one source at a time only when calibration is needed.
All cached pixels must already use the same units; this path does not infer
headers or rescale values. `dark_bias_subtracted=true` requires a supplied bias.
Omit `bayer` for mono/RGB; CFA offsets are zero or one. Source metadata may
override the default CFA layout and carries exposure and skipped flags. Empty
metadata uses defaults; otherwise its length must match the source count.

The file and image APIs use one measurement algorithm, including mono/RGB/CFA
planes and indexed geometry/calibration diagnostics. Sources are reported in
input order, so feedback UUIDs must have that same order. `enabled=false` on
`FlatRetryPolicy` skips retry decisions; the measurement functions remain
independently callable. Measurement algorithm version stays 2.

## Supplied sets and raw-flat response preparation

`flat_association` and `dark_association` are independent `automatic` (default)
or `one_set` choices on library and master-plan requests. C JSON uses
`flatAssociation` / `darkAssociation`. One-set includes all eligible inputs,
across nights and beyond the automatic 64-frame selection cap. It preserves
grades, explicit exclusions, source checks and physical compatibility. It
refuses incompatible sets and multiple/mixed external masters. Flat retries
can trim unstable frames but cannot switch to another session in this mode.

`library::plan_flat_responses` (`seiza_calibration_flat_response_plan_json`,
Python `calibration_flat_response_plan`) runs PSF Guard's session planning
(`plan_sessions`) over library request records. Eligible raw flats group by
family (rig, filter, focal length class, frame size, binning, gain, offset and
readout mode name), night and rotation, under the family, session and response
keys PSF Guard keeps its responses by, so a host that adopts this keeps its
cached planes. Each session samples at most `metric::MAX_FRAMES` flats, evenly
spaced. Sessions too small to make a response, and rigs grouped by time, are
not planned. The pedestal under a session's flats is the median of the
dark-flats within 5% of its exposure, else of the rig's biases at that gain and
offset (`PedestalTable`). Results include the native rig limits and night
boundaries needed to prepare a grouping book. Missing capture times are
reported as unplanned; hosts must not fabricate dates. Hosts supply measured
response planes and pair comparisons; this API does not claim pixel evidence
exists merely because sessions have been prepared.

## Light evidence, correction planning and personal choices

The same core boundary covers the remaining workflow:
`library::plan_flat_work(FlatWorkRequest {library,light_context})` and
`library::finish_flat_work(FlatWorkAnswer)`. Their C exports are
`seiza_calibration_flat_work_plan_json` and
`seiza_calibration_flat_work_finish_json` (camelCase); Python offers
`calibration_flat_work_plan` / `calibration_flat_work_finish` with mapping or
keyword calls and the same snake_case records.

Cached light metadata, catalog IDs/profiles/targets/exclusions and optional
pedestal estimates drive questions without paths being opened. The questions
are PSF Guard's family and change stages (`family_questions`,
`change_question`) over the request's lights: each imaging session's lights by
filter and target, then by angle within the rig's rotation tolerance; family
compatibility from the first light's sensor and optics; bounded samples;
representatives; evidence-current keys; the tie and missing-fit rules; and the
change verdict. A light's pedestal is its context's, else the rig's bias level
at its gain and offset. Native split/join and light-side/group choice rows pass
through grouping unchanged and report their resolved/orphaned status. Hosts
retain database and file/cache mechanisms. `evaluate` refuses light evidence
without the imaging sessions it was measured for: submit the `grouping` this
call returns.

Pixel entrypoints support cached arrays as well: response pedestal removal,
light-background generation, pairwise response choice, dust-only evidence maps,
change fitting, compact correction maps and correction application. Dust-only
evidence maps mask vignetting using the policy extracted from PSF Guard's job;
correction maps intentionally use the full compact change. These are distinct
operations. See the Python README's cached workflow for the feedback protocol.

No response sample plan alone constitutes measured light evidence. Missing images
remain retryable host failures; inconclusive pixels remain undecided and do not
become a positive family or side verdict. Current cached answers request no
sample reads, preserving PSF Guard's database/cache advantage over a file-only CLI.

Path-based hosts may key `LightChoiceEntry.file_name` by the exact light path.
The selector prefers that key when present and otherwise uses the basename,
preserving existing PSF Guard catalog semantics. This prevents same-name/time
files in different directories from sharing a personal choice.
