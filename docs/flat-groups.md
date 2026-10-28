# Shared measured flat grouping

`seiza_calibration::flat_response::groups` owns the measured grouping policy
extracted from PSF Guard. Rust hosts build a `Book` from their stored rows
(`groups::prepare`) and ask it where each light stands
(`Book::place_light`); Python binds the same (`FlatGroups`, `place_light`).
The C ABI's `seiza_flat_group_plan_json` (`library::json::flat_group_plan`)
does both in one call and returns the book with every light placed. See
[provenance](../seiza-calibration/src/flat_response/groups/PROVENANCE.md).

This stage builds measured families/groups and change events, resolves
overrides, constructs angle epochs, and places lights using time, imaging
continuity, and pixel evidence. A light inside a change window can remain
**undetermined**. An angle epoch widens candidates; it does not alone move a
light to another family. Disputed pixel moves retain their evidence basis.

Hosts supply measurements from Seiza's existing response/evidence kernels and
retain file/catalog I/O and persistence. Compatible families must already pass
hard sensor/optics gates. Personal choices do not bypass those gates. Automatic
results and resolved personal choices remain separate, as in the native book.

## Request

C sends one JSON request and receives allocated JSON; free results/errors with
`seiza_string_free`. Its record fields are **camelCase**. Python accepts the
same fields as keyword arguments, with **snake_case** record keys, and returns
a dict. Rust serde uses snake_case. Enum values keep snake_case in all three.
Collections default to empty; unknown record fields are errors. Identifiers
are opaque values and keep their spelling. Keyed collections are arrays of
pairs so translation never renames identifiers.

| Python/Rust field | Records / purpose |
| --- | --- |
| `responses` | `ResponseRow`: measured sessions and stable frame identities |
| `comparisons` | `ComparisonRow`: response pair broad/dust/sigma, spots, vignetting |
| `evidence` | `EvidenceRow`: imaging-session evidence about a change |
| `imaging` | `ImagingSession`: continuous light sessions |
| `imaging_lights` | Alternative to `imaging`: `[id, profile_or_null, captured_at]` rows |
| `profile_rigs` | `[profile_or_null, [rig_uuid, ...]]` for grouping raw imaging rows |
| `limits` | `{"rig_uuid": str, "limits": {"measured": bool, "dust": float, "broad": float, "window": int, "rotation": float}}` |
| `group_overrides` | `GroupOverrideRow`: stable split/join choices and reset tombstones |
| `night_boundaries` | `{"rig_uuid": str, "frame_uuid": str, "seconds": int}` local noon overrides |
| `epoch_samples` | `{"rig_uuid": str, "at": int, "offset": float}` sky-minus-mechanical samples |
| `family_evidence` | `FamilyEvidenceRow`: candidate response choices from lights' pixels |
| `light_choices` | `LightChoiceEntry`: light identity and before/after/group choices |
| `lights` | `{"file_name": str, "rig_uuid": str, "filter": str_or_null, "at": int_or_null, "angle": float_or_null, "compatible_families": [family_key, ...], "rotation_tolerance": float}` |

Shared record definitions:
[rows.rs](../seiza-calibration/src/flat_response/groups/rows.rs),
[model.rs](../seiza-calibration/src/flat_response/groups/model.rs),
[overrides.rs](../seiza-calibration/src/flat_response/groups/overrides.rs).
[request.json](../seiza-calibration/tests/fixtures/flat-groups/request.json)
is an executable example with captured PSF Guard expected output alongside it.
Optional fields accept null/omission. Failed/undated responses stay out of the
book, preserving the original policy. Every usable response's rig needs limits.

Times are Unix **seconds**, angles **degrees**, levels/pedestals **ADU**,
dust/broad/noise fractions of background. `window` is seconds. Night boundaries
are unwrapped local-noon shifts (`12h - UTC offset`) within one day of noon UTC,
defaulting to noon UTC. Capture times are bounded to ±2^50
seconds to keep differences safe. Dust/broad limits must be positive and
finite, rotations nonnegative and finite, windows in `[0, 1000000000000]`.
Supply either pre-grouped `imaging` or raw `imaging_lights`. Raw rows group by
profile, splitting at gaps over three hours, using the original PSF Guard rule.

## Result and measurement passes

The snapshot includes `algorithm_version`, sessions, families (with groups,
alike ranges and carrying classes), boundaries, events, override states,
outlier/spanning boundaries, angle terms, epochs and imaging sessions. Keyed
outputs are sorted arrays of pairs. Indices refer to this result's arrays.
Cache against `algorithm_version` **and all inputs**; it is not a fingerprint.

`pairs_to_compare` schedules response measurements. `sessions_in_windows`
gives `[event_index, imaging_index, evidence_key]` jobs;
`sessions_needing_evidence` identifies those without stored evidence.

Each `lights` result, in query order, contains:

- `automatic_family`: family index, candidates, angle shift and basis, or null.
  Empty `compatible_families` means no candidates.
- `family_probe`: candidate indices, representative session indices, and the
  key to use for a pixel pass's `FamilyEvidenceRow`; null without time/angle.
  Changed keys invalidate old evidence.
- `choices`: resolved personal side/group choices and orphaned choices with
  reasons, distinct from the automatic result.
- `events`: applicable indices, automatic placements, response pairs for
  correction maps, and reference session indices for evidence fits. Missing
  time gives null placement; insufficient evidence can give `undetermined`.

Build the book first to discover family keys, gate families for each light,
then place the lights. Pixel passes feed their evidence into the next
immutable set of rows. No files or previous results are modified.

## Relationship to seiza-mac #92

This supplies the measured flat-policy stage for
[seiza-mac #92](https://github.com/theatrus/seiza-mac/pull/92), checked against
its pinned PSF Guard fixture through Rust, C and Python. The 48 extracted unit
tests cover angle ranges/epochs, camera-side events, pixel corroboration,
outliers, spanning changes and evidence freshness.

The [library policy](calibration-library.md) adds peer grading/eligibility,
temporal/cohort selection and master dependency recipes. The
[session-aware expanded planner](calibration-sessions.md) calibrates original
files under their assignments before star detection and footprint planning.
The Mac app should consume these released C APIs and complete host parity
before enabling setup. Existing pixel processing baselines and output versions
remain unchanged.
