# File-backed calibration preparation

`seiza_stacking::calibration_files::FileCalibration` coordinates the native
calibration library for hosts that have files rather than a catalog database.
It owns discovery, physical file identities, measurements, response/evidence
caches, master dependency execution, native stability retries, pedestal fits,
and atomic corrected-master publication. Numerical policy remains in
`seiza_calibration::library`; pixel operations remain in their existing kernels.

The implementation was moved from Seiza Mac #92. macOS bookmarks and scoped
access, user preferences, UI, cache eviction and output grouping stay in hosts.
Windows and CLI callers can use the same coordinator. PSF Guard can continue to
use `RuleBook`, `QualityIndex`, `Library`, sessions and evidence stages directly,
with its database, cached measurements and already-loaded pixels. None of those
lower-level APIs requires this file coordinator or its cache layout.

## Native Rust

```rust,ignore
use seiza_stacking::{CancelSignal, FileCalibration};
use seiza_calibration::library::Association;

let mut files = FileCalibration::new(lights, cache_directory);
files.library = Some(calibration_directory);
files.provided = additional_files;
files.rules = validated_rule_book;
files.options.associations.flat = Association::Automatic;
files.options.associations.dark = Association::OneSet;
let review = files.analyze(cancel.clone(), |update| show_progress(update))?;
// Present review.frames, snapshot, flat_work, light_options and missing.
let prepared = files.prepare(&review.review_key, cancel, |update| show_progress(update))?;
seiza_stacking::calibration_files::verify_sources(&prepared.sources)?;
// Use prepared.paths, file_sessions and snapshot.assignments on either canvas.
```

`PreparationOptions` holds native builder/association policy and source-bound
quality/group/side choices. Defaults stop when a light's calibration is
incomplete. `Keep` prepares available calibration; `Exclude` returns original
order and `missing` flags so the host can omit those lights without reordering
the rest. The review's `missing` flags come before that choice, so changing
`options.incomplete` after analysis needs no new review. Review integrates no
masters. Execution rebuilds the review from
current sources/settings/choices before accepting its identity. A required
correction that cannot be produced is an error, never an uncorrected flat
presented as corrected. Source inputs are never modified.

Rust receives typed report records and native session/selection explanations.
There is no JSON call between the Rust coordinator and the calibration engine.
`PreparationError` says what the host does next: `Cancelled`; `ReviewChanged`
(analyze again; `changed` names the reviewed file that changed, when one did);
`Incomplete` (`IncompleteLights::Stop` refused lights without complete
calibration: review them or choose another policy); or `Failed(reason)`.
Cancellation interrupts cache lock waits,
measurement boundaries and master building; completed immutable cache entries
remain reusable. No partial master is published.

Progress is synchronous on the coordinating thread: a `PreparationPhase`,
completed and total. A zero total denotes discovery. Callbacks must not reenter the same cache;
platform adapters should dispatch UI updates without blocking on a waiting UI
thread. No callback is retained after the method returns.

## C ABI

`seiza_calibration_files_json(request, cancel, progress, context, error_out)` is
one thin workflow export over that object. The cancellation handle uses
`seiza_cancel_signal_create/cancel/free`; progress receives a borrowed phase
string, completed, total and context. Keep arguments/callbacks alive until the
blocking call returns. Run it off the UI thread; release result/error strings
with `seiza_string_free`.

An analysis request uses camelCase properties:

```json
{
  "operation": "analyze",
  "lights": ["/capture/a.fits", "/capture/b.xisf"],
  "library": "/calibration",
  "provided": [],
  "cache": "/cache/calibration",
  "settings": [{"key": "rotation_tolerance_deg", "value": 2}],
  "rigs": [],
  "flatAssociation": "automatic",
  "darkAssociation": "one_set",
  "incomplete": "stop"
}
```

Execute sends the same configuration, `operation: "execute"`, and the returned
`reviewKey`. Verify takes `operation: "verify"` and the returned `sources`;
schema takes `operation: "schema"` and returns native settings controls.
Every request property, nested row and report property is camelCase; enum
values and setting-key strings retain native spellings. Unknown and snake_case
properties are refused. The generated calibration-library JSON Schema describes
request and operation-dependent answers; bindings hold no selection policy.

## Python

```python
job = seiza.CalibrationFiles(
    lights, cache_directory, library=calibration_directory,
    rules=seiza.CalibrationRules({"rotation_tolerance_deg": 2}),
    flat_association="automatic", dark_association="one_set",
)
review = job.analyze(cancel=cancelled, progress=show_progress)
prepared = job.prepare(review["review_key"], cancel=cancelled)
seiza.CalibrationFiles.verify(prepared["sources"])
```

The class binds native methods directly with the GIL released during work.
Reports use snake_case. `cancel`/`progress` callback exceptions and Ctrl-C are
preserved. Python converts records directly with pythonize, without JSON text
or C-only request objects. Database callers retain the independent
`CalibrationLibrary`, quality, session and flat-evidence classes.

## Cache and source guarantees

The host selects an explicit cache directory. Calls serialize per cache through
a portable file lock. File aliases/hard links are recognized on supported
platforms; duplicate lights and calibration/light aliases are refused. Linked
files are read like any other; linked directories, and links into the cache,
are not followed, and the cache is excluded from discovery. A library file
refused for what it is (a light, an already calibrated frame) is remembered
by its stamp, so it is not decoded again on the next call.
Native readers/writers handle FITS and XISF through `seiza-image-io`.

Measurements, raw response planes, light bins, evidence, integrated masters and
corrected masters have separate identities. Rule changes regrade cached
measurements rather than rereading pixels. Master/evidence identities include
source stamps, native policy/metric versions, relevant options and output
versions. Settings and all source-bound choices participate in review identity;
the incomplete-light policy does not.

Stamps are metadata identities rather than pixel hashes: canonical path, size
and high-resolution modification time, plus Unix device/inode or other
platform creation metadata. Replaced/corrupt cached artifacts are rebuilt or
refused. Raw sources, staged files, response dependencies and final masters are
verified before the report returns. Call `verify_sources` before stacking and
retain the native stacker's own source/master validation and replay protection.
The host owns cache eviction; this API never prunes a library or cache.
