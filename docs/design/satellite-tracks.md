# Single-exposure satellite track overlays

Status: first prediction layer implemented; pixel-evidence matching proposed

## Goal and boundary

After Seiza has solved a single exposure, predict which Earth-orbiting objects
crossed the image while the shutter was open and draw their paths as an
optional annotation layer. Satellite prediction never participates in plate
solving and a missing time, observer, element source, or network connection
must not turn an otherwise successful solve into a failed solve unless the
caller explicitly requested the layer.

This feature does not operate on stacks. A stack's total integration is not a
continuous shutter-open interval and stacking normally rejects moving trails.
The library therefore accepts a `SingleExposure` value rather than a generic
image timestamp. Applications that know an image is stacked do not construct
that value.

The first result is a **predicted crossing**, not a pixel detection. An object
can be geometrically present while eclipsed, too faint, outside the camera's
actual shutter interval, or hidden by clouds. A later matcher may associate a
detected trail with one or more predictions and attach residuals and a
confidence, but it must preserve the prediction and pixel evidence as distinct
provenance layers.

## Observation contract

`SingleExposure` contains:

- UTC start and end timestamps;
- an ITRF Cartesian or geodetic observer location;
- metadata provenance (`Explicit`, `FitsBounds`, or
  `FitsDateObsAndExposure`).

The CLI resolves those values in this order:

1. explicit `--time`, `--exposure-seconds`, and observer arguments;
2. FITS `DATE-BEG` and `DATE-END`;
3. FITS `DATE-OBS` plus `XPOSURE`, `EXPTIME`, or `EXPOSURE`.

The FITS standard gives `DATE-BEG` and `DATE-END` unambiguous acquisition-bound
semantics. `DATE-OBS` is only a fallback because historical files have also
used it for an average observation time. `TIMESYS` must be absent or `UTC` for
the current CLI path. Standard observer keywords are preferred in this order:
`OBSGEO-X/Y/Z`, then `OBSGEO-B/L/H`. Explicit arguments override the header.
The CLI also recognizes the common non-standard `SITELAT`, `SITELONG`, and
optional `SITEALT` fallback used by capture applications, after the standard
keywords.

## Orbital elements and caching

Orbital elements are rapidly changing runtime inputs, not an `objects.bin`
section or a hosted static catalog bundle. `SatelliteCatalog::open` accepts
CCSDS OMM JSON and legacy two-/three-line element files for offline and
historical use.

`CelesTrakSource` supplies recent active-satellite OMM JSON. This first source
does not include the complete debris and rocket-body population. It stores one
validated response in the platform cache directory, never refreshes it more
often than CelesTrak's two-hour update interval, writes replacements
atomically, coordinates refreshes across processes, and may use a previously
validated stale snapshot if refresh fails.
The result records whether the snapshot was fresh, downloaded, or a stale
fallback. A non-success HTTP status is returned immediately without retrying;
403 and 429 map to a dedicated rate-limit error carrying any `Retry-After`
value, because CelesTrak blocks clients that re-download inside its update
window. Responses are requested gzip-compressed. Readers of a fresh snapshot
hold only a shared file lock so concurrent processes do not serialize; the
exclusive lock is taken for refresh, and the freshness check repeats after
acquiring it in case another process refreshed first. A snapshot whose
modification time is more than a few minutes in the future is treated as
maximally stale rather than permanently fresh.

Historical images require elements close to their acquisition epoch. A future
authenticated Space-Track `GP_HISTORY` provider can implement the same source
boundary. It should cache each historical response indefinitely rather than
repeat large history queries.

Stable identity uses the numeric NORAD catalog ID and, when available, the
COSPAR international designator. Display names are not keys. OMM is preferred
to fixed-width TLE for new downloads because it supports the expanding
catalog-number range and preserves metadata.

## Projection pipeline

For each element record:

1. propagate SGP4 positions in TEME across the shutter-open interval;
2. rotate TEME to ITRF and subtract the observer position for topocentric
   parallax;
3. rotate the relative vector to GCRF and derive RA/Dec;
4. compute local elevation, range, and sunlight fraction;
5. project samples through `Wcs::world_to_pixel` and clip segments to the image.

The implementation uses SatKit's offline IAU-76/FK5 approximate ITRF/GCRF
transform, documented at about one arcsecond, so it needs no Earth-orientation
download. Element uncertainty and camera timestamp error will often dominate,
but the approximation and element epoch are retained in the result provenance.

A coarse pass samples every ten seconds to discard tracks that do not intersect
the image. Because the true apparent path between two coarse samples is curved
while the test uses straight chords, the coarse pass clips against an image
rectangle padded by a tenth of each chord's length — a bound comfortably above
the worst-case sagitta for low-orbit rates — and skips the elevation gate, so
it can only over-admit; the fine pass remains authoritative. Candidate tracks
are then sampled at the configured fine step (one second by default). The
sample interval remains visible in the result so callers can choose a tighter
value for narrow, high-resolution fields, and each track reports its peak
apparent and image-plane rates so callers can detect when a crossing outruns
the sampling.
The default maximum absolute element age is seven days. Older records are
reported and skipped instead of silently extrapolating today's orbit back onto
an archival image; callers doing controlled research may override the policy.

## API and CLI

The library entry point is:

```rust,ignore
catalog.tracks_in_footprint(&wcs, dimensions, &exposure, &TrackOptions::default())
```

Each `SatelliteTrack` contains identity, element epoch, source provenance,
time-tagged topocentric samples, clipped pixel segments, elevation, range, and
sunlight fraction. `association` is currently always `Predicted`.

`tracks_in_footprint` borrows the catalog immutably — SGP4 initialization is
cached on a per-call scratch copy of each element — so one loaded catalog can
serve concurrent solves. The optional `serde` crate feature derives
`Serialize`/`Deserialize` on the exposure, option, and result types for
embedding in application APIs and caches.

The CLI enables the layer explicitly with either:

```shell
seiza solve image.fits ... \
  --satellites-celestrak \
  --observer-lat 37.3 --observer-lon -122.0 --observer-alt-m 50 \
  --annotate solved.png

seiza solve historical.fits ... \
  --satellites elements-near-exposure.json \
  --annotate solved.png
```

The annotation draws the predicted path, direction and identity in a distinct
color. CLI text says "predicted satellite tracks" and reports the element age;
it never calls an unmatched prediction observed or detected.

## Follow-on work

1. Add an authenticated, permanently cached Space-Track history provider.
2. Detect trail candidates after masking stars and compare their geometry with
   predicted paths.
3. Report all plausible identities when timing or elements do not distinguish
   nearby tracks; never force a single match.
4. Add optional timing-error and orbit-uncertainty envelopes.
5. Benchmark full active-catalog propagation and introduce observer/time-bucket
   indices only if measured workloads require them.
