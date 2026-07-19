# seiza-satellites

`seiza-satellites` predicts topocentric satellite paths through an already
solved Seiza image. It accepts a single shutter-open interval, an observer
location, a WCS solution, and OMM JSON or TLE orbital elements.

The output is a provenance-bearing prediction. It does not claim that a
satellite was detected in the pixels. Stacked images are deliberately outside
the API: callers must provide one `SingleExposure` interval.

For recent images, `CelesTrakSource` asynchronously downloads and caches the
current active-satellite OMM set. CelesTrak rate-limits repeated downloads, so
keep reusing one cache directory; a rate-limited refresh falls back to the
newest previously validated snapshot when one exists.

Downloaded snapshots form a durable history instead of being deleted after
the next refresh. The history is bounded by 5 GiB by default; once the ceiling
is exceeded, the oldest snapshots are removed until the cache fits, while the
newest snapshot is always retained. Applications can choose another ceiling
with `with_cache_size_limit_bytes`.

Cache-only consumers never need to rediscover the crate's private filename or
locking rules:

- `cached_snapshots` returns the recognized history in retrieval order;
- `load_cached` loads the newest valid snapshot without network access;
- `load_cached_for` loads the valid snapshot retrieved closest to a historical
  exposure, also without network access.

Every `SatelliteCatalog` exposes a SHA-256 `CatalogFingerprint`. Persist it
beside derived predictions and compare it with the active snapshot before
reusing them; retrieval time is kept as provenance but does not change the
content identity. Offline workflows can still open a local OMM JSON or TLE
file directly and receive the same fingerprint contract.

`SingleExposure::from_midpoint_and_duration` is available for FITS writers
whose reliable timestamp is `DATE-AVG` rather than shutter open.

The optional `serde` feature derives `Serialize`/`Deserialize` on the
prediction result types so applications can embed or persist them directly.
