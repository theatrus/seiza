# seiza-satellites

`seiza-satellites` predicts topocentric satellite paths through an already
solved Seiza image. It accepts a single shutter-open interval, an observer
location, a WCS solution, and OMM JSON or TLE orbital elements.

The output is a provenance-bearing prediction. It does not claim that a
satellite was detected in the pixels. Stacked images are deliberately outside
the API: callers must provide one `SingleExposure` interval.

## Predict tracks

After solving an image with `seiza`, load the current CelesTrak active OMM set
and search its tracks against that solution. The CelesTrak cache is persistent;
reuse one `CelesTrakSource` rather than downloading for every exposure.

```rust,no_run
use seiza_satellites::{
    CelesTrakSource, ExposureProvenance, ObserverLocation, SingleExposure,
    TrackOptions, UtcTimestamp,
};

# async fn predict(wcs: &seiza::wcs::Wcs) -> seiza_satellites::Result<()> {
let source = CelesTrakSource::platform_default()?;
let elements = source.load_active().await?;
let observer = ObserverLocation::geodetic(42.466, -71.1516, 150.0)?;
let exposure = SingleExposure::from_start_and_duration(
    UtcTimestamp::parse("2026-07-19T06:12:00Z")?,
    120.0,
    observer,
    ExposureProvenance::Explicit,
)?;
let result = elements.catalog.tracks_in_footprint(
    wcs,
    (6000, 4000),
    &exposure,
    &TrackOptions::default(),
)?;
for track in result.tracks {
    println!(
        "{}: {:.1} deg max elevation",
        track.identity.display_label(),
        track.maximum_elevation_deg(),
    );
}
# Ok(())
# }
```

For reproducible offline work, use `SatelliteCatalog::open` for a local OMM
JSON or TLE file. For a historical exposure, ask the public IAU SatChecker
service for epoch-appropriate elements on demand:

```rust,no_run
use seiza_satellites::SatCheckerSource;

# async fn historical(exposure: &seiza_satellites::SingleExposure) -> seiza_satellites::Result<()> {
let history = SatCheckerSource::platform_default()?;
let elements = history.load_for_exposure(exposure).await?;
println!("using {} historical elements", elements.catalog.len());
# Ok(())
# }
```

`load_for_exposure` performs no download until the application explicitly
requests tracks. A validated response is cached permanently, subject only to
the shared size ceiling, and reused for exposures within 12 hours by default.
Use `with_cache_reuse_window` to change that policy. Cache-only reprocessing
uses `load_cached_for_exposure` and never touches the network. Record the
returned `HistoricalCatalogSnapshot` and `CatalogFingerprint` alongside
derived results.

Reusable post-prediction analysis lives here as well, so applications do not
need to reimplement it:

- `SatelliteTrack::bright_trail_risk` summarizes generic illumination and
  image-plane geometry. It intentionally stops short of deciding whether an
  application should warn, reject, or grade an exposure.
- `TrackSearchResult::into_analysis` produces compact
  `SatelliteTrackAnalysis` records with identity, element provenance, clipped
  geometry, rates, risk, and optional pixel alignment while preserving search
  accounting. The large propagation sample vectors are deliberately omitted
  from this API/cache-oriented contract.
- `trail_alignment::PixelTrailAligner` searches monochrome image pixels in a
  bounded corridor around the complete predicted polyline. The result keeps
  aligned pixel evidence separate from the orbital prediction and distinguishes
  a tested negative (`NotDetected`) from a path that could not be tested
  (`NotEvaluated`). At least half of the sampled path must have complete
  center-line and sideband coverage by default; the measured coverage is
  included in the result.

The alignment input is row-major `u16` luminance plus an ADU conversion factor,
not an application-specific FITS type. Construct one aligner per frame and
reuse it for every predicted track.

For recent images, `CelesTrakSource` asynchronously downloads and caches the
current active-satellite OMM set. CelesTrak rate-limits repeated downloads, so
keep reusing one cache directory; a rate-limited refresh falls back to the
newest previously validated snapshot when one exists.

Downloaded CelesTrak and SatChecker snapshots form a shared durable history
instead of being deleted after the next refresh. The history is bounded by 5
GiB by default; once the ceiling is exceeded, the oldest downloads are removed
until the cache fits, while the newest snapshot is always retained.
Applications can choose another ceiling with `with_cache_size_limit_bytes`.
The ceiling is enforced by active and cache-only loads, without requiring a
successful download; `prune_cache` is also available for explicit maintenance.

Cache-only consumers never need to rediscover the crate's private filename or
locking rules:

- `cached_snapshots` returns the recognized history in retrieval order;
- `load_cached` loads the newest valid snapshot without network access;
- `load_cached_for` loads the valid snapshot retrieved closest to a historical
  exposure, also without network access.

`SatCheckerSource` provides the corresponding `cached_snapshots`,
`load_cached_for`, and `load_cached_for_exposure` APIs for epoch-query history.
Its snapshot provenance distinguishes the requested epoch from download time.

Every `SatelliteCatalog` exposes a SHA-256 `CatalogFingerprint`. Persist it
beside derived predictions and compare it with the active snapshot before
reusing them; retrieval time is kept as provenance but does not change the
content identity. Offline workflows can still open a local OMM JSON or TLE
file directly and receive the same fingerprint contract.

`SingleExposure::from_midpoint_and_duration` and
`SingleExposure::from_end_and_duration` cover FITS writers whose reliable
timestamp is `DATE-AVG` or shutter close rather than shutter open. Dedicated
provenance variants preserve which interpretation was used.

The optional `serde` feature derives `Serialize`/`Deserialize` on the
prediction result types so applications can embed or persist them directly.
