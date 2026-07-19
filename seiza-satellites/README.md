# seiza-satellites

`seiza-satellites` predicts topocentric satellite paths through an already
solved Seiza image. It accepts a single shutter-open interval, an observer
location, a WCS solution, and OMM JSON or TLE orbital elements.

The output is a provenance-bearing prediction. It does not claim that a
satellite was detected in the pixels. Stacked images are deliberately outside
the API: callers must provide one `SingleExposure` interval.

For recent images, `CelesTrakSource` asynchronously downloads and caches the
current active-satellite OMM set. Offline and historical workflows can open a
local element file instead. CelesTrak rate-limits repeated downloads, so keep
reusing one cache directory; a rate-limited refresh falls back to the newest
previously validated snapshot when one exists.

The optional `serde` feature derives `Serialize`/`Deserialize` on the
prediction result types so applications can embed or persist them directly.
