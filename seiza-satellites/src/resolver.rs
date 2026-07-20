use crate::{
    CacheState, CelesTrakLoad, CelesTrakSource, Result, SatCheckerLoad, SatCheckerSource,
    SatelliteCatalog, SeizaMirrorLoad, SeizaMirrorSource, SingleExposure, UtcTimestamp,
};
use directories::ProjectDirs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Exposure age after which a current active catalog is no longer selected.
/// This matches [`crate::TrackOptions::default`] element-age protection.
pub const CURRENT_CATALOG_MAX_AGE: Duration = Duration::from_secs(7 * 24 * 60 * 60);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum OrbitalCatalogProvider {
    CelesTrakActive,
    SeizaMirror,
    IauSatChecker,
}

/// Provider-neutral provenance for one resolved orbital catalog.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OrbitalCatalogSnapshot {
    pub provider: OrbitalCatalogProvider,
    pub cache_path: PathBuf,
    /// Epoch requested from a historical provider. Current catalogs have no
    /// query epoch.
    pub query_time: Option<UtcTimestamp>,
    pub retrieved_at: UtcTimestamp,
    pub size_bytes: u64,
}

#[derive(Debug)]
pub struct OrbitalCatalogLoad {
    pub catalog: SatelliteCatalog,
    pub state: CacheState,
    pub cache_path: PathBuf,
    pub snapshot: OrbitalCatalogSnapshot,
    pub warning: Option<String>,
}

/// Provider resolver for one exposure. Applications ask for elements by
/// exposure time; current-versus-historical source selection stays here.
#[derive(Clone, Debug)]
pub struct OrbitalCatalogSource {
    active: CelesTrakSource,
    mirror: SeizaMirrorSource,
    historical: SatCheckerSource,
}

impl OrbitalCatalogSource {
    pub fn new(cache_dir: impl Into<PathBuf>) -> Result<Self> {
        let cache_dir = cache_dir.into();
        Ok(Self {
            active: CelesTrakSource::new(cache_dir.clone())?,
            mirror: SeizaMirrorSource::new(cache_dir.clone())?,
            historical: SatCheckerSource::new(cache_dir)?,
        })
    }

    pub fn platform_default() -> Result<Self> {
        let cache_dir = ProjectDirs::from("fyi", "Seiza", "seiza")
            .map(|dirs| dirs.cache_dir().join("satellites"))
            .ok_or(crate::Error::NoCacheDirectory)?;
        Self::new(cache_dir)
    }

    pub fn cache_dir(&self) -> &Path {
        self.active.cache_dir()
    }

    pub fn active_source(&self) -> &CelesTrakSource {
        &self.active
    }

    pub fn historical_source(&self) -> &SatCheckerSource {
        &self.historical
    }

    pub fn mirror_source(&self) -> &SeizaMirrorSource {
        &self.mirror
    }

    /// Override the current-catalog endpoint for a caching proxy or fixture.
    pub fn with_active_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.active = self.active.with_endpoint(endpoint);
        self
    }

    /// Override the historical epoch endpoint for a mirror, proxy, or fixture.
    /// It must implement SatChecker's `epoch=<JD>&format=txt` contract.
    pub fn with_satchecker_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.historical = self.historical.with_endpoint(endpoint);
        self
    }

    /// Override the Seiza-hosted rolling manifest base URL.
    pub fn with_mirror_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.mirror = self.mirror.with_base_url(base_url);
        self
    }

    pub fn with_cache_size_limit_bytes(mut self, limit: u64) -> Self {
        self.active = self.active.with_cache_size_limit_bytes(limit);
        self.mirror = self.mirror.with_cache_size_limit_bytes(limit);
        self.historical = self.historical.with_cache_size_limit_bytes(limit);
        self
    }

    pub fn with_historical_reuse_window(mut self, window: Duration) -> Self {
        self.mirror = self.mirror.with_cache_reuse_window(window);
        self.historical = self.historical.with_cache_reuse_window(window);
        self
    }

    pub fn has_cached_catalogs(&self) -> Result<bool> {
        Ok(!self.active.cached_snapshots()?.is_empty()
            || !self.mirror.cached_snapshots()?.is_empty()
            || !self.historical.cached_snapshots()?.is_empty())
    }

    /// Resolve the appropriate catalog on demand for normal application use.
    pub async fn load_for_exposure(&self, exposure: &SingleExposure) -> Result<OrbitalCatalogLoad> {
        if should_use_historical(exposure.midpoint(), now_unix_seconds()) {
            self.load_historical_at(exposure.midpoint()).await
        } else {
            self.active
                .load_active()
                .await
                .map(OrbitalCatalogLoad::from)
        }
    }

    /// Explicitly prewarm one historical epoch through the configured
    /// historical provider. Normal applications should prefer
    /// [`Self::load_for_exposure`].
    pub async fn prewarm_historical(&self, time_utc: UtcTimestamp) -> Result<OrbitalCatalogLoad> {
        self.load_historical_at(time_utc).await
    }

    async fn load_historical_at(&self, time_utc: UtcTimestamp) -> Result<OrbitalCatalogLoad> {
        if let Some(load) = self.load_cached_historical(time_utc)? {
            let distance = load
                .snapshot
                .query_time
                .expect("historical catalogs have a query epoch")
                .seconds_since(time_utc)
                .abs();
            if distance <= self.historical.cache_reuse_window().as_secs_f64() {
                return Ok(load);
            }
        }
        match self.mirror.load_at(time_utc).await {
            Ok(load) => Ok(load.into()),
            Err(mirror_error) => {
                let mut load = OrbitalCatalogLoad::from(self.historical.load_at(time_utc).await?);
                if !matches!(mirror_error, crate::Error::HttpStatus { status: 404, .. }) {
                    load.warning = Some(format!(
                        "Seiza satellite mirror unavailable; used IAU SatChecker: {mirror_error}"
                    ));
                }
                Ok(load)
            }
        }
    }

    /// Fetch one exact historical epoch from the public origin for mirror
    /// publication. An exact cached origin response is reused, but the Seiza
    /// mirror and nearby-cache window are intentionally bypassed so a rolling
    /// publisher cannot keep republishing its previous bucket.
    pub async fn fetch_historical_origin(
        &self,
        time_utc: UtcTimestamp,
    ) -> Result<OrbitalCatalogLoad> {
        self.historical
            .clone()
            .with_cache_reuse_window(Duration::ZERO)
            .load_at(time_utc)
            .await
            .map(OrbitalCatalogLoad::from)
    }

    /// Resolve the best cached source for an exposure without network I/O.
    /// The preferred source is tried first; the other durable history is a
    /// fallback and downstream element-age checks still reject unsuitable
    /// extrapolation.
    pub fn load_cached_for_exposure(
        &self,
        exposure: &SingleExposure,
    ) -> Result<Option<OrbitalCatalogLoad>> {
        if should_use_historical(exposure.midpoint(), now_unix_seconds()) {
            if let Some(load) = self.load_cached_historical(exposure.midpoint())? {
                return Ok(Some(load));
            }
            return self
                .active
                .load_cached_for(exposure.midpoint())
                .map(|load| load.map(Into::into));
        }
        if let Some(load) = self.active.load_cached_for(exposure.midpoint())? {
            return Ok(Some(load.into()));
        }
        self.load_cached_historical(exposure.midpoint())
    }

    fn load_cached_historical(&self, time_utc: UtcTimestamp) -> Result<Option<OrbitalCatalogLoad>> {
        let mirror: Option<OrbitalCatalogLoad> =
            self.mirror.load_cached_for(time_utc)?.map(Into::into);
        let satchecker: Option<OrbitalCatalogLoad> =
            self.historical.load_cached_for(time_utc)?.map(Into::into);
        Ok(match (mirror, satchecker) {
            (Some(mirror), Some(satchecker)) => {
                let mirror_distance = mirror
                    .snapshot
                    .query_time
                    .expect("mirror history has a query epoch")
                    .seconds_since(time_utc)
                    .abs();
                let satchecker_distance = satchecker
                    .snapshot
                    .query_time
                    .expect("SatChecker history has a query epoch")
                    .seconds_since(time_utc)
                    .abs();
                Some(if mirror_distance <= satchecker_distance {
                    mirror
                } else {
                    satchecker
                })
            }
            (Some(load), None) | (None, Some(load)) => Some(load),
            (None, None) => None,
        })
    }
}

impl From<CelesTrakLoad> for OrbitalCatalogLoad {
    fn from(load: CelesTrakLoad) -> Self {
        let snapshot = OrbitalCatalogSnapshot {
            provider: OrbitalCatalogProvider::CelesTrakActive,
            cache_path: load.snapshot.cache_path.clone(),
            query_time: None,
            retrieved_at: load.snapshot.retrieved_at,
            size_bytes: load.snapshot.size_bytes,
        };
        Self {
            catalog: load.catalog,
            state: load.state,
            cache_path: load.cache_path,
            snapshot,
            warning: load.warning,
        }
    }
}

impl From<SatCheckerLoad> for OrbitalCatalogLoad {
    fn from(load: SatCheckerLoad) -> Self {
        let snapshot = OrbitalCatalogSnapshot {
            provider: OrbitalCatalogProvider::IauSatChecker,
            cache_path: load.snapshot.cache_path.clone(),
            query_time: Some(load.snapshot.query_time),
            retrieved_at: load.snapshot.downloaded_at,
            size_bytes: load.snapshot.size_bytes,
        };
        Self {
            catalog: load.catalog,
            state: load.state,
            cache_path: load.cache_path,
            snapshot,
            warning: None,
        }
    }
}

impl From<SeizaMirrorLoad> for OrbitalCatalogLoad {
    fn from(load: SeizaMirrorLoad) -> Self {
        let snapshot = OrbitalCatalogSnapshot {
            provider: OrbitalCatalogProvider::SeizaMirror,
            cache_path: load.snapshot.cache_path.clone(),
            query_time: Some(load.snapshot.query_time),
            retrieved_at: load.snapshot.downloaded_at,
            size_bytes: load.snapshot.size_bytes,
        };
        Self {
            catalog: load.catalog,
            state: load.state,
            cache_path: load.cache_path,
            snapshot,
            warning: None,
        }
    }
}

fn now_unix_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn should_use_historical(exposure_midpoint: UtcTimestamp, now_unix_seconds: f64) -> bool {
    now_unix_seconds - exposure_midpoint.unix_seconds() > CURRENT_CATALOG_MAX_AGE.as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TLE: &str = include_str!("../tests/data/iss-2024.tle");

    #[test]
    fn resolver_owns_current_versus_historical_selection() {
        let now = UtcTimestamp::parse("2026-07-19T12:00:00Z").unwrap();
        let recent = now
            .add_seconds(-CURRENT_CATALOG_MAX_AGE.as_secs_f64())
            .unwrap();
        let historical = recent.add_seconds(-1.0).unwrap();
        let future = now.add_seconds(24.0 * 60.0 * 60.0).unwrap();

        assert!(!should_use_historical(recent, now.unix_seconds()));
        assert!(should_use_historical(historical, now.unix_seconds()));
        assert!(!should_use_historical(future, now.unix_seconds()));
    }

    #[tokio::test]
    async fn prewarm_uses_nearby_durable_cache_before_any_provider_network() {
        let dir = tempfile::tempdir().unwrap();
        let time = UtcTimestamp::parse("2025-10-18T12:00:00Z").unwrap();
        let query_millis = (time.unix_seconds() * 1_000.0).round() as i64;
        std::fs::write(
            dir.path().join(format!(
                "satchecker-epoch-{query_millis}-cached-1760900000.tle"
            )),
            TLE,
        )
        .unwrap();
        let source = OrbitalCatalogSource::new(dir.path())
            .unwrap()
            .with_mirror_base_url("http://127.0.0.1:1/never-called")
            .with_satchecker_endpoint("http://127.0.0.1:1/never-called");

        let load = source.prewarm_historical(time).await.unwrap();
        assert_eq!(load.state, CacheState::Cached);
        assert_eq!(
            load.snapshot.provider,
            OrbitalCatalogProvider::IauSatChecker
        );
        assert_eq!(load.catalog.len(), 1);
    }
}
