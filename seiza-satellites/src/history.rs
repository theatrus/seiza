use crate::cache::{
    self, DEFAULT_SATELLITE_CACHE_SIZE_LIMIT_BYTES, LockMode, SATCHECKER_CACHE_PREFIX,
    SATCHECKER_CACHE_SUFFIX, TimeSnapshot, acquire_lock_in, now_timestamp,
};
use crate::{CacheState, Error, Result, SatelliteCatalog, SingleExposure, UtcTimestamp};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Public IAU Centre for the Protection of the Dark and Quiet Sky historical
/// TLE service.
pub const SATCHECKER_TLES_AT_EPOCH_URL: &str =
    "https://satchecker.cps.iau.org/tools/tles-at-epoch/";

/// Reuse a validated epoch query for other exposures in the same observing
/// night. Applications can tighten or disable this window when required.
pub const SATCHECKER_CACHE_REUSE_WINDOW: Duration = Duration::from_secs(12 * 60 * 60);

/// Provenance for one durable SatChecker historical-catalog response.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HistoricalCatalogSnapshot {
    pub cache_path: PathBuf,
    /// Epoch sent to SatChecker, normally the exposure midpoint.
    pub query_time: UtcTimestamp,
    /// Time at which this response was downloaded and validated.
    pub downloaded_at: UtcTimestamp,
    pub source_url: String,
    pub size_bytes: u64,
}

#[derive(Debug)]
pub struct SatCheckerLoad {
    pub catalog: SatelliteCatalog,
    pub state: CacheState,
    pub cache_path: PathBuf,
    pub snapshot: HistoricalCatalogSnapshot,
}

/// On-demand access to IAU SatChecker's epoch-appropriate historical TLEs.
///
/// A network request is made only when [`Self::load_for_exposure`] or
/// [`Self::load_at`] is called and no validated nearby response is cached.
/// Cache-only methods never perform network I/O.
#[derive(Clone, Debug)]
pub struct SatCheckerSource {
    client: reqwest::Client,
    cache_dir: PathBuf,
    endpoint: String,
    cache_size_limit_bytes: u64,
    cache_reuse_window: Duration,
}

impl SatCheckerSource {
    pub fn new(cache_dir: impl Into<PathBuf>) -> Result<Self> {
        let client = reqwest::Client::builder()
            .user_agent(format!("seiza-satellites/{}", env!("CARGO_PKG_VERSION")))
            .connect_timeout(Duration::from_secs(30))
            .read_timeout(Duration::from_secs(120))
            .build()
            .map_err(Error::HttpClient)?;
        Ok(Self {
            client,
            cache_dir: cache_dir.into(),
            endpoint: SATCHECKER_TLES_AT_EPOCH_URL.into(),
            cache_size_limit_bytes: DEFAULT_SATELLITE_CACHE_SIZE_LIMIT_BYTES,
            cache_reuse_window: SATCHECKER_CACHE_REUSE_WINDOW,
        })
    }

    pub fn platform_default() -> Result<Self> {
        Self::new(cache::platform_cache_dir()?)
    }

    /// Override the endpoint for an organization-local proxy or test fixture.
    /// The endpoint must accept `epoch` (Julian date) and `format=txt` query
    /// parameters using SatChecker's response contract.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Override the shared durable orbital-element cache ceiling. This is the
    /// same cache directory and bound used by [`crate::CelesTrakSource`].
    pub fn with_cache_size_limit_bytes(mut self, cache_size_limit_bytes: u64) -> Self {
        self.cache_size_limit_bytes = cache_size_limit_bytes;
        self
    }

    /// Override how close a cached SatChecker query must be to avoid another
    /// network request. A zero duration requires an exact cached epoch.
    pub fn with_cache_reuse_window(mut self, cache_reuse_window: Duration) -> Self {
        self.cache_reuse_window = cache_reuse_window;
        self
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn cache_size_limit_bytes(&self) -> u64 {
        self.cache_size_limit_bytes
    }

    pub fn cache_reuse_window(&self) -> Duration {
        self.cache_reuse_window
    }

    /// List recognized historical snapshots in query-epoch order without
    /// network I/O.
    pub fn cached_snapshots(&self) -> Result<Vec<HistoricalCatalogSnapshot>> {
        self.prune_cache()?;
        let _shared = acquire_lock_in(&self.cache_dir, LockMode::Shared)?;
        history_inventory_in(&self.cache_dir, &self.endpoint)
    }

    /// Load the valid cached historical response nearest an exposure without
    /// network I/O. This intentionally has no distance cutoff; the returned
    /// query epoch lets the caller decide whether the snapshot is suitable.
    pub fn load_cached_for_exposure(
        &self,
        exposure: &SingleExposure,
    ) -> Result<Option<SatCheckerLoad>> {
        self.load_cached_for(exposure.midpoint())
    }

    /// Load the valid cached historical response nearest a timestamp without
    /// network I/O.
    pub fn load_cached_for(&self, time_utc: UtcTimestamp) -> Result<Option<SatCheckerLoad>> {
        self.prune_cache()?;
        let _shared = acquire_lock_in(&self.cache_dir, LockMode::Shared)?;
        self.load_nearest_blocking(time_utc, None)
    }

    /// Resolve epoch-appropriate elements on demand for one exposure. A
    /// nearby validated response is reused; otherwise SatChecker is queried
    /// at the shutter midpoint and the result is cached atomically.
    pub async fn load_for_exposure(&self, exposure: &SingleExposure) -> Result<SatCheckerLoad> {
        self.load_at(exposure.midpoint()).await
    }

    /// Resolve epoch-appropriate elements on demand for a UTC timestamp.
    pub async fn load_at(&self, time_utc: UtcTimestamp) -> Result<SatCheckerLoad> {
        tokio::fs::create_dir_all(&self.cache_dir)
            .await
            .map_err(|source| Error::Io {
                path: self.cache_dir.clone(),
                source,
            })?;
        self.prune_cache_async().await?;

        {
            let _shared = self.acquire_lock(LockMode::Shared).await?;
            if let Some(load) = self
                .load_nearest_async(time_utc, Some(self.cache_reuse_window))
                .await?
            {
                return Ok(load);
            }
        }

        let _exclusive = self.acquire_lock(LockMode::Exclusive).await?;
        if let Some(load) = self
            .load_nearest_async(time_utc, Some(self.cache_reuse_window))
            .await?
        {
            return Ok(load);
        }
        self.download(time_utc).await
    }

    /// Enforce the shared history ceiling without network I/O.
    pub fn prune_cache(&self) -> Result<()> {
        cache::prune_cache_blocking(&self.cache_dir, self.cache_size_limit_bytes)
    }

    async fn prune_cache_async(&self) -> Result<()> {
        cache::prune_cache_async(&self.cache_dir, self.cache_size_limit_bytes).await
    }

    async fn acquire_lock(&self, mode: LockMode) -> Result<File> {
        cache::acquire_lock(&self.cache_dir, mode).await
    }

    async fn load_nearest_async(
        &self,
        time_utc: UtcTimestamp,
        maximum_distance: Option<Duration>,
    ) -> Result<Option<SatCheckerLoad>> {
        let cache_dir = self.cache_dir.clone();
        let endpoint = self.endpoint.clone();
        let snapshots =
            tokio::task::spawn_blocking(move || history_inventory_in(&cache_dir, &endpoint))
                .await
                .map_err(|error| Error::CacheLock(error.to_string()))??;
        cache::select_nearest_async(snapshots, time_utc, maximum_distance, async |snapshot| {
            self.load_snapshot_async(snapshot).await
        })
        .await
    }

    fn load_nearest_blocking(
        &self,
        time_utc: UtcTimestamp,
        maximum_distance: Option<Duration>,
    ) -> Result<Option<SatCheckerLoad>> {
        let snapshots = history_inventory_in(&self.cache_dir, &self.endpoint)?;
        cache::select_nearest_blocking(snapshots, time_utc, maximum_distance, |snapshot| {
            self.load_snapshot_blocking(snapshot)
        })
    }

    async fn download(&self, query_time: UtcTimestamp) -> Result<SatCheckerLoad> {
        let epoch = julian_date(query_time);
        let response = self
            .client
            .get(&self.endpoint)
            .query(&[("epoch", format!("{epoch:.8}")), ("format", "txt".into())])
            .send()
            .await
            .map_err(|source| Error::Http {
                url: self.endpoint.clone(),
                source,
            })?;
        let request_url = response.url().to_string();
        let status = response.status();
        if !status.is_success() {
            return Err(Error::HttpStatus {
                url: request_url,
                status: status.as_u16(),
            });
        }
        let body =
            cache::read_body_capped(response, cache::MAX_SATELLITE_RESPONSE_BYTES, &request_url)
                .await?;
        let payload = String::from_utf8(body).map_err(|error| Error::Elements {
            source_name: request_url.clone(),
            message: error.to_string(),
        })?;
        let downloaded_at = now_timestamp()?;
        let catalog = SatelliteCatalog::from_tle_text(&payload, request_url.clone())?
            .with_retrieved_at(downloaded_at);

        let query_millis = (query_time.unix_seconds() * 1_000.0).round() as i64;
        let downloaded_seconds = downloaded_at.unix_seconds().floor() as i64;
        let final_path = self.cache_dir.join(format!(
            "{SATCHECKER_CACHE_PREFIX}{query_millis}-cached-{downloaded_seconds}{SATCHECKER_CACHE_SUFFIX}"
        ));
        let temporary_path = self.cache_dir.join(format!(
            ".{SATCHECKER_CACHE_PREFIX}{query_millis}-{}-{}.partial",
            downloaded_seconds,
            std::process::id()
        ));
        cache::publish_atomically(&temporary_path, &final_path, payload.as_bytes()).await?;
        cache::enforce_size_limit_async(
            &self.cache_dir,
            Some(&final_path),
            self.cache_size_limit_bytes,
        )
        .await?;

        let snapshot = HistoricalCatalogSnapshot {
            cache_path: final_path.clone(),
            query_time,
            downloaded_at,
            source_url: request_url,
            size_bytes: payload.len() as u64,
        };
        Ok(SatCheckerLoad {
            catalog,
            state: CacheState::Downloaded,
            cache_path: final_path,
            snapshot,
        })
    }

    async fn load_snapshot_async(
        &self,
        snapshot: HistoricalCatalogSnapshot,
    ) -> Result<SatCheckerLoad> {
        let payload = tokio::fs::read_to_string(&snapshot.cache_path)
            .await
            .map_err(|source| Error::Io {
                path: snapshot.cache_path.clone(),
                source,
            })?;
        self.parsed_snapshot(payload, snapshot)
    }

    fn load_snapshot_blocking(
        &self,
        snapshot: HistoricalCatalogSnapshot,
    ) -> Result<SatCheckerLoad> {
        let payload =
            std::fs::read_to_string(&snapshot.cache_path).map_err(|source| Error::Io {
                path: snapshot.cache_path.clone(),
                source,
            })?;
        self.parsed_snapshot(payload, snapshot)
    }

    fn parsed_snapshot(
        &self,
        payload: String,
        snapshot: HistoricalCatalogSnapshot,
    ) -> Result<SatCheckerLoad> {
        let catalog = SatelliteCatalog::from_tle_text(&payload, snapshot.source_url.clone())?
            .with_retrieved_at(snapshot.downloaded_at);
        Ok(SatCheckerLoad {
            catalog,
            state: CacheState::Cached,
            cache_path: snapshot.cache_path.clone(),
            snapshot,
        })
    }
}

impl TimeSnapshot for HistoricalCatalogSnapshot {
    fn query_time(&self) -> UtcTimestamp {
        self.query_time
    }

    fn downloaded_at(&self) -> UtcTimestamp {
        self.downloaded_at
    }
}

fn history_inventory_in(
    cache_dir: &Path,
    endpoint: &str,
) -> Result<Vec<HistoricalCatalogSnapshot>> {
    let entries = match std::fs::read_dir(cache_dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(source) => {
            return Err(Error::Io {
                path: cache_dir.to_path_buf(),
                source,
            });
        }
    };
    let mut snapshots = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| Error::Io {
            path: cache_dir.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some((query_millis, downloaded_seconds)) = name
            .strip_prefix(SATCHECKER_CACHE_PREFIX)
            .and_then(|value| value.strip_suffix(SATCHECKER_CACHE_SUFFIX))
            .and_then(|value| value.split_once("-cached-"))
            .and_then(|(query, cached)| {
                Some((query.parse::<i64>().ok()?, cached.parse::<i64>().ok()?))
            })
        else {
            continue;
        };
        let metadata = entry.metadata().map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        if !metadata.is_file() {
            continue;
        }
        let query_time = UtcTimestamp::from_unix_seconds(query_millis as f64 / 1_000.0)?;
        let separator = if endpoint.contains('?') { '&' } else { '?' };
        snapshots.push(HistoricalCatalogSnapshot {
            cache_path: path,
            query_time,
            downloaded_at: UtcTimestamp::from_unix_seconds(downloaded_seconds as f64)?,
            source_url: format!(
                "{endpoint}{separator}epoch={:.8}&format=txt",
                julian_date(query_time)
            ),
            size_bytes: metadata.len(),
        });
    }
    snapshots.sort_by(|left, right| {
        left.query_time
            .unix_seconds()
            .total_cmp(&right.query_time.unix_seconds())
            .then_with(|| {
                left.downloaded_at
                    .unix_seconds()
                    .total_cmp(&right.downloaded_at.unix_seconds())
            })
    });
    Ok(snapshots)
}

/// Convert Unix UTC seconds to the Julian date expected by SatChecker.
fn julian_date(time_utc: UtcTimestamp) -> f64 {
    2_440_587.5 + time_utc.unix_seconds() / 86_400.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExposureProvenance, ObserverLocation};
    use std::io::{Read, Write};
    use std::sync::mpsc;

    const TLE: &str = include_str!("../tests/data/iss-2024.tle");

    fn serve_tle_once() -> (String, mpsc::Receiver<String>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (sender, receiver) = mpsc::channel();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let count = stream.read(&mut request).unwrap();
            sender
                .send(String::from_utf8_lossy(&request[..count]).into_owned())
                .unwrap();
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/plain\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{TLE}",
                        TLE.len()
                    )
                    .as_bytes(),
                )
                .unwrap();
        });
        (format!("http://{address}/tools/tles-at-epoch/"), receiver)
    }

    fn exposure(midpoint: UtcTimestamp) -> SingleExposure {
        SingleExposure::from_midpoint_and_duration(
            midpoint,
            60.0,
            ObserverLocation::geodetic(0.0, 0.0, 0.0).unwrap(),
            ExposureProvenance::Explicit,
        )
        .unwrap()
    }

    #[test]
    fn unix_epoch_has_the_standard_julian_date() {
        let epoch = UtcTimestamp::from_unix_seconds(0.0).unwrap();
        assert_eq!(julian_date(epoch), 2_440_587.5);
    }

    #[tokio::test]
    async fn downloads_epoch_tles_only_when_requested_and_then_reuses_them() {
        let dir = tempfile::tempdir().unwrap();
        let midpoint = UtcTimestamp::parse("2025-10-18T12:52:12.790Z").unwrap();
        let (endpoint, request) = serve_tle_once();
        let source = SatCheckerSource::new(dir.path())
            .unwrap()
            .with_endpoint(endpoint);

        assert!(source.cached_snapshots().unwrap().is_empty());
        let first = source.load_for_exposure(&exposure(midpoint)).await.unwrap();
        assert_eq!(first.state, CacheState::Downloaded);
        assert_eq!(first.catalog.len(), 1);
        assert_eq!(first.snapshot.query_time, midpoint);
        let request = request.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(request.starts_with("GET /tools/tles-at-epoch/?"));
        assert!(request.contains("epoch=2460967.03625914"));
        assert!(request.contains("format=txt"));

        let nearby = midpoint.add_seconds(60.0 * 60.0).unwrap();
        let second = source.load_at(nearby).await.unwrap();
        assert_eq!(second.state, CacheState::Cached);
        assert_eq!(second.cache_path, first.cache_path);
        assert_eq!(source.cached_snapshots().unwrap().len(), 1);
    }

    #[test]
    fn cache_only_lookup_selects_the_nearest_historical_query() {
        let dir = tempfile::tempdir().unwrap();
        for query in [100_000_i64, 300_000] {
            std::fs::write(
                dir.path().join(format!(
                    "{SATCHECKER_CACHE_PREFIX}{query}-cached-1000{SATCHECKER_CACHE_SUFFIX}"
                )),
                TLE,
            )
            .unwrap();
        }
        let source = SatCheckerSource::new(dir.path()).unwrap();
        let target = UtcTimestamp::from_unix_seconds(200.0).unwrap();

        let load = source.load_cached_for(target).unwrap().unwrap();

        assert_eq!(load.state, CacheState::Cached);
        assert_eq!(load.snapshot.query_time.unix_seconds(), 300.0);
        assert_eq!(load.catalog.len(), 1);
    }

    #[test]
    fn shared_ceiling_counts_current_and_historical_catalogs() {
        let dir = tempfile::tempdir().unwrap();
        let current = dir.path().join("celestrak-active-100.json");
        let historical = dir.path().join(format!(
            "{SATCHECKER_CACHE_PREFIX}200000-cached-200{SATCHECKER_CACHE_SUFFIX}"
        ));
        std::fs::write(&current, "current").unwrap();
        std::fs::write(&historical, TLE).unwrap();
        let source = SatCheckerSource::new(dir.path())
            .unwrap()
            .with_cache_size_limit_bytes(TLE.len() as u64);

        source.prune_cache().unwrap();

        assert!(!current.exists());
        assert!(historical.exists());
    }
}
