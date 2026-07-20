use crate::source::{
    DEFAULT_SATELLITE_CACHE_SIZE_LIMIT_BYTES, LockMode, SATCHECKER_CACHE_PREFIX,
    SATCHECKER_CACHE_SUFFIX, acquire_lock_in, enforce_cache_size_limit_in, now_timestamp,
};
use crate::{CacheState, Error, Result, SatelliteCatalog, SingleExposure, UtcTimestamp};
use directories::ProjectDirs;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::io::AsyncWriteExt;

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
        let cache_dir = ProjectDirs::from("fyi", "Seiza", "seiza")
            .map(|dirs| dirs.cache_dir().join("satellites"))
            .ok_or(Error::NoCacheDirectory)?;
        Self::new(cache_dir)
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
        history_inventory_in(&self.cache_dir)
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
        let _exclusive = acquire_lock_in(&self.cache_dir, LockMode::Exclusive)?;
        enforce_cache_size_limit_in(&self.cache_dir, None, self.cache_size_limit_bytes)
    }

    async fn prune_cache_async(&self) -> Result<()> {
        let cache_dir = self.cache_dir.clone();
        let limit = self.cache_size_limit_bytes;
        tokio::task::spawn_blocking(move || {
            let _exclusive = acquire_lock_in(&cache_dir, LockMode::Exclusive)?;
            enforce_cache_size_limit_in(&cache_dir, None, limit)
        })
        .await
        .map_err(|error| Error::CacheLock(error.to_string()))?
    }

    async fn acquire_lock(&self, mode: LockMode) -> Result<File> {
        let cache_dir = self.cache_dir.clone();
        tokio::task::spawn_blocking(move || acquire_lock_in(&cache_dir, mode))
            .await
            .map_err(|error| Error::CacheLock(error.to_string()))?
    }

    async fn load_nearest_async(
        &self,
        time_utc: UtcTimestamp,
        maximum_distance: Option<Duration>,
    ) -> Result<Option<SatCheckerLoad>> {
        let cache_dir = self.cache_dir.clone();
        let mut snapshots = tokio::task::spawn_blocking(move || history_inventory_in(&cache_dir))
            .await
            .map_err(|error| Error::CacheLock(error.to_string()))??;
        sort_by_distance(&mut snapshots, time_utc);
        for snapshot in snapshots {
            if !within_distance(&snapshot, time_utc, maximum_distance) {
                continue;
            }
            if let Ok(load) = self.load_snapshot_async(snapshot).await {
                return Ok(Some(load));
            }
        }
        Ok(None)
    }

    fn load_nearest_blocking(
        &self,
        time_utc: UtcTimestamp,
        maximum_distance: Option<Duration>,
    ) -> Result<Option<SatCheckerLoad>> {
        let mut snapshots = history_inventory_in(&self.cache_dir)?;
        sort_by_distance(&mut snapshots, time_utc);
        let mut last_error = None;
        for snapshot in snapshots {
            if !within_distance(&snapshot, time_utc, maximum_distance) {
                continue;
            }
            match self.load_snapshot_blocking(snapshot) {
                Ok(load) => return Ok(Some(load)),
                Err(error) => last_error = Some(error),
            }
        }
        match last_error {
            Some(error) => Err(error),
            None => Ok(None),
        }
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
        let payload = response.text().await.map_err(|source| Error::Http {
            url: request_url.clone(),
            source,
        })?;
        let downloaded_at = now_timestamp()?;
        let catalog = SatelliteCatalog::from_tle_text(&payload, request_url)?
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
        let publication = async {
            let mut file = tokio::fs::File::create(&temporary_path)
                .await
                .map_err(|source| Error::Io {
                    path: temporary_path.clone(),
                    source,
                })?;
            file.write_all(payload.as_bytes())
                .await
                .map_err(|source| Error::Io {
                    path: temporary_path.clone(),
                    source,
                })?;
            file.sync_all().await.map_err(|source| Error::Io {
                path: temporary_path.clone(),
                source,
            })?;
            drop(file);
            tokio::fs::rename(&temporary_path, &final_path)
                .await
                .map_err(|source| Error::Io {
                    path: final_path.clone(),
                    source,
                })?;
            Ok::<(), Error>(())
        }
        .await;
        if publication.is_err() {
            let _ = tokio::fs::remove_file(&temporary_path).await;
        }
        publication?;
        let cache_dir = self.cache_dir.clone();
        let retained = final_path.clone();
        let limit = self.cache_size_limit_bytes;
        tokio::task::spawn_blocking(move || {
            enforce_cache_size_limit_in(&cache_dir, Some(&retained), limit)
        })
        .await
        .map_err(|error| Error::CacheLock(error.to_string()))??;

        let snapshot = HistoricalCatalogSnapshot {
            cache_path: final_path.clone(),
            query_time,
            downloaded_at,
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
        let source = self.request_url(snapshot.query_time);
        let catalog = SatelliteCatalog::from_tle_text(&payload, source)?
            .with_retrieved_at(snapshot.downloaded_at);
        Ok(SatCheckerLoad {
            catalog,
            state: CacheState::Cached,
            cache_path: snapshot.cache_path.clone(),
            snapshot,
        })
    }

    fn request_url(&self, query_time: UtcTimestamp) -> String {
        let separator = if self.endpoint.contains('?') {
            '&'
        } else {
            '?'
        };
        format!(
            "{}{separator}epoch={:.8}&format=txt",
            self.endpoint,
            julian_date(query_time)
        )
    }
}

fn within_distance(
    snapshot: &HistoricalCatalogSnapshot,
    time_utc: UtcTimestamp,
    maximum_distance: Option<Duration>,
) -> bool {
    maximum_distance.is_none_or(|maximum| {
        snapshot.query_time.seconds_since(time_utc).abs() <= maximum.as_secs_f64()
    })
}

fn sort_by_distance(snapshots: &mut [HistoricalCatalogSnapshot], time_utc: UtcTimestamp) {
    snapshots.sort_by(|left, right| {
        left.query_time
            .seconds_since(time_utc)
            .abs()
            .total_cmp(&right.query_time.seconds_since(time_utc).abs())
            .then_with(|| {
                right
                    .query_time
                    .unix_seconds()
                    .total_cmp(&left.query_time.unix_seconds())
            })
            .then_with(|| {
                right
                    .downloaded_at
                    .unix_seconds()
                    .total_cmp(&left.downloaded_at.unix_seconds())
            })
    });
}

fn history_inventory_in(cache_dir: &Path) -> Result<Vec<HistoricalCatalogSnapshot>> {
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
        snapshots.push(HistoricalCatalogSnapshot {
            cache_path: path,
            query_time: UtcTimestamp::from_unix_seconds(query_millis as f64 / 1_000.0)?,
            downloaded_at: UtcTimestamp::from_unix_seconds(downloaded_seconds as f64)?,
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
