use crate::{Error, Result, SatelliteCatalog, UtcTimestamp};
use directories::ProjectDirs;
use fs2::FileExt;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::AsyncWriteExt;

pub const CELESTRAK_ACTIVE_OMM_URL: &str =
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=ACTIVE&FORMAT=JSON";
pub const CELESTRAK_MIN_REFRESH: Duration = Duration::from_secs(2 * 60 * 60);
/// Default upper bound shared by all durable orbital-element snapshots (5 GiB).
pub const DEFAULT_SATELLITE_CACHE_SIZE_LIMIT_BYTES: u64 = 5 * 1024 * 1024 * 1024;
/// Backward-compatible name for the shared orbital-element cache ceiling.
pub const DEFAULT_CELESTRAK_CACHE_SIZE_LIMIT_BYTES: u64 = DEFAULT_SATELLITE_CACHE_SIZE_LIMIT_BYTES;
const CACHE_PREFIX: &str = "celestrak-active-";
const CACHE_SUFFIX: &str = ".json";
pub(crate) const SATCHECKER_CACHE_PREFIX: &str = "satchecker-epoch-";
pub(crate) const SATCHECKER_CACHE_SUFFIX: &str = ".tle";
pub(crate) const SEIZA_MIRROR_CACHE_PREFIX: &str = "seiza-mirror-epoch-";
pub(crate) const SEIZA_MIRROR_CACHE_SUFFIX: &str = ".tle";
/// Upper bound on any single satellite provider response body (256 MiB).
pub(crate) const MAX_SATELLITE_RESPONSE_BYTES: u64 = 256 * 1024 * 1024;

/// Read a response body, refusing to buffer more than `limit` bytes.
///
/// The `Content-Length` header is only advisory — it is absent on chunked and
/// HTTP/2 responses (both of which CloudFront can produce), so a check on it
/// alone can be bypassed. This enforces the cap while streaming, so a
/// compromised or misconfigured origin can never make the client allocate an
/// unbounded body before the size/hash checks run.
pub(crate) async fn read_body_capped(
    mut response: reqwest::Response,
    limit: u64,
    url: &str,
) -> Result<Vec<u8>> {
    if response.content_length().is_some_and(|len| len > limit) {
        return Err(Error::ResponseTooLarge {
            url: url.to_string(),
            limit,
        });
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|source| Error::Http {
        url: url.to_string(),
        source,
    })? {
        if body.len() as u64 + chunk.len() as u64 > limit {
            return Err(Error::ResponseTooLarge {
                url: url.to_string(),
                limit,
            });
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CacheState {
    Fresh,
    Downloaded,
    StaleFallback,
    /// Loaded without any network access through a cache-only API.
    Cached,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum LockMode {
    Shared,
    Exclusive,
}

/// Snapshot mtimes slightly in the future are tolerated as clock jitter;
/// anything further ahead is treated as maximally stale so a bad clock can
/// never pin an old snapshot as fresh forever.
const FUTURE_MTIME_TOLERANCE: Duration = Duration::from_secs(5 * 60);

/// Public inventory record for one durable orbital-element snapshot.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CachedCatalogSnapshot {
    pub cache_path: PathBuf,
    pub retrieved_at: UtcTimestamp,
    pub size_bytes: u64,
}

#[derive(Debug)]
pub struct CelesTrakLoad {
    pub catalog: SatelliteCatalog,
    pub state: CacheState,
    pub cache_path: PathBuf,
    pub snapshot: CachedCatalogSnapshot,
    pub warning: Option<String>,
}

/// Cached asynchronous access to CelesTrak's current active-satellite OMM set.
#[derive(Clone, Debug)]
pub struct CelesTrakSource {
    client: reqwest::Client,
    cache_dir: PathBuf,
    endpoint: String,
    cache_size_limit_bytes: u64,
}

impl CelesTrakSource {
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
            endpoint: CELESTRAK_ACTIVE_OMM_URL.into(),
            cache_size_limit_bytes: DEFAULT_CELESTRAK_CACHE_SIZE_LIMIT_BYTES,
        })
    }

    pub fn platform_default() -> Result<Self> {
        let cache_dir = ProjectDirs::from("fyi", "Seiza", "seiza")
            .map(|dirs| dirs.cache_dir().join("satellites"))
            .ok_or(Error::NoCacheDirectory)?;
        Self::new(cache_dir)
    }

    /// Override the endpoint for an organization-local caching proxy or test
    /// fixture. The two-hour local refresh floor remains enforced.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Override the durable snapshot-history ceiling. The most recent
    /// snapshot is always retained, even when it alone exceeds this limit.
    pub fn with_cache_size_limit_bytes(mut self, cache_size_limit_bytes: u64) -> Self {
        self.cache_size_limit_bytes = cache_size_limit_bytes;
        self
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn cache_size_limit_bytes(&self) -> u64 {
        self.cache_size_limit_bytes
    }

    /// List recognized snapshots from oldest to newest without network I/O.
    pub fn cached_snapshots(&self) -> Result<Vec<CachedCatalogSnapshot>> {
        self.prune_cache_if_needed()?;
        let _shared = self.acquire_lock_blocking(LockMode::Shared)?;
        Ok(self
            .cache_inventory()?
            .into_iter()
            .map(|snapshot| snapshot.public())
            .collect())
    }

    /// Load the newest valid cached snapshot without network I/O.
    pub fn load_cached(&self) -> Result<Option<CelesTrakLoad>> {
        self.load_cached_near(None)
    }

    /// Load the valid cached snapshot retrieved closest to `time_utc` without
    /// network I/O. This is intended for reproducible historical exposures.
    pub fn load_cached_for(&self, time_utc: UtcTimestamp) -> Result<Option<CelesTrakLoad>> {
        self.load_cached_near(Some(time_utc))
    }

    fn load_cached_near(&self, time_utc: Option<UtcTimestamp>) -> Result<Option<CelesTrakLoad>> {
        self.prune_cache_if_needed()?;
        let _shared = self.acquire_lock_blocking(LockMode::Shared)?;
        let mut snapshots = self.cache_inventory()?;
        if let Some(time_utc) = time_utc {
            snapshots.sort_by(|left, right| {
                (left.retrieved_at.unix_seconds() - time_utc.unix_seconds())
                    .abs()
                    .total_cmp(&(right.retrieved_at.unix_seconds() - time_utc.unix_seconds()).abs())
                    .then_with(|| {
                        right
                            .retrieved_at
                            .unix_seconds()
                            .total_cmp(&left.retrieved_at.unix_seconds())
                    })
            });
        } else {
            snapshots.reverse();
        }

        let mut last_error = None;
        for snapshot in snapshots {
            match self.load_cache_blocking(&snapshot, CacheState::Cached, None) {
                Ok(load) => return Ok(Some(load)),
                Err(error) => last_error = Some(error),
            }
        }
        match last_error {
            Some(error) => Err(error),
            None => Ok(None),
        }
    }

    pub async fn load_active(&self) -> Result<CelesTrakLoad> {
        tokio::fs::create_dir_all(&self.cache_dir)
            .await
            .map_err(|source| Error::Io {
                path: self.cache_dir.clone(),
                source,
            })?;

        // Enforce the configured bound even when the newest snapshot is fresh
        // or the network is unavailable. Download is not a maintenance gate.
        self.prune_cache_if_needed_async().await?;

        // Concurrent readers of a fresh snapshot only need a shared lock;
        // refreshes take the exclusive lock below.
        {
            let _shared = self.acquire_lock(LockMode::Shared).await?;
            if let Some(load) = self
                .load_newest_valid_cache(CacheState::Fresh, Some(CELESTRAK_MIN_REFRESH), None)
                .await?
            {
                return Ok(load);
            }
        }

        let _exclusive = self.acquire_lock(LockMode::Exclusive).await?;
        // Another process may have refreshed while this one waited.
        if let Some(load) = self
            .load_newest_valid_cache(CacheState::Fresh, Some(CELESTRAK_MIN_REFRESH), None)
            .await?
        {
            return Ok(load);
        }

        match self.download().await {
            Ok(downloaded) => Ok(downloaded),
            Err(refresh_error) => {
                if let Some(load) = self
                    .load_newest_valid_cache(
                        CacheState::StaleFallback,
                        None,
                        Some(format!("CelesTrak refresh failed: {refresh_error}")),
                    )
                    .await?
                {
                    return Ok(load);
                }
                Err(refresh_error)
            }
        }
    }

    async fn acquire_lock(&self, mode: LockMode) -> Result<File> {
        let cache_dir = self.cache_dir.clone();
        tokio::task::spawn_blocking(move || acquire_lock_in(&cache_dir, mode))
            .await
            .map_err(|error| Error::CacheLock(error.to_string()))?
    }

    fn acquire_lock_blocking(&self, mode: LockMode) -> Result<File> {
        acquire_lock_in(&self.cache_dir, mode)
    }

    async fn download(&self) -> Result<CelesTrakLoad> {
        let response = self
            .client
            .get(&self.endpoint)
            .send()
            .await
            .map_err(|source| Error::Http {
                url: self.endpoint.clone(),
                source,
            })?;
        let status = response.status();
        if !status.is_success() {
            // CelesTrak answers 403 when it has rate-limited or blocked a
            // client that re-downloads within its two-hour update window.
            if status.as_u16() == 403 || status.as_u16() == 429 {
                let retry_after_seconds = response
                    .headers()
                    .get(reqwest::header::RETRY_AFTER)
                    .and_then(|value| value.to_str().ok())
                    .and_then(|value| value.trim().parse::<u64>().ok());
                return Err(Error::RateLimited {
                    url: self.endpoint.clone(),
                    status: status.as_u16(),
                    retry_after_seconds,
                });
            }
            return Err(Error::HttpStatus {
                url: self.endpoint.clone(),
                status: status.as_u16(),
            });
        }
        let payload = response.text().await.map_err(|source| Error::Http {
            url: self.endpoint.clone(),
            source,
        })?;
        let retrieved_at = now_timestamp()?;
        let catalog = SatelliteCatalog::from_omm_json(&payload, self.endpoint.clone())?
            .with_retrieved_at(retrieved_at);

        let timestamp = retrieved_at.unix_seconds().floor() as i64;
        let final_path = self
            .cache_dir
            .join(format!("{CACHE_PREFIX}{timestamp}{CACHE_SUFFIX}"));
        let temporary_path = self.cache_dir.join(format!(
            ".{CACHE_PREFIX}{timestamp}-{}.partial",
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
        self.enforce_cache_size_limit(Some(&final_path)).await?;
        let snapshot = CacheSnapshot {
            path: final_path.clone(),
            retrieved_at,
            age: Duration::ZERO,
            size_bytes: payload.len() as u64,
        };
        Ok(CelesTrakLoad {
            catalog,
            state: CacheState::Downloaded,
            cache_path: final_path,
            snapshot: snapshot.public(),
            warning: None,
        })
    }

    async fn load_cache(
        &self,
        snapshot: &CacheSnapshot,
        state: CacheState,
        warning: Option<String>,
    ) -> Result<CelesTrakLoad> {
        let payload = tokio::fs::read_to_string(&snapshot.path)
            .await
            .map_err(|source| Error::Io {
                path: snapshot.path.clone(),
                source,
            })?;
        let catalog = SatelliteCatalog::from_omm_json(&payload, self.endpoint.clone())?
            .with_retrieved_at(snapshot.retrieved_at);
        Ok(CelesTrakLoad {
            catalog,
            state,
            cache_path: snapshot.path.clone(),
            snapshot: snapshot.public(),
            warning,
        })
    }

    fn load_cache_blocking(
        &self,
        snapshot: &CacheSnapshot,
        state: CacheState,
        warning: Option<String>,
    ) -> Result<CelesTrakLoad> {
        let payload = std::fs::read_to_string(&snapshot.path).map_err(|source| Error::Io {
            path: snapshot.path.clone(),
            source,
        })?;
        let catalog = SatelliteCatalog::from_omm_json(&payload, self.endpoint.clone())?
            .with_retrieved_at(snapshot.retrieved_at);
        Ok(CelesTrakLoad {
            catalog,
            state,
            cache_path: snapshot.path.clone(),
            snapshot: snapshot.public(),
            warning,
        })
    }

    async fn cache_inventory_async(&self) -> Result<Vec<CacheSnapshot>> {
        let cache_dir = self.cache_dir.clone();
        tokio::task::spawn_blocking(move || cache_inventory_in(&cache_dir))
            .await
            .map_err(|error| Error::CacheLock(error.to_string()))?
    }

    async fn load_newest_valid_cache(
        &self,
        state: CacheState,
        maximum_age: Option<Duration>,
        warning: Option<String>,
    ) -> Result<Option<CelesTrakLoad>> {
        let mut snapshots = self.cache_inventory_async().await?;
        snapshots.reverse();
        for snapshot in snapshots {
            if maximum_age.is_some_and(|maximum| snapshot.age > maximum) {
                continue;
            }
            if let Ok(load) = self.load_cache(&snapshot, state, warning.clone()).await {
                return Ok(Some(load));
            }
        }
        Ok(None)
    }

    /// Enforce the configured history ceiling without performing network I/O.
    /// The newest snapshot is always retained.
    pub fn prune_cache(&self) -> Result<()> {
        let _exclusive = self.acquire_lock_blocking(LockMode::Exclusive)?;
        enforce_cache_size_limit_in(&self.cache_dir, None, self.cache_size_limit_bytes)
    }

    fn prune_cache_if_needed(&self) -> Result<()> {
        let exceeds_limit = {
            let _shared = self.acquire_lock_blocking(LockMode::Shared)?;
            self.cache_exceeds_limit()?
        };
        if exceeds_limit {
            self.prune_cache()?;
        }
        Ok(())
    }

    async fn prune_cache_if_needed_async(&self) -> Result<()> {
        let exceeds_limit = {
            let _shared = self.acquire_lock(LockMode::Shared).await?;
            self.cache_exceeds_limit_async().await?
        };
        if !exceeds_limit {
            return Ok(());
        }
        let _exclusive = self.acquire_lock(LockMode::Exclusive).await?;
        self.enforce_cache_size_limit(None).await
    }

    fn cache_exceeds_limit(&self) -> Result<bool> {
        Ok(managed_cache_inventory_in(&self.cache_dir)?
            .iter()
            .map(|snapshot| u128::from(snapshot.size_bytes))
            .sum::<u128>()
            > u128::from(self.cache_size_limit_bytes))
    }

    async fn cache_exceeds_limit_async(&self) -> Result<bool> {
        let cache_dir = self.cache_dir.clone();
        Ok(
            tokio::task::spawn_blocking(move || managed_cache_inventory_in(&cache_dir))
                .await
                .map_err(|error| Error::CacheLock(error.to_string()))??
                .iter()
                .map(|snapshot| u128::from(snapshot.size_bytes))
                .sum::<u128>()
                > u128::from(self.cache_size_limit_bytes),
        )
    }

    async fn enforce_cache_size_limit(&self, retained: Option<&Path>) -> Result<()> {
        let cache_dir = self.cache_dir.clone();
        let retained = retained.map(Path::to_path_buf);
        let limit = self.cache_size_limit_bytes;
        tokio::task::spawn_blocking(move || {
            enforce_cache_size_limit_in(&cache_dir, retained.as_deref(), limit)
        })
        .await
        .map_err(|error| Error::CacheLock(error.to_string()))?
    }

    fn cache_inventory(&self) -> Result<Vec<CacheSnapshot>> {
        cache_inventory_in(&self.cache_dir)
    }
}

#[derive(Clone, Debug)]
struct CacheSnapshot {
    path: PathBuf,
    retrieved_at: UtcTimestamp,
    age: Duration,
    size_bytes: u64,
}

impl CacheSnapshot {
    fn public(&self) -> CachedCatalogSnapshot {
        CachedCatalogSnapshot {
            cache_path: self.path.clone(),
            retrieved_at: self.retrieved_at,
            size_bytes: self.size_bytes,
        }
    }
}

pub(crate) fn acquire_lock_in(cache_dir: &Path, mode: LockMode) -> Result<File> {
    std::fs::create_dir_all(cache_dir).map_err(|source| Error::Io {
        path: cache_dir.to_path_buf(),
        source,
    })?;
    let path = cache_dir.join(".celestrak-active.lock");
    let file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&path)
        .map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
    match mode {
        LockMode::Shared => file.lock_shared(),
        LockMode::Exclusive => file.lock_exclusive(),
    }
    .map_err(|error| Error::CacheLock(error.to_string()))?;
    Ok(file)
}

fn cache_inventory_in(cache_dir: &Path) -> Result<Vec<CacheSnapshot>> {
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
    let now = SystemTime::now();
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
        if !name.starts_with(CACHE_PREFIX) || !name.ends_with(CACHE_SUFFIX) {
            continue;
        }
        let metadata = entry.metadata().map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        if !metadata.is_file() {
            continue;
        }
        let modified = metadata.modified().unwrap_or(UNIX_EPOCH);
        let retrieved_at = snapshot_retrieved_at(&path, modified)?;
        let age = snapshot_age_at(retrieved_at, now);
        snapshots.push(CacheSnapshot {
            path,
            retrieved_at,
            age,
            size_bytes: metadata.len(),
        });
    }
    snapshots.sort_by(|left, right| {
        left.retrieved_at
            .unix_seconds()
            .total_cmp(&right.retrieved_at.unix_seconds())
            .then_with(|| left.path.cmp(&right.path))
    });
    Ok(snapshots)
}

pub(crate) fn enforce_cache_size_limit_in(
    cache_dir: &Path,
    retained: Option<&Path>,
    limit: u64,
) -> Result<()> {
    remove_abandoned_partial_files(cache_dir)?;
    let mut snapshots = managed_cache_inventory_in(cache_dir)?;
    let newest = snapshots.last().map(|snapshot| snapshot.path.clone());
    let mut total = snapshots
        .iter()
        .map(|snapshot| snapshot.size_bytes)
        .sum::<u64>();
    for snapshot in snapshots.drain(..) {
        if total <= limit {
            break;
        }
        if retained.is_some_and(|retained| snapshot.path == retained)
            || newest
                .as_deref()
                .is_some_and(|newest| snapshot.path == newest)
        {
            continue;
        }
        std::fs::remove_file(&snapshot.path).map_err(|source| Error::Io {
            path: snapshot.path.clone(),
            source,
        })?;
        total = total.saturating_sub(snapshot.size_bytes);
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct ManagedCacheSnapshot {
    path: PathBuf,
    retained_at: f64,
    size_bytes: u64,
}

/// Inventory every cache artifact owned by this crate so current and
/// historical element sources share one configured size ceiling.
fn managed_cache_inventory_in(cache_dir: &Path) -> Result<Vec<ManagedCacheSnapshot>> {
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
        let metadata = entry.metadata().map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        if !metadata.is_file() {
            continue;
        }
        let modified = metadata.modified().unwrap_or(UNIX_EPOCH);
        let modified_seconds = modified
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        let retained_at = if name.starts_with(CACHE_PREFIX) && name.ends_with(CACHE_SUFFIX) {
            snapshot_retrieved_at(&path, modified)?.unix_seconds()
        } else if let Some(value) = name
            .strip_prefix(SATCHECKER_CACHE_PREFIX)
            .and_then(|value| value.strip_suffix(SATCHECKER_CACHE_SUFFIX))
            .and_then(|value| value.rsplit_once("-cached-").map(|(_, cached)| cached))
            .and_then(|value| value.parse::<f64>().ok())
        {
            value
        } else if let Some(value) = name
            .strip_prefix(SEIZA_MIRROR_CACHE_PREFIX)
            .and_then(|value| value.strip_suffix(SEIZA_MIRROR_CACHE_SUFFIX))
            .and_then(|value| value.split("-cached-").nth(1))
            .and_then(|value| value.split('-').next())
            .and_then(|value| value.parse::<f64>().ok())
        {
            value
        } else {
            continue;
        };
        snapshots.push(ManagedCacheSnapshot {
            path,
            retained_at: if retained_at.is_finite() {
                retained_at
            } else {
                modified_seconds
            },
            size_bytes: metadata.len(),
        });
    }
    snapshots.sort_by(|left, right| {
        left.retained_at
            .total_cmp(&right.retained_at)
            .then_with(|| left.path.cmp(&right.path))
    });
    Ok(snapshots)
}

fn remove_abandoned_partial_files(cache_dir: &Path) -> Result<()> {
    let entries = std::fs::read_dir(cache_dir).map_err(|source| Error::Io {
        path: cache_dir.to_path_buf(),
        source,
    })?;
    let partial_prefix = format!(".{CACHE_PREFIX}");
    let satchecker_partial_prefix = format!(".{SATCHECKER_CACHE_PREFIX}");
    let mirror_partial_prefix = format!(".{SEIZA_MIRROR_CACHE_PREFIX}");
    for entry in entries {
        let entry = entry.map_err(|source| Error::Io {
            path: cache_dir.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if (name.starts_with(&partial_prefix)
            || name.starts_with(&satchecker_partial_prefix)
            || name.starts_with(&mirror_partial_prefix))
            && name.ends_with(".partial")
        {
            std::fs::remove_file(&path).map_err(|source| Error::Io {
                path: path.clone(),
                source,
            })?;
        }
    }
    Ok(())
}

fn snapshot_retrieved_at(path: &Path, modified: SystemTime) -> Result<UtcTimestamp> {
    let filename_timestamp = path
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_prefix(CACHE_PREFIX))
        .and_then(|name| name.strip_suffix(CACHE_SUFFIX))
        .and_then(|value| value.parse::<u64>().ok())
        .map(|seconds| seconds as f64);
    let seconds = filename_timestamp.unwrap_or_else(|| {
        modified
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
    });
    UtcTimestamp::from_unix_seconds(seconds)
}

fn snapshot_age_at(timestamp: UtcTimestamp, now: SystemTime) -> Duration {
    let Ok(elapsed) = Duration::try_from_secs_f64(timestamp.unix_seconds()) else {
        return Duration::MAX;
    };
    let Some(retrieved) = UNIX_EPOCH.checked_add(elapsed) else {
        return Duration::MAX;
    };
    match now.duration_since(retrieved) {
        Ok(age) => age,
        Err(future) if future.duration() <= FUTURE_MTIME_TOLERANCE => Duration::ZERO,
        Err(_) => Duration::MAX,
    }
}

pub(crate) fn now_timestamp() -> Result<UtcTimestamp> {
    UtcTimestamp::from_unix_seconds(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const OMM: &str = r#"[{
        "OBJECT_NAME":"ISS (ZARYA)",
        "OBJECT_ID":"1998-067A",
        "EPOCH":"2024-05-02T12:00:00.000000",
        "MEAN_MOTION":15.5,
        "ECCENTRICITY":0.0005,
        "INCLINATION":51.64,
        "RA_OF_ASC_NODE":160.0,
        "ARG_OF_PERICENTER":80.0,
        "MEAN_ANOMALY":280.0,
        "NORAD_CAT_ID":25544,
        "BSTAR":0.0001
    }]"#;

    #[tokio::test]
    async fn fresh_valid_cache_avoids_network() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("celestrak-active-manual.json");
        tokio::fs::write(&path, OMM).await.unwrap();
        let source = CelesTrakSource::new(dir.path())
            .unwrap()
            .with_endpoint("http://127.0.0.1:1/never-called");
        let load = source.load_active().await.unwrap();
        assert_eq!(load.state, CacheState::Fresh);
        assert_eq!(load.catalog.len(), 1);
        assert_eq!(load.cache_path, path);
    }

    /// Serve a canned HTTP status for every connection on a background thread.
    fn serve_status(status_line: &'static str, extra_headers: &'static str) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            for stream in listener.incoming().flatten() {
                let mut stream = stream;
                let mut request = [0u8; 4096];
                let _ = stream.read(&mut request);
                let _ = stream.write_all(
                    format!(
                        "HTTP/1.1 {status_line}\r\n{extra_headers}content-length: 0\r\nconnection: close\r\n\r\n"
                    )
                    .as_bytes(),
                );
            }
        });
        format!("http://{address}/gp")
    }

    fn set_mtime(path: &Path, time: SystemTime) {
        File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(time)
            .unwrap();
    }

    #[tokio::test]
    async fn rate_limited_download_reports_retry_after() {
        let dir = tempfile::tempdir().unwrap();
        let source = CelesTrakSource::new(dir.path())
            .unwrap()
            .with_endpoint(serve_status("403 Forbidden", "retry-after: 7200\r\n"));
        match source.load_active().await.unwrap_err() {
            Error::RateLimited {
                status,
                retry_after_seconds,
                ..
            } => {
                assert_eq!(status, 403);
                assert_eq!(retry_after_seconds, Some(7200));
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stale_snapshot_survives_rate_limited_refresh() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("celestrak-active-manual.json");
        tokio::fs::write(&path, OMM).await.unwrap();
        set_mtime(&path, SystemTime::now() - Duration::from_secs(3 * 60 * 60));
        let source = CelesTrakSource::new(dir.path())
            .unwrap()
            .with_endpoint(serve_status("403 Forbidden", ""));
        let load = source.load_active().await.unwrap();
        assert_eq!(load.state, CacheState::StaleFallback);
        assert!(load.warning.unwrap().contains("rate-limits"));
    }

    #[tokio::test]
    async fn refresh_fallback_skips_a_corrupt_newer_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let valid = dir.path().join("celestrak-active-100.json");
        let corrupt = dir.path().join("celestrak-active-200.json");
        tokio::fs::write(&valid, OMM).await.unwrap();
        tokio::fs::write(&corrupt, "not orbital data")
            .await
            .unwrap();
        let source = CelesTrakSource::new(dir.path())
            .unwrap()
            .with_endpoint(serve_status("403 Forbidden", ""));

        let load = source.load_active().await.unwrap();

        assert_eq!(load.state, CacheState::StaleFallback);
        assert_eq!(load.cache_path, valid);
    }

    #[tokio::test]
    async fn future_mtime_is_stale_rather_than_fresh_forever() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("celestrak-active-manual.json");
        tokio::fs::write(&path, OMM).await.unwrap();
        set_mtime(&path, SystemTime::now() + Duration::from_secs(24 * 60 * 60));
        let source = CelesTrakSource::new(dir.path())
            .unwrap()
            .with_endpoint(serve_status("403 Forbidden", ""));
        // A snapshot stamped far in the future must trigger a refresh attempt
        // (leaving it usable only as a stale fallback), not read as fresh.
        let load = source.load_active().await.unwrap();
        assert_eq!(load.state, CacheState::StaleFallback);
    }

    #[test]
    fn cache_only_load_selects_the_snapshot_nearest_the_exposure() {
        let dir = tempfile::tempdir().unwrap();
        for timestamp in [100_u64, 300] {
            std::fs::write(
                dir.path()
                    .join(format!("{CACHE_PREFIX}{timestamp}{CACHE_SUFFIX}")),
                OMM,
            )
            .unwrap();
        }
        let source = CelesTrakSource::new(dir.path())
            .unwrap()
            .with_endpoint("http://127.0.0.1:1/never-called");

        let inventory = source.cached_snapshots().unwrap();
        assert_eq!(inventory.len(), 2);
        assert_eq!(inventory[0].retrieved_at.unix_seconds(), 100.0);
        assert_eq!(inventory[1].retrieved_at.unix_seconds(), 300.0);

        let target = UtcTimestamp::from_unix_seconds(200.0).unwrap();
        let loaded = source.load_cached_for(target).unwrap().unwrap();
        assert_eq!(loaded.state, CacheState::Cached);
        assert_eq!(loaded.snapshot.retrieved_at.unix_seconds(), 300.0);
        assert_eq!(loaded.catalog.fingerprint().content_sha256.len(), 64);
    }

    #[test]
    fn cache_only_load_skips_a_corrupt_nearer_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("celestrak-active-100.json"), OMM).unwrap();
        std::fs::write(
            dir.path().join("celestrak-active-200.json"),
            "not orbital data",
        )
        .unwrap();
        let source = CelesTrakSource::new(dir.path()).unwrap();

        let loaded = source
            .load_cached_for(UtcTimestamp::from_unix_seconds(200.0).unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(loaded.snapshot.retrieved_at.unix_seconds(), 100.0);
    }

    #[test]
    fn retention_evicts_oldest_only_after_the_size_ceiling() {
        let dir = tempfile::tempdir().unwrap();
        let size = OMM.len() as u64;
        let paths = [100_u64, 200, 300].map(|timestamp| {
            let path = dir
                .path()
                .join(format!("{CACHE_PREFIX}{timestamp}{CACHE_SUFFIX}"));
            std::fs::write(&path, OMM).unwrap();
            path
        });
        let partial = dir.path().join(".celestrak-active-300-123.partial");
        std::fs::write(&partial, OMM).unwrap();

        enforce_cache_size_limit_in(dir.path(), Some(&paths[2]), size * 3).unwrap();
        assert!(paths.iter().all(|path| path.exists()));
        assert!(!partial.exists());

        enforce_cache_size_limit_in(dir.path(), Some(&paths[2]), size * 2).unwrap();

        assert!(!paths[0].exists());
        assert!(paths[1].exists());
        assert!(paths[2].exists());
    }

    #[test]
    fn default_history_ceiling_is_five_gibibytes() {
        let dir = tempfile::tempdir().unwrap();
        let source = CelesTrakSource::new(dir.path()).unwrap();
        assert_eq!(source.cache_size_limit_bytes(), 5 * 1024 * 1024 * 1024);
    }

    #[test]
    fn cache_only_load_enforces_a_reduced_history_ceiling() {
        let dir = tempfile::tempdir().unwrap();
        let size = OMM.len() as u64;
        let paths = [100_u64, 200, 300].map(|timestamp| {
            let path = dir
                .path()
                .join(format!("{CACHE_PREFIX}{timestamp}{CACHE_SUFFIX}"));
            std::fs::write(&path, OMM).unwrap();
            path
        });
        let source = CelesTrakSource::new(dir.path())
            .unwrap()
            .with_cache_size_limit_bytes(size * 2);

        let loaded = source.load_cached().unwrap().unwrap();

        assert_eq!(loaded.cache_path, paths[2]);
        assert!(!paths[0].exists());
        assert!(paths[1].exists());
        assert!(paths[2].exists());
    }

    #[tokio::test]
    async fn fresh_cache_load_enforces_the_history_ceiling_without_downloading() {
        let dir = tempfile::tempdir().unwrap();
        let size = OMM.len() as u64;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let paths = [now - 2, now - 1, now].map(|timestamp| {
            let path = dir
                .path()
                .join(format!("{CACHE_PREFIX}{timestamp}{CACHE_SUFFIX}"));
            std::fs::write(&path, OMM).unwrap();
            path
        });
        let source = CelesTrakSource::new(dir.path())
            .unwrap()
            .with_endpoint("http://127.0.0.1:1/never-called")
            .with_cache_size_limit_bytes(size * 2);

        let loaded = source.load_active().await.unwrap();

        assert_eq!(loaded.state, CacheState::Fresh);
        assert_eq!(loaded.cache_path, paths[2]);
        assert!(!paths[0].exists());
        assert!(paths[1].exists());
        assert!(paths[2].exists());
    }
}
