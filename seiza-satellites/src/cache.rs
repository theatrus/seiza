//! Shared durable-cache machinery for the satellite element providers.
//!
//! CelesTrak (current), SatChecker (historical), and the Seiza mirror all keep
//! timestamped snapshots under one platform cache directory, guarded by one
//! shared advisory lock and bounded by one size ceiling. This module owns that
//! common substrate — locking, size eviction, atomic publication, timestamps —
//! while each provider keeps its own URL contract, filename scheme, and parser.

use crate::{Error, Result, UtcTimestamp};
use directories::ProjectDirs;
use fs2::FileExt;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::AsyncWriteExt;

// Filename schemes owned by each provider. The shared inventory and pruning
// recognize all of them so one ceiling governs the whole directory regardless
// of which source wrote a given snapshot.
pub(crate) const CELESTRAK_CACHE_PREFIX: &str = "celestrak-active-";
pub(crate) const CELESTRAK_CACHE_SUFFIX: &str = ".json";
pub(crate) const SATCHECKER_CACHE_PREFIX: &str = "satchecker-epoch-";
pub(crate) const SATCHECKER_CACHE_SUFFIX: &str = ".tle";
pub(crate) const SEIZA_MIRROR_CACHE_PREFIX: &str = "seiza-mirror-epoch-";
pub(crate) const SEIZA_MIRROR_CACHE_SUFFIX: &str = ".tle";

/// Default upper bound shared by all durable orbital-element snapshots (5 GiB).
pub const DEFAULT_SATELLITE_CACHE_SIZE_LIMIT_BYTES: u64 = 5 * 1024 * 1024 * 1024;
/// Backward-compatible name for the shared orbital-element cache ceiling.
pub const DEFAULT_CELESTRAK_CACHE_SIZE_LIMIT_BYTES: u64 = DEFAULT_SATELLITE_CACHE_SIZE_LIMIT_BYTES;

/// Upper bound on any single satellite provider response body (256 MiB).
pub(crate) const MAX_SATELLITE_RESPONSE_BYTES: u64 = 256 * 1024 * 1024;

/// Snapshot mtimes slightly in the future are tolerated as clock jitter;
/// anything further ahead is treated as maximally stale so a bad clock can
/// never pin an old snapshot as fresh forever.
const FUTURE_MTIME_TOLERANCE: Duration = Duration::from_secs(5 * 60);

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

/// Resolve the platform durable cache directory for satellite snapshots. Every
/// provider shares this directory so the ceiling and lock are truly shared.
pub(crate) fn platform_cache_dir() -> Result<PathBuf> {
    ProjectDirs::from("fyi", "Seiza", "seiza")
        .map(|dirs| dirs.cache_dir().join("satellites"))
        .ok_or(Error::NoCacheDirectory)
}

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

/// Acquire the shared advisory lock off the async runtime.
pub(crate) async fn acquire_lock(cache_dir: &Path, mode: LockMode) -> Result<File> {
    let cache_dir = cache_dir.to_path_buf();
    tokio::task::spawn_blocking(move || acquire_lock_in(&cache_dir, mode))
        .await
        .map_err(|error| Error::CacheLock(error.to_string()))?
}

/// Take the exclusive lock and enforce the ceiling. The newest snapshot is
/// always retained even when it alone exceeds the limit.
pub(crate) fn prune_cache_blocking(cache_dir: &Path, limit: u64) -> Result<()> {
    let _exclusive = acquire_lock_in(cache_dir, LockMode::Exclusive)?;
    enforce_cache_size_limit_in(cache_dir, None, limit)
}

/// [`prune_cache_blocking`] off the async runtime.
pub(crate) async fn prune_cache_async(cache_dir: &Path, limit: u64) -> Result<()> {
    let cache_dir = cache_dir.to_path_buf();
    tokio::task::spawn_blocking(move || prune_cache_blocking(&cache_dir, limit))
        .await
        .map_err(|error| Error::CacheLock(error.to_string()))?
}

/// Enforce the ceiling while retaining a just-written snapshot, off the async
/// runtime. Used immediately after a download so the fresh artifact is never
/// the one evicted.
pub(crate) async fn enforce_size_limit_async(
    cache_dir: &Path,
    retained: Option<&Path>,
    limit: u64,
) -> Result<()> {
    let cache_dir = cache_dir.to_path_buf();
    let retained = retained.map(Path::to_path_buf);
    tokio::task::spawn_blocking(move || {
        enforce_cache_size_limit_in(&cache_dir, retained.as_deref(), limit)
    })
    .await
    .map_err(|error| Error::CacheLock(error.to_string()))?
}

/// Write `bytes` to `temporary_path`, fsync, and atomically rename onto
/// `final_path`. The partial file is removed if any step fails, so a crashed
/// or interrupted download never leaves a half-written snapshot in place.
pub(crate) async fn publish_atomically(
    temporary_path: &Path,
    final_path: &Path,
    bytes: &[u8],
) -> Result<()> {
    let publication = async {
        let mut file = tokio::fs::File::create(temporary_path)
            .await
            .map_err(|source| Error::Io {
                path: temporary_path.to_path_buf(),
                source,
            })?;
        file.write_all(bytes).await.map_err(|source| Error::Io {
            path: temporary_path.to_path_buf(),
            source,
        })?;
        file.sync_all().await.map_err(|source| Error::Io {
            path: temporary_path.to_path_buf(),
            source,
        })?;
        drop(file);
        tokio::fs::rename(temporary_path, final_path)
            .await
            .map_err(|source| Error::Io {
                path: final_path.to_path_buf(),
                source,
            })?;
        Ok::<(), Error>(())
    }
    .await;
    if publication.is_err() {
        let _ = tokio::fs::remove_file(temporary_path).await;
    }
    publication
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

/// Total on-disk size of every managed snapshot in the directory. Current and
/// historical sources compare this against the one shared ceiling.
pub(crate) fn managed_cache_total_bytes(cache_dir: &Path) -> Result<u128> {
    Ok(managed_cache_inventory_in(cache_dir)?
        .iter()
        .map(|snapshot| u128::from(snapshot.size_bytes))
        .sum())
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
        let retained_at = if name.starts_with(CELESTRAK_CACHE_PREFIX)
            && name.ends_with(CELESTRAK_CACHE_SUFFIX)
        {
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
    let partial_prefix = format!(".{CELESTRAK_CACHE_PREFIX}");
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

/// The retrieval time of a CelesTrak snapshot: its filename timestamp, or the
/// file's mtime when the name carries no parseable timestamp.
pub(crate) fn snapshot_retrieved_at(path: &Path, modified: SystemTime) -> Result<UtcTimestamp> {
    let filename_timestamp = path
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_prefix(CELESTRAK_CACHE_PREFIX))
        .and_then(|name| name.strip_suffix(CELESTRAK_CACHE_SUFFIX))
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

pub(crate) fn snapshot_age_at(timestamp: UtcTimestamp, now: SystemTime) -> Duration {
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

/// A cached snapshot addressable by the epoch it answers for and when it was
/// fetched. Nearest-time selection orders candidates by these two.
pub(crate) trait TimeSnapshot {
    fn query_time(&self) -> UtcTimestamp;
    fn downloaded_at(&self) -> UtcTimestamp;
}

/// True when the snapshot's query epoch is within `maximum_distance` of `time`.
/// `None` imposes no bound — used by cache-only lookups that let the caller
/// judge suitability from the returned query epoch.
pub(crate) fn within_distance<S: TimeSnapshot>(
    snapshot: &S,
    time: UtcTimestamp,
    maximum_distance: Option<Duration>,
) -> bool {
    maximum_distance.is_none_or(|maximum| {
        snapshot.query_time().seconds_since(time).abs() <= maximum.as_secs_f64()
    })
}

/// Order snapshots nearest-first: by absolute distance from `time`, then the
/// newer query epoch, then the more recent download as a final tiebreak.
pub(crate) fn sort_by_distance<S: TimeSnapshot>(snapshots: &mut [S], time: UtcTimestamp) {
    snapshots.sort_by(|left, right| {
        left.query_time()
            .seconds_since(time)
            .abs()
            .total_cmp(&right.query_time().seconds_since(time).abs())
            .then_with(|| {
                right
                    .query_time()
                    .unix_seconds()
                    .total_cmp(&left.query_time().unix_seconds())
            })
            .then_with(|| {
                right
                    .downloaded_at()
                    .unix_seconds()
                    .total_cmp(&left.downloaded_at().unix_seconds())
            })
    });
}

/// Try cached snapshots nearest-first, returning the first that loads. This is
/// the cache-only path: if every in-window candidate fails to load, the last
/// error is surfaced; an empty set yields `Ok(None)`.
pub(crate) fn select_nearest_blocking<S, T>(
    mut snapshots: Vec<S>,
    time: UtcTimestamp,
    maximum_distance: Option<Duration>,
    mut load: impl FnMut(S) -> Result<T>,
) -> Result<Option<T>>
where
    S: TimeSnapshot,
{
    sort_by_distance(&mut snapshots, time);
    let mut last_error = None;
    for snapshot in snapshots {
        if !within_distance(&snapshot, time, maximum_distance) {
            continue;
        }
        match load(snapshot) {
            Ok(loaded) => return Ok(Some(loaded)),
            Err(error) => last_error = Some(error),
        }
    }
    match last_error {
        Some(error) => Err(error),
        None => Ok(None),
    }
}

/// Async counterpart used by the download path: a load failure is treated as a
/// cache miss and skipped, so resolution falls through to a network fetch. An
/// empty or entirely-unloadable set yields `Ok(None)`.
pub(crate) async fn select_nearest_async<S, T>(
    mut snapshots: Vec<S>,
    time: UtcTimestamp,
    maximum_distance: Option<Duration>,
    mut load: impl AsyncFnMut(S) -> Result<T>,
) -> Result<Option<T>>
where
    S: TimeSnapshot,
{
    sort_by_distance(&mut snapshots, time);
    for snapshot in snapshots {
        if !within_distance(&snapshot, time, maximum_distance) {
            continue;
        }
        if let Ok(loaded) = load(snapshot).await {
            return Ok(Some(loaded));
        }
    }
    Ok(None)
}

pub(crate) fn now_timestamp() -> Result<UtcTimestamp> {
    UtcTimestamp::from_unix_seconds(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64(),
    )
}
