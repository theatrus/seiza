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
const CACHE_PREFIX: &str = "celestrak-active-";
const CACHE_SUFFIX: &str = ".json";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CacheState {
    Fresh,
    Downloaded,
    StaleFallback,
}

#[derive(Clone, Copy, Debug)]
enum LockMode {
    Shared,
    Exclusive,
}

/// Snapshot mtimes slightly in the future are tolerated as clock jitter;
/// anything further ahead is treated as maximally stale so a bad clock can
/// never pin an old snapshot as fresh forever.
const FUTURE_MTIME_TOLERANCE: Duration = Duration::from_secs(5 * 60);

#[derive(Debug)]
pub struct CelesTrakLoad {
    pub catalog: SatelliteCatalog,
    pub state: CacheState,
    pub cache_path: PathBuf,
    pub warning: Option<String>,
}

/// Cached asynchronous access to CelesTrak's current active-satellite OMM set.
#[derive(Clone, Debug)]
pub struct CelesTrakSource {
    client: reqwest::Client,
    cache_dir: PathBuf,
    endpoint: String,
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

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub async fn load_active(&self) -> Result<CelesTrakLoad> {
        tokio::fs::create_dir_all(&self.cache_dir)
            .await
            .map_err(|source| Error::Io {
                path: self.cache_dir.clone(),
                source,
            })?;

        // Concurrent readers of a fresh snapshot only need a shared lock;
        // refreshes take the exclusive lock below.
        {
            let _shared = self.acquire_lock(LockMode::Shared).await?;
            if let Some(snapshot) = self.freshest_cache().await?
                && snapshot.age <= CELESTRAK_MIN_REFRESH
                && let Ok(load) = self.load_cache(&snapshot, CacheState::Fresh, None).await
            {
                return Ok(load);
            }
        }

        let _exclusive = self.acquire_lock(LockMode::Exclusive).await?;
        // Another process may have refreshed while this one waited.
        let cached = self.freshest_cache().await?;
        if let Some(snapshot) = cached.as_ref()
            && snapshot.age <= CELESTRAK_MIN_REFRESH
            && let Ok(load) = self.load_cache(snapshot, CacheState::Fresh, None).await
        {
            return Ok(load);
        }

        match self.download().await {
            Ok(downloaded) => Ok(downloaded),
            Err(refresh_error) => {
                if let Some(snapshot) = cached.as_ref()
                    && let Ok(load) = self
                        .load_cache(
                            snapshot,
                            CacheState::StaleFallback,
                            Some(format!("CelesTrak refresh failed: {refresh_error}")),
                        )
                        .await
                {
                    return Ok(load);
                }
                Err(refresh_error)
            }
        }
    }

    async fn acquire_lock(&self, mode: LockMode) -> Result<File> {
        let path = self.cache_dir.join(".celestrak-active.lock");
        let error_path = path.clone();
        tokio::task::spawn_blocking(move || {
            let file = std::fs::OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .truncate(false)
                .open(&path)?;
            match mode {
                LockMode::Shared => file.lock_shared()?,
                LockMode::Exclusive => file.lock_exclusive()?,
            }
            Ok::<_, std::io::Error>(file)
        })
        .await
        .map_err(|error| Error::CacheLock(error.to_string()))?
        .map_err(|source| Error::Io {
            path: error_path,
            source,
        })
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
        tokio::fs::rename(&temporary_path, &final_path)
            .await
            .map_err(|source| Error::Io {
                path: final_path.clone(),
                source,
            })?;
        self.prune_except(&final_path).await;
        Ok(CelesTrakLoad {
            catalog,
            state: CacheState::Downloaded,
            cache_path: final_path,
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
        let retrieved_at = UtcTimestamp::from_unix_seconds(
            snapshot
                .modified
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
        )?;
        let catalog = SatelliteCatalog::from_omm_json(&payload, self.endpoint.clone())?
            .with_retrieved_at(retrieved_at);
        Ok(CelesTrakLoad {
            catalog,
            state,
            cache_path: snapshot.path.clone(),
            warning,
        })
    }

    async fn freshest_cache(&self) -> Result<Option<CacheSnapshot>> {
        let mut entries = tokio::fs::read_dir(&self.cache_dir)
            .await
            .map_err(|source| Error::Io {
                path: self.cache_dir.clone(),
                source,
            })?;
        let mut newest: Option<CacheSnapshot> = None;
        while let Some(entry) = entries.next_entry().await.map_err(|source| Error::Io {
            path: self.cache_dir.clone(),
            source,
        })? {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if !name.starts_with(CACHE_PREFIX) || !name.ends_with(CACHE_SUFFIX) {
                continue;
            }
            let metadata = match entry.metadata().await {
                Ok(metadata) if metadata.is_file() => metadata,
                _ => continue,
            };
            let modified = metadata.modified().unwrap_or(UNIX_EPOCH);
            if newest
                .as_ref()
                .is_none_or(|snapshot| modified > snapshot.modified)
            {
                let age = match SystemTime::now().duration_since(modified) {
                    Ok(age) => age,
                    Err(future) if future.duration() <= FUTURE_MTIME_TOLERANCE => Duration::ZERO,
                    Err(_) => Duration::MAX,
                };
                newest = Some(CacheSnapshot {
                    path: entry.path(),
                    modified,
                    age,
                });
            }
        }
        Ok(newest)
    }

    async fn prune_except(&self, retained: &Path) {
        let Ok(mut entries) = tokio::fs::read_dir(&self.cache_dir).await else {
            return;
        };
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if path != retained && name.starts_with(CACHE_PREFIX) && name.ends_with(CACHE_SUFFIX) {
                let _ = tokio::fs::remove_file(path).await;
            }
        }
    }
}

#[derive(Debug)]
struct CacheSnapshot {
    path: PathBuf,
    modified: SystemTime,
    age: Duration,
}

fn now_timestamp() -> Result<UtcTimestamp> {
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
        let path = dir.path().join("celestrak-active-1.json");
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
        let path = dir.path().join("celestrak-active-1.json");
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
    async fn future_mtime_is_stale_rather_than_fresh_forever() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("celestrak-active-1.json");
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
}
