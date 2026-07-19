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
pub enum CacheState {
    Fresh,
    Downloaded,
    StaleFallback,
}

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
        let _lock = self.acquire_lock().await?;
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

    async fn acquire_lock(&self) -> Result<File> {
        let path = self.cache_dir.join(".celestrak-active.lock");
        let error_path = path.clone();
        tokio::task::spawn_blocking(move || {
            let file = std::fs::OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .truncate(false)
                .open(&path)?;
            file.lock_exclusive()?;
            Ok::<_, std::io::Error>(file)
        })
        .await
        .map_err(|error| Error::Propagation(format!("CelesTrak cache lock task failed: {error}")))?
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
                newest = Some(CacheSnapshot {
                    path: entry.path(),
                    modified,
                    age: SystemTime::now()
                        .duration_since(modified)
                        .unwrap_or_default(),
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
}
