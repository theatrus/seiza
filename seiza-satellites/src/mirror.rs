use crate::source::{
    DEFAULT_SATELLITE_CACHE_SIZE_LIMIT_BYTES, LockMode, SEIZA_MIRROR_CACHE_PREFIX,
    SEIZA_MIRROR_CACHE_SUFFIX, acquire_lock_in, enforce_cache_size_limit_in, now_timestamp,
};
use crate::{CacheState, Error, Result, SatelliteCatalog, UtcTimestamp, payload_sha256};
use directories::ProjectDirs;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::io::AsyncWriteExt;

pub const DEFAULT_SEIZA_SATELLITE_MIRROR_URL: &str = "https://downloads.seiza.fyi/satellites/v1";

const MAX_MIRROR_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const MAX_MIRROR_ARTIFACT_BYTES: u64 = 256 * 1024 * 1024;

/// One immutable historical orbital catalog offered by the Seiza mirror.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SatelliteMirrorEntry {
    pub query_time_utc: String,
    pub source_url: String,
    pub key: String,
    pub sha256: String,
    pub size_bytes: u64,
}

impl SatelliteMirrorEntry {
    pub fn query_time(&self) -> Result<UtcTimestamp> {
        UtcTimestamp::parse(&self.query_time_utc).map_err(|error| Error::MirrorManifest {
            source_name: self.key.clone(),
            message: format!("invalid query_time_utc: {error}"),
        })
    }

    pub fn expected_key(&self) -> String {
        format!("artifacts/{}/satellites.tle", self.sha256)
    }
}

/// Rolling pointer published after all immutable satellite artifacts.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SatelliteMirrorManifest {
    pub schema_version: u32,
    pub generated_at_utc: String,
    pub snapshots: Vec<SatelliteMirrorEntry>,
}

impl SatelliteMirrorManifest {
    pub const SCHEMA_VERSION: u32 = 1;

    pub fn parse(payload: &[u8], source_name: impl Into<String>) -> Result<Self> {
        let source_name = source_name.into();
        let manifest: Self =
            serde_json::from_slice(payload).map_err(|error| Error::MirrorManifest {
                source_name: source_name.clone(),
                message: error.to_string(),
            })?;
        manifest.validate(&source_name)?;
        Ok(manifest)
    }

    pub fn validate(&self, source_name: &str) -> Result<()> {
        if self.schema_version != Self::SCHEMA_VERSION {
            return Err(Error::MirrorManifest {
                source_name: source_name.into(),
                message: format!(
                    "unsupported schema_version {}; expected {}",
                    self.schema_version,
                    Self::SCHEMA_VERSION
                ),
            });
        }
        UtcTimestamp::parse(&self.generated_at_utc).map_err(|error| Error::MirrorManifest {
            source_name: source_name.into(),
            message: format!("invalid generated_at_utc: {error}"),
        })?;
        let mut previous = None;
        for entry in &self.snapshots {
            let query_time = entry.query_time()?;
            if previous.is_some_and(|previous: f64| previous >= query_time.unix_seconds()) {
                return Err(Error::MirrorManifest {
                    source_name: source_name.into(),
                    message: "snapshots must be strictly ordered by unique query time".into(),
                });
            }
            previous = Some(query_time.unix_seconds());
            if entry.source_url.trim().is_empty() {
                return Err(Error::MirrorManifest {
                    source_name: source_name.into(),
                    message: format!("{} has an empty source_url", entry.query_time_utc),
                });
            }
            if entry.sha256.len() != 64
                || !entry.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
                || entry.sha256.bytes().any(|byte| byte.is_ascii_uppercase())
            {
                return Err(Error::MirrorManifest {
                    source_name: source_name.into(),
                    message: format!("{} has an invalid lowercase SHA-256", entry.query_time_utc),
                });
            }
            if entry.key != entry.expected_key() {
                return Err(Error::MirrorManifest {
                    source_name: source_name.into(),
                    message: format!(
                        "{} must use content-addressed key {}",
                        entry.query_time_utc,
                        entry.expected_key()
                    ),
                });
            }
            if entry.size_bytes == 0 || entry.size_bytes > MAX_MIRROR_ARTIFACT_BYTES {
                return Err(Error::MirrorManifest {
                    source_name: source_name.into(),
                    message: format!(
                        "{} has invalid size {} (maximum {})",
                        entry.query_time_utc, entry.size_bytes, MAX_MIRROR_ARTIFACT_BYTES
                    ),
                });
            }
        }
        Ok(())
    }

    pub fn nearest(
        &self,
        time_utc: UtcTimestamp,
        maximum_distance: Duration,
    ) -> Result<Option<&SatelliteMirrorEntry>> {
        let mut nearest = None;
        for entry in &self.snapshots {
            let query_time = entry.query_time()?;
            let distance = query_time.seconds_since(time_utc).abs();
            if distance <= maximum_distance.as_secs_f64()
                && nearest.is_none_or(|(_, nearest_distance)| distance <= nearest_distance)
            {
                nearest = Some((entry, distance));
            }
        }
        Ok(nearest.map(|(entry, _)| entry))
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MirrorCatalogSnapshot {
    pub cache_path: PathBuf,
    pub query_time: UtcTimestamp,
    pub downloaded_at: UtcTimestamp,
    pub source_url: String,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Debug)]
pub struct SeizaMirrorLoad {
    pub catalog: SatelliteCatalog,
    pub state: CacheState,
    pub cache_path: PathBuf,
    pub snapshot: MirrorCatalogSnapshot,
}

#[derive(Clone, Debug)]
pub struct SeizaMirrorSource {
    client: reqwest::Client,
    cache_dir: PathBuf,
    base_url: String,
    cache_size_limit_bytes: u64,
    cache_reuse_window: Duration,
}

impl SeizaMirrorSource {
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
            base_url: DEFAULT_SEIZA_SATELLITE_MIRROR_URL.into(),
            cache_size_limit_bytes: DEFAULT_SATELLITE_CACHE_SIZE_LIMIT_BYTES,
            cache_reuse_window: crate::SATCHECKER_CACHE_REUSE_WINDOW,
        })
    }

    pub fn platform_default() -> Result<Self> {
        let cache_dir = ProjectDirs::from("fyi", "Seiza", "seiza")
            .map(|dirs| dirs.cache_dir().join("satellites"))
            .ok_or(Error::NoCacheDirectory)?;
        Self::new(cache_dir)
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    pub fn with_cache_size_limit_bytes(mut self, limit: u64) -> Self {
        self.cache_size_limit_bytes = limit;
        self
    }

    pub fn with_cache_reuse_window(mut self, window: Duration) -> Self {
        self.cache_reuse_window = window;
        self
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn cached_snapshots(&self) -> Result<Vec<MirrorCatalogSnapshot>> {
        self.prune_cache()?;
        let _shared = acquire_lock_in(&self.cache_dir, LockMode::Shared)?;
        mirror_inventory_in(&self.cache_dir, &self.base_url)
    }

    pub fn load_cached_for(&self, time_utc: UtcTimestamp) -> Result<Option<SeizaMirrorLoad>> {
        self.prune_cache()?;
        let _shared = acquire_lock_in(&self.cache_dir, LockMode::Shared)?;
        self.load_nearest_blocking(time_utc, None)
    }

    pub async fn load_at(&self, time_utc: UtcTimestamp) -> Result<SeizaMirrorLoad> {
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
        self.download_nearest(time_utc).await
    }

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

    async fn download_nearest(&self, time_utc: UtcTimestamp) -> Result<SeizaMirrorLoad> {
        let manifest_url = format!("{}/manifest.json", self.base_url);
        let response = self
            .client
            .get(&manifest_url)
            .send()
            .await
            .map_err(|source| Error::Http {
                url: manifest_url.clone(),
                source,
            })?;
        if !response.status().is_success() {
            return Err(Error::HttpStatus {
                url: manifest_url,
                status: response.status().as_u16(),
            });
        }
        if response
            .content_length()
            .is_some_and(|size| size > MAX_MIRROR_MANIFEST_BYTES)
        {
            return Err(Error::MirrorManifest {
                source_name: manifest_url,
                message: format!(
                    "manifest exceeds the {} byte limit",
                    MAX_MIRROR_MANIFEST_BYTES
                ),
            });
        }
        let manifest_bytes = response.bytes().await.map_err(|source| Error::Http {
            url: manifest_url.clone(),
            source,
        })?;
        if manifest_bytes.len() as u64 > MAX_MIRROR_MANIFEST_BYTES {
            return Err(Error::MirrorManifest {
                source_name: manifest_url,
                message: format!(
                    "manifest exceeds the {} byte limit",
                    MAX_MIRROR_MANIFEST_BYTES
                ),
            });
        }
        let manifest = SatelliteMirrorManifest::parse(&manifest_bytes, &manifest_url)?;
        let entry = manifest
            .nearest(time_utc, self.cache_reuse_window)?
            .ok_or_else(|| Error::MirrorManifest {
                source_name: manifest_url.clone(),
                message: format!(
                    "no snapshot is within {} seconds of {}",
                    self.cache_reuse_window.as_secs(),
                    time_utc.to_rfc3339()
                ),
            })?;
        let query_time = entry.query_time()?;
        let artifact_url = format!("{}/{}", self.base_url, entry.key);
        let response = self
            .client
            .get(&artifact_url)
            .send()
            .await
            .map_err(|source| Error::Http {
                url: artifact_url.clone(),
                source,
            })?;
        if !response.status().is_success() {
            return Err(Error::HttpStatus {
                url: artifact_url,
                status: response.status().as_u16(),
            });
        }
        if response
            .content_length()
            .is_some_and(|size| size != entry.size_bytes)
        {
            return Err(Error::MirrorManifest {
                source_name: artifact_url,
                message: format!(
                    "manifest size is {}, response Content-Length is {}",
                    entry.size_bytes,
                    response.content_length().unwrap_or_default()
                ),
            });
        }
        let payload = response.bytes().await.map_err(|source| Error::Http {
            url: artifact_url.clone(),
            source,
        })?;
        if payload.len() as u64 != entry.size_bytes {
            return Err(Error::MirrorManifest {
                source_name: artifact_url,
                message: format!(
                    "manifest size is {}, response size is {}",
                    entry.size_bytes,
                    payload.len()
                ),
            });
        }
        let actual_sha256 = payload_sha256(&payload);
        if actual_sha256 != entry.sha256 {
            return Err(Error::Integrity {
                source_name: artifact_url,
                expected: entry.sha256.clone(),
                actual: actual_sha256,
            });
        }
        let payload_text = std::str::from_utf8(&payload).map_err(|error| Error::Elements {
            source_name: artifact_url.clone(),
            message: error.to_string(),
        })?;
        let downloaded_at = now_timestamp()?;
        let catalog = SatelliteCatalog::from_tle_text(payload_text, artifact_url.clone())?
            .with_retrieved_at(downloaded_at);
        let final_path =
            mirror_cache_path(&self.cache_dir, query_time, downloaded_at, &entry.sha256);
        let temporary_path = self.cache_dir.join(format!(
            ".{SEIZA_MIRROR_CACHE_PREFIX}{}-{}.partial",
            (query_time.unix_seconds() * 1_000.0).round() as i64,
            std::process::id()
        ));
        let publication = async {
            let mut file = tokio::fs::File::create(&temporary_path)
                .await
                .map_err(|source| Error::Io {
                    path: temporary_path.clone(),
                    source,
                })?;
            file.write_all(&payload).await.map_err(|source| Error::Io {
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
        let snapshot = MirrorCatalogSnapshot {
            cache_path: final_path.clone(),
            query_time,
            downloaded_at,
            source_url: artifact_url,
            sha256: entry.sha256.clone(),
            size_bytes: payload.len() as u64,
        };
        Ok(SeizaMirrorLoad {
            catalog,
            state: CacheState::Downloaded,
            cache_path: final_path,
            snapshot,
        })
    }

    async fn load_nearest_async(
        &self,
        time_utc: UtcTimestamp,
        maximum_distance: Option<Duration>,
    ) -> Result<Option<SeizaMirrorLoad>> {
        let cache_dir = self.cache_dir.clone();
        let base_url = self.base_url.clone();
        let mut snapshots =
            tokio::task::spawn_blocking(move || mirror_inventory_in(&cache_dir, &base_url))
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
    ) -> Result<Option<SeizaMirrorLoad>> {
        let mut snapshots = mirror_inventory_in(&self.cache_dir, &self.base_url)?;
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

    async fn load_snapshot_async(
        &self,
        snapshot: MirrorCatalogSnapshot,
    ) -> Result<SeizaMirrorLoad> {
        let payload = tokio::fs::read(&snapshot.cache_path)
            .await
            .map_err(|source| Error::Io {
                path: snapshot.cache_path.clone(),
                source,
            })?;
        self.parsed_snapshot(payload, snapshot)
    }

    fn load_snapshot_blocking(&self, snapshot: MirrorCatalogSnapshot) -> Result<SeizaMirrorLoad> {
        let payload = std::fs::read(&snapshot.cache_path).map_err(|source| Error::Io {
            path: snapshot.cache_path.clone(),
            source,
        })?;
        self.parsed_snapshot(payload, snapshot)
    }

    fn parsed_snapshot(
        &self,
        payload: Vec<u8>,
        snapshot: MirrorCatalogSnapshot,
    ) -> Result<SeizaMirrorLoad> {
        let actual = payload_sha256(&payload);
        if actual != snapshot.sha256 {
            return Err(Error::Integrity {
                source_name: snapshot.cache_path.display().to_string(),
                expected: snapshot.sha256,
                actual,
            });
        }
        let payload = std::str::from_utf8(&payload).map_err(|error| Error::Elements {
            source_name: snapshot.source_url.clone(),
            message: error.to_string(),
        })?;
        let catalog = SatelliteCatalog::from_tle_text(payload, snapshot.source_url.clone())?
            .with_retrieved_at(snapshot.downloaded_at);
        Ok(SeizaMirrorLoad {
            catalog,
            state: CacheState::Cached,
            cache_path: snapshot.cache_path.clone(),
            snapshot,
        })
    }
}

fn mirror_cache_path(
    cache_dir: &Path,
    query_time: UtcTimestamp,
    downloaded_at: UtcTimestamp,
    sha256: &str,
) -> PathBuf {
    cache_dir.join(format!(
        "{SEIZA_MIRROR_CACHE_PREFIX}{}-cached-{}-{sha256}{SEIZA_MIRROR_CACHE_SUFFIX}",
        (query_time.unix_seconds() * 1_000.0).round() as i64,
        downloaded_at.unix_seconds().floor() as i64,
    ))
}

fn mirror_inventory_in(cache_dir: &Path, base_url: &str) -> Result<Vec<MirrorCatalogSnapshot>> {
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
        let Some((query_millis, downloaded_seconds, sha256)) = name
            .strip_prefix(SEIZA_MIRROR_CACHE_PREFIX)
            .and_then(|value| value.strip_suffix(SEIZA_MIRROR_CACHE_SUFFIX))
            .and_then(|value| {
                let (query, remainder) = value.split_once("-cached-")?;
                let (downloaded, sha256) = remainder.split_once('-')?;
                Some((
                    query.parse::<i64>().ok()?,
                    downloaded.parse::<i64>().ok()?,
                    sha256,
                ))
            })
        else {
            continue;
        };
        if sha256.len() != 64 || !sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            continue;
        }
        let sha256 = sha256.to_string();
        let metadata = entry.metadata().map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        if !metadata.is_file() {
            continue;
        }
        snapshots.push(MirrorCatalogSnapshot {
            cache_path: path,
            query_time: UtcTimestamp::from_unix_seconds(query_millis as f64 / 1_000.0)?,
            downloaded_at: UtcTimestamp::from_unix_seconds(downloaded_seconds as f64)?,
            source_url: format!("{base_url}/artifacts/{sha256}/satellites.tle"),
            sha256,
            size_bytes: metadata.len(),
        });
    }
    snapshots.sort_by(|left, right| {
        left.query_time
            .unix_seconds()
            .total_cmp(&right.query_time.unix_seconds())
    });
    Ok(snapshots)
}

fn within_distance(
    snapshot: &MirrorCatalogSnapshot,
    time_utc: UtcTimestamp,
    maximum_distance: Option<Duration>,
) -> bool {
    maximum_distance.is_none_or(|maximum| {
        snapshot.query_time.seconds_since(time_utc).abs() <= maximum.as_secs_f64()
    })
}

fn sort_by_distance(snapshots: &mut [MirrorCatalogSnapshot], time_utc: UtcTimestamp) {
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
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::sync::mpsc;

    const TLE: &str = include_str!("../tests/data/iss-2024.tle");

    fn serve_mirror(query_time: &str, artifact: &'static [u8]) -> (String, mpsc::Receiver<String>) {
        let sha256 = payload_sha256(TLE.as_bytes());
        let manifest = serde_json::to_vec(&SatelliteMirrorManifest {
            schema_version: SatelliteMirrorManifest::SCHEMA_VERSION,
            generated_at_utc: "2026-07-19T12:00:00Z".into(),
            snapshots: vec![SatelliteMirrorEntry {
                query_time_utc: query_time.into(),
                source_url: "https://satchecker.cps.iau.org/tools/tles-at-epoch/".into(),
                key: format!("artifacts/{sha256}/satellites.tle"),
                sha256,
                size_bytes: artifact.len() as u64,
            }],
        })
        .unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (sender, receiver) = mpsc::channel();
        std::thread::spawn(move || {
            for (request_number, stream) in listener.incoming().take(2).enumerate() {
                let mut stream = stream.unwrap();
                let mut request = [0_u8; 4096];
                let count = stream.read(&mut request).unwrap();
                let _ = sender.send(String::from_utf8_lossy(&request[..count]).into_owned());
                let (content_type, body): (&str, &[u8]) = if request_number == 0 {
                    ("application/json", &manifest)
                } else {
                    ("text/plain", artifact)
                };
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    body.len()
                )
                .unwrap();
                stream.write_all(body).unwrap();
            }
        });
        (format!("http://{address}"), receiver)
    }

    #[test]
    fn manifest_requires_sorted_content_addressed_entries() {
        let sha256 = "a".repeat(64);
        let manifest = SatelliteMirrorManifest {
            schema_version: SatelliteMirrorManifest::SCHEMA_VERSION,
            generated_at_utc: "2026-07-19T12:00:00Z".into(),
            snapshots: vec![SatelliteMirrorEntry {
                query_time_utc: "2025-10-18T12:52:12Z".into(),
                source_url: "https://satchecker.cps.iau.org/tools/tles-at-epoch/".into(),
                key: format!("artifacts/{sha256}/satellites.tle"),
                sha256,
                size_bytes: 4_274_736,
            }],
        };
        manifest.validate("test").unwrap();
        let encoded = serde_json::to_vec(&manifest).unwrap();
        assert_eq!(
            SatelliteMirrorManifest::parse(&encoded, "test").unwrap(),
            manifest
        );
    }

    #[test]
    fn manifest_rejects_non_content_addressed_keys() {
        let sha256 = "b".repeat(64);
        let manifest = SatelliteMirrorManifest {
            schema_version: 1,
            generated_at_utc: "2026-07-19T12:00:00Z".into(),
            snapshots: vec![SatelliteMirrorEntry {
                query_time_utc: "2025-10-18T12:52:12Z".into(),
                source_url: "upstream".into(),
                key: "latest.tle".into(),
                sha256,
                size_bytes: 10,
            }],
        };
        assert!(manifest.validate("test").is_err());
    }

    #[tokio::test]
    async fn downloads_verified_mirror_artifact_and_then_reuses_cache() {
        let dir = tempfile::tempdir().unwrap();
        let query_time = "2025-10-18T12:52:12Z";
        let (base_url, requests) = serve_mirror(query_time, TLE.as_bytes());
        let source = SeizaMirrorSource::new(dir.path())
            .unwrap()
            .with_base_url(base_url);
        let time = UtcTimestamp::parse(query_time).unwrap();

        let first = source.load_at(time).await.unwrap();
        assert_eq!(first.state, CacheState::Downloaded);
        assert_eq!(first.catalog.len(), 1);
        assert_eq!(first.snapshot.sha256, payload_sha256(TLE.as_bytes()));
        assert!(
            requests
                .recv_timeout(Duration::from_secs(2))
                .unwrap()
                .starts_with("GET /manifest.json ")
        );
        assert!(
            requests
                .recv_timeout(Duration::from_secs(2))
                .unwrap()
                .starts_with("GET /artifacts/")
        );

        let second = source.load_at(time).await.unwrap();
        assert_eq!(second.state, CacheState::Cached);
        assert_eq!(second.cache_path, first.cache_path);
    }

    #[tokio::test]
    async fn rejects_mirror_artifact_that_does_not_match_manifest_hash() {
        let dir = tempfile::tempdir().unwrap();
        let query_time = "2025-10-18T12:52:12Z";
        let (base_url, _) = serve_mirror(query_time, b"not the advertised TLE");
        let source = SeizaMirrorSource::new(dir.path())
            .unwrap()
            .with_base_url(base_url);

        assert!(matches!(
            source
                .load_at(UtcTimestamp::parse(query_time).unwrap())
                .await
                .unwrap_err(),
            Error::Integrity { .. }
        ));
    }
}
