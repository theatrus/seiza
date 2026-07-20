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

/// How a mirror artifact is stored at rest. `identity` is the raw TLE text;
/// `zstd` stores it zstd-compressed. The mirror compresses its own copy (the
/// upstream IAU endpoint does not compress, and the mirror is the fast path
/// consulted first), so the object at `key` is a `.zst` and the client
/// decompresses it back to the canonical TLE on download.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MirrorEncoding {
    #[default]
    Identity,
    Zstd,
}

impl MirrorEncoding {
    fn is_identity(&self) -> bool {
        matches!(self, MirrorEncoding::Identity)
    }
}

/// One immutable historical orbital catalog offered by the Seiza mirror.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SatelliteMirrorEntry {
    pub query_time_utc: String,
    pub source_url: String,
    pub key: String,
    /// SHA-256 and byte count of the DECOMPRESSED TLE — the artifact's logical
    /// identity, verified after any decode.
    pub sha256: String,
    pub size_bytes: u64,
    /// Encoding of the object stored at `key`. Absent in JSON = `identity`.
    #[serde(default, skip_serializing_if = "MirrorEncoding::is_identity")]
    pub encoding: MirrorEncoding,
    /// SHA-256 of the stored (encoded) bytes, verified on download before
    /// decoding. Present iff `encoding` is not `identity`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoded_sha256: Option<String>,
    /// Byte count of the stored (encoded) object. Present iff `encoding` is not
    /// `identity`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoded_size_bytes: Option<u64>,
}

impl SatelliteMirrorEntry {
    pub fn query_time(&self) -> Result<UtcTimestamp> {
        UtcTimestamp::parse(&self.query_time_utc).map_err(|error| Error::MirrorManifest {
            source_name: self.key.clone(),
            message: format!("invalid query_time_utc: {error}"),
        })
    }

    pub fn expected_key(&self) -> String {
        match self.encoding {
            MirrorEncoding::Identity => format!("artifacts/{}/satellites.tle", self.sha256),
            MirrorEncoding::Zstd => format!("artifacts/{}/satellites.tle.zst", self.sha256),
        }
    }

    /// SHA-256 of the bytes actually stored at `key` (encoded form).
    fn stored_sha256(&self) -> &str {
        self.encoded_sha256.as_deref().unwrap_or(&self.sha256)
    }

    /// Byte count of the bytes actually stored at `key`.
    fn stored_size_bytes(&self) -> u64 {
        self.encoded_size_bytes.unwrap_or(self.size_bytes)
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
            match entry.encoding {
                MirrorEncoding::Identity => {
                    if entry.encoded_sha256.is_some() || entry.encoded_size_bytes.is_some() {
                        return Err(Error::MirrorManifest {
                            source_name: source_name.into(),
                            message: format!(
                                "{} is identity-encoded but declares encoded_* fields",
                                entry.query_time_utc
                            ),
                        });
                    }
                }
                MirrorEncoding::Zstd => {
                    let sha =
                        entry
                            .encoded_sha256
                            .as_deref()
                            .ok_or_else(|| Error::MirrorManifest {
                                source_name: source_name.into(),
                                message: format!(
                                    "{} is zstd-encoded but has no encoded_sha256",
                                    entry.query_time_utc
                                ),
                            })?;
                    if sha.len() != 64
                        || !sha.bytes().all(|byte| byte.is_ascii_hexdigit())
                        || sha.bytes().any(|byte| byte.is_ascii_uppercase())
                    {
                        return Err(Error::MirrorManifest {
                            source_name: source_name.into(),
                            message: format!(
                                "{} has an invalid lowercase encoded SHA-256",
                                entry.query_time_utc
                            ),
                        });
                    }
                    match entry.encoded_size_bytes {
                        Some(size) if size > 0 && size <= MAX_MIRROR_ARTIFACT_BYTES => {}
                        _ => {
                            return Err(Error::MirrorManifest {
                                source_name: source_name.into(),
                                message: format!(
                                    "{} has an invalid encoded_size_bytes (maximum {})",
                                    entry.query_time_utc, MAX_MIRROR_ARTIFACT_BYTES
                                ),
                            });
                        }
                    }
                }
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
        let manifest_bytes =
            crate::source::read_body_capped(response, MAX_MIRROR_MANIFEST_BYTES, &manifest_url)
                .await?;
        let manifest = SatelliteMirrorManifest::parse(&manifest_bytes, &manifest_url)?;
        // A healthy manifest that simply does not cover this epoch is a normal,
        // expected outcome — a distinct signal from a broken/unreachable mirror,
        // so the resolver can fall through to SatChecker silently.
        let entry = manifest
            .nearest(time_utc, self.cache_reuse_window)?
            .ok_or_else(|| Error::MirrorNoCoverage {
                time: time_utc.to_rfc3339(),
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
        // The bytes on the wire are the STORED (possibly zstd-compressed) form.
        let stored_size = entry.stored_size_bytes();
        if response
            .content_length()
            .is_some_and(|size| size != stored_size)
        {
            return Err(Error::MirrorManifest {
                source_name: artifact_url,
                message: format!(
                    "manifest stored size is {stored_size}, response Content-Length is {}",
                    response.content_length().unwrap_or_default()
                ),
            });
        }
        let stored = crate::source::read_body_capped(response, stored_size, &artifact_url).await?;
        if stored.len() as u64 != stored_size {
            return Err(Error::MirrorManifest {
                source_name: artifact_url,
                message: format!(
                    "manifest stored size is {stored_size}, response size is {}",
                    stored.len()
                ),
            });
        }
        let stored_sha256 = payload_sha256(&stored);
        if stored_sha256 != entry.stored_sha256() {
            return Err(Error::Integrity {
                source_name: artifact_url.clone(),
                expected: entry.stored_sha256().to_string(),
                actual: stored_sha256,
            });
        }
        // Decode to the canonical uncompressed TLE, then verify its identity.
        let payload = decode_mirror_artifact(entry, stored, &artifact_url)?;
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

/// Decode a stored mirror artifact into the canonical uncompressed TLE. For
/// `zstd`, the declared uncompressed `size_bytes` caps the output, so a
/// decompression bomb or a wrong frame fails instead of exhausting memory.
fn decode_mirror_artifact(
    entry: &SatelliteMirrorEntry,
    stored: Vec<u8>,
    source_name: &str,
) -> Result<Vec<u8>> {
    match entry.encoding {
        MirrorEncoding::Identity => Ok(stored),
        MirrorEncoding::Zstd => {
            let capacity = usize::try_from(entry.size_bytes).map_err(|_| Error::Elements {
                source_name: source_name.to_string(),
                message: "declared uncompressed size exceeds the address space".into(),
            })?;
            let decoded =
                zstd::bulk::decompress(&stored, capacity).map_err(|error| Error::Elements {
                    source_name: source_name.to_string(),
                    message: format!("cannot zstd-decompress artifact: {error}"),
                })?;
            if decoded.len() as u64 != entry.size_bytes {
                return Err(Error::MirrorManifest {
                    source_name: source_name.to_string(),
                    message: format!(
                        "decoded size is {}, manifest declares {}",
                        decoded.len(),
                        entry.size_bytes
                    ),
                });
            }
            Ok(decoded)
        }
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
                encoding: MirrorEncoding::Identity,
                encoded_sha256: None,
                encoded_size_bytes: None,
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
                encoding: MirrorEncoding::Identity,
                encoded_sha256: None,
                encoded_size_bytes: None,
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
                encoding: MirrorEncoding::Identity,
                encoded_sha256: None,
                encoded_size_bytes: None,
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

    /// Serve a zstd-encoded mirror: manifest (json) then the compressed artifact.
    fn serve_zstd_mirror(query_time: &str) -> String {
        let uncompressed = TLE.as_bytes();
        let sha256 = payload_sha256(uncompressed);
        let compressed = zstd::bulk::compress(uncompressed, 19).unwrap();
        let encoded_sha256 = payload_sha256(&compressed);
        let manifest = serde_json::to_vec(&SatelliteMirrorManifest {
            schema_version: SatelliteMirrorManifest::SCHEMA_VERSION,
            generated_at_utc: "2026-07-19T12:00:00Z".into(),
            snapshots: vec![SatelliteMirrorEntry {
                query_time_utc: query_time.into(),
                source_url: "https://satchecker.cps.iau.org/tools/tles-at-epoch/".into(),
                key: format!("artifacts/{sha256}/satellites.tle.zst"),
                sha256,
                size_bytes: uncompressed.len() as u64,
                encoding: MirrorEncoding::Zstd,
                encoded_sha256: Some(encoded_sha256),
                encoded_size_bytes: Some(compressed.len() as u64),
            }],
        })
        .unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for (request_number, stream) in listener.incoming().take(2).enumerate() {
                let mut stream = stream.unwrap();
                let mut request = [0_u8; 4096];
                let _ = stream.read(&mut request);
                let (content_type, body): (&str, &[u8]) = if request_number == 0 {
                    ("application/json", &manifest)
                } else {
                    ("application/zstd", &compressed)
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
        format!("http://{address}")
    }

    #[tokio::test]
    async fn downloads_and_decompresses_zstd_mirror_artifact() {
        let dir = tempfile::tempdir().unwrap();
        let query_time = "2025-10-18T12:52:12Z";
        let base_url = serve_zstd_mirror(query_time);
        let source = SeizaMirrorSource::new(dir.path())
            .unwrap()
            .with_base_url(base_url);
        let time = UtcTimestamp::parse(query_time).unwrap();

        let load = source.load_at(time).await.unwrap();
        assert_eq!(load.state, CacheState::Downloaded);
        assert_eq!(load.catalog.len(), 1);
        // The on-disk cache holds the DECOMPRESSED TLE, keyed + verified by the
        // uncompressed sha — the "download-decompress on disk" contract.
        assert_eq!(load.snapshot.sha256, payload_sha256(TLE.as_bytes()));
        let cached = std::fs::read(&load.cache_path).unwrap();
        assert_eq!(cached, TLE.as_bytes());

        // The reuse path reads the uncompressed cache without a second fetch.
        let second = source.load_at(time).await.unwrap();
        assert_eq!(second.state, CacheState::Cached);
    }

    #[test]
    fn zstd_entry_validates_and_rejects_inconsistent_encoded_fields() {
        let sha256 = "a".repeat(64);
        let mut manifest = SatelliteMirrorManifest {
            schema_version: SatelliteMirrorManifest::SCHEMA_VERSION,
            generated_at_utc: "2026-07-19T12:00:00Z".into(),
            snapshots: vec![SatelliteMirrorEntry {
                query_time_utc: "2025-10-18T12:52:12Z".into(),
                source_url: "upstream".into(),
                key: format!("artifacts/{sha256}/satellites.tle.zst"),
                sha256: sha256.clone(),
                size_bytes: 4_274_736,
                encoding: MirrorEncoding::Zstd,
                encoded_sha256: Some("b".repeat(64)),
                encoded_size_bytes: Some(1_000_000),
            }],
        };
        manifest.validate("test").unwrap();

        // A zstd entry missing its encoded digest is rejected.
        manifest.snapshots[0].encoded_sha256 = None;
        assert!(manifest.validate("test").is_err());

        // An identity entry that still carries encoded_* fields is rejected.
        manifest.snapshots[0].encoding = MirrorEncoding::Identity;
        manifest.snapshots[0].key = format!("artifacts/{sha256}/satellites.tle");
        manifest.snapshots[0].encoded_sha256 = Some("b".repeat(64));
        assert!(manifest.validate("test").is_err());
    }
}
