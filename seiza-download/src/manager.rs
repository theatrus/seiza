use crate::error::io;
use crate::{
    BundleManifestDocument, CatalogSet, DEFAULT_BUNDLE_BASE_URL, Dataset, Error,
    LEGACY_V2_BUNDLE_BASE_URL, ManifestFile, ManifestTransport, Result,
};
use async_compression::tokio::bufread::ZstdDecoder;
use directories::ProjectDirs;
use fs2::FileExt;
use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_util::io::StreamReader;

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const READ_TIMEOUT: Duration = Duration::from_secs(300);
const IO_BUFFER_BYTES: usize = 1024 * 1024;

/// Controls when the small hosted bundle manifest may use the cached copy.
/// Catalog artifacts themselves are immutable and content-addressed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CachePolicy {
    /// Never access the network.
    OfflineOnly,
    /// Use any valid cached manifest, fetching only when none exists.
    PreferCached,
    /// Refresh a cached manifest after the given age. A stale manifest remains
    /// usable when the refresh fails.
    RefreshIfOlderThan(Duration),
    /// Require a fresh manifest from the server.
    ForceRefresh,
}

impl Default for CachePolicy {
    fn default() -> Self {
        Self::RefreshIfOlderThan(Duration::from_secs(24 * 60 * 60))
    }
}

/// Observable cache and transfer activity. Libraries do not print; callers
/// can translate these events into logs, progress bars, or UI state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DownloadEvent {
    FetchingManifest {
        url: String,
    },
    UsingCachedManifest {
        version: String,
        stale: bool,
    },
    CacheHit {
        name: String,
        path: PathBuf,
    },
    DownloadStarted {
        name: String,
        bytes: u64,
    },
    DownloadProgress {
        name: String,
        /// Transferred bytes: encoded bytes for a compressed transport.
        downloaded: u64,
        /// Total transfer size in bytes.
        total: u64,
        /// Bytes written to the local file so far. Equal to `downloaded` for
        /// an uncompressed transfer; ahead of it while decompressing.
        written: u64,
    },
    DownloadComplete {
        name: String,
        path: PathBuf,
    },
}

/// One verified immutable artifact in the local cache.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CatalogArtifact {
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
    pub path: PathBuf,
}

/// Paths for a coherent manifest version. Pass these directly to `seiza`'s
/// memory-mapped catalog readers to avoid an extra copy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CatalogBundle {
    pub version: String,
    artifacts: BTreeMap<String, CatalogArtifact>,
}

impl CatalogBundle {
    pub fn path(&self, dataset: Dataset) -> Result<&Path> {
        self.path_by_name(dataset.file_name())
    }

    pub fn path_by_name(&self, name: &str) -> Result<&Path> {
        self.artifacts
            .get(name)
            .map(|artifact| artifact.path.as_path())
            .ok_or_else(|| Error::MissingArtifact(name.into()))
    }

    pub fn artifacts(&self) -> impl ExactSizeIterator<Item = &CatalogArtifact> {
        self.artifacts.values()
    }

    /// Exhaustively re-hash the selected cached artifacts. Normal `ensure`
    /// calls intentionally use only manifest size metadata on cache hits.
    pub async fn verify(&self) -> Result<()> {
        for artifact in self.artifacts.values() {
            verify_artifact(&artifact.path, artifact).await?;
        }
        Ok(())
    }

    /// Copy the selected artifacts into the CLI-compatible flat directory.
    /// Applications should normally consume the cache paths directly.
    pub async fn materialize(&self, output: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
        let output = output.as_ref();
        tokio::fs::create_dir_all(output)
            .await
            .map_err(|source| io("create directory", output, source))?;
        let mut paths = Vec::with_capacity(self.artifacts.len());

        for artifact in self.artifacts.values() {
            let target = output.join(&artifact.name);
            if verify_artifact(&target, artifact).await.is_ok() {
                paths.push(target);
                continue;
            }

            let temp = output.join(format!(
                ".{}.part-{}-{}",
                artifact.name,
                std::process::id(),
                next_sequence()
            ));
            let install = async {
                tokio::fs::copy(&artifact.path, &temp)
                    .await
                    .map_err(|source| io("copy cached artifact to", &temp, source))?;
                verify_artifact(&temp, artifact).await?;
                replace_file(&temp, &target).await
            }
            .await;
            if let Err(error) = install {
                let _ = tokio::fs::remove_file(&temp).await;
                return Err(error);
            }
            paths.push(target);
        }
        Ok(paths)
    }
}

/// Configures a catalog manager without creating an async runtime.
#[derive(Clone, Debug)]
pub struct CatalogManagerBuilder {
    cache_dir: Option<PathBuf>,
    base_url: String,
    policy: CachePolicy,
}

impl Default for CatalogManagerBuilder {
    fn default() -> Self {
        Self {
            cache_dir: None,
            base_url: DEFAULT_BUNDLE_BASE_URL.into(),
            policy: CachePolicy::default(),
        }
    }
}

impl CatalogManagerBuilder {
    pub fn cache_dir(mut self, cache_dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(cache_dir.into());
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    pub fn policy(mut self, policy: CachePolicy) -> Self {
        self.policy = policy;
        self
    }

    pub fn build(self) -> Result<CatalogManager> {
        let cache_dir = self
            .cache_dir
            .or_else(|| std::env::var_os("SEIZA_CACHE_DIR").map(PathBuf::from))
            .or_else(|| {
                ProjectDirs::from("fyi", "Seiza", "seiza")
                    .map(|dirs| dirs.cache_dir().join("catalogs"))
            })
            .ok_or(Error::CacheDirectoryUnavailable)?;
        let client = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .read_timeout(READ_TIMEOUT)
            .user_agent(format!("seiza-download/{}", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|source| Error::Http {
                url: self.base_url.clone(),
                source,
            })?;
        Ok(CatalogManager {
            cache_dir,
            base_url: self.base_url,
            policy: self.policy,
            client,
        })
    }
}

/// Async manager for hosted manifests and immutable catalog artifacts.
#[derive(Clone, Debug)]
pub struct CatalogManager {
    cache_dir: PathBuf,
    base_url: String,
    policy: CachePolicy,
    client: reqwest::Client,
}

impl CatalogManager {
    pub fn builder() -> CatalogManagerBuilder {
        CatalogManagerBuilder::default()
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub async fn ensure(&self, set: &CatalogSet) -> Result<CatalogBundle> {
        self.ensure_with(set, |_| {}).await
    }

    pub async fn ensure_with<F>(&self, set: &CatalogSet, report: F) -> Result<CatalogBundle>
    where
        F: Fn(DownloadEvent) + Send + Sync,
    {
        let manifest = self.load_manifest(&report).await?;
        let plan = manifest.manifest.plan(set)?;
        let mut artifacts = BTreeMap::new();

        for file in plan {
            let transport = manifest.preferred_transport(&file.name);
            let path = self.ensure_file(&file, transport, &report).await?;
            artifacts.insert(
                file.name.clone(),
                CatalogArtifact {
                    name: file.name,
                    bytes: file.bytes,
                    sha256: file.sha256,
                    path,
                },
            );
        }

        Ok(CatalogBundle {
            version: manifest.manifest.version,
            artifacts,
        })
    }

    async fn load_manifest<F>(&self, report: &F) -> Result<BundleManifestDocument>
    where
        F: Fn(DownloadEvent) + Send + Sync,
    {
        let cached = self.manifest_path();
        match self.policy {
            CachePolicy::OfflineOnly => {
                let manifest = self
                    .read_cached_manifest()
                    .await?
                    .ok_or_else(|| Error::CachedManifestUnavailable(cached))?;
                report(DownloadEvent::UsingCachedManifest {
                    version: manifest.manifest.version.clone(),
                    stale: false,
                });
                Ok(manifest)
            }
            CachePolicy::PreferCached => {
                if let Ok(Some(manifest)) = self.read_cached_manifest().await {
                    report(DownloadEvent::UsingCachedManifest {
                        version: manifest.manifest.version.clone(),
                        stale: false,
                    });
                    Ok(manifest)
                } else {
                    self.fetch_manifest(report).await
                }
            }
            CachePolicy::ForceRefresh => self.fetch_manifest(report).await,
            CachePolicy::RefreshIfOlderThan(max_age) => {
                if self.cached_manifest_is_fresh(max_age).await?
                    && let Ok(Some(manifest)) = self.read_cached_manifest().await
                {
                    report(DownloadEvent::UsingCachedManifest {
                        version: manifest.manifest.version.clone(),
                        stale: false,
                    });
                    return Ok(manifest);
                }
                match self.fetch_manifest(report).await {
                    Ok(manifest) => Ok(manifest),
                    Err(fetch_error) => match self.read_cached_manifest().await {
                        Ok(Some(manifest)) => {
                            report(DownloadEvent::UsingCachedManifest {
                                version: manifest.manifest.version.clone(),
                                stale: true,
                            });
                            Ok(manifest)
                        }
                        _ => Err(fetch_error),
                    },
                }
            }
        }
    }

    async fn cached_manifest_is_fresh(&self, max_age: Duration) -> Result<bool> {
        let path = self.manifest_path();
        let metadata = match tokio::fs::metadata(&path).await {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(source) => return Err(io("read metadata for", path, source)),
        };
        let modified = metadata
            .modified()
            .map_err(|source| io("read modification time for", &path, source))?;
        Ok(SystemTime::now()
            .duration_since(modified)
            .map_or(true, |age| age <= max_age))
    }

    async fn read_cached_manifest(&self) -> Result<Option<BundleManifestDocument>> {
        let path = self.manifest_path();
        let bytes = match tokio::fs::read(&path).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(source) => return Err(io("read", path, source)),
        };
        BundleManifestDocument::parse(&bytes).map(Some)
    }

    async fn fetch_manifest<F>(&self, report: &F) -> Result<BundleManifestDocument>
    where
        F: Fn(DownloadEvent) + Send + Sync,
    {
        let url = format!("{}/manifest.json", self.base_url);
        report(DownloadEvent::FetchingManifest { url: url.clone() });
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|source| Error::Http {
                url: url.clone(),
                source,
            })?;
        if !response.status().is_success() {
            return Err(Error::HttpStatus {
                url,
                status: response.status().as_u16(),
            });
        }
        let bytes = response.bytes().await.map_err(|source| Error::Http {
            url: url.clone(),
            source,
        })?;
        let manifest = BundleManifestDocument::parse(&bytes)?;
        self.write_cached_manifest(&bytes).await?;
        Ok(manifest)
    }

    async fn write_cached_manifest(&self, bytes: &[u8]) -> Result<()> {
        let path = self.manifest_path();
        let parent = path.parent().expect("manifest path has a parent");
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|source| io("create directory", parent, source))?;
        let temp = parent.join(format!(
            ".{}.part-{}-{}",
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("catalog-manifest.json"),
            std::process::id(),
            next_sequence()
        ));
        let write = async {
            let mut output = tokio::fs::File::create(&temp)
                .await
                .map_err(|source| io("create", &temp, source))?;
            output
                .write_all(bytes)
                .await
                .map_err(|source| io("write", &temp, source))?;
            output
                .sync_all()
                .await
                .map_err(|source| io("sync", &temp, source))?;
            drop(output);
            replace_file(&temp, &path).await
        }
        .await;
        if write.is_err() {
            let _ = tokio::fs::remove_file(&temp).await;
        }
        write
    }

    async fn ensure_file<F>(
        &self,
        file: &ManifestFile,
        transport: Option<&ManifestTransport>,
        report: &F,
    ) -> Result<PathBuf>
    where
        F: Fn(DownloadEvent) + Send + Sync,
    {
        let target = self.object_path(file);
        if metadata_matches(&target, file.bytes).await? {
            report(DownloadEvent::CacheHit {
                name: file.name.clone(),
                path: target.clone(),
            });
            return Ok(target);
        }

        // Lock only cache misses, then re-check after waiting. This avoids
        // duplicate multi-gigabyte transfers across application processes.
        let _lock = self.acquire_artifact_lock(file).await?;
        if metadata_matches(&target, file.bytes).await? {
            report(DownloadEvent::CacheHit {
                name: file.name.clone(),
                path: target.clone(),
            });
            return Ok(target);
        }

        let partial_dir = self.cache_dir.join("partial");
        tokio::fs::create_dir_all(&partial_dir)
            .await
            .map_err(|source| io("create directory", &partial_dir, source))?;
        let temp = partial_dir.join(format!(
            "{}-{}-{}-{}.part",
            file.sha256,
            std::process::id(),
            next_sequence(),
            file.name
        ));
        let install = async {
            self.download_file(file, transport, &temp, report).await?;

            let parent = target.parent().expect("object path has a parent");
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|source| io("create directory", parent, source))?;
            if metadata_matches(&target, file.bytes).await? {
                tokio::fs::remove_file(&temp)
                    .await
                    .map_err(|source| io("remove", &temp, source))?;
            } else {
                replace_file(&temp, &target).await?;
            }
            Ok(())
        }
        .await;
        if let Err(error) = install {
            let _ = tokio::fs::remove_file(&temp).await;
            return Err(error);
        }
        report(DownloadEvent::DownloadComplete {
            name: file.name.clone(),
            path: target.clone(),
        });
        Ok(target)
    }

    async fn download_file<F>(
        &self,
        file: &ManifestFile,
        transport: Option<&ManifestTransport>,
        temp: &Path,
        report: &F,
    ) -> Result<()>
    where
        F: Fn(DownloadEvent) + Send + Sync,
    {
        if let Some(transport) = transport {
            self.download_zstd(file, transport, temp, report).await
        } else {
            self.download_identity(file, temp, report).await
        }
    }

    async fn download_identity<F>(&self, file: &ManifestFile, temp: &Path, report: &F) -> Result<()>
    where
        F: Fn(DownloadEvent) + Send + Sync,
    {
        let url = format!("{}/{}", self.base_url, file.artifact_key());
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|source| Error::Http {
                url: url.clone(),
                source,
            })?;
        if !response.status().is_success() {
            return Err(Error::HttpStatus {
                url,
                status: response.status().as_u16(),
            });
        }
        if let Some(bytes) = response.content_length()
            && bytes != file.bytes
        {
            return Err(Error::Size {
                name: file.name.clone(),
                expected: file.bytes,
                actual: bytes,
            });
        }

        report(DownloadEvent::DownloadStarted {
            name: file.name.clone(),
            bytes: file.bytes,
        });
        let mut output = tokio::fs::File::create(temp)
            .await
            .map_err(|source| io("create", temp, source))?;
        let mut stream = response.bytes_stream();
        let mut hasher = Sha256::new();
        let mut downloaded = 0u64;
        let mut reported = 0u64;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|source| Error::Http {
                url: url.clone(),
                source,
            })?;
            output
                .write_all(&chunk)
                .await
                .map_err(|source| io("write", temp, source))?;
            hasher.update(&chunk);
            downloaded += chunk.len() as u64;
            if downloaded == file.bytes || downloaded.saturating_sub(reported) >= 4 * 1024 * 1024 {
                report(DownloadEvent::DownloadProgress {
                    name: file.name.clone(),
                    downloaded,
                    total: file.bytes,
                    written: downloaded,
                });
                reported = downloaded;
            }
        }
        output
            .sync_all()
            .await
            .map_err(|source| io("sync", temp, source))?;
        drop(output);

        if downloaded != file.bytes {
            return Err(Error::Size {
                name: file.name.clone(),
                expected: file.bytes,
                actual: downloaded,
            });
        }
        let actual = lowercase_hex(&hasher.finalize());
        if actual != file.sha256 {
            return Err(Error::Checksum {
                name: file.name.clone(),
                expected: file.sha256.clone(),
                actual,
            });
        }
        Ok(())
    }

    async fn download_zstd<F>(
        &self,
        file: &ManifestFile,
        transport: &ManifestTransport,
        temp: &Path,
        report: &F,
    ) -> Result<()>
    where
        F: Fn(DownloadEvent) + Send + Sync,
    {
        let url = format!("{}/{}", self.base_url, transport.key);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|source| Error::Http {
                url: url.clone(),
                source,
            })?;
        if !response.status().is_success() {
            return Err(Error::HttpStatus {
                url,
                status: response.status().as_u16(),
            });
        }
        if let Some(bytes) = response.content_length()
            && bytes != transport.bytes
        {
            return Err(Error::Size {
                name: format!("{} (zstd transport)", file.name),
                expected: transport.bytes,
                actual: bytes,
            });
        }

        report(DownloadEvent::DownloadStarted {
            name: file.name.clone(),
            bytes: transport.bytes,
        });

        let downloaded = Arc::new(AtomicU64::new(0));
        let stream_downloaded = Arc::clone(&downloaded);
        let encoded_hasher = Arc::new(Mutex::new(Sha256::new()));
        let stream_hasher = Arc::clone(&encoded_hasher);
        let stream = response.bytes_stream().map(move |chunk| match chunk {
            Ok(chunk) => {
                stream_hasher
                    .lock()
                    .expect("encoded hash mutex poisoned")
                    .update(&chunk);
                stream_downloaded.fetch_add(chunk.len() as u64, Ordering::Relaxed);
                Ok(chunk)
            }
            Err(source) => Err(std::io::Error::other(source)),
        });
        let reader = StreamReader::new(stream);
        let mut decoder = ZstdDecoder::new(tokio::io::BufReader::new(reader));
        let mut output = tokio::fs::File::create(temp)
            .await
            .map_err(|source| io("create", temp, source))?;
        let mut decoded_hasher = Sha256::new();
        let mut decoded = 0u64;
        let mut reported = 0u64;
        let mut buffer = vec![0u8; IO_BUFFER_BYTES];
        loop {
            let read = decoder
                .read(&mut buffer)
                .await
                .map_err(|source| io("decompress into", temp, source))?;
            if read == 0 {
                break;
            }
            let next_decoded = decoded
                .checked_add(read as u64)
                .ok_or_else(|| Error::Size {
                    name: file.name.clone(),
                    expected: file.bytes,
                    actual: u64::MAX,
                })?;
            // Reject the decoded chunk before writing it when it would exceed
            // the canonical size. This bounds temporary disk usage as well as
            // the amount of work performed on a wrong or malicious frame.
            if next_decoded > file.bytes {
                return Err(Error::Size {
                    name: file.name.clone(),
                    expected: file.bytes,
                    actual: next_decoded,
                });
            }
            output
                .write_all(&buffer[..read])
                .await
                .map_err(|source| io("write", temp, source))?;
            decoded_hasher.update(&buffer[..read]);
            decoded = next_decoded;
            if decoded.saturating_sub(reported) >= 4 * 1024 * 1024 {
                report(DownloadEvent::DownloadProgress {
                    name: file.name.clone(),
                    downloaded: downloaded.load(Ordering::Relaxed),
                    total: transport.bytes,
                    written: decoded,
                });
                reported = decoded;
            }
        }
        drop(decoder);
        report(DownloadEvent::DownloadProgress {
            name: file.name.clone(),
            downloaded: downloaded.load(Ordering::Relaxed),
            total: transport.bytes,
            written: decoded,
        });
        output
            .sync_all()
            .await
            .map_err(|source| io("sync", temp, source))?;
        drop(output);

        let downloaded = downloaded.load(Ordering::Relaxed);
        if downloaded != transport.bytes {
            return Err(Error::Size {
                name: format!("{} (zstd transport)", file.name),
                expected: transport.bytes,
                actual: downloaded,
            });
        }
        let encoded_actual = lowercase_hex(
            &encoded_hasher
                .lock()
                .expect("encoded hash mutex poisoned")
                .clone()
                .finalize(),
        );
        if encoded_actual != transport.sha256 {
            return Err(Error::Checksum {
                name: format!("{} (zstd transport)", file.name),
                expected: transport.sha256.clone(),
                actual: encoded_actual,
            });
        }
        if decoded != file.bytes {
            return Err(Error::Size {
                name: file.name.clone(),
                expected: file.bytes,
                actual: decoded,
            });
        }
        let decoded_actual = lowercase_hex(&decoded_hasher.finalize());
        if decoded_actual != file.sha256 {
            return Err(Error::Checksum {
                name: file.name.clone(),
                expected: file.sha256.clone(),
                actual: decoded_actual,
            });
        }
        Ok(())
    }

    fn manifest_path(&self) -> PathBuf {
        self.cache_dir
            .join("manifests")
            .join(manifest_cache_file_name(&self.base_url))
    }

    fn object_path(&self, file: &ManifestFile) -> PathBuf {
        self.cache_dir
            .join("objects")
            .join(&file.sha256)
            .join(&file.name)
    }

    async fn acquire_artifact_lock(&self, file: &ManifestFile) -> Result<std::fs::File> {
        let lock_dir = self.cache_dir.join("locks");
        tokio::fs::create_dir_all(&lock_dir)
            .await
            .map_err(|source| io("create directory", &lock_dir, source))?;
        let path = lock_dir.join(format!("{}.lock", file.sha256));
        let task_path = path.clone();
        tokio::task::spawn_blocking(move || {
            let lock = std::fs::OpenOptions::new()
                .create(true)
                .read(true)
                .truncate(false)
                .write(true)
                .open(&task_path)?;
            FileExt::lock_exclusive(&lock)?;
            Ok::<_, std::io::Error>(lock)
        })
        .await
        .map_err(|error| Error::BackgroundTask(error.to_string()))?
        .map_err(|source| io("lock", path, source))
    }
}

async fn metadata_matches(path: &Path, expected_bytes: u64) -> Result<bool> {
    match tokio::fs::metadata(path).await {
        Ok(metadata) => Ok(metadata.is_file() && metadata.len() == expected_bytes),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(source) => Err(io("read metadata for", path, source)),
    }
}

async fn verify_artifact(path: &Path, artifact: &CatalogArtifact) -> Result<()> {
    let mut input = tokio::fs::File::open(path)
        .await
        .map_err(|source| io("open", path, source))?;
    let metadata = input
        .metadata()
        .await
        .map_err(|source| io("read metadata for", path, source))?;
    if metadata.len() != artifact.bytes {
        return Err(Error::Size {
            name: artifact.name.clone(),
            expected: artifact.bytes,
            actual: metadata.len(),
        });
    }

    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; IO_BUFFER_BYTES];
    loop {
        let read = input
            .read(&mut buffer)
            .await
            .map_err(|source| io("read", path, source))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let actual = lowercase_hex(&hasher.finalize());
    if actual != artifact.sha256 {
        return Err(Error::Checksum {
            name: artifact.name.clone(),
            expected: artifact.sha256.clone(),
            actual,
        });
    }
    Ok(())
}

async fn replace_file(temp: &Path, target: &Path) -> Result<()> {
    tokio::fs::rename(temp, target)
        .await
        .map_err(|source| io("rename", temp, source))
}

fn next_sequence() -> u64 {
    TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
}

fn lowercase_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn manifest_cache_file_name(base_url: &str) -> String {
    if base_url == DEFAULT_BUNDLE_BASE_URL {
        return "catalog-bundle-v4.json".into();
    }
    if base_url == LEGACY_V2_BUNDLE_BASE_URL {
        // Preserve the v0.5 cache location for explicitly configured v2 use.
        return "catalog-bundle-v2.json".into();
    }
    let digest = lowercase_hex(&Sha256::digest(base_url.as_bytes()));
    format!("catalog-bundle-{}.json", &digest[..16])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::REQUIRED_BUNDLE_FILES;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    fn sha256(bytes: &[u8]) -> String {
        lowercase_hex(&Sha256::digest(bytes))
    }

    fn cached_manifest(selected_bytes: &[u8]) -> BundleManifestDocument {
        BundleManifestDocument {
            manifest: crate::BundleManifest {
                version: "catalog-bundle-v4-test".into(),
                files: REQUIRED_BUNDLE_FILES
                    .iter()
                    .enumerate()
                    .map(|(index, &name)| {
                        let bytes = if name == "objects.bin" {
                            selected_bytes.to_vec()
                        } else {
                            vec![index as u8 + 1]
                        };
                        let sha256 = sha256(&bytes);
                        ManifestFile {
                            name: name.into(),
                            key: Some(format!("artifacts/{sha256}/{name}")),
                            bytes: bytes.len() as u64,
                            sha256,
                        }
                    })
                    .collect(),
            },
            transports: Vec::new(),
        }
    }

    #[tokio::test]
    async fn offline_cache_hit_does_not_hash_the_artifact() {
        let temp = tempfile::tempdir().unwrap();
        let original = b"object catalog";
        let manifest = cached_manifest(original);
        let manager = CatalogManager::builder()
            .cache_dir(temp.path())
            .base_url("http://127.0.0.1:1/unreachable")
            .policy(CachePolicy::OfflineOnly)
            .build()
            .unwrap();
        let manifest_path = manager.manifest_path();
        std::fs::create_dir_all(manifest_path.parent().unwrap()).unwrap();
        std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let entry = manifest
            .manifest
            .files
            .iter()
            .find(|file| file.name == "objects.bin")
            .unwrap();
        let object_path = temp
            .path()
            .join("objects")
            .join(&entry.sha256)
            .join(&entry.name);
        std::fs::create_dir_all(object_path.parent().unwrap()).unwrap();
        std::fs::write(&object_path, original).unwrap();

        let bundle = manager
            .ensure(&CatalogSet::dataset(Dataset::Objects))
            .await
            .unwrap();
        assert_eq!(bundle.path(Dataset::Objects).unwrap(), object_path);

        std::fs::write(&object_path, b"broken catalog").unwrap();
        assert_eq!(b"broken catalog".len(), original.len());
        manager
            .ensure(&CatalogSet::dataset(Dataset::Objects))
            .await
            .unwrap();
        assert!(bundle.verify().await.is_err());
    }

    #[tokio::test]
    async fn materialize_repairs_a_corrupt_same_size_output() {
        let temp = tempfile::tempdir().unwrap();
        let output = tempfile::tempdir().unwrap();
        let original = b"object catalog";
        let manifest = cached_manifest(original);
        let manager = CatalogManager::builder()
            .cache_dir(temp.path())
            .policy(CachePolicy::OfflineOnly)
            .build()
            .unwrap();
        let manifest_path = manager.manifest_path();
        std::fs::create_dir_all(manifest_path.parent().unwrap()).unwrap();
        std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let entry = manifest
            .manifest
            .files
            .iter()
            .find(|file| file.name == "objects.bin")
            .unwrap();
        let object_path = temp
            .path()
            .join("objects")
            .join(&entry.sha256)
            .join(&entry.name);
        std::fs::create_dir_all(object_path.parent().unwrap()).unwrap();
        std::fs::write(&object_path, original).unwrap();
        std::fs::write(output.path().join("objects.bin"), b"broken catalog").unwrap();

        let bundle = manager
            .ensure(&CatalogSet::dataset(Dataset::Objects))
            .await
            .unwrap();
        bundle.materialize(output.path()).await.unwrap();
        assert_eq!(
            std::fs::read(output.path().join("objects.bin")).unwrap(),
            original
        );
    }

    #[tokio::test]
    async fn materialize_removes_temporary_file_when_install_fails() {
        let cache = tempfile::tempdir().unwrap();
        let output = tempfile::tempdir().unwrap();
        let content = b"object catalog";
        let source = cache.path().join("objects.bin");
        std::fs::write(&source, content).unwrap();

        let name = "objects.bin".to_string();
        let bundle = CatalogBundle {
            version: "catalog-bundle-v4-test".into(),
            artifacts: BTreeMap::from([(
                name.clone(),
                CatalogArtifact {
                    name: name.clone(),
                    bytes: content.len() as u64,
                    sha256: sha256(content),
                    path: source,
                },
            )]),
        };

        // A directory at the destination forces the final file replacement to
        // fail after the artifact has been copied and verified.
        std::fs::create_dir(output.path().join(&name)).unwrap();
        assert!(bundle.materialize(output.path()).await.is_err());

        let entries = std::fs::read_dir(output.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect::<Vec<_>>();
        assert!(
            !entries
                .iter()
                .any(|entry| { entry.to_string_lossy().starts_with(".objects.bin.part-") })
        );
    }

    #[tokio::test]
    async fn remote_manifest_and_artifact_are_streamed_into_the_cache() {
        let content = b"hosted object catalog".to_vec();
        let manifest = cached_manifest(&content);
        let artifact_request = format!(
            "GET /{} ",
            manifest
                .manifest
                .files
                .iter()
                .find(|file| file.name == "objects.bin")
                .unwrap()
                .artifact_key()
        );
        let manifest_json = serde_json::to_vec(&manifest).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            // Two callers each refresh the manifest, but the per-artifact
            // lock permits only one objects.bin transfer.
            for _ in 0..3 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = [0u8; 4096];
                let read = stream.read(&mut request).unwrap();
                let request = String::from_utf8_lossy(&request[..read]);
                let body = if request.starts_with("GET /manifest.json ") {
                    manifest_json.as_slice()
                } else if request.starts_with(&artifact_request) {
                    content.as_slice()
                } else {
                    panic!("unexpected request: {request}");
                };
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .unwrap();
                stream.write_all(body).unwrap();
            }
        });

        let cache = tempfile::tempdir().unwrap();
        let manager = CatalogManager::builder()
            .cache_dir(cache.path())
            .base_url(format!("http://{address}"))
            .policy(CachePolicy::ForceRefresh)
            .build()
            .unwrap();
        let selection = CatalogSet::dataset(Dataset::Objects);
        let (first, second) = tokio::join!(manager.ensure(&selection), manager.ensure(&selection));
        let bundle = first.unwrap();
        let second = second.unwrap();
        bundle.verify().await.unwrap();
        second.verify().await.unwrap();
        assert_eq!(
            std::fs::read(bundle.path(Dataset::Objects).unwrap()).unwrap(),
            b"hosted object catalog"
        );
        server.join().unwrap();
    }

    #[tokio::test]
    async fn zstd_transport_is_streamed_into_the_uncompressed_cache() {
        let content = b"hosted object catalog\n".repeat(4096);
        let encoded = zstd::stream::encode_all(content.as_slice(), 22).unwrap();
        let mut manifest = cached_manifest(&content);
        let encoded_sha256 = sha256(&encoded);
        manifest.transports.push(ManifestTransport {
            name: "objects.bin".into(),
            encoding: "zstd".into(),
            key: format!("artifacts/{encoded_sha256}/objects.bin.zst"),
            bytes: encoded.len() as u64,
            sha256: encoded_sha256,
        });
        let artifact_request = format!(
            "GET /{} ",
            manifest.preferred_transport("objects.bin").unwrap().key
        );
        let manifest_json = serde_json::to_vec(&manifest).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = [0u8; 4096];
                let read = stream.read(&mut request).unwrap();
                let request = String::from_utf8_lossy(&request[..read]);
                let body = if request.starts_with("GET /manifest.json ") {
                    manifest_json.as_slice()
                } else if request.starts_with(&artifact_request) {
                    encoded.as_slice()
                } else {
                    panic!("unexpected request: {request}");
                };
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .unwrap();
                stream.write_all(body).unwrap();
            }
        });

        let cache = tempfile::tempdir().unwrap();
        let manager = CatalogManager::builder()
            .cache_dir(cache.path())
            .base_url(format!("http://{address}"))
            .policy(CachePolicy::ForceRefresh)
            .build()
            .unwrap();
        let events = Arc::new(Mutex::new(Vec::new()));
        let reported = Arc::clone(&events);
        let bundle = manager
            .ensure_with(&CatalogSet::dataset(Dataset::Objects), move |event| {
                reported.lock().unwrap().push(event);
            })
            .await
            .unwrap();
        bundle.verify().await.unwrap();
        assert_eq!(
            std::fs::read(bundle.path(Dataset::Objects).unwrap()).unwrap(),
            content
        );
        let events = events.lock().unwrap();
        assert!(events.iter().any(|event| matches!(
            event,
            DownloadEvent::DownloadStarted { name, bytes }
                if name == "objects.bin" && *bytes < content.len() as u64
        )));
        assert!(
            std::fs::read_dir(cache.path().join("partial"))
                .unwrap()
                .next()
                .is_none()
        );
        server.join().unwrap();
    }

    #[tokio::test]
    async fn invalid_zstd_transport_is_not_installed() {
        let content = b"hosted object catalog\n".repeat(128);
        let encoded = zstd::stream::encode_all(content.as_slice(), 22).unwrap();
        let mut manifest = cached_manifest(&content);
        let advertised_sha256 = sha256(b"not the encoded artifact");
        manifest.transports.push(ManifestTransport {
            name: "objects.bin".into(),
            encoding: "zstd".into(),
            key: format!("artifacts/{advertised_sha256}/objects.bin.zst"),
            bytes: encoded.len() as u64,
            sha256: advertised_sha256,
        });
        let logical_sha256 = manifest
            .manifest
            .files
            .iter()
            .find(|file| file.name == "objects.bin")
            .unwrap()
            .sha256
            .clone();
        let artifact_request = format!(
            "GET /{} ",
            manifest.preferred_transport("objects.bin").unwrap().key
        );
        let manifest_json = serde_json::to_vec(&manifest).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = [0u8; 4096];
                let read = stream.read(&mut request).unwrap();
                let request = String::from_utf8_lossy(&request[..read]);
                let body = if request.starts_with("GET /manifest.json ") {
                    manifest_json.as_slice()
                } else if request.starts_with(&artifact_request) {
                    encoded.as_slice()
                } else {
                    panic!("unexpected request: {request}");
                };
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .unwrap();
                stream.write_all(body).unwrap();
            }
        });

        let cache = tempfile::tempdir().unwrap();
        let manager = CatalogManager::builder()
            .cache_dir(cache.path())
            .base_url(format!("http://{address}"))
            .policy(CachePolicy::ForceRefresh)
            .build()
            .unwrap();
        let error = manager
            .ensure(&CatalogSet::dataset(Dataset::Objects))
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Checksum { name, .. } if name.contains("zstd transport")));
        assert!(
            !cache
                .path()
                .join("objects")
                .join(logical_sha256)
                .join("objects.bin")
                .exists()
        );
        assert!(
            std::fs::read_dir(cache.path().join("partial"))
                .unwrap()
                .next()
                .is_none()
        );
        server.join().unwrap();
    }

    #[tokio::test]
    async fn oversized_zstd_frame_aborts_before_decompressing_completely() {
        let content = b"hosted object catalog\n".repeat(128);
        // A ~64 MiB frame that transfers as a few kilobytes. The declared
        // canonical size is only a few KiB, so decoding must stop within the
        // first buffers instead of writing the complete expansion.
        let bomb = vec![0u8; 64 * 1024 * 1024];
        let encoded = zstd::stream::encode_all(bomb.as_slice(), 3).unwrap();
        let mut manifest = cached_manifest(&content);
        let encoded_sha256 = sha256(&encoded);
        manifest.transports.push(ManifestTransport {
            name: "objects.bin".into(),
            encoding: "zstd".into(),
            key: format!("artifacts/{encoded_sha256}/objects.bin.zst"),
            bytes: encoded.len() as u64,
            sha256: encoded_sha256,
        });
        let logical_sha256 = manifest
            .manifest
            .files
            .iter()
            .find(|file| file.name == "objects.bin")
            .unwrap()
            .sha256
            .clone();
        let artifact_request = format!(
            "GET /{} ",
            manifest.preferred_transport("objects.bin").unwrap().key
        );
        let manifest_json = serde_json::to_vec(&manifest).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = [0u8; 4096];
                let read = stream.read(&mut request).unwrap();
                let request = String::from_utf8_lossy(&request[..read]);
                let body = if request.starts_with("GET /manifest.json ") {
                    manifest_json.as_slice()
                } else if request.starts_with(&artifact_request) {
                    encoded.as_slice()
                } else {
                    panic!("unexpected request: {request}");
                };
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .unwrap();
                stream.write_all(body).unwrap();
            }
        });

        let cache = tempfile::tempdir().unwrap();
        let manager = CatalogManager::builder()
            .cache_dir(cache.path())
            .base_url(format!("http://{address}"))
            .policy(CachePolicy::ForceRefresh)
            .build()
            .unwrap();
        let error = manager
            .ensure(&CatalogSet::dataset(Dataset::Objects))
            .await
            .unwrap_err();
        match error {
            Error::Size {
                name,
                expected,
                actual,
            } => {
                assert_eq!(name, "objects.bin");
                assert_eq!(expected, content.len() as u64);
                assert!(actual > expected);
                assert!(
                    actual <= expected + IO_BUFFER_BYTES as u64,
                    "decoding continued beyond the first oversized chunk: {actual} bytes"
                );
            }
            other => panic!("expected a size error, got {other:?}"),
        }
        assert!(
            !cache
                .path()
                .join("objects")
                .join(logical_sha256)
                .join("objects.bin")
                .exists()
        );
        assert!(
            std::fs::read_dir(cache.path().join("partial"))
                .unwrap()
                .next()
                .is_none()
        );
        server.join().unwrap();
    }

    #[tokio::test]
    async fn explicitly_configured_v2_manifest_uses_the_legacy_flat_url() {
        let content = b"legacy object catalog".to_vec();
        let mut manifest = cached_manifest(&content);
        manifest.manifest.version = "catalog-bundle-v2-test".into();
        for file in &mut manifest.manifest.files {
            file.key = None;
        }
        let manifest_json = serde_json::to_vec(&manifest).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = [0u8; 4096];
                let read = stream.read(&mut request).unwrap();
                let request = String::from_utf8_lossy(&request[..read]);
                let body = if request.starts_with("GET /manifest.json ") {
                    manifest_json.as_slice()
                } else if request.starts_with("GET /objects.bin ") {
                    content.as_slice()
                } else {
                    panic!("unexpected request: {request}");
                };
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .unwrap();
                stream.write_all(body).unwrap();
            }
        });

        let cache = tempfile::tempdir().unwrap();
        let manager = CatalogManager::builder()
            .cache_dir(cache.path())
            .base_url(format!("http://{address}"))
            .policy(CachePolicy::ForceRefresh)
            .build()
            .unwrap();
        let bundle = manager
            .ensure(&CatalogSet::dataset(Dataset::Objects))
            .await
            .unwrap();
        bundle.verify().await.unwrap();
        assert_eq!(
            std::fs::read(bundle.path(Dataset::Objects).unwrap()).unwrap(),
            b"legacy object catalog"
        );
        server.join().unwrap();
    }

    #[test]
    fn manifest_caches_are_scoped_without_discarding_the_legacy_v2_path() {
        assert_eq!(
            manifest_cache_file_name(DEFAULT_BUNDLE_BASE_URL),
            "catalog-bundle-v4.json"
        );
        assert_eq!(
            manifest_cache_file_name(LEGACY_V2_BUNDLE_BASE_URL),
            "catalog-bundle-v2.json"
        );
        assert_ne!(
            manifest_cache_file_name("https://example.test/one"),
            manifest_cache_file_name("https://example.test/two")
        );
    }
}
