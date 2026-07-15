use crate::error::io;
use crate::{
    BundleManifest, CatalogSet, DEFAULT_BUNDLE_BASE_URL, Dataset, Error, ManifestFile, Result,
};
use directories::ProjectDirs;
use fs2::FileExt;
use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const READ_TIMEOUT: Duration = Duration::from_secs(300);

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
        downloaded: u64,
        total: u64,
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
        let plan = manifest.plan(set)?;
        let mut artifacts = BTreeMap::new();

        for file in plan {
            let path = self.ensure_file(&file, &report).await?;
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
            version: manifest.version,
            artifacts,
        })
    }

    async fn load_manifest<F>(&self, report: &F) -> Result<BundleManifest>
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
                    version: manifest.version.clone(),
                    stale: false,
                });
                Ok(manifest)
            }
            CachePolicy::PreferCached => {
                if let Ok(Some(manifest)) = self.read_cached_manifest().await {
                    report(DownloadEvent::UsingCachedManifest {
                        version: manifest.version.clone(),
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
                        version: manifest.version.clone(),
                        stale: false,
                    });
                    return Ok(manifest);
                }
                match self.fetch_manifest(report).await {
                    Ok(manifest) => Ok(manifest),
                    Err(fetch_error) => match self.read_cached_manifest().await {
                        Ok(Some(manifest)) => {
                            report(DownloadEvent::UsingCachedManifest {
                                version: manifest.version.clone(),
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

    async fn read_cached_manifest(&self) -> Result<Option<BundleManifest>> {
        let path = self.manifest_path();
        let bytes = match tokio::fs::read(&path).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(source) => return Err(io("read", path, source)),
        };
        BundleManifest::parse(&bytes).map(Some)
    }

    async fn fetch_manifest<F>(&self, report: &F) -> Result<BundleManifest>
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
        let manifest = BundleManifest::parse(&bytes)?;
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
            ".catalog-bundle-v2.json.part-{}-{}",
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

    async fn ensure_file<F>(&self, file: &ManifestFile, report: &F) -> Result<PathBuf>
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
            self.download_file(file, &temp, report).await?;

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

    async fn download_file<F>(&self, file: &ManifestFile, temp: &Path, report: &F) -> Result<()>
    where
        F: Fn(DownloadEvent) + Send + Sync,
    {
        let url = format!("{}/{}", self.base_url, file.name);
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
        let actual = format!("{:x}", hasher.finalize());
        if actual != file.sha256 {
            return Err(Error::Checksum {
                name: file.name.clone(),
                expected: file.sha256.clone(),
                actual,
            });
        }
        Ok(())
    }

    fn manifest_path(&self) -> PathBuf {
        self.cache_dir
            .join("manifests")
            .join("catalog-bundle-v2.json")
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
    let mut buffer = vec![0u8; 1024 * 1024];
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
    let actual = format!("{:x}", hasher.finalize());
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::REQUIRED_V2_FILES;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn cached_manifest(selected_bytes: &[u8]) -> BundleManifest {
        BundleManifest {
            version: "catalog-bundle-v2-test".into(),
            files: REQUIRED_V2_FILES
                .iter()
                .enumerate()
                .map(|(index, name)| {
                    let bytes = if *name == "objects.bin" {
                        selected_bytes.to_vec()
                    } else {
                        vec![index as u8 + 1]
                    };
                    ManifestFile {
                        name: (*name).into(),
                        bytes: bytes.len() as u64,
                        sha256: sha256(&bytes),
                    }
                })
                .collect(),
        }
    }

    #[tokio::test]
    async fn offline_cache_hit_does_not_hash_the_artifact() {
        let temp = tempfile::tempdir().unwrap();
        let original = b"object catalog";
        let manifest = cached_manifest(original);
        let manifest_path = temp.path().join("manifests/catalog-bundle-v2.json");
        std::fs::create_dir_all(manifest_path.parent().unwrap()).unwrap();
        std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let entry = manifest
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

        let manager = CatalogManager::builder()
            .cache_dir(temp.path())
            .base_url("http://127.0.0.1:1/unreachable")
            .policy(CachePolicy::OfflineOnly)
            .build()
            .unwrap();
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
        let manifest_path = temp.path().join("manifests/catalog-bundle-v2.json");
        std::fs::create_dir_all(manifest_path.parent().unwrap()).unwrap();
        std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let entry = manifest
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

        let manager = CatalogManager::builder()
            .cache_dir(temp.path())
            .policy(CachePolicy::OfflineOnly)
            .build()
            .unwrap();
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
            version: "catalog-bundle-v2-test".into(),
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
}
