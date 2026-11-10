//! One hosted endpoint: a small manifest under a base URL, cached as a
//! [`CachePolicy`] allows, and the immutable artifacts it names, installed
//! into a content-addressed cache. Catalog bundles and models each keep one;
//! what a manifest offers, and which of it a caller wants, stay with them.

use crate::bundle::{self, ArtifactSpec, EncodedSpec, Encoding, TransferEvent};
use crate::error::io;
use crate::{CachePolicy, DownloadEvent, Error, ManifestFile, ManifestTransport, Result};
use directories::ProjectDirs;
use fs2::FileExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use tokio::io::AsyncWriteExt;

/// A hosted manifest document, read from its JSON and validated.
pub(crate) trait HostedManifest: Sized {
    fn parse(json: &[u8]) -> Result<Self>;
    fn version(&self) -> &str;
}

#[derive(Clone, Debug)]
pub(crate) struct Hosted {
    pub(crate) cache_dir: PathBuf,
    base_url: String,
    policy: CachePolicy,
    client: reqwest::Client,
    /// The cached manifest's file name under `manifests/`.
    manifest_file: String,
}

impl Hosted {
    /// An endpoint caching in `cache_dir`, else `SEIZA_CACHE_DIR`, else
    /// `default_dir` in the platform's cache directory.
    pub(crate) fn new(
        cache_dir: Option<PathBuf>,
        default_dir: &str,
        base_url: String,
        policy: CachePolicy,
        manifest_file: String,
    ) -> Result<Self> {
        let cache_dir = cache_dir
            .or_else(|| std::env::var_os("SEIZA_CACHE_DIR").map(PathBuf::from))
            .or_else(|| {
                ProjectDirs::from("fyi", "Seiza", "seiza")
                    .map(|dirs| dirs.cache_dir().join(default_dir))
            })
            .ok_or(Error::CacheDirectoryUnavailable)?;
        let client = bundle::http_client(
            format!("seiza-download/{}", env!("CARGO_PKG_VERSION")),
            Duration::from_secs(300),
        )?;
        Ok(Self {
            cache_dir,
            base_url,
            policy,
            client,
            manifest_file,
        })
    }

    pub(crate) async fn load_manifest<M, F>(&self, report: &F) -> Result<M>
    where
        M: HostedManifest,
        F: Fn(DownloadEvent) + Send + Sync,
    {
        let cached = self.manifest_path();
        let using = |manifest: &M, stale: bool| {
            report(DownloadEvent::UsingCachedManifest {
                version: manifest.version().to_string(),
                stale,
            })
        };
        match self.policy {
            CachePolicy::OfflineOnly => {
                let manifest = self
                    .read_cached_manifest::<M>()
                    .await?
                    .ok_or_else(|| Error::CachedManifestUnavailable(cached))?;
                using(&manifest, false);
                Ok(manifest)
            }
            CachePolicy::PreferCached => {
                if let Ok(Some(manifest)) = self.read_cached_manifest::<M>().await {
                    using(&manifest, false);
                    Ok(manifest)
                } else {
                    self.fetch_manifest(report).await
                }
            }
            CachePolicy::ForceRefresh => self.fetch_manifest(report).await,
            CachePolicy::RefreshIfOlderThan(max_age) => {
                if self.cached_manifest_is_fresh(max_age).await?
                    && let Ok(Some(manifest)) = self.read_cached_manifest::<M>().await
                {
                    using(&manifest, false);
                    return Ok(manifest);
                }
                match self.fetch_manifest(report).await {
                    Ok(manifest) => Ok(manifest),
                    Err(fetch_error) => match self.read_cached_manifest::<M>().await {
                        Ok(Some(manifest)) => {
                            using(&manifest, true);
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

    async fn read_cached_manifest<M: HostedManifest>(&self) -> Result<Option<M>> {
        let path = self.manifest_path();
        let bytes = match tokio::fs::read(&path).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(source) => return Err(io("read", path, source)),
        };
        M::parse(&bytes).map(Some)
    }

    async fn fetch_manifest<M, F>(&self, report: &F) -> Result<M>
    where
        M: HostedManifest,
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
        let manifest = M::parse(&bytes)?;
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
            self.manifest_file,
            std::process::id(),
            bundle::next_sequence()
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
            bundle::replace_file(&temp, &path).await
        }
        .await;
        if write.is_err() {
            let _ = tokio::fs::remove_file(&temp).await;
        }
        write
    }

    /// The verified cache object for `file`, downloading it through
    /// `transport` when given, or as it is, on a miss.
    pub(crate) async fn ensure_file<F>(
        &self,
        file: &ManifestFile,
        transport: Option<&ManifestTransport>,
        report: &F,
    ) -> Result<PathBuf>
    where
        F: Fn(DownloadEvent) + Send + Sync,
    {
        let target = self.object_path(file);
        if bundle::size_matches(&target, file.bytes).await? {
            report(DownloadEvent::CacheHit {
                name: file.name.clone(),
                path: target.clone(),
            });
            return Ok(target);
        }

        // Lock only cache misses, then re-check after waiting. This avoids
        // duplicate multi-gigabyte transfers across application processes.
        let _lock = self.acquire_artifact_lock(file).await?;
        if bundle::size_matches(&target, file.bytes).await? {
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
            bundle::next_sequence(),
            file.name
        ));
        let install = async {
            self.download_file(file, transport, &temp, report).await?;

            let parent = target.parent().expect("object path has a parent");
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|source| io("create directory", parent, source))?;
            if bundle::size_matches(&target, file.bytes).await? {
                tokio::fs::remove_file(&temp)
                    .await
                    .map_err(|source| io("remove", &temp, source))?;
            } else {
                bundle::replace_file(&temp, &target).await?;
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
        let url = match transport {
            Some(transport) => format!("{}/{}", self.base_url, transport.key),
            None => format!("{}/{}", self.base_url, file.artifact_key()),
        };
        let encoded = transport.map(|transport| EncodedSpec {
            encoding: Encoding::Zstd,
            bytes: transport.bytes,
            sha256: &transport.sha256,
        });
        let spec = ArtifactSpec {
            label: &file.name,
            url: &url,
            decoded_bytes: file.bytes,
            decoded_sha256: &file.sha256,
            encoded,
        };
        bundle::stream_to_temp(&self.client, &spec, temp, |event| match event {
            TransferEvent::Started { total } => report(DownloadEvent::DownloadStarted {
                name: file.name.clone(),
                bytes: total,
            }),
            TransferEvent::Progress(progress) => report(DownloadEvent::DownloadProgress {
                name: file.name.clone(),
                downloaded: progress.downloaded,
                total: progress.total,
                written: progress.written,
            }),
        })
        .await
    }

    pub(crate) fn manifest_path(&self) -> PathBuf {
        self.cache_dir.join("manifests").join(&self.manifest_file)
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

/// The cached manifest's file name for an endpoint at `base_url` that has
/// no name of its own: `prefix` and the start of the URL's hash, so two
/// endpoints never share one.
pub(crate) fn scoped_manifest_file(prefix: &str, base_url: &str) -> String {
    let digest = bundle::sha256_hex(base_url.as_bytes());
    format!("{prefix}-{}.json", &digest[..16])
}
