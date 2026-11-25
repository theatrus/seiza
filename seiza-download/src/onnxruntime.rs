//! Pinned CPU runtime installation, shared by native and bound hosts.
//! Installation never loads a library or changes a host's selected provider.

use crate::bundle::{self, ArtifactSpec, TransferEvent};
use crate::error::io;
use crate::{DownloadEvent, Error, Result};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

#[derive(Clone, Deserialize)]
struct Archive {
    name: String,
    bytes: u64,
    sha256: String,
    #[serde(default)]
    sources: Vec<String>,
    files: BTreeMap<String, String>,
}

#[derive(Clone, Deserialize)]
struct Release {
    os: String,
    arch: String,
    library: String,
    #[serde(flatten)]
    archive: Archive,
    support: Option<Archive>,
}

#[derive(Deserialize)]
struct Catalog {
    version: String,
    sources: Vec<String>,
    archives: Vec<Release>,
}

/// A verified local CPU runtime. Pass `library` to the model backend options.
/// The installer does not load it; an already loaded runtime needs a restart
/// before another runtime can be selected.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InstalledOnnxRuntime {
    /// ONNX Runtime's pinned version.
    pub version: String,
    /// Absolute path to the library, beside its required dependencies.
    pub library: PathBuf,
}

#[derive(Deserialize, Serialize)]
struct Receipt {
    release: String,
    directory: String,
    files: BTreeMap<String, String>,
}

/// Installs the pinned runtime into a host-owned directory. Downloads use the
/// PSF Guard mirror first, then Microsoft, with the same size and SHA-256 pins.
/// Windows also receives the app-local CRT from a pinned Seiza package. Old
/// installs remain untouched so another process can keep a library open.
pub struct OnnxRuntimeInstaller {
    directory: PathBuf,
    version: String,
    release: Release,
    client: reqwest::Client,
}

impl OnnxRuntimeInstaller {
    /// Select the tested archive for the current OS and architecture. Does not
    /// access the network, create files, or load a library.
    pub fn new(directory: impl Into<PathBuf>) -> Result<Self> {
        let catalog: Catalog = serde_json::from_str(include_str!("onnxruntime.json"))?;
        let mut release = catalog
            .archives
            .into_iter()
            .find(|release| {
                release.os == std::env::consts::OS && release.arch == std::env::consts::ARCH
            })
            .ok_or_else(|| Error::RuntimePlatform {
                os: std::env::consts::OS.into(),
                arch: std::env::consts::ARCH.into(),
            })?;
        release.archive.sources = catalog.sources;
        let directory = directory.into();
        let directory =
            std::path::absolute(&directory).map_err(|source| io("resolve", directory, source))?;
        Ok(Self {
            directory,
            version: catalog.version,
            release,
            client: bundle::http_client(
                concat!("seiza-download/", env!("CARGO_PKG_VERSION")),
                Duration::from_secs(60),
            )?,
        })
    }

    fn identity(&self) -> String {
        format!(
            "{}:{}",
            self.release.archive.sha256,
            self.release
                .support
                .as_ref()
                .map_or("", |support| support.sha256.as_str())
        )
    }

    /// Verify every installed library and license against the install receipt.
    /// Missing, incomplete, altered, or older installations return `None` and
    /// can be repaired by `install`. This check never accesses the network.
    pub fn installed(&self) -> Result<Option<InstalledOnnxRuntime>> {
        let path = self.directory.join("current.json");
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(io("read", path, error)),
        };
        let Ok(receipt) = serde_json::from_slice::<Receipt>(&bytes) else {
            return Ok(None);
        };
        if receipt.release != self.identity() || !bare_name(&receipt.directory) {
            return Ok(None);
        }
        let expected = self.release.archive.files.values().chain(
            self.release
                .support
                .iter()
                .flat_map(|support| support.files.values()),
        );
        let names: std::collections::BTreeSet<_> = expected.collect();
        if names != receipt.files.keys().collect() {
            return Ok(None);
        }
        for (name, hash) in &receipt.files {
            if !bare_name(name) {
                return Ok(None);
            }
            let path = self.directory.join(&receipt.directory).join(name);
            match hash_file(&path) {
                Ok(actual) if &actual == hash => {}
                Ok(_) => return Ok(None),
                Err(Error::Io { source, .. }) if source.kind() == std::io::ErrorKind::NotFound => {
                    return Ok(None);
                }
                Err(error) => return Err(error),
            }
        }
        Ok(Some(InstalledOnnxRuntime {
            version: self.version.clone(),
            library: self
                .directory
                .join(receipt.directory)
                .join(&self.release.library),
        }))
    }

    /// Install or repair the current runtime, returning a verified library.
    /// `cancel` is checked while waiting, transferring and unpacking. Dropping
    /// the future removes its temporary files. Publication is one atomic
    /// receipt replacement; failures and cancellation preserve the old install.
    /// Extraction is bounded and runs on this thread; call off the UI thread.
    pub async fn install(
        &self,
        report: impl Fn(DownloadEvent),
        cancel: impl Fn() -> bool,
    ) -> Result<InstalledOnnxRuntime> {
        stopped(&cancel)?;
        if let Some(installed) = self.installed()? {
            return Ok(installed);
        }
        std::fs::create_dir_all(&self.directory)
            .map_err(|source| io("create directory", &self.directory, source))?;
        let lock_path = self.directory.join("install.lock");
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(|source| io("open", &lock_path, source))?;
        loop {
            stopped(&cancel)?;
            match lock.try_lock_exclusive() {
                Ok(()) => break,
                Err(error)
                    if error.raw_os_error() == fs2::lock_contended_error().raw_os_error() =>
                {
                    tokio::time::sleep(Duration::from_millis(50)).await
                }
                Err(error) => return Err(io("lock", lock_path, error)),
            }
        }
        if let Some(installed) = self.installed()? {
            return Ok(installed);
        }
        let staging = tempfile::Builder::new()
            .prefix(".onnxruntime-")
            .tempdir_in(&self.directory)
            .map_err(|source| io("create staging directory", &self.directory, source))?;
        let payload = staging.path().join("payload");
        std::fs::create_dir(&payload).map_err(|source| io("create directory", &payload, source))?;
        for archive in std::iter::once(&self.release.archive).chain(self.release.support.iter()) {
            let path = staging.path().join(&archive.name);
            self.fetch(archive, &path, &report, &cancel).await?;
            report(DownloadEvent::Installing {
                name: archive.name.clone(),
                path: payload.clone(),
            });
            unpack(archive, &path, &payload, &cancel)?;
        }
        let files = self
            .release
            .archive
            .files
            .values()
            .chain(self.release.support.iter().flat_map(|s| s.files.values()))
            .map(|name| Ok((name.clone(), hash_file(&payload.join(name))?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        stopped(&cancel)?;
        let directory = format!(
            "onnxruntime-{}-{}",
            self.version,
            staging.path().file_name().unwrap().to_string_lossy()
        );
        let destination = self.directory.join(&directory);
        let receipt = Receipt {
            release: self.identity(),
            directory,
            files,
        };
        let pointer = staging.path().join("current.json");
        std::fs::write(&pointer, serde_json::to_vec(&receipt)?)
            .map_err(|source| io("write", &pointer, source))?;
        std::fs::rename(&payload, &destination)
            .map_err(|source| io("install", &destination, source))?;
        if let Err(error) = std::fs::rename(&pointer, self.directory.join("current.json")) {
            let _ = std::fs::remove_dir_all(&destination);
            return Err(io("publish runtime receipt", pointer, error));
        }
        let installed = InstalledOnnxRuntime {
            version: self.version.clone(),
            library: destination.join(&self.release.library),
        };
        report(DownloadEvent::InstallComplete {
            name: "ONNX Runtime".into(),
            path: installed.library.clone(),
        });
        Ok(installed)
    }

    async fn fetch(
        &self,
        archive: &Archive,
        path: &Path,
        report: &impl Fn(DownloadEvent),
        cancel: &impl Fn() -> bool,
    ) -> Result<()> {
        let mut failures = Vec::new();
        for source in &archive.sources {
            stopped(cancel)?;
            let url = format!("{}/{}", source.trim_end_matches('/'), archive.name);
            let spec = ArtifactSpec {
                label: &archive.name,
                url: &url,
                decoded_bytes: archive.bytes,
                decoded_sha256: &archive.sha256,
                encoded: None,
            };
            let progress = |event| match event {
                TransferEvent::Started { total } => report(DownloadEvent::DownloadStarted {
                    name: archive.name.clone(),
                    bytes: total,
                }),
                TransferEvent::Progress(p) => report(DownloadEvent::DownloadProgress {
                    name: archive.name.clone(),
                    downloaded: p.downloaded,
                    total: p.total,
                    written: p.written,
                }),
            };
            let download = bundle::stream_to_temp(&self.client, &spec, path, progress);
            tokio::pin!(download);
            let result = loop {
                tokio::select! {
                    result = &mut download => break result,
                    _ = tokio::time::sleep(Duration::from_millis(50)) => stopped(cancel)?,
                }
            };
            match result {
                Ok(()) => return Ok(()),
                Err(error @ Error::Io { .. }) => return Err(error),
                Err(error) => failures.push(format!("{url}: {error}")),
            }
        }
        Err(Error::RuntimeSources(failures))
    }
}

fn stopped(cancel: &impl Fn() -> bool) -> Result<()> {
    if cancel() {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}

fn bare_name(name: &str) -> bool {
    let mut parts = Path::new(name).components();
    !name.contains(['/', '\\', ':'])
        && matches!(parts.next(), Some(Component::Normal(_)))
        && parts.next().is_none()
}

fn hash_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path).map_err(|source| io("open", path, source))?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let n = file
            .read(&mut buffer)
            .map_err(|source| io("read", path, source))?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(bundle::lowercase_hex(&hash.finalize()))
}

fn unpack(
    archive: &Archive,
    path: &Path,
    destination: &Path,
    cancel: &impl Fn() -> bool,
) -> Result<()> {
    let invalid = |error: &dyn std::fmt::Display| Error::RuntimeArchive(error.to_string());
    let file = std::fs::File::open(path).map_err(|source| io("open", path, source))?;
    let mut remaining = archive.files.clone();
    if archive.name.ends_with(".zip") {
        let mut zip = zip::ZipArchive::new(file).map_err(|error| invalid(&error))?;
        for (source, name) in &archive.files {
            stopped(cancel)?;
            let mut entry = zip.by_name(source).map_err(|error| invalid(&error))?;
            if !entry.is_file() || entry.is_symlink() {
                return Err(invalid(&"expected a regular file"));
            }
            extract(&mut entry, &destination.join(name), cancel)?;
            remaining.remove(source);
        }
    } else {
        let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(file));
        for entry in tar.entries().map_err(|error| invalid(&error))? {
            stopped(cancel)?;
            let mut entry = entry.map_err(|error| invalid(&error))?;
            let source = entry
                .path()
                .map_err(|error| invalid(&error))?
                .to_string_lossy()
                .into_owned();
            if let Some(name) = remaining.remove(&source) {
                if !entry.header().entry_type().is_file() {
                    return Err(invalid(&"expected a regular file"));
                }
                extract(&mut entry, &destination.join(name), cancel)?;
            }
        }
    }
    if !remaining.is_empty() {
        return Err(invalid(&format!("missing files: {:?}", remaining.keys())));
    }
    Ok(())
}

fn extract(reader: &mut impl Read, path: &Path, cancel: &impl Fn() -> bool) -> Result<()> {
    if !path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(bare_name)
    {
        return Err(Error::RuntimeArchive("unsafe file name".into()));
    }
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|source| io("create", path, source))?;
    let mut buffer = [0; 65536];
    let mut total = 0u64;
    loop {
        stopped(cancel)?;
        let n = reader
            .read(&mut buffer)
            .map_err(|source| io("unpack", path, source))?;
        if n == 0 {
            break;
        }
        total += n as u64;
        if total > 512 * 1024 * 1024 {
            return Err(Error::RuntimeArchive("library exceeds size limit".into()));
        }
        output
            .write_all(&buffer[..n])
            .map_err(|source| io("write", path, source))?;
    }
    output.sync_all().map_err(|source| io("sync", path, source))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn zipped() -> Vec<u8> {
        let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
        zip.start_file(
            "runtime/lib/test.dll",
            zip::write::SimpleFileOptions::default(),
        )
        .unwrap();
        zip.write_all(b"test library bytes").unwrap();
        zip.finish().unwrap().into_inner()
    }

    fn server(responses: Vec<(u16, Vec<u8>)>) -> (String, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let thread = std::thread::spawn(move || {
            for (status, body) in responses {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(10)))
                    .unwrap();
                let mut request = Vec::new();
                while !request.ends_with(b"\r\n\r\n") {
                    let mut byte = [0];
                    stream.read_exact(&mut byte).unwrap();
                    request.push(byte[0]);
                }
                write!(
                    stream,
                    "HTTP/1.1 {status} Test\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .unwrap();
                let _ = stream.write_all(&body);
            }
        });
        (url, thread)
    }

    fn installer(directory: &Path, sources: Vec<String>, bytes: &[u8]) -> OnnxRuntimeInstaller {
        OnnxRuntimeInstaller {
            directory: directory.into(),
            version: "test".into(),
            client: bundle::http_client("test", Duration::from_secs(2)).unwrap(),
            release: Release {
                os: "test".into(),
                arch: "test".into(),
                library: "test.dll".into(),
                support: None,
                archive: Archive {
                    name: "runtime.zip".into(),
                    bytes: bytes.len() as u64,
                    sha256: bundle::lowercase_hex(&Sha256::digest(bytes)),
                    sources,
                    files: BTreeMap::from([("runtime/lib/test.dll".into(), "test.dll".into())]),
                },
            },
        }
    }

    #[tokio::test]
    async fn failed_and_corrupt_sources_fall_back_then_work_offline() {
        let bytes = zipped();
        let (url, thread) = server(vec![
            (404, vec![]),
            (200, vec![b'x'; bytes.len()]),
            (200, bytes.clone()),
        ]);
        let dir = tempfile::tempdir().unwrap();
        let manager = installer(dir.path(), vec![url.clone(), url.clone(), url], &bytes);
        let installed = manager.install(|_| {}, || false).await.unwrap();
        thread.join().unwrap();
        assert_eq!(
            std::fs::read(&installed.library).unwrap(),
            b"test library bytes"
        );
        assert_eq!(
            manager.install(|_| {}, || false).await.unwrap().library,
            installed.library
        );
        assert_eq!(
            manager.installed().unwrap().unwrap().library,
            installed.library
        );
        assert!(!std::fs::read_dir(dir.path()).unwrap().any(|e| {
            e.unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".onnxruntime-")
        }));
    }

    #[tokio::test]
    async fn source_failures_preserve_the_previous_receipt() {
        let bytes = zipped();
        let (url, thread) = server(vec![
            (200, bytes.clone()),
            (404, vec![]),
            (200, vec![0; bytes.len()]),
        ]);
        let dir = tempfile::tempdir().unwrap();
        let mut manager = installer(dir.path(), vec![url.clone()], &bytes);
        let previous = manager.install(|_| {}, || false).await.unwrap();
        let receipt = std::fs::read(dir.path().join("current.json")).unwrap();
        manager.release.archive.sha256 = "0".repeat(64);
        manager.release.archive.sources = vec![url.clone(), url];
        match manager.install(|_| {}, || false).await {
            Err(Error::RuntimeSources(failures)) => assert_eq!(failures.len(), 2),
            other => panic!("{other:?}"),
        }
        thread.join().unwrap();
        assert_eq!(
            std::fs::read(dir.path().join("current.json")).unwrap(),
            receipt
        );
        assert!(previous.library.is_file());
    }

    #[tokio::test]
    async fn damaged_files_are_repaired_in_a_new_directory() {
        let bytes = zipped();
        let (url, thread) = server(vec![(200, bytes.clone()), (200, bytes.clone())]);
        let dir = tempfile::tempdir().unwrap();
        let manager = installer(dir.path(), vec![url], &bytes);
        let first = manager.install(|_| {}, || false).await.unwrap();
        std::fs::write(&first.library, b"corrupt").unwrap();
        assert!(manager.installed().unwrap().is_none());
        let second = manager.install(|_| {}, || false).await.unwrap();
        thread.join().unwrap();
        assert_ne!(first.library, second.library);
        assert_eq!(std::fs::read(first.library).unwrap(), b"corrupt");
        assert_eq!(
            std::fs::read(second.library).unwrap(),
            b"test library bytes"
        );
    }

    #[tokio::test]
    async fn cancelling_during_extraction_publishes_nothing() {
        let bytes = zipped();
        let (url, thread) = server(vec![(200, bytes.clone())]);
        let dir = tempfile::tempdir().unwrap();
        let manager = installer(dir.path(), vec![url], &bytes);
        let cancelled = AtomicBool::new(false);
        let report = |event| {
            if matches!(event, DownloadEvent::Installing { .. }) {
                cancelled.store(true, Ordering::Relaxed)
            }
        };
        assert!(matches!(
            manager
                .install(report, || cancelled.load(Ordering::Relaxed))
                .await,
            Err(Error::Cancelled)
        ));
        thread.join().unwrap();
        assert!(!dir.path().join("current.json").exists());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1); // lock only
    }

    #[tokio::test]
    async fn cancellation_before_start_never_contacts_a_source() {
        let dir = tempfile::tempdir().unwrap();
        let manager = installer(dir.path(), vec!["http://127.0.0.1:1".into()], &zipped());
        assert!(matches!(
            manager.install(|_| {}, || true).await,
            Err(Error::Cancelled)
        ));
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[tokio::test]
    async fn a_local_write_failure_does_not_try_another_source() {
        let bytes = zipped();
        let (url, thread) = server(vec![(200, bytes.clone())]);
        let dir = tempfile::tempdir().unwrap();
        let manager = installer(dir.path(), vec![url, "http://127.0.0.1:1".into()], &bytes);
        // A directory cannot be opened as the destination archive.
        let error = manager
            .fetch(&manager.release.archive, dir.path(), &|_| {}, &|| false)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Io { .. }));
        thread.join().unwrap();
    }

    #[tokio::test]
    async fn concurrent_installs_publish_one_complete_directory() {
        let bytes = zipped();
        let (url, thread) = server(vec![(200, bytes.clone())]);
        let dir = tempfile::tempdir().unwrap();
        let manager = installer(dir.path(), vec![url], &bytes);
        let (a, b) = tokio::join!(
            manager.install(|_| {}, || false),
            manager.install(|_| {}, || false)
        );
        assert_eq!(a.unwrap().library, b.unwrap().library);
        thread.join().unwrap();
    }

    #[test]
    fn corrupt_receipts_cannot_escape_the_install_directory() {
        let dir = tempfile::tempdir().unwrap();
        let manager = installer(dir.path(), vec![], &zipped());
        for directory in ["../outside", "..\\outside", "C:outside", "/outside"] {
            let receipt = Receipt {
                release: manager.identity(),
                directory: directory.into(),
                files: BTreeMap::new(),
            };
            std::fs::write(
                dir.path().join("current.json"),
                serde_json::to_vec(&receipt).unwrap(),
            )
            .unwrap();
            assert!(manager.installed().unwrap().is_none());
        }
    }

    #[test]
    fn packaging_and_installer_share_the_same_runtime_pins() {
        let catalog: Catalog = serde_json::from_str(include_str!("onnxruntime.json")).unwrap();
        // The packaging directory is not part of the published download crate.
        let env = std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../packaging/onnxruntime.env"),
        )
        .ok();
        if let Some(env) = &env {
            assert!(env.contains(&format!("ONNXRUNTIME_VERSION={}", catalog.version)));
        }
        for release in catalog.archives {
            if let Some(env) = &env {
                assert!(env.contains(&release.archive.sha256));
            }
            assert!(release.archive.files.values().all(|name| bare_name(name)));
            assert!(
                release
                    .archive
                    .files
                    .values()
                    .any(|name| name == &release.library)
            );
        }
    }
}
