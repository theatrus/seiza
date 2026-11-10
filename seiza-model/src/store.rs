use crate::bundle::{CARD_FILE, ModelBundle, Task};
use crate::error::ModelError;
use directories::ProjectDirs;
use std::path::{Path, PathBuf};

/// Names the directory models are installed in, for hosts that keep them
/// elsewhere.
const STORE_VARIABLE: &str = "SEIZA_MODEL_DIR";

/// The directory models are installed in, one bundle per model.
#[derive(Clone, Debug)]
pub struct ModelStore {
    root: PathBuf,
}

impl ModelStore {
    /// A store at `root`, which need not exist yet.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The store `SEIZA_MODEL_DIR` names, or the one in the platform's data
    /// directory.
    pub fn default_store() -> Result<Self, ModelError> {
        std::env::var_os(STORE_VARIABLE)
            .map(PathBuf::from)
            .or_else(|| {
                ProjectDirs::from("fyi", "Seiza", "seiza")
                    .map(|dirs| dirs.data_dir().join("models"))
            })
            .map(Self::new)
            .ok_or_else(|| {
                ModelError::InvalidOption(format!(
                    "no data directory is known here: set {STORE_VARIABLE}"
                ))
            })
    }

    /// The store's directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Every installed model, by name. A directory with no readable card is
    /// left out.
    pub fn list(&self) -> Result<Vec<ModelBundle>, ModelError> {
        let entries = match std::fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(source) => {
                return Err(ModelError::Io {
                    path: self.root.clone(),
                    source,
                });
            }
        };
        let mut bundles: Vec<ModelBundle> = entries
            .filter_map(Result::ok)
            .filter(|entry| entry.path().join(CARD_FILE).is_file())
            .filter_map(|entry| ModelBundle::open(entry.path()).ok())
            .collect();
        bundles.sort_by(|a, b| a.card().name.cmp(&b.card().name));
        Ok(bundles)
    }

    /// The installed model named `name`.
    pub fn find(&self, name: &str) -> Result<ModelBundle, ModelError> {
        self.list()?
            .into_iter()
            .find(|bundle| bundle.card().name == name)
            .ok_or_else(|| ModelError::ModelNotFound(name.to_string()))
    }

    /// The model `chosen` names: a bundle's directory, or an installed model.
    /// With nothing chosen, the one installed noise model.
    pub fn resolve(&self, chosen: Option<&str>) -> Result<ModelBundle, ModelError> {
        if let Some(chosen) = chosen {
            if Path::new(chosen).join(CARD_FILE).is_file() {
                return ModelBundle::open(chosen);
            }
            return self.find(chosen);
        }
        let mut models: Vec<ModelBundle> = self
            .list()?
            .into_iter()
            .filter(|bundle| bundle.card().task == Task::Denoise)
            .collect();
        match models.len() {
            0 => Err(ModelError::NoModelInstalled {
                store: self.root.clone(),
            }),
            1 => Ok(models.remove(0)),
            _ => Err(ModelError::SeveralModels {
                names: models
                    .iter()
                    .map(|bundle| bundle.card().name.clone())
                    .collect(),
            }),
        }
    }

    /// Copy the bundle at `source` into the store, replacing any model of the
    /// same name, once its graphs match its card. It appears whole or not at
    /// all.
    pub fn install(&self, source: impl AsRef<Path>) -> Result<ModelBundle, ModelError> {
        let bundle = ModelBundle::open(source)?;
        bundle.verify()?;
        self.install_staged(&bundle.card().name, |staging| {
            let files = std::iter::once(CARD_FILE)
                .chain(bundle.card().graphs.iter().map(|graph| graph.file.as_str()));
            for file in files {
                let from = bundle.root().join(file);
                std::fs::copy(&from, staging.join(file)).map_err(ModelError::io(&from))?;
            }
            Ok(())
        })
    }

    /// Have `fill` put a bundle in a staging directory, then put that in place
    /// of any model named `name`. Nothing replaces the model when `fill`
    /// fails.
    fn install_staged(
        &self,
        name: &str,
        fill: impl FnOnce(&Path) -> Result<(), ModelError>,
    ) -> Result<ModelBundle, ModelError> {
        std::fs::create_dir_all(&self.root).map_err(ModelError::io(&self.root))?;
        let target = self.root.join(name);
        let staging = self
            .root
            .join(format!(".installing-{name}-{}", std::process::id()));
        let result = (|| {
            std::fs::create_dir_all(&staging).map_err(ModelError::io(&staging))?;
            fill(&staging)?;
            if target.exists() {
                std::fs::remove_dir_all(&target).map_err(ModelError::io(&target))?;
            }
            std::fs::rename(&staging, &target).map_err(ModelError::io(&target))
        })();
        if result.is_err() {
            let _ = std::fs::remove_dir_all(&staging);
        }
        result?;
        ModelBundle::open(&target)
    }

    /// Delete the installed model named `name`.
    pub fn remove(&self, name: &str) -> Result<(), ModelError> {
        let bundle = self.find(name)?;
        std::fs::remove_dir_all(bundle.root()).map_err(ModelError::io(bundle.root()))
    }
}

#[cfg(feature = "download")]
impl ModelStore {
    /// Fetch the model `name` that `manager`'s hosted manifest offers and
    /// install it, replacing any model of its name, once its card matches the
    /// manifest and its graphs match its card. It appears whole or not at all.
    ///
    /// Its files are verified as they download and kept in `manager`'s cache,
    /// so installing it again needs no network. The store links them from
    /// there where the file system allows, so a model takes its space once.
    /// `report` hears the download's progress.
    pub async fn download<F>(
        &self,
        manager: &seiza_download::ModelManager,
        name: &str,
        report: F,
    ) -> Result<ModelBundle, ModelError>
    where
        F: Fn(seiza_download::DownloadEvent) + Send + Sync,
    {
        let files = manager.ensure_with(name, report).await?;
        let store = self.clone();
        // Hashing the graphs takes a second or two: keep it off the runtime.
        tokio::task::spawn_blocking(move || store.install_downloaded(&files))
            .await
            .map_err(|error| seiza_download::Error::BackgroundTask(error.to_string()))?
    }

    fn install_downloaded(
        &self,
        files: &seiza_download::ModelFiles,
    ) -> Result<ModelBundle, ModelError> {
        let card = files.artifact(seiza_download::MODEL_CARD_FILE)?;
        self.install_staged(&files.name, |staging| {
            for artifact in files.artifacts() {
                let to = staging.join(&artifact.name);
                // The cache's objects are never rewritten in place, so a link
                // to one stays the bytes that were verified.
                if std::fs::hard_link(&artifact.path, &to).is_err() {
                    std::fs::copy(&artifact.path, &to).map_err(ModelError::io(&artifact.path))?;
                }
            }
            let bundle = ModelBundle::open(staging)?;
            if bundle.card().name != files.name {
                return Err(ModelError::ModelContract(format!(
                    "the manifest offers {}, but its card names {}",
                    files.name,
                    bundle.card().name
                )));
            }
            // A cached file is checked by its size alone, so the card is
            // checked against the manifest here, and the graphs against it.
            if bundle.card_sha256() != card.sha256 {
                return Err(ModelError::HashMismatch {
                    file: card.path.clone(),
                    expected: card.sha256.clone(),
                    found: bundle.card_sha256().to_string(),
                });
            }
            bundle.verify()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundle::tests::write_bundle;

    #[test]
    fn a_model_installs_lists_and_goes() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        write_bundle(&source, "noise-test", "constant");
        let store = ModelStore::new(temp.path().join("store"));
        assert!(
            store.list().unwrap().is_empty(),
            "a store that does not exist yet is empty"
        );

        let installed = store.install(&source).unwrap();
        assert_eq!(installed.root(), store.root().join("noise-test"));
        installed.verify().unwrap();
        assert_eq!(store.list().unwrap().len(), 1);
        store.find("noise-test").unwrap();
        store.install(&source).unwrap();
        assert_eq!(
            store.list().unwrap().len(),
            1,
            "installing again replaces it"
        );

        store.remove("noise-test").unwrap();
        assert!(matches!(
            store.find("noise-test"),
            Err(ModelError::ModelNotFound(_))
        ));
        assert!(matches!(
            store.remove("noise-test"),
            Err(ModelError::ModelNotFound(_))
        ));
    }

    #[test]
    fn a_model_resolves_by_directory_by_name_or_alone() {
        let temp = tempfile::tempdir().unwrap();
        let (first, second) = (temp.path().join("first"), temp.path().join("second"));
        write_bundle(&first, "noise-first", "constant");
        write_bundle(&second, "noise-second", "image");
        let store = ModelStore::new(temp.path().join("store"));
        assert!(matches!(
            store.resolve(None),
            Err(ModelError::NoModelInstalled { .. })
        ));

        store.install(&first).unwrap();
        assert_eq!(store.resolve(None).unwrap().card().name, "noise-first");
        let by_path = store.resolve(Some(second.to_str().unwrap())).unwrap();
        assert_eq!(by_path.card().name, "noise-second");

        store.install(&second).unwrap();
        assert!(matches!(
            store.resolve(None),
            Err(ModelError::SeveralModels { .. })
        ));
        assert_eq!(
            store.resolve(Some("noise-first")).unwrap().card().name,
            "noise-first"
        );
        assert!(matches!(
            store.resolve(Some("nothing")),
            Err(ModelError::ModelNotFound(_))
        ));
    }

    #[test]
    fn a_bundle_that_does_not_match_its_card_never_installs() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        write_bundle(&source, "noise-test", "constant");
        std::fs::write(source.join("model-f32.onnx"), b"changed").unwrap();
        let store = ModelStore::new(temp.path().join("store"));
        assert!(matches!(
            store.install(&source),
            Err(ModelError::HashMismatch { .. })
        ));
        assert!(store.list().unwrap().is_empty());
    }
}

#[cfg(all(test, feature = "download"))]
mod download_tests {
    use super::*;
    use seiza_download::bundle::sha256_hex;
    use seiza_download::{CachePolicy, HostedModel, ManifestFile, ModelManager, ModelManifest};
    use std::io::{Read, Write};

    fn tiny() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny")
    }

    /// Serve a manifest offering the tiny fixture model as `name`, and its
    /// files, for `requests` requests.
    fn serve(name: &str, requests: usize) -> (String, std::thread::JoinHandle<()>) {
        let mut routes = Vec::new();
        let mut files = Vec::new();
        for file in ["card.json", "model-f32.onnx", "model-f16.onnx"] {
            let bytes = std::fs::read(tiny().join(file)).unwrap();
            let sha256 = sha256_hex(&bytes);
            let key = format!("artifacts/{sha256}/{file}");
            files.push(ManifestFile {
                name: file.to_string(),
                key: Some(key.clone()),
                bytes: bytes.len() as u64,
                sha256,
            });
            routes.push((key, bytes));
        }
        let manifest = ModelManifest {
            version: "model-bundle-v1-test".to_string(),
            models: vec![HostedModel {
                name: name.to_string(),
                files,
                extras: Vec::new(),
                transports: Vec::new(),
            }],
        };
        routes.push((
            "manifest.json".to_string(),
            serde_json::to_vec(&manifest).unwrap(),
        ));
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            for _ in 0..requests {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = [0u8; 4096];
                let read = stream.read(&mut request).unwrap();
                let request = String::from_utf8_lossy(&request[..read]);
                let (_, body) = routes
                    .iter()
                    .find(|(path, _)| request.starts_with(&format!("GET /{path} ")))
                    .unwrap_or_else(|| panic!("unexpected request: {request}"));
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .unwrap();
                stream.write_all(body).unwrap();
            }
        });
        (format!("http://{address}"), server)
    }

    fn manager(cache: &Path, url: &str, policy: CachePolicy) -> ModelManager {
        ModelManager::builder()
            .cache_dir(cache)
            .base_url(url)
            .policy(policy)
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn a_hosted_model_downloads_and_installs_whole() {
        let temp = tempfile::tempdir().unwrap();
        let cache = temp.path().join("cache");
        let (url, server) = serve("noise-tiny", 4);
        let store = ModelStore::new(temp.path().join("store"));
        let installed = store
            .download(
                &manager(&cache, &url, CachePolicy::ForceRefresh),
                "noise-tiny",
                |_| {},
            )
            .await
            .unwrap();
        server.join().unwrap();
        assert_eq!(installed.root(), store.root().join("noise-tiny"));
        installed.verify().unwrap();
        assert_eq!(
            installed.card_sha256(),
            ModelBundle::open(tiny()).unwrap().card_sha256(),
            "the card installs as it was published"
        );

        // The cache holds it now: no server is needed to install it again.
        let offline = manager(&cache, &url, CachePolicy::OfflineOnly);
        store.remove("noise-tiny").unwrap();
        let again = store
            .download(&offline, "noise-tiny", |_| {})
            .await
            .unwrap();
        again.verify().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let cached = offline.ensure("noise-tiny").await.unwrap();
            let graph = cached.artifact("model-f32.onnx").unwrap();
            let inode = |path: &Path| std::fs::metadata(path).unwrap().ino();
            assert_eq!(
                inode(&graph.path),
                inode(&again.root().join("model-f32.onnx")),
                "the store links the cache's file rather than copying it"
            );
        }

        assert!(matches!(
            store.download(&offline, "noise-other", |_| {}).await,
            Err(ModelError::Download(
                seiza_download::Error::ModelNotOffered { .. }
            ))
        ));
        assert_eq!(store.list().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn a_hosted_model_whose_card_names_another_never_installs() {
        let temp = tempfile::tempdir().unwrap();
        let (url, server) = serve("noise-other", 4);
        let store = ModelStore::new(temp.path().join("store"));
        let manager = manager(&temp.path().join("cache"), &url, CachePolicy::ForceRefresh);
        let result = store.download(&manager, "noise-other", |_| {}).await;
        server.join().unwrap();
        assert!(
            matches!(&result, Err(ModelError::ModelContract(reason)) if reason.contains("noise-tiny")),
            "{result:?}"
        );
        assert!(store.list().unwrap().is_empty());
        assert_eq!(
            std::fs::read_dir(store.root()).unwrap().count(),
            0,
            "no staging directory is left behind"
        );
    }
}
