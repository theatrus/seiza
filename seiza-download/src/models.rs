//! Hosted learned models: a manifest of named model bundles, published and
//! cached the way catalog bundles are.
//!
//! A hosted model is a card and the graphs it names, with content-addressed
//! keys and optional zstd transports, as a catalog bundle's files are. The
//! manager stops at verified files in its cache: `seiza-model` reads the
//! card and installs the model.

use crate::catalog::CatalogArtifact;
use crate::hosted::{Hosted, HostedManifest, scoped_manifest_file};
use crate::manifest::{
    preferred_transport, validate_content_key, validate_file_name, validate_sha256,
    validate_transports,
};
use crate::{
    CachePolicy, DEFAULT_MODEL_BASE_URL, DownloadEvent, Error, ManifestFile, ManifestTransport,
    Result,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The version every model manifest this build reads starts with.
const MODEL_MANIFEST_GENERATION: &str = "model-bundle-v1-";

/// The file every hosted model has: its card, which names the rest.
pub const MODEL_CARD_FILE: &str = "card.json";

/// One model the manifest offers.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct HostedModel {
    /// The model's name: its card's, and its directory once installed.
    pub name: String,
    /// What installing it fetches: [`MODEL_CARD_FILE`] and the graphs the
    /// card names.
    pub files: Vec<ManifestFile>,
    /// Files hosted with the model that installing it does not fetch, such
    /// as the training weights its graphs were exported from.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extras: Vec<ManifestFile>,
    /// Alternate transfer encodings of `files` and `extras`. The installed
    /// file is always the canonical one its entry describes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transports: Vec<ManifestTransport>,
}

impl HostedModel {
    /// The entry for the file `name`, from `files` or `extras`.
    pub fn file(&self, name: &str) -> Option<&ManifestFile> {
        self.files
            .iter()
            .chain(&self.extras)
            .find(|file| file.name == name)
    }

    /// Best transport for the file `name` that this version of the
    /// downloader understands.
    pub fn preferred_transport(&self, name: &str) -> Option<&ManifestTransport> {
        preferred_transport(&self.transports, name)
    }

    fn validate(&self) -> Result<()> {
        validate_file_name(&self.name)?;
        let mut offered = BTreeSet::new();
        for file in self.files.iter().chain(&self.extras) {
            validate_file_name(&file.name)?;
            if file.bytes == 0 {
                return Err(Error::Manifest(format!(
                    "model {}'s {} is empty",
                    self.name, file.name
                )));
            }
            validate_sha256(&file.sha256, &file.name)?;
            validate_content_key(file, "model artifact")?;
            if !offered.insert(file.name.as_str()) {
                return Err(Error::Manifest(format!(
                    "model {} lists {} twice",
                    self.name, file.name
                )));
            }
        }
        if !self.files.iter().any(|file| file.name == MODEL_CARD_FILE) {
            return Err(Error::Manifest(format!(
                "model {} has no {MODEL_CARD_FILE}",
                self.name
            )));
        }
        validate_transports(&self.transports, &offered)
    }
}

/// The hosted list of models, each with the files that install it.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ModelManifest {
    /// `model-bundle-v1-` and the day it was published.
    pub version: String,
    /// The models offered.
    pub models: Vec<HostedModel>,
}

impl ModelManifest {
    /// Read a manifest from its JSON, refusing one this build cannot honour.
    pub fn parse(json: &[u8]) -> Result<Self> {
        let manifest: Self = serde_json::from_slice(json)?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Refuse a manifest of another generation, a model named twice, or a
    /// model whose files are not content-addressed.
    pub fn validate(&self) -> Result<()> {
        if !self.version.starts_with(MODEL_MANIFEST_GENERATION) {
            return Err(Error::Manifest(format!(
                "unsupported version {}; expected {MODEL_MANIFEST_GENERATION}*",
                self.version
            )));
        }
        let mut names = BTreeSet::new();
        for model in &self.models {
            model.validate()?;
            if !names.insert(model.name.as_str()) {
                return Err(Error::Manifest(format!(
                    "model {} is listed twice",
                    model.name
                )));
            }
        }
        Ok(())
    }

    /// The model named `name`.
    pub fn model(&self, name: &str) -> Result<&HostedModel> {
        self.models
            .iter()
            .find(|model| model.name == name)
            .ok_or_else(|| Error::ModelNotOffered {
                name: name.to_string(),
                offered: self.models.iter().map(|model| model.name.clone()).collect(),
            })
    }
}

impl HostedManifest for ModelManifest {
    fn parse(json: &[u8]) -> Result<Self> {
        ModelManifest::parse(json)
    }

    fn version(&self) -> &str {
        &self.version
    }
}

/// A hosted model's files, verified in the local cache.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelFiles {
    /// The model's name.
    pub name: String,
    /// The version of the manifest that offered it.
    pub version: String,
    artifacts: Vec<CatalogArtifact>,
}

impl ModelFiles {
    /// Every file installing the model takes, the card among them.
    pub fn artifacts(&self) -> impl ExactSizeIterator<Item = &CatalogArtifact> {
        self.artifacts.iter()
    }

    /// The file `name`.
    pub fn artifact(&self, name: &str) -> Result<&CatalogArtifact> {
        self.artifacts
            .iter()
            .find(|artifact| artifact.name == name)
            .ok_or_else(|| Error::MissingArtifact(name.into()))
    }
}

/// Configures a [`ModelManager`] without creating an async runtime.
#[derive(Clone, Debug)]
pub struct ModelManagerBuilder {
    cache_dir: Option<PathBuf>,
    base_url: String,
    policy: CachePolicy,
}

impl Default for ModelManagerBuilder {
    fn default() -> Self {
        Self {
            cache_dir: None,
            base_url: DEFAULT_MODEL_BASE_URL.into(),
            policy: CachePolicy::default(),
        }
    }
}

impl ModelManagerBuilder {
    /// Cache downloads here instead of `SEIZA_CACHE_DIR`, or the platform
    /// cache directory's `models`.
    pub fn cache_dir(mut self, cache_dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(cache_dir.into());
        self
    }

    /// Read the manifest and models from this URL instead of
    /// [`DEFAULT_MODEL_BASE_URL`].
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    /// When the cached manifest may stand in for the server's.
    pub fn policy(mut self, policy: CachePolicy) -> Self {
        self.policy = policy;
        self
    }

    /// The manager.
    pub fn build(self) -> Result<ModelManager> {
        let manifest_file = if self.base_url == DEFAULT_MODEL_BASE_URL {
            "model-bundle-v1.json".to_string()
        } else {
            scoped_manifest_file("model-bundle", &self.base_url)
        };
        Ok(ModelManager {
            hosted: Hosted::new(
                self.cache_dir,
                "models",
                self.base_url,
                self.policy,
                manifest_file,
            )?,
        })
    }
}

/// Async manager for the hosted model manifest and the models it offers.
#[derive(Clone, Debug)]
pub struct ModelManager {
    hosted: Hosted,
}

impl ModelManager {
    /// A builder with the default URL, cache and policy.
    pub fn builder() -> ModelManagerBuilder {
        ModelManagerBuilder::default()
    }

    /// Where downloads are cached.
    pub fn cache_dir(&self) -> &Path {
        &self.hosted.cache_dir
    }

    /// The files that install the model `name`, from the cache, or downloaded
    /// and verified into it. Its extras are not fetched.
    pub async fn ensure(&self, name: &str) -> Result<ModelFiles> {
        self.ensure_with(name, |_| {}).await
    }

    /// Like [`ensure`](Self::ensure), reporting what it does.
    pub async fn ensure_with<F>(&self, name: &str, report: F) -> Result<ModelFiles>
    where
        F: Fn(DownloadEvent) + Send + Sync,
    {
        let manifest: ModelManifest = self.hosted.load_manifest(&report).await?;
        let model = manifest.model(name)?;
        let mut artifacts = Vec::with_capacity(model.files.len());
        for file in &model.files {
            let transport = model.preferred_transport(&file.name);
            let path = self.hosted.ensure_file(file, transport, &report).await?;
            artifacts.push(CatalogArtifact {
                name: file.name.clone(),
                bytes: file.bytes,
                sha256: file.sha256.clone(),
                path,
            });
        }
        Ok(ModelFiles {
            name: model.name.clone(),
            version: manifest.version,
            artifacts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundle;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    fn entry(name: &str, bytes: &[u8]) -> ManifestFile {
        let sha256 = bundle::sha256_hex(bytes);
        ManifestFile {
            name: name.into(),
            key: Some(format!("artifacts/{sha256}/{name}")),
            bytes: bytes.len() as u64,
            sha256,
        }
    }

    fn zstd_transport(name: &str, encoded: &[u8]) -> ManifestTransport {
        let sha256 = bundle::sha256_hex(encoded);
        ManifestTransport {
            name: name.into(),
            encoding: "zstd".into(),
            key: format!("artifacts/{sha256}/{name}.zst"),
            bytes: encoded.len() as u64,
            sha256,
        }
    }

    fn manifest() -> ModelManifest {
        ModelManifest {
            version: "model-bundle-v1-test".into(),
            models: vec![HostedModel {
                name: "noise-test".into(),
                files: vec![entry("card.json", b"{}"), entry("model-f32.onnx", b"graph")],
                extras: vec![entry("weights.pt", b"weights")],
                transports: Vec::new(),
            }],
        }
    }

    #[test]
    fn a_manifest_reads_back_and_ignores_fields_it_does_not_know() {
        let mut value = serde_json::to_value(manifest()).unwrap();
        value["future"] = serde_json::json!(1);
        value["models"][0]["summary"] = serde_json::json!("a later field");
        let parsed = ModelManifest::parse(&serde_json::to_vec(&value).unwrap()).unwrap();
        assert_eq!(parsed, manifest());
        let model = parsed.model("noise-test").unwrap();
        assert_eq!(model.file("weights.pt").unwrap().bytes, 7);
        assert!(model.file("missing").is_none());
    }

    #[test]
    fn a_name_not_offered_is_refused_with_those_that_are() {
        match manifest().model("noise-other") {
            Err(Error::ModelNotOffered { name, offered }) => {
                assert_eq!(name, "noise-other");
                assert_eq!(offered, ["noise-test"]);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_manifest_this_build_cannot_honour_is_refused() {
        let refused = |change: &dyn Fn(&mut ModelManifest), says: &str| {
            let mut manifest = manifest();
            change(&mut manifest);
            let error = manifest.validate().unwrap_err().to_string();
            assert!(error.contains(says), "{says}: {error}");
        };
        refused(
            &|m| m.version = "catalog-bundle-v4-test".into(),
            "unsupported version",
        );
        refused(
            &|m| {
                m.models[0]
                    .files
                    .retain(|file| file.name != MODEL_CARD_FILE)
            },
            "has no card.json",
        );
        refused(
            &|m| m.models[0].files[1].key = Some("model-f32.onnx".into()),
            "immutable key",
        );
        refused(
            &|m| m.models[0].extras.push(entry("model-f32.onnx", b"graph")),
            "twice",
        );
        refused(&|m| m.models.push(m.models[0].clone()), "listed twice");
        refused(&|m| m.models[0].name = "../outside".into(), "unsafe");
        refused(
            &|m| {
                m.models[0]
                    .transports
                    .push(zstd_transport("other.onnx", b"x"))
            },
            "unknown file",
        );
        let mut transport = zstd_transport("weights.pt", b"x");
        transport.key = "weights.pt.zst".into();
        refused(
            &|m| m.models[0].transports.push(transport.clone()),
            "immutable key",
        );
    }

    /// Serve `routes`, path to body, for `requests` requests, refusing any
    /// other path.
    fn serve(
        routes: Vec<(String, Vec<u8>)>,
        requests: usize,
    ) -> (String, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            for _ in 0..requests {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = [0u8; 4096];
                let read = stream.read(&mut request).unwrap();
                let request = String::from_utf8_lossy(&request[..read]);
                let body = routes
                    .iter()
                    .find(|(path, _)| request.starts_with(&format!("GET /{path} ")))
                    .map(|(_, body)| body.as_slice())
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

    #[tokio::test]
    async fn a_model_s_files_are_fetched_through_zstd_and_its_extras_are_not() {
        let card = b"{\"name\": \"noise-test\"}".to_vec();
        let graph = b"graph bytes ".repeat(4096);
        let encoded = zstd::stream::encode_all(graph.as_slice(), 19).unwrap();
        let mut manifest = manifest();
        let model = &mut manifest.models[0];
        model.files = vec![entry("card.json", &card), entry("model-f32.onnx", &graph)];
        model
            .transports
            .push(zstd_transport("model-f32.onnx", &encoded));
        let routes = vec![
            (model.files[0].artifact_key().to_string(), card.clone()),
            (model.transports[0].key.clone(), encoded),
            (
                "manifest.json".to_string(),
                serde_json::to_vec(&manifest).unwrap(),
            ),
        ];
        let (url, server) = serve(routes, 3);

        let cache = tempfile::tempdir().unwrap();
        let manager = ModelManager::builder()
            .cache_dir(cache.path())
            .base_url(&url)
            .policy(CachePolicy::ForceRefresh)
            .build()
            .unwrap();
        let files = manager.ensure("noise-test").await.unwrap();
        server.join().unwrap();
        assert_eq!(files.name, "noise-test");
        assert_eq!(files.version, "model-bundle-v1-test");
        assert_eq!(files.artifacts().len(), 2, "the weights are an extra");
        let fetched = files.artifact("model-f32.onnx").unwrap();
        assert_eq!(std::fs::read(&fetched.path).unwrap(), graph);
        assert_eq!(
            std::fs::read(&files.artifact("card.json").unwrap().path).unwrap(),
            card
        );
        assert!(matches!(
            files.artifact("weights.pt"),
            Err(Error::MissingArtifact(_))
        ));

        // A second call needs no server: the manifest and the files are cached.
        let offline = ModelManager::builder()
            .cache_dir(cache.path())
            .base_url("http://127.0.0.1:1/unreachable")
            .policy(CachePolicy::OfflineOnly)
            .build()
            .unwrap();
        assert!(
            offline.ensure("noise-test").await.is_err(),
            "a manifest is cached per URL"
        );
        let again = ModelManager::builder()
            .cache_dir(cache.path())
            .base_url(&url)
            .policy(CachePolicy::OfflineOnly)
            .build()
            .unwrap();
        assert_eq!(again.ensure("noise-test").await.unwrap(), files);
    }
}
