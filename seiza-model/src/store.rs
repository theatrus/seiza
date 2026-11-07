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
        std::fs::create_dir_all(&self.root).map_err(ModelError::io(&self.root))?;
        let name = &bundle.card().name;
        let target = self.root.join(name);
        let staging = self
            .root
            .join(format!(".installing-{name}-{}", std::process::id()));
        let result = (|| {
            std::fs::create_dir_all(&staging).map_err(ModelError::io(&staging))?;
            let files = std::iter::once(CARD_FILE)
                .chain(bundle.card().graphs.iter().map(|graph| graph.file.as_str()));
            for file in files {
                let from = bundle.root().join(file);
                std::fs::copy(&from, staging.join(file)).map_err(ModelError::io(&from))?;
            }
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
