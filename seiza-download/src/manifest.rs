use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Files that make a coherent hosted v2 catalog bundle.
pub const REQUIRED_V2_FILES: &[&str] = &[
    "blind-gaia16.idx",
    "minor-bodies.bin",
    "objects.bin",
    "stars-deep-gaia17.bin",
    "stars-gaia.bin",
    "stars-lite-tycho2.bin",
    "stars-lite-tycho2.ids.bin",
    "transients.bin",
];

/// A known artifact in the current hosted bundle.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Dataset {
    BlindGaia16,
    MinorBodies,
    Objects,
    StarsDeepGaia17,
    StarsGaia,
    StarsLiteTycho2,
    StarsLiteTycho2Identifiers,
    Transients,
}

impl Dataset {
    pub const fn file_name(self) -> &'static str {
        match self {
            Self::BlindGaia16 => "blind-gaia16.idx",
            Self::MinorBodies => "minor-bodies.bin",
            Self::Objects => "objects.bin",
            Self::StarsDeepGaia17 => "stars-deep-gaia17.bin",
            Self::StarsGaia => "stars-gaia.bin",
            Self::StarsLiteTycho2 => "stars-lite-tycho2.bin",
            Self::StarsLiteTycho2Identifiers => "stars-lite-tycho2.ids.bin",
            Self::Transients => "transients.bin",
        }
    }
}

/// A requested group of catalog artifacts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CatalogSet {
    names: Option<BTreeSet<String>>,
}

impl CatalogSet {
    /// Every artifact in the coherent bundle. Large; applications should
    /// normally request only the datasets they use.
    pub fn all() -> Self {
        Self { names: None }
    }

    pub fn empty() -> Self {
        Self {
            names: Some(BTreeSet::new()),
        }
    }

    pub fn dataset(dataset: Dataset) -> Self {
        Self::empty().with(dataset)
    }

    pub fn solver_lite() -> Self {
        Self::dataset(Dataset::StarsLiteTycho2)
    }

    pub fn solver_gaia() -> Self {
        Self::dataset(Dataset::StarsGaia)
    }

    pub fn blind_deep() -> Self {
        Self::dataset(Dataset::StarsDeepGaia17).with(Dataset::BlindGaia16)
    }

    /// Build a selection from hosted filenames. An empty iterator retains the
    /// CLI's existing meaning of the complete bundle.
    pub fn from_names<I, S>(names: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let names = names.into_iter().map(Into::into).collect::<BTreeSet<_>>();
        if names.is_empty() {
            return Ok(Self::all());
        }
        for name in &names {
            validate_file_name(name)?;
        }
        Ok(Self { names: Some(names) })
    }

    pub fn with(mut self, dataset: Dataset) -> Self {
        if let Some(names) = &mut self.names {
            names.insert(dataset.file_name().to_string());
        }
        self
    }

    pub fn is_all(&self) -> bool {
        self.names.is_none()
    }

    pub(crate) fn contains(&self, name: &str) -> bool {
        self.names.as_ref().is_none_or(|names| names.contains(name))
    }

    pub(crate) fn requested_names(&self) -> Option<&BTreeSet<String>> {
        self.names.as_ref()
    }
}

impl Default for CatalogSet {
    fn default() -> Self {
        Self::solver_lite()
    }
}

/// One file entry from the hosted manifest.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ManifestFile {
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
}

/// A complete, coherent set of hosted catalog artifacts.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BundleManifest {
    pub version: String,
    pub files: Vec<ManifestFile>,
}

impl BundleManifest {
    pub fn parse(json: &[u8]) -> Result<Self> {
        let manifest: Self = serde_json::from_slice(json)?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<()> {
        if !self.version.starts_with("catalog-bundle-v2-") {
            return Err(Error::Manifest(format!(
                "unsupported version {}",
                self.version
            )));
        }

        let mut offered = BTreeSet::new();
        for file in &self.files {
            validate_file_name(&file.name)?;
            if file.bytes == 0 {
                return Err(Error::Manifest(format!("{} is empty", file.name)));
            }
            if file.sha256.len() != 64
                || !file
                    .sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Err(Error::Manifest(format!(
                    "{} has an invalid lowercase SHA-256",
                    file.name
                )));
            }
            if !offered.insert(file.name.as_str()) {
                return Err(Error::Manifest(format!("duplicate file: {}", file.name)));
            }
        }

        let missing = REQUIRED_V2_FILES
            .iter()
            .filter(|name| !offered.contains(**name))
            .copied()
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(Error::Manifest(format!(
                "catalog bundle v2 is incomplete; missing: {}",
                missing.join(", ")
            )));
        }
        Ok(())
    }

    pub fn plan(&self, set: &CatalogSet) -> Result<Vec<ManifestFile>> {
        self.validate()?;
        let offered = self
            .files
            .iter()
            .map(|file| file.name.as_str())
            .collect::<BTreeSet<_>>();
        if let Some(requested) = set.requested_names() {
            let missing = requested
                .iter()
                .filter(|name| !offered.contains(name.as_str()))
                .cloned()
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                return Err(Error::Manifest(format!(
                    "requested file(s) unavailable: {}; the manifest offers: {}",
                    missing.join(", "),
                    offered.into_iter().collect::<Vec<_>>().join(", ")
                )));
            }
        }

        let plan = self
            .files
            .iter()
            .filter(|file| set.contains(&file.name))
            .cloned()
            .collect::<Vec<_>>();
        if plan.is_empty() {
            return Err(Error::Manifest("nothing matched the requested set".into()));
        }
        Ok(plan)
    }
}

fn validate_file_name(name: &str) -> Result<()> {
    if name.is_empty() || name == "." || name == ".." || name.contains('/') || name.contains('\\') {
        return Err(Error::Manifest(format!(
            "unsafe artifact filename: {name:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn manifest() -> BundleManifest {
        BundleManifest {
            version: "catalog-bundle-v2-test".into(),
            files: REQUIRED_V2_FILES
                .iter()
                .enumerate()
                .map(|(index, name)| ManifestFile {
                    name: (*name).into(),
                    bytes: index as u64 + 1,
                    sha256: hash(char::from(b"abcdef"[index % 6])),
                })
                .collect(),
        }
    }

    #[test]
    fn complete_manifest_and_explicit_selection_are_accepted() {
        let manifest = manifest();
        manifest.validate().unwrap();
        let set = CatalogSet::from_names(["objects.bin", "transients.bin"]).unwrap();
        let plan = manifest.plan(&set).unwrap();
        assert_eq!(
            plan.iter()
                .map(|file| file.name.as_str())
                .collect::<Vec<_>>(),
            ["objects.bin", "transients.bin"]
        );
    }

    #[test]
    fn incomplete_bundle_is_rejected_for_a_single_file_request() {
        let mut manifest = manifest();
        manifest
            .files
            .retain(|file| file.name != "stars-lite-tycho2.ids.bin");
        assert!(
            manifest
                .validate()
                .unwrap_err()
                .to_string()
                .contains("stars-lite-tycho2.ids.bin")
        );
    }

    #[test]
    fn traversal_and_unknown_names_are_rejected() {
        assert!(CatalogSet::from_names(["../objects.bin"]).is_err());
        let error = manifest()
            .plan(&CatalogSet::from_names(["unknown.bin"]).unwrap())
            .unwrap_err();
        assert!(error.to_string().contains("unknown.bin"));
    }

    #[test]
    fn default_set_is_small() {
        let set = CatalogSet::default();
        assert!(!set.is_all());
        assert!(set.contains(Dataset::StarsLiteTycho2.file_name()));
        assert!(!set.contains(Dataset::Objects.file_name()));
    }
}
