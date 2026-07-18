use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Files that make a coherent hosted catalog bundle.
pub const REQUIRED_BUNDLE_FILES: &[&str] = &[
    "blind-gaia16.idx",
    "minor-bodies.bin",
    "objects.bin",
    "stars-deep-gaia17.bin",
    "stars-gaia.bin",
    "stars-lite-tycho2.bin",
    "stars-lite-tycho2.ids.bin",
    "transients.bin",
];

/// Files required by the previous v2 bundle.
///
/// V2 and v4 intentionally contain the same logical datasets. New code should
/// use [`REQUIRED_BUNDLE_FILES`]; this alias remains so existing library users
/// do not need to change merely to read a legacy manifest.
pub const REQUIRED_V2_FILES: &[&str] = REQUIRED_BUNDLE_FILES;

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

/// One alternate transport for a hosted file.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ManifestTransport {
    /// Logical filename from [`BundleManifest::files`].
    pub name: String,
    /// Transport encoding understood by a downloader, currently `zstd`.
    /// Readers skip encodings they do not support.
    pub encoding: String,
    /// Immutable S3 key relative to the bundle base URL.
    pub key: String,
    /// Encoded transfer size in bytes.
    pub bytes: u64,
    /// SHA-256 of the encoded bytes.
    pub sha256: String,
}

/// One file entry from the hosted manifest.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ManifestFile {
    pub name: String,
    /// Immutable S3 key relative to the bundle base URL. Required by v4.
    /// Legacy v2 manifests omit it and resolve artifacts directly by `name`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    pub bytes: u64,
    pub sha256: String,
}

impl ManifestFile {
    /// Relative URL used to download this artifact.
    pub fn artifact_key(&self) -> &str {
        self.key.as_deref().unwrap_or(&self.name)
    }
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
        let generation = if self.version.starts_with("catalog-bundle-v2-") {
            2
        } else if self.version.starts_with("catalog-bundle-v4-") {
            4
        } else {
            return Err(Error::Manifest(format!(
                "unsupported version {}; expected catalog-bundle-v2-* or catalog-bundle-v4-*",
                self.version
            )));
        };

        let mut offered = BTreeSet::new();
        for file in &self.files {
            validate_file_name(&file.name)?;
            if file.bytes == 0 {
                return Err(Error::Manifest(format!("{} is empty", file.name)));
            }
            validate_sha256(&file.sha256, &file.name)?;
            match generation {
                2 if file.key.is_some() => {
                    return Err(Error::Manifest(format!(
                        "legacy v2 artifact {} must not define a key",
                        file.name
                    )));
                }
                4 => {
                    let expected = format!("artifacts/{}/{}", file.sha256, file.name);
                    if file.key.as_deref() != Some(expected.as_str()) {
                        return Err(Error::Manifest(format!(
                            "v4 artifact {} must use immutable key {}",
                            file.name, expected
                        )));
                    }
                }
                _ => {}
            }
            if !offered.insert(file.name.as_str()) {
                return Err(Error::Manifest(format!("duplicate file: {}", file.name)));
            }
        }

        let missing = REQUIRED_BUNDLE_FILES
            .iter()
            .filter(|name| !offered.contains(**name))
            .copied()
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(Error::Manifest(format!(
                "catalog bundle v{generation} is incomplete; missing: {}",
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

/// The extensible hosted JSON document. `manifest` remains the stable logical
/// bundle read by older clients; `transports` is an optional additive section.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BundleManifestDocument {
    #[serde(flatten)]
    pub manifest: BundleManifest,
    /// Alternate transfer representations. The installed/cache file is always
    /// the canonical uncompressed entry in `manifest.files`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transports: Vec<ManifestTransport>,
}

impl BundleManifestDocument {
    pub fn parse(json: &[u8]) -> Result<Self> {
        let document: Self = serde_json::from_slice(json)?;
        document.validate()?;
        Ok(document)
    }

    pub fn validate(&self) -> Result<()> {
        self.manifest.validate()?;
        if self.manifest.version.starts_with("catalog-bundle-v2-") && !self.transports.is_empty() {
            return Err(Error::Manifest(
                "legacy v2 manifests must not define alternate transports".into(),
            ));
        }

        let offered = self
            .manifest
            .files
            .iter()
            .map(|file| file.name.as_str())
            .collect::<BTreeSet<_>>();
        let mut transports = BTreeSet::new();
        for transport in &self.transports {
            validate_file_name(&transport.name)?;
            if !offered.contains(transport.name.as_str()) {
                return Err(Error::Manifest(format!(
                    "alternate transport references unknown file {}",
                    transport.name
                )));
            }
            if transport.encoding.is_empty()
                || !transport
                    .encoding
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
            {
                return Err(Error::Manifest(format!(
                    "{} has an invalid transport encoding {:?}",
                    transport.name, transport.encoding
                )));
            }
            if !transports.insert((transport.name.as_str(), transport.encoding.as_str())) {
                return Err(Error::Manifest(format!(
                    "{} has duplicate {} transports",
                    transport.name, transport.encoding
                )));
            }
            if transport.bytes == 0 {
                return Err(Error::Manifest(format!(
                    "{} {} transport is empty",
                    transport.name, transport.encoding
                )));
            }
            validate_sha256(
                &transport.sha256,
                &format!("{} {} transport", transport.name, transport.encoding),
            )?;
            validate_transport_key(transport)?;
        }
        Ok(())
    }

    /// Best transport this version of the downloader understands.
    pub fn preferred_transport(&self, name: &str) -> Option<&ManifestTransport> {
        self.transports
            .iter()
            .find(|transport| transport.name == name && transport.encoding == "zstd")
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

fn validate_sha256(sha256: &str, label: &str) -> Result<()> {
    if sha256.len() != 64
        || !sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(Error::Manifest(format!(
            "{label} has an invalid lowercase SHA-256"
        )));
    }
    Ok(())
}

fn validate_transport_key(transport: &ManifestTransport) -> Result<()> {
    let prefix = format!("artifacts/{}/", transport.sha256);
    let Some(leaf) = transport.key.strip_prefix(&prefix) else {
        return Err(Error::Manifest(format!(
            "{} {} transport must use an immutable key below {prefix}",
            transport.name, transport.encoding
        )));
    };
    validate_file_name(leaf)?;
    if transport.encoding == "zstd" {
        let expected = format!("{}.zst", transport.name);
        if leaf != expected {
            return Err(Error::Manifest(format!(
                "{} zstd transport must use filename {expected}",
                transport.name
            )));
        }
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
            version: "catalog-bundle-v4-test".into(),
            files: REQUIRED_BUNDLE_FILES
                .iter()
                .enumerate()
                .map(|(index, &name)| {
                    let sha256 = hash(char::from(b"abcdef"[index % 6]));
                    ManifestFile {
                        name: name.into(),
                        key: Some(format!("artifacts/{sha256}/{name}")),
                        bytes: index as u64 + 1,
                        sha256,
                    }
                })
                .collect(),
        }
    }

    fn legacy_v2_manifest() -> BundleManifest {
        let mut manifest = manifest();
        manifest.version = "catalog-bundle-v2-test".into();
        for file in &mut manifest.files {
            file.key = None;
        }
        manifest
    }

    fn transport(name: &str, kind: &str, suffix: &str) -> ManifestTransport {
        let sha256 = hash('1');
        ManifestTransport {
            name: name.into(),
            encoding: kind.into(),
            key: format!("artifacts/{sha256}/{name}.{suffix}"),
            bytes: 7,
            sha256,
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
    fn legacy_v2_manifest_without_keys_remains_accepted() {
        let manifest = legacy_v2_manifest();
        manifest.validate().unwrap();
        assert_eq!(manifest.files[0].artifact_key(), manifest.files[0].name);
    }

    #[test]
    fn v4_requires_content_addressed_artifact_keys() {
        let mut missing = manifest();
        missing.files[0].key = None;
        assert!(
            missing
                .validate()
                .unwrap_err()
                .to_string()
                .contains("immutable key")
        );

        let mut mismatched = manifest();
        mismatched.files[0].key = Some("artifacts/not-the-hash/objects.bin".into());
        assert!(
            mismatched
                .validate()
                .unwrap_err()
                .to_string()
                .contains("immutable key")
        );
    }

    #[test]
    fn v4_accepts_content_addressed_zstd_and_unknown_transports() {
        let manifest = manifest();
        let name = manifest.files[0].name.clone();
        let mut document = BundleManifestDocument {
            manifest,
            transports: vec![
                transport(&name, "zstd", "zst"),
                transport(&name, "future-codec", "future"),
            ],
        };
        document.validate().unwrap();
        assert_eq!(
            document.preferred_transport(&name).unwrap().encoding,
            "zstd"
        );

        document.transports[0].key = "objects.bin.zst".into();
        assert!(
            document
                .validate()
                .unwrap_err()
                .to_string()
                .contains("immutable key")
        );
    }

    #[test]
    fn unknown_json_fields_are_ignored() {
        let manifest = manifest();
        let name = manifest.files[0].name.clone();
        let document = BundleManifestDocument {
            manifest,
            transports: vec![transport(&name, "zstd", "zst")],
        };
        let mut value = serde_json::to_value(document).unwrap();
        value["future_manifest_field"] = serde_json::json!({ "format": 5 });
        value["files"][0]["future_file_field"] = serde_json::json!(true);
        value["transports"][0]["future_transport_field"] = serde_json::json!("ignored");
        let json = serde_json::to_vec(&value).unwrap();
        BundleManifestDocument::parse(&json).unwrap();
        // Models a released reader which knows only the stable logical fields.
        BundleManifest::parse(&json).unwrap();
    }

    #[test]
    fn legacy_v2_rejects_alternate_transports() {
        let manifest = legacy_v2_manifest();
        let name = manifest.files[0].name.clone();
        let document = BundleManifestDocument {
            manifest,
            transports: vec![transport(&name, "zstd", "zst")],
        };
        assert!(
            document
                .validate()
                .unwrap_err()
                .to_string()
                .contains("alternate transports")
        );
    }

    #[test]
    fn historical_v3_bundle_version_is_not_repurposed() {
        let mut manifest = manifest();
        manifest.version = "catalog-bundle-v3-test".into();
        assert!(
            manifest
                .validate()
                .unwrap_err()
                .to_string()
                .contains("unsupported version")
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
