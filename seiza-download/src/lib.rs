//! Async installation and caching for published Seiza catalog bundles and
//! learned models.
//!
//! This crate deliberately stops at verified local files. It never opens a
//! catalog or a model and does not depend on `seiza`; applications can await
//! downloads here and pass the returned paths to the memory-mapped readers in
//! `seiza`, or a model's to `seiza-model`.

pub mod bundle;
mod catalog;
mod error;
mod hosted;
mod manifest;
mod models;

pub use catalog::{
    CachePolicy, CatalogArtifact, CatalogBundle, CatalogManager, CatalogManagerBuilder,
    DownloadEvent,
};
pub use error::{Error, Result};
pub use manifest::{
    BundleManifest, BundleManifestDocument, CatalogSet, Dataset, ManifestFile, ManifestTransport,
    REQUIRED_BUNDLE_FILES, REQUIRED_V2_FILES,
};
pub use models::{
    HostedModel, MODEL_CARD_FILE, ModelFiles, ModelManager, ModelManagerBuilder, ModelManifest,
};

/// Current complete hosted catalog bundle.
///
/// `/data` and `/data/v2` remain frozen compatibility surfaces for v1 and the
/// complete v2 bundle. The historical `/data/v3` probe URL remains reserved
/// and may be absent. New clients never replace artifacts under those URLs.
pub const DEFAULT_BUNDLE_BASE_URL: &str = "https://downloads.seiza.fyi/data/v4";

/// Frozen complete v2 bundle for applications that explicitly need the
/// previous `SEIZAOB3` object artifact.
pub const LEGACY_V2_BUNDLE_BASE_URL: &str = "https://downloads.seiza.fyi/data/v2";

/// The hosted model manifest and the models it offers.
pub const DEFAULT_MODEL_BASE_URL: &str = "https://downloads.seiza.fyi/models/v1";
