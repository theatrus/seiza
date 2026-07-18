//! Async installation and caching for published Seiza catalog bundles.
//!
//! This crate deliberately stops at verified local files. It never opens a
//! catalog and does not depend on `seiza`; applications can await downloads
//! here and pass the returned paths to the memory-mapped readers in `seiza`.

mod error;
mod manager;
mod manifest;

pub use error::{Error, Result};
pub use manager::{
    CachePolicy, CatalogArtifact, CatalogBundle, CatalogManager, CatalogManagerBuilder,
    DownloadEvent,
};
pub use manifest::{
    BundleManifest, BundleManifestDocument, CatalogSet, Dataset, ManifestFile, ManifestTransport,
    REQUIRED_BUNDLE_FILES, REQUIRED_V2_FILES,
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
