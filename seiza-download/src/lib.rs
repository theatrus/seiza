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
pub use manifest::{BundleManifest, CatalogSet, Dataset, ManifestFile, REQUIRED_V2_FILES};

/// Current complete hosted catalog bundle.
///
/// The unversioned `/data` path remains a classic-v1 compatibility surface and
/// is intentionally not consulted by this crate.
pub const DEFAULT_BUNDLE_BASE_URL: &str = "https://downloads.seiza.fyi/data/v2";
