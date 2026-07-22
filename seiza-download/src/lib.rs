//! Async installation and caching for published Seiza catalog bundles.
//!
//! This crate deliberately stops at verified local files. It never opens a
//! catalog and does not depend on `seiza`; applications can await downloads
//! here and pass the returned paths to the memory-mapped readers in `seiza`.

pub mod bundle;
mod catalog;
mod error;
mod manifest;

pub use catalog::{
    CachePolicy, CatalogArtifact, CatalogBundle, CatalogManager, CatalogManagerBuilder,
    DownloadEvent,
};
pub use error::{Error, Result};
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

/// Select AWS-LC for Rustls unless the host process has already selected a
/// provider. Seiza HTTP clients call this before they build a client.
pub fn install_default_crypto_provider() {
    if rustls::crypto::CryptoProvider::get_default().is_none() {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    }
}
