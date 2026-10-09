//! seiza (星座) — star detection, WCS fitting, and near-field plate solving.
//!
//! The intended pipeline:
//! 1. [`detect`] finds stars (x, y, flux) in a decoded image.
//! 2. [`catalog`] provides reference stars around a hinted sky position.
//! 3. [`solve`] matches detected stars to catalog stars and fits a [`wcs::Wcs`].
//!
//! Solving is *seeded*: it expects an approximate center (RA/Dec hint) and an
//! approximate pixel scale. Blind solving is out of scope for now.
//!
//! # Threads
//!
//! Detection, solving and building a blind index split their work across
//! the Rayon pool of the calling thread, or the global pool when that thread
//! belongs to none. To hold them to a number of cores, call them inside a
//! pool that size with `pool.install(..)`. The crate starts no threads of its
//! own. A blind solve checks as many candidate fields at once as the pool
//! has threads, so pools of different sizes can settle on different, equally
//! valid solutions.

pub mod blind;
pub mod catalog;
pub mod constellations;
pub mod data_paths;
pub mod detect;
pub mod minor_bodies;
mod object_catalog_v3;
mod object_catalog_v4;
pub mod objects;
pub mod raster;
pub mod solve;
pub mod star_ids;
pub mod wcs;

pub use detect::{DetectBackend, DetectConfig, DetectedStar, detect_stars, detect_stars_luma_f32};
pub use wcs::{FitsCardValue, Sip, Wcs};

/// Host-neutral calibration response kernels.
pub use seiza_calibration as calibration;

/// Async installation and caching of published catalog bundles.
///
/// Available with the non-default `downloads` feature. Catalog opening itself
/// remains synchronous, local, and network-free.
#[cfg(feature = "downloads")]
pub use seiza_download as downloads;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("catalog error: {0}")]
    Catalog(String),
    #[error("solve failed: {0}")]
    Solve(String),
}
