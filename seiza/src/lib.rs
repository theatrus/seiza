//! seiza (星座) — star detection, WCS fitting, and near-field plate solving.
//!
//! The intended pipeline:
//! 1. [`detect`] finds stars (x, y, flux) in a decoded image.
//! 2. [`catalog`] provides reference stars around a hinted sky position.
//! 3. [`solve`] matches detected stars to catalog stars and fits a [`wcs::Wcs`].
//!
//! Solving is *seeded*: it expects an approximate center (RA/Dec hint) and an
//! approximate pixel scale. Blind solving is out of scope for now.

pub mod blind;
pub mod catalog;
pub mod detect;
pub mod minor_bodies;
pub mod objects;
pub mod solve;
pub mod star_ids;
pub mod wcs;

pub use detect::{DetectConfig, DetectedStar, detect_stars};
pub use wcs::Wcs;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("catalog error: {0}")]
    Catalog(String),
    #[error("solve failed: {0}")]
    Solve(String),
}
