//! Color composition for astrophotography: mono stacks combined into a
//! color picture.
//!
//! - [`combine_rgb`], [`combine_lrgb`], [`combine_narrowband`] and their
//!   relatives compose registered channels, with the palettes, LRGB
//!   luminance and Foraxx blends in [`NarrowbandPalette`]; [`ColorCrop`]
//!   trims the result to the area every channel covers.
//! - [`calibrate_color`] and its relatives balance RGB against Gaia
//!   photometry.
//! - [`balance_narrowband`] balances narrowband lines, and
//!   [`stretch_rgb_stars`] and [`screen_rgb_stars`] put stars back over a
//!   starless picture.
//! - [`write_color_fits_f32`] writes a composition as FITS.
//! - [`picture::Picture`] describes any color picture as layers, masks and
//!   finishing steps, and [`render::render`] renders one in tiles: the one
//!   compositor every composition and every host draws through.
//!
//! The channels are [`seiza_stacking::LinearImage`]s, as stacking makes
//! them, and calls return [`seiza_stacking::Error`].
//!
//! # Output version
//!
//! [`OUTPUT_VERSION`] names the release of this crate whose output a build
//! reproduces. A host that caches compositions keys them on it.

mod color;
mod color_calibration;
mod crop;
mod fits;
#[cfg(feature = "gpu")]
pub mod gpu;
mod narrowband_balance;
pub mod picture;
pub mod render;
pub mod space;
mod star_stretch;
pub mod tone;

pub(crate) use seiza_stacking::{AffineTransform, Result};
/// The types this crate's functions take and give, so a caller needs no
/// other crate to name them.
pub use seiza_stacking::{Error, LinearImage, ReferenceRegion};

/// The release of this crate whose composition output a build reproduces.
/// It moves to the next minor version when a change alters any output, and
/// `tests/output_version.rs` holds it to recorded results.
pub const OUTPUT_VERSION: &str = "0.1.0";

pub use color::{
    ColorComposition, ColorNormalization, ColorOptions, ColorTransfer, ForaxxOptions,
    LrgbLuminance, NarrowbandMatrix, NarrowbandMix, NarrowbandPalette, RgbLuminance, combine_lrgb,
    combine_narrowband, combine_narrowband_matrix, combine_rgb, combine_super_lrgb,
    combine_super_rgb,
};
pub use color_calibration::{
    CalibratedStretch, ColorCalibration, ColorCalibrationOptions, ColorCalibrationOutcome,
    ColorCalibrationSkip, ColorCalibrationSkipReason, ColorFit, ColorReferenceStar,
    ColorStarMeasurement, GaiaColorSource, GaiaColors, SOLAR_BP_RP, apply_color_gains,
    calibrate_and_stretch, calibrate_color, calibrate_color_image, neutralize_clipped,
    place_gaia_sources, transform_wcs,
};
pub use crop::{
    ChannelCoverage, ChannelSamples, ColorCrop, CropReport, covered_region, crop_report,
};
pub use fits::write_color_fits_f32;
pub use narrowband_balance::{
    NarrowbandBalanceOptions, NarrowbandBalanceReport, NarrowbandChannelBalance,
    balance_narrowband, screen_rgb_stars, screen_rgb_stars_reduced,
};
pub use star_stretch::{screen_rgb_star_layer, stretch_rgb_stars};
