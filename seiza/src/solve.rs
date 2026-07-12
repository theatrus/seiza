//! Near-field plate solving: match detected stars against catalog stars
//! around a hinted position and fit a [`crate::wcs::Wcs`].
//!
//! Not yet implemented — this module fixes the API shape so integration work
//! can proceed. The planned algorithm: triangle/quad geometric hashing over
//! the brightest detected stars vs. catalog stars in the hint region, RANSAC
//! over candidate correspondences, then a linear least-squares fit of the CD
//! matrix and reference point, iterated with residual clipping.

use crate::catalog::StarCatalog;
use crate::detect::DetectedStar;
use crate::wcs::Wcs;

/// Approximate knowledge required to seed a solve.
#[derive(Debug, Clone)]
pub struct SolveHint {
    /// Approximate field center, ICRS degrees (RA, Dec)
    pub center: (f64, f64),
    /// Search radius around the hinted center, degrees
    pub radius_deg: f64,
    /// Approximate pixel scale, arcseconds per pixel
    pub scale_arcsec_px: f64,
    /// Allowed relative scale error (e.g. 0.2 = ±20 %)
    pub scale_tolerance: f64,
}

/// A successful plate solution with quality metrics.
#[derive(Debug, Clone)]
pub struct Solution {
    pub wcs: Wcs,
    /// Number of detected stars matched to catalog stars
    pub matched_stars: usize,
    /// RMS of match residuals, arcseconds
    pub rms_arcsec: f64,
}

/// Solve the field for an image of `(width, height)` given detected stars,
/// a catalog, and a hint.
pub fn solve(
    _stars: &[DetectedStar],
    _catalog: &dyn StarCatalog,
    _hint: &SolveHint,
    _dimensions: (u32, u32),
) -> Result<Solution, crate::Error> {
    Err(crate::Error::Solve(
        "near-field solving is not implemented yet".to_string(),
    ))
}
