//! Star detection, PSF fitting, and sensor tilt analysis.
//!
//! Two detector families, extracted from PSF Guard where they graded tens of
//! thousands of real frames:
//!
//! - [`hocus_focus_star_detection`] — a port of the HocusFocus plugin for
//!   N.I.N.A. by George Hilios: à trous wavelet structure removal, kappa-sigma
//!   noise thresholds, hot-pixel filtering, and multi-criteria validation,
//!   with optional Gaussian/Moffat PSF fitting per star.
//! - [`nina_star_detection`] — a port of N.I.N.A.'s standard detector, kept
//!   because its star counts and HFR values are comparable with what N.I.N.A.
//!   itself recorded into an imaging catalog.
//!
//! These are *measurement* detectors: they exist to produce trustworthy star
//! counts, HFR, FWHM, and eccentricity for quality grading. The fast
//! threshold detector in the `seiza` crate serves registration and solving,
//! where speed matters and photometric fidelity does not; the two are
//! different tools, not rivals.
//!
//! [`tilt`] turns per-star PSF measurements into parallelogram and triangle
//! sensor-tilt diagrams plus field-curvature numbers.
//!
//! A porting note that is also a warranty: the detector reductions are
//! order-sensitive, and their numeric behavior decides star counts. The code
//! here moved verbatim from its previous home and is validated against real
//! frames, not only unit fixtures — treat any "cleanup" that reorders a
//! reduction as a behavior change and prove otherwise on a corpus.

pub mod accord_imaging;
pub mod debug;
pub mod hocus_focus_star_detection;
pub mod nina_star_detection;
pub mod psf_fitting;
pub mod star_contours;
pub mod tilt;

pub use hocus_focus_star_detection::{
    HocusFocusDetectionResult, HocusFocusParams, TelescopeClass, classify_pixel_scale,
    detect_stars_hocus_focus, pixel_scale_arcsec, recommend_detection_binning,
};
pub use psf_fitting::{PSFModel, PSFType};
pub use tilt::{
    CellStats, Corner, CornerHfr, TRIANGLE_INNER_RADIUS_FRACTION,
    TRIANGLE_MINIMUM_STARS_PER_REGION, TRIANGLE_OUTER_RADIUS_FRACTION, TiltStar, TiltSummary,
    TriangleCenterStats, TriangleSectorStats, TriangleTiltError, TriangleTiltSummary,
    analyze_cells, analyze_triangle, tilt_summary,
};
