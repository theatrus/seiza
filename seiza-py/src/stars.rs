//! Star detection, PSF fitting, and sensor tilt analysis — `seiza-stars`.
//!
//! The measurement detectors: trustworthy star counts, HFR, FWHM, and
//! eccentricity for quality grading, as distinct from the fast alignment
//! detector `seiza.detect_stars` uses for solving and registration.

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use seiza_stars::hocus_focus_star_detection::{
    HocusFocusParams, StructureRemovalMethod, TelescopeClass, detect_stars_hocus_focus,
};
use seiza_stars::psf_fitting::PSFType;
use seiza_stars::tilt;

/// One detected star with its measurements. `eccentricity`, `theta`, and
/// `r_squared` are present when a PSF model was fitted.
#[pyclass(name = "MeasuredStar", module = "seiza", frozen)]
#[derive(Clone)]
pub(crate) struct PyMeasuredStar {
    #[pyo3(get)]
    x: f64,
    #[pyo3(get)]
    y: f64,
    #[pyo3(get)]
    hfr: f64,
    #[pyo3(get)]
    fwhm: f64,
    #[pyo3(get)]
    brightness: f64,
    #[pyo3(get)]
    background: f64,
    #[pyo3(get)]
    snr: f64,
    #[pyo3(get)]
    flux: f64,
    #[pyo3(get)]
    pixel_count: usize,
    #[pyo3(get)]
    saturated: bool,
    #[pyo3(get)]
    eccentricity: Option<f64>,
    #[pyo3(get)]
    theta: Option<f64>,
    #[pyo3(get)]
    r_squared: Option<f64>,
}

/// A detection run's stars and frame-level statistics.
#[pyclass(name = "StarDetectionResult", module = "seiza", frozen)]
pub(crate) struct PyStarDetectionResult {
    #[pyo3(get)]
    stars: Vec<PyMeasuredStar>,
    #[pyo3(get)]
    average_hfr: f64,
    #[pyo3(get)]
    average_fwhm: f64,
    #[pyo3(get)]
    noise_sigma: f64,
    #[pyo3(get)]
    background_mean: f64,
    #[pyo3(get)]
    width: usize,
    #[pyo3(get)]
    height: usize,
}

/// One 3×3 grid cell's aggregate star statistics.
#[pyclass(name = "TiltCell", module = "seiza", frozen)]
#[derive(Clone)]
pub(crate) struct PyTiltCell {
    #[pyo3(get)]
    row: usize,
    #[pyo3(get)]
    col: usize,
    #[pyo3(get)]
    star_count: usize,
    #[pyo3(get)]
    median_hfr: Option<f64>,
    #[pyo3(get)]
    median_eccentricity: Option<f64>,
    #[pyo3(get)]
    mean_theta: Option<f64>,
    #[pyo3(get)]
    theta_coherence: f64,
}

/// ASTAP-style corner-vs-center tilt and curvature verdict. Corner names are
/// kebab-case: ``top-left``, ``top-right``, ``bottom-left``, ``bottom-right``.
#[pyclass(name = "TiltSummary", module = "seiza", frozen)]
pub(crate) struct PyTiltSummary {
    #[pyo3(get)]
    center_hfr: Option<f64>,
    /// ``[(corner, hfr_or_None), ...]`` for the four corners.
    #[pyo3(get)]
    corners: Vec<(String, Option<f64>)>,
    #[pyo3(get)]
    mean_hfr: Option<f64>,
    #[pyo3(get)]
    tilt_percent: Option<f64>,
    #[pyo3(get)]
    curvature_percent: Option<f64>,
    #[pyo3(get)]
    worst_corner: Option<String>,
    #[pyo3(get)]
    best_corner: Option<String>,
}

fn parse_psf_type(name: &str) -> PyResult<PSFType> {
    match name {
        "none" => Ok(PSFType::None),
        "gaussian" => Ok(PSFType::Gaussian),
        "moffat4" => Ok(PSFType::Moffat4),
        other => Err(PyValueError::new_err(format!(
            "unknown psf_type {other:?} (none, gaussian, moffat4)"
        ))),
    }
}

fn parse_structure_removal(name: &str) -> PyResult<StructureRemovalMethod> {
    match name {
        "filtered" => Ok(StructureRemovalMethod::Filtered),
        "atrous" => Ok(StructureRemovalMethod::Atrous),
        other => Err(PyValueError::new_err(format!(
            "unknown structure_removal {other:?} (filtered, atrous)"
        ))),
    }
}

fn parse_preset(name: &str) -> PyResult<TelescopeClass> {
    match name {
        "widefield" => Ok(TelescopeClass::WideField),
        "standard" => Ok(TelescopeClass::Standard),
        "longfocal" => Ok(TelescopeClass::LongFocalLength),
        other => Err(PyValueError::new_err(format!(
            "unknown preset {other:?} (widefield, standard, longfocal)"
        ))),
    }
}

/// Detect and measure stars in a mono ``uint16`` frame with the HocusFocus
/// detector: wavelet structure removal, kappa-sigma thresholds, hot-pixel
/// filtering, multi-criteria validation, and optional PSF fitting.
///
/// The preset resolves in this order: explicit ``preset``; else
/// ``focal_length_mm`` + ``pixel_size_um`` classify the pixel scale the way
/// the FITS-header path does; else the standard defaults. Explicit knobs
/// always override the preset they land on.
#[pyfunction]
#[pyo3(signature = (
    image,
    *,
    preset=None,
    focal_length_mm=None,
    pixel_size_um=None,
    psf_type="moffat4",
    structure_removal=None,
    detection_binning=None,
    keep_saturated=None,
    noise_reduction_radius=None,
    sensitivity=None,
))]
#[allow(clippy::too_many_arguments)]
fn detect_measured_stars(
    py: Python<'_>,
    image: PyReadonlyArray2<'_, u16>,
    preset: Option<&str>,
    focal_length_mm: Option<f64>,
    pixel_size_um: Option<f64>,
    psf_type: &str,
    structure_removal: Option<&str>,
    detection_binning: Option<usize>,
    keep_saturated: Option<bool>,
    noise_reduction_radius: Option<usize>,
    sensitivity: Option<f64>,
) -> PyResult<PyStarDetectionResult> {
    let shape = image.shape();
    let (height, width) = (shape[0], shape[1]);
    let data = image.as_array().iter().copied().collect::<Vec<u16>>();

    let mut params = match preset {
        Some(name) => HocusFocusParams::for_telescope_class(parse_preset(name)?),
        None => HocusFocusParams::for_frame_headers(focal_length_mm, pixel_size_um).0,
    };
    params.psf_type = parse_psf_type(psf_type)?;
    if let Some(method) = structure_removal {
        params.structure_removal = parse_structure_removal(method)?;
    }
    if let Some(binning) = detection_binning {
        params.detection_binning = binning.max(1);
    }
    if let Some(keep) = keep_saturated {
        params.keep_saturated_stars = keep;
    }
    if let Some(radius) = noise_reduction_radius {
        params.noise_reduction_radius = radius;
    }
    if let Some(value) = sensitivity {
        params.sensitivity = value;
    }

    let result =
        py.allow_threads(move || detect_stars_hocus_focus(&data, width, height, &params));
    Ok(PyStarDetectionResult {
        stars: result
            .stars
            .iter()
            .map(|star| PyMeasuredStar {
                x: star.position.0,
                y: star.position.1,
                hfr: star.hfr,
                fwhm: star.fwhm,
                brightness: star.brightness,
                background: star.background,
                snr: star.snr,
                flux: star.flux,
                pixel_count: star.pixel_count,
                saturated: star.saturated,
                eccentricity: star.psf_model.as_ref().map(|psf| psf.eccentricity),
                theta: star.psf_model.as_ref().map(|psf| psf.theta),
                r_squared: star.psf_model.as_ref().map(|psf| psf.r_squared),
            })
            .collect(),
        average_hfr: result.average_hfr,
        average_fwhm: result.average_fwhm,
        noise_sigma: result.noise_sigma,
        background_mean: result.background_mean,
        width,
        height,
    })
}

/// ASTAP-style sensor tilt and field-curvature analysis over a detection
/// result: the 3×3 grid's per-cell statistics and the corner-vs-center
/// verdict. Stars without a fitted PSF contribute HFR but no direction.
#[pyfunction]
fn tilt_analysis(
    result: &PyStarDetectionResult,
) -> PyResult<(Vec<PyTiltCell>, PyTiltSummary)> {
    let stars: Vec<tilt::TiltStar> = result
        .stars
        .iter()
        .map(|star| tilt::TiltStar {
            x: star.x,
            y: star.y,
            hfr: star.hfr,
            eccentricity: star.eccentricity.unwrap_or(0.0),
            theta: star.theta,
        })
        .collect();
    let cells = tilt::analyze_cells(&stars, result.width, result.height);
    let summary = tilt::tilt_summary(&cells);
    Ok((
        cells
            .iter()
            .map(|cell| PyTiltCell {
                row: cell.row,
                col: cell.col,
                star_count: cell.star_count,
                median_hfr: cell.median_hfr,
                median_eccentricity: cell.median_eccentricity,
                mean_theta: cell.mean_theta,
                theta_coherence: cell.theta_coherence,
            })
            .collect(),
        PyTiltSummary {
            center_hfr: summary.center_hfr,
            corners: summary
                .corners
                .iter()
                .map(|corner| (corner.corner.as_str().to_string(), corner.hfr))
                .collect(),
            mean_hfr: summary.mean_hfr,
            tilt_percent: summary.tilt_percent,
            curvature_percent: summary.curvature_percent,
            worst_corner: summary.worst_corner.map(|corner| corner.as_str().to_string()),
            best_corner: summary.best_corner.map(|corner| corner.as_str().to_string()),
        },
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMeasuredStar>()?;
    module.add_class::<PyStarDetectionResult>()?;
    module.add_class::<PyTiltCell>()?;
    module.add_class::<PyTiltSummary>()?;
    module.add_function(wrap_pyfunction!(detect_measured_stars, module)?)?;
    module.add_function(wrap_pyfunction!(tilt_analysis, module)?)?;
    Ok(())
}
