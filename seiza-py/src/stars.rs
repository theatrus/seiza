//! Star detection, PSF fitting, and sensor tilt analysis — `seiza-stars`.
//!
//! The measurement detectors: trustworthy star counts, HFR, FWHM, and
//! eccentricity for quality grading, as distinct from the fast alignment
//! detector `seiza.detect_stars` uses for solving and registration.

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use seiza_stars::hocus_focus_star_detection::{
    HocusFocusDetectionResult, MeasurementOptions, StructureRemovalMethod, TelescopeClass,
};
use seiza_stars::psf_fitting::PSFType;
use seiza_stars::tilt;
use std::path::PathBuf;

/// One detected star with its measurements. `eccentricity`, `theta`, and
/// `r_squared` are present when a PSF model was fitted. `theta` is the
/// ellipse major-axis orientation in radians over `[0, π)`.
#[pyclass(skip_from_py_object, name = "MeasuredStar", module = "seiza", frozen)]
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
#[pyclass(skip_from_py_object, name = "TiltCell", module = "seiza", frozen)]
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

/// Parallelogram corner-vs-center tilt and curvature verdict. Corner names
/// are kebab-case: ``top-left``, ``top-right``, ``bottom-left``,
/// ``bottom-right``.
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

/// Aggregate statistics for the circular center of a triangle tilt analysis.
#[pyclass(
    skip_from_py_object,
    name = "TriangleTiltCenter",
    module = "seiza",
    frozen
)]
#[derive(Clone)]
pub(crate) struct PyTriangleTiltCenter {
    #[pyo3(get)]
    star_count: usize,
    #[pyo3(get)]
    median_hfr: Option<f64>,
}

/// Aggregate statistics for one triangle adjustment-screw sector.
#[pyclass(
    skip_from_py_object,
    name = "TriangleTiltSector",
    module = "seiza",
    frozen
)]
#[derive(Clone)]
pub(crate) struct PyTriangleTiltSector {
    /// Stable 1-based adjustment-screw identifier.
    #[pyo3(get)]
    sector: u8,
    /// Image-coordinate degrees: zero points up, positive turns clockwise.
    #[pyo3(get)]
    axis_angle_degrees: f64,
    #[pyo3(get)]
    star_count: usize,
    #[pyo3(get)]
    median_hfr: Option<f64>,
}

/// Native triangle-sector tilt analysis and its confidence provenance.
#[pyclass(name = "TriangleTiltSummary", module = "seiza", frozen)]
pub(crate) struct PyTriangleTiltSummary {
    #[pyo3(get)]
    angle_degrees: f64,
    #[pyo3(get)]
    inner_radius_pixels: f64,
    #[pyo3(get)]
    outer_radius_pixels: f64,
    #[pyo3(get)]
    minimum_stars_per_region: usize,
    #[pyo3(get)]
    ready: bool,
    #[pyo3(get)]
    center: PyTriangleTiltCenter,
    #[pyo3(get)]
    sectors: Vec<PyTriangleTiltSector>,
    #[pyo3(get)]
    overall_median_hfr: Option<f64>,
    #[pyo3(get)]
    tilt_percent: Option<f64>,
    #[pyo3(get)]
    best_sector: Option<u8>,
    #[pyo3(get)]
    worst_sector: Option<u8>,
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
    TelescopeClass::from_name(name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown preset {name:?} (widefield, standard, longfocal)"
        ))
    })
}

/// The keyword arguments both detection calls take, as the core's options.
#[allow(clippy::too_many_arguments)]
fn measurement_options(
    preset: Option<&str>,
    focal_length_mm: Option<f64>,
    pixel_size_um: Option<f64>,
    psf_type: &str,
    structure_removal: Option<&str>,
    detection_binning: Option<usize>,
    keep_saturated: Option<bool>,
    noise_reduction_radius: Option<usize>,
    sensitivity: Option<f64>,
    target_star_count: Option<usize>,
) -> PyResult<MeasurementOptions> {
    Ok(MeasurementOptions {
        preset: preset.map(parse_preset).transpose()?,
        focal_length_mm,
        pixel_size_um,
        psf_type: parse_psf_type(psf_type)?,
        structure_removal: structure_removal.map(parse_structure_removal).transpose()?,
        detection_binning,
        keep_saturated,
        noise_reduction_radius,
        sensitivity,
        target_star_count,
    })
}

fn detection_result(
    result: HocusFocusDetectionResult,
    width: usize,
    height: usize,
) -> PyStarDetectionResult {
    PyStarDetectionResult {
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
                theta: star
                    .psf_model
                    .as_ref()
                    .map(seiza_stars::psf_fitting::PSFModel::major_axis_theta),
                r_squared: star.psf_model.as_ref().map(|psf| psf.r_squared),
            })
            .collect(),
        average_hfr: result.average_hfr,
        average_fwhm: result.average_fwhm,
        noise_sigma: result.noise_sigma,
        background_mean: result.background_mean,
        width,
        height,
    }
}

/// Detect and measure stars in a mono ``uint16`` frame with the HocusFocus
/// detector: wavelet structure removal, kappa-sigma thresholds, hot-pixel
/// filtering, multi-criteria validation, and optional PSF fitting.
///
/// The preset resolves in this order: explicit ``preset``; else
/// ``focal_length_mm`` + ``pixel_size_um`` classify the pixel scale the way
/// the FITS-header path does; else the standard defaults. Explicit knobs
/// always override the preset they land on. A positive
/// ``target_star_count`` retries with looser settings (a lower SNR gate,
/// then native resolution with no blur) until that many stars are
/// measured, and returns the best pass otherwise.
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
    target_star_count=None,
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
    target_star_count: Option<usize>,
) -> PyResult<PyStarDetectionResult> {
    let shape = image.shape();
    let (height, width) = (shape[0], shape[1]);
    let data = image.as_array().iter().copied().collect::<Vec<u16>>();
    let options = measurement_options(
        preset,
        focal_length_mm,
        pixel_size_um,
        psf_type,
        structure_removal,
        detection_binning,
        keep_saturated,
        noise_reduction_radius,
        sensitivity,
        target_star_count,
    )?;
    let result = py.detach(move || options.detect(&data, width, height));
    Ok(detection_result(result, width, height))
}

/// :func:`detect_measured_stars` on a FITS or XISF file: its linear 16-bit
/// luminance, with planar RGB collapsed to luminance and a raw Bayer mosaic
/// debayered first. When no ``preset`` is given, a ``focal_length_mm`` left
/// out is read from ``FOCALLEN``, ``FOCALLENGTH`` or ``FOCAL``, and a
/// ``pixel_size_um`` left out from ``XPIXSZ``; a value given wins, each on
/// its own. Other formats are refused.
#[pyfunction]
#[pyo3(signature = (
    path,
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
    target_star_count=None,
))]
#[allow(clippy::too_many_arguments)]
fn detect_measured_stars_file(
    py: Python<'_>,
    path: PathBuf,
    preset: Option<&str>,
    focal_length_mm: Option<f64>,
    pixel_size_um: Option<f64>,
    psf_type: &str,
    structure_removal: Option<&str>,
    detection_binning: Option<usize>,
    keep_saturated: Option<bool>,
    noise_reduction_radius: Option<usize>,
    sensitivity: Option<f64>,
    target_star_count: Option<usize>,
) -> PyResult<PyStarDetectionResult> {
    let options = measurement_options(
        preset,
        focal_length_mm,
        pixel_size_um,
        psf_type,
        structure_removal,
        detection_binning,
        keep_saturated,
        noise_reduction_radius,
        sensitivity,
        target_star_count,
    )?;
    let (result, width, height) = py
        .detach(move || {
            let (image, _) = seiza_image_io::open_astronomy(&path)?;
            let options = options.with_frame_optics(
                seiza_image_io::header_focal_length_mm(&image.headers),
                seiza_image_io::header_pixel_size_um(&image.headers),
            );
            let samples = seiza_image_io::luma_u16(&image);
            Ok::<_, seiza_image_io::Error>((
                options.detect(&samples, image.width, image.height),
                image.width,
                image.height,
            ))
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(detection_result(result, width, height))
}

/// The pixel scale, in arcseconds per pixel, of a camera with pixels
/// ``pixel_size_um`` micrometres across (binning included) behind a
/// telescope of focal length ``focal_length_mm`` millimetres; ``None``
/// unless both are positive.
#[pyfunction]
fn pixel_scale_arcsec(focal_length_mm: f64, pixel_size_um: f64) -> Option<f64> {
    seiza_stars::pixel_scale_arcsec(focal_length_mm, pixel_size_um)
}

/// The detection ``preset`` for a rig of ``scale_arcsec_px``:
/// ``"widefield"`` from 1.2"/px up, ``"longfocal"`` at 0.6"/px and below,
/// and ``"standard"`` between, or for a scale that is not a positive
/// number.
#[pyfunction]
fn classify_pixel_scale(scale_arcsec_px: f64) -> &'static str {
    seiza_stars::classify_pixel_scale(scale_arcsec_px).as_str()
}

/// The ``detection_binning``, 1 to 4, from an in-focus star's measured HFR
/// in captured pixels: the factor that brings it nearest the detector's
/// calibrated 3 pixels. 1 for an HFR that is not a positive number.
#[pyfunction]
fn recommend_detection_binning(measured_hfr_px: f64) -> usize {
    seiza_stars::recommend_detection_binning(measured_hfr_px)
}

fn tilt_stars(result: &PyStarDetectionResult) -> Vec<tilt::TiltStar> {
    result
        .stars
        .iter()
        .map(|star| tilt::TiltStar {
            x: star.x,
            y: star.y,
            hfr: star.hfr,
            eccentricity: star.eccentricity.unwrap_or(0.0),
            theta: star.theta,
        })
        .collect()
}

/// Parallelogram sensor-tilt and field-curvature analysis over a detection
/// result: the 3×3 grid's per-cell statistics and the corner-vs-center
/// verdict. Stars without a fitted PSF contribute HFR but no direction.
#[pyfunction]
fn tilt_analysis(result: &PyStarDetectionResult) -> PyResult<(Vec<PyTiltCell>, PyTiltSummary)> {
    let stars = tilt_stars(result);
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
            worst_corner: summary
                .worst_corner
                .map(|corner| corner.as_str().to_string()),
            best_corner: summary
                .best_corner
                .map(|corner| corner.as_str().to_string()),
        },
    ))
}

/// Triangle tilt analysis over an existing detection result without running
/// the detector again.
///
/// ``angle_degrees`` locates sector 1 in image coordinates: zero points to
/// the top and positive angles turn clockwise. The result normalizes it over
/// ``[0, 360)``. The center disk ends at 25% of the center-to-corner distance;
/// the outer annulus radius is half the shorter image dimension. Annular stars
/// are divided among three complete, half-open 120-degree sectors. The tilt
/// verdict is withheld until every sector contains at least
/// ``minimum_stars_per_region`` measurements.
#[pyfunction]
#[pyo3(signature = (result, *, angle_degrees=0.0))]
fn triangle_tilt_analysis(
    result: &PyStarDetectionResult,
    angle_degrees: f64,
) -> PyResult<PyTriangleTiltSummary> {
    let stars = tilt_stars(result);
    let summary = tilt::analyze_triangle(&stars, result.width, result.height, angle_degrees)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyTriangleTiltSummary {
        angle_degrees: summary.angle_degrees,
        inner_radius_pixels: summary.inner_radius_pixels,
        outer_radius_pixels: summary.outer_radius_pixels,
        minimum_stars_per_region: summary.minimum_stars_per_region,
        ready: summary.ready,
        center: PyTriangleTiltCenter {
            star_count: summary.center.star_count,
            median_hfr: summary.center.median_hfr,
        },
        sectors: summary
            .sectors
            .iter()
            .map(|sector| PyTriangleTiltSector {
                sector: sector.sector,
                axis_angle_degrees: sector.axis_angle_degrees,
                star_count: sector.star_count,
                median_hfr: sector.median_hfr,
            })
            .collect(),
        overall_median_hfr: summary.overall_median_hfr,
        tilt_percent: summary.tilt_percent,
        best_sector: summary.best_sector,
        worst_sector: summary.worst_sector,
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMeasuredStar>()?;
    module.add_class::<PyStarDetectionResult>()?;
    module.add_class::<PyTiltCell>()?;
    module.add_class::<PyTiltSummary>()?;
    module.add_class::<PyTriangleTiltCenter>()?;
    module.add_class::<PyTriangleTiltSector>()?;
    module.add_class::<PyTriangleTiltSummary>()?;
    module.add_function(wrap_pyfunction!(detect_measured_stars, module)?)?;
    module.add_function(wrap_pyfunction!(detect_measured_stars_file, module)?)?;
    module.add_function(wrap_pyfunction!(pixel_scale_arcsec, module)?)?;
    module.add_function(wrap_pyfunction!(classify_pixel_scale, module)?)?;
    module.add_function(wrap_pyfunction!(recommend_detection_binning, module)?)?;
    module.add_function(wrap_pyfunction!(tilt_analysis, module)?)?;
    module.add_function(wrap_pyfunction!(triangle_tilt_analysis, module)?)?;
    Ok(())
}
