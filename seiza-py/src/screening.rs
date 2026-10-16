//! Judging bias, dark and flat frames before a master is built, and
//! residual flat responses: the light a flat left uncorrected.

use crate::arrays::{float_array, float_image_view};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use seiza_calibration::{
    BiasScreening, DarkMeasureOptions, DarkMeasurement, DarkScreening, Finding, FlatFeature,
    FlatMeasureOptions, FlatMeasurement, FlatScreening, FlatSetReport, FrameVerdict,
    IlluminationProfile, Limits, LinearImageMut, LinearImageRef, ResidualFlatDiagnostics,
    ResidualFlatOptions,
};
use seiza_stacking::FitsFrame;
use std::path::PathBuf;

fn value_error(error: seiza_calibration::Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// A raw frame read through seiza-image-io, as the master builder reads it.
fn open_frame(py: Python<'_>, path: PathBuf) -> PyResult<FitsFrame> {
    py.detach(|| FitsFrame::open(&path))
        .map_err(|error| crate::stacking::StackError::new_err(error.to_string()))
}

fn view(frame: &FitsFrame) -> PyResult<LinearImageRef<'_>> {
    let image = &frame.image;
    LinearImageRef::new(&image.data, image.width, image.height, image.channels).map_err(value_error)
}

fn json_error(error: serde_json::Error) -> PyErr {
    PyValueError::new_err(format!("invalid measurement JSON: {error}"))
}

/// What :func:`measure_dark` found in one bias or dark frame, in the frame's
/// own units.
#[pyclass(
    skip_from_py_object,
    frozen,
    name = "DarkMeasurement",
    module = "seiza"
)]
#[derive(Clone)]
pub(crate) struct PyDarkMeasurement {
    inner: DarkMeasurement,
}

#[pymethods]
impl PyDarkMeasurement {
    /// The frame quality algorithm this was measured with. A stored
    /// measurement whose version differs from a new one's is stale.
    #[getter]
    fn algorithm_version(&self) -> u32 {
        self.inner.algorithm_version
    }
    #[getter]
    fn width(&self) -> usize {
        self.inner.width
    }
    #[getter]
    fn height(&self) -> usize {
        self.inner.height
    }
    /// Median of every sixteenth sample.
    #[getter]
    fn level(&self) -> f32 {
        self.inner.level
    }
    /// Robust spread of those samples around the level, glow and pattern
    /// included.
    #[getter]
    fn sigma(&self) -> f32 {
        self.inner.sigma
    }
    /// Pixel-scale noise.
    #[getter]
    fn noise(&self) -> f32 {
        self.inner.noise
    }
    #[getter]
    fn row_banding(&self) -> f32 {
        self.inner.row_banding
    }
    #[getter]
    fn column_banding(&self) -> f32 {
        self.inner.column_banding
    }
    #[getter]
    fn hot_pixels(&self) -> u64 {
        self.inner.hot_pixels
    }
    #[getter]
    fn hot_threshold(&self) -> f32 {
        self.inner.hot_threshold
    }

    /// This measurement as JSON, to keep and judge again later.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(json_error)
    }

    /// A measurement :meth:`to_json` wrote.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        serde_json::from_str(text)
            .map(|inner| Self { inner })
            .map_err(json_error)
    }

    fn __repr__(&self) -> String {
        format!(
            "DarkMeasurement(level={}, noise={}, hot_pixels={})",
            self.inner.level, self.inner.noise, self.inner.hot_pixels
        )
    }
}

/// A coarse map of how light falls across a flat.
#[pyclass(
    skip_from_py_object,
    frozen,
    name = "IlluminationProfile",
    module = "seiza"
)]
pub(crate) struct PyIlluminationProfile {
    inner: IlluminationProfile,
}

#[pymethods]
impl PyIlluminationProfile {
    #[getter]
    fn columns(&self) -> usize {
        self.inner.columns
    }
    #[getter]
    fn rows(&self) -> usize {
        self.inner.rows
    }
    /// Each cell's median over the median cell, as a ``(rows, columns)``
    /// array: a typical cell reads 1.
    #[getter]
    fn response<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        float_array(
            py,
            self.inner.columns,
            self.inner.rows,
            1,
            self.inner.response.clone(),
        )
    }
}

/// One compact dip in a flat, in image pixels.
#[pyclass(skip_from_py_object, frozen, name = "FlatFeature", module = "seiza")]
pub(crate) struct PyFlatFeature {
    inner: FlatFeature,
}

#[pymethods]
impl PyFlatFeature {
    #[getter]
    fn x(&self) -> f32 {
        self.inner.x
    }
    #[getter]
    fn y(&self) -> f32 {
        self.inner.y
    }
    #[getter]
    fn left(&self) -> u32 {
        self.inner.left
    }
    #[getter]
    fn top(&self) -> u32 {
        self.inner.top
    }
    #[getter]
    fn right(&self) -> u32 {
        self.inner.right
    }
    #[getter]
    fn bottom(&self) -> u32 {
        self.inner.bottom
    }
    #[getter]
    fn pixels(&self) -> u32 {
        self.inner.pixels
    }
    /// The deepest point, as a fraction of the light around it.
    #[getter]
    fn depth(&self) -> f32 {
        self.inner.depth
    }

    fn __repr__(&self) -> String {
        format!(
            "FlatFeature(x={}, y={}, depth={})",
            self.inner.x, self.inner.y, self.inner.depth
        )
    }
}

/// What :func:`measure_flat` found in one flat frame.
#[pyclass(
    skip_from_py_object,
    frozen,
    name = "FlatMeasurement",
    module = "seiza"
)]
#[derive(Clone)]
pub(crate) struct PyFlatMeasurement {
    inner: FlatMeasurement,
}

#[pymethods]
impl PyFlatMeasurement {
    #[getter]
    fn algorithm_version(&self) -> u32 {
        self.inner.algorithm_version
    }
    #[getter]
    fn width(&self) -> usize {
        self.inner.width
    }
    #[getter]
    fn height(&self) -> usize {
        self.inner.height
    }
    #[getter]
    fn full_scale(&self) -> f32 {
        self.inner.full_scale
    }
    #[getter]
    fn level(&self) -> f32 {
        self.inner.level
    }
    /// Median of each channel, or of each site of a CFA's 2x2 pattern.
    #[getter]
    fn channel_levels(&self) -> Vec<f32> {
        self.inner.channel_levels.clone()
    }
    #[getter]
    fn clipped_fraction(&self) -> f64 {
        self.inner.clipped_fraction
    }
    /// Pixel-scale noise, as a fraction of the light.
    #[getter]
    fn noise(&self) -> f32 {
        self.inner.noise
    }
    /// How deep a dip had to be to count as a feature in this frame.
    #[getter]
    fn detection_depth(&self) -> f32 {
        self.inner.detection_depth
    }
    #[getter]
    fn illumination(&self) -> PyIlluminationProfile {
        PyIlluminationProfile {
            inner: self.inner.illumination.clone(),
        }
    }
    /// Features found, including any beyond :attr:`features`.
    #[getter]
    fn feature_count(&self) -> usize {
        self.inner.feature_count
    }
    /// The deepest features, deepest first.
    #[getter]
    fn features(&self) -> Vec<PyFlatFeature> {
        self.inner
            .features
            .iter()
            .map(|&inner| PyFlatFeature { inner })
            .collect()
    }

    /// This measurement as JSON, to keep and judge again later.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(json_error)
    }

    /// A measurement :meth:`to_json` wrote.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        serde_json::from_str(text)
            .map(|inner| Self { inner })
            .map_err(json_error)
    }

    fn __repr__(&self) -> String {
        format!(
            "FlatMeasurement(level={}, clipped_fraction={}, feature_count={})",
            self.inner.level, self.inner.clipped_fraction, self.inner.feature_count
        )
    }
}

/// One limit a frame crossed, with the numbers behind it.
#[pyclass(skip_from_py_object, frozen, name = "Finding", module = "seiza")]
pub(crate) struct PyFinding {
    inner: Finding,
}

#[pymethods]
impl PyFinding {
    /// What was measured: ``"clipped"``, ``"level_excess"``,
    /// ``"hot_pixels"`` and so on.
    #[getter]
    fn code(&self) -> &'static str {
        self.inner.code.as_str()
    }
    /// ``"suspect"`` or ``"poor"``.
    #[getter]
    fn grade(&self) -> &'static str {
        self.inner.grade.as_str()
    }
    #[getter]
    fn value(&self) -> f64 {
        self.inner.value
    }
    /// The limit the value reached.
    #[getter]
    fn limit(&self) -> f64 {
        self.inner.limit
    }

    fn __repr__(&self) -> String {
        format!(
            "Finding(code={:?}, grade={:?}, value={}, limit={})",
            self.code(),
            self.grade(),
            self.inner.value,
            self.inner.limit
        )
    }
}

/// One frame's result: its worst grade and every limit it crossed.
#[pyclass(skip_from_py_object, frozen, name = "FrameVerdict", module = "seiza")]
pub(crate) struct PyFrameVerdict {
    inner: FrameVerdict,
}

#[pymethods]
impl PyFrameVerdict {
    /// ``"ok"``, ``"suspect"`` or ``"poor"``.
    #[getter]
    fn grade(&self) -> &'static str {
        self.inner.grade.as_str()
    }
    #[getter]
    fn findings(&self) -> Vec<PyFinding> {
        self.inner
            .findings
            .iter()
            .map(|finding| PyFinding {
                inner: finding.clone(),
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "FrameVerdict(grade={:?}, findings={})",
            self.grade(),
            self.inner.findings.len()
        )
    }
}

fn verdicts(frames: Vec<FrameVerdict>) -> Vec<PyFrameVerdict> {
    frames
        .into_iter()
        .map(|inner| PyFrameVerdict { inner })
        .collect()
}

/// :func:`screen_flats`' judgement of one flat set.
#[pyclass(skip_from_py_object, frozen, name = "FlatSetReport", module = "seiza")]
pub(crate) struct PyFlatSetReport {
    inner: FlatSetReport,
}

#[pymethods]
impl PyFlatSetReport {
    /// One verdict per frame, in input order.
    #[getter]
    fn frames(&self) -> Vec<PyFrameVerdict> {
        verdicts(self.inner.frames.clone())
    }
    /// The verdict on a master built from every frame.
    #[getter]
    fn master(&self) -> PyFrameVerdict {
        PyFrameVerdict {
            inner: self.inner.master.clone(),
        }
    }
    #[getter]
    fn steady_features(&self) -> usize {
        self.inner.steady_features
    }
    #[getter]
    fn changing_features(&self) -> usize {
        self.inner.changing_features
    }
    #[getter]
    fn carried_features(&self) -> usize {
        self.inner.carried_features
    }

    fn __repr__(&self) -> String {
        format!(
            "FlatSetReport(frames={}, master={:?}, changing_features={})",
            self.inner.frames.len(),
            self.inner.master.grade.as_str(),
            self.inner.changing_features
        )
    }
}

/// Measure one raw bias or dark frame from a FITS or XISF file.
#[pyfunction]
#[pyo3(signature = (path, *, hot_sigma=None))]
fn measure_dark(
    py: Python<'_>,
    path: PathBuf,
    hot_sigma: Option<f32>,
) -> PyResult<PyDarkMeasurement> {
    let mut options = DarkMeasureOptions::default();
    if let Some(hot_sigma) = hot_sigma {
        options.hot_sigma = hot_sigma;
    }
    let frame = open_frame(py, path)?;
    let image = view(&frame)?;
    py.detach(|| seiza_calibration::measure_dark(image, &options))
        .map(|inner| PyDarkMeasurement { inner })
        .map_err(value_error)
}

/// Measure one raw flat frame from a FITS or XISF file.
#[pyfunction]
#[pyo3(signature = (path, *, cfa=None, full_scale=None, feature_depth=None, feature_sigma=None))]
fn measure_flat(
    py: Python<'_>,
    path: PathBuf,
    cfa: Option<bool>,
    full_scale: Option<f32>,
    feature_depth: Option<f32>,
    feature_sigma: Option<f32>,
) -> PyResult<PyFlatMeasurement> {
    let mut options = FlatMeasureOptions::default();
    if let Some(value) = full_scale {
        options.full_scale = value;
    }
    if let Some(value) = feature_depth {
        options.feature_depth = value;
    }
    if let Some(value) = feature_sigma {
        options.feature_sigma = value;
    }
    let frame = open_frame(py, path)?;
    let cfa = cfa.unwrap_or(frame.bayer.is_some());
    let image = view(&frame)?;
    py.detach(|| seiza_calibration::measure_flat(image, cfa, &options))
        .map(|inner| PyFlatMeasurement { inner })
        .map_err(value_error)
}

fn limits(value: Option<(f64, f64)>, target: &mut Limits) {
    if let Some((suspect, poor)) = value {
        *target = Limits::new(suspect, poor);
    }
}

/// Judge a set of bias frames that should match. Each limit is a
/// ``(suspect, poor)`` pair; higher is worse.
#[pyfunction]
#[pyo3(signature = (measurements, *, level=None, noise=None, banding=None))]
fn screen_bias(
    measurements: Vec<PyRef<'_, PyDarkMeasurement>>,
    level: Option<(f64, f64)>,
    noise: Option<(f64, f64)>,
    banding: Option<(f64, f64)>,
) -> Vec<PyFrameVerdict> {
    let mut screening = BiasScreening::default();
    limits(level, &mut screening.level);
    limits(noise, &mut screening.noise);
    limits(banding, &mut screening.banding);
    let measurements: Vec<DarkMeasurement> = measurements
        .iter()
        .map(|measurement| measurement.inner.clone())
        .collect();
    verdicts(seiza_calibration::screen_bias(&measurements, &screening))
}

/// Judge a set of darks that should match. Each limit is a ``(suspect,
/// poor)`` pair; higher is worse.
#[pyfunction]
#[pyo3(signature = (measurements, *, level_excess=None, noise=None, hot_pixels=None, banding=None))]
fn screen_darks(
    measurements: Vec<PyRef<'_, PyDarkMeasurement>>,
    level_excess: Option<(f64, f64)>,
    noise: Option<(f64, f64)>,
    hot_pixels: Option<(f64, f64)>,
    banding: Option<(f64, f64)>,
) -> Vec<PyFrameVerdict> {
    let mut screening = DarkScreening::default();
    limits(level_excess, &mut screening.level_excess);
    limits(noise, &mut screening.noise);
    limits(hot_pixels, &mut screening.hot_pixels);
    limits(banding, &mut screening.banding);
    let measurements: Vec<DarkMeasurement> = measurements
        .iter()
        .map(|measurement| measurement.inner.clone())
        .collect();
    verdicts(seiza_calibration::screen_darks(&measurements, &screening))
}

/// Judge a set of flats that should match. Each limit is a ``(suspect,
/// poor)`` pair; higher is worse except for ``dim``.
#[pyfunction]
#[pyo3(signature = (
    measurements,
    *,
    clipped=None,
    bright=None,
    dim=None,
    illumination=None,
    features=None,
    changing_features=None,
    carried_features=None,
    steady_presence=None,
    steady_depth_range=None,
    match_distance=None
))]
#[allow(clippy::too_many_arguments)]
fn screen_flats(
    py: Python<'_>,
    measurements: Vec<PyRef<'_, PyFlatMeasurement>>,
    clipped: Option<(f64, f64)>,
    bright: Option<(f64, f64)>,
    dim: Option<(f64, f64)>,
    illumination: Option<(f64, f64)>,
    features: Option<(f64, f64)>,
    changing_features: Option<(f64, f64)>,
    carried_features: Option<(f64, f64)>,
    steady_presence: Option<f64>,
    steady_depth_range: Option<f64>,
    match_distance: Option<u32>,
) -> PyResult<PyFlatSetReport> {
    let mut screening = FlatScreening::default();
    limits(clipped, &mut screening.clipped);
    limits(bright, &mut screening.bright);
    limits(dim, &mut screening.dim);
    limits(illumination, &mut screening.illumination);
    limits(features, &mut screening.features);
    limits(changing_features, &mut screening.changing_features);
    limits(carried_features, &mut screening.carried_features);
    if let Some(value) = steady_presence {
        screening.steady_presence = value;
    }
    if let Some(value) = steady_depth_range {
        screening.steady_depth_range = value;
    }
    if let Some(value) = match_distance {
        screening.match_distance = value;
    }
    let measurements: Vec<FlatMeasurement> = measurements
        .iter()
        .map(|measurement| measurement.inner.clone())
        .collect();
    py.detach(|| seiza_calibration::screen_flats(&measurements, &screening))
        .map(|inner| PyFlatSetReport { inner })
        .map_err(value_error)
}

/// The evidence behind a residual flat response from
/// :func:`build_residual_flat`.
#[pyclass(
    skip_from_py_object,
    frozen,
    name = "ResidualFlatDiagnostics",
    module = "seiza"
)]
pub(crate) struct PyResidualFlatDiagnostics {
    inner: ResidualFlatDiagnostics,
}

#[pymethods]
impl PyResidualFlatDiagnostics {
    /// Keep it with a stored response.
    #[getter]
    fn algorithm_version(&self) -> u32 {
        self.inner.algorithm_version
    }
    /// Crops used.
    #[getter]
    fn sample_count(&self) -> usize {
        self.inner.sample_count
    }
    /// Pixel-channel samples with enough shared evidence to correct.
    #[getter]
    fn corrected_samples(&self) -> usize {
        self.inner.corrected_samples
    }
    #[getter]
    fn total_samples(&self) -> usize {
        self.inner.total_samples
    }
    /// Corrected pixels in the largest connected region.
    #[getter]
    fn largest_connected_pixels(&self) -> usize {
        self.inner.largest_connected_pixels
    }
    #[getter]
    fn minimum_response(&self) -> f32 {
        self.inner.minimum_response
    }
    /// The largest gain the response applies.
    #[getter]
    fn maximum_applied_gain(&self) -> f32 {
        self.inner.maximum_applied_gain
    }

    fn __repr__(&self) -> String {
        format!(
            "ResidualFlatDiagnostics(sample_count={}, corrected_samples={}, maximum_applied_gain={})",
            self.inner.sample_count, self.inner.corrected_samples, self.inner.maximum_applied_gain
        )
    }
}

/// Build a residual flat response from crops of lights that share one
/// place on the detector. Returns the response, shaped as one crop, and its
/// diagnostics.
#[pyfunction]
#[pyo3(signature = (
    samples,
    *,
    minimum_samples=None,
    minimum_depth=None,
    minimum_consensus=None,
    maximum_gain=None,
    background_edge_fraction=None,
    smoothing_sigma=None,
    edge_feather_fraction=None,
    minimum_corrected_samples=None,
    minimum_connected_pixels=None
))]
#[allow(clippy::too_many_arguments)]
fn build_residual_flat<'py>(
    py: Python<'py>,
    samples: Vec<PyReadonlyArrayDyn<'py, f32>>,
    minimum_samples: Option<usize>,
    minimum_depth: Option<f32>,
    minimum_consensus: Option<f32>,
    maximum_gain: Option<f32>,
    background_edge_fraction: Option<f32>,
    smoothing_sigma: Option<f32>,
    edge_feather_fraction: Option<f32>,
    minimum_corrected_samples: Option<usize>,
    minimum_connected_pixels: Option<usize>,
) -> PyResult<(Bound<'py, PyArrayDyn<f32>>, PyResidualFlatDiagnostics)> {
    let mut options = ResidualFlatOptions::default();
    for (given, target) in [
        (minimum_samples, &mut options.minimum_samples),
        (
            minimum_corrected_samples,
            &mut options.minimum_corrected_samples,
        ),
        (
            minimum_connected_pixels,
            &mut options.minimum_connected_pixels,
        ),
    ] {
        if let Some(value) = given {
            *target = value;
        }
    }
    for (given, target) in [
        (minimum_depth, &mut options.minimum_depth),
        (minimum_consensus, &mut options.minimum_consensus),
        (maximum_gain, &mut options.maximum_gain),
        (
            background_edge_fraction,
            &mut options.background_edge_fraction,
        ),
        (smoothing_sigma, &mut options.smoothing_sigma),
        (edge_feather_fraction, &mut options.edge_feather_fraction),
    ] {
        if let Some(value) = given {
            *target = value;
        }
    }
    let crops = samples
        .iter()
        .map(float_image_view)
        .collect::<PyResult<Vec<_>>>()?;
    let views = crops
        .iter()
        .map(|crop| LinearImageRef::new(crop.data, crop.width, crop.height, crop.channels))
        .collect::<Result<Vec<_>, _>>()
        .map_err(value_error)?;
    let built = py
        .detach(|| seiza_calibration::build_residual_flat_patch(&views, &options))
        .map_err(value_error)?;
    let diagnostics = PyResidualFlatDiagnostics {
        inner: built.diagnostics,
    };
    let (width, height, channels, response) = built.patch.into_parts();
    Ok((
        float_array(py, width, height, channels, response)?,
        diagnostics,
    ))
}

/// Check a residual flat response: every value finite and in (0, 1].
/// Raises ``ValueError`` when one is not.
#[pyfunction]
fn validate_residual_flat_response(response: PyReadonlyArrayDyn<'_, f32>) -> PyResult<()> {
    let response = float_image_view(&response)?;
    let response = LinearImageRef::new(
        response.data,
        response.width,
        response.height,
        response.channels,
    )
    .map_err(value_error)?;
    seiza_calibration::validate_residual_flat_response(response).map_err(value_error)
}

/// Divide a residual flat response out of a light, with the response's
/// top-left corner at pixel (``x``, ``y``). Returns the corrected copy.
#[pyfunction]
fn apply_residual_flat_response<'py>(
    py: Python<'py>,
    response: PyReadonlyArrayDyn<'py, f32>,
    image: PyReadonlyArrayDyn<'py, f32>,
    x: usize,
    y: usize,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let response = float_image_view(&response)?;
    let image = float_image_view(&image)?;
    let mut corrected = image.data.to_vec();
    let response = LinearImageRef::new(
        response.data,
        response.width,
        response.height,
        response.channels,
    )
    .map_err(value_error)?;
    let light = LinearImageMut::new(&mut corrected, image.width, image.height, image.channels)
        .map_err(value_error)?;
    py.detach(|| seiza_calibration::apply_residual_flat_response_at(response, light, x, y))
        .map_err(value_error)?;
    float_array(py, image.width, image.height, image.channels, corrected)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDarkMeasurement>()?;
    module.add_class::<PyIlluminationProfile>()?;
    module.add_class::<PyFlatFeature>()?;
    module.add_class::<PyFlatMeasurement>()?;
    module.add_class::<PyFinding>()?;
    module.add_class::<PyFrameVerdict>()?;
    module.add_class::<PyFlatSetReport>()?;
    module.add_class::<PyResidualFlatDiagnostics>()?;
    module.add_function(wrap_pyfunction!(measure_dark, module)?)?;
    module.add_function(wrap_pyfunction!(measure_flat, module)?)?;
    module.add_function(wrap_pyfunction!(screen_bias, module)?)?;
    module.add_function(wrap_pyfunction!(screen_darks, module)?)?;
    module.add_function(wrap_pyfunction!(screen_flats, module)?)?;
    module.add_function(wrap_pyfunction!(build_residual_flat, module)?)?;
    module.add_function(wrap_pyfunction!(validate_residual_flat_response, module)?)?;
    module.add_function(wrap_pyfunction!(apply_residual_flat_response, module)?)?;
    Ok(())
}
