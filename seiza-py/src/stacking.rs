use crate::arrays::{image_array, linear_image, u32_array};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use seiza_stacking::{
    BatchStackOptions, CalibrationMasters, CancelSignal, DarkLevelScreening, DeltaSigmaOptions,
    Demosaic, DrizzleOptions, DrizzleResult, FitsFrame, FlatStarMaskingOptions,
    FlatStarMaskingStatistics, FrameDisposition, FrameWeighting, LinearImage, LiveStacker,
    MasterBuildOptions, MasterDark, MasterFrameKind, MasterRejectionOptions, NormalizationMode,
    PipelineOptions, PipelineReport, RegistrationModel, RejectionMode, SnrSample, StackOptions,
    StackSnapshot, build_master_from_fits, build_master_from_fits_with_scratch, checkpoint_depths,
    measure_depth, path_identity, paths_refer_to_same_file, write_drizzle_fits_f32, write_fits_f32,
    write_master_fits_f32,
};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pyo3::create_exception!(
    seiza,
    StackError,
    PyRuntimeError,
    "Calibration, registration, or stacking failed."
);

fn stack_error(error: seiza_stacking::Error) -> PyErr {
    StackError::new_err(error.to_string())
}

/// What one pipelined run did.
#[pyclass(skip_from_py_object, frozen, name = "PipelineReport", module = "seiza")]
#[derive(Clone, Copy)]
pub(crate) struct PyPipelineReport {
    /// Frames the accumulator took.
    #[pyo3(get)]
    integrated: usize,
    /// Frames an admission gate turned away.
    #[pyo3(get)]
    rejected: usize,
    /// Paths that could not be read, or repeated one already stacked.
    #[pyo3(get)]
    failed: usize,
}

#[pymethods]
impl PyPipelineReport {
    fn __repr__(&self) -> String {
        format!(
            "PipelineReport(integrated={}, rejected={}, failed={})",
            self.integrated, self.rejected, self.failed
        )
    }
}

impl PyPipelineReport {
    fn from_rust(report: PipelineReport) -> Self {
        Self {
            integrated: report.integrated,
            rejected: report.rejected,
            failed: report.failed,
        }
    }
}

#[pyclass(skip_from_py_object, frozen, name = "StackOptions", module = "seiza")]
#[derive(Clone, Default)]
pub(crate) struct PyStackOptions {
    inner: StackOptions,
}

#[pymethods]
impl PyStackOptions {
    #[new]
    #[pyo3(signature = (
        *,
        normalization="global",
        local_tile_size=256,
        rejection="delta-sigma",
        sigma_low=3.0,
        sigma_high=3.0,
        rejection_warmup=5,
        rejection_minimum_sigma=1.0e-6,
        detection_sigma=4.0,
        maximum_stars=200,
        triangle_stars=24,
        descriptor_tolerance=0.015,
        scale_tolerance=0.08,
        match_tolerance_pixels=2.5,
        maximum_drift_pixels=256.0,
        maximum_drift_fraction=0.15,
        minimum_matches=6,
        maximum_candidates=384,
        maximum_registration_rms=2.0,
        maximum_scale_deviation=0.04,
        maximum_rotation_degrees=10.0,
        minimum_overlap=0.60,
        minimum_normalization_gain=0.25,
        maximum_normalization_gain=4.0,
        minimum_integrated_fraction=0.50,
        demosaic="vng",
        match_tolerances=None,
        registration_model="similarity",
        weighting="equal"
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        normalization: &str,
        local_tile_size: usize,
        rejection: &str,
        sigma_low: f32,
        sigma_high: f32,
        rejection_warmup: u32,
        rejection_minimum_sigma: f32,
        detection_sigma: f32,
        maximum_stars: usize,
        triangle_stars: usize,
        descriptor_tolerance: f64,
        scale_tolerance: f64,
        match_tolerance_pixels: f64,
        maximum_drift_pixels: f64,
        maximum_drift_fraction: f64,
        minimum_matches: usize,
        maximum_candidates: usize,
        maximum_registration_rms: f64,
        maximum_scale_deviation: f64,
        maximum_rotation_degrees: f64,
        minimum_overlap: f32,
        minimum_normalization_gain: f32,
        maximum_normalization_gain: f32,
        minimum_integrated_fraction: f32,
        demosaic: &str,
        match_tolerances: Option<crate::calibration::PyMatchTolerances>,
        registration_model: &str,
        weighting: &str,
    ) -> PyResult<Self> {
        let weighting = match weighting {
            "equal" => FrameWeighting::Equal,
            "inverse-noise-variance" | "inverse_noise_variance" => {
                FrameWeighting::inverse_noise_variance()
            }
            value => {
                return Err(PyValueError::new_err(format!(
                    "weighting must be 'equal' or 'inverse-noise-variance', not {value:?}"
                )));
            }
        };
        let registration_model = match registration_model {
            "similarity" => RegistrationModel::Similarity,
            "affine" => RegistrationModel::Affine,
            "quadratic" => RegistrationModel::Quadratic,
            value => {
                return Err(PyValueError::new_err(format!(
                    "registration_model must be 'similarity', 'affine', or 'quadratic', \
                     not {value:?}"
                )));
            }
        };
        let demosaic = match demosaic {
            "vng" => Demosaic::Vng,
            "mhc" => Demosaic::Mhc,
            "bilinear" => Demosaic::Bilinear,
            value => {
                return Err(PyValueError::new_err(format!(
                    "demosaic must be 'vng', 'mhc', or 'bilinear', not {value:?}"
                )));
            }
        };
        let normalization = match normalization {
            "none" => NormalizationMode::None,
            "global" => NormalizationMode::Global,
            "local" => NormalizationMode::Local {
                tile_size: local_tile_size,
            },
            "local-background" | "local_background" => NormalizationMode::LocalBackground {
                tile_size: local_tile_size,
            },
            value => {
                return Err(PyValueError::new_err(format!(
                    "normalization must be 'none', 'global', 'local', or 'local-background', \
                     not {value:?}"
                )));
            }
        };
        let rejection = match rejection {
            "none" => RejectionMode::None,
            "delta-sigma" | "delta_sigma" => RejectionMode::DeltaSigma(DeltaSigmaOptions {
                low_sigma: sigma_low,
                high_sigma: sigma_high,
                warmup_samples: rejection_warmup,
                minimum_sigma: rejection_minimum_sigma,
            }),
            value => {
                return Err(PyValueError::new_err(format!(
                    "rejection must be 'none' or 'delta-sigma', not {value:?}"
                )));
            }
        };
        let mut inner = StackOptions {
            normalization,
            rejection,
            weighting,
            ..StackOptions::default()
        };
        inner.registration.detection_sigma = detection_sigma;
        inner.registration.maximum_stars = maximum_stars;
        inner.registration.triangle_stars = triangle_stars;
        inner.registration.descriptor_tolerance = descriptor_tolerance;
        inner.registration.scale_tolerance = scale_tolerance;
        inner.registration.match_tolerance_pixels = match_tolerance_pixels;
        inner.registration.maximum_drift_pixels = maximum_drift_pixels;
        inner.registration.maximum_drift_fraction = maximum_drift_fraction;
        inner.registration.minimum_matches = minimum_matches;
        inner.registration.maximum_candidates = maximum_candidates;
        inner.registration.model = registration_model;
        inner.acceptance.maximum_registration_rms_pixels = maximum_registration_rms;
        inner.acceptance.maximum_scale_deviation = maximum_scale_deviation;
        inner.acceptance.maximum_rotation_degrees = maximum_rotation_degrees;
        inner.acceptance.minimum_overlap_fraction = minimum_overlap;
        inner.acceptance.minimum_normalization_gain = minimum_normalization_gain;
        inner.acceptance.maximum_normalization_gain = maximum_normalization_gain;
        inner.acceptance.minimum_integrated_fraction = minimum_integrated_fraction;
        inner.demosaic = demosaic;
        inner.match_tolerances = crate::calibration::tolerances_or_default(match_tolerances);
        inner
            .validate()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    fn normalization(&self) -> &'static str {
        match self.inner.normalization {
            NormalizationMode::None => "none",
            NormalizationMode::Global => "global",
            NormalizationMode::Local { .. } => "local",
            NormalizationMode::LocalBackground { .. } => "local-background",
        }
    }

    /// How Bayer frames are demosaiced: "vng", "mhc", or "bilinear".
    #[getter]
    fn demosaic(&self) -> &'static str {
        match self.inner.demosaic {
            Demosaic::Vng => "vng",
            Demosaic::Mhc => "mhc",
            Demosaic::Bilinear => "bilinear",
        }
    }

    /// Geometry fitted after the similarity match: "similarity", "affine",
    /// or "quadratic", which follows lens distortion.
    #[getter]
    fn registration_model(&self) -> &'static str {
        match self.inner.registration.model {
            RegistrationModel::Similarity => "similarity",
            RegistrationModel::Affine => "affine",
            RegistrationModel::Quadratic => "quadratic",
        }
    }

    #[getter]
    fn local_tile_size(&self) -> Option<usize> {
        match self.inner.normalization {
            NormalizationMode::Local { tile_size }
            | NormalizationMode::LocalBackground { tile_size } => Some(tile_size),
            _ => None,
        }
    }

    /// How frames are weighted in the mean: "equal", or
    /// "inverse-noise-variance", which weights each frame by the reference
    /// frame's noise over its own, squared.
    #[getter]
    fn weighting(&self) -> &'static str {
        if self.inner.weighting.is_equal() {
            "equal"
        } else {
            "inverse-noise-variance"
        }
    }

    #[getter]
    fn rejection(&self) -> &'static str {
        match self.inner.rejection {
            RejectionMode::None => "none",
            RejectionMode::DeltaSigma(_) => "delta-sigma",
        }
    }

    #[getter]
    fn maximum_drift_pixels(&self) -> f64 {
        self.inner.registration.maximum_drift_pixels
    }

    #[getter]
    fn maximum_drift_fraction(&self) -> f64 {
        self.inner.registration.maximum_drift_fraction
    }

    /// How close a light's readings must be to its masters' for them to
    /// calibrate it: the gate a stack opened with these options judges its
    /// reference and every pushed light under.
    #[getter]
    fn match_tolerances(&self) -> crate::calibration::PyMatchTolerances {
        crate::calibration::PyMatchTolerances {
            inner: self.inner.match_tolerances,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "StackOptions(normalization={:?}, rejection={:?}, maximum_drift_pixels={:.1}, maximum_drift_fraction={:.3})",
            self.normalization(),
            self.rejection(),
            self.maximum_drift_pixels(),
            self.maximum_drift_fraction(),
        )
    }
}

#[pyclass(
    skip_from_py_object,
    frozen,
    name = "FrameDisposition",
    module = "seiza"
)]
#[derive(Clone)]
pub(crate) struct PyFrameDisposition {
    #[pyo3(get)]
    source: Option<PathBuf>,
    #[pyo3(get)]
    accepted: bool,
    #[pyo3(get)]
    reason: Option<String>,
    #[pyo3(get)]
    matched_stars: Option<usize>,
    #[pyo3(get)]
    registration_rms_pixels: Option<f64>,
    #[pyo3(get)]
    registration_drift_pixels: Option<f64>,
    #[pyo3(get)]
    scale: Option<f64>,
    #[pyo3(get)]
    rotation_degrees: Option<f64>,
    #[pyo3(get)]
    translation_x: Option<f64>,
    #[pyo3(get)]
    translation_y: Option<f64>,
    #[pyo3(get)]
    normalization_mean_gain: Option<f32>,
    #[pyo3(get)]
    normalization_mean_offset: Option<f32>,
    #[pyo3(get)]
    overlap_fraction: Option<f32>,
    #[pyo3(get)]
    integrated_fraction: Option<f32>,
    #[pyo3(get)]
    accepted_samples: Option<usize>,
    #[pyo3(get)]
    rejected_samples: Option<usize>,
    /// Each channel's measured noise; empty under equal weighting.
    #[pyo3(get)]
    noise: Option<Vec<f32>>,
    /// Each channel's weight in the stack mean; empty under equal weighting.
    #[pyo3(get)]
    weight: Option<Vec<f32>>,
    /// The core's JSON of the frame's mapping, read by the `mapping` getter.
    mapping_json: Option<String>,
}

impl PyFrameDisposition {
    fn from_rust(source: Option<PathBuf>, disposition: FrameDisposition) -> Self {
        match disposition {
            FrameDisposition::Accepted(diagnostics) => Self {
                source,
                accepted: true,
                reason: None,
                matched_stars: Some(diagnostics.matched_stars),
                registration_rms_pixels: Some(diagnostics.registration_rms_pixels),
                registration_drift_pixels: Some(diagnostics.registration_drift_pixels),
                scale: Some(diagnostics.transform.scale),
                rotation_degrees: Some(diagnostics.transform.rotation_radians.to_degrees()),
                translation_x: Some(diagnostics.transform.translation_x),
                translation_y: Some(diagnostics.transform.translation_y),
                normalization_mean_gain: Some(diagnostics.normalization_mean_gain),
                normalization_mean_offset: Some(diagnostics.normalization_mean_offset),
                overlap_fraction: Some(diagnostics.overlap_fraction),
                integrated_fraction: Some(diagnostics.integrated_fraction),
                accepted_samples: Some(diagnostics.accepted_samples),
                rejected_samples: Some(diagnostics.rejected_samples),
                mapping_json: serde_json::to_string(&diagnostics.mapping).ok(),
                noise: Some(diagnostics.noise),
                weight: Some(diagnostics.weight),
            },
            FrameDisposition::Rejected(reason) => Self {
                source,
                accepted: false,
                reason: Some(reason.to_string()),
                matched_stars: None,
                registration_rms_pixels: None,
                registration_drift_pixels: None,
                scale: None,
                rotation_degrees: None,
                translation_x: None,
                translation_y: None,
                normalization_mean_gain: None,
                normalization_mean_offset: None,
                overlap_fraction: None,
                integrated_fraction: None,
                accepted_samples: None,
                rejected_samples: None,
                noise: None,
                weight: None,
                mapping_json: None,
            },
        }
    }

    /// A path the pipeline could not use at all — unreadable, or a repeat of
    /// one already stacked. Reported in place rather than raising, so one bad
    /// path in a night's listing does not lose the rest.
    fn failed(source: PathBuf, reason: String) -> Self {
        Self {
            source: Some(source),
            accepted: false,
            reason: Some(reason),
            matched_stars: None,
            registration_rms_pixels: None,
            registration_drift_pixels: None,
            scale: None,
            rotation_degrees: None,
            translation_x: None,
            translation_y: None,
            normalization_mean_gain: None,
            normalization_mean_offset: None,
            overlap_fraction: None,
            integrated_fraction: None,
            accepted_samples: None,
            rejected_samples: None,
            noise: None,
            weight: None,
            mapping_json: None,
        }
    }
}

/// Turn JSON the core wrote into Python objects.
fn json_object(py: Python<'_>, json: &str) -> PyResult<Py<PyAny>> {
    Ok(py.import("json")?.call_method1("loads", (json,))?.unbind())
}

#[pymethods]
impl PyFrameDisposition {
    fn __bool__(&self) -> bool {
        self.accepted
    }

    /// How an accepted frame maps onto the reference grid, its
    /// registration and normalization, as a dict in the core's versioned
    /// form; `None` for a rejected frame.
    #[getter]
    fn mapping(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.mapping_json
            .as_deref()
            .map(|json| json_object(py, json))
            .transpose()
    }

    fn __repr__(&self) -> String {
        if self.accepted {
            format!(
                "FrameDisposition(accepted=True, matched_stars={}, rms_pixels={:.3})",
                self.matched_stars.unwrap_or_default(),
                self.registration_rms_pixels.unwrap_or_default(),
            )
        } else {
            format!(
                "FrameDisposition(accepted=False, reason={:?})",
                self.reason.as_deref().unwrap_or("unknown rejection")
            )
        }
    }
}

#[pyclass(frozen, name = "StackResult", module = "seiza")]
pub(crate) struct PyStackResult {
    #[pyo3(get)]
    output: PathBuf,
    #[pyo3(get)]
    accepted_frames: u32,
    #[pyo3(get)]
    rejected_frames: u32,
    #[pyo3(get)]
    width: usize,
    #[pyo3(get)]
    height: usize,
    #[pyo3(get)]
    channels: usize,
    /// The light that set normalization: the stack's reference frame, or
    /// with a reference image, the channel reference.
    #[pyo3(get)]
    reference_light: PathBuf,
    frames: Vec<PyFrameDisposition>,
}

#[pymethods]
impl PyStackResult {
    #[getter]
    fn frames(&self, py: Python<'_>) -> PyResult<Vec<Py<PyFrameDisposition>>> {
        self.frames
            .iter()
            .cloned()
            .map(|frame| Py::new(py, frame))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "StackResult(output={:?}, accepted_frames={}, rejected_frames={}, shape={}x{}x{})",
            self.output,
            self.accepted_frames,
            self.rejected_frames,
            self.width,
            self.height,
            self.channels,
        )
    }
}

/// How good a frame would be as the stack's reference; see
/// :func:`choose_reference`.
#[pyclass(skip_from_py_object, frozen, name = "ReferenceScore", module = "seiza")]
#[derive(Clone)]
pub(crate) struct PyReferenceScore {
    /// Stars detected at five sigma on the half-resolution luminance.
    #[pyo3(get)]
    stars: usize,
    /// Median star area in half-resolution pixels; smaller is sharper.
    #[pyo3(get)]
    median_star_area: f32,
    /// Median sky level of the half-resolution luminance.
    #[pyo3(get)]
    background: f32,
    /// How far the sky departs from flat; cloud and twilight raise it.
    #[pyo3(get)]
    background_variation: f32,
    /// The bright stars' signal over the sky noise; higher is better.
    #[pyo3(get)]
    score: f32,
}

impl From<seiza_stacking::ReferenceScore> for PyReferenceScore {
    fn from(score: seiza_stacking::ReferenceScore) -> Self {
        Self {
            stars: score.stars,
            median_star_area: score.median_star_area,
            background: score.background,
            background_variation: score.background_variation,
            score: score.score,
        }
    }
}

#[pymethods]
impl PyReferenceScore {
    fn __repr__(&self) -> String {
        format!(
            "ReferenceScore(score={:.3}, stars={}, median_star_area={:.1}, background_variation={:.4})",
            self.score, self.stars, self.median_star_area, self.background_variation,
        )
    }
}

/// The reference :func:`choose_reference` chose, with every frame's score.
#[pyclass(frozen, name = "ReferenceChoice", module = "seiza")]
pub(crate) struct PyReferenceChoice {
    /// The chosen frame's index in the list given.
    #[pyo3(get)]
    reference_index: usize,
    /// The chosen frame's path.
    #[pyo3(get)]
    reference_path: PathBuf,
    /// Every scored frame's index, best reference first.
    #[pyo3(get)]
    ranking: Vec<usize>,
    scores: Vec<Option<PyReferenceScore>>,
}

#[pymethods]
impl PyReferenceChoice {
    /// One score per frame, in the order given; ``None`` for a frame that
    /// could not be read or scored.
    #[getter]
    fn scores(&self, py: Python<'_>) -> PyResult<Vec<Option<Py<PyReferenceScore>>>> {
        self.scores
            .iter()
            .map(|score| score.clone().map(|score| Py::new(py, score)).transpose())
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "ReferenceChoice(reference_index={}, reference_path={:?}, ranking={:?})",
            self.reference_index, self.reference_path, self.ranking,
        )
    }
}

/// The channel reference
/// :meth:`LiveStacker.open_fits_with_reference_image_candidates` used.
#[pyclass(frozen, name = "ChannelReferenceSelection", module = "seiza")]
pub(crate) struct PyChannelReferenceSelection {
    /// Its index in the list given.
    #[pyo3(get)]
    index: usize,
    /// Its path.
    #[pyo3(get)]
    path: PathBuf,
    /// ``(index, path, reason)`` for each earlier light that did not align.
    #[pyo3(get)]
    skipped: Vec<(usize, PathBuf, String)>,
}

#[pymethods]
impl PyChannelReferenceSelection {
    fn __repr__(&self) -> String {
        format!(
            "ChannelReferenceSelection(index={}, path={:?}, skipped={})",
            self.index,
            self.path,
            self.skipped.len(),
        )
    }
}

/// With ``reference="auto"`` and a reference image, how many of the best
/// lights :func:`stack_fits` tries in turn, as the CLI does.
const CHANNEL_REFERENCE_ATTEMPTS: usize = 5;

#[pyclass(frozen, name = "StackSnapshot", module = "seiza")]
pub(crate) struct PyStackSnapshot {
    inner: StackSnapshot,
}

#[pymethods]
impl PyStackSnapshot {
    #[getter]
    fn width(&self) -> usize {
        self.inner.image.width
    }

    #[getter]
    fn height(&self) -> usize {
        self.inner.image.height
    }

    #[getter]
    fn channels(&self) -> usize {
        self.inner.image.channels
    }

    #[getter]
    fn accepted_frames(&self) -> u32 {
        self.inner.accepted_frames
    }

    #[getter]
    fn rejected_frames(&self) -> u32 {
        self.inner.rejected_frames
    }

    /// Return a copy of the current linear mean as a 2D mono or HWC RGB array.
    #[getter]
    fn image<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        image_array(py, &self.inner.image)
    }

    /// Return a copy of the per-sample variance.
    #[getter]
    fn variance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        image_array(py, &self.inner.variance)
    }

    /// Return a copy of the accepted-sample count map.
    #[getter]
    fn coverage<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<u32>>> {
        u32_array(
            py,
            &self.inner.coverage,
            self.inner.image.width,
            self.inner.image.height,
            self.inner.image.channels,
        )
    }

    /// Return a copy of the rejected-sample count map.
    #[getter]
    fn rejected_samples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<u32>>> {
        u32_array(
            py,
            &self.inner.rejected_samples,
            self.inner.image.width,
            self.inner.image.height,
            self.inner.image.channels,
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "StackSnapshot(shape={}x{}x{}, accepted_frames={}, rejected_frames={})",
            self.width(),
            self.height(),
            self.channels(),
            self.accepted_frames(),
            self.rejected_frames(),
        )
    }
}

/// A drizzled stack from :meth:`LiveStacker.reintegrate_drizzled`.
#[pyclass(frozen, name = "DrizzleResult", module = "seiza")]
pub(crate) struct PyDrizzleResult {
    inner: DrizzleResult,
    accepted_frames: u32,
    rejected_frames: u32,
    reference_headers: Vec<(String, seiza_fits::HeaderValue)>,
    input_paths: Vec<PathBuf>,
}

#[pymethods]
impl PyDrizzleResult {
    #[getter]
    fn width(&self) -> usize {
        self.inner.image.width
    }

    #[getter]
    fn height(&self) -> usize {
        self.inner.image.height
    }

    #[getter]
    fn channels(&self) -> usize {
        self.inner.image.channels
    }

    /// Output pixels per reference pixel along each axis.
    #[getter]
    fn scale(&self) -> u32 {
        self.inner.scale
    }

    #[getter]
    fn accepted_frames(&self) -> u32 {
        self.accepted_frames
    }

    #[getter]
    fn rejected_frames(&self) -> u32 {
        self.rejected_frames
    }

    /// Return a copy of the drizzled linear mean as a 2D mono or HWC RGB
    /// array; samples no drop reached are NaN.
    #[getter]
    fn image<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        image_array(py, &self.inner.image)
    }

    /// Return a copy of the total weight behind each output sample: drop
    /// area in output pixels times frame weight.
    #[getter]
    fn weight<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        image_array(py, &self.inner.weight)
    }

    /// Write the drizzled image as linear FITS (or XISF for a ``.xisf``
    /// path), with the reference WCS scaled to the drizzle grid.
    fn write_fits(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        if self
            .input_paths
            .iter()
            .any(|input| paths_refer_to_same_file(input, &path))
        {
            return Err(PyValueError::new_err(
                "output path must not refer to a stack input or calibration master",
            ));
        }
        py.detach(|| {
            write_drizzle_fits_f32(
                path,
                &self.inner,
                self.accepted_frames,
                self.rejected_frames,
                &self.reference_headers,
            )
        })
        .map_err(stack_error)
    }

    fn __repr__(&self) -> String {
        format!(
            "DrizzleResult(shape={}x{}x{}, scale={}, accepted_frames={})",
            self.width(),
            self.height(),
            self.channels(),
            self.scale(),
            self.accepted_frames,
        )
    }
}

fn batch_options(
    sigma_low: f32,
    sigma_high: f32,
    scratch_directory: Option<PathBuf>,
) -> PyResult<BatchStackOptions> {
    for sigma in [sigma_low, sigma_high] {
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(PyValueError::new_err("rejection sigmas must be positive"));
        }
    }
    Ok(BatchStackOptions {
        rejection: MasterRejectionOptions {
            low_sigma: sigma_low,
            high_sigma: sigma_high,
        },
        scratch_directory,
        ..BatchStackOptions::default()
    })
}

#[pyclass(name = "LiveStacker", module = "seiza")]
pub(crate) struct PyLiveStacker {
    inner: Option<LiveStacker>,
}

#[pymethods]
impl PyLiveStacker {
    #[new]
    #[pyo3(signature = (
        reference,
        *,
        options=None,
        bias=None,
        dark=None,
        flat=None,
        dark_exposure_seconds=None
    ))]
    fn new(
        py: Python<'_>,
        reference: PathBuf,
        options: Option<PyRef<'_, PyStackOptions>>,
        bias: Option<PathBuf>,
        dark: Option<PathBuf>,
        flat: Option<PathBuf>,
        dark_exposure_seconds: Option<f64>,
    ) -> PyResult<Self> {
        validate_exposure_override(dark.as_ref(), dark_exposure_seconds, "dark")?;
        let input_paths = [
            Some(reference.clone()),
            bias.clone(),
            dark.clone(),
            flat.clone(),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        validate_distinct_paths(&input_paths)?;
        let options = options
            .as_ref()
            .map_or_else(StackOptions::default, |options| options.inner.clone());
        let inner = py
            .detach(move || {
                LiveStacker::open_fits(
                    reference,
                    bias.as_deref(),
                    dark.as_deref(),
                    flat.as_deref(),
                    dark_exposure_seconds,
                    options,
                )
            })
            .map_err(stack_error)?;
        Ok(Self { inner: Some(inner) })
    }

    #[staticmethod]
    #[pyo3(signature = (reference, *, options=None))]
    fn from_array(
        py: Python<'_>,
        reference: PyReadonlyArrayDyn<'_, f32>,
        options: Option<PyRef<'_, PyStackOptions>>,
    ) -> PyResult<Self> {
        let reference = linear_image(reference)?;
        let options = options
            .as_ref()
            .map_or_else(StackOptions::default, |options| options.inner.clone());
        let inner = py
            .detach(move || LiveStacker::from_linear(reference, options))
            .map_err(stack_error)?;
        Ok(Self { inner: Some(inner) })
    }

    /// Open a stack from a FITS or XISF frame already calibrated and
    /// debayered, keeping its header cards for the outputs, so a caller can
    /// correct a frame between calibration and registration. A raw CFA frame
    /// is refused. Later frames go in as arrays, through :meth:`push`.
    #[staticmethod]
    #[pyo3(signature = (path, *, options=None))]
    fn open_prepared_fits(
        py: Python<'_>,
        path: PathBuf,
        options: Option<PyRef<'_, PyStackOptions>>,
    ) -> PyResult<Self> {
        let options = options
            .as_ref()
            .map_or_else(StackOptions::default, |options| options.inner.clone());
        let inner = py
            .detach(move || LiveStacker::from_prepared_frame(FitsFrame::open(path)?, options))
            .map_err(stack_error)?;
        Ok(Self { inner: Some(inner) })
    }

    /// Open a stack aligned to ``reference_image``, an image shared by the
    /// stacks of several filters so their outputs share one pixel grid and
    /// WCS. ``channel_reference``, one of this filter's lights, sets
    /// normalization and noise weighting. The stack starts empty: push the
    /// channel reference like every other light. The reference image adds no
    /// pixels unless it is pushed as well, and outputs may not replace it.
    #[staticmethod]
    #[pyo3(signature = (
        reference_image,
        channel_reference,
        *,
        options=None,
        bias=None,
        dark=None,
        flat=None,
        dark_exposure_seconds=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn open_fits_with_reference_image(
        py: Python<'_>,
        reference_image: PathBuf,
        channel_reference: PathBuf,
        options: Option<PyRef<'_, PyStackOptions>>,
        bias: Option<PathBuf>,
        dark: Option<PathBuf>,
        flat: Option<PathBuf>,
        dark_exposure_seconds: Option<f64>,
    ) -> PyResult<Self> {
        validate_exposure_override(dark.as_ref(), dark_exposure_seconds, "dark")?;
        for image in [&reference_image, &channel_reference] {
            let paths = std::iter::once(image.clone())
                .chain(
                    [bias.clone(), dark.clone(), flat.clone()]
                        .into_iter()
                        .flatten(),
                )
                .collect::<Vec<_>>();
            validate_distinct_paths(&paths)?;
        }
        let options = options
            .as_ref()
            .map_or_else(StackOptions::default, |options| options.inner.clone());
        let inner = py
            .detach(move || {
                LiveStacker::open_fits_with_reference_image(
                    reference_image,
                    channel_reference,
                    bias.as_deref(),
                    dark.as_deref(),
                    flat.as_deref(),
                    dark_exposure_seconds,
                    options,
                )
            })
            .map_err(stack_error)?;
        Ok(Self { inner: Some(inner) })
    }

    /// :meth:`open_fits_with_reference_image` with ``channel_references``,
    /// this filter's lights best first, tried in turn until one aligns to
    /// the reference image, which is read and prepared once. For an
    /// automatic choice, pass the first few paths of
    /// :func:`choose_reference`'s ``ranking``. A light that cannot be read or
    /// calibrated, or whose channel count differs from the reference
    /// image's, raises at once, since the next light would fail the same way.
    ///
    /// Returns the stack and a :class:`ChannelReferenceSelection` naming the
    /// light it used and why each earlier one was skipped. Push every light,
    /// the channel reference included.
    #[staticmethod]
    #[pyo3(signature = (
        reference_image,
        channel_references,
        *,
        options=None,
        bias=None,
        dark=None,
        flat=None,
        dark_exposure_seconds=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn open_fits_with_reference_image_candidates(
        py: Python<'_>,
        reference_image: PathBuf,
        channel_references: Vec<PathBuf>,
        options: Option<PyRef<'_, PyStackOptions>>,
        bias: Option<PathBuf>,
        dark: Option<PathBuf>,
        flat: Option<PathBuf>,
        dark_exposure_seconds: Option<f64>,
    ) -> PyResult<(Self, PyChannelReferenceSelection)> {
        if channel_references.is_empty() {
            return Err(PyValueError::new_err(
                "at least one channel reference is required",
            ));
        }
        validate_exposure_override(dark.as_ref(), dark_exposure_seconds, "dark")?;
        for image in std::iter::once(&reference_image).chain(&channel_references) {
            let paths = std::iter::once(image.clone())
                .chain(
                    [bias.clone(), dark.clone(), flat.clone()]
                        .into_iter()
                        .flatten(),
                )
                .collect::<Vec<_>>();
            validate_distinct_paths(&paths)?;
        }
        let options = options
            .as_ref()
            .map_or_else(StackOptions::default, |options| options.inner.clone());
        let (inner, used, skipped) = py
            .detach(|| {
                let mut skipped = Vec::new();
                LiveStacker::open_fits_with_reference_image_candidates(
                    &reference_image,
                    &channel_references,
                    bias.as_deref(),
                    dark.as_deref(),
                    flat.as_deref(),
                    dark_exposure_seconds,
                    options,
                    |index, reason| skipped.push((index, reason.to_string())),
                )
                .map(|(inner, used)| (inner, used, skipped))
            })
            .map_err(stack_error)?;
        let selection = PyChannelReferenceSelection {
            index: used,
            path: channel_references[used].clone(),
            skipped: skipped
                .into_iter()
                .map(|(index, reason)| (index, channel_references[index].clone(), reason))
                .collect(),
        };
        Ok((Self { inner: Some(inner) }, selection))
    }

    /// Plan a canvas that holds a connected set of overlapping lights, such
    /// as mosaic panels or a dithered stack whose edges should be kept, and
    /// start an empty stack on it. The first path anchors the WCS and
    /// photometric scale; put :func:`choose_reference`'s choice first to
    /// anchor on the clearest light. Panels may connect through other
    /// panels, and frames may differ in rotation by any amount. A frame that
    /// cannot join is rejected, with the reason, when pushed; this raises only
    /// when nothing joins the anchor, or the canvas would exceed
    /// ``max_canvas_megasamples`` million samples (pixels times channels; a
    /// run peaks at about 40 bytes per sample).
    ///
    /// With ``registration_model="quadratic"`` in ``options``, a frame lying
    /// mostly within one panel is warped onto it, which follows lens
    /// distortion across a meridian flip; each panel's first frame keeps
    /// its similarity. Use a ``minimum_overlap`` near 0.1 for panels.
    ///
    /// ``cancel``, a callable, is asked before each frame is read; Ctrl-C
    /// also stops planning. The stack starts empty: push every light, the
    /// anchor included.
    #[staticmethod]
    #[pyo3(signature = (
        paths,
        *,
        options=None,
        bias=None,
        dark=None,
        flat=None,
        dark_exposure_seconds=None,
        max_canvas_megasamples=256,
        cancel=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn open_fits_expanding(
        py: Python<'_>,
        paths: Vec<PathBuf>,
        options: Option<PyRef<'_, PyStackOptions>>,
        bias: Option<PathBuf>,
        dark: Option<PathBuf>,
        flat: Option<PathBuf>,
        dark_exposure_seconds: Option<f64>,
        max_canvas_megasamples: usize,
        cancel: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        if paths.is_empty() {
            return Err(PyValueError::new_err("at least one light is required"));
        }
        validate_exposure_override(dark.as_ref(), dark_exposure_seconds, "dark")?;
        for light in &paths {
            let inputs = std::iter::once(light.clone())
                .chain(
                    [bias.clone(), dark.clone(), flat.clone()]
                        .into_iter()
                        .flatten(),
                )
                .collect::<Vec<_>>();
            validate_distinct_paths(&inputs)?;
        }
        let maximum_samples = max_canvas_megasamples
            .checked_mul(1_000_000)
            .filter(|samples| *samples > 0)
            .ok_or_else(|| PyValueError::new_err("max_canvas_megasamples must be positive"))?;
        let options = options
            .as_ref()
            .map_or_else(StackOptions::default, |options| options.inner.clone());
        // Planning reads every light with the GIL released, taking it back
        // before each one to ask whether to stop.
        let raised: Mutex<Option<PyErr>> = Mutex::new(None);
        let planned = py.detach(|| {
            LiveStacker::open_fits_expanding(
                &paths,
                bias.as_deref(),
                dark.as_deref(),
                flat.as_deref(),
                dark_exposure_seconds,
                options,
                maximum_samples,
                |_, _, _, _| {
                    let stop = Python::attach(|py| {
                        let asked = py.check_signals().and_then(|()| match &cancel {
                            Some(cancel) => cancel
                                .call0(py)
                                .and_then(|value| value.bind(py).is_truthy()),
                            None => Ok(false),
                        });
                        asked.unwrap_or_else(|error| {
                            // Re-raised once planning unwinds, so the caller
                            // sees KeyboardInterrupt or their own exception.
                            *raised.lock().unwrap() = Some(error);
                            true
                        })
                    });
                    if stop {
                        seiza_stacking::Continue::No
                    } else {
                        seiza_stacking::Continue::Yes
                    }
                },
            )
        });
        if let Some(error) = raised.lock().unwrap().take() {
            return Err(error);
        }
        Ok(Self {
            inner: Some(planned.map_err(stack_error)?),
        })
    }

    /// Reopen a versioned on-disk context and continue adding frames.
    #[staticmethod]
    fn open_context(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let inner = py
            .detach(|| LiveStacker::open_context(path))
            .map_err(stack_error)?;
        Ok(Self { inner: Some(inner) })
    }

    /// Which of the active masters a prospective light could accept, and why
    /// each refused master was set aside. The question to ask before pushing
    /// a frame in a mode that must warn rather than fail.
    ///
    /// Returns ``(kept, dropped)``: ``kept`` names the master kinds that are
    /// loaded and acceptable (``"bias"``, ``"dark"``, ``"flat"``); ``dropped``
    /// carries one human-readable reason per refused master, each naming the
    /// differing fields and both readings.
    ///
    /// ``tolerances=None`` asks under the stack's own; see
    /// :meth:`match_tolerances`. Given tolerances replace the stack's whole:
    /// a field left unset takes the built-in default, not the stack's value.
    /// Raises :class:`StackError` for a tolerance that is NaN, infinite,
    /// negative, or a rotation above 180 degrees.
    #[pyo3(signature = (light, tolerances=None))]
    fn compatible_calibration(
        &self,
        light: &crate::calibration::PyFrameSignature,
        tolerances: Option<crate::calibration::PyMatchTolerances>,
    ) -> PyResult<(Vec<&'static str>, Vec<String>)> {
        let stacker = self.active()?;
        let tolerances = crate::calibration::tolerances_or(tolerances, stacker.match_tolerances());
        let (kept, dropped) = stacker
            .calibration()
            .compatible_for_light_with(&light.inner, &tolerances)
            .map_err(stack_error)?;
        let mut kinds = Vec::new();
        if kept.has_bias() {
            kinds.push("bias");
        }
        if kept.has_dark() {
            kinds.push("dark");
        }
        if kept.has_flat() {
            kinds.push("flat");
        }
        Ok((kinds, dropped))
    }

    /// Whether every active master accepts a prospective light: the
    /// yes-or-no form of :meth:`compatible_calibration`, asking the same
    /// questions, so it passes exactly when that drops nothing.
    ///
    /// Returns ``None`` when the light is acceptable, and always with no
    /// masters loaded. Raises :class:`StackError` naming why a master
    /// refuses it — the reading and, for a measured gap, the tolerance in
    /// force. A host that widened a gate to choose its masters asks with the
    /// same ``tolerances`` here; ``None`` means the stack's own, and given
    /// tolerances replace them as in :meth:`compatible_calibration`.
    /// Unusable tolerances raise :class:`StackError` too.
    #[pyo3(signature = (light, tolerances=None))]
    fn validate_calibration(
        &self,
        light: &crate::calibration::PyFrameSignature,
        tolerances: Option<crate::calibration::PyMatchTolerances>,
    ) -> PyResult<()> {
        let stacker = self.active()?;
        let tolerances = crate::calibration::tolerances_or(tolerances, stacker.match_tolerances());
        stacker
            .calibration()
            .validate_light_signature_with(&light.inner, &tolerances)
            .map_err(stack_error)
    }

    /// The tolerances lights are judged under now: those the stack opened
    /// with (:attr:`StackOptions.match_tolerances`) or was last given.
    fn match_tolerances(&self) -> PyResult<crate::calibration::PyMatchTolerances> {
        Ok(crate::calibration::PyMatchTolerances {
            inner: self.active()?.match_tolerances(),
        })
    }

    /// Judge the lights pushed from now on under ``tolerances``, or the
    /// defaults when ``None``.
    ///
    /// A host that widened a gate to choose its masters — a rotator that
    /// scatters more than two degrees, or a flat taken from the nearest
    /// other angle on purpose — sets the same gate here, or the light is
    /// refused when it is pushed. Frames already admitted keep the gate they
    /// were admitted under, and :meth:`reintegrate` judges each by its own,
    /// so the gate can be widened for one light and narrowed after it. A
    /// saved context keeps every gate. Raises :class:`StackError` for
    /// tolerances that are negative or not finite, and for a stack started
    /// from prepared pixels.
    #[pyo3(signature = (tolerances=None))]
    fn set_match_tolerances(
        &mut self,
        tolerances: Option<crate::calibration::PyMatchTolerances>,
    ) -> PyResult<()> {
        let tolerances = crate::calibration::tolerances_or_default(tolerances);
        self.active_mut()
            .and_then(|stacker| stacker.set_match_tolerances(tolerances))
            .map_err(stack_error)
    }

    /// Atomically checkpoint this live stack without consuming it.
    fn save_context(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let stacker = self.active()?;
        py.detach(|| stacker.save_context(path))
            .map_err(stack_error)
    }

    #[getter]
    fn accepted_frames(&self) -> PyResult<u32> {
        Ok(self.active()?.view().accepted_frames)
    }

    #[getter]
    fn rejected_frames(&self) -> PyResult<u32> {
        Ok(self.active()?.view().rejected_frames)
    }

    /// False for a stack aligned to a separate reference image, which starts
    /// with no accepted frames.
    #[getter]
    fn reference_integrated(&self) -> PyResult<bool> {
        Ok(self.active()?.reference_is_integrated())
    }

    /// The reference frame's noise in each channel, measured as each pushed
    /// frame's is: the unit of frame weights. Empty under equal weighting.
    #[getter]
    fn reference_noise(&self) -> PyResult<Vec<f32>> {
        Ok(self.active()?.reference_noise().to_vec())
    }

    /// The reference frame's mapping onto the stack grid, the identity, in
    /// the same form as :attr:`FrameDisposition.mapping`.
    #[getter]
    fn reference_mapping(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let json = serde_json::to_string(&self.active()?.reference_mapping())
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        json_object(py, &json)
    }

    fn push_fits(&mut self, py: Python<'_>, path: PathBuf) -> PyResult<PyFrameDisposition> {
        if self
            .active()?
            .input_paths()
            .iter()
            .any(|input| paths_refer_to_same_file(input, &path))
        {
            return Err(PyValueError::new_err(format!(
                "FITS frame {} has already been used by this stack",
                path.display()
            )));
        }
        let disposition = py
            .detach(|| self.active_mut()?.push_fits(&path))
            .map_err(stack_error)?;
        Ok(PyFrameDisposition::from_rust(Some(path), disposition))
    }

    /// Stack many paths at once, preparing several frames in parallel.
    ///
    /// Reads, calibration, registration and normalization overlap across
    /// frames while integration stays in the order given, so the result is
    /// identical to pushing the same paths one at a time — measured 2.0x
    /// faster on local storage and 3.0x with a 300ms read latency.
    ///
    /// Returns the per-frame dispositions in order and a summary. A path that
    /// cannot be read, or that repeats one already stacked, is reported in
    /// place with `accepted` false and a reason, and the run carries on; check
    /// the summary's `failed` count rather than assuming success.
    ///
    /// `workers` is the read concurrency as well as the compute concurrency.
    /// Leave it unset for local frames; raise it when the frames are remote,
    /// since this library cannot tell a network mount from a local disk.
    ///
    /// `normalized_full_scale` puts a frame the file declares as normalized —
    /// `bounds="0:1"`, as PixInsight writes — onto that scale as it is read.
    /// Pass 65535.0 when the other frames are 16-bit camera data, or leave it
    /// unset to keep every sample exactly as stored.
    #[pyo3(signature = (paths, workers=None, max_in_flight_bytes=None, normalized_full_scale=None))]
    fn push_fits_pipelined(
        &mut self,
        py: Python<'_>,
        paths: Vec<PathBuf>,
        workers: Option<usize>,
        max_in_flight_bytes: Option<usize>,
        normalized_full_scale: Option<f32>,
    ) -> PyResult<(Vec<PyFrameDisposition>, PyPipelineReport)> {
        let mut options = PipelineOptions {
            workers,
            normalized_full_scale,
            ..PipelineOptions::default()
        };
        if let Some(bytes) = max_in_flight_bytes {
            options.max_in_flight_bytes = bytes;
        }

        // The callback runs on the calling thread, so collecting here and
        // building the Python objects afterwards keeps the GIL out of the
        // stacking loop entirely.
        let mut outcomes: Vec<(PathBuf, Result<FrameDisposition, String>)> = Vec::new();
        let report = py
            .detach(|| {
                let stacker = self.active_mut()?;
                stacker.push_fits_pipelined(&paths, &options, |path, outcome| {
                    outcomes.push((
                        path.to_path_buf(),
                        outcome.map_err(|error| error.to_string()),
                    ));
                    seiza_stacking::Continue::Yes
                })
            })
            .map_err(stack_error)?;

        let dispositions = outcomes
            .into_iter()
            .map(|(path, outcome)| match outcome {
                Ok(disposition) => PyFrameDisposition::from_rust(Some(path), disposition),
                Err(message) => PyFrameDisposition::failed(path, message),
            })
            .collect();
        Ok((dispositions, PyPipelineReport::from_rust(report)))
    }

    /// Push an already linear, calibrated, and channel-compatible NumPy frame.
    fn push(
        &mut self,
        py: Python<'_>,
        image: PyReadonlyArrayDyn<'_, f32>,
    ) -> PyResult<PyFrameDisposition> {
        let image = linear_image(image)?;
        let disposition = py
            .detach(|| self.active_mut()?.push_linear(image))
            .map_err(stack_error)?;
        Ok(PyFrameDisposition::from_rust(None, disposition))
    }

    /// Copy the current accumulator into an immutable Python snapshot.
    /// Read how deep this stack is, without copying the accumulator.
    ///
    /// Returns ``None`` when the frame is too small or too little of it is
    /// covered to measure — an ordinary answer early in a build, not an error.
    /// Call it at the depths :func:`checkpoint_depths` names.
    fn measure_depth(&self, py: Python<'_>) -> PyResult<Option<PySnrSample>> {
        let stacker = self.active()?;
        Ok(py
            .detach(|| measure_depth(stacker.view()))
            .map(|sample| PySnrSample { inner: sample }))
    }

    /// The lens distortion an expanded canvas was planned through, as a
    /// dict, or ``None`` when planning measured none.
    #[getter]
    fn expanded_lens<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, pyo3::types::PyDict>>> {
        let Some(lens) = self.active()?.expanded_lens() else {
            return Ok(None);
        };
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("center_x", lens.center_x)?;
        dict.set_item("center_y", lens.center_y)?;
        dict.set_item("decenter_pixels", lens.decenter_pixels)?;
        dict.set_item("k1", lens.k1)?;
        dict.set_item("k2", lens.k2)?;
        dict.set_item("largest_shift_pixels", lens.largest_shift_pixels)?;
        dict.set_item("overlap_rms_pixels", lens.overlap_rms_pixels)?;
        dict.set_item(
            "undistorted_overlap_rms_pixels",
            lens.undistorted_overlap_rms_pixels,
        )?;
        dict.set_item("focal_length_pixels", lens.focal_length_pixels)?;
        Ok(Some(dict))
    }

    /// What planning an expanded canvas noted: why it skipped fitting one
    /// lens to every panel (their camera or optics differ, or their headers
    /// do not say), or why it kept its first plan when the lens-corrected
    /// one failed. Empty for an ordinary stack and a reopened context.
    #[getter]
    fn expansion_planning_notes(&self) -> PyResult<Vec<String>> {
        Ok(self.active()?.expansion_planning_notes().to_vec())
    }

    fn snapshot(&self, py: Python<'_>) -> PyResult<PyStackSnapshot> {
        let stacker = self.active()?;
        let snapshot = py.detach(|| stacker.snapshot()).map_err(stack_error)?;
        Ok(PyStackSnapshot { inner: snapshot })
    }

    /// Keep each frame admitted from now on in a scratch file, registered but
    /// not normalized, so :meth:`reintegrate` and :meth:`reintegrate_drizzled`
    /// read it back instead of calibrating, demosaicing and resampling the
    /// frame's source again; the result is the same either way. Call it
    /// before pushing frames. The files take about four bytes per output
    /// sample per admitted frame, a sixteenth more under local background
    /// normalization, in a new directory inside ``scratch_directory`` (the
    /// system temporary directory by default) that is removed when the
    /// stacker is finished or freed. A stack that integrates Bayer photosites
    /// keeps each frame's photosites. A stack that cannot be reintegrated
    /// keeps nothing.
    #[pyo3(signature = (*, scratch_directory=None))]
    fn retain_frames_for_reintegration(
        &mut self,
        py: Python<'_>,
        scratch_directory: Option<PathBuf>,
    ) -> PyResult<()> {
        py.detach(|| {
            self.active_mut()?
                .retain_frames_for_reintegration(scratch_directory.as_deref())
        })
        .map_err(stack_error)
    }

    /// Integrate every admitted frame again with three-pass, leave-one-out
    /// rejection, which removes trails in the reference and warm-up frames
    /// that online rejection cannot revisit. Each frame is reread from its
    /// source file, unless :meth:`retain_frames_for_reintegration` kept it;
    /// prepared frames wait in ``scratch_directory`` (the system temporary
    /// directory by default) between passes. The live stack is left as it
    /// was.
    #[pyo3(signature = (*, sigma_low=3.0, sigma_high=3.0, scratch_directory=None))]
    fn reintegrate(
        &self,
        py: Python<'_>,
        sigma_low: f32,
        sigma_high: f32,
        scratch_directory: Option<PathBuf>,
    ) -> PyResult<PyStackSnapshot> {
        let stacker = self.active()?;
        let options = batch_options(sigma_low, sigma_high, scratch_directory)?;
        let result = py
            .detach(|| stacker.reintegrate(&options, |_, _, _| {}))
            .map_err(stack_error)?;
        Ok(PyStackSnapshot {
            inner: result.snapshot,
        })
    }

    /// :meth:`reintegrate`, also drizzling every admitted frame onto a grid
    /// ``scale`` (1 to 4) times finer than the reference, as PixInsight's
    /// DrizzleIntegration does. A pixel the integration rejected is left
    /// out; Bayer frames drop each photosite into its own colour.
    /// ``drop_shrink`` is each drop's side in source pixels, by default 0.9
    /// for monochrome and 1.0 for Bayer frames. Returns the reintegrated
    /// stack and the drizzled one.
    #[pyo3(signature = (
        *,
        scale=1,
        drop_shrink=None,
        sigma_low=3.0,
        sigma_high=3.0,
        scratch_directory=None
    ))]
    fn reintegrate_drizzled(
        &self,
        py: Python<'_>,
        scale: u32,
        drop_shrink: Option<f32>,
        sigma_low: f32,
        sigma_high: f32,
        scratch_directory: Option<PathBuf>,
    ) -> PyResult<(PyStackSnapshot, PyDrizzleResult)> {
        let stacker = self.active()?;
        let options = batch_options(sigma_low, sigma_high, scratch_directory)?;
        let drizzle = DrizzleOptions { scale, drop_shrink };
        drizzle
            .validate()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let (result, drizzled) = py
            .detach(|| stacker.reintegrate_drizzled(&options, &drizzle, |_, _, _| {}))
            .map_err(stack_error)?;
        let drizzled = PyDrizzleResult {
            inner: drizzled,
            accepted_frames: result.snapshot.accepted_frames,
            rejected_frames: result.snapshot.rejected_frames,
            reference_headers: stacker.reference_headers().to_vec(),
            input_paths: stacker.protected_paths(),
        };
        Ok((
            PyStackSnapshot {
                inner: result.snapshot,
            },
            drizzled,
        ))
    }

    /// Consume the accumulator, optionally write a linear FITS file, and return its arrays.
    #[pyo3(signature = (output=None))]
    fn finish(&mut self, py: Python<'_>, output: Option<PathBuf>) -> PyResult<PyStackSnapshot> {
        if let Some(output) = &output
            && self
                .active()?
                .protected_paths()
                .iter()
                .any(|input| paths_refer_to_same_file(input, output))
        {
            return Err(PyValueError::new_err(
                "output path must not refer to a stack input or calibration master",
            ));
        }
        let stacker = self.inner.take().ok_or_else(finished_error)?;
        let snapshot = py
            .detach(move || {
                let headers = stacker.reference_headers().to_vec();
                let snapshot = stacker.into_snapshot()?;
                if let Some(output) = output {
                    write_fits_f32(output, &snapshot, &headers)?;
                }
                Ok::<_, seiza_stacking::Error>(snapshot)
            })
            .map_err(stack_error)?;
        Ok(PyStackSnapshot { inner: snapshot })
    }

    fn __repr__(&self) -> String {
        match self.inner.as_ref() {
            Some(stacker) => {
                let view = stacker.view();
                format!(
                    "LiveStacker(shape={}x{}x{}, accepted_frames={}, rejected_frames={})",
                    view.width,
                    view.height,
                    view.channels,
                    view.accepted_frames,
                    view.rejected_frames,
                )
            }
            None => "LiveStacker(finished=True)".into(),
        }
    }
}

impl PyLiveStacker {
    fn active(&self) -> PyResult<&LiveStacker> {
        self.inner.as_ref().ok_or_else(finished_error)
    }

    fn active_mut(&mut self) -> Result<&mut LiveStacker, seiza_stacking::Error> {
        self.inner.as_mut().ok_or_else(|| {
            seiza_stacking::Error::Stack("live stack has already been finished".into())
        })
    }
}

fn finished_error() -> PyErr {
    PyRuntimeError::new_err("live stack has already been finished")
}

#[pyclass(frozen, name = "MasterResult", module = "seiza")]
pub(crate) struct PyMasterResult {
    #[pyo3(get)]
    output: PathBuf,
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    width: usize,
    #[pyo3(get)]
    height: usize,
    #[pyo3(get)]
    channels: usize,
    #[pyo3(get)]
    input_frames: usize,
    #[pyo3(get)]
    accepted_samples: u64,
    #[pyo3(get)]
    rejected_samples: u64,
    #[pyo3(get)]
    fallback_pixels: u64,
    #[pyo3(get)]
    bias_subtracted: bool,
    #[pyo3(get)]
    dark_subtracted: bool,
    #[pyo3(get)]
    normalized: bool,
    #[pyo3(get)]
    exposure_seconds: Option<f64>,
    input_statistics: Vec<(u64, u64)>,
    /// Samples star masking left out, across every input.
    #[pyo3(get)]
    masked_samples: u64,
    /// The samples star masking left out of each accepted input, in the
    /// order of `input_statistics`.
    #[pyo3(get)]
    input_masked_samples: Vec<u64>,
    /// The rejection the build used: `NONE`, `MEDIAN_MAD` or `LEAVE_ONE_OUT`.
    #[pyo3(get)]
    rejection_method: &'static str,
    flat_star_masking_statistics: Option<FlatStarMaskingStatistics>,
}

/// Star masking for sky flats: what :func:`build_flat` leaves out of each
/// flat before it normalizes and combines them.
///
/// Stars brighter than `detection_sigma` above the sky are masked out to
/// `halo_radius_factor` times their size, and at least
/// `minimum_radius_pixels`. `saturation_level`, in the inputs' raw units,
/// marks saturated patches; by default it comes from `SATURATE`, `SATLEVEL`
/// or the FITS encoding. Every pixel must keep at least
/// `minimum_clean_samples` unmasked inputs (two or more).
#[pyclass(frozen, name = "FlatStarMasking", module = "seiza")]
pub(crate) struct PyFlatStarMasking {
    #[pyo3(get)]
    detection_sigma: f32,
    #[pyo3(get)]
    halo_radius_factor: f32,
    #[pyo3(get)]
    minimum_radius_pixels: f32,
    #[pyo3(get)]
    saturation_level: Option<f32>,
    #[pyo3(get)]
    minimum_clean_samples: usize,
}

impl From<&FlatStarMaskingOptions> for PyFlatStarMasking {
    fn from(options: &FlatStarMaskingOptions) -> Self {
        Self {
            detection_sigma: options.detection_sigma,
            halo_radius_factor: options.halo_radius_factor,
            minimum_radius_pixels: options.minimum_radius_pixels,
            saturation_level: options.saturation_level,
            minimum_clean_samples: options.minimum_clean_samples,
        }
    }
}

impl From<&PyFlatStarMasking> for FlatStarMaskingOptions {
    fn from(options: &PyFlatStarMasking) -> Self {
        Self {
            detection_sigma: options.detection_sigma,
            halo_radius_factor: options.halo_radius_factor,
            minimum_radius_pixels: options.minimum_radius_pixels,
            saturation_level: options.saturation_level,
            minimum_clean_samples: options.minimum_clean_samples,
        }
    }
}

#[pymethods]
impl PyFlatStarMasking {
    #[new]
    #[pyo3(signature = (
        *,
        detection_sigma=None,
        halo_radius_factor=None,
        minimum_radius_pixels=None,
        saturation_level=None,
        minimum_clean_samples=None
    ))]
    fn new(
        detection_sigma: Option<f32>,
        halo_radius_factor: Option<f32>,
        minimum_radius_pixels: Option<f32>,
        saturation_level: Option<f32>,
        minimum_clean_samples: Option<usize>,
    ) -> Self {
        let defaults = FlatStarMaskingOptions::default();
        Self {
            detection_sigma: detection_sigma.unwrap_or(defaults.detection_sigma),
            halo_radius_factor: halo_radius_factor.unwrap_or(defaults.halo_radius_factor),
            minimum_radius_pixels: minimum_radius_pixels.unwrap_or(defaults.minimum_radius_pixels),
            saturation_level: saturation_level.or(defaults.saturation_level),
            minimum_clean_samples: minimum_clean_samples.unwrap_or(defaults.minimum_clean_samples),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "FlatStarMasking(detection_sigma={}, halo_radius_factor={}, minimum_radius_pixels={}, saturation_level={:?}, minimum_clean_samples={})",
            self.detection_sigma,
            self.halo_radius_factor,
            self.minimum_radius_pixels,
            self.saturation_level,
            self.minimum_clean_samples
        )
    }
}

#[pymethods]
impl PyMasterResult {
    #[getter]
    fn input_statistics(&self) -> Vec<(u64, u64)> {
        self.input_statistics.clone()
    }

    /// What flat star masking did, or `None` when it did not run: a dict
    /// with the `options` used (a :class:`FlatStarMasking`),
    /// `masked_samples`, `masked_pixels`, the `minimum_clean_samples` and
    /// `maximum_clean_samples` any pixel kept, `low_coverage_samples`,
    /// `unmasked_saturation_samples` and `unknown_saturation_inputs`.
    #[getter]
    fn flat_star_masking<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, pyo3::types::PyDict>>> {
        let Some(statistics) = &self.flat_star_masking_statistics else {
            return Ok(None);
        };
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("options", PyFlatStarMasking::from(&statistics.options))?;
        dict.set_item("masked_samples", statistics.masked_samples)?;
        dict.set_item("masked_pixels", statistics.masked_pixels)?;
        dict.set_item("minimum_clean_samples", statistics.minimum_clean_samples)?;
        dict.set_item("maximum_clean_samples", statistics.maximum_clean_samples)?;
        dict.set_item("low_coverage_samples", statistics.low_coverage_samples)?;
        dict.set_item(
            "unmasked_saturation_samples",
            statistics.unmasked_saturation_samples,
        )?;
        dict.set_item(
            "unknown_saturation_inputs",
            statistics.unknown_saturation_inputs,
        )?;
        Ok(Some(dict))
    }

    fn __repr__(&self) -> String {
        format!(
            "MasterResult(kind={:?}, output={:?}, input_frames={}, rejected_samples={})",
            self.kind, self.output, self.input_frames, self.rejected_samples,
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    images,
    output,
    *,
    options=None,
    bias=None,
    dark=None,
    flat=None,
    dark_exposure_seconds=None,
    reference_image=None,
    reference="first"
))]
#[allow(clippy::too_many_arguments)]
fn stack_fits(
    py: Python<'_>,
    images: Vec<PathBuf>,
    output: PathBuf,
    options: Option<PyRef<'_, PyStackOptions>>,
    bias: Option<PathBuf>,
    dark: Option<PathBuf>,
    flat: Option<PathBuf>,
    dark_exposure_seconds: Option<f64>,
    reference_image: Option<PathBuf>,
    reference: &str,
) -> PyResult<PyStackResult> {
    if images.len() < 2 {
        return Err(PyValueError::new_err(
            "stack_fits requires at least two light frames",
        ));
    }
    let automatic = match reference {
        "first" => false,
        "auto" => true,
        other => {
            return Err(PyValueError::new_err(format!(
                "reference must be \"first\" or \"auto\", not {other:?}"
            )));
        }
    };
    validate_exposure_override(dark.as_ref(), dark_exposure_seconds, "dark")?;
    validate_output_path(
        &images,
        &output,
        [bias.as_ref(), dark.as_ref(), flat.as_ref()],
    )?;
    if let Some(reference_image) = &reference_image {
        // The reference image may be one of the lights, but not a master or
        // the output.
        if paths_refer_to_same_file(reference_image, &output) {
            return Err(PyValueError::new_err(
                "output path must not refer to the reference image",
            ));
        }
        let paths = std::iter::once(reference_image.clone())
            .chain(
                [bias.clone(), dark.clone(), flat.clone()]
                    .into_iter()
                    .flatten(),
            )
            .collect::<Vec<_>>();
        validate_distinct_paths(&paths)?;
    }
    let options = options
        .as_ref()
        .map_or_else(StackOptions::default, |options| options.inner.clone());
    py.detach(move || {
        let calibration = load_calibration(bias, dark, flat, dark_exposure_seconds)?;
        let mut images = images;
        // The lights that may become the reference light, best first. With
        // a reference image, the next ones stand in when the best does not
        // align to it.
        let mut candidates = vec![0];
        if automatic {
            let (_, scores) = seiza_stacking::choose_reference(&images, 8)?;
            candidates = seiza_stacking::rank_references(&scores);
            candidates.truncate(CHANNEL_REFERENCE_ATTEMPTS);
        }
        // The reference light goes first. Without a reference image it is
        // the stack's reference; with one, it sets only normalization and
        // noise weighting and is stacked like the rest.
        let (mut stacker, chosen) = match &reference_image {
            Some(reference_image) => {
                let (stacker, used) = LiveStacker::with_reference_image_candidates(
                    FitsFrame::open(reference_image)?,
                    candidates
                        .iter()
                        .map(|&index| FitsFrame::open(&images[index])),
                    calibration,
                    options,
                    |_, _| {},
                )?;
                (stacker, candidates[used])
            }
            None => (
                LiveStacker::new(
                    FitsFrame::open(&images[candidates[0]])?,
                    calibration,
                    options,
                )?,
                candidates[0],
            ),
        };
        let reference_light = images.remove(chosen);
        images.insert(0, reference_light.clone());
        let lights = &images[usize::from(reference_image.is_none())..];
        let mut frames = Vec::new();
        for path in lights {
            let frame = FitsFrame::open(path)?;
            frames.push(PyFrameDisposition::from_rust(
                Some(path.clone()),
                stacker.push(frame)?,
            ));
        }
        if stacker.view().accepted_frames == 0 {
            return Err(seiza_stacking::Error::Stack(
                "no light frames were accepted against the reference image".into(),
            ));
        }
        let headers = stacker.reference_headers().to_vec();
        let snapshot = stacker.into_snapshot()?;
        write_fits_f32(&output, &snapshot, &headers)?;
        Ok(PyStackResult {
            output,
            accepted_frames: snapshot.accepted_frames,
            rejected_frames: snapshot.rejected_frames,
            width: snapshot.image.width,
            height: snapshot.image.height,
            channels: snapshot.image.channels,
            reference_light,
            frames,
        })
    })
    .map_err(stack_error)
}

/// Score every image as a possible stack reference and choose one, as the
/// CLI's ``--reference auto`` does: among frames with nearly the best
/// stars, the one with the flattest sky. Only reads the images, at most
/// ``concurrency`` at once.
#[pyfunction(name = "choose_reference")]
#[pyo3(signature = (images, *, concurrency=8))]
fn py_choose_reference(
    py: Python<'_>,
    images: Vec<PathBuf>,
    concurrency: usize,
) -> PyResult<PyReferenceChoice> {
    if images.is_empty() {
        return Err(PyValueError::new_err(
            "choose_reference requires at least one image",
        ));
    }
    let (reference_index, scores) = py
        .detach(|| seiza_stacking::choose_reference(&images, concurrency))
        .map_err(stack_error)?;
    Ok(PyReferenceChoice {
        reference_index,
        reference_path: images[reference_index].clone(),
        ranking: seiza_stacking::rank_references(&scores),
        scores: scores
            .into_iter()
            .map(|score| score.map(PyReferenceScore::from))
            .collect(),
    })
}

#[pyfunction]
#[pyo3(signature = (images, output, *, sigma_low=3.0, sigma_high=3.0, cancel=None))]
fn build_bias(
    py: Python<'_>,
    images: Vec<PathBuf>,
    output: PathBuf,
    sigma_low: f32,
    sigma_high: f32,
    cancel: Option<Py<PyAny>>,
) -> PyResult<PyMasterResult> {
    build_master(
        py,
        images,
        output,
        MasterFrameKind::Bias,
        None,
        None,
        None,
        None,
        sigma_low,
        sigma_high,
        cancel,
        MasterExtras::default(),
    )
}

#[pyfunction]
#[pyo3(signature = (
    images,
    output,
    *,
    bias=None,
    exposure_seconds=None,
    sigma_low=3.0,
    sigma_high=3.0,
    cancel=None,
    dark_level_screening=None
))]
#[allow(clippy::too_many_arguments)]
fn build_dark(
    py: Python<'_>,
    images: Vec<PathBuf>,
    output: PathBuf,
    bias: Option<PathBuf>,
    exposure_seconds: Option<f64>,
    sigma_low: f32,
    sigma_high: f32,
    cancel: Option<Py<PyAny>>,
    dark_level_screening: Option<f32>,
) -> PyResult<PyMasterResult> {
    build_master(
        py,
        images,
        output,
        MasterFrameKind::Dark,
        bias,
        None,
        None,
        exposure_seconds,
        sigma_low,
        sigma_high,
        cancel,
        MasterExtras {
            dark_level_screening: dark_level_screening
                .map(|max_excess_sigma| DarkLevelScreening { max_excess_sigma }),
            ..MasterExtras::default()
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    images,
    output,
    *,
    bias=None,
    dark_flat=None,
    dark_flat_exposure_seconds=None,
    exposure_seconds=None,
    sigma_low=3.0,
    sigma_high=3.0,
    cancel=None,
    flat_star_masking=None,
    scratch_directory=None
))]
#[allow(clippy::too_many_arguments)]
fn build_flat(
    py: Python<'_>,
    images: Vec<PathBuf>,
    output: PathBuf,
    bias: Option<PathBuf>,
    dark_flat: Option<PathBuf>,
    dark_flat_exposure_seconds: Option<f64>,
    exposure_seconds: Option<f64>,
    sigma_low: f32,
    sigma_high: f32,
    cancel: Option<Py<PyAny>>,
    flat_star_masking: Option<PyRef<'_, PyFlatStarMasking>>,
    scratch_directory: Option<PathBuf>,
) -> PyResult<PyMasterResult> {
    validate_exposure_override(dark_flat.as_ref(), dark_flat_exposure_seconds, "dark_flat")?;
    build_master(
        py,
        images,
        output,
        MasterFrameKind::Flat,
        bias,
        dark_flat,
        dark_flat_exposure_seconds,
        exposure_seconds,
        sigma_low,
        sigma_high,
        cancel,
        MasterExtras {
            flat_star_masking: flat_star_masking
                .as_deref()
                .map(FlatStarMaskingOptions::from),
            scratch_directory,
            ..MasterExtras::default()
        },
    )
}

/// Master build options only some kinds take.
#[derive(Default)]
struct MasterExtras {
    flat_star_masking: Option<FlatStarMaskingOptions>,
    dark_level_screening: Option<DarkLevelScreening>,
    scratch_directory: Option<PathBuf>,
}

#[allow(clippy::too_many_arguments)]
fn build_master(
    py: Python<'_>,
    images: Vec<PathBuf>,
    output: PathBuf,
    kind: MasterFrameKind,
    bias: Option<PathBuf>,
    dark: Option<PathBuf>,
    dark_exposure_seconds: Option<f64>,
    exposure_seconds: Option<f64>,
    sigma_low: f32,
    sigma_high: f32,
    cancel: Option<Py<PyAny>>,
    extras: MasterExtras,
) -> PyResult<PyMasterResult> {
    validate_output_path(&images, &output, [bias.as_ref(), dark.as_ref()])?;
    // A master over dozens of frames runs for minutes with the GIL released,
    // so the build takes the GIL back once per input to ask whether it should
    // stop. That covers Ctrl-C as well as an explicit `cancel` predicate.
    let raised: Arc<Mutex<Option<PyErr>>> = Arc::new(Mutex::new(None));
    let signal = {
        let raised = Arc::clone(&raised);
        CancelSignal::new(move || {
            Python::attach(|py| {
                let asked = py.check_signals().and_then(|()| match &cancel {
                    Some(cancel) => cancel
                        .call0(py)
                        .and_then(|value| value.bind(py).is_truthy()),
                    None => Ok(false),
                });
                match asked {
                    Ok(stop) => stop,
                    Err(error) => {
                        // Re-raised once the build unwinds, so the caller sees
                        // KeyboardInterrupt or their own exception rather than
                        // a generic StackError.
                        *raised.lock().unwrap() = Some(error);
                        true
                    }
                }
            })
        })
    };
    let built = py.detach(move || {
        let bias = bias.map(load_bias).transpose()?;
        let dark = dark
            .map(|path| {
                let frame = FitsFrame::open(path)?;
                MasterDark::from_fits_frame(frame, dark_exposure_seconds)
            })
            .transpose()?;
        let options = MasterBuildOptions {
            rejection: MasterRejectionOptions {
                low_sigma: sigma_low,
                high_sigma: sigma_high,
            },
            exposure_seconds,
            bias,
            dark,
            cancel: Some(signal),
            progress: None,
            defect_suppression: None,
            flat_star_masking: extras.flat_star_masking,
            dark_level_screening: extras.dark_level_screening,
        };
        let master = match &extras.scratch_directory {
            Some(directory) => {
                build_master_from_fits_with_scratch(&images, kind, &options, directory)?
            }
            None => build_master_from_fits(&images, kind, &options)?,
        };
        write_master_fits_f32(&output, &master)?;
        Ok(PyMasterResult {
            output,
            kind: master.kind.as_str().into(),
            width: master.image.width,
            height: master.image.height,
            channels: master.image.channels,
            input_frames: master.input_frames,
            accepted_samples: master.accepted_samples,
            rejected_samples: master.rejected_samples,
            fallback_pixels: master.fallback_pixels,
            bias_subtracted: master.bias_subtracted,
            dark_subtracted: master.dark_subtracted,
            normalized: master.normalized,
            exposure_seconds: master.exposure_seconds,
            input_statistics: master
                .input_statistics
                .iter()
                .map(|stats| (stats.accepted_samples, stats.rejected_samples))
                .collect(),
            masked_samples: master.masked_samples,
            input_masked_samples: master
                .input_statistics
                .iter()
                .map(|stats| stats.masked_samples)
                .collect(),
            rejection_method: master.rejection_method.as_str(),
            flat_star_masking_statistics: master.flat_star_masking,
        })
    });
    if let Some(error) = raised.lock().unwrap().take() {
        return Err(error);
    }
    built.map_err(stack_error)
}

fn load_calibration(
    bias: Option<PathBuf>,
    dark: Option<PathBuf>,
    flat: Option<PathBuf>,
    dark_exposure_seconds: Option<f64>,
) -> seiza_stacking::Result<CalibrationMasters> {
    CalibrationMasters::from_fits_paths(
        bias.as_deref(),
        dark.as_deref(),
        flat.as_deref(),
        dark_exposure_seconds,
    )
}

fn load_bias(path: PathBuf) -> seiza_stacking::Result<LinearImage> {
    let frame = FitsFrame::open(path)?;
    frame.validate_master_kind("BIAS")?;
    Ok(frame.image)
}

fn validate_output_path<const N: usize>(
    inputs: &[PathBuf],
    output: &Path,
    other_inputs: [Option<&PathBuf>; N],
) -> PyResult<()> {
    let mut all_inputs = inputs.iter().map(PathBuf::as_path).collect::<Vec<_>>();
    all_inputs.extend(other_inputs.into_iter().flatten().map(PathBuf::as_path));
    let output_identity = path_identity(output);
    let mut seen = std::collections::HashSet::new();
    for input in all_inputs {
        let identity = path_identity(input);
        if identity == output_identity {
            return Err(PyValueError::new_err(
                "output path must not refer to an input frame",
            ));
        }
        if !seen.insert(identity) {
            return Err(PyValueError::new_err(format!(
                "duplicate input path {}",
                input.display()
            )));
        }
    }
    Ok(())
}

fn validate_distinct_paths(paths: &[PathBuf]) -> PyResult<()> {
    let mut seen = std::collections::HashSet::new();
    for path in paths {
        if !seen.insert(path_identity(path)) {
            return Err(PyValueError::new_err(format!(
                "duplicate input path {}",
                path.display()
            )));
        }
    }
    Ok(())
}

fn validate_exposure_override(
    master: Option<&PathBuf>,
    exposure_seconds: Option<f64>,
    argument: &str,
) -> PyResult<()> {
    if exposure_seconds.is_some() && master.is_none() {
        return Err(PyValueError::new_err(format!(
            "{argument}_exposure_seconds requires {argument}"
        )));
    }
    Ok(())
}

/// One reading of an accumulator, in the stack's own units. Only ratios
/// between readings of the same stack mean anything.
#[pyclass(skip_from_py_object, frozen, name = "SnrSample", module = "seiza")]
#[derive(Clone)]
pub(crate) struct PySnrSample {
    inner: SnrSample,
}

#[pymethods]
impl PySnrSample {
    /// Frames the accumulator had taken when this was measured.
    #[getter]
    fn frames(&self) -> u32 {
        self.inner.frames
    }

    /// Pixel-scale noise of the integration, averaged across channels.
    #[getter]
    fn noise(&self) -> f64 {
        self.inner.noise
    }

    /// Median sample, averaged across channels.
    #[getter]
    fn background(&self) -> f64 {
        self.inner.background
    }

    /// How far the brightest one percent sits above the background.
    #[getter]
    fn signal(&self) -> f64 {
        self.inner.signal
    }

    /// Per-channel noise: one entry for mono, three for a debayered stack.
    #[getter]
    fn channel_noise(&self) -> Vec<f64> {
        self.inner.channel_noise.clone()
    }

    /// This reading's signal against its own noise.
    ///
    /// Correct for one depth. To compare depths, divide one signal — normally
    /// the deepest measured one — by each depth's noise instead. The
    /// brightest-percent statistic is itself lifted by noise where a stack is
    /// shallow, so reading each depth against its own signal flatters the
    /// early frames.
    #[getter]
    fn snr(&self) -> f64 {
        self.inner.snr()
    }

    fn __repr__(&self) -> String {
        format!(
            "SnrSample(frames={}, noise={:.4}, signal={:.4}, snr={:.2})",
            self.inner.frames,
            self.inner.noise,
            self.inner.signal,
            self.inner.snr()
        )
    }
}

/// The depths a build of ``total`` frames should measure at: the doubling
/// ladder, and the full set.
///
/// Doubling keeps the count of measurements logarithmic, so a five-hundred
/// frame stack is interrupted nine times rather than five hundred, and the
/// points still spread evenly on a log axis.
#[pyfunction]
#[pyo3(name = "checkpoint_depths")]
pub(crate) fn py_checkpoint_depths(total: usize) -> Vec<usize> {
    checkpoint_depths(total)
}

/// Measure a linear image's noise in each channel, the estimator behind
/// frame weights: the median absolute deviation of second differences along
/// each axis, as a Gaussian standard deviation in the image's own units,
/// keeping the noisier axis. Returns ``None`` when the image is too small or
/// has too few finite samples to measure.
#[pyfunction]
fn frame_noise(py: Python<'_>, image: PyReadonlyArrayDyn<'_, f32>) -> PyResult<Option<Vec<f32>>> {
    let image = linear_image(image)?;
    Ok(py.detach(|| seiza_stacking::frame_noise(&image)))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyStackOptions>()?;
    module.add_class::<PyFrameDisposition>()?;
    module.add_class::<PyPipelineReport>()?;
    module.add_class::<PyStackResult>()?;
    module.add_class::<PyReferenceScore>()?;
    module.add_class::<PyReferenceChoice>()?;
    module.add_class::<PyChannelReferenceSelection>()?;
    module.add_class::<PyStackSnapshot>()?;
    module.add_class::<PySnrSample>()?;
    module.add_function(wrap_pyfunction!(py_checkpoint_depths, module)?)?;
    module.add_class::<PyLiveStacker>()?;
    module.add_class::<PyDrizzleResult>()?;
    module.add_class::<PyMasterResult>()?;
    module.add_class::<PyFlatStarMasking>()?;
    module.add_function(wrap_pyfunction!(frame_noise, module)?)?;
    module.add_function(wrap_pyfunction!(stack_fits, module)?)?;
    module.add_function(wrap_pyfunction!(py_choose_reference, module)?)?;
    module.add_function(wrap_pyfunction!(build_bias, module)?)?;
    module.add_function(wrap_pyfunction!(build_dark, module)?)?;
    module.add_function(wrap_pyfunction!(build_flat, module)?)?;
    module.add("StackError", module.py().get_type::<StackError>())?;
    Ok(())
}
