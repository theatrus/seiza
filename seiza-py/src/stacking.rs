use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use seiza_stacking::{
    CalibrationMasters, DeltaSigmaOptions, FitsFrame, FrameDisposition, LinearImage, LiveStacker,
    MasterBuildOptions, MasterDark, MasterFlat, MasterFrameKind, MasterRejectionOptions,
    NormalizationMode, RejectionMode, StackOptions, StackSnapshot, build_master_from_fits,
    paths_refer_to_same_file, write_fits_f32, write_master_fits_f32,
};
use std::path::{Path, PathBuf};

pyo3::create_exception!(
    seiza,
    StackError,
    PyRuntimeError,
    "Calibration, registration, or stacking failed."
);

fn stack_error(error: seiza_stacking::Error) -> PyErr {
    StackError::new_err(error.to_string())
}

#[pyclass(frozen, name = "StackOptions", module = "seiza")]
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
        minimum_integrated_fraction=0.50
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
    ) -> PyResult<Self> {
        let normalization = match normalization {
            "none" => NormalizationMode::None,
            "global" => NormalizationMode::Global,
            "local" => NormalizationMode::Local {
                tile_size: local_tile_size,
            },
            value => {
                return Err(PyValueError::new_err(format!(
                    "normalization must be 'none', 'global', or 'local', not {value:?}"
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
        inner.acceptance.maximum_registration_rms_pixels = maximum_registration_rms;
        inner.acceptance.maximum_scale_deviation = maximum_scale_deviation;
        inner.acceptance.maximum_rotation_degrees = maximum_rotation_degrees;
        inner.acceptance.minimum_overlap_fraction = minimum_overlap;
        inner.acceptance.minimum_normalization_gain = minimum_normalization_gain;
        inner.acceptance.maximum_normalization_gain = maximum_normalization_gain;
        inner.acceptance.minimum_integrated_fraction = minimum_integrated_fraction;
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
        }
    }

    #[getter]
    fn local_tile_size(&self) -> Option<usize> {
        match self.inner.normalization {
            NormalizationMode::Local { tile_size } => Some(tile_size),
            _ => None,
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

#[pyclass(frozen, name = "FrameDisposition", module = "seiza")]
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
            },
        }
    }
}

#[pymethods]
impl PyFrameDisposition {
    fn __bool__(&self) -> bool {
        self.accepted
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

#[pyclass(name = "LiveStacker", module = "seiza")]
pub(crate) struct PyLiveStacker {
    inner: Option<LiveStacker>,
    input_paths: Vec<PathBuf>,
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
            .allow_threads(move || {
                let calibration = load_calibration(bias, dark, flat, dark_exposure_seconds)?;
                let reference = FitsFrame::open(reference)?;
                LiveStacker::new(reference, calibration, options)
            })
            .map_err(stack_error)?;
        Ok(Self {
            inner: Some(inner),
            input_paths,
        })
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
            .allow_threads(move || LiveStacker::from_linear(reference, options))
            .map_err(stack_error)?;
        Ok(Self {
            inner: Some(inner),
            input_paths: Vec::new(),
        })
    }

    #[getter]
    fn accepted_frames(&self) -> PyResult<u32> {
        Ok(self.active()?.view().accepted_frames)
    }

    #[getter]
    fn rejected_frames(&self) -> PyResult<u32> {
        Ok(self.active()?.view().rejected_frames)
    }

    fn push_fits(&mut self, py: Python<'_>, path: PathBuf) -> PyResult<PyFrameDisposition> {
        if self
            .input_paths
            .iter()
            .any(|input| paths_refer_to_same_file(input, &path))
        {
            return Err(PyValueError::new_err(format!(
                "FITS frame {} has already been used by this stack",
                path.display()
            )));
        }
        let disposition = py
            .allow_threads(|| {
                let frame = FitsFrame::open(&path)?;
                self.active_mut()?.push(frame)
            })
            .map_err(stack_error)?;
        self.input_paths.push(path.clone());
        Ok(PyFrameDisposition::from_rust(Some(path), disposition))
    }

    /// Push an already linear, calibrated, and channel-compatible NumPy frame.
    fn push(
        &mut self,
        py: Python<'_>,
        image: PyReadonlyArrayDyn<'_, f32>,
    ) -> PyResult<PyFrameDisposition> {
        let image = linear_image(image)?;
        let disposition = py
            .allow_threads(|| self.active_mut()?.push_linear(image))
            .map_err(stack_error)?;
        Ok(PyFrameDisposition::from_rust(None, disposition))
    }

    /// Copy the current accumulator into an immutable Python snapshot.
    fn snapshot(&self, py: Python<'_>) -> PyResult<PyStackSnapshot> {
        let stacker = self.active()?;
        let snapshot = py
            .allow_threads(|| stacker.snapshot())
            .map_err(stack_error)?;
        Ok(PyStackSnapshot { inner: snapshot })
    }

    /// Consume the accumulator, optionally write a linear FITS file, and return its arrays.
    #[pyo3(signature = (output=None))]
    fn finish(&mut self, py: Python<'_>, output: Option<PathBuf>) -> PyResult<PyStackSnapshot> {
        if let Some(output) = &output
            && self
                .input_paths
                .iter()
                .any(|input| paths_refer_to_same_file(input, output))
        {
            return Err(PyValueError::new_err(
                "output path must not refer to a stack input or calibration master",
            ));
        }
        let stacker = self.inner.take().ok_or_else(finished_error)?;
        let snapshot = py
            .allow_threads(move || {
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
    bias_subtracted: bool,
    #[pyo3(get)]
    dark_subtracted: bool,
    #[pyo3(get)]
    normalized: bool,
    #[pyo3(get)]
    exposure_seconds: Option<f64>,
    input_statistics: Vec<(u64, u64)>,
}

#[pymethods]
impl PyMasterResult {
    #[getter]
    fn input_statistics(&self) -> Vec<(u64, u64)> {
        self.input_statistics.clone()
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
    dark_exposure_seconds=None
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
) -> PyResult<PyStackResult> {
    if images.len() < 2 {
        return Err(PyValueError::new_err(
            "stack_fits requires at least two light frames",
        ));
    }
    validate_exposure_override(dark.as_ref(), dark_exposure_seconds, "dark")?;
    validate_output_path(
        &images,
        &output,
        [bias.as_ref(), dark.as_ref(), flat.as_ref()],
    )?;
    let options = options
        .as_ref()
        .map_or_else(StackOptions::default, |options| options.inner.clone());
    py.allow_threads(move || {
        let calibration = load_calibration(bias, dark, flat, dark_exposure_seconds)?;
        let mut paths = images.into_iter();
        let reference_path = paths.next().expect("two paths were validated");
        let reference = FitsFrame::open(&reference_path)?;
        let mut stacker = LiveStacker::new(reference, calibration, options)?;
        let mut frames = Vec::new();
        for path in paths {
            let frame = FitsFrame::open(&path)?;
            frames.push(PyFrameDisposition::from_rust(
                Some(path),
                stacker.push(frame)?,
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
            frames,
        })
    })
    .map_err(stack_error)
}

#[pyfunction]
#[pyo3(signature = (images, output, *, sigma_low=3.0, sigma_high=3.0))]
fn build_bias(
    py: Python<'_>,
    images: Vec<PathBuf>,
    output: PathBuf,
    sigma_low: f32,
    sigma_high: f32,
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
    sigma_high=3.0
))]
fn build_dark(
    py: Python<'_>,
    images: Vec<PathBuf>,
    output: PathBuf,
    bias: Option<PathBuf>,
    exposure_seconds: Option<f64>,
    sigma_low: f32,
    sigma_high: f32,
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
    sigma_high=3.0
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
    )
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
) -> PyResult<PyMasterResult> {
    validate_output_path(&images, &output, [bias.as_ref(), dark.as_ref()])?;
    py.allow_threads(move || {
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
        };
        let master = build_master_from_fits(&images, kind, &options)?;
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
            bias_subtracted: master.bias_subtracted,
            dark_subtracted: master.dark_subtracted,
            normalized: master.normalized,
            exposure_seconds: master.exposure_seconds,
            input_statistics: master
                .input_statistics
                .iter()
                .map(|stats| (stats.accepted_samples, stats.rejected_samples))
                .collect(),
        })
    })
    .map_err(stack_error)
}

fn load_calibration(
    bias: Option<PathBuf>,
    dark: Option<PathBuf>,
    flat: Option<PathBuf>,
    dark_exposure_seconds: Option<f64>,
) -> seiza_stacking::Result<CalibrationMasters> {
    let bias = bias.map(load_bias).transpose()?;
    let dark = dark
        .map(|path| {
            let frame = FitsFrame::open(path)?;
            MasterDark::from_fits_frame(frame, dark_exposure_seconds)
        })
        .transpose()?;
    let flat = flat
        .map(|path| FitsFrame::open(path).and_then(MasterFlat::from_fits_frame))
        .transpose()?;
    CalibrationMasters::new(bias, dark, flat)
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
    for (index, input) in all_inputs.iter().enumerate() {
        if all_inputs[..index]
            .iter()
            .any(|previous| paths_refer_to_same_file(input, previous))
        {
            return Err(PyValueError::new_err(format!(
                "duplicate input path {}",
                input.display()
            )));
        }
        if paths_refer_to_same_file(input, output) {
            return Err(PyValueError::new_err(
                "output path must not refer to an input frame",
            ));
        }
    }
    Ok(())
}

fn validate_distinct_paths(paths: &[PathBuf]) -> PyResult<()> {
    for (index, path) in paths.iter().enumerate() {
        if paths[..index]
            .iter()
            .any(|previous| paths_refer_to_same_file(path, previous))
        {
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

fn linear_image(array: PyReadonlyArrayDyn<'_, f32>) -> PyResult<LinearImage> {
    let shape = array.shape();
    let (height, width, channels) = match shape {
        [height, width] => (*height, *width, 1),
        [height, width, 3] => (*height, *width, 3),
        _ => {
            return Err(PyValueError::new_err(
                "stacking arrays must have shape (height, width) or (height, width, 3)",
            ));
        }
    };
    let data = array
        .as_slice()
        .map_err(|_| PyValueError::new_err("stacking arrays must be C-contiguous"))?
        .to_vec();
    LinearImage::new(width, height, channels, data).map_err(stack_error)
}

fn array_shape(width: usize, height: usize, channels: usize) -> Vec<usize> {
    if channels == 1 {
        vec![height, width]
    } else {
        vec![height, width, channels]
    }
}

fn image_array<'py>(py: Python<'py>, image: &LinearImage) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let shape = array_shape(image.width, image.height, image.channels);
    let array = ArrayD::from_shape_vec(IxDyn(&shape), image.data.clone())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(py))
}

fn u32_array<'py>(
    py: Python<'py>,
    values: &[u32],
    width: usize,
    height: usize,
    channels: usize,
) -> PyResult<Bound<'py, PyArrayDyn<u32>>> {
    let shape = array_shape(width, height, channels);
    let array = ArrayD::from_shape_vec(IxDyn(&shape), values.to_vec())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(py))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyStackOptions>()?;
    module.add_class::<PyFrameDisposition>()?;
    module.add_class::<PyStackResult>()?;
    module.add_class::<PyStackSnapshot>()?;
    module.add_class::<PyLiveStacker>()?;
    module.add_class::<PyMasterResult>()?;
    module.add_function(wrap_pyfunction!(stack_fits, module)?)?;
    module.add_function(wrap_pyfunction!(build_bias, module)?)?;
    module.add_function(wrap_pyfunction!(build_dark, module)?)?;
    module.add_function(wrap_pyfunction!(build_flat, module)?)?;
    module.add("StackError", module.py().get_type::<StackError>())?;
    Ok(())
}
