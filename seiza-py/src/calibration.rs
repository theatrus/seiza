//! Deciding which calibration frames belong together, and recovering a
//! camera pedestal when no bias or dark is available.

use crate::arrays::linear_image;
use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use seiza_calibration::{
    FrameRole, FrameSignature, LightRotation, LinearImageRef, MatchTolerances, coherent_subset,
    coherent_subset_indices, exposure_matches, fit_flat_pedestal, optics_match, rotation_matches,
    sensor_matches, sort_by_proximity, temperature_matches,
};
use seiza_stacking::{
    CalibrationPlan, CalibrationPlanKind, CalibrationPlanRequest, CalibrationRecord,
    FrameCalibrationState, FrameMetadata, FrameProbe, FrameSourceRole, plan_calibration,
    probe_frame,
};
use std::path::PathBuf;

/// What a frame was shot with, as far as matching cares.
///
/// Every field is optional. A missing value on the *candidate* side
/// disqualifies it and a missing value on the *reference* side accepts
/// whatever it is offered: a light that does not record its gain cannot rule
/// anything out, while a calibration frame that does not record its gain
/// cannot prove it belongs. Rotation is the exception — unknown on either
/// side matches, because treating a missing angle as a mismatch would strip
/// flats from every frame shot before anyone recorded one.
#[pyclass(from_py_object, name = "FrameSignature", module = "seiza")]
#[derive(Clone, Default)]
pub(crate) struct PyFrameSignature {
    pub(crate) inner: FrameSignature,
}

#[pymethods]
impl PyFrameSignature {
    #[new]
    #[pyo3(signature = (
        camera=None,
        telescope=None,
        width=None,
        height=None,
        channels=None,
        binning_x=None,
        binning_y=None,
        gain=None,
        offset=None,
        readout_mode=None,
        bayer_pattern=None,
        filter=None,
        focal_length_mm=None,
        rotation_deg=None,
        exposure_seconds=None,
        camera_temp_c=None,
        captured_at_unix=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        camera: Option<String>,
        telescope: Option<String>,
        width: Option<i64>,
        height: Option<i64>,
        channels: Option<i64>,
        binning_x: Option<i64>,
        binning_y: Option<i64>,
        gain: Option<i64>,
        offset: Option<i64>,
        readout_mode: Option<i64>,
        bayer_pattern: Option<String>,
        filter: Option<String>,
        focal_length_mm: Option<f64>,
        rotation_deg: Option<f64>,
        exposure_seconds: Option<f64>,
        camera_temp_c: Option<f64>,
        captured_at_unix: Option<i64>,
    ) -> Self {
        // Field by field from the default, because the Rust struct is
        // `#[non_exhaustive]`: anything it gains later starts out unknown here
        // rather than breaking this constructor.
        let mut inner = FrameSignature::default();
        inner.camera = camera;
        inner.telescope = telescope;
        inner.width = width;
        inner.height = height;
        inner.channels = channels;
        inner.binning_x = binning_x;
        inner.binning_y = binning_y;
        inner.gain = gain;
        inner.offset = offset;
        inner.readout_mode = readout_mode;
        inner.bayer_pattern = bayer_pattern;
        inner.filter = filter;
        inner.focal_length_mm = focal_length_mm;
        inner.rotation_deg = rotation_deg;
        inner.exposure_seconds = exposure_seconds;
        inner.camera_temp_c = camera_temp_c;
        inner.captured_at_unix = captured_at_unix;
        Self { inner }
    }

    #[getter]
    fn camera(&self) -> Option<String> {
        self.inner.camera.clone()
    }
    #[getter]
    fn telescope(&self) -> Option<String> {
        self.inner.telescope.clone()
    }
    #[getter]
    fn filter(&self) -> Option<String> {
        self.inner.filter.clone()
    }
    #[getter]
    fn bayer_pattern(&self) -> Option<String> {
        self.inner.bayer_pattern.clone()
    }
    #[getter]
    fn width(&self) -> Option<i64> {
        self.inner.width
    }
    #[getter]
    fn height(&self) -> Option<i64> {
        self.inner.height
    }
    #[getter]
    fn channels(&self) -> Option<i64> {
        self.inner.channels
    }
    #[getter]
    fn binning_x(&self) -> Option<i64> {
        self.inner.binning_x
    }
    #[getter]
    fn binning_y(&self) -> Option<i64> {
        self.inner.binning_y
    }
    #[getter]
    fn gain(&self) -> Option<i64> {
        self.inner.gain
    }
    #[getter]
    fn offset(&self) -> Option<i64> {
        self.inner.offset
    }
    #[getter]
    fn readout_mode(&self) -> Option<i64> {
        self.inner.readout_mode
    }
    #[getter]
    fn focal_length_mm(&self) -> Option<f64> {
        self.inner.focal_length_mm
    }
    #[getter]
    fn rotation_deg(&self) -> Option<f64> {
        self.inner.rotation_deg
    }
    #[getter]
    fn exposure_seconds(&self) -> Option<f64> {
        self.inner.exposure_seconds
    }
    #[getter]
    fn camera_temp_c(&self) -> Option<f64> {
        self.inner.camera_temp_c
    }
    #[getter]
    fn captured_at_unix(&self) -> Option<i64> {
        self.inner.captured_at_unix
    }

    fn __repr__(&self) -> String {
        // Python spelling, not Rust's: a `Some("Ha")` in a REPL is a leak of
        // what this happens to be written in.
        fn text(value: &Option<String>) -> String {
            value
                .as_ref()
                .map_or_else(|| "None".to_owned(), |value| format!("{value:?}"))
        }
        fn number<T: std::fmt::Display>(value: &Option<T>) -> String {
            value
                .as_ref()
                .map_or_else(|| "None".to_owned(), |value| value.to_string())
        }
        // Every field matching actually turns on. Two signatures that differ
        // only in gain or binning have to look different, or a user asking
        // "why does this dark not match?" is shown a repr that hides why.
        format!(
            "FrameSignature(camera={}, telescope={}, filter={}, size={}x{}, channels={}, \
             binning={}x{}, gain={}, offset={}, readout_mode={}, bayer_pattern={}, \
             focal_length_mm={}, rotation_deg={}, exposure_seconds={}, camera_temp_c={}, \
             captured_at_unix={})",
            text(&self.inner.camera),
            text(&self.inner.telescope),
            text(&self.inner.filter),
            number(&self.inner.width),
            number(&self.inner.height),
            number(&self.inner.channels),
            number(&self.inner.binning_x),
            number(&self.inner.binning_y),
            number(&self.inner.gain),
            number(&self.inner.offset),
            number(&self.inner.readout_mode),
            text(&self.inner.bayer_pattern),
            number(&self.inner.focal_length_mm),
            number(&self.inner.rotation_deg),
            number(&self.inner.exposure_seconds),
            number(&self.inner.camera_temp_c),
            number(&self.inner.captured_at_unix)
        )
    }
}

/// How close two readings have to be to count as the same.
#[pyclass(from_py_object, name = "MatchTolerances", module = "seiza")]
#[derive(Clone, Copy)]
pub(crate) struct PyMatchTolerances {
    pub(crate) inner: MatchTolerances,
}

#[pymethods]
impl PyMatchTolerances {
    #[new]
    #[pyo3(signature = (
        exposure_seconds=None,
        exposure_fraction=None,
        dark_temperature_c=None,
        master_temperature_c=None,
        rotation_deg=None,
        focal_length_mm=None,
        flat_session_seconds=None,
        light_rotation=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        exposure_seconds: Option<f64>,
        exposure_fraction: Option<f64>,
        dark_temperature_c: Option<f64>,
        master_temperature_c: Option<f64>,
        rotation_deg: Option<f64>,
        focal_length_mm: Option<f64>,
        flat_session_seconds: Option<u64>,
        light_rotation: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut inner = MatchTolerances::default();
        if let Some(value) = exposure_seconds {
            inner.exposure_seconds = value;
        }
        if let Some(value) = exposure_fraction {
            inner.exposure_fraction = value;
        }
        if let Some(value) = dark_temperature_c {
            inner.dark_temperature_c = value;
        }
        if let Some(value) = master_temperature_c {
            inner.master_temperature_c = value;
        }
        if let Some(value) = rotation_deg {
            inner.rotation_deg = value;
        }
        if let Some(value) = focal_length_mm {
            inner.focal_length_mm = value;
        }
        if let Some(value) = flat_session_seconds {
            inner.flat_session_seconds = value;
        }
        if let Some(value) = light_rotation {
            inner.light_rotation = parse_light_rotation(value)?;
        }
        Ok(Self { inner })
    }

    #[getter]
    fn exposure_seconds(&self) -> f64 {
        self.inner.exposure_seconds
    }
    #[getter]
    fn exposure_fraction(&self) -> f64 {
        self.inner.exposure_fraction
    }
    #[getter]
    fn dark_temperature_c(&self) -> f64 {
        self.inner.dark_temperature_c
    }
    #[getter]
    fn master_temperature_c(&self) -> f64 {
        self.inner.master_temperature_c
    }
    #[getter]
    fn rotation_deg(&self) -> f64 {
        self.inner.rotation_deg
    }
    #[getter]
    fn focal_length_mm(&self) -> f64 {
        self.inner.focal_length_mm
    }
    #[getter]
    fn flat_session_seconds(&self) -> u64 {
        self.inner.flat_session_seconds
    }
    /// How a light's own rotator angle is read when a flat is matched to
    /// it: ``"recorded"``, an angle the host measured in degrees, or
    /// ``"unknown"`` when the angle does not decide the light's flats.
    #[getter]
    fn light_rotation(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(match self.inner.light_rotation {
            LightRotation::Recorded => "recorded".into_pyobject(py)?.into_any().unbind(),
            LightRotation::Measured { degrees } => degrees.into_pyobject(py)?.into_any().unbind(),
            LightRotation::Unknown => "unknown".into_pyobject(py)?.into_any().unbind(),
        })
    }
}

/// ``"recorded"``, ``"unknown"``, or a measured angle in degrees.
fn parse_light_rotation(value: &Bound<'_, PyAny>) -> PyResult<LightRotation> {
    if let Ok(text) = value.extract::<String>() {
        return match text.as_str() {
            "recorded" => Ok(LightRotation::Recorded),
            "unknown" => Ok(LightRotation::Unknown),
            _ => Err(PyValueError::new_err(
                "light_rotation must be \"recorded\", \"unknown\" or an angle in degrees",
            )),
        };
    }
    let degrees: f64 = value.extract().map_err(|_| {
        PyTypeError::new_err(
            "light_rotation must be \"recorded\", \"unknown\" or an angle in degrees",
        )
    })?;
    if !degrees.is_finite() {
        return Err(PyValueError::new_err(
            "a measured light rotation must be finite",
        ));
    }
    Ok(LightRotation::Measured { degrees })
}

pub(crate) fn tolerances_or_default(tolerances: Option<PyMatchTolerances>) -> MatchTolerances {
    tolerances_or(tolerances, MatchTolerances::default())
}

pub(crate) fn tolerances_or(
    tolerances: Option<PyMatchTolerances>,
    fallback: MatchTolerances,
) -> MatchTolerances {
    tolerances.map_or(fallback, |value| value.inner)
}

/// Whether two frames came off the same sensor in the same mode.
#[pyfunction]
#[pyo3(name = "sensor_matches")]
fn py_sensor_matches(reference: &PyFrameSignature, candidate: &PyFrameSignature) -> bool {
    sensor_matches(&reference.inner, &candidate.inner)
}

/// Whether a flat describes the same optical path as what it would correct.
#[pyfunction]
#[pyo3(name = "optics_match", signature = (reference, candidate, tolerances=None))]
fn py_optics_match(
    reference: &PyFrameSignature,
    candidate: &PyFrameSignature,
    tolerances: Option<PyMatchTolerances>,
) -> bool {
    optics_match(
        &reference.inner,
        &candidate.inner,
        &tolerances_or_default(tolerances),
    )
}

/// Every sensor field two frames disagree on, with both readings — for
/// example ``gain light=100 master=200`` — or a note that the signatures
/// disagree when no single field can be blamed.
#[pyfunction]
#[pyo3(name = "describe_sensor_mismatch")]
fn py_describe_sensor_mismatch(
    reference: &PyFrameSignature,
    candidate: &PyFrameSignature,
) -> String {
    seiza_calibration::describe_sensor_mismatch(&reference.inner, &candidate.inner)
}

/// Every optical field two frames disagree on, with both readings; rotation
/// additionally reports the gap against the tolerance — for example
/// ``rotation light=101.93deg master=104.24deg (2.31 deg apart, tolerance
/// 2.00)``.
#[pyfunction]
#[pyo3(name = "describe_optics_mismatch", signature = (reference, candidate, tolerances=None))]
fn py_describe_optics_mismatch(
    reference: &PyFrameSignature,
    candidate: &PyFrameSignature,
    tolerances: Option<PyMatchTolerances>,
) -> String {
    seiza_calibration::describe_optics_mismatch(
        &reference.inner,
        &candidate.inner,
        &tolerances_or_default(tolerances),
    )
}

/// Whether two rotator angles are close enough to share a flat. Wraps at 360,
/// and unknown on either side matches.
#[pyfunction]
#[pyo3(name = "rotation_matches", signature = (reference, candidate, tolerance_deg=None))]
fn py_rotation_matches(
    reference: Option<f64>,
    candidate: Option<f64>,
    tolerance_deg: Option<f64>,
) -> bool {
    // Defaulted from the tolerance struct rather than repeated here, so the
    // crate default and the Python default cannot drift apart.
    let tolerance_deg = tolerance_deg.unwrap_or(MatchTolerances::default().rotation_deg);
    rotation_matches(reference, candidate, tolerance_deg)
}

/// Whether a dark's exposure suits the frame it would be subtracted from.
/// Reads ``exposure_seconds`` from both signatures.
#[pyfunction]
#[pyo3(name = "exposure_matches", signature = (reference, candidate, tolerances=None))]
fn py_exposure_matches(
    reference: &PyFrameSignature,
    candidate: &PyFrameSignature,
    tolerances: Option<PyMatchTolerances>,
) -> bool {
    exposure_matches(
        &reference.inner,
        &candidate.inner,
        &tolerances_or_default(tolerances),
    )
}

/// Whether a dark's sensor temperature suits the frame it would be subtracted
/// from. Reads ``camera_temp_c`` from both signatures.
#[pyfunction]
#[pyo3(name = "temperature_matches", signature = (reference, candidate, tolerances=None))]
fn py_temperature_matches(
    reference: &PyFrameSignature,
    candidate: &PyFrameSignature,
    tolerances: Option<PyMatchTolerances>,
) -> bool {
    temperature_matches(
        &reference.inner,
        &candidate.inner,
        &tolerances_or_default(tolerances),
    )
}

/// The subset of ``candidates`` that can actually be averaged into one master.
///
/// Matching says what may be *used*; this says what may be *combined*. Frames
/// that each suit the light can still disagree with each other. Pass
/// [`coherent_subset`], as positions into ``candidates`` rather than copies.
///
/// A ``FrameSignature`` is only the part of a frame that decides whether it
/// belongs with another, so a copy cannot be traced back to whatever record it
/// came from. A caller holding its own objects — rows, paths, dataclasses —
/// asks for these and indexes its own list.
#[pyfunction]
#[pyo3(name = "coherent_subset_indices", signature = (candidates, flats=false, minimum=2, tolerances=None))]
fn py_coherent_subset_indices(
    candidates: Vec<PyFrameSignature>,
    flats: bool,
    minimum: usize,
    tolerances: Option<PyMatchTolerances>,
) -> Vec<usize> {
    let signatures: Vec<FrameSignature> = candidates
        .into_iter()
        .map(|candidate| candidate.inner)
        .collect();
    coherent_subset_indices(
        &signatures,
        if flats {
            FrameRole::Flat
        } else {
            FrameRole::Other
        },
        minimum,
        &tolerances_or_default(tolerances),
    )
}

/// ``flats=True`` to additionally require a shared session and rotator angle.
#[pyfunction]
#[pyo3(name = "coherent_subset", signature = (candidates, flats=false, minimum=2, tolerances=None))]
fn py_coherent_subset(
    candidates: Vec<PyFrameSignature>,
    flats: bool,
    minimum: usize,
    tolerances: Option<PyMatchTolerances>,
) -> Vec<PyFrameSignature> {
    let signatures: Vec<FrameSignature> = candidates
        .into_iter()
        .map(|candidate| candidate.inner)
        .collect();
    coherent_subset(
        &signatures,
        if flats {
            FrameRole::Flat
        } else {
            FrameRole::Other
        },
        minimum,
        &tolerances_or_default(tolerances),
    )
    .into_iter()
    .map(|inner| PyFrameSignature { inner })
    .collect()
}

/// Order candidates by how close in time they were shot to ``reference_unix``,
/// nearest first. Frames with no capture time sort last.
#[pyfunction]
#[pyo3(name = "sort_by_proximity", signature = (frames, reference_unix=None))]
fn py_sort_by_proximity(
    frames: Vec<PyFrameSignature>,
    reference_unix: Option<i64>,
) -> Vec<PyFrameSignature> {
    let mut signatures: Vec<FrameSignature> =
        frames.iter().map(|frame| frame.inner.clone()).collect();
    sort_by_proximity(&mut signatures, reference_unix);
    signatures
        .into_iter()
        .map(|inner| PyFrameSignature { inner })
        .collect()
}

/// Fit the pedestal in ``light``, in the light's own units.
///
/// Dividing by a flat only works on a signal that starts at zero, and without
/// a bias or dark master the camera's offset is still there. Sky background
/// varies with the flat's own response, so the intercept of that line is the
/// part that does not.
///
/// Returns ``None`` when the frame cannot support a fit: too few usable tiles,
/// a flat too uniform to give the line a lever, or a slope saying the model
/// does not describe this field. Carry on without a pedestal.
///
/// Raises ``ValueError`` when the arrays themselves are wrong — different
/// shapes, or more than one channel. That is a caller mistake rather than a
/// field this cannot fit, and keeping the two apart stops a bug from reading
/// as "too few tiles".
///
/// The fit reads low by roughly 0.8 times the frame's noise, by construction:
/// the per-tile sky is taken below the median so stars cannot drag it up. That
/// is the safe direction, and it cancels when comparing two frames.
#[pyfunction]
#[pyo3(name = "fit_flat_pedestal")]
fn py_fit_flat_pedestal(
    py: Python<'_>,
    light: PyReadonlyArrayDyn<'_, f32>,
    flat: PyReadonlyArrayDyn<'_, f32>,
) -> PyResult<Option<f32>> {
    let light = linear_image(light)?;
    let flat = linear_image(flat)?;
    py.detach(|| {
        let light = LinearImageRef::new(&light.data, light.width, light.height, light.channels)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let flat = LinearImageRef::new(&flat.data, flat.width, flat.height, flat.channels)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        fit_flat_pedestal(light, flat)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
    })
}

/// Calibration steps a frame's header says are already done.
#[pyclass(
    skip_from_py_object,
    frozen,
    name = "FrameCalibrationState",
    module = "seiza"
)]
#[derive(Clone, Copy)]
pub(crate) struct PyFrameCalibrationState {
    inner: FrameCalibrationState,
}

#[pymethods]
impl PyFrameCalibrationState {
    #[getter]
    fn bias_subtracted(&self) -> bool {
        self.inner.bias_subtracted
    }
    #[getter]
    fn dark_subtracted(&self) -> bool {
        self.inner.dark_subtracted
    }
    #[getter]
    fn flat_normalized(&self) -> bool {
        self.inner.flat_normalized
    }

    fn __repr__(&self) -> String {
        format!(
            "FrameCalibrationState(bias_subtracted={}, dark_subtracted={}, flat_normalized={})",
            python_bool(self.inner.bias_subtracted),
            python_bool(self.inner.dark_subtracted),
            python_bool(self.inner.flat_normalized),
        )
    }
}

fn python_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

/// What a frame's header says it is: its role, whether it is a master, what
/// it was shot with, and which calibration it has had.
#[pyclass(skip_from_py_object, frozen, name = "FrameMetadata", module = "seiza")]
#[derive(Clone)]
pub(crate) struct PyFrameMetadata {
    pub(crate) inner: FrameMetadata,
}

#[pymethods]
impl PyFrameMetadata {
    /// ``"bias"``, ``"dark"``, ``"dark-flat"``, ``"flat"``, ``"light"`` or
    /// ``"unknown"``.
    #[getter]
    fn role(&self) -> &'static str {
        self.inner.role.as_str()
    }
    #[getter]
    fn is_master(&self) -> bool {
        self.inner.is_master
    }
    #[getter]
    fn signature(&self) -> PyFrameSignature {
        PyFrameSignature {
            inner: self.inner.signature.clone(),
        }
    }
    #[getter]
    fn calibration_state(&self) -> PyFrameCalibrationState {
        PyFrameCalibrationState {
            inner: self.inner.calibration_state,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "FrameMetadata(role={:?}, is_master={})",
            self.role(),
            python_bool(self.inner.is_master),
        )
    }
}

/// A FITS or XISF header read by :func:`probe_frame`.
#[pyclass(skip_from_py_object, frozen, name = "FrameProbe", module = "seiza")]
#[derive(Clone)]
pub(crate) struct PyFrameProbe {
    inner: FrameProbe,
}

#[pymethods]
impl PyFrameProbe {
    #[getter]
    fn path(&self) -> PathBuf {
        self.inner.path.clone()
    }
    /// ``"FITS"`` or ``"XISF"``.
    #[getter]
    fn format(&self) -> &'static str {
        self.inner.format.name()
    }
    /// The frame type as the file wrote it (``IMAGETYP``, ``OBSTYPE`` or
    /// ``FRAME``), before it was read as a role.
    #[getter]
    fn raw_image_type(&self) -> Option<String> {
        self.inner.raw_image_type.clone()
    }
    #[getter]
    fn role(&self) -> &'static str {
        self.inner.metadata.role.as_str()
    }
    #[getter]
    fn is_master(&self) -> bool {
        self.inner.metadata.is_master
    }
    #[getter]
    fn signature(&self) -> PyFrameSignature {
        PyFrameSignature {
            inner: self.inner.metadata.signature.clone(),
        }
    }
    #[getter]
    fn calibration_state(&self) -> PyFrameCalibrationState {
        PyFrameCalibrationState {
            inner: self.inner.metadata.calibration_state,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "FrameProbe(path={:?}, format={:?}, role={:?}, is_master={})",
            self.inner.path.display().to_string(),
            self.format(),
            self.role(),
            python_bool(self.inner.metadata.is_master),
        )
    }
}

/// Read a FITS or XISF header into a :class:`FrameProbe`: the frame's role,
/// whether it is a master, its :class:`FrameSignature` and its calibration
/// state. No pixels are decoded.
#[pyfunction]
#[pyo3(name = "probe_frame")]
fn py_probe_frame(py: Python<'_>, path: PathBuf) -> PyResult<PyFrameProbe> {
    py.detach(|| probe_frame(&path))
        .map(|inner| PyFrameProbe { inner })
        .map_err(|error| crate::stacking::StackError::new_err(calibration_message(error)))
}

/// One frame :func:`plan_calibration` weighs, for a caller that holds its
/// own records rather than :class:`FrameProbe` results.
#[pyclass(
    skip_from_py_object,
    frozen,
    name = "CalibrationRecord",
    module = "seiza"
)]
#[derive(Clone)]
pub(crate) struct PyCalibrationRecord {
    inner: CalibrationRecord,
}

#[pymethods]
impl PyCalibrationRecord {
    #[new]
    #[pyo3(signature = (path, signature, *, role="unknown"))]
    fn new(path: PathBuf, signature: PyFrameSignature, role: &str) -> PyResult<Self> {
        let role = role
            .parse::<FrameSourceRole>()
            .map_err(|error| PyValueError::new_err(calibration_message(error)))?;
        Ok(Self {
            inner: CalibrationRecord {
                path,
                role,
                signature: signature.inner,
            },
        })
    }
    #[getter]
    fn path(&self) -> PathBuf {
        self.inner.path.clone()
    }
    #[getter]
    fn role(&self) -> &'static str {
        self.inner.role.as_str()
    }
    #[getter]
    fn signature(&self) -> PyFrameSignature {
        PyFrameSignature {
            inner: self.inner.signature.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CalibrationRecord(path={:?}, role={:?})",
            self.inner.path.display().to_string(),
            self.role(),
        )
    }
}

/// A :class:`CalibrationRecord`, or a :class:`FrameProbe` read as one.
fn calibration_record(value: &Bound<'_, PyAny>) -> PyResult<CalibrationRecord> {
    if let Ok(record) = value.cast::<PyCalibrationRecord>() {
        return Ok(record.get().inner.clone());
    }
    if let Ok(probe) = value.cast::<PyFrameProbe>() {
        let probe = &probe.get().inner;
        return Ok(CalibrationRecord {
            path: probe.path.clone(),
            role: probe.metadata.role,
            signature: probe.metadata.signature.clone(),
        });
    }
    Err(PyTypeError::new_err(
        "calibration records must be CalibrationRecord or FrameProbe objects",
    ))
}

/// The calibration frames :func:`plan_calibration` chose.
#[pyclass(
    skip_from_py_object,
    frozen,
    name = "CalibrationPlan",
    module = "seiza"
)]
pub(crate) struct PyCalibrationPlan {
    inner: CalibrationPlan,
}

#[pymethods]
impl PyCalibrationPlan {
    /// ``"bias"``, ``"dark"``, ``"dark-flat"`` or ``"flat"``.
    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind.as_str()
    }
    /// The fewest frames a usable set holds, at least one.
    #[getter]
    fn minimum(&self) -> usize {
        self.inner.minimum
    }
    /// Whether :attr:`selected_paths` holds at least :attr:`minimum` frames.
    #[getter]
    fn ready(&self) -> bool {
        self.inner.ready
    }
    /// Every candidate that suits all the frames served, nearest in time
    /// first.
    #[getter]
    fn matched_paths(&self) -> Vec<PathBuf> {
        self.inner.matched_paths.clone()
    }
    /// The matched candidates that can be combined into one master.
    #[getter]
    fn selected_paths(&self) -> Vec<PathBuf> {
        self.inner.selected_paths.clone()
    }
    /// Each candidate left out, as ``(path, reason)``: ``"role-mismatch"``,
    /// ``"sensor-mismatch"``, ``"missing-exposure"``, ``"exposure-mismatch"``,
    /// ``"temperature-mismatch"``, ``"optics-mismatch"`` or
    /// ``"outside-coherent-set"``.
    #[getter]
    fn excluded(&self) -> Vec<(PathBuf, &'static str)> {
        self.inner
            .excluded
            .iter()
            .map(|exclusion| (exclusion.path.clone(), exclusion.reason.as_str()))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "CalibrationPlan(kind={:?}, ready={}, selected={}, excluded={})",
            self.kind(),
            python_bool(self.inner.ready),
            self.inner.selected_paths.len(),
            self.inner.excluded.len(),
        )
    }
}

/// Choose the calibration frames that suit every frame they serve, and can
/// be combined with each other, under the whole matching policy.
///
/// ``kind`` is ``"bias"``, ``"dark"``, ``"dark-flat"`` or ``"flat"``.
/// ``reference`` is the light the frames serve (for dark flats, the flat
/// they calibrate); its capture time orders the candidates. ``references``
/// lists every frame served, ``reference`` among them, when there are
/// several. Records are :class:`FrameProbe` results or
/// :class:`CalibrationRecord` objects. ``bias_available`` says a usable bias
/// master exists, so darks of another exposure can be scaled.
///
/// Raises ``ValueError`` for a bad request: an unknown kind, references of
/// the wrong role, repeated paths, or tolerances that are negative or not
/// finite.
#[pyfunction]
#[pyo3(
    name = "plan_calibration",
    signature = (
        kind,
        reference,
        candidates,
        *,
        references=None,
        minimum=2,
        tolerances=None,
        bias_available=false
    )
)]
#[allow(clippy::too_many_arguments)]
fn py_plan_calibration(
    py: Python<'_>,
    kind: &str,
    reference: &Bound<'_, PyAny>,
    candidates: Vec<Bound<'_, PyAny>>,
    references: Option<Vec<Bound<'_, PyAny>>>,
    minimum: usize,
    tolerances: Option<PyMatchTolerances>,
    bias_available: bool,
) -> PyResult<PyCalibrationPlan> {
    let bad_request = |error| PyValueError::new_err(calibration_message(error));
    let request = CalibrationPlanRequest {
        kind: kind.parse::<CalibrationPlanKind>().map_err(bad_request)?,
        reference: calibration_record(reference)?,
        references: references
            .unwrap_or_default()
            .iter()
            .map(calibration_record)
            .collect::<PyResult<_>>()?,
        candidates: candidates
            .iter()
            .map(calibration_record)
            .collect::<PyResult<_>>()?,
        minimum,
        tolerances: tolerances_or_default(tolerances),
        bias_available,
    };
    py.detach(|| plan_calibration(request))
        .map(|inner| PyCalibrationPlan { inner })
        .map_err(bad_request)
}

/// A calibration error's own words, without the "calibration error: " its
/// display adds.
fn calibration_message(error: seiza_stacking::Error) -> String {
    match error {
        seiza_stacking::Error::Calibration(message) => message,
        other => other.to_string(),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyFrameSignature>()?;
    module.add_class::<PyMatchTolerances>()?;
    module.add_function(wrap_pyfunction!(py_sensor_matches, module)?)?;
    module.add_function(wrap_pyfunction!(py_optics_match, module)?)?;
    module.add_function(wrap_pyfunction!(py_rotation_matches, module)?)?;
    module.add_function(wrap_pyfunction!(py_describe_sensor_mismatch, module)?)?;
    module.add_function(wrap_pyfunction!(py_describe_optics_mismatch, module)?)?;
    module.add_function(wrap_pyfunction!(py_exposure_matches, module)?)?;
    module.add_function(wrap_pyfunction!(py_temperature_matches, module)?)?;
    module.add_function(wrap_pyfunction!(py_coherent_subset, module)?)?;
    module.add_function(wrap_pyfunction!(py_coherent_subset_indices, module)?)?;
    module.add_function(wrap_pyfunction!(py_sort_by_proximity, module)?)?;
    module.add_function(wrap_pyfunction!(py_fit_flat_pedestal, module)?)?;
    module.add_class::<PyFrameCalibrationState>()?;
    module.add_class::<PyFrameMetadata>()?;
    module.add_class::<PyFrameProbe>()?;
    module.add_class::<PyCalibrationRecord>()?;
    module.add_class::<PyCalibrationPlan>()?;
    module.add_function(wrap_pyfunction!(py_probe_frame, module)?)?;
    module.add_function(wrap_pyfunction!(py_plan_calibration, module)?)?;
    Ok(())
}
