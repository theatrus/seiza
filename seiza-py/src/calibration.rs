//! Deciding which calibration frames belong together, and recovering a
//! camera pedestal when no bias or dark is available.

use crate::arrays::linear_image;
use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::*;
use seiza_calibration::{
    FrameRole, FrameSignature, LinearImageRef, MatchTolerances, coherent_subset, exposure_matches,
    fit_flat_pedestal, optics_match, rotation_matches, sensor_matches, sort_by_proximity,
    temperature_matches,
};

/// What a frame was shot with, as far as matching cares.
///
/// Every field is optional. A missing value on the *candidate* side
/// disqualifies it and a missing value on the *reference* side accepts
/// whatever it is offered: a light that does not record its gain cannot rule
/// anything out, while a calibration frame that does not record its gain
/// cannot prove it belongs. Rotation is the exception — unknown on either
/// side matches, because treating a missing angle as a mismatch would strip
/// flats from every frame shot before anyone recorded one.
#[pyclass(name = "FrameSignature", module = "seiza")]
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
#[pyclass(name = "MatchTolerances", module = "seiza")]
#[derive(Clone, Copy)]
pub(crate) struct PyMatchTolerances {
    inner: MatchTolerances,
}

#[pymethods]
impl PyMatchTolerances {
    #[new]
    #[pyo3(signature = (
        exposure_seconds=None,
        dark_temperature_c=None,
        master_temperature_c=None,
        rotation_deg=None,
        focal_length_mm=None,
        flat_session_seconds=None,
    ))]
    fn new(
        exposure_seconds: Option<f64>,
        dark_temperature_c: Option<f64>,
        master_temperature_c: Option<f64>,
        rotation_deg: Option<f64>,
        focal_length_mm: Option<f64>,
        flat_session_seconds: Option<u64>,
    ) -> Self {
        let mut inner = MatchTolerances::default();
        if let Some(value) = exposure_seconds {
            inner.exposure_seconds = value;
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
        Self { inner }
    }

    #[getter]
    fn exposure_seconds(&self) -> f64 {
        self.inner.exposure_seconds
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
}

fn tolerances_or_default(tolerances: Option<PyMatchTolerances>) -> MatchTolerances {
    tolerances.map_or_else(MatchTolerances::default, |value| value.inner)
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
    py.allow_threads(|| {
        let light = LinearImageRef::new(&light.data, light.width, light.height, light.channels)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let flat = LinearImageRef::new(&flat.data, flat.width, flat.height, flat.channels)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        fit_flat_pedestal(light, flat)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyFrameSignature>()?;
    module.add_class::<PyMatchTolerances>()?;
    module.add_function(wrap_pyfunction!(py_sensor_matches, module)?)?;
    module.add_function(wrap_pyfunction!(py_optics_match, module)?)?;
    module.add_function(wrap_pyfunction!(py_rotation_matches, module)?)?;
    module.add_function(wrap_pyfunction!(py_exposure_matches, module)?)?;
    module.add_function(wrap_pyfunction!(py_temperature_matches, module)?)?;
    module.add_function(wrap_pyfunction!(py_coherent_subset, module)?)?;
    module.add_function(wrap_pyfunction!(py_sort_by_proximity, module)?)?;
    module.add_function(wrap_pyfunction!(py_fit_flat_pedestal, module)?)?;
    Ok(())
}
