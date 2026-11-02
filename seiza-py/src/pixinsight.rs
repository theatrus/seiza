//! PixInsight's MLDenoise, run headless by an installed PixInsight —
//! `seiza-stacking`'s `pixinsight` module.
//!
//! [`pixinsight_ml_denoise_image`] denoises an array in linear units,
//! mapped onto PixInsight's 0..1 range and back, and
//! [`pixinsight_ml_denoise_file`] denoises a FITS or XISF file.
//! [`pixinsight_ml_denoise_model`] reads a model file's header.

use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use seiza_stacking::{
    MlDenoiseLocalSupport, MlDenoiseModel, MlDenoiseOptions, MlDenoisePrecision, PixInsight,
};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::rc_astro::cancel_signal;

/// MLDenoise's local support: a mask made from the image's brightness that
/// protects bright structures. `clip_low` (-20 to 20, in robust sigma from
/// the median) sets where the mask clips, `background` (0 to 1) the mask's
/// mean background level, and `smoothness` (0 to 7 pixels) the radius of
/// the median filter that smooths it.
#[pyclass(
    skip_from_py_object,
    name = "MlDenoiseLocalSupport",
    module = "seiza",
    frozen
)]
#[derive(Clone)]
pub(crate) struct PyMlDenoiseLocalSupport {
    #[pyo3(get)]
    clip_low: f64,
    #[pyo3(get)]
    background: f64,
    #[pyo3(get)]
    smoothness: u32,
}

#[pymethods]
impl PyMlDenoiseLocalSupport {
    #[new]
    #[pyo3(signature = (*, clip_low=0.0, background=0.25, smoothness=2))]
    fn new(clip_low: f64, background: f64, smoothness: u32) -> Self {
        Self {
            clip_low,
            background,
            smoothness,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "MlDenoiseLocalSupport(clip_low={}, background={}, smoothness={})",
            self.clip_low, self.background, self.smoothness
        )
    }
}

/// What an MLDenoise model file declares in its header. `header_sha256`
/// identifies the model: a host caching results keys on it with the
/// PixInsight version and the options. Two releases of a model can share
/// `weights_sha256` and still give different results.
#[pyclass(skip_from_py_object, name = "MlDenoiseModel", module = "seiza", frozen)]
#[derive(Clone)]
pub(crate) struct PyMlDenoiseModel {
    #[pyo3(get)]
    process: String,
    #[pyo3(get)]
    format_version: String,
    #[pyo3(get)]
    created: Option<String>,
    #[pyo3(get)]
    mono: bool,
    #[pyo3(get)]
    rgb: bool,
    #[pyo3(get)]
    weights_sha256: Option<String>,
    #[pyo3(get)]
    header_sha256: String,
}

impl From<MlDenoiseModel> for PyMlDenoiseModel {
    fn from(model: MlDenoiseModel) -> Self {
        Self {
            process: model.process,
            format_version: model.format_version,
            created: model.created,
            mono: model.mono,
            rgb: model.rgb,
            weights_sha256: model.weights_sha256,
            header_sha256: model.header_sha256,
        }
    }
}

#[pymethods]
impl PyMlDenoiseModel {
    fn __repr__(&self) -> String {
        format!(
            "MlDenoiseModel(process={:?}, format_version={:?}, mono={}, rgb={})",
            self.process,
            self.format_version,
            if self.mono { "True" } else { "False" },
            if self.rgb { "True" } else { "False" }
        )
    }
}

/// A file MLDenoise processed, from [`pixinsight_ml_denoise_file`].
#[pyclass(skip_from_py_object, name = "MlDenoiseRun", module = "seiza", frozen)]
pub(crate) struct PyMlDenoiseRun {
    /// The file written.
    #[pyo3(get)]
    output: PathBuf,
    /// The PixInsight version that ran, such as "1.9.6".
    #[pyo3(get)]
    pixinsight_version: String,
    /// The model's header.
    #[pyo3(get)]
    model: PyMlDenoiseModel,
    /// Seconds PixInsight spent opening, denoising and saving the image.
    #[pyo3(get)]
    seconds: f64,
}

/// An image MLDenoise processed in memory, from
/// [`pixinsight_ml_denoise_image`].
#[pyclass(skip_from_py_object, name = "MlDenoised", module = "seiza", frozen)]
pub(crate) struct PyMlDenoised {
    /// The denoised image, in the input's units and shape.
    #[pyo3(get)]
    image: Py<numpy::PyArrayDyn<f32>>,
    /// The PixInsight version that ran.
    #[pyo3(get)]
    pixinsight_version: String,
    /// The model's header.
    #[pyo3(get)]
    model: PyMlDenoiseModel,
    /// Seconds PixInsight spent opening, denoising and saving the image.
    #[pyo3(get)]
    seconds: f64,
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMlDenoiseLocalSupport>()?;
    module.add_class::<PyMlDenoiseModel>()?;
    module.add_class::<PyMlDenoiseRun>()?;
    module.add_class::<PyMlDenoised>()?;
    module.add_function(wrap_pyfunction!(pixinsight_locate, module)?)?;
    module.add_function(wrap_pyfunction!(pixinsight_ml_denoise_model, module)?)?;
    module.add_function(wrap_pyfunction!(pixinsight_ml_denoise_file, module)?)?;
    module.add_function(wrap_pyfunction!(pixinsight_ml_denoise_image, module)?)?;
    Ok(())
}

fn pixinsight(executable: Option<PathBuf>) -> PyResult<PixInsight> {
    match executable {
        Some(path) => Ok(PixInsight::with_executable(path)),
        None => PixInsight::locate()
            .ok_or_else(|| PyFileNotFoundError::new_err("PixInsight was not found")),
    }
}

/// A refused request is a `ValueError`; anything else that stopped a run
/// is a `RuntimeError`.
fn python_error(error: seiza_stacking::Error) -> PyErr {
    match error {
        seiza_stacking::Error::ExternalToolRefused { .. } => {
            PyValueError::new_err(error.to_string())
        }
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

fn existing_model(model: PathBuf) -> PyResult<PathBuf> {
    if model.exists() {
        Ok(model)
    } else {
        Err(PyFileNotFoundError::new_err(format!(
            "no MLDenoise model at {}",
            model.display()
        )))
    }
}

#[allow(clippy::too_many_arguments)]
fn options(
    model: PathBuf,
    amount: f64,
    overlap_percent: u32,
    precision: &str,
    local_support: Option<PyMlDenoiseLocalSupport>,
    timeout_seconds: Option<f64>,
) -> PyResult<MlDenoiseOptions> {
    let precision: MlDenoisePrecision = precision.parse().map_err(python_error)?;
    let options = MlDenoiseOptions {
        amount,
        overlap_percent,
        precision,
        local_support: local_support.map(|support| MlDenoiseLocalSupport {
            clip_low: support.clip_low,
            background: support.background,
            smoothness: support.smoothness,
        }),
        timeout_seconds,
        ..MlDenoiseOptions::new(model)
    };
    options.validate().map_err(python_error)?;
    Ok(options)
}

/// The PixInsight executable in its standard install location, or on
/// PATH, or None when PixInsight is not installed.
#[pyfunction]
pub(crate) fn pixinsight_locate() -> Option<String> {
    PixInsight::locate().map(|pixinsight| pixinsight.executable().display().to_string())
}

/// Read the header of an MLDenoise model file (`.xmlm`), without reading
/// its weights.
#[pyfunction]
pub(crate) fn pixinsight_ml_denoise_model(path: PathBuf) -> PyResult<PyMlDenoiseModel> {
    let path = existing_model(path)?;
    MlDenoiseModel::read(&path)
        .map(PyMlDenoiseModel::from)
        .map_err(python_error)
}

/// Denoise a FITS or XISF file with PixInsight's MLDenoise and write the
/// result to `output` (FITS or XISF, by extension) as 32-bit floats on the
/// input's scale, with the input's headers. `model` is the `.xmlm` file.
/// `amount` (0 to 1) blends the denoised image with the original,
/// `overlap_percent` (12 to 50) sets how much inference tiles overlap,
/// `precision` is "fast" or "high" GPU inference, and `local_support`
/// protects bright structures. `timeout_seconds` closes PixInsight and fails
/// the run after that long; the default is ten minutes plus 20 seconds per
/// million samples. PixInsight reports no progress; Ctrl-C or `cancel`
/// returning true stops the run within a quarter of a second and closes
/// PixInsight. A refused request raises `ValueError`, a missing model
/// `FileNotFoundError`, and a failed run `RuntimeError`.
#[pyfunction]
#[pyo3(signature = (input, output, model, *, amount=0.9, overlap_percent=25, precision="fast", local_support=None, timeout_seconds=None, executable=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn pixinsight_ml_denoise_file(
    py: Python<'_>,
    input: PathBuf,
    output: PathBuf,
    model: PathBuf,
    amount: f64,
    overlap_percent: u32,
    precision: &str,
    local_support: Option<PyRef<'_, PyMlDenoiseLocalSupport>>,
    timeout_seconds: Option<f64>,
    executable: Option<PathBuf>,
    cancel: Option<Py<PyAny>>,
) -> PyResult<PyMlDenoiseRun> {
    let options = options(
        existing_model(model)?,
        amount,
        overlap_percent,
        precision,
        local_support.map(|support| support.clone()),
        timeout_seconds,
    )?;
    let pixinsight = pixinsight(executable)?;
    let raised: Arc<Mutex<Option<PyErr>>> = Arc::new(Mutex::new(None));
    let signal = cancel_signal(&raised, cancel);
    let outcome =
        py.detach(|| pixinsight.ml_denoise_file(&input, &output, &options, Some(&signal)));
    if let Ok(mut slot) = raised.lock()
        && let Some(error) = slot.take()
    {
        return Err(error);
    }
    let run = outcome.map_err(python_error)?;
    Ok(PyMlDenoiseRun {
        output: run.output,
        pixinsight_version: run.pixinsight_version,
        model: run.model.into(),
        seconds: run.seconds,
    })
}

/// Denoise an image in memory with PixInsight's MLDenoise: a float32 array
/// of shape (height, width) or (height, width, 3). PixInsight works on
/// 0..1: an image within 0..1 keeps its scale, and any other is offset by
/// its most negative sample and divided by 65535 or its span, whichever is
/// larger, as PixInsight opens a 16-bit camera frame. The result is mapped
/// back, so it returns in the input's units and shape. NaN and infinite
/// samples, such as a stack's uncovered edges, come back unchanged. A one-channel image needs
/// a model with a mono route and an RGB image one with an RGB route. The
/// options work as for `pixinsight_ml_denoise_file`; no file is left
/// behind.
#[pyfunction]
#[pyo3(signature = (image, model, *, amount=0.9, overlap_percent=25, precision="fast", local_support=None, timeout_seconds=None, executable=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn pixinsight_ml_denoise_image<'py>(
    py: Python<'py>,
    image: numpy::PyReadonlyArrayDyn<'py, f32>,
    model: PathBuf,
    amount: f64,
    overlap_percent: u32,
    precision: &str,
    local_support: Option<PyRef<'_, PyMlDenoiseLocalSupport>>,
    timeout_seconds: Option<f64>,
    executable: Option<PathBuf>,
    cancel: Option<Py<PyAny>>,
) -> PyResult<PyMlDenoised> {
    let linear = crate::arrays::linear_image(image)?;
    let options = options(
        existing_model(model)?,
        amount,
        overlap_percent,
        precision,
        local_support.map(|support| support.clone()),
        timeout_seconds,
    )?;
    let pixinsight = pixinsight(executable)?;
    let raised: Arc<Mutex<Option<PyErr>>> = Arc::new(Mutex::new(None));
    let signal = cancel_signal(&raised, cancel);
    let outcome = py.detach(|| pixinsight.ml_denoise_image(&linear, &options, Some(&signal)));
    if let Ok(mut slot) = raised.lock()
        && let Some(error) = slot.take()
    {
        return Err(error);
    }
    let denoised = outcome.map_err(python_error)?;
    Ok(PyMlDenoised {
        image: crate::arrays::into_image_array(py, denoised.image)?.unbind(),
        pixinsight_version: denoised.pixinsight_version,
        model: denoised.model.into(),
        seconds: denoised.seconds,
    })
}
