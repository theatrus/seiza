//! Learned models — `seiza-model`.
//!
//! `NoiseModel` opens a model on a backend and runs it over an image, which is
//! all it needs. The `Denoised` that returns keeps what the model found, so
//! choosing how much noise to keep costs a blend and no inference.

use crate::arrays::{float_array, float_image_view};
use crate::stacking::{keep_first, python_cancel_signal};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use seiza_model::{
    Backend, BackendOptions, ChannelScales, DenoiseOptions, DenoiseReport, Denoised, ImageView,
    ModelBundle, ModelError, ModelStore, NoiseBands, NoiseModel, Precision, Progress, runtime_info,
};
use std::path::PathBuf;
use std::sync::Mutex;

/// The `onnxruntime` package's library, which `pip install seiza[ml]` brings,
/// for a caller that names none. `SEIZA_ONNXRUNTIME` still comes first. The
/// package is found, not imported, since that would load its own copy.
fn package_library(py: Python<'_>) -> Option<PathBuf> {
    if std::env::var_os("SEIZA_ONNXRUNTIME").is_some() {
        return None;
    }
    let spec = py
        .import("importlib.util")
        .ok()?
        .call_method1("find_spec", ("onnxruntime",))
        .ok()?;
    let package: PathBuf = spec.getattr("origin").ok()?.extract().ok()?;
    let entries = std::fs::read_dir(package.parent()?.join("capi")).ok()?;
    entries.flatten().map(|entry| entry.path()).find(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name == "onnxruntime.dll" || name.starts_with("libonnxruntime."))
    })
}

fn model_error(error: ModelError) -> PyErr {
    crate::EngineError::new_err(error.to_string())
}

fn store_at(store: Option<PathBuf>) -> PyResult<ModelStore> {
    match store {
        Some(root) => Ok(ModelStore::new(root)),
        None => ModelStore::default_store().map_err(model_error),
    }
}

/// The scales one channel was read in: its `background` and its median
/// `noise`, which is what σ̄ means in a model's output.
#[pyclass(skip_from_py_object, name = "ChannelScales", module = "seiza", frozen)]
#[derive(Clone)]
pub(crate) struct PyChannelScales {
    #[pyo3(get)]
    background: f32,
    #[pyo3(get)]
    noise: f32,
}

impl From<&ChannelScales> for PyChannelScales {
    fn from(scales: &ChannelScales) -> Self {
        Self {
            background: scales.background,
            noise: scales.noise,
        }
    }
}

#[pymethods]
impl PyChannelScales {
    fn __repr__(&self) -> String {
        format!(
            "ChannelScales(background={}, noise={})",
            self.background, self.noise
        )
    }
}

/// What ran, for a file's provenance and a cache key: a cache keys on
/// `sha256`, `backend`, `precision` and `MODEL_OUTPUT_VERSION`, not on the
/// tile, which does not change the output.
#[pyclass(skip_from_py_object, name = "DenoiseReport", module = "seiza", frozen)]
pub(crate) struct PyDenoiseReport {
    #[pyo3(get)]
    model: String,
    #[pyo3(get)]
    sha256: String,
    #[pyo3(get)]
    backend: String,
    #[pyo3(get)]
    precision: String,
    #[pyo3(get)]
    tile: usize,
    #[pyo3(get)]
    noise_map: String,
    #[pyo3(get)]
    channels: Vec<PyChannelScales>,
}

impl From<&DenoiseReport> for PyDenoiseReport {
    fn from(report: &DenoiseReport) -> Self {
        Self {
            model: report.model.clone(),
            sha256: report.sha256.clone(),
            backend: report.backend.to_string(),
            precision: report.precision.to_string(),
            tile: report.tile,
            noise_map: report.noise_map.to_string(),
            channels: report.channels.iter().map(Into::into).collect(),
        }
    }
}

#[pymethods]
impl PyDenoiseReport {
    fn __repr__(&self) -> String {
        format!(
            "DenoiseReport(model={:?}, backend={:?}, precision={:?}, tile={})",
            self.model, self.backend, self.precision, self.tile
        )
    }
}

/// A run's result. It keeps what the model found, so `blend` costs no
/// inference.
#[pyclass(skip_from_py_object, name = "Denoised", module = "seiza", frozen)]
pub(crate) struct PyDenoised {
    result: Denoised,
}

#[pymethods]
impl PyDenoised {
    /// The image with `high` of the grain and `low` of the mottle the model
    /// found taken out, split at a Gaussian of `scale_px`, shaped as the
    /// image given was.
    #[pyo3(signature = (high=1.0, low=1.0, scale_px=5.0))]
    fn blend<'py>(
        &self,
        py: Python<'py>,
        high: f32,
        low: f32,
        scale_px: f32,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let bands = NoiseBands {
            high,
            low,
            scale_px,
        };
        let blended = py
            .detach(|| self.result.blend(&bands))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        float_array(
            py,
            self.result.width(),
            self.result.height(),
            self.result.channels(),
            blended,
        )
    }

    /// What the model found in one channel: `f(x) − x`, in the image's units.
    #[pyo3(signature = (channel=0))]
    fn residual<'py>(
        &self,
        py: Python<'py>,
        channel: usize,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let found = self
            .result
            .residual(channel)
            .ok_or_else(|| PyIndexError::new_err(format!("no channel {channel}")))?;
        float_array(
            py,
            self.result.width(),
            self.result.height(),
            1,
            found.to_vec(),
        )
    }

    /// What ran.
    #[getter]
    fn report(&self) -> PyDenoiseReport {
        (&self.result.report).into()
    }

    fn __repr__(&self) -> String {
        format!(
            "Denoised(model={:?}, width={}, height={}, channels={})",
            self.result.report.model,
            self.result.width(),
            self.result.height(),
            self.result.channels()
        )
    }
}

/// A noise model, open on a backend. `model` names an installed model or a
/// bundle's directory; `None` means the one installed noise model. With no
/// `backend`, the best one this machine can run and the model's card allows
/// is chosen.
#[pyclass(skip_from_py_object, name = "NoiseModel", module = "seiza", frozen)]
pub(crate) struct PyNoiseModel {
    model: Mutex<NoiseModel>,
    backend: Backend,
    tile: usize,
}

#[pymethods]
impl PyNoiseModel {
    #[new]
    #[pyo3(signature = (model=None, backend=None, *, store=None, library=None, cache_dir=None, tile=None, precision=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        model: Option<String>,
        backend: Option<String>,
        store: Option<PathBuf>,
        library: Option<PathBuf>,
        cache_dir: Option<PathBuf>,
        tile: Option<usize>,
        precision: Option<String>,
    ) -> PyResult<Self> {
        let precision = precision
            .as_deref()
            .map(str::parse::<Precision>)
            .transpose()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let backend = backend
            .as_deref()
            .map(str::parse::<Backend>)
            .transpose()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let options = BackendOptions {
            library: library.or_else(|| package_library(py)),
            cache_dir,
            tile,
            precision,
        };
        let bundle = store_at(store)?
            .resolve(model.as_deref())
            .map_err(model_error)?;
        let opened = py.detach(|| {
            let backend = match backend {
                Some(backend) => backend,
                None => Backend::detect(&bundle, &options)?,
            };
            NoiseModel::open(&bundle, backend, &options)
        });
        let model = opened.map_err(model_error)?;
        Ok(Self {
            backend: model.backend(),
            tile: model.tiling().tile,
            model: Mutex::new(model),
        })
    }

    /// The backend this model runs on.
    #[getter]
    fn backend(&self) -> String {
        self.backend.to_string()
    }

    /// The side of the tiles this model runs.
    #[getter]
    fn tile(&self) -> usize {
        self.tile
    }

    /// Run the model over `image`, float32 of shape (height, width) or
    /// (height, width, 3), NaN where nothing covers a pixel. `progress` is
    /// called with (channel, channels, tile, tiles) after each tile, and
    /// `cancel`, a callable returning true, stops the run.
    #[pyo3(signature = (image, progress=None, cancel=None))]
    fn denoise(
        &self,
        py: Python<'_>,
        image: PyReadonlyArrayDyn<'_, f32>,
        progress: Option<Py<PyAny>>,
        cancel: Option<Py<PyAny>>,
    ) -> PyResult<PyDenoised> {
        let view = float_image_view(&image)?;
        let view = ImageView::new(view.data, view.width, view.height, view.channels)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let (signal, raised) = python_cancel_signal(cancel);
        let report = |step: Progress| {
            let Some(progress) = &progress else { return };
            let called = Python::attach(|py| {
                progress
                    .call1(py, (step.channel, step.channels, step.tile, step.tiles))
                    .map(|_| ())
            });
            if let Err(error) = called {
                keep_first(&raised, error);
            }
        };
        let stop = || signal.is_cancelled();
        let options = DenoiseOptions {
            progress: Some(&report),
            cancel: Some(&stop),
        };
        let result = py.detach(|| {
            let mut model = self
                .model
                .lock()
                .map_err(|_| ModelError::Runtime("a run on this model panicked".to_string()))?;
            model.denoise(view, &options)
        });
        if let Some(error) = raised.lock().ok().and_then(|mut slot| slot.take()) {
            return Err(error);
        }
        Ok(PyDenoised {
            result: result.map_err(model_error)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "NoiseModel(backend={:?}, tile={})",
            self.backend(),
            self.tile
        )
    }
}

/// Which ONNX Runtime this process loaded.
#[pyclass(skip_from_py_object, name = "ModelRuntime", module = "seiza", frozen)]
pub(crate) struct PyModelRuntime {
    #[pyo3(get)]
    path: PathBuf,
    #[pyo3(get)]
    version: String,
}

/// Whether a backend runs here: `unavailable` holds why it cannot, and
/// `automatic` whether a model picks it unasked.
#[pyclass(skip_from_py_object, name = "ModelBackend", module = "seiza", frozen)]
pub(crate) struct PyModelBackend {
    #[pyo3(get)]
    backend: String,
    #[pyo3(get)]
    unavailable: Option<String>,
    #[pyo3(get)]
    automatic: bool,
}

#[pymethods]
impl PyModelBackend {
    fn __repr__(&self) -> String {
        format!(
            "ModelBackend(backend={:?}, unavailable={:?}, automatic={})",
            self.backend, self.unavailable, self.automatic
        )
    }
}

/// An installed model.
#[pyclass(skip_from_py_object, name = "InstalledModel", module = "seiza", frozen)]
pub(crate) struct PyInstalledModel {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    path: PathBuf,
    #[pyo3(get)]
    task: String,
    #[pyo3(get)]
    noise_map: String,
    #[pyo3(get)]
    precisions: Vec<String>,
}

impl From<&ModelBundle> for PyInstalledModel {
    fn from(bundle: &ModelBundle) -> Self {
        let card = bundle.card();
        Self {
            name: card.name.clone(),
            path: bundle.root().to_path_buf(),
            task: card.task.to_string(),
            noise_map: card.reading.noise_map.to_string(),
            precisions: card
                .graphs
                .iter()
                .map(|graph| graph.precision.to_string())
                .collect(),
        }
    }
}

#[pymethods]
impl PyInstalledModel {
    fn __repr__(&self) -> String {
        format!("InstalledModel(name={:?}, task={:?})", self.name, self.task)
    }
}

/// Which backends run here, and why the others do not.
#[pyfunction]
#[pyo3(signature = (library=None))]
fn model_backends(py: Python<'_>, library: Option<PathBuf>) -> Vec<PyModelBackend> {
    let options = BackendOptions {
        library: library.or_else(|| package_library(py)),
        ..Default::default()
    };
    Backend::available(&options)
        .into_iter()
        .map(|status| PyModelBackend {
            backend: status.backend.to_string(),
            unavailable: status.unavailable,
            automatic: status.automatic,
        })
        .collect()
}

/// The ONNX Runtime this process loaded, or `None` when no library was found.
#[pyfunction]
#[pyo3(signature = (library=None))]
fn model_runtime(py: Python<'_>, library: Option<PathBuf>) -> Option<PyModelRuntime> {
    let options = BackendOptions {
        library: library.or_else(|| package_library(py)),
        ..Default::default()
    };
    runtime_info(&options).ok().map(|info| PyModelRuntime {
        path: info.path,
        version: info.version,
    })
}

/// The installed models.
#[pyfunction]
#[pyo3(signature = (store=None))]
fn installed_models(store: Option<PathBuf>) -> PyResult<Vec<PyInstalledModel>> {
    let models = store_at(store)?.list().map_err(model_error)?;
    Ok(models.iter().map(Into::into).collect())
}

/// Install the bundle at `source`, once its graphs match its card, replacing
/// any model of its name.
#[pyfunction]
#[pyo3(signature = (source, store=None))]
fn install_model(
    py: Python<'_>,
    source: PathBuf,
    store: Option<PathBuf>,
) -> PyResult<PyInstalledModel> {
    let store = store_at(store)?;
    let bundle = py.detach(|| store.install(source)).map_err(model_error)?;
    Ok((&bundle).into())
}

/// Check a model's graphs against its card, raising when one differs. `model`
/// is as for `NoiseModel`.
#[pyfunction]
#[pyo3(signature = (model=None, store=None))]
fn verify_model(py: Python<'_>, model: Option<String>, store: Option<PathBuf>) -> PyResult<()> {
    let store = store_at(store)?;
    py.detach(|| store.resolve(model.as_deref())?.verify())
        .map_err(model_error)
}

/// Delete the installed model named `name`.
#[pyfunction]
#[pyo3(signature = (name, store=None))]
fn remove_model(name: &str, store: Option<PathBuf>) -> PyResult<()> {
    store_at(store)?.remove(name).map_err(model_error)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNoiseModel>()?;
    m.add_class::<PyDenoised>()?;
    m.add_class::<PyDenoiseReport>()?;
    m.add_class::<PyChannelScales>()?;
    m.add_class::<PyModelBackend>()?;
    m.add_class::<PyModelRuntime>()?;
    m.add_class::<PyInstalledModel>()?;
    m.add_function(wrap_pyfunction!(model_backends, m)?)?;
    m.add_function(wrap_pyfunction!(model_runtime, m)?)?;
    m.add_function(wrap_pyfunction!(installed_models, m)?)?;
    m.add_function(wrap_pyfunction!(install_model, m)?)?;
    m.add_function(wrap_pyfunction!(verify_model, m)?)?;
    m.add_function(wrap_pyfunction!(remove_model, m)?)?;
    m.add("MODEL_OUTPUT_VERSION", seiza_model::OUTPUT_VERSION)?;
    Ok(())
}
