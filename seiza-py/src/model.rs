//! Learned models — `seiza-model`.
//!
//! `NoiseModel` opens a model on a backend and runs it over an image, which is
//! all it needs. The `Denoised` that returns keeps what the model found, so
//! choosing how much noise to keep costs a blend and no inference.
//! `SharpenModel` does the same for a sharpening model, which measures the
//! image's PSF, point sources and noise itself.

use crate::arrays::{float_array, float_image_view};
use crate::convert::to_python;
use crate::stacking::{keep_first, python_cancel_signal};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use seiza::downloads::{CachePolicy, DownloadEvent, ModelManager, OnnxRuntimeInstaller};
use seiza_model::{
    Backend, BackendOptions, ChannelScales, DenoiseOptions, DenoiseReport, Denoised,
    FileRestoration, HoleGuard, ImageView, ModelBundle, ModelError, ModelStore, NoiseBands,
    NoiseModel, Precision, Progress, Published, RestorationArea, RestorationImage, SharpenModel,
    SharpenOptions, SharpenReport, Sharpened, SharpenedChannel, Task, runtime_info,
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

/// How a model's card has the image's holes kept from the network: pixels
/// more than `depth` of the image map's noise below the sky's floor, widened by
/// `margin` pixels, are filled from the mean of the others within
/// `fill_radius`, and put back after where they differ from the output by more
/// than `restore` of the map's noise.
#[pyclass(
    skip_from_py_object,
    name = "HoleGuard",
    module = "seiza",
    frozen,
    eq,
    hash
)]
#[derive(Clone)]
pub(crate) struct PyHoleGuard {
    #[pyo3(get)]
    depth: f64,
    #[pyo3(get)]
    margin: usize,
    #[pyo3(get)]
    fill_radius: usize,
    #[pyo3(get)]
    restore: f64,
}

impl From<&HoleGuard> for PyHoleGuard {
    fn from(guard: &HoleGuard) -> Self {
        Self {
            depth: guard.depth,
            margin: guard.margin,
            fill_radius: guard.fill_radius,
            restore: guard.restore,
        }
    }
}

/// Two guards are equal when their settings are, bit for bit, so a cache keyed
/// on a guard finds it again.
impl PartialEq for PyHoleGuard {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl std::hash::Hash for PyHoleGuard {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key().hash(state);
    }
}

impl PyHoleGuard {
    fn key(&self) -> (u64, usize, usize, u64) {
        (
            self.depth.to_bits(),
            self.margin,
            self.fill_radius,
            self.restore.to_bits(),
        )
    }
}

#[pymethods]
impl PyHoleGuard {
    fn __repr__(&self) -> String {
        format!(
            "HoleGuard(depth={}, margin={}, fill_radius={}, restore={})",
            self.depth, self.margin, self.fill_radius, self.restore
        )
    }
}

/// What ran, for a file's provenance and a cache key: a cache keys on
/// `card_sha256`, `backend`, `precision` and `MODEL_OUTPUT_VERSION`, not on
/// the tile, which does not change the output. The card names its graphs by
/// their hashes and everything the image is read by.
#[pyclass(skip_from_py_object, name = "DenoiseReport", module = "seiza", frozen)]
pub(crate) struct PyDenoiseReport {
    #[pyo3(get)]
    model: String,
    #[pyo3(get)]
    sha256: String,
    #[pyo3(get)]
    card_sha256: String,
    #[pyo3(get)]
    backend: String,
    #[pyo3(get)]
    precision: String,
    #[pyo3(get)]
    tile: usize,
    #[pyo3(get)]
    noise_map: String,
    #[pyo3(get)]
    holes: Option<PyHoleGuard>,
    #[pyo3(get)]
    channels: Vec<PyChannelScales>,
}

impl From<&DenoiseReport> for PyDenoiseReport {
    fn from(report: &DenoiseReport) -> Self {
        Self {
            model: report.model.clone(),
            sha256: report.sha256.clone(),
            card_sha256: report.card_sha256.clone(),
            backend: report.backend.to_string(),
            precision: report.precision.to_string(),
            tile: report.tile,
            noise_map: report.noise_map.to_string(),
            holes: report.holes.as_ref().map(Into::into),
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
    #[pyo3(signature = (model=None, backend=None, *, store=None, library=None, cache_dir=None, tile=None, precision=None, device_id=None))]
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
        device_id: Option<i32>,
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
            device_id,
        };
        let bundle = store_at(store)?
            .resolve(model.as_deref(), Task::Denoise)
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
        let result = self.run_with_progress(py, progress, cancel, |model, options| {
            model.denoise(view, options).map_err(model_error)
        })?;
        Ok(PyDenoised { result })
    }

    /// Restore a full linear file, or its center crop, without first constructing a Python array.
    #[pyo3(signature = (path, *, center=false, region=None, progress=None, cancel=None))]
    #[allow(clippy::too_many_arguments)]
    fn denoise_file(
        &self,
        py: Python<'_>,
        path: PathBuf,
        center: bool,
        region: Option<(usize, usize, usize, usize)>,
        progress: Option<Py<PyAny>>,
        cancel: Option<Py<PyAny>>,
    ) -> PyResult<PyFileRestoration> {
        if center && region.is_some() {
            return Err(PyValueError::new_err("choose center or region, not both"));
        }
        let area = if let Some((x, y, width, height)) = region {
            RestorationArea::Region {
                x,
                y,
                width,
                height,
            }
        } else if center {
            RestorationArea::Center
        } else {
            RestorationArea::Full
        };
        let result = self.run_with_progress(py, progress, cancel, |model, options| {
            RestorationImage::open(&path, area)
                .and_then(|image| image.denoise(model, options))
                .map_err(|error| crate::EngineError::new_err(error.to_string()))
        })?;
        Ok(PyFileRestoration { result })
    }

    fn __repr__(&self) -> String {
        format!(
            "NoiseModel(backend={:?}, tile={})",
            self.backend(),
            self.tile
        )
    }
}

impl PyNoiseModel {
    fn run_with_progress<T: Send>(
        &self,
        py: Python<'_>,
        progress: Option<Py<PyAny>>,
        cancel: Option<Py<PyAny>>,
        run: impl FnOnce(&mut NoiseModel, &DenoiseOptions<'_>) -> PyResult<T> + Send,
    ) -> PyResult<T> {
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
                .map_err(|_| crate::EngineError::new_err("a run on this model panicked"))?;
            run(&mut model, &options)
        });
        if let Some(error) = raised.lock().ok().and_then(|mut slot| slot.take()) {
            return Err(error);
        }
        result
    }
}

/// A full image or center preview and its reusable model residual.
#[pyclass(
    skip_from_py_object,
    name = "FileRestoration",
    module = "seiza",
    frozen
)]
pub(crate) struct PyFileRestoration {
    result: FileRestoration,
}

#[pymethods]
impl PyFileRestoration {
    /// Blend and write a full-resolution linear file, retaining its metadata.
    #[pyo3(signature = (path, *, high=1.0, low=1.0, scale_px=5.0))]
    fn write_linear(
        &self,
        py: Python<'_>,
        path: PathBuf,
        high: f32,
        low: f32,
        scale_px: f32,
    ) -> PyResult<PyDenoiseReport> {
        let bands = NoiseBands {
            high,
            low,
            scale_px,
        };
        py.detach(|| self.result.write_linear(&path, &bands))
            .map_err(|error| crate::EngineError::new_err(error.to_string()))?;
        Ok(self.result.report().into())
    }

    /// Write the restored preview crop in linear samples, without full-frame WCS.
    #[pyo3(signature = (path, *, high=1.0, low=1.0, scale_px=5.0))]
    fn write_preview_linear(
        &self,
        py: Python<'_>,
        path: PathBuf,
        high: f32,
        low: f32,
        scale_px: f32,
    ) -> PyResult<PyDenoiseReport> {
        py.detach(|| {
            self.result.write_preview_linear(
                &path,
                &NoiseBands {
                    high,
                    low,
                    scale_px,
                },
            )
        })
        .map_err(|error| crate::EngineError::new_err(error.to_string()))?;
        Ok(self.result.report().into())
    }

    /// Write the regional result within the unchanged full source image.
    #[pyo3(signature = (source, output, *, high=1.0, low=1.0, scale_px=5.0))]
    fn write_preview_in_context(
        &self,
        py: Python<'_>,
        source: PathBuf,
        output: PathBuf,
        high: f32,
        low: f32,
        scale_px: f32,
    ) -> PyResult<PyDenoiseReport> {
        py.detach(|| {
            self.result.write_preview_in_context(
                &source,
                &output,
                &NoiseBands {
                    high,
                    low,
                    scale_px,
                },
            )
        })
        .map_err(|error| crate::EngineError::new_err(error.to_string()))?;
        Ok(self.result.report().into())
    }

    /// Blend and write the original and restored crop with one display stretch.
    #[pyo3(signature = (original, restored, *, high=1.0, low=1.0, scale_px=5.0))]
    fn write_comparison(
        &self,
        py: Python<'_>,
        original: PathBuf,
        restored: PathBuf,
        high: f32,
        low: f32,
        scale_px: f32,
    ) -> PyResult<PyDenoiseReport> {
        let bands = NoiseBands {
            high,
            low,
            scale_px,
        };
        py.detach(|| self.result.write_comparison(&original, &restored, &bands))
            .map_err(|error| crate::EngineError::new_err(error.to_string()))?;
        Ok(self.result.report().into())
    }
}

/// What one channel of a sharpened image was read by: its `background` and
/// median `noise`; the `psf_rows` and `psf_columns` of nodes its PSF field was
/// sampled at, the `psf_stars` it was measured from, and whether too few of
/// them left one PSF for the whole image (`psf_global`); the median of the
/// nodes' widths, `psf_width_px`; how many `points`, point sources, were
/// found and rendered; and how many pixels a side it was also sharpened
/// `binned` by, for the scaled pass (1: not at all).
#[pyclass(
    skip_from_py_object,
    name = "SharpenedChannel",
    module = "seiza",
    frozen
)]
#[derive(Clone)]
pub(crate) struct PySharpenedChannel {
    #[pyo3(get)]
    background: f32,
    #[pyo3(get)]
    noise: f32,
    #[pyo3(get)]
    psf_rows: usize,
    #[pyo3(get)]
    psf_columns: usize,
    #[pyo3(get)]
    psf_stars: usize,
    #[pyo3(get)]
    psf_global: bool,
    #[pyo3(get)]
    psf_width_px: f32,
    #[pyo3(get)]
    points: usize,
    #[pyo3(get)]
    binned: usize,
}

impl From<&SharpenedChannel> for PySharpenedChannel {
    fn from(read: &SharpenedChannel) -> Self {
        Self {
            background: read.background,
            noise: read.noise,
            psf_rows: read.psf_nodes.0,
            psf_columns: read.psf_nodes.1,
            psf_stars: read.psf_stars,
            psf_global: read.psf_global,
            psf_width_px: read.psf_width_px,
            points: read.points,
            binned: read.binned,
        }
    }
}

#[pymethods]
impl PySharpenedChannel {
    fn __repr__(&self) -> String {
        format!(
            "SharpenedChannel(psf_width_px={}, psf_stars={}, points={}, noise={}, binned={})",
            self.psf_width_px, self.psf_stars, self.points, self.noise, self.binned
        )
    }
}

/// What ran in a sharpening, for a file's provenance and a cache key: key
/// cached results on `card_sha256`, `backend`, `precision`, `stars`, `rest`,
/// `rolloff`, `binned_rest`, `full_size_only` and `MODEL_OUTPUT_VERSION`.
/// `sha256` is the body graph's. `binned_rest` is the share asked of the
/// rest on a binned image, None where the card has no scaled pass or the
/// run was held to full size.
#[pyclass(skip_from_py_object, name = "SharpenReport", module = "seiza", frozen)]
#[derive(Clone)]
pub(crate) struct PySharpenReport {
    #[pyo3(get)]
    model: String,
    #[pyo3(get)]
    sha256: String,
    #[pyo3(get)]
    card_sha256: String,
    #[pyo3(get)]
    backend: String,
    #[pyo3(get)]
    precision: String,
    #[pyo3(get)]
    tile: usize,
    #[pyo3(get)]
    stars: f32,
    #[pyo3(get)]
    rest: f32,
    #[pyo3(get)]
    rolloff: f32,
    #[pyo3(get)]
    binned_rest: Option<f32>,
    #[pyo3(get)]
    full_size_only: bool,
    #[pyo3(get)]
    channels: Vec<PySharpenedChannel>,
}

impl From<&SharpenReport> for PySharpenReport {
    fn from(report: &SharpenReport) -> Self {
        Self {
            model: report.model.clone(),
            sha256: report.sha256.clone(),
            card_sha256: report.card_sha256.clone(),
            backend: report.backend.to_string(),
            precision: report.precision.to_string(),
            tile: report.tile,
            stars: report.stars,
            rest: report.rest,
            rolloff: report.rolloff,
            binned_rest: report.binned_rest,
            full_size_only: report.full_size_only,
            channels: report.channels.iter().map(Into::into).collect(),
        }
    }
}

#[pymethods]
impl PySharpenReport {
    fn __repr__(&self) -> String {
        format!(
            "SharpenReport(model={:?}, backend={:?}, precision={:?}, stars={}, rest={})",
            self.model, self.backend, self.precision, self.stars, self.rest
        )
    }
}

/// A sharpened image and what ran.
#[pyclass(skip_from_py_object, name = "Sharpened", module = "seiza", frozen)]
pub(crate) struct PySharpened {
    result: Sharpened,
}

#[pymethods]
impl PySharpened {
    /// The sharpened image, shaped as the image given was. A pixel that was
    /// not finite is NaN.
    #[getter]
    fn image<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        float_array(
            py,
            self.result.width(),
            self.result.height(),
            self.result.channels(),
            self.result.image().to_vec(),
        )
    }

    /// What ran.
    #[getter]
    fn report(&self) -> PySharpenReport {
        (&self.result.report).into()
    }

    fn __repr__(&self) -> String {
        format!(
            "Sharpened(width={}, height={}, channels={})",
            self.result.width(),
            self.result.height(),
            self.result.channels()
        )
    }
}

/// A sharpening model, open on a backend. `model` names an installed model or
/// a bundle's directory; `None` means the one installed sharpening model. With
/// no `backend`, the best one this machine can run and the model's card allows
/// is chosen.
#[pyclass(skip_from_py_object, name = "SharpenModel", module = "seiza", frozen)]
pub(crate) struct PySharpenModel {
    model: Mutex<SharpenModel>,
    backend: Backend,
    tile: usize,
}

/// A sharpening's refusal as Python raises it: too few stars for a PSF as
/// `TooFewStarsError`, and a bad share or image as `ValueError`.
fn sharpen_error(error: ModelError) -> PyErr {
    match error {
        ModelError::PsfField(seiza_stars::PsfFieldError::TooFewStars { .. }) => {
            crate::psf_field::TooFewStarsError::new_err(error.to_string())
        }
        ModelError::InvalidOption(_) | ModelError::InvalidImage(_) => {
            PyValueError::new_err(error.to_string())
        }
        other => model_error(other),
    }
}

/// What a Python caller asks of a sharpening, before its progress callback
/// and cancel signal.
struct Asked {
    stars: Option<f32>,
    rest: Option<f32>,
    rolloff: f32,
    binned_rest: Option<f32>,
    full_size_only: bool,
}

impl PySharpenModel {
    /// Run `run` on the model with what is `asked`, the caller's progress
    /// callback and cancel signal, and the interpreter released.
    fn run<T: Send>(
        &self,
        py: Python<'_>,
        asked: Asked,
        progress: Option<Py<PyAny>>,
        cancel: Option<Py<PyAny>>,
        run: impl FnOnce(&mut SharpenModel, &SharpenOptions<'_>) -> PyResult<T> + Send,
    ) -> PyResult<T> {
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
        let options = SharpenOptions {
            stars: asked.stars,
            rest: asked.rest,
            rolloff: asked.rolloff,
            binned_rest: asked.binned_rest,
            full_size_only: asked.full_size_only,
            progress: Some(&report),
            cancel: Some(&stop),
        };
        let result = py.detach(|| {
            let mut model = self.model.lock().map_err(|_| {
                model_error(ModelError::Runtime(
                    "a run on this model panicked".to_string(),
                ))
            })?;
            run(&mut model, &options)
        });
        if let Some(error) = raised.lock().ok().and_then(|mut slot| slot.take()) {
            return Err(error);
        }
        result
    }
}

#[pymethods]
impl PySharpenModel {
    #[new]
    #[pyo3(signature = (model=None, backend=None, *, store=None, library=None, cache_dir=None, tile=None, precision=None, device_id=None))]
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
        device_id: Option<i32>,
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
            device_id,
        };
        let bundle = store_at(store)?
            .resolve(model.as_deref(), Task::Sharpen)
            .map_err(model_error)?;
        let opened = py.detach(|| {
            let backend = match backend {
                Some(backend) => backend,
                None => Backend::detect(&bundle, &options)?,
            };
            SharpenModel::open(&bundle, backend, &options)
        });
        let model = opened.map_err(model_error)?;
        Ok(Self {
            backend: model.backend(),
            tile: model.tiling().tile,
            model: Mutex::new(model),
        })
    }

    /// The backend this model's body runs on.
    #[getter]
    fn backend(&self) -> String {
        self.backend.to_string()
    }

    /// The side of the tiles this model runs.
    #[getter]
    fn tile(&self) -> usize {
        self.tile
    }

    /// Sharpen `image`, float32 of shape (height, width) or (height, width,
    /// 3), NaN where nothing covers a pixel. Its PSF and how that changes
    /// over the field, its point sources and its noise are measured from it,
    /// each channel's on its own. `stars` is the share of its width every
    /// star is brought to, and `rest` the share asked of everything that is
    /// not a star: above 0 and at most 1, which leaves things as they are,
    /// and the model's own when `None`. No star is asked to be narrower than
    /// the model's floor. `rolloff` holds back the finest part of the change
    /// where it lies under that many times the image's noise. Where the
    /// model's card has a scaled pass, an image whose stars are far wider
    /// than the model was trained on is also sharpened binned, toward the
    /// width the card names, and the two runs put together by scale:
    /// `binned_rest` is the share asked of what is not a star there (the
    /// card's when `None`), and `full_size_only` keeps the run at full size.
    /// `progress` is called with (channel, channels, tile, tiles) after each
    /// tile, the binned run's counted in, and `cancel`, a callable returning
    /// true, stops the run. An image with too few stars to measure a PSF from
    /// raises `TooFewStarsError`. Where only the binned image has too few, the
    /// channel is sharpened at full size alone.
    #[pyo3(signature = (image, stars=None, rest=None, *, rolloff=0.0, binned_rest=None, full_size_only=false, progress=None, cancel=None))]
    #[allow(clippy::too_many_arguments)]
    fn sharpen(
        &self,
        py: Python<'_>,
        image: PyReadonlyArrayDyn<'_, f32>,
        stars: Option<f32>,
        rest: Option<f32>,
        rolloff: f32,
        binned_rest: Option<f32>,
        full_size_only: bool,
        progress: Option<Py<PyAny>>,
        cancel: Option<Py<PyAny>>,
    ) -> PyResult<PySharpened> {
        let view = float_image_view(&image)?;
        let view = ImageView::new(view.data, view.width, view.height, view.channels)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let asked = Asked {
            stars,
            rest,
            rolloff,
            binned_rest,
            full_size_only,
        };
        let result = self.run(py, asked, progress, cancel, |model, options| {
            model.sharpen(view, options).map_err(sharpen_error)
        })?;
        Ok(PySharpened { result })
    }

    /// Sharpen the full linear FITS or XISF file at `path` and write the
    /// result to `output` as linear 32-bit FITS, with the input's geometry
    /// and cards that say what ran, without first constructing a Python
    /// array. The other arguments are `sharpen`'s.
    #[pyo3(signature = (path, output, stars=None, rest=None, *, rolloff=0.0, binned_rest=None, full_size_only=false, progress=None, cancel=None))]
    #[allow(clippy::too_many_arguments)]
    fn sharpen_file(
        &self,
        py: Python<'_>,
        path: PathBuf,
        output: PathBuf,
        stars: Option<f32>,
        rest: Option<f32>,
        rolloff: f32,
        binned_rest: Option<f32>,
        full_size_only: bool,
        progress: Option<Py<PyAny>>,
        cancel: Option<Py<PyAny>>,
    ) -> PyResult<PySharpenReport> {
        let file_error = |error: seiza_model::FileError| match error {
            seiza_model::FileError::Model(error) => sharpen_error(error),
            other => crate::EngineError::new_err(other.to_string()),
        };
        let asked = Asked {
            stars,
            rest,
            rolloff,
            binned_rest,
            full_size_only,
        };
        self.run(py, asked, progress, cancel, |model, options| {
            let image = RestorationImage::open(&path, RestorationArea::Full).map_err(file_error)?;
            let result = image.sharpen(model, options).map_err(file_error)?;
            result.write_linear(&output).map_err(file_error)?;
            Ok(result.report().into())
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "SharpenModel(backend={:?}, tile={})",
            self.backend(),
            self.tile
        )
    }
}

/// What an installed sharpening model asks for unless told: the shares of
/// their width `stars` and the `rest` are brought to; the narrowest a star is
/// asked to be, `floor_px`; the `sharpest` share it was trained to be asked
/// for; and, where its card has a scaled pass, the stars' width an image is
/// binned toward, `scaled_px`, and the share asked of the rest there,
/// `binned_rest` (both `None` without one).
#[pyclass(
    skip_from_py_object,
    name = "SharpenStrengths",
    module = "seiza",
    frozen
)]
#[derive(Clone)]
pub(crate) struct PySharpenStrengths {
    #[pyo3(get)]
    stars: f64,
    #[pyo3(get)]
    rest: f64,
    #[pyo3(get)]
    floor_px: f64,
    #[pyo3(get)]
    sharpest: f64,
    #[pyo3(get)]
    scaled_px: Option<f64>,
    #[pyo3(get)]
    binned_rest: Option<f64>,
}

#[pymethods]
impl PySharpenStrengths {
    fn __repr__(&self) -> String {
        format!(
            "SharpenStrengths(stars={}, rest={}, floor_px={}, sharpest={})",
            self.stars, self.rest, self.floor_px, self.sharpest
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
    card_sha256: String,
    #[pyo3(get)]
    noise_map: String,
    #[pyo3(get)]
    holes: Option<PyHoleGuard>,
    #[pyo3(get)]
    sharpen: Option<PySharpenStrengths>,
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
            card_sha256: bundle.card_sha256().to_string(),
            noise_map: card.reading.noise_map.to_string(),
            holes: card.reading.holes.as_ref().map(Into::into),
            sharpen: card.sharpen.as_ref().map(|sharpen| PySharpenStrengths {
                stars: sharpen.strengths.stars,
                rest: sharpen.strengths.rest,
                floor_px: sharpen.floor_px,
                sharpest: sharpen.sharpest,
                scaled_px: sharpen.scaled.map(|scaled| scaled.stars_px),
                binned_rest: sharpen.scaled.map(|scaled| scaled.rest),
            }),
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

/// Download the model `name` that Seiza's hosted manifest offers, or with no
/// name the latest noise model it names, and install it, replacing any model
/// of its name, once its card matches the manifest and its graphs match its
/// card. `base_url` reads another copy of the hosted models, and `cache_dir`
/// keeps the downloads somewhere other than the default cache. `progress`, when
/// given, is called with each step as a dict: its kind under `"event"`, as the
/// C ABI's callback has it, beside its fields. An exception it raises is raised
/// once the download ends.
#[pyfunction]
#[pyo3(signature = (name=None, store=None, *, base_url=None, cache_dir=None, progress=None))]
fn download_model(
    py: Python<'_>,
    name: Option<&str>,
    store: Option<PathBuf>,
    base_url: Option<String>,
    cache_dir: Option<PathBuf>,
    progress: Option<Py<PyAny>>,
) -> PyResult<PyInstalledModel> {
    let model = match name {
        Some(name) => Published::Named(name),
        None => Published::Latest(Task::Denoise),
    };
    let store = store_at(store)?;
    let mut builder = ModelManager::builder().policy(CachePolicy::ForceRefresh);
    if let Some(base_url) = base_url {
        builder = builder.base_url(base_url);
    }
    if let Some(cache_dir) = cache_dir {
        builder = builder.cache_dir(cache_dir);
    }
    let manager = builder.build().map_err(|error| model_error(error.into()))?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let raised = Mutex::new(None);
    let report = |event: DownloadEvent| {
        let Some(callback) = &progress else { return };
        Python::attach(|py| {
            if raised.lock().is_ok_and(|slot| slot.is_some()) {
                return;
            }
            let result = to_python(py, &event).and_then(|value| callback.call1(py, (value,)));
            if let Err(error) = result {
                keep_first(&raised, error);
            }
        });
    };
    let bundle = py.detach(|| runtime.block_on(store.download(&manager, model, report)));
    if let Some(error) = raised.lock().ok().and_then(|mut slot| slot.take()) {
        return Err(error);
    }
    Ok((&bundle.map_err(model_error)?).into())
}

/// Check models' graphs, and a sharpening model's encoder, against their
/// cards, raising at the first that differs. `model` names an installed model
/// or a bundle's directory; `None` checks every installed model, noise and
/// sharpening alike, and passes when none is installed.
#[pyfunction]
#[pyo3(signature = (model=None, store=None))]
fn verify_model(py: Python<'_>, model: Option<String>, store: Option<PathBuf>) -> PyResult<()> {
    let store = store_at(store)?;
    py.detach(|| store.verify(model.as_deref()))
        .map(drop)
        .map_err(model_error)
}

/// Delete the installed model named `name`.
#[pyfunction]
#[pyo3(signature = (name, store=None))]
fn remove_model(name: &str, store: Option<PathBuf>) -> PyResult<()> {
    store_at(store)?.remove(name).map_err(model_error)
}

/// Fetch the hosted model catalog without downloading any model files.
#[pyfunction]
#[pyo3(signature = (*, base_url=None, cache_dir=None))]
fn model_catalog(
    py: Python<'_>,
    base_url: Option<String>,
    cache_dir: Option<PathBuf>,
) -> PyResult<Py<PyAny>> {
    let catalog = py
        .detach(|| -> Result<_, ModelError> {
            let mut builder = ModelManager::builder().policy(CachePolicy::ForceRefresh);
            if let Some(url) = base_url {
                builder = builder.base_url(url);
            }
            if let Some(path) = cache_dir {
                builder = builder.cache_dir(path);
            }
            let manager = builder.build()?;
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|error| ModelError::Runtime(error.to_string()))?;
            Ok(runtime.block_on(manager.manifest())?)
        })
        .map_err(model_error)?;
    to_python(py, &catalog).map(Bound::unbind)
}

/// Verify the shared installed runtime without downloading or loading it.
#[pyfunction]
fn installed_onnxruntime(py: Python<'_>, directory: PathBuf) -> PyResult<Py<PyAny>> {
    let installed = py
        .detach(|| OnnxRuntimeInstaller::new(directory)?.installed())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    to_python(py, &installed).map(Bound::unbind)
}

/// Install the pinned CPU runtime and its dependencies. Downloads are verified
/// and installed atomically. Existing valid files work offline. Does not load
/// the library. Progress receives dictionaries; cancel returns True to stop.
#[pyfunction]
#[pyo3(signature = (directory, *, progress=None, cancel=None))]
fn install_onnxruntime(
    py: Python<'_>,
    directory: PathBuf,
    progress: Option<Py<PyAny>>,
    cancel: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let installer = OnnxRuntimeInstaller::new(directory)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let raised = Mutex::new(None);
    let report = |event: DownloadEvent| {
        let Some(callback) = &progress else { return };
        Python::attach(|py| {
            if raised.lock().is_ok_and(|slot| slot.is_some()) {
                return;
            }
            if let Err(error) = to_python(py, &event).and_then(|value| callback.call1(py, (value,)))
            {
                keep_first(&raised, error);
            }
        });
    };
    let stop = || {
        if raised.lock().is_ok_and(|slot| slot.is_some()) {
            return true;
        }
        let Some(callback) = &cancel else {
            return false;
        };
        Python::attach(|py| {
            match callback
                .call0(py)
                .and_then(|value| value.extract::<bool>(py))
            {
                Ok(cancelled) => cancelled,
                Err(error) => {
                    keep_first(&raised, error);
                    true
                }
            }
        })
    };
    let installed = py.detach(|| runtime.block_on(installer.install(report, stop)));
    if let Some(error) = raised.lock().ok().and_then(|mut slot| slot.take()) {
        return Err(error);
    }
    let installed = installed.map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    to_python(py, &installed).map(Bound::unbind)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(installed_onnxruntime, m)?)?;
    m.add_function(wrap_pyfunction!(install_onnxruntime, m)?)?;
    m.add_class::<PyFileRestoration>()?;
    m.add_class::<PyNoiseModel>()?;
    m.add_class::<PyDenoised>()?;
    m.add_class::<PyDenoiseReport>()?;
    m.add_class::<PyChannelScales>()?;
    m.add_class::<PyHoleGuard>()?;
    m.add_class::<PyModelBackend>()?;
    m.add_class::<PyModelRuntime>()?;
    m.add_class::<PyInstalledModel>()?;
    m.add_class::<PySharpenModel>()?;
    m.add_class::<PySharpened>()?;
    m.add_class::<PySharpenReport>()?;
    m.add_class::<PySharpenedChannel>()?;
    m.add_class::<PySharpenStrengths>()?;
    m.add_function(wrap_pyfunction!(model_backends, m)?)?;
    m.add_function(wrap_pyfunction!(model_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(model_runtime, m)?)?;
    m.add_function(wrap_pyfunction!(installed_models, m)?)?;
    m.add_function(wrap_pyfunction!(install_model, m)?)?;
    m.add_function(wrap_pyfunction!(download_model, m)?)?;
    m.add_function(wrap_pyfunction!(verify_model, m)?)?;
    m.add_function(wrap_pyfunction!(remove_model, m)?)?;
    m.add("MODEL_OUTPUT_VERSION", seiza_model::OUTPUT_VERSION)?;
    Ok(())
}
