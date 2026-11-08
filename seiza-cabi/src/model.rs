//! Learned models (`seiza-model`), as Python's `NoiseModel` runs them.
//!
//! [`seiza_noise_model_open`] opens a model on a backend, and
//! [`seiza_noise_model_denoise`] runs it over an image, which is all it needs.
//! The [`SeizaDenoised`] that returns keeps what the model found, so
//! [`seiza_denoised_blend`] chooses how much noise to keep with no second
//! inference. [`seiza_model_backends_json`] says which backends run here and
//! why the others do not.

use super::{
    SeizaCancelSignal, clear_error, ffi_result, optional_path, owned_json, required_f32_slice,
    required_f32_slice_mut, required_path, required_str,
};
use seiza_model::{
    Backend, BackendOptions, DenoiseOptions, DenoiseReport, Denoised, HoleGuard, ImageView,
    ModelBundle, ModelStore, NoiseBands, NoiseModel, Precision, Progress, runtime_info,
};
use serde::{Deserialize, Serialize};
use std::ffi::{CString, c_char, c_void};
use std::path::PathBuf;
use std::ptr;
use std::sync::atomic::Ordering;

/// An opaque noise model, open on a backend. Release it with
/// [`seiza_noise_model_free`]. One thread may run it at a time.
pub struct SeizaNoiseModel {
    model: NoiseModel,
}

/// A run's result. Release it with [`seiza_denoised_free`]. It is read-only,
/// and what it lends stays valid until it is freed.
pub struct SeizaDenoised {
    result: Denoised,
    report_json: CString,
}

/// Told after each tile, on the thread running the model: the channel from 0,
/// the channels, the tiles done in this channel, and its tiles.
pub type SeizaModelProgressCallback =
    Option<unsafe extern "C" fn(usize, usize, usize, usize, *mut c_void)>;

/// How a model is found and run. Every field may be left out: `model` then
/// means the one installed noise model and `backend` the best one here.
#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
struct ModelOpenRequest {
    backend: Option<String>,
    store: Option<PathBuf>,
    library: Option<PathBuf>,
    cache_dir: Option<PathBuf>,
    tile: Option<usize>,
    precision: Option<String>,
}

impl ModelOpenRequest {
    fn parse(json: *const c_char) -> Result<Self, String> {
        if json.is_null() {
            return Ok(Self::default());
        }
        let json = required_str(json, "model options")?;
        if json.is_empty() {
            return Ok(Self::default());
        }
        serde_json::from_str(&json).map_err(|error| format!("invalid model options: {error}"))
    }

    fn options(&self) -> Result<BackendOptions, String> {
        let precision = self
            .precision
            .as_deref()
            .map(str::parse::<Precision>)
            .transpose()
            .map_err(|error| error.to_string())?;
        Ok(BackendOptions {
            library: self.library.clone(),
            cache_dir: self.cache_dir.clone(),
            tile: self.tile,
            precision,
        })
    }
}

fn store_at(store: Option<PathBuf>) -> Result<ModelStore, String> {
    match store {
        Some(root) => Ok(ModelStore::new(root)),
        None => ModelStore::default_store().map_err(|error| error.to_string()),
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelRuntimeResponse {
    path: PathBuf,
    version: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelBackendResponse {
    backend: String,
    unavailable: Option<String>,
    automatic: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelBackendsResponse {
    runtime: Option<ModelRuntimeResponse>,
    backends: Vec<ModelBackendResponse>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ChannelScalesResponse {
    background: f32,
    noise: f32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct HoleGuardResponse {
    depth: f64,
    margin: usize,
    fill_radius: usize,
    restore: f64,
}

impl From<&HoleGuard> for HoleGuardResponse {
    fn from(guard: &HoleGuard) -> Self {
        Self {
            depth: guard.depth,
            margin: guard.margin,
            fill_radius: guard.fill_radius,
            restore: guard.restore,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DenoiseReportResponse {
    model: String,
    sha256: String,
    backend: String,
    precision: String,
    tile: usize,
    noise_map: String,
    holes: Option<HoleGuardResponse>,
    channels: Vec<ChannelScalesResponse>,
}

impl From<&DenoiseReport> for DenoiseReportResponse {
    fn from(report: &DenoiseReport) -> Self {
        Self {
            model: report.model.clone(),
            sha256: report.sha256.clone(),
            backend: report.backend.to_string(),
            precision: report.precision.to_string(),
            tile: report.tile,
            noise_map: report.noise_map.to_string(),
            holes: report.holes.as_ref().map(Into::into),
            channels: report
                .channels
                .iter()
                .map(|scales| ChannelScalesResponse {
                    background: scales.background,
                    noise: scales.noise,
                })
                .collect(),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct InstalledModelResponse {
    name: String,
    path: PathBuf,
    task: String,
    noise_map: String,
    holes: Option<HoleGuardResponse>,
    precisions: Vec<String>,
}

impl From<&ModelBundle> for InstalledModelResponse {
    fn from(bundle: &ModelBundle) -> Self {
        let card = bundle.card();
        Self {
            name: card.name.clone(),
            path: bundle.root().to_path_buf(),
            task: card.task.to_string(),
            noise_map: card.reading.noise_map.to_string(),
            holes: card.reading.holes.as_ref().map(Into::into),
            precisions: card
                .graphs
                .iter()
                .map(|graph| graph.precision.to_string())
                .collect(),
        }
    }
}

#[unsafe(no_mangle)]
/// Open a noise model.
///
/// `model` names an installed model or a bundle's directory; null means the one
/// installed noise model. `options_json` may be null, or hold `backend`,
/// `store`, `library`, `cacheDir`, `tile` and `precision`. With no `backend`,
/// the best one this machine can run and the model's card allows is chosen.
/// The first open on Core ML compiles the model, which can take some time.
/// Release the result with [`seiza_noise_model_free`].
///
/// # Safety
/// `model` and `options_json` must be null or NUL-terminated strings. When
/// non-null, `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_noise_model_open(
    model: *const c_char,
    options_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut SeizaNoiseModel {
    clear_error(error_out);
    ffi_result(error_out, || {
        let chosen = if model.is_null() {
            None
        } else {
            Some(required_str(model, "model")?)
        };
        let request = ModelOpenRequest::parse(options_json)?;
        let options = request.options()?;
        let bundle = store_at(request.store.clone())?
            .resolve(chosen.as_deref())
            .map_err(|error| error.to_string())?;
        let backend = match request.backend.as_deref() {
            Some(word) => word.parse::<Backend>().map_err(|error| error.to_string())?,
            None => Backend::detect(&bundle, &options).map_err(|error| error.to_string())?,
        };
        let model =
            NoiseModel::open(&bundle, backend, &options).map_err(|error| error.to_string())?;
        Ok(Box::into_raw(Box::new(SeizaNoiseModel { model })))
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// The backend an open model runs on, as a string the caller frees with
/// `seiza_string_free`.
///
/// # Safety
/// `model` must be a live [`SeizaNoiseModel`]. When non-null, `error_out` must
/// point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_noise_model_backend(
    model: *const SeizaNoiseModel,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let model = unsafe { model.as_ref() }.ok_or("model is required")?;
        CString::new(model.model.backend().to_string())
            .map(CString::into_raw)
            .map_err(|error| error.to_string())
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// The side of the tiles an open model runs.
///
/// # Safety
/// `model` must be null or a live [`SeizaNoiseModel`].
pub unsafe extern "C" fn seiza_noise_model_tile(model: *const SeizaNoiseModel) -> usize {
    unsafe { model.as_ref() }.map_or(0, |model| model.model.tiling().tile)
}

#[unsafe(no_mangle)]
/// Run a model over an image, which is all it needs: a linear stack from any
/// stacker, NaN where nothing covers a pixel. Samples are row-major with
/// channels interleaved, and each channel runs alone. Release the result with
/// [`seiza_denoised_free`].
///
/// # Safety
/// `model` must be a live [`SeizaNoiseModel`] no other thread is using, and
/// `data` must point to `data_length` floats. `progress` may be null, and is
/// passed `context`. `cancel` must be null or a live [`SeizaCancelSignal`]
/// retained until this call returns. When non-null, `error_out` must point to
/// writable storage for one pointer.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn seiza_noise_model_denoise(
    model: *mut SeizaNoiseModel,
    data: *const f32,
    data_length: usize,
    width: usize,
    height: usize,
    channels: usize,
    progress: SeizaModelProgressCallback,
    cancel: *const SeizaCancelSignal,
    context: *mut c_void,
    error_out: *mut *mut c_char,
) -> *mut SeizaDenoised {
    clear_error(error_out);
    ffi_result(error_out, || {
        let model = unsafe { model.as_mut() }.ok_or("model is required")?;
        let data = unsafe { required_f32_slice(data, data_length, "image")? };
        let view =
            ImageView::new(data, width, height, channels).map_err(|error| error.to_string())?;
        // Held as integers so that the closures are `Sync`: the core calls
        // them only on this thread.
        let context = context as usize;
        let signal = cancel as usize;
        let report = |step: Progress| {
            if let Some(callback) = progress {
                let Progress {
                    channel,
                    channels,
                    tile,
                    tiles,
                } = step;
                unsafe { callback(channel, channels, tile, tiles, context as *mut c_void) };
            }
        };
        let stop = || {
            unsafe { (signal as *const SeizaCancelSignal).as_ref() }
                .is_some_and(|signal| signal.cancelled.load(Ordering::Relaxed))
        };
        let options = DenoiseOptions {
            progress: Some(&report),
            cancel: Some(&stop),
        };
        let result = model
            .model
            .denoise(view, &options)
            .map_err(|error| error.to_string())?;
        let report_json = serde_json::to_string(&DenoiseReportResponse::from(&result.report))
            .map_err(|error| error.to_string())?;
        let report_json = CString::new(report_json).map_err(|error| error.to_string())?;
        Ok(Box::into_raw(Box::new(SeizaDenoised {
            result,
            report_json,
        })))
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Write the image with `high` of the grain and `low` of the mottle the model
/// found taken out, split at a Gaussian of `scale_px`. It costs no inference.
///
/// # Safety
/// `denoised` must be a live [`SeizaDenoised`], and `out` must point to
/// `out_length` writable floats, as many as the image given. When non-null,
/// `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_denoised_blend(
    denoised: *const SeizaDenoised,
    high: f32,
    low: f32,
    scale_px: f32,
    out: *mut f32,
    out_length: usize,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let denoised = unsafe { denoised.as_ref() }.ok_or("denoised is required")?;
        let bands = NoiseBands {
            high,
            low,
            scale_px,
        };
        let blended = denoised
            .result
            .blend(&bands)
            .map_err(|error| error.to_string())?;
        if blended.len() != out_length {
            return Err(format!(
                "output has {out_length} floats; expected {}",
                blended.len()
            ));
        }
        let out = unsafe { required_f32_slice_mut(out, out_length, "output")? };
        out.copy_from_slice(&blended);
        Ok(true)
    })
    .unwrap_or(false)
}

#[unsafe(no_mangle)]
/// What the model found in one channel, `f(x) − x` in the image's units, lent
/// until the result is freed. Null when there is no such channel; `length_out`
/// then holds 0.
///
/// # Safety
/// `denoised` must be null or a live [`SeizaDenoised`]. When non-null,
/// `length_out` must point to writable storage for one `size_t`.
pub unsafe extern "C" fn seiza_denoised_residual(
    denoised: *const SeizaDenoised,
    channel: usize,
    length_out: *mut usize,
) -> *const f32 {
    let found = unsafe { denoised.as_ref() }.and_then(|denoised| denoised.result.residual(channel));
    if !length_out.is_null() {
        unsafe { *length_out = found.map_or(0, <[f32]>::len) };
    }
    found.map_or(ptr::null(), <[f32]>::as_ptr)
}

#[unsafe(no_mangle)]
/// What ran, as JSON lent until the result is freed: `model`, `sha256`,
/// `backend`, `precision`, `tile`, `noiseMap`, `holes`, and `channels`, each
/// with its `background` and `noise`. `holes` is null, or how the image's holes
/// were kept from the network, as the model's card asks: `depth`, `margin`,
/// `fillRadius` and `restore`. A host keys cached results on `sha256`,
/// `backend`, `precision`, `holes` and [`seiza_model_output_version`].
///
/// # Safety
/// `denoised` must be null or a live [`SeizaDenoised`].
pub unsafe extern "C" fn seiza_denoised_report_json(
    denoised: *const SeizaDenoised,
) -> *const c_char {
    unsafe { denoised.as_ref() }.map_or(ptr::null(), |denoised| denoised.report_json.as_ptr())
}

#[unsafe(no_mangle)]
/// Release a run's result.
///
/// # Safety
/// `denoised` must be null or a pointer from [`seiza_noise_model_denoise`] that
/// has not already been freed.
pub unsafe extern "C" fn seiza_denoised_free(denoised: *mut SeizaDenoised) {
    if !denoised.is_null() {
        unsafe { drop(Box::from_raw(denoised)) };
    }
}

#[unsafe(no_mangle)]
/// Release an open model.
///
/// # Safety
/// `model` must be null or a pointer from [`seiza_noise_model_open`] that has
/// not already been freed.
pub unsafe extern "C" fn seiza_noise_model_free(model: *mut SeizaNoiseModel) {
    if !model.is_null() {
        unsafe { drop(Box::from_raw(model)) };
    }
}

#[unsafe(no_mangle)]
/// Which ONNX Runtime this process loaded and which backends run here, as JSON
/// the caller frees with `seiza_string_free`: `runtime`, with its `path` and
/// `version`, or null when no library was found; and `backends`, each with
/// `backend`, `unavailable`, the reason it cannot run or null, and `automatic`,
/// whether a model picks it unasked. `options_json` may be null, or hold
/// `library`.
///
/// # Safety
/// `options_json` must be null or a NUL-terminated string. When non-null,
/// `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_model_backends_json(
    options_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let options = ModelOpenRequest::parse(options_json)?.options()?;
        let runtime = runtime_info(&options)
            .ok()
            .map(|info| ModelRuntimeResponse {
                path: info.path,
                version: info.version,
            });
        let backends = Backend::available(&options)
            .into_iter()
            .map(|status| ModelBackendResponse {
                backend: status.backend.to_string(),
                unavailable: status.unavailable,
                automatic: status.automatic,
            })
            .collect();
        owned_json(&ModelBackendsResponse { runtime, backends })
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// The installed models, as a JSON array the caller frees with
/// `seiza_string_free`, each with `name`, `path`, `task`, `noiseMap`, `holes`
/// (as [`seiza_denoised_report_json`] gives it) and `precisions`. `store` may
/// be null for the default store.
///
/// # Safety
/// `store` must be null or a NUL-terminated string. When non-null, `error_out`
/// must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_model_list_json(
    store: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let models = store_at(optional_path(store)?)?
            .list()
            .map_err(|error| error.to_string())?;
        let models: Vec<InstalledModelResponse> = models.iter().map(Into::into).collect();
        owned_json(&models)
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Install the bundle at `source`, once its graphs match its card, replacing
/// any model of its name. Returns the model as [`seiza_model_list_json`]
/// describes it, as JSON the caller frees with `seiza_string_free`.
///
/// # Safety
/// `source` must be a NUL-terminated string and `store` null or one. When
/// non-null, `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_model_install_json(
    source: *const c_char,
    store: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let source = required_path(source, "source")?;
        let bundle = store_at(optional_path(store)?)?
            .install(source)
            .map_err(|error| error.to_string())?;
        owned_json(&InstalledModelResponse::from(&bundle))
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Check a model's graphs against its card. `model` is as for
/// [`seiza_noise_model_open`]; `store` may be null for the default store.
///
/// # Safety
/// `model` and `store` must be null or NUL-terminated strings. When non-null,
/// `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_model_verify(
    model: *const c_char,
    store: *const c_char,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let chosen = if model.is_null() {
            None
        } else {
            Some(required_str(model, "model")?)
        };
        store_at(optional_path(store)?)?
            .resolve(chosen.as_deref())
            .and_then(|bundle| bundle.verify())
            .map_err(|error| error.to_string())?;
        Ok(true)
    })
    .unwrap_or(false)
}

#[unsafe(no_mangle)]
/// Delete the installed model named `name`. `store` may be null for the default
/// store.
///
/// # Safety
/// `name` must be a NUL-terminated string and `store` null or one. When
/// non-null, `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_model_remove(
    name: *const c_char,
    store: *const c_char,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let name = required_str(name, "name")?;
        store_at(optional_path(store)?)?
            .remove(&name)
            .map_err(|error| error.to_string())?;
        Ok(true)
    })
    .unwrap_or(false)
}

#[unsafe(no_mangle)]
/// The version of `seiza-model` whose output this build reproduces, a static
/// string a host keys cached results on.
pub extern "C" fn seiza_model_output_version() -> *const c_char {
    static VERSION: std::sync::OnceLock<CString> = std::sync::OnceLock::new();
    VERSION
        .get_or_init(|| CString::new(seiza_model::OUTPUT_VERSION).unwrap_or_default())
        .as_ptr()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::ffi::CStr;
    use std::path::Path;

    /// Read and free a string the C ABI handed over.
    fn taken(value: *mut c_char) -> String {
        let text = unsafe { CStr::from_ptr(value) }
            .to_str()
            .unwrap()
            .to_string();
        unsafe { crate::seiza_string_free(value) };
        text
    }

    fn fixtures() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../seiza-model/tests/fixtures")
    }

    fn text(path: &Path) -> CString {
        CString::new(path.to_str().unwrap()).unwrap()
    }

    fn image() -> Vec<f32> {
        let bytes = std::fs::read(fixtures().join("image.f32")).unwrap();
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect()
    }

    #[test]
    fn the_output_version_is_the_crate_s_own() {
        let version = unsafe { CStr::from_ptr(seiza_model_output_version()) };
        assert_eq!(version.to_str().unwrap(), seiza_model::OUTPUT_VERSION);
    }

    #[test]
    fn a_model_installs_verifies_lists_and_goes() {
        let temp = tempfile::tempdir().unwrap();
        let store = text(&temp.path().join("store"));
        let source = text(&fixtures().join("tiny"));
        let name = CString::new("noise-tiny").unwrap();
        let mut error = ptr::null_mut();
        unsafe {
            let installed = seiza_model_install_json(source.as_ptr(), store.as_ptr(), &mut error);
            assert!(error.is_null());
            let installed: Value = serde_json::from_str(&taken(installed)).unwrap();
            assert_eq!(installed["name"], "noise-tiny");
            assert_eq!(installed["noiseMap"], "constant");
            assert!(installed["holes"].is_null(), "the tiny model's card has no guard");

            let listed = seiza_model_list_json(store.as_ptr(), &mut error);
            let listed: Value = serde_json::from_str(&taken(listed)).unwrap();
            assert_eq!(listed.as_array().unwrap().len(), 1);
            assert_eq!(listed[0]["precisions"], serde_json::json!(["f32", "f16"]));

            assert!(seiza_model_verify(
                name.as_ptr(),
                store.as_ptr(),
                &mut error
            ));
            assert!(seiza_model_verify(ptr::null(), store.as_ptr(), &mut error));
            assert!(seiza_model_remove(
                name.as_ptr(),
                store.as_ptr(),
                &mut error
            ));
            assert!(!seiza_model_remove(
                name.as_ptr(),
                store.as_ptr(),
                &mut error
            ));
            assert!(!error.is_null(), "a second removal is an error");
            assert!(taken(error).contains("noise-tiny"));
        }
    }

    #[test]
    fn invalid_tiling_cannot_replace_an_installed_model() {
        let temp = tempfile::tempdir().unwrap();
        let store = text(&temp.path().join("store"));
        let source = temp.path().join("source");
        std::fs::create_dir(&source).unwrap();
        for entry in std::fs::read_dir(fixtures().join("tiny")).unwrap() {
            let entry = entry.unwrap();
            std::fs::copy(entry.path(), source.join(entry.file_name())).unwrap();
        }
        let source_text = text(&source);
        let path = source.join("card.json");
        let good: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let mut error = ptr::null_mut();
        let installed =
            unsafe { seiza_model_install_json(source_text.as_ptr(), store.as_ptr(), &mut error) };
        assert!(error.is_null());
        taken(installed);
        for (field, value) in [
            ("margin", usize::MAX / 2 + 1),
            ("ramp", usize::MAX),
            ("tile", 0),
            ("tile", 63),
            ("tile", 16),
            ("multiple", 0),
        ] {
            let mut bad = good.clone();
            bad["tiling"][field] = serde_json::json!(value);
            std::fs::write(&path, serde_json::to_vec(&bad).unwrap()).unwrap();
            let installed = unsafe {
                seiza_model_install_json(source_text.as_ptr(), store.as_ptr(), &mut error)
            };
            assert!(installed.is_null(), "{field}={value}");
            assert!(!error.is_null());
            assert!(taken(error).contains("tiling"));
            error = ptr::null_mut();
            let name = CString::new("noise-tiny").unwrap();
            assert!(unsafe { seiza_model_verify(name.as_ptr(), store.as_ptr(), &mut error) });
            assert!(error.is_null());
        }
    }

    #[test]
    fn bad_options_are_refused_before_anything_loads() {
        let unknown = CString::new(r#"{"backend": "cpu", "speed": 9}"#).unwrap();
        let mut error = ptr::null_mut();
        let source = text(&fixtures().join("tiny"));
        let model =
            unsafe { seiza_noise_model_open(source.as_ptr(), unknown.as_ptr(), &mut error) };
        assert!(model.is_null());
        assert!(taken(error).contains("speed"));

        let backend = CString::new(r#"{"backend": "metal"}"#).unwrap();
        let mut error = ptr::null_mut();
        let model =
            unsafe { seiza_noise_model_open(source.as_ptr(), backend.as_ptr(), &mut error) };
        assert!(model.is_null());
        assert!(taken(error).contains("metal"));
    }

    #[test]
    fn the_backends_call_answers_with_or_without_a_library() {
        let mut error = ptr::null_mut();
        let answer = unsafe { seiza_model_backends_json(ptr::null(), &mut error) };
        assert!(error.is_null());
        let answer: Value = serde_json::from_str(&taken(answer)).unwrap();
        let backends = answer["backends"].as_array().unwrap();
        assert!(backends.iter().any(|status| status["backend"] == "cpu"));
    }

    unsafe extern "C" fn count_tile(_: usize, _: usize, _: usize, _: usize, context: *mut c_void) {
        unsafe { *(context as *mut usize) += 1 };
    }

    /// Needs ONNX Runtime, so it runs only where `SEIZA_ONNXRUNTIME` names it.
    #[test]
    fn a_tiny_model_runs_and_blends_from_c() {
        if std::env::var_os("SEIZA_ONNXRUNTIME").is_none() {
            eprintln!("skipped: SEIZA_ONNXRUNTIME names no ONNX Runtime library");
            return;
        }
        let (width, height) = (300, 230);
        let pixels = image();
        let source = text(&fixtures().join("tiny"));
        let options = CString::new(r#"{"backend": "cpu"}"#).unwrap();
        let mut error = ptr::null_mut();
        let mut tiles = 0_usize;
        unsafe {
            let model = seiza_noise_model_open(source.as_ptr(), options.as_ptr(), &mut error);
            assert!(!model.is_null());
            assert_eq!(taken(seiza_noise_model_backend(model, &mut error)), "cpu");
            assert_eq!(seiza_noise_model_tile(model), 64);
            let denoised = seiza_noise_model_denoise(
                model,
                pixels.as_ptr(),
                pixels.len(),
                width,
                height,
                1,
                Some(count_tile),
                ptr::null(),
                (&raw mut tiles).cast(),
                &mut error,
            );
            assert!(!denoised.is_null());
            assert!(tiles > 1);

            let report = CStr::from_ptr(seiza_denoised_report_json(denoised));
            let report: Value = serde_json::from_str(report.to_str().unwrap()).unwrap();
            assert_eq!(report["model"], "noise-tiny");
            assert_eq!(report["precision"], "f32");
            assert_eq!(report["noiseMap"], "constant");
            assert!(report["holes"].is_null());

            let mut length = 0;
            let residual = seiza_denoised_residual(denoised, 0, &mut length);
            assert_eq!(length, width * height);
            let residual = std::slice::from_raw_parts(residual, length);
            assert!(seiza_denoised_residual(denoised, 1, &mut length).is_null());
            assert_eq!(length, 0);

            let mut out = vec![0.0_f32; pixels.len()];
            let blended = seiza_denoised_blend(
                denoised,
                1.0,
                1.0,
                5.0,
                out.as_mut_ptr(),
                out.len(),
                &mut error,
            );
            assert!(blended);
            let index = 150 * width + 150;
            assert_eq!(out[index], pixels[index] + residual[index]);
            assert!(out[0].is_nan());
            assert!(!seiza_denoised_blend(
                denoised,
                2.0,
                1.0,
                5.0,
                out.as_mut_ptr(),
                out.len(),
                &mut error
            ));
            assert!(taken(error).contains("amounts"));

            seiza_denoised_free(denoised);
            seiza_noise_model_free(model);
        }
    }
}
