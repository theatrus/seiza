use image::DynamicImage;
use seiza::blind::{BlindIndex, BlindParams, solve_blind};
use seiza::catalog::{StarCatalog, tiles::TileCatalog};
use seiza::downloads::{CachePolicy, CatalogManager, CatalogSet, Dataset, DownloadEvent};
use seiza::minor_bodies::{MinorBodyCatalog, MinorBodyKind};
use seiza::objects::{
    GeometryData, GeometryQuality, GeometryRole, ObjectCatalog, ObjectGeometry, ObjectKind,
    ObjectQuery, SkyRegion,
};
use seiza::wcs::Wcs;
use seiza::{DetectBackend, DetectConfig, detect_stars, detect_stars_luma_f32};
use seiza_background::{BackgroundConfig, BackgroundFit, CorrectionMode, fit_background_masked};
use seiza_deconvolution::{DeconvolutionConfig, deconvolve};
use seiza_fits::{FitsImage, HeaderValue, RgbImage16, Statistics, StretchParams};
use seiza_stacking::{
    CalibrationMasters, FitsFrame, FrameDiagnostics, FrameDisposition, LinearImage, LiveStacker,
    StackOptions, StackSnapshot as RustStackSnapshot, paths_refer_to_same_file, write_fits_f32,
};
use seiza_stretch::{StretchConfig, StretchStack};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Instant, SystemTime};

static VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StretchConfigRequest {
    Single(StretchConfig),
    Stack(StretchStack),
}

impl StretchConfigRequest {
    fn into_stack(self) -> StretchStack {
        match self {
            Self::Single(config) => StretchStack::single(config),
            Self::Stack(stack) => stack,
        }
    }
}

#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(default)]
struct BackgroundRenderRequest {
    mode: CorrectionMode,
    config: BackgroundConfig,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
struct DeconvolutionRenderRequest {
    psf_fwhm_pixels: f32,
    #[serde(default = "default_deconvolution_iterations")]
    iterations: usize,
    #[serde(default = "default_deconvolution_amount")]
    amount: f32,
    #[serde(default = "default_deconvolution_noise_fraction")]
    noise_fraction: f32,
    #[serde(default = "default_deconvolution_max_correction")]
    max_correction: f32,
}

const fn default_deconvolution_iterations() -> usize {
    4
}

const fn default_deconvolution_amount() -> f32 {
    0.35
}

const fn default_deconvolution_noise_fraction() -> f32 {
    0.001
}

const fn default_deconvolution_max_correction() -> f32 {
    2.0
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct InteractivePreviewCacheKey {
    path: PathBuf,
    file_size: u64,
    modified: Option<SystemTime>,
    max_dimension: u32,
    background: Option<String>,
}

struct PreparedFitsRender {
    source_width: usize,
    source_height: usize,
    planes: usize,
    color_kind: &'static str,
    render_width: usize,
    render_height: usize,
    channels: usize,
    data: Vec<f32>,
    statistics: Value,
    input_histogram: Value,
    background_metadata: Option<Value>,
    headers: Map<String, Value>,
    interactive_preview: bool,
}

struct PreparedStretchInput<'a> {
    data: Cow<'a, [f32]>,
    input_histogram: Value,
    deconvolution_metadata: Option<Value>,
}

type InteractivePreviewCache =
    Mutex<VecDeque<(InteractivePreviewCacheKey, Arc<PreparedFitsRender>)>>;

static INTERACTIVE_PREVIEW_CACHE: OnceLock<InteractivePreviewCache> = OnceLock::new();
const INTERACTIVE_PREVIEW_CACHE_CAPACITY: usize = 2;

#[derive(Debug, Deserialize)]
struct ProcessedRenderRequest {
    stretch: StretchConfigRequest,
    #[serde(default)]
    background: Option<BackgroundRenderRequest>,
    #[serde(default)]
    deconvolution: Option<DeconvolutionRenderRequest>,
    #[serde(default)]
    interactive_preview: bool,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ImageRenderConfigRequest {
    Processed(ProcessedRenderRequest),
    Stretch(StretchConfigRequest),
}

impl ImageRenderConfigRequest {
    fn into_parts(
        self,
    ) -> (
        StretchStack,
        Option<BackgroundRenderRequest>,
        Option<DeconvolutionRenderRequest>,
        bool,
    ) {
        match self {
            Self::Processed(request) => (
                request.stretch.into_stack(),
                request.background,
                request.deconvolution,
                request.interactive_preview,
            ),
            Self::Stretch(request) => (request.into_stack(), None, None, false),
        }
    }
}

pub type SeizaCatalogSetupProgressCallback =
    Option<unsafe extern "C" fn(*const c_char, *mut c_void)>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
enum CatalogSetupPreset {
    StandardBlind = 0,
    DeepestBlind = 1,
    All = 2,
}

impl CatalogSetupPreset {
    fn from_raw(value: u32) -> Result<Self, String> {
        match value {
            0 => Ok(Self::StandardBlind),
            1 => Ok(Self::DeepestBlind),
            2 => Ok(Self::All),
            _ => Err(format!("unsupported catalog setup preset: {value}")),
        }
    }

    fn datasets(self) -> &'static [Dataset] {
        match self {
            Self::StandardBlind => &[
                Dataset::Objects,
                Dataset::MinorBodies,
                Dataset::Transients,
                Dataset::StarsDeepGaia17,
                Dataset::BlindGaia16,
            ],
            Self::DeepestBlind => &[
                Dataset::Objects,
                Dataset::MinorBodies,
                Dataset::Transients,
                Dataset::StarsDeepGaia20,
                Dataset::BlindGaia16,
            ],
            Self::All => &[
                Dataset::Objects,
                Dataset::MinorBodies,
                Dataset::Transients,
                Dataset::StarsLiteTycho2,
                Dataset::StarsLiteTycho2Identifiers,
                Dataset::StarsGaia,
                Dataset::StarsDeepGaia17,
                Dataset::StarsDeepGaia20,
                Dataset::BlindGaia16,
            ],
        }
    }

    fn selection(self) -> Result<CatalogSet, String> {
        CatalogSet::from_names(
            self.datasets()
                .iter()
                .map(|dataset| dataset.file_name().to_string()),
        )
        .map_err(|error| error.to_string())
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CatalogComponentStatus {
    available: bool,
    path: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CatalogStatusResponse {
    directory: String,
    ready_for_solving: bool,
    ready_for_overlays: bool,
    star_catalog: CatalogComponentStatus,
    blind_index: CatalogComponentStatus,
    objects: CatalogComponentStatus,
    transients: CatalogComponentStatus,
    minor_bodies: CatalogComponentStatus,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CatalogSetupProgressResponse {
    phase: &'static str,
    message: String,
    file_name: Option<String>,
    files_completed: usize,
    files_total: usize,
    bytes_completed: Option<u64>,
    bytes_total: Option<u64>,
    written_bytes: Option<u64>,
}

#[derive(Clone, Copy)]
struct CatalogSetupReporter {
    callback: SeizaCatalogSetupProgressCallback,
    context: usize,
    files_total: usize,
}

impl CatalogSetupReporter {
    fn report(&self, event: CatalogSetupProgressResponse) {
        let Some(callback) = self.callback else {
            return;
        };
        let Ok(json) = serde_json::to_string(&event) else {
            return;
        };
        let Ok(json) = CString::new(json) else { return };
        unsafe { callback(json.as_ptr(), self.context as *mut c_void) };
    }

    fn simple(&self, phase: &'static str, message: impl Into<String>) {
        self.report(CatalogSetupProgressResponse {
            phase,
            message: message.into(),
            file_name: None,
            files_completed: 0,
            files_total: self.files_total,
            bytes_completed: None,
            bytes_total: None,
            written_bytes: None,
        });
    }

    fn download_event(&self, event: DownloadEvent, files_completed: usize) {
        match event {
            DownloadEvent::FetchingManifest { .. } => {
                self.simple("manifest", "Checking the Seiza catalog manifest…")
            }
            DownloadEvent::UsingCachedManifest { version, stale } => self.simple(
                "manifest",
                if stale {
                    format!("Using cached catalog manifest {version} while offline")
                } else {
                    format!("Using catalog manifest {version}")
                },
            ),
            DownloadEvent::CacheHit { name, .. } => self.report(CatalogSetupProgressResponse {
                phase: "preparing",
                message: format!("Found {name} in the download cache"),
                file_name: Some(name),
                files_completed,
                files_total: self.files_total,
                bytes_completed: None,
                bytes_total: None,
                written_bytes: None,
            }),
            DownloadEvent::DownloadStarted { name, bytes } => {
                self.report(CatalogSetupProgressResponse {
                    phase: "downloading",
                    message: format!("Downloading {name}"),
                    file_name: Some(name),
                    files_completed,
                    files_total: self.files_total,
                    bytes_completed: Some(0),
                    bytes_total: Some(bytes),
                    written_bytes: Some(0),
                })
            }
            DownloadEvent::DownloadProgress {
                name,
                downloaded,
                total,
                written,
            } => self.report(CatalogSetupProgressResponse {
                phase: "downloading",
                message: format!("Downloading {name}"),
                file_name: Some(name),
                files_completed,
                files_total: self.files_total,
                bytes_completed: Some(downloaded),
                bytes_total: Some(total),
                written_bytes: Some(written),
            }),
            DownloadEvent::DownloadComplete { name, .. } => {
                self.report(CatalogSetupProgressResponse {
                    phase: "preparing",
                    message: format!("Downloaded {name}"),
                    file_name: Some(name),
                    files_completed,
                    files_total: self.files_total,
                    bytes_completed: None,
                    bytes_total: None,
                    written_bytes: None,
                })
            }
            DownloadEvent::Verifying { name } => self.report(CatalogSetupProgressResponse {
                phase: "verifying",
                message: format!("Verifying {name}"),
                file_name: Some(name),
                files_completed,
                files_total: self.files_total,
                bytes_completed: None,
                bytes_total: None,
                written_bytes: None,
            }),
            DownloadEvent::Installing { name, .. } => self.report(CatalogSetupProgressResponse {
                phase: "installing",
                message: format!("Installing {name}"),
                file_name: Some(name),
                files_completed,
                files_total: self.files_total,
                bytes_completed: None,
                bytes_total: None,
                written_bytes: None,
            }),
            DownloadEvent::InstallComplete { name, .. } => {
                self.report(CatalogSetupProgressResponse {
                    phase: "installing",
                    message: format!("Installed {name}"),
                    file_name: Some(name),
                    files_completed,
                    files_total: self.files_total,
                    bytes_completed: None,
                    bytes_total: None,
                    written_bytes: None,
                })
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
enum RgbStretchMode {
    Auto = 0,
    LinkedAuto = 1,
    Linear = 2,
}

impl RgbStretchMode {
    fn from_raw(value: u32) -> Result<Self, String> {
        match value {
            0 => Ok(Self::Auto),
            1 => Ok(Self::LinkedAuto),
            2 => Ok(Self::Linear),
            _ => Err(format!("unsupported RGB stretch mode: {value}")),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::LinkedAuto => "linked-auto",
            Self::Linear => "linear",
        }
    }
}

/// An opaque, owned rendered image. C sees only a pointer; release it with
/// [`seiza_rendered_image_free`]. Not `repr(C)`: it is only ever handed to C as
/// an opaque pointer, so its layout is private (and cbindgen forward-declares
/// it).
pub struct SeizaRenderedImage {
    width: u32,
    height: u32,
    /// Canonical pixel buffer, RGBA8 (macOS / CoreGraphics byte order).
    rgba: Vec<u8>,
    /// BGRA8 view (Direct2D / WinUI byte order) computed from `rgba` on first
    /// request and cached. Only one byte order is used per consumer, so the
    /// copy is paid lazily and at most once.
    bgra: OnceLock<Vec<u8>>,
    metadata_json: CString,
}

/// An opaque, owned 16-bit rendered image. Its RGBA samples are native-endian
/// `u16` values suitable for a high-bit-depth image encoder. C sees only a
/// pointer; release it with [`seiza_rendered_image16_free`].
pub struct SeizaRenderedImage16 {
    width: u32,
    height: u32,
    rgba: Vec<u16>,
    metadata_json: CString,
}

/// An opaque fitted background model. Release it with
/// [`seiza_background_model_free`]. Its diagnostics string is borrowed and
/// remains valid until the model is freed.
pub struct SeizaBackgroundModel {
    fit: BackgroundFit,
    diagnostics_json: CString,
}

/// An opaque incremental stacker. Release it with
/// [`seiza_live_stacker_free`], or consume it with
/// [`seiza_live_stacker_finish`].
pub struct SeizaLiveStacker {
    stacker: LiveStacker,
    input_paths: Vec<PathBuf>,
}

/// An immutable owned stack result. Its image and count pointers are borrowed
/// until [`seiza_stack_snapshot_free`] is called.
pub struct SeizaStackSnapshot {
    snapshot: RustStackSnapshot,
    reference_headers: Vec<(String, HeaderValue)>,
    input_paths: Vec<PathBuf>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StackDispositionResponse {
    source: Option<String>,
    accepted: bool,
    reason: Option<String>,
    diagnostics: Option<StackDiagnosticsResponse>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StackDiagnosticsResponse {
    matched_stars: usize,
    registration_rms_pixels: f64,
    registration_drift_pixels: f64,
    scale: f64,
    rotation_degrees: f64,
    translation_x: f64,
    translation_y: f64,
    normalization_mean_gain: f32,
    normalization_mean_offset: f32,
    overlap_fraction: f32,
    integrated_fraction: f32,
    accepted_samples: usize,
    rejected_samples: usize,
}

/// Additive background subtraction mode for
/// [`seiza_background_model_correct_in_place`].
pub const SEIZA_BACKGROUND_CORRECTION_SUBTRACT: u32 = 0;
/// Multiplicative background division mode for
/// [`seiza_background_model_correct_in_place`].
pub const SEIZA_BACKGROUND_CORRECTION_DIVIDE: u32 = 1;

fn background_correction_mode(value: u32) -> Result<CorrectionMode, String> {
    match value {
        SEIZA_BACKGROUND_CORRECTION_SUBTRACT => Ok(CorrectionMode::Subtract),
        SEIZA_BACKGROUND_CORRECTION_DIVIDE => Ok(CorrectionMode::Divide),
        _ => Err(format!("unsupported background correction mode: {value}")),
    }
}

impl SeizaRenderedImage {
    /// The BGRA8 view, derived from the canonical RGBA on first use.
    fn bgra(&self) -> &[u8] {
        self.bgra.get_or_init(|| rgba_to_bgra(self.rgba.clone()))
    }
}

/// Swap the red and blue channels in place, converting RGBA8 to BGRA8.
fn rgba_to_bgra(mut pixels: Vec<u8>) -> Vec<u8> {
    for pixel in pixels.chunks_exact_mut(4) {
        pixel.swap(0, 2);
    }
    pixels
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SolveResponse {
    center_ra_degrees: f64,
    center_dec_degrees: f64,
    scale_arcsec_per_pixel: f64,
    matched_stars: usize,
    rms_arcsec: f64,
    detected_stars: usize,
    elapsed_milliseconds: u128,
    detected_star_positions: Vec<ImagePointResponse>,
    catalog_star_positions: Vec<CatalogStarPointResponse>,
    object_positions: Vec<ObjectPointResponse>,
    object_catalog_error: Option<String>,
    capture_time: Option<String>,
    overlay_availability: BTreeMap<String, bool>,
    overlay_unavailable_reasons: BTreeMap<String, String>,
    overlay_counts: BTreeMap<String, usize>,
    wcs: WcsResponse,
}

#[derive(Serialize)]
struct ImagePointResponse {
    x: f64,
    y: f64,
}

#[derive(Serialize)]
struct CatalogStarPointResponse {
    x: f64,
    y: f64,
    magnitude: f32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ObjectPointResponse {
    stable_id: Option<String>,
    name: String,
    common_name: String,
    kind: String,
    source: String,
    catalog_source: Option<String>,
    x: f64,
    y: f64,
    semi_major_pixels: f64,
    semi_minor_pixels: f64,
    angle_degrees: Option<f64>,
    prominence: Option<f64>,
    ra_degrees: Option<f64>,
    dec_degrees: Option<f64>,
    discovered: Option<String>,
    near_capture: Option<bool>,
    distance_au: Option<f64>,
    motion_arcsec_per_hour: Option<f64>,
    direction_position_angle_degrees: Option<f64>,
    direction_image_angle_degrees: Option<f64>,
    outlines: Vec<ObjectOutlineResponse>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ObjectOutlineResponse {
    geometry_id: String,
    source_record_id: String,
    role: String,
    quality: String,
    level: Option<String>,
    contours: Vec<ObjectContourResponse>,
}

#[derive(Debug, Serialize)]
struct ObjectContourResponse {
    closed: bool,
    points: Vec<[f64; 2]>,
}

#[derive(Serialize)]
struct WcsResponse {
    crval: [f64; 2],
    crpix: [f64; 2],
    cd: [[f64; 2]; 2],
    sip: Option<SipResponse>,
}

#[derive(Serialize)]
struct SipResponse {
    order: u8,
    a: Vec<f64>,
    b: Vec<f64>,
    ap: Vec<f64>,
    bp: Vec<f64>,
}

#[unsafe(no_mangle)]
pub extern "C" fn seiza_core_version() -> *const c_char {
    VERSION.as_ptr().cast()
}

#[unsafe(no_mangle)]
/// Applies damped Richardson-Lucy deconvolution to interleaved linear `float`
/// samples in place. `channels` must be one or three and `data_length` must
/// equal `width * height * channels`. RGB samples are pixel-interleaved.
/// The operation is synchronous, retains no pointer after returning, and leaves
/// the input unchanged when validation or restoration fails.
///
/// # Safety
/// `data` must point to `data_length` writable floats. When non-null,
/// `error_out` must point to writable storage for one pointer.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn seiza_deconvolve_in_place(
    data: *mut f32,
    data_length: usize,
    width: usize,
    height: usize,
    channels: usize,
    psf_fwhm_pixels: f32,
    iterations: usize,
    amount: f32,
    noise_fraction: f32,
    max_correction: f32,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let data = unsafe { required_f32_slice_mut(data, data_length, "deconvolution input")? };
        let config = DeconvolutionConfig {
            psf_fwhm_pixels,
            iterations,
            amount,
            noise_fraction,
            max_correction,
        };
        let restored = deconvolve(data, width, height, channels, &config)
            .map_err(|error| error.to_string())?;
        data.copy_from_slice(&restored.data);
        Ok(())
    })
    .is_some()
}

#[unsafe(no_mangle)]
/// Fits a compact background model to interleaved linear `float` samples.
///
/// `channels` must be one or three and `data_length` must equal
/// `width * height * channels`. RGB samples are pixel-interleaved. Pass null
/// `mask` with zero `mask_length` for automatic fitting, or `width * height`
/// bytes where `1` excludes a pixel. The fitted model owns its compact data and
/// does not borrow either input buffer after this call returns.
/// Pass null or empty `config_json` for `BackgroundConfig::default()`; otherwise
/// provide a serialized `seiza-background` `BackgroundConfig`.
///
/// # Safety
/// `data` must point to `data_length` readable floats. A non-null `mask` must
/// point to `mask_length` readable bytes containing only zero or one. A
/// non-null `config_json` must be NUL-terminated. When non-null, `error_out`
/// must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_background_fit(
    data: *const f32,
    data_length: usize,
    width: usize,
    height: usize,
    channels: usize,
    mask: *const u8,
    mask_length: usize,
    config_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut SeizaBackgroundModel {
    clear_error(error_out);
    ffi_result(error_out, || {
        if !matches!(channels, 1 | 3) {
            return Err("background fitting requires one or three channels".into());
        }
        let pixels = width
            .checked_mul(height)
            .ok_or_else(|| "background image dimensions overflow".to_string())?;
        let expected = pixels
            .checked_mul(channels)
            .ok_or_else(|| "background image dimensions overflow".to_string())?;
        if data_length != expected {
            return Err(format!(
                "background input has {data_length} floats; expected {expected}"
            ));
        }
        if mask.is_null() {
            if mask_length != 0 {
                return Err("background mask is null but mask_length is non-zero".into());
            }
        } else if mask_length != pixels {
            return Err(format!(
                "background mask has {mask_length} bytes; expected {pixels}"
            ));
        }
        let data = unsafe { required_f32_slice(data, data_length, "background input")? };
        let mask = unsafe { optional_mask(mask, mask_length)? };
        let config = background_config(config_json)?;
        let fit = fit_background_masked(data, width, height, channels, mask.as_deref(), &config)
            .map_err(|error| error.to_string())?;
        let diagnostics_json =
            CString::new(serde_json::to_string(&fit).map_err(|error| error.to_string())?)
                .map_err(|_| "background diagnostics contain a NUL byte".to_string())?;
        Ok(SeizaBackgroundModel {
            fit,
            diagnostics_json,
        })
    })
    .map_or(ptr::null_mut(), |model| Box::into_raw(Box::new(model)))
}

#[unsafe(no_mangle)]
/// # Safety
/// `model` must be null or a live pointer returned by [`seiza_background_fit`].
pub unsafe extern "C" fn seiza_background_model_width(model: *const SeizaBackgroundModel) -> usize {
    unsafe { model.as_ref().map_or(0, |model| model.fit.width) }
}

#[unsafe(no_mangle)]
/// # Safety
/// `model` must be null or a live pointer returned by [`seiza_background_fit`].
pub unsafe extern "C" fn seiza_background_model_height(
    model: *const SeizaBackgroundModel,
) -> usize {
    unsafe { model.as_ref().map_or(0, |model| model.fit.height) }
}

#[unsafe(no_mangle)]
/// # Safety
/// `model` must be null or a live pointer returned by [`seiza_background_fit`].
pub unsafe extern "C" fn seiza_background_model_channels(
    model: *const SeizaBackgroundModel,
) -> usize {
    unsafe { model.as_ref().map_or(0, |model| model.fit.channels) }
}

#[unsafe(no_mangle)]
/// Returns the number of floats required by render and correction buffers.
///
/// # Safety
/// `model` must be null or a live pointer returned by [`seiza_background_fit`].
pub unsafe extern "C" fn seiza_background_model_data_length(
    model: *const SeizaBackgroundModel,
) -> usize {
    unsafe {
        model.as_ref().map_or(0, |model| {
            model.fit.width * model.fit.height * model.fit.channels
        })
    }
}

#[unsafe(no_mangle)]
/// Returns borrowed fitted coefficients, references, samples, and diagnostics
/// as JSON. The string remains valid until the model is freed.
///
/// # Safety
/// `model` must be null or a live pointer returned by [`seiza_background_fit`].
pub unsafe extern "C" fn seiza_background_model_diagnostics_json(
    model: *const SeizaBackgroundModel,
) -> *const c_char {
    unsafe {
        model
            .as_ref()
            .map_or(ptr::null(), |model| model.diagnostics_json.as_ptr())
    }
}

#[unsafe(no_mangle)]
/// Renders a fitted background into a caller-owned interleaved float buffer.
///
/// # Safety
/// `model` must be a live pointer returned by [`seiza_background_fit`].
/// `output` must point to `output_length` writable floats. When non-null,
/// `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_background_model_render(
    model: *const SeizaBackgroundModel,
    output: *mut f32,
    output_length: usize,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let model = unsafe { required_background_model(model)? };
        let output = unsafe { required_f32_slice_mut(output, output_length, "background output")? };
        model
            .fit
            .render_model_into(output)
            .map_err(|error| error.to_string())
    })
    .is_some()
}

#[unsafe(no_mangle)]
/// Corrects an interleaved linear float buffer in place. Use
/// `SEIZA_BACKGROUND_CORRECTION_SUBTRACT` for additive subtraction or
/// `SEIZA_BACKGROUND_CORRECTION_DIVIDE` for multiplicative division.
///
/// # Safety
/// `model` must be a live pointer returned by [`seiza_background_fit`]. `data`
/// must point to `data_length` writable floats. When non-null, `error_out` must
/// point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_background_model_correct_in_place(
    model: *const SeizaBackgroundModel,
    data: *mut f32,
    data_length: usize,
    mode: u32,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let model = unsafe { required_background_model(model)? };
        let data = unsafe { required_f32_slice_mut(data, data_length, "background input")? };
        let mode = background_correction_mode(mode)?;
        model
            .fit
            .correct_in_place(data, mode)
            .map_err(|error| error.to_string())
    })
    .is_some()
}

#[unsafe(no_mangle)]
/// # Safety
/// `model` must be null or a pointer returned by [`seiza_background_fit`] that
/// has not already been freed.
pub unsafe extern "C" fn seiza_background_model_free(model: *mut SeizaBackgroundModel) {
    if !model.is_null() {
        unsafe { drop(Box::from_raw(model)) };
    }
}

#[unsafe(no_mangle)]
/// Creates an incremental stack from a copied linear mono or interleaved RGB
/// reference frame. Array frames are assumed to be calibrated and debayered.
/// Pass null or empty `options_json` for `StackOptions::default()`.
///
/// # Safety
/// `reference` must point to `reference_length` readable floats. A non-null
/// `options_json` must be NUL-terminated. When non-null, `error_out` must point
/// to writable storage for one pointer.
pub unsafe extern "C" fn seiza_live_stacker_create(
    reference: *const f32,
    reference_length: usize,
    width: usize,
    height: usize,
    channels: usize,
    options_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut SeizaLiveStacker {
    clear_error(error_out);
    ffi_result(error_out, || {
        let reference = unsafe {
            linear_image_from_ffi(
                reference,
                reference_length,
                width,
                height,
                channels,
                "stack reference",
            )?
        };
        let options = stack_options(options_json)?;
        let stacker =
            LiveStacker::from_linear(reference, options).map_err(|error| error.to_string())?;
        Ok(SeizaLiveStacker {
            stacker,
            input_paths: Vec::new(),
        })
    })
    .map_or(ptr::null_mut(), |stacker| Box::into_raw(Box::new(stacker)))
}

#[unsafe(no_mangle)]
/// Opens a FITS reference and optional integrated bias, dark, and flat masters.
/// A positive `dark_exposure_seconds` overrides the dark FITS metadata; zero
/// uses the metadata. Pass null or empty `options_json` for defaults. All files
/// are fully read during this call and are not kept open afterward.
///
/// # Safety
/// `reference_path` must be a valid NUL-terminated string. Optional paths and
/// `options_json` may be null; when non-null they must be NUL-terminated. When
/// non-null, `error_out` must point to writable storage for one pointer.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn seiza_live_stacker_open_fits(
    reference_path: *const c_char,
    bias_path: *const c_char,
    dark_path: *const c_char,
    flat_path: *const c_char,
    dark_exposure_seconds: f64,
    options_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut SeizaLiveStacker {
    clear_error(error_out);
    ffi_result(error_out, || {
        let reference_path = required_path(reference_path, "stack reference path")?;
        let bias_path = optional_path(bias_path)?;
        let dark_path = optional_path(dark_path)?;
        let flat_path = optional_path(flat_path)?;
        let input_paths = [
            Some(reference_path.clone()),
            bias_path.clone(),
            dark_path.clone(),
            flat_path.clone(),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        validate_distinct_stack_paths(&input_paths)?;
        if dark_path.is_none() && dark_exposure_seconds != 0.0 {
            return Err("a master-dark exposure override requires a dark path".into());
        }
        let dark_exposure_seconds =
            optional_positive_seconds(dark_exposure_seconds, "master-dark exposure override")?;
        let options = stack_options(options_json)?;
        let calibration = CalibrationMasters::from_fits_paths(
            bias_path.as_deref(),
            dark_path.as_deref(),
            flat_path.as_deref(),
            dark_exposure_seconds,
        )
        .map_err(|error| error.to_string())?;
        let reference = FitsFrame::open(&reference_path).map_err(|error| error.to_string())?;
        let stacker =
            LiveStacker::new(reference, calibration, options).map_err(|error| error.to_string())?;
        Ok(SeizaLiveStacker {
            stacker,
            input_paths,
        })
    })
    .map_or(ptr::null_mut(), |stacker| Box::into_raw(Box::new(stacker)))
}

#[unsafe(no_mangle)]
/// Registers and offers one copied, calibrated linear frame to the stack.
/// Returns owned disposition JSON for both accepted and rejected frames; free
/// it with [`seiza_string_free`]. A rejected frame is a successful call and is
/// represented by `accepted: false` rather than `error_out`.
///
/// # Safety
/// `stacker` must be a live pointer returned by a `seiza_live_stacker_*`
/// constructor. `frame` must point to `frame_length` readable floats. When
/// non-null, `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_live_stacker_push_linear_json(
    stacker: *mut SeizaLiveStacker,
    frame: *const f32,
    frame_length: usize,
    width: usize,
    height: usize,
    channels: usize,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let frame = unsafe {
            linear_image_from_ffi(frame, frame_length, width, height, channels, "stack frame")?
        };
        let stacker = unsafe { required_live_stacker_mut(stacker)? };
        let disposition = stacker
            .stacker
            .push_linear(frame)
            .map_err(|error| error.to_string())?;
        owned_json(&stack_disposition_response(None, disposition))
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Opens, calibrates, registers, and offers one FITS frame to the stack. The
/// returned disposition JSON is owned and must be freed with
/// [`seiza_string_free`]. Each source path may be offered only once.
///
/// # Safety
/// `stacker` must be a live pointer returned by a `seiza_live_stacker_*`
/// constructor. `path` must be a valid NUL-terminated string. When non-null,
/// `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_live_stacker_push_fits_json(
    stacker: *mut SeizaLiveStacker,
    path: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let path = required_path(path, "stack frame path")?;
        let stacker = unsafe { required_live_stacker_mut(stacker)? };
        if stacker
            .input_paths
            .iter()
            .any(|input| paths_refer_to_same_file(input, &path))
        {
            return Err(format!(
                "FITS frame {} has already been used by this stack",
                path.display()
            ));
        }
        let frame = FitsFrame::open(&path).map_err(|error| error.to_string())?;
        let disposition = stacker
            .stacker
            .push(frame)
            .map_err(|error| error.to_string())?;
        stacker.input_paths.push(path.clone());
        owned_json(&stack_disposition_response(Some(&path), disposition))
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// # Safety
/// `stacker` must be null or a live `SeizaLiveStacker` pointer.
pub unsafe extern "C" fn seiza_live_stacker_width(stacker: *const SeizaLiveStacker) -> usize {
    unsafe {
        stacker
            .as_ref()
            .map_or(0, |stacker| stacker.stacker.view().width)
    }
}

#[unsafe(no_mangle)]
/// # Safety
/// `stacker` must be null or a live `SeizaLiveStacker` pointer.
pub unsafe extern "C" fn seiza_live_stacker_height(stacker: *const SeizaLiveStacker) -> usize {
    unsafe {
        stacker
            .as_ref()
            .map_or(0, |stacker| stacker.stacker.view().height)
    }
}

#[unsafe(no_mangle)]
/// # Safety
/// `stacker` must be null or a live `SeizaLiveStacker` pointer.
pub unsafe extern "C" fn seiza_live_stacker_channels(stacker: *const SeizaLiveStacker) -> usize {
    unsafe {
        stacker
            .as_ref()
            .map_or(0, |stacker| stacker.stacker.view().channels)
    }
}

#[unsafe(no_mangle)]
/// Returns the sample count for every live-view and snapshot buffer.
///
/// # Safety
/// `stacker` must be null or a live `SeizaLiveStacker` pointer.
pub unsafe extern "C" fn seiza_live_stacker_data_length(stacker: *const SeizaLiveStacker) -> usize {
    unsafe {
        stacker.as_ref().map_or(0, |stacker| {
            let view = stacker.stacker.view();
            view.width * view.height * view.channels
        })
    }
}

#[unsafe(no_mangle)]
/// # Safety
/// `stacker` must be null or a live `SeizaLiveStacker` pointer.
pub unsafe extern "C" fn seiza_live_stacker_accepted_frames(
    stacker: *const SeizaLiveStacker,
) -> u32 {
    unsafe {
        stacker
            .as_ref()
            .map_or(0, |stacker| stacker.stacker.view().accepted_frames)
    }
}

#[unsafe(no_mangle)]
/// # Safety
/// `stacker` must be null or a live `SeizaLiveStacker` pointer.
pub unsafe extern "C" fn seiza_live_stacker_rejected_frames(
    stacker: *const SeizaLiveStacker,
) -> u32 {
    unsafe {
        stacker
            .as_ref()
            .map_or(0, |stacker| stacker.stacker.view().rejected_frames)
    }
}

#[unsafe(no_mangle)]
/// Borrows the current interleaved linear mean without copying it. Zero-
/// coverage samples are undefined. The pointer remains valid only until the
/// next mutable stacker operation or the stacker is freed/finished.
///
/// # Safety
/// `stacker` must be null or a live `SeizaLiveStacker` pointer.
pub unsafe extern "C" fn seiza_live_stacker_mean(stacker: *const SeizaLiveStacker) -> *const f32 {
    unsafe {
        stacker
            .as_ref()
            .map_or(ptr::null(), |stacker| stacker.stacker.view().mean.as_ptr())
    }
}

#[unsafe(no_mangle)]
/// Borrows the accepted-observation count for each image sample. The pointer
/// has the same lifetime as [`seiza_live_stacker_mean`].
///
/// # Safety
/// `stacker` must be null or a live `SeizaLiveStacker` pointer.
pub unsafe extern "C" fn seiza_live_stacker_coverage(
    stacker: *const SeizaLiveStacker,
) -> *const u32 {
    unsafe {
        stacker.as_ref().map_or(ptr::null(), |stacker| {
            stacker.stacker.view().coverage.as_ptr()
        })
    }
}

#[unsafe(no_mangle)]
/// Borrows the rejected-observation count for each image sample. The pointer
/// has the same lifetime as [`seiza_live_stacker_mean`].
///
/// # Safety
/// `stacker` must be null or a live `SeizaLiveStacker` pointer.
pub unsafe extern "C" fn seiza_live_stacker_rejected_samples(
    stacker: *const SeizaLiveStacker,
) -> *const u32 {
    unsafe {
        stacker.as_ref().map_or(ptr::null(), |stacker| {
            stacker.stacker.view().rejected_samples.as_ptr()
        })
    }
}

#[unsafe(no_mangle)]
/// Copies the current mean, variance, coverage, and rejection maps into an
/// immutable owned snapshot. Prefer the borrowed live view for display-only
/// updates and [`seiza_live_stacker_finish`] for copy-free finalization.
///
/// # Safety
/// `stacker` must be a live `SeizaLiveStacker` pointer. When non-null,
/// `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_live_stacker_snapshot(
    stacker: *const SeizaLiveStacker,
    error_out: *mut *mut c_char,
) -> *mut SeizaStackSnapshot {
    clear_error(error_out);
    ffi_result(error_out, || {
        let stacker = unsafe { required_live_stacker(stacker)? };
        let reference_headers = stacker.stacker.reference_headers().to_vec();
        let snapshot = stacker
            .stacker
            .snapshot()
            .map_err(|error| error.to_string())?;
        Ok(SeizaStackSnapshot {
            snapshot,
            reference_headers,
            input_paths: stacker.input_paths.clone(),
        })
    })
    .map_or(ptr::null_mut(), |snapshot| {
        Box::into_raw(Box::new(snapshot))
    })
}

#[unsafe(no_mangle)]
/// Consumes a live stacker and moves its full-frame state into an immutable
/// snapshot without cloning it. Once a non-null live handle is accepted,
/// `*stacker` is set to null and consumed even if finalization reports an
/// error.
///
/// # Safety
/// `stacker` must point to writable storage containing null or a live pointer
/// returned by a `seiza_live_stacker_*` constructor. When non-null, `error_out`
/// must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_live_stacker_finish(
    stacker: *mut *mut SeizaLiveStacker,
    error_out: *mut *mut c_char,
) -> *mut SeizaStackSnapshot {
    clear_error(error_out);
    ffi_result(error_out, || {
        if stacker.is_null() {
            return Err("live stacker pointer storage is required".into());
        }
        let live = unsafe { *stacker };
        if live.is_null() {
            return Err("live stacker is required".into());
        }
        unsafe { *stacker = ptr::null_mut() };
        let live = unsafe { Box::from_raw(live) };
        let SeizaLiveStacker {
            stacker,
            input_paths,
        } = *live;
        let reference_headers = stacker.reference_headers().to_vec();
        let snapshot = stacker.into_snapshot().map_err(|error| error.to_string())?;
        Ok(SeizaStackSnapshot {
            snapshot,
            reference_headers,
            input_paths,
        })
    })
    .map_or(ptr::null_mut(), |snapshot| {
        Box::into_raw(Box::new(snapshot))
    })
}

#[unsafe(no_mangle)]
/// # Safety
/// `stacker` must be null or a live pointer returned by a
/// `seiza_live_stacker_*` constructor and must not already be finished/freed.
pub unsafe extern "C" fn seiza_live_stacker_free(stacker: *mut SeizaLiveStacker) {
    if !stacker.is_null() {
        unsafe { drop(Box::from_raw(stacker)) };
    }
}

#[unsafe(no_mangle)]
/// # Safety
/// `snapshot` must be null or a live `SeizaStackSnapshot` pointer.
pub unsafe extern "C" fn seiza_stack_snapshot_width(snapshot: *const SeizaStackSnapshot) -> usize {
    unsafe {
        snapshot
            .as_ref()
            .map_or(0, |value| value.snapshot.image.width)
    }
}

#[unsafe(no_mangle)]
/// # Safety
/// `snapshot` must be null or a live `SeizaStackSnapshot` pointer.
pub unsafe extern "C" fn seiza_stack_snapshot_height(snapshot: *const SeizaStackSnapshot) -> usize {
    unsafe {
        snapshot
            .as_ref()
            .map_or(0, |value| value.snapshot.image.height)
    }
}

#[unsafe(no_mangle)]
/// # Safety
/// `snapshot` must be null or a live `SeizaStackSnapshot` pointer.
pub unsafe extern "C" fn seiza_stack_snapshot_channels(
    snapshot: *const SeizaStackSnapshot,
) -> usize {
    unsafe {
        snapshot
            .as_ref()
            .map_or(0, |value| value.snapshot.image.channels)
    }
}

#[unsafe(no_mangle)]
/// Returns the sample count for every snapshot buffer.
///
/// # Safety
/// `snapshot` must be null or a live `SeizaStackSnapshot` pointer.
pub unsafe extern "C" fn seiza_stack_snapshot_data_length(
    snapshot: *const SeizaStackSnapshot,
) -> usize {
    unsafe {
        snapshot
            .as_ref()
            .map_or(0, |value| value.snapshot.image.sample_count())
    }
}

#[unsafe(no_mangle)]
/// # Safety
/// `snapshot` must be null or a live `SeizaStackSnapshot` pointer.
pub unsafe extern "C" fn seiza_stack_snapshot_accepted_frames(
    snapshot: *const SeizaStackSnapshot,
) -> u32 {
    unsafe {
        snapshot
            .as_ref()
            .map_or(0, |value| value.snapshot.accepted_frames)
    }
}

#[unsafe(no_mangle)]
/// # Safety
/// `snapshot` must be null or a live `SeizaStackSnapshot` pointer.
pub unsafe extern "C" fn seiza_stack_snapshot_rejected_frames(
    snapshot: *const SeizaStackSnapshot,
) -> u32 {
    unsafe {
        snapshot
            .as_ref()
            .map_or(0, |value| value.snapshot.rejected_frames)
    }
}

#[unsafe(no_mangle)]
/// Borrows the immutable interleaved linear mean until the snapshot is freed.
///
/// # Safety
/// `snapshot` must be null or a live `SeizaStackSnapshot` pointer.
pub unsafe extern "C" fn seiza_stack_snapshot_image(
    snapshot: *const SeizaStackSnapshot,
) -> *const f32 {
    unsafe {
        snapshot
            .as_ref()
            .map_or(ptr::null(), |value| value.snapshot.image.data.as_ptr())
    }
}

#[unsafe(no_mangle)]
/// Borrows the immutable per-sample variance until the snapshot is freed.
///
/// # Safety
/// `snapshot` must be null or a live `SeizaStackSnapshot` pointer.
pub unsafe extern "C" fn seiza_stack_snapshot_variance(
    snapshot: *const SeizaStackSnapshot,
) -> *const f32 {
    unsafe {
        snapshot
            .as_ref()
            .map_or(ptr::null(), |value| value.snapshot.variance.data.as_ptr())
    }
}

#[unsafe(no_mangle)]
/// Borrows the immutable per-sample accepted count until the snapshot is freed.
///
/// # Safety
/// `snapshot` must be null or a live `SeizaStackSnapshot` pointer.
pub unsafe extern "C" fn seiza_stack_snapshot_coverage(
    snapshot: *const SeizaStackSnapshot,
) -> *const u32 {
    unsafe {
        snapshot
            .as_ref()
            .map_or(ptr::null(), |value| value.snapshot.coverage.as_ptr())
    }
}

#[unsafe(no_mangle)]
/// Borrows the immutable per-sample rejection count until the snapshot is
/// freed.
///
/// # Safety
/// `snapshot` must be null or a live `SeizaStackSnapshot` pointer.
pub unsafe extern "C" fn seiza_stack_snapshot_rejected_samples(
    snapshot: *const SeizaStackSnapshot,
) -> *const u32 {
    unsafe {
        snapshot.as_ref().map_or(ptr::null(), |value| {
            value.snapshot.rejected_samples.as_ptr()
        })
    }
}

#[unsafe(no_mangle)]
/// Writes the immutable stack as an unstretched 32-bit floating-point FITS,
/// preserving compatible reference headers.
///
/// # Safety
/// `snapshot` must be a live `SeizaStackSnapshot` pointer. `path` must be a
/// valid NUL-terminated string. When non-null, `error_out` must point to
/// writable storage for one pointer.
pub unsafe extern "C" fn seiza_stack_snapshot_write_fits(
    snapshot: *const SeizaStackSnapshot,
    path: *const c_char,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let snapshot = unsafe { required_stack_snapshot(snapshot)? };
        let path = required_path(path, "stack output path")?;
        if snapshot
            .input_paths
            .iter()
            .any(|input| paths_refer_to_same_file(input, &path))
        {
            return Err(
                "stack output path must not refer to an input frame or calibration master".into(),
            );
        }
        write_fits_f32(path, &snapshot.snapshot, &snapshot.reference_headers)
            .map_err(|error| error.to_string())
    })
    .is_some()
}

#[unsafe(no_mangle)]
/// # Safety
/// `snapshot` must be null or a live pointer returned by a snapshot/finalize
/// function and must not already be freed.
pub unsafe extern "C" fn seiza_stack_snapshot_free(snapshot: *mut SeizaStackSnapshot) {
    if !snapshot.is_null() {
        unsafe { drop(Box::from_raw(snapshot)) };
    }
}

#[unsafe(no_mangle)]
/// Returns catalog readiness and resolved component paths as JSON.
///
/// # Safety
/// `catalog_directory` may be null or a valid NUL-terminated string. When
/// non-null, `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_catalog_status_json(
    catalog_directory: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let catalog_directory = optional_path(catalog_directory)?;
        let status = catalog_status(catalog_directory.as_deref());
        let json = serde_json::to_string(&status).map_err(|error| error.to_string())?;
        CString::new(json)
            .map(CString::into_raw)
            .map_err(|_| "catalog status contains a NUL byte".to_string())
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Downloads and installs a solver-ready Seiza catalog preset.
///
/// Preset `0` is the standard G≤17 blind-solving package, `1` is the optional
/// G≤20 package, and `2` installs every published catalog. The call is
/// synchronous and must run off the UI thread. Progress JSON is valid only for
/// the duration of each callback.
///
/// # Safety
/// `catalog_directory` may be null or a valid NUL-terminated string. `context`
/// is passed through untouched to `progress`. When non-null, `error_out` must
/// point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_catalog_setup(
    catalog_directory: *const c_char,
    preset: u32,
    progress: SeizaCatalogSetupProgressCallback,
    context: *mut c_void,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let catalog_directory = optional_path(catalog_directory)?;
        let preset = CatalogSetupPreset::from_raw(preset)?;
        run_catalog_setup(
            catalog_directory.as_deref(),
            preset,
            CatalogSetupReporter {
                callback: progress,
                context: context as usize,
                files_total: preset.datasets().len(),
            },
        )
    })
    .is_some()
}

#[unsafe(no_mangle)]
/// Opens and renders an image for the C ABI.
///
/// # Safety
/// `path` must be a valid NUL-terminated string. When non-null, `error_out`
/// must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_rendered_image_open(
    path: *const c_char,
    target_median: f64,
    shadows_clip: f64,
    max_dimension: u32,
    error_out: *mut *mut c_char,
) -> *mut SeizaRenderedImage {
    open_rendered_image(
        path,
        target_median,
        shadows_clip,
        max_dimension,
        RgbStretchMode::Auto,
        error_out,
    )
}

#[unsafe(no_mangle)]
/// Opens and renders an image with an explicit RGB stretch mode.
///
/// Mode `0` is per-channel auto, `1` is linked auto, and `2` is linear.
/// Non-RGB FITS and standard raster images ignore this setting.
///
/// # Safety
/// `path` must be a valid NUL-terminated string. When non-null, `error_out`
/// must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_rendered_image_open_with_rgb_stretch(
    path: *const c_char,
    target_median: f64,
    shadows_clip: f64,
    max_dimension: u32,
    rgb_stretch_mode: u32,
    error_out: *mut *mut c_char,
) -> *mut SeizaRenderedImage {
    clear_error(error_out);
    ffi_result(error_out, || {
        let mode = RgbStretchMode::from_raw(rgb_stretch_mode)?;
        render_image(path, target_median, shadows_clip, max_dimension, mode)
    })
    .map_or(ptr::null_mut(), |image| Box::into_raw(Box::new(image)))
}

#[unsafe(no_mangle)]
/// Opens a FITS image and renders it with parameterized processing described by
/// `config_json`. The value may be one serialized `seiza-stretch`
/// `StretchConfig` (the original schema), a non-empty array of configs, or an
/// object with `stretch`, optional `background`, optional `deconvolution`, and
/// optional `interactive_preview` fields. Array stages are applied in order
/// using `f32` intermediates and converted to RGBA only after the final stage.
/// Background correction and deconvolution, when requested, are applied to
/// linear samples in that order before the first stretch stage. Interactive
/// preview mode bounds the linear samples to `max_dimension` before processing
/// and reuses the source/background-prepared pixels across stretch and
/// deconvolution edits; full renders should leave it false.
///
/// # Safety
/// `path` and `config_json` must be valid NUL-terminated strings. When non-null,
/// `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_rendered_image_open_with_stretch_config(
    path: *const c_char,
    config_json: *const c_char,
    max_dimension: u32,
    error_out: *mut *mut c_char,
) -> *mut SeizaRenderedImage {
    clear_error(error_out);
    ffi_result(error_out, || {
        let path = required_path(path, "image path")?;
        let config_json = required_str(config_json, "stretch config JSON")?;
        let request: ImageRenderConfigRequest = serde_json::from_str(&config_json)
            .map_err(|error| format!("invalid image processing config JSON: {error}"))?;
        let (stack, background, deconvolution, interactive_preview) = request.into_parts();
        if interactive_preview {
            render_cached_interactive_preview(
                &path,
                &stack,
                background.as_ref(),
                deconvolution.as_ref(),
                max_dimension,
            )
        } else {
            let fits = FitsImage::open(&path)
                .map_err(|error| format!("failed to open {}: {error}", path.display()))?;
            render_fits_with_pipeline(
                fits,
                &stack,
                background.as_ref(),
                deconvolution.as_ref(),
                max_dimension,
                false,
            )
        }
    })
    .map_or(ptr::null_mut(), |image| Box::into_raw(Box::new(image)))
}

#[unsafe(no_mangle)]
/// Opens and renders an image to native-endian RGBA16 for high-bit-depth
/// export. This is a separate allocation from the RGBA8 preview API, so normal
/// preview renders do not pay the memory cost of both pixel formats.
///
/// # Safety
/// `path` must be a valid NUL-terminated string. When non-null, `error_out`
/// must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_rendered_image16_open(
    path: *const c_char,
    target_median: f64,
    shadows_clip: f64,
    max_dimension: u32,
    error_out: *mut *mut c_char,
) -> *mut SeizaRenderedImage16 {
    open_rendered_image16(
        path,
        target_median,
        shadows_clip,
        max_dimension,
        RgbStretchMode::Auto,
        error_out,
    )
}

#[unsafe(no_mangle)]
/// Opens and renders an image to native-endian RGBA16 with an explicit RGB
/// stretch mode. Mode `0` is per-channel auto, `1` is linked auto, and `2` is
/// linear. Non-RGB FITS and standard raster images ignore this setting.
///
/// # Safety
/// `path` must be a valid NUL-terminated string. When non-null, `error_out`
/// must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_rendered_image16_open_with_rgb_stretch(
    path: *const c_char,
    target_median: f64,
    shadows_clip: f64,
    max_dimension: u32,
    rgb_stretch_mode: u32,
    error_out: *mut *mut c_char,
) -> *mut SeizaRenderedImage16 {
    clear_error(error_out);
    ffi_result(error_out, || {
        let mode = RgbStretchMode::from_raw(rgb_stretch_mode)?;
        render_image16(path, target_median, shadows_clip, max_dimension, mode)
    })
    .map_or(ptr::null_mut(), |image| Box::into_raw(Box::new(image)))
}

#[unsafe(no_mangle)]
/// Opens a FITS image and renders its parameterized processing stack to
/// native-endian RGBA16. The JSON schema and processing order are identical to
/// [`seiza_rendered_image_open_with_stretch_config`], but the final stretch is
/// quantized directly from `f32` to `u16` instead of passing through RGBA8.
///
/// # Safety
/// `path` and `config_json` must be valid NUL-terminated strings. When non-null,
/// `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_rendered_image16_open_with_stretch_config(
    path: *const c_char,
    config_json: *const c_char,
    max_dimension: u32,
    error_out: *mut *mut c_char,
) -> *mut SeizaRenderedImage16 {
    clear_error(error_out);
    ffi_result(error_out, || {
        let path = required_path(path, "image path")?;
        let config_json = required_str(config_json, "stretch config JSON")?;
        let request: ImageRenderConfigRequest = serde_json::from_str(&config_json)
            .map_err(|error| format!("invalid image processing config JSON: {error}"))?;
        let (stack, background, deconvolution, interactive_preview) = request.into_parts();
        if interactive_preview {
            render_cached_interactive_preview16(
                &path,
                &stack,
                background.as_ref(),
                deconvolution.as_ref(),
                max_dimension,
            )
        } else {
            let fits = FitsImage::open(&path)
                .map_err(|error| format!("failed to open {}: {error}", path.display()))?;
            render_fits_with_pipeline16(
                fits,
                &stack,
                background.as_ref(),
                deconvolution.as_ref(),
                max_dimension,
                false,
            )
        }
    })
    .map_or(ptr::null_mut(), |image| Box::into_raw(Box::new(image)))
}

fn open_rendered_image(
    path: *const c_char,
    target_median: f64,
    shadows_clip: f64,
    max_dimension: u32,
    rgb_stretch_mode: RgbStretchMode,
    error_out: *mut *mut c_char,
) -> *mut SeizaRenderedImage {
    clear_error(error_out);
    ffi_result(error_out, || {
        render_image(
            path,
            target_median,
            shadows_clip,
            max_dimension,
            rgb_stretch_mode,
        )
    })
    .map_or(ptr::null_mut(), |image| Box::into_raw(Box::new(image)))
}

fn open_rendered_image16(
    path: *const c_char,
    target_median: f64,
    shadows_clip: f64,
    max_dimension: u32,
    rgb_stretch_mode: RgbStretchMode,
    error_out: *mut *mut c_char,
) -> *mut SeizaRenderedImage16 {
    clear_error(error_out);
    ffi_result(error_out, || {
        render_image16(
            path,
            target_median,
            shadows_clip,
            max_dimension,
            rgb_stretch_mode,
        )
    })
    .map_or(ptr::null_mut(), |image| Box::into_raw(Box::new(image)))
}

fn render_image(
    path: *const c_char,
    target_median: f64,
    shadows_clip: f64,
    max_dimension: u32,
    rgb_stretch_mode: RgbStretchMode,
) -> Result<SeizaRenderedImage, String> {
    let path = required_path(path, "image path")?;
    let params = StretchParams {
        target_median: target_median.clamp(0.01, 0.95),
        shadows_clip: shadows_clip.clamp(-10.0, 0.0),
    };
    render_path(&path, &params, max_dimension, rgb_stretch_mode)
}

fn render_image16(
    path: *const c_char,
    target_median: f64,
    shadows_clip: f64,
    max_dimension: u32,
    rgb_stretch_mode: RgbStretchMode,
) -> Result<SeizaRenderedImage16, String> {
    let path = required_path(path, "image path")?;
    let params = StretchParams {
        target_median: target_median.clamp(0.01, 0.95),
        shadows_clip: shadows_clip.clamp(-10.0, 0.0),
    };
    render_path16(&path, &params, max_dimension, rgb_stretch_mode)
}

#[unsafe(no_mangle)]
/// # Safety
/// `image` must be null or a live pointer returned by
/// [`seiza_rendered_image_open`].
pub unsafe extern "C" fn seiza_rendered_image_width(image: *const SeizaRenderedImage) -> u32 {
    unsafe { image.as_ref().map_or(0, |image| image.width) }
}

#[unsafe(no_mangle)]
/// # Safety
/// `image` must be null or a live pointer returned by
/// [`seiza_rendered_image_open`].
pub unsafe extern "C" fn seiza_rendered_image_height(image: *const SeizaRenderedImage) -> u32 {
    unsafe { image.as_ref().map_or(0, |image| image.height) }
}

#[unsafe(no_mangle)]
/// # Safety
/// `image` must be null or a live pointer returned by
/// [`seiza_rendered_image_open`]. The returned buffer is valid until the image
/// is freed.
pub unsafe extern "C" fn seiza_rendered_image_rgba(image: *const SeizaRenderedImage) -> *const u8 {
    unsafe {
        image
            .as_ref()
            .map_or(ptr::null(), |image| image.rgba.as_ptr())
    }
}

#[unsafe(no_mangle)]
/// # Safety
/// `image` must be null or a live pointer returned by
/// [`seiza_rendered_image_open`].
pub unsafe extern "C" fn seiza_rendered_image_rgba_length(
    image: *const SeizaRenderedImage,
) -> usize {
    unsafe { image.as_ref().map_or(0, |image| image.rgba.len()) }
}

#[unsafe(no_mangle)]
/// Returns the image as BGRA8 (Direct2D / WinUI byte order), computed from the
/// canonical RGBA on first use and cached. The returned buffer is valid until
/// the image is freed.
///
/// # Safety
/// `image` must be null or a live pointer returned by
/// [`seiza_rendered_image_open`].
pub unsafe extern "C" fn seiza_rendered_image_bgra(image: *const SeizaRenderedImage) -> *const u8 {
    unsafe {
        image
            .as_ref()
            .map_or(ptr::null(), |image| image.bgra().as_ptr())
    }
}

#[unsafe(no_mangle)]
/// # Safety
/// `image` must be null or a live pointer returned by
/// [`seiza_rendered_image_open`].
pub unsafe extern "C" fn seiza_rendered_image_bgra_length(
    image: *const SeizaRenderedImage,
) -> usize {
    unsafe { image.as_ref().map_or(0, |image| image.bgra().len()) }
}

#[unsafe(no_mangle)]
/// # Safety
/// `image` must be null or a live pointer returned by
/// [`seiza_rendered_image_open`]. The returned string is valid until the image
/// is freed.
pub unsafe extern "C" fn seiza_rendered_image_metadata_json(
    image: *const SeizaRenderedImage,
) -> *const c_char {
    unsafe {
        image
            .as_ref()
            .map_or(ptr::null(), |image| image.metadata_json.as_ptr())
    }
}

#[unsafe(no_mangle)]
/// # Safety
/// `image` must be null or a pointer returned by [`seiza_rendered_image_open`]
/// that has not already been freed.
pub unsafe extern "C" fn seiza_rendered_image_free(image: *mut SeizaRenderedImage) {
    if !image.is_null() {
        unsafe { drop(Box::from_raw(image)) };
    }
}

#[unsafe(no_mangle)]
/// # Safety
/// `image` must be null or a live pointer returned by a
/// `seiza_rendered_image16_open*` function.
pub unsafe extern "C" fn seiza_rendered_image16_width(image: *const SeizaRenderedImage16) -> u32 {
    unsafe { image.as_ref().map_or(0, |image| image.width) }
}

#[unsafe(no_mangle)]
/// # Safety
/// `image` must be null or a live pointer returned by a
/// `seiza_rendered_image16_open*` function.
pub unsafe extern "C" fn seiza_rendered_image16_height(image: *const SeizaRenderedImage16) -> u32 {
    unsafe { image.as_ref().map_or(0, |image| image.height) }
}

#[unsafe(no_mangle)]
/// Returns borrowed native-endian RGBA16 samples. The returned buffer remains
/// valid until the image is freed.
///
/// # Safety
/// `image` must be null or a live pointer returned by a
/// `seiza_rendered_image16_open*` function.
pub unsafe extern "C" fn seiza_rendered_image16_rgba(
    image: *const SeizaRenderedImage16,
) -> *const u16 {
    unsafe {
        image
            .as_ref()
            .map_or(ptr::null(), |image| image.rgba.as_ptr())
    }
}

#[unsafe(no_mangle)]
/// Returns the RGBA16 buffer length in `uint16_t` elements, not bytes.
///
/// # Safety
/// `image` must be null or a live pointer returned by a
/// `seiza_rendered_image16_open*` function.
pub unsafe extern "C" fn seiza_rendered_image16_rgba_length(
    image: *const SeizaRenderedImage16,
) -> usize {
    unsafe { image.as_ref().map_or(0, |image| image.rgba.len()) }
}

#[unsafe(no_mangle)]
/// Returns borrowed render metadata JSON. The string remains valid until the
/// image is freed.
///
/// # Safety
/// `image` must be null or a live pointer returned by a
/// `seiza_rendered_image16_open*` function.
pub unsafe extern "C" fn seiza_rendered_image16_metadata_json(
    image: *const SeizaRenderedImage16,
) -> *const c_char {
    unsafe {
        image
            .as_ref()
            .map_or(ptr::null(), |image| image.metadata_json.as_ptr())
    }
}

#[unsafe(no_mangle)]
/// # Safety
/// `image` must be null or a pointer returned by a
/// `seiza_rendered_image16_open*` function that has not already been freed.
pub unsafe extern "C" fn seiza_rendered_image16_free(image: *mut SeizaRenderedImage16) {
    if !image.is_null() {
        unsafe { drop(Box::from_raw(image)) };
    }
}

#[unsafe(no_mangle)]
/// Solves an image and returns a JSON string for the C ABI.
///
/// # Safety
/// `path` must be a valid NUL-terminated string. `catalog_directory` may be
/// null or a valid NUL-terminated string. When non-null, `error_out` must point
/// to writable storage for one pointer.
pub unsafe extern "C" fn seiza_solve_image_json(
    path: *const c_char,
    catalog_directory: *const c_char,
    minimum_scale_arcsec_per_pixel: f64,
    maximum_scale_arcsec_per_pixel: f64,
    sip_order: u8,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let started = Instant::now();
        let path = required_path(path, "image path")?;
        let catalog_directory = optional_path(catalog_directory)?;
        let detection_config = DetectConfig {
            max_stars: 600,
            ..Default::default()
        };
        let (width, height, mut stars, raster_fallback, capture_time) = if is_fits_path(&path) {
            let fits = FitsImage::open(&path).map_err(|error| error.to_string())?;
            let width = u32::try_from(fits.width).map_err(|_| "image width is too large")?;
            let height = u32::try_from(fits.height).map_err(|_| "image height is too large")?;
            let capture_time = fits_capture_time(&fits);
            let luma = fits.to_luma_f32();
            let stars = detect_stars_luma_f32(&luma, width, height, &detection_config);
            (width, height, stars, None, capture_time)
        } else {
            let image = image::open(&path)
                .map_err(|error| format!("failed to open {}: {error}", path.display()))?;
            let width = image.width();
            let height = image.height();
            let stars = detect_stars(&image, &detection_config);
            let fallback = is_converted_8bit_color(&image).then_some(image);
            (width, height, stars, fallback, None)
        };
        let acquisition_jd = capture_time.as_deref().and_then(parse_iso_jd);

        let star_path = seiza::data_paths::star_data(catalog_directory.as_deref())
            .map_err(|error| error.to_string())?;
        let index_path = seiza::data_paths::blind_index(catalog_directory.as_deref())
            .map_err(|error| error.to_string())?
            .ok_or_else(|| {
                "no blind index found; install a complete Seiza catalog bundle first".to_string()
            })?;
        let catalog = TileCatalog::open(&star_path)
            .map_err(|error| format!("failed to open {}: {error}", star_path.display()))?;
        let index = BlindIndex::open(&index_path)
            .map_err(|error| format!("failed to open {}: {error}", index_path.display()))?;

        let params = BlindParams {
            min_scale_arcsec_px: minimum_scale_arcsec_per_pixel.max(0.01),
            max_scale_arcsec_px: maximum_scale_arcsec_per_pixel
                .max(minimum_scale_arcsec_per_pixel.max(0.01)),
            index_mag_limit: index.index_mag_limit(),
            max_pattern_deg: index.max_pattern_deg(),
            sip_order: sip_order.min(5),
            ..Default::default()
        };
        let solution = match solve_blind(&stars, &catalog, &index, &params, (width, height)) {
            Ok(solution) => solution,
            Err(primary_error) => {
                let Some(image) = raster_fallback else {
                    return Err(primary_error.to_string());
                };
                stars = detect_stars(
                    &image,
                    &DetectConfig {
                        backend: DetectBackend::F32,
                        ..detection_config
                    },
                );
                solve_blind(&stars, &catalog, &index, &params, (width, height))
                    .map_err(|error| error.to_string())?
            }
        };
        let center = solution
            .wcs
            .pixel_to_world(width as f64 / 2.0, height as f64 / 2.0);
        let detected_star_positions = stars
            .iter()
            .take(300)
            .map(|star| ImagePointResponse {
                x: star.x,
                y: star.y,
            })
            .collect();
        let field_radius_degrees =
            (width as f64).hypot(height as f64) / 2.0 * solution.wcs.scale_arcsec_per_px() / 3600.0
                * 1.1;
        let catalog_star_positions: Vec<_> = catalog
            .cone_search(center.0, center.1, field_radius_degrees.max(0.05), 1_000)
            .into_iter()
            .filter(|star| star.mag <= 10.0)
            .filter_map(|star| {
                let (x, y) = solution.wcs.world_to_pixel(star.ra, star.dec)?;
                (x >= 0.0 && y >= 0.0 && x < width as f64 && y < height as f64).then_some(
                    CatalogStarPointResponse {
                        x,
                        y,
                        magnitude: star.mag,
                    },
                )
            })
            .take(300)
            .collect();
        let mut object_positions = Vec::new();
        let mut overlay_availability = BTreeMap::from([
            ("deep_sky".into(), false),
            ("named_stars".into(), false),
            ("field_stars".into(), true),
            ("transients".into(), false),
            ("historical_transients".into(), false),
            ("minor_bodies".into(), false),
            ("grid".into(), true),
        ]);
        let mut overlay_unavailable_reasons = BTreeMap::new();

        let object_catalog_result = (|| -> Result<ObjectCatalog, String> {
            let object_path = seiza::data_paths::objects(catalog_directory.as_deref())
                .map_err(|error| error.to_string())?;
            ObjectCatalog::open(&object_path)
                .map_err(|error| format!("failed to open {}: {error}", object_path.display()))
        })();
        let object_catalog_error = match object_catalog_result {
            Ok(object_catalog) => {
                overlay_availability.insert("deep_sky".into(), true);
                overlay_availability.insert("named_stars".into(), true);
                if let Err(error) = append_object_catalog(
                    &mut object_positions,
                    &object_catalog,
                    &solution.wcs,
                    (width, height),
                    acquisition_jd,
                    false,
                ) {
                    overlay_availability.insert("deep_sky".into(), false);
                    overlay_availability.insert("named_stars".into(), false);
                    overlay_unavailable_reasons.insert("deep_sky".into(), error.clone());
                    overlay_unavailable_reasons.insert("named_stars".into(), error.clone());
                    Some(error)
                } else {
                    None
                }
            }
            Err(error) => {
                overlay_unavailable_reasons.insert("deep_sky".into(), error.clone());
                overlay_unavailable_reasons.insert("named_stars".into(), error.clone());
                Some(error)
            }
        };

        match open_object_catalog(
            seiza::data_paths::transients(catalog_directory.as_deref()),
            "transient",
        ) {
            Ok(transient_catalog) => {
                overlay_availability.insert("transients".into(), true);
                overlay_availability.insert("historical_transients".into(), true);
                if let Err(error) = append_object_catalog(
                    &mut object_positions,
                    &transient_catalog,
                    &solution.wcs,
                    (width, height),
                    acquisition_jd,
                    true,
                ) {
                    overlay_availability.insert("transients".into(), false);
                    overlay_availability.insert("historical_transients".into(), false);
                    overlay_unavailable_reasons.insert("transients".into(), error.clone());
                    overlay_unavailable_reasons.insert("historical_transients".into(), error);
                }
            }
            Err(error) => {
                overlay_unavailable_reasons.insert("transients".into(), error.clone());
                overlay_unavailable_reasons.insert("historical_transients".into(), error);
            }
        }

        match open_minor_body_catalog(catalog_directory.as_deref()) {
            Ok(minor_body_catalog) => {
                if let Some(jd) = acquisition_jd {
                    overlay_availability.insert("minor_bodies".into(), true);
                    append_minor_bodies(
                        &mut object_positions,
                        &minor_body_catalog,
                        &solution.wcs,
                        (width, height),
                        jd,
                    );
                } else {
                    overlay_unavailable_reasons.insert(
                        "minor_bodies".into(),
                        "Solar-system positions require a FITS DATE-OBS acquisition time".into(),
                    );
                }
            }
            Err(error) => {
                overlay_unavailable_reasons.insert("minor_bodies".into(), error);
            }
        }

        let mut overlay_counts = BTreeMap::from([
            ("deep_sky".into(), 0),
            ("named_stars".into(), 0),
            ("field_stars".into(), catalog_star_positions.len()),
            ("transients".into(), 0),
            ("historical_transients".into(), 0),
            ("minor_bodies".into(), 0),
        ]);
        for object in &object_positions {
            let layer = overlay_layer_name(&object.kind);
            *overlay_counts.entry(layer.into()).or_insert(0) += 1;
            if object.kind == "transient" && object.near_capture == Some(false) {
                *overlay_counts
                    .entry("historical_transients".into())
                    .or_insert(0) += 1;
            }
        }
        let sip = solution.wcs.sip.as_ref().map(|sip| SipResponse {
            order: sip.order,
            a: sip.a.clone(),
            b: sip.b.clone(),
            ap: sip.ap.clone(),
            bp: sip.bp.clone(),
        });
        let response = SolveResponse {
            center_ra_degrees: center.0,
            center_dec_degrees: center.1,
            scale_arcsec_per_pixel: solution.wcs.scale_arcsec_per_px(),
            matched_stars: solution.matched_stars,
            rms_arcsec: solution.rms_arcsec,
            detected_stars: stars.len(),
            elapsed_milliseconds: started.elapsed().as_millis(),
            detected_star_positions,
            catalog_star_positions,
            object_positions,
            object_catalog_error,
            capture_time,
            overlay_availability,
            overlay_unavailable_reasons,
            overlay_counts,
            wcs: WcsResponse {
                crval: [solution.wcs.crval.0, solution.wcs.crval.1],
                crpix: [solution.wcs.crpix.0, solution.wcs.crpix.1],
                cd: solution.wcs.cd,
                sip,
            },
        };
        let json = serde_json::to_string(&response).map_err(|error| error.to_string())?;
        CString::new(json).map_err(|_| "solution JSON contains a null byte".to_string())
    })
    .map_or(ptr::null_mut(), CString::into_raw)
}

fn open_object_catalog(
    path: Result<PathBuf, seiza::data_paths::DataPathError>,
    label: &str,
) -> Result<ObjectCatalog, String> {
    let path = path.map_err(|error| error.to_string())?;
    ObjectCatalog::open(&path)
        .map_err(|error| format!("failed to open {label} catalog {}: {error}", path.display()))
}

fn open_minor_body_catalog(catalog_directory: Option<&Path>) -> Result<MinorBodyCatalog, String> {
    let path =
        seiza::data_paths::minor_bodies(catalog_directory).map_err(|error| error.to_string())?;
    MinorBodyCatalog::open(&path).map_err(|error| {
        format!(
            "failed to open minor-body catalog {}: {error}",
            path.display()
        )
    })
}

fn append_object_catalog(
    output: &mut Vec<ObjectPointResponse>,
    catalog: &ObjectCatalog,
    wcs: &Wcs,
    dimensions: (u32, u32),
    capture_jd: Option<f64>,
    force_transient: bool,
) -> Result<(), String> {
    let prominence_by_id: HashMap<String, f64> = catalog
        .query_region(
            &SkyRegion::Polygon {
                vertices: wcs.footprint(dimensions.0, dimensions.1).to_vec(),
            },
            &ObjectQuery::default(),
        )
        .map_err(|error| error.to_string())?
        .into_iter()
        .map(|hit| (hit.object.metadata.id, hit.predicted_prominence))
        .collect();
    let placed = catalog
        .objects_in_footprint(wcs, dimensions)
        .map_err(|error| error.to_string())?;
    for placed in placed {
        let transient = force_transient || placed.object.kind == ObjectKind::Transient;
        let stable_id =
            (!placed.object.metadata.id.is_empty()).then(|| placed.object.metadata.id.clone());
        let prominence = stable_id
            .as_ref()
            .and_then(|id| prominence_by_id.get(id))
            .copied();
        let outlines = stable_id
            .as_deref()
            .map(|id| projected_outlines(catalog, id, wcs))
            .unwrap_or_default();
        let discovered = transient
            .then(|| transient_discovery_date(&placed.object.common_name))
            .flatten();
        let near_capture =
            transient.then(|| transient_near_capture(discovered.as_deref(), capture_jd));
        let catalog_source = (!placed.object.metadata.source.is_empty())
            .then(|| placed.object.metadata.source.clone());
        output.push(ObjectPointResponse {
            stable_id,
            name: placed.object.name,
            common_name: placed.object.common_name,
            kind: if force_transient {
                "transient".into()
            } else {
                placed.object.kind.as_str().into()
            },
            source: if transient {
                "transient".into()
            } else {
                "deep_sky".into()
            },
            catalog_source,
            x: placed.x,
            y: placed.y,
            semi_major_pixels: placed.semi_major_px,
            semi_minor_pixels: placed.semi_minor_px,
            angle_degrees: placed.angle_deg,
            prominence,
            ra_degrees: Some(placed.object.ra),
            dec_degrees: Some(placed.object.dec),
            discovered,
            near_capture,
            distance_au: None,
            motion_arcsec_per_hour: None,
            direction_position_angle_degrees: None,
            direction_image_angle_degrees: None,
            outlines,
        });
    }
    Ok(())
}

fn append_minor_bodies(
    output: &mut Vec<ObjectPointResponse>,
    catalog: &MinorBodyCatalog,
    wcs: &Wcs,
    dimensions: (u32, u32),
    acquisition_jd: f64,
) {
    for placed in catalog.objects_in_footprint(wcs, dimensions, acquisition_jd, 18.0) {
        let kind = match placed.body.kind {
            MinorBodyKind::Comet => "comet",
            MinorBodyKind::Asteroid => "asteroid",
        };
        output.push(ObjectPointResponse {
            stable_id: None,
            name: placed.body.name,
            common_name: format!("V~{:.1}, {:.2} AU", placed.mag, placed.delta_au),
            kind: kind.into(),
            source: "minor_body".into(),
            catalog_source: None,
            x: placed.x,
            y: placed.y,
            semi_major_pixels: 0.0,
            semi_minor_pixels: 0.0,
            angle_degrees: Some(0.0),
            prominence: None,
            ra_degrees: Some(placed.ra),
            dec_degrees: Some(placed.dec),
            discovered: None,
            near_capture: Some(true),
            distance_au: Some(placed.delta_au),
            motion_arcsec_per_hour: placed.motion_arcsec_per_hour,
            direction_position_angle_degrees: placed.direction_pa_deg,
            direction_image_angle_degrees: placed
                .direction_pa_deg
                .and_then(|angle| direction_image_angle(wcs, placed.ra, placed.dec, angle)),
            outlines: Vec::new(),
        });
    }
}

fn direction_image_angle(wcs: &Wcs, ra: f64, dec: f64, pa_deg: f64) -> Option<f64> {
    let (x, y) = wcs.world_to_pixel(ra, dec)?;
    let epsilon = 1.0 / 60.0;
    let north = wcs.world_to_pixel(ra, (dec + epsilon).min(90.0))?;
    let east = wcs.world_to_pixel(ra + epsilon / dec.to_radians().cos().abs().max(1e-6), dec)?;
    let normalize = |point: (f64, f64)| {
        let vector = (point.0 - x, point.1 - y);
        let length = vector.0.hypot(vector.1).max(1e-12);
        (vector.0 / length, vector.1 / length)
    };
    let north = normalize(north);
    let east = normalize(east);
    let (sin, cos) = pa_deg.to_radians().sin_cos();
    Some(
        (north.1 * cos + east.1 * sin)
            .atan2(north.0 * cos + east.0 * sin)
            .to_degrees(),
    )
}

fn fits_capture_time(fits: &FitsImage) -> Option<String> {
    ["DATE-OBS", "DATE-BEG", "DATE-AVG"]
        .into_iter()
        .find_map(|key| {
            fits.headers
                .iter()
                .find(|(name, _)| name == key)
                .and_then(|(_, value)| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
        })
}

/// Parse the FITS ISO-8601 forms used by Seiza into a Julian date.
fn parse_iso_jd(value: &str) -> Option<f64> {
    let value = value.trim().trim_end_matches('Z');
    let (date, clock) = value.split_once('T').unwrap_or((value, "0:0:0"));
    let mut date_parts = date.split('-');
    let year: i32 = date_parts.next()?.parse().ok()?;
    let month: u32 = date_parts.next()?.parse().ok()?;
    let day: u32 = date_parts.next()?.parse().ok()?;
    let mut clock_parts = clock.split(':');
    let hour: f64 = clock_parts.next()?.parse().ok()?;
    let minute: f64 = clock_parts.next().unwrap_or("0").parse().ok()?;
    let second: f64 = clock_parts.next().unwrap_or("0").parse().ok()?;
    let day_fraction = day as f64 + (hour + minute / 60.0 + second / 3_600.0) / 24.0;
    Some(seiza::minor_bodies::julian_date(year, month, day_fraction))
}

fn transient_discovery_date(details: &str) -> Option<String> {
    let value = details
        .split(", ")
        .find_map(|part| part.strip_prefix("disc. "))?;
    let mut parts = value.split('/');
    let year: i32 = parts.next()?.trim().parse().ok()?;
    let month: u32 = parts.next()?.trim().parse().ok()?;
    let day: u32 = parts.next()?.trim().parse().ok()?;
    parse_iso_jd(&format!("{year:04}-{month:02}-{day:02}"))?;
    Some(format!("{year:04}-{month:02}-{day:02}"))
}

fn transient_near_capture(discovered: Option<&str>, capture_jd: Option<f64>) -> bool {
    let (Some(discovered), Some(capture_jd)) = (discovered, capture_jd) else {
        return true;
    };
    let Some(discovered_jd) = parse_iso_jd(discovered) else {
        return true;
    };
    discovered_jd >= capture_jd - 365.0 && discovered_jd <= capture_jd + 30.0
}

fn overlay_layer_name(kind: &str) -> &'static str {
    match kind {
        "star" | "double-star" => "named_stars",
        "transient" => "transients",
        "comet" | "asteroid" => "minor_bodies",
        _ => "deep_sky",
    }
}

fn projected_outlines(
    catalog: &ObjectCatalog,
    canonical_id: &str,
    wcs: &Wcs,
) -> Vec<ObjectOutlineResponse> {
    let Ok(geometries) = catalog.geometries(canonical_id) else {
        return Vec::new();
    };
    project_outline_geometries(geometries, wcs)
}

fn project_outline_geometries(
    geometries: Vec<ObjectGeometry>,
    wcs: &Wcs,
) -> Vec<ObjectOutlineResponse> {
    geometries
        .into_iter()
        .filter_map(|geometry| {
            let GeometryData::OutlineSet { level, contours } = geometry.data else {
                return None;
            };
            let contours = contours
                .into_iter()
                .filter_map(|contour| {
                    let points = contour
                        .vertices
                        .into_iter()
                        .map(|(ra, dec)| wcs.world_to_pixel(ra, dec).map(|(x, y)| [x, y]))
                        .collect::<Option<Vec<_>>>()?;
                    let minimum_points = if contour.closed { 3 } else { 2 };
                    (points.len() >= minimum_points).then_some(ObjectContourResponse {
                        closed: contour.closed,
                        points,
                    })
                })
                .collect::<Vec<_>>();
            (!contours.is_empty()).then_some(ObjectOutlineResponse {
                geometry_id: geometry.id,
                source_record_id: geometry.source_record_id,
                role: geometry_role_name(geometry.role).into(),
                quality: geometry_quality_name(geometry.quality).into(),
                level,
                contours,
            })
        })
        .collect()
}

fn geometry_role_name(role: GeometryRole) -> &'static str {
    match role {
        GeometryRole::CatalogExtent => "catalog-extent",
        GeometryRole::PreferredRender => "preferred-render",
        GeometryRole::FallbackExtent => "fallback-extent",
        GeometryRole::BrightnessLevel => "brightness-level",
        GeometryRole::Component => "component",
    }
}

fn geometry_quality_name(quality: GeometryQuality) -> &'static str {
    match quality {
        GeometryQuality::Catalog => "catalog",
        GeometryQuality::Curated => "curated",
        GeometryQuality::Estimated => "estimated",
        GeometryQuality::Derived => "derived",
    }
}

#[unsafe(no_mangle)]
/// # Safety
/// `value` must be null or a string returned by this library that has not
/// already been freed.
pub unsafe extern "C" fn seiza_string_free(value: *mut c_char) {
    if !value.is_null() {
        unsafe { drop(CString::from_raw(value)) };
    }
}

fn is_fits_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("fits")
                || extension.eq_ignore_ascii_case("fit")
                || extension.eq_ignore_ascii_case("fts")
        })
}

fn render_path(
    path: &Path,
    params: &StretchParams,
    max_dimension: u32,
    rgb_stretch_mode: RgbStretchMode,
) -> Result<SeizaRenderedImage, String> {
    if is_fits_path(path) {
        let fits = FitsImage::open(path).map_err(|error| error.to_string())?;
        render_fits(fits, params, max_dimension, rgb_stretch_mode)
    } else {
        let image = image::open(path)
            .map_err(|error| format!("failed to open {}: {error}", path.display()))?;
        render_raster(image, raster_format(path), max_dimension)
    }
}

fn render_path16(
    path: &Path,
    params: &StretchParams,
    max_dimension: u32,
    rgb_stretch_mode: RgbStretchMode,
) -> Result<SeizaRenderedImage16, String> {
    if is_fits_path(path) {
        let fits = FitsImage::open(path).map_err(|error| error.to_string())?;
        render_fits16(fits, params, max_dimension, rgb_stretch_mode)
    } else {
        let image = image::open(path)
            .map_err(|error| format!("failed to open {}: {error}", path.display()))?;
        render_raster16(image, raster_format(path), max_dimension)
    }
}

fn render_fits(
    fits: FitsImage,
    params: &StretchParams,
    max_dimension: u32,
    rgb_stretch_mode: RgbStretchMode,
) -> Result<SeizaRenderedImage, String> {
    let source_width = fits.width;
    let source_height = fits.height;
    let statistics = fits.statistics();
    let color_kind = if fits.planes == 3 {
        "planar-rgb"
    } else if fits.bayer_pattern().is_some() {
        "bayer"
    } else {
        "mono"
    };

    let rgb = fits.debayer().or_else(|| fits.rgb_planes());
    let input_histogram = if let Some(rgb) = &rgb {
        input_histogram_u16_json(&rgb.data, 3, false)
    } else {
        input_histogram_u16_json(&fits.to_u16(), 1, true)
    };
    let rgba = if let Some(rgb) = rgb {
        stretch_rgb(&rgb, params, rgb_stretch_mode)
    } else {
        let gray = fits.stretch_to_u8(params);
        gray.into_iter()
            .flat_map(|value| [value, value, value, 255])
            .collect()
    };
    let display_histogram = display_histogram_json(&rgba);
    let (width, height, rgba) = downsample_rgba(
        source_width,
        source_height,
        rgba,
        usize::try_from(max_dimension).unwrap_or(usize::MAX),
    );

    let mut headers = Map::new();
    for (key, value) in &fits.headers {
        headers.insert(key.clone(), header_json(value));
    }
    let metadata = json!({
        "width": source_width,
        "height": source_height,
        "planes": fits.planes,
        "format": "FITS",
        "colorKind": color_kind,
        "rgbStretchMode": matches!(color_kind, "planar-rgb" | "bayer")
            .then(|| rgb_stretch_mode.name()),
        "statistics": statistics_json(&statistics),
        "inputHistogram": input_histogram,
        "displayHistogram": display_histogram,
        "headers": headers,
    });
    let metadata_json = CString::new(metadata.to_string())
        .map_err(|_| "metadata JSON contains a null byte".to_string())?;
    Ok(SeizaRenderedImage {
        width: u32::try_from(width).map_err(|_| "rendered width is too large")?,
        height: u32::try_from(height).map_err(|_| "rendered height is too large")?,
        rgba,
        bgra: OnceLock::new(),
        metadata_json,
    })
}

fn render_fits16(
    fits: FitsImage,
    params: &StretchParams,
    max_dimension: u32,
    rgb_stretch_mode: RgbStretchMode,
) -> Result<SeizaRenderedImage16, String> {
    let source_width = fits.width;
    let source_height = fits.height;
    let statistics = fits.statistics();
    let color_kind = if fits.planes == 3 {
        "planar-rgb"
    } else if fits.bayer_pattern().is_some() {
        "bayer"
    } else {
        "mono"
    };

    let rgb = fits.debayer().or_else(|| fits.rgb_planes());
    let input_histogram = if let Some(rgb) = &rgb {
        input_histogram_u16_json(&rgb.data, 3, false)
    } else {
        input_histogram_u16_json(&fits.to_u16(), 1, true)
    };
    let rgba = if let Some(rgb) = rgb {
        stretch_rgb16(&rgb, params, rgb_stretch_mode)
    } else {
        fits.stretch_to_u16(params)
            .into_iter()
            .flat_map(|value| [value, value, value, u16::MAX])
            .collect()
    };
    let display_histogram = display_histogram_u16_json(&rgba);
    let (width, height, rgba) = downsample_rgba(
        source_width,
        source_height,
        rgba,
        usize::try_from(max_dimension).unwrap_or(usize::MAX),
    );

    let mut headers = Map::new();
    for (key, value) in &fits.headers {
        headers.insert(key.clone(), header_json(value));
    }
    let metadata = json!({
        "width": source_width,
        "height": source_height,
        "planes": fits.planes,
        "format": "FITS",
        "colorKind": color_kind,
        "bitsPerComponent": 16,
        "rgbStretchMode": matches!(color_kind, "planar-rgb" | "bayer")
            .then(|| rgb_stretch_mode.name()),
        "statistics": statistics_json(&statistics),
        "inputHistogram": input_histogram,
        "displayHistogram": display_histogram,
        "headers": headers,
    });
    let metadata_json = CString::new(metadata.to_string())
        .map_err(|_| "metadata JSON contains a null byte".to_string())?;
    Ok(SeizaRenderedImage16 {
        width: u32::try_from(width).map_err(|_| "rendered width is too large")?,
        height: u32::try_from(height).map_err(|_| "rendered height is too large")?,
        rgba,
        metadata_json,
    })
}

/// Render a FITS image with a parameterized `seiza-stretch` config. The stretch
/// math lives entirely in `seiza-stretch`; this only marshals FITS pixels into
/// the interleaved `f32` the pipeline expects and assembles the RGBA result and
/// metadata, matching [`render_fits`]'s output shape.
#[cfg(test)]
fn render_fits_with_config(
    fits: FitsImage,
    config: &StretchConfig,
    max_dimension: u32,
) -> Result<SeizaRenderedImage, String> {
    render_fits_with_stack(fits, &StretchStack::single(config.clone()), max_dimension)
}

#[cfg(test)]
fn render_fits_with_stack(
    fits: FitsImage,
    stack: &StretchStack,
    max_dimension: u32,
) -> Result<SeizaRenderedImage, String> {
    render_fits_with_pipeline(fits, stack, None, None, max_dimension, false)
}

fn render_fits_with_pipeline(
    fits: FitsImage,
    stack: &StretchStack,
    background: Option<&BackgroundRenderRequest>,
    deconvolution: Option<&DeconvolutionRenderRequest>,
    max_dimension: u32,
    interactive_preview: bool,
) -> Result<SeizaRenderedImage, String> {
    let prepared = prepare_fits_render(fits, background, max_dimension, interactive_preview)?;
    render_prepared_fits(&prepared, stack, deconvolution, max_dimension, false)
}

fn render_fits_with_pipeline16(
    fits: FitsImage,
    stack: &StretchStack,
    background: Option<&BackgroundRenderRequest>,
    deconvolution: Option<&DeconvolutionRenderRequest>,
    max_dimension: u32,
    interactive_preview: bool,
) -> Result<SeizaRenderedImage16, String> {
    let prepared = prepare_fits_render(fits, background, max_dimension, interactive_preview)?;
    render_prepared_fits16(&prepared, stack, deconvolution, max_dimension, false)
}

fn prepare_fits_render(
    fits: FitsImage,
    background: Option<&BackgroundRenderRequest>,
    max_dimension: u32,
    interactive_preview: bool,
) -> Result<PreparedFitsRender, String> {
    let source_width = fits.width;
    let source_height = fits.height;
    let statistics = fits.statistics();
    let color_kind = if fits.planes == 3 {
        "planar-rgb"
    } else if fits.bayer_pattern().is_some() {
        "bayer"
    } else {
        "mono"
    };

    let rgb = fits.debayer().or_else(|| fits.rgb_planes());
    let original_input_histogram = if let Some(rgb) = &rgb {
        input_histogram_u16_json(&rgb.data, 3, false)
    } else {
        input_histogram_u16_json(&fits.to_u16(), 1, true)
    };

    // The pipeline consumes interleaved f32 samples normalized to [0, 1], the
    // same convention as `FitsImage::to_luma_f32`.
    let (data, channels): (Vec<f32>, usize) = match &rgb {
        Some(rgb) => (
            rgb.data
                .iter()
                .map(|&value| f32::from(value) / f32::from(u16::MAX))
                .collect(),
            3,
        ),
        None => (fits.to_luma_f32(), 1),
    };
    let (render_width, render_height, mut data) = if interactive_preview {
        downsample_interleaved_f32(
            source_width,
            source_height,
            data,
            channels,
            usize::try_from(max_dimension).unwrap_or(usize::MAX),
        )
    } else {
        (source_width, source_height, data)
    };
    let (input_histogram, background_metadata) = if let Some(background) = background {
        let fit = fit_background_masked(
            &data,
            render_width,
            render_height,
            channels,
            None,
            &background.config,
        )
        .map_err(|error| format!("failed to fit image background: {error}"))?;
        fit.correct_in_place(&mut data, background.mode)
            .map_err(|error| format!("failed to correct image background: {error}"))?;
        let metadata = json!({
            "mode": background.mode,
            "diagnostics": &fit.diagnostics,
            "reference": &fit.reference,
        });
        (input_histogram_f32_json(&data, channels), Some(metadata))
    } else {
        (original_input_histogram, None)
    };
    let mut headers = Map::new();
    for (key, value) in &fits.headers {
        headers.insert(key.clone(), header_json(value));
    }

    Ok(PreparedFitsRender {
        source_width,
        source_height,
        planes: fits.planes,
        color_kind,
        render_width,
        render_height,
        channels,
        data,
        statistics: statistics_json(&statistics),
        input_histogram,
        background_metadata,
        headers,
        interactive_preview,
    })
}

fn prepare_stretch_input<'a>(
    prepared: &'a PreparedFitsRender,
    deconvolution: Option<&DeconvolutionRenderRequest>,
) -> Result<PreparedStretchInput<'a>, String> {
    let Some(request) = deconvolution else {
        return Ok(PreparedStretchInput {
            data: Cow::Borrowed(&prepared.data),
            input_histogram: prepared.input_histogram.clone(),
            deconvolution_metadata: None,
        });
    };

    let scale = (prepared.render_width as f32 / prepared.source_width as f32)
        .min(prepared.render_height as f32 / prepared.source_height as f32);
    let effective_psf_fwhm_pixels = (request.psf_fwhm_pixels * scale).max(0.25);
    let config = DeconvolutionConfig {
        psf_fwhm_pixels: effective_psf_fwhm_pixels,
        iterations: request.iterations,
        amount: request.amount,
        noise_fraction: request.noise_fraction,
        max_correction: request.max_correction,
    };
    let restored = deconvolve(
        &prepared.data,
        prepared.render_width,
        prepared.render_height,
        prepared.channels,
        &config,
    )
    .map_err(|error| format!("failed to deconvolve image: {error}"))?;
    let channels = restored
        .channels
        .iter()
        .map(|channel| {
            json!({
                "inputFlux": channel.input_flux,
                "outputFlux": channel.output_flux,
                "inputPeak": channel.input_peak,
                "outputPeak": channel.output_peak,
            })
        })
        .collect::<Vec<_>>();
    let input_histogram = input_histogram_f32_json(&restored.data, prepared.channels);
    Ok(PreparedStretchInput {
        data: Cow::Owned(restored.data),
        input_histogram,
        deconvolution_metadata: Some(json!({
            "psfFwhmPixels": request.psf_fwhm_pixels,
            "effectivePsfFwhmPixels": effective_psf_fwhm_pixels,
            "iterations": request.iterations,
            "amount": request.amount,
            "noiseFraction": request.noise_fraction,
            "maxCorrection": request.max_correction,
            "channels": channels,
        })),
    })
}

fn render_prepared_fits(
    prepared: &PreparedFitsRender,
    stack: &StretchStack,
    deconvolution: Option<&DeconvolutionRenderRequest>,
    max_dimension: u32,
    interactive_preview_cache_hit: bool,
) -> Result<SeizaRenderedImage, String> {
    let PreparedStretchInput {
        data,
        input_histogram,
        deconvolution_metadata,
    } = prepare_stretch_input(prepared, deconvolution)?;
    let stretched = stack
        .apply_u8(&data, prepared.channels)
        .map_err(|error| error.to_string())?
        .data;
    let rgba: Vec<u8> = if prepared.channels == 3 {
        stretched
            .chunks_exact(3)
            .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], 255])
            .collect()
    } else {
        stretched
            .into_iter()
            .flat_map(|value| [value, value, value, 255])
            .collect()
    };

    let display_histogram = display_histogram_json(&rgba);
    let (width, height, rgba) = downsample_rgba(
        prepared.render_width,
        prepared.render_height,
        rgba,
        usize::try_from(max_dimension).unwrap_or(usize::MAX),
    );
    let metadata = json!({
        "width": prepared.source_width,
        "height": prepared.source_height,
        "planes": prepared.planes,
        "format": "FITS",
        "colorKind": prepared.color_kind,
        "stretchStages": stack.len(),
        "interactivePreview": prepared.interactive_preview,
        "interactivePreviewCacheHit": interactive_preview_cache_hit,
        "backgroundProcessing": prepared.background_metadata,
        "deconvolutionProcessing": deconvolution_metadata,
        "statistics": prepared.statistics,
        "inputHistogram": input_histogram,
        "displayHistogram": display_histogram,
        "headers": prepared.headers,
    });
    let metadata_json = CString::new(metadata.to_string())
        .map_err(|_| "metadata JSON contains a null byte".to_string())?;
    Ok(SeizaRenderedImage {
        width: u32::try_from(width).map_err(|_| "rendered width is too large")?,
        height: u32::try_from(height).map_err(|_| "rendered height is too large")?,
        rgba,
        bgra: OnceLock::new(),
        metadata_json,
    })
}

fn render_prepared_fits16(
    prepared: &PreparedFitsRender,
    stack: &StretchStack,
    deconvolution: Option<&DeconvolutionRenderRequest>,
    max_dimension: u32,
    interactive_preview_cache_hit: bool,
) -> Result<SeizaRenderedImage16, String> {
    let PreparedStretchInput {
        data,
        input_histogram,
        deconvolution_metadata,
    } = prepare_stretch_input(prepared, deconvolution)?;
    let stretched = stack
        .apply_u16(&data, prepared.channels)
        .map_err(|error| error.to_string())?
        .data;
    let rgba: Vec<u16> = if prepared.channels == 3 {
        stretched
            .chunks_exact(3)
            .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], u16::MAX])
            .collect()
    } else {
        stretched
            .into_iter()
            .flat_map(|value| [value, value, value, u16::MAX])
            .collect()
    };

    let display_histogram = display_histogram_u16_json(&rgba);
    let (width, height, rgba) = downsample_rgba(
        prepared.render_width,
        prepared.render_height,
        rgba,
        usize::try_from(max_dimension).unwrap_or(usize::MAX),
    );
    let metadata = json!({
        "width": prepared.source_width,
        "height": prepared.source_height,
        "planes": prepared.planes,
        "format": "FITS",
        "colorKind": prepared.color_kind,
        "bitsPerComponent": 16,
        "stretchStages": stack.len(),
        "interactivePreview": prepared.interactive_preview,
        "interactivePreviewCacheHit": interactive_preview_cache_hit,
        "backgroundProcessing": prepared.background_metadata,
        "deconvolutionProcessing": deconvolution_metadata,
        "statistics": prepared.statistics,
        "inputHistogram": input_histogram,
        "displayHistogram": display_histogram,
        "headers": prepared.headers,
    });
    let metadata_json = CString::new(metadata.to_string())
        .map_err(|_| "metadata JSON contains a null byte".to_string())?;
    Ok(SeizaRenderedImage16 {
        width: u32::try_from(width).map_err(|_| "rendered width is too large")?,
        height: u32::try_from(height).map_err(|_| "rendered height is too large")?,
        rgba,
        metadata_json,
    })
}

fn render_cached_interactive_preview(
    path: &Path,
    stack: &StretchStack,
    background: Option<&BackgroundRenderRequest>,
    deconvolution: Option<&DeconvolutionRenderRequest>,
    max_dimension: u32,
) -> Result<SeizaRenderedImage, String> {
    let key = interactive_preview_cache_key(path, background, max_dimension)?;
    let cache = INTERACTIVE_PREVIEW_CACHE
        .get_or_init(|| Mutex::new(VecDeque::with_capacity(INTERACTIVE_PREVIEW_CACHE_CAPACITY)));

    if let Some(prepared) = cached_interactive_preview(cache, &key)? {
        return render_prepared_fits(&prepared, stack, deconvolution, max_dimension, true);
    }

    let fits = FitsImage::open(path)
        .map_err(|error| format!("failed to open {}: {error}", path.display()))?;
    let prepared = Arc::new(prepare_fits_render(fits, background, max_dimension, true)?);
    let prepared = store_interactive_preview(cache, key, prepared)?;
    render_prepared_fits(&prepared, stack, deconvolution, max_dimension, false)
}

fn render_cached_interactive_preview16(
    path: &Path,
    stack: &StretchStack,
    background: Option<&BackgroundRenderRequest>,
    deconvolution: Option<&DeconvolutionRenderRequest>,
    max_dimension: u32,
) -> Result<SeizaRenderedImage16, String> {
    let key = interactive_preview_cache_key(path, background, max_dimension)?;
    let cache = INTERACTIVE_PREVIEW_CACHE
        .get_or_init(|| Mutex::new(VecDeque::with_capacity(INTERACTIVE_PREVIEW_CACHE_CAPACITY)));

    if let Some(prepared) = cached_interactive_preview(cache, &key)? {
        return render_prepared_fits16(&prepared, stack, deconvolution, max_dimension, true);
    }

    let fits = FitsImage::open(path)
        .map_err(|error| format!("failed to open {}: {error}", path.display()))?;
    let prepared = Arc::new(prepare_fits_render(fits, background, max_dimension, true)?);
    let prepared = store_interactive_preview(cache, key, prepared)?;
    render_prepared_fits16(&prepared, stack, deconvolution, max_dimension, false)
}

fn interactive_preview_cache_key(
    path: &Path,
    background: Option<&BackgroundRenderRequest>,
    max_dimension: u32,
) -> Result<InteractivePreviewCacheKey, String> {
    let metadata = std::fs::metadata(path)
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?;
    Ok(InteractivePreviewCacheKey {
        path: path.canonicalize().unwrap_or_else(|_| path.to_path_buf()),
        file_size: metadata.len(),
        modified: metadata.modified().ok(),
        max_dimension,
        background: background
            .map(serde_json::to_string)
            .transpose()
            .map_err(|error| format!("failed to identify background processing: {error}"))?,
    })
}

fn cached_interactive_preview(
    cache: &InteractivePreviewCache,
    key: &InteractivePreviewCacheKey,
) -> Result<Option<Arc<PreparedFitsRender>>, String> {
    let mut entries = cache
        .lock()
        .map_err(|_| "interactive preview cache lock is poisoned".to_string())?;
    let Some(index) = entries.iter().position(|(candidate, _)| candidate == key) else {
        return Ok(None);
    };
    let entry = entries
        .remove(index)
        .ok_or_else(|| "interactive preview cache entry disappeared".to_string())?;
    let prepared = Arc::clone(&entry.1);
    entries.push_front(entry);
    Ok(Some(prepared))
}

fn store_interactive_preview(
    cache: &InteractivePreviewCache,
    key: InteractivePreviewCacheKey,
    prepared: Arc<PreparedFitsRender>,
) -> Result<Arc<PreparedFitsRender>, String> {
    let mut entries = cache
        .lock()
        .map_err(|_| "interactive preview cache lock is poisoned".to_string())?;
    if let Some(index) = entries.iter().position(|(candidate, _)| candidate == &key) {
        let existing = entries
            .remove(index)
            .ok_or_else(|| "interactive preview cache entry disappeared".to_string())?;
        let prepared = Arc::clone(&existing.1);
        entries.push_front(existing);
        return Ok(prepared);
    }
    entries.push_front((key, Arc::clone(&prepared)));
    entries.truncate(INTERACTIVE_PREVIEW_CACHE_CAPACITY);
    Ok(prepared)
}

fn render_raster(
    image: DynamicImage,
    format: &'static str,
    max_dimension: u32,
) -> Result<SeizaRenderedImage, String> {
    let source_width = image.width();
    let source_height = image.height();
    let (planes, color_kind) = raster_encoding(&image);
    let input_histogram = raster_input_histogram_json(&image);
    let statistics = raster_statistics_json(image.to_luma8().as_raw());
    let rgba = image.to_rgba8().into_raw();
    let display_histogram = display_histogram_json(&rgba);
    let (width, height, rgba) = downsample_rgba(
        usize::try_from(source_width).map_err(|_| "image width is too large")?,
        usize::try_from(source_height).map_err(|_| "image height is too large")?,
        rgba,
        usize::try_from(max_dimension).unwrap_or(usize::MAX),
    );
    let metadata = json!({
        "width": source_width,
        "height": source_height,
        "planes": planes,
        "format": format,
        "colorKind": color_kind,
        "statistics": statistics,
        "inputHistogram": input_histogram,
        "displayHistogram": display_histogram,
        "headers": Map::<String, Value>::new(),
    });
    let metadata_json = CString::new(metadata.to_string())
        .map_err(|_| "metadata JSON contains a null byte".to_string())?;
    Ok(SeizaRenderedImage {
        width: u32::try_from(width).map_err(|_| "rendered width is too large")?,
        height: u32::try_from(height).map_err(|_| "rendered height is too large")?,
        rgba,
        bgra: OnceLock::new(),
        metadata_json,
    })
}

fn render_raster16(
    image: DynamicImage,
    format: &'static str,
    max_dimension: u32,
) -> Result<SeizaRenderedImage16, String> {
    let source_width = image.width();
    let source_height = image.height();
    let (planes, color_kind) = raster_encoding(&image);
    let input_histogram = raster_input_histogram_json(&image);
    let luma = image.to_luma16();
    let statistics = statistics_json(&seiza_fits::statistics_u16(luma.as_raw()));
    let rgba = image.to_rgba16().into_raw();
    let display_histogram = display_histogram_u16_json(&rgba);
    let (width, height, rgba) = downsample_rgba(
        usize::try_from(source_width).map_err(|_| "image width is too large")?,
        usize::try_from(source_height).map_err(|_| "image height is too large")?,
        rgba,
        usize::try_from(max_dimension).unwrap_or(usize::MAX),
    );
    let metadata = json!({
        "width": source_width,
        "height": source_height,
        "planes": planes,
        "format": format,
        "colorKind": color_kind,
        "bitsPerComponent": 16,
        "statistics": statistics,
        "inputHistogram": input_histogram,
        "displayHistogram": display_histogram,
        "headers": Map::<String, Value>::new(),
    });
    let metadata_json = CString::new(metadata.to_string())
        .map_err(|_| "metadata JSON contains a null byte".to_string())?;
    Ok(SeizaRenderedImage16 {
        width: u32::try_from(width).map_err(|_| "rendered width is too large")?,
        height: u32::try_from(height).map_err(|_| "rendered height is too large")?,
        rgba,
        metadata_json,
    })
}

fn raster_format(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("jpg" | "jpeg" | "jfif") => "JPEG",
        Some("png") => "PNG",
        Some("tif" | "tiff") => "TIFF",
        _ => "Raster",
    }
}

fn raster_encoding(image: &DynamicImage) -> (usize, &'static str) {
    match image {
        DynamicImage::ImageLuma8(_) => (1, "mono-8"),
        DynamicImage::ImageLumaA8(_) => (2, "mono-alpha-8"),
        DynamicImage::ImageRgb8(_) => (3, "rgb-8"),
        DynamicImage::ImageRgba8(_) => (4, "rgba-8"),
        DynamicImage::ImageLuma16(_) => (1, "mono-16"),
        DynamicImage::ImageLumaA16(_) => (2, "mono-alpha-16"),
        DynamicImage::ImageRgb16(_) => (3, "rgb-16"),
        DynamicImage::ImageRgba16(_) => (4, "rgba-16"),
        DynamicImage::ImageRgb32F(_) => (3, "rgb-f32"),
        DynamicImage::ImageRgba32F(_) => (4, "rgba-f32"),
        _ => (usize::from(image.color().channel_count()), "raster"),
    }
}

fn is_converted_8bit_color(image: &DynamicImage) -> bool {
    matches!(
        image,
        DynamicImage::ImageLumaA8(_) | DynamicImage::ImageRgb8(_) | DynamicImage::ImageRgba8(_)
    )
}

fn raster_statistics_json(values: &[u8]) -> Value {
    let mut histogram = [0_u64; 256];
    let mut sum = 0_u64;
    for &value in values {
        histogram[usize::from(value)] += 1;
        sum += u64::from(value);
    }
    let count = values.len() as u64;
    let quantile = |histogram: &[u64; 256], rank: u64| -> u8 {
        let mut seen = 0_u64;
        for (value, &frequency) in histogram.iter().enumerate() {
            seen += frequency;
            if seen > rank {
                return value as u8;
            }
        }
        0
    };
    let minimum = histogram
        .iter()
        .position(|&frequency| frequency > 0)
        .unwrap_or(0) as u8;
    let maximum = histogram
        .iter()
        .rposition(|&frequency| frequency > 0)
        .unwrap_or(0) as u8;
    let median = quantile(&histogram, count.saturating_sub(1) / 2);
    let mut deviation_histogram = [0_u64; 256];
    for (value, &frequency) in histogram.iter().enumerate() {
        deviation_histogram[value.abs_diff(usize::from(median))] += frequency;
    }
    let mad = quantile(&deviation_histogram, count.saturating_sub(1) / 2);
    json!({
        "minimum": minimum,
        "maximum": maximum,
        "mean": if count == 0 { 0.0 } else { sum as f64 / count as f64 },
        "median": median,
        "mad": mad,
    })
}

fn display_histogram_json(rgba: &[u8]) -> Value {
    let mut red = [0_u64; 256];
    let mut green = [0_u64; 256];
    let mut blue = [0_u64; 256];
    for pixel in rgba.chunks_exact(4) {
        red[usize::from(pixel[0])] += 1;
        green[usize::from(pixel[1])] += 1;
        blue[usize::from(pixel[2])] += 1;
    }
    json!({
        "red": red.as_slice(),
        "green": green.as_slice(),
        "blue": blue.as_slice(),
        "lowerBound": 0.0,
        "upperBound": 255.0,
    })
}

fn display_histogram_u16_json(rgba: &[u16]) -> Value {
    let mut red = [0_u64; 256];
    let mut green = [0_u64; 256];
    let mut blue = [0_u64; 256];
    for pixel in rgba.chunks_exact(4) {
        red[usize::from(pixel[0] >> 8)] += 1;
        green[usize::from(pixel[1] >> 8)] += 1;
        blue[usize::from(pixel[2] >> 8)] += 1;
    }
    histogram_json(&red, &green, &blue, 0.0, f64::from(u16::MAX))
}

fn raster_input_histogram_json(image: &DynamicImage) -> Value {
    match image {
        DynamicImage::ImageLuma8(image) => input_histogram_u8_json(image.as_raw(), 1, true),
        DynamicImage::ImageLumaA8(image) => input_histogram_u8_json(image.as_raw(), 2, true),
        DynamicImage::ImageRgb8(image) => input_histogram_u8_json(image.as_raw(), 3, false),
        DynamicImage::ImageRgba8(image) => input_histogram_u8_json(image.as_raw(), 4, false),
        DynamicImage::ImageLuma16(image) => input_histogram_u16_json(image.as_raw(), 1, true),
        DynamicImage::ImageLumaA16(image) => input_histogram_u16_json(image.as_raw(), 2, true),
        DynamicImage::ImageRgb16(image) => input_histogram_u16_json(image.as_raw(), 3, false),
        DynamicImage::ImageRgba16(image) => input_histogram_u16_json(image.as_raw(), 4, false),
        DynamicImage::ImageRgb32F(image) => input_histogram_f32_json(image.as_raw(), 3),
        DynamicImage::ImageRgba32F(image) => input_histogram_f32_json(image.as_raw(), 4),
        _ => display_histogram_json(image.to_rgba8().as_raw()),
    }
}

fn input_histogram_u8_json(samples: &[u8], stride: usize, monochrome: bool) -> Value {
    let mut red = [0_u64; 256];
    let mut green = [0_u64; 256];
    let mut blue = [0_u64; 256];
    for pixel in samples.chunks_exact(stride) {
        let red_value = usize::from(pixel[0]);
        let green_value = if monochrome {
            red_value
        } else {
            usize::from(pixel[1])
        };
        let blue_value = if monochrome {
            red_value
        } else {
            usize::from(pixel[2])
        };
        red[red_value] += 1;
        green[green_value] += 1;
        blue[blue_value] += 1;
    }
    histogram_json(&red, &green, &blue, 0.0, 255.0)
}

fn input_histogram_u16_json(samples: &[u16], stride: usize, monochrome: bool) -> Value {
    let mut red = [0_u64; 256];
    let mut green = [0_u64; 256];
    let mut blue = [0_u64; 256];
    for pixel in samples.chunks_exact(stride) {
        let bin = |value: u16| usize::from(value >> 8);
        let red_value = bin(pixel[0]);
        let green_value = if monochrome { red_value } else { bin(pixel[1]) };
        let blue_value = if monochrome { red_value } else { bin(pixel[2]) };
        red[red_value] += 1;
        green[green_value] += 1;
        blue[blue_value] += 1;
    }
    histogram_json(&red, &green, &blue, 0.0, f64::from(u16::MAX))
}

fn input_histogram_f32_json(samples: &[f32], stride: usize) -> Value {
    let mut red = [0_u64; 256];
    let mut green = [0_u64; 256];
    let mut blue = [0_u64; 256];
    let bin = |value: f32| {
        if value.is_finite() {
            (value.clamp(0.0, 1.0) * 255.0).round() as usize
        } else {
            0
        }
    };
    for pixel in samples.chunks_exact(stride) {
        let red_value = bin(pixel[0]);
        let green_value = if stride == 1 {
            red_value
        } else {
            bin(pixel[1])
        };
        let blue_value = if stride == 1 {
            red_value
        } else {
            bin(pixel[2])
        };
        red[red_value] += 1;
        green[green_value] += 1;
        blue[blue_value] += 1;
    }
    histogram_json(&red, &green, &blue, 0.0, 1.0)
}

fn histogram_json(
    red: &[u64; 256],
    green: &[u64; 256],
    blue: &[u64; 256],
    lower_bound: f64,
    upper_bound: f64,
) -> Value {
    json!({
        "red": red.as_slice(),
        "green": green.as_slice(),
        "blue": blue.as_slice(),
        "lowerBound": lower_bound,
        "upperBound": upper_bound,
    })
}

fn stretch_rgb(rgb: &RgbImage16, params: &StretchParams, mode: RgbStretchMode) -> Vec<u8> {
    let stretched = match mode {
        RgbStretchMode::Auto => {
            let channels = rgb_channels(rgb);
            let channels = channels.map(|channel| {
                let statistics = seiza_fits::statistics_u16(&channel);
                seiza_fits::stretch_u16_to_u8(&channel, &statistics, params)
            });
            (0..rgb.width * rgb.height)
                .flat_map(|index| [channels[0][index], channels[1][index], channels[2][index]])
                .collect()
        }
        RgbStretchMode::LinkedAuto => {
            let statistics = linked_rgb_statistics(rgb);
            seiza_fits::stretch_u16_to_u8(&rgb.data, &statistics, params)
        }
        RgbStretchMode::Linear => rgb.data.iter().copied().map(linear_u16_to_u8).collect(),
    };
    stretched
        .chunks_exact(3)
        .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], 255])
        .collect()
}

fn stretch_rgb16(rgb: &RgbImage16, params: &StretchParams, mode: RgbStretchMode) -> Vec<u16> {
    let stretched = match mode {
        RgbStretchMode::Auto => {
            let channels = rgb_channels(rgb);
            let channels = channels.map(|channel| {
                let statistics = seiza_fits::statistics_u16(&channel);
                seiza_fits::stretch_u16_to_u16(&channel, &statistics, params)
            });
            (0..rgb.width * rgb.height)
                .flat_map(|index| [channels[0][index], channels[1][index], channels[2][index]])
                .collect()
        }
        RgbStretchMode::LinkedAuto => {
            let statistics = linked_rgb_statistics(rgb);
            seiza_fits::stretch_u16_to_u16(&rgb.data, &statistics, params)
        }
        RgbStretchMode::Linear => rgb.data.clone(),
    };
    stretched
        .chunks_exact(3)
        .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], u16::MAX])
        .collect()
}

fn rgb_channels(rgb: &RgbImage16) -> [Vec<u16>; 3] {
    let mut channels = [Vec::new(), Vec::new(), Vec::new()];
    for pixel in rgb.data.chunks_exact(3) {
        channels[0].push(pixel[0]);
        channels[1].push(pixel[1]);
        channels[2].push(pixel[2]);
    }
    channels
}

fn linked_rgb_statistics(rgb: &RgbImage16) -> Statistics {
    let statistics = rgb_channels(rgb).map(|channel| seiza_fits::statistics_u16(&channel));
    Statistics {
        min: statistics.iter().map(|value| value.min).min().unwrap_or(0),
        max: statistics.iter().map(|value| value.max).max().unwrap_or(0),
        mean: statistics.iter().map(|value| value.mean).sum::<f64>() / 3.0,
        std_dev: statistics.iter().map(|value| value.std_dev).sum::<f64>() / 3.0,
        median: (statistics
            .iter()
            .map(|value| f64::from(value.median))
            .sum::<f64>()
            / 3.0)
            .round() as u16,
        mad: statistics.iter().map(|value| value.mad).sum::<f64>() / 3.0,
        count: rgb.data.len(),
    }
}

fn linear_u16_to_u8(value: u16) -> u8 {
    ((u32::from(value) * 255 + 32_767) / 65_535) as u8
}

fn downsample_rgba<T: Copy>(
    width: usize,
    height: usize,
    rgba: Vec<T>,
    max_dimension: usize,
) -> (usize, usize, Vec<T>) {
    if max_dimension == 0 || width.max(height) <= max_dimension {
        return (width, height, rgba);
    }
    let scale = max_dimension as f64 / width.max(height) as f64;
    let output_width = ((width as f64 * scale).round() as usize).max(1);
    let output_height = ((height as f64 * scale).round() as usize).max(1);
    let mut output = Vec::with_capacity(output_width * output_height * 4);
    for y in 0..output_height {
        let source_y = y * height / output_height;
        for x in 0..output_width {
            let source_x = x * width / output_width;
            let offset = (source_y * width + source_x) * 4;
            output.extend_from_slice(&rgba[offset..offset + 4]);
        }
    }
    (output_width, output_height, output)
}

/// Bounds an interactive render before expensive processing. Bilinear sampling
/// keeps the preview representative without spending time on source-resolution
/// background fitting and stretch stages. Full and non-interactive renders do
/// not use this path.
fn downsample_interleaved_f32(
    width: usize,
    height: usize,
    pixels: Vec<f32>,
    channels: usize,
    max_dimension: usize,
) -> (usize, usize, Vec<f32>) {
    if max_dimension == 0 || width.max(height) <= max_dimension {
        return (width, height, pixels);
    }

    let scale = max_dimension as f64 / width.max(height) as f64;
    let output_width = ((width as f64 * scale).round() as usize).max(1);
    let output_height = ((height as f64 * scale).round() as usize).max(1);
    let mut output = vec![0.0; output_width * output_height * channels];
    let scale_x = width as f64 / output_width as f64;
    let scale_y = height as f64 / output_height as f64;

    for output_y in 0..output_height {
        let source_y =
            ((output_y as f64 + 0.5) * scale_y - 0.5).clamp(0.0, height.saturating_sub(1) as f64);
        let y0 = source_y.floor() as usize;
        let y1 = (y0 + 1).min(height - 1);
        let y_weight = (source_y - y0 as f64) as f32;

        for output_x in 0..output_width {
            let source_x = ((output_x as f64 + 0.5) * scale_x - 0.5)
                .clamp(0.0, width.saturating_sub(1) as f64);
            let x0 = source_x.floor() as usize;
            let x1 = (x0 + 1).min(width - 1);
            let x_weight = (source_x - x0 as f64) as f32;
            let output_start = (output_y * output_width + output_x) * channels;

            for channel in 0..channels {
                let top_left = pixels[(y0 * width + x0) * channels + channel];
                let top_right = pixels[(y0 * width + x1) * channels + channel];
                let bottom_left = pixels[(y1 * width + x0) * channels + channel];
                let bottom_right = pixels[(y1 * width + x1) * channels + channel];
                let top = top_left + (top_right - top_left) * x_weight;
                let bottom = bottom_left + (bottom_right - bottom_left) * x_weight;
                output[output_start + channel] = top + (bottom - top) * y_weight;
            }
        }
    }

    (output_width, output_height, output)
}

fn header_json(value: &HeaderValue) -> Value {
    match value {
        HeaderValue::Integer(value) => json!(value),
        HeaderValue::Float(value) if value.is_finite() => json!(value),
        HeaderValue::Float(value) => json!(value.to_string()),
        HeaderValue::String(value) => json!(value),
        HeaderValue::Logical(value) => json!(value),
        HeaderValue::Raw(value) => json!(value),
    }
}

fn statistics_json(statistics: &Statistics) -> Value {
    json!({
        "minimum": statistics.min,
        "maximum": statistics.max,
        "mean": statistics.mean,
        "median": statistics.median,
        "mad": statistics.mad,
    })
}

fn catalog_status(catalog_directory: Option<&Path>) -> CatalogStatusResponse {
    let directory = catalog_directory
        .map(Path::to_path_buf)
        .unwrap_or_else(seiza::data_paths::default_catalog_dir);
    let star_catalog = component_status(seiza::data_paths::star_data(catalog_directory));
    let blind_index = optional_component_status(seiza::data_paths::blind_index(catalog_directory));
    let objects = component_status(seiza::data_paths::objects(catalog_directory));
    let transients = component_status(seiza::data_paths::transients(catalog_directory));
    let minor_bodies = component_status(seiza::data_paths::minor_bodies(catalog_directory));
    CatalogStatusResponse {
        directory: directory.to_string_lossy().into_owned(),
        ready_for_solving: star_catalog.available && blind_index.available,
        ready_for_overlays: objects.available && transients.available && minor_bodies.available,
        star_catalog,
        blind_index,
        objects,
        transients,
        minor_bodies,
    }
}

fn component_status<E: std::fmt::Display>(result: Result<PathBuf, E>) -> CatalogComponentStatus {
    match result {
        Ok(path) => CatalogComponentStatus {
            available: true,
            path: Some(path.to_string_lossy().into_owned()),
        },
        Err(_) => CatalogComponentStatus {
            available: false,
            path: None,
        },
    }
}

fn optional_component_status<E: std::fmt::Display>(
    result: Result<Option<PathBuf>, E>,
) -> CatalogComponentStatus {
    match result {
        Ok(Some(path)) => CatalogComponentStatus {
            available: true,
            path: Some(path.to_string_lossy().into_owned()),
        },
        Ok(None) | Err(_) => CatalogComponentStatus {
            available: false,
            path: None,
        },
    }
}

fn run_catalog_setup(
    catalog_directory: Option<&Path>,
    preset: CatalogSetupPreset,
    reporter: CatalogSetupReporter,
) -> Result<(), String> {
    let output = catalog_directory
        .map(Path::to_path_buf)
        .unwrap_or_else(seiza::data_paths::default_catalog_dir);
    reporter.simple(
        "preparing",
        format!("Preparing catalog setup in {}", output.display()),
    );
    let selection = preset.selection()?;
    let manager = CatalogManager::builder()
        .policy(CachePolicy::ForceRefresh)
        .build()
        .map_err(|error| error.to_string())?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("failed to start the catalog download runtime: {error}"))?;
    let download_reporter = reporter;
    let bundle = runtime.block_on(async move {
        manager
            .ensure_with(&selection, move |event| {
                download_reporter.download_event(event, 0)
            })
            .await
    });
    let bundle = bundle.map_err(|error| error.to_string())?;
    let installed_count = AtomicUsize::new(0);
    let install_reporter = reporter;
    runtime
        .block_on(bundle.materialize_with(&output, move |event| {
            let files_completed = if matches!(&event, DownloadEvent::InstallComplete { .. }) {
                installed_count.fetch_add(1, Ordering::Relaxed) + 1
            } else {
                installed_count.load(Ordering::Relaxed)
            };
            install_reporter.download_event(event, files_completed);
        }))
        .map_err(|error| {
            format!(
                "failed to install catalogs in {}: {error}",
                output.display()
            )
        })?;
    reporter.report(CatalogSetupProgressResponse {
        phase: "complete",
        message: format!("Catalogs are ready in {}", output.display()),
        file_name: None,
        files_completed: reporter.files_total,
        files_total: reporter.files_total,
        bytes_completed: None,
        bytes_total: None,
        written_bytes: None,
    });
    Ok(())
}

fn background_config(config_json: *const c_char) -> Result<BackgroundConfig, String> {
    if config_json.is_null() {
        return Ok(BackgroundConfig::default());
    }
    let config_json = required_str(config_json, "background config JSON")?;
    if config_json.trim().is_empty() {
        return Ok(BackgroundConfig::default());
    }
    serde_json::from_str(&config_json)
        .map_err(|error| format!("invalid background config JSON: {error}"))
}

fn stack_options(options_json: *const c_char) -> Result<StackOptions, String> {
    let options = if options_json.is_null() {
        StackOptions::default()
    } else {
        let options_json = required_str(options_json, "stack options JSON")?;
        if options_json.trim().is_empty() {
            StackOptions::default()
        } else {
            serde_json::from_str(&options_json)
                .map_err(|error| format!("invalid stack options JSON: {error}"))?
        }
    };
    options.validate().map_err(|error| error.to_string())?;
    Ok(options)
}

unsafe fn linear_image_from_ffi(
    data: *const f32,
    length: usize,
    width: usize,
    height: usize,
    channels: usize,
    name: &str,
) -> Result<LinearImage, String> {
    if width == 0 || height == 0 || !matches!(channels, 1 | 3) {
        return Err(format!(
            "{name} dimensions must be non-zero and channels must be one or three"
        ));
    }
    let expected = width
        .checked_mul(height)
        .and_then(|value| value.checked_mul(channels))
        .ok_or_else(|| format!("{name} dimensions overflow"))?;
    if length != expected {
        return Err(format!("{name} has {length} floats; expected {expected}"));
    }
    let data = unsafe { required_f32_slice(data, length, name)? };
    LinearImage::new(width, height, channels, data.to_vec()).map_err(|error| error.to_string())
}

fn optional_positive_seconds(value: f64, name: &str) -> Result<Option<f64>, String> {
    if value == 0.0 {
        return Ok(None);
    }
    if !value.is_finite() || value <= 0.0 {
        return Err(format!("{name} must be zero or a positive finite number"));
    }
    Ok(Some(value))
}

fn validate_distinct_stack_paths(paths: &[PathBuf]) -> Result<(), String> {
    for (index, path) in paths.iter().enumerate() {
        if paths[..index]
            .iter()
            .any(|previous| paths_refer_to_same_file(path, previous))
        {
            return Err(format!("duplicate stack input path {}", path.display()));
        }
    }
    Ok(())
}

fn stack_disposition_response(
    source: Option<&Path>,
    disposition: FrameDisposition,
) -> StackDispositionResponse {
    match disposition {
        FrameDisposition::Accepted(diagnostics) => StackDispositionResponse {
            source: source.map(|path| path.to_string_lossy().into_owned()),
            accepted: true,
            reason: None,
            diagnostics: Some(stack_diagnostics_response(diagnostics)),
        },
        FrameDisposition::Rejected(reason) => StackDispositionResponse {
            source: source.map(|path| path.to_string_lossy().into_owned()),
            accepted: false,
            reason: Some(reason.to_string()),
            diagnostics: None,
        },
    }
}

fn stack_diagnostics_response(diagnostics: FrameDiagnostics) -> StackDiagnosticsResponse {
    StackDiagnosticsResponse {
        matched_stars: diagnostics.matched_stars,
        registration_rms_pixels: diagnostics.registration_rms_pixels,
        registration_drift_pixels: diagnostics.registration_drift_pixels,
        scale: diagnostics.transform.scale,
        rotation_degrees: diagnostics.transform.rotation_radians.to_degrees(),
        translation_x: diagnostics.transform.translation_x,
        translation_y: diagnostics.transform.translation_y,
        normalization_mean_gain: diagnostics.normalization_mean_gain,
        normalization_mean_offset: diagnostics.normalization_mean_offset,
        overlap_fraction: diagnostics.overlap_fraction,
        integrated_fraction: diagnostics.integrated_fraction,
        accepted_samples: diagnostics.accepted_samples,
        rejected_samples: diagnostics.rejected_samples,
    }
}

fn owned_json(value: &impl Serialize) -> Result<*mut c_char, String> {
    let json = serde_json::to_string(value).map_err(|error| error.to_string())?;
    CString::new(json)
        .map(CString::into_raw)
        .map_err(|_| "serialized JSON contains a NUL byte".into())
}

unsafe fn required_f32_slice<'a>(
    data: *const f32,
    length: usize,
    name: &str,
) -> Result<&'a [f32], String> {
    if data.is_null() {
        return Err(format!("{name} is required"));
    }
    if !(data as usize).is_multiple_of(std::mem::align_of::<f32>()) {
        return Err(format!("{name} is not aligned for float samples"));
    }
    Ok(unsafe { std::slice::from_raw_parts(data, length) })
}

unsafe fn required_f32_slice_mut<'a>(
    data: *mut f32,
    length: usize,
    name: &str,
) -> Result<&'a mut [f32], String> {
    if data.is_null() {
        return Err(format!("{name} is required"));
    }
    if !(data as usize).is_multiple_of(std::mem::align_of::<f32>()) {
        return Err(format!("{name} is not aligned for float samples"));
    }
    Ok(unsafe { std::slice::from_raw_parts_mut(data, length) })
}

unsafe fn optional_mask(mask: *const u8, length: usize) -> Result<Option<Vec<bool>>, String> {
    if mask.is_null() {
        return if length == 0 {
            Ok(None)
        } else {
            Err("background mask is null but mask_length is non-zero".into())
        };
    }
    let bytes = unsafe { std::slice::from_raw_parts(mask, length) };
    if bytes.iter().any(|value| *value > 1) {
        return Err("background mask entries must be zero or one".into());
    }
    Ok(Some(bytes.iter().map(|value| *value != 0).collect()))
}

unsafe fn required_background_model<'a>(
    model: *const SeizaBackgroundModel,
) -> Result<&'a SeizaBackgroundModel, String> {
    unsafe { model.as_ref() }.ok_or_else(|| "background model is required".into())
}

unsafe fn required_live_stacker<'a>(
    stacker: *const SeizaLiveStacker,
) -> Result<&'a SeizaLiveStacker, String> {
    unsafe { stacker.as_ref() }.ok_or_else(|| "live stacker is required".into())
}

unsafe fn required_live_stacker_mut<'a>(
    stacker: *mut SeizaLiveStacker,
) -> Result<&'a mut SeizaLiveStacker, String> {
    unsafe { stacker.as_mut() }.ok_or_else(|| "live stacker is required".into())
}

unsafe fn required_stack_snapshot<'a>(
    snapshot: *const SeizaStackSnapshot,
) -> Result<&'a SeizaStackSnapshot, String> {
    unsafe { snapshot.as_ref() }.ok_or_else(|| "stack snapshot is required".into())
}

fn required_path(value: *const c_char, name: &str) -> Result<PathBuf, String> {
    optional_path(value)?.ok_or_else(|| format!("{name} is required"))
}

fn required_str(value: *const c_char, name: &str) -> Result<String, String> {
    if value.is_null() {
        return Err(format!("{name} is required"));
    }
    unsafe { CStr::from_ptr(value) }
        .to_str()
        .map(str::to_owned)
        .map_err(|_| format!("{name} is not valid UTF-8"))
}

fn optional_path(value: *const c_char) -> Result<Option<PathBuf>, String> {
    if value.is_null() {
        return Ok(None);
    }
    let value = unsafe { CStr::from_ptr(value) }
        .to_str()
        .map_err(|_| "path is not valid UTF-8".to_string())?;
    if value.is_empty() {
        Ok(None)
    } else {
        Ok(Some(Path::new(value).to_path_buf()))
    }
}

fn ffi_result<T>(
    error_out: *mut *mut c_char,
    body: impl FnOnce() -> Result<T, String>,
) -> Option<T> {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(value)) => Some(value),
        Ok(Err(error)) => {
            set_error(error_out, error);
            None
        }
        Err(_) => {
            set_error(error_out, "Seiza core panicked".to_string());
            None
        }
    }
}

fn clear_error(error_out: *mut *mut c_char) {
    if !error_out.is_null() {
        unsafe { *error_out = ptr::null_mut() };
    }
}

fn set_error(error_out: *mut *mut c_char, error: String) {
    if error_out.is_null() {
        return;
    }
    let sanitized = error.replace('\0', "�");
    if let Ok(error) = CString::new(sanitized) {
        unsafe { *error_out = error.into_raw() };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn card(value: &str) -> [u8; 80] {
        let mut card = [b' '; 80];
        card[..value.len()].copy_from_slice(value.as_bytes());
        card
    }

    fn synthetic_fits() -> Vec<u8> {
        let mut bytes = Vec::new();
        for value in [
            "SIMPLE  =                    T",
            "BITPIX  =                   16",
            "NAXIS   =                    2",
            "NAXIS1  =                    2",
            "NAXIS2  =                    2",
            "BZERO   =                32768",
            "OBJECT  = 'M42'",
            "DATE-OBS= '2025-07-20T12:34:56.5Z'",
            "END",
        ] {
            bytes.extend_from_slice(&card(value));
        }
        bytes.resize(2880, b' ');
        for value in [0_i16, 100, 1000, 20_000] {
            bytes.write_all(&value.to_be_bytes()).unwrap();
        }
        bytes.resize(5760, 0);
        bytes
    }

    fn background_plane(width: usize, height: usize) -> Vec<f32> {
        let mut image = Vec::with_capacity(width * height);
        for y in 0..height {
            let y = 2.0 * y as f32 / (height - 1) as f32 - 1.0;
            for x in 0..width {
                let x = 2.0 * x as f32 / (width - 1) as f32 - 1.0;
                image.push(0.2 + 0.08 * x - 0.04 * y);
            }
        }
        image
    }

    fn gaussian_star(size: usize, fwhm: f32) -> Vec<f32> {
        let center = size / 2;
        let sigma = fwhm / 2.354_82;
        let mut image = Vec::with_capacity(size * size);
        for y in 0..size {
            for x in 0..size {
                let radius_squared = ((x as isize - center as isize).pow(2)
                    + (y as isize - center as isize).pow(2))
                    as f32;
                image.push((-0.5 * radius_squared / sigma.powi(2)).exp());
            }
        }
        let flux = image.iter().sum::<f32>();
        image.iter_mut().for_each(|sample| *sample /= flux);
        image
    }

    fn stacking_star_field(width: usize, height: usize) -> Vec<f32> {
        let positions = [
            (19.7_f32, 16.4_f32),
            (71.3, 28.1),
            (132.2, 34.8),
            (43.1, 49.7),
            (103.4, 58.3),
            (22.8, 70.2),
            (82.7, 76.5),
            (143.1, 87.8),
            (54.4, 96.2),
            (116.8, 104.1),
            (31.2, 113.0),
            (91.5, 118.4),
        ];
        let mut image = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let noise = ((x * 17 + y * 31) % 23) as f32 * 0.12 - 1.32;
                let mut value = 100.0 + noise;
                for (index, (star_x, star_y)) in positions.iter().enumerate() {
                    let dx = x as f32 - star_x;
                    let dy = y as f32 - star_y;
                    value +=
                        (900.0 + index as f32 * 130.0) * (-(dx.mul_add(dx, dy * dy)) / 3.2).exp();
                }
                image.push(value);
            }
        }
        image
    }

    fn no_adjustment_stack_options() -> CString {
        CString::new(
            r#"{
                "normalization": {"mode": "none"},
                "rejection": {"mode": "none"}
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn deconvolution_cabi_restores_in_place_and_reports_errors() {
        let size = 41;
        let center = size / 2;
        let mut image = gaussian_star(size, 2.8);
        let input_peak = image[center * size + center];
        let input_flux = image.iter().sum::<f32>();
        let mut error = ptr::null_mut();

        assert!(unsafe {
            seiza_deconvolve_in_place(
                image.as_mut_ptr(),
                image.len(),
                size,
                size,
                1,
                2.8,
                4,
                0.35,
                0.001,
                2.0,
                &mut error,
            )
        });
        assert!(error.is_null());
        assert!(image[center * size + center] > input_peak);
        assert!((image.iter().sum::<f32>() - input_flux).abs() < 1.0e-5);

        assert!(!unsafe {
            seiza_deconvolve_in_place(
                image.as_mut_ptr(),
                image.len() - 1,
                size,
                size,
                1,
                2.8,
                4,
                0.35,
                0.001,
                2.0,
                &mut error,
            )
        });
        let message = unsafe { CStr::from_ptr(error) }.to_str().unwrap();
        assert!(message.contains("expected"));
        unsafe { seiza_string_free(error) };
    }

    #[test]
    fn background_cabi_fits_renders_and_corrects_a_model() {
        let (width, height) = (96, 72);
        let image = background_plane(width, height);
        let config = CString::new(
            r#"{"model":{"kind":"polynomial","degree":1,"ridge":0.0},"sample_radius":2}"#,
        )
        .unwrap();
        let mut error = ptr::null_mut();
        let model = unsafe {
            seiza_background_fit(
                image.as_ptr(),
                image.len(),
                width,
                height,
                1,
                ptr::null(),
                0,
                config.as_ptr(),
                &mut error,
            )
        };
        assert!(!model.is_null());
        assert!(error.is_null());
        assert_eq!(unsafe { seiza_background_model_width(model) }, width);
        assert_eq!(unsafe { seiza_background_model_height(model) }, height);
        assert_eq!(unsafe { seiza_background_model_channels(model) }, 1);
        assert_eq!(
            unsafe { seiza_background_model_data_length(model) },
            image.len()
        );

        let diagnostics = unsafe { seiza_background_model_diagnostics_json(model) };
        let diagnostics: Value =
            serde_json::from_str(unsafe { CStr::from_ptr(diagnostics) }.to_str().unwrap()).unwrap();
        assert!(
            diagnostics["diagnostics"]["accepted_samples"]
                .as_u64()
                .unwrap()
                > 10
        );

        let mut rendered = vec![0.0; image.len()];
        assert!(unsafe {
            seiza_background_model_render(model, rendered.as_mut_ptr(), rendered.len(), &mut error)
        });
        let mse = rendered
            .iter()
            .zip(&image)
            .map(|(actual, expected)| f64::from(*actual - *expected).powi(2))
            .sum::<f64>()
            / rendered.len() as f64;
        let rmse = mse.sqrt();
        assert!(rmse < 0.003, "background RMSE was {rmse}");

        let mut corrected = image.clone();
        assert!(unsafe {
            seiza_background_model_correct_in_place(
                model,
                corrected.as_mut_ptr(),
                corrected.len(),
                SEIZA_BACKGROUND_CORRECTION_SUBTRACT,
                &mut error,
            )
        });
        let left = corrected[height / 2 * width + 3];
        let right = corrected[height / 2 * width + width - 4];
        assert!((left - right).abs() < 0.003);
        unsafe { seiza_background_model_free(model) };
    }

    #[test]
    fn background_cabi_rejects_invalid_mask_bytes() {
        let (width, height) = (32, 32);
        let image = background_plane(width, height);
        let mut mask = vec![0_u8; width * height];
        mask[10] = 2;
        let mut error = ptr::null_mut();
        let model = unsafe {
            seiza_background_fit(
                image.as_ptr(),
                image.len(),
                width,
                height,
                1,
                mask.as_ptr(),
                mask.len(),
                ptr::null(),
                &mut error,
            )
        };
        assert!(model.is_null());
        assert!(!error.is_null());
        let message = unsafe { CStr::from_ptr(error) }.to_str().unwrap();
        assert!(message.contains("zero or one"));
        unsafe { seiza_string_free(error) };
    }

    #[test]
    fn stacking_cabi_pushes_views_snapshots_and_finishes_without_copying() {
        let (width, height) = (160, 128);
        let image = stacking_star_field(width, height);
        let config = no_adjustment_stack_options();
        let mut error = ptr::null_mut();
        let mut stacker = unsafe {
            seiza_live_stacker_create(
                image.as_ptr(),
                image.len(),
                width,
                height,
                1,
                config.as_ptr(),
                &mut error,
            )
        };
        assert!(!stacker.is_null());
        assert!(error.is_null());
        assert_eq!(unsafe { seiza_live_stacker_width(stacker) }, width);
        assert_eq!(unsafe { seiza_live_stacker_height(stacker) }, height);
        assert_eq!(unsafe { seiza_live_stacker_channels(stacker) }, 1);
        assert_eq!(
            unsafe { seiza_live_stacker_data_length(stacker) },
            image.len()
        );

        let initial_mean =
            unsafe { std::slice::from_raw_parts(seiza_live_stacker_mean(stacker), image.len()) };
        assert_eq!(initial_mean, image);
        let initial_coverage = unsafe {
            std::slice::from_raw_parts(seiza_live_stacker_coverage(stacker), image.len())
        };
        assert!(initial_coverage.iter().all(|count| *count == 1));

        let disposition_json = unsafe {
            seiza_live_stacker_push_linear_json(
                stacker,
                image.as_ptr(),
                image.len(),
                width,
                height,
                1,
                &mut error,
            )
        };
        assert!(!disposition_json.is_null());
        assert!(error.is_null());
        let disposition: Value = serde_json::from_str(
            unsafe { CStr::from_ptr(disposition_json) }
                .to_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(disposition["accepted"], true);
        assert!(disposition["diagnostics"]["matchedStars"].as_u64().unwrap() >= 6);
        unsafe { seiza_string_free(disposition_json) };
        assert_eq!(unsafe { seiza_live_stacker_accepted_frames(stacker) }, 2);
        assert_eq!(unsafe { seiza_live_stacker_rejected_frames(stacker) }, 0);

        let snapshot = unsafe { seiza_live_stacker_snapshot(stacker, &mut error) };
        assert!(!snapshot.is_null());
        let coverage = unsafe {
            std::slice::from_raw_parts(seiza_stack_snapshot_coverage(snapshot), image.len())
        };
        assert!(coverage.iter().all(|count| *count == 2));
        let variance = unsafe {
            std::slice::from_raw_parts(seiza_stack_snapshot_variance(snapshot), image.len())
        };
        assert!(variance.iter().all(|value| value.abs() < f32::EPSILON));
        unsafe { seiza_stack_snapshot_free(snapshot) };

        let snapshot = unsafe { seiza_live_stacker_finish(&mut stacker, &mut error) };
        assert!(!snapshot.is_null());
        assert!(stacker.is_null());
        assert_eq!(unsafe { seiza_stack_snapshot_accepted_frames(snapshot) }, 2);
        assert_eq!(
            unsafe { seiza_stack_snapshot_data_length(snapshot) },
            image.len()
        );
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("stack.fits");
        let output_c = CString::new(output.to_str().unwrap()).unwrap();
        assert!(unsafe {
            seiza_stack_snapshot_write_fits(snapshot, output_c.as_ptr(), &mut error)
        });
        assert_eq!(
            FitsImage::open(&output)
                .unwrap()
                .header_f64("STACKCNT")
                .unwrap(),
            2.0
        );
        unsafe { seiza_stack_snapshot_free(snapshot) };
    }

    #[test]
    fn stacking_cabi_opens_fits_and_rejects_duplicate_paths() {
        let (width, height) = (160, 128);
        let data = stacking_star_field(width, height);
        let image = LinearImage::new(width, height, 1, data).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let first = directory.path().join("light-001.fits");
        let second = directory.path().join("light-002.fits");
        seiza_stacking::write_processed_image_fits_f32(&first, &image, &[], &[]).unwrap();
        seiza_stacking::write_processed_image_fits_f32(&second, &image, &[], &[]).unwrap();
        let first_c = CString::new(first.to_str().unwrap()).unwrap();
        let second_c = CString::new(second.to_str().unwrap()).unwrap();
        let config = no_adjustment_stack_options();
        let mut error = ptr::null_mut();
        let stacker = unsafe {
            seiza_live_stacker_open_fits(
                first_c.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                0.0,
                config.as_ptr(),
                &mut error,
            )
        };
        assert!(!stacker.is_null());
        let disposition =
            unsafe { seiza_live_stacker_push_fits_json(stacker, second_c.as_ptr(), &mut error) };
        assert!(!disposition.is_null());
        unsafe { seiza_string_free(disposition) };

        let duplicate =
            unsafe { seiza_live_stacker_push_fits_json(stacker, second_c.as_ptr(), &mut error) };
        assert!(duplicate.is_null());
        assert!(!error.is_null());
        assert!(
            unsafe { CStr::from_ptr(error) }
                .to_str()
                .unwrap()
                .contains("already been used")
        );
        unsafe { seiza_string_free(error) };
        error = ptr::null_mut();
        let snapshot = unsafe { seiza_live_stacker_snapshot(stacker, &mut error) };
        assert!(!snapshot.is_null());
        assert!(!unsafe {
            seiza_stack_snapshot_write_fits(snapshot, first_c.as_ptr(), &mut error)
        });
        assert!(
            unsafe { CStr::from_ptr(error) }
                .to_str()
                .unwrap()
                .contains("must not refer")
        );
        unsafe {
            seiza_string_free(error);
            seiza_stack_snapshot_free(snapshot);
            seiza_live_stacker_free(stacker);
        }
    }

    #[test]
    fn stacking_cabi_rejects_unknown_configuration_fields() {
        let image = stacking_star_field(160, 128);
        let config = CString::new(r#"{"mystery":true}"#).unwrap();
        let mut error = ptr::null_mut();
        let stacker = unsafe {
            seiza_live_stacker_create(
                image.as_ptr(),
                image.len(),
                160,
                128,
                1,
                config.as_ptr(),
                &mut error,
            )
        };
        assert!(stacker.is_null());
        assert!(!error.is_null());
        assert!(
            unsafe { CStr::from_ptr(error) }
                .to_str()
                .unwrap()
                .contains("unknown field")
        );
        unsafe { seiza_string_free(error) };

        let reference = CString::new("reference.fits").unwrap();
        error = ptr::null_mut();
        let stacker = unsafe {
            seiza_live_stacker_open_fits(
                reference.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                60.0,
                ptr::null(),
                &mut error,
            )
        };
        assert!(stacker.is_null());
        assert!(
            unsafe { CStr::from_ptr(error) }
                .to_str()
                .unwrap()
                .contains("requires a dark path")
        );
        unsafe { seiza_string_free(error) };
    }

    #[test]
    fn stacking_cabi_returns_frame_rejection_as_disposition_json() {
        let (width, height) = (160, 128);
        let image = stacking_star_field(width, height);
        let config = no_adjustment_stack_options();
        let mut error = ptr::null_mut();
        let stacker = unsafe {
            seiza_live_stacker_create(
                image.as_ptr(),
                image.len(),
                width,
                height,
                1,
                config.as_ptr(),
                &mut error,
            )
        };
        let rgb = image
            .iter()
            .flat_map(|value| [*value; 3])
            .collect::<Vec<_>>();
        let disposition_json = unsafe {
            seiza_live_stacker_push_linear_json(
                stacker,
                rgb.as_ptr(),
                rgb.len(),
                width,
                height,
                3,
                &mut error,
            )
        };
        assert!(!disposition_json.is_null());
        assert!(error.is_null());
        let disposition: Value = serde_json::from_str(
            unsafe { CStr::from_ptr(disposition_json) }
                .to_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(disposition["accepted"], false);
        assert!(disposition["reason"].as_str().unwrap().contains("channel"));
        assert_eq!(unsafe { seiza_live_stacker_rejected_frames(stacker) }, 1);
        unsafe {
            seiza_string_free(disposition_json);
            seiza_live_stacker_free(stacker);
        }
    }

    #[test]
    fn bgra_view_swaps_red_and_blue_and_caches() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("swatch.png");
        image::RgbImage::from_fn(2, 1, |x, _| image::Rgb([10 + x as u8, 20, 30]))
            .save(&path)
            .unwrap();

        let image = render_path(&path, &StretchParams::default(), 0, RgbStretchMode::Auto).unwrap();
        assert_eq!(image.rgba.len(), image.bgra().len());
        for (rgba, bgra) in image.rgba.chunks_exact(4).zip(image.bgra().chunks_exact(4)) {
            assert_eq!(
                [bgra[0], bgra[1], bgra[2], bgra[3]],
                [rgba[2], rgba[1], rgba[0], rgba[3]]
            );
        }
        // The cached buffer is reused across calls.
        assert_eq!(image.bgra().as_ptr(), image.bgra().as_ptr());
    }

    #[test]
    fn renders_a_fits_with_a_parameterized_stretch_config() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("test.fits");
        std::fs::write(&path, synthetic_fits()).unwrap();

        let config = StretchConfig::auto_mtf(StretchParams::default(), 4096);
        let image = render_fits_with_config(FitsImage::open(&path).unwrap(), &config, 0).unwrap();
        assert_eq!((image.width, image.height), (2, 2));
        assert_eq!(image.rgba.len(), 16);
        // Config round-trips through JSON, the form the FFI accepts.
        let json = serde_json::to_string(&config).unwrap();
        let parsed: StretchConfig = serde_json::from_str(&json).unwrap();
        assert!(
            render_fits_with_config(FitsImage::open(&path).unwrap(), &parsed, 0)
                .unwrap()
                .rgba
                .len()
                == 16
        );
        let metadata: Value = serde_json::from_str(image.metadata_json.to_str().unwrap()).unwrap();
        assert_eq!(metadata["format"], "FITS");
        assert!(metadata["displayHistogram"].is_object());
    }

    #[test]
    fn parameterized_rgba16_render_retains_sub_u8_distinctions() {
        let fits = FitsImage {
            width: 4,
            height: 1,
            planes: 1,
            pixels: seiza_fits::Pixels::F32(vec![0.0, 0.5, 0.5001, 1.0]),
            headers: Vec::new(),
        };
        let config: StretchConfig = serde_json::from_value(json!({
            "model": { "type": "identity" },
            "color_strategy": "linked",
            "max_analysis_samples": 4096
        }))
        .unwrap();
        let stack = StretchStack::single(config);
        let image8 = render_fits_with_pipeline(fits.clone(), &stack, None, None, 0, false).unwrap();
        let image16 = render_fits_with_pipeline16(fits, &stack, None, None, 0, false).unwrap();

        assert_eq!(image8.rgba[4], image8.rgba[8]);
        assert_ne!(image16.rgba[4], image16.rgba[8]);
        assert_eq!(image16.rgba[4], 32_768);
        assert_eq!(image16.rgba[8], 32_774);
        assert!(
            image16
                .rgba
                .chunks_exact(4)
                .all(|pixel| pixel[3] == u16::MAX)
        );
        let metadata: Value =
            serde_json::from_str(image16.metadata_json.to_str().unwrap()).unwrap();
        assert_eq!(metadata["bitsPerComponent"], 16);
        assert_eq!(metadata["displayHistogram"]["upperBound"], 65_535.0);
    }

    #[test]
    fn rgba16_cabi_exposes_element_count_and_borrowed_pixels() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("test.fits");
        std::fs::write(&path, synthetic_fits()).unwrap();
        let path = CString::new(path.to_string_lossy().as_bytes()).unwrap();
        let config = CString::new(
            r#"{"model":{"type":"identity"},"color_strategy":"linked","max_analysis_samples":4096}"#,
        )
        .unwrap();
        let mut error = ptr::null_mut();

        let image = unsafe {
            seiza_rendered_image16_open_with_stretch_config(
                path.as_ptr(),
                config.as_ptr(),
                0,
                &mut error,
            )
        };
        assert!(!image.is_null());
        assert!(error.is_null());
        assert_eq!(unsafe { seiza_rendered_image16_width(image) }, 2);
        assert_eq!(unsafe { seiza_rendered_image16_height(image) }, 2);
        let length = unsafe { seiza_rendered_image16_rgba_length(image) };
        assert_eq!(length, 16);
        let pixels =
            unsafe { std::slice::from_raw_parts(seiza_rendered_image16_rgba(image), length) };
        assert!(pixels.chunks_exact(4).all(|pixel| pixel[3] == u16::MAX));
        let metadata = unsafe { CStr::from_ptr(seiza_rendered_image16_metadata_json(image)) };
        let metadata: Value = serde_json::from_slice(metadata.to_bytes()).unwrap();
        assert_eq!(metadata["bitsPerComponent"], 16);
        unsafe { seiza_rendered_image16_free(image) };

        assert_eq!(unsafe { seiza_rendered_image16_width(ptr::null()) }, 0);
        assert_eq!(
            unsafe { seiza_rendered_image16_rgba_length(ptr::null()) },
            0
        );
        assert!(unsafe { seiza_rendered_image16_rgba(ptr::null()) }.is_null());
    }

    #[test]
    fn renders_an_ordered_f32_stretch_stack_and_accepts_single_config_json() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("test.fits");
        std::fs::write(&path, synthetic_fits()).unwrap();

        let first: StretchConfig = serde_json::from_value(json!({
            "model": { "type": "linear", "black": 0.0, "white": 0.75 },
            "color_strategy": "linked",
            "max_analysis_samples": 4096
        }))
        .unwrap();
        let second: StretchConfig = serde_json::from_value(json!({
            "model": { "type": "linear", "black": 0.0, "white": 0.5 },
            "color_strategy": "linked",
            "max_analysis_samples": 4096
        }))
        .unwrap();

        let single_request: StretchConfigRequest =
            serde_json::from_str(&serde_json::to_string(&first).unwrap()).unwrap();
        assert_eq!(
            single_request.into_stack().stages(),
            std::slice::from_ref(&first)
        );

        let stack_json = serde_json::to_string(&[first.clone(), second.clone()]).unwrap();
        let stack = serde_json::from_str::<StretchConfigRequest>(&stack_json)
            .unwrap()
            .into_stack();
        let single = render_fits_with_config(FitsImage::open(&path).unwrap(), &first, 0).unwrap();
        let stacked = render_fits_with_stack(FitsImage::open(&path).unwrap(), &stack, 0).unwrap();

        assert_ne!(stacked.rgba, single.rgba);
        let metadata: Value =
            serde_json::from_str(stacked.metadata_json.to_str().unwrap()).unwrap();
        assert_eq!(metadata["stretchStages"], 2);
    }

    #[test]
    fn render_config_composes_background_correction_before_the_stretch_stack() {
        let first: StretchConfig = serde_json::from_value(json!({
            "model": { "type": "identity" },
            "color_strategy": "linked",
            "max_analysis_samples": 4096
        }))
        .unwrap();
        let request: ImageRenderConfigRequest = serde_json::from_value(json!({
            "stretch": [first],
            "background": {
                "mode": "subtract",
                "config": {
                    "model": { "kind": "polynomial", "degree": 1, "ridge": 0.0 },
                    "sample_radius": 2
                }
            }
        }))
        .unwrap();
        let (stack, background, deconvolution, interactive_preview) = request.into_parts();
        assert_eq!(stack.len(), 1);
        assert!(!interactive_preview);
        assert!(deconvolution.is_none());
        let background = background.unwrap();
        assert_eq!(background.mode, CorrectionMode::Subtract);
        assert_eq!(background.config.sample_radius, Some(2));
    }

    #[test]
    fn fits_render_pipeline_reports_and_applies_background_correction() {
        let (width, height) = (96, 72);
        let fits = FitsImage {
            width,
            height,
            planes: 1,
            pixels: seiza_fits::Pixels::F32(background_plane(width, height)),
            headers: Vec::new(),
        };
        let stretch: StretchConfig = serde_json::from_value(json!({
            "model": { "type": "identity" },
            "color_strategy": "linked",
            "max_analysis_samples": 4096
        }))
        .unwrap();
        let stack = StretchStack::single(stretch);
        let background = BackgroundRenderRequest {
            mode: CorrectionMode::Subtract,
            config: serde_json::from_value(json!({
                "model": { "kind": "polynomial", "degree": 1, "ridge": 0.0 },
                "sample_radius": 2
            }))
            .unwrap(),
        };

        let uncorrected = render_fits_with_stack(fits.clone(), &stack, 0).unwrap();
        let corrected =
            render_fits_with_pipeline(fits, &stack, Some(&background), None, 0, false).unwrap();
        assert_ne!(corrected.rgba, uncorrected.rgba);

        let metadata: Value =
            serde_json::from_str(corrected.metadata_json.to_str().unwrap()).unwrap();
        assert_eq!(metadata["backgroundProcessing"]["mode"], "subtract");
        assert!(metadata["backgroundProcessing"]["diagnostics"].is_object());
        assert_eq!(metadata["inputHistogram"]["lowerBound"], 0.0);
        assert_eq!(metadata["inputHistogram"]["upperBound"], 1.0);
        assert_eq!(
            metadata["inputHistogram"]["red"],
            metadata["inputHistogram"]["green"]
        );
        assert_eq!(
            metadata["inputHistogram"]["red"],
            metadata["inputHistogram"]["blue"]
        );
    }

    #[test]
    fn fits_render_pipeline_reports_and_applies_deconvolution_before_stretching() {
        let size = 41;
        let center = size / 2;
        let mut pixels = vec![0.01; size * size];
        pixels[center * size + center] = 0.7;
        pixels[center * size + center - 1] = 0.35;
        pixels[center * size + center + 1] = 0.35;
        pixels[(center - 1) * size + center] = 0.35;
        pixels[(center + 1) * size + center] = 0.35;
        let fits = FitsImage {
            width: size,
            height: size,
            planes: 1,
            pixels: seiza_fits::Pixels::F32(pixels),
            headers: Vec::new(),
        };
        let stretch: StretchConfig = serde_json::from_value(json!({
            "model": { "type": "identity" },
            "color_strategy": "linked",
            "max_analysis_samples": 4096
        }))
        .unwrap();
        let stack = StretchStack::single(stretch);
        let deconvolution = DeconvolutionRenderRequest {
            psf_fwhm_pixels: 2.8,
            iterations: 4,
            amount: 0.35,
            noise_fraction: 0.001,
            max_correction: 2.0,
        };

        let plain = render_fits_with_stack(fits.clone(), &stack, 0).unwrap();
        let restored =
            render_fits_with_pipeline(fits, &stack, None, Some(&deconvolution), 0, false).unwrap();
        assert_ne!(restored.rgba, plain.rgba);

        let metadata: Value =
            serde_json::from_str(restored.metadata_json.to_str().unwrap()).unwrap();
        let requested_fwhm = metadata["deconvolutionProcessing"]["psfFwhmPixels"]
            .as_f64()
            .unwrap();
        let effective_fwhm = metadata["deconvolutionProcessing"]["effectivePsfFwhmPixels"]
            .as_f64()
            .unwrap();
        assert!((requested_fwhm - 2.8).abs() < 1.0e-5);
        assert!((effective_fwhm - 2.8).abs() < 1.0e-5);
        assert_eq!(metadata["deconvolutionProcessing"]["iterations"], 4);
        assert_eq!(
            metadata["deconvolutionProcessing"]["channels"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn interactive_preview_bounds_linear_samples_before_processing() {
        let request: ImageRenderConfigRequest = serde_json::from_value(json!({
            "stretch": [{
                "model": { "type": "identity" },
                "color_strategy": "linked",
                "max_analysis_samples": 4096
            }],
            "deconvolution": {
                "psf_fwhm_pixels": 3.0
            },
            "interactive_preview": true
        }))
        .unwrap();
        let (stack, background, deconvolution, interactive_preview) = request.into_parts();
        assert!(background.is_none());
        let deconvolution = deconvolution.unwrap();
        assert_eq!(deconvolution.iterations, 4);
        assert_eq!(deconvolution.amount, 0.35);
        assert!(interactive_preview);

        let fits = FitsImage {
            width: 400,
            height: 200,
            planes: 1,
            pixels: seiza_fits::Pixels::F32(background_plane(400, 200)),
            headers: Vec::new(),
        };
        let preview = render_fits_with_pipeline(
            fits,
            &stack,
            None,
            Some(&deconvolution),
            100,
            interactive_preview,
        )
        .unwrap();
        assert_eq!((preview.width, preview.height), (100, 50));
        let metadata: Value =
            serde_json::from_str(preview.metadata_json.to_str().unwrap()).unwrap();
        assert_eq!(metadata["width"], 400);
        assert_eq!(metadata["height"], 200);
        assert_eq!(metadata["interactivePreview"], true);
        assert_eq!(
            metadata["deconvolutionProcessing"]["effectivePsfFwhmPixels"],
            0.75
        );
    }

    #[test]
    fn interactive_preview_reuses_prepared_pixels_across_stretch_edits() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("cached-preview.fits");
        std::fs::write(&path, synthetic_fits()).unwrap();
        let stretch: StretchConfig = serde_json::from_value(json!({
            "model": { "type": "identity" },
            "color_strategy": "linked",
            "max_analysis_samples": 4096
        }))
        .unwrap();
        let stack = StretchStack::single(stretch);

        let deconvolution = DeconvolutionRenderRequest {
            psf_fwhm_pixels: 3.0,
            iterations: 4,
            amount: 0.35,
            noise_fraction: 0.001,
            max_correction: 2.0,
        };
        let first = render_cached_interactive_preview(&path, &stack, None, None, 100).unwrap();
        let second =
            render_cached_interactive_preview(&path, &stack, None, Some(&deconvolution), 100)
                .unwrap();
        let first_metadata: Value =
            serde_json::from_str(first.metadata_json.to_str().unwrap()).unwrap();
        let second_metadata: Value =
            serde_json::from_str(second.metadata_json.to_str().unwrap()).unwrap();
        assert_eq!(first_metadata["interactivePreviewCacheHit"], false);
        assert_eq!(second_metadata["interactivePreviewCacheHit"], true);
        assert!(second_metadata["deconvolutionProcessing"].is_object());

        let background = BackgroundRenderRequest::default();
        assert_ne!(
            interactive_preview_cache_key(&path, None, 100).unwrap(),
            interactive_preview_cache_key(&path, Some(&background), 100).unwrap()
        );
    }

    #[test]
    fn rejects_an_empty_stretch_stack() {
        assert!(serde_json::from_str::<StretchConfigRequest>("[]").is_err());
    }

    #[test]
    fn renders_a_synthetic_fits_and_reports_metadata() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("test.fits");
        std::fs::write(&path, synthetic_fits()).unwrap();

        let image = render_fits(
            FitsImage::open(&path).unwrap(),
            &StretchParams::default(),
            0,
            RgbStretchMode::Auto,
        )
        .unwrap();
        assert_eq!((image.width, image.height), (2, 2));
        assert_eq!(image.rgba.len(), 16);
        let metadata: Value = serde_json::from_str(image.metadata_json.to_str().unwrap()).unwrap();
        assert_eq!(metadata["headers"]["OBJECT"], "M42");
        assert_eq!(metadata["format"], "FITS");
        assert_eq!(metadata["colorKind"], "mono");
        assert_eq!(metadata["inputHistogram"]["lowerBound"], 0.0);
        assert_eq!(metadata["inputHistogram"]["upperBound"], 65_535.0);
        for channel in ["red", "green", "blue"] {
            for histogram in ["inputHistogram", "displayHistogram"] {
                let bins = metadata[histogram][channel].as_array().unwrap();
                assert_eq!(bins.len(), 256);
                assert_eq!(bins.iter().map(|bin| bin.as_u64().unwrap()).sum::<u64>(), 4);
            }
        }
    }

    #[test]
    fn renders_a_png_and_reports_raster_metadata() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("test.png");
        let source = image::RgbImage::from_fn(3, 2, |x, y| {
            image::Rgb([(x * 70) as u8, (y * 90) as u8, 150])
        });
        source.save(&path).unwrap();

        let image = render_path(&path, &StretchParams::default(), 0, RgbStretchMode::Auto).unwrap();
        assert_eq!((image.width, image.height), (3, 2));
        assert_eq!(image.rgba.len(), 24);
        let metadata: Value = serde_json::from_str(image.metadata_json.to_str().unwrap()).unwrap();
        assert_eq!(metadata["format"], "PNG");
        assert_eq!(metadata["colorKind"], "rgb-8");
        assert_eq!(metadata["headers"], json!({}));
        assert_eq!(metadata["inputHistogram"]["lowerBound"], 0.0);
        assert_eq!(metadata["inputHistogram"]["upperBound"], 255.0);
        for channel in ["red", "green", "blue"] {
            for histogram in ["inputHistogram", "displayHistogram"] {
                let bins = metadata[histogram][channel].as_array().unwrap();
                assert_eq!(bins.len(), 256);
                assert_eq!(bins.iter().map(|bin| bin.as_u64().unwrap()).sum::<u64>(), 6);
            }
        }
    }

    #[test]
    fn rgba16_raster_render_preserves_sixteen_bit_png_samples() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("test16.png");
        let source = image::ImageBuffer::<image::Rgba<u16>, Vec<u16>>::from_raw(
            2,
            1,
            vec![
                1_000,
                1_001,
                32_768,
                u16::MAX,
                4_000,
                8_000,
                16_000,
                u16::MAX,
            ],
        )
        .unwrap();
        source.save(&path).unwrap();

        let image =
            render_path16(&path, &StretchParams::default(), 0, RgbStretchMode::Auto).unwrap();
        assert_eq!((image.width, image.height), (2, 1));
        assert_eq!(
            image.rgba,
            [
                1_000,
                1_001,
                32_768,
                u16::MAX,
                4_000,
                8_000,
                16_000,
                u16::MAX
            ]
        );
        let metadata: Value = serde_json::from_str(image.metadata_json.to_str().unwrap()).unwrap();
        assert_eq!(metadata["format"], "PNG");
        assert_eq!(metadata["colorKind"], "rgba-16");
        assert_eq!(metadata["bitsPerComponent"], 16);
    }

    #[test]
    fn rgba16_render_buffer_round_trips_through_png_and_tiff_encoders() {
        let fits = FitsImage {
            width: 2,
            height: 1,
            planes: 1,
            pixels: seiza_fits::Pixels::U16(vec![1_000, 1_001]),
            headers: Vec::new(),
        };
        let config: StretchConfig = serde_json::from_value(json!({
            "model": { "type": "identity" },
            "color_strategy": "linked",
            "max_analysis_samples": 4096
        }))
        .unwrap();
        let image =
            render_fits_with_pipeline16(fits, &StretchStack::single(config), None, None, 0, false)
                .unwrap();
        let directory = tempfile::tempdir().unwrap();

        for (name, format) in [
            ("export.png", image::ImageFormat::Png),
            ("export.tiff", image::ImageFormat::Tiff),
        ] {
            let path = directory.path().join(name);
            let buffer = image::ImageBuffer::<image::Rgba<u16>, Vec<u16>>::from_raw(
                image.width,
                image.height,
                image.rgba.clone(),
            )
            .unwrap();
            buffer.save_with_format(&path, format).unwrap();
            let decoded = image::open(&path).unwrap();
            assert_eq!(decoded.color(), image::ColorType::Rgba16);
            assert_eq!(decoded.to_rgba16().into_raw(), image.rgba);
        }
    }

    #[test]
    fn downsampling_preserves_aspect_ratio() {
        let rgba = vec![255; 400 * 200 * 4];
        let (width, height, pixels) = downsample_rgba(400, 200, rgba, 100);
        assert_eq!((width, height), (100, 50));
        assert_eq!(pixels.len(), 100 * 50 * 4);
    }

    #[test]
    fn rgb_linear_and_linked_auto_use_shared_channel_mappings() {
        let rgb = RgbImage16 {
            width: 2,
            height: 2,
            data: vec![
                0, 32_768, 65_535, 500, 1_000, 2_000, 4_000, 8_000, 16_000, 20_000, 30_000, 40_000,
            ],
        };
        let params = StretchParams::default();

        let linear = stretch_rgb(&rgb, &params, RgbStretchMode::Linear);
        assert_eq!(&linear[..4], &[0, 128, 255, 255]);

        let statistics = linked_rgb_statistics(&rgb);
        assert_eq!(statistics.median, 8_167);
        assert!((statistics.mad - 7_166.666_666_666_667).abs() < 1e-9);
        let expected = seiza_fits::stretch_u16_to_u8(&rgb.data, &statistics, &params);
        let linked = stretch_rgb(&rgb, &params, RgbStretchMode::LinkedAuto);
        for (pixel, expected) in linked.chunks_exact(4).zip(expected.chunks_exact(3)) {
            assert_eq!(&pixel[..3], expected);
            assert_eq!(pixel[3], 255);
        }
    }

    #[test]
    fn rgb_stretch_mode_rejects_unknown_abi_values() {
        assert_eq!(RgbStretchMode::from_raw(0), Ok(RgbStretchMode::Auto));
        assert_eq!(RgbStretchMode::from_raw(1), Ok(RgbStretchMode::LinkedAuto));
        assert_eq!(RgbStretchMode::from_raw(2), Ok(RgbStretchMode::Linear));
        assert!(RgbStretchMode::from_raw(3).is_err());
    }

    #[test]
    fn background_correction_mode_rejects_unknown_abi_values() {
        assert_eq!(
            background_correction_mode(SEIZA_BACKGROUND_CORRECTION_SUBTRACT),
            Ok(CorrectionMode::Subtract)
        );
        assert_eq!(
            background_correction_mode(SEIZA_BACKGROUND_CORRECTION_DIVIDE),
            Ok(CorrectionMode::Divide)
        );
        assert!(background_correction_mode(2).is_err());
    }

    #[test]
    fn catalog_setup_presets_include_solver_and_overlay_data() {
        let standard = CatalogSetupPreset::StandardBlind.datasets();
        assert!(standard.contains(&Dataset::StarsDeepGaia17));
        assert!(standard.contains(&Dataset::BlindGaia16));
        assert!(standard.contains(&Dataset::Objects));
        assert!(standard.contains(&Dataset::Transients));
        assert!(standard.contains(&Dataset::MinorBodies));

        let deepest = CatalogSetupPreset::DeepestBlind.datasets();
        assert!(deepest.contains(&Dataset::StarsDeepGaia20));
        assert!(!deepest.contains(&Dataset::StarsDeepGaia17));

        let all = CatalogSetupPreset::All.datasets();
        assert!(all.len() > standard.len());
        assert!(all.contains(&Dataset::StarsLiteTycho2Identifiers));
    }

    #[test]
    fn catalog_status_requires_a_star_catalog_and_blind_index() {
        let directory = tempfile::tempdir().unwrap();
        for name in [
            "stars-deep-gaia17.bin",
            "objects.bin",
            "transients.bin",
            "minor-bodies.bin",
        ] {
            std::fs::write(directory.path().join(name), []).unwrap();
        }

        let incomplete = catalog_status(Some(directory.path()));
        assert!(!incomplete.ready_for_solving);
        assert!(incomplete.ready_for_overlays);

        std::fs::write(directory.path().join("blind-gaia16.idx"), []).unwrap();
        let ready = catalog_status(Some(directory.path()));
        assert!(ready.ready_for_solving);
        assert!(ready.ready_for_overlays);
        assert!(
            ready
                .star_catalog
                .path
                .unwrap()
                .ends_with("stars-deep-gaia17.bin")
        );
    }

    #[test]
    fn new_downloader_events_map_to_setup_progress() {
        unsafe extern "C" fn capture_progress(json: *const c_char, context: *mut c_void) {
            let json = unsafe { CStr::from_ptr(json) }.to_str().unwrap();
            let events = unsafe { &mut *context.cast::<Vec<Value>>() };
            events.push(serde_json::from_str(json).unwrap());
        }

        let mut events = Vec::<Value>::new();
        let reporter = CatalogSetupReporter {
            callback: Some(capture_progress),
            context: (&mut events as *mut Vec<Value>) as usize,
            files_total: 3,
        };
        let path = PathBuf::from("/tmp/catalog.bin");

        reporter.download_event(
            DownloadEvent::Verifying {
                name: "catalog.bin".into(),
            },
            0,
        );
        reporter.download_event(
            DownloadEvent::Installing {
                name: "catalog.bin".into(),
                path: path.clone(),
            },
            0,
        );
        reporter.download_event(
            DownloadEvent::InstallComplete {
                name: "catalog.bin".into(),
                path,
            },
            1,
        );

        assert_eq!(events[0]["phase"], "verifying");
        assert_eq!(events[1]["phase"], "installing");
        assert_eq!(events[2]["message"], "Installed catalog.bin");
        assert_eq!(events[2]["filesCompleted"], 1);
        assert_eq!(events[2]["filesTotal"], 3);
    }

    #[test]
    fn projects_catalog_outline_geometry_into_image_pixels() {
        let wcs = Wcs::from_center_scale_rotation((10.0, 20.0), (100.0, 100.0), 3.6, 0.0, false);
        let expected = [(30.0, 40.0), (70.0, 40.0), (50.0, 80.0)];
        let vertices = expected
            .iter()
            .map(|&(x, y)| wcs.pixel_to_world(x, y))
            .collect();
        let outlines = project_outline_geometries(
            vec![ObjectGeometry {
                id: "openngc:NGC1#outline-1".into(),
                source_record_id: "openngc:NGC1".into(),
                role: GeometryRole::BrightnessLevel,
                quality: GeometryQuality::Catalog,
                method: "OpenNGC outline".into(),
                evidence: String::new(),
                data: GeometryData::OutlineSet {
                    level: Some("1".into()),
                    contours: vec![seiza::objects::ObjectContour {
                        closed: true,
                        vertices,
                    }],
                },
            }],
            &wcs,
        );

        assert_eq!(outlines.len(), 1);
        assert_eq!(outlines[0].role, "brightness-level");
        assert_eq!(outlines[0].quality, "catalog");
        assert_eq!(outlines[0].level.as_deref(), Some("1"));
        assert!(outlines[0].contours[0].closed);
        for (actual, expected) in outlines[0].contours[0].points.iter().zip(expected) {
            assert!((actual[0] - expected.0).abs() < 1e-6);
            assert!((actual[1] - expected.1).abs() < 1e-6);
        }
    }

    #[test]
    fn parses_fits_acquisition_time_for_dynamic_catalogs() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("dated.fits");
        std::fs::write(&path, synthetic_fits()).unwrap();
        let fits = FitsImage::open(&path).unwrap();

        assert_eq!(
            fits_capture_time(&fits).as_deref(),
            Some("2025-07-20T12:34:56.5Z")
        );
        assert!(parse_iso_jd("2025-07-20T12:34:56.5Z").is_some());
        assert!(parse_iso_jd("not-a-date").is_none());
    }

    #[test]
    fn object_overlay_keeps_named_stars_and_dates_transients() {
        use seiza::objects::{ObjectMetadata, SkyObject};

        let object = |name: &str, common_name: &str, kind: ObjectKind| SkyObject {
            kind,
            ra: 10.0,
            dec: 20.0,
            mag: Some(4.0),
            major_arcmin: None,
            minor_arcmin: None,
            position_angle_deg: None,
            name: name.into(),
            common_name: common_name.into(),
            metadata: ObjectMetadata {
                id: format!("test:{name}"),
                source: "test-catalog".into(),
                aliases: Vec::new(),
                parent_ids: Vec::new(),
                alternate_ids: Vec::new(),
                alternate_sources: Vec::new(),
            },
        };
        let wcs = Wcs::from_center_scale_rotation((10.0, 20.0), (50.0, 50.0), 3.6, 0.0, false);
        let catalog = ObjectCatalog::new(vec![
            object("Sirius", "Dog Star", ObjectKind::Star),
            object("NGC 1", "Test Galaxy", ObjectKind::Galaxy),
        ]);
        let mut output = Vec::new();

        append_object_catalog(&mut output, &catalog, &wcs, (100, 100), None, false).unwrap();

        assert_eq!(output.len(), 2);
        assert!(output.iter().any(|object| object.kind == "star"));
        assert!(output.iter().any(|object| object.kind == "galaxy"));

        let transient_catalog = ObjectCatalog::new(vec![object(
            "SN 2020abc",
            "disc. 2020/01/01",
            ObjectKind::Galaxy,
        )]);
        let mut transients = Vec::new();
        append_object_catalog(
            &mut transients,
            &transient_catalog,
            &wcs,
            (100, 100),
            parse_iso_jd("2025-07-20T12:00:00Z"),
            true,
        )
        .unwrap();
        assert_eq!(transients[0].kind, "transient");
        assert_eq!(transients[0].discovered.as_deref(), Some("2020-01-01"));
        assert_eq!(transients[0].near_capture, Some(false));
    }
}
