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
use seiza_deconvolution::{DeconvolutionConfig, deconvolve, deconvolve_masked};
use seiza_fits::{FitsImage, HeaderValue, RgbImage16, Statistics, StretchParams};
use seiza_stacking::{
    CancelSignal, ChannelCoverage, ChannelSamples, ColorCrop, CropReport, ExternalParameterKind,
    ExternalParameterValue, ExternalToolRequest, ExternalToolSchema, FitsFrame, FrameDiagnostics,
    FrameDisposition, FrameInputMode, FrameSourceRole, ImpulseFilterOptions, LinearImage,
    LiveStacker, MasterBuildOptions, MasterDark, MasterFrame, MasterFrameKind,
    MasterRejectionOptions, RcAstroCli, ReferenceRegion,
    StackExportSnapshot as RustStackExportSnapshot, StackOptions,
    StackSnapshot as RustStackSnapshot, build_master_from_fits,
    checkpoint_depths as rust_checkpoint_depths, crop_report, measure_depth as rust_measure_depth,
    path_identity, paths_refer_to_same_file, write_fits_f32, write_master_fits_f32,
    write_stack_export_fits_f32,
};
use seiza_stretch::{
    ResolvedSampleDomain, SampleDomain, StretchAnalysis, StretchConfig, StretchStack,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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

#[derive(Debug, Deserialize, Serialize)]
#[serde(default)]
struct BackgroundRenderRequest {
    mode: CorrectionMode,
    /// Fraction of the fitted correction to apply, in `[0, 1]`.
    strength: f64,
    config: BackgroundConfig,
}

impl Default for BackgroundRenderRequest {
    fn default() -> Self {
        Self {
            mode: CorrectionMode::default(),
            strength: 1.0,
            config: BackgroundConfig::default(),
        }
    }
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
    physical_samples: bool,
    background: Option<String>,
}

struct PreparedFitsRender {
    source_format: &'static str,
    source_width: usize,
    source_height: usize,
    planes: usize,
    color_kind: &'static str,
    render_width: usize,
    render_height: usize,
    channels: usize,
    data: Vec<f32>,
    validity_mask: Option<Vec<bool>>,
    statistics: Value,
    input_histogram: Value,
    background_metadata: Option<Value>,
    headers: Map<String, Value>,
    interactive_preview: bool,
    live_stack: Option<LivePreviewMetadata>,
}

struct PreparedStretchInput<'a> {
    data: Cow<'a, [f32]>,
    input_histogram: Value,
    deconvolution_metadata: Option<Value>,
    sample_domain: ResolvedSampleDomain,
}

#[derive(Clone, Copy)]
struct RenderPipelineOptions<'a> {
    background: Option<&'a BackgroundRenderRequest>,
    deconvolution: Option<&'a DeconvolutionRenderRequest>,
    sample_domain: &'a SampleDomain,
    max_dimension: u32,
    interactive_preview: bool,
}

type InteractivePreviewCache =
    Mutex<VecDeque<(InteractivePreviewCacheKey, Arc<PreparedFitsRender>)>>;

static INTERACTIVE_PREVIEW_CACHE: OnceLock<InteractivePreviewCache> = OnceLock::new();
const INTERACTIVE_PREVIEW_CACHE_CAPACITY: usize = 2;

#[derive(Debug, Deserialize)]
struct ProcessedRenderRequest {
    stretch: StretchConfigRequest,
    #[serde(default)]
    sample_domain: Option<SampleDomain>,
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
    Processed(Box<ProcessedRenderRequest>),
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
        Option<SampleDomain>,
    ) {
        match self {
            Self::Processed(request) => {
                let ProcessedRenderRequest {
                    stretch,
                    sample_domain,
                    background,
                    deconvolution,
                    interactive_preview,
                } = *request;
                (
                    stretch.into_stack(),
                    background,
                    deconvolution,
                    interactive_preview,
                    sample_domain,
                )
            }
            Self::Stretch(request) => (request.into_stack(), None, None, false, None),
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

    /// Reports a phase update whose byte counters are unset. Download progress,
    /// which carries byte counters, builds its own response.
    fn report_phase(
        &self,
        phase: &'static str,
        message: impl Into<String>,
        file_name: Option<String>,
        files_completed: usize,
    ) {
        self.report(CatalogSetupProgressResponse {
            phase,
            message: message.into(),
            file_name,
            files_completed,
            files_total: self.files_total,
            bytes_completed: None,
            bytes_total: None,
            written_bytes: None,
        });
    }

    fn simple(&self, phase: &'static str, message: impl Into<String>) {
        self.report_phase(phase, message, None, 0);
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
            DownloadEvent::CacheHit { name, .. } => self.report_phase(
                "preparing",
                format!("Found {name} in the download cache"),
                Some(name),
                files_completed,
            ),
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
            DownloadEvent::DownloadComplete { name, .. } => self.report_phase(
                "preparing",
                format!("Downloaded {name}"),
                Some(name),
                files_completed,
            ),
            DownloadEvent::Verifying { name } => self.report_phase(
                "verifying",
                format!("Verifying {name}"),
                Some(name),
                files_completed,
            ),
            DownloadEvent::Installing { name, .. } => self.report_phase(
                "installing",
                format!("Installing {name}"),
                Some(name),
                files_completed,
            ),
            DownloadEvent::InstallComplete { name, .. } => self.report_phase(
                "installing",
                format!("Installed {name}"),
                Some(name),
                files_completed,
            ),
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
}

/// An opaque, thread-safe cooperative cancellation flag. Release it with
/// [`seiza_cancel_signal_free`] after every operation borrowing it has
/// returned.
pub struct SeizaCancelSignal {
    cancelled: Arc<AtomicBool>,
}

/// An immutable owned stack result. Its image and count pointers are borrowed
/// until [`seiza_stack_snapshot_free`] is called.
pub struct SeizaStackSnapshot {
    snapshot: RustStackSnapshot,
    reference_headers: Vec<(String, HeaderValue)>,
    input_paths: Vec<PathBuf>,
}

/// A compact immutable live-stack result for non-destructive output. It owns
/// only the finalized mean, reference headers, scalar frame counts, and the
/// small source-path ledger. Release it with
/// [`seiza_stack_export_snapshot_free`].
pub struct SeizaStackExportSnapshot {
    snapshot: RustStackExportSnapshot,
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

/// One pipelined run: every frame's outcome in order, plus the tallies a
/// caller needs when it is not inspecting each one.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StackPipelineResponse {
    frames: Vec<StackDispositionResponse>,
    integrated: usize,
    rejected: usize,
    failed: usize,
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

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct LiveStackStateResponse {
    schema_version: u32,
    core_version: &'static str,
    configuration_fingerprint: String,
    width: usize,
    height: usize,
    channels: usize,
    accepted_frames: u32,
    rejected_frames: u32,
    input_mode: &'static str,
    input_paths: Vec<String>,
    reference_frame: LiveStackReferenceResponse,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct LiveStackReferenceResponse {
    role: CalibrationFrameRole,
    is_master: bool,
    signature: FrameProbeSignature,
    calibration_state: FrameCalibrationStateResponse,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct LivePreviewMetadata {
    schema_version: u32,
    accepted_frames: u32,
    rejected_frames: u32,
    input_mode: &'static str,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum CalibrationFrameRole {
    Bias,
    Dark,
    DarkFlat,
    Flat,
    Light,
    #[default]
    Unknown,
}

impl From<FrameSourceRole> for CalibrationFrameRole {
    fn from(value: FrameSourceRole) -> Self {
        match value {
            FrameSourceRole::Bias => Self::Bias,
            FrameSourceRole::Dark => Self::Dark,
            FrameSourceRole::DarkFlat => Self::DarkFlat,
            FrameSourceRole::Flat => Self::Flat,
            FrameSourceRole::Light => Self::Light,
            FrameSourceRole::Unknown => Self::Unknown,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
struct FrameProbeSignature {
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
}

impl From<&FrameProbeSignature> for seiza_calibration::FrameSignature {
    fn from(value: &FrameProbeSignature) -> Self {
        let mut signature = Self::default();
        signature.camera = value.camera.clone();
        signature.telescope = value.telescope.clone();
        signature.width = value.width;
        signature.height = value.height;
        signature.channels = value.channels;
        signature.binning_x = value.binning_x;
        signature.binning_y = value.binning_y;
        signature.gain = value.gain;
        signature.offset = value.offset;
        signature.readout_mode = value.readout_mode;
        signature.bayer_pattern = value.bayer_pattern.clone();
        signature.filter = value.filter.clone();
        signature.focal_length_mm = value.focal_length_mm;
        signature.rotation_deg = value.rotation_deg;
        signature.exposure_seconds = value.exposure_seconds;
        signature.camera_temp_c = value.camera_temp_c;
        signature.captured_at_unix = value.captured_at_unix;
        signature
    }
}

impl From<&seiza_calibration::FrameSignature> for FrameProbeSignature {
    fn from(value: &seiza_calibration::FrameSignature) -> Self {
        Self {
            camera: value.camera.clone(),
            telescope: value.telescope.clone(),
            width: value.width,
            height: value.height,
            channels: value.channels,
            binning_x: value.binning_x,
            binning_y: value.binning_y,
            gain: value.gain,
            offset: value.offset,
            readout_mode: value.readout_mode,
            bayer_pattern: value.bayer_pattern.clone(),
            filter: value.filter.clone(),
            focal_length_mm: value.focal_length_mm,
            rotation_deg: value.rotation_deg,
            exposure_seconds: value.exposure_seconds,
            camera_temp_c: value.camera_temp_c,
            captured_at_unix: value.captured_at_unix,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct FrameCalibrationStateResponse {
    bias_subtracted: bool,
    dark_subtracted: bool,
    flat_normalized: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct FrameProbeResponse {
    schema_version: u32,
    path: String,
    format: &'static str,
    role: CalibrationFrameRole,
    raw_image_type: Option<String>,
    is_master: bool,
    signature: FrameProbeSignature,
    calibration_state: FrameCalibrationStateResponse,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum CalibrationPlanKind {
    Bias,
    Dark,
    DarkFlat,
    Flat,
}

impl CalibrationPlanKind {
    fn role(self) -> CalibrationFrameRole {
        match self {
            Self::Bias => CalibrationFrameRole::Bias,
            Self::Dark => CalibrationFrameRole::Dark,
            Self::DarkFlat => CalibrationFrameRole::DarkFlat,
            Self::Flat => CalibrationFrameRole::Flat,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Bias => "bias",
            Self::Dark => "dark",
            Self::DarkFlat => "dark-flat",
            Self::Flat => "flat",
        }
    }

    fn uses_dark_matching(self) -> bool {
        matches!(self, Self::Dark | Self::DarkFlat)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CalibrationPlanRecord {
    path: String,
    #[serde(default)]
    role: CalibrationFrameRole,
    signature: FrameProbeSignature,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CalibrationPlanTolerancesRequest {
    exposure_seconds: Option<f64>,
    exposure_fraction: Option<f64>,
    dark_temperature_c: Option<f64>,
    master_temperature_c: Option<f64>,
    rotation_deg: Option<f64>,
    focal_length_mm: Option<f64>,
    flat_session_seconds: Option<u64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CalibrationPlanRequest {
    kind: CalibrationPlanKind,
    reference: CalibrationPlanRecord,
    #[serde(default)]
    references: Vec<CalibrationPlanRecord>,
    candidates: Vec<CalibrationPlanRecord>,
    #[serde(default = "default_calibration_plan_minimum")]
    minimum: usize,
    #[serde(default)]
    tolerances: CalibrationPlanTolerancesRequest,
    #[serde(default)]
    dependencies: CalibrationPlanDependenciesRequest,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CalibrationPlanDependenciesRequest {
    /// An actually built, usable bias master will isolate dark current and
    /// make exposure scaling safe. This is a fact, not user intent.
    bias_available: bool,
}

const fn default_calibration_plan_minimum() -> usize {
    2
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CalibrationPlanExclusionResponse {
    path: String,
    reason: &'static str,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CalibrationPlanResponse {
    schema_version: u32,
    kind: &'static str,
    minimum: usize,
    ready: bool,
    matched_paths: Vec<String>,
    selected_paths: Vec<String>,
    excluded: Vec<CalibrationPlanExclusionResponse>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum MasterBuildKindRequest {
    Bias,
    Dark,
    Flat,
}

impl MasterBuildKindRequest {
    fn into_core(self) -> MasterFrameKind {
        match self {
            Self::Bias => MasterFrameKind::Bias,
            Self::Dark => MasterFrameKind::Dark,
            Self::Flat => MasterFrameKind::Flat,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
struct MasterRejectionRequest {
    low_sigma: f32,
    high_sigma: f32,
}

impl Default for MasterRejectionRequest {
    fn default() -> Self {
        let defaults = MasterRejectionOptions::default();
        Self {
            low_sigma: defaults.low_sigma,
            high_sigma: defaults.high_sigma,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
struct ImpulseFilterRequest {
    low_sigma: f32,
    high_sigma: f32,
}

impl Default for ImpulseFilterRequest {
    fn default() -> Self {
        let defaults = ImpulseFilterOptions::default();
        Self {
            low_sigma: defaults.low_sigma,
            high_sigma: defaults.high_sigma,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct MasterBuildRequest {
    kind: MasterBuildKindRequest,
    inputs: Vec<PathBuf>,
    output: PathBuf,
    #[serde(default)]
    bias: Option<PathBuf>,
    #[serde(default)]
    dark: Option<PathBuf>,
    #[serde(default)]
    dark_exposure_seconds: Option<f64>,
    #[serde(default)]
    exposure_seconds: Option<f64>,
    #[serde(default)]
    rejection: MasterRejectionRequest,
    #[serde(default)]
    defect_suppression: Option<ImpulseFilterRequest>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct MasterBuildInputResponse {
    path: String,
    accepted_samples: u64,
    rejected_samples: u64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct MasterBuildSkippedInputResponse {
    path: String,
    reason: String,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct MasterBuildRejectionResponse {
    low_sigma: f32,
    high_sigma: f32,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct MasterBuildResponse {
    schema_version: u32,
    kind: &'static str,
    output: String,
    width: usize,
    height: usize,
    channels: usize,
    requested_frames: usize,
    input_frames: usize,
    accepted_samples: u64,
    rejected_samples: u64,
    fallback_pixels: u64,
    defect_pixels_replaced: u64,
    bias_subtracted: bool,
    dark_subtracted: bool,
    normalized: bool,
    output_exposure_seconds: Option<f64>,
    rejection: MasterBuildRejectionResponse,
    inputs: Vec<MasterBuildInputResponse>,
    skipped_inputs: Vec<MasterBuildSkippedInputResponse>,
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
struct CropReportResponse<'a> {
    mode: &'static str,
    grid: SizeResponse,
    region: RegionResponse,
    retained_fraction: f64,
    off_center_limit_pixels: f64,
    channels: Vec<ChannelCoverageResponse<'a>>,
}

impl<'a> CropReportResponse<'a> {
    fn new(crop: ColorCrop, report: &'a CropReport) -> Self {
        Self {
            mode: crop.name(),
            grid: SizeResponse {
                width: report.grid_width,
                height: report.grid_height,
            },
            region: RegionResponse::new(report.region),
            retained_fraction: report.retained_fraction(),
            off_center_limit_pixels: CropReport::off_center_limit_pixels(
                report.grid_width,
                report.grid_height,
            ),
            channels: report
                .channels
                .iter()
                .map(ChannelCoverageResponse::new)
                .collect(),
        }
    }
}

#[derive(Serialize)]
struct ChannelCoverageResponse<'a> {
    name: &'a str,
    region: RegionResponse,
    covered_pixels: usize,
    center_offset_x: f64,
    center_offset_y: f64,
    center_offset_pixels: f64,
    off_center: bool,
}

impl<'a> ChannelCoverageResponse<'a> {
    fn new(coverage: &'a ChannelCoverage) -> Self {
        Self {
            name: &coverage.name,
            region: RegionResponse::new(coverage.region),
            covered_pixels: coverage.covered_pixels,
            center_offset_x: coverage.center_offset_x,
            center_offset_y: coverage.center_offset_y,
            center_offset_pixels: coverage.center_offset_pixels(),
            off_center: coverage.off_center,
        }
    }
}

#[derive(Serialize)]
struct SizeResponse {
    width: usize,
    height: usize,
}

#[derive(Serialize)]
struct RegionResponse {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

impl RegionResponse {
    fn new(region: ReferenceRegion) -> Self {
        Self {
            x: region.x,
            y: region.y,
            width: region.width,
            height: region.height,
        }
    }
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
        if !matches!(channels, 1 | 3) {
            return Err("deconvolution requires one or three channels".into());
        }
        let expected = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(channels))
            .ok_or_else(|| "deconvolution image dimensions overflow".to_string())?;
        if data_length != expected {
            return Err(format!(
                "deconvolution input has {data_length} floats; expected {expected}"
            ));
        }
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
/// Reports the region a color crop would keep across aligned channels, and
/// what each channel covers of the shared grid.
///
/// A pixel counts as covered when every sample of every channel there is
/// finite, so the reported region is the inner area common to all of them.
/// `crop` selects `none`, `bounds`, or `inscribed`; `bounds` keeps the box the
/// covered pixels span, and `inscribed` the largest rectangle every channel
/// covers in full. The report also names any channel whose coverage sits far
/// enough from the others to look like a pointing error rather than dither.
///
/// `names` and `channels` are parallel arrays of `channel_count` entries. Every
/// channel holds `data_length` interleaved linear floats on the same
/// `width` by `height` grid, with `samples_per_pixel` of one or three. The call
/// is synchronous and retains no pointer. The returned JSON is owned by the
/// caller and must be released with [`seiza_string_free`].
///
/// # Safety
/// `names` must point to `channel_count` NUL-terminated strings and `channels`
/// to `channel_count` arrays of `data_length` readable floats. `crop` must be
/// NUL-terminated. When non-null, `error_out` must point to writable storage
/// for one pointer.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn seiza_color_crop_report_json(
    names: *const *const c_char,
    channels: *const *const f32,
    channel_count: usize,
    data_length: usize,
    width: usize,
    height: usize,
    samples_per_pixel: usize,
    crop: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        if channel_count == 0 {
            return Err("crop report requires at least one channel".into());
        }
        if width == 0 || height == 0 {
            return Err("crop report requires a non-empty grid".into());
        }
        if !matches!(samples_per_pixel, 1 | 3) {
            return Err("crop report requires one or three samples per pixel".into());
        }
        let expected = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(samples_per_pixel))
            .ok_or_else(|| "crop report image dimensions overflow".to_string())?;
        if data_length != expected {
            return Err(format!(
                "each crop report channel has {data_length} floats; expected {expected}"
            ));
        }
        if names.is_null() || channels.is_null() {
            return Err("crop report channel names and samples are required".into());
        }
        let crop = required_str(crop, "crop mode")?
            .parse::<ColorCrop>()
            .map_err(|error| error.to_string())?;
        let names = unsafe { std::slice::from_raw_parts(names, channel_count) };
        let channels = unsafe { std::slice::from_raw_parts(channels, channel_count) };
        let mut labels = Vec::with_capacity(channel_count);
        let mut samples = Vec::with_capacity(channel_count);
        for (index, (name, data)) in names.iter().zip(channels).enumerate() {
            labels.push(required_str(*name, &format!("channel {index} name"))?);
            samples.push(unsafe {
                required_f32_slice(*data, data_length, &format!("channel {index}"))?
            });
        }
        let borrowed = labels
            .iter()
            .zip(&samples)
            .map(|(name, data)| ChannelSamples {
                name,
                data,
                width,
                height,
                channels: samples_per_pixel,
            })
            .collect::<Vec<_>>();
        let report = crop_report(&borrowed, crop).map_err(|error| error.to_string())?;
        let json = serde_json::to_string(&CropReportResponse::new(crop, &report))
            .map_err(|error| error.to_string())?;
        CString::new(json)
            .map(CString::into_raw)
            .map_err(|_| "crop report contains a NUL byte".to_string())
    })
    .unwrap_or(ptr::null_mut())
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
        if !mask.is_null() && mask_length != pixels {
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
/// `model` must be null or a live pointer returned by [`seiza_background_fit`];
/// a null model returns an error. `output` must point to `output_length`
/// writable floats. When non-null, `error_out` must point to writable storage
/// for one pointer.
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
/// Corrects an interleaved linear float buffer in place with a fractional
/// strength. Zero leaves the image unchanged; one applies the full correction.
///
/// # Safety
/// `model` must be a live pointer returned by [`seiza_background_fit`]. `data`
/// must point to `data_length` writable floats. When non-null, `error_out` must
/// point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_background_model_correct_in_place_with_strength(
    model: *const SeizaBackgroundModel,
    data: *mut f32,
    data_length: usize,
    mode: u32,
    strength: f64,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let model = unsafe { required_background_model(model)? };
        let data = unsafe { required_f32_slice_mut(data, data_length, "background input")? };
        let mode = background_correction_mode(mode)?;
        model
            .fit
            .correct_in_place_with_strength(data, mode, strength)
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
/// Create a thread-safe cooperative cancellation flag. The returned owner is
/// initially clear and must be released with [`seiza_cancel_signal_free`].
pub extern "C" fn seiza_cancel_signal_create() -> *mut SeizaCancelSignal {
    Box::into_raw(Box::new(SeizaCancelSignal {
        cancelled: Arc::new(AtomicBool::new(false)),
    }))
}

#[unsafe(no_mangle)]
/// Ask work borrowing this signal to stop. The call is thread-safe and may be
/// made from a UI thread while a worker is inside a cancellable operation.
///
/// # Safety
/// `signal` must be null or a live pointer returned by
/// [`seiza_cancel_signal_create`]. It must not be freed concurrently.
pub unsafe extern "C" fn seiza_cancel_signal_cancel(signal: *const SeizaCancelSignal) {
    if let Some(signal) = unsafe { signal.as_ref() } {
        signal.cancelled.store(true, Ordering::Relaxed);
    }
}

#[unsafe(no_mangle)]
/// # Safety
/// `signal` must be null or a pointer returned by
/// [`seiza_cancel_signal_create`] that has not already been freed. Do not free
/// it until every operation borrowing it has returned.
pub unsafe extern "C" fn seiza_cancel_signal_free(signal: *mut SeizaCancelSignal) {
    if !signal.is_null() {
        unsafe { drop(Box::from_raw(signal)) };
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
        Ok(SeizaLiveStacker { stacker })
    })
    .map_or(ptr::null_mut(), |stacker| Box::into_raw(Box::new(stacker)))
}

#[unsafe(no_mangle)]
/// Opens a FITS or XISF reference and optional integrated bias, dark, and flat
/// masters. A positive `dark_exposure_seconds` overrides the dark metadata; zero
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
        let stacker = LiveStacker::open_fits(
            &reference_path,
            bias_path.as_deref(),
            dark_path.as_deref(),
            flat_path.as_deref(),
            dark_exposure_seconds,
            options,
        )
        .map_err(|error| error.to_string())?;
        Ok(SeizaLiveStacker { stacker })
    })
    .map_or(ptr::null_mut(), |stacker| Box::into_raw(Box::new(stacker)))
}

#[unsafe(no_mangle)]
/// Reopens a versioned live-stack context previously written by
/// [`seiza_live_stacker_save_context`]. The restored handle retains its
/// original registration reference, calibration, online rejection moments,
/// frame counters, and source-path ledger, and may immediately accept more
/// frames.
///
/// # Safety
/// `context_path` must be a valid NUL-terminated path. When non-null,
/// `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_live_stacker_open_context(
    context_path: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut SeizaLiveStacker {
    clear_error(error_out);
    ffi_result(error_out, || {
        let context_path = required_path(context_path, "stack context path")?;
        let stacker =
            LiveStacker::open_context(&context_path).map_err(|error| error.to_string())?;
        Ok(SeizaLiveStacker { stacker })
    })
    .map_or(ptr::null_mut(), |stacker| Box::into_raw(Box::new(stacker)))
}

#[unsafe(no_mangle)]
/// Atomically checkpoints every piece of state required to reopen this live
/// stack and continue integrating with identical online rejection behavior.
/// The live handle remains usable after the checkpoint completes.
///
/// # Safety
/// `stacker` must be a live `SeizaLiveStacker` pointer. `context_path` must be
/// a valid NUL-terminated path. When non-null, `error_out` must point to
/// writable storage for one pointer.
pub unsafe extern "C" fn seiza_live_stacker_save_context(
    stacker: *const SeizaLiveStacker,
    context_path: *const c_char,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let stacker = unsafe { required_live_stacker(stacker)? };
        let context_path = required_path(context_path, "stack context path")?;
        stacker
            .stacker
            .save_context(context_path)
            .map_err(|error| error.to_string())
    })
    .is_some()
}

/// Which of the stacker's active masters a prospective light could accept,
/// and why each refused master was set aside. The answer to ask BEFORE
/// pushing a frame in a mode that must warn rather than fail.
///
/// JSON shape: `{"schemaVersion":1,"kept":["bias","dark"],"dropped":
/// [{"kind":"flat","reason":"flat set aside: rotation light=..."}]}` — `kept`
/// lists only masters that are loaded and acceptable; a master that was never
/// loaded appears in neither list. `tolerances` may be null for the defaults.
/// Returns a string released with [`seiza_string_free`], or null with
/// `error_out` set.
///
/// # Safety
///
/// `stacker` must be a live stacker from this library, with this read
/// externally synchronized against mutable operations. `signature` must
/// reference an initialized `SeizaFrameSignature`; `tolerances` must be null
/// or reference an initialized `SeizaMatchTolerances`. When non-null,
/// `error_out` must point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_live_stacker_compatible_calibration_json(
    stacker: *const SeizaLiveStacker,
    signature: *const SeizaFrameSignature,
    tolerances: *const SeizaMatchTolerances,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let stacker = unsafe { required_live_stacker(stacker)? };
        let light = unsafe { frame_signature(signature) }?;
        let tolerances = unsafe { match_tolerances(tolerances) };
        let masters = stacker.stacker.calibration();
        let (kept, dropped) = masters.compatible_for_light_with(&light, &tolerances);
        let kind_of = |reason: &str| -> &'static str {
            if reason.starts_with("bias") {
                "bias"
            } else if reason.starts_with("dark") {
                "dark"
            } else {
                "flat"
            }
        };
        owned_json(&CompatibleCalibrationResponse {
            schema_version: 1,
            kept: active_master_kinds(&kept),
            dropped: dropped
                .iter()
                .map(|reason| DroppedMaster {
                    kind: kind_of(reason),
                    reason: reason.clone(),
                })
                .collect(),
        })
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Returns an owned JSON snapshot of the live stack's resumable identity and
/// counters. Free it with [`seiza_string_free`].
///
/// `inputPaths` is the native, ordered source/calibration ledger used for
/// duplicate-input and output protection. `configurationFingerprint` is a
/// lowercase SHA-256 identity of stack options, current calibration content,
/// and input mode; it excludes counters and paths, changes when calibration
/// changes, and survives a context round trip.
///
/// # Safety
/// `stacker` must be a live `SeizaLiveStacker` pointer. When non-null,
/// `error_out` must point to writable storage for one pointer. Externally
/// synchronize this read with every mutable operation on the same stacker.
pub unsafe extern "C" fn seiza_live_stacker_state_json(
    stacker: *const SeizaLiveStacker,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let stacker = unsafe { required_live_stacker(stacker)? };
        let view = stacker.stacker.view();
        let reference_metadata = stacker.stacker.reference_metadata();
        owned_json(&LiveStackStateResponse {
            schema_version: 1,
            core_version: env!("CARGO_PKG_VERSION"),
            configuration_fingerprint: stacker.stacker.configuration_fingerprint().to_owned(),
            width: view.width,
            height: view.height,
            channels: view.channels,
            accepted_frames: view.accepted_frames,
            rejected_frames: view.rejected_frames,
            input_mode: live_stack_input_mode_name(stacker.stacker.input_mode()),
            input_paths: stacker
                .stacker
                .input_paths()
                .iter()
                .map(|path| path.to_string_lossy().into_owned())
                .collect(),
            reference_frame: LiveStackReferenceResponse {
                role: CalibrationFrameRole::from(reference_metadata.role),
                is_master: reference_metadata.is_master,
                signature: FrameProbeSignature::from(&reference_metadata.signature),
                calibration_state: FrameCalibrationStateResponse {
                    bias_subtracted: reference_metadata.calibration_state.bias_subtracted,
                    dark_subtracted: reference_metadata.calibration_state.dark_subtracted,
                    flat_normalized: reference_metadata.calibration_state.flat_normalized,
                },
            },
        })
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Load and atomically replace the masters used by later FITS/XISF pushes.
///
/// Each master path is optional. Passing all null/empty paths clears the
/// calibration set. A positive `dark_exposure_seconds` overrides the dark
/// header; zero uses the header and requires no override. All supplied files
/// are fully decoded and the complete set is validated against the immutable
/// registration reference before either active calibration or the input-path
/// ledger changes. Already integrated frames are not recalibrated. A stack
/// created from prepared arrays refuses this operation.
///
/// # Safety
/// `stacker` must be a live `SeizaLiveStacker` pointer. Non-null paths must be
/// valid NUL-terminated strings. When non-null, `error_out` must point to
/// writable storage for one pointer.
pub unsafe extern "C" fn seiza_live_stacker_set_calibration_fits(
    stacker: *mut SeizaLiveStacker,
    bias_path: *const c_char,
    dark_path: *const c_char,
    flat_path: *const c_char,
    dark_exposure_seconds: f64,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let bias_path = optional_path(bias_path)?;
        let dark_path = optional_path(dark_path)?;
        let flat_path = optional_path(flat_path)?;
        if dark_path.is_none() && dark_exposure_seconds != 0.0 {
            return Err("a master-dark exposure override requires a dark path".into());
        }
        let dark_exposure_seconds =
            optional_positive_seconds(dark_exposure_seconds, "master-dark exposure override")?;
        let stacker = unsafe { required_live_stacker_mut(stacker)? };
        stacker
            .stacker
            .set_calibration_from_fits_paths(
                bias_path.as_deref(),
                dark_path.as_deref(),
                flat_path.as_deref(),
                dark_exposure_seconds,
            )
            .map_err(|error| error.to_string())
    })
    .is_some()
}

#[unsafe(no_mangle)]
/// Render a bounded RGBA8 preview directly from the current live mean.
///
/// `config_json` accepts the same stretch / optional background / optional
/// deconvolution / optional `sample_domain` schema as
/// [`seiza_rendered_image_open_with_stretch_config`]. Physical-domain mapping
/// is presentation-only and runs after physical background correction and
/// deconvolution, immediately before stretch. When omitted, file-backed
/// calibrate-and-prepare stacks use linked robust physical normalization;
/// prepared-array stacks retain the legacy unit-linear interpretation.
/// `max_dimension` must be positive and bounds the linear buffer before that
/// processing. The returned image owns its pixels and remains valid after the
/// stack changes; free it with [`seiza_rendered_image_free`].
///
/// # Safety
/// `stacker` must be a live `SeizaLiveStacker` pointer and `config_json` a
/// valid NUL-terminated string. When non-null, `error_out` must point to
/// writable storage for one pointer. Externally synchronize this read with
/// every mutable operation on the same stacker.
pub unsafe extern "C" fn seiza_live_stacker_render_preview(
    stacker: *const SeizaLiveStacker,
    config_json: *const c_char,
    max_dimension: u32,
    error_out: *mut *mut c_char,
) -> *mut SeizaRenderedImage {
    clear_error(error_out);
    ffi_result(error_out, || {
        if max_dimension == 0 {
            return Err("live preview maximum dimension must be positive".into());
        }
        let config_json = required_str(config_json, "stretch config JSON")?;
        let request: ImageRenderConfigRequest = serde_json::from_str(&config_json)
            .map_err(|error| format!("invalid stretch config JSON: {error}"))?;
        let (stretch, background, deconvolution, _, sample_domain) = request.into_parts();
        let stacker = unsafe { required_live_stacker(stacker)? };
        let sample_domain = sample_domain
            .unwrap_or_else(|| default_live_preview_sample_domain(stacker.stacker.input_mode()));
        let prepared =
            prepare_live_stack_render(&stacker.stacker, background.as_ref(), max_dimension)?;
        render_prepared_fits(
            &prepared,
            &stretch,
            deconvolution.as_ref(),
            &sample_domain,
            max_dimension,
            false,
        )
    })
    .map_or(ptr::null_mut(), |image| Box::into_raw(Box::new(image)))
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
/// Opens, calibrates, registers, and offers one FITS or XISF frame to the stack.
/// Stacks created from an array reject this path, including after a context
/// restore. The returned disposition JSON is owned and must be freed with
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
        let disposition = stacker
            .stacker
            .push_fits(&path)
            .map_err(|error| error.to_string())?;
        owned_json(&stack_disposition_response(Some(&path), disposition))
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Offer many FITS or XISF paths at once, preparing several frames in
/// parallel. Returns owned JSON that the caller frees with
/// [`seiza_string_free`].
///
/// Reads, calibration, registration and normalization overlap across frames
/// while integration stays in the order given, so the result is identical to
/// offering the same paths one at a time. `paths_json` is a JSON array of
/// strings. `workers` is the read concurrency as well as the compute
/// concurrency; pass 0 to derive it, or raise it when the frames are remote.
/// `max_in_flight_bytes` bounds the memory a derived count may use; pass 0 for
/// the default. `normalized_full_scale` puts a frame the file declares as
/// normalized (`bounds="0:1"`, as PixInsight writes) onto that scale as it is
/// read — pass 65535.0 when the other frames are 16-bit camera data, or 0 to
/// leave every sample exactly as stored.
///
/// A path that cannot be read, or that repeats one already stacked, appears in
/// the `frames` array with `accepted` false and a `reason`, and the run carries
/// on. Check `failed` rather than assuming an absent error means every frame
/// landed. There is no cancellation here: a C caller wanting that should offer
/// paths in batches.
///
/// # Safety
/// `stacker` must be a live pointer returned by a `seiza_live_stacker_*`
/// constructor. `paths_json` must be a valid NUL-terminated string. When
/// non-null, `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_live_stacker_push_fits_pipelined_json(
    stacker: *mut SeizaLiveStacker,
    paths_json: *const c_char,
    workers: usize,
    max_in_flight_bytes: usize,
    normalized_full_scale: f32,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let paths_json = required_str(paths_json, "stack frame paths")?;
        let paths: Vec<PathBuf> = serde_json::from_str::<Vec<String>>(&paths_json)
            .map_err(|error| format!("stack frame paths must be a JSON array of strings: {error}"))?
            .into_iter()
            .map(PathBuf::from)
            .collect();
        let stacker = unsafe { required_live_stacker_mut(stacker)? };

        let mut options = seiza_stacking::PipelineOptions {
            workers: (workers > 0).then_some(workers),
            ..seiza_stacking::PipelineOptions::default()
        };
        if max_in_flight_bytes > 0 {
            options.max_in_flight_bytes = max_in_flight_bytes;
        }
        if normalized_full_scale > 0.0 {
            options.normalized_full_scale = Some(normalized_full_scale);
        }

        let mut frames = Vec::with_capacity(paths.len());
        let report = stacker
            .stacker
            .push_fits_pipelined(&paths, &options, |path, outcome| {
                frames.push(match outcome {
                    Ok(disposition) => stack_disposition_response(Some(path), disposition),
                    Err(error) => StackDispositionResponse {
                        source: Some(path.to_string_lossy().into_owned()),
                        accepted: false,
                        reason: Some(error.to_string()),
                        diagnostics: None,
                    },
                });
                seiza_stacking::Continue::Yes
            })
            .map_err(|error| error.to_string())?;

        owned_json(&StackPipelineResponse {
            frames,
            integrated: report.integrated,
            rejected: report.rejected,
            failed: report.failed,
        })
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
            input_paths: stacker.stacker.input_paths().to_vec(),
        })
    })
    .map_or(ptr::null_mut(), |snapshot| {
        Box::into_raw(Box::new(snapshot))
    })
}

#[unsafe(no_mangle)]
/// Copies only the finalized mean and scalar frame counts into an immutable
/// export owner. Unlike [`seiza_live_stacker_snapshot`], this does not clone
/// variance, per-sample coverage, or per-sample rejection maps.
///
/// The returned owner is independent of `stacker`: after this call returns it
/// may be transferred to a worker thread, written, and freed without holding
/// the live stacker's synchronization lock while ingestion continues.
///
/// # Safety
/// `stacker` must be a live `SeizaLiveStacker` pointer. When non-null,
/// `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_live_stacker_export_snapshot(
    stacker: *const SeizaLiveStacker,
    error_out: *mut *mut c_char,
) -> *mut SeizaStackExportSnapshot {
    clear_error(error_out);
    ffi_result(error_out, || {
        let stacker = unsafe { required_live_stacker(stacker)? };
        let reference_headers = stacker.stacker.reference_headers().to_vec();
        let input_paths = stacker.stacker.input_paths().to_vec();
        let snapshot = stacker
            .stacker
            .export_snapshot()
            .map_err(|error| error.to_string())?;
        Ok(SeizaStackExportSnapshot {
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
        let SeizaLiveStacker { stacker } = *live;
        let reference_headers = stacker.reference_headers().to_vec();
        let input_paths = stacker.input_paths().to_vec();
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
/// Writes a compact immutable export snapshot as an unstretched 32-bit
/// floating-point FITS, preserving compatible reference headers. The same
/// extension-based XISF behavior as [`seiza_stack_snapshot_write_fits`] is
/// retained.
///
/// This call reads only the export owner and may run on a worker thread after
/// [`seiza_live_stacker_export_snapshot`] returns; the live stacker is no
/// longer involved.
///
/// # Safety
/// `snapshot` must be a live `SeizaStackExportSnapshot` pointer. `path` must
/// be a valid NUL-terminated string. When non-null, `error_out` must point to
/// writable storage for one pointer. Do not free `snapshot` until this call
/// returns.
pub unsafe extern "C" fn seiza_stack_export_snapshot_write_fits(
    snapshot: *const SeizaStackExportSnapshot,
    path: *const c_char,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let snapshot = unsafe { required_stack_export_snapshot(snapshot)? };
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
        write_stack_export_fits_f32(path, &snapshot.snapshot, &snapshot.reference_headers)
            .map_err(|error| error.to_string())
    })
    .is_some()
}

#[unsafe(no_mangle)]
/// # Safety
/// `snapshot` must be null or a live pointer returned by
/// [`seiza_live_stacker_export_snapshot`] and must not already be freed.
pub unsafe extern "C" fn seiza_stack_export_snapshot_free(snapshot: *mut SeizaStackExportSnapshot) {
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
/// This builds its own blocking Tokio runtime for the download, so it must not
/// be called from a thread already inside an async runtime; doing so panics,
/// which this call catches and returns through `error_out`.
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
        Ok(RgbStretchMode::Auto),
        error_out,
    )
}

#[unsafe(no_mangle)]
/// Opens and renders an image with an explicit RGB stretch mode.
///
/// Mode `0` is per-channel auto, `1` is linked auto, and `2` is linear.
/// Non-RGB FITS/XISF and standard raster images ignore this setting.
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
    open_rendered_image(
        path,
        target_median,
        shadows_clip,
        max_dimension,
        RgbStretchMode::from_raw(rgb_stretch_mode),
        error_out,
    )
}

#[unsafe(no_mangle)]
/// Opens a FITS or XISF image and renders it with parameterized processing described by
/// `config_json`. The value may be one serialized `seiza-stretch`
/// `StretchConfig` (the original schema), a non-empty array of configs, or an
/// object with `stretch`, optional `background`, optional `deconvolution`,
/// optional `sample_domain`, and optional `interactive_preview` fields. Array
/// stages are applied in order using `f32` intermediates and converted to RGBA
/// only after the final stage. Background correction and deconvolution, when
/// requested, are applied to linear samples in that order. Sample-domain
/// mapping then converts physical values to the unit-linear domain expected by
/// stretch, without changing the source image or scientific outputs. Omitting
/// it retains the legacy unit-linear interpretation for file renders.
/// Interactive preview mode bounds the linear samples to `max_dimension`
/// before processing and reuses the source/background-prepared pixels across
/// stretch, domain, and deconvolution edits; full renders should leave it
/// false. Metadata reports both the requested and resolved sample domain.
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
        let (stack, background, deconvolution, interactive_preview, sample_domain) =
            request.into_parts();
        let sample_domain = sample_domain.unwrap_or_default();
        if interactive_preview {
            render_cached_interactive_preview(
                &path,
                &stack,
                background.as_ref(),
                deconvolution.as_ref(),
                &sample_domain,
                max_dimension,
            )
        } else {
            let (image, format) = open_astronomy_image(&path)?;
            render_astronomy_with_pipeline(
                image,
                format,
                &stack,
                RenderPipelineOptions {
                    background: background.as_ref(),
                    deconvolution: deconvolution.as_ref(),
                    sample_domain: &sample_domain,
                    max_dimension,
                    interactive_preview: false,
                },
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
/// linear. Non-RGB FITS/XISF and standard raster images ignore this setting.
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
/// Opens a FITS or XISF image and renders its parameterized processing stack to
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
        let (stack, background, deconvolution, interactive_preview, sample_domain) =
            request.into_parts();
        let sample_domain = sample_domain.unwrap_or_default();
        if interactive_preview {
            render_cached_interactive_preview16(
                &path,
                &stack,
                background.as_ref(),
                deconvolution.as_ref(),
                &sample_domain,
                max_dimension,
            )
        } else {
            let (image, format) = open_astronomy_image(&path)?;
            render_astronomy_with_pipeline16(
                image,
                format,
                &stack,
                RenderPipelineOptions {
                    background: background.as_ref(),
                    deconvolution: deconvolution.as_ref(),
                    sample_domain: &sample_domain,
                    max_dimension,
                    interactive_preview: false,
                },
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
    rgb_stretch_mode: Result<RgbStretchMode, String>,
    error_out: *mut *mut c_char,
) -> *mut SeizaRenderedImage {
    clear_error(error_out);
    ffi_result(error_out, || {
        render_image(
            path,
            target_median,
            shadows_clip,
            max_dimension,
            rgb_stretch_mode?,
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
        let (width, height, mut stars, raster_fallback, capture_time) =
            if is_astronomy_image_path(&path) {
                let (image, _) = open_astronomy_image(&path)?;
                let width = u32::try_from(image.width).map_err(|_| "image width is too large")?;
                let height =
                    u32::try_from(image.height).map_err(|_| "image height is too large")?;
                let capture_time = fits_capture_time(&image);
                let luma = image.to_luma_f32();
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
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AstronomyImageFormat {
    Fits,
    Xisf,
}

impl AstronomyImageFormat {
    fn name(self) -> &'static str {
        match self {
            Self::Fits => "FITS",
            Self::Xisf => "XISF",
        }
    }
}

fn astronomy_image_format(path: &Path) -> Option<AstronomyImageFormat> {
    if seiza_xisf::is_xisf_path(path) {
        return Some(AstronomyImageFormat::Xisf);
    }
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("fits")
                || extension.eq_ignore_ascii_case("fit")
                || extension.eq_ignore_ascii_case("fts")
        })
        .then_some(AstronomyImageFormat::Fits)
}

fn is_astronomy_image_path(path: &Path) -> bool {
    astronomy_image_format(path).is_some()
}

fn open_astronomy_image(path: &Path) -> Result<(FitsImage, AstronomyImageFormat), String> {
    let format = astronomy_image_format(path)
        .ok_or_else(|| format!("{} is not a FITS or XISF path", path.display()))?;
    let image = match format {
        AstronomyImageFormat::Fits => FitsImage::open(path)
            .map_err(|error| format!("failed to open {}: {error}", path.display()))?,
        AstronomyImageFormat::Xisf => seiza_xisf::open(path)
            .map_err(|error| format!("failed to open {}: {error}", path.display()))?,
    };
    Ok((image, format))
}

fn render_path(
    path: &Path,
    params: &StretchParams,
    max_dimension: u32,
    rgb_stretch_mode: RgbStretchMode,
) -> Result<SeizaRenderedImage, String> {
    if is_astronomy_image_path(path) {
        let (image, format) = open_astronomy_image(path)?;
        render_astronomy_image(image, format, params, max_dimension, rgb_stretch_mode)
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
    if is_astronomy_image_path(path) {
        let (image, format) = open_astronomy_image(path)?;
        render_astronomy_image16(image, format, params, max_dimension, rgb_stretch_mode)
    } else {
        let image = image::open(path)
            .map_err(|error| format!("failed to open {}: {error}", path.display()))?;
        render_raster16(image, raster_format(path), max_dimension)
    }
}

/// Classifies a FITS image as planar RGB, Bayer-mosaicked, or mono for the
/// `colorKind` metadata field.
fn fits_color_kind(fits: &FitsImage) -> &'static str {
    if fits.planes == 3 {
        "planar-rgb"
    } else if fits.bayer_pattern().is_some() {
        "bayer"
    } else {
        "mono"
    }
}

/// Copies the FITS header cards into the JSON map shape used in render metadata.
fn fits_headers_json(fits: &FitsImage) -> Map<String, Value> {
    let mut headers = Map::new();
    for (key, value) in &fits.headers {
        headers.insert(key.clone(), header_json(value));
    }
    headers
}

fn render_astronomy_image(
    fits: FitsImage,
    source_format: AstronomyImageFormat,
    params: &StretchParams,
    max_dimension: u32,
    rgb_stretch_mode: RgbStretchMode,
) -> Result<SeizaRenderedImage, String> {
    let source_width = fits.width;
    let source_height = fits.height;
    let statistics = fits.statistics();
    let color_kind = fits_color_kind(&fits);

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

    let headers = fits_headers_json(&fits);
    let metadata = json!({
        "width": source_width,
        "height": source_height,
        "planes": fits.planes,
        "format": source_format.name(),
        "colorKind": color_kind,
        "bitsPerComponent": 8,
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

fn render_astronomy_image16(
    fits: FitsImage,
    source_format: AstronomyImageFormat,
    params: &StretchParams,
    max_dimension: u32,
    rgb_stretch_mode: RgbStretchMode,
) -> Result<SeizaRenderedImage16, String> {
    let source_width = fits.width;
    let source_height = fits.height;
    let statistics = fits.statistics();
    let color_kind = fits_color_kind(&fits);

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

    let headers = fits_headers_json(&fits);
    let metadata = json!({
        "width": source_width,
        "height": source_height,
        "planes": fits.planes,
        "format": source_format.name(),
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

/// Render an astronomy image with a parameterized `seiza-stretch` config. The
/// stretch math lives entirely in `seiza-stretch`; this only marshals linear
/// pixels into the interleaved `f32` the pipeline expects and assembles the
/// RGBA result and metadata for the requested component depth.
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
    render_astronomy_with_pipeline(
        fits,
        AstronomyImageFormat::Fits,
        stack,
        RenderPipelineOptions {
            background: None,
            deconvolution: None,
            sample_domain: &SampleDomain::UnitLinear,
            max_dimension,
            interactive_preview: false,
        },
    )
}

fn render_astronomy_with_pipeline(
    image: FitsImage,
    source_format: AstronomyImageFormat,
    stack: &StretchStack,
    options: RenderPipelineOptions<'_>,
) -> Result<SeizaRenderedImage, String> {
    let prepared = prepare_fits_render(
        image,
        source_format,
        options.background,
        options.sample_domain,
        options.max_dimension,
        options.interactive_preview,
    )?;
    render_prepared_fits(
        &prepared,
        stack,
        options.deconvolution,
        options.sample_domain,
        options.max_dimension,
        false,
    )
}

fn render_astronomy_with_pipeline16(
    image: FitsImage,
    source_format: AstronomyImageFormat,
    stack: &StretchStack,
    options: RenderPipelineOptions<'_>,
) -> Result<SeizaRenderedImage16, String> {
    let prepared = prepare_fits_render(
        image,
        source_format,
        options.background,
        options.sample_domain,
        options.max_dimension,
        options.interactive_preview,
    )?;
    render_prepared_fits16(
        &prepared,
        stack,
        options.deconvolution,
        options.sample_domain,
        options.max_dimension,
        false,
    )
}

fn prepare_fits_render(
    fits: FitsImage,
    source_format: AstronomyImageFormat,
    background: Option<&BackgroundRenderRequest>,
    sample_domain: &SampleDomain,
    max_dimension: u32,
    interactive_preview: bool,
) -> Result<PreparedFitsRender, String> {
    let source_width = fits.width;
    let source_height = fits.height;
    let source_planes = fits.planes;
    let color_kind = fits_color_kind(&fits);

    let headers = fits_headers_json(&fits);
    let (data, channels, original_input_histogram, statistics) = match sample_domain {
        SampleDomain::UnitLinear => {
            let statistics = statistics_json(&fits.statistics());
            let rgb = fits.debayer().or_else(|| fits.rgb_planes());
            let input_histogram = if let Some(rgb) = &rgb {
                input_histogram_u16_json(&rgb.data, 3, false)
            } else {
                input_histogram_u16_json(&fits.to_u16(), 1, true)
            };
            // Legacy file renders offer the stretch pipeline interleaved f32
            // samples normalized to [0, 1].
            let (data, channels) = match &rgb {
                Some(rgb) => (
                    rgb.data
                        .iter()
                        .map(|&value| f32::from(value) / f32::from(u16::MAX))
                        .collect(),
                    3,
                ),
                None => (fits.to_luma_f32(), 1),
            };
            (data, channels, input_histogram, statistics)
        }
        SampleDomain::PhysicalLinear { .. } => {
            let frame = FitsFrame::from_fits(fits, None)
                .and_then(FitsFrame::into_prepared)
                .map_err(|error| format!("failed to prepare physical render samples: {error}"))?;
            let channels = frame.image.channels;
            let data = frame.image.data;
            let input_histogram = input_histogram_scaled_f32_json(&data, channels);
            let statistics = linear_f32_statistics_json(&data, channels)?;
            (data, channels, input_histogram, statistics)
        }
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
        fit.correct_in_place_with_strength(&mut data, background.mode, background.strength)
            .map_err(|error| format!("failed to correct image background: {error}"))?;
        let metadata = json!({
            "mode": background.mode,
            "strength": background.strength,
            "model": fit.model.family_name(),
            "diagnostics": &fit.diagnostics,
            "reference": &fit.reference,
        });
        let input_histogram = match sample_domain {
            SampleDomain::UnitLinear => input_histogram_f32_json(&data, channels),
            SampleDomain::PhysicalLinear { .. } => input_histogram_scaled_f32_json(&data, channels),
        };
        (input_histogram, Some(metadata))
    } else {
        (original_input_histogram, None)
    };
    Ok(PreparedFitsRender {
        source_format: source_format.name(),
        source_width,
        source_height,
        planes: source_planes,
        color_kind,
        render_width,
        render_height,
        channels,
        data,
        validity_mask: None,
        statistics,
        input_histogram,
        background_metadata,
        headers,
        interactive_preview,
        live_stack: None,
    })
}

fn prepare_live_stack_render(
    stacker: &LiveStacker,
    background: Option<&BackgroundRenderRequest>,
    max_dimension: u32,
) -> Result<PreparedFitsRender, String> {
    let view = stacker.view();
    let source_width = view.width;
    let source_height = view.height;
    let channels = view.channels;
    let (render_width, render_height, mut data, validity_mask) =
        sample_live_stack_view(view, usize::try_from(max_dimension).unwrap_or(usize::MAX))?;
    let statistics = linear_f32_statistics_json(&data, channels)?;
    let (input_histogram, background_metadata) = if let Some(background) = background {
        let exclusion_mask = validity_mask.iter().map(|valid| !valid).collect::<Vec<_>>();
        let fit = fit_background_masked(
            &data,
            render_width,
            render_height,
            channels,
            Some(&exclusion_mask),
            &background.config,
        )
        .map_err(|error| format!("failed to fit image background: {error}"))?;
        fit.correct_in_place_with_strength(&mut data, background.mode, background.strength)
            .map_err(|error| format!("failed to correct image background: {error}"))?;
        let metadata = json!({
            "mode": background.mode,
            "strength": background.strength,
            "model": fit.model.family_name(),
            "diagnostics": &fit.diagnostics,
            "reference": &fit.reference,
        });
        (
            input_histogram_scaled_f32_json(&data, channels),
            Some(metadata),
        )
    } else {
        (input_histogram_scaled_f32_json(&data, channels), None)
    };
    let mut headers = Map::new();
    for (key, value) in stacker.reference_headers() {
        headers.insert(key.clone(), header_json(value));
    }
    let state = stacker.view();

    Ok(PreparedFitsRender {
        source_format: "Live stack",
        source_width,
        source_height,
        planes: channels,
        color_kind: if channels == 3 { "planar-rgb" } else { "mono" },
        render_width,
        render_height,
        channels,
        data,
        validity_mask: Some(validity_mask),
        statistics,
        input_histogram,
        background_metadata,
        headers,
        interactive_preview: true,
        live_stack: Some(LivePreviewMetadata {
            schema_version: 1,
            accepted_frames: state.accepted_frames,
            rejected_frames: state.rejected_frames,
            input_mode: live_stack_input_mode_name(stacker.input_mode()),
        }),
    })
}

/// Copy at most `max_dimension` squared pixels from a borrowed live view.
/// Bilinear samples ignore uncovered neighbors, so the undefined mean values
/// behind zero coverage never enter the preview.
fn sample_live_stack_view(
    view: seiza_stacking::StackView<'_>,
    max_dimension: usize,
) -> Result<(usize, usize, Vec<f32>, Vec<bool>), String> {
    if max_dimension == 0 {
        return Err("live preview maximum dimension must be positive".into());
    }
    let scale = (max_dimension as f64 / view.width.max(view.height) as f64).min(1.0);
    let output_width = ((view.width as f64 * scale).round() as usize).max(1);
    let output_height = ((view.height as f64 * scale).round() as usize).max(1);
    let output_samples = output_width
        .checked_mul(output_height)
        .and_then(|pixels| pixels.checked_mul(view.channels))
        .ok_or("live preview dimensions overflow")?;
    let mut output = vec![0.0_f32; output_samples];
    let mut validity_mask = vec![false; output_width * output_height];

    if output_width == view.width && output_height == view.height {
        for (pixel, output_pixel) in output.chunks_exact_mut(view.channels).enumerate() {
            let mut valid = true;
            for (channel, value) in output_pixel.iter_mut().enumerate() {
                let index = pixel * view.channels + channel;
                if view.coverage[index] > 0 && view.mean[index].is_finite() {
                    *value = view.mean[index];
                } else {
                    *value = f32::NAN;
                    valid = false;
                }
            }
            validity_mask[pixel] = valid;
            if !valid {
                output_pixel.fill(f32::NAN);
            }
        }
        return Ok((output_width, output_height, output, validity_mask));
    }

    let scale_x = view.width as f64 / output_width as f64;
    let scale_y = view.height as f64 / output_height as f64;
    for output_y in 0..output_height {
        let source_y = ((output_y as f64 + 0.5) * scale_y - 0.5)
            .clamp(0.0, view.height.saturating_sub(1) as f64);
        let y0 = source_y.floor() as usize;
        let y1 = (y0 + 1).min(view.height - 1);
        let wy = (source_y - y0 as f64) as f32;
        for output_x in 0..output_width {
            let source_x = ((output_x as f64 + 0.5) * scale_x - 0.5)
                .clamp(0.0, view.width.saturating_sub(1) as f64);
            let x0 = source_x.floor() as usize;
            let x1 = (x0 + 1).min(view.width - 1);
            let wx = (source_x - x0 as f64) as f32;
            let neighbors = [
                (x0, y0, (1.0 - wx) * (1.0 - wy)),
                (x1, y0, wx * (1.0 - wy)),
                (x0, y1, (1.0 - wx) * wy),
                (x1, y1, wx * wy),
            ];
            for channel in 0..view.channels {
                let mut sum = 0.0_f32;
                let mut weight = 0.0_f32;
                for (x, y, spatial_weight) in neighbors {
                    let index = (y * view.width + x) * view.channels + channel;
                    if view.coverage[index] > 0 && view.mean[index].is_finite() {
                        sum += view.mean[index] * spatial_weight;
                        weight += spatial_weight;
                    }
                }
                if weight > 0.0 {
                    output[(output_y * output_width + output_x) * view.channels + channel] =
                        sum / weight;
                } else {
                    output[(output_y * output_width + output_x) * view.channels + channel] =
                        f32::NAN;
                }
            }
            let output_pixel = output_y * output_width + output_x;
            validity_mask[output_pixel] = output
                [output_pixel * view.channels..(output_pixel + 1) * view.channels]
                .iter()
                .all(|value| value.is_finite());
            if !validity_mask[output_pixel] {
                output[output_pixel * view.channels..(output_pixel + 1) * view.channels]
                    .fill(f32::NAN);
            }
        }
    }
    Ok((output_width, output_height, output, validity_mask))
}

fn linear_f32_statistics_json(samples: &[f32], channels: usize) -> Result<Value, String> {
    let analysis = StretchAnalysis::analyze(samples, channels, 262_144)
        .map_err(|error| format!("failed to analyze linear render samples: {error}"))?;
    let statistics = analysis.linked_statistics();
    let (sum, count) = samples
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold((0.0_f64, 0_u64), |(sum, count), value| {
            (sum + f64::from(value), count + 1)
        });
    Ok(json!({
        "minimum": statistics.min,
        "maximum": statistics.max,
        "mean": (count > 0).then_some(sum / count as f64),
        "median": statistics.median,
        "mad": statistics.mad,
        "sampleCount": statistics.count,
        "scale": Value::Null,
        "normalized": Value::Null,
    }))
}

fn prepare_stretch_input<'a>(
    prepared: &'a PreparedFitsRender,
    deconvolution: Option<&DeconvolutionRenderRequest>,
    sample_domain: &SampleDomain,
) -> Result<PreparedStretchInput<'a>, String> {
    let (mut data, input_histogram, deconvolution_metadata) = if let Some(request) = deconvolution {
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
        let restored = deconvolve_masked(
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
        let input_histogram = if prepared.validity_mask.is_some()
            || matches!(sample_domain, SampleDomain::PhysicalLinear { .. })
        {
            input_histogram_scaled_f32_json(&restored.data, prepared.channels)
        } else {
            input_histogram_f32_json(&restored.data, prepared.channels)
        };
        (
            Cow::Owned(restored.data),
            input_histogram,
            Some(json!({
                "algorithmVersion": seiza_deconvolution::ALGORITHM_VERSION,
                "psfFwhmPixels": request.psf_fwhm_pixels,
                "effectivePsfFwhmPixels": effective_psf_fwhm_pixels,
                "iterations": request.iterations,
                "amount": request.amount,
                "noiseFraction": request.noise_fraction,
                "maxCorrection": request.max_correction,
                "channels": channels,
            })),
        )
    } else {
        (
            Cow::Borrowed(prepared.data.as_slice()),
            prepared.input_histogram.clone(),
            None,
        )
    };

    let resolved_sample_domain = sample_domain
        .resolve(&data, prepared.channels)
        .map_err(|error| format!("failed to resolve render sample domain: {error}"))?;
    if !matches!(resolved_sample_domain, ResolvedSampleDomain::UnitLinear) {
        resolved_sample_domain
            .apply_in_place(data.to_mut(), prepared.channels)
            .map_err(|error| format!("failed to apply render sample domain: {error}"))?;
    }

    Ok(PreparedStretchInput {
        data,
        input_histogram,
        deconvolution_metadata,
        sample_domain: resolved_sample_domain,
    })
}

fn render_prepared_fits(
    prepared: &PreparedFitsRender,
    stack: &StretchStack,
    deconvolution: Option<&DeconvolutionRenderRequest>,
    requested_sample_domain: &SampleDomain,
    max_dimension: u32,
    interactive_preview_cache_hit: bool,
) -> Result<SeizaRenderedImage, String> {
    let PreparedStretchInput {
        data,
        input_histogram,
        deconvolution_metadata,
        sample_domain: resolved_sample_domain,
    } = prepare_stretch_input(prepared, deconvolution, requested_sample_domain)?;
    let stretched = stack
        .apply_u8(&data, prepared.channels)
        .map_err(|error| error.to_string())?
        .data;
    let mut rgba: Vec<u8> = if prepared.channels == 3 {
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
    if let Some(mask) = &prepared.validity_mask {
        for (pixel, valid) in rgba.chunks_exact_mut(4).zip(mask) {
            if !valid {
                pixel.fill(0);
            }
        }
    }

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
        "format": prepared.source_format,
        "colorKind": prepared.color_kind,
        "stretchStages": stack.len(),
        "interactivePreview": prepared.interactive_preview,
        "interactivePreviewCacheHit": interactive_preview_cache_hit,
        "liveStack": prepared.live_stack,
        "backgroundProcessing": prepared.background_metadata,
        "deconvolutionProcessing": deconvolution_metadata,
        "sampleDomain": {
            "requested": requested_sample_domain,
            "resolved": resolved_sample_domain,
        },
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
    requested_sample_domain: &SampleDomain,
    max_dimension: u32,
    interactive_preview_cache_hit: bool,
) -> Result<SeizaRenderedImage16, String> {
    let PreparedStretchInput {
        data,
        input_histogram,
        deconvolution_metadata,
        sample_domain: resolved_sample_domain,
    } = prepare_stretch_input(prepared, deconvolution, requested_sample_domain)?;
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
        "format": prepared.source_format,
        "colorKind": prepared.color_kind,
        "bitsPerComponent": 16,
        "stretchStages": stack.len(),
        "interactivePreview": prepared.interactive_preview,
        "interactivePreviewCacheHit": interactive_preview_cache_hit,
        "liveStack": prepared.live_stack,
        "backgroundProcessing": prepared.background_metadata,
        "deconvolutionProcessing": deconvolution_metadata,
        "sampleDomain": {
            "requested": requested_sample_domain,
            "resolved": resolved_sample_domain,
        },
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
    sample_domain: &SampleDomain,
    max_dimension: u32,
) -> Result<SeizaRenderedImage, String> {
    let key = interactive_preview_cache_key(path, background, sample_domain, max_dimension)?;
    let cache = INTERACTIVE_PREVIEW_CACHE
        .get_or_init(|| Mutex::new(VecDeque::with_capacity(INTERACTIVE_PREVIEW_CACHE_CAPACITY)));

    if let Some(prepared) = cached_interactive_preview(cache, &key)? {
        return render_prepared_fits(
            &prepared,
            stack,
            deconvolution,
            sample_domain,
            max_dimension,
            true,
        );
    }

    let (image, format) = open_astronomy_image(path)?;
    let prepared = Arc::new(prepare_fits_render(
        image,
        format,
        background,
        sample_domain,
        max_dimension,
        true,
    )?);
    let prepared = store_interactive_preview(cache, key, prepared)?;
    render_prepared_fits(
        &prepared,
        stack,
        deconvolution,
        sample_domain,
        max_dimension,
        false,
    )
}

fn render_cached_interactive_preview16(
    path: &Path,
    stack: &StretchStack,
    background: Option<&BackgroundRenderRequest>,
    deconvolution: Option<&DeconvolutionRenderRequest>,
    sample_domain: &SampleDomain,
    max_dimension: u32,
) -> Result<SeizaRenderedImage16, String> {
    let key = interactive_preview_cache_key(path, background, sample_domain, max_dimension)?;
    let cache = INTERACTIVE_PREVIEW_CACHE
        .get_or_init(|| Mutex::new(VecDeque::with_capacity(INTERACTIVE_PREVIEW_CACHE_CAPACITY)));

    if let Some(prepared) = cached_interactive_preview(cache, &key)? {
        return render_prepared_fits16(
            &prepared,
            stack,
            deconvolution,
            sample_domain,
            max_dimension,
            true,
        );
    }

    let (image, format) = open_astronomy_image(path)?;
    let prepared = Arc::new(prepare_fits_render(
        image,
        format,
        background,
        sample_domain,
        max_dimension,
        true,
    )?);
    let prepared = store_interactive_preview(cache, key, prepared)?;
    render_prepared_fits16(
        &prepared,
        stack,
        deconvolution,
        sample_domain,
        max_dimension,
        false,
    )
}

fn interactive_preview_cache_key(
    path: &Path,
    background: Option<&BackgroundRenderRequest>,
    sample_domain: &SampleDomain,
    max_dimension: u32,
) -> Result<InteractivePreviewCacheKey, String> {
    let metadata = std::fs::metadata(path)
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?;
    Ok(InteractivePreviewCacheKey {
        path: path.canonicalize().unwrap_or_else(|_| path.to_path_buf()),
        file_size: metadata.len(),
        modified: metadata.modified().ok(),
        max_dimension,
        physical_samples: matches!(sample_domain, SampleDomain::PhysicalLinear { .. }),
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
        "bitsPerComponent": 8,
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
    let mean = if count == 0 {
        0.0
    } else {
        sum as f64 / count as f64
    };
    json!({
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "median": median,
        "mad": mad,
        "scale": 255,
        "normalized": normalized_statistics_json(
            f64::from(minimum),
            f64::from(maximum),
            mean,
            f64::from(median),
            f64::from(mad),
            255.0,
        ),
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

fn input_histogram_scaled_f32_json(samples: &[f32], stride: usize) -> Value {
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    for value in samples.iter().copied().filter(|value| value.is_finite()) {
        minimum = minimum.min(value);
        maximum = maximum.max(value);
    }
    if !minimum.is_finite() || !maximum.is_finite() {
        return histogram_json(&[0; 256], &[0; 256], &[0; 256], 0.0, 1.0);
    }
    let span = (maximum - minimum).max(f32::EPSILON);
    let bin = |value: f32| (((value - minimum) / span).clamp(0.0, 1.0) * 255.0).round() as usize;
    let mut red = [0_u64; 256];
    let mut green = [0_u64; 256];
    let mut blue = [0_u64; 256];
    for pixel in samples.chunks_exact(stride) {
        if !pixel.iter().all(|value| value.is_finite()) {
            continue;
        }
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
    histogram_json(&red, &green, &blue, f64::from(minimum), f64::from(maximum))
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

/// The five summary values on a 0-1 scale, so consumers can compare
/// statistics across render depths and native sample scales.
fn normalized_statistics_json(
    minimum: f64,
    maximum: f64,
    mean: f64,
    median: f64,
    mad: f64,
    scale: f64,
) -> Value {
    json!({
        "minimum": minimum / scale,
        "maximum": maximum / scale,
        "mean": mean / scale,
        "median": median / scale,
        "mad": mad / scale,
    })
}

fn statistics_json(statistics: &Statistics) -> Value {
    json!({
        "minimum": statistics.min,
        "maximum": statistics.max,
        "mean": statistics.mean,
        "median": statistics.median,
        "mad": statistics.mad,
        "scale": 65_535,
        "normalized": normalized_statistics_json(
            f64::from(statistics.min),
            f64::from(statistics.max),
            statistics.mean,
            f64::from(statistics.median),
            statistics.mad,
            65_535.0,
        ),
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
    optional_component_status(result.map(Some))
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
    reporter.report_phase(
        "complete",
        format!("Catalogs are ready in {}", output.display()),
        None,
        reporter.files_total,
    );
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
    let mut seen = std::collections::HashSet::new();
    for path in paths {
        if !seen.insert(path_identity(path)) {
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

fn live_stack_input_mode_name(mode: FrameInputMode) -> &'static str {
    match mode {
        FrameInputMode::CalibrateAndPrepare => "calibrate-and-prepare",
        FrameInputMode::PreparedOnly => "prepared-only",
    }
}

fn default_live_preview_sample_domain(mode: FrameInputMode) -> SampleDomain {
    match mode {
        FrameInputMode::CalibrateAndPrepare => SampleDomain::PhysicalLinear {
            normalization: Default::default(),
        },
        FrameInputMode::PreparedOnly => SampleDomain::UnitLinear,
    }
}

fn probe_frame_header(path: &Path) -> Result<FrameProbeResponse, String> {
    let format = astronomy_image_format(path)
        .ok_or_else(|| format!("{} is not a FITS or XISF path", path.display()))?;
    let headers = match format {
        AstronomyImageFormat::Fits => seiza_fits::read_header(path)
            .map_err(|error| format!("failed to read FITS header {}: {error}", path.display()))?,
        AstronomyImageFormat::Xisf => seiza_xisf::read_header(path)
            .map_err(|error| format!("failed to read XISF header {}: {error}", path.display()))?,
    };
    let metadata = seiza_stacking::FrameMetadata::from_headers(&headers);
    let raw_image_type = probe_header_text(&headers, &["IMAGETYP", "OBSTYPE", "FRAME"]);

    Ok(FrameProbeResponse {
        schema_version: 1,
        path: path.to_string_lossy().into_owned(),
        format: format.name(),
        role: CalibrationFrameRole::from(metadata.role),
        raw_image_type,
        is_master: metadata.is_master,
        signature: FrameProbeSignature::from(&metadata.signature),
        calibration_state: FrameCalibrationStateResponse {
            bias_subtracted: metadata.calibration_state.bias_subtracted,
            dark_subtracted: metadata.calibration_state.dark_subtracted,
            flat_normalized: metadata.calibration_state.flat_normalized,
        },
    })
}

fn probe_header_text(headers: &[(String, HeaderValue)], keys: &[&str]) -> Option<String> {
    let value = keys.iter().find_map(|key| {
        headers
            .iter()
            .find(|(candidate, _)| candidate.eq_ignore_ascii_case(key))
            .map(|(_, value)| value)
    })?;
    let value = match value {
        HeaderValue::String(value) | HeaderValue::Raw(value) => value.trim(),
        _ => return None,
    };
    (!value.is_empty()).then(|| value.to_owned())
}

fn build_calibration_plan(
    request: CalibrationPlanRequest,
) -> Result<CalibrationPlanResponse, String> {
    let CalibrationPlanRequest {
        kind,
        reference: primary_record,
        references,
        candidates,
        minimum,
        tolerances,
        dependencies,
    } = request;
    let minimum = minimum.max(1);
    let tolerances = resolve_calibration_plan_tolerances(tolerances)?;
    if primary_record.path.trim().is_empty() {
        return Err("primary calibration reference path must not be empty".into());
    }
    let expected_reference_role = if kind == CalibrationPlanKind::DarkFlat {
        CalibrationFrameRole::Flat
    } else {
        CalibrationFrameRole::Light
    };
    let references = if references.is_empty() {
        vec![primary_record.clone()]
    } else {
        references
    };
    let reference_paths = references
        .iter()
        .map(|record| PathBuf::from(&record.path))
        .collect::<Vec<_>>();
    if reference_paths
        .iter()
        .any(|path| path.as_os_str().is_empty())
    {
        return Err("calibration reference paths must not be empty".into());
    }
    validate_distinct_stack_paths(&reference_paths)
        .map_err(|error| error.replace("stack input", "calibration reference"))?;
    let primary_in_references = references.iter().find(|record| {
        paths_refer_to_same_file(Path::new(&record.path), Path::new(&primary_record.path))
    });
    let Some(primary_in_references) = primary_in_references else {
        return Err("calibration references must include the primary reference path".into());
    };
    if primary_in_references.role != primary_record.role
        || primary_in_references.signature != primary_record.signature
    {
        return Err("the primary calibration reference must match its record in references".into());
    }
    if primary_record.role != expected_reference_role
        || references
            .iter()
            .any(|record| record.role != expected_reference_role)
    {
        return Err(format!(
            "{} calibration references must all have role {}",
            kind.as_str(),
            match expected_reference_role {
                CalibrationFrameRole::Flat => "flat",
                _ => "light",
            }
        ));
    }
    let reference = seiza_calibration::FrameSignature::from(&primary_record.signature);
    let reference_signatures = references
        .iter()
        .map(|record| seiza_calibration::FrameSignature::from(&record.signature))
        .collect::<Vec<_>>();
    let scalable_dark = kind.uses_dark_matching() && dependencies.bias_available;
    let dark_targets_have_exposure = !kind.uses_dark_matching()
        || reference_signatures
            .iter()
            .all(has_positive_signature_exposure);

    let paths = candidates
        .iter()
        .map(|candidate| PathBuf::from(&candidate.path))
        .collect::<Vec<_>>();
    if paths.iter().any(|path| path.as_os_str().is_empty()) {
        return Err("calibration candidate paths must not be empty".into());
    }
    validate_distinct_stack_paths(&paths)
        .map_err(|error| error.replace("stack input", "calibration candidate"))?;

    let mut excluded = Vec::new();
    let mut matched = Vec::new();
    for record in candidates {
        let signature = seiza_calibration::FrameSignature::from(&record.signature);
        let reason = if record.role != kind.role() {
            Some("role-mismatch")
        } else if !reference_signatures
            .iter()
            .all(|light| seiza_calibration::sensor_matches(light, &signature))
        {
            Some("sensor-mismatch")
        } else if kind.uses_dark_matching()
            && (!dark_targets_have_exposure || !has_positive_signature_exposure(&signature))
        {
            Some("missing-exposure")
        } else if kind.uses_dark_matching()
            && !scalable_dark
            && !reference_signatures
                .iter()
                .all(|light| seiza_calibration::exposure_matches(light, &signature, &tolerances))
        {
            Some("exposure-mismatch")
        } else if kind.uses_dark_matching()
            && !reference_signatures
                .iter()
                .all(|light| seiza_calibration::temperature_matches(light, &signature, &tolerances))
        {
            Some("temperature-mismatch")
        } else if kind == CalibrationPlanKind::Flat
            && !reference_signatures
                .iter()
                .all(|light| seiza_calibration::optics_match(light, &signature, &tolerances))
        {
            Some("optics-mismatch")
        } else {
            None
        };
        if let Some(reason) = reason {
            excluded.push(CalibrationPlanExclusionResponse {
                path: record.path,
                reason,
            });
        } else {
            matched.push((record, signature));
        }
    }

    let mut ordered_signatures = matched
        .iter()
        .map(|(_, signature)| signature.clone())
        .collect::<Vec<_>>();
    seiza_calibration::sort_by_proximity(&mut ordered_signatures, reference.captured_at_unix);
    let mut unordered = matched;
    let mut ordered = Vec::with_capacity(unordered.len());
    for signature in ordered_signatures {
        let index = unordered
            .iter()
            .position(|(_, candidate)| *candidate == signature)
            .ok_or_else(|| "calibration plan ordering lost a candidate".to_string())?;
        ordered.push(unordered.remove(index));
    }
    let matched_paths = ordered
        .iter()
        .map(|(record, _)| record.path.clone())
        .collect::<Vec<_>>();
    let ordered_signatures = ordered
        .iter()
        .map(|(_, signature)| signature.clone())
        .collect::<Vec<_>>();
    let coherent = coherent_calibration_signatures(&ordered_signatures, kind, minimum, &tolerances);
    let mut selected = vec![false; ordered.len()];
    for signature in coherent {
        if let Some(index) = ordered
            .iter()
            .enumerate()
            .find(|(index, (_, candidate))| !selected[*index] && *candidate == signature)
            .map(|(index, _)| index)
        {
            selected[index] = true;
        }
    }
    let mut selected_paths = Vec::new();
    for (index, (record, _)) in ordered.into_iter().enumerate() {
        if selected[index] {
            selected_paths.push(record.path);
        } else {
            excluded.push(CalibrationPlanExclusionResponse {
                path: record.path,
                reason: "outside-coherent-set",
            });
        }
    }

    Ok(CalibrationPlanResponse {
        schema_version: 1,
        kind: kind.as_str(),
        minimum,
        ready: selected_paths.len() >= minimum,
        matched_paths,
        selected_paths,
        excluded,
    })
}

fn has_positive_signature_exposure(signature: &seiza_calibration::FrameSignature) -> bool {
    signature
        .exposure_seconds
        .is_some_and(|value| value.is_finite() && value > 0.0)
}

fn coherent_calibration_signatures(
    signatures: &[seiza_calibration::FrameSignature],
    kind: CalibrationPlanKind,
    minimum: usize,
    tolerances: &seiza_calibration::MatchTolerances,
) -> Vec<seiza_calibration::FrameSignature> {
    let mut first = None;
    for anchor in signatures {
        let compatible_group = signatures
            .iter()
            .filter(|candidate| {
                internally_compatible_calibration_signatures(anchor, candidate, kind, tolerances)
            })
            .cloned()
            .collect::<Vec<_>>();
        let cluster = seiza_calibration::coherent_subset(
            &compatible_group,
            if kind == CalibrationPlanKind::Flat {
                seiza_calibration::FrameRole::Flat
            } else {
                seiza_calibration::FrameRole::Other
            },
            minimum,
            tolerances,
        );
        if cluster.len() >= minimum.max(1) {
            return cluster;
        }
        if first.is_none() {
            first = Some(cluster);
        }
    }
    first.unwrap_or_default()
}

fn internally_compatible_calibration_signatures(
    left: &seiza_calibration::FrameSignature,
    right: &seiza_calibration::FrameSignature,
    kind: CalibrationPlanKind,
    tolerances: &seiza_calibration::MatchTolerances,
) -> bool {
    // Light-to-calibration matching is deliberately asymmetric: unknown
    // metadata on a light cannot rule a candidate out. A master set needs the
    // stricter answer, because two candidates with conflicting known settings
    // must never be averaged merely because every target omitted that field.
    seiza_calibration::sensor_consistent(left, right)
        && (!kind.uses_dark_matching()
            || seiza_calibration::exposure_matches(left, right, tolerances))
        && (kind != CalibrationPlanKind::Flat
            || seiza_calibration::optics_consistent(left, right, tolerances))
}

fn resolve_calibration_plan_tolerances(
    request: CalibrationPlanTolerancesRequest,
) -> Result<seiza_calibration::MatchTolerances, String> {
    let mut tolerances = seiza_calibration::MatchTolerances::default();
    let finite_nonnegative = |name: &str, value: Option<f64>, target: &mut f64| {
        if let Some(value) = value {
            if !value.is_finite() || value < 0.0 {
                return Err(format!(
                    "calibration {name} tolerance must be finite and non-negative"
                ));
            }
            *target = value;
        }
        Ok(())
    };
    finite_nonnegative(
        "exposure-seconds",
        request.exposure_seconds,
        &mut tolerances.exposure_seconds,
    )?;
    finite_nonnegative(
        "exposure-fraction",
        request.exposure_fraction,
        &mut tolerances.exposure_fraction,
    )?;
    finite_nonnegative(
        "dark-temperature",
        request.dark_temperature_c,
        &mut tolerances.dark_temperature_c,
    )?;
    finite_nonnegative(
        "master-temperature",
        request.master_temperature_c,
        &mut tolerances.master_temperature_c,
    )?;
    finite_nonnegative(
        "rotation",
        request.rotation_deg,
        &mut tolerances.rotation_deg,
    )?;
    finite_nonnegative(
        "focal-length",
        request.focal_length_mm,
        &mut tolerances.focal_length_mm,
    )?;
    if let Some(seconds) = request.flat_session_seconds {
        tolerances.flat_session_seconds = seconds;
    }
    Ok(tolerances)
}

fn build_master_request(
    request: MasterBuildRequest,
    cancellation: Option<CancelSignal>,
) -> Result<MasterBuildResponse, String> {
    let kind = request.kind.into_core();
    if request.output.as_os_str().is_empty() {
        return Err("master output path must not be empty".into());
    }
    if request
        .inputs
        .iter()
        .any(|path| path.as_os_str().is_empty())
    {
        return Err("master input paths must not be empty".into());
    }
    if request.dark.is_none() && request.dark_exposure_seconds.is_some() {
        return Err("a master-dark exposure override requires a dark path".into());
    }
    validate_optional_positive_seconds(request.dark_exposure_seconds, "master-dark exposure")?;
    validate_optional_positive_seconds(request.exposure_seconds, "master exposure")?;
    if !request.rejection.low_sigma.is_finite()
        || request.rejection.low_sigma <= 0.0
        || !request.rejection.high_sigma.is_finite()
        || request.rejection.high_sigma <= 0.0
    {
        return Err("master rejection sigmas must be positive finite numbers".into());
    }
    if let Some(filter) = request.defect_suppression {
        if kind != MasterFrameKind::Flat {
            return Err("defect suppression is supported only for flat masters".into());
        }
        if !filter.low_sigma.is_finite()
            || filter.low_sigma <= 0.0
            || !filter.high_sigma.is_finite()
            || filter.high_sigma <= 0.0
        {
            return Err("defect-suppression sigmas must be positive finite numbers".into());
        }
    }
    match kind {
        MasterFrameKind::Bias if request.bias.is_some() || request.dark.is_some() => {
            return Err("bias masters cannot use bias or dark calibration inputs".into());
        }
        MasterFrameKind::Dark if request.dark.is_some() => {
            return Err("dark masters cannot use a dark calibration input".into());
        }
        _ => {}
    }

    let mut all_paths = request.inputs.clone();
    all_paths.extend(request.bias.iter().cloned());
    all_paths.extend(request.dark.iter().cloned());
    all_paths.push(request.output.clone());
    validate_distinct_stack_paths(&all_paths)
        .map_err(|error| error.replace("stack input", "master input/output"))?;

    let bias = request
        .bias
        .as_deref()
        .map(FitsFrame::open)
        .transpose()
        .map_err(|error| error.to_string())?
        .map(|frame| {
            frame.validate_master_kind("BIAS")?;
            Ok::<_, seiza_stacking::Error>(frame.image)
        })
        .transpose()
        .map_err(|error| error.to_string())?;
    let dark = request
        .dark
        .as_deref()
        .map(FitsFrame::open)
        .transpose()
        .map_err(|error| error.to_string())?
        .map(|frame| MasterDark::from_fits_frame(frame, request.dark_exposure_seconds))
        .transpose()
        .map_err(|error| error.to_string())?;
    let defect_suppression = request
        .defect_suppression
        .map(|filter| ImpulseFilterOptions {
            low_sigma: filter.low_sigma,
            high_sigma: filter.high_sigma,
        });
    let options = MasterBuildOptions {
        rejection: MasterRejectionOptions {
            low_sigma: request.rejection.low_sigma,
            high_sigma: request.rejection.high_sigma,
        },
        exposure_seconds: request.exposure_seconds,
        bias,
        dark,
        cancel: cancellation,
        defect_suppression,
    };
    let master = build_master_from_fits(&request.inputs, kind, &options)
        .map_err(|error| error.to_string())?;
    let (inputs, skipped_inputs) = master_build_provenance(&request.inputs, &master)?;
    // Validate every piece of provenance before publishing the output. A
    // response that cannot account for its requested inputs must not leave a
    // master behind for a caller to trust despite receiving an error.
    write_master_fits_f32(&request.output, &master).map_err(|error| error.to_string())?;
    Ok(MasterBuildResponse {
        schema_version: 2,
        kind: master.kind.as_str(),
        output: request.output.to_string_lossy().into_owned(),
        width: master.image.width,
        height: master.image.height,
        channels: master.image.channels,
        requested_frames: request.inputs.len(),
        input_frames: master.input_frames,
        accepted_samples: master.accepted_samples,
        rejected_samples: master.rejected_samples,
        fallback_pixels: master.fallback_pixels,
        defect_pixels_replaced: master.defect_pixels_replaced,
        bias_subtracted: master.bias_subtracted,
        dark_subtracted: master.dark_subtracted,
        normalized: master.normalized,
        output_exposure_seconds: master.exposure_seconds,
        rejection: MasterBuildRejectionResponse {
            low_sigma: master.rejection.low_sigma,
            high_sigma: master.rejection.high_sigma,
        },
        inputs,
        skipped_inputs,
    })
}

fn master_build_provenance(
    requested_inputs: &[PathBuf],
    master: &MasterFrame,
) -> Result<
    (
        Vec<MasterBuildInputResponse>,
        Vec<MasterBuildSkippedInputResponse>,
    ),
    String,
> {
    if master.input_frames < 2 {
        return Err(format!(
            "master input accounting is invalid: only {} accepted frame(s)",
            master.input_frames
        ));
    }
    if master.input_frames != master.input_statistics.len() {
        return Err(format!(
            "master input accounting is invalid: {} accepted frames but {} per-input tallies",
            master.input_frames,
            master.input_statistics.len()
        ));
    }
    let accounted_frames = master
        .input_frames
        .checked_add(master.skipped_inputs.len())
        .ok_or_else(|| "master input accounting overflowed".to_string())?;
    if accounted_frames != requested_inputs.len() {
        return Err(format!(
            "master input accounting is invalid: {} requested, {} accepted, and {} skipped",
            requested_inputs.len(),
            master.input_frames,
            master.skipped_inputs.len()
        ));
    }
    for (index, skipped) in master.skipped_inputs.iter().enumerate() {
        let requested_matches = requested_inputs
            .iter()
            .filter(|path| **path == skipped.path)
            .count();
        if requested_matches != 1 {
            return Err(format!(
                "master input accounting is invalid: skipped path {} occurs {requested_matches} times in the request",
                skipped.path.display()
            ));
        }
        if master.skipped_inputs[..index]
            .iter()
            .any(|previous| previous.path == skipped.path)
        {
            return Err(format!(
                "master input accounting is invalid: skipped path {} was reported more than once",
                skipped.path.display()
            ));
        }
    }

    let mut statistics = master.input_statistics.iter();
    let mut inputs = Vec::with_capacity(master.input_frames);
    for path in requested_inputs {
        if master
            .skipped_inputs
            .iter()
            .any(|skipped| skipped.path == *path)
        {
            continue;
        }
        let tally = statistics.next().ok_or_else(|| {
            "master input accounting is invalid: an accepted path has no per-input tally"
                .to_string()
        })?;
        inputs.push(MasterBuildInputResponse {
            path: path.to_string_lossy().into_owned(),
            accepted_samples: tally.accepted_samples,
            rejected_samples: tally.rejected_samples,
        });
    }
    if statistics.next().is_some() || inputs.len() != master.input_frames {
        return Err(format!(
            "master input accounting is invalid: mapped {} accepted paths for {} frames",
            inputs.len(),
            master.input_frames
        ));
    }
    let accepted_samples = master
        .input_statistics
        .iter()
        .try_fold(0_u64, |total, tally| {
            total.checked_add(tally.accepted_samples)
        })
        .ok_or_else(|| "master accepted-sample accounting overflowed".to_string())?;
    let rejected_samples = master
        .input_statistics
        .iter()
        .try_fold(0_u64, |total, tally| {
            total.checked_add(tally.rejected_samples)
        })
        .ok_or_else(|| "master rejected-sample accounting overflowed".to_string())?;
    if accepted_samples != master.accepted_samples || rejected_samples != master.rejected_samples {
        return Err(format!(
            "master sample accounting is invalid: per-input totals are {accepted_samples} accepted/{rejected_samples} rejected but the master reports {}/{}",
            master.accepted_samples, master.rejected_samples
        ));
    }

    let skipped_inputs = master
        .skipped_inputs
        .iter()
        .map(|skipped| MasterBuildSkippedInputResponse {
            path: skipped.path.to_string_lossy().into_owned(),
            reason: skipped.reason.clone(),
        })
        .collect();
    Ok((inputs, skipped_inputs))
}

fn validate_optional_positive_seconds(value: Option<f64>, name: &str) -> Result<(), String> {
    if value.is_some_and(|value| !value.is_finite() || value <= 0.0) {
        return Err(format!("{name} must be a positive finite number"));
    }
    Ok(())
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CompatibleCalibrationResponse {
    schema_version: u32,
    kept: Vec<&'static str>,
    dropped: Vec<DroppedMaster>,
}

#[derive(Serialize)]
struct DroppedMaster {
    kind: &'static str,
    reason: String,
}

/// The master kinds a set actually holds, in the order a reader expects.
fn active_master_kinds(masters: &seiza_stacking::CalibrationMasters) -> Vec<&'static str> {
    let mut kinds = Vec::new();
    if masters.has_bias() {
        kinds.push("bias");
    }
    if masters.has_dark() {
        kinds.push("dark");
    }
    if masters.has_flat() {
        kinds.push("flat");
    }
    kinds
}

/// Options for [`seiza_stars_detect_luma_u16_json`] and
/// [`seiza_stars_detect_path_json`]. Everything optional; unknown fields are
/// an error so a typo cannot silently run the defaults.
#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct StarDetectOptions {
    preset: Option<String>,
    focal_length_mm: Option<f64>,
    pixel_size_um: Option<f64>,
    psf_type: Option<String>,
    structure_removal: Option<String>,
    detection_binning: Option<usize>,
    keep_saturated: Option<bool>,
    noise_reduction_radius: Option<usize>,
    sensitivity: Option<f64>,
    /// Optional first adjustment-screw axis for triangle tilt analysis. Any
    /// finite degree value is accepted and normalized in the response.
    triangle_angle_degrees: Option<f64>,
    /// When present and positive, detection retries with progressively more
    /// permissive settings until it measures at least this many stars
    /// (ASTAP-style), returning the best pass otherwise. Absent or zero runs
    /// the single configured pass exactly as before.
    target_star_count: Option<usize>,
}

const STAR_FOCAL_LENGTH_HEADER_KEYS: &[&str] = &["FOCALLEN", "FOCALLENGTH", "FOCAL"];
const STAR_PIXEL_SIZE_HEADER_KEYS: &[&str] = &["XPIXSZ"];

impl StarDetectOptions {
    /// Fill only the detector classification inputs the caller omitted.
    /// An explicit preset is a complete classification choice and therefore
    /// never consults frame metadata.
    fn with_frame_headers(mut self, image: &FitsImage) -> Self {
        if self.preset.is_some() {
            return self;
        }
        if self.focal_length_mm.is_none() {
            self.focal_length_mm = first_positive_header(image, STAR_FOCAL_LENGTH_HEADER_KEYS);
        }
        if self.pixel_size_um.is_none() {
            self.pixel_size_um = first_positive_header(image, STAR_PIXEL_SIZE_HEADER_KEYS);
        }
        self
    }

    fn into_params(
        self,
    ) -> Result<seiza_stars::hocus_focus_star_detection::HocusFocusParams, String> {
        use seiza_stars::hocus_focus_star_detection::{
            HocusFocusParams, StructureRemovalMethod, TelescopeClass,
        };
        let mut params = match self.preset.as_deref() {
            Some("widefield") => HocusFocusParams::for_telescope_class(TelescopeClass::WideField),
            Some("standard") => HocusFocusParams::for_telescope_class(TelescopeClass::Standard),
            Some("longfocal") => {
                HocusFocusParams::for_telescope_class(TelescopeClass::LongFocalLength)
            }
            Some(other) => {
                return Err(format!(
                    "unknown preset {other:?} (widefield, standard, longfocal)"
                ));
            }
            None => HocusFocusParams::for_frame_headers(self.focal_length_mm, self.pixel_size_um).0,
        };
        params.psf_type = match self.psf_type.as_deref() {
            None | Some("moffat4") => seiza_stars::psf_fitting::PSFType::Moffat4,
            Some("gaussian") => seiza_stars::psf_fitting::PSFType::Gaussian,
            Some("none") => seiza_stars::psf_fitting::PSFType::None,
            Some(other) => {
                return Err(format!(
                    "unknown psfType {other:?} (none, gaussian, moffat4)"
                ));
            }
        };
        if let Some(method) = self.structure_removal.as_deref() {
            params.structure_removal = match method {
                "filtered" => StructureRemovalMethod::Filtered,
                "atrous" => StructureRemovalMethod::Atrous,
                other => {
                    return Err(format!(
                        "unknown structureRemoval {other:?} (filtered, atrous)"
                    ));
                }
            };
        }
        if let Some(binning) = self.detection_binning {
            params.detection_binning = binning.max(1);
        }
        if let Some(keep) = self.keep_saturated {
            params.keep_saturated_stars = keep;
        }
        if let Some(radius) = self.noise_reduction_radius {
            params.noise_reduction_radius = radius;
        }
        if let Some(value) = self.sensitivity {
            params.sensitivity = value;
        }
        Ok(params)
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DetectedStarJson {
    x: f64,
    y: f64,
    hfr: f64,
    fwhm: f64,
    brightness: f64,
    background: f64,
    snr: f64,
    flux: f64,
    pixel_count: usize,
    saturated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    eccentricity: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    theta: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    r_squared: Option<f64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TiltCellJson {
    row: usize,
    col: usize,
    star_count: usize,
    median_hfr: Option<f64>,
    median_eccentricity: Option<f64>,
    mean_theta: Option<f64>,
    theta_coherence: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TiltCornerJson {
    corner: &'static str,
    hfr: Option<f64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TiltSummaryJson {
    center_hfr: Option<f64>,
    corners: Vec<TiltCornerJson>,
    mean_hfr: Option<f64>,
    tilt_percent: Option<f64>,
    curvature_percent: Option<f64>,
    worst_corner: Option<&'static str>,
    best_corner: Option<&'static str>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TriangleCenterJson {
    star_count: usize,
    median_hfr: Option<f64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TriangleSectorJson {
    sector: u8,
    axis_angle_degrees: f64,
    star_count: usize,
    median_hfr: Option<f64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TriangleTiltJson {
    angle_degrees: f64,
    inner_radius_pixels: f64,
    outer_radius_pixels: f64,
    minimum_stars_per_region: usize,
    ready: bool,
    center: TriangleCenterJson,
    sectors: Vec<TriangleSectorJson>,
    overall_median_hfr: Option<f64>,
    tilt_percent: Option<f64>,
    best_sector: Option<u8>,
    worst_sector: Option<u8>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StarDetectResponse {
    schema_version: u32,
    width: usize,
    height: usize,
    major_axis_orientations_normalized: bool,
    average_hfr: f64,
    average_fwhm: f64,
    noise_sigma: f64,
    background_mean: f64,
    stars: Vec<DetectedStarJson>,
    cells: Vec<TiltCellJson>,
    tilt: TiltSummaryJson,
    #[serde(skip_serializing_if = "Option::is_none")]
    triangle_tilt: Option<TriangleTiltJson>,
}

fn first_positive_header(image: &FitsImage, keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|key| {
        image
            .headers
            .iter()
            .filter(|(candidate, _)| candidate.eq_ignore_ascii_case(key))
            .filter_map(|(_, value)| value.as_f64())
            .find(|value| value.is_finite() && *value > 0.0)
    })
}

unsafe fn parse_star_detect_options(
    options_json: *const c_char,
) -> Result<StarDetectOptions, String> {
    if options_json.is_null() {
        return Ok(StarDetectOptions::default());
    }
    let text = unsafe { CStr::from_ptr(options_json) }
        .to_str()
        .map_err(|_| "options_json is not valid UTF-8".to_string())?;
    serde_json::from_str(text).map_err(|error| error.to_string())
}

/// Linear 16-bit luminance for measurement. Mono u16 stays borrowed; planar
/// RGB is collapsed to luminance, and a raw Bayer mosaic is debayered before
/// its luminance is measured.
fn astronomy_luma_u16(image: &FitsImage) -> Cow<'_, [u16]> {
    match image.debayer() {
        Some(rgb) => Cow::Owned(rgb.to_luma_u16()),
        None => image.to_u16(),
    }
}

fn detect_stars_response(
    samples: &[u16],
    width: usize,
    height: usize,
    options: StarDetectOptions,
) -> Result<StarDetectResponse, String> {
    let expected = width
        .checked_mul(height)
        .ok_or_else(|| format!("image dimensions {width}x{height} overflow"))?;
    if samples.len() != expected {
        return Err(format!(
            "data length {} does not match {width}x{height}",
            samples.len()
        ));
    }
    let triangle_angle_degrees = options.triangle_angle_degrees;
    let target_star_count = options.target_star_count;
    let params = options.into_params()?;
    let result = match target_star_count {
        Some(target) if target > 0 => {
            seiza_stars::hocus_focus_star_detection::detect_stars_hocus_focus_adaptive(
                samples, width, height, &params, target,
            )
        }
        _ => seiza_stars::hocus_focus_star_detection::detect_stars_hocus_focus(
            samples, width, height, &params,
        ),
    };
    let stars: Vec<DetectedStarJson> = result
        .stars
        .iter()
        .map(|star| DetectedStarJson {
            x: star.position.0,
            y: star.position.1,
            hfr: star.hfr,
            fwhm: star.fwhm,
            brightness: star.brightness,
            background: star.background,
            snr: star.snr,
            flux: star.flux,
            pixel_count: star.pixel_count,
            saturated: star.saturated,
            eccentricity: star.psf_model.as_ref().map(|psf| psf.eccentricity),
            theta: star
                .psf_model
                .as_ref()
                .map(seiza_stars::psf_fitting::PSFModel::major_axis_theta),
            r_squared: star.psf_model.as_ref().map(|psf| psf.r_squared),
        })
        .collect();
    let tilt_stars: Vec<seiza_stars::tilt::TiltStar> = stars
        .iter()
        .map(|star| seiza_stars::tilt::TiltStar {
            x: star.x,
            y: star.y,
            hfr: star.hfr,
            eccentricity: star.eccentricity.unwrap_or(0.0),
            theta: star.theta,
        })
        .collect();
    let cells = seiza_stars::tilt::analyze_cells(&tilt_stars, width, height);
    let summary = seiza_stars::tilt::tilt_summary(&cells);
    let triangle_tilt = triangle_angle_degrees
        .map(|angle| seiza_stars::tilt::analyze_triangle(&tilt_stars, width, height, angle))
        .transpose()
        .map_err(|error| error.to_string())?
        .map(|triangle| TriangleTiltJson {
            angle_degrees: triangle.angle_degrees,
            inner_radius_pixels: triangle.inner_radius_pixels,
            outer_radius_pixels: triangle.outer_radius_pixels,
            minimum_stars_per_region: triangle.minimum_stars_per_region,
            ready: triangle.ready,
            center: TriangleCenterJson {
                star_count: triangle.center.star_count,
                median_hfr: triangle.center.median_hfr,
            },
            sectors: triangle
                .sectors
                .iter()
                .map(|sector| TriangleSectorJson {
                    sector: sector.sector,
                    axis_angle_degrees: sector.axis_angle_degrees,
                    star_count: sector.star_count,
                    median_hfr: sector.median_hfr,
                })
                .collect(),
            overall_median_hfr: triangle.overall_median_hfr,
            tilt_percent: triangle.tilt_percent,
            best_sector: triangle.best_sector,
            worst_sector: triangle.worst_sector,
        });
    Ok(StarDetectResponse {
        schema_version: 1,
        width,
        height,
        major_axis_orientations_normalized: true,
        average_hfr: result.average_hfr,
        average_fwhm: result.average_fwhm,
        noise_sigma: result.noise_sigma,
        background_mean: result.background_mean,
        stars,
        cells: cells
            .iter()
            .map(|cell| TiltCellJson {
                row: cell.row,
                col: cell.col,
                star_count: cell.star_count,
                median_hfr: cell.median_hfr,
                median_eccentricity: cell.median_eccentricity,
                mean_theta: cell.mean_theta,
                theta_coherence: cell.theta_coherence,
            })
            .collect(),
        tilt: TiltSummaryJson {
            center_hfr: summary.center_hfr,
            corners: summary
                .corners
                .iter()
                .map(|corner| TiltCornerJson {
                    corner: corner.corner.as_str(),
                    hfr: corner.hfr,
                })
                .collect(),
            mean_hfr: summary.mean_hfr,
            tilt_percent: summary.tilt_percent,
            curvature_percent: summary.curvature_percent,
            worst_corner: summary.worst_corner.map(seiza_stars::tilt::Corner::as_str),
            best_corner: summary.best_corner.map(seiza_stars::tilt::Corner::as_str),
        },
        triangle_tilt,
    })
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

unsafe fn required_stack_export_snapshot<'a>(
    snapshot: *const SeizaStackExportSnapshot,
) -> Result<&'a SeizaStackExportSnapshot, String> {
    unsafe { snapshot.as_ref() }.ok_or_else(|| "stack export snapshot is required".into())
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

    /// A tolerance a caller cannot express is one that falls back, not one
    /// that silently matches nothing.
    #[test]
    fn tolerance_overrides_fall_back_when_they_cannot_decide() {
        let defaults = seiza_calibration::MatchTolerances::default();

        // No struct at all, and a zeroed one, both mean "every default".
        assert_eq!(unsafe { match_tolerances(ptr::null()) }, defaults);
        let zeroed = SeizaMatchTolerances::default();
        assert_eq!(unsafe { match_tolerances(&zeroed) }, defaults);

        // A flag set to a usable value is taken.
        let mut tuned = SeizaMatchTolerances {
            known: SEIZA_TOLERANCE_HAS_ROTATION,
            rotation_deg: 5.0,
            ..Default::default()
        };
        assert_eq!(unsafe { match_tolerances(&tuned) }.rotation_deg, 5.0);
        // ...and only that one.
        assert_eq!(
            unsafe { match_tolerances(&tuned) }.exposure_seconds,
            defaults.exposure_seconds
        );

        // Zero is a real ask: these must be exactly equal.
        tuned.rotation_deg = 0.0;
        assert_eq!(unsafe { match_tolerances(&tuned) }.rotation_deg, 0.0);

        // Neither of these can decide anything, so neither is taken.
        // `(a - b).abs() <= -1.0` is false for every pair, which would match
        // nothing and explain nothing.
        for nonsense in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            tuned.rotation_deg = nonsense;
            assert_eq!(
                unsafe { match_tolerances(&tuned) }.rotation_deg,
                defaults.rotation_deg,
                "{nonsense} should fall back to the default"
            );
        }

        // The session window is a u64 and cannot be negative or NaN, so it is
        // taken whenever its flag is set — including zero.
        let session = SeizaMatchTolerances {
            known: SEIZA_TOLERANCE_HAS_FLAT_SESSION,
            flat_session_seconds: 0,
            ..Default::default()
        };
        assert_eq!(
            unsafe { match_tolerances(&session) }.flat_session_seconds,
            0
        );
    }

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

    fn synthetic_u16_fits(
        width: usize,
        height: usize,
        samples: &[u16],
        extra_headers: &[(&str, &str)],
    ) -> Vec<u8> {
        assert_eq!(samples.len(), width * height);
        let mut bytes = Vec::new();
        for (keyword, value) in [
            ("SIMPLE", "T".to_string()),
            ("BITPIX", "16".to_string()),
            ("NAXIS", "2".to_string()),
            ("NAXIS1", width.to_string()),
            ("NAXIS2", height.to_string()),
            ("BZERO", "32768".to_string()),
        ] {
            bytes.extend_from_slice(&card(&format!("{keyword:<8}= {value:>20}")));
        }
        for (keyword, value) in extra_headers {
            bytes.extend_from_slice(&card(&format!("{keyword:<8}= {value:>20}")));
        }
        bytes.extend_from_slice(&card("END"));
        bytes.resize(bytes.len().next_multiple_of(2880), b' ');
        for value in samples {
            bytes.extend_from_slice(&(*value ^ 0x8000).to_be_bytes());
        }
        bytes.resize(bytes.len().next_multiple_of(2880), 0);
        bytes
    }

    fn call_star_buffer(
        samples: &[u16],
        width: usize,
        height: usize,
        options: Option<&str>,
    ) -> Result<Value, String> {
        let options = options.map(|text| CString::new(text).unwrap());
        let mut error = ptr::null_mut();
        let json = unsafe {
            seiza_stars_detect_luma_u16_json(
                samples.as_ptr(),
                samples.len(),
                width,
                height,
                options.as_ref().map_or(ptr::null(), |text| text.as_ptr()),
                &mut error,
            )
        };
        if json.is_null() {
            assert!(!error.is_null());
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { seiza_string_free(error) };
            return Err(message);
        }
        assert!(error.is_null());
        let parsed = serde_json::from_str(unsafe { CStr::from_ptr(json) }.to_str().unwrap())
            .map_err(|error| error.to_string());
        unsafe { seiza_string_free(json) };
        parsed
    }

    fn call_star_path(path: &Path, options: Option<&str>) -> Result<Value, String> {
        let path = CString::new(path.to_str().unwrap()).unwrap();
        let options = options.map(|text| CString::new(text).unwrap());
        let mut error = ptr::null_mut();
        let json = unsafe {
            seiza_stars_detect_path_json(
                path.as_ptr(),
                options.as_ref().map_or(ptr::null(), |text| text.as_ptr()),
                &mut error,
            )
        };
        if json.is_null() {
            assert!(!error.is_null());
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { seiza_string_free(error) };
            return Err(message);
        }
        assert!(error.is_null());
        let parsed = serde_json::from_str(unsafe { CStr::from_ptr(json) }.to_str().unwrap())
            .map_err(|error| error.to_string());
        unsafe { seiza_string_free(json) };
        parsed
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

    /// A 128x128 channel blank across its top `blank_rows` rows.
    fn blank_topped_channel(blank_rows: usize) -> Vec<f32> {
        (0..128 * 128)
            .map(|index| {
                if index / 128 < blank_rows {
                    f32::NAN
                } else {
                    0.25
                }
            })
            .collect()
    }

    #[test]
    fn crop_report_cabi_reports_the_region_and_names_a_stray_channel() {
        let channels = [
            blank_topped_channel(0),
            blank_topped_channel(2),
            blank_topped_channel(80),
        ];
        let names = ["H-alpha", "OIII", "SII"]
            .map(|name| CString::new(name).unwrap())
            .to_vec();
        let name_pointers = names
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<*const c_char>>();
        let data_pointers = channels
            .iter()
            .map(|channel| channel.as_ptr())
            .collect::<Vec<*const f32>>();
        let crop = CString::new("inscribed").unwrap();
        let mut error = ptr::null_mut();

        let json = unsafe {
            seiza_color_crop_report_json(
                name_pointers.as_ptr(),
                data_pointers.as_ptr(),
                name_pointers.len(),
                channels[0].len(),
                128,
                128,
                1,
                crop.as_ptr(),
                &mut error,
            )
        };
        assert!(error.is_null());
        let report: Value =
            serde_json::from_str(unsafe { CStr::from_ptr(json) }.to_str().unwrap()).unwrap();
        unsafe { seiza_string_free(json) };

        assert_eq!(report["mode"], "inscribed");
        assert_eq!(report["region"]["y"], 80);
        assert_eq!(report["region"]["height"], 48);
        assert_eq!(report["grid"]["width"], 128);
        let entries = report["channels"].as_array().unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0]["name"], "H-alpha");
        assert_eq!(entries[0]["off_center"], false);
        assert_eq!(entries[2]["name"], "SII");
        assert_eq!(entries[2]["off_center"], true);
        assert!(entries[2]["center_offset_pixels"].as_f64().unwrap() > 32.0);

        let missing = unsafe {
            seiza_color_crop_report_json(
                name_pointers.as_ptr(),
                data_pointers.as_ptr(),
                name_pointers.len(),
                channels[0].len() - 1,
                128,
                128,
                1,
                crop.as_ptr(),
                &mut error,
            )
        };
        assert!(missing.is_null());
        let message = unsafe { CStr::from_ptr(error) }.to_str().unwrap();
        assert!(message.contains("expected"), "{message}");
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

        let mut half_corrected = image.clone();
        assert!(unsafe {
            seiza_background_model_correct_in_place_with_strength(
                model,
                half_corrected.as_mut_ptr(),
                half_corrected.len(),
                SEIZA_BACKGROUND_CORRECTION_SUBTRACT,
                0.5,
                &mut error,
            )
        });
        let left = height / 2 * width + 3;
        let right = height / 2 * width + width - 4;
        assert!(
            ((half_corrected[right] - half_corrected[left]) - (image[right] - image[left]) * 0.5)
                .abs()
                < 0.003
        );

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
        assert!((corrected[left] - corrected[right]).abs() < 0.003);
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
    fn lightweight_export_snapshot_writes_off_thread_while_ingestion_continues() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SeizaStackExportSnapshot>();

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
        assert!(!stacker.is_null());
        let export = unsafe { seiza_live_stacker_export_snapshot(stacker, &mut error) };
        assert!(!export.is_null());
        assert!(error.is_null());

        let later = image.iter().map(|value| value + 10.0).collect::<Vec<_>>();
        let disposition = unsafe {
            seiza_live_stacker_push_linear_json(
                stacker,
                later.as_ptr(),
                later.len(),
                width,
                height,
                1,
                &mut error,
            )
        };
        assert!(!disposition.is_null());
        unsafe { seiza_string_free(disposition) };
        assert_eq!(unsafe { seiza_live_stacker_accepted_frames(stacker) }, 2);

        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("frozen-export.fits");
        let output_for_worker = output.clone();
        let export = unsafe { Box::from_raw(export) };
        std::thread::spawn(move || {
            let export = Box::into_raw(export);
            let output_c = CString::new(output_for_worker.to_str().unwrap()).unwrap();
            let mut error = ptr::null_mut();
            assert!(unsafe {
                seiza_stack_export_snapshot_write_fits(export, output_c.as_ptr(), &mut error)
            });
            assert!(error.is_null());
            unsafe { seiza_stack_export_snapshot_free(export) };
        })
        .join()
        .unwrap();

        let written = FitsImage::open(&output).unwrap();
        assert_eq!(written.header_f64("STACKCNT"), Some(1.0));
        let seiza_fits::Pixels::F32(values) = written.pixels else {
            panic!("stack export must remain floating point");
        };
        assert_eq!(values, image, "the worker wrote the frozen mean");
        unsafe { seiza_live_stacker_free(stacker) };
    }

    #[test]
    fn stacking_cabi_checkpoints_reopens_and_continues() {
        let (width, height) = (160, 128);
        let image = stacking_star_field(width, height);
        let config = no_adjustment_stack_options();
        let directory = tempfile::tempdir().unwrap();
        let context_path = directory.path().join("live.seiza-stack");
        let context_c = CString::new(context_path.to_str().unwrap()).unwrap();
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
        assert!(!stacker.is_null());
        let disposition = unsafe {
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
        assert!(!disposition.is_null());
        unsafe { seiza_string_free(disposition) };
        assert!(unsafe {
            seiza_live_stacker_save_context(stacker, context_c.as_ptr(), &mut error)
        });
        assert!(error.is_null());
        unsafe { seiza_live_stacker_free(stacker) };

        let resumed = unsafe { seiza_live_stacker_open_context(context_c.as_ptr(), &mut error) };
        assert!(!resumed.is_null());
        assert!(error.is_null());
        assert_eq!(unsafe { seiza_live_stacker_accepted_frames(resumed) }, 2);
        let missing_path = CString::new("does-not-need-to-exist.fits").unwrap();
        let disposition = unsafe {
            seiza_live_stacker_push_fits_json(resumed, missing_path.as_ptr(), &mut error)
        };
        assert!(disposition.is_null());
        assert!(!error.is_null());
        assert!(
            unsafe { CStr::from_ptr(error) }
                .to_str()
                .unwrap()
                .contains("use push_linear")
        );
        unsafe { seiza_string_free(error) };
        error = ptr::null_mut();
        let disposition = unsafe {
            seiza_live_stacker_push_linear_json(
                resumed,
                image.as_ptr(),
                image.len(),
                width,
                height,
                1,
                &mut error,
            )
        };
        assert!(!disposition.is_null());
        unsafe { seiza_string_free(disposition) };
        assert_eq!(unsafe { seiza_live_stacker_accepted_frames(resumed) }, 3);
        unsafe { seiza_live_stacker_free(resumed) };
    }

    #[test]
    fn stacking_cabi_opens_fits_and_rejects_duplicate_paths() {
        let (width, height) = (160, 128);
        let data = stacking_star_field(width, height);
        let image = LinearImage::new(width, height, 1, data).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let first = directory.path().join("light-001.fits");
        let second = directory.path().join("light-002.fits");
        let context = directory.path().join("live.seiza-stack");
        seiza_stacking::write_processed_image_fits_f32(&first, &image, &[], &[]).unwrap();
        seiza_stacking::write_processed_image_fits_f32(&second, &image, &[], &[]).unwrap();
        let first_c = CString::new(first.to_str().unwrap()).unwrap();
        let second_c = CString::new(second.to_str().unwrap()).unwrap();
        let context_c = CString::new(context.to_str().unwrap()).unwrap();
        let config = no_adjustment_stack_options();
        let mut error = ptr::null_mut();
        let mut stacker = unsafe {
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
        assert!(unsafe {
            seiza_live_stacker_save_context(stacker, context_c.as_ptr(), &mut error)
        });
        unsafe { seiza_live_stacker_free(stacker) };
        stacker = unsafe { seiza_live_stacker_open_context(context_c.as_ptr(), &mut error) };
        assert!(!stacker.is_null());

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
    fn live_state_calibration_swap_and_bounded_preview_share_native_state() {
        let (width, height) = (160, 128);
        let reference =
            LinearImage::new(width, height, 1, stacking_star_field(width, height)).unwrap();
        let bias = LinearImage::new(width, height, 1, vec![2.0; width * height]).unwrap();
        let wrong_bias = LinearImage::new(80, 64, 1, vec![3.0; 80 * 64]).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let reference_path = directory.path().join("reference.fits");
        let bias_path = directory.path().join("master-bias.fits");
        let wrong_bias_path = directory.path().join("wrong-bias.fits");
        seiza_stacking::write_processed_image_fits_f32(&reference_path, &reference, &[], &[])
            .unwrap();
        seiza_stacking::write_processed_image_fits_f32(&bias_path, &bias, &[], &[]).unwrap();
        seiza_stacking::write_processed_image_fits_f32(&wrong_bias_path, &wrong_bias, &[], &[])
            .unwrap();
        let reference_c = CString::new(reference_path.to_str().unwrap()).unwrap();
        let bias_c = CString::new(bias_path.to_str().unwrap()).unwrap();
        let wrong_bias_c = CString::new(wrong_bias_path.to_str().unwrap()).unwrap();
        let config = no_adjustment_stack_options();
        let mut error = ptr::null_mut();
        let stacker = unsafe {
            seiza_live_stacker_open_fits(
                reference_c.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                0.0,
                config.as_ptr(),
                &mut error,
            )
        };
        assert!(!stacker.is_null());

        let read_state = |error: &mut *mut c_char| {
            let state = unsafe { seiza_live_stacker_state_json(stacker, error) };
            assert!(!state.is_null());
            let parsed =
                serde_json::from_str::<Value>(unsafe { CStr::from_ptr(state) }.to_str().unwrap())
                    .unwrap();
            unsafe { seiza_string_free(state) };
            parsed
        };
        let initial = read_state(&mut error);
        assert_eq!(initial["schemaVersion"], 1);
        assert_eq!(initial["coreVersion"], env!("CARGO_PKG_VERSION"));
        assert_eq!(initial["inputMode"], "calibrate-and-prepare");
        assert_eq!(initial["inputPaths"].as_array().unwrap().len(), 1);
        assert_eq!(initial["referenceFrame"]["isMaster"], false);
        assert_eq!(initial["referenceFrame"]["signature"]["width"], width);
        assert_eq!(initial["referenceFrame"]["signature"]["height"], height);
        assert_eq!(initial["referenceFrame"]["signature"]["channels"], 1);
        assert_eq!(
            initial["referenceFrame"]["calibrationState"]["biasSubtracted"],
            false
        );
        assert_eq!(
            initial["configurationFingerprint"].as_str().unwrap().len(),
            64
        );

        assert!(unsafe {
            seiza_live_stacker_set_calibration_fits(
                stacker,
                bias_c.as_ptr(),
                ptr::null(),
                ptr::null(),
                0.0,
                &mut error,
            )
        });
        let calibrated = read_state(&mut error);
        assert_ne!(
            calibrated["configurationFingerprint"],
            initial["configurationFingerprint"]
        );
        assert_eq!(calibrated["inputPaths"].as_array().unwrap().len(), 2);

        assert!(!unsafe {
            seiza_live_stacker_set_calibration_fits(
                stacker,
                wrong_bias_c.as_ptr(),
                ptr::null(),
                ptr::null(),
                0.0,
                &mut error,
            )
        });
        assert!(
            unsafe { CStr::from_ptr(error) }
                .to_str()
                .unwrap()
                .contains("dimensions")
        );
        unsafe { seiza_string_free(error) };
        error = ptr::null_mut();
        let after_failure = read_state(&mut error);
        assert_eq!(
            after_failure["configurationFingerprint"],
            calibrated["configurationFingerprint"]
        );
        assert_eq!(after_failure["inputPaths"], calibrated["inputPaths"]);

        let stretch = CString::new(
            r#"{"stretch":[{"model":{"type":"auto-mtf","target_median":0.2,"shadows_clip":-2.8},"color_strategy":"unlinked","max_analysis_samples":200000}]}"#,
        )
        .unwrap();
        let physical_mean_before = unsafe { (*stacker).stacker.view().mean.to_vec() };
        let preview =
            unsafe { seiza_live_stacker_render_preview(stacker, stretch.as_ptr(), 64, &mut error) };
        assert!(!preview.is_null());
        assert!(error.is_null());
        assert!(unsafe { seiza_rendered_image_width(preview) } <= 64);
        assert!(unsafe { seiza_rendered_image_height(preview) } <= 64);
        let metadata: Value = serde_json::from_str(
            unsafe { CStr::from_ptr(seiza_rendered_image_metadata_json(preview)) }
                .to_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(metadata["format"], "Live stack");
        assert_eq!(metadata["liveStack"]["acceptedFrames"], 1);
        assert_eq!(metadata["interactivePreview"], true);
        assert_eq!(
            metadata["sampleDomain"]["requested"]["type"],
            "physical-linear"
        );
        assert_eq!(
            metadata["sampleDomain"]["requested"]["normalization"]["type"],
            "robust-percentile"
        );
        assert_eq!(
            metadata["sampleDomain"]["resolved"]["type"],
            "physical-linear"
        );
        let black = metadata["sampleDomain"]["resolved"]["black"]
            .as_f64()
            .unwrap();
        let white = metadata["sampleDomain"]["resolved"]["white"]
            .as_f64()
            .unwrap();
        assert!(black.is_finite() && white.is_finite() && white > black);
        let rgba = unsafe {
            std::slice::from_raw_parts(
                seiza_rendered_image_rgba(preview),
                seiza_rendered_image_rgba_length(preview),
            )
        };
        let display_codes = rgba
            .chunks_exact(4)
            .filter(|pixel| pixel[3] != 0)
            .map(|pixel| pixel[0])
            .collect::<std::collections::BTreeSet<_>>();
        assert!(
            display_codes.len() > 16,
            "physical live preview collapsed to {} gray codes",
            display_codes.len()
        );
        assert_eq!(
            unsafe { (*stacker).stacker.view().mean },
            physical_mean_before,
            "presentation mapping changed the physical live mean"
        );
        unsafe { seiza_rendered_image_free(preview) };

        assert!(unsafe {
            seiza_live_stacker_set_calibration_fits(
                stacker,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                0.0,
                &mut error,
            )
        });
        let cleared = read_state(&mut error);
        assert_eq!(
            cleared["configurationFingerprint"],
            initial["configurationFingerprint"]
        );
        // The used-master history remains output-protective after clearing.
        assert_eq!(cleared["inputPaths"].as_array().unwrap().len(), 2);
        unsafe { seiza_live_stacker_free(stacker) };
    }

    #[test]
    fn live_preview_defaults_follow_the_stacker_input_mode() {
        assert_eq!(
            default_live_preview_sample_domain(FrameInputMode::PreparedOnly),
            SampleDomain::UnitLinear
        );
        assert_eq!(
            default_live_preview_sample_domain(FrameInputMode::CalibrateAndPrepare),
            SampleDomain::PhysicalLinear {
                normalization: Default::default(),
            }
        );

        let (width, height) = (160, 128);
        let scale = stacking_star_field(width, height)
            .into_iter()
            .map(|value| value / 3_000.0)
            .collect::<Vec<_>>();
        let options = no_adjustment_stack_options();
        let config = CString::new(
            r#"{"model":{"type":"identity"},"color_strategy":"linked","max_analysis_samples":4096}"#,
        )
        .unwrap();
        let mut error = ptr::null_mut();
        let stacker = unsafe {
            seiza_live_stacker_create(
                scale.as_ptr(),
                scale.len(),
                width,
                height,
                1,
                options.as_ptr(),
                &mut error,
            )
        };
        assert!(!stacker.is_null());
        assert!(error.is_null());
        let preview =
            unsafe { seiza_live_stacker_render_preview(stacker, config.as_ptr(), 64, &mut error) };
        assert!(!preview.is_null());
        assert!(error.is_null());
        let metadata: Value = serde_json::from_str(
            unsafe { CStr::from_ptr(seiza_rendered_image_metadata_json(preview)) }
                .to_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(metadata["liveStack"]["inputMode"], "prepared-only");
        assert_eq!(metadata["sampleDomain"]["requested"]["type"], "unit-linear");
        assert_eq!(metadata["sampleDomain"]["resolved"]["type"], "unit-linear");
        unsafe {
            seiza_rendered_image_free(preview);
            seiza_live_stacker_free(stacker);
        }
    }

    #[test]
    fn explicit_physical_sample_domain_maps_display_buffers_without_mutating_source() {
        let source = vec![0.0_f32, 16_384.0, 32_768.0, 49_152.0, 65_535.0];
        let prepared = PreparedFitsRender {
            source_format: "Live stack",
            source_width: source.len(),
            source_height: 1,
            planes: 1,
            color_kind: "mono",
            render_width: source.len(),
            render_height: 1,
            channels: 1,
            data: source.clone(),
            validity_mask: None,
            statistics: json!({}),
            input_histogram: json!({}),
            background_metadata: None,
            headers: Map::new(),
            interactive_preview: true,
            live_stack: None,
        };
        let stack = StretchStack::single(
            serde_json::from_value(json!({
                "model": { "type": "identity" },
                "color_strategy": "linked",
                "max_analysis_samples": 4096
            }))
            .unwrap(),
        );
        let sample_domain = SampleDomain::PhysicalLinear {
            normalization: seiza_stretch::SampleNormalization::ExplicitRange {
                black: 0.0,
                white: 65_535.0,
            },
        };

        let image8 =
            render_prepared_fits(&prepared, &stack, None, &sample_domain, 0, false).unwrap();
        let image16 =
            render_prepared_fits16(&prepared, &stack, None, &sample_domain, 0, false).unwrap();

        assert_eq!(prepared.data, source, "render mutated physical source data");
        assert_eq!(
            image8
                .rgba
                .chunks_exact(4)
                .map(|pixel| pixel[0])
                .collect::<Vec<_>>(),
            [0, 64, 128, 191, 255]
        );
        assert_eq!(
            image16
                .rgba
                .chunks_exact(4)
                .map(|pixel| pixel[0])
                .collect::<Vec<_>>(),
            [0, 16_384, 32_768, 49_152, 65_535]
        );
        for metadata in [&image8.metadata_json, &image16.metadata_json] {
            let metadata: Value = serde_json::from_str(metadata.to_str().unwrap()).unwrap();
            assert_eq!(
                metadata["sampleDomain"]["requested"]["type"],
                "physical-linear"
            );
            assert_eq!(
                metadata["sampleDomain"]["requested"]["normalization"]["type"],
                "explicit-range"
            );
            assert_eq!(metadata["sampleDomain"]["resolved"]["black"], 0.0);
            assert_eq!(metadata["sampleDomain"]["resolved"]["white"], 65_535.0);
        }
    }

    #[test]
    fn file_render_physical_domain_decodes_physical_samples_before_mapping() {
        let fits = FitsImage {
            width: 5,
            height: 1,
            planes: 1,
            pixels: seiza_fits::Pixels::U16(vec![0, 16_384, 32_768, 49_152, 65_535]),
            headers: Vec::new(),
        };
        let stack = StretchStack::single(
            serde_json::from_value(json!({
                "model": { "type": "identity" },
                "color_strategy": "linked",
                "max_analysis_samples": 4096
            }))
            .unwrap(),
        );
        let sample_domain = SampleDomain::PhysicalLinear {
            normalization: seiza_stretch::SampleNormalization::ExplicitRange {
                black: 0.0,
                white: 65_535.0,
            },
        };

        let image = render_astronomy_with_pipeline(
            fits,
            AstronomyImageFormat::Fits,
            &stack,
            RenderPipelineOptions {
                background: None,
                deconvolution: None,
                sample_domain: &sample_domain,
                max_dimension: 0,
                interactive_preview: false,
            },
        )
        .unwrap();

        assert_eq!(
            image
                .rgba
                .chunks_exact(4)
                .map(|pixel| pixel[0])
                .collect::<Vec<_>>(),
            [0, 64, 128, 191, 255]
        );
        let metadata: Value = serde_json::from_str(image.metadata_json.to_str().unwrap()).unwrap();
        assert_eq!(metadata["sampleDomain"]["resolved"]["black"], 0.0);
        assert_eq!(metadata["sampleDomain"]["resolved"]["white"], 65_535.0);
        assert_eq!(metadata["statistics"]["minimum"], 0.0);
        assert_eq!(metadata["statistics"]["maximum"], 65_535.0);
        assert_eq!(metadata["inputHistogram"]["lowerBound"], 0.0);
        assert_eq!(metadata["inputHistogram"]["upperBound"], 65_535.0);
    }

    #[test]
    fn stacking_cabi_refuses_double_calibration_and_incompatible_lights() {
        let (width, height) = (160, 128);
        let image = LinearImage::new(width, height, 1, stacking_star_field(width, height)).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let reference_path = directory.path().join("reference.fits");
        let bias_path = directory.path().join("master-bias.fits");
        let calibrated_path = directory.path().join("already-calibrated.fits");
        let wrong_gain_path = directory.path().join("wrong-gain.fits");
        let master_light_path = directory.path().join("master-as-light.fits");
        let raw_dark_path = directory.path().join("raw-dark-as-light.fits");
        let cards = |gain, extra: Option<(&str, HeaderValue)>| {
            let mut cards = vec![
                seiza_fits::WriteHeaderCard::new("IMAGETYP", HeaderValue::String("LIGHT".into())),
                seiza_fits::WriteHeaderCard::new("GAIN", HeaderValue::Integer(gain)),
                seiza_fits::WriteHeaderCard::new("EXPTIME", HeaderValue::Float(60.0)),
            ];
            if let Some((name, value)) = extra {
                cards.push(seiza_fits::WriteHeaderCard::new(name, value));
            }
            cards
        };
        seiza_stacking::write_processed_image_fits_f32(
            &reference_path,
            &image,
            &[],
            &cards(100, None),
        )
        .unwrap();
        seiza_stacking::write_processed_image_fits_f32(
            &bias_path,
            &LinearImage::new(width, height, 1, vec![2.0; width * height]).unwrap(),
            &[],
            &[
                seiza_fits::WriteHeaderCard::new("SEIZAMST", HeaderValue::String("BIAS".into())),
                seiza_fits::WriteHeaderCard::new("GAIN", HeaderValue::Integer(100)),
            ],
        )
        .unwrap();
        seiza_stacking::write_processed_image_fits_f32(
            &calibrated_path,
            &image,
            &[],
            &cards(100, Some(("BIASSUB", HeaderValue::Logical(true)))),
        )
        .unwrap();
        seiza_stacking::write_processed_image_fits_f32(
            &wrong_gain_path,
            &image,
            &[],
            &cards(200, None),
        )
        .unwrap();
        seiza_stacking::write_processed_image_fits_f32(
            &master_light_path,
            &image,
            &[],
            &cards(100, Some(("SEIZAMST", HeaderValue::String("DARK".into())))),
        )
        .unwrap();
        seiza_stacking::write_processed_image_fits_f32(
            &raw_dark_path,
            &image,
            &[],
            &[
                seiza_fits::WriteHeaderCard::new("IMAGETYP", HeaderValue::String("DARK".into())),
                seiza_fits::WriteHeaderCard::new("GAIN", HeaderValue::Integer(100)),
                seiza_fits::WriteHeaderCard::new("EXPTIME", HeaderValue::Float(60.0)),
            ],
        )
        .unwrap();

        let reference_c = CString::new(reference_path.to_str().unwrap()).unwrap();
        let bias_c = CString::new(bias_path.to_str().unwrap()).unwrap();
        let calibrated_c = CString::new(calibrated_path.to_str().unwrap()).unwrap();
        let wrong_gain_c = CString::new(wrong_gain_path.to_str().unwrap()).unwrap();
        let master_light_c = CString::new(master_light_path.to_str().unwrap()).unwrap();
        let raw_dark_c = CString::new(raw_dark_path.to_str().unwrap()).unwrap();
        let config = no_adjustment_stack_options();
        let mut error = ptr::null_mut();

        for (invalid_reference, reason) in [
            (&master_light_c, "master cannot be used as a light"),
            (&raw_dark_c, "dark frame cannot be used as a light"),
        ] {
            let rejected = unsafe {
                seiza_live_stacker_open_fits(
                    invalid_reference.as_ptr(),
                    bias_c.as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    0.0,
                    config.as_ptr(),
                    &mut error,
                )
            };
            assert!(rejected.is_null());
            assert!(
                unsafe { CStr::from_ptr(error) }
                    .to_str()
                    .unwrap()
                    .contains(reason)
            );
            unsafe { seiza_string_free(error) };
            error = ptr::null_mut();
        }

        let stacker = unsafe {
            seiza_live_stacker_open_fits(
                reference_c.as_ptr(),
                bias_c.as_ptr(),
                ptr::null(),
                ptr::null(),
                0.0,
                config.as_ptr(),
                &mut error,
            )
        };
        assert!(!stacker.is_null());
        for (path, reason) in [
            (&calibrated_c, "double-calibrate"),
            (&wrong_gain_c, "sensor or readout mode"),
        ] {
            let disposition =
                unsafe { seiza_live_stacker_push_fits_json(stacker, path.as_ptr(), &mut error) };
            assert!(!disposition.is_null());
            let response: Value =
                serde_json::from_str(unsafe { CStr::from_ptr(disposition) }.to_str().unwrap())
                    .unwrap();
            assert_eq!(response["accepted"], false);
            assert!(response["reason"].as_str().unwrap().contains(reason));
            unsafe { seiza_string_free(disposition) };
        }
        unsafe { seiza_live_stacker_free(stacker) };

        let no_masters = unsafe {
            seiza_live_stacker_open_fits(
                calibrated_c.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                0.0,
                config.as_ptr(),
                &mut error,
            )
        };
        assert!(!no_masters.is_null());
        unsafe { seiza_live_stacker_free(no_masters) };
    }

    #[test]
    fn live_preview_sampling_excludes_uncovered_pixels_instead_of_making_black_data() {
        let mean = vec![10.0_f32; 16];
        let mut coverage = vec![1_u32; 16];
        for index in [0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15] {
            coverage[index] = 0;
        }
        let rejected = vec![0_u32; 16];
        let view = seiza_stacking::StackView {
            width: 4,
            height: 4,
            channels: 1,
            mean: &mean,
            coverage: &coverage,
            rejected_samples: &rejected,
            accepted_frames: 2,
            rejected_frames: 0,
        };
        let (_, _, sampled, mask) = sample_live_stack_view(view, 4).unwrap();
        assert!(sampled[0].is_nan());
        assert!(!mask[0]);
        assert_eq!(sampled[5], 10.0);
        assert!(mask[5]);
    }

    #[test]
    fn frame_probe_is_header_only_and_normalizes_role_and_signature() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("header-only.fits");
        let image = LinearImage::new(2, 2, 1, vec![1.0; 4]).unwrap();
        let cards = vec![
            seiza_fits::WriteHeaderCard::new(
                "IMAGETYP",
                HeaderValue::String("Master Dark Frame".into()),
            ),
            seiza_fits::WriteHeaderCard::new("INSTRUME", HeaderValue::String("ASI2600MM".into())),
            seiza_fits::WriteHeaderCard::new("XBINNING", HeaderValue::Integer(2)),
            seiza_fits::WriteHeaderCard::new("YBINNING", HeaderValue::Integer(2)),
            seiza_fits::WriteHeaderCard::new("GAIN", HeaderValue::Integer(100)),
            seiza_fits::WriteHeaderCard::new("EXPTIME", HeaderValue::Float(60.0)),
            seiza_fits::WriteHeaderCard::new(
                "DATE-OBS",
                HeaderValue::String("2026-01-02T03:04:05Z".into()),
            ),
            seiza_fits::WriteHeaderCard::new("BIASSUB", HeaderValue::Logical(true)),
        ];
        seiza_stacking::write_processed_image_fits_f32(&path, &image, &[], &cards).unwrap();
        // Leave a complete FITS header but no pixel payload. A full decoder
        // fails; the metadata-only API must still succeed.
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(2880)
            .unwrap();
        assert!(FitsImage::open(&path).is_err());

        let path_c = CString::new(path.to_str().unwrap()).unwrap();
        let mut error = ptr::null_mut();
        let result = unsafe { seiza_probe_frame_json(path_c.as_ptr(), &mut error) };
        assert!(!result.is_null());
        assert!(error.is_null());
        let probe: Value =
            serde_json::from_str(unsafe { CStr::from_ptr(result) }.to_str().unwrap()).unwrap();
        unsafe { seiza_string_free(result) };
        assert_eq!(probe["format"], "FITS");
        assert_eq!(probe["role"], "dark");
        assert_eq!(probe["isMaster"], true);
        assert_eq!(probe["signature"]["camera"], "ASI2600MM");
        assert_eq!(probe["signature"]["width"], 2);
        assert_eq!(probe["signature"]["height"], 2);
        assert_eq!(probe["signature"]["channels"], 1);
        assert_eq!(probe["signature"]["binningX"], 2);
        assert_eq!(probe["signature"]["gain"], 100);
        assert!(probe["signature"]["capturedAtUnix"].as_i64().is_some());
        assert_eq!(probe["calibrationState"]["biasSubtracted"], true);
    }

    #[test]
    fn calibration_plan_sorts_by_proximity_and_reports_exclusions() {
        let request = CString::new(
            json!({
                "kind": "dark",
                "minimum": 2,
                "reference": {
                    "path": "light.fits",
                    "role": "light",
                    "signature": {
                        "camera": "ASI2600MM", "width": 100, "height": 80,
                        "channels": 1, "exposureSeconds": 60.0,
                        "cameraTempC": -10.0, "capturedAtUnix": 1000
                    }
                },
                "candidates": [
                    {
                        "path": "dark-later.fits", "role": "dark",
                        "signature": {
                            "camera": "ASI2600MM", "width": 100, "height": 80,
                            "channels": 1, "exposureSeconds": 60.0,
                            "cameraTempC": -9.8, "capturedAtUnix": 1100
                        }
                    },
                    {
                        "path": "dark-nearest.fits", "role": "dark",
                        "signature": {
                            "camera": "ASI2600MM", "width": 100, "height": 80,
                            "channels": 1, "exposureSeconds": 60.0,
                            "cameraTempC": -10.1, "capturedAtUnix": 1005
                        }
                    },
                    {
                        "path": "wrong-exposure.fits", "role": "dark",
                        "signature": {
                            "camera": "ASI2600MM", "width": 100, "height": 80,
                            "channels": 1, "exposureSeconds": 120.0,
                            "cameraTempC": -10.0, "capturedAtUnix": 1001
                        }
                    }
                ]
            })
            .to_string(),
        )
        .unwrap();
        let mut error = ptr::null_mut();
        let result = unsafe { seiza_calibration_plan_json(request.as_ptr(), &mut error) };
        assert!(!result.is_null());
        let plan: Value =
            serde_json::from_str(unsafe { CStr::from_ptr(result) }.to_str().unwrap()).unwrap();
        unsafe { seiza_string_free(result) };
        assert_eq!(plan["ready"], true);
        assert_eq!(plan["matchedPaths"][0], "dark-nearest.fits");
        assert_eq!(plan["selectedPaths"].as_array().unwrap().len(), 2);
        assert_eq!(plan["excluded"][0]["path"], "wrong-exposure.fits");
        assert_eq!(plan["excluded"][0]["reason"], "exposure-mismatch");
    }

    #[test]
    fn calibration_plan_matches_dark_flats_to_the_selected_flat() {
        let request = CString::new(
            json!({
                "kind": "dark-flat",
                "minimum": 2,
                "reference": {
                    "path": "flat-reference.fits",
                    "role": "flat",
                    "signature": {
                        "camera": "ASI2600MM", "width": 100, "height": 80,
                        "channels": 1, "exposureSeconds": 2.0,
                        "cameraTempC": -10.0, "capturedAtUnix": 1000
                    }
                },
                "candidates": [
                    {
                        "path": "dark-flat-nearest.fits", "role": "dark-flat",
                        "signature": {
                            "camera": "ASI2600MM", "width": 100, "height": 80,
                            "channels": 1, "exposureSeconds": 2.0,
                            "cameraTempC": -10.1, "capturedAtUnix": 1005
                        }
                    },
                    {
                        "path": "dark-flat-later.fits", "role": "dark-flat",
                        "signature": {
                            "camera": "ASI2600MM", "width": 100, "height": 80,
                            "channels": 1, "exposureSeconds": 2.0,
                            "cameraTempC": -9.8, "capturedAtUnix": 1100
                        }
                    },
                    {
                        "path": "dark-flat-warm-session.fits", "role": "dark-flat",
                        "signature": {
                            "camera": "ASI2600MM", "width": 100, "height": 80,
                            "channels": 1, "exposureSeconds": 2.0,
                            "cameraTempC": -7.5, "capturedAtUnix": 1006
                        }
                    },
                    {
                        "path": "ordinary-dark.fits", "role": "dark",
                        "signature": {
                            "camera": "ASI2600MM", "width": 100, "height": 80,
                            "channels": 1, "exposureSeconds": 2.0,
                            "cameraTempC": -10.0, "capturedAtUnix": 1001
                        }
                    },
                    {
                        "path": "wrong-exposure.fits", "role": "dark-flat",
                        "signature": {
                            "camera": "ASI2600MM", "width": 100, "height": 80,
                            "channels": 1, "exposureSeconds": 300.0,
                            "cameraTempC": -10.0, "capturedAtUnix": 1002
                        }
                    }
                ]
            })
            .to_string(),
        )
        .unwrap();
        let mut error = ptr::null_mut();
        let result = unsafe { seiza_calibration_plan_json(request.as_ptr(), &mut error) };
        assert!(!result.is_null());
        assert!(error.is_null());
        let plan: Value =
            serde_json::from_str(unsafe { CStr::from_ptr(result) }.to_str().unwrap()).unwrap();
        unsafe { seiza_string_free(result) };

        assert_eq!(plan["kind"], "dark-flat");
        assert_eq!(plan["ready"], true);
        assert_eq!(plan["matchedPaths"][0], "dark-flat-nearest.fits");
        assert_eq!(plan["selectedPaths"].as_array().unwrap().len(), 2);
        assert!(plan["excluded"].as_array().unwrap().iter().any(|entry| {
            entry["path"] == "ordinary-dark.fits" && entry["reason"] == "role-mismatch"
        }));
        assert!(plan["excluded"].as_array().unwrap().iter().any(|entry| {
            entry["path"] == "wrong-exposure.fits" && entry["reason"] == "exposure-mismatch"
        }));
        assert!(plan["excluded"].as_array().unwrap().iter().any(|entry| {
            entry["path"] == "dark-flat-warm-session.fits"
                && entry["reason"] == "outside-coherent-set"
        }));
    }

    #[test]
    fn calibration_plan_requires_one_safe_set_for_every_target() {
        let reference = json!({
            "path": "light-60.fits", "role": "light",
            "signature": {
                "camera": "ASI2600MM", "width": 100, "height": 80,
                "channels": 1, "exposureSeconds": 60.0,
                "cameraTempC": -10.0, "capturedAtUnix": 1000
            }
        });
        let references = json!([
            reference.clone(),
            {
                "path": "light-120.fits", "role": "light",
                "signature": {
                    "camera": "ASI2600MM", "width": 100, "height": 80,
                    "channels": 1, "exposureSeconds": 120.0,
                    "cameraTempC": -10.5, "capturedAtUnix": 1010
                }
            }
        ]);
        let candidates = json!([
            {
                "path": "dark-60-a.fits", "role": "dark",
                "signature": {
                    "camera": "ASI2600MM", "width": 100, "height": 80,
                    "channels": 1, "exposureSeconds": 60.0,
                    "cameraTempC": -10.0, "capturedAtUnix": 1001
                }
            },
            {
                "path": "dark-60-b.fits", "role": "dark",
                "signature": {
                    "camera": "ASI2600MM", "width": 100, "height": 80,
                    "channels": 1, "exposureSeconds": 60.0,
                    "cameraTempC": -9.8, "capturedAtUnix": 1002
                }
            },
            {
                "path": "dark-120.fits", "role": "dark",
                "signature": {
                    "camera": "ASI2600MM", "width": 100, "height": 80,
                    "channels": 1, "exposureSeconds": 120.0,
                    "cameraTempC": -10.0, "capturedAtUnix": 1003
                }
            },
            {
                "path": "dark-unknown.fits", "role": "dark",
                "signature": {
                    "camera": "ASI2600MM", "width": 100, "height": 80,
                    "channels": 1, "cameraTempC": -10.0
                }
            }
        ]);
        let invoke = |bias_available| {
            let request = CString::new(
                json!({
                    "kind": "dark", "minimum": 2,
                    "reference": reference.clone(),
                    "references": references.clone(),
                    "dependencies": {"biasAvailable": bias_available},
                    "candidates": candidates.clone()
                })
                .to_string(),
            )
            .unwrap();
            let mut error = ptr::null_mut();
            let result = unsafe { seiza_calibration_plan_json(request.as_ptr(), &mut error) };
            assert!(!result.is_null());
            assert!(error.is_null());
            let plan: Value =
                serde_json::from_str(unsafe { CStr::from_ptr(result) }.to_str().unwrap()).unwrap();
            unsafe { seiza_string_free(result) };
            plan
        };

        let unscaled = invoke(false);
        assert_eq!(unscaled["ready"], false);
        assert!(
            unscaled["excluded"]
                .as_array()
                .unwrap()
                .iter()
                .any(|entry| {
                    entry["path"] == "dark-60-a.fits" && entry["reason"] == "exposure-mismatch"
                })
        );

        let scalable = invoke(true);
        assert_eq!(scalable["ready"], true);
        assert_eq!(scalable["matchedPaths"].as_array().unwrap().len(), 3);
        assert_eq!(scalable["selectedPaths"].as_array().unwrap().len(), 2);
        assert!(
            scalable["selectedPaths"]
                .as_array()
                .unwrap()
                .iter()
                .all(|path| path.as_str().unwrap().starts_with("dark-60-"))
        );
        assert!(
            scalable["excluded"]
                .as_array()
                .unwrap()
                .iter()
                .any(|entry| {
                    entry["path"] == "dark-unknown.fits" && entry["reason"] == "missing-exposure"
                })
        );
        assert!(
            scalable["excluded"]
                .as_array()
                .unwrap()
                .iter()
                .any(|entry| {
                    entry["path"] == "dark-120.fits" && entry["reason"] == "outside-coherent-set"
                })
        );
    }

    #[test]
    fn calibration_plan_checks_flat_optics_against_every_reference() {
        let primary = json!({
            "path":"ha-light.fits", "role":"light",
            "signature":{"camera":"ASI2600MM","width":100,"height":80,
                         "channels":1,"filter":"Ha"}
        });
        let request = CString::new(
            json!({
                "kind":"flat", "minimum":1,
                "reference":primary.clone(),
                "references":[
                    primary.clone(),
                    {"path":"oiii-light.fits","role":"light",
                     "signature":{"camera":"ASI2600MM","width":100,"height":80,
                                  "channels":1,"filter":"OIII"}}
                ],
                "candidates":[
                    {"path":"ha-flat.fits","role":"flat",
                     "signature":{"camera":"ASI2600MM","width":100,"height":80,
                                  "channels":1,"filter":"Ha"}}
                ]
            })
            .to_string(),
        )
        .unwrap();
        let mut error = ptr::null_mut();
        let result = unsafe { seiza_calibration_plan_json(request.as_ptr(), &mut error) };
        assert!(!result.is_null());
        let plan: Value =
            serde_json::from_str(unsafe { CStr::from_ptr(result) }.to_str().unwrap()).unwrap();
        unsafe { seiza_string_free(result) };
        assert_eq!(plan["ready"], false);
        assert_eq!(plan["excluded"][0]["reason"], "optics-mismatch");
    }

    #[test]
    fn calibration_plan_never_combines_conflicting_sensor_settings_hidden_by_target_metadata() {
        let request = CString::new(
            json!({
                "kind":"bias", "minimum":2,
                "reference":{
                    "path":"light.fits", "role":"light",
                    "signature":{"camera":"ASI2600MM","width":100,"height":80,
                                 "channels":1}
                },
                "candidates":[
                    {"path":"bias-100-a.fits","role":"bias",
                     "signature":{"camera":"ASI2600MM","width":100,"height":80,
                                  "channels":1,"gain":100,"capturedAtUnix":1000}},
                    {"path":"bias-200.fits","role":"bias",
                     "signature":{"camera":"ASI2600MM","width":100,"height":80,
                                  "channels":1,"gain":200,"capturedAtUnix":1001}},
                    {"path":"bias-100-b.fits","role":"bias",
                     "signature":{"camera":"ASI2600MM","width":100,"height":80,
                                  "channels":1,"gain":100,"capturedAtUnix":1002}}
                ]
            })
            .to_string(),
        )
        .unwrap();
        let mut error = ptr::null_mut();
        let result = unsafe { seiza_calibration_plan_json(request.as_ptr(), &mut error) };
        assert!(!result.is_null());
        assert!(error.is_null());
        let plan: Value =
            serde_json::from_str(unsafe { CStr::from_ptr(result) }.to_str().unwrap()).unwrap();
        unsafe { seiza_string_free(result) };
        assert_eq!(plan["ready"], true);
        assert_eq!(
            plan["selectedPaths"],
            json!(["bias-100-a.fits", "bias-100-b.fits"])
        );
        assert!(plan["excluded"].as_array().unwrap().iter().any(|entry| {
            entry["path"] == "bias-200.fits" && entry["reason"] == "outside-coherent-set"
        }));
    }

    #[test]
    fn calibration_plan_never_combines_conflicting_flats_when_target_optics_are_unknown() {
        let request = CString::new(
            json!({
                "kind":"flat", "minimum":2,
                "reference":{
                    "path":"light.fits", "role":"light",
                    "signature":{"camera":"ASI2600MM","width":100,"height":80,
                                 "channels":1}
                },
                "candidates":[
                    {"path":"ha-flat.fits","role":"flat",
                     "signature":{"camera":"ASI2600MM","width":100,"height":80,
                                  "channels":1,"filter":"Ha"}},
                    {"path":"oiii-flat.fits","role":"flat",
                     "signature":{"camera":"ASI2600MM","width":100,"height":80,
                                  "channels":1,"filter":"OIII"}}
                ]
            })
            .to_string(),
        )
        .unwrap();
        let mut error = ptr::null_mut();
        let result = unsafe { seiza_calibration_plan_json(request.as_ptr(), &mut error) };
        assert!(!result.is_null());
        assert!(error.is_null());
        let plan: Value =
            serde_json::from_str(unsafe { CStr::from_ptr(result) }.to_str().unwrap()).unwrap();
        unsafe { seiza_string_free(result) };
        assert_eq!(plan["ready"], false);
        assert_eq!(plan["selectedPaths"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn master_builder_publishes_atomically_reports_stats_and_can_cancel() {
        let directory = tempfile::tempdir().unwrap();
        let first = directory.path().join("bias-001.fits");
        let second = directory.path().join("bias-002.fits");
        let wrong = directory.path().join("bias-wrong.fits");
        let output = directory.path().join("master-bias.fits");
        let cancelled_output = directory.path().join("cancelled-master.fits");
        let preserved_output = directory.path().join("preserved.fits");
        let first_image = LinearImage::new(8, 8, 1, vec![100.0; 64]).unwrap();
        let second_image = LinearImage::new(8, 8, 1, vec![102.0; 64]).unwrap();
        let wrong_image = LinearImage::new(4, 4, 1, vec![101.0; 16]).unwrap();
        seiza_stacking::write_processed_image_fits_f32(&first, &first_image, &[], &[]).unwrap();
        seiza_stacking::write_processed_image_fits_f32(&second, &second_image, &[], &[]).unwrap();
        seiza_stacking::write_processed_image_fits_f32(&wrong, &wrong_image, &[], &[]).unwrap();
        let request_for = |inputs: &[&Path], output: &Path| {
            CString::new(
                json!({
                    "kind": "bias",
                    "inputs": inputs,
                    "output": output,
                    "rejection": {"lowSigma": 3.0, "highSigma": 3.0}
                })
                .to_string(),
            )
            .unwrap()
        };
        let mut error = ptr::null_mut();
        let request = request_for(&[&first, &second], &output);
        let result = unsafe {
            seiza_calibration_build_master_json(request.as_ptr(), ptr::null(), &mut error)
        };
        assert!(!result.is_null());
        let report: Value =
            serde_json::from_str(unsafe { CStr::from_ptr(result) }.to_str().unwrap()).unwrap();
        unsafe { seiza_string_free(result) };
        assert_eq!(report["schemaVersion"], 2);
        assert_eq!(report["kind"], "bias");
        assert_eq!(report["requestedFrames"], 2);
        assert_eq!(report["inputFrames"], 2);
        assert_eq!(report["inputs"].as_array().unwrap().len(), 2);
        assert_eq!(report["skippedInputs"], json!([]));
        let written = FitsImage::open(&output).unwrap();
        assert_eq!(written.header_str("SEIZAMST"), Some("BIAS"));
        assert_eq!(written.header_f64("NCOMBINE"), Some(2.0));

        let signal = seiza_cancel_signal_create();
        unsafe { seiza_cancel_signal_cancel(signal) };
        let request = request_for(&[&first, &second], &cancelled_output);
        let result =
            unsafe { seiza_calibration_build_master_json(request.as_ptr(), signal, &mut error) };
        assert!(result.is_null());
        assert!(!cancelled_output.exists());
        assert!(
            unsafe { CStr::from_ptr(error) }
                .to_str()
                .unwrap()
                .contains("cancelled")
        );
        unsafe {
            seiza_string_free(error);
            seiza_cancel_signal_free(signal);
        }

        std::fs::write(&preserved_output, b"existing output").unwrap();
        error = ptr::null_mut();
        let request = request_for(&[&first, &wrong], &preserved_output);
        let result = unsafe {
            seiza_calibration_build_master_json(request.as_ptr(), ptr::null(), &mut error)
        };
        assert!(result.is_null());
        assert_eq!(
            std::fs::read(&preserved_output).unwrap(),
            b"existing output"
        );
        unsafe { seiza_string_free(error) };
    }

    #[test]
    fn master_builder_reports_a_skipped_middle_input_without_mislabeling_tallies() {
        let directory = tempfile::tempdir().unwrap();
        let first = directory.path().join("flat-001.fits");
        let stray = directory.path().join("flat-stray.fits");
        let third = directory.path().join("flat-002.fits");
        let output = directory.path().join("master-flat.fits");
        let write_flat = |path: &Path, rotation: f64, value: f32| {
            let image = LinearImage::new(8, 8, 1, vec![value; 64]).unwrap();
            seiza_stacking::write_processed_image_fits_f32(
                path,
                &image,
                &[],
                &[
                    seiza_fits::WriteHeaderCard::new(
                        "IMAGETYP",
                        HeaderValue::String("FLAT".into()),
                    ),
                    seiza_fits::WriteHeaderCard::new("FILTER", HeaderValue::String("R".into())),
                    seiza_fits::WriteHeaderCard::new("ROTATANG", HeaderValue::Float(rotation)),
                ],
            )
            .unwrap();
        };
        write_flat(&first, 10.0, 100.0);
        write_flat(&stray, 90.0, 150.0);
        write_flat(&third, 10.0, 110.0);

        let request = CString::new(
            json!({
                "kind": "flat",
                "inputs": [&first, &stray, &third],
                "output": &output
            })
            .to_string(),
        )
        .unwrap();
        let mut error = ptr::null_mut();
        let result = unsafe {
            seiza_calibration_build_master_json(request.as_ptr(), ptr::null(), &mut error)
        };
        assert!(!result.is_null());
        assert!(error.is_null());
        let report: Value =
            serde_json::from_str(unsafe { CStr::from_ptr(result) }.to_str().unwrap()).unwrap();
        unsafe { seiza_string_free(result) };

        assert_eq!(report["schemaVersion"], 2);
        assert_eq!(report["requestedFrames"], 3);
        assert_eq!(report["inputFrames"], 2);
        assert_eq!(
            report["inputs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|entry| entry["path"].as_str().unwrap())
                .collect::<Vec<_>>(),
            [first.to_str().unwrap(), third.to_str().unwrap()]
        );
        let skipped = report["skippedInputs"].as_array().unwrap();
        assert_eq!(skipped.len(), 1);
        assert_eq!(skipped[0]["path"], stray.to_str().unwrap());
        assert!(skipped[0]["reason"].as_str().unwrap().contains("optical"));
        assert_eq!(
            FitsImage::open(&output).unwrap().header_f64("NCOMBINE"),
            Some(2.0)
        );
    }

    #[test]
    fn master_builder_preserves_output_when_fewer_than_two_inputs_survive() {
        let directory = tempfile::tempdir().unwrap();
        let first = directory.path().join("flat-001.fits");
        let stray = directory.path().join("flat-stray.fits");
        let output = directory.path().join("preserved-master.fits");
        let image = LinearImage::new(8, 8, 1, vec![100.0; 64]).unwrap();
        for (path, rotation) in [(&first, 10.0), (&stray, 90.0)] {
            seiza_stacking::write_processed_image_fits_f32(
                path,
                &image,
                &[],
                &[
                    seiza_fits::WriteHeaderCard::new(
                        "IMAGETYP",
                        HeaderValue::String("FLAT".into()),
                    ),
                    seiza_fits::WriteHeaderCard::new("FILTER", HeaderValue::String("R".into())),
                    seiza_fits::WriteHeaderCard::new("ROTATANG", HeaderValue::Float(rotation)),
                ],
            )
            .unwrap();
        }
        std::fs::write(&output, b"existing output").unwrap();
        let request = CString::new(
            json!({"kind":"flat", "inputs":[&first, &stray], "output":&output}).to_string(),
        )
        .unwrap();
        let mut error = ptr::null_mut();
        let result = unsafe {
            seiza_calibration_build_master_json(request.as_ptr(), ptr::null(), &mut error)
        };
        assert!(result.is_null());
        assert!(!error.is_null());
        let message = unsafe { CStr::from_ptr(error) }.to_str().unwrap();
        assert!(message.contains("only 1 of 2"), "{message}");
        assert_eq!(std::fs::read(&output).unwrap(), b"existing output");
        unsafe { seiza_string_free(error) };
    }

    #[test]
    fn star_detection_over_the_c_surface_measures_and_judges_tilt() {
        // The synthetic star field the stacking tests register against.
        // The measurement pipeline is tuned for real PSFs and under-detects
        // synthetic gaussians (the caveat its original test file carried),
        // so this exercises the C plumbing — options, JSON shape, the tilt
        // fold-in, the error path — not detector quality, which is
        // corpus-validated.
        let (width, height) = (160usize, 128usize);
        let field = stacking_star_field(width, height);
        let data: Vec<u16> = field.iter().map(|value| *value as u16).collect();

        let options = CString::new(
            r#"{"psfType":"gaussian","preset":"standard","triangleAngleDegrees":-30}"#,
        )
        .unwrap();
        let mut error = ptr::null_mut();
        let json = unsafe {
            seiza_stars_detect_luma_u16_json(
                data.as_ptr(),
                data.len(),
                width,
                height,
                options.as_ptr(),
                &mut error,
            )
        };
        assert!(!json.is_null());
        assert!(error.is_null());
        let parsed: Value =
            serde_json::from_str(unsafe { CStr::from_ptr(json) }.to_str().unwrap()).unwrap();
        unsafe { seiza_string_free(json) };
        assert_eq!(parsed["schemaVersion"], 1);
        assert_eq!(parsed["width"], width);
        assert_eq!(parsed["height"], height);
        assert_eq!(parsed["majorAxisOrientationsNormalized"], true);
        assert!(
            !parsed["stars"].as_array().unwrap().is_empty(),
            "no stars found"
        );
        assert!(parsed["averageHfr"].as_f64().unwrap() > 0.0);
        assert_eq!(parsed["cells"].as_array().unwrap().len(), 9);
        assert!(parsed["tilt"].is_object());
        let triangle = &parsed["triangleTilt"];
        assert_eq!(triangle["angleDegrees"], 330.0);
        let expected_inner = 0.25 * (width as f64 / 2.0).hypot(height as f64 / 2.0);
        let expected_outer = 0.5 * width.min(height) as f64;
        assert!((triangle["innerRadiusPixels"].as_f64().unwrap() - expected_inner).abs() < 1e-12);
        assert_eq!(triangle["outerRadiusPixels"], expected_outer);
        assert_eq!(triangle["minimumStarsPerRegion"], 3);
        assert!(triangle["center"].is_object());
        let center_count = triangle["center"]["starCount"].as_u64().unwrap();
        assert_eq!(
            triangle["center"]["medianHfr"].is_number(),
            center_count > 0
        );
        let sectors = triangle["sectors"].as_array().unwrap();
        assert_eq!(sectors.len(), 3);
        for (index, (sector, axis)) in sectors
            .iter()
            .zip([(1, 330.0), (2, 90.0), (3, 210.0)])
            .enumerate()
        {
            assert_eq!(sector["sector"], axis.0, "sector {index}");
            assert_eq!(sector["axisAngleDegrees"], axis.1, "sector {index}");
            let count = sector["starCount"].as_u64().unwrap();
            assert_eq!(sector["medianHfr"].is_number(), count > 0);
        }
        let expected_ready = sectors
            .iter()
            .all(|sector| sector["starCount"].as_u64().unwrap() >= 3);
        assert_eq!(triangle["ready"], expected_ready);
        let annular_count: u64 = sectors
            .iter()
            .map(|sector| sector["starCount"].as_u64().unwrap())
            .sum();
        assert_eq!(triangle["overallMedianHfr"].is_number(), annular_count > 0);
        for field in ["tiltPercent", "bestSector", "worstSector"] {
            assert_eq!(triangle[field].is_number(), expected_ready, "{field}");
        }
        if expected_ready {
            assert!(triangle["tiltPercent"].as_f64().unwrap() >= 0.0);
            for field in ["bestSector", "worstSector"] {
                assert!((1..=3).contains(&triangle[field].as_u64().unwrap()));
            }
        }
        let star = &parsed["stars"][0];
        assert!(star["eccentricity"].is_number(), "PSF was fitted");
        let theta = star["theta"].as_f64().expect("PSF orientation");
        assert!((0.0..std::f64::consts::PI).contains(&theta), "{theta}");
        for cell in parsed["cells"].as_array().unwrap() {
            if let Some(theta) = cell["meanTheta"].as_f64() {
                assert!((0.0..std::f64::consts::PI).contains(&theta), "{theta}");
            }
        }

        // A typo in the options is an error, never silently the defaults.
        let typo = CString::new(r#"{"psftype":"gaussian"}"#).unwrap();
        let mut error = ptr::null_mut();
        let json = unsafe {
            seiza_stars_detect_luma_u16_json(
                data.as_ptr(),
                data.len(),
                width,
                height,
                typo.as_ptr(),
                &mut error,
            )
        };
        assert!(json.is_null());
        assert!(!error.is_null());
        assert!(
            unsafe { CStr::from_ptr(error) }
                .to_str()
                .unwrap()
                .contains("unknown field"),
        );
        unsafe { seiza_string_free(error) };

        let bad_angle = CString::new(r#"{"triangleAngleDegrees":"up"}"#).unwrap();
        let mut error = ptr::null_mut();
        let json = unsafe {
            seiza_stars_detect_luma_u16_json(
                data.as_ptr(),
                data.len(),
                width,
                height,
                bad_angle.as_ptr(),
                &mut error,
            )
        };
        assert!(json.is_null());
        assert!(!error.is_null());
        assert!(
            unsafe { CStr::from_ptr(error) }
                .to_str()
                .unwrap()
                .contains("invalid type"),
        );
        unsafe { seiza_string_free(error) };
    }

    #[test]
    fn star_detection_target_star_count_retries_toward_the_target() {
        // A grid of small faint stars that strict binned-and-blurred
        // detection misses: the adaptive ladder's native-resolution rung
        // must measure at least as many as the strict single pass, and the
        // option must parse over the C surface.
        let (width, height) = (256usize, 256usize);
        let mut data = vec![1000u16; width * height];
        for row in 0..5usize {
            for col in 0..5usize {
                let (cx, cy) = (40.0 + col as f64 * 44.0, 40.0 + row as f64 * 44.0);
                for dy in -7i64..=7 {
                    for dx in -7i64..=7 {
                        let x = (cx as i64 + dx) as usize;
                        let y = (cy as i64 + dy) as usize;
                        let d2 = (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2);
                        let value = 3000.0 * (-d2 / (2.0 * 1.3 * 1.3)).exp();
                        data[y * width + x] =
                            (f64::from(data[y * width + x]) + value).min(65535.0) as u16;
                    }
                }
            }
        }

        let run = |options: &str| -> usize {
            let options = CString::new(options).unwrap();
            let mut error = ptr::null_mut();
            let json = unsafe {
                seiza_stars_detect_luma_u16_json(
                    data.as_ptr(),
                    data.len(),
                    width,
                    height,
                    options.as_ptr(),
                    &mut error,
                )
            };
            assert!(!json.is_null());
            assert!(error.is_null());
            let parsed: Value =
                serde_json::from_str(unsafe { CStr::from_ptr(json) }.to_str().unwrap()).unwrap();
            unsafe { seiza_string_free(json) };
            parsed["stars"].as_array().unwrap().len()
        };

        let strict =
            r#"{"psfType":"none","preset":"standard","detectionBinning":2,"sensitivity":30}"#;
        let adaptive = r#"{"psfType":"none","preset":"standard","detectionBinning":2,"sensitivity":30,"targetStarCount":25}"#;
        let single_pass = run(strict);
        let with_target = run(adaptive);
        assert!(single_pass < 25, "premise: strict pass misses stars");
        assert_eq!(with_target, 25, "the ladder measures the full field");

        // Zero keeps the single-pass behavior bit-for-bit.
        let zero_target = r#"{"psfType":"none","preset":"standard","detectionBinning":2,"sensitivity":30,"targetStarCount":0}"#;
        assert_eq!(run(zero_target), single_pass);
    }

    #[test]
    fn star_detection_rejects_invalid_slice_dimensions_before_borrowing() {
        let mut error = ptr::null_mut();
        let sample = 0u16;
        let json =
            unsafe { seiza_stars_detect_luma_u16_json(&sample, 0, 0, 1, ptr::null(), &mut error) };
        assert!(json.is_null());
        assert!(!error.is_null());
        let message = unsafe { CStr::from_ptr(error) }.to_str().unwrap();
        assert!(message.contains("must be non-zero"), "{message}");
        unsafe { seiza_string_free(error) };

        let oversized = isize::MAX as usize / std::mem::size_of::<u16>() + 1;
        let mut error = ptr::null_mut();
        let json = unsafe {
            seiza_stars_detect_luma_u16_json(
                std::ptr::NonNull::<u16>::dangling().as_ptr(),
                oversized,
                oversized,
                1,
                ptr::null(),
                &mut error,
            )
        };
        assert!(json.is_null());
        assert!(!error.is_null());
        let message = unsafe { CStr::from_ptr(error) }.to_str().unwrap();
        assert!(message.contains("larger than a slice"), "{message}");
        unsafe { seiza_string_free(error) };
    }

    #[test]
    fn fits_path_detection_is_byte_for_byte_the_buffer_contract() {
        let (width, height) = (160usize, 128usize);
        let field = stacking_star_field(width, height);
        let samples: Vec<u16> = field.iter().map(|value| *value as u16).collect();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("measured-field.FiTs");
        std::fs::write(
            &path,
            synthetic_u16_fits(
                width,
                height,
                &samples,
                &[("FOCALLEN", "2000.0"), ("XPIXSZ", "3.76")],
            ),
        )
        .unwrap();
        let options =
            Some(r#"{"psfType":"gaussian","preset":"standard","triangleAngleDegrees":390}"#);

        let direct = call_star_buffer(&samples, width, height, options).unwrap();
        let from_path = call_star_path(&path, options).unwrap();

        assert_eq!(from_path, direct);
        assert_eq!(from_path["width"], width);
        assert_eq!(from_path["height"], height);
        assert_eq!(from_path["majorAxisOrientationsNormalized"], true);
        assert_eq!(from_path["triangleTilt"]["angleDegrees"], 30.0);
    }

    #[test]
    fn xisf_path_detection_uses_the_astronomy_loader_and_refuses_rasters() {
        let (width, height) = (160usize, 128usize);
        let field = stacking_star_field(width, height);
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("measured-field.xisf");
        seiza_xisf::write_f32_image(
            &path,
            width,
            height,
            seiza_fits::F32ImageData::Mono(&field),
            &[
                seiza_fits::WriteHeaderCard::new("FOCALLEN", HeaderValue::Float(2000.0)),
                seiza_fits::WriteHeaderCard::new("XPIXSZ", HeaderValue::Float(3.76)),
            ],
        )
        .unwrap();

        let result = call_star_path(&path, Some(r#"{"psfType":"none"}"#)).unwrap();
        assert_eq!(result["schemaVersion"], 1);
        assert_eq!(result["width"], width);
        assert_eq!(result["height"], height);
        assert_eq!(result["cells"].as_array().unwrap().len(), 9);
        assert!(result.get("triangleTilt").is_none());

        let bad_options = call_star_path(&path, Some(r#"{"psftype":"none"}"#)).unwrap_err();
        assert!(bad_options.contains("unknown field"), "{bad_options}");
        let missing = directory.path().join("missing.fits");
        let bad_options = call_star_path(&missing, Some(r#"{"psftype":"none"}"#)).unwrap_err();
        assert!(bad_options.contains("unknown field"), "{bad_options}");

        let raster = directory.path().join("ordinary.png");
        std::fs::write(&raster, b"not actually a raster").unwrap();
        let message = call_star_path(&raster, None).unwrap_err();
        assert!(message.contains("not a FITS or XISF path"), "{message}");
    }

    #[test]
    fn path_detection_headers_only_fill_options_the_caller_omitted() {
        use seiza_stars::hocus_focus_star_detection::StructureRemovalMethod;

        let image = FitsImage {
            width: 2,
            height: 2,
            planes: 1,
            pixels: seiza_fits::Pixels::U16(vec![1, 2, 3, 4]),
            headers: vec![
                ("FOCALLEN".into(), HeaderValue::Float(-1.0)),
                ("focallength".into(), HeaderValue::Float(2000.0)),
                ("xpixsz".into(), HeaderValue::Float(3.76)),
            ],
        };

        // The usable aliases are case-insensitive, as FITS-derived XISF
        // metadata is not required to preserve FITS keyword casing.
        let from_headers = StarDetectOptions::default()
            .with_frame_headers(&image)
            .into_params()
            .unwrap();
        assert_eq!(from_headers.structure_layers, 5);
        assert_eq!(from_headers.noise_reduction_radius, 0);

        // Caller pixel size wins while the missing focal length still comes
        // from the frame, yielding the wide-field class.
        let caller_pixel = StarDetectOptions {
            pixel_size_um: Some(12.0),
            ..Default::default()
        }
        .with_frame_headers(&image)
        .into_params()
        .unwrap();
        assert_eq!(caller_pixel.structure_layers, 3);

        // Caller focal length also wins independently while pixel size comes
        // from the frame.
        let caller_focal = StarDetectOptions {
            focal_length_mm: Some(500.0),
            ..Default::default()
        }
        .with_frame_headers(&image)
        .into_params()
        .unwrap();
        assert_eq!(caller_focal.structure_layers, 3);

        // An explicit preset is complete and all explicit fine tuning still
        // lands after it.
        let explicit = StarDetectOptions {
            preset: Some("standard".into()),
            focal_length_mm: Some(50.0),
            pixel_size_um: Some(50.0),
            structure_removal: Some("atrous".into()),
            detection_binning: Some(3),
            sensitivity: Some(7.5),
            ..Default::default()
        }
        .with_frame_headers(&image)
        .into_params()
        .unwrap();
        assert_eq!(explicit.structure_layers, 4);
        assert_eq!(explicit.structure_removal, StructureRemovalMethod::Atrous);
        assert_eq!(explicit.detection_binning, 3);
        assert_eq!(explicit.sensitivity, 7.5);

        let luma = astronomy_luma_u16(&image);
        assert!(matches!(luma, Cow::Borrowed(_)), "mono u16 should not copy");
        assert_eq!(luma.as_ref(), [1, 2, 3, 4]);

        let planar_rgb = FitsImage {
            width: 2,
            height: 1,
            planes: 3,
            pixels: seiza_fits::Pixels::U16(vec![10, 40, 20, 50, 30, 60]),
            headers: vec![],
        };
        assert_eq!(astronomy_luma_u16(&planar_rgb).as_ref(), [20, 50]);

        let bayer = FitsImage {
            width: 2,
            height: 2,
            planes: 1,
            pixels: seiza_fits::Pixels::U16(vec![10_000, 20_000, 30_000, 40_000]),
            headers: vec![("BAYERPAT".into(), HeaderValue::String("RGGB".into()))],
        };
        let expected = bayer.debayer().unwrap().to_luma_u16();
        assert_eq!(astronomy_luma_u16(&bayer).as_ref(), expected);
    }

    #[test]
    fn a_refusal_over_the_c_surface_names_the_field_and_both_readings() {
        // The zeroed struct means "nothing recorded"; set only what the case
        // needs, plus the known bits that make those readings count.
        let mut light: SeizaFrameSignature = unsafe { std::mem::zeroed() };
        light.rotation_deg = 101.93;
        light.known = SEIZA_FRAME_HAS_ROTATION;
        let mut flat: SeizaFrameSignature = unsafe { std::mem::zeroed() };
        flat.rotation_deg = 104.24;
        flat.known = SEIZA_FRAME_HAS_ROTATION;

        let mut error = ptr::null_mut();
        let text = unsafe {
            seiza_calibration_describe_optics_mismatch(&light, &flat, ptr::null(), &mut error)
        };
        assert!(!text.is_null());
        assert!(error.is_null());
        let reason = unsafe { CStr::from_ptr(text) }.to_str().unwrap().to_owned();
        unsafe { seiza_string_free(text) };
        assert!(reason.contains("101.93"), "{reason}");
        assert!(reason.contains("104.24"), "{reason}");
        assert!(reason.contains("deg apart"), "{reason}");
    }

    #[test]
    fn a_stacker_answers_what_a_prospective_light_could_accept() {
        // A stack with no masters loaded keeps both lists empty: nothing to
        // accept, nothing refused. The shape is what a C host parses, so the
        // schema is asserted, not just the emptiness.
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
        assert!(!stacker.is_null());
        let mut light: SeizaFrameSignature = unsafe { std::mem::zeroed() };
        light.rotation_deg = 101.93;
        light.known = SEIZA_FRAME_HAS_ROTATION;
        let json = unsafe {
            seiza_live_stacker_compatible_calibration_json(stacker, &light, ptr::null(), &mut error)
        };
        assert!(!json.is_null());
        assert!(error.is_null());
        let parsed: Value =
            serde_json::from_str(unsafe { CStr::from_ptr(json) }.to_str().unwrap()).unwrap();
        unsafe { seiza_string_free(json) };
        assert_eq!(parsed["schemaVersion"], 1);
        assert_eq!(parsed["kept"].as_array().unwrap().len(), 0);
        assert_eq!(parsed["dropped"].as_array().unwrap().len(), 0);
        unsafe { seiza_live_stacker_free(stacker) };
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
        let image8 = render_astronomy_with_pipeline(
            fits.clone(),
            AstronomyImageFormat::Fits,
            &stack,
            RenderPipelineOptions {
                background: None,
                deconvolution: None,
                sample_domain: &SampleDomain::UnitLinear,
                max_dimension: 0,
                interactive_preview: false,
            },
        )
        .unwrap();
        let image16 = render_astronomy_with_pipeline16(
            fits,
            AstronomyImageFormat::Fits,
            &stack,
            RenderPipelineOptions {
                background: None,
                deconvolution: None,
                sample_domain: &SampleDomain::UnitLinear,
                max_dimension: 0,
                interactive_preview: false,
            },
        )
        .unwrap();

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
    fn render_config_accepts_a_presentation_only_physical_sample_domain() {
        let request: ImageRenderConfigRequest = serde_json::from_value(json!({
            "stretch": [{
                "model": { "type": "identity" },
                "color_strategy": "linked",
                "max_analysis_samples": 4096
            }],
            "sample_domain": {
                "type": "physical-linear",
                "normalization": {
                    "type": "robust-percentile",
                    "black_percentile": 0.001,
                    "white_percentile": 0.999,
                    "max_analysis_samples": 200000
                }
            }
        }))
        .unwrap();

        let (stack, background, deconvolution, interactive_preview, sample_domain) =
            request.into_parts();
        assert_eq!(stack.len(), 1);
        assert!(background.is_none());
        assert!(deconvolution.is_none());
        assert!(!interactive_preview);
        assert_eq!(
            sample_domain,
            Some(SampleDomain::PhysicalLinear {
                normalization: seiza_stretch::SampleNormalization::RobustPercentile {
                    black_percentile: 0.001,
                    white_percentile: 0.999,
                    max_analysis_samples: 200_000,
                },
            })
        );
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
                    "sample_radius": 2,
                    "protected_regions": [{
                        "kind": "polygon",
                        "points": [[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]]
                    }]
                }
            }
        }))
        .unwrap();
        let (stack, background, deconvolution, interactive_preview, sample_domain) =
            request.into_parts();
        assert_eq!(stack.len(), 1);
        assert!(!interactive_preview);
        assert!(deconvolution.is_none());
        assert!(sample_domain.is_none());
        let background = background.unwrap();
        assert_eq!(background.mode, CorrectionMode::Subtract);
        assert_eq!(background.strength, 1.0);
        assert_eq!(background.config.sample_radius, Some(2));
        assert_eq!(background.config.protected_regions.len(), 1);
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
            strength: 0.5,
            config: serde_json::from_value(json!({
                "model": { "kind": "polynomial", "degree": 1, "ridge": 0.0 },
                "sample_radius": 2
            }))
            .unwrap(),
        };

        let uncorrected = render_fits_with_stack(fits.clone(), &stack, 0).unwrap();
        let corrected = render_astronomy_with_pipeline(
            fits,
            AstronomyImageFormat::Fits,
            &stack,
            RenderPipelineOptions {
                background: Some(&background),
                deconvolution: None,
                sample_domain: &SampleDomain::UnitLinear,
                max_dimension: 0,
                interactive_preview: false,
            },
        )
        .unwrap();
        assert_ne!(corrected.rgba, uncorrected.rgba);

        let metadata: Value =
            serde_json::from_str(corrected.metadata_json.to_str().unwrap()).unwrap();
        assert_eq!(metadata["backgroundProcessing"]["mode"], "subtract");
        assert_eq!(metadata["backgroundProcessing"]["strength"], 0.5);
        assert_eq!(metadata["backgroundProcessing"]["model"], "polynomial");
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
        for sample in pixels.iter_mut().take(size) {
            *sample = f32::NAN;
        }
        for row in pixels.chunks_exact_mut(size) {
            row[0] = f32::NAN;
        }
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
        let restored = render_astronomy_with_pipeline(
            fits,
            AstronomyImageFormat::Fits,
            &stack,
            RenderPipelineOptions {
                background: None,
                deconvolution: Some(&deconvolution),
                sample_domain: &SampleDomain::UnitLinear,
                max_dimension: 0,
                interactive_preview: false,
            },
        )
        .unwrap();
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
            metadata["deconvolutionProcessing"]["algorithmVersion"],
            u64::from(seiza_deconvolution::ALGORITHM_VERSION)
        );
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
        let (stack, background, deconvolution, interactive_preview, sample_domain) =
            request.into_parts();
        assert!(background.is_none());
        assert!(sample_domain.is_none());
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
        let preview = render_astronomy_with_pipeline(
            fits,
            AstronomyImageFormat::Fits,
            &stack,
            RenderPipelineOptions {
                background: None,
                deconvolution: Some(&deconvolution),
                sample_domain: &SampleDomain::UnitLinear,
                max_dimension: 100,
                interactive_preview,
            },
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
        let automatic_domain = SampleDomain::PhysicalLinear {
            normalization: Default::default(),
        };
        let first =
            render_cached_interactive_preview(&path, &stack, None, None, &automatic_domain, 100)
                .unwrap();
        let second = render_cached_interactive_preview(
            &path,
            &stack,
            None,
            Some(&deconvolution),
            &automatic_domain,
            100,
        )
        .unwrap();
        let explicit_domain = SampleDomain::PhysicalLinear {
            normalization: seiza_stretch::SampleNormalization::ExplicitRange {
                black: 0.0,
                white: 0.5,
            },
        };
        let remapped =
            render_cached_interactive_preview(&path, &stack, None, None, &explicit_domain, 100)
                .unwrap();
        let first_metadata: Value =
            serde_json::from_str(first.metadata_json.to_str().unwrap()).unwrap();
        let second_metadata: Value =
            serde_json::from_str(second.metadata_json.to_str().unwrap()).unwrap();
        let remapped_metadata: Value =
            serde_json::from_str(remapped.metadata_json.to_str().unwrap()).unwrap();
        assert_eq!(first_metadata["interactivePreviewCacheHit"], false);
        assert_eq!(second_metadata["interactivePreviewCacheHit"], true);
        assert_eq!(remapped_metadata["interactivePreviewCacheHit"], true);
        assert!(second_metadata["deconvolutionProcessing"].is_object());
        assert_eq!(remapped_metadata["sampleDomain"]["resolved"]["white"], 0.5);
        assert_ne!(first.rgba, remapped.rgba);

        let background = BackgroundRenderRequest::default();
        assert_ne!(
            interactive_preview_cache_key(&path, None, &automatic_domain, 100).unwrap(),
            interactive_preview_cache_key(&path, Some(&background), &automatic_domain, 100)
                .unwrap()
        );
        assert_ne!(
            interactive_preview_cache_key(&path, None, &SampleDomain::UnitLinear, 100).unwrap(),
            interactive_preview_cache_key(&path, None, &automatic_domain, 100).unwrap()
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

        let image = render_astronomy_image(
            FitsImage::open(&path).unwrap(),
            AstronomyImageFormat::Fits,
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
    fn astronomy_render_metadata_identifies_xisf_sources_at_both_depths() {
        let source = FitsImage {
            width: 2,
            height: 2,
            planes: 1,
            pixels: seiza_fits::Pixels::F32(vec![0.0, 0.25, 0.5, 1.0]),
            headers: Vec::new(),
        };
        let image8 = render_astronomy_image(
            source.clone(),
            AstronomyImageFormat::Xisf,
            &StretchParams::default(),
            0,
            RgbStretchMode::Auto,
        )
        .unwrap();
        let image16 = render_astronomy_image16(
            source,
            AstronomyImageFormat::Xisf,
            &StretchParams::default(),
            0,
            RgbStretchMode::Auto,
        )
        .unwrap();
        for metadata_json in [&image8.metadata_json, &image16.metadata_json] {
            let metadata: Value = serde_json::from_str(metadata_json.to_str().unwrap()).unwrap();
            assert_eq!(metadata["format"], "XISF");
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
        let image = render_astronomy_with_pipeline16(
            fits,
            AstronomyImageFormat::Fits,
            &StretchStack::single(config),
            RenderPipelineOptions {
                background: None,
                deconvolution: None,
                sample_domain: &SampleDomain::UnitLinear,
                max_dimension: 0,
                interactive_preview: false,
            },
        )
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
    fn normalized_statistics_agree_across_render_depths() {
        let luma_u8: Vec<u8> = vec![0, 64, 128, 255];
        let luma_u16: Vec<u16> = luma_u8
            .iter()
            .map(|&value| u16::from(value) * 257)
            .collect();
        let raster = raster_statistics_json(&luma_u8);
        let fits = statistics_json(&seiza_fits::statistics_u16(&luma_u16));
        assert_eq!(raster["scale"], 255);
        assert_eq!(fits["scale"], 65_535);
        for field in ["minimum", "maximum", "median", "mean"] {
            let raster = raster["normalized"][field].as_f64().unwrap();
            let fits = fits["normalized"][field].as_f64().unwrap();
            assert!(
                (raster - fits).abs() < 1.0e-9,
                "{field}: {raster} vs {fits}"
            );
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

// ---------------------------------------------------------------------------
// File metadata and calibration-master orchestration
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
/// Read only a FITS/XISF header and return normalized frame-role, calibration
/// state, and matching-signature JSON. No pixel payload is decoded. The owned
/// string must be released with [`seiza_string_free`].
///
/// # Safety
/// `path` must be a valid NUL-terminated string. When non-null, `error_out`
/// must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_probe_frame_json(
    path: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let path = required_path(path, "frame path")?;
        owned_json(&probe_frame_header(&path)?)
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Select a proximity-ordered, internally coherent set of calibration-frame
/// probe records without duplicating Seiza's matching policy in a host.
///
/// `request_json` contains `kind` (`bias`, `dark`, `dark-flat`, or `flat`), a
/// `reference` record, `candidates`, optional `minimum` (default 2), and
/// optional camelCase tolerances. Dark flats use dark exposure/temperature
/// matching and should reference the selected flat they will calibrate. Each
/// record has `path`, `role`, and the `signature` object returned by
/// [`seiza_probe_frame_json`]; extra probe fields are ignored.
/// The owned response reports `matchedPaths`, coherent `selectedPaths`,
/// `ready`, and one stable exclusion reason per omitted candidate. Free it
/// with [`seiza_string_free`].
///
/// # Safety
/// `request_json` must be a valid NUL-terminated string. When non-null,
/// `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_calibration_plan_json(
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let request_json = required_str(request_json, "calibration plan JSON")?;
        let request: CalibrationPlanRequest = serde_json::from_str(&request_json)
            .map_err(|error| format!("invalid calibration plan JSON: {error}"))?;
        owned_json(&build_calibration_plan(request)?)
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Build and atomically publish one calibration master from raw FITS/XISF
/// paths, returning an owned JSON integration report.
///
/// The camelCase request has `kind`, at least two `inputs`, and `output`;
/// optional fields are `bias`, `dark`, `darkExposureSeconds`,
/// `exposureSeconds`, `rejection` (`lowSigma`, `highSigma`), and
/// `defectSuppression` (`lowSigma`, `highSigma`). `dark` is a dark-flat when
/// building a flat. Defect suppression is accepted only for flats, because a
/// dark master must retain the hot pixels it is meant to subtract.
///
/// Construction is a two-pass leave-one-out sigma-clipped mean and may take
/// minutes. Run it off the UI thread. `cancel` may be null; otherwise another
/// thread may call [`seiza_cancel_signal_cancel`]. Cancellation is checked
/// between input frames, returns failure through `error_out`, and publishes no
/// output. FITS/XISF writers publish through a same-directory temporary file,
/// so every successful output is atomic. Free the response with
/// [`seiza_string_free`].
///
/// Response schema 2 reports `requestedFrames`, accepted `inputFrames` and
/// accepted-only per-frame `inputs`, plus `skippedInputs` entries with `path`
/// and `reason` for metadata disagreements the integrator set aside. The
/// accepted and skipped paths are disjoint and account for every requested
/// input in request order.
///
/// # Safety
/// `request_json` must be a valid NUL-terminated string. `cancel` must be null
/// or a live [`SeizaCancelSignal`] retained until this call returns. When
/// non-null, `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_calibration_build_master_json(
    request_json: *const c_char,
    cancel: *const SeizaCancelSignal,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let request_json = required_str(request_json, "master build JSON")?;
        let request: MasterBuildRequest = serde_json::from_str(&request_json)
            .map_err(|error| format!("invalid master build JSON: {error}"))?;
        let cancellation = unsafe { cancel.as_ref() }
            .map(|signal| CancelSignal::from(Arc::clone(&signal.cancelled)));
        owned_json(&build_master_request(request, cancellation)?)
    })
    .unwrap_or(ptr::null_mut())
}

// ---------------------------------------------------------------------------
// Reading how deep a stack is
// ---------------------------------------------------------------------------

/// The most channels a measurement reports separately: mono or debayered RGB.
pub const SEIZA_SNR_MAX_CHANNELS: usize = 3;

/// One reading of an accumulator, in the stack's own units. Only ratios
/// between readings of the same stack mean anything.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SeizaSnrSample {
    /// Frames the accumulator had taken when this was measured.
    pub frames: u32,
    /// Pixel-scale noise of the integration, averaged across channels.
    pub noise: f64,
    /// Median sample, averaged across channels.
    pub background: f64,
    /// How far the brightest one percent sits above the background.
    pub signal: f64,
    /// `signal / noise` for this one reading. To compare depths, divide one
    /// signal — normally the deepest measured — by each depth's noise instead:
    /// the brightest-percent statistic is itself lifted by noise where a stack
    /// is shallow, so reading each depth against its own signal flatters the
    /// early frames.
    pub snr: f64,
    /// How many entries of `channel_noise` are meaningful.
    pub channel_count: usize,
    /// Per-channel noise.
    pub channel_noise: [f64; SEIZA_SNR_MAX_CHANNELS],
}

/// Read how deep a live stack is, without copying the accumulator.
///
/// Returns 1 and fills `sample` when the stack could be measured, 0 when it
/// could not — too small a frame, or too little of it covered, which is an
/// ordinary answer early in a build rather than an error — and -1 on failure,
/// with `error_out` set.
///
/// **Test the return against 1, not for truth.** -1 is non-zero, so
/// `if (seiza_live_stacker_measure_depth(...))` is true on failure. `sample`
/// is cleared to zero on entry and only carries a reading when the return is
/// exactly 1.
///
/// # Safety
///
/// `stacker` must be a live `SeizaLiveStacker` pointer and `sample` must point
/// at writable storage for one `SeizaSnrSample`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_live_stacker_measure_depth(
    stacker: *const SeizaLiveStacker,
    sample: *mut SeizaSnrSample,
    error_out: *mut *mut c_char,
) -> i32 {
    clear_error(error_out);
    // Cleared on entry like the error slot, so a caller that tests the return
    // for truth rather than against 1 reads zeros rather than whatever was on
    // the stack.
    if !sample.is_null() {
        unsafe { *sample = SeizaSnrSample::default() };
    }
    ffi_result(error_out, || {
        let stacker = unsafe { required_live_stacker(stacker)? };
        if sample.is_null() {
            return Err("sample is required".into());
        }
        let Some(measured) = rust_measure_depth(stacker.stacker.view()) else {
            return Ok(0);
        };
        let mut channel_noise = [0.0f64; SEIZA_SNR_MAX_CHANNELS];
        let channel_count = measured.channel_noise.len().min(SEIZA_SNR_MAX_CHANNELS);
        channel_noise[..channel_count].copy_from_slice(&measured.channel_noise[..channel_count]);
        unsafe {
            *sample = SeizaSnrSample {
                frames: measured.frames,
                noise: measured.noise,
                background: measured.background,
                signal: measured.signal,
                snr: measured.snr(),
                channel_count,
                channel_noise,
            };
        }
        Ok(1)
    })
    .unwrap_or(-1)
}

/// The depths a build of `total` frames should measure at: the doubling
/// ladder, and the full set.
///
/// Writes up to `out_len` depths into `out` and returns how many the ladder
/// has in total, which may exceed `out_len`. Pass a null `out` with a zero
/// `out_len` to ask for the count alone.
///
/// # Safety
///
/// `out` must point at writable storage for `out_len` values, or be null when
/// `out_len` is zero.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_checkpoint_depths(
    total: usize,
    out: *mut usize,
    out_len: usize,
) -> usize {
    let depths = rust_checkpoint_depths(total);
    if !out.is_null() && out_len > 0 {
        let copied = depths.len().min(out_len);
        unsafe { ptr::copy_nonoverlapping(depths.as_ptr(), out, copied) };
    }
    depths.len()
}

// ---------------------------------------------------------------------------
// Deciding which calibration frames belong together
// ---------------------------------------------------------------------------

/// Which numeric fields of a [`SeizaFrameSignature`] were actually recorded.
///
/// A cleared bit means "not recorded", which is not the same as zero: a camera
/// really can be set to gain 0, and a frame that does not say what gain it used
/// is a different thing from one that says zero.
///
/// The flags exist so that a zeroed struct — `= {0}`, `calloc`, `memset`, all
/// of them ordinary C — means "nothing recorded" rather than "every setting is
/// zero". Getting that backwards silently inverts every comparison, so the
/// safe reading is the one you get for free.
pub const SEIZA_FRAME_HAS_WIDTH: u32 = 1 << 0;
pub const SEIZA_FRAME_HAS_HEIGHT: u32 = 1 << 1;
pub const SEIZA_FRAME_HAS_CHANNELS: u32 = 1 << 2;
pub const SEIZA_FRAME_HAS_BINNING_X: u32 = 1 << 3;
pub const SEIZA_FRAME_HAS_BINNING_Y: u32 = 1 << 4;
pub const SEIZA_FRAME_HAS_GAIN: u32 = 1 << 5;
pub const SEIZA_FRAME_HAS_OFFSET: u32 = 1 << 6;
pub const SEIZA_FRAME_HAS_READOUT_MODE: u32 = 1 << 7;
pub const SEIZA_FRAME_HAS_FOCAL_LENGTH: u32 = 1 << 8;
pub const SEIZA_FRAME_HAS_ROTATION: u32 = 1 << 9;
pub const SEIZA_FRAME_HAS_EXPOSURE: u32 = 1 << 10;
pub const SEIZA_FRAME_HAS_CAMERA_TEMP: u32 = 1 << 11;
pub const SEIZA_FRAME_HAS_CAPTURED_AT: u32 = 1 << 12;

/// What a frame was shot with, as far as matching cares.
///
/// Text fields are null when unknown. Numeric fields are read only when their
/// `SEIZA_FRAME_HAS_*` bit is set in `known`, so a zeroed struct means nothing
/// was recorded — the safe reading, and the one plain `= {0}` gives you. A
/// non-finite value also reads as unknown, whatever its bit says.
///
/// Integer-valued settings are carried as doubles so the struct has one shape;
/// every value a camera reports is exact in a double.
///
/// To fill one: zero it, assign the fields the frame recorded, and set each
/// one's flag — `light.gain = 100;` then
/// `light.known |= SEIZA_FRAME_HAS_GAIN;`. Text fields need no flag.
///
/// C++ callers should write `SeizaFrameSignature light{};` rather than
/// `= {0}`; the C spelling is correct but warns under
/// `-Wmissing-field-initializers`. Both zero the struct, which is what
/// "nothing recorded" is.
///
/// A missing value on the *candidate* side disqualifies it and a missing value
/// on the *reference* side accepts what it is offered: a light that does not
/// record its gain cannot rule anything out, while a calibration frame that
/// does not record its gain cannot prove it belongs. Rotation is the exception
/// — unknown on either side matches.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SeizaFrameSignature {
    /// Bitwise OR of the `SEIZA_FRAME_HAS_*` flags for the numeric fields
    /// this frame actually recorded.
    pub known: u32,
    pub camera: *const c_char,
    pub telescope: *const c_char,
    pub bayer_pattern: *const c_char,
    pub filter: *const c_char,
    pub width: f64,
    pub height: f64,
    pub channels: f64,
    pub binning_x: f64,
    pub binning_y: f64,
    pub gain: f64,
    pub offset: f64,
    pub readout_mode: f64,
    pub focal_length_mm: f64,
    pub rotation_deg: f64,
    pub exposure_seconds: f64,
    pub camera_temp_c: f64,
    pub captured_at_unix: f64,
}

impl Default for SeizaFrameSignature {
    fn default() -> Self {
        // All zero, which is exactly what a C caller gets from `= {0}`: no
        // flags set, so nothing recorded.
        Self {
            known: 0,
            camera: ptr::null(),
            telescope: ptr::null(),
            bayer_pattern: ptr::null(),
            filter: ptr::null(),
            width: 0.0,
            height: 0.0,
            channels: 0.0,
            binning_x: 0.0,
            binning_y: 0.0,
            gain: 0.0,
            offset: 0.0,
            readout_mode: 0.0,
            focal_length_mm: 0.0,
            rotation_deg: 0.0,
            exposure_seconds: 0.0,
            camera_temp_c: 0.0,
            captured_at_unix: 0.0,
        }
    }
}

/// Fill a signature with "nothing recorded". Equivalent to zeroing it, and
/// offered so callers in languages without designated initializers have a
/// spelling.
///
/// # Safety
///
/// `signature` must point at writable storage for one `SeizaFrameSignature`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_frame_signature_init(signature: *mut SeizaFrameSignature) {
    if !signature.is_null() {
        unsafe { *signature = SeizaFrameSignature::default() };
    }
}

/// A numeric field is a reading only when its bit is set and the value is
/// finite.
fn optional_number(known: u32, flag: u32, value: f64) -> Option<f64> {
    (known & flag != 0 && value.is_finite()).then_some(value)
}

fn optional_integer(known: u32, flag: u32, value: f64) -> Option<i64> {
    optional_number(known, flag, value).map(|value| value as i64)
}

unsafe fn optional_text(value: *const c_char) -> Result<Option<String>, String> {
    if value.is_null() {
        return Ok(None);
    }
    unsafe { CStr::from_ptr(value) }
        .to_str()
        .map(|text| Some(text.to_owned()))
        .map_err(|error| format!("signature text is not valid UTF-8: {error}"))
}

unsafe fn frame_signature(
    signature: *const SeizaFrameSignature,
) -> Result<seiza_calibration::FrameSignature, String> {
    let signature = unsafe { signature.as_ref() }.ok_or("frame signature is required")?;
    // Built field by field from the default rather than as a literal: the Rust
    // struct is `#[non_exhaustive]`, so it can gain fields without breaking
    // this, and anything it gains starts out as "unknown" here.
    let mut converted = seiza_calibration::FrameSignature::default();
    converted.camera = unsafe { optional_text(signature.camera) }?;
    converted.telescope = unsafe { optional_text(signature.telescope) }?;
    converted.bayer_pattern = unsafe { optional_text(signature.bayer_pattern) }?;
    converted.filter = unsafe { optional_text(signature.filter) }?;
    let known = signature.known;
    converted.width = optional_integer(known, SEIZA_FRAME_HAS_WIDTH, signature.width);
    converted.height = optional_integer(known, SEIZA_FRAME_HAS_HEIGHT, signature.height);
    converted.channels = optional_integer(known, SEIZA_FRAME_HAS_CHANNELS, signature.channels);
    converted.binning_x = optional_integer(known, SEIZA_FRAME_HAS_BINNING_X, signature.binning_x);
    converted.binning_y = optional_integer(known, SEIZA_FRAME_HAS_BINNING_Y, signature.binning_y);
    converted.gain = optional_integer(known, SEIZA_FRAME_HAS_GAIN, signature.gain);
    converted.offset = optional_integer(known, SEIZA_FRAME_HAS_OFFSET, signature.offset);
    converted.readout_mode =
        optional_integer(known, SEIZA_FRAME_HAS_READOUT_MODE, signature.readout_mode);
    converted.focal_length_mm = optional_number(
        known,
        SEIZA_FRAME_HAS_FOCAL_LENGTH,
        signature.focal_length_mm,
    );
    converted.rotation_deg =
        optional_number(known, SEIZA_FRAME_HAS_ROTATION, signature.rotation_deg);
    converted.exposure_seconds =
        optional_number(known, SEIZA_FRAME_HAS_EXPOSURE, signature.exposure_seconds);
    converted.camera_temp_c =
        optional_number(known, SEIZA_FRAME_HAS_CAMERA_TEMP, signature.camera_temp_c);
    converted.captured_at_unix = optional_integer(
        known,
        SEIZA_FRAME_HAS_CAPTURED_AT,
        signature.captured_at_unix,
    );
    Ok(converted)
}

/// Whether two frames came off the same sensor in the same mode. Returns 1 for
/// a match, 0 for a mismatch, -1 on failure with `error_out` set.
///
/// # Safety
///
/// Both pointers must reference initialized `SeizaFrameSignature` values, with
/// any text fields either null or valid UTF-8 C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_calibration_sensor_matches(
    reference: *const SeizaFrameSignature,
    candidate: *const SeizaFrameSignature,
    error_out: *mut *mut c_char,
) -> i32 {
    clear_error(error_out);
    ffi_result(error_out, || {
        let reference = unsafe { frame_signature(reference) }?;
        let candidate = unsafe { frame_signature(candidate) }?;
        Ok(i32::from(seiza_calibration::sensor_matches(
            &reference, &candidate,
        )))
    })
    .unwrap_or(-1)
}

/// Which fields of a [`SeizaMatchTolerances`] the caller set. A cleared bit
/// takes the built-in default for that tolerance.
///
/// A zero tolerance is a legitimate ask — "these must be exactly equal" — so
/// it cannot double as "unset". The flags keep both expressible, and make a
/// zeroed struct mean "every default", which is what a caller who zeroes one
/// almost certainly wants.
pub const SEIZA_TOLERANCE_HAS_EXPOSURE: u32 = 1 << 0;
pub const SEIZA_TOLERANCE_HAS_DARK_TEMPERATURE: u32 = 1 << 1;
pub const SEIZA_TOLERANCE_HAS_MASTER_TEMPERATURE: u32 = 1 << 2;
pub const SEIZA_TOLERANCE_HAS_ROTATION: u32 = 1 << 3;
pub const SEIZA_TOLERANCE_HAS_FOCAL_LENGTH: u32 = 1 << 4;
pub const SEIZA_TOLERANCE_HAS_FLAT_SESSION: u32 = 1 << 5;
pub const SEIZA_TOLERANCE_HAS_EXPOSURE_FRACTION: u32 = 1 << 6;

/// How close two readings have to be to count as the same.
///
/// Every field is optional: set a `SEIZA_TOLERANCE_HAS_*` bit to override that
/// tolerance, leave it clear to take the default. A zeroed struct therefore
/// means "all defaults", and passing a null pointer anywhere one of these is
/// accepted means the same. An override that is negative or not a number is
/// ignored in favour of the default.
///
/// The defaults are what a rig's own scatter needs rather than what a
/// specification promises. [`seiza_match_tolerances_default`] fills one in if
/// you want to read or adjust them.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SeizaMatchTolerances {
    /// Bitwise OR of the `SEIZA_TOLERANCE_HAS_*` flags this struct overrides.
    pub known: u32,
    /// Dark exposure against light exposure: the floor, in seconds. The
    /// comparison takes whichever of this and `exposure_fraction` is larger.
    pub exposure_seconds: f64,
    /// Dark exposure against light exposure: the proportional part, as a
    /// fraction of the longer of the two. Past about a minute this is what
    /// decides.
    pub exposure_fraction: f64,
    /// Dark sensor temperature against light sensor temperature, in Celsius.
    pub dark_temperature_c: f64,
    /// Sensor temperature within one master's input set, in Celsius.
    pub master_temperature_c: f64,
    /// Rotator angle between a flat and what it corrects, in degrees.
    pub rotation_deg: f64,
    /// Focal length between a flat and what it corrects, in millimetres.
    pub focal_length_mm: f64,
    /// How far apart flats in one master may have been shot, in seconds.
    pub flat_session_seconds: u64,
}

/// Fill `tolerances` with the built-in defaults and every flag set, so a
/// caller can read them or adjust one and pass the rest through unchanged.
///
/// # Safety
///
/// `tolerances` must point at writable storage for one
/// `SeizaMatchTolerances`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_match_tolerances_default(tolerances: *mut SeizaMatchTolerances) {
    if tolerances.is_null() {
        return;
    }
    let defaults = seiza_calibration::MatchTolerances::default();
    unsafe {
        *tolerances = SeizaMatchTolerances {
            known: SEIZA_TOLERANCE_HAS_EXPOSURE
                | SEIZA_TOLERANCE_HAS_EXPOSURE_FRACTION
                | SEIZA_TOLERANCE_HAS_DARK_TEMPERATURE
                | SEIZA_TOLERANCE_HAS_MASTER_TEMPERATURE
                | SEIZA_TOLERANCE_HAS_ROTATION
                | SEIZA_TOLERANCE_HAS_FOCAL_LENGTH
                | SEIZA_TOLERANCE_HAS_FLAT_SESSION,
            exposure_seconds: defaults.exposure_seconds,
            exposure_fraction: defaults.exposure_fraction,
            dark_temperature_c: defaults.dark_temperature_c,
            master_temperature_c: defaults.master_temperature_c,
            rotation_deg: defaults.rotation_deg,
            focal_length_mm: defaults.focal_length_mm,
            flat_session_seconds: defaults.flat_session_seconds,
        };
    }
}

/// A null pointer, a zeroed struct, or any cleared flag all fall back to the
/// built-in default for that tolerance. So does an override that is not a
/// number, or is negative: neither can decide anything, and taking them
/// literally would quietly match nothing.
unsafe fn match_tolerances(
    tolerances: *const SeizaMatchTolerances,
) -> seiza_calibration::MatchTolerances {
    let mut resolved = seiza_calibration::MatchTolerances::default();
    let Some(overrides) = (unsafe { tolerances.as_ref() }) else {
        return resolved;
    };
    // Finite *and* not negative. `(a - b).abs() <= -1.0` is false for every
    // pair, so a negative tolerance would match nothing and say nothing about
    // why — the same silent failure a zeroed signature used to cause, reached
    // by a different mistake.
    let set =
        |flag: u32, value: f64| overrides.known & flag != 0 && value.is_finite() && value >= 0.0;
    if set(SEIZA_TOLERANCE_HAS_EXPOSURE, overrides.exposure_seconds) {
        resolved.exposure_seconds = overrides.exposure_seconds;
    }
    if set(
        SEIZA_TOLERANCE_HAS_EXPOSURE_FRACTION,
        overrides.exposure_fraction,
    ) {
        resolved.exposure_fraction = overrides.exposure_fraction;
    }
    if set(
        SEIZA_TOLERANCE_HAS_DARK_TEMPERATURE,
        overrides.dark_temperature_c,
    ) {
        resolved.dark_temperature_c = overrides.dark_temperature_c;
    }
    if set(
        SEIZA_TOLERANCE_HAS_MASTER_TEMPERATURE,
        overrides.master_temperature_c,
    ) {
        resolved.master_temperature_c = overrides.master_temperature_c;
    }
    if set(SEIZA_TOLERANCE_HAS_ROTATION, overrides.rotation_deg) {
        resolved.rotation_deg = overrides.rotation_deg;
    }
    if set(SEIZA_TOLERANCE_HAS_FOCAL_LENGTH, overrides.focal_length_mm) {
        resolved.focal_length_mm = overrides.focal_length_mm;
    }
    if overrides.known & SEIZA_TOLERANCE_HAS_FLAT_SESSION != 0 {
        resolved.flat_session_seconds = overrides.flat_session_seconds;
    }
    resolved
}

/// Whether a flat describes the same optical path as what it would correct —
/// filter, telescope, focal length and rotator angle. Returns 1, 0, or -1 as
/// [`seiza_calibration_sensor_matches`] does.
///
/// `tolerances` may be null for the defaults; see [`SeizaMatchTolerances`].
///
/// # Safety
///
/// As [`seiza_calibration_sensor_matches`]. `tolerances` must be null or
/// reference an initialized `SeizaMatchTolerances`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_calibration_optics_match(
    reference: *const SeizaFrameSignature,
    candidate: *const SeizaFrameSignature,
    tolerances: *const SeizaMatchTolerances,
    error_out: *mut *mut c_char,
) -> i32 {
    clear_error(error_out);
    ffi_result(error_out, || {
        let reference = unsafe { frame_signature(reference) }?;
        let candidate = unsafe { frame_signature(candidate) }?;
        let tolerances = unsafe { match_tolerances(tolerances) };
        Ok(i32::from(seiza_calibration::optics_match(
            &reference,
            &candidate,
            &tolerances,
        )))
    })
    .unwrap_or(-1)
}

/// Whether a dark's exposure and sensor temperature suit the frame it would be
/// subtracted from. Reads `exposure_seconds` and `camera_temp_c` from both
/// signatures. Returns 1, 0, or -1 as
/// [`seiza_calibration_sensor_matches`] does.
///
/// `tolerances` may be null for the defaults; see [`SeizaMatchTolerances`].
///
/// # Safety
///
/// As [`seiza_calibration_sensor_matches`]. `tolerances` must be null or
/// reference an initialized `SeizaMatchTolerances`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_calibration_dark_matches(
    reference: *const SeizaFrameSignature,
    candidate: *const SeizaFrameSignature,
    tolerances: *const SeizaMatchTolerances,
    error_out: *mut *mut c_char,
) -> i32 {
    clear_error(error_out);
    ffi_result(error_out, || {
        let reference = unsafe { frame_signature(reference) }?;
        let candidate = unsafe { frame_signature(candidate) }?;
        let tolerances = unsafe { match_tolerances(tolerances) };
        let matched = seiza_calibration::exposure_matches(&reference, &candidate, &tolerances)
            && seiza_calibration::temperature_matches(&reference, &candidate, &tolerances);
        Ok(i32::from(matched))
    })
    .unwrap_or(-1)
}

/// Detect and measure stars in a mono 16-bit frame with the HocusFocus
/// detector (seiza-stars): wavelet structure removal, kappa-sigma thresholds,
/// hot-pixel filtering, multi-criteria validation, and optional PSF fitting —
/// the measurement detector, distinct from the fast alignment detector the
/// solver uses. `data` holds `width * height` samples, row-major.
///
/// `options_json` may be null for the defaults, or an object with any of:
/// `preset` ("widefield" | "standard" | "longfocal"), `focalLengthMm` +
/// `pixelSizeUm` (classify the pixel scale when no preset is given),
/// `psfType` ("none" | "gaussian" | "moffat4"; default "moffat4"),
/// `structureRemoval` ("filtered" | "atrous"), `detectionBinning`,
/// `keepSaturated`, `noiseReductionRadius`, `sensitivity`, optional
/// `triangleAngleDegrees`, and optional `targetStarCount`. Unknown fields
/// are an error, so a typo cannot silently run the defaults. A triangle
/// angle may be any finite degree value; the response normalizes it over
/// `[0, 360)`, with zero pointing to the top of the image and positive
/// angles turning clockwise. A positive `targetStarCount` retries detection
/// with progressively more permissive settings (a relaxed SNR gate, then
/// native-resolution unblurred detection) until at least that many stars
/// are measured, returning the best pass otherwise.
///
/// Returns owned JSON, released with [`seiza_string_free`]:
/// `{"schemaVersion":1,"width":..,"height":..,
/// "majorAxisOrientationsNormalized":true,"averageHfr":..,
/// "averageFwhm":..,"noiseSigma":..,"backgroundMean":..,
/// "stars":[{"x":..,"y":..,"hfr":..,"fwhm":..,"brightness":..,
/// "background":..,"snr":..,"flux":..,"pixelCount":..,"saturated":..,
/// "eccentricity":?,"theta":?,"rSquared":?}],"cells":[..3x3 region
/// statistics..],"tilt":{..parallelogram corner verdict..},
/// "triangleTilt":?}`. Fitted `theta` and cell
/// `meanTheta` are normalized ellipse major-axis orientations in radians over
/// `[0, π)`. The tilt analysis is folded in because it is derived and cheap,
/// so every consumer reads the same verdict from one call.
///
/// `triangleTilt` is omitted unless `triangleAngleDegrees` was supplied. It
/// reports the normalized angle, center/annulus radii in pixels, the native
/// minimum-star confidence policy and readiness, center statistics, and three
/// ordered sectors with 1-based sector ID, axis angle, star count, and median
/// HFR. The center is `radius < 0.25 * hypot(width/2, height/2)`. The annulus
/// is `innerRadiusPixels <= radius <= 0.5 * min(width, height)` and is split
/// into three complete, half-open 120-degree sectors. `overallMedianHfr` is
/// the median of every usable annular star. `tiltPercent`, `bestSector`, and
/// `worstSector` are null until every sector has at least
/// `minimumStarsPerRegion` measurements; otherwise tilt is
/// `100 * (worst median - best median) / overallMedianHfr`.
///
/// # Safety
///
/// `data` must reference `width * height` readable `uint16_t` samples; both
/// dimensions must be non-zero.
/// `options_json` must be null or a NUL-terminated UTF-8 string. `error_out`
/// must be null or point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_stars_detect_luma_u16_json(
    data: *const u16,
    len: usize,
    width: usize,
    height: usize,
    options_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        if data.is_null() {
            return Err("data is required".into());
        }
        let expected = width
            .checked_mul(height)
            .ok_or_else(|| format!("image dimensions {width}x{height} overflow"))?;
        if expected == 0 {
            return Err("image dimensions must be non-zero".into());
        }
        if len != expected {
            return Err(format!("data length {len} does not match {width}x{height}"));
        }
        // `slice::from_raw_parts` limits the byte span, not the element
        // count, to `isize::MAX`. A real frame cannot approach this, but an
        // FFI caller controls all three dimensions, so reject it before the
        // unsafe slice construction instead of relying on an unstated Rust
        // precondition.
        if len
            .checked_mul(std::mem::size_of::<u16>())
            .is_none_or(|bytes| bytes > isize::MAX as usize)
        {
            return Err("image is larger than a slice can describe".into());
        }
        if !(data as usize).is_multiple_of(std::mem::align_of::<u16>()) {
            return Err("data is not aligned for uint16_t samples".into());
        }
        let samples = unsafe { std::slice::from_raw_parts(data, len) };
        let options = unsafe { parse_star_detect_options(options_json) }?;
        owned_json(&detect_stars_response(samples, width, height, options)?)
    })
    .unwrap_or(ptr::null_mut())
}

/// Open a FITS or XISF image and run the same measurement detector and tilt
/// analysis as [`seiza_stars_detect_luma_u16_json`] on its linear 16-bit
/// luminance. Mono u16 is measured without a display-render copy; planar RGB
/// is collapsed to luminance, raw Bayer data is debayered to luminance, and
/// other numeric sample types use the astronomy loader's linear u16 mapping.
/// Raster image formats are deliberately refused.
///
/// `options_json` has the same shape as the buffer entry point. When no
/// `preset` is given, an omitted `focalLengthMm` is read from `FOCALLEN`,
/// `FOCALLENGTH`, or `FOCAL`, and an omitted `pixelSizeUm` is read from
/// `XPIXSZ`; caller-provided values win independently. An explicit `preset`
/// is the complete classification choice and does not consult those headers.
/// The returned schema-1 JSON, including `width` and `height`, has the same
/// shape as the buffer entry point and is byte-for-byte identical for the
/// same samples and resolved options.
///
/// # Safety
///
/// `path` must reference a NUL-terminated UTF-8 FITS or XISF path.
/// `options_json` must be null or a NUL-terminated UTF-8 string. `error_out`
/// must be null or point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_stars_detect_path_json(
    path: *const c_char,
    options_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let path = required_path(path, "image path")?;
        // Refuse caller mistakes before opening what may be a multi-hundred
        // megabyte astronomy frame.
        let options = unsafe { parse_star_detect_options(options_json) }?;
        let (image, _) = open_astronomy_image(&path)?;
        let options = options.with_frame_headers(&image);
        let samples = astronomy_luma_u16(&image);
        owned_json(&detect_stars_response(
            &samples,
            image.width,
            image.height,
            options,
        )?)
    })
    .unwrap_or(ptr::null_mut())
}

/// Why two frames' sensor readings refuse to match, as a human-readable
/// string naming every differing field and both readings — for example
/// `gain light=100 master=200`. Never empty on a mismatch; on frames that
/// match it still reports (the caller decides when to ask). Returns a string
/// the caller owns and must release with [`seiza_string_free`], or null with
/// `error_out` set.
///
/// # Safety
///
/// As [`seiza_calibration_sensor_matches`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_calibration_describe_sensor_mismatch(
    reference: *const SeizaFrameSignature,
    candidate: *const SeizaFrameSignature,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let reference = unsafe { frame_signature(reference) }?;
        let candidate = unsafe { frame_signature(candidate) }?;
        let text = seiza_calibration::describe_sensor_mismatch(&reference, &candidate);
        CString::new(text)
            .map(CString::into_raw)
            .map_err(|_| "mismatch description contains a NUL byte".to_string())
    })
    .unwrap_or(ptr::null_mut())
}

/// Why a flat's optical path refuses the frame it would correct: every
/// differing field with both readings, and for rotation the gap against the
/// tolerance — for example `rotation light=101.93deg master=104.24deg (2.31
/// deg apart, tolerance 2.00)`. `tolerances` may be null for the defaults.
/// Returns a string released with [`seiza_string_free`], or null with
/// `error_out` set.
///
/// # Safety
///
/// As [`seiza_calibration_optics_match`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_calibration_describe_optics_mismatch(
    reference: *const SeizaFrameSignature,
    candidate: *const SeizaFrameSignature,
    tolerances: *const SeizaMatchTolerances,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let reference = unsafe { frame_signature(reference) }?;
        let candidate = unsafe { frame_signature(candidate) }?;
        let tolerances = unsafe { match_tolerances(tolerances) };
        let text = seiza_calibration::describe_optics_mismatch(&reference, &candidate, &tolerances);
        CString::new(text)
            .map(CString::into_raw)
            .map_err(|_| "mismatch description contains a NUL byte".to_string())
    })
    .unwrap_or(ptr::null_mut())
}

/// Whether two rotator angles are close enough to share a flat. Wraps at 360,
/// and a non-finite angle on either side matches — a missing angle means the
/// rig had no rotator, or the record predates keeping one.
///
/// Takes its tolerance directly rather than a [`SeizaMatchTolerances`], being
/// a single comparison with a single tolerance.
/// [`seiza_match_tolerances_default`] gives the value the other entry points
/// use.
#[unsafe(no_mangle)]
pub extern "C" fn seiza_calibration_rotation_matches(
    reference_deg: f64,
    candidate_deg: f64,
    tolerance_deg: f64,
) -> i32 {
    i32::from(seiza_calibration::rotation_matches(
        Some(reference_deg),
        Some(candidate_deg),
        tolerance_deg,
    ))
}

/// Fit the camera pedestal in `light`, in the light's own units.
///
/// Dividing by a flat only works on a signal that starts at zero; without a
/// bias or dark master the offset is still there. Sky background varies with
/// the flat's own response, so the intercept of that line is the part that
/// does not.
///
/// Returns 1 and writes `pedestal` when the fit succeeded, 0 when the frame
/// cannot support one — too few usable tiles, a flat too uniform to give the
/// line a lever, or a slope saying the model does not describe this field —
/// and -1 when the caller passed something unusable, with `error_out` set.
///
/// **Test the return against 1, not for truth.** -1 is non-zero. `pedestal` is
/// set to NaN on entry and only carries a fit when the return is exactly 1 —
/// zero would be ambiguous, since a camera with no offset fits exactly that.
///
/// The fit reads low by roughly 0.8 times the frame's noise, by construction.
/// That is the safe direction, and it cancels when comparing two frames.
///
/// # Safety
///
/// `light` and `flat` must each point at `width * height` readable floats.
/// `pedestal` must point at writable storage for one float.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_calibration_fit_flat_pedestal(
    light: *const f32,
    flat: *const f32,
    width: usize,
    height: usize,
    pedestal: *mut f32,
    error_out: *mut *mut c_char,
) -> i32 {
    clear_error(error_out);
    if !pedestal.is_null() {
        // NaN, not zero: a camera with no offset fits a pedestal of exactly
        // zero, so clearing to zero would leave "declined" and "fitted zero"
        // indistinguishable for a caller who ignores the return code.
        unsafe { *pedestal = f32::NAN };
    }
    ffi_result(error_out, || {
        if light.is_null() || flat.is_null() || pedestal.is_null() {
            return Err("light, flat and pedestal are required".into());
        }
        let samples = width
            .checked_mul(height)
            .ok_or("image dimensions overflow")?;
        if samples == 0 {
            return Err("image dimensions must be non-zero".into());
        }
        // `slice::from_raw_parts` requires the slice to be at most isize::MAX
        // *bytes*, not elements. Unreachable with real frames, but this is a
        // stated contract on an unsafe function, so it is checked rather than
        // assumed.
        if samples
            .checked_mul(std::mem::size_of::<f32>())
            .is_none_or(|bytes| bytes > isize::MAX as usize)
        {
            return Err("image is larger than a slice can describe".into());
        }
        let light = unsafe { std::slice::from_raw_parts(light, samples) };
        let flat = unsafe { std::slice::from_raw_parts(flat, samples) };
        let light = seiza_calibration::LinearImageRef::new(light, width, height, 1)
            .map_err(|error| error.to_string())?;
        let flat = seiza_calibration::LinearImageRef::new(flat, width, height, 1)
            .map_err(|error| error.to_string())?;
        // An error here is a caller mistake — mismatched sizes, a colour frame
        // — and is reported as one. A frame this simply cannot fit is 0.
        let Some(fitted) =
            seiza_calibration::fit_flat_pedestal(light, flat).map_err(|error| error.to_string())?
        else {
            return Ok(0);
        };
        unsafe { *pedestal = fitted };
        Ok(1)
    })
    .unwrap_or(-1)
}

// ---------------------------------------------------------------------------
// RC-Astro external tools (BlurXTerminator, StarXTerminator, NoiseXTerminator)
// ---------------------------------------------------------------------------

fn rc_astro_cli(executable: Option<PathBuf>, host: Option<String>) -> Result<RcAstroCli, String> {
    let cli = match executable {
        Some(path) => RcAstroCli::with_executable(path),
        None => RcAstroCli::locate().ok_or_else(|| "rc-astro was not found on PATH".to_string())?,
    };
    // RC-Astro's --host names the integrator to their support; a cabi
    // consumer that says nothing is a cabi-based application.
    Ok(cli.with_host(host.unwrap_or_else(|| format!("seiza-cabi-{}", env!("CARGO_PKG_VERSION")))))
}

/// The path of the `rc-astro` executable on `PATH`, as a string released
/// with [`seiza_string_free`]. Returns null with `error_out` untouched when
/// the CLI is simply not installed — absence is a state, not an error.
///
/// # Safety
///
/// `error_out` must be null or point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_rc_astro_locate(error_out: *mut *mut c_char) -> *mut c_char {
    clear_error(error_out);
    let Some(cli) = RcAstroCli::locate() else {
        return ptr::null_mut();
    };
    ffi_result(error_out, || {
        CString::new(cli.executable().display().to_string())
            .map(CString::into_raw)
            .map_err(|_| "rc-astro path contains a NUL byte".to_string())
    })
    .unwrap_or(ptr::null_mut())
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RcAstroParameterResponse {
    name: String,
    /// Absent for a GUI-only parameter that cannot be set through the CLI.
    #[serde(skip_serializing_if = "Option::is_none")]
    flag: Option<String>,
    label: String,
    description: String,
    /// "float", "bool", or "int".
    r#type: &'static str,
    default: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max: Option<f64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RcAstroSchemaResponse {
    schema_version: u32,
    /// The tool's own --json contract version (v3 through v6 are known).
    contract_version: u32,
    cli_version: String,
    key: String,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    ml_version: Option<i64>,
    licensed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    license_message: Option<String>,
    parameters: Vec<RcAstroParameterResponse>,
}

fn rc_astro_schema_response(schema: &ExternalToolSchema) -> RcAstroSchemaResponse {
    RcAstroSchemaResponse {
        schema_version: 1,
        contract_version: schema.schema_version,
        cli_version: schema.cli_version.clone(),
        key: schema.key.clone(),
        name: schema.name.clone(),
        ml_version: schema.ml_version,
        licensed: schema.licensed,
        license_message: schema.license_message.clone(),
        parameters: schema
            .parameters
            .iter()
            .map(|parameter| {
                let (kind, default, min, max) = match &parameter.kind {
                    ExternalParameterKind::Float { default, min, max } => (
                        "float",
                        serde_json::json!(default),
                        Some(*min).filter(|value| value.is_finite()),
                        Some(*max).filter(|value| value.is_finite()),
                    ),
                    ExternalParameterKind::Bool { default } => {
                        ("bool", serde_json::json!(default), None, None)
                    }
                    ExternalParameterKind::Int { default, min, max } => (
                        "int",
                        serde_json::json!(default),
                        (*min != i64::MIN).then_some(*min as f64),
                        (*max != i64::MAX).then_some(*max as f64),
                    ),
                };
                RcAstroParameterResponse {
                    name: parameter.name.clone(),
                    flag: parameter.flag.clone(),
                    label: parameter.label.clone(),
                    description: parameter.description.clone(),
                    r#type: kind,
                    default,
                    min,
                    max,
                }
            })
            .collect(),
    }
}

/// One RC-Astro file-processing request for
/// [`seiza_rc_astro_process_file_json`]. Unknown fields are an error so a
/// typo cannot silently run the defaults.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RcAstroProcessRequest {
    /// "bxt", "sxt", or "nxt".
    tool: String,
    input: PathBuf,
    output: PathBuf,
    /// Parameter values keyed by schema name; anything absent keeps the
    /// tool's default. Booleans for switches, numbers for the rest.
    #[serde(default)]
    parameters: serde_json::Map<String, serde_json::Value>,
    /// "auto", "cpu", "gpu", or "gpuN"; absent uses the tool's saved default.
    #[serde(default)]
    device: Option<String>,
    /// Executable path; absent searches PATH.
    #[serde(default)]
    executable: Option<PathBuf>,
    /// Integrator identification for RC-Astro support, e.g. "MyApp-1.2".
    #[serde(default)]
    host: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RcAstroProcessResponse {
    schema_version: u32,
    primary: PathBuf,
    /// Extra files written beside the output — StarXTerminator's stars
    /// image when the "stars" parameter was on.
    sidecars: Vec<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    device: Option<String>,
    warnings: Vec<String>,
    cli_version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    ml_version: Option<i64>,
}

fn rc_astro_parameter_values(
    parameters: &serde_json::Map<String, serde_json::Value>,
) -> Result<Vec<(String, ExternalParameterValue)>, String> {
    parameters
        .iter()
        .map(|(name, value)| {
            let value = match value {
                serde_json::Value::Bool(flag) => ExternalParameterValue::Bool(*flag),
                serde_json::Value::Number(number) => match number.as_i64() {
                    Some(whole) => ExternalParameterValue::Int(whole),
                    None => ExternalParameterValue::Float(
                        number
                            .as_f64()
                            .ok_or_else(|| format!("parameter {name:?} is not a finite number"))?,
                    ),
                },
                _ => return Err(format!("parameter {name:?} must be a boolean or a number")),
            };
            Ok((name.clone(), value))
        })
        .collect()
}

/// One RC-Astro tool's live contract as schema-1 JSON: its parameters with
/// flags, types, ranges, and defaults, plus CLI/model versions and license
/// state, read from `rc-astro <tool> --json`. Flags change between CLI
/// builds, so build requests from this document rather than hard-coding
/// them. `executable` may be null to search PATH. Returns a string released
/// with [`seiza_string_free`], or null with `error_out` set.
///
/// # Safety
///
/// `tool` must be a NUL-terminated UTF-8 string. `executable` must be null
/// or a NUL-terminated UTF-8 path. When non-null, `error_out` must point to
/// writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_rc_astro_tool_schema_json(
    executable: *const c_char,
    tool: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let tool = required_str(tool, "rc-astro tool")?;
        let executable = optional_path(executable)?;
        let cli = rc_astro_cli(executable, None)?;
        let schema = cli.tool_schema(&tool).map_err(|error| error.to_string())?;
        owned_json(&rc_astro_schema_response(&schema))
    })
    .unwrap_or(ptr::null_mut())
}

/// Run one RC-Astro tool on an image file. The request names the tool, the
/// input and output paths, and parameter values keyed by schema name (see
/// [`seiza_rc_astro_tool_schema_json`]); whole-number floats are accepted
/// for float parameters. The run streams the tool's progress internally,
/// kills a child silent for ten minutes, and honors `cancel` within half a
/// second. The response lists the written files — StarXTerminator's stars
/// sidecar included — and the device the tool reported using. Returns a
/// string released with [`seiza_string_free`], or null with `error_out`
/// set.
///
/// # Safety
///
/// `request_json` must be a NUL-terminated UTF-8 string. `cancel` must be
/// null or a live [`SeizaCancelSignal`] retained until this call returns.
/// When non-null, `error_out` must point to writable storage for one
/// pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_rc_astro_process_file_json(
    request_json: *const c_char,
    cancel: *const SeizaCancelSignal,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let request_json = required_str(request_json, "rc-astro request JSON")?;
        let request: RcAstroProcessRequest = serde_json::from_str(&request_json)
            .map_err(|error| format!("invalid rc-astro request JSON: {error}"))?;
        let cli = rc_astro_cli(request.executable.clone(), request.host.clone())?;
        let schema = cli
            .tool_schema(&request.tool)
            .map_err(|error| error.to_string())?;
        let tool_request = ExternalToolRequest {
            tool: request.tool.clone(),
            parameters: rc_astro_parameter_values(&request.parameters)?,
            device: request.device.clone(),
        };
        let cancellation = unsafe { cancel.as_ref() }
            .map(|signal| CancelSignal::from(Arc::clone(&signal.cancelled)));
        let run = cli
            .run_on_file(
                &schema,
                &tool_request,
                &request.input,
                &request.output,
                cancellation.as_ref(),
                &mut |_| {},
            )
            .map_err(|error| error.to_string())?;
        owned_json(&RcAstroProcessResponse {
            schema_version: 1,
            primary: run.primary,
            sidecars: run.sidecars,
            device: run.device,
            warnings: run.warnings,
            cli_version: schema.cli_version.clone(),
            ml_version: schema.ml_version,
        })
    })
    .unwrap_or(ptr::null_mut())
}

#[cfg(test)]
mod rc_astro_tests {
    use super::*;

    /// A stand-in rc-astro: answers `<tool> --json` with a schema document
    /// and otherwise copies the input to the output (with a stars sidecar
    /// when asked), emitting the real event-stream shape.
    #[cfg(unix)]
    fn fake_rc_astro(directory: &std::path::Path) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = directory.join("rc-astro");
        let script = r#"#!/bin/sh
if [ "$2" = "--json" ] && [ $# -eq 2 ]; then
  cat <<'SCHEMA'
{"schemaVersion": 6, "cliVersion": "2.6.6", "key": "sxt",
 "name": "RC-Astro StarXTerminator", "mlVersion": 11,
 "license": {"status": "permanent", "valid": true, "message": "Permanently licensed"},
 "parameters": [
   {"label": "Tile Overlap", "name": "overlap", "flag": "--overlap",
    "description": "Fractional overlap", "type": "float",
    "default": 0.2, "min": 0.0, "max": 0.5},
   {"label": "Generate Star Image", "name": "stars", "flag": "--stars",
    "description": "Also write a stars-only image", "type": "bool", "default": false},
   {"label": "Iterations", "name": "it", "flag": "--it",
    "description": "Passes", "type": "int", "default": 2, "min": 1, "max": 5},
   {"label": "Color Separation", "name": "csep",
    "description": "GUI only", "type": "bool", "default": false}
 ]}
SCHEMA
  exit 0
fi
out=""
input=""
stars=0
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    --stars) stars=1; shift ;;
    --host|--depth|--device) shift 2 ;;
    --*|sxt|bxt|nxt) shift ;;
    *) input="$1"; shift ;;
  esac
done
cp "$input" "$out"
echo '{"event":"info","topic":"device","device":"cpu"}'
echo "{\"event\":\"status\",\"phase\":\"complete\",\"output\":\"$out\"}"
if [ "$stars" = 1 ]; then
  sidecar="${out%.fits}-stars.fits"
  cp "$input" "$sidecar"
  echo "{\"event\":\"status\",\"phase\":\"complete\",\"output\":\"$sidecar\"}"
fi
"#;
        std::fs::write(&path, script).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    fn call_json(json: *mut c_char, error: *mut c_char) -> Result<serde_json::Value, String> {
        if json.is_null() {
            assert!(!error.is_null());
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { seiza_string_free(error) };
            return Err(message);
        }
        let value = unsafe { CStr::from_ptr(json) }
            .to_string_lossy()
            .into_owned();
        unsafe { seiza_string_free(json) };
        Ok(serde_json::from_str(&value).unwrap())
    }

    #[cfg(unix)]
    #[test]
    fn the_schema_and_a_run_round_trip_through_the_abi() {
        let directory = tempfile::tempdir().unwrap();
        let executable = fake_rc_astro(directory.path());
        let executable_c = CString::new(executable.to_str().unwrap()).unwrap();
        let tool = CString::new("sxt").unwrap();

        let mut error: *mut c_char = ptr::null_mut();
        let schema = unsafe {
            seiza_rc_astro_tool_schema_json(executable_c.as_ptr(), tool.as_ptr(), &mut error)
        };
        let schema = call_json(schema, error).unwrap();
        assert_eq!(schema["contractVersion"], 6);
        assert_eq!(schema["licensed"], true);
        assert_eq!(schema["parameters"][1]["name"], "stars");
        assert_eq!(schema["parameters"][1]["type"], "bool");
        assert_eq!(schema["parameters"][2]["type"], "int");
        assert_eq!(schema["parameters"][2]["min"], 1.0);
        assert_eq!(schema["parameters"][2]["max"], 5.0);
        // A GUI-only parameter carries no flag key at all.
        assert!(schema["parameters"][3].get("flag").is_none());

        let input = directory.path().join("in.fits");
        std::fs::write(&input, b"fake fits").unwrap();
        let output = directory.path().join("out.fits");
        // "it": 3.0 exercises the float-for-int coercion a consumer hits
        // when it clamps to the schema's numeric bounds and sends the
        // result back.
        let request = serde_json::json!({
            "tool": "sxt",
            "input": input,
            "output": output,
            "parameters": {"stars": true, "overlap": 0.3, "it": 3.0},
            "executable": executable,
        });
        let request_c = CString::new(request.to_string()).unwrap();
        let mut error: *mut c_char = ptr::null_mut();
        let response = unsafe {
            seiza_rc_astro_process_file_json(request_c.as_ptr(), ptr::null(), &mut error)
        };
        let response = call_json(response, error).unwrap();
        assert_eq!(response["primary"], output.to_str().unwrap());
        assert_eq!(response["sidecars"].as_array().unwrap().len(), 1);
        assert_eq!(response["device"], "cpu");
        assert_eq!(response["cliVersion"], "2.6.6");
        assert!(output.is_file());
    }

    #[test]
    fn a_malformed_request_reports_a_clear_error() {
        let request_c = CString::new(r#"{"tool": "sxt"}"#).unwrap();
        let mut error: *mut c_char = ptr::null_mut();
        let response = unsafe {
            seiza_rc_astro_process_file_json(request_c.as_ptr(), ptr::null(), &mut error)
        };
        let message = call_json(response, error).unwrap_err();
        assert!(
            message.contains("invalid rc-astro request JSON"),
            "{message}"
        );
    }
}
