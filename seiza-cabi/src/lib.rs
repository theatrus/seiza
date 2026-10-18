use image::DynamicImage;
mod constellations;
mod flat_response;
mod parallax;
mod resample;
mod satellites;
use resample::{downsample_interleaved_f32, downsample_rgba};
use seiza::blind::{BlindIndex, BlindParams};
use seiza::catalog::{StarCatalog, tiles::TileCatalog};
use seiza::downloads::{CachePolicy, CatalogManager, CatalogSet, Dataset, DownloadEvent};
use seiza::field_objects::{FieldObject, FieldObjects, FieldOutline};
use seiza::prior::SolveSource;
use seiza::{
    DetectBackend, DetectConfig, Error as SeizaError, SolveLimits, detect_stars,
    detect_stars_luma_f32,
};
use seiza_background::{BackgroundConfig, BackgroundFit, CorrectionMode, fit_background_masked};
use seiza_deconvolution::{DeconvolutionConfig, deconvolve, deconvolve_masked};
use seiza_fits::{FitsImage, HeaderValue, RgbImage16, Statistics, StretchParams};
use seiza_stacking::{
    CancelSignal, ChannelCoverage, ChannelSamples, ColorComposition, ColorCrop, ColorNormalization,
    ColorOptions, CropReport, DarkLevelScreening, ExternalParameterKind, ExternalParameterValue,
    ExternalToolRequest, ExternalToolSchema, FitsFrame, FlatStarMaskingOptions,
    FlatStarMaskingStatistics, ForaxxOptions, FrameDiagnostics, FrameDisposition, FrameInputMode,
    FrameSourceRole, ImpulseFilterOptions, LinearImage, LiveStacker, LrgbLuminance,
    MasterBuildOptions, MasterDark, MasterFrame, MasterFrameKind, MasterRejectionOptions,
    NarrowbandMatrix, NarrowbandMix, NarrowbandPalette, RcAstroCli, ReferenceRegion,
    RegisteredFrameMapping, RgbLuminance, StackExportSnapshot as RustStackExportSnapshot,
    StackOptions, StackSnapshot as RustStackSnapshot, build_master_from_fits,
    build_master_from_fits_with_scratch, checkpoint_depths as rust_checkpoint_depths, combine_lrgb,
    combine_narrowband, combine_narrowband_matrix, combine_rgb, combine_super_lrgb,
    combine_super_rgb, crop_report, measure_depth as rust_measure_depth, path_identity,
    paths_refer_to_same_file, write_color_fits_f32, write_fits_f32, write_master_fits_f32,
    write_stack_export_fits_f32,
};
use seiza_stretch::{
    ImageShape, ResolvedSampleDomain, SampleDomain, StretchAnalysis, StretchConfig, StretchStack,
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
    GaiaPhotometry = 3,
}

impl CatalogSetupPreset {
    fn from_raw(value: u32) -> Result<Self, String> {
        match value {
            0 => Ok(Self::StandardBlind),
            1 => Ok(Self::DeepestBlind),
            2 => Ok(Self::All),
            3 => Ok(Self::GaiaPhotometry),
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
                Dataset::GaiaPhotometry,
                Dataset::ObjectDistances,
                Dataset::StarDistances,
            ],
            Self::GaiaPhotometry => &[Dataset::GaiaPhotometry],
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

/// A drizzled stack from [`seiza_live_stacker_reintegrate_drizzled`]. Its
/// image and weight pointers are borrowed until
/// [`seiza_drizzle_result_free`] is called.
pub struct SeizaDrizzleResult {
    result: seiza_stacking::DrizzleResult,
    accepted_frames: u32,
    rejected_frames: u32,
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
struct StackReferenceResponse {
    schema_version: u32,
    reference_index: usize,
    reference_path: String,
    ranking: Vec<usize>,
    scores: Vec<Option<StackReferenceScoreResponse>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ChannelReferenceSelectionResponse {
    schema_version: u32,
    channel_reference_index: usize,
    channel_reference_path: String,
    skipped: Vec<SkippedChannelReferenceResponse>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SkippedChannelReferenceResponse {
    index: usize,
    path: String,
    reason: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StackReferenceScoreResponse {
    stars: usize,
    median_star_area: f32,
    background: f32,
    background_variation: f32,
    score: f32,
}

impl From<seiza_stacking::ReferenceScore> for StackReferenceScoreResponse {
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
    /// The frame's registration and normalization, in the core's own
    /// versioned JSON, for a host to keep beside the frame.
    mapping: Box<RegisteredFrameMapping>,
    /// Each channel's measured noise; empty under equal weighting.
    noise: Vec<f32>,
    /// Each channel's weight in the stack mean; empty under equal weighting.
    weight: Vec<f32>,
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
    /// False for a stack aligned to a separate reference image, which starts
    /// with no accepted frames; see
    /// [`seiza_live_stacker_open_fits_with_reference_image`].
    reference_integrated: bool,
    /// That reference image, or null.
    reference_image_path: Option<String>,
    /// Why [`seiza_live_stacker_reintegrate`] cannot replay this stack, or
    /// null when it can.
    reintegration_unavailable: Option<String>,
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

impl From<CalibrationFrameRole> for FrameSourceRole {
    fn from(value: CalibrationFrameRole) -> Self {
        match value {
            CalibrationFrameRole::Bias => Self::Bias,
            CalibrationFrameRole::Dark => Self::Dark,
            CalibrationFrameRole::DarkFlat => Self::DarkFlat,
            CalibrationFrameRole::Flat => Self::Flat,
            CalibrationFrameRole::Light => Self::Light,
            CalibrationFrameRole::Unknown => Self::Unknown,
        }
    }
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

impl From<seiza_stacking::FrameCalibrationState> for FrameCalibrationStateResponse {
    fn from(state: seiza_stacking::FrameCalibrationState) -> Self {
        Self {
            bias_subtracted: state.bias_subtracted,
            dark_subtracted: state.dark_subtracted,
            flat_normalized: state.flat_normalized,
        }
    }
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
    fn into_core(self) -> seiza_stacking::CalibrationPlanKind {
        match self {
            Self::Bias => seiza_stacking::CalibrationPlanKind::Bias,
            Self::Dark => seiza_stacking::CalibrationPlanKind::Dark,
            Self::DarkFlat => seiza_stacking::CalibrationPlanKind::DarkFlat,
            Self::Flat => seiza_stacking::CalibrationPlanKind::Flat,
        }
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
    /// How a light's own rotator angle is read when flats are matched to
    /// it: `{"basis": "recorded"}` (the default), `{"basis": "measured",
    /// "degrees": 82.2}` for an angle the host measured, or `{"basis":
    /// "unknown"}` when the angle does not decide the light's flats.
    light_rotation: Option<seiza_calibration::LightRotation>,
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

impl CalibrationPlanRecord {
    fn into_core(self) -> seiza_stacking::CalibrationRecord {
        seiza_stacking::CalibrationRecord {
            path: PathBuf::from(self.path),
            role: self.role.into(),
            signature: seiza_calibration::FrameSignature::from(&self.signature),
        }
    }
}

impl CalibrationPlanTolerancesRequest {
    /// The defaults, with each given tolerance in place of its default. The
    /// planner checks that every one is finite and not negative.
    fn into_core(self) -> seiza_calibration::MatchTolerances {
        let mut tolerances = seiza_calibration::MatchTolerances::default();
        let fields = [
            (self.exposure_seconds, &mut tolerances.exposure_seconds),
            (self.exposure_fraction, &mut tolerances.exposure_fraction),
            (self.dark_temperature_c, &mut tolerances.dark_temperature_c),
            (
                self.master_temperature_c,
                &mut tolerances.master_temperature_c,
            ),
            (self.rotation_deg, &mut tolerances.rotation_deg),
            (self.focal_length_mm, &mut tolerances.focal_length_mm),
        ];
        for (given, target) in fields {
            if let Some(value) = given {
                *target = value;
            }
        }
        if let Some(seconds) = self.flat_session_seconds {
            tolerances.flat_session_seconds = seconds;
        }
        if let Some(light_rotation) = self.light_rotation {
            tolerances.light_rotation = light_rotation;
        }
        tolerances
    }
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

/// Star masking for sky flats; each field left out keeps the core's default.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
struct FlatStarMaskingRequest {
    detection_sigma: f32,
    halo_radius_factor: f32,
    minimum_radius_pixels: f32,
    saturation_level: Option<f32>,
    minimum_clean_samples: usize,
}

impl Default for FlatStarMaskingRequest {
    fn default() -> Self {
        Self::from(&FlatStarMaskingOptions::default())
    }
}

impl From<&FlatStarMaskingOptions> for FlatStarMaskingRequest {
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

impl From<FlatStarMaskingRequest> for FlatStarMaskingOptions {
    fn from(request: FlatStarMaskingRequest) -> Self {
        Self {
            detection_sigma: request.detection_sigma,
            halo_radius_factor: request.halo_radius_factor,
            minimum_radius_pixels: request.minimum_radius_pixels,
            saturation_level: request.saturation_level,
            minimum_clean_samples: request.minimum_clean_samples,
        }
    }
}

/// Dark level screening; a field left out keeps the core's default.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
struct DarkLevelScreeningRequest {
    max_excess_sigma: f32,
}

impl Default for DarkLevelScreeningRequest {
    fn default() -> Self {
        Self {
            max_excess_sigma: DarkLevelScreening::default().max_excess_sigma,
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
    #[serde(default)]
    flat_star_masking: Option<FlatStarMaskingRequest>,
    #[serde(default)]
    dark_level_screening: Option<DarkLevelScreeningRequest>,
    /// An existing directory for a flat build's temporary pixels, in place
    /// of the system's temporary directory.
    #[serde(default)]
    scratch_directory: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct MasterBuildInputResponse {
    path: String,
    accepted_samples: u64,
    rejected_samples: u64,
    masked_samples: u64,
}

/// What flat star masking did, when it ran.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct FlatStarMaskingResponse {
    options: FlatStarMaskingRequest,
    masked_samples: u64,
    masked_pixels: u64,
    minimum_clean_samples: usize,
    maximum_clean_samples: usize,
    low_coverage_samples: u64,
    unmasked_saturation_samples: u64,
    unknown_saturation_inputs: usize,
}

impl From<&FlatStarMaskingStatistics> for FlatStarMaskingResponse {
    fn from(statistics: &FlatStarMaskingStatistics) -> Self {
        Self {
            options: FlatStarMaskingRequest::from(&statistics.options),
            masked_samples: statistics.masked_samples,
            masked_pixels: statistics.masked_pixels,
            minimum_clean_samples: statistics.minimum_clean_samples,
            maximum_clean_samples: statistics.maximum_clean_samples,
            low_coverage_samples: statistics.low_coverage_samples,
            unmasked_saturation_samples: statistics.unmasked_saturation_samples,
            unknown_saturation_inputs: statistics.unknown_saturation_inputs,
        }
    }
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
    masked_samples: u64,
    fallback_pixels: u64,
    defect_pixels_replaced: u64,
    bias_subtracted: bool,
    dark_subtracted: bool,
    normalized: bool,
    output_exposure_seconds: Option<f64>,
    rejection: MasterBuildRejectionResponse,
    rejection_method: &'static str,
    flat_star_masking: Option<FlatStarMaskingResponse>,
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
    false_match_probability: f64,
    /// `"header-checked"`, `"header-hint"`, `"blind"`, or
    /// `"header-unchecked"` when nothing solved and the file's own WCS
    /// stands unchecked.
    solved_from: &'static str,
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

impl From<FieldObject> for ObjectPointResponse {
    fn from(object: FieldObject) -> Self {
        Self {
            stable_id: object.stable_id,
            name: object.name,
            common_name: object.common_name,
            kind: object.kind.into(),
            source: object.source.into(),
            catalog_source: object.catalog_source,
            x: object.x,
            y: object.y,
            semi_major_pixels: object.semi_major_px,
            semi_minor_pixels: object.semi_minor_px,
            angle_degrees: object.angle_deg,
            prominence: object.prominence,
            ra_degrees: Some(object.ra_deg),
            dec_degrees: Some(object.dec_deg),
            discovered: object.discovered,
            near_capture: object.near_capture,
            distance_au: object.distance_au,
            motion_arcsec_per_hour: object.motion_arcsec_per_hour,
            direction_position_angle_degrees: object.direction_position_angle_deg,
            direction_image_angle_degrees: object.direction_image_angle_deg,
            outlines: object
                .outlines
                .into_iter()
                .map(ObjectOutlineResponse::from)
                .collect(),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ObjectOutlineResponse {
    geometry_id: String,
    source_record_id: String,
    role: String,
    quality: String,
    /// How the outline was made.
    method: String,
    /// What the outline rests on.
    evidence: String,
    level: Option<String>,
    contours: Vec<ObjectContourResponse>,
}

impl From<FieldOutline> for ObjectOutlineResponse {
    fn from(outline: FieldOutline) -> Self {
        Self {
            geometry_id: outline.geometry_id,
            source_record_id: outline.source_record_id,
            role: outline.role.as_str().into(),
            quality: outline.quality.as_str().into(),
            method: outline.method,
            evidence: outline.evidence,
            level: outline.level,
            contours: outline
                .contours
                .into_iter()
                .map(|contour| ObjectContourResponse {
                    closed: contour.closed,
                    points: contour.points.into_iter().map(|(x, y)| [x, y]).collect(),
                })
                .collect(),
        }
    }
}

#[derive(Debug, Serialize)]
struct ObjectContourResponse {
    closed: bool,
    points: Vec<[f64; 2]>,
}

/// A WCS as the C ABI returns it. `crpix` counts pixels from 0, with each
/// pixel's center on a whole number, as `pixelOrigin` says; add 1 for a
/// FITS header.
#[derive(Serialize)]
struct WcsResponse {
    crval: [f64; 2],
    crpix: [f64; 2],
    cd: [[f64; 2]; 2],
    sip: Option<SipResponse>,
    #[serde(rename = "pixelOrigin")]
    pixel_origin: u8,
}

impl WcsResponse {
    fn new(wcs: &seiza::Wcs) -> Self {
        Self {
            crval: [wcs.crval.0, wcs.crval.1],
            crpix: [wcs.crpix.0, wcs.crpix.1],
            cd: wcs.cd,
            sip: wcs.sip.as_ref().map(|sip| SipResponse {
                order: sip.order,
                a: sip.a.clone(),
                b: sip.b.clone(),
                ap: sip.ap.clone(),
                bp: sip.bp.clone(),
            }),
            pixel_origin: 0,
        }
    }
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

/// [`seiza_color_combine_rgb_json`]'s request. Paths are FITS or XISF mono
/// stacks on one pixel grid; `output` is the colour image to write.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ColorRgbRequest {
    red: String,
    green: String,
    blue: String,
    output: String,
    luminance_mode: Option<String>,
    normalization: Option<String>,
    black_percentile: Option<f32>,
    white_percentile: Option<f32>,
    normalization_samples: Option<usize>,
    crop: Option<String>,
}

/// [`seiza_color_combine_lrgb_json`]'s request.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ColorLrgbRequest {
    luminance: String,
    red: String,
    green: String,
    blue: String,
    output: String,
    luminance_weight: Option<f32>,
    luminance_mode: Option<String>,
    normalization: Option<String>,
    black_percentile: Option<f32>,
    white_percentile: Option<f32>,
    normalization_samples: Option<usize>,
    crop: Option<String>,
}

/// [`seiza_color_combine_narrowband_json`]'s request.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ColorNarrowbandRequest {
    ha: String,
    oiii: String,
    sii: Option<String>,
    output: String,
    palette: Option<String>,
    foraxx_target_median: Option<f32>,
    foraxx_shadows_clip: Option<f32>,
    normalization: Option<String>,
    black_percentile: Option<f32>,
    white_percentile: Option<f32>,
    normalization_samples: Option<usize>,
    crop: Option<String>,
}

/// [`seiza_color_combine_narrowband_matrix_json`]'s request.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ColorNarrowbandMatrixRequest {
    ha: String,
    oiii: String,
    sii: Option<String>,
    output: String,
    matrix: NarrowbandMatrixRequest,
    normalization: Option<String>,
    black_percentile: Option<f32>,
    white_percentile: Option<f32>,
    normalization_samples: Option<usize>,
    crop: Option<String>,
}

/// The filter weights of each output channel.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct NarrowbandMatrixRequest {
    red: NarrowbandMixRequest,
    green: NarrowbandMixRequest,
    blue: NarrowbandMixRequest,
}

/// One output channel's weights; a filter left out weighs zero.
#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
struct NarrowbandMixRequest {
    sii: f32,
    ha: f32,
    oiii: f32,
}

impl From<NarrowbandMixRequest> for NarrowbandMix {
    fn from(mix: NarrowbandMixRequest) -> Self {
        Self {
            sii: mix.sii,
            ha: mix.ha,
            oiii: mix.oiii,
        }
    }
}

impl From<NarrowbandMatrixRequest> for NarrowbandMatrix {
    fn from(matrix: NarrowbandMatrixRequest) -> Self {
        Self {
            red: matrix.red.into(),
            green: matrix.green.into(),
            blue: matrix.blue.into(),
        }
    }
}

/// What a colour composition wrote.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ColorCompositionResponse<'a> {
    schema_version: u32,
    label: &'a str,
    transfer: String,
    region: RegionResponse,
    crop: Option<CropReportResponse<'a>>,
}

/// The shared composition settings, with the core's defaults for those left
/// out.
fn color_options(
    normalization: Option<&str>,
    black_percentile: Option<f32>,
    white_percentile: Option<f32>,
    normalization_samples: Option<usize>,
    crop: Option<&str>,
) -> Result<ColorOptions, String> {
    let normalization = match normalization.map(str::to_ascii_lowercase).as_deref() {
        None | Some("percentile") => match ColorNormalization::default() {
            ColorNormalization::Percentile {
                black_percentile: black,
                white_percentile: white,
                max_samples,
            } => ColorNormalization::Percentile {
                black_percentile: black_percentile.unwrap_or(black),
                white_percentile: white_percentile.unwrap_or(white),
                max_samples: normalization_samples.unwrap_or(max_samples),
            },
            ColorNormalization::None => ColorNormalization::None,
        },
        Some("none") => ColorNormalization::None,
        Some(other) => {
            return Err(format!(
                "normalization must be \"none\" or \"percentile\", not {other:?}"
            ));
        }
    };
    let crop = crop
        .unwrap_or("none")
        .parse::<ColorCrop>()
        .map_err(|error| error.to_string())?;
    Ok(ColorOptions {
        normalization,
        crop,
        ..ColorOptions::default()
    })
}

/// Read the channel stacks `inputs` (role and path), compose them, and write
/// the result to `output` with the first channel's WCS. Every input and the
/// output must be a different file. `cancel` is checked before each read and
/// before the write.
fn compose_color_files(
    inputs: &[(&str, &str)],
    output: &str,
    label: &str,
    options: &ColorOptions,
    cancel: *const SeizaCancelSignal,
    compose: impl FnOnce(&[LinearImage]) -> seiza_stacking::Result<ColorComposition>,
) -> Result<*mut c_char, String> {
    let cancelled = || {
        unsafe { cancel.as_ref() }.is_some_and(|signal| signal.cancelled.load(Ordering::Relaxed))
    };
    let output = Path::new(output);
    let mut seen = HashMap::new();
    for (role, path) in inputs
        .iter()
        .map(|(role, path)| (*role, Path::new(*path)))
        .chain(std::iter::once(("output", output)))
    {
        if path.as_os_str().is_empty() {
            return Err(format!("the {role} path is required"));
        }
        if let Some(other) = seen.insert(path_identity(path), role) {
            return Err(format!(
                "the {other} and {role} paths refer to the same file {}",
                path.display()
            ));
        }
    }
    let mut images = Vec::with_capacity(inputs.len());
    let mut reference_headers = Vec::new();
    for (index, (role, path)) in inputs.iter().enumerate() {
        if cancelled() {
            return Err(seiza_stacking::Error::Cancelled.to_string());
        }
        let frame = FitsFrame::open(path)
            .map_err(|error| format!("failed to read the {role} stack {path}: {error}"))?;
        if index == 0 {
            reference_headers = frame.headers;
        }
        images.push(frame.image);
    }
    let composition = compose(&images).map_err(|error| error.to_string())?;
    if cancelled() {
        return Err(seiza_stacking::Error::Cancelled.to_string());
    }
    write_color_fits_f32(output, &composition, &reference_headers, label)
        .map_err(|error| error.to_string())?;
    owned_json(&ColorCompositionResponse {
        schema_version: 1,
        label,
        transfer: composition.transfer.fits_name().to_ascii_lowercase(),
        region: RegionResponse::new(composition.region),
        crop: composition
            .crop
            .as_ref()
            .map(|report| CropReportResponse::new(options.crop, report)),
    })
}

#[unsafe(no_mangle)]
/// Combines aligned red, green and blue mono stacks into one RGB image and
/// writes it, as `seiza color rgb` does without registering the channels.
///
/// `request_json` is an object with the `red`, `green` and `blue` FITS or
/// XISF paths, all on one pixel grid, and the `output` path, written as XISF
/// for a `.xisf` name and FITS otherwise. The output carries the red stack's
/// WCS, moved by any crop. Optional fields:
/// - `luminanceMode`: `"native"` (the default) keeps the channels as they
///   are; `"super"` scales each pixel to a synthetic luminance of R + G + B,
///   which may exceed one.
/// - `normalization`: `"percentile"` (the default) maps each channel's
///   `blackPercentile` (0.001) and `whitePercentile` (0.995), estimated from
///   up to `normalizationSamples` (1000000) samples, to 0 and 1; `"none"`
///   keeps the samples.
/// - `crop`: `"none"` (the default), `"bounds"` or `"inscribed"`, as in
///   [`seiza_color_crop_report_json`].
///
/// Returns owned schema-1 JSON: the `label` written to `SEIZACLR` (`RGB` or
/// `SUPER-RGB`), the `transfer` of the samples (`"linear"` here), the
/// `region` of the grid the output covers, and `crop`, the object
/// [`seiza_color_crop_report_json`] returns, when a crop was asked for (null
/// otherwise). Free it with [`seiza_string_free`]. `cancel` is checked before
/// each stack is read and before the output is written.
///
/// # Safety
/// `request_json` must be a valid NUL-terminated string. `cancel` must be
/// null or a live [`SeizaCancelSignal`] retained until this call returns.
/// When non-null, `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_color_combine_rgb_json(
    request_json: *const c_char,
    cancel: *const SeizaCancelSignal,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let request: ColorRgbRequest = parse_request(request_json, "RGB composition")?;
        let mode = match request
            .luminance_mode
            .as_deref()
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            None | Some("native") => RgbLuminance::Native,
            Some("super") => RgbLuminance::Super,
            Some(other) => {
                return Err(format!(
                    "luminanceMode must be \"native\" or \"super\", not {other:?}"
                ));
            }
        };
        let options = color_options(
            request.normalization.as_deref(),
            request.black_percentile,
            request.white_percentile,
            request.normalization_samples,
            request.crop.as_deref(),
        )?;
        compose_color_files(
            &[
                ("red", &request.red),
                ("green", &request.green),
                ("blue", &request.blue),
            ],
            &request.output,
            mode.label(),
            &options,
            cancel,
            |images| match mode {
                RgbLuminance::Native => combine_rgb(&images[0], &images[1], &images[2], &options),
                RgbLuminance::Super => {
                    combine_super_rgb(&images[0], &images[1], &images[2], &options)
                }
            },
        )
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Combines aligned luminance, red, green and blue mono stacks into one LRGB
/// image and writes it, as `seiza color lrgb` does without registering the
/// channels.
///
/// `request_json` is an object with the `luminance`, `red`, `green` and
/// `blue` paths and the `output` path, as in
/// [`seiza_color_combine_rgb_json`]; the output carries the luminance
/// stack's WCS. `luminanceMode` `"replace"` (the default) swaps in the
/// luminance, weighted by `luminanceWeight` from 0 (RGB alone) to 1 (the
/// default, L alone); `"super"` targets the additive L + R + G + B, which may
/// exceed one, and takes no weight. `normalization`, `blackPercentile`,
/// `whitePercentile`, `normalizationSamples` and `crop` are those of
/// [`seiza_color_combine_rgb_json`], and so are the returned JSON (with the
/// label `LRGB` or `SUPER-LRGB`) and `cancel`.
///
/// # Safety
/// As [`seiza_color_combine_rgb_json`].
pub unsafe extern "C" fn seiza_color_combine_lrgb_json(
    request_json: *const c_char,
    cancel: *const SeizaCancelSignal,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let request: ColorLrgbRequest = parse_request(request_json, "LRGB composition")?;
        let super_luminance = match request
            .luminance_mode
            .as_deref()
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            None | Some("replace") => false,
            Some("super") => true,
            Some(other) => {
                return Err(format!(
                    "luminanceMode must be \"replace\" or \"super\", not {other:?}"
                ));
            }
        };
        let mode = LrgbLuminance::new(super_luminance, request.luminance_weight)
            .map_err(color_request_error)?;
        let options = color_options(
            request.normalization.as_deref(),
            request.black_percentile,
            request.white_percentile,
            request.normalization_samples,
            request.crop.as_deref(),
        )?;
        compose_color_files(
            &[
                ("luminance", &request.luminance),
                ("red", &request.red),
                ("green", &request.green),
                ("blue", &request.blue),
            ],
            &request.output,
            mode.label(),
            &options,
            cancel,
            |images| {
                let [luminance, red, green, blue] = images else {
                    unreachable!("four channels were read")
                };
                match mode {
                    LrgbLuminance::Replace { luminance_weight } => {
                        combine_lrgb(luminance, red, green, blue, luminance_weight, &options)
                    }
                    LrgbLuminance::Super => {
                        combine_super_lrgb(luminance, red, green, blue, &options)
                    }
                }
            },
        )
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Maps aligned H-alpha, OIII and, for most palettes, SII mono stacks onto an
/// RGB palette and writes the result, as `seiza color narrowband` does
/// without registering the channels.
///
/// `request_json` is an object with the `ha` and `oiii` paths, `sii` when the
/// palette uses it, and the `output` path, as in
/// [`seiza_color_combine_rgb_json`]; the output carries the H-alpha stack's
/// WCS. `palette` is one of `SHO` (the default), `SOH`, `HSO`, `HOS`, `OSH`,
/// `OHS`, `HOO`, `Foraxx-SHO` or `Foraxx-HOO`, in any case; `HOO` and
/// `Foraxx-HOO` take no `sii`, and the rest need one. The Foraxx palettes
/// stretch each channel first, setting its median to `foraxxTargetMedian`
/// (0.2) and its black point `foraxxShadowsClip` (-2.8) normal-equivalent
/// MADs below the median, so their output is display-referred. The other
/// fields are those of [`seiza_color_combine_rgb_json`], and so are the
/// returned JSON (with the palette's upper-case name as the label and
/// `"display"` as the Foraxx palettes' transfer) and `cancel`.
///
/// # Safety
/// As [`seiza_color_combine_rgb_json`].
pub unsafe extern "C" fn seiza_color_combine_narrowband_json(
    request_json: *const c_char,
    cancel: *const SeizaCancelSignal,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let request: ColorNarrowbandRequest =
            parse_request(request_json, "narrowband composition")?;
        let palette = request
            .palette
            .as_deref()
            .unwrap_or("sho")
            .parse::<NarrowbandPalette>()
            .map_err(|error| error.to_string())?;
        palette
            .check_sii(request.sii.is_some())
            .map_err(color_request_error)?;
        let defaults = ForaxxOptions::default();
        let foraxx = ForaxxOptions {
            target_median: request
                .foraxx_target_median
                .unwrap_or(defaults.target_median),
            shadows_clip: request.foraxx_shadows_clip.unwrap_or(defaults.shadows_clip),
        };
        let options = color_options(
            request.normalization.as_deref(),
            request.black_percentile,
            request.white_percentile,
            request.normalization_samples,
            request.crop.as_deref(),
        )?;
        let mut inputs = vec![("H-alpha", request.ha.as_str()), ("OIII", &request.oiii)];
        if let Some(sii) = &request.sii {
            inputs.push(("SII", sii));
        }
        compose_color_files(
            &inputs,
            &request.output,
            palette.name(),
            &options,
            cancel,
            |images| {
                combine_narrowband(
                    &images[0],
                    &images[1],
                    images.get(2),
                    palette,
                    &options,
                    &foraxx,
                )
            },
        )
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Mixes aligned H-alpha, OIII and optional SII mono stacks into RGB through
/// a caller's own linear matrix, and writes the result.
///
/// `request_json` is an object with the `ha` and `oiii` paths, an optional
/// `sii` path, the `output` path, and `matrix`: `{"red": {...}, "green":
/// {...}, "blue": {...}}`, each output channel giving its `sii`, `ha` and
/// `oiii` weights (a weight left out is zero). Weights must be finite and
/// some must be non-zero. A matrix that weights SII needs an `sii` stack; one
/// that does not ignores any given. The output carries the H-alpha stack's
/// WCS and the label `NARROWBAND-MATRIX`, and is linear light. The other
/// fields are those of [`seiza_color_combine_rgb_json`], and so are the
/// returned JSON and `cancel`.
///
/// # Safety
/// As [`seiza_color_combine_rgb_json`].
pub unsafe extern "C" fn seiza_color_combine_narrowband_matrix_json(
    request_json: *const c_char,
    cancel: *const SeizaCancelSignal,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let request: ColorNarrowbandMatrixRequest =
            parse_request(request_json, "narrowband matrix composition")?;
        let matrix = NarrowbandMatrix::from(request.matrix);
        let options = color_options(
            request.normalization.as_deref(),
            request.black_percentile,
            request.white_percentile,
            request.normalization_samples,
            request.crop.as_deref(),
        )?;
        let mut inputs = vec![("H-alpha", request.ha.as_str()), ("OIII", &request.oiii)];
        if let Some(sii) = &request.sii {
            inputs.push(("SII", sii));
        }
        compose_color_files(
            &inputs,
            &request.output,
            NarrowbandMatrix::LABEL,
            &options,
            cancel,
            |images| {
                combine_narrowband_matrix(&images[0], &images[1], images.get(2), matrix, &options)
            },
        )
    })
    .unwrap_or(ptr::null_mut())
}

/// A request the core refused before any work, in the core's own words.
fn color_request_error(error: seiza_stacking::Error) -> String {
    match error {
        seiza_stacking::Error::Color(message) => message,
        other => other.to_string(),
    }
}

/// Parse a required JSON request object.
fn parse_request<T: serde::de::DeserializeOwned>(
    request_json: *const c_char,
    name: &str,
) -> Result<T, String> {
    let request_json = required_str(request_json, &format!("{name} request"))?;
    serde_json::from_str(&request_json).map_err(|error| format!("invalid {name} request: {error}"))
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

/// A Gaia DR3 source in the JSON [`seiza_gaia_photometry_cone_json`]
/// returns and [`seiza_color_calibrate_json`] reads.
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct GaiaSourceJson {
    ra: f64,
    dec: f64,
    #[serde(default)]
    pmra: Option<f64>,
    #[serde(default)]
    pmdec: Option<f64>,
    g: f32,
    #[serde(default)]
    bp: Option<f32>,
    #[serde(default)]
    rp: Option<f32>,
    #[serde(default)]
    ruwe: Option<f32>,
    /// BP − RP, used in place of `bp` and `rp`: what
    /// [`seiza_gaia_photometry_catalog_cone_json`] gives.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    bp_rp: Option<f32>,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase", default, deny_unknown_fields)]
struct ColorCalibrationOptionsJson {
    white_bp_rp: Option<f32>,
    /// The BP − RP range of the stars the fits use, as `[bluest, reddest]`.
    fit_bp_rp_range: Option<[f32; 2]>,
    aperture_radius: Option<f64>,
    /// Robust sigma beyond which a star leaves a fit.
    rejection_sigma: Option<f64>,
    /// The fewest stars either fit may rest on.
    minimum_stars: Option<usize>,
    neutralize_background: Option<bool>,
    /// The faintest Gaia G magnitude a calibrator may have.
    calibrator_max_g: Option<f32>,
    /// Julian year of the observation, for proper motion.
    observation_epoch: Option<f64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ColorFitJson {
    intercept: f64,
    slope: f64,
    scatter: f64,
    stars: usize,
}

/// One star a colour calibration measured.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ColorStarMeasurementJson {
    x: f64,
    y: f64,
    g: f32,
    bp_rp: f32,
    /// Background-subtracted, aperture-corrected fluxes in R, G, B.
    flux: [f64; 3],
    /// Whether the red and the blue fit kept the star.
    used: [bool; 2],
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ColorCalibrationJson {
    schema_version: u32,
    gains: [f32; 3],
    offsets: [f32; 3],
    background: [f32; 3],
    red_fit: ColorFitJson,
    blue_fit: ColorFitJson,
    white_bp_rp: f32,
    aperture_radius: f64,
    aperture_correction: [f64; 3],
    stars_offered: usize,
    stars_measured: usize,
    measurements: Vec<ColorStarMeasurementJson>,
}

impl GaiaSourceJson {
    fn source(&self) -> seiza_stacking::GaiaColorSource {
        seiza_stacking::GaiaColorSource {
            ra: self.ra,
            dec: self.dec,
            pmra: self.pmra,
            pmdec: self.pmdec,
            g: self.g,
            bp_rp: self
                .bp_rp
                .or_else(|| self.bp.zip(self.rp).map(|(bp, rp)| bp - rp)),
            ruwe: self.ruwe,
        }
    }
}

impl ColorCalibrationOptionsJson {
    /// The options a call gave, or the defaults for a null pointer.
    fn parse(options_json: *const c_char) -> Result<Self, String> {
        if options_json.is_null() {
            return Ok(Self::default());
        }
        serde_json::from_str(&required_str(options_json, "options")?)
            .map_err(|error| format!("invalid colour calibration options: {error}"))
    }

    fn calibration_options(&self) -> seiza_stacking::ColorCalibrationOptions {
        let defaults = seiza_stacking::ColorCalibrationOptions::default();
        seiza_stacking::ColorCalibrationOptions {
            white_bp_rp: self.white_bp_rp.unwrap_or(defaults.white_bp_rp),
            fit_bp_rp_range: self
                .fit_bp_rp_range
                .map_or(defaults.fit_bp_rp_range, |[bluest, reddest]| {
                    (bluest, reddest)
                }),
            aperture_radius: self.aperture_radius,
            rejection_sigma: self.rejection_sigma.unwrap_or(defaults.rejection_sigma),
            minimum_stars: self.minimum_stars.unwrap_or(defaults.minimum_stars),
            neutralize_background: self
                .neutralize_background
                .unwrap_or(defaults.neutralize_background),
            calibrator_max_g: self.calibrator_max_g,
        }
    }
}

impl ColorCalibrationJson {
    fn new(calibration: &seiza_stacking::ColorCalibration) -> Self {
        let fit = |fit: &seiza_stacking::ColorFit| ColorFitJson {
            intercept: fit.intercept,
            slope: fit.slope,
            scatter: fit.scatter,
            stars: fit.stars,
        };
        Self {
            schema_version: 1,
            gains: calibration.gains,
            offsets: calibration.offsets,
            background: calibration.background,
            red_fit: fit(&calibration.red_fit),
            blue_fit: fit(&calibration.blue_fit),
            white_bp_rp: calibration.white_bp_rp,
            aperture_radius: calibration.aperture_radius,
            aperture_correction: calibration.aperture_correction,
            stars_offered: calibration.stars_offered,
            stars_measured: calibration.stars_measured,
            measurements: calibration
                .measurements
                .iter()
                .map(|star| ColorStarMeasurementJson {
                    x: star.x,
                    y: star.y,
                    g: star.g,
                    bp_rp: star.bp_rp,
                    flux: star.flux,
                    used: star.used,
                })
                .collect(),
        }
    }
}

/// What a colour calibration against Gaia did: `status` is `calibrated`,
/// with the `calibration`, or `skipped`, with the `reason` and its `detail`.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ColorCalibrationOutcomeJson {
    schema_version: u32,
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<seiza_stacking::ColorCalibrationSkipReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    calibration: Option<ColorCalibrationJson>,
}

impl ColorCalibrationOutcomeJson {
    fn new(outcome: &seiza_stacking::ColorCalibrationOutcome) -> Self {
        match outcome {
            seiza_stacking::ColorCalibrationOutcome::Calibrated(calibration) => Self {
                schema_version: 1,
                status: "calibrated",
                reason: None,
                detail: None,
                calibration: Some(ColorCalibrationJson::new(calibration)),
            },
            seiza_stacking::ColorCalibrationOutcome::Skipped(skip) => Self {
                schema_version: 1,
                status: "skipped",
                reason: Some(skip.reason),
                detail: Some(skip.detail.clone()),
                calibration: None,
            },
        }
    }
}

/// A colour calibration and stretch, as
/// [`seiza_color_calibrate_and_stretch_json`] returns it.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CalibratedStretchJson {
    schema_version: u32,
    calibration: ColorCalibrationOutcomeJson,
    stretch_plans: Vec<seiza_stretch::StretchPlan>,
    /// Stages that asked to stretch each channel on its own and were linked
    /// because calibration ran.
    linked_stages: Vec<usize>,
}

/// A pixel mapping from one image's grid to another's: `matrix` times the
/// source position, plus the translation.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AffineTransformJson {
    matrix: [[f64; 2]; 2],
    #[serde(default)]
    translation_x: f64,
    #[serde(default)]
    translation_y: f64,
}

/// The inputs shared by the colour calibration calls that read Gaia
/// themselves: the image's solution, if any, and where its Gaia colours
/// come from.
struct GaiaColorInputs {
    wcs: Option<seiza::Wcs>,
    sources: Option<Vec<seiza_stacking::GaiaColorSource>>,
    catalog_path: Option<PathBuf>,
    options: ColorCalibrationOptionsJson,
}

impl GaiaColorInputs {
    fn parse(
        wcs_json: *const c_char,
        gaia_json: *const c_char,
        catalog_path: *const c_char,
        options_json: *const c_char,
    ) -> Result<Self, String> {
        let wcs = if wcs_json.is_null() {
            None
        } else {
            let request: parallax::WcsRequest =
                serde_json::from_str(&required_str(wcs_json, "WCS")?)
                    .map_err(|error| format!("invalid WCS: {error}"))?;
            Some(request.wcs()?)
        };
        let sources = if gaia_json.is_null() {
            None
        } else {
            let gaia: Vec<GaiaSourceJson> =
                serde_json::from_str(&required_str(gaia_json, "Gaia sources")?)
                    .map_err(|error| format!("Gaia sources must be a JSON array: {error}"))?;
            Some(gaia.iter().map(GaiaSourceJson::source).collect())
        };
        Ok(Self {
            wcs,
            sources,
            catalog_path: optional_path(catalog_path)?,
            options: ColorCalibrationOptionsJson::parse(options_json)?,
        })
    }

    fn gaia(&self) -> seiza_stacking::GaiaColors<'_> {
        match &self.sources {
            Some(sources) => seiza_stacking::GaiaColors::Sources {
                sources,
                epoch: self.options.observation_epoch,
            },
            None => seiza_stacking::GaiaColors::CatalogPath(self.catalog_path.as_deref()),
        }
    }
}

/// The interleaved RGB image a caller lends, to read and then overwrite.
unsafe fn rgb_image_from_ffi<'a>(
    data: *mut f32,
    length: usize,
    width: usize,
    height: usize,
) -> Result<(&'a mut [f32], LinearImage), String> {
    let image = unsafe { linear_image_from_ffi(data, length, width, height, 3, "image")? };
    let samples = unsafe { std::slice::from_raw_parts_mut(data, length) };
    Ok((samples, image))
}

#[unsafe(no_mangle)]
/// Fetch Gaia DR3 photometry within `radius_deg` of `(ra, dec)` (degrees)
/// down to G `max_mag` from the ESA Gaia archive, for
/// [`seiza_color_calibrate_json`].
///
/// Returns an owned JSON array of `{ra, dec, pmra, pmdec, g, bp, rp, ruwe}`
/// objects (J2016.0 positions; proper motions in mas/yr; absent values
/// null). The call blocks on the network, a wide field for up to several
/// minutes; cache the result. Free it, or an `error_out` string, with
/// [`seiza_string_free`].
///
/// # Safety
/// When non-null, `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_gaia_photometry_cone_json(
    ra: f64,
    dec: f64,
    radius_deg: f64,
    max_mag: f32,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let downloader =
            seiza_sources::SourceDownloader::new().map_err(|error| error.to_string())?;
        let stars = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| error.to_string())?
            .block_on(downloader.gaia_photometry_cone(ra, dec, radius_deg, max_mag))
            .map_err(|error| error.to_string())?;
        owned_json(
            &stars
                .into_iter()
                .map(|star| GaiaSourceJson {
                    ra: star.ra,
                    dec: star.dec,
                    pmra: star.pmra,
                    pmdec: star.pmdec,
                    g: star.g,
                    bp: star.bp,
                    rp: star.rp,
                    ruwe: star.ruwe,
                    bp_rp: None,
                })
                .collect::<Vec<_>>(),
        )
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Read Gaia DR3 photometry within `radius_deg` of `(ra, dec)` (degrees)
/// down to G `max_mag` from the local catalog [`seiza_catalog_setup`]
/// installs (`stars-gaia-photometry.bin`), for
/// [`seiza_color_calibrate_json`]. `catalog_path` is the catalog file or the
/// directory that holds it; null looks in the standard places.
///
/// Returns an owned JSON array of sources like
/// [`seiza_gaia_photometry_cone_json`]'s, but with positions already at the
/// catalog's epoch, so `pmra` and `pmdec` are null, and the colour as
/// `bpRp` in place of `bp` and `rp`; `bpRp` is left out when Gaia's is
/// missing or its astrometry is unreliable. Free the result, or an
/// `error_out` string, with [`seiza_string_free`].
///
/// # Safety
/// `catalog_path` must be null or a valid NUL-terminated string. When
/// non-null, `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_gaia_photometry_catalog_cone_json(
    catalog_path: *const c_char,
    ra: f64,
    dec: f64,
    radius_deg: f64,
    max_mag: f32,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let catalog_path = optional_path(catalog_path)?;
        let path = seiza::data_paths::gaia_photometry(catalog_path.as_deref())
            .map_err(|error| error.to_string())?
            .ok_or("no Gaia photometry catalog is installed")?;
        let catalog = seiza::catalog::PhotometryCatalog::open(&path)
            .map_err(|error| format!("failed to open {}: {error}", path.display()))?;
        owned_json(
            &catalog
                .cone_search(ra, dec, radius_deg, max_mag)
                .into_iter()
                .map(|star| GaiaSourceJson {
                    ra: star.ra,
                    dec: star.dec,
                    pmra: None,
                    pmdec: None,
                    g: star.g,
                    bp: None,
                    rp: None,
                    ruwe: None,
                    bp_rp: star.calibration_bp_rp(),
                })
                .collect::<Vec<_>>(),
        )
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Fit photometric colour calibration for an interleaved linear RGB image
/// against Gaia DR3 star colours.
///
/// `wcs_json` is a JSON object of the image's FITS WCS keywords (`CTYPE1`,
/// `CRVAL1`, `CRPIX1`, `CD1_1` or `CDELT1`, SIP terms, …) with numeric or
/// string values; `gaia_json` is an array of sources as
/// [`seiza_gaia_photometry_cone_json`] or
/// [`seiza_gaia_photometry_catalog_cone_json`] returns; a `bpRp` value is
/// used in place of `bp` and `rp`. `options_json` may be null
/// or an object with `whiteBpRp` (default 0.82, the Sun), `fitBpRpRange`
/// (the `[bluest, reddest]` BP − RP of the stars the fits use; default
/// `[0, 1.8]`), `apertureRadius` (pixels; default twice the measured FWHM),
/// `rejectionSigma` (default 3), `minimumStars` (the fewest stars either fit
/// may rest on; default 20, at least 3), `neutralizeBackground` (default
/// true), `calibratorMaxG` (the faintest Gaia G a calibrator may have;
/// default two magnitudes brighter than the faintest source given) and
/// `observationEpoch` (a Julian year, for proper motion).
///
/// Returns owned schema-1 JSON with `gains` and `offsets` for R, G, B (apply
/// them with [`seiza_color_calibration_apply`]), each channel's
/// `background`, the `redFit` and `blueFit` lines of instrumental colour
/// against BP − RP, the `apertureRadius` and `apertureCorrection` used, star
/// counts, and `measurements`: each cleanly measured star's `x`, `y`, `g`,
/// `bpRp`, R, G, B `flux`, and `used`, whether the red and the blue fit kept
/// it. The image is not changed. Free the result, or an `error_out` string,
/// with [`seiza_string_free`].
///
/// # Safety
/// `data` must point to `length` readable floats, `width * height * 3` of
/// them. The JSON arguments must be valid NUL-terminated UTF-8 strings, and
/// `options_json` may be null. When non-null, `error_out` must point to
/// writable storage for one pointer.
pub unsafe extern "C" fn seiza_color_calibrate_json(
    data: *const f32,
    length: usize,
    width: usize,
    height: usize,
    wcs_json: *const c_char,
    gaia_json: *const c_char,
    options_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let image = unsafe { linear_image_from_ffi(data, length, width, height, 3, "image")? };
        let wcs: serde_json::Map<String, Value> =
            serde_json::from_str(&required_str(wcs_json, "WCS keywords")?)
                .map_err(|error| format!("WCS keywords must be a JSON object: {error}"))?;
        let wcs = seiza::Wcs::from_fits_values(
            |key| match wcs.get(key)? {
                Value::Number(number) => number.as_f64(),
                Value::String(text) => text.trim().parse().ok(),
                _ => None,
            },
            |key| wcs.get(key)?.as_str().map(str::to_owned),
        )
        .ok_or("the WCS keywords do not describe a TAN projection")?;
        let gaia: Vec<GaiaSourceJson> =
            serde_json::from_str(&required_str(gaia_json, "Gaia sources")?)
                .map_err(|error| format!("Gaia sources must be a JSON array: {error}"))?;
        let options = ColorCalibrationOptionsJson::parse(options_json)?;
        let sources = gaia.iter().map(GaiaSourceJson::source).collect::<Vec<_>>();
        let stars = seiza_stacking::place_gaia_sources(
            &wcs,
            width,
            height,
            options.observation_epoch,
            &sources,
        );
        let calibration =
            seiza_stacking::calibrate_color(&image, &stars, &options.calibration_options())
                .map_err(|error| error.to_string())?;
        owned_json(&ColorCalibrationJson::new(&calibration))
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Apply colour calibration gains and offsets, as
/// [`seiza_color_calibrate_json`] returns them, to an interleaved linear RGB
/// image in place: each finite sample becomes `sample * gain + offset` for
/// its channel.
///
/// # Safety
/// `data` must point to `length` writable floats, a multiple of three.
/// `gains` and `offsets` must each point to three readable floats. When
/// non-null, `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_color_calibration_apply(
    data: *mut f32,
    length: usize,
    gains: *const f32,
    offsets: *const f32,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        if data.is_null() || gains.is_null() || offsets.is_null() {
            return Err("image, gains and offsets are required".into());
        }
        let gains = unsafe { gains.cast::<[f32; 3]>().read_unaligned() };
        let offsets = unsafe { offsets.cast::<[f32; 3]>().read_unaligned() };
        let data = unsafe { std::slice::from_raw_parts_mut(data, length) };
        seiza_stacking::apply_color_gains(data, gains, offsets).map_err(|error| error.to_string())
    })
    .is_some()
}

#[unsafe(no_mangle)]
/// Colour calibrate an interleaved linear RGB image in place against Gaia
/// DR3 star colours: place the Gaia stars through the image's solution, fit
/// the gains, and apply them.
///
/// `wcs_json` is the image's own solution as [`seiza_solve_image_json`]
/// returns it (`crval`, `crpix`, `cd`, optional `sip` and `pixelOrigin`);
/// for an image made from a solved frame, derive it with
/// [`seiza_wcs_transform_json`]. Null means there is none. `gaia_json` is
/// an array of sources as [`seiza_gaia_photometry_cone_json`] returns, and
/// when given it is used and `catalog_path` is ignored. When it is null the
/// Gaia colours come from the offline catalog at `catalog_path` (the file
/// or the directory that holds it; null looks in the standard places).
/// `options_json` is as [`seiza_color_calibrate_json`] takes it; its
/// `observationEpoch` applies to `gaia_json` sources only, since the
/// catalog's positions are already at its own epoch.
///
/// When calibration cannot run, the image is left as it was. Either way
/// the call returns owned JSON: `{"schemaVersion": 1, "status":
/// "calibrated", "calibration": {...}}` with the calibration as
/// [`seiza_color_calibrate_json`] gives it, or `{"schemaVersion": 1,
/// "status": "skipped", "reason": ..., "detail": ...}` with `reason` one of
/// `no_solution`, `no_catalog`, `no_stars`, `too_few_stars`, `not_rgb`,
/// `invalid_options` and `fit_failed`. Errors are for malformed arguments
/// only. Free the
/// result, or an `error_out` string, with [`seiza_string_free`].
///
/// # Safety
/// `data` must point to `length` writable floats, `width * height * 3` of
/// them. The JSON and path arguments must each be null or a valid
/// NUL-terminated UTF-8 string. When non-null, `error_out` must point to
/// writable storage for one pointer.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn seiza_color_calibrate_image_json(
    data: *mut f32,
    length: usize,
    width: usize,
    height: usize,
    wcs_json: *const c_char,
    gaia_json: *const c_char,
    catalog_path: *const c_char,
    options_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let (samples, mut image) = unsafe { rgb_image_from_ffi(data, length, width, height)? };
        let inputs = GaiaColorInputs::parse(wcs_json, gaia_json, catalog_path, options_json)?;
        let outcome = seiza_stacking::calibrate_color_image(
            &mut image,
            inputs.wcs.as_ref(),
            inputs.gaia(),
            &inputs.options.calibration_options(),
        );
        samples.copy_from_slice(&image.data);
        owned_json(&ColorCalibrationOutcomeJson::new(&outcome))
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Colour calibrate an interleaved linear RGB image against Gaia, as
/// [`seiza_color_calibrate_image_json`] does, then display-stretch it, and
/// write the stretched image over `data`. `stretch_json` is a stretch
/// configuration or an array of them applied in order, as
/// [`seiza_rendered_image_open_with_stretch_config`] takes its `stretch`.
/// A statistics-driven stretch such as Auto-MTF sees the calibrated colours.
/// Once calibration has run, a stage whose `color_strategy` is `unlinked`
/// is linked instead, so the balance survives; when calibration cannot run,
/// every stage runs as given and the image is still stretched. Returns
/// owned JSON `{"schemaVersion": 1, "calibration": {...}, "stretchPlans":
/// [...], "linkedStages": [...]}`: the calibration outcome as
/// [`seiza_color_calibrate_image_json`] gives it, each stage's resolved
/// stretch plan, and the stages (counted from 0) that were linked. Fails only on malformed arguments or a failed stretch, and
/// then leaves `data` as it was. Free the result, or an `error_out` string,
/// with [`seiza_string_free`].
///
/// # Safety
/// As [`seiza_color_calibrate_image_json`]; `stretch_json` must be a valid
/// NUL-terminated UTF-8 string.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn seiza_color_calibrate_and_stretch_json(
    data: *mut f32,
    length: usize,
    width: usize,
    height: usize,
    wcs_json: *const c_char,
    gaia_json: *const c_char,
    catalog_path: *const c_char,
    options_json: *const c_char,
    stretch_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let (samples, image) = unsafe { rgb_image_from_ffi(data, length, width, height)? };
        let inputs = GaiaColorInputs::parse(wcs_json, gaia_json, catalog_path, options_json)?;
        let stretch: StretchConfigRequest =
            serde_json::from_str(&required_str(stretch_json, "stretch")?)
                .map_err(|error| format!("invalid stretch: {error}"))?;
        let result = seiza_stacking::calibrate_and_stretch(
            image,
            inputs.wcs.as_ref(),
            inputs.gaia(),
            &inputs.options.calibration_options(),
            &stretch.into_stack(),
        )
        .map_err(|error| error.to_string())?;
        samples.copy_from_slice(&result.image.data);
        owned_json(&CalibratedStretchJson {
            schema_version: 1,
            calibration: ColorCalibrationOutcomeJson::new(&result.calibration),
            stretch_plans: result.stretch_plans,
            linked_stages: result.linked_stages,
        })
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// The solution of an image made from a solved one by a pixel mapping, such
/// as registration, a turn, binning, drizzle or a crop. `wcs_json` is the
/// solved image's WCS as [`seiza_solve_image_json`] returns it;
/// `transform_json` is `{"matrix": [[a, b], [c, d]], "translationX": tx,
/// "translationY": ty}`, taking a solved-image pixel `(x, y)` to `(a x + b
/// y + tx, c x + d y + ty)` on the new image. The transform always counts
/// pixels from 0 with their centres on whole numbers, whatever the WCS's
/// `pixelOrigin`: a crop at `(x, y)` is the identity matrix with
/// translation `(-x, -y)`, a half turn of a `w` × `h` image is `-1` on the
/// diagonal with translation `(w - 1, h - 1)`, and drizzle or binning by a
/// scale `s` is `s` on the diagonal with translation `((s - 1) / 2, (s - 1)
/// / 2)`.
///
/// Returns the new image's WCS as owned JSON in the same form, `crpix`
/// counting from 0. Fails when the mapping cannot be inverted, or when the
/// WCS has SIP terms and the mapping is more than a translation. Free the
/// result, or an `error_out` string, with [`seiza_string_free`].
///
/// # Safety
/// Both JSON arguments must be valid NUL-terminated UTF-8 strings. When
/// non-null, `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_wcs_transform_json(
    wcs_json: *const c_char,
    transform_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let request: parallax::WcsRequest = serde_json::from_str(&required_str(wcs_json, "WCS")?)
            .map_err(|error| format!("invalid WCS: {error}"))?;
        let transform: AffineTransformJson =
            serde_json::from_str(&required_str(transform_json, "transform")?)
                .map_err(|error| format!("invalid transform: {error}"))?;
        let wcs = seiza_stacking::transform_wcs(
            &request.wcs()?,
            seiza_stacking::AffineTransform {
                matrix: transform.matrix,
                translation_x: transform.translation_x,
                translation_y: transform.translation_y,
            },
        )
        .map_err(|error| error.to_string())?;
        owned_json(&WcsResponse::new(&wcs))
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Choose a reference from an ordered JSON array of FITS or XISF path strings.
/// Uses the native stacking reference scorer: among frames with nearly the
/// best star quality, choose the flattest background. This only reads inputs;
/// it does not calibrate, register, stack, or modify them.
///
/// Returns owned schema-1 JSON with `referenceIndex` (zero-based),
/// `referencePath` (exactly the input string at that index), `ranking` (the
/// indices of every scored frame, best reference first, so `referenceIndex`
/// leads it), and `scores` (one entry per input in the original order). A
/// score is null for an unreadable or unscorable frame; otherwise it
/// contains `stars`, `medianStarArea`, `background`, `backgroundVariation`,
/// and `score`.
/// `medianStarArea` is in half-resolution pixels, and background measurements
/// use the scorer's half-resolution luminance units, not display pixels.
/// See `seiza_stacking::ReferenceScore` for the native measurements.
///
/// `concurrency` bounds how many source frames are opened and scored at once;
/// 0 uses 1. The synchronous call has no cancellation callback and retains
/// no input pointers. It fails if the JSON is invalid or no frame can be
/// scored. Free the returned JSON, or an `error_out` string on failure, with
/// [`seiza_string_free`]. Relative paths are resolved by file reads against
/// the process working directory; the returned spelling is unchanged.
///
/// # Safety
/// `paths_json` must point to a valid NUL-terminated UTF-8 string. When
/// non-null, `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_stack_choose_reference_json(
    paths_json: *const c_char,
    concurrency: usize,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let paths_json = required_str(paths_json, "stack reference paths")?;
        let paths: Vec<String> = serde_json::from_str(&paths_json).map_err(|error| {
            format!("stack reference paths must be a JSON array of strings: {error}")
        })?;
        let native_paths = paths.iter().map(PathBuf::from).collect::<Vec<_>>();
        let (reference_index, scores) =
            seiza_stacking::choose_reference(&native_paths, concurrency)
                .map_err(|error| error.to_string())?;
        owned_json(&StackReferenceResponse {
            schema_version: 1,
            reference_index,
            reference_path: paths[reference_index].clone(),
            ranking: seiza_stacking::rank_references(&scores),
            scores: scores
                .into_iter()
                .map(|score| score.map(StackReferenceScoreResponse::from))
                .collect(),
        })
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Opens a FITS or XISF reference and optional integrated bias, dark, and flat
/// masters. A positive `dark_exposure_seconds` overrides the dark metadata; zero
/// uses the metadata. Pass null or empty `options_json` for defaults; its
/// fields are `seiza_stacking::StackOptions`'s, so a host can choose, for
/// example, `"demosaic": "vng" | "mhc" | "bilinear"`,
/// `"interpolation": "bilinear" | "lanczos3"`,
/// `"normalization": {"mode": "local-background", "options": {"tile_size": 256}}`,
/// `"weighting": {"mode": "inverse-noise-variance"}`,
/// `"registration": {"model": "quadratic"}`, or
/// `"match_tolerances": {"rotation_deg": 5.0}`, the gate the reference and
/// every pushed light are judged under against the masters (tolerances left
/// out keep their defaults). All files
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
/// Opens a stack aligned to `reference_image_path`, a FITS or XISF image
/// shared by the stacks of several filters, so that their outputs share one
/// pixel grid and WCS. The reference image sets only the grid: it is
/// demosaiced when it is a raw colour frame, but not calibrated, and it adds
/// no pixels unless it is also pushed as a light.
///
/// `channel_reference_path`, one of this filter's lights, is calibrated,
/// aligned to the reference image, and sets normalization and noise
/// weighting. It is not taken as a light here: the stack starts with no
/// accepted frames, so push it like every other light. Both images must
/// prepare to the same channel count, mono for mono lights and RGB for RGB or
/// Bayer lights. The masters, `dark_exposure_seconds` and `options_json` are
/// those of [`seiza_live_stacker_open_fits`]. The state JSON reports
/// `referenceIntegrated` false and the `referenceImagePath`, and snapshots,
/// drizzles and contexts refuse to replace the reference image.
///
/// # Safety
/// `reference_image_path` and `channel_reference_path` must be valid
/// NUL-terminated strings. Optional paths and `options_json` may be null;
/// when non-null they must be NUL-terminated. When non-null, `error_out` must
/// point to writable storage for one pointer.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn seiza_live_stacker_open_fits_with_reference_image(
    reference_image_path: *const c_char,
    channel_reference_path: *const c_char,
    bias_path: *const c_char,
    dark_path: *const c_char,
    flat_path: *const c_char,
    dark_exposure_seconds: f64,
    options_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut SeizaLiveStacker {
    clear_error(error_out);
    ffi_result(error_out, || {
        let reference_image_path =
            required_path(reference_image_path, "stack reference image path")?;
        let channel_reference_path =
            required_path(channel_reference_path, "stack channel reference path")?;
        let ReferenceImageStack { stacker, .. } = open_with_reference_image(
            &reference_image_path,
            &[channel_reference_path],
            [bias_path, dark_path, flat_path],
            dark_exposure_seconds,
            options_json,
        )?;
        Ok(SeizaLiveStacker { stacker })
    })
    .map_or(ptr::null_mut(), |stacker| Box::into_raw(Box::new(stacker)))
}

#[unsafe(no_mangle)]
/// [`seiza_live_stacker_open_fits_with_reference_image`] with a JSON array
/// of channel reference paths, best first. Each is tried in turn until one
/// aligns to the reference image, which is read and prepared once. For an
/// automatic choice, pass the first few lights of the `ranking` from
/// [`seiza_stack_choose_reference_json`]; the CLI tries five. A light that
/// cannot be read or calibrated, or whose channel count differs from the
/// reference image's, ends the search with an error, since the next light
/// would fail the same way. The masters, `dark_exposure_seconds` and
/// `options_json` are those of [`seiza_live_stacker_open_fits`].
///
/// On success, when `selection_json_out` is non-null, it receives owned
/// schema-1 JSON: `channelReferenceIndex` (zero-based, into the array),
/// `channelReferencePath` (exactly that input string), and `skipped`, an
/// `{index, path, reason}` entry for each earlier light that did not align.
/// Free it with [`seiza_string_free`]. The stack starts with no accepted
/// frames: push every light, the channel reference included.
///
/// # Safety
/// `reference_image_path` and `channel_reference_paths_json` must be valid
/// NUL-terminated strings. Optional paths and `options_json` may be null;
/// when non-null they must be NUL-terminated. When non-null,
/// `selection_json_out` and `error_out` must each point to writable storage
/// for one pointer.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn seiza_live_stacker_open_fits_with_reference_image_candidates(
    reference_image_path: *const c_char,
    channel_reference_paths_json: *const c_char,
    bias_path: *const c_char,
    dark_path: *const c_char,
    flat_path: *const c_char,
    dark_exposure_seconds: f64,
    options_json: *const c_char,
    selection_json_out: *mut *mut c_char,
    error_out: *mut *mut c_char,
) -> *mut SeizaLiveStacker {
    clear_error(error_out);
    if !selection_json_out.is_null() {
        unsafe { *selection_json_out = ptr::null_mut() };
    }
    ffi_result(error_out, || {
        let reference_image_path =
            required_path(reference_image_path, "stack reference image path")?;
        let paths_json = required_str(channel_reference_paths_json, "channel reference paths")?;
        let paths: Vec<String> = serde_json::from_str(&paths_json).map_err(|error| {
            format!("channel reference paths must be a JSON array of strings: {error}")
        })?;
        if paths.is_empty() {
            return Err("at least one channel reference path is required".into());
        }
        let native_paths = paths.iter().map(PathBuf::from).collect::<Vec<_>>();
        let ReferenceImageStack {
            stacker,
            used,
            skipped,
        } = open_with_reference_image(
            &reference_image_path,
            &native_paths,
            [bias_path, dark_path, flat_path],
            dark_exposure_seconds,
            options_json,
        )?;
        if !selection_json_out.is_null() {
            let selection = owned_json(&ChannelReferenceSelectionResponse {
                schema_version: 1,
                channel_reference_index: used,
                channel_reference_path: paths[used].clone(),
                skipped: skipped
                    .into_iter()
                    .map(|(index, reason)| SkippedChannelReferenceResponse {
                        index,
                        path: paths[index].clone(),
                        reason,
                    })
                    .collect(),
            })?;
            unsafe { *selection_json_out = selection };
        }
        Ok(SeizaLiveStacker { stacker })
    })
    .map_or(ptr::null_mut(), |stacker| Box::into_raw(Box::new(stacker)))
}

/// A stack opened on a reference image.
struct ReferenceImageStack {
    stacker: LiveStacker,
    /// The index of the channel reference it uses.
    used: usize,
    /// The index and reason of each earlier channel reference that did not
    /// align.
    skipped: Vec<(usize, String)>,
}

/// Open a stack on a reference image, trying `channel_reference_paths` in
/// turn.
fn open_with_reference_image(
    reference_image_path: &Path,
    channel_reference_paths: &[PathBuf],
    [bias_path, dark_path, flat_path]: [*const c_char; 3],
    dark_exposure_seconds: f64,
    options_json: *const c_char,
) -> Result<ReferenceImageStack, String> {
    let bias_path = optional_path(bias_path)?;
    let dark_path = optional_path(dark_path)?;
    let flat_path = optional_path(flat_path)?;
    let masters = [bias_path.clone(), dark_path.clone(), flat_path.clone()];
    // The reference image may be a light, even a channel reference, but
    // neither may be a master.
    for image in std::iter::once(reference_image_path)
        .chain(channel_reference_paths.iter().map(PathBuf::as_path))
    {
        let paths = std::iter::once(image.to_path_buf())
            .chain(masters.iter().flatten().cloned())
            .collect::<Vec<_>>();
        validate_distinct_stack_paths(&paths)?;
    }
    if dark_path.is_none() && dark_exposure_seconds != 0.0 {
        return Err("a master-dark exposure override requires a dark path".into());
    }
    let dark_exposure_seconds =
        optional_positive_seconds(dark_exposure_seconds, "master-dark exposure override")?;
    let options = stack_options(options_json)?;
    let mut skipped = Vec::new();
    let (stacker, used) = LiveStacker::open_fits_with_reference_image_candidates(
        reference_image_path,
        channel_reference_paths,
        bias_path.as_deref(),
        dark_path.as_deref(),
        flat_path.as_deref(),
        dark_exposure_seconds,
        options,
        |index, reason| skipped.push((index, reason.to_string())),
    )
    .map_err(|error| error.to_string())?;
    Ok(ReferenceImageStack {
        stacker,
        used,
        skipped,
    })
}

#[unsafe(no_mangle)]
/// Plans a canvas that holds a connected set of overlapping lights, such as
/// mosaic panels or a dithered stack whose edges should be kept, and starts
/// an empty stack on it, as the CLI's `--reference expand` does.
///
/// `paths_json` is a JSON array of FITS or XISF light paths; the first
/// anchors the WCS and photometric scale, so put the choice of
/// [`seiza_stack_choose_reference_json`] first to anchor on the clearest
/// light. Panels may connect through other panels, and lights may differ in
/// rotation by any amount. A light that cannot join is rejected, with the
/// reason, when pushed; this call fails only when nothing joins the anchor,
/// a light cannot be read, or the canvas would exceed
/// `max_canvas_megasamples` million samples (pixels times channels; zero
/// uses 256). A run peaks at about 40 bytes per sample. With
/// `"registration": {"model": "quadratic"}` in `options_json`, a light lying
/// mostly within one panel is warped onto it, which follows lens distortion
/// across a meridian flip. For panels, set
/// `"acceptance": {"minimum_overlap_fraction": 0.1}` or near it.
/// Bayer drizzle and drizzled reintegration are not supported on the canvas.
///
/// The masters, `dark_exposure_seconds` and `options_json` are those of
/// [`seiza_live_stacker_open_fits`]; every light must differ from every
/// master. `cancel` is checked before each light is read. The stack starts
/// with no accepted frames: push every light, the anchor included.
/// [`seiza_live_stacker_expanded_lens_json`] and
/// [`seiza_live_stacker_expansion_planning_notes_json`] report what planning
/// found.
///
/// # Safety
/// `paths_json` must be a valid NUL-terminated string. Optional paths and
/// `options_json` may be null; when non-null they must be NUL-terminated.
/// `cancel` must be null or a live [`SeizaCancelSignal`] retained until this
/// call returns. When non-null, `error_out` must point to writable storage
/// for one pointer.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn seiza_live_stacker_open_fits_expanding(
    paths_json: *const c_char,
    bias_path: *const c_char,
    dark_path: *const c_char,
    flat_path: *const c_char,
    dark_exposure_seconds: f64,
    options_json: *const c_char,
    max_canvas_megasamples: usize,
    cancel: *const SeizaCancelSignal,
    error_out: *mut *mut c_char,
) -> *mut SeizaLiveStacker {
    clear_error(error_out);
    ffi_result(error_out, || {
        let paths_json = required_str(paths_json, "expanding stack light paths")?;
        let paths: Vec<PathBuf> = serde_json::from_str::<Vec<String>>(&paths_json)
            .map_err(|error| {
                format!("expanding stack light paths must be a JSON array of strings: {error}")
            })?
            .into_iter()
            .map(PathBuf::from)
            .collect();
        if paths.is_empty() {
            return Err("at least one light is required".into());
        }
        let bias_path = optional_path(bias_path)?;
        let dark_path = optional_path(dark_path)?;
        let flat_path = optional_path(flat_path)?;
        for light in &paths {
            let inputs = std::iter::once(light.clone())
                .chain(
                    [bias_path.clone(), dark_path.clone(), flat_path.clone()]
                        .into_iter()
                        .flatten(),
                )
                .collect::<Vec<_>>();
            validate_distinct_stack_paths(&inputs)?;
        }
        if dark_path.is_none() && dark_exposure_seconds != 0.0 {
            return Err("a master-dark exposure override requires a dark path".into());
        }
        let dark_exposure_seconds =
            optional_positive_seconds(dark_exposure_seconds, "master-dark exposure override")?;
        let megasamples = if max_canvas_megasamples == 0 {
            256
        } else {
            max_canvas_megasamples
        };
        let maximum_samples = megasamples
            .checked_mul(1_000_000)
            .ok_or_else(|| "max_canvas_megasamples is too large".to_string())?;
        let options = stack_options(options_json)?;
        let cancel = unsafe { cancel.as_ref() };
        let stacker = LiveStacker::open_fits_expanding(
            &paths,
            bias_path.as_deref(),
            dark_path.as_deref(),
            flat_path.as_deref(),
            dark_exposure_seconds,
            options,
            maximum_samples,
            |_, _, _, _| {
                if cancel.is_some_and(|signal| signal.cancelled.load(Ordering::Relaxed)) {
                    seiza_stacking::Continue::No
                } else {
                    seiza_stacking::Continue::Yes
                }
            },
        )
        .map_err(|error| error.to_string())?;
        Ok(SeizaLiveStacker { stacker })
    })
    .map_or(ptr::null_mut(), |stacker| Box::into_raw(Box::new(stacker)))
}

/// The lens distortion an expanded canvas was planned through.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ExpandedLensResponse {
    schema_version: u32,
    center_x: f64,
    center_y: f64,
    decenter_pixels: f64,
    k1: f64,
    k2: f64,
    largest_shift_pixels: f64,
    overlap_rms_pixels: Option<f64>,
    undistorted_overlap_rms_pixels: Option<f64>,
    focal_length_pixels: Option<f64>,
}

#[unsafe(no_mangle)]
/// The lens distortion a canvas from
/// [`seiza_live_stacker_open_fits_expanding`] was planned through, as owned
/// JSON, or the JSON text `null` when planning measured none, as for any
/// stack not opened on an expanding canvas.
///
/// The object is schema 1: the distortion's axis `centerX` and `centerY` in
/// sensor pixels, counted from 0, and its `decenterPixels` from the sensor's
/// centre; `k1` and `k2`, the coefficients of `q` and `q²`, where `q` is a
/// position's squared distance from the axis over the frame's half diagonal
/// squared; `largestShiftPixels`, the largest shift it makes within that half
/// diagonal; `overlapRmsPixels` and `undistortedOverlapRmsPixels`, the RMS
/// distance between the stars the panels' first frames pair where they
/// overlap, through the lens and fitted without it (null for a reopened
/// context); and `focalLengthPixels`, by which panels were placed on the
/// anchor's tangent plane, when the anchor's headers gave one (null
/// otherwise). Free the string with [`seiza_string_free`].
///
/// # Safety
/// `stacker` must be a live `SeizaLiveStacker` pointer, with this read
/// externally synchronized against mutable operations. When non-null,
/// `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_live_stacker_expanded_lens_json(
    stacker: *const SeizaLiveStacker,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let stacker = unsafe { required_live_stacker(stacker)? };
        owned_json(
            &stacker
                .stacker
                .expanded_lens()
                .map(|lens| ExpandedLensResponse {
                    schema_version: 1,
                    center_x: lens.center_x,
                    center_y: lens.center_y,
                    decenter_pixels: lens.decenter_pixels,
                    k1: lens.k1,
                    k2: lens.k2,
                    largest_shift_pixels: lens.largest_shift_pixels,
                    overlap_rms_pixels: lens.overlap_rms_pixels,
                    undistorted_overlap_rms_pixels: lens.undistorted_overlap_rms_pixels,
                    focal_length_pixels: lens.focal_length_pixels,
                }),
        )
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// What planning a canvas from [`seiza_live_stacker_open_fits_expanding`]
/// noted, as an owned JSON array of strings: why it skipped fitting one lens
/// to every panel (their camera or optics differ, or their headers do not
/// say), or why it kept its first plan when the lens-corrected one failed.
/// The array is empty for an ordinary stack and for a reopened context,
/// which does not plan again. Free the string with [`seiza_string_free`].
///
/// # Safety
/// As [`seiza_live_stacker_expanded_lens_json`].
pub unsafe extern "C" fn seiza_live_stacker_expansion_planning_notes_json(
    stacker: *const SeizaLiveStacker,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let stacker = unsafe { required_live_stacker(stacker)? };
        owned_json(&stacker.stacker.expansion_planning_notes())
    })
    .unwrap_or(ptr::null_mut())
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
/// loaded appears in neither list. `tolerances` may be null for the stack's
/// own: those it opened with (`match_tolerances` in its options JSON) or
/// was last given by [`seiza_live_stacker_set_match_tolerances`]. Returns a
/// string released with [`seiza_string_free`], or null with `error_out`
/// set.
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
        let masters = stacker.stacker.calibration();
        let tolerances = unsafe { stack_match_tolerances(&stacker.stacker, tolerances) };
        let (kept, dropped) = masters
            .compatible_for_light_with(&light, &tolerances)
            .map_err(|error| error.to_string())?;
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

/// Whether every active master of the stacker accepts a prospective light:
/// the yes-or-no form of [`seiza_live_stacker_compatible_calibration_json`],
/// asking exactly the same questions, so it answers 1 exactly when that
/// drops nothing.
///
/// Returns 1 when the light is acceptable (and always with no masters
/// loaded), 0 when a master refuses it — with `error_out` set to why,
/// naming the reading and, for a measured gap, the tolerance in force, as
/// in `master flat does not match the light frame's optical configuration:
/// rotation light=113.8deg master=3.2deg (110.60 deg apart, tolerance
/// 2.00)` — and -1 with `error_out` set when an argument is invalid.
///
/// `tolerances` may be null for the stack's own, as in
/// [`seiza_live_stacker_compatible_calibration_json`]. A host that widened
/// a gate to choose its masters asks with the same gate here.
///
/// # Safety
///
/// As [`seiza_live_stacker_compatible_calibration_json`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_live_stacker_validate_calibration(
    stacker: *const SeizaLiveStacker,
    signature: *const SeizaFrameSignature,
    tolerances: *const SeizaMatchTolerances,
    error_out: *mut *mut c_char,
) -> i32 {
    clear_error(error_out);
    ffi_result(error_out, || {
        let stacker = unsafe { required_live_stacker(stacker)? };
        let light = unsafe { frame_signature(signature) }?;
        let tolerances = unsafe { stack_match_tolerances(&stacker.stacker, tolerances) };
        Ok(stacker
            .stacker
            .calibration()
            .validate_light_signature_with(&light, &tolerances))
    })
    .map_or(-1, |validated| match validated {
        Ok(()) => 1,
        Err(error) => {
            let message = match error {
                seiza_stacking::Error::Calibration(message) => message,
                other => other.to_string(),
            };
            set_error(error_out, message);
            0
        }
    })
}

/// A caller's tolerances, or with none given the stack's own, which its
/// pushes are judged under.
unsafe fn stack_match_tolerances(
    stacker: &LiveStacker,
    tolerances: *const SeizaMatchTolerances,
) -> seiza_calibration::MatchTolerances {
    if tolerances.is_null() {
        stacker.match_tolerances()
    } else {
        unsafe { match_tolerances(tolerances) }
    }
}

/// Judge the lights pushed from now on under `tolerances`: how close a
/// light's readings must be to the masters' for them to calibrate it. A
/// host that widened a gate to choose its masters — a rotator that scatters
/// more than two degrees, or a flat taken from the nearest other angle on
/// purpose — sets the same gate here, or the light is refused when it is
/// pushed. To judge the reference under it too, open the stack with
/// `match_tolerances` in its options JSON instead.
///
/// Frames already admitted keep the gate they were admitted under, and a
/// reintegration judges each by its own, so a gate can be widened for one
/// light and narrowed again after it. A saved context keeps it, and the
/// configuration fingerprint changes with it. Null, a zeroed struct and any
/// cleared flag mean the defaults, as everywhere a `SeizaMatchTolerances`
/// is taken. Returns false with `error_out` set for a stack created from
/// prepared arrays, whose frames are never calibrated.
///
/// # Safety
///
/// `stacker` must be a live stacker from this library, with this call
/// externally synchronized against every other operation on it.
/// `tolerances` must be null or reference an initialized
/// `SeizaMatchTolerances`. When non-null, `error_out` must point to
/// writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_live_stacker_set_match_tolerances(
    stacker: *mut SeizaLiveStacker,
    tolerances: *const SeizaMatchTolerances,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let stacker = unsafe { required_live_stacker_mut(stacker)? };
        let tolerances = unsafe { match_tolerances(tolerances) };
        stacker
            .stacker
            .set_match_tolerances(tolerances)
            .map_err(|error| error.to_string())
    })
    .is_some()
}

/// Fill `tolerances` with the stack's own, every flag set: the gate its
/// lights are judged under now.
///
/// `SeizaMatchTolerances` has no field for how the light's angle is read
/// (`MatchTolerances::light_rotation`): a stack whose gate reads lights at
/// an angle the host measured, as one opened from a context saved that
/// way, reports its other tolerances here as if it read the recorded
/// angle, and setting these back gives it the recorded reading.
///
/// # Safety
///
/// `stacker` must be a live stacker from this library, with this read
/// externally synchronized against mutable operations. `tolerances` must
/// point at writable storage for one `SeizaMatchTolerances`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_live_stacker_match_tolerances(
    stacker: *const SeizaLiveStacker,
    tolerances: *mut SeizaMatchTolerances,
) -> bool {
    let (Some(stacker), false) = (unsafe { stacker.as_ref() }, tolerances.is_null()) else {
        return false;
    };
    unsafe { *tolerances = SeizaMatchTolerances::from(stacker.stacker.match_tolerances()) };
    true
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
            input_mode: stacker.stacker.input_mode().as_str(),
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
                calibration_state: FrameCalibrationStateResponse::from(
                    reference_metadata.calibration_state,
                ),
            },
            reference_integrated: stacker.stacker.reference_is_integrated(),
            reference_image_path: stacker
                .stacker
                .reference_image_path()
                .map(|path| path.to_string_lossy().into_owned()),
            reintegration_unavailable: stacker.stacker.reintegration_unavailable(),
        })
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Returns the reference frame's per-channel noise as an owned JSON array of
/// numbers, measured the way each pushed frame's `diagnostics.noise` is. It
/// is the unit of frame weights: the reference frame has weight 1 in every
/// channel. The array is empty when the stack weights frames equally. Free
/// it, or an `error_out` string, with [`seiza_string_free`].
///
/// # Safety
/// `stacker` must be a live `SeizaLiveStacker` pointer. When non-null,
/// `error_out` must point to writable storage for one pointer. Externally
/// synchronize this read with every mutable operation on the same stacker.
pub unsafe extern "C" fn seiza_live_stacker_reference_noise_json(
    stacker: *const SeizaLiveStacker,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let stacker = unsafe { required_live_stacker(stacker)? };
        owned_json(&stacker.stacker.reference_noise())
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Returns the reference frame's mapping onto the stack grid, the identity,
/// as owned JSON in the same versioned form as each pushed frame's
/// `diagnostics.mapping`, so a host can keep one record per frame, the
/// reference included. Free it, or an `error_out` string, with
/// [`seiza_string_free`].
///
/// # Safety
/// `stacker` must be a live `SeizaLiveStacker` pointer. When non-null,
/// `error_out` must point to writable storage for one pointer. Externally
/// synchronize this read with every mutable operation on the same stacker.
pub unsafe extern "C" fn seiza_live_stacker_reference_mapping_json(
    stacker: *const SeizaLiveStacker,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let stacker = unsafe { required_live_stacker(stacker)? };
        owned_json(&stacker.stacker.reference_mapping())
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Opens a stack from a FITS or XISF frame the host has already calibrated
/// and debayered, keeping its header cards for the stack's outputs. This
/// lets a host correct a frame between calibration and registration. A raw
/// CFA frame (one with a Bayer layout) is refused. The stack takes later
/// frames only as arrays, through [`seiza_live_stacker_push_linear_json`].
/// Pass null or empty `options_json` for defaults, as for
/// [`seiza_live_stacker_open_fits`].
///
/// # Safety
/// `prepared_path` must be a valid NUL-terminated string. A non-null
/// `options_json` must be NUL-terminated. When non-null, `error_out` must
/// point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_live_stacker_open_prepared_fits(
    prepared_path: *const c_char,
    options_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut SeizaLiveStacker {
    clear_error(error_out);
    ffi_result(error_out, || {
        let prepared_path = required_path(prepared_path, "prepared reference path")?;
        let options = stack_options(options_json)?;
        let frame = FitsFrame::open(&prepared_path).map_err(|error| error.to_string())?;
        let stacker =
            LiveStacker::from_prepared_frame(frame, options).map_err(|error| error.to_string())?;
        Ok(SeizaLiveStacker { stacker })
    })
    .map_or(ptr::null_mut(), |stacker| Box::into_raw(Box::new(stacker)))
}

#[unsafe(no_mangle)]
/// Measures a linear image's per-channel pixel noise, the estimator behind
/// frame weights: the median absolute deviation of second differences along
/// each axis, as a Gaussian standard deviation in the image's own units,
/// keeping the noisier axis. `channels` is 1 for mono or 3 for interleaved
/// RGB.
///
/// Returns an owned JSON array with one number per channel, or `null` when
/// the image is too small or too empty of finite samples to measure. Free
/// it, or an `error_out` string, with [`seiza_string_free`].
///
/// # Safety
/// `data` must point to `length` readable floats, `width * height *
/// channels` of them. When non-null, `error_out` must point to writable
/// storage for one pointer.
pub unsafe extern "C" fn seiza_frame_noise_json(
    data: *const f32,
    length: usize,
    width: usize,
    height: usize,
    channels: usize,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let image =
            unsafe { linear_image_from_ffi(data, length, width, height, channels, "image")? };
        owned_json(&seiza_stacking::frame_noise(&image))
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
            input_paths: stacker.stacker.protected_paths(),
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
        let input_paths = stacker.stacker.protected_paths();
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
/// Keep each frame this stack admits from now on in a scratch file,
/// registered but not normalized, so [`seiza_live_stacker_reintegrate`] and
/// [`seiza_live_stacker_reintegrate_drizzled`] read it back instead of
/// calibrating, demosaicing and resampling the frame's source again. The
/// result is bit-identical either way.
///
/// Call this before pushing frames to a stack that will be reintegrated. The
/// files take about four bytes per output sample per admitted frame (313 MB
/// for each frame of a 26 MP colour sensor), and a sixteenth as much again
/// under local background normalization, in a new directory inside
/// `scratch_directory`, or the system temporary directory when that is null
/// or empty. Freeing or finishing the stacker removes the directory. Frames
/// admitted before the call, such as those of a stack reopened from a saved
/// context, are prepared from their sources when replayed. A stack that
/// integrates Bayer photosites keeps each frame's photosites. A second call
/// keeps the first directory. A stack that cannot be reintegrated keeps
/// nothing and the call succeeds.
///
/// # Safety
/// `stacker` must be a live `SeizaLiveStacker` pointer, externally
/// synchronized with every other operation on it. `scratch_directory` must
/// be null or a valid NUL-terminated path. When non-null, `error_out` must
/// point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_live_stacker_retain_frames_for_reintegration(
    stacker: *mut SeizaLiveStacker,
    scratch_directory: *const c_char,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let scratch_directory = optional_path(scratch_directory)?;
        let stacker = unsafe { required_live_stacker_mut(stacker)? };
        stacker
            .stacker
            .retain_frames_for_reintegration(scratch_directory.as_deref())
            .map_err(|error| error.to_string())
    })
    .is_some()
}

/// Progress callback for [`seiza_live_stacker_reintegrate`]: the pass (0
/// while estimating statistics, 1 while refining them without the samples
/// the estimate rejects, 2 while integrating), the zero-based frame index,
/// the admitted-frame count, and the caller's context pointer. Called on the
/// thread that made the call. Frames kept in scratch files are read in bands
/// of rows, all three passes running on each band in turn: the callback
/// hears pass 0 for each frame read whole first (to fit its background
/// normalization again), then pass 2 for the frame indices in turn, spread
/// over the bands. Frames read whole for each pass are announced per pass
/// and frame before each read.
pub type SeizaStackReintegrateProgressCallback =
    Option<unsafe extern "C" fn(u32, usize, usize, *mut c_void)>;

#[unsafe(no_mangle)]
/// Integrate every admitted frame again with three-pass, leave-one-out
/// rejection and return the result as a new snapshot.
///
/// Online rejection cannot revisit the reference frame or the warm-up frames
/// it admitted before it had statistics, so a satellite or aircraft trail in
/// one of them stays in the live mean. This reads each admitted frame again,
/// from the scratch file kept for it or else its source file, prepared
/// exactly as the live pass did (the same calibration masters, cosmetic
/// filter, debayering, and recorded registration and normalization), and
/// rejects samples more than `low_sigma` below or `high_sigma` above the
/// other frames. A value of zero or less uses the default of 3. Star
/// detection and registration do not run again, and every admitted frame
/// takes part.
///
/// The live stacker is not changed and may keep integrating afterwards. The
/// stack must be replayable: `reintegrationUnavailable` in
/// [`seiza_live_stacker_state_json`] says why not when it is not, and this
/// call fails with the same message. A source file changed since it was
/// stacked also fails the call. Memory use is about 20 bytes per output
/// sample plus the bands of frames being read, at most 256 MiB, independent
/// of the frame count; up to four frames are held whole at once while their
/// normalization is fitted.
///
/// # Safety
/// `stacker` must be a live `SeizaLiveStacker` pointer, externally
/// synchronized with every mutable operation on it for the duration of the
/// call. `cancel` must be null or a live [`SeizaCancelSignal`] retained until
/// this call returns. `context` is passed through untouched to `progress`.
/// When non-null, `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_live_stacker_reintegrate(
    stacker: *const SeizaLiveStacker,
    low_sigma: f32,
    high_sigma: f32,
    cancel: *const SeizaCancelSignal,
    progress: SeizaStackReintegrateProgressCallback,
    context: *mut c_void,
    error_out: *mut *mut c_char,
) -> *mut SeizaStackSnapshot {
    clear_error(error_out);
    // Raw pointers are not UnwindSafe; the callback runs on this thread and
    // the context's lifetime is the caller's promise.
    let context = context as usize;
    ffi_result(error_out, || {
        let stacker = unsafe { required_live_stacker(stacker)? };
        let sigma = |value: f32| {
            if value.is_finite() && value > 0.0 {
                value
            } else {
                3.0
            }
        };
        let options = seiza_stacking::BatchStackOptions {
            rejection: seiza_stacking::MasterRejectionOptions {
                low_sigma: sigma(low_sigma),
                high_sigma: sigma(high_sigma),
            },
            cancel: unsafe { cancel.as_ref() }
                .map(|signal| CancelSignal::from(Arc::clone(&signal.cancelled))),
            ..seiza_stacking::BatchStackOptions::default()
        };
        let result = stacker
            .stacker
            .reintegrate(&options, |pass, index, count| {
                if let Some(callback) = progress {
                    let pass = match pass {
                        seiza_stacking::BatchStackPass::Estimate => 0,
                        seiza_stacking::BatchStackPass::Refine => 1,
                        seiza_stacking::BatchStackPass::Integrate => 2,
                    };
                    unsafe { callback(pass, index, count, context as *mut c_void) };
                }
            })
            .map_err(|error| error.to_string())?;
        Ok(SeizaStackSnapshot {
            snapshot: result.snapshot,
            reference_headers: stacker.stacker.reference_headers().to_vec(),
            input_paths: stacker.stacker.protected_paths(),
        })
    })
    .map_or(ptr::null_mut(), |snapshot| {
        Box::into_raw(Box::new(snapshot))
    })
}

#[unsafe(no_mangle)]
/// [`seiza_live_stacker_reintegrate`], also drizzling every admitted frame
/// onto a grid `scale` (1 to 4) times finer than the reference, as
/// PixInsight's DrizzleIntegration follows ImageIntegration.
///
/// During the final pass each frame is read once more from its source file
/// and calibrated, but not debayered or resampled. Every source pixel,
/// shrunk to a drop `drop_shrink` of a pixel on a side, is carried through
/// the frame's recorded registration and spread over the output pixels it
/// overlaps. A pixel whose registered sample the integration rejected is
/// left out, and each is normalized and weighted as the integration did its
/// sample. Bayer frames drop each photosite into its own colour. A
/// `drop_shrink` of zero or less uses WBPP's defaults: 0.9 for monochrome
/// frames and 1.0 for Bayer frames.
///
/// Returns the drizzled result, and stores the reintegrated stack in
/// `*snapshot_out` when that is non-null; free each with its own `_free`
/// function. On failure both are null. `low_sigma`, `high_sigma`, `cancel`,
/// `progress` and `context` work as in [`seiza_live_stacker_reintegrate`];
/// progress reports pass 2 for the combined integrating and drizzling pass.
/// The drizzle adds about eight bytes per output sample to reintegration's
/// memory.
///
/// # Safety
/// As for [`seiza_live_stacker_reintegrate`]. When non-null, `snapshot_out`
/// must point to writable storage for one pointer.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn seiza_live_stacker_reintegrate_drizzled(
    stacker: *const SeizaLiveStacker,
    low_sigma: f32,
    high_sigma: f32,
    scale: u32,
    drop_shrink: f32,
    cancel: *const SeizaCancelSignal,
    progress: SeizaStackReintegrateProgressCallback,
    context: *mut c_void,
    snapshot_out: *mut *mut SeizaStackSnapshot,
    error_out: *mut *mut c_char,
) -> *mut SeizaDrizzleResult {
    clear_error(error_out);
    if !snapshot_out.is_null() {
        unsafe { *snapshot_out = ptr::null_mut() };
    }
    // Raw pointers are not UnwindSafe; the callback runs on this thread and
    // the context's lifetime is the caller's promise.
    let context = context as usize;
    let snapshot_out = snapshot_out as usize;
    ffi_result(error_out, || {
        let stacker = unsafe { required_live_stacker(stacker)? };
        let sigma = |value: f32| {
            if value.is_finite() && value > 0.0 {
                value
            } else {
                3.0
            }
        };
        let options = seiza_stacking::BatchStackOptions {
            rejection: seiza_stacking::MasterRejectionOptions {
                low_sigma: sigma(low_sigma),
                high_sigma: sigma(high_sigma),
            },
            cancel: unsafe { cancel.as_ref() }
                .map(|signal| CancelSignal::from(Arc::clone(&signal.cancelled))),
            ..seiza_stacking::BatchStackOptions::default()
        };
        let drizzle = seiza_stacking::DrizzleOptions {
            scale,
            drop_shrink: (drop_shrink.is_finite() && drop_shrink > 0.0).then_some(drop_shrink),
        };
        let (result, drizzled) = stacker
            .stacker
            .reintegrate_drizzled(&options, &drizzle, |pass, index, count| {
                if let Some(callback) = progress {
                    let pass = match pass {
                        seiza_stacking::BatchStackPass::Estimate => 0,
                        seiza_stacking::BatchStackPass::Refine => 1,
                        seiza_stacking::BatchStackPass::Integrate => 2,
                    };
                    unsafe { callback(pass, index, count, context as *mut c_void) };
                }
            })
            .map_err(|error| error.to_string())?;
        let reference_headers = stacker.stacker.reference_headers().to_vec();
        let input_paths = stacker.stacker.protected_paths();
        let drizzled = SeizaDrizzleResult {
            result: drizzled,
            accepted_frames: result.snapshot.accepted_frames,
            rejected_frames: result.snapshot.rejected_frames,
            reference_headers: reference_headers.clone(),
            input_paths: input_paths.clone(),
        };
        let snapshot_out = snapshot_out as *mut *mut SeizaStackSnapshot;
        if !snapshot_out.is_null() {
            let snapshot = SeizaStackSnapshot {
                snapshot: result.snapshot,
                reference_headers,
                input_paths,
            };
            unsafe { *snapshot_out = Box::into_raw(Box::new(snapshot)) };
        }
        Ok(drizzled)
    })
    .map_or(ptr::null_mut(), |drizzled| {
        Box::into_raw(Box::new(drizzled))
    })
}

#[unsafe(no_mangle)]
/// # Safety
/// `drizzle` must be null or a live `SeizaDrizzleResult` pointer.
pub unsafe extern "C" fn seiza_drizzle_result_width(drizzle: *const SeizaDrizzleResult) -> usize {
    unsafe { drizzle.as_ref().map_or(0, |value| value.result.image.width) }
}

#[unsafe(no_mangle)]
/// # Safety
/// `drizzle` must be null or a live `SeizaDrizzleResult` pointer.
pub unsafe extern "C" fn seiza_drizzle_result_height(drizzle: *const SeizaDrizzleResult) -> usize {
    unsafe {
        drizzle
            .as_ref()
            .map_or(0, |value| value.result.image.height)
    }
}

#[unsafe(no_mangle)]
/// # Safety
/// `drizzle` must be null or a live `SeizaDrizzleResult` pointer.
pub unsafe extern "C" fn seiza_drizzle_result_channels(
    drizzle: *const SeizaDrizzleResult,
) -> usize {
    unsafe {
        drizzle
            .as_ref()
            .map_or(0, |value| value.result.image.channels)
    }
}

#[unsafe(no_mangle)]
/// Output pixels per reference pixel along each axis.
///
/// # Safety
/// `drizzle` must be null or a live `SeizaDrizzleResult` pointer.
pub unsafe extern "C" fn seiza_drizzle_result_scale(drizzle: *const SeizaDrizzleResult) -> u32 {
    unsafe { drizzle.as_ref().map_or(0, |value| value.result.scale) }
}

#[unsafe(no_mangle)]
/// Returns the sample count of the image and weight buffers.
///
/// # Safety
/// `drizzle` must be null or a live `SeizaDrizzleResult` pointer.
pub unsafe extern "C" fn seiza_drizzle_result_data_length(
    drizzle: *const SeizaDrizzleResult,
) -> usize {
    unsafe {
        drizzle
            .as_ref()
            .map_or(0, |value| value.result.image.sample_count())
    }
}

#[unsafe(no_mangle)]
/// # Safety
/// `drizzle` must be null or a live `SeizaDrizzleResult` pointer.
pub unsafe extern "C" fn seiza_drizzle_result_accepted_frames(
    drizzle: *const SeizaDrizzleResult,
) -> u32 {
    unsafe { drizzle.as_ref().map_or(0, |value| value.accepted_frames) }
}

#[unsafe(no_mangle)]
/// # Safety
/// `drizzle` must be null or a live `SeizaDrizzleResult` pointer.
pub unsafe extern "C" fn seiza_drizzle_result_rejected_frames(
    drizzle: *const SeizaDrizzleResult,
) -> u32 {
    unsafe { drizzle.as_ref().map_or(0, |value| value.rejected_frames) }
}

#[unsafe(no_mangle)]
/// Borrows the interleaved drizzled mean until the result is freed. Output
/// samples no drop reached are `NaN`.
///
/// # Safety
/// `drizzle` must be null or a live `SeizaDrizzleResult` pointer.
pub unsafe extern "C" fn seiza_drizzle_result_image(
    drizzle: *const SeizaDrizzleResult,
) -> *const f32 {
    unsafe {
        drizzle
            .as_ref()
            .map_or(ptr::null(), |value| value.result.image.data.as_ptr())
    }
}

#[unsafe(no_mangle)]
/// Borrows the total weight behind each output sample (drop area in output
/// pixels times frame weight) until the result is freed.
///
/// # Safety
/// `drizzle` must be null or a live `SeizaDrizzleResult` pointer.
pub unsafe extern "C" fn seiza_drizzle_result_weight(
    drizzle: *const SeizaDrizzleResult,
) -> *const f32 {
    unsafe {
        drizzle
            .as_ref()
            .map_or(ptr::null(), |value| value.result.weight.data.as_ptr())
    }
}

#[unsafe(no_mangle)]
/// Writes the drizzled image as an unstretched 32-bit floating-point FITS,
/// or monolithic XISF for a `.xisf` path, with the reference WCS scaled to
/// the drizzle grid.
///
/// # Safety
/// `drizzle` must be a live `SeizaDrizzleResult` pointer. `path` must be a
/// valid NUL-terminated string. When non-null, `error_out` must point to
/// writable storage for one pointer.
pub unsafe extern "C" fn seiza_drizzle_result_write_fits(
    drizzle: *const SeizaDrizzleResult,
    path: *const c_char,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let drizzle = unsafe { drizzle.as_ref() }.ok_or("drizzle result is required")?;
        let path = required_path(path, "drizzle output path")?;
        if drizzle
            .input_paths
            .iter()
            .any(|input| paths_refer_to_same_file(input, &path))
        {
            return Err(
                "drizzle output path must not refer to an input frame or calibration master".into(),
            );
        }
        seiza_stacking::write_drizzle_fits_f32(
            path,
            &drizzle.result,
            drizzle.accepted_frames,
            drizzle.rejected_frames,
            &drizzle.reference_headers,
        )
        .map_err(|error| error.to_string())
    })
    .is_some()
}

#[unsafe(no_mangle)]
/// # Safety
/// `drizzle` must be null or a live pointer returned by
/// [`seiza_live_stacker_reintegrate_drizzled`] and must not already be freed.
pub unsafe extern "C" fn seiza_drizzle_result_free(drizzle: *mut SeizaDrizzleResult) {
    if !drizzle.is_null() {
        unsafe { drop(Box::from_raw(drizzle)) };
    }
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
        let input_paths = stacker.protected_paths();
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
/// G≤20 package, `2` installs every published catalog, and `3` installs only
/// the Gaia photometry catalog that colour calibration reads. The call is
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

/// A blind plate solution of an image file against the catalogs in
/// `catalog_directory`, with the stars found in it, its size and capture
/// date, and the star catalog it was solved against.
struct BlindSolved {
    width: u32,
    height: u32,
    stars: Vec<seiza::DetectedStar>,
    catalog: TileCatalog,
    capture_time: Option<String>,
    solution: seiza::solve::Solution,
    /// Where the solution came from.
    source: SolveSource,
}

/// Solve the image at `path`, a FITS, XISF or raster file, with what the
/// file says about itself (see [`seiza::prior`]): its own WCS checked, else
/// a hinted solve around its pointing and scale, else blind; and, when
/// nothing solves, its own WCS unchecked.
///
/// FITS and XISF images are detected first on an MTF stretch, as the CLI
/// does, which suits processed images, and again on their linear samples
/// when that does not solve; a raster converted from 8-bit colour is
/// detected again in float. A raster's EXIF focal length gives the first
/// scale to try. Only when the second detection also fails does an
/// unchecked header WCS stand.
fn blind_solve_path(
    path: &Path,
    catalog_directory: Option<&Path>,
    minimum_scale_arcsec_per_pixel: f64,
    maximum_scale_arcsec_per_pixel: f64,
    sip_order: u8,
    limits: &SolveLimits,
) -> Result<BlindSolved, String> {
    /// How to detect again when the first detection does not solve.
    enum Retry {
        None,
        Linear(FitsImage),
        Float(DynamicImage),
    }
    let detection_config = DetectConfig {
        max_stars: 600,
        ..Default::default()
    };
    let (width, height, first_stars, retry, capture_time, prior) = if is_astronomy_image_path(path)
    {
        let read = seiza_image_io::read_astronomy(path).map_err(|error| error.to_string())?;
        let prior = read.solve_prior();
        let image = read.image;
        let width = u32::try_from(image.width).map_err(|_| "image width is too large")?;
        let height = u32::try_from(image.height).map_err(|_| "image height is too large")?;
        let capture_time = seiza_image_io::capture_time_from_headers(&image.headers);
        let stretched = image.stretch_to_u8(&StretchParams::default());
        let stars = image::GrayImage::from_raw(width, height, stretched)
            .map(|gray| detect_stars(&DynamicImage::ImageLuma8(gray), &detection_config))
            .unwrap_or_default();
        (
            width,
            height,
            stars,
            Retry::Linear(image),
            capture_time,
            prior,
        )
    } else {
        let image = open_raster(path)?;
        let width = image.width();
        let height = image.height();
        let stars = detect_stars(&image, &detection_config);
        let metadata = seiza_image_io::raster::PhotoMetadata::read(path);
        let prior = seiza::prior::SolvePrior {
            scale_arcsec_px: metadata
                .scale_hint((width, height))
                .map(|hint| hint.nominal_arcsec_per_pixel),
            ..seiza::prior::SolvePrior::default()
        };
        let retry = if is_converted_8bit_color(&image) {
            Retry::Float(image)
        } else {
            Retry::None
        };
        (
            width,
            height,
            stars,
            retry,
            metadata.capture_time_utc,
            prior,
        )
    };
    let star_path =
        seiza::data_paths::star_data(catalog_directory).map_err(|error| error.to_string())?;
    let index_path = seiza::data_paths::blind_index_beside(catalog_directory, &star_path)
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
    let dimensions = (width, height);
    let solve = |stars: &[seiza::DetectedStar], params: &BlindParams| {
        seiza::prior::solve_with_prior(
            stars,
            &catalog,
            Some(&index),
            &prior,
            params,
            dimensions,
            limits,
        )
    };
    let stopped =
        |error: &SeizaError| matches!(error, SeizaError::TimedOut | SeizaError::Cancelled);
    let unchecked =
        |found: &seiza::prior::PriorSolution| found.source == SolveSource::HeaderUnchecked;

    // With a second detection to come, the first spends fewer blind
    // hypotheses, as the CLI's does.
    let first_params = match retry {
        Retry::None => params.clone(),
        _ => BlindParams {
            max_hypotheses: params.max_hypotheses.min(FIRST_DETECTION_HYPOTHESES),
            ..params.clone()
        },
    };
    let first = solve(&first_stars, &first_params);
    let (found, stars) = match (first, retry) {
        (Err(error), _) if stopped(&error) => return Err(error.to_string()),
        (Ok(found), _) if !unchecked(&found) => (found, first_stars),
        (first, Retry::None) => (first.map_err(|error| error.to_string())?, first_stars),
        (first, retry) => {
            let stars = match retry {
                Retry::Linear(image) => {
                    detect_stars_luma_f32(&image.to_luma_f32(), width, height, &detection_config)
                }
                Retry::Float(image) => detect_stars(
                    &image,
                    &DetectConfig {
                        backend: DetectBackend::F32,
                        ..detection_config
                    },
                ),
                Retry::None => unreachable!("handled above"),
            };
            match (solve(&stars, &params), first) {
                (Err(error), _) if stopped(&error) => return Err(error.to_string()),
                (Ok(found), _) if !unchecked(&found) => (found, stars),
                // Neither detection solved: the header's own WCS stands.
                (Ok(found), _) | (Err(_), Ok(found)) => (found, stars),
                (Err(error), Err(_)) => return Err(error.to_string()),
            }
        }
    };
    Ok(BlindSolved {
        width,
        height,
        stars,
        catalog,
        capture_time,
        solution: found.solution,
        source: found.source,
    })
}

/// Blind hypotheses the first detection of an image spends when a second
/// detection follows, as the CLI's `--detection-fallback-hypotheses`.
const FIRST_DETECTION_HYPOTHESES: usize = 64;

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
    unsafe {
        seiza_solve_image_with_limits_json(
            path,
            catalog_directory,
            minimum_scale_arcsec_per_pixel,
            maximum_scale_arcsec_per_pixel,
            sip_order,
            0.0,
            ptr::null(),
            error_out,
        )
    }
}

#[unsafe(no_mangle)]
/// [`seiza_solve_image_json`] that gives up after `time_limit_seconds`
/// (zero for no limit) or when `cancel` is cancelled. A solve that stops
/// returns null with `error_out` set to `solve stopped at its time limit` or
/// `cancelled by the caller`. Detection and loading are not counted against
/// the limit; the solve checks it between windows and hypotheses.
///
/// # Safety
/// As [`seiza_solve_image_json`]. `cancel` must be null or a live
/// [`SeizaCancelSignal`] retained until this call returns.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn seiza_solve_image_with_limits_json(
    path: *const c_char,
    catalog_directory: *const c_char,
    minimum_scale_arcsec_per_pixel: f64,
    maximum_scale_arcsec_per_pixel: f64,
    sip_order: u8,
    time_limit_seconds: f64,
    cancel: *const SeizaCancelSignal,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let time_limit = optional_positive_seconds(time_limit_seconds, "solve time limit")?;
        let limits = time_limit
            .map(|seconds| SolveLimits::within(std::time::Duration::from_secs_f64(seconds)))
            .unwrap_or_default();
        let limits = match unsafe { cancel.as_ref() } {
            Some(signal) => limits.with_cancel(CancelSignal::from(Arc::clone(&signal.cancelled))),
            None => limits,
        };
        let started = Instant::now();
        let path = required_path(path, "image path")?;
        let catalog_directory = optional_path(catalog_directory)?;
        let BlindSolved {
            width,
            height,
            stars,
            catalog,
            capture_time,
            solution,
            source,
        } = blind_solve_path(
            &path,
            catalog_directory.as_deref(),
            minimum_scale_arcsec_per_pixel,
            maximum_scale_arcsec_per_pixel,
            sip_order,
            &limits,
        )?;
        let middle = seiza::image_center(width, height);
        let center = solution.wcs.pixel_to_world(middle.0, middle.1);
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
        let FieldObjects {
            objects,
            object_catalog_error,
            transient_catalog_error,
            minor_body_error,
        } = seiza::field_objects::field_objects(
            catalog_directory.as_deref(),
            &solution.wcs,
            (width, height),
            capture_time.as_deref(),
        );
        let mut overlay_availability =
            BTreeMap::from([("field_stars".into(), true), ("grid".into(), true)]);
        let mut overlay_unavailable_reasons = BTreeMap::new();
        for (layers, error) in [
            (&["deep_sky", "named_stars"][..], &object_catalog_error),
            (
                &["transients", "historical_transients"][..],
                &transient_catalog_error,
            ),
            (&["minor_bodies"][..], &minor_body_error),
        ] {
            for &layer in layers {
                overlay_availability.insert(layer.into(), error.is_none());
                if let Some(error) = error {
                    overlay_unavailable_reasons.insert(layer.into(), error.clone());
                }
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
        for object in &objects {
            *overlay_counts.entry(object.layer().into()).or_insert(0) += 1;
            if object.kind == "transient" && object.near_capture == Some(false) {
                *overlay_counts
                    .entry("historical_transients".into())
                    .or_insert(0) += 1;
            }
        }
        let object_positions = objects.into_iter().map(ObjectPointResponse::from).collect();
        let response = SolveResponse {
            center_ra_degrees: center.0,
            center_dec_degrees: center.1,
            scale_arcsec_per_pixel: solution.wcs.scale_arcsec_per_px(),
            matched_stars: solution.matched_stars,
            rms_arcsec: solution.rms_arcsec,
            false_match_probability: solution.false_match_probability,
            solved_from: source.as_str(),
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
            wcs: WcsResponse::new(&solution.wcs),
        };
        let json = serde_json::to_string(&response).map_err(|error| error.to_string())?;
        CString::new(json).map_err(|_| "solution JSON contains a null byte".to_string())
    })
    .map_or(ptr::null_mut(), CString::into_raw)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CaptureTimeResponse {
    capture_time: Option<String>,
}

#[unsafe(no_mangle)]
/// When the image at `path` was taken, as `{"captureTime": "..."}`: from a
/// FITS or XISF header's `DATE-OBS`, `DATE-BEG` or `DATE-AVG`, read without
/// decoding the pixels, or from a photo's EXIF, in UTC. `captureTime` is
/// null when the file does not say. It is the time
/// [`seiza_solve_image_json`] dates transients and places comets and
/// asteroids by. Free the result, or an `error_out` string, with
/// [`seiza_string_free`].
///
/// # Safety
/// `path` must be a valid NUL-terminated string. When non-null, `error_out`
/// must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_read_capture_time_json(
    path: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let path = required_path(path, "image path")?;
        let capture_time =
            seiza_image_io::read_capture_time(&path).map_err(|error| error.to_string())?;
        owned_json(&CaptureTimeResponse { capture_time })
    })
    .unwrap_or(ptr::null_mut())
}

/// FITS header cards in the order they are written, as a JSON object of
/// keyword to value.
struct FitsCardsJson(Vec<(String, seiza::FitsCardValue)>);

impl Serialize for FitsCardsJson {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.0.iter().map(|(keyword, value)| {
            let value = match value {
                seiza::FitsCardValue::Text(text) => json!(text),
                seiza::FitsCardValue::Integer(number) => json!(number),
                seiza::FitsCardValue::Number(number) => json!(number),
            };
            (keyword, value)
        }))
    }
}

#[derive(Serialize)]
struct WcsFitsHeaderResponse {
    cards: FitsCardsJson,
    text: String,
}

#[unsafe(no_mangle)]
/// A WCS as FITS header cards. `wcs_json` is a WCS as
/// [`seiza_solve_image_json`] returns it: `crval`, `crpix`, `cd`, an
/// optional `sip` and an optional `pixelOrigin` (0, the default, when
/// `crpix` counts pixels from 0; 1 when it counts from 1, as a FITS header
/// does).
///
/// Returns `{"cards": {...}, "text": "..."}`: `cards` holds each keyword
/// and its value in the order a header writes them (`CTYPE1` and `CTYPE2`,
/// TAN or TAN-SIP; `CUNIT`, `EQUINOX`, `CRVAL`, 1-based `CRPIX`, the `CD`
/// matrix, and every SIP keyword when the WCS has SIP terms), and `text`
/// the same cards as a minimal FITS header: 80-column cards ending with
/// `END`. `cards` can be passed as the WCS keywords of
/// [`seiza_color_calibrate_json`]. Free the result, or an `error_out`
/// string, with [`seiza_string_free`].
///
/// # Safety
/// `wcs_json` must be a valid NUL-terminated UTF-8 string. When non-null,
/// `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_wcs_fits_header_json(
    wcs_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let request: parallax::WcsRequest = serde_json::from_str(&required_str(wcs_json, "WCS")?)
            .map_err(|error| format!("invalid WCS: {error}"))?;
        let wcs = request.wcs()?;
        owned_json(&WcsFitsHeaderResponse {
            cards: FitsCardsJson(wcs.fits_header_cards()),
            text: wcs.fits_header_text(),
        })
    })
    .unwrap_or(ptr::null_mut())
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ScaleHintResponse {
    nominal_arcsec_per_pixel: f64,
    min_arcsec_per_pixel: f64,
    max_arcsec_per_pixel: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PhotoMetadataResponse {
    make: Option<String>,
    model: Option<String>,
    date_time_original: Option<String>,
    sub_sec_time_original: Option<String>,
    offset_time_original: Option<String>,
    capture_time_utc: Option<String>,
    capture_time_source: Option<seiza_image_io::raster::CaptureTimeSource>,
    exposure_seconds: Option<f64>,
    gps_latitude_deg: Option<f64>,
    gps_longitude_deg: Option<f64>,
    gps_altitude_m: Option<f64>,
    gps_altitude_ref: Option<u32>,
    gps_image_direction_deg: Option<f64>,
    gps_image_direction_ref: Option<String>,
    focal_length_mm: Option<f64>,
    focal_length_35mm: Option<f64>,
    exif_orientation: Option<u32>,
    warnings: Vec<String>,
    scale_hint: Option<ScaleHintResponse>,
    scale_ranges: Option<Vec<[f64; 2]>>,
}

#[unsafe(no_mangle)]
/// The EXIF fields of a JPEG, TIFF, HEIF, PNG or WebP file that help a
/// plate solve: `captureTimeUtc` (from the original time and its offset,
/// or from the GPS stamps, as `captureTimeSource` says:
/// `date_time_original` or `gps`), the GPS position, and the focal length
/// with its 35 mm equivalent, with the camera, exposure and orientation.
/// Missing or damaged EXIF gives null fields and `warnings`, never an error.
///
/// Given the decoded image's `width` and `height` (after its EXIF
/// orientation; 0 for either leaves them out), `scaleHint` is the pixel
/// scale the 35 mm-equivalent focal length implies (null without one), and
/// `scaleRanges` the `[min, max]` pixel-scale ranges, in arcseconds per
/// pixel, to try in order with a blind solve. Without them both are null.
/// Free the result, or an `error_out` string, with [`seiza_string_free`].
///
/// # Safety
/// `path` must be a valid NUL-terminated string. When non-null, `error_out`
/// must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_read_photo_metadata_json(
    path: *const c_char,
    width: u32,
    height: u32,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        use seiza_image_io::raster::{PhotoMetadata, ScaleSearch};
        let path = required_path(path, "photo path")?;
        let metadata = PhotoMetadata::read(&path);
        let (scale_hint, scale_ranges) = if width > 0 && height > 0 {
            let search = ScaleSearch::new(&metadata, (width, height), None, None)
                .map_err(|error| error.to_string())?;
            let hint = metadata
                .scale_hint((width, height))
                .map(|hint| ScaleHintResponse {
                    nominal_arcsec_per_pixel: hint.nominal_arcsec_per_pixel,
                    min_arcsec_per_pixel: hint.min_arcsec_per_pixel,
                    max_arcsec_per_pixel: hint.max_arcsec_per_pixel,
                });
            let ranges = search.ranges.iter().map(|&(min, max)| [min, max]).collect();
            (hint, Some(ranges))
        } else {
            (None, None)
        };
        owned_json(&PhotoMetadataResponse {
            make: metadata.make,
            model: metadata.model,
            date_time_original: metadata.date_time_original,
            sub_sec_time_original: metadata.sub_sec_time_original,
            offset_time_original: metadata.offset_time_original,
            capture_time_utc: metadata.capture_time_utc,
            capture_time_source: metadata.capture_time_source,
            exposure_seconds: metadata.exposure_seconds,
            gps_latitude_deg: metadata.gps_latitude_deg,
            gps_longitude_deg: metadata.gps_longitude_deg,
            gps_altitude_m: metadata.gps_altitude_m,
            gps_altitude_ref: metadata.gps_altitude_ref,
            gps_image_direction_deg: metadata.gps_image_direction_deg,
            gps_image_direction_ref: metadata.gps_image_direction_ref,
            focal_length_mm: metadata.focal_length_mm,
            focal_length_35mm: metadata.focal_length_35mm,
            exif_orientation: metadata.exif_orientation,
            warnings: metadata.warnings,
            scale_hint,
            scale_ranges,
        })
    })
    .unwrap_or(ptr::null_mut())
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

/// FITS or XISF, as the shared image reader names it; its `name()` says
/// which in rendered metadata.
use seiza_image_io::Format as AstronomyImageFormat;

fn astronomy_image_format(path: &Path) -> Option<AstronomyImageFormat> {
    AstronomyImageFormat::from_path(path).filter(|format| format.is_astronomy())
}

fn is_astronomy_image_path(path: &Path) -> bool {
    astronomy_image_format(path).is_some()
}

fn open_astronomy_image(path: &Path) -> Result<(FitsImage, AstronomyImageFormat), String> {
    if !is_astronomy_image_path(path) {
        return Err(format!("{} is not a FITS or XISF path", path.display()));
    }
    seiza_image_io::open_astronomy(path).map_err(|error| error.to_string())
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
        let image = open_raster(path)?;
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
        let image = open_raster(path)?;
        render_raster16(image, raster_format(path), max_dimension)
    }
}

/// Opens a JPEG, PNG or TIFF upright, applying its EXIF orientation, so
/// solving, rendering and overlays share the frame the CLI uses.
fn open_raster(path: &Path) -> Result<image::DynamicImage, String> {
    seiza_image_io::open_raster(path)
        .map(|raster| raster.pixels)
        .map_err(|error| error.to_string())
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
            input_mode: stacker.input_mode().as_str(),
        }),
    })
}

/// Reduce a borrowed live view with area averaging and bounded output storage.
/// Ignore uncovered samples so undefined means never enter the preview.
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

    let xs = resample::footprints(view.width, output_width);
    let ys = resample::footprints(view.height, output_height);
    for (output_y, y) in ys.iter().enumerate() {
        for (output_x, x) in xs.iter().enumerate() {
            for channel in 0..view.channels {
                let mut sum = 0.0_f64;
                let mut weight = 0.0_f64;
                for (dy, wy) in y.weights.iter().enumerate() {
                    for (dx, wx) in x.weights.iter().enumerate() {
                        let index =
                            ((y.start + dy) * view.width + x.start + dx) * view.channels + channel;
                        if view.coverage[index] > 0 && view.mean[index].is_finite() {
                            sum += f64::from(view.mean[index]) * wx * wy;
                            weight += wx * wy;
                        }
                    }
                }
                if weight > 0.0 {
                    output[(output_y * output_width + output_x) * view.channels + channel] =
                        (sum / weight) as f32;
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
    let shape = ImageShape::new(
        prepared.render_width,
        prepared.render_height,
        prepared.channels,
    );
    let stretched = stack
        .apply_u8_image(&data, shape)
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
    let shape = ImageShape::new(
        prepared.render_width,
        prepared.render_height,
        prepared.channels,
    );
    let stretched = stack
        .apply_u16_image(&data, shape)
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
    let star_data = seiza::data_paths::star_data(catalog_directory);
    // The index that ships with the catalog sits beside it.
    let blind_index = optional_component_status(match &star_data {
        Ok(star_path) => seiza::data_paths::blind_index_beside(catalog_directory, star_path),
        Err(_) => seiza::data_paths::blind_index(catalog_directory),
    });
    let star_catalog = component_status(star_data);
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
        mapping: diagnostics.mapping,
        noise: diagnostics.noise,
        weight: diagnostics.weight,
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
    let probe = seiza_stacking::probe_frame(path).map_err(stacking_message)?;
    let metadata = probe.metadata;
    Ok(FrameProbeResponse {
        schema_version: 1,
        path: probe.path.to_string_lossy().into_owned(),
        format: probe.format.name(),
        role: CalibrationFrameRole::from(metadata.role),
        raw_image_type: probe.raw_image_type,
        is_master: metadata.is_master,
        signature: FrameProbeSignature::from(&metadata.signature),
        calibration_state: FrameCalibrationStateResponse::from(metadata.calibration_state),
    })
}

/// A stacking error's own words, without the "calibration error: " that
/// its display adds.
fn stacking_message(error: seiza_stacking::Error) -> String {
    match error {
        seiza_stacking::Error::Calibration(message) => message,
        other => other.to_string(),
    }
}

fn build_calibration_plan(
    request: CalibrationPlanRequest,
) -> Result<CalibrationPlanResponse, String> {
    let plan = seiza_stacking::plan_calibration(seiza_stacking::CalibrationPlanRequest {
        kind: request.kind.into_core(),
        reference: request.reference.into_core(),
        references: request
            .references
            .into_iter()
            .map(CalibrationPlanRecord::into_core)
            .collect(),
        candidates: request
            .candidates
            .into_iter()
            .map(CalibrationPlanRecord::into_core)
            .collect(),
        minimum: request.minimum,
        tolerances: request.tolerances.into_core(),
        bias_available: request.dependencies.bias_available,
    })
    .map_err(stacking_message)?;
    let path_text = |path: PathBuf| path.to_string_lossy().into_owned();
    Ok(CalibrationPlanResponse {
        schema_version: 1,
        kind: plan.kind.as_str(),
        minimum: plan.minimum,
        ready: plan.ready,
        matched_paths: plan.matched_paths.into_iter().map(path_text).collect(),
        selected_paths: plan.selected_paths.into_iter().map(path_text).collect(),
        excluded: plan
            .excluded
            .into_iter()
            .map(|exclusion| CalibrationPlanExclusionResponse {
                path: path_text(exclusion.path),
                reason: exclusion.reason.as_str(),
            })
            .collect(),
    })
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
        progress: None,
        defect_suppression,
        flat_star_masking: request.flat_star_masking.map(FlatStarMaskingOptions::from),
        dark_level_screening: request
            .dark_level_screening
            .map(|screening| DarkLevelScreening {
                max_excess_sigma: screening.max_excess_sigma,
            }),
    };
    let master = match &request.scratch_directory {
        Some(directory) => {
            build_master_from_fits_with_scratch(&request.inputs, kind, &options, directory)
        }
        None => build_master_from_fits(&request.inputs, kind, &options),
    }
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
        masked_samples: master.masked_samples,
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
        rejection_method: master.rejection_method.as_str(),
        flat_star_masking: master
            .flat_star_masking
            .as_ref()
            .map(FlatStarMaskingResponse::from),
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
    let inputs = master
        .accepted_inputs(requested_inputs)
        .map_err(stacking_message)?
        .into_iter()
        .map(|(path, tally)| MasterBuildInputResponse {
            path: path.to_string_lossy().into_owned(),
            accepted_samples: tally.accepted_samples,
            rejected_samples: tally.rejected_samples,
            masked_samples: tally.masked_samples,
        })
        .collect();
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

impl StarDetectOptions {
    /// The core's measurement options: names checked, the rest carried
    /// over as given.
    fn measurement(&self) -> Result<seiza_stars::MeasurementOptions, String> {
        use seiza_stars::hocus_focus_star_detection::StructureRemovalMethod;
        let preset = self
            .preset
            .as_deref()
            .map(|name| {
                seiza_stars::TelescopeClass::from_name(name).ok_or_else(|| {
                    format!("unknown preset {name:?} (widefield, standard, longfocal)")
                })
            })
            .transpose()?;
        let psf_type = match self.psf_type.as_deref() {
            None | Some("moffat4") => seiza_stars::PSFType::Moffat4,
            Some("gaussian") => seiza_stars::PSFType::Gaussian,
            Some("none") => seiza_stars::PSFType::None,
            Some(other) => {
                return Err(format!(
                    "unknown psfType {other:?} (none, gaussian, moffat4)"
                ));
            }
        };
        let structure_removal = match self.structure_removal.as_deref() {
            None => None,
            Some("filtered") => Some(StructureRemovalMethod::Filtered),
            Some("atrous") => Some(StructureRemovalMethod::Atrous),
            Some(other) => {
                return Err(format!(
                    "unknown structureRemoval {other:?} (filtered, atrous)"
                ));
            }
        };
        Ok(seiza_stars::MeasurementOptions {
            preset,
            focal_length_mm: self.focal_length_mm,
            pixel_size_um: self.pixel_size_um,
            psf_type,
            structure_removal,
            detection_binning: self.detection_binning,
            keep_saturated: self.keep_saturated,
            noise_reduction_radius: self.noise_reduction_radius,
            sensitivity: self.sensitivity,
            target_star_count: self.target_star_count,
        })
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

fn detect_stars_response(
    samples: &[u16],
    width: usize,
    height: usize,
    options: StarDetectOptions,
    frame_optics: (Option<f64>, Option<f64>),
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
    let result = options
        .measurement()?
        .with_frame_optics(frame_optics.0, frame_optics.1)
        .detect(samples, width, height);
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

    #[test]
    fn stack_reference_cabi_preserves_order_scores_and_selected_path() {
        let directory = tempfile::tempdir().unwrap();
        let image = LinearImage::new(320, 256, 1, stacking_star_field(320, 256)).unwrap();
        let blank = LinearImage::new(320, 256, 1, vec![100.0; 320 * 256]).unwrap();
        let first = directory.path().join("clear-星.fits");
        let second = directory.path().join("other.xisf");
        let unscorable = directory.path().join("blank.fits");
        seiza_stacking::write_processed_image_fits_f32(&first, &image, &[], &[]).unwrap();
        seiza_stacking::write_processed_image_fits_f32(&second, &image, &[], &[]).unwrap();
        seiza_stacking::write_processed_image_fits_f32(&unscorable, &blank, &[], &[]).unwrap();
        let paths = vec![
            directory.path().join("missing.fits"),
            first,
            unscorable,
            second,
        ];
        let (expected_index, expected_scores) =
            seiza_stacking::choose_reference(&paths, 2).unwrap();
        assert!(matches!(expected_index, 1 | 3));
        let expected_ranking = seiza_stacking::rank_references(&expected_scores);
        assert_eq!(expected_ranking.len(), 2);
        assert_eq!(expected_ranking[0], expected_index);
        let strings = paths
            .iter()
            .map(|path| path.to_str().unwrap())
            .collect::<Vec<_>>();
        let paths_json = CString::new(serde_json::to_string(&strings).unwrap()).unwrap();

        // Reusing error_out clears it but never frees a previously owned error.
        let old_error = CString::new("previous error").unwrap().into_raw();
        let mut error = old_error;
        let raw_response =
            unsafe { seiza_stack_choose_reference_json(paths_json.as_ptr(), 2, &mut error) };
        assert!(error.is_null());
        assert!(!raw_response.is_null());
        let response: Value =
            serde_json::from_str(unsafe { CStr::from_ptr(raw_response) }.to_str().unwrap())
                .unwrap();
        assert_eq!(response["schemaVersion"], 1);
        assert_eq!(response["referenceIndex"], expected_index);
        assert_eq!(response["referencePath"], strings[expected_index]);
        assert_eq!(response["ranking"], serde_json::json!(expected_ranking));
        let scores = response["scores"].as_array().unwrap();
        assert_eq!(scores.len(), paths.len());
        for (actual, expected) in scores.iter().zip(expected_scores) {
            match expected {
                None => assert!(actual.is_null()),
                Some(score) => {
                    assert_eq!(actual["stars"], score.stars);
                    for (key, expected) in [
                        ("medianStarArea", score.median_star_area),
                        ("background", score.background),
                        ("backgroundVariation", score.background_variation),
                        ("score", score.score),
                    ] {
                        assert_eq!(actual[key].as_f64().unwrap() as f32, expected, "{key}");
                    }
                }
            }
        }
        assert!(scores[0].is_null());
        assert!(scores[2].is_null());
        unsafe {
            seiza_string_free(raw_response);
            seiza_string_free(old_error);
        }
        let sequential =
            unsafe { seiza_stack_choose_reference_json(paths_json.as_ptr(), 0, ptr::null_mut()) };
        assert!(!sequential.is_null());
        let sequential_json: Value =
            serde_json::from_str(unsafe { CStr::from_ptr(sequential) }.to_str().unwrap()).unwrap();
        assert_eq!(sequential_json, response);
        unsafe { seiza_string_free(sequential) };
    }

    #[test]
    fn stack_reference_cabi_rejects_invalid_or_unscoreable_inputs() {
        for (request, expected) in [
            (None, "stack reference paths is required"),
            (Some("not JSON"), "JSON array of strings"),
            (Some("{}"), "JSON array of strings"),
            (Some("[42]"), "JSON array of strings"),
            (Some("[]"), "no frame could be scored"),
            (
                Some(r#"["missing-stack-reference.fits"]"#),
                "no frame could be scored",
            ),
        ] {
            let request = request.map(|request| CString::new(request).unwrap());
            let mut error = ptr::null_mut();
            let response = unsafe {
                seiza_stack_choose_reference_json(
                    request.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
                    0,
                    &mut error,
                )
            };
            assert!(response.is_null());
            assert!(!error.is_null());
            let message = unsafe { CStr::from_ptr(error) }.to_str().unwrap();
            assert!(message.contains(expected), "{message}");
            unsafe { seiza_string_free(error) };
        }
        let invalid_utf8 = [0xff_u8, 0];
        let mut error = ptr::null_mut();
        let response = unsafe {
            seiza_stack_choose_reference_json(invalid_utf8.as_ptr().cast(), 0, &mut error)
        };
        assert!(response.is_null());
        assert!(
            unsafe { CStr::from_ptr(error) }
                .to_str()
                .unwrap()
                .contains("not valid UTF-8")
        );
        unsafe { seiza_string_free(error) };
        // Reporting an error is optional, as it is for other owned-JSON APIs.
        assert!(
            unsafe { seiza_stack_choose_reference_json(ptr::null(), 0, ptr::null_mut()) }.is_null()
        );
    }

    #[test]
    fn stack_options_json_chooses_the_demosaic() {
        let options = std::ffi::CString::new(r#"{"demosaic": "mhc"}"#).unwrap();
        assert_eq!(
            stack_options(options.as_ptr()).unwrap().demosaic,
            seiza_stacking::Demosaic::Mhc
        );
        assert_eq!(
            stack_options(ptr::null()).unwrap().demosaic,
            seiza_stacking::Demosaic::Vng
        );
        // Every spelling the open call's documentation gives.
        let documented = std::ffi::CString::new(
            r#"{"demosaic": "bilinear", "interpolation": "lanczos3",
                "normalization": {"mode": "local-background", "options": {"tile_size": 256}},
                "weighting": {"mode": "inverse-noise-variance"},
                "registration": {"model": "quadratic"}}"#,
        )
        .unwrap();
        let options = stack_options(documented.as_ptr()).unwrap();
        assert_eq!(options.demosaic, seiza_stacking::Demosaic::Bilinear);
        assert_eq!(
            options.interpolation,
            seiza_stacking::Interpolation::Lanczos3
        );
        assert!(!options.weighting.is_equal());
        assert_eq!(
            options.registration.model,
            seiza_stacking::RegistrationModel::Quadratic
        );
        let unknown = std::ffi::CString::new(r#"{"demosaic": "ahd"}"#).unwrap();
        assert!(stack_options(unknown.as_ptr()).is_err());
    }
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
        // Past half a turn is a mistake about units, by the rule
        // `MatchTolerances::validate` holds Rust callers to.
        for nonsense in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 360.0] {
            tuned.rotation_deg = nonsense;
            assert_eq!(
                unsafe { match_tolerances(&tuned) }.rotation_deg,
                defaults.rotation_deg,
                "{nonsense} should fall back to the default"
            );
        }
        // Half a turn itself is "any angle", and is taken.
        tuned.rotation_deg = 180.0;
        assert_eq!(unsafe { match_tolerances(&tuned) }.rotation_deg, 180.0);
        // One bad field costs only itself.
        let mixed = SeizaMatchTolerances {
            known: SEIZA_TOLERANCE_HAS_ROTATION | SEIZA_TOLERANCE_HAS_FOCAL_LENGTH,
            rotation_deg: f64::NAN,
            focal_length_mm: 15.0,
            ..Default::default()
        };
        let resolved = unsafe { match_tolerances(&mixed) };
        assert_eq!(resolved.rotation_deg, defaults.rotation_deg);
        assert_eq!(resolved.focal_length_mm, 15.0);

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

    fn wcs_headers() -> Vec<(String, HeaderValue)> {
        [
            ("CTYPE1", HeaderValue::String("RA---TAN".into())),
            ("CTYPE2", HeaderValue::String("DEC--TAN".into())),
            ("CRPIX1", HeaderValue::Float(16.5)),
            ("CRPIX2", HeaderValue::Float(12.5)),
            ("CRVAL1", HeaderValue::Float(83.8)),
            ("CRVAL2", HeaderValue::Float(-5.4)),
            ("CD1_1", HeaderValue::Float(-0.0003)),
            ("CD1_2", HeaderValue::Float(0.0)),
            ("CD2_1", HeaderValue::Float(0.0)),
            ("CD2_2", HeaderValue::Float(0.0003)),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect()
    }

    /// Write a 32 by 24 mono stack of `value`, blank above row `blank_rows`.
    fn color_channel(
        directory: &Path,
        name: &str,
        value: f32,
        blank_rows: usize,
        headers: &[(String, HeaderValue)],
    ) -> String {
        let (width, height) = (32, 24);
        let data = (0..width * height)
            .map(|index| {
                if index / width < blank_rows {
                    f32::NAN
                } else {
                    value
                }
            })
            .collect();
        let image = LinearImage::new(width, height, 1, data).unwrap();
        let path = directory.join(name);
        seiza_stacking::write_processed_image_fits_f32(&path, &image, headers, &[]).unwrap();
        path.to_str().unwrap().to_owned()
    }

    fn compose(
        call: unsafe extern "C" fn(
            *const c_char,
            *const SeizaCancelSignal,
            *mut *mut c_char,
        ) -> *mut c_char,
        request: &Value,
        cancel: *const SeizaCancelSignal,
    ) -> Result<Value, String> {
        let request = CString::new(request.to_string()).unwrap();
        let mut error = ptr::null_mut();
        let json = unsafe { call(request.as_ptr(), cancel, &mut error) };
        if json.is_null() {
            let message = unsafe { CStr::from_ptr(error) }
                .to_str()
                .unwrap()
                .to_owned();
            unsafe { seiza_string_free(error) };
            return Err(message);
        }
        assert!(error.is_null());
        let value =
            serde_json::from_str(unsafe { CStr::from_ptr(json) }.to_str().unwrap()).unwrap();
        unsafe { seiza_string_free(json) };
        Ok(value)
    }

    fn rgb_pixels(path: &str) -> Vec<f32> {
        let image = FitsFrame::open(path).unwrap().image;
        assert_eq!(image.channels, 3);
        image.data
    }

    #[test]
    fn color_cabi_composes_writes_and_reports_a_crop() {
        let directory = tempfile::tempdir().unwrap();
        let headers = wcs_headers();
        let red = color_channel(directory.path(), "red.fits", 0.2, 0, &headers);
        let green = color_channel(directory.path(), "green.xisf", 0.4, 2, &[]);
        let blue = color_channel(directory.path(), "blue.fits", 0.6, 4, &[]);
        let output = directory.path().join("rgb.fits");
        let output = output.to_str().unwrap();

        let report = compose(
            seiza_color_combine_rgb_json,
            &json!({
                "red": red, "green": green, "blue": blue, "output": output,
                "normalization": "none", "crop": "inscribed",
            }),
            ptr::null(),
        )
        .unwrap();
        assert_eq!(report["schemaVersion"], 1);
        assert_eq!(report["label"], "RGB");
        assert_eq!(report["transfer"], "linear");
        assert_eq!(
            report["region"],
            json!({"x": 0, "y": 4, "width": 32, "height": 20})
        );
        assert_eq!(report["crop"]["mode"], "inscribed");
        assert_eq!(report["crop"]["channels"].as_array().unwrap().len(), 3);

        let written = FitsImage::open(Path::new(output)).unwrap();
        assert_eq!(written.header_str("SEIZACLR"), Some("RGB"));
        assert_eq!(written.header_str("SEIZATRF"), Some("LINEAR"));
        // The crop moves the red stack's reference pixel by its origin.
        assert_eq!(written.header_f64("CRPIX2"), Some(12.5 - 4.0));
        let pixels = rgb_pixels(output);
        assert_eq!(pixels.len(), 32 * 20 * 3);
        assert_eq!(&pixels[..3], &[0.2, 0.4, 0.6]);

        // Super-luminance scales each pixel to R + G + B, uncropped.
        let output = directory.path().join("super.xisf");
        let output = output.to_str().unwrap();
        let report = compose(
            seiza_color_combine_rgb_json,
            &json!({
                "red": red, "green": green, "blue": blue, "output": output,
                "normalization": "none", "luminanceMode": "super",
            }),
            ptr::null(),
        )
        .unwrap();
        assert_eq!(report["label"], "SUPER-RGB");
        assert!(report["crop"].is_null());
        assert_eq!(report["region"]["height"], 24);
        let pixels = rgb_pixels(output);
        let sum = pixels[pixels.len() - 3..].iter().sum::<f32>();
        assert!(sum > 1.2, "super-luminance lifts the triplet: {sum}");
    }

    #[test]
    fn color_cabi_composes_lrgb_and_narrowband_palettes_and_matrices() {
        let directory = tempfile::tempdir().unwrap();
        let headers = wcs_headers();
        let path =
            |name: &str, value: f32| color_channel(directory.path(), name, value, 0, &headers);
        let (luminance, red, green, blue) = (
            path("l.fits", 0.5),
            path("r.fits", 0.2),
            path("g.fits", 0.4),
            path("b.fits", 0.6),
        );
        let output = directory.path().join("lrgb.fits");
        let output = output.to_str().unwrap();
        let report = compose(
            seiza_color_combine_lrgb_json,
            &json!({
                "luminance": luminance, "red": red, "green": green, "blue": blue,
                "output": output, "normalization": "none", "luminanceWeight": 0.5,
            }),
            ptr::null(),
        )
        .unwrap();
        assert_eq!(report["label"], "LRGB");
        assert_eq!(
            FitsImage::open(Path::new(output))
                .unwrap()
                .header_str("SEIZACLR"),
            Some("LRGB")
        );
        let error = compose(
            seiza_color_combine_lrgb_json,
            &json!({
                "luminance": luminance, "red": red, "green": green, "blue": blue,
                "output": output, "luminanceMode": "super", "luminanceWeight": 0.5,
            }),
            ptr::null(),
        )
        .unwrap_err();
        assert!(error.contains("luminance weight only applies"), "{error}");

        let (ha, oiii, sii) = (
            path("ha.fits", 0.3),
            path("oiii.fits", 0.6),
            path("sii.fits", 0.1),
        );
        let output = directory.path().join("sho.fits");
        let output = output.to_str().unwrap();
        let report = compose(
            seiza_color_combine_narrowband_json,
            &json!({
                "ha": ha, "oiii": oiii, "sii": sii, "output": output,
                "normalization": "none",
            }),
            ptr::null(),
        )
        .unwrap();
        assert_eq!(report["label"], "SHO");
        assert_eq!(&rgb_pixels(output)[..3], &[0.1, 0.3, 0.6]);
        let error = compose(
            seiza_color_combine_narrowband_json,
            &json!({"ha": ha, "oiii": oiii, "output": output, "palette": "sho"}),
            ptr::null(),
        )
        .unwrap_err();
        assert!(error.contains("requires an SII channel"), "{error}");
        let error = compose(
            seiza_color_combine_narrowband_json,
            &json!({"ha": ha, "oiii": oiii, "sii": sii, "output": output, "palette": "hoo"}),
            ptr::null(),
        )
        .unwrap_err();
        assert!(error.contains("does not use an SII channel"), "{error}");
        let report = compose(
            seiza_color_combine_narrowband_json,
            &json!({"ha": ha, "oiii": oiii, "output": output, "palette": "Foraxx-HOO", "normalization": "none"}),
            ptr::null(),
        )
        .unwrap();
        assert_eq!(report["label"], "FORAXX-HOO");
        assert_eq!(report["transfer"], "display");

        let output = directory.path().join("matrix.fits");
        let output = output.to_str().unwrap();
        let report = compose(
            seiza_color_combine_narrowband_matrix_json,
            &json!({
                "ha": ha, "oiii": oiii, "output": output, "normalization": "none",
                "matrix": {
                    "red": {"ha": 1.0},
                    "green": {"ha": 0.5, "oiii": 0.5},
                    "blue": {"oiii": 1.0},
                },
            }),
            ptr::null(),
        )
        .unwrap();
        assert_eq!(report["label"], NarrowbandMatrix::LABEL);
        let pixels = rgb_pixels(output);
        assert!((pixels[0] - 0.3).abs() < 1e-6);
        assert!((pixels[1] - 0.45).abs() < 1e-6);
        assert!((pixels[2] - 0.6).abs() < 1e-6);
        let error = compose(
            seiza_color_combine_narrowband_matrix_json,
            &json!({
                "ha": ha, "oiii": oiii, "output": output,
                "matrix": {"red": {"sii": 1.0}, "green": {"ha": 1.0}, "blue": {"oiii": 1.0}},
            }),
            ptr::null(),
        )
        .unwrap_err();
        assert!(error.contains("SII"), "{error}");
    }

    #[test]
    fn color_cabi_refuses_shared_paths_unknown_fields_and_a_cancelled_signal() {
        let directory = tempfile::tempdir().unwrap();
        let red = color_channel(directory.path(), "red.fits", 0.2, 0, &[]);
        let green = color_channel(directory.path(), "green.fits", 0.4, 0, &[]);
        let blue = color_channel(directory.path(), "blue.fits", 0.6, 0, &[]);
        let error = compose(
            seiza_color_combine_rgb_json,
            &json!({"red": red, "green": green, "blue": blue, "output": red}),
            ptr::null(),
        )
        .unwrap_err();
        assert!(error.contains("same file"), "{error}");
        let output = directory.path().join("rgb.fits");
        let output = output.to_str().unwrap();
        let error = compose(
            seiza_color_combine_rgb_json,
            &json!({"red": red, "green": green, "blue": blue, "output": output, "palette": "sho"}),
            ptr::null(),
        )
        .unwrap_err();
        assert!(error.contains("palette"), "{error}");
        let error = compose(
            seiza_color_combine_rgb_json,
            &json!({"red": red, "green": green, "blue": blue, "output": output, "crop": "tight"}),
            ptr::null(),
        )
        .unwrap_err();
        assert!(error.contains("tight"), "{error}");

        let cancel = seiza_cancel_signal_create();
        unsafe { seiza_cancel_signal_cancel(cancel) };
        let error = compose(
            seiza_color_combine_rgb_json,
            &json!({"red": red, "green": green, "blue": blue, "output": output}),
            cancel,
        )
        .unwrap_err();
        assert!(error.contains("cancel"), "{error}");
        assert!(!Path::new(output).exists());
        unsafe { seiza_cancel_signal_free(cancel) };
    }

    /// Three panels across one star field, each overlapping the next by 80
    /// pixels; the first and third share no sky.
    fn overlapping_panels(directory: &Path) -> Vec<String> {
        let (width, height, step) = (192, 160, 112);
        let sky_width = width + 2 * step;
        let mut state = 41_u64;
        let mut random = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        let stars = (0..140)
            .map(|_| {
                (
                    8.0 + random() * (sky_width - 16) as f64,
                    8.0 + random() * (height - 16) as f64,
                    400.0 + 3600.0 * random(),
                )
            })
            .collect::<Vec<_>>();
        (0..3)
            .map(|panel| {
                let offset = (panel * step) as f64;
                let mut data = vec![100.0_f32; width * height];
                for &(sx, sy, peak) in &stars {
                    let x = sx - offset;
                    for py in (sy as isize - 5).max(0)..(sy as isize + 6).min(height as isize) {
                        for px in (x as isize - 5).max(0)..(x as isize + 6).min(width as isize) {
                            let d2 = (px as f64 - x).powi(2) + (py as f64 - sy).powi(2);
                            data[py as usize * width + px as usize] +=
                                (peak * (-d2 / 3.2).exp()) as f32;
                        }
                    }
                }
                let image = LinearImage::new(width, height, 1, data).unwrap();
                let path = directory.join(format!("panel-{panel}.fits"));
                seiza_stacking::write_processed_image_fits_f32(&path, &image, &[], &[]).unwrap();
                path.to_str().unwrap().to_owned()
            })
            .collect()
    }

    fn open_expanding(
        paths: &Value,
        flat: Option<&str>,
        cancel: *const SeizaCancelSignal,
    ) -> Result<*mut SeizaLiveStacker, String> {
        let paths = CString::new(paths.to_string()).unwrap();
        let flat = flat.map(|flat| CString::new(flat).unwrap());
        let options = CString::new(
            r#"{"normalization": {"mode": "none"}, "rejection": {"mode": "none"},
                "acceptance": {"minimum_overlap_fraction": 0.1}}"#,
        )
        .unwrap();
        let mut error = ptr::null_mut();
        let stacker = unsafe {
            seiza_live_stacker_open_fits_expanding(
                paths.as_ptr(),
                ptr::null(),
                ptr::null(),
                flat.as_ref().map_or(ptr::null(), |flat| flat.as_ptr()),
                0.0,
                options.as_ptr(),
                0,
                cancel,
                &mut error,
            )
        };
        if stacker.is_null() {
            let message = unsafe { CStr::from_ptr(error) }
                .to_str()
                .unwrap()
                .to_owned();
            unsafe { seiza_string_free(error) };
            return Err(message);
        }
        assert!(error.is_null());
        Ok(stacker)
    }

    fn stacker_json(
        call: unsafe extern "C" fn(*const SeizaLiveStacker, *mut *mut c_char) -> *mut c_char,
        stacker: *const SeizaLiveStacker,
    ) -> Value {
        let mut error = ptr::null_mut();
        let json = unsafe { call(stacker, &mut error) };
        assert!(error.is_null());
        let value =
            serde_json::from_str(unsafe { CStr::from_ptr(json) }.to_str().unwrap()).unwrap();
        unsafe { seiza_string_free(json) };
        value
    }

    #[test]
    fn expanding_stack_cabi_holds_every_panel() {
        let directory = tempfile::tempdir().unwrap();
        let paths = overlapping_panels(directory.path());
        // Anchored on the last panel, the first joins only through the middle.
        let order = [&paths[2], &paths[0], &paths[1]];
        let stacker = open_expanding(&json!(order), None, ptr::null()).unwrap();
        assert_eq!(unsafe { seiza_live_stacker_accepted_frames(stacker) }, 0);
        for path in order {
            let path = CString::new(path.as_str()).unwrap();
            let mut error = ptr::null_mut();
            let json =
                unsafe { seiza_live_stacker_push_fits_json(stacker, path.as_ptr(), &mut error) };
            assert!(error.is_null());
            let disposition: Value =
                serde_json::from_str(unsafe { CStr::from_ptr(json) }.to_str().unwrap()).unwrap();
            unsafe { seiza_string_free(json) };
            assert_eq!(disposition["accepted"], true, "{disposition}");
        }
        assert_eq!(unsafe { seiza_live_stacker_accepted_frames(stacker) }, 3);
        let width = unsafe { seiza_live_stacker_width(stacker) };
        assert!((412..=420).contains(&width), "canvas width {width}");
        // Undistorted panels leave the lens nothing to explain.
        assert!(stacker_json(seiza_live_stacker_expanded_lens_json, stacker).is_null());
        assert!(stacker_json(seiza_live_stacker_expansion_planning_notes_json, stacker).is_array());
        unsafe { seiza_live_stacker_free(stacker) };

        // An ordinary stack has no lens and no notes.
        let image = stacking_star_field(160, 128);
        let options = no_adjustment_stack_options();
        let mut error = ptr::null_mut();
        let ordinary = unsafe {
            seiza_live_stacker_create(
                image.as_ptr(),
                image.len(),
                160,
                128,
                1,
                options.as_ptr(),
                &mut error,
            )
        };
        assert!(stacker_json(seiza_live_stacker_expanded_lens_json, ordinary).is_null());
        assert_eq!(
            stacker_json(seiza_live_stacker_expansion_planning_notes_json, ordinary),
            json!([])
        );
        unsafe { seiza_live_stacker_free(ordinary) };
    }

    #[test]
    fn expanding_stack_cabi_refuses_bad_inputs_and_honours_cancel() {
        let directory = tempfile::tempdir().unwrap();
        let paths = overlapping_panels(directory.path());
        let error = open_expanding(&json!([]), None, ptr::null()).unwrap_err();
        assert!(error.contains("at least one light"), "{error}");
        let error = open_expanding(&json!(paths), Some(&paths[1]), ptr::null()).unwrap_err();
        assert!(error.contains("duplicate"), "{error}");
        let error = open_expanding(&json!([&paths[0], &paths[2]]), None, ptr::null()).unwrap_err();
        assert!(error.contains("could not join"), "{error}");
        let cancel = seiza_cancel_signal_create();
        unsafe { seiza_cancel_signal_cancel(cancel) };
        let error = open_expanding(&json!(paths), None, cancel).unwrap_err();
        assert!(error.contains("cancel"), "{error}");
        unsafe { seiza_cancel_signal_free(cancel) };
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

    fn owned_json_value(json: *mut c_char) -> Value {
        assert!(!json.is_null());
        let value = serde_json::from_str(unsafe { CStr::from_ptr(json) }.to_str().unwrap());
        unsafe { seiza_string_free(json) };
        value.unwrap()
    }

    #[test]
    fn stacking_cabi_opens_a_prepared_frame_and_reports_noise_weights_and_mappings() {
        let (width, height) = (160, 128);
        let image = stacking_star_field(width, height);
        let directory = tempfile::tempdir().unwrap();
        let prepared = directory.path().join("prepared.fits");
        seiza_stacking::write_processed_image_fits_f32(
            &prepared,
            &LinearImage::new(width, height, 1, image.clone()).unwrap(),
            &[],
            &[seiza_fits::WriteHeaderCard::new(
                "OBJECT",
                HeaderValue::String("M 31".into()),
            )],
        )
        .unwrap();
        let path = CString::new(prepared.to_str().unwrap()).unwrap();
        let options = CString::new(
            r#"{
                "normalization": {"mode": "none"},
                "rejection": {"mode": "none"},
                "weighting": {"mode": "inverse-noise-variance"}
            }"#,
        )
        .unwrap();
        let mut error = ptr::null_mut();
        let mut stacker = unsafe {
            seiza_live_stacker_open_prepared_fits(path.as_ptr(), options.as_ptr(), &mut error)
        };
        assert!(!stacker.is_null());
        assert!(error.is_null());
        let state = owned_json_value(unsafe { seiza_live_stacker_state_json(stacker, &mut error) });
        assert_eq!(state["inputMode"], "prepared-only");
        assert_eq!(state["acceptedFrames"], 1);

        let reference_noise = owned_json_value(unsafe {
            seiza_live_stacker_reference_noise_json(stacker, &mut error)
        });
        let reference_noise = reference_noise.as_array().unwrap();
        assert_eq!(reference_noise.len(), 1);
        assert!(reference_noise[0].as_f64().unwrap() > 0.0);
        let reference_mapping = owned_json_value(unsafe {
            seiza_live_stacker_reference_mapping_json(stacker, &mut error)
        });
        assert_eq!(reference_mapping["schema_version"], 1);
        assert_eq!(reference_mapping["reference_width"], width);

        let measured = owned_json_value(unsafe {
            seiza_frame_noise_json(image.as_ptr(), image.len(), width, height, 1, &mut error)
        });
        assert_eq!(&measured, &Value::Array(reference_noise.clone()));
        let tiny = [1.0_f32; 4];
        let unmeasured = owned_json_value(unsafe {
            seiza_frame_noise_json(tiny.as_ptr(), 4, 2, 2, 1, &mut error)
        });
        assert_eq!(unmeasured, Value::Null);
        assert!(unsafe { seiza_frame_noise_json(tiny.as_ptr(), 4, 4, 4, 1, &mut error) }.is_null());
        unsafe { seiza_string_free(error) };
        error = ptr::null_mut();

        let disposition = owned_json_value(unsafe {
            seiza_live_stacker_push_linear_json(
                stacker,
                image.as_ptr(),
                image.len(),
                width,
                height,
                1,
                &mut error,
            )
        });
        assert_eq!(disposition["accepted"], true);
        let diagnostics = &disposition["diagnostics"];
        assert_eq!(diagnostics["noise"].as_array().unwrap().len(), 1);
        let weight = diagnostics["weight"].as_array().unwrap();
        assert_eq!(weight.len(), 1);
        assert!(
            (weight[0].as_f64().unwrap() - 1.0).abs() < 1e-3,
            "{weight:?}"
        );
        assert_eq!(diagnostics["mapping"]["schema_version"], 1);
        assert!(diagnostics["mapping"]["transform"].is_object());

        let snapshot = unsafe { seiza_live_stacker_finish(&mut stacker, &mut error) };
        assert!(!snapshot.is_null());
        unsafe { seiza_stack_snapshot_free(snapshot) };

        // A raw CFA frame cannot skip preparation.
        let raw = directory.path().join("raw.fits");
        seiza_stacking::write_processed_image_fits_f32(
            &raw,
            &LinearImage::new(width, height, 1, image).unwrap(),
            &[],
            &[seiza_fits::WriteHeaderCard::new(
                "BAYERPAT",
                HeaderValue::String("RGGB".into()),
            )],
        )
        .unwrap();
        let raw = CString::new(raw.to_str().unwrap()).unwrap();
        let refused =
            unsafe { seiza_live_stacker_open_prepared_fits(raw.as_ptr(), ptr::null(), &mut error) };
        assert!(refused.is_null());
        assert!(
            unsafe { CStr::from_ptr(error) }
                .to_str()
                .unwrap()
                .contains("Bayer")
        );
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
    fn stacking_cabi_aligns_a_channel_to_a_shared_reference_image() {
        let directory = tempfile::tempdir().unwrap();
        let write = |name: &str, width: usize, height: usize| {
            let path = directory.path().join(name);
            let image = LinearImage::new(width, height, 1, stacking_star_field(width, height));
            seiza_stacking::write_processed_image_fits_f32(&path, &image.unwrap(), &[], &[])
                .unwrap();
            CString::new(path.to_str().unwrap()).unwrap()
        };
        // A larger shared image sets the grid every channel stacks onto.
        let reference = write("common.fits", 176, 144);
        let lights = [
            write("red-001.fits", 160, 128),
            write("red-002.fits", 160, 128),
        ];
        let bias = directory.path().join("bias.fits");
        seiza_stacking::write_processed_image_fits_f32(
            &bias,
            &LinearImage::new(160, 128, 1, vec![0.0; 160 * 128]).unwrap(),
            &[],
            &[],
        )
        .unwrap();
        let bias = CString::new(bias.to_str().unwrap()).unwrap();
        let context = directory.path().join("red.seiza-stack");
        let context = CString::new(context.to_str().unwrap()).unwrap();
        let config = no_adjustment_stack_options();
        let open = |reference: &CString, error: &mut *mut c_char| unsafe {
            seiza_live_stacker_open_fits_with_reference_image(
                reference.as_ptr(),
                lights[0].as_ptr(),
                bias.as_ptr(),
                ptr::null(),
                ptr::null(),
                0.0,
                config.as_ptr(),
                error,
            )
        };
        let state = |stacker: *const SeizaLiveStacker| {
            let mut error = ptr::null_mut();
            let json = unsafe { seiza_live_stacker_state_json(stacker, &mut error) };
            assert!(!json.is_null());
            let value: serde_json::Value =
                serde_json::from_str(unsafe { CStr::from_ptr(json) }.to_str().unwrap()).unwrap();
            unsafe { seiza_string_free(json) };
            value
        };

        let mut error = ptr::null_mut();
        assert!(open(&bias, &mut error).is_null());
        assert!(
            unsafe { CStr::from_ptr(error) }
                .to_str()
                .unwrap()
                .contains("duplicate stack input path")
        );
        unsafe { seiza_string_free(error) };
        error = ptr::null_mut();

        let mut stacker = open(&reference, &mut error);
        assert!(!stacker.is_null(), "{:?}", unsafe { CStr::from_ptr(error) });
        let opened = state(stacker);
        assert_eq!(opened["referenceIntegrated"], false);
        assert_eq!(opened["acceptedFrames"], 0);
        assert_eq!(
            (opened["width"].clone(), opened["height"].clone()),
            (176.into(), 144.into())
        );
        assert!(
            opened["referenceImagePath"]
                .as_str()
                .unwrap()
                .ends_with("common.fits")
        );
        for light in &lights {
            let disposition =
                unsafe { seiza_live_stacker_push_fits_json(stacker, light.as_ptr(), &mut error) };
            assert!(!disposition.is_null());
            unsafe { seiza_string_free(disposition) };
        }
        assert_eq!(unsafe { seiza_live_stacker_accepted_frames(stacker) }, 2);
        assert!(!unsafe {
            seiza_live_stacker_save_context(stacker, reference.as_ptr(), &mut error)
        });
        unsafe { seiza_string_free(error) };
        error = ptr::null_mut();
        assert!(unsafe { seiza_live_stacker_save_context(stacker, context.as_ptr(), &mut error) });
        unsafe { seiza_live_stacker_free(stacker) };

        stacker = unsafe { seiza_live_stacker_open_context(context.as_ptr(), &mut error) };
        assert!(!stacker.is_null());
        let resumed = state(stacker);
        assert_eq!(resumed["referenceIntegrated"], false);
        assert_eq!(resumed["referenceImagePath"], opened["referenceImagePath"]);
        assert_eq!(resumed["acceptedFrames"], 2);
        let snapshot = unsafe { seiza_live_stacker_snapshot(stacker, &mut error) };
        assert!(!snapshot.is_null());
        assert!(!unsafe {
            seiza_stack_snapshot_write_fits(snapshot, reference.as_ptr(), &mut error)
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
    fn stacking_cabi_tries_channel_references_until_one_aligns() {
        let directory = tempfile::tempdir().unwrap();
        let write = |name: &str, image: LinearImage| {
            let path = directory.path().join(name);
            seiza_stacking::write_processed_image_fits_f32(&path, &image, &[], &[]).unwrap();
            path.to_str().unwrap().to_owned()
        };
        let field = |width: usize, height: usize| {
            LinearImage::new(width, height, 1, stacking_star_field(width, height)).unwrap()
        };
        let reference = CString::new(write("common.fits", field(176, 144))).unwrap();
        // A quarter turn of a light cannot align to the shared image.
        let light = field(160, 128);
        let turned = (0..light.data.len())
            .map(|index| light.data[(127 - index % 128) * 160 + index / 128])
            .collect::<Vec<_>>();
        let turned = write(
            "red-turned.fits",
            LinearImage::new(128, 160, 1, turned).unwrap(),
        );
        let light = write("red-001.fits", light);
        let bias = write(
            "bias.fits",
            LinearImage::new(160, 128, 1, vec![0.0; 160 * 128]).unwrap(),
        );
        let bias_c = CString::new(bias.as_str()).unwrap();
        let config = no_adjustment_stack_options();
        // The turned light's size differs from the bias's, which would end
        // the search, so only the duplicate-path check uses the bias.
        let open_with = |bias: *const c_char,
                         candidates: &[&str],
                         selection: *mut *mut c_char,
                         error: &mut *mut c_char| {
            let candidates = CString::new(serde_json::to_string(candidates).unwrap()).unwrap();
            unsafe {
                seiza_live_stacker_open_fits_with_reference_image_candidates(
                    reference.as_ptr(),
                    candidates.as_ptr(),
                    bias,
                    ptr::null(),
                    ptr::null(),
                    0.0,
                    config.as_ptr(),
                    selection,
                    error,
                )
            }
        };
        let open = |candidates: &[&str], selection: *mut *mut c_char, error: &mut *mut c_char| {
            open_with(ptr::null(), candidates, selection, error)
        };
        let take_error = |error: &mut *mut c_char| {
            assert!(!error.is_null());
            let message = unsafe { CStr::from_ptr(*error) }
                .to_str()
                .unwrap()
                .to_owned();
            unsafe { seiza_string_free(*error) };
            *error = ptr::null_mut();
            message
        };

        let mut error = ptr::null_mut();
        let mut selection = ptr::null_mut();
        let stacker = open(&[&turned, &light], &mut selection, &mut error);
        assert!(!stacker.is_null(), "{:?}", unsafe { CStr::from_ptr(error) });
        let chosen: serde_json::Value =
            serde_json::from_str(unsafe { CStr::from_ptr(selection) }.to_str().unwrap()).unwrap();
        unsafe { seiza_string_free(selection) };
        assert_eq!(chosen["schemaVersion"], 1);
        assert_eq!(chosen["channelReferenceIndex"], 1);
        assert_eq!(chosen["channelReferencePath"], light.as_str());
        let skipped = chosen["skipped"].as_array().unwrap();
        assert_eq!(skipped.len(), 1);
        assert_eq!(skipped[0]["index"], 0);
        assert_eq!(skipped[0]["path"], turned.as_str());
        assert!(!skipped[0]["reason"].as_str().unwrap().is_empty());
        assert_eq!(unsafe { seiza_live_stacker_accepted_frames(stacker) }, 0);
        assert_eq!(unsafe { seiza_live_stacker_width(stacker) }, 176);
        unsafe { seiza_live_stacker_free(stacker) };

        // The selection is optional.
        let stacker = open(&[&light], ptr::null_mut(), &mut error);
        assert!(!stacker.is_null());
        unsafe { seiza_live_stacker_free(stacker) };

        // On failure, the selection stays null and the error says why.
        selection = CString::new("stale").unwrap().into_raw();
        let stale = selection;
        assert!(open(&[&turned], &mut selection, &mut error).is_null());
        assert!(selection.is_null());
        unsafe { seiza_string_free(stale) };
        let message = take_error(&mut error);
        assert!(message.contains("could not be aligned"), "{message}");
        assert!(open(&[], ptr::null_mut(), &mut error).is_null());
        assert!(take_error(&mut error).contains("at least one"));
        assert!(
            open_with(
                bias_c.as_ptr(),
                &[&light, &bias],
                ptr::null_mut(),
                &mut error
            )
            .is_null()
        );
        assert!(take_error(&mut error).contains("duplicate stack input path"));
    }

    #[test]
    fn retained_frames_reintegrate_identically_through_the_cabi() {
        let (width, height) = (160, 128);
        let clean = stacking_star_field(width, height);
        let directory = tempfile::tempdir().unwrap();
        let paths = (0..6)
            .map(|index| {
                let path = directory.path().join(format!("light-{index:03}.fits"));
                let noisy = clean
                    .iter()
                    .enumerate()
                    .map(|(sample, value)| value + ((sample * 7 + index * 13) % 5) as f32 * 0.1)
                    .collect::<Vec<_>>();
                seiza_stacking::write_processed_image_fits_f32(
                    &path,
                    &LinearImage::new(width, height, 1, noisy).unwrap(),
                    &[],
                    &[],
                )
                .unwrap();
                path.to_str().unwrap().to_owned()
            })
            .collect::<Vec<_>>();
        let scratch = directory.path().join("scratch");
        std::fs::create_dir(&scratch).unwrap();
        let scratch_c = CString::new(scratch.to_str().unwrap()).unwrap();
        let reference_c = CString::new(paths[0].as_str()).unwrap();
        let others_c = CString::new(serde_json::to_string(&paths[1..]).unwrap()).unwrap();
        let config = no_adjustment_stack_options();
        let mut error = ptr::null_mut();
        let replay = |retain: bool| {
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
            if retain {
                assert!(unsafe {
                    seiza_live_stacker_retain_frames_for_reintegration(
                        stacker,
                        scratch_c.as_ptr(),
                        &mut error,
                    )
                });
            }
            let response = unsafe {
                seiza_live_stacker_push_fits_pipelined_json(
                    stacker,
                    others_c.as_ptr(),
                    2,
                    0,
                    0.0,
                    &mut error,
                )
            };
            assert!(!response.is_null());
            unsafe { seiza_string_free(response) };
            let snapshot = unsafe {
                seiza_live_stacker_reintegrate(
                    stacker,
                    0.0,
                    0.0,
                    ptr::null(),
                    None,
                    ptr::null_mut(),
                    &mut error,
                )
            };
            assert!(!snapshot.is_null());
            let image = unsafe {
                std::slice::from_raw_parts(
                    seiza_stack_snapshot_image(snapshot),
                    seiza_stack_snapshot_data_length(snapshot),
                )
            }
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
            unsafe { seiza_stack_snapshot_free(snapshot) };
            (stacker, image)
        };
        let (retained, kept) = replay(true);
        assert_eq!(std::fs::read_dir(&scratch).unwrap().count(), 1);
        let (fresh, expected) = replay(false);
        assert_eq!(kept, expected);
        unsafe {
            seiza_live_stacker_free(retained);
            seiza_live_stacker_free(fresh);
        }
        assert_eq!(std::fs::read_dir(&scratch).unwrap().count(), 0);
        assert!(!unsafe {
            seiza_live_stacker_retain_frames_for_reintegration(
                ptr::null_mut(),
                ptr::null(),
                &mut error,
            )
        });
        unsafe { seiza_string_free(error) };
    }

    #[test]
    fn reintegration_cabi_rejects_a_reference_trail_the_live_stack_kept() {
        let (width, height) = (160, 128);
        let clean = stacking_star_field(width, height);
        let mut trailed = clean.clone();
        let trail_row = 60;
        for value in &mut trailed[trail_row * width + 10..trail_row * width + 150] {
            *value += 5000.0;
        }
        let directory = tempfile::tempdir().unwrap();
        let reference = directory.path().join("light-000.fits");
        seiza_stacking::write_processed_image_fits_f32(
            &reference,
            &LinearImage::new(width, height, 1, trailed).unwrap(),
            &[],
            &[],
        )
        .unwrap();
        let mut others = Vec::new();
        for index in 1..6 {
            let path = directory.path().join(format!("light-{index:03}.fits"));
            let noisy = clean
                .iter()
                .enumerate()
                .map(|(sample, value)| value + ((sample * 7 + index * 13) % 5) as f32 * 0.1)
                .collect::<Vec<_>>();
            seiza_stacking::write_processed_image_fits_f32(
                &path,
                &LinearImage::new(width, height, 1, noisy).unwrap(),
                &[],
                &[],
            )
            .unwrap();
            others.push(path.to_str().unwrap().to_owned());
        }
        let reference_c = CString::new(reference.to_str().unwrap()).unwrap();
        let paths_c = CString::new(serde_json::to_string(&others).unwrap()).unwrap();
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
        let response = unsafe {
            seiza_live_stacker_push_fits_pipelined_json(
                stacker,
                paths_c.as_ptr(),
                2,
                0,
                0.0,
                &mut error,
            )
        };
        assert!(!response.is_null(), "{:?}", unsafe {
            error.as_ref().map(|e| CStr::from_ptr(e))
        });
        unsafe { seiza_string_free(response) };
        assert_eq!(unsafe { seiza_live_stacker_accepted_frames(stacker) }, 6);

        let state = unsafe { seiza_live_stacker_state_json(stacker, &mut error) };
        let parsed =
            serde_json::from_str::<Value>(unsafe { CStr::from_ptr(state) }.to_str().unwrap())
                .unwrap();
        unsafe { seiza_string_free(state) };
        assert!(parsed["reintegrationUnavailable"].is_null(), "{parsed}");

        unsafe extern "C" fn count_reads(
            pass: u32,
            index: usize,
            count: usize,
            context: *mut c_void,
        ) {
            let reads = unsafe { &mut *(context as *mut Vec<(u32, usize, usize)>) };
            reads.push((pass, index, count));
        }
        let mut reads = Vec::<(u32, usize, usize)>::new();
        let snapshot = unsafe {
            seiza_live_stacker_reintegrate(
                stacker,
                0.0,
                0.0,
                ptr::null(),
                Some(count_reads),
                (&mut reads as *mut Vec<(u32, usize, usize)>).cast(),
                &mut error,
            )
        };
        assert!(!snapshot.is_null(), "{:?}", unsafe {
            error.as_ref().map(|e| CStr::from_ptr(e))
        });
        // Each frame is prepared whole and kept in a scratch file (pass 0),
        // then the three passes run on bands of every frame (pass 2).
        let expected = [0, 2]
            .into_iter()
            .flat_map(|pass| (0..6).map(move |index| (pass, index, 6)))
            .collect::<Vec<_>>();
        assert_eq!(reads, expected);
        let sample = trail_row * width + 80;
        let replayed = unsafe {
            std::slice::from_raw_parts(
                seiza_stack_snapshot_image(snapshot),
                seiza_stack_snapshot_data_length(snapshot),
            )
        };
        let live = unsafe {
            std::slice::from_raw_parts(
                seiza_live_stacker_mean(stacker),
                seiza_live_stacker_data_length(stacker),
            )
        };
        assert!(
            live[sample] - clean[sample] > 500.0,
            "the live mean keeps the trail"
        );
        assert!(
            (replayed[sample] - clean[sample]).abs() < 1.0,
            "reintegration rejects it: {} vs {}",
            replayed[sample],
            clean[sample]
        );
        assert_eq!(unsafe { seiza_stack_snapshot_accepted_frames(snapshot) }, 6);

        // The drizzled replay rejects the same trail on a grid twice as
        // fine, and hands back the reintegrated stack alongside.
        let mut integrated = ptr::null_mut();
        let drizzled = unsafe {
            seiza_live_stacker_reintegrate_drizzled(
                stacker,
                0.0,
                0.0,
                2,
                0.0,
                ptr::null(),
                None,
                ptr::null_mut(),
                &mut integrated,
                &mut error,
            )
        };
        assert!(!drizzled.is_null(), "{:?}", unsafe {
            error.as_ref().map(|e| CStr::from_ptr(e))
        });
        assert!(!integrated.is_null());
        unsafe {
            assert_eq!(seiza_drizzle_result_scale(drizzled), 2);
            assert_eq!(seiza_drizzle_result_width(drizzled), width * 2);
            assert_eq!(seiza_drizzle_result_height(drizzled), height * 2);
            assert_eq!(seiza_drizzle_result_channels(drizzled), 1);
            assert_eq!(seiza_drizzle_result_accepted_frames(drizzled), 6);
        }
        let image = unsafe {
            std::slice::from_raw_parts(
                seiza_drizzle_result_image(drizzled),
                seiza_drizzle_result_data_length(drizzled),
            )
        };
        let weight = unsafe {
            std::slice::from_raw_parts(
                seiza_drizzle_result_weight(drizzled),
                seiza_drizzle_result_data_length(drizzled),
            )
        };
        // Reference row 60 covers output rows 120 and 121.
        for row in [120, 121] {
            let output = row * width * 2 + 160;
            assert!(weight[output] > 0.0);
            assert!(
                (image[output] - clean[sample]).abs() < 50.0,
                "the drizzle leaves the trail out: {} vs {}",
                image[output],
                clean[sample]
            );
        }
        let integrated_image = unsafe {
            std::slice::from_raw_parts(
                seiza_stack_snapshot_image(integrated),
                seiza_stack_snapshot_data_length(integrated),
            )
        };
        assert_eq!(
            integrated_image[sample].to_bits(),
            replayed[sample].to_bits()
        );
        let output = directory.path().join("drizzled.fits");
        let output_c = CString::new(output.to_str().unwrap()).unwrap();
        assert!(unsafe {
            seiza_drizzle_result_write_fits(drizzled, output_c.as_ptr(), &mut error)
        });
        let written = FitsFrame::open(&output).unwrap();
        assert_eq!(
            (written.image.width, written.image.height),
            (width * 2, height * 2)
        );
        assert!(!unsafe {
            seiza_drizzle_result_write_fits(drizzled, reference_c.as_ptr(), &mut error)
        });
        unsafe {
            seiza_string_free(error);
            error = ptr::null_mut();
            seiza_drizzle_result_free(drizzled);
            seiza_stack_snapshot_free(integrated);
        }

        // A cancelled replay stops with an error rather than a snapshot.
        let cancel = seiza_cancel_signal_create();
        unsafe { seiza_cancel_signal_cancel(cancel) };
        let cancelled = unsafe {
            seiza_live_stacker_reintegrate(
                stacker,
                3.0,
                3.0,
                cancel,
                None,
                ptr::null_mut(),
                &mut error,
            )
        };
        assert!(cancelled.is_null());
        assert!(!error.is_null());
        unsafe {
            seiza_string_free(error);
            seiza_cancel_signal_free(cancel);
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
    fn live_preview_area_sampling_keeps_off_center_signal_and_ignores_gaps() {
        let mut mean = vec![0.0_f32; 8 * 8 * 3];
        mean[..3].copy_from_slice(&[16.0, 32.0, 48.0]);
        let mut coverage = vec![1_u32; mean.len()];
        // One invalid sample in the first footprint, and an entirely empty
        // bottom-right footprint. Neither may supply black image data.
        coverage[3] = 0;
        mean[3] = 9999.0;
        for y in 4..8 {
            for x in 4..8 {
                coverage[(y * 8 + x) * 3..(y * 8 + x + 1) * 3].fill(0);
            }
        }
        let rejected = vec![0_u32; mean.len()];
        let view = seiza_stacking::StackView {
            width: 8,
            height: 8,
            channels: 3,
            mean: &mean,
            coverage: &coverage,
            rejected_samples: &rejected,
            accepted_frames: 2,
            rejected_frames: 0,
        };
        let (w, h, sampled, mask) = sample_live_stack_view(view, 2).unwrap();
        assert_eq!((w, h), (2, 2));
        assert!((sampled[0] - 16.0 / 15.0).abs() < 1e-6);
        assert_eq!(&sampled[1..3], &[2.0, 3.0]);
        assert_eq!(mask, [true, true, true, false]);
        assert!(sampled[9..].iter().all(|value| value.is_nan()));
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

    /// Deterministic noise in [-0.5, 0.5) for test frames.
    fn screening_noise(seed: u64, count: usize) -> Vec<f32> {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    /// Call a JSON-in, JSON-out C export and parse its answer, or return its
    /// error.
    fn call_json(
        call: unsafe extern "C" fn(*const c_char, *mut *mut c_char) -> *mut c_char,
        request: &Value,
    ) -> Result<Value, String> {
        let request = CString::new(request.to_string()).unwrap();
        let mut error = ptr::null_mut();
        let result = unsafe { call(request.as_ptr(), &mut error) };
        if result.is_null() {
            let message = unsafe { CStr::from_ptr(error) }
                .to_str()
                .unwrap()
                .to_owned();
            unsafe { seiza_string_free(error) };
            return Err(message);
        }
        assert!(error.is_null());
        let parsed = serde_json::from_str(unsafe { CStr::from_ptr(result) }.to_str().unwrap());
        unsafe { seiza_string_free(result) };
        Ok(parsed.unwrap())
    }

    #[test]
    fn calibration_frames_are_measured_from_files_and_screened_as_sets() {
        let directory = tempfile::tempdir().unwrap();
        let (width, height) = (64, 64);
        // Four darks at one level; the last caught light and sits well above.
        let mut dark_paths = Vec::new();
        for index in 0..4 {
            let lift = if index == 3 { 40.0 } else { 0.0 };
            let data = screening_noise(index, width * height)
                .into_iter()
                .map(|noise| 500.0 + lift + 8.0 * noise)
                .collect();
            let path = directory.path().join(format!("dark-{index}.fits"));
            seiza_stacking::write_processed_image_fits_f32(
                &path,
                &LinearImage::new(width, height, 1, data).unwrap(),
                &[],
                &[],
            )
            .unwrap();
            dark_paths.push(path);
        }
        let darks: Vec<Value> = dark_paths
            .iter()
            .map(|path| {
                call_json(
                    seiza_calibration_measure_dark_json,
                    &json!({"path": path, "hotSigma": 10.0}),
                )
                .unwrap()
            })
            .collect();
        // The C measurement is the core's, in camelCase.
        let frame = FitsFrame::open(&dark_paths[0]).unwrap();
        let core = seiza_calibration::measure_dark(
            seiza_calibration::LinearImageRef::new(&frame.image.data, width, height, 1).unwrap(),
            &seiza_calibration::DarkMeasureOptions::default(),
        )
        .unwrap();
        let expected: Value =
            serde_json::from_str(&serde_json::to_string(&DarkMeasurementJson::from(core)).unwrap())
                .unwrap();
        assert_eq!(darks[0], expected);
        assert_eq!(darks[0]["width"], 64);
        assert!(darks[0]["rowBanding"].is_number());
        assert_eq!(
            darks[0]["algorithmVersion"],
            seiza_calibration::FRAME_QUALITY_ALGORITHM_VERSION
        );

        let screened = call_json(
            seiza_calibration_screen_darks_json,
            &json!({"measurements": darks}),
        )
        .unwrap();
        assert_eq!(screened["schemaVersion"], 1);
        let frames = screened["frames"].as_array().unwrap();
        assert_eq!(frames.len(), 4);
        assert_eq!(frames[0]["grade"], "ok");
        assert_eq!(frames[3]["grade"], "poor");
        assert_eq!(frames[3]["findings"][0]["code"], "level_excess");
        // A limit the host gives replaces the default.
        let lenient = call_json(
            seiza_calibration_screen_darks_json,
            &json!({"measurements": darks, "levelExcess": {"suspect": 1.0e9, "poor": 2.0e9}}),
        )
        .unwrap();
        assert!(
            lenient["frames"][3]["findings"]
                .as_array()
                .unwrap()
                .iter()
                .all(|finding| finding["code"] != "level_excess")
        );
        // Measurements of a dark set read as bias too; the odd level shows.
        let bias = call_json(
            seiza_calibration_screen_bias_json,
            &json!({"measurements": darks}),
        )
        .unwrap();
        assert_eq!(bias["frames"][3]["findings"][0]["code"], "level");
        // An unknown limit is refused rather than ignored.
        let refused = call_json(
            seiza_calibration_screen_bias_json,
            &json!({"measurements": darks, "levelExcess": {"suspect": 1.0, "poor": 2.0}}),
        );
        assert!(refused.unwrap_err().contains("levelExcess"));

        // Three good flats and one clipped.
        let mut flats = Vec::new();
        for index in 0..4 {
            let data = if index == 3 {
                vec![65535.0; width * height]
            } else {
                screening_noise(10 + index, width * height)
                    .into_iter()
                    .map(|noise| 30000.0 + 100.0 * noise)
                    .collect()
            };
            let path = directory.path().join(format!("flat-{index}.fits"));
            seiza_stacking::write_processed_image_fits_f32(
                &path,
                &LinearImage::new(width, height, 1, data).unwrap(),
                &[],
                &[],
            )
            .unwrap();
            flats.push(
                call_json(
                    seiza_calibration_measure_flat_json,
                    &json!({"path": path, "fullScale": 65535.0}),
                )
                .unwrap(),
            );
        }
        assert_eq!(flats[0]["channelLevels"].as_array().unwrap().len(), 1);
        assert!(flats[0]["illumination"]["response"].is_array());
        assert_eq!(
            flats[0]["algorithmVersion"],
            seiza_calibration::FLAT_QUALITY_ALGORITHM_VERSION
        );
        // An obstruction without the rest of the light cone is refused, not
        // ignored.
        let refused = call_json(
            seiza_calibration_measure_flat_json,
            &json!({"path": directory.path().join("flat-0.fits"), "obstruction": 0.35}),
        )
        .unwrap_err();
        assert!(refused.contains("focal ratio"), "{refused}");
        let report = call_json(
            seiza_calibration_screen_flats_json,
            &json!({"measurements": flats}),
        )
        .unwrap();
        assert_eq!(report["schemaVersion"], 1);
        assert_eq!(report["frames"][0]["grade"], "ok");
        assert_eq!(report["frames"][3]["grade"], "poor");
        assert_eq!(report["frames"][3]["findings"][0]["code"], "clipped");
        assert!(report["master"]["grade"].is_string());
        assert!(report["steadyFeatures"].is_number());

        // A missing file and a flat set of mixed sizes fail with a reason.
        let missing = directory.path().join("missing.fits");
        assert!(
            call_json(
                seiza_calibration_measure_dark_json,
                &json!({"path": missing})
            )
            .is_err()
        );
        let mut mixed = flats[0].clone();
        mixed["width"] = json!(32);
        let error = call_json(
            seiza_calibration_screen_flats_json,
            &json!({"measurements": [flats[0].clone(), mixed]}),
        )
        .unwrap_err();
        assert!(error.contains("cannot form one set"), "{error}");
    }

    #[test]
    fn flat_features_are_camel_case_both_ways() {
        let mut feature = seiza_calibration::FlatFeature::default();
        feature.x = 10.0;
        feature.y = 20.0;
        feature.right = 40;
        feature.bottom = 40;
        feature.pixels = 1257;
        feature.depth = 0.003;
        feature.shape = seiza_calibration::dust::DustShape::Disk;
        feature.radius = 20.0;
        feature.distance_mm = Some(1.5);
        feature.seen_in = Some(seiza_calibration::dust::SeenIn { frames: 9, of: 10 });
        let json = serde_json::to_value(FlatFeatureJson::from(feature)).unwrap();
        assert_eq!(json["distanceMm"], 1.5);
        assert_eq!(json["seenIn"]["frames"], 9);
        assert!(json.get("distance_mm").is_none() && json.get("hole").is_none());
        let back: FlatFeatureJson = serde_json::from_value(json).unwrap();
        assert_eq!(seiza_calibration::FlatFeature::from(back), feature);
    }

    /// A 32x32 crop of sky with a ring of lost light at its centre and a
    /// star that moves from crop to crop, as seiza-calibration's own test
    /// builds it.
    fn residual_crop(width: usize, offset: f32, shadow: bool) -> Vec<f32> {
        (0..width)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let background = 1_000.0 + 1.5 * x as f32 + 0.75 * y as f32 + offset;
                    let radius = (x as f32 - 15.5).hypot(y as f32 - 15.5);
                    let response = if shadow && (6.0..=9.0).contains(&radius) {
                        0.8
                    } else {
                        1.0
                    };
                    let star = if (x + offset as usize) % width == y {
                        2_000.0
                    } else {
                        0.0
                    };
                    background * response + star
                })
            })
            .collect()
    }

    #[test]
    fn residual_flat_responses_are_built_checked_and_applied_as_the_core_does() {
        let crops: Vec<Vec<f32>> = (0..7)
            .map(|index| residual_crop(32, index as f32 * 3.0, true))
            .collect();
        let pointers: Vec<*const f32> = crops.iter().map(|crop| crop.as_ptr()).collect();
        let options =
            CString::new(r#"{"smoothingSigma": 0.8, "edgeFeatherFraction": 0.0}"#).unwrap();
        let mut response = vec![0.0f32; 32 * 32];
        let mut error = ptr::null_mut();
        let diagnostics = unsafe {
            seiza_calibration_build_residual_flat_json(
                pointers.as_ptr(),
                pointers.len(),
                32,
                32,
                1,
                options.as_ptr(),
                response.as_mut_ptr(),
                response.len(),
                &mut error,
            )
        };
        assert!(!diagnostics.is_null());
        let parsed: Value =
            serde_json::from_str(unsafe { CStr::from_ptr(diagnostics) }.to_str().unwrap()).unwrap();
        unsafe { seiza_string_free(diagnostics) };

        // The same pixels and evidence as the core.
        let views: Vec<_> = crops
            .iter()
            .map(|crop| seiza_calibration::LinearImageRef::new(crop, 32, 32, 1).unwrap())
            .collect();
        let core = seiza_calibration::build_residual_flat_patch(
            &views,
            &seiza_calibration::ResidualFlatOptions {
                smoothing_sigma: 0.8,
                edge_feather_fraction: 0.0,
                ..seiza_calibration::ResidualFlatOptions::default()
            },
        )
        .unwrap();
        assert_eq!(
            response
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            core.patch
                .response()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(parsed["sampleCount"], 7);
        assert_eq!(
            parsed["correctedSamples"],
            core.diagnostics.corrected_samples
        );
        assert_eq!(
            parsed["algorithmVersion"],
            seiza_calibration::RESIDUAL_FLAT_ALGORITHM_VERSION
        );

        assert!(unsafe {
            seiza_calibration_validate_residual_flat_response(
                response.as_ptr(),
                response.len(),
                32,
                32,
                1,
                &mut error,
            )
        });
        let mut light = residual_crop(40, 0.0, false);
        let mut expected = light.clone();
        core.patch
            .apply_at(
                seiza_calibration::LinearImageMut::new(&mut expected, 40, 40, 1).unwrap(),
                4,
                4,
            )
            .unwrap();
        assert!(unsafe {
            seiza_calibration_apply_residual_flat_response(
                response.as_ptr(),
                response.len(),
                32,
                32,
                light.as_mut_ptr(),
                light.len(),
                40,
                40,
                1,
                4,
                4,
                &mut error,
            )
        });
        assert_eq!(light, expected);

        // A response past the light's edge leaves the light alone.
        let before = light.clone();
        assert!(!unsafe {
            seiza_calibration_apply_residual_flat_response(
                response.as_ptr(),
                response.len(),
                32,
                32,
                light.as_mut_ptr(),
                light.len(),
                40,
                40,
                1,
                20,
                20,
                &mut error,
            )
        });
        assert!(!error.is_null());
        unsafe { seiza_string_free(error) };
        assert_eq!(light, before);
        // A gain above one is not a response.
        let mut bad = response.clone();
        bad[0] = 1.5;
        let mut error = ptr::null_mut();
        assert!(!unsafe {
            seiza_calibration_validate_residual_flat_response(
                bad.as_ptr(),
                bad.len(),
                32,
                32,
                1,
                &mut error,
            )
        });
        assert!(!error.is_null());
        unsafe { seiza_string_free(error) };
        // Crops without a shared shadow give no response.
        let plain: Vec<Vec<f32>> = (0..7)
            .map(|index| residual_crop(32, index as f32 * 3.0, false))
            .collect();
        let pointers: Vec<*const f32> = plain.iter().map(|crop| crop.as_ptr()).collect();
        let mut error = ptr::null_mut();
        let none = unsafe {
            seiza_calibration_build_residual_flat_json(
                pointers.as_ptr(),
                pointers.len(),
                32,
                32,
                1,
                ptr::null(),
                response.as_mut_ptr(),
                response.len(),
                &mut error,
            )
        };
        assert!(none.is_null());
        assert!(!error.is_null());
        unsafe { seiza_string_free(error) };
    }

    #[test]
    fn registered_frames_are_integrated_with_leave_one_out_rejection() {
        unsafe extern "C" fn count(pass: u32, index: usize, total: usize, context: *mut c_void) {
            let calls = unsafe { &mut *(context as *mut Vec<(u32, usize, usize)>) };
            calls.push((pass, index, total));
        }
        let directory = tempfile::tempdir().unwrap();
        let (width, height) = (16, 16);
        let mut inputs = Vec::new();
        for index in 0..5 {
            let mut data: Vec<f32> = screening_noise(20 + index, width * height)
                .into_iter()
                .map(|noise| 10.0 + noise)
                .collect();
            if index == 2 {
                // A trail in one frame, and an edge it does not cover.
                data[40] = 1_000.0;
                data[0] = f32::NAN;
            }
            let path = directory.path().join(format!("registered-{index}.fits"));
            seiza_stacking::write_processed_image_fits_f32(
                &path,
                &LinearImage::new(width, height, 1, data).unwrap(),
                &[],
                &[seiza_fits::WriteHeaderCard::new(
                    "OBJECT",
                    HeaderValue::String(format!("frame {index}")),
                )],
            )
            .unwrap();
            inputs.push(path);
        }
        let request = CString::new(json!({"inputs": inputs}).to_string()).unwrap();
        let mut calls: Vec<(u32, usize, usize)> = Vec::new();
        let mut report = ptr::null_mut();
        let mut error = ptr::null_mut();
        let snapshot = unsafe {
            seiza_stack_integrate_registered_json(
                request.as_ptr(),
                ptr::null(),
                Some(count),
                &mut calls as *mut _ as *mut c_void,
                &mut report,
                &mut error,
            )
        };
        assert!(!snapshot.is_null());
        assert!(error.is_null());
        let stacked = unsafe { &*snapshot };
        assert!(stacked.snapshot.image.data[40] < 11.0);
        assert_eq!(stacked.snapshot.accepted_frames, 5);
        assert!(
            calls
                .iter()
                .any(|&(pass, _, total)| pass == 2 && total == 5)
        );
        let parsed: Value =
            serde_json::from_str(unsafe { CStr::from_ptr(report) }.to_str().unwrap()).unwrap();
        unsafe { seiza_string_free(report) };
        let frames = parsed["frames"].as_array().unwrap();
        assert_eq!(frames.len(), 5);
        assert_eq!(frames[2]["path"], inputs[2].to_str().unwrap());
        assert_eq!(frames[2]["finiteSamples"], width * height - 1);
        assert!(frames[2]["integratedSamples"].as_u64().unwrap() < (width * height - 1) as u64);
        assert_eq!(frames[0]["finiteSamples"], width * height);

        // It writes, and never over an input.
        let output = directory.path().join("stack.fits");
        let output_c = CString::new(output.to_str().unwrap()).unwrap();
        assert!(unsafe {
            seiza_stack_snapshot_write_fits(snapshot, output_c.as_ptr(), &mut error)
        });
        assert_eq!(FitsFrame::open(&output).unwrap().image.width, width);
        let input_c = CString::new(inputs[0].to_str().unwrap()).unwrap();
        assert!(!unsafe {
            seiza_stack_snapshot_write_fits(snapshot, input_c.as_ptr(), &mut error)
        });
        unsafe { seiza_string_free(error) };
        unsafe { seiza_stack_snapshot_free(snapshot) };

        // Frames of different shapes are refused.
        let small = directory.path().join("small.fits");
        seiza_stacking::write_processed_image_fits_f32(
            &small,
            &LinearImage::new(8, 8, 1, vec![10.0; 64]).unwrap(),
            &[],
            &[],
        )
        .unwrap();
        let request =
            CString::new(json!({"inputs": [inputs[0], inputs[1], small]}).to_string()).unwrap();
        let mut error = ptr::null_mut();
        let mut report = ptr::null_mut();
        let refused = unsafe {
            seiza_stack_integrate_registered_json(
                request.as_ptr(),
                ptr::null(),
                None,
                ptr::null_mut(),
                &mut report,
                &mut error,
            )
        };
        assert!(refused.is_null() && report.is_null() && !error.is_null());
        unsafe { seiza_string_free(error) };
    }

    #[test]
    fn channels_fitted_apart_share_one_background_surface() {
        let (width, height) = (96, 64);
        let channel = |tilt: f32, curve: f32| -> Vec<f32> {
            (0..height)
                .flat_map(|y| {
                    (0..width).map(move |x| {
                        let u = x as f32 / width as f32 - 0.5;
                        let v = y as f32 / height as f32 - 0.5;
                        100.0 + tilt * u + curve * (u * u + v * v)
                    })
                })
                .collect()
        };
        let channels = [channel(0.0, 0.0), channel(8.0, 0.0), channel(4.0, 30.0)];
        let config = CString::new(
            json!({"model": {"kind": "automatic", "max_degree": 2}, "search_steps": 0}).to_string(),
        )
        .unwrap();
        let mut error = ptr::null_mut();
        let models: Vec<*mut SeizaBackgroundModel> = channels
            .iter()
            .map(|data| {
                let model = unsafe {
                    seiza_background_fit(
                        data.as_ptr(),
                        data.len(),
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
                model
            })
            .collect();
        let pointers: Vec<*const SeizaBackgroundModel> =
            models.iter().map(|&model| model as *const _).collect();
        let shared = unsafe {
            seiza_background_select_shared_model_json(
                pointers.as_ptr(),
                pointers.len(),
                config.as_ptr(),
                &mut error,
            )
        };
        assert!(!shared.is_null());
        let text = unsafe { CStr::from_ptr(shared) }
            .to_str()
            .unwrap()
            .to_owned();
        unsafe { seiza_string_free(shared) };
        let shared: Value = serde_json::from_str(&text).unwrap();
        let automatic: BackgroundConfig = serde_json::from_str(config.to_str().unwrap()).unwrap();
        let fits: Vec<&BackgroundFit> = models
            .iter()
            .map(|&model| unsafe { &(*model).fit })
            .collect();
        let expected =
            seiza_background::select_shared_model(&automatic.model, &fits).map(|model| {
                BackgroundConfig {
                    model,
                    ..automatic.clone()
                }
            });
        assert!(expected.is_some());
        assert_eq!(shared, serde_json::to_value(&expected).unwrap());
        assert_ne!(shared["model"]["kind"], "automatic");

        // A config that is not automatic has no shared choice.
        let fixed = CString::new(
            json!({"model": {"kind": "polynomial", "degree": 1, "ridge": 1.0e-8}}).to_string(),
        )
        .unwrap();
        let none = unsafe {
            seiza_background_select_shared_model_json(
                pointers.as_ptr(),
                pointers.len(),
                fixed.as_ptr(),
                &mut error,
            )
        };
        assert!(!none.is_null());
        assert_eq!(unsafe { CStr::from_ptr(none) }.to_str().unwrap(), "null");
        unsafe { seiza_string_free(none) };
        for model in models {
            unsafe { seiza_background_model_free(model) };
        }
    }

    #[test]
    fn a_stack_and_the_planner_read_a_light_at_the_angle_the_host_measured() {
        let (width, height) = (160, 128);
        let directory = tempfile::tempdir().unwrap();
        let reference_path = directory.path().join("reference.fits");
        let flat_path = directory.path().join("master-flat.fits");
        let rotation =
            |degrees| seiza_fits::WriteHeaderCard::new("ROTATANG", HeaderValue::Float(degrees));
        seiza_stacking::write_processed_image_fits_f32(
            &reference_path,
            &LinearImage::new(width, height, 1, stacking_star_field(width, height)).unwrap(),
            &[],
            &[
                seiza_fits::WriteHeaderCard::new("IMAGETYP", HeaderValue::String("LIGHT".into())),
                rotation(3.2),
            ],
        )
        .unwrap();
        seiza_stacking::write_processed_image_fits_f32(
            &flat_path,
            &LinearImage::new(width, height, 1, vec![1.0; width * height]).unwrap(),
            &[],
            &[
                seiza_fits::WriteHeaderCard::new("SEIZAMST", HeaderValue::String("FLAT".into())),
                rotation(3.2),
            ],
        )
        .unwrap();
        let reference_c = CString::new(reference_path.to_str().unwrap()).unwrap();
        let flat_c = CString::new(flat_path.to_str().unwrap()).unwrap();
        let config = no_adjustment_stack_options();
        let mut error = ptr::null_mut();
        let stacker = unsafe {
            seiza_live_stacker_open_fits(
                reference_c.as_ptr(),
                ptr::null(),
                ptr::null(),
                flat_c.as_ptr(),
                0.0,
                config.as_ptr(),
                &mut error,
            )
        };
        assert!(!stacker.is_null());

        // A light whose rotator reads 113.8 degrees, though the host
        // measured it at 3.5: the flat's angle.
        let mut light: SeizaFrameSignature = unsafe { std::mem::zeroed() };
        light.width = width as f64;
        light.height = height as f64;
        light.channels = 1.0;
        light.rotation_deg = 113.8;
        light.known = SEIZA_FRAME_HAS_WIDTH
            | SEIZA_FRAME_HAS_HEIGHT
            | SEIZA_FRAME_HAS_CHANNELS
            | SEIZA_FRAME_HAS_ROTATION;
        let validate = || {
            let mut error = ptr::null_mut();
            let validated = unsafe {
                seiza_live_stacker_validate_calibration(stacker, &light, ptr::null(), &mut error)
            };
            if !error.is_null() {
                unsafe { seiza_string_free(error) };
            }
            validated
        };
        let read = || {
            let (mut basis, mut degrees) = (u32::MAX, f64::NAN);
            assert!(unsafe {
                seiza_live_stacker_light_rotation(stacker, &mut basis, &mut degrees)
            });
            (basis, degrees)
        };
        assert_eq!(read(), (SEIZA_LIGHT_ROTATION_RECORDED, 0.0));
        assert_eq!(validate(), 0);

        assert!(unsafe {
            seiza_live_stacker_set_light_rotation(
                stacker,
                SEIZA_LIGHT_ROTATION_MEASURED,
                3.5,
                &mut error,
            )
        });
        assert_eq!(read(), (SEIZA_LIGHT_ROTATION_MEASURED, 3.5));
        assert_eq!(validate(), 1);
        // The other tolerances stay as they were.
        let mut own = SeizaMatchTolerances::default();
        assert!(unsafe { seiza_live_stacker_match_tolerances(stacker, &mut own) });
        assert_eq!(own.rotation_deg, 2.0);

        assert!(unsafe {
            seiza_live_stacker_set_light_rotation(
                stacker,
                SEIZA_LIGHT_ROTATION_UNKNOWN,
                0.0,
                &mut error,
            )
        });
        assert_eq!(read(), (SEIZA_LIGHT_ROTATION_UNKNOWN, 0.0));
        assert_eq!(validate(), 1);

        // Setting the whole gate reads the recorded angle again.
        assert!(unsafe {
            seiza_live_stacker_set_match_tolerances(stacker, ptr::null(), &mut error)
        });
        assert_eq!(read().0, SEIZA_LIGHT_ROTATION_RECORDED);

        // A bad basis or angle is refused and changes nothing.
        for (basis, degrees) in [(7, 0.0), (SEIZA_LIGHT_ROTATION_MEASURED, f64::NAN)] {
            let mut error = ptr::null_mut();
            assert!(!unsafe {
                seiza_live_stacker_set_light_rotation(stacker, basis, degrees, &mut error)
            });
            assert!(!error.is_null());
            unsafe { seiza_string_free(error) };
        }
        assert_eq!(read().0, SEIZA_LIGHT_ROTATION_RECORDED);
        assert!(!unsafe {
            seiza_live_stacker_light_rotation(ptr::null(), ptr::null_mut(), ptr::null_mut())
        });
        unsafe { seiza_live_stacker_free(stacker) };

        // The planner picks the flat for a light at its measured angle.
        let plan = |light_rotation: Value| {
            call_json(
                seiza_calibration_plan_json,
                &json!({
                    "kind": "flat", "minimum": 1,
                    "reference": {"path": "light.fits", "role": "light",
                                  "signature": {"camera": "ASI2600MM", "width": 100,
                                                "height": 80, "channels": 1,
                                                "rotationDeg": 113.8}},
                    "candidates": [{"path": "flat.fits", "role": "flat",
                                    "signature": {"camera": "ASI2600MM", "width": 100,
                                                  "height": 80, "channels": 1,
                                                  "rotationDeg": 3.2}}],
                    "tolerances": {"lightRotation": light_rotation}
                }),
            )
        };
        let recorded = plan(json!({"basis": "recorded"})).unwrap();
        assert_eq!(recorded["selectedPaths"].as_array().unwrap().len(), 0);
        let measured = plan(json!({"basis": "measured", "degrees": 3.5})).unwrap();
        assert_eq!(measured["selectedPaths"][0], "flat.fits");
        let unknown = plan(json!({"basis": "unknown"})).unwrap();
        assert_eq!(unknown["selectedPaths"][0], "flat.fits");
        assert!(plan(json!({"basis": "measured"})).is_err());
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

    fn build_master_report(request: Value) -> Result<Value, String> {
        let request = CString::new(request.to_string()).unwrap();
        let mut error = ptr::null_mut();
        let result = unsafe {
            seiza_calibration_build_master_json(request.as_ptr(), ptr::null(), &mut error)
        };
        if result.is_null() {
            let message = unsafe { CStr::from_ptr(error) }
                .to_str()
                .unwrap()
                .to_owned();
            unsafe { seiza_string_free(error) };
            return Err(message);
        }
        let report = serde_json::from_str(unsafe { CStr::from_ptr(result) }.to_str().unwrap());
        unsafe { seiza_string_free(result) };
        Ok(report.unwrap())
    }

    #[test]
    fn master_builder_masks_stars_in_sky_flats_in_a_chosen_scratch_directory() {
        let directory = tempfile::tempdir().unwrap();
        let scratch = tempfile::tempdir().unwrap();
        let (width, height) = (96, 80);
        let paths = (0..6)
            .map(|frame| {
                let path = directory.path().join(format!("sky-flat-{frame}.fits"));
                let illumination = 1000.0 + frame as f32 * 300.0;
                let (star_x, star_y) = (20.0 + frame as f32 * 11.0, 40.0);
                let data = (0..width * height)
                    .map(|index| {
                        let (x, y) = ((index % width) as f32, (index / width) as f32);
                        let grain = ((index * 37 + 11) % 17) as f32 * 0.002 - 0.016;
                        let radius = (x - star_x).powi(2) + (y - star_y).powi(2);
                        let star = 8.0 * (-radius / 8.0).exp();
                        illumination * (1.0 + grain + star) + 200.0
                    })
                    .collect();
                seiza_stacking::write_processed_image_fits_f32(
                    &path,
                    &LinearImage::new(width, height, 1, data).unwrap(),
                    &[],
                    &[seiza_fits::WriteHeaderCard::new(
                        "IMAGETYP",
                        HeaderValue::String("FLAT".into()),
                    )],
                )
                .unwrap();
                path
            })
            .collect::<Vec<_>>();
        let output = directory.path().join("master-flat.fits");
        let report = build_master_report(json!({
            "kind": "flat",
            "inputs": &paths,
            "output": &output,
            "flatStarMasking": {"detectionSigma": 5.0},
            "scratchDirectory": scratch.path(),
        }))
        .unwrap();
        assert_eq!(report["rejectionMethod"], "MEDIAN_MAD");
        let masking = &report["flatStarMasking"];
        assert_eq!(masking["options"]["detectionSigma"], 5.0);
        assert_eq!(masking["options"]["minimumCleanSamples"], 2);
        assert_eq!(masking["maskedSamples"], report["maskedSamples"]);
        let masked = report["maskedSamples"].as_u64().unwrap();
        assert!(masked > 0, "{report}");
        let per_input: u64 = report["inputs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|input| input["maskedSamples"].as_u64().unwrap())
            .sum();
        assert_eq!(per_input, masked);
        assert_eq!(
            report["acceptedSamples"].as_u64().unwrap()
                + report["rejectedSamples"].as_u64().unwrap()
                + masked,
            (width * height * paths.len()) as u64
        );
        // The build cleans up its scratch file.
        assert_eq!(std::fs::read_dir(scratch.path()).unwrap().count(), 0);

        // The scratch directory reaches the core: one that does not exist fails.
        let missing = directory.path().join("no-such-directory");
        assert!(
            build_master_report(json!({
                "kind": "flat",
                "inputs": &paths,
                "output": directory.path().join("unwritten.fits"),
                "scratchDirectory": missing,
            }))
            .is_err()
        );
        // Without masking, the response says so.
        let plain = build_master_report(json!({
            "kind": "flat",
            "inputs": &paths,
            "output": directory.path().join("plain-flat.fits"),
        }))
        .unwrap();
        assert_eq!(plain["flatStarMasking"], Value::Null);
        assert_eq!(plain["maskedSamples"], 0);
        // Masking is for flats only; the core refuses it on a bias.
        let error = build_master_report(json!({
            "kind": "bias",
            "inputs": &paths,
            "output": directory.path().join("bias.fits"),
            "flatStarMasking": {},
        }))
        .unwrap_err();
        assert!(error.contains("star masking"), "{error}");
    }

    #[test]
    fn master_builder_screens_darks_that_caught_stray_light() {
        let directory = tempfile::tempdir().unwrap();
        let leaks = [0.0_f32, 0.5, 0.0, 40.0, 400.0];
        let paths = leaks
            .iter()
            .enumerate()
            .map(|(index, &leak)| {
                let path = directory.path().join(format!("dark-{index}.fits"));
                let data = (0..32 * 32)
                    .map(|pixel| {
                        let (x, y) = ((pixel % 32) as f32 - 15.5, (pixel / 32) as f32 - 15.5);
                        let noise = ((pixel * 37 + 11) % 17) as f32 * 0.6 - 4.8;
                        let falloff = 1.0 - 0.06 * (x * x + y * y) / (2.0 * 15.5 * 15.5);
                        503.0 + noise + leak * falloff
                    })
                    .collect();
                seiza_stacking::write_processed_image_fits_f32(
                    &path,
                    &LinearImage::new(32, 32, 1, data).unwrap(),
                    &[],
                    &[],
                )
                .unwrap();
                path
            })
            .collect::<Vec<_>>();
        let report = build_master_report(json!({
            "kind": "dark",
            "inputs": &paths,
            "output": directory.path().join("master-dark.fits"),
            "darkLevelScreening": {"maxExcessSigma": 1.0},
        }))
        .unwrap();
        assert_eq!(report["inputFrames"], 3);
        assert_eq!(report["rejectionMethod"], "LEAVE_ONE_OUT");
        let skipped = report["skippedInputs"].as_array().unwrap();
        assert_eq!(
            skipped
                .iter()
                .map(|entry| entry["path"].as_str().unwrap())
                .collect::<Vec<_>>(),
            [paths[3].to_str().unwrap(), paths[4].to_str().unwrap()]
        );
        assert!(
            skipped[0]["reason"]
                .as_str()
                .unwrap()
                .starts_with("stray light")
        );
        // `{}` takes the default limit; screening is for darks only.
        let error = build_master_report(json!({
            "kind": "bias",
            "inputs": &paths,
            "output": directory.path().join("bias.fits"),
            "darkLevelScreening": {},
        }))
        .unwrap_err();
        assert!(error.contains("level screening"), "{error}");
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
        let params = |options: StarDetectOptions| {
            options
                .measurement()
                .unwrap()
                .with_frame_optics(
                    seiza_image_io::header_focal_length_mm(&image.headers),
                    seiza_image_io::header_pixel_size_um(&image.headers),
                )
                .params()
        };

        // The usable aliases are case-insensitive, as FITS-derived XISF
        // metadata is not required to preserve FITS keyword casing.
        let from_headers = params(StarDetectOptions::default());
        assert_eq!(from_headers.structure_layers, 5);
        assert_eq!(from_headers.noise_reduction_radius, 0);

        // Caller pixel size wins while the missing focal length still comes
        // from the frame, yielding the wide-field class.
        let caller_pixel = params(StarDetectOptions {
            pixel_size_um: Some(12.0),
            ..Default::default()
        });
        assert_eq!(caller_pixel.structure_layers, 3);

        // Caller focal length also wins independently while pixel size comes
        // from the frame.
        let caller_focal = params(StarDetectOptions {
            focal_length_mm: Some(500.0),
            ..Default::default()
        });
        assert_eq!(caller_focal.structure_layers, 3);

        // An explicit preset is complete and all explicit fine tuning still
        // lands after it.
        let explicit = params(StarDetectOptions {
            preset: Some("standard".into()),
            focal_length_mm: Some(50.0),
            pixel_size_um: Some(50.0),
            structure_removal: Some("atrous".into()),
            detection_binning: Some(3),
            sensitivity: Some(7.5),
            ..Default::default()
        });
        assert_eq!(explicit.structure_layers, 4);
        assert_eq!(explicit.structure_removal, StructureRemovalMethod::Atrous);
        assert_eq!(explicit.detection_binning, 3);
        assert_eq!(explicit.sensitivity, 7.5);

        let luma = seiza_image_io::luma_u16(&image);
        assert!(matches!(luma, Cow::Borrowed(_)), "mono u16 should not copy");
        assert_eq!(luma.as_ref(), [1, 2, 3, 4]);

        let planar_rgb = FitsImage {
            width: 2,
            height: 1,
            planes: 3,
            pixels: seiza_fits::Pixels::U16(vec![10, 40, 20, 50, 30, 60]),
            headers: vec![],
        };
        assert_eq!(seiza_image_io::luma_u16(&planar_rgb).as_ref(), [20, 50]);

        let bayer = FitsImage {
            width: 2,
            height: 2,
            planes: 1,
            pixels: seiza_fits::Pixels::U16(vec![10_000, 20_000, 30_000, 40_000]),
            headers: vec![("BAYERPAT".into(), HeaderValue::String("RGGB".into()))],
        };
        let expected = bayer.debayer().unwrap().to_luma_u16();
        assert_eq!(seiza_image_io::luma_u16(&bayer).as_ref(), expected);
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
    fn a_stacker_validates_a_light_under_the_callers_tolerances() {
        // A flat from 3.2 degrees and a light at 113.8: refused under the
        // default two-degree gate, with the gate named; accepted when the
        // host asks with the gate it chose the flat under. The yes-or-no
        // answer and the kept/dropped answer agree either way.
        unsafe fn take_error(error: *mut c_char) -> String {
            if error.is_null() {
                return String::new();
            }
            let message = unsafe { CStr::from_ptr(error) }
                .to_str()
                .unwrap()
                .to_owned();
            unsafe { seiza_string_free(error) };
            message
        }
        let (width, height) = (160, 128);
        let directory = tempfile::tempdir().unwrap();
        let reference_path = directory.path().join("reference.fits");
        let flat_path = directory.path().join("master-flat.fits");
        let rotation =
            |degrees| seiza_fits::WriteHeaderCard::new("ROTATANG", HeaderValue::Float(degrees));
        seiza_stacking::write_processed_image_fits_f32(
            &reference_path,
            &LinearImage::new(width, height, 1, stacking_star_field(width, height)).unwrap(),
            &[],
            &[
                seiza_fits::WriteHeaderCard::new("IMAGETYP", HeaderValue::String("LIGHT".into())),
                rotation(3.2),
            ],
        )
        .unwrap();
        seiza_stacking::write_processed_image_fits_f32(
            &flat_path,
            &LinearImage::new(width, height, 1, vec![1.0; width * height]).unwrap(),
            &[],
            &[
                seiza_fits::WriteHeaderCard::new("SEIZAMST", HeaderValue::String("FLAT".into())),
                rotation(3.2),
            ],
        )
        .unwrap();
        let reference_c = CString::new(reference_path.to_str().unwrap()).unwrap();
        let flat_c = CString::new(flat_path.to_str().unwrap()).unwrap();
        let config = no_adjustment_stack_options();
        let mut error = ptr::null_mut();
        let stacker = unsafe {
            seiza_live_stacker_open_fits(
                reference_c.as_ptr(),
                ptr::null(),
                ptr::null(),
                flat_c.as_ptr(),
                0.0,
                config.as_ptr(),
                &mut error,
            )
        };
        assert!(!stacker.is_null(), "{}", unsafe { take_error(error) });

        let mut light: SeizaFrameSignature = unsafe { std::mem::zeroed() };
        light.width = width as f64;
        light.height = height as f64;
        light.channels = 1.0;
        light.rotation_deg = 113.8;
        light.known = SEIZA_FRAME_HAS_WIDTH
            | SEIZA_FRAME_HAS_HEIGHT
            | SEIZA_FRAME_HAS_CHANNELS
            | SEIZA_FRAME_HAS_ROTATION;
        let wide = SeizaMatchTolerances {
            known: SEIZA_TOLERANCE_HAS_ROTATION,
            rotation_deg: 180.0,
            ..SeizaMatchTolerances::default()
        };
        let dropped = |tolerances: *const SeizaMatchTolerances| {
            let mut error = ptr::null_mut();
            let json = unsafe {
                seiza_live_stacker_compatible_calibration_json(
                    stacker, &light, tolerances, &mut error,
                )
            };
            assert!(!json.is_null());
            let parsed: Value =
                serde_json::from_str(unsafe { CStr::from_ptr(json) }.to_str().unwrap()).unwrap();
            unsafe { seiza_string_free(json) };
            parsed["dropped"].as_array().unwrap().len()
        };

        // The stack's own tolerances, which are the defaults.
        let validated = unsafe {
            seiza_live_stacker_validate_calibration(stacker, &light, ptr::null(), &mut error)
        };
        assert_eq!(validated, 0);
        let reason = unsafe { take_error(error) };
        assert!(reason.contains("113.8"), "{reason}");
        assert!(reason.contains("tolerance 2.00"), "{reason}");
        assert_eq!(dropped(ptr::null()), 1);

        let mut error = ptr::null_mut();
        let validated =
            unsafe { seiza_live_stacker_validate_calibration(stacker, &light, &wide, &mut error) };
        assert_eq!(validated, 1);
        assert!(error.is_null());
        assert_eq!(dropped(&wide), 0);

        // An invalid argument is -1, never mistaken for a refusal.
        let validated = unsafe {
            seiza_live_stacker_validate_calibration(ptr::null(), &light, ptr::null(), &mut error)
        };
        assert_eq!(validated, -1);
        assert!(unsafe { take_error(error) }.contains("live stacker is required"));

        // The stack's own gate: set, read back, and used wherever a host
        // gives none.
        let mut error = ptr::null_mut();
        assert!(unsafe { seiza_live_stacker_set_match_tolerances(stacker, &wide, &mut error) });
        let mut own = SeizaMatchTolerances::default();
        assert!(unsafe { seiza_live_stacker_match_tolerances(stacker, &mut own) });
        assert_eq!(own.rotation_deg, 180.0);
        assert_eq!(
            own.known,
            SeizaMatchTolerances::from(seiza_calibration::MatchTolerances::default()).known
        );
        assert_eq!(
            unsafe {
                seiza_live_stacker_validate_calibration(stacker, &light, ptr::null(), &mut error)
            },
            1
        );
        assert_eq!(dropped(ptr::null()), 0);
        // Null puts the defaults back.
        assert!(unsafe {
            seiza_live_stacker_set_match_tolerances(stacker, ptr::null(), &mut error)
        });
        assert_eq!(dropped(ptr::null()), 1);
        unsafe { seiza_live_stacker_free(stacker) };

        // A reference is judged under the gate in the options it opens
        // with.
        let turned_path = directory.path().join("turned.fits");
        seiza_stacking::write_processed_image_fits_f32(
            &turned_path,
            &LinearImage::new(width, height, 1, stacking_star_field(width, height)).unwrap(),
            &[],
            &[
                seiza_fits::WriteHeaderCard::new("IMAGETYP", HeaderValue::String("LIGHT".into())),
                rotation(113.8),
            ],
        )
        .unwrap();
        let turned_c = CString::new(turned_path.to_str().unwrap()).unwrap();
        let open = |options: &CString| {
            let mut error = ptr::null_mut();
            let stacker = unsafe {
                seiza_live_stacker_open_fits(
                    turned_c.as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    flat_c.as_ptr(),
                    0.0,
                    options.as_ptr(),
                    &mut error,
                )
            };
            (stacker, unsafe { take_error(error) })
        };
        let (refused, reason) = open(&config);
        assert!(refused.is_null());
        assert!(reason.contains("tolerance 2.00"), "{reason}");
        let gated = CString::new(
            r#"{
                "normalization": {"mode": "none"},
                "rejection": {"mode": "none"},
                "match_tolerances": {"rotation_deg": 180.0}
            }"#,
        )
        .unwrap();
        let (stacker, reason) = open(&gated);
        assert!(!stacker.is_null(), "{reason}");
        let mut own = SeizaMatchTolerances::default();
        assert!(unsafe { seiza_live_stacker_match_tolerances(stacker, &mut own) });
        assert_eq!(own.rotation_deg, 180.0);
        assert_eq!(own.dark_temperature_c, 3.0);
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
    fn phone_jpegs_render_upright_in_8_and_16_bits() {
        use seiza_image_io::raster::test_support::{Tag, Value, field, jpeg_with_exif};
        let directory = tempfile::tempdir().unwrap();
        // Stored 40×30; orientation 6 means the viewer turns it to 30×40.
        let path = directory.path().join("phone.jpg");
        std::fs::write(
            &path,
            jpeg_with_exif(
                &image::DynamicImage::new_rgb8(40, 30),
                &[field(Tag::Orientation, Value::Short(vec![6]))],
            ),
        )
        .unwrap();
        let image = render_path(&path, &StretchParams::default(), 0, RgbStretchMode::Auto).unwrap();
        assert_eq!((image.width, image.height), (30, 40));
        let image16 =
            render_path16(&path, &StretchParams::default(), 0, RgbStretchMode::Auto).unwrap();
        assert_eq!((image16.width, image16.height), (30, 40));
        let raster = open_raster(&path).unwrap();
        assert_eq!((raster.width(), raster.height()), (30, 40));
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
    fn renders_the_layered_stretch_with_the_image_shape() {
        let (width, height) = (48, 32);
        let pixels = (0..width * height)
            .map(|index| {
                if index % 97 == 0 {
                    0.9
                } else {
                    0.05 + (index * 7919 % 101) as f32 * 0.0002
                }
            })
            .collect::<Vec<_>>();
        let fits = FitsImage {
            width,
            height,
            planes: 1,
            pixels: seiza_fits::Pixels::F32(pixels),
            headers: Vec::new(),
        };
        let request: StretchConfigRequest = serde_json::from_value(json!({
            "model": {
                "type": "layered",
                "target_background": 0.2,
                "aggressiveness": 0.5,
                "contrast_recovery": 0.5,
                "dynamic_range_compression": 0.5,
                "scale_separation": 16
            },
            "color_strategy": "linked",
            "max_analysis_samples": 4096
        }))
        .unwrap();
        let stack = request.into_stack();
        let options = || RenderPipelineOptions {
            background: None,
            deconvolution: None,
            sample_domain: &SampleDomain::UnitLinear,
            max_dimension: 0,
            interactive_preview: false,
        };
        let image8 = render_astronomy_with_pipeline(
            fits.clone(),
            AstronomyImageFormat::Fits,
            &stack,
            options(),
        )
        .unwrap();
        let image16 =
            render_astronomy_with_pipeline16(fits, AstronomyImageFormat::Fits, &stack, options())
                .unwrap();
        assert_eq!((image8.width, image8.height), (48, 32));
        assert_eq!(image16.rgba.len(), image8.rgba.len());
        // The stars stand out from the background in both renders.
        assert!(image8.rgba[0] > image8.rgba[4]);
        assert!(image16.rgba[0] > image16.rgba[4]);
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
        let rgba = vec![255_u8; 400 * 200 * 4];
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

        assert_eq!(
            CatalogSetupPreset::from_raw(3).unwrap().datasets(),
            [Dataset::GaiaPhotometry]
        );
        assert!(
            CatalogSetupPreset::All
                .datasets()
                .contains(&Dataset::GaiaPhotometry)
        );
        assert!(
            CatalogSetupPreset::All
                .datasets()
                .contains(&Dataset::ObjectDistances)
        );
        assert!(
            CatalogSetupPreset::All
                .datasets()
                .contains(&Dataset::StarDistances)
        );

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
    fn object_outlines_carry_how_they_were_made() {
        use seiza::field_objects::FieldContour;
        use seiza::objects::{GeometryQuality, GeometryRole};

        let outline = ObjectOutlineResponse::from(FieldOutline {
            geometry_id: "openngc:NGC1#outline-1".into(),
            source_record_id: "openngc:NGC1".into(),
            role: GeometryRole::BrightnessLevel,
            quality: GeometryQuality::Catalog,
            method: "OpenNGC outline".into(),
            evidence: "survey isophote".into(),
            level: Some("1".into()),
            contours: vec![FieldContour {
                closed: true,
                points: vec![(30.0, 40.0), (70.0, 40.0), (50.0, 80.0)],
            }],
        });
        let json = serde_json::to_value(&outline).unwrap();
        assert_eq!(json["geometryId"], "openngc:NGC1#outline-1");
        assert_eq!(json["role"], "brightness-level");
        assert_eq!(json["quality"], "catalog");
        assert_eq!(json["method"], "OpenNGC outline");
        assert_eq!(json["evidence"], "survey isophote");
        assert_eq!(json["level"], "1");
        assert_eq!(json["contours"][0]["closed"], true);
        assert_eq!(json["contours"][0]["points"][2], json!([50.0, 80.0]));
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
    fn reads_the_capture_time_a_solve_dates_objects_by() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("dated.fits");
        std::fs::write(&path, synthetic_fits()).unwrap();
        let fits = FitsImage::open(&path).unwrap();

        assert_eq!(
            seiza_image_io::capture_time_from_headers(&fits.headers).as_deref(),
            Some("2025-07-20T12:34:56.5Z")
        );
        let c_path = CString::new(path.to_str().unwrap()).unwrap();
        let mut error = ptr::null_mut();
        let json = unsafe { seiza_read_capture_time_json(c_path.as_ptr(), &mut error) };
        assert!(error.is_null());
        let value: Value =
            serde_json::from_str(unsafe { CStr::from_ptr(json) }.to_str().unwrap()).unwrap();
        unsafe { seiza_string_free(json) };
        assert_eq!(value["captureTime"], "2025-07-20T12:34:56.5Z");

        let missing =
            CString::new(directory.path().join("missing.fits").to_str().unwrap()).unwrap();
        let json = unsafe { seiza_read_capture_time_json(missing.as_ptr(), &mut error) };
        assert!(json.is_null());
        assert!(!error.is_null());
        unsafe { seiza_string_free(error) };
    }

    #[test]
    fn object_points_keep_their_names_and_dates() {
        let point = ObjectPointResponse::from(FieldObject {
            stable_id: Some("test:SN 2020abc".into()),
            name: "SN 2020abc".into(),
            common_name: "disc. 2020/01/01".into(),
            kind: "transient",
            source: "transient",
            catalog_source: Some("test-catalog".into()),
            x: 50.0,
            y: 49.0,
            semi_major_px: 0.0,
            semi_minor_px: 0.0,
            angle_deg: None,
            prominence: Some(0.5),
            ra_deg: 10.0,
            dec_deg: 20.0,
            discovered: Some("2020-01-01".into()),
            near_capture: Some(false),
            distance_au: None,
            motion_arcsec_per_hour: None,
            direction_position_angle_deg: None,
            direction_image_angle_deg: None,
            outlines: Vec::new(),
        });
        let json = serde_json::to_value(&point).unwrap();
        assert_eq!(json["stableId"], "test:SN 2020abc");
        assert_eq!(json["kind"], "transient");
        assert_eq!(json["catalogSource"], "test-catalog");
        assert_eq!(json["raDegrees"], 10.0);
        assert_eq!(json["discovered"], "2020-01-01");
        assert_eq!(json["nearCapture"], false);
        assert_eq!(json["outlines"], json!([]));
    }

    /// What a C call returned: its JSON, or its error, freed either way.
    fn returned_json(json: *mut c_char, error: *mut c_char) -> Result<Value, String> {
        let take = |text: *mut c_char| {
            let owned = unsafe { CStr::from_ptr(text) }
                .to_string_lossy()
                .into_owned();
            unsafe { seiza_string_free(text) };
            owned
        };
        if json.is_null() {
            assert!(!error.is_null());
            return Err(take(error));
        }
        assert!(error.is_null());
        Ok(serde_json::from_str(&take(json)).unwrap())
    }

    #[test]
    fn a_wcs_becomes_fits_header_cards() {
        let wcs =
            seiza::Wcs::from_center_scale_rotation((150.0, 35.0), (99.0, 49.0), 2.0, 15.0, false);
        let header = |request: Value| {
            let text = CString::new(request.to_string()).unwrap();
            let mut error = ptr::null_mut();
            let json = unsafe { seiza_wcs_fits_header_json(text.as_ptr(), &mut error) };
            returned_json(json, error)
        };
        let from_zero = header(json!({
            "crval": [wcs.crval.0, wcs.crval.1],
            "crpix": [99.0, 49.0],
            "cd": wcs.cd,
        }))
        .unwrap();
        assert_eq!(from_zero["cards"]["CTYPE1"], "RA---TAN");
        assert_eq!(from_zero["cards"]["CRPIX1"], 100.0);
        assert_eq!(from_zero["cards"]["CRPIX2"], 50.0);
        assert_eq!(from_zero["cards"]["CD1_2"], wcs.cd[0][1]);
        assert_eq!(from_zero["text"], wcs.fits_header_text());

        let from_one = header(json!({
            "crval": [wcs.crval.0, wcs.crval.1],
            "crpix": [100.0, 50.0],
            "cd": wcs.cd,
            "pixelOrigin": 1,
        }))
        .unwrap();
        assert_eq!(from_one, from_zero);

        let message = header(json!({"crval": [1.0, 2.0]})).unwrap_err();
        assert!(message.contains("invalid WCS"), "{message}");
    }

    #[test]
    fn a_photo_gives_its_exif_and_scale_ranges() {
        use seiza_image_io::raster::test_support::{
            Tag, Value as ExifValue, ascii, field, jpeg_with_exif,
        };
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("phone.jpg");
        std::fs::write(
            &path,
            jpeg_with_exif(
                &image::DynamicImage::new_rgb8(40, 30),
                &[
                    ascii(Tag::Make, "Seiza"),
                    field(Tag::FocalLengthIn35mmFilm, ExifValue::Short(vec![24])),
                ],
            ),
        )
        .unwrap();
        let metadata = |width: u32, height: u32| {
            let path = CString::new(path.to_str().unwrap()).unwrap();
            let mut error = ptr::null_mut();
            let json =
                unsafe { seiza_read_photo_metadata_json(path.as_ptr(), width, height, &mut error) };
            returned_json(json, error).unwrap()
        };

        let alone = metadata(0, 0);
        assert_eq!(alone["make"], "Seiza");
        assert_eq!(alone["focalLength35mm"], 24.0);
        assert_eq!(alone["captureTimeUtc"], Value::Null);
        assert_eq!(alone["scaleHint"], Value::Null);
        assert_eq!(alone["scaleRanges"], Value::Null);

        let sized = metadata(4000, 3000);
        let nominal = sized["scaleHint"]["nominalArcsecPerPixel"]
            .as_f64()
            .unwrap();
        assert!(nominal > 10.0 && nominal < 100.0, "{nominal}");
        assert_eq!(
            sized["scaleRanges"][0][0].as_f64().unwrap(),
            sized["scaleHint"]["minArcsecPerPixel"].as_f64().unwrap()
        );
    }

    #[test]
    fn the_local_gaia_catalog_answers_a_cone() {
        use seiza::catalog::{PhotometricStar, PhotometryCatalogBuilder};
        let directory = tempfile::tempdir().unwrap();
        let mut builder = PhotometryCatalogBuilder::new(90, 2016.0, "test");
        for (ra, g, bp_rp, reliable) in [
            (10.0, 8.0, Some(0.8), true),
            (10.01, 9.0, Some(1.2), false),
            (10.02, 16.0, Some(0.5), true),
            (40.0, 8.0, Some(0.8), true),
        ] {
            builder.add(PhotometricStar {
                ra,
                dec: 20.0,
                g,
                bp_rp,
                reliable,
            });
        }
        builder
            .write_to(&directory.path().join("stars-gaia-photometry.bin"))
            .unwrap();

        let cone = |path: &Path| {
            let path = CString::new(path.to_str().unwrap()).unwrap();
            let mut error = ptr::null_mut();
            let json = unsafe {
                seiza_gaia_photometry_catalog_cone_json(
                    path.as_ptr(),
                    10.0,
                    20.0,
                    0.5,
                    15.0,
                    &mut error,
                )
            };
            returned_json(json, error)
        };
        let stars = cone(directory.path()).unwrap();
        let stars = stars.as_array().unwrap();
        assert_eq!(stars.len(), 2);
        let bright = stars.iter().find(|star| star["g"] == 8.0).unwrap();
        assert!((bright["bpRp"].as_f64().unwrap() - 0.8).abs() < 0.01);
        assert_eq!(bright["pmra"], Value::Null);
        let unreliable = stars.iter().find(|star| star["g"] == 9.0).unwrap();
        assert!(unreliable.get("bpRp").is_none());

        // The sources read back as colour calibration's input.
        let sources: Vec<GaiaSourceJson> = serde_json::from_value(json!(stars)).unwrap();
        assert!(sources.iter().any(|source| source.bp_rp.is_some()));

        assert!(cone(&directory.path().join("missing")).is_err());
    }

    #[test]
    fn pixel_scale_advice_follows_the_core() {
        let scale = seiza_stars_pixel_scale_arcsec(518.0, 3.76);
        assert!((scale - 1.497).abs() < 0.01, "{scale}");
        assert!(seiza_stars_pixel_scale_arcsec(0.0, 3.76).is_nan());
        let class = |scale: f64| {
            unsafe { CStr::from_ptr(seiza_stars_classify_pixel_scale(scale)) }
                .to_str()
                .unwrap()
        };
        assert_eq!(class(1.5), "widefield");
        assert_eq!(class(1.0), "standard");
        assert_eq!(class(0.36), "longfocal");
        assert_eq!(class(f64::NAN), "standard");
        assert_eq!(seiza_stars_recommend_detection_binning(6.2), 2);
        assert_eq!(seiza_stars_recommend_detection_binning(-1.0), 1);
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
/// [`seiza_probe_frame_json`]; extra probe fields are ignored. To plan a
/// light's flats by an angle the host measured for it, give
/// `tolerances.lightRotation` as `{"basis": "measured", "degrees": 82.2}`,
/// or `{"basis": "unknown"}` when its angle does not decide its flats; the
/// flats keep their recorded angles.
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
/// `exposureSeconds`, `rejection` (`lowSigma`, `highSigma`),
/// `defectSuppression` (`lowSigma`, `highSigma`), `flatStarMasking`,
/// `darkLevelScreening` and `scratchDirectory`. `dark` is a dark-flat when
/// building a flat. Defect suppression is accepted only for flats, because a
/// dark master must retain the hot pixels it is meant to subtract.
///
/// `flatStarMasking` (flats only) leaves stars and broad saturated patches
/// out of each sky flat before it is normalized and combined. Its fields,
/// each optional, are `detectionSigma` (default 4), `haloRadiusFactor`
/// (default 3), `minimumRadiusPixels` (default 6), `saturationLevel` (in the
/// inputs' raw units; default from `SATURATE`, `SATLEVEL` or the FITS
/// encoding) and `minimumCleanSamples` (the fewest unmasked inputs every
/// pixel must keep; default 2); `{}` takes every default.
/// `darkLevelScreening` (darks only) sets aside darks whose level sits more
/// than `maxExcessSigma` (default 1) times the quietest dark's pixel noise
/// above it, the mark of stray light; they appear in `skippedInputs`.
/// `scratchDirectory`, an existing directory, holds a flat build's temporary
/// pixels in place of the system's temporary directory; bias and dark
/// builds need none.
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
/// accepted-only per-frame `inputs` (`path`, `acceptedSamples`,
/// `rejectedSamples`, `maskedSamples`), plus `skippedInputs` entries with
/// `path` and `reason` for frames the integrator set aside. The accepted and
/// skipped paths are disjoint and account for every requested input in
/// request order. `maskedSamples` counts the samples star masking left out,
/// `rejectionMethod` is the rejection the build used (`NONE`, `MEDIAN_MAD`
/// or `LEAVE_ONE_OUT`), and `flatStarMasking` is null unless masking ran;
/// then it holds the `options` used, `maskedSamples`, `maskedPixels`, the
/// `minimumCleanSamples` and `maximumCleanSamples` any pixel kept,
/// `lowCoverageSamples`, `unmaskedSaturationSamples` and
/// `unknownSaturationInputs`.
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
/// means "all defaults", and so does a null pointer, except where a live
/// stacker call says null means the stack's own gate. An override that is
/// negative, not a number, infinite, or a rotation above 180 degrees is
/// ignored in favour of the default. Unset fields always take the built-in
/// defaults, never the stack's own gate: a non-null struct replaces it
/// whole.
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
    unsafe {
        *tolerances = SeizaMatchTolerances::from(seiza_calibration::MatchTolerances::default());
    }
}

impl From<seiza_calibration::MatchTolerances> for SeizaMatchTolerances {
    /// Every tolerance, with every flag set. How the light's angle is read
    /// (`light_rotation`) has no field here and is dropped: the result
    /// reads the recorded angle.
    fn from(tolerances: seiza_calibration::MatchTolerances) -> Self {
        Self {
            known: SEIZA_TOLERANCE_HAS_EXPOSURE
                | SEIZA_TOLERANCE_HAS_EXPOSURE_FRACTION
                | SEIZA_TOLERANCE_HAS_DARK_TEMPERATURE
                | SEIZA_TOLERANCE_HAS_MASTER_TEMPERATURE
                | SEIZA_TOLERANCE_HAS_ROTATION
                | SEIZA_TOLERANCE_HAS_FOCAL_LENGTH
                | SEIZA_TOLERANCE_HAS_FLAT_SESSION,
            exposure_seconds: tolerances.exposure_seconds,
            exposure_fraction: tolerances.exposure_fraction,
            dark_temperature_c: tolerances.dark_temperature_c,
            master_temperature_c: tolerances.master_temperature_c,
            rotation_deg: tolerances.rotation_deg,
            focal_length_mm: tolerances.focal_length_mm,
            flat_session_seconds: tolerances.flat_session_seconds,
        }
    }
}

/// A null pointer, a zeroed struct, or any cleared flag all fall back to the
/// built-in default for that tolerance. So does an override
/// `MatchTolerances::validate` refuses — not a number, infinite, negative, or
/// a rotation past 180 degrees: taking it literally would quietly match
/// nothing, or leave a stack context that cannot be reopened. Rust refuses
/// these with an error; C keeps the fallback it has always had.
unsafe fn match_tolerances(
    tolerances: *const SeizaMatchTolerances,
) -> seiza_calibration::MatchTolerances {
    let mut resolved = seiza_calibration::MatchTolerances::default();
    let Some(overrides) = (unsafe { tolerances.as_ref() }) else {
        return resolved;
    };
    // Each override is judged alone against the defaults, by the same rule
    // Rust callers are held to, so one bad field does not cost the others.
    type Set = fn(&mut seiza_calibration::MatchTolerances, f64);
    let fields: [(u32, f64, Set); 6] = [
        (
            SEIZA_TOLERANCE_HAS_EXPOSURE,
            overrides.exposure_seconds,
            |tolerances, value| tolerances.exposure_seconds = value,
        ),
        (
            SEIZA_TOLERANCE_HAS_EXPOSURE_FRACTION,
            overrides.exposure_fraction,
            |tolerances, value| tolerances.exposure_fraction = value,
        ),
        (
            SEIZA_TOLERANCE_HAS_DARK_TEMPERATURE,
            overrides.dark_temperature_c,
            |tolerances, value| tolerances.dark_temperature_c = value,
        ),
        (
            SEIZA_TOLERANCE_HAS_MASTER_TEMPERATURE,
            overrides.master_temperature_c,
            |tolerances, value| tolerances.master_temperature_c = value,
        ),
        (
            SEIZA_TOLERANCE_HAS_ROTATION,
            overrides.rotation_deg,
            |tolerances, value| tolerances.rotation_deg = value,
        ),
        (
            SEIZA_TOLERANCE_HAS_FOCAL_LENGTH,
            overrides.focal_length_mm,
            |tolerances, value| tolerances.focal_length_mm = value,
        ),
    ];
    for (flag, value, set) in fields {
        let mut candidate = resolved;
        set(&mut candidate, value);
        if overrides.known & flag != 0 && candidate.validate().is_ok() {
            resolved = candidate;
        }
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
        owned_json(&detect_stars_response(
            samples,
            width,
            height,
            options,
            (None, None),
        )?)
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
        let frame_optics = (
            seiza_image_io::header_focal_length_mm(&image.headers),
            seiza_image_io::header_pixel_size_um(&image.headers),
        );
        let samples = seiza_image_io::luma_u16(&image);
        owned_json(&detect_stars_response(
            &samples,
            image.width,
            image.height,
            options,
            frame_optics,
        )?)
    })
    .unwrap_or(ptr::null_mut())
}

/// The pixel scale, in arcseconds per pixel, of a camera with pixels
/// `pixel_size_um` micrometres across (binning included) behind a
/// telescope of focal length `focal_length_mm` millimetres; NaN unless
/// both are finite and positive. [`seiza_stars_classify_pixel_scale`]
/// turns it into a detection preset.
#[unsafe(no_mangle)]
pub extern "C" fn seiza_stars_pixel_scale_arcsec(focal_length_mm: f64, pixel_size_um: f64) -> f64 {
    seiza_stars::pixel_scale_arcsec(focal_length_mm, pixel_size_um).unwrap_or(f64::NAN)
}

/// The star detection preset for a rig of `arcsec_per_pixel`, as the
/// `preset` option of [`seiza_stars_detect_luma_u16_json`] takes it:
/// `widefield` from 1.2"/px up, `longfocal` at 0.6"/px and below, and
/// `standard` between, or for a scale that is not a positive number. The
/// string is static: do not free it.
#[unsafe(no_mangle)]
pub extern "C" fn seiza_stars_classify_pixel_scale(arcsec_per_pixel: f64) -> *const c_char {
    use seiza_stars::TelescopeClass;
    match seiza_stars::classify_pixel_scale(arcsec_per_pixel) {
        TelescopeClass::WideField => c"widefield".as_ptr(),
        TelescopeClass::Standard => c"standard".as_ptr(),
        TelescopeClass::LongFocalLength => c"longfocal".as_ptr(),
    }
}

/// The detection binning, 1 to 4, for the `detectionBinning` option of
/// [`seiza_stars_detect_luma_u16_json`], from an in-focus star's measured
/// HFR in captured pixels: the factor that brings it nearest the
/// detector's calibrated 3 pixels. 1 for an HFR that is not a positive
/// number.
#[unsafe(no_mangle)]
pub extern "C" fn seiza_stars_recommend_detection_binning(measured_hfr_pixels: f64) -> usize {
    seiza_stars::recommend_detection_binning(measured_hfr_pixels)
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

/// Progress callback for [`seiza_rc_astro_process_file_json`]: the fraction
/// complete in `[0, 1]` as the tool reports it, plus the caller's context
/// pointer. Called on the thread that made the call.
pub type SeizaRcAstroProgressCallback = Option<unsafe extern "C" fn(f32, *mut c_void)>;

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
/// set. `progress` (nullable) receives the fraction complete in `[0, 1]`
/// as the tool reports it, on the calling thread, with `context` passed
/// through untouched.
///
/// # Safety
///
/// `request_json` must be a NUL-terminated UTF-8 string. `cancel` must be
/// null or a live [`SeizaCancelSignal`] retained until this call returns.
/// `context` is passed through untouched to `progress`. When non-null,
/// `error_out` must point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_rc_astro_process_file_json(
    request_json: *const c_char,
    cancel: *const SeizaCancelSignal,
    progress: SeizaRcAstroProgressCallback,
    context: *mut c_void,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    // Raw pointers are not UnwindSafe; the callback runs on this thread and
    // the context's lifetime is the caller's promise, so the address is
    // carried as a plain integer across the catch_unwind boundary.
    let context = context as usize;
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
        let mut report = |fraction: f32| {
            if let Some(callback) = progress {
                unsafe { callback(fraction, context as *mut c_void) };
            }
        };
        let run = cli
            .run_on_file(
                &schema,
                &tool_request,
                &request.input,
                &request.output,
                cancellation.as_ref(),
                &mut report,
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
echo '{"event":"progress","done":50.0}'
echo '{"event":"progress","done":100.0}'
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
        unsafe extern "C" fn collect(fraction: f32, context: *mut c_void) {
            let fractions = unsafe { &mut *(context as *mut Vec<f32>) };
            fractions.push(fraction);
        }
        let mut fractions: Vec<f32> = Vec::new();
        let request_c = CString::new(request.to_string()).unwrap();
        let mut error: *mut c_char = ptr::null_mut();
        let response = unsafe {
            seiza_rc_astro_process_file_json(
                request_c.as_ptr(),
                ptr::null(),
                Some(collect),
                (&raw mut fractions).cast(),
                &mut error,
            )
        };
        let response = call_json(response, error).unwrap();
        assert_eq!(fractions, vec![0.5, 1.0]);
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
            seiza_rc_astro_process_file_json(
                request_c.as_ptr(),
                ptr::null(),
                None,
                ptr::null_mut(),
                &mut error,
            )
        };
        let message = call_json(response, error).unwrap_err();
        assert!(
            message.contains("invalid rc-astro request JSON"),
            "{message}"
        );
    }

    /// A camera that records a solar-coloured star at R/G 0.6 and B/G 1.4:
    /// the C API fits gains of 1/0.6 and 1/1.4 from WCS keywords and Gaia
    /// rows, and applies them.
    #[test]
    fn color_calibration_cabi_fits_and_applies_channel_gains() {
        let (width, height) = (1200, 1000);
        let wcs =
            seiza::Wcs::from_center_scale_rotation((56.75, 24.1), (600.0, 500.0), 3.0, 10.0, false);
        let sky = [120.0_f32, 100.0, 90.0];
        let mut data = (0..width * height * 3)
            .map(|index| sky[index % 3] + ((index * 7919) % 13) as f32 * 0.3)
            .collect::<Vec<_>>();
        let mut gaia = Vec::new();
        for index in 0..140 {
            let x = 30.0 + ((index * 7919) % 1140) as f64 + ((index % 7) as f64) * 0.13;
            let y = 30.0 + ((index * 6271) % 940) as f64 + ((index % 5) as f64) * 0.17;
            let bp_rp = 0.1 + ((index * 37) % 160) as f64 / 100.0;
            let g = 9.0 + ((index * 13) % 40) as f64 / 10.0;
            let (ra, dec) = wcs.pixel_to_world(x, y);
            gaia.push(json!({
                "ra": ra, "dec": dec, "g": g, "bp": g + 0.3, "rp": g + 0.3 - bp_rp, "ruwe": 1.0
            }));
            let flux = 2.0e5 * 10f64.powf(-0.4 * (g - 9.0));
            let delta = bp_rp - 0.82;
            let ratios = [
                0.6 * 10f64.powf(0.2 * delta),
                1.0,
                1.4 * 10f64.powf(-0.32 * delta),
            ];
            for py in (y as usize).saturating_sub(10)..(y as usize + 11).min(height) {
                for px in (x as usize).saturating_sub(10)..(x as usize + 11).min(width) {
                    let r2 = (px as f64 - x).powi(2) + (py as f64 - y).powi(2);
                    let profile = (-r2 / 4.5).exp() / (std::f64::consts::PI * 4.5);
                    for (channel, ratio) in ratios.iter().enumerate() {
                        data[(py * width + px) * 3 + channel] += (flux * ratio * profile) as f32;
                    }
                }
            }
        }
        let keywords = wcs
            .fits_header_cards()
            .into_iter()
            .map(|(key, value)| {
                let value = match value {
                    seiza::FitsCardValue::Text(text) => json!(text),
                    seiza::FitsCardValue::Integer(number) => json!(number),
                    seiza::FitsCardValue::Number(number) => json!(number),
                };
                (key, value)
            })
            .collect::<Map<_, _>>();
        let wcs_c = CString::new(Value::Object(keywords).to_string()).unwrap();
        let gaia_c = CString::new(Value::Array(gaia).to_string()).unwrap();
        let mut error = ptr::null_mut();
        let response = unsafe {
            seiza_color_calibrate_json(
                data.as_ptr(),
                data.len(),
                width,
                height,
                wcs_c.as_ptr(),
                gaia_c.as_ptr(),
                ptr::null(),
                &mut error,
            )
        };
        assert!(!response.is_null(), "{:?}", unsafe {
            error.as_ref().map(|e| CStr::from_ptr(e))
        });
        let report: Value =
            serde_json::from_str(unsafe { CStr::from_ptr(response) }.to_str().unwrap()).unwrap();
        unsafe { seiza_string_free(response) };
        assert_eq!(report["schemaVersion"], 1);
        let triple = |key: &str| {
            let values = report[key].as_array().unwrap();
            [0, 1, 2].map(|channel| values[channel].as_f64().unwrap() as f32)
        };
        let (gains, offsets) = (triple("gains"), triple("offsets"));
        assert!((gains[0] - 1.0 / 0.6).abs() < 0.03, "{gains:?}");
        assert!((gains[2] - 1.0 / 1.4).abs() < 0.02, "{gains:?}");
        let measurements = report["measurements"].as_array().unwrap();
        assert_eq!(
            measurements.len() as u64,
            report["starsMeasured"].as_u64().unwrap()
        );
        let star = &measurements[0];
        for key in ["x", "y", "g", "bpRp"] {
            assert!(star[key].is_number(), "{star}");
        }
        assert_eq!(star["flux"].as_array().unwrap().len(), 3);
        assert_eq!(star["used"].as_array().unwrap().len(), 2);

        // The fit's own options reach the core.
        let calibrate = |options: &str, error: &mut *mut c_char| {
            let options = CString::new(options).unwrap();
            let response = unsafe {
                seiza_color_calibrate_json(
                    data.as_ptr(),
                    data.len(),
                    width,
                    height,
                    wcs_c.as_ptr(),
                    gaia_c.as_ptr(),
                    options.as_ptr(),
                    error,
                )
            };
            (!response.is_null()).then(|| {
                let report: Value =
                    serde_json::from_str(unsafe { CStr::from_ptr(response) }.to_str().unwrap())
                        .unwrap();
                unsafe { seiza_string_free(response) };
                report
            })
        };
        assert!(calibrate(r#"{"minimumStars": 500}"#, &mut error).is_none());
        unsafe { seiza_string_free(error) };
        error = ptr::null_mut();
        let narrow = calibrate(
            r#"{"fitBpRpRange": [0.3, 1.4], "rejectionSigma": 2.5, "minimumStars": 5, "calibratorMaxG": 10.0}"#,
            &mut error,
        )
        .expect("narrowed fit");
        let narrow_stars = narrow["measurements"].as_array().unwrap();
        assert!(!narrow_stars.is_empty());
        assert!(narrow_stars.len() < measurements.len());
        assert!(
            narrow_stars
                .iter()
                .all(|star| star["g"].as_f64().unwrap() <= 10.0)
        );
        assert!(narrow_stars.iter().all(|star| {
            let bp_rp = star["bpRp"].as_f64().unwrap();
            let used = star["used"].as_array().unwrap();
            (0.3 - 1e-6..=1.4 + 1e-6).contains(&bp_rp) || used.iter().all(|used| used == false)
        }));

        let before = data[0..3].to_vec();
        assert!(unsafe {
            seiza_color_calibration_apply(
                data.as_mut_ptr(),
                data.len(),
                gains.as_ptr(),
                offsets.as_ptr(),
                &mut error,
            )
        });
        for channel in 0..3 {
            let expected = before[channel].mul_add(gains[channel], offsets[channel]);
            assert_eq!(data[channel], expected);
        }
        // The core refuses a buffer that is not whole RGB pixels.
        assert!(!unsafe {
            seiza_color_calibration_apply(
                data.as_mut_ptr(),
                4,
                gains.as_ptr(),
                offsets.as_ptr(),
                &mut error,
            )
        });
        unsafe { seiza_string_free(error) };
        error = ptr::null_mut();
        // Malformed input is refused rather than guessed at.
        let bad = CString::new("{\"CTYPE1\": \"RA---SIN\"}").unwrap();
        let refused = unsafe {
            seiza_color_calibrate_json(
                data.as_ptr(),
                data.len(),
                width,
                height,
                bad.as_ptr(),
                gaia_c.as_ptr(),
                ptr::null(),
                &mut error,
            )
        };
        assert!(refused.is_null() && !error.is_null());
        unsafe { seiza_string_free(error) };
    }

    /// An image made from a solved frame by a half turn and a crop: the C
    /// API carries the frame's solution onto it, calibrates against Gaia
    /// rows, stretches, and reports why when it cannot calibrate.
    #[test]
    fn color_calibrate_and_stretch_cabi_follows_a_derived_solution() {
        let (width, height) = (1200, 1000);
        let frame =
            seiza::Wcs::from_center_scale_rotation((56.75, 24.1), (649.5, 549.5), 3.0, 10.0, false);
        // Turned half round about the frame's 1300 x 1100 grid, then cropped
        // at (40, 60).
        let (turn_x, turn_y, crop_x, crop_y) = (1299.0, 1099.0, 40.0, 60.0);
        let to_frame = |x: f64, y: f64| (turn_x - (x + crop_x), turn_y - (y + crop_y));
        let sky = [120.0_f32, 100.0, 90.0];
        let mut data = (0..width * height * 3)
            .map(|index| sky[index % 3] + ((index * 7919) % 13) as f32 * 0.3)
            .collect::<Vec<_>>();
        let mut gaia = Vec::new();
        for index in 0..140 {
            let x = 30.0 + ((index * 7919) % 1140) as f64 + ((index % 7) as f64) * 0.13;
            let y = 30.0 + ((index * 6271) % 940) as f64 + ((index % 5) as f64) * 0.17;
            let bp_rp = 0.1 + ((index * 37) % 160) as f64 / 100.0;
            let g = 9.0 + ((index * 13) % 40) as f64 / 10.0;
            let (frame_x, frame_y) = to_frame(x, y);
            let (ra, dec) = frame.pixel_to_world(frame_x, frame_y);
            gaia.push(json!({ "ra": ra, "dec": dec, "g": g, "bpRp": bp_rp }));
            let flux = 2.0e5 * 10f64.powf(-0.4 * (g - 9.0));
            let delta = bp_rp - 0.82;
            let ratios = [
                0.6 * 10f64.powf(0.2 * delta),
                1.0,
                1.4 * 10f64.powf(-0.32 * delta),
            ];
            for py in (y as usize).saturating_sub(10)..(y as usize + 11).min(height) {
                for px in (x as usize).saturating_sub(10)..(x as usize + 11).min(width) {
                    let r2 = (px as f64 - x).powi(2) + (py as f64 - y).powi(2);
                    let profile = (-r2 / 4.5).exp() / (std::f64::consts::PI * 4.5);
                    for (channel, ratio) in ratios.iter().enumerate() {
                        data[(py * width + px) * 3 + channel] += (flux * ratio * profile) as f32;
                    }
                }
            }
        }
        let frame_json = CString::new(
            json!({
                "crval": [frame.crval.0, frame.crval.1],
                "crpix": [frame.crpix.0, frame.crpix.1],
                "cd": frame.cd,
            })
            .to_string(),
        )
        .unwrap();
        let transform = CString::new(
            json!({
                "matrix": [[-1.0, 0.0], [0.0, -1.0]],
                "translationX": turn_x - crop_x,
                "translationY": turn_y - crop_y,
            })
            .to_string(),
        )
        .unwrap();
        let mut error = ptr::null_mut();
        let derived = call_json(
            unsafe {
                seiza_wcs_transform_json(frame_json.as_ptr(), transform.as_ptr(), &mut error)
            },
            error,
        )
        .unwrap();
        assert_eq!(derived["pixelOrigin"], 0);
        let wcs_c = CString::new(derived.to_string()).unwrap();
        let gaia_c = CString::new(Value::Array(gaia).to_string()).unwrap();
        let stretch = CString::new(
            json!({
                "model": {"type": "auto-mtf", "target_median": 0.25, "shadows_clip": -2.8},
                "color_strategy": "linked",
                "max_analysis_samples": 200000,
            })
            .to_string(),
        )
        .unwrap();

        let mut stretched = data.clone();
        let report = call_json(
            unsafe {
                seiza_color_calibrate_and_stretch_json(
                    stretched.as_mut_ptr(),
                    stretched.len(),
                    width,
                    height,
                    wcs_c.as_ptr(),
                    gaia_c.as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    stretch.as_ptr(),
                    &mut error,
                )
            },
            error,
        )
        .unwrap();
        assert_eq!(report["schemaVersion"], 1);
        assert_eq!(report["calibration"]["status"], "calibrated", "{report}");
        let gains = &report["calibration"]["calibration"]["gains"];
        assert!(
            (gains[0].as_f64().unwrap() - 1.0 / 0.6).abs() < 0.03,
            "{gains}"
        );
        assert!(
            (gains[2].as_f64().unwrap() - 1.0 / 1.4).abs() < 0.03,
            "{gains}"
        );
        assert_eq!(report["stretchPlans"].as_array().unwrap().len(), 1);
        assert_eq!(report["linkedStages"], json!([]));
        assert!(stretched.iter().all(|value| (0.0..=1.0).contains(value)));

        // A per-channel stage is linked once calibration has run.
        let unlinked = CString::new(
            json!([{
                "model": {"type": "auto-mtf", "target_median": 0.25, "shadows_clip": -2.8},
                "color_strategy": "unlinked",
                "max_analysis_samples": 200000,
            }])
            .to_string(),
        )
        .unwrap();
        let mut relinked = data.clone();
        let report = call_json(
            unsafe {
                seiza_color_calibrate_and_stretch_json(
                    relinked.as_mut_ptr(),
                    relinked.len(),
                    width,
                    height,
                    wcs_c.as_ptr(),
                    gaia_c.as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    unlinked.as_ptr(),
                    &mut error,
                )
            },
            error,
        )
        .unwrap();
        assert_eq!(report["linkedStages"], json!([0]));
        assert_eq!(relinked, stretched);

        // Calibration alone applies the gains in place.
        let mut calibrated = data.clone();
        let outcome = call_json(
            unsafe {
                seiza_color_calibrate_image_json(
                    calibrated.as_mut_ptr(),
                    calibrated.len(),
                    width,
                    height,
                    wcs_c.as_ptr(),
                    gaia_c.as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    &mut error,
                )
            },
            error,
        )
        .unwrap();
        assert_eq!(outcome["status"], "calibrated");
        assert_ne!(calibrated, data);

        // Without a solution, or without a catalog, it says why and leaves
        // the pixels alone.
        let empty = tempfile::tempdir().unwrap();
        let empty_c = CString::new(empty.path().to_str().unwrap()).unwrap();
        for (wcs, gaia, catalog, reason) in [
            (ptr::null(), gaia_c.as_ptr(), ptr::null(), "no_solution"),
            (wcs_c.as_ptr(), ptr::null(), empty_c.as_ptr(), "no_catalog"),
        ] {
            let mut unchanged = data.clone();
            let outcome = call_json(
                unsafe {
                    seiza_color_calibrate_image_json(
                        unchanged.as_mut_ptr(),
                        unchanged.len(),
                        width,
                        height,
                        wcs,
                        gaia,
                        catalog,
                        ptr::null(),
                        &mut error,
                    )
                },
                error,
            )
            .unwrap();
            assert_eq!(outcome["status"], "skipped");
            assert_eq!(outcome["reason"], reason, "{outcome}");
            assert!(outcome["detail"].is_string());
            assert_eq!(unchanged, data);
        }

        // A turn cannot carry SIP terms.
        let distorted = CString::new(
            json!({
                "crval": [56.75, 24.1], "crpix": [649.5, 549.5], "cd": frame.cd,
                "sip": {"order": 2, "a": [1e-6, 0.0, 0.0], "b": [0.0, 0.0, 1e-6], "ap": [], "bp": []},
            })
            .to_string(),
        )
        .unwrap();
        let refused =
            unsafe { seiza_wcs_transform_json(distorted.as_ptr(), transform.as_ptr(), &mut error) };
        assert!(call_json(refused, error).is_err());
    }
}

// ---------------------------------------------------------------------------
// Screening bias, dark and flat frames before a master is built
// ---------------------------------------------------------------------------

/// Settings for measuring one bias or dark; each left out keeps the core's
/// default.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DarkMeasureRequest {
    path: String,
    #[serde(default)]
    hot_sigma: Option<f32>,
}

/// Settings for measuring one flat; each left out keeps the core's default.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct FlatMeasureRequest {
    path: String,
    #[serde(default)]
    cfa: Option<bool>,
    #[serde(default)]
    full_scale: Option<f32>,
    #[serde(default)]
    feature_depth: Option<f32>,
    #[serde(default)]
    feature_sigma: Option<f32>,
    #[serde(default)]
    large_features: Option<bool>,
    #[serde(default)]
    focal_ratio: Option<f32>,
    #[serde(default)]
    pixel_size_um: Option<f32>,
    #[serde(default)]
    obstruction: Option<f32>,
}

/// The light cone from a request's optional numbers: `None` when none is
/// given; an error when some are given without both the focal ratio and the
/// pixel size.
fn optics(
    focal_ratio: Option<f32>,
    pixel_size_um: Option<f32>,
    obstruction: Option<f32>,
) -> Result<Option<seiza_calibration::dust::Optics>, String> {
    seiza_calibration::dust::Optics::from_parts(focal_ratio, pixel_size_um, obstruction)
        .map_err(calibration_message)
}

/// `seiza_calibration::DarkMeasurement` in camelCase, both ways: hosts keep
/// what a measure call returned and hand it back to a screen.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct DarkMeasurementJson {
    #[serde(default = "frame_quality_algorithm_version")]
    algorithm_version: u32,
    width: usize,
    height: usize,
    level: f32,
    sigma: f32,
    noise: f32,
    row_banding: f32,
    column_banding: f32,
    hot_pixels: u64,
    hot_threshold: f32,
}

/// `seiza_calibration::FlatMeasurement` in camelCase, both ways.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct FlatMeasurementJson {
    #[serde(default = "first_flat_quality_algorithm_version")]
    algorithm_version: u32,
    width: usize,
    height: usize,
    full_scale: f32,
    level: f32,
    channel_levels: Vec<f32>,
    clipped_fraction: f64,
    noise: f32,
    detection_depth: f32,
    illumination: seiza_calibration::IlluminationProfile,
    feature_count: usize,
    features: Vec<FlatFeatureJson>,
    #[serde(default)]
    vignetting: Vec<seiza_calibration::dust::Vignetting>,
}

/// `seiza_calibration::FlatFeature` in camelCase, both ways.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct FlatFeatureJson {
    x: f32,
    y: f32,
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
    pixels: u32,
    depth: f32,
    #[serde(default)]
    shape: seiza_calibration::dust::DustShape,
    #[serde(default)]
    radius: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    hole: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    distance_mm: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    seen_in: Option<seiza_calibration::dust::SeenIn>,
}

impl From<seiza_calibration::FlatFeature> for FlatFeatureJson {
    fn from(feature: seiza_calibration::FlatFeature) -> Self {
        Self {
            x: feature.x,
            y: feature.y,
            left: feature.left,
            top: feature.top,
            right: feature.right,
            bottom: feature.bottom,
            pixels: feature.pixels,
            depth: feature.depth,
            shape: feature.shape,
            radius: feature.radius,
            hole: feature.hole,
            distance_mm: feature.distance_mm,
            seen_in: feature.seen_in,
        }
    }
}

impl From<FlatFeatureJson> for seiza_calibration::FlatFeature {
    fn from(feature: FlatFeatureJson) -> Self {
        let mut core = Self::default();
        core.x = feature.x;
        core.y = feature.y;
        core.left = feature.left;
        core.top = feature.top;
        core.right = feature.right;
        core.bottom = feature.bottom;
        core.pixels = feature.pixels;
        core.depth = feature.depth;
        core.shape = feature.shape;
        core.radius = feature.radius;
        core.hole = feature.hole;
        core.distance_mm = feature.distance_mm;
        core.seen_in = feature.seen_in;
        core
    }
}

const fn frame_quality_algorithm_version() -> u32 {
    seiza_calibration::FRAME_QUALITY_ALGORITHM_VERSION
}

const fn first_flat_quality_algorithm_version() -> u32 {
    1
}

impl From<seiza_calibration::DarkMeasurement> for DarkMeasurementJson {
    fn from(measurement: seiza_calibration::DarkMeasurement) -> Self {
        Self {
            algorithm_version: measurement.algorithm_version,
            width: measurement.width,
            height: measurement.height,
            level: measurement.level,
            sigma: measurement.sigma,
            noise: measurement.noise,
            row_banding: measurement.row_banding,
            column_banding: measurement.column_banding,
            hot_pixels: measurement.hot_pixels,
            hot_threshold: measurement.hot_threshold,
        }
    }
}

impl From<DarkMeasurementJson> for seiza_calibration::DarkMeasurement {
    fn from(measurement: DarkMeasurementJson) -> Self {
        Self {
            algorithm_version: measurement.algorithm_version,
            width: measurement.width,
            height: measurement.height,
            level: measurement.level,
            sigma: measurement.sigma,
            noise: measurement.noise,
            row_banding: measurement.row_banding,
            column_banding: measurement.column_banding,
            hot_pixels: measurement.hot_pixels,
            hot_threshold: measurement.hot_threshold,
        }
    }
}

impl From<seiza_calibration::FlatMeasurement> for FlatMeasurementJson {
    fn from(measurement: seiza_calibration::FlatMeasurement) -> Self {
        Self {
            algorithm_version: measurement.algorithm_version,
            width: measurement.width,
            height: measurement.height,
            full_scale: measurement.full_scale,
            level: measurement.level,
            channel_levels: measurement.channel_levels,
            clipped_fraction: measurement.clipped_fraction,
            noise: measurement.noise,
            detection_depth: measurement.detection_depth,
            illumination: measurement.illumination,
            feature_count: measurement.feature_count,
            features: measurement.features.into_iter().map(Into::into).collect(),
            vignetting: measurement.vignetting,
        }
    }
}

impl From<FlatMeasurementJson> for seiza_calibration::FlatMeasurement {
    fn from(measurement: FlatMeasurementJson) -> Self {
        Self {
            algorithm_version: measurement.algorithm_version,
            width: measurement.width,
            height: measurement.height,
            full_scale: measurement.full_scale,
            level: measurement.level,
            channel_levels: measurement.channel_levels,
            clipped_fraction: measurement.clipped_fraction,
            noise: measurement.noise,
            detection_depth: measurement.detection_depth,
            illumination: measurement.illumination,
            feature_count: measurement.feature_count,
            features: measurement.features.into_iter().map(Into::into).collect(),
            vignetting: measurement.vignetting,
        }
    }
}

/// Limits for screening a bias set; each left out keeps the core's default.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct BiasScreeningRequest {
    measurements: Vec<DarkMeasurementJson>,
    #[serde(default)]
    level: Option<seiza_calibration::Limits>,
    #[serde(default)]
    noise: Option<seiza_calibration::Limits>,
    #[serde(default)]
    banding: Option<seiza_calibration::Limits>,
}

/// Limits for screening a dark set; each left out keeps the core's default.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DarkScreeningRequest {
    measurements: Vec<DarkMeasurementJson>,
    #[serde(default)]
    level_excess: Option<seiza_calibration::Limits>,
    #[serde(default)]
    noise: Option<seiza_calibration::Limits>,
    #[serde(default)]
    hot_pixels: Option<seiza_calibration::Limits>,
    #[serde(default)]
    banding: Option<seiza_calibration::Limits>,
}

/// Limits for screening a flat set; each left out keeps the core's default.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct FlatScreeningRequest {
    measurements: Vec<FlatMeasurementJson>,
    #[serde(default)]
    clipped: Option<seiza_calibration::Limits>,
    #[serde(default)]
    bright: Option<seiza_calibration::Limits>,
    #[serde(default)]
    dim: Option<seiza_calibration::Limits>,
    #[serde(default)]
    illumination: Option<seiza_calibration::Limits>,
    #[serde(default)]
    features: Option<seiza_calibration::Limits>,
    #[serde(default)]
    changing_features: Option<seiza_calibration::Limits>,
    #[serde(default)]
    carried_features: Option<seiza_calibration::Limits>,
    #[serde(default)]
    steady_presence: Option<f64>,
    #[serde(default)]
    steady_depth_range: Option<f64>,
    #[serde(default)]
    match_distance: Option<u32>,
}

/// One verdict per frame of a bias or dark set, in request order.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct FrameScreeningResponse {
    schema_version: u32,
    frames: Vec<seiza_calibration::FrameVerdict>,
}

/// `seiza_calibration::FlatSetReport` in camelCase.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct FlatScreeningResponse {
    schema_version: u32,
    frames: Vec<seiza_calibration::FrameVerdict>,
    master: seiza_calibration::FrameVerdict,
    steady_features: usize,
    changing_features: usize,
    carried_features: usize,
}

fn calibration_message(error: seiza_calibration::Error) -> String {
    error.to_string()
}

/// A raw frame read through seiza-image-io, as the master builder reads it.
fn open_calibration_frame(path: &str) -> Result<FitsFrame, String> {
    FitsFrame::open(path).map_err(stacking_message)
}

fn frame_view(frame: &FitsFrame) -> Result<seiza_calibration::LinearImageRef<'_>, String> {
    let image = &frame.image;
    seiza_calibration::LinearImageRef::new(&image.data, image.width, image.height, image.channels)
        .map_err(calibration_message)
}

#[unsafe(no_mangle)]
/// Measure one raw bias or dark frame, as a step toward judging its set
/// with [`seiza_calibration_screen_bias_json`] or
/// [`seiza_calibration_screen_darks_json`].
///
/// `request_json` is `{"path": ..., "hotSigma": 10}`; `hotSigma`, the
/// height above the level in units of pixel noise at which a sample counts
/// as hot, may be left out. The file is read through seiza-image-io, FITS
/// or XISF, before any calibration. The owned response holds the
/// measurement: `algorithmVersion`, `width`, `height`, `level`, `sigma`,
/// `noise`, `rowBanding`, `columnBanding`, `hotPixels` and `hotThreshold`,
/// in the frame's own units. It is small: keep it, and pass it back to a
/// screen whenever the set changes rather than reading the frame again. A
/// stored measurement whose `algorithmVersion` differs from a new one's is
/// stale. Free the response with [`seiza_string_free`].
///
/// # Safety
/// `request_json` must be a valid NUL-terminated string. When non-null,
/// `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_calibration_measure_dark_json(
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let request: DarkMeasureRequest = parse_request(request_json, "dark measure")?;
        let mut options = seiza_calibration::DarkMeasureOptions::default();
        if let Some(hot_sigma) = request.hot_sigma {
            options.hot_sigma = hot_sigma;
        }
        let frame = open_calibration_frame(&request.path)?;
        let measurement = seiza_calibration::measure_dark(frame_view(&frame)?, &options)
            .map_err(calibration_message)?;
        owned_json(&DarkMeasurementJson::from(measurement))
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Measure one raw flat frame, as a step toward judging its set with
/// [`seiza_calibration_screen_flats_json`].
///
/// `request_json` is `{"path": ...}` with optional `cfa` (true for a raw
/// colour frame with one channel; left out, true when the header records a
/// Bayer pattern), `fullScale` (the value a saturated sample reads, 65535 by
/// default), `featureDepth` (the shallowest dip counted, as a fraction of
/// the light around it, 0.02 by default) and `featureSigma` (how far above
/// the noise a dip must reach, 5 by default). Give the raw samples, before
/// bias or dark subtraction: levels and clipping are judged against
/// `fullScale`. The owned response holds `algorithmVersion`, `width`,
/// `height`, `fullScale`, `level`, `channelLevels`, `clippedFraction`,
/// `noise`, `detectionDepth`, `illumination` (`columns`, `rows` and a
/// row-major `response` of cell medians, 1 for a typical cell),
/// `featureCount`, the `features` — large disks, donuts and patches first,
/// then compact and irregular dips, each with `x`, `y`, `left`, `top`,
/// `right`, `bottom`, `pixels`, `depth`, `shape` (`compact`, `irregular`,
/// `disk`, `donut` or `patch`), `radius`, and when known `hole`,
/// `distanceMm` and `seenIn` — and `vignetting` (`edge`, `x`, `y`, `area`,
/// `depth`). Optional `largeFeatures` (default true) turns the search for
/// large ones off. Optional `focalRatio` and `pixelSizeUm`, given together,
/// with an optional `obstruction`, describe the light cone, so each disk gets
/// a distance from the sensor; any of them without both of the first two is
/// refused. Keep the response and pass it back to a screen whenever the set
/// changes. Free it with [`seiza_string_free`].
///
/// # Safety
/// As [`seiza_calibration_measure_dark_json`].
pub unsafe extern "C" fn seiza_calibration_measure_flat_json(
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let request: FlatMeasureRequest = parse_request(request_json, "flat measure")?;
        let mut options = seiza_calibration::FlatMeasureOptions::default();
        if let Some(full_scale) = request.full_scale {
            options.full_scale = full_scale;
        }
        if let Some(feature_depth) = request.feature_depth {
            options.feature_depth = feature_depth;
        }
        if let Some(feature_sigma) = request.feature_sigma {
            options.feature_sigma = feature_sigma;
        }
        if let Some(large_features) = request.large_features {
            options.large_features = large_features;
        }
        options.optics = optics(
            request.focal_ratio,
            request.pixel_size_um,
            request.obstruction,
        )?;
        let frame = open_calibration_frame(&request.path)?;
        let cfa = request.cfa.unwrap_or(frame.bayer.is_some());
        let measurement = seiza_calibration::measure_flat(frame_view(&frame)?, cfa, &options)
            .map_err(calibration_message)?;
        owned_json(&FlatMeasurementJson::from(measurement))
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Judge a set of bias frames that should match: one camera, gain, offset,
/// readout mode and binning.
///
/// `request_json` holds `measurements`, each as
/// [`seiza_calibration_measure_dark_json`] returned it, and optional limits
/// `level`, `noise` and `banding`, each `{"suspect": ..., "poor": ...}`:
/// the value at which a frame turns suspect, then poor (higher is worse).
/// Banding is judged on each frame alone; with three or more frames, each
/// is also compared with the set's median level and noise. The owned
/// response holds `frames`, one verdict per measurement in request order:
/// `grade` (`ok`, `suspect` or `poor`) and `findings`, each with `code`
/// (`level`, `noise`, `row_banding` or `column_banding`), `grade`, the
/// measured `value` and the `limit` it reached. The verdicts say what the
/// pixels show, never why. Free it with [`seiza_string_free`].
///
/// # Safety
/// As [`seiza_calibration_measure_dark_json`].
pub unsafe extern "C" fn seiza_calibration_screen_bias_json(
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let request: BiasScreeningRequest = parse_request(request_json, "bias screening")?;
        let mut screening = seiza_calibration::BiasScreening::default();
        if let Some(limits) = request.level {
            screening.level = limits;
        }
        if let Some(limits) = request.noise {
            screening.noise = limits;
        }
        if let Some(limits) = request.banding {
            screening.banding = limits;
        }
        let measurements: Vec<_> = request.measurements.into_iter().map(Into::into).collect();
        owned_json(&FrameScreeningResponse {
            schema_version: 1,
            frames: seiza_calibration::screen_bias(&measurements, &screening),
        })
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Judge a set of darks that should match: one camera mode, one exposure,
/// one sensor temperature.
///
/// `request_json` holds `measurements`, each as
/// [`seiza_calibration_measure_dark_json`] returned it, and optional limits
/// `levelExcess`, `noise`, `hotPixels` and `banding`, each `{"suspect":
/// ..., "poor": ...}` (higher is worse). With two or more frames each
/// frame's level is compared with the quietest frame's, since light can only
/// add to a dark; with three or more, noise and hot pixels are compared with
/// the median of the frames whose level passed. The response is shaped as
/// [`seiza_calibration_screen_bias_json`]'s, with the codes `level_excess`,
/// `noise`, `hot_pixels`, `row_banding` and `column_banding`. Free it with
/// [`seiza_string_free`].
///
/// # Safety
/// As [`seiza_calibration_measure_dark_json`].
pub unsafe extern "C" fn seiza_calibration_screen_darks_json(
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let request: DarkScreeningRequest = parse_request(request_json, "dark screening")?;
        let mut screening = seiza_calibration::DarkScreening::default();
        if let Some(limits) = request.level_excess {
            screening.level_excess = limits;
        }
        if let Some(limits) = request.noise {
            screening.noise = limits;
        }
        if let Some(limits) = request.hot_pixels {
            screening.hot_pixels = limits;
        }
        if let Some(limits) = request.banding {
            screening.banding = limits;
        }
        let measurements: Vec<_> = request.measurements.into_iter().map(Into::into).collect();
        owned_json(&FrameScreeningResponse {
            schema_version: 1,
            frames: seiza_calibration::screen_darks(&measurements, &screening),
        })
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Judge a set of flats that should match: one filter, one session, one
/// rotation, one optical train.
///
/// `request_json` holds `measurements`, each as
/// [`seiza_calibration_measure_flat_json`] returned it, and optional limits
/// `clipped`, `bright`, `dim` (lower is worse), `illumination`, `features`,
/// `changingFeatures` and `carriedFeatures`, each `{"suspect": ...,
/// "poor": ...}`, with `steadyPresence` (the share of frames a steady
/// feature is found in, 0.9 by default), `steadyDepthRange` (how much its
/// depth may vary, 0.5 by default) and `matchDistance` (pixels between
/// boxes that are one feature, 2 by default). Every frame is checked alone
/// for clipping, exposure and clear compact features; with three or more,
/// each is also compared with the set. The owned response holds `frames`,
/// one verdict per measurement in request order, as
/// [`seiza_calibration_screen_bias_json`] gives them; `master`, the verdict
/// on a master built from every frame (whether changing features would
/// survive into it); and the counts `steadyFeatures`, `changingFeatures`
/// and `carriedFeatures`. Fails when the measurements differ in size. Free
/// it with [`seiza_string_free`].
///
/// # Safety
/// As [`seiza_calibration_measure_dark_json`].
pub unsafe extern "C" fn seiza_calibration_screen_flats_json(
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let request: FlatScreeningRequest = parse_request(request_json, "flat screening")?;
        let mut screening = seiza_calibration::FlatScreening::default();
        let limits = [
            (request.clipped, &mut screening.clipped),
            (request.bright, &mut screening.bright),
            (request.dim, &mut screening.dim),
            (request.illumination, &mut screening.illumination),
            (request.features, &mut screening.features),
            (request.changing_features, &mut screening.changing_features),
            (request.carried_features, &mut screening.carried_features),
        ];
        for (given, target) in limits {
            if let Some(value) = given {
                *target = value;
            }
        }
        if let Some(value) = request.steady_presence {
            screening.steady_presence = value;
        }
        if let Some(value) = request.steady_depth_range {
            screening.steady_depth_range = value;
        }
        if let Some(value) = request.match_distance {
            screening.match_distance = value;
        }
        let measurements: Vec<_> = request.measurements.into_iter().map(Into::into).collect();
        let report = seiza_calibration::screen_flats(&measurements, &screening)
            .map_err(calibration_message)?;
        owned_json(&FlatScreeningResponse {
            schema_version: 1,
            frames: report.frames,
            master: report.master,
            steady_features: report.steady_features,
            changing_features: report.changing_features,
            carried_features: report.carried_features,
        })
    })
    .unwrap_or(ptr::null_mut())
}

// ---------------------------------------------------------------------------
// Residual flat responses
// ---------------------------------------------------------------------------

/// Settings for building a residual flat response; each left out keeps the
/// core's default.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ResidualFlatOptionsRequest {
    minimum_samples: Option<usize>,
    minimum_depth: Option<f32>,
    minimum_consensus: Option<f32>,
    maximum_gain: Option<f32>,
    background_edge_fraction: Option<f32>,
    smoothing_sigma: Option<f32>,
    edge_feather_fraction: Option<f32>,
    minimum_corrected_samples: Option<usize>,
    minimum_connected_pixels: Option<usize>,
}

impl ResidualFlatOptionsRequest {
    fn into_core(self) -> seiza_calibration::ResidualFlatOptions {
        let mut options = seiza_calibration::ResidualFlatOptions::default();
        let counts = [
            (self.minimum_samples, &mut options.minimum_samples),
            (
                self.minimum_corrected_samples,
                &mut options.minimum_corrected_samples,
            ),
            (
                self.minimum_connected_pixels,
                &mut options.minimum_connected_pixels,
            ),
        ];
        for (given, target) in counts {
            if let Some(value) = given {
                *target = value;
            }
        }
        let fractions = [
            (self.minimum_depth, &mut options.minimum_depth),
            (self.minimum_consensus, &mut options.minimum_consensus),
            (self.maximum_gain, &mut options.maximum_gain),
            (
                self.background_edge_fraction,
                &mut options.background_edge_fraction,
            ),
            (self.smoothing_sigma, &mut options.smoothing_sigma),
            (
                self.edge_feather_fraction,
                &mut options.edge_feather_fraction,
            ),
        ];
        for (given, target) in fractions {
            if let Some(value) = given {
                *target = value;
            }
        }
        options
    }
}

/// `seiza_calibration::ResidualFlatDiagnostics` in camelCase.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ResidualFlatDiagnosticsResponse {
    schema_version: u32,
    algorithm_version: u32,
    sample_count: usize,
    corrected_samples: usize,
    total_samples: usize,
    largest_connected_pixels: usize,
    minimum_response: f32,
    maximum_applied_gain: f32,
}

/// The number of samples in a `width` x `height` x `channels` buffer.
fn residual_flat_samples(width: usize, height: usize, channels: usize) -> Result<usize, String> {
    width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or_else(|| "residual flat dimensions overflow".to_string())
}

#[unsafe(no_mangle)]
/// Build a residual flat response from crops of light frames that share one
/// place on the detector, and write it into the caller's buffer.
///
/// A residual flat corrects a dip the flat left behind, such as dust that
/// moved after the flats were taken. Each of the `sample_count` pointers in
/// `samples` is one crop of `width` x `height` x `channels` interleaved
/// linear samples, every crop cut from the same detector position of a
/// different light. The host must first show that the dip stays fixed on the
/// detector while the sky moves; this call does not decide what the dip is.
/// Each crop's own edge sets a local background, and only a loss of light
/// most crops share is kept. `options_json` may be null or empty for the
/// defaults, or hold any of `minimumSamples` (5), `minimumDepth` (0.005),
/// `minimumConsensus` (0.7), `maximumGain` (1.2), `backgroundEdgeFraction`
/// (0.2), `smoothingSigma` (2), `edgeFeatherFraction` (0.12),
/// `minimumCorrectedSamples` (16) and `minimumConnectedPixels` (64).
///
/// `response_out` receives `response_length` samples, which must equal one
/// crop's: the response, 1 where nothing is corrected and lower where light
/// is lost, never below `1 / maximumGain`. Apply it with
/// [`seiza_calibration_apply_residual_flat_response`]. Returns owned JSON
/// with `algorithmVersion` (keep it with a stored response), `sampleCount`,
/// `correctedSamples`, `totalSamples`, `largestConnectedPixels`,
/// `minimumResponse` and `maximumAppliedGain`, or null with `error_out` set
/// when the crops show too little to correct. Free it with
/// [`seiza_string_free`].
///
/// # Safety
/// `samples` must point to `sample_count` pointers, each to one crop's
/// readable floats. `response_out` must point to `response_length` writable
/// floats. A non-null `options_json` must be NUL-terminated. When non-null,
/// `error_out` must point to writable storage for one pointer.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn seiza_calibration_build_residual_flat_json(
    samples: *const *const f32,
    sample_count: usize,
    width: usize,
    height: usize,
    channels: usize,
    options_json: *const c_char,
    response_out: *mut f32,
    response_length: usize,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        if samples.is_null() {
            return Err("residual flat crops are required".into());
        }
        let length = residual_flat_samples(width, height, channels)?;
        if response_length != length {
            return Err(format!(
                "residual flat response buffer has {response_length} floats; expected {length}"
            ));
        }
        let options = if options_json.is_null() {
            ResidualFlatOptionsRequest::default()
        } else {
            let text = required_str(options_json, "residual flat options JSON")?;
            if text.trim().is_empty() {
                ResidualFlatOptionsRequest::default()
            } else {
                serde_json::from_str(&text)
                    .map_err(|error| format!("invalid residual flat options JSON: {error}"))?
            }
        }
        .into_core();
        let pointers = unsafe { std::slice::from_raw_parts(samples, sample_count) };
        let views = pointers
            .iter()
            .map(|&data| {
                let data = unsafe { required_f32_slice(data, length, "residual flat crop")? };
                seiza_calibration::LinearImageRef::new(data, width, height, channels)
                    .map_err(calibration_message)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let built = seiza_calibration::build_residual_flat_patch(&views, &options)
            .map_err(calibration_message)?;
        let response_out =
            unsafe { required_f32_slice_mut(response_out, length, "residual flat response")? };
        response_out.copy_from_slice(built.patch.response());
        let diagnostics = built.diagnostics;
        owned_json(&ResidualFlatDiagnosticsResponse {
            schema_version: 1,
            algorithm_version: diagnostics.algorithm_version,
            sample_count: diagnostics.sample_count,
            corrected_samples: diagnostics.corrected_samples,
            total_samples: diagnostics.total_samples,
            largest_connected_pixels: diagnostics.largest_connected_pixels,
            minimum_response: diagnostics.minimum_response,
            maximum_applied_gain: diagnostics.maximum_applied_gain,
        })
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Check a residual flat response the host stored or made itself before it
/// keeps or applies it: every value must be finite and in (0, 1]. Returns
/// true when it is, or false with `error_out` set.
///
/// # Safety
/// `response` must point to `response_length` readable floats, which must
/// equal `width * height * channels`. When non-null, `error_out` must point
/// to writable storage for one pointer.
pub unsafe extern "C" fn seiza_calibration_validate_residual_flat_response(
    response: *const f32,
    response_length: usize,
    width: usize,
    height: usize,
    channels: usize,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let data =
            unsafe { required_f32_slice(response, response_length, "residual flat response")? };
        let view = seiza_calibration::LinearImageRef::new(data, width, height, channels)
            .map_err(calibration_message)?;
        seiza_calibration::validate_residual_flat_response(view).map_err(calibration_message)
    })
    .is_some()
}

#[unsafe(no_mangle)]
/// Divide a residual flat response out of a light, in place, with the
/// response's top-left corner at pixel (`x`, `y`) of the light.
///
/// The response is `response_width` x `response_height` with the light's
/// channel count, as [`seiza_calibration_build_residual_flat_json`] wrote
/// it or as the host stored it; it is checked first as
/// [`seiza_calibration_validate_residual_flat_response`] does. The light
/// must be sampled as the crops were, and the response must lie inside it.
/// Returns true, or false with `error_out` set and the light unchanged.
///
/// # Safety
/// `response` must point to `response_length` readable floats and `image`
/// to `image_length` writable floats, each equal to its width times height
/// times channels. When non-null, `error_out` must point to writable storage
/// for one pointer.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn seiza_calibration_apply_residual_flat_response(
    response: *const f32,
    response_length: usize,
    response_width: usize,
    response_height: usize,
    image: *mut f32,
    image_length: usize,
    image_width: usize,
    image_height: usize,
    channels: usize,
    x: usize,
    y: usize,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let response =
            unsafe { required_f32_slice(response, response_length, "residual flat response")? };
        let response = seiza_calibration::LinearImageRef::new(
            response,
            response_width,
            response_height,
            channels,
        )
        .map_err(calibration_message)?;
        let image = unsafe { required_f32_slice_mut(image, image_length, "light image")? };
        let image =
            seiza_calibration::LinearImageMut::new(image, image_width, image_height, channels)
                .map_err(calibration_message)?;
        seiza_calibration::apply_residual_flat_response_at(response, image, x, y)
            .map_err(calibration_message)
    })
    .is_some()
}

// ---------------------------------------------------------------------------
// Integrating frames registered elsewhere
// ---------------------------------------------------------------------------

/// Frames already calibrated, registered and normalized onto one grid.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RegisteredIntegrationRequest {
    inputs: Vec<String>,
    #[serde(default)]
    rejection: MasterRejectionRequest,
    #[serde(default)]
    minimum_sigma: Option<f32>,
    #[serde(default)]
    frame_weights: Option<Vec<Vec<f32>>>,
}

/// What the final pass did with each input, in request order.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RegisteredIntegrationResponse {
    schema_version: u32,
    frames: Vec<RegisteredFrameResponse>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RegisteredFrameResponse {
    path: String,
    finite_samples: usize,
    integrated_samples: usize,
}

#[unsafe(no_mangle)]
/// Integrate FITS or XISF frames that were calibrated, registered and
/// normalized elsewhere, with the three-pass leave-one-out rejection
/// [`seiza_live_stacker_reintegrate`] uses, and return the result as a new
/// snapshot.
///
/// `request_json` holds `inputs`, the paths of the frames, all one size on
/// one pixel grid, with samples a frame does not cover stored as NaN; and
/// optional `rejection` (`lowSigma`, `highSigma`, 3 by default),
/// `minimumSigma` (a noise floor in the frames' units, 1e-6 by default) and
/// `frameWeights` (one list per input with one positive weight per channel,
/// as a live stack's diagnostics give them; left out, every frame weighs
/// the same). Each input is read through seiza-image-io once per pass, and
/// must read the same each time. Nothing is registered or normalized again,
/// and every input takes part. Pixels with fewer than three samples are
/// averaged without rejection. Memory is about 40 bytes per output sample
/// plus one frame.
///
/// When `report_out` is non-null it receives owned JSON holding `frames`,
/// one per input in request order, with its `path`, `finiteSamples` (the
/// samples it covers) and `integratedSamples` (those rejection kept); free
/// it with [`seiza_string_free`]. `cancel` and `progress` work as for
/// [`seiza_live_stacker_reintegrate`], with the input's index and the input
/// count. The snapshot writes with the first input's WCS, and
/// refuses to write over an input. Release it with
/// [`seiza_stack_snapshot_free`]. On failure the snapshot and report are
/// null and `error_out` says why.
///
/// # Safety
/// `request_json` must be a valid NUL-terminated string. `cancel` must be
/// null or a live [`SeizaCancelSignal`] retained until this call returns.
/// `context` is passed through untouched to `progress`. When non-null,
/// `report_out` and `error_out` must each point to writable storage for one
/// pointer.
pub unsafe extern "C" fn seiza_stack_integrate_registered_json(
    request_json: *const c_char,
    cancel: *const SeizaCancelSignal,
    progress: SeizaStackReintegrateProgressCallback,
    context: *mut c_void,
    report_out: *mut *mut c_char,
    error_out: *mut *mut c_char,
) -> *mut SeizaStackSnapshot {
    clear_error(error_out);
    clear_error(report_out);
    // Raw pointers are not UnwindSafe; the callback runs on this thread and
    // the context's lifetime is the caller's promise.
    let context = context as usize;
    ffi_result(error_out, || {
        let request: RegisteredIntegrationRequest =
            parse_request(request_json, "registered integration")?;
        let paths: Vec<PathBuf> = request.inputs.iter().map(PathBuf::from).collect();
        let mut options = seiza_stacking::BatchStackOptions {
            rejection: MasterRejectionOptions {
                low_sigma: request.rejection.low_sigma,
                high_sigma: request.rejection.high_sigma,
            },
            cancel: unsafe { cancel.as_ref() }
                .map(|signal| CancelSignal::from(Arc::clone(&signal.cancelled))),
            frame_weights: request.frame_weights,
            ..seiza_stacking::BatchStackOptions::default()
        };
        if let Some(minimum_sigma) = request.minimum_sigma {
            options.minimum_sigma = minimum_sigma;
        }
        let count = paths.len();
        let mut reference_headers = None;
        let result = seiza_stacking::integrate_registered_frames(count, &options, |pass, index| {
            if let Some(callback) = progress {
                let pass = match pass {
                    seiza_stacking::BatchStackPass::Estimate => 0,
                    seiza_stacking::BatchStackPass::Refine => 1,
                    seiza_stacking::BatchStackPass::Integrate => 2,
                };
                unsafe { callback(pass, index, count, context as *mut c_void) };
            }
            let frame = FitsFrame::open(&paths[index])?;
            if reference_headers.is_none() && index == 0 {
                reference_headers = Some(frame.headers);
            }
            Ok(frame.image)
        })
        .map_err(stacking_message)?;
        if !report_out.is_null() {
            let report = RegisteredIntegrationResponse {
                schema_version: 1,
                frames: request
                    .inputs
                    .iter()
                    .zip(&result.frames)
                    .map(|(path, frame)| RegisteredFrameResponse {
                        path: path.clone(),
                        finite_samples: frame.finite_samples,
                        integrated_samples: frame.integrated_samples,
                    })
                    .collect(),
            };
            let report = owned_json(&report)?;
            unsafe { *report_out = report };
        }
        Ok(SeizaStackSnapshot {
            snapshot: result.snapshot,
            reference_headers: reference_headers.unwrap_or_default(),
            input_paths: paths,
        })
    })
    .map_or(ptr::null_mut(), |snapshot| {
        Box::into_raw(Box::new(snapshot))
    })
}

// ---------------------------------------------------------------------------
// One background surface for the channels of a colour image
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
/// Choose one background surface for the channels of one image that were
/// fitted one at a time.
///
/// Automatic model selection runs per fit, so channels fitted apart can
/// each land on a different surface, a constant in one and a curve in
/// another, and correcting them with different shapes leaves a colour cast
/// that grows toward the edges. `models` holds `model_count` models from
/// [`seiza_background_fit`], each fitted with an automatic `model` in
/// `config_json`, the same `BackgroundConfig` JSON the fits took (null or
/// empty for the default). This averages each candidate surface's held-out
/// error over every fit and chooses as automatic selection does for one.
///
/// Returns owned `BackgroundConfig` JSON: `config_json` with its `model`
/// replaced by the chosen surface. Fit each channel again with it. Returns
/// the JSON `null` when there is no shared choice: the config is not
/// automatic, a fit holds no selection diagnostics, or no candidate was
/// scored in every fit; keep the fits as they are then. Null with
/// `error_out` set means the arguments were wrong. Free the string with
/// [`seiza_string_free`].
///
/// # Safety
/// `models` must point to `model_count` live model pointers. A non-null
/// `config_json` must be NUL-terminated. When non-null, `error_out` must
/// point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_background_select_shared_model_json(
    models: *const *const SeizaBackgroundModel,
    model_count: usize,
    config_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        if models.is_null() {
            return Err("background models are required".into());
        }
        let config = background_config(config_json)?;
        let pointers = unsafe { std::slice::from_raw_parts(models, model_count) };
        let fits = pointers
            .iter()
            .map(|&model| unsafe { required_background_model(model) }.map(|model| &model.fit))
            .collect::<Result<Vec<_>, String>>()?;
        let shared = seiza_background::select_shared_model(&config.model, &fits)
            .map(|model| BackgroundConfig { model, ..config });
        owned_json(&shared)
    })
    .unwrap_or(ptr::null_mut())
}

// ---------------------------------------------------------------------------
// A light angle the host measured
// ---------------------------------------------------------------------------

/// The light's recorded rotator angle is compared, as for any frame.
pub const SEIZA_LIGHT_ROTATION_RECORDED: u32 = 0;
/// An angle the host measured is compared in place of the recorded one.
pub const SEIZA_LIGHT_ROTATION_MEASURED: u32 = 1;
/// The host has measured that the angle does not decide the light's flats:
/// any flat angle passes.
pub const SEIZA_LIGHT_ROTATION_UNKNOWN: u32 = 2;

#[unsafe(no_mangle)]
/// Set how the stack's gate reads a light's rotator angle when it matches
/// the master flat to the lights pushed from now on, keeping its other
/// tolerances.
///
/// `basis` is [`SEIZA_LIGHT_ROTATION_RECORDED`] (the default),
/// [`SEIZA_LIGHT_ROTATION_MEASURED`], with `degrees` the angle the host
/// measured for the light (by plate solving it, or by fitting its sky
/// background against the candidate flats), or
/// [`SEIZA_LIGHT_ROTATION_UNKNOWN`] when the host has shown the angle does
/// not decide which flats the light follows; `degrees` is read only for a
/// measured angle. The flat's own angle is always the recorded one.
/// [`SeizaMatchTolerances`] is a fixed layout with no room for this, so it
/// has its own call: [`seiza_live_stacker_set_match_tolerances`] replaces
/// the whole gate and reads the recorded angle again, so call this after
/// it. A saved context keeps the setting, and the configuration fingerprint
/// changes with it. Returns false with `error_out` set for an unknown basis,
/// a measured angle that is not finite, or a stack created from prepared
/// arrays.
///
/// # Safety
/// `stacker` must be a live stacker from this library, with this call
/// externally synchronized against every other operation on it. When
/// non-null, `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_live_stacker_set_light_rotation(
    stacker: *mut SeizaLiveStacker,
    basis: u32,
    degrees: f64,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let stacker = unsafe { required_live_stacker_mut(stacker)? };
        let light_rotation = match basis {
            SEIZA_LIGHT_ROTATION_RECORDED => seiza_calibration::LightRotation::Recorded,
            SEIZA_LIGHT_ROTATION_MEASURED => seiza_calibration::LightRotation::Measured { degrees },
            SEIZA_LIGHT_ROTATION_UNKNOWN => seiza_calibration::LightRotation::Unknown,
            other => return Err(format!("unknown light rotation basis {other}")),
        };
        let tolerances = seiza_calibration::MatchTolerances {
            light_rotation,
            ..stacker.stacker.match_tolerances()
        };
        stacker
            .stacker
            .set_match_tolerances(tolerances)
            .map_err(stacking_message)
    })
    .is_some()
}

#[unsafe(no_mangle)]
/// Read how the stack's gate reads a light's rotator angle now: one of the
/// `SEIZA_LIGHT_ROTATION_*` values into `basis_out`, and into `degrees_out`
/// the measured angle, or zero for the other bases. Either output may be
/// null. Returns false when `stacker` is null.
///
/// # Safety
/// `stacker` must be null or a live stacker from this library, with this
/// read externally synchronized against mutable operations. Each non-null
/// output must point to writable storage for one value.
pub unsafe extern "C" fn seiza_live_stacker_light_rotation(
    stacker: *const SeizaLiveStacker,
    basis_out: *mut u32,
    degrees_out: *mut f64,
) -> bool {
    let Some(stacker) = (unsafe { stacker.as_ref() }) else {
        return false;
    };
    let (basis, degrees) = match stacker.stacker.match_tolerances().light_rotation {
        seiza_calibration::LightRotation::Recorded => (SEIZA_LIGHT_ROTATION_RECORDED, 0.0),
        seiza_calibration::LightRotation::Measured { degrees } => {
            (SEIZA_LIGHT_ROTATION_MEASURED, degrees)
        }
        seiza_calibration::LightRotation::Unknown => (SEIZA_LIGHT_ROTATION_UNKNOWN, 0.0),
    };
    if !basis_out.is_null() {
        unsafe { *basis_out = basis };
    }
    if !degrees_out.is_null() {
        unsafe { *degrees_out = degrees };
    }
    true
}
