use crate::{
    BayerLayout, CalibrationMasters, Error, FitsFrame, FrameMetadata, LinearImage,
    NormalizationMap, NormalizationMode, RegisteredFrameMapping, Registrar, RegistrationOptions,
    Result, SimilarityTransform, context, path_identity, paths_refer_to_same_file,
    resample_to_reference,
};
use rayon::prelude::*;
use seiza_fits::HeaderValue;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Thresholds for per-sample delta-sigma rejection during live stacking.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct DeltaSigmaOptions {
    /// Reject a sample this many sigma below the running mean.
    pub low_sigma: f32,
    /// Reject a sample this many sigma above the running mean.
    pub high_sigma: f32,
    /// Observations a sample needs before rejection starts.
    pub warmup_samples: u32,
    /// Floor on the running sigma, so a near-constant sample stays inclusive.
    pub minimum_sigma: f32,
}

impl Default for DeltaSigmaOptions {
    fn default() -> Self {
        Self {
            low_sigma: 3.0,
            high_sigma: 3.0,
            warmup_samples: 5,
            minimum_sigma: 1.0e-6,
        }
    }
}

/// Which per-sample rejection rule the stack applies.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(tag = "mode", content = "options", rename_all = "kebab-case")]
pub enum RejectionMode {
    /// Keep every finite sample.
    None,
    /// Reject samples that stray too far from the running mean.
    DeltaSigma(DeltaSigmaOptions),
}

impl Default for RejectionMode {
    fn default() -> Self {
        Self::DeltaSigma(DeltaSigmaOptions::default())
    }
}

/// Everything that governs how frames are aligned, matched, rejected, and
/// admitted into a stack.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct StackOptions {
    /// Star-matching and transform-fitting options.
    pub registration: RegistrationOptions,
    /// Background-matching mode.
    pub normalization: NormalizationMode,
    /// Per-sample rejection rule.
    pub rejection: RejectionMode,
    /// Whole-frame admission gates.
    pub acceptance: FrameAcceptanceCriteria,
    /// Replace impulse pixels in each frame after calibration and before
    /// debayering. The defense for lights whose calibration has no dark
    /// master to subtract their hot pixels; `None` (the default) leaves
    /// frames untouched.
    pub cosmetic: Option<crate::cosmetic::ImpulseFilterOptions>,
}

impl StackOptions {
    /// Validate registration, normalization, rejection, and admission bounds.
    pub fn validate(&self) -> Result<()> {
        self.registration.validate()?;
        if matches!(self.normalization, NormalizationMode::Local { tile_size } if tile_size < 16) {
            return Err(Error::Stack(
                "local normalization tile size must be at least 16 pixels".into(),
            ));
        }
        if let RejectionMode::DeltaSigma(rejection) = self.rejection
            && (!rejection.low_sigma.is_finite()
                || rejection.low_sigma <= 0.0
                || !rejection.high_sigma.is_finite()
                || rejection.high_sigma <= 0.0
                || rejection.warmup_samples < 2
                || !rejection.minimum_sigma.is_finite()
                || rejection.minimum_sigma <= 0.0)
        {
            return Err(Error::Stack("invalid delta-sigma options".into()));
        }
        if let Some(cosmetic) = &self.cosmetic
            && (!cosmetic.low_sigma.is_finite()
                || cosmetic.low_sigma <= 0.0
                || !cosmetic.high_sigma.is_finite()
                || cosmetic.high_sigma <= 0.0)
        {
            return Err(Error::Stack(
                "cosmetic filter sigmas must be positive finite numbers".into(),
            ));
        }
        let acceptance = self.acceptance;
        if !acceptance.maximum_registration_rms_pixels.is_finite()
            || acceptance.maximum_registration_rms_pixels <= 0.0
            || !acceptance.maximum_scale_deviation.is_finite()
            || !(0.0..1.0).contains(&acceptance.maximum_scale_deviation)
            || !acceptance.maximum_rotation_degrees.is_finite()
            || !(0.0..=180.0).contains(&acceptance.maximum_rotation_degrees)
            || !acceptance.minimum_overlap_fraction.is_finite()
            || !(0.0..=1.0).contains(&acceptance.minimum_overlap_fraction)
            || !acceptance.minimum_normalization_gain.is_finite()
            || acceptance.minimum_normalization_gain <= 0.0
            || !acceptance.maximum_normalization_gain.is_finite()
            || acceptance.maximum_normalization_gain < acceptance.minimum_normalization_gain
            || !acceptance.minimum_integrated_fraction.is_finite()
            || !(0.0..=1.0).contains(&acceptance.minimum_integrated_fraction)
        {
            return Err(Error::Stack("invalid frame acceptance criteria".into()));
        }
        Ok(())
    }
}

/// Admission gates applied before an additive live-stack update becomes
/// permanent.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct FrameAcceptanceCriteria {
    /// Largest registration RMS residual, in pixels, still accepted.
    pub maximum_registration_rms_pixels: f64,
    /// Largest departure of the transform's scale from unity still accepted.
    pub maximum_scale_deviation: f64,
    /// Maximum rotation away from either the reference orientation or its
    /// 180-degree meridian-flipped orientation.
    pub maximum_rotation_degrees: f64,
    /// Smallest fraction of the frame that must overlap the reference.
    pub minimum_overlap_fraction: f32,
    /// Smallest normalization gain, anywhere in the map, still accepted.
    pub minimum_normalization_gain: f32,
    /// Largest normalization gain, anywhere in the map, still accepted.
    pub maximum_normalization_gain: f32,
    /// Smallest fraction of samples that must survive rejection to admit the
    /// frame.
    pub minimum_integrated_fraction: f32,
}

impl Default for FrameAcceptanceCriteria {
    fn default() -> Self {
        Self {
            maximum_registration_rms_pixels: 2.0,
            maximum_scale_deviation: 0.04,
            maximum_rotation_degrees: 10.0,
            minimum_overlap_fraction: 0.60,
            minimum_normalization_gain: 0.25,
            maximum_normalization_gain: 4.0,
            minimum_integrated_fraction: 0.50,
        }
    }
}

/// Measurements recorded for a frame that passed every admission gate.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct FrameDiagnostics {
    /// Transform used to align the frame.
    pub transform: SimilarityTransform,
    /// Star pairs supporting the registration.
    pub matched_stars: usize,
    /// Registration RMS residual, in pixels.
    pub registration_rms_pixels: f64,
    /// Frame-center displacement under the transform, in pixels.
    pub registration_drift_pixels: f64,
    /// Mean normalization gain applied.
    pub normalization_mean_gain: f32,
    /// Mean normalization offset applied.
    pub normalization_mean_offset: f32,
    /// Versioned transform and normalization provenance for this frame.
    pub mapping: Box<crate::RegisteredFrameMapping>,
    /// Fraction of the frame that overlapped the reference.
    pub overlap_fraction: f32,
    /// Fraction of samples that survived rejection.
    pub integrated_fraction: f32,
    /// Samples integrated from this frame.
    pub accepted_samples: usize,
    /// Samples rejected from this frame.
    pub rejected_samples: usize,
}

/// Why a frame was turned away from the stack.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum FrameRejectionReason {
    /// Calibration masters could not be applied.
    #[error("calibration failed: {0}")]
    Calibration(String),
    /// The frame's shape or channel count did not match the stack.
    #[error("incompatible image: {0}")]
    IncompatibleImage(String),
    /// No transform reached the match threshold.
    #[error("registration failed: {0}")]
    Registration(String),
    /// Registration succeeded but its residual was too large.
    #[error("registration RMS {measured:.3}px exceeds {maximum:.3}px")]
    RegistrationRms {
        /// Measured RMS residual, in pixels.
        measured: f64,
        /// Allowed RMS residual, in pixels.
        maximum: f64,
    },
    /// The transform's scale departed too far from unity.
    #[error("scale deviation {measured:.5} exceeds {maximum:.5}")]
    ScaleDeviation {
        /// Measured scale deviation.
        measured: f64,
        /// Allowed scale deviation.
        maximum: f64,
    },
    /// The transform's rotation was too far from a valid pier orientation.
    #[error(
        "rotation deviation {measured_degrees:.3}deg from the nearest normal or meridian-flipped orientation exceeds {maximum_degrees:.3}deg"
    )]
    Rotation {
        /// Measured rotation deviation, in degrees.
        measured_degrees: f64,
        /// Allowed rotation deviation, in degrees.
        maximum_degrees: f64,
    },
    /// Too little of the frame overlapped the reference.
    #[error("overlap fraction {measured:.3} is below {minimum:.3}")]
    InsufficientOverlap {
        /// Measured overlap fraction.
        measured: f32,
        /// Required overlap fraction.
        minimum: f32,
    },
    /// Background matching failed.
    #[error("normalization failed: {0}")]
    Normalization(String),
    /// A normalization gain fell outside the accepted range.
    #[error(
        "normalization gain range {measured_minimum:.3}..={measured_maximum:.3} is outside {minimum:.3}..={maximum:.3}"
    )]
    NormalizationGain {
        /// Smallest gain in the map.
        measured_minimum: f32,
        /// Largest gain in the map.
        measured_maximum: f32,
        /// Smallest accepted gain.
        minimum: f32,
        /// Largest accepted gain.
        maximum: f32,
    },
    /// Too few samples would survive rejection to be worth integrating.
    #[error("integrated sample fraction {measured:.3} is below {minimum:.3}")]
    InsufficientIntegratedSamples {
        /// Measured surviving fraction.
        measured: f32,
        /// Required surviving fraction.
        minimum: f32,
    },
}

/// The outcome of pushing one frame: admitted with diagnostics, or turned away
/// with a reason.
#[derive(Clone, Debug)]
pub enum FrameDisposition {
    /// The frame was integrated; carries its measurements.
    Accepted(FrameDiagnostics),
    /// The frame was turned away; carries why.
    Rejected(FrameRejectionReason),
}

/// A full copy of the current stack estimate and its coverage masks.
#[derive(Clone, Debug)]
pub struct StackSnapshot {
    /// Current mean image; zero-coverage samples are masked with `NaN`.
    pub image: LinearImage,
    /// Per-sample variance of the integrated observations.
    pub variance: LinearImage,
    /// Accepted observation count for every image sample.
    pub coverage: Vec<u32>,
    /// Rejected observation count for every image sample.
    pub rejected_samples: Vec<u32>,
    /// Number of frames admitted so far.
    pub accepted_frames: u32,
    /// Number of frames turned away so far.
    pub rejected_frames: u32,
}

/// A compact immutable copy of the current stack for non-destructive output.
///
/// Unlike [`StackSnapshot`], this owns only the finalized mean and scalar frame
/// counts. It deliberately omits variance and both per-sample count maps, so a
/// caller can hand it to an output worker while the live accumulator continues
/// without cloning four additional full-frame buffers.
#[derive(Clone, Debug)]
pub struct StackExportSnapshot {
    /// Current mean image; zero-coverage samples are masked with `NaN`.
    pub image: LinearImage,
    /// Number of frames admitted when the export snapshot was captured.
    pub accepted_frames: u32,
    /// Number of frames turned away when the export snapshot was captured.
    pub rejected_frames: u32,
}

/// Zero-copy access to the current online estimate. Samples with zero
/// coverage have an undefined mean and must be masked by `coverage`.
#[derive(Clone, Copy, Debug)]
pub struct StackView<'a> {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Channel count.
    pub channels: usize,
    /// Current running mean; mask by `coverage`.
    pub mean: &'a [f32],
    /// Accepted observation count for every sample.
    pub coverage: &'a [u32],
    /// Rejected observation count for every sample.
    pub rejected_samples: &'a [u32],
    /// Number of frames admitted so far.
    pub accepted_frames: u32,
    /// Number of frames turned away so far.
    pub rejected_frames: u32,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
/// Selects whether live-stack inputs are calibrated by the stacker or arrive
/// as already-prepared linear images.
pub enum FrameInputMode {
    /// File inputs are decoded, calibrated, and prepared before integration.
    #[default]
    CalibrateAndPrepare,
    /// The caller supplies prepared linear images; file inputs are rejected.
    PreparedOnly,
}

/// Incremental, bounded-memory image stack. Frames are registered to the
/// immutable first accepted frame and integrated immediately.
pub struct LiveStacker {
    // Constant for the duration of one pipelined batch, so `pipeline` may
    // share these across preparation threads while this thread holds
    // `&mut self` for the accumulator. `set_calibration` swaps the masters
    // between batches; the borrow checker keeps a swap out of a live batch.
    pub(crate) options: StackOptions,
    pub(crate) calibration: CalibrationMasters,
    pub(crate) reference: LinearImage,
    reference_metadata: FrameMetadata,
    pub(crate) registrar: Registrar,
    accumulator: Accumulator,
    reference_headers: Vec<(String, HeaderValue)>,
    accepted_frames: u32,
    rejected_frames: u32,
    input_paths: Vec<PathBuf>,
    input_mode: FrameInputMode,
    configuration_fingerprint: String,
}

impl LiveStacker {
    /// Start a stack from a reference FITS frame, calibrating and preparing it
    /// as the immutable alignment target.
    pub fn new(
        mut reference: FitsFrame,
        calibration: CalibrationMasters,
        options: StackOptions,
    ) -> Result<Self> {
        calibration.validate_light_frame(&reference)?;
        let reference_metadata = reference.metadata();
        calibration.apply(
            &mut reference.image,
            reference.exposure_seconds,
            reference.bayer,
        )?;
        if let Some(filter) = &options.cosmetic {
            crate::cosmetic::suppress_impulses(&mut reference.image, reference.bayer, filter)?;
        }
        let reference = reference.into_prepared()?;
        Self::from_prepared(
            reference.image,
            reference.headers,
            reference_metadata,
            calibration,
            options,
            FrameInputMode::CalibrateAndPrepare,
        )
    }

    /// Start a stack from an already-prepared linear reference, with no
    /// calibration and no header metadata. Every later frame must use
    /// [`Self::push_linear`].
    pub fn from_linear(reference: LinearImage, options: StackOptions) -> Result<Self> {
        let reference_metadata = FrameMetadata::from_image(&reference, &[]);
        Self::from_prepared(
            reference,
            Vec::new(),
            reference_metadata,
            CalibrationMasters::default(),
            options,
            FrameInputMode::PreparedOnly,
        )
    }

    /// Start a stack from a frame that a caller has already calibrated and
    /// prepared, while retaining its FITS headers for the output.
    ///
    /// This is the extension point for bounded corrections that must run
    /// between ordinary calibration and registration. A raw CFA frame is
    /// rejected so it cannot bypass preparation by mistake. Every later frame
    /// must use [`Self::push_linear`].
    pub fn from_prepared_frame(reference: FitsFrame, options: StackOptions) -> Result<Self> {
        if reference.bayer.is_some() {
            return Err(Error::Stack(
                "an already-prepared reference frame must not retain a Bayer layout".into(),
            ));
        }
        let reference_metadata = reference.metadata();
        Self::from_prepared(
            reference.image,
            reference.headers,
            reference_metadata,
            CalibrationMasters::default(),
            options,
            FrameInputMode::PreparedOnly,
        )
    }

    fn from_prepared(
        reference: LinearImage,
        reference_headers: Vec<(String, HeaderValue)>,
        reference_metadata: FrameMetadata,
        calibration: CalibrationMasters,
        options: StackOptions,
        input_mode: FrameInputMode,
    ) -> Result<Self> {
        options.validate()?;
        let configuration_fingerprint =
            stack_configuration_fingerprint(&options, &calibration, input_mode)?;
        let registrar = Registrar::new(&reference, options.registration.clone())?;
        let mut accumulator = Accumulator::new(reference.sample_count());
        accumulator.integrate(&reference.data, RejectionMode::None);
        Ok(Self {
            options,
            calibration,
            reference,
            reference_metadata,
            registrar,
            accumulator,
            reference_headers,
            accepted_frames: 1,
            rejected_frames: 0,
            input_paths: Vec::new(),
            input_mode,
            configuration_fingerprint,
        })
    }

    /// Start a stack from FITS or XISF paths and retain every source and
    /// calibration path for duplicate-input and output-path protection.
    pub fn open_fits(
        reference_path: impl AsRef<Path>,
        bias_path: Option<&Path>,
        dark_path: Option<&Path>,
        flat_path: Option<&Path>,
        dark_exposure_seconds: Option<f64>,
        options: StackOptions,
    ) -> Result<Self> {
        let reference_path = reference_path.as_ref();
        let input_paths = [Some(reference_path), bias_path, dark_path, flat_path]
            .into_iter()
            .flatten()
            .map(path_identity)
            .collect::<Vec<_>>();
        for (index, path) in input_paths.iter().enumerate() {
            if input_paths[..index]
                .iter()
                .any(|other| paths_refer_to_same_file(other, path))
            {
                return Err(Error::Stack(format!(
                    "stack input path {} is used more than once",
                    path.display()
                )));
            }
        }
        let calibration = CalibrationMasters::from_fits_paths(
            bias_path,
            dark_path,
            flat_path,
            dark_exposure_seconds,
        )?;
        let reference = FitsFrame::open(reference_path)?;
        let mut stacker = Self::new(reference, calibration, options)?;
        stacker.input_paths = input_paths;
        Ok(stacker)
    }

    /// Restore an atomically checkpointed live stack, including its immutable
    /// registration reference, calibration, online moments, and source ledger.
    pub fn open_context(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let restored = context::read(path)?;
        let registrar = Registrar::new(&restored.reference, restored.options.registration.clone())
            .map_err(|error| Error::StackContextRead {
                path: path.to_path_buf(),
                message: error.to_string(),
            })?;
        let configuration_fingerprint = stack_configuration_fingerprint(
            &restored.options,
            &restored.calibration,
            restored.input_mode,
        )?;
        Ok(Self {
            options: restored.options,
            calibration: restored.calibration,
            reference: restored.reference,
            reference_metadata: restored.reference_metadata,
            registrar,
            accumulator: Accumulator {
                mean: restored.mean,
                m2: restored.m2,
                count: restored.count,
                rejected: restored.rejected,
            },
            reference_headers: restored.reference_headers,
            accepted_frames: restored.accepted_frames,
            rejected_frames: restored.rejected_frames,
            input_paths: restored.input_paths,
            input_mode: restored.input_mode,
            configuration_fingerprint,
        })
    }

    /// Atomically checkpoint all state required to reopen this stack and keep
    /// integrating frames with identical online rejection behavior.
    pub fn save_context(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if self
            .input_paths
            .iter()
            .any(|input| paths_refer_to_same_file(input, path))
        {
            return Err(Error::StackContextWrite {
                path: path.to_path_buf(),
                message: "context path must not replace a stack input or calibration master".into(),
            });
        }
        context::write(
            path,
            context::ContextWriteState {
                options: &self.options,
                calibration: &self.calibration,
                reference: &self.reference,
                reference_headers: &self.reference_headers,
                reference_metadata: &self.reference_metadata,
                mean: &self.accumulator.mean,
                m2: &self.accumulator.m2,
                count: &self.accumulator.count,
                rejected: &self.accumulator.rejected,
                accepted_frames: self.accepted_frames,
                rejected_frames: self.rejected_frames,
                input_paths: &self.input_paths,
                input_mode: self.input_mode,
            },
        )
    }

    /// Replace the calibration masters applied to every frame pushed from
    /// now on.
    ///
    /// This is how a stack spanning several capture sessions calibrates each
    /// session with its own masters: push one session's frames as a batch,
    /// swap, push the next. Nothing already integrated is touched — the
    /// reference frame keeps the masters it was calibrated with at
    /// [`LiveStacker::new`], and a saved context records only the masters
    /// current at [`LiveStacker::save_context`] time, so a caller resuming a
    /// multi-session stack must call this again before pushing the next
    /// session's frames.
    ///
    /// The masters must fit the stack's geometry — dimensions and Bayer
    /// layout are checked against the registration reference eagerly, and a
    /// master missing its compatibility metadata is refused here once.
    ///
    /// What is deliberately NOT checked here is the reference frame's own
    /// acquisition signature. These masters calibrate the frames pushed
    /// while they are active, not the reference, which is already
    /// integrated. A multi-session stack swaps masters at each session
    /// boundary, and a later night's flat legitimately disagrees with the
    /// reference's rotator angle; judging it against the reference refused
    /// the swap and killed a hundred-frame stack at frame 45 over a flat
    /// that matched every frame it would actually touch. Each pushed light
    /// is validated against the active masters individually, which is the
    /// check that actually protects the pixels — and a light that fails it
    /// is rejected alone, never the stack. A stack started from prepared
    /// pixels refuses the call: its frames bypass calibration entirely.
    /// The calibration masters currently applied to pushed frames.
    ///
    /// Read access so a host can ask questions of the active set — which
    /// masters a prospective light could accept, and why not — without
    /// keeping its own copy in sync with every swap.
    pub fn calibration(&self) -> &CalibrationMasters {
        &self.calibration
    }

    pub fn set_calibration(&mut self, calibration: CalibrationMasters) -> Result<()> {
        self.require_fits_input_mode()?;
        calibration.validate_master_set_signatures()?;
        crate::context::validate_calibration(&self.reference, &calibration)
            .map_err(Error::Calibration)?;
        let configuration_fingerprint =
            stack_configuration_fingerprint(&self.options, &calibration, self.input_mode)?;
        self.calibration = calibration;
        self.configuration_fingerprint = configuration_fingerprint;
        Ok(())
    }

    /// Load and atomically replace the calibration masters used by later
    /// file inputs, retaining their paths in the stack's resumable input
    /// ledger.
    ///
    /// All supplied files are decoded and the complete set is validated
    /// against the registration reference before either the active masters or
    /// the path ledger changes. Passing no paths clears calibration. Existing
    /// integrated frames are never recalibrated.
    ///
    /// The same master may be selected again later: a multi-session stack can
    /// switch from one night's masters to another's and back without making a
    /// duplicate ledger entry. Supplied paths must still be distinct from one
    /// another within this call.
    pub fn set_calibration_from_fits_paths(
        &mut self,
        bias_path: Option<&Path>,
        dark_path: Option<&Path>,
        flat_path: Option<&Path>,
        dark_exposure_seconds: Option<f64>,
    ) -> Result<()> {
        self.require_fits_input_mode()?;
        if dark_path.is_none() && dark_exposure_seconds.is_some() {
            return Err(Error::Calibration(
                "a master-dark exposure override requires a dark path".into(),
            ));
        }
        if dark_exposure_seconds.is_some_and(|seconds| !seconds.is_finite() || seconds <= 0.0) {
            return Err(Error::Calibration(
                "master-dark exposure override must be a positive finite number".into(),
            ));
        }
        let paths = [bias_path, dark_path, flat_path]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        for (index, path) in paths.iter().enumerate() {
            if paths[..index]
                .iter()
                .any(|previous| paths_refer_to_same_file(path, previous))
            {
                return Err(Error::Calibration(format!(
                    "duplicate calibration input {}",
                    path.display()
                )));
            }
        }

        // Loading and validation happen before assignment. Once
        // `set_calibration` succeeds, recording identities cannot fail.
        let calibration = CalibrationMasters::from_fits_paths(
            bias_path,
            dark_path,
            flat_path,
            dark_exposure_seconds,
        )?;
        self.set_calibration(calibration)?;
        for path in paths {
            if !self.is_duplicate_input(path) {
                self.record_input_path(path);
            }
        }
        Ok(())
    }

    /// Calibrate, prepare, and try to integrate a FITS frame, reporting whether
    /// it was admitted or turned away. Stacks created from prepared pixels
    /// reject this path so later inputs cannot skip the caller's preparation.
    pub fn push(&mut self, mut frame: FitsFrame) -> Result<FrameDisposition> {
        self.require_fits_input_mode()?;
        if let Err(error) = self.calibration.validate_light_frame(&frame) {
            let message = match error {
                Error::Calibration(message) => message,
                other => other.to_string(),
            };
            return Ok(self.reject(FrameRejectionReason::Calibration(message)));
        }
        if let Err(error) =
            self.calibration
                .apply(&mut frame.image, frame.exposure_seconds, frame.bayer)
        {
            let message = match error {
                Error::Calibration(message) => message,
                other => other.to_string(),
            };
            return Ok(self.reject(FrameRejectionReason::Calibration(message)));
        }
        if let Some(filter) = &self.options.cosmetic
            && let Err(error) =
                crate::cosmetic::suppress_impulses(&mut frame.image, frame.bayer, filter)
        {
            return Ok(self.reject(FrameRejectionReason::Calibration(error.to_string())));
        }
        let frame = match frame.into_prepared() {
            Ok(frame) => frame,
            Err(error) => {
                return Ok(self.reject(FrameRejectionReason::IncompatibleImage(error.to_string())));
            }
        };
        self.push_linear(frame.image)
    }

    /// Open and offer one FITS or XISF path, rejecting duplicate source or
    /// calibration paths and retaining the path in resumable context state.
    /// Stacks created from prepared pixels reject this path.
    pub fn push_fits(&mut self, path: impl AsRef<Path>) -> Result<FrameDisposition> {
        self.require_fits_input_mode()?;
        let path = path.as_ref();
        if self.is_duplicate_input(path) {
            return Err(Error::Stack(format!(
                "FITS frame {} has already been used by this stack",
                path.display()
            )));
        }
        let frame = FitsFrame::open(path)?;
        let disposition = self.push(frame)?;
        self.record_input_path(path);
        Ok(disposition)
    }

    /// The identities of every path already taken, for a caller checking many
    /// candidates. One canonicalization each, rather than one per pair.
    pub(crate) fn input_identities(&self) -> std::collections::HashSet<PathBuf> {
        self.input_paths
            .iter()
            .map(|path| path_identity(path))
            .collect()
    }

    fn is_duplicate_input(&self, path: &Path) -> bool {
        self.input_paths
            .iter()
            .any(|input| paths_refer_to_same_file(input, path))
    }

    /// Retain a consumed path in resumable context state.
    pub(crate) fn record_input_path(&mut self, path: &Path) {
        self.input_paths.push(path_identity(path));
    }

    /// Register, normalize, and try to integrate an already-prepared linear
    /// frame, applying every admission gate.
    pub fn push_linear(&mut self, frame: LinearImage) -> Result<FrameDisposition> {
        let prepared = prepare_frame(&self.reference, &self.registrar, &self.options, frame)?;
        Ok(self.integrate_prepared(prepared))
    }

    /// Borrow the stack as two disjoint halves: the immutable state every
    /// frame's preparation reads, and the mutable state integration owns.
    ///
    /// This is what lets `pipeline` prepare frames on other threads while this
    /// thread integrates. The borrow checker enforces the split that makes the
    /// concurrency sound, rather than a comment promising it.
    pub(crate) fn split_for_pipeline(&mut self) -> (PreparationHalf<'_>, IntegrationHalf<'_>) {
        (
            PreparationHalf {
                reference: &self.reference,
                registrar: &self.registrar,
                calibration: &self.calibration,
                options: &self.options,
            },
            IntegrationHalf {
                accumulator: &mut self.accumulator,
                options: &self.options,
                accepted_frames: &mut self.accepted_frames,
                rejected_frames: &mut self.rejected_frames,
                input_paths: &mut self.input_paths,
            },
        )
    }

    /// Integrate what [`prepare_frame`] produced, in the caller's order.
    pub(crate) fn integrate_prepared(&mut self, prepared: PreparedFrame) -> FrameDisposition {
        let (_, mut integration) = self.split_for_pipeline();
        integration.integrate(prepared)
    }
    /// Copy the current estimate and coverage masks into an owned snapshot.
    pub fn snapshot(&self) -> Result<StackSnapshot> {
        let (mean, variance) = self.accumulator.snapshot();
        Ok(StackSnapshot {
            image: LinearImage::new(
                self.reference.width,
                self.reference.height,
                self.reference.channels,
                mean,
            )?,
            variance: LinearImage::new(
                self.reference.width,
                self.reference.height,
                self.reference.channels,
                variance,
            )?,
            coverage: self.accumulator.count.clone(),
            rejected_samples: self.accumulator.rejected.clone(),
            accepted_frames: self.accepted_frames,
            rejected_frames: self.rejected_frames,
        })
    }

    /// Copy only the state required to write an immutable stack image.
    ///
    /// The returned owner is independent of this stacker and may be moved to
    /// another thread. Capturing it copies one `f32` per image sample; it does
    /// not copy variance, coverage, or rejected-sample maps.
    pub fn export_snapshot(&self) -> Result<StackExportSnapshot> {
        Ok(StackExportSnapshot {
            image: LinearImage::new(
                self.reference.width,
                self.reference.height,
                self.reference.channels,
                self.accumulator.mean_snapshot(),
            )?,
            accepted_frames: self.accepted_frames,
            rejected_frames: self.rejected_frames,
        })
    }

    /// Borrow the current mean and masks without copying full-frame state.
    /// This is the preferred source for a live display renderer.
    pub fn view(&self) -> StackView<'_> {
        StackView {
            width: self.reference.width,
            height: self.reference.height,
            channels: self.reference.channels,
            mean: &self.accumulator.mean,
            coverage: &self.accumulator.count,
            rejected_samples: &self.accumulator.rejected,
            accepted_frames: self.accepted_frames,
            rejected_frames: self.rejected_frames,
        }
    }

    /// Return the identity processing mapping for the prepared reference
    /// frame. Callers can persist this beside mappings returned for later
    /// frames and use one extraction path for the whole stack.
    pub fn reference_mapping(&self) -> RegisteredFrameMapping {
        RegisteredFrameMapping::identity(&self.reference)
    }

    /// Consume the live state and move its full-frame buffers into a final
    /// snapshot. Batch callers should prefer this to avoid snapshot copies.
    pub fn into_snapshot(self) -> Result<StackSnapshot> {
        let (mean, variance, coverage, rejected_samples) = self.accumulator.into_snapshot();
        Ok(StackSnapshot {
            image: LinearImage::new(
                self.reference.width,
                self.reference.height,
                self.reference.channels,
                mean,
            )?,
            variance: LinearImage::new(
                self.reference.width,
                self.reference.height,
                self.reference.channels,
                variance,
            )?,
            coverage,
            rejected_samples,
            accepted_frames: self.accepted_frames,
            rejected_frames: self.rejected_frames,
        })
    }

    /// Header cards carried from the reference frame, for writing outputs.
    pub fn reference_headers(&self) -> &[(String, HeaderValue)] {
        &self.reference_headers
    }

    /// Normalized acquisition and calibration metadata of the immutable
    /// reference source.
    pub fn reference_metadata(&self) -> &FrameMetadata {
        &self.reference_metadata
    }

    /// Source and calibration paths already used by this stack.
    pub fn input_paths(&self) -> &[PathBuf] {
        &self.input_paths
    }

    /// Which kind of inputs this stack accepts after its reference.
    pub fn input_mode(&self) -> FrameInputMode {
        self.input_mode
    }

    /// Stable SHA-256 identity of the stack options, current calibration
    /// content, and input mode.
    ///
    /// The fingerprint deliberately excludes counters, accumulated pixels,
    /// and source paths. It therefore stays fixed while one compatible batch
    /// grows, changes when calibration is swapped, and recomputes to the same
    /// value after a context round trip.
    pub fn configuration_fingerprint(&self) -> &str {
        &self.configuration_fingerprint
    }

    pub(crate) fn require_fits_input_mode(&self) -> Result<()> {
        if self.input_mode == FrameInputMode::PreparedOnly {
            return Err(Error::Stack(
                "this stack was started from prepared pixels; use push_linear for every later frame"
                    .into(),
            ));
        }
        Ok(())
    }

    fn reject(&mut self, reason: FrameRejectionReason) -> FrameDisposition {
        self.rejected_frames += 1;
        FrameDisposition::Rejected(reason)
    }
}

fn stack_configuration_fingerprint(
    options: &StackOptions,
    calibration: &CalibrationMasters,
    input_mode: FrameInputMode,
) -> Result<String> {
    let options = serde_json::to_vec(options)
        .map_err(|error| Error::Stack(format!("failed to fingerprint stack options: {error}")))?;
    let mut hasher = Sha256::new();
    hasher.update(b"seiza-live-stack-configuration-v1\0");
    hasher.update([match input_mode {
        FrameInputMode::CalibrateAndPrepare => 0,
        FrameInputMode::PreparedOnly => 1,
    }]);
    hash_bytes(&mut hasher, &options);
    hash_optional_image(&mut hasher, calibration.bias.as_ref());
    hash_optional_signature(&mut hasher, calibration.bias_signature.as_ref())?;
    hash_optional_image(&mut hasher, calibration.dark_signal.as_ref());
    hash_optional_f64(&mut hasher, calibration.dark_exposure_seconds);
    hasher.update([u8::from(calibration.dark_scaling_safe)]);
    hash_optional_signature(&mut hasher, calibration.dark_signature.as_ref())?;
    hash_optional_bayer(&mut hasher, calibration.dark_bayer);
    hash_optional_image(&mut hasher, calibration.flat_response.as_ref());
    hash_optional_signature(&mut hasher, calibration.flat_signature.as_ref())?;
    hash_optional_bayer(&mut hasher, calibration.flat_bayer);
    let digest = hasher.finalize();
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn hash_optional_signature(
    hasher: &mut Sha256,
    signature: Option<&seiza_calibration::FrameSignature>,
) -> Result<()> {
    let Some(signature) = signature else {
        hasher.update([0]);
        return Ok(());
    };
    hasher.update([1]);
    let bytes = serde_json::to_vec(signature).map_err(|error| {
        Error::Stack(format!(
            "failed to fingerprint calibration metadata: {error}"
        ))
    })?;
    hash_bytes(hasher, &bytes);
    Ok(())
}

fn hash_bytes(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn hash_optional_image(hasher: &mut Sha256, image: Option<&LinearImage>) {
    let Some(image) = image else {
        hasher.update([0]);
        return;
    };
    hasher.update([1]);
    hasher.update((image.width as u64).to_le_bytes());
    hasher.update((image.height as u64).to_le_bytes());
    hasher.update((image.channels as u64).to_le_bytes());
    for sample in &image.data {
        hasher.update(sample.to_bits().to_le_bytes());
    }
}

fn hash_optional_f64(hasher: &mut Sha256, value: Option<f64>) {
    match value {
        Some(value) => {
            hasher.update([1]);
            hasher.update(value.to_bits().to_le_bytes());
        }
        None => hasher.update([0]),
    }
}

fn hash_optional_bayer(hasher: &mut Sha256, value: Option<BayerLayout>) {
    let Some(value) = value else {
        hasher.update([0]);
        return;
    };
    hasher.update([1]);
    hash_bytes(hasher, value.pattern.as_str().as_bytes());
    hasher.update((value.x_offset as u64).to_le_bytes());
    hasher.update((value.y_offset as u64).to_le_bytes());
}

/// A frame carried from preparation to integration.
///
/// Preparation reads only immutable stack state — the reference image, the
/// registrar's star catalogue, and the options — so it is a pure function of
/// the frame and can run for many frames at once. Integration is the part
/// that cannot.
pub(crate) enum PreparedFrame {
    /// Turned away by a gate that does not depend on the accumulator.
    Rejected(FrameRejectionReason),
    /// Registered and normalized, waiting its turn to be integrated.
    Ready(Box<ReadyFrame>),
}

/// A registered, normalized frame and the measurements taken along the way.
pub(crate) struct ReadyFrame {
    registered: LinearImage,
    transform: SimilarityTransform,
    matched_stars: usize,
    registration_rms_pixels: f64,
    registration_drift_pixels: f64,
    normalization_mean_gain: f32,
    normalization_mean_offset: f32,
    mapping: Box<crate::RegisteredFrameMapping>,
    overlap_fraction: f32,
}

/// Register and normalize one frame against the immutable reference.
///
/// Every gate applied here — channel count, registration quality, scale,
/// rotation, overlap, normalization gain — reads only the reference and the
/// options, so it reaches the same verdict whatever else is in flight. That is
/// what lets the pipeline prepare frames out of order and still match a
/// sequential run exactly.
pub(crate) fn prepare_frame(
    reference: &LinearImage,
    registrar: &Registrar,
    options: &StackOptions,
    frame: LinearImage,
) -> Result<PreparedFrame> {
    if reference.channels != frame.channels {
        return Ok(PreparedFrame::Rejected(
            FrameRejectionReason::IncompatibleImage(format!(
                "frame has {} channel(s) but stack has {}",
                frame.channels, reference.channels
            )),
        ));
    }
    let registration = match registrar.register(&frame) {
        Ok(registration) => registration,
        Err(error) => {
            let message = match error {
                Error::Registration(message) => message,
                other => other.to_string(),
            };
            return Ok(PreparedFrame::Rejected(FrameRejectionReason::Registration(
                message,
            )));
        }
    };
    let criteria = options.acceptance;
    if registration.rms_error_pixels > criteria.maximum_registration_rms_pixels {
        return Ok(PreparedFrame::Rejected(
            FrameRejectionReason::RegistrationRms {
                measured: registration.rms_error_pixels,
                maximum: criteria.maximum_registration_rms_pixels,
            },
        ));
    }
    let scale_deviation = (registration.transform.scale - 1.0).abs();
    if scale_deviation > criteria.maximum_scale_deviation {
        return Ok(PreparedFrame::Rejected(
            FrameRejectionReason::ScaleDeviation {
                measured: scale_deviation,
                maximum: criteria.maximum_scale_deviation,
            },
        ));
    }
    let rotation_deviation_degrees =
        rotation_deviation_degrees(registration.transform.rotation_radians);
    if rotation_deviation_degrees > criteria.maximum_rotation_degrees {
        return Ok(PreparedFrame::Rejected(FrameRejectionReason::Rotation {
            measured_degrees: rotation_deviation_degrees,
            maximum_degrees: criteria.maximum_rotation_degrees,
        }));
    }
    let mut registered = resample_to_reference(
        &frame,
        reference.width,
        reference.height,
        registration.transform,
    )?;
    let finite_samples = registered
        .data
        .par_iter()
        .filter(|value| value.is_finite())
        .count();
    let overlap_fraction = finite_samples as f32 / registered.sample_count() as f32;
    if overlap_fraction < criteria.minimum_overlap_fraction {
        return Ok(PreparedFrame::Rejected(
            FrameRejectionReason::InsufficientOverlap {
                measured: overlap_fraction,
                minimum: criteria.minimum_overlap_fraction,
            },
        ));
    }
    let normalization =
        match NormalizationMap::estimate(reference, &registered, options.normalization) {
            Ok(normalization) => normalization,
            Err(error) => {
                let message = match error {
                    Error::Normalization(message) => message,
                    other => other.to_string(),
                };
                return Ok(PreparedFrame::Rejected(
                    FrameRejectionReason::Normalization(message),
                ));
            }
        };
    let (minimum_gain, maximum_gain) = normalization.gain_range();
    if minimum_gain < criteria.minimum_normalization_gain
        || maximum_gain > criteria.maximum_normalization_gain
    {
        return Ok(PreparedFrame::Rejected(
            FrameRejectionReason::NormalizationGain {
                measured_minimum: minimum_gain,
                measured_maximum: maximum_gain,
                minimum: criteria.minimum_normalization_gain,
                maximum: criteria.maximum_normalization_gain,
            },
        ));
    }
    if !matches!(options.normalization, NormalizationMode::None)
        && let Err(error) = normalization.apply(&mut registered)
    {
        let message = match error {
            Error::Normalization(message) => message,
            other => other.to_string(),
        };
        return Ok(PreparedFrame::Rejected(
            FrameRejectionReason::Normalization(message),
        ));
    }
    let normalization_mean_gain = normalization.mean_gain();
    let normalization_mean_offset = normalization.mean_offset();
    let mapping = crate::RegisteredFrameMapping::new(
        reference.width,
        reference.height,
        registration.transform,
        normalization,
    )?;
    Ok(PreparedFrame::Ready(Box::new(ReadyFrame {
        registered,
        transform: registration.transform,
        matched_stars: registration.matched_stars,
        registration_rms_pixels: registration.rms_error_pixels,
        registration_drift_pixels: registration.drift_pixels,
        normalization_mean_gain,
        normalization_mean_offset,
        mapping: Box::new(mapping),
        overlap_fraction,
    })))
}

/// The immutable half of a stack: everything preparing a frame may read.
pub(crate) struct PreparationHalf<'a> {
    pub(crate) reference: &'a LinearImage,
    pub(crate) registrar: &'a Registrar,
    pub(crate) calibration: &'a CalibrationMasters,
    pub(crate) options: &'a StackOptions,
}

/// The mutable half of a stack: the accumulator and the run's tallies.
pub(crate) struct IntegrationHalf<'a> {
    accumulator: &'a mut Accumulator,
    options: &'a StackOptions,
    accepted_frames: &'a mut u32,
    rejected_frames: &'a mut u32,
    input_paths: &'a mut Vec<PathBuf>,
}

impl IntegrationHalf<'_> {
    /// Integrate one prepared frame. Must be called in submission order:
    /// whether a frame survives `minimum_integrated_fraction` depends on every
    /// frame integrated before it.
    pub(crate) fn integrate(&mut self, prepared: PreparedFrame) -> FrameDisposition {
        let ready = match prepared {
            PreparedFrame::Rejected(reason) => return self.reject(reason),
            PreparedFrame::Ready(ready) => ready,
        };
        let ReadyFrame {
            registered,
            transform,
            matched_stars,
            registration_rms_pixels,
            registration_drift_pixels,
            normalization_mean_gain,
            normalization_mean_offset,
            mapping,
            overlap_fraction,
        } = *ready;

        let (would_accept, _) = self
            .accumulator
            .classify(&registered.data, self.options.rejection);
        let integrated_fraction = would_accept as f32 / registered.sample_count() as f32;
        if integrated_fraction < self.options.acceptance.minimum_integrated_fraction {
            return self.reject(FrameRejectionReason::InsufficientIntegratedSamples {
                measured: integrated_fraction,
                minimum: self.options.acceptance.minimum_integrated_fraction,
            });
        }
        let (accepted_samples, rejected_samples) = self
            .accumulator
            .integrate(&registered.data, self.options.rejection);
        *self.accepted_frames += 1;
        FrameDisposition::Accepted(FrameDiagnostics {
            transform,
            matched_stars,
            registration_rms_pixels,
            registration_drift_pixels,
            normalization_mean_gain,
            normalization_mean_offset,
            mapping,
            overlap_fraction,
            integrated_fraction,
            accepted_samples,
            rejected_samples,
        })
    }

    fn reject(&mut self, reason: FrameRejectionReason) -> FrameDisposition {
        *self.rejected_frames += 1;
        FrameDisposition::Rejected(reason)
    }

    /// Retain a consumed path in resumable context state, given the identity
    /// the caller has already resolved. Canonicalizing is a filesystem call,
    /// and this runs on the one serial stage a pipeline waits on.
    pub(crate) fn record_input_identity(&mut self, identity: PathBuf) {
        self.input_paths.push(identity);
    }
}

struct Accumulator {
    mean: Vec<f32>,
    m2: Vec<f32>,
    count: Vec<u32>,
    rejected: Vec<u32>,
}

impl Accumulator {
    fn new(samples: usize) -> Self {
        Self {
            mean: vec![0.0; samples],
            m2: vec![0.0; samples],
            count: vec![0; samples],
            rejected: vec![0; samples],
        }
    }

    fn integrate(&mut self, samples: &[f32], rejection: RejectionMode) -> (usize, usize) {
        self.mean
            .par_iter_mut()
            .zip(self.m2.par_iter_mut())
            .zip(self.count.par_iter_mut())
            .zip(self.rejected.par_iter_mut())
            .zip(samples.par_iter())
            .map(|((((mean, m2), count), rejected), &sample)| {
                if !sample.is_finite() {
                    return (0, 0);
                }
                if should_reject_sample(*mean, *m2, *count, sample, rejection) {
                    *rejected = rejected.saturating_add(1);
                    return (0, 1);
                }
                let next_count = count.saturating_add(1);
                let delta = sample - *mean;
                *mean += delta / next_count as f32;
                let delta_after = sample - *mean;
                *m2 += delta * delta_after;
                *count = next_count;
                (1, 0)
            })
            .reduce(
                || (0, 0),
                |left, right| (left.0 + right.0, left.1 + right.1),
            )
    }

    fn classify(&self, samples: &[f32], rejection: RejectionMode) -> (usize, usize) {
        self.mean
            .par_iter()
            .zip(self.m2.par_iter())
            .zip(self.count.par_iter())
            .zip(samples.par_iter())
            .map(|(((mean, m2), count), &sample)| {
                if !sample.is_finite() {
                    (0, 0)
                } else if should_reject_sample(*mean, *m2, *count, sample, rejection) {
                    (0, 1)
                } else {
                    (1, 0)
                }
            })
            .reduce(
                || (0, 0),
                |left, right| (left.0 + right.0, left.1 + right.1),
            )
    }

    fn snapshot(&self) -> (Vec<f32>, Vec<f32>) {
        let mean = self.mean_snapshot();
        let variance = self
            .m2
            .iter()
            .zip(&self.count)
            .map(|(&m2, &count)| finalized_variance(m2, count))
            .collect();
        (mean, variance)
    }

    fn mean_snapshot(&self) -> Vec<f32> {
        self.mean
            .par_iter()
            .zip(self.count.par_iter())
            .map(|(&mean, &count)| finalized_mean(mean, count))
            .collect()
    }

    fn into_snapshot(mut self) -> (Vec<f32>, Vec<f32>, Vec<u32>, Vec<u32>) {
        for (mean, &count) in self.mean.iter_mut().zip(&self.count) {
            *mean = finalized_mean(*mean, count);
        }
        for (m2, &count) in self.m2.iter_mut().zip(&self.count) {
            *m2 = finalized_variance(*m2, count);
        }
        (self.mean, self.m2, self.count, self.rejected)
    }
}

/// A sample never observed has an undefined mean; mask it so downstream
/// renderers can drop it by coverage.
fn finalized_mean(mean: f32, count: u32) -> f32 {
    if count == 0 { f32::NAN } else { mean }
}

/// Convert Welford's running sum of squares into the sample variance, which
/// needs at least two observations.
fn finalized_variance(m2: f32, count: u32) -> f32 {
    if count > 1 {
        m2 / (count - 1) as f32
    } else {
        0.0
    }
}

fn should_reject_sample(
    mean: f32,
    m2: f32,
    count: u32,
    sample: f32,
    rejection: RejectionMode,
) -> bool {
    match rejection {
        RejectionMode::None => false,
        RejectionMode::DeltaSigma(options) if count >= options.warmup_samples && count > 1 => {
            let sigma = (m2 / (count - 1) as f32).sqrt().max(options.minimum_sigma);
            let delta = sample - mean;
            delta < -options.low_sigma * sigma || delta > options.high_sigma * sigma
        }
        RejectionMode::DeltaSigma(_) => false,
    }
}

/// Angular distance from the closest valid German-equatorial-mount pier
/// orientation. A meridian flip rotates the camera by 180 degrees, so a
/// transform near either zero or half a turn has the same admission error.
fn rotation_deviation_degrees(rotation_radians: f64) -> f64 {
    let modulo_half_turn = rotation_radians.to_degrees().rem_euclid(180.0);
    modulo_half_turn.min(180.0 - modulo_half_turn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BayerLayout;

    fn stacking_star_field(width: usize, height: usize) -> LinearImage {
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
        let mut data = Vec::with_capacity(width * height);
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
                data.push(value);
            }
        }
        LinearImage::new(width, height, 1, data).unwrap()
    }

    fn offset_image(reference: &LinearImage, offset: f32) -> LinearImage {
        LinearImage::new(
            reference.width,
            reference.height,
            reference.channels,
            reference.data.iter().map(|value| value + offset).collect(),
        )
        .unwrap()
    }

    #[test]
    fn delta_sigma_rejects_late_outlier_without_moving_mean() {
        let mut accumulator = Accumulator::new(1);
        let rejection = RejectionMode::DeltaSigma(DeltaSigmaOptions {
            warmup_samples: 4,
            low_sigma: 3.0,
            high_sigma: 3.0,
            minimum_sigma: 0.01,
        });
        for value in [10.0, 10.1, 9.9, 10.05] {
            accumulator.integrate(&[value], rejection);
        }
        let before = accumulator.mean[0];
        let (_, rejected) = accumulator.integrate(&[1000.0], rejection);
        assert_eq!(rejected, 1);
        assert_eq!(accumulator.count[0], 4);
        assert_eq!(accumulator.mean[0], before);
    }

    #[test]
    fn export_snapshot_owns_only_a_frozen_finalized_mean() {
        let mut accumulator = Accumulator::new(2);
        accumulator.integrate(&[5.0, f32::NAN], RejectionMode::None);
        let mean = accumulator.mean_snapshot();
        assert_eq!(mean[0], 5.0);
        assert!(mean[1].is_nan(), "zero coverage must be finalized as NaN");

        let reference = stacking_star_field(160, 128);
        let mut stacker = LiveStacker::from_linear(
            reference.clone(),
            StackOptions {
                normalization: NormalizationMode::None,
                rejection: RejectionMode::None,
                ..StackOptions::default()
            },
        )
        .unwrap();
        let export = stacker.export_snapshot().unwrap();
        assert_eq!(export.image.data, reference.data);
        assert_eq!(export.accepted_frames, 1);
        assert_eq!(export.rejected_frames, 0);

        assert!(matches!(
            stacker.push_linear(offset_image(&reference, 10.0)).unwrap(),
            FrameDisposition::Accepted(_)
        ));
        assert_eq!(export.image.data, reference.data, "the export is immutable");
        assert_eq!(export.accepted_frames, 1);
        assert_eq!(stacker.view().accepted_frames, 2);
        assert!(
            stacker
                .export_snapshot()
                .unwrap()
                .image
                .data
                .iter()
                .zip(&export.image.data)
                .any(|(current, frozen)| current != frozen)
        );
    }

    #[test]
    fn meridian_flip_rotation_is_measured_from_half_a_turn() {
        assert!(rotation_deviation_degrees(179.307_f64.to_radians()) < 0.7);
        assert!(rotation_deviation_degrees((-179.307_f64).to_radians()) < 0.7);
        assert!((rotation_deviation_degrees(12.0_f64.to_radians()) - 12.0).abs() < 1.0e-10);
        assert!((rotation_deviation_degrees(90.0_f64.to_radians()) - 90.0).abs() < 1.0e-10);
    }

    #[test]
    fn rejects_invalid_online_options_before_allocating_state() {
        let options = StackOptions {
            rejection: RejectionMode::DeltaSigma(DeltaSigmaOptions {
                warmup_samples: 1,
                ..DeltaSigmaOptions::default()
            }),
            ..StackOptions::default()
        };
        assert!(options.validate().is_err());
    }

    #[test]
    fn stack_options_support_partial_json_and_reject_unknown_fields() {
        let options: StackOptions = serde_json::from_str(
            r#"{
                "registration": {"maximum_drift_pixels": 512.0},
                "normalization": {"mode": "local", "options": {"tile_size": 128}},
                "rejection": {"mode": "none"},
                "acceptance": {"minimum_overlap_fraction": 0.75}
            }"#,
        )
        .unwrap();
        assert_eq!(options.registration.maximum_drift_pixels, 512.0);
        assert_eq!(
            options.normalization,
            NormalizationMode::Local { tile_size: 128 }
        );
        assert!(matches!(options.rejection, RejectionMode::None));
        assert_eq!(options.acceptance.minimum_overlap_fraction, 0.75);
        assert_eq!(options.registration.maximum_stars, 200);
        options.validate().unwrap();

        let json = serde_json::to_string(&options).unwrap();
        let round_trip: StackOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(round_trip.registration.maximum_drift_pixels, 512.0);
        assert!(serde_json::from_str::<StackOptions>(r#"{"mystery": true}"#).is_err());
    }

    #[test]
    fn prepared_frame_constructor_retains_headers_and_rejects_raw_cfa() {
        let image = stacking_star_field(160, 128);
        let frame = FitsFrame {
            image: image.clone(),
            headers: vec![("OBJECT".into(), HeaderValue::String("M 31".into()))],
            exposure_seconds: Some(60.0),
            bayer: None,
            source: None,
            bounds: None,
        };
        let mut stacker = LiveStacker::from_prepared_frame(frame, StackOptions::default()).unwrap();
        assert_eq!(
            stacker.reference_headers(),
            [("OBJECT".into(), HeaderValue::String("M 31".into()))]
        );

        let standard_push = FitsFrame {
            image: image.clone(),
            headers: Vec::new(),
            exposure_seconds: None,
            bayer: None,
            source: None,
            bounds: None,
        };
        let error = stacker.push(standard_push).unwrap_err().to_string();
        assert!(error.contains("use push_linear"), "{error}");
        let error = stacker
            .push_fits("does-not-need-to-exist.fits")
            .unwrap_err()
            .to_string();
        assert!(error.contains("use push_linear"), "{error}");
        assert_eq!(stacker.view().accepted_frames, 1);
        assert_eq!(stacker.view().rejected_frames, 0);
        assert!(matches!(
            stacker.push_linear(image.clone()).unwrap(),
            FrameDisposition::Accepted(_)
        ));

        let raw = FitsFrame {
            image,
            headers: Vec::new(),
            exposure_seconds: None,
            bayer: Some(BayerLayout {
                pattern: seiza_fits::BayerPattern::Rggb,
                x_offset: 0,
                y_offset: 0,
            }),
            source: None,
            bounds: None,
        };
        assert!(LiveStacker::from_prepared_frame(raw, StackOptions::default()).is_err());
    }

    #[test]
    fn context_resume_is_identical_to_uninterrupted_online_integration() {
        let reference = stacking_star_field(160, 128);
        let options = StackOptions {
            normalization: NormalizationMode::None,
            rejection: RejectionMode::DeltaSigma(DeltaSigmaOptions {
                warmup_samples: 4,
                minimum_sigma: 0.01,
                ..DeltaSigmaOptions::default()
            }),
            ..StackOptions::default()
        };
        let mut frames = [0.10, -0.10, 0.05, -0.05]
            .map(|offset| offset_image(&reference, offset))
            .to_vec();
        let mut partial_outlier = offset_image(&reference, 0.0);
        let center = partial_outlier.height / 2 * partial_outlier.width + partial_outlier.width / 2;
        partial_outlier.data[center] += 1_000.0;
        frames.push(partial_outlier);
        frames.push(offset_image(&reference, 0.02));
        let mut uninterrupted =
            LiveStacker::from_linear(reference.clone(), options.clone()).unwrap();
        for frame in frames.iter().cloned() {
            uninterrupted.push_linear(frame).unwrap();
        }

        let mut checkpointed = LiveStacker::from_linear(reference, options).unwrap();
        for frame in frames[..3].iter().cloned() {
            checkpointed.push_linear(frame).unwrap();
        }
        checkpointed
            .input_paths
            .push(PathBuf::from("light-001.fits"));
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("session.seiza-stack");
        checkpointed.save_context(&path).unwrap();
        let mut resumed = LiveStacker::open_context(&path).unwrap();
        assert_eq!(resumed.input_paths(), [PathBuf::from("light-001.fits")]);
        let standard_push = FitsFrame {
            image: frames[0].clone(),
            headers: Vec::new(),
            exposure_seconds: None,
            bayer: None,
            source: None,
            bounds: None,
        };
        let error = resumed.push(standard_push).unwrap_err().to_string();
        assert!(error.contains("use push_linear"), "{error}");
        for frame in frames[3..].iter().cloned() {
            resumed.push_linear(frame).unwrap();
        }
        resumed.save_context(&path).unwrap();
        let resumed = LiveStacker::open_context(&path).unwrap();

        let expected = uninterrupted.into_snapshot().unwrap();
        let actual = resumed.into_snapshot().unwrap();
        assert_eq!(actual.image.data, expected.image.data);
        assert_eq!(actual.variance.data, expected.variance.data);
        assert_eq!(actual.coverage, expected.coverage);
        assert_eq!(actual.rejected_samples, expected.rejected_samples);
        assert!(actual.rejected_samples.iter().sum::<u32>() > 0);
        assert_eq!(actual.accepted_frames, expected.accepted_frames);
        assert_eq!(actual.rejected_frames, expected.rejected_frames);
    }

    #[test]
    fn legacy_contexts_open_but_fail_closed_until_masters_are_reloaded() {
        let image = stacking_star_field(160, 128);
        let frame = || FitsFrame {
            image: image.clone(),
            headers: vec![("IMAGETYP".into(), HeaderValue::String("LIGHT".into()))],
            exposure_seconds: Some(60.0),
            bayer: None,
            source: None,
            bounds: None,
        };
        let calibration = CalibrationMasters::new(
            Some(LinearImage::new(160, 128, 1, vec![2.0; 160 * 128]).unwrap()),
            None,
            None,
        )
        .unwrap();
        let stacker =
            LiveStacker::new(frame(), calibration.clone(), StackOptions::default()).unwrap();
        let original_fingerprint = stacker.configuration_fingerprint().to_owned();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("legacy-v1.seiza-stack");
        context::write_legacy_v1(
            &path,
            context::ContextWriteState {
                options: &stacker.options,
                calibration: &stacker.calibration,
                reference: &stacker.reference,
                reference_headers: &stacker.reference_headers,
                reference_metadata: &stacker.reference_metadata,
                mean: &stacker.accumulator.mean,
                m2: &stacker.accumulator.m2,
                count: &stacker.accumulator.count,
                rejected: &stacker.accumulator.rejected,
                accepted_frames: stacker.accepted_frames,
                rejected_frames: stacker.rejected_frames,
                input_paths: &stacker.input_paths,
                input_mode: stacker.input_mode,
            },
        )
        .unwrap();

        let mut restored = LiveStacker::open_context(&path).unwrap();
        assert_ne!(
            restored.configuration_fingerprint(),
            original_fingerprint,
            "missing v1 signatures are part of the migrated identity"
        );
        let rejected = restored.push(frame()).unwrap();
        assert!(matches!(
            rejected,
            FrameDisposition::Rejected(FrameRejectionReason::Calibration(ref reason))
                if reason.contains("reload calibration masters")
        ));

        restored.set_calibration(calibration).unwrap();
        assert!(matches!(
            restored.push(frame()).unwrap(),
            FrameDisposition::Accepted(_)
        ));
    }

    #[test]
    fn truncated_context_is_rejected() {
        let reference = stacking_star_field(160, 128);
        let stacker = LiveStacker::from_linear(reference, StackOptions::default()).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("session.seiza-stack");
        stacker.save_context(&path).unwrap();
        let length = std::fs::metadata(&path).unwrap().len();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(length - 1)
            .unwrap();
        assert!(matches!(
            LiveStacker::open_context(&path),
            Err(Error::StackContextRead { .. })
        ));
    }

    #[test]
    fn cosmetic_correction_cleans_a_hot_pixel_from_reference_and_pushed_frames() {
        // Same field twice, both with the same defective pixel — exactly a
        // sensor defect with no dark master to subtract it. Both the
        // reference path (`new`) and the push path must clean it, and the
        // stars must survive the filter untouched enough to register.
        let hot = 90 * 160 + 60;
        let frame = |exposure| {
            let mut image = stacking_star_field(160, 128);
            image.data[hot] = 60_000.0;
            FitsFrame {
                image,
                headers: Vec::new(),
                exposure_seconds: Some(exposure),
                bayer: None,
                source: None,
                bounds: None,
            }
        };
        let options = StackOptions {
            cosmetic: Some(crate::cosmetic::ImpulseFilterOptions::default()),
            ..StackOptions::default()
        };
        let mut stacker =
            LiveStacker::new(frame(60.0), CalibrationMasters::default(), options).unwrap();
        assert!(matches!(
            stacker.push(frame(60.0)).unwrap(),
            FrameDisposition::Accepted(_)
        ));
        let snapshot = stacker.snapshot().unwrap();
        assert!(
            snapshot.image.data[hot] < 200.0,
            "the defect must be gone from the integration: {}",
            snapshot.image.data[hot]
        );
    }

    #[test]
    fn context_preserves_calibration_headers_and_source_ledger() {
        let reference = stacking_star_field(160, 128);
        let calibration_image = LinearImage::new(160, 128, 1, vec![2.0; 160 * 128]).unwrap();
        let mut stacker = LiveStacker::from_linear(reference, StackOptions::default()).unwrap();
        let bayer = BayerLayout {
            pattern: seiza_fits::BayerPattern::Rggb,
            x_offset: 1,
            y_offset: 0,
        };
        stacker.calibration = CalibrationMasters::new(
            Some(calibration_image.clone()),
            Some(crate::MasterDark {
                image: calibration_image.clone(),
                exposure_seconds: Some(300.0),
                bias_subtracted: false,
                bayer: Some(bayer),
            }),
            Some(crate::MasterFlat::raw_with_bayer(
                LinearImage::new(160, 128, 1, vec![4.0; 160 * 128]).unwrap(),
                bayer,
            )),
        )
        .unwrap();
        stacker.reference_headers = vec![
            ("OBJECT".into(), HeaderValue::String("M 31".into())),
            ("ODDVAL".into(), HeaderValue::Float(f64::NAN)),
        ];
        stacker.input_paths = vec![PathBuf::from("reference.fits"), PathBuf::from("dark.fits")];
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("session.seiza-stack");
        stacker.save_context(&path).unwrap();

        let restored = LiveStacker::open_context(&path).unwrap();
        assert_eq!(
            restored.calibration.bias.unwrap().data,
            vec![2.0; 160 * 128]
        );
        assert_eq!(restored.calibration.dark_exposure_seconds, Some(300.0));
        assert_eq!(
            restored.calibration.dark_bayer.unwrap().pattern,
            seiza_fits::BayerPattern::Rggb
        );
        assert_eq!(
            restored.reference_headers[0],
            ("OBJECT".into(), HeaderValue::String("M 31".into()))
        );
        assert!(matches!(
            restored.reference_headers[1].1,
            HeaderValue::Float(value) if value.is_nan()
        ));
        assert_eq!(stacker.input_paths, restored.input_paths);
    }

    #[test]
    fn path_calibration_swap_is_atomic_updates_the_ledger_and_fingerprints() {
        let directory = tempfile::tempdir().unwrap();
        let reference_path = directory.path().join("reference.fits");
        let bias_path = directory.path().join("master-bias.fits");
        let wrong_bias_path = directory.path().join("wrong-master-bias.fits");
        let context_path = directory.path().join("live.seiza-stack");
        let reference = stacking_star_field(160, 128);
        let bias = LinearImage::new(160, 128, 1, vec![2.0; 160 * 128]).unwrap();
        let wrong_bias = LinearImage::new(80, 64, 1, vec![3.0; 80 * 64]).unwrap();
        crate::write_processed_image_fits_f32(&reference_path, &reference, &[], &[]).unwrap();
        crate::write_processed_image_fits_f32(&bias_path, &bias, &[], &[]).unwrap();
        crate::write_processed_image_fits_f32(&wrong_bias_path, &wrong_bias, &[], &[]).unwrap();

        let mut stacker = LiveStacker::open_fits(
            &reference_path,
            None,
            None,
            None,
            None,
            StackOptions::default(),
        )
        .unwrap();
        assert_eq!(stacker.input_mode(), FrameInputMode::CalibrateAndPrepare);
        let empty_fingerprint = stacker.configuration_fingerprint().to_owned();
        stacker
            .set_calibration_from_fits_paths(Some(&bias_path), None, None, None)
            .unwrap();
        let calibrated_fingerprint = stacker.configuration_fingerprint().to_owned();
        assert_ne!(calibrated_fingerprint, empty_fingerprint);
        assert_eq!(calibrated_fingerprint.len(), 64);
        assert!(
            stacker
                .input_paths()
                .iter()
                .any(|path| paths_refer_to_same_file(path, &bias_path))
        );
        let paths_before_failure = stacker.input_paths().to_vec();
        let bias_before_failure = stacker.calibration.bias.clone().unwrap();

        assert!(
            stacker
                .set_calibration_from_fits_paths(Some(&wrong_bias_path), None, None, None)
                .is_err()
        );
        assert_eq!(stacker.configuration_fingerprint(), calibrated_fingerprint);
        assert_eq!(stacker.input_paths(), paths_before_failure);
        assert_eq!(
            stacker.calibration.bias.as_ref(),
            Some(&bias_before_failure)
        );

        // Selecting the same master again neither fails nor duplicates it.
        stacker
            .set_calibration_from_fits_paths(Some(&bias_path), None, None, None)
            .unwrap();
        assert_eq!(stacker.input_paths(), paths_before_failure);
        stacker.save_context(&context_path).unwrap();
        let restored = LiveStacker::open_context(&context_path).unwrap();
        assert_eq!(restored.configuration_fingerprint(), calibrated_fingerprint);
        assert_eq!(restored.input_paths(), paths_before_failure);

        let mut stacker = restored;
        stacker
            .set_calibration_from_fits_paths(None, None, None, None)
            .unwrap();
        assert_eq!(stacker.configuration_fingerprint(), empty_fingerprint);
        // Clearing calibration does not erase history needed for safe output.
        assert_eq!(stacker.input_paths(), paths_before_failure);
    }

    #[test]
    fn prepared_stack_refuses_path_calibration_without_mutation() {
        let reference = stacking_star_field(160, 128);
        let mut stacker = LiveStacker::from_linear(reference, StackOptions::default()).unwrap();
        let fingerprint = stacker.configuration_fingerprint().to_owned();
        let error = stacker
            .set_calibration_from_fits_paths(None, None, None, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("prepared pixels"), "{error}");
        assert_eq!(stacker.configuration_fingerprint(), fingerprint);
        assert!(stacker.input_paths().is_empty());
    }
}
