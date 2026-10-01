use crate::{
    BayerLayout, CalibrationMasters, Error, FitsFrame, FrameMetadata, LinearImage,
    NormalizationMap, NormalizationMode, ReferenceRegion, RegisteredFrameMapping, Registrar,
    RegistrationOptions, Result, SimilarityTransform, context, path_identity,
    paths_refer_to_same_file,
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

/// How much each admitted frame counts toward the stack mean.
///
/// The default, [`FrameWeighting::Equal`], gives every frame the same weight
/// and is exactly the behaviour of earlier releases. It is omitted from
/// serialized [`StackOptions`], so it leaves the configuration fingerprint and
/// saved contexts unchanged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "kebab-case", deny_unknown_fields)]
pub enum FrameWeighting {
    /// Every admitted frame has weight 1.
    #[default]
    Equal,
    /// Weight each frame by the inverse of its noise variance, relative to the
    /// reference frame.
    ///
    /// Each channel's noise is measured on the calibrated frame before
    /// resampling and scaled by that channel's normalization gain, so it is in
    /// the units the frame is stacked in. The weight is
    /// `(reference_noise / frame_noise)^2`, clamped to
    /// `minimum_weight..=maximum_weight`. The reference frame therefore has
    /// weight 1, and a frame twice as noisy as the reference has weight 0.25.
    ///
    /// In this mode the stack's variance output is the variance of a frame
    /// with weight 1 (a frame as noisy as the reference), not the plain
    /// sample variance.
    InverseNoiseVariance {
        /// Smallest weight a frame can receive. Must be in `(0, 1]`.
        #[serde(default = "default_minimum_weight")]
        minimum_weight: f32,
        /// Largest weight a frame can receive. Must be at least 1.
        #[serde(default = "default_maximum_weight")]
        maximum_weight: f32,
    },
}

fn default_minimum_weight() -> f32 {
    0.05
}

fn default_maximum_weight() -> f32 {
    20.0
}

impl FrameWeighting {
    /// Inverse-noise-variance weighting with the default bounds: 0.05 to 20.
    pub fn inverse_noise_variance() -> Self {
        Self::InverseNoiseVariance {
            minimum_weight: default_minimum_weight(),
            maximum_weight: default_maximum_weight(),
        }
    }

    /// Whether every frame has the same weight.
    pub fn is_equal(&self) -> bool {
        matches!(self, Self::Equal)
    }

    /// Check that the weight bounds are finite and bracket 1.
    pub fn validate(&self) -> Result<()> {
        if let Self::InverseNoiseVariance {
            minimum_weight,
            maximum_weight,
        } = *self
            && !(minimum_weight.is_finite()
                && maximum_weight.is_finite()
                && minimum_weight > 0.0
                && minimum_weight <= 1.0
                && maximum_weight >= 1.0)
        {
            return Err(Error::Stack(
                "frame weight bounds must be finite with 0 < minimum <= 1 <= maximum".into(),
            ));
        }
        Ok(())
    }

    /// The weight of a frame whose normalized noise is `frame_noise`, given
    /// the reference frame's noise in the same channel. `None` for
    /// [`FrameWeighting::Equal`].
    fn weight(&self, reference_noise: f32, frame_noise: f32) -> Option<f32> {
        let Self::InverseNoiseVariance {
            minimum_weight,
            maximum_weight,
        } = *self
        else {
            return None;
        };
        let ratio = f64::from(reference_noise) / f64::from(frame_noise);
        let weight = if ratio.is_finite() {
            (ratio * ratio) as f32
        } else {
            maximum_weight
        };
        Some(weight.clamp(minimum_weight, maximum_weight))
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
    /// How much each admitted frame counts toward the mean. The default,
    /// [`FrameWeighting::Equal`], is not serialized, so existing options,
    /// fingerprints and contexts keep their exact bytes.
    #[serde(default, skip_serializing_if = "FrameWeighting::is_equal")]
    pub weighting: FrameWeighting,
    /// How a Bayer frame's colors reach the accumulator. The default,
    /// [`CfaIntegration::Demosaic`], is not serialized, so existing options,
    /// fingerprints and contexts keep their exact bytes.
    #[serde(default, skip_serializing_if = "CfaIntegration::is_demosaic")]
    pub cfa_integration: CfaIntegration,
    /// How registration resamples each frame onto the reference grid. The
    /// default, [`Interpolation::Bilinear`], is not serialized, so existing
    /// options, fingerprints and contexts keep their exact bytes.
    #[serde(default, skip_serializing_if = "crate::Interpolation::is_bilinear")]
    pub interpolation: crate::Interpolation,
}

/// How a Bayer frame's colors reach the accumulator. Registration and
/// normalization use the demosaiced frame either way.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CfaIntegration {
    /// Integrate the demosaiced frame, every channel at every pixel.
    #[default]
    Demosaic,
    /// Bayer drizzle: each registered pixel takes only the nearest source
    /// photosite, in the one channel that photosite records, and the stack
    /// fills the colors in from frames that land on different photosites.
    ///
    /// Nothing is interpolated, so stars keep their sharpness and colors
    /// keep their place, but each channel sees a third to a quarter of the
    /// samples. It pays when frames are dithered by several pixels; on a
    /// 98-frame M45 stack that drifted only 10 to 20 pixels in a night it
    /// gave lower SNR than demosaicing. Channels no frame reached at a pixel
    /// are filled from the same channel's neighbours in snapshots, with
    /// their coverage left at zero. Frames without a Bayer layout integrate
    /// as demosaiced frames would.
    BayerDrizzle,
}

impl CfaIntegration {
    /// Whether this is the default, which options leave unserialized.
    pub fn is_demosaic(&self) -> bool {
        *self == Self::Demosaic
    }
}

impl StackOptions {
    /// Validate registration, normalization, rejection, weighting, and
    /// admission bounds.
    pub fn validate(&self) -> Result<()> {
        self.registration.validate()?;
        self.weighting.validate()?;
        if matches!(
            self.normalization,
            NormalizationMode::Local { tile_size } | NormalizationMode::LocalBackground { tile_size }
                if tile_size < 16
        ) {
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
    /// Pixel-scale noise of each channel, measured on the calibrated frame
    /// before resampling and scaled by the channel's normalization gain.
    /// Empty when [`StackOptions::weighting`] is [`FrameWeighting::Equal`],
    /// because noise is then not measured.
    pub noise: Vec<f32>,
    /// Weight of each channel in the stack mean. Empty when
    /// [`StackOptions::weighting`] is [`FrameWeighting::Equal`], where every
    /// frame has weight 1. [`LiveStacker::reintegrate`] replays with these
    /// automatically. To replay frames yourself, persist them (the reference
    /// frame has weight 1 in every channel) and pass them as
    /// [`crate::BatchStackOptions::frame_weights`].
    pub weight: Vec<f32>,
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
    /// Per-sample variance of the integrated observations. With frame
    /// weighting it is the variance of a weight-1 frame.
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
    /// Per-channel noise of the reference frame, the unit of frame weights.
    /// Empty when frames are weighted equally.
    reference_noise: Vec<f32>,
    accumulator: Accumulator,
    reference_headers: Vec<(String, HeaderValue)>,
    pub(crate) accepted_frames: u32,
    pub(crate) rejected_frames: u32,
    input_paths: Vec<PathBuf>,
    input_mode: FrameInputMode,
    configuration_fingerprint: String,
    pub(crate) ledger: crate::replay::Ledger,
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
        let source = crate::replay::FrameSource::of(&reference);
        let reference_metadata = reference.metadata();
        calibration.apply(
            &mut reference.image,
            reference.exposure_seconds,
            reference.bayer,
        )?;
        if let Some(filter) = &options.cosmetic {
            crate::cosmetic::suppress_impulses(&mut reference.image, reference.bayer, filter)?;
        }
        let (reference, cfa) = reference.into_prepared_with_layout()?;
        let mut stacker = Self::from_prepared(
            reference.image,
            reference.headers,
            reference_metadata,
            calibration,
            options,
            FrameInputMode::CalibrateAndPrepare,
            cfa,
        )?;
        stacker.ledger.set_reference_source(source);
        Ok(stacker)
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
            None,
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
            None,
        )
    }

    fn from_prepared(
        reference: LinearImage,
        reference_headers: Vec<(String, HeaderValue)>,
        reference_metadata: FrameMetadata,
        calibration: CalibrationMasters,
        options: StackOptions,
        input_mode: FrameInputMode,
        reference_cfa: Option<BayerLayout>,
    ) -> Result<Self> {
        options.validate()?;
        let configuration_fingerprint =
            stack_configuration_fingerprint(&options, &calibration, input_mode)?;
        let registrar = Registrar::new(&reference, options.registration.clone())?;
        let reference_noise = if options.weighting.is_equal() {
            Vec::new()
        } else {
            measure_reference_noise(&reference)?
        };
        let mut accumulator =
            Accumulator::new(reference.sample_count(), !options.weighting.is_equal());
        // The reference is the unit of weight: it integrates with weight 1.
        // Under Bayer drizzle it contributes its own photosites, like every
        // later frame, while the demosaiced image stays the alignment target.
        match reference_cfa.filter(|_| options.cfa_integration == CfaIntegration::BayerDrizzle) {
            Some(layout) => {
                let photosites = crate::registration::resample_region_photosites(
                    &reference,
                    reference.width,
                    reference.height,
                    ReferenceRegion {
                        x: 0,
                        y: 0,
                        width: reference.width,
                        height: reference.height,
                    },
                    SimilarityTransform::IDENTITY,
                    layout,
                )?;
                accumulator.integrate(&photosites.data, RejectionMode::None, None);
            }
            None => {
                accumulator.integrate(&reference.data, RejectionMode::None, None);
            }
        }
        let mut ledger = crate::replay::Ledger::new(&reference);
        if !reference_noise.is_empty() {
            ledger.set_reference_weighting(&reference_noise);
        }
        Ok(Self {
            options,
            calibration,
            reference,
            reference_metadata,
            registrar,
            reference_noise,
            accumulator,
            reference_headers,
            accepted_frames: 1,
            rejected_frames: 0,
            input_paths: Vec::new(),
            input_mode,
            configuration_fingerprint,
            ledger,
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
        stacker
            .ledger
            .set_current_calibration(crate::replay::CalibrationRecord::from_paths(
                bias_path,
                dark_path,
                flat_path,
                dark_exposure_seconds,
            ));
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
            reference_noise: restored.reference_noise,
            accumulator: Accumulator {
                mean: restored.mean,
                m2: restored.m2,
                count: restored.count,
                rejected: restored.rejected,
                weight_sum: restored.weight_sum,
            },
            reference_headers: restored.reference_headers,
            accepted_frames: restored.accepted_frames,
            rejected_frames: restored.rejected_frames,
            input_paths: restored.input_paths,
            input_mode: restored.input_mode,
            configuration_fingerprint,
            ledger: restored
                .ledger
                .unwrap_or_else(crate::replay::Ledger::legacy),
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
                weight_sum: self.accumulator.weight_sum.as_deref(),
                reference_noise: &self.reference_noise,
                accepted_frames: self.accepted_frames,
                rejected_frames: self.rejected_frames,
                input_paths: &self.input_paths,
                input_mode: self.input_mode,
                ledger: &self.ledger,
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
        self.ledger
            .begin_calibration(crate::replay::CalibrationRecord::default());
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
        self.ledger
            .set_current_calibration(crate::replay::CalibrationRecord::from_paths(
                bias_path,
                dark_path,
                flat_path,
                dark_exposure_seconds,
            ));
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
        let source = crate::replay::FrameSource::of(&frame);
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
        let (frame, cfa) = match frame.into_prepared_with_layout() {
            Ok(prepared) => prepared,
            Err(error) => {
                return Ok(self.reject(FrameRejectionReason::IncompatibleImage(error.to_string())));
            }
        };
        let prepared = prepare_frame(
            &self.reference,
            &self.registrar,
            &self.options,
            &self.reference_noise,
            frame.image,
            cfa,
        )?
        .with_source(source);
        Ok(self.integrate_prepared(prepared))
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
        let prepared = prepare_frame(
            &self.reference,
            &self.registrar,
            &self.options,
            &self.reference_noise,
            frame,
            None,
        )?;
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
                reference_noise: &self.reference_noise,
            },
            IntegrationHalf {
                accumulator: &mut self.accumulator,
                options: &self.options,
                accepted_frames: &mut self.accepted_frames,
                rejected_frames: &mut self.rejected_frames,
                input_paths: &mut self.input_paths,
                ledger: &mut self.ledger,
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
        let (mut mean, variance) = self.accumulator.snapshot();
        self.fill_photosite_gaps(&mut mean, &self.accumulator.count);
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
        let mut mean = self.accumulator.mean_snapshot();
        self.fill_photosite_gaps(&mut mean, &self.accumulator.count);
        Ok(StackExportSnapshot {
            image: LinearImage::new(
                self.reference.width,
                self.reference.height,
                self.reference.channels,
                mean,
            )?,
            accepted_frames: self.accepted_frames,
            rejected_frames: self.rejected_frames,
        })
    }

    /// Under Bayer drizzle, fill each channel no frame reached at a pixel
    /// from that channel's covered neighbours; see [`fill_photosite_gaps`].
    pub(crate) fn fill_photosite_gaps(&self, mean: &mut [f32], coverage: &[u32]) {
        if self.options.cfa_integration == CfaIntegration::BayerDrizzle {
            fill_photosite_gaps(
                mean,
                coverage,
                self.reference.width,
                self.reference.channels,
            );
        }
    }

    /// Borrow the current mean and masks without copying full-frame state.
    /// Under Bayer drizzle the borrowed mean is not gap-filled: a channel no
    /// frame has reached reads `NaN` with zero coverage.
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
        let (mut mean, variance, coverage, rejected_samples) = self.accumulator.into_snapshot();
        if self.options.cfa_integration == CfaIntegration::BayerDrizzle {
            fill_photosite_gaps(
                &mut mean,
                &coverage,
                self.reference.width,
                self.reference.channels,
            );
        }
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

    /// Per-channel noise of the reference frame, measured the same way as
    /// [`FrameDiagnostics::noise`]. It is the unit of frame weights: the
    /// reference frame has weight 1 in every channel. Empty when
    /// [`StackOptions::weighting`] is [`FrameWeighting::Equal`].
    pub fn reference_noise(&self) -> &[f32] {
        &self.reference_noise
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

/// Measure the reference frame's per-channel noise, which weighted stacks
/// need before they can weigh any other frame.
fn measure_reference_noise(reference: &LinearImage) -> Result<Vec<f32>> {
    crate::snr::frame_noise(reference)
        .filter(|noise| noise.iter().all(|value| value.is_finite() && *value > 0.0))
        .ok_or_else(|| {
            Error::Stack(
                "frame weighting needs a reference frame whose noise can be measured".into(),
            )
        })
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
    /// The file the frame came from, for a later reintegration.
    source: Option<crate::replay::FrameSource>,
    /// Per-channel normalized noise; empty for equal weighting.
    noise: Vec<f32>,
    /// Per-channel weight; empty for equal weighting.
    weight: Vec<f32>,
    /// Whether `registered` holds one photosite sample per pixel (Bayer
    /// drizzle) rather than every channel.
    photosites: bool,
}

impl PreparedFrame {
    /// Attach the file a ready frame came from.
    pub(crate) fn with_source(mut self, source: Option<crate::replay::FrameSource>) -> Self {
        if let Self::Ready(ready) = &mut self {
            ready.source = source;
        }
        self
    }
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
    reference_noise: &[f32],
    frame: LinearImage,
    cfa: Option<BayerLayout>,
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
    let geometry =
        crate::registration::FrameGeometry::of(registration.transform, registration.warp.as_ref());
    let full_reference = ReferenceRegion {
        x: 0,
        y: 0,
        width: reference.width,
        height: reference.height,
    };
    let mut registered = crate::registration::resample_region_geometry(
        &frame,
        reference.width,
        reference.height,
        full_reference,
        geometry,
        options.interpolation.into(),
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
    let normalization = match NormalizationMap::estimate_with_gains(
        reference,
        &registered,
        options.normalization,
        matches!(
            options.normalization,
            NormalizationMode::LocalBackground { .. }
        )
        .then(|| {
            crate::normalization::photometric_gains(
                reference,
                &registered,
                &registrar.reference_star_positions(),
            )
        })
        .flatten()
        .as_deref(),
    ) {
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
    // Bayer drizzle measures overlap and normalization on the interpolated
    // frame above, which samples every channel evenly, then integrates the
    // photosites themselves with the same normalization.
    let photosites = cfa.filter(|_| options.cfa_integration == CfaIntegration::BayerDrizzle);
    if let Some(layout) = photosites {
        registered = crate::registration::resample_region_geometry(
            &frame,
            reference.width,
            reference.height,
            full_reference,
            geometry,
            crate::registration::Sampling::NearestPhotosite(layout),
        )?;
        if !matches!(options.normalization, NormalizationMode::None) {
            normalization.apply(&mut registered)?;
        }
    }
    // Noise is read from the calibrated frame before resampling. Bilinear
    // resampling averages neighbouring pixels by an amount that depends on
    // each frame's sub-pixel shift, which would bias the weights.
    let (noise, weight) = if options.weighting.is_equal() {
        (Vec::new(), Vec::new())
    } else {
        let Some(raw_noise) = crate::snr::frame_noise(&frame) else {
            return Ok(PreparedFrame::Rejected(
                FrameRejectionReason::IncompatibleImage(
                    "frame noise could not be measured for weighting".into(),
                ),
            ));
        };
        let noise: Vec<f32> = raw_noise
            .iter()
            .enumerate()
            .map(|(channel, &raw)| raw * normalization.channel_mean_gain(channel).abs())
            .collect();
        let weight = noise
            .iter()
            .zip(reference_noise)
            .map(|(&noise, &reference)| {
                options
                    .weighting
                    .weight(reference, noise)
                    .expect("weighting is not equal")
            })
            .collect();
        (noise, weight)
    };
    let normalization_mean_gain = normalization.mean_gain();
    let normalization_mean_offset = normalization.mean_offset();
    let mut mapping = crate::RegisteredFrameMapping::new(
        reference.width,
        reference.height,
        registration.transform,
        normalization,
    )?;
    mapping.set_warp(registration.warp.clone())?;
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
        source: None,
        noise,
        weight,
        photosites: photosites.is_some(),
    })))
}

/// The immutable half of a stack: everything preparing a frame may read.
pub(crate) struct PreparationHalf<'a> {
    pub(crate) reference: &'a LinearImage,
    pub(crate) registrar: &'a Registrar,
    pub(crate) calibration: &'a CalibrationMasters,
    pub(crate) options: &'a StackOptions,
    pub(crate) reference_noise: &'a [f32],
}

/// The mutable half of a stack: the accumulator and the run's tallies.
pub(crate) struct IntegrationHalf<'a> {
    accumulator: &'a mut Accumulator,
    options: &'a StackOptions,
    accepted_frames: &'a mut u32,
    rejected_frames: &'a mut u32,
    input_paths: &'a mut Vec<PathBuf>,
    ledger: &'a mut crate::replay::Ledger,
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
            source,
            noise,
            weight,
            photosites,
        } = *ready;
        let weights = (!weight.is_empty()).then_some(weight.as_slice());

        let (would_accept, _) =
            self.accumulator
                .classify(&registered.data, self.options.rejection, weights);
        // A photosite frame offers one sample per pixel, not one per channel.
        let offered = if photosites {
            registered.pixel_count()
        } else {
            registered.sample_count()
        };
        let integrated_fraction = would_accept as f32 / offered as f32;
        if integrated_fraction < self.options.acceptance.minimum_integrated_fraction {
            return self.reject(FrameRejectionReason::InsufficientIntegratedSamples {
                measured: integrated_fraction,
                minimum: self.options.acceptance.minimum_integrated_fraction,
            });
        }
        let (accepted_samples, rejected_samples) =
            self.accumulator
                .integrate(&registered.data, self.options.rejection, weights);
        *self.accepted_frames += 1;
        self.ledger.admit(
            source,
            (*mapping).clone(),
            crate::replay::FrameWeightRecord {
                noise: noise.clone(),
                weight: weight.clone(),
            },
        );
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
            noise,
            weight,
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

/// Online per-sample moments.
///
/// With equal weights this is Welford's algorithm. With frame weights it is
/// West's weighted form: `weight_sum` holds each sample's total weight, `mean`
/// is the weighted mean, and `m2` is the weighted sum of squared deviations.
/// Weights are relative to the reference frame, so `m2 / (count - 1)` still
/// estimates the variance of a frame with weight 1. `count` always counts
/// frames, never weight: coverage, warm-up, and depth readings use it.
struct Accumulator {
    mean: Vec<f32>,
    m2: Vec<f32>,
    count: Vec<u32>,
    rejected: Vec<u32>,
    /// Per-sample sum of weights; `None` when frames are weighted equally.
    weight_sum: Option<Vec<f32>>,
}

impl Accumulator {
    fn new(samples: usize, weighted: bool) -> Self {
        Self {
            mean: vec![0.0; samples],
            m2: vec![0.0; samples],
            count: vec![0; samples],
            rejected: vec![0; samples],
            weight_sum: weighted.then(|| vec![0.0; samples]),
        }
    }

    /// Integrate one frame. `weights` holds one weight per channel and is
    /// ignored by an equal-weight accumulator; a weighted accumulator given
    /// `None` uses weight 1.
    fn integrate(
        &mut self,
        samples: &[f32],
        rejection: RejectionMode,
        weights: Option<&[f32]>,
    ) -> (usize, usize) {
        let Some(weight_sum) = self.weight_sum.as_mut() else {
            debug_assert!(weights.is_none(), "equal-weight stack given weights");
            // Fixed chunks with a plain inner loop: each sample's update is
            // independent, so this is the same arithmetic as one pass per
            // sample, without a rayon split and tuple reduce per element.
            return self
                .mean
                .par_chunks_mut(ACCUMULATE_CHUNK)
                .zip(self.m2.par_chunks_mut(ACCUMULATE_CHUNK))
                .zip(self.count.par_chunks_mut(ACCUMULATE_CHUNK))
                .zip(self.rejected.par_chunks_mut(ACCUMULATE_CHUNK))
                .zip(samples.par_chunks(ACCUMULATE_CHUNK))
                .map(|((((means, m2s), counts), rejecteds), samples)| {
                    let (mut accepted, mut refused) = (0, 0);
                    for ((((mean, m2), count), rejected), &sample) in means
                        .iter_mut()
                        .zip(m2s.iter_mut())
                        .zip(counts.iter_mut())
                        .zip(rejecteds.iter_mut())
                        .zip(samples)
                    {
                        if !sample.is_finite() {
                            continue;
                        }
                        if should_reject_sample(*mean, *m2, *count, sample, rejection) {
                            *rejected = rejected.saturating_add(1);
                            refused += 1;
                            continue;
                        }
                        let next_count = count.saturating_add(1);
                        let delta = sample - *mean;
                        *mean += delta / next_count as f32;
                        let delta_after = sample - *mean;
                        *m2 += delta * delta_after;
                        *count = next_count;
                        accepted += 1;
                    }
                    (accepted, refused)
                })
                .reduce(
                    || (0, 0),
                    |left, right| (left.0 + right.0, left.1 + right.1),
                );
        };
        let unit = [1.0_f32];
        let weights = weights.unwrap_or(&unit);
        let channels = weights.len();
        self.mean
            .par_iter_mut()
            .zip(self.m2.par_iter_mut())
            .zip(self.count.par_iter_mut())
            .zip(self.rejected.par_iter_mut())
            .zip(weight_sum.par_iter_mut())
            .zip(samples.par_iter())
            .enumerate()
            .map(
                |(index, (((((mean, m2), count), rejected), weight_sum), &sample))| {
                    if !sample.is_finite() {
                        return (0, 0);
                    }
                    let weight = weights[index % channels];
                    if should_reject_weighted_sample(*mean, *m2, *count, sample, weight, rejection)
                    {
                        *rejected = rejected.saturating_add(1);
                        return (0, 1);
                    }
                    // West's weighted update. With weight 1 every operation
                    // matches the equal-weight update bit for bit.
                    let next_weight = *weight_sum + weight;
                    let delta = sample - *mean;
                    let weighted_delta = delta * weight;
                    *mean += weighted_delta / next_weight;
                    let delta_after = sample - *mean;
                    *m2 += weighted_delta * delta_after;
                    *weight_sum = next_weight;
                    *count = count.saturating_add(1);
                    (1, 0)
                },
            )
            .reduce(
                || (0, 0),
                |left, right| (left.0 + right.0, left.1 + right.1),
            )
    }

    fn classify(
        &self,
        samples: &[f32],
        rejection: RejectionMode,
        weights: Option<&[f32]>,
    ) -> (usize, usize) {
        if self.weight_sum.is_none() {
            debug_assert!(weights.is_none(), "equal-weight stack given weights");
            return self
                .mean
                .par_chunks(ACCUMULATE_CHUNK)
                .zip(self.m2.par_chunks(ACCUMULATE_CHUNK))
                .zip(self.count.par_chunks(ACCUMULATE_CHUNK))
                .zip(samples.par_chunks(ACCUMULATE_CHUNK))
                .map(|(((means, m2s), counts), samples)| {
                    let (mut accepted, mut refused) = (0, 0);
                    for (((&mean, &m2), &count), &sample) in
                        means.iter().zip(m2s).zip(counts).zip(samples)
                    {
                        if !sample.is_finite() {
                        } else if should_reject_sample(mean, m2, count, sample, rejection) {
                            refused += 1;
                        } else {
                            accepted += 1;
                        }
                    }
                    (accepted, refused)
                })
                .reduce(
                    || (0, 0),
                    |left, right| (left.0 + right.0, left.1 + right.1),
                );
        }
        let unit = [1.0_f32];
        let weights = weights.unwrap_or(&unit);
        let channels = weights.len();
        self.mean
            .par_iter()
            .zip(self.m2.par_iter())
            .zip(self.count.par_iter())
            .zip(samples.par_iter())
            .enumerate()
            .map(|(index, (((mean, m2), count), &sample))| {
                if !sample.is_finite() {
                    (0, 0)
                } else if should_reject_weighted_sample(
                    *mean,
                    *m2,
                    *count,
                    sample,
                    weights[index % channels],
                    rejection,
                ) {
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

/// Fill each channel of a Bayer-drizzled mean that no frame reached at a
/// pixel, where some other channel was reached, with the mean of the same
/// channel's covered samples within one pixel, or else within two. Coverage
/// is left at zero, so the gaps stay visible in the coverage map. Gaps are
/// rare once frames have drifted a few pixels, so the fills are collected
/// and then written.
pub(crate) fn fill_photosite_gaps(
    mean: &mut [f32],
    coverage: &[u32],
    width: usize,
    channels: usize,
) {
    if channels != 3 || width == 0 {
        return;
    }
    let height = mean.len() / (width * channels);
    let read: &[f32] = mean;
    let fills = (0..height)
        .into_par_iter()
        .flat_map_iter(|y| {
            (0..width).flat_map(move |x| {
                let pixel = (y * width + x) * channels;
                let reached = (0..channels).any(|channel| coverage[pixel + channel] > 0);
                (0..channels).filter_map(move |channel| {
                    let index = pixel + channel;
                    if !reached || coverage[index] > 0 {
                        return None;
                    }
                    (1..=2_usize).find_map(|radius| {
                        let (mut sum, mut count) = (0.0_f64, 0_u32);
                        for ny in y.saturating_sub(radius)..(y + radius + 1).min(height) {
                            for nx in x.saturating_sub(radius)..(x + radius + 1).min(width) {
                                let neighbour = (ny * width + nx) * channels + channel;
                                if coverage[neighbour] > 0 && read[neighbour].is_finite() {
                                    sum += f64::from(read[neighbour]);
                                    count += 1;
                                }
                            }
                        }
                        (count > 0).then(|| (index, (sum / f64::from(count)) as f32))
                    })
                })
            })
        })
        .collect::<Vec<_>>();
    for (index, value) in fills {
        mean[index] = value;
    }
}

/// Samples per parallel task in the accumulator's per-sample passes.
const ACCUMULATE_CHUNK: usize = 16 * 1024;

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

/// Delta-sigma rejection for a weighted stack. `m2 / (count - 1)` estimates
/// the variance of a weight-1 frame, so a sample from a frame of weight `w`
/// is expected to scatter with variance `m2 / (count - 1) / w`. With weight 1
/// this is [`should_reject_sample`] bit for bit.
fn should_reject_weighted_sample(
    mean: f32,
    m2: f32,
    count: u32,
    sample: f32,
    weight: f32,
    rejection: RejectionMode,
) -> bool {
    match rejection {
        RejectionMode::None => false,
        RejectionMode::DeltaSigma(options) if count >= options.warmup_samples && count > 1 => {
            let sigma = (m2 / (count - 1) as f32 / weight)
                .sqrt()
                .max(options.minimum_sigma);
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
        let mut accumulator = Accumulator::new(1, false);
        let rejection = RejectionMode::DeltaSigma(DeltaSigmaOptions {
            warmup_samples: 4,
            low_sigma: 3.0,
            high_sigma: 3.0,
            minimum_sigma: 0.01,
        });
        for value in [10.0, 10.1, 9.9, 10.05] {
            accumulator.integrate(&[value], rejection, None);
        }
        let before = accumulator.mean[0];
        let (_, rejected) = accumulator.integrate(&[1000.0], rejection, None);
        assert_eq!(rejected, 1);
        assert_eq!(accumulator.count[0], 4);
        assert_eq!(accumulator.mean[0], before);
    }

    #[test]
    fn batch_rejection_removes_trail_that_live_warmup_cannot_revisit() {
        let frames: Vec<_> = (0..20)
            .map(|index| {
                LinearImage::new(1, 1, 1, vec![if index == 0 { 11_000.0 } else { 1000.0 }]).unwrap()
            })
            .collect();
        let mut online = Accumulator::new(1, false);
        for frame in &frames {
            online.integrate(&frame.data, RejectionMode::default(), None);
        }
        assert!(
            online.mean[0] > 1400.0,
            "the live reference trail stays in its mean"
        );
        let completed = crate::integrate_registered_frames(
            frames.len(),
            &crate::BatchStackOptions::default(),
            |_, index| Ok(frames[index].clone()),
        )
        .unwrap()
        .snapshot;
        assert_eq!(completed.image.data, vec![1000.0]);
        assert_eq!(completed.coverage, vec![19]);
        assert_eq!(completed.rejected_samples, vec![1]);
    }

    #[test]
    fn export_snapshot_owns_only_a_frozen_finalized_mean() {
        let mut accumulator = Accumulator::new(2, false);
        accumulator.integrate(&[5.0, f32::NAN], RejectionMode::None, None);
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
                weight_sum: None,
                reference_noise: &[],
                accepted_frames: stacker.accepted_frames,
                rejected_frames: stacker.rejected_frames,
                input_paths: &stacker.input_paths,
                input_mode: stacker.input_mode,
                ledger: &stacker.ledger,
            },
        )
        .unwrap();

        let mut restored = LiveStacker::open_context(&path).unwrap();
        // A v1 context has no ledger, so the stack cannot be replayed.
        assert!(restored.reintegration_unavailable().is_some());
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

    /// An RGGB mosaic of a colored star field moved by `(dx, dy)`: each
    /// photosite records its own channel of the continuous scene.
    fn bayer_star_field(width: usize, height: usize, dx: f32, dy: f32) -> LinearImage {
        const SKY: [f32; 3] = [400.0, 300.0, 200.0];
        const COLOR: [f32; 3] = [1.0, 0.8, 0.6];
        let stars = [
            (21.7_f32, 18.4_f32),
            (71.3, 28.1),
            (132.2, 34.8),
            (43.1, 49.7),
            (103.4, 58.3),
            (24.8, 72.2),
            (82.7, 76.5),
            (141.1, 87.8),
            (54.4, 96.2),
            (116.8, 104.1),
            (33.2, 110.0),
            (91.5, 116.4),
        ];
        let layout = BayerLayout {
            pattern: seiza_fits::BayerPattern::Rggb,
            x_offset: 0,
            y_offset: 0,
        };
        let data = (0..width * height)
            .map(|index| {
                let (x, y) = (index % width, index / width);
                let channel = layout.channel_at(x, y);
                let mut value = SKY[channel];
                for (star, (star_x, star_y)) in stars.iter().enumerate() {
                    let rx = x as f32 - star_x - dx;
                    let ry = y as f32 - star_y - dy;
                    value += COLOR[channel]
                        * (2_000.0 + star as f32 * 150.0)
                        * (-(rx * rx + ry * ry) / 3.2).exp();
                }
                value
            })
            .collect();
        LinearImage::new(width, height, 1, data).unwrap()
    }

    #[test]
    fn bayer_drizzle_integrates_one_photosite_per_pixel_and_replays_the_same() {
        let (width, height) = (160, 128);
        let shifts = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (2.3, 0.6),
            (0.4, 2.2),
            (3.1, 3.3),
            (1.6, 2.7),
        ];
        let directory = tempfile::tempdir().unwrap();
        let bayer = [("BAYERPAT".to_string(), HeaderValue::String("RGGB".into()))];
        let paths = shifts
            .iter()
            .enumerate()
            .map(|(index, &(dx, dy))| {
                let path = directory.path().join(format!("light-{index}.fits"));
                crate::write_processed_image_fits_f32(
                    &path,
                    &bayer_star_field(width, height, dx, dy),
                    &bayer,
                    &[],
                )
                .unwrap();
                path
            })
            .collect::<Vec<_>>();
        let options = StackOptions {
            normalization: NormalizationMode::None,
            rejection: RejectionMode::None,
            cfa_integration: CfaIntegration::BayerDrizzle,
            ..StackOptions::default()
        };
        let mut stacker =
            LiveStacker::open_fits(&paths[0], None, None, None, None, options).unwrap();
        for path in &paths[1..] {
            assert!(
                matches!(
                    stacker.push_fits(path).unwrap(),
                    FrameDisposition::Accepted(_)
                ),
                "{}",
                path.display()
            );
        }

        // Every frame offers exactly one sample per pixel it covers, in that
        // photosite's channel, so away from the edges the three channels'
        // coverage sums to the frame count. Gaps are filled, and the sky
        // keeps its color: nothing mixed the channels.
        let check = |snapshot: &StackSnapshot| {
            let mut sky = [Vec::new(), Vec::new(), Vec::new()];
            for y in 8..height - 8 {
                for x in 8..width - 8 {
                    let pixel = (y * width + x) * 3;
                    let covered: u32 = snapshot.coverage[pixel..pixel + 3].iter().sum();
                    assert_eq!(covered as usize, shifts.len(), "({x}, {y})");
                    for (channel, samples) in sky.iter_mut().enumerate() {
                        let value = snapshot.image.data[pixel + channel];
                        assert!(value.is_finite(), "({x}, {y}) channel {channel}");
                        samples.push(value);
                    }
                }
            }
            for (channel, expected) in [400.0, 300.0, 200.0].into_iter().enumerate() {
                let median = seiza_stats::median_in_place(&mut sky[channel]).unwrap();
                assert!(
                    (median - expected).abs() < 1.0,
                    "channel {channel}: {median}"
                );
            }
        };
        check(&stacker.snapshot().unwrap());

        let replayed = stacker
            .reintegrate(&crate::BatchStackOptions::default(), |_, _, _| {})
            .unwrap();
        let mut sky = [Vec::new(), Vec::new(), Vec::new()];
        for y in 8..height - 8 {
            for x in 8..width - 8 {
                for (channel, samples) in sky.iter_mut().enumerate() {
                    let value = replayed.snapshot.image.data[(y * width + x) * 3 + channel];
                    assert!(value.is_finite());
                    samples.push(value);
                }
            }
        }
        for (channel, expected) in [400.0, 300.0, 200.0].into_iter().enumerate() {
            let median = seiza_stats::median_in_place(&mut sky[channel]).unwrap();
            assert!(
                (median - expected).abs() < 1.0,
                "replayed channel {channel}: {median}"
            );
        }
    }

    #[test]
    fn photosite_gaps_fill_from_the_same_channel_and_keep_zero_coverage() {
        // A 3x3 RGB image whose centre pixel has red and green but no blue.
        let mut mean = vec![f32::NAN; 27];
        let mut coverage = vec![0_u32; 27];
        for pixel in 0..9 {
            for channel in 0..3 {
                if pixel == 4 && channel == 2 {
                    continue;
                }
                mean[pixel * 3 + channel] = (channel as f32 + 1.0) * 100.0 + pixel as f32;
                coverage[pixel * 3 + channel] = 1;
            }
        }
        fill_photosite_gaps(&mut mean, &coverage, 3, 3);
        let neighbours = (0..9).filter(|&pixel| pixel != 4).map(|p| 300.0 + p as f32);
        let expected = neighbours.sum::<f32>() / 8.0;
        assert!((mean[4 * 3 + 2] - expected).abs() < 1.0e-3);
        assert_eq!(coverage[4 * 3 + 2], 0);
    }

    #[test]
    fn quadratic_registration_warps_survive_a_context_round_trip() {
        let options = StackOptions {
            normalization: NormalizationMode::None,
            registration: RegistrationOptions {
                model: crate::RegistrationModel::Quadratic,
                ..RegistrationOptions::default()
            },
            ..StackOptions::default()
        };
        let mut stacker =
            LiveStacker::from_linear(crate::registration::test_star_field(false), options).unwrap();
        let FrameDisposition::Accepted(diagnostics) = stacker
            .push_linear(crate::registration::test_star_field(true))
            .unwrap()
        else {
            panic!("the distorted frame registers");
        };
        let warp = diagnostics
            .mapping
            .warp()
            .cloned()
            .expect("a quadratic warp");
        assert!(diagnostics.registration_rms_pixels < 0.1);

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("warped.seiza-stack");
        stacker.save_context(&path).unwrap();
        let restored = LiveStacker::open_context(&path).unwrap();
        assert_eq!(restored.ledger.warps(), vec![None, Some(warp)]);
        assert_eq!(
            restored.options.registration.model,
            crate::RegistrationModel::Quadratic
        );
    }

    /// Local background normalization matches every frame's background to
    /// the reference frame's, so a gradient only the reference has reaches
    /// the whole online stack. Reintegration refits each frame's background
    /// against an integration of the best frames, where it largely averages
    /// out.
    #[test]
    fn reintegration_refits_backgrounds_against_an_integrated_reference() {
        let directory = tempfile::tempdir().unwrap();
        let field = crate::registration::test_star_field(false);
        let width = field.width;
        let paths = (0..10)
            .map(|frame| {
                let mut state = 0x9e37_79b9_u32.wrapping_mul(frame + 1) | 1;
                let data = field
                    .data
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        let noise = (state % 2001) as f32 / 100.0 - 10.0;
                        let gradient = if frame == 0 {
                            (index % width) as f32 * 0.5
                        } else {
                            0.0
                        };
                        value + noise + gradient
                    })
                    .collect();
                let path = directory.path().join(format!("light-{frame}.fits"));
                crate::write_processed_image_fits_f32(
                    &path,
                    &LinearImage::new(width, field.height, 1, data).unwrap(),
                    &[],
                    &[],
                )
                .unwrap();
                path
            })
            .collect::<Vec<_>>();
        let options = StackOptions {
            normalization: NormalizationMode::LocalBackground { tile_size: 64 },
            ..StackOptions::default()
        };
        let mut stacker =
            LiveStacker::open_fits(&paths[0], None, None, None, None, options).unwrap();
        for path in &paths[1..] {
            assert!(matches!(
                stacker.push_fits(path).unwrap(),
                FrameDisposition::Accepted(_)
            ));
        }
        // Median of a column band, away from the outermost tiles.
        let band = |image: &LinearImage, from: usize| {
            let mut values = (40..image.height - 40)
                .flat_map(|y| (from..from + 40).map(move |x| (x, y)))
                .map(|(x, y)| image.data[y * image.width + x])
                .collect::<Vec<_>>();
            seiza_stats::median_in_place(&mut values).unwrap()
        };
        let slope = |image: &LinearImage| band(image, width - 120) - band(image, 80);
        let online = slope(&stacker.snapshot().unwrap().image);
        let replayed = stacker
            .reintegrate(&crate::BatchStackOptions::default(), |_, _, _| {})
            .unwrap();
        let refit = slope(&replayed.snapshot.image);
        assert!(
            online > 150.0,
            "the online stack carries the gradient: {online}"
        );
        assert!(
            refit < 0.3 * online,
            "reintegrated {refit} against online {online}"
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

#[cfg(test)]
mod weighting_tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn digest(values: &[f32]) -> String {
        let mut hasher = Sha256::new();
        for value in values {
            hasher.update(value.to_bits().to_le_bytes());
        }
        hex(&hasher.finalize())
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    const WIDTH: usize = 160;
    const HEIGHT: usize = 128;
    const STARS: [(f32, f32); 12] = [
        (19.7, 16.4),
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

    /// The star field the other stack tests use, with its fixed pattern
    /// noise, so the golden digests below describe a realistic run.
    fn field() -> LinearImage {
        let mut data = Vec::with_capacity(WIDTH * HEIGHT);
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let noise = ((x * 17 + y * 31) % 23) as f32 * 0.12 - 1.32;
                let mut value = 100.0 + noise;
                for (index, (star_x, star_y)) in STARS.iter().enumerate() {
                    let dx = x as f32 - star_x;
                    let dy = y as f32 - star_y;
                    value +=
                        (900.0 + index as f32 * 130.0) * (-(dx.mul_add(dx, dy * dy)) / 3.2).exp();
                }
                data.push(value);
            }
        }
        LinearImage::new(WIDTH, HEIGHT, 1, data).unwrap()
    }

    /// Bright stars on a flat sky with no noise: the truth a noisy stack is
    /// measured against.
    fn clean_field() -> LinearImage {
        let mut data = Vec::with_capacity(WIDTH * HEIGHT);
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let mut value = 1000.0_f32;
                for (index, (star_x, star_y)) in STARS.iter().enumerate() {
                    let dx = x as f32 - star_x;
                    let dy = y as f32 - star_y;
                    value += (20_000.0 + index as f32 * 1_500.0)
                        * (-(dx.mul_add(dx, dy * dy)) / 3.2).exp();
                }
                data.push(value);
            }
        }
        LinearImage::new(WIDTH, HEIGHT, 1, data).unwrap()
    }

    /// A deterministic normal generator.
    struct Gaussian(u64);

    impl Gaussian {
        fn uniform(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            (-2.0 * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
        }
    }

    fn noisy(truth: &LinearImage, sigma: f64, seed: u64) -> LinearImage {
        let mut rng = Gaussian(seed);
        LinearImage::new(
            truth.width,
            truth.height,
            truth.channels,
            truth
                .data
                .iter()
                .map(|value| (f64::from(*value) + rng.normal() * sigma) as f32)
                .collect(),
        )
        .unwrap()
    }

    /// Frames for the golden run: the pattern field plus a varying ripple
    /// and one late outlier, so rejection has work to do.
    fn golden_frames(reference: &LinearImage) -> Vec<LinearImage> {
        [0.3_f32, -0.2, 0.1, -0.4, 0.25]
            .iter()
            .enumerate()
            .map(|(index, offset)| {
                let mut frame = reference.clone();
                for (sample, value) in frame.data.iter_mut().enumerate() {
                    *value += offset + ((sample * 13 + index * 7) % 17) as f32 * 0.3;
                }
                if index == 4 {
                    frame.data[5000] += 5000.0;
                }
                frame
            })
            .collect()
    }

    fn golden_options(weighting: FrameWeighting) -> StackOptions {
        StackOptions {
            rejection: RejectionMode::DeltaSigma(DeltaSigmaOptions {
                warmup_samples: 3,
                minimum_sigma: 0.01,
                ..DeltaSigmaOptions::default()
            }),
            weighting,
            ..StackOptions::default()
        }
    }

    fn batch_frames(reference: &LinearImage) -> Vec<LinearImage> {
        (0..8)
            .map(|index| {
                let mut frame = reference.clone();
                for (sample, value) in frame.data.iter_mut().enumerate() {
                    *value += ((sample * 13 + index * 7) % 17) as f32 * 0.3;
                }
                if index == 2 {
                    frame.data[700] += 9000.0;
                }
                frame
            })
            .collect()
    }

    /// Weighting where every frame clamps to weight 1.
    fn unit_weighting() -> FrameWeighting {
        FrameWeighting::InverseNoiseVariance {
            minimum_weight: 1.0,
            maximum_weight: 1.0,
        }
    }

    /// Digests recorded from `main` (seiza-stacking 0.17.0) before frame
    /// weighting existed. Equal weighting must keep producing exactly these
    /// bytes.
    ///
    /// The configuration fingerprint hashes serialized options only, so it is
    /// checked on every platform. The pixel, variance, and context digests
    /// were recorded on Linux x86-64: the test field uses `exp`, and
    /// registration uses trigonometry, whose last bits come from each
    /// platform's math library. Those digests are checked only there. The
    /// in-process tests that compare unit weights with equal weights bit for
    /// bit run everywhere.
    #[test]
    fn equal_weighting_matches_the_release_before_weighting() {
        let reference = field();
        let mut stacker =
            LiveStacker::from_linear(reference.clone(), golden_options(FrameWeighting::Equal))
                .unwrap();
        for frame in golden_frames(&reference) {
            let FrameDisposition::Accepted(diagnostics) = stacker.push_linear(frame).unwrap()
            else {
                panic!("golden frames are accepted");
            };
            assert!(diagnostics.noise.is_empty() && diagnostics.weight.is_empty());
        }
        assert!(stacker.reference_noise().is_empty());
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("golden.seiza-stack");
        stacker.save_context(&path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[8..12], &3_u32.to_le_bytes(), "equal stacks write v3");
        let snapshot = stacker.snapshot().unwrap();
        assert_eq!(
            stacker.configuration_fingerprint(),
            "d60e7ea72f889e480f3c7865b62f606b73c474629f2ce033f2921c239a6a2bed"
        );
        assert!(snapshot.rejected_samples.iter().sum::<u32>() > 0);
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            assert_eq!(
                hex(&Sha256::digest(&bytes)),
                "88fe4b86a6ef8928d62d5630e47e3500dd24419fbf94887ad3e8b118c41793e4"
            );
            assert_eq!(
                digest(&snapshot.image.data),
                "03e76662472764fa23b29b7dc17ff51872783095c8531ba094fa274113fdd523"
            );
            assert_eq!(
                digest(&snapshot.variance.data),
                "78202b30a58da1d67aca708eb811fbe29bb2701ad27110112199f9416a23d3be"
            );
            assert_eq!(snapshot.rejected_samples.iter().sum::<u32>(), 334);
        }

        let frames = batch_frames(&reference);
        let batch = crate::integrate_registered_frames(
            frames.len(),
            &crate::BatchStackOptions::default(),
            |_, index| Ok(frames[index].clone()),
        )
        .unwrap()
        .snapshot;
        assert_eq!(batch.rejected_samples.iter().sum::<u32>(), 1);
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            assert_eq!(
                digest(&batch.image.data),
                "226ad323408cc5cf365ff6670be69a60f0d88f5312362a484a61de43b3dd5bd9"
            );
            assert_eq!(
                digest(&batch.variance.data),
                "f416eb68eca424d5ec5ce2338d576fa867e116d169e24bbb0433342fb6378645"
            );
        }
    }

    #[test]
    fn equal_weighting_is_not_serialized_and_weighted_options_parse() {
        let equal = serde_json::to_string(&StackOptions::default()).unwrap();
        assert!(!equal.contains("weighting"), "{equal}");

        let options: StackOptions =
            serde_json::from_str(r#"{"weighting": {"mode": "inverse-noise-variance"}}"#).unwrap();
        assert_eq!(options.weighting, FrameWeighting::inverse_noise_variance());
        options.validate().unwrap();
        let json = serde_json::to_string(&options).unwrap();
        assert!(json.contains(r#""weighting":{"mode":"inverse-noise-variance","minimum_weight":0.05,"maximum_weight":20.0}"#), "{json}");
        let round_trip: StackOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(round_trip.weighting, options.weighting);

        let equal: StackOptions =
            serde_json::from_str(r#"{"weighting": {"mode": "equal"}}"#).unwrap();
        assert!(equal.weighting.is_equal());
        assert!(
            serde_json::from_str::<StackOptions>(
                r#"{"weighting": {"mode": "inverse-noise-variance", "mystery": 1}}"#
            )
            .is_err()
        );

        for (minimum_weight, maximum_weight) in [
            (0.0, 20.0),
            (1.5, 20.0),
            (0.05, 0.5),
            (f32::NAN, 20.0),
            (0.05, f32::INFINITY),
        ] {
            let options = StackOptions {
                weighting: FrameWeighting::InverseNoiseVariance {
                    minimum_weight,
                    maximum_weight,
                },
                ..StackOptions::default()
            };
            assert!(
                options.validate().is_err(),
                "{minimum_weight}..{maximum_weight}"
            );
        }
        let weighted = StackOptions {
            weighting: FrameWeighting::inverse_noise_variance(),
            ..StackOptions::default()
        };
        let equal = LiveStacker::from_linear(field(), StackOptions::default()).unwrap();
        let weighted = LiveStacker::from_linear(field(), weighted).unwrap();
        assert_ne!(
            equal.configuration_fingerprint(),
            weighted.configuration_fingerprint(),
            "weighting is part of the stack's identity"
        );
        assert_eq!(weighted.reference_noise().len(), 1);
        assert!(weighted.reference_noise()[0] > 0.0);
    }

    #[test]
    fn unit_weights_integrate_bit_identically_to_equal_weights() {
        let reference = field();
        let mut equal =
            LiveStacker::from_linear(reference.clone(), golden_options(FrameWeighting::Equal))
                .unwrap();
        let mut weighted =
            LiveStacker::from_linear(reference.clone(), golden_options(unit_weighting())).unwrap();
        for frame in golden_frames(&reference) {
            let expected = equal.push_linear(frame.clone()).unwrap();
            let actual = weighted.push_linear(frame).unwrap();
            let (FrameDisposition::Accepted(expected), FrameDisposition::Accepted(actual)) =
                (expected, actual)
            else {
                panic!("both stacks accept every frame");
            };
            assert_eq!(actual.weight, vec![1.0]);
            assert_eq!(actual.noise.len(), 1);
            assert_eq!(actual.accepted_samples, expected.accepted_samples);
            assert_eq!(actual.rejected_samples, expected.rejected_samples);
        }
        let expected = equal.snapshot().unwrap();
        let actual = weighted.snapshot().unwrap();
        assert!(expected.rejected_samples.iter().sum::<u32>() > 0);
        assert_eq!(bits(&actual.image.data), bits(&expected.image.data));
        assert_eq!(bits(&actual.variance.data), bits(&expected.variance.data));
        assert_eq!(actual.coverage, expected.coverage);
        assert_eq!(actual.rejected_samples, expected.rejected_samples);

        let frames = batch_frames(&reference);
        let run = |frame_weights| {
            crate::integrate_registered_frames(
                frames.len(),
                &crate::BatchStackOptions {
                    frame_weights,
                    ..crate::BatchStackOptions::default()
                },
                |_, index| Ok(frames[index].clone()),
            )
            .unwrap()
            .snapshot
        };
        let expected = run(None);
        let actual = run(Some(vec![vec![1.0]; frames.len()]));
        assert_eq!(expected.rejected_samples.iter().sum::<u32>(), 1);
        assert_eq!(bits(&actual.image.data), bits(&expected.image.data));
        assert_eq!(bits(&actual.variance.data), bits(&expected.variance.data));
        assert_eq!(actual.coverage, expected.coverage);
        assert_eq!(actual.rejected_samples, expected.rejected_samples);
    }

    /// Frames of two noise levels over the same sky, interleaved; the first
    /// is a quiet one and serves as the reference.
    fn mixed_noise_frames() -> (LinearImage, Vec<(f64, LinearImage)>) {
        let truth = clean_field();
        let frames = (0..30)
            .map(|index| {
                let sigma = if index % 3 == 2 { 25.0 } else { 10.0 };
                (
                    sigma,
                    noisy(&truth, sigma, 0x9e37_79b9_7f4a_7c15 ^ (index as u64 + 1)),
                )
            })
            .collect();
        (truth, frames)
    }

    /// Standard deviation of the stack about the truth, over sky pixels away
    /// from stars.
    fn sky_residual(stack: &LinearImage, truth: &LinearImage) -> f64 {
        let (mut sum, mut count) = (0.0_f64, 0usize);
        for (value, truth) in stack.data.iter().zip(&truth.data) {
            if *truth < 1000.5 && value.is_finite() {
                sum += (f64::from(*value) - f64::from(*truth)).powi(2);
                count += 1;
            }
        }
        (sum / count as f64).sqrt()
    }

    fn stack_mixed(
        weighting: FrameWeighting,
        rejection: RejectionMode,
    ) -> (StackSnapshot, Vec<(f64, FrameDiagnostics)>) {
        let (_, frames) = mixed_noise_frames();
        let options = StackOptions {
            normalization: NormalizationMode::None,
            rejection,
            weighting,
            ..StackOptions::default()
        };
        let mut frames = frames.into_iter();
        let (_, reference) = frames.next().unwrap();
        let mut stacker = LiveStacker::from_linear(reference, options).unwrap();
        let diagnostics = frames
            .map(|(sigma, frame)| match stacker.push_linear(frame).unwrap() {
                FrameDisposition::Accepted(diagnostics) => (sigma, diagnostics),
                FrameDisposition::Rejected(reason) => panic!("frame rejected: {reason}"),
            })
            .collect();
        (stacker.into_snapshot().unwrap(), diagnostics)
    }

    #[test]
    fn inverse_noise_weighting_reaches_the_optimal_stack_noise() {
        let (truth, frames) = mixed_noise_frames();
        let optimal = 1.0
            / frames
                .iter()
                .map(|(sigma, _)| 1.0 / (sigma * sigma))
                .sum::<f64>()
                .sqrt();
        let (equal, _) = stack_mixed(FrameWeighting::Equal, RejectionMode::None);
        let (weighted, diagnostics) = stack_mixed(
            FrameWeighting::inverse_noise_variance(),
            RejectionMode::None,
        );
        for (sigma, frame) in &diagnostics {
            let expected = (10.0 / sigma).powi(2) as f32;
            assert!(
                (frame.weight[0] / expected - 1.0).abs() < 0.08,
                "sigma {sigma}: weight {} expected {expected}",
                frame.weight[0]
            );
            // The estimator reads a few percent high on this small, star-rich
            // field; the bias is shared by every frame, so weights are not.
            assert!((f64::from(frame.noise[0]) / sigma - 1.0).abs() < 0.10);
        }
        let equal_noise = sky_residual(&equal.image, &truth);
        let weighted_noise = sky_residual(&weighted.image, &truth);
        eprintln!("optimal={optimal:.4} equal={equal_noise:.4} weighted={weighted_noise:.4}");
        assert!(weighted_noise < equal_noise * 0.8);
        assert!(
            (weighted_noise / optimal - 1.0).abs() < 0.05,
            "weighted {weighted_noise} vs optimal {optimal}"
        );
        // The depth reader sees the same improvement.
        let view = |snapshot: &StackSnapshot| {
            crate::measure_depth(StackView {
                width: snapshot.image.width,
                height: snapshot.image.height,
                channels: 1,
                mean: &snapshot.image.data,
                coverage: &snapshot.coverage,
                rejected_samples: &snapshot.rejected_samples,
                accepted_frames: snapshot.accepted_frames,
                rejected_frames: 0,
            })
            .unwrap()
            .noise
        };
        assert!(view(&weighted) < view(&equal));
        assert_eq!(weighted.coverage, equal.coverage, "coverage counts frames");
        // Variance output is that of a weight-1 (reference-quality) frame.
        let sky_variance = weighted
            .variance
            .data
            .iter()
            .zip(&truth.data)
            .filter(|(_, truth)| **truth < 1000.5)
            .map(|(variance, _)| f64::from(*variance))
            .sum::<f64>()
            / weighted
                .variance
                .data
                .iter()
                .zip(&truth.data)
                .filter(|(_, truth)| **truth < 1000.5)
                .count() as f64;
        assert!(
            (sky_variance.sqrt() / 10.0 - 1.0).abs() < 0.05,
            "unit-weight sigma {}",
            sky_variance.sqrt()
        );
    }

    #[test]
    fn weighted_rejection_scales_sigma_to_each_frames_noise() {
        let rejection = RejectionMode::DeltaSigma(DeltaSigmaOptions {
            warmup_samples: 5,
            ..DeltaSigmaOptions::default()
        });
        let rejected_fraction = |diagnostics: &[(f64, FrameDiagnostics)], noisy: bool| {
            let (rejected, total) = diagnostics
                .iter()
                .filter(|(sigma, _)| (*sigma > 20.0) == noisy)
                .fold((0, 0), |(rejected, total), (_, frame)| {
                    (
                        rejected + frame.rejected_samples,
                        total + frame.rejected_samples + frame.accepted_samples,
                    )
                });
            rejected as f64 / total as f64
        };
        let (_, weighted) = stack_mixed(FrameWeighting::inverse_noise_variance(), rejection);
        let (_, equal) = stack_mixed(FrameWeighting::Equal, rejection);
        let weighted_noisy = rejected_fraction(&weighted, true);
        let weighted_quiet = rejected_fraction(&weighted, false);
        let equal_noisy = rejected_fraction(&equal, true);
        eprintln!(
            "rejected: weighted noisy={weighted_noisy:.4} quiet={weighted_quiet:.4} equal noisy={equal_noisy:.4}"
        );
        // Live delta-sigma estimates sigma from few frames early on, so it
        // clips more than the 0.27% a 3-sigma cut takes from Gaussian noise.
        // Weighted rejection clips both kinds of frame at about the same
        // rate; pooled sigma clips the noisy frames far harder.
        assert!(weighted_noisy < 0.04, "{weighted_noisy}");
        assert!(weighted_quiet < 0.04, "{weighted_quiet}");
        assert!(
            (weighted_noisy / weighted_quiet - 1.0).abs() < 0.3,
            "{weighted_noisy} vs {weighted_quiet}"
        );
        assert!(equal_noisy > 4.0 * weighted_noisy, "{equal_noisy}");

        // A real outlier in a noisy frame is still removed.
        let (_, frames) = mixed_noise_frames();
        let mut frames = frames.into_iter();
        let (_, reference) = frames.next().unwrap();
        let mut stacker = LiveStacker::from_linear(
            reference,
            StackOptions {
                normalization: NormalizationMode::None,
                rejection,
                weighting: FrameWeighting::inverse_noise_variance(),
                ..StackOptions::default()
            },
        )
        .unwrap();
        let trail = 40 * WIDTH + 5;
        for (index, (_, mut frame)) in frames.enumerate() {
            if index == 16 {
                frame.data[trail] += 2_000.0;
            }
            stacker.push_linear(frame).unwrap();
        }
        let snapshot = stacker.snapshot().unwrap();
        assert!((snapshot.image.data[trail] - 1000.0).abs() < 20.0);
        assert_eq!(snapshot.rejected_samples[trail], 1);
    }

    #[test]
    fn weighted_batch_replay_matches_optimal_noise_and_removes_trails() {
        let (truth, frames) = mixed_noise_frames();
        let mut images: Vec<LinearImage> = frames.iter().map(|(_, frame)| frame.clone()).collect();
        let trail = 60 * WIDTH + 3;
        images[0].data[trail] += 3_000.0;
        let weights: Vec<Vec<f32>> = frames
            .iter()
            .map(|(sigma, _)| vec![(100.0 / (sigma * sigma)) as f32])
            .collect();
        let result = crate::integrate_registered_frames(
            images.len(),
            &crate::BatchStackOptions {
                frame_weights: Some(weights),
                ..crate::BatchStackOptions::default()
            },
            |_, index| Ok(images[index].clone()),
        )
        .unwrap();
        let equal = crate::integrate_registered_frames(
            images.len(),
            &crate::BatchStackOptions::default(),
            |_, index| Ok(images[index].clone()),
        )
        .unwrap();
        let optimal = 1.0
            / frames
                .iter()
                .map(|(sigma, _)| 1.0 / (sigma * sigma))
                .sum::<f64>()
                .sqrt();
        let weighted_noise = sky_residual(&result.snapshot.image, &truth);
        let equal_noise = sky_residual(&equal.snapshot.image, &truth);
        eprintln!("batch optimal={optimal:.4} equal={equal_noise:.4} weighted={weighted_noise:.4}");
        assert!((weighted_noise / optimal - 1.0).abs() < 0.05);
        assert!(weighted_noise < equal_noise * 0.8);
        assert_eq!(result.snapshot.rejected_samples[trail], 1);
        let noisy_rejected: usize = frames
            .iter()
            .zip(&result.frames)
            .filter(|((sigma, _), _)| *sigma > 20.0)
            .map(|(_, frame)| frame.finite_samples - frame.integrated_samples)
            .sum();
        let noisy_total: usize = frames
            .iter()
            .zip(&result.frames)
            .filter(|((sigma, _), _)| *sigma > 20.0)
            .map(|(_, frame)| frame.finite_samples)
            .sum();
        assert!((noisy_rejected as f64 / noisy_total as f64) < 0.012);

        let options = |frame_weights| crate::BatchStackOptions {
            frame_weights: Some(frame_weights),
            ..crate::BatchStackOptions::default()
        };
        let load = |_, _| LinearImage::new(1, 1, 1, vec![1.0]);
        assert!(crate::integrate_registered_frames(2, &options(vec![vec![1.0]]), load).is_err());
        assert!(
            crate::integrate_registered_frames(2, &options(vec![vec![1.0], vec![0.0]]), load)
                .is_err()
        );
        assert!(
            crate::integrate_registered_frames(2, &options(vec![vec![1.0], vec![f32::NAN]]), load)
                .is_err()
        );
        assert!(
            crate::integrate_registered_frames(
                2,
                &options(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]),
                load
            )
            .is_err(),
            "weights must match the channel count"
        );
    }

    fn weighted_stack() -> (LiveStacker, Vec<LinearImage>) {
        let (_, frames) = mixed_noise_frames();
        let mut frames = frames.into_iter().map(|(_, frame)| frame);
        let reference = frames.next().unwrap();
        let stacker = LiveStacker::from_linear(
            reference,
            StackOptions {
                normalization: NormalizationMode::None,
                rejection: RejectionMode::DeltaSigma(DeltaSigmaOptions {
                    warmup_samples: 4,
                    ..DeltaSigmaOptions::default()
                }),
                weighting: FrameWeighting::inverse_noise_variance(),
                ..StackOptions::default()
            },
        )
        .unwrap();
        (stacker, frames.take(9).collect())
    }

    #[test]
    fn weighted_context_round_trips_bit_exactly_as_version_4() {
        let (mut uninterrupted, frames) = weighted_stack();
        let (mut checkpointed, _) = weighted_stack();
        for frame in &frames[..5] {
            uninterrupted.push_linear(frame.clone()).unwrap();
            checkpointed.push_linear(frame.clone()).unwrap();
        }
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("weighted.seiza-stack");
        checkpointed.save_context(&path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[8..12], &4_u32.to_le_bytes());
        let mut resumed = LiveStacker::open_context(&path).unwrap();
        assert_eq!(
            bits(resumed.reference_noise()),
            bits(checkpointed.reference_noise())
        );
        assert_eq!(
            bits(resumed.accumulator.weight_sum.as_ref().unwrap()),
            bits(checkpointed.accumulator.weight_sum.as_ref().unwrap())
        );
        assert_eq!(
            resumed.configuration_fingerprint(),
            checkpointed.configuration_fingerprint()
        );
        let records = resumed.ledger.weights();
        assert_eq!(records, checkpointed.ledger.weights());
        assert_eq!(records.len(), 6);
        assert_eq!(records[0].weight, vec![1.0]);
        assert_eq!(
            bits(&records[0].noise),
            bits(checkpointed.reference_noise())
        );
        assert!(records[1..].iter().all(|record| record.weight.len() == 1));
        for frame in &frames[5..] {
            uninterrupted.push_linear(frame.clone()).unwrap();
            resumed.push_linear(frame.clone()).unwrap();
        }
        let expected = uninterrupted.into_snapshot().unwrap();
        let actual = resumed.into_snapshot().unwrap();
        assert_eq!(bits(&actual.image.data), bits(&expected.image.data));
        assert_eq!(bits(&actual.variance.data), bits(&expected.variance.data));
        assert_eq!(actual.coverage, expected.coverage);
        assert_eq!(actual.rejected_samples, expected.rejected_samples);
    }

    /// Rewrite the compressed payload of a context file.
    fn tamper(path: &Path, edit: impl FnOnce(&mut Vec<u8>, usize)) {
        let bytes = std::fs::read(path).unwrap();
        let mut payload = zstd::stream::decode_all(&bytes[12..]).unwrap();
        let metadata_length = u64::from_le_bytes(payload[..8].try_into().unwrap()) as usize;
        edit(&mut payload, 8 + metadata_length);
        let mut encoder = zstd::stream::write::Encoder::new(Vec::new(), 1).unwrap();
        encoder.include_checksum(true).unwrap();
        std::io::Write::write_all(&mut encoder, &payload).unwrap();
        let mut rewritten = bytes[..12].to_vec();
        rewritten.extend(encoder.finish().unwrap());
        std::fs::write(path, rewritten).unwrap();
    }

    #[test]
    fn tampered_weighted_contexts_are_refused() {
        let (mut stacker, frames) = weighted_stack();
        for frame in &frames[..3] {
            stacker.push_linear(frame.clone()).unwrap();
        }
        let samples = WIDTH * HEIGHT;
        // Reference, mean, m2, coverage, rejected, then weight sums.
        let weight_offset = move |arrays_start: usize| arrays_start + 5 * samples * 4;
        let directory = tempfile::tempdir().unwrap();
        type Edit<'a> = Box<dyn Fn(&mut Vec<u8>, usize) + 'a>;
        let cases: [(&str, Edit); 6] = [
            (
                "non-finite weight sum",
                Box::new(move |payload, start| {
                    let at = weight_offset(start) + 40;
                    payload[at..at + 4].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
                }),
            ),
            (
                "zero weight on a covered sample",
                Box::new(move |payload, start| {
                    let at = weight_offset(start) + 80;
                    payload[at..at + 4].copy_from_slice(&0.0_f32.to_bits().to_le_bytes());
                }),
            ),
            (
                "negative weight sum",
                Box::new(move |payload, start| {
                    let at = weight_offset(start);
                    payload[at..at + 4].copy_from_slice(&(-1.0_f32).to_bits().to_le_bytes());
                }),
            ),
            (
                "zero reference noise",
                Box::new(move |payload, start| {
                    let at = weight_offset(start) + samples * 4;
                    payload[at..at + 4].copy_from_slice(&0.0_f32.to_bits().to_le_bytes());
                }),
            ),
            (
                "truncated frame weight records",
                Box::new(move |payload, _| {
                    payload.pop();
                }),
            ),
            (
                "missing weight sums",
                Box::new(move |payload, start| {
                    let at = weight_offset(start);
                    payload.drain(at..at + samples * 4 + 4);
                }),
            ),
        ];
        for (name, edit) in cases {
            let path = directory.path().join(format!("{name}.seiza-stack"));
            stacker.save_context(&path).unwrap();
            LiveStacker::open_context(&path).unwrap();
            tamper(&path, |payload, start| edit(payload, start));
            assert!(
                matches!(
                    LiveStacker::open_context(&path),
                    Err(Error::StackContextRead { .. })
                ),
                "{name}"
            );
        }

        // A weighted stack relabelled as version 3, container and metadata
        // both, is refused: version 3 cannot hold weighted options.
        let path = directory.path().join("relabelled.seiza-stack");
        stacker.save_context(&path).unwrap();
        tamper(&path, |payload, start| {
            let metadata = std::str::from_utf8(&payload[8..start]).unwrap();
            let relabelled = metadata.replacen(r#""schema_version":4"#, r#""schema_version":3"#, 1);
            assert_ne!(metadata, relabelled);
            payload.splice(8..start, relabelled.into_bytes());
        });
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[8..12].copy_from_slice(&3_u32.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        let Err(error) = LiveStacker::open_context(&path) else {
            panic!("a relabelled weighted context must be refused");
        };
        let error = error.to_string();
        assert!(error.contains("cannot hold a weighted stack"), "{error}");
    }
}
