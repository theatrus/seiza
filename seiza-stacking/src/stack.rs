use crate::image::Placed;
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
#[derive(Clone, Debug, Deserialize, Serialize)]
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
    /// Sensor defects repaired in each light after calibration and before
    /// the cosmetic filter and debayering: what [`crate::detect_defects`]
    /// learns from dithered lights whose calibration has no master dark to
    /// subtract their hot pixels. A light of another size is left as it is.
    /// The default, `None`, is not serialized, so existing options,
    /// fingerprints and contexts keep their exact bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub defects: Option<crate::DefectMap>,
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
    /// How Bayer frames are demosaiced. The default, [`crate::Demosaic::Vng`],
    /// is not serialized, so existing options, fingerprints and contexts
    /// keep their exact bytes.
    #[serde(default, skip_serializing_if = "crate::Demosaic::is_vng")]
    pub demosaic: crate::Demosaic,
    /// How close a light's readings must be to its calibration masters'
    /// for them to calibrate it: the gate the reference, every pushed light
    /// and every replayed frame are judged under. A host that matched its
    /// masters to its lights under wider tolerances — a rig whose rotator
    /// scatters more than two degrees — sets the same ones here, or a master
    /// it chose on purpose is refused at the last step. Change it during a
    /// stack with [`LiveStacker::set_match_tolerances`]. The default,
    /// [`seiza_calibration::MatchTolerances::default`], is not serialized,
    /// so existing options, fingerprints and contexts keep their exact bytes.
    ///
    /// Only the tolerances a light is admitted by are written:
    /// `master_temperature_c` and `flat_session_seconds` decide how masters
    /// are built, so they change neither options nor fingerprint, and read
    /// back as their defaults.
    #[serde(
        default,
        skip_serializing_if = "crate::calibration::LightGate::admits_as_defaults",
        with = "crate::calibration::light_gate"
    )]
    pub match_tolerances: seiza_calibration::MatchTolerances,
    /// Area-mean bin each light `bin_factor` by `bin_factor`, from 1 to 4,
    /// after calibration, defect repair, the cosmetic filter and debayering,
    /// and before star detection. Registration, resampling, normalization,
    /// rejection and integration then work on a quarter of the pixels at 2
    /// and a sixteenth at 4, and the stack's grid is the binned reference's.
    /// [`LiveStacker::reintegrate_full_resolution`] later reintegrates the
    /// same frames at the sensor's own resolution without searching for
    /// their transforms again.
    ///
    /// Every pixel-valued option keeps its meaning in the sensor's pixels at
    /// any factor (the match tolerance, drift floor, registration RMS gate
    /// and local normalization tile), and registration RMS and drift are
    /// reported in them. Each frame's mapping, transform and the stack's
    /// outputs are on the binned grid. [`CfaIntegration::BayerDrizzle`]
    /// needs single photosites, so it is refused above 1. The default, 1, is
    /// not serialized, so existing options, fingerprints and contexts keep
    /// their exact bytes.
    #[serde(
        default = "native_bin_factor",
        skip_serializing_if = "is_native_bin_factor"
    )]
    pub bin_factor: u32,
    /// Which passes may run on the GPU, for every frame the stack is pushed
    /// and for its reintegration, which [`crate::BatchStackOptions::gpu`]
    /// can only narrow. They decide where the arithmetic runs, which moves
    /// the stack by rounding at most, so
    /// [`LiveStacker::configuration_fingerprint`] leaves them out; a saved
    /// context keeps them. The default, every pass, is not serialized, so
    /// existing options and contexts keep their exact bytes.
    #[serde(default, skip_serializing_if = "crate::GpuOptions::is_default")]
    pub gpu: crate::GpuOptions,
}

impl Default for StackOptions {
    fn default() -> Self {
        Self {
            registration: RegistrationOptions::default(),
            normalization: NormalizationMode::default(),
            rejection: RejectionMode::default(),
            acceptance: FrameAcceptanceCriteria::default(),
            cosmetic: None,
            defects: None,
            weighting: FrameWeighting::default(),
            cfa_integration: CfaIntegration::default(),
            interpolation: crate::Interpolation::default(),
            demosaic: crate::Demosaic::default(),
            match_tolerances: seiza_calibration::MatchTolerances::default(),
            bin_factor: native_bin_factor(),
            gpu: crate::GpuOptions::default(),
        }
    }
}

fn native_bin_factor() -> u32 {
    1
}

fn is_native_bin_factor(bin_factor: &u32) -> bool {
    *bin_factor == 1
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
        if let Some(defects) = &self.defects {
            defects.validate()?;
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
        crate::calibration::check_tolerances(&self.match_tolerances)?;
        if !(1..=4).contains(&self.bin_factor) {
            return Err(Error::Stack(format!(
                "bin factor must be 1, 2, 3 or 4, not {}",
                self.bin_factor
            )));
        }
        if self.bin_factor > 1 && self.cfa_integration == CfaIntegration::BayerDrizzle {
            return Err(Error::Stack(
                "Bayer drizzle integrates single photosites, which binning averages away; \
                 use a bin factor of 1"
                    .into(),
            ));
        }
        if self.demosaic == crate::Demosaic::Superpixel && !self.bin_factor.is_multiple_of(2) {
            return Err(Error::Stack(
                "a superpixel demosaic bins by 2 as it debayers, so it needs a bin factor of \
                 2 or 4"
                    .into(),
            ));
        }
        self.gpu.validate()
    }

    /// Where each channel of a light prepared from `cfa` samples the sky:
    /// off its pixels' centres after a superpixel demosaic, and on them
    /// otherwise.
    pub(crate) fn channel_sites(
        &self,
        cfa: Option<BayerLayout>,
    ) -> Option<crate::registration::ChannelSites> {
        self.channel_sites_at(cfa, self.bin_factor)
    }

    /// [`Self::channel_sites`] of a light binned by `bin_factor` in place of
    /// the stack's own.
    pub(crate) fn channel_sites_at(
        &self,
        cfa: Option<BayerLayout>,
        bin_factor: u32,
    ) -> Option<crate::registration::ChannelSites> {
        if self.demosaic != crate::Demosaic::Superpixel || !bin_factor.is_multiple_of(2) {
            return None;
        }
        Some(crate::registration::ChannelSites::superpixel(
            cfa?, bin_factor,
        ))
    }

    /// How pixel-valued options and constants read on the working grid.
    pub(crate) fn scale(&self) -> crate::binning::Scale {
        crate::binning::Scale::new(self.bin_factor)
    }

    /// Repair a calibrated light's sensor defects, then filter its
    /// impulses, as every light is before debayering.
    pub(crate) fn repair(&self, frame: &mut FitsFrame) -> Result<()> {
        if let Some(defects) = &self.defects {
            defects.repair(&mut frame.image, frame.bayer, frame.exposure_seconds)?;
        }
        if let Some(filter) = &self.cosmetic {
            crate::cosmetic::suppress_impulses(&mut frame.image, frame.bayer, filter)?;
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

impl FrameAcceptanceCriteria {
    /// The `minimum_overlap_fraction` for a canvas expanded to hold mosaic
    /// panels, where it is the fraction of the smaller footprint a frame must
    /// share with the canvas to join it. The CLI, the C ABI and Python use it
    /// when the caller sets none; [`LiveStacker::with_expanding_reference`]
    /// itself takes the options as they are. Panels planned for 10%
    /// overlap land short of 10%: on a real 2x2 mosaic planned that way, they
    /// overlapped by 0.088 to 0.094, and at 0.10 only the anchor's panel
    /// joined; at this fraction all four did, registering at a median 0.1px
    /// RMS. Overlap does not guard against chance matches: a match must
    /// pair more stars than chance would and a share of those the overlap
    /// should hold, and pass the scale and residual checks.
    pub const EXPANDED_MINIMUM_OVERLAP_FRACTION: f32 = 0.05;
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
    /// Registration RMS residual, in the sensor's pixels at any
    /// [`StackOptions::bin_factor`].
    pub registration_rms_pixels: f64,
    /// Frame-center displacement under the transform, in the sensor's
    /// pixels.
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

/// A binned stack reintegrated at the sensor's full resolution by
/// [`LiveStacker::reintegrate_full_resolution`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct FullResolutionStack {
    /// The reintegrated stack, on a grid [`StackOptions::bin_factor`] times
    /// as fine as the binned stack's, and each frame's statistics.
    pub result: crate::BatchStackResult,
    /// The reference frame's headers for that grid: its WCS carried there,
    /// its binning and pixel-size cards the sensor's again, and no
    /// `SEIZABIN`.
    pub reference_headers: Vec<(String, HeaderValue)>,
    /// Admitted frames whose transform was refined against stars detected
    /// at full resolution, or against the canvas the other frames build
    /// there: none without `refine`.
    pub refined_frames: usize,
    /// The median registration RMS of the admitted frames, in the sensor's
    /// pixels: a refined frame's own refit, and for every other frame the
    /// RMS the binned pass measured, which its carried transform is expected
    /// to keep. `None` when no frame's is known, as for a stack reopened
    /// from a context and not refined.
    pub registration_rms_pixels: Option<f64>,
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

impl FrameInputMode {
    /// `calibrate-and-prepare` or `prepared-only`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CalibrateAndPrepare => "calibrate-and-prepare",
            Self::PreparedOnly => "prepared-only",
        }
    }
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
    alignment_reference: Option<LinearImage>,
    planned_registrations: std::collections::HashMap<PathBuf, crate::expand::PlannedRegistration>,
    /// Registers frames to an expanded canvas through its panels.
    pub(crate) canvas_registrar: Option<crate::expand::CanvasRegistrar>,
    /// Per-channel noise of the reference frame, the unit of frame weights.
    /// Empty when frames are weighted equally.
    reference_noise: Vec<f32>,
    /// The reference's [`crate::normalization::photometric_ceilings`],
    /// measured the first time a frame needs them rather than for every
    /// frame: on a 26 MP colour reference that pass was 11% of preparing a
    /// frame.
    reference_ceilings: std::sync::OnceLock<Vec<f32>>,
    accumulator: Accumulator,
    reference_headers: Vec<(String, HeaderValue)>,
    pub(crate) accepted_frames: u32,
    pub(crate) rejected_frames: u32,
    input_paths: Vec<PathBuf>,
    input_mode: FrameInputMode,
    configuration_fingerprint: String,
    pub(crate) ledger: crate::replay::Ledger,
    /// Each admitted frame's registered image, kept on disk for a later
    /// reintegration; see [`Self::retain_frames_for_reintegration`].
    pub(crate) frame_cache: Option<crate::replay::FrameCache>,
    /// The Bayer layout the reference was demosaiced from, when this stack
    /// prepared it; a stack reopened from a context does not know it.
    pub(crate) reference_cfa: Option<BayerLayout>,
    /// Whether the prepared reference is also the first admitted image.
    pub(crate) reference_is_integrated: bool,
    /// The reference image a stack from
    /// [`Self::open_fits_with_reference_image`] aligns to. Outputs must not
    /// replace it, though it is no input until pushed as a light.
    reference_image_path: Option<PathBuf>,
}

enum ReferenceContribution {
    Expanded {
        noise: Vec<f32>,
        canvas: Box<crate::expand::CanvasRegistrar>,
    },
    Integrated(Option<BayerLayout>),
    External {
        image: LinearImage,
        registrar: Box<Registrar>,
        noise: Vec<f32>,
    },
}

impl LiveStacker {
    /// Start a stack from a reference FITS frame, calibrating and preparing it
    /// as the immutable alignment target.
    pub fn new(
        reference: FitsFrame,
        calibration: CalibrationMasters,
        options: StackOptions,
    ) -> Result<Self> {
        let source = crate::replay::FrameSource::of(&reference);
        let reference_metadata = reference.metadata();
        let clipped = crate::saturation::raw_clip_mask(&reference);
        let (reference, cfa) =
            prepare_light(reference, &calibration, &options.match_tolerances, &options)
                .map_err(LightFailure::into_error)?;
        let clipped = developed_clip_mask(clipped, cfa, &options);
        let mut stacker = Self::from_prepared(
            reference.image,
            reference.headers,
            reference_metadata,
            calibration,
            options,
            FrameInputMode::CalibrateAndPrepare,
            ReferenceContribution::Integrated(cfa),
        )?;
        // The reference is integrated where it lies, as it was prepared.
        if let Some(clipped) = clipped {
            let (width, height, channels) = (
                stacker.reference.width,
                stacker.reference.height,
                stacker.reference.channels,
            );
            stacker.accumulator.add_clipped(
                &clipped,
                ReferenceRegion {
                    x: 0,
                    y: 0,
                    width,
                    height,
                },
                width,
                channels,
            );
        }
        stacker.ledger.set_reference_source(source);
        Ok(stacker)
    }

    /// Start an empty stack on an external alignment grid. The calibrated
    /// channel reference supplies normalization and noise weighting, after
    /// registration onto that grid. The external image supplies only star
    /// positions, output dimensions, and WCS; its brightness is never used.
    /// Push all lights explicitly, including the channel reference.
    pub fn with_reference_image(
        reference: FitsFrame,
        channel_reference: FitsFrame,
        calibration: CalibrationMasters,
        options: StackOptions,
    ) -> Result<Self> {
        Self::with_reference_image_candidates(
            reference,
            [Ok::<_, Error>(channel_reference)],
            calibration,
            options,
            |_, _| {},
        )
        .map(|(stacker, _)| stacker)
    }

    /// [`Self::with_reference_image`] with channel references to try in
    /// turn, best first, until one aligns to the external image, which is
    /// prepared once. `rejected` hears the index and reason of each one that
    /// does not align. Returns the stack and the index of the channel
    /// reference it uses.
    ///
    /// A candidate that cannot be read or calibrated, or whose channel count
    /// differs from the external image's, ends the search with an error:
    /// the next candidate would fail the same way.
    pub fn with_reference_image_candidates<E: From<Error>>(
        reference: FitsFrame,
        channel_references: impl IntoIterator<Item = std::result::Result<FitsFrame, E>>,
        calibration: CalibrationMasters,
        options: StackOptions,
        mut rejected: impl FnMut(usize, &FrameRejectionReason),
    ) -> std::result::Result<(Self, usize), E> {
        options.validate()?;
        // The external image is debayered and binned as the lights are, so
        // that registration finds them at the scale it expects, and its
        // colours line up as theirs will.
        let (mut alignment, cfa) = develop(reference, &options, options.bin_factor)?;
        alignment.image = co_site(
            alignment.image,
            options.channel_sites(cfa),
            options.interpolation,
            &options.gpu,
        )?;
        let registrar = Registrar::at_scale(
            &alignment.image,
            &options.registration,
            options.scale(),
            &options.gpu,
        )?;
        // Use the regular registration gates, with no photometric comparison
        // to the external image. This image is a target, not an observation.
        let geometry_options = StackOptions {
            normalization: NormalizationMode::None,
            weighting: FrameWeighting::Equal,
            cfa_integration: CfaIntegration::Demosaic,
            ..options.clone()
        };
        let mut first_rejection = None;
        let mut tried = 0;
        for (index, channel_reference) in channel_references.into_iter().enumerate() {
            let channel_reference = channel_reference?;
            tried += 1;
            let metadata = channel_reference.metadata();
            let (channel, channel_cfa) = prepare_light(
                channel_reference,
                &calibration,
                &options.match_tolerances,
                &options,
            )
            .map_err(LightFailure::into_error)?;
            let noise = if options.weighting.is_equal() {
                Vec::new()
            } else {
                measure_reference_noise(&channel.image)?
            };
            match prepare_frame(
                &alignment.image,
                FrameRegistration {
                    registrar: &registrar,
                    canvas: None,
                    known: None,
                    ceilings: &std::sync::OnceLock::new(),
                },
                &geometry_options,
                &[],
                channel.image,
                channel_cfa,
                None,
                None,
            )? {
                PreparedFrame::Ready(ready) => {
                    let stacker = Self::from_prepared(
                        ready.registered,
                        alignment.headers,
                        metadata,
                        calibration,
                        options,
                        FrameInputMode::CalibrateAndPrepare,
                        ReferenceContribution::External {
                            image: alignment.image,
                            registrar: Box::new(registrar),
                            noise,
                        },
                    )?;
                    return Ok((stacker, index));
                }
                PreparedFrame::Rejected(reason @ FrameRejectionReason::IncompatibleImage(_)) => {
                    return Err(Error::Stack(format!(
                        "channel reference could not be aligned to the reference image: {reason}"
                    ))
                    .into());
                }
                PreparedFrame::Rejected(reason) => {
                    rejected(index, &reason);
                    first_rejection.get_or_insert(reason);
                }
            }
        }
        Err(Error::Stack(match (first_rejection, tried) {
            (None, _) => "no channel reference was given".into(),
            (Some(reason), 1) => {
                format!("channel reference could not be aligned to the reference image: {reason}")
            }
            (Some(reason), tried) => format!(
                "none of {tried} channel references could be aligned to the reference image; \
                 the first: {reason}"
            ),
        })
        .into())
    }

    /// Plan the union of a connected set of overlapping light frames, then
    /// start an empty stack on that grid. The first path anchors the WCS and
    /// photometric scale. Every path must subsequently be pushed explicitly.
    /// Planning places frames by similarity at any rotation. A frame that
    /// cannot join is rejected, with the reason, when pushed; planning fails
    /// when nothing joins the first path, an input changes or cannot be read,
    /// or the canvas limit is exceeded. `maximum_samples` bounds
    /// the canvas's pixels times channels; with frames being prepared, a
    /// stack peaks at about 40 bytes per sample. Each frame registers against the panels it overlaps, so a
    /// narrow overlap need not hold the canvas's brightest stars, and a frame
    /// lying mostly within one panel also gets the registration model's
    /// warp against that panel. Each panel starts from a frame on the
    /// anchor's side of the pier where one connects. Bayer drizzle and
    /// drizzled reintegration are not supported on the canvas.
    pub fn with_expanding_reference(
        paths: &[PathBuf],
        calibration: CalibrationMasters,
        options: StackOptions,
        maximum_samples: usize,
        progress: impl FnMut(&Path, usize, usize, usize) -> crate::Continue,
    ) -> Result<Self> {
        let options = Self::expanding_options(options)?;
        let plan = crate::expand::plan(paths, &calibration, &options, maximum_samples, progress)?;
        Self::from_expansion(plan, calibration, options)
    }

    /// `options` checked and adjusted for planning an expanded canvas.
    pub(crate) fn expanding_options(mut options: StackOptions) -> Result<StackOptions> {
        options.validate()?;
        if options.cfa_integration == CfaIntegration::BayerDrizzle {
            return Err(Error::Stack(
                "an expanded canvas does not support Bayer drizzle".into(),
            ));
        }
        // Sessions differ in rotation whenever the camera comes off between
        // them; matching, residual, and scale checks guard against a false
        // placement instead.
        options.acceptance.maximum_rotation_degrees = 180.0;
        Ok(options)
    }

    /// An empty stack on the canvas `plan` holds, planned with `options`
    /// from [`Self::expanding_options`].
    pub(crate) fn from_expansion(
        plan: crate::expand::Expansion,
        calibration: CalibrationMasters,
        mut options: StackOptions,
    ) -> Result<Self> {
        // Panels bound each registration's search; the canvas diagonal bounds
        // any frame's displacement from its center. Options keep sensor
        // pixels, and the canvas is on the binned grid.
        options.registration.maximum_drift_pixels = options
            .scale()
            .native_length((plan.image.width as f64).hypot(plan.image.height as f64));
        let mut stacker = Self::from_prepared(
            plan.image,
            plan.headers,
            plan.metadata,
            calibration,
            options,
            FrameInputMode::CalibrateAndPrepare,
            ReferenceContribution::Expanded {
                noise: plan.noise,
                canvas: Box::new(plan.canvas),
            },
        )?;
        stacker.planned_registrations = plan.registrations;
        Ok(stacker)
    }

    /// [`Self::with_expanding_reference`] from FITS or XISF paths, with the
    /// duplicate-input and output-path protection of [`Self::open_fits`] and
    /// the masters' paths kept, so a reopened context can still reintegrate.
    /// Every light must differ from every master. Push every light, the
    /// anchor included.
    #[allow(clippy::too_many_arguments)]
    pub fn open_fits_expanding<P: AsRef<Path>>(
        paths: &[P],
        bias_path: Option<&Path>,
        dark_path: Option<&Path>,
        flat_path: Option<&Path>,
        dark_exposure_seconds: Option<f64>,
        options: StackOptions,
        maximum_samples: usize,
        progress: impl FnMut(&Path, usize, usize, usize) -> crate::Continue,
    ) -> Result<Self> {
        let masters = [bias_path, dark_path, flat_path];
        for path in paths {
            distinct_input_paths(std::iter::once(Some(path.as_ref())).chain(masters))?;
        }
        let calibration = CalibrationMasters::from_fits_paths(
            bias_path,
            dark_path,
            flat_path,
            dark_exposure_seconds,
        )?;
        let paths = paths
            .iter()
            .map(|path| path.as_ref().to_path_buf())
            .collect::<Vec<_>>();
        let mut stacker = Self::with_expanding_reference(
            &paths,
            calibration,
            options,
            maximum_samples,
            progress,
        )?;
        stacker.input_paths = distinct_input_paths(masters)?;
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

    /// The lens distortion an expanded canvas was planned through, when
    /// planning found the panels' overlaps bent by one.
    pub fn expanded_lens(&self) -> Option<crate::ExpandedLens> {
        self.canvas_registrar
            .as_ref()
            .and_then(crate::expand::CanvasRegistrar::lens_summary)
    }

    /// Notes from expansion planning: how many lights did not join the
    /// canvas, and how close those that matched a panel came to the minimum
    /// overlap; metadata conflicts that skipped the shared lens fit; and a
    /// lens-corrected plan that failed. Empty for ordinary stacks and
    /// reopened contexts, which do not rerun planning.
    pub fn expansion_planning_notes(&self) -> &[String] {
        self.canvas_registrar
            .as_ref()
            .map_or(&[], |canvas| canvas.planning_notes())
    }

    /// The largest frame displacement registration accepts, in the sensor's
    /// pixels: on an expanded canvas, its diagonal.
    pub fn effective_maximum_drift_pixels(&self) -> f64 {
        let scale = self.options.scale();
        scale.native_length(
            scale
                .registration(&self.options.registration)
                .effective_maximum_drift_pixels(self.reference.width, self.reference.height),
        )
    }

    /// How deep this stack is: [`crate::measure_depth`] of its
    /// [`Self::view`], with the bin factor its lights took.
    pub fn measure_depth(&self) -> Option<crate::SnrSample> {
        crate::measure_depth(self.view()).map(|sample| crate::SnrSample {
            bin_factor: self.options.bin_factor,
            ..sample
        })
    }

    /// The factor each light is binned by: [`StackOptions::bin_factor`].
    /// The reference, snapshots and every frame's mapping are on the binned
    /// grid; see [`Self::reintegrate_full_resolution`].
    pub fn bin_factor(&self) -> u32 {
        self.options.bin_factor
    }

    /// [`Self::reintegrate`] at the sensor's full resolution: every admitted
    /// frame of a binned stack read again unbinned and integrated, with
    /// three-pass leave-one-out rejection, onto a grid
    /// [`Self::bin_factor`] times as fine as this stack's. At a bin factor of
    /// 1 it is [`Self::reintegrate`].
    ///
    /// No admission decision is revisited, and no transform is searched for
    /// again: each frame's mapping is carried exactly from the binned grid
    /// (see [`RegisteredFrameMapping::rebinned`]), with its weight and
    /// normalization. With `refine`, a single field then detects each
    /// frame's stars at full resolution, as it does the reference frame's,
    /// read again from its source, and refits the frame's similarity from
    /// that seed without searching, and the registration model's warp
    /// around it. Binning costs registration little at 2 and up to a few
    /// tenths of a pixel at 4 on tight stars, which the refit recovers. A
    /// frame whose stars do not pair, or whose refit fails the registration
    /// RMS gate, keeps its carried mapping.
    ///
    /// An expanded canvas has no full-resolution reference, so it builds
    /// one: every frame is carried onto the finer canvas through its
    /// carried mapping and summed, and each frame's similarity is then
    /// refit against the stars of the canvas the other frames make over its
    /// footprint. A frame warped onto its panel, by the registration model
    /// or a lens, keeps its carried warp, as does one whose refit would move
    /// a corner further than the match tolerance, as a narrow overlap could
    /// pull it. [`FullResolutionStack`] reports the median RMS, refined or
    /// carried, in the sensor's pixels, so a host can warn.
    ///
    /// The refit reads each frame once on a single field. On a canvas it
    /// first reads every frame, then reads the unwarped frames again; a
    /// canvas with only warped frames needs no refit reads. Registered
    /// images wait in [`crate::BatchStackOptions::scratch_directory`] for
    /// the integration that follows; `progress` hears each refit read as
    /// [`crate::BatchStackPass::Estimate`] for each frame in turn, before the
    /// reintegration's own progress. The
    /// live stack is left as it was. Its frames are refused as
    /// [`Self::reintegrate`] refuses them, and memory is reintegration's on
    /// the finer grid.
    pub fn reintegrate_full_resolution(
        &self,
        options: &crate::BatchStackOptions,
        refine: bool,
        mut progress: impl FnMut(crate::BatchStackPass, usize, usize),
    ) -> Result<FullResolutionStack> {
        if let Some(reason) = self.reintegration_unavailable() {
            return Err(Error::Stack(reason));
        }
        let live = self.ledger.registration_rms();
        if self.options.bin_factor == 1 {
            return Ok(FullResolutionStack {
                result: self.reintegrate(options, progress)?,
                reference_headers: self.reference_headers.clone(),
                refined_frames: 0,
                registration_rms_pixels: median_registration_rms(&live, &[]),
            });
        }
        let (full, refined) = self.full_resolution(options, refine, &mut progress)?;
        let result = full.reintegrate(options, &mut progress)?;
        Ok(FullResolutionStack {
            result,
            reference_headers: full.reference_headers,
            refined_frames: refined.iter().flatten().count(),
            registration_rms_pixels: median_registration_rms(&live, &refined),
        })
    }

    /// This binned stack at the sensor's full resolution, ready to
    /// reintegrate: every frame's mapping carried there exactly, and
    /// refined when `refine` asks, with each refined frame's RMS. See
    /// [`Self::reintegrate_full_resolution`].
    pub(crate) fn full_resolution(
        &self,
        options: &crate::BatchStackOptions,
        refine: bool,
        progress: &mut dyn FnMut(crate::BatchStackPass, usize, usize),
    ) -> Result<(Self, Vec<Option<f64>>)> {
        // Validate an explicitly chosen scratch directory before reading
        // the native reference or allocating a full mosaic-sized grid.
        let frame_cache = self.full_resolution_cache(options.scratch_directory.as_deref())?;
        let bin = self.options.bin_factor;
        let factor = bin as usize;
        let (width, height, channels) = (
            self.reference.width * factor,
            self.reference.height * factor,
            self.reference.channels,
        );
        let full_reference = if refine && self.canvas_registrar.is_none() {
            self.full_resolution_reference()?
                .map(|image| {
                    image.crop(ReferenceRegion {
                        x: 0,
                        y: 0,
                        width,
                        height,
                    })
                })
                .transpose()?
        } else {
            None
        };
        let registrar = full_reference
            .as_ref()
            .map(|reference| {
                Registrar::at_scale(
                    reference,
                    &self.options.registration,
                    crate::binning::Scale::NATIVE,
                    &self.options.gpu,
                )
            })
            .transpose()?;
        // A grid of the finer size: the reference frame itself where it was
        // read again, which the replay reads only for its shape.
        let reference = match full_reference {
            Some(reference) => reference,
            None => LinearImage::new(
                width,
                height,
                channels,
                vec![f32::NAN; width * height * channels],
            )?,
        };
        let mut full = Self {
            options: StackOptions {
                bin_factor: 1,
                // A superpixel is a bin of 2; full resolution demosaics.
                demosaic: match self.options.demosaic {
                    crate::Demosaic::Superpixel => crate::Demosaic::Vng,
                    demosaic => demosaic,
                },
                ..self.options.clone()
            },
            calibration: self.calibration.clone(),
            reference,
            reference_metadata: self.reference_metadata.clone(),
            registrar: registrar.clone().unwrap_or_else(|| self.registrar.clone()),
            alignment_reference: None,
            planned_registrations: Default::default(),
            // Replay reads only whether there is one, to resample each frame
            // over its footprint.
            canvas_registrar: self.canvas_registrar.clone(),
            reference_noise: self.reference_noise.clone(),
            // Of the full-resolution reference, not this one's.
            reference_ceilings: std::sync::OnceLock::new(),
            accumulator: Accumulator::new(0, !self.options.weighting.is_equal()),
            reference_headers: crate::fits::rebinned_headers(&self.reference_headers, bin, 1),
            accepted_frames: self.accepted_frames,
            rejected_frames: self.rejected_frames,
            input_paths: self.input_paths.clone(),
            input_mode: self.input_mode,
            configuration_fingerprint: self.configuration_fingerprint.clone(),
            ledger: self.ledger.rebinned(1)?,
            frame_cache,
            reference_cfa: None,
            reference_is_integrated: self.reference_is_integrated,
            reference_image_path: self.reference_image_path.clone(),
        };
        let refined = match &registrar {
            Some(registrar) => {
                full.refine_full_resolution(registrar, options.cancel.as_ref(), progress)?
            }
            None if refine && full.canvas_registrar.is_some() => {
                full.refine_canvas_full_resolution(options.cancel.as_ref(), progress)?
            }
            None => Vec::new(),
        };
        Ok((full, refined))
    }

    /// The reference frame at the sensor's full resolution, read again: the
    /// integrated reference frame's source, or the external reference
    /// image. `None` for an external image held only in memory.
    fn full_resolution_reference(&self) -> Result<Option<LinearImage>> {
        if self.reference_is_integrated {
            return self.read_full_resolution(0).map(Some);
        }
        let Some(path) = &self.reference_image_path else {
            return Ok(None);
        };
        let (frame, _) = develop(FitsFrame::open(path)?, &self.options, 1)?;
        Ok(Some(frame.image))
    }

    /// Start a stack from an already-prepared linear reference, with no
    /// calibration and no header metadata. Every later frame must use
    /// [`Self::push_linear`]. The reference, and every frame pushed, is
    /// binned by [`StackOptions::bin_factor`].
    pub fn from_linear(reference: LinearImage, options: StackOptions) -> Result<Self> {
        options.validate()?;
        let reference_metadata = FrameMetadata::from_image(&reference, &[]);
        let reference = reference.into_binned(options.bin_factor)?;
        Self::from_prepared(
            reference,
            Vec::new(),
            reference_metadata,
            CalibrationMasters::default(),
            options,
            FrameInputMode::PreparedOnly,
            ReferenceContribution::Integrated(None),
        )
    }

    /// Start a stack from a frame that a caller has already calibrated and
    /// prepared, while retaining its FITS headers for the output.
    ///
    /// This is the extension point for bounded corrections that must run
    /// between ordinary calibration and registration. A raw CFA frame is
    /// rejected so it cannot bypass preparation by mistake. Every later frame
    /// must use [`Self::push_linear`]. The reference, and every frame pushed,
    /// is binned by [`StackOptions::bin_factor`].
    pub fn from_prepared_frame(reference: FitsFrame, options: StackOptions) -> Result<Self> {
        if reference.bayer.is_some() {
            return Err(Error::Stack(
                "an already-prepared reference frame must not retain a Bayer layout".into(),
            ));
        }
        options.validate()?;
        let reference_metadata = reference.metadata();
        let reference = reference.binned(options.bin_factor)?;
        Self::from_prepared(
            reference.image,
            reference.headers,
            reference_metadata,
            CalibrationMasters::default(),
            options,
            FrameInputMode::PreparedOnly,
            ReferenceContribution::Integrated(None),
        )
    }

    fn from_prepared(
        reference: LinearImage,
        reference_headers: Vec<(String, HeaderValue)>,
        reference_metadata: FrameMetadata,
        calibration: CalibrationMasters,
        options: StackOptions,
        input_mode: FrameInputMode,
        contribution: ReferenceContribution,
    ) -> Result<Self> {
        options.validate()?;
        let configuration_fingerprint =
            stack_configuration_fingerprint(&options, &calibration, input_mode)?;
        let reference_is_integrated = matches!(contribution, ReferenceContribution::Integrated(_));
        let mut canvas_registrar = None;
        let (registrar, reference_noise, reference_cfa, alignment_reference) = match contribution {
            ReferenceContribution::Expanded { noise, canvas } => {
                let registrar = canvas.anchor_registrar();
                canvas_registrar = Some(*canvas);
                (registrar, noise, None, None)
            }
            ReferenceContribution::Integrated(cfa) => (
                Registrar::at_scale(
                    &reference,
                    &options.registration,
                    options.scale(),
                    &options.gpu,
                )?,
                if options.weighting.is_equal() {
                    Vec::new()
                } else {
                    measure_reference_noise(&reference)?
                },
                cfa,
                None,
            ),
            ReferenceContribution::External {
                image,
                registrar,
                noise,
            } => (*registrar, noise, None, Some(image)),
        };
        // A reference whose colours sample the sky off its pixels' centres
        // is resampled onto them once, as every later frame is when it is
        // registered; its stars and noise were taken as the lights' are.
        let reference = if reference_is_integrated {
            co_site(
                reference,
                options.channel_sites(reference_cfa),
                options.interpolation,
                &options.gpu,
            )?
        } else {
            reference
        };
        let mut accumulator =
            Accumulator::new(reference.sample_count(), !options.weighting.is_equal());
        if reference_is_integrated {
            // The reference is the unit of weight: it integrates with weight 1.
            // Under Bayer drizzle it contributes its own photosites, like every
            // later frame, while the demosaiced image stays the alignment target.
            match reference_cfa.filter(|_| options.cfa_integration == CfaIntegration::BayerDrizzle)
            {
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
        }
        let mut ledger = if reference_is_integrated {
            crate::replay::Ledger::new(&reference, options.match_tolerances)
        } else {
            crate::replay::Ledger::empty()
        };
        ledger.set_bin_factor(options.bin_factor);
        ledger.set_reference_channel_sites(options.channel_sites(reference_cfa));
        if !reference_noise.is_empty() {
            ledger.set_reference_weighting(&reference_noise);
        }
        Ok(Self {
            options,
            calibration,
            reference,
            reference_metadata,
            registrar,
            alignment_reference,
            planned_registrations: Default::default(),
            canvas_registrar,
            reference_noise,
            reference_ceilings: std::sync::OnceLock::new(),
            accumulator,
            reference_headers,
            accepted_frames: u32::from(reference_is_integrated),
            rejected_frames: 0,
            input_paths: Vec::new(),
            input_mode,
            configuration_fingerprint,
            ledger,
            frame_cache: None,
            reference_cfa,
            reference_is_integrated,
            reference_image_path: None,
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
        let input_paths =
            distinct_input_paths([Some(reference_path), bias_path, dark_path, flat_path])?;
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

    /// [`Self::with_reference_image`] from FITS or XISF paths, with the
    /// duplicate-input and output-path protection of [`Self::open_fits`].
    ///
    /// The stack starts empty: push the channel reference like every other
    /// light. The reference image may also be one of the lights, though not a
    /// calibration master, and [`Self::protected_paths`] keeps outputs off it.
    pub fn open_fits_with_reference_image(
        reference_image_path: impl AsRef<Path>,
        channel_reference_path: impl AsRef<Path>,
        bias_path: Option<&Path>,
        dark_path: Option<&Path>,
        flat_path: Option<&Path>,
        dark_exposure_seconds: Option<f64>,
        options: StackOptions,
    ) -> Result<Self> {
        Self::open_fits_with_reference_image_candidates(
            reference_image_path,
            &[channel_reference_path],
            bias_path,
            dark_path,
            flat_path,
            dark_exposure_seconds,
            options,
            |_, _| {},
        )
        .map(|(stacker, _)| stacker)
    }

    /// [`Self::open_fits_with_reference_image`] with channel references to
    /// try in turn, best first, as [`Self::with_reference_image_candidates`]
    /// does. Returns the stack and the index of the channel reference it
    /// uses.
    #[allow(clippy::too_many_arguments)]
    pub fn open_fits_with_reference_image_candidates<P: AsRef<Path>>(
        reference_image_path: impl AsRef<Path>,
        channel_reference_paths: &[P],
        bias_path: Option<&Path>,
        dark_path: Option<&Path>,
        flat_path: Option<&Path>,
        dark_exposure_seconds: Option<f64>,
        options: StackOptions,
        rejected: impl FnMut(usize, &FrameRejectionReason),
    ) -> Result<(Self, usize)> {
        let reference_image_path = reference_image_path.as_ref();
        let masters = [bias_path, dark_path, flat_path];
        distinct_input_paths(std::iter::once(Some(reference_image_path)).chain(masters))?;
        for path in channel_reference_paths {
            distinct_input_paths(std::iter::once(Some(path.as_ref())).chain(masters))?;
        }
        let calibration = CalibrationMasters::from_fits_paths(
            bias_path,
            dark_path,
            flat_path,
            dark_exposure_seconds,
        )?;
        let (mut stacker, used) = Self::with_reference_image_candidates(
            FitsFrame::open(reference_image_path)?,
            channel_reference_paths.iter().map(FitsFrame::open),
            calibration,
            options,
            rejected,
        )?;
        stacker.input_paths = distinct_input_paths(masters)?;
        stacker.reference_image_path = Some(path_identity(reference_image_path));
        stacker
            .ledger
            .set_current_calibration(crate::replay::CalibrationRecord::from_paths(
                bias_path,
                dark_path,
                flat_path,
                dark_exposure_seconds,
            ));
        Ok((stacker, used))
    }

    /// Restore an atomically checkpointed live stack, including its immutable
    /// registration reference, calibration, online moments, and source ledger.
    pub fn open_context(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut restored = context::read(path)?;
        if restored.expanded_lens.is_some() {
            // Early lens contexts retained the raw anchor's WCS even though
            // their pixels were already on the undistorted canvas.
            restored
                .reference_headers
                .retain(|(key, _)| !crate::fits::preserve_wcs_key(key));
        }
        let read_error = |error: Error| Error::StackContextRead {
            path: path.to_path_buf(),
            message: error.to_string(),
        };
        let canvas_registrar = (!restored.expanded_panels.is_empty())
            .then(|| {
                crate::expand::CanvasRegistrar::new(
                    &restored.reference,
                    &restored.expanded_panels,
                    restored.expanded_panel_stars,
                    restored.expanded_lens,
                    &restored.options.registration,
                    restored.options.scale(),
                    &restored.options.gpu,
                )
                .and_then(|canvas| canvas.with_projections(restored.expanded_panel_projections))
            })
            .transpose()
            .map_err(read_error)?;
        let registrar = match &canvas_registrar {
            Some(canvas) => canvas.anchor_registrar(),
            None => Registrar::at_scale(
                restored
                    .alignment_reference
                    .as_ref()
                    .unwrap_or(&restored.reference),
                &restored.options.registration,
                restored.options.scale(),
                &restored.options.gpu,
            )
            .map_err(read_error)?,
        };
        // A context's compact ledger leaves each mapping's bin factor out;
        // the options carry it.
        let mut ledger = restored.ledger.unwrap_or_else(crate::replay::Ledger::empty);
        ledger.set_bin_factor(restored.options.bin_factor);
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
            alignment_reference: restored.alignment_reference,
            planned_registrations: Default::default(),
            canvas_registrar,
            reference_noise: restored.reference_noise,
            reference_ceilings: std::sync::OnceLock::new(),
            accumulator: Accumulator::from_moments(
                restored.mean,
                restored.m2,
                restored.count,
                restored.rejected,
                restored.weight_sum,
                restored.clipped,
            ),
            reference_headers: restored.reference_headers,
            accepted_frames: restored.accepted_frames,
            rejected_frames: restored.rejected_frames,
            input_paths: restored.input_paths,
            input_mode: restored.input_mode,
            configuration_fingerprint,
            ledger,
            frame_cache: None,
            reference_cfa: None,
            reference_is_integrated: restored.reference_is_integrated,
            reference_image_path: restored.reference_image_path,
        })
    }

    /// Atomically checkpoint all state required to reopen this stack and keep
    /// integrating frames with identical online rejection behavior.
    pub fn save_context(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if self
            .protected_paths()
            .iter()
            .any(|input| paths_refer_to_same_file(input, path))
        {
            return Err(Error::StackContextWrite {
                path: path.to_path_buf(),
                message: "context path must not replace a stack input or calibration master".into(),
            });
        }
        let moments = self.accumulator.moments();
        self.accumulator.check_kept()?;
        context::write(
            path,
            context::ContextWriteState {
                reference_is_integrated: self.reference_is_integrated,
                alignment_reference: self.alignment_reference.as_ref(),
                reference_image_path: self.reference_image_path.as_deref(),
                options: &self.options,
                calibration: &self.calibration,
                reference: &self.reference,
                reference_headers: &self.reference_headers,
                reference_metadata: &self.reference_metadata,
                mean: moments.mean,
                m2: moments.m2,
                count: moments.count,
                rejected: moments.rejected,
                weight_sum: moments.weight_sum,
                reference_noise: &self.reference_noise,
                accepted_frames: self.accepted_frames,
                rejected_frames: self.rejected_frames,
                input_paths: &self.input_paths,
                input_mode: self.input_mode,
                ledger: &self.ledger,
                expanded_panels: &self
                    .canvas_registrar
                    .as_ref()
                    .map(crate::expand::CanvasRegistrar::regions)
                    .unwrap_or_default(),
                expanded_panel_stars: self
                    .canvas_registrar
                    .as_ref()
                    .map(crate::expand::CanvasRegistrar::panel_stars)
                    .unwrap_or_default(),
                expanded_panel_projections: self
                    .canvas_registrar
                    .as_ref()
                    .map(crate::expand::CanvasRegistrar::panel_projections)
                    .unwrap_or_default(),
                expanded_lens: self
                    .canvas_registrar
                    .as_ref()
                    .and_then(crate::expand::CanvasRegistrar::lens),
                clipped: self.accumulator.clipped.as_deref(),
            },
        )
    }

    /// The calibration masters currently applied to pushed frames.
    ///
    /// Read access so a host can ask questions of the active set — which
    /// masters a prospective light could accept, and why not — without
    /// keeping its own copy in sync with every swap.
    pub fn calibration(&self) -> &CalibrationMasters {
        &self.calibration
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
    ///
    /// Each light is judged under [`StackOptions::match_tolerances`], which
    /// a swap leaves as it was. Masters supplied here live only in memory:
    /// once they are replaced, the frames admitted under them can no longer
    /// be reintegrated (see [`LiveStacker::reintegration_unavailable`]),
    /// because replay has no file to load them from again.
    pub fn set_calibration(&mut self, calibration: CalibrationMasters) -> Result<()> {
        self.require_fits_input_mode()?;
        calibration.validate_master_set_signatures()?;
        crate::context::validate_calibration(
            &self.reference,
            &calibration,
            self.reference_is_integrated,
        )
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
    ///
    /// Lights pushed under these masters are judged under
    /// [`StackOptions::match_tolerances`], which the swap leaves as it was.
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
    pub fn push(&mut self, frame: FitsFrame) -> Result<FrameDisposition> {
        self.require_fits_input_mode()?;
        let source = crate::replay::FrameSource::of(&frame);
        let clipped = crate::saturation::raw_clip_mask(&frame);
        let (frame, cfa) = match prepare_light(
            frame,
            &self.calibration,
            &self.options.match_tolerances,
            &self.options,
        ) {
            Ok(prepared) => prepared,
            Err(failure) => return Ok(self.reject(failure.rejection())),
        };
        let cache = self
            .frame_cache
            .as_ref()
            .filter(|_| source.is_some())
            .map(crate::replay::FrameCache::files);
        let prepared = prepare_frame(
            &self.reference,
            FrameRegistration {
                registrar: &self.registrar,
                canvas: self.canvas_registrar.as_ref(),
                known: crate::expand::registration_for(
                    &self.planned_registrations,
                    frame.source.as_ref(),
                )?,
                ceilings: &self.reference_ceilings,
            },
            &self.options,
            &self.reference_noise,
            frame.image,
            cfa,
            cache,
            developed_clip_mask(clipped, cfa, &self.options),
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
    /// frame, applying every admission gate. It is binned by
    /// [`StackOptions::bin_factor`] first.
    pub fn push_linear(&mut self, frame: LinearImage) -> Result<FrameDisposition> {
        let frame = match frame.into_binned(self.options.bin_factor) {
            Ok(frame) => frame,
            Err(error) => {
                return Ok(self.reject(FrameRejectionReason::IncompatibleImage(error.to_string())));
            }
        };
        // A frame with no source file cannot be replayed, so none is kept.
        let prepared = prepare_frame(
            &self.reference,
            FrameRegistration {
                registrar: &self.registrar,
                canvas: self.canvas_registrar.as_ref(),
                known: None,
                ceilings: &self.reference_ceilings,
            },
            &self.options,
            &self.reference_noise,
            frame,
            None,
            None,
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
        let (cache_files, cache_admissions) = match self.frame_cache.as_mut() {
            Some(cache) => {
                let (files, admissions) = cache.split();
                (Some(files), Some(admissions))
            }
            None => (None, None),
        };
        (
            PreparationHalf {
                reference: &self.reference,
                registrar: &self.registrar,
                planned_registrations: &self.planned_registrations,
                canvas_registrar: self.canvas_registrar.as_ref(),
                calibration: &self.calibration,
                options: &self.options,
                reference_noise: &self.reference_noise,
                reference_ceilings: &self.reference_ceilings,
                frame_cache: cache_files,
            },
            IntegrationHalf {
                accumulator: &mut self.accumulator,
                options: &self.options,
                accepted_frames: &mut self.accepted_frames,
                rejected_frames: &mut self.rejected_frames,
                input_paths: &mut self.input_paths,
                ledger: &mut self.ledger,
                frame_cache: cache_admissions,
                expanded_canvas: self.canvas_registrar.is_some(),
                reference_width: self.reference.width,
                reference_height: self.reference.height,
            },
        )
    }

    /// Integrate what [`prepare_frame`] produced, in the caller's order.
    pub(crate) fn integrate_prepared(&mut self, prepared: PreparedFrame) -> FrameDisposition {
        let (_, mut integration) = self.split_for_pipeline();
        integration.integrate(prepared)
    }
    /// Copy the current estimate and coverage masks into an owned snapshot.
    /// How much of each pixel's integrated observations came from clipped
    /// raw pixels, from 0 to 1, as a mono image on the stack's grid: where
    /// the sensor saturated, a stack records only that the sky was at least
    /// that bright, and its colour there means nothing. `None` when no
    /// integrated frame clipped, or its saturation level was unknown.
    /// [`crate::neutralize_clipped`] renders those pixels neutral after
    /// colour calibration.
    pub fn clipped_fraction(&self) -> Option<LinearImage> {
        let channels = self.reference.channels;
        let fraction = self.accumulator.clipped_fraction(channels)?;
        LinearImage::new(self.reference.width, self.reference.height, 1, fraction).ok()
    }

    pub fn snapshot(&self) -> Result<StackSnapshot> {
        let (mut mean, variance) = self.accumulator.snapshot();
        self.accumulator.check_kept()?;
        let moments = self.accumulator.moments();
        self.fill_photosite_gaps(&mut mean, moments.count);
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
            coverage: moments.count.to_vec(),
            rejected_samples: moments.rejected.to_vec(),
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
        self.accumulator.check_kept()?;
        self.fill_photosite_gaps(&mut mean, self.accumulator.moments().count);
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
        let moments = self.accumulator.moments();
        StackView {
            width: self.reference.width,
            height: self.reference.height,
            channels: self.reference.channels,
            mean: moments.mean,
            coverage: moments.count,
            rejected_samples: moments.rejected,
            accepted_frames: self.accepted_frames,
            rejected_frames: self.rejected_frames,
        }
    }

    /// Return the identity processing mapping for the prepared reference
    /// frame. Callers can persist this beside mappings returned for later
    /// frames and use one extraction path for the whole stack.
    pub fn reference_mapping(&self) -> RegisteredFrameMapping {
        let mut mapping = RegisteredFrameMapping::identity(&self.reference);
        mapping.set_bin_factor(self.options.bin_factor);
        if self.reference_is_integrated {
            mapping.set_channel_sites(self.ledger.reference_channel_sites());
        }
        mapping
    }

    /// Consume the live state and move its full-frame buffers into a final
    /// snapshot. Batch callers should prefer this to avoid snapshot copies.
    pub fn into_snapshot(self) -> Result<StackSnapshot> {
        self.accumulator.check_kept()?;
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

    /// Every path an output must not replace: [`Self::input_paths`] and the
    /// reference image of a stack from
    /// [`Self::open_fits_with_reference_image`].
    pub fn protected_paths(&self) -> Vec<PathBuf> {
        self.input_paths
            .iter()
            .chain(&self.reference_image_path)
            .cloned()
            .collect()
    }

    /// The reference image a stack from
    /// [`Self::open_fits_with_reference_image`] aligns to.
    pub fn reference_image_path(&self) -> Option<&Path> {
        self.reference_image_path.as_deref()
    }

    /// Whether the reference frame is part of the stack. It is, except in a
    /// stack aligned to a separate reference image
    /// ([`Self::with_reference_image`]), which starts with no frames.
    pub fn reference_is_integrated(&self) -> bool {
        self.reference_is_integrated
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

    /// The tolerances lights are judged under now; see
    /// [`StackOptions::match_tolerances`].
    pub fn match_tolerances(&self) -> seiza_calibration::MatchTolerances {
        self.options.match_tolerances
    }

    /// Judge the lights pushed from now on under `tolerances`.
    ///
    /// Frames already admitted keep the tolerances they were admitted under,
    /// and a reintegration judges each by its own, so a light given a flat
    /// from another rotator angle on purpose can be admitted under a wider
    /// gate and the gate narrowed again after it. The stack's configuration
    /// fingerprint changes with the tolerances, and a saved context keeps
    /// them. Refused for tolerances that are negative or not finite, and for
    /// a stack started from prepared pixels, whose frames are never
    /// calibrated.
    pub fn set_match_tolerances(
        &mut self,
        tolerances: seiza_calibration::MatchTolerances,
    ) -> Result<()> {
        self.require_fits_input_mode()?;
        crate::calibration::check_tolerances(&tolerances)?;
        let options = StackOptions {
            match_tolerances: tolerances,
            ..self.options.clone()
        };
        self.configuration_fingerprint =
            stack_configuration_fingerprint(&options, &self.calibration, self.input_mode)?;
        self.options = options;
        Ok(())
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

/// The identities of `paths`, refusing any two that name the same file.
fn distinct_input_paths<'a>(
    paths: impl IntoIterator<Item = Option<&'a Path>>,
) -> Result<Vec<PathBuf>> {
    let mut identities: Vec<PathBuf> = Vec::new();
    for path in paths.into_iter().flatten().map(path_identity) {
        if identities
            .iter()
            .any(|other| paths_refer_to_same_file(other, &path))
        {
            return Err(Error::Stack(format!(
                "stack input path {} is used more than once",
                path.display()
            )));
        }
        identities.push(path);
    }
    Ok(identities)
}

/// The step at which preparing a light failed, which decides how a live
/// stack turns the frame away. The reference, planning and replay paths pass
/// the error on as it is.
pub(crate) enum LightFailure {
    /// The masters refused the light, or applying them failed.
    Calibration(Error),
    /// Repairing its defects or filtering its impulses failed.
    Repair(Error),
    /// Debayering or binning failed.
    Image(Error),
}

impl LightFailure {
    /// The reason a live stack reports for turning the light away.
    pub(crate) fn rejection(self) -> FrameRejectionReason {
        match self {
            Self::Calibration(Error::Calibration(message)) => {
                FrameRejectionReason::Calibration(message)
            }
            Self::Calibration(error) | Self::Repair(error) => {
                FrameRejectionReason::Calibration(error.to_string())
            }
            Self::Image(error) => FrameRejectionReason::IncompatibleImage(error.to_string()),
        }
    }

    /// The error itself, for a path that passes it on. Not a `From`
    /// conversion: [`Error`] converting only from itself is what lets a
    /// closure's `?` infer its error type.
    pub(crate) fn into_error(self) -> Error {
        match self {
            Self::Calibration(error) | Self::Repair(error) | Self::Image(error) => error,
        }
    }
}

/// Check a light against its masters under `tolerances`, calibrate it, then
/// repair its defects and filter its impulses: every step a light takes on
/// its sensor's own photosites, before debayering.
pub(crate) fn calibrate_light(
    frame: &mut FitsFrame,
    calibration: &CalibrationMasters,
    tolerances: &seiza_calibration::MatchTolerances,
    options: &StackOptions,
) -> std::result::Result<(), LightFailure> {
    calibration
        .validate_light_frame_with(frame, tolerances)
        .map_err(LightFailure::Calibration)?;
    calibration
        .apply(&mut frame.image, frame.exposure_seconds, frame.bayer)
        .map_err(LightFailure::Calibration)?;
    options.repair(frame).map_err(LightFailure::Repair)
}

/// Prepare a light read from a file, as every path that stacks, plans or
/// replays one does: [`calibrate_light`], then debayer, then bin by
/// [`StackOptions::bin_factor`].
/// A frame's raw clip mask, taken before calibration, carried through the
/// demosaic and binning [`develop`] gave the frame; `None` when it had none
/// or cannot follow.
pub(crate) fn developed_clip_mask(
    mask: Option<LinearImage>,
    cfa: Option<BayerLayout>,
    options: &StackOptions,
) -> Option<LinearImage> {
    crate::saturation::develop_clip_mask(mask?, cfa, options.demosaic, options.bin_factor).ok()
}

pub(crate) fn prepare_light(
    frame: FitsFrame,
    calibration: &CalibrationMasters,
    tolerances: &seiza_calibration::MatchTolerances,
    options: &StackOptions,
) -> std::result::Result<(FitsFrame, Option<BayerLayout>), LightFailure> {
    prepare_light_at(frame, calibration, tolerances, options, options.bin_factor)
}

/// [`prepare_light`], binned by `bin_factor` in place of the stack's own: 1
/// for the sensor's full resolution.
pub(crate) fn prepare_light_at(
    mut frame: FitsFrame,
    calibration: &CalibrationMasters,
    tolerances: &seiza_calibration::MatchTolerances,
    options: &StackOptions,
    bin_factor: u32,
) -> std::result::Result<(FitsFrame, Option<BayerLayout>), LightFailure> {
    calibrate_light(&mut frame, calibration, tolerances, options)?;
    develop(frame, options, bin_factor).map_err(LightFailure::Image)
}

/// Debayer a calibrated frame as `options` say, and bin it by `bin_factor`
/// in all: a superpixel demosaic is the first 2x2 of an even bin, and at an
/// odd one, such as the sensor's full resolution, VNG stands in for it.
/// Returns the frame and the Bayer layout it was debayered from.
pub(crate) fn develop(
    frame: FitsFrame,
    options: &StackOptions,
    bin_factor: u32,
) -> Result<(FitsFrame, Option<BayerLayout>)> {
    let superpixel = options.demosaic == crate::Demosaic::Superpixel;
    let demosaic = if superpixel && !bin_factor.is_multiple_of(2) {
        crate::Demosaic::Vng
    } else {
        options.demosaic
    };
    let (frame, cfa) = frame.into_prepared_with_layout(demosaic, &options.gpu)?;
    let binned = if cfa.is_some() && demosaic == crate::Demosaic::Superpixel {
        2
    } else {
        1
    };
    Ok((frame.binned(bin_factor / binned)?, cfa))
}

/// `image`, whose channels sample the sky at `sites`, resampled so every
/// channel samples its pixels' centres, as registration carries every frame
/// it resamples; `image` itself without sites. Resampled on the GPU where
/// `gpu` allows it.
pub(crate) fn co_site(
    image: LinearImage,
    sites: Option<crate::registration::ChannelSites>,
    interpolation: crate::Interpolation,
    gpu: &crate::GpuOptions,
) -> Result<LinearImage> {
    if sites.is_none() {
        return Ok(image);
    }
    crate::registration::resample_region_sited(
        &image,
        image.width,
        image.height,
        ReferenceRegion {
            x: 0,
            y: 0,
            width: image.width,
            height: image.height,
        },
        crate::registration::FrameGeometry::Similarity(SimilarityTransform::IDENTITY),
        interpolation.into(),
        sites,
        gpu,
    )
}

/// The median of each admitted frame's registration RMS: its `refined` one
/// where there is one, and its `live` one otherwise, where known.
fn median_registration_rms(live: &[Option<f64>], refined: &[Option<f64>]) -> Option<f64> {
    let mut values = live
        .iter()
        .enumerate()
        .filter_map(|(index, &live)| refined.get(index).copied().flatten().or(live))
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[values.len() / 2])
}

/// Measure noise before any registration resampling changes its variance.
pub(crate) fn measure_reference_noise(reference: &LinearImage) -> Result<Vec<f32>> {
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
    // Where the arithmetic runs is no part of the configuration.
    let options = serde_json::to_vec(&StackOptions {
        gpu: crate::GpuOptions::default(),
        ..options.clone()
    })
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
    /// The frame's samples over `region` of the reference grid; every sample
    /// beyond it is `NaN`. The whole grid except on an expanded canvas.
    registered: LinearImage,
    region: ReferenceRegion,
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
    /// The registered image before normalization, written for a later
    /// reintegration when the stack keeps frames for one.
    staged: Option<crate::replay::StagedFrame>,
    /// How much of each pixel of `region` the frame's clipped raw pixels
    /// reach, from 0 to 1, when any of them clipped.
    clipped: Option<crate::image::ClipMask>,
    /// The normalization `registered` still waits for, when it was applied
    /// to `on_gpu` instead: the CPU's copy needs it only if the frame is
    /// integrated there after all.
    pending_normalization: Option<NormalizationMap>,
    /// `registered`'s samples, normalized, where the GPU resampled them, for
    /// a live stack's accumulator there; `registered` then keeps its shape
    /// and not its samples.
    #[cfg(feature = "gpu")]
    on_gpu: Option<crate::gpu::GpuFrame>,
    /// `registered`'s finite samples.
    finite_samples: usize,
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

/// How [`prepare_frame`] finds a frame's transform: a planned registration
/// when one is known, else through an expanded canvas's panels, else
/// against the reference.
pub(crate) struct FrameRegistration<'a> {
    pub registrar: &'a Registrar,
    pub canvas: Option<&'a crate::expand::CanvasRegistrar>,
    pub known: Option<std::result::Result<&'a crate::RegistrationResult, &'a str>>,
    /// The reference's photometric ceilings once measured, for every frame
    /// prepared against the same reference.
    pub ceilings: &'a std::sync::OnceLock<Vec<f32>>,
}

impl FrameRegistration<'_> {
    fn register(&self, frame: &LinearImage) -> Result<crate::RegistrationResult> {
        match (self.known, self.canvas) {
            (Some(Ok(known)), _) => Ok(known.clone()),
            (Some(Err(reason)), _) => Err(Error::Registration(format!(
                "could not join the mosaic: {reason}"
            ))),
            (None, Some(canvas)) => canvas.register(frame),
            (None, None) => self.registrar.register(frame),
        }
    }

    /// The reference stars photometric normalization measures.
    fn reference_star_positions(&self) -> Vec<(f64, f64)> {
        match self.canvas {
            Some(canvas) => canvas.star_positions(),
            None => self.registrar.reference_star_positions(),
        }
    }

    /// The [`crate::normalization::photometric_ceilings`] of `reference`,
    /// which an expanded canvas measured once, as do [`Self::ceilings`]
    /// the first time they are asked for.
    fn reference_ceilings(&self, reference: &LinearImage) -> &[f32] {
        match self.canvas {
            Some(canvas) if !resamples_whole_canvas() => canvas.photometric_ceilings(),
            _ => self
                .ceilings
                .get_or_init(|| crate::normalization::photometric_ceilings(reference)),
        }
    }
}

/// Register and normalize one frame against the immutable reference.
///
/// Every gate applied here — channel count, registration quality, scale,
/// rotation, overlap, normalization gain — reads only the reference and the
/// options, so it reaches the same verdict whatever else is in flight. That is
/// what lets the pipeline prepare frames out of order and still match a
/// sequential run exactly.
///
/// With `cache`, the registered image is written there before it is
/// normalized, which is the image a reintegration would otherwise make again
/// from the source; see [`LiveStacker::retain_frames_for_reintegration`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_frame(
    reference: &LinearImage,
    target: FrameRegistration<'_>,
    options: &StackOptions,
    reference_noise: &[f32],
    frame: LinearImage,
    cfa: Option<BayerLayout>,
    cache: Option<&crate::replay::CacheFiles>,
    clipped: Option<LinearImage>,
) -> Result<PreparedFrame> {
    // The frame's samples the GPU holds, for detection and resampling, go
    // back to the pool when this returns.
    #[cfg(feature = "gpu")]
    let _resident = crate::gpu::ResidentScope;
    if reference.channels != frame.channels {
        return Ok(PreparedFrame::Rejected(
            FrameRejectionReason::IncompatibleImage(format!(
                "frame has {} channel(s) but stack has {}",
                frame.channels, reference.channels
            )),
        ));
    }
    let registration = match target.register(&frame) {
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
    // Registration measures in working pixels; gates and reports read the
    // sensor's own.
    let scale = options.scale();
    let rms_pixels = scale.native_length(registration.rms_error_pixels);
    if rms_pixels > criteria.maximum_registration_rms_pixels {
        return Ok(PreparedFrame::Rejected(
            FrameRejectionReason::RegistrationRms {
                measured: rms_pixels,
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
    // A frame reaches only the part of an expanded canvas its footprint
    // covers, and every sample beyond it would resample to `NaN`, so only
    // that part is resampled, normalized and integrated. A fixed grid is
    // resampled whole.
    let region = match target.canvas {
        Some(_) if !resamples_whole_canvas() => crate::registration::footprint_region(
            frame.width,
            frame.height,
            reference.width,
            reference.height,
            geometry,
            registration.transform,
        ),
        _ => ReferenceRegion {
            x: 0,
            y: 0,
            width: reference.width,
            height: reference.height,
        },
    };
    // The frame's clip mask, where its raw pixels clipped, goes onto the
    // grid as the frame does: bilinearly, so a pixel reads how much of it
    // they reach, in the same pass where the GPU resamples.
    let mask = clipped.filter(|mask| (mask.width, mask.height) == (frame.width, frame.height));
    let (mut registered, clipped) = crate::registration::resample_region_sited_masked(
        &frame,
        mask.as_ref(),
        reference.width,
        reference.height,
        region,
        geometry,
        options.interpolation.into(),
        options.channel_sites(cfa),
        &options.gpu,
    )?;
    // Where the GPU resampled the frame, its samples stay there for the
    // normalization and a GPU accumulator to read, when there is one.
    #[cfg(feature = "gpu")]
    let mut on_gpu = (options.gpu.accumulator && crate::gpu::accumulating())
        .then(|| {
            crate::gpu::take_resident(
                &registered.data,
                (registered.width, registered.height, registered.channels),
            )
        })
        .flatten();
    let finite_samples = registered
        .data
        .par_iter()
        .filter(|value| value.is_finite())
        .count();
    // On an expanded canvas, coverage is relative to the frame's own finite
    // footprint; a fixed grid measures coverage of the reference.
    let offered = if target.canvas.is_some() {
        (frame.data.iter().filter(|value| value.is_finite()).count() as f64
            * registration.transform.scale.powi(2)) as f32
    } else {
        registered.sample_count() as f32
    };
    let overlap_fraction = (finite_samples as f32 / offered.max(1.0)).min(1.0);
    if overlap_fraction < criteria.minimum_overlap_fraction {
        return Ok(PreparedFrame::Rejected(
            FrameRejectionReason::InsufficientOverlap {
                measured: overlap_fraction,
                minimum: criteria.minimum_overlap_fraction,
            },
        ));
    }
    let gains = matches!(
        options.normalization,
        NormalizationMode::LocalBackground { .. }
    )
    .then(|| {
        crate::normalization::photometric_gains(
            reference,
            target.reference_ceilings(reference),
            Placed::on(&registered, region, reference),
            &target.reference_star_positions(),
        )
    })
    .flatten();
    let placed = Placed::on(&registered, region, reference);
    // An expanded canvas keeps the first value each sample received, so its
    // background is a patchwork of single frames' skies, each with its own
    // gradient. Matched tile by tile, every frame would take on that
    // patchwork, and so would the stack, where the frames' own gradients
    // otherwise average away. There local background normalization keeps
    // its photometric gain and one offset over the frame's footprint; a
    // reintegration fits the tiles against a reference integrated from the
    // best frames.
    let mode = match (scale.normalization(options.normalization), target.canvas) {
        (NormalizationMode::LocalBackground { .. }, Some(_)) => {
            NormalizationMode::LocalBackground {
                tile_size: reference.width.max(reference.height),
            }
        }
        (mode, _) => mode,
    };
    let normalization =
        match NormalizationMap::estimate_with_gains(reference, placed, mode, gains.as_deref()) {
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
    // Bayer drizzle measures overlap and normalization on the interpolated
    // frame above, which samples every channel evenly, then integrates the
    // photosites themselves with the same normalization.
    let photosites = cfa.filter(|_| options.cfa_integration == CfaIntegration::BayerDrizzle);
    // Written before normalization changes it, and deleted again if a later
    // gate turns the frame away. Under Bayer drizzle the photosites are kept
    // instead, below, with the samples of this image a background fit reads.
    let (mut staged, background) = match (cache, photosites) {
        (Some(cache), None) => {
            let placed = Placed::on(&registered, region, reference);
            (
                cache.stage(
                    placed,
                    cache.background_samples(placed),
                    crate::replay::keeps_zeros(options, cfa),
                ),
                None,
            )
        }
        (Some(cache), Some(_)) => (
            None,
            cache.background_samples(Placed::on(&registered, region, reference)),
        ),
        (None, _) => (None, None),
    };
    // Normalized on the GPU when the frame lies there for its accumulator,
    // and the CPU's copy only if the frame comes back to the CPU.
    #[cfg(feature = "gpu")]
    if photosites.is_some() {
        on_gpu = None;
    }
    let mut pending_normalization = None;
    if !matches!(options.normalization, NormalizationMode::None) {
        #[cfg(feature = "gpu")]
        let normalized_there = on_gpu.as_mut().is_some_and(|frame| {
            normalization
                .gpu_coefficients(
                    (region.x, region.y),
                    (registered.width, registered.height, registered.channels),
                )
                .is_some_and(|coefficients| frame.normalize(&coefficients).is_ok())
        });
        #[cfg(not(feature = "gpu"))]
        let normalized_there = false;
        if normalized_there {
            pending_normalization = Some(normalization.clone());
        } else {
            #[cfg(feature = "gpu")]
            {
                on_gpu = None;
            }
            if let Err(error) = normalization.apply_region(&mut registered, region.x, region.y) {
                let message = match error {
                    Error::Normalization(message) => message,
                    other => other.to_string(),
                };
                return Ok(PreparedFrame::Rejected(
                    FrameRejectionReason::Normalization(message),
                ));
            }
        }
    }
    if let Some(layout) = photosites {
        registered = crate::registration::resample_region_geometry(
            &frame,
            reference.width,
            reference.height,
            region,
            geometry,
            crate::registration::Sampling::NearestPhotosite(layout),
            &options.gpu,
        )?;
        staged = cache.and_then(|cache| {
            cache.stage(
                Placed::on(&registered, region, reference),
                background,
                crate::replay::keeps_zeros(options, cfa),
            )
        });
        if !matches!(options.normalization, NormalizationMode::None) {
            normalization.apply_region(&mut registered, region.x, region.y)?;
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
    mapping.set_bin_factor(options.bin_factor);
    mapping.set_channel_sites(options.channel_sites(cfa));
    // A frame whose normalized samples lie on the GPU for its accumulator
    // keeps no second copy here, which a frame queued for integration would
    // hold for as long as it waits.
    #[cfg(feature = "gpu")]
    if on_gpu.is_some() {
        registered.data = Vec::new();
    }
    Ok(PreparedFrame::Ready(Box::new(ReadyFrame {
        registered,
        finite_samples,
        region,
        transform: registration.transform,
        matched_stars: registration.matched_stars,
        registration_rms_pixels: rms_pixels,
        registration_drift_pixels: scale.native_length(registration.drift_pixels),
        normalization_mean_gain,
        normalization_mean_offset,
        mapping: Box::new(mapping),
        overlap_fraction,
        source: None,
        noise,
        weight,
        photosites: photosites.is_some(),
        staged,
        clipped,
        pending_normalization,
        #[cfg(feature = "gpu")]
        on_gpu,
    })))
}

#[cfg(test)]
thread_local! {
    /// Set by a test to prepare frames on an expanded canvas as before
    /// their footprints bounded them, to compare the two: each resampled
    /// over the whole canvas, which photometry reads for its brightest
    /// sample.
    static WHOLE_CANVAS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether [`prepare_frame`] prepares a frame over the whole of an expanded
/// canvas: only for a test comparing that with its footprint.
#[cfg(test)]
pub(crate) fn resamples_whole_canvas() -> bool {
    WHOLE_CANVAS.get()
}

#[cfg(not(test))]
pub(crate) fn resamples_whole_canvas() -> bool {
    false
}

/// The immutable half of a stack: everything preparing a frame may read.
pub(crate) struct PreparationHalf<'a> {
    pub(crate) reference: &'a LinearImage,
    pub(crate) registrar: &'a Registrar,
    pub(crate) planned_registrations:
        &'a std::collections::HashMap<PathBuf, crate::expand::PlannedRegistration>,
    pub(crate) canvas_registrar: Option<&'a crate::expand::CanvasRegistrar>,
    pub(crate) calibration: &'a CalibrationMasters,
    pub(crate) options: &'a StackOptions,
    pub(crate) reference_noise: &'a [f32],
    pub(crate) reference_ceilings: &'a std::sync::OnceLock<Vec<f32>>,
    /// Where a stack kept for reintegration writes each frame's registered
    /// image.
    pub(crate) frame_cache: Option<&'a crate::replay::CacheFiles>,
}

/// The mutable half of a stack: the accumulator and the run's tallies.
pub(crate) struct IntegrationHalf<'a> {
    accumulator: &'a mut Accumulator,
    options: &'a StackOptions,
    accepted_frames: &'a mut u32,
    rejected_frames: &'a mut u32,
    input_paths: &'a mut Vec<PathBuf>,
    ledger: &'a mut crate::replay::Ledger,
    frame_cache: Option<crate::replay::CacheAdmissions<'a>>,
    /// Integrated fractions count the frame's finite footprint.
    expanded_canvas: bool,
    /// The reference grid's width and height, which a frame's region lies
    /// on.
    reference_width: usize,
    reference_height: usize,
}

/// What [`Accumulator::admit_staged`] did with a frame on the GPU.
#[cfg(feature = "gpu")]
enum Staged {
    /// Admitted there or not: what classification accepted, and the
    /// integration's counts when it was integrated.
    Admitted((usize, Option<(usize, usize)>)),
    /// The moments are not on the GPU, so the frame is handed back.
    Elsewhere(crate::gpu::GpuFrame),
    /// The GPU failed; the moments are back on the CPU.
    Failed,
}

/// The fewest accepted samples whose `fraction` is not below `minimum`, as
/// the frame's admission compares them, so a frame is admitted exactly when
/// it accepts at least as many; past `u32::MAX`, more than a frame holds,
/// when no count up to it is.
fn admission_count(fraction: &impl Fn(usize) -> f32, minimum: f32) -> usize {
    // `fraction` grows with the count, so the counts it falls short at come
    // first.
    let (mut low, mut high) = (0, u32::MAX as usize + 1);
    while low < high {
        let middle = low + (high - low) / 2;
        if fraction(middle) < minimum {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low
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
            region,
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
            staged,
            clipped,
            pending_normalization,
            #[cfg(feature = "gpu")]
            on_gpu,
            finite_samples,
        } = *ready;
        let mut registered = registered;
        #[cfg_attr(not(feature = "gpu"), allow(unused_mut))]
        let mut pending_normalization = pending_normalization;
        let weights = (!weight.is_empty()).then_some(weight.as_slice());

        // A photosite frame offers one sample per pixel, not one per channel.
        let offered = if self.expanded_canvas {
            finite_samples
        } else if photosites {
            registered.pixel_count()
        } else {
            registered.pixel_count() * registered.channels
        };
        let minimum = self.options.acceptance.minimum_integrated_fraction;
        let fraction = |accepted: usize| accepted as f32 / offered.max(1) as f32;
        let admit_at = admission_count(&fraction, minimum);
        #[cfg(feature = "gpu")]
        self.accumulator.offer_gpu(&self.options.gpu);
        // The frame's normalized samples on the GPU, for the accumulator
        // there; else the CPU's copy, normalized now if it waited.
        #[cfg(feature = "gpu")]
        let admitted = match on_gpu {
            None => None,
            Some(staged) => match self.accumulator.admit_staged(
                Placed {
                    image: &registered,
                    region,
                    width: self.reference_width,
                    height: self.reference_height,
                },
                staged,
                self.options.rejection,
                weights,
                admit_at,
            ) {
                Staged::Admitted(admitted) => Some(admitted),
                // The moments are on the CPU: the frame comes back, already
                // normalized.
                Staged::Elsewhere(staged) => {
                    match staged.download() {
                        Some(samples) if registered.data.is_empty() => {
                            registered.data = samples;
                            pending_normalization = None;
                        }
                        _ if registered.data.is_empty() => {
                            return self.reject(FrameRejectionReason::Normalization(
                                "the GPU could not give back the frame it normalized".into(),
                            ));
                        }
                        _ => {}
                    }
                    None
                }
                Staged::Failed if registered.data.is_empty() => {
                    return self.reject(FrameRejectionReason::Normalization(
                        "the GPU failed holding the frame it normalized".into(),
                    ));
                }
                Staged::Failed => None,
            },
        };
        #[cfg(not(feature = "gpu"))]
        let admitted = None;
        let (would_accept, integrated) = match admitted {
            Some(admitted) => admitted,
            None => {
                if let Some(normalization) = &pending_normalization
                    && let Err(error) =
                        normalization.apply_region(&mut registered, region.x, region.y)
                {
                    return self.reject(FrameRejectionReason::Normalization(error.to_string()));
                }
                self.accumulator.admit_frame(
                    Placed {
                        image: &registered,
                        region,
                        width: self.reference_width,
                        height: self.reference_height,
                    },
                    self.options.rejection,
                    weights,
                    admit_at,
                )
            }
        };
        let integrated_fraction = fraction(would_accept);
        let Some((accepted_samples, rejected_samples)) = integrated else {
            return self.reject(FrameRejectionReason::InsufficientIntegratedSamples {
                measured: integrated_fraction,
                minimum,
            });
        };
        if let Some(clipped) = &clipped {
            self.accumulator.add_clipped_mask(
                clipped,
                region,
                self.reference_width,
                registered.channels,
            );
        }
        *self.accepted_frames += 1;
        self.ledger.admit(
            source,
            (*mapping).clone(),
            crate::replay::FrameWeightRecord {
                noise: noise.clone(),
                weight: weight.clone(),
            },
            self.options.match_tolerances,
        );
        self.ledger.record_registration_rms(registration_rms_pixels);
        if let Some(cache) = &mut self.frame_cache {
            cache.admit(staged);
        }
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
    /// Per-pixel sum of how much of each integrated frame's pixel its
    /// clipped raw pixels reached; `None` until a frame with any arrives.
    clipped: Option<Vec<f32>>,
    /// With the `gpu` feature, the moments on the GPU once a frame large
    /// enough for it arrives; the vectors above are then empty, and
    /// [`Self::moments`] reads the GPU's back.
    #[cfg(feature = "gpu")]
    gpu: Option<crate::gpu::GpuMoments>,
    /// The GPU's moments read back, once a frame, when first asked for.
    #[cfg(feature = "gpu")]
    mirror: std::sync::OnceLock<crate::gpu::MomentVectors>,
    /// Whether the GPU was offered these moments, so a stack it cannot
    /// take does not ask every frame.
    #[cfg(feature = "gpu")]
    offered: bool,
    /// The GPU lost the moments and could not give them back: what is
    /// left is not this stack.
    lost: std::sync::atomic::AtomicBool,
}

impl Accumulator {
    fn new(samples: usize, weighted: bool) -> Self {
        Self::from_moments(
            vec![0.0; samples],
            vec![0.0; samples],
            vec![0; samples],
            vec![0; samples],
            weighted.then(|| vec![0.0; samples]),
            None,
        )
    }

    /// An accumulator holding these moments, as a saved context restores.
    fn from_moments(
        mean: Vec<f32>,
        m2: Vec<f32>,
        count: Vec<u32>,
        rejected: Vec<u32>,
        weight_sum: Option<Vec<f32>>,
        clipped: Option<Vec<f32>>,
    ) -> Self {
        Self {
            mean,
            m2,
            count,
            rejected,
            weight_sum,
            clipped,
            #[cfg(feature = "gpu")]
            gpu: None,
            #[cfg(feature = "gpu")]
            mirror: std::sync::OnceLock::new(),
            #[cfg(feature = "gpu")]
            offered: false,
            lost: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// The moments, wherever they are: the GPU's are read back once a
    /// frame, the first time they are asked for.
    fn moments(&self) -> MomentsRef<'_> {
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &self.gpu {
            let mirror = self.mirror.get_or_init(|| match gpu.read() {
                Ok(moments) => moments,
                Err(reason) => {
                    eprintln!("seiza-stacking: the GPU lost the live stack: {reason}");
                    self.lost.store(true, std::sync::atomic::Ordering::Relaxed);
                    let samples = gpu.samples();
                    crate::gpu::MomentVectors {
                        mean: vec![f32::NAN; samples],
                        m2: vec![0.0; samples],
                        count: vec![0; samples],
                        rejected: vec![0; samples],
                        weight_sum: gpu.weighted().then(|| vec![0.0; samples]),
                    }
                }
            });
            return MomentsRef {
                mean: &mirror.mean,
                m2: &mirror.m2,
                count: &mirror.count,
                rejected: &mirror.rejected,
                weight_sum: mirror.weight_sum.as_deref(),
            };
        }
        MomentsRef {
            mean: &self.mean,
            m2: &self.m2,
            count: &self.count,
            rejected: &self.rejected,
            weight_sum: self.weight_sum.as_deref(),
        }
    }

    /// Samples the moments hold, wherever they are.
    fn samples(&self) -> usize {
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &self.gpu {
            return gpu.samples();
        }
        self.mean.len()
    }

    /// An error when the GPU lost the moments.
    fn check_kept(&self) -> Result<()> {
        if self.lost.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(Error::Stack(
                "the GPU lost the live stack's state; reintegrate or restack its frames".into(),
            ));
        }
        Ok(())
    }

    /// Move the moments to the GPU the first time a frame arrives, when
    /// they are large enough for it and `options` allow it.
    #[cfg(feature = "gpu")]
    fn offer_gpu(&mut self, options: &crate::GpuOptions) {
        if self.offered {
            return;
        }
        self.offered = true;
        let moments = crate::gpu::MomentSlices {
            mean: &self.mean,
            m2: &self.m2,
            count: &self.count,
            rejected: &self.rejected,
            weight_sum: self.weight_sum.as_deref(),
        };
        if let Some(gpu) = crate::gpu::GpuMoments::upload(moments, options) {
            self.gpu = Some(gpu);
            // The GPU holds them now; free the CPU's.
            self.mean = Vec::new();
            self.m2 = Vec::new();
            self.count = Vec::new();
            self.rejected = Vec::new();
            self.weight_sum = self.weight_sum.as_ref().map(|_| Vec::new());
        }
    }

    /// Bring the moments back from the GPU after it failed, so the CPU
    /// carries on; when it cannot give them back, the stack is lost.
    #[cfg(feature = "gpu")]
    fn leave_gpu(&mut self, reason: &str) {
        let Some(gpu) = self.gpu.take() else {
            return;
        };
        let moments = match self.mirror.take() {
            Some(mirror) => Ok(mirror),
            None => gpu.read(),
        };
        match moments {
            Ok(moments) => {
                eprintln!("seiza-stacking: integrating on the CPU from here on: {reason}");
                self.mean = moments.mean;
                self.m2 = moments.m2;
                self.count = moments.count;
                self.rejected = moments.rejected;
                self.weight_sum = moments.weight_sum;
            }
            Err(read) => {
                eprintln!("seiza-stacking: the GPU lost the live stack: {reason}; {read}");
                self.lost.store(true, std::sync::atomic::Ordering::Relaxed);
                let samples = gpu.samples();
                self.mean = vec![0.0; samples];
                self.m2 = vec![0.0; samples];
                self.count = vec![0; samples];
                self.rejected = vec![0; samples];
                self.weight_sum = gpu.weighted().then(|| vec![0.0; samples]);
            }
        }
    }

    /// [`Self::admit_frame`] for a frame whose samples, normalized, already
    /// lie on the GPU as `staged`, when the moments are there too.
    #[cfg(feature = "gpu")]
    fn admit_staged(
        &mut self,
        frame: Placed<'_>,
        staged: crate::gpu::GpuFrame,
        rejection: RejectionMode,
        weights: Option<&[f32]>,
        admit_at: usize,
    ) -> Staged {
        let Some(gpu) = self.gpu.as_mut() else {
            return Staged::Elsewhere(staged);
        };
        let unit = [1.0_f32];
        match gpu.admit(
            &frame.image.data,
            Some(staged),
            gpu_placement(frame),
            weights.unwrap_or(&unit),
            gpu_rejection(rejection),
            admit_at,
        ) {
            Ok((accepted, integrated)) => {
                if integrated.is_some() {
                    // The moments changed; a copy read back before is stale.
                    self.mirror = std::sync::OnceLock::new();
                }
                Staged::Admitted((accepted, integrated))
            }
            Err(reason) => {
                self.leave_gpu(&reason);
                Staged::Failed
            }
        }
    }

    /// Classify `frame` against the moments and, if rejection would accept
    /// at least `admit_at` of its samples, integrate it: how many it would
    /// accept and refuse, and the integration's accepted and rejected counts
    /// when it was integrated. On the GPU, when the moments are there, in
    /// one submission; else [`Self::classify_placed`] and
    /// [`Self::integrate_placed`].
    fn admit_frame(
        &mut self,
        frame: Placed<'_>,
        rejection: RejectionMode,
        weights: Option<&[f32]>,
        admit_at: usize,
    ) -> (usize, Option<(usize, usize)>) {
        #[cfg(feature = "gpu")]
        {
            if let Some(gpu) = &mut self.gpu {
                let unit = [1.0_f32];
                match gpu.admit(
                    &frame.image.data,
                    None,
                    gpu_placement(frame),
                    weights.unwrap_or(&unit),
                    gpu_rejection(rejection),
                    admit_at,
                ) {
                    Ok((accepted, integrated)) => {
                        if integrated.is_some() {
                            // The moments changed; a copy read back before
                            // is stale.
                            self.mirror = std::sync::OnceLock::new();
                        }
                        return (accepted, integrated);
                    }
                    Err(reason) => self.leave_gpu(&reason),
                }
            }
        }
        let (accepted, _) = self.classify_placed(frame, rejection, weights);
        let integrated =
            (accepted >= admit_at).then(|| self.integrate_placed(frame, rejection, weights));
        (accepted, integrated)
    }

    fn add_clipped_mask(
        &mut self,
        mask: &crate::image::ClipMask,
        region: ReferenceRegion,
        grid_width: usize,
        channels: usize,
    ) {
        match mask {
            crate::image::ClipMask::Dense(mask) => {
                self.add_clipped(mask, region, grid_width, channels)
            }
            #[cfg(feature = "gpu")]
            crate::image::ClipMask::Sparse {
                width,
                height,
                entries,
            } => {
                if (*width, *height) != (region.width, region.height) || entries.is_empty() {
                    return;
                }
                let pixels = self.samples() / channels.max(1);
                let clipped = self.clipped.get_or_insert_with(|| vec![0.0; pixels]);
                // Each entry names a different pixel. Frames still arrive
                // in admission order, so these are the dense path's exact
                // additions, with its zero and non-finite work omitted.
                for &(index, value) in entries {
                    if index < width * height && value.is_finite() && value > 0.0 {
                        let at = (region.y + index / width) * grid_width + region.x + index % width;
                        clipped[at] += value.min(1.0);
                    }
                }
            }
        }
    }

    /// Add a frame's clipped share of each pixel of `region`, a mono image
    /// as wide as the region, on a grid `grid_width` pixels wide whose
    /// samples hold `channels` channels each.
    fn add_clipped(
        &mut self,
        mask: &LinearImage,
        region: ReferenceRegion,
        grid_width: usize,
        channels: usize,
    ) {
        if mask.channels != 1
            || (mask.width, mask.height) != (region.width, region.height)
            || !mask.data.par_iter().any(|&value| value > 0.0)
        {
            return;
        }
        let pixels = self.samples() / channels.max(1);
        let clipped = self.clipped.get_or_insert_with(|| vec![0.0; pixels]);
        clipped[region.y * grid_width..(region.y + region.height) * grid_width]
            .par_chunks_mut(grid_width)
            .zip(mask.data.par_chunks(region.width))
            .for_each(|(row, values)| {
                for (sum, &value) in row[region.x..region.x + region.width]
                    .iter_mut()
                    .zip(values)
                {
                    if value.is_finite() && value > 0.0 {
                        *sum += value.min(1.0);
                    }
                }
            });
    }

    /// Each pixel's clipped share of the observations it has had, from 0 to
    /// 1, counting its first channel's accepted and rejected samples; `None`
    /// when no integrated frame clipped.
    fn clipped_fraction(&self, channels: usize) -> Option<Vec<f32>> {
        let clipped = self.clipped.as_ref()?;
        let channels = channels.max(1);
        let moments = self.moments();
        Some(
            clipped
                .par_iter()
                .enumerate()
                .map(|(pixel, &sum)| {
                    let observations =
                        moments.count[pixel * channels] + moments.rejected[pixel * channels];
                    if observations == 0 {
                        0.0
                    } else {
                        (sum / observations as f32).min(1.0)
                    }
                })
                .collect(),
        )
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
                    integrate_run(means, m2s, counts, rejecteds, samples, rejection)
                })
                .reduce(|| (0, 0), add_counts);
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
                    integrate_weighted(
                        (mean, m2, count, rejected, weight_sum),
                        sample,
                        weights[index % channels],
                        rejection,
                    )
                },
            )
            .reduce(|| (0, 0), add_counts)
    }

    /// [`Self::integrate`] for a frame placed on the reference grid, visiting
    /// only its region's rows, each as wide as the region. Every sample
    /// beyond the region is `NaN`, which integration skips, so this is
    /// [`Self::integrate`] of the grid-sized frame.
    fn integrate_placed(
        &mut self,
        frame: Placed<'_>,
        rejection: RejectionMode,
        weights: Option<&[f32]>,
    ) -> (usize, usize) {
        if frame.is_whole() {
            return self.integrate(&frame.image.data, rejection, weights);
        }
        let RegionRows {
            rows,
            span,
            grid_row,
        } = RegionRows::of(frame);
        let samples = frame.image.data.par_chunks(span.len());
        let Some(weight_sum) = self.weight_sum.as_mut() else {
            debug_assert!(weights.is_none(), "equal-weight stack given weights");
            return self.mean[rows.clone()]
                .par_chunks_mut(grid_row)
                .zip(self.m2[rows.clone()].par_chunks_mut(grid_row))
                .zip(self.count[rows.clone()].par_chunks_mut(grid_row))
                .zip(self.rejected[rows].par_chunks_mut(grid_row))
                .zip(samples)
                .map(|((((means, m2s), counts), rejecteds), samples)| {
                    integrate_run(
                        &mut means[span.clone()],
                        &mut m2s[span.clone()],
                        &mut counts[span.clone()],
                        &mut rejecteds[span.clone()],
                        samples,
                        rejection,
                    )
                })
                .reduce(|| (0, 0), add_counts);
        };
        let unit = [1.0_f32];
        let weights = weights.unwrap_or(&unit);
        let channels = weights.len();
        // A row of the region starts at a pixel, so a sample's channel is
        // its place in the row's, as it is in the grid's.
        self.mean[rows.clone()]
            .par_chunks_mut(grid_row)
            .zip(self.m2[rows.clone()].par_chunks_mut(grid_row))
            .zip(self.count[rows.clone()].par_chunks_mut(grid_row))
            .zip(self.rejected[rows.clone()].par_chunks_mut(grid_row))
            .zip(weight_sum[rows].par_chunks_mut(grid_row))
            .zip(samples)
            .map(
                |(((((means, m2s), counts), rejecteds), weight_sums), samples)| {
                    means[span.clone()]
                    .iter_mut()
                    .zip(&mut m2s[span.clone()])
                    .zip(&mut counts[span.clone()])
                    .zip(&mut rejecteds[span.clone()])
                    .zip(&mut weight_sums[span.clone()])
                    .zip(samples)
                    .enumerate()
                    .fold(
                        (0, 0),
                        |total, (index, (((((mean, m2), count), rejected), weight_sum), &sample))| {
                            add_counts(
                                total,
                                integrate_weighted(
                                    (mean, m2, count, rejected, weight_sum),
                                    sample,
                                    weights[index % channels],
                                    rejection,
                                ),
                            )
                        },
                    )
                },
            )
            .reduce(|| (0, 0), add_counts)
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
                    classify_run(means, m2s, counts, samples, rejection)
                })
                .reduce(|| (0, 0), add_counts);
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
            .map(|(index, (((&mean, &m2), &count), &sample))| {
                classify_weighted(
                    (mean, m2, count),
                    sample,
                    weights[index % channels],
                    rejection,
                )
            })
            .reduce(|| (0, 0), add_counts)
    }

    /// [`Self::classify`] for a frame placed on the reference grid, as
    /// [`Self::integrate_placed`] visits it.
    fn classify_placed(
        &self,
        frame: Placed<'_>,
        rejection: RejectionMode,
        weights: Option<&[f32]>,
    ) -> (usize, usize) {
        if frame.is_whole() {
            return self.classify(&frame.image.data, rejection, weights);
        }
        let RegionRows {
            rows,
            span,
            grid_row,
        } = RegionRows::of(frame);
        let moments = self.mean[rows.clone()]
            .par_chunks(grid_row)
            .zip(self.m2[rows.clone()].par_chunks(grid_row))
            .zip(self.count[rows].par_chunks(grid_row))
            .zip(frame.image.data.par_chunks(span.len()));
        if self.weight_sum.is_none() {
            debug_assert!(weights.is_none(), "equal-weight stack given weights");
            return moments
                .map(|(((means, m2s), counts), samples)| {
                    classify_run(
                        &means[span.clone()],
                        &m2s[span.clone()],
                        &counts[span.clone()],
                        samples,
                        rejection,
                    )
                })
                .reduce(|| (0, 0), add_counts);
        }
        let unit = [1.0_f32];
        let weights = weights.unwrap_or(&unit);
        let channels = weights.len();
        moments
            .map(|(((means, m2s), counts), samples)| {
                means[span.clone()]
                    .iter()
                    .zip(&m2s[span.clone()])
                    .zip(&counts[span.clone()])
                    .zip(samples)
                    .enumerate()
                    .fold(
                        (0, 0),
                        |total, (index, (((&mean, &m2), &count), &sample))| {
                            add_counts(
                                total,
                                classify_weighted(
                                    (mean, m2, count),
                                    sample,
                                    weights[index % channels],
                                    rejection,
                                ),
                            )
                        },
                    )
            })
            .reduce(|| (0, 0), add_counts)
    }

    fn snapshot(&self) -> (Vec<f32>, Vec<f32>) {
        let mean = self.mean_snapshot();
        let moments = self.moments();
        let variance = moments
            .m2
            .iter()
            .zip(moments.count)
            .map(|(&m2, &count)| finalized_variance(m2, count))
            .collect();
        (mean, variance)
    }

    fn mean_snapshot(&self) -> Vec<f32> {
        let moments = self.moments();
        moments
            .mean
            .par_iter()
            .zip(moments.count.par_iter())
            .map(|(&mean, &count)| finalized_mean(mean, count))
            .collect()
    }

    fn into_snapshot(mut self) -> (Vec<f32>, Vec<f32>, Vec<u32>, Vec<u32>) {
        #[cfg(feature = "gpu")]
        if self.gpu.is_some() {
            // The moments come home, from the copy read back if there is one.
            let moments = match self.mirror.take() {
                Some(mirror) => mirror,
                None => {
                    let _ = self.moments();
                    self.mirror.take().unwrap_or_default()
                }
            };
            self.mean = moments.mean;
            self.m2 = moments.m2;
            self.count = moments.count;
            self.rejected = moments.rejected;
            self.gpu = None;
        }
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

/// A live stack's moments, borrowed from wherever they are.
struct MomentsRef<'a> {
    mean: &'a [f32],
    m2: &'a [f32],
    count: &'a [u32],
    rejected: &'a [u32],
    #[allow(dead_code)]
    weight_sum: Option<&'a [f32]>,
}

/// Where `frame`'s samples land on the grid, for the GPU.
#[cfg(feature = "gpu")]
fn gpu_placement(frame: Placed<'_>) -> crate::gpu::FramePlacement {
    let channels = frame.channels();
    crate::gpu::FramePlacement {
        first_row: frame.region.y,
        rows: frame.region.height,
        first_sample: frame.region.x * channels,
        span: frame.region.width * channels,
        grid_row: frame.width * channels,
    }
}

/// `rejection` for the GPU.
#[cfg(feature = "gpu")]
fn gpu_rejection(rejection: RejectionMode) -> Option<crate::gpu::GpuRejection> {
    match rejection {
        RejectionMode::None => None,
        RejectionMode::DeltaSigma(options) => Some(crate::gpu::GpuRejection {
            low_sigma: options.low_sigma,
            high_sigma: options.high_sigma,
            warmup: options.warmup_samples,
            minimum_sigma: options.minimum_sigma,
        }),
    }
}

/// Samples per parallel task in the accumulator's per-sample passes.
const ACCUMULATE_CHUNK: usize = 16 * 1024;

/// The accumulator samples a placed frame's region covers: `span` of each
/// whole row of the grid in `rows`, rows `grid_row` samples long.
struct RegionRows {
    rows: std::ops::Range<usize>,
    span: std::ops::Range<usize>,
    grid_row: usize,
}

impl RegionRows {
    fn of(frame: Placed<'_>) -> Self {
        let channels = frame.channels();
        let grid_row = frame.width * channels;
        let start = frame.region.x * channels;
        Self {
            rows: frame.region.y * grid_row..(frame.region.y + frame.region.height) * grid_row,
            span: start..start + frame.region.width * channels,
            grid_row,
        }
    }
}

/// Sum two (accepted, refused) tallies.
fn add_counts(left: (usize, usize), right: (usize, usize)) -> (usize, usize) {
    (left.0 + right.0, left.1 + right.1)
}

/// Integrate `samples` into equal-weight moments one by one, by Welford's
/// update, skipping those not finite and counting those rejection refuses.
/// Returns how many were accepted and how many refused.
fn integrate_run(
    means: &mut [f32],
    m2s: &mut [f32],
    counts: &mut [u32],
    rejecteds: &mut [u32],
    samples: &[f32],
    rejection: RejectionMode,
) -> (usize, usize) {
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
}

/// Integrate one sample of a frame of weight `weight` into a weighted
/// accumulator's mean, m2, count, rejected count and weight sum, by West's
/// update, unless rejection refuses it. Returns `(1, 0)` when it is
/// accepted, `(0, 1)` when refused and `(0, 0)` when it is not finite.
#[inline(always)]
fn integrate_weighted(
    (mean, m2, count, rejected, weight_sum): (&mut f32, &mut f32, &mut u32, &mut u32, &mut f32),
    sample: f32,
    weight: f32,
    rejection: RejectionMode,
) -> (usize, usize) {
    if !sample.is_finite() {
        return (0, 0);
    }
    if should_reject_weighted_sample(*mean, *m2, *count, sample, weight, rejection) {
        *rejected = rejected.saturating_add(1);
        return (0, 1);
    }
    // West's weighted update. With weight 1 every operation matches the
    // equal-weight update bit for bit.
    let next_weight = *weight_sum + weight;
    let delta = sample - *mean;
    let weighted_delta = delta * weight;
    *mean += weighted_delta / next_weight;
    let delta_after = sample - *mean;
    *m2 += weighted_delta * delta_after;
    *weight_sum = next_weight;
    *count = count.saturating_add(1);
    (1, 0)
}

/// How many of `samples` [`integrate_run`] would accept and refuse.
fn classify_run(
    means: &[f32],
    m2s: &[f32],
    counts: &[u32],
    samples: &[f32],
    rejection: RejectionMode,
) -> (usize, usize) {
    let (mut accepted, mut refused) = (0, 0);
    for (((&mean, &m2), &count), &sample) in means.iter().zip(m2s).zip(counts).zip(samples) {
        if !sample.is_finite() {
        } else if should_reject_sample(mean, m2, count, sample, rejection) {
            refused += 1;
        } else {
            accepted += 1;
        }
    }
    (accepted, refused)
}

/// What [`integrate_weighted`] would return for one sample, given its
/// mean, m2 and count.
#[inline(always)]
fn classify_weighted(
    (mean, m2, count): (f32, f32, u32),
    sample: f32,
    weight: f32,
    rejection: RejectionMode,
) -> (usize, usize) {
    if !sample.is_finite() {
        (0, 0)
    } else if should_reject_weighted_sample(mean, m2, count, sample, weight, rejection) {
        (0, 1)
    } else {
        (1, 0)
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
pub(crate) fn rotation_deviation_degrees(rotation_radians: f64) -> f64 {
    let modulo_half_turn = rotation_radians.to_degrees().rem_euclid(180.0);
    modulo_half_turn.min(180.0 - modulo_half_turn)
}

#[cfg(test)]
#[path = "stack_canvas_tests.rs"]
mod canvas_tests;

#[cfg(test)]
#[path = "stack_binned_tests.rs"]
mod binned_tests;

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame is admitted at exactly the counts whose fraction the
    /// comparison would let through, at every minimum, odd ones included.
    #[test]
    fn the_admission_count_is_where_the_fraction_comparison_turns() {
        for offered in [0_usize, 1, 2, 3, 7, 1000, 1_048_577, 78_274_944] {
            let fraction = |accepted: usize| accepted as f32 / offered.max(1) as f32;
            for minimum in [
                -1.0,
                0.0,
                f32::MIN_POSITIVE,
                0.1,
                0.5,
                1.0 / 3.0,
                0.999_999,
                1.0,
                1.5,
                f32::INFINITY,
                f32::NAN,
            ] {
                let count = admission_count(&fraction, minimum);
                for accepted in [0, 1, count.saturating_sub(1), count, count + 1, offered] {
                    if accepted > u32::MAX as usize {
                        continue;
                    }
                    // As `integrate` decides: rejected below the minimum.
                    let refused = fraction(accepted) < minimum;
                    assert_eq!(
                        accepted >= count,
                        !refused,
                        "{offered} offered, minimum {minimum}, {accepted} accepted"
                    );
                }
            }
        }
    }
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

    /// Where a frame's raw pixels reach the sensor's ceiling, the stack
    /// records how much of each pixel's observations clipped, and a context
    /// keeps it.
    #[test]
    fn the_stack_records_where_frames_clipped_and_a_context_keeps_it() {
        let directory = tempfile::tempdir().unwrap();
        let raw = |image: LinearImage| FitsFrame {
            image,
            headers: vec![
                ("BITPIX".into(), HeaderValue::Integer(16)),
                ("BZERO".into(), HeaderValue::Float(32768.0)),
            ],
            exposure_seconds: None,
            bayer: None,
            source: None,
            bounds: None,
        };
        let field = stacking_star_field(160, 128);
        let mut stacker = LiveStacker::new(
            raw(field.clone()),
            CalibrationMasters::default(),
            StackOptions::default(),
        )
        .unwrap();
        assert!(stacker.clipped_fraction().is_none());
        // The brightest star's core, clipped in the second frame only.
        let core = [(91, 118), (92, 118), (91, 119), (92, 119)];
        let mut clipped = field.clone();
        for &(x, y) in &core {
            clipped.data[y * 160 + x] = 65535.0;
        }
        assert!(matches!(
            stacker.push(raw(clipped)).unwrap(),
            FrameDisposition::Accepted(_)
        ));
        let fraction = stacker.clipped_fraction().unwrap();
        assert_eq!(
            (fraction.width, fraction.height, fraction.channels),
            (160, 128, 1)
        );
        // One of the two observations clipped at the core, none in the sky;
        // the frame's sub-pixel registration spreads the share a little.
        let at = |x: usize, y: usize| fraction.data[y * 160 + x];
        for &(x, y) in &core {
            assert!((0.3..=0.5).contains(&at(x, y)), "({x}, {y}): {}", at(x, y));
        }
        let total: f32 = fraction.data.iter().sum();
        assert!((total - 2.0).abs() < 0.1, "{total}");
        assert_eq!(at(10, 10), 0.0);
        let path = directory.path().join("clipped.seiza-stack");
        stacker.save_context(&path).unwrap();
        let reopened = LiveStacker::open_context(&path).unwrap();
        assert_eq!(reopened.clipped_fraction().unwrap().data, fraction.data);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn sparse_clipping_matches_dense_across_regions_and_frame_order() {
        let (width, height, channels) = (43, 31, 3);
        let mut dense = Accumulator::new(width * height * channels, false);
        let mut sparse = Accumulator::new(width * height * channels, false);
        for frame in 0..10 {
            let region = ReferenceRegion {
                x: 3 + frame,
                y: 5,
                width: 13,
                height: 17,
            };
            let data = (0..region.width * region.height)
                .map(|index| match (index + frame) % 13 {
                    0 => f32::NAN,
                    1 => -0.25,
                    2 => 2.0,
                    3 => 1.0e-7,
                    4 => 0.37,
                    _ => 0.0,
                })
                .collect::<Vec<_>>();
            let entries = data
                .iter()
                .enumerate()
                .rev()
                .filter(|(_, value)| value.is_finite() && **value > 0.0)
                .map(|(index, value)| (index, *value))
                .collect();
            dense.add_clipped(
                &LinearImage::new(region.width, region.height, 1, data).unwrap(),
                region,
                width,
                channels,
            );
            sparse.add_clipped_mask(
                &crate::image::ClipMask::Sparse {
                    width: region.width,
                    height: region.height,
                    entries,
                },
                region,
                width,
                channels,
            );
            assert!(
                dense
                    .clipped
                    .as_ref()
                    .unwrap()
                    .iter()
                    .zip(sparse.clipped.as_ref().unwrap())
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
    }

    /// A stack's GPU options decide where its arithmetic runs, not what it
    /// stacks: its fingerprint leaves them out, and a context keeps them.
    #[test]
    fn gpu_options_leave_the_fingerprint_and_a_context_keeps_them() {
        let directory = tempfile::tempdir().unwrap();
        let field = stacking_star_field(160, 128);
        let open = |gpu| {
            let frame = FitsFrame {
                image: field.clone(),
                headers: Vec::new(),
                exposure_seconds: None,
                bayer: None,
                source: None,
                bounds: None,
            };
            let options = StackOptions {
                gpu,
                ..StackOptions::default()
            };
            LiveStacker::new(frame, CalibrationMasters::default(), options).unwrap()
        };
        let everywhere = open(crate::GpuOptions::default());
        let narrowed = crate::GpuOptions {
            detection: false,
            concurrency: 1,
            ..crate::GpuOptions::default()
        };
        let stacker = open(narrowed);
        assert_eq!(
            stacker.configuration_fingerprint(),
            everywhere.configuration_fingerprint()
        );
        let path = directory.path().join("gpu.seiza-stack");
        stacker.save_context(&path).unwrap();
        let reopened = LiveStacker::open_context(&path).unwrap();
        assert_eq!(reopened.options.gpu, narrowed);
        assert_eq!(
            reopened.configuration_fingerprint(),
            everywhere.configuration_fingerprint()
        );
        // The defaults are not written, so existing contexts keep their bytes.
        let written = serde_json::to_value(StackOptions::default()).unwrap();
        assert!(written.get("gpu").is_none(), "{written}");
        assert!(
            StackOptions {
                gpu: crate::GpuOptions {
                    concurrency: 0,
                    ..crate::GpuOptions::default()
                },
                ..StackOptions::default()
            }
            .validate()
            .is_err()
        );
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
    fn external_reference_replays_cached_uncached_and_restored_lights() {
        let directory = tempfile::tempdir().unwrap();
        let light = stacking_star_field(160, 128);
        let mut reference = stacking_star_field(176, 144);
        for value in &mut reference.data {
            *value = *value * 2.0 + 500.0;
        }
        let reference_path = directory.path().join("reference.fits");
        crate::write_processed_image_fits_f32(&reference_path, &reference, &[], &[]).unwrap();
        let paths = (0..3)
            .map(|index| {
                let path = directory.path().join(format!("light-{index}.fits"));
                crate::write_processed_image_fits_f32(
                    &path,
                    &offset_image(&light, index as f32),
                    &[],
                    &[],
                )
                .unwrap();
                path
            })
            .collect::<Vec<_>>();
        let bits = |image: &LinearImage| image.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for normalization in [
            NormalizationMode::None,
            NormalizationMode::LocalBackground { tile_size: 32 },
        ] {
            let options = StackOptions {
                normalization,
                weighting: FrameWeighting::inverse_noise_variance(),
                ..StackOptions::default()
            };
            let make = || {
                LiveStacker::with_reference_image(
                    FitsFrame::open(&reference_path).unwrap(),
                    FitsFrame::open(&paths[0]).unwrap(),
                    CalibrationMasters::new(
                        Some(LinearImage::new(160, 128, 1, vec![0.0; 160 * 128]).unwrap()),
                        None,
                        None,
                    )
                    .unwrap(),
                    options.clone(),
                )
                .unwrap()
            };
            let mut uncached = make();
            let mut cached = make();
            assert_eq!(
                cached.reference_noise(),
                measure_reference_noise(&light).unwrap()
            );
            assert_eq!(cached.view().accepted_frames, 0);
            assert!(cached.view().coverage.iter().all(|&count| count == 0));
            cached
                .retain_frames_for_reintegration(Some(directory.path()))
                .unwrap();
            for path in &paths {
                assert!(matches!(
                    uncached.push_fits(path).unwrap(),
                    FrameDisposition::Accepted(_)
                ));
                assert!(matches!(
                    cached.push_fits(path).unwrap(),
                    FrameDisposition::Accepted(_)
                ));
            }
            assert_eq!(
                bits(&uncached.snapshot().unwrap().image),
                bits(&cached.snapshot().unwrap().image)
            );
            let context = directory.path().join("external.seiza-stack");
            cached.save_context(&context).unwrap();
            let mut restored = LiveStacker::open_context(&context).unwrap();
            assert_eq!(
                bits(restored.alignment_reference.as_ref().unwrap()),
                bits(&reference)
            );
            restored
                .retain_frames_for_reintegration(Some(directory.path()))
                .unwrap();
            let batch = crate::BatchStackOptions {
                scratch_directory: Some(directory.path().to_path_buf()),
                ..Default::default()
            };
            let expected = uncached.reintegrate(&batch, |_, _, _| {}).unwrap().snapshot;
            assert_eq!(expected.accepted_frames, 3);
            for stacker in [&cached, &restored] {
                let actual = stacker.reintegrate(&batch, |_, _, _| {}).unwrap().snapshot;
                assert_eq!(actual.accepted_frames, 3);
                assert_eq!(bits(&actual.image), bits(&expected.image));
                assert_eq!(actual.coverage, expected.coverage);
            }
            let offset = if matches!(normalization, NormalizationMode::None) {
                1.0
            } else {
                0.0
            };
            assert!(
                (expected.image.data[70 * 176 + 70] - light.data[70 * 160 + 70] - offset).abs()
                    < 0.1
            );
            let (replayed, drizzled) = cached
                .reintegrate_drizzled(
                    &batch,
                    &crate::DrizzleOptions {
                        scale: 1,
                        drop_shrink: Some(1.0),
                    },
                    |_, _, _| {},
                )
                .unwrap();
            assert_eq!(replayed.snapshot.accepted_frames, 3);
            assert_eq!((drizzled.image.width, drizzled.image.height), (176, 144));
            let extra = directory.path().join("extra.fits");
            crate::write_processed_image_fits_f32(&extra, &offset_image(&light, 0.5), &[], &[])
                .unwrap();
            for stacker in [&mut uncached, &mut cached, &mut restored] {
                assert!(matches!(
                    stacker.push_fits(&extra).unwrap(),
                    FrameDisposition::Accepted(_)
                ));
            }
            assert_eq!(
                bits(&uncached.snapshot().unwrap().image),
                bits(&restored.snapshot().unwrap().image)
            );
        }
    }

    #[test]
    fn a_stack_opened_on_a_reference_image_protects_it_and_keeps_its_path() {
        let directory = tempfile::tempdir().unwrap();
        let light = stacking_star_field(160, 128);
        let reference = stacking_star_field(176, 144);
        let reference_path = directory.path().join("reference.fits");
        crate::write_processed_image_fits_f32(&reference_path, &reference, &[], &[]).unwrap();
        let bias_path = directory.path().join("bias.fits");
        crate::write_processed_image_fits_f32(
            &bias_path,
            &LinearImage::new(160, 128, 1, vec![0.0; 160 * 128]).unwrap(),
            &[],
            &[],
        )
        .unwrap();
        let paths = (0..2)
            .map(|index| {
                let path = directory.path().join(format!("light-{index}.fits"));
                crate::write_processed_image_fits_f32(
                    &path,
                    &offset_image(&light, index as f32),
                    &[],
                    &[],
                )
                .unwrap();
                path
            })
            .collect::<Vec<_>>();
        let open = |reference: &Path, channel: &Path| {
            LiveStacker::open_fits_with_reference_image(
                reference,
                channel,
                Some(&bias_path),
                None,
                None,
                None,
                StackOptions::default(),
            )
        };
        // Neither image may be a calibration master.
        assert!(open(&bias_path, &paths[0]).is_err());
        assert!(open(&reference_path, &bias_path).is_err());

        let mut stacker = open(&reference_path, &paths[0]).unwrap();
        assert!(!stacker.reference_is_integrated());
        assert_eq!(stacker.view().accepted_frames, 0);
        assert_eq!((stacker.view().width, stacker.view().height), (176, 144));
        assert_eq!(stacker.input_paths(), [path_identity(&bias_path)]);
        assert_eq!(
            stacker.protected_paths(),
            [path_identity(&bias_path), path_identity(&reference_path)]
        );
        for path in &paths {
            assert!(matches!(
                stacker.push_fits(path).unwrap(),
                FrameDisposition::Accepted(_)
            ));
        }
        assert_eq!(stacker.view().accepted_frames, 2);
        assert!(stacker.save_context(&reference_path).is_err());

        let context = directory.path().join("stack.seiza-stack");
        stacker.save_context(&context).unwrap();
        let restored = LiveStacker::open_context(&context).unwrap();
        assert!(!restored.reference_is_integrated());
        assert_eq!(
            restored.reference_image_path(),
            Some(path_identity(&reference_path).as_path())
        );
        assert_eq!(restored.protected_paths(), stacker.protected_paths());

        // A context marked as aligned to an external image must hold that
        // image, or it would reopen aligned to its channel light instead.
        let bytes = std::fs::read(&context).unwrap();
        let payload = zstd::stream::decode_all(&bytes[12..]).unwrap();
        let metadata_end = 8 + u64::from_le_bytes(payload[..8].try_into().unwrap()) as usize;
        let mut metadata: serde_json::Value =
            serde_json::from_slice(&payload[8..metadata_end]).unwrap();
        let fields = metadata.as_object_mut().unwrap();
        assert_eq!(fields["external_reference"], true);
        fields.remove("alignment_reference").unwrap();
        fields.remove("reference_image_path").unwrap();
        let metadata = serde_json::to_vec(&metadata).unwrap();
        let alignment_bytes = 176 * 144 * 4;
        let mut stripped = (metadata.len() as u64).to_le_bytes().to_vec();
        stripped.extend(&metadata);
        stripped.extend(&payload[metadata_end..payload.len() - alignment_bytes]);
        let mut rewritten = bytes[..12].to_vec();
        rewritten.extend(zstd::stream::encode_all(&stripped[..], 1).unwrap());
        std::fs::write(&context, rewritten).unwrap();
        let Err(error) = LiveStacker::open_context(&context) else {
            panic!("an external-reference context without its alignment image must be refused");
        };
        assert!(
            error.to_string().contains("no alignment reference"),
            "{error}"
        );

        // The reference image may be a light as well, pushed like any other.
        let mut stacker = open(&paths[1], &paths[0]).unwrap();
        for path in &paths {
            assert!(matches!(
                stacker.push_fits(path).unwrap(),
                FrameDisposition::Accepted(_)
            ));
        }
        assert_eq!(stacker.view().accepted_frames, 2);
    }

    /// `image` turned a quarter turn.
    fn quarter_turn(image: &LinearImage) -> LinearImage {
        let (width, height) = (image.height, image.width);
        let channels = image.channels;
        let data = (0..width * height)
            .flat_map(|index| {
                let (x, y) = (index % width, index / width);
                let base = ((image.height - 1 - x) * image.width + y) * channels;
                image.data[base..base + channels].iter().copied()
            })
            .collect();
        LinearImage::new(width, height, channels, data).unwrap()
    }

    #[test]
    fn the_next_channel_reference_is_tried_when_one_does_not_align() {
        let light = stacking_star_field(160, 128);
        let reference = stacking_star_field(176, 144);
        let frame = |image: &LinearImage| FitsFrame {
            image: image.clone(),
            headers: Vec::new(),
            exposure_seconds: None,
            bayer: None,
            source: None,
            bounds: None,
        };
        let open = |candidates: &[&LinearImage], heard: &mut Vec<usize>| {
            LiveStacker::with_reference_image_candidates(
                frame(&reference),
                candidates.iter().map(|image| Ok::<_, Error>(frame(image))),
                CalibrationMasters::default(),
                StackOptions::default(),
                |index, _| heard.push(index),
            )
        };
        let turned = quarter_turn(&light);

        let mut heard = Vec::new();
        let (stacker, used) = open(&[&turned, &light], &mut heard).unwrap();
        assert_eq!((used, heard), (1, vec![0]));
        assert_eq!(stacker.view().accepted_frames, 0);
        assert_eq!((stacker.view().width, stacker.view().height), (176, 144));

        let mut heard = Vec::new();
        let Err(error) = open(&[&turned, &turned], &mut heard) else {
            panic!("no channel reference aligns");
        };
        assert_eq!(heard, [0, 1]);
        let error = error.to_string();
        assert!(error.contains("none of 2 channel references"), "{error}");

        // A light whose channel count differs from the reference image's
        // ends the search: the next light would differ too.
        let color = LinearImage::new(
            160,
            128,
            3,
            light.data.iter().flat_map(|&value| [value; 3]).collect(),
        )
        .unwrap();
        let mut heard = Vec::new();
        assert!(open(&[&color, &light], &mut heard).is_err());
        assert!(heard.is_empty());
    }

    /// The GPU's accumulator classifies and integrates as the CPU's does,
    /// weighted or not, for whole frames and frames over part of the grid,
    /// with outliers for delta-sigma rejection to refuse.
    #[cfg(feature = "gpu")]
    #[test]
    fn the_gpu_accumulator_matches_the_cpu() {
        let (width, height) = (1200_usize, 1000_usize);
        let rejection = RejectionMode::DeltaSigma(DeltaSigmaOptions::default());
        for weighted in [false, true] {
            let mut cpu = Accumulator::new(width * height, weighted);
            let mut gpu = Accumulator::new(width * height, weighted);
            gpu.offer_gpu(&crate::GpuOptions::default());
            if gpu.gpu.is_none() {
                eprintln!("no GPU: the GPU accumulator not compared");
                return;
            }
            let mut state = 0x2545_f491_4f6c_dd1d_u64;
            let mut uniform = move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f32 / (1_u64 << 53) as f32
            };
            for frame in 0..12 {
                // Every third frame covers only part of the grid.
                let region = if frame % 3 == 2 {
                    ReferenceRegion {
                        x: 101,
                        y: 57,
                        width: 700,
                        height: 600,
                    }
                } else {
                    ReferenceRegion {
                        x: 0,
                        y: 0,
                        width,
                        height,
                    }
                };
                let data: Vec<f32> = (0..region.width * region.height)
                    .map(|index| {
                        let roll = uniform();
                        if roll < 0.01 {
                            f32::NAN
                        } else if roll > 0.995 {
                            5_000.0
                        } else {
                            100.0 + (index % 97) as f32 + (uniform() - 0.5) * 20.0
                        }
                    })
                    .collect();
                let image = LinearImage::new(region.width, region.height, 1, data).unwrap();
                let placed = Placed {
                    image: &image,
                    region,
                    width,
                    height,
                };
                let weights = [0.6 + 0.1 * frame as f32];
                let weights = weighted.then_some(&weights[..]);
                // Single precision on both, but a GPU may fuse a multiply
                // and add the CPU rounds apart, so a sample on a threshold
                // can fall the other way: at most one in 100 000.
                let close = |(a, b): (usize, usize), (c, d): (usize, usize)| {
                    let total = (c + d).max(1);
                    a.abs_diff(c) * 100_000 <= total && b.abs_diff(d) * 100_000 <= total
                };
                let (classified, expected) = cpu.admit_frame(placed, rejection, weights, 0);
                let (would_accept, actual) = gpu.admit_frame(placed, rejection, weights, 0);
                let (expected, actual) = (expected.unwrap(), actual.unwrap());
                assert!(
                    close((would_accept, 0), (classified, 0)),
                    "frame {frame} weighted {weighted}: classified {would_accept} against \
                     {classified}"
                );
                assert!(
                    close(actual, expected),
                    "frame {frame} weighted {weighted}: integrated {actual:?} against {expected:?}"
                );
                // A frame short of its admission count is classified, and
                // left out of the moments on both.
                let refused_at = classified + classified / 2 + 1;
                assert_eq!(
                    cpu.admit_frame(placed, rejection, weights, refused_at).1,
                    None
                );
                assert_eq!(
                    gpu.admit_frame(placed, rejection, weights, refused_at).1,
                    None
                );
            }
            let (cpu, gpu) = (cpu.moments(), gpu.moments());
            let differing = gpu
                .count
                .iter()
                .zip(cpu.count)
                .zip(gpu.rejected.iter().zip(cpu.rejected))
                .filter(|((a, b), (c, d))| a != b || c != d)
                .count();
            assert!(
                differing * 100_000 <= cpu.count.len(),
                "weighted {weighted}: {differing} samples counted apart"
            );
            let apart = gpu
                .mean
                .iter()
                .zip(cpu.mean)
                .filter(|(gpu, cpu)| (**gpu - **cpu).abs() > 1.0e-4 * cpu.abs().max(1.0))
                .count();
            assert!(
                apart * 100_000 <= cpu.mean.len(),
                "weighted {weighted}: {apart} means apart"
            );
        }
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
                reference_is_integrated: stacker.reference_is_integrated,
                alignment_reference: stacker.alignment_reference.as_ref(),
                reference_image_path: stacker.reference_image_path.as_deref(),
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
                expanded_panels: &[],
                expanded_panel_stars: Vec::new(),
                expanded_panel_projections: Vec::new(),
                expanded_lens: None,
                clipped: None,
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

    /// Contexts saved before expanded canvases kept their panels' stars
    /// hold the panels alone; reopened, each panel matches the stars of the
    /// canvas beneath it, and frames still join.
    #[test]
    fn expanded_context_without_panel_stars_still_registers_frames() {
        let (width, height, step) = (320_usize, 240_usize, 200_usize);
        let sky_width = width + step;
        let mut state = 0x5eed_u64;
        let mut random = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as f64 / (1u64 << 31) as f64
        };
        let stars = (0..450)
            .map(|_| {
                (
                    random() * sky_width as f64,
                    random() * height as f64,
                    300.0 + 3000.0 * random(),
                )
            })
            .collect::<Vec<_>>();
        let directory = tempfile::tempdir().unwrap();
        let panel = |name: &str, offset: f64| {
            let mut data = vec![100.0_f32; width * height];
            for &(sx, sy, peak) in &stars {
                let x = sx - offset;
                for py in (sy as isize - 4).max(0)..(sy as isize + 5).min(height as isize) {
                    for px in (x as isize - 4).max(0)..(x as isize + 5).min(width as isize) {
                        let d2 = (px as f64 - x).powi(2) + (py as f64 - sy).powi(2);
                        data[py as usize * width + px as usize] +=
                            (peak * (-d2 / 3.2).exp()) as f32;
                    }
                }
            }
            let path = directory.path().join(name);
            seiza_fits::write_f32_image(
                &path,
                width,
                height,
                seiza_fits::F32ImageData::Mono(&data),
                &[],
            )
            .unwrap();
            path
        };
        let paths = [panel("a.fits", 0.0), panel("b.fits", step as f64)];
        let mut options = StackOptions {
            normalization: NormalizationMode::None,
            ..StackOptions::default()
        };
        options.acceptance.minimum_overlap_fraction = 0.1;
        let mut stacker = LiveStacker::with_expanding_reference(
            &paths,
            CalibrationMasters::default(),
            options,
            usize::MAX,
            |_, _, _, _| crate::Continue::Yes,
        )
        .unwrap();
        let placed = paths
            .iter()
            .map(|path| match stacker.push_fits(path).unwrap() {
                FrameDisposition::Accepted(diagnostics) => diagnostics.transform,
                other => panic!("{other:?}"),
            })
            .collect::<Vec<_>>();
        let canvas = stacker.canvas_registrar.as_ref().unwrap();
        let path = directory.path().join("before-panel-stars.seiza-stack");
        context::write(
            &path,
            context::ContextWriteState {
                reference_is_integrated: stacker.reference_is_integrated,
                alignment_reference: stacker.alignment_reference.as_ref(),
                reference_image_path: stacker.reference_image_path.as_deref(),
                options: &stacker.options,
                calibration: &stacker.calibration,
                reference: &stacker.reference,
                reference_headers: &stacker.reference_headers,
                reference_metadata: &stacker.reference_metadata,
                mean: &stacker.accumulator.mean,
                m2: &stacker.accumulator.m2,
                count: &stacker.accumulator.count,
                rejected: &stacker.accumulator.rejected,
                weight_sum: stacker.accumulator.weight_sum.as_deref(),
                reference_noise: &stacker.reference_noise,
                accepted_frames: stacker.accepted_frames,
                rejected_frames: stacker.rejected_frames,
                input_paths: &stacker.input_paths,
                input_mode: stacker.input_mode,
                ledger: &stacker.ledger,
                expanded_panels: &canvas.regions(),
                // As written before panels kept their stars.
                expanded_panel_stars: Vec::new(),
                expanded_panel_projections: Vec::new(),
                expanded_lens: None,
                clipped: None,
            },
        )
        .unwrap();
        let mut restored = LiveStacker::open_context(&path).unwrap();
        assert_eq!(
            restored.canvas_registrar.as_ref().unwrap().regions(),
            canvas.regions()
        );
        let late = panel("late.fits", step as f64 + 3.4);
        let FrameDisposition::Accepted(diagnostics) = restored.push_fits(&late).unwrap() else {
            panic!("a frame joins a context saved without panel stars");
        };
        // It lands where the panel it repeats was planned, 3.4px further on.
        let (x, y) = diagnostics.transform.apply(0.0, 0.0);
        let (px, py) = placed[1].apply(3.4, 0.0);
        assert!((x - px).hypot(y - py) < 0.5, "{x},{y} against {px},{py}");
        assert!(diagnostics.registration_rms_pixels < 0.5, "{diagnostics:?}");
    }

    #[test]
    fn older_lens_context_drops_raw_anchor_wcs_on_reopening() {
        let reference = stacking_star_field(160, 128);
        let mut stacker = LiveStacker::from_linear(reference, StackOptions::default()).unwrap();
        stacker.reference_is_integrated = false;
        stacker.canvas_registrar = Some(
            crate::expand::CanvasRegistrar::new(
                &stacker.reference,
                &[ReferenceRegion {
                    x: 0,
                    y: 0,
                    width: 160,
                    height: 128,
                }],
                None,
                Some(crate::lens::LensDistortion::none(160, 128)),
                &stacker.options.registration,
                crate::binning::Scale::NATIVE,
                &crate::GpuOptions::default(),
            )
            .unwrap(),
        );
        // Simulate a context written before raw WCS was invalidated.
        stacker.reference_headers = vec![
            ("CRPIX1".into(), HeaderValue::Float(80.5)),
            ("CTYPE1".into(), HeaderValue::String("RA---TAN-SIP".into())),
            ("A_2_0".into(), HeaderValue::Float(1.0e-5)),
            ("EXPTIME".into(), HeaderValue::Float(60.0)),
        ];
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("old-lens.context");
        stacker.save_context(&path).unwrap();
        let restored = LiveStacker::open_context(&path).unwrap();
        assert!(restored.expanded_lens().is_some());
        assert_eq!(
            restored.reference_headers(),
            &[("EXPTIME".into(), HeaderValue::Float(60.0))]
        );
    }

    #[test]
    fn a_light_admitted_at_its_measured_angle_replays_at_it() {
        // A light recorded at 113.8 degrees whose own pixels showed it
        // follows the flat at 3.2: admitted under a gate that reads it at
        // its measured angle, and replayed under that gate after the stack
        // narrows again, and after a saved context.
        use seiza_calibration::{LightRotation, MatchTolerances};
        let directory = tempfile::tempdir().unwrap();
        let star_field = stacking_star_field(160, 128);
        let write = |name: &str, image: &LinearImage, cards: Vec<seiza_fits::WriteHeaderCard>| {
            let path = directory.path().join(name);
            crate::write_processed_image_fits_f32(&path, image, &[], &cards).unwrap();
            path
        };
        let rotation =
            |degrees| seiza_fits::WriteHeaderCard::new("ROTATANG", HeaderValue::Float(degrees));
        let light = |name: &str, offset: f32, degrees: f64| {
            write(
                name,
                &offset_image(&star_field, offset),
                vec![
                    seiza_fits::WriteHeaderCard::new(
                        "IMAGETYP",
                        HeaderValue::String("LIGHT".into()),
                    ),
                    rotation(degrees),
                ],
            )
        };
        let flat = write(
            "flat.fits",
            &LinearImage::new(160, 128, 1, vec![1.0; 160 * 128]).unwrap(),
            vec![
                seiza_fits::WriteHeaderCard::new("SEIZAMST", HeaderValue::String("FLAT".into())),
                rotation(3.2),
            ],
        );
        let reference = light("reference.fits", 0.0, 3.2);
        let mut stacker = LiveStacker::open_fits(
            &reference,
            None,
            None,
            Some(&flat),
            None,
            StackOptions::default(),
        )
        .unwrap();
        let measured = MatchTolerances {
            light_rotation: LightRotation::Measured { degrees: 3.0 },
            ..MatchTolerances::default()
        };
        stacker.set_match_tolerances(measured).unwrap();
        assert!(matches!(
            stacker.push_fits(light("turned.fits", 1.0, 113.8)).unwrap(),
            FrameDisposition::Accepted(_)
        ));
        stacker
            .set_match_tolerances(MatchTolerances::default())
            .unwrap();
        assert!(matches!(
            stacker
                .push_fits(light("turned-2.fits", 2.0, 113.8))
                .unwrap(),
            FrameDisposition::Rejected(FrameRejectionReason::Calibration(_))
        ));
        let batch = crate::BatchStackOptions::default();
        let replayed = stacker.reintegrate(&batch, |_, _, _| {}).unwrap();
        assert_eq!(replayed.snapshot.accepted_frames, 2);

        let path = directory.path().join("measured.seiza-stack");
        stacker.save_context(&path).unwrap();
        let restored = LiveStacker::open_context(&path).unwrap();
        assert_eq!(restored.match_tolerances(), MatchTolerances::default());
        // The stack's own gate would refuse the turned light: it replays
        // because the context kept the gate it was admitted under.
        let replayed = restored.reintegrate(&batch, |_, _, _| {}).unwrap();
        assert_eq!(replayed.snapshot.accepted_frames, 2);
    }

    #[test]
    fn a_stack_judges_each_light_under_its_gate_and_replays_it_so() {
        // A flat from 3.2 degrees, chosen on purpose for a light at 113.8:
        // the stack admits it under a gate opened for that light, judges
        // later lights under the narrowed gate again, and replays each
        // frame under the gate it was admitted under, after a saved context
        // too.
        use seiza_calibration::MatchTolerances;
        let directory = tempfile::tempdir().unwrap();
        let star_field = stacking_star_field(160, 128);
        let write = |name: &str, image: &LinearImage, cards: Vec<seiza_fits::WriteHeaderCard>| {
            let path = directory.path().join(name);
            crate::write_processed_image_fits_f32(&path, image, &[], &cards).unwrap();
            path
        };
        let rotation =
            |degrees| seiza_fits::WriteHeaderCard::new("ROTATANG", HeaderValue::Float(degrees));
        let light = |name: &str, offset: f32, degrees: f64| {
            write(
                name,
                &offset_image(&star_field, offset),
                vec![
                    seiza_fits::WriteHeaderCard::new(
                        "IMAGETYP",
                        HeaderValue::String("LIGHT".into()),
                    ),
                    rotation(degrees),
                ],
            )
        };
        let flat = write(
            "flat.fits",
            &LinearImage::new(160, 128, 1, vec![1.0; 160 * 128]).unwrap(),
            vec![
                seiza_fits::WriteHeaderCard::new("SEIZAMST", HeaderValue::String("FLAT".into())),
                rotation(3.2),
            ],
        );
        let reference = light("reference.fits", 0.0, 3.2);
        let turned = light("turned.fits", 1.0, 113.8);
        let open = |reference: &Path, match_tolerances| {
            LiveStacker::open_fits(
                reference,
                None,
                None,
                Some(&flat),
                None,
                StackOptions {
                    match_tolerances,
                    ..StackOptions::default()
                },
            )
        };
        let wide = MatchTolerances {
            rotation_deg: 180.0,
            ..MatchTolerances::default()
        };

        // A reference is judged under the gate the stack opens with.
        assert!(open(&turned, MatchTolerances::default()).is_err());
        assert_eq!(open(&turned, wide).unwrap().match_tolerances(), wide);

        let mut stacker = open(&reference, MatchTolerances::default()).unwrap();
        let narrow_fingerprint = stacker.configuration_fingerprint().to_owned();
        assert!(matches!(
            stacker.push_fits(&turned).unwrap(),
            FrameDisposition::Rejected(FrameRejectionReason::Calibration(ref reason))
                if reason.contains("tolerance 2.00")
        ));

        stacker.set_match_tolerances(wide).unwrap();
        assert_ne!(stacker.configuration_fingerprint(), narrow_fingerprint);
        assert!(matches!(
            stacker
                .push_fits(light("turned-2.fits", 2.0, 113.8))
                .unwrap(),
            FrameDisposition::Accepted(_)
        ));
        stacker
            .set_match_tolerances(MatchTolerances::default())
            .unwrap();
        assert_eq!(stacker.configuration_fingerprint(), narrow_fingerprint);
        assert!(matches!(
            stacker
                .push_fits(light("turned-3.fits", 3.0, 113.8))
                .unwrap(),
            FrameDisposition::Rejected(FrameRejectionReason::Calibration(_))
        ));
        assert!(matches!(
            stacker.push_fits(light("level.fits", 3.0, 3.4)).unwrap(),
            FrameDisposition::Accepted(_)
        ));
        assert_eq!(stacker.view().accepted_frames, 3);

        // The frame admitted under the wide gate replays under it, though
        // the stack's own gate is narrow again.
        let batch = crate::BatchStackOptions::default();
        let replayed = stacker.reintegrate(&batch, |_, _, _| {}).unwrap();
        assert_eq!(replayed.snapshot.accepted_frames, 3);

        let path = directory.path().join("gated.seiza-stack");
        stacker.save_context(&path).unwrap();
        let mut restored = LiveStacker::open_context(&path).unwrap();
        assert_eq!(restored.match_tolerances(), MatchTolerances::default());
        assert_eq!(restored.configuration_fingerprint(), narrow_fingerprint);
        let replayed = restored.reintegrate(&batch, |_, _, _| {}).unwrap();
        assert_eq!(replayed.snapshot.accepted_frames, 3);

        // Tolerances that cannot decide anything, or cannot be saved, are
        // refused, and leave the stack as it was.
        for broken in [
            MatchTolerances {
                rotation_deg: f64::INFINITY,
                ..MatchTolerances::default()
            },
            MatchTolerances {
                dark_temperature_c: -1.0,
                ..MatchTolerances::default()
            },
            MatchTolerances {
                focal_length_mm: f64::NAN,
                ..MatchTolerances::default()
            },
        ] {
            assert!(restored.set_match_tolerances(broken).is_err());
            assert!(open(&reference, broken).is_err());
        }
        assert_eq!(restored.match_tolerances(), MatchTolerances::default());
    }

    #[test]
    fn unusable_tolerances_are_refused_where_they_enter() {
        // JSON writes an infinite or NaN tolerance as null, so a stack
        // holding one saved a context that failed to open; NaN and a
        // negative value refused every light. They are refused where they
        // enter, and at save time should one get past that.
        let image = stacking_star_field(160, 128);
        let frame = |rotation_deg: f64| FitsFrame {
            image: image.clone(),
            headers: vec![
                ("IMAGETYP".into(), HeaderValue::String("LIGHT".into())),
                ("ROTATANG".into(), HeaderValue::Float(rotation_deg)),
            ],
            exposure_seconds: Some(60.0),
            bayer: None,
            source: None,
            bounds: None,
        };
        let mut calibration = CalibrationMasters::new(
            None,
            None,
            Some(crate::MasterFlat::raw(
                LinearImage::new(160, 128, 1, vec![1.0; 160 * 128]).unwrap(),
            )),
        )
        .unwrap();
        calibration.flat_signature.as_mut().unwrap().rotation_deg = Some(3.2);
        let mut light = seiza_calibration::FrameSignature::default();
        light.rotation_deg = Some(113.8);
        let mut stacker =
            LiveStacker::new(frame(3.2), calibration.clone(), StackOptions::default()).unwrap();

        let unusable = [
            ("rotation_deg", f64::INFINITY),
            ("rotation_deg", f64::NAN),
            ("rotation_deg", 360.0),
            ("dark_temperature_c", -1.0),
            ("focal_length_mm", f64::NEG_INFINITY),
        ];
        for (field, value) in unusable {
            let mut tolerances = seiza_calibration::MatchTolerances::default();
            match field {
                "rotation_deg" => tolerances.rotation_deg = value,
                "dark_temperature_c" => tolerances.dark_temperature_c = value,
                _ => tolerances.focal_length_mm = value,
            }
            let refused = |result: Result<()>| {
                let error = result.unwrap_err().to_string();
                assert!(
                    error.contains(field) && error.contains("invalid match tolerance"),
                    "{field}={value}: {error}"
                );
            };
            refused(stacker.set_match_tolerances(tolerances));
            refused(
                LiveStacker::new(
                    frame(3.2),
                    calibration.clone(),
                    StackOptions {
                        match_tolerances: tolerances,
                        ..StackOptions::default()
                    },
                )
                .map(drop),
            );
            refused(calibration.validate_light_signature_with(&light, &tolerances));
            refused(calibration.validate_light_frame_with(&frame(113.8), &tolerances));
            refused(
                calibration
                    .compatible_for_light_with(&light, &tolerances)
                    .map(drop),
            );
            // Refused before the masters are asked anything, so an empty
            // set does not hide the mistake.
            refused(
                CalibrationMasters::default().validate_light_signature_with(&light, &tolerances),
            );
        }
        assert_eq!(
            stacker.match_tolerances(),
            seiza_calibration::MatchTolerances::default()
        );

        // Should unusable tolerances reach the stack some other way, the
        // context is not written.
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("infinite.seiza-stack");
        let mut broken =
            LiveStacker::new(frame(3.2), calibration, StackOptions::default()).unwrap();
        broken.options.match_tolerances.rotation_deg = f64::INFINITY;
        let error = broken.save_context(&path).unwrap_err().to_string();
        assert!(error.contains("rotation_deg"), "{error}");
        assert!(!path.exists());

        // The widest usable rotation gate admits any angle and survives
        // a saved context.
        let any_angle = seiza_calibration::MatchTolerances {
            rotation_deg: seiza_calibration::MatchTolerances::MAX_ROTATION_DEG,
            ..seiza_calibration::MatchTolerances::default()
        };
        stacker.set_match_tolerances(any_angle).unwrap();
        assert!(matches!(
            stacker.push(frame(113.8)).unwrap(),
            FrameDisposition::Accepted(_)
        ));
        stacker.save_context(&path).unwrap();
        let restored = LiveStacker::open_context(&path).unwrap();
        assert_eq!(restored.match_tolerances(), any_angle);
        assert_eq!(
            restored.configuration_fingerprint(),
            stacker.configuration_fingerprint()
        );

        // A light angle the host measured: the gate follows it, refuses a
        // light measured elsewhere, and a saved context keeps it.
        let measured = |degrees| seiza_calibration::MatchTolerances {
            light_rotation: seiza_calibration::LightRotation::Measured { degrees },
            ..seiza_calibration::MatchTolerances::default()
        };
        let any_angle_fingerprint = stacker.configuration_fingerprint().to_owned();
        stacker.set_match_tolerances(measured(60.0)).unwrap();
        assert!(matches!(
            stacker.push(frame(3.2)).unwrap(),
            FrameDisposition::Rejected(FrameRejectionReason::Calibration(_))
        ));
        stacker.set_match_tolerances(measured(3.0)).unwrap();
        assert_ne!(stacker.configuration_fingerprint(), any_angle_fingerprint);
        assert!(matches!(
            stacker.push(frame(113.8)).unwrap(),
            FrameDisposition::Accepted(_)
        ));
        stacker.save_context(&path).unwrap();
        let restored = LiveStacker::open_context(&path).unwrap();
        assert_eq!(restored.match_tolerances(), measured(3.0));
        assert_eq!(
            restored.configuration_fingerprint(),
            stacker.configuration_fingerprint()
        );
    }

    #[test]
    fn master_building_tolerances_leave_a_stack_identity_alone() {
        // Only the gates a light is admitted by belong to the stack's
        // identity and context; a host passing its master-building
        // settings through changes neither.
        let reference = stacking_star_field(160, 128);
        let calibration = CalibrationMasters::new(
            None,
            None,
            Some(crate::MasterFlat::raw(
                LinearImage::new(160, 128, 1, vec![1.0; 160 * 128]).unwrap(),
            )),
        )
        .unwrap();
        let light = || FitsFrame {
            image: reference.clone(),
            headers: vec![("IMAGETYP".into(), HeaderValue::String("LIGHT".into()))],
            exposure_seconds: Some(60.0),
            bayer: None,
            source: None,
            bounds: None,
        };
        let mut stacker = LiveStacker::new(light(), calibration, StackOptions::default()).unwrap();
        let default_fingerprint = stacker.configuration_fingerprint().to_owned();
        let building = seiza_calibration::MatchTolerances {
            master_temperature_c: 0.5,
            flat_session_seconds: 3_600,
            ..seiza_calibration::MatchTolerances::default()
        };
        stacker.set_match_tolerances(building).unwrap();
        assert_eq!(stacker.configuration_fingerprint(), default_fingerprint);
        assert!(matches!(
            stacker.push(light()).unwrap(),
            FrameDisposition::Accepted(_)
        ));
        // Nor does a frame admitted under them need its gate written down.
        assert!(stacker.ledger.tolerances().is_none());

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("building.seiza-stack");
        stacker.save_context(&path).unwrap();
        let restored = LiveStacker::open_context(&path).unwrap();
        assert_eq!(restored.configuration_fingerprint(), default_fingerprint);
        assert_eq!(
            restored.match_tolerances(),
            seiza_calibration::MatchTolerances::default()
        );
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

    /// Dithered Bayer lights with no dark to subtract their hot photosites,
    /// each brighter than any star: demosaiced, they look like stars that
    /// never move. The learned map keeps them out of registration, which
    /// then follows the stars, and out of the stack, the reference's
    /// included, which online rejection never judges.
    #[test]
    fn a_defect_map_keeps_hot_photosites_out_of_registration_and_the_stack() {
        let (width, height) = (160, 128);
        let hot = [
            (30, 40),
            (95, 21),
            (131, 77),
            (48, 100),
            (110, 111),
            (70, 60),
            (17, 90),
            (140, 30),
        ];
        let shifts = [
            (0.0, 0.0),
            (4.0, -3.0),
            (-3.0, 5.0),
            (6.0, 2.0),
            (-5.0, -4.0),
            (2.0, 6.0),
        ];
        let directory = tempfile::tempdir().unwrap();
        let bayer = [("BAYERPAT".to_string(), HeaderValue::String("RGGB".into()))];
        let paths = shifts
            .iter()
            .enumerate()
            .map(|(frame, &(dx, dy))| {
                let mut image = bayer_star_field(width, height, dx, dy);
                for (index, value) in image.data.iter_mut().enumerate() {
                    let noise = (index as u64 * 2_654_435_761 + frame as u64 * 40_503) % 997;
                    *value += noise as f32 / 997.0 * 8.0 - 4.0;
                }
                for &(x, y) in &hot {
                    image.data[y * width + x] += 30_000.0;
                }
                let path = directory.path().join(format!("light-{frame}.fits"));
                crate::write_processed_image_fits_f32(&path, &image, &bayer, &[]).unwrap();
                path
            })
            .collect::<Vec<_>>();
        let map = crate::detect_defects(
            &paths,
            &CalibrationMasters::default(),
            &crate::DefectDetectionOptions::default(),
        )
        .unwrap();
        let mut expected = hot.map(|(x, y)| (y * width + x) as u32);
        expected.sort_unstable();
        assert_eq!(map.pixels(), expected);

        let options = StackOptions {
            defects: Some(map),
            ..StackOptions::default()
        };
        let mut stacker =
            LiveStacker::open_fits(&paths[0], None, None, None, None, options).unwrap();
        for (path, &(dx, dy)) in paths[1..].iter().zip(&shifts[1..]) {
            let FrameDisposition::Accepted(diagnostics) = stacker.push_fits(path).unwrap() else {
                panic!("{} must be admitted", path.display());
            };
            let dither = f64::from(f32::hypot(dx, dy));
            assert!(
                (diagnostics.registration_drift_pixels - dither).abs() < 0.5,
                "registered {} px off a {dither} px dither",
                diagnostics.registration_drift_pixels
            );
        }
        let snapshot = stacker.snapshot().unwrap();
        for &(x, y) in &hot {
            for channel in 0..3 {
                let value = snapshot.image.data[(y * width + x) * 3 + channel];
                assert!(value < 1_000.0, "({x}, {y}) channel {channel}: {value}");
            }
        }
    }

    #[test]
    fn a_defect_map_travels_in_the_options_and_the_context() {
        let plain = serde_json::to_value(StackOptions::default()).unwrap();
        assert!(plain.get("defects").is_none());
        let map = crate::DefectMap::new(160, 128, vec![90 * 160 + 60]).unwrap();
        let options = StackOptions {
            defects: Some(map.clone()),
            ..StackOptions::default()
        };
        let json = serde_json::to_string(&options).unwrap();
        let parsed: StackOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.defects.as_ref(), Some(&map));
        let unordered = StackOptions {
            defects: serde_json::from_str(r#"{"width":4,"height":4,"pixels":[3,1]}"#).unwrap(),
            ..StackOptions::default()
        };
        assert!(unordered.validate().is_err());

        let fingerprint = |options: StackOptions| {
            stack_configuration_fingerprint(
                &options,
                &CalibrationMasters::default(),
                FrameInputMode::CalibrateAndPrepare,
            )
            .unwrap()
        };
        assert_ne!(
            fingerprint(options.clone()),
            fingerprint(StackOptions::default())
        );

        let stacker = LiveStacker::from_linear(stacking_star_field(160, 128), options).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("defects.seiza-stack");
        stacker.save_context(&path).unwrap();
        let restored = LiveStacker::open_context(&path).unwrap();
        assert_eq!(restored.options.defects, Some(map));
        assert_eq!(
            restored.configuration_fingerprint(),
            stacker.configuration_fingerprint()
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

    /// Drizzling Bayer frames drops each photosite into its own colour, so
    /// the sky keeps its colour, and a trail the integration rejected stays
    /// out, whether the stack demosaiced or Bayer-drizzled its frames.
    #[test]
    fn drizzled_bayer_frames_keep_their_colours_and_reject_a_trail() {
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
        const TRAIL_ROW: usize = 60;
        let directory = tempfile::tempdir().unwrap();
        let bayer = [("BAYERPAT".to_string(), HeaderValue::String("RGGB".into()))];
        let paths = shifts
            .iter()
            .enumerate()
            .map(|(index, &(dx, dy))| {
                let path = directory.path().join(format!("light-{index}.fits"));
                let mut image = bayer_star_field(width, height, dx, dy);
                if index == 3 {
                    for y in TRAIL_ROW - 1..=TRAIL_ROW + 1 {
                        for x in 10..width - 10 {
                            image.data[y * width + x] += 20_000.0;
                        }
                    }
                }
                crate::write_processed_image_fits_f32(&path, &image, &bayer, &[]).unwrap();
                path
            })
            .collect::<Vec<_>>();
        for cfa_integration in [CfaIntegration::Demosaic, CfaIntegration::BayerDrizzle] {
            let options = StackOptions {
                normalization: NormalizationMode::None,
                rejection: RejectionMode::None,
                cfa_integration,
                ..StackOptions::default()
            };
            let mut stacker =
                LiveStacker::open_fits(&paths[0], None, None, None, None, options).unwrap();
            for path in &paths[1..] {
                stacker.push_fits(path).unwrap();
            }
            let (_, drizzled) = stacker
                .reintegrate_drizzled(
                    &crate::BatchStackOptions::default(),
                    &crate::DrizzleOptions::default(),
                    |_, _, _| {},
                )
                .unwrap();
            let image = &drizzled.image;
            assert_eq!(
                (image.width, image.height, image.channels),
                (width, height, 3)
            );
            let mut sky = [Vec::new(), Vec::new(), Vec::new()];
            let mut trail = [Vec::new(), Vec::new(), Vec::new()];
            for y in 8..height - 8 {
                for x in 8..width - 8 {
                    for channel in 0..3 {
                        let value = image.data[(y * width + x) * 3 + channel];
                        assert!(value.is_finite(), "{cfa_integration:?} ({x}, {y})");
                        // Frame 3 moved by (1, 1): its trail lies a row higher on
                        // the reference grid.
                        if y.abs_diff(TRAIL_ROW - 1) <= 1 {
                            trail[channel].push(value);
                        } else {
                            sky[channel].push(value);
                        }
                    }
                }
            }
            for (channel, expected) in [400.0, 300.0, 200.0].into_iter().enumerate() {
                let sky = seiza_stats::median_in_place(&mut sky[channel]).unwrap();
                let trail = seiza_stats::median_in_place(&mut trail[channel]).unwrap();
                assert!(
                    (sky - expected).abs() < 2.0 && (trail - expected).abs() < 20.0,
                    "{cfa_integration:?} channel {channel}: sky {sky}, trail {trail}"
                );
            }
        }
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
    fn retained_bayer_frames_replay_bit_identically() {
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
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
                let mut image = bayer_star_field(width, height, dx, dy);
                let mut state = 0x9e37_79b9_u32.wrapping_mul(index as u32 + 1) | 1;
                for value in &mut image.data {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    *value += (state % 2001) as f32 / 100.0 - 10.0;
                }
                let path = directory.path().join(format!("light-{index}.fits"));
                crate::write_processed_image_fits_f32(&path, &image, &bayer, &[]).unwrap();
                path
            })
            .collect::<Vec<_>>();
        let scratch = directory.path().join("scratch");
        std::fs::create_dir(&scratch).unwrap();
        let stack = |options: &StackOptions, retain: bool| {
            let mut stacker =
                LiveStacker::open_fits(&paths[0], None, None, None, None, options.clone()).unwrap();
            if retain {
                stacker
                    .retain_frames_for_reintegration(Some(&scratch))
                    .unwrap();
            }
            for path in &paths[1..] {
                assert!(matches!(
                    stacker.push_fits(path).unwrap(),
                    FrameDisposition::Accepted(_)
                ));
            }
            stacker
        };
        // Demosaiced, resampled with Lanczos-3, weighted, and refitted to an
        // integrated background, as the command line does.
        let options = StackOptions {
            normalization: NormalizationMode::LocalBackground { tile_size: 32 },
            interpolation: crate::Interpolation::Lanczos3,
            weighting: FrameWeighting::inverse_noise_variance(),
            ..StackOptions::default()
        };
        let retained = stack(&options, true);
        let fresh = stack(&options, false);
        let batch = crate::BatchStackOptions::default();
        let drizzle = crate::DrizzleOptions::default();
        let (integrated, drizzled) = retained
            .reintegrate_drizzled(&batch, &drizzle, |_, _, _| {})
            .unwrap();
        let (expected, expected_drizzled) = fresh
            .reintegrate_drizzled(&batch, &drizzle, |_, _, _| {})
            .unwrap();
        assert_eq!(
            bits(&integrated.snapshot.image.data),
            bits(&expected.snapshot.image.data)
        );
        assert_eq!(
            bits(&integrated.snapshot.variance.data),
            bits(&expected.snapshot.variance.data)
        );
        assert_eq!(integrated.snapshot.coverage, expected.snapshot.coverage);
        assert_eq!(
            integrated.snapshot.rejected_samples,
            expected.snapshot.rejected_samples
        );
        assert_eq!(
            bits(&drizzled.image.data),
            bits(&expected_drizzled.image.data)
        );
        assert_eq!(
            bits(&drizzled.weight.data),
            bits(&expected_drizzled.weight.data)
        );
        drop(retained);
        assert_eq!(std::fs::read_dir(&scratch).unwrap().count(), 0);

        // Bayer drizzle keeps each frame's photosites, and under local
        // background normalization the samples of its interpolated image
        // that the background fit reads, and replays them in bands: to the
        // bit what the whole-frame passes make from the sources, whether the
        // backgrounds are fitted again, the recorded maps are used, or there
        // are none.
        for normalization in [
            NormalizationMode::LocalBackground { tile_size: 32 },
            NormalizationMode::Global,
            NormalizationMode::None,
        ] {
            let photosites = StackOptions {
                cfa_integration: CfaIntegration::BayerDrizzle,
                normalization,
                ..options.clone()
            };
            let retained = stack(&photosites, true);
            let directories = std::fs::read_dir(&scratch)
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .collect::<Vec<_>>();
            assert_eq!(directories.len(), 1);
            let mut kept = std::fs::read_dir(&directories[0])
                .unwrap()
                .map(|entry| entry.unwrap().file_name().into_string().unwrap())
                .collect::<Vec<_>>();
            kept.sort();
            let mut expected_files = (0..shifts.len())
                .flat_map(|index| {
                    let samples =
                        matches!(normalization, NormalizationMode::LocalBackground { .. })
                            .then(|| format!("frame-{index}.samples"));
                    std::iter::once(format!("frame-{index}.f32")).chain(samples)
                })
                .collect::<Vec<_>>();
            expected_files.sort();
            assert_eq!(kept, expected_files, "{normalization:?}");
            let fresh = stack(&photosites, false);
            let (whole, whole_drizzled) = fresh
                .replay_with(&batch, Some(&drizzle), |_, _, _| {}, false)
                .unwrap();
            let whole_drizzled = whole_drizzled.unwrap();
            for stacker in [&retained, &fresh] {
                let mut passes = Vec::new();
                let (replayed, drizzled) = stacker
                    .reintegrate_drizzled(&batch, &drizzle, |pass, _, _| passes.push(pass))
                    .unwrap();
                // The bands ran: no frame was read for a pass of its own.
                assert!(!passes.contains(&crate::BatchStackPass::Refine));
                assert_eq!(
                    bits(&replayed.snapshot.image.data),
                    bits(&whole.snapshot.image.data),
                    "{normalization:?}"
                );
                assert_eq!(
                    bits(&replayed.snapshot.variance.data),
                    bits(&whole.snapshot.variance.data)
                );
                assert_eq!(replayed.snapshot.coverage, whole.snapshot.coverage);
                assert_eq!(
                    replayed.snapshot.rejected_samples,
                    whole.snapshot.rejected_samples
                );
                assert_eq!(bits(&drizzled.image.data), bits(&whole_drizzled.image.data));
                assert_eq!(
                    bits(&drizzled.weight.data),
                    bits(&whole_drizzled.weight.data)
                );
            }
            drop(retained);
            assert_eq!(std::fs::read_dir(&scratch).unwrap().count(), 0);
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

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
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
