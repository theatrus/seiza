use crate::{
    CalibrationMasters, Error, FitsFrame, LinearImage, NormalizationMap, NormalizationMode,
    RegisteredFrameMapping, Registrar, RegistrationOptions, Result, SimilarityTransform, context,
    path_identity, paths_refer_to_same_file, resample_to_reference,
};
use rayon::prelude::*;
use seiza_fits::HeaderValue;
use serde::{Deserialize, Serialize};
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

/// Incremental, bounded-memory image stack. Frames are registered to the
/// immutable first accepted frame and integrated immediately.
pub struct LiveStacker {
    options: StackOptions,
    calibration: CalibrationMasters,
    reference: LinearImage,
    registrar: Registrar,
    accumulator: Accumulator,
    reference_headers: Vec<(String, HeaderValue)>,
    accepted_frames: u32,
    rejected_frames: u32,
    input_paths: Vec<PathBuf>,
}

impl LiveStacker {
    /// Start a stack from a reference FITS frame, calibrating and preparing it
    /// as the immutable alignment target.
    pub fn new(
        mut reference: FitsFrame,
        calibration: CalibrationMasters,
        options: StackOptions,
    ) -> Result<Self> {
        calibration.apply(
            &mut reference.image,
            reference.exposure_seconds,
            reference.bayer,
        )?;
        let reference = reference.into_prepared()?;
        Self::from_prepared(reference.image, reference.headers, calibration, options)
    }

    /// Start a stack from an already-prepared linear reference, with no
    /// calibration and no header metadata.
    pub fn from_linear(reference: LinearImage, options: StackOptions) -> Result<Self> {
        Self::from_prepared(
            reference,
            Vec::new(),
            CalibrationMasters::default(),
            options,
        )
    }

    fn from_prepared(
        reference: LinearImage,
        reference_headers: Vec<(String, HeaderValue)>,
        calibration: CalibrationMasters,
        options: StackOptions,
    ) -> Result<Self> {
        options.validate()?;
        let registrar = Registrar::new(&reference, options.registration.clone())?;
        let mut accumulator = Accumulator::new(reference.sample_count());
        accumulator.integrate(&reference.data, RejectionMode::None);
        Ok(Self {
            options,
            calibration,
            reference,
            registrar,
            accumulator,
            reference_headers,
            accepted_frames: 1,
            rejected_frames: 0,
            input_paths: Vec::new(),
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
        Ok(Self {
            options: restored.options,
            calibration: restored.calibration,
            reference: restored.reference,
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
                mean: &self.accumulator.mean,
                m2: &self.accumulator.m2,
                count: &self.accumulator.count,
                rejected: &self.accumulator.rejected,
                accepted_frames: self.accepted_frames,
                rejected_frames: self.rejected_frames,
                input_paths: &self.input_paths,
            },
        )
    }

    /// Calibrate, prepare, and try to integrate a FITS frame, reporting whether
    /// it was admitted or turned away.
    pub fn push(&mut self, mut frame: FitsFrame) -> Result<FrameDisposition> {
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
    pub fn push_fits(&mut self, path: impl AsRef<Path>) -> Result<FrameDisposition> {
        let path = path.as_ref();
        if self
            .input_paths
            .iter()
            .any(|input| paths_refer_to_same_file(input, path))
        {
            return Err(Error::Stack(format!(
                "FITS frame {} has already been used by this stack",
                path.display()
            )));
        }
        let frame = FitsFrame::open(path)?;
        let disposition = self.push(frame)?;
        self.input_paths.push(path_identity(path));
        Ok(disposition)
    }

    /// Register, normalize, and try to integrate an already-prepared linear
    /// frame, applying every admission gate.
    pub fn push_linear(&mut self, frame: LinearImage) -> Result<FrameDisposition> {
        if self.reference.channels != frame.channels {
            return Ok(self.reject(FrameRejectionReason::IncompatibleImage(format!(
                "frame has {} channel(s) but stack has {}",
                frame.channels, self.reference.channels
            ))));
        }
        let registration = match self.registrar.register(&frame) {
            Ok(registration) => registration,
            Err(error) => {
                let message = match error {
                    Error::Registration(message) => message,
                    other => other.to_string(),
                };
                return Ok(self.reject(FrameRejectionReason::Registration(message)));
            }
        };
        let criteria = self.options.acceptance;
        if registration.rms_error_pixels > criteria.maximum_registration_rms_pixels {
            return Ok(self.reject(FrameRejectionReason::RegistrationRms {
                measured: registration.rms_error_pixels,
                maximum: criteria.maximum_registration_rms_pixels,
            }));
        }
        let scale_deviation = (registration.transform.scale - 1.0).abs();
        if scale_deviation > criteria.maximum_scale_deviation {
            return Ok(self.reject(FrameRejectionReason::ScaleDeviation {
                measured: scale_deviation,
                maximum: criteria.maximum_scale_deviation,
            }));
        }
        let rotation_deviation_degrees =
            rotation_deviation_degrees(registration.transform.rotation_radians);
        if rotation_deviation_degrees > criteria.maximum_rotation_degrees {
            return Ok(self.reject(FrameRejectionReason::Rotation {
                measured_degrees: rotation_deviation_degrees,
                maximum_degrees: criteria.maximum_rotation_degrees,
            }));
        }
        let mut registered = resample_to_reference(
            &frame,
            self.reference.width,
            self.reference.height,
            registration.transform,
        )?;
        let finite_samples = registered
            .data
            .par_iter()
            .filter(|value| value.is_finite())
            .count();
        let overlap_fraction = finite_samples as f32 / registered.sample_count() as f32;
        if overlap_fraction < criteria.minimum_overlap_fraction {
            return Ok(self.reject(FrameRejectionReason::InsufficientOverlap {
                measured: overlap_fraction,
                minimum: criteria.minimum_overlap_fraction,
            }));
        }
        let normalization = match NormalizationMap::estimate(
            &self.reference,
            &registered,
            self.options.normalization,
        ) {
            Ok(normalization) => normalization,
            Err(error) => {
                let message = match error {
                    Error::Normalization(message) => message,
                    other => other.to_string(),
                };
                return Ok(self.reject(FrameRejectionReason::Normalization(message)));
            }
        };
        let (minimum_gain, maximum_gain) = normalization.gain_range();
        if minimum_gain < criteria.minimum_normalization_gain
            || maximum_gain > criteria.maximum_normalization_gain
        {
            return Ok(self.reject(FrameRejectionReason::NormalizationGain {
                measured_minimum: minimum_gain,
                measured_maximum: maximum_gain,
                minimum: criteria.minimum_normalization_gain,
                maximum: criteria.maximum_normalization_gain,
            }));
        }
        if !matches!(self.options.normalization, NormalizationMode::None)
            && let Err(error) = normalization.apply(&mut registered)
        {
            let message = match error {
                Error::Normalization(message) => message,
                other => other.to_string(),
            };
            return Ok(self.reject(FrameRejectionReason::Normalization(message)));
        }
        let normalization_mean_gain = normalization.mean_gain();
        let normalization_mean_offset = normalization.mean_offset();
        let mapping = crate::RegisteredFrameMapping::new(
            self.reference.width,
            self.reference.height,
            registration.transform,
            normalization,
        )?;
        let (would_accept, _) = self
            .accumulator
            .classify(&registered.data, self.options.rejection);
        let integrated_fraction = would_accept as f32 / registered.sample_count() as f32;
        if integrated_fraction < criteria.minimum_integrated_fraction {
            return Ok(
                self.reject(FrameRejectionReason::InsufficientIntegratedSamples {
                    measured: integrated_fraction,
                    minimum: criteria.minimum_integrated_fraction,
                }),
            );
        }
        let (accepted_samples, rejected_samples) = self
            .accumulator
            .integrate(&registered.data, self.options.rejection);
        self.accepted_frames += 1;
        Ok(FrameDisposition::Accepted(FrameDiagnostics {
            transform: registration.transform,
            matched_stars: registration.matched_stars,
            registration_rms_pixels: registration.rms_error_pixels,
            registration_drift_pixels: registration.drift_pixels,
            normalization_mean_gain,
            normalization_mean_offset,
            mapping: Box::new(mapping),
            overlap_fraction,
            integrated_fraction,
            accepted_samples,
            rejected_samples,
        }))
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

    /// Source and calibration paths already used by this stack.
    pub fn input_paths(&self) -> &[PathBuf] {
        &self.input_paths
    }

    fn reject(&mut self, reason: FrameRejectionReason) -> FrameDisposition {
        self.rejected_frames += 1;
        FrameDisposition::Rejected(reason)
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
        let mean = self
            .mean
            .iter()
            .zip(&self.count)
            .map(|(&mean, &count)| finalized_mean(mean, count))
            .collect();
        let variance = self
            .m2
            .iter()
            .zip(&self.count)
            .map(|(&m2, &count)| finalized_variance(m2, count))
            .collect();
        (mean, variance)
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
    fn context_preserves_calibration_headers_and_source_ledger() {
        let reference = stacking_star_field(160, 128);
        let calibration_image = LinearImage::new(160, 128, 1, vec![2.0; 160 * 128]).unwrap();
        let mut stacker = LiveStacker::from_linear(reference, StackOptions::default()).unwrap();
        stacker.calibration = CalibrationMasters {
            bias: Some(calibration_image.clone()),
            dark_signal: Some(calibration_image.clone()),
            dark_exposure_seconds: Some(300.0),
            dark_bayer: Some(BayerLayout {
                pattern: seiza_fits::BayerPattern::Rggb,
                x_offset: 1,
                y_offset: 0,
            }),
            flat_response: Some(calibration_image),
            flat_bayer: Some(BayerLayout {
                pattern: seiza_fits::BayerPattern::Rggb,
                x_offset: 1,
                y_offset: 0,
            }),
        };
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
}
