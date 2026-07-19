use crate::{
    CalibrationMasters, Error, FitsFrame, LinearImage, NormalizationMap, NormalizationMode,
    Registrar, RegistrationOptions, Result, SimilarityTransform,
};
use rayon::prelude::*;
use seiza_fits::HeaderValue;

#[derive(Clone, Copy, Debug)]
pub struct DeltaSigmaOptions {
    pub low_sigma: f32,
    pub high_sigma: f32,
    pub warmup_samples: u32,
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

#[derive(Clone, Copy, Debug)]
pub enum RejectionMode {
    None,
    DeltaSigma(DeltaSigmaOptions),
}

impl Default for RejectionMode {
    fn default() -> Self {
        Self::DeltaSigma(DeltaSigmaOptions::default())
    }
}

#[derive(Clone, Debug, Default)]
pub struct StackOptions {
    pub registration: RegistrationOptions,
    pub normalization: NormalizationMode,
    pub rejection: RejectionMode,
    pub acceptance: FrameAcceptanceCriteria,
}

impl StackOptions {
    fn validate(&self) -> Result<()> {
        let registration = &self.registration;
        if !registration.detection_sigma.is_finite()
            || registration.detection_sigma <= 0.0
            || registration.triangle_stars < 3
            || registration.maximum_stars < registration.minimum_matches.max(3)
            || registration.minimum_matches < 3
            || registration.maximum_candidates == 0
            || !registration.descriptor_tolerance.is_finite()
            || registration.descriptor_tolerance <= 0.0
            || !registration.scale_tolerance.is_finite()
            || !(0.0..1.0).contains(&registration.scale_tolerance)
            || !registration.match_tolerance_pixels.is_finite()
            || registration.match_tolerance_pixels <= 0.0
        {
            return Err(Error::Stack("invalid registration options".into()));
        }
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
#[derive(Clone, Copy, Debug)]
pub struct FrameAcceptanceCriteria {
    pub maximum_registration_rms_pixels: f64,
    pub maximum_scale_deviation: f64,
    pub maximum_rotation_degrees: f64,
    pub minimum_overlap_fraction: f32,
    pub minimum_normalization_gain: f32,
    pub maximum_normalization_gain: f32,
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

#[derive(Clone, Debug)]
pub struct FrameDiagnostics {
    pub transform: SimilarityTransform,
    pub matched_stars: usize,
    pub registration_rms_pixels: f64,
    pub normalization_mean_gain: f32,
    pub normalization_mean_offset: f32,
    pub overlap_fraction: f32,
    pub integrated_fraction: f32,
    pub accepted_samples: usize,
    pub rejected_samples: usize,
}

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum FrameRejectionReason {
    #[error("incompatible image: {0}")]
    IncompatibleImage(String),
    #[error("registration failed: {0}")]
    Registration(String),
    #[error("registration RMS {measured:.3}px exceeds {maximum:.3}px")]
    RegistrationRms { measured: f64, maximum: f64 },
    #[error("scale deviation {measured:.5} exceeds {maximum:.5}")]
    ScaleDeviation { measured: f64, maximum: f64 },
    #[error("rotation {measured_degrees:.3}deg exceeds {maximum_degrees:.3}deg")]
    Rotation {
        measured_degrees: f64,
        maximum_degrees: f64,
    },
    #[error("overlap fraction {measured:.3} is below {minimum:.3}")]
    InsufficientOverlap { measured: f32, minimum: f32 },
    #[error("normalization gain {measured:.3} is outside {minimum:.3}..={maximum:.3}")]
    NormalizationGain {
        measured: f32,
        minimum: f32,
        maximum: f32,
    },
    #[error("integrated sample fraction {measured:.3} is below {minimum:.3}")]
    InsufficientIntegratedSamples { measured: f32, minimum: f32 },
}

#[derive(Clone, Debug)]
pub enum FrameDisposition {
    Accepted(FrameDiagnostics),
    Rejected(FrameRejectionReason),
}

#[derive(Clone, Debug)]
pub struct StackSnapshot {
    pub image: LinearImage,
    pub variance: LinearImage,
    /// Accepted observation count for every image sample.
    pub coverage: Vec<u32>,
    /// Rejected observation count for every image sample.
    pub rejected_samples: Vec<u32>,
    pub accepted_frames: u32,
    pub rejected_frames: u32,
}

/// Zero-copy access to the current online estimate. Samples with zero
/// coverage have an undefined mean and must be masked by `coverage`.
#[derive(Clone, Copy, Debug)]
pub struct StackView<'a> {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub mean: &'a [f32],
    pub coverage: &'a [u32],
    pub rejected_samples: &'a [u32],
    pub accepted_frames: u32,
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
}

impl LiveStacker {
    pub fn new(
        mut reference: FitsFrame,
        calibration: CalibrationMasters,
        options: StackOptions,
    ) -> Result<Self> {
        calibration.apply(&mut reference.image, reference.exposure_seconds)?;
        let reference = reference.into_prepared()?;
        Self::from_prepared(reference.image, reference.headers, calibration, options)
    }

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
        })
    }

    pub fn push(&mut self, mut frame: FitsFrame) -> Result<FrameDisposition> {
        self.calibration
            .apply(&mut frame.image, frame.exposure_seconds)?;
        let frame = frame.into_prepared()?;
        self.push_linear(frame.image)
    }

    pub fn push_linear(&mut self, frame: LinearImage) -> Result<FrameDisposition> {
        if !self.reference.dimensions_match(&frame) {
            self.rejected_frames += 1;
            return Ok(FrameDisposition::Rejected(
                FrameRejectionReason::IncompatibleImage(format!(
                    "frame is {}x{}x{} but stack is {}x{}x{}",
                    frame.width,
                    frame.height,
                    frame.channels,
                    self.reference.width,
                    self.reference.height,
                    self.reference.channels
                )),
            ));
        }
        let registration = match self.registrar.register(&frame) {
            Ok(registration) => registration,
            Err(error) => {
                self.rejected_frames += 1;
                let message = match error {
                    Error::Registration(message) => message,
                    other => other.to_string(),
                };
                return Ok(FrameDisposition::Rejected(
                    FrameRejectionReason::Registration(message),
                ));
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
        let rotation_degrees = registration.transform.rotation_radians.to_degrees().abs();
        if rotation_degrees > criteria.maximum_rotation_degrees {
            return Ok(self.reject(FrameRejectionReason::Rotation {
                measured_degrees: rotation_degrees,
                maximum_degrees: criteria.maximum_rotation_degrees,
            }));
        }
        let mut registered = resample(
            &frame,
            self.reference.width,
            self.reference.height,
            registration.transform,
        );
        let finite_samples = registered
            .data
            .iter()
            .filter(|value| value.is_finite())
            .count();
        let overlap_fraction = finite_samples as f32 / registered.sample_count() as f32;
        if overlap_fraction < criteria.minimum_overlap_fraction {
            return Ok(self.reject(FrameRejectionReason::InsufficientOverlap {
                measured: overlap_fraction,
                minimum: criteria.minimum_overlap_fraction,
            }));
        }
        let normalization =
            NormalizationMap::estimate(&self.reference, &registered, self.options.normalization)?;
        let normalization_gain = normalization.mean_gain();
        if normalization_gain < criteria.minimum_normalization_gain
            || normalization_gain > criteria.maximum_normalization_gain
        {
            return Ok(self.reject(FrameRejectionReason::NormalizationGain {
                measured: normalization_gain,
                minimum: criteria.minimum_normalization_gain,
                maximum: criteria.maximum_normalization_gain,
            }));
        }
        normalization.apply(&mut registered)?;
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
            normalization_mean_gain: normalization.mean_gain(),
            normalization_mean_offset: normalization.mean_offset(),
            overlap_fraction,
            integrated_fraction,
            accepted_samples,
            rejected_samples,
        }))
    }

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

    pub fn reference_headers(&self) -> &[(String, HeaderValue)] {
        &self.reference_headers
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
        let mut accepted = 0;
        let mut rejected = 0;
        for (index, &sample) in samples.iter().enumerate() {
            if !sample.is_finite() {
                continue;
            }
            let should_reject = self.should_reject(index, sample, rejection);
            if should_reject {
                self.rejected[index] = self.rejected[index].saturating_add(1);
                rejected += 1;
                continue;
            }
            let count = self.count[index].saturating_add(1);
            let delta = sample - self.mean[index];
            self.mean[index] += delta / count as f32;
            let delta_after = sample - self.mean[index];
            self.m2[index] += delta * delta_after;
            self.count[index] = count;
            accepted += 1;
        }
        (accepted, rejected)
    }

    fn classify(&self, samples: &[f32], rejection: RejectionMode) -> (usize, usize) {
        let mut accepted = 0;
        let mut rejected = 0;
        for (index, &sample) in samples.iter().enumerate() {
            if !sample.is_finite() {
                continue;
            }
            if self.should_reject(index, sample, rejection) {
                rejected += 1;
            } else {
                accepted += 1;
            }
        }
        (accepted, rejected)
    }

    fn should_reject(&self, index: usize, sample: f32, rejection: RejectionMode) -> bool {
        match rejection {
            RejectionMode::None => false,
            RejectionMode::DeltaSigma(options)
                if self.count[index] >= options.warmup_samples && self.count[index] > 1 =>
            {
                let sigma = (self.m2[index] / (self.count[index] - 1) as f32)
                    .sqrt()
                    .max(options.minimum_sigma);
                let delta = sample - self.mean[index];
                delta < -options.low_sigma * sigma || delta > options.high_sigma * sigma
            }
            RejectionMode::DeltaSigma(_) => false,
        }
    }

    fn snapshot(&self) -> (Vec<f32>, Vec<f32>) {
        let mean = self
            .mean
            .iter()
            .zip(&self.count)
            .map(|(mean, count)| if *count == 0 { f32::NAN } else { *mean })
            .collect();
        let variance = self
            .m2
            .iter()
            .zip(&self.count)
            .map(|(m2, count)| {
                if *count > 1 {
                    *m2 / (*count - 1) as f32
                } else {
                    0.0
                }
            })
            .collect();
        (mean, variance)
    }

    fn into_snapshot(mut self) -> (Vec<f32>, Vec<f32>, Vec<u32>, Vec<u32>) {
        for (mean, count) in self.mean.iter_mut().zip(&self.count) {
            if *count == 0 {
                *mean = f32::NAN;
            }
        }
        for (m2, count) in self.m2.iter_mut().zip(&self.count) {
            *m2 = if *count > 1 {
                *m2 / (*count - 1) as f32
            } else {
                0.0
            };
        }
        (self.mean, self.m2, self.count, self.rejected)
    }
}

fn resample(
    source: &LinearImage,
    width: usize,
    height: usize,
    transform: SimilarityTransform,
) -> LinearImage {
    let channels = source.channels;
    let mut data = vec![f32::NAN; width * height * channels];
    data.par_chunks_mut(channels)
        .enumerate()
        .for_each(|(pixel, output)| {
            let x = pixel % width;
            let y = pixel / width;
            let (source_x, source_y) = transform.inverse_apply(x as f64, y as f64);
            if source_x < 0.0
                || source_y < 0.0
                || source_x >= (source.width - 1) as f64
                || source_y >= (source.height - 1) as f64
            {
                return;
            }
            let x0 = source_x.floor() as usize;
            let y0 = source_y.floor() as usize;
            let tx = (source_x - x0 as f64) as f32;
            let ty = (source_y - y0 as f64) as f32;
            for (channel, output_sample) in output.iter_mut().enumerate() {
                let sample =
                    |x: usize, y: usize| source.data[(y * source.width + x) * channels + channel];
                let values = [
                    sample(x0, y0),
                    sample(x0 + 1, y0),
                    sample(x0, y0 + 1),
                    sample(x0 + 1, y0 + 1),
                ];
                if values.iter().all(|value| value.is_finite()) {
                    let top = values[0] * (1.0 - tx) + values[1] * tx;
                    let bottom = values[2] * (1.0 - tx) + values[3] * tx;
                    *output_sample = top * (1.0 - ty) + bottom * ty;
                }
            }
        });
    LinearImage {
        width,
        height,
        channels,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn bilinear_resampling_uses_inverse_transform() {
        let source =
            LinearImage::new(4, 4, 1, (0..16).map(|value| value as f32).collect()).unwrap();
        let registered = resample(
            &source,
            4,
            4,
            SimilarityTransform {
                translation_x: 1.0,
                ..SimilarityTransform::IDENTITY
            },
        );
        assert_eq!(registered.data[1], source.data[0]);
        assert!(registered.data[0].is_nan());
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
}
