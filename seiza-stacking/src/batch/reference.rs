//! The three-pass integration as it stood before the passes were reworked,
//! kept verbatim as a test reference: every faster path must reproduce it
//! bit for bit. Frames come from memory, so the loader plumbing and the
//! change check are left out; neither touches a sample.

use super::{BatchFrameDiagnostics, BatchStackOptions, BatchStackResult, SampleFate};
use crate::{LinearImage, StackSnapshot};
use rayon::prelude::*;
use statrs::distribution::{Continuous, ContinuousCDF, Normal, StudentsT};

/// The reference result, and every frame's sample fates from the final pass.
pub(crate) fn integrate(
    frames: &[LinearImage],
    options: &BatchStackOptions,
) -> (BatchStackResult, Vec<Vec<SampleFate>>) {
    if options.frame_weights.is_some() {
        return integrate_weighted_frames(frames, options);
    }
    let frame_count = frames.len();
    let rejection = Rejection::new(frame_count, options);
    let mut fates_by_frame = Vec::new();

    let mut shape = None;
    let mut first = Vec::<FirstEstimate>::new();
    for image in frames {
        if shape.is_none() {
            shape = Some((image.width, image.height, image.channels));
            first.resize(image.sample_count(), FirstEstimate::default());
        }
        first
            .par_iter_mut()
            .zip(image.data.par_iter())
            .for_each(|(first, &sample)| {
                if sample.is_finite() {
                    first.take(sample, 1.0);
                }
            });
    }
    first
        .par_iter_mut()
        .for_each(|first| first.finish(&rejection));

    let mut kept = vec![Moments::default(); first.len()];
    for image in frames {
        kept.par_iter_mut()
            .zip(first.par_iter())
            .zip(image.data.par_iter())
            .for_each(|((kept, first), &sample)| {
                if sample.is_finite() && first.keeps(sample, 1.0, &rejection) {
                    *kept = kept.with(sample, 1.0);
                }
            });
    }

    let mut integrated = vec![0.0_f32; first.len()];
    let mut variance = vec![0.0_f32; first.len()];
    let mut coverage = vec![0_u32; first.len()];
    let mut rejected = vec![0_u32; first.len()];
    let mut frame_diagnostics = Vec::with_capacity(frame_count);
    for image in frames {
        let fates = integrated
            .par_iter_mut()
            .zip(variance.par_iter_mut())
            .zip(coverage.par_iter_mut())
            .zip(rejected.par_iter_mut())
            .enumerate()
            .map(
                |(sample_index, (((integrated, variance), coverage), rejected))| {
                    let sample = image.data[sample_index];
                    if !sample.is_finite() {
                        return SampleFate::Missing;
                    }
                    if kept_peers(
                        &first[sample_index],
                        kept[sample_index],
                        sample,
                        1.0,
                        &rejection,
                    )
                    .rejects(sample, 1.0, &rejection)
                    {
                        *rejected += 1;
                        return SampleFate::Rejected;
                    }
                    *coverage += 1;
                    let delta = f64::from(sample) - f64::from(*integrated);
                    let next_mean = f64::from(*integrated) + delta / f64::from(*coverage);
                    *variance =
                        (f64::from(*variance) + delta * (f64::from(sample) - next_mean)) as f32;
                    *integrated = next_mean as f32;
                    SampleFate::Integrated
                },
            )
            .collect::<Vec<_>>();
        frame_diagnostics.push(diagnostics(&fates));
        fates_by_frame.push(fates);
    }
    finish(
        shape,
        frame_count,
        integrated,
        variance,
        coverage,
        rejected,
        frame_diagnostics,
        fates_by_frame,
    )
}

fn integrate_weighted_frames(
    frames: &[LinearImage],
    options: &BatchStackOptions,
) -> (BatchStackResult, Vec<Vec<SampleFate>>) {
    let frame_weights = options
        .frame_weights
        .as_deref()
        .expect("weighted integration needs frame weights");
    let frame_count = frames.len();
    let rejection = Rejection::new(frame_count, options);
    let weight_at = |index: usize, sample_index: usize, channels: usize| {
        f64::from(frame_weights[index][sample_index % channels])
    };
    let mut fates_by_frame = Vec::new();

    let mut shape = None;
    let mut first = Vec::<FirstEstimate>::new();
    for (index, image) in frames.iter().enumerate() {
        if shape.is_none() {
            shape = Some((image.width, image.height, image.channels));
            first.resize(image.sample_count(), FirstEstimate::default());
        }
        let channels = image.channels;
        first
            .par_iter_mut()
            .zip(image.data.par_iter())
            .enumerate()
            .for_each(|(sample_index, (first, &sample))| {
                if sample.is_finite() {
                    first.take(sample, weight_at(index, sample_index, channels));
                }
            });
    }
    first
        .par_iter_mut()
        .for_each(|first| first.finish(&rejection));

    let mut kept = vec![Moments::default(); first.len()];
    for (index, image) in frames.iter().enumerate() {
        let channels = image.channels;
        kept.par_iter_mut()
            .zip(first.par_iter())
            .zip(image.data.par_iter())
            .enumerate()
            .for_each(|(sample_index, ((kept, first), &sample))| {
                let weight = weight_at(index, sample_index, channels);
                if sample.is_finite() && first.keeps(sample, weight, &rejection) {
                    *kept = kept.with(sample, weight);
                }
            });
    }

    let mut integrated = vec![0.0_f32; first.len()];
    let mut variance = vec![0.0_f32; first.len()];
    let mut integrated_weight = vec![0.0_f32; first.len()];
    let mut coverage = vec![0_u32; first.len()];
    let mut rejected = vec![0_u32; first.len()];
    let mut frame_diagnostics = Vec::with_capacity(frame_count);
    for (index, image) in frames.iter().enumerate() {
        let channels = image.channels;
        let fates = integrated
            .par_iter_mut()
            .zip(variance.par_iter_mut())
            .zip(integrated_weight.par_iter_mut())
            .zip(coverage.par_iter_mut())
            .zip(rejected.par_iter_mut())
            .enumerate()
            .map(
                |(
                    sample_index,
                    ((((integrated, variance), integrated_weight), coverage), rejected),
                )| {
                    let sample = image.data[sample_index];
                    if !sample.is_finite() {
                        return SampleFate::Missing;
                    }
                    let weight = weight_at(index, sample_index, channels);
                    if kept_peers(
                        &first[sample_index],
                        kept[sample_index],
                        sample,
                        weight,
                        &rejection,
                    )
                    .rejects(sample, weight, &rejection)
                    {
                        *rejected += 1;
                        return SampleFate::Rejected;
                    }
                    *coverage += 1;
                    let next_weight = f64::from(*integrated_weight) + weight;
                    let weighted_delta = (f64::from(sample) - f64::from(*integrated)) * weight;
                    let next_mean = f64::from(*integrated) + weighted_delta / next_weight;
                    *variance = (f64::from(*variance)
                        + weighted_delta * (f64::from(sample) - next_mean))
                        as f32;
                    *integrated = next_mean as f32;
                    *integrated_weight = next_weight as f32;
                    SampleFate::Integrated
                },
            )
            .collect::<Vec<_>>();
        frame_diagnostics.push(diagnostics(&fates));
        fates_by_frame.push(fates);
    }
    finish(
        shape,
        frame_count,
        integrated,
        variance,
        coverage,
        rejected,
        frame_diagnostics,
        fates_by_frame,
    )
}

fn diagnostics(fates: &[SampleFate]) -> BatchFrameDiagnostics {
    BatchFrameDiagnostics {
        finite_samples: fates
            .iter()
            .filter(|&&fate| fate != SampleFate::Missing)
            .count(),
        integrated_samples: fates
            .iter()
            .filter(|&&fate| fate == SampleFate::Integrated)
            .count(),
    }
}

#[allow(clippy::too_many_arguments)]
fn finish(
    shape: Option<(usize, usize, usize)>,
    frame_count: usize,
    mut integrated: Vec<f32>,
    mut variance: Vec<f32>,
    coverage: Vec<u32>,
    rejected: Vec<u32>,
    frames: Vec<BatchFrameDiagnostics>,
    fates: Vec<Vec<SampleFate>>,
) -> (BatchStackResult, Vec<Vec<SampleFate>>) {
    for ((integrated, variance), &coverage) in
        integrated.iter_mut().zip(&mut variance).zip(&coverage)
    {
        if coverage == 0 {
            *integrated = f32::NAN;
        }
        *variance = if coverage > 1 {
            (*variance / (coverage - 1) as f32).max(0.0)
        } else {
            0.0
        };
    }
    let (width, height, channels) = shape.expect("at least one frame");
    (
        BatchStackResult {
            snapshot: StackSnapshot {
                image: LinearImage::new(width, height, channels, integrated).unwrap(),
                variance: LinearImage::new(width, height, channels, variance).unwrap(),
                coverage,
                rejected_samples: rejected,
                accepted_frames: frame_count as u32,
                rejected_frames: 0,
            },
            frames,
        },
        fates,
    )
}

fn rejection_thresholds(frame_count: usize, options: &BatchStackOptions) -> Vec<(f64, f64)> {
    let normal = Normal::new(0.0, 1.0).expect("valid standard normal parameters");
    let low_tail = normal.sf(f64::from(options.rejection.low_sigma));
    let high_tail = normal.sf(f64::from(options.rejection.high_sigma));
    (0..=frame_count)
        .map(|count| {
            if count < 3 {
                return (f64::INFINITY, f64::INFINITY);
            }
            let peers = (count - 1) as f64;
            let predictive = StudentsT::new(0.0, (1.0 + 1.0 / peers).sqrt(), peers - 1.0)
                .expect("at least two peers give valid predictive parameters");
            (
                -predictive.inverse_cdf(low_tail),
                -predictive.inverse_cdf(high_tail),
            )
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Default)]
struct Moments {
    mean: f32,
    m2: f32,
    weight_sum: f32,
    count: u32,
}

impl Moments {
    fn with(self, value: f32, weight: f64) -> Self {
        let (mean, m2) = (f64::from(self.mean), f64::from(self.m2));
        let weight_sum = f64::from(self.weight_sum) + weight;
        let value = f64::from(value);
        let weighted_delta = (value - mean) * weight;
        let next_mean = mean + weighted_delta / weight_sum;
        Self {
            mean: next_mean as f32,
            m2: (m2 + weighted_delta * (value - next_mean)) as f32,
            weight_sum: weight_sum as f32,
            count: self.count + 1,
        }
    }

    fn without(self, value: f32, weight: f64) -> Self {
        let weight_sum = f64::from(self.weight_sum) - weight;
        if self.count <= 1 || weight_sum.is_nan() || weight_sum <= 0.0 {
            return Self::default();
        }
        let (mean, m2) = (f64::from(self.mean), f64::from(self.m2));
        let value = f64::from(value);
        let other_mean = mean + weight * (mean - value) / weight_sum;
        Self {
            mean: other_mean as f32,
            m2: (m2 - weight * (value - mean) * (value - other_mean)).max(0.0) as f32,
            weight_sum: weight_sum as f32,
            count: self.count - 1,
        }
    }

    fn scaled(self, factor: f64) -> Self {
        Self {
            m2: (f64::from(self.m2) * factor) as f32,
            ..self
        }
    }

    fn rejects(self, value: f32, weight: f64, rejection: &Rejection) -> bool {
        let weight_sum = f64::from(self.weight_sum);
        if self.count < 2 || weight_sum.is_nan() || weight_sum <= 0.0 {
            return false;
        }
        let peers = f64::from(self.count);
        let mean = f64::from(self.mean);
        let scale = if weight == 1.0 && weight_sum == peers {
            1.0
        } else {
            ((1.0 / weight + 1.0 / weight_sum) / (1.0 + 1.0 / peers)).sqrt()
        };
        let sigma = ((f64::from(self.m2) / (peers - 1.0)).sqrt() * scale)
            .max(f64::from(rejection.minimum_sigma))
            .max(mean.abs() * PRECISION_STEPS * f64::from(f32::EPSILON));
        let (low, high) = rejection.thresholds[self.count as usize + 1];
        let residual = f64::from(value) - mean;
        residual < -low * sigma || residual > high * sigma
    }
}

const PRECISION_STEPS: f64 = 8.0;

struct Rejection {
    thresholds: Vec<(f64, f64)>,
    minimum_sigma: f32,
    expected_extreme_square: Vec<f64>,
    clipped_variance: f64,
}

impl Rejection {
    fn new(frame_count: usize, options: &BatchStackOptions) -> Self {
        let normal = Normal::new(0.0, 1.0).expect("valid standard normal parameters");
        let (low, high) = (
            f64::from(options.rejection.low_sigma),
            f64::from(options.rejection.high_sigma),
        );
        let kept = normal.cdf(high) - normal.cdf(-low);
        let (density_low, density_high) = (normal.pdf(low), normal.pdf(high));
        let clipped_variance = 1.0
            - (low * density_low + high * density_high) / kept
            - ((density_low - density_high) / kept).powi(2);
        const STEPS: usize = 1600;
        let (from, to) = (-8.0_f64, 8.0_f64);
        let step = (to - from) / STEPS as f64;
        let grid = (0..=STEPS)
            .map(|index| {
                let z = from + index as f64 * step;
                (z, normal.pdf(z), normal.cdf(z))
            })
            .collect::<Vec<_>>();
        let expected_extreme_square = (0..=frame_count)
            .map(|count| {
                if (count as u32) < TRIMMED_FIRST_ESTIMATE {
                    return 0.0;
                }
                let n = count as f64;
                grid.iter()
                    .map(|&(z, density, cumulative)| {
                        z * z * n * density * cumulative.powf(n - 1.0) * step
                    })
                    .sum()
            })
            .collect();
        Self {
            thresholds: rejection_thresholds(frame_count, options),
            minimum_sigma: options.minimum_sigma,
            expected_extreme_square,
            clipped_variance: clipped_variance.clamp(0.05, 1.0),
        }
    }

    fn trim_factor(&self, count: u32, without_candidate: bool) -> f64 {
        let n = f64::from(count);
        let extremes = 2.0 * self.expected_extreme_square[count as usize];
        let (degrees, expected) = if without_candidate {
            (n - 4.0, n - 2.0 - extremes)
        } else {
            (n - 3.0, n - 1.0 - extremes)
        };
        if expected > 0.0 {
            degrees / expected
        } else {
            1.0
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct FirstEstimate {
    moments: Moments,
    high: f32,
    low: f32,
    high_weight_or_member_factor: f32,
    low_weight_or_extreme_factor: f32,
}

impl Default for FirstEstimate {
    fn default() -> Self {
        Self {
            moments: Moments::default(),
            high: f32::NEG_INFINITY,
            low: f32::INFINITY,
            high_weight_or_member_factor: 0.0,
            low_weight_or_extreme_factor: 0.0,
        }
    }
}

impl FirstEstimate {
    fn take(&mut self, value: f32, weight: f64) {
        self.moments = self.moments.with(value, weight);
        if value > self.high {
            self.high = value;
            self.high_weight_or_member_factor = weight as f32;
        }
        if value < self.low {
            self.low = value;
            self.low_weight_or_extreme_factor = weight as f32;
        }
    }

    fn finish(&mut self, rejection: &Rejection) {
        let count = self.moments.count;
        if count < TRIMMED_FIRST_ESTIMATE {
            self.high = f32::NAN;
            self.low = f32::NAN;
            self.high_weight_or_member_factor = 1.0;
            self.low_weight_or_extreme_factor = 1.0;
            return;
        }
        self.moments = self
            .moments
            .without(self.high, f64::from(self.high_weight_or_member_factor))
            .without(self.low, f64::from(self.low_weight_or_extreme_factor));
        self.high_weight_or_member_factor = rejection.trim_factor(count, true) as f32;
        self.low_weight_or_extreme_factor = rejection.trim_factor(count, false) as f32;
    }

    fn peers(&self, value: f32, weight: f64) -> Moments {
        if value == self.high || value == self.low {
            self.moments
                .scaled(f64::from(self.low_weight_or_extreme_factor))
        } else {
            self.moments
                .without(value, weight)
                .scaled(f64::from(self.high_weight_or_member_factor))
        }
    }

    fn keeps(&self, value: f32, weight: f64, rejection: &Rejection) -> bool {
        !self.peers(value, weight).rejects(value, weight, rejection)
    }
}

const TRIMMED_FIRST_ESTIMATE: u32 = 10;

fn kept_peers(
    first: &FirstEstimate,
    kept: Moments,
    value: f32,
    weight: f64,
    rejection: &Rejection,
) -> Moments {
    let peers = if first.keeps(value, weight, rejection) {
        kept.without(value, weight)
    } else {
        kept
    };
    peers.scaled(1.0 / rejection.clipped_variance)
}
