use crate::{CancelSignal, Error, LinearImage, MasterRejectionOptions, Result, StackSnapshot};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use statrs::distribution::{Continuous, ContinuousCDF, Normal, StudentsT};

/// Rejection options for a completed stack of registered, normalized frames.
#[derive(Clone, Debug)]
pub struct BatchStackOptions {
    /// Nominal Gaussian low and high sigma thresholds. Student-t prediction
    /// limits preserve their tail probabilities when few peers estimate noise.
    pub rejection: MasterRejectionOptions,
    /// Noise floor in the physical sample units of the registered frames.
    pub minimum_sigma: f32,
    /// Optional cancellation checked before and after each frame load.
    pub cancel: Option<CancelSignal>,
    /// Optional per-frame, per-channel weights, in loader index order.
    ///
    /// `None` (the default) weighs every frame equally and gives exactly the
    /// result of earlier releases. `Some` must hold one entry per frame, each
    /// with one finite, positive weight per image channel. Pass the weights a
    /// live stack recorded in [`crate::FrameDiagnostics::weight`] (1 for the
    /// reference frame), so replay does not measure noise again. Weights are
    /// relative to a frame of weight 1: `minimum_sigma` and the variance
    /// output then describe such a frame.
    pub frame_weights: Option<Vec<Vec<f32>>>,
    /// Where [`crate::LiveStacker::reintegrate`] keeps each frame's prepared
    /// image between passes, or `None` for the system temporary directory.
    /// It needs about four bytes per output sample per admitted frame (29 GB
    /// for 92 frames of a 26 MP colour sensor), so choose a disk with room:
    /// where `/tmp` is held in memory, the system default may be too small.
    /// Without room, reintegration prepares each frame again on every pass.
    /// A stack told to [`crate::LiveStacker::retain_frames_for_reintegration`]
    /// keeps its frames in the directory given there instead.
    pub scratch_directory: Option<std::path::PathBuf>,
}

impl Default for BatchStackOptions {
    fn default() -> Self {
        Self {
            rejection: MasterRejectionOptions::default(),
            minimum_sigma: 1.0e-6,
            cancel: None,
            frame_weights: None,
            scratch_directory: None,
        }
    }
}

/// Which of the three sequential reads is requesting an input frame, in the
/// order they run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchStackPass {
    /// Estimate moments using every finite registered sample.
    Estimate,
    /// Reread frames and estimate moments again from only the samples the
    /// first estimate keeps.
    Refine,
    /// Reread frames and accumulate only samples which survive rejection.
    Integrate,
}

/// Per-frame sample decisions from the completed rejection pass.
#[derive(Clone, Debug)]
pub struct BatchFrameDiagnostics {
    /// Finite registered samples before rejection, excluding missing borders.
    pub finite_samples: usize,
    /// Samples which contributed to the final mean.
    pub integrated_samples: usize,
}

/// Completed image and per-frame decisions in loader index order.
#[derive(Clone, Debug)]
pub struct BatchStackResult {
    /// Final clipped mean, sample variance, and per-sample coverage/rejections.
    pub snapshot: StackSnapshot,
    /// Sample statistics for each supplied frame.
    pub frames: Vec<BatchFrameDiagnostics>,
}

/// Integrate already-registered frames with three-pass leave-one-out
/// rejection.
///
/// Unlike live delta-sigma rejection, this revisits the reference and warm-up
/// frames, so an isolated early transient cannot remain in the final average.
/// Pixels with fewer than three finite observations are averaged without
/// rejection. A Student-t prediction limit accounts for uncertain noise
/// estimates at low depth while preserving the configured Gaussian tail
/// probabilities.
///
/// One large outlier inflates the dispersion a pixel's other samples are
/// judged by, which would let moderate outliers at that pixel through: a
/// bright satellite trail in one frame would leave a line of the hazier
/// frames' samples behind it. So the first pass estimates moments from every
/// sample, the second estimates them again from only the samples the first
/// keeps, and the third rejects each sample against the second, leaving the
/// sample's own contribution out when the second counted it. The first
/// estimate also leaves out each pixel's largest and smallest sample once it
/// has ten, so a single trail cannot set the scale the second is clipped by.
/// Both estimates are corrected for the variance that trimming and clipping
/// take from Gaussian noise, so plain noise is rejected at the configured
/// rate.
/// Three or more large outliers at one pixel can still mask one another; this
/// is not a median/MAD estimator.
///
/// The loader must return the same calibrated, registered, normalized image
/// for an index on every pass. Shapes and sample digests are checked before
/// accumulation. The caller owns registration, whole-frame admission, and
/// source provenance; no registration, normalization, or frame-level quality
/// decisions are repeated here. All supplied frames count as admitted.
///
/// Memory is proportional to the output dimensions, not the number of frames:
/// approximately 64 bytes per sample (68 with frame weights) plus one loaded
/// input and small per-frame digests. Drop any online accumulator before
/// calling this when memory is tight.
///
/// With [`BatchStackOptions::frame_weights`] every pass uses West's weighted
/// mean and variance. A candidate is compared to the weighted mean of the
/// other frames, and its allowed deviation grows as `sqrt(1/w + 1/W_o)`,
/// where `w` is its weight and `W_o` the others' total weight. With every
/// weight 1 the result is bit-identical to the unweighted one.
pub fn integrate_registered_frames(
    frame_count: usize,
    options: &BatchStackOptions,
    load: impl FnMut(BatchStackPass, usize) -> Result<LinearImage>,
) -> Result<BatchStackResult> {
    integrate_registered_frames_observed(frame_count, options, load, None)
}

/// What the final pass did with one registered sample.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SampleFate {
    /// Not finite: the frame does not cover it.
    Missing,
    /// Finite and rejected.
    Rejected,
    /// Finite and averaged in.
    Integrated,
}

impl SampleFate {
    /// Finite and integrated sample counts.
    fn counts(self) -> (usize, usize) {
        match self {
            Self::Missing => (0, 0),
            Self::Rejected => (1, 0),
            Self::Integrated => (1, 1),
        }
    }
}

/// What the final pass did with each sample of each frame, in loader index
/// order.
pub(crate) type RejectionObserver<'a> = &'a mut dyn FnMut(usize, Vec<SampleFate>) -> Result<()>;

/// Count a frame's finite and integrated samples, handing their fates to
/// `observe` when there is one. Without an observer the fates are only
/// counted, never stored.
fn settle_frame(
    index: usize,
    fates: impl IndexedParallelIterator<Item = SampleFate>,
    observe: &mut Option<RejectionObserver<'_>>,
) -> Result<(usize, usize)> {
    let sum = |left: (usize, usize), right: (usize, usize)| (left.0 + right.0, left.1 + right.1);
    match observe {
        Some(observe) => {
            let fates = fates.collect::<Vec<_>>();
            let counts = fates
                .par_iter()
                .map(|fate| fate.counts())
                .reduce(|| (0, 0), sum);
            observe(index, fates)?;
            Ok(counts)
        }
        None => Ok(fates.map(SampleFate::counts).reduce(|| (0, 0), sum)),
    }
}

/// [`integrate_registered_frames`], also handing each frame's rejections to
/// `observe` as the final pass integrates it.
pub(crate) fn integrate_registered_frames_observed(
    frame_count: usize,
    options: &BatchStackOptions,
    mut load: impl FnMut(BatchStackPass, usize) -> Result<LinearImage>,
    mut observe: Option<RejectionObserver<'_>>,
) -> Result<BatchStackResult> {
    if frame_count == 0 || frame_count > u32::MAX as usize {
        return Err(Error::Stack(
            "batch stack frame count must fit a nonzero u32".into(),
        ));
    }
    if !options.rejection.low_sigma.is_finite()
        || options.rejection.low_sigma <= 0.0
        || !options.rejection.high_sigma.is_finite()
        || options.rejection.high_sigma <= 0.0
        || !options.minimum_sigma.is_finite()
        || options.minimum_sigma <= 0.0
    {
        return Err(Error::Stack("invalid batch rejection options".into()));
    }
    if let Some(weights) = &options.frame_weights {
        if weights.len() != frame_count {
            return Err(Error::Stack(format!(
                "batch stack has {} frame weights for {frame_count} frames",
                weights.len()
            )));
        }
        if weights.iter().any(|frame| {
            frame.is_empty()
                || frame
                    .iter()
                    .any(|weight| !weight.is_finite() || *weight <= 0.0)
        }) {
            return Err(Error::Stack(
                "batch frame weights must be finite and positive".into(),
            ));
        }
    }
    if options.frame_weights.is_some() {
        return integrate_weighted_frames(frame_count, options, load, observe);
    }
    let rejection = Rejection::new(frame_count, options);

    // First estimate: every finite sample, and each pixel's extremes.
    let mut shape = None;
    let mut first = Vec::<FirstEstimate>::new();
    let mut digests = Vec::with_capacity(frame_count);
    for index in 0..frame_count {
        let image = load_unchanged(options, &mut load, BatchStackPass::Estimate, index, shape)?;
        if shape.is_none() {
            shape = Some((image.width, image.height, image.channels));
            first.resize(image.sample_count(), FirstEstimate::default());
        }
        digests.push(sample_digest(&image.data));
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

    // Second estimate: only the samples the first keeps.
    let mut kept = vec![Moments::default(); first.len()];
    for (index, expected_digest) in digests.iter().enumerate() {
        let image = load_matching(
            options,
            &mut load,
            BatchStackPass::Refine,
            index,
            shape,
            expected_digest,
        )?;
        kept.par_iter_mut()
            .zip(first.par_iter())
            .zip(image.data.par_iter())
            .for_each(|((kept, first), &sample)| {
                if sample.is_finite() && first.keeps(sample, 1.0, &rejection) {
                    *kept = kept.with(sample, 1.0);
                }
            });
    }

    // Integrate the samples the second estimate keeps.
    let mut integrated = vec![0.0_f32; first.len()];
    let mut variance = vec![0.0_f32; first.len()];
    let mut coverage = vec![0_u32; first.len()];
    let mut rejected = vec![0_u32; first.len()];
    let mut frames = Vec::with_capacity(frame_count);
    for (index, expected_digest) in digests.iter().enumerate() {
        let image = load_matching(
            options,
            &mut load,
            BatchStackPass::Integrate,
            index,
            shape,
            expected_digest,
        )?;
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
            );
        let (finite_samples, integrated_samples) = settle_frame(index, fates, &mut observe)?;
        frames.push(BatchFrameDiagnostics {
            finite_samples,
            integrated_samples,
        });
    }
    check_cancelled(options)?;
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
    let (width, height, channels) = shape.expect("at least one frame was read");
    Ok(BatchStackResult {
        snapshot: StackSnapshot {
            image: LinearImage::new(width, height, channels, integrated)?,
            variance: LinearImage::new(width, height, channels, variance)?,
            coverage,
            rejected_samples: rejected,
            accepted_frames: frame_count as u32,
            rejected_frames: 0,
        },
        frames,
    })
}

/// The weighted form of [`integrate_registered_frames`]: the same three
/// passes, with West's weighted moments and a weighted integration.
fn integrate_weighted_frames(
    frame_count: usize,
    options: &BatchStackOptions,
    mut load: impl FnMut(BatchStackPass, usize) -> Result<LinearImage>,
    mut observe: Option<RejectionObserver<'_>>,
) -> Result<BatchStackResult> {
    let frame_weights = options
        .frame_weights
        .as_deref()
        .expect("weighted integration needs frame weights");
    let rejection = Rejection::new(frame_count, options);
    let weight_at = |index: usize, sample_index: usize, channels: usize| {
        f64::from(frame_weights[index][sample_index % channels])
    };

    let mut shape = None;
    let mut first = Vec::<FirstEstimate>::new();
    let mut digests = Vec::with_capacity(frame_count);
    for (index, weights) in frame_weights.iter().enumerate() {
        let image = load_unchanged(options, &mut load, BatchStackPass::Estimate, index, shape)?;
        validate_frame_weights(index, weights, image.channels)?;
        if shape.is_none() {
            shape = Some((image.width, image.height, image.channels));
            first.resize(image.sample_count(), FirstEstimate::default());
        }
        digests.push(sample_digest(&image.data));
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
    for (index, expected_digest) in digests.iter().enumerate() {
        let image = load_matching(
            options,
            &mut load,
            BatchStackPass::Refine,
            index,
            shape,
            expected_digest,
        )?;
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
    let mut frames = Vec::with_capacity(frame_count);
    for (index, expected_digest) in digests.iter().enumerate() {
        let image = load_matching(
            options,
            &mut load,
            BatchStackPass::Integrate,
            index,
            shape,
            expected_digest,
        )?;
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
            );
        let (finite_samples, integrated_samples) = settle_frame(index, fates, &mut observe)?;
        frames.push(BatchFrameDiagnostics {
            finite_samples,
            integrated_samples,
        });
    }
    check_cancelled(options)?;
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
    let (width, height, channels) = shape.expect("at least one frame was read");
    Ok(BatchStackResult {
        snapshot: StackSnapshot {
            image: LinearImage::new(width, height, channels, integrated)?,
            variance: LinearImage::new(width, height, channels, variance)?,
            coverage,
            rejected_samples: rejected,
            accepted_frames: frame_count as u32,
            rejected_frames: 0,
        },
        frames,
    })
}

fn validate_frame_weights(index: usize, weights: &[f32], channels: usize) -> Result<()> {
    if weights.len() != channels {
        return Err(Error::Stack(format!(
            "frame {index} has {} weights for {channels} channel(s)",
            weights.len()
        )));
    }
    Ok(())
}

fn check_cancelled(options: &BatchStackOptions) -> Result<()> {
    if options
        .cancel
        .as_ref()
        .is_some_and(CancelSignal::is_cancelled)
    {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}

fn validate_image(image: &LinearImage, shape: Option<(usize, usize, usize)>) -> Result<()> {
    if image.width == 0
        || image.height == 0
        || !matches!(image.channels, 1 | 3)
        || image
            .width
            .checked_mul(image.height)
            .and_then(|n| n.checked_mul(image.channels))
            != Some(image.data.len())
        || shape.is_some_and(|shape| shape != (image.width, image.height, image.channels))
    {
        return Err(Error::Stack(
            "batch frames must share a valid registered image shape".into(),
        ));
    }
    Ok(())
}

/// SHA-256 over the SHA-256 of each 1 Mi-sample chunk of little-endian
/// samples, so the passes' change check runs on every core.
fn sample_digest(samples: &[f32]) -> [u8; 32] {
    const CHUNK: usize = 1 << 20;
    let chunks = samples
        .par_chunks(CHUNK)
        .map(|chunk| {
            let mut hash = Sha256::new();
            let mut bytes = [0_u8; 4096];
            for part in chunk.chunks(bytes.len() / 4) {
                for (sample, output) in part.iter().zip(bytes.chunks_exact_mut(4)) {
                    output.copy_from_slice(&sample.to_le_bytes());
                }
                hash.update(&bytes[..part.len() * 4]);
            }
            <[u8; 32]>::from(hash.finalize())
        })
        .collect::<Vec<_>>();
    let mut hash = Sha256::new();
    hash.update((samples.len() as u64).to_le_bytes());
    for chunk in chunks {
        hash.update(chunk);
    }
    hash.finalize().into()
}

fn rejection_thresholds(frame_count: usize, options: &BatchStackOptions) -> Vec<(f64, f64)> {
    let normal = Normal::new(0.0, 1.0).expect("valid standard normal parameters");
    // The lower tail avoids cancellation in 1 - CDF for large sigma settings.
    let low_tail = normal.sf(f64::from(options.rejection.low_sigma));
    let high_tail = normal.sf(f64::from(options.rejection.high_sigma));
    (0..=frame_count)
        .map(|count| {
            if count < 3 {
                return (f64::INFINITY, f64::INFINITY);
            }
            // Each candidate is a new observation relative to its N-1 peers:
            // studentized predictive residuals have N-2 degrees of freedom.
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

/// Read one frame for a pass, checking cancellation around the read and
/// the shape against the first pass.
fn load_unchanged(
    options: &BatchStackOptions,
    load: &mut impl FnMut(BatchStackPass, usize) -> Result<LinearImage>,
    pass: BatchStackPass,
    index: usize,
    shape: Option<(usize, usize, usize)>,
) -> Result<LinearImage> {
    check_cancelled(options)?;
    let image = load(pass, index)?;
    check_cancelled(options)?;
    validate_image(&image, shape)?;
    Ok(image)
}

/// [`load_unchanged`] for a later pass, which must see the samples the first
/// pass saw.
fn load_matching(
    options: &BatchStackOptions,
    load: &mut impl FnMut(BatchStackPass, usize) -> Result<LinearImage>,
    pass: BatchStackPass,
    index: usize,
    shape: Option<(usize, usize, usize)>,
    expected_digest: &[u8; 32],
) -> Result<LinearImage> {
    let image = load_unchanged(options, load, pass, index, shape)?;
    if sample_digest(&image.data) != *expected_digest {
        return Err(Error::Stack(format!(
            "registered frame {index} changed between batch passes"
        )));
    }
    Ok(image)
}

/// West's weighted running moments of one pixel's samples, stored in single
/// precision to hold the pass's memory to 16 bytes per sample and updated in
/// double precision. With every weight 1 they are Welford's: `weight_sum`
/// equals `count`.
#[derive(Clone, Copy, Debug, Default)]
struct Moments {
    mean: f32,
    m2: f32,
    weight_sum: f32,
    count: u32,
}

impl Moments {
    /// These moments with one more sample of weight `weight`.
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

    /// These moments with one sample of weight `weight` taken back out.
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

    /// The same moments with `m2` scaled by `factor`, correcting a variance
    /// estimate known to run low.
    fn scaled(self, factor: f64) -> Self {
        Self {
            m2: (f64::from(self.m2) * factor) as f32,
            ..self
        }
    }

    /// Whether a sample of weight `weight` lies outside the prediction limits
    /// these moments, its peers, set for it. A peer set describes the other
    /// samples, so the thresholds are those for `count + 1` observations.
    /// `m2 / (count - 1)` estimates a weight-1 sample's variance; the
    /// candidate's residual from the peers' weighted mean has variance
    /// `s^2 (1/w + 1/W)`, and the Student-t thresholds already carry the
    /// unweighted factor `1 + 1/count`, so sigma is scaled by the ratio of
    /// the two, which is exactly 1 when every weight is 1. Sigma never falls
    /// below the configured floor, nor below a few steps of single precision
    /// at the mean, since the samples themselves carry no finer distinction.
    /// Fewer than two peers reject nothing.
    fn rejects(self, value: f32, weight: f64, rejection: &Rejection) -> bool {
        let weight_sum = f64::from(self.weight_sum);
        if self.count < 2 || weight_sum.is_nan() || weight_sum <= 0.0 {
            return false;
        }
        let peers = f64::from(self.count);
        let mean = f64::from(self.mean);
        // Equal weights make the scale exactly 1; skip its divisions.
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

/// Steps of single precision at a pixel's mean below which samples are
/// treated as equal.
const PRECISION_STEPS: f64 = 8.0;

/// What every pass needs to judge a sample: the prediction limits by depth,
/// the noise floor, and the corrections for the two estimates' known low
/// variance.
struct Rejection {
    thresholds: Vec<(f64, f64)>,
    minimum_sigma: f32,
    /// By a pixel's sample count `n`: `E[z^2]` of the largest of `n`
    /// standard normal samples, which the trimmed first estimate leaves out
    /// along with the smallest.
    expected_extreme_square: Vec<f64>,
    /// The variance of a standard normal clipped at the configured sigmas,
    /// which is what the kept moments of Gaussian noise estimate.
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
        // E[z^2] of the maximum of n samples: the integral of
        // z^2 n phi(z) Phi(z)^(n-1) over a grid wide enough for any count.
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

    /// The factor that corrects a trimmed first estimate of `count` samples,
    /// which also leaves out the candidate when `without_candidate`.
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

/// One pixel's first estimate. Pass 1 accumulates every sample's moments
/// and the largest and smallest sample; [`Self::finish`] then takes those two
/// out of the moments once the pixel has [`TRIMMED_FIRST_ESTIMATE`] samples
/// and records the variance corrections, so later passes pay for at most one
/// leave-one-out step per sample.
#[derive(Clone, Copy, Debug)]
struct FirstEstimate {
    moments: Moments,
    high: f32,
    low: f32,
    /// Until [`Self::finish`], the weight of the largest sample; after it,
    /// the variance correction for a candidate that is not an extreme.
    high_weight_or_member_factor: f32,
    /// Until [`Self::finish`], the weight of the smallest sample; after it,
    /// the variance correction for a candidate that is one.
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

    /// Trim the extremes and record the corrections. A pixel too shallow to
    /// trim keeps every sample and marks its extremes `NaN`, which no sample
    /// equals.
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

    /// A sample's peers in the first estimate: every other sample, less the
    /// extremes, with the variance corrected for trimming Gaussian noise. A
    /// sample that is itself an extreme is already out.
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

    /// Whether the first estimate keeps a sample.
    fn keeps(&self, value: f32, weight: f64, rejection: &Rejection) -> bool {
        !self.peers(value, weight).rejects(value, weight, rejection)
    }
}

/// Samples a pixel needs before the first estimate leaves its extremes out.
/// Below this the correction for trimming is too large to trust.
const TRIMMED_FIRST_ESTIMATE: u32 = 10;

/// The second estimate's peers for one sample: the kept moments, less the
/// sample when the first estimate kept it, with the variance corrected for
/// the clipping that chose them.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    fn integrate(values: &[Vec<f32>], options: &BatchStackOptions) -> StackSnapshot {
        integrate_registered_frames(values.len(), options, |_, index| {
            LinearImage::new(values[index].len(), 1, 1, values[index].clone())
        })
        .unwrap()
        .snapshot
    }

    /// One frame's bright trail used to inflate the dispersion every other
    /// sample at that pixel was judged by, so hazy frames a little brighter
    /// than the rest survived there and nowhere else, drawing the trail's
    /// line in the stack. Taken from the samples at one pixel of a 98-frame
    /// M45 stack: about 300 of noise on a 16 800 sky, a trail 50 000 above it
    /// in one frame, and four hazy frames 1 600 to 9 300 above it. The trail
    /// and the three largest hazy samples must go, and the pixel must land
    /// within the stack's own noise of the sky.
    #[test]
    fn one_large_outlier_does_not_mask_moderate_ones() {
        let sky = 16_800.0;
        let mut values = (0..98)
            .map(|index| vec![sky + ((index * 37) % 21) as f32 * 50.0 - 500.0])
            .collect::<Vec<_>>();
        let outliers = [
            (17, 66_723.0),
            (96, 26_084.0),
            (91, 20_239.0),
            (97, 20_161.0),
        ];
        for (frame, value) in outliers {
            values[frame][0] = value;
        }
        values[92][0] = 18_383.0;
        let check = |options: &BatchStackOptions| {
            let result = integrate_registered_frames(values.len(), options, |_, index| {
                LinearImage::new(1, 1, 1, values[index].clone())
            })
            .unwrap();
            for (frame, _) in outliers {
                assert_eq!(
                    result.frames[frame].integrated_samples, 0,
                    "frame {frame} kept"
                );
            }
            let snapshot = result.snapshot;
            let stack_noise = 300.0 / (snapshot.coverage[0] as f32).sqrt();
            assert!(
                (snapshot.image.data[0] - sky).abs() < stack_noise,
                "{} against {sky}",
                snapshot.image.data[0]
            );
        };
        check(&BatchStackOptions::default());
        // Equal frame weights take the weighted path.
        check(&BatchStackOptions {
            frame_weights: Some(vec![vec![1.0]; values.len()]),
            ..BatchStackOptions::default()
        });
    }

    #[test]
    fn isolated_bright_and_dark_trails_are_removed_at_every_position_and_depth() {
        for depth in [3, 5, 10, 20, 100] {
            for position in 0..depth {
                let mut values = vec![vec![1000.0, 50_000.0, 12_000.0, -50.0]; depth];
                values[position][0] += 20_000.0;
                values[position][3] -= 1000.0;
                let snapshot = integrate(&values, &BatchStackOptions::default());
                assert_eq!(
                    snapshot.image.data,
                    vec![1000.0, 50_000.0, 12_000.0, -50.0],
                    "depth={depth} trail={position}"
                );
                assert_eq!(
                    snapshot.coverage,
                    vec![
                        depth as u32 - 1,
                        depth as u32,
                        depth as u32,
                        depth as u32 - 1
                    ]
                );
                assert_eq!(snapshot.rejected_samples, vec![1, 0, 0, 1]);
            }
        }
    }

    #[test]
    fn two_intermittent_trails_are_removed_with_sufficient_depth() {
        for depth in [20, 100] {
            for positions in [[0, 1], [0, depth - 1], [depth / 2, depth - 1]] {
                let mut values = vec![vec![1000.0, 30_000.0]; depth];
                for position in positions {
                    values[position][0] += 10_000.0;
                }
                let snapshot = integrate(&values, &BatchStackOptions::default());
                assert_eq!(snapshot.image.data, vec![1000.0, 30_000.0]);
                assert_eq!(snapshot.coverage, vec![depth as u32 - 2, depth as u32]);
                assert_eq!(snapshot.rejected_samples, vec![2, 0]);
            }
        }
    }

    #[test]
    fn noisy_star_field_retains_flux_while_rejecting_crossing_trails() {
        let width = 32;
        let height = 24;
        let depth = 20;
        let mut frames = Vec::new();
        let mut expected = vec![0.0_f64; width * height];
        let mut expected_count = vec![0_u32; width * height];
        for frame in 0..depth {
            let mut samples = Vec::new();
            for y in 0..height {
                for x in 0..width {
                    let r2 = (x as f32 - 13.5).powi(2) + (y as f32 - 10.5).powi(2);
                    let noise = ((frame * 7 + x * 3 + y * 11) % 11) as f32 - 5.0;
                    let value = 1000.0 + 40_000.0 * (-r2 / 8.0).exp() + noise;
                    let trail = (frame == 0 && y == 10) || (frame == 19 && x == 13);
                    samples.push(value + if trail { 20_000.0 } else { 0.0 });
                    if !trail {
                        expected[y * width + x] += f64::from(value);
                        expected_count[y * width + x] += 1;
                    }
                }
            }
            frames.push(LinearImage::new(width, height, 1, samples).unwrap());
        }
        let result =
            integrate_registered_frames(depth, &BatchStackOptions::default(), |_, index| {
                Ok(frames[index].clone())
            })
            .unwrap();
        assert_eq!(result.frames[0].finite_samples, width * height);
        assert_eq!(result.frames[0].integrated_samples, width * height - width);
        assert_eq!(
            result.frames[19].integrated_samples,
            width * height - height
        );
        let snapshot = result.snapshot;
        for (index, &sample) in snapshot.image.data.iter().enumerate() {
            let target = expected[index] / f64::from(expected_count[index]);
            assert!(
                (f64::from(sample) - target).abs() < 0.02,
                "sample {index}: {sample}, expected {target}"
            );
            assert_eq!(snapshot.coverage[index], expected_count[index]);
        }
    }

    #[test]
    #[ignore = "set SEIZA_STACKING_REAL_FRAME to a local FITS/XISF light"]
    fn real_frame_pixels_survive_injected_early_and_late_crossing_trails() {
        let path = std::env::var("SEIZA_STACKING_REAL_FRAME").expect("fixture path required");
        let frame = crate::FitsFrame::open(path)
            .unwrap()
            .into_prepared()
            .unwrap();
        let region = crate::ReferenceRegion {
            x: frame.image.width / 2 - 128,
            y: frame.image.height / 2 - 128,
            width: 256,
            height: 256,
        };
        let source = frame.image.crop(region).unwrap();
        let depth = 20;
        let mut expected = vec![0.0_f64; source.sample_count()];
        let mut counts = vec![0_u32; source.sample_count()];
        let mut inputs = Vec::new();
        for index in 0..depth {
            let mut image = source.clone();
            for (sample, value) in image.data.iter_mut().enumerate() {
                if !value.is_finite() {
                    continue;
                }
                let pixel = sample / image.channels;
                let (x, y) = (pixel % image.width, pixel / image.width);
                let noise = ((index * 7 + x * 3 + y * 11) % 11) as f32 - 5.0;
                *value += noise;
                let trail = (index == 0 && x == 128) || (index == depth - 1 && y == 128);
                if trail {
                    *value += 100_000.0;
                } else {
                    expected[sample] += f64::from(*value);
                    counts[sample] += 1;
                }
            }
            inputs.push(image);
        }
        let result =
            integrate_registered_frames(depth, &BatchStackOptions::default(), |_, index| {
                Ok(inputs[index].clone())
            })
            .unwrap();
        for (sample, &value) in result.snapshot.image.data.iter().enumerate() {
            if counts[sample] == 0 {
                assert!(value.is_nan());
                continue;
            }
            assert_eq!(result.snapshot.coverage[sample], counts[sample]);
            let target = expected[sample] / f64::from(counts[sample]);
            assert!(
                (f64::from(value) - target).abs() < 0.0625,
                "sample {sample}: {value}, expected {target}"
            );
        }
    }

    #[test]
    fn finite_coverage_is_per_sample_and_low_depth_is_not_clipped() {
        let frames = [
            vec![1.0, 3.0, f32::NAN, f32::NAN, 7.0, 2.0],
            vec![1.0, 30.0, 6.0, f32::INFINITY, 7.0, f32::NAN],
            vec![99.0, f32::NAN, f32::NAN, f32::NEG_INFINITY, 7.0, f32::NAN],
        ];
        let snapshot = integrate(&frames, &BatchStackOptions::default());
        assert_eq!(&snapshot.image.data[..3], &[1.0, 16.5, 6.0]);
        assert!(snapshot.image.data[3].is_nan());
        assert_eq!(&snapshot.image.data[4..], &[7.0, 2.0]);
        assert_eq!(snapshot.coverage, vec![2, 2, 1, 0, 3, 1]);
        assert_eq!(snapshot.rejected_samples, vec![1, 0, 0, 0, 0, 0]);
        assert_eq!(snapshot.variance.data, vec![0.0, 364.5, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn rgb_samples_keep_physical_scale_and_independent_masks() {
        let frames = [
            vec![10.0, 2000.0, 30_000.0],
            vec![10.0, 9000.0, 30_000.0],
            vec![10.0, 2000.0, 30_000.0],
        ];
        let snapshot = integrate_registered_frames(3, &BatchStackOptions::default(), |_, index| {
            LinearImage::new(1, 1, 3, frames[index].clone())
        })
        .unwrap()
        .snapshot;
        assert_eq!(snapshot.image.channels, 3);
        assert_eq!(snapshot.image.data, vec![10.0, 2000.0, 30_000.0]);
        assert_eq!(snapshot.coverage, vec![3, 2, 3]);
    }

    #[test]
    fn explicit_noise_floor_preserves_near_constant_samples() {
        let snapshot = integrate(
            &[vec![1000.0], vec![1000.0], vec![1001.0]],
            &BatchStackOptions {
                minimum_sigma: 1.0,
                ..BatchStackOptions::default()
            },
        );
        assert_eq!(snapshot.coverage, vec![3]);
        assert!((snapshot.image.data[0] - 1000.3333).abs() < 0.001);
    }

    #[test]
    fn independent_gaussian_noise_retains_unbiased_finite_means() {
        let pixels = 16_384;
        let mut seed = 0x923a_583d_13b7_049e_u64;
        let mut uniform = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
        };
        for depth in [3, 5, 20, 100] {
            let mut inputs = Vec::new();
            let mut raw = vec![0.0_f64; pixels];
            for _ in 0..depth {
                let values: Vec<_> = raw
                    .iter_mut()
                    .map(|mean| {
                        let noise = (-2.0 * uniform().ln()).sqrt()
                            * (std::f64::consts::TAU * uniform()).cos();
                        let value = (1000.0 + 10.0 * noise) as f32;
                        *mean += f64::from(value) / depth as f64;
                        value
                    })
                    .collect();
                inputs.push(values);
            }
            let snapshot = integrate(&inputs, &BatchStackOptions::default());
            let bias = snapshot
                .image
                .data
                .iter()
                .map(|&value| f64::from(value) - 1000.0)
                .sum::<f64>()
                / pixels as f64;
            let rms = (snapshot
                .image
                .data
                .iter()
                .map(|&value| (f64::from(value) - 1000.0).powi(2))
                .sum::<f64>()
                / pixels as f64)
                .sqrt();
            let raw_rms = (raw
                .iter()
                .map(|value| (value - 1000.0).powi(2))
                .sum::<f64>()
                / pixels as f64)
                .sqrt();
            let rejected: u32 = snapshot.rejected_samples.iter().sum();
            let rejected_fraction = f64::from(rejected) / (pixels * depth) as f64;
            eprintln!(
                "Gaussian depth={depth}: bias={bias:.5} clipped_rmse={rms:.5} mean_rmse={raw_rms:.5} rejected={rejected_fraction:.5}"
            );
            assert!(snapshot.image.data.iter().all(|value| value.is_finite()));
            assert!(snapshot.coverage.iter().all(|&count| count > 0));
            assert!(bias.abs() < 0.2);
            assert!(rms / raw_rms < 1.03);
            assert!((rejected_fraction - 0.0027).abs() < 0.001);
        }
    }

    #[test]
    fn predictive_thresholds_keep_asymmetric_tail_probabilities() {
        let options = BatchStackOptions {
            rejection: MasterRejectionOptions {
                low_sigma: 2.0,
                high_sigma: 4.0,
            },
            ..BatchStackOptions::default()
        };
        let thresholds = rejection_thresholds(100, &options);
        let normal = Normal::new(0.0, 1.0).unwrap();
        for count in [3, 5, 20, 100] {
            let peers = (count - 1) as f64;
            let predictive = StudentsT::new(0.0, (1.0 + 1.0 / peers).sqrt(), peers - 1.0).unwrap();
            let (low, high) = thresholds[count];
            assert!(low < high);
            assert!((predictive.cdf(-low) - normal.sf(2.0)).abs() < 1.0e-10);
            assert!((predictive.cdf(-high) - normal.sf(4.0)).abs() < 1.0e-10);
        }
        assert!(thresholds[3].0 > thresholds[5].0);
        assert!(thresholds[5].0 > thresholds[100].0);
        let high = rejection_thresholds(
            3,
            &BatchStackOptions {
                rejection: MasterRejectionOptions {
                    low_sigma: 40.0,
                    high_sigma: 40.0,
                },
                ..BatchStackOptions::default()
            },
        );
        assert_eq!(high[3], (f64::INFINITY, f64::INFINITY));
    }

    #[test]
    fn cancellation_and_loader_errors_do_not_return_partial_images() {
        let flag = Arc::new(AtomicBool::new(false));
        let options = BatchStackOptions {
            cancel: Some(Arc::clone(&flag).into()),
            ..BatchStackOptions::default()
        };
        let mut calls = 0;
        let result = integrate_registered_frames(3, &options, |pass, index| {
            calls += 1;
            if pass == BatchStackPass::Integrate && index == 0 {
                flag.store(true, Ordering::Relaxed);
            }
            LinearImage::new(1, 1, 1, vec![10.0])
        });
        assert!(matches!(result, Err(Error::Cancelled)));
        assert_eq!(calls, 7);
        let error = integrate_registered_frames(3, &BatchStackOptions::default(), |pass, _| {
            if pass == BatchStackPass::Integrate {
                Err(Error::Stack("source disappeared".into()))
            } else {
                LinearImage::new(1, 1, 1, vec![10.0])
            }
        })
        .unwrap_err();
        assert!(error.to_string().contains("source disappeared"));
    }

    #[test]
    fn changed_shapes_samples_and_invalid_options_are_rejected() {
        let options = BatchStackOptions::default();
        assert!(integrate_registered_frames(0, &options, |_, _| unreachable!()).is_err());
        assert!(
            integrate_registered_frames(
                2,
                &BatchStackOptions {
                    minimum_sigma: f32::NAN,
                    ..options.clone()
                },
                |_, _| unreachable!()
            )
            .is_err()
        );
        assert!(
            integrate_registered_frames(2, &options, |_, index| LinearImage::new(
                index + 1,
                1,
                1,
                vec![1.0; index + 1]
            ))
            .is_err()
        );
        let error = integrate_registered_frames(2, &options, |pass, _| {
            LinearImage::new(
                1,
                1,
                1,
                vec![if pass == BatchStackPass::Estimate {
                    1.0
                } else {
                    2.0
                }],
            )
        })
        .unwrap_err();
        assert!(error.to_string().contains("changed between batch passes"));
        assert!(
            integrate_registered_frames(1, &options, |_, _| Ok(LinearImage {
                width: 2,
                height: 2,
                channels: 1,
                data: vec![1.0]
            }))
            .is_err()
        );
    }
}
