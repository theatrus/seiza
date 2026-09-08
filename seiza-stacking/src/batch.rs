use crate::{CancelSignal, Error, LinearImage, MasterRejectionOptions, Result, StackSnapshot};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use statrs::distribution::{ContinuousCDF, Normal, StudentsT};

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
}

impl Default for BatchStackOptions {
    fn default() -> Self {
        Self {
            rejection: MasterRejectionOptions::default(),
            minimum_sigma: 1.0e-6,
            cancel: None,
        }
    }
}

/// Which of the two sequential reads is requesting an input frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchStackPass {
    /// Estimate moments using every finite registered sample.
    Estimate,
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

/// Integrate already-registered frames with two-pass leave-one-out rejection.
///
/// Unlike live delta-sigma rejection, this revisits the reference and warm-up
/// frames, so an isolated early transient cannot remain in the final average.
/// Pixels with fewer than three finite observations are averaged without
/// rejection. A Student-t prediction limit accounts for uncertain noise
/// estimates at low depth while preserving the configured Gaussian tail
/// probabilities. Several overlapping transients can still mask one another by
/// inflating the estimated dispersion; this is not a median/MAD estimator.
///
/// The loader must return the same calibrated, registered, normalized image
/// for an index on both passes. Shapes and sample digests are checked before
/// accumulation. The caller owns registration, whole-frame admission, and
/// source provenance; no registration, normalization, or frame-level quality
/// decisions are repeated here. All supplied frames count as admitted.
///
/// Memory is proportional to the output dimensions, not the number of frames:
/// approximately 36 bytes per sample plus one loaded input and small per-frame
/// digests. Drop any online accumulator before calling this when memory is tight.
pub fn integrate_registered_frames(
    frame_count: usize,
    options: &BatchStackOptions,
    mut load: impl FnMut(BatchStackPass, usize) -> Result<LinearImage>,
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
    let thresholds = rejection_thresholds(frame_count, options);
    let mut shape = None;
    let mut mean = Vec::<f64>::new();
    let mut m2 = Vec::<f64>::new();
    let mut count = Vec::<u32>::new();
    let mut digests = Vec::with_capacity(frame_count);
    for index in 0..frame_count {
        check_cancelled(options)?;
        let image = load(BatchStackPass::Estimate, index)?;
        check_cancelled(options)?;
        validate_image(&image, shape)?;
        if shape.is_none() {
            shape = Some((image.width, image.height, image.channels));
            mean.resize(image.sample_count(), 0.0);
            m2.resize(image.sample_count(), 0.0);
            count.resize(image.sample_count(), 0);
        }
        digests.push(sample_digest(&image.data));
        mean.par_iter_mut()
            .zip(m2.par_iter_mut())
            .zip(count.par_iter_mut())
            .zip(image.data.par_iter())
            .for_each(|(((mean, m2), count), &sample)| {
                if sample.is_finite() {
                    *count += 1;
                    let sample = f64::from(sample);
                    let delta = sample - *mean;
                    *mean += delta / f64::from(*count);
                    *m2 += delta * (sample - *mean);
                }
            });
    }
    let mut integrated = vec![0.0_f32; mean.len()];
    let mut variance = vec![0.0_f32; mean.len()];
    let mut coverage = vec![0_u32; mean.len()];
    let mut rejected = vec![0_u32; mean.len()];
    let mut frames = Vec::with_capacity(frame_count);
    for (index, expected_digest) in digests.iter().enumerate() {
        check_cancelled(options)?;
        let image = load(BatchStackPass::Integrate, index)?;
        check_cancelled(options)?;
        validate_image(&image, shape)?;
        if sample_digest(&image.data) != *expected_digest {
            return Err(Error::Stack(format!(
                "registered frame {index} changed between batch passes"
            )));
        }
        let (finite_samples, integrated_samples) = integrated
            .par_iter_mut()
            .zip(variance.par_iter_mut())
            .zip(coverage.par_iter_mut())
            .zip(rejected.par_iter_mut())
            .enumerate()
            .map(
                |(sample_index, (((integrated, variance), coverage), rejected))| {
                    let sample = image.data[sample_index];
                    if !sample.is_finite() {
                        return (0, 0);
                    }
                    if rejects_sample(
                        sample,
                        mean[sample_index],
                        m2[sample_index],
                        count[sample_index],
                        thresholds[count[sample_index] as usize],
                        options.minimum_sigma,
                    ) {
                        *rejected += 1;
                        return (1, 0);
                    }
                    *coverage += 1;
                    let delta = f64::from(sample) - f64::from(*integrated);
                    let next_mean = f64::from(*integrated) + delta / f64::from(*coverage);
                    *variance =
                        (f64::from(*variance) + delta * (f64::from(sample) - next_mean)) as f32;
                    *integrated = next_mean as f32;
                    (1, 1)
                },
            )
            .reduce(
                || (0, 0),
                |left, right| (left.0 + right.0, left.1 + right.1),
            );
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

fn sample_digest(samples: &[f32]) -> [u8; 32] {
    let mut hash = Sha256::new();
    let mut bytes = [0_u8; 4096];
    for chunk in samples.chunks(bytes.len() / 4) {
        for (sample, output) in chunk.iter().zip(bytes.chunks_exact_mut(4)) {
            output.copy_from_slice(&sample.to_le_bytes());
        }
        hash.update(&bytes[..chunk.len() * 4]);
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

fn rejects_sample(
    value: f32,
    mean: f64,
    m2: f64,
    count: u32,
    thresholds: (f64, f64),
    minimum_sigma: f32,
) -> bool {
    if count < 3 {
        return false;
    }
    // Remove the candidate's contribution before measuring its deviation so
    // that a bright trail does not raise its own rejection threshold.
    let value = f64::from(value);
    let others = f64::from(count - 1);
    let other_mean = mean + (mean - value) / others;
    let other_m2 = (m2 - (value - mean) * (value - other_mean)).max(0.0);
    let sigma = (other_m2 / (others - 1.0))
        .sqrt()
        .max(f64::from(minimum_sigma));
    let residual = value - other_mean;
    residual < -thresholds.0 * sigma || residual > thresholds.1 * sigma
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
        assert_eq!(calls, 4);
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
