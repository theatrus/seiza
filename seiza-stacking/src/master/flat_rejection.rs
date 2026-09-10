use super::{MasterBuildOptions, MasterInputStatistics, MasterRejectionOptions, check_cancelled};
use crate::{Error, Result};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

const TILE_BYTES: usize = 64 * 1024 * 1024;
const IO_SAMPLES: usize = 16 * 1024;

/// Prepared, normalized frames in frame-major order. The named temporary file
/// owns cleanup even when decoding, cancellation, or integration fails.
pub(super) struct FlatScratch {
    file: tempfile::NamedTempFile,
    samples: usize,
    frames: usize,
}

pub(super) struct IntegratedFlat {
    pub samples: Vec<f32>,
    pub input_statistics: Vec<MasterInputStatistics>,
    pub accepted_samples: u64,
    pub rejected_samples: u64,
    pub fallback_pixels: u64,
}

impl FlatScratch {
    pub fn new(directory: Option<&Path>) -> Result<Self> {
        let mut builder = tempfile::Builder::new();
        builder.prefix("seiza-flat-");
        let file = match directory {
            Some(directory) => builder.tempfile_in(directory),
            None => builder.tempfile(),
        }
        .map_err(scratch_error)?;
        Ok(Self {
            file,
            samples: 0,
            frames: 0,
        })
    }

    pub fn append(&mut self, samples: &[f32], options: &MasterBuildOptions) -> Result<()> {
        if self.frames > 0 && samples.len() != self.samples {
            return Err(Error::Calibration("flat scratch dimensions changed".into()));
        }
        let frames = self.frames.checked_add(1).ok_or_else(size_error)?;
        byte_offset(samples.len(), frames, 0)?;
        let mut bytes = Vec::with_capacity(IO_SAMPLES * 4);
        for chunk in samples.chunks(IO_SAMPLES) {
            check_cancelled(options)?;
            bytes.clear();
            for sample in chunk {
                bytes.extend_from_slice(&sample.to_le_bytes());
            }
            self.file.write_all(&bytes).map_err(scratch_error)?;
        }
        self.samples = samples.len();
        self.frames = frames;
        Ok(())
    }

    pub fn integrate(self, options: &MasterBuildOptions) -> Result<IntegratedFlat> {
        self.integrate_with_budget(options, TILE_BYTES)
    }

    fn integrate_with_budget(
        mut self,
        options: &MasterBuildOptions,
        tile_bytes: usize,
    ) -> Result<IntegratedFlat> {
        let tile_samples = tile_samples(self.samples, self.frames, tile_bytes)?;
        let total_samples = u64::try_from(self.samples)
            .ok()
            .and_then(|samples| samples.checked_mul(u64::try_from(self.frames).ok()?))
            .ok_or_else(size_error)?;
        let mut result = IntegratedFlat {
            samples: vec![0.0; self.samples],
            input_statistics: vec![
                MasterInputStatistics {
                    accepted_samples: 0,
                    rejected_samples: 0,
                };
                self.frames
            ],
            accepted_samples: 0,
            rejected_samples: 0,
            fallback_pixels: 0,
        };
        let mut tile = vec![0.0; tile_samples * self.frames];
        let mut bytes = vec![0_u8; tile_samples * 4];
        let mut statistics = vec![0.0; self.frames];

        for start in (0..self.samples).step_by(tile_samples) {
            check_cancelled(options)?;
            let length = tile_samples.min(self.samples - start);
            // Read each normalized frame's tile once, transposing so all
            // temporal samples for one sensor pixel are contiguous.
            for frame in 0..self.frames {
                check_cancelled(options)?;
                self.file
                    .seek(SeekFrom::Start(byte_offset(self.samples, frame, start)?))
                    .map_err(scratch_error)?;
                let bytes = &mut bytes[..length * 4];
                self.file.read_exact(bytes).map_err(scratch_error)?;
                for (index, bytes) in bytes.chunks_exact(4).enumerate() {
                    tile[index * self.frames + frame] =
                        f32::from_le_bytes(bytes.try_into().expect("four-byte sample"));
                }
            }
            for (index, values) in tile[..length * self.frames]
                .chunks_exact(self.frames)
                .enumerate()
            {
                if index % 1024 == 0 {
                    check_cancelled(options)?;
                }
                let bounds = rejection_bounds(values, &mut statistics, options.rejection);
                let mut sum = 0.0_f64;
                let mut kept = 0;
                for (value, counts) in values.iter().zip(&mut result.input_statistics) {
                    if self.frames >= 3 && bounds.rejects(*value) {
                        counts.rejected_samples += 1;
                        result.rejected_samples += 1;
                    } else {
                        sum += f64::from(*value);
                        kept += 1;
                        counts.accepted_samples += 1;
                    }
                }
                result.samples[start + index] = if kept == 0 {
                    result.fallback_pixels += 1;
                    bounds.center as f32
                } else {
                    (sum / kept as f64) as f32
                };
            }
        }
        result.accepted_samples = total_samples - result.rejected_samples;
        Ok(result)
    }
}

fn tile_samples(samples: usize, frames: usize, budget: usize) -> Result<usize> {
    if samples == 0 || frames < 2 {
        return Err(Error::Calibration(
            "flat integration requires at least two nonempty frames".into(),
        ));
    }
    // Payload: temporal tile, one input tile's bytes, and one pixel's MAD
    // workspace. The output image and small per-frame tallies are separate.
    let statistics_bytes = frames.checked_mul(4).ok_or_else(size_error)?;
    let bytes_per_sample = frames
        .checked_add(1)
        .and_then(|frames| frames.checked_mul(4))
        .ok_or_else(size_error)?;
    let available = budget
        .checked_sub(statistics_bytes)
        .ok_or_else(size_error)?;
    let count = available / bytes_per_sample;
    if count == 0 {
        return Err(Error::Calibration(
            "flat integration tile budget cannot hold one pixel across all inputs".into(),
        ));
    }
    Ok(samples.min(count))
}

fn byte_offset(samples: usize, frame: usize, start: usize) -> Result<u64> {
    samples
        .checked_mul(frame)
        .and_then(|offset| offset.checked_add(start))
        .and_then(|offset| offset.checked_mul(4))
        .and_then(|offset| u64::try_from(offset).ok())
        .ok_or_else(size_error)
}

struct RejectionBounds {
    center: f64,
    low: f64,
    high: f64,
}

impl RejectionBounds {
    fn rejects(&self, value: f32) -> bool {
        let residual = f64::from(value) - self.center;
        residual < -self.low || residual > self.high
    }
}

fn rejection_bounds(
    values: &[f32],
    statistics: &mut [f32],
    options: MasterRejectionOptions,
) -> RejectionBounds {
    // Scaling only extreme finite inputs prevents overflow in the shared
    // f32 median/MAD helpers without changing ordinary normalized flats.
    let maximum = values.iter().map(|value| value.abs()).fold(1.0, f32::max);
    let scale = if maximum > f32::MAX / 4.0 {
        maximum
    } else {
        1.0
    };
    for (destination, value) in statistics.iter_mut().zip(values) {
        *destination = *value / scale;
    }
    let center = seiza_stats::median_in_place(statistics).expect("nonempty temporal sample");
    let sigma =
        seiza_stats::robust_sigma_in_place(statistics, center).expect("nonempty temporal sample");
    let center = f64::from(center) * f64::from(scale);
    let sigma = f64::from(sigma) * f64::from(scale);
    let tolerance = f64::from(f32::EPSILON) * center.abs().max(1.0) * 8.0;
    RejectionBounds {
        center,
        low: (f64::from(options.low_sigma) * sigma).max(tolerance),
        high: (f64::from(options.high_sigma) * sigma).max(tolerance),
    }
}

fn scratch_error(error: std::io::Error) -> Error {
    Error::Calibration(format!("flat master scratch storage: {error}"))
}

fn size_error() -> Error {
    Error::Calibration("flat master scratch dimensions exceed the supported size".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CancelSignal;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn integrate(frames: &[Vec<f32>], budget: usize) -> IntegratedFlat {
        let directory = tempfile::tempdir().unwrap();
        let mut scratch = FlatScratch::new(Some(directory.path())).unwrap();
        let options = MasterBuildOptions::default();
        for frame in frames {
            scratch.append(frame, &options).unwrap();
        }
        let result = scratch.integrate_with_budget(&options, budget).unwrap();
        assert_eq!(directory.path().read_dir().unwrap().count(), 0);
        result
    }

    #[test]
    fn noisy_temporal_clipping_matches_across_tile_sizes_and_input_order() {
        let frames = (0..8)
            .map(|frame| {
                (0..37)
                    .map(|pixel| {
                        let noise = ((frame + pixel) % 6) as f32 * 0.001 - 0.0025;
                        let transient = if frame >= 6 && pixel % 3 == 0 {
                            1.0
                        } else {
                            0.0
                        };
                        1.0 + noise + transient
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let full = integrate(&frames, TILE_BYTES);
        // 32 bytes of per-pixel scratch plus 36 bytes for a one-pixel tile.
        let tiny = integrate(&frames, 68);
        let reversed = integrate(&frames.into_iter().rev().collect::<Vec<_>>(), 68);
        assert_eq!(full.samples, tiny.samples);
        assert_eq!(full.samples, reversed.samples);
        assert_eq!(full.input_statistics, tiny.input_statistics);
        assert_eq!(
            full.input_statistics,
            reversed
                .input_statistics
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
        );
        assert_eq!(full.rejected_samples, 26);
        assert_eq!(full.accepted_samples + full.rejected_samples, 8 * 37);
        for sample in full.samples {
            assert!((sample - 1.0).abs() < 0.003, "{sample}");
        }
    }

    #[test]
    fn two_frame_and_evenly_split_sets_do_not_guess_which_signal_is_real() {
        let two = integrate(&[vec![1.0], vec![2.0]], TILE_BYTES);
        assert_eq!(two.samples, [1.5]);
        assert_eq!(two.rejected_samples, 0);
        let split = integrate(&[vec![1.0], vec![1.0], vec![2.0], vec![2.0]], TILE_BYTES);
        assert_eq!(split.samples, [1.5]);
        assert_eq!(split.rejected_samples, 0);
    }

    #[test]
    fn an_overly_tight_threshold_falls_back_to_the_median_not_the_outlier_mean() {
        let directory = tempfile::tempdir().unwrap();
        let mut scratch = FlatScratch::new(Some(directory.path())).unwrap();
        let options = MasterBuildOptions {
            rejection: MasterRejectionOptions {
                low_sigma: 0.1,
                high_sigma: 0.1,
            },
            ..Default::default()
        };
        for value in [0.8, 0.9, 1.0, 1.1, 1.2, 10.0] {
            scratch.append(&[value], &options).unwrap();
        }
        let result = scratch.integrate(&options).unwrap();
        assert!((result.samples[0] - 1.05).abs() < 1.0e-6);
        assert_eq!(result.fallback_pixels, 1);
        assert_eq!(result.rejected_samples, 6);
        assert_eq!(result.accepted_samples, 0);
        assert_eq!(directory.path().read_dir().unwrap().count(), 0);
    }

    #[test]
    fn robust_bounds_reject_both_tails_and_tolerate_flat_rounding() {
        let values = [1.0, 1.0, 1.0, 1.0 + f32::EPSILON, 0.1, 10.0];
        let mut workspace = [0.0; 6];
        let bounds = rejection_bounds(&values, &mut workspace, MasterRejectionOptions::default());
        assert!(values[..4].iter().all(|value| !bounds.rejects(*value)));
        assert!(bounds.rejects(values[4]));
        assert!(bounds.rejects(values[5]));
    }

    #[test]
    fn large_finite_responses_do_not_overflow_the_robust_center() {
        let value = f32::MAX * 0.5;
        let result = integrate(&[vec![value], vec![value], vec![f32::MAX]], TILE_BYTES);
        assert_eq!(result.samples, [value]);
        assert_eq!(result.rejected_samples, 1);
    }

    #[test]
    fn cancellation_during_tile_reads_removes_scratch() {
        let directory = tempfile::tempdir().unwrap();
        let mut scratch = FlatScratch::new(Some(directory.path())).unwrap();
        for _ in 0..8 {
            scratch
                .append(&[1.0; 8], &MasterBuildOptions::default())
                .unwrap();
        }
        assert_eq!(directory.path().read_dir().unwrap().count(), 1);
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);
        let options = MasterBuildOptions {
            cancel: Some(CancelSignal::new(move || {
                counter.fetch_add(1, Ordering::Relaxed) >= 3
            })),
            ..Default::default()
        };
        assert!(matches!(scratch.integrate(&options), Err(Error::Cancelled)));
        assert_eq!(calls.load(Ordering::Relaxed), 4);
        assert_eq!(directory.path().read_dir().unwrap().count(), 0);
    }

    #[test]
    fn truncated_scratch_fails_and_is_removed() {
        let directory = tempfile::tempdir().unwrap();
        let mut scratch = FlatScratch::new(Some(directory.path())).unwrap();
        let options = MasterBuildOptions::default();
        for _ in 0..3 {
            scratch.append(&[1.0; 8], &options).unwrap();
        }
        scratch.file.as_file().set_len(4).unwrap();
        let error = match scratch.integrate(&options) {
            Ok(_) => panic!("truncated scratch must fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("scratch storage"));
        assert_eq!(directory.path().read_dir().unwrap().count(), 0);
    }

    #[test]
    fn tile_planning_and_offsets_are_checked() {
        assert_eq!(tile_samples(100, 8, 68).unwrap(), 1);
        assert!(tile_samples(100, 8, 67).is_err());
        assert!(tile_samples(100, usize::MAX, TILE_BYTES).is_err());
        assert!(tile_samples(0, 8, TILE_BYTES).is_err());
        assert!(tile_samples(100, 1, TILE_BYTES).is_err());
        assert!(byte_offset(usize::MAX, 2, 0).is_err());
    }

    #[test]
    #[ignore = "manual 64-frame, 1 MP flat-integration performance check"]
    fn benchmark_flat_integration() {
        let directory = tempfile::tempdir().unwrap();
        let options = MasterBuildOptions::default();
        let mut scratch = FlatScratch::new(Some(directory.path())).unwrap();
        let mut samples = vec![0.0; 1_000_000];
        for frame in 0..64 {
            for (pixel, sample) in samples.iter_mut().enumerate() {
                let noise = (pixel as u32)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(frame * 1_048_573)
                    % 997;
                *sample = 1.0 + noise as f32 * 0.00001;
                if frame < 8 && pixel % 997 == 0 {
                    *sample += 2.0;
                }
            }
            scratch.append(&samples, &options).unwrap();
        }
        let started = std::time::Instant::now();
        let result = scratch.integrate(&options).unwrap();
        eprintln!(
            "64-frame 1 MP normalized flat integration: {:?}; {} rejected samples",
            started.elapsed(),
            result.rejected_samples
        );
        assert!(
            result
                .samples
                .iter()
                .all(|value| (value - 1.0).abs() < 0.02)
        );
        assert!(result.rejected_samples > 0);
    }
}
