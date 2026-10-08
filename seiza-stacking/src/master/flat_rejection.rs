use super::{
    MasterBuildOptions, MasterBuildStage, MasterInputStatistics, MasterRejectionOptions,
    check_cancelled, report_progress,
};
use crate::{Error, Result};
use rayon::prelude::*;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

const TILE_BYTES: usize = 64 * 1024 * 1024;
/// Samples encoded and written to scratch storage at a time.
const IO_SAMPLES: usize = 1024 * 1024;
/// Pixels one rayon task combines. Every pixel is clipped and averaged on its
/// own, over its frames in order, so the split never changes a result.
const COMBINE_CHUNK: usize = 1024;
/// Pixels combined between two cancellation checks. The checks stay on the
/// calling thread, so a host's cancel callback is never called from a worker.
const COMBINE_BATCH: usize = 64 * 1024;

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
    pub masked_samples: u64,
    pub masked_output_samples: u64,
    pub minimum_clean_samples: usize,
    pub maximum_clean_samples: usize,
    pub low_coverage_samples: u64,
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
        let mut bytes = vec![0_u8; IO_SAMPLES.min(samples.len()) * 4];
        for chunk in samples.chunks(IO_SAMPLES) {
            check_cancelled(options)?;
            let bytes = &mut bytes[..chunk.len() * 4];
            bytes
                .par_chunks_mut(16 * 1024 * 4)
                .zip(chunk.par_chunks(16 * 1024))
                .for_each(|(bytes, chunk)| {
                    for (bytes, sample) in bytes.chunks_exact_mut(4).zip(chunk) {
                        bytes.copy_from_slice(&sample.to_le_bytes());
                    }
                });
            self.file.write_all(bytes).map_err(scratch_error)?;
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
        let mut samples = vec![0.0; self.samples];
        // Each frame's tile, frame after frame, as stored. Workers gather one
        // short run of pixels at a time into pixel-major order, which keeps
        // the transpose in cache instead of striding across the whole tile.
        let mut tile = vec![0_u8; tile_samples * self.frames * 4];
        let mut totals = CombineTally::new(self.frames);
        let masking = options.flat_star_masking.is_some();
        let minimum_clean = options
            .flat_star_masking
            .as_ref()
            .map(|masking| masking.minimum_clean_samples);

        let tiles = self.samples.div_ceil(tile_samples);
        for (tile_index, start) in (0..self.samples).step_by(tile_samples).enumerate() {
            check_cancelled(options)?;
            report_progress(options, MasterBuildStage::Combine, tile_index, tiles);
            let length = tile_samples.min(self.samples - start);
            let tile = &mut tile[..length * self.frames * 4];
            // Read each normalized frame's tile once.
            for (frame, bytes) in tile.chunks_exact_mut(length * 4).enumerate() {
                check_cancelled(options)?;
                self.file
                    .seek(SeekFrom::Start(byte_offset(self.samples, frame, start)?))
                    .map_err(scratch_error)?;
                self.file.read_exact(bytes).map_err(scratch_error)?;
            }
            let tile = &*tile;
            let output = &mut samples[start..start + length];
            for (batch, output) in output.chunks_mut(COMBINE_BATCH).enumerate() {
                check_cancelled(options)?;
                let batch_start = batch * COMBINE_BATCH;
                let tally = output
                    .par_chunks_mut(COMBINE_CHUNK)
                    .enumerate()
                    .map_init(
                        || vec![0.0; COMBINE_CHUNK * self.frames],
                        |values, (chunk, output)| {
                            let first = batch_start + chunk * COMBINE_CHUNK;
                            let values = &mut values[..output.len() * self.frames];
                            gather_pixels(tile, length, first, values, masking)?;
                            Ok(combine_pixels(
                                values,
                                output,
                                self.frames,
                                options.rejection,
                                minimum_clean,
                            ))
                        },
                    )
                    .try_reduce(
                        || CombineTally::new(self.frames),
                        |left, right| Ok(left.merge(right)),
                    )?;
                totals = totals.merge(tally);
            }
        }
        let result = IntegratedFlat {
            samples,
            input_statistics: totals.input_statistics,
            accepted_samples: total_samples - totals.rejected_samples - totals.masked_samples,
            rejected_samples: totals.rejected_samples,
            masked_samples: totals.masked_samples,
            masked_output_samples: totals.masked_output_samples,
            minimum_clean_samples: totals.minimum_clean_samples,
            maximum_clean_samples: totals.maximum_clean_samples,
            low_coverage_samples: totals.low_coverage_samples,
            fallback_pixels: totals.fallback_pixels,
        };
        let insufficient_samples = totals.insufficient_samples;
        if insufficient_samples > 0 {
            return Err(Error::InsufficientFlatCoverage {
                insufficient_samples,
                required_clean_samples: options
                    .flat_star_masking
                    .as_ref()
                    .unwrap()
                    .minimum_clean_samples,
                minimum_clean_samples: result.minimum_clean_samples,
                maximum_clean_samples: result.maximum_clean_samples,
                masked_samples: result.masked_samples,
            });
        }
        report_progress(options, MasterBuildStage::Combine, tiles, tiles);
        Ok(result)
    }
}

/// Decode pixels `first..first + values.len() / frames` of a frame-major
/// tile of `length` samples per frame into `values`, each pixel's temporal
/// samples together. Pure copies: the result does not depend on the split.
fn gather_pixels(
    tile: &[u8],
    length: usize,
    first: usize,
    values: &mut [f32],
    masking: bool,
) -> Result<()> {
    let frames = tile.len() / (length * 4);
    let pixels = values.len() / frames;
    for (frame, bytes) in tile.chunks_exact(length * 4).enumerate() {
        let bytes = &bytes[first * 4..(first + pixels) * 4];
        for (index, bytes) in bytes.chunks_exact(4).enumerate() {
            let value = f32::from_le_bytes(bytes.try_into().expect("four-byte sample"));
            if !value.is_finite() && !(masking && value.is_nan()) {
                return Err(Error::FlatStarMasking(
                    "unexpected non-finite sample in flat scratch storage".into(),
                ));
            }
            values[index * frames + frame] = value;
        }
    }
    Ok(())
}

/// Counts from combining a run of pixels. Every field is an integer count, a
/// minimum, or a maximum, so merging runs in any order gives the serial totals.
struct CombineTally {
    input_statistics: Vec<MasterInputStatistics>,
    rejected_samples: u64,
    masked_samples: u64,
    masked_output_samples: u64,
    minimum_clean_samples: usize,
    maximum_clean_samples: usize,
    low_coverage_samples: u64,
    fallback_pixels: u64,
    insufficient_samples: u64,
}

impl CombineTally {
    fn new(frames: usize) -> Self {
        Self {
            input_statistics: vec![
                MasterInputStatistics {
                    accepted_samples: 0,
                    rejected_samples: 0,
                    masked_samples: 0,
                };
                frames
            ],
            rejected_samples: 0,
            masked_samples: 0,
            masked_output_samples: 0,
            minimum_clean_samples: frames,
            maximum_clean_samples: 0,
            low_coverage_samples: 0,
            fallback_pixels: 0,
            insufficient_samples: 0,
        }
    }

    fn merge(mut self, other: Self) -> Self {
        for (total, counts) in self.input_statistics.iter_mut().zip(other.input_statistics) {
            total.accepted_samples += counts.accepted_samples;
            total.rejected_samples += counts.rejected_samples;
            total.masked_samples += counts.masked_samples;
        }
        self.rejected_samples += other.rejected_samples;
        self.masked_samples += other.masked_samples;
        self.masked_output_samples += other.masked_output_samples;
        self.minimum_clean_samples = self.minimum_clean_samples.min(other.minimum_clean_samples);
        self.maximum_clean_samples = self.maximum_clean_samples.max(other.maximum_clean_samples);
        self.low_coverage_samples += other.low_coverage_samples;
        self.fallback_pixels += other.fallback_pixels;
        self.insufficient_samples += other.insufficient_samples;
        self
    }
}

/// Clip and average a run of pixels, `values` holding each pixel's `frames`
/// temporal samples together. Unmasked pixels with nothing left fall back to
/// their temporal median; masked builds write NaN there and count the pixel
/// as short of `minimum_clean` retained samples, so the build fails instead.
fn combine_pixels(
    values: &[f32],
    output: &mut [f32],
    frames: usize,
    rejection: MasterRejectionOptions,
    minimum_clean: Option<usize>,
) -> CombineTally {
    let mut tally = CombineTally::new(frames);
    let mut keys = vec![0; frames];
    for (values, output) in values.chunks_exact(frames).zip(output) {
        let bounds = rejection_bounds(values, &mut keys, rejection);
        let available = values.iter().filter(|value| value.is_finite()).count();
        if available < frames {
            tally.masked_output_samples += 1;
        }
        let mut sum = 0.0_f64;
        let mut kept = 0;
        for (value, counts) in values.iter().zip(&mut tally.input_statistics) {
            if value.is_nan() {
                counts.masked_samples += 1;
                tally.masked_samples += 1;
            } else if available >= 3 && bounds.rejects(*value) {
                counts.rejected_samples += 1;
                tally.rejected_samples += 1;
            } else {
                sum += f64::from(*value);
                kept += 1;
                counts.accepted_samples += 1;
            }
        }
        tally.minimum_clean_samples = tally.minimum_clean_samples.min(kept);
        tally.maximum_clean_samples = tally.maximum_clean_samples.max(kept);
        tally.low_coverage_samples += u64::from(kept < 3);
        if minimum_clean.is_some_and(|minimum| kept < minimum) {
            tally.insufficient_samples += 1;
        }
        *output = if kept == 0 && minimum_clean.is_some() {
            // Never expose this incomplete image: the coverage error
            // below discards it, including on all-masked pixels.
            f32::NAN
        } else if kept == 0 {
            tally.fallback_pixels += 1;
            bounds.center as f32
        } else {
            (sum / kept as f64) as f32
        };
    }
    tally
}

fn tile_samples(samples: usize, frames: usize, budget: usize) -> Result<usize> {
    if samples == 0 || frames < 2 {
        return Err(Error::Calibration(
            "flat integration requires at least two nonempty frames".into(),
        ));
    }
    // Payload: every input's tile bytes, one more sample per pixel, and one
    // pixel's MAD workspace. The output image, small per-frame tallies, and
    // each worker's gather buffer of `COMBINE_CHUNK` pixels are separate.
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
    keys: &mut [i32],
    options: MasterRejectionOptions,
) -> RejectionBounds {
    // Scaling only extreme finite inputs prevents overflow in the f32
    // median/MAD arithmetic without changing ordinary normalized flats.
    let maximum = values
        .iter()
        .filter(|value| value.is_finite())
        .map(|value| value.abs())
        .fold(1.0, f32::max);
    let scale = if maximum > f32::MAX / 4.0 {
        maximum
    } else {
        1.0
    };
    let mut count = 0;
    for value in values.iter().filter(|value| value.is_finite()) {
        keys[count] = order_key(*value / scale);
        count += 1;
    }
    if count == 0 {
        return RejectionBounds {
            center: 0.0,
            low: 0.0,
            high: 0.0,
        };
    }
    let (center, mad) = median_and_deviation(&mut keys[..count]);
    let sigma = mad * seiza_stats::NORMAL_MAD_SCALE_F32;
    let center = f64::from(center) * f64::from(scale);
    let sigma = f64::from(sigma) * f64::from(scale);
    let tolerance = f64::from(f32::EPSILON) * center.abs().max(1.0) * 8.0;
    RejectionBounds {
        center,
        low: (f64::from(options.low_sigma) * sigma).max(tolerance),
        high: (f64::from(options.high_sigma) * sigma).max(tolerance),
    }
}

/// An integer that orders as `f32::total_cmp` orders the float. Integers sort
/// faster than floats compared through a closure, and the map is one to one,
/// so the sorted floats come back bit for bit.
fn order_key(value: f32) -> i32 {
    let bits = value.to_bits() as i32;
    bits ^ ((((bits >> 31) as u32) >> 1) as i32)
}

/// The float whose [`order_key`] this is.
fn from_order_key(key: i32) -> f32 {
    f32::from_bits((key ^ ((((key >> 31) as u32) >> 1) as i32)) as u32)
}

/// The median of a nonempty finite sample, given as [`order_key`]s, and the
/// median of its absolute deviations from that median: bit for bit what
/// `seiza_stats::median_in_place` and `robust_sigma_in_place` (before its
/// scale factor) return.
///
/// Both are order statistics, so any exact method gives the same bits. One
/// sort replaces two selections: in sorted order the deviations shrink toward
/// the median from below and grow away from it above, and rounding keeps both
/// runs monotone, so walking the two runs outward from the median meets the
/// deviations in ascending order.
fn median_and_deviation(keys: &mut [i32]) -> (f32, f32) {
    keys.sort_unstable();
    let value = |index: usize| from_order_key(keys[index]);
    let count = keys.len();
    let middle = count / 2;
    let even = count.is_multiple_of(2);
    let center = if even {
        (value(middle - 1) + value(middle)) * 0.5
    } else {
        value(middle)
    };
    let split = keys.partition_point(|key| from_order_key(*key) < center);
    let (mut below, mut above) = (split, split);
    let (mut previous, mut current) = (0.0, 0.0);
    for _ in 0..=middle {
        let take_below = below > 0
            && (above == count
                || (value(below - 1) - center).abs() <= (value(above) - center).abs());
        previous = current;
        current = if take_below {
            below -= 1;
            (value(below) - center).abs()
        } else {
            above += 1;
            (value(above - 1) - center).abs()
        };
    }
    let deviation = if even {
        (previous + current) * 0.5
    } else {
        current
    };
    (center, deviation)
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

    /// The flat combine as it ran on one thread before the per-pixel work
    /// moved onto rayon and the median/MAD onto one sort, minus the scratch
    /// file. Kept to show the new code changes no bit.
    mod serial {
        use super::super::{IntegratedFlat, RejectionBounds};
        use crate::master::{MasterBuildOptions, MasterInputStatistics, MasterRejectionOptions};
        use crate::{Error, Result};

        pub fn rejection_bounds(
            values: &[f32],
            statistics: &mut [f32],
            options: MasterRejectionOptions,
        ) -> RejectionBounds {
            let mut count = 0;
            for value in values.iter().filter(|value| value.is_finite()) {
                statistics[count] = *value;
                count += 1;
            }
            let statistics = &mut statistics[..count];
            if statistics.is_empty() {
                return RejectionBounds {
                    center: 0.0,
                    low: 0.0,
                    high: 0.0,
                };
            }
            let maximum = statistics
                .iter()
                .map(|value| value.abs())
                .fold(1.0, f32::max);
            let scale = if maximum > f32::MAX / 4.0 {
                maximum
            } else {
                1.0
            };
            for value in statistics.iter_mut() {
                *value /= scale;
            }
            let center =
                seiza_stats::median_in_place(statistics).expect("nonempty temporal sample");
            let sigma = seiza_stats::robust_sigma_in_place(statistics, center)
                .expect("nonempty temporal sample");
            let center = f64::from(center) * f64::from(scale);
            let sigma = f64::from(sigma) * f64::from(scale);
            let tolerance = f64::from(f32::EPSILON) * center.abs().max(1.0) * 8.0;
            RejectionBounds {
                center,
                low: (f64::from(options.low_sigma) * sigma).max(tolerance),
                high: (f64::from(options.high_sigma) * sigma).max(tolerance),
            }
        }

        pub fn integrate(
            frames: &[Vec<f32>],
            options: &MasterBuildOptions,
        ) -> Result<IntegratedFlat> {
            let samples = frames[0].len();
            let count = frames.len();
            let mut result = IntegratedFlat {
                samples: vec![0.0; samples],
                input_statistics: vec![
                    MasterInputStatistics {
                        accepted_samples: 0,
                        rejected_samples: 0,
                        masked_samples: 0,
                    };
                    count
                ],
                accepted_samples: 0,
                rejected_samples: 0,
                masked_samples: 0,
                masked_output_samples: 0,
                minimum_clean_samples: count,
                maximum_clean_samples: 0,
                low_coverage_samples: 0,
                fallback_pixels: 0,
            };
            let mut tile = vec![0.0; samples * count];
            for (frame, values) in frames.iter().enumerate() {
                for (index, value) in values.iter().enumerate() {
                    if !value.is_finite()
                        && !(options.flat_star_masking.is_some() && value.is_nan())
                    {
                        return Err(Error::FlatStarMasking(
                            "unexpected non-finite sample in flat scratch storage".into(),
                        ));
                    }
                    tile[index * count + frame] = *value;
                }
            }
            let mut statistics = vec![0.0; count];
            let mut insufficient_samples = 0_u64;
            for (index, values) in tile.chunks_exact(count).enumerate() {
                let bounds = rejection_bounds(values, &mut statistics, options.rejection);
                let available = values.iter().filter(|value| value.is_finite()).count();
                if available < count {
                    result.masked_output_samples += 1;
                }
                let mut sum = 0.0_f64;
                let mut kept = 0;
                for (value, counts) in values.iter().zip(&mut result.input_statistics) {
                    if value.is_nan() {
                        counts.masked_samples += 1;
                        result.masked_samples += 1;
                    } else if available >= 3 && bounds.rejects(*value) {
                        counts.rejected_samples += 1;
                        result.rejected_samples += 1;
                    } else {
                        sum += f64::from(*value);
                        kept += 1;
                        counts.accepted_samples += 1;
                    }
                }
                result.minimum_clean_samples = result.minimum_clean_samples.min(kept);
                result.maximum_clean_samples = result.maximum_clean_samples.max(kept);
                result.low_coverage_samples += u64::from(kept < 3);
                if options
                    .flat_star_masking
                    .as_ref()
                    .is_some_and(|masking| kept < masking.minimum_clean_samples)
                {
                    insufficient_samples += 1;
                }
                result.samples[index] = if kept == 0 && options.flat_star_masking.is_some() {
                    f32::NAN
                } else if kept == 0 {
                    result.fallback_pixels += 1;
                    bounds.center as f32
                } else {
                    (sum / kept as f64) as f32
                };
            }
            result.accepted_samples =
                (samples * count) as u64 - result.rejected_samples - result.masked_samples;
            if insufficient_samples > 0 {
                return Err(Error::InsufficientFlatCoverage {
                    insufficient_samples,
                    required_clean_samples: options
                        .flat_star_masking
                        .as_ref()
                        .unwrap()
                        .minimum_clean_samples,
                    minimum_clean_samples: result.minimum_clean_samples,
                    maximum_clean_samples: result.maximum_clean_samples,
                    masked_samples: result.masked_samples,
                });
            }
            Ok(result)
        }
    }

    struct Random(u64);

    impl Random {
        fn unit(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1_u64 << 24) as f32
        }
    }

    /// Normalized sky flats: response near one with noise, a star crossing
    /// some frames, a few pixels every frame disagrees on, identical samples,
    /// and, when `masked`, star masks (NaN) over parts of some frames.
    fn sky_flats(frames: usize, pixels: usize, masked: bool) -> Vec<Vec<f32>> {
        let mut random = Random(0x2545_f491_4f6c_dd1d);
        (0..frames)
            .map(|frame| {
                (0..pixels)
                    .map(|pixel| {
                        let noise = (random.unit() - 0.5) * 0.004;
                        let response = 0.9 + (pixel % 977) as f32 * 1.0e-4;
                        match pixel % 89 {
                            _ if masked && (pixel + 3 * frame) % 41 == 0 => f32::NAN,
                            0 => 1.0,
                            1 => frame as f32 * 0.25 + noise,
                            2 if frame % 3 == 0 => response + 0.7 + noise,
                            _ if (pixel / 5 + frame * 11) % 53 == 0 => response + 1.5,
                            _ => response + noise,
                        }
                    })
                    .collect()
            })
            .collect()
    }

    fn assert_same_flat(actual: &IntegratedFlat, expected: &IntegratedFlat) {
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&actual.samples), bits(&expected.samples));
        assert_eq!(actual.input_statistics, expected.input_statistics);
        assert_eq!(actual.accepted_samples, expected.accepted_samples);
        assert_eq!(actual.rejected_samples, expected.rejected_samples);
        assert_eq!(actual.masked_samples, expected.masked_samples);
        assert_eq!(actual.masked_output_samples, expected.masked_output_samples);
        assert_eq!(actual.minimum_clean_samples, expected.minimum_clean_samples);
        assert_eq!(actual.maximum_clean_samples, expected.maximum_clean_samples);
        assert_eq!(actual.low_coverage_samples, expected.low_coverage_samples);
        assert_eq!(actual.fallback_pixels, expected.fallback_pixels);
    }

    #[test]
    fn parallel_flat_combine_matches_the_serial_combine_bit_for_bit() {
        // More pixels than one batch, so batches, chunks and tiles all split.
        let pixels = COMBINE_BATCH + 3 * COMBINE_CHUNK + 17;
        for (frames, masked, sigma) in [
            (2, false, 3.0),
            (3, false, 3.0),
            (8, false, 3.0),
            (8, false, 0.1),
            (9, true, 3.0),
            (31, false, 2.0),
        ] {
            let inputs = sky_flats(frames, pixels, masked);
            let options = MasterBuildOptions {
                rejection: MasterRejectionOptions {
                    low_sigma: sigma,
                    high_sigma: sigma,
                },
                flat_star_masking: masked.then(super::super::FlatStarMaskingOptions::default),
                ..Default::default()
            };
            let expected = serial::integrate(&inputs, &options).unwrap();
            if frames >= 3 {
                assert!(expected.rejected_samples > 0);
            }
            if sigma < 1.0 {
                assert!(expected.fallback_pixels > 0);
            }
            if masked {
                assert!(expected.masked_samples > 0);
            }
            for budget in [TILE_BYTES, (frames + 1) * 4 * 10_007 + frames * 4] {
                let directory = tempfile::tempdir().unwrap();
                let mut scratch = FlatScratch::new(Some(directory.path())).unwrap();
                for frame in &inputs {
                    scratch.append(frame, &options).unwrap();
                }
                let actual = scratch.integrate_with_budget(&options, budget).unwrap();
                assert_same_flat(&actual, &expected);
            }
        }
    }

    #[test]
    fn one_sort_finds_the_selected_median_and_deviation_bit_for_bit() {
        let mut random = Random(0x9e37_79b9_7f4a_7c15);
        let options = MasterRejectionOptions::default();
        let mut keys = vec![0; 80];
        let mut statistics = vec![0.0; 80];
        for trial in 0..20_000 {
            let count = 1 + trial % 79;
            let values = (0..count)
                .map(|index| {
                    let unit = random.unit();
                    match trial % 9 {
                        // Ordinary normalized flats.
                        0 | 1 => 1.0 + (unit - 0.5) * 0.01,
                        // Few distinct values: ties at the median and in the
                        // deviations.
                        2 => (unit * 4.0).floor() * 0.25,
                        // Both zeros, which order apart but compare equal.
                        3 => [0.0, -0.0, 1.0, -1.0][index % 4] * unit.round(),
                        // Mixed signs and magnitudes.
                        4 => (unit - 0.5) * 10_f32.powi((index % 13) as i32 - 6),
                        // Values near the overflow guard.
                        5 => (unit - 0.3) * f32::MAX,
                        // Subnormals.
                        6 => unit * f32::MIN_POSITIVE,
                        // Masked samples among real ones.
                        7 if index % 3 == 0 => f32::NAN,
                        _ => 1.0 + unit,
                    }
                })
                .collect::<Vec<_>>();
            let actual = rejection_bounds(&values, &mut keys, options);
            let expected = serial::rejection_bounds(&values, &mut statistics, options);
            assert_eq!(
                (
                    actual.center.to_bits(),
                    actual.low.to_bits(),
                    actual.high.to_bits()
                ),
                (
                    expected.center.to_bits(),
                    expected.low.to_bits(),
                    expected.high.to_bits()
                ),
                "{values:?}"
            );
        }
    }

    #[test]
    fn order_keys_sort_as_total_cmp_and_round_trip() {
        let mut values = vec![
            f32::NEG_INFINITY,
            -f32::MAX,
            -1.0,
            -f32::MIN_POSITIVE,
            -1.0e-45,
            -0.0,
            0.0,
            1.0e-45,
            f32::MIN_POSITIVE,
            1.0,
            f32::MAX,
            f32::INFINITY,
        ];
        let keys = values
            .iter()
            .map(|value| order_key(*value))
            .collect::<Vec<_>>();
        assert!(keys.windows(2).all(|pair| pair[0] < pair[1]), "{keys:?}");
        for (key, value) in keys.iter().zip(&values) {
            assert_eq!(from_order_key(*key).to_bits(), value.to_bits());
        }
        values.reverse();
        values.sort_unstable_by_key(|value| order_key(*value));
        let mut by_total_cmp = values.clone();
        by_total_cmp.sort_unstable_by(f32::total_cmp);
        assert_eq!(
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            by_total_cmp
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
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
        let mut workspace = [0; 6];
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
    fn combining_reports_each_tile_and_then_the_end() {
        let directory = tempfile::tempdir().unwrap();
        let mut scratch = FlatScratch::new(Some(directory.path())).unwrap();
        let reports = Arc::new(std::sync::Mutex::new(Vec::new()));
        let options = MasterBuildOptions {
            progress: Some(crate::MasterProgress::new({
                let reports = Arc::clone(&reports);
                move |progress| reports.lock().unwrap().push(progress)
            })),
            ..MasterBuildOptions::default()
        };
        for offset in 0..3 {
            let frame = (0..10)
                .map(|index| 1.0 + (index + offset) as f32 * 0.01)
                .collect::<Vec<_>>();
            scratch.append(&frame, &options).unwrap();
        }
        let budget = 76;
        let tiles = 10_usize.div_ceil(tile_samples(10, 3, budget).unwrap());
        assert!(tiles > 1, "the budget must split the image");
        scratch.integrate_with_budget(&options, budget).unwrap();
        let reports = reports.lock().unwrap();
        let expected = (0..=tiles)
            .map(|done| super::super::MasterBuildProgress {
                stage: MasterBuildStage::Combine,
                done,
                total: tiles,
            })
            .collect::<Vec<_>>();
        assert_eq!(*reports, expected);
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
    fn masks_partition_counts_and_two_retained_samples_are_not_clipped() {
        let directory = tempfile::tempdir().unwrap();
        let mut scratch = FlatScratch::new(Some(directory.path())).unwrap();
        let options = MasterBuildOptions {
            flat_star_masking: Some(super::super::FlatStarMaskingOptions::default()),
            ..Default::default()
        };
        for values in [[1.0, 1.0], [1.0, 1.0], [f32::NAN, 10.0], [f32::NAN, 1.0]] {
            scratch.append(&values, &options).unwrap();
        }
        let result = scratch.integrate_with_budget(&options, 36).unwrap();
        assert_eq!(result.samples, [1.0, 1.0]);
        assert_eq!(result.masked_samples, 2);
        assert_eq!(result.rejected_samples, 1);
        assert_eq!(result.accepted_samples, 5);
        assert_eq!(result.masked_output_samples, 1);
        assert_eq!(result.minimum_clean_samples, 2);
        assert_eq!(result.maximum_clean_samples, 3);
        assert_eq!(result.low_coverage_samples, 1);
        assert_eq!(result.input_statistics[2].masked_samples, 1);
        assert_eq!(result.input_statistics[2].rejected_samples, 1);
        assert_eq!(directory.path().read_dir().unwrap().count(), 0);
    }

    #[test]
    fn insufficient_masked_or_post_clipped_coverage_fails_without_a_fallback() {
        for values in [
            vec![f32::NAN; 4],
            vec![1.0, f32::NAN, f32::NAN, f32::NAN],
            vec![0.8, 1.0, 1.2, 10.0],
        ] {
            let directory = tempfile::tempdir().unwrap();
            let mut scratch = FlatScratch::new(Some(directory.path())).unwrap();
            let options = MasterBuildOptions {
                flat_star_masking: Some(super::super::FlatStarMaskingOptions::default()),
                rejection: MasterRejectionOptions {
                    low_sigma: 0.1,
                    high_sigma: 0.1,
                },
                ..Default::default()
            };
            for value in values {
                scratch.append(&[value], &options).unwrap();
            }
            assert!(matches!(
                scratch.integrate(&options),
                Err(Error::InsufficientFlatCoverage {
                    insufficient_samples: 1,
                    required_clean_samples: 2,
                    ..
                })
            ));
            assert_eq!(directory.path().read_dir().unwrap().count(), 0);
        }
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
