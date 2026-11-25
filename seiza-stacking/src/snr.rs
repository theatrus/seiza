//! Reading how deep a stack is, while it is being built.
//!
//! An accumulator already holds the integration of every frame pushed so far,
//! so measuring how the noise falls with depth is a matter of looking at that
//! running mean a few times on the way past — not of integrating the same
//! frames again once per depth. [`checkpoint_depths`] picks the depths a
//! caller should look at, and [`measure_depth`] reads the accumulator without
//! copying it.
//!
//! This module measures. It draws no conclusions: what the curve means —
//! whether more frames are worth shooting, where the returns flattened — is
//! left to the caller, whose exposure model and product vocabulary this crate
//! has no business guessing at.
//!
//! # What a measurement contains
//!
//! - **Noise** is the robust spread of second differences taken along both
//!   image axes, scaled to a standard deviation. Second differences cancel a
//!   planar sky gradient, the median throws the stars away, and keeping the
//!   noisier axis leaves row or column banding visible instead of averaging it
//!   out. What is left is the pixel-scale noise of the integration — the
//!   quantity that falls as the square root of the frame count when averaging
//!   is working.
//! - **Background** is the median sample.
//! - **Signal** is how far the brightest one percent of the frame sits above
//!   that background. The fraction is fixed rather than a multiple of the
//!   noise, so the same part of the sky is measured at every depth; a
//!   threshold that moved with the noise would widen as the stack deepened and
//!   write its own trend into the result.
//!
//! Statistics come from a capped subsample of rows, so a sixty-megapixel stack
//! costs what a six-megapixel one costs.
//!
//! # Comparing depths
//!
//! The brightest-percent statistic is itself lifted by noise where a stack is
//! shallow — a one-frame stack's brightest percent is part star and part noise
//! peak. A caller comparing depths should therefore read every depth's ratio
//! against one signal, normally the deepest measured one, so the shape of the
//! curve comes from the noise alone. [`SnrSample::snr`] is offered for a
//! single reading; it is not the right thing to plot across depths.

use crate::LinearImage;
use crate::stack::StackView;
use rayon::prelude::*;
use seiza_stats::{NORMAL_MAD_SCALE, median_in_place};
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;

/// Rows read per measurement, at most. Noise and level statistics converge
/// long before a full frame is consumed, and this keeps the cost of a
/// measurement flat across sensor sizes.
const MAX_SAMPLED_ROWS: usize = 512;

/// Row and column pairs [`photosite_noise`] reads, at most.
const PHOTOSITE_ROW_PAIRS: usize = 128;
const PHOTOSITE_COLUMN_PAIRS: usize = 512;

/// Fewer samples than this and the medians are not worth reporting.
const MIN_SAMPLES: usize = 1024;

/// The brightest share of the frame that stands in for the target's signal.
/// Public because a caller plotting the curve may want to say what the signal
/// figure actually measures.
pub const SIGNAL_FRACTION: f64 = 0.01;

/// A second difference has coefficients 1, -2, 1, so independent samples with
/// sigma noise produce a difference with sqrt(6) sigma.
const SECOND_DIFFERENCE_GAIN_SQUARED: f64 = 6.0;

/// One reading of an accumulator.
///
/// Units are the stack's own: whatever scale the frames were integrated on.
/// Only ratios between readings of the same stack mean anything.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SnrSample {
    /// Frames the accumulator had taken when this was measured.
    pub frames: u32,
    /// Pixel-scale noise of the integration, averaged across channels.
    pub noise: f64,
    /// Median sample, averaged across channels.
    pub background: f64,
    /// How far the brightest [`SIGNAL_FRACTION`] of the frame sits above the
    /// background, averaged across channels.
    pub signal: f64,
    /// Per-channel noise: one entry for mono, three for a debayered stack.
    /// Worth looking at on colour data, where one channel is often noisier
    /// than the others.
    pub channel_noise: Vec<f64>,
    /// The factor the stack's lights were binned by (see
    /// [`crate::StackOptions::bin_factor`]). Binning by `b` averages
    /// `b²` samples into each pixel, so its noise falls by about `b`: noise
    /// and SNR compare only between readings of one bin factor, while the
    /// shape of the depth curve compares across them.
    /// [`crate::LiveStacker::measure_depth`] gives the stack's; [`measure_depth`]
    /// of a bare [`StackView`] cannot know it, and gives 1.
    #[serde(default = "native_bin_factor")]
    pub bin_factor: u32,
}

fn native_bin_factor() -> u32 {
    1
}

impl SnrSample {
    /// This reading's signal against its own noise.
    ///
    /// Correct for one depth. To compare depths, read each one against a
    /// single signal instead — see the module documentation.
    pub fn snr(&self) -> f64 {
        if self.noise > 0.0 {
            self.signal / self.noise
        } else {
            0.0
        }
    }
}

/// The depths a build of `total` frames should measure at: the doubling
/// ladder, and the full set.
///
/// Doubling keeps the count of measurements logarithmic, so a five-hundred
/// frame stack is interrupted nine times rather than five hundred, and the
/// points still spread evenly once the curve is drawn against a log axis.
///
/// Measuring is cheap, but reaching the accumulator between pushes is not
/// free for a caller that pipelines its frame preparation: every depth is a
/// batch boundary. That is the cost this ladder is shaped to keep small.
pub fn checkpoint_depths(total: usize) -> Vec<usize> {
    let mut depths = Vec::new();
    let mut depth = 1usize;
    while depth < total {
        depths.push(depth);
        // Saturating, not wrapping: `total` is an unconstrained `usize`, and
        // doubling past the top would wrap to zero and loop forever pushing
        // zeros. Unreachable from a real frame count, but this is public.
        depth = depth.saturating_mul(2);
        if depth == usize::MAX {
            break;
        }
    }
    if total > 0 {
        depths.push(total);
    }
    depths
}

/// Read a live accumulator.
///
/// `None` when the frame is too small, when too little of it is covered to
/// measure, or when the samples carry no noise at all — all of which are
/// ordinary answers early in a build, not errors.
///
/// A [`StackView`] whose dimensions do not describe its slices also answers
/// `None`. Its fields are public, so nothing stops a caller assembling one by
/// hand, and this reads neighbouring samples by arithmetic on those
/// dimensions: a library should decline such a view rather than panic inside
/// somebody else's stack frame.
pub fn measure_depth(view: StackView<'_>) -> Option<SnrSample> {
    if view.width < 3 || view.height < 3 || view.channels == 0 {
        return None;
    }
    let channels = view.channels;
    let samples = view
        .width
        .checked_mul(view.height)
        .and_then(|pixels| pixels.checked_mul(channels))?;
    if view.mean.len() != samples || view.coverage.len() != samples {
        return None;
    }
    let interior_rows = view.height - 2;
    let stride = interior_rows.div_ceil(MAX_SAMPLED_ROWS).max(1);
    let mut channel_noise = Vec::with_capacity(channels);
    let mut channel_background = Vec::with_capacity(channels);
    let mut channel_signal = Vec::with_capacity(channels);

    for channel in 0..channels {
        let mut levels: Vec<f32> = Vec::new();
        let mut horizontal: Vec<f32> = Vec::new();
        let mut vertical: Vec<f32> = Vec::new();
        let mut y = 1usize;
        while y + 1 < view.height {
            let row = y * view.width;
            for x in 1..view.width - 1 {
                let index = (row + x) * channels + channel;
                let coverage = view.coverage[index];
                let value = view.mean[index];
                if coverage == 0 || !value.is_finite() {
                    continue;
                }
                levels.push(value);

                // A second difference removes any linear gradient. All three
                // samples need equal coverage so a dithered edge, where one
                // neighbour is thinner than the others, cannot masquerade as
                // pixel structure.
                let left = index - channels;
                let right = index + channels;
                if view.coverage[left] == coverage
                    && view.coverage[right] == coverage
                    && view.mean[left].is_finite()
                    && view.mean[right].is_finite()
                {
                    horizontal.push(view.mean[left] - 2.0 * value + view.mean[right]);
                }

                let above = index - view.width * channels;
                let below = index + view.width * channels;
                if view.coverage[above] == coverage
                    && view.coverage[below] == coverage
                    && view.mean[above].is_finite()
                    && view.mean[below].is_finite()
                {
                    vertical.push(view.mean[above] - 2.0 * value + view.mean[below]);
                }
            }
            y += stride;
        }

        if horizontal.len() < MIN_SAMPLES
            || vertical.len() < MIN_SAMPLES
            || levels.len() < MIN_SAMPLES
        {
            return None;
        }
        // Keep the noisier axis: that makes horizontal and vertical banding
        // equally visible instead of averaging either one away.
        let noise =
            second_difference_noise(&mut horizontal)?.max(second_difference_noise(&mut vertical)?);
        if noise <= 0.0 || !noise.is_finite() {
            return None;
        }
        let background = f64::from(median_in_place(&mut levels)?);
        channel_noise.push(noise);
        channel_background.push(background);
        channel_signal.push(brightest_above(&mut levels, background));
    }

    Some(SnrSample {
        frames: view.accepted_frames,
        noise: mean(&channel_noise),
        background: mean(&channel_background),
        signal: mean(&channel_signal),
        channel_noise,
        bin_factor: native_bin_factor(),
    })
}

/// Per-channel pixel-scale noise of one image, as a standard deviation in the
/// image's own units.
///
/// This is the estimator [`crate::FrameWeighting::InverseNoiseVariance`]
/// uses to weight frames, and the one [`measure_depth`] uses on a stack: the
/// median absolute deviation of second differences taken along each axis,
/// scaled to a Gaussian standard deviation, keeping the noisier axis. Second
/// differences cancel a sky gradient and the median ignores stars. At most
/// 512 rows are read, so the cost does not grow with sensor height.
/// Non-finite samples, and differences that touch one, are skipped.
///
/// It reads neighbouring pixels, so it measures noise that is independent
/// from one to the next. Noise they share, as after drizzle or resampling,
/// reads low; [`noise_at_scale`] measures it.
///
/// The result has one entry per channel, interleaved channel order. It is
/// `None` when the image is smaller than 3 by 3 pixels, its buffer does not
/// match its dimensions, or any channel has fewer than 1024 usable
/// differences on either axis.
pub fn frame_noise(image: &LinearImage) -> Option<Vec<f32>> {
    let (width, height, channels) = (image.width, image.height, image.channels);
    if width < 3 || height < 3 || channels == 0 {
        return None;
    }
    if width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(channels))
        != Some(image.data.len())
    {
        return None;
    }
    let data = &image.data;
    let stride = (height - 2).div_ceil(MAX_SAMPLED_ROWS).max(1);
    (0..channels)
        .into_par_iter()
        .map(|channel| {
            // Room for every sampled row, so the vectors are not copied as
            // they grow.
            let rows = (height - 2).div_ceil(stride);
            let mut horizontal: Vec<f32> = Vec::with_capacity(rows * (width - 2));
            let mut vertical: Vec<f32> = Vec::with_capacity(rows * (width - 2));
            let mut y = 1usize;
            while y + 1 < height {
                let row = y * width;
                for x in 1..width - 1 {
                    let index = (row + x) * channels + channel;
                    let value = data[index];
                    if !value.is_finite() {
                        continue;
                    }
                    let (left, right) = (data[index - channels], data[index + channels]);
                    if left.is_finite() && right.is_finite() {
                        horizontal.push(left - 2.0 * value + right);
                    }
                    let (above, below) = (
                        data[index - width * channels],
                        data[index + width * channels],
                    );
                    if above.is_finite() && below.is_finite() {
                        vertical.push(above - 2.0 * value + below);
                    }
                }
                y += stride;
            }
            let noise = second_difference_noise(&mut horizontal)?
                .max(second_difference_noise(&mut vertical)?);
            Some(noise as f32)
        })
        .collect()
}

/// Per-channel noise of an image at the scale of `block` by `block` pixels,
/// as the standard deviation of those blocks' means in the image's own units.
///
/// [`frame_noise`] reads neighbouring pixels, so it under-reads noise that
/// neighbours share, which drizzle and resampling leave: one 2x drizzle at
/// full drop size read 2.7 times low. Means of blocks larger than the shared
/// reach take the correlation in. The same blocks on the sky, 8 pixels of a
/// 2x drizzle or 4 of the native stack, then compare the two. On that
/// drizzle, 4-pixel blocks read 16% low, 8-pixel blocks 5 to 8%, 16-pixel
/// blocks 5%: choose blocks several times the resampling's reach.
///
/// For independent noise it is [`frame_noise`] divided by `block`, and
/// [`frame_noise`] itself at 1. Above 1 the second differences are between
/// blocks two apart, since neighbouring blocks share the noise at their
/// common edge. The median passes over stars, but structure that covers much
/// of the image and curves within four blocks reads as noise: subtract a
/// model or a deeper image first, or measure a region of sky.
///
/// A block with a non-finite sample is left out. `None` as for
/// [`frame_noise`], with blocks for pixels: each axis needs 1024 usable
/// differences, about 36 by 36 blocks.
pub fn noise_at_scale(image: &LinearImage, block: NonZeroUsize) -> Option<Vec<f32>> {
    let block = block.get();
    if block == 1 {
        return frame_noise(image);
    }
    let (width, height, channels) = (image.width, image.height, image.channels);
    let samples = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(channels));
    if channels == 0 || samples != Some(image.data.len()) {
        return None;
    }
    (0..channels)
        .into_par_iter()
        .map(|channel| {
            let means = block_means(image, channel, block, None);
            grid_noise(&means, width / block, height / block).map(|noise| noise as f32)
        })
        .collect()
}

/// The means of one channel's whole blocks of `block` by `block` pixels, row
/// by row, NaN where a block holds a non-finite sample or a pixel `mask`, one
/// entry per pixel, sets.
pub(crate) fn block_means(
    image: &LinearImage,
    channel: usize,
    block: usize,
    mask: Option<&[bool]>,
) -> Vec<f32> {
    let (columns, rows) = (image.width / block, image.height / block);
    let area = block as f64 * block as f64;
    let mut means = vec![f32::NAN; columns * rows];
    if columns == 0 {
        return means;
    }
    means
        .par_chunks_mut(columns)
        .enumerate()
        .for_each(|(row, means)| {
            let mut sums = vec![0.0_f64; columns];
            for y in row * block..(row + 1) * block {
                for (column, sum) in sums.iter_mut().enumerate() {
                    for x in column * block..(column + 1) * block {
                        let pixel = y * image.width + x;
                        // A NaN or infinite sample leaves the sum so.
                        *sum += match mask {
                            Some(mask) if mask[pixel] => f64::NAN,
                            _ => f64::from(image.data[pixel * image.channels + channel]),
                        };
                    }
                }
            }
            for (mean, sum) in means.iter_mut().zip(sums) {
                if sum.is_finite() {
                    *mean = (sum / area) as f32;
                }
            }
        });
    means
}

/// The noise of a grid of block means, from second differences between
/// blocks two apart, keeping the noisier axis.
pub(crate) fn grid_noise(means: &[f32], columns: usize, rows: usize) -> Option<f64> {
    const GAP: usize = 2;
    if columns <= 2 * GAP || rows <= 2 * GAP {
        return None;
    }
    let mut horizontal = Vec::with_capacity(means.len());
    let mut vertical = Vec::with_capacity(means.len());
    for (index, &value) in means.iter().enumerate() {
        let (x, y) = (index % columns, index / columns);
        if x >= GAP && x + GAP < columns {
            horizontal.push(means[index - GAP] - 2.0 * value + means[index + GAP]);
        }
        if y >= GAP && y + GAP < rows {
            let reach = GAP * columns;
            vertical.push(means[index - reach] - 2.0 * value + means[index + reach]);
        }
    }
    horizontal.retain(|difference| difference.is_finite());
    vertical.retain(|difference| difference.is_finite());
    let noise =
        second_difference_noise(&mut horizontal)?.max(second_difference_noise(&mut vertical)?);
    Some(noise)
}

/// Per-channel noise of the samples an image was built from, for the
/// Lanczos-3 clamp's noise gate.
///
/// [`frame_noise`] reads neighbouring pixels, which a demosaiced frame has
/// correlated: through a bilinear demosaic, red and blue read almost no
/// noise, since most of their interpolated samples have no second
/// difference at all, and through Malvar-He-Cutler's about 0.67 of the
/// photosites' noise. A demosaic leaves each colour's own photosites as
/// they were, every second pixel on both axes, so this takes second
/// differences two pixels apart, within each of the four row and column
/// parities separately, and keeps the noisiest parity and axis. For a
/// demosaiced channel that is the parity of its photosites, whose noise is
/// the sensor's; interpolated parities read lower and are passed over. For a
/// mono or planar frame every parity reads the same noise, and a row or
/// column pattern alternating every pixel cancels.
///
/// At most 256 rows and 1024 columns are read, in adjacent pairs so every
/// parity is seen, about 65 000 differences per parity and axis: enough to
/// fix the noise to a fraction of a percent, at a cost that does not grow
/// with the sensor. Non-finite samples, and differences that touch one, are
/// skipped. `None` when the image is smaller than 5 by 5 pixels, its buffer
/// does not match its dimensions, or any parity of any channel has fewer
/// than 1024 usable differences on either axis.
pub(crate) fn photosite_noise(image: &LinearImage) -> Option<Vec<f32>> {
    let (width, height, channels) = (image.width, image.height, image.channels);
    if width < 5 || height < 5 || channels == 0 {
        return None;
    }
    if width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(channels))
        != Some(image.data.len())
    {
        return None;
    }
    let data = &image.data;
    let (step_x, step_y) = (2 * channels, 2 * width * channels);
    // Rows 2 + 2p and 3 + 2p for pair p, each with two rows above and below,
    // and columns likewise.
    let (row_pairs, column_pairs) = ((height - 4) / 2, (width - 4) / 2);
    let row_stride = row_pairs.div_ceil(PHOTOSITE_ROW_PAIRS).max(1);
    let column_stride = column_pairs.div_ceil(PHOTOSITE_COLUMN_PAIRS).max(1);
    (0..channels)
        .into_par_iter()
        .map(|channel| {
            // By parity, (y % 2) * 2 + x % 2, then axis.
            let mut differences: [[Vec<f32>; 2]; 4] = Default::default();
            for row_pair in (0..row_pairs).step_by(row_stride) {
                for y in [2 + 2 * row_pair, 3 + 2 * row_pair] {
                    for x in (0..column_pairs)
                        .step_by(column_stride)
                        .flat_map(|pair| [2 + 2 * pair, 3 + 2 * pair])
                    {
                        let index = (y * width + x) * channels + channel;
                        let value = data[index];
                        if !value.is_finite() {
                            continue;
                        }
                        let parity = &mut differences[(y % 2) * 2 + x % 2];
                        let (left, right) = (data[index - step_x], data[index + step_x]);
                        if left.is_finite() && right.is_finite() {
                            parity[0].push(left - 2.0 * value + right);
                        }
                        let (above, below) = (data[index - step_y], data[index + step_y]);
                        if above.is_finite() && below.is_finite() {
                            parity[1].push(above - 2.0 * value + below);
                        }
                    }
                }
            }
            let mut noise = 0.0_f64;
            for axis in differences.iter_mut().flatten() {
                noise = noise.max(second_difference_noise(axis)?);
            }
            Some(noise as f32)
        })
        .collect()
}

/// The standard deviation the second differences imply, or `None` when the
/// sample is too small or degenerate.
fn second_difference_noise(differences: &mut [f32]) -> Option<f64> {
    if differences.len() < MIN_SAMPLES {
        return None;
    }
    let center = median_in_place(differences)?;
    for difference in differences.iter_mut() {
        *difference = (*difference - center).abs();
    }
    let deviation = f64::from(median_in_place(differences)?);
    let noise = deviation * NORMAL_MAD_SCALE / SECOND_DIFFERENCE_GAIN_SQUARED.sqrt();
    (noise >= 0.0 && noise.is_finite()).then_some(noise)
}

/// How far the brightest [`SIGNAL_FRACTION`] of the samples sits above the
/// background. Reorders `values`.
fn brightest_above(values: &mut [f32], background: f64) -> f64 {
    let cut = ((values.len() as f64) * (1.0 - SIGNAL_FRACTION)) as usize;
    let cut = cut.min(values.len().saturating_sub(1));
    let (_, pivot, brightest) = values.select_nth_unstable_by(cut, f32::total_cmp);
    let mut total = f64::from(*pivot);
    let mut count = 1usize;
    for value in brightest {
        total += f64::from(*value);
        count += 1;
    }
    (total / count as f64 - background).max(0.0)
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic normal generator, so a measurement can be checked
    /// against a number the test chose.
    struct Noise(u64);

    impl Noise {
        fn next_uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64).mul_add(0.999_999, 5e-7)
        }

        fn next_normal(&mut self) -> f64 {
            let first = self.next_uniform();
            let second = self.next_uniform();
            (-2.0 * first.ln()).sqrt() * (std::f64::consts::TAU * second).cos()
        }
    }

    struct Frame {
        data: Vec<f32>,
        coverage: Vec<u32>,
        width: usize,
        height: usize,
    }

    impl Frame {
        fn view(&self, frames: u32) -> StackView<'_> {
            StackView {
                width: self.width,
                height: self.height,
                channels: 1,
                mean: &self.data,
                coverage: &self.coverage,
                rejected_samples: &self.coverage,
                accepted_frames: frames,
                rejected_frames: 0,
            }
        }
    }

    /// A flat field with the given noise, a sky gradient, and a bright patch
    /// standing in for a target.
    fn frame(width: usize, height: usize, sigma: f64, seed: u64) -> Frame {
        let mut noise = Noise(seed);
        let mut data = vec![0.0f32; width * height];
        for (index, sample) in data.iter_mut().enumerate() {
            let (x, y) = (index % width, index / width);
            let object = if y < height / 10 { 500.0 } else { 0.0 };
            let gradient = x as f32 * 0.05 + y as f32 * 0.03;
            *sample = 1000.0 + object + gradient + (noise.next_normal() * sigma) as f32;
        }
        Frame {
            coverage: vec![1; data.len()],
            data,
            width,
            height,
        }
    }

    #[test]
    fn checkpoint_depths_double_and_end_on_the_full_set() {
        assert_eq!(checkpoint_depths(0), Vec::<usize>::new());
        assert_eq!(checkpoint_depths(1), vec![1]);
        assert_eq!(checkpoint_depths(5), vec![1, 2, 4, 5]);
        assert_eq!(checkpoint_depths(16), vec![1, 2, 4, 8, 16]);
        assert_eq!(checkpoint_depths(100), vec![1, 2, 4, 8, 16, 32, 64, 100]);
        // The whole point of the ladder: measuring is cheap, but every depth
        // costs a caller a batch boundary.
        assert_eq!(checkpoint_depths(500).len(), 10);
        // An unconstrained `usize` must not wrap past the top and loop.
        assert!(checkpoint_depths(usize::MAX).len() < 70);
    }

    #[test]
    fn noise_is_measured_through_a_sky_gradient_and_a_bright_target() {
        let field = frame(400, 400, 12.0, 7);
        let sample = measure_depth(field.view(1)).expect("measurable");
        assert!(
            (sample.noise - 12.0).abs() < 0.6,
            "measured {} for a sigma of 12",
            sample.noise
        );
        // The gradient runs across the frame and the target sits 500 above it;
        // neither may leak into the noise.
        assert!(sample.signal > 400.0, "{}", sample.signal);
        assert!(sample.snr() > 30.0, "{}", sample.snr());
    }

    #[test]
    fn averaging_four_times_the_frames_halves_the_measured_noise() {
        let shallow = frame(400, 400, 20.0, 11);
        let deep = frame(400, 400, 10.0, 13);
        let shallow = measure_depth(shallow.view(4)).expect("shallow");
        let deep = measure_depth(deep.view(16)).expect("deep");
        let ratio = deep.noise / shallow.noise;
        assert!((ratio - 0.5).abs() < 0.05, "ratio {ratio}");
    }

    #[test]
    fn row_banding_is_not_averaged_away_by_the_other_axis() {
        // Banding is structure along one axis only. Measuring a single axis,
        // or averaging the two, would hide it.
        let mut field = frame(400, 400, 4.0, 17);
        for y in 0..field.height {
            if y % 2 == 0 {
                continue;
            }
            for x in 0..field.width {
                field.data[y * field.width + x] += 30.0;
            }
        }
        let sample = measure_depth(field.view(8)).expect("measurable");
        assert!(
            sample.noise > 12.0,
            "banding should dominate the reading, got {}",
            sample.noise
        );
    }

    #[test]
    fn uncovered_samples_are_not_measured() {
        let mut field = frame(400, 400, 12.0, 19);
        // Half the frame never had a frame land on it. Those samples are zero
        // and would read as enormous noise if they were counted.
        for index in 0..field.data.len() {
            if index % field.width >= field.width / 2 {
                field.coverage[index] = 0;
            }
        }
        let sample = measure_depth(field.view(4)).expect("measurable");
        assert!((sample.noise - 12.0).abs() < 0.8, "{}", sample.noise);
    }

    #[test]
    fn a_thin_dithered_edge_does_not_read_as_pixel_structure() {
        // One neighbour covered by fewer frames is noisier for a reason that
        // is not pixel structure, so the difference is skipped.
        let mut field = frame(400, 400, 8.0, 23);
        for y in 0..field.height {
            let index = y * field.width + field.width / 2;
            field.coverage[index] = 1;
            field.data[index] += 400.0;
        }
        for coverage in field.coverage.iter_mut() {
            if *coverage != 1 {
                *coverage = 4;
            }
        }
        for (index, coverage) in field.coverage.iter_mut().enumerate() {
            if index % field.width != field.width / 2 {
                *coverage = 4;
            }
        }
        let sample = measure_depth(field.view(4)).expect("measurable");
        assert!((sample.noise - 8.0).abs() < 0.8, "{}", sample.noise);
    }

    #[test]
    fn a_frame_too_small_or_too_empty_is_not_measured() {
        let tiny = frame(2, 2, 12.0, 29);
        assert!(measure_depth(tiny.view(1)).is_none());
        let small = frame(20, 20, 12.0, 31);
        assert!(measure_depth(small.view(1)).is_none(), "too few samples");
    }

    #[test]
    fn a_view_that_does_not_describe_its_own_slices_is_declined() {
        // `StackView` has public fields, so a caller can assemble one whose
        // dimensions and buffers disagree. Reading neighbours is arithmetic on
        // those dimensions, so this has to be refused rather than indexed.
        let field = frame(400, 400, 12.0, 37);
        let honest = field.view(4);
        assert!(
            measure_depth(honest).is_some(),
            "the truthful view measures"
        );

        let lying = |width, height, channels| StackView {
            width,
            height,
            channels,
            mean: &field.data,
            coverage: &field.coverage,
            rejected_samples: &field.coverage,
            accepted_frames: 4,
            rejected_frames: 0,
        };
        // Dimensions larger than the buffer, a channel count the buffer cannot
        // hold, and a zero channel count that `max(1)` used to paper over.
        assert!(measure_depth(lying(4000, 400, 1)).is_none());
        assert!(measure_depth(lying(400, 400, 3)).is_none());
        assert!(measure_depth(lying(400, 400, 0)).is_none());

        // Slices that disagree with each other, not just with the dimensions.
        let short_coverage = vec![4u32; field.data.len() - 1];
        assert!(
            measure_depth(StackView {
                width: field.width,
                height: field.height,
                channels: 1,
                mean: &field.data,
                coverage: &short_coverage,
                rejected_samples: &short_coverage,
                accepted_frames: 4,
                rejected_frames: 0,
            })
            .is_none()
        );
    }

    #[test]
    fn frame_noise_reads_each_channel_of_one_image() {
        let (width, height) = (256, 200);
        let mut noise = Noise(7);
        let mut data = Vec::with_capacity(width * height * 3);
        for index in 0..width * height {
            let (x, y) = (index % width, index / width);
            let gradient = x as f64 * 0.05 + y as f64 * 0.03;
            for sigma in [5.0, 10.0, 20.0] {
                data.push((500.0 + gradient + noise.next_normal() * sigma) as f32);
            }
        }
        data[1234] = f32::NAN;
        let image = LinearImage::new(width, height, 3, data).unwrap();
        let measured = crate::frame_noise(&image).unwrap();
        for (measured, sigma) in measured.iter().zip([5.0_f32, 10.0, 20.0]) {
            assert!(
                (measured / sigma - 1.0).abs() < 0.05,
                "{measured} vs {sigma}"
            );
        }
        let tiny = LinearImage::new(2, 2, 1, vec![1.0; 4]).unwrap();
        assert_eq!(crate::frame_noise(&tiny), None);
    }

    fn white(size: usize, sigma: f64, seed: u64) -> LinearImage {
        let mut noise = Noise(seed);
        let data = (0..size * size)
            .map(|index| {
                let gradient = (index % size) as f64 * 0.05 + (index / size) as f64 * 0.03;
                (500.0 + gradient + noise.next_normal() * sigma) as f32
            })
            .collect();
        LinearImage::new(size, size, 1, data).unwrap()
    }

    fn scaled(image: &LinearImage, block: usize) -> f64 {
        f64::from(noise_at_scale(image, NonZeroUsize::new(block).unwrap()).unwrap()[0])
    }

    #[test]
    fn independent_noise_falls_with_the_block_and_matches_the_frames_at_one() {
        let image = white(640, 8.0, 21);
        assert_eq!(
            noise_at_scale(&image, NonZeroUsize::MIN),
            frame_noise(&image)
        );
        for block in [2, 4, 8] {
            let expected = 8.0 / block as f64;
            let measured = scaled(&image, block);
            assert!(
                (measured / expected - 1.0).abs() < 0.07,
                "block {block}: {measured}"
            );
        }
    }

    /// Each pixel of `image` drawn as a square two pixels wide, the mean of
    /// several copies at fractional shifts, as a 2x drizzle leaves them.
    fn drizzled(size: usize, sigma: f64, frames: u64, seed: u64) -> LinearImage {
        let fine = size * 2;
        let mut data = vec![0.0_f64; fine * fine];
        let mut shifts = Noise(99);
        for frame in 0..frames {
            let native = white(size, sigma, seed + frame);
            let (dx, dy) = (shifts.next_uniform(), shifts.next_uniform());
            // A fine pixel covers half of a native one along each axis.
            let axis = |fine_index: usize, shift: f64| {
                let start = (fine_index as f64 + shift * 2.0) / 2.0;
                let (first, inside) = (start.floor(), start.fract());
                let next = ((inside - 0.5) / 0.5).clamp(0.0, 1.0);
                let last = size - 1;
                (
                    (first as usize).min(last),
                    (first as usize + 1).min(last),
                    next,
                )
            };
            for y in 0..fine {
                let (y0, y1, fy) = axis(y, dy);
                for x in 0..fine {
                    let (x0, x1, fx) = axis(x, dx);
                    let at = |x: usize, y: usize| f64::from(native.data[y * size + x]);
                    data[y * fine + x] += (1.0 - fy) * ((1.0 - fx) * at(x0, y0) + fx * at(x1, y0))
                        + fy * ((1.0 - fx) * at(x0, y1) + fx * at(x1, y1));
                }
            }
        }
        let data = data
            .into_iter()
            .map(|sum| (sum / frames as f64) as f32)
            .collect();
        LinearImage::new(fine, fine, 1, data).unwrap()
    }

    #[test]
    fn a_drizzles_noise_reads_as_its_native_stacks_at_one_scale() {
        let (sigma, frames) = (8.0, 8);
        let image = drizzled(320, sigma, frames, 1_000);
        let native = sigma / (frames as f64).sqrt();
        // Neighbouring pixels share their noise, so per pixel it reads low.
        let per_pixel = f64::from(frame_noise(&image).unwrap()[0]);
        assert!(per_pixel < 0.5 * native, "{per_pixel} vs {native}");
        // 8 fine pixels are 4 native ones.
        let measured = scaled(&image, 8) * 4.0;
        assert!(
            (measured / native - 1.0).abs() < 0.12,
            "{measured} vs {native}"
        );
    }

    /// The true scatter of the block means is that of two drizzles'
    /// difference, whose sky cancels. At this size differences between
    /// neighbouring blocks would read it 15% low.
    #[test]
    fn small_blocks_of_a_drizzle_read_their_means_true_scatter() {
        let image = drizzled(320, 8.0, 8, 1_000);
        let other = drizzled(320, 8.0, 8, 2_000);
        let differences = block_means(&image, 0, 2, None)
            .into_iter()
            .zip(block_means(&other, 0, 2, None))
            .map(|(a, b)| f64::from(a - b))
            .collect::<Vec<_>>();
        let mean = mean(&differences);
        let variance = differences
            .iter()
            .map(|difference| (difference - mean).powi(2))
            .sum::<f64>()
            / differences.len() as f64;
        let scatter = (variance / 2.0).sqrt();
        let measured = scaled(&image, 2);
        assert!(
            (measured / scatter - 1.0).abs() < 0.05,
            "{measured} vs {scatter}"
        );
    }

    #[test]
    fn blocks_skip_gaps_and_refuse_images_too_small() {
        let mut image = white(640, 8.0, 22);
        for pixel in image.data.iter_mut().take(640 * 100) {
            *pixel = f32::NAN;
        }
        let measured = scaled(&image, 4);
        assert!((measured / 2.0 - 1.0).abs() < 0.07, "{measured}");
        let block = |block| NonZeroUsize::new(block).unwrap();
        assert_eq!(noise_at_scale(&image, block(64)), None);
        assert_eq!(noise_at_scale(&image, block(10_000)), None);
        assert_eq!(noise_at_scale(&image, NonZeroUsize::MAX), None);
        let means = block_means(&image, 0, 4, None);
        assert_eq!(means.len(), 160 * 160);
        assert!(means[..160 * 25].iter().all(|mean| mean.is_nan()));
        let first = (400..404)
            .flat_map(|y| (0..4).map(move |x| (x, y)))
            .map(|(x, y)| f64::from(image.data[y * 640 + x]))
            .sum::<f64>()
            / 16.0;
        assert!((f64::from(means[100 * 160]) - first).abs() < 1.0e-3);
    }
}
