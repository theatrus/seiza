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

use crate::stack::StackView;
use seiza_stats::{NORMAL_MAD_SCALE, median_in_place};

/// Rows read per measurement, at most. Noise and level statistics converge
/// long before a full frame is consumed, and this keeps the cost of a
/// measurement flat across sensor sizes.
const MAX_SAMPLED_ROWS: usize = 512;

/// Fewer samples than this and the medians are not worth reporting.
const MIN_SAMPLES: usize = 1024;

/// The brightest share of the frame that stands in for the target's signal.
const SIGNAL_FRACTION: f64 = 0.01;

/// A second difference has coefficients 1, -2, 1, so independent samples with
/// sigma noise produce a difference with sqrt(6) sigma.
const SECOND_DIFFERENCE_GAIN_SQUARED: f64 = 6.0;

/// One reading of an accumulator.
///
/// Units are the stack's own: whatever scale the frames were integrated on.
/// Only ratios between readings of the same stack mean anything.
#[derive(Clone, Debug, PartialEq)]
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
        depth *= 2;
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
    })
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
}
