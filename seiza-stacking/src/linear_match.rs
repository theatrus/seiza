//! Matching one finished image's scale and level to another's.
//!
//! [`match_linear`] finds the gain and offset that put an image on a
//! reference's scale, `reference ≈ gain * image + offset`, from the pixels
//! the two share. It is for comparing masters of different depth, so it must
//! not depend on which of the two is the noisier.
//!
//! Regressing the reference on the image does: noise in the image pulls the
//! gain toward 0, to a half where the image's noise equals its signal, and
//! the noisier master then looks cleaner than it is. Regressing the image on
//! the reference and inverting moves the fault to the reference's noise,
//! which pushes the gain up: it doubles where that equals the signal. The
//! fit here (Deming's) weighs both noises, as it measures them, and is free
//! of either. With a clean reference it is the second regression.

use crate::LinearImage;
use crate::snr::{block_means, grid_noise};
use rayon::prelude::*;
use seiza_stats::{median_f64, robust_sigma_f64};
use std::num::NonZeroUsize;
use thiserror::Error;

/// How many blocks a fit needs.
const MINIMUM_BLOCKS: usize = 256;
/// How many times clipping may run again.
const MAXIMUM_PASSES: usize = 20;
/// The shares of blocks left out at each end of the line in the fits that
/// start the clipping: little, which keeps a compact target's signal, up to
/// enough to pass over a crowded field's saturated stars.
const FIRST_FIT_TRIMS: [f64; 4] = [0.0, 0.005, 0.05, 0.2];
/// How many roundings of the largest sample a block may lie from the line
/// and still count as on it.
const ROUNDINGS: f64 = 4.0;
/// How many times its own error a gain must be to count as measured.
const MINIMUM_GAIN_SNR: f64 = 3.0;
/// What the two noises' own errors add to the variance of the log of their
/// ratio, times the number of blocks: 6 each from second differences by MAD
/// (simulated), less 4 since they follow the noise that the blocks hold.
const NOISE_RATIO_VARIANCE: f64 = 4.0;

/// How [`match_linear`] samples and clips.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearMatchOptions {
    /// Edge in pixels of the blocks whose means the fit reads (default 4).
    /// Larger blocks lower the noise and pass over small misalignments and
    /// seeing differences.
    pub bin: NonZeroUsize,
    /// How far, in robust sigmas of the fit's residuals, a block may lie
    /// from the line before it is left out (default 3).
    pub clip_sigma: f64,
}

impl Default for LinearMatchOptions {
    fn default() -> Self {
        Self {
            bin: NonZeroUsize::new(4).unwrap(),
            clip_sigma: 3.0,
        }
    }
}

/// One channel's fit: `reference ≈ gain * image + offset`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChannelMatch {
    /// What to multiply the image by.
    pub gain: f64,
    /// What to add after that.
    pub offset: f64,
    /// Standard error of `gain` from the two images' noise, which it takes
    /// as independent from block to block. It knows nothing of real
    /// differences between the images, such as a gradient.
    pub gain_error: f64,
    /// Blocks the fit used.
    pub blocks: usize,
    /// Blocks the clipping left out.
    pub rejected: usize,
}

/// Why [`match_linear`] gave no fit.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum LinearMatchError {
    #[error(
        "the image and the reference differ in size or channels, or a buffer does not match its size"
    )]
    Shape,
    #[error("the mask must have one entry per pixel: {expected}, not {found}")]
    Mask { expected: usize, found: usize },
    #[error("the clipping sigma must be finite and greater than zero")]
    ClipSigma,
    #[error("channel {channel} has too few blocks to fit after masking and clipping")]
    TooFewBlocks { channel: usize },
    #[error("channel {channel} of the two images shares too little signal to measure a gain")]
    NoCommonSignal { channel: usize },
}

/// Fit the gain and offset that put `image` on `reference`'s scale, one
/// [`ChannelMatch`] per channel.
///
/// The images must be registered to one grid. The fit reads means of blocks
/// of [`LinearMatchOptions::bin`] pixels, leaving out any block that holds a
/// non-finite sample in either image or a pixel `mask` excludes: `mask` has
/// one entry per pixel, row by row, `true` where it excludes the pixel. It also
/// leaves out blocks that hold one value in both images, such as an empty
/// border, and clips blocks far from the line, such as stars that saturate in
/// one image.
pub fn match_linear(
    image: &LinearImage,
    reference: &LinearImage,
    mask: Option<&[bool]>,
    options: &LinearMatchOptions,
) -> Result<Vec<ChannelMatch>, LinearMatchError> {
    let (width, height, channels) = (image.width, image.height, image.channels);
    let samples = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(channels));
    if (width, height, channels) != (reference.width, reference.height, reference.channels)
        || channels == 0
        || samples != Some(image.data.len())
        || samples != Some(reference.data.len())
    {
        return Err(LinearMatchError::Shape);
    }
    if let Some(mask) = mask
        && mask.len() != width * height
    {
        return Err(LinearMatchError::Mask {
            expected: width * height,
            found: mask.len(),
        });
    }
    if !options.clip_sigma.is_finite() || options.clip_sigma <= 0.0 {
        return Err(LinearMatchError::ClipSigma);
    }
    (0..channels)
        .into_par_iter()
        .map(|channel| match_channel(image, reference, mask, options, channel))
        // The lowest failing channel's error, not the first to come.
        .collect::<Vec<_>>()
        .into_iter()
        .collect()
}

/// Whether one channel of an image is a single value over the block, or over
/// 2 by 2 pixels for a block of one.
fn is_flat(image: &LinearImage, channel: usize, bin: usize, column: usize, row: usize) -> bool {
    let edge = bin.max(2);
    if image.width < edge || image.height < edge {
        return false;
    }
    let left = (column * bin).min(image.width - edge);
    let top = (row * bin).min(image.height - edge);
    let sample = |x: usize, y: usize| image.data[(y * image.width + x) * image.channels + channel];
    let first = sample(left, top);
    (top..top + edge).all(|y| (left..left + edge).all(|x| sample(x, y) == first))
}

fn match_channel(
    image: &LinearImage,
    reference: &LinearImage,
    mask: Option<&[bool]>,
    options: &LinearMatchOptions,
    channel: usize,
) -> Result<ChannelMatch, LinearMatchError> {
    let too_few = LinearMatchError::TooFewBlocks { channel };
    let no_signal = LinearMatchError::NoCommonSignal { channel };
    let bin = options.bin.get();
    let (columns, rows) = (image.width / bin, image.height / bin);
    let mut xs = block_means(image, channel, bin, mask);
    let mut rs = block_means(reference, channel, bin, mask);
    // Flat blocks would pass for a noise of zero.
    xs.par_iter_mut()
        .zip(&mut rs)
        .enumerate()
        .for_each(|(index, (x, r))| {
            let (column, row) = (index % columns, index / columns);
            let flat = |image| is_flat(image, channel, bin, column, row);
            if !x.is_finite() || !r.is_finite() || (flat(image) && flat(reference)) {
                (*x, *r) = (f32::NAN, f32::NAN);
            }
        });
    let noise = |means: &[f32]| grid_noise(means, columns, rows).ok_or(too_few);
    let noise = Noise {
        image: noise(&xs)?,
        reference: noise(&rs)?,
    };
    let pairs = xs
        .iter()
        .zip(&rs)
        .filter(|(x, _)| x.is_finite())
        .map(|(&x, &r)| (f64::from(x), f64::from(r)))
        .collect::<Vec<_>>();
    if pairs.len() < MINIMUM_BLOCKS {
        return Err(too_few);
    }

    // A few saturated stars can turn a first fit so far that clipping never
    // finds them, so several start it and the line nearest the blocks wins.
    let sorted = along_the_line(&pairs, noise);
    let mut best: Option<(f64, Line)> = None;
    let mut error = no_signal;
    for trim in FIRST_FIT_TRIMS {
        let trim = (sorted.len() as f64 * trim) as usize;
        let start = Line::fit(&sorted[trim..sorted.len() - trim], noise, None);
        let Some(start) = start else { continue };
        match start.clipped(&pairs, noise, options.clip_sigma) {
            Ok(line) => {
                let distance = line.distance(&pairs, noise);
                if best.as_ref().is_none_or(|(least, _)| distance < *least) {
                    best = Some((distance, line));
                }
            }
            Err(Unclipped::TooFewBlocks) => error = too_few,
            Err(Unclipped::NoLine) => {}
        }
    }
    let (_, line) = best.ok_or(error)?;
    if line.gain < MINIMUM_GAIN_SNR * line.gain_error {
        return Err(no_signal);
    }
    Ok(ChannelMatch {
        gain: line.gain,
        offset: line.offset,
        gain_error: line.gain_error,
        blocks: line.blocks,
        rejected: pairs.len() - line.blocks,
    })
}

/// The noise of the two images' block means, each in its own units.
#[derive(Clone, Copy)]
struct Noise {
    image: f64,
    reference: f64,
}

/// Why clipping gave no line.
enum Unclipped {
    TooFewBlocks,
    NoLine,
}

/// A clip that chose the blocks of a fit.
#[derive(Clone, Copy)]
struct Clip {
    /// The gain of the line the clip measured from.
    gain: f64,
    /// The share of a normal residual's variance that the clip leaves.
    variance: f64,
}

/// The variance of a unit normal deviate clipped at `limit` either side.
fn clipped_variance(limit: f64) -> f64 {
    const STEPS: usize = 64;
    if limit >= 8.0 {
        return 1.0;
    }
    // Simpson's rule for the two integrals.
    let step = limit / STEPS as f64;
    let (mut mass, mut square) = (0.0, 0.0);
    for index in 0..=STEPS {
        let weight = match index {
            0 | STEPS => 1.0,
            _ if index % 2 == 1 => 4.0,
            _ => 2.0,
        };
        let x = index as f64 * step;
        let density = weight * (-0.5 * x * x).exp();
        mass += density;
        square += density * x * x;
    }
    square / mass
}

/// The pairs in order along the line. Their places on it are weighed by the
/// noise, so a block's noise does not decide which end it falls from.
fn along_the_line(pairs: &[(f64, f64)], noise: Noise) -> Vec<(f64, f64)> {
    let spread = |values: Vec<f64>| {
        let center = median_f64(&values).unwrap_or(0.0);
        robust_sigma_f64(&values, center).unwrap_or(0.0)
    };
    let image = spread(pairs.iter().map(|pair| pair.0).collect());
    let reference = spread(pairs.iter().map(|pair| pair.1).collect());
    if image <= 0.0 || reference <= 0.0 {
        return pairs.to_vec();
    }
    let gain = reference / image;
    // Each sample's inverse variance, on the reference's scale.
    let image_noise = (gain * noise.image).powi(2);
    let reference_noise = noise.reference.powi(2);
    let total = image_noise + reference_noise;
    let image_weight = if total > 0.0 {
        reference_noise / total
    } else {
        0.5
    };
    let place = |&(x, r): &(f64, f64)| image_weight * gain * x + (1.0 - image_weight) * r;
    let mut sorted = pairs.to_vec();
    sorted.sort_unstable_by(|a, b| place(a).total_cmp(&place(b)));
    sorted
}

#[derive(Clone, Copy)]
struct Line {
    gain: f64,
    offset: f64,
    gain_error: f64,
    blocks: usize,
}

impl Line {
    fn residuals(&self, pairs: &[(f64, f64)]) -> Vec<f64> {
        pairs
            .iter()
            .map(|&(x, r)| r - self.gain * x - self.offset)
            .collect()
    }

    /// The rounding of `pairs`' residuals from this line.
    fn rounding(&self, pairs: &[(f64, f64)]) -> f64 {
        let level = pairs.iter().fold(0.0_f64, |level, (x, r)| {
            level.max(r.abs()).max((self.gain * x).abs())
        });
        f64::from(f32::EPSILON) * level
    }

    /// This line fitted again to the pairs within `clip_sigma` robust sigmas
    /// of it, until they no longer change.
    fn clipped(
        mut self,
        pairs: &[(f64, f64)],
        noise: Noise,
        clip_sigma: f64,
    ) -> Result<Self, Unclipped> {
        let mut kept = Vec::new();
        for _ in 0..MAXIMUM_PASSES {
            let residuals = self.residuals(pairs);
            let center = median_f64(&residuals).ok_or(Unclipped::NoLine)?;
            let sigma = robust_sigma_f64(&residuals, center).ok_or(Unclipped::NoLine)?;
            let reach = (clip_sigma * sigma).max(ROUNDINGS * self.rounding(pairs));
            let keep = residuals
                .iter()
                .map(|residual| (residual - center).abs() <= reach)
                .collect::<Vec<_>>();
            if keep == kept {
                break;
            }
            let used = pairs
                .iter()
                .zip(&keep)
                .filter_map(|(&pair, &keep)| keep.then_some(pair))
                .collect::<Vec<_>>();
            if used.len() < MINIMUM_BLOCKS {
                return Err(Unclipped::TooFewBlocks);
            }
            let clip = Clip {
                gain: self.gain,
                variance: if sigma > 0.0 {
                    clipped_variance(reach / sigma)
                } else {
                    1.0
                },
            };
            self = Self::fit(&used, noise, Some(clip)).ok_or(Unclipped::NoLine)?;
            kept = keep;
        }
        Ok(self)
    }

    /// How far all the pairs lie from this line, in units of their noise
    /// across it, on a scale where outliers count little.
    fn distance(&self, pairs: &[(f64, f64)], noise: Noise) -> f64 {
        let variance = (noise.reference.powi(2) + (self.gain * noise.image).powi(2))
            .max(self.rounding(pairs).powi(2))
            .max(f64::MIN_POSITIVE);
        self.residuals(pairs)
            .iter()
            .map(|residual| (residual * residual / variance).ln_1p())
            .sum()
    }

    /// The line nearest `pairs`, each distance in units of its noise, or
    /// `None` when they do not rise together. `clip` is the clip that chose
    /// the pairs.
    fn fit(pairs: &[(f64, f64)], noise: Noise, clip: Option<Clip>) -> Option<Self> {
        let count = pairs.len() as f64;
        let mean_x = pairs.iter().map(|pair| pair.0).sum::<f64>() / count;
        let mean_r = pairs.iter().map(|pair| pair.1).sum::<f64>() / count;
        let (mut xx, mut rr, mut xr) = (0.0, 0.0, 0.0);
        for (x, r) in pairs {
            let (x, r) = (x - mean_x, r - mean_r);
            xx += x * x;
            rr += r * r;
            xr += x * r;
        }
        let (mut xx, mut rr, mut xr) = (xx / count, rr / count, xr / count);
        // Blocks near a line agree with it, whatever line it is, and would
        // hold the next fit to it. Undo that: choosing blocks by a residual
        // leaves a regression on the residual as it was and shrinks the
        // residual's variance by a known share (Pearson's and Lawley's
        // correction).
        let clipped = clip.map_or(1.0, |clip| clip.variance);
        if let Some(Clip { gain, variance }) = clip {
            let xe = xr - gain * xx;
            let ee = rr - 2.0 * gain * xr + gain * gain * xx;
            if ee > 0.0 && variance < 1.0 {
                let slope = xe / ee;
                let (wide_ee, wide_xe) = (ee / variance, xe / variance);
                xx += slope * (wide_xe - xe);
                xr = gain * xx + wide_xe;
                rr = gain * gain * xx + 2.0 * gain * wide_xe + wide_ee;
            }
        }
        // Also refuses NaN.
        if !(xr > 0.0 && xx > 0.0 && rr > 0.0) {
            return None;
        }
        let (image, reference) = (noise.image.powi(2), noise.reference.powi(2));
        let gain = if image <= 0.0 && reference <= 0.0 {
            (rr / xx).sqrt()
        } else if image <= 0.0 {
            xr / xx
        } else if reference <= 0.0 {
            rr / xr
        } else {
            let ratio = reference / image;
            let difference = rr - ratio * xx;
            let root = (difference * difference + 4.0 * ratio * xr * xr).sqrt();
            // Two forms of one root: each is exact where the other cancels.
            if difference < 0.0 {
                2.0 * ratio * xr / (root - difference)
            } else {
                (difference + root) / (2.0 * xr)
            }
        };
        if !(gain > 0.0 && gain.is_finite()) {
            return None;
        }
        // Fuller, Measurement Error Models, section 1.3, with the image's
        // signal variance `xr / gain` and independent noises.
        let signal = xr / gain;
        let residual = reference + gain * gain * image;
        // What the clipped blocks tell of the line's slope.
        let weight = count * clipped;
        let fitted = (signal * residual + image * reference) / (weight * signal * signal);
        // How the gain moves with the log of the noises' ratio.
        let by_ratio = if residual > 0.0 {
            gain * image * reference / (signal * residual)
        } else {
            0.0
        };
        let variance = fitted + by_ratio * by_ratio * NOISE_RATIO_VARIANCE / count;
        Some(Self {
            gain,
            offset: mean_r - gain * mean_x,
            gain_error: variance.sqrt(),
            blocks: pairs.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: usize = 512;

    /// A deterministic generator of unit normal deviates.
    struct Normal(u64);

    impl Normal {
        fn uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64 / (1_u64 << 53) as f64).mul_add(0.999_999, 5e-7)
        }

        fn next(&mut self) -> f64 {
            (-2.0 * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
        }
    }

    /// A smooth nebula, of unit spread across the frame.
    fn nebula(x: usize, y: usize) -> f64 {
        let (x, y) = (x as f64 / SIZE as f64, y as f64 / SIZE as f64);
        let wave = (4.0 * x + 1.0).sin() * (5.0 * y).cos() + (7.0 * (x + y)).sin() * 0.6;
        wave / 0.62
    }

    /// `level + scale * nebula` with noise of `sigma` per pixel.
    fn master(level: f64, scale: f64, sigma: f64, seed: u64) -> LinearImage {
        let mut normal = Normal(seed);
        let data = (0..SIZE * SIZE)
            .map(|index| {
                let signal = scale * nebula(index % SIZE, index / SIZE);
                (level + signal + sigma * normal.next()) as f32
            })
            .collect();
        LinearImage::new(SIZE, SIZE, 1, data).unwrap()
    }

    fn fit(image: &LinearImage, reference: &LinearImage) -> ChannelMatch {
        match_linear(image, reference, None, &LinearMatchOptions::default()).unwrap()[0]
    }

    /// Binning by 4 divides the noise by 4, so a pixel noise of 4 is a block
    /// noise equal to the signal.
    #[test]
    fn the_gain_does_not_depend_on_which_image_is_noisy() {
        for (image_noise, reference_noise) in [
            (0.4, 0.04),
            (4.0, 0.04),
            (12.0, 0.04),
            (0.4, 4.0),
            (4.0, 4.0),
        ] {
            // reference = 2 * image + 30
            let image = master(100.0, 0.5, 0.5 * image_noise, 1);
            let reference = master(230.0, 1.0, reference_noise, 2);
            let fit = fit(&image, &reference);
            assert!(
                (fit.gain / 2.0 - 1.0).abs() < 0.06,
                "noise {image_noise}, {reference_noise}: {fit:?}"
            );
            let level = fit.gain * 100.0 + fit.offset;
            assert!((level - 230.0).abs() < 0.2, "{fit:?}");
            assert!(
                (fit.gain - 2.0).abs() < 4.0 * fit.gain_error.max(0.004),
                "{fit:?}"
            );
        }
    }

    #[test]
    fn the_two_directions_are_each_others_inverse() {
        let image = master(100.0, 0.5, 2.0, 3);
        let reference = master(230.0, 1.0, 3.0, 4);
        let there = fit(&image, &reference);
        let back = fit(&reference, &image);
        assert!(
            (there.gain * back.gain - 1.0).abs() < 0.01,
            "{there:?} {back:?}"
        );
    }

    #[test]
    fn identical_images_match_at_one_and_zero() {
        let image = master(100.0, 1.0, 1.0, 5);
        let fit = fit(&image, &image);
        assert!((fit.gain - 1.0).abs() < 1.0e-9, "{fit:?}");
        assert!(fit.offset.abs() < 1.0e-6, "{fit:?}");
        assert_eq!(fit.rejected, 0);
    }

    /// Clipping across a line keeps the blocks that agree with it, so without
    /// its correction half of these passed, with gains of any size.
    #[test]
    fn noise_alone_has_no_gain() {
        for clip_sigma in [2.0, 3.0] {
            let options = LinearMatchOptions {
                bin: NonZeroUsize::MIN,
                clip_sigma,
            };
            for seed in 0..8 {
                let image = master(100.0, 0.0, 1.0, 1_000 + 2 * seed);
                let reference = master(230.0, 0.0, 1.0, 1_001 + 2 * seed);
                assert_eq!(
                    match_linear(&image, &reference, None, &options),
                    Err(LinearMatchError::NoCommonSignal { channel: 0 }),
                    "seed {seed}, clip {clip_sigma}"
                );
            }
        }
    }

    /// A tight clip tells less of the slope, and noise of four times the
    /// signal in both makes the noise measurement's own error count.
    #[test]
    fn the_gain_error_is_the_gains_scatter() {
        const TRIALS: u64 = 96;
        for (clip_sigma, noise) in [(1.0, 1.0), (3.0, 4.0)] {
            let options = LinearMatchOptions {
                clip_sigma,
                ..LinearMatchOptions::default()
            };
            let fits = (0..TRIALS)
                .into_par_iter()
                .map(|seed| {
                    let image = master(100.0, 0.5, 2.0 * noise, 2_000 + 2 * seed);
                    let reference = master(230.0, 1.0, 4.0 * noise, 2_001 + 2 * seed);
                    match_linear(&image, &reference, None, &options).unwrap()[0]
                })
                .collect::<Vec<_>>();
            let count = TRIALS as f64;
            let mean = fits.iter().map(|fit| fit.gain).sum::<f64>() / count;
            let squares = fits.iter().map(|fit| (fit.gain - mean).powi(2));
            let scatter = (squares.sum::<f64>() / (count - 1.0)).sqrt();
            let error = fits.iter().map(|fit| fit.gain_error).sum::<f64>() / count;
            assert!((mean / 2.0 - 1.0).abs() < 0.04, "{mean}");
            assert!(
                (0.8..1.25).contains(&(scatter / error)),
                "clip {clip_sigma}: scatter {scatter}, error {error}"
            );
        }
    }

    /// A model's noise reads as almost nothing beside the reference's.
    #[test]
    fn a_noiseless_model_matches_a_noisy_reference() {
        let blob = |sigma: f64| {
            let mut normal = Normal(20);
            let data = (0..SIZE * SIZE)
                .map(|index| {
                    let (x, y) = ((index % SIZE) as f64 - 200.0, (index / SIZE) as f64 - 310.0);
                    let signal = 10.0 * (-(x * x + y * y) / 800.0).exp();
                    (signal + sigma * normal.next()) as f32
                })
                .collect();
            LinearImage::new(SIZE, SIZE, 1, data).unwrap()
        };
        let there = fit(&blob(0.0), &blob(0.2));
        let back = fit(&blob(0.2), &blob(0.0));
        assert!((there.gain - 1.0).abs() < 0.01, "{there:?}");
        assert!((back.gain - 1.0).abs() < 0.01, "{back:?}");
    }

    /// Most of both frames is one value, as where a sky was clipped, and
    /// stars saturate in one.
    #[test]
    fn a_flat_sky_does_not_stop_the_clipping() {
        for (sigma, bin) in [(0.0, 4), (0.3, 4), (0.0, 1), (0.3, 1)] {
            let options = LinearMatchOptions {
                bin: NonZeroUsize::new(bin).unwrap(),
                ..LinearMatchOptions::default()
            };
            let mut image = master(100.0, 10.0, sigma, 21);
            let mut reference = master(230.0, 20.0, sigma, 22);
            for pixel in 0..SIZE * SIZE * 6 / 10 {
                (image.data[pixel], reference.data[pixel]) = (100.0, 230.0);
            }
            for star in 0..100 {
                let (cx, cy) = (11 + star * 67 % 490, 9 + star * 139 % 490);
                for index in 0..64 {
                    let pixel = (cy + index / 8) * SIZE + cx + index % 8;
                    image.data[pixel] += 5_000.0;
                    reference.data[pixel] += 900.0;
                }
            }
            let fit = match_linear(&image, &reference, None, &options).unwrap()[0];
            assert!((fit.gain / 2.0 - 1.0).abs() < 0.01, "{fit:?}");
            assert!((fit.offset - 30.0).abs() < 1.0, "{fit:?}");
        }
    }

    /// A ramp in whole numbers has no noise at all to measure lines by.
    #[test]
    fn a_noiseless_pair_still_passes_over_saturated_stars() {
        let ramp = |level: f32, gain: f32| {
            let data = (0..SIZE * SIZE)
                .map(|index| level + gain * (index % SIZE + index / SIZE) as f32)
                .collect();
            LinearImage::new(SIZE, SIZE, 1, data).unwrap()
        };
        let (mut image, mut reference) = (ramp(100.0, 1.0), ramp(230.0, 2.0));
        for star in 0..100 {
            let (cx, cy) = (11 + star * 67 % 490, 9 + star * 139 % 490);
            for index in 0..64 {
                let pixel = (cy + index / 8) * SIZE + cx + index % 8;
                image.data[pixel] += 50_000.0;
                reference.data[pixel] += 9_000.0;
            }
        }
        let fit = fit(&image, &reference);
        assert!((fit.gain / 2.0 - 1.0).abs() < 1.0e-4, "{fit:?}");
    }

    #[test]
    fn an_exact_pair_keeps_every_block() {
        let image = master(0.05, 0.01, 0.0, 23);
        let mut reference = image.clone();
        for value in &mut reference.data {
            *value = *value * 2.0 + 0.03;
        }
        let fit = fit(&image, &reference);
        assert!((fit.gain - 2.0).abs() < 1.0e-5, "{fit:?}");
        assert_eq!((fit.blocks, fit.rejected), (128 * 128, 0), "{fit:?}");
    }

    #[test]
    fn saturated_stars_in_one_image_do_not_turn_the_line() {
        let mut image = master(100.0, 0.5, 0.5, 6);
        let mut reference = master(230.0, 1.0, 0.2, 7);
        for star in 0..120 {
            let (cx, cy) = (11 + star * 67 % 490, 9 + star * 139 % 490);
            for index in 0..64 {
                let pixel = (cy + index / 8) * SIZE + cx + index % 8;
                image.data[pixel] += 5_000.0;
                reference.data[pixel] += 900.0;
            }
        }
        let fit = fit(&image, &reference);
        assert!((fit.gain / 2.0 - 1.0).abs() < 0.03, "{fit:?}");
        assert!(fit.rejected >= 120, "{fit:?}");
    }

    /// A target over 4% of an otherwise empty sky still fixes the gain.
    #[test]
    fn a_compact_target_fixes_the_gain() {
        let blob = |level: f64, scale: f64, sigma: f64, seed| {
            let mut normal = Normal(seed);
            let data = (0..SIZE * SIZE)
                .map(|index| {
                    let (x, y) = ((index % SIZE) as f64 - 200.0, (index / SIZE) as f64 - 310.0);
                    let signal = scale * 10.0 * (-(x * x + y * y) / 3_200.0).exp();
                    (level + signal + sigma * normal.next()) as f32
                })
                .collect();
            LinearImage::new(SIZE, SIZE, 1, data).unwrap()
        };
        for (image_noise, reference_noise) in [(2.0, 0.2), (2.0, 4.0)] {
            let fit = fit(
                &blob(100.0, 0.5, image_noise, 18),
                &blob(230.0, 1.0, reference_noise, 19),
            );
            assert!((fit.gain / 2.0 - 1.0).abs() < 0.04, "{fit:?}");
        }
    }

    #[test]
    fn masked_and_non_finite_pixels_are_left_out() {
        let mut image = master(100.0, 0.5, 0.5, 8);
        let reference = master(230.0, 1.0, 0.2, 9);
        let clean = fit(&image, &reference);
        let mut mask = vec![false; SIZE * SIZE];
        for y in 0..SIZE {
            for x in 0..SIZE / 4 {
                image.data[y * SIZE + x] = if y < 8 { f32::NAN } else { 4_000.0 };
                mask[y * SIZE + x] = true;
            }
        }
        let options = LinearMatchOptions::default();
        let fit = match_linear(&image, &reference, Some(&mask), &options).unwrap()[0];
        assert!((fit.gain / clean.gain - 1.0).abs() < 0.02, "{fit:?}");
        assert!(fit.blocks + fit.rejected == 128 * 96, "{fit:?}");
    }

    #[test]
    fn colour_channels_fit_apart() {
        let planes = [
            master(100.0, 0.5, 0.5, 10),
            master(60.0, 2.0, 0.5, 11),
            master(10.0, 1.0, 0.5, 12),
        ];
        let reference = master(230.0, 1.0, 0.2, 13);
        let interleave = |planes: [&LinearImage; 3]| {
            let data = (0..SIZE * SIZE)
                .flat_map(|pixel| planes.map(|plane| plane.data[pixel]))
                .collect();
            LinearImage::new(SIZE, SIZE, 3, data).unwrap()
        };
        let image = interleave([&planes[0], &planes[1], &planes[2]]);
        let reference = interleave([&reference; 3]);
        let options = LinearMatchOptions::default();
        let fits = match_linear(&image, &reference, None, &options).unwrap();
        for (fit, gain) in fits.iter().zip([2.0, 0.5, 1.0]) {
            assert!((fit.gain / gain - 1.0).abs() < 0.03, "{fit:?}");
        }
    }

    #[test]
    fn refusals_say_why() {
        let options = LinearMatchOptions::default();
        let image = master(100.0, 1.0, 1.0, 14);
        fn refused(
            image: &LinearImage,
            reference: &LinearImage,
            mask: Option<&[bool]>,
            options: &LinearMatchOptions,
        ) -> LinearMatchError {
            match_linear(image, reference, mask, options).unwrap_err()
        }
        let small = LinearImage::new(4, 4, 1, vec![1.0; 16]).unwrap();
        assert_eq!(
            refused(&image, &small, None, &options),
            LinearMatchError::Shape
        );
        assert_eq!(
            refused(&image, &image, Some(&[true; 3]), &options),
            LinearMatchError::Mask {
                expected: SIZE * SIZE,
                found: 3
            }
        );
        for clip_sigma in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let options = LinearMatchOptions {
                clip_sigma,
                ..options
            };
            assert_eq!(
                refused(&image, &image, None, &options),
                LinearMatchError::ClipSigma
            );
        }
        assert_eq!(
            refused(&small, &small, None, &options),
            LinearMatchError::TooFewBlocks { channel: 0 }
        );
        // A buffer shorter than its size, in both images.
        let mut short = image.clone();
        short.data.truncate(100);
        assert_eq!(
            refused(&short, &short, None, &options),
            LinearMatchError::Shape
        );
        // A tight clip, and blocks larger than any image.
        let tight = LinearMatchOptions {
            clip_sigma: 0.01,
            ..options
        };
        let other = master(100.0, 1.0, 1.0, 24);
        assert_eq!(
            refused(&image, &other, None, &tight),
            LinearMatchError::TooFewBlocks { channel: 0 }
        );
        let huge = LinearMatchOptions {
            bin: NonZeroUsize::MAX,
            ..options
        };
        assert_eq!(
            refused(&image, &image, None, &huge),
            LinearMatchError::TooFewBlocks { channel: 0 }
        );
        // The first of the channels that fail.
        let rgb = |planes: [&LinearImage; 3]| {
            let data = (0..SIZE * SIZE)
                .flat_map(|pixel| planes.map(|plane| plane.data[pixel]))
                .collect();
            LinearImage::new(SIZE, SIZE, 3, data).unwrap()
        };
        let negative = master(1.0, -1.0, 1.0, 25);
        let mut empty = image.clone();
        empty.data.fill(f32::NAN);
        for _ in 0..16 {
            assert_eq!(
                refused(
                    &rgb([&image, &image, &empty]),
                    &rgb([&negative, &image, &empty]),
                    None,
                    &options
                ),
                LinearMatchError::NoCommonSignal { channel: 0 }
            );
        }
        // Noise alone, and a negative.
        assert_eq!(
            refused(
                &master(1.0, 0.0, 1.0, 15),
                &master(1.0, 0.0, 1.0, 16),
                None,
                &options
            ),
            LinearMatchError::NoCommonSignal { channel: 0 }
        );
        assert_eq!(
            refused(&image, &master(1.0, -1.0, 1.0, 17), None, &options),
            LinearMatchError::NoCommonSignal { channel: 0 }
        );
    }
}
