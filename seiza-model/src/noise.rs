//! The noise a network reads, made from the image alone.
//!
//! `image_sigma` is the harness's function of the same name, and matches it
//! on fixtures. The arithmetic follows it closely, in the same precision at
//! each step, because a model is trained on exactly what it makes.

use crate::bundle::{NoiseInput, Reading};
use crate::sky::upsample;
use crate::stats::{median_f32, median_f64};
use rayon::prelude::*;

/// The median of a χ² with one degree of freedom over its mean: a second
/// difference's square, of white noise, has it.
const CHI2_1_MEDIAN: f64 = 0.4549;
/// The signal term is fitted on the blocks whose median is this many times the
/// sky's noise above the background.
const IMAGE_GAIN_LEVEL: f64 = 5.0;
/// The fewest such blocks the signal term is fitted on.
const IMAGE_GAIN_BLOCKS: usize = 20;
/// No block's sky variance falls below this share of the image's.
const SKY_VARIANCE_FLOOR: f64 = 0.25;

/// The noise map `reading` asks for, NaN where the image is not finite.
pub(crate) fn input_sigma(
    plane: &[f32],
    width: usize,
    height: usize,
    reading: &Reading,
) -> Vec<f32> {
    let mapped = image_sigma(
        plane,
        width,
        height,
        reading.sky_block,
        reading.image_map_scale,
    );
    match reading.noise_map {
        NoiseInput::Image => mapped,
        NoiseInput::Constant => {
            let mut finite: Vec<f32> = mapped.iter().copied().filter(|v| v.is_finite()).collect();
            let level = median_f32(&mut finite).unwrap_or(f32::NAN);
            mapped
                .iter()
                .map(|v| if v.is_finite() { level } else { f32::NAN })
                .collect()
        }
    }
}

/// A noise map from the image alone. The sky's variance is that of the
/// image's second differences along each axis, as medians over `block`-pixel
/// blocks: a planar gradient cancels in them, and the stars drop out. Spread
/// back smoothly, the blocks keep vignetting and thinly covered edges. A term
/// growing with the signal is fitted on the blocks a bright nebula fills, and
/// taken out of those blocks' own variance first, so the sky is not counted
/// twice. `scale` sets the map's level.
pub(crate) fn image_sigma(
    plane: &[f32],
    width: usize,
    height: usize,
    block: usize,
    scale: f64,
) -> Vec<f32> {
    let rows = (height / block).max(1);
    let columns = (width / block).max(1);
    let at = |x: usize, y: usize| f64::from(plane[y * width + x]);

    let cells: Vec<(f64, f64)> = (0..rows * columns)
        .into_par_iter()
        .map(|cell| {
            let (row, column) = (cell / columns, cell % columns);
            let (top, left) = (row * block, column * block);
            let (bottom, right) = (
                ((row + 1) * block).min(height),
                ((column + 1) * block).min(width),
            );
            let capacity = (bottom - top) * (right - left);
            let mut across = Vec::with_capacity(capacity);
            let mut down = Vec::with_capacity(capacity);
            let mut values = Vec::with_capacity(capacity);
            for y in top..bottom {
                for x in left..right {
                    let centre = at(x, y);
                    if centre.is_finite() {
                        values.push(centre);
                    }
                    if x >= 1 && x + 1 < width {
                        let difference = at(x - 1, y) - 2.0 * centre + at(x + 1, y);
                        if difference.is_finite() {
                            across.push(difference * difference);
                        }
                    }
                    if y >= 1 && y + 1 < height {
                        let difference = at(x, y - 1) - 2.0 * centre + at(x, y + 1);
                        if difference.is_finite() {
                            down.push(difference * difference);
                        }
                    }
                }
            }
            let across = median_f64(&mut across).unwrap_or(f64::NAN);
            let down = median_f64(&mut down).unwrap_or(f64::NAN);
            let value = median_f64(&mut values).unwrap_or(f64::NAN);
            // A second difference of white noise has six times its variance.
            (0.5 * (across + down) / (6.0 * CHI2_1_MEDIAN), value)
        })
        .collect();

    let mut sampled = Vec::new();
    for y in (0..height).step_by(7) {
        for x in (0..width).step_by(7) {
            let value = at(x, y);
            if value.is_finite() {
                sampled.push(value);
            }
        }
    }
    let background = median_f64(&mut sampled).unwrap_or(0.0);

    let variance_blocks: Vec<f64> = cells.iter().map(|cell| cell.0).collect();
    let signal_blocks: Vec<f64> = cells.iter().map(|cell| cell.1 - background).collect();
    let finite: Vec<bool> = variance_blocks
        .iter()
        .zip(&signal_blocks)
        .map(|(variance, signal)| variance.is_finite() && signal.is_finite())
        .collect();
    let select = |keep: &dyn Fn(usize) -> bool| -> Vec<f64> {
        (0..variance_blocks.len())
            .filter(|&index| keep(index))
            .map(|index| variance_blocks[index])
            .collect()
    };
    let sky_variance = median_f64(&mut select(&|index| finite[index])).unwrap_or(1.0);
    let level = IMAGE_GAIN_LEVEL * sky_variance.sqrt();
    let nebula: Vec<bool> = (0..variance_blocks.len())
        .map(|index| finite[index] && signal_blocks[index] > level)
        .collect();
    let mut gain = 0.0_f64;
    if nebula.iter().filter(|&&inside| inside).count() >= IMAGE_GAIN_BLOCKS {
        let floor = median_f64(&mut select(&|index| finite[index] && !nebula[index]))
            .unwrap_or(sky_variance);
        let mut slopes: Vec<f64> = (0..variance_blocks.len())
            .filter(|&index| nebula[index])
            .map(|index| (variance_blocks[index] - floor) / signal_blocks[index])
            .collect();
        gain = median_f64(&mut slopes).unwrap_or(0.0).max(0.0);
    }

    let sky_blocks: Vec<f64> = variance_blocks
        .iter()
        .zip(&signal_blocks)
        .map(|(&variance, &signal)| {
            let signal = if signal.is_nan() {
                0.0
            } else {
                signal.max(0.0)
            };
            let sky = variance - gain * signal;
            let sky = if sky.is_finite() { sky } else { sky_variance };
            sky.max(SKY_VARIANCE_FLOOR * sky_variance)
        })
        .collect();
    let sky = upsample(&sky_blocks, rows, columns, height, width, block, 0.0);

    plane
        .par_iter()
        .zip(sky.par_iter())
        .map(|(&value, &sky)| {
            if !value.is_finite() {
                return f32::NAN;
            }
            let signal = (f64::from(value) - background).max(0.0);
            (scale * (f64::from(sky) + gain * signal).sqrt()) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundle::{NoisePlane, SkyInput};

    /// A repeatable generator of normal deviates.
    struct Normal(u64);

    impl Normal {
        fn uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
        }

        fn sample(&mut self) -> f64 {
            let radius = (-2.0 * self.uniform().ln()).sqrt();
            radius * (2.0 * std::f64::consts::PI * self.uniform()).cos()
        }
    }

    fn reading(noise_map: NoiseInput) -> Reading {
        Reading {
            noise_map,
            noise_plane: NoisePlane::Asinh,
            sky: SkyInput::Flat,
            sky_block: 32,
            flat_block: 256,
            image_map_scale: 1.0,
        }
    }

    /// As the harness's test: a gradient, stars, a noisier strip on the left
    /// and on the right a nebula whose shot noise adds to the variance.
    #[test]
    fn the_image_noise_map_finds_the_sky_s_noise_and_a_nebula_s() {
        let (width, height, sky_sigma, gain) = (640_usize, 512_usize, 2.0_f64, 0.05_f64);
        let mut normal = Normal(5);
        let stars: Vec<(f64, f64)> = (0..80)
            .map(|_| {
                (
                    normal.uniform() * height as f64,
                    normal.uniform() * width as f64,
                )
            })
            .collect();
        let sigma_at = |x: usize| {
            let strip = if x < 64 { 4.0 } else { 1.0 };
            let nebula = if x > 400 { 2000.0 } else { 0.0 };
            (strip * sky_sigma * sky_sigma + gain * nebula).sqrt()
        };
        let mut image = vec![0.0_f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let nebula = if x > 400 { 2000.0 } else { 0.0 };
                let mut signal = 100.0 + 0.05 * x as f64 + 0.03 * y as f64 + nebula;
                for &(cy, cx) in &stars {
                    let distance = (y as f64 - cy).powi(2) + (x as f64 - cx).powi(2);
                    if distance < 100.0 {
                        signal += 3000.0 * (-distance / (2.0 * 1.5 * 1.5)).exp();
                    }
                }
                image[y * width + x] = (signal + normal.sample() * sigma_at(x)) as f32;
            }
        }
        for y in 0..20 {
            image[y * width..y * width + 20].fill(f32::NAN);
        }
        let mapped = image_sigma(&image, width, height, 32, 1.0);
        assert!(mapped[0].is_nan() && mapped[19 * width + 19].is_nan());
        assert!(mapped[20 * width + 20].is_finite());

        let median_over = |xs: std::ops::Range<usize>, ys: std::ops::Range<usize>| {
            let mut values = Vec::new();
            for y in ys {
                for x in xs.clone() {
                    values.push(mapped[y * width + x]);
                }
            }
            median_f32(&mut values).unwrap() as f64
        };
        let sky = median_over(96..368, 0..height);
        assert!(
            (sky / sky_sigma - 1.0).abs() < 0.08,
            "the sky's noise, the stars left out: {sky}"
        );
        let edge = median_over(8..48, 24..height);
        assert!(
            (edge / (2.0 * sky_sigma) - 1.0).abs() < 0.15,
            "the noisier edge: {edge}"
        );
        let inside = median_over(440..width, 0..height);
        assert!(
            (inside / sigma_at(500) - 1.0).abs() < 0.1,
            "the nebula's: {inside}"
        );
    }

    #[test]
    fn a_constant_map_holds_the_image_map_s_median_where_the_image_is_finite() {
        let (width, height) = (96, 64);
        let mut normal = Normal(11);
        let mut image: Vec<f32> = (0..width * height)
            .map(|_| (100.0 + 3.0 * normal.sample()) as f32)
            .collect();
        image[0] = f32::NAN;
        let constant = input_sigma(&image, width, height, &reading(NoiseInput::Constant));
        let mapped = input_sigma(&image, width, height, &reading(NoiseInput::Image));
        let mut finite: Vec<f32> = mapped.iter().copied().filter(|v| v.is_finite()).collect();
        let level = median_f32(&mut finite).unwrap();
        assert!(constant[0].is_nan());
        assert!(constant[1..].iter().all(|&value| value == level));
        assert!(
            (level / 3.0 - 1.0).abs() < 0.15,
            "the image's noise: {level}"
        );
    }
}
