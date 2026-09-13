//! Area averaging for reduced previews. Every source pixel covered by an
//! output pixel contributes, including fractional edges. Unlike point or
//! bilinear sampling, the filter grows with the reduction ratio.

pub(super) struct Footprint {
    pub(super) start: usize,
    pub(super) weights: Vec<f64>,
}

pub(super) fn footprints(source: usize, target: usize) -> Vec<Footprint> {
    let scale = source as f64 / target as f64;
    (0..target)
        .map(|index| {
            let left = index as f64 * scale;
            let right = ((index + 1) as f64 * scale).min(source as f64);
            let start = left.floor() as usize;
            let end = (right.ceil() as usize).min(source);
            Footprint {
                start,
                weights: (start..end)
                    .map(|pixel| (right.min((pixel + 1) as f64) - left.max(pixel as f64)) / scale)
                    .collect(),
            }
        })
        .collect()
}

fn dimensions(width: usize, height: usize, limit: usize) -> (usize, usize) {
    if limit == 0 || width.max(height) <= limit {
        return (width, height);
    }
    let scale = limit as f64 / width.max(height) as f64;
    (
        ((width as f64 * scale).round() as usize).max(1),
        ((height as f64 * scale).round() as usize).max(1),
    )
}

pub(crate) trait RgbaSample: Copy + Into<f64> {
    fn rounded(value: f64) -> Self;
}

impl RgbaSample for u8 {
    fn rounded(value: f64) -> Self {
        value.round() as Self
    }
}

impl RgbaSample for u16 {
    fn rounded(value: f64) -> Self {
        value.round() as Self
    }
}

/// Resample straight-alpha display samples in their existing transfer space.
/// Weight colors by alpha so missing stack coverage cannot add a dark fringe.
/// Full-size exports return the original buffer without changing any samples.
pub(crate) fn downsample_rgba<T: RgbaSample>(
    width: usize,
    height: usize,
    rgba: Vec<T>,
    max_dimension: usize,
) -> (usize, usize, Vec<T>) {
    let (output_width, output_height) = dimensions(width, height, max_dimension);
    if (width, height) == (output_width, output_height) {
        return (width, height, rgba);
    }
    let xs = footprints(width, output_width);
    let ys = footprints(height, output_height);
    let mut output = Vec::with_capacity(output_width * output_height * 4);
    for y in &ys {
        for x in &xs {
            let mut sum = [0.0; 4];
            for (dy, wy) in y.weights.iter().enumerate() {
                for (dx, wx) in x.weights.iter().enumerate() {
                    let offset = ((y.start + dy) * width + x.start + dx) * 4;
                    let alpha_weight = rgba[offset + 3].into() * wx * wy;
                    for channel in 0..3 {
                        sum[channel] += rgba[offset + channel].into() * alpha_weight;
                    }
                    sum[3] += alpha_weight;
                }
            }
            for channel in 0..3 {
                output.push(T::rounded(if sum[3] > 0.0 {
                    sum[channel] / sum[3]
                } else {
                    0.0
                }));
            }
            output.push(T::rounded(sum[3]));
        }
    }
    (output_width, output_height, output)
}

/// Bounds interactive work before background fitting and stretch stages.
/// Average linear samples, excluding non-finite samples per channel. Keep an
/// all-invalid footprint invalid rather than inventing a black measurement.
pub(crate) fn downsample_interleaved_f32(
    width: usize,
    height: usize,
    pixels: Vec<f32>,
    channels: usize,
    max_dimension: usize,
) -> (usize, usize, Vec<f32>) {
    let (output_width, output_height) = dimensions(width, height, max_dimension);
    if (width, height) == (output_width, output_height) {
        return (width, height, pixels);
    }
    let xs = footprints(width, output_width);
    let ys = footprints(height, output_height);
    let mut output = Vec::with_capacity(output_width * output_height * channels);
    let mut sums = vec![0.0; channels];
    let mut weights = vec![0.0; channels];
    for y in &ys {
        for x in &xs {
            sums.fill(0.0);
            weights.fill(0.0);
            for (dy, wy) in y.weights.iter().enumerate() {
                for (dx, wx) in x.weights.iter().enumerate() {
                    let offset = ((y.start + dy) * width + x.start + dx) * channels;
                    let weight = wx * wy;
                    for channel in 0..channels {
                        let value = pixels[offset + channel];
                        if value.is_finite() {
                            sums[channel] += f64::from(value) * weight;
                            weights[channel] += weight;
                        }
                    }
                }
            }
            for channel in 0..channels {
                output.push(if weights[channel] > 0.0 {
                    (sums[channel] / weights[channel]) as f32
                } else {
                    f32::NAN
                });
            }
        }
    }
    (output_width, output_height, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkerboard_averages_instead_of_aliasing() {
        let pixels = (0..64)
            .flat_map(|i| {
                let value = if (i / 8 + i % 8) % 2 == 0 { 0_u8 } else { 255 };
                [value, value, value, 255]
            })
            .collect();
        let (w, h, result) = downsample_rgba(8, 8, pixels, 2);
        assert_eq!((w, h), (2, 2));
        assert_eq!(result, [128, 128, 128, 255].repeat(4));
    }

    #[test]
    fn fractional_footprints_include_edges_and_preserve_mean() {
        let (_, _, result) = downsample_interleaved_f32(5, 1, vec![0., 0., 10., 0., 0.], 1, 2);
        assert_eq!(result, [2., 2.]);
        let (_, _, result) = downsample_interleaved_f32(5, 1, vec![10., 0., 0., 0., 0.], 1, 2);
        assert_eq!(result, [4., 0.]);
    }

    #[test]
    fn averages_each_rgb_channel_without_losing_u16_precision() {
        let (_, _, result) = downsample_rgba(
            2,
            1,
            vec![1000_u16, 2000, 3000, 65535, 1002, 2004, 3006, 65535],
            1,
        );
        assert_eq!(result, [1001, 2002, 3003, 65535]);
        let (_, _, result) =
            downsample_interleaved_f32(2, 1, vec![-10., 100., 1000., 30., 200., 2000.], 3, 1);
        assert_eq!(result, [10., 150., 1500.]);
    }

    #[test]
    fn transparent_pixels_do_not_add_dark_or_colored_fringes() {
        let (_, _, result) = downsample_rgba(2, 1, vec![200_u8, 80, 40, 255, 0, 255, 0, 0], 1);
        assert_eq!(result, [200, 80, 40, 128]);
        let (_, _, result) = downsample_rgba(2, 1, [255_u8, 0, 0, 0].repeat(2), 1);
        assert_eq!(result, [0, 0, 0, 0]);
    }

    #[test]
    fn full_size_and_upscale_requests_leave_samples_untouched() {
        let pixels = [17_u16, 129, 2049, 65535].repeat(6);
        for limit in [0, 3, 10] {
            let (w, h, result) = downsample_rgba(3, 2, pixels.clone(), limit);
            assert_eq!((w, h), (3, 2));
            assert_eq!(result, pixels);
        }
    }

    #[test]
    fn constant_fields_survive_non_integer_reduction() {
        let (w, h, result) = downsample_rgba(17, 11, [51_u8, 102, 153, 255].repeat(187), 7);
        assert_eq!((w, h), (7, 5));
        assert_eq!(result, [51, 102, 153, 255].repeat(35));
    }

    #[test]
    fn linear_preview_uses_whole_footprint_not_only_center_samples() {
        let mut pixels = vec![0.; 64];
        pixels[0] = 16.;
        let (_, _, result) = downsample_interleaved_f32(8, 8, pixels, 1, 2);
        assert_eq!(result, [1., 0., 0., 0.]);
    }

    #[test]
    fn four_by_four_reduction_lowers_uncorrelated_noise() {
        let mut state = 123456789_u32;
        let pixels: Vec<f32> = (0..256 * 256)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 8) as f32 / 16777216.0
            })
            .collect();
        let statistics = |values: &[f32]| {
            let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / values.len() as f64;
            let variance = values
                .iter()
                .map(|&v| (f64::from(v) - mean).powi(2))
                .sum::<f64>()
                / values.len() as f64;
            (mean, variance.sqrt())
        };
        let before = statistics(&pixels);
        let (_, _, reduced) = downsample_interleaved_f32(256, 256, pixels, 1, 64);
        let after = statistics(&reduced);
        assert!((before.0 - after.0).abs() < 1e-7);
        assert!((after.1 / before.1 - 0.25).abs() < 0.02);
    }

    #[test]
    fn linear_preview_handles_missing_samples_per_channel() {
        let (_, _, result) = downsample_interleaved_f32(
            2,
            1,
            vec![f32::NAN, 10., f32::INFINITY, 20., 30., f32::NAN],
            3,
            1,
        );
        assert_eq!(&result[..2], &[20., 20.]);
        assert!(result[2].is_nan());
    }
}
