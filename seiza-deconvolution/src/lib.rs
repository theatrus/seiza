//! Conservative deconvolution for linear astrophotography images.
//!
//! The initial implementation is a damped Richardson-Lucy update with a
//! spatially invariant circular Gaussian point-spread function (PSF). It is
//! deliberately narrow: callers must provide a measured stellar FWHM, and
//! defaults use a small iteration count plus partial blending to reduce noise
//! amplification and ringing. This is a classical restoration experiment, not
//! a learned image model and not a substitute for a spatially varying PSF.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const FWHM_TO_SIGMA: f32 = 1.0 / 2.354_82;

/// Bump this when the same inputs and settings may yield different pixels.
pub const ALGORITHM_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeconvolutionConfig {
    /// Measured stellar full width at half maximum, in pixels.
    pub psf_fwhm_pixels: f32,
    /// Richardson-Lucy update count. Small values are intentionally preferred.
    pub iterations: usize,
    /// Blend of the restored estimate into the original, in `[0, 1]`.
    pub amount: f32,
    /// Residual damping floor as a fraction of each channel's sample range.
    pub noise_fraction: f32,
    /// Per-iteration multiplicative correction limit, greater than `1`.
    pub max_correction: f32,
}

impl DeconvolutionConfig {
    /// Conservative settings for a measured stellar FWHM.
    pub fn conservative(psf_fwhm_pixels: f32) -> Self {
        Self {
            psf_fwhm_pixels,
            iterations: 4,
            amount: 0.35,
            noise_fraction: 0.001,
            max_correction: 2.0,
        }
    }

    /// Check that every setting lies in its supported range.
    pub fn validate(self) -> Result<()> {
        if !self.psf_fwhm_pixels.is_finite() || !(0.25..=100.0).contains(&self.psf_fwhm_pixels) {
            return Err(DeconvolutionError::Invalid(
                "PSF FWHM must be finite and in [0.25, 100] pixels".into(),
            ));
        }
        if self.iterations == 0 || self.iterations > 50 {
            return Err(DeconvolutionError::Invalid(
                "iterations must be in 1..=50".into(),
            ));
        }
        if !self.amount.is_finite() || !(0.0..=1.0).contains(&self.amount) {
            return Err(DeconvolutionError::Invalid(
                "amount must be finite and in [0, 1]".into(),
            ));
        }
        if !self.noise_fraction.is_finite() || !(0.0..=0.25).contains(&self.noise_fraction) {
            return Err(DeconvolutionError::Invalid(
                "noise fraction must be finite and in [0, 0.25]".into(),
            ));
        }
        if !self.max_correction.is_finite() || !(1.0..=100.0).contains(&self.max_correction) {
            return Err(DeconvolutionError::Invalid(
                "maximum correction must be finite and in [1, 100]".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChannelDiagnostics {
    pub input_flux: f64,
    pub output_flux: f64,
    pub input_peak: f32,
    pub output_peak: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeconvolutionResult {
    /// Row-major samples with the same interleaved channel layout as the input.
    pub data: Vec<f32>,
    pub channels: Vec<ChannelDiagnostics>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeconvolutionError {
    #[error("invalid deconvolution request: {0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, DeconvolutionError>;

/// Apply damped Richardson-Lucy deconvolution to a linear mono or interleaved
/// RGB image.
///
/// Each channel is restored independently with the same circular Gaussian
/// PSF. This function requires finite samples; use [`deconvolve_masked`] for
/// registered images with missing border pixels. Negative linear values use a
/// reversible per-channel offset, and each channel keeps its input flux.
pub fn deconvolve(
    data: &[f32],
    width: usize,
    height: usize,
    channel_count: usize,
    config: &DeconvolutionConfig,
) -> Result<DeconvolutionResult> {
    deconvolve_impl(data, width, height, channel_count, config, false)
}

/// Apply damped Richardson-Lucy deconvolution while treating non-finite
/// samples as a missing-data mask.
///
/// The output keeps each masked sample non-finite. Convolutions divide by the
/// valid PSF support near mask edges, so missing registration borders do not
/// darken or contaminate nearby pixels.
pub fn deconvolve_masked(
    data: &[f32],
    width: usize,
    height: usize,
    channel_count: usize,
    config: &DeconvolutionConfig,
) -> Result<DeconvolutionResult> {
    deconvolve_impl(data, width, height, channel_count, config, true)
}

fn deconvolve_impl(
    data: &[f32],
    width: usize,
    height: usize,
    channel_count: usize,
    config: &DeconvolutionConfig,
    allow_masked: bool,
) -> Result<DeconvolutionResult> {
    config.validate()?;
    if width == 0 || height == 0 || !matches!(channel_count, 1 | 3) {
        return Err(DeconvolutionError::Invalid(
            "dimensions must be non-zero and channel count must be 1 or 3".into(),
        ));
    }
    let expected = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(channel_count))
        .ok_or_else(|| DeconvolutionError::Invalid("image dimensions overflow".into()))?;
    if data.len() != expected {
        return Err(DeconvolutionError::Invalid(format!(
            "pixel buffer has {} samples; expected {expected}",
            data.len()
        )));
    }
    let kernel = gaussian_kernel(config.psf_fwhm_pixels);
    let convolver = SeparableConvolver::new(width, height, &kernel);
    if channel_count == 1 {
        let restored = restore_channel(data, &convolver, config, allow_masked)?;
        return Ok(DeconvolutionResult {
            data: restored.data,
            channels: vec![restored.diagnostics],
        });
    }

    let restored = (0..channel_count)
        .into_par_iter()
        .map(|channel| {
            let input = data
                .iter()
                .skip(channel)
                .step_by(channel_count)
                .copied()
                .collect::<Vec<_>>();
            restore_channel(&input, &convolver, config, allow_masked)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut output = vec![0.0; data.len()];
    let mut diagnostics = Vec::with_capacity(channel_count);
    for (channel, restored) in restored.into_iter().enumerate() {
        for (index, sample) in restored.data.into_iter().enumerate() {
            output[index * channel_count + channel] = sample;
        }
        diagnostics.push(restored.diagnostics);
    }
    Ok(DeconvolutionResult {
        data: output,
        channels: diagnostics,
    })
}

struct RestoredChannel {
    data: Vec<f32>,
    diagnostics: ChannelDiagnostics,
}

fn restore_channel(
    input: &[f32],
    convolver: &SeparableConvolver<'_>,
    config: &DeconvolutionConfig,
    allow_masked: bool,
) -> Result<RestoredChannel> {
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    let mut input_flux = 0.0_f64;
    let mut finite_samples = 0_usize;
    for &sample in input {
        if !sample.is_finite() {
            if allow_masked {
                continue;
            }
            return Err(DeconvolutionError::Invalid(
                "all input samples must be finite".into(),
            ));
        }
        finite_samples += 1;
        minimum = minimum.min(sample);
        maximum = maximum.max(sample);
        input_flux += f64::from(sample);
    }
    if finite_samples == 0 {
        return Err(DeconvolutionError::Invalid(
            "at least one input sample must be finite".into(),
        ));
    }
    let input_peak = maximum;
    let range = maximum - minimum;
    if range <= f32::EPSILON || config.amount == 0.0 {
        return Ok(RestoredChannel {
            data: input.to_vec(),
            diagnostics: ChannelDiagnostics {
                input_flux,
                output_flux: input_flux,
                input_peak,
                output_peak: input_peak,
            },
        });
    }

    let offset = (-minimum).max(0.0);
    let observed = input
        .iter()
        .map(|&sample| {
            if sample.is_finite() {
                (sample + offset).max(0.0)
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let observed_flux = observed
        .iter()
        .map(|&sample| f64::from(sample))
        .sum::<f64>();
    let epsilon = (range * 1.0e-7).max(f32::MIN_POSITIVE);
    let noise_floor = range * config.noise_fraction;
    let minimum_correction = config.max_correction.recip();
    let mut estimate = observed.clone();
    let mut predicted = vec![0.0; input.len()];
    let mut ratio = vec![0.0; input.len()];
    let mut correction = vec![0.0; input.len()];
    let mut convolution_scratch = vec![0.0; input.len()];
    let mask_support = if finite_samples == input.len() {
        None
    } else {
        let mask = input
            .iter()
            .map(|sample| if sample.is_finite() { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let mut support = vec![0.0; input.len()];
        convolver.convolve_into(&mask, &mut support, &mut convolution_scratch);
        Some((mask, support))
    };

    for _ in 0..config.iterations {
        convolver.convolve_into(&estimate, &mut predicted, &mut convolution_scratch);
        normalize_masked_convolution(&mut predicted, mask_support.as_ref());
        if let Some((mask, _)) = &mask_support {
            ratio
                .par_iter_mut()
                .zip(&observed)
                .zip(&predicted)
                .zip(mask)
                .for_each(|(((ratio, &actual), &model), &valid)| {
                    *ratio = if valid == 0.0 {
                        0.0
                    } else {
                        (actual / model.max(epsilon))
                            .clamp(minimum_correction, config.max_correction)
                    };
                });
        } else {
            ratio
                .par_iter_mut()
                .zip(&observed)
                .zip(&predicted)
                .for_each(|((ratio, &actual), &model)| {
                    *ratio = (actual / model.max(epsilon))
                        .clamp(minimum_correction, config.max_correction);
                });
        }
        convolver.convolve_into(&ratio, &mut correction, &mut convolution_scratch);
        normalize_masked_convolution(&mut correction, mask_support.as_ref());
        if let Some((mask, _)) = &mask_support {
            estimate
                .par_iter_mut()
                .zip(&predicted)
                .zip(&observed)
                .zip(&correction)
                .zip(mask)
                .for_each(
                    |((((estimate, &predicted), &observed), &correction), &valid)| {
                        if valid != 0.0 {
                            update_estimate(estimate, predicted, observed, correction, noise_floor);
                        }
                    },
                );
        } else {
            estimate
                .par_iter_mut()
                .zip(&predicted)
                .zip(&observed)
                .zip(&correction)
                .for_each(|(((estimate, &predicted), &observed), &correction)| {
                    update_estimate(estimate, predicted, observed, correction, noise_floor);
                });
        }
    }

    let estimated_flux = estimate
        .iter()
        .map(|&sample| f64::from(sample))
        .sum::<f64>();
    let flux_scale = if estimated_flux > f64::from(epsilon) {
        (observed_flux / estimated_flux) as f32
    } else {
        1.0
    };
    drop(observed);
    drop(ratio);
    drop(correction);
    drop(convolution_scratch);

    let amount = config.amount;
    let masked = mask_support.is_some();
    drop(mask_support);
    let mut data = if masked { input.to_vec() } else { predicted };
    if masked {
        data.par_iter_mut()
            .zip(&estimate)
            .for_each(|(output, &estimate)| {
                let original = *output;
                if original.is_finite() {
                    let restored = estimate.mul_add(flux_scale, -offset);
                    *output = amount.mul_add(restored - original, original);
                }
            });
    } else {
        data.par_iter_mut().zip(input).zip(&estimate).for_each(
            |((output, &original), &estimate)| {
                let restored = estimate.mul_add(flux_scale, -offset);
                *output = amount.mul_add(restored - original, original);
            },
        );
    }
    let output_flux = data
        .iter()
        .filter(|sample| sample.is_finite())
        .map(|&sample| f64::from(sample))
        .sum::<f64>();
    let output_peak = data
        .iter()
        .copied()
        .filter(|sample| sample.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    Ok(RestoredChannel {
        data,
        diagnostics: ChannelDiagnostics {
            input_flux,
            output_flux,
            input_peak,
            output_peak,
        },
    })
}

#[inline]
fn update_estimate(
    estimate: &mut f32,
    predicted: f32,
    observed: f32,
    correction: f32,
    noise_floor: f32,
) {
    let proposed = (*estimate * correction).max(0.0);
    let residual = (observed - predicted).abs();
    let activity = if noise_floor > 0.0 {
        residual / (residual + noise_floor)
    } else {
        1.0
    };
    *estimate = activity.mul_add(proposed - *estimate, *estimate);
}

fn normalize_masked_convolution(output: &mut [f32], mask_support: Option<&(Vec<f32>, Vec<f32>)>) {
    let Some((mask, support)) = mask_support else {
        return;
    };
    output
        .par_iter_mut()
        .zip(mask)
        .zip(support)
        .for_each(|((output, &valid), &support)| {
            *output = if valid == 0.0 {
                0.0
            } else {
                *output / support.max(f32::MIN_POSITIVE)
            };
        });
}

fn gaussian_kernel(fwhm_pixels: f32) -> Vec<f32> {
    let sigma = fwhm_pixels * FWHM_TO_SIGMA;
    let radius = (3.0 * sigma).ceil().max(1.0) as isize;
    let mut kernel = (-radius..=radius)
        .map(|offset| (-0.5 * (offset as f32 / sigma).powi(2)).exp())
        .collect::<Vec<_>>();
    let sum = kernel.iter().sum::<f32>();
    kernel.iter_mut().for_each(|weight| *weight /= sum);
    kernel
}

struct SeparableConvolver<'a> {
    width: usize,
    height: usize,
    kernel: &'a [f32],
    horizontal_indices: Vec<usize>,
    vertical_indices: Vec<usize>,
}

impl<'a> SeparableConvolver<'a> {
    fn new(width: usize, height: usize, kernel: &'a [f32]) -> Self {
        Self {
            width,
            height,
            kernel,
            horizontal_indices: reflected_indices(width, kernel.len()),
            vertical_indices: reflected_indices(height, kernel.len()),
        }
    }

    fn convolve_into(&self, input: &[f32], output: &mut [f32], scratch: &mut [f32]) {
        debug_assert_eq!(input.len(), self.width * self.height);
        debug_assert_eq!(output.len(), input.len());
        debug_assert_eq!(scratch.len(), input.len());
        let kernel_length = self.kernel.len();

        scratch
            .par_chunks_mut(self.width)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, output) in row.iter_mut().enumerate() {
                    let indices =
                        &self.horizontal_indices[x * kernel_length..(x + 1) * kernel_length];
                    *output = self
                        .kernel
                        .iter()
                        .zip(indices)
                        .map(|(&weight, &source_x)| weight * input[y * self.width + source_x])
                        .sum();
                }
            });

        output
            .par_chunks_mut(self.width)
            .enumerate()
            .for_each(|(y, row)| {
                let y_indices = &self.vertical_indices[y * kernel_length..(y + 1) * kernel_length];
                for (x, output) in row.iter_mut().enumerate() {
                    *output = self
                        .kernel
                        .iter()
                        .zip(y_indices)
                        .map(|(&weight, &source_y)| weight * scratch[source_y * self.width + x])
                        .sum();
                }
            });
    }
}

fn reflected_indices(length: usize, kernel_length: usize) -> Vec<usize> {
    let radius = (kernel_length / 2) as isize;
    (0..length)
        .flat_map(|index| {
            (0..kernel_length)
                .map(move |tap| reflect(index as isize + tap as isize - radius, length))
        })
        .collect()
}

#[cfg(test)]
fn convolve_separable(input: &[f32], width: usize, height: usize, kernel: &[f32]) -> Vec<f32> {
    let convolver = SeparableConvolver::new(width, height, kernel);
    let mut output = vec![0.0; input.len()];
    let mut scratch = vec![0.0; input.len()];
    convolver.convolve_into(input, &mut output, &mut scratch);
    output
}

fn reflect(index: isize, length: usize) -> usize {
    if length == 1 {
        return 0;
    }
    let period = 2 * (length as isize - 1);
    let index = index.rem_euclid(period);
    if index >= length as isize {
        (period - index) as usize
    } else {
        index as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> DeconvolutionConfig {
        DeconvolutionConfig {
            psf_fwhm_pixels: 2.8,
            iterations: 6,
            amount: 0.5,
            noise_fraction: 0.0,
            max_correction: 3.0,
        }
    }

    #[test]
    fn conservative_defaults_are_deliberately_light() {
        let config = DeconvolutionConfig::conservative(3.0);
        assert_eq!(config.iterations, 4);
        assert_eq!(config.amount, 0.35);
        assert!(config.noise_fraction > 0.0);
    }

    #[test]
    fn restores_peak_and_reduces_second_moment_of_blurred_star() {
        let size = 41;
        let center = size / 2;
        let mut point = vec![0.0; size * size];
        point[center * size + center] = 1.0;
        let kernel = gaussian_kernel(config().psf_fwhm_pixels);
        let blurred = convolve_separable(&point, size, size, &kernel);
        let restored = deconvolve(&blurred, size, size, 1, &config()).unwrap();

        let moment = |image: &[f32]| {
            image
                .iter()
                .enumerate()
                .map(|(index, &sample)| {
                    let x = index % size;
                    let y = index / size;
                    let radius_squared = ((x as isize - center as isize).pow(2)
                        + (y as isize - center as isize).pow(2))
                        as f64;
                    f64::from(sample) * radius_squared
                })
                .sum::<f64>()
        };
        assert!(restored.data[center * size + center] > blurred[center * size + center]);
        assert!(moment(&restored.data) < moment(&blurred));
        assert!((restored.data.iter().sum::<f32>() - 1.0).abs() < 1.0e-4);
    }

    #[test]
    fn constant_field_and_rgb_layout_are_preserved() {
        let input = [2.0, 4.0, 8.0].repeat(9 * 7);
        let restored = deconvolve(&input, 9, 7, 3, &config()).unwrap();
        assert_eq!(restored.data, input);
        assert_eq!(restored.channels.len(), 3);
    }

    #[test]
    fn restores_one_rgb_channel_without_mixing_constant_channels() {
        let size = 21;
        let center = size / 2;
        let mut point = vec![0.0; size * size];
        point[center * size + center] = 1.0;
        let kernel = gaussian_kernel(config().psf_fwhm_pixels);
        let blurred = convolve_separable(&point, size, size, &kernel);
        let input = blurred
            .iter()
            .flat_map(|&red| [red, 4.0, 8.0])
            .collect::<Vec<_>>();

        let restored = deconvolve(&input, size, size, 3, &config()).unwrap();

        assert!(restored.data[(center * size + center) * 3] > blurred[center * size + center]);
        assert!(
            restored
                .data
                .chunks_exact(3)
                .all(|pixel| pixel[1] == 4.0 && pixel[2] == 8.0)
        );
    }

    #[test]
    fn accepts_negative_linear_samples_without_changing_flux() {
        let mut input = vec![-0.25; 13 * 13];
        input[6 * 13 + 6] = 4.0;
        let restored = deconvolve(&input, 13, 13, 1, &config()).unwrap();
        let input_flux = input.iter().sum::<f32>();
        let output_flux = restored.data.iter().sum::<f32>();
        assert!((output_flux - input_flux).abs() < 1.0e-3);
        assert!(restored.data.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn masked_entry_point_keeps_the_finite_fast_path_exact() {
        let mut input = vec![-0.25; 13 * 13];
        input[6 * 13 + 6] = 4.0;

        let strict = deconvolve(&input, 13, 13, 1, &config()).unwrap();
        let masked = deconvolve_masked(&input, 13, 13, 1, &config()).unwrap();

        assert_eq!(masked, strict);
    }

    #[test]
    fn rejects_invalid_configuration_and_nonfinite_samples() {
        let mut invalid = config();
        invalid.amount = 1.1;
        assert!(deconvolve(&[1.0], 1, 1, 1, &invalid).is_err());
        assert!(deconvolve(&[f32::NAN], 1, 1, 1, &config()).is_err());
        assert!(deconvolve(&[1.0, 2.0], 1, 1, 1, &config()).is_err());
    }

    #[test]
    fn masked_constant_field_keeps_its_values_and_mask() {
        let width = 13;
        let height = 11;
        let mut input = vec![4.0; width * height];
        for y in 0..height {
            input[y * width] = f32::NAN;
        }
        for sample in input.iter_mut().take(width) {
            *sample = f32::NAN;
        }

        let restored = deconvolve_masked(&input, width, height, 1, &config()).unwrap();

        for (before, after) in input.iter().zip(&restored.data) {
            if before.is_finite() {
                assert_eq!(before, after);
            } else {
                assert!(after.is_nan());
            }
        }
    }

    #[test]
    fn masked_star_restoration_keeps_registration_border_out_of_the_solution() {
        let size = 41;
        let center = size / 2;
        let mut point = vec![0.0; size * size];
        point[center * size + center] = 1.0;
        let kernel = gaussian_kernel(config().psf_fwhm_pixels);
        let mut input = convolve_separable(&point, size, size, &kernel);
        for y in 0..size {
            input[y * size] = f32::NAN;
            input[y * size + 1] = f32::NAN;
        }
        for sample in input.iter_mut().take(size) {
            *sample = f32::NAN;
        }
        let input_flux = input
            .iter()
            .filter(|sample| sample.is_finite())
            .sum::<f32>();

        let restored = deconvolve_masked(&input, size, size, 1, &config()).unwrap();

        assert!(restored.data[center * size + center] > input[center * size + center]);
        assert!(
            restored
                .data
                .iter()
                .zip(&input)
                .all(|(after, before)| after.is_finite() == before.is_finite())
        );
        let output_flux = restored
            .data
            .iter()
            .filter(|sample| sample.is_finite())
            .sum::<f32>();
        assert!((output_flux - input_flux).abs() < 1.0e-4);
    }
}
