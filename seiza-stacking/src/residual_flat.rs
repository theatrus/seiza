//! Bounded multiplicative response patches estimated from repeated light frames.
//!
//! This module does not classify a feature as dust. Callers must first show
//! that the feature stays fixed on the detector while sky content moves.

use crate::{Error, LinearImage, Result};
use seiza_imgproc::{
    BorderMode,
    blur::gaussian_blur_f32,
    components::{Connectivity, largest_connected_component},
};
use seiza_stats::{median_in_place, robust_sigma_f64};
use serde::{Deserialize, Serialize};

const MAX_PLANE_SAMPLES: usize = 30_000;

/// Bump when equal inputs and options may yield different response pixels.
pub const RESIDUAL_FLAT_ALGORITHM_VERSION: u32 = 1;

const fn residual_flat_algorithm_version() -> u32 {
    RESIDUAL_FLAT_ALGORITHM_VERSION
}

/// Controls robust response estimation and the maximum correction gain.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ResidualFlatOptions {
    /// Fewest detector-aligned light-frame crops needed for an estimate.
    pub minimum_samples: usize,
    /// Fractional response loss that must remain after smoothing.
    pub minimum_depth: f32,
    /// Fraction of samples that must show the loss at a pixel.
    pub minimum_consensus: f32,
    /// Largest multiplier that correction may apply to any sample.
    pub maximum_gain: f32,
    /// Fraction of each edge used to estimate the local background plane.
    pub background_edge_fraction: f32,
    /// Gaussian sigma used to suppress stars and pixel noise before consensus.
    pub smoothing_sigma: f32,
    /// Fraction of each patch edge blended back to a neutral response.
    pub edge_feather_fraction: f32,
    /// Fewest corrected pixel-channel samples accepted as a useful patch.
    pub minimum_corrected_samples: usize,
    /// Fewest adjacent corrected detector pixels accepted as a useful patch.
    pub minimum_connected_pixels: usize,
}

impl Default for ResidualFlatOptions {
    fn default() -> Self {
        Self {
            minimum_samples: 5,
            minimum_depth: 0.005,
            minimum_consensus: 0.7,
            maximum_gain: 1.2,
            background_edge_fraction: 0.2,
            smoothing_sigma: 2.0,
            edge_feather_fraction: 0.12,
            minimum_corrected_samples: 16,
            minimum_connected_pixels: 64,
        }
    }
}

impl ResidualFlatOptions {
    fn validate(&self) -> Result<()> {
        if self.minimum_samples < 3 {
            return Err(residual_error("minimum_samples must be at least 3"));
        }
        if !self.minimum_depth.is_finite() || !(0.0..0.5).contains(&self.minimum_depth) {
            return Err(residual_error(
                "minimum_depth must be finite and between 0 and 0.5",
            ));
        }
        if !self.minimum_consensus.is_finite() || !(0.5..=1.0).contains(&self.minimum_consensus) {
            return Err(residual_error(
                "minimum_consensus must be finite and between 0.5 and 1",
            ));
        }
        if !self.maximum_gain.is_finite() || !(1.0..=2.0).contains(&self.maximum_gain) {
            return Err(residual_error(
                "maximum_gain must be finite and between 1 and 2",
            ));
        }
        if !self.background_edge_fraction.is_finite()
            || !(0.05..=0.45).contains(&self.background_edge_fraction)
        {
            return Err(residual_error(
                "background_edge_fraction must be finite and between 0.05 and 0.45",
            ));
        }
        if !self.smoothing_sigma.is_finite() || !(0.0..=32.0).contains(&self.smoothing_sigma) {
            return Err(residual_error(
                "smoothing_sigma must be finite and between 0 and 32",
            ));
        }
        if !self.edge_feather_fraction.is_finite()
            || !(0.0..=0.45).contains(&self.edge_feather_fraction)
        {
            return Err(residual_error(
                "edge_feather_fraction must be finite and between 0 and 0.45",
            ));
        }
        if self.minimum_corrected_samples == 0 {
            return Err(residual_error(
                "minimum_corrected_samples must be greater than zero",
            ));
        }
        if self.minimum_connected_pixels == 0 {
            return Err(residual_error(
                "minimum_connected_pixels must be greater than zero",
            ));
        }
        Ok(())
    }
}

/// Measurements retained with a generated response patch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidualFlatDiagnostics {
    /// Numeric algorithm revision for cache keys and provenance.
    #[serde(default = "residual_flat_algorithm_version")]
    pub algorithm_version: u32,
    /// Number of source crops used.
    pub sample_count: usize,
    /// Pixel-channel samples with enough repeated evidence to correct.
    pub corrected_samples: usize,
    /// Total pixel-channel samples in the patch.
    pub total_samples: usize,
    /// Corrected detector pixels in the largest connected region.
    pub largest_connected_pixels: usize,
    /// Lowest response retained before division.
    pub minimum_response: f32,
    /// Largest gain the generated patch will apply.
    pub maximum_applied_gain: f32,
}

/// A normalized response patch. Values at one are neutral; lower values
/// describe bounded attenuation that correction divides out.
#[derive(Clone, Debug, PartialEq)]
pub struct ResidualFlatPatch {
    response: LinearImage,
}

impl ResidualFlatPatch {
    /// Validate a cached or externally generated normalized response.
    pub fn from_response(response: LinearImage) -> Result<Self> {
        if response
            .data
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0 || *value > 1.0)
        {
            return Err(residual_error(
                "residual-flat response values must be finite and in (0, 1]",
            ));
        }
        Ok(Self { response })
    }

    /// The normalized multiplicative response image.
    pub fn response(&self) -> &LinearImage {
        &self.response
    }

    /// Consume the patch and return its normalized response image.
    pub fn into_response(self) -> LinearImage {
        self.response
    }

    /// Divide this response out of a same-sampling image at a detector origin.
    pub fn apply_at(&self, image: &mut LinearImage, x: usize, y: usize) -> Result<()> {
        if image.channels != self.response.channels {
            return Err(residual_error(format!(
                "light has {} channels but residual-flat patch has {}",
                image.channels, self.response.channels
            )));
        }
        let end_x = x
            .checked_add(self.response.width)
            .ok_or_else(|| residual_error("residual-flat horizontal bounds overflow"))?;
        let end_y = y
            .checked_add(self.response.height)
            .ok_or_else(|| residual_error("residual-flat vertical bounds overflow"))?;
        if end_x > image.width || end_y > image.height {
            return Err(residual_error(format!(
                "residual-flat patch at ({x}, {y}) extends past the {}x{} light frame",
                image.width, image.height
            )));
        }

        for patch_y in 0..self.response.height {
            let light_start = ((y + patch_y) * image.width + x) * image.channels;
            let patch_start = patch_y * self.response.width * self.response.channels;
            let sample_count = self.response.width * self.response.channels;
            let light_row = &mut image.data[light_start..light_start + sample_count];
            let response_row = &self.response.data[patch_start..patch_start + sample_count];
            for (value, response) in light_row.iter_mut().zip(response_row) {
                *value /= *response;
            }
        }
        Ok(())
    }
}

/// A generated response patch and the evidence summary used to accept it.
#[derive(Clone, Debug, PartialEq)]
pub struct ResidualFlatBuild {
    pub patch: ResidualFlatPatch,
    pub diagnostics: ResidualFlatDiagnostics,
}

/// Estimate a bounded multiplicative response from detector-aligned crops.
///
/// Each crop gets a robust local background plane from its outer edge. The
/// builder divides that plane out, smooths the crop, and retains only dark
/// response shared by the requested fraction of source frames. Callers must
/// supply crops from the same detector position and sampling.
pub fn build_residual_flat_patch(
    samples: &[LinearImage],
    options: &ResidualFlatOptions,
) -> Result<ResidualFlatBuild> {
    options.validate()?;
    if samples.len() < options.minimum_samples {
        return Err(residual_error(format!(
            "found {} source crops; need at least {}",
            samples.len(),
            options.minimum_samples
        )));
    }
    let reference = samples
        .first()
        .ok_or_else(|| residual_error("no source crops were supplied"))?;
    if reference.width < 5 || reference.height < 5 {
        return Err(residual_error("source crops must be at least 5x5 pixels"));
    }
    if samples
        .iter()
        .any(|sample| !reference.dimensions_match(sample))
    {
        return Err(residual_error(
            "all source crops must have matching dimensions and channels",
        ));
    }

    let normalized = samples
        .iter()
        .map(|sample| normalize_sample(sample, options))
        .collect::<Result<Vec<_>>>()?;
    let threshold = 1.0 - options.minimum_depth;
    let response_floor = 1.0 / options.maximum_gain;
    let mut response = vec![1.0_f32; reference.sample_count()];
    let mut values = Vec::with_capacity(normalized.len());

    for sample_index in 0..response.len() {
        values.clear();
        values.extend(
            normalized
                .iter()
                .map(|sample| sample[sample_index])
                .filter(|value| value.is_finite() && *value > 0.0),
        );
        if values.len() < options.minimum_samples {
            continue;
        }
        let agreeing = values.iter().filter(|value| **value <= threshold).count();
        let consensus = agreeing as f32 / values.len() as f32;
        let median = median_in_place(&mut values).expect("values are non-empty");
        if median <= threshold && consensus >= options.minimum_consensus {
            let feather = edge_feather(
                sample_index / reference.channels,
                reference.width,
                reference.height,
                options.edge_feather_fraction,
            );
            let bounded = median.clamp(response_floor, 1.0);
            let value = 1.0 - feather * (1.0 - bounded);
            if value < 1.0 - f32::EPSILON {
                response[sample_index] = value;
            }
        }
    }

    let corrected_mask = response
        .chunks_exact(reference.channels)
        .map(|pixel| u8::from(pixel.iter().any(|value| *value < 1.0 - f32::EPSILON)))
        .collect::<Vec<_>>();
    let largest_component = largest_connected_component(
        &corrected_mask,
        reference.width,
        reference.height,
        Connectivity::Eight,
    );
    let largest_connected_pixels = largest_component
        .as_ref()
        .map_or(0, |component| component.pixels.len());
    if largest_connected_pixels < options.minimum_connected_pixels {
        return Err(residual_error(format!(
            "the largest repeated attenuation region has {largest_connected_pixels} connected pixels; need at least {}",
            options.minimum_connected_pixels
        )));
    }

    let mut retained_pixels = vec![false; reference.pixel_count()];
    for pixel in largest_component
        .expect("the accepted component is present")
        .pixels
    {
        retained_pixels[pixel] = true;
    }
    let mut corrected_samples = 0;
    let mut minimum_response = 1.0_f32;
    for (pixel, retained) in retained_pixels.into_iter().enumerate() {
        let start = pixel * reference.channels;
        let values = &mut response[start..start + reference.channels];
        if !retained {
            values.fill(1.0);
            continue;
        }
        for value in values {
            if *value < 1.0 - f32::EPSILON {
                corrected_samples += 1;
                minimum_response = minimum_response.min(*value);
            }
        }
    }
    if corrected_samples < options.minimum_corrected_samples {
        return Err(residual_error(format!(
            "only {corrected_samples} pixel-channel samples remained in the coherent attenuation region; need at least {}",
            options.minimum_corrected_samples
        )));
    }

    let response = LinearImage::new(
        reference.width,
        reference.height,
        reference.channels,
        response,
    )?;
    let patch = ResidualFlatPatch::from_response(response)?;
    Ok(ResidualFlatBuild {
        diagnostics: ResidualFlatDiagnostics {
            algorithm_version: RESIDUAL_FLAT_ALGORITHM_VERSION,
            sample_count: samples.len(),
            corrected_samples,
            total_samples: reference.sample_count(),
            largest_connected_pixels,
            minimum_response,
            maximum_applied_gain: 1.0 / minimum_response,
        },
        patch,
    })
}

fn normalize_sample(image: &LinearImage, options: &ResidualFlatOptions) -> Result<Vec<f32>> {
    let mut normalized = vec![f32::NAN; image.sample_count()];
    for channel in 0..image.channels {
        let plane = fit_background_plane(image, channel, options.background_edge_fraction)?;
        let mut channel_data = Vec::with_capacity(image.pixel_count());
        for y in 0..image.height {
            let normalized_y = normalized_coordinate(y, image.height);
            for x in 0..image.width {
                let normalized_x = normalized_coordinate(x, image.width);
                let background = plane[0] + plane[1] * normalized_x + plane[2] * normalized_y;
                let value = image.data[(y * image.width + x) * image.channels + channel];
                channel_data.push(
                    if value.is_finite() && background.is_finite() && background > 0.0 {
                        (f64::from(value) / background) as f32
                    } else {
                        f32::NAN
                    },
                );
            }
        }
        let channel_data = smooth_channel(
            &channel_data,
            image.width,
            image.height,
            options.smoothing_sigma,
        );
        for (pixel, value) in channel_data.into_iter().enumerate() {
            normalized[pixel * image.channels + channel] = value;
        }
    }
    Ok(normalized)
}

fn smooth_channel(data: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 {
        return data.to_vec();
    }
    let minimum_dimension = width.min(height);
    let maximum_kernel = if minimum_dimension.is_multiple_of(2) {
        minimum_dimension - 1
    } else {
        minimum_dimension
    };
    let requested = ((f64::from(sigma) * 6.0).ceil() as usize | 1).max(3);
    let kernel = requested.min(maximum_kernel);
    let values = data
        .iter()
        .map(|value| if value.is_finite() { *value } else { 0.0 })
        .collect::<Vec<_>>();
    let weights = data
        .iter()
        .map(|value| if value.is_finite() { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    let values = gaussian_blur_f32(
        &values,
        width,
        height,
        kernel,
        f64::from(sigma),
        BorderMode::Reflect,
    );
    let weights = gaussian_blur_f32(
        &weights,
        width,
        height,
        kernel,
        f64::from(sigma),
        BorderMode::Reflect,
    );
    values
        .into_iter()
        .zip(weights)
        .map(|(value, weight)| {
            if weight > 1.0e-6 {
                value / weight
            } else {
                f32::NAN
            }
        })
        .collect()
}

#[derive(Clone, Copy)]
struct PlaneSample {
    x: f64,
    y: f64,
    value: f64,
}

fn fit_background_plane(
    image: &LinearImage,
    channel: usize,
    edge_fraction: f32,
) -> Result<[f64; 3]> {
    let edge_x = ((image.width as f32 * edge_fraction).ceil() as usize).max(1);
    let edge_y = ((image.height as f32 * edge_fraction).ceil() as usize).max(1);
    let edge_count = image
        .pixel_count()
        .saturating_sub(
            image.width.saturating_sub(edge_x * 2) * image.height.saturating_sub(edge_y * 2),
        )
        .max(1);
    let stride = edge_count.div_ceil(MAX_PLANE_SAMPLES).max(1);
    let mut points = Vec::with_capacity(edge_count.min(MAX_PLANE_SAMPLES));
    let mut candidate = 0_usize;
    for y in 0..image.height {
        for x in 0..image.width {
            if x >= edge_x
                && x < image.width.saturating_sub(edge_x)
                && y >= edge_y
                && y < image.height.saturating_sub(edge_y)
            {
                continue;
            }
            let value = image.data[(y * image.width + x) * image.channels + channel];
            if value.is_finite() && value > 0.0 {
                if candidate.is_multiple_of(stride) {
                    points.push(PlaneSample {
                        x: normalized_coordinate(x, image.width),
                        y: normalized_coordinate(y, image.height),
                        value: f64::from(value),
                    });
                }
                candidate += 1;
            }
        }
    }
    if points.len() < 12 {
        return Err(residual_error(format!(
            "channel {channel} has only {} usable edge samples",
            points.len()
        )));
    }

    let mut active = vec![true; points.len()];
    let mut plane = solve_plane(&points, &active)
        .ok_or_else(|| residual_error("local background plane is singular"))?;
    for _ in 0..3 {
        let residuals = points
            .iter()
            .zip(&active)
            .filter(|(_, active)| **active)
            .map(|(point, _)| point.value - evaluate_plane(plane, point.x, point.y))
            .collect::<Vec<_>>();
        let center = seiza_stats::median_f64(&residuals).unwrap_or(0.0);
        let sigma = robust_sigma_f64(&residuals, center).unwrap_or(0.0);
        if sigma <= f64::EPSILON {
            break;
        }
        let limit = 3.5 * sigma;
        let mut changed = false;
        for (index, point) in points.iter().enumerate() {
            let keep =
                (point.value - evaluate_plane(plane, point.x, point.y) - center).abs() <= limit;
            changed |= keep != active[index];
            active[index] = keep;
        }
        if active.iter().filter(|active| **active).count() < 12 {
            return Err(residual_error(
                "too few edge samples remained after background rejection",
            ));
        }
        if !changed {
            break;
        }
        plane = solve_plane(&points, &active)
            .ok_or_else(|| residual_error("local background plane is singular"))?;
    }
    Ok(plane)
}

fn solve_plane(points: &[PlaneSample], active: &[bool]) -> Option<[f64; 3]> {
    let mut normal = [[0.0_f64; 4]; 3];
    for (point, active) in points.iter().zip(active) {
        if !active {
            continue;
        }
        let row = [1.0, point.x, point.y];
        for i in 0..3 {
            for j in 0..3 {
                normal[i][j] += row[i] * row[j];
            }
            normal[i][3] += row[i] * point.value;
        }
    }
    for pivot in 0..3 {
        let swap = (pivot..3).max_by(|left, right| {
            normal[*left][pivot]
                .abs()
                .total_cmp(&normal[*right][pivot].abs())
        })?;
        normal.swap(pivot, swap);
        if normal[pivot][pivot].abs() <= f64::EPSILON {
            return None;
        }
        let divisor = normal[pivot][pivot];
        for value in &mut normal[pivot][pivot..=3] {
            *value /= divisor;
        }
        for row in 0..3 {
            if row == pivot {
                continue;
            }
            let scale = normal[row][pivot];
            let pivot_values = normal[pivot];
            for (value, pivot_value) in normal[row][pivot..=3]
                .iter_mut()
                .zip(&pivot_values[pivot..=3])
            {
                *value -= scale * *pivot_value;
            }
        }
    }
    Some([normal[0][3], normal[1][3], normal[2][3]])
}

fn evaluate_plane(plane: [f64; 3], x: f64, y: f64) -> f64 {
    plane[0] + plane[1] * x + plane[2] * y
}

fn normalized_coordinate(value: usize, length: usize) -> f64 {
    if length <= 1 {
        0.0
    } else {
        2.0 * value as f64 / (length - 1) as f64 - 1.0
    }
}

fn edge_feather(pixel: usize, width: usize, height: usize, fraction: f32) -> f32 {
    if fraction <= 0.0 {
        return 1.0;
    }
    let x = pixel % width;
    let y = pixel / width;
    let distance = x.min(width - 1 - x).min(y.min(height - 1 - y)) as f32;
    let feather_width = width.min(height) as f32 * fraction;
    (distance / feather_width).clamp(0.0, 1.0)
}

fn residual_error(message: impl Into<String>) -> Error {
    Error::Calibration(format!("residual flat: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_with_shadow(width: usize, height: usize, offset: f32, shadow: bool) -> LinearImage {
        let data = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let background = 1_000.0 + 1.5 * x as f32 + 0.75 * y as f32 + offset;
                    let dx = x as f32 - 15.5;
                    let dy = y as f32 - 15.5;
                    let radius = dx.hypot(dy);
                    let response = if shadow && (6.0..=9.0).contains(&radius) {
                        0.8
                    } else {
                        1.0
                    };
                    let moving_star = if (x + offset as usize) % width == y {
                        2_000.0
                    } else {
                        0.0
                    };
                    background * response + moving_star
                })
            })
            .collect();
        LinearImage::new(width, height, 1, data).unwrap()
    }

    #[test]
    fn builds_and_applies_a_bounded_repeated_shadow() {
        let samples = (0..7)
            .map(|index| sample_with_shadow(32, 32, index as f32 * 3.0, true))
            .collect::<Vec<_>>();
        let options = ResidualFlatOptions {
            smoothing_sigma: 0.8,
            edge_feather_fraction: 0.0,
            ..ResidualFlatOptions::default()
        };
        let built = build_residual_flat_patch(&samples, &options).unwrap();
        assert_eq!(
            built.diagnostics.algorithm_version,
            RESIDUAL_FLAT_ALGORITHM_VERSION
        );
        assert_eq!(built.diagnostics.sample_count, 7);
        assert!(built.diagnostics.corrected_samples > 40);
        assert!(built.diagnostics.largest_connected_pixels > 20);
        assert!(built.diagnostics.minimum_response >= 1.0 / options.maximum_gain);
        assert!(built.diagnostics.maximum_applied_gain <= options.maximum_gain + 1.0e-5);

        let mut light = sample_with_shadow(40, 40, 0.0, false);
        let light_x = 4 + 15;
        let light_y = 4 + 8;
        let before = light.data[light_y * 40 + light_x];
        built.patch.apply_at(&mut light, 4, 4).unwrap();
        assert!(light.data[light_y * 40 + light_x] > before);
    }

    #[test]
    fn rejects_a_shadow_without_cross_frame_consensus() {
        let samples = (0..7)
            .map(|index| sample_with_shadow(32, 32, index as f32 * 3.0, index == 0))
            .collect::<Vec<_>>();
        let result = build_residual_flat_patch(&samples, &ResidualFlatOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn rejects_scattered_repeated_pixels_without_a_coherent_region() {
        let sample = || {
            let data = (0..32)
                .flat_map(|y| {
                    (0..32).map(move |x| {
                        if (8..=24).contains(&x)
                            && (8..=24).contains(&y)
                            && x % 4 == 0
                            && y % 4 == 0
                        {
                            800.0
                        } else {
                            1_000.0
                        }
                    })
                })
                .collect();
            LinearImage::new(32, 32, 1, data).unwrap()
        };
        let samples = (0..7).map(|_| sample()).collect::<Vec<_>>();
        let options = ResidualFlatOptions {
            smoothing_sigma: 0.0,
            edge_feather_fraction: 0.0,
            minimum_connected_pixels: 2,
            ..ResidualFlatOptions::default()
        };
        let error = build_residual_flat_patch(&samples, &options)
            .unwrap_err()
            .to_string();
        assert!(error.contains("connected pixels"), "{error}");
    }

    #[test]
    fn removes_disconnected_corrections_from_an_accepted_patch() {
        let sample = || {
            let mut data = vec![1_000.0_f32; 64 * 64];
            for y in 24..36 {
                for x in 24..36 {
                    data[y * 64 + x] = 800.0;
                }
            }
            data[10 * 64 + 10] = 800.0;
            LinearImage::new(64, 64, 1, data).unwrap()
        };
        let samples = (0..5).map(|_| sample()).collect::<Vec<_>>();
        let options = ResidualFlatOptions {
            smoothing_sigma: 0.0,
            edge_feather_fraction: 0.0,
            minimum_connected_pixels: 64,
            ..ResidualFlatOptions::default()
        };

        let built = build_residual_flat_patch(&samples, &options).unwrap();

        assert_eq!(built.diagnostics.largest_connected_pixels, 12 * 12);
        assert_eq!(built.diagnostics.corrected_samples, 12 * 12);
        assert_eq!(built.patch.response().data[10 * 64 + 10], 1.0);
        assert!(built.patch.response().data[30 * 64 + 30] < 1.0);
    }

    #[test]
    fn legacy_diagnostics_default_to_the_initial_algorithm_version() {
        let diagnostics: ResidualFlatDiagnostics = serde_json::from_value(serde_json::json!({
            "sample_count": 5,
            "corrected_samples": 64,
            "total_samples": 1024,
            "largest_connected_pixels": 64,
            "minimum_response": 0.95,
            "maximum_applied_gain": 1.0526316
        }))
        .unwrap();

        assert_eq!(
            diagnostics.algorithm_version,
            RESIDUAL_FLAT_ALGORITHM_VERSION
        );
    }

    #[test]
    fn default_depth_keeps_a_one_percent_repeated_shadow() {
        let samples = (0..7)
            .map(|index| {
                let offset = index as f32 * 3.0;
                let data = (0..48)
                    .flat_map(|y| {
                        (0..48).map(move |x| {
                            let background = 1_000.0 + x as f32 + 0.5 * y as f32 + offset;
                            let radius = (x as f32 - 23.5).hypot(y as f32 - 23.5);
                            let response = if (8.0..=15.0).contains(&radius) {
                                0.99
                            } else {
                                1.0
                            };
                            background * response
                        })
                    })
                    .collect();
                LinearImage::new(48, 48, 1, data).unwrap()
            })
            .collect::<Vec<_>>();

        let built = build_residual_flat_patch(&samples, &ResidualFlatOptions::default()).unwrap();
        assert!(built.diagnostics.corrected_samples > 100);
        assert!(built.diagnostics.minimum_response < 0.995);
    }

    #[test]
    fn rejects_too_few_samples_and_invalid_cached_response() {
        let samples = vec![sample_with_shadow(32, 32, 0.0, true); 2];
        assert!(build_residual_flat_patch(&samples, &ResidualFlatOptions::default()).is_err());
        let invalid = LinearImage::new(2, 2, 1, vec![1.0, 0.0, 1.1, f32::NAN]).unwrap();
        assert!(ResidualFlatPatch::from_response(invalid).is_err());
    }

    #[test]
    fn validates_patch_bounds_and_channels_before_mutating() {
        let patch =
            ResidualFlatPatch::from_response(LinearImage::new(2, 2, 1, vec![0.9; 4]).unwrap())
                .unwrap();
        let mut rgb = LinearImage::new(4, 4, 3, vec![1.0; 48]).unwrap();
        let original = rgb.clone();
        assert!(patch.apply_at(&mut rgb, 0, 0).is_err());
        assert_eq!(rgb, original);

        let mut mono = LinearImage::new(4, 4, 1, vec![1.0; 16]).unwrap();
        let original = mono.clone();
        assert!(patch.apply_at(&mut mono, 3, 3).is_err());
        assert_eq!(mono, original);
    }
}
