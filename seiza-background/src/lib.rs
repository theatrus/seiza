//! Robust background and gradient modelling for linear astrophotography images.

use rayon::prelude::*;
use seiza_stats::{median_f64 as median, median_in_place, robust_sigma_in_place};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Parameters for estimating a smooth background from linear image samples.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BackgroundConfig {
    /// Smooth surface family fitted to accepted background samples.
    pub model: ModelConfig,
    /// Number of deterministic seed positions along the image's longest axis.
    pub samples_per_axis: usize,
    /// Radius of each square sample window. `None` chooses from image size.
    pub sample_radius: Option<usize>,
    /// Maximum number of local low-background moves made by each seed.
    pub search_steps: usize,
    /// Robust sigma threshold for rejecting locally noisy sample windows.
    pub sample_rejection_sigma: f64,
    /// Robust sigma threshold for rejecting samples inconsistent with the fit.
    pub fit_rejection_sigma: f64,
    /// Maximum robust refit/rejection passes.
    pub fit_rejection_iterations: usize,
    /// Fractional border excluded from automatic sampling, in `[0, 0.45)`.
    pub border_fraction: f64,
}

impl Default for BackgroundConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            samples_per_axis: 12,
            sample_radius: None,
            search_steps: 4,
            sample_rejection_sigma: 3.5,
            fit_rejection_sigma: 3.0,
            fit_rejection_iterations: 3,
            border_fraction: 0.03,
        }
    }
}

/// Surface families available to the background estimator.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ModelConfig {
    /// A total-degree polynomial in normalized image coordinates.
    Polynomial {
        /// Total polynomial degree. Zero is a constant pedestal; four is the
        /// highest supported degree.
        degree: u8,
        /// Scale-independent Tikhonov regularization applied to non-constant
        /// coefficients after coordinate normalization.
        ridge: f64,
    },
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::Polynomial {
            degree: 2,
            ridge: 1.0e-8,
        }
    }
}

/// How a fitted background is removed from the input.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionMode {
    /// Remove an additive gradient while retaining the robust background level.
    #[default]
    Subtract,
    /// Correct a multiplicative field response while retaining image scale.
    Divide,
}

/// Why a candidate sample was retained or rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SampleStatus {
    Accepted,
    RejectedNoise,
    RejectedResidual,
}

/// One measured background window, useful for diagnostics and overlays.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BackgroundSample {
    /// Zero-indexed sample center in the input image.
    pub x: usize,
    pub y: usize,
    /// Per-channel robust median of the sample window.
    pub values: Vec<f32>,
    /// Mean normal-equivalent per-channel MAD in the sample window.
    pub dispersion: f32,
    /// Weight used by the final least-squares fit.
    pub weight: f32,
    pub status: SampleStatus,
}

/// Counts and resolved parameters from background fitting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FitDiagnostics {
    pub candidate_samples: usize,
    pub accepted_samples: usize,
    pub rejected_noise: usize,
    pub rejected_residual: usize,
    pub rejection_iterations: usize,
    pub sample_radius: usize,
}

/// A compact surface fitted independently for each channel.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum FittedModel {
    Polynomial {
        degree: u8,
        /// One coefficient vector per channel. Terms are ordered by increasing
        /// total degree, then decreasing x exponent within each degree.
        coefficients: Vec<Vec<f64>>,
    },
}

/// A fitted background, small enough to retain without a full image-sized map.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BackgroundFit {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub model: FittedModel,
    /// Robust per-channel background level retained by correction operations.
    pub reference: Vec<f64>,
    pub samples: Vec<BackgroundSample>,
    pub diagnostics: FitDiagnostics,
}

#[derive(Debug, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum Error {
    #[error("invalid image: {0}")]
    InvalidImage(String),
    #[error("invalid background configuration: {0}")]
    InvalidConfig(String),
    #[error("invalid fitted background: {0}")]
    InvalidFit(String),
    #[error("not enough usable background samples: found {found}, need at least {required}")]
    NotEnoughSamples { found: usize, required: usize },
    #[error("background surface fit is singular")]
    SingularFit,
    #[error("multiplicative background reference is zero or non-finite for channel {channel}")]
    InvalidReference { channel: usize },
    #[error("multiplicative background is unsafe at ({x}, {y}), channel {channel}")]
    InvalidDivisor { x: usize, y: usize, channel: usize },
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug)]
struct RawSample {
    x: usize,
    y: usize,
    values: Vec<f64>,
    dispersion: f64,
    weight: f64,
    status: SampleStatus,
}

/// Fit a background without an exclusion mask.
pub fn fit_background(
    data: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    config: &BackgroundConfig,
) -> Result<BackgroundFit> {
    fit_background_masked(data, width, height, channels, None, config)
}

/// Fit a background while excluding pixels whose mask entry is `true`.
///
/// The mask has one entry per pixel, independent of the channel count. This is
/// the extension point for user regions, source masks, and future learned
/// structure masks.
pub fn fit_background_masked(
    data: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    exclusion_mask: Option<&[bool]>,
    config: &BackgroundConfig,
) -> Result<BackgroundFit> {
    validate_image(data, width, height, channels, exclusion_mask)?;
    let (degree, ridge) = validate_config(config)?;
    let radius = resolved_radius(config.sample_radius, width, height);
    let mut samples = collect_samples(
        data,
        width,
        height,
        channels,
        exclusion_mask,
        config,
        radius,
    );
    let required = basis_len(degree)
        .saturating_mul(2)
        .max(basis_len(degree) + 2);
    if samples.len() < required {
        return Err(Error::NotEnoughSamples {
            found: samples.len(),
            required,
        });
    }

    let dispersions: Vec<f64> = samples.iter().map(|sample| sample.dispersion).collect();
    let dispersion_median = median(&dispersions).unwrap_or(0.0);
    let dispersion_sigma =
        seiza_stats::robust_sigma_f64(&dispersions, dispersion_median).unwrap_or(0.0);
    let noise_limit = dispersion_median + config.sample_rejection_sigma * dispersion_sigma;
    for sample in &mut samples {
        if dispersion_sigma > 0.0 && sample.dispersion > noise_limit {
            sample.status = SampleStatus::RejectedNoise;
        }
    }
    if accepted_count(&samples) < required {
        // A small or unusually structured frame can make dispersion rejection
        // too aggressive. Keep only the quietest candidates needed for a
        // valid fit instead of admitting every noisy window.
        samples.sort_by(|a, b| a.dispersion.total_cmp(&b.dispersion));
        for (index, sample) in samples.iter_mut().enumerate() {
            sample.status = if index < required {
                SampleStatus::Accepted
            } else {
                SampleStatus::RejectedNoise
            };
        }
    }

    let weight_scale = dispersion_median.max(1.0e-12);
    for sample in &mut samples {
        let relative = sample.dispersion / weight_scale;
        sample.weight = (1.0 / (1.0 + relative * relative)).clamp(0.05, 1.0);
    }

    let mut coefficients = fit_channels(&samples, width, height, channels, degree, ridge)?;
    let mut rejection_iterations = 0;
    for _ in 0..config.fit_rejection_iterations {
        let residuals: Vec<(usize, Vec<f64>)> = samples
            .iter()
            .enumerate()
            .filter(|(_, sample)| sample.status == SampleStatus::Accepted)
            .map(|(index, sample)| {
                let residuals = (0..channels)
                    .map(|channel| {
                        sample.values[channel]
                            - evaluate_coefficients(
                                &coefficients[channel],
                                degree,
                                normalized_coordinate(sample.x, width),
                                normalized_coordinate(sample.y, height),
                            )
                    })
                    .collect();
                (index, residuals)
            })
            .collect();
        let channel_limits: Vec<(f64, f64)> = (0..channels)
            .map(|channel| {
                let values: Vec<f64> = residuals
                    .iter()
                    .map(|(_, residuals)| residuals[channel])
                    .collect();
                let center = median(&values).unwrap_or(0.0);
                (
                    center,
                    seiza_stats::robust_sigma_f64(&values, center).unwrap_or(0.0),
                )
            })
            .collect();
        if channel_limits.iter().all(|(_, sigma)| *sigma <= 1.0e-12) {
            break;
        }
        let rejected: Vec<usize> = residuals
            .iter()
            .filter(|(_, residuals)| {
                residuals
                    .iter()
                    .zip(&channel_limits)
                    .any(|(residual, (center, sigma))| {
                        *sigma > 1.0e-12
                            && (*residual - *center).abs() > config.fit_rejection_sigma * *sigma
                    })
            })
            .map(|(index, _)| *index)
            .collect();
        if rejected.is_empty() || accepted_count(&samples).saturating_sub(rejected.len()) < required
        {
            break;
        }
        for index in rejected {
            samples[index].status = SampleStatus::RejectedResidual;
        }
        coefficients = fit_channels(&samples, width, height, channels, degree, ridge)?;
        rejection_iterations += 1;
    }

    let reference = (0..channels)
        .map(|channel| {
            let values: Vec<f64> = samples
                .iter()
                .filter(|sample| sample.status == SampleStatus::Accepted)
                .map(|sample| sample.values[channel])
                .collect();
            median(&values).ok_or(Error::NotEnoughSamples {
                found: 0,
                required: 1,
            })
        })
        .collect::<Result<_>>()?;
    let rejected_noise = samples
        .iter()
        .filter(|sample| sample.status == SampleStatus::RejectedNoise)
        .count();
    let rejected_residual = samples
        .iter()
        .filter(|sample| sample.status == SampleStatus::RejectedResidual)
        .count();
    let accepted_samples = accepted_count(&samples);
    let candidate_samples = samples.len();
    let samples = samples
        .into_iter()
        .map(|sample| BackgroundSample {
            x: sample.x,
            y: sample.y,
            values: sample
                .values
                .into_iter()
                .map(|value| value as f32)
                .collect(),
            dispersion: sample.dispersion as f32,
            weight: sample.weight as f32,
            status: sample.status,
        })
        .collect();

    Ok(BackgroundFit {
        width,
        height,
        channels,
        model: FittedModel::Polynomial {
            degree,
            coefficients,
        },
        reference,
        samples,
        diagnostics: FitDiagnostics {
            candidate_samples,
            accepted_samples,
            rejected_noise,
            rejected_residual,
            rejection_iterations,
            sample_radius: radius,
        },
    })
}

impl BackgroundFit {
    /// Validate dimensions, reference levels, and fitted surface coefficients.
    ///
    /// Fits produced by [`fit_background`] are already valid. This is useful
    /// when accepting a deserialized or externally constructed fit.
    pub fn validate(&self) -> Result<()> {
        self.validated_sample_count()?;
        if self.reference.len() != self.channels {
            return Err(Error::InvalidFit(format!(
                "reference has {} channels; expected {}",
                self.reference.len(),
                self.channels
            )));
        }
        if self.reference.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidFit(
                "reference levels must all be finite".into(),
            ));
        }
        match &self.model {
            FittedModel::Polynomial {
                degree,
                coefficients,
            } => {
                if *degree > 4 {
                    return Err(Error::InvalidFit(
                        "polynomial degree must be between 0 and 4".into(),
                    ));
                }
                if coefficients.len() != self.channels {
                    return Err(Error::InvalidFit(format!(
                        "polynomial has {} channel coefficient sets; expected {}",
                        coefficients.len(),
                        self.channels
                    )));
                }
                let expected = basis_len(*degree);
                if let Some((channel, actual)) =
                    coefficients
                        .iter()
                        .enumerate()
                        .find_map(|(channel, values)| {
                            (values.len() != expected).then_some((channel, values.len()))
                        })
                {
                    return Err(Error::InvalidFit(format!(
                        "polynomial channel {channel} has {actual} coefficients; expected {expected}"
                    )));
                }
                if coefficients
                    .iter()
                    .flatten()
                    .any(|coefficient| !coefficient.is_finite())
                {
                    return Err(Error::InvalidFit(
                        "polynomial coefficients must all be finite".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Evaluate one channel of the fitted background at a pixel coordinate.
    pub fn value_at(&self, x: usize, y: usize, channel: usize) -> Result<f64> {
        self.validate()?;
        if x >= self.width || y >= self.height || channel >= self.channels {
            return Err(Error::InvalidImage(
                "background evaluation coordinate is outside the fitted image".into(),
            ));
        }
        let x = normalized_coordinate(x, self.width);
        let y = normalized_coordinate(y, self.height);
        match &self.model {
            FittedModel::Polynomial {
                degree,
                coefficients,
            } => Ok(evaluate_coefficients(&coefficients[channel], *degree, x, y)),
        }
    }

    /// Render the fitted background as interleaved `f32` samples.
    pub fn render_model(&self) -> Result<Vec<f32>> {
        self.validate()?;
        let mut output = vec![0.0; self.validated_sample_count()?];
        self.render_model_into_validated(&mut output);
        Ok(output)
    }

    /// Render the fitted background into a caller-provided interleaved buffer.
    pub fn render_model_into(&self, output: &mut [f32]) -> Result<()> {
        self.validate()?;
        self.validate_buffer_len(output.len())?;
        self.render_model_into_validated(output);
        Ok(())
    }

    fn render_model_into_validated(&self, output: &mut [f32]) {
        output
            .par_chunks_mut(self.width * self.channels)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..self.width {
                    for channel in 0..self.channels {
                        row[x * self.channels + channel] =
                            self.value_unchecked(x, y, channel) as f32;
                    }
                }
            });
    }

    /// Return a corrected copy of an interleaved image.
    pub fn correct(&self, data: &[f32], mode: CorrectionMode) -> Result<Vec<f32>> {
        self.validate()?;
        self.validate_buffer_len(data.len())?;
        let mut corrected = data.to_vec();
        self.correct_in_place_validated(&mut corrected, mode)?;
        Ok(corrected)
    }

    /// Correct an interleaved image in place without allocating a model image.
    pub fn correct_in_place(&self, data: &mut [f32], mode: CorrectionMode) -> Result<()> {
        self.validate()?;
        self.validate_buffer_len(data.len())?;
        self.correct_in_place_validated(data, mode)
    }

    fn correct_in_place_validated(&self, data: &mut [f32], mode: CorrectionMode) -> Result<()> {
        if mode == CorrectionMode::Divide {
            for (channel, reference) in self.reference.iter().copied().enumerate() {
                if !reference.is_finite() || reference.abs() <= 1.0e-12 {
                    return Err(Error::InvalidReference { channel });
                }
            }
            (0..self.width * self.height)
                .into_par_iter()
                .try_for_each(|pixel| -> Result<()> {
                    let x = pixel % self.width;
                    let y = pixel / self.width;
                    for channel in 0..self.channels {
                        let background = self.value_unchecked(x, y, channel);
                        let reference = self.reference[channel];
                        let floor = reference.abs().mul_add(1.0e-9, 1.0e-12);
                        if !background.is_finite()
                            || background.abs() <= floor
                            || background.is_sign_positive() != reference.is_sign_positive()
                        {
                            return Err(Error::InvalidDivisor { x, y, channel });
                        }
                    }
                    Ok(())
                })?;
        }
        data.par_chunks_mut(self.width * self.channels)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..self.width {
                    for channel in 0..self.channels {
                        let value = &mut row[x * self.channels + channel];
                        if !value.is_finite() {
                            continue;
                        }
                        let background = self.value_unchecked(x, y, channel);
                        let reference = self.reference[channel];
                        *value = match mode {
                            CorrectionMode::Subtract => {
                                (f64::from(*value) - background + reference) as f32
                            }
                            CorrectionMode::Divide => {
                                (f64::from(*value) / background * reference) as f32
                            }
                        };
                    }
                }
            });
        Ok(())
    }

    fn validated_sample_count(&self) -> Result<usize> {
        if self.width == 0 || self.height == 0 || self.channels == 0 {
            return Err(Error::InvalidFit(
                "dimensions and channel count must be non-zero".into(),
            ));
        }
        self.width
            .checked_mul(self.height)
            .and_then(|pixels| pixels.checked_mul(self.channels))
            .ok_or_else(|| Error::InvalidFit("image dimensions overflow".into()))
    }

    fn validate_buffer_len(&self, actual: usize) -> Result<()> {
        let expected = self
            .width
            .checked_mul(self.height)
            .and_then(|pixels| pixels.checked_mul(self.channels))
            .ok_or_else(|| Error::InvalidImage("fitted image dimensions overflow".into()))?;
        if actual != expected {
            return Err(Error::InvalidImage(format!(
                "pixel buffer has {actual} samples; expected {expected}"
            )));
        }
        Ok(())
    }

    fn value_unchecked(&self, x: usize, y: usize, channel: usize) -> f64 {
        let x = normalized_coordinate(x, self.width);
        let y = normalized_coordinate(y, self.height);
        match &self.model {
            FittedModel::Polynomial {
                degree,
                coefficients,
            } => evaluate_coefficients(&coefficients[channel], *degree, x, y),
        }
    }
}

fn validate_image(
    data: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    mask: Option<&[bool]>,
) -> Result<()> {
    if width == 0 || height == 0 || channels == 0 {
        return Err(Error::InvalidImage(
            "dimensions and channel count must be non-zero".into(),
        ));
    }
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| Error::InvalidImage("image dimensions overflow".into()))?;
    let expected = pixels
        .checked_mul(channels)
        .ok_or_else(|| Error::InvalidImage("image dimensions overflow".into()))?;
    if data.len() != expected {
        return Err(Error::InvalidImage(format!(
            "pixel buffer has {} samples; expected {expected}",
            data.len()
        )));
    }
    if let Some(mask) = mask
        && mask.len() != pixels
    {
        return Err(Error::InvalidImage(format!(
            "exclusion mask has {} entries; expected {pixels}",
            mask.len()
        )));
    }
    Ok(())
}

fn validate_config(config: &BackgroundConfig) -> Result<(u8, f64)> {
    if config.samples_per_axis < 3 {
        return Err(Error::InvalidConfig(
            "samples_per_axis must be at least 3".into(),
        ));
    }
    if config.samples_per_axis > 512 {
        return Err(Error::InvalidConfig(
            "samples_per_axis must not exceed 512".into(),
        ));
    }
    if let Some(radius) = config.sample_radius
        && radius == 0
    {
        return Err(Error::InvalidConfig(
            "sample_radius must be greater than zero".into(),
        ));
    }
    if config.search_steps > 64 {
        return Err(Error::InvalidConfig(
            "search_steps must not exceed 64".into(),
        ));
    }
    if config.fit_rejection_iterations > 16 {
        return Err(Error::InvalidConfig(
            "fit_rejection_iterations must not exceed 16".into(),
        ));
    }
    for (name, value) in [
        ("sample_rejection_sigma", config.sample_rejection_sigma),
        ("fit_rejection_sigma", config.fit_rejection_sigma),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::InvalidConfig(format!(
                "{name} must be finite and greater than zero"
            )));
        }
    }
    if !config.border_fraction.is_finite()
        || config.border_fraction < 0.0
        || config.border_fraction >= 0.45
    {
        return Err(Error::InvalidConfig(
            "border_fraction must be finite and in [0, 0.45)".into(),
        ));
    }
    match config.model {
        ModelConfig::Polynomial { degree, ridge } => {
            if degree > 4 {
                return Err(Error::InvalidConfig(
                    "polynomial degree must be between 0 and 4".into(),
                ));
            }
            if !ridge.is_finite() || ridge < 0.0 {
                return Err(Error::InvalidConfig(
                    "polynomial ridge must be finite and non-negative".into(),
                ));
            }
            Ok((degree, ridge))
        }
    }
}

fn resolved_radius(requested: Option<usize>, width: usize, height: usize) -> usize {
    let max_radius = width.min(height).saturating_sub(1) / 4;
    requested
        .unwrap_or_else(|| ((height as f64 * 0.025).round() as usize).clamp(3, 25))
        .min(max_radius.max(1))
}

#[allow(clippy::too_many_arguments)]
fn collect_samples(
    data: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    mask: Option<&[bool]>,
    config: &BackgroundConfig,
    radius: usize,
) -> Vec<RawSample> {
    let border_x = ((width as f64 * config.border_fraction).round() as usize).max(radius);
    let border_y = ((height as f64 * config.border_fraction).round() as usize).max(radius);
    let min_x = border_x.min(width.saturating_sub(radius + 1));
    let max_x = width.saturating_sub(border_x + 1).max(min_x);
    let min_y = border_y.min(height.saturating_sub(radius + 1));
    let max_y = height.saturating_sub(border_y + 1).max(min_y);
    let longest = width.max(height) as f64;
    let x_count =
        ((config.samples_per_axis as f64 * width as f64 / longest).round() as usize).max(3);
    let y_count =
        ((config.samples_per_axis as f64 * height as f64 / longest).round() as usize).max(3);
    let step = radius.max(1);
    let mut positions = BTreeSet::new();
    let mut samples = Vec::with_capacity(x_count * y_count);
    for yi in 0..y_count {
        for xi in 0..x_count {
            let seed_x = grid_position(xi, x_count, min_x, max_x);
            let seed_y = grid_position(yi, y_count, min_y, max_y);
            if let Some(sample) = descend_sample(
                data,
                width,
                height,
                channels,
                mask,
                seed_x,
                seed_y,
                radius,
                step,
                config.search_steps,
                min_x,
                max_x,
                min_y,
                max_y,
            ) && positions.insert((sample.x, sample.y))
            {
                samples.push(sample);
            }
        }
    }
    samples
}

fn grid_position(index: usize, count: usize, min: usize, max: usize) -> usize {
    if count <= 1 || min >= max {
        return min;
    }
    min + ((max - min) as f64 * index as f64 / (count - 1) as f64).round() as usize
}

#[allow(clippy::too_many_arguments)]
fn descend_sample(
    data: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    mask: Option<&[bool]>,
    seed_x: usize,
    seed_y: usize,
    radius: usize,
    step: usize,
    search_steps: usize,
    min_x: usize,
    max_x: usize,
    min_y: usize,
    max_y: usize,
) -> Option<RawSample> {
    let mut best = window_statistics(data, width, height, channels, mask, seed_x, seed_y, radius)?;
    let origin = (seed_x, seed_y);
    for _ in 0..search_steps {
        let mut next = best.clone();
        for dy in [-1_isize, 0, 1] {
            for dx in [-1_isize, 0, 1] {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let x = best
                    .x
                    .saturating_add_signed(dx * step as isize)
                    .clamp(min_x, max_x);
                let y = best
                    .y
                    .saturating_add_signed(dy * step as isize)
                    .clamp(min_y, max_y);
                if x.abs_diff(origin.0) > search_steps * step
                    || y.abs_diff(origin.1) > search_steps * step
                {
                    continue;
                }
                if let Some(candidate) =
                    window_statistics(data, width, height, channels, mask, x, y, radius)
                    && sample_score(&candidate) < sample_score(&next)
                {
                    next = candidate;
                }
            }
        }
        if next.x == best.x && next.y == best.y {
            break;
        }
        best = next;
    }
    Some(best)
}

fn sample_score(sample: &RawSample) -> f64 {
    sample.values.iter().sum::<f64>() / sample.values.len() as f64 + 0.25 * sample.dispersion
}

#[allow(clippy::too_many_arguments)]
fn window_statistics(
    data: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    mask: Option<&[bool]>,
    x: usize,
    y: usize,
    radius: usize,
) -> Option<RawSample> {
    let x0 = x.saturating_sub(radius);
    let x1 = (x + radius).min(width - 1);
    let y0 = y.saturating_sub(radius);
    let y1 = (y + radius).min(height - 1);
    let mut values = vec![Vec::new(); channels];
    for py in y0..=y1 {
        for px in x0..=x1 {
            let pixel = py * width + px;
            if mask.is_some_and(|mask| mask[pixel]) {
                continue;
            }
            let start = pixel * channels;
            if data[start..start + channels]
                .iter()
                .any(|value| !value.is_finite())
            {
                continue;
            }
            for channel in 0..channels {
                values[channel].push(data[start + channel]);
            }
        }
    }
    let area = (x1 - x0 + 1) * (y1 - y0 + 1);
    if values[0].len() < area.div_ceil(4).max(9) {
        return None;
    }
    let mut medians = Vec::with_capacity(channels);
    let mut dispersions = Vec::with_capacity(channels);
    for channel_values in &mut values {
        let channel_median = median_in_place(channel_values)?;
        let dispersion = robust_sigma_in_place(channel_values, channel_median)?;
        medians.push(f64::from(channel_median));
        dispersions.push(f64::from(dispersion));
    }
    Some(RawSample {
        x,
        y,
        values: medians,
        dispersion: dispersions.iter().sum::<f64>() / channels as f64,
        weight: 1.0,
        status: SampleStatus::Accepted,
    })
}

fn accepted_count(samples: &[RawSample]) -> usize {
    samples
        .iter()
        .filter(|sample| sample.status == SampleStatus::Accepted)
        .count()
}

fn fit_channels(
    samples: &[RawSample],
    width: usize,
    height: usize,
    channels: usize,
    degree: u8,
    ridge: f64,
) -> Result<Vec<Vec<f64>>> {
    (0..channels)
        .map(|channel| fit_channel(samples, width, height, channel, degree, ridge))
        .collect()
}

fn fit_channel(
    samples: &[RawSample],
    width: usize,
    height: usize,
    channel: usize,
    degree: u8,
    ridge: f64,
) -> Result<Vec<f64>> {
    let count = basis_len(degree);
    let mut normal = vec![vec![0.0; count]; count];
    let mut rhs = vec![0.0; count];
    for sample in samples
        .iter()
        .filter(|sample| sample.status == SampleStatus::Accepted)
    {
        let basis = polynomial_basis(
            degree,
            normalized_coordinate(sample.x, width),
            normalized_coordinate(sample.y, height),
        );
        for row in 0..count {
            rhs[row] += sample.weight * basis[row] * sample.values[channel];
            for column in 0..count {
                normal[row][column] += sample.weight * basis[row] * basis[column];
            }
        }
    }
    let scale = (0..count).map(|index| normal[index][index]).sum::<f64>() / count as f64;
    for (index, row) in normal.iter_mut().enumerate().skip(1) {
        row[index] += ridge * scale.max(1.0);
    }
    solve_linear_system(normal, rhs)
}

fn solve_linear_system(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Result<Vec<f64>> {
    let size = rhs.len();
    for pivot in 0..size {
        let best = (pivot..size)
            .max_by(|&a, &b| matrix[a][pivot].abs().total_cmp(&matrix[b][pivot].abs()))
            .expect("non-empty pivot range");
        if matrix[best][pivot].abs() <= 1.0e-14 {
            return Err(Error::SingularFit);
        }
        matrix.swap(pivot, best);
        rhs.swap(pivot, best);
        let divisor = matrix[pivot][pivot];
        for value in &mut matrix[pivot][pivot..] {
            *value /= divisor;
        }
        rhs[pivot] /= divisor;
        let pivot_row = matrix[pivot].clone();
        for row in 0..size {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            if factor == 0.0 {
                continue;
            }
            for (value, pivot_value) in matrix[row][pivot..].iter_mut().zip(&pivot_row[pivot..]) {
                *value -= factor * pivot_value;
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    Ok(rhs)
}

fn basis_len(degree: u8) -> usize {
    let degree = usize::from(degree);
    (degree + 1) * (degree + 2) / 2
}

fn polynomial_basis(degree: u8, x: f64, y: f64) -> Vec<f64> {
    let mut basis = Vec::with_capacity(basis_len(degree));
    for total in 0..=u32::from(degree) {
        for x_power in (0..=total).rev() {
            let y_power = total - x_power;
            basis.push(x.powi(x_power as i32) * y.powi(y_power as i32));
        }
    }
    basis
}

fn evaluate_coefficients(coefficients: &[f64], degree: u8, x: f64, y: f64) -> f64 {
    let mut result = 0.0;
    let mut index = 0;
    for total in 0..=u32::from(degree) {
        for x_power in (0..=total).rev() {
            let y_power = total - x_power;
            result += coefficients[index] * x.powi(x_power as i32) * y.powi(y_power as i32);
            index += 1;
        }
    }
    result
}

fn normalized_coordinate(value: usize, extent: usize) -> f64 {
    if extent <= 1 {
        0.0
    } else {
        2.0 * value as f64 / (extent - 1) as f64 - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(width: usize, height: usize, channels: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(width * height * channels);
        for y in 0..height {
            let ny = normalized_coordinate(y, height) as f32;
            for x in 0..width {
                let nx = normalized_coordinate(x, width) as f32;
                for channel in 0..channels {
                    let channel = channel as f32;
                    data.push(
                        0.2 + 0.04 * channel + (0.08 - 0.015 * channel) * nx
                            - (0.05 - 0.01 * channel) * ny,
                    );
                }
            }
        }
        data
    }

    #[test]
    fn recovers_and_subtracts_a_color_plane_with_bright_sources() {
        let (width, height, channels) = (128, 96, 3);
        let expected = plane(width, height, channels);
        let mut image = expected.clone();
        for &(cx, cy) in &[(20_usize, 18_usize), (63, 51), (104, 72), (88, 20)] {
            for y in cy - 2..=cy + 2 {
                for x in cx - 2..=cx + 2 {
                    let distance = x.abs_diff(cx) + y.abs_diff(cy);
                    let signal = 0.7 / (distance + 1) as f32;
                    for channel in 0..channels {
                        image[(y * width + x) * channels + channel] += signal;
                    }
                }
            }
        }
        let config = BackgroundConfig {
            model: ModelConfig::Polynomial {
                degree: 1,
                ridge: 0.0,
            },
            samples_per_axis: 10,
            sample_radius: Some(3),
            ..BackgroundConfig::default()
        };
        let fit = fit_background(&image, width, height, channels, &config).unwrap();
        let model = fit.render_model().unwrap();
        let mse = model
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| f64::from(*actual - *expected).powi(2))
            .sum::<f64>()
            / model.len() as f64;
        let rmse = mse.sqrt();
        assert!(rmse < 0.003, "model RMSE was {rmse}");
        let corrected = fit.correct(&image, CorrectionMode::Subtract).unwrap();
        for channel in 0..channels {
            let left = corrected[(height / 2 * width + 5) * channels + channel];
            let right = corrected[(height / 2 * width + width - 6) * channels + channel];
            assert!((left - right).abs() < 0.004);
        }
        assert!(fit.diagnostics.accepted_samples >= 20);
    }

    #[test]
    fn divide_removes_a_multiplicative_plane_and_preserves_nonfinite_pixels() {
        let (width, height) = (96, 72);
        let background = plane(width, height, 1);
        let mut image: Vec<f32> = background.iter().map(|value| value * 2.5).collect();
        image[0] = f32::NAN;
        let config = BackgroundConfig {
            model: ModelConfig::Polynomial {
                degree: 1,
                ridge: 0.0,
            },
            samples_per_axis: 8,
            sample_radius: Some(2),
            border_fraction: 0.05,
            ..BackgroundConfig::default()
        };
        let fit = fit_background(&image, width, height, 1, &config).unwrap();
        fit.correct_in_place(&mut image, CorrectionMode::Divide)
            .unwrap();
        assert!(image[0].is_nan());
        let low = image[width * (height / 2) + 3];
        let high = image[width * (height / 2) + width - 4];
        assert!((low - high).abs() < 0.004);
    }

    #[test]
    fn invalid_divisor_does_not_partially_mutate_an_in_place_image() {
        let fit = BackgroundFit {
            width: 2,
            height: 1,
            channels: 1,
            model: FittedModel::Polynomial {
                degree: 1,
                coefficients: vec![vec![0.5, -0.5, 0.0]],
            },
            reference: vec![1.0],
            samples: Vec::new(),
            diagnostics: FitDiagnostics {
                candidate_samples: 0,
                accepted_samples: 0,
                rejected_noise: 0,
                rejected_residual: 0,
                rejection_iterations: 0,
                sample_radius: 1,
            },
        };
        let mut image = vec![2.0, 2.0];
        let original = image.clone();
        assert_eq!(
            fit.correct_in_place(&mut image, CorrectionMode::Divide),
            Err(Error::InvalidDivisor {
                x: 1,
                y: 0,
                channel: 0
            })
        );
        assert_eq!(image, original);
    }

    #[test]
    fn invalid_division_reference_does_not_mutate_an_in_place_image() {
        let fit = BackgroundFit {
            width: 2,
            height: 1,
            channels: 1,
            model: FittedModel::Polynomial {
                degree: 0,
                coefficients: vec![vec![1.0]],
            },
            reference: vec![0.0],
            samples: Vec::new(),
            diagnostics: FitDiagnostics {
                candidate_samples: 0,
                accepted_samples: 0,
                rejected_noise: 0,
                rejected_residual: 0,
                rejection_iterations: 0,
                sample_radius: 1,
            },
        };
        let mut image = vec![2.0, 2.0];
        let original = image.clone();
        assert_eq!(
            fit.correct_in_place(&mut image, CorrectionMode::Divide),
            Err(Error::InvalidReference { channel: 0 })
        );
        assert_eq!(image, original);
    }

    #[test]
    fn correction_rejects_a_mismatched_buffer() {
        let image = plane(64, 48, 1);
        let fit = fit_background(&image, 64, 48, 1, &BackgroundConfig::default()).unwrap();
        assert!(matches!(
            fit.correct(&image[..image.len() - 1], CorrectionMode::Subtract),
            Err(Error::InvalidImage(message)) if message.contains("expected 3072")
        ));
    }

    #[test]
    fn rendering_rejects_a_mismatched_buffer() {
        let image = plane(64, 48, 1);
        let fit = fit_background(&image, 64, 48, 1, &BackgroundConfig::default()).unwrap();
        let mut output = vec![0.0; image.len() - 1];
        assert!(matches!(
            fit.render_model_into(&mut output),
            Err(Error::InvalidImage(message)) if message.contains("expected 3072")
        ));
    }

    #[test]
    fn malformed_deserialized_fit_fails_instead_of_panicking() {
        let fit = BackgroundFit {
            width: 2,
            height: 2,
            channels: 1,
            model: FittedModel::Polynomial {
                degree: 2,
                coefficients: vec![vec![1.0]],
            },
            reference: vec![1.0],
            samples: Vec::new(),
            diagnostics: FitDiagnostics {
                candidate_samples: 0,
                accepted_samples: 0,
                rejected_noise: 0,
                rejected_residual: 0,
                rejection_iterations: 0,
                sample_radius: 1,
            },
        };
        assert!(matches!(fit.render_model(), Err(Error::InvalidFit(_))));
        assert!(matches!(
            fit.correct(&[1.0; 4], CorrectionMode::Subtract),
            Err(Error::InvalidFit(_))
        ));
    }

    #[test]
    fn exclusion_mask_can_remove_a_large_structure_from_sampling() {
        let (width, height) = (80, 80);
        let mut image = plane(width, height, 1);
        let mut mask = vec![false; width * height];
        for y in 20..60 {
            for x in 20..60 {
                image[y * width + x] += 0.5;
                mask[y * width + x] = true;
            }
        }
        let config = BackgroundConfig {
            model: ModelConfig::Polynomial {
                degree: 1,
                ridge: 0.0,
            },
            samples_per_axis: 9,
            sample_radius: Some(2),
            ..BackgroundConfig::default()
        };
        let fit = fit_background_masked(&image, width, height, 1, Some(&mask), &config).unwrap();
        assert!(
            fit.samples
                .iter()
                .all(|sample| !mask[sample.y * width + sample.x])
        );
        assert!(
            (fit.value_at(40, 40, 0).unwrap()
                - f64::from(plane(width, height, 1)[40 * width + 40]))
            .abs()
                < 0.005
        );
    }

    #[test]
    fn configuration_and_fit_round_trip_through_json() {
        let config = BackgroundConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        assert_eq!(
            serde_json::from_str::<BackgroundConfig>(&json).unwrap(),
            config
        );
        let image = plane(64, 48, 1);
        let fit = fit_background(&image, 64, 48, 1, &config).unwrap();
        let json = serde_json::to_string(&fit).unwrap();
        let decoded = serde_json::from_str::<BackgroundFit>(&json).unwrap();
        assert_eq!(decoded.width, fit.width);
        assert_eq!(decoded.height, fit.height);
        assert_eq!(decoded.channels, fit.channels);
        assert_eq!(decoded.samples, fit.samples);
        assert_eq!(decoded.diagnostics, fit.diagnostics);
        assert!((decoded.reference[0] - fit.reference[0]).abs() < 1.0e-15);
        let (
            FittedModel::Polynomial { coefficients, .. },
            FittedModel::Polynomial {
                coefficients: expected,
                ..
            },
        ) = (&decoded.model, &fit.model);
        assert!(
            coefficients[0]
                .iter()
                .zip(&expected[0])
                .all(|(actual, expected)| (*actual - *expected).abs() < 1.0e-15)
        );
    }
}
