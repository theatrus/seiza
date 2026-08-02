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
    /// Normalized target or structure bounds excluded from sampling. Solved
    /// image/catalog projections can populate these without a full-size mask.
    pub protected_regions: Vec<ProtectedRegion>,
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
            protected_regions: Vec::new(),
        }
    }
}

/// Image-relative bounds that protect known targets or extended structures.
/// Coordinates and radii are fractions of image width and height.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ProtectedRegion {
    /// A rotated ellipse projected into the image by a solver or catalog.
    Ellipse {
        center: [f64; 2],
        radii: [f64; 2],
        #[serde(default)]
        rotation_degrees: f64,
    },
    /// One projected closed catalog outline. Use more than one region for
    /// disconnected contours.
    Polygon { points: Vec<[f64; 2]> },
}

impl ProtectedRegion {
    /// Normalize a solver-projected pixel contour for reuse at any render size.
    pub fn polygon_from_pixels(points: &[[f64; 2]], width: usize, height: usize) -> Result<Self> {
        if width < 2 || height < 2 {
            return Err(Error::InvalidImage(
                "protected polygon normalization needs image dimensions of at least 2 by 2".into(),
            ));
        }
        let region = Self::Polygon {
            points: points
                .iter()
                .map(|point| {
                    [
                        point[0] / (width - 1) as f64,
                        point[1] / (height - 1) as f64,
                    ]
                })
                .collect(),
        };
        validate_protected_regions(std::slice::from_ref(&region))?;
        Ok(region)
    }
}

/// Surface families available to the background estimator.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ModelConfig {
    /// Select a conservative polynomial or a thin-plate radial-basis surface
    /// from held-out background samples.
    Automatic {
        /// Highest polynomial degree considered by automatic selection.
        #[serde(default = "default_automatic_max_degree")]
        max_degree: u8,
        /// Regularization applied to polynomial candidates.
        #[serde(default = "default_polynomial_ridge")]
        ridge: f64,
        /// Thin-plate spline smoothing. Larger values produce a stiffer model.
        #[serde(default = "default_rbf_smoothing")]
        rbf_smoothing: f64,
        /// Largest number of radial-basis control points retained for a fit.
        #[serde(default = "default_rbf_max_control_points")]
        max_control_points: usize,
        /// Include the flexible radial-basis candidate. This is off by default
        /// because held-out samples can still share real extended emission.
        #[serde(default)]
        allow_radial_basis: bool,
        /// Fractional validation-error improvement required before selecting a
        /// more flexible model.
        #[serde(default = "default_minimum_improvement")]
        minimum_improvement: f64,
    },
    /// A total-degree polynomial in normalized image coordinates.
    Polynomial {
        /// Total polynomial degree. Zero is a constant pedestal; four is the
        /// highest supported degree.
        degree: u8,
        /// Scale-independent Tikhonov regularization applied to non-constant
        /// coefficients after coordinate normalization.
        ridge: f64,
    },
    /// A smoothed thin-plate radial-basis surface with an affine tail.
    RadialBasis {
        /// Thin-plate spline smoothing. Zero interpolates the control points;
        /// larger values produce a stiffer surface.
        #[serde(default = "default_rbf_smoothing")]
        smoothing: f64,
        /// Largest number of accepted samples retained as control points.
        #[serde(default = "default_rbf_max_control_points")]
        max_control_points: usize,
    },
}

const fn default_automatic_max_degree() -> u8 {
    2
}

const fn default_polynomial_ridge() -> f64 {
    1.0e-8
}

const fn default_rbf_smoothing() -> f64 {
    0.01
}

const fn default_rbf_max_control_points() -> usize {
    192
}

const fn default_minimum_improvement() -> f64 {
    0.12
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::Polynomial {
            degree: 2,
            ridge: default_polynomial_ridge(),
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
    #[serde(default)]
    pub protected_regions: usize,
    /// Candidate validation scores when automatic model selection was used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_selection: Option<ModelSelectionDiagnostics>,
}

/// Held-out errors and the selected surface from automatic model selection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelSelectionDiagnostics {
    pub selected: String,
    pub candidates: Vec<ModelCandidateDiagnostics>,
}

/// One surface considered by automatic model selection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelCandidateDiagnostics {
    pub model: String,
    /// Median absolute held-out residual, normalized per channel.
    pub validation_error: f64,
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
    RadialBasis {
        smoothing: f64,
        /// Control points in normalized image coordinates.
        centers: Vec<[f64; 2]>,
        /// One coefficient vector per channel. Radial weights come first,
        /// followed by the constant, x, and y affine terms.
        coefficients: Vec<Vec<f64>>,
    },
}

impl FittedModel {
    /// Stable short name for diagnostics and file metadata.
    pub const fn family_name(&self) -> &'static str {
        match self {
            Self::Polynomial { .. } => "polynomial",
            Self::RadialBasis { .. } => "radial_basis",
        }
    }
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
    #[error("invalid correction strength: expected a finite value in [0, 1], got {0}")]
    InvalidStrength(f64),
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

#[derive(Clone, Copy, Debug)]
enum SurfaceSpec {
    Polynomial {
        degree: u8,
        ridge: f64,
    },
    RadialBasis {
        smoothing: f64,
        max_control_points: usize,
    },
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
    validate_config(config)?;
    let radius = resolved_radius(config.sample_radius, width, height);
    let mut samples = collect_samples(
        data,
        width,
        height,
        channels,
        exclusion_mask,
        &config.protected_regions,
        config,
        radius,
    );
    let required = required_samples(&config.model);
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

    let (surface, model_selection) =
        select_surface(&samples, width, height, channels, &config.model)?;
    let mut model = fit_surface(&samples, width, height, channels, surface)?;
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
                            - evaluate_model_normalized(
                                &model,
                                normalized_coordinate(sample.x, width),
                                normalized_coordinate(sample.y, height),
                                channel,
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
        model = fit_surface(&samples, width, height, channels, surface)?;
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
        model,
        reference,
        samples,
        diagnostics: FitDiagnostics {
            candidate_samples,
            accepted_samples,
            rejected_noise,
            rejected_residual,
            rejection_iterations,
            sample_radius: radius,
            protected_regions: config.protected_regions.len(),
            model_selection,
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
            FittedModel::RadialBasis {
                smoothing,
                centers,
                coefficients,
            } => {
                if !smoothing.is_finite() || *smoothing < 0.0 {
                    return Err(Error::InvalidFit(
                        "radial-basis smoothing must be finite and non-negative".into(),
                    ));
                }
                if centers.len() < 4
                    || centers
                        .iter()
                        .flatten()
                        .any(|coordinate| !coordinate.is_finite())
                {
                    return Err(Error::InvalidFit(
                        "radial-basis model needs at least four finite centers".into(),
                    ));
                }
                if coefficients.len() != self.channels {
                    return Err(Error::InvalidFit(format!(
                        "radial-basis model has {} channel coefficient sets; expected {}",
                        coefficients.len(),
                        self.channels
                    )));
                }
                let expected = centers.len() + 3;
                if let Some((channel, actual)) =
                    coefficients
                        .iter()
                        .enumerate()
                        .find_map(|(channel, values)| {
                            (values.len() != expected).then_some((channel, values.len()))
                        })
                {
                    return Err(Error::InvalidFit(format!(
                        "radial-basis channel {channel} has {actual} coefficients; expected {expected}"
                    )));
                }
                if coefficients
                    .iter()
                    .flatten()
                    .any(|coefficient| !coefficient.is_finite())
                {
                    return Err(Error::InvalidFit(
                        "radial-basis coefficients must all be finite".into(),
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
        Ok(evaluate_model_normalized(&self.model, x, y, channel))
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
        self.correct_with_strength(data, mode, 1.0)
    }

    /// Return a corrected copy with a fractional correction strength.
    ///
    /// A strength of zero leaves finite input samples unchanged; one applies
    /// the full fitted correction.
    pub fn correct_with_strength(
        &self,
        data: &[f32],
        mode: CorrectionMode,
        strength: f64,
    ) -> Result<Vec<f32>> {
        self.validate()?;
        self.validate_buffer_len(data.len())?;
        validate_strength(strength)?;
        let mut corrected = data.to_vec();
        self.correct_in_place_validated(&mut corrected, mode, strength)?;
        Ok(corrected)
    }

    /// Correct an interleaved image in place without allocating a model image.
    pub fn correct_in_place(&self, data: &mut [f32], mode: CorrectionMode) -> Result<()> {
        self.correct_in_place_with_strength(data, mode, 1.0)
    }

    /// Correct an interleaved image in place with a fractional strength.
    pub fn correct_in_place_with_strength(
        &self,
        data: &mut [f32],
        mode: CorrectionMode,
        strength: f64,
    ) -> Result<()> {
        self.validate()?;
        self.validate_buffer_len(data.len())?;
        validate_strength(strength)?;
        self.correct_in_place_validated(data, mode, strength)
    }

    fn correct_in_place_validated(
        &self,
        data: &mut [f32],
        mode: CorrectionMode,
        strength: f64,
    ) -> Result<()> {
        if strength == 0.0 {
            return Ok(());
        }
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
                                (f64::from(*value) - strength * (background - reference)) as f32
                            }
                            CorrectionMode::Divide => {
                                let full_factor = reference / background;
                                (f64::from(*value) * strength.mul_add(full_factor - 1.0, 1.0))
                                    as f32
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
        evaluate_model_normalized(&self.model, x, y, channel)
    }
}

fn validate_strength(strength: f64) -> Result<()> {
    if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
        return Err(Error::InvalidStrength(strength));
    }
    Ok(())
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

fn validate_config(config: &BackgroundConfig) -> Result<()> {
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
    validate_protected_regions(&config.protected_regions)?;
    match config.model {
        ModelConfig::Automatic {
            max_degree,
            ridge,
            rbf_smoothing,
            max_control_points,
            allow_radial_basis,
            minimum_improvement,
        } => {
            validate_polynomial(max_degree, ridge)?;
            if allow_radial_basis {
                validate_rbf(rbf_smoothing, max_control_points)?;
            }
            if !minimum_improvement.is_finite() || !(0.0..=0.75).contains(&minimum_improvement) {
                return Err(Error::InvalidConfig(
                    "automatic minimum_improvement must be finite and in [0, 0.75]".into(),
                ));
            }
            Ok(())
        }
        ModelConfig::Polynomial { degree, ridge } => validate_polynomial(degree, ridge),
        ModelConfig::RadialBasis {
            smoothing,
            max_control_points,
        } => validate_rbf(smoothing, max_control_points),
    }
}

fn validate_protected_regions(regions: &[ProtectedRegion]) -> Result<()> {
    if regions.len() > 4_096 {
        return Err(Error::InvalidConfig(
            "protected_regions must not contain more than 4096 regions".into(),
        ));
    }
    let mut total_points = 0_usize;
    for region in regions {
        match region {
            ProtectedRegion::Ellipse {
                center,
                radii,
                rotation_degrees,
            } => {
                if center.iter().any(|value| !value.is_finite())
                    || radii
                        .iter()
                        .any(|value| !value.is_finite() || *value <= 0.0 || *value > 4.0)
                    || !rotation_degrees.is_finite()
                {
                    return Err(Error::InvalidConfig(
                        "protected ellipse needs finite coordinates, radii in (0, 4], and a finite rotation"
                            .into(),
                    ));
                }
            }
            ProtectedRegion::Polygon { points } => {
                if points.len() < 3 {
                    return Err(Error::InvalidConfig(
                        "protected polygon needs at least three points".into(),
                    ));
                }
                total_points = total_points.saturating_add(points.len());
                if points
                    .iter()
                    .flatten()
                    .any(|coordinate| !coordinate.is_finite())
                {
                    return Err(Error::InvalidConfig(
                        "protected polygon coordinates must be finite".into(),
                    ));
                }
            }
        }
    }
    if total_points > 20_000 {
        return Err(Error::InvalidConfig(
            "protected polygon data must not exceed 20000 points".into(),
        ));
    }
    Ok(())
}

fn validate_polynomial(degree: u8, ridge: f64) -> Result<()> {
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
    Ok(())
}

fn validate_rbf(smoothing: f64, max_control_points: usize) -> Result<()> {
    if !smoothing.is_finite() || smoothing < 0.0 {
        return Err(Error::InvalidConfig(
            "radial-basis smoothing must be finite and non-negative".into(),
        ));
    }
    if !(16..=512).contains(&max_control_points) {
        return Err(Error::InvalidConfig(
            "radial-basis max_control_points must be between 16 and 512".into(),
        ));
    }
    Ok(())
}

fn required_samples(model: &ModelConfig) -> usize {
    match model {
        ModelConfig::Polynomial { degree, .. } => polynomial_required(*degree),
        ModelConfig::RadialBasis { .. } => 8,
        ModelConfig::Automatic { max_degree, .. } => polynomial_required(*max_degree),
    }
}

fn polynomial_required(degree: u8) -> usize {
    basis_len(degree)
        .saturating_mul(2)
        .max(basis_len(degree) + 2)
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
    protected_regions: &[ProtectedRegion],
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
                protected_regions,
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
    protected_regions: &[ProtectedRegion],
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
    let mut best = window_statistics(
        data,
        width,
        height,
        channels,
        mask,
        protected_regions,
        seed_x,
        seed_y,
        radius,
    )?;
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
                if let Some(candidate) = window_statistics(
                    data,
                    width,
                    height,
                    channels,
                    mask,
                    protected_regions,
                    x,
                    y,
                    radius,
                ) && sample_score(&candidate) < sample_score(&next)
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
    protected_regions: &[ProtectedRegion],
    x: usize,
    y: usize,
    radius: usize,
) -> Option<RawSample> {
    if sample_center_is_excluded(x, y, width, height, radius, mask, protected_regions) {
        return None;
    }
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

fn sample_center_is_excluded(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    radius: usize,
    mask: Option<&[bool]>,
    protected_regions: &[ProtectedRegion],
) -> bool {
    if mask.is_some_and(|mask| mask[y * width + x]) {
        return true;
    }
    if protected_regions.is_empty() {
        return false;
    }
    let point = [unit_coordinate(x, width), unit_coordinate(y, height)];
    let padding = radius as f64 / width.min(height).saturating_sub(1).max(1) as f64;
    protected_regions
        .iter()
        .any(|region| region_contains_with_padding(region, point, padding))
}

fn unit_coordinate(value: usize, extent: usize) -> f64 {
    if extent <= 1 {
        0.5
    } else {
        value as f64 / (extent - 1) as f64
    }
}

#[cfg(test)]
fn region_contains(region: &ProtectedRegion, point: [f64; 2]) -> bool {
    region_contains_with_padding(region, point, 0.0)
}

fn region_contains_with_padding(region: &ProtectedRegion, point: [f64; 2], padding: f64) -> bool {
    match region {
        ProtectedRegion::Ellipse {
            center,
            radii,
            rotation_degrees,
        } => {
            let angle = rotation_degrees.to_radians();
            let (sin, cos) = angle.sin_cos();
            let dx = point[0] - center[0];
            let dy = point[1] - center[1];
            let x = cos.mul_add(dx, sin * dy) / (radii[0] + padding);
            let y = (-sin).mul_add(dx, cos * dy) / (radii[1] + padding);
            x.mul_add(x, y * y) <= 1.0
        }
        ProtectedRegion::Polygon { points } => {
            point_in_polygon(point, points)
                || (padding > 0.0 && polygon_distance_squared(point, points) <= padding * padding)
        }
    }
}

fn polygon_distance_squared(point: [f64; 2], polygon: &[[f64; 2]]) -> f64 {
    let mut distance = f64::INFINITY;
    let mut previous = polygon[polygon.len() - 1];
    for &current in polygon {
        distance = distance.min(segment_distance_squared(point, previous, current));
        previous = current;
    }
    distance
}

fn segment_distance_squared(point: [f64; 2], start: [f64; 2], end: [f64; 2]) -> f64 {
    let dx = end[0] - start[0];
    let dy = end[1] - start[1];
    let length_squared = dx.mul_add(dx, dy * dy);
    if length_squared <= 1.0e-24 {
        let px = point[0] - start[0];
        let py = point[1] - start[1];
        return px.mul_add(px, py * py);
    }
    let projection = ((point[0] - start[0]).mul_add(dx, (point[1] - start[1]) * dy)
        / length_squared)
        .clamp(0.0, 1.0);
    let px = point[0] - start[0] - projection * dx;
    let py = point[1] - start[1] - projection * dy;
    px.mul_add(px, py * py)
}

fn point_in_polygon(point: [f64; 2], polygon: &[[f64; 2]]) -> bool {
    let mut inside = false;
    let mut previous = polygon[polygon.len() - 1];
    for &current in polygon {
        if point_on_segment(point, previous, current) {
            return true;
        }
        if (current[1] > point[1]) != (previous[1] > point[1]) {
            let crossing_x = (previous[0] - current[0]) * (point[1] - current[1])
                / (previous[1] - current[1])
                + current[0];
            if point[0] < crossing_x {
                inside = !inside;
            }
        }
        previous = current;
    }
    inside
}

fn point_on_segment(point: [f64; 2], start: [f64; 2], end: [f64; 2]) -> bool {
    let dx = end[0] - start[0];
    let dy = end[1] - start[1];
    let cross = (point[0] - start[0]).mul_add(dy, -(point[1] - start[1]) * dx);
    let scale = dx.abs().max(dy.abs()).max(1.0);
    if cross.abs() > 1.0e-12 * scale {
        return false;
    }
    let dot = (point[0] - start[0]).mul_add(dx, (point[1] - start[1]) * dy);
    dot >= 0.0 && dot <= dx.mul_add(dx, dy * dy)
}

fn accepted_count(samples: &[RawSample]) -> usize {
    samples
        .iter()
        .filter(|sample| sample.status == SampleStatus::Accepted)
        .count()
}

fn select_surface(
    samples: &[RawSample],
    width: usize,
    height: usize,
    channels: usize,
    config: &ModelConfig,
) -> Result<(SurfaceSpec, Option<ModelSelectionDiagnostics>)> {
    match *config {
        ModelConfig::Polynomial { degree, ridge } => {
            Ok((SurfaceSpec::Polynomial { degree, ridge }, None))
        }
        ModelConfig::RadialBasis {
            smoothing,
            max_control_points,
        } => Ok((
            SurfaceSpec::RadialBasis {
                smoothing,
                max_control_points,
            },
            None,
        )),
        ModelConfig::Automatic {
            max_degree,
            ridge,
            rbf_smoothing,
            max_control_points,
            allow_radial_basis,
            minimum_improvement,
        } => {
            let mut candidates: Vec<SurfaceSpec> = (0..=max_degree)
                .map(|degree| SurfaceSpec::Polynomial { degree, ridge })
                .collect();
            if allow_radial_basis {
                candidates.push(SurfaceSpec::RadialBasis {
                    smoothing: rbf_smoothing,
                    max_control_points,
                });
            }

            let scored: Vec<(SurfaceSpec, f64)> = candidates
                .into_iter()
                .filter_map(|candidate| {
                    cross_validation_error(samples, width, height, channels, candidate)
                        .ok()
                        .map(|score| (candidate, score))
                })
                .collect();
            let Some(&(mut selected, mut selected_error)) = scored.first() else {
                return Err(Error::SingularFit);
            };
            for &(candidate, error) in scored.iter().skip(1) {
                if error + 0.01 < selected_error * (1.0 - minimum_improvement) {
                    selected = candidate;
                    selected_error = error;
                }
            }
            let diagnostics = ModelSelectionDiagnostics {
                selected: surface_label(selected),
                candidates: scored
                    .into_iter()
                    .map(|(candidate, validation_error)| ModelCandidateDiagnostics {
                        model: surface_label(candidate),
                        validation_error,
                    })
                    .collect(),
            };
            Ok((selected, Some(diagnostics)))
        }
    }
}

fn surface_label(surface: SurfaceSpec) -> String {
    match surface {
        SurfaceSpec::Polynomial { degree, .. } => format!("polynomial_{degree}"),
        SurfaceSpec::RadialBasis { .. } => "radial_basis".into(),
    }
}

fn cross_validation_error(
    samples: &[RawSample],
    width: usize,
    height: usize,
    channels: usize,
    surface: SurfaceSpec,
) -> Result<f64> {
    let accepted: Vec<usize> = samples
        .iter()
        .enumerate()
        .filter_map(|(index, sample)| (sample.status == SampleStatus::Accepted).then_some(index))
        .collect();
    let folds = 4.min(accepted.len() / 4).max(2);
    let scales: Vec<f64> = (0..channels)
        .map(|channel| {
            let values: Vec<f64> = accepted
                .iter()
                .map(|&index| samples[index].values[channel])
                .collect();
            let center = median(&values).unwrap_or(0.0);
            seiza_stats::robust_sigma_f64(&values, center)
                .unwrap_or(0.0)
                .max(center.abs() * 1.0e-6)
                .max(1.0e-12)
        })
        .collect();
    let mut residuals = Vec::with_capacity(accepted.len() * channels);
    for fold in 0..folds {
        let mut training = samples.to_vec();
        for (position, &index) in accepted.iter().enumerate() {
            if position % folds == fold {
                training[index].status = SampleStatus::RejectedResidual;
            }
        }
        let model = fit_surface(&training, width, height, channels, surface)?;
        for (position, &index) in accepted.iter().enumerate() {
            if position % folds != fold {
                continue;
            }
            let sample = &samples[index];
            let x = normalized_coordinate(sample.x, width);
            let y = normalized_coordinate(sample.y, height);
            for (channel, &scale) in scales.iter().enumerate() {
                residuals.push(
                    (sample.values[channel] - evaluate_model_normalized(&model, x, y, channel))
                        .abs()
                        / scale,
                );
            }
        }
    }
    median(&residuals).ok_or(Error::SingularFit)
}

fn fit_surface(
    samples: &[RawSample],
    width: usize,
    height: usize,
    channels: usize,
    surface: SurfaceSpec,
) -> Result<FittedModel> {
    match surface {
        SurfaceSpec::Polynomial { degree, ridge } => Ok(FittedModel::Polynomial {
            degree,
            coefficients: fit_polynomial_channels(samples, width, height, channels, degree, ridge)?,
        }),
        SurfaceSpec::RadialBasis {
            smoothing,
            max_control_points,
        } => fit_radial_basis(
            samples,
            width,
            height,
            channels,
            smoothing,
            max_control_points,
        ),
    }
}

fn fit_polynomial_channels(
    samples: &[RawSample],
    width: usize,
    height: usize,
    channels: usize,
    degree: u8,
    ridge: f64,
) -> Result<Vec<Vec<f64>>> {
    (0..channels)
        .map(|channel| fit_polynomial_channel(samples, width, height, channel, degree, ridge))
        .collect()
}

fn fit_polynomial_channel(
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

#[allow(clippy::too_many_arguments)]
fn fit_radial_basis(
    samples: &[RawSample],
    width: usize,
    height: usize,
    channels: usize,
    smoothing: f64,
    max_control_points: usize,
) -> Result<FittedModel> {
    let mut accepted: Vec<&RawSample> = samples
        .iter()
        .filter(|sample| sample.status == SampleStatus::Accepted)
        .collect();
    accepted.sort_by_key(|sample| (sample.y, sample.x));
    let controls: Vec<&RawSample> = if accepted.len() <= max_control_points {
        accepted
    } else {
        (0..max_control_points)
            .map(|index| {
                let source = index * (accepted.len() - 1) / (max_control_points - 1);
                accepted[source]
            })
            .collect()
    };
    if controls.len() < 4 {
        return Err(Error::NotEnoughSamples {
            found: controls.len(),
            required: 4,
        });
    }
    let centers: Vec<[f64; 2]> = controls
        .iter()
        .map(|sample| {
            [
                normalized_coordinate(sample.x, width),
                normalized_coordinate(sample.y, height),
            ]
        })
        .collect();
    let count = centers.len();
    let size = count + 3;
    let coefficients = (0..channels)
        .map(|channel| {
            let mut matrix = vec![vec![0.0; size]; size];
            let mut rhs = vec![0.0; size];
            for row in 0..count {
                for column in 0..count {
                    matrix[row][column] = thin_plate_distance(centers[row], centers[column]);
                }
                matrix[row][row] += smoothing / controls[row].weight.max(0.05);
                matrix[row][count] = 1.0;
                matrix[row][count + 1] = centers[row][0];
                matrix[row][count + 2] = centers[row][1];
                matrix[count][row] = 1.0;
                matrix[count + 1][row] = centers[row][0];
                matrix[count + 2][row] = centers[row][1];
                rhs[row] = controls[row].values[channel];
            }
            solve_linear_system(matrix, rhs)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(FittedModel::RadialBasis {
        smoothing,
        centers,
        coefficients,
    })
}

fn thin_plate_distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let radius_squared = dx.mul_add(dx, dy * dy);
    if radius_squared <= 1.0e-24 {
        0.0
    } else {
        0.5 * radius_squared * radius_squared.ln()
    }
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

fn evaluate_model_normalized(model: &FittedModel, x: f64, y: f64, channel: usize) -> f64 {
    match model {
        FittedModel::Polynomial {
            degree,
            coefficients,
        } => evaluate_coefficients(&coefficients[channel], *degree, x, y),
        FittedModel::RadialBasis {
            centers,
            coefficients,
            ..
        } => {
            let channel = &coefficients[channel];
            let radial = centers
                .iter()
                .zip(channel)
                .map(|(&center, &weight)| weight * thin_plate_distance([x, y], center))
                .sum::<f64>();
            let affine = &channel[centers.len()..];
            radial + affine[0] + affine[1] * x + affine[2] * y
        }
    }
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
                protected_regions: 0,
                model_selection: None,
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
                protected_regions: 0,
                model_selection: None,
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
                protected_regions: 0,
                model_selection: None,
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
    fn normalized_catalog_outline_protects_a_solved_target() {
        let (width, height) = (80, 80);
        let expected = plane(width, height, 1);
        let mut image = expected.clone();
        for y in 20..60 {
            for x in 20..60 {
                image[y * width + x] += 0.5;
            }
        }
        let outline = ProtectedRegion::Polygon {
            points: vec![[0.24, 0.24], [0.76, 0.24], [0.76, 0.76], [0.24, 0.76]],
        };
        let config = BackgroundConfig {
            model: ModelConfig::Polynomial {
                degree: 1,
                ridge: 0.0,
            },
            samples_per_axis: 9,
            sample_radius: Some(2),
            protected_regions: vec![outline.clone()],
            ..BackgroundConfig::default()
        };
        let fit = fit_background(&image, width, height, 1, &config).unwrap();
        assert_eq!(fit.diagnostics.protected_regions, 1);
        assert!(fit.samples.iter().all(|sample| {
            !region_contains(
                &outline,
                [
                    unit_coordinate(sample.x, width),
                    unit_coordinate(sample.y, height),
                ],
            )
        }));
        assert!(
            (fit.value_at(40, 40, 0).unwrap() - f64::from(expected[40 * width + 40])).abs() < 0.005
        );
    }

    #[test]
    fn rotated_ellipse_and_polygon_edges_are_inclusive() {
        let ellipse = ProtectedRegion::Ellipse {
            center: [0.5, 0.5],
            radii: [0.3, 0.1],
            rotation_degrees: 90.0,
        };
        assert!(region_contains(&ellipse, [0.5, 0.75]));
        assert!(!region_contains(&ellipse, [0.75, 0.5]));
        let polygon = ProtectedRegion::Polygon {
            points: vec![[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]],
        };
        assert!(region_contains(&polygon, [0.5, 0.2]));
        assert!(region_contains(&polygon, [0.5, 0.4]));
        assert!(!region_contains(&polygon, [0.1, 0.1]));

        let normalized = ProtectedRegion::polygon_from_pixels(
            &[[20.0, 10.0], [80.0, 10.0], [50.0, 40.0]],
            101,
            51,
        )
        .unwrap();
        assert!(region_contains(&normalized, [0.5, 0.4]));
    }

    fn curved_gradient(width: usize, height: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(width * height);
        for y in 0..height {
            let ny = normalized_coordinate(y, height);
            for x in 0..width {
                let nx = normalized_coordinate(x, width);
                data.push(
                    (0.3 + 0.04 * nx - 0.025 * ny
                        + 0.055
                            * (std::f64::consts::PI * nx).sin()
                            * (std::f64::consts::FRAC_PI_2 * ny).cos()) as f32,
                );
            }
        }
        data
    }

    #[test]
    fn radial_basis_recovers_a_smooth_irregular_gradient() {
        let (width, height) = (112, 84);
        let image = curved_gradient(width, height);
        let config = BackgroundConfig {
            model: ModelConfig::RadialBasis {
                smoothing: 0.002,
                max_control_points: 192,
            },
            samples_per_axis: 14,
            sample_radius: Some(1),
            search_steps: 0,
            ..BackgroundConfig::default()
        };
        let fit = fit_background(&image, width, height, 1, &config).unwrap();
        assert!(matches!(fit.model, FittedModel::RadialBasis { .. }));
        let model = fit.render_model().unwrap();
        let rmse = model
            .iter()
            .zip(&image)
            .map(|(actual, expected)| f64::from(*actual - *expected).powi(2))
            .sum::<f64>()
            / model.len() as f64;
        let rmse = rmse.sqrt();
        assert!(rmse < 0.004, "radial-basis model RMSE was {rmse}");
    }

    #[test]
    fn automatic_selection_uses_held_out_samples_for_model_choice() {
        let (width, height) = (112, 84);
        let config = BackgroundConfig {
            model: ModelConfig::Automatic {
                max_degree: 2,
                ridge: 1.0e-8,
                rbf_smoothing: 0.002,
                max_control_points: 192,
                allow_radial_basis: true,
                minimum_improvement: 0.08,
            },
            samples_per_axis: 14,
            sample_radius: Some(1),
            search_steps: 0,
            ..BackgroundConfig::default()
        };
        let curved = curved_gradient(width, height);
        let curved_fit = fit_background(&curved, width, height, 1, &config).unwrap();
        let selection = curved_fit.diagnostics.model_selection.as_ref().unwrap();
        assert_eq!(selection.selected, "radial_basis");
        assert!(selection.candidates.len() >= 4);

        let mut conservative = config.clone();
        let ModelConfig::Automatic {
            allow_radial_basis, ..
        } = &mut conservative.model
        else {
            unreachable!();
        };
        *allow_radial_basis = false;
        let conservative_fit = fit_background(&curved, width, height, 1, &conservative).unwrap();
        let conservative_selection = conservative_fit
            .diagnostics
            .model_selection
            .as_ref()
            .unwrap();
        assert!(matches!(
            conservative_fit.model,
            FittedModel::Polynomial { .. }
        ));
        assert!(
            conservative_selection
                .candidates
                .iter()
                .all(|candidate| candidate.model != "radial_basis")
        );

        let target_outline = ProtectedRegion::Polygon {
            points: vec![[0.35, 0.3], [0.7, 0.3], [0.7, 0.75], [0.35, 0.75]],
        };
        let mut protected_image = curved.clone();
        for y in 26..63 {
            for x in 40..78 {
                protected_image[y * width + x] += 0.5;
            }
        }
        let mut protected_config = config.clone();
        protected_config.protected_regions = vec![target_outline];
        let protected_fit =
            fit_background(&protected_image, width, height, 1, &protected_config).unwrap();
        assert_eq!(
            protected_fit
                .diagnostics
                .model_selection
                .as_ref()
                .unwrap()
                .selected,
            "radial_basis"
        );
        let center = height / 2 * width + width / 2;
        assert!(
            (protected_fit.value_at(width / 2, height / 2, 0).unwrap() - f64::from(curved[center]))
                .abs()
                < 0.015
        );

        let planar = plane(width, height, 1);
        let planar_fit = fit_background(&planar, width, height, 1, &config).unwrap();
        assert_eq!(
            planar_fit
                .diagnostics
                .model_selection
                .as_ref()
                .unwrap()
                .selected,
            "polynomial_1"
        );
    }

    #[test]
    fn correction_strength_blends_from_unchanged_to_full_correction() {
        let (width, height) = (96, 72);
        let image = plane(width, height, 1);
        let config = BackgroundConfig {
            model: ModelConfig::Polynomial {
                degree: 1,
                ridge: 0.0,
            },
            sample_radius: Some(2),
            ..BackgroundConfig::default()
        };
        let fit = fit_background(&image, width, height, 1, &config).unwrap();
        let unchanged = fit
            .correct_with_strength(&image, CorrectionMode::Subtract, 0.0)
            .unwrap();
        assert_eq!(unchanged, image);
        let half = fit
            .correct_with_strength(&image, CorrectionMode::Subtract, 0.5)
            .unwrap();
        let full = fit.correct(&image, CorrectionMode::Subtract).unwrap();
        let left = height / 2 * width + 3;
        let right = height / 2 * width + width - 4;
        let original_span = image[right] - image[left];
        let half_span = half[right] - half[left];
        let full_span = full[right] - full[left];
        assert!((half_span - original_span * 0.5).abs() < 0.002);
        assert!(full_span.abs() < 0.002);
        assert_eq!(
            fit.correct_with_strength(&image, CorrectionMode::Subtract, 1.1),
            Err(Error::InvalidStrength(1.1))
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
        ) = (&decoded.model, &fit.model)
        else {
            panic!("default background model should remain polynomial");
        };
        assert!(
            coefficients[0]
                .iter()
                .zip(&expected[0])
                .all(|(actual, expected)| (*actual - *expected).abs() < 1.0e-15)
        );
    }
}
