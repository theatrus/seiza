//! Reusable parameterized display stretching for astrophotography pipelines.
//!
//! Requested [`StretchModel`] values are resolved against a reusable
//! [`StretchAnalysis`] into deterministic [`StretchPlan`] values. This keeps
//! image-dependent parameter selection separate from curve application and
//! permits interactive callers to reuse analysis across parameter changes.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const NORMAL_MAD_SCALE: f64 = 1.4826;
const LUMA_RED: f64 = 0.2126;
const LUMA_GREEN: f64 = 0.7152;
const LUMA_BLUE: f64 = 0.0722;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum StretchError {
    #[error("{0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, StretchError>;

/// How one or more channels share analysis and transfer curves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ColorStrategy {
    /// Use one analysis distribution and one curve for every channel.
    #[default]
    Linked,
    /// Analyze and stretch each channel independently.
    Unlinked,
    /// Analyze Rec.709 luminance and scale non-negative RGB triplets by one
    /// common factor, preserving chromaticity while keeping the result in gamut.
    LuminancePreserving,
}

/// Parameters for a deterministic Generalized Hyperbolic Stretch (GHS).
///
/// These follow the parameterization documented by the Generalized Hyperbolic
/// Stretch PixInsight process. Black and white provide an explicit input range;
/// the remaining values describe the normalized GHS transfer curve.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GhsParams {
    /// Logarithmic stretch factor `S = ln(D + 1)`, in `[0, 20]`.
    pub stretch_factor: f64,
    /// Local stretch intensity `b`, in `[-5, 15]`.
    pub local_intensity: f64,
    /// Symmetry point, in `[0, 1]`.
    pub symmetry_point: f64,
    /// Linear shadow-protection boundary, in `[0, symmetry_point]`.
    pub protect_shadows: f64,
    /// Linear highlight-protection boundary, in `[symmetry_point, 1]`.
    pub protect_highlights: f64,
    /// Explicit input black point.
    pub black: f64,
    /// Explicit input white point.
    pub white: f64,
}

/// A parameterized stretch request, before image-dependent values are resolved.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum StretchModel {
    /// Clamp already display-referred samples to `[0, 1]`.
    Identity,
    /// Affinely map an explicit input range to `[0, 1]`.
    Linear { black: f64, white: f64 },
    /// Apply an asinh curve after mapping an explicit input range.
    Asinh {
        black: f64,
        white: f64,
        strength: f64,
    },
    /// Resolve black and white from robust percentiles, then apply asinh.
    PercentileAsinh {
        black_percentile: f64,
        white_percentile: f64,
        strength: f64,
    },
    /// Apply an already resolved PixInsight/N.I.N.A.-family MTF curve.
    Mtf {
        shadows: f64,
        midtone: f64,
        highlights: f64,
    },
    /// Apply a manual Generalized Hyperbolic Stretch.
    Ghs(GhsParams),
    /// Resolve the existing median/MAD Auto-MTF model from image statistics.
    AutoMtf(StretchParams),
}

/// Complete pipeline request for analysis and stretch resolution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StretchConfig {
    pub model: StretchModel,
    pub color_strategy: ColorStrategy,
    /// Maximum pooled scalar samples retained; auxiliary channel and
    /// luminance distributions reuse the same sampled pixels.
    pub max_analysis_samples: usize,
}

impl StretchConfig {
    pub fn percentile_asinh(
        black_percentile: f64,
        white_percentile: f64,
        strength: f64,
        max_analysis_samples: usize,
    ) -> Self {
        Self {
            model: StretchModel::PercentileAsinh {
                black_percentile,
                white_percentile,
                strength,
            },
            color_strategy: ColorStrategy::Linked,
            max_analysis_samples,
        }
    }

    pub fn auto_mtf(params: StretchParams, max_analysis_samples: usize) -> Self {
        Self {
            model: StretchModel::AutoMtf(params),
            color_strategy: ColorStrategy::Linked,
            max_analysis_samples,
        }
    }

    /// Analyze input using this configuration's bounded sample limit.
    pub fn analyze(&self, data: &[f32], channel_count: usize) -> Result<StretchAnalysis> {
        StretchAnalysis::analyze(data, channel_count, self.max_analysis_samples)
    }

    /// Resolve this request against reusable image analysis.
    pub fn resolve(&self, analysis: &StretchAnalysis) -> Result<StretchPlan> {
        if !self.model.requires_analysis() {
            return self.resolve_explicit(analysis.channel_count);
        }
        if self.max_analysis_samples == 0 {
            return Err(StretchError::Invalid(
                "maximum analysis samples must be greater than zero".into(),
            ));
        }
        let distributions = match self.color_strategy {
            ColorStrategy::Linked => vec![&analysis.linked],
            ColorStrategy::Unlinked => analysis.channels.iter().collect(),
            ColorStrategy::LuminancePreserving => {
                vec![analysis.luminance.as_ref().ok_or_else(|| {
                    StretchError::Invalid(
                        "luminance-preserving stretch requires exactly three channels".into(),
                    )
                })?]
            }
        };
        let curves = distributions
            .into_iter()
            .map(|distribution| self.model.resolve(distribution))
            .collect::<Result<Vec<_>>>()?;
        StretchPlan::from_resolved(analysis.channel_count, self.color_strategy, curves)
    }

    /// Resolve a deterministic model without collecting image statistics.
    ///
    /// Percentile-asinh and Auto-MTF return an error here because their
    /// parameters depend on a [`StretchAnalysis`].
    pub fn resolve_explicit(&self, channel_count: usize) -> Result<StretchPlan> {
        let curve = self.model.resolve_explicit()?;
        let curves = match self.color_strategy {
            ColorStrategy::Unlinked => vec![curve; channel_count],
            ColorStrategy::Linked | ColorStrategy::LuminancePreserving => vec![curve],
        };
        StretchPlan::from_resolved(channel_count, self.color_strategy, curves)
    }
}

/// Parameters for the existing median/MAD Auto-MTF model.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct StretchParams {
    /// Where the median should land in the output (N.I.N.A. default 0.2).
    pub target_median: f64,
    /// Shadow clipping point in MADs relative to the median (default -2.8).
    pub shadows_clip: f64,
}

impl Default for StretchParams {
    fn default() -> Self {
        Self {
            target_median: 0.2,
            shadows_clip: -2.8,
        }
    }
}

/// Robust statistics estimated from the bounded analysis sample.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RobustStatistics {
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub mad: f64,
    pub count: usize,
}

#[derive(Clone, Debug)]
struct Distribution {
    sorted: Vec<f32>,
    statistics: RobustStatistics,
}

impl Distribution {
    fn new(mut values: Vec<f32>, name: &str) -> Result<Self> {
        values.retain(|value| value.is_finite());
        if values.is_empty() {
            return Err(StretchError::Invalid(format!(
                "{name} contains no finite samples"
            )));
        }
        values.sort_unstable_by(f32::total_cmp);
        let median = f64::from(values[values.len() / 2]);
        let mut deviations = values
            .iter()
            .map(|value| (f64::from(*value) - median).abs())
            .collect::<Vec<_>>();
        deviations.sort_unstable_by(f64::total_cmp);
        let statistics = RobustStatistics {
            min: f64::from(values[0]),
            max: f64::from(*values.last().expect("non-empty sample")),
            median,
            mad: deviations[deviations.len() / 2],
            count: values.len(),
        };
        Ok(Self {
            sorted: values,
            statistics,
        })
    }

    fn percentile(&self, percentile: f64) -> f64 {
        let index =
            ((self.sorted.len() as f64 * percentile).floor() as usize).min(self.sorted.len() - 1);
        f64::from(self.sorted[index])
    }
}

/// Bounded analysis that can be reused while interactive parameters change.
#[derive(Clone, Debug)]
pub struct StretchAnalysis {
    channel_count: usize,
    linked: Distribution,
    channels: Vec<Distribution>,
    luminance: Option<Distribution>,
}

impl StretchAnalysis {
    /// Analyze mono or interleaved RGB `f32` samples with bounded memory.
    pub fn analyze(data: &[f32], channel_count: usize, max_samples: usize) -> Result<Self> {
        if channel_count == 0 || data.is_empty() || !data.len().is_multiple_of(channel_count) {
            return Err(StretchError::Invalid(
                "image samples must have non-zero dimensions and complete interleaved pixels"
                    .into(),
            ));
        }
        if max_samples < channel_count {
            return Err(StretchError::Invalid(
                "maximum analysis samples must accommodate one complete pixel".into(),
            ));
        }
        let pixel_count = data.len() / channel_count;
        let maximum_pixels = (max_samples / channel_count).max(1);
        let pixel_stride = pixel_count.div_ceil(maximum_pixels).max(1);
        let mut linked = Vec::with_capacity(max_samples.min(data.len()));
        let mut channels = (0..channel_count)
            .map(|_| Vec::with_capacity(maximum_pixels.min(pixel_count)))
            .collect::<Vec<_>>();
        let mut luminance =
            (channel_count == 3).then(|| Vec::with_capacity(maximum_pixels.min(pixel_count)));

        for pixel in data.chunks_exact(channel_count).step_by(pixel_stride) {
            for (channel, value) in pixel.iter().copied().enumerate() {
                if value.is_finite() {
                    linked.push(value);
                    channels[channel].push(value);
                }
            }
            if let Some(luminance) = &mut luminance
                && pixel.iter().all(|value| value.is_finite())
            {
                luminance.push(
                    (LUMA_RED * f64::from(pixel[0])
                        + LUMA_GREEN * f64::from(pixel[1])
                        + LUMA_BLUE * f64::from(pixel[2])) as f32,
                );
            }
        }

        Ok(Self {
            channel_count,
            linked: Distribution::new(linked, "image")?,
            channels: channels
                .into_iter()
                .enumerate()
                .map(|(index, values)| Distribution::new(values, &format!("channel {index}")))
                .collect::<Result<Vec<_>>>()?,
            luminance: luminance
                .map(|values| Distribution::new(values, "luminance"))
                .transpose()?,
        })
    }

    pub fn channel_count(&self) -> usize {
        self.channel_count
    }

    pub fn linked_statistics(&self) -> RobustStatistics {
        self.linked.statistics
    }

    pub fn channel_statistics(&self) -> Vec<RobustStatistics> {
        self.channels
            .iter()
            .map(|channel| channel.statistics)
            .collect()
    }

    pub fn luminance_statistics(&self) -> Option<RobustStatistics> {
        self.luminance.as_ref().map(|luma| luma.statistics)
    }
}

/// A concrete transfer curve with no unresolved image-dependent parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ResolvedCurve {
    Identity,
    Linear {
        black: f64,
        white: f64,
    },
    Asinh {
        black: f64,
        white: f64,
        strength: f64,
    },
    Mtf {
        shadows: f64,
        midtone: f64,
        highlights: f64,
    },
    Ghs(ResolvedGhs),
}

/// Precomputed GHS transfer state used by a resolved plan.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ResolvedGhs {
    params: GhsParams,
    d: f64,
    low_value: f64,
    low_slope: f64,
    high_value: f64,
    high_slope: f64,
    normalization_min: f64,
    normalization_span: f64,
}

impl ResolvedGhs {
    fn new(params: GhsParams) -> Result<Self> {
        validate_ghs(params)?;
        let d = params.stretch_factor.exp_m1();
        let low_distance = params.symmetry_point - params.protect_shadows;
        let high_distance = params.protect_highlights - params.symmetry_point;
        let low_value = -ghs_base(d, params.local_intensity, low_distance);
        let low_slope = ghs_derivative(d, params.local_intensity, low_distance);
        let high_value = ghs_base(d, params.local_intensity, high_distance);
        let high_slope = ghs_derivative(d, params.local_intensity, high_distance);
        let normalization_min = low_slope * -params.protect_shadows + low_value;
        let normalization_max = high_slope * (1.0 - params.protect_highlights) + high_value;
        let normalization_span = normalization_max - normalization_min;
        if !normalization_span.is_finite() || normalization_span <= 0.0 {
            return Err(StretchError::Invalid(
                "GHS parameters do not produce a finite increasing curve".into(),
            ));
        }
        Ok(Self {
            params,
            d,
            low_value,
            low_slope,
            high_value,
            high_slope,
            normalization_min,
            normalization_span,
        })
    }

    /// Return the requested GHS parameters retained for provenance and UI use.
    pub fn params(self) -> GhsParams {
        self.params
    }

    fn map(self, value: f64) -> f64 {
        let x =
            ((value - self.params.black) / (self.params.white - self.params.black)).clamp(0.0, 1.0);
        let raw = if x < self.params.protect_shadows {
            self.low_slope * (x - self.params.protect_shadows) + self.low_value
        } else if x < self.params.symmetry_point {
            -ghs_base(
                self.d,
                self.params.local_intensity,
                self.params.symmetry_point - x,
            )
        } else if x < self.params.protect_highlights {
            ghs_base(
                self.d,
                self.params.local_intensity,
                x - self.params.symmetry_point,
            )
        } else {
            self.high_slope * (x - self.params.protect_highlights) + self.high_value
        };
        (raw - self.normalization_min) / self.normalization_span
    }
}

impl ResolvedCurve {
    pub fn map(self, value: f32) -> f32 {
        if !value.is_finite() {
            return 0.0;
        }
        let value = f64::from(value);
        let mapped = match self {
            Self::Identity => value,
            Self::Linear { black, white } => (value - black) / (white - black),
            Self::Asinh {
                black,
                white,
                strength,
            } => {
                let linear = ((value - black) / (white - black)).max(0.0);
                (strength * linear).asinh() / strength.asinh()
            }
            Self::Mtf {
                shadows,
                midtone,
                highlights,
            } => {
                let input = 1.0 - highlights + value - shadows;
                midtones_transfer_function(midtone, input)
            }
            Self::Ghs(ghs) => ghs.map(value),
        };
        mapped.clamp(0.0, 1.0) as f32
    }
}

/// Resolved, reusable stretch curves for a fixed channel strategy.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StretchPlan {
    channels: usize,
    color_strategy: ColorStrategy,
    curves: Vec<ResolvedCurve>,
}

impl StretchPlan {
    /// Build a plan from explicit curves without performing image analysis.
    pub fn from_resolved(
        channels: usize,
        color_strategy: ColorStrategy,
        curves: Vec<ResolvedCurve>,
    ) -> Result<Self> {
        let valid = match color_strategy {
            ColorStrategy::Linked => channels > 0 && curves.len() == 1,
            ColorStrategy::Unlinked => channels > 0 && curves.len() == channels,
            ColorStrategy::LuminancePreserving => channels == 3 && curves.len() == 1,
        };
        if !valid {
            return Err(StretchError::Invalid(
                "resolved curves do not match the requested channel strategy".into(),
            ));
        }
        Ok(Self {
            channels,
            color_strategy,
            curves,
        })
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn color_strategy(&self) -> ColorStrategy {
        self.color_strategy
    }

    pub fn curves(&self) -> &[ResolvedCurve] {
        &self.curves
    }

    pub fn apply_f32(&self, data: &[f32], channel_count: usize) -> Result<Vec<f32>> {
        self.validate_input(data, channel_count)?;
        let output = match self.color_strategy {
            ColorStrategy::Linked => data
                .par_iter()
                .map(|value| self.curves[0].map(*value))
                .collect(),
            ColorStrategy::Unlinked => data
                .par_iter()
                .enumerate()
                .map(|(index, value)| self.curves[index % channel_count].map(*value))
                .collect(),
            ColorStrategy::LuminancePreserving => data
                .par_chunks_exact(3)
                .flat_map_iter(|pixel| self.map_luminance_pixel(pixel))
                .collect(),
        };
        Ok(output)
    }

    pub fn apply_u8(&self, data: &[f32], channel_count: usize) -> Result<Vec<u8>> {
        self.validate_input(data, channel_count)?;
        let output = match self.color_strategy {
            ColorStrategy::Linked => data
                .par_iter()
                .map(|value| to_u8(self.curves[0].map(*value)))
                .collect(),
            ColorStrategy::Unlinked => data
                .par_iter()
                .enumerate()
                .map(|(index, value)| to_u8(self.curves[index % channel_count].map(*value)))
                .collect(),
            ColorStrategy::LuminancePreserving => data
                .par_chunks_exact(3)
                .flat_map_iter(|pixel| self.map_luminance_pixel(pixel).map(to_u8))
                .collect(),
        };
        Ok(output)
    }

    fn validate_input(&self, data: &[f32], channel_count: usize) -> Result<()> {
        if channel_count != self.channels
            || channel_count == 0
            || !data.len().is_multiple_of(channel_count)
        {
            return Err(StretchError::Invalid(
                "stretch plan and input channel layout differ".into(),
            ));
        }
        Ok(())
    }

    fn map_luminance_pixel(&self, pixel: &[f32]) -> [f32; 3] {
        if pixel.iter().any(|value| !value.is_finite()) {
            return [0.0; 3];
        }
        let luminance = (LUMA_RED * f64::from(pixel[0])
            + LUMA_GREEN * f64::from(pixel[1])
            + LUMA_BLUE * f64::from(pixel[2])) as f32;
        let target = self.curves[0].map(luminance);
        if luminance > 1.0e-8 {
            let mut scale = target / luminance;
            let maximum = pixel.iter().copied().fold(0.0_f32, f32::max);
            if maximum * scale > 1.0 {
                scale = 1.0 / maximum;
            }
            [
                (pixel[0] * scale).clamp(0.0, 1.0),
                (pixel[1] * scale).clamp(0.0, 1.0),
                (pixel[2] * scale).clamp(0.0, 1.0),
            ]
        } else {
            [target; 3]
        }
    }
}

impl StretchModel {
    /// Whether this model needs an image analysis before it can be resolved.
    pub fn requires_analysis(&self) -> bool {
        matches!(self, Self::PercentileAsinh { .. } | Self::AutoMtf(_))
    }

    fn resolve(&self, distribution: &Distribution) -> Result<ResolvedCurve> {
        let curve = match *self {
            Self::PercentileAsinh {
                black_percentile,
                white_percentile,
                strength,
            } => {
                validate_percentiles(black_percentile, white_percentile)?;
                validate_strength(strength)?;
                let black = distribution.percentile(black_percentile);
                let white = distribution.percentile(white_percentile);
                validate_range(black, white)?;
                ResolvedCurve::Asinh {
                    black,
                    white,
                    strength,
                }
            }
            Self::AutoMtf(params) => auto_mtf_curve(distribution.statistics, params)?,
            _ => self.resolve_explicit()?,
        };
        Ok(curve)
    }

    fn resolve_explicit(&self) -> Result<ResolvedCurve> {
        let curve = match *self {
            Self::Identity => ResolvedCurve::Identity,
            Self::Linear { black, white } => {
                validate_range(black, white)?;
                ResolvedCurve::Linear { black, white }
            }
            Self::Asinh {
                black,
                white,
                strength,
            } => {
                validate_range(black, white)?;
                validate_strength(strength)?;
                ResolvedCurve::Asinh {
                    black,
                    white,
                    strength,
                }
            }
            Self::Mtf {
                shadows,
                midtone,
                highlights,
            } => {
                validate_mtf(shadows, midtone, highlights)?;
                ResolvedCurve::Mtf {
                    shadows,
                    midtone,
                    highlights,
                }
            }
            Self::Ghs(params) => {
                validate_ghs(params)?;
                if params.stretch_factor == 0.0 {
                    ResolvedCurve::Linear {
                        black: params.black,
                        white: params.white,
                    }
                } else {
                    ResolvedCurve::Ghs(ResolvedGhs::new(params)?)
                }
            }
            Self::PercentileAsinh { .. } | Self::AutoMtf(_) => {
                return Err(StretchError::Invalid(
                    "data-driven stretch model requires image analysis".into(),
                ));
            }
        };
        Ok(curve)
    }
}

fn validate_ghs(params: GhsParams) -> Result<()> {
    validate_range(params.black, params.white)?;
    if !params.stretch_factor.is_finite()
        || !(0.0..=20.0).contains(&params.stretch_factor)
        || !params.local_intensity.is_finite()
        || !(-5.0..=15.0).contains(&params.local_intensity)
        || !params.symmetry_point.is_finite()
        || !(0.0..=1.0).contains(&params.symmetry_point)
        || !params.protect_shadows.is_finite()
        || !(0.0..=params.symmetry_point).contains(&params.protect_shadows)
        || !params.protect_highlights.is_finite()
        || !(params.symmetry_point..=1.0).contains(&params.protect_highlights)
    {
        return Err(StretchError::Invalid(
            "GHS parameters must satisfy S in [0, 20], b in [-5, 15], and 0 <= LP <= SP <= HP <= 1"
                .into(),
        ));
    }
    Ok(())
}

// Formulae from the Generalized Hyperbolic Stretch reference implementation:
// https://www.ghsastro.co.uk/doc/tools/GeneralizedHyperbolicStretch/GeneralizedHyperbolicStretch.html
fn ghs_base(d: f64, b: f64, x: f64) -> f64 {
    const SPECIAL_EPSILON: f64 = 1.0e-8;
    if (b + 1.0).abs() < SPECIAL_EPSILON {
        (d * x).ln_1p()
    } else if b < 0.0 {
        (1.0 - (1.0 - b * d * x).powf((b + 1.0) / b)) / (d * (b + 1.0))
    } else if b.abs() < SPECIAL_EPSILON {
        1.0 - (-d * x).exp()
    } else if (b - 1.0).abs() < SPECIAL_EPSILON {
        1.0 - (1.0 + d * x).recip()
    } else {
        1.0 - (1.0 + b * d * x).powf(-1.0 / b)
    }
}

fn ghs_derivative(d: f64, b: f64, x: f64) -> f64 {
    const SPECIAL_EPSILON: f64 = 1.0e-8;
    if (b + 1.0).abs() < SPECIAL_EPSILON {
        d / (1.0 + d * x)
    } else if b < 0.0 {
        (1.0 - b * d * x).powf(1.0 / b)
    } else if b.abs() < SPECIAL_EPSILON {
        d * (-d * x).exp()
    } else if (b - 1.0).abs() < SPECIAL_EPSILON {
        d * (1.0 + d * x).powi(-2)
    } else {
        d * (1.0 + b * d * x).powf(-(1.0 + b) / b)
    }
}

fn auto_mtf_curve(statistics: RobustStatistics, params: StretchParams) -> Result<ResolvedCurve> {
    if !params.target_median.is_finite() || !(0.0..1.0).contains(&params.target_median) {
        return Err(StretchError::Invalid(
            "target median must be finite and between zero and one".into(),
        ));
    }
    if !params.shadows_clip.is_finite() || params.shadows_clip > 0.0 {
        return Err(StretchError::Invalid(
            "shadows clip must be finite and non-positive".into(),
        ));
    }
    let normal_mad = statistics.mad * NORMAL_MAD_SCALE;
    if normal_mad <= f64::from(f32::EPSILON) {
        return Ok(ResolvedCurve::Identity);
    }
    let (shadows, midtone, highlights) = if statistics.median > 0.5 {
        let highlights = statistics.median - params.shadows_clip * normal_mad;
        let midtone = midtones_transfer_function(
            params.target_median,
            1.0 - (highlights - statistics.median),
        );
        (0.0, midtone, highlights)
    } else {
        let shadows = (statistics.median + params.shadows_clip * normal_mad).max(0.0);
        let midtone = midtones_transfer_function(params.target_median, statistics.median - shadows);
        (shadows, midtone, 1.0)
    };
    validate_mtf(shadows, midtone, highlights)?;
    Ok(ResolvedCurve::Mtf {
        shadows,
        midtone,
        highlights,
    })
}

fn validate_range(black: f64, white: f64) -> Result<()> {
    if !black.is_finite() || !white.is_finite() || white <= black {
        return Err(StretchError::Invalid(
            "stretch black and white points must be finite and increasing".into(),
        ));
    }
    Ok(())
}

fn validate_strength(strength: f64) -> Result<()> {
    if !strength.is_finite() || strength <= 0.0 {
        return Err(StretchError::Invalid(
            "asinh strength must be finite and greater than zero".into(),
        ));
    }
    Ok(())
}

fn validate_percentiles(black: f64, white: f64) -> Result<()> {
    if !black.is_finite()
        || !white.is_finite()
        || !(0.0..=1.0).contains(&black)
        || !(0.0..=1.0).contains(&white)
        || white <= black
    {
        return Err(StretchError::Invalid(
            "stretch percentiles must be finite, within zero and one, and increasing".into(),
        ));
    }
    Ok(())
}

fn validate_mtf(shadows: f64, midtone: f64, highlights: f64) -> Result<()> {
    if !shadows.is_finite()
        || !highlights.is_finite()
        || highlights <= shadows
        || !midtone.is_finite()
        || !(0.0..1.0).contains(&midtone)
    {
        return Err(StretchError::Invalid(
            "MTF shadows, midtone, and highlights must form a finite valid curve".into(),
        ));
    }
    Ok(())
}

fn to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// The PixInsight/N.I.N.A. midtones transfer function.
pub fn midtones_transfer_function(midtone: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    (midtone - 1.0) * x / ((2.0 * midtone - 1.0) * x - midtone)
}

/// Exact `u16` histogram statistics retained for `seiza-fits` compatibility.
#[derive(Debug, Clone, PartialEq)]
pub struct Statistics {
    pub min: u16,
    pub max: u16,
    pub mean: f64,
    pub std_dev: f64,
    pub median: u16,
    pub mad: f64,
    pub count: usize,
}

pub fn statistics_u16(data: &[u16]) -> Statistics {
    let mut histogram = vec![0u32; 65_536];
    let mut sum = 0u64;
    let mut sum_sq = 0u128;
    for &value in data {
        histogram[value as usize] += 1;
        sum += u64::from(value);
        sum_sq += u128::from(value) * u128::from(value);
    }
    let count = data.len();
    if count == 0 {
        return Statistics {
            min: 0,
            max: 0,
            mean: 0.0,
            std_dev: 0.0,
            median: 0,
            mad: 0.0,
            count: 0,
        };
    }

    let min = histogram.iter().position(|count| *count > 0).unwrap_or(0) as u16;
    let max = (65_535
        - histogram
            .iter()
            .rev()
            .position(|count| *count > 0)
            .unwrap_or(0)) as u16;
    let half = count.div_ceil(2) as u64;
    let mut cumulative = 0u64;
    let mut median = 0u16;
    for (value, bin_count) in histogram.iter().enumerate() {
        cumulative += u64::from(*bin_count);
        if cumulative >= half {
            median = value as u16;
            break;
        }
    }
    let median_index = usize::from(median);
    let mut inside = u64::from(histogram[median_index]);
    let mut mad = 0u32;
    while inside < half && mad < 65_535 {
        mad += 1;
        if let Some(count) = histogram.get(median_index + mad as usize) {
            inside += u64::from(*count);
        }
        if mad as usize <= median_index {
            inside += u64::from(histogram[median_index - mad as usize]);
        }
    }
    let mean = sum as f64 / count as f64;
    let variance = (sum_sq as f64 / count as f64 - mean * mean).max(0.0);
    Statistics {
        min,
        max,
        mean,
        std_dev: variance.sqrt(),
        median,
        mad: f64::from(mad),
        count,
    }
}

/// Stretch `u16` data directly to `u8` through the existing 65,536-entry LUT.
pub fn stretch_u16_to_u8(data: &[u16], stats: &Statistics, params: &StretchParams) -> Vec<u8> {
    let map = stretch_u16_map(stats, params);
    data.iter().map(|value| map[usize::from(*value)]).collect()
}

fn stretch_u16_map(stats: &Statistics, params: &StretchParams) -> Vec<u8> {
    let normalized_median = f64::from(stats.median) / 65_535.0;
    let normalized_mad = stats.mad / 65_535.0 * NORMAL_MAD_SCALE;
    let (shadows, midtone, highlights) = if normalized_median > 0.5 {
        let highlights = normalized_median - params.shadows_clip * normalized_mad;
        let midtone = midtones_transfer_function(
            params.target_median,
            1.0 - (highlights - normalized_median),
        );
        (0.0, midtone, highlights)
    } else {
        let shadows = (normalized_median + params.shadows_clip * normalized_mad).max(0.0);
        let midtone = midtones_transfer_function(params.target_median, normalized_median - shadows);
        (shadows, midtone, 1.0)
    };

    (0..65_536)
        .map(|value| {
            let input = 1.0 - highlights + value as f64 / 65_535.0 - shadows;
            to_u8(midtones_transfer_function(midtone, input) as f32)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_asinh_matches_the_existing_preview_map() {
        let mut data = vec![0.0, 0.5, 1.0];
        data.extend((0..997).map(|index| index as f32 / 996.0));
        let analysis = StretchAnalysis::analyze(&data, 1, 200_000).unwrap();
        let config = StretchConfig::percentile_asinh(0.01, 0.995, 10.0, 200_000);
        let plan = config.resolve(&analysis).unwrap();
        let ResolvedCurve::Asinh {
            black,
            white,
            strength,
        } = plan.curves()[0]
        else {
            panic!("expected an asinh curve");
        };
        assert_eq!(black, f64::from(analysis.linked.sorted[10]));
        assert_eq!(white, f64::from(analysis.linked.sorted[995]));
        assert_eq!(strength, 10.0);
    }

    #[test]
    fn analysis_is_bounded_and_color_aware() {
        let data = [0.1, 0.2, 0.3].repeat(100_000);
        let analysis = StretchAnalysis::analyze(&data, 3, 3_000).unwrap();
        assert!(analysis.linked_statistics().count <= 3_000);
        assert_eq!(analysis.channel_statistics().len(), 3);
        assert!(analysis.luminance_statistics().is_some());
    }

    #[test]
    fn luminance_preserving_strategy_keeps_rgb_ratios() {
        let data = [0.2, 0.4, 0.1];
        let analysis = StretchAnalysis::analyze(&data, 3, 100).unwrap();
        let config = StretchConfig {
            model: StretchModel::Asinh {
                black: 0.0,
                white: 1.0,
                strength: 10.0,
            },
            color_strategy: ColorStrategy::LuminancePreserving,
            max_analysis_samples: 100,
        };
        let output = config
            .resolve(&analysis)
            .unwrap()
            .apply_f32(&data, 3)
            .unwrap();
        assert!((output[0] / output[1] - 0.5).abs() < 1.0e-6);
        assert!((output[2] / output[1] - 0.25).abs() < 1.0e-6);
        assert!(output.iter().all(|value| (0.0..=1.0).contains(value)));
    }

    #[test]
    fn ghs_exponential_subtype_matches_the_reference_equation() {
        let data = [0.0, 0.25, 1.0];
        let analysis = StretchAnalysis::analyze(&data, 1, data.len()).unwrap();
        let config = StretchConfig {
            model: StretchModel::Ghs(GhsParams {
                stretch_factor: 10.0_f64.ln_1p(),
                local_intensity: 0.0,
                symmetry_point: 0.0,
                protect_shadows: 0.0,
                protect_highlights: 1.0,
                black: 0.0,
                white: 1.0,
            }),
            color_strategy: ColorStrategy::Linked,
            max_analysis_samples: data.len(),
        };
        let output = config
            .resolve(&analysis)
            .unwrap()
            .apply_f32(&data, 1)
            .unwrap();
        let expected = (1.0 - (-2.5_f64).exp()) / (1.0 - (-10.0_f64).exp());
        assert_eq!(output[0], 0.0);
        assert!((f64::from(output[1]) - expected).abs() < 1.0e-6);
        assert_eq!(output[2], 1.0);
    }

    #[test]
    fn ghs_protection_boundaries_are_continuous_and_monotonic() {
        let data = (0..=1_000)
            .map(|index| index as f32 / 1_000.0)
            .collect::<Vec<_>>();
        let analysis = StretchAnalysis::analyze(&data, 1, data.len()).unwrap();
        let config = StretchConfig {
            model: StretchModel::Ghs(GhsParams {
                stretch_factor: 4.0,
                local_intensity: -1.0,
                symmetry_point: 0.35,
                protect_shadows: 0.1,
                protect_highlights: 0.8,
                black: 0.0,
                white: 1.0,
            }),
            color_strategy: ColorStrategy::Linked,
            max_analysis_samples: data.len(),
        };
        let output = config
            .resolve(&analysis)
            .unwrap()
            .apply_f32(&data, 1)
            .unwrap();
        assert!(output.windows(2).all(|pair| pair[0] <= pair[1]));
        for boundary in [100, 350, 800] {
            assert!((output[boundary + 1] - output[boundary - 1]).abs() < 0.02);
        }
    }

    #[test]
    fn every_ghs_subtype_is_finite_and_monotonic() {
        let data = (0..=1_000)
            .map(|index| index as f32 / 1_000.0)
            .collect::<Vec<_>>();
        let analysis = StretchAnalysis::analyze(&data, 1, data.len()).unwrap();
        for local_intensity in [-5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 15.0] {
            let config = StretchConfig {
                model: StretchModel::Ghs(GhsParams {
                    stretch_factor: 8.0,
                    local_intensity,
                    symmetry_point: 0.4,
                    protect_shadows: 0.1,
                    protect_highlights: 0.9,
                    black: 0.0,
                    white: 1.0,
                }),
                color_strategy: ColorStrategy::Linked,
                max_analysis_samples: data.len(),
            };
            let output = config
                .resolve(&analysis)
                .unwrap()
                .apply_f32(&data, 1)
                .unwrap();
            assert_eq!(output[0], 0.0, "b={local_intensity}");
            assert_eq!(*output.last().unwrap(), 1.0, "b={local_intensity}");
            assert!(
                output
                    .windows(2)
                    .all(|pair| pair[0].is_finite() && pair[0] <= pair[1]),
                "b={local_intensity}"
            );
        }
    }

    #[test]
    fn zero_strength_ghs_retains_the_explicit_input_range() {
        let config = StretchConfig {
            model: StretchModel::Ghs(GhsParams {
                stretch_factor: 0.0,
                local_intensity: 0.0,
                symmetry_point: 0.5,
                protect_shadows: 0.0,
                protect_highlights: 1.0,
                black: 0.25,
                white: 0.75,
            }),
            color_strategy: ColorStrategy::Linked,
            max_analysis_samples: 3,
        };
        let output = config
            .resolve_explicit(1)
            .unwrap()
            .apply_f32(&[0.25, 0.5, 0.75], 1)
            .unwrap();
        assert_eq!(output, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn explicit_models_resolve_without_image_analysis() {
        let config = StretchConfig {
            model: StretchModel::Linear {
                black: 0.0,
                white: 1.0,
            },
            color_strategy: ColorStrategy::Linked,
            max_analysis_samples: 0,
        };
        let plan = config.resolve_explicit(1).unwrap();
        assert_eq!(plan.apply_f32(&[f32::NAN, 0.5], 1).unwrap(), [0.0, 0.5]);

        let automatic = StretchConfig::auto_mtf(StretchParams::default(), 100);
        assert!(automatic.resolve_explicit(1).is_err());
    }

    #[test]
    fn resolved_ghs_plan_is_serializable_for_pipeline_provenance() {
        let analysis = StretchAnalysis::analyze(&[0.0, 0.5, 1.0], 1, 3).unwrap();
        let config = StretchConfig {
            model: StretchModel::Ghs(GhsParams {
                stretch_factor: 2.0,
                local_intensity: 1.0,
                symmetry_point: 0.25,
                protect_shadows: 0.0,
                protect_highlights: 1.0,
                black: 0.0,
                white: 1.0,
            }),
            color_strategy: ColorStrategy::Linked,
            max_analysis_samples: 3,
        };
        let plan = config.resolve(&analysis).unwrap();
        let json = serde_json::to_value(plan).unwrap();
        assert_eq!(json["curves"][0]["type"], "ghs");
    }

    #[test]
    fn auto_mtf_places_the_median_at_the_target() {
        let data = (0..1_001)
            .map(|index| 0.1 + index as f32 * 0.0001)
            .collect::<Vec<_>>();
        let analysis = StretchAnalysis::analyze(&data, 1, data.len()).unwrap();
        let config = StretchConfig::auto_mtf(StretchParams::default(), data.len());
        let plan = config.resolve(&analysis).unwrap();
        let median = analysis.linked_statistics().median as f32;
        assert!((plan.curves()[0].map(median) - 0.2).abs() < 1.0e-5);
    }

    #[test]
    fn exact_u16_statistics_and_lut_retain_existing_behavior() {
        let mut data = vec![600u16; 100_000];
        for (index, value) in data.iter_mut().enumerate() {
            *value += ((index * 37) % 41) as u16;
        }
        data[0] = 60_000;
        data[1] = 55_000;
        let statistics = statistics_u16(&data);
        let output = stretch_u16_to_u8(&data, &statistics, &StretchParams::default());
        let mut sorted = output.clone();
        sorted.sort_unstable();
        assert!((f64::from(sorted[sorted.len() / 2]) - 0.2 * 255.0).abs() < 16.0);
        assert!(output[0] > 200);
        assert!(sorted[100] < 60);
    }

    #[test]
    fn invalid_models_fail_during_resolution() {
        let analysis = StretchAnalysis::analyze(&[0.0, 1.0], 1, 2).unwrap();
        let config = StretchConfig {
            model: StretchModel::PercentileAsinh {
                black_percentile: 0.9,
                white_percentile: 0.1,
                strength: 0.0,
            },
            color_strategy: ColorStrategy::Linked,
            max_analysis_samples: 2,
        };
        assert!(config.resolve(&analysis).is_err());
    }
}
