//! Reusable parameterized display stretching for astrophotography pipelines.
//!
//! Requested [`StretchModel`] values are resolved against a reusable
//! [`StretchAnalysis`] into deterministic [`StretchPlan`] values. This keeps
//! image-dependent parameter selection separate from curve application and
//! permits interactive callers to reuse analysis across parameter changes.
//!
//! # Threads
//!
//! Analysis and curve application split large images across the Rayon pool
//! of the calling thread, or the global pool when that thread belongs to
//! none. To hold them to a number of cores, call them inside a pool that
//! size, as in `pool.install(|| plan.apply_u16(&samples, 3))`. The crate
//! starts no threads of its own.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use seiza_stats::NORMAL_MAD_SCALE;

mod layered;
pub use layered::{LayeredParams, ResolvedLayered};

const LUMA_RED: f64 = 0.2126;
const LUMA_GREEN: f64 = 0.7152;
const LUMA_BLUE: f64 = 0.0722;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum StretchError {
    #[error("{0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, StretchError>;

/// Default lower percentile used to map physical linear samples into the
/// unit-linear domain expected by display stretches.
pub const DEFAULT_SAMPLE_BLACK_PERCENTILE: f64 = 0.001;
/// Default upper percentile used to map physical linear samples into the
/// unit-linear domain expected by display stretches.
pub const DEFAULT_SAMPLE_WHITE_PERCENTILE: f64 = 0.999;
/// Default scalar-sample budget used while resolving a robust sample domain.
pub const DEFAULT_SAMPLE_ANALYSIS_SAMPLES: usize = 200_000;

/// Numeric domain of linear samples offered to a display-stretch pipeline.
///
/// Stretch curves operate in a unit-linear domain. Callers whose pixels are
/// already expressed that way can use [`Self::UnitLinear`]. Physical camera or
/// calibrated samples retain their scientific units until a caller explicitly
/// selects [`Self::PhysicalLinear`], which resolves one linked range across all
/// channels and maps that range to zero and one.
///
/// This is deliberately independent of whether an image is linear-light or
/// display-referred. It describes numeric scale, not transfer semantics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
pub enum SampleDomain {
    /// Samples are already in the unit-linear domain expected by stretch
    /// curves. Resolving and applying this domain is an identity operation.
    #[default]
    UnitLinear,
    /// Samples retain physical linear values and must be mapped to unit-linear
    /// before stretching.
    PhysicalLinear {
        #[serde(default)]
        normalization: SampleNormalization,
    },
}

/// How a physical linear sample domain is resolved into unit-linear values.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
pub enum SampleNormalization {
    /// Resolve a linked black and white point from a bounded, pixel-aware
    /// sample. The same range is used for every channel, preserving channel
    /// relationships without introducing an implicit per-channel balance.
    RobustPercentile {
        #[serde(default = "default_sample_black_percentile")]
        black_percentile: f64,
        #[serde(default = "default_sample_white_percentile")]
        white_percentile: f64,
        #[serde(default = "default_sample_analysis_samples")]
        max_analysis_samples: usize,
    },
    /// Use a caller-supplied physical range. Values outside the range are not
    /// clamped here; the selected stretch curve decides how to handle them.
    ExplicitRange { black: f32, white: f32 },
}

impl Default for SampleNormalization {
    fn default() -> Self {
        Self::RobustPercentile {
            black_percentile: DEFAULT_SAMPLE_BLACK_PERCENTILE,
            white_percentile: DEFAULT_SAMPLE_WHITE_PERCENTILE,
            max_analysis_samples: DEFAULT_SAMPLE_ANALYSIS_SAMPLES,
        }
    }
}

const fn default_sample_black_percentile() -> f64 {
    DEFAULT_SAMPLE_BLACK_PERCENTILE
}

const fn default_sample_white_percentile() -> f64 {
    DEFAULT_SAMPLE_WHITE_PERCENTILE
}

const fn default_sample_analysis_samples() -> usize {
    DEFAULT_SAMPLE_ANALYSIS_SAMPLES
}

/// Concrete mapping from an input sample domain to unit-linear values.
///
/// This value is serializable for render metadata and can be persisted by a
/// presentation client that wants to lock an automatically resolved range for
/// later frames without changing the underlying physical stack.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
pub enum ResolvedSampleDomain {
    /// Identity mapping for samples already expressed as unit-linear values.
    UnitLinear,
    /// Affine physical-to-unit mapping. `black` maps to zero and `white` maps
    /// to one; finite outliers remain outside that interval until stretching.
    PhysicalLinear { black: f32, white: f32 },
}

impl SampleDomain {
    /// Resolve this domain for mono or interleaved multi-channel samples.
    ///
    /// Robust resolution samples complete pixels, so bounded RGB analysis
    /// cannot accidentally observe only one color plane. Non-finite values are
    /// excluded from range selection.
    pub fn resolve(&self, data: &[f32], channel_count: usize) -> Result<ResolvedSampleDomain> {
        validate_sample_buffer(data, channel_count)?;
        match *self {
            Self::UnitLinear => Ok(ResolvedSampleDomain::UnitLinear),
            Self::PhysicalLinear { normalization } => match normalization {
                SampleNormalization::ExplicitRange { black, white } => {
                    validate_sample_range(black, white)?;
                    Ok(ResolvedSampleDomain::PhysicalLinear { black, white })
                }
                SampleNormalization::RobustPercentile {
                    black_percentile,
                    white_percentile,
                    max_analysis_samples,
                } => {
                    validate_sample_percentiles(black_percentile, white_percentile)?;
                    if max_analysis_samples < channel_count {
                        return Err(StretchError::Invalid(
                            "maximum sample-domain analysis samples must accommodate one complete pixel"
                                .into(),
                        ));
                    }
                    let (black, white) = robust_sample_range(
                        data,
                        channel_count,
                        black_percentile,
                        white_percentile,
                        max_analysis_samples,
                    )?;
                    Ok(ResolvedSampleDomain::PhysicalLinear { black, white })
                }
            },
        }
    }
}

impl ResolvedSampleDomain {
    /// The input values that map to zero and one, respectively.
    pub fn input_range(self) -> (f32, f32) {
        match self {
            Self::UnitLinear => (0.0, 1.0),
            Self::PhysicalLinear { black, white } => (black, white),
        }
    }

    /// Copy samples and apply this mapping to the copy.
    pub fn apply(self, data: &[f32], channel_count: usize) -> Result<Vec<f32>> {
        let mut mapped = data.to_vec();
        self.apply_in_place(&mut mapped, channel_count)?;
        Ok(mapped)
    }

    /// Apply this mapping in place while preserving every non-finite sample.
    ///
    /// The physical mapping is affine and intentionally does not clamp. This
    /// keeps highlights and shadows available to the subsequent stretch and
    /// leaves `NaN` coverage masks intact.
    pub fn apply_in_place(self, data: &mut [f32], channel_count: usize) -> Result<()> {
        validate_sample_buffer(data, channel_count)?;
        let Self::PhysicalLinear { black, white } = self else {
            return Ok(());
        };
        let span = validate_sample_range(black, white)?;
        data.par_iter_mut().for_each(|value| {
            if value.is_finite() {
                *value = (*value - black) / span;
            }
        });
        Ok(())
    }
}

fn validate_sample_buffer(data: &[f32], channel_count: usize) -> Result<usize> {
    if channel_count == 0 || data.is_empty() || !data.len().is_multiple_of(channel_count) {
        return Err(StretchError::Invalid(
            "sample-domain input must contain non-empty complete interleaved pixels".into(),
        ));
    }
    Ok(data.len() / channel_count)
}

fn validate_sample_percentiles(black: f64, white: f64) -> Result<()> {
    if !black.is_finite()
        || !white.is_finite()
        || !(0.0..=1.0).contains(&black)
        || !(0.0..=1.0).contains(&white)
        || white <= black
    {
        return Err(StretchError::Invalid(
            "sample-domain percentiles must be finite, within zero and one, and increasing".into(),
        ));
    }
    Ok(())
}

fn validate_sample_range(black: f32, white: f32) -> Result<f32> {
    let span = white - black;
    if !black.is_finite() || !white.is_finite() || !span.is_finite() || span <= 0.0 {
        return Err(StretchError::Invalid(
            "sample-domain black and white points must be finite and separated".into(),
        ));
    }
    Ok(span)
}

fn robust_sample_range(
    data: &[f32],
    channel_count: usize,
    black_percentile: f64,
    white_percentile: f64,
    max_analysis_samples: usize,
) -> Result<(f32, f32)> {
    let pixel_count = data.len() / channel_count;
    let maximum_pixels = max_analysis_samples / channel_count;
    let pixel_stride = pixel_count.div_ceil(maximum_pixels).max(1);
    let capacity = maximum_pixels.min(pixel_count) * channel_count;
    let mut sample = Vec::with_capacity(capacity);
    for pixel in data.chunks_exact(channel_count).step_by(pixel_stride) {
        sample.extend(pixel.iter().copied().filter(|value| value.is_finite()));
    }
    if sample.len() < 2 {
        return Err(StretchError::Invalid(
            "physical sample domain contains too few finite samples".into(),
        ));
    }
    sample.sort_unstable_by(f32::total_cmp);
    let percentile = |value: f64| {
        let index = ((sample.len() - 1) as f64 * value).round() as usize;
        sample[index]
    };
    let black = percentile(black_percentile);
    let white = percentile(white_percentile);
    validate_sample_range(black, white)?;
    Ok((black, white))
}

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

/// Parameters for the median anchor stretch: a closed-form midtones-family curve
/// that places the background median exactly on a target.
///
/// The black point sits `black_point_sigma` robust sigmas below the median,
/// the noise taken from the lower half of the distribution so stars cannot
/// inflate it (`None` uses the image minimum and clips no black). Samples are
/// rescaled so the black point maps to 0 and 1 stays at 1, then
/// `S(r) = (M - 1)·T·r / (M·(T + r - 1) - T·r)` maps the rescaled median `M`
/// to the target `T` exactly, with `S(0) = 0` and `S(1) = 1`. Optional
/// steps follow in order: a curves boost that lifts the tones above the
/// target, a soft-knee highlight compression, and a rescale so the brightest
/// analysed sample reaches 1.
///
/// In JSON every field may be left out and takes its default; an explicit
/// `"black_point_sigma": null` uses the image minimum.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MedianAnchorParams {
    /// Where the background median lands, in `(0, 1)` (default 0.25).
    pub target_median: f64,
    /// Black point in robust sigmas below the median, above 0 (default
    /// 2.7); `None` uses the image minimum, as does a background with no
    /// measurable noise.
    pub black_point_sigma: Option<f64>,
    /// Curves boost strength in `[0, 0.9]`: lifts the quarter tones above
    /// the target while black, the target and white stay put (default 0).
    pub curves_boost: f64,
    /// Highlight compression in `[0, 1]` above `hdr_knee` (default 0).
    pub hdr_amount: f64,
    /// Where highlight compression starts, in `(0, 1)` (default 0.75).
    pub hdr_knee: f64,
    /// Rescale so the brightest analysed sample reaches 1 (default off).
    pub normalize: bool,
}

impl Default for MedianAnchorParams {
    fn default() -> Self {
        Self {
            target_median: 0.25,
            black_point_sigma: Some(2.7),
            curves_boost: 0.0,
            hdr_amount: 0.0,
            hdr_knee: 0.75,
            normalize: false,
        }
    }
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
    /// Place the median exactly on a target with a closed-form curve, from
    /// a robust black point; see [`MedianAnchorParams`].
    MedianAnchor(MedianAnchorParams),
    /// Stretch luminance with the median anchor curve while treating large
    /// and fine structure apart; see [`LayeredParams`]. It needs the image's
    /// width and height, so it runs only through the `*_image` methods.
    Layered(LayeredParams),
}

/// Why a shape-less method refuses a layered stage.
const LAYERED_NEEDS_SHAPE: &str = "the layered stretch needs the image's width and height; \
     apply it with apply_f32_image, apply_u8_image or apply_u16_image";

/// The width, height and channel count of interleaved samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageShape {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
}

impl ImageShape {
    pub fn new(width: usize, height: usize, channels: usize) -> Self {
        Self {
            width,
            height,
            channels,
        }
    }

    fn validate(self, data: &[f32]) -> Result<()> {
        let samples = self
            .width
            .checked_mul(self.height)
            .and_then(|pixels| pixels.checked_mul(self.channels));
        if self.width == 0 || self.height == 0 || samples != Some(data.len()) {
            return Err(StretchError::Invalid(
                "image width, height and channels do not match the samples".into(),
            ));
        }
        Ok(())
    }
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

/// One or more display-stretch configurations applied in order.
///
/// Every stage resolves against the output of the previous stage. Intermediate
/// samples remain `f32`; conversion to an integer display format happens only
/// after the final stage. A stack is never empty.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "Vec<StretchConfig>", into = "Vec<StretchConfig>")]
pub struct StretchStack {
    stages: Vec<StretchConfig>,
}

/// Fine-grained lifecycle emitted while one stretch-stack stage is processed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StretchStageState {
    Resolving,
    Applying,
    Completed,
}

/// Progress for one stage in an ordered stretch stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StretchStageProgress {
    /// Zero-based index of the active stage.
    pub stage_index: usize,
    pub stage_count: usize,
    pub state: StretchStageState,
}

/// Output pixels and the resolved plan retained for every applied stage.
#[derive(Clone, Debug, PartialEq)]
pub struct StretchStackOutput<T> {
    pub data: Vec<T>,
    pub plans: Vec<StretchPlan>,
}

impl StretchStack {
    pub fn new(stages: Vec<StretchConfig>) -> Result<Self> {
        if stages.is_empty() {
            return Err(StretchError::Invalid(
                "stretch stack must contain at least one stage".into(),
            ));
        }
        Ok(Self { stages })
    }

    pub fn single(config: StretchConfig) -> Self {
        Self {
            stages: vec![config],
        }
    }

    pub fn stages(&self) -> &[StretchConfig] {
        &self.stages
    }

    pub fn len(&self) -> usize {
        self.stages.len()
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    /// Apply every stage and retain a display-referred `f32` result.
    pub fn apply_f32(&self, data: &[f32], channel_count: usize) -> Result<StretchStackOutput<f32>> {
        self.apply_f32_with_progress(data, channel_count, |_| {})
    }

    /// Apply every stage, reporting resolve/apply/completion boundaries.
    pub fn apply_f32_with_progress(
        &self,
        data: &[f32],
        channel_count: usize,
        progress: impl FnMut(StretchStageProgress),
    ) -> Result<StretchStackOutput<f32>> {
        apply_stack_f32(&self.stages, data, channel_count, None, progress)
    }

    /// Apply every stage and convert the final display result to `u8`.
    pub fn apply_u8(&self, data: &[f32], channel_count: usize) -> Result<StretchStackOutput<u8>> {
        self.apply_u8_with_progress(data, channel_count, |_| {})
    }

    /// Apply every stage, reporting resolve/apply/completion boundaries, and
    /// convert to `u8` only after the final stage.
    pub fn apply_u8_with_progress(
        &self,
        data: &[f32],
        channel_count: usize,
        mut progress: impl FnMut(StretchStageProgress),
    ) -> Result<StretchStackOutput<u8>> {
        let output = apply_stack_f32(&self.stages, data, channel_count, None, &mut progress)?;
        let data = output.data.into_par_iter().map(to_u8).collect::<Vec<_>>();
        Ok(StretchStackOutput {
            data,
            plans: output.plans,
        })
    }

    /// Apply every stage and convert the final display result to `u16`.
    pub fn apply_u16(&self, data: &[f32], channel_count: usize) -> Result<StretchStackOutput<u16>> {
        self.apply_u16_with_progress(data, channel_count, |_| {})
    }

    /// Apply every stage, reporting resolve/apply/completion boundaries, and
    /// convert to `u16` only after the final stage.
    pub fn apply_u16_with_progress(
        &self,
        data: &[f32],
        channel_count: usize,
        mut progress: impl FnMut(StretchStageProgress),
    ) -> Result<StretchStackOutput<u16>> {
        let output = apply_stack_f32(&self.stages, data, channel_count, None, &mut progress)?;
        let data = output.data.into_par_iter().map(to_u16).collect::<Vec<_>>();
        Ok(StretchStackOutput {
            data,
            plans: output.plans,
        })
    }

    /// Apply every stage to an image of a known shape, the layered stretch
    /// included, and retain a display-referred `f32` result. Other models
    /// give the same samples as [`Self::apply_f32`].
    pub fn apply_f32_image(
        &self,
        data: &[f32],
        shape: ImageShape,
    ) -> Result<StretchStackOutput<f32>> {
        self.apply_f32_image_with_progress(data, shape, |_| {})
    }

    /// [`Self::apply_f32_image`], reporting resolve/apply/completion
    /// boundaries.
    pub fn apply_f32_image_with_progress(
        &self,
        data: &[f32],
        shape: ImageShape,
        progress: impl FnMut(StretchStageProgress),
    ) -> Result<StretchStackOutput<f32>> {
        apply_stack_f32(&self.stages, data, shape.channels, Some(shape), progress)
    }

    /// [`Self::apply_f32_image`], converted to `u8` after the final stage.
    pub fn apply_u8_image(
        &self,
        data: &[f32],
        shape: ImageShape,
    ) -> Result<StretchStackOutput<u8>> {
        self.apply_u8_image_with_progress(data, shape, |_| {})
    }

    /// [`Self::apply_u8_image`], reporting resolve/apply/completion
    /// boundaries.
    pub fn apply_u8_image_with_progress(
        &self,
        data: &[f32],
        shape: ImageShape,
        mut progress: impl FnMut(StretchStageProgress),
    ) -> Result<StretchStackOutput<u8>> {
        let output = apply_stack_f32(
            &self.stages,
            data,
            shape.channels,
            Some(shape),
            &mut progress,
        )?;
        let data = output.data.into_par_iter().map(to_u8).collect::<Vec<_>>();
        Ok(StretchStackOutput {
            data,
            plans: output.plans,
        })
    }

    /// [`Self::apply_f32_image`], converted to `u16` after the final stage.
    pub fn apply_u16_image(
        &self,
        data: &[f32],
        shape: ImageShape,
    ) -> Result<StretchStackOutput<u16>> {
        self.apply_u16_image_with_progress(data, shape, |_| {})
    }

    /// [`Self::apply_u16_image`], reporting resolve/apply/completion
    /// boundaries.
    pub fn apply_u16_image_with_progress(
        &self,
        data: &[f32],
        shape: ImageShape,
        mut progress: impl FnMut(StretchStageProgress),
    ) -> Result<StretchStackOutput<u16>> {
        let output = apply_stack_f32(
            &self.stages,
            data,
            shape.channels,
            Some(shape),
            &mut progress,
        )?;
        let data = output.data.into_par_iter().map(to_u16).collect::<Vec<_>>();
        Ok(StretchStackOutput {
            data,
            plans: output.plans,
        })
    }
}

impl TryFrom<Vec<StretchConfig>> for StretchStack {
    type Error = StretchError;

    fn try_from(stages: Vec<StretchConfig>) -> Result<Self> {
        Self::new(stages)
    }
}

impl From<StretchStack> for Vec<StretchConfig> {
    fn from(stack: StretchStack) -> Self {
        stack.stages
    }
}

fn apply_stack_f32(
    stages: &[StretchConfig],
    data: &[f32],
    channel_count: usize,
    shape: Option<ImageShape>,
    mut progress: impl FnMut(StretchStageProgress),
) -> Result<StretchStackOutput<f32>> {
    match shape {
        Some(shape) => shape.validate(data)?,
        None => {
            if let Some(index) = stages
                .iter()
                .position(|config| matches!(config.model, StretchModel::Layered(_)))
            {
                return Err(StretchError::Invalid(format!(
                    "stretch stage {}/{}: {LAYERED_NEEDS_SHAPE}",
                    index + 1,
                    stages.len()
                )));
            }
        }
    }
    let mut current = data.to_vec();
    let mut plans = Vec::with_capacity(stages.len());
    for (stage_index, config) in stages.iter().enumerate() {
        let event = |state| StretchStageProgress {
            stage_index,
            stage_count: stages.len(),
            state,
        };
        progress(event(StretchStageState::Resolving));
        let plan = config
            .resolve_for(&current, channel_count)
            .map_err(|error| {
                StretchError::Invalid(format!(
                    "failed to resolve stretch stage {}/{}: {error}",
                    stage_index + 1,
                    stages.len()
                ))
            })?;
        progress(event(StretchStageState::Applying));
        let applied = match shape {
            Some(shape) => plan.apply_f32_image(&current, shape),
            None => plan.apply_f32(&current, channel_count),
        };
        current = applied.map_err(|error| {
            StretchError::Invalid(format!(
                "failed to apply stretch stage {}/{}: {error}",
                stage_index + 1,
                stages.len()
            ))
        })?;
        plans.push(plan);
        progress(event(StretchStageState::Completed));
    }
    Ok(StretchStackOutput {
        data: current,
        plans,
    })
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

    /// Resolve this request for an image, analyzing only data-driven models.
    pub fn resolve_for(&self, data: &[f32], channel_count: usize) -> Result<StretchPlan> {
        if self.model.requires_analysis() {
            let analysis = self.analyze(data, channel_count)?;
            self.resolve(&analysis)
        } else {
            self.resolve_explicit(channel_count)
        }
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
        if let StretchModel::Layered(params) = self.model {
            return self.resolve_layered(analysis, params);
        }
        let distributions = match self.color_strategy {
            ColorStrategy::Linked => vec![&analysis.linked],
            ColorStrategy::Unlinked if analysis.channel_count == 1 => vec![&analysis.linked],
            ColorStrategy::Unlinked => analysis
                .channels
                .iter()
                .enumerate()
                .map(|(index, channel)| {
                    channel.as_ref().ok_or_else(|| {
                        StretchError::Invalid(format!("channel {index} contains no finite samples"))
                    })
                })
                .collect::<Result<Vec<_>>>()?,
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

    /// The layered stretch works on luminance and carries colour itself, so
    /// it takes one curve, from the channel of a mono image or the luminance
    /// of an RGB one.
    fn resolve_layered(
        &self,
        analysis: &StretchAnalysis,
        params: LayeredParams,
    ) -> Result<StretchPlan> {
        if self.color_strategy == ColorStrategy::Unlinked {
            return Err(StretchError::Invalid(
                "the layered stretch carries colour by luminance; use the linked or \
                 luminance-preserving color strategy"
                    .into(),
            ));
        }
        let distribution = match analysis.channel_count {
            1 => &analysis.linked,
            3 => analysis.luminance.as_ref().ok_or_else(|| {
                StretchError::Invalid("luminance contains no finite samples".into())
            })?,
            _ => {
                return Err(StretchError::Invalid(
                    "the layered stretch takes mono or RGB images".into(),
                ));
            }
        };
        let curve = ResolvedCurve::Layered(ResolvedLayered::new(
            distribution.statistics,
            distribution.lower_half_noise(),
            params,
        )?);
        StretchPlan::from_resolved(analysis.channel_count, self.color_strategy, vec![curve])
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
        let median = values[values.len() / 2];
        let mut deviations = values
            .iter()
            .map(|value| (*value - median).abs())
            .collect::<Vec<_>>();
        deviations.sort_unstable_by(f32::total_cmp);
        let statistics = RobustStatistics {
            min: f64::from(values[0]),
            max: f64::from(*values.last().expect("non-empty sample")),
            median: f64::from(median),
            mad: f64::from(deviations[deviations.len() / 2]),
            count: values.len(),
        };
        Ok(Self {
            sorted: values,
            statistics,
        })
    }

    /// Robust noise from the samples at or below the median: the normal
    /// scale times their median distance below the median, which stars and
    /// bright structure cannot inflate. For Gaussian noise that median
    /// distance is the same 0.6745 sigma as a two-sided MAD.
    fn lower_half_noise(&self) -> f64 {
        let lower = &self.sorted[..=self.sorted.len() / 2];
        let median = lower[lower.len() - 1];
        f64::from(median - lower[lower.len() / 2]) * NORMAL_MAD_SCALE
    }

    fn percentile(&self, percentile: f64) -> f32 {
        let index =
            ((self.sorted.len() as f64 * percentile).floor() as usize).min(self.sorted.len() - 1);
        self.sorted[index]
    }
}

/// Bounded analysis that can be reused while interactive parameters change.
#[derive(Clone, Debug)]
pub struct StretchAnalysis {
    channel_count: usize,
    linked: Distribution,
    channels: Vec<Option<Distribution>>,
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
        let mut channels = if channel_count == 1 {
            Vec::new()
        } else {
            (0..channel_count)
                .map(|_| Vec::with_capacity(maximum_pixels.min(pixel_count)))
                .collect::<Vec<_>>()
        };
        let mut luminance =
            (channel_count == 3).then(|| Vec::with_capacity(maximum_pixels.min(pixel_count)));

        for pixel in data.chunks_exact(channel_count).step_by(pixel_stride) {
            for (channel, value) in pixel.iter().copied().enumerate() {
                if value.is_finite() {
                    linked.push(value);
                    if let Some(samples) = channels.get_mut(channel) {
                        samples.push(value);
                    }
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
                .map(|(index, values)| {
                    (!values.is_empty())
                        .then(|| Distribution::new(values, &format!("channel {index}")))
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?,
            luminance: luminance
                .and_then(|values| {
                    (!values.is_empty()).then(|| Distribution::new(values, "luminance"))
                })
                .transpose()?,
        })
    }

    pub fn channel_count(&self) -> usize {
        self.channel_count
    }

    pub fn linked_statistics(&self) -> RobustStatistics {
        self.linked.statistics
    }

    pub fn channel_statistics(&self) -> Vec<Option<RobustStatistics>> {
        if self.channel_count == 1 {
            return vec![Some(self.linked.statistics)];
        }
        self.channels
            .iter()
            .map(|channel| channel.as_ref().map(|channel| channel.statistics))
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
        black: f32,
        white: f32,
        strength: f32,
    },
    Mtf {
        shadows: f32,
        midtone: f64,
        highlights: f32,
    },
    Ghs(ResolvedGhs),
    MedianAnchor(ResolvedMedianAnchor),
    /// A layered stretch. As a per-sample curve it maps only through its
    /// global curve; the full stretch needs the image, through
    /// [`StretchPlan::apply_f32_image`] and its twins.
    Layered(ResolvedLayered),
}

/// A median anchor stretch with its image-dependent values resolved.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ResolvedMedianAnchor {
    /// Input black point.
    pub black: f32,
    /// The median after the black point is taken off, in `(0, 1)`.
    pub median: f64,
    /// Where the median lands.
    pub target: f64,
    /// Curves boost knots from the target up, as `(x, y)`, or none.
    pub boost: Option<[(f64, f64); 3]>,
    /// Highlight compression knee and the curve's end slope, or none.
    pub highlights: Option<(f64, f64)>,
    /// Factor that brings the brightest analysed sample to 1, else 1.
    pub scale: f64,
}

impl ResolvedMedianAnchor {
    fn new(statistics: RobustStatistics, noise: f64, params: MedianAnchorParams) -> Result<Self> {
        validate_median_anchor(params)?;
        let median = statistics.median;
        let black = match params.black_point_sigma {
            // A background with no measurable noise would put the black point
            // on the median, which would then map to 0; the minimum is used.
            Some(sigma) if noise > 0.0 => (median - sigma * noise).max(statistics.min),
            _ => statistics.min,
        }
        .min(0.99);
        // A median at the minimum leaves nothing to place: the rescaled
        // median is held just inside the curve's domain.
        let rescaled = ((median - black) / (1.0 - black)).clamp(1e-6, 1.0 - 1e-6);
        let boost = (params.curves_boost > 0.0).then(|| {
            let target = params.target_median;
            let lift = 1.0 - params.curves_boost;
            let quarter = 0.25 * (1.0 - target) + target;
            let three_quarters = 0.75 * (1.0 - target) + target;
            [
                (target, target),
                (quarter, quarter.powf(lift)),
                (three_quarters, three_quarters.powf(lift * lift)),
            ]
        });
        let highlights = (params.hdr_amount > 0.0).then_some((
            params.hdr_knee,
            1.0 + HDR_MAX_EXTRA_SLOPE * params.hdr_amount,
        ));
        let mut resolved = Self {
            black: black as f32,
            median: rescaled,
            target: params.target_median,
            boost,
            highlights,
            scale: 1.0,
        };
        if params.normalize {
            let brightest = resolved.map(statistics.max);
            if brightest > f64::EPSILON {
                resolved.scale = 1.0 / brightest;
            }
        }
        Ok(resolved)
    }

    fn map(&self, value: f64) -> f64 {
        let black = f64::from(self.black);
        let rescaled = ((value - black) / (1.0 - black)).clamp(0.0, 1.0);
        let (m, t) = (self.median, self.target);
        let denominator = m * (t + rescaled - 1.0) - t * rescaled;
        let mut out = if denominator.abs() <= f64::EPSILON {
            rescaled
        } else {
            (m - 1.0) * t * rescaled / denominator
        };
        if let Some(knots) = self.boost {
            out = boost_curve(knots, out);
        }
        if let Some((knee, end_slope)) = self.highlights {
            out = compress_highlights(out, knee, end_slope);
        }
        out * self.scale
    }
}

/// The most a full highlight compression steepens the curve's end above 1:
/// a cubic Hermite from slope 1 to slope `1 + 1.8` stays monotonic.
const HDR_MAX_EXTRA_SLOPE: f64 = 1.8;

/// Piecewise-linear lift of the tones above the target, through `knots` and
/// on to `(1, 1)`; below the target the curve is unchanged.
fn boost_curve(knots: [(f64, f64); 3], value: f64) -> f64 {
    if value <= knots[0].0 {
        return value;
    }
    let points = [knots[0], knots[1], knots[2], (1.0, 1.0)];
    for pair in points.windows(2) {
        let ((x0, y0), (x1, y1)) = (pair[0], pair[1]);
        if value <= x1 {
            let share = if x1 > x0 {
                (value - x0) / (x1 - x0)
            } else {
                1.0
            };
            return y0 + share * (y1 - y0);
        }
    }
    1.0
}

/// Soft-knee highlight compression: unchanged up to `knee`, then a cubic
/// Hermite from slope 1 at the knee to `end_slope` at 1, which keeps 1 at 1
/// and lowers what lies between.
fn compress_highlights(value: f64, knee: f64, end_slope: f64) -> f64 {
    if value <= knee {
        return value;
    }
    let width = 1.0 - knee;
    let t = ((value - knee) / width).min(1.0);
    let (t2, t3) = (t * t, t * t * t);
    let shape = (t3 - 2.0 * t2 + t) + (-2.0 * t3 + 3.0 * t2) + end_slope * (t3 - t2);
    knee + width * shape
}

fn validate_median_anchor(params: MedianAnchorParams) -> Result<()> {
    let open_unit = |value: f64| value.is_finite() && value > 0.0 && value < 1.0;
    if !open_unit(params.target_median) {
        return Err(StretchError::Invalid(
            "target median must be finite and between zero and one".into(),
        ));
    }
    if let Some(sigma) = params.black_point_sigma
        && (!sigma.is_finite() || sigma <= 0.0)
    {
        return Err(StretchError::Invalid(
            "black point sigma must be finite and positive".into(),
        ));
    }
    if !params.curves_boost.is_finite() || !(0.0..=0.9).contains(&params.curves_boost) {
        return Err(StretchError::Invalid(
            "curves boost must be between 0 and 0.9".into(),
        ));
    }
    if !params.hdr_amount.is_finite() || !(0.0..=1.0).contains(&params.hdr_amount) {
        return Err(StretchError::Invalid(
            "highlight compression must be between 0 and 1".into(),
        ));
    }
    if !open_unit(params.hdr_knee) {
        return Err(StretchError::Invalid(
            "highlight knee must be between zero and one".into(),
        ));
    }
    Ok(())
}

/// Precomputed GHS transfer state used by a resolved plan.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ResolvedGhs {
    params: GhsParams,
    #[serde(skip)]
    d: f64,
    #[serde(skip)]
    low_value: f64,
    #[serde(skip)]
    low_slope: f64,
    #[serde(skip)]
    high_value: f64,
    #[serde(skip)]
    high_slope: f64,
    #[serde(skip)]
    normalization_min: f64,
    #[serde(skip)]
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

    #[inline(always)]
    fn map(&self, value: f64) -> f64 {
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
        CurveMapper::new(self).map(value)
    }
}

/// A [`ResolvedCurve`] ready to map many samples: `asinh(strength)`, which
/// normalizes the asinh curve and depends only on the curve, is worked out
/// once instead of once per sample. It is the same value either way, so
/// every sample maps to the same bits.
#[derive(Clone, Copy, Debug)]
struct CurveMapper {
    curve: ResolvedCurve,
    /// `asinh(strength)` for an asinh curve, unused otherwise.
    asinh_strength: f32,
}

impl CurveMapper {
    fn new(curve: ResolvedCurve) -> Self {
        let asinh_strength = match curve {
            ResolvedCurve::Asinh { strength, .. } => strength.asinh(),
            _ => 0.0,
        };
        Self {
            curve,
            asinh_strength,
        }
    }

    #[inline(always)]
    fn map(self, value: f32) -> f32 {
        if !value.is_finite() {
            return 0.0;
        }
        let mapped = match self.curve {
            ResolvedCurve::Identity => f64::from(value),
            ResolvedCurve::Linear { black, white } => (f64::from(value) - black) / (white - black),
            ResolvedCurve::Asinh {
                black,
                white,
                strength,
            } => {
                let linear = ((value - black) / (white - black)).max(0.0);
                f64::from((strength * linear).asinh() / self.asinh_strength)
            }
            ResolvedCurve::Mtf {
                shadows,
                midtone,
                highlights,
            } => {
                let input = 1.0 - f64::from(highlights) + f64::from(value - shadows);
                midtones_transfer_function(midtone, input)
            }
            ResolvedCurve::Ghs(ghs) => ghs.map(f64::from(value)),
            ResolvedCurve::MedianAnchor(anchor) => anchor.map(f64::from(value)),
            ResolvedCurve::Layered(layered) => layered.curve.map(f64::from(value)),
        };
        mapped.clamp(0.0, 1.0) as f32
    }

    /// Map `input` into `output` through `convert`, with one loop per kind
    /// of curve so that no sample branches on the kind.
    fn map_into<T, F>(self, input: &[f32], output: &mut [T], convert: F)
    where
        F: Fn(f32) -> T + Copy,
    {
        #[inline(always)]
        fn run<T>(input: &[f32], output: &mut [T], map: impl Fn(f32) -> T) {
            for (output, &value) in output.iter_mut().zip(input) {
                *output = map(value);
            }
        }
        // Each arm repeats the call, but the curve's kind is known inside it,
        // so the inlined `map` keeps only that kind's operations.
        match self.curve {
            ResolvedCurve::Identity => run(input, output, |value| convert(self.map(value))),
            ResolvedCurve::Linear { .. } => run(input, output, |value| convert(self.map(value))),
            ResolvedCurve::Asinh { .. } => run(input, output, |value| convert(self.map(value))),
            ResolvedCurve::Mtf { .. } => run(input, output, |value| convert(self.map(value))),
            ResolvedCurve::Ghs(_) => run(input, output, |value| convert(self.map(value))),
            ResolvedCurve::MedianAnchor(_) => run(input, output, |value| convert(self.map(value))),
            ResolvedCurve::Layered(_) => run(input, output, |value| convert(self.map(value))),
        }
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
        self.apply_mapped(data, channel_count, std::convert::identity)
    }

    pub fn apply_u8(&self, data: &[f32], channel_count: usize) -> Result<Vec<u8>> {
        self.apply_mapped(data, channel_count, to_u8)
    }

    pub fn apply_u16(&self, data: &[f32], channel_count: usize) -> Result<Vec<u16>> {
        self.apply_mapped(data, channel_count, to_u16)
    }

    /// Apply this plan to an image of a known shape, which a layered plan
    /// needs. Other plans give the same samples as [`Self::apply_f32`].
    pub fn apply_f32_image(&self, data: &[f32], shape: ImageShape) -> Result<Vec<f32>> {
        self.apply_image(data, shape, std::convert::identity)
    }

    /// [`Self::apply_f32_image`], converted to `u8`.
    pub fn apply_u8_image(&self, data: &[f32], shape: ImageShape) -> Result<Vec<u8>> {
        self.apply_image(data, shape, to_u8)
    }

    /// [`Self::apply_f32_image`], converted to `u16`.
    pub fn apply_u16_image(&self, data: &[f32], shape: ImageShape) -> Result<Vec<u16>> {
        self.apply_image(data, shape, to_u16)
    }

    fn apply_image<T, F>(&self, data: &[f32], shape: ImageShape, convert: F) -> Result<Vec<T>>
    where
        T: Copy + Default + Send,
        F: Fn(f32) -> T + Copy + Send + Sync,
    {
        shape.validate(data)?;
        let ResolvedCurve::Layered(layered) = self.curves[0] else {
            return self.apply_mapped(data, shape.channels, convert);
        };
        self.validate_input(data, shape.channels)?;
        let stretched = layered::apply(&layered, data, shape);
        Ok(stretched.into_par_iter().map(convert).collect())
    }

    fn apply_mapped<T, F>(&self, data: &[f32], channel_count: usize, convert: F) -> Result<Vec<T>>
    where
        T: Copy + Default + Send,
        F: Fn(f32) -> T + Copy + Send + Sync,
    {
        self.validate_input(data, channel_count)?;
        if matches!(self.curves[0], ResolvedCurve::Layered(_)) {
            return Err(StretchError::Invalid(LAYERED_NEEDS_SHAPE.into()));
        }
        let mappers = self
            .curves
            .iter()
            .copied()
            .map(CurveMapper::new)
            .collect::<Vec<_>>();
        let mut output = vec![T::default(); data.len()];
        // Whole pixels per task, so each chunk starts on a first channel.
        let chunk = channel_count * MAP_CHUNK_PIXELS;
        let chunks = output.par_chunks_mut(chunk).zip(data.par_chunks(chunk));
        match self.color_strategy {
            ColorStrategy::Linked => {
                chunks.for_each(|(output, input)| mappers[0].map_into(input, output, convert))
            }
            ColorStrategy::Unlinked => chunks.for_each(|(output, input)| {
                for (output, input) in output
                    .chunks_exact_mut(channel_count)
                    .zip(input.chunks_exact(channel_count))
                {
                    for ((output, &value), mapper) in output.iter_mut().zip(input).zip(&mappers) {
                        *output = convert(mapper.map(value));
                    }
                }
            }),
            ColorStrategy::LuminancePreserving => chunks.for_each(|(output, input)| {
                for (output, pixel) in output.chunks_exact_mut(3).zip(input.chunks_exact(3)) {
                    let mapped = map_luminance_pixel(mappers[0], pixel);
                    for (output, value) in output.iter_mut().zip(mapped) {
                        *output = convert(value);
                    }
                }
            }),
        }
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
}

/// Pixels one rayon task maps in [`StretchPlan`]'s whole-image application.
const MAP_CHUNK_PIXELS: usize = 1 << 14;

/// Stretch an RGB pixel's Rec.709 luminance and scale its channels by the
/// same factor, keeping them in gamut.
fn map_luminance_pixel(curve: CurveMapper, pixel: &[f32]) -> [f32; 3] {
    if pixel.iter().any(|value| !value.is_finite()) {
        return [0.0; 3];
    }
    let luminance = (LUMA_RED * f64::from(pixel[0])
        + LUMA_GREEN * f64::from(pixel[1])
        + LUMA_BLUE * f64::from(pixel[2])) as f32;
    let target = curve.map(luminance);
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

impl StretchModel {
    /// Whether this model needs an image analysis before it can be resolved.
    pub fn requires_analysis(&self) -> bool {
        matches!(
            self,
            Self::PercentileAsinh { .. }
                | Self::AutoMtf(_)
                | Self::MedianAnchor(_)
                | Self::Layered(_)
        )
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
                let white = distribution
                    .percentile(white_percentile)
                    .max(black + f32::EPSILON);
                ResolvedCurve::Asinh {
                    black,
                    white,
                    strength: strength as f32,
                }
            }
            Self::AutoMtf(params) => auto_mtf_curve(distribution.statistics, params)?,
            Self::MedianAnchor(params) => ResolvedCurve::MedianAnchor(ResolvedMedianAnchor::new(
                distribution.statistics,
                distribution.lower_half_noise(),
                params,
            )?),
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
                validate_f32_range(black, white)?;
                validate_strength(strength)?;
                ResolvedCurve::Asinh {
                    black: black as f32,
                    white: white as f32,
                    strength: strength as f32,
                }
            }
            Self::Mtf {
                shadows,
                midtone,
                highlights,
            } => {
                validate_mtf(shadows, midtone, highlights)?;
                ResolvedCurve::Mtf {
                    shadows: shadows as f32,
                    midtone,
                    highlights: highlights as f32,
                }
            }
            Self::Ghs(params) => {
                if params.stretch_factor == 0.0 {
                    validate_ghs(params)?;
                    ResolvedCurve::Linear {
                        black: params.black,
                        white: params.white,
                    }
                } else {
                    ResolvedCurve::Ghs(ResolvedGhs::new(params)?)
                }
            }
            Self::PercentileAsinh { .. }
            | Self::AutoMtf(_)
            | Self::MedianAnchor(_)
            | Self::Layered(_) => {
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
    if !params.shadows_clip.is_finite()
        || !(params.shadows_clip as f32).is_finite()
        || params.shadows_clip > 0.0
    {
        return Err(StretchError::Invalid(
            "shadows clip must be finite and non-positive".into(),
        ));
    }
    let median = statistics.median as f32;
    let normal_mad = statistics.mad as f32 * NORMAL_MAD_SCALE as f32;
    let shadows_clip = params.shadows_clip as f32;
    if normal_mad <= f32::EPSILON {
        return Ok(ResolvedCurve::Identity);
    }
    let (shadows, midtone, highlights) = if median > 0.5 {
        let highlights = median - shadows_clip * normal_mad;
        let midtone = midtones_transfer_function(
            params.target_median,
            f64::from(1.0 - (highlights - median)),
        );
        (0.0, midtone, highlights)
    } else {
        let shadows = (median + shadows_clip * normal_mad).max(0.0);
        let midtone = midtones_transfer_function(params.target_median, f64::from(median - shadows));
        (shadows, midtone, 1.0)
    };
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

fn validate_f32_range(black: f64, white: f64) -> Result<()> {
    validate_range(black, white)?;
    let black = black as f32;
    let white = white as f32;
    if !black.is_finite() || !white.is_finite() || white <= black {
        return Err(StretchError::Invalid(
            "stretch black and white points must remain finite and increasing at f32 precision"
                .into(),
        ));
    }
    Ok(())
}

fn validate_strength(strength: f64) -> Result<()> {
    let strength_f32 = strength as f32;
    if !strength.is_finite() || !strength_f32.is_finite() || strength_f32 <= 0.0 {
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
    let shadows_f32 = shadows as f32;
    let highlights_f32 = highlights as f32;
    if !shadows.is_finite()
        || !highlights.is_finite()
        || highlights <= shadows
        || !shadows_f32.is_finite()
        || !highlights_f32.is_finite()
        || highlights_f32 <= shadows_f32
        || !midtone.is_finite()
        || !(0.0..1.0).contains(&midtone)
    {
        return Err(StretchError::Invalid(
            "MTF shadows, midtone, and highlights must form a finite valid curve".into(),
        ));
    }
    Ok(())
}

// Quantizes like the legacy u16 LUT's `* 255.0 + 0.5` (identical for the
// clamped non-negative domain); the LUT keeps its own form because it is
// frozen for byte-for-byte preview compatibility.
//
// Both round half away from zero as `f32::round` does, without its libm call
// on x86-64's SSE2 baseline. The scaled value is never below zero, so the
// cast truncates it to its whole part, the subtraction leaves the fraction
// exactly, and a fraction of a half or more rounds up. NaN casts to 0 both
// ways.
#[inline(always)]
fn to_u8(value: f32) -> u8 {
    let scaled = value.clamp(0.0, 1.0) * 255.0;
    let whole = scaled as u8;
    whole + u8::from(scaled - f32::from(whole) >= 0.5)
}

#[inline(always)]
fn to_u16(value: f32) -> u16 {
    let scaled = value.clamp(0.0, 1.0) * f32::from(u16::MAX);
    let whole = scaled as u16;
    whole + u16::from(scaled - f32::from(whole) >= 0.5)
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

/// Samples one rayon task counts or maps in the `u16` statistics and LUTs.
const U16_CHUNK: usize = 1 << 18;

pub fn statistics_u16(data: &[u16]) -> Statistics {
    // Chunks count into histograms of their own, merged by adding counts.
    // The sums come from the merged histogram: every sample is in it once,
    // and integer sums do not depend on their order.
    let histogram = data
        .par_chunks(U16_CHUNK)
        .fold(
            || vec![0u32; 65_536],
            |mut histogram, chunk| {
                for &value in chunk {
                    histogram[usize::from(value)] += 1;
                }
                histogram
            },
        )
        .reduce(
            || vec![0u32; 65_536],
            |mut histogram, other| {
                for (count, other) in histogram.iter_mut().zip(&other) {
                    *count += other;
                }
                histogram
            },
        );
    let (sum, sum_sq) =
        histogram
            .iter()
            .enumerate()
            .fold((0u64, 0u128), |(sum, sum_sq), (value, &samples)| {
                let value = value as u64;
                (
                    sum + value * u64::from(samples),
                    sum_sq + u128::from(value * value) * u128::from(samples),
                )
            });
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
///
/// The 8-bit value is the stretched 16-bit value shifted right by 8 — the
/// same conversion N.I.N.A. applies when it renders a stretched frame at 8
/// bits, so `stretch_u16_to_u8` is always `stretch_u16_to_u16 >> 8`.
pub fn stretch_u16_to_u8(data: &[u16], stats: &Statistics, params: &StretchParams) -> Vec<u8> {
    let map = stretch_u16_map(stats, params);
    data.par_iter()
        .with_min_len(U16_CHUNK)
        .map(|value| map[usize::from(*value)])
        .collect()
}

/// Stretch `u16` data directly to `u16` through a 65,536-entry LUT, retaining
/// the full display precision of the resolved curve.
///
/// Reproduces N.I.N.A.'s `AutoStretch` bit for bit, including its
/// `DenormalizeUShort` rounding (round half up below mid-scale, truncate
/// above).
pub fn stretch_u16_to_u16(data: &[u16], stats: &Statistics, params: &StretchParams) -> Vec<u16> {
    let (shadows, midtone, highlights) = stretch_u16_curve(stats, params);
    let map = (0..65_536)
        .map(|value| {
            let input = 1.0 - highlights + value as f64 / 65_535.0 - shadows;
            let stretched = midtones_transfer_function(midtone, input);
            denormalize_u16(stretched)
        })
        .collect::<Vec<_>>();
    data.par_iter()
        .with_min_len(U16_CHUNK)
        .map(|value| map[usize::from(*value)])
        .collect()
}

fn stretch_u16_map(stats: &Statistics, params: &StretchParams) -> Vec<u8> {
    let (shadows, midtone, highlights) = stretch_u16_curve(stats, params);
    (0..65_536)
        .map(|value| {
            let input = 1.0 - highlights + value as f64 / 65_535.0 - shadows;
            let stretched = midtones_transfer_function(midtone, input);
            (denormalize_u16(stretched) >> 8) as u8
        })
        .collect()
}

/// N.I.N.A.'s `DenormalizeUShort`: `(ushort)(val * 65535 + (val < 0.5 ? 0.5
/// : 0.0))` — round half up below mid-scale, truncate above.
fn denormalize_u16(value: f64) -> u16 {
    let scaled = value.clamp(0.0, 1.0) * 65_535.0;
    let rounded = if scaled < 32_767.5 {
        scaled + 0.5
    } else {
        scaled
    };
    rounded as u16
}

/// The N.I.N.A. auto-stretch curve. Operation order matches N.I.N.A.'s
/// `ImageStatistics`-driven stretch exactly, and shadows are deliberately
/// NOT clamped at zero — on very dark frames N.I.N.A. lets the shadow point
/// go negative, which brightens the background, and reproducing its output
/// requires doing the same.
fn stretch_u16_curve(stats: &Statistics, params: &StretchParams) -> (f64, f64, f64) {
    let normalized_median = f64::from(stats.median) / 65_535.0;
    let normalized_mad = stats.mad / 65_535.0;
    if normalized_median > 0.5 {
        let highlights =
            normalized_median - params.shadows_clip * normalized_mad * NORMAL_MAD_SCALE;
        let midtone = midtones_transfer_function(
            params.target_median,
            1.0 - (highlights - normalized_median),
        );
        (0.0, midtone, highlights)
    } else {
        let shadows = normalized_median + params.shadows_clip * normalized_mad * NORMAL_MAD_SCALE;
        let midtone = midtones_transfer_function(params.target_median, normalized_median - shadows);
        (shadows, midtone, 1.0)
    }
}

/// Auto-stretch arbitrary linear floating-point samples to display `u8`
/// with the same N.I.N.A.-style MTF mapping as [`stretch_u16_to_u8`].
///
/// Derived images such as stacks have no meaningful fixed integer bit
/// depth. The median, MAD, and a robust highlight (99.9th percentile) are
/// estimated from a bounded sample, the data is normalized to that range,
/// and the MTF is applied. Returns `None` when the sample is empty or
/// degenerate (no usable range). Non-finite input pixels map to 0.
pub fn auto_stretch_f32_to_u8(data: &[f32], params: &StretchParams) -> Option<Vec<u8>> {
    let mut sample = data
        .iter()
        .step_by((data.len() / 300_000).max(1))
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if sample.is_empty() {
        return None;
    }
    sample.sort_unstable_by(f32::total_cmp);
    let median = f64::from(sample[sample.len() / 2]);
    let mut deviations = sample
        .iter()
        .map(|value| (f64::from(*value) - median).abs())
        .collect::<Vec<_>>();
    deviations.sort_unstable_by(f64::total_cmp);
    let mad = deviations[deviations.len() / 2];
    let shadows = median + params.shadows_clip * mad * NORMAL_MAD_SCALE;
    let highlights = f64::from(sample[sample.len() * 999 / 1000]);
    let range = highlights - shadows;
    if !range.is_finite() || range <= f64::EPSILON {
        return None;
    }
    let normalized_median = ((median - shadows) / range).clamp(0.0, 1.0);
    let midtones = midtones_transfer_function(params.target_median, normalized_median);
    Some(
        data.iter()
            .map(|value| {
                if !value.is_finite() {
                    return 0;
                }
                let normalized = ((f64::from(*value) - shadows) / range).clamp(0.0, 1.0);
                (midtones_transfer_function(midtones, normalized) * 255.0).round() as u8
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent transcription of N.I.N.A.'s AutoStretch (as ported and
    /// validated in PSF Guard): operation order, no shadow clamp, and
    /// DenormalizeUShort rounding.
    fn nina_reference_stretch_u16(
        data: &[u16],
        stats: &Statistics,
        params: &StretchParams,
    ) -> Vec<u16> {
        let normalized_median = f64::from(stats.median) / 65_535.0;
        let normalized_mad = stats.mad / 65_535.0;
        let scale_factor = 1.4826;
        let (shadows, midtone, highlights) = if normalized_median > 0.5 {
            let highlights =
                normalized_median - params.shadows_clip * normalized_mad * scale_factor;
            let midtone = midtones_transfer_function(
                params.target_median,
                1.0 - (highlights - normalized_median),
            );
            (0.0, midtone, highlights)
        } else {
            let shadows = normalized_median + params.shadows_clip * normalized_mad * scale_factor;
            let midtone =
                midtones_transfer_function(params.target_median, normalized_median - shadows);
            (shadows, midtone, 1.0)
        };
        let map = (0..65_536)
            .map(|value| {
                let input = 1.0 - highlights + value as f64 / 65_535.0 - shadows;
                let stretched = midtones_transfer_function(midtone, input);
                let scaled = stretched.clamp(0.0, 1.0) * 65_535.0;
                (if scaled < 32_767.5 {
                    scaled + 0.5
                } else {
                    scaled
                }) as u16
            })
            .collect::<Vec<_>>();
        data.iter().map(|value| map[usize::from(*value)]).collect()
    }

    #[test]
    fn sample_domain_defaults_are_opt_in_and_serde_friendly() {
        assert_eq!(SampleDomain::default(), SampleDomain::UnitLinear);
        assert_eq!(
            SampleNormalization::default(),
            SampleNormalization::RobustPercentile {
                black_percentile: 0.001,
                white_percentile: 0.999,
                max_analysis_samples: 200_000,
            }
        );

        let domain = serde_json::from_value::<SampleDomain>(serde_json::json!({
            "type": "physical-linear"
        }))
        .unwrap();
        assert_eq!(
            domain,
            SampleDomain::PhysicalLinear {
                normalization: SampleNormalization::default(),
            }
        );
        assert_eq!(
            serde_json::to_value(domain).unwrap(),
            serde_json::json!({
                "type": "physical-linear",
                "normalization": {
                    "type": "robust-percentile",
                    "black_percentile": 0.001,
                    "white_percentile": 0.999,
                    "max_analysis_samples": 200_000,
                }
            })
        );
    }

    #[test]
    fn unit_linear_domain_is_an_exact_identity_including_non_finite_values() {
        let input = [f32::NAN, f32::NEG_INFINITY, -1.0, 0.5, 2.0, f32::INFINITY];
        let resolved = SampleDomain::UnitLinear.resolve(&input, 1).unwrap();
        assert_eq!(resolved, ResolvedSampleDomain::UnitLinear);
        assert_eq!(resolved.input_range(), (0.0, 1.0));

        let output = resolved.apply(&input, 1).unwrap();
        assert!(output[0].is_nan());
        assert_eq!(&output[1..], &input[1..]);
    }

    #[test]
    fn explicit_physical_range_maps_without_clamping_and_preserves_nan() {
        let domain = SampleDomain::PhysicalLinear {
            normalization: SampleNormalization::ExplicitRange {
                black: 100.0,
                white: 1_100.0,
            },
        };
        let input = [0.0, 100.0, 600.0, 1_100.0, 1_200.0, f32::NAN];
        let resolved = domain.resolve(&input, 1).unwrap();
        assert_eq!(resolved.input_range(), (100.0, 1_100.0));
        let output = resolved.apply(&input, 1).unwrap();
        assert_eq!(&output[..5], &[-0.1, 0.0, 0.5, 1.0, 1.1]);
        assert!(output[5].is_nan());
        assert_eq!(
            serde_json::to_value(resolved).unwrap(),
            serde_json::json!({
                "type": "physical-linear",
                "black": 100.0,
                "white": 1_100.0,
            })
        );

        let tiny = SampleDomain::PhysicalLinear {
            normalization: SampleNormalization::ExplicitRange {
                black: 1.0e-9,
                white: 2.0e-9,
            },
        }
        .resolve(&[1.0e-9, 1.5e-9, 2.0e-9], 1)
        .unwrap();
        let tiny_output = tiny.apply(&[1.0e-9, 1.5e-9, 2.0e-9], 1).unwrap();
        assert_eq!((tiny_output[0], tiny_output[2]), (0.0, 1.0));
        assert!((tiny_output[1] - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn robust_domain_samples_complete_rgb_pixels_with_a_scalar_budget() {
        let data = (0..10)
            .flat_map(|pixel| {
                let base = pixel as f32 * 100.0;
                [base + 1.0, base + 2.0, base + 3.0]
            })
            .collect::<Vec<_>>();
        let domain = SampleDomain::PhysicalLinear {
            normalization: SampleNormalization::RobustPercentile {
                black_percentile: 0.0,
                white_percentile: 1.0,
                // Two complete RGB pixels. With ten pixels, the selected
                // pixels are zero and five; later extremes stay unsampled.
                max_analysis_samples: 6,
            },
        };
        assert_eq!(
            domain.resolve(&data, 3).unwrap(),
            ResolvedSampleDomain::PhysicalLinear {
                black: 1.0,
                white: 503.0,
            }
        );
    }

    #[test]
    fn robust_domain_uses_one_linked_range_and_ignores_nan_for_resolution() {
        let input = [f32::NAN, 100.0, 200.0, 300.0, 400.0, 500.0];
        let domain = SampleDomain::PhysicalLinear {
            normalization: SampleNormalization::RobustPercentile {
                black_percentile: 0.0,
                white_percentile: 1.0,
                max_analysis_samples: input.len(),
            },
        };
        let resolved = domain.resolve(&input, 3).unwrap();
        assert_eq!(
            resolved,
            ResolvedSampleDomain::PhysicalLinear {
                black: 100.0,
                white: 500.0,
            }
        );
        let output = resolved.apply(&input, 3).unwrap();
        assert!(output[0].is_nan());
        assert_eq!(&output[1..], &[0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn robust_domain_is_affine_scale_invariant() {
        let mut input = (0..1_000).map(|value| value as f32).collect::<Vec<_>>();
        input[17] = f32::NAN;
        let transformed = input
            .iter()
            .map(|value| {
                if value.is_finite() {
                    *value * 4.5 + 1_234.0
                } else {
                    *value
                }
            })
            .collect::<Vec<_>>();
        let domain = SampleDomain::PhysicalLinear {
            normalization: SampleNormalization::RobustPercentile {
                black_percentile: 0.1,
                white_percentile: 0.9,
                max_analysis_samples: 1_000,
            },
        };
        let first = domain.resolve(&input, 1).unwrap().apply(&input, 1).unwrap();
        let second = domain
            .resolve(&transformed, 1)
            .unwrap()
            .apply(&transformed, 1)
            .unwrap();
        for (left, right) in first.iter().zip(&second) {
            if left.is_nan() {
                assert!(right.is_nan());
            } else {
                assert!((left - right).abs() < 2.0e-6, "{left} != {right}");
            }
        }
    }

    #[test]
    fn sample_domain_rejects_invalid_buffers_parameters_and_flat_ranges() {
        assert!(SampleDomain::UnitLinear.resolve(&[], 1).is_err());
        assert!(SampleDomain::UnitLinear.resolve(&[0.0], 0).is_err());
        assert!(SampleDomain::UnitLinear.resolve(&[0.0, 1.0], 3).is_err());

        for normalization in [
            SampleNormalization::ExplicitRange {
                black: 1.0,
                white: 1.0,
            },
            SampleNormalization::ExplicitRange {
                black: f32::NAN,
                white: 1.0,
            },
            SampleNormalization::RobustPercentile {
                black_percentile: 0.9,
                white_percentile: 0.1,
                max_analysis_samples: 3,
            },
            SampleNormalization::RobustPercentile {
                black_percentile: 0.0,
                white_percentile: 1.0,
                max_analysis_samples: 2,
            },
        ] {
            let domain = SampleDomain::PhysicalLinear { normalization };
            assert!(domain.resolve(&[1.0, 2.0, 3.0], 3).is_err());
        }

        let flat = SampleDomain::PhysicalLinear {
            normalization: SampleNormalization::default(),
        };
        assert!(flat.resolve(&[42.0; 1_000], 1).is_err());
    }

    #[test]
    fn domain_mapping_does_not_change_existing_stretch_entry_points() {
        let stack = StretchStack::single(StretchConfig {
            model: StretchModel::Identity,
            color_strategy: ColorStrategy::Linked,
            max_analysis_samples: 2,
        });
        let physical = [100.0, 200.0];
        assert_eq!(stack.apply_u8(&physical, 1).unwrap().data, [255, 255]);

        let domain = SampleDomain::PhysicalLinear {
            normalization: SampleNormalization::ExplicitRange {
                black: 0.0,
                white: 1_000.0,
            },
        };
        let mapped = domain
            .resolve(&physical, 1)
            .unwrap()
            .apply(&physical, 1)
            .unwrap();
        assert_eq!(stack.apply_u8(&mapped, 1).unwrap().data, [26, 51]);
    }

    /// `ResolvedCurve::map` as it was before `asinh(strength)` was worked
    /// out once per curve.
    fn previous_map(curve: ResolvedCurve, value: f32) -> f32 {
        if !value.is_finite() {
            return 0.0;
        }
        let mapped = match curve {
            ResolvedCurve::Identity => f64::from(value),
            ResolvedCurve::Linear { black, white } => (f64::from(value) - black) / (white - black),
            ResolvedCurve::Asinh {
                black,
                white,
                strength,
            } => {
                let linear = ((value - black) / (white - black)).max(0.0);
                f64::from((strength * linear).asinh() / strength.asinh())
            }
            ResolvedCurve::Mtf {
                shadows,
                midtone,
                highlights,
            } => {
                let input = 1.0 - f64::from(highlights) + f64::from(value - shadows);
                midtones_transfer_function(midtone, input)
            }
            ResolvedCurve::Ghs(ghs) => ghs.map(f64::from(value)),
            ResolvedCurve::MedianAnchor(anchor) => anchor.map(f64::from(value)),
            ResolvedCurve::Layered(layered) => layered.curve.map(f64::from(value)),
        };
        mapped.clamp(0.0, 1.0) as f32
    }

    fn previous_to_u8(value: f32) -> u8 {
        (value.clamp(0.0, 1.0) * 255.0).round() as u8
    }

    fn previous_to_u16(value: f32) -> u16 {
        (value.clamp(0.0, 1.0) * f32::from(u16::MAX)).round() as u16
    }

    /// `StretchPlan::apply_mapped` as it was: every sample through the
    /// plan's curves one at a time.
    fn previous_apply<T>(plan: &StretchPlan, data: &[f32], convert: impl Fn(f32) -> T) -> Vec<T> {
        let channels = plan.channels;
        match plan.color_strategy {
            ColorStrategy::Linked => data
                .iter()
                .map(|value| convert(previous_map(plan.curves[0], *value)))
                .collect(),
            ColorStrategy::Unlinked => data
                .iter()
                .enumerate()
                .map(|(index, value)| convert(previous_map(plan.curves[index % channels], *value)))
                .collect(),
            ColorStrategy::LuminancePreserving => data
                .chunks_exact(3)
                .flat_map(|pixel| {
                    let mapped = if pixel.iter().any(|value| !value.is_finite()) {
                        [0.0; 3]
                    } else {
                        let luminance = (LUMA_RED * f64::from(pixel[0])
                            + LUMA_GREEN * f64::from(pixel[1])
                            + LUMA_BLUE * f64::from(pixel[2]))
                            as f32;
                        let target = previous_map(plan.curves[0], luminance);
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
                    };
                    mapped.map(&convert)
                })
                .collect(),
        }
    }

    #[test]
    fn rounding_without_libm_matches_f32_round() {
        let mut state = 0xbb67_ae85_84ca_a73b_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut values = vec![
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            1.0e-40,
            1.0,
            1.0 + f32::EPSILON,
            f32::MAX,
            -1.0,
        ];
        // Every value whose scaled form lies within a few steps of a whole
        // number or a half, where rounding is decided.
        for scale in [255.0_f32, 65_535.0] {
            for step in 0..=(2 * scale as u32) {
                let center = step as f32 / 2.0 / scale;
                let mut below = center;
                let mut above = center;
                for _ in 0..4 {
                    below = below.next_down();
                    above = above.next_up();
                    values.extend([below, above]);
                }
                values.push(center);
            }
        }
        values.extend((0..200_000).map(|_| f32::from_bits(next() as u32)));
        values.extend((0..200_000).map(|_| (next() >> 40) as f32 / (1 << 24) as f32));
        for value in values {
            assert_eq!(to_u8(value), previous_to_u8(value), "{value:e}");
            assert_eq!(to_u16(value), previous_to_u16(value), "{value:e}");
        }
    }

    /// Every one of the 2^32 inputs; run with `--release --ignored`.
    #[test]
    #[ignore]
    fn rounding_without_libm_matches_f32_round_for_every_input() {
        let mismatches = (0..=u32::MAX)
            .into_par_iter()
            .filter(|&bits| {
                let value = f32::from_bits(bits);
                to_u8(value) != previous_to_u8(value) || to_u16(value) != previous_to_u16(value)
            })
            .count();
        assert_eq!(mismatches, 0);
    }

    #[test]
    fn plans_map_every_sample_as_before() {
        let mut state = 0x3c6e_f372_fe94_f82b_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let curves = [
            ResolvedCurve::Identity,
            ResolvedCurve::Linear {
                black: 0.02,
                white: 0.8,
            },
            ResolvedCurve::Linear {
                black: -0.5,
                white: 3.0e9,
            },
            ResolvedCurve::Asinh {
                black: 0.01,
                white: 0.9,
                strength: 10.0,
            },
            ResolvedCurve::Asinh {
                black: 0.1,
                white: 0.2,
                strength: 500.0,
            },
            ResolvedCurve::Mtf {
                shadows: 0.05,
                midtone: 0.12,
                highlights: 0.97,
            },
            StretchModel::Ghs(GhsParams {
                stretch_factor: 5.0,
                local_intensity: 2.0,
                symmetry_point: 0.1,
                protect_shadows: 0.02,
                protect_highlights: 0.9,
                black: 0.0,
                white: 1.0,
            })
            .resolve_explicit()
            .unwrap(),
            StretchModel::Ghs(GhsParams {
                stretch_factor: 3.0,
                local_intensity: -1.0,
                symmetry_point: 0.3,
                protect_shadows: 0.0,
                protect_highlights: 1.0,
                black: 0.0,
                white: 1.0,
            })
            .resolve_explicit()
            .unwrap(),
        ];
        let sample = |next: &mut dyn FnMut() -> u64| match next() % 50 {
            0 => f32::NAN,
            1 => f32::INFINITY,
            2 => f32::NEG_INFINITY,
            3 => -0.0,
            4 => 0.0,
            5 => -0.25,
            6 => 7.0,
            7 => 1.0e-9,
            _ => (next() >> 40) as f32 / (1 << 24) as f32,
        };
        // Empty, single-pixel, and chunk-crossing images.
        for pixels in [0, 1, 5, MAP_CHUNK_PIXELS + 7] {
            for (index, &curve) in curves.iter().enumerate() {
                let other = curves[(index + 3) % curves.len()];
                let third = curves[(index + 5) % curves.len()];
                let plans = [
                    StretchPlan::from_resolved(1, ColorStrategy::Linked, vec![curve]).unwrap(),
                    StretchPlan::from_resolved(3, ColorStrategy::Linked, vec![curve]).unwrap(),
                    StretchPlan::from_resolved(
                        3,
                        ColorStrategy::Unlinked,
                        vec![curve, other, third],
                    )
                    .unwrap(),
                    StretchPlan::from_resolved(3, ColorStrategy::LuminancePreserving, vec![curve])
                        .unwrap(),
                ];
                for plan in plans {
                    let data = (0..pixels * plan.channels)
                        .map(|_| sample(&mut next))
                        .collect::<Vec<_>>();
                    let label = format!("{curve:?} {:?} {pixels}", plan.color_strategy);
                    let f32s = plan.apply_f32(&data, plan.channels).unwrap();
                    let expected = previous_apply(&plan, &data, |value| value);
                    assert_eq!(f32s.len(), expected.len(), "{label}");
                    for (got, expected) in f32s.iter().zip(&expected) {
                        assert_eq!(got.to_bits(), expected.to_bits(), "{label}");
                    }
                    assert_eq!(
                        plan.apply_u8(&data, plan.channels).unwrap(),
                        previous_apply(&plan, &data, previous_to_u8),
                        "{label}"
                    );
                    assert_eq!(
                        plan.apply_u16(&data, plan.channels).unwrap(),
                        previous_apply(&plan, &data, previous_to_u16),
                        "{label}"
                    );
                    for &value in data.iter().take(64) {
                        assert_eq!(
                            curve.map(value).to_bits(),
                            previous_map(curve, value).to_bits(),
                            "{label}"
                        );
                    }
                }
            }
        }
    }

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
        assert_eq!(black, analysis.linked.sorted[10]);
        assert_eq!(white, analysis.linked.sorted[995]);
        assert_eq!(strength, 10.0);
    }

    #[test]
    fn midtones_transfer_retains_previous_boundaries() {
        assert_eq!(midtones_transfer_function(0.5, 0.0), 0.0);
        assert_eq!(midtones_transfer_function(0.5, 1.0), 1.0);
        assert!((midtones_transfer_function(0.5, 0.5) - 0.5).abs() < 1.0e-12);
        assert!(midtones_transfer_function(0.25, 0.25) > 0.45);
    }

    #[test]
    fn analysis_is_bounded_and_color_aware() {
        let data = [0.1, 0.2, 0.3].repeat(100_000);
        let analysis = StretchAnalysis::analyze(&data, 3, 3_000).unwrap();
        assert!(analysis.linked_statistics().count <= 3_000);
        assert!(analysis.channel_statistics().iter().all(Option::is_some));
        assert!(analysis.luminance_statistics().is_some());
    }

    #[test]
    fn linked_analysis_does_not_require_every_color_channel() {
        let data = [0.1, f32::NAN, 0.3].repeat(10);
        let analysis = StretchAnalysis::analyze(&data, 3, 30).unwrap();
        let statistics = analysis.channel_statistics();
        assert!(statistics[0].is_some());
        assert!(statistics[1].is_none());
        assert!(statistics[2].is_some());
        assert!(analysis.luminance_statistics().is_none());

        let linked = StretchConfig::percentile_asinh(0.01, 0.995, 10.0, 30);
        assert!(linked.resolve(&analysis).is_ok());
        let unlinked = StretchConfig {
            color_strategy: ColorStrategy::Unlinked,
            ..linked.clone()
        };
        assert!(unlinked.resolve(&analysis).is_err());
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
        assert_eq!(json["curves"][0]["params"]["stretch_factor"], 2.0);
        assert!(json["curves"][0].get("normalization_span").is_none());
    }

    #[test]
    fn parameterized_config_round_trips_for_pipeline_storage() {
        let config = StretchConfig {
            model: StretchModel::Ghs(GhsParams {
                stretch_factor: 3.0,
                local_intensity: -0.5,
                symmetry_point: 0.35,
                protect_shadows: 0.1,
                protect_highlights: 0.85,
                black: 0.01,
                white: 0.95,
            }),
            color_strategy: ColorStrategy::LuminancePreserving,
            max_analysis_samples: 123_456,
        };
        let encoded = serde_json::to_string(&config).unwrap();
        let decoded = serde_json::from_str::<StretchConfig>(&encoded).unwrap();
        assert_eq!(decoded, config);
    }

    #[test]
    fn ordered_stack_resolves_each_stage_against_the_previous_output() {
        let stack = StretchStack::new(vec![
            StretchConfig {
                model: StretchModel::Linear {
                    black: 0.0,
                    white: 0.8,
                },
                color_strategy: ColorStrategy::Linked,
                max_analysis_samples: 16,
            },
            StretchConfig {
                model: StretchModel::Linear {
                    black: 0.25,
                    white: 0.75,
                },
                color_strategy: ColorStrategy::Linked,
                max_analysis_samples: 16,
            },
        ])
        .unwrap();
        let mut events = Vec::new();
        let output = stack
            .apply_f32_with_progress(&[0.0, 0.2, 0.4, 0.8], 1, |event| events.push(event))
            .unwrap();
        assert_eq!(output.data, [0.0, 0.0, 0.5, 1.0]);
        assert_eq!(output.plans.len(), 2);
        assert_eq!(events.len(), 6);
        assert_eq!(events[0].state, StretchStageState::Resolving);
        assert_eq!(events[2].state, StretchStageState::Completed);
        assert_eq!(events[3].stage_index, 1);
        assert_eq!(events[5].state, StretchStageState::Completed);
    }

    #[test]
    fn ordered_stack_converts_to_u8_only_after_the_final_stage() {
        let stack = StretchStack::new(vec![
            StretchConfig {
                model: StretchModel::Linear {
                    black: 0.0,
                    white: 0.75,
                },
                color_strategy: ColorStrategy::Linked,
                max_analysis_samples: 4,
            },
            StretchConfig {
                model: StretchModel::Linear {
                    black: 0.0,
                    white: 0.5,
                },
                color_strategy: ColorStrategy::Linked,
                max_analysis_samples: 4,
            },
        ])
        .unwrap();
        let output = stack.apply_u8(&[0.1875], 1).unwrap();
        assert_eq!(output.data, [128]);
        assert_eq!(output.plans.len(), 2);
    }

    #[test]
    fn ordered_stack_u16_output_retains_sub_u8_distinctions() {
        let stack = StretchStack::single(StretchConfig {
            model: StretchModel::Identity,
            color_strategy: ColorStrategy::Linked,
            max_analysis_samples: 2,
        });
        let input = [0.5, 0.5001];
        let output_u8 = stack.apply_u8(&input, 1).unwrap();
        let output_u16 = stack.apply_u16(&input, 1).unwrap();

        assert_eq!(output_u8.data[0], output_u8.data[1]);
        assert_ne!(output_u16.data[0], output_u16.data[1]);
        assert_eq!(output_u16.data, [32_768, 32_774]);
        assert_eq!(output_u16.plans.len(), 1);
    }

    #[test]
    fn u16_autostretch_is_not_an_upscaled_u8_lut() {
        // Realistic dark-sky statistics (a full-range uniform ramp has MAD
        // comparable to its median, which N.I.N.A.'s unclamped shadow point
        // degenerates to a binary map — faithfully, but useless here).
        let input = (0..=16_000).step_by(7).collect::<Vec<_>>();
        let statistics = statistics_u16(&input);
        let output = stretch_u16_to_u16(&input, &statistics, &StretchParams::default());

        assert!(output.iter().any(|value| value % 257 != 0));
    }

    #[test]
    fn ordered_stack_rejects_empty_construction_and_deserialization() {
        assert!(StretchStack::new(Vec::new()).is_err());
        assert!(serde_json::from_str::<StretchStack>("[]").is_err());
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

    /// `statistics_u16`'s counting and sums as they were before they ran on
    /// rayon: one histogram and running sums over every sample in turn.
    fn previous_histogram_sums(data: &[u16]) -> (Vec<u32>, u64, u128) {
        let mut histogram = vec![0u32; 65_536];
        let mut sum = 0u64;
        let mut sum_sq = 0u128;
        for &value in data {
            histogram[value as usize] += 1;
            sum += u64::from(value);
            sum_sq += u128::from(value) * u128::from(value);
        }
        (histogram, sum, sum_sq)
    }

    /// The `u16` stretches as they were before they ran on rayon.
    fn previous_stretches(
        data: &[u16],
        stats: &Statistics,
        params: &StretchParams,
    ) -> (Vec<u8>, Vec<u16>) {
        let map = stretch_u16_map(stats, params);
        let narrow = data.iter().map(|value| map[usize::from(*value)]).collect();
        let (shadows, midtone, highlights) = stretch_u16_curve(stats, params);
        let map = (0..65_536)
            .map(|value| {
                let input = 1.0 - highlights + value as f64 / 65_535.0 - shadows;
                denormalize_u16(midtones_transfer_function(midtone, input))
            })
            .collect::<Vec<_>>();
        let wide = data.iter().map(|value| map[usize::from(*value)]).collect();
        (narrow, wide)
    }

    #[test]
    fn parallel_u16_statistics_and_stretches_match_the_serial_ones() {
        let mut state = 0x6a09_e667_f3bc_c908_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for count in [0, 1, 2, 3, 1000, U16_CHUNK, 2 * U16_CHUNK + 5] {
            for spread in [1_u64, 300, 65_536] {
                let data = (0..count)
                    .map(|_| (next() % spread) as u16 + if spread == 1 { 65_535 } else { 0 })
                    .collect::<Vec<_>>();
                let statistics = statistics_u16(&data);
                // The rest of the statistics comes from these, as before.
                let (histogram, sum, sum_sq) = previous_histogram_sums(&data);
                assert_eq!(statistics.count, count);
                if count > 0 {
                    let mean = sum as f64 / count as f64;
                    let variance = (sum_sq as f64 / count as f64 - mean * mean).max(0.0);
                    assert_eq!(
                        statistics.mean.to_bits(),
                        mean.to_bits(),
                        "{count} {spread}"
                    );
                    assert_eq!(
                        statistics.std_dev.to_bits(),
                        variance.sqrt().to_bits(),
                        "{count} {spread}"
                    );
                    let min = histogram.iter().position(|count| *count > 0).unwrap() as u16;
                    let max = histogram.iter().rposition(|count| *count > 0).unwrap() as u16;
                    assert_eq!((statistics.min, statistics.max), (min, max));
                }
                let params = StretchParams::default();
                let (narrow, wide) = previous_stretches(&data, &statistics, &params);
                assert_eq!(stretch_u16_to_u8(&data, &statistics, &params), narrow);
                assert_eq!(stretch_u16_to_u16(&data, &statistics, &params), wide);
            }
        }
    }

    #[test]
    fn u16_statistics_match_the_previous_sorted_reference() {
        let mut state = 0xABCDEF12345_u64;
        let data = (0..100_000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as u16) / 3 + 500
            })
            .collect::<Vec<_>>();
        let stats = statistics_u16(&data);
        let mut sorted = data.clone();
        sorted.sort_unstable();
        let expected_median = sorted[sorted.len().div_ceil(2) - 1];
        let mut deviations = data
            .iter()
            .map(|value| (i32::from(*value) - i32::from(expected_median)).unsigned_abs() as u16)
            .collect::<Vec<_>>();
        deviations.sort_unstable();
        let expected_mad = deviations[deviations.len().div_ceil(2) - 1];
        assert_eq!(stats.min, sorted[0]);
        assert_eq!(stats.max, *sorted.last().unwrap());
        assert_eq!(stats.median, expected_median);
        assert!((stats.mad - f64::from(expected_mad)).abs() <= 1.0);
        assert_eq!(stats.count, data.len());
    }

    #[test]
    fn u16_stretch_is_byte_identical_to_nina() {
        let data = (0..=u16::MAX).collect::<Vec<_>>();
        for statistics in [
            Statistics {
                min: 0,
                max: u16::MAX,
                mean: 8_000.0,
                std_dev: 1_000.0,
                median: 6_000,
                mad: 400.0,
                count: data.len(),
            },
            Statistics {
                min: 0,
                max: u16::MAX,
                mean: 50_000.0,
                std_dev: 1_000.0,
                median: 52_000,
                mad: 600.0,
                count: data.len(),
            },
            // Dark narrowband frame whose shadow point goes negative: the
            // clamp N.I.N.A. does NOT apply must not reappear.
            Statistics {
                min: 0,
                max: 4_000,
                mean: 320.0,
                std_dev: 260.0,
                median: 300,
                mad: 250.0,
                count: data.len(),
            },
        ] {
            let expected =
                nina_reference_stretch_u16(&data, &statistics, &StretchParams::default());
            let actual = stretch_u16_to_u16(&data, &statistics, &StretchParams::default());
            assert_eq!(actual, expected);

            // The u8 path is exactly the u16 result rendered at 8 bits.
            let expected_u8 = expected.iter().map(|v| (v >> 8) as u8).collect::<Vec<_>>();
            let actual_u8 = stretch_u16_to_u8(&data, &statistics, &StretchParams::default());
            assert_eq!(actual_u8, expected_u8);
        }
    }

    #[test]
    fn dark_frame_shadows_go_negative_like_nina() {
        // median + clip * mad * 1.4826 < 0 for these statistics; with the
        // old clamp the zero pixel mapped to 0, without it the background
        // lifts. Guard the semantic so the clamp cannot silently return.
        let stats = Statistics {
            min: 0,
            max: 4_000,
            mean: 320.0,
            std_dev: 260.0,
            median: 300,
            mad: 250.0,
            count: 1,
        };
        let params = StretchParams::default();
        let stretched = stretch_u16_to_u16(&[0], &stats, &params);
        assert!(
            stretched[0] > 0,
            "negative shadow point must lift the zero pixel, got {}",
            stretched[0]
        );
    }

    #[test]
    fn denormalize_matches_nina_denormalize_ushort() {
        assert_eq!(denormalize_u16(0.0), 0);
        assert_eq!(denormalize_u16(1.0), 65_535);
        // Below mid-scale: round half up. 0.25 * 65535 = 16383.75 -> 16384.
        assert_eq!(denormalize_u16(0.25), 16_384);
        // Above mid-scale: truncate. 0.75 * 65535 = 49151.25 -> 49151, and
        // a value that would round up under half-up stays truncated:
        // 0.9 * 65535 = 58981.5 -> 58981.
        assert_eq!(denormalize_u16(0.75), 49_151);
        assert_eq!(denormalize_u16(0.9), 58_981);
        assert_eq!(denormalize_u16(-0.5), 0);
        assert_eq!(denormalize_u16(1.5), 65_535);
    }

    #[test]
    fn f32_auto_stretch_places_background_near_requested_median() {
        let mut data = (0..10_000)
            .map(|index| 1000.0 + (index % 19) as f32 - 9.0)
            .collect::<Vec<_>>();
        data.extend([0.0, 25_000.0, f32::NAN]);
        let stretched = auto_stretch_f32_to_u8(&data, &StretchParams::default()).unwrap();
        let mut background = stretched[..10_000].to_vec();
        background.sort_unstable();
        assert!((40..=65).contains(&background[background.len() / 2]));
        assert_eq!(stretched.last(), Some(&0));
    }

    #[test]
    fn f32_auto_stretch_rejects_degenerate_input() {
        assert!(auto_stretch_f32_to_u8(&[], &StretchParams::default()).is_none());
        assert!(auto_stretch_f32_to_u8(&[f32::NAN; 4], &StretchParams::default()).is_none());
        assert!(auto_stretch_f32_to_u8(&[3.0; 100], &StretchParams::default()).is_none());
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

    /// A dim, noisy background with a few bright stars, in `[0, 1]`.
    fn sky() -> Vec<f32> {
        let mut data = (0..10_000)
            .map(|index| 0.05 + ((index * 7919) % 101) as f32 * 0.0002)
            .collect::<Vec<_>>();
        for star in data.iter_mut().step_by(997) {
            *star = 0.9;
        }
        data
    }

    fn median_anchor(params: MedianAnchorParams) -> StretchConfig {
        StretchConfig {
            model: StretchModel::MedianAnchor(params),
            color_strategy: ColorStrategy::Linked,
            max_analysis_samples: 20_000,
        }
    }

    fn median_anchor_curve(data: &[f32], params: MedianAnchorParams) -> ResolvedCurve {
        median_anchor(params).resolve_for(data, 1).unwrap().curves()[0]
    }

    #[test]
    fn median_anchor_places_the_median_exactly_on_the_target() {
        let data = sky();
        let median = StretchAnalysis::analyze(&data, 1, 20_000)
            .unwrap()
            .linked_statistics()
            .median as f32;
        for target in [0.1, 0.25, 0.5] {
            let curve = median_anchor_curve(
                &data,
                MedianAnchorParams {
                    target_median: target,
                    ..MedianAnchorParams::default()
                },
            );
            assert!(
                (curve.map(median) - target as f32).abs() < 1.0e-5,
                "{target}"
            );
        }
    }

    #[test]
    fn median_anchor_black_point_sits_sigmas_below_the_median_or_at_the_minimum() {
        let data = sky();
        // One sigma lands inside this narrow background; 2.7 would pass its
        // minimum, where the black point stops.
        let ResolvedCurve::MedianAnchor(clipped) = median_anchor_curve(
            &data,
            MedianAnchorParams {
                black_point_sigma: Some(1.0),
                ..MedianAnchorParams::default()
            },
        ) else {
            panic!("a median anchor curve");
        };
        let statistics = StretchAnalysis::analyze(&data, 1, 20_000)
            .unwrap()
            .linked_statistics();
        // The black point is under the median, above the minimum, and maps to 0.
        assert!(f64::from(clipped.black) < statistics.median);
        assert!(f64::from(clipped.black) > statistics.min);
        assert_eq!(ResolvedCurve::MedianAnchor(clipped).map(clipped.black), 0.0);
        let ResolvedCurve::MedianAnchor(unclipped) = median_anchor_curve(
            &data,
            MedianAnchorParams {
                black_point_sigma: None,
                ..MedianAnchorParams::default()
            },
        ) else {
            panic!("a median anchor curve");
        };
        assert_eq!(f64::from(unclipped.black), statistics.min);
    }

    #[test]
    fn median_anchor_curves_stay_pinned_and_in_order_with_every_option() {
        let data = sky();
        let options = [
            MedianAnchorParams::default(),
            MedianAnchorParams {
                curves_boost: 0.5,
                ..MedianAnchorParams::default()
            },
            MedianAnchorParams {
                hdr_amount: 1.0,
                hdr_knee: 0.6,
                ..MedianAnchorParams::default()
            },
            MedianAnchorParams {
                curves_boost: 0.9,
                hdr_amount: 1.0,
                ..MedianAnchorParams::default()
            },
        ];
        for params in options {
            let curve = median_anchor_curve(&data, params);
            assert_eq!(curve.map(1.0), 1.0, "{params:?}");
            let mut previous = curve.map(0.0);
            assert_eq!(previous, 0.0);
            for step in 1..=2_000 {
                let value = curve.map(step as f32 / 2_000.0);
                assert!(value >= previous, "{params:?} at step {step}");
                previous = value;
            }
        }
    }

    #[test]
    fn median_anchor_boost_lifts_only_above_the_target_and_compression_only_above_the_knee() {
        let data = sky();
        let plain = median_anchor_curve(&data, MedianAnchorParams::default());
        let boosted = median_anchor_curve(
            &data,
            MedianAnchorParams {
                curves_boost: 0.5,
                ..MedianAnchorParams::default()
            },
        );
        let compressed = median_anchor_curve(
            &data,
            MedianAnchorParams {
                hdr_amount: 1.0,
                ..MedianAnchorParams::default()
            },
        );
        for step in 0..=1_000 {
            let input = step as f32 / 1_000.0;
            let (base, lifted, lowered) =
                (plain.map(input), boosted.map(input), compressed.map(input));
            if base <= 0.25 {
                assert_eq!(lifted, base);
            } else {
                assert!(lifted >= base);
            }
            if base <= 0.75 {
                assert_eq!(lowered, base);
            } else {
                assert!(lowered <= base);
            }
        }
        assert!(boosted.map(0.5) > plain.map(0.5));
        assert!(compressed.map(0.95) < plain.map(0.95));
    }

    #[test]
    fn median_anchor_normalize_brings_the_brightest_sample_to_one() {
        let data = sky()
            .into_iter()
            .map(|value| value * 0.5)
            .collect::<Vec<_>>();
        let plain = median_anchor_curve(&data, MedianAnchorParams::default());
        let normalized = median_anchor_curve(
            &data,
            MedianAnchorParams {
                normalize: true,
                ..MedianAnchorParams::default()
            },
        );
        assert!(plain.map(0.45) < 1.0);
        assert!((normalized.map(0.45) - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn median_anchor_works_unlinked_and_preserving_luminance() {
        let mono = sky();
        let rgb = mono
            .iter()
            .flat_map(|&value| [value, value * 0.8, value * 1.1])
            .collect::<Vec<_>>();
        for color_strategy in [ColorStrategy::Unlinked, ColorStrategy::LuminancePreserving] {
            let config = StretchConfig {
                color_strategy,
                ..median_anchor(MedianAnchorParams::default())
            };
            let plan = config.resolve_for(&rgb, 3).unwrap();
            let out = plan.apply_f32(&rgb, 3).unwrap();
            assert_eq!(out.len(), rgb.len());
            assert!(out.iter().all(|value| (0.0..=1.0).contains(value)));
        }
    }

    #[test]
    fn median_anchor_refuses_settings_out_of_range_and_round_trips_as_json() {
        let data = sky();
        for params in [
            MedianAnchorParams {
                target_median: 1.0,
                ..MedianAnchorParams::default()
            },
            MedianAnchorParams {
                black_point_sigma: Some(-1.0),
                ..MedianAnchorParams::default()
            },
            MedianAnchorParams {
                black_point_sigma: Some(0.0),
                ..MedianAnchorParams::default()
            },
            MedianAnchorParams {
                curves_boost: 0.95,
                ..MedianAnchorParams::default()
            },
            MedianAnchorParams {
                hdr_amount: 1.5,
                ..MedianAnchorParams::default()
            },
            MedianAnchorParams {
                hdr_knee: 1.0,
                ..MedianAnchorParams::default()
            },
        ] {
            assert!(
                median_anchor(params).resolve_for(&data, 1).is_err(),
                "{params:?}"
            );
        }
        let model = StretchModel::MedianAnchor(MedianAnchorParams {
            black_point_sigma: None,
            curves_boost: 0.3,
            ..MedianAnchorParams::default()
        });
        let json = serde_json::to_string(&model).unwrap();
        assert!(json.contains("\"type\":\"median-anchor\""), "{json}");
        assert_eq!(serde_json::from_str::<StretchModel>(&json).unwrap(), model);
        // The optional settings may be left out.
        let terse: StretchModel = serde_json::from_str(
            r#"{"type":"median-anchor","target_median":0.2,"black_point_sigma":3.0}"#,
        )
        .unwrap();
        assert_eq!(
            terse,
            StretchModel::MedianAnchor(MedianAnchorParams {
                target_median: 0.2,
                black_point_sigma: Some(3.0),
                ..MedianAnchorParams::default()
            })
        );
        // Every setting may be left out, the black point included; null
        // asks for the image minimum.
        let bare: StretchModel = serde_json::from_str(r#"{"type":"median-anchor"}"#).unwrap();
        assert_eq!(
            bare,
            StretchModel::MedianAnchor(MedianAnchorParams::default())
        );
        let unclipped: StretchModel =
            serde_json::from_str(r#"{"type":"median-anchor","black_point_sigma":null}"#).unwrap();
        assert_eq!(
            unclipped,
            StretchModel::MedianAnchor(MedianAnchorParams {
                black_point_sigma: None,
                ..MedianAnchorParams::default()
            })
        );
    }

    #[test]
    fn median_anchor_noise_is_the_gaussian_sigma_and_ignores_stars() {
        // Twelve uniforms sum to a near-Gaussian with sigma 1: here sigma
        // 0.004 about 0.05, with one sample in fifty a bright star.
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut uniform = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        let data = (0..100_000)
            .map(|index| {
                if index % 50 == 0 {
                    0.8
                } else {
                    let normal = (0..12).map(|_| uniform()).sum::<f64>() - 6.0;
                    (0.05 + 0.004 * normal) as f32
                }
            })
            .collect::<Vec<_>>();
        let analysis = StretchAnalysis::analyze(&data, 1, 200_000).unwrap();
        let noise = analysis.linked.lower_half_noise();
        assert!((noise / 0.004 - 1.0).abs() < 0.03, "{noise}");
        let ResolvedCurve::MedianAnchor(anchor) =
            median_anchor_curve(&data, MedianAnchorParams::default())
        else {
            panic!("a median anchor curve");
        };
        let median = analysis.linked_statistics().median;
        let sigmas = (median - f64::from(anchor.black)) / 0.004;
        assert!((sigmas - 2.7).abs() < 0.1, "{sigmas}");
    }

    #[test]
    fn median_anchor_without_measurable_noise_uses_the_minimum() {
        // A quarter and more of the samples sit on the median, so the noise
        // measures 0; the black point falls back to the minimum and the
        // median still lands on the target.
        let data = (0..1_000)
            .map(|index| match index {
                0..10 => 0.02,
                10..600 => 0.1,
                _ => 0.1 + (index - 600) as f32 * 0.002,
            })
            .collect::<Vec<_>>();
        let curve = median_anchor_curve(&data, MedianAnchorParams::default());
        let ResolvedCurve::MedianAnchor(anchor) = curve else {
            panic!("a median anchor curve");
        };
        assert_eq!(anchor.black, 0.02);
        assert!((curve.map(0.1) - 0.25).abs() < 1.0e-5);
    }

    const FIELD: (usize, usize) = (128, 96);

    /// Background near 0.05 with some noise, a bright plateau at 0.5 holding
    /// a fine checkerboard, and a small star peaking at 0.5 on the background.
    fn field() -> Vec<f32> {
        let (width, height) = FIELD;
        (0..width * height)
            .map(|index| {
                let (x, y) = (index % width, index / width);
                let noise = ((index * 7919) % 101) as f32 / 101.0 - 0.5;
                let mut value = 0.05 + 0.004 * noise;
                if (70..110).contains(&x) && (30..70).contains(&y) {
                    value = 0.5 + if (x + y) % 2 == 0 { 0.01 } else { -0.01 };
                }
                let (dx, dy) = (x as f32 - 30.0, y as f32 - 48.0);
                value + 0.45 * (-(dx * dx + dy * dy) / (2.0 * 1.5 * 1.5)).exp()
            })
            .collect()
    }

    fn field_shape(channels: usize) -> ImageShape {
        ImageShape::new(FIELD.0, FIELD.1, channels)
    }

    fn layered(params: LayeredParams) -> StretchConfig {
        StretchConfig {
            model: StretchModel::Layered(params),
            color_strategy: ColorStrategy::Linked,
            max_analysis_samples: 200_000,
        }
    }

    fn layered_output(data: &[f32], channels: usize, params: LayeredParams) -> Vec<f32> {
        StretchStack::single(layered(params))
            .apply_f32_image(data, field_shape(channels))
            .unwrap()
            .data
    }

    /// Mean step between neighbouring pixels inside the plateau, where the
    /// checkerboard alternates.
    fn plateau_contrast(output: &[f32]) -> f64 {
        let steps = (40..60)
            .flat_map(|y| {
                (80..100).map(move |x| {
                    let index = y * FIELD.0 + x;
                    f64::from((output[index + 1] - output[index]).abs())
                })
            })
            .collect::<Vec<_>>();
        steps.iter().sum::<f64>() / steps.len() as f64
    }

    #[test]
    fn layered_without_recovery_or_compression_is_the_median_anchor_curve() {
        let data = field();
        let params = LayeredParams {
            aggressiveness: 0.2,
            ..LayeredParams::default()
        };
        let plain = median_anchor(MedianAnchorParams {
            target_median: params.target_background,
            black_point_sigma: Some(6.0 - 4.5 * params.aggressiveness),
            ..MedianAnchorParams::default()
        });
        let config = StretchConfig {
            max_analysis_samples: 200_000,
            ..plain
        };
        let expected = config
            .resolve_for(&data, 1)
            .unwrap()
            .apply_f32(&data, 1)
            .unwrap();
        assert_eq!(layered_output(&data, 1, params), expected);

        // A grey RGB image keeps its luminance curve on every channel.
        let grey = data.iter().flat_map(|&v| [v; 3]).collect::<Vec<_>>();
        let output = layered_output(&grey, 3, params);
        for (pixel, &value) in output.chunks_exact(3).zip(&expected) {
            for channel in pixel {
                assert!((channel - value).abs() < 1.0e-5, "{pixel:?} {value}");
            }
        }
    }

    #[test]
    fn layered_places_the_background_median_near_the_target() {
        let data = field();
        for target in [0.1, 0.15, 0.3] {
            for params in [
                LayeredParams {
                    target_background: target,
                    ..LayeredParams::default()
                },
                LayeredParams {
                    target_background: target,
                    contrast_recovery: 0.5,
                    dynamic_range_compression: 1.0,
                    ..LayeredParams::default()
                },
            ] {
                let mut output = layered_output(&data, 1, params);
                output.sort_unstable_by(f32::total_cmp);
                let median = output[output.len() / 2];
                assert!(
                    (median - target as f32).abs() < 0.01,
                    "{params:?}: {median}"
                );
            }
        }
    }

    #[test]
    fn contrast_recovery_raises_fine_contrast_on_bright_structure() {
        let data = field();
        let base = LayeredParams {
            scale_separation: 32,
            ..LayeredParams::default()
        };
        let plain = plateau_contrast(&layered_output(&data, 1, base));
        let mut previous = plain;
        for recovery in [0.1, 0.25, 0.5, 1.0] {
            let contrast = plateau_contrast(&layered_output(
                &data,
                1,
                LayeredParams {
                    contrast_recovery: recovery,
                    ..base
                },
            ));
            assert!(contrast >= previous, "{recovery}: {contrast} vs {previous}");
            previous = contrast;
        }
        // Detail gains up to four times, and no more.
        assert!(previous > plain * 3.0, "{previous} vs {plain}");
        assert!(previous < plain * 4.01, "{previous} vs {plain}");
    }

    #[test]
    fn compression_lowers_a_star_but_not_a_broad_plateau() {
        let mut data = field();
        // A smooth plateau of the star's brightness.
        for y in 30..70 {
            for x in 70..110 {
                data[y * FIELD.0 + x] = 0.5;
            }
        }
        let plain = layered_output(&data, 1, LayeredParams::default());
        let compressed = layered_output(
            &data,
            1,
            LayeredParams {
                dynamic_range_compression: 1.0,
                ..LayeredParams::default()
            },
        );
        let star = 48 * FIELD.0 + 30;
        let plateau = 50 * FIELD.0 + 90;
        assert!(plain[star] > 0.5 && (plain[star] - plain[plateau]).abs() < 0.01);
        assert!(
            compressed[star] < plain[star] - 0.02,
            "{}",
            compressed[star]
        );
        assert_eq!(compressed[plateau], plain[plateau]);
    }

    #[test]
    fn layered_rgb_stays_in_range_and_missing_samples_come_out_zero() {
        let mono = field();
        let mut rgb = mono
            .iter()
            .flat_map(|&v| [v * 1.3, v, v * 0.7])
            .collect::<Vec<_>>();
        rgb[5 * 3] = f32::NAN;
        rgb[9 * 3 + 2] = f32::INFINITY;
        let mut mono = mono;
        mono[11] = f32::NEG_INFINITY;
        for params in [
            LayeredParams::default(),
            LayeredParams {
                contrast_recovery: 1.0,
                dynamic_range_compression: 1.0,
                saturation_amount: 1.0,
                saturation_boost: 1.0,
                lightness_mask: false,
                scale_separation: 8,
                ..LayeredParams::default()
            },
        ] {
            let output = layered_output(&rgb, 3, params);
            assert!(output.iter().all(|v| (0.0..=1.0).contains(v)), "{params:?}");
            assert_eq!(&output[5 * 3..6 * 3], &[0.0; 3]);
            assert_eq!(&output[9 * 3..10 * 3], &[0.0; 3]);
            assert_eq!(layered_output(&mono, 1, params)[11], 0.0);
        }
    }

    #[test]
    fn layered_saturation_lifts_colour_and_keeps_hue() {
        let rgb = field()
            .iter()
            .flat_map(|&v| [v * 1.2, v, v * 0.8])
            .collect::<Vec<_>>();
        let plain = layered_output(&rgb, 3, LayeredParams::default());
        let lifted = layered_output(
            &rgb,
            3,
            LayeredParams {
                saturation_amount: 1.0,
                ..LayeredParams::default()
            },
        );
        let saturation = |p: &[f32]| {
            let max = p[0].max(p[1]).max(p[2]);
            (max - p[0].min(p[1]).min(p[2])) / max
        };
        // Near the black point blue clips to 0, so those pixels are fully
        // saturated already; every other one gains colour.
        let (mut raised, mut eligible) = (0, 0);
        for (before, after) in plain.chunks_exact(3).zip(lifted.chunks_exact(3)) {
            let value = before[0].max(before[1]).max(before[2]);
            if value < 0.01 {
                continue;
            }
            assert!((value - after[0].max(after[1]).max(after[2])).abs() < 1.0e-6);
            assert!(saturation(after) >= saturation(before) - 1.0e-6);
            if saturation(before) < 0.95 {
                eligible += 1;
                raised += usize::from(saturation(after) > saturation(before) + 1.0e-4);
            }
            // Red stays the largest channel and blue the smallest.
            assert!(after[0] >= after[1] && after[1] >= after[2], "{after:?}");
        }
        assert!(
            eligible > 100 && raised == eligible,
            "{raised} of {eligible}"
        );
    }

    #[test]
    fn shape_less_methods_refuse_the_layered_stretch() {
        let data = field();
        let stack = StretchStack::new(vec![
            StretchConfig::auto_mtf(StretchParams::default(), 4096),
            layered(LayeredParams::default()),
        ])
        .unwrap();
        for error in [
            stack.apply_f32(&data, 1).unwrap_err(),
            stack.apply_u8(&data, 1).unwrap_err(),
            stack.apply_u16(&data, 1).unwrap_err(),
        ] {
            assert!(
                error.to_string().contains("stage 2/2")
                    && error
                        .to_string()
                        .contains("needs the image's width and height"),
                "{error}"
            );
        }
        let plan = layered(LayeredParams::default())
            .resolve_for(&data, 1)
            .unwrap();
        for error in [
            plan.apply_f32(&data, 1).unwrap_err(),
            plan.apply_u8(&data, 1).map(|_| ()).unwrap_err(),
            plan.apply_u16(&data, 1).map(|_| ()).unwrap_err(),
        ] {
            assert!(
                error
                    .to_string()
                    .contains("needs the image's width and height")
            );
        }
        assert!(
            layered(LayeredParams::default())
                .resolve_explicit(1)
                .is_err()
        );
        // The shape must match the samples.
        assert!(
            stack
                .apply_f32_image(&data, ImageShape::new(FIELD.0 + 1, FIELD.1, 1))
                .is_err()
        );
        let rgb = data.iter().flat_map(|&v| [v; 3]).collect::<Vec<_>>();
        assert!(
            StretchStack::single(StretchConfig {
                color_strategy: ColorStrategy::Unlinked,
                ..layered(LayeredParams::default())
            })
            .apply_f32_image(&rgb, field_shape(3))
            .is_err()
        );
    }

    #[test]
    fn image_methods_leave_every_other_model_unchanged() {
        let data = field();
        let rgb = data
            .iter()
            .flat_map(|&v| [v * 1.1, v, v * 0.9])
            .collect::<Vec<_>>();
        let stack = StretchStack::new(vec![
            StretchConfig::percentile_asinh(0.001, 0.999, 50.0, 4096),
            StretchConfig {
                color_strategy: ColorStrategy::LuminancePreserving,
                ..median_anchor(MedianAnchorParams::default())
            },
        ])
        .unwrap();
        let shape = field_shape(3);
        assert_eq!(
            stack.apply_f32_image(&rgb, shape).unwrap(),
            stack.apply_f32(&rgb, 3).unwrap()
        );
        assert_eq!(
            stack.apply_u8_image(&rgb, shape).unwrap(),
            stack.apply_u8(&rgb, 3).unwrap()
        );
        assert_eq!(
            stack.apply_u16_image(&rgb, shape).unwrap(),
            stack.apply_u16(&rgb, 3).unwrap()
        );
    }

    #[test]
    fn layered_refuses_settings_out_of_range_and_round_trips_as_json() {
        let data = field();
        for params in [
            LayeredParams {
                target_background: 0.0,
                ..LayeredParams::default()
            },
            LayeredParams {
                aggressiveness: 1.5,
                ..LayeredParams::default()
            },
            LayeredParams {
                dynamic_range_compression: -0.1,
                ..LayeredParams::default()
            },
            LayeredParams {
                scale_separation: 4,
                ..LayeredParams::default()
            },
            LayeredParams {
                scale_separation: 2048,
                ..LayeredParams::default()
            },
            LayeredParams {
                contrast_recovery: f64::NAN,
                ..LayeredParams::default()
            },
            LayeredParams {
                saturation_amount: 2.0,
                ..LayeredParams::default()
            },
            LayeredParams {
                saturation_boost: -1.0,
                ..LayeredParams::default()
            },
        ] {
            assert!(layered(params).resolve_for(&data, 1).is_err(), "{params:?}");
        }
        let model = StretchModel::Layered(LayeredParams {
            contrast_recovery: 0.6,
            lightness_mask: false,
            ..LayeredParams::default()
        });
        let json = serde_json::to_string(&model).unwrap();
        assert!(json.contains("\"type\":\"layered\""), "{json}");
        assert_eq!(serde_json::from_str::<StretchModel>(&json).unwrap(), model);
        let terse: StretchModel = serde_json::from_str(
            r#"{"type":"layered","target_background":0.15,"aggressiveness":0.5}"#,
        )
        .unwrap();
        assert_eq!(terse, StretchModel::Layered(LayeredParams::default()));
        // The resolved plan serializes for provenance.
        let plan = layered(LayeredParams::default())
            .resolve_for(&data, 1)
            .unwrap();
        let resolved = serde_json::to_value(&plan).unwrap();
        assert_eq!(resolved["curves"][0]["type"], "layered");
    }
}
