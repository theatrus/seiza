//! Core compression on its own: bright structure's large-scale range taken
//! out before any stretch, as the HDR layered stretch takes it out before
//! its curve.
//!
//! Above the background, each value is scaled toward the median by a gain
//! that follows a blur of the log of the image, so a bright core's broad
//! glow comes down while the structure smaller than the blur keeps its
//! contrast. The blur is measured once from the whole image at
//! [`MAP_REDUCTION`] times less resolution and read back between its
//! pixels, as a layered stretch's maps are.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::layered::{blur, core_gain, core_log};
use crate::{LayeredMaps, MAP_REDUCTION, Result, StretchAnalysis, StretchError};

/// How much of bright structure's large-scale range is taken out, from
/// where, and at what size.
///
/// In JSON every field may be left out and takes its default.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CoreCompression {
    /// How much of the range in the log of luminance is taken out, in
    /// `[0, 1]` (default 0.5). 0 compresses nothing.
    pub amount: f64,
    /// Where compression starts, in robust sigmas above the background
    /// (default 100).
    pub threshold: f64,
    /// The size of the structure compressed, as a blur in pixels, from 1
    /// through 256 (default 12): about a third of the bright object's
    /// radius.
    pub scale: f64,
}

impl Default for CoreCompression {
    fn default() -> Self {
        Self {
            amount: 0.5,
            threshold: 100.0,
            scale: 12.0,
        }
    }
}

/// A core compression resolved for one image: its settings, and the
/// image's background median and noise, in the units the image's samples
/// are in when the compression reads them.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResolvedCore {
    pub params: CoreCompression,
    pub median: f64,
    pub noise: f64,
}

impl ResolvedCore {
    /// The compression of an image of `median` and lower-half `noise`.
    pub fn new(params: CoreCompression, median: f64, noise: f64) -> Result<Self> {
        let within =
            |value: f64, low: f64, high: f64| value.is_finite() && value >= low && value <= high;
        if !within(params.amount, 0.0, 1.0) {
            return Err(StretchError::Invalid(
                "core compression must be between 0 and 1".into(),
            ));
        }
        if !(params.threshold.is_finite() && params.threshold > 0.0) {
            return Err(StretchError::Invalid(
                "the core compression's threshold must be above 0".into(),
            ));
        }
        if !within(params.scale, 1.0, 256.0) {
            return Err(StretchError::Invalid(
                "the core compression's scale must be from 1 through 256 pixels".into(),
            ));
        }
        if !median.is_finite() || !noise.is_finite() {
            return Err(StretchError::Invalid(
                "the image has no finite background to compress above".into(),
            ));
        }
        Ok(Self {
            params,
            median,
            noise,
        })
    }

    /// The compression of the image `analysis` describes: its luminance's
    /// median and lower-half noise for RGB, its samples' for mono.
    pub fn from_analysis(params: CoreCompression, analysis: &StretchAnalysis) -> Result<Self> {
        let distribution = match analysis.channel_count {
            3 => analysis.luminance.as_ref().ok_or_else(|| {
                StretchError::Invalid("luminance contains no finite samples".into())
            })?,
            _ => &analysis.linked,
        };
        Self::new(
            params,
            distribution.statistics.median,
            distribution.lower_half_noise(),
        )
    }

    /// The blurred log the compression reads, measured from `reduced`: the
    /// image averaged by [`crate::reduce_luminance`], `width` by `height`
    /// map pixels. The maps hold it as their core, with the brightest
    /// value after the compression as their white.
    pub fn measure(&self, reduced: &[f32], width: usize, height: usize) -> LayeredMaps {
        if self.params.amount <= 0.0 {
            return LayeredMaps {
                width,
                height,
                white: reduced
                    .iter()
                    .copied()
                    .filter(|value| value.is_finite())
                    .fold(0.0_f32, f32::max),
                ..LayeredMaps::default()
            };
        }
        let logs: Vec<f32> = reduced
            .par_iter()
            .map(|&value| {
                if value.is_finite() {
                    core_log(f64::from(value), self.median, self.noise) as f32
                } else {
                    f32::NAN
                }
            })
            .collect();
        let core = blur(
            &logs,
            width,
            height,
            (self.params.scale / MAP_REDUCTION as f64).max(0.25),
        );
        let white = reduced
            .iter()
            .zip(&core)
            .map(|(&value, &base)| self.compress(value, Some(base)))
            .filter(|value| value.is_finite())
            .fold(0.0_f32, f32::max);
        LayeredMaps {
            width,
            height,
            core,
            large: Vec::new(),
            white,
        }
    }

    /// `value` with bright structure's range compressed, given the blurred
    /// log at its pixel; as it is without one.
    pub fn compress(&self, value: f32, base: Option<f32>) -> f32 {
        let Some(base) = base else {
            return value;
        };
        if !value.is_finite() || !base.is_finite() || self.params.amount <= 0.0 {
            return value;
        }
        let gain = core_gain(f64::from(base), self.params.amount, self.params.threshold);
        (self.median + (f64::from(value) - self.median) * gain) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce_luminance;

    #[test]
    fn a_bright_core_comes_down_and_the_background_stays() {
        // A sky of 0.1 with noise, and a broad bright core in the middle.
        let (width, height) = (128, 128);
        let mut image = vec![0.0_f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let (dx, dy) = (x as f32 - 64.0, y as f32 - 64.0);
                let core = 0.85 * (-(dx * dx + dy * dy) / (2.0 * 12.0 * 12.0)).exp();
                let noise = (((x * 7919 + y * 104_729) % 1000) as f32 / 1000.0 - 0.5) * 0.004;
                image[y * width + x] = 0.1 + core + noise;
            }
        }
        let analysis = StretchAnalysis::analyze(&image, 1, 100_000).unwrap();
        let core = ResolvedCore::from_analysis(
            CoreCompression {
                amount: 0.7,
                threshold: 20.0,
                scale: 12.0,
            },
            &analysis,
        )
        .unwrap();
        let (reduced, map_width, map_height) = reduce_luminance(&image, width, height);
        let maps = core.measure(&reduced, map_width, map_height);
        let at = |x: usize, y: usize| core.compress(image[y * width + x], maps.sample(x, y).core);
        let centre = at(64, 64);
        assert!(centre < 0.6, "the core came down to {centre}");
        assert!(centre > 0.1, "and stays above the sky at {centre}");
        let sky = at(4, 4);
        assert!((sky - image[4 * width + 4]).abs() < 1e-4, "the sky stays");
        assert!(maps.white < 0.95 && maps.white > 0.1);
        // Nothing to take out leaves every value as it is.
        let none = ResolvedCore {
            params: CoreCompression {
                amount: 0.0,
                ..core.params
            },
            ..core
        };
        assert_eq!(none.compress(0.8, Some(9.0)), 0.8);
    }
}
