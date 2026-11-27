//! An HDR tail on a stretch curve: above a knee, the curve continues into
//! the headroom above SDR white instead of flattening toward 1, so the
//! brightest part of an image keeps its structure on an HDR display.
//!
//! The curve's output is read as a value on the extended sRGB curve, as an
//! HDR picture's values are. Below the knee the curve is as it was; above
//! it, the display light rises as a power of the input above the curve's
//! black, starting with the curve's own slope and relaxing toward the power
//! that reaches 2 to the headroom times SDR white's light at the white. The
//! join is smooth: the value and its slope match at the knee.

use crate::ResolvedCurve;
use serde::{Deserialize, Serialize};

/// The value on the extended sRGB curve of `light`, linear light as a
/// multiple of SDR white's: the sRGB curve, continued past 1.
pub fn extended_srgb_encode(light: f32) -> f32 {
    if light <= 0.003_130_8 {
        12.92 * light.max(0.0)
    } else {
        1.055 * light.powf(1.0 / 2.4) - 0.055
    }
}

/// The light of `value` on the extended sRGB curve: the inverse of
/// [`extended_srgb_encode`].
pub fn extended_srgb_decode(value: f32) -> f32 {
    if value <= 0.040_45 {
        value.max(0.0) / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// How an HDR tail is asked for.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HdrTailParams {
    /// The curve's output where the tail starts, from 0.3 to 0.95.
    pub knee: f64,
    /// How far above SDR white the tail reaches at the white, in stops,
    /// from 0 to 4. At 0 the tail rolls off smoothly to 1 instead of the
    /// curve's clip.
    pub headroom: f64,
    /// The input, in the curve's unit domain, that reaches the top of the
    /// headroom: the brightest part of the image to keep.
    pub white: f64,
    /// How quickly the tail turns from the curve's slope to its own, in
    /// natural-log units of the input; 1 by default.
    pub relax: f64,
}

impl Default for HdrTailParams {
    fn default() -> Self {
        Self {
            knee: 0.75,
            headroom: 2.0,
            white: 1.0,
            relax: 1.0,
        }
    }
}

/// An HDR tail resolved for one curve.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HdrTail {
    /// The curve's black: the largest input it maps to 0.
    black: f64,
    /// The input above the black where the tail starts.
    knee_input: f64,
    /// The natural log of the display light at the knee.
    knee_light: f64,
    /// The curve's slope at the knee, in log light per log input.
    slope: f64,
    /// The tail's own slope, in the same units.
    power: f64,
    relax: f64,
    /// The brightest value the tail gives.
    peak: f32,
}

impl HdrTail {
    /// The tail of `curve` that `params` ask for. `None` when the curve
    /// never reaches the knee below the white, or the white is at or below
    /// the knee: the curve is then left as it is.
    pub fn resolve(curve: ResolvedCurve, params: HdrTailParams) -> Option<Self> {
        let knee = params.knee.clamp(0.3, 0.95);
        let headroom = params.headroom.clamp(0.0, 4.0);
        let relax = params.relax.max(0.05);
        let map = |x: f64| f64::from(curve.map(x as f32));
        let white = params.white;
        if !white.is_finite() || white <= 0.0 || map(white) < knee {
            return None;
        }
        // The curve rises from its black to the knee below the white.
        let black = bisect(0.0, white, |x| map(x) > 0.0);
        let knee_at = bisect(black, white, |x| map(x) >= knee);
        let knee_input = knee_at - black;
        if knee_input <= 0.0 || white - black <= knee_input * 1.0001 {
            return None;
        }
        let light = |x: f64| {
            f64::from(extended_srgb_decode(map(x) as f32))
                .max(1e-12)
                .ln()
        };
        let knee_light = f64::from(extended_srgb_decode(knee as f32)).ln();
        // The curve's slope at the knee in log light per log input, from
        // just below it, where the curve still is the curve.
        let step: f64 = 0.02;
        let below = black + knee_input * (-step).exp();
        let slope = ((knee_light - light(below)) / step).max(1e-3);
        let span = ((white - black) / knee_input).ln();
        let wanted = headroom * std::f64::consts::LN_2 - knee_light;
        let turned = relax * (1.0 - (-span / relax).exp());
        let power = ((wanted - slope * turned) / (span - turned)).max(1e-3);
        Some(Self {
            black,
            knee_input,
            knee_light,
            slope,
            power,
            relax,
            peak: extended_srgb_encode(headroom.exp2() as f32),
        })
    }

    /// The brightest value the tail gives: the value on the extended sRGB
    /// curve of 2 to the headroom.
    pub fn peak(&self) -> f32 {
        self.peak
    }

    /// `curve` with this tail: the curve below the knee, the tail above it.
    pub fn map(&self, curve: ResolvedCurve, value: f32) -> f32 {
        let above = f64::from(value) - self.black;
        if above.is_nan() || above <= self.knee_input {
            return curve.map(value);
        }
        let u = (above / self.knee_input).ln();
        let log_light = self.knee_light
            + self.power * u
            + (self.slope - self.power) * self.relax * (1.0 - (-u / self.relax).exp());
        extended_srgb_encode(log_light.exp() as f32).clamp(0.0, self.peak)
    }
}

/// The point between `low` and `high` where `reached` turns true, for a
/// `reached` that is false at `low` and stays true once true.
fn bisect(mut low: f64, mut high: f64, reached: impl Fn(f64) -> bool) -> f64 {
    if reached(low) {
        return low;
    }
    for _ in 0..60 {
        let middle = 0.5 * (low + high);
        if reached(middle) {
            high = middle;
        } else {
            low = middle;
        }
    }
    high
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{StretchAnalysis, StretchConfig, StretchModel};

    /// A median anchor stretch resolved for a faint sky with a bright core.
    fn curve() -> ResolvedCurve {
        let mut samples: Vec<f32> = (0..20_000)
            .map(|index| 0.01 + 0.002 * ((index * 7919 % 1000) as f32 / 1000.0))
            .collect();
        samples.extend((0..200).map(|index| 0.05 + index as f32 / 200.0 * 0.9));
        let analysis = StretchAnalysis::analyze(&samples, 1, samples.len()).unwrap();
        let config = StretchConfig {
            model: StretchModel::MedianAnchor(Default::default()),
            color_strategy: Default::default(),
            max_analysis_samples: samples.len(),
        };
        config.resolve(&analysis).unwrap().curves()[0]
    }

    #[test]
    fn below_the_knee_the_curve_is_as_it_was() {
        let curve = curve();
        let tail = HdrTail::resolve(curve, HdrTailParams::default()).unwrap();
        for x in [0.0_f32, 0.011, 0.012, 0.02, 0.03] {
            if curve.map(x) < 0.74 {
                assert_eq!(tail.map(curve, x), curve.map(x), "{x}");
            }
        }
    }

    #[test]
    fn the_tail_joins_smoothly_and_reaches_the_peak_at_the_white() {
        let curve = curve();
        let params = HdrTailParams {
            headroom: 3.0,
            ..Default::default()
        };
        let tail = HdrTail::resolve(curve, params).unwrap();
        assert!((tail.peak() - 2.454).abs() < 1e-3, "{}", tail.peak());
        // At the white, the top of the headroom.
        assert!((tail.map(curve, 1.0) - tail.peak()).abs() < 2e-3);
        // Rising all the way.
        let mut last = 0.0;
        for index in 0..=4000 {
            let x = index as f32 / 4000.0;
            let value = tail.map(curve, x);
            assert!(value >= last, "{x}: {value} after {last}");
            last = value;
        }
        // No kink at the knee: the slope just above matches the slope just
        // below, in log light per log input.
        let knee = tail.black + tail.knee_input;
        let log_light = |x: f64| f64::from(extended_srgb_decode(tail.map(curve, x as f32))).ln();
        let h = 0.002;
        let below = (log_light(knee * (1.0 - h)) - log_light(knee * (1.0 - 3.0 * h))) / (2.0 * h);
        let above = (log_light(knee * (1.0 + 3.0 * h)) - log_light(knee * (1.0 + h))) / (2.0 * h);
        assert!(
            (above / below - 1.0).abs() < 0.15,
            "{below} below, {above} above"
        );
    }

    #[test]
    fn no_headroom_rolls_off_to_white() {
        let curve = curve();
        let params = HdrTailParams {
            headroom: 0.0,
            ..Default::default()
        };
        let tail = HdrTail::resolve(curve, params).unwrap();
        assert!((tail.peak() - 1.0).abs() < 1e-6);
        assert!(tail.map(curve, 0.5) < 1.0);
        assert!((tail.map(curve, 1.0) - 1.0).abs() < 2e-3);
    }

    #[test]
    fn a_white_below_the_knee_leaves_the_curve_alone() {
        let curve = curve();
        let params = HdrTailParams {
            white: 0.01,
            ..Default::default()
        };
        assert!(HdrTail::resolve(curve, params).is_none());
    }

    #[test]
    fn the_extended_curve_inverts() {
        for light in [0.0_f32, 0.002, 0.18, 1.0, 4.0, 16.0] {
            let back = extended_srgb_decode(extended_srgb_encode(light));
            assert!(
                (back - light).abs() < 1e-4 * light.max(1.0),
                "{light}: {back}"
            );
        }
    }
}
