//! Tone and color: Camera Raw's Basic controls for a finished picture.
//!
//! Exposure, Contrast, Highlights, Shadows, Whites and Blacks make one curve
//! of the luminance, [`Tone::curve`], which [`Tone::apply`] applies with each
//! pixel's color kept, before Vibrance and Saturation. A GPU that draws the
//! curve from a table fills it with [`Tone::curve_row`].

use crate::render::{contrast_curve, luma};
use serde::{Deserialize, Serialize};

/// The settings of a Tone and color step. Every control at 0 leaves the
/// picture as it is; a control left out of its serialized form is 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Tone {
    /// Stops, from -3 to 3.
    pub exposure: f32,
    /// From -1 to 1, as every control below.
    pub contrast: f32,
    pub highlights: f32,
    pub shadows: f32,
    pub whites: f32,
    pub blacks: f32,
    pub vibrance: f32,
    pub saturation: f32,
}

/// Each control: its field name, its label, and its range. [`Tone::held`]
/// keeps a value within its range.
pub const CONTROLS: [(&str, &str, f32, f32); 8] = [
    ("exposure", "Exposure", -3.0, 3.0),
    ("contrast", "Contrast", -1.0, 1.0),
    ("highlights", "Highlights", -1.0, 1.0),
    ("shadows", "Shadows", -1.0, 1.0),
    ("whites", "Whites", -1.0, 1.0),
    ("blacks", "Blacks", -1.0, 1.0),
    ("vibrance", "Vibrance", -1.0, 1.0),
    ("saturation", "Saturation", -1.0, 1.0),
];

/// How far Highlights or Shadows at 1 moves the tone it changes most. Any
/// larger and the two together could make the curve turn back.
const SHIFT: f32 = 0.14;
/// How far Whites at 1 brings the white point down.
const WHITE_REACH: f32 = 0.3;
/// How far Blacks at -1 brings the black point up.
const BLACK_REACH: f32 = 0.1;

impl Tone {
    /// The same settings with each control held to its range.
    pub fn held(self) -> Self {
        let hold = |value: f32, index: usize| {
            let (_, _, low, high) = CONTROLS[index];
            if value.is_finite() {
                value.clamp(low, high)
            } else {
                0.0
            }
        };
        Self {
            exposure: hold(self.exposure, 0),
            contrast: hold(self.contrast, 1),
            highlights: hold(self.highlights, 2),
            shadows: hold(self.shadows, 3),
            whites: hold(self.whites, 4),
            blacks: hold(self.blacks, 5),
            vibrance: hold(self.vibrance, 6),
            saturation: hold(self.saturation, 7),
        }
    }

    /// The step applied to one pixel: the curve of its luminance with its
    /// color kept and in gamut, then Vibrance and Saturation.
    pub fn apply(&self, color: [f32; 3]) -> [f32; 3] {
        let luminance = luma(color);
        let mapped = self.curve(luminance);
        let scaled = if luminance <= 1.0e-8 {
            [mapped; 3]
        } else {
            let mut scale = mapped / luminance;
            let maximum = color[0].max(color[1]).max(color[2]).max(0.0);
            if maximum * scale > 1.0 {
                scale = 1.0 / maximum;
            }
            color.map(|channel| (channel * scale).clamp(0.0, 1.0))
        };
        let tone = self.held();
        colorfulness(scaled, tone.vibrance, tone.saturation)
    }

    /// The curve as `width` entries, entry `i` at `(i / (width - 1))²`, for
    /// a GPU that reads it from a table indexed by the square root.
    pub fn curve_row(&self, width: usize) -> Vec<f32> {
        let last = width.saturating_sub(1).max(1) as f32;
        (0..width)
            .map(|index| self.curve((index as f32 / last).powi(2)))
            .collect()
    }

    /// The luminance curve at `x`, from 0 to 1.
    pub fn curve(&self, x: f32) -> f32 {
        let tone = self.held();
        let mut y = x.max(0.0) * tone.exposure.exp2();
        // Whites and Blacks move the white and black points.
        let white = 1.0 - WHITE_REACH * tone.whites;
        let black = -BLACK_REACH * tone.blacks;
        y = ((y - black) / (white - black)).clamp(0.0, 1.0);
        // Shadows and Highlights move the tones a third and two thirds of
        // the way up most; 6.75 makes each bump 1 at its peak.
        y += SHIFT * tone.shadows * 6.75 * y * (1.0 - y) * (1.0 - y);
        y += SHIFT * tone.highlights * 6.75 * y * y * (1.0 - y);
        contrast_curve(y.clamp(0.0, 1.0), tone.contrast)
    }
}

/// Vibrance and saturation, each from -1 to 1. Vibrance changes a muted
/// color more than a vivid one.
pub(crate) fn colorfulness(color: [f32; 3], vibrance: f32, saturation: f32) -> [f32; 3] {
    let luminance = luma(color);
    let high = color[0].max(color[1]).max(color[2]);
    let low = color[0].min(color[1]).min(color[2]);
    let vivid = if high > 1.0e-6 {
        ((high - low) / high).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let gain =
        (1.0 + saturation.clamp(-1.0, 1.0)) * (1.0 + vibrance.clamp(-1.0, 1.0) * (1.0 - vivid));
    color.map(|channel| (luminance + (channel - luminance) * gain).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn controls_at_zero_leave_the_picture_as_it_is() {
        let tone: Tone = serde_json::from_value(json!({})).unwrap();
        assert_eq!(tone, Tone::default());
        for x in [0.0_f32, 0.01, 0.2, 0.5, 0.9, 1.0] {
            assert!((tone.curve(x) - x).abs() < 1e-6, "{x}");
        }
    }

    #[test]
    fn each_control_moves_the_tones_it_names() {
        let at = |settings: serde_json::Value, x: f32| {
            serde_json::from_value::<Tone>(settings).unwrap().curve(x)
        };
        assert!((at(json!({ "exposure": 1.0 }), 0.2) - 0.4).abs() < 1e-6);
        assert!(at(json!({ "shadows": 1.0 }), 0.3) > 0.35);
        assert!((at(json!({ "shadows": 1.0 }), 0.9) - 0.9).abs() < 0.02);
        assert!(at(json!({ "highlights": -1.0 }), 0.7) < 0.65);
        assert!((at(json!({ "highlights": -1.0 }), 0.1) - 0.1).abs() < 0.02);
        // Whites up clips the brightest tones; Blacks down clips the darkest.
        assert_eq!(at(json!({ "whites": 1.0 }), 0.75), 1.0);
        assert_eq!(at(json!({ "blacks": -1.0 }), 0.05), 0.0);
        // Contrast darkens below the middle and brightens above it.
        assert!(at(json!({ "contrast": 1.0 }), 0.25) < 0.25);
        assert!(at(json!({ "contrast": 1.0 }), 0.75) > 0.75);
    }

    #[test]
    fn the_curve_never_turns_back() {
        for values in [
            [1.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0, 1.0],
        ] {
            let [highlights, shadows, whites, blacks] = values;
            let tone = Tone {
                highlights,
                shadows,
                whites,
                blacks,
                contrast: values[0],
                ..Tone::default()
            };
            let row = tone.curve_row(4096);
            for pair in row.windows(2) {
                assert!(pair[1] >= pair[0] - 1e-6, "{values:?}");
            }
        }
    }

    #[test]
    fn settings_out_of_range_are_held_to_it() {
        let tone: Tone =
            serde_json::from_value(json!({ "exposure": 9.0, "contrast": -4.0 })).unwrap();
        let held = tone.held();
        assert_eq!(held.exposure, 3.0);
        assert_eq!(held.contrast, -1.0);
        assert_eq!(tone.curve(0.1), held.curve(0.1));
    }
}
