//! Tone and color: Camera Raw's Basic controls for a finished picture.
//!
//! Exposure, Whites and Blacks set the luminance's range. Highlights and
//! Shadows move the tones of each pixel's neighborhood rather than of the
//! pixel itself, so a bright star in a dark field keeps its contrast while
//! the field is lifted. Texture and Clarity raise or lower the detail
//! around a pixel at two scales, and Dehaze takes away a veil estimated
//! from the darkest channel nearby. Contrast, then Vibrance and Saturation,
//! follow. The luminance is changed with each pixel's color kept.
//!
//! A neighborhood is described by [`Neighborhood`]: means over a square
//! window of the luminance, its square, the darkest channel and the
//! product of the two. [`neighborhoods`] computes them for a picture, and
//! [`Tone::apply_around`] applies the step with them. From those means the
//! step takes an edge-preserving base, as a guided filter does: the mean,
//! drawn toward the pixel where the window's variance is large. So a
//! star's edge is its own base and gets no halo. [`Tone::apply`] is the
//! step with a neighborhood of the pixel alone, which no local control
//! changes: a curve of the pixel's luminance, [`Tone::curve`].

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
    /// Fine detail, over [`DETAIL_RADIUS`].
    pub texture: f32,
    /// Local contrast in the midtones, over [`LOCAL_RADIUS`].
    pub clarity: f32,
    /// Positive takes away a veil over the picture; negative adds one.
    pub dehaze: f32,
    pub vibrance: f32,
    pub saturation: f32,
}

/// Each control: its field name, its label, and its range, in Camera Raw's
/// order. [`Tone::held`] keeps a value within its range.
pub const CONTROLS: [(&str, &str, f32, f32); 11] = [
    ("exposure", "Exposure", -3.0, 3.0),
    ("contrast", "Contrast", -1.0, 1.0),
    ("highlights", "Highlights", -1.0, 1.0),
    ("shadows", "Shadows", -1.0, 1.0),
    ("whites", "Whites", -1.0, 1.0),
    ("blacks", "Blacks", -1.0, 1.0),
    ("texture", "Texture", -1.0, 1.0),
    ("clarity", "Clarity", -1.0, 1.0),
    ("dehaze", "Dehaze", -1.0, 1.0),
    ("vibrance", "Vibrance", -1.0, 1.0),
    ("saturation", "Saturation", -1.0, 1.0),
];

/// The half-width, in pixels, of the window Highlights, Shadows, Clarity
/// and Dehaze read: a square of `2 × LOCAL_RADIUS + 1` pixels.
pub const LOCAL_RADIUS: usize = 48;
/// The half-width, in pixels, of the window Texture reads.
pub const DETAIL_RADIUS: usize = 3;

/// How far Highlights or Shadows at 1 moves the tone it changes most. Any
/// larger and the two together could make the curve turn back.
const SHIFT: f32 = 0.14;
/// How far Whites at 1 brings the white point down.
const WHITE_REACH: f32 = 0.3;
/// How far Blacks at -1 brings the black point up.
const BLACK_REACH: f32 = 0.1;
/// The variance of luminance, over [`LOCAL_RADIUS`], at which a window
/// counts as half edge: below it the base is the window's mean, above it
/// the pixel. A standard deviation of 0.05.
const LOCAL_EDGE: f32 = 0.0025;
/// The same over [`DETAIL_RADIUS`], a standard deviation of 0.03: detail
/// smaller than this, such as noise in a flat sky, is left as it is.
const DETAIL_EDGE: f32 = 0.0009;
/// How much Clarity at 1 adds of the detail around the local base.
const CLARITY_GAIN: f32 = 1.0;
/// How much Texture at 1 adds of the detail around the fine base.
const TEXTURE_GAIN: f32 = 1.5;
/// The share of the veil Dehaze at 1 takes away.
const DEHAZE_STRENGTH: f32 = 0.95;
/// How far Dehaze at -1 draws each pixel toward its local base.
const HAZE: f32 = 0.5;

/// Means over the square windows around one pixel of a picture: over
/// [`LOCAL_RADIUS`], of the luminance, its square, the darkest channel and
/// the luminance times the darkest channel; over [`DETAIL_RADIUS`], of the
/// luminance and its square.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Neighborhood {
    pub local: [f32; 4],
    pub detail: [f32; 2],
}

impl Neighborhood {
    /// The neighborhood of a pixel alone.
    pub fn of_pixel(color: [f32; 3]) -> Self {
        let [luminance, darkest] = sample(color);
        Self {
            local: [
                luminance,
                luminance * luminance,
                darkest,
                luminance * darkest,
            ],
            detail: [luminance, luminance * luminance],
        }
    }
}

/// The luminance and the darkest channel of a pixel, 0 where it is not
/// covered.
fn sample(color: [f32; 3]) -> [f32; 2] {
    if color.iter().all(|channel| channel.is_finite()) {
        [luma(color), color[0].min(color[1]).min(color[2])]
    } else {
        [0.0; 2]
    }
}

/// Each pixel's [`Neighborhood`] in a picture of `width` by `height`
/// pixels, red, green and blue interleaved. A window near the picture's
/// edge holds only the pixels inside it.
pub fn neighborhoods(rgb: &[f32], width: usize, height: usize) -> Vec<Neighborhood> {
    let samples: Vec<[f32; 2]> = rgb
        .as_chunks::<3>()
        .0
        .iter()
        .map(|pixel| sample(*pixel))
        .collect();
    let plane = |value: &dyn Fn([f32; 2]) -> f32| -> Vec<f32> {
        samples.iter().map(|sample| value(*sample)).collect()
    };
    let luminance = plane(&|[l, _]| l);
    let squared = plane(&|[l, _]| l * l);
    let local = [
        box_mean(&luminance, width, height, LOCAL_RADIUS),
        box_mean(&squared, width, height, LOCAL_RADIUS),
        box_mean(&plane(&|[_, d]| d), width, height, LOCAL_RADIUS),
        box_mean(&plane(&|[l, d]| l * d), width, height, LOCAL_RADIUS),
    ];
    let detail = [
        box_mean(&luminance, width, height, DETAIL_RADIUS),
        box_mean(&squared, width, height, DETAIL_RADIUS),
    ];
    (0..samples.len())
        .map(|at| Neighborhood {
            local: local.each_ref().map(|plane| plane[at]),
            detail: detail.each_ref().map(|plane| plane[at]),
        })
        .collect()
}

/// The mean of `plane` over the square of `2 × radius + 1` pixels around
/// each pixel, the square cut at the plane's edges: a mean across, then a
/// mean down. Each window is summed afresh in 64 bits, in order, so a
/// pixel's mean depends only on its window, wherever a tile starts.
fn box_mean(plane: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    use rayon::prelude::*;
    if plane.is_empty() {
        return Vec::new();
    }
    let window = |index: usize, count: usize| {
        (
            index.saturating_sub(radius),
            (index + radius).min(count - 1),
        )
    };
    let mut across = vec![0.0_f32; plane.len()];
    across
        .par_chunks_mut(width)
        .zip(plane.par_chunks(width))
        .for_each(|(out, row)| {
            for (x, mean) in out.iter_mut().enumerate() {
                let (low, high) = window(x, width);
                let sum: f64 = row[low..=high].iter().map(|value| f64::from(*value)).sum();
                *mean = (sum / (high - low + 1) as f64) as f32;
            }
        });
    let mut down = vec![0.0_f32; plane.len()];
    down.par_chunks_mut(width).enumerate().for_each(|(y, out)| {
        let (low, high) = window(y, height);
        for (x, mean) in out.iter_mut().enumerate() {
            let sum: f64 = (low..=high)
                .map(|row| f64::from(across[row * width + x]))
                .sum();
            *mean = (sum / (high - low + 1) as f64) as f32;
        }
    });
    down
}

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
            texture: hold(self.texture, 6),
            clarity: hold(self.clarity, 7),
            dehaze: hold(self.dehaze, 8),
            vibrance: hold(self.vibrance, 9),
            saturation: hold(self.saturation, 10),
        }
    }

    /// Whether the step reads each pixel's neighborhood: whether a control
    /// that does is set.
    pub fn is_local(&self) -> bool {
        let tone = self.held();
        [
            tone.highlights,
            tone.shadows,
            tone.texture,
            tone.clarity,
            tone.dehaze,
        ]
        .iter()
        .any(|value| *value != 0.0)
    }

    /// The step applied to one pixel with a neighborhood of the pixel
    /// alone.
    pub fn apply(&self, color: [f32; 3]) -> [f32; 3] {
        let [luminance, _] = sample(color);
        self.apply_around(color, luminance, &Neighborhood::of_pixel(color))
    }

    /// The step applied to one pixel, `color`, whose neighborhood is
    /// `around`. `luminance` is the luminance the neighborhood was measured
    /// with at this pixel; it differs from `color`'s where a heal changed
    /// the pixel after the neighborhood was measured.
    pub fn apply_around(&self, color: [f32; 3], luminance: f32, around: &Neighborhood) -> [f32; 3] {
        let tone = self.held();
        let [mean, mean_squared, dark, mean_product] = around.local;
        // The edge-preserving bases, and the veil: the darkest channel's
        // base, guided by the luminance.
        let variance = (mean_squared - mean * mean).max(0.0);
        let edge = variance / (variance + LOCAL_EDGE);
        let mut base = mean + edge * (luminance - mean);
        let guided = (mean_product - mean * dark) / (variance + LOCAL_EDGE);
        // No brighter than the window's darkest channel on average: a large
        // bright area, such as a galaxy's core, is not a veil.
        let veil = (dark + guided * (luminance - mean))
            .min(dark)
            .clamp(0.0, 1.0);
        let [fine_mean, fine_squared] = around.detail;
        let fine_variance = (fine_squared - fine_mean * fine_mean).max(0.0);
        let detailed = fine_variance / (fine_variance + DETAIL_EDGE);
        let mut fine = fine_mean + detailed * (luminance - fine_mean);

        let mut color = color;
        let mut own = luma(color);
        // Dehaze changes each channel by one line, so the luminance and its
        // bases follow it by the same line.
        let line = if tone.dehaze > 0.0 {
            let lifted = DEHAZE_STRENGTH * tone.dehaze * veil;
            let scale = 1.0 / (1.0 - lifted);
            Some((scale, -lifted * scale))
        } else if tone.dehaze < 0.0 {
            let toward = HAZE * -tone.dehaze;
            Some((1.0 - toward, toward * base))
        } else {
            None
        };
        if let Some((scale, offset)) = line {
            let map = |value: f32| scale.mul_add(value, offset);
            color = color.map(|channel| map(channel).max(0.0));
            own = map(own);
            base = map(base);
            fine = map(fine);
        }

        // Exposure, Whites and Blacks: one line too.
        let gain = tone.exposure.exp2();
        let white = 1.0 - WHITE_REACH * tone.whites;
        let black = -BLACK_REACH * tone.blacks;
        let range = |value: f32| (value.max(0.0) * gain - black) / (white - black);
        let pixel = range(own);
        let local = range(base);
        let toned = local.clamp(0.0, 1.0);
        // Shadows and Highlights move the base's tones a third and two
        // thirds of the way up most, keeping the detail around it; 6.75
        // makes each bump 1 at its peak. Highlights moves the tone Shadows
        // left.
        let lifted = toned + SHIFT * tone.shadows * 6.75 * toned * (1.0 - toned) * (1.0 - toned);
        let moved = lifted + SHIFT * tone.highlights * 6.75 * lifted * lifted * (1.0 - lifted);
        let mut value = pixel + (moved - toned);
        // Clarity scales the detail around the local base, most in the
        // midtones; Texture the detail around the fine base, where there is
        // more of it than noise.
        value += tone.clarity * CLARITY_GAIN * (pixel - local) * 4.0 * toned * (1.0 - toned);
        value += tone.texture * TEXTURE_GAIN * detailed * (pixel - range(fine));
        let mapped = contrast_curve(value.clamp(0.0, 1.0), tone.contrast);

        let luminance = luma(color);
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
        colorfulness(scaled, tone.vibrance, tone.saturation)
    }

    /// The luminance curve of a pixel on its own at `x`, from 0 to 1.
    pub fn curve(&self, x: f32) -> f32 {
        let gray = x.max(0.0);
        luma(self.apply([gray; 3]))
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
            let row: Vec<f32> = (0..4096)
                .map(|index| tone.curve((index as f32 / 4095.0).powi(2)))
                .collect();
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

    /// A gray picture of `width` by `height` with `value(x, y)` at each
    /// pixel, and each pixel's neighborhood in it.
    fn gray(
        width: usize,
        height: usize,
        value: impl Fn(usize, usize) -> f32,
    ) -> (Vec<f32>, Vec<Neighborhood>) {
        let rgb: Vec<f32> = (0..width * height)
            .flat_map(|at| [value(at % width, at / width); 3])
            .collect();
        let around = neighborhoods(&rgb, width, height);
        (rgb, around)
    }

    fn at(tone: &Tone, rgb: &[f32], around: &[Neighborhood], index: usize) -> [f32; 3] {
        let color = [rgb[index * 3], rgb[index * 3 + 1], rgb[index * 3 + 2]];
        tone.apply_around(color, luma(color), &around[index])
    }

    #[test]
    fn box_means_match_a_sum_over_each_window() {
        let (width, height, radius) = (13, 7, 3);
        let plane: Vec<f32> = (0..width * height)
            .map(|at| ((at * 7919) % 101) as f32 / 100.0)
            .collect();
        let means = box_mean(&plane, width, height, radius);
        for y in 0..height {
            for x in 0..width {
                let (mut sum, mut count) = (0.0, 0.0);
                for v in y.saturating_sub(radius)..=(y + radius).min(height - 1) {
                    for u in x.saturating_sub(radius)..=(x + radius).min(width - 1) {
                        sum += plane[v * width + u];
                        count += 1.0;
                    }
                }
                assert!((means[y * width + x] - sum / count).abs() < 1e-6, "{x},{y}");
            }
        }
    }

    #[test]
    fn a_pixel_alone_is_the_plain_curve() {
        let tone = Tone {
            shadows: 0.7,
            highlights: -0.4,
            clarity: 1.0,
            texture: 1.0,
            ..Tone::default()
        };
        assert!(tone.is_local());
        assert!(!Tone::default().is_local());
        // Alone, a pixel has no detail around it: Clarity and Texture do
        // nothing, and Shadows and Highlights move its own tone.
        let plain = Tone {
            clarity: 0.0,
            texture: 0.0,
            ..tone
        };
        for x in [0.02_f32, 0.2, 0.5, 0.8] {
            assert!((tone.curve(x) - plain.curve(x)).abs() < 1e-6, "{x}");
        }
        assert!(tone.curve(0.3) > 0.32);
    }

    /// The per-pixel curve before neighborhoods, as it was written.
    fn earlier_curve(tone: &Tone, x: f32) -> f32 {
        let mut y = x.max(0.0) * tone.exposure.exp2();
        let white = 1.0 - WHITE_REACH * tone.whites;
        let black = -BLACK_REACH * tone.blacks;
        y = ((y - black) / (white - black)).clamp(0.0, 1.0);
        y += SHIFT * tone.shadows * 6.75 * y * (1.0 - y) * (1.0 - y);
        y += SHIFT * tone.highlights * 6.75 * y * y * (1.0 - y);
        contrast_curve(y.clamp(0.0, 1.0), tone.contrast)
    }

    #[test]
    fn a_pixel_alone_follows_the_earlier_curve() {
        for (shadows, highlights) in [(1.0, 1.0), (-1.0, 1.0), (0.4, -0.3), (1.0, -1.0)] {
            let tone = Tone {
                exposure: 0.5,
                contrast: 0.3,
                shadows,
                highlights,
                whites: 0.2,
                blacks: -0.2,
                ..Tone::default()
            };
            for index in 0..=200 {
                let x = index as f32 / 160.0;
                let (now, before) = (tone.curve(x), earlier_curve(&tone, x));
                assert!(
                    (now - before).abs() < 1e-6,
                    "{shadows} {highlights} at {x}: {now} {before}"
                );
            }
        }
    }

    #[test]
    fn dehaze_keeps_a_large_bright_area() {
        // A gray square of 0.9, 41 pixels across, in a sky of 0.1.
        let (width, height) = (121, 121);
        let (rgb, around) = gray(width, height, |x, y| {
            if (40..81).contains(&x) && (40..81).contains(&y) {
                0.9
            } else {
                0.1
            }
        });
        let dehaze = Tone {
            dehaze: 1.0,
            ..Tone::default()
        };
        let core = at(&dehaze, &rgb, &around, 60 * width + 60)[0];
        assert!(core > 0.8, "the core stays bright: {core}");
        let sky = at(&dehaze, &rgb, &around, 5 * width + 5)[0];
        assert!(sky < 0.02, "the sky's veil goes: {sky}");
    }

    #[test]
    fn shadows_lift_a_dark_field_and_keep_a_star_s_contrast() {
        // A dark field with one bright pixel in the middle.
        let (width, height) = (61, 61);
        let middle = 30 * width + 30;
        let (rgb, around) = gray(
            width,
            height,
            |x, y| {
                if (x, y) == (30, 30) { 0.8 } else { 0.1 }
            },
        );
        let shadows = Tone {
            shadows: 1.0,
            ..Tone::default()
        };
        let field = at(&shadows, &rgb, &around, 0)[0];
        let star = at(&shadows, &rgb, &around, middle)[0];
        assert!(field > 0.15, "the field is lifted: {field}");
        // On its own a pixel of 0.8 would barely move; in a dark field it
        // moves with the field, so the star stays as far above it.
        assert!(star - field > 0.8 - 0.1 - 0.02, "{star} over {field}");
    }

    #[test]
    fn clarity_and_texture_raise_detail_around_their_bases() {
        // A smooth ramp with a faint ripple of 3 pixels across it.
        let (width, height) = (120, 9);
        let ripple = |x: usize| if x.is_multiple_of(3) { 0.03 } else { 0.0 };
        let (rgb, around) = gray(width, height, |x, _| {
            0.3 + 0.3 * x as f32 / 120.0 + ripple(x)
        });
        let spread = |tone: &Tone| {
            let row: Vec<f32> = (40..80)
                .map(|x| at(tone, &rgb, &around, 4 * width + x)[0])
                .collect();
            row.windows(3).map(|w| (w[0] - w[1]).abs()).sum::<f32>()
        };
        let base = spread(&Tone::default());
        assert!(
            spread(&Tone {
                texture: 1.0,
                ..Tone::default()
            }) > base * 1.2
        );
        assert!(
            spread(&Tone {
                texture: -1.0,
                ..Tone::default()
            }) < base * 0.9
        );
        assert!(
            spread(&Tone {
                clarity: 1.0,
                ..Tone::default()
            }) > base * 1.2
        );
    }

    #[test]
    fn dehaze_takes_away_a_gray_veil_and_keeps_a_colored_glow() {
        // A gray veil of 0.2 with a red glow on its right half.
        let (width, height) = (60, 9);
        let rgb: Vec<f32> = (0..width * height)
            .flat_map(|at| {
                if at % width < 30 {
                    [0.2, 0.2, 0.2]
                } else {
                    [0.6, 0.2, 0.2]
                }
            })
            .collect();
        let around = neighborhoods(&rgb, width, height);
        let dehaze = Tone {
            dehaze: 1.0,
            ..Tone::default()
        };
        let sky = at(&dehaze, &rgb, &around, 4 * width + 5);
        let glow = at(&dehaze, &rgb, &around, 4 * width + 55);
        assert!(sky[0] < 0.05, "the veil goes: {sky:?}");
        // The glow keeps more red than green: its color grows stronger.
        assert!(glow[0] > 0.4 && glow[1] < 0.05, "{glow:?}");
        // Negative Dehaze adds a veil: the sky's contrast with the glow
        // falls.
        let haze = Tone {
            dehaze: -1.0,
            ..Tone::default()
        };
        let sky = at(&haze, &rgb, &around, 4 * width + 25);
        let glow = at(&haze, &rgb, &around, 4 * width + 35);
        assert!(luma(glow) - luma(sky) < 0.8 * (luma([0.6, 0.2, 0.2]) - 0.2));
    }
}
