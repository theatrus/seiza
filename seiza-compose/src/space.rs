//! The color space a picture is composed in, and the way from it to a
//! display's.
//!
//! Every space here has D65 white and the sRGB transfer curve, extended past
//! 1 for HDR as [`crate::tone::extended_srgb_encode`] gives it; they differ
//! in their primaries. A picture's values are encoded on that curve, so a
//! change of space decodes, converts the light through a matrix and encodes
//! again.

use crate::tone::{extended_srgb_decode, extended_srgb_encode};

/// The primaries a picture's red, green and blue are.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum WorkingSpace {
    /// sRGB's, the narrowest: what a picture is in unless its host says
    /// otherwise.
    #[default]
    Srgb,
    /// Display P3's, as Apple's wide-gamut displays show.
    DisplayP3,
    /// ITU-R BT.2020's, wide enough to keep any color a display can show,
    /// as Apple's `ITUR_2020_sRGBGamma` space pairs them with the sRGB curve.
    Rec2020,
}

impl WorkingSpace {
    /// Linear red, green and blue to CIE XYZ, by rows.
    pub fn to_xyz(self) -> [[f32; 3]; 3] {
        match self {
            Self::Srgb => [
                [0.412_390_8, 0.357_584_3, 0.180_480_8],
                [0.212_639, 0.715_168_7, 0.072_192_3],
                [0.019_330_8, 0.119_194_8, 0.950_532_2],
            ],
            Self::DisplayP3 => [
                [0.486_570_9, 0.265_667_7, 0.198_217_3],
                [0.228_974_6, 0.691_738_5, 0.079_286_9],
                [0.0, 0.045_113_4, 1.043_944_4],
            ],
            Self::Rec2020 => [
                [0.636_958, 0.144_616_9, 0.168_881],
                [0.262_700_2, 0.677_998_1, 0.059_301_7],
                [0.0, 0.028_072_7, 1.060_985_1],
            ],
        }
    }

    /// The weights that give a color's luminance from its red, green and
    /// blue: the middle row of [`Self::to_xyz`], with sRGB's and Rec.
    /// 2020's as ITU-R BT.709 and BT.2020 round them, so a picture in sRGB
    /// renders as it did before pictures had a space.
    pub fn luma_weights(self) -> [f32; 3] {
        match self {
            Self::Srgb => [0.2126, 0.7152, 0.0722],
            Self::DisplayP3 => self.to_xyz()[1],
            Self::Rec2020 => [0.2627, 0.6780, 0.0593],
        }
    }

    /// A color's luminance, with this space's weights.
    pub fn luma(self, color: [f32; 3]) -> f32 {
        let [r, g, b] = self.luma_weights();
        r.mul_add(color[0], g.mul_add(color[1], b * color[2]))
    }

    /// Linear light in this space to linear light in `target`, by rows.
    pub fn matrix_to(self, target: WorkingSpace) -> [[f32; 3]; 3] {
        multiply(inverse(target.to_xyz()), self.to_xyz())
    }

    /// `color`, encoded in this space, encoded in `target` instead: the
    /// same light. A color outside `target`'s gamut keeps its negative or
    /// out-of-range channels' light as the extended curve allows; use
    /// [`Self::map_to_display`] for a display.
    pub fn convert(self, color: [f32; 3], target: WorkingSpace) -> [f32; 3] {
        let light = apply(self.matrix_to(target), color.map(signed_decode));
        light.map(signed_encode)
    }

    /// `color`, encoded in this space, for a display whose primaries are
    /// `target`'s: converted, then brought inside the display's gamut. A
    /// color the display cannot show keeps its luminance and its hue as it
    /// is desaturated toward gray until no channel is negative. Values above
    /// 1, HDR's highlights, are kept.
    pub fn map_to_display(self, color: [f32; 3], target: WorkingSpace) -> [f32; 3] {
        let light = apply(
            self.matrix_to(target),
            color.map(|value| extended_srgb_decode(value.max(0.0))),
        );
        within_gamut(light, target.luma_weights()).map(extended_srgb_encode)
    }
}

/// Linear `light` with no channel below 0: the color desaturated toward the
/// gray of its luminance just enough. A color with no luminance is black.
pub fn within_gamut(light: [f32; 3], weights: [f32; 3]) -> [f32; 3] {
    let least = light[0].min(light[1]).min(light[2]);
    if least >= 0.0 {
        return light;
    }
    let luminance = weights[0].mul_add(
        light[0],
        weights[1].mul_add(light[1], weights[2] * light[2]),
    );
    if luminance <= 0.0 {
        return [0.0; 3];
    }
    // Along the line from gray to the color, the point where the lowest
    // channel reaches 0.
    let share = luminance / (luminance - least);
    light.map(|channel| (luminance + share * (channel - luminance)).max(0.0))
}

/// The extended sRGB decode, mirrored for negative values.
fn signed_decode(value: f32) -> f32 {
    extended_srgb_decode(value.abs()).copysign(value)
}

/// The extended sRGB encode, mirrored for negative light.
fn signed_encode(light: f32) -> f32 {
    extended_srgb_encode(light.abs()).copysign(light)
}

fn apply(matrix: [[f32; 3]; 3], color: [f32; 3]) -> [f32; 3] {
    matrix.map(|row| row[0].mul_add(color[0], row[1].mul_add(color[1], row[2] * color[2])))
}

fn multiply(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| (0..3).map(|k| a[row][k] * b[k][column]).sum())
    })
}

fn inverse(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let m = m.map(|row| row.map(f64::from));
    let cofactor =
        |r0: usize, r1: usize, c0: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    let adjugate = [
        [
            cofactor(1, 2, 1, 2),
            -cofactor(0, 2, 1, 2),
            cofactor(0, 1, 1, 2),
        ],
        [
            -cofactor(1, 2, 0, 2),
            cofactor(0, 2, 0, 2),
            -cofactor(0, 1, 0, 2),
        ],
        [
            cofactor(1, 2, 0, 1),
            -cofactor(0, 2, 0, 1),
            cofactor(0, 1, 0, 1),
        ],
    ];
    let determinant: f64 = (0..3)
        .map(|column| m[0][column] * adjugate[column][0])
        .sum();
    adjugate.map(|row| row.map(|value| (value / determinant) as f32))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPACES: [WorkingSpace; 3] = [
        WorkingSpace::Srgb,
        WorkingSpace::DisplayP3,
        WorkingSpace::Rec2020,
    ];

    fn close(a: [f32; 3], b: [f32; 3], tolerance: f32) -> bool {
        a.iter().zip(&b).all(|(a, b)| (a - b).abs() <= tolerance)
    }

    #[test]
    fn white_is_white_in_every_space() {
        for from in SPACES {
            for to in SPACES {
                let white = from.convert([1.0; 3], to);
                assert!(
                    close(white, [1.0; 3], 1e-4),
                    "{from:?} to {to:?}: {white:?}"
                );
                assert!((to.luma([1.0; 3]) - 1.0).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn a_conversion_and_back_is_the_same_color() {
        let color = [0.8, 0.3, 0.1];
        for from in SPACES {
            for to in SPACES {
                let back = to.convert(from.convert(color, to), from);
                assert!(close(back, color, 1e-4), "{from:?} to {to:?}: {back:?}");
            }
        }
    }

    #[test]
    fn srgb_and_display_p3_red_are_inside_rec_2020() {
        for from in [WorkingSpace::Srgb, WorkingSpace::DisplayP3] {
            let red = from.convert([1.0, 0.0, 0.0], WorkingSpace::Rec2020);
            // Display P3's red is a hair outside: -0.0012 of blue light.
            assert!(red.iter().all(|channel| *channel >= -0.02), "{red:?}");
            assert!(red[0] > red[1] && red[1] > 0.05, "{red:?}");
        }
    }

    #[test]
    fn a_color_outside_the_display_keeps_its_luminance_and_hue() {
        // Rec. 2020's green is outside Display P3.
        let green = [0.0, 1.0, 0.0];
        let converted = WorkingSpace::Rec2020.convert(green, WorkingSpace::DisplayP3);
        assert!(
            converted.iter().any(|channel| *channel < 0.0),
            "{converted:?}"
        );
        let mapped = WorkingSpace::Rec2020.map_to_display(green, WorkingSpace::DisplayP3);
        assert!(mapped.iter().all(|channel| *channel >= 0.0), "{mapped:?}");
        let luminance =
            |color: [f32; 3]| WorkingSpace::DisplayP3.luma(color.map(extended_srgb_decode));
        let wanted = WorkingSpace::Rec2020.luma(green.map(extended_srgb_decode));
        assert!((luminance(mapped) - wanted).abs() < 1e-4, "{mapped:?}");
        assert!(mapped[1] > mapped[0] && mapped[1] > mapped[2], "{mapped:?}");
        // A color the display shows is left as it is.
        let inside = [0.6, 0.4, 0.3];
        let shown = WorkingSpace::DisplayP3.map_to_display(inside, WorkingSpace::DisplayP3);
        assert!(close(shown, inside, 1e-4), "{shown:?}");
    }

    #[test]
    fn hdr_highlights_stay_above_sdr_white() {
        let bright = [1.6, 1.5, 1.4];
        let shown = WorkingSpace::Rec2020.map_to_display(bright, WorkingSpace::DisplayP3);
        assert!(shown.iter().all(|channel| *channel > 1.0), "{shown:?}");
    }
}
