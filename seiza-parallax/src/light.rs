//! Images of screen-blend "light".
//!
//! A stretched star image is laid over its starless image with a screen
//! blend, `1 − (1 − a)(1 − b)`. Writing a display value `v` as the light
//! `L = −ln(1 − v)` turns the screen blend into addition: the light of a
//! screened stack of layers is the sum of the layers' lights. So a star image
//! cut into one layer per star plus a leftover layer, each moved on its own,
//! puts itself back together wherever the layers line up again.
//!
//! The light of a pixel is taken from its brightest channel, with the other
//! channels kept in the same proportion. Per channel, a near-white star core
//! such as (0.95, 0.92, 1.0) would become light of very different sizes, and
//! brightening or shrinking the star would turn it magenta. Scaled together,
//! the channels keep the star's hue. Where two layers of different hue
//! overlap, the blend is then close to a screen blend rather than exact.

use image::{Rgb, Rgb32FImage, RgbImage};
use rayon::prelude::*;

/// The brightest display value kept, so white maps to a finite light.
const MAX_DISPLAY: f32 = 1.0 - 1.0 / 65_536.0;

/// The light of display value `value`, which runs 0 to 1; a NaN, as a
/// float image's blank border holds, is dark.
pub fn light_of(value: f32) -> f32 {
    if value.is_nan() {
        return 0.0;
    }
    -(1.0 - value.clamp(0.0, MAX_DISPLAY)).ln()
}

/// The display value of `light`.
pub fn display_of(light: f32) -> f32 {
    1.0 - (-light.max(0.0)).exp()
}

/// The light of an RGB display pixel: its brightest channel's light, shared
/// across the channels in their display proportions.
pub fn pixel_light(display: [f32; 3]) -> [f32; 3] {
    // A NaN is dark; `clamp` would keep it, and one NaN star pixel would
    // pass for a star 200 pixels wide.
    let display = display.map(|value| {
        if value.is_nan() {
            0.0
        } else {
            value.clamp(0.0, 1.0)
        }
    });
    let peak = display[0].max(display[1]).max(display[2]);
    if peak <= 0.0 {
        return [0.0; 3];
    }
    let light = light_of(peak) / peak;
    display.map(|value| value * light)
}

/// The display pixel of `light`, inverting [`pixel_light`].
pub fn pixel_display(light: [f32; 3]) -> [f32; 3] {
    let light = light.map(|value| value.max(0.0));
    let peak = light[0].max(light[1]).max(light[2]);
    if peak <= 0.0 {
        return [0.0; 3];
    }
    let display = display_of(peak) / peak;
    light.map(|value| value * display)
}

/// An RGB image of light, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct LightImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<[f32; 3]>,
}

impl LightImage {
    /// A dark image.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![[0.0; 3]; width * height],
        }
    }

    /// The light of a stretched image whose values run 0 to 1.
    pub fn from_display(image: &Rgb32FImage) -> Self {
        let (width, height) = (image.width() as usize, image.height() as usize);
        let pixels = image
            .as_raw()
            .par_chunks_exact(3)
            .map(|pixel| pixel_light([pixel[0], pixel[1], pixel[2]]))
            .collect();
        Self {
            width,
            height,
            pixels,
        }
    }

    /// The image as 8-bit display values.
    pub fn to_display_rgb8(&self) -> RgbImage {
        let mut out = RgbImage::new(self.width as u32, self.height as u32);
        out.as_mut()
            .par_chunks_exact_mut(3)
            .zip(self.pixels.par_iter())
            .for_each(|(out, light)| {
                let display = pixel_display(*light);
                for channel in 0..3 {
                    out[channel] = (display[channel] * 255.0 + 0.5) as u8;
                }
            });
        out
    }

    /// The image as display values.
    pub fn to_display(&self) -> Rgb32FImage {
        let mut out = Rgb32FImage::new(self.width as u32, self.height as u32);
        for (pixel, light) in out.pixels_mut().zip(&self.pixels) {
            *pixel = Rgb(pixel_display(*light));
        }
        out
    }

    #[inline]
    pub(crate) fn at(&self, x: usize, y: usize) -> [f32; 3] {
        self.pixels[y * self.width + x]
    }

    /// Bilinear light at `(x, y)` in pixel-centre coordinates (pixel `i`
    /// covers `i − 0.5` to `i + 0.5`), dark outside the image.
    #[inline]
    pub fn sample(&self, x: f32, y: f32) -> [f32; 3] {
        let (x0, y0) = (floor(x), floor(y));
        let (tx, ty) = (x - x0 as f32, y - y0 as f32);
        // Most samples have all four neighbours inside the image.
        if x0 >= 0 && y0 >= 0 && (x0 as usize) + 1 < self.width && (y0 as usize) + 1 < self.height {
            let index = y0 as usize * self.width + x0 as usize;
            let (a, b) = (self.pixels[index], self.pixels[index + 1]);
            let (c, d) = (
                self.pixels[index + self.width],
                self.pixels[index + self.width + 1],
            );
            let mut sum = [0.0_f32; 3];
            for channel in 0..3 {
                let top = a[channel] + (b[channel] - a[channel]) * tx;
                let bottom = c[channel] + (d[channel] - c[channel]) * tx;
                sum[channel] = top + (bottom - top) * ty;
            }
            return sum;
        }
        let mut sum = [0.0_f32; 3];
        for (dy, wy) in [(0, 1.0 - ty), (1, ty)] {
            for (dx, wx) in [(0, 1.0 - tx), (1, tx)] {
                let (px, py) = (x0 + dx, y0 + dy);
                if px < 0 || py < 0 || px >= self.width as isize || py >= self.height as isize {
                    continue;
                }
                let weight = wx * wy;
                let light = self.at(px as usize, py as usize);
                for channel in 0..3 {
                    sum[channel] += weight * light[channel];
                }
            }
        }
        sum
    }

    /// [`Self::halved`] on one thread, for a small image.
    pub(crate) fn halved_serial(&self) -> Self {
        let (width, height) = (self.width.div_ceil(2), self.height.div_ceil(2));
        let mut pixels = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                pixels.push(self.mean_of_four(x, y));
            }
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    /// The mean of the pixels of the 2×2 block `(x, y)` that lie in the
    /// image.
    fn mean_of_four(&self, x: usize, y: usize) -> [f32; 3] {
        let mut sum = [0.0_f32; 3];
        let mut count = 0.0;
        for (sx, sy) in [
            (2 * x, 2 * y),
            (2 * x + 1, 2 * y),
            (2 * x, 2 * y + 1),
            (2 * x + 1, 2 * y + 1),
        ] {
            if sx < self.width && sy < self.height {
                let light = self.at(sx, sy);
                for channel in 0..3 {
                    sum[channel] += light[channel];
                }
                count += 1.0;
            }
        }
        sum.map(|value| value / count)
    }

    /// Half the size, each pixel the mean of the four it covers.
    pub(crate) fn halved(&self) -> Self {
        let (width, height) = (self.width.div_ceil(2), self.height.div_ceil(2));
        let pixels = (0..height)
            .into_par_iter()
            .flat_map_iter(|y| (0..width).map(move |x| self.mean_of_four(x, y)))
            .collect();
        Self {
            width,
            height,
            pixels,
        }
    }
}

/// An image with its successive halvings, sampled at the level whose pixels
/// match the output's, so a shrunk view does not alias.
#[derive(Clone, Debug)]
pub struct Pyramid {
    levels: Vec<LightImage>,
}

impl Pyramid {
    pub fn new(image: LightImage) -> Self {
        let mut levels = vec![image];
        while let Some(last) = levels.last()
            && last.width > 64
            && last.height > 64
        {
            let next = last.halved();
            levels.push(next);
        }
        Self { levels }
    }

    pub fn base(&self) -> &LightImage {
        &self.levels[0]
    }

    /// The level for a view where one output pixel spans `footprint` base
    /// pixels, and that level's scale relative to the base.
    pub(crate) fn level_for(&self, footprint: f32) -> (&LightImage, f32) {
        let index = if footprint > 1.0 {
            (footprint.log2().floor() as usize).min(self.levels.len() - 1)
        } else {
            0
        };
        (&self.levels[index], (1u64 << index) as f32)
    }

    /// The two levels either side of a view where one output pixel spans
    /// `footprint` base pixels, each with its scale, and how far toward the
    /// coarser one the view lies, 0 to 1: blending them changes the
    /// sharpness smoothly as the footprint does.
    pub(crate) fn levels_between(
        &self,
        footprint: f32,
    ) -> ((&LightImage, f32), (&LightImage, f32), f32) {
        let finer = self.level_for(footprint);
        let index = (finer.1 as u64).trailing_zeros() as usize;
        if footprint <= 1.0 || index + 1 >= self.levels.len() {
            return (finer, finer, 0.0);
        }
        let toward = (footprint.log2() - index as f32).clamp(0.0, 1.0);
        (finer, (&self.levels[index + 1], finer.1 * 2.0), toward)
    }

    /// Bilinear light at base coordinates `(x, y)` from `level` of scale
    /// `scale`.
    #[inline]
    pub(crate) fn sample_level(level: &LightImage, scale: f32, x: f32, y: f32) -> [f32; 3] {
        // Pixel centres of level k sit at base coordinates
        // `scale * i + (scale − 1) / 2`. A point within the image's edge
        // pixels, outside the outer centres, takes the edge's light rather
        // than fading toward the dark beyond.
        let edge = |at: f32, size: usize| {
            let at = (at - (scale - 1.0) / 2.0) / scale;
            if (-0.5..=size as f32 - 0.5).contains(&at) {
                at.clamp(0.0, size.saturating_sub(1) as f32)
            } else {
                at
            }
        };
        level.sample(edge(x, level.width), edge(y, level.height))
    }
}

/// `value` rounded down, without the library call `f32::floor` makes on
/// targets lacking a rounding instruction: it runs for every pixel.
#[inline]
pub(crate) fn floor(value: f32) -> isize {
    let truncated = value as isize;
    if (truncated as f32) > value {
        truncated - 1
    } else {
        truncated
    }
}

/// [`floor`] for `f64`.
#[inline]
pub(crate) fn floor64(value: f64) -> isize {
    let truncated = value as isize;
    if (truncated as f64) > value {
        truncated - 1
    } else {
        truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_coarse_level_takes_the_edge_light_up_to_the_image_edge() {
        // Pyramids halve while a level is over 64 pixels a side.
        let mut image = LightImage::new(130, 130);
        image.pixels.fill([1.0; 3]);
        let pyramid = Pyramid::new(image);
        // Base coordinates on the outer pixel centres, and half a pixel
        // past them, read the edge's light at every level; farther out is
        // dark.
        for (scale, level) in [(1.0, &pyramid.levels[0]), (2.0, &pyramid.levels[1])] {
            for (x, y) in [(0.0, 0.0), (129.0, 129.0), (-0.5, 3.0), (129.5, 3.0)] {
                let light = Pyramid::sample_level(level, scale, x, y);
                assert!(
                    (light[0] - 1.0).abs() < 1e-6,
                    "scale {scale} at ({x}, {y}): {light:?}"
                );
            }
            assert!(Pyramid::sample_level(level, scale, -3.0, 3.0)[0] < 0.5);
        }
    }

    #[test]
    fn a_nan_pixel_is_dark() {
        assert_eq!(pixel_light([f32::NAN, 0.5, f32::NAN])[0], 0.0);
        assert!(
            pixel_light([f32::NAN, 0.5, 0.2])
                .iter()
                .all(|value| value.is_finite())
        );
        assert_eq!(light_of(f32::NAN), 0.0);
    }

    #[test]
    fn screen_blend_is_addition_of_light() {
        for (a, b) in [(0.2_f32, 0.5_f32), (0.0, 0.9), (0.7, 0.7)] {
            let screened = 1.0 - (1.0 - a) * (1.0 - b);
            let added = display_of(light_of(a) + light_of(b));
            assert!((screened - added).abs() < 1e-6, "{a} {b}");
        }
        assert_eq!(display_of(light_of(0.0)), 0.0);
        assert!(light_of(1.0).is_finite());
    }

    #[test]
    fn pixel_light_round_trips_and_keeps_hue_when_scaled() {
        for display in [
            [0.95_f32, 0.92, 1.0],
            [0.2, 0.5, 0.1],
            [0.0, 0.0, 0.0],
            [0.3, 0.3, 0.3],
        ] {
            let back = pixel_display(pixel_light(display));
            for channel in 0..3 {
                assert!(
                    (back[channel] - display[channel]).abs() < 1e-4,
                    "{display:?} {back:?}"
                );
            }
        }
        // A near-white core brightened stays near white, not magenta.
        let core = pixel_light([0.95, 0.92, 0.99]);
        let brighter = pixel_display(core.map(|value| value * 1.5));
        assert!(
            brighter[0] / brighter[1] < 0.95 / 0.92 + 1e-3,
            "{brighter:?}"
        );
        // Grey layers still screen exactly.
        let screened = 1.0 - (1.0 - 0.3) * (1.0 - 0.5);
        let added = pixel_display(
            [0, 1, 2]
                .map(|channel| pixel_light([0.3; 3])[channel] + pixel_light([0.5; 3])[channel]),
        );
        assert!((added[0] - screened).abs() < 1e-6);
    }

    #[test]
    fn bilinear_sampling_hits_pixel_centres_and_fades_outside() {
        let mut image = LightImage::new(2, 1);
        image.pixels = vec![[1.0; 3], [3.0; 3]];
        assert_eq!(image.sample(0.0, 0.0), [1.0; 3]);
        assert_eq!(image.sample(0.5, 0.0), [2.0; 3]);
        assert_eq!(image.sample(2.0, 0.0), [0.0; 3]);
    }

    #[test]
    fn pyramid_levels_average_and_line_up() {
        let mut image = LightImage::new(256, 256);
        for (index, pixel) in image.pixels.iter_mut().enumerate() {
            *pixel = [(index % 256) as f32; 3];
        }
        let pyramid = Pyramid::new(image);
        let (level, scale) = pyramid.level_for(2.5);
        assert_eq!(scale, 2.0);
        assert_eq!(level.width, 128);
        // Level 1 pixel 0 averages base columns 0 and 1, centred at 0.5.
        let value = Pyramid::sample_level(level, scale, 0.5, 0.5)[0];
        assert!((value - 0.5).abs() < 1e-5, "{value}");
    }
}
