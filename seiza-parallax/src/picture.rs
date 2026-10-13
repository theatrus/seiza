//! Images laid over a parallax video's frames: a watermark image, such as a
//! logo, and title cards, each with its transparency kept.
//!
//! An image is read once, when a video is set up, multiplied by its alpha
//! (premultiplied) and scaled to the size it is drawn at, so drawing it is
//! one blend a pixel. Scaling premultiplied light keeps dark and light
//! fringes out of soft edges. It is blended over the finished 8-bit frame
//! in sRGB, as the labels are, so it looks as it does in the program that
//! made it.

use image::imageops::{self, FilterType};
use image::{RgbImage, Rgba32FImage, RgbaImage};
use rayon::prelude::*;
use std::path::Path;

/// An image read to draw over frames, with whether any of it is
/// transparent.
#[derive(Clone, Debug)]
pub(crate) struct Artwork {
    pub image: RgbaImage,
    pub transparent: bool,
}

impl Artwork {
    /// The image at `path`, any raster `seiza-image-io` reads, upright by
    /// its EXIF orientation; one without alpha is opaque.
    pub(crate) fn open(path: &Path) -> Result<Self, String> {
        let raster = seiza_image_io::open_raster(path).map_err(|error| error.to_string())?;
        let image = raster.pixels.to_rgba8();
        if image.width() == 0 || image.height() == 0 {
            return Err(format!("{} is empty", path.display()));
        }
        let transparent = image.pixels().any(|pixel| pixel[3] < u8::MAX);
        Ok(Self { image, transparent })
    }

    /// Its width over its height.
    pub(crate) fn aspect(&self) -> f64 {
        self.image.width() as f64 / self.image.height() as f64
    }
}

/// An image as it is drawn: premultiplied by its alpha, at its drawn size.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Picture {
    pixels: RgbaImage,
}

impl Picture {
    /// `artwork` drawn at `width` × `height`: copied at its own size, else
    /// scaled with a Lanczos filter.
    pub(crate) fn new(artwork: &Artwork, (width, height): (u32, u32)) -> Self {
        let (width, height) = (width.max(1), height.max(1));
        let source = &artwork.image;
        if source.dimensions() == (width, height) {
            let mut pixels = source.clone();
            for pixel in pixels.pixels_mut() {
                let alpha = pixel[3] as u32;
                for channel in 0..3 {
                    pixel[channel] = ((pixel[channel] as u32 * alpha + 127) / 255) as u8;
                }
            }
            return Self { pixels };
        }
        // Scaled in floating point, so faint edges keep their colour.
        let premultiplied = Rgba32FImage::from_fn(source.width(), source.height(), |x, y| {
            let pixel = source.get_pixel(x, y);
            let alpha = pixel[3] as f32 / 255.0;
            image::Rgba([
                pixel[0] as f32 / 255.0 * alpha,
                pixel[1] as f32 / 255.0 * alpha,
                pixel[2] as f32 / 255.0 * alpha,
                alpha,
            ])
        });
        let scaled = imageops::resize(&premultiplied, width, height, FilterType::Lanczos3);
        let quantize = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
        let pixels = RgbaImage::from_fn(width, height, |x, y| {
            let pixel = scaled.get_pixel(x, y);
            // The filter overshoots near hard edges; light never exceeds
            // its own cover.
            let alpha = pixel[3].clamp(0.0, 1.0);
            image::Rgba([
                quantize(pixel[0].min(alpha)),
                quantize(pixel[1].min(alpha)),
                quantize(pixel[2].min(alpha)),
                quantize(alpha),
            ])
        });
        Self { pixels }
    }

    pub(crate) fn size(&self) -> (u32, u32) {
        self.pixels.dimensions()
    }

    /// Lay it over `canvas` with its top-left pixel at `(left, top)`, at
    /// `opacity`: each pixel becomes `light · opacity + under · (1 − alpha
    /// · opacity)`.
    pub(crate) fn draw(&self, canvas: &mut RgbImage, (left, top): (i64, i64), opacity: f32) {
        let opacity = opacity.clamp(0.0, 1.0);
        if opacity <= 0.0 {
            return;
        }
        let (width, height) = (canvas.width() as i64, canvas.height() as i64);
        let (own_width, own_height) = (self.pixels.width() as i64, self.pixels.height() as i64);
        let (first_x, last_x) = (left.max(0), (left + own_width).min(width));
        let (first_y, last_y) = (top.max(0), (top + own_height).min(height));
        if first_x >= last_x || first_y >= last_y {
            return;
        }
        let row_bytes = width as usize * 3;
        canvas
            .par_chunks_mut(row_bytes)
            .enumerate()
            .skip(first_y as usize)
            .take((last_y - first_y) as usize)
            .for_each(|(y, row)| {
                let own_y = (y as i64 - top) as u32;
                for x in first_x..last_x {
                    let pixel = self.pixels.get_pixel((x - left) as u32, own_y);
                    if pixel[3] == 0 {
                        continue;
                    }
                    let keep = 1.0 - pixel[3] as f32 / 255.0 * opacity;
                    let under = &mut row[x as usize * 3..x as usize * 3 + 3];
                    for channel in 0..3 {
                        let value = pixel[channel] as f32 * opacity + under[channel] as f32 * keep;
                        under[channel] = value.round().clamp(0.0, 255.0) as u8;
                    }
                }
            });
    }
}

/// The largest size an image of shape `aspect` (width over height) takes
/// inside a `width` × `height` frame, and where its top-left corner goes
/// to centre it.
pub(crate) fn fitted(aspect: f64, (width, height): (usize, usize)) -> ((u32, u32), (i64, i64)) {
    let (frame_width, frame_height) = (width as f64, height as f64);
    let (drawn_width, drawn_height) = if aspect >= frame_width / frame_height {
        (frame_width, frame_width / aspect)
    } else {
        (frame_height * aspect, frame_height)
    };
    let size = (
        (drawn_width.round() as u32).clamp(1, width as u32),
        (drawn_height.round() as u32).clamp(1, height as u32),
    );
    let at = (
        (width as i64 - size.0 as i64) / 2,
        (height as i64 - size.1 as i64) / 2,
    );
    (size, at)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    fn artwork(width: u32, height: u32, pixel: [u8; 4]) -> Artwork {
        Artwork {
            image: RgbaImage::from_pixel(width, height, image::Rgba(pixel)),
            transparent: pixel[3] < u8::MAX,
        }
    }

    #[test]
    fn a_picture_blends_by_its_alpha_and_the_opacity() {
        let canvas = || RgbImage::from_pixel(4, 3, Rgb([100, 100, 100]));
        // Clear leaves the frame as it was; opaque replaces it.
        let mut frame = canvas();
        Picture::new(&artwork(4, 3, [255, 0, 0, 0]), (4, 3)).draw(&mut frame, (0, 0), 1.0);
        assert_eq!(frame, canvas());
        Picture::new(&artwork(4, 3, [255, 0, 0, 255]), (4, 3)).draw(&mut frame, (0, 0), 1.0);
        assert_eq!(frame.get_pixel(2, 1), &Rgb([255, 0, 0]));
        // Half covered at half opacity: a quarter of the way.
        let mut frame = canvas();
        Picture::new(&artwork(4, 3, [200, 200, 200, 128]), (4, 3)).draw(&mut frame, (0, 0), 0.5);
        let value = frame.get_pixel(0, 0)[0];
        assert!((124..=126).contains(&value), "{value}");
        // Placed partly off the frame, only what lands on it is drawn.
        let mut frame = canvas();
        Picture::new(&artwork(2, 2, [0, 0, 0, 255]), (2, 2)).draw(&mut frame, (3, 2), 1.0);
        assert_eq!(frame.get_pixel(3, 2), &Rgb([0, 0, 0]));
        assert_eq!(frame.get_pixel(2, 2), &Rgb([100, 100, 100]));
        assert_eq!(frame.get_pixel(3, 1), &Rgb([100, 100, 100]));
    }

    #[test]
    fn a_scaled_picture_keeps_its_colour_at_soft_edges() {
        // White at a quarter cover, shrunk: its light stays a quarter
        // white, not darkened toward the clear pixels' black.
        let shrunk = Picture::new(&artwork(40, 20, [255, 255, 255, 64]), (10, 5));
        assert_eq!(shrunk.size(), (10, 5));
        let pixel = shrunk.pixels.get_pixel(5, 2);
        assert_eq!(pixel[3], 64);
        assert!(pixel[0].abs_diff(64) <= 1, "{pixel:?}");
    }

    #[test]
    fn a_card_fits_inside_the_frame_and_is_centred() {
        // The frame's own shape fills it.
        assert_eq!(fitted(16.0 / 9.0, (1920, 1080)), ((1920, 1080), (0, 0)));
        // A landscape card on a portrait frame: a band across the middle.
        assert_eq!(fitted(16.0 / 9.0, (1080, 1920)), ((1080, 608), (0, 656)));
        // A square on a landscape frame: centred across.
        assert_eq!(fitted(1.0, (1920, 1080)), ((1080, 1080), (420, 0)));
    }
}
