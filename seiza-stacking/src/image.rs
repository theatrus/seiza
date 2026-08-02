use crate::{Error, ReferenceRegion, Result};
use seiza_fits::{BayerPattern, debayer_rgb_f32};

/// A row-major, interleaved linear image with one or three channels.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearImage {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Channel count: 1 for mono, 3 for interleaved RGB.
    pub channels: usize,
    /// Row-major, channel-interleaved samples.
    pub data: Vec<f32>,
}

impl LinearImage {
    /// Build an image, checking that the sample count matches the dimensions
    /// and that the channel count is 1 or 3.
    pub fn new(width: usize, height: usize, channels: usize, data: Vec<f32>) -> Result<Self> {
        if width == 0 || height == 0 || !matches!(channels, 1 | 3) {
            return Err(Error::InvalidImage(
                "dimensions must be non-zero and channels must be 1 or 3".into(),
            ));
        }
        let expected = width
            .checked_mul(height)
            .and_then(|value| value.checked_mul(channels))
            .ok_or_else(|| Error::InvalidImage("image dimensions overflow".into()))?;
        if data.len() != expected {
            return Err(Error::InvalidImage(format!(
                "pixel buffer has {} samples; expected {expected}",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            channels,
            data,
        })
    }

    /// Total number of samples, counting every channel.
    pub fn sample_count(&self) -> usize {
        self.data.len()
    }

    /// Number of pixels, ignoring channels.
    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }

    /// Whether another image has the same width, height, and channel count.
    pub fn dimensions_match(&self, other: &Self) -> bool {
        self.width == other.width && self.height == other.height && self.channels == other.channels
    }

    /// Copy a region of this image into a new image of that size.
    ///
    /// The region is in this image's pixel coordinates and must lie inside it.
    pub fn crop(&self, region: ReferenceRegion) -> Result<Self> {
        let past_right = region.x.checked_add(region.width);
        let past_bottom = region.y.checked_add(region.height);
        if region.width == 0
            || region.height == 0
            || past_right.is_none_or(|edge| edge > self.width)
            || past_bottom.is_none_or(|edge| edge > self.height)
        {
            return Err(Error::InvalidImage(format!(
                "crop region {}x{} at ({}, {}) does not fit a {}x{} image",
                region.width, region.height, region.x, region.y, self.width, self.height
            )));
        }
        if region.x == 0
            && region.y == 0
            && region.width == self.width
            && region.height == self.height
        {
            return Ok(self.clone());
        }
        let mut data = Vec::with_capacity(region.width * region.height * self.channels);
        for row in region.y..region.y + region.height {
            let start = (row * self.width + region.x) * self.channels;
            data.extend_from_slice(&self.data[start..start + region.width * self.channels]);
        }
        Self::new(region.width, region.height, self.channels, data)
    }

    /// One luminance value per pixel: the sample itself for mono, Rec.709 luma
    /// for RGB.
    pub fn luminance(&self) -> Vec<f32> {
        if self.channels == 1 {
            return self.data.clone();
        }
        self.data
            .chunks_exact(3)
            .map(|pixel| rec709_luma(pixel[0], pixel[1], pixel[2]))
            .collect()
    }

    pub(crate) fn debayer(self, layout: BayerLayout) -> Result<Self> {
        if self.channels != 1 {
            return Err(Error::InvalidImage(
                "only a one-channel CFA image can be debayered".into(),
            ));
        }
        let rgb = debayer_rgb_f32(
            &self.data,
            self.width,
            self.height,
            layout.pattern,
            layout.x_offset,
            layout.y_offset,
        );
        Self::new(rgb.width, rgb.height, 3, rgb.data)
    }
}

/// Raw color-filter-array sampling of a one-channel frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BayerLayout {
    /// The CFA color order.
    pub pattern: BayerPattern,
    /// Horizontal offset of the pattern origin, in pixels.
    pub x_offset: usize,
    /// Vertical offset of the pattern origin, in pixels.
    pub y_offset: usize,
}

/// Rec.709 luma from linear RGB samples.
pub(crate) fn rec709_luma(red: f32, green: f32, blue: f32) -> f32 {
    0.2126_f32.mul_add(red, 0.7152_f32.mul_add(green, 0.0722 * blue))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_copies_the_requested_region() {
        let image = LinearImage::new(4, 3, 1, (0..12).map(|v| v as f32).collect()).unwrap();
        let region = ReferenceRegion {
            x: 1,
            y: 1,
            width: 2,
            height: 2,
        };
        let cropped = image.crop(region).unwrap();
        assert_eq!(cropped.width, 2);
        assert_eq!(cropped.height, 2);
        assert_eq!(cropped.data, [5.0, 6.0, 9.0, 10.0]);
    }

    #[test]
    fn crop_keeps_every_channel_of_an_rgb_pixel() {
        let image = LinearImage::new(2, 1, 3, (0..6).map(|v| v as f32).collect()).unwrap();
        let cropped = image
            .crop(ReferenceRegion {
                x: 1,
                y: 0,
                width: 1,
                height: 1,
            })
            .unwrap();
        assert_eq!(cropped.data, [3.0, 4.0, 5.0]);
    }

    #[test]
    fn crop_rejects_a_region_outside_the_image() {
        let image = LinearImage::new(2, 2, 1, vec![0.0; 4]).unwrap();
        for region in [
            ReferenceRegion {
                x: 1,
                y: 0,
                width: 2,
                height: 1,
            },
            ReferenceRegion {
                x: 0,
                y: 0,
                width: 0,
                height: 1,
            },
            ReferenceRegion {
                x: usize::MAX,
                y: 0,
                width: 1,
                height: 1,
            },
        ] {
            let error = image.crop(region).unwrap_err();
            assert!(error.to_string().contains("does not fit"), "{region:?}");
        }
    }

    #[test]
    fn debayer_preserves_samples_at_native_color_sites() {
        let raw = LinearImage::new(4, 4, 1, (0..16).map(|v| v as f32).collect()).unwrap();
        let rgb = raw
            .debayer(BayerLayout {
                pattern: BayerPattern::Rggb,
                x_offset: 0,
                y_offset: 0,
            })
            .unwrap();
        assert_eq!(rgb.channels, 3);
        assert_eq!(rgb.data[0], 0.0);
        assert_eq!(rgb.data[4], 1.0);
        assert_eq!(rgb.data[(3 * 4 + 3) * 3 + 2], 15.0);
    }
}
