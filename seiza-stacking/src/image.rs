use crate::{Error, Result};
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
