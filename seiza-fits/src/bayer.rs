//! Bilinear debayering for one-shot-color camera FITS files.
//!
//! OSC cameras write the raw color filter array: each pixel carries only
//! one of R/G/B, laid out in a 2×2 mosaic named by the `BAYERPAT` header
//! (with optional `XBAYROFF`/`YBAYROFF` origin offsets). Missing channel
//! samples are reconstructed by averaging every carrier of that channel
//! in the 3×3 neighborhood — bilinear interpolation, adequate for star
//! detection and display.

/// The 2×2 color filter array layout, named by its top-left origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BayerPattern {
    Rggb,
    Bggr,
    Grbg,
    Gbrg,
}

impl BayerPattern {
    /// Parse a `BAYERPAT` header value.
    pub fn parse(name: &str) -> Option<BayerPattern> {
        match name.trim().to_ascii_uppercase().as_str() {
            "RGGB" => Some(Self::Rggb),
            "BGGR" => Some(Self::Bggr),
            "GRBG" => Some(Self::Grbg),
            "GBRG" => Some(Self::Gbrg),
            _ => None,
        }
    }

    /// Canonical FITS `BAYERPAT` spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rggb => "RGGB",
            Self::Bggr => "BGGR",
            Self::Grbg => "GRBG",
            Self::Gbrg => "GBRG",
        }
    }

    /// Channel (0 = R, 1 = G, 2 = B) captured at pixel `(x, y)`, after
    /// applying the pattern origin offsets.
    fn channel_at(self, x: usize, y: usize, x_off: usize, y_off: usize) -> usize {
        let (col, row) = ((x + x_off) & 1, (y + y_off) & 1);
        match self {
            Self::Rggb => [[0, 1], [1, 2]][row][col],
            Self::Bggr => [[2, 1], [1, 0]][row][col],
            Self::Grbg => [[1, 0], [2, 1]][row][col],
            Self::Gbrg => [[1, 2], [0, 1]][row][col],
        }
    }
}

/// Interleaved 16-bit RGB image produced by [`debayer_rgb16`].
#[derive(Debug, Clone)]
pub struct RgbImage16 {
    pub width: usize,
    pub height: usize,
    /// `width * height * 3` samples, RGB interleaved, row-major
    pub data: Vec<u16>,
}

/// Interleaved linear floating-point RGB image produced by [`debayer_rgb_f32`].
#[derive(Debug, Clone)]
pub struct RgbImageF32 {
    pub width: usize,
    pub height: usize,
    /// `width * height * 3` samples, RGB interleaved, row-major
    pub data: Vec<f32>,
}

trait DebayerSample: Copy + Default {
    type Sum: Default;

    fn add(sum: &mut Self::Sum, value: Self);
    fn average(sum: Self::Sum, count: u32) -> Self;
}

impl DebayerSample for u16 {
    type Sum = u32;

    fn add(sum: &mut Self::Sum, value: Self) {
        *sum += u32::from(value);
    }

    fn average(sum: Self::Sum, count: u32) -> Self {
        (sum / count.max(1)) as u16
    }
}

impl DebayerSample for f32 {
    type Sum = f64;

    fn add(sum: &mut Self::Sum, value: Self) {
        *sum += f64::from(value);
    }

    fn average(sum: Self::Sum, count: u32) -> Self {
        (sum / f64::from(count.max(1))) as f32
    }
}

impl RgbImage16 {
    /// Collapse to luminance as `(R + 2G + B) / 4`.
    pub fn to_luma_u16(&self) -> Vec<u16> {
        self.data
            .chunks_exact(3)
            .map(|px| ((px[0] as u32 + 2 * px[1] as u32 + px[2] as u32) / 4) as u16)
            .collect()
    }
}

/// Bilinear-debayer a raw CFA frame into interleaved RGB.
pub fn debayer_rgb16(
    mosaic: &[u16],
    width: usize,
    height: usize,
    pattern: BayerPattern,
    x_offset: usize,
    y_offset: usize,
) -> RgbImage16 {
    RgbImage16 {
        width,
        height,
        data: debayer_interleaved(mosaic, width, height, pattern, x_offset, y_offset),
    }
}

/// Bilinear-debayer a linear floating-point CFA frame into interleaved RGB.
///
/// This uses the same kernel and native-sample preservation as [`debayer_rgb16`]
/// without quantizing calibrated sensor values.
pub fn debayer_rgb_f32(
    mosaic: &[f32],
    width: usize,
    height: usize,
    pattern: BayerPattern,
    x_offset: usize,
    y_offset: usize,
) -> RgbImageF32 {
    RgbImageF32 {
        width,
        height,
        data: debayer_interleaved(mosaic, width, height, pattern, x_offset, y_offset),
    }
}

fn debayer_interleaved<T: DebayerSample>(
    mosaic: &[T],
    width: usize,
    height: usize,
    pattern: BayerPattern,
    x_offset: usize,
    y_offset: usize,
) -> Vec<T> {
    assert_eq!(mosaic.len(), width * height);
    let mut data = vec![T::default(); width * height * 3];

    for y in 0..height {
        for x in 0..width {
            let mut sums: [T::Sum; 3] = std::array::from_fn(|_| T::Sum::default());
            let mut counts = [0u32; 3];
            for ny in y.saturating_sub(1)..(y + 2).min(height) {
                for nx in x.saturating_sub(1)..(x + 2).min(width) {
                    let channel = pattern.channel_at(nx, ny, x_offset, y_offset);
                    T::add(&mut sums[channel], mosaic[ny * width + nx]);
                    counts[channel] += 1;
                }
            }
            let own = pattern.channel_at(x, y, x_offset, y_offset);
            let out = &mut data[(y * width + x) * 3..(y * width + x) * 3 + 3];
            for (channel, (sum, count)) in sums.into_iter().zip(counts).enumerate() {
                out[channel] = if channel == own {
                    mosaic[y * width + x]
                } else {
                    T::average(sum, count)
                };
            }
        }
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_parsing() {
        assert_eq!(BayerPattern::parse("RGGB"), Some(BayerPattern::Rggb));
        assert_eq!(BayerPattern::parse(" bggr "), Some(BayerPattern::Bggr));
        assert_eq!(BayerPattern::parse("XTRANS"), None);
    }

    #[test]
    fn flat_color_field_reconstructs_exactly() {
        // A flat scene of R=4000, G=2000, B=1000 sampled through an RGGB
        // mosaic must debayer back to the flat scene everywhere
        let (w, h) = (8, 6);
        let mut mosaic = vec![0u16; w * h];
        for y in 0..h {
            for x in 0..w {
                mosaic[y * w + x] = match (y & 1, x & 1) {
                    (0, 0) => 4000,
                    (0, 1) | (1, 0) => 2000,
                    (1, 1) => 1000,
                    _ => unreachable!(),
                };
            }
        }
        let rgb = debayer_rgb16(&mosaic, w, h, BayerPattern::Rggb, 0, 0);
        for px in rgb.data.chunks_exact(3) {
            assert_eq!(px, &[4000, 2000, 1000]);
        }
        let luma = rgb.to_luma_u16();
        assert!(luma.iter().all(|&v| v == (4000 + 2 * 2000 + 1000) / 4));
    }

    #[test]
    fn offsets_shift_the_pattern() {
        // With a (1, 1) offset, RGGB behaves like BGGR at the origin
        assert_eq!(BayerPattern::Rggb.channel_at(0, 0, 1, 1), 2);
        assert_eq!(BayerPattern::Bggr.channel_at(0, 0, 0, 0), 2);
    }

    #[test]
    fn integer_and_float_kernels_preserve_all_native_color_sites() {
        let integers = (0_u16..16).collect::<Vec<_>>();
        let floats = integers
            .iter()
            .map(|&value| f32::from(value))
            .collect::<Vec<_>>();
        let integer_rgb = debayer_rgb16(&integers, 4, 4, BayerPattern::Rggb, 0, 0);
        let float_rgb = debayer_rgb_f32(&floats, 4, 4, BayerPattern::Rggb, 0, 0);

        for y in 0..4 {
            for x in 0..4 {
                let channel = BayerPattern::Rggb.channel_at(x, y, 0, 0);
                let output = (y * 4 + x) * 3 + channel;
                let input = y * 4 + x;
                assert_eq!(integer_rgb.data[output], integers[input]);
                assert_eq!(float_rgb.data[output], floats[input]);
            }
        }
    }
}
