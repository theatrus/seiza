//! The raw level at which a frame's sensor clips, and which of its pixels
//! reach it.
//!
//! A clipped sample records only that the sky was at least that bright, and
//! a stack, scaling each frame's channels apart, leaves a clipped star core
//! at a different level in each channel. Colour calibration's gains then
//! tint it. The stacker records which of its samples came from clipped raw
//! pixels, so that colour calibration can render them neutral rather than
//! guess from the finished stack, where neither a fraction of the maximum
//! (moved by flat-fielding) nor a flat top (rounded by dithering and seeing)
//! finds them.

use crate::{BayerLayout, FitsFrame, LinearImage, Result};
use rayon::prelude::*;
use seiza_fits::HeaderValue;

/// A frame's raw saturation ceiling from its headers: `SATURATE` or
/// `SATLEVEL`, or else the largest value its integer FITS encoding holds.
/// Never an observed range, which a frame need not reach.
pub(crate) fn saturation_level(headers: &[(String, HeaderValue)]) -> Option<f32> {
    let number = |name: &str| {
        headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .and_then(|(_, value)| value.as_f64())
    };
    for name in ["SATURATE", "SATLEVEL"] {
        if let Some(level) = number(name).filter(|value| value.is_finite() && *value > 0.0) {
            return (level <= f64::from(f32::MAX)).then_some(level as f32);
        }
    }
    let bits = number("BITPIX")?;
    let (minimum, maximum) = match bits {
        8.0 => (0.0, 255.0),
        16.0 => (-32768.0, 32767.0),
        32.0 => (-2147483648.0, 2147483647.0),
        _ => return None,
    };
    let scale = number("BSCALE").unwrap_or(1.0);
    let offset = number("BZERO").unwrap_or(0.0);
    let level = if scale > 0.0 {
        maximum * scale + offset
    } else {
        minimum * scale + offset
    };
    (scale.is_finite()
        && scale != 0.0
        && level.is_finite()
        && level > 0.0
        && level <= f64::from(f32::MAX))
    .then_some(level as f32)
}

/// Which pixels of `frame`, read and not yet calibrated, hold a sample at
/// its saturation ceiling, within half a unit: a mono image of ones there
/// and zeros elsewhere. `None` when the ceiling is unknown or no sample
/// reaches it, which costs a frame nothing further.
pub(crate) fn raw_clip_mask(frame: &FitsFrame) -> Option<LinearImage> {
    let level = saturation_level(&frame.headers)? - 0.5;
    let image = &frame.image;
    let channels = image.channels;
    let mask = image
        .data
        .par_chunks(channels)
        .map(|pixel| f32::from(u8::from(pixel.iter().any(|&value| value >= level))))
        .collect::<Vec<_>>();
    if !mask.par_iter().any(|&value| value > 0.0) {
        return None;
    }
    LinearImage::new(image.width, image.height, 1, mask).ok()
}

/// A raw clip mask carried through what [`crate::stack::develop`] does to
/// its frame: a Bayer frame's demosaic reads each pixel's neighbours, so a
/// clipped photosite spreads two pixels, unless a superpixel takes each
/// 2x2 whole; then binning by `bin_factor` in all. A binned pixel is
/// clipped when any pixel it holds was.
pub(crate) fn develop_clip_mask(
    mask: LinearImage,
    cfa: Option<BayerLayout>,
    demosaic: crate::Demosaic,
    bin_factor: u32,
) -> Result<LinearImage> {
    let superpixel =
        cfa.is_some() && demosaic == crate::Demosaic::Superpixel && bin_factor.is_multiple_of(2);
    let (mask, remaining) = if superpixel {
        (any_in_bins(&mask, 2)?, bin_factor / 2)
    } else if cfa.is_some() {
        (dilated(&mask, 2), bin_factor)
    } else {
        (mask, bin_factor)
    };
    if remaining <= 1 {
        return Ok(mask);
    }
    any_in_bins(&mask, remaining)
}

/// One where any pixel of a `factor`-square bin is clipped, as
/// [`LinearImage::binned`] lays the bins out.
fn any_in_bins(mask: &LinearImage, factor: u32) -> Result<LinearImage> {
    let mut binned = mask.binned(factor)?;
    for value in &mut binned.data {
        *value = f32::from(u8::from(*value > 0.0));
    }
    Ok(binned)
}

/// `mask` with every clipped pixel spread `radius` pixels in each direction:
/// along each row, then down the columns, every output row taking the
/// largest of the spread rows within `radius` of it. Each row of both passes
/// is independent and reads memory in order, so both run on every core.
fn dilated(mask: &LinearImage, radius: usize) -> LinearImage {
    let (width, height) = (mask.width, mask.height);
    let mut rows = vec![0.0_f32; width * height];
    rows.par_chunks_mut(width)
        .zip(mask.data.par_chunks(width))
        .for_each(|(output, line)| {
            for (index, value) in output.iter_mut().enumerate() {
                let start = index.saturating_sub(radius);
                let end = (index + radius + 1).min(width);
                *value = line[start..end].iter().copied().fold(0.0, f32::max);
            }
        });
    let mut data = vec![0.0_f32; width * height];
    data.par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, output)| {
            let (start, end) = (y.saturating_sub(radius), (y + radius + 1).min(height));
            for row in rows[start * width..end * width].chunks_exact(width) {
                for (value, &spread) in output.iter_mut().zip(row) {
                    *value = value.max(spread);
                }
            }
        });
    LinearImage {
        width,
        height,
        channels: 1,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seiza_fits::BayerPattern;

    fn unknown(width: usize, height: usize, channels: usize, data: Vec<f32>) -> FitsFrame {
        FitsFrame {
            image: LinearImage::new(width, height, channels, data).unwrap(),
            headers: Vec::new(),
            exposure_seconds: None,
            bayer: None,
            source: None,
            bounds: None,
        }
    }

    fn frame(width: usize, height: usize, channels: usize, data: Vec<f32>) -> FitsFrame {
        let mut frame = unknown(width, height, channels, data);
        frame.headers.extend([
            ("BITPIX".into(), HeaderValue::Integer(16)),
            ("BZERO".into(), HeaderValue::Float(32768.0)),
        ]);
        frame
    }

    #[test]
    fn saturation_uses_raw_encoding_or_explicit_metadata_not_observed_ranges() {
        let mut headers = vec![("DATAMAX".to_string(), HeaderValue::Float(65535.0))];
        assert_eq!(saturation_level(&headers), None);
        headers.extend([
            ("BITPIX".into(), HeaderValue::Integer(16)),
            ("BZERO".into(), HeaderValue::Float(32768.0)),
        ]);
        assert_eq!(saturation_level(&headers), Some(65535.0));
        headers.push(("SATURATE".into(), HeaderValue::Float(60000.0)));
        assert_eq!(saturation_level(&headers), Some(60000.0));
    }

    #[test]
    fn the_raw_mask_marks_samples_at_the_ceiling_and_none_without_one() {
        let mut data = vec![100.0_f32; 16];
        data[5] = 65535.0;
        let mask = raw_clip_mask(&frame(4, 4, 1, data.clone())).unwrap();
        assert_eq!(mask.data.iter().filter(|&&value| value == 1.0).count(), 1);
        assert_eq!(mask.data[5], 1.0);
        // Below the ceiling, or with no ceiling known: no mask.
        data[5] = 65000.0;
        assert!(raw_clip_mask(&frame(4, 4, 1, data.clone())).is_none());
        data[5] = 65535.0;
        assert!(raw_clip_mask(&unknown(4, 4, 1, data)).is_none());
    }

    #[test]
    fn a_bayer_mask_spreads_two_pixels_and_binning_keeps_any_clipped_pixel() {
        let mut data = vec![0.0_f32; 64];
        data[3 * 8 + 3] = 1.0;
        let mask = LinearImage::new(8, 8, 1, data).unwrap();
        let layout = BayerLayout {
            pattern: BayerPattern::Rggb,
            x_offset: 0,
            y_offset: 0,
        };
        let spread =
            develop_clip_mask(mask.clone(), Some(layout), crate::Demosaic::Vng, 1).unwrap();
        for y in 0..8 {
            for x in 0..8 {
                let inside = (1..=5).contains(&x) && (1..=5).contains(&y);
                assert_eq!(
                    spread.data[y * 8 + x],
                    f32::from(u8::from(inside)),
                    "({x}, {y})"
                );
            }
        }
        // A mono frame is not demosaiced, so its mask does not spread.
        let mono = develop_clip_mask(mask.clone(), None, crate::Demosaic::Vng, 1).unwrap();
        assert_eq!(mono.data, mask.data);
        // Binned by 2, the clipped pixel's bin is clipped, and only it.
        let binned = develop_clip_mask(mask.clone(), None, crate::Demosaic::Vng, 2).unwrap();
        assert_eq!((binned.width, binned.height), (4, 4));
        assert_eq!(binned.data.iter().filter(|&&value| value == 1.0).count(), 1);
        assert_eq!(binned.data[4 + 1], 1.0);
        // A superpixel demosaic takes each 2x2 whole, without spreading.
        let superpixel =
            develop_clip_mask(mask, Some(layout), crate::Demosaic::Superpixel, 2).unwrap();
        assert_eq!((superpixel.width, superpixel.height), (4, 4));
        assert_eq!(
            superpixel
                .data
                .iter()
                .filter(|&&value| value == 1.0)
                .count(),
            1
        );
    }
}
