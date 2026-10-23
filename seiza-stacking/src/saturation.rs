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

/// Which pixels of a frame clipped: ones there and zeros elsewhere, on a
/// mono grid of the frame's size.
///
/// A frame clips a few thousand of its millions of pixels, in star cores,
/// so the mask is kept as their list when it is short enough: an image of
/// ones and zeros would be written, copied to the GPU and read there whole,
/// a gigabyte of memory traffic a 61 MP frame, for those few thousand.
pub(crate) enum RawClip {
    /// The clipped pixels' indices, ascending.
    Listed {
        width: usize,
        height: usize,
        pixels: Vec<u32>,
    },
    Dense(LinearImage),
}

impl RawClip {
    pub(crate) fn dimensions(&self) -> (usize, usize) {
        match self {
            Self::Listed { width, height, .. } => (*width, *height),
            Self::Dense(mask) => (mask.width, mask.height),
        }
    }

    /// The mask as an image, made from its list when it is one.
    pub(crate) fn to_dense(&self) -> std::borrow::Cow<'_, LinearImage> {
        match self {
            Self::Dense(mask) => std::borrow::Cow::Borrowed(mask),
            Self::Listed {
                width,
                height,
                pixels,
            } => std::borrow::Cow::Owned(listed_image(*width, *height, pixels)),
        }
    }

    pub(crate) fn into_dense(self) -> LinearImage {
        match self {
            Self::Dense(mask) => mask,
            Self::Listed {
                width,
                height,
                pixels,
            } => listed_image(width, height, &pixels),
        }
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn view(&self) -> ClipView<'_> {
        match self {
            Self::Listed {
                width,
                height,
                pixels,
            } => ClipView::Listed {
                width: *width,
                height: *height,
                pixels,
            },
            Self::Dense(mask) => ClipView::Image(mask),
        }
    }

    /// Give a dense mask's samples back for a later frame.
    pub(crate) fn recycle(self) {
        if let Self::Dense(mask) = self {
            crate::recycle::give(mask.data);
        }
    }
}

/// A clip mask as the GPU resamples it: a [`RawClip`]'s list, or an image.
#[cfg(feature = "gpu")]
#[derive(Clone, Copy)]
pub(crate) enum ClipView<'a> {
    Listed {
        width: usize,
        height: usize,
        pixels: &'a [u32],
    },
    Image(&'a LinearImage),
}

#[cfg(feature = "gpu")]
impl ClipView<'_> {
    /// Width, height and channels.
    pub(crate) fn shape(&self) -> (usize, usize, usize) {
        match self {
            Self::Listed { width, height, .. } => (*width, *height, 1),
            Self::Image(mask) => (mask.width, mask.height, mask.channels),
        }
    }
}

fn listed_image(width: usize, height: usize, pixels: &[u32]) -> LinearImage {
    let mut data = crate::recycle::take(width * height);
    data.resize(width * height, 0.0);
    for &pixel in pixels {
        data[pixel as usize] = 1.0;
    }
    LinearImage {
        width,
        height,
        channels: 1,
        data,
    }
}

/// The most clipped pixels a mask lists rather than holds whole: one in
/// 128, as for the GPU's compact clip readback.
fn listed_capacity(pixels: usize) -> usize {
    pixels.div_ceil(128).clamp(1024, 1 << 18)
}

/// Which pixels of `frame`, read and not yet calibrated, hold a sample at
/// its saturation ceiling, within half a unit. `None` when the ceiling is
/// unknown or no sample reaches it, which costs a frame nothing further.
pub(crate) fn raw_clip_mask(frame: &FitsFrame) -> Option<RawClip> {
    let level = saturation_level(&frame.headers)? - 0.5;
    let image = &frame.image;
    let (width, height, channels) = (image.width, image.height, image.channels);
    let clipped = |pixel: &[f32]| pixel.iter().any(|&value| value >= level);
    if u32::try_from(width * height).is_ok() {
        let capacity = listed_capacity(width * height);
        let mut pixels: Vec<u32> = image
            .data
            .par_chunks_exact(channels)
            .enumerate()
            .filter(|(_, pixel)| clipped(pixel))
            .map(|(index, _)| index as u32)
            .take_any(capacity + 1)
            .collect();
        if pixels.is_empty() {
            return None;
        }
        if pixels.len() <= capacity {
            pixels.par_sort_unstable();
            return Some(RawClip::Listed {
                width,
                height,
                pixels,
            });
        }
    }
    let mut mask = crate::recycle::take(width * height);
    image
        .data
        .par_chunks(channels)
        .map(|pixel| f32::from(u8::from(clipped(pixel))))
        .collect_into_vec(&mut mask);
    if !mask.par_iter().any(|&value| value > 0.0) {
        crate::recycle::give(mask);
        return None;
    }
    LinearImage::new(width, height, 1, mask)
        .ok()
        .map(RawClip::Dense)
}

/// A raw clip mask carried through what [`crate::stack::develop`] does to
/// its frame: a Bayer frame's demosaic reads each pixel's neighbours, so a
/// clipped photosite spreads two pixels, unless a superpixel takes each
/// 2x2 whole; then binning by `bin_factor` in all. A binned pixel is
/// clipped when any pixel it holds was. A mono frame left at its size keeps
/// its mask as it was, listed or not.
pub(crate) fn develop_clip_mask(
    mask: RawClip,
    cfa: Option<BayerLayout>,
    demosaic: crate::Demosaic,
    bin_factor: u32,
) -> Result<RawClip> {
    if cfa.is_none() && bin_factor <= 1 {
        return Ok(mask);
    }
    let mask = mask.into_dense();
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
        return Ok(RawClip::Dense(mask));
    }
    any_in_bins(&mask, remaining).map(RawClip::Dense)
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
        assert!(matches!(&mask, RawClip::Listed { pixels, .. } if pixels == &[5]));
        let mask = mask.into_dense();
        assert_eq!(mask.data.iter().filter(|&&value| value == 1.0).count(), 1);
        assert_eq!(mask.data[5], 1.0);
        // Below the ceiling, or with no ceiling known: no mask.
        data[5] = 65000.0;
        assert!(raw_clip_mask(&frame(4, 4, 1, data.clone())).is_none());
        data[5] = 65535.0;
        assert!(raw_clip_mask(&unknown(4, 4, 1, data)).is_none());
    }

    #[test]
    fn a_mask_too_full_to_list_is_held_whole_and_both_read_alike() {
        // One pixel in 128 is listed at most, and never fewer than 1024.
        let (width, height) = (512, 512);
        let capacity = listed_capacity(width * height);
        let mut data = vec![100.0_f32; width * height * 3];
        let clip = |data: &mut Vec<f32>, pixel: usize| data[pixel * 3 + pixel % 3] = 65535.0;
        for pixel in (0..width * height).step_by(97).take(capacity) {
            clip(&mut data, pixel);
        }
        let listed = raw_clip_mask(&frame(width, height, 3, data.clone())).unwrap();
        let RawClip::Listed { pixels, .. } = &listed else {
            panic!("a mask within its capacity is listed");
        };
        assert_eq!(pixels.len(), capacity);
        assert!(pixels.windows(2).all(|pair| pair[0] < pair[1]));
        let expected = listed.to_dense().into_owned();
        // One more and the mask is held whole, with the same samples.
        clip(&mut data, 1);
        let dense = raw_clip_mask(&frame(width, height, 3, data)).unwrap();
        assert!(matches!(dense, RawClip::Dense(_)));
        let mut more = expected.data.clone();
        more[1] = 1.0;
        assert_eq!(dense.into_dense().data, more);
        assert_eq!(
            expected.data.iter().filter(|&&value| value == 1.0).count(),
            capacity
        );
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
        let develop = |mask: &LinearImage, cfa, demosaic, bin_factor| {
            develop_clip_mask(RawClip::Dense(mask.clone()), cfa, demosaic, bin_factor)
                .unwrap()
                .into_dense()
        };
        let spread = develop(&mask, Some(layout), crate::Demosaic::Vng, 1);
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
        let mono = develop(&mask, None, crate::Demosaic::Vng, 1);
        assert_eq!(mono.data, mask.data);
        // A listed mask stays listed.
        let listed = RawClip::Listed {
            width: 8,
            height: 8,
            pixels: vec![3 * 8 + 3],
        };
        let kept = develop_clip_mask(listed, None, crate::Demosaic::Vng, 1).unwrap();
        assert!(matches!(&kept, RawClip::Listed { pixels, .. } if pixels == &[27]));
        assert_eq!(kept.into_dense().data, mask.data);
        // Binned by 2, the clipped pixel's bin is clipped, and only it.
        let binned = develop(&mask, None, crate::Demosaic::Vng, 2);
        assert_eq!((binned.width, binned.height), (4, 4));
        assert_eq!(binned.data.iter().filter(|&&value| value == 1.0).count(), 1);
        assert_eq!(binned.data[4 + 1], 1.0);
        // A superpixel demosaic takes each 2x2 whole, without spreading.
        let superpixel = develop(&mask, Some(layout), crate::Demosaic::Superpixel, 2);
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
