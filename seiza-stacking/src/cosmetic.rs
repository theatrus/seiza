//! Impulse-pixel suppression for calibration masters and light frames.
//!
//! A hot or dead sensor pixel is consistent across every frame of a set, so
//! the across-frame sigma clipping used when integrating a master keeps it:
//! statistically it is not an outlier, it is the pixel's behavior. The only
//! way to remove it is spatial — the pixel disagrees with its immediate
//! neighborhood far beyond the local noise.
//!
//! [`suppress_impulses`] replaces each sample that deviates from the median
//! of its same-plane neighbors by more than a threshold times the local
//! robust scale (MAD) with that median. On a raw CFA frame, "same-plane"
//! steps by two so a red photosite is only ever compared with red
//! photosites. Structure wider than one pixel — stars, dust shadows,
//! vignetting — raises the neighbors along with the center and survives.

use crate::{BayerLayout, Error, LinearImage, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Consistency factor between the median absolute deviation and the
/// standard deviation of a normal distribution.
const MAD_TO_SIGMA: f32 = 1.4826;

/// Thresholds for [`suppress_impulses`], in units of the local robust
/// standard deviation estimated from the same-plane neighborhood.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct ImpulseFilterOptions {
    /// Replace a sample this many sigma below its neighborhood median.
    pub low_sigma: f32,
    /// Replace a sample this many sigma above its neighborhood median.
    pub high_sigma: f32,
}

impl Default for ImpulseFilterOptions {
    /// Conservative by design: a defective pixel sits hundreds of sigma
    /// from its neighborhood, while the core of a sharply sampled star is
    /// tens. The default removes the former without touching the latter.
    fn default() -> Self {
        Self {
            low_sigma: 16.0,
            high_sigma: 16.0,
        }
    }
}

impl ImpulseFilterOptions {
    fn validate(&self) -> Result<()> {
        if !self.low_sigma.is_finite()
            || self.low_sigma <= 0.0
            || !self.high_sigma.is_finite()
            || self.high_sigma <= 0.0
        {
            return Err(Error::Calibration(
                "impulse filter sigmas must be positive finite numbers".into(),
            ));
        }
        Ok(())
    }
}

/// Replace impulse samples with their same-plane neighborhood median.
///
/// `bayer` marks the image as raw CFA data: neighbors then step by two so
/// each photosite is compared only with its own color plane. Interleaved
/// multi-channel images are filtered per channel. Every replacement reads
/// the original samples, never other replacements. Returns the number of
/// samples replaced.
pub fn suppress_impulses(
    image: &mut LinearImage,
    bayer: Option<BayerLayout>,
    options: &ImpulseFilterOptions,
) -> Result<usize> {
    options.validate()?;
    if bayer.is_some() && image.channels != 1 {
        return Err(Error::Calibration(
            "a CFA impulse filter needs a one-channel image".into(),
        ));
    }
    let step = if bayer.is_some() { 2usize } else { 1usize };
    let width = image.width;
    let height = image.height;
    let channels = image.channels;
    if width <= 2 * step || height <= 2 * step {
        return Ok(0);
    }

    let source = image.data.clone();
    let replaced = image
        .data
        .par_chunks_mut(width * channels)
        .enumerate()
        .map(|(y, row)| {
            let mut replaced = 0usize;
            let mut neighbors = [0.0f32; 8];
            for x in 0..width {
                for channel in 0..channels {
                    let value = row[x * channels + channel];
                    if !value.is_finite() {
                        continue;
                    }
                    let mut count = 0usize;
                    for dy in [-(step as isize), 0, step as isize] {
                        let Some(sample_y) = y.checked_add_signed(dy).filter(|&y| y < height)
                        else {
                            continue;
                        };
                        for dx in [-(step as isize), 0, step as isize] {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let Some(sample_x) = x.checked_add_signed(dx).filter(|&x| x < width)
                            else {
                                continue;
                            };
                            let neighbor =
                                source[(sample_y * width + sample_x) * channels + channel];
                            if neighbor.is_finite() {
                                neighbors[count] = neighbor;
                                count += 1;
                            }
                        }
                    }
                    if count < 5 {
                        continue;
                    }
                    let median = median_in_place(&mut neighbors[..count]);
                    let mut deviations = [0.0f32; 8];
                    for (deviation, neighbor) in deviations.iter_mut().zip(&neighbors[..count]) {
                        *deviation = (neighbor - median).abs();
                    }
                    let sigma = (MAD_TO_SIGMA * median_in_place(&mut deviations[..count]))
                        .max(f32::MIN_POSITIVE);
                    let deviation = value - median;
                    if deviation > options.high_sigma * sigma
                        || -deviation > options.low_sigma * sigma
                    {
                        row[x * channels + channel] = median;
                        replaced += 1;
                    }
                }
            }
            replaced
        })
        .sum();
    Ok(replaced)
}

/// Median of a small scratch slice; reorders it.
fn median_in_place(values: &mut [f32]) -> f32 {
    values.sort_unstable_by(f32::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        values[middle]
    } else {
        (values[middle - 1] + values[middle]) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noisy_gradient(width: usize, height: usize) -> LinearImage {
        // A smooth ramp plus deterministic per-pixel jitter, so the local
        // MAD is realistic rather than zero.
        let data = (0..width * height)
            .map(|index| {
                let x = index % width;
                let y = index / width;
                let jitter = ((index * 2_654_435_761) % 97) as f32 / 97.0 - 0.5;
                100.0 + x as f32 + y as f32 * 0.5 + jitter
            })
            .collect();
        LinearImage::new(width, height, 1, data).unwrap()
    }

    #[test]
    fn replaces_a_hot_pixel_and_reports_it() {
        let mut image = noisy_gradient(32, 32);
        let hot = 17 * 32 + 11;
        image.data[hot] = 60_000.0;
        let replaced =
            suppress_impulses(&mut image, None, &ImpulseFilterOptions::default()).unwrap();
        assert_eq!(replaced, 1);
        assert!(
            (image.data[hot] - 128.0).abs() < 16.0,
            "hot pixel must land near its neighborhood: {}",
            image.data[hot]
        );
    }

    #[test]
    fn replaces_a_dead_pixel_via_the_low_threshold() {
        let mut image = noisy_gradient(32, 32);
        let dead = 9 * 32 + 21;
        image.data[dead] = 0.0;
        let replaced =
            suppress_impulses(&mut image, None, &ImpulseFilterOptions::default()).unwrap();
        assert_eq!(replaced, 1);
        assert!(image.data[dead] > 100.0);
    }

    #[test]
    fn leaves_smooth_structure_alone() {
        let mut image = noisy_gradient(64, 64);
        let original = image.data.clone();
        let replaced =
            suppress_impulses(&mut image, None, &ImpulseFilterOptions::default()).unwrap();
        assert_eq!(replaced, 0);
        assert_eq!(image.data, original);
    }

    #[test]
    fn a_cfa_frame_compares_only_its_own_plane() {
        // Alternating planes: even columns near 1000, odd columns near 100.
        // A same-plane filter sees each plane as flat; a naive 3x3 filter
        // would flag half the frame.
        let width = 32;
        let height = 32;
        let data = (0..width * height)
            .map(|index| {
                let x = index % width;
                let jitter = ((index * 2_654_435_761) % 89) as f32 / 89.0 - 0.5;
                if x % 2 == 0 {
                    1_000.0 + jitter
                } else {
                    100.0 + jitter
                }
            })
            .collect();
        let mut image = LinearImage::new(width, height, 1, data).unwrap();
        let layout = BayerLayout {
            pattern: seiza_fits::BayerPattern::Rggb,
            x_offset: 0,
            y_offset: 0,
        };
        let hot = 15 * width + 14;
        image.data[hot] = 50_000.0;
        let replaced =
            suppress_impulses(&mut image, Some(layout), &ImpulseFilterOptions::default()).unwrap();
        assert_eq!(replaced, 1, "only the injected hot photosite may change");
        assert!((image.data[hot] - 1_000.0).abs() < 2.0);
    }

    #[test]
    fn rejects_invalid_sigmas_and_cfa_multichannel() {
        let mut image = noisy_gradient(16, 16);
        assert!(
            suppress_impulses(
                &mut image,
                None,
                &ImpulseFilterOptions {
                    low_sigma: 0.0,
                    high_sigma: 16.0
                }
            )
            .is_err()
        );
        let mut rgb = LinearImage::new(4, 4, 3, vec![1.0; 48]).unwrap();
        let layout = BayerLayout {
            pattern: seiza_fits::BayerPattern::Rggb,
            x_offset: 0,
            y_offset: 0,
        };
        assert!(
            suppress_impulses(&mut rgb, Some(layout), &ImpulseFilterOptions::default()).is_err()
        );
    }
}
