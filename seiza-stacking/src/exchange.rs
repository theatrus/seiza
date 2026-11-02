//! Putting a linear image on the 0 to 1 that external tools work on, and
//! taking their result back.
//!
//! PixInsight and the tools that follow its conventions, such as RC-Astro's,
//! work on samples from 0 to 1 and clamp anything outside. An image whose
//! samples already lie within 0 to 1 keeps its scale. Any other image is
//! offset by its most negative sample, if it has one, and divided by 65535
//! or its span, whichever is larger: how PixInsight opens a 16-bit camera
//! frame. Dividing by the span alone made a result depend on the image's
//! brightest sample; a dim crop then sat near the top of the range.
//!
//! Samples that are not finite, such as the uncovered edges of a stack, are
//! given their channel's median plus noise of the channel's robust sigma on
//! the way out, and are put back unchanged on the way in. A noiseless fill
//! is not enough: PixInsight's MLDenoise sees a flat region as part of the
//! image and then barely denoises any of it.

use crate::image::LinearImage;

/// The full scale an image on a camera's scale is mapped from: 16-bit
/// counts, which PixInsight also uses for an integer FITS file.
pub(crate) const CAMERA_FULL_SCALE: f32 = 65535.0;

/// How an image is put on 0 to 1 and taken back.
#[derive(Debug, PartialEq)]
pub(crate) struct Exchange {
    /// Subtracted before scaling: the most negative sample, or 0.
    pub(crate) offset: f32,
    /// Divided by after subtracting the offset.
    pub(crate) scale: f32,
    /// What a sample that is not finite is given, per channel.
    fill: Vec<Background>,
}

/// A channel's background: its median and robust sigma.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Background {
    median: f32,
    sigma: f32,
}

impl Exchange {
    /// The map for `image`, or none when the image has no finite sample.
    pub(crate) fn of(image: &LinearImage) -> Option<Self> {
        let mut peak = f32::NEG_INFINITY;
        let mut floor = f32::INFINITY;
        for sample in &image.data {
            if sample.is_finite() {
                peak = peak.max(*sample);
                floor = floor.min(*sample);
            }
        }
        if !peak.is_finite() || !floor.is_finite() {
            return None;
        }
        let (offset, scale) = if floor >= 0.0 && peak <= 1.0 {
            (0.0, 1.0)
        } else {
            let offset = floor.min(0.0);
            (offset, (peak - offset).max(CAMERA_FULL_SCALE))
        };
        let channels = image.channels.max(1);
        let fill = (0..channels)
            .map(|channel| channel_background(&image.data, channel, channels))
            .collect();
        Some(Self {
            offset,
            scale,
            fill,
        })
    }

    /// The image's samples on 0 to 1, with every sample that is not finite
    /// replaced by its channel's background and noise. The noise depends
    /// only on the sample's position, so the same image gives the same
    /// exchange.
    pub(crate) fn normalize(&self, image: &LinearImage) -> Vec<f32> {
        let channels = self.fill.len();
        image
            .data
            .iter()
            .enumerate()
            .map(|(index, sample)| {
                let sample = if sample.is_finite() {
                    *sample
                } else {
                    let background = self.fill[index % channels];
                    background.median + background.sigma * standard_normal(index as u64)
                };
                (sample - self.offset) / self.scale
            })
            .collect()
    }

    /// Put an image the tool returned back on the original's scale, and the
    /// original's samples that were not finite back in place.
    pub(crate) fn restore(&self, processed: &mut [f32], original: &[f32]) {
        for (sample, original) in processed.iter_mut().zip(original) {
            *sample = if original.is_finite() {
                *sample * self.scale + self.offset
            } else {
                *original
            };
        }
    }

    /// [`Self::restore`] for a difference of two images on 0 to 1, such as
    /// StarXTerminator's stars image (the original minus the starless
    /// result): the offset cancels out of a difference, so only the scale
    /// is restored.
    pub(crate) fn restore_difference(&self, processed: &mut [f32], original: &[f32]) {
        for (sample, original) in processed.iter_mut().zip(original) {
            *sample = if original.is_finite() {
                *sample * self.scale
            } else {
                *original
            };
        }
    }
}

/// One channel's background from at most about a million of its finite
/// samples: their median, and 1.4826 times their median absolute deviation.
fn channel_background(data: &[f32], channel: usize, channels: usize) -> Background {
    let count = data.len() / channels;
    let step = (count / 1_000_000).max(1);
    let mut samples: Vec<f32> = (0..count)
        .step_by(step)
        .map(|pixel| data[pixel * channels + channel])
        .filter(|sample| sample.is_finite())
        .collect();
    if samples.is_empty() {
        return Background {
            median: 0.0,
            sigma: 0.0,
        };
    }
    let middle = samples.len() / 2;
    let median = *samples.select_nth_unstable_by(middle, f32::total_cmp).1;
    for sample in &mut samples {
        *sample = (*sample - median).abs();
    }
    let deviation = *samples.select_nth_unstable_by(middle, f32::total_cmp).1;
    Background {
        median,
        sigma: 1.4826 * deviation,
    }
}

/// A standard normal value that depends only on `position`: Box-Muller over
/// two splitmix64 values.
fn standard_normal(position: u64) -> f32 {
    let unit = |seed: u64| {
        let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let (u, v) = (unit(position * 2), unit(position * 2 + 1));
    ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The map as [`crate::EXTERNAL_EXCHANGE_VERSION`] 2 defines it. A change
    /// that fails this test changes external tools' results for the same
    /// input: move the version, then record the new values.
    #[test]
    fn the_map_matches_its_version() {
        assert_eq!(crate::EXTERNAL_EXCHANGE_VERSION, 2);
        let image = LinearImage::new(2, 2, 1, vec![f32::NAN, 1_000.0, 1_100.0, -20.0]).unwrap();
        let exchange = Exchange::of(&image).unwrap();
        assert_eq!((exchange.offset, exchange.scale), (-20.0, 65535.0));
        let normalized = exchange.normalize(&image);
        assert_eq!(normalized[1], 1_020.0 / 65535.0);
        assert_eq!(normalized[3], 0.0);
        // The fill's noise at the first two positions. A tolerance, since
        // platforms' logarithm and cosine can differ in the last bit.
        assert!((standard_normal(0) - -0.455_219).abs() < 1e-6);
        assert!((standard_normal(1) - 0.775_653).abs() < 1e-6);
    }

    #[test]
    fn a_normalized_image_keeps_its_scale_and_a_camera_image_maps_from_16_bits() {
        let image = |data: Vec<f32>| LinearImage::new(2, 2, 1, data).unwrap();
        let normalized = Exchange::of(&image(vec![0.0, 0.1, 0.5, 1.0])).unwrap();
        assert_eq!((normalized.offset, normalized.scale), (0.0, 1.0));
        // A dim camera image keeps its place on 16 bits, whatever its peak.
        let dim = Exchange::of(&image(vec![900.0, 1000.0, 1100.0, 1200.0])).unwrap();
        assert_eq!((dim.offset, dim.scale), (0.0, 65535.0));
        let negative = Exchange::of(&image(vec![-100.0, 0.0, 900.0, 70000.0])).unwrap();
        assert_eq!((negative.offset, negative.scale), (-100.0, 70100.0));
        assert!(Exchange::of(&image(vec![f32::NAN; 4])).is_none());
    }

    #[test]
    fn samples_that_are_not_finite_go_out_as_background_and_come_back_unchanged() {
        let data = vec![100.0, f32::NAN, 300.0, f32::INFINITY, 500.0, 700.0];
        let image = LinearImage::new(3, 2, 1, data.clone()).unwrap();
        let exchange = Exchange::of(&image).unwrap();
        assert_eq!(
            exchange.fill,
            vec![Background {
                median: 500.0,
                sigma: 1.4826 * 200.0
            }]
        );
        let normalized = exchange.normalize(&image);
        assert!(normalized.iter().all(|sample| sample.is_finite()));
        // Background with noise, the same each time.
        assert_ne!(normalized[1], 500.0 / 65535.0);
        assert_eq!(normalized, exchange.normalize(&image));

        let mut processed = normalized.clone();
        exchange.restore(&mut processed, &data);
        assert!(processed[1].is_nan());
        assert_eq!(processed[3], f32::INFINITY);
        assert!((processed[4] - 500.0).abs() < 1e-3);

        // A difference takes the scale alone, and keeps the gaps.
        let mut difference = vec![0.5; 6];
        exchange.restore_difference(&mut difference, &data);
        assert!(difference[1].is_nan());
        assert!((difference[0] - 0.5 * 65535.0).abs() < 1e-2);

        // RGB takes each channel's own background.
        let rgb = LinearImage::new(
            3,
            1,
            3,
            vec![10.0, 20.0, 30.0, f32::NAN, 22.0, f32::NAN, 12.0, 24.0, 36.0],
        )
        .unwrap();
        let exchange = Exchange::of(&rgb).unwrap();
        let medians: Vec<f32> = exchange.fill.iter().map(|fill| fill.median).collect();
        assert_eq!(medians, vec![12.0, 22.0, 36.0]);

        // The noise is standard normal.
        let draws: Vec<f32> = (0..20_000).map(standard_normal).collect();
        let mean = draws.iter().sum::<f32>() / draws.len() as f32;
        let variance = draws.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / draws.len() as f32;
        assert!(
            mean.abs() < 0.03 && (variance - 1.0).abs() < 0.05,
            "{mean} {variance}"
        );
    }
}
