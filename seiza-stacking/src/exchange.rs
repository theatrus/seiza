//! Putting a linear image on the 0 to 1 that external tools work on, and
//! taking their result back.
//!
//! PixInsight and the tools that follow its conventions, such as RC-Astro's,
//! work on samples from 0 to 1 and clamp anything outside. An image already
//! on 0 to 1 keeps its scale: one with no sample below -1 and at most one
//! sample in a thousand ([`OVERSHOOT_SHARE`]) above 1, such as a normalized
//! stack whose saturated star cores overshoot a little through resampling
//! and rejection. It is offset by its most negative sample, if it has one,
//! and divided by 1 minus that offset, so 1 stays 1; the samples above 1 go
//! to the tool at 1, as PixInsight would clamp them. Any other image is
//! offset the same way and divided by 65535 or its span, whichever is
//! larger: how PixInsight opens a 16-bit camera frame. Dividing by the span
//! alone made a result depend on the image's brightest sample; a dim crop
//! then sat near the top of the range. Taking a 0 to 1 stack with 191
//! star-core samples at 1.023 for a camera frame put its sky at 2e-7, and
//! BlurXTerminator 2.6.11 returned an image of zeros.
//!
//! Samples that are not finite, such as the uncovered edges of a stack, are
//! given their channel's median plus noise of the channel's robust sigma on
//! the way out, and are put back unchanged on the way in. A noiseless fill
//! is not enough: PixInsight's MLDenoise sees a flat region as part of the
//! image and then barely denoises any of it.
//!
//! So are samples more than [`ARTIFACT_SIGMAS`] robust sigmas below their
//! channel's median, which no sky reaches: a stack's resampling rings to
//! thousands below zero beside a saturated star. Taken as the most
//! negative sample, a few dozen of them lifted the whole sky up the range
//! by as much as their depth, and on such a pedestal NoiseXTerminator and
//! MLDenoise barely denoised at all.

use crate::image::LinearImage;

/// The full scale an image on a camera's scale is mapped from: 16-bit
/// counts, which PixInsight also uses for an integer FITS file.
pub(crate) const CAMERA_FULL_SCALE: f32 = 65535.0;

/// How many of its channel's robust sigmas below the channel's median a
/// sample is taken for an artifact and kept out of the map, as a sample
/// that is not finite is. A sky, even an image's darkest, lies within a few.
pub(crate) const ARTIFACT_SIGMAS: f32 = 30.0;

/// The largest share of an image's kept samples that may lie above 1 with
/// the image still taken as on 0 to 1: one in a thousand, so its 99.9th
/// percentile is within the range. A camera frame's sky alone lies above 1.
pub(crate) const OVERSHOOT_SHARE: f64 = 0.001;

/// How an image is put on 0 to 1 and taken back.
#[derive(Debug, PartialEq)]
pub(crate) struct Exchange {
    /// Subtracted before scaling: the most negative sample kept, or 0.
    pub(crate) offset: f32,
    /// Divided by after subtracting the offset.
    pub(crate) scale: f32,
    /// Each channel's background: what a sample not kept is given, and
    /// where the artifacts below it start.
    fill: Vec<Background>,
}

/// A channel's background: its median and robust sigma.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Background {
    median: f32,
    sigma: f32,
}

impl Background {
    /// Whether `sample` takes part in the map: finite, and not an artifact
    /// far below the background. With no spread, every finite sample does.
    fn keeps(&self, sample: f32) -> bool {
        sample.is_finite()
            && (self.sigma <= 0.0 || sample >= self.median - ARTIFACT_SIGMAS * self.sigma)
    }
}

impl Exchange {
    /// The map for `image`, or none when the image has no finite sample.
    pub(crate) fn of(image: &LinearImage) -> Option<Self> {
        let channels = image.channels.max(1);
        let fill: Vec<Background> = (0..channels)
            .map(|channel| channel_background(&image.data, channel, channels))
            .collect();
        let mut peak = f32::NEG_INFINITY;
        let mut floor = f32::INFINITY;
        let (mut kept, mut above_one) = (0usize, 0usize);
        for (index, sample) in image.data.iter().enumerate() {
            if fill[index % channels].keeps(*sample) {
                peak = peak.max(*sample);
                floor = floor.min(*sample);
                kept += 1;
                above_one += usize::from(*sample > 1.0);
            }
        }
        if !peak.is_finite() || !floor.is_finite() {
            return None;
        }
        let offset = floor.min(0.0);
        let on_unit_range = floor >= -1.0 && above_one as f64 <= OVERSHOOT_SHARE * kept as f64;
        let scale = if on_unit_range {
            1.0 - offset
        } else {
            (peak - offset).max(CAMERA_FULL_SCALE)
        };
        Some(Self {
            offset,
            scale,
            fill,
        })
    }

    /// The image's samples on 0 to 1, with every sample the map does not
    /// keep replaced by its channel's background and noise, and a sample
    /// above the range (an image on 0 to 1's overshoot) at 1. The noise
    /// depends only on the sample's position, so the same image gives the
    /// same exchange.
    pub(crate) fn normalize(&self, image: &LinearImage) -> Vec<f32> {
        let channels = self.fill.len();
        image
            .data
            .iter()
            .enumerate()
            .map(|(index, sample)| {
                let background = self.fill[index % channels];
                let sample = if background.keeps(*sample) {
                    *sample
                } else {
                    background.median + background.sigma * standard_normal(index as u64)
                };
                ((sample - self.offset) / self.scale).min(1.0)
            })
            .collect()
    }

    /// The original sample that 1 on the tool's scale stands for.
    fn top(&self) -> f32 {
        self.scale + self.offset
    }

    /// Put an image the tool returned back on the original's scale, and the
    /// original's samples the map did not keep back in place. A sample that
    /// went out at 1 takes the tool's result, as any kept sample does: the
    /// tool saw a star core there, and a starless image must not keep it.
    pub(crate) fn restore(&self, processed: &mut [f32], original: &[f32]) {
        let channels = self.fill.len();
        for (index, (sample, original)) in processed.iter_mut().zip(original).enumerate() {
            *sample = if self.fill[index % channels].keeps(*original) {
                *sample * self.scale + self.offset
            } else {
                *original
            };
        }
    }

    /// [`Self::restore`] for a difference of two images on 0 to 1, such as
    /// StarXTerminator's stars image (the original minus the starless
    /// result): the offset cancels out of a difference, so only the scale
    /// is restored. A sample that is not finite stays a gap; an artifact,
    /// which the result keeps as it was, differs by nothing. A sample that
    /// went out at 1 adds what it lost there, so the result plus the
    /// difference still gives back the original.
    pub(crate) fn restore_difference(&self, processed: &mut [f32], original: &[f32]) {
        let channels = self.fill.len();
        let top = self.top();
        for (index, (sample, original)) in processed.iter_mut().zip(original).enumerate() {
            *sample = if !original.is_finite() {
                *original
            } else if self.fill[index % channels].keeps(*original) {
                *sample * self.scale + (*original - top).max(0.0)
            } else {
                0.0
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

/// [`standard_normal`] for other modules' tests.
#[cfg(test)]
pub(crate) fn tests_noise(position: u64) -> f32 {
    standard_normal(position)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The map as [`crate::EXTERNAL_EXCHANGE_VERSION`] 4 defines it. A change
    /// that fails this test changes external tools' results for the same
    /// input: move the version, then record the new values.
    #[test]
    fn the_map_matches_its_version() {
        assert_eq!(crate::EXTERNAL_EXCHANGE_VERSION, 4);
        let image = LinearImage::new(2, 2, 1, vec![f32::NAN, 1_000.0, 1_100.0, -20.0]).unwrap();
        let exchange = Exchange::of(&image).unwrap();
        assert_eq!((exchange.offset, exchange.scale), (-20.0, 65535.0));
        let normalized = exchange.normalize(&image);
        assert_eq!(normalized[1], 1_020.0 / 65535.0);
        assert_eq!(normalized[3], 0.0);
        // A sample 30 robust sigmas below the median neither offsets the
        // map nor widens it (version 2 took -90000 as the offset).
        let ringing =
            LinearImage::new(5, 1, 1, vec![f32::NAN, 1_000.0, 1_100.0, -20.0, -90_000.0]).unwrap();
        let exchange = Exchange::of(&ringing).unwrap();
        assert_eq!((exchange.offset, exchange.scale), (-20.0, 65535.0));
        assert_eq!(
            exchange.fill,
            vec![Background {
                median: 1_000.0,
                sigma: 1.4826 * 1_020.0
            }]
        );
        // A 0 to 1 image with one sample in a thousand above 1 keeps its
        // scale, and a slight negative offsets it on that scale (version 3
        // took it for a camera frame and divided by 65535).
        let mut unit: Vec<f32> = (0..1_000).map(|i| 0.01 + i as f32 * 1e-4).collect();
        unit[0] = -0.002;
        unit[1] = 1.023;
        let unit = LinearImage::new(1_000, 1, 1, unit).unwrap();
        let exchange = Exchange::of(&unit).unwrap();
        assert_eq!((exchange.offset, exchange.scale), (-0.002, 1.0 + 0.002));
        let normalized = exchange.normalize(&unit);
        assert_eq!((normalized[0], normalized[1]), (0.0, 1.0));
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

    #[test]
    fn a_normalized_stack_whose_star_cores_overshoot_keeps_its_scale() {
        // A 0 to 1 stack's sky at 0.013, and 191 saturated star-core samples
        // that resampling and rejection left at 1.023.
        let mut data: Vec<f32> = (0..200_000)
            .map(|i| 0.013 + 0.0005 * standard_normal(i + 11))
            .collect();
        for core in 0..191 {
            data[997 * core + 500] = 1.023;
        }
        let image = LinearImage::new(1_000, 200, 1, data.clone()).unwrap();
        let exchange = Exchange::of(&image).unwrap();
        assert_eq!((exchange.offset, exchange.scale), (0.0, 1.0));
        // The sky reaches the tool where it is, not at 2e-7, and the cores
        // at 1, as PixInsight would clamp them.
        let normalized = exchange.normalize(&image);
        assert_eq!(normalized[0], data[0]);
        assert_eq!(normalized[500], 1.0);
        // The tool's result comes back as it is, cores included: a starless
        // image has no core left to put back.
        let mut processed = normalized.clone();
        processed[500] = 0.013;
        exchange.restore(&mut processed, &data);
        assert_eq!((processed[0], processed[500]), (data[0], 0.013));
        // The stars image takes what the cores lost, so starless plus stars
        // still gives back the stack.
        let mut stars: Vec<f32> = normalized
            .iter()
            .zip(&processed)
            .map(|(n, p)| n - p)
            .collect();
        exchange.restore_difference(&mut stars, &data);
        assert!((processed[500] + stars[500] - 1.023).abs() < 1e-6);
        assert_eq!(stars[0], 0.0);

        // More than one sample in a thousand above 1 is a camera frame's,
        // and so is a single sample below -1.
        for core in 191..201 {
            data[997 * core + 500] = 1.023;
        }
        let image = LinearImage::new(1_000, 200, 1, data.clone()).unwrap();
        assert_eq!(Exchange::of(&image).unwrap().scale, CAMERA_FULL_SCALE);
        let below = LinearImage::new(2, 2, 1, vec![-1.5, 0.1, 0.2, 0.3]).unwrap();
        assert_eq!(
            Exchange::of(&below).unwrap(),
            Exchange {
                offset: -1.5,
                scale: CAMERA_FULL_SCALE,
                fill: vec![channel_background(&below.data, 0, 1)],
            }
        );
    }

    #[test]
    fn ringing_far_below_the_sky_neither_lifts_it_nor_reaches_the_tool() {
        // A sky of 130 with noise of 10, a saturated star, and the few deep
        // negatives a stack's resampling leaves beside it.
        let mut data: Vec<f32> = (0..10_000)
            .map(|i| 130.0 + 10.0 * standard_normal(i + 7))
            .collect();
        data[5_000] = 68_900.0;
        data[5_001] = -7_300.0;
        data[5_099] = -4_100.0;
        let image = LinearImage::new(100, 100, 1, data.clone()).unwrap();
        let exchange = Exchange::of(&image).unwrap();
        // The sky stays at the foot of the range, where it would be without
        // them, not lifted by 7300; the star's peak sets the scale.
        assert_eq!((exchange.offset, exchange.scale), (0.0, 68_900.0));
        let normalized = exchange.normalize(&image);
        assert!((normalized[0] - data[0] / 68_900.0).abs() < 1e-9);
        // They go out as background with noise, inside the range.
        for index in [5_001, 5_099] {
            assert!(normalized[index] > 50.0 / 68_900.0 && normalized[index] < 250.0 / 68_900.0);
        }
        // And come back as they were; the stars image shows nothing there.
        let mut processed = normalized.clone();
        exchange.restore(&mut processed, &data);
        assert_eq!((processed[5_001], processed[5_099]), (-7_300.0, -4_100.0));
        assert!(
            (processed[0] - data[0]).abs() < 1e-3 && (processed[5_000] - 68_900.0).abs() < 1e-2
        );
        let mut stars = vec![0.25; data.len()];
        exchange.restore_difference(&mut stars, &data);
        assert_eq!((stars[5_001], stars[5_099]), (0.0, 0.0));
        assert!((stars[0] - 0.25 * 68_900.0).abs() < 1e-2);

        // An image of nothing but one value keeps every sample.
        let flat = LinearImage::new(2, 2, 1, vec![-5.0; 4]).unwrap();
        let exchange = Exchange::of(&flat).unwrap();
        assert_eq!((exchange.offset, exchange.scale), (-5.0, 65535.0));
    }
}
