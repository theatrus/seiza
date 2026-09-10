use super::{MasterBuildOptions, check_cancelled};
use crate::{Error, FitsFrame, LinearImage, Result};
use seiza_imgproc::components::{Connectivity, connected_components};
use serde::{Deserialize, Serialize};

/// Native per-input star masking for sky-flat construction. Pixels are excluded,
/// never replaced. All radii are measured on the original sensor grid.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FlatStarMaskingOptions {
    pub detection_sigma: f32,
    pub halo_radius_factor: f32,
    pub minimum_radius_pixels: f32,
    /// Explicit saturation ceiling in the input's raw sample units. Without an
    /// override, only SATURATE/SATLEVEL or an integer FITS encoding is used.
    pub saturation_level: Option<f32>,
    /// Required unmasked retained samples after sigma clipping, at every output
    /// sample. Two permits a mean without clipping; three permits robust clipping.
    pub minimum_clean_samples: usize,
}

impl Default for FlatStarMaskingOptions {
    fn default() -> Self {
        Self {
            detection_sigma: 4.0,
            halo_radius_factor: 3.0,
            minimum_radius_pixels: 6.0,
            saturation_level: None,
            minimum_clean_samples: 2,
        }
    }
}

impl FlatStarMaskingOptions {
    pub(super) fn validate(&self) -> Result<()> {
        if !self.detection_sigma.is_finite()
            || self.detection_sigma <= 0.0
            || !self.halo_radius_factor.is_finite()
            || self.halo_radius_factor < 1.0
            || !self.minimum_radius_pixels.is_finite()
            || self.minimum_radius_pixels < 1.0
            || self
                .saturation_level
                .is_some_and(|value| !value.is_finite() || value <= 0.0)
            || self.minimum_clean_samples < 2
        {
            return Err(Error::FlatStarMasking("invalid star-mask thresholds, radius, or minimum coverage (at least two retained inputs required)".into()));
        }
        Ok(())
    }
}

/// Coverage describes unmasked retained samples, not a guarantee that all
/// remaining samples are linear or artifact-free.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FlatStarMaskingStatistics {
    pub options: FlatStarMaskingOptions,
    pub masked_samples: u64,
    /// Output sensor pixels touched by at least one input mask.
    pub masked_pixels: u64,
    pub minimum_clean_samples: usize,
    pub maximum_clean_samples: usize,
    /// Output samples with fewer than three retained inputs, hence no robust
    /// rejection or only minimal retained coverage.
    pub low_coverage_samples: u64,
    /// Isolated saturated detector impulses left to existing defect handling,
    /// unless another detected star footprint covered them.
    pub unmasked_saturation_samples: u64,
    /// Inputs whose encoding/headers did not establish a saturation ceiling.
    pub unknown_saturation_inputs: usize,
}

struct Circle {
    x: f64,
    y: f64,
    radius: f64,
}

pub(super) struct SaturationSeeds {
    pub known_ceiling: bool,
    circles: Vec<Circle>,
    impulses: Vec<(usize, u64)>,
}

fn saturation_level(frame: &FitsFrame, options: &FlatStarMaskingOptions) -> Option<f32> {
    if let Some(level) = options.saturation_level {
        return Some(level);
    }
    let number = |name: &str| {
        frame
            .headers
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

pub(super) fn saturation_seeds(
    frame: &FitsFrame,
    masking: &FlatStarMaskingOptions,
    options: &MasterBuildOptions,
) -> Result<SaturationSeeds> {
    let Some(level) = saturation_level(frame, masking) else {
        return Ok(SaturationSeeds {
            known_ceiling: false,
            circles: Vec::new(),
            impulses: Vec::new(),
        });
    };
    let image = &frame.image;
    let mut saturated = vec![0_u8; image.width * image.height];
    for (index, pixel) in image.data.chunks_exact(image.channels).enumerate() {
        if index % 65536 == 0 {
            check_cancelled(options)?;
        }
        saturated[index] = u8::from(pixel.iter().any(|value| *value >= level));
    }
    check_cancelled(options)?;
    let components =
        connected_components(&saturated, image.width, image.height, Connectivity::Eight);
    let mut seeds = SaturationSeeds {
        known_ceiling: true,
        circles: Vec::new(),
        impulses: Vec::new(),
    };
    for component in components {
        check_cancelled(options)?;
        if component.pixels.len() >= 3 {
            // A bounding-circle radius also covers elongated saturated cores,
            // independent of the star detector's roundness admission gate.
            let radius = f64::from(masking.halo_radius_factor)
                * ((component.width() as f64).hypot(component.height() as f64) / 2.0);
            seeds.circles.push(Circle {
                x: (component.min_x + component.max_x) as f64 / 2.0,
                y: (component.min_y + component.max_y) as f64 / 2.0,
                radius: radius.max(f64::from(masking.minimum_radius_pixels)),
            });
        } else {
            for index in component.pixels {
                let start = index * image.channels;
                let count = image.data[start..start + image.channels]
                    .iter()
                    .filter(|value| **value >= level)
                    .count() as u64;
                seeds.impulses.push((index, count));
            }
        }
    }
    Ok(seeds)
}

pub(super) fn mask_stars(
    image: &mut LinearImage,
    masking: &FlatStarMaskingOptions,
    mut saturation: SaturationSeeds,
    options: &MasterBuildOptions,
) -> Result<u64> {
    let width = image.width.div_ceil(2);
    let height = image.height.div_ceil(2);
    let mut luma = vec![0.0; width * height];
    // Analysis only: the 2x2 average removes raw-CFA phase alternation without
    // demosaicing or modifying the original samples used by the master.
    for y in 0..height {
        check_cancelled(options)?;
        for x in 0..width {
            let mut sum = 0.0_f64;
            let mut count = 0;
            for raw_y in y * 2..(y * 2 + 2).min(image.height) {
                for raw_x in x * 2..(x * 2 + 2).min(image.width) {
                    let start = (raw_y * image.width + raw_x) * image.channels;
                    for value in &image.data[start..start + image.channels] {
                        sum += f64::from(*value);
                        count += 1;
                    }
                }
            }
            luma[y * width + x] = (sum / count as f64) as f32;
        }
    }
    let (minimum, maximum) = luma
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), &value| {
            (low.min(value), high.max(value))
        });
    if maximum > minimum {
        let range = f64::from(maximum) - f64::from(minimum);
        for value in &mut luma {
            *value = ((f64::from(*value) - f64::from(minimum)) / range) as f32;
        }
        check_cancelled(options)?;
        let config = seiza::DetectConfig {
            backend: seiza::DetectBackend::F32,
            sigma: masking.detection_sigma,
            min_area: 3,
            max_area: u32::MAX,
            max_stars: usize::MAX,
            ..Default::default()
        };
        let stars = seiza::detect_stars_luma_f32(&luma, width as u32, height as u32, &config);
        check_cancelled(options)?;
        for star in stars {
            saturation.circles.push(Circle {
                x: (star.x * 2.0 + 0.5).min(image.width as f64 - 1.0),
                y: (star.y * 2.0 + 0.5).min(image.height as f64 - 1.0),
                radius: (f64::from(masking.halo_radius_factor)
                    * 2.0
                    * (f64::from(star.area) / std::f64::consts::PI).sqrt())
                .max(f64::from(masking.minimum_radius_pixels)),
            });
        }
    }
    drop(luma);
    let mut mask = vec![false; image.width * image.height];
    for circle in saturation.circles {
        check_cancelled(options)?;
        let x_start = (circle.x - circle.radius).floor().max(0.0) as usize;
        let x_end = ((circle.x + circle.radius).ceil() as usize).min(image.width - 1);
        let y_start = (circle.y - circle.radius).floor().max(0.0) as usize;
        let y_end = ((circle.y + circle.radius).ceil() as usize).min(image.height - 1);
        let squared = circle.radius * circle.radius;
        for y in y_start..=y_end {
            check_cancelled(options)?;
            for x in x_start..=x_end {
                if (x as f64 - circle.x).powi(2) + (y as f64 - circle.y).powi(2) <= squared {
                    mask[y * image.width + x] = true;
                }
            }
        }
    }
    let unmasked_saturation_samples = saturation
        .impulses
        .iter()
        .filter(|(index, _)| !mask[*index])
        .map(|(_, count)| *count)
        .sum();
    for (index, pixel) in image.data.chunks_exact_mut(image.channels).enumerate() {
        if index % 65536 == 0 {
            check_cancelled(options)?;
        }
        if mask[index] {
            pixel.fill(f32::NAN);
        }
    }
    Ok(unmasked_saturation_samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use seiza_fits::HeaderValue;

    fn frame(width: usize, height: usize, channels: usize) -> FitsFrame {
        FitsFrame {
            image: LinearImage::new(
                width,
                height,
                channels,
                vec![1000.0; width * height * channels],
            )
            .unwrap(),
            headers: Vec::new(),
            exposure_seconds: None,
            bayer: None,
            source: None,
            bounds: None,
        }
    }

    fn mask(frame: &mut FitsFrame, masking: &FlatStarMaskingOptions) -> u64 {
        let options = MasterBuildOptions::default();
        let saturation = saturation_seeds(frame, masking, &options).unwrap();
        mask_stars(&mut frame.image, masking, saturation, &options).unwrap()
    }

    #[test]
    fn saturation_uses_raw_encoding_or_explicit_metadata_not_observed_ranges() {
        let mut frame = frame(4, 4, 1);
        let options = FlatStarMaskingOptions::default();
        frame.bounds = Some((0.0, 65535.0));
        frame
            .headers
            .push(("DATAMAX".into(), HeaderValue::Float(65535.0)));
        assert_eq!(saturation_level(&frame, &options), None);
        frame.headers.extend([
            ("BITPIX".into(), HeaderValue::Integer(16)),
            ("BZERO".into(), HeaderValue::Float(32768.0)),
        ]);
        assert_eq!(saturation_level(&frame, &options), Some(65535.0));
        frame
            .headers
            .push(("SATURATE".into(), HeaderValue::Float(60000.0)));
        assert_eq!(saturation_level(&frame, &options), Some(60000.0));
        assert_eq!(
            saturation_level(
                &frame,
                &FlatStarMaskingOptions {
                    saturation_level: Some(50000.0),
                    ..options
                }
            ),
            Some(50000.0)
        );
    }

    #[test]
    fn isolated_saturation_is_reported_without_creating_a_star_hole() {
        let mut frame = frame(64, 64, 1);
        let index = 30 * 64 + 30;
        frame.image.data[index] = 65535.0;
        let ignored = mask(
            &mut frame,
            &FlatStarMaskingOptions {
                saturation_level: Some(65535.0),
                ..Default::default()
            },
        );
        assert_eq!(ignored, 1);
        assert!(frame.image.data.iter().all(|value| value.is_finite()));
        assert_eq!(frame.image.data[index], 65535.0);
    }

    #[test]
    fn broad_saturation_expands_even_for_an_elongated_core_at_an_odd_edge() {
        let mut frame = frame(65, 63, 3);
        for y in 50..63 {
            frame.image.data[(y * 65 + 64) * 3 + 1] = 65535.0;
        }
        mask(
            &mut frame,
            &FlatStarMaskingOptions {
                saturation_level: Some(65535.0),
                ..Default::default()
            },
        );
        for y in 50..63 {
            assert!(
                frame.image.data[(y * 65 + 64) * 3..(y * 65 + 64) * 3 + 3]
                    .iter()
                    .all(|value| value.is_nan())
            );
        }
        assert!(
            frame
                .image
                .data
                .iter()
                .take(3)
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn native_star_footprint_masks_all_channels_and_preserves_other_samples() {
        let mut frame = frame(128, 128, 3);
        for y in 0..128 {
            for x in 0..128 {
                let signal = 10000.0
                    * (-((x as f64 - 65.5).powi(2) + (y as f64 - 63.5).powi(2)) / 8.0).exp() as f32;
                frame.image.data[(y * 128 + x) * 3] += signal;
            }
        }
        mask(&mut frame, &FlatStarMaskingOptions::default());
        let core = (64 * 128 + 66) * 3;
        assert!(
            frame.image.data[core..core + 3]
                .iter()
                .all(|value| value.is_nan())
        );
        assert!(
            frame.image.data[(64 * 128 + 71) * 3].is_nan(),
            "halo must be masked beyond the core"
        );
        assert_eq!(&frame.image.data[..3], &[1000.0; 3]);
    }
}
