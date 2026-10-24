//! Display-only narrowband balance and replacement of narrowband stars.
use crate::{Error, LinearImage, Result};
use rayon::prelude::*;
use seiza_stretch::midtones_transfer_function;
use serde::{Deserialize, Serialize};

/// Aesthetic midtone balance, with H-alpha as the reference.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NarrowbandBalanceOptions {
    /// Whether the third plane is physical SII. For HOO, pass OIII in the
    /// third plane and set this false; only H-alpha/OIII are measured.
    pub has_sii: bool,
    /// 0 protects the minimum; 1 protects the median background.
    pub shadow_protection: f64,
    /// Additional OIII boost after matching to H-alpha; 1 adds no boost.
    pub oiii_boost: f64,
    /// Additional SII boost after matching to H-alpha; 1 adds no boost.
    pub sii_boost: f64,
    /// Maximum common pixels sampled for statistics.
    pub max_samples: usize,
}

impl Default for NarrowbandBalanceOptions {
    fn default() -> Self {
        Self {
            has_sii: true,
            shadow_protection: 1.0,
            oiii_boost: 1.0,
            sii_boost: 1.0,
            max_samples: 1_000_000,
        }
    }
}

/// Statistics and the applied curve for one physical emission channel.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NarrowbandChannelBalance {
    /// Physical filter, independent of the eventual RGB palette.
    pub channel: String,
    /// `reference`, `applied`, `identity`, or `skipped`.
    pub status: String,
    /// Why a curve could not be measured or applied.
    pub reason: Option<String>,
    pub samples: usize,
    pub median: Option<f64>,
    pub background: Option<f64>,
    pub positive_deviation: Option<f64>,
    pub midtone: Option<f64>,
}

/// Resolved display curves; a skipped channel is left unchanged.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NarrowbandBalanceReport {
    pub channels: Vec<NarrowbandChannelBalance>,
}

fn invalid(message: &str) -> Error {
    Error::Color(message.into())
}

fn valid_image(image: &LinearImage, channels: &[usize]) -> Result<()> {
    if !channels.contains(&image.channels)
        || image.width == 0
        || image.height == 0
        || image
            .width
            .checked_mul(image.height)
            .and_then(|n| n.checked_mul(image.channels))
            != Some(image.data.len())
    {
        return Err(invalid(
            "Image dimensions or channel count do not match this display operation",
        ));
    }
    if image
        .data
        .iter()
        .any(|v| v.is_finite() && !(0.0..=1.0).contains(v))
    {
        return Err(invalid(
            "Display samples must be in [0, 1]; stretch linear sensor data first",
        ));
    }
    Ok(())
}

/// Balance interleaved display channels in physical order H-alpha, OIII,
/// optionally SII. This applies nonlinear curves, so the output is
/// display-referred. Do not use it for photometry or continuum fitting.
///
/// Statistics use common finite coverage and deterministic bounded sampling.
/// Positive deviation is the mean of strictly positive median deviations.
/// A 99th-percentile signal less than four pixel-difference noise sigmas is
/// skipped, rather than boosting a noise-only channel. This is a conservative
/// display heuristic, not evidence that an emission line is absent.
/// Invalid options or samples fail before any pixel is changed.
pub fn balance_narrowband(
    image: &mut LinearImage,
    options: &NarrowbandBalanceOptions,
) -> Result<NarrowbandBalanceReport> {
    valid_image(image, &[3])?;
    if !options.shadow_protection.is_finite()
        || !(0.0..=1.0).contains(&options.shadow_protection)
        || [options.oiii_boost, options.sii_boost]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
        || !(4..=1_000_000).contains(&options.max_samples)
    {
        return Err(invalid(
            "Shadow protection must be 0 to 1, boosts positive and finite, and max_samples 4 to 1000000",
        ));
    }
    let stride = image.channels;
    let count = if options.has_sii { 3 } else { 2 };
    let step = image.pixel_count().div_ceil(options.max_samples).max(1);
    let indices: Vec<_> = (0..image.pixel_count())
        .step_by(step)
        .filter(|i| {
            image.data[i * stride..i * stride + count]
                .iter()
                .all(|v| v.is_finite())
        })
        .collect();
    let mut reports = Vec::with_capacity(count);
    for channel in 0..count {
        let mut values: Vec<f64> = indices
            .iter()
            .map(|i| f64::from(image.data[i * stride + channel]))
            .collect();
        values.sort_unstable_by(f64::total_cmp);
        let mut report = NarrowbandChannelBalance {
            channel: ["ha", "oiii", "sii"][channel].into(),
            status: "skipped".into(),
            reason: None,
            samples: values.len(),
            median: None,
            background: None,
            positive_deviation: None,
            midtone: None,
        };
        if values.len() < 4 {
            report.reason = Some("too_few_samples".into());
            reports.push(report);
            continue;
        }
        let q = (values[(values.len() - 1) / 2] + values[values.len() / 2]) / 2.0;
        let positive: Vec<_> = values.iter().filter(|v| **v > q).map(|v| *v - q).collect();
        let d = if positive.is_empty() {
            0.0
        } else {
            positive.iter().sum::<f64>() / positive.len() as f64
        };
        let b = (1.0 - options.shadow_protection) * values[0] + options.shadow_protection * q;
        report.median = Some(q);
        report.background = Some(b);
        report.positive_deviation = Some(d);
        let mut differences: Vec<f64> = indices
            .iter()
            .filter_map(|i| {
                if i % image.width + 1 == image.width {
                    return None;
                }
                let next = image.data[(i + 1) * stride + channel];
                next.is_finite()
                    .then(|| (f64::from(next) - f64::from(image.data[i * stride + channel])).abs())
            })
            .collect();
        differences.sort_unstable_by(f64::total_cmp);
        let noise = if differences.is_empty() {
            0.0
        } else {
            differences[differences.len() / 2] / 0.9538725524
        };
        let tail = values[(values.len() - 1) * 99 / 100] - q;
        if d <= f64::EPSILON || b >= 1.0 || tail <= 4.0 * noise {
            report.reason = Some("too_little_signal".into());
        } else {
            report.status = if channel == 0 {
                "reference"
            } else {
                "identity"
            }
            .into();
        }
        reports.push(report);
    }
    let reference = &reports[0];
    let target = reference
        .background
        .zip(reference.median.zip(reference.positive_deviation))
        .map(|(b, (q, d))| (q + d - b) / (1.0 - b));
    let reference_ok = reference.status == "reference";
    for (channel, report) in reports.iter_mut().enumerate().skip(1) {
        if !reference_ok {
            report.status = "skipped".into();
            report.reason = Some("unusable_ha_reference".into());
            continue;
        }
        if report.status == "skipped" {
            continue;
        }
        let b = report.background.unwrap();
        let u = (report.median.unwrap() + report.positive_deviation.unwrap() - b) / (1.0 - b);
        let target = target.unwrap();
        let boost = if channel == 1 {
            options.oiii_boost
        } else {
            options.sii_boost
        };
        let m = u * (1.0 - target) / (u - 2.0 * u * target + target) / boost;
        if !m.is_finite()
            || !(0.0..1.0).contains(&m)
            || m == 0.0
            || !(0.0..1.0).contains(&u)
            || !(0.0..1.0).contains(&target)
        {
            report.status = "skipped".into();
            report.reason = Some("unusable_curve".into());
            continue;
        }
        report.midtone = Some(m);
        report.status = if (m - 0.5).abs() < 1e-12 {
            "identity"
        } else {
            "applied"
        }
        .into();
    }
    image.data.par_chunks_mut(stride).for_each(|pixel| {
        if pixel[..count].iter().any(|v| !v.is_finite()) {
            pixel.fill(f32::NAN);
            return;
        }
        for (channel, report) in reports.iter().enumerate().skip(1) {
            if let Some(m) = report.midtone {
                let b = report.background.unwrap();
                let value = f64::from(pixel[channel]);
                if value > b {
                    pixel[channel] = (b
                        + (1.0 - b) * midtones_transfer_function(m, (value - b) / (1.0 - b)))
                        as f32;
                }
            }
        }
    });
    Ok(NarrowbandBalanceReport { channels: reports })
}

/// Screen RGB stars onto a starless display nebula, in place. All images are
/// aligned interleaved RGB in [0, 1]. `original` and `starless` must have had
/// the same color-preserving display transfer applied. Derive a screen stars
/// layer by unscreening the starless RGB from the original RGB. This avoids
/// adding a second nebula or applying a linear stars residual as display data.
/// Negative extraction noise is suppressed only in this display result.
/// Missing coverage in any input masks the whole output pixel.
pub fn screen_rgb_stars(
    nebula: &mut LinearImage,
    original: &LinearImage,
    starless: &LinearImage,
) -> Result<()> {
    for image in [&*nebula, original, starless] {
        valid_image(image, &[3])?;
        if (image.width, image.height) != (nebula.width, nebula.height) {
            return Err(invalid("Stars and nebula must share one pixel grid"));
        }
    }
    nebula
        .data
        .par_chunks_mut(3)
        .zip(original.data.par_chunks(3))
        .zip(starless.data.par_chunks(3))
        .for_each(|((out, original), starless)| {
            if out
                .iter()
                .chain(original)
                .chain(starless)
                .any(|v| !v.is_finite())
            {
                out.fill(f32::NAN);
                return;
            }
            for channel in 0..3 {
                let base = starless[channel];
                let stars = if base >= 1.0 {
                    0.0
                } else {
                    ((original[channel] - base) / (1.0 - base)).clamp(0.0, 1.0)
                };
                out[channel] = 1.0 - (1.0 - out[channel]) * (1.0 - stars);
            }
        });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rgb(values: Vec<f32>) -> LinearImage {
        LinearImage::new(values.len() / 3, 1, 3, values).unwrap()
    }
    #[test]
    fn stars_reconstruct_original_and_mask_missing_coverage() {
        let original = rgb(vec![0.3, 0.6, 0.9, 0.4, 0.5, f32::NAN]);
        let starless = rgb(vec![0.1, 0.2, 0.3, 0.1, 0.2, 0.3]);
        let mut nebula = starless.clone();
        screen_rgb_stars(&mut nebula, &original, &starless).unwrap();
        for i in 0..3 {
            assert!((nebula.data[i] - original.data[i]).abs() < 1e-6);
        }
        assert!(nebula.data[3..].iter().all(|v| v.is_nan()));
    }
    #[test]
    fn balance_protects_shadows_and_ha_and_skips_flat_channels() {
        let mut data = Vec::new();
        for i in 0..100 {
            let ha = if i < 60 {
                0.1
            } else {
                0.1 + (i - 59) as f32 / 60.0
            };
            data.extend([ha, ha / 2.0, 0.1]);
        }
        let mut image = LinearImage::new(100, 1, 3, data.clone()).unwrap();
        let report = balance_narrowband(&mut image, &NarrowbandBalanceOptions::default()).unwrap();
        assert_eq!(report.channels[1].status, "applied");
        assert_eq!(
            report.channels[2].reason.as_deref(),
            Some("too_little_signal")
        );
        assert!(image.data[3 * 99 + 1] > data[3 * 99 + 1]);
        for i in 0..100 {
            assert_eq!(image.data[i * 3], data[i * 3]);
            assert_eq!(image.data[i * 3 + 2], 0.1);
        }
        assert_eq!(&image.data[..180], &data[..180]);
    }
    #[test]
    fn invalid_samples_and_options_fail_without_mutation() {
        let mut image = rgb(vec![0.1, 0.2, 0.3]);
        let before = image.data.clone();
        assert!(
            balance_narrowband(
                &mut image,
                &NarrowbandBalanceOptions {
                    oiii_boost: f64::NAN,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert_eq!(image.data, before);
        assert!(
            screen_rgb_stars(&mut image, &rgb(vec![2.0, 0.1, 0.2]), &rgb(before.clone())).is_err()
        );
        assert_eq!(image.data, before);
    }

    #[test]
    fn stars_keep_saturated_signal_above_a_nearly_white_base() {
        let mut nebula = rgb(vec![0.2; 3]);
        screen_rgb_stars(
            &mut nebula,
            &rgb(vec![1.0; 3]),
            &rgb(vec![1.0 - f32::EPSILON; 3]),
        )
        .unwrap();
        assert_eq!(nebula.data, vec![1.0; 3]);
    }

    #[test]
    fn hoo_ignores_the_unused_third_plane_and_skips_noise() {
        let mut image = LinearImage::new(
            100,
            1,
            3,
            (0..100)
                .flat_map(|i| {
                    [
                        if i < 60 {
                            0.1
                        } else {
                            0.1 + (i - 59) as f32 / 60.0
                        },
                        if i % 2 == 0 { 0.09 } else { 0.11 },
                        f32::NAN,
                    ]
                })
                .collect(),
        )
        .unwrap();
        let before = image.data.clone();
        let report = balance_narrowband(
            &mut image,
            &NarrowbandBalanceOptions {
                has_sii: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(report.channels.len(), 2);
        assert_eq!(report.channels[0].status, "reference");
        assert_eq!(
            report.channels[1].reason.as_deref(),
            Some("too_little_signal")
        );
        for (out, input) in image.data.chunks(3).zip(before.chunks(3)) {
            assert_eq!(&out[..2], &input[..2]);
            assert!(out[2].is_nan());
        }
    }
}
