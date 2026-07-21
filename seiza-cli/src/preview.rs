use anyhow::{Context, Result};
use seiza_stacking::LinearImage;
use seiza_stretch::{ColorStrategy, ResolvedCurve, StretchConfig, StretchPlan};
use std::path::Path;

const MAXIMUM_SAMPLES: usize = 200_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PreviewTransfer {
    LinearLight,
    DisplayReferred,
}

pub(crate) fn write_preview(
    image: &LinearImage,
    path: &Path,
    transfer: PreviewTransfer,
) -> Result<()> {
    let plan = preview_plan(image, transfer)?;
    let pixels = plan
        .apply_u8(&image.data, image.channels)
        .context("could not apply preview stretch")?;
    write_display_image(path, image.width, image.height, image.channels, pixels)?;
    Ok(())
}

pub(crate) fn write_display_image(
    path: &Path,
    width: usize,
    height: usize,
    channels: usize,
    pixels: Vec<u8>,
) -> Result<()> {
    if channels == 1 {
        image::GrayImage::from_raw(width as u32, height as u32, pixels)
            .ok_or_else(|| anyhow::anyhow!("display image dimensions do not match"))?
            .save(path)?;
    } else {
        image::RgbImage::from_raw(width as u32, height as u32, pixels)
            .ok_or_else(|| anyhow::anyhow!("display image dimensions do not match"))?
            .save(path)?;
    }
    Ok(())
}

fn preview_plan(image: &LinearImage, transfer: PreviewTransfer) -> Result<StretchPlan> {
    match transfer {
        PreviewTransfer::LinearLight => {
            StretchConfig::percentile_asinh(0.01, 0.995, 10.0, MAXIMUM_SAMPLES)
                .resolve_for(&image.data, image.channels)
                .context("could not resolve preview stretch")
        }
        PreviewTransfer::DisplayReferred => StretchPlan::from_resolved(
            image.channels,
            ColorStrategy::Linked,
            vec![ResolvedCurve::Identity],
        )
        .context("could not resolve display-referred preview"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn previous_linear_preview(image: &LinearImage) -> Vec<u8> {
        let maximum_pixels = (MAXIMUM_SAMPLES / image.channels).max(1);
        let pixel_stride = image.pixel_count().div_ceil(maximum_pixels).max(1);
        let mut sample = image
            .data
            .chunks_exact(image.channels)
            .step_by(pixel_stride)
            .flat_map(|pixel| pixel.iter().copied())
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        sample.sort_unstable_by(f32::total_cmp);
        let black = sample[sample.len() / 100];
        let white = sample[sample.len() * 995 / 1000].max(black + f32::EPSILON);
        image
            .data
            .iter()
            .map(|value| {
                if !value.is_finite() {
                    return 0;
                }
                let linear = ((*value - black) / (white - black)).max(0.0);
                let display = (10.0 * linear).asinh() / 10.0_f32.asinh();
                (display.clamp(0.0, 1.0) * 255.0).round() as u8
            })
            .collect()
    }

    #[test]
    fn direct_display_map_clamps_and_rejects_non_finite_values() {
        let plan =
            StretchPlan::from_resolved(1, ColorStrategy::Linked, vec![ResolvedCurve::Identity])
                .unwrap();
        assert_eq!(
            plan.apply_u8(&[-1.0, 0.5, 2.0, f32::NAN], 1).unwrap(),
            [0, 128, 255, 0]
        );
    }

    #[test]
    fn rgb_preview_sampling_cannot_alias_one_color_plane() {
        let data = [0.0, 0.5, 1.0].repeat(448 * 448);
        let image = LinearImage::new(448, 448, 3, data).unwrap();
        let plan = preview_plan(&image, PreviewTransfer::LinearLight).unwrap();
        assert_eq!(
            plan.curves(),
            &[ResolvedCurve::Asinh {
                black: 0.0,
                white: 1.0,
                strength: 10.0,
            }]
        );
    }

    #[test]
    fn linear_preview_is_byte_identical_to_the_previous_curve() {
        let data = (0..131_072)
            .flat_map(|index| {
                let value = ((index * 7_919) % 65_521) as f32 / 65_520.0;
                [value * 0.7, value, f32::NAN]
            })
            .collect::<Vec<_>>();
        let image = LinearImage::new(512, 256, 3, data).unwrap();
        let expected = previous_linear_preview(&image);
        let actual = preview_plan(&image, PreviewTransfer::LinearLight)
            .unwrap()
            .apply_u8(&image.data, image.channels)
            .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn flat_linear_preview_keeps_the_previous_black_output() {
        let image = LinearImage::new(4, 4, 1, vec![1_000.0; 16]).unwrap();
        let expected = previous_linear_preview(&image);
        let actual = preview_plan(&image, PreviewTransfer::LinearLight)
            .unwrap()
            .apply_u8(&image.data, image.channels)
            .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(actual, vec![0; 16]);
    }
}
