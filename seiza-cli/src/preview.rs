use anyhow::{Context, Result};
use seiza_stacking::LinearImage;
use seiza_stretch::{ColorStrategy, ResolvedCurve, StretchAnalysis, StretchConfig, StretchPlan};
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
    if image.channels == 1 {
        image::GrayImage::from_raw(image.width as u32, image.height as u32, pixels)
            .ok_or_else(|| anyhow::anyhow!("preview dimension mismatch"))?
            .save(path)?;
    } else {
        image::RgbImage::from_raw(image.width as u32, image.height as u32, pixels)
            .ok_or_else(|| anyhow::anyhow!("preview dimension mismatch"))?
            .save(path)?;
    }
    Ok(())
}

fn preview_plan(image: &LinearImage, transfer: PreviewTransfer) -> Result<StretchPlan> {
    match transfer {
        PreviewTransfer::LinearLight => {
            let analysis = StretchAnalysis::analyze(&image.data, image.channels, MAXIMUM_SAMPLES)
                .context("could not analyze image for preview")?;
            StretchConfig::percentile_asinh(0.01, 0.995, 10.0, MAXIMUM_SAMPLES)
                .resolve(&analysis)
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
}
