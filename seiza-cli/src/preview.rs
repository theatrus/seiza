use anyhow::Result;
use seiza_stacking::LinearImage;
use std::path::Path;

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
    let map = match transfer {
        PreviewTransfer::LinearLight => linear_preview_map(image)?,
        PreviewTransfer::DisplayReferred => DisplayMap {
            black: 0.0,
            white: 1.0,
            asinh_strength: None,
        },
    };
    if image.channels == 1 {
        let pixels = image.data.iter().map(|value| map.map(*value)).collect();
        image::GrayImage::from_raw(image.width as u32, image.height as u32, pixels)
            .ok_or_else(|| anyhow::anyhow!("preview dimension mismatch"))?
            .save(path)?;
    } else {
        let pixels = image.data.iter().map(|value| map.map(*value)).collect();
        image::RgbImage::from_raw(image.width as u32, image.height as u32, pixels)
            .ok_or_else(|| anyhow::anyhow!("preview dimension mismatch"))?
            .save(path)?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct DisplayMap {
    black: f32,
    white: f32,
    asinh_strength: Option<f32>,
}

impl DisplayMap {
    fn map(self, value: f32) -> u8 {
        if !value.is_finite() {
            return 0;
        }
        let linear = ((value - self.black) / (self.white - self.black)).max(0.0);
        let display = self.asinh_strength.map_or(linear, |strength| {
            (strength * linear).asinh() / strength.asinh()
        });
        (display.clamp(0.0, 1.0) * 255.0).round() as u8
    }
}

fn linear_preview_map(image: &LinearImage) -> Result<DisplayMap> {
    let stride = (image.data.len() / 200_000).max(1);
    let mut sample = image
        .data
        .iter()
        .step_by(stride)
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if sample.is_empty() {
        anyhow::bail!("image has no finite samples to preview");
    }
    sample.sort_unstable_by(f32::total_cmp);
    let black = sample[sample.len() / 100];
    let white = sample[sample.len() * 995 / 1000].max(black + f32::EPSILON);
    Ok(DisplayMap {
        black,
        white,
        asinh_strength: Some(10.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_display_map_clamps_and_rejects_non_finite_values() {
        let map = DisplayMap {
            black: 0.0,
            white: 1.0,
            asinh_strength: None,
        };
        assert_eq!(map.map(-1.0), 0);
        assert_eq!(map.map(0.5), 128);
        assert_eq!(map.map(2.0), 255);
        assert_eq!(map.map(f32::NAN), 0);
    }
}
