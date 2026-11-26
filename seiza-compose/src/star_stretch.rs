//! Independent display stretching of stars extracted from linear RGB.
use crate::{Error, LinearImage, Result, screen_rgb_stars_reduced};
use rayon::prelude::*;

fn invalid(message: &str) -> Error {
    Error::Color(message.into())
}
fn rgb(image: &LinearImage) -> Result<()> {
    if image.channels != 3
        || image.width == 0
        || image.height == 0
        || image
            .width
            .checked_mul(image.height)
            .and_then(|n| n.checked_mul(3))
            != Some(image.data.len())
    {
        return Err(invalid(
            "Star stretching requires a nonempty interleaved RGB image",
        ));
    }
    Ok(())
}

/// Extract a linear stars-only residual and apply the SetiAstro-style rational
/// tone `3^amount*x / (1 + (3^amount-1)*x)`. Both inputs must be aligned LINEAR
/// RGB, before any display stretch, with the same physical units and calibration.
/// `starless` is the linear star-separation result. This does not run a tool.
///
/// Amount is finite in [0,8]; 5 is the script's tone default. Lower amounts
/// make stars fainter. Zero applies no nonlinear lift, not star removal.
/// One shared scale, max(1, peak(original)-min(0,min(original))), matches the
/// linear external-tool normalization. The background offset cancels in the
/// residual; never normalize each channel or the stars-only image separately.
/// Negative residuals and samples above the source range are clipped only in
/// this new display layer. Nonfinite coverage masks the entire RGB pixel.
///
/// This independently implements the published tone formula, not PixInsight's
/// optional ColorSaturation or SCNR processes. Reference: SetiAstro Star Stretch
/// v2.6, https://www.setiastro.com/pjsr-scripts . Linear inputs are unchanged.
pub fn stretch_rgb_stars(
    original: &LinearImage,
    starless: &LinearImage,
    amount: f64,
) -> Result<LinearImage> {
    rgb(original)?;
    rgb(starless)?;
    if (original.width, original.height) != (starless.width, starless.height) {
        return Err(invalid(
            "Original and starless RGB must share one pixel grid",
        ));
    }
    if !amount.is_finite() || !(0.0..=8.0).contains(&amount) {
        return Err(invalid(
            "RGB star stretch amount must be between zero and eight",
        ));
    }
    let (mut floor, mut peak) = (f32::INFINITY, f32::NEG_INFINITY);
    for pixel in original.data.as_chunks::<3>().0 {
        if pixel.iter().all(|v| v.is_finite()) {
            for value in pixel {
                floor = floor.min(*value);
                peak = peak.max(*value);
            }
        }
    }
    if !peak.is_finite() {
        return Err(invalid("Original RGB has no finite pixels"));
    }
    let scale = (f64::from(peak) - f64::from(floor.min(0.0))).max(1.0);
    let gain = 3.0_f64.powf(amount);
    let mut result = original.clone();
    result
        .data
        .par_chunks_mut(3)
        .zip(original.data.par_chunks(3))
        .zip(starless.data.par_chunks(3))
        .for_each(|((out, input), base)| {
            if input.iter().chain(base).any(|v| !v.is_finite()) {
                out.fill(f32::NAN);
                return;
            }
            for c in 0..3 {
                let x = ((f64::from(input[c]) - f64::from(base[c])) / scale).clamp(0.0, 1.0);
                out[c] = (gain * x / (1.0 + (gain - 1.0) * x)) as f32;
            }
        });
    Ok(result)
}

/// Screen an already stretched stars-only RGB layer over a display nebula.
/// All samples must be in [0,1], on one grid. Optional reduction in [0,1]
/// applies linked one-pixel cross erosion to the stars before screening;
/// zero preserves their shape. Missing coverage masks the whole output pixel.
/// Validation errors leave the nebula unchanged. This does not stretch again.
pub fn screen_rgb_star_layer(
    nebula: &mut LinearImage,
    stars: &LinearImage,
    reduction: f64,
) -> Result<()> {
    rgb(stars)?;
    let zero = LinearImage::new(stars.width, stars.height, 3, vec![0.0; stars.data.len()])?;
    screen_rgb_stars_reduced(nebula, stars, &zero, reduction)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn image(data: Vec<f32>) -> LinearImage {
        LinearImage::new(data.len() / 3, 1, 3, data).unwrap()
    }
    #[test]
    fn tone_matches_analytic_values_and_smaller_amount_subdues_stars() {
        let original = image(vec![0.001, 0.01, 1.0]);
        let base = image(vec![0.0; 3]);
        let lower = stretch_rgb_stars(&original, &base, 3.0).unwrap();
        let higher = stretch_rgb_stars(&original, &base, 5.0).unwrap();
        for (actual, expected) in lower.data.iter().zip([0.027 / 1.026, 0.27 / 1.26, 1.0]) {
            assert!((f64::from(*actual) - expected).abs() < 1e-7);
        }
        assert!(lower.data[0] < higher.data[0]);
        assert_eq!(lower.data[2], 1.0);
        assert_eq!(
            stretch_rgb_stars(&original, &base, 0.0).unwrap().data,
            original.data
        );
    }
    #[test]
    fn physical_units_and_background_cancel_without_per_channel_scaling() {
        let original = image(vec![-0.1, 0.2, 0.9, 0.4, 0.5, 0.6]);
        let base = image(vec![-0.11, 0.18, 0.86, 0.4, 0.55, 0.6]);
        let unit = stretch_rgb_stars(&original, &base, 3.0).unwrap();
        let physical = stretch_rgb_stars(
            &image(original.data.iter().map(|v| v * 1000.0).collect()),
            &image(base.data.iter().map(|v| v * 1000.0).collect()),
            3.0,
        )
        .unwrap();
        for (a, b) in unit.data.iter().zip(physical.data) {
            assert!((a - b).abs() < 1e-6);
        }
        assert_eq!(&unit.data[3..], &[0.0, 0.0, 0.0]);
    }
    #[test]
    fn invalid_options_and_coverage() {
        let original = image(vec![1.0, 0.5, 0.2, 0.1, f32::NAN, 0.2]);
        let base = image(vec![0.0; 6]);
        for amount in [-1.0, 8.01, f64::NAN, f64::INFINITY] {
            assert!(stretch_rgb_stars(&original, &base, amount).is_err());
        }
        let stars = stretch_rgb_stars(&original, &base, 3.0).unwrap();
        assert!(stars.data[3..].iter().all(|v| v.is_nan()));
        let mut nebula = image(vec![0.2; 6]);
        screen_rgb_star_layer(&mut nebula, &stars, 0.4).unwrap();
        assert!(nebula.data[3..].iter().all(|v| v.is_nan()));
        let before = image(vec![0.2; 6]);
        let mut nebula = before.clone();
        assert!(screen_rgb_star_layer(&mut nebula, &stars, 2.0).is_err());
        assert_eq!(nebula.data, before.data);
    }
}
