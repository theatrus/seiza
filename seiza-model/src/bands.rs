use crate::error::ModelError;
use rayon::prelude::*;

/// How much of the noise a model finds is taken out. The removed noise is
/// split by a Gaussian of `scale_px` into the low band, the mottle, and the
/// high band, the grain left over, and `high` and `low` of each are taken out.
/// Ones take out all of it. Keeping a share `k` of white noise leaves an image
/// as noisy as a stack `1/k²` as deep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoiseBands {
    /// The share of the pixel-scale grain taken out, from 0 to 1.
    pub high: f32,
    /// The share of the wider mottle taken out, from 0 to 1.
    pub low: f32,
    /// The Gaussian scale in pixels dividing the grain from the mottle.
    pub scale_px: f32,
}

impl Default for NoiseBands {
    fn default() -> Self {
        Self {
            high: 1.0,
            low: 1.0,
            scale_px: 5.0,
        }
    }
}

impl NoiseBands {
    /// Refuse amounts outside 0 to 1 and a scale that is not positive.
    pub fn validate(&self) -> Result<(), ModelError> {
        let amount = 0.0..=1.0;
        if !(amount.contains(&self.high) && amount.contains(&self.low) && self.scale_px > 0.0) {
            return Err(ModelError::InvalidOption(format!(
                "amounts run from 0 to 1 and the scale is positive: high {}, low {}, scale {}",
                self.high, self.low, self.scale_px
            )));
        }
        Ok(())
    }

    /// One plane with these shares of the model's `residual`, `f(x) − x`,
    /// added: `x − a_high·r_high − a_low·r_low`.
    pub(crate) fn apply(
        &self,
        image: &[f32],
        residual: &[f32],
        width: usize,
        height: usize,
    ) -> Vec<f32> {
        if self.high == self.low {
            let amount = self.high;
            return image
                .par_iter()
                .zip(residual.par_iter())
                .map(|(value, removed)| value + amount * removed)
                .collect();
        }
        let blurred = gaussian_blur(residual, width, height, self.scale_px);
        let (high, wider) = (self.high, self.low - self.high);
        image
            .par_iter()
            .zip(residual.par_iter())
            .zip(blurred.par_iter())
            .map(|((value, removed), mottle)| value + high * removed + wider * mottle)
            .collect()
    }
}

/// A Gaussian blur of scale `sigma_px`, cut at three of it, the edge pixels
/// repeated outward. `plane` must be finite.
fn gaussian_blur(plane: &[f32], width: usize, height: usize, sigma_px: f32) -> Vec<f32> {
    let radius = (3.0 * sigma_px).ceil().max(1.0) as isize;
    let mut kernel: Vec<f32> = (-radius..=radius)
        .map(|offset| (-0.5 * (offset as f32 / sigma_px).powi(2)).exp())
        .collect();
    let sum: f32 = kernel.iter().sum();
    kernel.iter_mut().for_each(|weight| *weight /= sum);

    let mut rows = vec![0.0_f32; plane.len()];
    rows.par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, line)| {
            let source = &plane[y * width..(y + 1) * width];
            for (x, out) in line.iter_mut().enumerate() {
                let mut value = 0.0;
                for (tap, weight) in kernel.iter().enumerate() {
                    let at = (x as isize + tap as isize - radius).clamp(0, width as isize - 1);
                    value += weight * source[at as usize];
                }
                *out = value;
            }
        });
    let mut blurred = vec![0.0_f32; plane.len()];
    blurred
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, line)| {
            for (tap, weight) in kernel.iter().enumerate() {
                let at =
                    (y as isize + tap as isize - radius).clamp(0, height as isize - 1) as usize;
                let source = &rows[at * width..(at + 1) * width];
                for (out, value) in line.iter_mut().zip(source) {
                    *out += weight * value;
                }
            }
        });
    blurred
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_amounts_add_the_residual_and_none_leave_the_image() {
        let image = [10.0, 20.0, f32::NAN, 40.0];
        let residual = [-1.0, -2.0, -3.0, -4.0];
        let whole = NoiseBands::default().apply(&image, &residual, 2, 2);
        assert_eq!(whole[0], 9.0);
        assert!(whole[2].is_nan(), "a pixel with no data stays without");
        let none = NoiseBands {
            high: 0.0,
            low: 0.0,
            scale_px: 5.0,
        };
        assert_eq!(none.apply(&image, &residual, 2, 2)[3], 40.0);
    }

    #[test]
    fn a_flat_residual_is_all_low_band() {
        let image = vec![100.0_f32; 64 * 64];
        let residual = vec![-8.0_f32; 64 * 64];
        let low_only = NoiseBands {
            high: 0.0,
            low: 0.5,
            scale_px: 3.0,
        };
        let kept = low_only.apply(&image, &residual, 64, 64);
        assert!(
            kept.iter().all(|value| (value - 96.0).abs() < 1e-3),
            "{}",
            kept[0]
        );
    }

    #[test]
    fn a_blur_keeps_the_sum_of_a_point_away_from_the_edges() {
        let mut plane = vec![0.0_f32; 65 * 65];
        plane[32 * 65 + 32] = 1.0;
        let blurred = gaussian_blur(&plane, 65, 65, 2.0);
        let sum: f32 = blurred.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "{sum}");
        assert!(
            blurred[32 * 65 + 32] > blurred[32 * 65 + 34],
            "peaked at the point"
        );
    }

    #[test]
    fn amounts_out_of_range_are_refused() {
        assert!(
            NoiseBands {
                high: 1.5,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            NoiseBands {
                scale_px: 0.0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(NoiseBands::default().validate().is_ok());
    }
}
