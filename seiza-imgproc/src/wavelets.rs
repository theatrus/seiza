//! Multi-scale structure removal for astronomical images.
//!
//! Two decompositions are provided:
//!
//! - [`StructureRemover::remove_structures_filtered`]: Gaussian pyramid for
//!   the first three layers, then edge-aware domain transform filtering —
//!   the pipeline PSF Guard historically ran through OpenCV
//!   (`GaussianBlur` + `ximgproc::dtFilter`), reproduced in f32 like the
//!   original `CV_32F` path.
//! - [`StructureRemover::remove_structures_atrous`]: the à trous B3-spline
//!   wavelet transform, matching the HocusFocus reference implementation.

use crate::blur::gaussian_blur_f32;
use crate::border::BorderMode;
use crate::dtfilter::dt_filter_nc;

pub struct StructureRemover {
    pub layers: usize,
}

impl StructureRemover {
    pub fn new(layers: usize) -> Self {
        Self { layers }
    }

    /// Remove large-scale structure; returns the residual (small structures
    /// plus noise). Layers 0-2 subtract Gaussian-smoothed versions with
    /// `sigma = 0.8 * 2^layer`; deeper layers subtract a self-guided domain
    /// transform result (`sigma_spatial = 10 * 2^layer`, `sigma_color =
    /// 0.1`, one iteration). Arithmetic is f32 end-to-end, like the OpenCV
    /// `CV_32F` path this reproduces.
    pub fn remove_structures_filtered(
        &self,
        data: &[f64],
        width: usize,
        height: usize,
    ) -> Vec<f64> {
        assert_eq!(data.len(), width * height);
        let residual = self.remove_structures_filtered_f32(
            &data.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            width,
            height,
        );
        residual.into_iter().map(|v| v as f64).collect()
    }

    /// [`Self::remove_structures_filtered`] without the f64 boundary: takes
    /// and returns f32, which is the arithmetic the pipeline runs in anyway.
    /// Callers whose source data is exactly representable in f32 (integer
    /// camera data) get bit-identical results to the f64 entry point while
    /// skipping two full-image conversions.
    pub fn remove_structures_filtered_f32(
        &self,
        data: &[f32],
        width: usize,
        height: usize,
    ) -> Vec<f32> {
        assert_eq!(data.len(), width * height);
        let mut residual: Vec<f32> = data.to_vec();

        for layer in 0..self.layers {
            let scale = 1usize << layer;
            let kernel_size = 2 * scale + 1;

            let filtered = if layer < 3 {
                let sigma = scale as f64 * 0.8;
                gaussian_blur_f32(
                    &residual,
                    width,
                    height,
                    kernel_size,
                    sigma,
                    BorderMode::Reflect,
                )
            } else {
                dt_filter_nc(
                    &residual,
                    &residual,
                    width,
                    height,
                    10.0 * scale as f64,
                    0.1,
                    1,
                )
            };

            for (r, f) in residual.iter_mut().zip(filtered.iter()) {
                *r -= *f;
            }
        }

        residual
    }

    /// À trous B3-spline wavelet residual, matching HocusFocus exactly:
    /// per-layer separable [1/16, 1/4, 3/8, 1/4, 1/16] smoothing with
    /// spacing `2^layer` and edge weight renormalization, subtracted from
    /// the running residual.
    pub fn remove_structures_atrous(&self, data: &[f64], width: usize, height: usize) -> Vec<f64> {
        assert_eq!(data.len(), width * height);
        let mut residual = data.to_vec();

        for layer in 0..self.layers {
            let scale = (1usize << layer) as i32;
            let mut temp = vec![0.0; width * height];

            let coeffs = [0.0625, 0.25, 0.375, 0.25, 0.0625];
            let offsets = [-2i32, -1, 0, 1, 2];

            // Horizontal pass
            for y in 0..height {
                for x in 0..width {
                    let mut sum = 0.0;
                    let mut weight = 0.0;
                    for i in 0..5 {
                        let sx = x as i32 + offsets[i] * scale;
                        if sx >= 0 && sx < width as i32 {
                            sum += residual[y * width + sx as usize] * coeffs[i];
                            weight += coeffs[i];
                        }
                    }
                    temp[y * width + x] = if weight > 0.0 { sum / weight } else { 0.0 };
                }
            }

            // Vertical pass
            let mut smoothed = vec![0.0; width * height];
            for y in 0..height {
                for x in 0..width {
                    let mut sum = 0.0;
                    let mut weight = 0.0;
                    for i in 0..5 {
                        let sy = y as i32 + offsets[i] * scale;
                        if sy >= 0 && sy < height as i32 {
                            sum += temp[sy as usize * width + x] * coeffs[i];
                            weight += coeffs[i];
                        }
                    }
                    smoothed[y * width + x] = if weight > 0.0 { sum / weight } else { 0.0 };
                }
            }

            for i in 0..residual.len() {
                residual[i] -= smoothed[i];
            }
        }

        residual
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atrous_uniform_input_gives_near_zero_residual() {
        let data = vec![1.0; 25];
        let remover = StructureRemover::new(2);
        let residual = remover.remove_structures_atrous(&data, 5, 5);
        let sum: f64 = residual.iter().map(|x| x.abs()).sum();
        assert!(sum < 1.0);
    }

    #[test]
    fn filtered_removes_gradient_keeps_peak() {
        // Large-scale gradient with a compact bright peak: the residual
        // should retain far more of the peak than of the gradient.
        let w = 32;
        let h = 32;
        let mut data = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = (x + y) as f64 * 20.0;
            }
        }
        data[16 * w + 16] += 400.0;
        let remover = StructureRemover::new(3);
        let residual = remover.remove_structures_filtered(&data, w, h);
        let peak = residual[16 * w + 16];
        let bg = residual[8 * w + 8].abs();
        assert!(peak > 100.0, "peak should survive: {peak}");
        assert!(peak > 4.0 * bg, "peak {peak} vs background {bg}");
    }

    #[test]
    fn f32_entry_is_bit_identical_for_integer_data() {
        // Camera data (u16) is exactly representable in f32, so the f32
        // entry point must agree with the f64 wrapper bit for bit.
        let w = 41;
        let h = 23;
        let mut state = 7u64;
        let data_u16: Vec<u16> = (0..w * h)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state >> 48) as u16
            })
            .collect();
        let as_f64: Vec<f64> = data_u16.iter().map(|&v| v as f64).collect();
        let as_f32: Vec<f32> = data_u16.iter().map(|&v| v as f32).collect();
        let remover = StructureRemover::new(4);
        let via_f64 = remover.remove_structures_filtered(&as_f64, w, h);
        let via_f32 = remover.remove_structures_filtered_f32(&as_f32, w, h);
        for (a, b) in via_f64.iter().zip(via_f32.iter()) {
            assert_eq!(*a as f32, *b);
        }
    }

    #[test]
    fn both_paths_have_correct_length() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let remover = StructureRemover::new(4);
        assert_eq!(remover.remove_structures_filtered(&data, 10, 10).len(), 100);
        assert_eq!(remover.remove_structures_atrous(&data, 10, 10).len(), 100);
    }
}
