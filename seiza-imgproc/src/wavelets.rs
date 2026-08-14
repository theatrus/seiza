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

            subtract_in_place(&mut residual, &filtered);
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

    /// Small-scale structure map via the à trous B3-spline smoothing chain,
    /// matching HocusFocus `StarDetectorVersion` 2 (`AtrousWaveletFast`).
    ///
    /// Each layer smooths the PREVIOUS smoothed layer with the sparse
    /// separable 5-tap kernel `[1/16, 1/4, 3/8, 1/4, 1/16]` at tap spacing
    /// `2^layer` (taps at ±2^layer and ±2^(layer+1)), using OpenCV
    /// `BORDER_REFLECT` semantics (edge pixel repeats: index −1 → 0, −2 → 1,
    /// with multi-bounce when a tap offset exceeds the image). The final
    /// smoothed chain is subtracted from the source once and clamped at
    /// zero, leaving stars while removing nebulosity and gradients.
    ///
    /// This differs from [`Self::remove_structures_atrous`], which subtracts
    /// each layer's smoothing from a running residual — the composition
    /// `(I−S_n)…(I−S_0)` rather than the reference `I − S_n…S_0` — and
    /// renormalizes edge weights instead of reflecting.
    pub fn structure_map_atrous_chain(
        &self,
        data: &[f32],
        width: usize,
        height: usize,
    ) -> Vec<f32> {
        assert_eq!(data.len(), width * height);
        let mut smoothed = data.to_vec();
        let mut temp = vec![0.0f32; width * height];

        for layer in 0..self.layers {
            let scale = 1usize << layer;
            atrous_smooth_rows(&smoothed, &mut temp, width, scale);
            atrous_smooth_columns(&temp, &mut smoothed, width, height, scale);
        }

        // One fused subtract + clamp: map = max(src − smoothed chain, 0).
        let mut map = smoothed;
        #[cfg(feature = "parallel")]
        if map.len() >= crate::blur::PAR_MIN_PIXELS {
            use rayon::prelude::*;
            map.par_chunks_mut(4096)
                .zip(data.par_chunks(4096))
                .for_each(|(m_chunk, d_chunk)| {
                    for (m, &d) in m_chunk.iter_mut().zip(d_chunk.iter()) {
                        *m = (d - *m).max(0.0);
                    }
                });
            return map;
        }
        for (m, &d) in map.iter_mut().zip(data.iter()) {
            *m = (d - *m).max(0.0);
        }
        map
    }
}

/// OpenCV `BORDER_REFLECT` (`gfedcb|abcdefgh|gfedcba`): −1 → 0, −2 → 1;
/// `n` → `n−1`. Bounces until in range, so tap offsets beyond the image
/// stay defined.
#[inline]
fn reflect_index(mut index: isize, len: usize) -> usize {
    let len = len as isize;
    loop {
        if index < 0 {
            index = -index - 1;
        } else if index >= len {
            index = 2 * len - 1 - index;
        } else {
            return index as usize;
        }
    }
}

const B3_TAPS: [f32; 5] = [0.0625, 0.25, 0.375, 0.25, 0.0625];

/// Horizontal sparse 5-tap B3 smoothing at tap spacing `scale`, reflect
/// borders. Rows are independent; split across threads under `parallel`.
/// Per-pixel tap order is fixed (−2s, −s, 0, +s, +2s), so results do not
/// depend on the thread count.
fn atrous_smooth_rows(src: &[f32], dst: &mut [f32], width: usize, scale: usize) {
    let row_op = |(src_row, dst_row): (&[f32], &mut [f32])| {
        let s = scale as isize;
        let w = width as isize;
        // Interior pixels need no reflection: 2·scale ≤ x < width − 2·scale.
        let interior_start = (2 * scale).min(width);
        let interior_end = width.saturating_sub(2 * scale).max(interior_start);
        for (x, out) in dst_row.iter_mut().enumerate().take(interior_start) {
            *out = tap5_reflect(src_row, x as isize, s, w);
        }
        for x in interior_start..interior_end {
            dst_row[x] = B3_TAPS[0] * src_row[x - 2 * scale]
                + B3_TAPS[1] * src_row[x - scale]
                + B3_TAPS[2] * src_row[x]
                + B3_TAPS[3] * src_row[x + scale]
                + B3_TAPS[4] * src_row[x + 2 * scale];
        }
        for (x, out) in dst_row.iter_mut().enumerate().skip(interior_end) {
            *out = tap5_reflect(src_row, x as isize, s, w);
        }
    };

    #[cfg(feature = "parallel")]
    if src.len() >= crate::blur::PAR_MIN_PIXELS {
        use rayon::prelude::*;
        src.par_chunks(width)
            .zip(dst.par_chunks_mut(width))
            .for_each(row_op);
        return;
    }
    src.chunks(width)
        .zip(dst.chunks_mut(width))
        .for_each(row_op);
}

#[inline]
fn tap5_reflect(row: &[f32], x: isize, scale: isize, len: isize) -> f32 {
    B3_TAPS[0] * row[reflect_index(x - 2 * scale, len as usize)]
        + B3_TAPS[1] * row[reflect_index(x - scale, len as usize)]
        + B3_TAPS[2] * row[reflect_index(x, len as usize)]
        + B3_TAPS[3] * row[reflect_index(x + scale, len as usize)]
        + B3_TAPS[4] * row[reflect_index(x + 2 * scale, len as usize)]
}

/// Vertical sparse 5-tap B3 smoothing at tap spacing `scale`, reflect
/// borders. Each output row reads five source rows and writes one, so the
/// whole-row arithmetic vectorizes and rows split across threads with a
/// fixed tap order.
fn atrous_smooth_columns(src: &[f32], dst: &mut [f32], width: usize, height: usize, scale: usize) {
    let row_op = |(y, dst_row): (usize, &mut [f32])| {
        let source_row = |offset: isize| -> &[f32] {
            let sy = reflect_index(y as isize + offset, height);
            &src[sy * width..(sy + 1) * width]
        };
        let s = scale as isize;
        let (r0, r1, r2, r3, r4) = (
            source_row(-2 * s),
            source_row(-s),
            source_row(0),
            source_row(s),
            source_row(2 * s),
        );
        for x in 0..width {
            dst_row[x] = B3_TAPS[0] * r0[x]
                + B3_TAPS[1] * r1[x]
                + B3_TAPS[2] * r2[x]
                + B3_TAPS[3] * r3[x]
                + B3_TAPS[4] * r4[x];
        }
    };

    #[cfg(feature = "parallel")]
    if src.len() >= crate::blur::PAR_MIN_PIXELS {
        use rayon::prelude::*;
        dst.par_chunks_mut(width).enumerate().for_each(row_op);
        return;
    }
    dst.chunks_mut(width).enumerate().for_each(row_op);
}

/// Elementwise `residual -= filtered`; row-split under the `parallel`
/// feature (element order within each subtraction is unchanged, and the
/// operations are independent, so results are bit-identical).
fn subtract_in_place(residual: &mut [f32], filtered: &[f32]) {
    #[cfg(feature = "parallel")]
    if residual.len() >= crate::blur::PAR_MIN_PIXELS {
        use rayon::prelude::*;
        residual
            .par_chunks_mut(4096)
            .zip(filtered.par_chunks(4096))
            .for_each(|(r_chunk, f_chunk)| {
                for (r, f) in r_chunk.iter_mut().zip(f_chunk.iter()) {
                    *r -= *f;
                }
            });
        return;
    }
    for (r, f) in residual.iter_mut().zip(filtered.iter()) {
        *r -= *f;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive oracle for the chain: same math as HocusFocus's legacy dense
    /// SepFilter2D path — per layer, convolve the previous smoothed layer
    /// with the zero-padded B3 kernel of width 2^(layer+2)+1 (only five taps
    /// non-zero) under BORDER_REFLECT — then subtract once and clamp.
    fn structure_map_dense_oracle(
        data: &[f64],
        width: usize,
        height: usize,
        layers: usize,
    ) -> Vec<f64> {
        let reflect = |mut i: isize, n: usize| -> usize {
            let n = n as isize;
            loop {
                if i < 0 {
                    i = -i - 1;
                } else if i >= n {
                    i = 2 * n - 1 - i;
                } else {
                    return i as usize;
                }
            }
        };
        let taps = [0.0625f64, 0.25, 0.375, 0.25, 0.0625];
        let mut smoothed = data.to_vec();
        for layer in 0..layers {
            let scale = (1isize << layer) as isize;
            let mut horizontal = vec![0.0; width * height];
            for y in 0..height {
                for x in 0..width {
                    let mut sum = 0.0;
                    for (t, tap) in taps.iter().enumerate() {
                        let offset = (t as isize - 2) * scale;
                        sum += tap * smoothed[y * width + reflect(x as isize + offset, width)];
                    }
                    horizontal[y * width + x] = sum;
                }
            }
            for y in 0..height {
                for x in 0..width {
                    let mut sum = 0.0;
                    for (t, tap) in taps.iter().enumerate() {
                        let offset = (t as isize - 2) * scale;
                        sum += tap * horizontal[reflect(y as isize + offset, height) * width + x];
                    }
                    smoothed[y * width + x] = sum;
                }
            }
        }
        data.iter()
            .zip(smoothed.iter())
            .map(|(&d, &s)| (d - s).max(0.0))
            .collect()
    }

    #[test]
    fn atrous_chain_matches_dense_oracle() {
        // Deterministic pseudo-random image, sized so multi-bounce
        // reflection is exercised at 3 layers (max tap offset 8 vs width 13).
        let (w, h) = (13, 11);
        let mut state = 0x2545F4914F6CDD1Du64;
        let data: Vec<f64> = (0..w * h)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % 60000) as f64
            })
            .collect();
        let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();

        for layers in [1, 2, 3] {
            let oracle = structure_map_dense_oracle(&data, w, h, layers);
            let fast = StructureRemover::new(layers).structure_map_atrous_chain(&data_f32, w, h);
            for (i, (&o, &f)) in oracle.iter().zip(fast.iter()).enumerate() {
                assert!(
                    (o - f as f64).abs() <= 0.05,
                    "layers={layers} pixel {i}: oracle {o} vs fast {f}"
                );
            }
        }
    }

    #[test]
    fn atrous_chain_removes_gradient_keeps_star() {
        // A strong linear gradient (nebulosity stand-in) with a compact
        // Gaussian star: the map should keep the star and drop the gradient.
        let (w, h) = (64, 64);
        let mut data = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = (x + y) as f32 * 100.0;
            }
        }
        let (cx, cy) = (32.0f32, 32.0f32);
        for y in 24..40 {
            for x in 24..40 {
                let d2 = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
                data[y * w + x] += 20000.0 * (-d2 / (2.0 * 1.5f32.powi(2))).exp();
            }
        }
        let map = StructureRemover::new(4).structure_map_atrous_chain(&data, w, h);
        let star = map[32 * w + 32];
        let background = map[8 * w + 40];
        assert!(star > 10000.0, "star peak should survive, got {star}");
        assert!(
            background < star / 100.0,
            "gradient should be removed, got {background} vs star {star}"
        );
    }

    #[test]
    fn atrous_chain_is_clamped_and_sized() {
        let data = vec![5.0f32; 6 * 4];
        let map = StructureRemover::new(2).structure_map_atrous_chain(&data, 6, 4);
        assert_eq!(map.len(), 24);
        assert!(map.iter().all(|&v| v >= 0.0), "clamped at zero");
        assert!(map.iter().all(|&v| v < 1e-3), "uniform input has no detail");
    }

    #[test]
    fn reflect_index_bounces_like_opencv_border_reflect() {
        assert_eq!(reflect_index(-1, 8), 0);
        assert_eq!(reflect_index(-2, 8), 1);
        assert_eq!(reflect_index(8, 8), 7);
        assert_eq!(reflect_index(9, 8), 6);
        // Multi-bounce: offsets beyond one full reflection.
        assert_eq!(reflect_index(-9, 8), 7);
        assert_eq!(reflect_index(17, 8), 1);
        assert_eq!(reflect_index(0, 1), 0);
        assert_eq!(reflect_index(-3, 1), 0);
    }

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
