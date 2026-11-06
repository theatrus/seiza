//! Two-dimensional FFTs on small tiles, built on `rustfft`.

use std::sync::Arc;

use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

/// The smallest length at or above `length` whose only prime factors are
/// 2, 3 and 5, which `rustfft` transforms fastest.
pub(crate) fn smooth_length(length: usize) -> usize {
    let mut candidate = length.max(1);
    loop {
        let mut rest = candidate;
        for factor in [2, 3, 5] {
            while rest.is_multiple_of(factor) {
                rest /= factor;
            }
        }
        if rest == 1 {
            return candidate;
        }
        candidate += 1;
    }
}

/// Forward and inverse transforms of a `width × height` complex tile, row
/// by row. The inverse is not normalised: it multiplies by `width · height`.
pub(crate) struct Fft2 {
    pub(crate) width: usize,
    pub(crate) height: usize,
    rows: Arc<dyn Fft<f32>>,
    columns: Arc<dyn Fft<f32>>,
    rows_inverse: Arc<dyn Fft<f32>>,
    columns_inverse: Arc<dyn Fft<f32>>,
}

impl Fft2 {
    pub(crate) fn new(width: usize, height: usize) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        Self {
            width,
            height,
            rows: planner.plan_fft_forward(width),
            columns: planner.plan_fft_forward(height),
            rows_inverse: planner.plan_fft_inverse(width),
            columns_inverse: planner.plan_fft_inverse(height),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.width * self.height
    }

    /// Transform `data` in place; `scratch` holds one tile.
    pub(crate) fn forward(&self, data: &mut [Complex32], scratch: &mut [Complex32]) {
        self.transform(data, scratch, &self.rows, &self.columns);
    }

    /// [`Self::forward`] for a tile whose only non-zero samples lie in
    /// `rows`: the other rows' transforms are zero, so only these are run.
    pub(crate) fn forward_sparse_rows(
        &self,
        data: &mut [Complex32],
        scratch: &mut [Complex32],
        rows: &[usize],
    ) {
        debug_assert_eq!(data.len(), self.len());
        for &row in rows {
            self.rows
                .process(&mut data[row * self.width..(row + 1) * self.width]);
        }
        transpose(data, scratch, self.width, self.height);
        self.columns.process(scratch);
        transpose(scratch, data, self.height, self.width);
    }

    /// Transform `data` back in place, unnormalised.
    pub(crate) fn inverse(&self, data: &mut [Complex32], scratch: &mut [Complex32]) {
        self.transform(data, scratch, &self.rows_inverse, &self.columns_inverse);
    }

    fn transform(
        &self,
        data: &mut [Complex32],
        scratch: &mut [Complex32],
        rows: &Arc<dyn Fft<f32>>,
        columns: &Arc<dyn Fft<f32>>,
    ) {
        debug_assert_eq!(data.len(), self.len());
        debug_assert_eq!(scratch.len(), self.len());
        rows.process(data);
        transpose(data, scratch, self.width, self.height);
        columns.process(scratch);
        transpose(scratch, data, self.height, self.width);
    }
}

/// `output` is `input` (`width × height`, row by row) transposed.
fn transpose(input: &[Complex32], output: &mut [Complex32], width: usize, height: usize) {
    const BLOCK: usize = 32;
    for row_block in (0..height).step_by(BLOCK) {
        for column_block in (0..width).step_by(BLOCK) {
            for y in row_block..(row_block + BLOCK).min(height) {
                for x in column_block..(column_block + BLOCK).min(width) {
                    output[x * height + y] = input[y * width + x];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smooth_lengths_have_small_factors() {
        assert_eq!(smooth_length(1), 1);
        assert_eq!(smooth_length(7), 8);
        assert_eq!(smooth_length(97), 100);
        assert_eq!(smooth_length(1057), 1080);
    }

    #[test]
    fn forward_then_inverse_round_trips() {
        let (width, height) = (12, 10);
        let fft = Fft2::new(width, height);
        let original: Vec<Complex32> = (0..width * height)
            .map(|index| Complex32::new((index % 7) as f32 - 3.0, 0.0))
            .collect();
        let mut data = original.clone();
        let mut scratch = vec![Complex32::default(); fft.len()];
        fft.forward(&mut data, &mut scratch);
        // The DC term is the sum.
        let sum: f32 = original.iter().map(|value| value.re).sum();
        assert!((data[0].re - sum).abs() < 1e-3);
        fft.inverse(&mut data, &mut scratch);
        let scale = (width * height) as f32;
        for (got, want) in data.iter().zip(&original) {
            assert!((got.re / scale - want.re).abs() < 1e-4);
            assert!((got.im / scale).abs() < 1e-4);
        }
    }
}
