//! Cutting star stamps out of an image, centred to a fraction of a pixel.

use super::candidates::clipped_stats;

/// Taps on each side of a Lanczos-3 interpolation.
pub(super) const LANCZOS_REACH: usize = 3;

/// The Lanczos-3 kernel.
fn lanczos3(x: f64) -> f64 {
    let x = x.abs();
    if x < 1e-12 {
        1.0
    } else if x >= 3.0 {
        0.0
    } else {
        let px = std::f64::consts::PI * x;
        3.0 * px.sin() * (px / 3.0).sin() / (px * px)
    }
}

/// The six Lanczos-3 weights that interpolate at `x0 + t` from samples at
/// `x0 − 2 ..= x0 + 3`, for `t` in `[0, 1)`, normalised to sum to one.
pub(super) fn lanczos_weights(t: f64) -> [f64; 6] {
    let mut weights = [0.0; 6];
    for (k, weight) in weights.iter_mut().enumerate() {
        *weight = lanczos3(t - (k as f64 - 2.0));
    }
    let total: f64 = weights.iter().sum();
    weights.iter_mut().for_each(|weight| *weight /= total);
    weights
}

/// One channel of an interleaved image.
#[derive(Clone, Copy)]
pub(super) struct Plane<'a> {
    pub(super) data: &'a [f32],
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) channels: usize,
    pub(super) channel: usize,
}

impl Plane<'_> {
    /// The sample at `(x, y)`, or NaN outside the image.
    pub(super) fn at(&self, x: isize, y: isize) -> f64 {
        if x < 0 || y < 0 || x >= self.width as isize || y >= self.height as isize {
            return f64::NAN;
        }
        f64::from(self.data[(y as usize * self.width + x as usize) * self.channels + self.channel])
    }

    /// Whether every sample within Chebyshev distance `reach` of `(x, y)`
    /// is inside the image and finite.
    pub(super) fn window_is_finite(&self, x: isize, y: isize, reach: isize) -> bool {
        if x - reach < 0
            || y - reach < 0
            || x + reach >= self.width as isize
            || y + reach >= self.height as isize
        {
            return false;
        }
        (y - reach..=y + reach)
            .all(|py| (x - reach..=x + reach).all(|px| self.at(px, py).is_finite()))
    }

    /// The clipped median and MAD noise of the square ring at Chebyshev
    /// distance `inner ..= outer` around `(x, y)`.
    pub(super) fn ring_background(
        &self,
        x: isize,
        y: isize,
        inner: isize,
        outer: isize,
    ) -> Option<(f64, f64)> {
        let mut samples = Vec::with_capacity((8 * outer * (outer - inner + 1)) as usize);
        for py in y - outer..=y + outer {
            for px in x - outer..=x + outer {
                if (px - x).abs().max((py - y).abs()) < inner {
                    continue;
                }
                let value = self.at(px, py);
                if value.is_finite() {
                    samples.push(value as f32);
                }
            }
        }
        clipped_stats(&mut samples).map(|(centre, sigma)| (f64::from(centre), f64::from(sigma)))
    }

    /// The `size × size` stamp of `value − background` centred on `(cx, cy)`,
    /// interpolated with Lanczos-3, or `None` when it reaches a non-finite
    /// sample or the edge.
    pub(super) fn shifted_stamp(
        &self,
        cx: f64,
        cy: f64,
        size: usize,
        background: f64,
    ) -> Option<Vec<f64>> {
        let half = (size / 2) as isize;
        let (x0, y0) = (cx.floor(), cy.floor());
        let (wx, wy) = (lanczos_weights(cx - x0), lanczos_weights(cy - y0));
        let (x0, y0) = (x0 as isize, y0 as isize);
        let reach = LANCZOS_REACH as isize;
        // Rows the vertical pass needs, each interpolated horizontally.
        let first_row = y0 - half - (reach - 1);
        let last_row = y0 + half + reach;
        let row_count = (last_row - first_row + 1) as usize;
        let mut rows = vec![0.0; row_count * size];
        for (row_index, py) in (first_row..=last_row).enumerate() {
            for u in 0..size {
                let px = x0 + u as isize - half;
                let mut sum = 0.0;
                for (k, weight) in wx.iter().enumerate() {
                    let value = self.at(px + k as isize - 2, py);
                    if !value.is_finite() {
                        return None;
                    }
                    sum += weight * (value - background);
                }
                rows[row_index * size + u] = sum;
            }
        }
        let mut stamp = vec![0.0; size * size];
        for v in 0..size {
            // Output row v sits at y0 + v − half; its taps start 2 above.
            let start = (v as isize - half + y0 - 2 - first_row) as usize;
            for u in 0..size {
                stamp[v * size + u] = wy
                    .iter()
                    .enumerate()
                    .map(|(k, weight)| weight * rows[(start + k) * size + u])
                    .sum();
            }
        }
        Some(stamp)
    }
}

/// The sum of `stamp` within `radius` of its centre.
pub(super) fn aperture_sum(stamp: &[f64], size: usize, radius: f64) -> f64 {
    let centre = (size / 2) as f64;
    let mut sum = 0.0;
    for v in 0..size {
        for u in 0..size {
            let (dx, dy) = (u as f64 - centre, v as f64 - centre);
            if dx * dx + dy * dy <= radius * radius {
                sum += stamp[v * size + u];
            }
        }
    }
    sum
}

/// Each pixel's distance from a stamp's centre.
pub(super) fn radii(size: usize) -> Vec<f64> {
    let centre = (size / 2) as f64;
    (0..size * size)
        .map(|index| ((index % size) as f64 - centre).hypot((index / size) as f64 - centre))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shifted_stamp_centres_a_smooth_star() {
        let (width, height) = (64, 64);
        let (cx, cy) = (31.37, 32.71);
        let sigma = 2.2;
        let data: Vec<f32> = (0..width * height)
            .map(|index| {
                let (x, y) = ((index % width) as f64, (index / width) as f64);
                let r2 = (x - cx).powi(2) + (y - cy).powi(2);
                (5.0 + 100.0 * (-0.5 * r2 / (sigma * sigma)).exp()) as f32
            })
            .collect();
        let plane = Plane {
            data: &data,
            width,
            height,
            channels: 1,
            channel: 0,
        };
        let stamp = plane.shifted_stamp(cx, cy, 21, 5.0).unwrap();
        let centre = 10;
        for v in 0..21 {
            for u in 0..21 {
                let r2 = ((u as f64 - centre as f64).powi(2)) + (v as f64 - centre as f64).powi(2);
                let expected = 100.0 * (-0.5 * r2 / (sigma * sigma)).exp();
                assert!(
                    (stamp[v * 21 + u] - expected).abs() < 0.5,
                    "({u}, {v}): {} vs {expected}",
                    stamp[v * 21 + u]
                );
            }
        }
    }
}
