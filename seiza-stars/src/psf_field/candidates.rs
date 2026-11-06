//! Finding and measuring the stars a PSF field is built from.
//!
//! A coarse mesh gives the background and noise, a lightly smoothed
//! residual gives the peaks, and adaptive moments give each candidate's
//! centre, size and elongation. Nothing here decides which stars are used;
//! that is the field's job.

use rayon::prelude::*;

/// Side of a background mesh cell, in pixels.
pub(super) const MESH_CELL: usize = 64;

/// Samples a mesh cell draws at most, spread evenly over it.
const MESH_SAMPLES: usize = 4096;

/// σ of the smoothing that detection runs on, in pixels.
const DETECTION_SMOOTHING: f32 = 1.0;

/// A grid of robust background levels and noise, interpolated between the
/// cells' centres.
pub(super) struct Mesh {
    columns: usize,
    rows: usize,
    width: usize,
    height: usize,
    background: Vec<f32>,
    sigma: Vec<f32>,
}

impl Mesh {
    /// Measure `data` (non-finite samples ignored) in cells of
    /// [`MESH_CELL`] pixels.
    pub(super) fn measure(data: &[f32], width: usize, height: usize) -> Self {
        let columns = width.div_ceil(MESH_CELL).max(1);
        let rows = height.div_ceil(MESH_CELL).max(1);
        let stats: Vec<(f32, f32)> = (0..rows * columns)
            .into_par_iter()
            .map(|index| {
                let (row, column) = (index / columns, index % columns);
                let (x0, x1) = (column * MESH_CELL, ((column + 1) * MESH_CELL).min(width));
                let (y0, y1) = (row * MESH_CELL, ((row + 1) * MESH_CELL).min(height));
                let count = (x1 - x0) * (y1 - y0);
                let stride = ((count as f64 / MESH_SAMPLES as f64).sqrt().ceil() as usize).max(1);
                let mut samples = Vec::with_capacity(count / (stride * stride) + 1);
                for y in (y0..y1).step_by(stride) {
                    for &value in data[y * width + x0..y * width + x1].iter().step_by(stride) {
                        if value.is_finite() {
                            samples.push(value);
                        }
                    }
                }
                clipped_stats(&mut samples).unwrap_or((f32::NAN, f32::NAN))
            })
            .collect();
        let mut background: Vec<f32> = stats.iter().map(|stat| stat.0).collect();
        let mut sigma: Vec<f32> = stats.iter().map(|stat| stat.1).collect();
        fill_missing(&mut background);
        fill_missing(&mut sigma);
        Self {
            columns,
            rows,
            width,
            height,
            background,
            sigma,
        }
    }

    fn interpolate(&self, values: &[f32], x: f64, y: f64) -> f32 {
        let half = MESH_CELL as f64 / 2.0 - 0.5;
        let gx = ((x - half) / MESH_CELL as f64).clamp(0.0, (self.columns - 1) as f64);
        let gy = ((y - half) / MESH_CELL as f64).clamp(0.0, (self.rows - 1) as f64);
        let (c0, r0) = (gx.floor() as usize, gy.floor() as usize);
        let (c1, r1) = ((c0 + 1).min(self.columns - 1), (r0 + 1).min(self.rows - 1));
        let (tx, ty) = ((gx - c0 as f64) as f32, (gy - r0 as f64) as f32);
        let at = |row: usize, column: usize| values[row * self.columns + column];
        let top = at(r0, c0) * (1.0 - tx) + at(r0, c1) * tx;
        let bottom = at(r1, c0) * (1.0 - tx) + at(r1, c1) * tx;
        top * (1.0 - ty) + bottom * ty
    }

    /// The background level at a pixel.
    pub(super) fn background_at(&self, x: f64, y: f64) -> f32 {
        self.interpolate(&self.background, x, y)
    }

    /// The noise at a pixel, as a standard deviation.
    pub(super) fn sigma_at(&self, x: f64, y: f64) -> f32 {
        self.interpolate(&self.sigma, x, y)
    }

    /// `data` less the interpolated background, with non-finite samples
    /// set to zero.
    pub(super) fn residual(&self, data: &[f32]) -> Vec<f32> {
        let mut residual = vec![0.0; data.len()];
        residual
            .par_chunks_mut(self.width)
            .enumerate()
            .for_each(|(y, row)| {
                let source = &data[y * self.width..(y + 1) * self.width];
                for (x, (output, &value)) in row.iter_mut().zip(source).enumerate() {
                    *output = if value.is_finite() {
                        value - self.background_at(x as f64, y as f64)
                    } else {
                        0.0
                    };
                }
            });
        debug_assert_eq!(residual.len(), self.width * self.height);
        residual
    }
}

/// Replace NaN cells by the median of the finite ones.
fn fill_missing(values: &mut [f32]) {
    let mut finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let fill = if finite.is_empty() {
        0.0
    } else {
        median(&mut finite)
    };
    for value in values.iter_mut().filter(|value| !value.is_finite()) {
        *value = fill;
    }
}

/// The median of `values`, which it reorders.
pub(super) fn median(values: &mut [f32]) -> f32 {
    debug_assert!(!values.is_empty());
    let middle = values.len() / 2;
    let (_, &mut upper, _) = values.select_nth_unstable_by(middle, f32::total_cmp);
    if values.len() % 2 == 1 {
        upper
    } else {
        let lower = values[..middle]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        0.5 * (lower + upper)
    }
}

/// The median of `values` as f64, which it reorders.
pub(super) fn median_f64(values: &mut [f64]) -> f64 {
    debug_assert!(!values.is_empty());
    let middle = values.len() / 2;
    let (_, &mut upper, _) = values.select_nth_unstable_by(middle, f64::total_cmp);
    if values.len() % 2 == 1 {
        upper
    } else {
        let lower = values[..middle]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lower + upper)
    }
}

/// A 3σ-clipped median and its MAD noise, or `None` for fewer than 16
/// samples.
pub(super) fn clipped_stats(samples: &mut Vec<f32>) -> Option<(f32, f32)> {
    if samples.len() < 16 {
        return None;
    }
    let mut centre = 0.0;
    let mut sigma = 0.0;
    for _ in 0..4 {
        centre = median(samples);
        let mut deviations: Vec<f32> = samples.iter().map(|v| (v - centre).abs()).collect();
        sigma = 1.4826 * median(&mut deviations);
        if sigma <= 0.0 {
            break;
        }
        let before = samples.len();
        samples.retain(|v| (v - centre).abs() <= 3.0 * sigma);
        if samples.len() == before || samples.len() < 16 {
            break;
        }
    }
    if !samples.is_empty() {
        centre = median(samples);
    }
    Some((centre, sigma))
}

/// `residual` smoothed by a Gaussian of [`DETECTION_SMOOTHING`] px.
pub(super) fn smooth(residual: &[f32], width: usize, height: usize) -> Vec<f32> {
    let radius = 3_isize;
    let kernel: Vec<f32> = (-radius..=radius)
        .map(|offset| (-0.5 * (offset as f32 / DETECTION_SMOOTHING).powi(2)).exp())
        .collect();
    let total: f32 = kernel.iter().sum();
    let kernel: Vec<f32> = kernel.iter().map(|weight| weight / total).collect();
    let clamp = |index: isize, length: usize| index.clamp(0, length as isize - 1) as usize;
    let mut rows = vec![0.0; residual.len()];
    rows.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        let source = &residual[y * width..(y + 1) * width];
        for (x, output) in row.iter_mut().enumerate() {
            *output = kernel
                .iter()
                .enumerate()
                .map(|(tap, weight)| {
                    weight * source[clamp(x as isize + tap as isize - radius, width)]
                })
                .sum();
        }
    });
    let mut output = vec![0.0; residual.len()];
    output
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for (tap, weight) in kernel.iter().enumerate() {
                let source_y = clamp(y as isize + tap as isize - radius, height);
                let source = &rows[source_y * width..(source_y + 1) * width];
                for (sum, value) in row.iter_mut().zip(source) {
                    *sum += weight * value;
                }
            }
        });
    output
}

/// A local maximum of the smoothed residual.
#[derive(Clone, Copy, Debug)]
pub(super) struct Peak {
    pub(super) x: usize,
    pub(super) y: usize,
    /// The smoothed residual at the peak.
    pub(super) value: f32,
}

/// Local maxima over 5×5 pixels of `smooth` that stand `sigma` times the
/// mesh noise above the mesh's level, brightest first.
pub(super) fn find_peaks(
    smooth: &[f32],
    width: usize,
    height: usize,
    noise: &Mesh,
    sigma: f32,
) -> Vec<Peak> {
    if width < 5 || height < 5 {
        return Vec::new();
    }
    let mut peaks: Vec<Peak> = (2..height - 2)
        .into_par_iter()
        .flat_map_iter(|y| {
            (2..width - 2).filter_map(move |x| {
                let value = smooth[y * width + x];
                let (fx, fy) = (x as f64, y as f64);
                let excess = value - noise.background_at(fx, fy);
                if excess.is_nan() || excess <= sigma * noise.sigma_at(fx, fy) {
                    return None;
                }
                for dy in -2_isize..=2 {
                    for dx in -2_isize..=2 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let index = (y as isize + dy) as usize * width + (x as isize + dx) as usize;
                        let neighbour = smooth[index];
                        // Ties go to the first pixel in reading order.
                        let earlier = dy < 0 || (dy == 0 && dx < 0);
                        if neighbour > value || (earlier && neighbour == value) {
                            return None;
                        }
                    }
                }
                Some(Peak { x, y, value })
            })
        })
        .collect();
    peaks.sort_by(|a, b| b.value.total_cmp(&a.value));
    peaks
}

/// A star's position and size from adaptive (matched Gaussian) moments.
#[derive(Clone, Copy, Debug)]
pub(super) struct Moments {
    pub(super) x: f64,
    pub(super) y: f64,
    /// The star's covariance, deweighted: `2 ×` the weighted one at
    /// convergence, which is exact for a Gaussian star.
    pub(super) xx: f64,
    pub(super) xy: f64,
    pub(super) yy: f64,
}

impl Moments {
    /// The Gaussian-equivalent FWHM: `2.3548 · sqrt` of the geometric mean
    /// of the axes' variances.
    pub(super) fn fwhm(&self) -> f64 {
        let determinant = (self.xx * self.yy - self.xy * self.xy).max(0.0);
        2.354_820_045 * determinant.sqrt().sqrt()
    }
}

/// Adaptive moments of `value(x, y) − background` within `radius` of the
/// start, or `None` when they do not converge to a positive shape.
pub(super) fn adaptive_moments(
    value: impl Fn(isize, isize) -> f64,
    start_x: f64,
    start_y: f64,
    radius: f64,
    background: f64,
    initial_sigma: f64,
) -> Option<Moments> {
    let (mut x, mut y) = (start_x, start_y);
    let mut weight = [
        initial_sigma * initial_sigma,
        0.0,
        initial_sigma * initial_sigma,
    ];
    let reach = radius.ceil() as isize;
    for _ in 0..40 {
        let determinant = weight[0] * weight[2] - weight[1] * weight[1];
        if determinant.is_nan() || determinant <= 0.0 {
            return None;
        }
        let (ixx, ixy, iyy) = (
            weight[2] / determinant,
            -weight[1] / determinant,
            weight[0] / determinant,
        );
        let (cx, cy) = (x.round() as isize, y.round() as isize);
        let (mut sum, mut sx, mut sy) = (0.0, 0.0, 0.0);
        let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
        for py in cy - reach..=cy + reach {
            for px in cx - reach..=cx + reach {
                let (dx, dy) = (px as f64 - x, py as f64 - y);
                if dx * dx + dy * dy > radius * radius {
                    continue;
                }
                let w = (-0.5 * (ixx * dx * dx + 2.0 * ixy * dx * dy + iyy * dy * dy)).exp();
                let sample = (value(px, py) - background) * w;
                sum += sample;
                sx += sample * dx;
                sy += sample * dy;
                sxx += sample * dx * dx;
                sxy += sample * dx * dy;
                syy += sample * dy * dy;
            }
        }
        if sum.is_nan() || sum <= 0.0 {
            return None;
        }
        let (mx, my) = (sx / sum, sy / sum);
        let next = [
            2.0 * (sxx / sum - mx * mx),
            2.0 * (sxy / sum - mx * my),
            2.0 * (syy / sum - my * my),
        ];
        x += mx;
        y += my;
        if (x - start_x).abs() > radius || (y - start_y).abs() > radius {
            return None;
        }
        let change = (next[0] - weight[0]).abs() + (next[2] - weight[2]).abs();
        // Damp the shape update: a star's covariance can oscillate between
        // too narrow and too wide when the weight starts far from it.
        weight = [
            0.5 * (weight[0] + next[0]),
            0.5 * (weight[1] + next[1]),
            0.5 * (weight[2] + next[2]),
        ];
        if weight[0] <= 0.01 || weight[2] <= 0.01 {
            return None;
        }
        let largest = (radius / 2.0).powi(2);
        weight[0] = weight[0].min(largest);
        weight[2] = weight[2].min(largest);
        if mx.abs() < 1e-4 && my.abs() < 1e-4 && change < 1e-4 * (weight[0] + weight[2]) {
            break;
        }
    }
    Some(Moments {
        x,
        y,
        xx: weight[0],
        xy: weight[1],
        yy: weight[2],
    })
}

/// A spatial index over peaks, for neighbour tests.
pub(super) struct PeakGrid<'a> {
    peaks: &'a [Peak],
    cell: usize,
    columns: usize,
    rows: usize,
    buckets: Vec<Vec<u32>>,
}

impl<'a> PeakGrid<'a> {
    pub(super) fn new(peaks: &'a [Peak], width: usize, height: usize, cell: usize) -> Self {
        let cell = cell.max(1);
        let columns = width.div_ceil(cell).max(1);
        let rows = height.div_ceil(cell).max(1);
        let mut buckets = vec![Vec::new(); columns * rows];
        for (index, peak) in peaks.iter().enumerate() {
            buckets[(peak.y / cell) * columns + peak.x / cell].push(index as u32);
        }
        Self {
            peaks,
            cell,
            columns,
            rows,
            buckets,
        }
    }

    /// The brightest smoothed value of a peak within `radius` of `(x, y)`
    /// but farther than `exclude` from it.
    pub(super) fn brightest_near(&self, x: f64, y: f64, radius: f64, exclude: f64) -> f32 {
        let mut brightest = f32::NEG_INFINITY;
        self.for_each_near(x, y, radius, exclude, |peak| {
            brightest = brightest.max(peak.value);
        });
        brightest
    }

    /// The peaks within `radius` of `(x, y)` but farther than `exclude`
    /// from it.
    pub(super) fn near(&self, x: f64, y: f64, radius: f64, exclude: f64) -> Vec<Peak> {
        let mut found = Vec::new();
        self.for_each_near(x, y, radius, exclude, |peak| found.push(peak));
        found
    }

    fn for_each_near(
        &self,
        x: f64,
        y: f64,
        radius: f64,
        exclude: f64,
        mut visit: impl FnMut(Peak),
    ) {
        let reach = (radius / self.cell as f64).ceil() as isize;
        let (column, row) = (
            (x.max(0.0) as usize / self.cell) as isize,
            (y.max(0.0) as usize / self.cell) as isize,
        );
        for r in row - reach..=row + reach {
            for c in column - reach..=column + reach {
                if r < 0 || c < 0 || r >= self.rows as isize || c >= self.columns as isize {
                    continue;
                }
                for &index in &self.buckets[r as usize * self.columns + c as usize] {
                    let peak = self.peaks[index as usize];
                    let distance = (peak.x as f64 - x).hypot(peak.y as f64 - y);
                    if distance <= radius && distance > exclude {
                        visit(peak);
                    }
                }
            }
        }
    }
}

/// The level at which the brightest stars clip, from the peaks' raw values:
/// three or more within 2% of one another, at 90% of the brightest or
/// above, mark a clip level, and stars from 80% of it up count as
/// saturated. A stack's clipped cores share one level closely, as unclipped
/// stars do not; a single hot pixel above them does not hide it. `None`
/// when no such cluster exists.
pub(super) fn clip_level(mut peak_values: Vec<f32>) -> Option<f32> {
    peak_values.retain(|value| value.is_finite());
    peak_values.sort_by(|a, b| b.total_cmp(a));
    let brightest = *peak_values.first()?;
    if brightest.is_nan() || brightest <= 0.0 {
        return None;
    }
    for start in 0..peak_values.len() {
        let top = peak_values[start];
        if top < 0.9 * brightest {
            break;
        }
        let clustered = peak_values[start..]
            .iter()
            .take_while(|&&value| value >= 0.98 * top)
            .count();
        if clustered >= 3 {
            let mut cluster = peak_values[start..start + clustered].to_vec();
            return Some(0.8 * median(&mut cluster));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_moments_measure_a_gaussian() {
        let (sx, sy) = (2.0_f64, 1.4_f64);
        let (cx, cy) = (20.3, 19.6);
        let value = |x: isize, y: isize| {
            let (dx, dy) = (x as f64 - cx, y as f64 - cy);
            10.0 + 1000.0 * (-0.5 * (dx * dx / (sx * sx) + dy * dy / (sy * sy))).exp()
        };
        let moments = adaptive_moments(value, 20.0, 20.0, 12.0, 10.0, 1.5).unwrap();
        assert!((moments.x - cx).abs() < 1e-3, "{moments:?}");
        assert!((moments.y - cy).abs() < 1e-3, "{moments:?}");
        assert!((moments.xx - sx * sx).abs() < 0.05, "{moments:?}");
        assert!((moments.yy - sy * sy).abs() < 0.05, "{moments:?}");
        assert!(moments.xy.abs() < 1e-3);
    }

    #[test]
    fn clip_level_needs_a_pile_up() {
        assert_eq!(clip_level(vec![100.0, 60.0, 40.0, 30.0]), None);
        // Bright stars spread over a few percent are not a clip level.
        assert_eq!(clip_level(vec![100.0, 97.0, 94.0, 91.0, 88.0]), None);
        let level = clip_level(vec![70_000.0, 69_800.0, 69_500.0, 40_000.0]).unwrap();
        assert!((level - 0.8 * 69_800.0).abs() < 1.0);
        // A hot pixel above the clipped cores does not hide them.
        let level = clip_level(vec![67_200.0, 62_090.0, 61_600.0, 61_470.0, 50_000.0]).unwrap();
        assert!((level - 0.8 * 61_600.0).abs() < 1.0, "{level}");
    }

    #[test]
    fn mesh_measures_background_and_noise() {
        let (width, height) = (128, 96);
        let mut state = 7_u64;
        let mut noise = move || {
            // A sum of uniforms is close enough to normal for a MAD.
            let mut total = 0.0;
            for _ in 0..12 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                total += (state >> 11) as f64 / (1u64 << 53) as f64;
            }
            (total - 6.0) as f32
        };
        let data: Vec<f32> = (0..width * height).map(|_| 100.0 + 3.0 * noise()).collect();
        let mesh = Mesh::measure(&data, width, height);
        assert!((mesh.background_at(50.0, 40.0) - 100.0).abs() < 0.5);
        assert!((mesh.sigma_at(50.0, 40.0) - 3.0).abs() < 0.4);
    }
}
