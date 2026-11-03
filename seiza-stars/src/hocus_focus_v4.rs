//! HocusFocus 4's star detector (`StarDetector.cs` at release v4.0.0.21),
//! run when [`HocusFocusParams::version`] is [`HocusFocusVersion::V4`].
//!
//! Like HocusFocus it works on the frame scaled to 0..1 (each 16-bit value
//! over 65536) and reports brightnesses back in ADU. The stages, in order:
//!
//! 1. hot pixels replaced by their 3 × 3 median where they differ from it by
//!    more than `hotpixel_threshold`;
//! 2. a Gaussian blur of `2r + 1` taps (σ = 0.159758 × taps) gives the
//!    noise-reduced image; its kappa-sigma noise thresholds structures, and
//!    the unblurred image's own noise clips and gates stars;
//! 3. the à trous B3-spline residual of `structure_layers` layers is
//!    subtracted from the noise-reduced image (clamped at 0), and the result
//!    blurred with `2 × layers + 1` taps;
//! 4. the structure map is binarized against a per-pixel surface, the
//!    128-pixel block median plus `noise_clipping_multiplier` block noise
//!    upsampled bilinearly, or against one global threshold;
//! 5. 8-connected regions are candidates. Each is gated as HocusFocus gates
//!    it, against a local background plane fitted to the ring around it,
//!    centred by an iterative windowed centroid, and measured for HFR over a
//!    circular aperture. Stars whose ring shows a one-sided excess (a
//!    neighbour) are dropped.

use crate::hocus_focus_star_detection::{HocusFocusParams, HocusFocusStar, MeasurementAverage};
use crate::psf_fitting::{PSFFitter, PSFType};
use rayon::prelude::*;

/// Full scale of HocusFocus's normalized image: it divides 16-bit values by
/// `1 << 16`.
const SCALE: f64 = 65536.0;

/// What the v4 pipeline found, in ADU and the image's pixels.
pub(crate) struct V4Result {
    pub stars: Vec<HocusFocusStar>,
    pub average_hfr: f64,
    pub average_fwhm: f64,
    pub noise_sigma: f64,
    pub background_mean: f64,
}

/// Run HocusFocus 4's detector on hot-pixel-filtered `data`.
/// The inner region an autofocus outer crop leaves out of the search.
#[derive(Debug, Clone, Copy)]
pub(crate) struct InnerCrop {
    /// The full frame's inner-crop fractions, which HocusFocus scales by the
    /// cropped frame's size to clear structures.
    pub ratios: crate::hocus_focus_star_detection::CentredRatio,
    /// The full frame's inner rectangle in this frame's pixels
    /// (x, y, width, height); a star whose box touches it is dropped.
    pub excluded: (f64, f64, f64, f64),
}

impl InnerCrop {
    /// The same crop for a frame binned by `factor`.
    pub(crate) fn binned(self, factor: usize) -> Self {
        let f = factor as f64;
        let (x, y, w, h) = self.excluded;
        Self {
            ratios: self.ratios,
            excluded: (x / f, y / f, w / f, h / f),
        }
    }
}

pub(crate) fn detect(
    data: &[u16],
    width: usize,
    height: usize,
    params: &HocusFocusParams,
    inner: Option<InnerCrop>,
) -> V4Result {
    let mut src: Vec<f32> = data
        .par_iter()
        .map(|&v| (v as f64 / SCALE) as f32)
        .collect();
    let r = params.noise_reduction_radius;
    let measure_blurred = r > 0 && params.measurement_noise_reduction;
    if measure_blurred {
        src = gaussian_blur(&src, width, height, 2 * r + 1);
    }
    let noise_reduced = if r > 0 && !measure_blurred {
        gaussian_blur(&src, width, height, 2 * r + 1)
    } else {
        src.clone()
    };
    let structure_noise = kappa_sigma(&noise_reduced, params.noise_clipping_multiplier);
    let measurement_noise = if r > 0 && !measure_blurred {
        kappa_sigma(&src, params.noise_clipping_multiplier)
    } else {
        structure_noise
    };
    let sigma_grid = params.adaptive_binarization.then(|| {
        local_grid(
            &noise_reduced,
            width,
            height,
            params.adaptive_block_size.max(1),
        )
    });

    let mut structure = noise_reduced;
    atrous_subtract(&mut structure, width, height, params.structure_layers);
    let structure = gaussian_blur(&structure, width, height, params.structure_layers * 2 + 1);
    let binary: Vec<bool> = match &sigma_grid {
        Some(sigma) => {
            let median = local_grid(&structure, width, height, params.adaptive_block_size.max(1));
            adaptive_binarize(
                &structure,
                width,
                height,
                &median,
                sigma,
                params.noise_clipping_multiplier,
            )
        }
        None => {
            let threshold =
                histogram_median(&structure) + params.noise_clipping_multiplier * structure_noise.0;
            structure
                .par_iter()
                .map(|&v| v as f64 > threshold)
                .collect()
        }
    };
    let mut binary = binary;
    if let Some(crop) = inner {
        let (x0, y0, w, h) = crop.ratios.pixels_floor(width, height);
        for y in y0..(y0 + h).min(height) {
            for value in &mut binary[y * width + x0.min(width)..y * width + (x0 + w).min(width)] {
                *value = false;
            }
        }
    }
    let candidates = if params.connected_components {
        connected_regions(&binary, width, height)
    } else {
        walked_regions(&binary, width, height)
    };
    crate::debug_detection!(
        "HocusFocus 4: {} candidates, structure noise {:.3}, measurement noise {:.3} ADU",
        candidates.len(),
        structure_noise.0 * SCALE,
        measurement_noise.0 * SCALE
    );

    let sigma = measurement_noise.0;
    let rejected = std::sync::Mutex::new([0usize; 10]);
    let mut stars: Vec<HocusFocusStar> = candidates
        .par_iter()
        .filter(|candidate| {
            // HocusFocus's OutsideROI: a star whose box is inside or
            // touches the inner region is not counted.
            inner.is_none_or(|crop| {
                let (ex, ey, ew, eh) = crop.excluded;
                let (bx, by, bw, bh) = candidate.bounds;
                let (bx, by, bw, bh) = (bx as f64, by as f64, bw as f64, bh as f64);
                bx + bw <= ex || ex + ew <= bx || by + bh <= ey || ey + eh <= by
            })
        })
        .filter_map(|candidate| {
            evaluate(&src, data, width, height, candidate, params, sigma)
                .map_err(|gate| {
                    rejected.lock().unwrap()[gate as usize] += 1;
                })
                .ok()
        })
        .collect();
    stars.sort_by(|a, b| {
        (a.position.1, a.position.0)
            .partial_cmp(&(b.position.1, b.position.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    crate::debug_detection!(
        "HocusFocus 4: accepted {}, rejected [small, border, distorted, degenerate, \
         low sensitivity, off centre, flat, HFR failed, HFR too small, contaminated] = {:?}",
        stars.len(),
        rejected.lock().unwrap()
    );

    let (average_hfr, average_fwhm) = summarize(&stars, params);
    V4Result {
        stars,
        average_hfr,
        average_fwhm,
        noise_sigma: sigma * SCALE,
        background_mean: measurement_noise.1 * SCALE,
    }
}

/// HocusFocus's frame HFR and FWHM: over unsaturated stars when at least
/// three remain, the median or the outlier-trimmed mean.
fn summarize(stars: &[HocusFocusStar], params: &HocusFocusParams) -> (f64, f64) {
    let unsaturated = stars.iter().filter(|s| !s.saturated).collect::<Vec<_>>();
    let pool = if unsaturated.len() >= 3 {
        unsaturated
    } else {
        stars.iter().collect()
    };
    if pool.len() < 2 {
        return pool.first().map_or((0.0, 0.0), |s| (s.hfr, s.fwhm));
    }
    let hfrs = pool.iter().map(|s| s.hfr).collect::<Vec<_>>();
    let fwhms = pool.iter().map(|s| s.fwhm).collect::<Vec<_>>();
    match params.measurement_average {
        MeasurementAverage::Median => (median_mad(&hfrs).0, median_mad(&fwhms).0),
        MeasurementAverage::MeanOutliers => {
            let n = pool.len() as f64;
            (hfrs.iter().sum::<f64>() / n, fwhms.iter().sum::<f64>() / n)
        }
    }
}

fn median_mad(values: &[f64]) -> (f64, f64) {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = |s: &[f64]| {
        let n = s.len();
        if n.is_multiple_of(2) {
            (s[n / 2 - 1] + s[n / 2]) / 2.0
        } else {
            s[n / 2]
        }
    };
    let median = middle(&sorted);
    let mut deviations = sorted
        .iter()
        .map(|v| (v - median).abs())
        .collect::<Vec<_>>();
    deviations.sort_by(f64::total_cmp);
    (median, 1.483 * middle(&deviations))
}

/// OpenCV's `BORDER_REFLECT` index (`dcba|abcd`).
fn reflect(mut i: isize, n: usize) -> usize {
    let n = n as isize;
    if n == 1 {
        return 0;
    }
    while i < 0 || i >= n {
        i = if i < 0 { -i - 1 } else { 2 * n - 1 - i };
    }
    i as usize
}

/// HocusFocus's `ConvolveGaussian`: a separable Gaussian of `taps` taps and
/// σ = 0.159758 × taps, with reflected borders.
fn gaussian_blur(src: &[f32], width: usize, height: usize, taps: usize) -> Vec<f32> {
    let sigma = 0.159758 * taps as f64;
    let half = (taps / 2) as isize;
    let mut kernel = (0..taps)
        .map(|i| {
            let x = i as f64 - half as f64;
            (-x * x / (2.0 * sigma * sigma)).exp()
        })
        .collect::<Vec<_>>();
    let sum = kernel.iter().sum::<f64>();
    kernel.iter_mut().for_each(|k| *k /= sum);
    separable(src, width, height, &kernel)
}

/// Convolve rows then columns with a centred `kernel`, reflecting borders.
fn separable(src: &[f32], width: usize, height: usize, kernel: &[f64]) -> Vec<f32> {
    let half = (kernel.len() / 2) as isize;
    let mut rows = vec![0f32; src.len()];
    rows.par_chunks_mut(width).enumerate().for_each(|(y, out)| {
        let row = &src[y * width..(y + 1) * width];
        for (x, o) in out.iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for (k, &w) in kernel.iter().enumerate() {
                acc += w * row[reflect(x as isize + k as isize - half, width)] as f64;
            }
            *o = acc as f32;
        }
    });
    let mut out = vec![0f32; src.len()];
    out.par_chunks_mut(width).enumerate().for_each(|(y, o)| {
        let sources = kernel
            .iter()
            .enumerate()
            .map(|(k, &w)| (w, reflect(y as isize + k as isize - half, height)))
            .collect::<Vec<_>>();
        for (x, value) in o.iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for &(w, sy) in &sources {
                acc += w * rows[sy * width + x] as f64;
            }
            *value = acc as f32;
        }
    });
    out
}

/// HocusFocus's `AtrousWaveletFast.ComputeResidualAndSubtractInPlace`: the
/// residual after `layers` B3-spline à trous smoothings, subtracted from the
/// image and clamped to 0..1.
fn atrous_subtract(image: &mut [f32], width: usize, height: usize, layers: usize) {
    if layers == 0 {
        return;
    }
    const W0: f32 = 0.375;
    const W1: f32 = 0.25;
    const W2: f32 = 0.0625;
    let mut current = image.to_vec();
    let mut horizontal = vec![0f32; image.len()];
    for layer in 0..layers {
        let s = 1isize << layer;
        horizontal
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, out)| {
                let row = &current[y * width..(y + 1) * width];
                for (x, o) in out.iter_mut().enumerate() {
                    let x = x as isize;
                    let at = |i: isize| row[reflect(i, width)];
                    *o = W2 * (at(x - 2 * s) + at(x + 2 * s))
                        + W1 * (at(x - s) + at(x + s))
                        + W0 * at(x);
                }
            });
        let last = layer + 1 == layers;
        let original: &[f32] = image;
        let mut next = vec![0f32; image.len()];
        next.par_chunks_mut(width).enumerate().for_each(|(y, out)| {
            let y = y as isize;
            let row = |i: isize| &horizontal[reflect(i, height) * width..][..width];
            let (m2, m1, c0, p1, p2) = (
                row(y - 2 * s),
                row(y - s),
                row(y),
                row(y + s),
                row(y + 2 * s),
            );
            for (x, o) in out.iter_mut().enumerate() {
                let r = W2 * (m2[x] + p2[x]) + W1 * (m1[x] + p1[x]) + W0 * c0[x];
                *o = if last {
                    (original[y as usize * width + x] - r).clamp(0.0, 1.0)
                } else {
                    r
                };
            }
        });
        current = next;
    }
    image.copy_from_slice(&current);
}

/// HocusFocus's `KappaSigmaNoiseEstimate`: the mean and spread of the
/// positive pixels below mean + `clip` σ, iterated up to five times until σ
/// moves by at most 1e-5. Returns `(sigma, mean)`.
fn kappa_sigma(image: &[f32], clip: f64) -> (f64, f64) {
    let mut threshold = f32::MAX;
    let (mut last_sigma, mut last_mean) = (1.0f64, 1.0f64);
    for iteration in 0..5 {
        // C#'s float.Epsilon is the smallest subnormal.
        let tiny = f32::from_bits(1);
        let (lo, hi) = (tiny, threshold - tiny);
        let (n, sum, squares) = image
            .par_iter()
            .filter(|&&v| v >= lo && v <= hi)
            .fold(
                || (0u64, 0f64, 0f64),
                |(n, s, q), &v| (n + 1, s + v as f64, q + v as f64 * v as f64),
            )
            .reduce(|| (0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        if n == 0 {
            break;
        }
        let mean = sum / n as f64;
        let sigma = (squares / n as f64 - mean * mean).max(0.0).sqrt();
        if iteration > 0 && (sigma - last_sigma).abs() <= 1e-5 {
            last_sigma = sigma;
            break;
        }
        threshold = (mean + clip * sigma) as f32;
        last_sigma = sigma;
        last_mean = mean;
    }
    (last_sigma, last_mean)
}

/// HocusFocus's histogram median of a 0..1 image: 65536 buckets, with the
/// position inside the median bucket interpolated as it does.
fn histogram_median(image: &[f32]) -> f64 {
    const BUCKETS: usize = 1 << 16;
    let histogram = image
        .par_iter()
        .fold(
            || vec![0u64; BUCKETS],
            |mut h, &v| {
                let i =
                    ((v as f64 * BUCKETS as f64).floor() as isize).clamp(0, BUCKETS as isize - 1);
                h[i as usize] += 1;
                h
            },
        )
        .reduce(
            || vec![0u64; BUCKETS],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                a
            },
        );
    let target = image.len() as f64 / 2.0;
    let mut count = 0u64;
    for (i, &h) in histogram.iter().enumerate() {
        count += h;
        if count as f64 >= target {
            let ratio = (count as f64 - target) / h as f64;
            return (i as f64 + ratio) / BUCKETS as f64;
        }
    }
    1.0
}

/// Per-block medians and noise (1.4826 × MAD, at least 1e-6), the blocks
/// `block` pixels square with partial blocks at the right and bottom.
struct Grid {
    rows: usize,
    cols: usize,
    median: Vec<f32>,
    sigma: Vec<f32>,
}

fn local_grid(image: &[f32], width: usize, height: usize, block: usize) -> Grid {
    let cols = width.div_ceil(block);
    let rows = height.div_ceil(block);
    let cells = (0..rows * cols)
        .into_par_iter()
        .map(|cell| {
            let (gy, gx) = (cell / cols, cell % cols);
            let mut values = Vec::with_capacity(block * block);
            for y in gy * block..((gy + 1) * block).min(height) {
                values.extend_from_slice(
                    &image[y * width + gx * block..y * width + ((gx + 1) * block).min(width)],
                );
            }
            values.sort_unstable_by(f32::total_cmp);
            let median = values[values.len() >> 1];
            let mut deviations = values
                .iter()
                .map(|v| (v - median).abs())
                .collect::<Vec<_>>();
            deviations.sort_unstable_by(f32::total_cmp);
            let sigma = ((deviations[deviations.len() >> 1] as f64 * 1.4826) as f32).max(1e-6);
            (median, sigma)
        })
        .collect::<Vec<_>>();
    Grid {
        rows,
        cols,
        median: cells.iter().map(|c| c.0).collect(),
        sigma: cells.iter().map(|c| c.1).collect(),
    }
}

/// HocusFocus's adaptive binarization: the block median plus `clip` block
/// noise, resized to the image by OpenCV's bilinear `INTER_LINEAR`; a pixel
/// is structure when it exceeds the surface.
fn adaptive_binarize(
    image: &[f32],
    width: usize,
    height: usize,
    median: &Grid,
    sigma: &Grid,
    clip: f64,
) -> Vec<bool> {
    let grid = median
        .median
        .iter()
        .zip(&sigma.sigma)
        .map(|(&m, &s)| (m as f64 + clip * s as f64) as f32)
        .collect::<Vec<_>>();
    let (gw, gh) = (median.cols, median.rows);
    // OpenCV's source coordinate and weights for one axis.
    let axis = |out: usize, n_out: usize, n_in: usize| {
        let f = (out as f64 + 0.5) * (n_in as f64 / n_out as f64) - 0.5;
        let i0 = f.floor();
        let t = (f - i0) as f32;
        let i0 = i0 as isize;
        let a = i0.clamp(0, n_in as isize - 1) as usize;
        let b = (i0 + 1).clamp(0, n_in as isize - 1) as usize;
        if i0 < 0 { (0, 0, 0.0) } else { (a, b, t) }
    };
    let columns = (0..width).map(|x| axis(x, width, gw)).collect::<Vec<_>>();
    let mut out = vec![false; image.len()];
    out.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        let (y0, y1, ty) = axis(y, height, gh);
        for (x, o) in row.iter_mut().enumerate() {
            let (x0, x1, tx) = columns[x];
            let top = grid[y0 * gw + x0] * (1.0 - tx) + grid[y0 * gw + x1] * tx;
            let bottom = grid[y1 * gw + x0] * (1.0 - tx) + grid[y1 * gw + x1] * tx;
            *o = image[y * width + x] > top * (1.0 - ty) + bottom * ty;
        }
    });
    out
}

/// A candidate region: its bounding box (x, y, width, height) and pixels.
struct Region {
    bounds: (usize, usize, usize, usize),
    points: Vec<(usize, usize)>,
}

/// HocusFocus's `CollectStarCandidatesConnected`: 8-connected regions in
/// raster order of their first run, each region's pixels run by run.
fn connected_regions(binary: &[bool], width: usize, height: usize) -> Vec<Region> {
    // Runs per row as (y, start, end).
    let mut runs: Vec<(usize, usize, usize)> = Vec::new();
    let mut row_start = vec![0usize; height + 1];
    for y in 0..height {
        row_start[y] = runs.len();
        let row = &binary[y * width..(y + 1) * width];
        let mut x = 0;
        while x < width {
            if row[x] {
                let s = x;
                while x < width && row[x] {
                    x += 1;
                }
                runs.push((y, s, x));
            } else {
                x += 1;
            }
        }
    }
    row_start[height] = runs.len();
    let mut parent = (0..runs.len()).collect::<Vec<_>>();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for y in 1..height {
        let (prev, this) = (
            row_start[y - 1]..row_start[y],
            row_start[y]..row_start[y + 1],
        );
        let mut j = prev.start;
        for id in this {
            let (_, s, e) = runs[id];
            while j < prev.end && runs[j].2 < s {
                j += 1;
            }
            let mut k = j;
            while k < prev.end && runs[k].1 <= e {
                let (a, b) = (find(&mut parent, k), find(&mut parent, id));
                if a != b {
                    let (lo, hi) = (a.min(b), a.max(b));
                    parent[hi] = lo;
                }
                k += 1;
            }
        }
    }
    let mut index = vec![usize::MAX; runs.len()];
    let mut regions: Vec<Vec<usize>> = Vec::new();
    for id in 0..runs.len() {
        let root = find(&mut parent, id);
        if index[root] == usize::MAX {
            index[root] = regions.len();
            regions.push(Vec::new());
        }
        regions[index[root]].push(id);
    }
    regions
        .into_iter()
        .map(|ids| {
            let (mut x0, mut x1, mut y0, mut y1) = (usize::MAX, 0, usize::MAX, 0);
            let mut points = Vec::new();
            for id in ids {
                let (y, s, e) = runs[id];
                x0 = x0.min(s);
                x1 = x1.max(e - 1);
                y0 = y0.min(y);
                y1 = y1.max(y);
                points.extend((s..e).map(|x| (x, y)));
            }
            Region {
                bounds: (x0, y0, x1 - x0 + 1, y1 - y0 + 1),
                points,
            }
        })
        .collect()
}

/// HocusFocus's legacy walker (`CollectStarCandidatesLegacy`): grow a box
/// down and right from each lit pixel, then clear the whole box.
fn walked_regions(binary: &[bool], width: usize, height: usize) -> Vec<Region> {
    let mut map = binary.to_vec();
    let mut regions = Vec::new();
    let (x_right, y_bottom) = (width - 1, height - 1);
    for y_top in 0..y_bottom {
        for x_left in 0..x_right {
            if !map[y_top * width + x_left] {
                continue;
            }
            let mut points = Vec::new();
            let (mut bx, mut bw) = (x_left, 1usize);
            let bh;
            let mut y = y_top;
            loop {
                let row = y * width;
                let mut added = 0;
                if map[row + x_left] {
                    points.push((x_left, y));
                    added += 1;
                }
                let mut start = x_left;
                if added > 0 {
                    while start > 0 && map[row + start - 1] {
                        start -= 1;
                        points.push((start, y));
                        added += 1;
                    }
                }
                let mut end = x_left;
                while end < x_right {
                    if !map[row + end + 1] {
                        if added > 0 || end >= bx + bw {
                            break;
                        }
                        end += 1;
                    } else {
                        end += 1;
                        points.push((end, y));
                        added += 1;
                    }
                }
                if start < bx {
                    bw += bx - start;
                    bx = start;
                }
                if end + 1 > bx + bw {
                    bw = end + 1 - bx;
                }
                if added == 0 {
                    bh = y - y_top;
                    break;
                }
                if y == y_bottom {
                    bh = y - y_top + 1;
                    break;
                }
                y += 1;
            }
            for yy in y_top..y_top + bh {
                for xx in bx..bx + bw {
                    map[yy * width + xx] = false;
                }
            }
            regions.push(Region {
                bounds: (bx, y_top, bw, bh),
                points,
            });
        }
    }
    regions
}

/// Why a candidate is not a star; indexes the debug census.
#[derive(Clone, Copy)]
enum Gate {
    TooSmall = 0,
    OnBorder = 1,
    TooDistorted = 2,
    Degenerate = 3,
    LowSensitivity = 4,
    NotCentered = 5,
    TooFlat = 6,
    HfrFailed = 7,
    TooLowHfr = 8,
    Contaminated = 9,
}

/// The background plane `b0 + b1 (x − x0) + b2 (y − y0)`.
#[derive(Clone, Copy)]
struct Plane {
    origin: (f64, f64),
    b: [f64; 3],
}

impl Plane {
    fn at(&self, x: f64, y: f64) -> f64 {
        self.b[0] + self.b[1] * (x - self.origin.0) + self.b[2] * (y - self.origin.1)
    }
}

fn bilinear(src: &[f32], width: usize, height: usize, x: f64, y: f64) -> f64 {
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(width - 1), (y0 + 1).min(height - 1));
    let (fx, fy) = (x - x0 as f64, y - y0 as f64);
    let at = |x: usize, y: usize| src[y * width + x] as f64;
    let top = at(x0, y0) + fx * (at(x1, y0) - at(x0, y0));
    let bottom = at(x0, y1) + fx * (at(x1, y1) - at(x0, y1));
    top + fy * (bottom - top)
}

/// HocusFocus's `EvaluateStarCandidate` with its v4 defaults (no defocus or
/// donut options).
fn evaluate(
    src: &[f32],
    raw: &[u16],
    width: usize,
    height: usize,
    region: &Region,
    params: &HocusFocusParams,
    sigma: f64,
) -> Result<HocusFocusStar, Gate> {
    let (bx, by, bw, bh) = region.bounds;
    if bw < params.min_star_size || bh < params.min_star_size {
        return Err(Gate::TooSmall);
    }
    if bx == 0 || by == 0 || bx + bw == width || by + bh == height {
        return Err(Gate::OnBorder);
    }
    let d = bw.max(bh) as f64;
    if (region.points.len() as f64 / d / d) < params.max_distortion {
        return Err(Gate::TooDistorted);
    }
    let c = parameters(src, width, height, region, params, sigma).ok_or(Gate::Degenerate)?;
    let saturation = params.saturation_threshold / SCALE;
    let saturated = c.background + c.peak >= saturation;
    if c.normalized_brightness / sigma <= params.sensitivity {
        return Err(Gate::LowSensitivity);
    }
    let tol_w = bw as f64 * params.star_center_tolerance;
    let tol_h = bh as f64 * params.star_center_tolerance;
    let min_x = bx as f64 + (bw as f64 - tol_w) / 2.0;
    let min_y = by as f64 + (bh as f64 - tol_h) / 2.0;
    let (cx, cy) = c.center;
    if cx < min_x || cx > min_x + tol_w || cy < min_y || cy > min_y + tol_h {
        return Err(Gate::NotCentered);
    }
    if c.star_median >= params.peak_response * c.peak {
        return Err(Gate::TooFlat);
    }
    let hfr = measure_hfr(
        src,
        width,
        height,
        region.bounds,
        c.center,
        &c.plane,
        params,
        sigma,
    )
    .ok_or(Gate::HfrFailed)?;
    if hfr <= params.min_hfr {
        return Err(Gate::TooLowHfr);
    }
    if c.contaminated && params.contamination_sensitivity > 0.0 {
        return Err(Gate::Contaminated);
    }
    let background = c.background * SCALE;
    let peak = c.peak * SCALE;
    let psf_model = (params.psf_type != PSFType::None)
        .then(|| {
            PSFFitter::new(params.psf_type).fit_star(
                raw,
                width,
                height,
                cx,
                cy,
                bw as f64,
                bh as f64,
                background,
                background + peak,
            )
        })
        .flatten();
    let fwhm = psf_model.as_ref().map_or(hfr * 2.0 * 1.177, |p| p.fwhm);
    Ok(HocusFocusStar {
        position: c.center,
        hfr,
        fwhm,
        brightness: background + peak,
        background,
        snr: c.peak / sigma.max(f64::MIN_POSITIVE),
        flux: c.total_flux * SCALE,
        pixel_count: region.points.len(),
        saturated,
        psf_model,
    })
}

struct Candidate {
    center: (f64, f64),
    background: f64,
    plane: Plane,
    total_flux: f64,
    peak: f64,
    normalized_brightness: f64,
    star_median: f64,
    contaminated: bool,
}

/// HocusFocus 4's `ComputeStarParameters`.
fn parameters(
    src: &[f32],
    width: usize,
    height: usize,
    region: &Region,
    params: &HocusFocusParams,
    sigma: f64,
) -> Option<Candidate> {
    let (bx, by, bw, bh) = region.bounds;
    let e = params.background_box_expansion;
    let (x0, y0) = (bx.saturating_sub(e), by.saturating_sub(e));
    let (x1, y1) = ((bx + bw + e).min(width), (by + bh + e).min(height));
    let (cx, cy) = (bx as f64 + bw as f64 / 2.0, by as f64 + bh as f64 / 2.0);
    let mut ring = Vec::new();
    for y in y0..y1 {
        for x in x0..x1 {
            if x < bx || x >= bx + bw || y < by || y >= by + bh {
                ring.push((
                    (x as f64 - cx) as f32,
                    (y as f64 - cy) as f32,
                    src[y * width + x],
                ));
            }
        }
    }
    if ring.is_empty() {
        return None;
    }
    let mut sorted = ring.iter().map(|r| r.2).collect::<Vec<_>>();
    sorted.sort_unstable_by(f32::total_cmp);
    let median = sorted[sorted.len() >> 1] as f64;
    let local_sigma = if sorted.len() >= 8 {
        let mut dev = sorted
            .iter()
            .map(|&v| (v as f64 - median).abs())
            .collect::<Vec<_>>();
        dev.sort_by(f64::total_cmp);
        1.4826 * median_of(&dev)
    } else {
        0.0
    };
    let fallback = if local_sigma > 0.0 {
        local_sigma
    } else {
        sigma
    };
    let (plane, contaminated) =
        background_plane(&ring, (cx, cy), fallback, params.contamination_sensitivity).unwrap_or((
            Plane {
                origin: (cx, cy),
                b: [median, 0.0, 0.0],
            },
            false,
        ));

    let clip = params.star_clipping_multiplier * sigma;
    let above = region
        .points
        .iter()
        .filter_map(|&(x, y)| {
            let v = src[y * width + x] as f64 - plane.at(x as f64, y as f64);
            (v > clip).then_some(v)
        })
        .collect::<Vec<_>>();
    // HocusFocus starts its minimum at 1 and its maximum at 0.
    let (lo, hi) = above
        .iter()
        .fold((1.0f64, 0.0f64), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    if above.len() == 1 || hi <= lo {
        return None;
    }
    let total = above.iter().sum::<f64>();
    let peak = above.iter().fold(0.0f64, |p, &v| p.max(v));
    let mut values = above.clone();
    values.sort_by(f64::total_cmp);
    let star_median = median_of(&values);
    let mean = total / above.len() as f64;

    // Iterative centroid: all clipped pixels, then twice within a circle of
    // half the box's smaller side around the last estimate.
    let aperture = bw.min(bh) as f64 / 2.0;
    let weighted = |center: Option<(f64, f64)>| {
        let (mut sx, mut sy, mut sz) = (0.0, 0.0, 0.0);
        for &(x, y) in &region.points {
            let (xf, yf) = (x as f64, y as f64);
            let flux = src[y * width + x] as f64 - plane.at(xf, yf);
            if flux <= clip {
                continue;
            }
            if let Some((ccx, ccy)) = center
                && (xf - ccx).powi(2) + (yf - ccy).powi(2) > aperture * aperture
            {
                continue;
            }
            sx += flux * xf;
            sy += flux * yf;
            sz += flux;
        }
        (sz > 0.0).then(|| (sx / sz, sy / sz))
    };
    let mut center = weighted(None)?;
    for _ in 0..2 {
        match weighted(Some(center)) {
            Some(next) => center = next,
            None => break,
        }
    }
    Some(Candidate {
        center,
        background: plane.at(center.0, center.1),
        plane,
        total_flux: total,
        peak,
        normalized_brightness: peak - (1.0 - params.peak_response) * mean,
        star_median,
        contaminated,
    })
}

fn median_of(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// HocusFocus's `ComputeGradientContamination`: a Huber-weighted plane fit
/// to the ring (four reweightings), and whether any eighth of the ring sits
/// more than `sensitivity` standard errors above it. `None` when the ring is
/// too small or degenerate.
fn background_plane(
    ring: &[(f32, f32, f32)],
    origin: (f64, f64),
    fallback_sigma: f64,
    sensitivity: f64,
) -> Option<(Plane, bool)> {
    let n = ring.len();
    if n < 12 {
        return None;
    }
    let mut w = vec![1.0f64; n];
    let mut b = [0.0f64; 3];
    let residual = |b: &[f64; 3], i: usize| {
        let (x, y, v) = ring[i];
        v as f64 - (b[0] + b[1] * x as f64 + b[2] * y as f64)
    };
    let upper_median = |mut a: Vec<f64>| {
        a.sort_by(f64::total_cmp);
        a[a.len() >> 1]
    };
    for _ in 0..4 {
        b = solve_plane(ring, &w)?;
        let sigma = 1.4826 * upper_median((0..n).map(|i| residual(&b, i).abs()).collect());
        if sigma <= 0.0 {
            break;
        }
        for (i, wi) in w.iter_mut().enumerate() {
            let z = residual(&b, i).abs() / sigma;
            *wi = if z <= 1.345 { 1.0 } else { 1.345 / z };
        }
    }
    let plane = Plane { origin, b };
    let mut local = 1.4826 * upper_median((0..n).map(|i| residual(&b, i).abs()).collect());
    if local <= 0.0 {
        local = fallback_sigma;
    }
    if sensitivity <= 0.0 || local <= 0.0 {
        return Some((plane, false));
    }
    let mut sectors = vec![Vec::new(); 8];
    for (i, &(x, y, _)) in ring.iter().enumerate() {
        sectors[octant(x as f64, y as f64)].push(residual(&b, i));
    }
    let contaminated = sectors.into_iter().any(|mut s| {
        if s.len() < 8 {
            return false;
        }
        s.sort_by(f64::total_cmp);
        let median = s[s.len() >> 1];
        let se = 1.2533 * local / (s.len() as f64).sqrt();
        se > 0.0 && median > sensitivity * se
    });
    Some((plane, contaminated))
}

fn octant(dx: f64, dy: f64) -> usize {
    let wide = dx.abs() >= dy.abs();
    match (dx >= 0.0, dy >= 0.0) {
        (true, true) => {
            if wide {
                0
            } else {
                1
            }
        }
        (true, false) => {
            if wide {
                7
            } else {
                6
            }
        }
        (false, true) => {
            if wide {
                3
            } else {
                2
            }
        }
        (false, false) => {
            if wide {
                4
            } else {
                5
            }
        }
    }
}

/// Weighted least squares for `v ≈ b0 + b1 x + b2 y`, by Gaussian elimination
/// with partial pivoting as HocusFocus solves it.
fn solve_plane(ring: &[(f32, f32, f32)], w: &[f64]) -> Option<[f64; 3]> {
    let mut a = [[0.0f64; 3]; 3];
    let mut g = [0.0f64; 3];
    for (&(x, y, v), &wi) in ring.iter().zip(w) {
        let (x, y, v) = (x as f64, y as f64, v as f64);
        let row = [1.0, x, y];
        for r in 0..3 {
            for c in 0..3 {
                a[r][c] += wi * row[r] * row[c];
            }
            g[r] += wi * row[r] * v;
        }
    }
    for col in 0..3 {
        let pivot = (col..3).max_by(|&p, &q| a[p][col].abs().total_cmp(&a[q][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        g.swap(col, pivot);
        for r in 0..3 {
            if r == col {
                continue;
            }
            let f = a[r][col] / a[col][col];
            let pivot_row = a[col];
            for (value, &p) in a[r].iter_mut().zip(&pivot_row).skip(col) {
                *value -= f * p;
            }
            g[r] -= f * g[col];
        }
    }
    Some([g[0] / a[0][0], g[1] / a[1][1], g[2] / a[2][2]])
}

/// HocusFocus 4's `MeasureStar`: the flux-weighted mean radius of bilinear
/// samples, aligned on the centre, within a circle of half the box's smaller
/// side (partially weighting its last half pixel), counting each sample's
/// full brightness above the plane when it clears the clip.
#[allow(clippy::too_many_arguments)]
fn measure_hfr(
    src: &[f32],
    width: usize,
    height: usize,
    bounds: (usize, usize, usize, usize),
    center: (f64, f64),
    plane: &Plane,
    params: &HocusFocusParams,
    sigma: f64,
) -> Option<f64> {
    let (bx, by, bw, bh) = bounds;
    let (cx, cy) = center;
    let step = if params.pixel_sample_size > 0.0 {
        params.pixel_sample_size
    } else {
        1.0
    };
    let aperture = bw.min(bh) as f64 / 2.0;
    let clip = params.star_clipping_multiplier * sigma;
    let start_x = cx - step * ((cx - bx as f64) / step).floor();
    let start_y = cy - step * ((cy - by as f64) / step).floor();
    let (end_x, end_y) = ((bx + bw) as f64, (by + bh) as f64);
    let (mut weighted, mut total) = (0.0, 0.0);
    let mut y = start_y;
    while y <= end_y {
        let mut x = start_x;
        while x <= end_x {
            let distance = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
            if distance <= aperture + 0.5 {
                let flux = bilinear(src, width, height, x, y) - plane.at(x, y);
                if flux > clip {
                    let weight = 1.0 - (distance - (aperture - 0.5)).max(0.0);
                    weighted += weight * flux * distance;
                    total += weight * flux;
                }
            }
            x += step;
        }
        y += step;
    }
    (total > 0.0).then(|| weighted / total)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 600 × 400 field of 54 σ = 2 stars on a noisy sky, with their
    /// centres.
    fn star_field() -> (Vec<u16>, Vec<(f64, f64)>) {
        let (width, height, sigma) = (600usize, 400usize, 2.0f64);
        let mut seed = 7u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut image = (0..width * height)
            .map(|_| 1000.0 + 30.0 * (next() + next() + next() - 1.5))
            .collect::<Vec<f64>>();
        let mut centres = Vec::new();
        for row in 0..6 {
            for column in 0..9 {
                let (cx, cy) = (
                    40.0 + column as f64 * 65.0 + next(),
                    40.0 + row as f64 * 64.0 + next(),
                );
                let peak = 3000.0 + 20000.0 * next();
                for y in (cy as usize - 14)..(cy as usize + 14) {
                    for x in (cx as usize - 14)..(cx as usize + 14) {
                        let d2 = (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2);
                        image[y * width + x] += peak * (-d2 / (2.0 * sigma * sigma)).exp();
                    }
                }
                centres.push((cx, cy));
            }
        }
        let data = image.into_iter().map(|v| v.round() as u16).collect();
        (data, centres)
    }

    #[test]
    fn an_autofocus_crop_limits_detection_to_its_region() {
        let (data, centres) = star_field();
        let detect = |inner: f64, outer: f64| {
            crate::hocus_focus_star_detection::detect_stars_hocus_focus(
                &data,
                600,
                400,
                &HocusFocusParams {
                    autofocus_inner_crop_ratio: inner,
                    autofocus_outer_crop_ratio: outer,
                    ..HocusFocusParams::v4()
                },
            )
        };
        let whole = detect(1.0, 1.0);
        // An inner ratio alone: only the centred half of each side, with
        // positions still in the full frame.
        let centre = detect(0.5, 1.0);
        let inside =
            |&(x, y): &(f64, f64)| (150.0..450.0).contains(&x) && (100.0..300.0).contains(&y);
        assert!(
            centre.stars.iter().all(|s| inside(&s.position)),
            "{:?}",
            centre.stars.iter().map(|s| s.position).collect::<Vec<_>>()
        );
        // Stars whose blob reaches the crop's edge are border stars there.
        let well_inside =
            |&(x, y): &(f64, f64)| (162.0..438.0).contains(&x) && (112.0..288.0).contains(&y);
        let expected = centres.iter().filter(|c| well_inside(c)).count();
        assert!(
            centre.stars.len() + 2 >= expected && centre.stars.len() <= expected,
            "{} of {expected}",
            centre.stars.len()
        );
        assert!(centre.stars.iter().all(|s| {
            centres
                .iter()
                .any(|c| (c.0 - s.position.0).abs() < 0.5 && (c.1 - s.position.1).abs() < 0.5)
        }));
        // Outer 0.9 with inner 0.5: the ring between them.
        let ring = detect(0.5, 0.9);
        assert!(ring.stars.iter().all(|s| !inside(&s.position)));
        assert!(ring.stars.len() + centre.stars.len() <= whole.stars.len());
        assert!(ring.stars.len() > whole.stars.len() / 2);
    }

    #[test]
    fn finds_and_measures_a_star_field() {
        let (width, height, sigma) = (600usize, 400usize, 2.0f64);
        let mut seed = 7u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut image = (0..width * height)
            .map(|_| 1000.0 + 30.0 * (next() + next() + next() - 1.5))
            .collect::<Vec<f64>>();
        let mut placed = 0;
        for row in 0..6 {
            for column in 0..9 {
                let (cx, cy) = (
                    40.0 + column as f64 * 65.0 + next(),
                    40.0 + row as f64 * 64.0 + next(),
                );
                let peak = 3000.0 + 20000.0 * next();
                for y in (cy as usize - 14)..(cy as usize + 14) {
                    for x in (cx as usize - 14)..(cx as usize + 14) {
                        let d2 = (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2);
                        image[y * width + x] += peak * (-d2 / (2.0 * sigma * sigma)).exp();
                    }
                }
                placed += 1;
            }
        }
        let data = image
            .into_iter()
            .map(|v| v.round() as u16)
            .collect::<Vec<_>>();
        let found = crate::hocus_focus_star_detection::detect_stars_hocus_focus(
            &data,
            width,
            height,
            &HocusFocusParams::v4(),
        );
        assert!(
            found.stars.len() >= placed * 9 / 10,
            "{} of {placed}",
            found.stars.len()
        );
        // Above the clip, the sampled mean radius of a σ = 2 Gaussian lies a
        // little under the unclipped 1.2533σ.
        assert!(
            found.average_hfr > 1.9 && found.average_hfr < 2.6,
            "{}",
            found.average_hfr
        );
        assert!(found.stars.iter().all(|s| !s.saturated));
    }

    #[test]
    fn reflect_matches_opencv_border_reflect() {
        assert_eq!(
            (-3..8).map(|i| reflect(i, 5)).collect::<Vec<_>>(),
            vec![2, 1, 0, 0, 1, 2, 3, 4, 4, 3, 2]
        );
    }

    #[test]
    fn plane_fit_recovers_a_gradient_and_flags_a_neighbour() {
        let mut ring = Vec::new();
        for y in -8i32..=8 {
            for x in -8i32..=8 {
                if x.abs() >= 6 || y.abs() >= 6 {
                    ring.push((
                        x as f32,
                        y as f32,
                        (0.1 + 0.002 * x as f64 + 0.001 * y as f64) as f32,
                    ));
                }
            }
        }
        let (plane, contaminated) = background_plane(&ring, (0.0, 0.0), 1e-4, 5.0).unwrap();
        assert!((plane.b[1] - 0.002).abs() < 1e-6 && (plane.b[2] - 0.001).abs() < 1e-6);
        assert!(!contaminated);
        // A bright neighbour across one eighth of the ring, on a noisy sky.
        let mut seed = 3u64;
        let mut noise = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 40) as f64 / (1u64 << 24) as f64 - 0.5) * 1e-3
        };
        for r in ring.iter_mut() {
            r.2 = (0.1 + noise()) as f32
                + if r.0 > 0.0 && r.1 > 0.0 && r.0 >= r.1 {
                    0.01
                } else {
                    0.0
                };
        }
        let (_, contaminated) = background_plane(&ring, (0.0, 0.0), 1e-4, 5.0).unwrap();
        assert!(contaminated);
    }

    #[test]
    fn connected_regions_join_diagonals_and_keep_raster_order() {
        #[rustfmt::skip]
        let binary = [
            1, 0, 0, 1,
            0, 1, 0, 1,
            0, 0, 0, 0,
            1, 1, 0, 0,
        ].map(|v| v == 1);
        let regions = connected_regions(&binary, 4, 4);
        let bounds = regions.iter().map(|r| r.bounds).collect::<Vec<_>>();
        assert_eq!(bounds, vec![(0, 0, 2, 2), (3, 0, 1, 2), (0, 3, 2, 1)]);
    }

    #[test]
    fn hfr_within_the_aperture_measures_a_gaussian() {
        let (size, sigma) = (41usize, 2.0f64);
        let src = (0..size * size)
            .map(|i| {
                let (x, y) = ((i % size) as f64 - 20.0, (i / size) as f64 - 20.0);
                (0.01 + 0.3 * (-(x * x + y * y) / (2.0 * sigma * sigma)).exp()) as f32
            })
            .collect::<Vec<_>>();
        let plane = Plane {
            origin: (20.0, 20.0),
            b: [0.01, 0.0, 0.0],
        };
        let params = HocusFocusParams::v4();
        let hfr = measure_hfr(
            &src,
            size,
            size,
            (12, 12, 17, 17),
            (20.0, 20.0),
            &plane,
            &params,
            1e-5,
        )
        .unwrap();
        // The brightness-weighted mean radius of a Gaussian is 1.2533σ.
        assert!((hfr - 1.2533 * sigma).abs() < 0.1, "{hfr}");
    }
}
