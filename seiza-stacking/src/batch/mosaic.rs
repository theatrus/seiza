//! Match panel backgrounds without fitting away their shared sky.
//!
//! Panel means first share an additive plane, fitted simultaneously through
//! their overlaps. Only dim cells constrain the planes. The remaining smooth
//! differences taper across overlaps using distance to each panel's footprint;
//! the correction is additive, so full-resolution star detail and noise stay
//! in the rejection pass. No sky is inferred in uncovered pixels.
//!
//! The separation of overlap matching and background blending follows the
//! approach described by Mega Merge Mosaic (Apache-2.0),
//! https://github.com/Astrometrical/mega-merge-mosaic/blob/main/docs/DESIGN.md.
//! This implementation uses Seiza's registered samples and integration paths.
use super::{BandSource, BatchStackOptions, check_cancelled};
use crate::{Error, LinearImage, Result};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use std::collections::{BTreeMap, VecDeque};

const TERMS: usize = 3;
const SIDE: usize = 8;
const POINTS: usize = SIDE * SIDE;
const SAMPLE_BYTES: usize = 128 * 1024 * 1024;

#[derive(Clone, Copy, Default)]
struct Sample {
    sum: f64,
    weight: f64,
}
impl Sample {
    fn mean(self) -> Option<f64> {
        (self.weight > 0.0).then(|| self.sum / self.weight)
    }
}

pub(super) struct Samples {
    shape: (usize, usize, usize),
    cell: usize,
    columns: usize,
    rows: usize,
    frame_panels: Vec<usize>,
    samples: Vec<Vec<Sample>>,
}

impl Samples {
    pub(super) fn new(image: &LinearImage, panels: &[u32]) -> Self {
        let mut ids = panels.to_vec();
        ids.sort_unstable();
        ids.dedup();
        let ids: BTreeMap<_, _> = ids.into_iter().enumerate().map(|(i, id)| (id, i)).collect();
        // Keep corresponding sample positions until panels can be compared.
        // Increase spacing with panel/channel count to bound aggregate storage,
        // rather than retaining a decoded image for every panel.
        let cells = SAMPLE_BYTES / (ids.len() * image.channels * POINTS * size_of::<Sample>());
        let axis = (cells as f64).sqrt().floor().clamp(1.0, 128.0) as usize;
        let cell = 16.max(image.width.max(image.height).div_ceil(axis));
        let (columns, rows) = (image.width.div_ceil(cell), image.height.div_ceil(cell));
        Self {
            shape: (image.width, image.height, image.channels),
            cell,
            columns,
            rows,
            frame_panels: panels.iter().map(|id| ids[id]).collect(),
            samples: vec![
                vec![Sample::default(); columns * rows * POINTS * image.channels];
                ids.len()
            ],
        }
    }

    fn position(&self, tile: usize, point: usize) -> (usize, usize) {
        position(self.shape, self.cell, self.columns, tile, point)
    }

    pub(super) fn add(
        &mut self,
        index: usize,
        image: &LinearImage,
        options: &BatchStackOptions,
    ) -> Result<()> {
        super::validate_image(image, Some(self.shape))?;
        let channels = image.channels;
        if let Some(weights) = &options.frame_weights
            && weights[index].len() != channels
        {
            return Err(Error::Stack(
                "mosaic frame weights do not match its channels".into(),
            ));
        }
        let panel = self.frame_panels[index];
        let (shape, cell, columns) = (self.shape, self.cell, self.columns);
        self.samples[panel]
            .par_chunks_mut(POINTS * channels)
            .enumerate()
            .for_each(|(tile, samples)| {
                for point in 0..POINTS {
                    let (x, y) = position(shape, cell, columns, tile, point);
                    for channel in 0..channels {
                        let value = image.data[(y * image.width + x) * channels + channel];
                        if !value.is_finite() {
                            continue;
                        }
                        let weight = options
                            .frame_weights
                            .as_ref()
                            .map_or(1.0, |w| f64::from(w[index][channel]));
                        let sample = &mut samples[point * channels + channel];
                        sample.sum += f64::from(value) * weight;
                        sample.weight += weight;
                    }
                }
            });
        Ok(())
    }

    fn at(&self, panel: usize, tile: usize, point: usize, channel: usize) -> Sample {
        self.samples[panel][(tile * POINTS + point) * self.shape.2 + channel]
    }

    // Compare the same positions before any clipping or averaging. Shared
    // stars and emission then cancel even at partially covered panel edges.
    fn pair(&self, a: usize, b: usize, tile: usize, channel: usize) -> Option<([f64; TERMS], f64)> {
        let mut pairs = Vec::with_capacity(POINTS);
        for point in 0..POINTS {
            if let (Some(va), Some(vb)) = (
                self.at(a, tile, point, channel).mean(),
                self.at(b, tile, point, channel).mean(),
            ) {
                pairs.push((point, vb - va));
            }
        }
        if pairs.is_empty() {
            return None;
        }
        let mut differences = pairs.iter().map(|p| p.1).collect::<Vec<_>>();
        let (center, limit) = clipping_range(&mut differences);
        let (mut phi, mut sum, mut count) = ([0.0; TERMS], 0.0, 0.0);
        for (point, diff) in pairs {
            if (diff - center).abs() > limit {
                continue;
            }
            let (x, y) = self.position(tile, point);
            let basis = [
                1.0,
                x as f64 / self.shape.0 as f64 - 0.5,
                y as f64 / self.shape.1 as f64 - 0.5,
            ];
            for term in 0..TERMS {
                phi[term] += basis[term];
            }
            sum += diff;
            count += 1.0;
        }
        for value in &mut phi {
            *value /= count;
        }
        Some((phi, sum / count))
    }

    pub(super) fn finish(self, options: &BatchStackOptions) -> Result<Corrections> {
        let channels = self.shape.2;
        let means = self
            .samples
            .iter()
            .map(|samples| {
                samples
                    .chunks(POINTS * channels)
                    .flat_map(|cell| {
                        (0..channels)
                            .map(|c| {
                                let (sum, count) = cell
                                    .iter()
                                    .skip(c)
                                    .step_by(channels)
                                    .filter_map(|s| s.mean())
                                    .fold((0.0, 0), |(sum, n), v| (sum + v, n + 1));
                                if count == 0 {
                                    f32::NAN
                                } else {
                                    (sum / f64::from(count)) as f32
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut planes = vec![vec![[0.0; TERMS]; channels]; means.len()];
        for channel in 0..channels {
            check_cancelled(options)?;
            self.fit(&means, channel, &mut planes, options)?;
        }
        let distances = means
            .iter()
            .map(|mean| {
                (0..channels)
                    .map(|c| distances(mean, self.columns, self.rows, channels, c))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut residuals = vec![vec![f32::NAN; self.columns * self.rows * channels]; means.len()];
        for tile in 0..self.columns * self.rows {
            if tile % self.columns == 0 {
                check_cancelled(options)?;
            }
            for channel in 0..channels {
                let mut differences = (0..means.len())
                    .map(|_| Vec::with_capacity(POINTS))
                    .collect::<Vec<_>>();
                for point in 0..POINTS {
                    let (x, y) = self.position(tile, point);
                    let phi = [
                        1.0,
                        x as f64 / self.shape.0 as f64 - 0.5,
                        y as f64 / self.shape.1 as f64 - 0.5,
                    ];
                    let mut total = 0.0;
                    let mut weight = 0.0;
                    for panel in 0..means.len() {
                        let sample = self.at(panel, tile, point, channel);
                        if let Some(value) = sample.mean() {
                            let w = (distances[panel][channel][tile] / 8.0).min(1.0).powi(2)
                                * sample.weight;
                            total += w * (value + dot(&planes[panel][channel], &phi));
                            weight += w;
                        }
                    }
                    if weight == 0.0 {
                        continue;
                    }
                    for panel in 0..means.len() {
                        if let Some(value) = self.at(panel, tile, point, channel).mean() {
                            differences[panel]
                                .push(total / weight - value - dot(&planes[panel][channel], &phi));
                        }
                    }
                }
                for panel in 0..means.len() {
                    if !differences[panel].is_empty() {
                        // Clip differences, never panel brightness: reject a
                        // sampled cosmic ray without clipping shared stars.
                        residuals[panel][tile * channels + channel] =
                            clipped_mean(&mut differences[panel]) as f32;
                    }
                }
            }
        }
        for residual in &mut residuals {
            extend(residual, self.columns, self.rows, channels);
        }
        Ok(Corrections {
            shape: self.shape,
            cell: self.cell,
            columns: self.columns,
            rows: self.rows,
            frame_panels: self.frame_panels,
            planes,
            residuals,
        })
    }

    fn fit(
        &self,
        means: &[Vec<f32>],
        channel: usize,
        planes: &mut [Vec<[f64; TERMS]>],
        options: &BatchStackOptions,
    ) -> Result<()> {
        let channels = self.shape.2;
        let thresholds = means
            .iter()
            .map(|mean| {
                let mut values = mean
                    .iter()
                    .skip(channel)
                    .step_by(channels)
                    .copied()
                    .filter(|v| v.is_finite())
                    .collect::<Vec<_>>();
                if values.is_empty() {
                    return f32::NEG_INFINITY;
                }
                let center = median(&mut values);
                for value in &mut values {
                    *value = (*value - center).abs();
                }
                center + 3.0 * median(&mut values)
            })
            .collect::<Vec<_>>();
        let mut edges = vec![];
        for a in 0..means.len() {
            check_cancelled(options)?;
            for b in a + 1..means.len() {
                let mut cells = vec![];
                for tile in 0..self.columns * self.rows {
                    let (va, vb) = (
                        means[a][tile * channels + channel],
                        means[b][tile * channels + channel],
                    );
                    if va.is_finite()
                        && vb.is_finite()
                        && va <= thresholds[a]
                        && vb <= thresholds[b]
                        && let Some((phi, diff)) = self.pair(a, b, tile, channel)
                    {
                        cells.push((phi, diff, true));
                    }
                }
                if cells.len() >= 12 {
                    edges.push((a, b, cells));
                }
            }
        }
        let n = means.len() * TERMS;
        let iterations = if edges.is_empty() { 0 } else { 4 };
        for _ in 0..iterations {
            check_cancelled(options)?;
            let mut matrix = DMatrix::<f64>::zeros(n, n);
            let mut right = DVector::<f64>::zeros(n);
            for (a, b, cells) in &edges {
                let weight = 1.0 / cells.iter().filter(|c| c.2).count().max(1) as f64;
                for &(phi, diff, keep) in cells {
                    if keep {
                        for (p, sign_p) in [(*a, 1.0), (*b, -1.0)] {
                            for i in 0..TERMS {
                                right[p * TERMS + i] += weight * sign_p * phi[i] * diff;
                                for (q, sign_q) in [(*a, 1.0), (*b, -1.0)] {
                                    for j in 0..TERMS {
                                        matrix[(p * TERMS + i, q * TERMS + j)] +=
                                            weight * sign_p * sign_q * phi[i] * phi[j];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Penalize correction energy over each panel's actual footprint.
            // This fixes the common-plane gauge and bounds extrapolation from
            // thin overlaps without assuming the sky itself is flat.
            for (panel, mean) in means.iter().enumerate() {
                let count = mean
                    .iter()
                    .skip(channel)
                    .step_by(channels)
                    .filter(|v| v.is_finite())
                    .count()
                    .max(1);
                for tile in 0..self.columns * self.rows {
                    if mean[tile * channels + channel].is_finite() {
                        let phi = basis(self.shape, self.cell, self.columns, tile);
                        for i in 0..TERMS {
                            for j in 0..TERMS {
                                matrix[(panel * TERMS + i, panel * TERMS + j)] +=
                                    1.0e-3 * phi[i] * phi[j] / count as f64;
                            }
                        }
                    }
                }
            }
            for i in 0..n {
                matrix[(i, i)] += 1.0e-10;
            }
            let solution = matrix
                .cholesky()
                .ok_or_else(|| Error::Stack("mosaic background fit is singular".into()))?
                .solve(&right);
            for panel in 0..means.len() {
                for term in 0..TERMS {
                    planes[panel][channel][term] = solution[panel * TERMS + term];
                }
            }
            for (a, b, cells) in &mut edges {
                let mut residuals = cells
                    .iter()
                    .map(|&(phi, diff, _)| {
                        (dot(&planes[*a][channel], &phi) - dot(&planes[*b][channel], &phi) - diff)
                            as f32
                    })
                    .collect::<Vec<_>>();
                let center = median(&mut residuals);
                for v in &mut residuals {
                    *v = (*v - center).abs();
                }
                let limit = (3.0 * 1.4826 * median(&mut residuals)).max(1.0e-4);
                for (phi, diff, keep) in cells {
                    let residual =
                        dot(&planes[*a][channel], phi) - dot(&planes[*b][channel], phi) - *diff;
                    *keep = (residual - f64::from(center)).abs() <= f64::from(limit);
                }
            }
        }
        // Do not let a signal-only overlap contradict gradients already
        // measured through a chain of dim overlaps. Match only disconnected
        // background groups, by a constant after their planes are fitted.
        let mut groups = (0..means.len()).collect::<Vec<_>>();
        for (a, b, cells) in &edges {
            if !cells.iter().any(|c| c.2) {
                continue;
            }
            let (from, to) = (groups[*b], groups[*a]);
            for group in &mut groups {
                if *group == from {
                    *group = to;
                }
            }
        }
        self.match_levels(means, channel, planes, &groups, options)
    }

    fn match_levels(
        &self,
        means: &[Vec<f32>],
        channel: usize,
        planes: &mut [Vec<[f64; TERMS]>],
        groups: &[usize],
        options: &BatchStackOptions,
    ) -> Result<()> {
        let mut ids = groups.to_vec();
        ids.sort_unstable();
        ids.dedup();
        if ids.len() < 2 {
            return Ok(());
        }
        let groups = groups
            .iter()
            .map(|g| ids.binary_search(g).unwrap())
            .collect::<Vec<_>>();
        let n = ids.len();
        let mut matrix = DMatrix::<f64>::zeros(n, n);
        let mut right = DVector::<f64>::zeros(n);
        for a in 0..means.len() {
            check_cancelled(options)?;
            for b in a + 1..means.len() {
                let (ga, gb) = (groups[a], groups[b]);
                if ga == gb {
                    continue;
                }
                let mut residuals = vec![];
                for tile in 0..self.columns * self.rows {
                    if let Some((phi, diff)) = self.pair(a, b, tile, channel) {
                        residuals.push(
                            (diff + dot(&planes[b][channel], &phi) - dot(&planes[a][channel], &phi))
                                as f32,
                        );
                    }
                }
                if residuals.is_empty() {
                    continue;
                }
                // Shared celestial structure cancels in paired residuals.
                // There is no spatial fit to bright emission.
                let delta = f64::from(median(&mut residuals));
                matrix[(ga, ga)] += 1.0;
                matrix[(gb, gb)] += 1.0;
                matrix[(ga, gb)] -= 1.0;
                matrix[(gb, ga)] -= 1.0;
                right[ga] += delta;
                right[gb] -= delta;
            }
        }
        for &group in &groups {
            matrix[(group, group)] += 1.0e-6;
        }
        let levels = matrix
            .cholesky()
            .ok_or_else(|| Error::Stack("mosaic background levels are singular".into()))?
            .solve(&right);
        for (panel, &group) in groups.iter().enumerate() {
            planes[panel][channel][0] += levels[group];
        }
        Ok(())
    }
}

fn median(values: &mut [f32]) -> f32 {
    let index = values.len() / 2;
    *values.select_nth_unstable_by(index, f32::total_cmp).1
}

fn median_f64(values: &mut [f64]) -> f64 {
    let index = values.len() / 2;
    *values.select_nth_unstable_by(index, f64::total_cmp).1
}

fn clipping_range(values: &mut [f64]) -> (f64, f64) {
    let center = median_f64(values);
    let mut deviations = values
        .iter()
        .map(|v| (v - center).abs())
        .collect::<Vec<_>>();
    (
        center,
        (4.0 * 1.4826 * median_f64(&mut deviations)).max(1.0e-4),
    )
}

fn clipped_mean(values: &mut [f64]) -> f64 {
    let (center, limit) = clipping_range(values);
    let (sum, count) = values
        .iter()
        .filter(|v| (**v - center).abs() <= limit)
        .fold((0.0, 0), |(sum, count), &value| (sum + value, count + 1));
    sum / f64::from(count)
}

fn position(
    shape: (usize, usize, usize),
    cell: usize,
    columns: usize,
    tile: usize,
    point: usize,
) -> (usize, usize) {
    let (x, y) = ((tile % columns) * cell, (tile / columns) * cell);
    let (width, height) = (cell.min(shape.0 - x), cell.min(shape.1 - y));
    // Adjacent pairs avoid sampling only one parity of an alternating
    // channel mask near an edge. Pair centres still span the complete cell.
    let offset = |index: usize, length: usize| {
        if length == 1 {
            0
        } else {
            (index / 2 * (length - 2) + 1) / (SIDE / 2 - 1) + index % 2
        }
    };
    (
        x + offset(point % SIDE, width),
        y + offset(point / SIDE, height),
    )
}

fn basis(shape: (usize, usize, usize), cell: usize, columns: usize, tile: usize) -> [f64; TERMS] {
    [
        1.0,
        (((tile % columns) as f64 + 0.5) * cell as f64 - 0.5) / shape.0 as f64 - 0.5,
        (((tile / columns) as f64 + 0.5) * cell as f64 - 0.5) / shape.1 as f64 - 0.5,
    ]
}
fn dot(a: &[f64; TERMS], b: &[f64; TERMS]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn distances(
    values: &[f32],
    columns: usize,
    rows: usize,
    channels: usize,
    channel: usize,
) -> Vec<f64> {
    let mut d = (0..columns * rows)
        .map(|i| {
            if values[i * channels + channel].is_finite() {
                1.0e9_f64
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    for y in 0..rows {
        for x in 0..columns {
            let i = y * columns + x;
            d[i] = d[i]
                .min(if x == 0 { 1.0 } else { d[i - 1] + 1.0 })
                .min(if y == 0 { 1.0 } else { d[i - columns] + 1.0 });
        }
    }
    for y in (0..rows).rev() {
        for x in (0..columns).rev() {
            let i = y * columns + x;
            d[i] = d[i]
                .min(if x + 1 == columns {
                    1.0
                } else {
                    d[i + 1] + 1.0
                })
                .min(if y + 1 == rows {
                    1.0
                } else {
                    d[i + columns] + 1.0
                });
        }
    }
    d
}

// Extend residuals through uncovered cells for interpolation at the footprint's
// rim. Only the correction extends; missing image samples remain missing.
fn extend(values: &mut [f32], columns: usize, rows: usize, channels: usize) {
    for channel in 0..channels {
        let mut queue = VecDeque::new();
        for tile in 0..columns * rows {
            if values[tile * channels + channel].is_finite() {
                queue.push_back(tile);
            }
        }
        if queue.is_empty() {
            for value in values.iter_mut().skip(channel).step_by(channels) {
                *value = 0.0;
            }
            continue;
        }
        while let Some(tile) = queue.pop_front() {
            let (x, y) = (tile % columns, tile / columns);
            for next in [
                x.checked_sub(1).map(|_| tile - 1),
                (x + 1 < columns).then_some(tile + 1),
                y.checked_sub(1).map(|_| tile - columns),
                (y + 1 < rows).then_some(tile + columns),
            ]
            .into_iter()
            .flatten()
            {
                if !values[next * channels + channel].is_finite() {
                    values[next * channels + channel] = values[tile * channels + channel];
                    queue.push_back(next);
                }
            }
        }
    }
}

pub(super) struct Corrections {
    shape: (usize, usize, usize),
    cell: usize,
    columns: usize,
    rows: usize,
    frame_panels: Vec<usize>,
    planes: Vec<Vec<[f64; TERMS]>>,
    residuals: Vec<Vec<f32>>,
}
impl Corrections {
    pub(super) fn apply(&self, index: usize, top: usize, data: &mut [f32]) {
        self.apply_at(index, top * self.shape.0 * self.shape.2, data);
    }

    fn apply_at(&self, index: usize, first: usize, data: &mut [f32]) {
        let (width, height, channels) = self.shape;
        let panel = self.frame_panels[index];
        const RUN: usize = 4096;
        data.par_chunks_mut(RUN)
            .enumerate()
            .for_each(|(run, values)| {
                for (sample, value) in values.iter_mut().enumerate() {
                    if !value.is_finite() {
                        continue;
                    }
                    let sample = first + run * RUN + sample;
                    let (pixel, c) = (sample / channels, sample % channels);
                    let (x, y) = (pixel % width, pixel / width);
                    let phi = [
                        1.0,
                        x as f64 / width as f64 - 0.5,
                        y as f64 / height as f64 - 0.5,
                    ];
                    let gx = ((x as f64 + 0.5) / self.cell as f64 - 0.5)
                        .clamp(0.0, (self.columns - 1) as f64);
                    let gy = ((y as f64 + 0.5) / self.cell as f64 - 0.5)
                        .clamp(0.0, (self.rows - 1) as f64);
                    let (xx, yy) = (gx as usize, gy as usize);
                    let (fx, fy) = (gx - xx as f64, gy - yy as f64);
                    let at = |x: usize, y: usize| {
                        f64::from(self.residuals[panel][(y * self.columns + x) * channels + c])
                    };
                    let a = at(xx, yy) * (1.0 - fx) + at((xx + 1).min(self.columns - 1), yy) * fx;
                    let b = at(xx, (yy + 1).min(self.rows - 1)) * (1.0 - fx)
                        + at((xx + 1).min(self.columns - 1), (yy + 1).min(self.rows - 1)) * fx;
                    *value = (f64::from(*value)
                        + dot(&self.planes[panel][c], &phi)
                        + a * (1.0 - fy)
                        + b * fy) as f32;
                }
            });
    }
}

pub(super) struct Corrected<'a, T> {
    pub(super) source: &'a T,
    pub(super) corrections: Option<&'a Corrections>,
}
impl<T: BandSource> BandSource for Corrected<'_, T> {
    fn read(&self, index: usize, top: usize, band: &mut [f32]) -> Result<()> {
        self.source.read(index, top, band)?;
        if let Some(maps) = self.corrections {
            maps.apply(index, top, band);
        }
        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn read_into(
        &self,
        index: usize,
        top: usize,
        samples: usize,
        write: &mut dyn FnMut(usize, &[f32]),
    ) -> Result<()> {
        let Some(maps) = self.corrections else {
            return self.source.read_into(index, top, samples, write);
        };
        // Preserve the source's bounded decoded runs, including those that
        // start partway through a colour pixel or row. No extra input band.
        let mut corrected = Vec::new();
        self.source
            .read_into(index, top, samples, &mut |offset, values| {
                corrected.clear();
                corrected.extend_from_slice(values);
                maps.apply_at(
                    index,
                    top * maps.shape.0 * maps.shape.2 + offset,
                    &mut corrected,
                );
                write(offset, &corrected);
            })
    }

    fn region(&self, index: usize) -> Option<crate::ReferenceRegion> {
        self.source.region(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BatchStackOptions, GpuOptions, integrate_registered_frames,
        integrate_registered_frames_banded,
    };

    fn sky(x: usize, y: usize, channel: usize) -> f32 {
        let (xx, yy) = (x as f32 / 240.0, y as f32 / 224.0);
        50.0 * (channel + 1) as f32
            + 180.0 * (-((xx - 0.47).powi(2) + (yy - 0.51).powi(2)) / 0.019).exp()
            + if (x, y) == (130, 110) { 3000.0 } else { 0.0 }
    }

    fn panels(channels: usize, gradients: bool) -> Vec<LinearImage> {
        (0..8)
            .map(|frame| {
                let panel = frame / 2;
                let (left, top) = ((panel % 2) * 80, (panel / 2) * 80);
                let data = (0..240 * 224 * channels)
                    .map(|sample| {
                        let (pixel, c) = (sample / channels, sample % channels);
                        let (x, y) = (pixel % 240, pixel / 240);
                        if x < left
                            || x >= left + 160
                            || y < top
                            || y >= top + 144
                            || (x, y) == (117, 107)
                        {
                            return f32::NAN;
                        }
                        let background = if gradients {
                            [0.0, 27.0, -13.0, 6.0][panel]
                                + [0.0, -17.0, 12.0, 8.0][panel] * x as f32 / 240.0
                                + [0.0, 22.0, -14.0, 31.0][panel] * y as f32 / 224.0
                        } else {
                            0.0
                        };
                        sky(x, y, c) + background * (c + 1) as f32
                    })
                    .collect();
                LinearImage::new(240, 224, channels, data).unwrap()
            })
            .collect()
    }

    // A global additive plane is unobservable through overlap differences.
    // Remove only that gauge when measuring recovery of the known sky.
    fn error_after_common_plane(image: &LinearImage, channel: usize) -> f64 {
        let mut matrix = nalgebra::Matrix3::<f64>::zeros();
        let mut right = nalgebra::Vector3::<f64>::zeros();
        let mut samples = vec![];
        for y in 0..image.height {
            for x in 0..image.width {
                let value = image.data[(y * image.width + x) * image.channels + channel];
                if !value.is_finite() {
                    continue;
                }
                let phi = nalgebra::Vector3::new(
                    1.0,
                    x as f64 / image.width as f64,
                    y as f64 / image.height as f64,
                );
                let diff = f64::from(value - sky(x, y, channel));
                matrix += phi * phi.transpose();
                right += phi * diff;
                samples.push((phi, diff));
            }
        }
        let fit = matrix.cholesky().unwrap().solve(&right);
        (samples
            .iter()
            .map(|(phi, diff)| (diff - phi.dot(&fit)).powi(2))
            .sum::<f64>()
            / samples.len() as f64)
            .sqrt()
    }

    #[test]
    fn mismatched_panel_skies_recover_shared_nebula_and_star_contrast() {
        let frames = panels(3, true);
        let options = BatchStackOptions {
            mosaic_panels: Some((0..8).map(|i| i / 2).collect()),
            frame_weights: Some((0..8).map(|i| vec![1.0 + i as f32, 2.0, 0.5]).collect()),
            gpu: GpuOptions {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let whole = integrate_registered_frames(8, &options, |_, i| Ok(frames[i].clone())).unwrap();
        let uncorrected = integrate_registered_frames(
            8,
            &BatchStackOptions {
                mosaic_panels: None,
                ..options.clone()
            },
            |_, i| Ok(frames[i].clone()),
        )
        .unwrap();
        for c in 0..3 {
            let corrected = error_after_common_plane(&whole.snapshot.image, c);
            let original = error_after_common_plane(&uncorrected.snapshot.image, c);
            assert!(
                corrected < 0.35 && corrected < original / 10.0,
                "channel {c}: {original} -> {corrected}"
            );
            let star = whole.snapshot.image.data[(110 * 240 + 130) * 3 + c];
            let adjacent = whole.snapshot.image.data[(110 * 240 + 129) * 3 + c];
            assert!(
                (star - adjacent - sky(130, 110, c) + sky(129, 110, c)).abs() < 1.0,
                "channel {c}: the 3000 ADU star lost contrast"
            );
        }
        let hole = (107 * 240 + 117) * 3;
        assert!(
            whole.snapshot.image.data[hole..hole + 3]
                .iter()
                .all(|v| v.is_nan())
        );
        assert_eq!(&whole.snapshot.coverage[hole..hole + 3], &[0, 0, 0]);
        let scratch = tempfile::tempdir().unwrap();
        let options = BatchStackOptions {
            scratch_directory: Some(scratch.path().to_path_buf()),
            band_memory_bytes: 32 * 1024,
            ..options
        };
        let banded = integrate_registered_frames_banded(
            8,
            &options,
            |i| Ok(frames[i].clone()),
            |_, _, _| {},
        )
        .unwrap();
        assert_eq!(whole.snapshot.coverage, banded.snapshot.coverage);
        assert!(
            whole
                .snapshot
                .image
                .data
                .iter()
                .zip(&banded.snapshot.image.data)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert_eq!(std::fs::read_dir(scratch.path()).unwrap().count(), 0);
    }

    #[test]
    fn an_agreed_sky_keeps_its_diffuse_structure_and_missing_samples() {
        let frames = panels(1, false);
        let options = BatchStackOptions {
            mosaic_panels: Some((0..8).map(|i| i / 2).collect()),
            gpu: GpuOptions {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let result =
            integrate_registered_frames(8, &options, |_, i| Ok(frames[i].clone())).unwrap();
        for y in 0..224 {
            for x in 0..240 {
                let value = result.snapshot.image.data[y * 240 + x];
                if (x, y) == (117, 107) {
                    assert!(value.is_nan());
                } else {
                    assert!((value - sky(x, y, 0)).abs() < 0.001, "({x},{y}): {value}");
                }
            }
        }
    }

    #[test]
    fn zero_and_negative_samples_are_data_and_sources_cannot_change() {
        let frames = (0..4)
            .map(|i| {
                let mut image =
                    LinearImage::new(64, 64, 1, vec![-5.0 + 18.0 * (i / 2) as f32; 64 * 64])
                        .unwrap();
                image.data[20 * 64 + 20] = 18.0 * (i / 2) as f32;
                image.data[10 * 64 + 10] = f32::NAN;
                image
            })
            .collect::<Vec<_>>();
        let options = BatchStackOptions {
            mosaic_panels: Some(vec![0, 0, 1, 1]),
            gpu: GpuOptions {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let result =
            integrate_registered_frames(4, &options, |_, i| Ok(frames[i].clone())).unwrap();
        assert_eq!(result.snapshot.coverage[20 * 64 + 20], 4);
        assert_eq!(result.snapshot.coverage[10 * 64 + 10], 0);
        assert!(
            (result.snapshot.image.data[20 * 64 + 20]
                - result.snapshot.image.data[20 * 64 + 19]
                - 5.0)
                .abs()
                < 0.001
        );
        let mut calls = 0;
        let error = integrate_registered_frames(4, &options, |_, i| {
            calls += 1;
            let mut image = frames[i].clone();
            if calls > 4 {
                image.data[0] += 1.0;
            }
            Ok(image)
        })
        .unwrap_err();
        assert!(error.to_string().contains("changed between batch passes"));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn streamed_colour_corrections_match_bands_at_arbitrary_sample_offsets() {
        struct Source {
            image: LinearImage,
        }
        impl BandSource for Source {
            fn read(&self, _: usize, top: usize, band: &mut [f32]) -> Result<()> {
                let first = top * self.image.width * self.image.channels;
                band.copy_from_slice(&self.image.data[first..first + band.len()]);
                Ok(())
            }
            fn read_into(
                &self,
                _: usize,
                top: usize,
                samples: usize,
                write: &mut dyn FnMut(usize, &[f32]),
            ) -> Result<()> {
                let first = top * self.image.width * self.image.channels;
                for (run, values) in self.image.data[first..first + samples]
                    .chunks(1001)
                    .enumerate()
                {
                    write(run * 1001, values);
                }
                Ok(())
            }
        }
        let images = panels(3, true);
        let options = BatchStackOptions::default();
        let mut samples = Samples::new(&images[0], &[0, 0, 1, 1, 2, 2, 3, 3]);
        for (i, image) in images.iter().enumerate() {
            samples.add(i, image, &options).unwrap();
        }
        let maps = samples.finish(&options).unwrap();
        let source = Source {
            image: images[2].clone(),
        };
        let corrected = Corrected {
            source: &source,
            corrections: Some(&maps),
        };
        let top = 67;
        let mut band = vec![0.0; 240 * 3 * 59];
        corrected.read(2, top, &mut band).unwrap();
        let mut streamed = vec![0.0; band.len()];
        corrected
            .read_into(2, top, band.len(), &mut |offset, values| {
                streamed[offset..offset + values.len()].copy_from_slice(values)
            })
            .unwrap();
        assert!(
            band.iter()
                .zip(streamed)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }

    #[test]
    fn paired_differences_cancel_structure_with_partial_cells_and_channel_holes() {
        let images = (0..2)
            .map(|panel| {
                let data = (0..25 * 19 * 3)
                    .map(|sample| {
                        let (pixel, c) = (sample / 3, sample % 3);
                        let (x, y) = (pixel % 25, pixel / 25);
                        if (c == 0 && x % 2 == panel) || (c == 1 && (x + y) % 3 == panel) {
                            return f32::NAN;
                        }
                        100.0 + (x * x + y * 7) as f32 + 18.0 * panel as f32
                    })
                    .collect();
                LinearImage::new(25, 19, 3, data).unwrap()
            })
            .collect::<Vec<_>>();
        let mut samples = Samples::new(&images[0], &[0, 1]);
        for (i, image) in images.iter().enumerate() {
            samples
                .add(i, image, &BatchStackOptions::default())
                .unwrap();
        }
        for tile in 0..samples.columns * samples.rows {
            // R has disjoint support between panels: never compare different
            // skies simply because their values occupy the same coarse cell.
            assert!(samples.pair(0, 1, tile, 0).is_none());
            for c in 1..3 {
                let (_, difference) = samples.pair(0, 1, tile, c).unwrap();
                assert!((difference - 18.0).abs() < 1.0e-9);
            }
        }
    }

    #[test]
    fn paired_clipping_keeps_fractional_weighted_offsets_at_large_levels() {
        let images = [100.0, 101.0, 1_000_100.0, 1_000_100.0]
            .map(|value| LinearImage::new(16, 16, 1, vec![value; 256]).unwrap());
        let options = BatchStackOptions {
            frame_weights: Some(vec![vec![1.0], vec![2.0], vec![1.0], vec![2.0]]),
            ..Default::default()
        };
        let mut samples = Samples::new(&images[0], &[0, 0, 1, 1]);
        for (i, image) in images.iter().enumerate() {
            samples.add(i, image, &options).unwrap();
        }
        let (phi, difference) = samples.pair(0, 1, 0, 0).unwrap();
        assert!(phi.iter().all(|v| v.is_finite()));
        assert!((difference - (1_000_100.0 - 302.0 / 3.0)).abs() < 1.0e-8);
    }

    #[test]
    fn panel_ids_are_validated_before_loading() {
        for ids in [vec![0], (0..257).collect()] {
            let options = BatchStackOptions {
                mosaic_panels: Some(ids),
                ..Default::default()
            };
            let result = integrate_registered_frames(257, &options, |_, _| {
                panic!("invalid panel IDs must not load frames")
            });
            assert!(result.is_err());
        }
    }
}
