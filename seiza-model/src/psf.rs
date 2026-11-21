//! What a sharpening model is told about an image, made from the image alone:
//! its PSF field at a grid of nodes, each node's stamp and width, its point
//! sources, and the maps made from them. These follow the harness's
//! functions of the same names (`fitted_stamp`, `stamp_width`, `ratio_map`,
//! `detections`, `field_point_map`, `sampled_features`), because a model is
//! trained on exactly what they make.

#[cfg(test)]
use crate::bundle::StructureRemoval;
use crate::bundle::{PointsReading, PsfReading, StampEncoder, StampPlane};
use crate::error::ModelError;
use crate::stats::{median_f32, median_f64};
use rayon::prelude::*;
use seiza_stars::{MeasurementOptions, PsfField, PsfFieldError, SparseFieldPolicy};

/// The widths a stamp's Gaussian is sought over, in pixels of σ, and how
/// many: evenly in their logarithm.
const WIDTH_SIGMAS: (f64, f64, usize) = (0.25, 30.0, 300);
/// A Gaussian's FWHM over its σ, as the harness rounds it.
const FWHM_PER_SIGMA: f64 = 2.3548;
/// The detector reads 16-bit counts: the image from the 0.01st percentile of
/// every seventh pixel each way, at 0, to its largest value, at 65535.
const COUNT_STRIDE: usize = 7;
const COUNT_LOW_PERCENTILE: f64 = 0.01;
const COUNT_MOST: f64 = 65535.0;
/// A group of clipped peaks: three or more within 2% of one another, at 90%
/// of the brightest or above. Stars clip from 80% of their median.
const CLIP_CLUSTER: usize = 3;
const CLIP_WITHIN: f64 = 0.98;
const CLIP_BRIGHTEST: f64 = 0.9;
const CLIP_LEVEL: f64 = 0.8;

/// An image's PSF at a grid of nodes: both edges of each axis, at most the
/// card's `grid_px` apart.
pub(crate) struct Field {
    /// The nodes' rows, in pixels.
    pub(crate) ys: Vec<f64>,
    /// The nodes' columns, in pixels.
    pub(crate) xs: Vec<f64>,
    /// Each node's stamp at the encoder's side, row by row of nodes.
    pub(crate) stamps: Vec<Vec<f32>>,
    /// Each node's width in pixels ([`stamp_width`]), NaN where it has none.
    pub(crate) widths: Vec<f64>,
    /// The stamps' side.
    pub(crate) side: usize,
    /// How many stars the field was measured from.
    pub(crate) stars: usize,
    /// Whether too few stars left one PSF for the whole image.
    pub(crate) global: bool,
}

/// Node positions along an axis of `length` pixels, as the harness places
/// them: evenly from 0 to `length - 1`, at most `grid_px` apart.
pub(crate) fn grid_nodes(length: usize, grid_px: usize) -> Vec<f64> {
    let last = length.saturating_sub(1) as f64;
    let count = ((last / grid_px as f64).ceil() as usize + 1).max(2);
    let step = last / (count - 1) as f64;
    (0..count)
        .map(|index| {
            if index + 1 == count {
                last
            } else {
                index as f64 * step
            }
        })
        .collect()
}

impl Field {
    /// Measure `plane`'s PSF field and sample it at the nodes. With too few
    /// stars for a field that varies, one PSF is fitted for the whole image;
    /// with too few for that, the measurement's refusal comes back.
    pub(crate) fn measure(
        plane: &[f32],
        width: usize,
        height: usize,
        reading: &PsfReading,
        side: usize,
    ) -> Result<Self, ModelError> {
        let measure = |sparse| PsfField::measure(plane, width, height, 1, &reading.options(sparse));
        let field = match measure(SparseFieldPolicy::Refuse) {
            Err(PsfFieldError::TooFewStars { .. }) => measure(SparseFieldPolicy::Global)?,
            other => other?,
        };
        let (xs, ys) = (
            grid_nodes(width, reading.grid_px),
            grid_nodes(height, reading.grid_px),
        );
        let nodes: Vec<(f64, f64)> = ys
            .iter()
            .flat_map(|&y| xs.iter().map(move |&x| (x, y)))
            .collect();
        let sampled: Vec<(Vec<f32>, f64)> = nodes
            .par_iter()
            .map(|&(x, y)| {
                let stamp = field.stamp_at(x, y);
                let width = stamp_width(&stamp.data, stamp.size, reading.width_core);
                (fitted_stamp(&stamp.data, stamp.size, side), width)
            })
            .collect();
        let (stamps, widths) = sampled.into_iter().unzip();
        // The harness keeps the nodes in single precision, and reads them
        // back from it.
        let kept = |nodes: Vec<f64>| {
            nodes
                .into_iter()
                .map(|node| f64::from(node as f32))
                .collect()
        };
        Ok(Self {
            ys: kept(ys),
            xs: kept(xs),
            stamps,
            widths,
            side,
            stars: field.star_count(),
            global: field.is_global(),
        })
    }

    /// The median of the nodes' widths, in pixels; None where no node has
    /// one.
    pub(crate) fn typical_width(&self) -> Option<f64> {
        let mut finite: Vec<f64> = self
            .widths
            .iter()
            .copied()
            .filter(|w| w.is_finite())
            .collect();
        median_f64(&mut finite)
    }

    /// The ratio a network reads, FWHM(target) / FWHM(image), over an image
    /// of `width` × `height`: each star brought to `strength` of its width,
    /// never below `floor` pixels and never above its own, bilinear between
    /// the nodes. None where no node has a width.
    pub(crate) fn ratio_map(
        &self,
        width: usize,
        height: usize,
        strength: f64,
        floor: f64,
    ) -> Option<Vec<f32>> {
        let typical = self.typical_width()?;
        let nodes: Vec<f64> = self
            .widths
            .iter()
            .map(|&known| {
                let known = if known.is_finite() { known } else { typical };
                strength.max(floor / known).min(1.0)
            })
            .collect();
        let columns = self.xs.len();
        let across: Vec<(usize, f64)> = (0..width).map(|x| bracket(&self.xs, x as f64)).collect();
        let mut out = vec![0.0_f32; width * height];
        out.par_chunks_mut(width).enumerate().for_each(|(y, line)| {
            let (row, down) = bracket(&self.ys, y as f64);
            let (upper, lower) = (&nodes[row * columns..], &nodes[(row + 1) * columns..]);
            for (value, &(column, share)) in line.iter_mut().zip(&across) {
                let top = (1.0 - share) * upper[column] + share * upper[column + 1];
                let bottom = (1.0 - share) * lower[column] + share * lower[column + 1];
                *value = ((1.0 - down) * top + down * bottom) as f32;
            }
        });
        Some(out)
    }

    /// The nodes' `vectors`, `features` numbers each, interpolated
    /// bilinearly at the middles of a tile's coarsest pixels, `multiple`
    /// image pixels a side: `features` planes of `side / multiple` squared,
    /// for the tile whose corner is at (`left`, `top`).
    pub(crate) fn sampled_features(
        &self,
        vectors: &[f32],
        features: usize,
        (left, top): (usize, usize),
        side: usize,
        multiple: usize,
        out: &mut [f32],
    ) {
        let cells = side / multiple;
        let columns = self.xs.len();
        let middle = |corner: usize, cell: usize| {
            corner as f64 + (cell as f64 + 0.5) * multiple as f64 - 0.5
        };
        let across: Vec<(usize, f64)> = (0..cells)
            .map(|cell| bracket(&self.xs, middle(left, cell)))
            .collect();
        let down: Vec<(usize, f64)> = (0..cells)
            .map(|cell| bracket(&self.ys, middle(top, cell)))
            .collect();
        for (feature, plane) in out.chunks_mut(cells * cells).enumerate() {
            let node = |row: usize, column: usize| {
                f64::from(vectors[(row * columns + column) * features + feature])
            };
            for (line, &(row, share_down)) in plane.chunks_mut(cells).zip(&down) {
                for (value, &(column, share)) in line.iter_mut().zip(&across) {
                    let upper = (1.0 - share) * node(row, column) + share * node(row, column + 1);
                    let lower =
                        (1.0 - share) * node(row + 1, column) + share * node(row + 1, column + 1);
                    *value = ((1.0 - share_down) * upper + share_down * lower) as f32;
                }
            }
        }
    }
}

/// The node below `position` among `nodes`, and how far towards the next it
/// lies, held at both ends.
fn bracket(nodes: &[f64], position: f64) -> (usize, f64) {
    let last = nodes.len() - 2;
    let below = nodes
        .partition_point(|&node| node <= position)
        .saturating_sub(1)
        .min(last);
    let span = nodes[below + 1] - nodes[below];
    let share = if span > 0.0 {
        (position - nodes[below]) / span
    } else {
        0.0
    };
    (below, share.clamp(0.0, 1.0))
}

/// A stamp of `size` pixels a side as one of `side`, about its middle: padded
/// with zeros, or cut and brought back to a sum of one.
pub(crate) fn fitted_stamp(stamp: &[f32], size: usize, side: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; side * side];
    if size <= side {
        let pad = (side - size) / 2;
        for (row, line) in stamp.chunks(size).enumerate() {
            out[(row + pad) * side + pad..][..size].copy_from_slice(line);
        }
        return out;
    }
    let cut = (size - side) / 2;
    for (row, line) in out.chunks_mut(side).enumerate() {
        line.copy_from_slice(&stamp[(row + cut) * size + cut..][..side]);
    }
    let total: f64 = out.iter().map(|&value| f64::from(value)).sum();
    for value in &mut out {
        *value = (f64::from(*value) / total) as f32;
    }
    out
}

/// A stamp's FWHM as a round Gaussian's, centred on the stamp's middle and
/// fitted by least squares to the pixels at `core` of its peak or more: one
/// measure for every stamp, sharp or wide. NaN for a stamp with no positive
/// peak.
pub(crate) fn stamp_width(stamp: &[f32], size: usize, core: f64) -> f64 {
    let peak = stamp
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .fold(f32::NEG_INFINITY, f32::max);
    if !(peak > 0.0 && peak.is_finite()) {
        return f64::NAN;
    }
    let middle = (size as f64 - 1.0) / 2.0;
    // The level in single precision, as the harness compares it.
    let level = (core * f64::from(peak)) as f32;
    let pixels: Vec<(f64, f64)> = stamp
        .iter()
        .enumerate()
        .filter(|&(_, &value)| value >= level)
        .map(|(index, &value)| {
            let (dx, dy) = (
                (index % size) as f64 - middle,
                (index / size) as f64 - middle,
            );
            (dx * dx + dy * dy, f64::from(value))
        })
        .collect();
    let (first, last, count) = WIDTH_SIGMAS;
    let step = (last.log10() - first.log10()) / (count - 1) as f64;
    let sigma = |index: usize| match index {
        0 => first,
        index if index + 1 == count => last,
        index => 10.0_f64.powf(first.log10() + index as f64 * step),
    };
    let residual = |index: usize| {
        let scale = 2.0 * sigma(index) * sigma(index);
        let (mut cross, mut own) = (0.0, 0.0);
        for &(squared, value) in &pixels {
            let model = (-squared / scale).exp();
            cross += model * value;
            own += model * model;
        }
        let amplitude = cross / own;
        pixels
            .iter()
            .map(|&(squared, value)| (value - amplitude * (-squared / scale).exp()).powi(2))
            .sum::<f64>()
    };
    let residuals: Vec<f64> = (0..count).map(residual).collect();
    let best = (0..count)
        .min_by(|&a, &b| residuals[a].total_cmp(&residuals[b]))
        .unwrap_or(0)
        .clamp(1, count - 2);
    // A parabola through the best three, in log σ, as the grid is.
    let (low, middle_value, high) = (residuals[best - 1], residuals[best], residuals[best + 1]);
    let bend = low - 2.0 * middle_value + high;
    let shift = if bend > 0.0 {
        0.5 * (low - high) / bend
    } else {
        0.0
    };
    FWHM_PER_SIGMA * sigma(best) * (shift.clamp(-1.0, 1.0) * (sigma(1) / sigma(0)).ln()).exp()
}

/// The planes an encoder reads of a stamp, one after the other.
pub(crate) fn stamp_planes(stamp: &[f32], encoder: &StampEncoder) -> Vec<f32> {
    let peak = stamp.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let wings = encoder.wings as f32;
    let top = (1.0 / encoder.wings).asinh() as f32;
    let mut planes = Vec::with_capacity(stamp.len() * encoder.planes.len());
    for plane in &encoder.planes {
        planes.extend(stamp.iter().map(|&value| {
            let scaled = if peak > 0.0 { value / peak } else { 0.0 };
            match plane {
                StampPlane::Peak => scaled,
                StampPlane::Wings => (scaled / wings).asinh() / top,
            }
        }));
    }
    planes
}

/// One of an image's point sources.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Point {
    /// Its column, in pixels.
    pub(crate) x: f32,
    /// Its row, in pixels.
    pub(crate) y: f32,
    /// Its flux, in the image's units; NaN for a clipped core the detector
    /// turned away.
    pub(crate) flux: f32,
    /// The largest of the 3 × 3 pixels about it.
    pub(crate) peak: f32,
}

/// The image as the 16-bit counts the detector reads, and the image value one
/// count stands for.
fn scaled_counts(plane: &[f32], width: usize, height: usize) -> (Vec<u16>, f64) {
    let mut sample: Vec<f32> = (0..height)
        .step_by(COUNT_STRIDE)
        .flat_map(|y| {
            plane[y * width..(y + 1) * width]
                .iter()
                .step_by(COUNT_STRIDE)
        })
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if sample.is_empty() {
        return (vec![0; plane.len()], 0.0);
    }
    sample.sort_unstable_by(f32::total_cmp);
    let at = (sample.len() - 1) as f64 * COUNT_LOW_PERCENTILE / 100.0;
    let (below, share) = (at.floor() as usize, at.fract());
    let above = (below + 1).min(sample.len() - 1);
    let low =
        f64::from(sample[below]) + share * (f64::from(sample[above]) - f64::from(sample[below]));
    let low = f64::from(low as f32);
    let filler = median_f32(&mut sample).unwrap_or(0.0);
    let high = f64::from(
        plane
            .iter()
            .copied()
            .filter(|v| !v.is_nan())
            .fold(f32::NEG_INFINITY, f32::max),
    );
    let (low_f32, gain) = (low as f32, (COUNT_MOST / (high - low).max(1e-6)) as f32);
    let counts = plane
        .par_iter()
        .map(|&value| {
            let value = if value.is_finite() { value } else { filler };
            ((value - low_f32) * gain).clamp(0.0, COUNT_MOST as f32) as u16
        })
        .collect();
    (counts, (high.max(low + 1e-6) - low) / COUNT_MOST)
}

/// The largest finite value in the 3 × 3 pixels about a position, which is
/// rounded to the nearest pixel, a half to the even one.
fn raw_peak(plane: &[f32], width: usize, height: usize, x: f64, y: f64) -> f64 {
    let nearest = |value: f64, length: usize| {
        (value.round_ties_even() as isize).clamp(0, length as isize - 1)
    };
    let (column, row) = (nearest(x, width), nearest(y, height));
    let mut peak = f64::NEG_INFINITY;
    for down in -1..=1 {
        for across in -1..=1 {
            let (r, c) = (
                (row + down).clamp(0, height as isize - 1) as usize,
                (column + across).clamp(0, width as isize - 1) as usize,
            );
            let value = plane[r * width + c];
            if value.is_finite() {
                peak = peak.max(f64::from(value));
            }
        }
    }
    peak
}

/// The image's objects' peaks, brightest first, each with its column and row:
/// the largest value of every block of `block` pixels a side that is the
/// largest of the 3 × 3 blocks about it. A clipped core spread over several
/// blocks gives each of them, all at its clip level.
fn block_peaks(plane: &[f32], width: usize, height: usize, block: usize) -> Vec<(f64, f64, f64)> {
    let (rows, columns) = (height / block, width / block);
    let maxima: Vec<(f32, usize)> = (0..rows * columns)
        .into_par_iter()
        .map(|index| {
            let (row, column) = (index / columns, index % columns);
            let mut best = (f32::NEG_INFINITY, 0);
            for y in 0..block {
                let line = &plane[(row * block + y) * width + column * block..][..block];
                for (x, &value) in line.iter().enumerate() {
                    if value.is_finite() && value > best.0 {
                        best = (value, y * block + x);
                    }
                }
            }
            best
        })
        .collect();
    let mut peaks = Vec::new();
    for row in 0..rows {
        for column in 0..columns {
            let (value, offset) = maxima[row * columns + column];
            if !value.is_finite() {
                continue;
            }
            let around = (row.saturating_sub(1)..(row + 2).min(rows))
                .flat_map(|r| {
                    (column.saturating_sub(1)..(column + 2).min(columns)).map(move |c| (r, c))
                })
                .map(|(r, c)| maxima[r * columns + c].0)
                .fold(f32::NEG_INFINITY, f32::max);
            if value >= around {
                peaks.push((
                    f64::from(value),
                    (column * block + offset % block) as f64,
                    (row * block + offset / block) as f64,
                ));
            }
        }
    }
    peaks.sort_by(|a, b| b.0.total_cmp(&a.0));
    peaks
}

/// The level at which an image's brightest stars clip, from its peaks,
/// brightest first; None where no group of them shares one.
fn clip_level(peaks: &[f64]) -> Option<f64> {
    let brightest = *peaks.first()?;
    if brightest <= 0.0 {
        return None;
    }
    for (start, &top) in peaks.iter().enumerate() {
        if top < CLIP_BRIGHTEST * brightest {
            break;
        }
        // The values run down, so those within 2% of this one follow it.
        let clustered = peaks[start..]
            .iter()
            .filter(|&&value| value >= CLIP_WITHIN * top)
            .count();
        if clustered >= CLIP_CLUSTER {
            let mut group = peaks[start..start + clustered].to_vec();
            return median_f64(&mut group).map(|median| CLIP_LEVEL * median);
        }
    }
    None
}

/// Every star the measuring detector finds on the image, unsmoothed, with
/// the card's first structure removal; the clipped cores it turns away,
/// each at its brightest pixel with its peak alone; then the stars each
/// later removal finds beyond the core radius of every point listed before
/// it. The first removal's list and its cores are so the same as with that
/// removal alone, and a later removal's star near a clipped core does not
/// take the core's place.
pub(crate) fn detections(
    plane: &[f32],
    width: usize,
    height: usize,
    reading: &PointsReading,
) -> Vec<Point> {
    let peaks = block_peaks(plane, width, height, reading.peak_block);
    let brightest: Vec<f64> = peaks
        .iter()
        .take(reading.clip_peaks)
        .map(|peak| peak.0)
        .collect();
    let level = clip_level(&brightest);
    let (counts, scale) = scaled_counts(plane, width, height);
    let reach = reading.core_radius * reading.core_radius;
    let near = |known: &[(f64, f64)], x: f64, y: f64| {
        known
            .iter()
            .any(|&(kx, ky)| (kx - x).powi(2) + (ky - y).powi(2) <= reach)
    };
    let mut known: Vec<(f64, f64)> = Vec::new();
    let mut points: Vec<Point> = Vec::new();
    for (pass, removal) in reading.structure_removals.iter().enumerate() {
        let options = MeasurementOptions {
            noise_reduction_radius: Some(reading.noise_reduction_radius),
            structure_removal: Some(removal.method()),
            ..Default::default()
        };
        // Against the points listed before this removal, not each other.
        let listed = known.len();
        for star in options.detect(&counts, width, height).stars {
            let (x, y) = star.position;
            if pass > 0 && near(&known[..listed], x, y) {
                continue;
            }
            known.push((x, y));
            points.push(Point {
                x: x as f32,
                y: y as f32,
                flux: (star.flux * scale) as f32,
                peak: raw_peak(plane, width, height, x, y) as f32,
            });
        }
        if pass > 0 {
            continue;
        }
        if let Some(level) = level {
            for &(value, x, y) in peaks.iter().take_while(|peak| peak.0 >= level) {
                if !near(&known, x, y) {
                    known.push((x, y));
                    points.push(Point {
                        x: x as f32,
                        y: y as f32,
                        flux: f32::NAN,
                        peak: value as f32,
                    });
                }
            }
        }
    }
    points
}

/// The point sources rendered over an image of `width` × `height`, each with
/// the stamp of the field's node nearest it, at its flux and sub-pixel place,
/// in the image's units above `background`. A clipped core the detector
/// turned away is given its peak over the stamp's, which a clipped peak makes
/// too low but places where it is.
///
/// The harness lays each point down as four bilinear shares and convolves
/// them with the stamp by a Fourier transform; laying the stamp down four
/// times is the same sum.
pub(crate) fn field_point_map(
    points: &[Point],
    field: &Field,
    width: usize,
    height: usize,
    background: f32,
) -> Vec<f32> {
    let side = field.side;
    let half = (side / 2) as isize;
    // Each node's stamp, cut to the rows and columns that hold anything.
    struct Held {
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        peak: f32,
    }
    let cut: Vec<Option<Held>> = field
        .stamps
        .iter()
        .map(|stamp| {
            let held = |index: usize| stamp[index] != 0.0;
            let mut rows =
                (0..side).filter(|&row| (0..side).any(|column| held(row * side + column)));
            let mut columns =
                (0..side).filter(|&column| (0..side).any(|row| held(row * side + column)));
            let (top, left) = (rows.next()?, columns.next()?);
            Some(Held {
                top,
                bottom: rows.next_back().unwrap_or(top),
                left,
                right: columns.next_back().unwrap_or(left),
                peak: stamp.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            })
        })
        .collect();
    let nearest = |nodes: &[f64], position: f32| {
        (0..nodes.len())
            .min_by(|&a, &b| {
                (f64::from(position) - nodes[a])
                    .abs()
                    .total_cmp(&(f64::from(position) - nodes[b]).abs())
            })
            .unwrap_or(0)
    };
    // A point's stamp, and the four corners its flux is shared between.
    struct Laid {
        node: usize,
        row: isize,
        column: isize,
        shares: [f64; 4],
    }
    let laid: Vec<Laid> = points
        .iter()
        .filter_map(|point| {
            let node = nearest(&field.ys, point.y) * field.xs.len() + nearest(&field.xs, point.x);
            let peak = cut[node].as_ref()?.peak;
            let flux = if point.flux.is_finite() {
                f64::from(point.flux)
            } else {
                ((f64::from(point.peak) - f64::from(background)) / f64::from(peak).max(1e-12))
                    .max(0.0)
            };
            let (x, y) = (f64::from(point.x), f64::from(point.y));
            let inside = x >= 0.0 && y >= 0.0 && x < width as f64 && y < height as f64;
            if !(inside && flux.is_finite()) {
                return None;
            }
            let (across, down) = (x - x.floor(), y - y.floor());
            Some(Laid {
                node,
                row: y.floor() as isize,
                column: x.floor() as isize,
                shares: [
                    flux * (1.0 - down) * (1.0 - across),
                    flux * (1.0 - down) * across,
                    flux * down * (1.0 - across),
                    flux * down * across,
                ],
            })
        })
        .collect();
    let mut out = vec![0.0_f64; width * height];
    out.par_chunks_mut(width).enumerate().for_each(|(y, line)| {
        let y = y as isize;
        for point in &laid {
            let Some(&Held {
                top,
                bottom,
                left,
                right,
                ..
            }) = cut[point.node].as_ref()
            else {
                continue;
            };
            let stamp = &field.stamps[point.node];
            for (corner, &share) in point.shares.iter().enumerate() {
                let (row, column) = (
                    point.row + (corner / 2) as isize,
                    point.column + (corner % 2) as isize,
                );
                // The stamp's row that falls on this line.
                let source = y - row + half;
                if share == 0.0 || source < top as isize || source > bottom as isize {
                    continue;
                }
                let values = &stamp[source as usize * side..][..side];
                let first = (column - half + left as isize).max(0);
                let end = (column - half + right as isize + 1).min(width as isize);
                for x in first..end {
                    line[x as usize] += share * f64::from(values[(x - column + half) as usize]);
                }
            }
        }
    });
    out.into_par_iter().map(|value| value as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian(side: usize, sigma: f64) -> Vec<f32> {
        let middle = (side as f64 - 1.0) / 2.0;
        let mut stamp: Vec<f32> = (0..side * side)
            .map(|index| {
                let (dx, dy) = (
                    (index % side) as f64 - middle,
                    (index / side) as f64 - middle,
                );
                (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp() as f32
            })
            .collect();
        let total: f32 = stamp.iter().sum();
        stamp.iter_mut().for_each(|value| *value /= total);
        stamp
    }

    fn field(stamps: Vec<Vec<f32>>, side: usize, xs: Vec<f64>, ys: Vec<f64>) -> Field {
        let widths = stamps
            .iter()
            .map(|stamp| stamp_width(stamp, side, 0.05))
            .collect();
        Field {
            ys,
            xs,
            stamps,
            widths,
            side,
            stars: 0,
            global: false,
        }
    }

    #[test]
    fn a_stamp_s_width_is_its_gaussian_s() {
        for sigma in [0.8, 1.5, 3.0, 6.0] {
            let width = stamp_width(&gaussian(65, sigma), 65, 0.05);
            assert!(
                (width / (2.3548 * sigma) - 1.0).abs() < 2e-3,
                "{sigma}: {width}"
            );
        }
        assert!(stamp_width(&[0.0; 81], 9, 0.05).is_nan());
    }

    #[test]
    fn a_stamp_is_padded_or_cut_about_its_middle() {
        let small = gaussian(9, 1.0);
        let padded = fitted_stamp(&small, 9, 15);
        assert_eq!(padded[7 * 15 + 7], small[4 * 9 + 4]);
        assert_eq!(padded.iter().filter(|&&value| value != 0.0).count(), 81);
        let cut = fitted_stamp(&gaussian(15, 3.0), 15, 9);
        assert!(
            (cut.iter().sum::<f32>() - 1.0).abs() < 1e-5,
            "a cut stamp sums to one again"
        );
        assert_eq!(cut.iter().copied().fold(0.0, f32::max), cut[4 * 9 + 4]);
    }

    #[test]
    fn nodes_stand_on_both_edges_at_most_the_grid_apart() {
        assert_eq!(grid_nodes(257, 256), [0.0, 256.0]);
        assert_eq!(grid_nodes(258, 256), [0.0, 128.5, 257.0]);
        assert_eq!(grid_nodes(100, 256), [0.0, 99.0]);
    }

    #[test]
    fn a_ratio_keeps_the_floor_and_never_widens() {
        let side = 33;
        let stamps = vec![
            gaussian(side, 1.0),
            gaussian(side, 3.0),
            gaussian(side, 1.0),
            gaussian(side, 3.0),
        ];
        let field = field(stamps, side, vec![0.0, 99.0], vec![0.0, 49.0]);
        let map = field.ratio_map(100, 50, 0.5, 1.5).unwrap();
        let (narrow, wide) = (field.widths[0], field.widths[1]);
        assert!(
            (f64::from(map[0]) - 1.5 / narrow).abs() < 1e-6,
            "the floor holds a narrow star"
        );
        assert!(
            (f64::from(map[99]) - 0.5).abs() < 1e-6,
            "a wide star takes the share: {wide}"
        );
        assert!(map.iter().all(|&ratio| ratio > 0.0 && ratio <= 1.0));
        let middle = f64::from(map[25 * 100 + 50]);
        let expected = (1.5 / narrow) * (1.0 - 50.0 / 99.0) + 0.5 * (50.0 / 99.0);
        assert!(
            (middle - expected).abs() < 1e-6,
            "bilinear between the nodes"
        );
    }

    #[test]
    fn vectors_are_held_past_the_outer_nodes_and_run_between_them() {
        let side = 9;
        let field = field(
            vec![gaussian(side, 1.0); 4],
            side,
            vec![0.0, 63.0],
            vec![0.0, 63.0],
        );
        // Two features: the node's column, and its row.
        let vectors = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; 2 * 16];
        field.sampled_features(&vectors, 2, (32, 48), 32, 8, &mut out);
        let middles = [35.5, 43.5, 51.5, 59.5].map(|x: f64| (x / 63.0) as f32);
        for (cell, value) in out[..4].iter().enumerate() {
            assert!((value - middles[cell]).abs() < 1e-6, "{out:?}");
        }
        assert!((out[16] - (51.5_f32 / 63.0)).abs() < 1e-6);
        assert_eq!(
            out[16 + 12],
            1.0,
            "a tile past the last node holds its vector"
        );
    }

    #[test]
    fn a_point_is_laid_down_as_its_stamp_at_its_flux_and_place() {
        let side = 9;
        let field = field(
            vec![gaussian(side, 1.2); 4],
            side,
            vec![0.0, 39.0],
            vec![0.0, 29.0],
        );
        let point = |x, y, flux, peak| Point { x, y, flux, peak };
        let points = [
            point(12.0, 10.0, 500.0, 0.0),
            point(30.25, 20.5, 80.0, 0.0),
            point(1.0, 1.0, f32::NAN, 60.0),
        ];
        let map = field_point_map(&points, &field, 40, 30, 10.0);
        let stamp = &field.stamps[0];
        assert!((map[10 * 40 + 12] - 500.0 * stamp[4 * side + 4]).abs() < 1e-3);
        assert!((map[8 * 40 + 13] - 500.0 * stamp[2 * side + 5]).abs() < 1e-3);
        let total: f64 = map.iter().map(|&value| f64::from(value)).sum();
        let core = 50.0 / stamp[4 * side + 4];
        let lost: f32 = (0..side * side)
            .filter(|index| index / side < 3 || index % side < 3)
            .map(|index| stamp[index])
            .sum();
        assert!(
            (total - f64::from(580.0 + core * (1.0 - lost))).abs() < 0.05,
            "flux is kept, less what a corner star throws off the image: {total}"
        );
        // A star between pixels shares its flux between them.
        let (left, right) = (map[20 * 40 + 30], map[20 * 40 + 31]);
        assert!(left > right && right > 0.0);
        assert!(
            (map[20 * 40 + 30] - map[21 * 40 + 30]).abs() < 1e-5,
            "half a pixel down"
        );
    }

    #[test]
    fn a_stamp_s_planes_are_its_share_of_the_peak_and_its_wings() {
        let encoder = StampEncoder {
            file: String::new(),
            sha256: String::new(),
            input: String::new(),
            output: String::new(),
            planes: vec![StampPlane::Peak, StampPlane::Wings],
            side: 9,
            wings: 1e-3,
            features: 4,
        };
        let stamp = gaussian(9, 1.0);
        let planes = stamp_planes(&stamp, &encoder);
        assert_eq!(planes.len(), 162);
        assert_eq!(planes[40], 1.0);
        assert!(
            (planes[81 + 40] - 1.0).abs() < 1e-6,
            "the peak's wings plane is one"
        );
        assert!(
            planes[81] > 10.0 * planes[0],
            "the wings plane lifts a faint corner"
        );
    }

    #[test]
    fn clipped_cores_share_a_level_and_unclipped_peaks_do_not() {
        assert_eq!(
            clip_level(&[1000.0, 995.0, 990.0, 400.0]),
            Some(0.8 * 995.0)
        );
        assert_eq!(clip_level(&[1000.0, 900.0, 800.0, 700.0]), None);
        assert_eq!(clip_level(&[]), None);
    }

    #[test]
    fn a_clipped_core_the_detector_turns_away_is_found_by_its_peak() {
        let (width, height) = (96, 64);
        let mut plane = vec![10.0_f32; width * height];
        for (cx, cy) in [(20, 20), (60, 20), (40, 45)] {
            for dy in -2_i32..=2 {
                for dx in -2_i32..=2 {
                    plane[(cy + dy) as usize * width + (cx + dx) as usize] = 5000.0;
                }
            }
        }
        let peaks = block_peaks(&plane, width, height, 16);
        assert!(peaks.len() >= 3 && peaks[0].0 == 5000.0);
        let reading = PointsReading {
            noise_reduction_radius: 0,
            structure_removals: vec![StructureRemoval::Filtered, StructureRemoval::Atrous],
            clip_peaks: 300,
            peak_block: 16,
            core_radius: 24.0,
        };
        let points = detections(&plane, width, height, &reading);
        let cores: Vec<&Point> = points.iter().filter(|point| point.flux.is_nan()).collect();
        for (cx, cy) in [(20.0_f32, 20.0_f32), (60.0, 20.0), (40.0, 45.0)] {
            let near = points
                .iter()
                .filter(|p| (p.x - cx).abs() < 4.0 && (p.y - cy).abs() < 4.0)
                .count();
            assert_eq!(near, 1, "each star once, detected or not: {points:?}");
        }
        assert!(cores.iter().all(|core| core.peak == 5000.0));
    }
}
