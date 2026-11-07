//! The sky, the image's scales, and the planes a network reads. These follow
//! the harness's functions of the same names, in the same precision at each
//! step, because a model is trained on exactly what they make.

use crate::bundle::{NoisePlane, Reading, SkyInput};
use crate::stats::median_f32;
use rayon::prelude::*;

/// Every this many finite pixels, `scales` takes one.
const SCALE_STRIDE: usize = 17;

/// A grid of block values, one per `block` × `block` pixels, spread bilinearly
/// over `height` × `width` pixels, past the outer blocks' centres held at
/// theirs. A grid already extended by [`extended`] passes an `offset` of 1, so
/// that it carries on along the slope instead.
pub(crate) fn upsample(
    grid: &[f64],
    rows: usize,
    columns: usize,
    height: usize,
    width: usize,
    block: usize,
    offset: f32,
) -> Vec<f32> {
    let position = |index: usize, limit: usize| -> (usize, usize, f64) {
        let value = (index as f32 + 0.5) / block as f32 - 0.5 + offset;
        let value = value.clamp(0.0, (limit - 1) as f32);
        let lower = value.floor() as usize;
        (
            lower,
            (lower + 1).min(limit - 1),
            f64::from(value) - lower as f64,
        )
    };
    let across: Vec<(usize, usize, f64)> = (0..width).map(|x| position(x, columns)).collect();
    let mut out = vec![0.0_f32; width * height];
    out.par_chunks_mut(width).enumerate().for_each(|(y, line)| {
        let (y0, y1, wy) = position(y, rows);
        let (first, second) = (
            &grid[y0 * columns..][..columns],
            &grid[y1 * columns..][..columns],
        );
        for (pixel, &(x0, x1, wx)) in line.iter_mut().zip(&across) {
            let top = first[x0] * (1.0 - wx) + first[x1] * wx;
            let bottom = second[x0] * (1.0 - wx) + second[x1] * wx;
            *pixel = (top * (1.0 - wy) + bottom * wy) as f32;
        }
    });
    out
}

/// A grid with one more value on each side, carried on along the slope from
/// the edge's neighbour, or repeated along a side one value long.
fn extended(grid: &[f32], rows: usize, columns: usize) -> (Vec<f32>, usize, usize) {
    let past = |near: f32, inner: Option<f32>| match inner {
        Some(inner) => 2.0 * near - inner,
        None => near,
    };
    let wide = columns + 2;
    let mut lines = vec![0.0_f32; (rows + 2) * columns];
    for x in 0..columns {
        let at = |y: usize| grid[y * columns + x];
        let inner = |y: usize| (rows > 1).then(|| at(y));
        lines[x] = past(at(0), inner(1));
        for y in 0..rows {
            lines[(y + 1) * columns + x] = at(y);
        }
        lines[(rows + 1) * columns + x] = past(at(rows - 1), inner(rows.saturating_sub(2)));
    }
    let mut out = vec![0.0_f32; (rows + 2) * wide];
    for y in 0..rows + 2 {
        let line = &lines[y * columns..][..columns];
        let inner = |x: usize| (columns > 1).then(|| line[x]);
        out[y * wide] = past(line[0], inner(1));
        out[y * wide + 1..y * wide + 1 + columns].copy_from_slice(line);
        out[y * wide + wide - 1] = past(line[columns - 1], inner(columns.saturating_sub(2)));
    }
    (out, rows + 2, wide)
}

/// The slow change of an image's sky, less its own median, so that the image
/// less it keeps its level: medians over `block`-pixel blocks (one block where
/// the image is smaller), a 3 × 3 median over those against a star or a knot
/// that fills a block, spread back bilinearly and carried on past the outer
/// blocks' centres, as a gradient is.
pub(crate) fn smooth_sky(plane: &[f32], width: usize, height: usize, block: usize) -> Vec<f32> {
    let block = block.min(height).min(width);
    let (rows, columns) = (height / block, width / block);
    let mut grid: Vec<f32> = (0..rows * columns)
        .into_par_iter()
        .map(|cell| {
            let (top, left) = (cell / columns * block, cell % columns * block);
            let mut values = Vec::with_capacity(block * block);
            for y in top..top + block {
                values.extend(
                    plane[y * width + left..y * width + left + block]
                        .iter()
                        .filter(|value| value.is_finite()),
                );
            }
            median_f32(&mut values).unwrap_or(f32::NAN)
        })
        .collect();
    let mut finite: Vec<f32> = grid
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let fill = median_f32(&mut finite).unwrap_or(f32::NAN);
    for value in &mut grid {
        if !value.is_finite() {
            *value = fill;
        }
    }
    // Carried on past the edge, so that a gradient's edge blocks keep their
    // values through the median.
    let (padded, _, wide) = extended(&grid, rows, columns);
    let mut smoothed = vec![0.0_f32; rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            let mut around = [0.0_f32; 9];
            for (index, value) in around.iter_mut().enumerate() {
                *value = padded[(row + index / 3) * wide + column + index % 3];
            }
            smoothed[row * columns + column] = median_f32(&mut around).unwrap_or(f32::NAN);
        }
    }
    let centre = median_f32(&mut smoothed.clone()).unwrap_or(0.0);
    let less: Vec<f32> = smoothed.iter().map(|value| value - centre).collect();
    let (grown, grown_rows, grown_columns) = extended(&less, rows, columns);
    let grown: Vec<f64> = grown.iter().map(|&value| f64::from(value)).collect();
    upsample(&grown, grown_rows, grown_columns, height, width, block, 1.0)
}

/// The image the network reads: with a flat sky, less its smooth sky.
pub(crate) fn seen(plane: &[f32], width: usize, height: usize, reading: &Reading) -> Vec<f32> {
    match reading.sky {
        SkyInput::Global => plane.to_vec(),
        SkyInput::Flat => {
            let sky = smooth_sky(plane, width, height, reading.flat_block);
            plane
                .par_iter()
                .zip(sky.par_iter())
                .map(|(value, sky)| value - sky)
                .collect()
        }
    }
}

/// An image's background and median noise, from every 17th of the pixels both
/// planes hold. `None` when they share no finite pixel.
pub(crate) fn scales(image: &[f32], sigma: &[f32]) -> Option<(f32, f32)> {
    let (mut backgrounds, mut noises) = (Vec::new(), Vec::new());
    let finite = image
        .iter()
        .zip(sigma)
        .filter(|(value, sigma)| value.is_finite() && sigma.is_finite());
    for (value, sigma) in finite.step_by(SCALE_STRIDE) {
        backgrounds.push(*value);
        noises.push(*sigma);
    }
    Some((median_f32(&mut backgrounds)?, median_f32(&mut noises)?))
}

/// The two planes the network reads, one after the other, the noise map as
/// `reading` says; NaN becomes the background.
pub(crate) fn network_input(
    image: &[f32],
    sigma: &[f32],
    background: f32,
    noise: f32,
    plane: NoisePlane,
) -> Vec<f32> {
    let mut planes = vec![0.0_f32; image.len() * 2];
    let (first, second) = planes.split_at_mut(image.len());
    first
        .par_iter_mut()
        .zip(image.par_iter())
        .for_each(|(out, &value)| {
            let scaled = (value - background) / noise;
            *out = if scaled.is_nan() { 0.0 } else { scaled.asinh() };
        });
    second
        .par_iter_mut()
        .zip(sigma.par_iter())
        .for_each(|(out, &value)| {
            let scaled = value / noise;
            let scaled = if scaled.is_nan() { 1.0 } else { scaled };
            *out = match plane {
                NoisePlane::Asinh => scaled.asinh(),
                NoisePlane::Linear => scaled,
            };
        });
    planes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsampling_holds_past_the_outer_centres_and_runs_between_them() {
        let grid = [0.0, 10.0];
        let line = upsample(&grid, 1, 2, 1, 8, 4, 0.0);
        assert_eq!(line[0], 0.0, "held before the first centre");
        assert_eq!(line[7], 10.0, "held past the last");
        assert!(line.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn extending_a_grid_carries_its_slope_on() {
        let (grid, rows, columns) = extended(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        assert_eq!((rows, columns), (4, 4));
        assert_eq!(&grid[4..8], &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(grid[0], -2.0, "the slope along both axes");
        let (line, rows, _) = extended(&[1.0, 2.0], 1, 2);
        assert_eq!(rows, 3);
        assert_eq!(&line[0..4], &line[4..8], "a side one value long repeats");
    }

    #[test]
    fn the_smooth_sky_finds_a_gradient_and_keeps_the_image_s_level() {
        let (width, height, block) = (512, 256, 64);
        let image: Vec<f32> = (0..width * height)
            .map(|index| 1000.0 + 0.5 * (index % width) as f32)
            .collect();
        let sky = smooth_sky(&image, width, height, block);
        let flat: Vec<f32> = image
            .iter()
            .zip(&sky)
            .map(|(value, sky)| value - sky)
            .collect();
        let (low, high) = flat
            .iter()
            .fold((f32::MAX, f32::MIN), |(low, high), &value| {
                (low.min(value), high.max(value))
            });
        assert!(high - low < 1.0, "the gradient comes out: {low} to {high}");
        let mean = flat.iter().map(|&value| f64::from(value)).sum::<f64>() / flat.len() as f64;
        assert!((mean - 1127.75).abs() < 1.0, "the level stays: {mean}");
    }

    #[test]
    fn the_network_reads_nan_as_the_background_and_a_missing_map_as_one() {
        let planes = network_input(
            &[f32::NAN, 110.0],
            &[f32::NAN, 2.0],
            100.0,
            2.0,
            NoisePlane::Asinh,
        );
        assert_eq!(planes[0], 0.0);
        assert!((planes[1] - 5.0_f32.asinh()).abs() < 1e-6);
        assert!((planes[2] - 1.0_f32.asinh()).abs() < 1e-6);
        assert!((planes[3] - 1.0_f32.asinh()).abs() < 1e-6);
    }

    #[test]
    fn scales_sample_every_17th_pixel_both_planes_hold() {
        let image: Vec<f32> = (0..170).map(|index| index as f32).collect();
        let sigma = vec![2.0_f32; 170];
        assert_eq!(scales(&image, &sigma), Some((76.5, 2.0)));
        assert_eq!(scales(&[f32::NAN], &[1.0]), None);
    }
}
