//! Keeping an image's holes from a network.
//!
//! A stack can hold pixels far below its sky that every frame holds, so no
//! rejection takes them out: hot pixels a dark took too much of, dead ones a
//! missing flat leaves. A network that meets one fills it and overshoots, and
//! turns the hole into a bright spike ringed by speckle. Whatever made a hole,
//! the model is not the place to decide it is a fault. So a model whose card
//! asks for it ([`Reading::holes`]) never sees them: pixels deeper below the
//! sky's floor than any noise reaches are filled from their surroundings
//! before the network runs, and put back after, with the lobes registration
//! spread round them.
//!
//! This is the harness's `holes` module, held to it by the parity fixtures.

use crate::bundle::{HoleGuard, Reading};
use crate::noise::image_sigma;
use crate::sky::upsample;
use crate::stats::median_f64;
use rayon::prelude::*;

/// Each block's floor is this percentile of its pixels, read between order
/// statistics as numpy reads it by default.
const FLOOR_PERCENTILE: f64 = 20.0;
/// The image's background is the median of every this many pixels along each
/// axis, as the noise map takes it.
const BACKGROUND_STRIDE: usize = 7;

/// What a guarded network reads, and what the guard needs to put the holes
/// back.
pub(crate) struct Guarded {
    /// The image, its holes filled from their surroundings.
    pub(crate) read: Vec<f32>,
    /// The pixels the network does not see.
    pub(crate) mask: Vec<bool>,
    /// The image map the holes are measured in.
    noise_map: Vec<f32>,
}

/// The image `guard` has the network read in place of `plane`, or `None` when
/// it holds no hole and the network reads the image as it is.
pub(crate) fn guarded(
    plane: &[f32],
    width: usize,
    height: usize,
    guard: &HoleGuard,
    reading: &Reading,
) -> Option<Guarded> {
    let noise_map = image_sigma(
        plane,
        width,
        height,
        reading.sky_block,
        reading.image_map_scale,
    );
    let floor = sky_floor(plane, width, height, reading.sky_block);
    let deep: Vec<bool> = plane
        .par_iter()
        .zip(floor.par_iter().zip(noise_map.par_iter()))
        .map(|(&value, (&floor, &noise))| {
            value.is_finite()
                && f64::from(floor) - f64::from(value) > guard.depth * f64::from(noise)
        })
        .collect();
    if !deep.contains(&true) {
        return None;
    }
    let mask: Vec<bool> = grown(&deep, width, height, guard.margin)
        .into_iter()
        .zip(plane)
        .map(|(inside, value)| inside && value.is_finite())
        .collect();
    let read = filled(plane, &mask, width, height, guard.fill_radius, &floor);
    Some(Guarded {
        read,
        mask,
        noise_map,
    })
}

/// What a network `found` in the guarded image, as a residual of `plane`: the
/// network's own outside the mask, and within it what takes `plane` to the
/// output, with the input put back where it differs from the output by more
/// than `guard.restore` of the image map.
pub(crate) fn guarded_residual(
    plane: &[f32],
    guarded: &Guarded,
    found: &[f32],
    guard: &HoleGuard,
) -> Vec<f32> {
    plane
        .par_iter()
        .zip(found.par_iter())
        .enumerate()
        .map(|(index, (&value, &found))| {
            if !guarded.mask[index] {
                return found;
            }
            let output = guarded.read[index] + found;
            let differs = (f64::from(value) - f64::from(output)).abs()
                > guard.restore * f64::from(guarded.noise_map[index]);
            if differs { 0.0 } else { output - value }
        })
        .collect()
}

/// The sky's floor under each pixel: the lowest of a block's
/// `FLOOR_PERCENTILE` and its eight neighbours', spread bilinearly, and never
/// above the image's background. The blocks are the noise map's; one a large
/// star or a nebula's edge lifts is passed over for a neighbour on the sky, and
/// a dark lane within a nebula wider than them, which stays above the sky, is
/// not a hole.
pub(crate) fn sky_floor(plane: &[f32], width: usize, height: usize, block: usize) -> Vec<f32> {
    let rows = (height / block).max(1);
    let columns = (width / block).max(1);
    let lows: Vec<f64> = (0..rows * columns)
        .into_par_iter()
        .map(|cell| {
            let (top, left) = (cell / columns * block, cell % columns * block);
            let mut values = Vec::with_capacity(block * block);
            for y in top..(top + block).min(height) {
                let line = &plane[y * width..][..width];
                values.extend(
                    line[left..(left + block).min(width)]
                        .iter()
                        .filter(|value| value.is_finite())
                        .map(|&value| f64::from(value)),
                );
            }
            percentile(&mut values, FLOOR_PERCENTILE).unwrap_or(f64::INFINITY)
        })
        .collect();
    let mut sampled = Vec::new();
    for y in (0..height).step_by(BACKGROUND_STRIDE) {
        for x in (0..width).step_by(BACKGROUND_STRIDE) {
            let value = plane[y * width + x];
            if value.is_finite() {
                sampled.push(f64::from(value));
            }
        }
    }
    let background = median_f64(&mut sampled).unwrap_or(0.0);
    let floor: Vec<f64> = (0..rows * columns)
        .map(|cell| {
            let (row, column) = (cell / columns, cell % columns);
            let mut lowest = f64::INFINITY;
            for near in row.saturating_sub(1)..=(row + 1).min(rows - 1) {
                for across in column.saturating_sub(1)..=(column + 1).min(columns - 1) {
                    lowest = lowest.min(lows[near * columns + across]);
                }
            }
            if lowest.is_finite() {
                lowest
            } else {
                background
            }
        })
        .collect();
    let background = background as f32;
    upsample(&floor, rows, columns, height, width, block, 0.0)
        .into_iter()
        .map(|value| value.min(background))
        .collect()
}

/// The `percent` percentile of `values`, between order statistics as numpy's
/// default reads it. `None` when there are none.
fn percentile(values: &mut [f64], percent: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let count = values.len();
    let rank = percent / 100.0 * (count - 1) as f64;
    let lower = rank.floor() as usize;
    let (_, &mut below, above) = values.select_nth_unstable_by(lower, f64::total_cmp);
    if lower + 1 == count {
        return Some(below);
    }
    let next = above.iter().copied().fold(f64::INFINITY, f64::min);
    Some(below + (next - below) * (rank - lower as f64))
}

/// `mask` widened by `radius` pixels in every direction, to a square.
fn grown(mask: &[bool], width: usize, height: usize, radius: usize) -> Vec<bool> {
    let mut rows = vec![false; mask.len()];
    rows.par_chunks_mut(width)
        .zip(mask.par_chunks(width))
        .for_each(|(out, line)| {
            for (x, inside) in line.iter().enumerate() {
                if *inside {
                    out[x.saturating_sub(radius)..=(x + radius).min(width - 1)].fill(true);
                }
            }
        });
    let mut out = vec![false; mask.len()];
    out.par_chunks_mut(width).enumerate().for_each(|(y, line)| {
        for near in y.saturating_sub(radius)..=(y + radius).min(height - 1) {
            for (pixel, &inside) in line.iter_mut().zip(&rows[near * width..][..width]) {
                *pixel |= inside;
            }
        }
    });
    out
}

/// The sum over the box `radius` round each pixel, cut at the image's edges.
fn box_sums(plane: &[f64], width: usize, height: usize, radius: usize) -> Vec<f64> {
    let wide = width + 1;
    let mut total = vec![0.0_f64; (height + 1) * wide];
    for y in 0..height {
        let mut line = 0.0;
        for x in 0..width {
            line += plane[y * width + x];
            total[(y + 1) * wide + x + 1] = total[y * wide + x + 1] + line;
        }
    }
    let mut out = vec![0.0_f64; width * height];
    out.par_chunks_mut(width).enumerate().for_each(|(y, line)| {
        let (top, bottom) = (y.saturating_sub(radius), (y + radius + 1).min(height));
        for (x, sum) in line.iter_mut().enumerate() {
            let (left, right) = (x.saturating_sub(radius), (x + radius + 1).min(width));
            *sum = total[bottom * wide + right] - total[top * wide + right]
                - total[bottom * wide + left]
                + total[top * wide + left];
        }
    });
    out
}

/// `plane` with each masked pixel the mean of the unmasked ones within
/// `radius`, or the floor where there are none.
fn filled(
    plane: &[f32],
    mask: &[bool],
    width: usize,
    height: usize,
    radius: usize,
    floor: &[f32],
) -> Vec<f32> {
    let kept: Vec<bool> = plane
        .iter()
        .zip(mask)
        .map(|(value, &inside)| value.is_finite() && !inside)
        .collect();
    let values: Vec<f64> = plane
        .iter()
        .zip(&kept)
        .map(|(&value, &keep)| if keep { f64::from(value) } else { 0.0 })
        .collect();
    let ones: Vec<f64> = kept
        .iter()
        .map(|&keep| if keep { 1.0 } else { 0.0 })
        .collect();
    let totals = box_sums(&values, width, height, radius);
    let counts = box_sums(&ones, width, height, radius);
    plane
        .par_iter()
        .enumerate()
        .map(|(index, &value)| {
            if !mask[index] {
                value
            } else if counts[index] > 0.5 {
                (totals[index] / counts[index]) as f32
            } else {
                floor[index]
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundle::{NoiseInput, NoisePlane, SkyInput};

    const GUARD: HoleGuard = HoleGuard {
        depth: 30.0,
        margin: 3,
        fill_radius: 12,
        restore: 5.0,
    };

    fn reading() -> Reading {
        Reading {
            noise_map: NoiseInput::Constant,
            noise_plane: NoisePlane::Asinh,
            sky: SkyInput::Flat,
            sky_block: 32,
            flat_block: 256,
            image_map_scale: 1.13,
            holes: Some(GUARD),
        }
    }

    /// A sky of uniform noise on a gentle gradient.
    fn sky(width: usize, height: usize) -> Vec<f32> {
        let mut state = 11_u64;
        (0..width * height)
            .map(|index| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise = 6.0 * ((state >> 40) as f32 / (1_u32 << 24) as f32 - 0.5);
                100.0 + 0.02 * (index % width) as f32 + noise
            })
            .collect()
    }

    #[test]
    fn a_percentile_reads_between_order_statistics() {
        assert_eq!(percentile(&mut [4.0, 1.0, 3.0, 2.0, 5.0], 20.0), Some(1.8));
        assert_eq!(percentile(&mut [7.0], 20.0), Some(7.0));
        assert_eq!(percentile(&mut [], 20.0), None);
    }

    #[test]
    fn boxes_and_growth_cut_at_the_edges() {
        let sums = box_sums(&[1.0; 35], 7, 5, 1);
        assert_eq!((sums[0], sums[2 * 7 + 3], sums[34]), (4.0, 9.0, 4.0));
        let mut point = vec![false; 81];
        point[4] = true;
        let count = grown(&point, 9, 9, 2).iter().filter(|&&on| on).count();
        assert_eq!(count, 15, "cut at the top edge");
    }

    #[test]
    fn an_image_without_holes_is_read_as_it_is() {
        let (width, height) = (160, 128);
        let mut image = sky(width, height);
        // A bright star, and a dip the network takes out itself.
        for y in 50..60 {
            for x in 70..80 {
                image[y * width + x] += 40_000.0;
            }
        }
        image[90 * width + 30] -= 30.0;
        assert!(guarded(&image, width, height, &GUARD, &reading()).is_none());
    }

    #[test]
    fn a_hole_is_filled_kept_from_the_network_and_put_back() {
        let (width, height) = (160, 128);
        let mut image = sky(width, height);
        let hole = 60 * width + 80;
        image[hole] -= 3000.0;
        image[hole + 1] -= 2000.0;
        let guarded = guarded(&image, width, height, &GUARD, &reading()).unwrap();
        assert!(guarded.mask[hole] && guarded.mask[hole - 3 * width - 3]);
        assert!(!guarded.mask[hole - 4 * width], "the margin is 3 pixels");
        assert!(
            (guarded.read[hole] - 101.6).abs() < 3.0,
            "filled near the sky: {}",
            guarded.read[hole]
        );
        let unmasked: Vec<usize> = (0..image.len()).filter(|&i| !guarded.mask[i]).collect();
        assert!(unmasked.iter().all(|&i| guarded.read[i] == image[i]));

        // A network that takes out all the noise, leaving the gradient.
        let smooth = |index: usize| 100.0 + 0.02 * (index % width) as f32;
        let found: Vec<f32> = (0..image.len())
            .map(|index| smooth(index) - guarded.read[index])
            .collect();
        let residual = guarded_residual(&image, &guarded, &found, &GUARD);
        assert_eq!(residual[hole], 0.0, "the hole comes back as it was");
        assert_eq!(residual[hole + 1], 0.0);
        let margin = hole + 2 * width;
        assert!(
            (image[margin] + residual[margin] - smooth(margin)).abs() < 1e-3,
            "the margin's noise is taken out"
        );
        assert!(unmasked.iter().all(|&i| residual[i] == found[i]));
    }
}
