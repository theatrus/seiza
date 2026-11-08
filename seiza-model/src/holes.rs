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
use crate::noise::{background, image_map};
use crate::sky::upsample;
use crate::stats::median_f64;
use rayon::prelude::*;

/// Each block's floor is this percentile of its pixels, read between order
/// statistics as numpy reads it by default.
const FLOOR_PERCENTILE: f64 = 20.0;

/// What the guard found in an image.
pub(crate) enum Guard {
    /// No hole: the network reads the image as it is, with this image map.
    Clear(Vec<f32>),
    /// Holes, which the network reads filled.
    Holes(Guarded),
}

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

/// The image `guard` has the network read in place of `plane`. Where the image
/// map holds no noise, nothing is a hole.
pub(crate) fn guarded(
    plane: &[f32],
    width: usize,
    height: usize,
    guard: &HoleGuard,
    reading: &Reading,
) -> Guard {
    let noise_map = image_map(plane, width, height, reading);
    let floor = sky_floor(plane, width, height, reading.sky_block);
    let deep: Vec<bool> = plane
        .par_iter()
        .zip(floor.par_iter().zip(noise_map.par_iter()))
        .map(|(&value, (&floor, &noise))| {
            let noise = f64::from(noise);
            value.is_finite()
                && noise > 0.0
                && f64::from(floor) - f64::from(value) > guard.depth * noise
        })
        .collect();
    if !deep.contains(&true) {
        return Guard::Clear(noise_map);
    }
    let mask: Vec<bool> = grown(&deep, width, height, guard.margin)
        .into_iter()
        .zip(plane)
        .map(|(inside, value)| inside && value.is_finite())
        .collect();
    let read = filled(plane, &mask, width, height, guard.fill_radius, &floor);
    Guard::Holes(Guarded {
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
    let background = background(plane, width, height);
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
                    out[x.saturating_sub(radius)..=x.saturating_add(radius).min(width - 1)]
                        .fill(true);
                }
            }
        });
    let mut out = vec![false; mask.len()];
    out.par_chunks_mut(width).enumerate().for_each(|(y, line)| {
        for near in y.saturating_sub(radius)..=y.saturating_add(radius).min(height - 1) {
            for (pixel, &inside) in line.iter_mut().zip(&rows[near * width..][..width]) {
                *pixel |= inside;
            }
        }
    });
    out
}

/// `plane` with each masked pixel the median of the unmasked ones within
/// `radius`, cut at the image's edges, or the floor where there are none. A
/// star or a knot in the box barely moves it.
fn filled(
    plane: &[f32],
    mask: &[bool],
    width: usize,
    height: usize,
    radius: usize,
    floor: &[f32],
) -> Vec<f32> {
    let masked: Vec<usize> = (0..plane.len()).filter(|&index| mask[index]).collect();
    let fills: Vec<f32> = masked
        .par_iter()
        .map(|&index| {
            let (y, x) = (index / width, index % width);
            let mut around = Vec::new();
            for row in y.saturating_sub(radius)..y.saturating_add(radius + 1).min(height) {
                for column in x.saturating_sub(radius)..x.saturating_add(radius + 1).min(width) {
                    let at = row * width + column;
                    if !mask[at] && plane[at].is_finite() {
                        around.push(f64::from(plane[at]));
                    }
                }
            }
            median_f64(&mut around).map_or(floor[index], |median| median as f32)
        })
        .collect();
    let mut read = plane.to_vec();
    for (index, fill) in masked.into_iter().zip(fills) {
        read[index] = fill;
    }
    read
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
            noise_correlation: None,
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
    fn growth_cuts_at_the_edges() {
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
        let found = guarded(&image, width, height, &GUARD, &reading());
        assert!(matches!(found, Guard::Clear(_)));
    }

    #[test]
    fn a_map_without_noise_holds_no_hole() {
        let mut image = vec![100.0_f32; 96 * 96];
        image[40 * 96 + 40] = 0.0;
        let found = guarded(&image, 96, 96, &GUARD, &reading());
        assert!(matches!(found, Guard::Clear(_)));
    }

    #[test]
    fn a_star_beside_a_hole_does_not_lift_its_fill() {
        let (width, height) = (160, 128);
        let mut image = sky(width, height);
        for y in 0..height {
            for x in 0..width {
                let distance = (y as f32 - 60.0).powi(2) + (x as f32 - 88.0).powi(2);
                image[y * width + x] += 4000.0 * (-distance / (2.0 * 1.6 * 1.6)).exp();
            }
        }
        let hole = 60 * width + 80;
        image[hole] -= 3000.0;
        let Guard::Holes(guarded) = guarded(&image, width, height, &GUARD, &reading()) else {
            panic!("the hole is found");
        };
        assert!(
            (guarded.read[hole] - 101.6).abs() < 3.0,
            "the sky's level, not a plateau lit by the star: {}",
            guarded.read[hole]
        );
    }

    #[test]
    fn a_hole_is_filled_kept_from_the_network_and_put_back() {
        let (width, height) = (160, 128);
        let mut image = sky(width, height);
        let hole = 60 * width + 80;
        image[hole] -= 3000.0;
        image[hole + 1] -= 2000.0;
        let Guard::Holes(guarded) = guarded(&image, width, height, &GUARD, &reading()) else {
            panic!("the hole is found");
        };
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
