//! Finding the stars in a star image.
//!
//! A star image holds nothing but stars on an almost black, almost noiseless
//! ground, so a threshold joins a crowded cluster's halos into one region.
//! Each star is instead a local peak of the lightly smoothed light: crowded
//! stars stay apart, and a saturated core, flat on top, is one star.

use crate::light::LightImage;
use rayon::prelude::*;

/// A star found in a star image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FoundStar {
    /// Light-weighted centroid, image pixels.
    pub x: f64,
    pub y: f64,
    /// Light above the background in a small window, summed over channels.
    pub flux: f64,
    /// Pixels at half the peak's light or more around it: a measure of the
    /// core's size, which grows with saturation.
    pub area: u32,
}

/// Stars in `light`, brightest first. A peak must stand `sigma` noise
/// levels above the background, outside the core of a stronger star.
pub fn find_stars(light: &LightImage, sigma: f32) -> Vec<FoundStar> {
    let (width, height) = (light.width, light.height);
    if width < 5 || height < 5 {
        return Vec::new();
    }
    let sum: Vec<f32> = light
        .pixels
        .par_iter()
        .map(|pixel| pixel[0] + pixel[1] + pixel[2])
        .collect();
    let smooth = binomial(&sum, width, height);
    let (background, noise) = median_and_noise(&smooth);
    let threshold = background + (sigma * noise).max(0.02);

    // Local maxima, row by row.
    let mut peaks: Vec<(usize, usize, f32)> = (2..height - 2)
        .into_par_iter()
        .flat_map_iter(|y| {
            let smooth = &smooth;
            (2..width - 2).filter_map(move |x| {
                let value = smooth[y * width + x];
                if value < threshold {
                    return None;
                }
                for dy in -1_isize..=1 {
                    for dx in -1_isize..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let neighbour =
                            smooth[(y as isize + dy) as usize * width + (x as isize + dx) as usize];
                        // Ties go to the first pixel in reading order, so a
                        // flat top yields its corner, not every pixel.
                        let earlier = dy < 0 || (dy == 0 && dx < 0);
                        if neighbour > value || (earlier && neighbour == value) {
                            return None;
                        }
                    }
                }
                Some((x, y, value))
            })
        })
        .collect();
    peaks.sort_by(|a, b| b.2.total_cmp(&a.2));

    // Strongest first, each peak measured and kept unless it lies in the
    // core of a stronger kept star: a saturated star's flat top has peaks
    // at several of its corners.
    const CELL: f64 = 64.0;
    let cell = |x: f64, y: f64| ((x / CELL).floor() as i64, (y / CELL).floor() as i64);
    let mut grid: std::collections::HashMap<(i64, i64), Vec<usize>> =
        std::collections::HashMap::new();
    let mut found: Vec<FoundStar> = Vec::new();
    let mut cores: Vec<f64> = Vec::new();
    let mut widest = 2.0_f64;
    for &(x, y, peak) in &peaks {
        // A star rises clearly above its surroundings; a ripple on a halo
        // does not, however high the halo is.
        let base = surroundings(&smooth, width, height, x, y, peak);
        let rise = peak - base;
        if rise < (sigma * noise).max(0.02).max(0.05 * base) {
            continue;
        }
        let (fx, fy) = (x as f64, y as f64);
        let span = (widest / CELL).ceil() as i64;
        let (cx, cy) = cell(fx, fy);
        let inside = (cy - span..=cy + span).any(|row| {
            (cx - span..=cx + span).any(|column| {
                grid.get(&(column, row)).is_some_and(|indices| {
                    indices.iter().any(|&index| {
                        (found[index].x - fx).hypot(found[index].y - fy) <= cores[index]
                    })
                })
            })
        });
        if inside {
            continue;
        }
        let star = measure(&sum, width, height, x, y, peak, base);
        let core = (star.area as f64 / std::f64::consts::PI).sqrt() + 2.0;
        widest = widest.max(core);
        grid.entry(cell(star.x, star.y))
            .or_default()
            .push(found.len());
        found.push(star);
        cores.push(core);
    }
    found.sort_by(|a, b| b.flux.total_cmp(&a.flux));
    found
}

/// The light around the peak at `(x, y)`: the median of a ring well clear
/// of its core. The core ends at the first ring, three pixels out at least,
/// that falls below the peak (a saturated star's flat top pushes it out),
/// and the surroundings are read at twice that radius plus two.
fn surroundings(smooth: &[f32], width: usize, height: usize, x: usize, y: usize, peak: f32) -> f32 {
    let ring_median = |radius: usize| -> Option<f32> {
        let steps = radius * 8;
        let mut ring: Vec<f32> = (0..steps)
            .filter_map(|step| {
                let angle = step as f64 / steps as f64 * std::f64::consts::TAU;
                let px = (x as f64 + radius as f64 * angle.cos()).round();
                let py = (y as f64 + radius as f64 * angle.sin()).round();
                (px >= 0.0 && py >= 0.0 && px < width as f64 && py < height as f64)
                    .then(|| smooth[py as usize * width + px as usize])
            })
            .collect();
        if ring.is_empty() {
            return None;
        }
        let middle = ring.len() / 2;
        Some(*ring.select_nth_unstable_by(middle, f32::total_cmp).1)
    };
    let core = (3..=100_usize)
        .find(|&radius| ring_median(radius).is_none_or(|median| median < peak * 0.999))
        .unwrap_or(100);
    ring_median(2 * core + 2).unwrap_or(0.0)
}

/// Centroid, flux and core size of the peak at `(x, y)` above `base`.
fn measure(
    sum: &[f32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    peak: f32,
    base: f32,
) -> FoundStar {
    let half = base + (peak - base) / 2.0;
    let ring_median = |(cx, cy): (usize, usize), radius: usize| -> f32 {
        let steps = (radius * 8).max(8);
        let mut ring: Vec<f32> = (0..steps)
            .filter_map(|step| {
                let angle = step as f64 / steps as f64 * std::f64::consts::TAU;
                let px = (cx as f64 + radius as f64 * angle.cos()).round();
                let py = (cy as f64 + radius as f64 * angle.sin()).round();
                (px >= 0.0 && py >= 0.0 && px < width as f64 && py < height as f64)
                    .then(|| sum[py as usize * width + px as usize])
            })
            .collect();
        if ring.is_empty() {
            return 0.0;
        }
        let middle = ring.len() / 2;
        *ring.select_nth_unstable_by(middle, f32::total_cmp).1
    };
    // The core reaches the first ring whose median falls below half the
    // peak above its surroundings; a median, so the neighbouring stars a
    // ring crosses in a crowded cluster do not carry it on.
    let core = |centre: (usize, usize)| {
        (1..=200_usize)
            .find(|&radius| ring_median(centre, radius) < half)
            .unwrap_or(200)
            - 1
    };
    // The peak of a saturated star sits on the edge of its flat top, so
    // the window moves to the centroid and the core is measured again.
    let (mut centre, mut position, mut total) = ((x, y), (x as f64, y as f64), 0.0);
    let mut radius = core(centre);
    for _ in 0..4 {
        let reach = radius.max(2) + 1;
        let (mut sx, mut sy, mut weight) = (0.0_f64, 0.0_f64, 0.0_f64);
        for py in centre.1.saturating_sub(reach)..=(centre.1 + reach).min(height - 1) {
            for px in centre.0.saturating_sub(reach)..=(centre.0 + reach).min(width - 1) {
                let value = (sum[py * width + px] - base).max(0.0) as f64;
                sx += value * px as f64;
                sy += value * py as f64;
                weight += value;
            }
        }
        if weight <= 0.0 {
            break;
        }
        position = (sx / weight, sy / weight);
        total = weight;
        let moved = (position.0.round() as usize, position.1.round() as usize);
        if moved == centre {
            break;
        }
        centre = moved;
        radius = core(centre);
    }
    let area = (std::f64::consts::PI * (radius as f64 + 0.5).powi(2)).round() as u32;
    FoundStar {
        x: position.0,
        y: position.1,
        flux: total,
        area: area.max(1),
    }
}

/// A 3x3 binomial blur: (1 2 1) across, then down.
fn binomial(values: &[f32], width: usize, height: usize) -> Vec<f32> {
    let across: Vec<f32> = values
        .par_chunks(width)
        .flat_map_iter(|row| {
            (0..width).map(move |x| {
                let left = row[x.saturating_sub(1)];
                let right = row[(x + 1).min(width - 1)];
                (left + 2.0 * row[x] + right) / 4.0
            })
        })
        .collect();
    (0..height)
        .into_par_iter()
        .flat_map_iter(|y| {
            let across = &across;
            let (up, down) = (y.saturating_sub(1), (y + 1).min(height - 1));
            (0..width).map(move |x| {
                (across[up * width + x] + 2.0 * across[y * width + x] + across[down * width + x])
                    / 4.0
            })
        })
        .collect()
}

/// The median of a sample of `values`, and the noise as a scaled median
/// absolute deviation.
fn median_and_noise(values: &[f32]) -> (f32, f32) {
    let step = (values.len() / 200_000).max(1);
    let mut sample: Vec<f32> = values.iter().step_by(step).copied().collect();
    let middle = sample.len() / 2;
    let median = *sample.select_nth_unstable_by(middle, f32::total_cmp).1;
    let mut deviations: Vec<f32> = sample.iter().map(|value| (value - median).abs()).collect();
    let mad = *deviations.select_nth_unstable_by(middle, f32::total_cmp).1;
    (median, 1.4826 * mad)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(width: usize, height: usize, stars: &[(f64, f64, f32, f64)]) -> LightImage {
        let mut image = LightImage::new(width, height);
        for (index, pixel) in image.pixels.iter_mut().enumerate() {
            let (x, y) = ((index % width) as f64, (index / width) as f64);
            let mut light = 0.003 * ((index * 7919 % 13) as f32 / 13.0);
            for &(sx, sy, peak, spread) in stars {
                let r2 = (x - sx).powi(2) + (y - sy).powi(2);
                light += peak * (-r2 / (2.0 * spread * spread)).exp() as f32;
            }
            // Saturated cores flatten at the display maximum's light.
            let light = light.min(3.0);
            *pixel = [light; 3];
        }
        image
    }

    #[test]
    fn crowded_and_saturated_stars_are_each_found_once() {
        // A saturated star with a wide halo, two faint stars inside the
        // halo, and a pair 6 pixels apart.
        // A canvas mostly of empty sky, as a star image is.
        let image = field(
            400,
            300,
            &[
                (40.0, 50.0, 30.0, 6.0),
                (40.0, 50.0, 0.5, 25.0),
                (62.0, 50.0, 0.8, 1.2),
                (40.0, 72.0, 0.8, 1.2),
                (95.0, 20.0, 1.0, 1.2),
                (101.0, 20.0, 1.0, 1.2),
            ],
        );
        let found = find_stars(&image, 5.0);
        let near = |x: f64, y: f64| {
            found
                .iter()
                .filter(|star| (star.x - x).hypot(star.y - y) < 2.0)
                .count()
        };
        assert_eq!(near(40.0, 50.0), 1, "{found:?}");
        assert_eq!(near(62.0, 50.0), 1, "{found:?}");
        assert_eq!(near(40.0, 72.0), 1, "{found:?}");
        assert_eq!(near(95.0, 20.0), 1, "{found:?}");
        assert_eq!(near(101.0, 20.0), 1, "{found:?}");
        // The saturated star comes first and has the largest core.
        assert!((found[0].x - 40.0).abs() < 1.0 && (found[0].y - 50.0).abs() < 1.0);
        assert!(found[0].area > found.iter().skip(1).map(|star| star.area).max().unwrap());
    }
}
