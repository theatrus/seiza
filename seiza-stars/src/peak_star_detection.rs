//! Star detection in a star-only image, as local peaks.
//!
//! A star image (a star mask, or StarXTerminator's stars image) holds
//! nothing but stars on an almost black, almost noiseless ground, so a
//! threshold joins a crowded cluster's halos into one region. Each star is
//! instead a local peak of the lightly smoothed light: crowded stars stay
//! apart, and a saturated core, flat on top, is one star.
//!
//! This finds where stars are and how much light each holds, for cutting
//! them out or matching them to a catalog; it does not measure their shape
//! for grading as the other detectors here do.

use rayon::prelude::*;

/// A star found as a peak.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PeakStar {
    /// Light-weighted centroid, image pixels.
    pub x: f64,
    pub y: f64,
    /// Light above the background in a small window, summed over channels.
    pub flux: f64,
    /// Pixels at half the peak's light or more around it: a measure of the
    /// core's size, which grows with saturation.
    pub area: u32,
}

/// Stars in `light`, one linear brightness value per pixel in rows of
/// `width`, brightest first. For a colour image, pass each pixel's channels
/// summed. A peak must stand `sigma` noise levels above the background, and
/// at least 0.02 above it, outside the core of a stronger star.
///
/// # Panics
///
/// If `light` is not `width * height` values.
pub fn find_peak_stars(light: &[f32], width: usize, height: usize, sigma: f32) -> Vec<PeakStar> {
    assert_eq!(
        light.len(),
        width * height,
        "light must be width * height values"
    );
    if width < 5 || height < 5 {
        return Vec::new();
    }
    // A NaN or infinite value (a float image's blank border) counts as no
    // light: a NaN passes every comparison's negation, and would be taken
    // for a peak whose core never ends.
    let finite: Vec<f32>;
    let sum = if light.iter().all(|value| value.is_finite()) {
        light
    } else {
        finite = light
            .iter()
            .map(|&value| if value.is_finite() { value } else { 0.0 })
            .collect();
        &finite
    };
    let smooth = binomial(sum, width, height);
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
    let mut found: Vec<PeakStar> = Vec::new();
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
        let star = measure(sum, width, height, x, y, peak, base);
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
) -> PeakStar {
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
    PeakStar {
        x: position.0,
        y: position.1,
        flux: total,
        area: area.max(1),
    }
}

/// `stars`, brightest first, with those inside a brighter star's saturated
/// core folded into it: a flat-topped core can show more than one peak.
/// Stars in the halo beyond it stay stars of their own.
pub fn fold_core_fragments(stars: Vec<PeakStar>) -> Vec<PeakStar> {
    let reach = |star: &PeakStar| 1.5 * (star.area as f64 / std::f64::consts::PI).sqrt() + 1.0;
    let mut kept: Vec<PeakStar> = Vec::with_capacity(stars.len());
    let mut grid: std::collections::HashMap<(i64, i64), Vec<usize>> =
        std::collections::HashMap::new();
    const CELL: f64 = 32.0;
    let cell = |x: f64, y: f64| ((x / CELL).floor() as i64, (y / CELL).floor() as i64);
    // The largest reach among kept stars bounds how far to look.
    let mut widest = 0.0_f64;
    for star in stars {
        let span = (widest / CELL).ceil() as i64;
        let (cx, cy) = cell(star.x, star.y);
        let inside = (cy - span..=cy + span).any(|row| {
            (cx - span..=cx + span).any(|column| {
                grid.get(&(column, row)).is_some_and(|indices| {
                    indices.iter().any(|&index| {
                        let brighter = &kept[index];
                        (brighter.x - star.x).hypot(brighter.y - star.y) <= reach(brighter)
                    })
                })
            })
        });
        if inside {
            continue;
        }
        widest = widest.max(reach(&star));
        grid.entry(cell(star.x, star.y))
            .or_default()
            .push(kept.len());
        kept.push(star);
    }
    kept
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

    /// Brightness summed over three equal channels.
    fn field(width: usize, height: usize, stars: &[(f64, f64, f32, f64)]) -> Vec<f32> {
        let mut image = vec![0.0_f32; width * height];
        for (index, pixel) in image.iter_mut().enumerate() {
            let (x, y) = ((index % width) as f64, (index / width) as f64);
            let mut light = 0.003 * ((index * 7919 % 13) as f32 / 13.0);
            for &(sx, sy, peak, spread) in stars {
                let r2 = (x - sx).powi(2) + (y - sy).powi(2);
                light += peak * (-r2 / (2.0 * spread * spread)).exp() as f32;
            }
            // Saturated cores flatten at the display maximum's light.
            let light = light.min(3.0);
            *pixel = 3.0 * light;
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
        let found = find_peak_stars(&image, 400, 300, 5.0);
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

    #[test]
    fn a_nan_pixel_is_dark_and_hides_no_star() {
        let stars = [
            (100.0, 100.0, 1.0, 1.2),
            (200.0, 150.0, 1.0, 1.2),
            (300.0, 200.0, 1.0, 1.2),
            (60.0, 250.0, 1.0, 1.2),
        ];
        let clean = find_peak_stars(&field(400, 300, &stars), 400, 300, 5.0);
        let mut image = field(400, 300, &stars);
        image[150 * 400 + 230] = f32::NAN;
        image[10 * 400 + 10] = f32::INFINITY;
        let found = find_peak_stars(&image, 400, 300, 5.0);
        assert_eq!(found.len(), clean.len(), "{found:?}");
        assert_eq!(found.len(), 4, "{found:?}");
    }

    #[test]
    fn peaks_in_a_saturated_core_fold_into_it() {
        let star = |x: f64, y: f64, flux: f64, area: u32| PeakStar { x, y, flux, area };
        // A saturated core of area 314 (radius 10) reaches 16 pixels: a
        // second peak on it folds in, a star in the halo beyond does not.
        let folded = fold_core_fragments(vec![
            star(100.0, 100.0, 1e6, 314),
            star(108.0, 100.0, 1e3, 4),
            star(120.0, 100.0, 1e3, 4),
            star(150.0, 100.0, 1e3, 4),
        ]);
        let places: Vec<(f64, f64)> = folded.iter().map(|star| (star.x, star.y)).collect();
        assert_eq!(places, vec![(100.0, 100.0), (120.0, 100.0), (150.0, 100.0)]);
    }
}
