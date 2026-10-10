//! How much light the dust on the background plane lets through, from how
//! many stars shine through it.
//!
//! Dust hides the stars behind it, so where a deep image shows few stars
//! the dust is thick. Counted over cells, smoothed, and set against the
//! counts where the sky is clearest, the shortfall gives the share of light
//! that gets through. When the camera moves, a far star or galaxy slides
//! behind other dust than it was photographed through, and dims or
//! brightens by the ratio of the two. Dust thick enough to black out the
//! glow behind it shows too in the starless image's own darkness, finer
//! than the counts can see.

use crate::light::LightImage;
use rayon::prelude::*;

/// Light from behind the dust brightens at most this much as it slides out
/// from behind thicker dust: the star counts are noisy, and a star the
/// image barely showed should not blaze.
const MAX_BRIGHTENING: f32 = 1.5;

/// The least transmission the map gives: black dust lets through next to
/// nothing.
const LEAST: f32 = 0.01;

/// The darkness map's cell size, image pixels: fine enough for a small
/// black globule.
const FINE: f64 = 16.0;

/// The share of light the dust lets through, over the image.
#[derive(Clone, Debug)]
pub struct Dust {
    /// Cell size, image pixels.
    cell: f64,
    columns: usize,
    rows: usize,
    transmission: Vec<f32>,
    /// The power the share of stars seen, or of light, is raised to.
    opacity: f32,
    /// Stars per pixel where the sky is clear.
    clear_density: f32,
}

impl Dust {
    /// The dust in front of `stars`, the places of the stars seen behind it,
    /// over a `width` × `height` image, or `None` when there are too few to
    /// count. Cells are sized to hold a few dozen stars where the sky is
    /// clear; the counts are smoothed over a few cells and set against the
    /// densest tenth of the sky.
    ///
    /// The light let through is the share of stars seen to the power
    /// `opacity`. Dust that dims every star by a magnitude hides only the
    /// faintest of them, so the share of stars seen falls more slowly than
    /// the light does, and 1 understates the dust.
    pub fn from_star_counts(
        stars: &[(f64, f64)],
        width: usize,
        height: usize,
        opacity: f32,
    ) -> Option<Self> {
        if stars.len() < 500 || width == 0 || height == 0 {
            return None;
        }
        let area = width as f64 * height as f64;
        let cell = (30.0 * area / stars.len() as f64).sqrt().clamp(16.0, 512.0);
        let columns = (width as f64 / cell).ceil() as usize;
        let rows = (height as f64 / cell).ceil() as usize;
        let mut counts = vec![0.0_f32; columns * rows];
        for &(x, y) in stars {
            if !(x >= 0.0 && y >= 0.0 && x < width as f64 && y < height as f64) {
                continue;
            }
            let (column, row) = ((x / cell) as usize, (y / cell) as usize);
            counts[row.min(rows - 1) * columns + column.min(columns - 1)] += 1.0;
        }
        // Stars per pixel, with the image's edge cutting the last cells
        // short.
        let inside = |index: usize, size: usize, cells: usize| {
            if index + 1 < cells {
                cell
            } else {
                size as f64 - index as f64 * cell
            }
        };
        let mut density: Vec<f32> = (0..rows * columns)
            .map(|index| {
                let (row, column) = (index / columns, index % columns);
                let pixels = inside(column, width, columns) * inside(row, height, rows);
                counts[index] / pixels.max(1.0) as f32
            })
            .collect();
        smooth(&mut density, columns, rows, 1.5);
        let mut sorted = density.clone();
        let index = sorted.len() * 9 / 10;
        let (_, &mut clear, _) = sorted.select_nth_unstable_by(index, f32::total_cmp);
        if clear <= 0.0 {
            return None;
        }
        let transmission = density
            .iter()
            .map(|density| (density / clear).min(1.0).powf(opacity).max(LEAST))
            .collect();
        Some(Self {
            cell,
            columns,
            rows,
            transmission,
            opacity,
            clear_density: clear,
        })
    }

    /// This map on a finer grid, made darker where the starless image is
    /// darker than its surroundings and the stars run short: a globule
    /// thick enough to black out the light behind it, smaller than the
    /// star counts can see. A dark patch with its full share of stars is
    /// clear sky, not dust. `stars` are the places
    /// [`Self::from_star_counts`] counted.
    pub fn with_darkness(self, starless: &LightImage, stars: &[(f64, f64)]) -> Self {
        let (width, height) = (starless.width, starless.height);
        let cell = FINE.min(self.cell);
        let columns = (width as f64 / cell).ceil() as usize;
        let rows = (height as f64 / cell).ceil() as usize;
        let light: Vec<f32> = (0..rows)
            .into_par_iter()
            .flat_map_iter(|row| {
                (0..columns).map(move |column| {
                    let (left, top) = (
                        (column as f64 * cell) as usize,
                        (row as f64 * cell) as usize,
                    );
                    let right = (((column + 1) as f64 * cell) as usize).min(width);
                    let bottom = (((row + 1) as f64 * cell) as usize).min(height);
                    let mut sum = 0.0_f32;
                    for y in top..bottom {
                        for x in left..right {
                            let pixel = starless.at(x, y);
                            sum += (pixel[0] + pixel[1] + pixel[2]) / 3.0;
                        }
                    }
                    sum / ((right - left) * (bottom - top)).max(1) as f32
                })
            })
            .collect();
        // The light around each cell, a few hundred pixels across, and the
        // image's black: its darkest cell, the sky's own floor.
        let mut around = light.clone();
        smooth(&mut around, columns, rows, 6.0);
        let black = light.iter().copied().fold(f32::INFINITY, f32::min);
        // The share of the clear sky's stars within a few dozen pixels.
        let mut counts = vec![0.0_f32; columns * rows];
        for &(x, y) in stars {
            if x >= 0.0 && y >= 0.0 && x < width as f64 && y < height as f64 {
                let (column, row) = ((x / cell) as usize, (y / cell) as usize);
                counts[row.min(rows - 1) * columns + column.min(columns - 1)] += 1.0;
            }
        }
        smooth(&mut counts, columns, rows, 2.0);
        let expected = self.clear_density * (cell * cell) as f32;
        let transmission = (0..rows * columns)
            .map(|index| {
                let (row, column) = (index / columns, index % columns);
                let counted = self.at((column as f64 + 0.5) * cell, (row as f64 + 0.5) * cell);
                let dark = if around[index] > black {
                    ((light[index] - black) / (around[index] - black)).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                // The darkness counts in full where half the clear sky's
                // stars or fewer show, and not at all where all of them do.
                let stars = (counts[index] / expected).min(1.0);
                let belief = ((1.0 - stars) / 0.5).clamp(0.0, 1.0);
                let dark = 1.0 - belief * (1.0 - dark);
                counted.min(dark.powf(self.opacity)).max(LEAST)
            })
            .collect();
        Self {
            cell,
            columns,
            rows,
            transmission,
            ..self
        }
    }

    /// The share of light let through at image pixel `(x, y)`, between
    /// cell centres.
    pub fn at(&self, x: f64, y: f64) -> f32 {
        let fx = (x / self.cell - 0.5).clamp(0.0, (self.columns - 1) as f64);
        let fy = (y / self.cell - 0.5).clamp(0.0, (self.rows - 1) as f64);
        // Both are clamped at zero, so truncating rounds them down.
        let (column, row) = (fx as usize, fy as usize);
        let (next_column, next_row) = (
            (column + 1).min(self.columns - 1),
            (row + 1).min(self.rows - 1),
        );
        let (tx, ty) = ((fx - column as f64) as f32, (fy - row as f64) as f32);
        let value = |column: usize, row: usize| self.transmission[row * self.columns + column];
        let top = value(column, row) * (1.0 - tx) + value(next_column, row) * tx;
        let bottom = value(column, next_row) * (1.0 - tx) + value(next_column, next_row) * tx;
        top * (1.0 - ty) + bottom * ty
    }

    /// How light from behind the dust that was photographed through it at
    /// image pixel `seen` changes when it shows through it at `now`.
    pub fn change(&self, seen: (f64, f64), now: (f64, f64)) -> f32 {
        (self.at(now.0, now.1) / self.at(seen.0, seen.1)).min(MAX_BRIGHTENING)
    }

    /// The map as cells, row by row, and its width in cells: for showing it.
    pub fn cells(&self) -> (&[f32], usize) {
        (&self.transmission, self.columns)
    }
}

/// A Gaussian blur of `sigma` cells over a `columns` × `rows` grid, each
/// cell weighted by how much of the kernel lies on the grid, so the edges
/// do not darken.
fn smooth(values: &mut [f32], columns: usize, rows: usize, sigma: f64) {
    let radius = (3.0 * sigma).ceil() as isize;
    let kernel: Vec<f32> = (-radius..=radius)
        .map(|offset| (-(offset as f64).powi(2) / (2.0 * sigma * sigma)).exp() as f32)
        .collect();
    let pass = |values: &[f32], along: usize, across: usize, at: &dyn Fn(usize, usize) -> usize| {
        let mut out = vec![0.0_f32; values.len()];
        for line in 0..across {
            for position in 0..along {
                let (mut sum, mut weight) = (0.0_f32, 0.0_f32);
                for (offset, &k) in (-radius..=radius).zip(&kernel) {
                    let other = position as isize + offset;
                    if other < 0 || other >= along as isize {
                        continue;
                    }
                    sum += k * values[at(other as usize, line)];
                    weight += k;
                }
                out[at(position, line)] = sum / weight;
            }
        }
        out
    };
    let across = pass(values, columns, rows, &|column, row| row * columns + column);
    let down = pass(&across, rows, columns, &|row, column| {
        row * columns + column
    });
    values.copy_from_slice(&down);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stars on a jittered grid, `spacing` apart, thinned to one in `keep`
    /// inside a disc.
    fn field(
        width: usize,
        height: usize,
        spacing: f64,
        disc: (f64, f64, f64),
        keep: usize,
    ) -> Vec<(f64, f64)> {
        let mut stars = Vec::new();
        let mut index = 0_usize;
        let mut y = 0.5 * spacing;
        while y < height as f64 {
            let mut x = 0.5 * spacing;
            while x < width as f64 {
                index += 1;
                let jitter = ((index * 7919) % 97) as f64 / 97.0 - 0.5;
                let (sx, sy) = (x + jitter * spacing * 0.5, y - jitter * spacing * 0.3);
                let inside = (sx - disc.0).hypot(sy - disc.1) < disc.2;
                if !inside || index.is_multiple_of(keep) {
                    stars.push((sx, sy));
                }
                x += spacing;
            }
            y += spacing;
        }
        stars
    }

    #[test]
    fn thick_dust_shows_as_a_shortage_of_stars() {
        // A dark cloud letting through a tenth of the stars.
        let stars = field(2000, 1500, 10.0, (1000.0, 750.0, 300.0), 10);
        let dust = Dust::from_star_counts(&stars, 2000, 1500, 1.0).expect("enough stars");
        let clear = dust.at(200.0, 200.0);
        let thick = dust.at(1000.0, 750.0);
        assert!(clear > 0.9, "{clear}");
        assert!(thick < 0.25, "{thick}");
        // The edges of the image are not mistaken for dust.
        assert!(dust.at(1999.0, 1499.0) > 0.8, "{}", dust.at(1999.0, 1499.0));
        // Light seen through clear sky dims as it slides behind the cloud,
        // and brightens only so far the other way.
        assert!(dust.change((200.0, 200.0), (1000.0, 750.0)) < 0.3);
        assert_eq!(
            dust.change((1000.0, 750.0), (200.0, 200.0)),
            MAX_BRIGHTENING
        );
    }

    #[test]
    fn a_black_globule_blocks_what_the_counts_miss() {
        // Stars everywhere but in a black globule too small for the counts.
        let stars = field(2000, 1500, 10.0, (700.0, 500.0, 40.0), 1000);
        let mut starless = LightImage::new(2000, 1500);
        for (index, pixel) in starless.pixels.iter_mut().enumerate() {
            let (x, y) = ((index % 2000) as f64, (index / 2000) as f64);
            let glow = if (x - 700.0).hypot(y - 500.0) < 40.0 {
                0.002
            } else {
                0.3
            };
            *pixel = [glow; 3];
        }
        let counted = Dust::from_star_counts(&stars, 2000, 1500, 2.0).unwrap();
        // The counts barely see it.
        assert!(
            counted.at(700.0, 500.0) > 0.4,
            "{}",
            counted.at(700.0, 500.0)
        );
        let dust = counted.with_darkness(&starless, &stars);
        assert!(dust.at(700.0, 500.0) <= 0.05, "{}", dust.at(700.0, 500.0));
        assert!(dust.at(1200.0, 900.0) > 0.9, "{}", dust.at(1200.0, 900.0));
    }

    #[test]
    fn a_dark_patch_full_of_stars_is_clear_sky() {
        // Bright dust all around a dark gap with its full share of stars.
        let stars = field(2000, 1500, 10.0, (0.0, 0.0, 0.0), 1);
        let mut starless = LightImage::new(2000, 1500);
        for (index, pixel) in starless.pixels.iter_mut().enumerate() {
            let (x, y) = ((index % 2000) as f64, (index / 2000) as f64);
            let glow = if (x - 700.0).hypot(y - 500.0) < 150.0 {
                0.002
            } else {
                0.4
            };
            *pixel = [glow; 3];
        }
        let dust = Dust::from_star_counts(&stars, 2000, 1500, 2.0)
            .unwrap()
            .with_darkness(&starless, &stars);
        assert!(dust.at(700.0, 500.0) > 0.8, "{}", dust.at(700.0, 500.0));
    }

    #[test]
    fn too_few_stars_make_no_map() {
        assert!(Dust::from_star_counts(&[(1.0, 1.0); 10], 100, 100, 1.0).is_none());
    }

    #[test]
    fn more_opacity_darkens_the_same_shortage() {
        let stars = field(2000, 1500, 10.0, (1000.0, 750.0, 300.0), 4);
        let light = Dust::from_star_counts(&stars, 2000, 1500, 1.0).unwrap();
        let heavy = Dust::from_star_counts(&stars, 2000, 1500, 2.0).unwrap();
        let (a, b) = (light.at(1000.0, 750.0), heavy.at(1000.0, 750.0));
        assert!((b - a * a).abs() < 1e-3, "{a} {b}");
        assert!(heavy.at(200.0, 200.0) > 0.85);
    }
}
