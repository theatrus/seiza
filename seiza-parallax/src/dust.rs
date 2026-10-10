//! How much light the dust on the background plane lets through, from how
//! many stars shine through it.
//!
//! Dust hides the stars behind it, so where a deep image shows few stars
//! the dust is thick. Counted over cells, smoothed, and set against the
//! counts where the sky is clearest, the shortfall gives the share of light
//! that gets through. When the camera moves, a far star or galaxy slides
//! behind other dust than it was photographed through, and dims or
//! brightens by the ratio of the two.

/// Light from behind the dust brightens at most this much as it slides out
/// from behind thicker dust: the star counts are noisy, and a star the
/// image barely showed should not blaze.
const MAX_BRIGHTENING: f32 = 1.5;

/// The least transmission the map gives: no stars at all in a cell may be
/// chance as well as dust.
const LEAST: f32 = 0.05;

/// The share of light the dust lets through, over the image.
#[derive(Clone, Debug)]
pub struct Dust {
    /// Cell size, image pixels.
    cell: f64,
    columns: usize,
    rows: usize,
    transmission: Vec<f32>,
}

impl Dust {
    /// The dust in front of `stars`, the places of the stars seen behind it,
    /// over a `width` × `height` image, or `None` when there are too few to
    /// count. Cells are sized to hold a few dozen stars where the sky is
    /// clear; the counts are smoothed over a few cells and set against the
    /// densest tenth of the sky.
    pub fn from_star_counts(stars: &[(f64, f64)], width: usize, height: usize) -> Option<Self> {
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
            .map(|density| (density / clear).clamp(LEAST, 1.0))
            .collect();
        Some(Self {
            cell,
            columns,
            rows,
            transmission,
        })
    }

    /// The share of light let through at image pixel `(x, y)`, between
    /// cell centres.
    pub fn at(&self, x: f64, y: f64) -> f32 {
        let fx = (x / self.cell - 0.5).clamp(0.0, (self.columns - 1) as f64);
        let fy = (y / self.cell - 0.5).clamp(0.0, (self.rows - 1) as f64);
        let (column, row) = (fx.floor() as usize, fy.floor() as usize);
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
        let dust = Dust::from_star_counts(&stars, 2000, 1500).expect("enough stars");
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
    fn too_few_stars_make_no_map() {
        assert!(Dust::from_star_counts(&[(1.0, 1.0); 10], 100, 100).is_none());
    }
}
