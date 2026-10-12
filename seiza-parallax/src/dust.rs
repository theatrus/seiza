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
//!
//! A [`DustMap`] is the same dust as an image a person can paint: white
//! where the dust is opaque, black where the sky is clear. One given to the
//! video takes the place of the counted map, and is taken at its word: a
//! star seen where it is opaque lies in front. The counted map can be
//! written out as one to start from.

use crate::light::LightImage;
use rayon::prelude::*;
use std::path::Path;

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

/// Dust that lets through this share of the light or less hides everything
/// behind it, so a star seen through a given map's opaque dust must lie in
/// front.
pub const OPAQUE: f32 = 0.05;

/// The finest cell a given map is kept at, image pixels: a painted map
/// needs no finer, and a full-size one would cost a quarter of the image.
const GIVEN_CELL: f64 = 4.0;

/// How far a map's shape may stray from the image's and still fit it.
const SHAPE_TOLERANCE: f64 = 0.01;

/// How far past 0 to 1 a float map's samples may stray, rounding, and still
/// be read as a map.
const SAMPLE_SLACK: f32 = 1e-3;

/// Dust as an image: its opacity, 0 where the sky is clear to 1 where
/// nothing behind shows, row by row. It spans the whole image at any size of
/// the same shape.
#[derive(Clone, Debug, PartialEq)]
pub struct DustMap {
    pub width: usize,
    pub height: usize,
    pub opacity: Vec<f32>,
}

impl DustMap {
    /// The map in the image at `path`, white opaque, its format chosen by
    /// its extension: PNG, JPEG, TIFF, FITS or XISF. A colour image gives
    /// each pixel's mean channel. Integer samples read as 0 to 1; float ones
    /// must already lie there, give or take a rounding.
    pub fn open(path: &Path) -> Result<Self, String> {
        let (width, height, opacity) =
            open_unit_image(path, "dust map", "0 (clear) and 1 (opaque)")?;
        Ok(Self {
            width,
            height,
            opacity,
        })
    }

    /// Write the map to `path`, white opaque, in the format its extension
    /// names: a 16-bit grayscale PNG or TIFF, or a 32-bit float FITS or XISF.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        self.check()?;
        let image = seiza_image_io::FloatImage::gray(self.width, self.height, self.opacity.clone());
        seiza_image_io::save(path, &image).map_err(|error| error.to_string())
    }

    /// Whether it is a map: a size, an opacity for every pixel, each 0 to 1.
    pub fn check(&self) -> Result<(), String> {
        if self.width == 0 || self.height == 0 {
            return Err("the dust map is empty".into());
        }
        if self.opacity.len() != self.width * self.height {
            return Err(format!(
                "the dust map has {} values for {}×{} pixels",
                self.opacity.len(),
                self.width,
                self.height
            ));
        }
        if !self
            .opacity
            .iter()
            .all(|opacity| (0.0..=1.0).contains(opacity))
        {
            return Err("the dust map's opacity must lie between 0 and 1".into());
        }
        Ok(())
    }
}

/// The gray image at `path`, its format chosen by its extension, as its
/// size and its values row by row, each 0 to 1. A colour image gives each
/// pixel's mean channel. Integer samples read as 0 to 1; float ones must
/// already lie there, give or take a rounding, or the error names it the
/// `what` and says what 0 and 1 mean (`range`).
pub(crate) fn open_unit_image(
    path: &Path,
    what: &str,
    range: &str,
) -> Result<(usize, usize, Vec<f32>), String> {
    let image = seiza_image_io::open(path).map_err(|error| error.to_string())?;
    let mut values = image.to_gray();
    let (least, most) = values.iter().fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(least, most), &value| (least.min(value), most.max(value)),
    );
    if !(least >= -SAMPLE_SLACK && most <= 1.0 + SAMPLE_SLACK) {
        return Err(format!(
            "the {what} {} runs from {least} to {most}; its values must lie between {range}",
            path.display()
        ));
    }
    for value in &mut values {
        *value = value.clamp(0.0, 1.0);
    }
    Ok((image.width, image.height, values))
}

/// Whether a map of `map` pixels has the shape of an image of `image`
/// pixels, near enough to span it.
pub(crate) fn same_shape(map: (usize, usize), image: (usize, usize)) -> bool {
    let shape = map.0 as f64 / map.1 as f64;
    let wanted = image.0 as f64 / image.1 as f64;
    (shape / wanted - 1.0).abs() <= SHAPE_TOLERANCE
}

/// The share of light a map's `opacity` lets through: [`LEAST`] where it
/// is opaque, all of it where clear, so a map read back gives the dust it
/// was written from.
fn transmission_of(opacity: f32) -> f32 {
    LEAST + (1.0 - LEAST) * (1.0 - opacity)
}

/// The opacity a map shows for `transmission`, as [`transmission_of`]
/// reads it.
fn opacity_of(transmission: f32) -> f32 {
    ((1.0 - transmission) / (1.0 - LEAST)).clamp(0.0, 1.0)
}

/// The share of light the dust lets through, over the image.
#[derive(Clone, Debug)]
pub struct Dust {
    /// The image's size, pixels.
    width: usize,
    height: usize,
    /// Cell size, image pixels.
    cell: f64,
    columns: usize,
    rows: usize,
    transmission: Vec<f32>,
    /// The power the share of stars seen, or of light, is raised to.
    opacity: f32,
    /// Stars per pixel where the sky is clear.
    clear_density: f32,
    /// Whether a map gave it, rather than the star counts.
    given: bool,
}

/// How many cells of `cell` pixels span `size`: the whole ones, the last
/// taking any remainder, so no cell is a sliver.
fn cells(size: usize, cell: f64) -> usize {
    ((size as f64 / cell).floor() as usize).max(1)
}

/// The pixels cell `index` of `count` spans along a side of `size`.
fn span(index: usize, count: usize, size: usize, cell: f64) -> (usize, usize) {
    let start = ((index as f64 * cell) as usize).min(size);
    let end = if index + 1 >= count {
        size
    } else {
        (((index + 1) as f64 * cell) as usize).min(size)
    };
    (start, end)
}

/// The area in pixels of cell `index` of a `columns` × `rows` grid of
/// `cell`-pixel cells over a `width` × `height` image.
fn cell_area(
    index: usize,
    (columns, rows): (usize, usize),
    (width, height): (usize, usize),
    cell: f64,
) -> f32 {
    let (left, right) = span(index % columns, columns, width, cell);
    let (top, bottom) = span(index / columns, rows, height, cell);
    ((right - left) * (bottom - top)).max(1) as f32
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
        let (columns, rows) = (cells(width, cell), cells(height, cell));
        let mut counts = vec![0.0_f32; columns * rows];
        for &(x, y) in stars {
            if !(x >= 0.0 && y >= 0.0 && x < width as f64 && y < height as f64) {
                continue;
            }
            let (column, row) = ((x / cell) as usize, (y / cell) as usize);
            counts[row.min(rows - 1) * columns + column.min(columns - 1)] += 1.0;
        }
        // Stars per pixel, the last cells taking what the whole cells leave.
        let mut density: Vec<f32> = (0..rows * columns)
            .map(|index| counts[index] / cell_area(index, (columns, rows), (width, height), cell))
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
            width,
            height,
            cell,
            columns,
            rows,
            transmission,
            opacity,
            clear_density: clear,
            given: false,
        })
    }

    /// The dust `map` draws over a `width` × `height` image, or why it
    /// cannot: a map of another shape. A map coarser than a few pixels is
    /// kept at its own size and smoothed between its pixels; a finer one is
    /// averaged into cells of a few pixels.
    pub fn from_map(map: &DustMap, width: usize, height: usize) -> Result<Self, String> {
        map.check()?;
        if width == 0 || height == 0 {
            return Err("the image is empty".into());
        }
        if !same_shape((map.width, map.height), (width, height)) {
            return Err(format!(
                "the dust map is {}×{}, not the shape of the {width}×{height} image",
                map.width, map.height
            ));
        }
        let cell = (width as f64 / map.width as f64).max(GIVEN_CELL);
        let (columns, rows) = (cells(width, cell), cells(height, cell));
        // Map pixels per image pixel, each way.
        let (across, down) = (
            map.width as f64 / width as f64,
            map.height as f64 / height as f64,
        );
        let transmission = (0..rows * columns)
            .into_par_iter()
            .map(|index| {
                let (left, right) = span(index % columns, columns, width, cell);
                let (top, bottom) = span(index / columns, rows, height, cell);
                // The map pixels whose centres the cell covers, or the one
                // under its centre when it covers none.
                let first = |start: usize, scale: f64| (start as f64 * scale - 0.5).ceil().max(0.0);
                let last = |end: usize, scale: f64, size: usize| {
                    ((end as f64 * scale - 0.5).ceil() - 1.0).min(size as f64 - 1.0)
                };
                let (mut x0, mut x1) = (first(left, across), last(right, across, map.width));
                let (mut y0, mut y1) = (first(top, down), last(bottom, down, map.height));
                if x1 < x0 {
                    let centre = ((left + right) as f64 / 2.0 * across).floor();
                    (x0, x1) = (
                        centre.min(map.width as f64 - 1.0),
                        centre.min(map.width as f64 - 1.0),
                    );
                }
                if y1 < y0 {
                    let centre = ((top + bottom) as f64 / 2.0 * down).floor();
                    (y0, y1) = (
                        centre.min(map.height as f64 - 1.0),
                        centre.min(map.height as f64 - 1.0),
                    );
                }
                let (mut sum, mut count) = (0.0_f32, 0.0_f32);
                for y in y0 as usize..=y1 as usize {
                    for x in x0 as usize..=x1 as usize {
                        sum += map.opacity[y * map.width + x];
                        count += 1.0;
                    }
                }
                transmission_of(sum / count)
            })
            .collect();
        Ok(Self {
            width,
            height,
            cell,
            columns,
            rows,
            transmission,
            opacity: 1.0,
            clear_density: 0.0,
            given: true,
        })
    }

    /// This dust as a `width` × `height` map over the image, white opaque.
    pub fn to_map(&self, width: usize, height: usize) -> DustMap {
        let (across, down) = (
            self.width as f64 / width.max(1) as f64,
            self.height as f64 / height.max(1) as f64,
        );
        let opacity = (0..width * height)
            .into_par_iter()
            .map(|index| {
                let (x, y) = ((index % width) as f64 + 0.5, (index / width) as f64 + 0.5);
                opacity_of(self.at(x * across, y * down))
            })
            .collect();
        DustMap {
            width,
            height,
            opacity,
        }
    }

    /// Whether the dust at image pixel `(x, y)` hides everything behind it.
    #[inline]
    pub fn is_opaque(&self, x: f64, y: f64) -> bool {
        self.at(x, y) <= OPAQUE
    }

    /// The image's size, pixels.
    pub fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Whether a map gave this dust, rather than the star counts.
    pub fn given(&self) -> bool {
        self.given
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
        let (columns, rows) = (cells(width, cell), cells(height, cell));
        let light: Vec<f32> = (0..rows)
            .into_par_iter()
            .flat_map_iter(|row| {
                (0..columns).map(move |column| {
                    let (left, right) = span(column, columns, width, cell);
                    let (top, bottom) = span(row, rows, height, cell);
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
        // The clear sky's count for the same pixels, the edge cells' size
        // and the smoothing alike.
        let mut expected: Vec<f32> = (0..rows * columns)
            .map(|index| {
                self.clear_density * cell_area(index, (columns, rows), (width, height), cell)
            })
            .collect();
        smooth(&mut expected, columns, rows, 2.0);
        let transmission = (0..rows * columns)
            .map(|index| {
                let (row, column) = (index / columns, index % columns);
                let counted = self.at((column as f64 + 0.5) * cell, (row as f64 + 0.5) * cell);
                // Only surroundings standing clearly above the black say
                // what is dark: over a gentle gradient, such as vignetting
                // toward a darkest corner, a hair's difference would read as
                // a globule.
                let dark = if around[index] - black > 0.05 * around[index] {
                    ((light[index] - black) / (around[index] - black)).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                // The darkness counts in full where half the clear sky's
                // stars or fewer show, and not at all where all of them do.
                let stars = (counts[index] / expected[index].max(f32::MIN_POSITIVE)).min(1.0);
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
    #[inline]
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
    #[inline]
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
    fn a_thin_last_cell_is_not_taken_for_dust() {
        // Clear sky everywhere, and a starless image dimming a tenth toward
        // its right edge, as vignetting does; widths that leave the last
        // cells a few pixels wide.
        for width in [1975, 1990, 2001, 2003, 2008, 2017] {
            let stars = field(width, 1500, 10.0, (-1e4, -1e4, 1.0), 1);
            let mut starless = LightImage::new(width, 1500);
            for (index, pixel) in starless.pixels.iter_mut().enumerate() {
                let x = (index % width) as f32 / width as f32;
                *pixel = [0.3 * (1.0 - 0.1 * x); 3];
            }
            let counted = Dust::from_star_counts(&stars, width, 1500, 3.0).unwrap();
            let edge = (width - 1) as f64;
            assert!(
                counted.at(edge, 750.0) > 0.8,
                "{width}: {}",
                counted.at(edge, 750.0)
            );
            assert!(
                counted.at(edge, 1499.0) > 0.8,
                "{width}: {}",
                counted.at(edge, 1499.0)
            );
            // The darkness adds nothing to what the counts say there.
            let before = counted.at(edge, 750.0);
            let dust = counted.with_darkness(&starless, &stars);
            for x in [edge, edge - 8.0] {
                assert!(
                    dust.at(x, 750.0) > before - 0.05,
                    "{width} at {x}: {}",
                    dust.at(x, 750.0)
                );
            }
        }
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

    /// A `width` × `height` map, opaque left of `edge`, a share of the
    /// width, and clear right of it.
    fn half_map(width: usize, height: usize, edge: f64) -> DustMap {
        DustMap {
            width,
            height,
            opacity: (0..width * height)
                .map(|index| {
                    if ((index % width) as f64 + 0.5) < edge * width as f64 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
        }
    }

    #[test]
    fn a_painted_map_of_any_size_covers_the_image() {
        for (width, height) in [(50, 30), (1000, 600), (333, 200)] {
            let dust = Dust::from_map(&half_map(width, height, 0.5), 1000, 600).unwrap();
            assert!(dust.given());
            assert_eq!(dust.size(), (1000, 600));
            // Opaque lets through next to nothing, and hides what is behind.
            assert!((dust.at(100.0, 300.0) - LEAST).abs() < 1e-6, "{width}");
            assert!(dust.is_opaque(100.0, 300.0));
            // Clear lets everything through.
            assert!((dust.at(900.0, 300.0) - 1.0).abs() < 1e-6, "{width}");
            assert!(!dust.is_opaque(900.0, 300.0));
        }
    }

    #[test]
    fn a_map_of_another_shape_is_refused() {
        let error = Dust::from_map(&half_map(50, 50, 0.5), 1000, 600).unwrap_err();
        assert!(error.contains("50×50"), "{error}");
        let mut short = half_map(10, 6, 0.5);
        short.opacity.pop();
        assert!(Dust::from_map(&short, 1000, 600).is_err());
        let mut bright = half_map(10, 6, 0.5);
        bright.opacity[0] = 1.5;
        assert!(Dust::from_map(&bright, 1000, 600).is_err());
    }

    #[test]
    fn a_fine_map_is_averaged_into_cells() {
        // Opaque and clear pixels in turn: the cells see half of each.
        let map = DustMap {
            width: 1000,
            height: 600,
            opacity: (0..600_000)
                .map(|index| ((index % 1000) % 2) as f32)
                .collect(),
        };
        let dust = Dust::from_map(&map, 1000, 600).unwrap();
        let half = transmission_of(0.5);
        assert!(
            (dust.at(500.0, 300.0) - half).abs() < 1e-3,
            "{}",
            dust.at(500.0, 300.0)
        );
    }

    #[test]
    fn counted_dust_written_out_and_read_back_is_the_same_dust() {
        let stars = field(2000, 1500, 10.0, (1000.0, 750.0, 300.0), 10);
        let counted = Dust::from_star_counts(&stars, 2000, 1500, 2.0).unwrap();
        let directory = tempfile::tempdir().unwrap();
        for extension in ["png", "tif", "fits", "xisf"] {
            let path = directory.path().join(format!("dust.{extension}"));
            counted.to_map(1000, 750).save(&path).unwrap();
            let read = DustMap::open(&path).unwrap();
            assert_eq!((read.width, read.height), (1000, 750), "{extension}");
            let again = Dust::from_map(&read, 2000, 1500).unwrap();
            for (x, y) in [
                (200.0, 200.0),
                (1000.0, 750.0),
                (1250.0, 750.0),
                (1999.0, 1499.0),
            ] {
                let (before, after) = (counted.at(x, y), again.at(x, y));
                assert!(
                    (before - after).abs() < 0.02,
                    "{extension} at {x},{y}: {before} then {after}"
                );
            }
        }
        assert!(!counted.is_opaque(200.0, 200.0));
    }

    #[test]
    fn a_float_map_outside_zero_to_one_is_refused() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("counts.fits");
        let image = seiza_image_io::FloatImage::gray(2, 1, vec![0.0, 40_000.0]);
        seiza_image_io::save(&path, &image).unwrap();
        let error = DustMap::open(&path).unwrap_err();
        assert!(error.contains("between 0"), "{error}");
        assert!(DustMap::open(&directory.path().join("missing.png")).is_err());
        let error = half_map(4, 2, 0.5)
            .save(&directory.path().join("map.jpg"))
            .unwrap_err();
        assert!(error.contains(".png"), "{error}");
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
