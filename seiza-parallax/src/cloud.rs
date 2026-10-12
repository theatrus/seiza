//! Dust clouds: the dust the star counts find, in clouds that each lie at
//! one distance.
//!
//! A dark nebula is the thickest part of a cloud of dust that reaches out
//! around it, glowing faintly where starlight falls on it and hiding the
//! stars behind it. The cloud and its dark nebulae lie at one depth, so a
//! dark nebula moves with the cloud it lies in: with the nebula when the
//! cloud holds it, else with the whole cloud on a plane of its own.
//!
//! A cloud's distance is where the stars seen through it run short. Gaia
//! counts the stars to its magnitude limit, however brightly the image
//! glows, and gives their distances. In front of a cloud there are as many
//! stars at each distance as over clear sky; behind it, fewer. Published
//! distances to dark clouds are measured the same way.

use crate::dust::{Dust, smooth};
use crate::plane::Region;

/// Dust that the star counts find lets through this share of the light or
/// less belongs to a cloud: the faint dust joining dark nebulae to the
/// clouds around them as well as the thick.
const DUSTY: f32 = 0.7;

/// The core of a cloud, whose stars place it, lets through this share of
/// the light or less: past it, the cloud's stars are mostly those of the
/// thin dust about it, too few hidden to show where it lies.
const THICK: f32 = 0.3;

/// Sky the star counts find letting through this share of the light or
/// more is clear.
const CLEAR: f32 = 0.9;

/// The cells a cloud's region fades out over past its edge.
const SOFTEN: f64 = 0.7;

/// No cell holds a cloud.
const NONE: usize = usize::MAX;

/// The dust clouds over an image: the cells of the star counts' dust map
/// thick enough to count, joined where they touch, and the clear sky.
pub(crate) struct Clouds {
    /// The image's size, pixels.
    width: usize,
    height: usize,
    /// Cell size, image pixels, and the grid's size.
    cell: f64,
    columns: usize,
    rows: usize,
    /// Each cell's cloud, or [`NONE`].
    labels: Vec<usize>,
    /// Each cell's pixels: left, top, right and bottom.
    boxes: Vec<(usize, usize, usize, usize)>,
    /// Whether each cell's sky is clear.
    clear: Vec<bool>,
    /// Whether each cell lies in its cloud's core.
    core: Vec<bool>,
    count: usize,
}

impl Clouds {
    /// The clouds of `dust`, as the star counts map it: cells that let
    /// through [`DUSTY`] or less, joined to their eight neighbours.
    pub(crate) fn find(dust: &Dust) -> Self {
        let (cells, columns) = dust.cells();
        let rows = cells.len() / columns;
        let (width, height) = dust.size();
        let mut labels = vec![NONE; cells.len()];
        let mut count = 0;
        let mut pending = Vec::new();
        for start in 0..cells.len() {
            if labels[start] != NONE || cells[start] > DUSTY {
                continue;
            }
            labels[start] = count;
            pending.push(start);
            while let Some(index) = pending.pop() {
                let (column, row) = ((index % columns) as isize, (index / columns) as isize);
                for (dx, dy) in [
                    (-1, -1),
                    (0, -1),
                    (1, -1),
                    (-1, 0),
                    (1, 0),
                    (-1, 1),
                    (0, 1),
                    (1, 1),
                ] {
                    let (x, y) = (column + dx, row + dy);
                    if x < 0 || y < 0 || x >= columns as isize || y >= rows as isize {
                        continue;
                    }
                    let other = y as usize * columns + x as usize;
                    if labels[other] == NONE && cells[other] <= DUSTY {
                        labels[other] = count;
                        pending.push(other);
                    }
                }
            }
            count += 1;
        }
        // Each cloud's core: its cells at or below THICK, else its
        // thickest, halfway from its thickest cell to its edge.
        let mut thickest = vec![f32::INFINITY; count];
        for (&label, &passed) in labels.iter().zip(cells) {
            if label != NONE {
                thickest[label] = thickest[label].min(passed);
            }
        }
        let core = labels
            .iter()
            .zip(cells)
            .map(|(&label, &passed)| {
                label != NONE && passed <= THICK.max((thickest[label] + DUSTY) / 2.0)
            })
            .collect();
        Self {
            width,
            height,
            cell: dust.cell_size(),
            columns,
            rows,
            labels,
            boxes: (0..cells.len()).map(|index| dust.cell_box(index)).collect(),
            clear: cells.iter().map(|&passed| passed >= CLEAR).collect(),
            core,
            count,
        }
    }

    /// How many clouds there are.
    pub(crate) fn len(&self) -> usize {
        self.count
    }

    /// The cell under image point `(x, y)`, if it lies on the image.
    fn cell_at(&self, x: f64, y: f64) -> Option<usize> {
        if !(x >= 0.0 && y >= 0.0 && x < self.width as f64 && y < self.height as f64) {
            return None;
        }
        let column = ((x / self.cell) as usize).min(self.columns - 1);
        let row = ((y / self.cell) as usize).min(self.rows - 1);
        Some(row * self.columns + column)
    }

    /// The cloud under image point `(x, y)`, if any.
    pub(crate) fn at(&self, x: f64, y: f64) -> Option<usize> {
        self.cell_at(x, y)
            .map(|index| self.labels[index])
            .filter(|&label| label != NONE)
    }

    /// The cloud under image point `(x, y)`, else the nearest within
    /// `reach` cells of it.
    pub(crate) fn near(&self, (x, y): (f64, f64), reach: usize) -> Option<usize> {
        let index = self.cell_at(x, y)?;
        let (column, row) = (index % self.columns, index / self.columns);
        let mut best: Option<(usize, usize)> = None;
        for other_row in row.saturating_sub(reach)..=(row + reach).min(self.rows - 1) {
            for other_column in
                column.saturating_sub(reach)..=(column + reach).min(self.columns - 1)
            {
                let label = self.labels[other_row * self.columns + other_column];
                let away = other_row.abs_diff(row).pow(2) + other_column.abs_diff(column).pow(2);
                if label != NONE && best.is_none_or(|(_, nearest)| away < nearest) {
                    best = Some((label, away));
                }
            }
        }
        best.map(|(label, _)| label)
    }

    /// The cloud whose core lies under image point `(x, y)`, if any.
    pub(crate) fn core_at(&self, x: f64, y: f64) -> Option<usize> {
        self.cell_at(x, y)
            .filter(|&index| self.core[index])
            .map(|index| self.labels[index])
    }

    /// Whether the sky at image point `(x, y)` is clear.
    pub(crate) fn is_clear(&self, x: f64, y: f64) -> bool {
        self.cell_at(x, y).is_some_and(|index| self.clear[index])
    }

    /// The cloud most of `region` lies in, if a quarter of it or more lies
    /// in one.
    pub(crate) fn holding(&self, region: &Region) -> Option<usize> {
        let step = ((self.cell / 4.0) as usize).max(1);
        let mut shares = vec![0.0_f64; self.count];
        let mut total = 0.0;
        for y in (region.top..region.top + region.height).step_by(step) {
            for x in (region.left..region.left + region.width).step_by(step) {
                let share = region.at(x, y) as f64;
                if share <= 0.0 {
                    continue;
                }
                total += share;
                if let Some(label) = self.at(x as f64 + 0.5, y as f64 + 0.5) {
                    shares[label] += share;
                }
            }
        }
        let (label, &most) = shares
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))?;
        (total > 0.0 && most >= 0.25 * total).then_some(label)
    }

    /// The pixels cloud `label` covers.
    pub(crate) fn area(&self, label: usize) -> f64 {
        self.cells_area(|index| self.labels[index] == label)
    }

    /// The pixels cloud `label`'s core covers.
    pub(crate) fn core_area(&self, label: usize) -> f64 {
        self.cells_area(|index| self.core[index] && self.labels[index] == label)
    }

    /// The pixels the cells `chosen` cover.
    fn cells_area(&self, chosen: impl Fn(usize) -> bool) -> f64 {
        self.boxes
            .iter()
            .enumerate()
            .filter(|(index, _)| chosen(*index))
            .map(|(_, &(left, top, right, bottom))| ((right - left) * (bottom - top)) as f64)
            .sum()
    }

    /// The pixels of clear sky.
    pub(crate) fn clear_area(&self) -> f64 {
        self.cells_area(|index| self.clear[index])
    }

    /// Cloud `label` as a region of the image, fading out over a cell or
    /// so past its edge.
    pub(crate) fn region(&self, label: usize) -> Option<Region> {
        let mut inside: Vec<f32> = self
            .labels
            .iter()
            .map(|&other| if other == label { 1.0 } else { 0.0 })
            .collect();
        smooth(&mut inside, self.columns, self.rows, SOFTEN);
        // The cells it reaches, and the pixels they span.
        let (mut low, mut high) = ((usize::MAX, usize::MAX), (0, 0));
        for (index, &share) in inside.iter().enumerate() {
            if share >= 0.01 {
                let (column, row) = (index % self.columns, index / self.columns);
                low = (low.0.min(column), low.1.min(row));
                high = (high.0.max(column), high.1.max(row));
            }
        }
        if low.0 == usize::MAX {
            return None;
        }
        let (left, top, _, _) = self.boxes[low.1 * self.columns + low.0];
        let (_, _, right, bottom) = self.boxes[high.1 * self.columns + high.0];
        let sample = |x: f64, y: f64| -> f32 {
            let fx = (x / self.cell - 0.5).clamp(0.0, (self.columns - 1) as f64);
            let fy = (y / self.cell - 0.5).clamp(0.0, (self.rows - 1) as f64);
            let (column, row) = (fx as usize, fy as usize);
            let (next_column, next_row) = (
                (column + 1).min(self.columns - 1),
                (row + 1).min(self.rows - 1),
            );
            let (tx, ty) = ((fx - column as f64) as f32, (fy - row as f64) as f32);
            let at = |column: usize, row: usize| inside[row * self.columns + column];
            let upper = at(column, row) * (1.0 - tx) + at(next_column, row) * tx;
            let lower = at(column, next_row) * (1.0 - tx) + at(next_column, next_row) * tx;
            upper * (1.0 - ty) + lower * ty
        };
        let coverage = (top..bottom)
            .flat_map(|y| {
                (left..right).map(move |x| {
                    let share = sample(x as f64 + 0.5, y as f64 + 0.5);
                    if share < 0.01 { 0.0 } else { share.min(1.0) }
                })
            })
            .collect();
        Region {
            left,
            top,
            width: right - left,
            height: bottom - top,
            coverage,
        }
        .trimmed((self.width, self.height))
    }
}

/// A cloud's distance as the star counts find it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CountedDistance {
    /// Parsecs.
    pub distance_pc: f64,
    /// The nearest and farthest the counts allow.
    pub lower_pc: f64,
    pub upper_pc: f64,
    /// The stars seen in front of it.
    pub in_front: usize,
    /// The share of the stars behind it that show, against those in front,
    /// each against the clear sky's.
    pub passed: f64,
}

/// The fewest stars in front of a cloud that place it.
const LEAST_IN_FRONT: usize = 5;

/// Behind a cloud, at most this share of the stars may show, against
/// those in front of it.
const MOST_PASSED: f64 = 0.7;

/// In front of a cloud, its stars must be at least this share as many for
/// their area as the clear sky's: fewer, and they lie behind it too.
const LEAST_IN_FRONT_SHARE: f64 = 0.5;

/// How much likelier, as a natural log, the counts must be with a cloud at
/// some distance than with none.
const LEAST_GAIN: f64 = 12.0;

/// Distances whose likelihood lies within this, as a natural log, of the
/// best one's are allowed: wider than for a smooth likelihood, as a split's
/// rises and falls from star to star.
const WITHIN: f64 = 4.0;

/// Gaia's distances to the stars counted are good to about this share,
/// so a cloud's range is never narrower.
const PRECISION: f64 = 0.1;

/// The distance of a cloud from the distances of the stars seen through it,
/// `inside`, against those of the stars over clear sky, `clear`, whose area
/// is `area` times the cloud's, or `None` when the counts show no cloud or
/// cannot place it.
///
/// Inside a cloud, the stars in front of it are as many, for each star of
/// the clear sky at the same distances, as the stars behind it would be
/// without it. The cloud lies where that share falls, and the counts are
/// split there where that is likeliest. The split compares shares, so the
/// sky's stars thinning across the image, as they do away from the Milky
/// Way, does not show as a cloud; but the stars in front must be near as
/// many for their area as the clear sky's, or the cloud lies in front of
/// them too. The cloud lies past the last star in front of it, by about as
/// far as the next star in front would most likely have come.
pub(crate) fn counted_distance(
    inside: &[f64],
    clear: &[f64],
    area: f64,
) -> Option<CountedDistance> {
    let sorted = |values: &[f64]| {
        let mut values: Vec<f64> = values
            .iter()
            .copied()
            .filter(|distance| distance.is_finite() && *distance > 0.0)
            .collect();
        values.sort_by(f64::total_cmp);
        values
    };
    let (inside, clear) = (sorted(inside), sorted(clear));
    let (stars, sky) = (inside.len(), clear.len());
    if stars < 2 * LEAST_IN_FRONT || sky < 2 * LEAST_IN_FRONT {
        return None;
    }
    // Each split: the stars in front, the clear sky's stars no farther,
    // how much likelier the counts are split there than not, and the share
    // that shows behind.
    struct Split {
        in_front: usize,
        clear_in_front: usize,
        gain: f64,
        passed: f64,
    }
    let mut splits = Vec::new();
    let mut clear_in_front = 0;
    for in_front in 1..=stars {
        // A split falls between two distances.
        if in_front < stars && inside[in_front] == inside[in_front - 1] {
            continue;
        }
        let last = inside[in_front - 1];
        while clear_in_front < sky && clear[clear_in_front] <= last {
            clear_in_front += 1;
        }
        if clear_in_front == 0 || clear_in_front == sky {
            continue;
        }
        let share = clear_in_front as f64 / sky as f64;
        let front = in_front as f64 / (stars as f64 * share);
        let behind = (stars - in_front) as f64 / (stars as f64 * (1.0 - share));
        let for_area = in_front as f64 * area / clear_in_front as f64;
        if behind >= front || for_area < LEAST_IN_FRONT_SHARE {
            continue;
        }
        let term = |count: usize, ratio: f64| {
            if count == 0 {
                0.0
            } else {
                count as f64 * ratio.ln()
            }
        };
        splits.push(Split {
            in_front,
            clear_in_front,
            gain: term(in_front, front) + term(stars - in_front, behind),
            passed: behind / front,
        });
    }
    let placed: Vec<&Split> = splits
        .iter()
        .filter(|split| split.in_front >= LEAST_IN_FRONT)
        .collect();
    let best = placed.iter().max_by(|a, b| a.gain.total_cmp(&b.gain))?;
    if best.gain < LEAST_GAIN || best.passed > MOST_PASSED {
        return None;
    }
    // Where the next star in front would have come, `expected` stars on
    // from the last, by the clear sky's stars past it: never past the next
    // star seen, which lies behind the cloud.
    let reach = |split: &Split, expected: f64| {
        let per_clear = split.in_front as f64 / split.clear_in_front as f64;
        let steps = ((expected / per_clear).ceil() as usize).max(1);
        let last = inside[split.in_front - 1];
        let at = clear
            .get(split.clear_in_front + steps - 1)
            .copied()
            .unwrap_or(clear[sky - 1])
            .max(last);
        match inside.get(split.in_front) {
            Some(&next) => at.min(next),
            None => at,
        }
    };
    let allowed: Vec<&&Split> = placed
        .iter()
        .filter(|split| split.gain >= best.gain - WITHIN)
        .collect();
    let (nearest, farthest) = (allowed.first()?, allowed.last()?);
    Some(CountedDistance {
        distance_pc: reach(best, std::f64::consts::LN_2),
        lower_pc: inside[nearest.in_front - 1] * (1.0 - PRECISION),
        upper_pc: reach(farthest, 3.0) * (1.0 + PRECISION),
        in_front: best.in_front,
        passed: best.passed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small, repeatable random number generator: xorshift.
    struct Random(u64);

    impl Random {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// `count` stars spread evenly through a cone out to `far` parsecs.
    fn stars(random: &mut Random, count: usize, far: f64) -> Vec<f64> {
        (0..count).map(|_| far * random.next().cbrt()).collect()
    }

    #[test]
    fn a_cloud_lies_where_the_stars_run_short() {
        let mut random = Random(7);
        let clear = stars(&mut random, 3000, 1000.0);
        // A cloud at 300 pc lets a fifth of the stars behind it show.
        let inside: Vec<f64> = stars(&mut random, 1500, 1000.0)
            .into_iter()
            .filter(|&distance| distance < 300.0 || random.next() < 0.2)
            .collect();
        let found = counted_distance(&inside, &clear, 2.0).expect("a cloud");
        assert!((found.distance_pc - 300.0).abs() < 30.0, "{found:?}");
        assert!(found.lower_pc <= found.distance_pc && found.distance_pc <= found.upper_pc);
        assert!(
            found.lower_pc > 230.0 && found.upper_pc < 380.0,
            "{found:?}"
        );
        assert!((found.passed - 0.2).abs() < 0.1, "{found:?}");
    }

    #[test]
    fn stars_as_many_as_the_clear_skys_show_no_cloud() {
        let mut random = Random(11);
        let clear = stars(&mut random, 3000, 1000.0);
        let inside = stars(&mut random, 1500, 1000.0);
        assert_eq!(counted_distance(&inside, &clear, 2.0), None);
        // Fewer at every distance, as where the image cuts the area short.
        let thinned: Vec<f64> = inside.into_iter().filter(|_| random.next() < 0.3).collect();
        assert_eq!(counted_distance(&thinned, &clear, 2.0), None);
    }

    #[test]
    fn a_cloud_with_too_few_stars_in_front_is_not_placed() {
        let mut random = Random(13);
        let clear = stars(&mut random, 3000, 1000.0);
        // So near that hardly a star lies in front.
        let inside: Vec<f64> = stars(&mut random, 1500, 1000.0)
            .into_iter()
            .filter(|&distance| distance < 60.0 || random.next() < 0.2)
            .collect();
        assert!(
            inside.iter().filter(|&&distance| distance < 60.0).count() < LEAST_IN_FRONT,
            "the test's cloud has too many stars in front"
        );
        assert_eq!(counted_distance(&inside, &clear, 2.0), None);
    }
}
