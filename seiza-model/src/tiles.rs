//! Overlapping tiles: where they start, and how their outputs blend.
//!
//! Every run cuts an image into tiles, so memory stays fixed whatever its
//! size. Within `margin` of a tile's edge an output saw past the edge, into
//! the convolutions' zero padding, so it takes no weight there unless that
//! edge is the image's own. Past the margin, a tile's weight rises as a raised
//! cosine across the rest of the overlap.
//!
//! A network that halves the image samples it on a grid, so what it makes of a
//! pixel depends on where the tile holding it starts, unless every tile starts
//! on that grid. So the stride is cut to a multiple of it, and the image is
//! padded by reflection at its far edges until the last tile starts on it too.

use crate::error::ModelError;
use rayon::prelude::*;

/// How a model runs over an image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tiling {
    /// A tile's side in pixels.
    pub tile: usize,
    /// How far neighbouring tiles overlap, before the stride is cut to the
    /// grid.
    pub overlap: usize,
    /// How far in from an inner edge a tile takes no weight.
    pub margin: usize,
    /// Every tile starts on a multiple of this.
    pub multiple: usize,
}

/// Runs a network over one tile. Tests drive the tile loop through a closure,
/// so it needs no ONNX Runtime.
pub(crate) trait TileRunner {
    /// Fill `residual`, `side` × `side`, from `planes`, each of the same size,
    /// one after the other.
    fn run(&mut self, planes: &[f32], side: usize, residual: &mut [f32]) -> Result<(), ModelError>;
}

impl<F> TileRunner for F
where
    F: FnMut(&[f32], usize, &mut [f32]) -> Result<(), ModelError>,
{
    fn run(&mut self, planes: &[f32], side: usize, residual: &mut [f32]) -> Result<(), ModelError> {
        self(planes, side, residual)
    }
}

/// Tile starts along one axis: a stride of `tile - overlap`, with the last tile
/// flush with the far edge.
pub(crate) fn origins(length: usize, tile: usize, overlap: usize) -> Vec<usize> {
    if length <= tile {
        return vec![0];
    }
    let mut starts: Vec<usize> = (0..length - tile).step_by(tile - overlap).collect();
    starts.push(length - tile);
    starts
}

/// A tile's blending weight along one axis: zero within `margin` of an inner
/// edge, a raised cosine across the rest of the overlap, flat between. `first`
/// and `last` mark edges that are the image's own.
pub(crate) fn window(
    tile: usize,
    overlap: usize,
    margin: usize,
    first: bool,
    last: bool,
) -> Vec<f32> {
    let ramp_length = overlap - 2 * margin;
    let mut rising = vec![0.0_f32; margin];
    rising.extend((0..ramp_length).map(|index| {
        0.5 - 0.5 * (std::f32::consts::PI * (index as f32 + 0.5) / ramp_length as f32).cos()
    }));
    let mut weight = vec![1.0_f32; tile];
    if !first {
        weight[..rising.len()].copy_from_slice(&rising);
    }
    if !last {
        for (value, ramp) in weight[tile - rising.len()..]
            .iter_mut()
            .zip(rising.iter().rev())
        {
            *value = value.min(*ramp);
        }
    }
    weight
}

/// Where an index past the end of an axis of `length` falls, by reflection.
fn reflected(index: usize, length: usize) -> usize {
    if length == 1 {
        return 0;
    }
    let period = 2 * (length - 1);
    let folded = index % period;
    if folded < length {
        folded
    } else {
        period - folded
    }
}

/// How far an axis is padded: to a whole tile, or until its last tile starts on
/// the grid.
fn padding(length: usize, tile: usize, multiple: usize) -> usize {
    if length <= tile {
        tile - length
    } else {
        (length - tile).next_multiple_of(multiple) - (length - tile)
    }
}

impl Tiling {
    /// The overlap left once the stride is cut to the grid, which widens it,
    /// never narrows it.
    fn grid_overlap(&self) -> Result<usize, ModelError> {
        let Tiling {
            tile,
            overlap,
            margin,
            multiple,
        } = *self;
        if multiple == 0 || tile % multiple != 0 {
            return Err(ModelError::InvalidOption(format!(
                "a tile of {tile} is not a multiple of {multiple}"
            )));
        }
        let stride = tile.saturating_sub(overlap) / multiple * multiple;
        if stride < 1 {
            return Err(ModelError::InvalidOption(format!(
                "an overlap of {overlap} leaves no stride of {multiple} in a tile of {tile}"
            )));
        }
        let overlap = tile - stride;
        if overlap < 2 * margin + 1 {
            return Err(ModelError::InvalidOption(format!(
                "an overlap of {overlap} leaves no ramp past two margins of {margin}"
            )));
        }
        Ok(overlap)
    }

    /// How many tiles an image of this size takes.
    pub fn tiles(&self, width: usize, height: usize) -> Result<usize, ModelError> {
        let overlap = self.grid_overlap()?;
        let full = |length| length + padding(length, self.tile, self.multiple);
        Ok(origins(full(height), self.tile, overlap).len()
            * origins(full(width), self.tile, overlap).len())
    }
}

/// Run `runner` over overlapping tiles of `planes`, `channels` planes of
/// `width` × `height` one after the other, and blend the results, each scaled
/// by `scale`, into one plane. `stop` is asked before each tile, and `report`
/// told after it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn infer(
    runner: &mut dyn TileRunner,
    planes: &[f32],
    channels: usize,
    width: usize,
    height: usize,
    tiling: &Tiling,
    scale: f32,
    stop: &dyn Fn() -> bool,
    report: &mut dyn FnMut(usize, usize),
) -> Result<Vec<f32>, ModelError> {
    let tile = tiling.tile;
    let overlap = tiling.grid_overlap()?;
    let full_width = width + padding(width, tile, tiling.multiple);
    let full_height = height + padding(height, tile, tiling.multiple);
    let rows = origins(full_height, tile, overlap);
    let columns = origins(full_width, tile, overlap);
    let row_windows = windows(&rows, tile, overlap, tiling.margin);
    let column_windows = windows(&columns, tile, overlap, tiling.margin);

    // Tiles lie on a product grid, so the weights they lay down are the product
    // of each axis's sum, and no plane of them need be kept.
    let row_weight = weight_sums(&rows, &row_windows, tile, full_height);
    let column_weight = weight_sums(&columns, &column_windows, tile, full_width);
    let mut total = vec![0.0_f64; full_width * full_height];

    let source_x: Vec<usize> = (0..full_width).map(|x| reflected(x, width)).collect();
    let source_y: Vec<usize> = (0..full_height).map(|y| reflected(y, height)).collect();
    let mut input = vec![0.0_f32; channels * tile * tile];
    let mut residual = vec![0.0_f32; tile * tile];
    let count = rows.len() * columns.len();
    let mut done = 0;
    for (row, &top) in rows.iter().enumerate() {
        for (column, &left) in columns.iter().enumerate() {
            if stop() {
                return Err(ModelError::Cancelled);
            }
            input
                .par_chunks_mut(tile)
                .enumerate()
                .for_each(|(line, out)| {
                    let (channel, y) = (line / tile, line % tile);
                    let source =
                        &planes[channel * width * height + source_y[top + y] * width..][..width];
                    for (value, &x) in out.iter_mut().zip(&source_x[left..left + tile]) {
                        *value = source[x];
                    }
                });
            runner.run(&input, tile, &mut residual)?;
            if residual.iter().any(|value| !value.is_finite()) {
                return Err(ModelError::NonFiniteOutput { x: left, y: top });
            }
            let (down, across) = (&row_windows[row], &column_windows[column]);
            total[top * full_width..(top + tile) * full_width]
                .par_chunks_mut(full_width)
                .enumerate()
                .for_each(|(y, line)| {
                    if down[y] == 0.0 {
                        return;
                    }
                    let out = &mut line[left..left + tile];
                    let result = &residual[y * tile..(y + 1) * tile];
                    for ((sum, &value), &weight) in out.iter_mut().zip(result).zip(across) {
                        *sum += f64::from(value * scale * (down[y] * weight));
                    }
                });
            done += 1;
            report(done, count);
        }
    }

    let mut blended = vec![0.0_f32; width * height];
    blended
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, line)| {
            for (x, out) in line.iter_mut().enumerate() {
                let weight = (row_weight[y] * column_weight[x]).max(1e-12);
                *out = (total[y * full_width + x] / weight) as f32;
            }
        });
    Ok(blended)
}

fn windows(starts: &[usize], tile: usize, overlap: usize, margin: usize) -> Vec<Vec<f32>> {
    let last = starts.len() - 1;
    (0..starts.len())
        .map(|index| window(tile, overlap, margin, index == 0, index == last))
        .collect()
}

fn weight_sums(starts: &[usize], windows: &[Vec<f32>], tile: usize, length: usize) -> Vec<f64> {
    let mut sums = vec![0.0_f64; length];
    for (&start, window) in starts.iter().zip(windows) {
        for (sum, &weight) in sums[start..start + tile].iter_mut().zip(window) {
            *sum += f64::from(weight);
        }
    }
    sums
}

#[cfg(test)]
mod tests {
    use super::*;

    const TILING: Tiling = Tiling {
        tile: 64,
        overlap: 36,
        margin: 8,
        multiple: 16,
    };

    fn run(
        runner: &mut dyn TileRunner,
        planes: &[f32],
        width: usize,
        height: usize,
        tiling: &Tiling,
    ) -> Vec<f32> {
        infer(
            runner,
            planes,
            1,
            width,
            height,
            tiling,
            1.0,
            &|| false,
            &mut |_, _| {},
        )
        .unwrap()
    }

    #[test]
    fn tiles_cover_an_axis_and_end_flush_with_it() {
        assert_eq!(origins(100, 200, 10), vec![0]);
        assert_eq!(origins(160, 64, 16), vec![0, 48, 96]);
    }

    #[test]
    fn a_window_is_zero_in_the_margin_and_flat_at_the_image_s_own_edge() {
        let inner = window(64, 36, 8, false, false);
        assert!(inner[..8].iter().all(|&weight| weight == 0.0));
        assert!(inner[56..].iter().all(|&weight| weight == 0.0));
        assert_eq!(inner[32], 1.0);
        let first = window(64, 36, 8, true, false);
        assert_eq!(first[0], 1.0);
        assert_eq!(first[63], 0.0);
    }

    #[test]
    fn crossing_windows_sum_to_one_past_the_margins() {
        // The raised cosine's halves add to one, so a neighbour's weight fills
        // what this tile's gives up.
        let ramp = window(64, 36, 8, false, true);
        for offset in 0..20 {
            let sum = ramp[8 + offset] + ramp[8 + 19 - offset];
            assert!((sum - 1.0).abs() < 1e-5, "{offset}: {sum}");
        }
    }

    #[test]
    fn reflection_folds_an_index_back_without_repeating_the_edge() {
        let folded: Vec<usize> = (0..8).map(|index| reflected(index, 4)).collect();
        assert_eq!(folded, vec![0, 1, 2, 3, 2, 1, 0, 1]);
        assert_eq!(reflected(5, 1), 0);
    }

    #[test]
    fn a_stride_cut_to_the_grid_widens_the_overlap() {
        assert_eq!(TILING.grid_overlap().unwrap(), 48);
        let odd = Tiling { tile: 60, ..TILING };
        assert!(odd.grid_overlap().is_err(), "a tile off the grid");
        let tight = Tiling {
            tile: 48,
            overlap: 48,
            ..TILING
        };
        assert!(tight.grid_overlap().is_err(), "no stride left");
    }

    #[test]
    fn a_pixelwise_runner_blends_back_to_the_same_image() {
        let (width, height) = (200, 130);
        let image: Vec<f32> = (0..width * height)
            .map(|index| (index % 977) as f32)
            .collect();
        let mut double = |planes: &[f32], _: usize, residual: &mut [f32]| {
            for (out, value) in residual.iter_mut().zip(planes) {
                *out = 2.0 * value;
            }
            Ok(())
        };
        let blended = run(&mut double, &image, width, height, &TILING);
        for (index, (value, source)) in blended.iter().zip(&image).enumerate() {
            assert!(
                (value - 2.0 * source).abs() < 2e-3,
                "{index}: {value} against {source}"
            );
        }
    }

    #[test]
    fn an_image_smaller_than_a_tile_is_padded_and_cut_back() {
        let (width, height) = (40, 20);
        let image: Vec<f32> = (0..width * height).map(|index| index as f32).collect();
        let mut seen = 0;
        let mut copy = |planes: &[f32], side: usize, residual: &mut [f32]| {
            seen = side;
            residual.copy_from_slice(&planes[..side * side]);
            Ok(())
        };
        let blended = run(&mut copy, &image, width, height, &TILING);
        assert_eq!(seen, 64);
        assert_eq!(blended, image);
    }

    #[test]
    fn a_second_plane_reaches_the_runner_after_the_first() {
        let (width, height) = (64, 64);
        let mut planes = vec![1.0_f32; width * height];
        planes.extend(vec![3.0_f32; width * height]);
        let mut subtract = |planes: &[f32], side: usize, residual: &mut [f32]| {
            let (first, second) = planes.split_at(side * side);
            for ((out, a), b) in residual.iter_mut().zip(first).zip(second) {
                *out = b - a;
            }
            Ok(())
        };
        let blended = infer(
            &mut subtract,
            &planes,
            2,
            width,
            height,
            &TILING,
            0.5,
            &|| false,
            &mut |_, _| {},
        )
        .unwrap();
        assert!(blended.iter().all(|&value| value == 1.0), "scaled by 0.5");
    }

    #[test]
    fn a_stop_ends_the_run_and_a_bad_tile_is_refused() {
        let image = vec![0.0_f32; 200 * 200];
        let mut zero = |_: &[f32], _: usize, residual: &mut [f32]| {
            residual.fill(0.0);
            Ok(())
        };
        let stopped = infer(
            &mut zero,
            &image,
            1,
            200,
            200,
            &TILING,
            1.0,
            &|| true,
            &mut |_, _| {},
        );
        assert!(matches!(stopped, Err(ModelError::Cancelled)));
        let mut bad = |_: &[f32], _: usize, residual: &mut [f32]| {
            residual.fill(f32::NAN);
            Ok(())
        };
        let refused = infer(
            &mut bad,
            &image,
            1,
            200,
            200,
            &TILING,
            1.0,
            &|| false,
            &mut |_, _| {},
        );
        assert!(matches!(
            refused,
            Err(ModelError::NonFiniteOutput { x: 0, y: 0 })
        ));
    }

    #[test]
    fn the_tile_count_matches_what_a_run_visits() {
        let image = vec![0.0_f32; 200 * 130];
        let mut visits = 0;
        let mut zero = |_: &[f32], _: usize, residual: &mut [f32]| {
            residual.fill(0.0);
            visits += 1;
            Ok(())
        };
        run(&mut zero, &image, 200, 130, &TILING);
        assert_eq!(TILING.tiles(200, 130).unwrap(), visits);
    }
}
