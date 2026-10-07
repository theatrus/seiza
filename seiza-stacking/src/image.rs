use crate::{Error, ReferenceRegion, Result};
use rayon::prelude::*;
use seiza_fits::{BayerPattern, debayer_rgb_f32_rows};

/// A row-major, interleaved linear image with one or three channels.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearImage {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Channel count: 1 for mono, 3 for interleaved RGB.
    pub channels: usize,
    /// Row-major, channel-interleaved samples.
    pub data: Vec<f32>,
}

impl LinearImage {
    /// Build an image, checking that the sample count matches the dimensions
    /// and that the channel count is 1 or 3.
    pub fn new(width: usize, height: usize, channels: usize, data: Vec<f32>) -> Result<Self> {
        if width == 0 || height == 0 || !matches!(channels, 1 | 3) {
            return Err(Error::InvalidImage(
                "dimensions must be non-zero and channels must be 1 or 3".into(),
            ));
        }
        let expected = width
            .checked_mul(height)
            .and_then(|value| value.checked_mul(channels))
            .ok_or_else(|| Error::InvalidImage("image dimensions overflow".into()))?;
        if data.len() != expected {
            return Err(Error::InvalidImage(format!(
                "pixel buffer has {} samples; expected {expected}",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            channels,
            data,
        })
    }

    /// Total number of samples, counting every channel.
    pub fn sample_count(&self) -> usize {
        self.data.len()
    }

    /// Number of pixels, ignoring channels.
    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }

    /// Whether another image has the same width, height, and channel count.
    pub fn dimensions_match(&self, other: &Self) -> bool {
        self.width == other.width && self.height == other.height && self.channels == other.channels
    }

    /// Copy a region of this image into a new image of that size.
    ///
    /// The region is in this image's pixel coordinates and must lie inside it.
    pub fn crop(&self, region: ReferenceRegion) -> Result<Self> {
        let past_right = region.x.checked_add(region.width);
        let past_bottom = region.y.checked_add(region.height);
        if region.width == 0
            || region.height == 0
            || past_right.is_none_or(|edge| edge > self.width)
            || past_bottom.is_none_or(|edge| edge > self.height)
        {
            return Err(Error::InvalidImage(format!(
                "crop region {}x{} at ({}, {}) does not fit a {}x{} image",
                region.width, region.height, region.x, region.y, self.width, self.height
            )));
        }
        if region.x == 0
            && region.y == 0
            && region.width == self.width
            && region.height == self.height
        {
            return Ok(self.clone());
        }
        let mut data = Vec::with_capacity(region.width * region.height * self.channels);
        for row in region.y..region.y + region.height {
            let start = (row * self.width + region.x) * self.channels;
            data.extend_from_slice(&self.data[start..start + region.width * self.channels]);
        }
        Self::new(region.width, region.height, self.channels, data)
    }

    /// One luminance value per pixel: the sample itself for mono, Rec.709 luma
    /// for RGB.
    pub fn luminance(&self) -> Vec<f32> {
        if self.channels == 1 {
            return self.data.clone();
        }
        self.data
            .chunks_exact(3)
            .map(|pixel| rec709_luma(pixel[0], pixel[1], pixel[2]))
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn debayer(self, layout: BayerLayout) -> Result<Self> {
        self.debayer_with(layout, Demosaic::default())
    }

    /// Debayer a one-channel CFA frame with the given method.
    pub(crate) fn debayer_with(self, layout: BayerLayout, demosaic: Demosaic) -> Result<Self> {
        if self.channels != 1 {
            return Err(Error::InvalidImage(
                "only a one-channel CFA image can be debayered".into(),
            ));
        }
        // VNG estimates each missing colour from colour differences, which
        // assumes white-balanced channels; raw sensor channels are not (on an
        // ASI2600MC green ran about twice red). So the mosaic is balanced to
        // green by the channels' medians first and each channel scaled back
        // after. Every photosite keeps its own sample exactly.
        let balance = channel_balance(&self.data, self.width, self.height, layout);
        let balanced = self
            .data
            .par_chunks(self.width.max(1))
            .enumerate()
            .flat_map_iter(|(y, row)| {
                row.iter()
                    .enumerate()
                    .map(move |(x, &value)| value * balance[layout.channel_at(x, y)])
            })
            .collect::<Vec<_>>();
        // Rows depend only on the mosaic, so bands of them debayer in
        // parallel with the same samples a single pass produces.
        const BAND_ROWS: usize = 32;
        let mut rgb = vec![0.0_f32; self.data.len() * 3];
        let row_samples = self.width * 3;
        if row_samples > 0 {
            rgb.par_chunks_mut(row_samples * BAND_ROWS)
                .enumerate()
                .for_each(|(band, rows)| {
                    let first_row = band * BAND_ROWS;
                    debayer_rgb_f32_rows(
                        &balanced,
                        self.width,
                        self.height,
                        layout.pattern,
                        layout.x_offset,
                        layout.y_offset,
                        first_row,
                        rows,
                    );
                    match demosaic {
                        Demosaic::Vng => {
                            vng_rows(&balanced, self.width, self.height, layout, first_row, rows)
                        }
                        Demosaic::Mhc => malvar_he_cutler_rows(
                            &balanced,
                            self.width,
                            self.height,
                            layout,
                            first_row,
                            rows,
                        ),
                        Demosaic::Bilinear => {}
                    }
                    for (band_row, out_row) in rows.chunks_exact_mut(row_samples).enumerate() {
                        let y = first_row + band_row;
                        for (x, pixel) in out_row.chunks_exact_mut(3).enumerate() {
                            for (channel, value) in pixel.iter_mut().enumerate() {
                                *value /= balance[channel];
                            }
                            pixel[layout.channel_at(x, y)] = self.data[y * self.width + x];
                        }
                    }
                });
        }
        Self::new(self.width, self.height, 3, rgb)
    }
}

/// How a stack fills in the two colours each Bayer photosite does not record.
/// Every method balances the mosaic's channels to green first and keeps each
/// photosite's own sample exactly.
///
/// On a 98-frame M45 stack (FWHM, red-blue centroid offset, and red near a
/// star relative to its colour):
///
/// | method   | FWHM  | offset | red 2-3 px out |
/// |----------|-------|--------|----------------|
/// | VNG      | 2.7px | 0.10px | 0.75-1.16      |
/// | MHC      | 2.4px | 0.08px | 1.4-2.0        |
/// | bilinear | 2.9px | 0.43px | 1.0-1.3        |
#[derive(Clone, Copy, Debug, Default, serde::Deserialize, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Demosaic {
    /// Variable number of gradients (Chang, Cheung and Pang, 1999),
    /// PixInsight's default: interpolates only along the smoothest
    /// directions, so stars keep their colour across their profile.
    #[default]
    Vng,
    /// Malvar, He and Cutler's gradient-corrected linear interpolation:
    /// the sharpest, but it rings around small stars.
    Mhc,
    /// The mean of each colour's neighbours: fastest and softest, and it
    /// pulls each colour's centroid toward its own photosites.
    Bilinear,
}

impl Demosaic {
    /// Whether this is the default, which options leave unserialized.
    pub fn is_vng(&self) -> bool {
        *self == Self::Vng
    }
}

/// Overwrite the interpolated samples of pixels at least two from every edge
/// with Malvar, He and Cutler's gradient-corrected estimates ("High-quality
/// linear interpolation for demosaicing of Bayer-patterned color images",
/// ICASSP 2004), leaving the bilinear estimates on the two-pixel border.
///
/// The sharpest of the three demosaics (2.4px stars on M45, against 2.7 for
/// VNG), but its linear correction rings around stars a few pixels wide: on
/// the balanced mosaic it leaves red about 1.5 to 2 times a star's colour two
/// pixels out.
fn malvar_he_cutler_rows(
    mosaic: &[f32],
    width: usize,
    height: usize,
    layout: BayerLayout,
    first_row: usize,
    out: &mut [f32],
) {
    if width < 5 || height < 5 {
        return;
    }
    for (band_row, out_row) in out.chunks_exact_mut(width * 3).enumerate() {
        let y = first_row + band_row;
        if y < 2 || y + 2 >= height {
            continue;
        }
        for x in 2..width - 2 {
            let at = |dx: isize, dy: isize| {
                mosaic[(y as isize + dy) as usize * width + (x as isize + dx) as usize]
            };
            let center = at(0, 0);
            let pixel = &mut out_row[x * 3..x * 3 + 3];
            let own = layout.channel_at(x, y);
            let orth1 = at(-1, 0) + at(1, 0) + at(0, -1) + at(0, 1);
            let orth2 = at(-2, 0) + at(2, 0) + at(0, -2) + at(0, 2);
            let diag = at(-1, -1) + at(1, -1) + at(-1, 1) + at(1, 1);
            if own == 1 {
                let horizontal = layout.channel_at(x + 1, y);
                let vertical = layout.channel_at(x, y + 1);
                let row_estimate =
                    (5.0 * center + 4.0 * (at(-1, 0) + at(1, 0)) - (at(-2, 0) + at(2, 0)) - diag
                        + 0.5 * (at(0, -2) + at(0, 2)))
                        / 8.0;
                let column_estimate =
                    (5.0 * center + 4.0 * (at(0, -1) + at(0, 1)) - (at(0, -2) + at(0, 2)) - diag
                        + 0.5 * (at(-2, 0) + at(2, 0)))
                        / 8.0;
                pixel[horizontal] = row_estimate;
                pixel[vertical] = column_estimate;
            } else {
                pixel[1] = (4.0 * center + 2.0 * orth1 - orth2) / 8.0;
                pixel[2 - own] = (6.0 * center + 2.0 * diag - 1.5 * orth2) / 8.0;
            }
        }
    }
}

/// The eight directions VNG weighs, as (row, column) steps.
const VNG_DIRECTIONS: [(isize, isize); 8] = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
];

/// For each direction, the neighbours whose samples estimate the colours
/// there, as (row, column) offsets: every pixel of the 5x5 window ahead of
/// the centre and within one step of the ray through it. A narrower set,
/// as in Chang, Cheung and Pang's paper, gave stars about 3% sharper on M45
/// but swung a small star's red and blue by up to 1.9 times its colour two
/// pixels out; this one keeps them within 0.75 to 1.16.
fn vng_neighbourhoods() -> [Vec<(isize, isize)>; 8] {
    VNG_DIRECTIONS.map(|(dr, dc)| {
        let mut taps = Vec::new();
        for row in -2..=2_isize {
            for column in -2..=2_isize {
                let ahead = row * dr + column * dc;
                let across = (row * dc - column * dr).abs();
                if ahead > 0 && across <= 1 {
                    taps.push((row, column));
                }
            }
        }
        taps
    })
}

/// Index of a (row, column) offset in a pixel's 5x5 window.
fn vng_cell(row: isize, column: isize) -> u8 {
    ((row + 2) * 5 + column + 2) as u8
}

/// A direction's neighbours as window cells and colours.
#[derive(Clone, Copy, Default)]
struct VngTaps {
    cells: [(u8, u8); 6],
    len: usize,
}

/// Each direction's [`VngTaps`] per Bayer phase.
type VngTapTables = [[VngTaps; 8]; 4];

/// Each direction's gradient pairs as window cells.
type VngGradientPairs = [[(u8, u8); 4]; 8];

/// Each direction's neighbours per Bayer phase (column parity, row parity)
/// of the raw coordinates, and its gradient pairs as window cells, the last
/// two counting half.
fn vng_tables(layout: BayerLayout) -> (VngTapTables, VngGradientPairs) {
    let neighbourhoods = vng_neighbourhoods();
    // `channel_at` applies the pattern's origin offsets itself.
    let tables = std::array::from_fn(|phase| {
        let (px, py) = ((phase & 1) as isize, (phase >> 1) as isize);
        let channel = |row: isize, column: isize| {
            layout.channel_at(
                (px + column).rem_euclid(2) as usize,
                (py + row).rem_euclid(2) as usize,
            ) as u8
        };
        std::array::from_fn(|direction| {
            let mut taps = VngTaps::default();
            for (slot, &(row, column)) in taps.cells.iter_mut().zip(&neighbourhoods[direction]) {
                *slot = (vng_cell(row, column), channel(row, column));
            }
            taps.len = neighbourhoods[direction].len().min(6);
            taps
        })
    });
    let gradient_pairs = VNG_DIRECTIONS.map(|(dr, dc)| {
        let (pr, pc) = (dc, -dr);
        [
            (vng_cell(dr, dc), vng_cell(-dr, -dc)),
            (vng_cell(2 * dr, 2 * dc), vng_cell(0, 0)),
            (vng_cell(pr + dr, pc + dc), vng_cell(pr - dr, pc - dc)),
            (vng_cell(-pr + dr, -pc + dc), vng_cell(-pr - dr, -pc - dc)),
        ]
    });
    (tables, gradient_pairs)
}

/// Pixels [`vng_rows`] estimates at once: same-colour pixels two apart.
const VNG_LANES: usize = 8;

/// `sum / count` in every lane. A direction counts one to four samples of a
/// colour, and dividing by a power of two gives exactly what multiplying by
/// its reciprocal does, so only a count of three needs a division.
#[inline(always)]
fn vng_mean(sum: &[f32; VNG_LANES], count: u32) -> [f32; VNG_LANES] {
    let mut mean = [0.0_f32; VNG_LANES];
    if count.is_power_of_two() {
        let reciprocal = 1.0 / count as f32;
        for (mean, &sum) in mean.iter_mut().zip(sum) {
            *mean = sum * reciprocal;
        }
    } else {
        let count = count as f32;
        for (mean, &sum) in mean.iter_mut().zip(sum) {
            *mean = sum / count;
        }
    }
    mean
}

/// Overwrite the interpolated samples of pixels at least two from every edge
/// with variable-number-of-gradients estimates (Chang, Cheung and Pang,
/// 1999), the method PixInsight uses by default.
///
/// For each pixel, a gradient is measured in each of eight directions from
/// same-colour differences two pixels apart, on the line through the pixel
/// and, at half weight, the lines beside it; only directions whose gradient
/// is at most `1.5 min + 0.5 (max - min)` are used. In each, every channel is
/// averaged over the neighbours toward that direction, and each missing
/// channel is the pixel's own sample plus the mean, over those directions,
/// of that channel's difference from its own. Because it never interpolates
/// across a steep edge, a star a few pixels wide keeps its colour profile:
/// Malvar-He-Cutler's linear correction, tried first, rang around such
/// stars, leaving red 1.5 to 2 times the star's colour two pixels out.
///
/// The pixels of one row and column parity share a Bayer phase, so they
/// share every table, and [`VNG_LANES`] of them run side by side as vector
/// lanes: each row of the window is split into its even and odd columns, so
/// a window cell of eight such pixels is eight adjacent samples. Every lane
/// works out every direction and keeps or drops its result by selection
/// rather than a branch, adding the kept ones in direction order, so each
/// pixel goes through the same operations in the same order as it would
/// alone and the output is bit-identical. The row loops live in this
/// multiversioned function itself, with an AVX2 clone that runs all eight
/// lanes in one vector; a closure handed to rayon would not inherit the
/// clone's target features.
#[multiversion::multiversion(targets("x86_64+avx2"))]
fn vng_rows(
    mosaic: &[f32],
    width: usize,
    height: usize,
    layout: BayerLayout,
    first_row: usize,
    out: &mut [f32],
) {
    const LANES: usize = VNG_LANES;
    if width < 5 || height < 5 {
        return;
    }
    let (tables, gradient_pairs) = vng_tables(layout);
    // The window's five rows, each as its even and then its odd columns,
    // padded so a full vector can be read past the last pixel.
    let stride = width / 2 + 1 + LANES;
    let mut planes = vec![0.0_f32; 10 * stride];
    for (band_row, out_row) in out.chunks_exact_mut(width * 3).enumerate() {
        let y = first_row + band_row;
        if y < 2 || y + 2 >= height {
            continue;
        }
        for row in 0..5 {
            let start = (y + row - 2) * width;
            let source = &mosaic[start..start + width];
            let (even, odd) = planes[2 * row * stride..(2 * row + 2) * stride].split_at_mut(stride);
            for (index, pair) in source.chunks_exact(2).enumerate() {
                even[index] = pair[0];
                odd[index] = pair[1];
            }
            if width % 2 == 1 {
                even[width / 2] = source[width - 1];
            }
        }
        for parity in 0..2 {
            let phase = parity | ((y & 1) << 1);
            let own = layout.channel_at(parity, y);
            let taps_by_direction = &tables[phase];
            // Each direction's neighbour count per colour; a direction
            // missing a colour never counts.
            let counts_by_direction = taps_by_direction.map(|taps| {
                let mut counts = [0_u32; 3];
                counts[own] += 1;
                for &(_, channel) in &taps.cells[..taps.len] {
                    counts[channel as usize] += 1;
                }
                counts
            });
            let mut x = 2 + parity;
            while x < width - 2 {
                let lanes = (width - 2 - x).div_ceil(2).min(LANES);
                let mut window = [[0.0_f32; LANES]; 25];
                for row in 0..5 {
                    for column in 0..5 {
                        let plane = 2 * row + ((parity + column) & 1);
                        let start = plane * stride + ((x + column - 2) >> 1);
                        window[row * 5 + column].copy_from_slice(&planes[start..start + LANES]);
                    }
                }
                let mut gradients = [[0.0_f32; LANES]; 8];
                for (gradient, pairs) in gradients.iter_mut().zip(&gradient_pairs) {
                    let (a0, b0) = (&window[pairs[0].0 as usize], &window[pairs[0].1 as usize]);
                    let (a1, b1) = (&window[pairs[1].0 as usize], &window[pairs[1].1 as usize]);
                    let (a2, b2) = (&window[pairs[2].0 as usize], &window[pairs[2].1 as usize]);
                    let (a3, b3) = (&window[pairs[3].0 as usize], &window[pairs[3].1 as usize]);
                    for lane in 0..LANES {
                        gradient[lane] = (a0[lane] - b0[lane]).abs()
                            + (a1[lane] - b1[lane]).abs()
                            + 0.5 * ((a2[lane] - b2[lane]).abs() + (a3[lane] - b3[lane]).abs());
                    }
                }
                let mut minimum = [f32::INFINITY; LANES];
                let mut maximum = [f32::NEG_INFINITY; LANES];
                for gradient in &gradients {
                    for lane in 0..LANES {
                        minimum[lane] = minimum[lane].min(gradient[lane]);
                        maximum[lane] = maximum[lane].max(gradient[lane]);
                    }
                }
                let mut threshold = [0.0_f32; LANES];
                for lane in 0..LANES {
                    threshold[lane] = 1.5 * minimum[lane] + 0.5 * (maximum[lane] - minimum[lane]);
                }
                let center = window[12];
                let mut differences = [[0.0_f32; LANES]; 3];
                let mut used = [0_u32; LANES];
                for ((gradient, taps), counts) in gradients
                    .iter()
                    .zip(taps_by_direction)
                    .zip(&counts_by_direction)
                {
                    if counts.contains(&0) {
                        continue;
                    }
                    // A comparison with NaN fails, so a NaN gradient or
                    // threshold keeps the direction, as in the per-pixel loop.
                    let mut dropped = [false; LANES];
                    for lane in 0..LANES {
                        dropped[lane] = gradient[lane] > threshold[lane];
                    }
                    let mut sums = [[0.0_f32; LANES]; 3];
                    for lane in 0..LANES {
                        sums[own][lane] += center[lane];
                    }
                    for &(cell, channel) in &taps.cells[..taps.len] {
                        let values = &window[cell as usize];
                        let sum = &mut sums[channel as usize];
                        for lane in 0..LANES {
                            sum[lane] += values[lane];
                        }
                    }
                    let own_mean = vng_mean(&sums[own], counts[own]);
                    // The pixel's own channel is never estimated, so its
                    // difference is not formed.
                    for channel in 0..3 {
                        if channel == own {
                            continue;
                        }
                        let mean = vng_mean(&sums[channel], counts[channel]);
                        let difference = &mut differences[channel];
                        for lane in 0..LANES {
                            let next = difference[lane] + (mean[lane] - own_mean[lane]);
                            difference[lane] = if dropped[lane] {
                                difference[lane]
                            } else {
                                next
                            };
                        }
                    }
                    for lane in 0..LANES {
                        used[lane] += u32::from(!dropped[lane]);
                    }
                }
                for lane in 0..lanes {
                    if used[lane] == 0 {
                        continue;
                    }
                    let pixel_x = x + 2 * lane;
                    let pixel = &mut out_row[pixel_x * 3..pixel_x * 3 + 3];
                    for (channel, value) in pixel.iter_mut().enumerate() {
                        if channel != own {
                            *value = center[lane] + differences[channel][lane] / used[lane] as f32;
                        }
                    }
                }
                x += 2 * LANES;
            }
        }
    }
}

/// Per-channel factors that bring a mosaic's red and blue photosites to its
/// green ones' level, from every 7th photosite of each.
fn channel_balance(mosaic: &[f32], width: usize, height: usize, layout: BayerLayout) -> [f32; 3] {
    let mut sites: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for y in (0..height).step_by(7) {
        for x in (0..width).step_by(7) {
            let value = mosaic[y * width + x];
            if value.is_finite() {
                sites[layout.channel_at(x, y)].push(value);
            }
        }
    }
    let level = |values: &mut Vec<f32>| -> Option<f32> {
        seiza_stats::median_in_place(values).filter(|median| *median > 0.0)
    };
    let levels = sites.each_mut().map(level);
    match levels {
        [Some(red), Some(green), Some(blue)] => [green / red, 1.0, green / blue],
        _ => [1.0; 3],
    }
}

/// Raw color-filter-array sampling of a one-channel frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BayerLayout {
    /// The CFA color order at the first stored pixel, with ROWORDER resolved.
    pub pattern: BayerPattern,
    /// Horizontal offset of the pattern origin, in pixels.
    pub x_offset: usize,
    /// Vertical offset of the pattern origin, in pixels.
    pub y_offset: usize,
}

impl BayerLayout {
    /// The channel (0 red, 1 green, 2 blue) a photosite at `(x, y)` records.
    pub(crate) fn channel_at(self, x: usize, y: usize) -> usize {
        let (column, row) = ((x + self.x_offset) & 1, (y + self.y_offset) & 1);
        match self.pattern {
            BayerPattern::Rggb => [[0, 1], [1, 2]][row][column],
            BayerPattern::Bggr => [[2, 1], [1, 0]][row][column],
            BayerPattern::Grbg => [[1, 0], [2, 1]][row][column],
            BayerPattern::Gbrg => [[1, 2], [0, 1]][row][column],
        }
    }
}

/// Rec.709 luma from linear RGB samples.
pub(crate) fn rec709_luma(red: f32, green: f32, blue: f32) -> f32 {
    0.2126_f32.mul_add(red, 0.7152_f32.mul_add(green, 0.0722 * blue))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `vng_rows` as it was before same-colour pixels ran as vector lanes:
    /// one pixel at a time, skipping the directions it drops.
    fn vng_rows_per_pixel(
        mosaic: &[f32],
        width: usize,
        height: usize,
        layout: BayerLayout,
        first_row: usize,
        out: &mut [f32],
    ) {
        if width < 5 || height < 5 {
            return;
        }
        /// Index of a (row, column) offset in a pixel's 5x5 window.
        fn cell(row: isize, column: isize) -> u8 {
            ((row + 2) * 5 + column + 2) as u8
        }
        /// A direction's neighbours as window cells and colours.
        #[derive(Clone, Copy, Default)]
        struct Taps {
            cells: [(u8, u8); 6],
            len: usize,
        }
        let neighbourhoods = vng_neighbourhoods();
        // Per Bayer phase (column parity, row parity) of the raw coordinates;
        // `channel_at` applies the pattern's origin offsets itself.
        let tables: [[Taps; 8]; 4] = std::array::from_fn(|phase| {
            let (px, py) = ((phase & 1) as isize, (phase >> 1) as isize);
            let channel = |row: isize, column: isize| {
                layout.channel_at(
                    (px + column).rem_euclid(2) as usize,
                    (py + row).rem_euclid(2) as usize,
                ) as u8
            };
            std::array::from_fn(|direction| {
                let mut taps = Taps::default();
                for (slot, &(row, column)) in taps.cells.iter_mut().zip(&neighbourhoods[direction])
                {
                    *slot = (cell(row, column), channel(row, column));
                }
                taps.len = neighbourhoods[direction].len().min(6);
                taps
            })
        });
        // Gradient pairs per direction as window cells; the last two count half.
        let gradient_pairs: [[(u8, u8); 4]; 8] = VNG_DIRECTIONS.map(|(dr, dc)| {
            let (pr, pc) = (dc, -dr);
            [
                (cell(dr, dc), cell(-dr, -dc)),
                (cell(2 * dr, 2 * dc), cell(0, 0)),
                (cell(pr + dr, pc + dc), cell(pr - dr, pc - dc)),
                (cell(-pr + dr, -pc + dc), cell(-pr - dr, -pc - dc)),
            ]
        });
        for (band_row, out_row) in out.chunks_exact_mut(width * 3).enumerate() {
            let y = first_row + band_row;
            if y < 2 || y + 2 >= height {
                continue;
            }
            let rows: [&[f32]; 5] = std::array::from_fn(|row| {
                let start = (y + row - 2) * width;
                &mosaic[start..start + width]
            });
            for x in 2..width - 2 {
                let mut window = [0.0_f32; 25];
                for (row, values) in rows.iter().enumerate() {
                    window[row * 5..row * 5 + 5].copy_from_slice(&values[x - 2..x + 3]);
                }
                let at = |cell: u8| window[cell as usize];
                let mut gradients = [0.0_f32; 8];
                for (gradient, pairs) in gradients.iter_mut().zip(&gradient_pairs) {
                    *gradient = (at(pairs[0].0) - at(pairs[0].1)).abs()
                        + (at(pairs[1].0) - at(pairs[1].1)).abs()
                        + 0.5
                            * ((at(pairs[2].0) - at(pairs[2].1)).abs()
                                + (at(pairs[3].0) - at(pairs[3].1)).abs());
                }
                let minimum = gradients.iter().copied().fold(f32::INFINITY, f32::min);
                let maximum = gradients.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let threshold = 1.5 * minimum + 0.5 * (maximum - minimum);
                let phase = (x & 1) | ((y & 1) << 1);
                let own = layout.channel_at(x, y);
                let center = window[12];
                let mut differences = [0.0_f32; 3];
                let mut used = 0_u32;
                for (&gradient, taps) in gradients.iter().zip(&tables[phase]) {
                    if gradient > threshold {
                        continue;
                    }
                    let mut sums = [0.0_f32; 3];
                    let mut counts = [0_u32; 3];
                    sums[own] += center;
                    counts[own] += 1;
                    for &(cell, channel) in &taps.cells[..taps.len] {
                        sums[channel as usize] += at(cell);
                        counts[channel as usize] += 1;
                    }
                    if counts.iter().all(|&count| count > 0) {
                        let own_mean = sums[own] / counts[own] as f32;
                        for channel in 0..3 {
                            differences[channel] +=
                                sums[channel] / counts[channel] as f32 - own_mean;
                        }
                        used += 1;
                    }
                }
                if used == 0 {
                    continue;
                }
                let pixel = &mut out_row[x * 3..x * 3 + 3];
                for (channel, value) in pixel.iter_mut().enumerate() {
                    if channel != own {
                        *value = center + differences[channel] / used as f32;
                    }
                }
            }
        }
    }

    #[test]
    fn crop_copies_the_requested_region() {
        let image = LinearImage::new(4, 3, 1, (0..12).map(|v| v as f32).collect()).unwrap();
        let region = ReferenceRegion {
            x: 1,
            y: 1,
            width: 2,
            height: 2,
        };
        let cropped = image.crop(region).unwrap();
        assert_eq!(cropped.width, 2);
        assert_eq!(cropped.height, 2);
        assert_eq!(cropped.data, [5.0, 6.0, 9.0, 10.0]);
    }

    #[test]
    fn crop_keeps_every_channel_of_an_rgb_pixel() {
        let image = LinearImage::new(2, 1, 3, (0..6).map(|v| v as f32).collect()).unwrap();
        let cropped = image
            .crop(ReferenceRegion {
                x: 1,
                y: 0,
                width: 1,
                height: 1,
            })
            .unwrap();
        assert_eq!(cropped.data, [3.0, 4.0, 5.0]);
    }

    #[test]
    fn crop_rejects_a_region_outside_the_image() {
        let image = LinearImage::new(2, 2, 1, vec![0.0; 4]).unwrap();
        for region in [
            ReferenceRegion {
                x: 1,
                y: 0,
                width: 2,
                height: 1,
            },
            ReferenceRegion {
                x: 0,
                y: 0,
                width: 0,
                height: 1,
            },
            ReferenceRegion {
                x: usize::MAX,
                y: 0,
                width: 1,
                height: 1,
            },
        ] {
            let error = image.crop(region).unwrap_err();
            assert!(error.to_string().contains("does not fit"), "{region:?}");
        }
    }

    /// An RGGB mosaic of a scene with the given colour, seen through a sensor
    /// whose red and blue respond at `sensitivity` of green.
    fn mosaic(
        width: usize,
        height: usize,
        sensitivity: [f32; 3],
        scene: impl Fn(f32, f32) -> [f32; 3],
    ) -> LinearImage {
        let layout = BayerLayout {
            pattern: BayerPattern::Rggb,
            x_offset: 0,
            y_offset: 0,
        };
        let data = (0..width * height)
            .map(|index| {
                let (x, y) = (index % width, index / width);
                let channel = layout.channel_at(x, y);
                scene(x as f32, y as f32)[channel] * sensitivity[channel]
            })
            .collect();
        LinearImage::new(width, height, 1, data).unwrap()
    }

    fn rggb() -> BayerLayout {
        BayerLayout {
            pattern: BayerPattern::Rggb,
            x_offset: 0,
            y_offset: 0,
        }
    }

    #[test]
    fn vng_lanes_match_the_per_pixel_vng_bit_for_bit() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let patterns = [
            BayerPattern::Rggb,
            BayerPattern::Bggr,
            BayerPattern::Grbg,
            BayerPattern::Gbrg,
        ];
        // Odd and even widths, rows shorter than one vector of lanes and
        // ones ending in a partial vector; smooth sky with bright spikes,
        // steps, and non-finite and signed-zero samples.
        for (width, height) in [(5, 5), (6, 5), (7, 9), (16, 7), (17, 12), (37, 11), (40, 9)] {
            for (index, pattern) in patterns.into_iter().enumerate() {
                let layout = BayerLayout {
                    pattern,
                    x_offset: index & 1,
                    y_offset: index >> 1,
                };
                let mosaic = (0..width * height)
                    .map(|pixel| match next() % 53 {
                        0 => f32::NAN,
                        1 => f32::INFINITY,
                        2 => -0.0,
                        3..=8 => 3000.0 + (next() % 9000) as f32,
                        _ => {
                            let step = if pixel % width > width / 2 {
                                400.0
                            } else {
                                0.0
                            };
                            100.0 + step + (next() >> 40) as f32 / (1 << 20) as f32
                        }
                    })
                    .collect::<Vec<_>>();
                let start = (0..width * height * 3)
                    .map(|_| (next() >> 40) as f32)
                    .collect::<Vec<_>>();
                // The whole frame as one band, and bands of three rows.
                for band_rows in [height, 3] {
                    let mut got = start.clone();
                    let mut expected = start.clone();
                    for (band, (got, expected)) in got
                        .chunks_mut(width * 3 * band_rows)
                        .zip(expected.chunks_mut(width * 3 * band_rows))
                        .enumerate()
                    {
                        let first_row = band * band_rows;
                        vng_rows(&mosaic, width, height, layout, first_row, got);
                        vng_rows_per_pixel(&mosaic, width, height, layout, first_row, expected);
                    }
                    // Bit for bit, except that a NaN may differ in sign
                    // where a window holds both an infinite and a NaN
                    // sample: inf - inf gives -NaN on x86, and where it
                    // meets the +NaN sample in an addition x86 passes on the
                    // first operand's, which the compiler may order either
                    // way (the unoptimized and AVX2 builds differ here).
                    for (sample, (a, b)) in got.iter().zip(&expected).enumerate() {
                        assert!(
                            a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                            "{pattern:?} {width}x{height}, {band_rows}-row bands, sample {sample}: \
                             {a} vs {b}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_flat_colour_field_demosaics_exactly() {
        let sensitivity = [0.5, 1.0, 0.7];
        for demosaic in [Demosaic::Vng, Demosaic::Mhc, Demosaic::Bilinear] {
            let rgb = mosaic(24, 20, sensitivity, |_, _| [800.0, 1000.0, 600.0])
                .debayer_with(rggb(), demosaic)
                .unwrap();
            for pixel in rgb.data.chunks_exact(3) {
                for (channel, expected) in [400.0_f32, 1000.0, 420.0].into_iter().enumerate() {
                    assert!(
                        (pixel[channel] - expected).abs() < 1.0e-2,
                        "{demosaic:?}: {pixel:?}"
                    );
                }
            }
        }
    }

    /// A star a few pixels wide must keep its colour across its profile.
    /// One frame's demosaic error at a pixel depends on where the star falls
    /// on the colour pattern; a stack of drifting frames averages that out,
    /// but not a bias that every placement shares. Malvar-He-Cutler, tried
    /// first, had such a bias: red 1.5 to 2 times the star's colour two
    /// pixels out on real stacks. So the colour of the ring 1.5 to 3 pixels
    /// from the centre is averaged over sixteen placements spanning the
    /// pattern's 2x2 period.
    #[test]
    fn a_small_coloured_star_keeps_its_colour_across_its_profile() {
        let (width, height) = (40, 40);
        let sensitivity = [0.5, 1.0, 0.7];
        let colour = [0.9_f32, 1.0, 0.6];
        let sky = 200.0;
        let mut sums = [0.0_f64; 3];
        for step in 0..16 {
            // Over a full 2x2 period of the colour pattern.
            let (cx, cy) = (
                19.0 + (step % 4) as f32 * 0.5,
                20.0 + (step / 4) as f32 * 0.5,
            );
            let rgb = mosaic(width, height, sensitivity, |x, y| {
                let star =
                    20_000.0 * (-((x - cx).powi(2) + (y - cy).powi(2)) / (2.0 * 1.1 * 1.1)).exp();
                colour.map(|c| sky + c * star)
            })
            .debayer(rggb())
            .unwrap();
            for y in 15..=25 {
                for x in 15..=25 {
                    let distance = (x as f32 - cx).hypot(y as f32 - cy);
                    if !(1.5..=3.0).contains(&distance) {
                        continue;
                    }
                    let pixel = &rgb.data[(y * width + x) * 3..(y * width + x) * 3 + 3];
                    for channel in 0..3 {
                        sums[channel] += f64::from(pixel[channel] / sensitivity[channel] - sky);
                    }
                }
            }
        }
        for channel in [0, 2] {
            let relative = sums[channel] / sums[1] / f64::from(colour[channel]);
            assert!(
                (0.9..=1.1).contains(&relative),
                "channel {channel}: {relative}"
            );
        }
    }

    #[test]
    fn debayer_preserves_samples_at_native_color_sites() {
        // Large enough that MHC and VNG reach the interior.
        let raw = LinearImage::new(8, 8, 1, (0..64).map(|v| v as f32 + 1.0).collect()).unwrap();
        for demosaic in [Demosaic::Vng, Demosaic::Mhc, Demosaic::Bilinear] {
            let rgb = raw.clone().debayer_with(rggb(), demosaic).unwrap();
            assert_eq!(rgb.channels, 3);
            for y in 0..8 {
                for x in 0..8 {
                    let own = rggb().channel_at(x, y);
                    assert_eq!(
                        rgb.data[(y * 8 + x) * 3 + own],
                        raw.data[y * 8 + x],
                        "{demosaic:?} ({x}, {y})"
                    );
                }
            }
        }
    }
}
