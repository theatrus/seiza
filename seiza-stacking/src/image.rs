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

    pub(crate) fn debayer(self, layout: BayerLayout) -> Result<Self> {
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
                    vng_rows(&balanced, self.width, self.height, layout, first_row, rows);
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
fn vng_rows(
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
            for (slot, &(row, column)) in taps.cells.iter_mut().zip(&neighbourhoods[direction]) {
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
                        differences[channel] += sums[channel] / counts[channel] as f32 - own_mean;
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
    fn a_flat_colour_field_demosaics_exactly() {
        let sensitivity = [0.5, 1.0, 0.7];
        let rgb = mosaic(24, 20, sensitivity, |_, _| [800.0, 1000.0, 600.0])
            .debayer(rggb())
            .unwrap();
        for pixel in rgb.data.chunks_exact(3) {
            for (channel, expected) in [400.0_f32, 1000.0, 420.0].into_iter().enumerate() {
                assert!((pixel[channel] - expected).abs() < 1.0e-2, "{pixel:?}");
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
        let raw = LinearImage::new(4, 4, 1, (0..16).map(|v| v as f32).collect()).unwrap();
        let rgb = raw
            .debayer(BayerLayout {
                pattern: BayerPattern::Rggb,
                x_offset: 0,
                y_offset: 0,
            })
            .unwrap();
        assert_eq!(rgb.channels, 3);
        assert_eq!(rgb.data[0], 0.0);
        assert_eq!(rgb.data[4], 1.0);
        assert_eq!(rgb.data[(3 * 4 + 3) * 3 + 2], 15.0);
    }
}
