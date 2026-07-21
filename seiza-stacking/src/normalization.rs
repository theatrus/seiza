use crate::{Error, LinearImage, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(tag = "mode", content = "options", rename_all = "kebab-case")]
pub enum NormalizationMode {
    None,
    #[default]
    Global,
    Local {
        tile_size: usize,
    },
}

#[derive(Clone, Debug)]
pub struct NormalizationMap {
    width: usize,
    height: usize,
    channels: usize,
    tile_size: usize,
    columns: usize,
    rows: usize,
    gains: Vec<f32>,
    offsets: Vec<f32>,
}

impl NormalizationMap {
    pub fn identity(image: &LinearImage) -> Self {
        Self {
            width: image.width,
            height: image.height,
            channels: image.channels,
            tile_size: image.width.max(image.height),
            columns: 1,
            rows: 1,
            gains: vec![1.0; image.channels],
            offsets: vec![0.0; image.channels],
        }
    }

    pub fn estimate(
        reference: &LinearImage,
        source: &LinearImage,
        mode: NormalizationMode,
    ) -> Result<Self> {
        if !reference.dimensions_match(source) {
            return Err(Error::Normalization(
                "reference and source dimensions must match".into(),
            ));
        }
        match mode {
            NormalizationMode::None => Ok(Self::identity(source)),
            NormalizationMode::Global => {
                let mut map = Self::identity(source);
                for channel in 0..source.channels {
                    let (gain, offset) = affine_for_region(
                        reference,
                        source,
                        channel,
                        0,
                        0,
                        source.width,
                        source.height,
                    )?;
                    map.gains[channel] = gain;
                    map.offsets[channel] = offset;
                }
                Ok(map)
            }
            NormalizationMode::Local { tile_size } => {
                if tile_size < 16 {
                    return Err(Error::Normalization(
                        "local normalization tile size must be at least 16 pixels".into(),
                    ));
                }
                let columns = source.width.div_ceil(tile_size);
                let rows = source.height.div_ceil(tile_size);
                let cell_count = columns * rows * source.channels;
                let coefficients = (0..cell_count)
                    .into_par_iter()
                    .map(|index| {
                        let channel = index % source.channels;
                        let cell = index / source.channels;
                        let column = cell % columns;
                        let row = cell / columns;
                        let x = column * tile_size;
                        let y = row * tile_size;
                        let width = tile_size.min(source.width - x);
                        let height = tile_size.min(source.height - y);
                        affine_for_region(reference, source, channel, x, y, width, height)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let (gains, offsets) = coefficients.into_iter().unzip();
                Ok(Self {
                    width: source.width,
                    height: source.height,
                    channels: source.channels,
                    tile_size,
                    columns,
                    rows,
                    gains,
                    offsets,
                })
            }
        }
    }

    pub fn apply(&self, image: &mut LinearImage) -> Result<()> {
        if image.width != self.width
            || image.height != self.height
            || image.channels != self.channels
        {
            return Err(Error::Normalization(
                "normalization map and image dimensions do not match".into(),
            ));
        }
        if self.columns == 1 && self.rows == 1 {
            image.data.par_chunks_mut(image.channels).for_each(|pixel| {
                for (channel, value) in pixel.iter_mut().enumerate() {
                    if value.is_finite() {
                        *value = value.mul_add(self.gains[channel], self.offsets[channel]);
                    }
                }
            });
            return Ok(());
        }

        let x_weights = (0..image.width)
            .map(|x| axis_weights(x, self.columns, self.tile_size))
            .collect::<Vec<_>>();
        let row_samples = image.width * image.channels;
        image
            .data
            .par_chunks_mut(row_samples)
            .enumerate()
            .for_each(|(y, row)| {
                let y_weights = axis_weights(y, self.rows, self.tile_size);
                for (x, pixel) in row.chunks_exact_mut(self.channels).enumerate() {
                    let x_weights = x_weights[x];
                    let top_left = (y_weights.low * self.columns + x_weights.low) * self.channels;
                    let top_right = (y_weights.low * self.columns + x_weights.high) * self.channels;
                    let bottom_left =
                        (y_weights.high * self.columns + x_weights.low) * self.channels;
                    let bottom_right =
                        (y_weights.high * self.columns + x_weights.high) * self.channels;
                    for (channel, value) in pixel.iter_mut().enumerate() {
                        if !value.is_finite() {
                            continue;
                        }
                        let gain = bilinear(
                            self.gains[top_left + channel],
                            self.gains[top_right + channel],
                            self.gains[bottom_left + channel],
                            self.gains[bottom_right + channel],
                            x_weights.fraction,
                            y_weights.fraction,
                        );
                        let offset = bilinear(
                            self.offsets[top_left + channel],
                            self.offsets[top_right + channel],
                            self.offsets[bottom_left + channel],
                            self.offsets[bottom_right + channel],
                            x_weights.fraction,
                            y_weights.fraction,
                        );
                        *value = value.mul_add(gain, offset);
                    }
                }
            });
        Ok(())
    }

    pub fn mean_gain(&self) -> f32 {
        self.gains.iter().sum::<f32>() / self.gains.len() as f32
    }

    pub fn mean_offset(&self) -> f32 {
        self.offsets.iter().sum::<f32>() / self.offsets.len() as f32
    }

    /// Smallest and largest gain in the map. Live admission checks the full
    /// range so a pathological local tile cannot hide behind a reasonable
    /// mean gain.
    pub fn gain_range(&self) -> (f32, f32) {
        self.gains.iter().copied().fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(minimum, maximum), gain| (minimum.min(gain), maximum.max(gain)),
        )
    }
}

#[derive(Clone, Copy)]
struct AxisWeights {
    low: usize,
    high: usize,
    fraction: f32,
}

fn axis_weights(coordinate: usize, cells: usize, tile_size: usize) -> AxisWeights {
    let grid = ((coordinate as f32 + 0.5) / tile_size as f32 - 0.5).clamp(0.0, (cells - 1) as f32);
    let low = grid.floor() as usize;
    let high = (low + 1).min(cells - 1);
    AxisWeights {
        low,
        high,
        fraction: if low == high { 0.0 } else { grid - low as f32 },
    }
}

fn bilinear(
    top_left: f32,
    top_right: f32,
    bottom_left: f32,
    bottom_right: f32,
    x: f32,
    y: f32,
) -> f32 {
    let top = top_left * (1.0 - x) + top_right * x;
    let bottom = bottom_left * (1.0 - x) + bottom_right * x;
    top * (1.0 - y) + bottom * y
}

fn affine_for_region(
    reference: &LinearImage,
    source: &LinearImage,
    channel: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> Result<(f32, f32)> {
    let stride = (width * height / 20_000).max(1);
    let mut reference_values = Vec::new();
    let mut source_values = Vec::new();
    let mut sample_index = 0;
    for row in y..y + height {
        for column in x..x + width {
            if sample_index % stride == 0 {
                let index = (row * source.width + column) * source.channels + channel;
                let reference_value = reference.data[index];
                let source_value = source.data[index];
                if reference_value.is_finite() && source_value.is_finite() {
                    reference_values.push(reference_value);
                    source_values.push(source_value);
                }
            }
            sample_index += 1;
        }
    }
    if reference_values.len() < 32 {
        return Err(Error::Normalization(
            "too few overlapping finite pixels for normalization".into(),
        ));
    }
    let reference_median = median(&mut reference_values);
    let source_median = median(&mut source_values);
    let reference_sigma = robust_sigma(&mut reference_values, reference_median);
    let source_sigma = robust_sigma(&mut source_values, source_median);
    if !reference_sigma.is_finite() || !source_sigma.is_finite() || source_sigma <= 1.0e-8 {
        return Err(Error::Normalization(
            "normalization region has no usable dispersion".into(),
        ));
    }
    let gain = reference_sigma / source_sigma;
    if !gain.is_finite() {
        return Err(Error::Normalization(
            "normalization produced a non-finite gain".into(),
        ));
    }
    Ok((gain, reference_median - gain * source_median))
}

fn median(values: &mut [f32]) -> f32 {
    let middle = values.len() / 2;
    let even = values.len().is_multiple_of(2);
    let (lower, median, _) =
        values.select_nth_unstable_by(middle, |left, right| left.total_cmp(right));
    if even {
        let lower_median = lower
            .iter()
            .max_by(|left, right| left.total_cmp(right))
            .expect("an even non-empty sample has a lower partition");
        (*lower_median + *median) * 0.5
    } else {
        *median
    }
}

fn robust_sigma(values: &mut [f32], center: f32) -> f32 {
    for value in values.iter_mut() {
        *value = (*value - center).abs();
    }
    median(values) * 1.4826
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_selection_matches_sorted_even_and_odd_samples() {
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0, 5.0]), 3.0);
    }

    #[test]
    fn global_normalization_recovers_affine_background() {
        let reference = LinearImage::new(
            16,
            16,
            1,
            (0..256).map(|value| value as f32 + 20.0).collect(),
        )
        .unwrap();
        let source = LinearImage::new(
            16,
            16,
            1,
            reference
                .data
                .iter()
                .map(|value| value * 2.0 + 8.0)
                .collect(),
        )
        .unwrap();
        let map =
            NormalizationMap::estimate(&reference, &source, NormalizationMode::Global).unwrap();
        let mut normalized = source;
        map.apply(&mut normalized).unwrap();
        assert!((map.mean_gain() - 0.5).abs() < 1.0e-5);
        assert!((normalized.data[100] - reference.data[100]).abs() < 1.0e-3);
    }

    #[test]
    fn preserves_extreme_gain_for_admission_instead_of_clamping() {
        let reference = LinearImage::new(
            16,
            16,
            1,
            (0..256).map(|value| value as f32 * 10.0).collect(),
        )
        .unwrap();
        let source =
            LinearImage::new(16, 16, 1, (0..256).map(|value| value as f32).collect()).unwrap();
        let map =
            NormalizationMap::estimate(&reference, &source, NormalizationMode::Global).unwrap();
        let (minimum, maximum) = map.gain_range();
        assert!((minimum - 10.0).abs() < 1.0e-5);
        assert!((maximum - 10.0).abs() < 1.0e-5);
    }

    #[test]
    fn local_normalization_rejects_an_unusable_tile() {
        let reference =
            LinearImage::new(32, 32, 1, (0..1024).map(|value| value as f32).collect()).unwrap();
        let mut source = reference.clone();
        for y in 16..32 {
            for x in 16..32 {
                source.data[y * 32 + x] = f32::NAN;
            }
        }
        assert!(
            NormalizationMap::estimate(
                &reference,
                &source,
                NormalizationMode::Local { tile_size: 16 },
            )
            .is_err()
        );
    }
}
