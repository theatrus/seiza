use crate::{Error, LinearImage, Result};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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
                let mut map = Self {
                    width: source.width,
                    height: source.height,
                    channels: source.channels,
                    tile_size,
                    columns,
                    rows,
                    gains: vec![1.0; cell_count],
                    offsets: vec![0.0; cell_count],
                };
                for row in 0..rows {
                    for column in 0..columns {
                        let x = column * tile_size;
                        let y = row * tile_size;
                        let width = tile_size.min(source.width - x);
                        let height = tile_size.min(source.height - y);
                        for channel in 0..source.channels {
                            if let Ok((gain, offset)) =
                                affine_for_region(reference, source, channel, x, y, width, height)
                            {
                                let index = map.index(column, row, channel);
                                map.gains[index] = gain;
                                map.offsets[index] = offset;
                            }
                        }
                    }
                }
                Ok(map)
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
        for y in 0..image.height {
            for x in 0..image.width {
                for channel in 0..image.channels {
                    let (gain, offset) = self.sample(x, y, channel);
                    let index = (y * image.width + x) * image.channels + channel;
                    let value = image.data[index];
                    if value.is_finite() {
                        image.data[index] = value.mul_add(gain, offset);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn mean_gain(&self) -> f32 {
        self.gains.iter().sum::<f32>() / self.gains.len() as f32
    }

    pub fn mean_offset(&self) -> f32 {
        self.offsets.iter().sum::<f32>() / self.offsets.len() as f32
    }

    fn index(&self, column: usize, row: usize, channel: usize) -> usize {
        (row * self.columns + column) * self.channels + channel
    }

    fn sample(&self, x: usize, y: usize, channel: usize) -> (f32, f32) {
        if self.columns == 1 && self.rows == 1 {
            return (self.gains[channel], self.offsets[channel]);
        }
        let grid_x =
            ((x as f32 + 0.5) / self.tile_size as f32 - 0.5).clamp(0.0, (self.columns - 1) as f32);
        let grid_y =
            ((y as f32 + 0.5) / self.tile_size as f32 - 0.5).clamp(0.0, (self.rows - 1) as f32);
        let x0 = grid_x.floor() as usize;
        let y0 = grid_y.floor() as usize;
        let x1 = (x0 + 1).min(self.columns - 1);
        let y1 = (y0 + 1).min(self.rows - 1);
        let tx = if x0 == x1 {
            0.0
        } else {
            grid_x - grid_x.floor()
        };
        let ty = if y0 == y1 {
            0.0
        } else {
            grid_y - grid_y.floor()
        };
        let interpolate = |values: &[f32]| {
            let top = values[self.index(x0, y0, channel)] * (1.0 - tx)
                + values[self.index(x1, y0, channel)] * tx;
            let bottom = values[self.index(x0, y1, channel)] * (1.0 - tx)
                + values[self.index(x1, y1, channel)] * tx;
            top * (1.0 - ty) + bottom * ty
        };
        (interpolate(&self.gains), interpolate(&self.offsets))
    }
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
    let gain = if source_sigma > 1.0e-8 && reference_sigma.is_finite() {
        (reference_sigma / source_sigma).clamp(0.25, 4.0)
    } else {
        1.0
    };
    Ok((gain, reference_median - gain * source_median))
}

fn median(values: &mut [f32]) -> f32 {
    values.sort_unstable_by(f32::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
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
}
