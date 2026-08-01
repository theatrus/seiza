use crate::{Error, LinearImage, Result};
use rayon::prelude::*;
use seiza_stats::{median_in_place, robust_sigma_in_place};
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};

const NORMALIZATION_MAP_SCHEMA_VERSION: u32 = 1;

/// How a frame's background is matched to the reference before stacking.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(tag = "mode", content = "options", rename_all = "kebab-case")]
pub enum NormalizationMode {
    /// Leave samples untouched.
    None,
    /// One gain and offset for the whole frame per channel.
    #[default]
    Global,
    /// A grid of per-tile gains and offsets, interpolated across the frame.
    Local {
        /// Tile edge length in pixels; must be at least 16.
        tile_size: usize,
    },
}

/// Per-channel gain and offset that map a source frame onto the reference
/// background, either globally or over a tile grid.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NormalizationMap {
    schema_version: u32,
    width: usize,
    height: usize,
    channels: usize,
    tile_size: usize,
    columns: usize,
    rows: usize,
    gains: Vec<f32>,
    offsets: Vec<f32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NormalizationMapWire {
    schema_version: u32,
    width: usize,
    height: usize,
    channels: usize,
    tile_size: usize,
    columns: usize,
    rows: usize,
    gains: Vec<f32>,
    offsets: Vec<f32>,
}

impl<'de> Deserialize<'de> for NormalizationMap {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = NormalizationMapWire::deserialize(deserializer)?;
        let map = Self {
            schema_version: wire.schema_version,
            width: wire.width,
            height: wire.height,
            channels: wire.channels,
            tile_size: wire.tile_size,
            columns: wire.columns,
            rows: wire.rows,
            gains: wire.gains,
            offsets: wire.offsets,
        };
        map.validate().map_err(D::Error::custom)?;
        Ok(map)
    }
}

impl NormalizationMap {
    /// A map that leaves an image of the given shape unchanged.
    pub fn identity(image: &LinearImage) -> Self {
        Self {
            schema_version: NORMALIZATION_MAP_SCHEMA_VERSION,
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

    /// Fit gains and offsets that match `source`'s background to `reference`
    /// using robust median and dispersion statistics.
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
                    schema_version: NORMALIZATION_MAP_SCHEMA_VERSION,
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

    /// Rescale an image in place with the fitted map. Non-finite samples are
    /// left untouched; local maps interpolate gains and offsets between tiles.
    pub fn apply(&self, image: &mut LinearImage) -> Result<()> {
        self.validate()?;
        if image.width != self.width
            || image.height != self.height
            || image.channels != self.channels
        {
            return Err(Error::Normalization(
                "normalization map and image dimensions do not match".into(),
            ));
        }
        self.apply_region(image, 0, 0)
    }

    /// Apply this map to a crop whose origin is expressed in the full
    /// registered image grid used to estimate the map.
    pub fn apply_region(
        &self,
        image: &mut LinearImage,
        origin_x: usize,
        origin_y: usize,
    ) -> Result<()> {
        self.validate()?;
        let right = origin_x
            .checked_add(image.width)
            .ok_or_else(|| Error::Normalization("normalization region overflows".into()))?;
        let bottom = origin_y
            .checked_add(image.height)
            .ok_or_else(|| Error::Normalization("normalization region overflows".into()))?;
        if image.channels != self.channels || right > self.width || bottom > self.height {
            return Err(Error::Normalization(
                "normalization region exceeds the fitted image grid".into(),
            ));
        }
        if self.columns == 1 && self.rows == 1 {
            return self.apply_global(image);
        }

        let x_weights = (0..image.width)
            .map(|x| axis_weights(origin_x + x, self.columns, self.tile_size))
            .collect::<Vec<_>>();
        let row_samples = image.width * image.channels;
        image
            .data
            .par_chunks_mut(row_samples)
            .enumerate()
            .for_each(|(y, row)| {
                let y_weights = axis_weights(origin_y + y, self.rows, self.tile_size);
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

    /// Apply a one-tile global map to any image with the same channel count.
    /// This is useful after another geometric resampling because a constant
    /// per-channel affine transform does not depend on pixel coordinates.
    pub fn apply_global(&self, image: &mut LinearImage) -> Result<()> {
        self.validate()?;
        if self.columns != 1 || self.rows != 1 {
            return Err(Error::Normalization(
                "normalization map is not global".into(),
            ));
        }
        if image.channels != self.channels {
            return Err(Error::Normalization(
                "normalization channel count does not match".into(),
            ));
        }
        image.data.par_chunks_mut(image.channels).for_each(|pixel| {
            for (channel, value) in pixel.iter_mut().enumerate() {
                if value.is_finite() {
                    *value = value.mul_add(self.gains[channel], self.offsets[channel]);
                }
            }
        });
        Ok(())
    }

    /// Check the serialized map shape and every coefficient before use.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != NORMALIZATION_MAP_SCHEMA_VERSION {
            return Err(Error::Normalization(format!(
                "unsupported normalization map schema version {}",
                self.schema_version
            )));
        }
        if self.width == 0 || self.height == 0 || self.channels == 0 || self.tile_size == 0 {
            return Err(Error::Normalization(
                "normalization map dimensions must be non-zero".into(),
            ));
        }
        let expected_columns = self.width.div_ceil(self.tile_size);
        let expected_rows = self.height.div_ceil(self.tile_size);
        if self.columns != expected_columns || self.rows != expected_rows {
            return Err(Error::Normalization(
                "normalization tile grid does not match image dimensions".into(),
            ));
        }
        if (self.columns > 1 || self.rows > 1) && self.tile_size < 16 {
            return Err(Error::Normalization(
                "local normalization tile size must be at least 16 pixels".into(),
            ));
        }
        let coefficient_count = self
            .columns
            .checked_mul(self.rows)
            .and_then(|cells| cells.checked_mul(self.channels))
            .ok_or_else(|| Error::Normalization("normalization map dimensions overflow".into()))?;
        if self.gains.len() != coefficient_count || self.offsets.len() != coefficient_count {
            return Err(Error::Normalization(
                "normalization coefficient count does not match the tile grid".into(),
            ));
        }
        if !self
            .gains
            .iter()
            .chain(&self.offsets)
            .all(|coefficient| coefficient.is_finite())
        {
            return Err(Error::Normalization(
                "normalization coefficients must be finite".into(),
            ));
        }
        Ok(())
    }

    /// Width of the reference grid used to estimate this map.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Height of the reference grid used to estimate this map.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Channel count expected by this map.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Whether this map uses one constant affine transform per channel.
    pub fn is_global(&self) -> bool {
        self.columns == 1 && self.rows == 1
    }

    /// Mean gain across all channels and tiles.
    pub fn mean_gain(&self) -> f32 {
        self.gains.iter().sum::<f32>() / self.gains.len() as f32
    }

    /// Mean offset across all channels and tiles.
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
    let (Some(reference_median), Some(source_median)) = (
        median_in_place(&mut reference_values),
        median_in_place(&mut source_values),
    ) else {
        return Err(Error::Normalization(
            "too few overlapping finite pixels for normalization".into(),
        ));
    };
    let reference_sigma =
        robust_sigma_in_place(&mut reference_values, reference_median).unwrap_or(f32::NAN);
    let source_sigma = robust_sigma_in_place(&mut source_values, source_median).unwrap_or(f32::NAN);
    if !reference_sigma.is_finite() || !source_sigma.is_finite() || source_sigma <= 1.0e-8 {
        return Err(Error::Normalization(
            "normalization region has no usable dispersion".into(),
        ));
    }
    // Dispersion matching: the gain equalizes robust contrast against the
    // reference, which corrects transparency and exposure differences. It
    // assumes dispersion changes come from those global factors; a frame
    // whose extra dispersion has another cause (e.g. seeing) is still scaled
    // toward the reference.
    let gain = reference_sigma / source_sigma;
    if !gain.is_finite() {
        return Err(Error::Normalization(
            "normalization produced a non-finite gain".into(),
        ));
    }
    Ok((gain, reference_median - gain * source_median))
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

    #[test]
    fn region_application_matches_the_same_part_of_the_full_image() {
        let map = NormalizationMap {
            schema_version: NORMALIZATION_MAP_SCHEMA_VERSION,
            width: 32,
            height: 32,
            channels: 1,
            tile_size: 16,
            columns: 2,
            rows: 2,
            gains: vec![1.0, 2.0, 3.0, 4.0],
            offsets: vec![0.0; 4],
        };
        let mut full = LinearImage::new(32, 32, 1, vec![1.0; 32 * 32]).unwrap();
        map.apply(&mut full).unwrap();
        let mut crop = LinearImage::new(2, 2, 1, vec![1.0; 4]).unwrap();
        map.apply_region(&mut crop, 15, 15).unwrap();

        assert_eq!(
            crop.data,
            vec![
                full.data[15 * 32 + 15],
                full.data[15 * 32 + 16],
                full.data[16 * 32 + 15],
                full.data[16 * 32 + 16],
            ]
        );
        assert!(map.apply_global(&mut crop).is_err());
    }

    #[test]
    fn normalization_maps_round_trip_for_cached_provenance() {
        let map =
            NormalizationMap::identity(&LinearImage::new(4, 3, 3, vec![0.0; 4 * 3 * 3]).unwrap());
        let encoded = serde_json::to_vec(&map).unwrap();
        let decoded = serde_json::from_slice::<NormalizationMap>(&encoded).unwrap();

        assert_eq!(decoded, map);
    }

    #[test]
    fn malformed_serialized_maps_are_rejected_before_pixel_work() {
        let map = NormalizationMap::identity(&LinearImage::new(4, 3, 1, vec![0.0; 12]).unwrap());
        let mut value = serde_json::to_value(map).unwrap();
        value["gains"] = serde_json::json!([]);

        assert!(serde_json::from_value::<NormalizationMap>(value).is_err());
    }

    #[test]
    fn global_region_application_keeps_per_channel_coefficients() {
        let map = NormalizationMap {
            schema_version: NORMALIZATION_MAP_SCHEMA_VERSION,
            width: 8,
            height: 6,
            channels: 3,
            tile_size: 8,
            columns: 1,
            rows: 1,
            gains: vec![1.0, 2.0, 3.0],
            offsets: vec![10.0, 20.0, 30.0],
        };
        let mut crop = LinearImage::new(1, 1, 3, vec![2.0; 3]).unwrap();

        map.apply_global(&mut crop).unwrap();

        assert_eq!(crop.data, vec![12.0, 24.0, 36.0]);
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
