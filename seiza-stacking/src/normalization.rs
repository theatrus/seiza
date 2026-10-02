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
    /// One gain per channel for the whole frame, as [`Self::Global`] fits,
    /// and a grid of background offsets interpolated across it.
    ///
    /// Frames whose sky gradient differs from the reference's leave a step
    /// in the stack wherever the set of frames covering a pixel changes, so
    /// a drifting or flipped session draws its frames' edges as faint
    /// streaks. A per-tile offset matches each frame's background to the
    /// reference's and removes them, without [`Self::Local`]'s per-tile
    /// gains, which a tile of cloud, nebula or frame edge can drive far
    /// enough to reject the frame. The gain is the global one, so it follows
    /// transparency and frame weighting still sees each frame's noise. Each
    /// tile's offset compares medians of the samples both frames cover;
    /// tiles with too few are filled from their neighbours, and the grid is
    /// smoothed over three tiles.
    LocalBackground {
        /// Tile edge length in pixels; must be at least 16.
        tile_size: usize,
    },
}

impl NormalizationMap {
    /// A one-tile map with this map's mean gain and offset per channel.
    pub(crate) fn global_equivalent(&self) -> Self {
        let tiles = (self.columns * self.rows) as f32;
        let mean = |values: &[f32], channel: usize| {
            values
                .iter()
                .skip(channel)
                .step_by(self.channels)
                .sum::<f32>()
                / tiles
        };
        Self {
            schema_version: NORMALIZATION_MAP_SCHEMA_VERSION,
            width: self.width,
            height: self.height,
            channels: self.channels,
            tile_size: self.width.max(self.height),
            columns: 1,
            rows: 1,
            gains: (0..self.channels).map(|c| mean(&self.gains, c)).collect(),
            offsets: (0..self.channels).map(|c| mean(&self.offsets, c)).collect(),
        }
    }

    /// [`Self::estimate`], with per-channel gains measured elsewhere (from
    /// star photometry) for [`NormalizationMode::LocalBackground`]. Other
    /// modes, and a missing measurement, fit as [`Self::estimate`] does.
    pub(crate) fn estimate_with_gains(
        reference: &LinearImage,
        source: &LinearImage,
        mode: NormalizationMode,
        gains: Option<&[f32]>,
    ) -> Result<Self> {
        match (mode, gains) {
            (NormalizationMode::LocalBackground { tile_size }, Some(gains))
                if gains.len() == source.channels && reference.dimensions_match(source) =>
            {
                if tile_size < 16 {
                    return Err(Error::Normalization(
                        "local normalization tile size must be at least 16 pixels".into(),
                    ));
                }
                let globals = gains
                    .iter()
                    .enumerate()
                    .map(|(channel, &gain)| {
                        let (reference_median, source_median) = tile_medians(
                            reference,
                            source,
                            channel,
                            0,
                            0,
                            source.width,
                            source.height,
                        )
                        .ok_or_else(|| {
                            Error::Normalization(
                                "too few overlapping finite pixels for normalization".into(),
                            )
                        })?;
                        Ok((gain, reference_median - gain * source_median))
                    })
                    .collect::<Result<Vec<_>>>()?;
                local_background_with_gains(reference, source, tile_size, &globals)
            }
            _ => Self::estimate(reference, source, mode),
        }
    }

    /// Fit [`NormalizationMode::LocalBackground`] offsets against `reference`
    /// while keeping this map's per-channel gains, as fitted against the
    /// stack's reference frame. A gain fitted against a reference of
    /// another noise level, such as an integration of many frames, would
    /// follow the noise rather than the signal.
    pub(crate) fn refit_background(
        &self,
        reference: &LinearImage,
        source: &LinearImage,
        tile_size: usize,
    ) -> Result<Self> {
        if source.channels != self.channels {
            return Err(Error::Normalization(
                "normalization channel count does not match".into(),
            ));
        }
        let tiles = (self.columns * self.rows) as f32;
        let globals = (0..self.channels)
            .map(|channel| {
                let offset = self
                    .offsets
                    .iter()
                    .skip(channel)
                    .step_by(self.channels)
                    .sum::<f32>()
                    / tiles;
                (self.channel_mean_gain(channel), offset)
            })
            .collect::<Vec<_>>();
        local_background_with_gains(reference, source, tile_size, &globals)
    }
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
            NormalizationMode::LocalBackground { tile_size } => {
                if tile_size < 16 {
                    return Err(Error::Normalization(
                        "local normalization tile size must be at least 16 pixels".into(),
                    ));
                }
                local_background(reference, source, tile_size)
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

    /// Per-sample gains and offsets for drizzle, which reads them at
    /// scattered positions: each column's and row's tile weights are worked
    /// out once.
    pub(crate) fn sampler(&self) -> CoefficientSampler<'_> {
        let global = self.columns == 1 && self.rows == 1;
        let weights = |count: usize, cells: usize| {
            if global {
                Vec::new()
            } else {
                (0..count)
                    .map(|coordinate| axis_weights(coordinate, cells, self.tile_size))
                    .collect()
            }
        };
        CoefficientSampler {
            map: self,
            columns: weights(self.width, self.columns),
            rows: weights(self.height, self.rows),
        }
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
        let channels = image.channels;
        image
            .data
            .par_chunks_mut(channels * 4096)
            .for_each(|pixels| apply_affine(pixels, &self.gains, &self.offsets));
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

    /// Mean gain of one channel across all tiles.
    pub(crate) fn channel_mean_gain(&self, channel: usize) -> f32 {
        let gains = self.gains.iter().skip(channel).step_by(self.channels);
        let tiles = self.columns * self.rows;
        gains.sum::<f32>() / tiles as f32
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

/// See [`NormalizationMap::sampler`].
pub(crate) struct CoefficientSampler<'a> {
    map: &'a NormalizationMap,
    columns: Vec<AxisWeights>,
    rows: Vec<AxisWeights>,
}

impl CoefficientSampler<'_> {
    /// The gain and offset [`NormalizationMap::apply`] gives sample
    /// `(x, y, channel)`.
    pub(crate) fn at(&self, x: usize, y: usize, channel: usize) -> (f32, f32) {
        let map = self.map;
        if self.columns.is_empty() {
            return (map.gains[channel], map.offsets[channel]);
        }
        let (x_weights, y_weights) = (self.columns[x], self.rows[y]);
        let at = |row: usize, column: usize| (row * map.columns + column) * map.channels + channel;
        let corners = [
            at(y_weights.low, x_weights.low),
            at(y_weights.low, x_weights.high),
            at(y_weights.high, x_weights.low),
            at(y_weights.high, x_weights.high),
        ];
        let interpolate = |values: &[f32]| {
            bilinear(
                values[corners[0]],
                values[corners[1]],
                values[corners[2]],
                values[corners[3]],
                x_weights.fraction,
                y_weights.fraction,
            )
        };
        (interpolate(&map.gains), interpolate(&map.offsets))
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

/// [`NormalizationMode::LocalBackground`]: the global gain per channel and a
/// smoothed grid of background offsets.
///
/// The gain is [`NormalizationMode::Global`]'s, from the whole frame's
/// dispersion, which carries stars and nebulosity as well as noise and so
/// follows transparency. A gain from each tile's dispersion would match the
/// frames' sky noise instead, scaling a hazy frame to the reference's noise
/// rather than its signal, which also hides the noise frame weighting reads.
/// Each tile's offset then carries its gained source median onto the
/// reference median.
fn local_background(
    reference: &LinearImage,
    source: &LinearImage,
    tile_size: usize,
) -> Result<NormalizationMap> {
    let channels = source.channels;
    let globals = (0..channels)
        .map(|channel| {
            affine_for_region(
                reference,
                source,
                channel,
                0,
                0,
                source.width,
                source.height,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    local_background_with_gains(reference, source, tile_size, &globals)
}

/// Background offsets per tile for fixed per-channel gains, each paired with
/// the global offset used where no tile of that channel can be measured.
fn local_background_with_gains(
    reference: &LinearImage,
    source: &LinearImage,
    tile_size: usize,
    globals: &[(f32, f32)],
) -> Result<NormalizationMap> {
    let channels = source.channels;
    let columns = source.width.div_ceil(tile_size);
    let rows = source.height.div_ceil(tile_size);
    let mut offsets = (0..columns * rows * channels)
        .into_par_iter()
        .map(|index| {
            let channel = index % channels;
            let cell = index / channels;
            let (x, y) = ((cell % columns) * tile_size, (cell / columns) * tile_size);
            tile_medians(
                reference,
                source,
                channel,
                x,
                y,
                tile_size.min(source.width - x),
                tile_size.min(source.height - y),
            )
            .map(|(reference_median, source_median)| {
                reference_median - globals[channel].0 * source_median
            })
            .filter(|offset| offset.is_finite())
        })
        .collect::<Vec<_>>();
    for (channel, &(_, global_offset)) in globals.iter().enumerate() {
        fill_and_smooth_offsets(
            &mut offsets,
            columns,
            rows,
            channels,
            channel,
            global_offset,
        );
    }
    Ok(NormalizationMap {
        schema_version: NORMALIZATION_MAP_SCHEMA_VERSION,
        width: source.width,
        height: source.height,
        channels,
        tile_size,
        columns,
        rows,
        gains: (0..columns * rows)
            .flat_map(|_| globals.iter().map(|&(gain, _)| gain))
            .collect(),
        offsets: offsets
            .into_iter()
            .map(|offset| offset.unwrap_or(0.0))
            .collect(),
    })
}

/// The medians of one region and channel in both frames, over samples both
/// cover, or `None` when they share too few.
fn tile_medians(
    reference: &LinearImage,
    source: &LinearImage,
    channel: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> Option<(f32, f32)> {
    let stride = (width * height / 4_000).max(1);
    let mut reference_values = Vec::new();
    let mut source_values = Vec::new();
    for sample_index in (0..width * height).step_by(stride) {
        let index = ((y + sample_index / width) * source.width + x + sample_index % width)
            * source.channels
            + channel;
        let (reference_value, source_value) = (reference.data[index], source.data[index]);
        if reference_value.is_finite() && source_value.is_finite() {
            reference_values.push(reference_value);
            source_values.push(source_value);
        }
    }
    if reference_values.len() < 64 {
        return None;
    }
    Some((
        median_in_place(&mut reference_values)?,
        median_in_place(&mut source_values)?,
    ))
}

/// Fill one channel's missing tile offsets from their neighbours, growing
/// outward until every tile has one (or taking the global offset when none
/// does), then average each over the 3x3 tiles around it.
fn fill_and_smooth_offsets(
    offsets: &mut [Option<f32>],
    columns: usize,
    rows: usize,
    channels: usize,
    channel: usize,
    global_offset: f32,
) {
    let at = |column: usize, row: usize| (row * columns + column) * channels + channel;
    if (0..rows).all(|row| (0..columns).all(|column| offsets[at(column, row)].is_none())) {
        for row in 0..rows {
            for column in 0..columns {
                offsets[at(column, row)] = Some(global_offset);
            }
        }
        return;
    }
    let neighbourhood = |values: &[Option<f32>], column: usize, row: usize| {
        let (mut sum, mut count) = (0.0_f64, 0_u32);
        for nr in row.saturating_sub(1)..(row + 2).min(rows) {
            for nc in column.saturating_sub(1)..(column + 2).min(columns) {
                if let Some(value) = values[at(nc, nr)] {
                    sum += f64::from(value);
                    count += 1;
                }
            }
        }
        (count > 0).then(|| (sum / f64::from(count)) as f32)
    };
    while (0..rows).any(|row| (0..columns).any(|column| offsets[at(column, row)].is_none())) {
        let snapshot = offsets.to_vec();
        for row in 0..rows {
            for column in 0..columns {
                if snapshot[at(column, row)].is_none() {
                    offsets[at(column, row)] = neighbourhood(&snapshot, column, row);
                }
            }
        }
    }
    let snapshot = offsets.to_vec();
    for row in 0..rows {
        for column in 0..columns {
            offsets[at(column, row)] = neighbourhood(&snapshot, column, row);
        }
    }
}

/// Aperture radius, and the annulus that measures each star's local
/// background, in pixels.
const PHOTOMETRY_APERTURE: f64 = 5.0;
const PHOTOMETRY_ANNULUS: (f64, f64) = (8.0, 12.0);
/// Stars needed for a photometric gain.
const PHOTOMETRY_MINIMUM_STARS: usize = 12;

/// Per-channel gains that carry `source`'s star fluxes onto `reference`'s:
/// the median ratio of background-subtracted aperture fluxes at `stars`,
/// positions on the reference grid both images share.
///
/// A gain from star photometry follows transparency alone. One from the
/// frames' dispersion also follows their gradients and cloud, which inflate
/// a hazy frame's dispersion; that gain then comes out low, scaling the
/// frame's noise down, so inverse-noise weighting gave cloudy frames the
/// largest weights. Stars near saturation in the reference are skipped.
/// `None` when fewer than [`PHOTOMETRY_MINIMUM_STARS`] measure cleanly in a
/// channel.
pub(crate) fn photometric_gains(
    reference: &LinearImage,
    source: &LinearImage,
    stars: &[(f64, f64)],
) -> Option<Vec<f32>> {
    if !reference.dimensions_match(source) {
        return None;
    }
    let channels = reference.channels;
    let (width, height) = (reference.width, reference.height);
    let outer = PHOTOMETRY_ANNULUS.1.ceil() as usize + 1;
    (0..channels)
        .map(|channel| {
            let ceiling = reference
                .data
                .par_iter()
                .skip(channel)
                .step_by(channels)
                .copied()
                .filter(|value| value.is_finite())
                .reduce(|| f32::MIN, f32::max);
            let mut ratios = stars
                .iter()
                .filter_map(|&(x, y)| {
                    let (cx, cy) = (x.round() as isize, y.round() as isize);
                    if cx < outer as isize
                        || cy < outer as isize
                        || cx + outer as isize >= width as isize
                        || cy + outer as isize >= height as isize
                    {
                        return None;
                    }
                    let (cx, cy) = (cx as usize, cy as usize);
                    let mut reference_annulus = Vec::new();
                    let mut source_annulus = Vec::new();
                    let mut aperture = Vec::new();
                    for py in cy - outer..=cy + outer {
                        for px in cx - outer..=cx + outer {
                            let distance = (px as f64 - x).hypot(py as f64 - y);
                            let index = (py * width + px) * channels + channel;
                            let (r, s) = (reference.data[index], source.data[index]);
                            if !r.is_finite() || !s.is_finite() {
                                return None;
                            }
                            if distance <= PHOTOMETRY_APERTURE {
                                aperture.push((r, s));
                            } else if (PHOTOMETRY_ANNULUS.0..=PHOTOMETRY_ANNULUS.1)
                                .contains(&distance)
                            {
                                reference_annulus.push(r);
                                source_annulus.push(s);
                            }
                        }
                    }
                    if aperture.iter().any(|&(r, _)| r >= 0.85 * ceiling) {
                        return None;
                    }
                    let reference_sky = median_in_place(&mut reference_annulus)?;
                    let source_sky = median_in_place(&mut source_annulus)?;
                    let (reference_flux, source_flux) =
                        aperture
                            .iter()
                            .fold((0.0_f64, 0.0_f64), |(rf, sf), &(r, s)| {
                                (
                                    rf + f64::from(r - reference_sky),
                                    sf + f64::from(s - source_sky),
                                )
                            });
                    (reference_flux > 0.0 && source_flux > 0.0)
                        .then(|| (reference_flux / source_flux) as f32)
                })
                .collect::<Vec<_>>();
            if ratios.len() < PHOTOMETRY_MINIMUM_STARS {
                return None;
            }
            median_in_place(&mut ratios).filter(|gain| gain.is_finite() && *gain > 0.0)
        })
        .collect()
}

/// `value * gain + offset` per channel on whole interleaved pixels, leaving
/// non-finite samples alone. `mul_add` rounds once whether the CPU fuses it
/// or libm does, so the dispatched builds agree with the baseline one exactly;
/// they only avoid a call per sample.
#[multiversion::multiversion(targets("x86_64+avx2+fma", "aarch64+neon"))]
fn apply_affine(pixels: &mut [f32], gains: &[f32], offsets: &[f32]) {
    let channels = gains.len();
    for pixel in pixels.chunks_exact_mut(channels) {
        for ((value, &gain), &offset) in pixel.iter_mut().zip(gains).zip(offsets) {
            if value.is_finite() {
                *value = value.mul_add(gain, offset);
            }
        }
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
    // Every `stride`-th pixel of the region in row-major order, visited
    // directly rather than by testing each pixel's index.
    for sample_index in (0..width * height).step_by(stride) {
        let row = y + sample_index / width;
        let column = x + sample_index % width;
        let index = (row * source.width + column) * source.channels + channel;
        let reference_value = reference.data[index];
        let source_value = source.data[index];
        if reference_value.is_finite() && source_value.is_finite() {
            reference_values.push(reference_value);
            source_values.push(source_value);
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

    /// A frame whose sky gradient the reference lacks: a single offset leaves
    /// the gradient, which draws a step wherever this frame's edge falls in
    /// a stack; per-tile offsets remove it, with one gain for the frame.
    #[test]
    fn local_background_removes_a_gradient_with_one_gain() {
        let (width, height) = (192, 128);
        // Sky noise of about 20 from a sum of uniform draws, and a sparse
        // lattice of stars.
        let noise = (0..width * height)
            .map(|index| {
                let mut state = (index as u32).wrapping_mul(0x9e37_79b9) | 1;
                (0..6)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        (state % 1000) as f32 / 1000.0 - 0.5
                    })
                    .sum::<f32>()
                    * 28.0
            })
            .collect::<Vec<_>>();
        let scene = |index: usize| {
            let (x, y) = (index % width, index / width);
            1000.0 + noise[index] + if (x + 3 * y) % 41 == 0 { 400.0 } else { 0.0 }
        };
        let reference =
            LinearImage::new(width, height, 1, (0..width * height).map(scene).collect()).unwrap();
        let source = LinearImage::new(
            width,
            height,
            1,
            (0..width * height)
                .map(|index| 0.8 * scene(index) + 30.0 + (index % width) as f32 * 0.15)
                .collect(),
        )
        .unwrap();
        let worst = |mode| {
            let map = NormalizationMap::estimate(&reference, &source, mode).unwrap();
            let mut normalized = source.clone();
            map.apply(&mut normalized).unwrap();
            // The largest error in the mean of each 16x16 block, away from
            // the outermost tiles, where the smoothing has fewer neighbours:
            // the background, not single samples' noise.
            let mut worst = 0.0_f32;
            for block_y in (32..height - 32).step_by(16) {
                for block_x in (32..width - 32).step_by(16) {
                    let mut sum = 0.0_f32;
                    for y in block_y..block_y + 16 {
                        for x in block_x..block_x + 16 {
                            let index = y * width + x;
                            sum += normalized.data[index] - reference.data[index];
                        }
                    }
                    worst = worst.max((sum / 256.0).abs());
                }
            }
            (map, worst)
        };
        let (_, global) = worst(NormalizationMode::Global);
        let (map, local) = worst(NormalizationMode::LocalBackground { tile_size: 32 });
        assert!(global > 6.0, "{global}");
        assert!(local < 3.0, "{local}");
        let (minimum, maximum) = map.gain_range();
        assert_eq!(minimum, maximum, "one gain across the frame");
        let global_map =
            NormalizationMap::estimate(&reference, &source, NormalizationMode::Global).unwrap();
        assert_eq!(map.gain_range(), global_map.gain_range(), "the global gain");
    }

    /// A frame with half the reference's star flux under a gradient and a
    /// patch of cloud: star photometry recovers the gain of 2, where the
    /// frames' dispersion, inflated by the cloud and gradient, does not.
    #[test]
    fn photometric_gain_follows_stars_not_cloud() {
        let reference = crate::registration::test_star_field(false);
        let (width, height) = (reference.width, reference.height);
        let source = LinearImage::new(
            width,
            height,
            1,
            reference
                .data
                .iter()
                .enumerate()
                .map(|(index, &value)| {
                    let (x, y) = ((index % width) as f32, (index / width) as f32);
                    let cloud =
                        300.0 * (-((x - 380.0).powi(2) + (y - 100.0).powi(2)) / 6000.0).exp();
                    0.5 * value + 0.4 * x + cloud
                })
                .collect(),
        )
        .unwrap();
        let stars = crate::Registrar::new(&reference, crate::RegistrationOptions::default())
            .unwrap()
            .reference_star_positions();
        let gains = photometric_gains(&reference, &source, &stars).expect("enough stars");
        assert!((gains[0] - 2.0).abs() < 0.05, "{gains:?}");
        let dispersion =
            NormalizationMap::estimate(&reference, &source, NormalizationMode::Global).unwrap();
        assert!(
            (dispersion.mean_gain() - 2.0).abs() > 0.2,
            "the dispersion gain {} should be misled",
            dispersion.mean_gain()
        );
    }

    #[test]
    fn local_background_fills_tiles_the_frame_does_not_cover() {
        let (width, height) = (64, 64);
        let reference = LinearImage::new(
            width,
            height,
            1,
            (0..width * height)
                .map(|index| 500.0 + (index % 29) as f32)
                .collect(),
        )
        .unwrap();
        let mut source = reference.clone();
        for (index, value) in source.data.iter_mut().enumerate() {
            *value = if index % width < 20 {
                f32::NAN
            } else {
                *value + 40.0
            };
        }
        let map = NormalizationMap::estimate(
            &reference,
            &source,
            NormalizationMode::LocalBackground { tile_size: 16 },
        )
        .unwrap();
        map.validate().unwrap();
        assert!(map.offsets.iter().all(|offset| (offset + 40.0).abs() < 1.0));
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
        // Drizzle reads the same coefficients one sample at a time.
        let sampler = map.sampler();
        for (x, y) in [(0, 0), (15, 16), (31, 7), (20, 31)] {
            let (gain, offset) = sampler.at(x, y, 0);
            assert_eq!(1.0_f32.mul_add(gain, offset), full.data[y * 32 + x]);
        }
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
    fn channel_mean_gain_averages_one_channel_across_tiles() {
        let map = NormalizationMap {
            schema_version: NORMALIZATION_MAP_SCHEMA_VERSION,
            width: 32,
            height: 16,
            channels: 3,
            tile_size: 16,
            columns: 2,
            rows: 1,
            gains: vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
            offsets: vec![0.0; 6],
        };
        map.validate().unwrap();
        assert_eq!(map.channel_mean_gain(0), 2.0);
        assert_eq!(map.channel_mean_gain(1), 3.0);
        assert_eq!(map.channel_mean_gain(2), 4.0);
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
