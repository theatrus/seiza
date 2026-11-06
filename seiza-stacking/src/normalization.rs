use crate::{Error, LinearImage, ReferenceRegion, Result, image::Placed};
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
    /// tile's offset is the median difference of corresponding samples,
    /// after applying the gain, so shared nebular structure cancels; a
    /// stack leaves out the samples about the reference's brightest stars,
    /// whose light seeing spreads differently in the two frames, and tiles
    /// with too few are filled from their neighbours, and the grid is
    /// smoothed over three tiles. On an expanded canvas the live pass fits a
    /// single tile: the canvas keeps the first value each sample received,
    /// so its background is a patchwork of single frames' skies that
    /// matching tile by tile would copy into the stack. A reintegration fits
    /// the tiles against a reference integrated from the best frames.
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
    /// star photometry) for [`NormalizationMode::LocalBackground`], and its
    /// background offsets fitted on the samples `mask` leaves, when there is
    /// one (see [`BackgroundMask`]). Other modes, and a missing measurement,
    /// fit as [`Self::estimate`] does, with the mask.
    pub(crate) fn estimate_with_gains(
        reference: &LinearImage,
        source: Placed<'_>,
        mode: NormalizationMode,
        gains: Option<&[f32]>,
        mask: Option<&BackgroundMask>,
    ) -> Result<Self> {
        match (mode, gains) {
            (NormalizationMode::LocalBackground { tile_size }, Some(gains))
                if gains.len() == source.channels() && source.matches(reference) =>
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
                        let whole = (0, 0, source.width, source.height);
                        // The frame's one offset, for channels no tile
                        // measures, from every sample where the mask
                        // leaves too few.
                        let offset = tile_offset(reference, source, channel, whole, gain, mask)
                            .or_else(|| {
                                mask.and_then(|_| {
                                    tile_offset(reference, source, channel, whole, gain, None)
                                })
                            })
                            .ok_or_else(|| {
                                Error::Normalization(
                                    "too few overlapping finite pixels for normalization".into(),
                                )
                            })?;
                        Ok((gain, offset))
                    })
                    .collect::<Result<Vec<_>>>()?;
                local_background_with_gains(reference, source, tile_size, &globals, mask)
            }
            _ => Self::estimate_masked(reference, source, mode, mask),
        }
    }

    /// Fit [`NormalizationMode::LocalBackground`] offsets against `reference`
    /// while keeping this map's per-channel gains, as fitted against the
    /// stack's reference frame. A gain fitted against a reference of
    /// another noise level, such as an integration of many frames, would
    /// follow the noise rather than the signal.
    ///
    /// Reintegration fits from the samples alone, with
    /// [`Self::refit_background_samples`]; this whole-image form stays as
    /// its test reference.
    #[cfg(test)]
    pub(crate) fn refit_background(
        &self,
        reference: &LinearImage,
        source: &LinearImage,
        tile_size: usize,
        mask: Option<&BackgroundMask>,
    ) -> Result<Self> {
        if source.channels != self.channels {
            return Err(Error::Normalization(
                "normalization channel count does not match".into(),
            ));
        }
        local_background_with_gains(
            reference,
            Placed::whole(source),
            tile_size,
            &self.background_globals(),
            mask,
        )
    }

    /// [`Self::refit_background`] from the samples `layout` says the fit
    /// reads, of the reference and of the source, rather than from whole
    /// images; the same fit, to the bit, leaving out those `mask` covers.
    pub(crate) fn refit_background_samples(
        &self,
        layout: &BackgroundSamples,
        reference: &[f32],
        source: &[f32],
        mask: Option<&BackgroundMask>,
    ) -> Result<Self> {
        if layout.channels != self.channels
            || reference.len() != layout.len()
            || source.len() != layout.len()
        {
            return Err(Error::Normalization(
                "background samples do not match the normalization".into(),
            ));
        }
        let globals = self.background_globals();
        Ok(local_background_from(
            (layout.width, layout.height, layout.channels),
            layout.tile_size,
            &globals,
            |cell, channel, (x, y, width, height)| {
                let samples = layout.block(cell, channel);
                let pairs = reference[samples.clone()]
                    .iter()
                    .zip(&source[samples])
                    .zip(tile_sample_pixels(layout.width, x, y, width, height))
                    .filter(|&(_, pixel)| !mask.is_some_and(|mask| mask.covers(pixel)))
                    .map(|((&reference, &source), _)| (reference, source));
                sample_offset(pairs, globals[channel].0)
            },
        ))
    }

    /// Fit additive residuals after applying this recorded spatial map.
    /// The base remains separate: a native sample grid and a rescaled map
    /// can have different tile sizes, so adding their coefficient arrays
    /// would change the base's interpolated surface. Missing tiles inherit
    /// neighbouring residuals; a channel without support keeps zero residual
    /// rather than erasing its established overlap corrections.
    pub(crate) fn refit_background_residual_samples(
        &self,
        layout: &BackgroundSamples,
        reference: &[f32],
        source: &[f32],
        mask: Option<&BackgroundMask>,
    ) -> Result<Self> {
        self.validate()?;
        if (layout.width, layout.height, layout.channels)
            != (self.width, self.height, self.channels)
            || reference.len() != layout.len()
            || source.len() != layout.len()
        {
            return Err(Error::Normalization(
                "background samples do not match the normalization".into(),
            ));
        }
        let sampler = self.sampler();
        let globals = vec![(1.0, 0.0); self.channels];
        Ok(local_background_from(
            (layout.width, layout.height, layout.channels),
            layout.tile_size,
            &globals,
            |cell, channel, (x, y, width, height)| {
                let block = layout.block(cell, channel);
                let pairs = reference[block.clone()]
                    .iter()
                    .zip(&source[block])
                    .zip(tile_sample_pixels(layout.width, x, y, width, height))
                    .filter(|&(_, pixel)| !mask.is_some_and(|mask| mask.covers(pixel)))
                    .map(|((&reference, &value), pixel)| {
                        if !value.is_finite() {
                            return (reference, value);
                        }
                        let (gain, offset) =
                            sampler.at(pixel % layout.width, pixel / layout.width, channel);
                        (reference, value.mul_add(gain, offset))
                    });
                sample_offset(pairs, 1.0)
            },
        ))
    }

    /// Each channel's gain, and its mean offset over the tiles, which a
    /// background refit keeps.
    fn background_globals(&self) -> Vec<(f32, f32)> {
        let tiles = (self.columns * self.rows) as f32;
        (0..self.channels)
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
            .collect()
    }
}

/// The samples [`NormalizationMode::LocalBackground`]'s fit reads from an
/// image of one shape: in each tile, in the order of the map's offsets,
/// every channel at an evenly spaced subset of the tile's pixels, about one
/// in sixteen of a full tile's, as [`tile_offset`] picks them.
///
/// A reintegration keeps each frame's samples beside its registered image,
/// and fits the frame's background from them rather than from the whole
/// image.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BackgroundSamples {
    width: usize,
    height: usize,
    channels: usize,
    tile_size: usize,
    columns: usize,
    /// Where each tile's and channel's samples start, and where the last
    /// ends.
    starts: Vec<usize>,
}

impl BackgroundSamples {
    /// The layout for images of `width` by `height` pixels of `channels`
    /// channels fitted in tiles of `tile_size` pixels.
    pub(crate) fn new(width: usize, height: usize, channels: usize, tile_size: usize) -> Self {
        let columns = width.div_ceil(tile_size);
        let rows = height.div_ceil(tile_size);
        let mut starts = Vec::with_capacity(columns * rows * channels + 1);
        let mut start = 0;
        for cell in 0..columns * rows {
            let (_, _, tile_width, tile_height) = tile(cell, columns, width, height, tile_size);
            let count = tile_sample_pixels(width, 0, 0, tile_width, tile_height).len();
            for _ in 0..channels {
                starts.push(start);
                start += count;
            }
        }
        starts.push(start);
        Self {
            width,
            height,
            channels,
            tile_size,
            columns,
            starts,
        }
    }

    /// The layout an image of `image`'s shape is fitted with.
    pub(crate) fn for_image(image: &LinearImage, tile_size: usize) -> Self {
        Self::new(image.width, image.height, image.channels, tile_size)
    }

    /// Samples per image.
    pub(crate) fn len(&self) -> usize {
        self.starts[self.starts.len() - 1]
    }

    /// Samples of tiles intersecting `region`, in their original order,
    /// one run per tile row, including every channel. Samples outside these
    /// runs are all the canonical `NaN` that [`Placed::sample`] supplies
    /// beyond a footprint. Boundary tiles stay whole, preserving missing
    /// pixels and NaN payloads within the footprint exactly.
    pub(crate) fn ranges_for_region(
        &self,
        region: ReferenceRegion,
    ) -> Option<impl Iterator<Item = std::ops::Range<usize>> + '_> {
        let right = region.x.checked_add(region.width)?;
        let bottom = region.y.checked_add(region.height)?;
        if region.width == 0 || region.height == 0 || right > self.width || bottom > self.height {
            return None;
        }
        let first_column = region.x / self.tile_size;
        let end_column = right.div_ceil(self.tile_size);
        let rows = region.y / self.tile_size..bottom.div_ceil(self.tile_size);
        Some(rows.map(move |row| {
            let first = (row * self.columns + first_column) * self.channels;
            let end = (row * self.columns + end_column) * self.channels;
            self.starts[first]..self.starts[end]
        }))
    }

    /// Where tile `cell`'s samples of `channel` lie.
    fn block(&self, cell: usize, channel: usize) -> std::ops::Range<usize> {
        let at = cell * self.channels + channel;
        self.starts[at]..self.starts[at + 1]
    }

    /// The channel of every sample, in order, block by block: a tile's
    /// samples of channel 0, then of channel 1, and so on.
    pub(crate) fn channels(&self) -> impl Iterator<Item = (usize, std::ops::Range<usize>)> + '_ {
        self.starts
            .windows(2)
            .enumerate()
            .map(|(at, bounds)| (at % self.channels, bounds[0]..bounds[1]))
    }

    /// `image`'s samples, as they are.
    pub(crate) fn of(&self, image: &LinearImage) -> Result<Vec<f32>> {
        self.of_placed(Placed::whole(image))
    }

    /// [`Self::of`] an image placed on a grid of this layout's shape.
    pub(crate) fn of_placed(&self, image: Placed<'_>) -> Result<Vec<f32>> {
        if (image.width, image.height, image.channels()) != (self.width, self.height, self.channels)
        {
            return Err(Error::Normalization(
                "image does not match its background sample layout".into(),
            ));
        }
        let mut samples = Vec::with_capacity(self.len());
        let rows = self.height.div_ceil(self.tile_size);
        for cell in 0..self.columns * rows {
            let (x, y, width, height) =
                tile(cell, self.columns, self.width, self.height, self.tile_size);
            for channel in 0..self.channels {
                samples.extend(
                    tile_sample_pixels(self.width, x, y, width, height)
                        .map(|pixel| image.sample(pixel * self.channels + channel)),
                );
            }
        }
        Ok(samples)
    }
}

/// Every pixel the existing CPU preparation may inspect, including the
/// fallback gain fit and the replay's background sample grid. No values are
/// approximated: gathered frames use these same CPU fits and sample order.
/// Photometry reads up to `reach` from each of `stars` along either axis,
/// the reference's [`ReferencePhotometry::reach`].
#[cfg(feature = "gpu")]
pub(crate) fn preparation_sample_indices(
    (width, height, channels): (usize, usize, usize),
    mode: NormalizationMode,
    stars: &[(f64, f64)],
    reach: usize,
    background: Option<&BackgroundSamples>,
) -> Vec<u32> {
    // Each pixel read, once, in a bit of its own: the stars' circles
    // overlap the tiles' samples and each other, and marking them costs a
    // bit a pixel where collecting, sorting and deduplicating every index
    // cost the indices of every channel of every pixel, many times over.
    let mut read = vec![0_u64; (width * height).div_ceil(64)];
    let mut mark = |pixel: usize| read[pixel / 64] |= 1 << (pixel % 64);
    let mut add = |(x, y, region_width, region_height): (usize, usize, usize, usize),
                   target: usize| {
        let stride = (region_width * region_height / target).max(1);
        for sample in (0..region_width * region_height).step_by(stride) {
            mark((y + sample / region_width) * width + x + sample % region_width);
        }
    };
    let whole = (0, 0, width, height);
    if !matches!(mode, NormalizationMode::None) {
        add(whole, 20_000);
    }
    if let NormalizationMode::LocalBackground { tile_size }
    | NormalizationMode::Local { tile_size } = mode
    {
        let columns = width.div_ceil(tile_size);
        let rows = height.div_ceil(tile_size);
        let target = if matches!(mode, NormalizationMode::LocalBackground { .. }) {
            4_000
        } else {
            20_000
        };
        for cell in 0..columns * rows {
            add(tile(cell, columns, width, height, tile_size), target);
        }
    }
    if matches!(mode, NormalizationMode::LocalBackground { .. }) {
        add(whole, 4_000);
        let (reach, within) = (reach as isize, (reach * reach) as f64);
        for &(x, y) in stars {
            let (cx, cy) = (x.round() as isize, y.round() as isize);
            // Photometry reads within a circle about each star, smaller for
            // a sharper frame, which may fit beside an edge that the whole
            // reach would cross.
            let rows = (cy - reach).max(0)..=(cy + reach).min(height as isize - 1);
            let columns = (cx - reach).max(0)..=(cx + reach).min(width as isize - 1);
            for py in rows {
                for px in columns.clone() {
                    if (px as f64 - x).powi(2) + (py as f64 - y).powi(2) <= within {
                        mark(py as usize * width + px as usize);
                    }
                }
            }
        }
    }
    if let Some(layout) = background {
        for cell in 0..layout.columns * layout.height.div_ceil(layout.tile_size) {
            let (x, y, tile_width, tile_height) = tile(
                cell,
                layout.columns,
                layout.width,
                layout.height,
                layout.tile_size,
            );
            for pixel in tile_sample_pixels(layout.width, x, y, tile_width, tile_height) {
                mark(pixel);
            }
        }
    }
    let mut indices = Vec::new();
    for (word, &bits) in read.iter().enumerate() {
        let mut bits = bits;
        while bits != 0 {
            let pixel = word * 64 + bits.trailing_zeros() as usize;
            indices.extend((0..channels).map(|channel| (pixel * channels + channel) as u32));
            bits &= bits - 1;
        }
    }
    indices
}

/// Tile `cell`'s position and size, counting tiles row by row, `columns`
/// to a row, over an image of `width` by `height`.
fn tile(
    cell: usize,
    columns: usize,
    width: usize,
    height: usize,
    tile_size: usize,
) -> (usize, usize, usize, usize) {
    let (x, y) = ((cell % columns) * tile_size, (cell / columns) * tile_size);
    (x, y, tile_size.min(width - x), tile_size.min(height - y))
}

/// The pixels [`tile_offset`] reads in the region at `(x, y)` of `width`
/// by `height` of an image `image_width` wide: every `stride`-th pixel in
/// row-major order, for about four thousand.
fn tile_sample_pixels(
    image_width: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> TileSamples {
    let stride = (width * height / 4_000).max(1);
    TileSamples {
        remaining: (width * height).div_ceil(stride),
        row: y,
        column: 0,
        stride,
        width,
        image_width,
        x,
    }
}

/// [`tile_sample_pixels`]' pixels, stepping a row and column rather than
/// dividing for each.
#[derive(Clone)]
struct TileSamples {
    remaining: usize,
    row: usize,
    column: usize,
    stride: usize,
    width: usize,
    image_width: usize,
    x: usize,
}

impl Iterator for TileSamples {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let pixel = self.row * self.image_width + self.x + self.column;
        self.column += self.stride;
        if self.column >= self.width {
            self.row += self.column / self.width;
            self.column %= self.width;
        }
        Some(pixel)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for TileSamples {}

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
        Self::identity_of(image.width, image.height, image.channels)
    }

    /// [`Self::identity`] for an image of `width` by `height` pixels of
    /// `channels` channels.
    fn identity_of(width: usize, height: usize, channels: usize) -> Self {
        Self {
            schema_version: NORMALIZATION_MAP_SCHEMA_VERSION,
            width,
            height,
            channels,
            tile_size: width.max(height),
            columns: 1,
            rows: 1,
            gains: vec![1.0; channels],
            offsets: vec![0.0; channels],
        }
    }

    /// Fit gains and offsets that match `source`'s background to `reference`
    /// using robust median and dispersion statistics.
    pub fn estimate(
        reference: &LinearImage,
        source: &LinearImage,
        mode: NormalizationMode,
    ) -> Result<Self> {
        Self::estimate_placed(reference, Placed::whole(source), mode)
    }

    /// [`Self::estimate`] for a source placed on a grid of the reference's
    /// shape.
    pub(crate) fn estimate_placed(
        reference: &LinearImage,
        source: Placed<'_>,
        mode: NormalizationMode,
    ) -> Result<Self> {
        Self::estimate_masked(reference, source, mode, None)
    }

    /// [`Self::estimate_placed`], with [`NormalizationMode::LocalBackground`]'s
    /// offsets fitted on the samples `mask` leaves.
    fn estimate_masked(
        reference: &LinearImage,
        source: Placed<'_>,
        mode: NormalizationMode,
        mask: Option<&BackgroundMask>,
    ) -> Result<Self> {
        if !source.matches(reference) {
            return Err(Error::Normalization(
                "reference and source dimensions must match".into(),
            ));
        }
        match mode {
            NormalizationMode::None => Ok(Self::identity_of(
                source.width,
                source.height,
                source.channels(),
            )),
            NormalizationMode::Global => {
                let mut map = Self::identity_of(source.width, source.height, source.channels());
                for channel in 0..source.channels() {
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
                local_background(reference, source, tile_size, mask)
            }
            NormalizationMode::Local { tile_size } => {
                if tile_size < 16 {
                    return Err(Error::Normalization(
                        "local normalization tile size must be at least 16 pixels".into(),
                    ));
                }
                let columns = source.width.div_ceil(tile_size);
                let rows = source.height.div_ceil(tile_size);
                let cell_count = columns * rows * source.channels();
                let coefficients = (0..cell_count)
                    .into_par_iter()
                    .map(|index| {
                        let channel = index % source.channels();
                        let cell = index / source.channels();
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
                    channels: source.channels(),
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
        let rows = RowNormalizer::new(self, origin_x, image.width);
        // Rows are taken in blocks, so each task works out a tile row's
        // horizontal interpolation once for every row it normalizes.
        const BLOCK_ROWS: usize = 64;
        let row_samples = image.width * image.channels;
        image
            .data
            .par_chunks_mut(row_samples * BLOCK_ROWS)
            .enumerate()
            .for_each_init(RowCoefficients::default, |coefficients, (block, data)| {
                rows.apply(coefficients, data, origin_y + block * BLOCK_ROWS);
            });
        Ok(())
    }

    /// What applying this map to a region `width` by `height` at
    /// `(origin_x, origin_y)` of its grid takes, for the GPU to apply it as
    /// [`Self::apply_region`] does: each column's and row's tile weights as
    /// [`RowNormalizer`] works them out, and the tiles' gains and offsets.
    /// `None` when the region does not fit the grid.
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_coefficients(
        &self,
        (origin_x, origin_y): (usize, usize),
        (width, height, channels): (usize, usize, usize),
    ) -> Option<crate::gpu::NormalizationCoefficients> {
        if self.validate().is_err()
            || channels != self.channels
            || origin_x + width > self.width
            || origin_y + height > self.height
        {
            return None;
        }
        let global = self.columns == 1 && self.rows == 1;
        let weights = |coordinates: std::ops::Range<usize>, cells: usize| {
            if global {
                return Vec::new();
            }
            coordinates
                .map(|coordinate| {
                    let weights = axis_weights(coordinate, cells, self.tile_size);
                    [
                        weights.low as u32,
                        weights.high as u32,
                        weights.fraction.to_bits(),
                    ]
                })
                .collect()
        };
        Some(crate::gpu::NormalizationCoefficients {
            global,
            tile_columns: self.columns as u32,
            columns: weights(origin_x..origin_x + width, self.columns),
            rows: weights(origin_y..origin_y + height, self.rows),
            gains: self.gains.clone(),
            offsets: self.offsets.clone(),
        })
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

    /// [`Self::apply_global`] on the samples `layout` takes of an image
    /// with this map's channels: the same arithmetic on each.
    pub(crate) fn apply_global_samples(
        &self,
        layout: &BackgroundSamples,
        samples: &mut [f32],
    ) -> Result<()> {
        self.validate()?;
        if self.columns != 1 || self.rows != 1 {
            return Err(Error::Normalization(
                "normalization map is not global".into(),
            ));
        }
        if layout.channels != self.channels || samples.len() != layout.len() {
            return Err(Error::Normalization(
                "background samples do not match the normalization".into(),
            ));
        }
        for (channel, block) in layout.channels() {
            apply_affine(
                &mut samples[block],
                &self.gains[channel..=channel],
                &self.offsets[channel..=channel],
            );
        }
        Ok(())
    }

    /// Apply the complete spatial map at the exact coordinates the layout
    /// sampled. The map and sample tiles need not coincide.
    pub(crate) fn apply_samples(
        &self,
        layout: &BackgroundSamples,
        samples: &mut [f32],
    ) -> Result<()> {
        self.validate()?;
        if (layout.width, layout.height, layout.channels)
            != (self.width, self.height, self.channels)
            || samples.len() != layout.len()
        {
            return Err(Error::Normalization(
                "background samples do not match the normalization".into(),
            ));
        }
        if self.is_global() {
            return self.apply_global_samples(layout, samples);
        }
        let sampler = self.sampler();
        for cell in 0..layout.columns * layout.height.div_ceil(layout.tile_size) {
            let (x, y, width, height) = tile(
                cell,
                layout.columns,
                layout.width,
                layout.height,
                layout.tile_size,
            );
            for channel in 0..layout.channels {
                for (value, pixel) in samples[layout.block(cell, channel)]
                    .iter_mut()
                    .zip(tile_sample_pixels(layout.width, x, y, width, height))
                {
                    if value.is_finite() {
                        let (gain, offset) =
                            sampler.at(pixel % layout.width, pixel / layout.width, channel);
                        *value = value.mul_add(gain, offset);
                    }
                }
            }
        }
        Ok(())
    }

    /// Whether a residual leaves every sample untouched. Skipping it also
    /// preserves the base map's signed zeroes exactly.
    pub(crate) fn is_identity(&self) -> bool {
        self.gains.iter().all(|&gain| gain == 1.0)
            && self.offsets.iter().all(|&offset| offset == 0.0)
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

    /// The same map on a grid `scale` times as fine: its tiles grow with it,
    /// so the tile grid and every coefficient stand and each finer pixel
    /// reads the weights of the coarse pixel it lies in.
    pub(crate) fn rescaled(&self, scale: usize) -> Self {
        Self {
            width: self.width * scale,
            height: self.height * scale,
            tile_size: self.tile_size * scale,
            ..self.clone()
        }
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

/// Applies a map to whole rows of an image or a crop of it, a run of rows
/// at a time, giving every sample exactly what [`NormalizationMap::apply`]
/// gives it.
///
/// Each sample's gain and offset are interpolated bilinearly between the
/// four tiles around it. Every row between two tile centres shares those
/// tiles, so the interpolation across columns is worked out once for such a
/// run of rows ([`RowCoefficients`]); each row then needs only the
/// interpolation down the column, with the same arithmetic in the same
/// order as [`bilinear`].
pub(crate) struct RowNormalizer<'a> {
    map: &'a NormalizationMap,
    /// The tile weights of each column, empty for a one-tile map.
    columns: Vec<AxisWeights>,
}

/// One task's interpolation across columns for the rows between two tile
/// centres: the gains and offsets along the tile row above and below.
#[derive(Default)]
pub(crate) struct RowCoefficients {
    /// The tile rows above and below that the vectors hold.
    tile_rows: Option<(usize, usize)>,
    top_gain: Vec<f32>,
    bottom_gain: Vec<f32>,
    top_offset: Vec<f32>,
    bottom_offset: Vec<f32>,
}

impl<'a> RowNormalizer<'a> {
    /// A normalizer for rows `width` pixels wide starting at column
    /// `origin_x` of the grid `map` was fitted on.
    pub(crate) fn new(map: &'a NormalizationMap, origin_x: usize, width: usize) -> Self {
        let columns = if map.columns == 1 && map.rows == 1 {
            Vec::new()
        } else {
            (0..width)
                .map(|x| axis_weights(origin_x + x, map.columns, map.tile_size))
                .collect()
        };
        Self { map, columns }
    }

    /// Normalize `data`, whole rows of interleaved samples, the first of
    /// which is row `first_row` of the fitted grid. Non-finite samples are
    /// left as they are.
    pub(crate) fn apply(
        &self,
        coefficients: &mut RowCoefficients,
        data: &mut [f32],
        first_row: usize,
    ) {
        let map = self.map;
        if self.columns.is_empty() {
            apply_affine(data, &map.gains, &map.offsets);
            return;
        }
        let channels = map.channels;
        let row_samples = self.columns.len() * channels;
        for (offset, row) in data.chunks_exact_mut(row_samples).enumerate() {
            let rows = axis_weights(first_row + offset, map.rows, map.tile_size);
            if coefficients.tile_rows != Some((rows.low, rows.high)) {
                coefficients.fill(map, &self.columns, rows.low, rows.high);
            }
            apply_interpolated(
                row,
                &coefficients.top_gain,
                &coefficients.bottom_gain,
                &coefficients.top_offset,
                &coefficients.bottom_offset,
                rows.fraction,
            );
        }
    }
}

impl RowCoefficients {
    /// Interpolate tile rows `top` and `bottom` across every column, as the
    /// first two lines of [`bilinear`] do.
    fn fill(&mut self, map: &NormalizationMap, columns: &[AxisWeights], top: usize, bottom: usize) {
        let channels = map.channels;
        let samples = columns.len() * channels;
        for vector in [
            &mut self.top_gain,
            &mut self.bottom_gain,
            &mut self.top_offset,
            &mut self.bottom_offset,
        ] {
            vector.resize(samples, 0.0);
        }
        let (top, bottom) = (top * map.columns, bottom * map.columns);
        for (x, weights) in columns.iter().enumerate() {
            let (left, right) = (1.0 - weights.fraction, weights.fraction);
            for channel in 0..channels {
                let at = x * channels + channel;
                let corner = |row: usize, column: usize| (row + column) * channels + channel;
                let (top_left, top_right) = (corner(top, weights.low), corner(top, weights.high));
                let (bottom_left, bottom_right) =
                    (corner(bottom, weights.low), corner(bottom, weights.high));
                self.top_gain[at] = map.gains[top_left] * left + map.gains[top_right] * right;
                self.bottom_gain[at] =
                    map.gains[bottom_left] * left + map.gains[bottom_right] * right;
                self.top_offset[at] = map.offsets[top_left] * left + map.offsets[top_right] * right;
                self.bottom_offset[at] =
                    map.offsets[bottom_left] * left + map.offsets[bottom_right] * right;
            }
        }
        self.tile_rows = Some((top / map.columns, bottom / map.columns));
    }
}

/// `value * gain + offset` on one row, with each gain and offset
/// interpolated down the column from the tile rows above and below, as the
/// last line of [`bilinear`] does. Non-finite samples are left alone.
/// `mul_add` rounds once however it is computed, so the dispatched builds
/// match the baseline one.
#[multiversion::multiversion(targets("x86_64+avx2+fma", "aarch64+neon"))]
fn apply_interpolated(
    row: &mut [f32],
    top_gain: &[f32],
    bottom_gain: &[f32],
    top_offset: &[f32],
    bottom_offset: &[f32],
    fraction: f32,
) {
    let above = 1.0 - fraction;
    for ((((value, &top_gain), &bottom_gain), &top_offset), &bottom_offset) in row
        .iter_mut()
        .zip(top_gain)
        .zip(bottom_gain)
        .zip(top_offset)
        .zip(bottom_offset)
    {
        let gain = top_gain * above + bottom_gain * fraction;
        let offset = top_offset * above + bottom_offset * fraction;
        if value.is_finite() {
            *value = value.mul_add(gain, offset);
        }
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
    #[inline(always)]
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

#[inline]
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
/// Each tile's offset is the median of paired residuals after applying
/// that gain. Subtracting corresponding samples cancels shared nebulosity
/// before fitting the background; subtracting two separate medians does not.
fn local_background(
    reference: &LinearImage,
    source: Placed<'_>,
    tile_size: usize,
    mask: Option<&BackgroundMask>,
) -> Result<NormalizationMap> {
    let channels = source.channels();
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
    local_background_with_gains(reference, source, tile_size, &globals, mask)
}

/// Background offsets per tile for fixed per-channel gains, each paired with
/// the global offset used where no tile of that channel can be measured,
/// from the samples `mask` leaves.
fn local_background_with_gains(
    reference: &LinearImage,
    source: Placed<'_>,
    tile_size: usize,
    globals: &[(f32, f32)],
    mask: Option<&BackgroundMask>,
) -> Result<NormalizationMap> {
    let gains: Vec<f32> = globals.iter().map(|&(gain, _)| gain).collect();
    #[cfg(feature = "gpu")]
    if let Some(sampled) = source.sampled
        && let Some(tiles) = sampled.tiles
        && tiles.fits(reference, source, tile_size, mask)
    {
        return Ok(local_background_by_tile(
            (source.width, source.height, source.channels()),
            tile_size,
            globals,
            |cell, _, offsets| tiles.offsets(cell, sampled.values, &gains, offsets),
        ));
    }
    Ok(local_background_by_tile(
        (source.width, source.height, source.channels()),
        tile_size,
        globals,
        |_, (x, y, width, height), offsets| {
            // A tile the source does not reach shares no samples with the
            // reference.
            if source.reaches(x, y, width, height) {
                tile_offsets(
                    reference,
                    source,
                    (x, y, width, height),
                    &gains,
                    mask,
                    offsets,
                );
            }
        },
    ))
}

/// [`local_background_with_gains`] over an image of `shape` (width, height,
/// channels), with `offset` giving the median paired residual in a tile,
/// by its index, channel and region, when they share enough samples.
fn local_background_from(
    shape: (usize, usize, usize),
    tile_size: usize,
    globals: &[(f32, f32)],
    offset: impl Fn(usize, usize, (usize, usize, usize, usize)) -> Option<f32> + Sync,
) -> NormalizationMap {
    local_background_by_tile(shape, tile_size, globals, |cell, region, offsets| {
        for (channel, slot) in offsets.iter_mut().enumerate() {
            *slot = offset(cell, channel, region);
        }
    })
}

/// [`local_background_from`], with `offsets` filling every channel's offset
/// of a tile at once, by the tile's index and region, so a tile's samples
/// are read once for all its channels.
fn local_background_by_tile(
    shape: (usize, usize, usize),
    tile_size: usize,
    globals: &[(f32, f32)],
    offsets: impl Fn(usize, (usize, usize, usize, usize), &mut [Option<f32>]) + Sync,
) -> NormalizationMap {
    let (width, height, channels) = shape;
    let columns = width.div_ceil(tile_size);
    let rows = height.div_ceil(tile_size);
    let mut offsets_by_tile = vec![None; columns * rows * channels];
    offsets_by_tile
        .par_chunks_mut(channels.max(1))
        .enumerate()
        .for_each(|(cell, tile_offsets)| {
            offsets(
                cell,
                tile(cell, columns, width, height, tile_size),
                tile_offsets,
            );
            for slot in tile_offsets.iter_mut() {
                *slot = slot.filter(|offset| offset.is_finite());
            }
        });
    let mut offsets = offsets_by_tile;
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
    NormalizationMap {
        schema_version: NORMALIZATION_MAP_SCHEMA_VERSION,
        width,
        height,
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
    }
}

/// The median residual in one region and channel, over samples both frames
/// cover after applying `gain` and `mask` leaves, or `None` when they share
/// too few.
fn tile_offset(
    reference: &LinearImage,
    source: Placed<'_>,
    channel: usize,
    (x, y, width, height): (usize, usize, usize, usize),
    gain: f32,
    mask: Option<&BackgroundMask>,
) -> Option<f32> {
    let pairs = tile_sample_pixels(source.width, x, y, width, height)
        .filter(|&pixel| !mask.is_some_and(|mask| mask.covers(pixel)))
        .map(|pixel| {
            let index = pixel * source.channels() + channel;
            (reference.data[index], source.sample(index))
        });
    sample_offset(pairs, gain)
}

/// Each local background tile's samples in a frame the GPU gathered: where
/// each sits among the gathered values, and the reference's value there,
/// in the order [`tile_offsets`] reads them. A frame's fit then reads two
/// short arrays in order, rather than its lookup table and the reference
/// at a stride, a cache line for every sample of both, and finds the same
/// residuals in the same order. Built once a stack, for its reference.
#[cfg(feature = "gpu")]
#[derive(Debug)]
pub(crate) struct GatheredTiles {
    shape: (usize, usize, usize),
    tile_size: usize,
    /// Where each tile's samples start, and one more for the last's end.
    starts: Vec<usize>,
    at: Vec<u32>,
    reference: Vec<f32>,
    /// Whether the samples a [`BackgroundMask`] covers are left out.
    masked: bool,
}

#[cfg(feature = "gpu")]
impl GatheredTiles {
    /// The tiles of `tile_size` over `reference`, whose every sample
    /// `lookup` places among a frame's gathered values, but those `mask`
    /// covers; `None` when one is not gathered.
    pub(crate) fn new(
        reference: &LinearImage,
        tile_size: usize,
        lookup: &[u32],
        mask: Option<&BackgroundMask>,
    ) -> Option<Self> {
        let (width, height, channels) = (reference.width, reference.height, reference.channels);
        if tile_size == 0 || lookup.len() != reference.data.len() {
            return None;
        }
        let columns = width.div_ceil(tile_size);
        let cells = columns * height.div_ceil(tile_size);
        let mut starts = Vec::with_capacity(cells + 1);
        let (mut at, mut values) = (Vec::new(), Vec::new());
        for cell in 0..cells {
            starts.push(at.len());
            let (x, y, tile_width, tile_height) = tile(cell, columns, width, height, tile_size);
            for pixel in tile_sample_pixels(width, x, y, tile_width, tile_height) {
                if mask.is_some_and(|mask| mask.covers(pixel)) {
                    continue;
                }
                for channel in 0..channels {
                    let index = pixel * channels + channel;
                    let position = lookup[index];
                    if position == u32::MAX {
                        return None;
                    }
                    at.push(position);
                    values.push(reference.data[index]);
                }
            }
        }
        starts.push(at.len());
        Some(Self {
            shape: (width, height, channels),
            tile_size,
            starts,
            at,
            reference: values,
            masked: mask.is_some(),
        })
    }

    /// Whether these are the tiles `tile_offsets` would read for `source`,
    /// a whole frame, against `reference`, with `mask`, the one they were
    /// built with when there is one.
    fn fits(
        &self,
        reference: &LinearImage,
        source: Placed<'_>,
        tile_size: usize,
        mask: Option<&BackgroundMask>,
    ) -> bool {
        let shape = (source.width, source.height, source.channels());
        source.is_whole()
            && self.tile_size == tile_size
            && self.shape == shape
            && self.masked == mask.is_some()
            && (reference.width, reference.height, reference.channels) == shape
    }

    /// [`tile_offsets`] for tile `cell` of a frame whose gathered samples
    /// are `values`.
    fn offsets(&self, cell: usize, values: &[f32], gains: &[f32], offsets: &mut [Option<f32>]) {
        let channels = self.shape.2;
        let range = self.starts[cell]..self.starts[cell + 1];
        let mut residuals: Vec<Vec<f32>> = (0..channels)
            .map(|_| Vec::with_capacity(range.len() / channels.max(1)))
            .collect();
        let samples = self.at[range.clone()].iter().zip(&self.reference[range]);
        for (sample, (&at, &reference_value)) in samples.enumerate() {
            let channel = sample % channels;
            let Some(&gain) = gains.get(channel) else {
                continue;
            };
            let source_value = values.get(at as usize).copied().unwrap_or(f32::NAN);
            let residual = reference_value - gain * source_value;
            if reference_value.is_finite() && source_value.is_finite() && residual.is_finite() {
                residuals[channel].push(residual);
            }
        }
        for (offset, residuals) in offsets.iter_mut().zip(&mut residuals) {
            *offset = (residuals.len() >= 64)
                .then(|| median_in_place(residuals))
                .flatten();
        }
    }
}

/// [`tile_offset`] for every channel of a tile, into `offsets`: each
/// sampled pixel's channels read together, rather than the tile walked once
/// a channel, and each channel's residuals in the same order.
fn tile_offsets(
    reference: &LinearImage,
    source: Placed<'_>,
    (x, y, width, height): (usize, usize, usize, usize),
    gains: &[f32],
    mask: Option<&BackgroundMask>,
    offsets: &mut [Option<f32>],
) {
    let channels = source.channels();
    let pixels = tile_sample_pixels(source.width, x, y, width, height);
    let mut residuals: Vec<Vec<f32>> = (0..channels)
        .map(|_| Vec::with_capacity(pixels.len()))
        .collect();
    for pixel in pixels {
        if mask.is_some_and(|mask| mask.covers(pixel)) {
            continue;
        }
        for (channel, (residuals, &gain)) in residuals.iter_mut().zip(gains).enumerate() {
            let index = pixel * channels + channel;
            let (reference_value, source_value) = (reference.data[index], source.sample(index));
            let residual = reference_value - gain * source_value;
            if reference_value.is_finite() && source_value.is_finite() && residual.is_finite() {
                residuals.push(residual);
            }
        }
    }
    for (offset, residuals) in offsets.iter_mut().zip(&mut residuals) {
        *offset = (residuals.len() >= 64)
            .then(|| median_in_place(residuals))
            .flatten();
    }
}

/// The median of finite paired residuals. Shared structure cancels before
/// the robust fit, while isolated mismatches cannot dominate the offset.
/// `None` when fewer than 64 usable pairs remain.
fn sample_offset(pairs: impl Iterator<Item = (f32, f32)>, gain: f32) -> Option<f32> {
    let mut residuals = Vec::new();
    for (reference_value, source_value) in pairs {
        let residual = reference_value - gain * source_value;
        if reference_value.is_finite() && source_value.is_finite() && residual.is_finite() {
            residuals.push(residual);
        }
    }
    if residuals.len() < 64 {
        return None;
    }
    median_in_place(&mut residuals)
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

/// Photometry measures each star in a circular aperture whose radius is
/// `PHOTOMETRY_APERTURE` times a FWHM, and its local background in the
/// annulus between `PHOTOMETRY_ANNULUS` times it: the FWHM of the broader of
/// the reference's stars and the frame's, so that both images are measured
/// in the same pixels, which hold nearly all of either's light.
const PHOTOMETRY_APERTURE: f64 = 3.0;
const PHOTOMETRY_ANNULUS: (f64, f64) = (4.0, 5.0);
/// The annulus a star's width is measured over, in FWHMs: closer than the
/// photometry's, since the half of the peak it finds the width at moves
/// little with the sky, and so a smaller stamp, which more stars fit.
const PHOTOMETRY_WIDTH_ANNULUS: (f64, f64) = (2.5, 3.5);
/// How far from each of the other stars photometry measures, in FWHMs, a
/// pixel of a star's annulus must lie to count toward its sky; and the
/// share of the annulus that must be left for the rest to be left out.
const PHOTOMETRY_NEIGHBOUR_MASK: f64 = 2.0;
const PHOTOMETRY_ANNULUS_LEFT: f64 = 0.25;
/// The narrowest FWHM the geometry scales with, in pixels: an aperture 6 px
/// in radius, for stars sampled more coarsely than that.
const PHOTOMETRY_MINIMUM_FWHM: f64 = 2.0;
/// How many times as broad as the reference's a frame's stars may be and
/// still be measured whole. Broader ones are measured at that scale, which
/// is as far as a frame gathered from the GPU reaches.
const PHOTOMETRY_BROADENING: f64 = 3.0;
/// The broadest stars a reference's FWHM is measured for, in pixels, and
/// the broadest a frame's are measured at when the reference's stars give
/// no width.
const PHOTOMETRY_MAXIMUM_FWHM: f64 = 32.0;
/// How far above its sky, in sigmas of the sky's noise, a star's peak must
/// stand for its width to count.
const PHOTOMETRY_WIDTH_SNR: f32 = 10.0;
/// Stars needed for a photometric gain, and for a FWHM.
const PHOTOMETRY_MINIMUM_STARS: usize = 12;
const PHOTOMETRY_WIDTH_STARS: usize = 3;
/// Each smaller scale photometry tries, where too few stars fit the last,
/// is this much smaller.
const PHOTOMETRY_SHRINK: f64 = 1.5;
/// How far about each star photometry measures a background fit leaves
/// its samples out, in the broadest FWHM a frame's photometry scales with.
const BACKGROUND_MASK: f64 = 1.5;

/// What photometric normalization reads of a reference, which a stack
/// measures once for every frame it normalizes against it: the ceilings at
/// once, and its stars' widths the first time a frame is measured.
#[derive(Clone, Debug)]
pub(crate) struct ReferencePhotometry {
    /// Each channel's [`photometric_ceilings`].
    ceilings: Vec<f32>,
    widths: std::sync::OnceLock<ReferenceWidths>,
    mask: std::sync::OnceLock<BackgroundMask>,
}

/// The FWHM of a reference's stars, and how broad a frame's photometry may
/// scale.
#[derive(Clone, Debug)]
struct ReferenceWidths {
    /// Each channel's FWHM at the reference's stars, in pixels, or `None`
    /// where too few measure.
    widths: Vec<Option<f64>>,
    /// The broadest FWHM a frame's photometry scales with:
    /// [`PHOTOMETRY_BROADENING`] times the reference's broadest channel's,
    /// or [`PHOTOMETRY_MAXIMUM_FWHM`] when no channel's is known.
    broadest: f64,
}

impl ReferencePhotometry {
    /// `reference`'s ceilings; its stars' widths are measured when a frame
    /// first needs them, through [`Self::widths`].
    pub(crate) fn measure(reference: &LinearImage) -> Self {
        Self {
            ceilings: photometric_ceilings(reference),
            widths: std::sync::OnceLock::new(),
            mask: std::sync::OnceLock::new(),
        }
    }

    /// The ceilings of `reference` again, a canvas that has grown, keeping
    /// the widths of its stars when they have been measured: frames add
    /// samples where the canvas had none, which change its brightest sample
    /// but not the seeing of the stars already there.
    pub(crate) fn update(&mut self, reference: &LinearImage) {
        self.ceilings = photometric_ceilings(reference);
    }

    /// The widths of `reference`'s stars at `stars`, measured the first
    /// time they are asked for: `reference` and `stars` are those the
    /// photometry was measured for.
    fn widths(&self, reference: &LinearImage, stars: &[(f64, f64)]) -> &ReferenceWidths {
        self.widths.get_or_init(|| {
            let (grid, channels) = ((reference.width, reference.height), reference.channels);
            let widths = (0..channels)
                .map(|channel| {
                    let at =
                        |x: usize, y: usize| reference.data[(y * grid.0 + x) * channels + channel];
                    let ceiling = self.ceilings.get(channel).copied().unwrap_or(f32::MIN);
                    let saturated = |x: usize, y: usize| at(x, y) >= 0.85 * ceiling;
                    image_width(
                        &at,
                        &saturated,
                        grid,
                        stars,
                        PHOTOMETRY_MINIMUM_FWHM,
                        PHOTOMETRY_MAXIMUM_FWHM,
                    )
                })
                .collect::<Vec<_>>();
            let broadest = widths
                .iter()
                .flatten()
                .copied()
                .reduce(f64::max)
                .map_or(PHOTOMETRY_MAXIMUM_FWHM, |width| {
                    PHOTOMETRY_BROADENING * width.max(PHOTOMETRY_MINIMUM_FWHM)
                });
            ReferenceWidths { widths, broadest }
        })
    }

    /// How far from each of `stars`, along either axis, photometry may read
    /// a frame normalized against `reference`, whose photometry this is.
    #[cfg(feature = "gpu")]
    pub(crate) fn reach(&self, reference: &LinearImage, stars: &[(f64, f64)]) -> usize {
        stamp(self.widths(reference, stars).broadest)
    }

    /// The [`BackgroundMask`] about `stars` on `reference`'s grid, made the
    /// first time it is asked for: `reference` and `stars` are those the
    /// photometry was measured for. A canvas that grows keeps it, since it
    /// is its stars' and not its samples'.
    pub(crate) fn mask(&self, reference: &LinearImage, stars: &[(f64, f64)]) -> &BackgroundMask {
        self.mask.get_or_init(|| {
            let radius = BACKGROUND_MASK * self.widths(reference, stars).broadest;
            BackgroundMask::around((reference.width, reference.height), stars, radius)
        })
    }
}

/// The pixels of a grid a local background fit leaves out: those within
/// [`BACKGROUND_MASK`] times the broadest FWHM a frame's photometry scales
/// with of the stars photometry measures, the brightest.
///
/// A tile's offset is the median of its paired residuals, the reference's
/// samples less the frame's after its gain. Seeing spreads a star's light
/// without changing the sky, but not alike in the two: where the reference
/// is the sharper, its cores stand far above the frame's in a few samples,
/// which a median leaves out, and the frame's wings a little above the
/// reference's across many more, which move it by their mean. The offsets
/// of a frame blurred from sigma 1.6 to 3.6 px came out 1.6 low on average
/// on a crowded field, against a sky noise of 4, and 3.1 low in the tile of
/// a bright star: the frame's sky lower than the reference's where stars
/// are, in the stack too. Leaving out the samples about the stars, in both
/// images alike, leaves the median the sky and what little light reaches
/// further. A tile the mask leaves too few samples takes its neighbours'
/// offset, as a tile beyond a frame's edge does.
#[derive(Clone, Debug)]
pub(crate) struct BackgroundMask {
    /// A bit for each pixel of the grid, row by row: whether it is left out.
    bits: Vec<u64>,
}

impl BackgroundMask {
    /// The pixels within `radius` of any of `stars` on a grid of `width` by
    /// `height`.
    fn around((width, height): (usize, usize), stars: &[(f64, f64)], radius: f64) -> Self {
        let mut bits = vec![0_u64; (width * height).div_ceil(64)];
        let within = radius * radius;
        let reach = radius.ceil() as isize;
        for &(x, y) in stars {
            let (cx, cy) = (x.round() as isize, y.round() as isize);
            let rows = (cy - reach).max(0)..=(cy + reach).min(height as isize - 1);
            let columns = (cx - reach).max(0)..=(cx + reach).min(width as isize - 1);
            for py in rows {
                for px in columns.clone() {
                    if (px as f64 - x).powi(2) + (py as f64 - y).powi(2) <= within {
                        let pixel = py as usize * width + px as usize;
                        bits[pixel / 64] |= 1 << (pixel % 64);
                    }
                }
            }
        }
        Self { bits }
    }

    /// Whether `pixel`, counted row by row over the grid, is left out.
    pub(crate) fn covers(&self, pixel: usize) -> bool {
        self.bits
            .get(pixel / 64)
            .is_some_and(|bits| bits >> (pixel % 64) & 1 == 1)
    }
}

/// Per-channel gains that carry `source`'s star fluxes onto `reference`'s:
/// the median ratio of background-subtracted aperture fluxes at `stars`,
/// positions on the reference grid both images share.
///
/// A gain from star photometry follows transparency alone. One from the
/// frames' dispersion also follows their gradients and cloud, which inflate
/// a hazy frame's dispersion; that gain then comes out low, scaling the
/// frame's noise down, so inverse-noise weighting gave cloudy frames the
/// largest weights. Stars near saturation in the reference are skipped, as
/// are stars with a flat top in either image. `None` when fewer than
/// [`PHOTOMETRY_MINIMUM_STARS`] measure cleanly in a channel at any scale.
///
/// Seeing spreads a star's light without taking any away, so the aperture
/// must hold nearly all of it in both images, or the gain follows the
/// seeing too: a fixed 5 px aperture lost more of a blurred frame's stars,
/// and scaled a frame whose Gaussian seeing had grown from sigma 1.6 to
/// 3.6 px up by 1.67. Each channel's aperture is [`PHOTOMETRY_APERTURE`]
/// times the FWHM of the broader of the two images' stars, measured on
/// both: the same pixels in each, so that neighbours and nebulosity in them
/// count alike, and nearly all of a star's light in either, 97% or more
/// even for the broad wings of a Moffat profile of beta 2.5. The gain of
/// that frame is now within 1% of the transparency, and of Moffat stars
/// as blurred within 3.5%. Frames up to [`PHOTOMETRY_BROADENING`] times as
/// blurred as the reference are measured whole. Where too few stars fit
/// that scale, beside the edges or among saturated neighbours, each scale
/// [`PHOTOMETRY_SHRINK`] times smaller is tried in turn, down to the
/// narrowest: a smaller aperture loses some of a blurred frame's light, but
/// a gain from the frames' dispersion would follow its cloud.
///
/// `photometry` is the reference's [`ReferencePhotometry`], which a stack
/// measures once rather than for every frame.
pub(crate) fn photometric_gains(
    reference: &LinearImage,
    photometry: &ReferencePhotometry,
    source: Placed<'_>,
    stars: &[(f64, f64)],
) -> Option<Vec<f32>> {
    photometric_gains_with(
        reference,
        photometry,
        source,
        stars,
        PHOTOMETRY_MINIMUM_STARS,
    )
}

/// [`photometric_gains`] from at least `minimum` cleanly measured stars,
/// with the reference's photometry measured for them.
#[cfg(test)]
pub(crate) fn photometric_gains_from(
    reference: &LinearImage,
    source: Placed<'_>,
    stars: &[(f64, f64)],
    minimum: usize,
) -> Option<Vec<f32>> {
    if !source.matches(reference) {
        return None;
    }
    photometric_gains_with(
        reference,
        &ReferencePhotometry::measure(reference),
        source,
        stars,
        minimum,
    )
}

/// The brightest finite sample of each channel of `reference`, or
/// `f32::MIN` for a channel with none: photometry skips a star whose
/// aperture reaches 85% of it.
fn photometric_ceilings(reference: &LinearImage) -> Vec<f32> {
    let channels = reference.channels;
    (0..channels)
        .map(|channel| {
            reference
                .data
                .par_iter()
                .skip(channel)
                .step_by(channels)
                .copied()
                .filter(|value| value.is_finite())
                .reduce(|| f32::MIN, f32::max)
        })
        .collect()
}

/// [`photometric_gains_from`] given the reference's
/// [`ReferencePhotometry`].
pub(crate) fn photometric_gains_with(
    reference: &LinearImage,
    photometry: &ReferencePhotometry,
    source: Placed<'_>,
    stars: &[(f64, f64)],
    minimum: usize,
) -> Option<Vec<f32>> {
    let channels = reference.channels;
    if !source.matches(reference) || photometry.ceilings.len() != channels {
        return None;
    }
    let ReferenceWidths { widths, broadest } = photometry.widths(reference, stars);
    let grid = (reference.width, reference.height);
    (0..channels)
        .map(|channel| {
            let ceiling = photometry.ceilings[channel];
            let reference_at =
                |x: usize, y: usize| reference.data[(y * grid.0 + x) * channels + channel];
            let source_at = |x: usize, y: usize| source.at(x, y, channel);
            let saturated = |x: usize, y: usize| reference_at(x, y) >= 0.85 * ceiling;
            let reference_width = widths[channel];
            let source_width = image_width(
                &source_at,
                &saturated,
                grid,
                stars,
                reference_width.unwrap_or(PHOTOMETRY_MINIMUM_FWHM),
                *broadest,
            );
            let mut scale = reference_width
                .into_iter()
                .chain(source_width)
                .fold(PHOTOMETRY_MINIMUM_FWHM, f64::max)
                .min(*broadest);
            loop {
                let mut ratios = stars
                    .par_iter()
                    .zip(neighbours(stars, scale, PHOTOMETRY_ANNULUS.1))
                    .filter_map(|(&star, neighbours)| {
                        let star = Star {
                            at: star,
                            neighbours: &neighbours,
                        };
                        star_ratio(&reference_at, &source_at, &saturated, grid, star, scale)
                    })
                    .collect::<Vec<_>>();
                if ratios.len() >= minimum {
                    return median_in_place(&mut ratios)
                        .filter(|gain| gain.is_finite() && *gain > 0.0);
                }
                if scale <= PHOTOMETRY_MINIMUM_FWHM {
                    return None;
                }
                scale = (scale / PHOTOMETRY_SHRINK).max(PHOTOMETRY_MINIMUM_FWHM);
            }
        })
        .collect()
}

/// How far from a star, along either axis, photometry at `scale` reads: to
/// its annulus's outer edge, and a pixel more.
fn stamp(scale: f64) -> usize {
    stamp_for(scale, PHOTOMETRY_ANNULUS.1)
}

/// How far from a star, along either axis, a measurement at `scale` reads
/// whose annulus ends `outer` times it out.
fn stamp_for(scale: f64, outer: f64) -> usize {
    (outer * scale).ceil() as usize + 1
}

/// A star photometry measures, and the others near it.
#[derive(Clone, Copy)]
struct Star<'a> {
    at: (f64, f64),
    neighbours: &'a [(f64, f64)],
}

impl Star<'_> {
    /// The square of pixels within `half` of the pixel nearest the star, as
    /// its first and last columns and rows, or `None` when it leaves a grid
    /// of `width` by `height`.
    fn square(
        &self,
        half: usize,
        (width, height): (usize, usize),
    ) -> Option<(usize, usize, usize, usize)> {
        let (x, y) = self.at;
        let (cx, cy, half) = (x.round() as isize, y.round() as isize, half as isize);
        if cx < half || cy < half || cx + half >= width as isize || cy + half >= height as isize {
            return None;
        }
        let (cx, cy, half) = (cx as usize, cy as usize, half as usize);
        Some((cx - half, cy - half, cx + half, cy + half))
    }

    /// Where each pixel of the star's square at `scale` lies, by its squared
    /// distance from the star: within `inside` of it, in its annulus,
    /// `annulus` times `scale` out, and clear of its neighbours, or in its
    /// annulus beside one. `visit` is given the pixel and its squared
    /// distance, in rows from the top and each row from the left, and stops
    /// the walk with `None`. Only the pixels within the annulus's outer edge
    /// are visited, and none of the gap between `inside` and its inner edge.
    fn walk(
        &self,
        (x0, y0, x1, y1): (usize, usize, usize, usize),
        inside: f64,
        annulus: (f64, f64),
        scale: f64,
        mut visit: impl FnMut(usize, usize, f64, Place) -> Option<()>,
    ) -> Option<()> {
        let (x, y) = self.at;
        let inside = inside * inside;
        let inner = (annulus.0 * scale).powi(2);
        let outer = (annulus.1 * scale).powi(2);
        let clear = (PHOTOMETRY_NEIGHBOUR_MASK * scale).powi(2);
        let reach = inside.max(outer);
        for py in y0..=y1 {
            let dy = py as f64 - y;
            let rise = dy * dy;
            if rise > reach {
                continue;
            }
            // The row's span within reach, and the gap between the inside
            // and the annulus to step over, each a pixel wide of its edge.
            let half = (reach - rise).sqrt();
            let first = (x - half).floor().max(x0 as f64) as usize;
            let last = (x + half).ceil().min(x1 as f64) as usize;
            let gap = (rise < inner).then(|| {
                let within = (inside - rise).max(0.0).sqrt() + 1.0;
                (within, (inner - rise).sqrt() - 1.0)
            });
            for px in first..=last {
                let dx = px as f64 - x;
                if let Some((within, beyond)) = gap
                    && dx.abs() > within
                    && dx.abs() < beyond
                {
                    continue;
                }
                let squared = dx * dx + rise;
                let place = if squared <= inside {
                    Place::Inside
                } else if (inner..=outer).contains(&squared) {
                    let beside = self.neighbours.iter().any(|&(nx, ny)| {
                        let (ex, ey) = (px as f64 - nx, py as f64 - ny);
                        ex * ex + ey * ey <= clear
                    });
                    if beside { Place::Beside } else { Place::Sky }
                } else {
                    continue;
                };
                visit(px, py, squared, place)?;
            }
        }
        Some(())
    }
}

/// Where a pixel of a star's square lies: see [`Star::walk`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum Place {
    Inside,
    Sky,
    Beside,
}

/// Each of `stars`' neighbours among them at `scale`: the others close
/// enough to reach into its annulus, ending `outer` times `scale` out, with
/// the pixels about them that [`PHOTOMETRY_NEIGHBOUR_MASK`] leaves out.
///
/// A median sky stands up to a neighbour's light where that stands far
/// above the noise in a few pixels, as in a sharp frame. A blurred frame
/// spreads the same light thinly across many pixels, under the noise, where
/// it lifts the median by its mean: on a crowded field blurred from sigma
/// 1.6 to 3.6 px that raised the gain by 3.6%. The stars photometry measures
/// are the brightest, whose light that mostly is, so each star's annulus
/// leaves out the pixels about the others, in both images alike.
fn neighbours(stars: &[(f64, f64)], scale: f64, outer: f64) -> Vec<Vec<(f64, f64)>> {
    let reach = (outer + PHOTOMETRY_NEIGHBOUR_MASK) * scale;
    let mut order = (0..stars.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| stars[left].0.total_cmp(&stars[right].0));
    stars
        .par_iter()
        .enumerate()
        .map(|(index, &(x, y))| {
            let first = order.partition_point(|&other| stars[other].0 < x - reach);
            order[first..]
                .iter()
                .take_while(|&&other| stars[other].0 <= x + reach)
                .filter(|&&other| other != index && (stars[other].1 - y).abs() <= reach)
                .map(|&other| stars[other])
                .collect()
        })
        .collect()
}

/// The ratio of `star`'s flux in the reference, which `reference_at`
/// reads, to its flux in the frame `source_at` reads: each the sum over the
/// aperture of [`PHOTOMETRY_APERTURE`] times `scale` in radius, less for
/// each pixel the median of the annulus between [`PHOTOMETRY_ANNULUS`]
/// times it, clear of the star's neighbours. `None` where its square leaves
/// the grid, where either image misses a sample it reads, where
/// `saturated` says a pixel of the aperture is, where either image's star
/// has a flat top, where its neighbours leave too little of the annulus,
/// or where either flux is not positive.
fn star_ratio(
    reference_at: &(impl Fn(usize, usize) -> f32 + Sync),
    source_at: &(impl Fn(usize, usize) -> f32 + Sync),
    saturated: &(impl Fn(usize, usize) -> bool + Sync),
    grid: (usize, usize),
    star: Star<'_>,
    scale: f64,
) -> Option<f32> {
    let square = star.square(stamp(scale), grid)?;
    let mut annulus = Vec::new();
    let mut aperture = Vec::new();
    star.walk(
        square,
        PHOTOMETRY_APERTURE * scale,
        PHOTOMETRY_ANNULUS,
        scale,
        |px, py, _, place| {
            let (r, s) = (reference_at(px, py), source_at(px, py));
            if !r.is_finite() || !s.is_finite() {
                return None;
            }
            match place {
                Place::Inside => aperture.push((px, py, r, s)),
                Place::Sky => annulus.push(((r, s), true)),
                Place::Beside => annulus.push(((r, s), false)),
            }
            Some(())
        },
    )?;
    if aperture.iter().any(|&(px, py, _, _)| saturated(px, py)) {
        return None;
    }
    let (mut reference_annulus, mut source_annulus): (Vec<_>, Vec<_>) =
        sky_samples(&annulus).into_iter().unzip();
    let reference_sky = median_in_place(&mut reference_annulus)?;
    let source_sky = median_in_place(&mut source_annulus)?;
    if flat_topped(aperture.iter().map(|&(_, _, r, _)| r - reference_sky))
        || flat_topped(aperture.iter().map(|&(_, _, _, s)| s - source_sky))
    {
        return None;
    }
    let (reference_flux, source_flux) =
        aperture
            .iter()
            .fold((0.0_f64, 0.0_f64), |(rf, sf), &(_, _, r, s)| {
                (
                    rf + f64::from(r - reference_sky),
                    sf + f64::from(s - source_sky),
                )
            });
    (reference_flux > 0.0 && source_flux > 0.0).then(|| (reference_flux / source_flux) as f32)
}

/// Whether a star whose samples stand `excess` above its sky has the flat
/// top of a clipped one: more of them within 3% of its peak than 13% of
/// those above half of it, and more than four. A Gaussian star keeps 4.4%
/// of its samples above half its peak within 3% of it, and a Moffat star of
/// beta 2.5 3.8%; a star clipped at half its peak keeps half. The
/// reference's ceiling cannot say where a frame clips: one exposed longer
/// or under a clearer sky clips stars the reference does not, whose lost
/// light would raise the gain, and a flat field lifts the stars of a
/// vignetted corner above those clipped at the centre.
fn flat_topped(excess: impl Iterator<Item = f32> + Clone) -> bool {
    let peak = excess.clone().fold(f32::MIN, f32::max);
    if peak <= 0.0 {
        return false;
    }
    let (top, half) = excess.fold((0_usize, 0_usize), |(top, half), value| {
        (
            top + usize::from(value >= 0.97 * peak),
            half + usize::from(value > 0.5 * peak),
        )
    });
    top > 4 && top as f64 > 0.13 * half as f64
}

/// The noise of a star's sky, from the second differences of the samples
/// of `sky`, `(x, y, value)` in rows from the left, that lie three side by
/// side: a slope beneath them, as of a too-small annulus on the star's own
/// flank, cancels from each, where the samples' own spread would count it.
/// `None` with fewer than eight such runs.
fn sky_noise(sky: &[(usize, usize, f32)]) -> Option<f32> {
    let mut differences = sky
        .windows(3)
        .filter(|run| run[0].1 == run[2].1 && run[0].0 + 2 == run[2].0)
        .map(|run| run[0].2 - 2.0 * run[1].2 + run[2].2)
        .collect::<Vec<_>>();
    if differences.len() < 8 {
        return None;
    }
    let centre = median_in_place(&mut differences)?;
    // A second difference of independent samples has six times their
    // variance.
    robust_sigma_in_place(&mut differences, centre).map(|sigma| sigma / 6.0_f32.sqrt())
}

/// The samples of a star's annulus its sky comes from, each marked
/// whether it lies clear of the star's neighbours: those clear of them when
/// they make up [`PHOTOMETRY_ANNULUS_LEFT`] of the annulus or more, and
/// otherwise all of it. Where neighbours crowd the annulus, more than they
/// leave, its sky comes from all of it, as it would without them.
fn sky_samples<T: Copy>(annulus: &[(T, bool)]) -> Vec<T> {
    let left = annulus.iter().filter(|&&(_, clear)| clear).count();
    let clear = left > 0 && left as f64 >= PHOTOMETRY_ANNULUS_LEFT * annulus.len() as f64;
    annulus
        .iter()
        .filter(|&&(_, sky)| sky || !clear)
        .map(|&(sample, _)| sample)
        .collect()
}

/// The median FWHM of an image's stars at `stars`, each as [`star_width`]
/// measures it in the image `at` reads: first at `start`, then at each
/// median found until it holds, at most `cap`, which a broader median is
/// taken for. A star seen at too small a scale is cut short by the window
/// and its sky is lifted by its wings, so a first median at too small a
/// scale is too narrow, and the next ones widen. A wider scale needs a
/// wider stamp, which fewer stars fit: when too few do, the median of
/// those that did is taken, or with none, the last median, which is then
/// the narrowest the stars can be. `None` when fewer than
/// [`PHOTOMETRY_WIDTH_STARS`] measure at `start`.
///
/// Colour calibration measures a FWHM too, from luminance second moments
/// about catalogue stars. Second moments weigh a star's wings, which seeing
/// and optics give every profile in different measure, and the window that
/// bounds them; the aperture here is sized in FWHM, which the half maximum
/// gives from the core alone, in each channel, through the samples of a
/// frame gathered from the GPU.
fn image_width(
    at: &(impl Fn(usize, usize) -> f32 + Sync),
    saturated: &(impl Fn(usize, usize) -> bool + Sync),
    grid: (usize, usize),
    stars: &[(f64, f64)],
    start: f64,
    cap: f64,
) -> Option<f64> {
    let mut scale = start.clamp(PHOTOMETRY_MINIMUM_FWHM, cap);
    let mut found = None;
    for _ in 0..6 {
        let widths = stars
            .par_iter()
            .zip(neighbours(stars, scale, PHOTOMETRY_WIDTH_ANNULUS.1))
            .filter_map(|(&star, neighbours)| {
                let star = Star {
                    at: star,
                    neighbours: &neighbours,
                };
                star_width(at, saturated, grid, star, scale)
            })
            .collect::<Vec<_>>();
        if widths.is_empty() || (found.is_none() && widths.len() < PHOTOMETRY_WIDTH_STARS) {
            break;
        }
        let width = seiza_stats::median_f64(&widths)?.min(cap);
        found = Some(width);
        let next = width.max(PHOTOMETRY_MINIMUM_FWHM);
        if (next - scale).abs() <= 0.02 * scale {
            break;
        }
        scale = next;
    }
    found
}

/// `star`'s FWHM at `scale`, in the image `at` reads: `2 sqrt(n / π)` for
/// the `n` pixels within 1.5 `scale` of it that stand more than half its
/// peak above its sky, the median of its annulus between
/// [`PHOTOMETRY_WIDTH_ANNULUS`] times `scale`, clear of its neighbours as
/// [`star_ratio`]'s is. Counting the pixels above half the peak reads the
/// core alone, where seeing shows, and none of the wings. `None` where its
/// square leaves the grid, where the image misses a sample it reads, where
/// `saturated` says a pixel near it is, where it has a flat top, where its
/// neighbours leave too little of the annulus, or where its peak stands
/// less than [`PHOTOMETRY_WIDTH_SNR`] times its sky's noise above its sky.
fn star_width(
    at: &(impl Fn(usize, usize) -> f32 + Sync),
    saturated: &(impl Fn(usize, usize) -> bool + Sync),
    grid: (usize, usize),
    star: Star<'_>,
    scale: f64,
) -> Option<f64> {
    let square = star.square(stamp_for(scale, PHOTOMETRY_WIDTH_ANNULUS.1), grid)?;
    let mut annulus = Vec::new();
    let mut core = Vec::new();
    star.walk(
        square,
        1.5 * scale,
        PHOTOMETRY_WIDTH_ANNULUS,
        scale,
        |px, py, squared, place| {
            let value = at(px, py);
            if !value.is_finite() {
                return None;
            }
            match place {
                Place::Inside if saturated(px, py) => return None,
                Place::Inside => core.push((squared, value)),
                Place::Sky => annulus.push(((px, py, value), true)),
                Place::Beside => annulus.push(((px, py, value), false)),
            }
            Some(())
        },
    )?;
    let sky = sky_samples(&annulus);
    let noise = sky_noise(&sky)?;
    let level = median_in_place(&mut sky.iter().map(|&(_, _, value)| value).collect::<Vec<_>>())?;
    let peak = core
        .iter()
        .filter(|&&(squared, _)| squared <= 1.5 * 1.5)
        .map(|&(_, value)| value - level)
        .fold(f32::MIN, f32::max);
    if !(peak > 0.0 && peak > PHOTOMETRY_WIDTH_SNR * noise) {
        return None;
    }
    let count = core
        .iter()
        .filter(|&&(_, value)| value - level > 0.5 * peak)
        .count();
    // A star broader than the window fills it above half its peak, and
    // would look flat-topped; its width here is the least it can be, and a
    // wider scale looks again.
    let edge = (1.5 * scale - 1.0).max(0.0).powi(2);
    let truncated = core
        .iter()
        .any(|&(squared, value)| squared > edge && value - level > 0.5 * peak);
    if !truncated && flat_topped(core.iter().map(|&(_, value)| value - level)) {
        return None;
    }
    Some(2.0 * (count as f64 / std::f64::consts::PI).sqrt())
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
    source: Placed<'_>,
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
        let index = (row * source.width + column) * source.channels() + channel;
        let reference_value = reference.data[index];
        let source_value = source.sample(index);
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

    #[cfg(feature = "gpu")]
    #[test]
    fn gathered_samples_preserve_every_preparation_fit() {
        let (width, height, channels) = (960, 720, 3);
        // A lattice of stars, and one near the left edge: a frame as sharp
        // as the reference is measured there in a square smaller than the
        // reach gathered about it, which the edge cuts.
        let stars = (0..20)
            .map(|index| {
                (
                    90.25 + (index % 5) as f64 * 180.0,
                    90.5 + (index / 5) as f64 * 170.0,
                )
            })
            .chain([(20.25, 360.5)])
            .collect::<Vec<_>>();
        let field = |sigma: f64, gain: f64, offset: f64| {
            (0..width * height * channels)
                .map(|index| {
                    let pixel = index / channels;
                    let (x, y) = ((pixel % width) as f64, (pixel / width) as f64);
                    let light = stars
                        .iter()
                        .filter(|&&(sx, sy)| {
                            (x - sx).abs() < 10.0 * sigma && (y - sy).abs() < 10.0 * sigma
                        })
                        .map(|&(sx, sy)| {
                            let r2 = (x - sx).powi(2) + (y - sy).powi(2);
                            3_000.0 / (sigma * sigma) * (-r2 / (2.0 * sigma * sigma)).exp()
                        })
                        .sum::<f64>();
                    let value = gain * (100.0 + 0.03 * x + 0.01 * y + light)
                        + offset
                        + ((index * 97 % 1009) as f64 / 1009.0 - 0.5) * 8.0;
                    if index % 701 == 0 && pixel / width < 3 {
                        f32::NAN
                    } else {
                        value as f32
                    }
                })
                .collect::<Vec<_>>()
        };
        let mut reference =
            LinearImage::new(width, height, channels, field(1.2, 1.0, 0.0)).unwrap();
        reference.data[..channels].fill(50_000.0);
        let reference_photometry = ReferencePhotometry::measure(&reference);
        // One frame as sharp as the reference, and one whose seeing is
        // twice as broad, which photometry measures further out.
        let sources = [(1.2, 1.1, 7.0), (2.4, 0.9, 5.0)].map(|(sigma, gain, offset)| {
            let data = field(sigma, gain, offset);
            (
                1.0 / gain as f32,
                LinearImage::new(width, height, channels, data).unwrap(),
            )
        });
        let background = BackgroundSamples::new(width, height, channels, 256);
        for (transparency, source, mode) in sources.iter().flat_map(|(transparency, source)| {
            [
                NormalizationMode::None,
                NormalizationMode::Global,
                NormalizationMode::Local { tile_size: 64 },
                NormalizationMode::LocalBackground { tile_size: 64 },
                NormalizationMode::LocalBackground { tile_size: 256 },
            ]
            .map(|mode| (*transparency, source, mode))
        }) {
            let indices = preparation_sample_indices(
                (width, height, channels),
                mode,
                &stars,
                reference_photometry.reach(&reference, &stars),
                Some(&background),
            );
            // With tiles large enough to be sampled sparsely, photometry's
            // squares leave most of the frame ungathered, so that a read
            // beyond them would fail.
            if mode == (NormalizationMode::LocalBackground { tile_size: 256 }) {
                assert!(indices.len() < source.data.len() / 2, "{mode:?}");
            }
            let mut lookup = vec![u32::MAX; source.data.len()];
            let values = indices
                .iter()
                .enumerate()
                .map(|(at, &index)| {
                    lookup[index as usize] = at as u32;
                    source.data[index as usize]
                })
                .collect::<Vec<_>>();
            let shape = LinearImage {
                width,
                height,
                channels,
                data: Vec::new(),
            };
            let sparse = Placed {
                sampled: Some(crate::image::SampledPixels {
                    lookup: &lookup,
                    values: &values,
                    tiles: None,
                }),
                ..Placed::whole(&shape)
            };
            let dense = Placed::whole(source);
            let bits = |values: &[f32]| {
                values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                bits(&background.of_placed(sparse).unwrap()),
                bits(&background.of_placed(dense).unwrap())
            );
            let photometry = matches!(mode, NormalizationMode::LocalBackground { .. });
            let gain = photometry
                .then(|| photometric_gains(&reference, &reference_photometry, dense, &stars))
                .flatten();
            let gathered_gain = photometry
                .then(|| photometric_gains(&reference, &reference_photometry, sparse, &stars))
                .flatten();
            assert_eq!(gain, gathered_gain);
            if photometry {
                let gain = gain.as_ref().expect("enough stars");
                assert!(
                    gain.iter()
                        .all(|gain| (gain / transparency - 1.0).abs() < 0.02),
                    "{gain:?} for {transparency}"
                );
            }
            // The samples about the stars the background fit leaves out, as
            // a stack leaves them out.
            let mask = photometry.then(|| reference_photometry.mask(&reference, &stars));
            // The tiles' samples read in order, as a stack reads them.
            let tiles = match mode {
                NormalizationMode::LocalBackground { tile_size } => Some(
                    GatheredTiles::new(&reference, tile_size, &lookup, mask)
                        .expect("every tile sample is gathered"),
                ),
                _ => None,
            };
            let tiled = Placed {
                sampled: sparse.sampled.map(|sampled| crate::image::SampledPixels {
                    tiles: tiles.as_ref(),
                    ..sampled
                }),
                ..sparse
            };
            for gains in [None, gain.as_deref()] {
                let expected =
                    NormalizationMap::estimate_with_gains(&reference, dense, mode, gains, mask)
                        .unwrap();
                for source in [sparse, tiled] {
                    let actual = NormalizationMap::estimate_with_gains(
                        &reference, source, mode, gains, mask,
                    )
                    .unwrap();
                    assert_eq!(bits(&expected.gains), bits(&actual.gains), "{mode:?}");
                    assert_eq!(bits(&expected.offsets), bits(&actual.offsets), "{mode:?}");
                }
            }
        }
    }

    /// A frame the GPU resampled, normalized where it lies, is the CPU's
    /// normalized frame: to the bit for one gain and offset per channel,
    /// fused on both, and within a few units in the last place for a tile
    /// grid, whose interpolation a GPU compiler may fuse where the CPU
    /// rounds each step; on the whole grid and on a region of it.
    #[cfg(feature = "gpu")]
    #[test]
    fn the_gpu_normalizes_a_frame_as_the_cpu_does() {
        if crate::gpu::adapter_description().is_none() {
            eprintln!("no GPU: GPU normalization not compared");
            return;
        }
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        let (grid_width, grid_height, channels) = (1300, 1000, 3);
        let source = LinearImage::new(
            grid_width,
            grid_height,
            channels,
            (0..grid_width * grid_height * channels)
                .map(|index| {
                    if index % 997 == 0 {
                        f32::NAN
                    } else {
                        (200.0 + 50.0 * next()) as f32
                    }
                })
                .collect(),
        )
        .unwrap();
        let tile_size = 128;
        let (columns, rows) = (
            grid_width.div_ceil(tile_size),
            grid_height.div_ceil(tile_size),
        );
        let tiled = NormalizationMap {
            schema_version: NORMALIZATION_MAP_SCHEMA_VERSION,
            width: grid_width,
            height: grid_height,
            channels,
            tile_size,
            columns,
            rows,
            gains: (0..columns * rows * channels)
                .map(|_| (0.9 + 0.2 * next()) as f32)
                .collect(),
            offsets: (0..columns * rows * channels)
                .map(|_| (-20.0 + 40.0 * next()) as f32)
                .collect(),
        };
        let global = NormalizationMap {
            columns: 1,
            rows: 1,
            tile_size: grid_width.max(grid_height),
            gains: vec![1.05, 0.97, 1.11],
            offsets: vec![-3.25, 7.5, 0.125],
            ..tiled.clone()
        };
        let identity = |x: f64, y: f64| (x, y);
        for region in [
            ReferenceRegion {
                x: 0,
                y: 0,
                width: grid_width,
                height: grid_height,
            },
            ReferenceRegion {
                x: 210,
                y: 130,
                width: 900,
                height: 800,
            },
        ] {
            for (map, exact) in [(&global, true), (&tiled, false)] {
                let mut registered = crate::gpu::resample(
                    &source,
                    region,
                    &identity,
                    crate::gpu::GpuSampling::Bilinear,
                    &crate::GpuOptions::default(),
                )
                .expect("the GPU resamples");
                let mut frame = crate::gpu::take_resident(
                    &registered.data,
                    (registered.width, registered.height, registered.channels),
                )
                .expect("the resampled frame stays on the GPU");
                let coefficients = map
                    .gpu_coefficients(
                        (region.x, region.y),
                        (registered.width, registered.height, channels),
                    )
                    .unwrap();
                frame.normalize(&coefficients, false).unwrap();
                map.apply_region(&mut registered, region.x, region.y)
                    .unwrap();
                let normalized = frame.download().unwrap();
                let mut largest = 0.0_f32;
                for (sample, (&gpu, &cpu)) in normalized.iter().zip(&registered.data).enumerate() {
                    if exact || !cpu.is_finite() {
                        assert_eq!(gpu.to_bits(), cpu.to_bits(), "sample {sample}");
                    } else {
                        largest = largest.max((gpu - cpu).abs() / cpu.abs().max(1.0));
                    }
                }
                assert!(largest < 1e-6, "{largest} apart");
            }
        }
    }

    /// The tile's sampled pixels, stepped by row and column, are the ones
    /// dividing each sample's place gives.
    #[test]
    fn stepped_tile_samples_are_the_divided_ones() {
        for (image_width, x, y, width, height) in [
            (6248, 0, 0, 256, 256),
            (6248, 5888, 3840, 104, 79),
            (100, 3, 7, 1, 1),
            (100, 3, 7, 61, 1),
            (100, 0, 0, 100, 100),
            (300, 17, 2, 37, 211),
        ] {
            let stride = (width * height / 4_000).max(1);
            let divided: Vec<usize> = (0..width * height)
                .step_by(stride)
                .map(|sample| (y + sample / width) * image_width + x + sample % width)
                .collect();
            let stepped = tile_sample_pixels(image_width, x, y, width, height);
            assert_eq!(stepped.len(), divided.len());
            assert_eq!(stepped.collect::<Vec<_>>(), divided);
        }
    }

    /// Every channel's tile offsets from one walk of the tile are those a
    /// walk a channel gives, to the bit: mono and RGB, a frame covering the
    /// whole grid or a region of it, with NaN, tiles cut by the edges and
    /// tiles with too few samples.
    #[test]
    fn one_walk_of_a_tile_gives_each_channels_offsets() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        for channels in [1, 3] {
            let (width, height) = (701, 433);
            let mut sample = |nan: f64| {
                (0..width * height * channels)
                    .map(|index| {
                        if next() < nan {
                            f32::NAN
                        } else {
                            (300.0 + 0.01 * (index % width) as f64 + 40.0 * next()) as f32
                        }
                    })
                    .collect::<Vec<_>>()
            };
            let reference = LinearImage::new(width, height, channels, sample(0.01)).unwrap();
            let whole = LinearImage::new(width, height, channels, sample(0.05)).unwrap();
            let region = ReferenceRegion {
                x: 130,
                y: 90,
                width: 300,
                height: 200,
            };
            let part = LinearImage::new(
                region.width,
                region.height,
                channels,
                (0..region.width * region.height * channels)
                    .map(|_| (310.0 + 30.0 * next()) as f32)
                    .collect(),
            )
            .unwrap();
            let globals: Vec<(f32, f32)> = (0..channels)
                .map(|channel| (0.9 + 0.05 * channel as f32, 1.5))
                .collect();
            for (source, tile_size) in [
                (Placed::whole(&whole), 64),
                (Placed::whole(&whole), 256),
                (Placed::on(&part, region, &reference), 64),
            ] {
                let walked =
                    local_background_with_gains(&reference, source, tile_size, &globals, None)
                        .unwrap();
                let by_channel = local_background_from(
                    (source.width, source.height, source.channels()),
                    tile_size,
                    &globals,
                    |_, channel, (x, y, width, height)| {
                        source
                            .reaches(x, y, width, height)
                            .then(|| {
                                tile_offset(
                                    &reference,
                                    source,
                                    channel,
                                    (x, y, width, height),
                                    globals[channel].0,
                                    None,
                                )
                            })
                            .flatten()
                    },
                );
                let bits = |map: &NormalizationMap| {
                    (
                        map.gains.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        map.offsets.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    )
                };
                assert_eq!(
                    bits(&walked),
                    bits(&by_channel),
                    "{channels} channels, {tile_size}"
                );
            }
        }
    }

    #[test]
    fn paired_background_offsets_do_not_follow_shared_nebulae() {
        let (width, height) = (256, 16);
        for gain in [0.6, 1.0, 1.8] {
            for amplitude in [0.0, 50.0, 200.0, 500.0] {
                let sky = |index: usize| {
                    let x = (index % width) as f32 / (width - 1) as f32;
                    100.0 + amplitude * (-0.5 * ((x - 0.25) / 0.15).powi(2)).exp()
                };
                let reference =
                    LinearImage::new(width, height, 1, (0..width * height).map(sky).collect())
                        .unwrap();
                let source = LinearImage::new(
                    width,
                    height,
                    1,
                    (0..width * height)
                        .map(|index| {
                            let gradient = 50.0 * (index % width) as f32 / (width - 1) as f32;
                            (sky(index) + gradient) / gain
                        })
                        .collect(),
                )
                .unwrap();
                let map = NormalizationMap::estimate_with_gains(
                    &reference,
                    Placed::whole(&source),
                    NormalizationMode::LocalBackground { tile_size: width },
                    Some(&[gain]),
                    None,
                )
                .unwrap();
                assert_eq!(map.mean_gain(), gain);
                // The median additive difference is -25 regardless of the
                // shared nebula's shape or brightness. Separate medians
                // produced offsets from -29.4 to -13.3 on this scene.
                assert!(
                    (map.mean_offset() + 25.0).abs() < 0.0002,
                    "gain {gain}, nebula {amplitude}: {}",
                    map.mean_offset()
                );
            }
        }
    }

    #[test]
    fn paired_background_offsets_require_finite_pairs_and_resist_outliers() {
        let source = [100.0; 100];
        let mut reference = [107.0; 100];
        reference[..12].fill(10_000.0);
        reference[12..20].fill(f32::NAN);
        assert_eq!(
            sample_offset(reference.iter().copied().zip(source.iter().copied()), 1.0),
            Some(7.0)
        );
        reference[..37].fill(f32::NAN);
        assert_eq!(
            sample_offset(reference.iter().copied().zip(source.iter().copied()), 1.0),
            None
        );
        assert_eq!(
            sample_offset([(f32::MAX, -f32::MAX); 64].into_iter(), 2.0),
            None
        );
    }

    /// Refitting a frame's background from its samples alone matches the
    /// refit from whole images to the bit, for one and three channels, over
    /// tiles cut short at the edges, with samples missing, a tile with too
    /// few to fit, and negative zeros; and the global map applied to the
    /// samples matches it applied to the image.
    #[test]
    fn background_refits_from_samples_alone_as_from_whole_images() {
        let mut state = 0x2545_f491_u32;
        let mut draw = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state % 10_000) as f32 / 10_000.0
        };
        for (channels, tile_size) in [(1, 16), (3, 32), (3, 40)] {
            let (width, height) = (151, 97);
            let image = |draw: &mut dyn FnMut() -> f32, level: f32| {
                let data = (0..width * height * channels)
                    .map(|index| {
                        let (x, y) = (index / channels % width, index / channels / width);
                        match draw() {
                            // A corner tile left almost empty.
                            _ if x < 9 && y < 9 => f32::NAN,
                            chance if chance < 0.05 => f32::NAN,
                            chance if chance < 0.06 => -0.0,
                            chance => level + 40.0 * chance + x as f32 * 0.2,
                        }
                    })
                    .collect();
                LinearImage::new(width, height, channels, data).unwrap()
            };
            let reference = image(&mut draw, 1000.0);
            let source = image(&mut draw, 700.0);
            let recorded =
                NormalizationMap::estimate(&reference, &source, NormalizationMode::Global).unwrap();
            // Masked about two places, as about stars, or not.
            let mask = BackgroundMask::around(
                (source.width, source.height),
                &[(40.5, 30.25), (100.0, 70.0)],
                18.0,
            );
            for mask in [None, Some(&mask)] {
                let expected = recorded
                    .refit_background(&reference, &source, tile_size, mask)
                    .unwrap();
                let layout = BackgroundSamples::for_image(&source, tile_size);
                let refitted = recorded
                    .refit_background_samples(
                        &layout,
                        &layout.of(&reference).unwrap(),
                        &layout.of(&source).unwrap(),
                        mask,
                    )
                    .unwrap();
                let bits = |values: &[f32]| {
                    values
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                };
                assert_eq!(refitted.offsets.len(), expected.offsets.len());
                assert_eq!(bits(&refitted.offsets), bits(&expected.offsets));
                assert_eq!(bits(&refitted.gains), bits(&expected.gains));
                assert_eq!(refitted, expected);

                let mut whole = source.clone();
                recorded.apply_global(&mut whole).unwrap();
                let mut samples = layout.of(&source).unwrap();
                recorded
                    .apply_global_samples(&layout, &mut samples)
                    .unwrap();
                assert_eq!(bits(&samples), bits(&layout.of(&whole).unwrap()));
            }
        }
    }

    #[test]
    fn spatial_normalization_samples_match_whole_images_on_other_grids() {
        let (width, height) = (519_usize, 391_usize);
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        for channels in [1, 3] {
            let image = LinearImage::new(
                width,
                height,
                channels,
                (0..width * height * channels)
                    .map(|index| match index % 101 {
                        0 => f32::from_bits(0x7fc0_0042),
                        1 => -0.0,
                        2 => 0.0,
                        3 => f32::INFINITY,
                        _ => (index % 997) as f32 / 8.0 - 30.0,
                    })
                    .collect(),
            )
            .unwrap();
            for map_tile in [32, 255, width] {
                let columns = width.div_ceil(map_tile);
                let rows = height.div_ceil(map_tile);
                let map = NormalizationMap {
                    schema_version: NORMALIZATION_MAP_SCHEMA_VERSION,
                    width,
                    height,
                    channels,
                    tile_size: map_tile,
                    columns,
                    rows,
                    gains: (0..columns * rows * channels)
                        .map(|i| 0.5 + (i % 7) as f32 / 8.0)
                        .collect(),
                    offsets: (0..columns * rows * channels)
                        .map(|i| (i % 11) as f32 / 4.0 - 2.0)
                        .collect(),
                };
                let mut whole = image.clone();
                map.apply(&mut whole).unwrap();
                for sample_tile in [16, 32, 256] {
                    let layout = BackgroundSamples::for_image(&image, sample_tile);
                    let mut samples = layout.of(&image).unwrap();
                    map.apply_samples(&layout, &mut samples).unwrap();
                    assert_eq!(
                        bits(&samples),
                        bits(&layout.of(&whole).unwrap()),
                        "map/sample grids {map_tile}/{sample_tile}, channels {channels}"
                    );
                }
                let wrong = BackgroundSamples::new(width + 1, height, channels, 32);
                assert!(
                    map.apply_samples(&wrong, &mut vec![0.0; wrong.len()])
                        .is_err()
                );
                let layout = BackgroundSamples::for_image(&image, 32);
                assert!(
                    map.apply_samples(&layout, &mut vec![0.0; layout.len() - 1])
                        .is_err()
                );
            }
        }
    }

    #[test]
    fn spatial_background_residual_preserves_the_base_and_only_adds_offsets() {
        let (width, height, tile_size) = (512_usize, 384_usize, 255_usize);
        for channels in [1, 3] {
            let columns = width.div_ceil(tile_size);
            let rows = height.div_ceil(tile_size);
            let base = NormalizationMap {
                schema_version: NORMALIZATION_MAP_SCHEMA_VERSION,
                width,
                height,
                channels,
                tile_size,
                columns,
                rows,
                gains: (0..columns * rows * channels)
                    .map(|i| 0.5 + (i % channels) as f32 / 2.0)
                    .collect(),
                offsets: (0..columns * rows * channels)
                    .map(|i| (i / channels) as f32 * 16.0 - 32.0)
                    .collect(),
            };
            let source = LinearImage::new(
                width,
                height,
                channels,
                (0..width * height * channels)
                    .map(|i| {
                        if i % 173 == 0 {
                            f32::NAN
                        } else {
                            512.0 + (i % 113) as f32
                        }
                    })
                    .collect(),
            )
            .unwrap();
            let mut normalized = source.clone();
            base.apply(&mut normalized).unwrap();
            let layout = BackgroundSamples::for_image(&source, 256);
            let samples = layout.of(&source).unwrap();
            for reference in [
                layout.of(&normalized).unwrap(),
                vec![f32::NAN; layout.len()],
            ] {
                let residual = base
                    .refit_background_residual_samples(&layout, &reference, &samples, None)
                    .unwrap();
                assert!(residual.is_identity());
            }
            let mut expected = normalized.clone();
            let correction = [7.25, -3.5, 15.0];
            for (i, value) in expected.data.iter_mut().enumerate() {
                if value.is_finite() {
                    *value += correction[i % channels];
                }
            }
            let saved = base.clone();
            let residual = base
                .refit_background_residual_samples(
                    &layout,
                    &layout.of(&expected).unwrap(),
                    &samples,
                    None,
                )
                .unwrap();
            assert_eq!(base, saved);
            assert!(residual.gains.iter().all(|&gain| gain == 1.0));
            for (i, &offset) in residual.offsets.iter().enumerate() {
                assert_eq!(offset, correction[i % channels]);
            }
            residual.apply(&mut normalized).unwrap();
            assert_eq!(
                normalized
                    .data
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .data
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            let wrong = BackgroundSamples::new(width + 1, height, channels, 256);
            assert!(
                base.refit_background_residual_samples(
                    &wrong,
                    &vec![0.0; wrong.len()],
                    &vec![0.0; wrong.len()],
                    None,
                )
                .is_err()
            );
        }
    }

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
        let gains = photometric_gains(
            &reference,
            &ReferencePhotometry::measure(&reference),
            Placed::whole(&source),
            &stars,
        )
        .expect("enough stars");
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

    /// The row normalizer must give every sample what the per-sample
    /// bilinear interpolation gives it: whole images, crops, and runs of
    /// rows taken in any grouping.
    #[test]
    fn row_normalizer_matches_per_sample_interpolation_bit_for_bit() {
        let (width, height, channels, tile_size): (usize, usize, usize, usize) = (83, 61, 3, 16);
        let (columns, rows) = (width.div_ceil(tile_size), height.div_ceil(tile_size));
        let coefficient =
            |index: usize, scale: f32| ((index * 37 % 101) as f32 / 101.0 - 0.5) * scale;
        let map = NormalizationMap {
            schema_version: NORMALIZATION_MAP_SCHEMA_VERSION,
            width,
            height,
            channels,
            tile_size,
            columns,
            rows,
            gains: (0..columns * rows * channels)
                .map(|index| 1.0 + coefficient(index, 0.3))
                .collect(),
            offsets: (0..columns * rows * channels)
                .map(|index| coefficient(index + 5, 700.0))
                .collect(),
        };
        let source = LinearImage::new(
            width,
            height,
            channels,
            (0..width * height * channels)
                .map(|index| match index % 97 {
                    0 => f32::NAN,
                    1 => f32::INFINITY,
                    _ => 1000.0 + (index * 7919 % 4093) as f32 * 1.37,
                })
                .collect(),
        )
        .unwrap();
        let sampler = map.sampler();
        let expected = source
            .data
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                let (pixel, channel) = (index / channels, index % channels);
                let (gain, offset) = sampler.at(pixel % width, pixel / width, channel);
                if value.is_finite() {
                    value.mul_add(gain, offset)
                } else {
                    value
                }
            })
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        let bits = |data: &[f32]| data.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
        let mut whole = source.clone();
        map.apply(&mut whole).unwrap();
        assert_eq!(bits(&whole.data), expected);
        let row_samples = width * channels;
        for run in [1, 5, 16, 17, height] {
            let normalizer = RowNormalizer::new(&map, 0, width);
            let mut coefficients = RowCoefficients::default();
            let mut data = source.data.clone();
            for (index, rows) in data.chunks_mut(row_samples * run).enumerate() {
                normalizer.apply(&mut coefficients, rows, index * run);
            }
            assert_eq!(bits(&data), expected, "runs of {run} rows");
        }
        let (x, y, crop_width, crop_height) = (13, 9, 40, 30);
        let mut crop = LinearImage::new(
            crop_width,
            crop_height,
            channels,
            (0..crop_height)
                .flat_map(|row| {
                    let start = ((y + row) * width + x) * channels;
                    source.data[start..start + crop_width * channels].to_vec()
                })
                .collect(),
        )
        .unwrap();
        map.apply_region(&mut crop, x, y).unwrap();
        for row in 0..crop_height {
            let start = ((y + row) * width + x) * channels;
            assert_eq!(
                bits(&crop.data[row * crop_width * channels..(row + 1) * crop_width * channels]),
                expected[start..start + crop_width * channels]
            );
        }
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

    /// The profile of a star under seeing, of unit flux.
    #[derive(Clone, Copy, Debug)]
    enum Seeing {
        Gaussian { sigma: f64 },
        Moffat { fwhm: f64, beta: f64 },
    }

    impl Seeing {
        /// The star's light at squared distance `r2`, and how far it is
        /// drawn: past that it holds less than 1e-4 of its flux.
        fn profile(self) -> (f64, impl Fn(f64) -> f64) {
            let (reach, scale, norm, beta) = match self {
                Self::Gaussian { sigma } => (6.0 * sigma, sigma, 0.0, 0.0),
                Self::Moffat { fwhm, beta } => {
                    let alpha = fwhm / (2.0 * (2.0_f64.powf(1.0 / beta) - 1.0).sqrt());
                    let reach = alpha * (1.0e4_f64.powf(1.0 / (beta - 1.0)) - 1.0).sqrt();
                    let norm = (beta - 1.0) / (std::f64::consts::PI * alpha * alpha);
                    (reach, alpha, norm, beta)
                }
            };
            let light = move |r2: f64| match self {
                Self::Gaussian { .. } => {
                    (-r2 / (2.0 * scale * scale)).exp()
                        / (2.0 * std::f64::consts::PI * scale * scale)
                }
                Self::Moffat { .. } => norm * (1.0 + r2 / (scale * scale)).powf(-beta),
            };
            (reach.ceil(), light)
        }
    }

    /// One field of `count` stars under `seeing`, its light, sky included,
    /// scaled by `transparency`, with noise of sigma about 4 that differs
    /// from frame to frame. Each star keeps its whole flux whatever the
    /// seeing.
    fn seeing_field(seeing: Seeing, count: usize, transparency: f32, frame: u32) -> LinearImage {
        let (width, height) = (640_usize, 512_usize);
        let mut state = 0x9e37_79b9_u32;
        let mut random = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            f64::from(state) / f64::from(u32::MAX)
        };
        let stars = (0..count)
            .map(|_| {
                (
                    48.0 + random() * (width as f64 - 96.0),
                    48.0 + random() * (height as f64 - 96.0),
                    3_000.0 + random() * 60_000.0,
                )
            })
            .collect::<Vec<_>>();
        let mut data = vec![200.0_f64; width * height];
        let (reach, light) = seeing.profile();
        for &(x, y, flux) in &stars {
            let (left, top) = ((x - reach).max(0.0) as usize, (y - reach).max(0.0) as usize);
            for py in top..((y + reach) as usize + 1).min(height) {
                for px in left..((x + reach) as usize + 1).min(width) {
                    let (dx, dy) = (px as f64 - x, py as f64 - y);
                    data[py * width + px] += flux * light(dx * dx + dy * dy);
                }
            }
        }
        let mut noise = 0x2545_f491_u32 ^ frame.wrapping_mul(0x85eb_ca6b);
        let data = data
            .into_iter()
            .map(|value| {
                let draw = (0..4)
                    .map(|_| {
                        noise ^= noise << 13;
                        noise ^= noise >> 17;
                        noise ^= noise << 5;
                        f64::from(noise) / f64::from(u32::MAX) - 0.5
                    })
                    .sum::<f64>();
                (value * f64::from(transparency) + 7.0 * draw) as f32
            })
            .collect();
        LinearImage::new(width, height, 1, data).unwrap()
    }

    /// The photometric gain of each frame of [`seeing_field`] of `count`
    /// stars under `seeings`, dimmed to `transparency`, against the first
    /// under `reference` and a clear sky.
    fn seeing_gains<const N: usize>(
        count: usize,
        reference: Seeing,
        seeings: [Seeing; N],
        transparency: f32,
    ) -> [(Seeing, f32); N] {
        let reference = seeing_field(reference, count, 1.0, 0);
        let stars = crate::Registrar::new(&reference, crate::RegistrationOptions::default())
            .unwrap()
            .reference_star_positions();
        let photometry = ReferencePhotometry::measure(&reference);
        let mut frame = 0;
        seeings.map(|seeing| {
            frame += 1;
            let source = seeing_field(seeing, count, transparency, frame);
            let gains = photometric_gains(&reference, &photometry, Placed::whole(&source), &stars);
            (seeing, gains.expect("enough stars")[0])
        })
    }

    /// A night whose Gaussian seeing grows from sigma 1.6 to 3.6 px.
    fn gaussian_night() -> [Seeing; 5] {
        [1.6, 2.1, 2.6, 3.1, 3.6].map(|sigma| Seeing::Gaussian { sigma })
    }

    /// Seeing spreads a star's light without taking any of it away, so a
    /// night whose seeing grows from sigma 1.6 to 3.6 px under a steady sky
    /// needs no gain. A 5 px aperture lost more of each frame's stars than
    /// the last, and scaled them from 1.00 to 1.67. The field's 180 bright
    /// stars crowd each other's annuli as no real field's brightest do.
    #[test]
    fn photometric_gain_does_not_follow_seeing() {
        let night = gaussian_night();
        let gains = seeing_gains(180, night[0], night, 1.0);
        assert!(
            gains.iter().all(|(_, gain)| (gain - 1.0).abs() < 0.02),
            "{gains:?}"
        );
    }

    /// A frame dimmed 20% by haze needs a gain of 1.25, at any seeing; a
    /// 5 px aperture measured 1.25 to 2.09.
    #[test]
    fn photometric_gain_follows_transparency_at_any_seeing() {
        let night = gaussian_night();
        let gains = seeing_gains(180, night[0], night, 0.8);
        assert!(
            gains
                .iter()
                .all(|(_, gain)| (gain / 1.25 - 1.0).abs() < 0.02),
            "{gains:?}"
        );
    }

    /// Seeing's profiles have broader wings than a Gaussian's, which hold
    /// more of a star's light outside any aperture and lay more of it over
    /// the neighbours' annuli: Moffat profiles of beta 2.5, the broadest
    /// wings seeing gives, whose FWHM grows from 3.8 to 8.5 px as the
    /// Gaussian night's does. The aperture leaves 2% more of the most
    /// blurred frame's stars out than of the reference's; on the crowded
    /// field of 180 stars, their neighbours' wings in their annuli add
    /// about as much again.
    #[test]
    fn photometric_gain_does_not_follow_seeing_with_broad_wings() {
        let night = [3.77, 4.94, 6.12, 7.3, 8.48].map(|fwhm| Seeing::Moffat { fwhm, beta: 2.5 });
        for (count, tolerance) in [(80, 0.04), (180, 0.06)] {
            let gains = seeing_gains(count, night[0], night, 1.0);
            assert!(
                gains.iter().all(|(_, gain)| (gain - 1.0).abs() < tolerance),
                "{count} stars: {gains:?}"
            );
        }
    }

    /// A `width` by `height` image of Gaussian stars of `sigma` at
    /// `stars`, the first of them of `flux` and each of the others a tenth
    /// fainter than the last, on a sky of 100 with a little noise that
    /// differs from `frame` to frame.
    fn star_image(
        (width, height): (usize, usize),
        stars: &[(f64, f64)],
        sigma: f64,
        flux: f64,
        frame: u64,
    ) -> LinearImage {
        let reach = 8.0 * sigma;
        let norm = flux / (2.0 * std::f64::consts::PI * sigma * sigma);
        let brightness = |index: usize| norm * 0.9_f64.powi(index as i32);
        let data = (0..width * height)
            .map(|index| {
                let (x, y) = ((index % width) as f64, (index / width) as f64);
                let light = stars
                    .iter()
                    .enumerate()
                    .filter(|&(_, &(sx, sy))| (x - sx).abs() < reach && (y - sy).abs() < reach)
                    .map(|(index, &(sx, sy))| {
                        let r2 = (x - sx).powi(2) + (y - sy).powi(2);
                        brightness(index) * (-r2 / (2.0 * sigma * sigma)).exp()
                    })
                    .sum::<f64>();
                let mut hash = (index as u64 + 1)
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .wrapping_add(frame.wrapping_mul(0xbf58_476d_1ce4_e5b9));
                hash ^= hash >> 29;
                let noise = (hash >> 40) as f64 / (1u64 << 24) as f64 - 0.5;
                (100.0 + light + noise) as f32
            })
            .collect();
        LinearImage::new(width, height, 1, data).unwrap()
    }

    /// Stars 30 px from the edges and from each other fit a sharp frame's
    /// stamps but not those of a frame blurred to 7 px FWHM, which reach 37
    /// px. Photometry measures them at the largest scale that leaves
    /// enough, rather than giving up on the frame, whose gain would then
    /// come from its dispersion.
    #[test]
    fn photometry_shrinks_to_fit_rather_than_giving_up() {
        let stars = (0..16)
            .map(|index| {
                (
                    30.0 + (index % 4) as f64 * 33.0,
                    30.0 + (index / 4) as f64 * 33.0,
                )
            })
            .collect::<Vec<_>>();
        let reference = star_image((160, 160), &stars, 1.27, 20_000.0, 0);
        let source = star_image((160, 160), &stars, 3.0, 20_000.0, 1);
        let gains = photometric_gains(
            &reference,
            &ReferencePhotometry::measure(&reference),
            Placed::whole(&source),
            &stars,
        );
        let gain = gains.expect("photometry at a smaller scale")[0];
        assert!((gain - 1.0).abs() < 0.03, "{gain}");
    }

    /// Stars of 15 px FWHM along a strip 52 px tall: measured first at a
    /// scale of 2, whose window holds only their cores, and then at the
    /// width that gives, whose stamps the strip must still hold. A width
    /// pass that fits no star at the larger scale returned the first,
    /// narrow median: 6 px.
    #[test]
    fn a_width_too_broad_for_most_stamps_is_not_the_narrow_first_guess() {
        let stars = (0..12)
            .map(|index| (40.0 + index as f64 * 80.0, 26.0))
            .collect::<Vec<_>>();
        let image = star_image((1000, 52), &stars, 15.0 / 2.3548, 200_000.0, 0);
        let width = ReferencePhotometry::measure(&image)
            .widths(&image, &stars)
            .widths[0];
        assert!(width.is_some_and(|width| width > 12.0), "{width:?}");
    }

    /// Mosaic panels are matched through the few stars of their overlap.
    /// When too few of them gave the reference a width, every frame was
    /// measured at a scale of 6 px at most, an aperture of about one FWHM
    /// for stars 12 to 18 px across, and the gain followed the seeing.
    #[test]
    fn a_few_stars_still_scale_the_aperture() {
        // The two brightest come within 85% of the reference's brightest
        // sample, as saturated stars do, and are left out.
        let stars = [
            (150.0, 150.0),
            (450.0, 150.0),
            (150.0, 450.0),
            (450.0, 450.0),
            (300.0, 300.0),
        ];
        let reference = star_image((600, 600), &stars, 12.0 / 2.3548, 300_000.0, 0);
        let source = star_image((600, 600), &stars, 18.0 / 2.3548, 300_000.0, 1);
        let gain = photometric_gains_from(&reference, Placed::whole(&source), &stars, 3)
            .expect("enough stars")[0];
        assert!((gain - 1.0).abs() < 0.02, "{gain}");
    }

    /// A frame exposed twice as long as the reference clips its brightest
    /// stars where the reference's stay below its ceiling. Their lost light
    /// raised the gain, half the reference's light, by as much as they
    /// lost, until the frame's flat tops were looked for too.
    #[test]
    fn photometry_leaves_out_stars_the_frame_clips() {
        let seeing = Seeing::Gaussian { sigma: 1.6 };
        let reference = seeing_field(seeing, 180, 1.0, 0);
        let mut source = seeing_field(seeing, 180, 2.0, 1);
        let ceiling = 0.35 * source.data.iter().copied().fold(f32::MIN, f32::max);
        for value in &mut source.data {
            *value = value.min(ceiling);
        }
        let stars = crate::Registrar::new(&reference, crate::RegistrationOptions::default())
            .unwrap()
            .reference_star_positions();
        let gain = photometric_gains(
            &reference,
            &ReferencePhotometry::measure(&reference),
            Placed::whole(&source),
            &stars,
        )
        .expect("enough stars")[0];
        assert!((gain / 0.5 - 1.0).abs() < 0.01, "{gain}");
    }

    /// The mean and the lowest of `map`'s tile offsets for channel 0.
    fn offset_spread(map: &NormalizationMap) -> (f32, f32) {
        let offsets = &map.offsets;
        let mean = offsets.iter().sum::<f32>() / offsets.len() as f32;
        (mean, offsets.iter().copied().fold(f32::MAX, f32::min))
    }

    /// The crowded field, under one sky, taken in seeing of sigma 1.6 px
    /// for the reference and 3.6 px for the frame, with the frame's gain of
    /// 1, the [`BackgroundMask`] about the reference's stars, and the
    /// frame's local background map fitted against the reference in tiles
    /// of 64 px, as a stack fits it.
    fn blurred_background() -> (LinearImage, LinearImage, BackgroundMask, NormalizationMap) {
        let reference = seeing_field(Seeing::Gaussian { sigma: 1.6 }, 180, 1.0, 0);
        let source = seeing_field(Seeing::Gaussian { sigma: 3.6 }, 180, 1.0, 5);
        let stars = crate::Registrar::new(&reference, crate::RegistrationOptions::default())
            .unwrap()
            .reference_star_positions();
        let mask = ReferencePhotometry::measure(&reference)
            .mask(&reference, &stars)
            .clone();
        let map = NormalizationMap::estimate_with_gains(
            &reference,
            Placed::whole(&source),
            NormalizationMode::LocalBackground { tile_size: 64 },
            Some(&[1.0]),
            Some(&mask),
        )
        .unwrap();
        (reference, source, mask, map)
    }

    /// Seeing spreads a star's light into its wings without changing the
    /// sky, so a frame blurred from sigma 1.6 to 3.6 px under the same sky
    /// needs no offset. The median of the paired residuals in a tile reads
    /// the wings, a little above the sky across many pixels, and leaves out
    /// the sharp reference's cores, far above it in a few: on this crowded
    /// field the offsets came out 1.65 low on average, against a sky noise
    /// of 4, and 3.1 low in the tile of a bright star, smoothed over its
    /// neighbours. Without the samples about the stars, 0.12 and 0.64.
    #[test]
    fn background_offsets_do_not_follow_seeing() {
        let (_, _, _, map) = blurred_background();
        let (mean, lowest) = offset_spread(&map);
        assert!(
            mean.abs() < 0.3 && lowest > -1.5,
            "mean {mean}, lowest {lowest}"
        );
    }

    /// A reintegration fits the same offsets again, from the frame's
    /// background samples alone, against an integrated reference.
    #[test]
    fn refitted_background_offsets_do_not_follow_seeing() {
        let (reference, source, mask, map) = blurred_background();
        let layout = BackgroundSamples::for_image(&source, 64);
        let refitted = map
            .refit_background_samples(
                &layout,
                &layout.of(&reference).unwrap(),
                &layout.of(&source).unwrap(),
                Some(&mask),
            )
            .unwrap();
        let (mean, lowest) = offset_spread(&refitted);
        assert!(
            mean.abs() < 0.3 && lowest > -1.5,
            "mean {mean}, lowest {lowest}"
        );
    }
}
