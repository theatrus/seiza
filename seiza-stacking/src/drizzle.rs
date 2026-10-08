//! Drizzle integration: each calibrated source pixel, shrunk to a drop,
//! is carried through its frame's registration onto an output grid that can
//! be finer than the reference, and spread over the output pixels it
//! overlaps in proportion to the overlap.
//!
//! Unlike resampling, a drop never mixes neighbouring source pixels, so
//! dithered frames resolve detail that interpolation smooths away, and at
//! twice the reference scale an undersampled star can be recovered at a
//! finer pixel. A Bayer frame drizzles its photosites: each lands in its own
//! colour only, with no demosaicing at all.
//!
//! The drizzle pass follows a rejecting integration on the reference grid,
//! as PixInsight's DrizzleIntegration follows ImageIntegration: a source
//! pixel whose registered sample the integration rejected is left out, and
//! each pixel is normalized with its frame's map at the registered position.

use crate::batch::{PackedFates, SampleFate};
use crate::image::BayerLayout;
use crate::normalization::CoefficientSampler;
use crate::registration::PolynomialWarp;
use crate::{Error, LinearImage, NormalizationMap, RegisteredFrameMapping, Result};
use rayon::prelude::*;

/// Drizzle output scale and drop size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrizzleOptions {
    /// Output pixels per reference pixel along each axis: 1 keeps the
    /// reference grid, 2 doubles its resolution. At most 4.
    pub scale: u32,
    /// Drop side as a fraction of a source pixel, in `(0, 1]`. Smaller drops
    /// keep more resolution and need more frames, with more dithering, to
    /// cover every output pixel. `None` uses WBPP's defaults: 0.9 for a
    /// monochrome frame and 1.0 for a Bayer frame, whose photosites of one
    /// colour already leave gaps between them.
    pub drop_shrink: Option<f32>,
}

impl Default for DrizzleOptions {
    /// WBPP's defaults: the reference scale, with drops sized by frame type.
    fn default() -> Self {
        Self {
            scale: 1,
            drop_shrink: None,
        }
    }
}

impl DrizzleOptions {
    /// Check the scale and drop size.
    pub fn validate(&self) -> Result<()> {
        if !(1..=4).contains(&self.scale) {
            return Err(Error::Stack("drizzle scale must be 1 to 4".into()));
        }
        if let Some(shrink) = self.drop_shrink
            && !(shrink.is_finite() && shrink > 0.0 && shrink <= 1.0)
        {
            return Err(Error::Stack(
                "drizzle drop shrink must be greater than 0 and at most 1".into(),
            ));
        }
        Ok(())
    }
}

/// A drizzled integration on a grid `scale` times the reference's.
#[derive(Clone, Debug)]
pub struct DrizzleResult {
    /// The weighted mean of every drop over each output pixel; `NaN` where
    /// no drop landed, as can happen along the edges or, with too little
    /// dithering, between a Bayer frame's photosites at a finer scale.
    pub image: LinearImage,
    /// The total weight behind each output sample: drop area in output
    /// pixels times frame weight, summed over frames.
    pub weight: LinearImage,
    /// The output scale relative to the reference grid.
    pub scale: u32,
}

/// Output rows and columns handled together. A tile's source pixels are
/// found from its corners, so tiles keep that search tight even where frames
/// are rotated against the reference.
const TILE: usize = 64;

/// Source pixels between the nodes at which a warped frame's
/// source-to-reference map is solved exactly. In between it is interpolated;
/// a quadratic lens warp bends far too little over this span to matter.
const GRID_STEP: usize = 16;

/// The running sums of a drizzle integration.
pub(crate) struct DrizzleAccumulator {
    options: DrizzleOptions,
    reference_width: usize,
    reference_height: usize,
    channels: usize,
    width: usize,
    height: usize,
    sum: Vec<f32>,
    weight: Vec<f32>,
}

/// One calibrated frame ready to drizzle.
pub(crate) struct DrizzleFrame<'a> {
    /// The calibrated frame before debayering: one channel with `layout` for
    /// a Bayer frame, otherwise one channel per output channel.
    pub(crate) image: &'a LinearImage,
    pub(crate) layout: Option<BayerLayout>,
    pub(crate) mapping: &'a RegisteredFrameMapping,
    pub(crate) normalization: &'a NormalizationMap,
    /// What the integration did with each registered sample of this frame,
    /// on the reference grid.
    pub(crate) fates: &'a PackedFates,
    /// Per-channel frame weight.
    pub(crate) weight: &'a [f32],
}

impl DrizzleAccumulator {
    pub(crate) fn new(
        options: DrizzleOptions,
        reference_width: usize,
        reference_height: usize,
        channels: usize,
    ) -> Result<Self> {
        options.validate()?;
        // As the integration the drizzle follows requires.
        if !matches!(channels, 1 | 3) {
            return Err(Error::Stack("a drizzle needs one or three channels".into()));
        }
        let scale = options.scale as usize;
        let width = reference_width * scale;
        let height = reference_height * scale;
        let samples = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(channels))
            .ok_or_else(|| Error::Stack("drizzle output dimensions overflow".into()))?;
        Ok(Self {
            options,
            reference_width,
            reference_height,
            channels,
            width,
            height,
            sum: vec![0.0; samples],
            weight: vec![0.0; samples],
        })
    }

    /// Drop every unrejected pixel of one frame onto the output grid.
    pub(crate) fn add(&mut self, frame: &DrizzleFrame<'_>) -> Result<()> {
        self.add_frames(std::slice::from_ref(frame))
    }

    /// Drop every unrejected pixel of each frame in turn onto the output
    /// grid, as [`Self::add`] would one frame after another, to the bit.
    ///
    /// Each output tile takes every frame before the next tile, so its sums
    /// stay in the core's cache rather than streaming the whole grid, eight
    /// bytes per output sample, through memory once per frame.
    pub(crate) fn add_frames(&mut self, frames: &[DrizzleFrame<'_>]) -> Result<()> {
        let channels = self.channels;
        let scale = f64::from(self.options.scale);
        let maps = frames
            .iter()
            .map(|frame| {
                let source = frame.image;
                let expected_channels = if frame.layout.is_some() { 1 } else { channels };
                if source.channels != expected_channels
                    || (frame.layout.is_some() && channels != 3)
                    || frame.weight.len() != channels
                    || frame.normalization.channels() != channels
                    || frame.normalization.width() != self.reference_width
                    || frame.normalization.height() != self.reference_height
                    || frame.fates.len() != self.reference_width * self.reference_height * channels
                {
                    return Err(Error::Stack(
                        "drizzle frame does not match the integration's shape".into(),
                    ));
                }
                let shrink = self
                    .options
                    .drop_shrink
                    .unwrap_or(if frame.layout.is_some() { 1.0 } else { 0.9 });
                Ok((
                    ForwardMap::new(
                        frame.mapping,
                        source.width,
                        source.height,
                        f64::from(shrink) * 0.5 * scale,
                    )?,
                    InverseMap::new(frame.mapping),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let contexts = frames
            .iter()
            .zip(&maps)
            .map(|(frame, (map, inverse))| BandContext {
                frame,
                cfa: frame.layout.map(|layout| {
                    [0, 1].map(|row| [0, 1].map(|column| layout.channel_at(column, row)))
                }),
                map,
                inverse,
                normalization: frame.normalization.sampler(),
                scale,
                width: self.width,
                height: self.height,
                channels,
                reference_width: self.reference_width,
                reference_height: self.reference_height,
            })
            .collect::<Vec<_>>();
        if contexts.is_empty() {
            return Ok(());
        }
        let band_samples = self.width * channels * TILE;
        self.sum
            .par_chunks_mut(band_samples)
            .zip(self.weight.par_chunks_mut(band_samples))
            .enumerate()
            .for_each(|(band, (sum, weight))| drizzle_band(&contexts, band * TILE, sum, weight));
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<DrizzleResult> {
        let image = self
            .sum
            .par_iter()
            .zip(self.weight.par_iter())
            .map(|(&sum, &weight)| if weight > 0.0 { sum / weight } else { f32::NAN })
            .collect();
        Ok(DrizzleResult {
            image: LinearImage::new(self.width, self.height, self.channels, image)?,
            weight: LinearImage::new(self.width, self.height, self.channels, self.weight)?,
            scale: self.options.scale,
        })
    }
}

struct BandContext<'a> {
    frame: &'a DrizzleFrame<'a>,
    /// A Bayer frame's channel at each photosite of its repeating 2x2
    /// cell, by row and column.
    cfa: Option<[[usize; 2]; 2]>,
    map: &'a ForwardMap,
    inverse: &'a InverseMap<'a>,
    normalization: CoefficientSampler<'a>,
    scale: f64,
    width: usize,
    height: usize,
    channels: usize,
    reference_width: usize,
    reference_height: usize,
}

/// The most output pixels a drop can touch along one axis. At four times
/// the scale a full drop turned 45 degrees spans about 5.7 output pixels,
/// so this leaves room for frames at a different image scale and for warps.
const SPAN: usize = 16;

/// Drizzle every frame into output rows `top..top + TILE`, held in `sum`
/// and `weight`, one tile of columns at a time, each tile taking the frames
/// in order.
///
/// Built for processors with SSE4.1, and with AVX2 and FMA, too, where the
/// rounding and fused multiply-add the drops need are single instructions
/// rather than calls into the maths library; each gives the same result to
/// the bit.
#[multiversion::multiversion(targets("x86_64+avx2+fma", "x86_64+sse4.1"))]
fn drizzle_band(frames: &[BandContext<'_>], top: usize, sum: &mut [f32], weight: &mut [f32]) {
    let (width, height) = (frames[0].width, frames[0].height);
    let bottom = (top + TILE).min(height);
    let mut left = 0;
    while left < width {
        let right = (left + TILE).min(width);
        for frame in frames {
            frame.tile(top, bottom, left, right, sum, weight);
        }
        left = right;
    }
}

/// One channel a source pixel drops into: its normalized value, and the
/// frame's weight in that channel.
#[derive(Clone, Copy, Default)]
struct ChannelDrop {
    channel: usize,
    value: f32,
    frame_weight: f32,
}

impl BandContext<'_> {
    #[inline(always)]
    fn tile(
        &self,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        sum: &mut [f32],
        weight: &mut [f32],
    ) {
        let source = self.frame.image;
        let Some((x0, y0, x1, y1)) = self.source_bounds(top, bottom, left, right) else {
            return;
        };
        let channels = self.channels;
        let row_samples = self.width * channels;
        let (tile_left, tile_right) = (left as f64, right as f64);
        let (tile_top, tile_bottom) = (top as f64, bottom as f64);
        let mut x_overlaps = [0.0_f64; SPAN];
        let mut y_overlaps = [0.0_f64; SPAN];
        // A parallelogram's overlap with each output pixel it can reach, by
        // row and then column.
        let mut areas = [[0.0_f64; SPAN]; SPAN];
        // The channels a source pixel drops into, at most three.
        let mut drops = [ChannelDrop::default(); 3];
        for y in y0..y1 {
            for x in x0..x1 {
                let (center, shape) = self.map.at(x, y);
                // Output coordinates put pixel k's edges at k and k + 1.
                let output_x = (center.0 + 0.5) * self.scale;
                let output_y = (center.1 + 0.5) * self.scale;
                let (min_x, min_y, max_x, max_y) = shape.bounds(output_x, output_y);
                if max_x <= tile_left
                    || min_x >= tile_right
                    || max_y <= tile_top
                    || min_y >= tile_bottom
                {
                    continue;
                }
                // The registered sample this pixel's centre falls on decides
                // its rejection and normalization.
                let reference_x = center.0.round();
                let reference_y = center.1.round();
                if reference_x < 0.0
                    || reference_y < 0.0
                    || reference_x >= self.reference_width as f64
                    || reference_y >= self.reference_height as f64
                {
                    continue;
                }
                let (reference_x, reference_y) = (reference_x as usize, reference_y as usize);
                let mut count = 0;
                match self.cfa {
                    // A Bayer photosite drops into its own colour only.
                    Some(cfa) => {
                        let channel = cfa[y & 1][x & 1];
                        let value = source.data[y * source.width + x];
                        if let Some(drop) =
                            self.channel_drop(reference_x, reference_y, channel, value)
                        {
                            drops[0] = drop;
                            count = 1;
                        }
                    }
                    None => {
                        let pixel = (y * source.width + x) * channels;
                        for channel in 0..channels {
                            let value = source.data[pixel + channel];
                            if let Some(drop) =
                                self.channel_drop(reference_x, reference_y, channel, value)
                            {
                                drops[count] = drop;
                                count += 1;
                            }
                        }
                    }
                }
                if count == 0 {
                    continue;
                }
                debug_assert!(
                    max_x - min_x < (SPAN - 1) as f64 && max_y - min_y < (SPAN - 1) as f64,
                    "a drop spans more output pixels than the overlap buffers hold"
                );
                let first_x = (min_x.max(tile_left).floor() as usize).max(left);
                let last_x = (max_x.min(tile_right).ceil() as usize)
                    .min(right)
                    .min(first_x + SPAN);
                let first_y = (min_y.max(tile_top).floor() as usize).max(top);
                let last_y = (max_y.min(tile_bottom).ceil() as usize)
                    .min(bottom)
                    .min(first_y + SPAN);
                match shape.rectangle {
                    // A rectangle's overlap with a pixel is the product of
                    // its overlaps along each axis.
                    Some((half_width, half_height)) => {
                        for (overlap, pixel) in x_overlaps.iter_mut().zip(first_x..last_x) {
                            *overlap = interval_overlap(output_x, half_width, pixel as f64);
                        }
                        for (overlap, pixel) in y_overlaps.iter_mut().zip(first_y..last_y) {
                            *overlap = interval_overlap(output_y, half_height, pixel as f64);
                        }
                    }
                    None => parallelogram_overlaps(
                        &shape.corners(output_x, output_y),
                        first_x..last_x,
                        first_y..last_y,
                        &mut areas,
                    ),
                }
                for drop in &drops[..count] {
                    for (row_offset, output_y_index) in (first_y..last_y).enumerate() {
                        let row = (output_y_index - top) * row_samples;
                        for (column_offset, output_x_index) in (first_x..last_x).enumerate() {
                            let area = match shape.rectangle {
                                Some(_) => x_overlaps[column_offset] * y_overlaps[row_offset],
                                None => areas[row_offset][column_offset],
                            };
                            if area <= 0.0 {
                                continue;
                            }
                            let drop_weight = area as f32 * drop.frame_weight;
                            let index = row + output_x_index * channels + drop.channel;
                            sum[index] += drop_weight * drop.value;
                            weight[index] += drop_weight;
                        }
                    }
                }
            }
        }
    }

    /// A source pixel's `value` in `channel`, normalized as the integration
    /// normalized the registered sample at `(x, y)` it falls on, unless it
    /// is not finite or the integration rejected that sample.
    #[inline(always)]
    fn channel_drop(&self, x: usize, y: usize, channel: usize, value: f32) -> Option<ChannelDrop> {
        if !value.is_finite() || self.fate(x, y, channel) == SampleFate::Rejected {
            return None;
        }
        let (gain, offset) = self.normalization.at(x, y, channel);
        Some(ChannelDrop {
            channel,
            value: value.mul_add(gain, offset),
            frame_weight: self.frame.weight[channel],
        })
    }

    /// What the integration did with the registered sample under a source
    /// pixel's centre. Under Bayer drizzle a registered pixel holds only the
    /// colour of its nearest photosite, so where the centre's pixel has no
    /// sample in this colour, the nearest neighbour that has one carries this
    /// photosite's sample.
    #[inline(always)]
    fn fate(&self, x: usize, y: usize, channel: usize) -> SampleFate {
        let at = |x: usize, y: usize| {
            self.frame
                .fates
                .get((y * self.reference_width + x) * self.channels + channel)
        };
        let fate = at(x, y);
        if fate != SampleFate::Missing || self.frame.layout.is_none() {
            return fate;
        }
        let neighbours = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
        ];
        neighbours
            .into_iter()
            .filter_map(|(dx, dy)| {
                let x = x
                    .checked_add_signed(dx)
                    .filter(|&x| x < self.reference_width)?;
                let y = y
                    .checked_add_signed(dy)
                    .filter(|&y| y < self.reference_height)?;
                Some(at(x, y))
            })
            .find(|&fate| fate != SampleFate::Missing)
            .unwrap_or(SampleFate::Missing)
    }

    /// The source pixels whose drops can reach an output tile: the bounding
    /// box of the tile's outline mapped back to the source, with a margin
    /// for the drop size and the warp's curvature between the sampled
    /// points.
    #[inline(always)]
    fn source_bounds(
        &self,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> Option<(usize, usize, usize, usize)> {
        let to_reference = |output: usize| output as f64 / self.scale - 0.5;
        let (reference_left, reference_right) = (to_reference(left), to_reference(right));
        let (reference_top, reference_bottom) = (to_reference(top), to_reference(bottom));
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for step_y in 0..=2 {
            for step_x in 0..=2 {
                let x =
                    reference_left + (reference_right - reference_left) * f64::from(step_x) / 2.0;
                let y =
                    reference_top + (reference_bottom - reference_top) * f64::from(step_y) / 2.0;
                let (source_x, source_y) = self.inverse.apply(x, y);
                min_x = min_x.min(source_x);
                min_y = min_y.min(source_y);
                max_x = max_x.max(source_x);
                max_y = max_y.max(source_y);
            }
        }
        const MARGIN: f64 = 2.0;
        let source = self.frame.image;
        let x0 = (min_x - MARGIN).floor().max(0.0);
        let y0 = (min_y - MARGIN).floor().max(0.0);
        let x1 = (max_x + MARGIN).ceil().min(source.width as f64);
        let y1 = (max_y + MARGIN).ceil().min(source.height as f64);
        (x0 < x1 && y0 < y1).then_some((x0 as usize, y0 as usize, x1 as usize, y1 as usize))
    }
}

/// The length of `[center - half, center + half]` inside `[pixel, pixel + 1]`.
#[inline(always)]
fn interval_overlap(center: f64, half: f64, pixel: f64) -> f64 {
    ((center + half).min(pixel + 1.0) - (center - half).max(pixel)).max(0.0)
}

/// A drop's shape on the output grid, about its centre: a parallelogram
/// given by its half-diagonals, or an axis-aligned rectangle when the frame
/// is (nearly) square to the reference.
#[derive(Clone, Copy, Debug)]
struct DropShape {
    /// Half the drop's edge vectors: the Jacobian's columns times half the
    /// drop side.
    u: (f64, f64),
    v: (f64, f64),
    /// Half width and half height, when the drop is treated as a rectangle.
    rectangle: Option<(f64, f64)>,
    /// Half the bounding box's width and height.
    extent: (f64, f64),
}

impl DropShape {
    /// The drop of a source pixel whose reference Jacobian is
    /// `[dx/dsx, dx/dsy, dy/dsx, dy/dsy]`, `half` output pixels from centre
    /// to edge along each source axis.
    fn new(jacobian: [f64; 4], half: f64) -> Self {
        let [a, b, c, d] = jacobian;
        let u = (a * half, c * half);
        let v = (b * half, d * half);
        // A rotation under about half a degree, or that much off a quarter
        // turn, moves a drop's corners by under 1% of its side: treat it as
        // the rectangle of the same centre and area.
        let straight = b.abs().max(c.abs()) <= 0.01 * a.abs().min(d.abs());
        let quarter = a.abs().max(d.abs()) <= 0.01 * b.abs().min(c.abs());
        let rectangle = (straight || quarter).then(|| {
            let area = (a * d - b * c).abs() * 4.0 * half * half;
            let (half_width, half_height) = if straight {
                (a.abs() * half, d.abs() * half)
            } else {
                (b.abs() * half, c.abs() * half)
            };
            // Keep the parallelogram's area exactly.
            let correction = (area / (4.0 * half_width * half_height)).sqrt();
            (half_width * correction, half_height * correction)
        });
        let extent = rectangle.unwrap_or((u.0.abs() + v.0.abs(), u.1.abs() + v.1.abs()));
        Self {
            u,
            v,
            rectangle,
            extent,
        }
    }

    #[inline(always)]
    fn bounds(&self, x: f64, y: f64) -> (f64, f64, f64, f64) {
        let (half_width, half_height) = self.extent;
        (
            x - half_width,
            y - half_height,
            x + half_width,
            y + half_height,
        )
    }

    #[inline(always)]
    fn corners(&self, x: f64, y: f64) -> [(f64, f64); 4] {
        let (u, v) = (self.u, self.v);
        [
            (x - u.0 - v.0, y - u.1 - v.1),
            (x + u.0 - v.0, y + u.1 - v.1),
            (x + u.0 + v.0, y + u.1 + v.1),
            (x - u.0 + v.0, y - u.1 + v.1),
        ]
    }

    /// The drop's area inside output pixel `[x, x + 1) × [y, y + 1)` when
    /// centred at `center`.
    #[cfg(test)]
    fn overlap(&self, center: (f64, f64), x: f64, y: f64) -> f64 {
        match self.rectangle {
            Some((half_width, half_height)) => {
                interval_overlap(center.0, half_width, x)
                    * interval_overlap(center.1, half_height, y)
            }
            None => clipped_area(&self.corners(center.0, center.1), x, y),
        }
    }
}

/// The area of a convex quadrilateral inside each unit square at
/// `(x, y)` for `x` in `columns` and `y` in `rows`, into
/// `areas[y - rows.start][x - columns.start]`, by clipping it against each
/// square's four edges: left, right, top and bottom, in that order.
///
/// The left and right edges are shared by a column of squares, so the
/// quadrilateral is clipped to each column once, then to each square of it.
/// Every square sees the same arithmetic as a clip against its own four
/// edges in turn.
#[inline(always)]
fn parallelogram_overlaps(
    corners: &[(f64, f64); 4],
    columns: std::ops::Range<usize>,
    rows: std::ops::Range<usize>,
    areas: &mut [[f64; SPAN]; SPAN],
) {
    let corners = Polygon::of(corners);
    let mut left = Polygon::default();
    let mut column = Polygon::default();
    let mut top = Polygon::default();
    let mut square = Polygon::default();
    for (column_offset, x) in columns.enumerate() {
        let x = x as f64;
        let kept = corners.clip::<false, false>(x, &mut left)
            && left.clip::<false, true>(x + 1.0, &mut column);
        for (row_offset, y) in rows.clone().enumerate() {
            areas[row_offset][column_offset] = if kept {
                column.square_area(y as f64, &mut top, &mut square)
            } else {
                0.0
            };
        }
    }
}

/// Room for a clipped polygon: a quadrilateral clipped to a square has at
/// most eight vertices. A power of two, so that masking an index keeps it
/// in bounds.
const CLIPPED: usize = 16;

/// The vertices of a convex polygon, coordinates apart.
#[derive(Clone, Copy)]
struct Polygon {
    x: [f64; CLIPPED],
    y: [f64; CLIPPED],
    count: usize,
}

impl Default for Polygon {
    fn default() -> Self {
        Self {
            x: [0.0; CLIPPED],
            y: [0.0; CLIPPED],
            count: 0,
        }
    }
}

impl Polygon {
    fn of(corners: &[(f64, f64); 4]) -> Self {
        let mut polygon = Self::default();
        for (index, &(x, y)) in corners.iter().enumerate() {
            polygon.x[index] = x;
            polygon.y[index] = y;
        }
        polygon.count = 4;
        polygon
    }

    /// This polygon, already clipped to a column of unit squares, clipped
    /// to the square at row `y`, top edge first, and its area there.
    #[inline(always)]
    fn square_area(&self, y: f64, top: &mut Polygon, square: &mut Polygon) -> f64 {
        if self.clip::<true, false>(y, top) && top.clip::<true, true>(y + 1.0, square) {
            square.area()
        } else {
            0.0
        }
    }

    /// Keep the part of this convex polygon on one side of an axis-aligned
    /// line in `clipped`, reporting whether three vertices or more are left:
    /// the side of `x = at`, or of `y = at` when `ROWS`, below `at` when
    /// `UPPER` and above it otherwise.
    ///
    /// Sutherland and Hodgman's way, one vertex after another: each adds
    /// the cut on its way in from the one before, if it crossed the line,
    /// and then itself, if it is inside.
    #[inline(always)]
    fn clip<const ROWS: bool, const UPPER: bool>(&self, at: f64, clipped: &mut Polygon) -> bool {
        let count = self.count;
        debug_assert!((3..=8).contains(&count));
        let inside = |x: f64, y: f64| {
            let coordinate = if ROWS { y } else { x };
            if UPPER {
                coordinate <= at
            } else {
                coordinate >= at
            }
        };
        let mut next = 0;
        let last = (count - 1) % CLIPPED;
        let (mut previous_x, mut previous_y) = (self.x[last], self.y[last]);
        let mut previous_inside = inside(previous_x, previous_y);
        for index in 0..count {
            let (x, y) = (self.x[index % CLIPPED], self.y[index % CLIPPED]);
            let current_inside = inside(x, y);
            if previous_inside != current_inside {
                let fraction = if ROWS {
                    (at - previous_y) / (y - previous_y)
                } else {
                    (at - previous_x) / (x - previous_x)
                };
                clipped.x[next % CLIPPED] = previous_x + (x - previous_x) * fraction;
                clipped.y[next % CLIPPED] = previous_y + (y - previous_y) * fraction;
                next += 1;
            }
            if current_inside {
                clipped.x[next % CLIPPED] = x;
                clipped.y[next % CLIPPED] = y;
                next += 1;
            }
            (previous_x, previous_y, previous_inside) = (x, y, current_inside);
        }
        // A convex polygon gains at most a vertex a clip, so a clipped
        // quadrilateral keeps at most eight.
        assert!(
            next <= 8,
            "a clipped quadrilateral has at most eight vertices"
        );
        clipped.count = next;
        next >= 3
    }

    /// The area, by the shoelace formula.
    #[inline(always)]
    fn area(&self) -> f64 {
        let (x, y) = (&self.x, &self.y);
        let last = (self.count - 1) % CLIPPED;
        let mut twice_area = 0.0;
        for index in 0..last {
            twice_area += x[index] * y[index + 1] - x[index + 1] * y[index];
        }
        twice_area += x[last] * y[0] - x[0] * y[last];
        twice_area.abs() * 0.5
    }
}

/// The area of a convex quadrilateral inside the unit square at `(x, y)`,
/// clipped as [`parallelogram_overlaps`] clips it.
#[cfg(test)]
fn clipped_area(corners: &[(f64, f64); 4], x: f64, y: f64) -> f64 {
    let corners = Polygon::of(corners);
    let (mut left, mut column) = (Polygon::default(), Polygon::default());
    if corners.clip::<false, false>(x, &mut left) && left.clip::<false, true>(x + 1.0, &mut column)
    {
        column.square_area(y, &mut Polygon::default(), &mut Polygon::default())
    } else {
        0.0
    }
}

/// Reference positions of source pixels, and the drop shape each takes.
enum ForwardMap {
    /// A similarity's rotation and scale as the cosine and sine terms
    /// [`crate::SimilarityTransform::apply`] works out for every point,
    /// worked out once.
    Similarity {
        cosine: f64,
        sine: f64,
        translation: (f64, f64),
        shape: DropShape,
    },
    /// A warped frame's map, solved at grid nodes and interpolated between,
    /// with one drop shape per grid cell from the cell's mean Jacobian.
    Grid {
        columns: usize,
        nodes: Vec<(f64, f64)>,
        shapes: Vec<DropShape>,
    },
}

impl ForwardMap {
    /// `half` is half the drop side in output pixels.
    fn new(
        mapping: &RegisteredFrameMapping,
        width: usize,
        height: usize,
        half: f64,
    ) -> Result<Self> {
        let transform = mapping.transform();
        transform.validate()?;
        let Some(warp) = mapping.warp() else {
            let cosine = transform.rotation_radians.cos() * transform.scale;
            let sine = transform.rotation_radians.sin() * transform.scale;
            return Ok(Self::Similarity {
                cosine,
                sine,
                translation: (transform.translation_x, transform.translation_y),
                shape: DropShape::new([cosine, -sine, sine, cosine], half),
            });
        };
        let columns = width.div_ceil(GRID_STEP) + 1;
        let rows = height.div_ceil(GRID_STEP) + 1;
        let nodes = (0..rows * columns)
            .into_par_iter()
            .map(|node| {
                let source_x = ((node % columns) * GRID_STEP) as f64;
                let source_y = ((node / columns) * GRID_STEP) as f64;
                invert_warp(warp, transform, source_x, source_y)
            })
            .collect::<Vec<_>>();
        let step = GRID_STEP as f64;
        let shapes = (0..rows * columns)
            .into_par_iter()
            .map(|cell| {
                let (column, row) = (cell % columns, cell / columns);
                if column + 1 >= columns || row + 1 >= rows {
                    return DropShape::new([1.0, 0.0, 0.0, 1.0], half);
                }
                let node = |c: usize, r: usize| nodes[r * columns + c];
                let (p00, p10) = (node(column, row), node(column + 1, row));
                let (p01, p11) = (node(column, row + 1), node(column + 1, row + 1));
                let along_x = (
                    (p10.0 - p00.0 + p11.0 - p01.0) / (2.0 * step),
                    (p10.1 - p00.1 + p11.1 - p01.1) / (2.0 * step),
                );
                let along_y = (
                    (p01.0 - p00.0 + p11.0 - p10.0) / (2.0 * step),
                    (p01.1 - p00.1 + p11.1 - p10.1) / (2.0 * step),
                );
                DropShape::new([along_x.0, along_y.0, along_x.1, along_y.1], half)
            })
            .collect();
        Ok(Self::Grid {
            columns,
            nodes,
            shapes,
        })
    }

    #[inline(always)]
    fn at(&self, x: usize, y: usize) -> ((f64, f64), &DropShape) {
        match self {
            Self::Similarity {
                cosine,
                sine,
                translation,
                shape,
            } => {
                let (x, y) = (x as f64, y as f64);
                (
                    (
                        cosine * x - sine * y + translation.0,
                        sine * x + cosine * y + translation.1,
                    ),
                    shape,
                )
            }
            Self::Grid {
                columns,
                nodes,
                shapes,
            } => {
                let (column, row) = (x / GRID_STEP, y / GRID_STEP);
                let fx = (x % GRID_STEP) as f64 / GRID_STEP as f64;
                let fy = (y % GRID_STEP) as f64 / GRID_STEP as f64;
                let cell = row * columns + column;
                let (p00, p10) = (nodes[cell], nodes[cell + 1]);
                let (p01, p11) = (nodes[cell + columns], nodes[cell + columns + 1]);
                let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
                let top = (lerp(p00.0, p10.0, fx), lerp(p00.1, p10.1, fx));
                let bottom = (lerp(p01.0, p11.0, fx), lerp(p01.1, p11.1, fx));
                (
                    (lerp(top.0, bottom.0, fy), lerp(top.1, bottom.1, fy)),
                    &shapes[cell],
                )
            }
        }
    }
}

/// The reference position a warp sends to `(x, y)` in the source, by
/// Newton's method from the frame's similarity estimate.
fn invert_warp(
    warp: &PolynomialWarp,
    transform: crate::SimilarityTransform,
    x: f64,
    y: f64,
) -> (f64, f64) {
    let mut point = transform.apply(x, y);
    for _ in 0..8 {
        let (source_x, source_y) = warp.apply(point.0, point.1);
        let (error_x, error_y) = (source_x - x, source_y - y);
        if error_x.abs() < 1.0e-6 && error_y.abs() < 1.0e-6 {
            break;
        }
        let (right_x, right_y) = warp.apply(point.0 + 1.0, point.1);
        let (down_x, down_y) = warp.apply(point.0, point.1 + 1.0);
        let (a, b) = (right_x - source_x, down_x - source_x);
        let (c, d) = (right_y - source_y, down_y - source_y);
        let determinant = a * d - b * c;
        if determinant.abs() < 1.0e-12 {
            break;
        }
        point.0 -= (d * error_x - b * error_y) / determinant;
        point.1 -= (a * error_y - c * error_x) / determinant;
    }
    point
}

/// Reference-to-source positions, the direction registration reads.
enum InverseMap<'a> {
    Similarity(crate::SimilarityTransform),
    Warp(&'a PolynomialWarp),
}

impl<'a> InverseMap<'a> {
    fn new(mapping: &'a RegisteredFrameMapping) -> Self {
        match mapping.warp() {
            Some(warp) => Self::Warp(warp),
            None => Self::Similarity(mapping.transform()),
        }
    }

    #[inline(always)]
    fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        match self {
            Self::Similarity(transform) => transform.inverse_apply(x, y),
            Self::Warp(warp) => warp.apply(x, y),
        }
    }
}

#[cfg(test)]
mod reference;

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic stream of numbers in `[0, 1)`.
    struct Noise(u64);

    impl Noise {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1_u64 << 53) as f64
        }
    }

    /// A local normalization map of random gains and offsets.
    fn local_map(
        width: usize,
        height: usize,
        channels: usize,
        tile_size: usize,
        noise: &mut Noise,
    ) -> NormalizationMap {
        let (columns, rows) = (width.div_ceil(tile_size), height.div_ceil(tile_size));
        let cells = columns * rows * channels;
        let gains = (0..cells)
            .map(|_| 0.8 + 0.4 * noise.next() as f32)
            .collect::<Vec<_>>();
        let offsets = (0..cells)
            .map(|_| 20.0 * noise.next() as f32 - 10.0)
            .collect::<Vec<_>>();
        serde_json::from_value(serde_json::json!({
            "schema_version": 1,
            "width": width,
            "height": height,
            "channels": channels,
            "tile_size": tile_size,
            "columns": columns,
            "rows": rows,
            "gains": gains,
            "offsets": offsets,
        }))
        .unwrap()
    }

    /// One frame's inputs to a drizzle.
    struct Inputs {
        image: LinearImage,
        layout: Option<BayerLayout>,
        mapping: RegisteredFrameMapping,
        normalization: NormalizationMap,
        fates: PackedFates,
        weight: Vec<f32>,
    }

    impl Inputs {
        fn frame(&self) -> DrizzleFrame<'_> {
            DrizzleFrame {
                image: &self.image,
                layout: self.layout,
                mapping: &self.mapping,
                normalization: &self.normalization,
                fates: &self.fates,
                weight: &self.weight,
            }
        }
    }

    /// A frame of `width` by `height` source pixels, the reference's size,
    /// carried onto it by `transform` and, when `warped`, a radial lens
    /// term, with some samples not finite and a quarter of them rejected.
    #[allow(clippy::too_many_arguments)]
    fn inputs(
        width: usize,
        height: usize,
        channels: usize,
        bayer: bool,
        transform: crate::SimilarityTransform,
        warped: bool,
        tile_size: usize,
        noise: &mut Noise,
    ) -> Inputs {
        let source_channels = if bayer { 1 } else { channels };
        let data = (0..width * height * source_channels)
            .map(|_| match noise.next() {
                draw if draw < 0.02 => f32::NAN,
                draw if draw < 0.025 => f32::INFINITY,
                draw if draw < 0.03 => -0.0,
                _ => 100.0 + 900.0 * noise.next() as f32,
            })
            .collect();
        let image = LinearImage::new(width, height, source_channels, data).unwrap();
        let layout = bayer.then_some(BayerLayout {
            pattern: seiza_fits::BayerPattern::Gbrg,
            x_offset: 1,
            y_offset: 0,
        });
        let identity = NormalizationMap::identity(
            &LinearImage::new(
                width,
                height,
                channels,
                vec![0.0; width * height * channels],
            )
            .unwrap(),
        );
        let mut mapping = RegisteredFrameMapping::new(width, height, transform, identity).unwrap();
        if warped {
            let (center_x, center_y) = (width as f64 / 2.0, height as f64 / 2.0);
            // Three pixels at the corners.
            let strength = 3.0 / center_x.hypot(center_y).powi(3);
            let pairs = (0..=12)
                .flat_map(|row| {
                    (0..=12).map(move |column| {
                        (
                            column as f64 * width as f64 / 12.0,
                            row as f64 * height as f64 / 12.0,
                        )
                    })
                })
                .map(|(x, y)| {
                    let (source_x, source_y) = transform.inverse_apply(x, y);
                    let radius = ((x - center_x).powi(2) + (y - center_y).powi(2)) * strength;
                    (
                        (x, y),
                        (
                            source_x + (x - center_x) * radius,
                            source_y + (y - center_y) * radius,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            mapping
                .set_warp(Some(PolynomialWarp::fit(2, width, height, &pairs).unwrap()))
                .unwrap();
        }
        let fates = (0..width * height * channels)
            .map(|_| match noise.next() {
                draw if draw < 0.1 => SampleFate::Missing,
                draw if draw < 0.35 => SampleFate::Rejected,
                _ => SampleFate::Integrated,
            })
            .collect::<Vec<_>>();
        Inputs {
            image,
            layout,
            mapping,
            normalization: local_map(width, height, channels, tile_size, noise),
            fates: PackedFates::from_fates(&fates),
            weight: (0..channels).map(|_| 0.5 + noise.next() as f32).collect(),
        }
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    /// Drizzle `frames` one at a time with the reference, one at a time
    /// with [`DrizzleAccumulator::add`], and all at once, and check the
    /// three agree to the bit.
    fn assert_matches_reference(
        options: DrizzleOptions,
        width: usize,
        height: usize,
        channels: usize,
        frames: &[Inputs],
    ) {
        let accumulator = || DrizzleAccumulator::new(options, width, height, channels).unwrap();
        let mut expected = accumulator();
        for frame in frames {
            reference::add(&mut expected, &frame.frame()).unwrap();
        }
        let mut each = accumulator();
        for frame in frames {
            each.add(&frame.frame()).unwrap();
        }
        let mut together = accumulator();
        together
            .add_frames(&frames.iter().map(Inputs::frame).collect::<Vec<_>>())
            .unwrap();
        assert!(expected.weight.iter().any(|&weight| weight > 0.0));
        for actual in [&each, &together] {
            assert!(
                bits(&actual.sum) == bits(&expected.sum),
                "{options:?}: sums differ"
            );
            assert!(
                bits(&actual.weight) == bits(&expected.weight),
                "{options:?}: weights differ"
            );
        }
        let (expected, together) = (expected.finish().unwrap(), together.finish().unwrap());
        assert!(bits(&together.image.data) == bits(&expected.image.data));
    }

    fn turned(
        degrees: f64,
        scale: f64,
        translation_x: f64,
        translation_y: f64,
    ) -> crate::SimilarityTransform {
        crate::SimilarityTransform {
            scale,
            rotation_radians: degrees.to_radians(),
            translation_x,
            translation_y,
        }
    }

    #[test]
    fn bayer_frames_drizzle_as_the_reference_does() {
        let (width, height) = (150, 110);
        let mut noise = Noise(0x2545_f491_4f6c_dd1d);
        // Nearly square to the reference, so drops are rectangles; then
        // turned by a meridian flip and a little more, so drops are
        // parallelograms; then a quarter turn. Some frames are warped, and
        // every frame hangs over an edge.
        let frames = [
            (turned(0.05, 1.0, 3.7, -2.2), false),
            (turned(-0.2, 1.0, -6.1, 4.4), true),
            (turned(179.2, 1.0, 152.3, 105.6), true),
            (turned(180.8, 1.0, 145.0, 111.9), false),
            (turned(90.3, 1.0, 130.0, -12.0), true),
        ]
        .map(|(transform, warped)| {
            inputs(width, height, 3, true, transform, warped, 32, &mut noise)
        });
        for scale in 1..=4 {
            for drop_shrink in [None, Some(0.55)] {
                let options = DrizzleOptions { scale, drop_shrink };
                assert_matches_reference(options, width, height, 3, &frames);
            }
        }
    }

    #[test]
    fn colour_and_monochrome_frames_drizzle_as_the_reference_does() {
        let (width, height) = (131, 97);
        let mut noise = Noise(0x9e37_79b9_7f4a_7c15);
        for channels in [1, 3] {
            let frames = [
                (turned(0.0, 1.0, 0.0, 0.0), false),
                (turned(23.0, 1.0, 30.0, -20.0), false),
                (turned(-37.0, 1.04, -15.0, 60.0), true),
                (turned(179.4, 0.97, 128.0, 99.0), true),
                (turned(-90.0, 1.0, 1.5, 130.5), false),
            ]
            .map(|(transform, warped)| {
                inputs(
                    width, height, channels, false, transform, warped, 16, &mut noise,
                )
            });
            for scale in [1, 2, 3] {
                for drop_shrink in [None, Some(1.0), Some(0.3)] {
                    let options = DrizzleOptions { scale, drop_shrink };
                    assert_matches_reference(options, width, height, channels, &frames);
                }
            }
        }
    }

    /// Drizzle a full-size frame of a 26-megapixel colour sensor at twice
    /// the scale with the reference and the reworked accumulation, and
    /// report how long each takes. Run with
    /// `cargo test --release -p seiza-stacking drizzle_timing -- --ignored --nocapture`.
    #[test]
    #[ignore = "a timing run on a full-size frame"]
    fn drizzle_timing() {
        let (width, height) = (6248, 4176);
        let mut noise = Noise(0x1234_5678_9abc_def1);
        let frames = [
            (turned(0.05, 1.0, 3.7, -2.2), true),
            (turned(179.2, 1.0, 6250.3, 4180.6), true),
        ]
        .map(|(transform, warped)| {
            inputs(width, height, 3, true, transform, warped, 256, &mut noise)
        });
        let options = DrizzleOptions {
            scale: 2,
            drop_shrink: None,
        };
        let accumulator = || DrizzleAccumulator::new(options, width, height, 3).unwrap();
        let median = |mut times: Vec<std::time::Duration>| {
            times.sort();
            times[times.len() / 2]
        };
        let timed = |add: &mut dyn FnMut(&mut DrizzleAccumulator)| {
            let mut sums = accumulator();
            // Fault the sums' pages in first, as an earlier frame has in a
            // real drizzle.
            sums.sum.fill(0.0);
            sums.weight.fill(0.0);
            let start = std::time::Instant::now();
            add(&mut sums);
            (start.elapsed(), sums)
        };
        for (name, frame) in ["straight", "flipped"].iter().zip(&frames) {
            let (mut before, mut after) = (Vec::new(), Vec::new());
            for _ in 0..3 {
                let (time, expected) =
                    timed(&mut |sums| reference::add(sums, &frame.frame()).unwrap());
                before.push(time);
                let (time, actual) = timed(&mut |sums| sums.add(&frame.frame()).unwrap());
                after.push(time);
                assert!(bits(&actual.sum) == bits(&expected.sum));
                assert!(bits(&actual.weight) == bits(&expected.weight));
            }
            println!(
                "{name}: reference {:?}, reworked {:?}",
                median(before),
                median(after)
            );
        }
        let batch = (0..4)
            .map(|index| frames[index % 2].frame())
            .collect::<Vec<_>>();
        let (mut before, mut after) = (Vec::new(), Vec::new());
        for _ in 0..3 {
            let (time, expected) = timed(&mut |sums| {
                for frame in &batch {
                    sums.add(frame).unwrap();
                }
            });
            before.push(time);
            let (time, actual) = timed(&mut |sums| sums.add_frames(&batch).unwrap());
            after.push(time);
            assert!(bits(&actual.sum) == bits(&expected.sum));
            assert!(bits(&actual.weight) == bits(&expected.weight));
        }
        println!(
            "four frames: one at a time {:?}, together {:?}",
            median(before),
            median(after)
        );
    }

    #[test]
    fn clipped_area_matches_known_overlaps() {
        let square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!((clipped_area(&square, 0.0, 0.0) - 1.0).abs() < 1e-12);
        assert!((clipped_area(&square, 0.5, 0.0) - 0.5).abs() < 1e-12);
        assert!((clipped_area(&square, 0.5, 0.5) - 0.25).abs() < 1e-12);
        assert_eq!(clipped_area(&square, 2.0, 0.0), 0.0);
        // A diamond of area 2 centred on a pixel corner covers half of
        // each of the four pixels around it.
        let diamond = [(1.0, 0.0), (2.0, 1.0), (1.0, 2.0), (0.0, 1.0)];
        for (x, y) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
            assert!((clipped_area(&diamond, x, y) - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn a_rotated_drop_spreads_its_whole_area() {
        let angle: f64 = 0.4;
        let jacobian = [angle.cos(), -angle.sin(), angle.sin(), angle.cos()];
        let shape = DropShape::new(jacobian, 0.45 * 2.0);
        assert!(shape.rectangle.is_none());
        let center = (10.3, 7.8);
        let (min_x, min_y, max_x, max_y) = shape.bounds(center.0, center.1);
        let mut total = 0.0;
        for y in min_y.floor() as i32..max_y.ceil() as i32 {
            for x in min_x.floor() as i32..max_x.ceil() as i32 {
                total += shape.overlap(center, f64::from(x), f64::from(y));
            }
        }
        assert!((total - 1.8 * 1.8).abs() < 1e-9, "{total}");
        // A nearly straight drop is a rectangle of the same area.
        let shape = DropShape::new([1.0, -0.002, 0.002, 1.0], 0.45);
        let (half_width, half_height) = shape.rectangle.unwrap();
        assert!((4.0 * half_width * half_height - 0.81 * (1.0 + 0.002 * 0.002)).abs() < 1e-12);
    }

    #[test]
    fn warp_inversion_recovers_the_reference_position() {
        let transform = crate::SimilarityTransform {
            scale: 1.0,
            rotation_radians: 0.01,
            translation_x: 3.0,
            translation_y: -2.0,
        };
        // A lens-like warp: the similarity plus a radial term.
        let pairs = (0..20)
            .flat_map(|row| (0..20).map(move |column| (column as f64 * 50.0, row as f64 * 40.0)))
            .map(|(x, y)| {
                let (source_x, source_y) = transform.inverse_apply(x, y);
                let radius = ((x - 500.0).powi(2) + (y - 400.0).powi(2)) * 2.0e-6;
                (
                    (x, y),
                    (
                        source_x + (x - 500.0) * radius,
                        source_y + (y - 400.0) * radius,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let warp = PolynomialWarp::fit(2, 1000, 800, &pairs).unwrap();
        for (x, y) in [(0.0, 0.0), (500.0, 400.0), (999.0, 10.0)] {
            let reference = invert_warp(&warp, transform, x, y);
            let (back_x, back_y) = warp.apply(reference.0, reference.1);
            assert!((back_x - x).abs() < 1e-5 && (back_y - y).abs() < 1e-5);
        }
    }
}
