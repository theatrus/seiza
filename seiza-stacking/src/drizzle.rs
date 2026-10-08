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
        let source = frame.image;
        let channels = self.channels;
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
        let scale = f64::from(self.options.scale);
        let map = ForwardMap::new(
            frame.mapping,
            source.width,
            source.height,
            f64::from(shrink) * 0.5 * scale,
        )?;
        let context = BandContext {
            frame,
            map: &map,
            inverse: &InverseMap::new(frame.mapping),
            normalization: frame.normalization.sampler(),
            scale,
            width: self.width,
            height: self.height,
            channels,
            reference_width: self.reference_width,
            reference_height: self.reference_height,
        };
        let band_samples = self.width * channels * TILE;
        self.sum
            .par_chunks_mut(band_samples)
            .zip(self.weight.par_chunks_mut(band_samples))
            .enumerate()
            .for_each(|(band, (sum, weight))| context.band(band * TILE, sum, weight));
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

impl BandContext<'_> {
    /// Drizzle into output rows `top..top + TILE`, held in `sum` and
    /// `weight`, one tile of columns at a time.
    fn band(&self, top: usize, sum: &mut [f32], weight: &mut [f32]) {
        let bottom = (top + TILE).min(self.height);
        let mut left = 0;
        while left < self.width {
            let right = (left + TILE).min(self.width);
            self.tile(top, bottom, left, right, sum, weight);
            left = right;
        }
    }

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
                if let Some((half_width, half_height)) = shape.rectangle {
                    // A rectangle's overlap with a pixel is the product of its
                    // overlaps along each axis.
                    for (overlap, pixel) in x_overlaps.iter_mut().zip(first_x..last_x) {
                        *overlap = interval_overlap(output_x, half_width, pixel as f64);
                    }
                    for (overlap, pixel) in y_overlaps.iter_mut().zip(first_y..last_y) {
                        *overlap = interval_overlap(output_y, half_height, pixel as f64);
                    }
                }
                let channel_range = match self.frame.layout {
                    Some(layout) => {
                        let channel = layout.channel_at(x, y);
                        channel..channel + 1
                    }
                    None => 0..channels,
                };
                for channel in channel_range {
                    let source_channel = if self.frame.layout.is_some() {
                        0
                    } else {
                        channel
                    };
                    let value =
                        source.data[(y * source.width + x) * source.channels + source_channel];
                    if !value.is_finite()
                        || self.fate(reference_x, reference_y, channel) == SampleFate::Rejected
                    {
                        continue;
                    }
                    let (gain, offset) = self.normalization.at(reference_x, reference_y, channel);
                    let value = value.mul_add(gain, offset);
                    let frame_weight = self.frame.weight[channel];
                    for (row_offset, output_y_index) in (first_y..last_y).enumerate() {
                        let row = (output_y_index - top) * row_samples;
                        for (column_offset, output_x_index) in (first_x..last_x).enumerate() {
                            let area = match shape.rectangle {
                                Some(_) => x_overlaps[column_offset] * y_overlaps[row_offset],
                                None => clipped_area(
                                    &shape.corners(output_x, output_y),
                                    output_x_index as f64,
                                    output_y_index as f64,
                                ),
                            };
                            if area <= 0.0 {
                                continue;
                            }
                            let drop_weight = area as f32 * frame_weight;
                            let index = row + output_x_index * channels + channel;
                            sum[index] += drop_weight * value;
                            weight[index] += drop_weight;
                        }
                    }
                }
            }
        }
    }

    /// What the integration did with the registered sample under a source
    /// pixel's centre. Under Bayer drizzle a registered pixel holds only the
    /// colour of its nearest photosite, so where the centre's pixel has no
    /// sample in this colour, the nearest neighbour that has one carries this
    /// photosite's sample.
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

    fn bounds(&self, x: f64, y: f64) -> (f64, f64, f64, f64) {
        let (half_width, half_height) = self.extent;
        (
            x - half_width,
            y - half_height,
            x + half_width,
            y + half_height,
        )
    }

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

/// The area of a convex quadrilateral inside the unit square at `(x, y)`,
/// by clipping it against the square's four edges.
fn clipped_area(corners: &[(f64, f64); 4], x: f64, y: f64) -> f64 {
    let mut polygon = [(0.0, 0.0); 8];
    let mut count = 4;
    polygon[..4].copy_from_slice(corners);
    let mut scratch = [(0.0, 0.0); 8];
    // Each edge keeps the side where `inside` holds and cuts crossing
    // segments where `cut` says.
    for edge in 0..4 {
        let inside = |point: (f64, f64)| match edge {
            0 => point.0 >= x,
            1 => point.0 <= x + 1.0,
            2 => point.1 >= y,
            _ => point.1 <= y + 1.0,
        };
        let cut = |from: (f64, f64), to: (f64, f64)| {
            let fraction = match edge {
                0 => (x - from.0) / (to.0 - from.0),
                1 => (x + 1.0 - from.0) / (to.0 - from.0),
                2 => (y - from.1) / (to.1 - from.1),
                _ => (y + 1.0 - from.1) / (to.1 - from.1),
            };
            (
                from.0 + (to.0 - from.0) * fraction,
                from.1 + (to.1 - from.1) * fraction,
            )
        };
        let mut next = 0;
        for index in 0..count {
            let current = polygon[index];
            let previous = polygon[(index + count - 1) % count];
            match (inside(previous), inside(current)) {
                (true, true) => {
                    scratch[next] = current;
                    next += 1;
                }
                (true, false) => {
                    scratch[next] = cut(previous, current);
                    next += 1;
                }
                (false, true) => {
                    scratch[next] = cut(previous, current);
                    scratch[next + 1] = current;
                    next += 2;
                }
                (false, false) => {}
            }
        }
        count = next;
        if count < 3 {
            return 0.0;
        }
        polygon[..count].copy_from_slice(&scratch[..count]);
    }
    let mut twice_area = 0.0;
    for index in 0..count {
        let (x0, y0) = polygon[index];
        let (x1, y1) = polygon[(index + 1) % count];
        twice_area += x0 * y1 - x1 * y0;
    }
    twice_area.abs() * 0.5
}

/// Reference positions of source pixels, and the drop shape each takes.
enum ForwardMap {
    Similarity {
        transform: crate::SimilarityTransform,
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
                transform,
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

    fn at(&self, x: usize, y: usize) -> ((f64, f64), &DropShape) {
        match self {
            Self::Similarity { transform, shape } => (transform.apply(x as f64, y as f64), shape),
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

    fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        match self {
            Self::Similarity(transform) => transform.inverse_apply(x, y),
            Self::Warp(warp) => warp.apply(x, y),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
