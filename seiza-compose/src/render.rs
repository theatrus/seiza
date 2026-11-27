//! The compositor: renders a [`Picture`] in tiles.
//!
//! The output is red, green and blue in 32-bit floats. Each tile reads its
//! inputs over the tile and a margin wide enough for everything that reads
//! beyond a pixel: a computed mask's blur, the heals, and a Tone step's
//! neighborhoods. Within the margin,
//! pixels outside the picture's region are its pixels reflected at its edge,
//! as Foraxx's smoothing always reflected them, so a tile gives the same
//! pixels as the whole picture would. Tiles render in parallel in the
//! caller's Rayon pool, read from a [`Source`] and go to a [`Sink`], so the
//! picture is never held whole unless the sink holds it.
//!
//! A pixel any layer's or computed mask's input does not cover (a
//! non-finite sample) is not covered in the output, and is `NaN` there.

use crate::picture::{
    Adjustment, Blend, ComputedMask, FinishKind, InputRead, LayerKind, LuminanceTarget, Picture,
    Segment,
};
use crate::space::WorkingSpace;
use crate::{Error, LinearImage, ReferenceRegion, Result};
use rayon::prelude::*;
use seiza_imgproc::BorderMode;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Where a picture's inputs come from: planes of one size, read a rectangle
/// at a time.
pub trait Source: Sync {
    /// The inputs' width and height.
    fn size(&self) -> (usize, usize);
    /// The samples of `input` over `rect`, row after row.
    fn read(&self, input: usize, rect: ReferenceRegion) -> Result<Vec<f32>>;
}

/// Where a rendered picture goes: tiles of red, green and blue, interleaved,
/// in any order and from any thread.
pub trait Sink: Sync {
    /// Takes the pixels of `rect`, given in the picture's coordinates.
    fn put(&self, rect: ReferenceRegion, rgb: &[f32]) -> Result<()>;
}

/// How [`render`] works through a picture.
pub struct RenderOptions<'a> {
    /// The side of a square tile, in pixels.
    pub tile: usize,
    /// Asked between tiles; when it gives `true`, the render ends with
    /// [`Error::Cancelled`].
    pub stop: Option<&'a (dyn Fn() -> bool + Sync)>,
    /// Given the share of tiles done, from 0 to 1, as tiles finish.
    pub progress: Option<&'a (dyn Fn(f32) + Sync)>,
}

impl Default for RenderOptions<'_> {
    fn default() -> Self {
        Self {
            tile: 512,
            stop: None,
            progress: None,
        }
    }
}

/// Renders `region` of `picture` from `source` into `sink`, in tiles.
/// `region` is where the picture lies on its inputs' grid, and the edge its
/// margins reflect at.
pub fn render(
    picture: &Picture,
    source: &dyn Source,
    region: ReferenceRegion,
    sink: &dyn Sink,
    options: &RenderOptions<'_>,
) -> Result<()> {
    let (width, height) = source.size();
    if region.width == 0
        || region.height == 0
        || region.x + region.width > width
        || region.y + region.height > height
    {
        return Err(Error::Color(format!(
            "the region {}x{} at ({}, {}) is not within the inputs' {width}x{height}",
            region.width, region.height, region.x, region.y
        )));
    }
    let plan = Plan::new(picture);
    let tile = options.tile.max(16);
    let tiles: Vec<ReferenceRegion> = (0..region.height.div_ceil(tile))
        .flat_map(|row| {
            (0..region.width.div_ceil(tile)).map(move |column| {
                let x = region.x + column * tile;
                let y = region.y + row * tile;
                ReferenceRegion {
                    x,
                    y,
                    width: tile.min(region.x + region.width - x),
                    height: tile.min(region.y + region.height - y),
                }
            })
        })
        .collect();
    let done = AtomicUsize::new(0);
    tiles.par_iter().try_for_each(|tile| {
        if options.stop.is_some_and(|stop| stop()) {
            return Err(Error::Cancelled);
        }
        let pixels = plan.render_tile(picture, source, region, *tile)?;
        sink.put(*tile, &pixels)?;
        let finished = done.fetch_add(1, Ordering::Relaxed) + 1;
        if let Some(progress) = options.progress {
            progress(finished as f32 / tiles.len() as f32);
        }
        Ok(())
    })
}

/// Renders `region` of `picture` from `source` into an image of its own.
pub fn render_image(
    picture: &Picture,
    source: &dyn Source,
    region: ReferenceRegion,
    options: &RenderOptions<'_>,
) -> Result<LinearImage> {
    let sink = ImageSink::new(region);
    render(picture, source, region, &sink, options)?;
    sink.into_image()
}

/// A painted mask over `region`, row after row, from 0 to 1: its strokes
/// drawn as the compositor draws them, each segment in order. A host uses
/// it to know where a step applies, such as a histogram of a masked step's
/// area.
pub fn paint_mask(segments: &[Segment], region: ReferenceRegion) -> Vec<f32> {
    let padded = Padded {
        x: region.x as i64,
        y: region.y as i64,
        width: region.width,
        height: region.height,
    };
    let mut mask = vec![0.0_f32; padded.pixels()];
    for segment in segments {
        draw_segment(&mut mask, &padded, segment);
    }
    mask
}

/// A [`Source`] over images already in memory, one per input.
pub struct ImageSource<'a> {
    images: Vec<&'a LinearImage>,
}

impl<'a> ImageSource<'a> {
    /// Inputs from these mono images, by their place in the list. They must
    /// share one size.
    pub fn new(images: Vec<&'a LinearImage>) -> Result<Self> {
        let first = images
            .first()
            .ok_or_else(|| Error::Color("a picture needs at least one input".into()))?;
        if images.iter().any(|image| {
            image.channels != 1 || image.width != first.width || image.height != first.height
        }) {
            return Err(Error::Color(
                "a picture's inputs must be mono images of one size".into(),
            ));
        }
        Ok(Self { images })
    }
}

impl Source for ImageSource<'_> {
    fn size(&self) -> (usize, usize) {
        (self.images[0].width, self.images[0].height)
    }

    fn read(&self, input: usize, rect: ReferenceRegion) -> Result<Vec<f32>> {
        let image = self
            .images
            .get(input)
            .ok_or_else(|| Error::Color(format!("the picture has no input {input}")))?;
        let mut out = Vec::with_capacity(rect.width * rect.height);
        for row in rect.y..rect.y + rect.height {
            let start = row * image.width + rect.x;
            out.extend_from_slice(&image.data[start..start + rect.width]);
        }
        Ok(out)
    }
}

/// A [`Sink`] that keeps the whole picture in memory.
pub struct ImageSink {
    region: ReferenceRegion,
    rgb: Mutex<Vec<f32>>,
}

impl ImageSink {
    /// A sink for `region`, black until tiles arrive.
    pub fn new(region: ReferenceRegion) -> Self {
        Self {
            region,
            rgb: Mutex::new(vec![0.0; region.width * region.height * 3]),
        }
    }

    /// The picture, as a three-channel image of the region's size.
    pub fn into_image(self) -> Result<LinearImage> {
        let rgb = self
            .rgb
            .into_inner()
            .map_err(|_| Error::Color("a tile's render panicked".into()))?;
        LinearImage::new(self.region.width, self.region.height, 3, rgb)
    }
}

impl Sink for ImageSink {
    fn put(&self, rect: ReferenceRegion, rgb: &[f32]) -> Result<()> {
        let mut out = self
            .rgb
            .lock()
            .map_err(|_| Error::Color("a tile's render panicked".into()))?;
        let row_length = rect.width * 3;
        for (row, line) in rgb.chunks(row_length).enumerate() {
            let y = rect.y - self.region.y + row;
            let start = (y * self.region.width + rect.x - self.region.x) * 3;
            out[start..start + row_length].copy_from_slice(line);
        }
        Ok(())
    }
}

/// A contrast curve: an S-curve for a positive amount, a flattening one
/// for a negative amount. 0 keeps `x`, held within 0 to 1.
pub(crate) fn contrast_curve(x: f32, amount: f32) -> f32 {
    let clamped = x.clamp(0.0, 1.0);
    if amount >= 0.0 {
        mix(
            clamped,
            clamped * clamped * (3.0 - 2.0 * clamped),
            amount.min(1.0),
        )
    } else {
        mix(clamped, 0.5, (-amount).min(1.0) * 0.5)
    }
}

/// The part of the inputs a tile is drawn from: the tile and its margin.
/// Its pixels outside the picture's region are reflections of pixels in it.
struct Padded {
    x: i64,
    y: i64,
    width: usize,
    height: usize,
}

impl Padded {
    fn pixels(&self) -> usize {
        self.width * self.height
    }

    /// The index of a pixel given relative to the padded area, if it is in it.
    fn at(&self, x: i64, y: i64) -> Option<usize> {
        (x >= 0 && y >= 0 && x < self.width as i64 && y < self.height as i64)
            .then(|| y as usize * self.width + x as usize)
    }
}

/// A tile's inputs with their margin: each input's plane, and the padded
/// area's left and top in the picture's pixels, its width and its height.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) struct PaddedPlanes {
    pub(crate) planes: Vec<Vec<f32>>,
    pub(crate) origin: [i64; 2],
    pub(crate) width: usize,
    pub(crate) height: usize,
}

/// What a picture needs worked out once: its inputs, and how far a tile's
/// pixels can read beyond it.
pub(crate) struct Plan {
    pub(crate) inputs: Vec<usize>,
    pub(crate) margin: usize,
}

impl Plan {
    pub(crate) fn new(picture: &Picture) -> Self {
        let blur = picture
            .computed
            .iter()
            .filter(|mask| mask.smoothing > 0.0)
            .map(|mask| seiza_imgproc::blur::auto_ksize_f32(f64::from(mask.smoothing)) / 2 + 1)
            .max()
            .unwrap_or(0);
        let reach = |offset: [i64; 2]| offset[0].unsigned_abs().max(offset[1].unsigned_abs());
        // A layer heal reads one level: before the transfer, then after it.
        let layer_heals = [true, false]
            .into_iter()
            .map(|before| {
                picture
                    .heals
                    .iter()
                    .filter(|heal| heal.before_transfer == before)
                    .map(|heal| reach(heal.offset))
                    .max()
                    .unwrap_or(0)
            })
            .sum::<u64>();
        // The steps in the stack, then the Finish steps.
        let stacked: Vec<&FinishKind> = picture
            .layers
            .iter()
            .filter_map(|layer| match &layer.kind {
                LayerKind::Step(kind) => Some(kind),
                _ => None,
            })
            .collect();
        let steps = || {
            stacked
                .iter()
                .copied()
                .chain(picture.finish.iter().map(|step| &step.kind))
        };
        // A heal in the stack reads the picture an earlier one changed, so
        // their reaches add; the Finish steps' heals read the same picture.
        let stacked_heals: u64 = stacked
            .iter()
            .filter_map(|kind| match kind {
                FinishKind::Heal { offset, .. } => Some(reach(*offset)),
                _ => None,
            })
            .sum();
        let finish_heals = picture
            .finish
            .iter()
            .filter_map(|step| match step.kind {
                FinishKind::Heal { offset, .. } => Some(reach(offset)),
                _ => None,
            })
            .max()
            .unwrap_or(0)
            + stacked_heals;
        // A Tone step that reads neighborhoods reads that far around, in
        // a picture an earlier such step already read around.
        let local = steps()
            .filter(|kind| matches!(kind, FinishKind::Tone(tone) if tone.is_local()))
            .count()
            * crate::tone::LOCAL_RADIUS;
        Self {
            inputs: picture.inputs(),
            margin: blur + (layer_heals + finish_heals) as usize + local,
        }
    }

    fn render_tile(
        &self,
        picture: &Picture,
        source: &dyn Source,
        region: ReferenceRegion,
        tile: ReferenceRegion,
    ) -> Result<Vec<f32>> {
        let margin = self.margin as i64;
        let padded = Padded {
            x: tile.x as i64 - margin,
            y: tile.y as i64 - margin,
            width: tile.width + 2 * self.margin,
            height: tile.height + 2 * self.margin,
        };
        let planes = self.read_planes(source, region, &padded)?;
        let plane = |input: usize| -> &[f32] {
            let at = self
                .inputs
                .binary_search(&input)
                .expect("every input a picture reads is read");
            &planes[at]
        };
        let masks: Vec<Vec<f32>> = picture
            .masks
            .iter()
            .map(|segments| {
                let mut mask = vec![0.0_f32; padded.pixels()];
                for segment in segments {
                    draw_segment(&mut mask, &padded, segment);
                }
                mask
            })
            .collect();
        let read = |read: &InputRead| -> Vec<f32> {
            let mut samples: Vec<f32> = match read.blended {
                Some((other, amount)) => plane(read.input)
                    .iter()
                    .zip(plane(other))
                    .map(|(first, second)| mix(*first, *second, amount.clamp(0.0, 1.0)))
                    .collect(),
                None => plane(read.input).to_vec(),
            };
            read.transfer.apply_guided(
                &mut samples,
                [padded.x, padded.y],
                padded.width,
                &|input| Some(plane(input)),
            );
            samples
        };
        // Where every input the picture reads is covered.
        let covered: Vec<bool> = (0..padded.pixels())
            .map(|at| {
                self.inputs
                    .iter()
                    .all(|input| plane(*input)[at].is_finite())
            })
            .collect();
        let computed: Vec<Vec<f32>> = picture
            .computed
            .iter()
            .map(|mask| computed_mask(mask, &padded, &masks, &covered, &read))
            .collect();
        let layers = LayerPlanes::new(picture, &padded, &masks, plane);
        // The layers blend up to each step in the stack, which then works on
        // what they made over the whole padded area, and on from there.
        let mut state = vec![Blended::default(); padded.pixels()];
        let mut start = 0;
        let ends = picture
            .layers
            .iter()
            .enumerate()
            .filter(|(_, layer)| matches!(layer.kind, LayerKind::Step(_)))
            .map(|(index, _)| index)
            .chain([picture.layers.len()]);
        for end in ends {
            let range = start..end;
            state.par_iter_mut().enumerate().for_each(|(at, blended)| {
                *blended = combine(
                    picture,
                    &layers,
                    &masks,
                    &computed,
                    at,
                    range.clone(),
                    *blended,
                );
            });
            if let Some(layer) = picture.layers.get(end)
                && let LayerKind::Step(kind) = &layer.kind
            {
                let cover: Vec<f32> = (0..padded.pixels())
                    .map(|at| layer_cover(layer, &masks, &computed, at))
                    .collect();
                let mut rgb: Vec<f32> = state.iter().flat_map(|blended| blended.color).collect();
                apply_step(
                    &picture.rendered_kind(kind),
                    &cover,
                    &mut rgb,
                    &padded,
                    picture.space,
                );
                for (blended, color) in state.iter_mut().zip(rgb.as_chunks::<3>().0) {
                    blended.color = *color;
                }
            }
            start = end + 1;
        }
        let combined: Vec<f32> = state.iter().flat_map(|blended| blended.color).collect();
        let finished = finish(picture, &padded, &masks, combined);
        let mut out = Vec::with_capacity(tile.width * tile.height * 3);
        for row in 0..tile.height {
            for column in 0..tile.width {
                let at = (row + self.margin) * padded.width + column + self.margin;
                if covered[at] {
                    out.extend_from_slice(&finished[at * 3..at * 3 + 3]);
                } else {
                    out.extend_from_slice(&[f32::NAN; 3]);
                }
            }
        }
        Ok(out)
    }

    /// Each input over `tile` and the margin around it, in the order of
    /// [`Self::inputs`]: the part inside the picture's region read from the
    /// source, the rest reflected from it.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn padded_planes(
        &self,
        source: &dyn Source,
        region: ReferenceRegion,
        tile: ReferenceRegion,
    ) -> Result<PaddedPlanes> {
        let margin = self.margin as i64;
        let padded = Padded {
            x: tile.x as i64 - margin,
            y: tile.y as i64 - margin,
            width: tile.width + 2 * self.margin,
            height: tile.height + 2 * self.margin,
        };
        let planes = self.read_planes(source, region, &padded)?;
        Ok(PaddedPlanes {
            planes,
            origin: [padded.x, padded.y],
            width: padded.width,
            height: padded.height,
        })
    }

    /// Each input over the padded area: the part inside the picture's region
    /// read from the source, the rest reflected from it.
    fn read_planes(
        &self,
        source: &dyn Source,
        region: ReferenceRegion,
        padded: &Padded,
    ) -> Result<Vec<Vec<f32>>> {
        let left = padded.x.max(region.x as i64) as usize;
        let top = padded.y.max(region.y as i64) as usize;
        let right = (padded.x + padded.width as i64).min((region.x + region.width) as i64) as usize;
        let bottom =
            (padded.y + padded.height as i64).min((region.y + region.height) as i64) as usize;
        let inside = ReferenceRegion {
            x: left,
            y: top,
            width: right - left,
            height: bottom - top,
        };
        let reflect = |position: i64, start: usize, length: usize| {
            BorderMode::Reflect.map((position - start as i64) as isize, length) + start
        };
        let columns: Vec<usize> = (0..padded.width as i64)
            .map(|x| reflect(padded.x + x, region.x, region.width) - inside.x)
            .collect();
        let rows: Vec<usize> = (0..padded.height as i64)
            .map(|y| reflect(padded.y + y, region.y, region.height) - inside.y)
            .collect();
        self.inputs
            .iter()
            .map(|input| {
                let samples = source.read(*input, inside)?;
                if samples.len() != inside.width * inside.height {
                    return Err(Error::Color(format!(
                        "input {input} gave {} samples for a {}x{} area",
                        samples.len(),
                        inside.width,
                        inside.height
                    )));
                }
                let mut plane = Vec::with_capacity(padded.pixels());
                for row in &rows {
                    let line = &samples[row * inside.width..(row + 1) * inside.width];
                    plane.extend(columns.iter().map(|column| line[*column]));
                }
                Ok(plane)
            })
            .collect()
    }
}

/// Each layer's values over the padded area, with its heals applied.
struct LayerPlanes {
    values: Vec<Option<Vec<f32>>>,
}

impl LayerPlanes {
    fn new<'a>(
        picture: &Picture,
        padded: &Padded,
        masks: &[Vec<f32>],
        plane: impl Fn(usize) -> &'a [f32],
    ) -> Self {
        let raw = |read: &InputRead| -> Vec<f32> {
            match read.blended {
                Some((other, amount)) => plane(read.input)
                    .iter()
                    .zip(plane(other))
                    .map(|(first, second)| mix(*first, *second, amount.clamp(0.0, 1.0)))
                    .collect(),
                None => plane(read.input).to_vec(),
            }
        };
        let values = picture
            .layers
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                let read = match &layer.kind {
                    LayerKind::Input { read, .. } => read,
                    LayerKind::Luminance {
                        read: Some(read), ..
                    } => read,
                    _ => return None,
                };
                let curve = match layer.kind {
                    LayerKind::Input { curve, .. } => curve,
                    _ => 0.0,
                };
                let heals: Vec<_> = picture
                    .heals
                    .iter()
                    .filter(|heal| heal.layer == index && heal.mask < masks.len())
                    .collect();
                let samples = raw(read);
                // Heals before the transfer copy the unhealed samples.
                let mut healed = samples.clone();
                for heal in heals.iter().filter(|heal| heal.before_transfer) {
                    copy_heal(
                        &mut healed,
                        &samples,
                        padded,
                        heal.offset,
                        heal.shift,
                        &masks[heal.mask],
                    );
                }
                read.transfer.apply_guided(
                    &mut healed,
                    [padded.x, padded.y],
                    padded.width,
                    &|input| Some(plane(input)),
                );
                if curve != 0.0 {
                    healed
                        .iter_mut()
                        .for_each(|value| *value = contrast_curve(*value, curve));
                }
                // Heals after it copy the transferred, healed values.
                let transferred = healed.clone();
                for heal in heals.iter().filter(|heal| !heal.before_transfer) {
                    copy_heal(
                        &mut healed,
                        &transferred,
                        padded,
                        heal.offset,
                        heal.shift,
                        &masks[heal.mask],
                    );
                }
                Some(healed)
            })
            .collect();
        Self { values }
    }
}

/// Where `mask` is painted, `values` take `from` at `offset` pixels away,
/// with `shift` added.
fn copy_heal(
    values: &mut [f32],
    from: &[f32],
    padded: &Padded,
    offset: [i64; 2],
    shift: f32,
    mask: &[f32],
) {
    for y in 0..padded.height as i64 {
        for x in 0..padded.width as i64 {
            let at = y as usize * padded.width + x as usize;
            if mask[at] <= 0.0 {
                continue;
            }
            if let Some(source) = padded.at(x + offset[0], y + offset[1]) {
                values[at] = mix(values[at], from[source] + shift, mask[at]);
            }
        }
    }
}

/// A computed mask over the padded area.
/// A pixel the picture does not cover, or the mask's inputs do not, has no
/// weight in its smoothing, as a channel Foraxx's blur lacks has none.
fn computed_mask(
    mask: &ComputedMask,
    padded: &Padded,
    masks: &[Vec<f32>],
    covered: &[bool],
    read: &impl Fn(&InputRead) -> Vec<f32>,
) -> Vec<f32> {
    let reads: Vec<Vec<f32>> = mask.reads.iter().map(read).collect();
    let mut weights = vec![1.0_f32; padded.pixels()];
    let mut shape: Vec<f32> = (0..padded.pixels())
        .map(|at| {
            if !covered[at] || reads.iter().any(|plane| !plane[at].is_finite()) {
                weights[at] = 0.0;
                return 0.0;
            }
            let product = reads
                .iter()
                .fold(1.0_f32, |product, plane| product * plane[at])
                .clamp(0.0, 1.0);
            let shaped = if mask.power_of_inverse {
                product.powf(1.0 - product)
            } else {
                product
            };
            if mask.reach == 1.0 {
                shaped
            } else {
                shaped.powf(mask.reach.clamp(0.05, 20.0))
            }
        })
        .collect();
    if mask.smoothing > 0.0 {
        let sigma = f64::from(mask.smoothing);
        let blur = |plane: &[f32]| {
            seiza_imgproc::blur::gaussian_blur_f32(
                plane,
                padded.width,
                padded.height,
                seiza_imgproc::blur::auto_ksize_f32(sigma),
                sigma,
                BorderMode::Reflect,
            )
        };
        // Uncovered pixels enter the blur as 0 with no weight, so the
        // blurred values over the blurred weights are the weighted mean of
        // the covered neighbours. Dividing everywhere, not only where some
        // pixel is uncovered, keeps a tile's pixels the same whatever other
        // tiles hold.
        let blurred = blur(&shape);
        let blurred_weights = blur(&weights);
        shape = blurred
            .iter()
            .zip(&blurred_weights)
            .map(|(value, weight)| {
                if *weight > 0.0 {
                    (value / weight).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            })
            .collect();
    }
    let painted = |index: Option<usize>| index.and_then(|index| masks.get(index));
    if let Some(added) = painted(mask.added) {
        shape
            .iter_mut()
            .zip(added)
            .for_each(|(value, add)| *value = value.max(*add));
    }
    if let Some(removed) = painted(mask.removed) {
        shape
            .iter_mut()
            .zip(removed)
            .for_each(|(value, remove)| *value *= 1.0 - remove);
    }
    if mask.inverted {
        shape.iter_mut().for_each(|value| *value = 1.0 - *value);
    }
    shape
}

/// What the layers blended so far leave at one pixel, at a step in the
/// stack or at the top: the blend, and the picture beneath each group it is
/// in, outermost first.
#[derive(Clone, Copy, Debug, Default)]
struct Blended {
    color: [f32; 3],
    beneath: [[f32; 3]; crate::picture::MAX_GROUP_DEPTH],
    /// How many groups the layers so far are inside.
    depth: usize,
    /// How many groups past [`crate::picture::MAX_GROUP_DEPTH`] are open;
    /// their members blend into the group around them.
    flattened: usize,
}

/// How much of a layer applies at one pixel of the padded area: its
/// opacity, within its mask, the mask of the layer it is clipped to and its
/// computed mask.
fn layer_cover(
    layer: &crate::picture::Layer,
    masks: &[Vec<f32>],
    computed: &[Vec<f32>],
    at: usize,
) -> f32 {
    let painted = |mask: Option<usize>| {
        mask.and_then(|mask| masks.get(mask))
            .map_or(1.0, |mask| mask[at])
    };
    let mut cover = layer.opacity.clamp(0.0, 1.0) * painted(layer.mask) * painted(layer.clip);
    if let Some(mask) = layer.computed.and_then(|mask| computed.get(mask)) {
        cover *= mask[at];
    }
    cover
}

/// The layers in `range` blended at one pixel of the padded area, onto what
/// the layers before them left. A step ends a range, so nothing waits to be
/// blended past its end.
fn combine(
    picture: &Picture,
    layers: &LayerPlanes,
    masks: &[Vec<f32>],
    computed: &[Vec<f32>],
    at: usize,
    range: std::ops::Range<usize>,
    from: Blended,
) -> Blended {
    let space = picture.space;
    let peak = picture.peak();
    let mut color = from.color;
    let mut beneath = from.beneath;
    let mut depth = from.depth;
    let mut flattened = from.flattened;
    let mut waiting = false;
    let mut own = [0.0_f32; 3];
    let mut own_cover = 0.0_f32;
    let mut own_mode = Blend::Normal;
    // For an input layer waiting to be blended: its tint and value, so an
    // added layer is one fused multiply-add per channel, as the recipes
    // have always computed their sums.
    let mut own_parts: Option<([f32; 3], f32)> = None;
    let flush = |color: [f32; 3],
                 own: [f32; 3],
                 cover: f32,
                 mode: Blend,
                 parts: Option<([f32; 3], f32)>| {
        match (mode, parts) {
            (Blend::Add, Some((tint, value))) if cover > 0.0 => {
                if cover >= 1.0 {
                    [0, 1, 2].map(|channel| tint[channel].mul_add(value, color[channel]))
                } else {
                    [0, 1, 2].map(|channel| (tint[channel] * value).mul_add(cover, color[channel]))
                }
            }
            _ => cover_mix(color, blend(mode, color, own, space, peak), cover),
        }
    };
    for (index, layer) in picture
        .layers
        .iter()
        .enumerate()
        .take(range.end)
        .skip(range.start)
    {
        let cover = layer_cover(layer, masks, computed, at);
        if let LayerKind::Adjustment(adjustment) = layer.kind
            && layer.own
        {
            if waiting {
                own = cover_mix(own, adjusted(own, adjustment, space), cover);
                own_parts = None;
            }
            continue;
        }
        if waiting {
            color = flush(color, own, own_cover, own_mode, own_parts.take());
            waiting = false;
        }
        match &layer.kind {
            LayerKind::Solid { color: solid } => {
                own = *solid;
                own_cover = cover;
                own_mode = layer.blend;
                waiting = true;
            }
            // A group nested past the deepest kept is flattened: its
            // members blend into the group around it, and its end does
            // nothing.
            LayerKind::GroupEnd if flattened > 0 => flattened -= 1,
            LayerKind::GroupEnd if depth > 0 => {
                own = color;
                own_cover = cover;
                own_mode = layer.blend;
                waiting = true;
                depth -= 1;
                color = beneath[depth];
            }
            // An end with no start ends nothing.
            LayerKind::GroupEnd => {}
            LayerKind::GroupBegin if depth == crate::picture::MAX_GROUP_DEPTH => flattened += 1,
            LayerKind::GroupBegin => {
                beneath[depth] = color;
                depth += 1;
                color = [0.0; 3];
            }
            LayerKind::Adjustment(adjustment) => {
                color = cover_mix(color, adjusted(color, *adjustment, space), cover);
            }
            // A step works on the whole area at the end of a range.
            LayerKind::Step(_) => {}
            LayerKind::Luminance { read, target } => {
                let value = read
                    .as_ref()
                    .and_then(|_| layers.values[index].as_ref())
                    .map(|values| values[at]);
                color = luminance(color, value, *target, cover, space);
            }
            LayerKind::Input { color: tint, .. } => {
                let value = layers.values[index]
                    .as_ref()
                    .map_or(0.0, |values| values[at]);
                own = tint.map(|channel| channel * value);
                own_parts = Some((*tint, value));
                own_cover = cover;
                own_mode = layer.blend;
                waiting = true;
            }
        }
    }
    if waiting {
        color = flush(color, own, own_cover, own_mode, own_parts);
    }
    Blended {
        color,
        beneath,
        depth,
        flattened,
    }
}

/// A step in the stack over the padded area's picture, `rgb`, applied as
/// far as `cover` says at each pixel. A heal copies, and a local Tone step
/// measures, the picture as the layers beneath left it.
fn apply_step(
    kind: &FinishKind,
    cover: &[f32],
    rgb: &mut [f32],
    padded: &Padded,
    space: WorkingSpace,
) {
    match kind {
        FinishKind::Heal { offset, shift } => {
            let from = rgb.to_vec();
            for y in 0..padded.height as i64 {
                for x in 0..padded.width as i64 {
                    let at = y as usize * padded.width + x as usize;
                    if cover[at] <= 0.0 {
                        continue;
                    }
                    if let Some(source) = padded.at(x + offset[0], y + offset[1]) {
                        for channel in 0..3 {
                            let copied = from[source * 3 + channel] + shift[channel];
                            rgb[at * 3 + channel] = mix(rgb[at * 3 + channel], copied, cover[at]);
                        }
                    }
                }
            }
        }
        FinishKind::Tone(tone) if tone.is_local() => {
            let around = crate::tone::neighborhoods(rgb, padded.width, padded.height, space);
            let measured: Vec<f32> = rgb
                .as_chunks::<3>()
                .0
                .iter()
                .map(|pixel| {
                    if pixel.iter().all(|channel| channel.is_finite()) {
                        space.luma(*pixel)
                    } else {
                        0.0
                    }
                })
                .collect();
            rgb.as_chunks_mut::<3>()
                .0
                .par_iter_mut()
                .enumerate()
                .for_each(|(at, pixel)| {
                    let toned = tone.apply_around(*pixel, measured[at], &around[at], space);
                    *pixel = cover_mix(*pixel, toned, cover[at]);
                });
        }
        kind => {
            let adjusted = adjust_finish(kind, rgb, space);
            for (at, (pixel, new)) in rgb
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(adjusted.as_chunks::<3>().0)
                .enumerate()
            {
                *pixel = cover_mix(*pixel, *new, cover[at]);
            }
        }
    }
}

/// A new luminance for `color`, its chromaticity kept, as LRGB composes.
fn luminance(
    color: [f32; 3],
    value: Option<f32>,
    target: LuminanceTarget,
    cover: f32,
    space: WorkingSpace,
) -> [f32; 3] {
    let own = space.luma(color);
    let sum = color[0] + color[1] + color[2];
    let wanted = match (target, value) {
        (LuminanceTarget::Replace, Some(value)) => (1.0 - cover).mul_add(own, cover * value),
        (LuminanceTarget::Replace, None) => own,
        (LuminanceTarget::Super, Some(value)) => value + sum,
        (LuminanceTarget::Super, None) => sum,
    };
    let scaled = if own > 1.0e-8 && wanted.is_finite() {
        let scale = wanted / own;
        color.map(|channel| channel * scale)
    } else {
        [wanted; 3]
    };
    match target {
        // Replacement weighs the cover in its target already.
        LuminanceTarget::Replace => scaled,
        LuminanceTarget::Super => cover_mix(color, scaled, cover),
    }
}

/// The Finish steps over the padded area. A heal copies the picture as the
/// steps before it left it, heals left out.
fn finish(picture: &Picture, padded: &Padded, masks: &[Vec<f32>], combined: Vec<f32>) -> Vec<f32> {
    if picture.finish.is_empty() {
        return combined;
    }
    let mut out = combined.clone();
    let mut plain = combined;
    for step in picture.finish_steps().iter() {
        let cover = |at: usize| {
            step.opacity.clamp(0.0, 1.0)
                * step
                    .mask
                    .and_then(|mask| masks.get(mask))
                    .map_or(1.0, |mask| mask[at])
        };
        match &step.kind {
            FinishKind::Heal { offset, shift } => {
                let from = plain.clone();
                for y in 0..padded.height as i64 {
                    for x in 0..padded.width as i64 {
                        let at = y as usize * padded.width + x as usize;
                        let applies = cover(at);
                        if applies <= 0.0 {
                            continue;
                        }
                        if let Some(source) = padded.at(x + offset[0], y + offset[1]) {
                            for channel in 0..3 {
                                let copied = from[source * 3 + channel] + shift[channel];
                                out[at * 3 + channel] = mix(out[at * 3 + channel], copied, applies);
                            }
                        }
                    }
                }
            }
            FinishKind::Tone(tone) if tone.is_local() => {
                // The neighborhoods are of the picture as the steps before
                // left it, heals left out, as a heal's source is.
                let around =
                    crate::tone::neighborhoods(&plain, padded.width, padded.height, picture.space);
                let measured: Vec<f32> = plain
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|pixel| {
                        if pixel.iter().all(|channel| channel.is_finite()) {
                            picture.space.luma(*pixel)
                        } else {
                            0.0
                        }
                    })
                    .collect();
                for buffer in [&mut out, &mut plain] {
                    buffer
                        .as_chunks_mut::<3>()
                        .0
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(at, pixel)| {
                            let toned =
                                tone.apply_around(*pixel, measured[at], &around[at], picture.space);
                            *pixel = cover_mix(*pixel, toned, cover(at));
                        });
                }
            }
            kind => {
                for buffer in [&mut out, &mut plain] {
                    let adjusted = adjust_finish(kind, buffer, picture.space);
                    for (at, (pixel, new)) in buffer
                        .as_chunks_mut::<3>()
                        .0
                        .iter_mut()
                        .zip(adjusted.as_chunks::<3>().0)
                        .enumerate()
                    {
                        *pixel = cover_mix(*pixel, *new, cover(at));
                    }
                }
            }
        }
    }
    out
}

/// A Finish step that maps each pixel alone, over interleaved RGB.
fn adjust_finish(kind: &FinishKind, rgb: &[f32], space: WorkingSpace) -> Vec<f32> {
    match kind {
        FinishKind::Stretch(plan) => {
            // Uncovered pixels are NaN in the output whatever a stretch
            // gives them; stretch them as black.
            let finite: Vec<f32> = rgb
                .iter()
                .map(|value| if value.is_finite() { *value } else { 0.0 })
                .collect();
            plan.apply_f32(&finite, 3).unwrap_or(finite)
        }
        FinishKind::Tone(tone) => rgb
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|pixel| tone.apply(*pixel, space))
            .collect(),
        FinishKind::Curves(amount) => rgb
            .iter()
            .map(|value| contrast_curve(*value, *amount))
            .collect(),
        FinishKind::Saturation(amount) => rgb
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|pixel| saturated(*pixel, *amount, space))
            .collect(),
        FinishKind::Heal { .. } => rgb.to_vec(),
    }
}

fn saturated(color: [f32; 3], amount: f32, space: WorkingSpace) -> [f32; 3] {
    let luminance = space.luma(color);
    color.map(|channel| mix(luminance, channel, amount).max(0.0))
}

/// How a layer's color combines with what is beneath it, before opacity.
/// `peak` is the brightest value the picture shows: screen works up to it,
/// so values above SDR white keep their order under a screened layer.
fn blend(
    mode: Blend,
    below: [f32; 3],
    layer: [f32; 3],
    space: WorkingSpace,
    peak: f32,
) -> [f32; 3] {
    match mode {
        Blend::Normal => layer,
        Blend::Screen => {
            let peak = peak.max(1.0);
            [0, 1, 2].map(|channel| screen(below[channel], layer[channel], peak))
        }
        Blend::Add => [
            below[0] + layer[0],
            below[1] + layer[1],
            below[2] + layer[2],
        ],
        Blend::Multiply => [
            below[0] * layer[0],
            below[1] * layer[1],
            below[2] * layer[2],
        ],
        Blend::Lighten => [
            below[0].max(layer[0]),
            below[1].max(layer[1]),
            below[2].max(layer[2]),
        ],
        Blend::Darken => [
            below[0].min(layer[0]),
            below[1].min(layer[1]),
            below[2].min(layer[2]),
        ],
        Blend::Luminance => {
            let shift = space.luma(layer) - space.luma(below);
            below.map(|channel| (channel + shift).max(0.0))
        }
    }
}

/// Screen over the range 0 to `peak`: `peak · (1 − (1 − a/peak)(1 − b/peak))`.
/// At a peak of 1 it is the usual screen. Above SDR white it keeps what is
/// beneath: a bright core under a star layer stays brighter where it was
/// brighter, rather than every value meeting at 1.
pub(crate) fn screen(below: f32, layer: f32, peak: f32) -> f32 {
    let below = below.clamp(0.0, peak);
    let layer = layer.clamp(0.0, peak);
    peak - (peak - below) * (peak - layer) / peak
}

fn adjusted(color: [f32; 3], adjustment: Adjustment, space: WorkingSpace) -> [f32; 3] {
    match adjustment {
        Adjustment::Exposure(stops) => color.map(|channel| channel * stops.exp2()),
        Adjustment::Curves(amount) => color.map(|channel| contrast_curve(channel, amount)),
        Adjustment::Saturation(amount) => saturated(color, amount, space),
    }
}

/// One brush segment over a mask: a capsule whose coverage falls off from
/// `radius * hardness` to `radius`, blended over what the mask holds.
fn draw_segment(mask: &mut [f32], padded: &Padded, segment: &Segment) {
    let radius = segment.radius.max(0.0);
    if radius <= 0.0 {
        return;
    }
    let left = segment.from[0].min(segment.to[0]) - radius;
    let right = segment.from[0].max(segment.to[0]) + radius;
    let top = segment.from[1].min(segment.to[1]) - radius;
    let bottom = segment.from[1].max(segment.to[1]) + radius;
    let clip = |value: f32, origin: i64, length: usize| {
        (value.floor() as i64 - origin).clamp(0, length as i64) as usize
    };
    let (x0, x1) = (
        clip(left, padded.x, padded.width),
        clip(right.ceil() + 1.0, padded.x, padded.width),
    );
    let (y0, y1) = (
        clip(top, padded.y, padded.height),
        clip(bottom.ceil() + 1.0, padded.y, padded.height),
    );
    let delta = [
        segment.to[0] - segment.from[0],
        segment.to[1] - segment.from[1],
    ];
    let length_squared = (delta[0] * delta[0] + delta[1] * delta[1]).max(0.0001);
    let inner = radius * segment.hardness.clamp(0.0, 1.0);
    for y in y0..y1 {
        for x in x0..x1 {
            // The pixel's center in the picture's coordinates.
            let doc = [
                (padded.x + x as i64) as f32 + 0.5,
                (padded.y + y as i64) as f32 + 0.5,
            ];
            let along = (((doc[0] - segment.from[0]) * delta[0]
                + (doc[1] - segment.from[1]) * delta[1])
                / length_squared)
                .clamp(0.0, 1.0);
            let nearest = [
                segment.from[0] + delta[0] * along,
                segment.from[1] + delta[1] * along,
            ];
            let distance = ((doc[0] - nearest[0]).powi(2) + (doc[1] - nearest[1]).powi(2)).sqrt();
            let coverage = if distance >= radius {
                0.0
            } else if distance <= inner {
                1.0
            } else {
                1.0 - smoothstep(inner, radius, distance)
            };
            if coverage <= 0.0 {
                continue;
            }
            let at = y * padded.width + x;
            mask[at] = mask[at] * (1.0 - coverage) + segment.value * coverage;
        }
    }
}

/// `color` and `layer` mixed by `cover`: `layer` itself where it covers in
/// full, so a fully covering layer keeps its exact samples.
fn cover_mix(color: [f32; 3], layer: [f32; 3], cover: f32) -> [f32; 3] {
    if cover >= 1.0 {
        layer
    } else if cover <= 0.0 {
        color
    } else {
        [
            mix(color[0], layer[0], cover),
            mix(color[1], layer[1], cover),
            mix(color[2], layer[2], cover),
        ]
    }
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::screen;

    #[test]
    fn below_white_curves_up_to_white_and_passes_the_headroom() {
        use crate::picture::{Transfer, TransferStage};
        let curve = seiza_stretch::ResolvedCurve::Mtf {
            shadows: 0.0,
            midtone: 0.25,
            highlights: 1.0,
        };
        let mut samples = [0.0, 0.25, 1.0, 1.5, 2.0];
        Transfer::of([TransferStage::BelowWhite(curve)]).apply(&mut samples);
        assert_eq!(samples[0], 0.0);
        assert!((samples[1] - 0.5).abs() < 1e-6, "{samples:?}");
        assert_eq!(&samples[2..], &[1.0, 1.5, 2.0]);
    }

    #[test]
    fn screen_works_up_to_the_peak_and_is_the_usual_screen_at_one() {
        // At a peak of 1 it is 1 - (1 - a)(1 - b).
        assert!((screen(0.5, 0.5, 1.0) - 0.75).abs() < 1e-6);
        assert_eq!(screen(1.0, 0.3, 1.0), 1.0);
        // Above SDR white, a brighter value beneath stays brighter, and a
        // layer of 0 leaves it as it is.
        let peak = 2.45;
        assert!((screen(2.0, 0.0, peak) - 2.0).abs() < 1e-6);
        assert!(screen(2.0, 1.0, peak) > screen(1.5, 1.0, peak));
        assert!(screen(2.0, 1.0, peak) > 2.0 && screen(2.0, 1.0, peak) <= peak);
        assert_eq!(screen(peak, 1.0, peak), peak);
    }
}
