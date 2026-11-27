//! The compositor on the GPU: a fast render of a [`Picture`] that follows
//! the CPU render in [`crate::render`].
//!
//! The CPU render is the reference. This one draws the same picture with
//! wgpu for a canvas that redraws as a person edits, and for hosts that want
//! speed over the last bits: its inputs and stretch tables are half floats,
//! a computed mask's smoothing is a 33-tap Gaussian clamped at the drawn
//! area's edge, and an uncovered pixel draws as black rather than `NaN`.
//! `tests/gpu.rs` holds the two renders together.
//!
//! A host owns the device, the input textures and where the picture goes:
//! it packs a picture once with [`GpuScene::new`], makes [`GpuTargets`] the
//! size of the area it draws, and calls [`GpuCompositor::draw`] with the
//! input tiles in view. The finished picture is left in the targets'
//! `finished` texture in the picture's working space; a host applies its own
//! display transform when it presents it. [`render_gpu`] does all of that
//! for a picture read from a [`crate::render::Source`].

mod display;
mod headless;

pub use display::{DISPLAY_SHADER, DisplayMapping};
pub use headless::{GpuContext, render_gpu};

use crate::picture::{
    Adjustment, Blend, FinishKind, InputRead, LayerKind, LuminanceTarget, Picture, Segment,
    Transfer,
};
use bytemuck::{Pod, Zeroable};
use seiza_stretch::{ColorStrategy, ResolvedCurve};

/// The most layers one draw combines.
pub const MAX_LAYERS: usize = 24;
/// The most Finish steps one draw applies.
pub const MAX_FINISH_STEPS: usize = 8;
/// The most painted masks one draw holds, besides the overlay.
pub const MAX_MASKS: usize = 15;
/// The most heals of single layers one draw applies.
pub const MAX_HEALS: usize = 16;
/// The most layers with heals of their own one draw applies.
pub const MAX_HEALED_LAYERS: usize = 4;
/// The most computed masks one draw holds.
pub const MAX_COMPUTED_MASKS: usize = 8;
/// The entries in one row of the stretch table.
pub const STRETCH_TABLE_WIDTH: usize = 4096;
/// The rows of the stretch table: one per transfer a layer or a computed
/// mask reads through, then three per Finish step that maps the picture
/// through curves.
const STRETCH_TABLE_ROWS: usize = MAX_LAYERS + 3 * MAX_FINISH_STEPS;
/// The mask layer that holds the overlay.
const OVERLAY_LAYER: u32 = MAX_MASKS as u32;

const MASK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
const SMOOTHED_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
const HEALED_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
/// The combined picture, before the Finish steps. Float, so a sum is not
/// clipped before a step that brings it back into range.
const COMBINED_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The local passes' measurements and means. Full floats: a window's
/// variance is the difference of two means, which half floats would lose.
const LOCAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
/// The most Tone steps one draw reads neighborhoods for; another is drawn
/// with each pixel alone and named in `left_out`.
pub const MAX_LOCAL_TONES: usize = 4;
/// The widest a local window is drawn, in drawn pixels either side, however
/// far the view is zoomed in.
const MAX_WINDOW_RADIUS: f64 = 256.0;
/// The most input tiles one draw reads.
const MAX_TILE_INSTANCES: u64 = 4096;

const GLOBALS_SHADER: &str = include_str!("globals.wgsl");
const LAYERS_SHADER: &str = include_str!("layers.wgsl");
const FINISH_SHADER: &str = include_str!("finish.wgsl");

/// A shader's whole source: the globals every pass shares, then its body.
fn shader_source(body: &str) -> String {
    format!("{GLOBALS_SHADER}\n{body}")
}

/// Where one of a picture's inputs is on the GPU: its layer in the input
/// textures, and the value a texel of 1 stands for. Input textures hold each
/// input's samples divided by its scale and kept within 0 to 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuInput {
    pub layer: usize,
    pub scale: f32,
}

/// One brush segment, as the GPU draws it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct GpuSegment {
    from: [f32; 2],
    to: [f32; 2],
    radius: f32,
    hardness: f32,
    value: f32,
    _pad: f32,
}

impl From<&Segment> for GpuSegment {
    fn from(segment: &Segment) -> Self {
        Self {
            from: segment.from,
            to: segment.to,
            radius: segment.radius,
            hardness: segment.hardness,
            value: segment.value,
            _pad: 0.0,
        }
    }
}

/// A picture packed for the GPU: its entries as the shaders read them, the
/// rows of its stretch table, and its masks' segments.
#[derive(Clone, Debug, Default)]
pub struct GpuScene {
    layers: Vec<[f32; 20]>,
    computed: Vec<([f32; 16], f32)>,
    finish: Vec<([f32; 20], Option<[i64; 2]>)>,
    /// The Finish steps whose neighborhoods the local passes measure, in
    /// order: each one's layer in the local textures is its place here.
    local: Vec<usize>,
    healed: Vec<[f32; 4]>,
    heals: Vec<([f32; 8], [i64; 2])>,
    rows: Vec<Vec<f32>>,
    finish_rows: Vec<[Vec<f32>; 3]>,
    masks: Vec<Vec<GpuSegment>>,
    /// The weights of the picture's luminance, from its working space.
    luma: [f32; 3],
    /// What the GPU cannot draw of the picture, by name.
    pub left_out: Vec<String>,
}

/// A transfer as a stretch-table row: entry `i` is the transfer of the
/// sample `(i / (width - 1))² × scale`, so a row is indexed by the square
/// root of a texel, where a stretch is steepest.
fn transfer_row(read: &InputRead, scale: f32) -> Vec<f32> {
    let last = (STRETCH_TABLE_WIDTH - 1) as f32;
    let mut samples: Vec<f32> = (0..STRETCH_TABLE_WIDTH)
        .map(|index| (index as f32 / last).powi(2) * scale)
        .collect();
    read.transfer.apply(&mut samples);
    samples
}

/// A curve as a stretch-table row of a picture's values from 0 to 1.
fn curve_row(curve: ResolvedCurve) -> Vec<f32> {
    let last = (STRETCH_TABLE_WIDTH - 1) as f32;
    (0..STRETCH_TABLE_WIDTH)
        .map(|index| curve.map((index as f32 / last).powi(2)).clamp(0.0, 1.0))
        .collect()
}

impl GpuScene {
    /// Packs `picture` for the GPU, with each of its inputs at `inputs[i]`.
    /// What the GPU cannot hold (more entries than its limits, a computed
    /// mask of more than two inputs) is left out and named in
    /// [`Self::left_out`].
    pub fn new(picture: &Picture, inputs: &[GpuInput]) -> Self {
        let mut scene = Self {
            luma: picture.space.luma_weights(),
            ..Self::default()
        };
        let input = |read: &InputRead| inputs.get(read.input).copied();
        let mask_index = |mask: Option<usize>| {
            mask.filter(|mask| *mask < picture.masks.len().min(MAX_MASKS))
                .map_or(-1.0, |mask| mask as f32)
        };
        // Each read through its transfer is a row of the stretch table. A
        // row depends only on the transfer and the input's scale, so reads
        // that share both share a row, as a layer and a mask that reads
        // its input do.
        let mut row_keys: Vec<(u32, Transfer)> = Vec::new();
        let mut row_of = |scene: &mut Self, read: &InputRead| -> Option<usize> {
            let place = input(read)?;
            let scale = place.scale.to_bits();
            if let Some(row) = row_keys
                .iter()
                .position(|(known, transfer)| *known == scale && *transfer == read.transfer)
            {
                return Some(row);
            }
            if scene.rows.len() == MAX_LAYERS {
                return None;
            }
            row_keys.push((scale, read.transfer.clone()));
            scene.rows.push(transfer_row(read, place.scale));
            Some(scene.rows.len() - 1)
        };
        let band = |read: &InputRead| -> [f32; 3] {
            let layer = input(read).map_or(-1.0, |input| input.layer as f32);
            let (blended, amount) = read
                .blended
                .and_then(|(other, amount)| Some((inputs.get(other)?.layer as f32, amount)))
                .map_or((-1.0, 0.0), |(layer, amount)| {
                    (layer, amount.clamp(0.0, 1.0))
                });
            [layer, blended, amount]
        };
        for (index, layer) in picture.layers.iter().enumerate() {
            if scene.layers.len() == MAX_LAYERS {
                scene
                    .left_out
                    .push(format!("layer {index}, past the GPU's {MAX_LAYERS}"));
                continue;
            }
            let computed = layer
                .computed
                .filter(|mask| *mask < picture.computed.len().min(MAX_COMPUTED_MASKS))
                .map_or(-1.0, |mask| mask as f32);
            let mut slot = [0.0_f32; 20];
            slot[3] = layer.opacity.clamp(0.0, 1.0);
            slot[5] = -1.0;
            slot[6] = mask_index(layer.clip);
            slot[9] = blend_index(layer.blend);
            slot[10] = -1.0;
            slot[14] = mask_index(layer.mask);
            slot[15] = if layer.own { 1.0 } else { 0.0 };
            slot[16] = computed;
            match &layer.kind {
                LayerKind::Input { read, color, curve } => {
                    let Some(row) = row_of(&mut scene, read) else {
                        // Kept in its place, hidden, so heals and masks keep
                        // their layers.
                        scene
                            .left_out
                            .push(format!("layer {index}, with no stretch row left"));
                        slot[3] = 0.0;
                        scene.layers.push(slot);
                        continue;
                    };
                    let [layer_band, blended, amount] = band(read);
                    slot[..3].copy_from_slice(color);
                    slot[4] = row as f32;
                    slot[7] = curve.clamp(-1.0, 1.0);
                    slot[8] = layer_band;
                    slot[10] = blended;
                    slot[11] = amount;
                }
                LayerKind::Solid { color } => {
                    slot[..3].copy_from_slice(color);
                    slot[12] = 12.0;
                }
                LayerKind::GroupBegin => slot[12] = 10.0,
                LayerKind::GroupEnd => slot[12] = 11.0,
                LayerKind::Adjustment(adjustment) => {
                    let (kind, amount) = match adjustment {
                        Adjustment::Exposure(stops) => (1.0, *stops),
                        Adjustment::Curves(amount) => (2.0, *amount),
                        Adjustment::Saturation(amount) => (3.0, *amount),
                    };
                    slot[12] = kind;
                    slot[13] = amount;
                }
                LayerKind::Luminance { read, target } => {
                    slot[12] = match target {
                        LuminanceTarget::Replace => 13.0,
                        LuminanceTarget::Super => 14.0,
                    };
                    slot[8] = -1.0;
                    if let Some(read) = read {
                        let Some(row) = row_of(&mut scene, read) else {
                            scene
                                .left_out
                                .push(format!("layer {index}, with no stretch row left"));
                            slot[3] = 0.0;
                            scene.layers.push(slot);
                            continue;
                        };
                        let [layer_band, blended, amount] = band(read);
                        slot[4] = row as f32;
                        slot[8] = layer_band;
                        slot[10] = blended;
                        slot[11] = amount;
                    }
                }
            }
            scene.layers.push(slot);
        }
        for (index, mask) in picture.computed.iter().take(MAX_COMPUTED_MASKS).enumerate() {
            if mask.reads.len() > 2 {
                scene
                    .left_out
                    .push(format!("computed mask {index}'s inputs past two"));
            }
            let mut slot = [0.0_f32; 16];
            for (place, read) in [0, 4]
                .into_iter()
                .zip(mask.reads.iter().take(2).map(Some).chain([None, None]))
            {
                let values = match read {
                    Some(read) => match row_of(&mut scene, read) {
                        Some(row) => {
                            let [layer_band, blended, amount] = band(read);
                            [layer_band, blended, amount, row as f32]
                        }
                        None => {
                            scene.left_out.push(format!(
                                "an input of computed mask {index}, with no stretch row left"
                            ));
                            [-1.0, -1.0, 0.0, 0.0]
                        }
                    },
                    None => [-1.0, -1.0, 0.0, 0.0],
                };
                slot[place..place + 4].copy_from_slice(&values);
            }
            slot[8] = if mask.power_of_inverse { 1.0 } else { 0.0 };
            slot[9] = if mask.inverted { 1.0 } else { 0.0 };
            slot[10] = mask.reach.clamp(0.05, 20.0);
            slot[12] = mask_index(mask.added);
            slot[13] = mask_index(mask.removed);
            scene.computed.push((slot, mask.smoothing.max(0.0)));
        }
        let mut healed_layers: Vec<usize> = Vec::new();
        for heal in &picture.heals {
            let known = healed_layers.iter().position(|layer| *layer == heal.layer);
            let Some(LayerKind::Input { read, .. }) =
                picture.layers.get(heal.layer).map(|layer| &layer.kind)
            else {
                continue;
            };
            if heal.layer >= scene.layers.len()
                || scene.heals.len() == MAX_HEALS
                || heal.mask >= picture.masks.len().min(MAX_MASKS)
                || (known.is_none() && healed_layers.len() == MAX_HEALED_LAYERS)
            {
                scene
                    .left_out
                    .push(format!("a heal of layer {}", heal.layer));
                continue;
            }
            let place = known.unwrap_or_else(|| {
                healed_layers.push(heal.layer);
                let [layer_band, blended, amount] = band(read);
                scene.healed.push([layer_band, blended, amount, 0.0]);
                healed_layers.len() - 1
            });
            scene.layers[heal.layer][5] = place as f32;
            scene.heals.push((
                [
                    heal.layer as f32,
                    if heal.before_transfer { 0.0 } else { 1.0 },
                    heal.mask as f32,
                    heal.shift,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                heal.offset,
            ));
        }
        for (index, step) in picture.finish.iter().enumerate() {
            if scene.finish.len() == MAX_FINISH_STEPS {
                scene.left_out.push(format!(
                    "finishing step {index}, past the GPU's {MAX_FINISH_STEPS}"
                ));
                continue;
            }
            let mut slot = [0.0_f32; 20];
            slot[2] = mask_index(step.mask);
            slot[3] = step.opacity.clamp(0.0, 1.0);
            let first_row = (MAX_LAYERS + 3 * scene.finish_rows.len()) as f32;
            let mut offset = None;
            match &step.kind {
                FinishKind::Curves(amount) => {
                    slot[0] = 1.0;
                    slot[1] = *amount;
                }
                FinishKind::Saturation(amount) => {
                    slot[0] = 2.0;
                    slot[1] = *amount;
                }
                FinishKind::Heal { offset: by, shift } => {
                    slot[0] = 3.0;
                    slot[8..11].copy_from_slice(shift);
                    offset = Some(*by);
                }
                FinishKind::Stretch(plan) => {
                    let curves = plan.curves();
                    let rows: [Vec<f32>; 3] = match plan.color_strategy() {
                        ColorStrategy::Unlinked => std::array::from_fn(|channel| {
                            curve_row(curves[channel.min(curves.len() - 1)])
                        }),
                        _ => std::array::from_fn(|_| curve_row(curves[0])),
                    };
                    slot[0] = 4.0;
                    slot[6] = first_row;
                    slot[7] = if plan.color_strategy() == ColorStrategy::LuminancePreserving {
                        1.0
                    } else {
                        0.0
                    };
                    scene.finish_rows.push(rows);
                }
                FinishKind::Tone(tone) => {
                    let held = tone.held();
                    slot[0] = 5.0;
                    slot[1] = held.exposure;
                    slot[4..8].copy_from_slice(&[
                        held.contrast,
                        held.highlights,
                        held.shadows,
                        held.whites,
                    ]);
                    slot[8..12].copy_from_slice(&[
                        held.blacks,
                        held.texture,
                        held.clarity,
                        held.dehaze,
                    ]);
                    slot[12] = held.vibrance;
                    slot[13] = held.saturation;
                    slot[15] = tone.detail_edge();
                    slot[16] = tone.peak();
                    // Up to MAX_LOCAL_TONES steps a draw reads
                    // neighborhoods for, each with its own layer, counted
                    // from 1; another is drawn with each pixel alone.
                    if tone.is_local() {
                        if scene.local.len() < MAX_LOCAL_TONES {
                            scene.local.push(scene.finish.len());
                            slot[14] = scene.local.len() as f32;
                        } else {
                            // Drawn with each pixel alone, where a veil is the
                            // pixel's own darkest channel: Dehaze is left out.
                            slot[11] = 0.0;
                            scene.left_out.push(format!(
                                "the neighborhoods of finishing step {index}, past the GPU's {MAX_LOCAL_TONES}"
                            ));
                        }
                    }
                }
            }
            scene.finish.push((slot, offset));
        }
        scene.masks = picture
            .masks
            .iter()
            .take(MAX_MASKS)
            .map(|segments| segments.iter().map(GpuSegment::from).collect())
            .collect();
        scene
    }
}

fn blend_index(blend: Blend) -> f32 {
    match blend {
        Blend::Normal => 0.0,
        Blend::Screen => 1.0,
        Blend::Add => 2.0,
        Blend::Multiply => 3.0,
        Blend::Lighten => 4.0,
        Blend::Darken => 5.0,
        Blend::Luminance => 6.0,
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    view: [f32; 4],
    screen: [f32; 4],
    counts: [f32; 4],
    layers: [[f32; 20]; MAX_LAYERS],
    finish: [[f32; 20]; MAX_FINISH_STEPS],
    healed: [[f32; 4]; MAX_HEALED_LAYERS],
    heals: [[f32; 8]; MAX_HEALS],
    computed: [[f32; 16]; MAX_COMPUTED_MASKS],
    local: [f32; 4],
    area: [f32; 4],
    luma: [f32; 4],
}

/// Where the picture sits in the area drawn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuView {
    /// The picture's pixel at the center of the area.
    pub center: [f64; 2],
    /// Drawn pixels per picture pixel.
    pub zoom: f64,
}

/// One input tile to draw: the bind group of its input textures, made with
/// [`GpuCompositor::tile_layout`], the rectangle it covers in picture pixels,
/// and the part of its textures that rectangle takes, from 0 to 1.
pub struct TileDraw<'a> {
    pub inputs: &'a wgpu::BindGroup,
    pub rect: [f32; 4],
    pub uv_rect: [f32; 4],
}

/// The part of the targets one draw fills: where the picture is laid out
/// (x, y, width and height in pixels; the view's center is at its middle),
/// and the part it draws to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrawArea {
    pub viewport: [u32; 4],
    pub scissor: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TileInstance {
    rect: [f32; 4],
    uv_rect: [f32; 4],
}

/// A growable buffer of stroke segments.
struct StrokeBuffer {
    buffer: wgpu::Buffer,
    capacity: usize,
}

impl StrokeBuffer {
    fn new(device: &wgpu::Device, capacity: usize) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stroke instances"),
            size: (capacity.max(1) * size_of::<GpuSegment>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { buffer, capacity }
    }

    fn reserve(&mut self, device: &wgpu::Device, count: usize) {
        if count > self.capacity {
            *self = Self::new(device, count.next_power_of_two());
        }
    }
}

/// The textures a draw works in, the size of the area drawn.
pub struct GpuTargets {
    width: u32,
    height: u32,
    combined: wgpu::TextureView,
    mask_layers: Vec<wgpu::TextureView>,
    finish_inputs: wgpu::BindGroup,
    layer_masks: wgpu::BindGroup,
    healed_layers: Vec<wgpu::TextureView>,
    smoothed: [Vec<wgpu::TextureView>; 2],
    smoothed_sources: [wgpu::BindGroup; 2],
    /// The finished picture, in the picture's working space.
    pub finished: wgpu::Texture,
    finished_view: wgpu::TextureView,
    /// The local passes' textures, made the first time a draw has a Tone
    /// step that reads neighborhoods, and again when a draw has more such
    /// steps than they have layers for.
    local: std::sync::Mutex<Option<std::sync::Arc<LocalTargets>>>,
}

/// What the local passes draw into, the size of the targets. One step's
/// passes draw its measurements, a mean across and each mean into single
/// textures, which are copied into the step's layer of three arrays: the
/// measurements, the local means and the detail means. A later step's
/// measurements and Finish read the arrays.
struct LocalTargets {
    /// The layers each array holds.
    layers: u32,
    measured: wgpu::Texture,
    local_means: wgpu::Texture,
    detail_means: wgpu::Texture,
    scratch: wgpu::Texture,
    scratch_view: wgpu::TextureView,
    across: wgpu::TextureView,
    means: wgpu::Texture,
    means_view: wgpu::TextureView,
    from_scratch: wgpu::BindGroup,
    from_across: wgpu::BindGroup,
    finish_inputs: wgpu::BindGroup,
}

impl GpuTargets {
    /// The targets' width and height.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// A view of the finished picture.
    pub fn finished_view(&self) -> &wgpu::TextureView {
        &self.finished_view
    }
}

/// The passes, pipelines and buffers that draw a [`GpuScene`].
pub struct GpuCompositor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    finished_format: wgpu::TextureFormat,
    globals: wgpu::Buffer,
    common: wgpu::BindGroup,
    stretch_table: wgpu::Texture,
    tile_layout: wgpu::BindGroupLayout,
    masks_layout: wgpu::BindGroupLayout,
    finish_layout: wgpu::BindGroupLayout,
    /// What a local window pass reads: one texture of full floats.
    window_layout: wgpu::BindGroupLayout,
    /// What Finish reads of the local passes.
    local_layout: wgpu::BindGroupLayout,
    /// Finish's local inputs when a draw has none: one texel of zeros.
    no_local: wgpu::BindGroup,
    local_measure_pipeline: wgpu::RenderPipeline,
    /// The local window across and down, then the detail window across
    /// and down.
    window_pipelines: [wgpu::RenderPipeline; 4],
    tile_pipeline: wgpu::RenderPipeline,
    healed_pipeline: wgpu::RenderPipeline,
    smoothed_pipeline: wgpu::RenderPipeline,
    blur_pipelines: [wgpu::RenderPipeline; 2],
    stroke_pipeline: wgpu::RenderPipeline,
    finish_pipeline: wgpu::RenderPipeline,
    tile_instances: wgpu::Buffer,
    masks: StrokeBuffer,
    mask_ranges: Vec<(u32, u32)>,
    overlay: StrokeBuffer,
    tables_version: Option<u64>,
    masks_version: Option<u64>,
}

impl GpuCompositor {
    /// The compositor on `device`, leaving finished pictures in textures of
    /// `finished_format` (a float format, such as `Rgba16Float`, keeps the
    /// working space's range).
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        finished_format: wgpu::TextureFormat,
    ) -> Self {
        let module = |label: &str, body: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(shader_source(body).into()),
            })
        };
        let layers_shader = module("compose layers", LAYERS_SHADER);
        let finish_shader = module("compose finish", FINISH_SHADER);
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compose globals"),
            size: size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let texture_entry =
            |binding: u32, view_dimension: wgpu::TextureViewDimension| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension,
                    multisampled: false,
                },
                count: None,
            };
        let common_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compose globals and sampler"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                texture_entry(2, wgpu::TextureViewDimension::D2),
            ],
        });
        let tile_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compose input tile"),
            entries: &[texture_entry(0, wgpu::TextureViewDimension::D2Array)],
        });
        let masks_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compose layer masks"),
            entries: &[
                texture_entry(0, wgpu::TextureViewDimension::D2Array),
                texture_entry(1, wgpu::TextureViewDimension::D2Array),
                texture_entry(2, wgpu::TextureViewDimension::D2Array),
            ],
        });
        let finish_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compose finish inputs"),
            entries: &[
                texture_entry(0, wgpu::TextureViewDimension::D2),
                texture_entry(1, wgpu::TextureViewDimension::D2Array),
            ],
        });
        // Full floats cannot be filtered on every GPU; the local passes only
        // load them.
        let unfiltered = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let window_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compose local window source"),
            entries: &[unfiltered(0)],
        });
        let layered = |binding: u32| wgpu::BindGroupLayoutEntry {
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            ..unfiltered(binding)
        };
        let local_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compose local finish inputs"),
            entries: &[layered(1), layered(2), layered(3)],
        });
        const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
            wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4];
        let pipeline = |label: &str,
                        shader: &wgpu::ShaderModule,
                        second: Option<&wgpu::BindGroupLayout>,
                        third: Option<&wgpu::BindGroupLayout>,
                        entries: (&str, &str),
                        instanced: bool,
                        targets: &[Option<wgpu::ColorTargetState>]| {
            let groups = [Some(&common_layout), second, third];
            let count = if third.is_some() {
                3
            } else if second.is_some() {
                2
            } else {
                1
            };
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &groups[..count],
                ..Default::default()
            });
            let instances = [Some(wgpu::VertexBufferLayout {
                array_stride: 32,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &INSTANCE_ATTRIBUTES,
            })];
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some(entries.0),
                    buffers: if instanced { &instances } else { &[] },
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some(entries.1),
                    targets,
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let tile_pipeline = pipeline(
            "compose layers",
            &layers_shader,
            Some(&tile_layout),
            Some(&masks_layout),
            ("tile_vertex", "tile_fragment"),
            true,
            &[Some(COMBINED_FORMAT.into())],
        );
        let healed_pipeline = pipeline(
            "compose healed layers",
            &layers_shader,
            Some(&tile_layout),
            None,
            ("tile_vertex", "healed_fragment"),
            true,
            &std::array::from_fn::<_, MAX_HEALED_LAYERS, _>(|_| Some(HEALED_FORMAT.into())),
        );
        let smoothed_targets =
            std::array::from_fn::<_, MAX_COMPUTED_MASKS, _>(|_| Some(SMOOTHED_FORMAT.into()));
        let smoothed_pipeline = pipeline(
            "compose computed masks",
            &layers_shader,
            Some(&tile_layout),
            None,
            ("tile_vertex", "computed_fragment"),
            true,
            &smoothed_targets,
        );
        // A blur reads the masks through the tile binding: both are arrays of
        // single-channel images.
        let blur_pipelines = ["blur_across_fragment", "blur_down_fragment"].map(|fragment| {
            pipeline(
                "compose mask blur",
                &layers_shader,
                Some(&tile_layout),
                None,
                ("whole_vertex", fragment),
                false,
                &smoothed_targets,
            )
        });
        let stroke_pipeline = pipeline(
            "compose strokes",
            &layers_shader,
            None,
            None,
            ("stroke_vertex", "stroke_fragment"),
            true,
            &[Some(wgpu::ColorTargetState {
                format: MASK_FORMAT,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::RED,
            })],
        );
        let finish_pipeline = pipeline(
            "compose finish",
            &finish_shader,
            Some(&finish_layout),
            Some(&local_layout),
            ("finish_vertex", "finish_fragment"),
            false,
            &[Some(finished_format.into())],
        );
        // The measurements run the Finish steps before the local one, whose
        // code can read the local inputs: they are bound, empty.
        let local_measure_pipeline = pipeline(
            "compose local measurements",
            &finish_shader,
            Some(&finish_layout),
            Some(&local_layout),
            ("finish_vertex", "local_measure_fragment"),
            false,
            &[Some(LOCAL_FORMAT.into())],
        );
        let window_pipelines = [
            "local_across_fragment",
            "local_down_fragment",
            "detail_across_fragment",
            "detail_down_fragment",
        ]
        .map(|fragment| {
            pipeline(
                "compose local window",
                &finish_shader,
                None,
                Some(&window_layout),
                ("finish_vertex", fragment),
                false,
                &[Some(LOCAL_FORMAT.into())],
            )
        });
        let no_local = {
            let zero = device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("compose no local inputs"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: LOCAL_FORMAT,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    ..Default::default()
                });
            local_finish_inputs(device, &local_layout, [&zero, &zero, &zero])
        };
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("compose linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let stretch_table = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("compose stretch table"),
            size: wgpu::Extent3d {
                width: STRETCH_TABLE_WIDTH as u32,
                height: STRETCH_TABLE_ROWS as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let stretch_view = stretch_table.create_view(&wgpu::TextureViewDescriptor::default());
        let common = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compose globals and sampler"),
            layout: &common_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: globals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&stretch_view),
                },
            ],
        });
        let tile_instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compose tile instances"),
            size: MAX_TILE_INSTANCES * size_of::<TileInstance>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            masks: StrokeBuffer::new(device, 4096),
            overlay: StrokeBuffer::new(device, 4096),
            device: device.clone(),
            queue: queue.clone(),
            finished_format,
            globals,
            common,
            stretch_table,
            tile_layout,
            masks_layout,
            finish_layout,
            window_layout,
            local_layout,
            no_local,
            local_measure_pipeline,
            window_pipelines,
            tile_pipeline,
            healed_pipeline,
            smoothed_pipeline,
            blur_pipelines,
            stroke_pipeline,
            finish_pipeline,
            tile_instances,
            mask_ranges: Vec::new(),
            tables_version: None,
            masks_version: None,
        }
    }

    /// The layout of an input tile's bind group: one `texture_2d_array` of
    /// filterable floats, one layer per input, at binding 0.
    pub fn tile_layout(&self) -> &wgpu::BindGroupLayout {
        &self.tile_layout
    }

    /// Targets for drawing an area of `width` by `height` pixels.
    pub fn targets(&self, width: u32, height: u32) -> GpuTargets {
        let device = &self.device;
        let (width, height) = (width.max(1), height.max(1));
        let texture = |label: &str, format: wgpu::TextureFormat, layers: u32| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: layers,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let layers_of = |texture: &wgpu::Texture, count: u32| -> Vec<wgpu::TextureView> {
            (0..count)
                .map(|layer| {
                    texture.create_view(&wgpu::TextureViewDescriptor {
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        base_array_layer: layer,
                        array_layer_count: Some(1),
                        ..Default::default()
                    })
                })
                .collect()
        };
        let array_of = |texture: &wgpu::Texture| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        };
        let combined = texture("compose combined", COMBINED_FORMAT, 1)
            .create_view(&wgpu::TextureViewDescriptor::default());
        let masks = texture("compose masks", MASK_FORMAT, MAX_MASKS as u32 + 1);
        let mask_layers = layers_of(&masks, MAX_MASKS as u32 + 1);
        let all_masks = array_of(&masks);
        let finish_inputs = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compose finish inputs"),
            layout: &self.finish_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&combined),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&all_masks),
                },
            ],
        });
        let healed = texture(
            "compose healed layers",
            HEALED_FORMAT,
            MAX_HEALED_LAYERS as u32,
        );
        let healed_layers = layers_of(&healed, MAX_HEALED_LAYERS as u32);
        let all_healed = array_of(&healed);
        let smoothed_sets = ["compose smoothed masks", "compose masks half blurred"].map(|label| {
            let texture = texture(label, SMOOTHED_FORMAT, MAX_COMPUTED_MASKS as u32);
            (
                layers_of(&texture, MAX_COMPUTED_MASKS as u32),
                array_of(&texture),
            )
        });
        let smoothed_sources = [0, 1].map(|set: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("compose masks to blur"),
                layout: &self.tile_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&smoothed_sets[set].1),
                }],
            })
        });
        let layer_masks = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compose layer masks"),
            layout: &self.masks_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&all_masks),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&all_healed),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&smoothed_sets[0].1),
                },
            ],
        });
        let finished = texture("compose finished", self.finished_format, 1);
        let finished_view = finished.create_view(&wgpu::TextureViewDescriptor::default());
        let [(first, _), (second, _)] = smoothed_sets;
        GpuTargets {
            width,
            height,
            combined,
            mask_layers,
            finish_inputs,
            layer_masks,
            healed_layers,
            smoothed: [first, second],
            smoothed_sources,
            finished,
            finished_view,
            local: std::sync::Mutex::new(None),
        }
    }

    /// The local passes' textures for targets of `width` by `height`, with
    /// `layers` layers in each array.
    fn local_targets(&self, width: u32, height: u32, layers: u32) -> LocalTargets {
        let texture = |label: &str, layers: u32, usage: wgpu::TextureUsages| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: layers,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: LOCAL_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | usage,
                view_formats: &[],
            })
        };
        let single = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC;
        let array = |label: &str| texture(label, layers, wgpu::TextureUsages::COPY_DST);
        let array_view = |texture: &wgpu::Texture| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        };
        let measured = array("compose local measurements");
        let local_means = array("compose local means");
        let detail_means = array("compose detail means");
        let scratch = texture("compose one step's measurements", 1, single);
        let scratch_view = scratch.create_view(&wgpu::TextureViewDescriptor::default());
        let across = texture(
            "compose local means across",
            1,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
        .create_view(&wgpu::TextureViewDescriptor::default());
        let means = texture("compose one step's means", 1, single);
        let means_view = means.create_view(&wgpu::TextureViewDescriptor::default());
        let source = |view: &wgpu::TextureView| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("compose local window source"),
                layout: &self.window_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                }],
            })
        };
        let from_scratch = source(&scratch_view);
        let from_across = source(&across);
        let finish_inputs = local_finish_inputs(
            &self.device,
            &self.local_layout,
            [
                &array_view(&measured),
                &array_view(&local_means),
                &array_view(&detail_means),
            ],
        );
        LocalTargets {
            layers,
            measured,
            local_means,
            detail_means,
            scratch,
            scratch_view,
            across,
            means,
            means_view,
            from_scratch,
            from_across,
            finish_inputs,
        }
    }

    /// Draws `scene` into `area` of `targets` from `tiles`, and submits the
    /// work. `version` names the scene's tables and masks: a draw of the same
    /// version uploads them only once; `None` always uploads. `overlay` is a
    /// stroke shown over the finished picture at `overlay_strength`. With
    /// `first`, the draw clears the finished picture before drawing.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        scene: &GpuScene,
        version: Option<u64>,
        targets: &GpuTargets,
        view: GpuView,
        area: DrawArea,
        tiles: &[TileDraw<'_>],
        overlay: &[Segment],
        overlay_strength: f32,
        first: bool,
    ) {
        let rect = within(area.viewport, targets.size());
        let scissor = within(area.scissor, targets.size());
        let area = picture_area(tiles, view, rect, scissor);
        self.write_globals(scene, targets, view, (rect, area), overlay_strength);
        if version.is_none() || self.tables_version != version {
            self.upload_tables(scene);
            self.tables_version = version;
        }
        if version.is_none() || self.masks_version != version {
            self.upload_masks(scene);
            self.masks_version = version;
        }
        let overlay: Vec<GpuSegment> = overlay.iter().map(GpuSegment::from).collect();
        self.overlay.reserve(&self.device, overlay.len());
        self.queue
            .write_buffer(&self.overlay.buffer, 0, bytemuck::cast_slice(&overlay));
        let instances: Vec<TileInstance> = tiles
            .iter()
            .take(MAX_TILE_INSTANCES as usize)
            .map(|tile| TileInstance {
                rect: tile.rect,
                uv_rect: tile.uv_rect,
            })
            .collect();
        self.queue
            .write_buffer(&self.tile_instances, 0, bytemuck::cast_slice(&instances));
        let groups: Vec<&wgpu::BindGroup> = tiles
            .iter()
            .take(MAX_TILE_INSTANCES as usize)
            .map(|tile| tile.inputs)
            .collect();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.passes(
            &mut encoder,
            scene,
            targets,
            &groups,
            overlay.len() as u32,
            (rect, scissor),
            first,
        );
        self.queue.submit([encoder.finish()]);
    }

    fn write_globals(
        &self,
        scene: &GpuScene,
        targets: &GpuTargets,
        view: GpuView,
        (rect, area): ([u32; 4], [u32; 4]),
        overlay_strength: f32,
    ) {
        let window =
            |radius: usize| (radius as f64 * view.zoom).round().min(MAX_WINDOW_RADIUS) as f32;
        let (width, height) = (f64::from(rect[2]), f64::from(rect[3]));
        let scale = [2.0 * view.zoom / width, -2.0 * view.zoom / height];
        let mut globals = Globals {
            view: [
                scale[0] as f32,
                scale[1] as f32,
                (-view.center[0] * scale[0]) as f32,
                (-view.center[1] * scale[1]) as f32,
            ],
            // Passes read masks and the combined image by target pixel, so
            // this is the targets' size, not the area's.
            screen: [
                targets.width as f32,
                targets.height as f32,
                overlay_strength,
                OVERLAY_LAYER as f32,
            ],
            counts: [
                scene.layers.len() as f32,
                scene.finish.len() as f32,
                scene.heals.len() as f32,
                0.0,
            ],
            layers: [[0.0; 20]; MAX_LAYERS],
            finish: [[0.0; 20]; MAX_FINISH_STEPS],
            healed: [[-1.0, -1.0, 0.0, 0.0]; MAX_HEALED_LAYERS],
            heals: [[0.0; 8]; MAX_HEALS],
            computed: [[0.0; 16]; MAX_COMPUTED_MASKS],
            local: [
                scene.local.first().map_or(-1.0, |step| *step as f32),
                window(crate::tone::LOCAL_RADIUS),
                window(crate::tone::DETAIL_RADIUS),
                0.0,
            ],
            area: area.map(|value| value as f32),
            luma: [scene.luma[0], scene.luma[1], scene.luma[2], 0.0],
        };
        for (slot, layer) in globals.layers.iter_mut().zip(&scene.layers) {
            *slot = *layer;
        }
        for (slot, (mask, smoothing)) in globals.computed.iter_mut().zip(&scene.computed) {
            *slot = *mask;
            // The blur's reach in drawn pixels.
            slot[14] = (f64::from(*smoothing) * view.zoom) as f32;
        }
        for (slot, healed) in globals.healed.iter_mut().zip(&scene.healed) {
            *slot = *healed;
        }
        let pixels = |offset: [i64; 2]| {
            [
                (offset[0] as f64 * view.zoom).round() as f32,
                (offset[1] as f64 * view.zoom).round() as f32,
            ]
        };
        for (slot, (heal, offset)) in globals.heals.iter_mut().zip(&scene.heals) {
            *slot = *heal;
            slot[4..6].copy_from_slice(&pixels(*offset));
        }
        for (slot, (step, offset)) in globals.finish.iter_mut().zip(&scene.finish) {
            *slot = *step;
            // Zoomed out, a detail window holds pixels the canvas averaged,
            // whose noise varies less by the square of the zoom: Texture's
            // edge follows, so it treats the same noise as noise.
            if slot[0] == 5.0 && view.zoom < 1.0 {
                slot[15] *= (view.zoom * view.zoom) as f32;
            }
            if let Some(offset) = offset {
                slot[4..6].copy_from_slice(&pixels(*offset));
            }
        }
        self.queue
            .write_buffer(&self.globals, 0, bytemuck::bytes_of(&globals));
    }

    fn upload_tables(&self, scene: &GpuScene) {
        let mut bits = vec![0u16; STRETCH_TABLE_WIDTH * STRETCH_TABLE_ROWS];
        let finish_rows = scene
            .finish_rows
            .iter()
            .take(MAX_FINISH_STEPS)
            .enumerate()
            .flat_map(|(table, curves)| {
                curves
                    .iter()
                    .enumerate()
                    .map(move |(channel, curve)| (MAX_LAYERS + 3 * table + channel, curve))
            });
        let rows = scene
            .rows
            .iter()
            .take(MAX_LAYERS)
            .enumerate()
            .chain(finish_rows);
        for (row, table) in rows {
            let target = &mut bits[row * STRETCH_TABLE_WIDTH..(row + 1) * STRETCH_TABLE_WIDTH];
            for (bits, value) in target.iter_mut().zip(table) {
                *bits = half::f16::from_f32(*value).to_bits();
            }
        }
        self.queue.write_texture(
            self.stretch_table.as_image_copy(),
            bytemuck::cast_slice(&bits),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(STRETCH_TABLE_WIDTH as u32 * 2),
                rows_per_image: None,
            },
            self.stretch_table.size(),
        );
    }

    fn upload_masks(&mut self, scene: &GpuScene) {
        let all: Vec<GpuSegment> = scene.masks.iter().flatten().copied().collect();
        self.masks.reserve(&self.device, all.len());
        self.queue
            .write_buffer(&self.masks.buffer, 0, bytemuck::cast_slice(&all));
        let mut first = 0;
        self.mask_ranges = scene
            .masks
            .iter()
            .map(|mask| {
                let range = (first, mask.len() as u32);
                first += mask.len() as u32;
                range
            })
            .collect();
    }

    /// Every pass of one draw: the masks, the layers, then the Finish steps
    /// into the finished picture.
    #[allow(clippy::too_many_arguments)]
    fn passes(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene: &GpuScene,
        targets: &GpuTargets,
        tiles: &[&wgpu::BindGroup],
        overlay: u32,
        (rect, scissor): ([u32; 4], [u32; 4]),
        first: bool,
    ) {
        let within = |pass: &mut wgpu::RenderPass| {
            pass.set_viewport(
                rect[0] as f32,
                rect[1] as f32,
                rect[2] as f32,
                rect[3] as f32,
                0.0,
                1.0,
            );
            pass.set_scissor_rect(scissor[0], scissor[1], scissor[2], scissor[3]);
        };
        let clear = |view| wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        };
        // Each mask into its own layer, and the overlay into the last.
        let strokes = self
            .mask_ranges
            .iter()
            .enumerate()
            .map(|(layer, range)| (layer, &self.masks.buffer, *range))
            .chain([(OVERLAY_LAYER as usize, &self.overlay.buffer, (0, overlay))]);
        for (layer, buffer, (first_segment, count)) in strokes {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("compose mask"),
                color_attachments: &[Some(clear(&targets.mask_layers[layer]))],
                ..Default::default()
            });
            within(&mut pass);
            if count > 0 {
                pass.set_pipeline(&self.stroke_pipeline);
                pass.set_bind_group(0, &self.common, &[]);
                pass.set_vertex_buffer(0, buffer.slice(..));
                pass.draw(0..4, first_segment..first_segment + count);
            }
        }
        let draw_tiles = |pass: &mut wgpu::RenderPass| {
            pass.set_vertex_buffer(0, self.tile_instances.slice(..));
            for (index, group) in tiles.iter().enumerate() {
                pass.set_bind_group(1, *group, &[]);
                pass.draw(0..4, index as u32..index as u32 + 1);
            }
        };
        if !scene.healed.is_empty() {
            let attachments: Vec<_> = targets
                .healed_layers
                .iter()
                .map(|view| Some(clear(view)))
                .collect();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("compose healed layers"),
                color_attachments: &attachments,
                ..Default::default()
            });
            within(&mut pass);
            pass.set_pipeline(&self.healed_pipeline);
            pass.set_bind_group(0, &self.common, &[]);
            draw_tiles(&mut pass);
        }
        if scene.computed.iter().any(|(_, smoothing)| *smoothing > 0.0) {
            {
                let attachments: Vec<_> = targets.smoothed[0]
                    .iter()
                    .map(|view| Some(clear(view)))
                    .collect();
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("compose computed masks"),
                    color_attachments: &attachments,
                    ..Default::default()
                });
                within(&mut pass);
                pass.set_pipeline(&self.smoothed_pipeline);
                pass.set_bind_group(0, &self.common, &[]);
                draw_tiles(&mut pass);
            }
            for (blur, pipeline) in self.blur_pipelines.iter().enumerate() {
                let attachments: Vec<_> = targets.smoothed[1 - blur]
                    .iter()
                    .map(|view| Some(clear(view)))
                    .collect();
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("compose mask blur"),
                    color_attachments: &attachments,
                    ..Default::default()
                });
                within(&mut pass);
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &self.common, &[]);
                pass.set_bind_group(1, &targets.smoothed_sources[blur], &[]);
                pass.draw(0..3, 0..1);
            }
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("compose layers"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.combined,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            within(&mut pass);
            pass.set_pipeline(&self.tile_pipeline);
            pass.set_bind_group(0, &self.common, &[]);
            pass.set_bind_group(2, &targets.layer_masks, &[]);
            draw_tiles(&mut pass);
        }
        // The local passes: the measurements of the picture before the
        // local Tone step, then their means over the local window and over
        // the detail window, each across then down.
        let local = (!scene.local.is_empty()).then(|| {
            let mut kept = targets.local.lock().expect("the local targets");
            let needed = scene.local.len() as u32;
            match kept.as_ref() {
                Some(local) if local.layers >= needed => local.clone(),
                _ => {
                    let made = std::sync::Arc::new(self.local_targets(
                        targets.width,
                        targets.height,
                        needed,
                    ));
                    *kept = Some(made.clone());
                    made
                }
            }
        });
        if let Some(local) = &local {
            for (layer, step) in scene.local.iter().enumerate() {
                // The measure pass reads which step it measures before from
                // the globals. A write reaches the GPU at the next submit, so
                // the passes so far go first.
                if layer > 0 {
                    let done = std::mem::replace(
                        encoder,
                        self.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default()),
                    );
                    self.queue.submit([done.finish()]);
                }
                self.queue.write_buffer(
                    &self.globals,
                    std::mem::offset_of!(Globals, local) as u64,
                    bytemuck::bytes_of(&(*step as f32)),
                );
                let pass_into =
                    |encoder: &mut wgpu::CommandEncoder,
                     label: &str,
                     target: &wgpu::TextureView,
                     pipeline: &wgpu::RenderPipeline,
                     groups: &[(u32, &wgpu::BindGroup)]| {
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some(label),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: target,
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            ..Default::default()
                        });
                        within(&mut pass);
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, &self.common, &[]);
                        for (index, group) in groups {
                            pass.set_bind_group(*index, *group, &[]);
                        }
                        pass.draw(0..3, 0..1);
                    };
                // The measurements run the steps before this one, the local
                // ones among them reading the layers already filled.
                pass_into(
                    encoder,
                    "compose local measurements",
                    &local.scratch_view,
                    &self.local_measure_pipeline,
                    &[(1, &targets.finish_inputs), (2, &local.finish_inputs)],
                );
                let [local_across, local_down, detail_across, detail_down] = &self.window_pipelines;
                for (across, down, array) in [
                    (local_across, local_down, &local.local_means),
                    (detail_across, detail_down, &local.detail_means),
                ] {
                    pass_into(
                        encoder,
                        "compose local window",
                        &local.across,
                        across,
                        &[(2, &local.from_scratch)],
                    );
                    pass_into(
                        encoder,
                        "compose local window",
                        &local.means_view,
                        down,
                        &[(2, &local.from_across)],
                    );
                    copy_into_layer(encoder, &local.means, array, layer as u32);
                }
                copy_into_layer(encoder, &local.scratch, &local.measured, layer as u32);
            }
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("compose finish"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &targets.finished_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: if first {
                        wgpu::LoadOp::Clear(wgpu::Color::BLACK)
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        within(&mut pass);
        pass.set_pipeline(&self.finish_pipeline);
        pass.set_bind_group(0, &self.common, &[]);
        pass.set_bind_group(1, &targets.finish_inputs, &[]);
        pass.set_bind_group(
            2,
            local
                .as_ref()
                .map_or(&self.no_local, |local| &local.finish_inputs),
            &[],
        );
        pass.draw(0..3, 0..1);
    }
}

/// The part of the targets a local window reads: the scissor, within the
/// viewport, within the picture the tiles draw, so a window never counts
/// pixels outside the picture or pixels this draw does not fill.
fn picture_area(
    tiles: &[TileDraw<'_>],
    view: GpuView,
    rect: [u32; 4],
    scissor: [u32; 4],
) -> [u32; 4] {
    let mut low = [scissor[0].max(rect[0]), scissor[1].max(rect[1])];
    let mut high = [
        (scissor[0] + scissor[2]).min(rect[0] + rect[2]),
        (scissor[1] + scissor[3]).min(rect[1] + rect[3]),
    ];
    if let Some(drawn) = tiles.iter().map(|tile| tile.rect).reduce(|a, b| {
        [
            a[0].min(b[0]),
            a[1].min(b[1]),
            a[2].max(b[2]),
            a[3].max(b[3]),
        ]
    }) {
        // Picture pixels to target pixels: the view's center is at the
        // viewport's middle.
        let to_target = |value: f32, axis: usize| {
            f64::from(rect[axis])
                + f64::from(rect[axis + 2]) / 2.0
                + (f64::from(value) - view.center[axis]) * view.zoom
        };
        for axis in 0..2 {
            let start = to_target(drawn[axis], axis).floor().max(0.0) as u32;
            let end = to_target(drawn[axis + 2], axis).ceil().max(0.0) as u32;
            low[axis] = low[axis].max(start);
            high[axis] = high[axis].min(end);
        }
    }
    [
        low[0],
        low[1],
        high[0].saturating_sub(low[0]).max(1),
        high[1].saturating_sub(low[1]).max(1),
    ]
}

/// Copies a single texture into one layer of an array of its size.
fn copy_into_layer(
    encoder: &mut wgpu::CommandEncoder,
    from: &wgpu::Texture,
    into: &wgpu::Texture,
    layer: u32,
) {
    encoder.copy_texture_to_texture(
        from.as_image_copy(),
        wgpu::TexelCopyTextureInfo {
            texture: into,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: layer,
            },
            aspect: wgpu::TextureAspect::All,
        },
        from.size(),
    );
}

/// Finish's bind group of the local passes' measurements and means.
fn local_finish_inputs(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    views: [&wgpu::TextureView; 3],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = views
        .iter()
        .enumerate()
        .map(|(index, view)| wgpu::BindGroupEntry {
            binding: index as u32 + 1,
            resource: wgpu::BindingResource::TextureView(view),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compose local finish inputs"),
        layout,
        entries: &entries,
    })
}

/// A rectangle kept inside targets of `size`.
fn within(rect: [u32; 4], size: (u32, u32)) -> [u32; 4] {
    let x = rect[0].min(size.0.saturating_sub(1));
    let y = rect[1].min(size.1.saturating_sub(1));
    [
        x,
        y,
        rect[2].clamp(1, size.0 - x),
        rect[3].clamp(1, size.1 - y),
    ]
}

#[cfg(test)]
mod tests {
    use super::{FINISH_SHADER, LAYERS_SHADER, shader_source};

    /// Every shader parses and validates, so a mistake in one is found by
    /// the tests rather than when a canvas starts.
    #[test]
    fn every_shader_compiles() {
        for (label, body) in [("layers", LAYERS_SHADER), ("finish", FINISH_SHADER)] {
            let source = shader_source(body);
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|error| panic!("{label}: {}", error.emit_to_string(&source)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|error| panic!("{label}: {error:?}"));
        }
    }
}
