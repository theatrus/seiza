// What every pass of the canvas shares.

// One layer of the Combine stage.
struct Layer {
    // rgb: the layer's color. a: its opacity, 0 when the layer is hidden.
    color: vec4f,
    // x: the row of the stretch table that holds this layer's stretch.
    // y: the layer's place among the healed layers, or -1 when it has no
    // heals of its own. z: the mask layer of the layer it is clipped to, or
    // -1. w: the layer's own curve amount.
    stretch: vec4f,
    // x: the band the layer shows, y: the blend mode. z: a second band
    // blended into the first before the stretch, or -1. w: how much of it.
    select: vec4f,
    // x: 0 for a layer that shows a band; 1 exposure, 2 curves, 3 saturation
    // for an adjustment of the layers beneath; 10 for the start of a group
    // and 11 for its end; 12 for a layer of its color everywhere. y: the
    // adjustment's amount.
    // z: the mask layer that limits the layer, or -1 for everywhere.
    // w: 1 for an adjustment of the band layer before it alone.
    adjust: vec4f,
    // x: the computed mask that limits the layer, or -1 for none.
    more: vec4f,
}

// A layered stretch, as `seiza_stretch::ResolvedLayered` with its maps and
// HDR tail; `layered_constants` in Rust packs it.
struct LayeredStep {
    // The median anchor curve: black, median, target; then the background
    // median of luminance.
    curve: vec4f,
    // x: luminance noise. y: contrast recovery. z: the curve's slope at the
    // background median. w: 1 when the maps hold a core compression base.
    detail: vec4f,
    // The core compression: x the log of its threshold, y how much it takes
    // out, z the floor under the log, w the knee's width.
    core: vec4f,
    // The HDR tail: x 1 when there is one, y the curve's black, z the input
    // above it where the tail starts, w the log of the light there.
    tail: vec4f,
    // The tail: x the curve's slope at the knee, y its own, z how quickly it
    // turns; w the brightest value the stretch gives.
    turn: vec4f,
    // x: the maps' layer. y: 1 when the maps hold a large-scale layer.
    // zw: the maps' width and height.
    maps: vec4f,
    // The guides, each a layer of the input tiles (-1 for none) and the
    // value a texel of 1 stands for: x, y the luminance a channel follows;
    // z, w its fine blur for star cores.
    guides: vec4f,
    // Star cores: x how far they are compressed, y the end slope of the
    // curve they are pulled toward, w its knee. z: the curve's black, for
    // a channel that follows a luminance.
    cores: vec4f,
    // x: the sky floor, the stretched level that becomes black. y: the
    // stretch-table row of what follows the layered stretch, over 0 to its
    // peak, or -1 for nothing.
    floor: vec4f,
}

// One step of the Finish stage.
struct FinishStep {
    // x: 1 curves, 2 saturation, 3 heal, 4 stretch, 5 tone and color.
    // y: amount; for tone and color, exposure. z: the mask layer that
    // limits the step, or -1 for everywhere. w: the step's opacity.
    kind: vec4f,
    // xy: for a heal, where the replacement comes from, in surface pixels.
    // z: for a stretch, the first of its three rows of the stretch table.
    // w: for a stretch, 1 when it maps luminance. For tone and color:
    // contrast, highlights, shadows, whites.
    heal: vec4f,
    // rgb: for a heal, the brightness added to the copied color. For tone
    // and color: blacks, texture, clarity, dehaze.
    shift: vec4f,
    // For tone and color, x: vibrance, y: saturation, z: the step's layer in
    // the local passes' textures counted from 1, or 0 when it reads none,
    // w: the variance of
    // fine detail Texture treats as half noise.
    local: vec4f,
    // For tone and color, x: the brightest value it gives, as a multiple
    // of SDR white: 1 for SDR.
    range: vec4f,
}

// A heal of one layer.
struct LayerHeal {
    // x: the layer healed. y: 0 for a heal of the band's samples before the
    // stretch, 1 for a heal of the stretched layer. z: the mask layer that
    // holds the heal's stroke. w: the brightness added to the copy.
    what: vec4f,
    // xy: where the copy comes from, in surface pixels.
    offset: vec4f,
}

// A mask computed at each pixel from one or two bands.
struct ComputedMask {
    // x: the band, or -1 for none. y: a second band blended into it before
    // the stretch, or -1. z: how much of the second. w: the row of the
    // stretch table that holds the band's stretch.
    first: vec4f,
    second: vec4f,
    // x: 1 to raise the product to the power of 1 minus itself. y: 1 for
    // 1 minus the result. z: the power the shaped product is raised to.
    shape: vec4f,
    // x: the mask layer of strokes that set the mask to 1, or -1. y: the
    // mask layer of strokes that set it to 0, or -1. z: how far the mask is
    // smoothed before the strokes, as the standard deviation of a blur in
    // surface pixels; 0 is not at all.
    touch: vec4f,
}

struct Globals {
    // clip = doc * view.xy + view.zw
    view: vec4f,
    // x, y: surface size in pixels. z: strength of the stroke overlay.
    // w: the mask layer that holds the overlay.
    screen: vec4f,
    // x: number of layers. y: number of finish steps. z: number of heals
    // of single layers.
    counts: vec4f,
    layers: array<Layer, 24>,
    finish: array<FinishStep, 8>,
    // Per healed layer: x: its band, y: a second band blended into the
    // first, or -1, z: how much of the second. A place not in use has -1.
    healed: array<vec4f, 4>,
    heals: array<LayerHeal, 16>,
    computed: array<ComputedMask, 8>,
    // x: the Finish step whose neighborhoods the local passes are
    // measuring, or -1. y: the half-width of the local window, z: of the detail window,
    // both in surface pixels.
    local: vec4f,
    // The part of the targets this draw fills: x, y, width and height. A
    // window reads only pixels inside it.
    area: vec4f,
    // xyz: the weights of the luminance, from the picture's working space;
    // w: the picture's peak, which screen works up to.
    luma: vec4f,
    // The layered stretches the layers read.
    layered: array<LayeredStep, 12>,
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var linear_sampler: sampler;
// Each layer's stretch, sampled: one row per layer. A row is indexed by the
// square root of the band's sample, so the dark end, where a stretch is
// steepest, gets most of the entries.
@group(0) @binding(2) var stretch_table: texture_2d<f32>;
// The layered stretches' maps, at a quarter of the picture's resolution, a
// layer each: red the core compression's base, green the large-scale layer.
@group(0) @binding(3) var layered_maps: texture_2d_array<f32>;

// An S-shaped contrast curve for a positive amount, and a flattening one for
// a negative amount. 0 leaves x unchanged.
fn contrast_curve(x: f32, amount: f32) -> f32 {
    let clamped = clamp(x, 0.0, 1.0);
    if amount >= 0.0 {
        return mix(clamped, clamped * clamped * (3.0 - 2.0 * clamped), min(amount, 1.0));
    }
    return mix(clamped, 0.5, min(-amount, 1.0) * 0.5);
}
