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
    // and 11 for its end; 12 for a layer of its color everywhere; 13 and 14
    // for a luminance layer; 15 for a Finish step in the stack, whose entry
    // in the Finish steps is y. y: the adjustment's amount.
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
    // x: number of layers. y: the Finish steps a pass applies end before
    // this one. z: number of heals of single layers. w: the first Finish
    // step a pass applies.
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
    // The layers one layers pass blends, for a stack drawn in stages: x:
    // the first, y: the one it stops before. z: 1 when the pass starts from
    // the staged picture, which a step that reads neighbors made of the
    // layers beneath; 0 when it starts from black.
    stage: vec4f,
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

// The constants of `seiza_compose::tone`, which says what each one is.
const SHIFT: f32 = 0.14;
const WHITE_REACH: f32 = 0.3;
const BLACK_REACH: f32 = 0.1;
const LOCAL_EDGE: f32 = 0.0025;
const CLARITY_GAIN: f32 = 1.0;
const TEXTURE_GAIN: f32 = 1.5;
const DEHAZE_STRENGTH: f32 = 0.95;
const HAZE: f32 = 0.5;
const SHOULDER_START: f32 = 0.8;

// One row of the stretch table at a value from 0 to 1. A row is indexed by
// the square root of the value, as a layer's row is.
fn table_value(value: f32, row: f32) -> f32 {
    let size = vec2f(textureDimensions(stretch_table));
    let at = vec2f(
        (sqrt(clamp(value, 0.0, 1.0)) * (size.x - 1.0) + 0.5) / size.x,
        (row + 0.5) / size.y,
    );
    return textureSampleLevel(stretch_table, linear_sampler, at, 0.0).r;
}

// A stretch of the picture. Its three rows map red, green and blue. With
// luminance, the first row maps the Rec. 709 luminance and the pixel is
// scaled by one factor, kept in gamut, as Seiza's luminance-preserving
// strategy applies a curve.
fn stretch_picture(color: vec3f, step: FinishStep) -> vec3f {
    let row = step.heal.z;
    if step.heal.w < 0.5 {
        return vec3f(
            table_value(color.r, row),
            table_value(color.g, row + 1.0),
            table_value(color.b, row + 2.0),
        );
    }
    let luminance = dot(color, globals.luma.xyz);
    let mapped = table_value(luminance, row);
    if luminance <= 1.0e-8 {
        return vec3f(mapped);
    }
    var scale = mapped / luminance;
    let maximum = max(max(color.r, color.g), max(color.b, 0.0));
    if maximum * scale > 1.0 {
        scale = 1.0 / maximum;
    }
    return clamp(color * scale, vec3f(0.0), vec3f(1.0));
}

// Vibrance and saturation, each from -1 to 1. Vibrance changes a muted
// color more than a vivid one. The CPU render's `colorfulness` is the same.
fn colorfulness(color: vec3f, vibrance: f32, saturation: f32) -> vec3f {
    let luminance = dot(color, globals.luma.xyz);
    let high = max(max(color.r, color.g), color.b);
    let low = min(min(color.r, color.g), color.b);
    var vivid = 0.0;
    if high > 1.0e-6 {
        vivid = clamp((high - low) / high, 0.0, 1.0);
    }
    let gain = (1.0 + saturation) * (1.0 + vibrance * (1.0 - vivid));
    return max(vec3f(luminance) + (color - vec3f(luminance)) * gain, vec3f(0.0));
}

// Tone and color with the means over a pixel's windows, as
// `Tone::apply_around` applies it. `measured` is the luminance the means
// were measured with at this pixel.
fn tone(start: vec3f, measured: f32, local: vec4f, detail: vec2f, step: FinishStep) -> vec3f {
    let exposure = step.kind.y;
    let contrast = step.heal.x;
    let highlights = step.heal.y;
    let shadows = step.heal.z;
    let whites = step.heal.w;
    let blacks = step.shift.x;
    let texture = step.shift.y;
    let clarity = step.shift.z;
    let dehaze = step.shift.w;
    let mean = local.x;
    let variance = max(local.y - mean * mean, 0.0);
    let edge = variance / (variance + LOCAL_EDGE);
    var base = mean + edge * (measured - mean);
    let guided = (local.w - mean * local.z) / (variance + LOCAL_EDGE);
    let veil = clamp(min(local.z + guided * (measured - mean), local.z), 0.0, 1.0);
    let fine_variance = max(detail.y - detail.x * detail.x, 0.0);
    let detailed = fine_variance / (fine_variance + step.local.w);
    var fine = detail.x;

    var color = start;
    var own = dot(color, globals.luma.xyz);
    var scale = 1.0;
    var offset = 0.0;
    if dehaze > 0.0 {
        let lifted = DEHAZE_STRENGTH * dehaze * veil;
        scale = 1.0 / (1.0 - lifted);
        offset = -lifted * scale;
    } else if dehaze < 0.0 {
        let toward = HAZE * -dehaze;
        scale = 1.0 - toward;
        offset = toward * base;
    }
    if dehaze != 0.0 {
        color = max(fma(vec3f(scale), color, vec3f(offset)), vec3f(0.0));
        own = fma(scale, own, offset);
        base = fma(scale, base, offset);
        fine = fma(scale, fine, offset);
    }

    let gain = exp2(exposure);
    let white = 1.0 - WHITE_REACH * whites;
    let black = -BLACK_REACH * blacks;
    let pixel = (max(own, 0.0) * gain - black) / (white - black);
    let around = (max(base, 0.0) * gain - black) / (white - black);
    let small = (max(fine, 0.0) * gain - black) / (white - black);
    let toned = clamp(around, 0.0, 1.0);
    let lifted = toned + SHIFT * shadows * 6.75 * toned * (1.0 - toned) * (1.0 - toned);
    let moved = lifted + SHIFT * highlights * 6.75 * lifted * lifted * (1.0 - lifted);
    var value = pixel + (moved - toned);
    value += clarity * CLARITY_GAIN * (pixel - around) * 4.0 * toned * (1.0 - toned);
    value += texture * TEXTURE_GAIN * detailed * (pixel - small);
    let peak = step.range.x;
    var mapped = contrast_curve(clamp(value, 0.0, 1.0), contrast);
    // An HDR picture's shoulder above SDR white, as `hdr_shoulder`.
    let room = peak - 1.0;
    if room > 0.0 && value > 1.0 {
        let over = value - 1.0;
        let straight = SHOULDER_START * room;
        if over <= straight {
            mapped += over;
        } else {
            let rest = room - straight;
            mapped += straight + rest * (1.0 - exp(-(over - straight) / rest));
        }
    }

    let luminance = dot(color, globals.luma.xyz);
    var scaled = vec3f(mapped);
    if luminance > 1.0e-8 {
        var factor = mapped / luminance;
        let maximum = max(max(color.r, color.g), max(color.b, 0.0));
        if maximum * factor > peak {
            factor = peak / maximum;
        }
        scaled = clamp(color * factor, vec3f(0.0), vec3f(peak));
    }
    return min(colorfulness(scaled, step.local.x, step.local.y), vec3f(peak));
}

// The local and detail means of a pixel alone.
fn alone(color: vec3f) -> array<vec4f, 2> {
    let luminance = dot(color, globals.luma.xyz);
    let darkest = min(min(color.r, color.g), color.b);
    return array<vec4f, 2>(
        vec4f(luminance, luminance * luminance, darkest, luminance * darkest),
        vec4f(luminance, luminance * luminance, 0.0, 0.0),
    );
}

// A step of the Finish stage that maps each pixel alone: a stretch, a
// curve, a saturation, or a Tone step with each pixel's own means. The
// Finish pass and a step in the stack of layers both apply it.
fn step_alone(color: vec3f, step: FinishStep) -> vec3f {
    if step.kind.x > 4.5 {
        let means = alone(color);
        return tone(color, means[0].x, means[0], means[1].xy, step);
    }
    if step.kind.x > 3.5 {
        return stretch_picture(color, step);
    }
    if step.kind.x < 1.5 {
        return vec3f(
            contrast_curve(color.r, step.kind.y),
            contrast_curve(color.g, step.kind.y),
            contrast_curve(color.b, step.kind.y),
        );
    }
    let luminance = dot(color, globals.luma.xyz);
    return max(mix(vec3f(luminance), color, step.kind.y), vec3f(0.0));
}
