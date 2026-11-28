// The Finish stage: steps applied to the combined image in order, each
// limited by its own mask, then the stroke overlay. The result stays in the
// picture's working space; a host applies its display transform after it.

@group(1) @binding(0) var combined: texture_2d<f32>;
@group(1) @binding(1) var masks: texture_2d_array<f32>;
// The cover of a step in the stack that reads neighbors, as the layers pass
// works it out: its opacity, its masks, its clip and its computed mask.
@group(1) @binding(2) var step_cover: texture_2d<f32>;

// The local passes. A window pass reads `local_source`. A later step's
// measurements and the Finish pass read what they measured, one layer per
// local Tone step: per pixel, the luminance and darkest channel of the
// picture as the steps before that step leave it, and the means over the
// local and detail windows.
@group(2) @binding(0) var local_source: texture_2d<f32>;
@group(2) @binding(1) var local_measured: texture_2d_array<f32>;
@group(2) @binding(2) var local_means: texture_2d_array<f32>;
@group(2) @binding(3) var detail_means: texture_2d_array<f32>;

@vertex
fn finish_vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    let corner = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
}

// How much of a step applies at a pixel.
fn coverage(pixel: vec2i, mask_layer: f32) -> f32 {
    if mask_layer < 0.0 {
        return 1.0;
    }
    return textureLoad(masks, pixel, i32(mask_layer), 0).r;
}

// How much of a step applies at a pixel: its opacity, within its mask. A
// step in the stack that reads neighbors (mask -2) has its whole cover in
// `step_cover`.
fn applies(pixel: vec2i, step: FinishStep) -> f32 {
    if step.kind.z < -1.5 {
        return textureLoad(step_cover, pixel, 0).r;
    }
    return coverage(pixel, step.kind.z) * step.kind.w;
}

// Whether a step maps each pixel alone: everything but a heal.
fn maps_pixels(step: FinishStep) -> bool {
    return step.kind.x < 2.5 || step.kind.x > 3.5;
}

// A step that maps each pixel alone, or a Tone step that reads the windows
// around `pixel`.
fn adjust(color: vec3f, step: FinishStep, pixel: vec2i) -> vec3f {
    if step.kind.x > 4.5 {
        if step.local.z > 0.5 {
            let layer = i32(step.local.z) - 1;
            return tone(
                color,
                textureLoad(local_measured, pixel, layer, 0).x,
                textureLoad(local_means, pixel, layer, 0),
                textureLoad(detail_means, pixel, layer, 0).xy,
                step,
            );
        }
    }
    return step_alone(color, step);
}

// The combined color at a pixel after the steps before `upto`, leaving heals
// out. A heal copies this from its source pixel.
fn before(pixel: vec2i, upto: i32) -> vec3f {
    var color = textureLoad(combined, pixel, 0).rgb;
    for (var index = i32(globals.counts.w); index < upto; index++) {
        let step = globals.finish[index];
        if maps_pixels(step) {
            color = mix(color, adjust(color, step, pixel), applies(pixel, step));
        }
    }
    return color;
}

// The local passes. The first measures, at each pixel, the luminance, its
// square, the darkest channel and their product, of the picture as the
// steps before the local Tone step leave it.
@fragment
fn local_measure_fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
    let color = before(vec2i(position.xy), i32(globals.local.x));
    return alone(color)[0];
}

// The mean of `local_source` over `radius` pixels either side of `pixel`,
// across or down, the window cut at the edge of the picture drawn.
fn window_mean(pixel: vec2i, across: bool, radius: i32) -> vec4f {
    let low = vec2i(globals.area.xy);
    let high = low + vec2i(globals.area.zw) - vec2i(1);
    var sum = vec4f(0.0);
    var count = 0.0;
    for (var step = -radius; step <= radius; step++) {
        var at = pixel + vec2i(0, step);
        if across {
            at = pixel + vec2i(step, 0);
        }
        if all(at >= low) && all(at <= high) {
            sum += textureLoad(local_source, at, 0);
            count += 1.0;
        }
    }
    return sum / max(count, 1.0);
}

@fragment
fn local_across_fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
    return window_mean(vec2i(position.xy), true, i32(globals.local.y));
}

@fragment
fn local_down_fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
    return window_mean(vec2i(position.xy), false, i32(globals.local.y));
}

@fragment
fn detail_across_fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
    return window_mean(vec2i(position.xy), true, i32(globals.local.z));
}

@fragment
fn detail_down_fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
    return window_mean(vec2i(position.xy), false, i32(globals.local.z));
}

// The combined color at a pixel after the steps from `counts.w` up to
// `counts.y`.
fn finished(pixel: vec2i) -> vec3f {
    let size = vec2i(globals.screen.xy);
    var color = textureLoad(combined, pixel, 0).rgb;
    let count = i32(globals.counts.y);
    for (var index = i32(globals.counts.w); index < count; index++) {
        let step = globals.finish[index];
        let covered = applies(pixel, step)
            * range_weight(step.range.y, step.range.z, step.range.w, dot(color, globals.luma.xyz));
        if maps_pixels(step) {
            color = mix(color, adjust(color, step, pixel), covered);
        } else {
            // A heal. Its source must be on the surface to be copied.
            let source = pixel + vec2i(step.heal.xy);
            if all(source >= vec2i(0)) && all(source < size) {
                color = mix(color, before(source, index) + step.shift.rgb, covered);
            }
        }
    }
    return color;
}

// A step in the stack that reads neighbors, on the layers beneath it: the
// staged picture the next stage of the layers starts from.
@fragment
fn step_fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
    return vec4f(finished(vec2i(position.xy)), 1.0);
}

@fragment
fn finish_fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
    let pixel = vec2i(position.xy);
    var color = finished(pixel);
    let overlay = textureLoad(masks, pixel, i32(globals.screen.w), 0).r * globals.screen.z;
    color = mix(color, vec3f(1.0, 0.25, 0.2), overlay * 0.45);
    return vec4f(color, 1.0);
}
