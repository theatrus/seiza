// The Finish stage: steps applied to the combined image in order, each
// limited by its own mask, then the stroke overlay. The result stays in the
// picture's working space; a host applies its display transform after it.

@group(1) @binding(0) var combined: texture_2d<f32>;
@group(1) @binding(1) var masks: texture_2d_array<f32>;

// The local passes. A window pass reads `local_source`. A later step's
// measurements and the Finish pass read what they measured, one layer per
// local Tone step: per pixel, the luminance and darkest channel of the
// picture as the steps before that step leave it, and the means over the
// local and detail windows.
@group(2) @binding(0) var local_source: texture_2d<f32>;
@group(2) @binding(1) var local_measured: texture_2d_array<f32>;
@group(2) @binding(2) var local_means: texture_2d_array<f32>;
@group(2) @binding(3) var detail_means: texture_2d_array<f32>;

const LUMA: vec3f = vec3f(0.2126, 0.7152, 0.0722);
// The constants of `seiza_compose::tone`, which says what each one is.
const SHIFT: f32 = 0.14;
const WHITE_REACH: f32 = 0.3;
const BLACK_REACH: f32 = 0.1;
const LOCAL_EDGE: f32 = 0.0025;
const CLARITY_GAIN: f32 = 1.0;
const TEXTURE_GAIN: f32 = 1.5;
const DEHAZE_STRENGTH: f32 = 0.95;
const HAZE: f32 = 0.5;

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

// How much of a step applies at a pixel: its opacity, within its mask.
fn applies(pixel: vec2i, step: FinishStep) -> f32 {
    return coverage(pixel, step.kind.z) * step.kind.w;
}

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
    let luminance = dot(color, LUMA);
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

// Whether a step maps each pixel alone: everything but a heal.
fn maps_pixels(step: FinishStep) -> bool {
    return step.kind.x < 2.5 || step.kind.x > 3.5;
}

// Vibrance and saturation, each from -1 to 1. Vibrance changes a muted
// color more than a vivid one. The CPU render's `colorfulness` is the same.
fn colorfulness(color: vec3f, vibrance: f32, saturation: f32) -> vec3f {
    let luminance = dot(color, LUMA);
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
    var own = dot(color, LUMA);
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
        mapped += room * (1.0 - exp(-(value - 1.0) / room));
    }

    let luminance = dot(color, LUMA);
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
    let luminance = dot(color, LUMA);
    let darkest = min(min(color.r, color.g), color.b);
    return array<vec4f, 2>(
        vec4f(luminance, luminance * luminance, darkest, luminance * darkest),
        vec4f(luminance, luminance * luminance, 0.0, 0.0),
    );
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
    let luminance = dot(color, LUMA);
    return max(mix(vec3f(luminance), color, step.kind.y), vec3f(0.0));
}

// The combined color at a pixel after the steps before `upto`, leaving heals
// out. A heal copies this from its source pixel.
fn before(pixel: vec2i, upto: i32) -> vec3f {
    var color = textureLoad(combined, pixel, 0).rgb;
    for (var index = 0; index < upto; index++) {
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

@fragment
fn finish_fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
    let pixel = vec2i(position.xy);
    let size = vec2i(globals.screen.xy);
    var color = textureLoad(combined, pixel, 0).rgb;
    let count = i32(globals.counts.y);
    for (var index = 0; index < count; index++) {
        let step = globals.finish[index];
        let covered = applies(pixel, step);
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
    let overlay = textureLoad(masks, pixel, i32(globals.screen.w), 0).r * globals.screen.z;
    color = mix(color, vec3f(1.0, 0.25, 0.2), overlay * 0.45);
    return vec4f(color, 1.0);
}
