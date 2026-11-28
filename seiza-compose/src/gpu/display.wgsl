// A picture's color for a display: `seiza_compose::space::WorkingSpace::
// map_to_display` on the GPU. A host's present pass includes this source and
// calls `to_display` with a `DisplayMapping` it fills from
// `seiza_compose::gpu::DisplayMapping`.

struct DisplayMapping {
    // Linear light in the working space to linear light in the display's.
    matrix: mat3x3f,
    // xyz: the display's luminance weights.
    weights: vec4f,
}

// The light of a value on the extended sRGB curve.
fn display_decode(value: f32) -> f32 {
    if value <= 0.04045 {
        return max(value, 0.0) / 12.92;
    }
    return pow((value + 0.055) / 1.055, 2.4);
}

// The value on the extended sRGB curve of linear light.
fn display_encode(light: f32) -> f32 {
    if light <= 0.0031308 {
        return 12.92 * max(light, 0.0);
    }
    return 1.055 * pow(light, 1.0 / 2.4) - 0.055;
}

// Linear light with no channel below 0: desaturated toward the gray of its
// luminance just enough. A color with no luminance is black.
fn display_within_gamut(light: vec3f, weights: vec3f) -> vec3f {
    let least = min(light.r, min(light.g, light.b));
    if least >= 0.0 {
        return light;
    }
    let luminance = dot(light, weights);
    if luminance <= 0.0 {
        return vec3f(0.0);
    }
    let share = luminance / (luminance - least);
    return max(vec3f(luminance) + share * (light - vec3f(luminance)), vec3f(0.0));
}

// A color encoded in the working space, encoded for the display.
fn to_display(color: vec3f, mapping: DisplayMapping) -> vec3f {
    let decoded = vec3f(
        display_decode(max(color.r, 0.0)),
        display_decode(max(color.g, 0.0)),
        display_decode(max(color.b, 0.0)),
    );
    let light = display_within_gamut(mapping.matrix * decoded, mapping.weights.xyz);
    return vec3f(display_encode(light.r), display_encode(light.g), display_encode(light.b));
}
