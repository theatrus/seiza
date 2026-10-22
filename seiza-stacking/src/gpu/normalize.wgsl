// A registered frame's normalization, applied where the GPU holds it, as
// `NormalizationMap::apply_region` applies it on the CPU: each finite
// sample times its gain plus its offset, fused as the CPU's `mul_add`, the
// gain and offset one per channel, or interpolated from the tiles' with the
// CPU's own column and row weights.

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    // 1 for one gain and offset per channel.
    global: u32,
    tile_columns: u32,
    // Invocations in a row of the dispatch.
    row: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Each column's and row's tile weights: the lower and upper tile, and the
// fraction toward the upper's bits, three words each.
@group(0) @binding(1) var<storage, read> columns: array<u32>;
@group(0) @binding(2) var<storage, read> rows: array<u32>;
@group(0) @binding(3) var<storage, read> gains: array<f32>;
@group(0) @binding(4) var<storage, read> offsets: array<f32>;
@group(0) @binding(5) var<storage, read_write> frame: array<f32>;

fn finite(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let sample = id.y * params.row + id.x;
    let channels = params.channels;
    if (sample >= params.width * params.height * channels) {
        return;
    }
    let value = frame[sample];
    if (!finite(value)) {
        return;
    }
    let channel = sample % channels;
    if (params.global == 1u) {
        frame[sample] = fma(value, gains[channel], offsets[channel]);
        return;
    }
    let pixel = sample / channels;
    let x = pixel % params.width;
    let y = pixel / params.width;
    let column_low = columns[x * 3u];
    let column_high = columns[x * 3u + 1u];
    let right = bitcast<f32>(columns[x * 3u + 2u]);
    let left = 1.0 - right;
    let row_low = rows[y * 3u];
    let row_high = rows[y * 3u + 1u];
    let below = bitcast<f32>(rows[y * 3u + 2u]);
    let above = 1.0 - below;
    let top = row_low * params.tile_columns;
    let bottom = row_high * params.tile_columns;
    let top_left = (top + column_low) * channels + channel;
    let top_right = (top + column_high) * channels + channel;
    let bottom_left = (bottom + column_low) * channels + channel;
    let bottom_right = (bottom + column_high) * channels + channel;
    let top_gain = gains[top_left] * left + gains[top_right] * right;
    let bottom_gain = gains[bottom_left] * left + gains[bottom_right] * right;
    let top_offset = offsets[top_left] * left + offsets[top_right] * right;
    let bottom_offset = offsets[bottom_left] * left + offsets[bottom_right] * right;
    let gain = top_gain * above + bottom_gain * below;
    let offset = top_offset * above + bottom_offset * below;
    frame[sample] = fma(value, gain, offset);
}
