// The first test of `defects::frame_deviations` on every sample: whether it
// may stand more than its plane's threshold from the extremes of its eight
// same-plane neighbours. A sample within the threshold of both cannot
// deviate beyond it, and nearly every sample is; the CPU examines only those
// flagged here, one bit each, 32 to a word.
//
// The test is loosened by a hundred-thousandth of the values compared, so a
// compiler free to reorder the arithmetic still flags every sample the
// CPU's exact test keeps; the CPU makes that test again on each.

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    // 2 on a raw colour frame, whose same-colour neighbours are two apart.
    step: u32,
    samples: u32,
    // Invocations in a row of the dispatch.
    row: u32,
    // `sigma` times each plane's noise, by channel * step^2 + plane.
    thresholds: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read_write> flags: array<u32>;

var<workgroup> words: array<atomic<u32>, 8>;

// Tested by its bits, since shader compilers may assume no float is NaN or
// infinite.
fn finite(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

fn flagged(sample: u32) -> bool {
    let value = data[sample];
    if (!finite(value)) {
        return false;
    }
    let pixel = sample / params.channels;
    let channel = sample % params.channels;
    let x = i32(pixel % params.width);
    let y = i32(pixel / params.width);
    let step = i32(params.step);
    var count = 0u;
    var low = 0.0;
    var high = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let nx = x + dx * step;
            let ny = y + dy * step;
            if ((dx == 0 && dy == 0) || nx < 0 || ny < 0 || nx >= i32(params.width)
                || ny >= i32(params.height)) {
                continue;
            }
            let neighbour = data[(u32(ny) * params.width + u32(nx)) * params.channels + channel];
            if (!finite(neighbour)) {
                continue;
            }
            if (count == 0u) {
                low = neighbour;
                high = neighbour;
            } else {
                low = min(low, neighbour);
                high = max(high, neighbour);
            }
            count += 1u;
        }
    }
    if (count < 5u) {
        return false;
    }
    let planes = params.step * params.step;
    let plane = u32(y % step) * params.step + u32(x % step);
    let threshold = params.thresholds[channel * planes + plane];
    let slack = 1e-5 * (abs(value) + abs(low) + abs(high) + abs(threshold));
    return value - low > threshold - slack || high - value > threshold - slack;
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    if (local < 8u) {
        atomicStore(&words[local], 0u);
    }
    workgroupBarrier();
    let sample = id.y * params.row + id.x;
    if (sample < params.samples && flagged(sample)) {
        atomicOr(&words[local / 32u], 1u << (local % 32u));
    }
    workgroupBarrier();
    // A workgroup's 256 samples are eight whole words, as a row of the
    // dispatch is a whole number of workgroups.
    let first = sample - local;
    if (local < 8u && first + local * 32u < params.samples) {
        flags[first / 32u + local] = atomicLoad(&words[local]);
    }
}
