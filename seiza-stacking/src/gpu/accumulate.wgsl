// The live stack's accumulator on the GPU, in f32 as on the CPU: delta-sigma
// classification of a frame against the moments so far, and Welford's
// update (West's, weighted) of the samples it keeps. A frame covers a region
// of the grid; its samples are the region's rows, each `span` samples long.

struct Params {
    // The region, in samples of the grid's rows.
    first_row: u32,
    rows: u32,
    first_sample: u32,
    span: u32,
    // Samples in a whole row of the grid.
    grid_row: u32,
    channels: u32,
    weighted: u32,
    // 1 for delta-sigma rejection, 0 for none.
    rejecting: u32,
    low_sigma: f32,
    high_sigma: f32,
    warmup: u32,
    minimum_sigma: f32,
    // This frame's weight per channel.
    weights: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> frame: array<f32>;
@group(0) @binding(2) var<storage, read_write> mean: array<f32>;
@group(0) @binding(3) var<storage, read_write> m2: array<f32>;
@group(0) @binding(4) var<storage, read_write> count: array<u32>;
@group(0) @binding(5) var<storage, read_write> rejected: array<u32>;
@group(0) @binding(6) var<storage, read_write> weight_sum: array<f32>;
// The frame's accepted and refused samples.
@group(0) @binding(7) var<storage, read_write> counters: array<atomic<u32>, 2>;

var<workgroup> local_counts: array<atomic<u32>, 2>;

fn finite(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

// Whether the moments so far refuse `sample`, as `should_reject_sample` and
// its weighted twin, which agree at weight 1.
fn refuses(sample: f32, at: u32, weight: f32) -> bool {
    let n = count[at];
    if (params.rejecting == 0u || n < params.warmup || n <= 1u) {
        return false;
    }
    var variance = m2[at] / f32(n - 1u);
    if (params.weighted == 1u) {
        variance = variance / weight;
    }
    let sigma = max(sqrt(variance), params.minimum_sigma);
    let delta = sample - mean[at];
    return delta < -params.low_sigma * sigma || delta > params.high_sigma * sigma;
}

// The grid sample and frame sample of an invocation, or false past the
// region.
fn place(id: vec3<u32>) -> vec3<u32> {
    // x runs along a region row's samples, y down its rows.
    let at = (params.first_row + id.y) * params.grid_row + params.first_sample + id.x;
    return vec3<u32>(at, id.y * params.span + id.x, (params.first_sample + id.x) % params.channels);
}

@compute @workgroup_size(64, 4)
fn classify(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) local: u32) {
    if (id.x < params.span && id.y < params.rows) {
        let placed = place(id);
        let sample = frame[placed.y];
        if (finite(sample)) {
            let refused = refuses(sample, placed.x, params.weights[placed.z]);
            atomicAdd(&local_counts[select(0u, 1u, refused)], 1u);
        }
    }
    workgroupBarrier();
    if (local == 0u) {
        let accepted = atomicLoad(&local_counts[0]);
        let refused = atomicLoad(&local_counts[1]);
        if (accepted > 0u) {
            atomicAdd(&counters[0], accepted);
        }
        if (refused > 0u) {
            atomicAdd(&counters[1], refused);
        }
    }
}

@compute @workgroup_size(64, 4)
fn integrate(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) local: u32) {
    if (id.x < params.span && id.y < params.rows) {
        let placed = place(id);
        let at = placed.x;
        let sample = frame[placed.y];
        if (finite(sample)) {
            let weight = params.weights[placed.z];
            if (refuses(sample, at, weight)) {
                rejected[at] = rejected[at] + select(1u, 0u, rejected[at] == 0xffffffffu);
                atomicAdd(&local_counts[1], 1u);
            } else {
                let n = count[at];
                let old_mean = mean[at];
                if (params.weighted == 1u) {
                    let next_weight = weight_sum[at] + weight;
                    let delta = sample - old_mean;
                    let weighted_delta = delta * weight;
                    let next_mean = old_mean + weighted_delta / next_weight;
                    mean[at] = next_mean;
                    m2[at] = m2[at] + weighted_delta * (sample - next_mean);
                    weight_sum[at] = next_weight;
                } else {
                    let next_count = n + select(1u, 0u, n == 0xffffffffu);
                    let delta = sample - old_mean;
                    let next_mean = old_mean + delta / f32(next_count);
                    mean[at] = next_mean;
                    m2[at] = m2[at] + delta * (sample - next_mean);
                }
                count[at] = n + select(1u, 0u, n == 0xffffffffu);
                atomicAdd(&local_counts[0], 1u);
            }
        }
    }
    workgroupBarrier();
    if (local == 0u) {
        let accepted = atomicLoad(&local_counts[0]);
        let refused = atomicLoad(&local_counts[1]);
        if (accepted > 0u) {
            atomicAdd(&counters[0], accepted);
        }
        if (refused > 0u) {
            atomicAdd(&counters[1], refused);
        }
    }
}
