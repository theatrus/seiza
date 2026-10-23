// Gather the exact samples the CPU's normalization and photometry read.
// Count coverage over the full frame using integer reductions.
struct Params {
    samples: u32,
    gathered: u32,
    row: u32,
    padding: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> frame: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> values: array<f32>;
@group(0) @binding(4) var<storage, read_write> finite: atomic<u32>;

@compute @workgroup_size(256)
fn gather(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x + id.y * params.row;
    if index < params.gathered {
        values[index] = frame[indices[index]];
    }
}

var<workgroup> counts: array<u32, 256>;

@compute @workgroup_size(256)
fn count_finite(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    let index = id.x + id.y * params.row;
    var count = 0u;
    if index < params.samples {
        count = select(0u, 1u, (bitcast<u32>(frame[index]) & 0x7f800000u) != 0x7f800000u);
    }
    counts[local] = count;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if local < stride {
            counts[local] += counts[local + stride];
        }
        workgroupBarrier();
        if stride == 1u { break; }
        stride /= 2u;
    }
    if local == 0u {
        atomicAdd(&finite, counts[0]);
    }
}
