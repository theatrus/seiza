// Write a few samples into a buffer the GPU holds: each entry of `writes` is
// a sample's index and its value's bits. The border of a debayered frame
// the CPU filled goes in this way, so the GPU's copy is the whole frame.

struct Params {
    entries: u32,
    // Invocations in a row of the dispatch.
    row: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> writes: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> frame: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let entry = id.y * params.row + id.x;
    if (entry < params.entries) {
        frame[writes[entry].x] = bitcast<f32>(writes[entry].y);
    }
}
