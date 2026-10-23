// Clamped Lanczos-3 or bilinear resampling onto a region of the reference
// grid, in f32: the taps, weight table, clamping rule and bilinear fallback
// of `registration::resample_row`, one invocation per output pixel and every
// channel of it. The source position of each pixel comes from a coarse grid
// of positions the host evaluated in f64, interpolated bilinearly.
// `with_mask` also resamples the frame's clip mask, bilinearly, at the same
// positions, as `main` would resample it alone.

struct Params {
    // The output region, its origin on the reference grid.
    width: u32,
    height: u32,
    origin_x: u32,
    origin_y: u32,
    source_width: u32,
    source_height: u32,
    channels: u32,
    grid_step: u32,
    grid_columns: u32,
    clamping: f32,
    // 0: clamped Lanczos-3, 1: bilinear.
    sampling: u32,
    // The grid node of the region's first column and row: the grid lies on
    // multiples of `grid_step` of the reference grid, whatever the region,
    // so a pixel's position is the same in every region that holds it.
    grid_origin: u32,
    // Per channel: the span a pass's taps must exceed before it may clamp.
    noise_spans: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> source: array<f32>;
// The source position of every `grid_step`-th reference pixel over the
// region, from node `grid_origin`, row-major: its whole pixels, and the
// fraction past them. Positions are interpolated relative to one node's
// whole pixels, so the fraction keeps its precision far from the origin.
struct Node {
    whole: vec2<i32>,
    fraction: vec2<f32>,
}
@group(0) @binding(2) var<storage, read> grid: array<Node>;
// 1025 steps of six normalized Lanczos-3 weights.
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
// The frame's clip mask, one sample a source pixel, and its resampling.
@group(0) @binding(5) var<storage, read> mask: array<f32>;
@group(0) @binding(6) var<storage, read_write> mask_output: array<u32>;
@group(0) @binding(7) var<storage, read_write> mask_count: atomic<u32>;

const NAN_BITS: u32 = 0x7fc00000u;

// By bits: a shader compiler may assume that no float is NaN.
fn finite(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

fn tap(x: u32, y: u32, channel: u32) -> f32 {
    return source[(y * params.source_width + x) * params.channels + channel];
}

fn clamped(w: u32, t: array<f32, 6>, noise_span: f32) -> f32 {
    let w0 = weights[w * 6u];
    let w1 = weights[w * 6u + 1u];
    let w2 = weights[w * 6u + 2u];
    let w3 = weights[w * 6u + 3u];
    let w4 = weights[w * 6u + 4u];
    let w5 = weights[w * 6u + 5u];
    let positive = w0 * t[0] + w2 * t[2] + w3 * t[3] + w5 * t[5];
    let negative = w1 * t[1] + w4 * t[4];
    if (abs(negative) > params.clamping * abs(positive)) {
        var highest = t[0];
        var lowest = t[0];
        for (var i = 1; i < 6; i++) {
            highest = max(highest, t[i]);
            lowest = min(lowest, t[i]);
        }
        if (highest - lowest > noise_span) {
            return positive / (w0 + w2 + w3 + w5);
        }
    }
    return positive + negative;
}

// Bilinear interpolation of four neighbouring samples, NaN if any is.
fn blend(a: f32, b: f32, c: f32, d: f32, tx: f32, ty: f32) -> f32 {
    if (!(finite(a) && finite(b) && finite(c) && finite(d))) {
        return bitcast<f32>(NAN_BITS);
    }
    let top = a * (1.0 - tx) + b * tx;
    let bottom = c * (1.0 - tx) + d * tx;
    return top * (1.0 - ty) + bottom * ty;
}

fn bilinear(x0: u32, y0: u32, tx: f32, ty: f32, channel: u32) -> f32 {
    let x1 = min(x0 + 1u, params.source_width - 1u);
    let y1 = min(y0 + 1u, params.source_height - 1u);
    return blend(
        tap(x0, y0, channel),
        tap(x1, y0, channel),
        tap(x0, y1, channel),
        tap(x1, y1, channel),
        tx,
        ty,
    );
}

// Where an output pixel reads the source: whole pixels and the fraction
// past them, or `valid` false where it reads nothing.
struct Position {
    valid: bool,
    x0: u32,
    y0: u32,
    tx: f32,
    ty: f32,
}

fn position(id: vec3<u32>) -> Position {
    let none = Position(false, 0u, 0u, 0.0, 0.0);
    let step = params.grid_step;
    let x = params.origin_x + id.x;
    let y = params.origin_y + id.y;
    let gx = x / step;
    let gy = y / step;
    let fx = f32(x - gx * step) / f32(step);
    let fy = f32(y - gy * step) / f32(step);
    let row = (gy - (params.grid_origin >> 16u)) * params.grid_columns
        + gx - (params.grid_origin & 0xffffu);
    let a = grid[row];
    let b = grid[row + 1u];
    let c = grid[row + params.grid_columns];
    let d = grid[row + params.grid_columns + 1u];
    // Each node's position relative to the first node's whole pixels: a few
    // tens of pixels, which f32 holds to a millionth of one.
    let base = a.whole;
    let relative = mix(
        mix(a.fraction, vec2<f32>(b.whole - base) + b.fraction, fx),
        mix(
            vec2<f32>(c.whole - base) + c.fraction,
            vec2<f32>(d.whole - base) + d.fraction,
            fx,
        ),
        fy,
    );
    if (!finite(relative.x) || !finite(relative.y)) {
        return none;
    }
    let steps = floor(relative);
    var whole = base + vec2<i32>(steps);
    var fraction = relative - steps;
    let max_x = i32(params.source_width) - 1;
    let max_y = i32(params.source_height) - 1;
    let epsilon = 1.0e-4;
    // Within the source, give or take rounding, as the CPU's within_source.
    if ((whole.x < 0 && !(whole.x == -1 && fraction.x >= 1.0 - epsilon))
        || (whole.y < 0 && !(whole.y == -1 && fraction.y >= 1.0 - epsilon))
        || (whole.x > max_x || (whole.x == max_x && fraction.x > epsilon))
        || (whole.y > max_y || (whole.y == max_y && fraction.y > epsilon))) {
        return none;
    }
    // Clamp onto the frame, as the CPU does for positions just outside.
    if (whole.x < 0) {
        whole.x = 0;
        fraction.x = 0.0;
    }
    if (whole.y < 0) {
        whole.y = 0;
        fraction.y = 0.0;
    }
    if (whole.x == max_x) {
        fraction.x = 0.0;
    }
    if (whole.y == max_y) {
        fraction.y = 0.0;
    }
    return Position(true, u32(whole.x), u32(whole.y), fraction.x, fraction.y);
}

// Resample every channel of output pixel `id` at `at`.
fn resample_pixel(id: vec3<u32>, at: Position) {
    let first = (id.y * params.width + id.x) * params.channels;
    if (!at.valid) {
        for (var channel = 0u; channel < params.channels; channel++) {
            output[first + channel] = bitcast<f32>(NAN_BITS);
        }
        return;
    }
    let x0 = at.x0;
    let y0 = at.y0;
    let tx = at.tx;
    let ty = at.ty;
    let window = params.sampling == 0u && x0 >= 2u && y0 >= 2u
        && x0 + 3u < params.source_width && y0 + 3u < params.source_height;
    let wx = min(u32(tx * 1024.0 + 0.5), 1024u);
    let wy = min(u32(ty * 1024.0 + 0.5), 1024u);
    for (var channel = 0u; channel < params.channels; channel++) {
        var value = bitcast<f32>(NAN_BITS);
        if (window) {
            let span = params.noise_spans[channel];
            var rows: array<f32, 6>;
            for (var r = 0u; r < 6u; r++) {
                let y = y0 + r - 2u;
                let t = array<f32, 6>(
                    tap(x0 - 2u, y, channel), tap(x0 - 1u, y, channel), tap(x0, y, channel),
                    tap(x0 + 1u, y, channel), tap(x0 + 2u, y, channel), tap(x0 + 3u, y, channel),
                );
                rows[r] = clamped(wx, t, span);
            }
            value = clamped(wy, rows, span);
        }
        if (!finite(value)) {
            value = bilinear(x0, y0, tx, ty, channel);
        }
        output[first + channel] = value;
    }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }
    resample_pixel(id, position(id));
}

fn resample_mask(at: Position) -> f32 {
    var value = bitcast<f32>(NAN_BITS);
    if (at.valid) {
        let x1 = min(at.x0 + 1u, params.source_width - 1u);
        let y1 = min(at.y0 + 1u, params.source_height - 1u);
        let row0 = at.y0 * params.source_width;
        let row1 = y1 * params.source_width;
        value = blend(mask[row0 + at.x0], mask[row0 + x1], mask[row1 + at.x0], mask[row1 + x1], at.tx, at.ty);
    }
    return value;
}

@compute @workgroup_size(16, 16)
fn with_mask(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }
    let at = position(id);
    resample_pixel(id, at);
    mask_output[id.y * params.width + id.x] = bitcast<u32>(resample_mask(at));
}

// Only clipping statistics need this mask. Return its nonzero contributions
// directly, without allocating or reading back a whole mask raster. Each
// pixel writes once, so list order cannot change any pixel's accumulated sum.
@compute @workgroup_size(16, 16)
fn with_sparse_mask(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }
    let at = position(id);
    resample_pixel(id, at);
    let value = resample_mask(at);
    if (finite(value) && value > 0.0) {
        let slot = atomicAdd(&mask_count, 1u);
        if (slot < arrayLength(&mask_output) / 2u) {
            mask_output[2u * slot] = id.y * params.width + id.x;
            mask_output[2u * slot + 1u] = bitcast<u32>(value);
        }
    }
}
