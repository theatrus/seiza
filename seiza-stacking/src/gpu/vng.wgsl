// VNG demosaicing of a Bayer mosaic's interior on the GPU, in f32: the
// gradients, threshold, directions and colour differences of
// `image::vng_rows`, one invocation per pixel two or more from every edge.
// The mosaic is balanced to green first and each estimate scaled back after,
// and every photosite keeps its own sample exactly, so a pixel comes out as
// `debayer_with` leaves it.
//
// The 5x5 window is 25 values, c0 to c24 row by row, and each direction's
// gradient and neighbours are written out cell by cell, so the window stays
// in registers: a window array indexed through tables, as the CPU's are,
// ran five times slower. Every direction's neighbours are `vng_neighbourhoods`'
// six cells, summed here by their parity class relative to the pixel
// (column parity | row parity << 1), each class one colour of the mosaic.

struct Params {
    width: u32,
    height: u32,
    // Each channel's factor that brings it to green's level.
    balance: vec4<f32>,
    // Each phase's channel, a phase being column parity | row parity << 1.
    channels: vec4<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> mosaic: array<f32>;
// The mosaic balanced to green, which `balance` writes for `main`.
@group(0) @binding(2) var<storage, read_write> balanced: array<f32>;
// Interleaved RGB for the whole image; only the interior is written.
@group(0) @binding(3) var<storage, read_write> rgb: array<f32>;

// Bring every photosite to green's level, as `image::balance_row`. A pass of
// its own: balanced in `main` as it read the window, the product was fused
// into the gradients' differences, and on a star's flank that flipped
// directions the CPU kept.
@compute @workgroup_size(16, 16)
fn balance(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }
    let phase = (id.x & 1u) | ((id.y & 1u) << 1u);
    let at = id.y * params.width + id.x;
    balanced[at] = mosaic[at] * params.balance[params.channels[phase]];
}

// NaN is tested by its bits, since shader compilers may assume no float is
// NaN.
fn is_nan(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u;
}

// The least and greatest gradient that is not NaN, as the CPU's fold of
// `f32::min` and `f32::max` from the infinities; `found` is false while
// every gradient so far is NaN.
struct Range {
    low: f32,
    high: f32,
    found: bool,
}

fn widen(range: ptr<function, Range>, gradient: f32) {
    if (is_nan(gradient)) {
        return;
    }
    if ((*range).found) {
        (*range).low = min((*range).low, gradient);
        (*range).high = max((*range).high, gradient);
    } else {
        *range = Range(gradient, gradient, true);
    }
}

// The sums of the two missing colours' differences from the pixel's own over
// the directions kept, and how many were kept.
struct Estimate {
    first: f32,
    second: f32,
    used: f32,
}

// Add one direction, its neighbours' `sums` and `counts` per parity class,
// the pixel's own sample counted in class 0. A green pixel's own colour is
// classes 0 and 3, and its missing ones class 1 (along its row) and class 2
// (along its column); a red or blue pixel's own is class 0, green classes 1
// and 2, and the other class 3.
fn direction(
    estimate: ptr<function, Estimate>,
    gradient: f32,
    sums: vec4<f32>,
    counts: vec4<f32>,
    green: bool,
    threshold: f32,
    keep_all: bool,
) {
    // A comparison with NaN fails, so a NaN gradient keeps the direction.
    if (!keep_all && !is_nan(gradient) && gradient > threshold) {
        return;
    }
    var own: f32;
    var first: f32;
    var second: f32;
    if (green) {
        own = (sums[0] + sums[3]) / (counts[0] + counts[3]);
        first = sums[1] / counts[1];
        second = sums[2] / counts[2];
    } else {
        own = sums[0] / counts[0];
        first = (sums[1] + sums[2]) / (counts[1] + counts[2]);
        second = sums[3] / counts[3];
    }
    (*estimate).first += first - own;
    (*estimate).second += second - own;
    (*estimate).used += 1.0;
}

// Each direction's neighbours per parity class, the pixel itself in class 0.
const ALONG_COLUMNS = vec4<f32>(2.0, 2.0, 1.0, 2.0);
const ALONG_ROWS = vec4<f32>(2.0, 1.0, 2.0, 2.0);
const DIAGONAL = vec4<f32>(2.0, 2.0, 2.0, 1.0);

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x + 2u;
    let y = id.y + 2u;
    if (x + 2u >= params.width || y + 2u >= params.height) {
        return;
    }
    let phase = (x & 1u) | ((y & 1u) << 1u);
    let own = params.channels[phase];
    let green = own == 1u;
    let row2 = y * params.width + x;
    let row0 = row2 - 2u * params.width;
    let row1 = row2 - params.width;
    let row3 = row2 + params.width;
    let row4 = row2 + 2u * params.width;
    let c0 = balanced[row0 - 2u];
    let c1 = balanced[row0 - 1u];
    let c2 = balanced[row0];
    let c3 = balanced[row0 + 1u];
    let c4 = balanced[row0 + 2u];
    let c5 = balanced[row1 - 2u];
    let c6 = balanced[row1 - 1u];
    let c7 = balanced[row1];
    let c8 = balanced[row1 + 1u];
    let c9 = balanced[row1 + 2u];
    let c10 = balanced[row2 - 2u];
    let c11 = balanced[row2 - 1u];
    let c12 = balanced[row2];
    let c13 = balanced[row2 + 1u];
    let c14 = balanced[row2 + 2u];
    let c15 = balanced[row3 - 2u];
    let c16 = balanced[row3 - 1u];
    let c17 = balanced[row3];
    let c18 = balanced[row3 + 1u];
    let c19 = balanced[row3 + 2u];
    let c20 = balanced[row4 - 2u];
    let c21 = balanced[row4 - 1u];
    let c22 = balanced[row4];
    let c23 = balanced[row4 + 1u];
    let c24 = balanced[row4 + 2u];
    // Each direction's gradient: same-colour differences two apart on the
    // line through the pixel and, at half weight, the lines beside it.
    let up = abs(c7 - c17) + abs(c2 - c12) + 0.5 * (abs(c8 - c18) + abs(c6 - c16));
    let down = abs(c17 - c7) + abs(c22 - c12) + 0.5 * (abs(c16 - c6) + abs(c18 - c8));
    let left = abs(c11 - c13) + abs(c10 - c12) + 0.5 * (abs(c6 - c8) + abs(c16 - c18));
    let right = abs(c13 - c11) + abs(c14 - c12) + 0.5 * (abs(c18 - c16) + abs(c8 - c6));
    let up_left = abs(c6 - c18) + abs(c0 - c12) + 0.5 * (abs(c2 - c14) + abs(c10 - c22));
    let up_right = abs(c8 - c16) + abs(c4 - c12) + 0.5 * (abs(c14 - c22) + abs(c2 - c10));
    let down_left = abs(c16 - c8) + abs(c20 - c12) + 0.5 * (abs(c10 - c2) + abs(c22 - c14));
    let down_right = abs(c18 - c6) + abs(c24 - c12) + 0.5 * (abs(c22 - c10) + abs(c14 - c2));
    var range = Range(0.0, 0.0, false);
    widen(&range, up);
    widen(&range, down);
    widen(&range, left);
    widen(&range, right);
    widen(&range, up_left);
    widen(&range, up_right);
    widen(&range, down_left);
    widen(&range, down_right);
    let threshold = 1.5 * range.low + 0.5 * (range.high - range.low);
    // With every gradient NaN the CPU's threshold is NaN, and so is it when
    // an infinite sample makes infinity less infinity; either keeps every
    // direction.
    let keep_all = !range.found || is_nan(threshold);
    var estimate = Estimate(0.0, 0.0, 0.0);
    // Each direction's neighbours by parity class: up is (-2, -1), (-2, 0),
    // (-2, 1), (-1, -1), (-1, 0) and (-1, 1) as (row, column), and so on.
    direction(&estimate, up, vec4<f32>(c12 + c2, c1 + c3, c7, c6 + c8), ALONG_COLUMNS, green, threshold, keep_all);
    direction(&estimate, down, vec4<f32>(c12 + c22, c21 + c23, c17, c16 + c18), ALONG_COLUMNS, green, threshold, keep_all);
    direction(&estimate, left, vec4<f32>(c12 + c10, c11, c5 + c15, c6 + c16), ALONG_ROWS, green, threshold, keep_all);
    direction(&estimate, right, vec4<f32>(c12 + c14, c13, c9 + c19, c8 + c18), ALONG_ROWS, green, threshold, keep_all);
    direction(&estimate, up_left, vec4<f32>(c12 + c0, c1 + c11, c5 + c7, c6), DIAGONAL, green, threshold, keep_all);
    direction(&estimate, up_right, vec4<f32>(c12 + c4, c3 + c13, c7 + c9, c8), DIAGONAL, green, threshold, keep_all);
    direction(&estimate, down_left, vec4<f32>(c12 + c20, c11 + c21, c15 + c17, c16), DIAGONAL, green, threshold, keep_all);
    direction(&estimate, down_right, vec4<f32>(c12 + c24, c13 + c23, c17 + c19, c18), DIAGONAL, green, threshold, keep_all);
    // The channels the two estimates are of.
    var first = 1u;
    var second = params.channels[phase ^ 3u];
    if (green) {
        first = params.channels[phase ^ 1u];
        second = params.channels[phase ^ 2u];
    }
    let out = row2 * 3u;
    rgb[out + own] = mosaic[row2];
    if (estimate.used == 0.0) {
        // As the CPU, which leaves the zero it started from.
        rgb[out + first] = 0.0;
        rgb[out + second] = 0.0;
    } else {
        rgb[out + first] = (c12 + estimate.first / estimate.used) / params.balance[first];
        rgb[out + second] = (c12 + estimate.second / estimate.used) / params.balance[second];
    }
}
