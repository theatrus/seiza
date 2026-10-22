// Star detection's per-pixel work on the GPU, as `registration::detect`
// and `seiza::detect_stars_luma_f32` do it: the frame's Rec.709 luminance,
// normalized to [0, 1] between its finite extremes, each tile's median and
// robust sigma, and the excess of every pixel above its tile's median by
// more than `sigma` of its noise. What passes is written in row-major
// order, as (pixel, excess), for `seiza::detect_stars_from_excess` to
// gather into stars on the CPU. Six passes, dispatched in order in one
// submission.

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    tile: u32,
    tiles_x: u32,
    tiles_y: u32,
    sigma: f32,
    // Entries `positives` holds; past them a pass counts but does not write.
    capacity: u32,
    // Invocations in a row of the per-pixel dispatches.
    row: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> frame: array<f32>;
@group(0) @binding(2) var<storage, read_write> luma: array<f32>;
// The finite luminance's least and greatest, as order keys.
@group(0) @binding(3) var<storage, read_write> extremes: array<atomic<u32>, 2>;
// Each tile's median and noise.
@group(0) @binding(4) var<storage, read_write> tiles: array<vec2<f32>>;
// Each row's count of positive pixels, then their running sum: the first
// entry of each row's in `positives`, and the total last.
@group(0) @binding(5) var<storage, read_write> rows: array<u32>;
@group(0) @binding(6) var<storage, read_write> positives: array<vec2<u32>>;

// Tested by its bits, since shader compilers may assume no float is NaN or
// infinite.
fn finite(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

// A float's bits as an integer that orders as `f32::total_cmp` orders
// floats, and back.
fn order_key(value: f32) -> u32 {
    let bits = bitcast<u32>(value);
    return bits ^ select(0x80000000u, 0xffffffffu, (bits & 0x80000000u) != 0u);
}

fn from_key(key: u32) -> f32 {
    return bitcast<f32>(key ^ select(0xffffffffu, 0x80000000u, (key & 0x80000000u) != 0u));
}

var<workgroup> lows: array<u32, 256>;
var<workgroup> highs: array<u32, 256>;

// The luminance, as `LinearImage::luminance`'s `rec709_luma` forms it, and
// its finite extremes: each workgroup's by a tree, then one atomic each.
@compute @workgroup_size(256)
fn luminance(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    let pixel = id.y * params.row + id.x;
    var low = 0xffffffffu;
    var high = 0u;
    if (pixel < params.width * params.height) {
        var value: f32;
        if (params.channels == 3u) {
            let at = pixel * 3u;
            value = fma(0.2126, frame[at], fma(0.7152, frame[at + 1u], 0.0722 * frame[at + 2u]));
        } else {
            value = frame[pixel];
        }
        luma[pixel] = value;
        if (finite(value)) {
            low = order_key(value);
            high = low;
        }
    }
    lows[local] = low;
    highs[local] = high;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (local < stride) {
            lows[local] = min(lows[local], lows[local + stride]);
            highs[local] = max(highs[local], highs[local + stride]);
        }
        workgroupBarrier();
    }
    if (local == 0u) {
        atomicMin(&extremes[0], lows[0]);
        atomicMax(&extremes[1], highs[0]);
    }
}

// `normalize_for_detection`, in f32 where the CPU works in f64: a sample
// can differ by a unit in the last place. Each workgroup reads the extremes
// once.
@compute @workgroup_size(256)
fn normalize(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    if (local == 0u) {
        lows[0] = atomicLoad(&extremes[0]);
        highs[0] = atomicLoad(&extremes[1]);
    }
    workgroupBarrier();
    let pixel = id.y * params.row + id.x;
    if (pixel >= params.width * params.height) {
        return;
    }
    let least_key = lows[0];
    let lowest = from_key(least_key);
    let highest = from_key(highs[0]);
    let value = luma[pixel];
    if (least_key == 0xffffffffu || !(highest > lowest) || !finite(value)) {
        luma[pixel] = 0.0;
    } else {
        luma[pixel] = (value - lowest) / (highest - lowest);
    }
}

const TILE_SAMPLES: u32 = 4096u;
// Samples each invocation of the background pass holds: a tile's worth
// over the workgroup.
const HELD: u32 = 16u;
const BINS: u32 = 1024u;
var<workgroup> histogram: array<atomic<u32>, 1024>;
var<workgroup> sums: array<u32, 256>;
var<workgroup> groups: array<u32, 16>;
var<workgroup> found: array<u32, 2>;

// The key of rank `rank` (from 0) among the keys the workgroup's
// invocations hold in `keys`, all within 0..=`span`: a radix select, ten
// bits a round, each round counting the keys of the range left into a
// histogram and keeping the bin that rank falls in. Exact, and a key past
// `span` is not counted.
fn select_rank(keys: ptr<function, array<u32, 16>>, local: u32, wanted: u32, span_in: u32) -> u32 {
    var base = 0u;
    var span = span_in;
    var rank = wanted;
    loop {
        if (span == 0u) {
            break;
        }
        let bits = 32u - countLeadingZeros(span);
        let shift = select(0u, bits - 10u, bits > 10u);
        for (var bin = local; bin < BINS; bin += 256u) {
            atomicStore(&histogram[bin], 0u);
        }
        workgroupBarrier();
        for (var i = 0u; i < HELD; i++) {
            let key = (*keys)[i];
            if (key >= base && key - base <= span) {
                atomicAdd(&histogram[(key - base) >> shift], 1u);
            }
        }
        workgroupBarrier();
        var sum = 0u;
        for (var j = 0u; j < 4u; j++) {
            sum += atomicLoad(&histogram[local * 4u + j]);
        }
        sums[local] = sum;
        workgroupBarrier();
        if (local < 16u) {
            var group_sum = 0u;
            for (var t = local * 16u; t < local * 16u + 16u; t++) {
                group_sum += sums[t];
            }
            groups[local] = group_sum;
        }
        workgroupBarrier();
        // One invocation walks the 16 groups' sums, the 16 sums of the
        // group the rank falls in, then the four bins of the one it falls
        // in there.
        if (local == 0u) {
            var before = 0u;
            var group = 0u;
            for (var g = 0u; g < 16u; g++) {
                if (rank < before + groups[g]) {
                    group = g;
                    break;
                }
                before += groups[g];
            }
            var owner = group * 16u;
            for (var t = group * 16u; t < group * 16u + 16u; t++) {
                if (rank < before + sums[t]) {
                    owner = t;
                    break;
                }
                before += sums[t];
            }
            var bin = owner * 4u;
            for (var j = 0u; j < 4u; j++) {
                let count = atomicLoad(&histogram[owner * 4u + j]);
                if (rank < before + count) {
                    bin = owner * 4u + j;
                    break;
                }
                before += count;
            }
            found[0] = bin;
            found[1] = before;
        }
        workgroupBarrier();
        let bin = found[0];
        rank -= found[1];
        base += bin << shift;
        span = min(span - (bin << shift), (1u << shift) - 1u);
        workgroupBarrier();
    }
    return base;
}

// The median of the `count` keys the workgroup holds, as `median_in_place`
// takes it.
fn held_median(keys: ptr<function, array<u32, 16>>, local: u32, count: u32, span: u32) -> f32 {
    let middle = count / 2u;
    var median = bitcast<f32>(select_rank(keys, local, middle, span));
    if (count % 2u == 0u) {
        median = (bitcast<f32>(select_rank(keys, local, middle - 1u, span)) + median) * 0.5;
    }
    return median;
}

// Each tile's median and robust sigma, as `seiza::detect`'s
// `estimate_background` takes them, by exact selection. The normalized
// luminance lies in 0..=1 with no NaN, so a sample's bits order as it does
// and span no more than 1.0's.
@compute @workgroup_size(256)
fn background(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    let x0 = group.x * params.tile;
    let y0 = group.y * params.tile;
    let tile_width = min(params.tile, params.width - x0);
    let tile_height = min(params.tile, params.height - y0);
    let count = tile_width * tile_height;
    var keys: array<u32, 16>;
    for (var i = 0u; i < HELD; i++) {
        let at = local + i * 256u;
        var key = 0xffffffffu;
        if (at < count) {
            key = bitcast<u32>(luma[(y0 + at / tile_width) * params.width + x0 + at % tile_width]);
        }
        keys[i] = key;
    }
    let one = bitcast<u32>(1.0);
    let median = held_median(&keys, local, count, one);
    // Each sample's distance from it, which `robust_sigma_in_place` takes
    // the median of.
    for (var i = 0u; i < HELD; i++) {
        if (keys[i] != 0xffffffffu) {
            keys[i] = bitcast<u32>(abs(bitcast<f32>(keys[i]) - median));
        }
    }
    let spread = held_median(&keys, local, count, one);
    if (local == 0u) {
        tiles[group.y * params.tiles_x + group.x] = vec2<f32>(median, max(spread * 1.4826, 1e-6));
    }
}

// A pixel's excess above its tile's median, when it passes the threshold.
fn excess_at(x: u32, y: u32) -> f32 {
    let tile = tiles[(y / params.tile) * params.tiles_x + x / params.tile];
    let excess = luma[y * params.width + x] - tile.x;
    // A NaN fails the comparison, as on the CPU; none reaches here.
    return select(0.0, excess, excess > params.sigma * tile.y);
}

var<workgroup> counts: array<u32, 256>;

// Each row's positive pixels: one workgroup a row, each invocation a run
// of it, and the runs' counts summed.
@compute @workgroup_size(256)
fn count_rows(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    let y = group.y * 65535u + group.x;
    let run = (params.width + 255u) / 256u;
    var found = 0u;
    if (y < params.height) {
        for (var x = local * run; x < min((local + 1u) * run, params.width); x++) {
            found += select(0u, 1u, excess_at(x, y) > 0.0);
        }
    }
    counts[local] = found;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (local < stride) {
            counts[local] += counts[local + stride];
        }
        workgroupBarrier();
    }
    if (local == 0u && y < params.height) {
        rows[y] = counts[0];
    }
}

// Each row's first entry in `positives`: the running sum of the counts, in
// `rows[height..]`, the total last.
@compute @workgroup_size(256)
fn offsets(@builtin(local_invocation_index) local: u32) {
    let run = (params.height + 255u) / 256u;
    let first = min(local * run, params.height);
    let last = min(first + run, params.height);
    var sum = 0u;
    for (var y = first; y < last; y++) {
        sum += rows[y];
    }
    counts[local] = sum;
    workgroupBarrier();
    // Inclusive running sum over the invocations' runs, then exclusive.
    for (var stride = 1u; stride < 256u; stride <<= 1u) {
        var add = 0u;
        if (local >= stride) {
            add = counts[local - stride];
        }
        workgroupBarrier();
        counts[local] += add;
        workgroupBarrier();
    }
    var running = counts[local] - sum;
    for (var y = first; y < last; y++) {
        rows[params.height + y] = running;
        running += rows[y];
    }
    if (local == 255u) {
        rows[2u * params.height] = counts[255];
    }
}

// Each row's positive pixels in order, at the row's offset, each
// invocation's run after the runs before it.
@compute @workgroup_size(256)
fn write_rows(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    let y = group.y * 65535u + group.x;
    let run = (params.width + 255u) / 256u;
    let first = local * run;
    let last = min(first + run, params.width);
    var found = 0u;
    if (y < params.height) {
        for (var x = first; x < last; x++) {
            found += select(0u, 1u, excess_at(x, y) > 0.0);
        }
    }
    counts[local] = found;
    workgroupBarrier();
    for (var stride = 1u; stride < 256u; stride <<= 1u) {
        var add = 0u;
        if (local >= stride) {
            add = counts[local - stride];
        }
        workgroupBarrier();
        counts[local] += add;
        workgroupBarrier();
    }
    if (y >= params.height) {
        return;
    }
    var at = rows[params.height + y] + counts[local] - found;
    for (var x = first; x < last; x++) {
        let excess = excess_at(x, y);
        if (excess > 0.0) {
            if (at < params.capacity) {
                positives[at] = vec2<u32>(y * params.width + x, bitcast<u32>(excess));
            }
            at += 1u;
        }
    }
}
