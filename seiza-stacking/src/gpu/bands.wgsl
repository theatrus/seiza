// The final integration's three passes over one band of every frame, one
// invocation per sample, in f32: the estimates, leave-one-out rejection and
// weighted integration of `batch::bands`, which runs them in f64 on moments
// stored in f32. Each invocation reads its sample of every frame three times.

struct Params {
    // Samples in the band.
    samples: u32,
    // Frames whose samples are in `values`, frame after frame.
    frames: u32,
    // Interleaved channels of a pixel.
    channels: u32,
    // Channels of a frame's weight: 1, or `channels` for a weighted colour
    // stack.
    weight_channels: u32,
    // 1 when frames are weighted: West's update into a weight sum.
    weighted: u32,
    minimum_sigma: f32,
    clipped_correction: f32,
    // Steps of single precision at the mean below which samples are equal.
    precision_floor: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> values: array<f32>;
// Per frame and weight channel: weight, 1 / weight.
@group(0) @binding(2) var<storage, read> weights: array<vec2<f32>>;
// By count: the low and high prediction limits.
@group(0) @binding(3) var<storage, read> thresholds: array<vec2<f32>>;
// By count: 1 + 1/count, the trim factors for a member and for an extreme.
@group(0) @binding(4) var<storage, read> factors: array<vec4<f32>>;
// Per sample: mean, m2, weight sum, unused.
@group(0) @binding(5) var<storage, read_write> moments_out: array<vec4<f32>>;
// Per sample: integrated and rejected counts.
@group(0) @binding(6) var<storage, read_write> tallies_out: array<vec2<u32>>;
// Per frame: finite and integrated samples.
@group(0) @binding(7) var<storage, read_write> frame_counts: array<atomic<u32>>;

const MAX_FRAMES: u32 = 512u;
const TRIMMED_FIRST_ESTIMATE: u32 = 10u;

var<workgroup> local_counts: array<atomic<u32>, 1024>;

fn finite(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

struct Moments {
    mean: f32,
    m2: f32,
    weight_sum: f32,
    count: u32,
}

fn with_sample(m: Moments, value: f32, weight: f32) -> Moments {
    let weight_sum = m.weight_sum + weight;
    let weighted_delta = (value - m.mean) * weight;
    let next_mean = m.mean + weighted_delta / weight_sum;
    return Moments(next_mean, m.m2 + weighted_delta * (value - next_mean), weight_sum, m.count + 1u);
}

fn without_sample(m: Moments, value: f32, weight: f32) -> Moments {
    let weight_sum = m.weight_sum - weight;
    if (m.count <= 1u || !finite(weight_sum) || weight_sum <= 0.0) {
        return Moments(0.0, 0.0, 0.0, 0u);
    }
    let other_mean = m.mean + weight * (m.mean - value) / weight_sum;
    return Moments(
        other_mean,
        max(m.m2 - weight * (value - m.mean) * (value - other_mean), 0.0),
        weight_sum,
        m.count - 1u,
    );
}

fn scaled(m: Moments, factor: f32) -> Moments {
    return Moments(m.mean, m.m2 * factor, m.weight_sum, m.count);
}

fn rejects(m: Moments, value: f32, weight: f32, inverse_weight: f32) -> bool {
    if (m.count < 2u || !finite(m.weight_sum) || m.weight_sum <= 0.0) {
        return false;
    }
    let peers = f32(m.count);
    var scale = 1.0;
    if (!(weight == 1.0 && m.weight_sum == peers)) {
        scale = sqrt((inverse_weight + 1.0 / m.weight_sum) / factors[m.count].x);
    }
    let sigma = max(
        max(sqrt(m.m2 / (peers - 1.0)) * scale, params.minimum_sigma),
        abs(m.mean) * params.precision_floor,
    );
    let limits = thresholds[m.count + 1u];
    let residual = value - m.mean;
    return residual < -limits.x * sigma || residual > limits.y * sigma;
}

struct First {
    moments: Moments,
    trimmed: bool,
    high: f32,
    low: f32,
    member_factor: f32,
    extreme_factor: f32,
}

fn keeps(first: First, value: f32, weight: f32, inverse_weight: f32) -> bool {
    var peers: Moments;
    if (first.trimmed && (value == first.high || value == first.low)) {
        peers = scaled(first.moments, first.extreme_factor);
    } else {
        peers = scaled(without_sample(first.moments, value, weight), first.member_factor);
    }
    return !rejects(peers, value, weight, inverse_weight);
}

fn weight_of(frame: u32, at: u32) -> vec2<f32> {
    return weights[frame * params.weight_channels + (at % params.channels) % params.weight_channels];
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    let at = id.y * groups.x * 256u + id.x;
    let frames = params.frames;
    if (at < params.samples) {
        // Pass 1: every finite sample, and the largest and smallest.
        var first = First(Moments(0.0, 0.0, 0.0, 0u), false, 0.0, 0.0, 1.0, 1.0);
        var high_weight = 0.0;
        var low_weight = 0.0;
        var any = false;
        for (var frame = 0u; frame < frames; frame++) {
            let value = values[frame * params.samples + at];
            if (!finite(value)) {
                continue;
            }
            let weight = weight_of(frame, at).x;
            first.moments = with_sample(first.moments, value, weight);
            if (!any || value > first.high) {
                first.high = value;
                high_weight = weight;
            }
            if (!any || value < first.low) {
                first.low = value;
                low_weight = weight;
            }
            any = true;
        }
        let count = first.moments.count;
        if (count >= TRIMMED_FIRST_ESTIMATE) {
            first.moments = without_sample(without_sample(first.moments, first.high, high_weight), first.low, low_weight);
            first.trimmed = true;
            first.member_factor = factors[count].y;
            first.extreme_factor = factors[count].z;
        }

        // Pass 2: the samples the first estimate keeps.
        var kept = Moments(0.0, 0.0, 0.0, 0u);
        for (var frame = 0u; frame < frames; frame++) {
            let value = values[frame * params.samples + at];
            if (!finite(value)) {
                continue;
            }
            let weight = weight_of(frame, at);
            if (keeps(first, value, weight.x, weight.y)) {
                kept = with_sample(kept, value, weight.x);
            }
        }

        // Pass 3: reject each against the others the second estimate kept,
        // and integrate the rest.
        var mean = 0.0;
        var m2 = 0.0;
        var weight_sum = 0.0;
        var coverage = 0u;
        var rejected = 0u;
        for (var frame = 0u; frame < frames; frame++) {
            let value = values[frame * params.samples + at];
            if (!finite(value)) {
                continue;
            }
            atomicAdd(&local_counts[2u * frame], 1u);
            let weight = weight_of(frame, at);
            var peers = kept;
            if (keeps(first, value, weight.x, weight.y)) {
                peers = without_sample(kept, value, weight.x);
            }
            if (rejects(scaled(peers, params.clipped_correction), value, weight.x, weight.y)) {
                rejected += 1u;
                continue;
            }
            coverage += 1u;
            atomicAdd(&local_counts[2u * frame + 1u], 1u);
            if (params.weighted == 1u) {
                let next_weight = weight_sum + weight.x;
                let weighted_delta = (value - mean) * weight.x;
                let next_mean = mean + weighted_delta / next_weight;
                m2 = m2 + weighted_delta * (value - next_mean);
                mean = next_mean;
                weight_sum = next_weight;
            } else {
                let delta = value - mean;
                let next_mean = mean + delta / f32(coverage);
                m2 = m2 + delta * (value - next_mean);
                mean = next_mean;
            }
        }
        moments_out[at] = vec4<f32>(mean, m2, weight_sum, 0.0);
        tallies_out[at] = vec2<u32>(coverage, rejected);
    }
    workgroupBarrier();
    for (var i = local; i < 2u * frames; i += 256u) {
        let count = atomicLoad(&local_counts[i]);
        if (count > 0u) {
            atomicAdd(&frame_counts[i], count);
        }
    }
}
