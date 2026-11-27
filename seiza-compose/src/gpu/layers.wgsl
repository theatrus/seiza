// The Combine stage: each layer stretches one band, colors it and blends it
// onto the layers beneath. Also the brush strokes that become masks.

@group(1) @binding(0) var tile: texture_2d_array<f32>;
// The masks, one per layer of the array, in surface pixels.
@group(2) @binding(0) var layer_masks: texture_2d_array<f32>;
// The samples of each layer that has heals of its own, in surface pixels.
@group(2) @binding(1) var healed_layers: texture_2d_array<f32>;
// The computed masks that are smoothed, in surface pixels.
@group(2) @binding(2) var smoothed_masks: texture_2d_array<f32>;

struct TileVertex {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    // Where the fragment is in the picture, in picture pixels.
    @location(1) doc: vec2f,
}

@vertex
fn tile_vertex(
    @builtin(vertex_index) index: u32,
    @location(0) rect: vec4f,
    @location(1) uv_rect: vec4f,
) -> TileVertex {
    let corner = vec2f(f32(index & 1u), f32((index >> 1u) & 1u));
    let doc = mix(rect.xy, rect.zw, corner);
    var out: TileVertex;
    out.position = vec4f(doc * globals.view.xy + globals.view.zw, 0.0, 1.0);
    out.uv = mix(uv_rect.xy, uv_rect.zw, corner);
    out.doc = doc;
    return out;
}

// ---- The layered stretch, as `seiza_stretch::ResolvedLayered` gives it.

// The value on the extended sRGB curve of linear light.
fn layered_encode(light: f32) -> f32 {
    if light <= 0.0031308 {
        return 12.92 * max(light, 0.0);
    }
    return 1.055 * pow(light, 1.0 / 2.4) - 0.055;
}

// The median anchor curve, and its slope.
fn anchor_rescaled(value: f32, curve: vec4f) -> f32 {
    return clamp((value - curve.x) / (1.0 - curve.x), 0.0, 1.0);
}

fn anchor_curve(value: f32, curve: vec4f) -> f32 {
    let r = anchor_rescaled(value, curve);
    let m = curve.y;
    let t = curve.z;
    let denominator = m * (t + r - 1.0) - t * r;
    if abs(denominator) <= 1.1920929e-7 {
        return r;
    }
    return (m - 1.0) * t * r / denominator;
}

fn anchor_slope(value: f32, curve: vec4f) -> f32 {
    let r = anchor_rescaled(value, curve);
    let m = curve.y;
    let t = curve.z;
    let denominator = m * (t + r - 1.0) - t * r;
    return (m - 1.0) * (t - 1.0) * m * t / (denominator * denominator * (1.0 - curve.x));
}

// The curve, continued into the headroom above its knee when there is a tail.
fn layered_curve(value: f32, step: LayeredStep) -> f32 {
    if step.tail.x > 0.5 {
        let above = value - step.tail.y;
        if above > step.tail.z {
            let u = log(above / step.tail.z);
            let light = step.tail.w + step.turn.y * u
                + (step.turn.x - step.turn.y) * step.turn.z * (1.0 - exp(-u / step.turn.z));
            return clamp(layered_encode(exp(light)), 0.0, step.turn.w);
        }
    }
    return anchor_curve(value, step.curve);
}

// One map of a layered stretch at picture position `doc`, interpolated
// between the centres of the map pixels around it; a NaN texel has no
// weight. NaN where all four are.
fn layered_map(step: LayeredStep, doc: vec2f, channel: i32) -> f32 {
    let size = step.maps.zw;
    let at = clamp(doc / 4.0 - 0.5, vec2f(0.0), size - 1.0);
    let low = vec2i(floor(at));
    let high = min(low + 1, vec2i(size) - 1);
    let t = at - vec2f(low);
    let layer = i32(step.maps.x);
    var sum = 0.0;
    var weight = 0.0;
    let corners = array<vec2i, 4>(low, vec2i(high.x, low.y), vec2i(low.x, high.y), high);
    let weights = array<f32, 4>(
        (1.0 - t.x) * (1.0 - t.y),
        t.x * (1.0 - t.y),
        (1.0 - t.x) * t.y,
        t.x * t.y,
    );
    for (var corner = 0; corner < 4; corner++) {
        let value = textureLoad(layered_maps, corners[corner], layer, 0)[channel];
        if value == value && weights[corner] > 0.0 {
            sum += value * weights[corner];
            weight += weights[corner];
        }
    }
    if weight > 0.0 {
        return sum / weight;
    }
    return bitcast<f32>(0x7fc00000u);
}

// The soft-knee highlight curve star cores are pulled toward.
fn star_core_curve(value: f32, knee: f32, end_slope: f32) -> f32 {
    if value <= knee {
        return value;
    }
    let width = 1.0 - knee;
    let t = min((value - knee) / width, 1.0);
    let t2 = t * t;
    let t3 = t2 * t;
    return knee + width * ((t3 - 2.0 * t2 + t) + (-2.0 * t3 + 3.0 * t2) + end_slope * (t3 - t2));
}

// A guide input's value at `uv`, from its layer of the input tiles.
fn layered_guide(uv: vec2f, layer: f32, scale: f32) -> f32 {
    return textureSampleLevel(tile, linear_sampler, uv, i32(layer), 0.0).r * scale;
}

// Layer `index`'s layered stretch of `value` at picture position `doc` and
// tile position `uv`, or `value` as it is when the layer has none.
fn layered(value: f32, index: i32, doc: vec2f, uv: vec2f) -> f32 {
    let at = i32(globals.layers[index].more.y);
    if at < 0 {
        return value;
    }
    let step = globals.layered[at];
    // The luminance the stretch reads: a guide's, which the channel then
    // follows, or the sample's own.
    var luminance = value;
    if step.guides.x >= 0.0 {
        luminance = layered_guide(uv, step.guides.x, step.guides.y);
    }
    var read = luminance;
    // The core compression: the part above the background scaled by one
    // smooth gain from the base.
    if step.detail.w > 0.5 {
        let base = layered_map(step, doc, 0);
        if base == base {
            let over = (base - step.core.x) / step.core.w;
            var soft = step.core.w * over;
            if over <= 20.0 {
                soft = step.core.w * log(1.0 + exp(over));
            }
            let gain = exp(-step.core.y * soft);
            read = step.curve.w + (luminance - step.curve.w) * gain;
        }
    }
    var out = layered_curve(read, step);
    // Contrast recovery: the detail around the large-scale layer keeps more
    // of the background's slope.
    if step.maps.y > 0.5 && step.detail.y > 0.0 {
        let large = layered_map(step, doc, 1);
        if large == large {
            let level = min(large, read);
            let gain = min(pow(step.detail.z / anchor_slope(level, step.curve), step.detail.y / 2.0), 4.0);
            let base = layered_curve(large, step);
            out = base + (out - base) * gain;
        }
    }
    // Star cores, against the luminance's fine blur.
    if step.cores.x > 0.0 && step.guides.z >= 0.0 {
        let fine = layered_guide(uv, step.guides.z, step.guides.w);
        let noise = max(step.detail.x, 1.0e-6);
        let mask = smoothstep(0.0, 1.0, clamp((luminance - fine - 3.0 * noise) / (12.0 * noise), 0.0, 1.0));
        out += step.cores.x * mask * (star_core_curve(out, step.cores.w, step.cores.y) - out);
    }
    // A channel that follows a luminance: its own value less the black,
    // scaled as the luminance was.
    if step.guides.x >= 0.0 {
        let black = step.cores.z;
        let base = max((luminance - black) / (1.0 - black), 0.0);
        if base > 1.0e-6 {
            // No less than one noise sigma, as `colour_floor` gives it.
            let floor = step.detail.x / (1.0 - black);
            let own = max((value - black) / (1.0 - black), 0.0);
            out = max(1.0 + (own - base) / max(base, floor), 0.0) * out;
        }
    }
    return clamp(out, 0.0, step.turn.w);
}

// A layer's stretch of a band's sample. The stretch itself is resolved by
// the host and arrives as a table; nothing here knows its form.
fn stretched(sample: f32, row: f32) -> f32 {
    let size = vec2f(textureDimensions(stretch_table));
    let at = vec2f(
        (sqrt(clamp(sample, 0.0, 1.0)) * (size.x - 1.0) + 0.5) / size.x,
        (row + 0.5) / size.y,
    );
    return textureSampleLevel(stretch_table, linear_sampler, at, 0.0).r;
}

// How a layer's color combines with what is beneath it, before opacity.
// The modes are in the order of the document's `BlendMode`.
fn blend(mode: f32, below: vec3f, layer: vec3f) -> vec3f {
    if mode < 0.5 {
        return layer;
    } else if mode < 1.5 {
        return 1.0 - (1.0 - clamp(below, vec3f(0.0), vec3f(1.0))) * (1.0 - clamp(layer, vec3f(0.0), vec3f(1.0)));
    } else if mode < 2.5 {
        return below + layer;
    } else if mode < 3.5 {
        return below * layer;
    } else if mode < 4.5 {
        return max(below, layer);
    } else if mode < 5.5 {
        return min(below, layer);
    }
    // Luminance: the layer's brightness with the color beneath it.
    let weights = globals.luma.xyz;
    return max(below + dot(layer, weights) - dot(below, weights), vec3f(0.0));
}

// What an adjustment layer makes of the color beneath it.
fn adjusted(color: vec3f, adjust: vec4f) -> vec3f {
    if adjust.x < 1.5 {
        return color * exp2(adjust.y);
    } else if adjust.x < 2.5 {
        return vec3f(
            contrast_curve(color.r, adjust.y),
            contrast_curve(color.g, adjust.y),
            contrast_curve(color.b, adjust.y),
        );
    }
    let luminance = dot(color, globals.luma.xyz);
    return max(mix(vec3f(luminance), color, adjust.y), vec3f(0.0));
}

// A band's sample in a tile, blended with its second band.
fn band_sample(uv: vec2f, band: f32, second: f32, amount: f32) -> f32 {
    var sample = textureSampleLevel(tile, linear_sampler, uv, i32(band), 0.0).r;
    if second >= 0.0 {
        let other = textureSampleLevel(tile, linear_sampler, uv, i32(second), 0.0).r;
        sample = mix(sample, other, amount);
    }
    return sample;
}

// A computed mask as its bands give it at a place in a tile: the product of
// their stretched samples, shaped and raised to the mask's reach.
fn computed_shape(index: i32, uv: vec2f) -> f32 {
    let mask = globals.computed[index];
    var product = 1.0;
    if mask.first.x >= 0.0 {
        product *= stretched(band_sample(uv, mask.first.x, mask.first.y, mask.first.z), mask.first.w);
    }
    if mask.second.x >= 0.0 {
        product *= stretched(band_sample(uv, mask.second.x, mask.second.y, mask.second.z), mask.second.w);
    }
    product = clamp(product, 0.0, 1.0);
    if mask.shape.x > 0.5 {
        product = pow(product, 1.0 - product);
    }
    return pow(product, mask.shape.z);
}

// A computed mask as a layer uses it: smoothed when it asks for that, so
// the bands' noise does not become speckle, then touched up by strokes that
// set it to 1 or to 0, then inverted or not.
fn computed_mask(index: i32, uv: vec2f, pixel: vec2i) -> f32 {
    let mask = globals.computed[index];
    var value = 0.0;
    if mask.touch.z > 0.0 {
        value = textureLoad(smoothed_masks, pixel, index, 0).r;
    } else {
        value = computed_shape(index, uv);
    }
    if mask.touch.x >= 0.0 {
        value = max(value, textureLoad(layer_masks, pixel, i32(mask.touch.x), 0).r);
    }
    if mask.touch.y >= 0.0 {
        value *= 1.0 - textureLoad(layer_masks, pixel, i32(mask.touch.y), 0).r;
    }
    if mask.shape.y > 0.5 {
        value = 1.0 - value;
    }
    return value;
}

struct EveryMask {
    @location(0) first: vec4f,
    @location(1) second: vec4f,
    @location(2) third: vec4f,
    @location(3) fourth: vec4f,
    @location(4) fifth: vec4f,
    @location(5) sixth: vec4f,
    @location(6) seventh: vec4f,
    @location(7) eighth: vec4f,
}

fn every_mask(values: array<f32, 8>) -> EveryMask {
    var out: EveryMask;
    out.first = vec4f(values[0]);
    out.second = vec4f(values[1]);
    out.third = vec4f(values[2]);
    out.fourth = vec4f(values[3]);
    out.fifth = vec4f(values[4]);
    out.sixth = vec4f(values[5]);
    out.seventh = vec4f(values[6]);
    out.eighth = vec4f(values[7]);
    return out;
}

// The computed masks that are smoothed, one per target, before the blurs.
@fragment
fn computed_fragment(in: TileVertex) -> EveryMask {
    var values = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (var index = 0; index < 8; index++) {
        if globals.computed[index].touch.z > 0.0 {
            values[index] = computed_shape(index, in.uv);
        }
    }
    return every_mask(values);
}

// One triangle over the whole target.
@vertex
fn whole_vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    let corner = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
}

// One direction of a Gaussian blur of each smoothed mask. A blur reads the
// masks through the tile binding. Thirty-three samples cover three standard
// deviations to each side; a wide blur spaces them further apart.
fn blurred(position: vec2f, direction: vec2f) -> EveryMask {
    let size = vec2f(textureDimensions(tile).xy);
    var values = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (var index = 0; index < 8; index++) {
        let sigma = globals.computed[index].touch.z;
        if sigma <= 0.0 {
            continue;
        }
        let spacing = sigma * 3.0 / 16.0;
        var sum = 0.0;
        var weights = 0.0;
        for (var tap = -16; tap <= 16; tap++) {
            let offset = f32(tap) * spacing;
            let weight = exp(-offset * offset / (2.0 * sigma * sigma));
            let at = (position + direction * offset) / size;
            sum += weight * textureSampleLevel(tile, linear_sampler, at, index, 0.0).r;
            weights += weight;
        }
        values[index] = sum / weights;
    }
    return every_mask(values);
}

@fragment
fn blur_across_fragment(@builtin(position) position: vec4f) -> EveryMask {
    return blurred(position.xy, vec2f(1.0, 0.0));
}

@fragment
fn blur_down_fragment(@builtin(position) position: vec4f) -> EveryMask {
    return blurred(position.xy, vec2f(0.0, 1.0));
}

struct HealedLayers {
    @location(0) first: vec4f,
    @location(1) second: vec4f,
    @location(2) third: vec4f,
    @location(3) fourth: vec4f,
}

// The samples of each layer that has heals of its own, one per target.
@fragment
fn healed_fragment(in: TileVertex) -> HealedLayers {
    var samples = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    for (var place = 0; place < 4; place++) {
        let healed = globals.healed[place];
        if healed.x >= 0.0 {
            samples[place] = band_sample(in.uv, healed.x, healed.y, healed.z);
        }
    }
    var out: HealedLayers;
    out.first = vec4f(samples[0]);
    out.second = vec4f(samples[1]);
    out.third = vec4f(samples[2]);
    out.fourth = vec4f(samples[3]);
    return out;
}

// A healed layer's sample at a surface pixel, with the heals of its band
// applied. A heal copies the sample as it was before any heal.
fn healed_sample(index: i32, place: i32, pixel: vec2i) -> f32 {
    let size = vec2i(globals.screen.xy);
    var sample = textureLoad(healed_layers, pixel, place, 0).r;
    let count = i32(globals.counts.z);
    for (var at = 0; at < count; at++) {
        let heal = globals.heals[at];
        if i32(heal.what.x) != index || heal.what.y > 0.5 {
            continue;
        }
        let source = pixel + vec2i(heal.offset.xy);
        if all(source >= vec2i(0)) && all(source < size) {
            let copy = textureLoad(healed_layers, source, place, 0).r + heal.what.w;
            sample = mix(sample, copy, textureLoad(layer_masks, pixel, i32(heal.what.z), 0).r);
        }
    }
    return sample;
}

// A healed layer's stretched value at a surface pixel, with the heals of
// the layer applied. A heal copies the value as it was before those heals.
// A layer's own contrast curve, when it has one; with none, the value as it
// is, as the CPU render leaves it.
fn layer_curve(value: f32, amount: f32) -> f32 {
    if amount == 0.0 {
        return value;
    }
    return contrast_curve(value, amount);
}

// A waiting layer blended over `color`. An added band layer is one fused
// multiply-add per channel, as the CPU render adds it.
fn flush(color: vec3f, own: vec3f, cover: f32, mode: f32, band: bool, tint: vec3f, value: f32) -> vec3f {
    if band && mode > 1.5 && mode < 2.5 && cover > 0.0 {
        if cover >= 1.0 {
            return fma(tint, vec3f(value), color);
        }
        return fma(tint * value, vec3f(cover), color);
    }
    if cover >= 1.0 {
        return blend(mode, color, own);
    }
    if cover <= 0.0 {
        return color;
    }
    return mix(color, blend(mode, color, own), cover);
}

// LRGB: a new luminance for `color`, its chromaticity kept. `replace`
// weighs the input's value against the color's own luminance by `cover`;
// otherwise the target is the input plus R + G + B, or that sum alone.
fn luminance_target(color: vec3f, value: f32, has: bool, replace: bool, cover: f32) -> vec3f {
    let own = dot(color, globals.luma.xyz);
    var wanted = own;
    if replace {
        if has {
            wanted = fma(1.0 - cover, own, cover * value);
        }
    } else if has {
        wanted = value + color.r + color.g + color.b;
    } else {
        wanted = color.r + color.g + color.b;
    }
    var scaled = vec3f(wanted);
    if own > 1.0e-8 {
        scaled = color * (wanted / own);
    }
    if replace || cover >= 1.0 {
        return scaled;
    }
    return mix(color, scaled, cover);
}

fn healed_value(index: i32, pixel: vec2i, doc: vec2f, uv: vec2f) -> f32 {
    let layer = globals.layers[index];
    let place = i32(layer.stretch.y);
    let size = vec2i(globals.screen.xy);
    var value = layer_curve(
        layered(stretched(healed_sample(index, place, pixel), layer.stretch.x), index, doc, uv),
        layer.stretch.w,
    );
    let count = i32(globals.counts.z);
    for (var at = 0; at < count; at++) {
        let heal = globals.heals[at];
        if i32(heal.what.x) != index || heal.what.y < 0.5 {
            continue;
        }
        let source = pixel + vec2i(heal.offset.xy);
        if all(source >= vec2i(0)) && all(source < size) {
            // The maps are smooth, so the copy reads them where it lands.
            let copy = layer_curve(
                layered(stretched(healed_sample(index, place, source), layer.stretch.x), index, doc, uv),
                layer.stretch.w,
            ) + heal.what.w;
            value = mix(value, copy, textureLoad(layer_masks, pixel, i32(heal.what.z), 0).r);
        }
    }
    return value;
}

@fragment
fn tile_fragment(in: TileVertex) -> @location(0) vec4f {
    // The layers blended so far.
    var color = vec3f(0.0);
    // The layers beneath a group, kept while the group's members blend on
    // their own.
    var beneath = vec3f(0.0);
    // The last band layer or group, which waits to be blended until its own
    // adjustments have been applied to it.
    var waiting = false;
    var own = vec3f(0.0);
    // The waiting layer's tint and value, when it shows a band.
    var own_band = false;
    var own_tint = vec3f(0.0);
    var own_value = 0.0;
    var own_cover = 0.0;
    var own_mode = 0.0;
    let pixel = vec2i(in.position.xy);
    let count = i32(globals.counts.x);
    for (var index = 0; index < count; index++) {
        let layer = globals.layers[index];
        // How much of the layer applies here: its opacity, within its mask
        // and within the mask of the layer it is clipped to.
        var cover = layer.color.a;
        if layer.adjust.z >= 0.0 {
            cover *= textureLoad(layer_masks, pixel, i32(layer.adjust.z), 0).r;
        }
        if layer.stretch.z >= 0.0 {
            cover *= textureLoad(layer_masks, pixel, i32(layer.stretch.z), 0).r;
        }
        if layer.more.x >= 0.0 {
            cover *= computed_mask(i32(layer.more.x), in.uv, pixel);
        }
        let kind = layer.adjust.x;
        if kind > 0.5 && kind < 9.5 && layer.adjust.w > 0.5 {
            // An adjustment of the waiting layer or group alone.
            if waiting {
                own = mix(own, adjusted(own, layer.adjust), cover);
                own_band = false;
            }
            continue;
        }
        if waiting {
            color = flush(color, own, own_cover, own_mode, own_band, own_tint, own_value);
            waiting = false;
        }
        if kind > 12.5 {
            // A luminance layer: 13 replaces, 14 is super-luminance.
            let has = layer.select.x >= 0.0;
            var value = 0.0;
            if has {
                value = stretched(
                    band_sample(in.uv, layer.select.x, layer.select.z, layer.select.w),
                    layer.stretch.x,
                );
            }
            color = luminance_target(color, value, has, kind < 13.5, cover);
            continue;
        }
        if kind > 11.5 {
            // A layer of its color everywhere.
            own = layer.color.rgb;
            own_band = false;
            own_cover = cover;
            own_mode = layer.select.y;
            waiting = true;
            continue;
        }
        if kind > 10.5 {
            // The end of a group: its members' blend is one layer now.
            own = color;
            own_band = false;
            own_cover = cover;
            own_mode = layer.select.y;
            waiting = true;
            color = beneath;
            continue;
        }
        if kind > 9.5 {
            // The start of a group: its members blend from nothing.
            beneath = color;
            color = vec3f(0.0);
            continue;
        }
        if kind > 0.5 {
            color = mix(color, adjusted(color, layer.adjust), cover);
            continue;
        }
        var value = 0.0;
        if layer.stretch.y >= 0.0 {
            value = healed_value(index, pixel, in.doc, in.uv);
        } else {
            let sample = band_sample(in.uv, layer.select.x, layer.select.z, layer.select.w);
            value = layer_curve(layered(stretched(sample, layer.stretch.x), index, in.doc, in.uv), layer.stretch.w);
        }
        own = layer.color.rgb * value;
        own_band = true;
        own_tint = layer.color.rgb;
        own_value = value;
        own_cover = cover;
        own_mode = layer.select.y;
        waiting = true;
    }
    if waiting {
        color = flush(color, own, own_cover, own_mode, own_band, own_tint, own_value);
    }
    return vec4f(color, 1.0);
}

struct StrokeVertex {
    @builtin(position) position: vec4f,
    @location(0) doc: vec2f,
    @location(1) @interpolate(flat) ends: vec4f,
    @location(2) @interpolate(flat) brush: vec4f,
}

// One brush segment, drawn as a quad around a capsule.
@vertex
fn stroke_vertex(
    @builtin(vertex_index) index: u32,
    @location(0) ends: vec4f,
    @location(1) brush: vec4f,
) -> StrokeVertex {
    let corner = vec2f(f32(index & 1u), f32((index >> 1u) & 1u)) * 2.0 - 1.0;
    let delta = ends.zw - ends.xy;
    let length = length(delta);
    var along = vec2f(1.0, 0.0);
    if length > 0.0001 {
        along = delta / length;
    }
    let across = vec2f(-along.y, along.x);
    let radius = brush.x;
    let doc = mix(ends.xy, ends.zw, corner.x * 0.5 + 0.5)
        + along * corner.x * radius
        + across * corner.y * radius;
    var out: StrokeVertex;
    out.position = vec4f(doc * globals.view.xy + globals.view.zw, 0.0, 1.0);
    out.doc = doc;
    out.ends = ends;
    out.brush = brush;
    return out;
}

@fragment
fn stroke_fragment(in: StrokeVertex) -> @location(0) vec4f {
    let delta = in.ends.zw - in.ends.xy;
    let along = clamp(
        dot(in.doc - in.ends.xy, delta) / max(dot(delta, delta), 0.0001),
        0.0,
        1.0,
    );
    let distance = distance(in.doc, in.ends.xy + delta * along);
    let radius = in.brush.x;
    let coverage = 1.0 - smoothstep(radius * in.brush.y, radius, distance);
    // brush.z is 1 to add to the mask and 0 to subtract from it.
    return vec4f(in.brush.z, 0.0, 0.0, coverage);
}
