//! The layered stretch: a median anchor curve on luminance that treats
//! large and fine structure apart.
//!
//! A blur splits luminance into a large-scale layer and the detail above it.
//! The large layer takes the global curve; detail keeps the curve's change
//! across it, with more gain where the curve flattens on bright structure.
//! Star cores can be compressed on their own, and colour follows luminance.
//!
//! The HDR layered stretch, [`LayeredHdrParams`], adds two steps for a
//! picture shown above SDR white. Before the curve, bright structure's
//! large-scale range is compressed in the log of luminance while its detail
//! is kept, as PixInsight's HDR multiscale transform does; luminance carries
//! one gain to every channel, so colour is kept. Above a knee the curve then
//! continues into the headroom, reaching its top at the image's brightest
//! large-scale luminance. The brightest object, such as a planetary
//! nebula's core, keeps its structure, and the sky and faint halo are as
//! the plain layered stretch makes them.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    HDR_MAX_EXTRA_SLOPE, HdrTail, HdrTailParams, ImageShape, LUMA_BLUE, LUMA_GREEN, LUMA_RED,
    MedianAnchorParams, ResolvedCurve, ResolvedMedianAnchor, Result, RobustStatistics,
    StretchError, compress_highlights,
};

/// Parameters for the layered stretch.
///
/// Luminance `L` takes the median anchor curve `S`, its black point
/// `6 − 4.5·aggressiveness` robust sigmas below the median. A blur of
/// `scale_separation / 4` pixels splits `L` into a large-scale layer `B`.
/// Contrast recovery `c` gives detail on bright structure more of the
/// background's slope: `Y = S(B) + (S(L) − S(B))·g`, with
/// `g = min(4, (S'(m) / S'(min(L, B)))^(c/2))` for the median `m`. Dynamic
/// range compression pulls star cores, found in a 2-pixel detail layer,
/// toward a soft-knee highlight curve. Colour follows luminance, and an
/// optional saturation lift leaves hue and value alone.
///
/// In JSON every field may be left out and takes its default.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LayeredParams {
    /// Where the background median lands, in `(0, 1)` (default 0.15).
    pub target_background: f64,
    /// How close the black point sits to the median, in `[0, 1]` (default
    /// 0.5): 6 robust sigmas below it at 0, 1.5 at 1.
    pub aggressiveness: f64,
    /// How far star cores are compressed, in `[0, 1]` (default 0).
    pub dynamic_range_compression: f64,
    /// Size in pixels that splits large structure from detail, from 8
    /// through 1024 (default 256).
    pub scale_separation: u32,
    /// How much of the background's slope detail on bright structure keeps,
    /// in `[0, 1]` (default 0).
    pub contrast_recovery: f64,
    /// Saturation lift, in `[0, 1]` (default 0).
    pub saturation_amount: f64,
    /// How much the lift favours weakly coloured pixels, in `[0, 1]`
    /// (default 0.5).
    pub saturation_boost: f64,
    /// Scale the saturation lift by each pixel's value, so dark noise gains
    /// little colour (default on).
    pub lightness_mask: bool,
}

impl Default for LayeredParams {
    fn default() -> Self {
        Self {
            target_background: 0.15,
            aggressiveness: 0.5,
            dynamic_range_compression: 0.0,
            scale_separation: 256,
            contrast_recovery: 0.0,
            saturation_amount: 0.0,
            saturation_boost: 0.5,
            lightness_mask: true,
        }
    }
}

/// What the HDR layered stretch adds to the layered stretch: how far above
/// SDR white it reaches, and how bright structure's range is compressed.
///
/// With `z` luminance above the background median in robust sigmas, the
/// gain on `z` is `exp(−core_compression · w · softplus((B − ln(T + 3)) / w))`,
/// where `B` is a blur of `core_scale` pixels of `ln(z + 3)`, `T` is
/// `core_threshold` and `w` is half a natural-log unit. Structure smaller than
/// the blur keeps its contrast. The curve's output above `knee` continues
/// on the extended sRGB curve as [`HdrTail`] does, reaching 2 to the
/// `headroom` times SDR white's light at the image's brightest luminance
/// after the compression, blurred by a pixel.
///
/// In JSON every field may be left out and takes its default.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LayeredHdr {
    /// How far above SDR white the brightest structure reaches, in stops,
    /// from 0 through 4 (default 2). At 0 the curve rolls off to white.
    pub headroom: f64,
    /// The curve's output where it starts to continue into the headroom,
    /// from 0.3 through 0.95 (default 0.75).
    pub knee: f64,
    /// How much of bright structure's large-scale range in the log of
    /// luminance is taken out, in `[0, 1]` (default 0.7). 0 compresses
    /// nothing.
    pub core_compression: f64,
    /// Where compression starts, in robust sigmas of luminance above the
    /// background (default 100).
    pub core_threshold: f64,
    /// The size of the large-scale structure that is compressed, as a blur
    /// in pixels, from 1 through 256 (default 12). About a third of the
    /// bright object's radius.
    pub core_scale: f64,
}

impl Default for LayeredHdr {
    fn default() -> Self {
        Self {
            headroom: 2.0,
            knee: 0.75,
            core_compression: 0.7,
            core_threshold: 100.0,
            core_scale: 12.0,
        }
    }
}

/// Parameters for the HDR layered stretch: the layered stretch's, and what
/// HDR adds. The layered stretch's dynamic range compression of star cores
/// is not used: the core compression takes its place.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LayeredHdrParams {
    #[serde(flatten)]
    pub layered: LayeredParams,
    #[serde(flatten)]
    pub hdr: LayeredHdr,
}

/// The floor under `z` in the log, in robust sigmas, so noise near the
/// background has no log to speak of.
pub const CORE_FLOOR: f64 = 3.0;
/// The width of the compression's knee, in natural-log units.
pub const CORE_KNEE_WIDTH: f64 = 0.5;
/// The most of the white's curve value the HDR tail's knee may be, so the
/// tail has room to rise to the top of the headroom.
const KNEE_BELOW_WHITE: f64 = 0.8;

/// The most contrast recovery multiplies detail by.
pub const MAX_DETAIL_GAIN: f64 = 4.0;
/// Blur sigma in pixels of the fine layer that finds star cores.
const FINE_SIGMA: f64 = 2.0;
/// Knee of the highlight curve the compression pulls toward.
const DRC_KNEE: f64 = 0.5;
/// Below this, a pixel's rescaled luminance is too dark to carry colour.
const COLOUR_EPSILON: f64 = 1.0e-6;

/// A layered stretch with its image-dependent values resolved.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResolvedLayered {
    /// The requested parameters.
    pub params: LayeredParams,
    /// The global curve `S`.
    pub curve: ResolvedMedianAnchor,
    /// Luminance median `m`.
    pub median: f64,
    /// Robust luminance noise, from the lower half of the samples.
    pub noise: f64,
    /// What the HDR layered stretch adds, or none for the layered stretch.
    #[serde(default)]
    pub hdr: Option<LayeredHdr>,
}

impl ResolvedLayered {
    pub(crate) fn new(
        statistics: RobustStatistics,
        noise: f64,
        params: LayeredParams,
        hdr: Option<LayeredHdr>,
    ) -> Result<Self> {
        validate(params)?;
        if let Some(hdr) = hdr {
            validate_hdr(hdr)?;
        }
        let curve = ResolvedMedianAnchor::new(
            statistics,
            noise,
            MedianAnchorParams {
                target_median: params.target_background,
                black_point_sigma: Some(6.0 - 4.5 * params.aggressiveness),
                curves_boost: 0.0,
                hdr_amount: 0.0,
                hdr_knee: MedianAnchorParams::default().hdr_knee,
                normalize: false,
            },
        )?;
        Ok(Self {
            params,
            curve,
            median: statistics.median,
            noise,
            hdr,
        })
    }

    pub(crate) fn is_valid(&self) -> bool {
        validate(self.params).is_ok()
            && self.hdr.is_none_or(|hdr| validate_hdr(hdr).is_ok())
            && self.curve.is_valid()
            && self.median.is_finite()
            && self.noise.is_finite()
            && self.noise >= 0.0
    }

    /// The global curve's slope `S'(x)`, from its rational form. Outside the
    /// black point and 1, the slope at the nearer end.
    pub fn slope(&self, value: f64) -> f64 {
        let black = f64::from(self.curve.black);
        let rescaled = ((value - black) / (1.0 - black)).clamp(0.0, 1.0);
        let (m, t) = (self.curve.median, self.curve.target);
        let denominator = m * (t + rescaled - 1.0) - t * rescaled;
        (m - 1.0) * (t - 1.0) * m * t / (denominator * denominator * (1.0 - black))
    }
}

fn validate(params: LayeredParams) -> Result<()> {
    let unit = |value: f64| value.is_finite() && (0.0..=1.0).contains(&value);
    if !(params.target_background.is_finite()
        && params.target_background > 0.0
        && params.target_background < 1.0)
    {
        return Err(StretchError::Invalid(
            "target background must be finite and between zero and one".into(),
        ));
    }
    for (value, name) in [
        (params.aggressiveness, "aggressiveness"),
        (
            params.dynamic_range_compression,
            "dynamic range compression",
        ),
        (params.contrast_recovery, "contrast recovery"),
        (params.saturation_amount, "saturation amount"),
        (params.saturation_boost, "saturation boost"),
    ] {
        if !unit(value) {
            return Err(StretchError::Invalid(format!(
                "{name} must be between 0 and 1"
            )));
        }
    }
    if !(8..=1024).contains(&params.scale_separation) {
        return Err(StretchError::Invalid(
            "scale separation must be from 8 through 1024 pixels".into(),
        ));
    }
    Ok(())
}

fn validate_hdr(hdr: LayeredHdr) -> Result<()> {
    let within =
        |value: f64, low: f64, high: f64| value.is_finite() && (low..=high).contains(&value);
    if !within(hdr.headroom, 0.0, 4.0) {
        return Err(StretchError::Invalid(
            "headroom must be from 0 through 4 stops".into(),
        ));
    }
    if !within(hdr.knee, 0.3, 0.95) {
        return Err(StretchError::Invalid(
            "the knee must be from 0.3 through 0.95".into(),
        ));
    }
    if !within(hdr.core_compression, 0.0, 1.0) {
        return Err(StretchError::Invalid(
            "core compression must be between 0 and 1".into(),
        ));
    }
    if !(hdr.core_threshold.is_finite() && hdr.core_threshold > 0.0) {
        return Err(StretchError::Invalid(
            "the core threshold must be above 0".into(),
        ));
    }
    if !within(hdr.core_scale, 1.0, 256.0) {
        return Err(StretchError::Invalid(
            "the core scale must be from 1 through 256 pixels".into(),
        ));
    }
    Ok(())
}

/// How many times smaller than the image the layered stretch's large-scale
/// maps are along each side. The structure they hold is smooth, so a
/// quarter of the resolution keeps it, and a host can measure the maps of a
/// whole image fast enough to follow a slider.
pub const MAP_REDUCTION: usize = 4;

/// The large-scale structure a layered stretch reads, measured once from the
/// whole image at [`MAP_REDUCTION`] times less resolution: what makes the
/// stretch depend on more than each pixel. Each map pixel stands for the
/// mean of a square of image pixels; between them a reader interpolates.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LayeredMaps {
    /// The maps' width and height in map pixels.
    pub width: usize,
    pub height: usize,
    /// For the HDR layered stretch's core compression: the blurred log of
    /// luminance above the background, in noise sigmas. Empty when the
    /// stretch compresses nothing.
    pub core: Vec<f32>,
    /// For contrast recovery: the large-scale layer of luminance, after the
    /// core compression. Empty without contrast recovery.
    pub large: Vec<f32>,
    /// The brightest large-scale luminance after the core compression: the
    /// HDR tail's white. 0 for the plain layered stretch.
    pub white: f32,
}

/// A layered stretch's maps at one image pixel: the core compression's base
/// and the large-scale layer, each `None` where the stretch has no such map.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MapSample {
    pub core: Option<f32>,
    pub large: Option<f32>,
}

impl LayeredMaps {
    /// The maps at image pixel `(x, y)`, interpolated between the centres
    /// of the map pixels around it.
    pub fn sample(&self, x: usize, y: usize) -> MapSample {
        let at = |plane: &[f32]| {
            (!plane.is_empty()).then(|| bilinear(plane, self.width, self.height, x, y))
        };
        MapSample {
            core: at(&self.core),
            large: at(&self.large),
        }
    }
}

/// Luminance with every `MAP_REDUCTION` square of pixels averaged, its
/// finite samples only; a square with none is NaN. Returns the reduced
/// width and height too.
pub fn reduce_luminance(
    luminance: &[f32],
    width: usize,
    height: usize,
) -> (Vec<f32>, usize, usize) {
    let (reduced_width, reduced_height) = (
        width.div_ceil(MAP_REDUCTION),
        height.div_ceil(MAP_REDUCTION),
    );
    let reduced = (0..reduced_width * reduced_height)
        .into_par_iter()
        .map(|at| {
            let (column, row) = (at % reduced_width, at / reduced_width);
            let (mut sum, mut count) = (0.0_f64, 0_u32);
            for y in row * MAP_REDUCTION..((row + 1) * MAP_REDUCTION).min(height) {
                for x in column * MAP_REDUCTION..((column + 1) * MAP_REDUCTION).min(width) {
                    let value = luminance[y * width + x];
                    if value.is_finite() {
                        sum += f64::from(value);
                        count += 1;
                    }
                }
            }
            if count == 0 {
                f32::NAN
            } else {
                (sum / f64::from(count)) as f32
            }
        })
        .collect();
    (reduced, reduced_width, reduced_height)
}

/// `plane`, `width` by `height` map pixels, at image pixel `(x, y)`: each
/// map pixel's centre is at the middle of the square it averages. A NaN
/// neighbour has no weight.
fn bilinear(plane: &[f32], width: usize, height: usize, x: usize, y: usize) -> f32 {
    let scale = MAP_REDUCTION as f64;
    let fx = ((x as f64 + 0.5) / scale - 0.5).clamp(0.0, (width - 1) as f64);
    let fy = ((y as f64 + 0.5) / scale - 0.5).clamp(0.0, (height - 1) as f64);
    let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(width - 1), (y0 + 1).min(height - 1));
    let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
    let (mut sum, mut weight) = (0.0, 0.0);
    for (px, py, w) in [
        (x0, y0, (1.0 - tx) * (1.0 - ty)),
        (x1, y0, tx * (1.0 - ty)),
        (x0, y1, (1.0 - tx) * ty),
        (x1, y1, tx * ty),
    ] {
        let value = plane[py * width + px];
        if value.is_finite() && w > 0.0 {
            sum += f64::from(value) * w;
            weight += w;
        }
    }
    if weight > 0.0 {
        (sum / weight) as f32
    } else {
        f32::NAN
    }
}

impl ResolvedLayered {
    /// The stretch's maps, measured from `reduced`: luminance averaged by
    /// [`reduce_luminance`], `width` by `height` map pixels.
    pub fn measure(&self, reduced: &[f32], width: usize, height: usize) -> LayeredMaps {
        let scale = MAP_REDUCTION as f64;
        let compressing = self.hdr.filter(|hdr| hdr.core_compression > 0.0);
        let core = match compressing {
            Some(hdr) => {
                let logs: Vec<f32> = reduced
                    .par_iter()
                    .map(|&value| {
                        if value.is_finite() {
                            self.core_log(f64::from(value)) as f32
                        } else {
                            f32::NAN
                        }
                    })
                    .collect();
                blur(&logs, width, height, (hdr.core_scale / scale).max(0.25))
            }
            None => Vec::new(),
        };
        let compressed: Vec<f32> = if core.is_empty() {
            reduced.to_vec()
        } else {
            reduced
                .par_iter()
                .zip(&core)
                .map(|(&value, &base)| self.compress(value, Some(base)))
                .collect()
        };
        let white = if self.hdr.is_some() {
            compressed
                .iter()
                .copied()
                .filter(|value| value.is_finite())
                .fold(0.0_f32, f32::max)
        } else {
            0.0
        };
        let large = if self.params.contrast_recovery > 0.0 {
            blur(
                &compressed,
                width,
                height,
                (f64::from(self.params.scale_separation) / 4.0 / scale).max(0.25),
            )
        } else {
            Vec::new()
        };
        LayeredMaps {
            width,
            height,
            core,
            large,
            white,
        }
    }

    /// The log the core compression blurs: luminance above the background
    /// in noise sigmas, with a floor.
    pub fn core_log(&self, value: f64) -> f64 {
        ((value - self.median).max(0.0) / self.noise.max(1.0e-9) + CORE_FLOOR).ln()
    }

    /// `value` with bright structure's range compressed, given the core
    /// compression's base at its pixel; as it is without one.
    pub fn compress(&self, value: f32, base: Option<f32>) -> f32 {
        let (Some(hdr), Some(base)) = (self.hdr, base) else {
            return value;
        };
        if !value.is_finite() || !base.is_finite() || hdr.core_compression <= 0.0 {
            return value;
        }
        let threshold = (hdr.core_threshold + CORE_FLOOR).ln();
        let over = (f64::from(base) - threshold) / CORE_KNEE_WIDTH;
        let soft = if over > 20.0 {
            CORE_KNEE_WIDTH * over
        } else {
            CORE_KNEE_WIDTH * over.exp().ln_1p()
        };
        let gain = (-hdr.core_compression * soft).exp();
        (self.median + (f64::from(value) - self.median) * gain) as f32
    }

    /// The HDR tail the stretch continues its curve with, for the white its
    /// maps found; none for the plain layered stretch. When the curve gives
    /// the white less than the knee, as after strong core compression, the
    /// knee comes down below it, so the brightest structure still reaches
    /// the top of the headroom.
    pub fn tail(&self, maps: &LayeredMaps) -> Option<HdrTail> {
        let hdr = self.hdr?;
        let curve = ResolvedCurve::MedianAnchor(self.curve);
        let at_white = f64::from(curve.map(maps.white));
        let knee = hdr.knee.min(KNEE_BELOW_WHITE * at_white);
        HdrTail::resolve(
            curve,
            HdrTailParams {
                knee,
                headroom: hdr.headroom,
                white: f64::from(maps.white),
                ..HdrTailParams::default()
            },
        )
    }

    /// The brightest value the stretch gives: the tail's peak, or 1.
    pub fn peak(&self, tail: Option<&HdrTail>) -> f32 {
        tail.map_or(1.0, HdrTail::peak)
    }

    /// The stretched luminance of one pixel whose luminance is `value` and
    /// whose maps are `at`, before star-core compression and colour: the
    /// core compressed, the curve (with `tail` above its knee), and the
    /// detail around the large-scale layer kept.
    pub fn stretch_luminance(&self, value: f32, at: MapSample, tail: Option<&HdrTail>) -> f64 {
        let curve = |value: f32| match tail {
            Some(tail) => f64::from(tail.map(ResolvedCurve::MedianAnchor(self.curve), value)),
            None => self.curve.map(f64::from(value)),
        };
        let read = self.compress(value, at.core);
        match at.large.filter(|large| large.is_finite()) {
            Some(large) if self.params.contrast_recovery > 0.0 => {
                // Detail darker than its surroundings takes the slope at its
                // own level, so background beside bright structure is not
                // pushed toward black.
                let level = f64::from(large.min(read));
                let gain = (self.slope(self.median) / self.slope(level))
                    .powf(self.params.contrast_recovery / 2.0)
                    .min(MAX_DETAIL_GAIN);
                let base = curve(large);
                base + (curve(read) - base) * gain
            }
            _ => curve(read),
        }
    }
}

/// Run the layered stretch over a mono or RGB image, already checked
/// against `shape`. Non-finite samples come out 0.
pub(crate) fn apply(layered: &ResolvedLayered, data: &[f32], shape: ImageShape) -> Vec<f32> {
    let ImageShape {
        width,
        height,
        channels,
    } = shape;
    let luminance = luminance(data, channels);
    let params = layered.params;
    // The large-scale structure, measured at a quarter of the resolution,
    // as a host measures it for an interactive canvas.
    let (reduced, reduced_width, reduced_height) = reduce_luminance(&luminance, width, height);
    let maps = layered.measure(&reduced, reduced_width, reduced_height);
    let tail = layered.tail(&maps);
    let peak = f64::from(layered.peak(tail.as_ref()));
    let mut stretched: Vec<f64> = luminance
        .par_iter()
        .enumerate()
        .map(|(at, &value)| {
            if value.is_finite() {
                layered.stretch_luminance(value, maps.sample(at % width, at / width), tail.as_ref())
            } else {
                0.0
            }
        })
        .collect();

    if params.dynamic_range_compression > 0.0 && layered.hdr.is_none() {
        let fine = blur(&luminance, width, height, FINE_SIGMA);
        let noise = layered.noise.max(1.0e-6);
        let end_slope = 1.0 + HDR_MAX_EXTRA_SLOPE;
        let amount = params.dynamic_range_compression;
        stretched
            .par_iter_mut()
            .zip(luminance.par_iter().zip(&fine))
            .for_each(|(out, (&value, &smooth))| {
                if !value.is_finite() {
                    return;
                }
                let detail = f64::from(value) - f64::from(smooth);
                let mask = smoothstep(((detail - 3.0 * noise) / (12.0 * noise)).clamp(0.0, 1.0));
                *out += amount * mask * (compress_highlights(*out, DRC_KNEE, end_slope) - *out);
            });
    }

    if channels == 1 {
        return stretched
            .into_par_iter()
            .map(|value| value.clamp(0.0, peak) as f32)
            .collect();
    }

    let black = f64::from(layered.curve.black);
    let mut output = vec![0.0_f32; data.len()];
    output
        .par_chunks_mut(3 * width)
        .zip(data.par_chunks(3 * width))
        .zip(stretched.par_chunks(width).zip(luminance.par_chunks(width)))
        .for_each(|((output, input), (stretched, luminance))| {
            for (((output, pixel), &value), &luma) in output
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(input.as_chunks::<3>().0)
                .zip(stretched)
                .zip(luminance)
            {
                if !luma.is_finite() {
                    continue;
                }
                let mut rgb = colour(pixel, luma, value.clamp(0.0, peak), black, peak);
                if params.saturation_amount > 0.0 {
                    saturate(&mut rgb, params);
                }
                for (output, value) in output.iter_mut().zip(rgb) {
                    *output = value.clamp(0.0, peak) as f32;
                }
            }
        });
    output
}

/// Rec.709 luminance for RGB, the channel for mono; NaN where any sample of
/// a pixel is not finite.
fn luminance(data: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        return data
            .par_iter()
            .map(|&value| if value.is_finite() { value } else { f32::NAN })
            .collect();
    }
    data.par_chunks_exact(3)
        .map(|pixel| {
            if pixel.iter().all(|value| value.is_finite()) {
                (LUMA_RED * f64::from(pixel[0])
                    + LUMA_GREEN * f64::from(pixel[1])
                    + LUMA_BLUE * f64::from(pixel[2])) as f32
            } else {
                f32::NAN
            }
        })
        .collect()
}

/// Give a pixel the stretched luminance `value`: its RGB, less the black
/// point, scaled by one factor so its luminance does the same, and brought
/// back within `peak` by its largest channel.
fn colour(pixel: &[f32], luma: f32, value: f64, black: f64, peak: f64) -> [f64; 3] {
    let span = 1.0 - black;
    let rescale = |sample: f32| ((f64::from(sample) - black) / span).max(0.0);
    let base = rescale(luma);
    if base <= COLOUR_EPSILON {
        return [value; 3];
    }
    let scale = value / base;
    let mut rgb = [
        rescale(pixel[0]) * scale,
        rescale(pixel[1]) * scale,
        rescale(pixel[2]) * scale,
    ];
    let maximum = rgb[0].max(rgb[1]).max(rgb[2]);
    if maximum > peak {
        for channel in &mut rgb {
            *channel *= peak / maximum;
        }
    }
    rgb
}

/// Lift HSV saturation `s` to `s + A·(1 − s)·s^e`, which stays monotonic
/// for `A ≤ 1`, keeping hue and value. Grey and black pixels stay as they
/// are.
fn saturate(rgb: &mut [f64; 3], params: LayeredParams) {
    let value = rgb[0].max(rgb[1]).max(rgb[2]);
    let minimum = rgb[0].min(rgb[1]).min(rgb[2]);
    if value <= 0.0 || value <= minimum {
        return;
    }
    let saturation = (value - minimum) / value;
    let mut amount = params.saturation_amount;
    if params.lightness_mask {
        amount *= value.min(1.0);
    }
    let exponent = 2.0 * (1.0 - params.saturation_boost);
    let lifted = (saturation + amount * (1.0 - saturation) * saturation.powf(exponent)).min(1.0);
    // Each channel's distance below the value is a share of the saturation
    // set by the hue, so scaling every distance by one ratio keeps the hue.
    let ratio = lifted / saturation;
    for channel in rgb.iter_mut() {
        *channel = value - (value - *channel) * ratio;
    }
}

fn smoothstep(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

/// Box widths for three passes that approximate a Gaussian of `sigma`.
fn box_radii(sigma: f64) -> [usize; 3] {
    let passes = 3.0;
    let ideal = (12.0 * sigma * sigma / passes + 1.0).sqrt();
    let mut lower = ideal.floor().max(1.0);
    if lower % 2.0 == 0.0 {
        lower -= 1.0;
    }
    let upper = lower + 2.0;
    let lower_count =
        ((12.0 * sigma * sigma - passes * lower * lower - 4.0 * passes * lower - 3.0 * passes)
            / (-4.0 * lower - 4.0))
            .round();
    let mut radii = [0; 3];
    for (index, radius) in radii.iter_mut().enumerate() {
        let width = if (index as f64) < lower_count {
            lower
        } else {
            upper
        };
        *radius = (width as usize - 1) / 2;
    }
    radii
}

/// A Gaussian-like blur of `sigma` pixels: three box passes along rows,
/// then along columns, each a running sum, so the cost does not grow with
/// the radius. Non-finite samples and the space past the edges carry no
/// weight; the result is divided by the weight that remains.
fn blur(values: &[f32], width: usize, height: usize, sigma: f64) -> Vec<f32> {
    let radii = box_radii(sigma);
    let any_missing = values.par_iter().any(|value| !value.is_finite());
    let filled = values
        .par_iter()
        .map(|&value| if value.is_finite() { value } else { 0.0 })
        .collect::<Vec<_>>();
    let mut blurred = separable_boxes(filled, width, height, radii);
    if any_missing {
        let weights = values
            .par_iter()
            .map(|value| f32::from(u8::from(value.is_finite())))
            .collect::<Vec<_>>();
        let weights = separable_boxes(weights, width, height, radii);
        blurred
            .par_iter_mut()
            .zip(&weights)
            .for_each(|(value, &weight)| {
                *value = if weight > 0.0 {
                    *value / weight
                } else {
                    f32::NAN
                };
            });
    } else {
        // Every sample counts, so the weight is a row profile times a column
        // profile: only the edges lose any.
        let row = box_passes_owned(vec![1.0; width], radii);
        let column = box_passes_owned(vec![1.0; height], radii);
        blurred
            .par_chunks_mut(width)
            .zip(&column)
            .for_each(|(line, &column)| {
                for (value, &row) in line.iter_mut().zip(&row) {
                    *value /= row * column;
                }
            });
    }
    blurred
}

fn separable_boxes(values: Vec<f32>, width: usize, height: usize, radii: [usize; 3]) -> Vec<f32> {
    let rows = box_rows(values, width, radii);
    let columns = box_rows(transpose(&rows, width, height), height, radii);
    transpose(&columns, height, width)
}

/// Three box passes along each row of `width` samples.
fn box_rows(mut values: Vec<f32>, width: usize, radii: [usize; 3]) -> Vec<f32> {
    values.par_chunks_mut(width).for_each_init(
        || vec![0.0_f32; width],
        |scratch, line| box_passes(line, scratch, radii),
    );
    values
}

fn box_passes_owned(mut line: Vec<f32>, radii: [usize; 3]) -> Vec<f32> {
    let mut scratch = vec![0.0; line.len()];
    box_passes(&mut line, &mut scratch, radii);
    line
}

fn box_passes(line: &mut [f32], scratch: &mut [f32], radii: [usize; 3]) {
    for radius in radii {
        box_mean(line, scratch, radius);
        line.copy_from_slice(scratch);
    }
}

/// Mean over a window of `2·radius + 1` samples with zeros past the ends,
/// kept as a running sum.
fn box_mean(input: &[f32], output: &mut [f32], radius: usize) {
    let length = input.len();
    let scale = 1.0 / (2 * radius + 1) as f64;
    let mut sum = input[..=radius.min(length - 1)]
        .iter()
        .map(|&value| f64::from(value))
        .sum::<f64>();
    for index in 0..length {
        output[index] = (sum * scale) as f32;
        if index + radius + 1 < length {
            sum += f64::from(input[index + radius + 1]);
        }
        if index >= radius {
            sum -= f64::from(input[index - radius]);
        }
    }
}

/// Transpose a row-major `width` × `height` image, in tiles.
fn transpose(values: &[f32], width: usize, height: usize) -> Vec<f32> {
    const TILE: usize = 32;
    let mut output = vec![0.0_f32; values.len()];
    output
        .par_chunks_mut(TILE * height)
        .enumerate()
        .for_each(|(tile, block)| {
            let first = tile * TILE;
            let columns = block.len() / height;
            for row_start in (0..height).step_by(TILE) {
                let row_end = (row_start + TILE).min(height);
                for column in 0..columns {
                    let target = &mut block[column * height..(column + 1) * height];
                    for row in row_start..row_end {
                        target[row] = values[row * width + first + column];
                    }
                }
            }
        });
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_radii_approximate_the_requested_gaussian() {
        for sigma in [2.0, 8.0, 64.0, 256.0] {
            let variance = box_radii(sigma)
                .iter()
                .map(|&radius| {
                    let width = (2 * radius + 1) as f64;
                    (width * width - 1.0) / 12.0
                })
                .sum::<f64>();
            assert!(
                (variance.sqrt() - sigma).abs() / sigma < 0.1,
                "{sigma}: {}",
                variance.sqrt()
            );
        }
    }

    #[test]
    fn blur_keeps_a_flat_image_flat_to_the_edges_and_skips_missing_samples() {
        let (width, height) = (37, 23);
        let mut values = vec![0.25_f32; width * height];
        for sigma in [2.0, 16.0, 64.0] {
            assert!(
                blur(&values, width, height, sigma)
                    .iter()
                    .all(|value| (value - 0.25).abs() < 1.0e-5),
                "{sigma}"
            );
        }
        values[5 * width + 7] = f32::NAN;
        values[0] = f32::INFINITY;
        let blurred = blur(&values, width, height, 4.0);
        assert!(blurred.iter().all(|value| (value - 0.25).abs() < 1.0e-5));
    }

    #[test]
    fn transpose_swaps_rows_and_columns() {
        let (width, height) = (70, 45);
        let values = (0..width * height).map(|i| i as f32).collect::<Vec<_>>();
        let swapped = transpose(&values, width, height);
        for y in 0..height {
            for x in 0..width {
                assert_eq!(swapped[x * height + y], values[y * width + x]);
            }
        }
        assert_eq!(transpose(&swapped, height, width), values);
    }

    #[test]
    fn slope_matches_the_curve() {
        let statistics = RobustStatistics {
            min: 0.01,
            max: 1.0,
            median: 0.05,
            mad: 0.004,
            count: 1000,
        };
        let layered =
            ResolvedLayered::new(statistics, 0.006, LayeredParams::default(), None).unwrap();
        for x in [0.04, 0.05, 0.1, 0.3, 0.7, 0.95] {
            let step = 1.0e-6;
            let numeric =
                (layered.curve.map(x + step) - layered.curve.map(x - step)) / (2.0 * step);
            assert!(
                (numeric - layered.slope(x)).abs() / numeric < 1.0e-4,
                "{x}: {numeric} vs {}",
                layered.slope(x)
            );
        }
    }

    #[test]
    fn saturation_keeps_hue_and_value_and_rises_with_saturation() {
        let hue = |[r, g, b]: [f64; 3]| {
            let (max, min) = (r.max(g).max(b), r.min(g).min(b));
            [
                (r - min) / (max - min),
                (g - min) / (max - min),
                (b - min) / (max - min),
            ]
        };
        for boost in [0.0, 0.5, 1.0] {
            for mask in [false, true] {
                let params = LayeredParams {
                    saturation_amount: 1.0,
                    saturation_boost: boost,
                    lightness_mask: mask,
                    ..LayeredParams::default()
                };
                let mut previous = 0.0;
                for step in 1..=200 {
                    let s = step as f64 / 200.0;
                    let original = [0.8, 0.8 * (1.0 - s * 0.4), 0.8 * (1.0 - s)];
                    let mut lifted = original;
                    saturate(&mut lifted, params);
                    let value = lifted[0].max(lifted[1]).max(lifted[2]);
                    assert!((value - 0.8).abs() < 1.0e-12);
                    let minimum = lifted[0].min(lifted[1]).min(lifted[2]);
                    let saturation = (value - minimum) / value;
                    assert!(saturation >= s - 1.0e-12, "{boost} {mask} {s}");
                    assert!(saturation >= previous - 1.0e-12, "{boost} {mask} {s}");
                    previous = saturation;
                    for (a, b) in hue(original).iter().zip(hue(lifted)) {
                        assert!((a - b).abs() < 1.0e-9, "{boost} {mask} {s}");
                    }
                }
            }
        }
        let mut grey = [0.4; 3];
        saturate(&mut grey, LayeredParams::default());
        assert_eq!(grey, [0.4; 3]);
    }
}
