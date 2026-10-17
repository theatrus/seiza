//! The layered stretch: a median anchor curve on luminance that treats
//! large and fine structure apart.
//!
//! A blur splits luminance into a large-scale layer and the detail above it.
//! The large layer takes the global curve; detail keeps the curve's change
//! across it, with more gain where the curve flattens on bright structure.
//! Star cores can be compressed on their own, and colour follows luminance.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    HDR_MAX_EXTRA_SLOPE, ImageShape, LUMA_BLUE, LUMA_GREEN, LUMA_RED, MedianAnchorParams,
    ResolvedMedianAnchor, Result, RobustStatistics, StretchError, compress_highlights,
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

/// The most contrast recovery multiplies detail by.
const MAX_DETAIL_GAIN: f64 = 4.0;
/// Blur sigma in pixels of the fine layer that finds star cores.
const FINE_SIGMA: f64 = 2.0;
/// Knee of the highlight curve the compression pulls toward.
const DRC_KNEE: f64 = 0.5;
/// Below this, a pixel's rescaled luminance is too dark to carry colour.
const COLOUR_EPSILON: f64 = 1.0e-6;

/// A layered stretch with its image-dependent values resolved.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ResolvedLayered {
    /// The requested parameters.
    pub params: LayeredParams,
    /// The global curve `S`.
    pub curve: ResolvedMedianAnchor,
    /// Luminance median `m`.
    pub median: f64,
    /// Robust luminance noise, from the lower half of the samples.
    pub noise: f64,
}

impl ResolvedLayered {
    pub(crate) fn new(
        statistics: RobustStatistics,
        noise: f64,
        params: LayeredParams,
    ) -> Result<Self> {
        validate(params)?;
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
        })
    }

    /// The global curve's slope `S'(x)`, from its rational form. Outside the
    /// black point and 1, the slope at the nearer end.
    fn slope(&self, value: f64) -> f64 {
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
    let curve = |value: f32| layered.curve.map(f64::from(value));

    let mut stretched = if params.contrast_recovery == 0.0 {
        luminance
            .par_iter()
            .map(|&value| if value.is_finite() { curve(value) } else { 0.0 })
            .collect::<Vec<_>>()
    } else {
        let large = blur(
            &luminance,
            width,
            height,
            f64::from(params.scale_separation) / 4.0,
        );
        let half = params.contrast_recovery / 2.0;
        let background_slope = layered.slope(layered.median);
        luminance
            .par_iter()
            .zip(&large)
            .map(|(&value, &base)| {
                if !value.is_finite() {
                    return 0.0;
                }
                // Detail darker than its surroundings takes the slope at its
                // own level, so background beside bright structure is not
                // pushed toward black.
                let level = f64::from(base.min(value));
                let gain = (background_slope / layered.slope(level))
                    .powf(half)
                    .min(MAX_DETAIL_GAIN);
                let base_value = curve(base);
                base_value + (curve(value) - base_value) * gain
            })
            .collect()
    };

    if params.dynamic_range_compression > 0.0 {
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
            .map(|value| value.clamp(0.0, 1.0) as f32)
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
                .chunks_exact_mut(3)
                .zip(input.chunks_exact(3))
                .zip(stretched)
                .zip(luminance)
            {
                if !luma.is_finite() {
                    continue;
                }
                let mut rgb = colour(pixel, luma, value.clamp(0.0, 1.0), black);
                if params.saturation_amount > 0.0 {
                    saturate(&mut rgb, params);
                }
                for (output, value) in output.iter_mut().zip(rgb) {
                    *output = value.clamp(0.0, 1.0) as f32;
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
/// back in gamut by its largest channel.
fn colour(pixel: &[f32], luma: f32, value: f64, black: f64) -> [f64; 3] {
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
    if maximum > 1.0 {
        for channel in &mut rgb {
            *channel /= maximum;
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
        amount *= value;
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
        let layered = ResolvedLayered::new(statistics, 0.006, LayeredParams::default()).unwrap();
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
