//! Forced aperture photometry at supplied positions on one linear mono plane.
//!
//! [`aperture_photometry`] sums a circular aperture at each position a caller
//! gives, less the local sky its annulus measures, and returns one
//! [`ApertureMeasurement`] per position. It detects nothing, moves no
//! position, and needs no reference image: a time series measures its target
//! and comparison stars where the caller says they are.
//!
//! The sky fit, [`fit_sky`], is the one continuum fitting measures its stars
//! with.
use crate::{Error, LinearImage, Result};
use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};

/// The fewest sky pixels a local background is fitted from, before and
/// after clipping.
const MINIMUM_SKY_PIXELS: usize = 20;
/// The share of an annulus's pixels that must be usable for a forced
/// measurement's background. A plane through less than half a ring is an
/// extrapolation to the aperture.
const MINIMUM_ANNULUS_FRACTION: f64 = 0.5;

/// A circular aperture and its sky annulus, in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ApertureGeometry {
    /// Radius of the aperture the flux is summed in.
    #[serde(alias = "aperture_radius")]
    pub aperture_radius: f64,
    /// Inner radius of the sky annulus, at least the aperture's radius.
    #[serde(alias = "annulus_inner")]
    pub annulus_inner: f64,
    /// Outer radius of the sky annulus.
    #[serde(alias = "annulus_outer")]
    pub annulus_outer: f64,
}

impl ApertureGeometry {
    /// An aperture of `aperture_radius` with a sky annulus between
    /// `annulus_inner` and `annulus_outer`, all in pixels.
    pub fn new(aperture_radius: f64, annulus_inner: f64, annulus_outer: f64) -> Self {
        Self {
            aperture_radius,
            annulus_inner,
            annulus_outer,
        }
    }

    fn validate(&self) -> Result<()> {
        let radii = [self.aperture_radius, self.annulus_inner, self.annulus_outer];
        if radii.iter().any(|radius| !radius.is_finite())
            || self.aperture_radius <= 0.0
            || self.annulus_inner < self.aperture_radius
            || self.annulus_outer <= self.annulus_inner
        {
            return Err(invalid(
                "Aperture geometry needs finite radii with 0 < aperture <= annulus inner < annulus outer",
            ));
        }
        if self.annulus_outer > 4096.0 {
            return Err(invalid("Annulus outer radius is limited to 4096 pixels"));
        }
        Ok(())
    }
}

/// What to measure and what is known about the samples.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AperturePhotometryOptions {
    /// The aperture and annulus every position is measured with.
    pub geometry: ApertureGeometry,
    /// The sample level at or above which a pixel is saturated, in the
    /// image's units: the sensor's or the calibrated frame's real ceiling,
    /// never an observed maximum. `None` flags nothing by level.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub saturation: Option<f64>,
    /// Electrons per sample unit. With it, a positive flux adds its own
    /// photon noise, `flux / gain`, to the variance; without it the error
    /// holds sky noise only. Ignored when an uncertainty map is supplied.
    /// The mean of `n` equal frames has `n` times one frame's gain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gain: Option<f64>,
}

impl AperturePhotometryOptions {
    /// Options for `geometry` with no saturation level and no gain.
    pub fn new(geometry: ApertureGeometry) -> Self {
        Self {
            geometry,
            saturation: None,
            gain: None,
        }
    }
}

/// Per-pixel planes on the image's grid, each row-major and
/// `width * height` long. All are optional.
#[derive(Clone, Copy, Debug, Default)]
pub struct ApertureMaps<'a> {
    /// Nonzero where a pixel must not be used: a defect, a cosmic ray, a
    /// neighbouring star. Such a pixel is left out of the aperture, whose
    /// sum then lacks it, and out of the sky.
    pub mask: Option<&'a [u8]>,
    /// Nonzero where the producer knows a pixel clipped, for samples whose
    /// scalar ceiling calibration or stacking has blurred.
    pub saturation: Option<&'a [u8]>,
    /// Each pixel's total 1-sigma noise in the image's units, the source's
    /// photon noise included. With it the flux error comes from these values
    /// and not from the annulus's scatter.
    pub uncertainty: Option<&'a [f32]>,
}

/// What is wrong with a measurement. A measurement with no flag set summed
/// its whole aperture over unsaturated pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ApertureFlags {
    /// A pixel of the aperture is at or above the saturation level, or set
    /// in the saturation map. The flux is a lower bound.
    pub saturated: bool,
    /// Part of the aperture is off the image, masked or not finite. The flux
    /// sums only the covered part, `aperture_area` pixels of it, and nothing
    /// is filled in.
    #[serde(alias = "missing_coverage")]
    pub missing_coverage: bool,
    /// No flux could be measured: the aperture has no usable pixel, or too
    /// little of the annulus is usable to give a background. The flux, its
    /// error and the magnitude are `None`.
    pub unusable: bool,
}

/// One position's measurement. Fluxes are in the image's sample units.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ApertureMeasurement {
    /// The requested position, unchanged: zero-based, pixel centres on whole
    /// numbers.
    pub x: f64,
    /// See `x`.
    pub y: f64,
    /// Background-subtracted aperture sum. Signed: a faint source in noise
    /// can measure below zero, and is reported so.
    pub flux: Option<f64>,
    /// 1-sigma error of `flux` under the noise model [`aperture_photometry`]
    /// documents. `None` with the flux, or when the uncertainty map has a
    /// value in the aperture that is negative or not finite.
    #[serde(alias = "flux_error")]
    pub flux_error: Option<f64>,
    /// Instrumental magnitude, `-2.5 log10(flux)`, with no zero point.
    /// `None` unless the flux is positive.
    pub magnitude: Option<f64>,
    /// `2.5 / ln(10) * flux_error / flux`, where there is a magnitude.
    #[serde(alias = "magnitude_error")]
    pub magnitude_error: Option<f64>,
    /// Local sky per pixel at the position, from the annulus.
    pub background: Option<f64>,
    /// Robust per-pixel scatter of the annulus about its fitted sky.
    #[serde(alias = "background_noise")]
    pub background_noise: Option<f64>,
    /// The aperture sum before the sky is taken off.
    #[serde(alias = "aperture_sum")]
    pub aperture_sum: Option<f64>,
    /// Area summed, in pixels: `pi * aperture_radius^2` when all of the
    /// aperture is covered.
    #[serde(alias = "aperture_area")]
    pub aperture_area: f64,
    /// Annulus pixels the sky was fitted from, after clipping.
    #[serde(alias = "annulus_pixels")]
    pub annulus_pixels: usize,
    /// Largest sample in the aperture, sky included.
    pub peak: Option<f64>,
    /// What is wrong with this measurement, if anything.
    pub flags: ApertureFlags,
}

fn invalid(message: &str) -> Error {
    Error::Photometry(message.into())
}

/// Median with the workspace's convention: an even count averages its two
/// middle values.
pub(crate) fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    sorted[(sorted.len() - 1) / 2] / 2.0 + sorted[sorted.len() / 2] / 2.0
}

/// A local sky: a plane fitted to annulus pixels, with the pixels that stand
/// away from it (a neighbour's light, a defect) clipped out.
pub(crate) struct Sky {
    pedestal: f64,
    plane: Vector3<f64>,
    /// Normal matrix of the samples used for the fitted plane.
    normal: Matrix3<f64>,
    /// Robust per-pixel scatter of the pixels kept, about the plane.
    pub(crate) noise: f64,
    /// Pixels kept after clipping.
    pub(crate) pixels: usize,
}

impl Sky {
    /// The sky at an offset from the position the annulus is centred on.
    pub(crate) fn at(&self, dx: f64, dy: f64) -> f64 {
        self.pedestal + self.plane.dot(&Vector3::new(1.0, dx, dy))
    }
}

/// Fit the sky to `(dx, dy, value)` annulus pixels, offsets taken from the
/// star's position. `None` when fewer than twenty pixels are given or
/// survive clipping at 3.5 robust sigma, or when they do not span a plane.
pub(crate) fn fit_sky(sky: &[(f64, f64, f64)]) -> Option<Sky> {
    if sky.len() < MINIMUM_SKY_PIXELS {
        return None;
    }
    let pedestal = median(&sky.iter().map(|s| s.2).collect::<Vec<_>>());
    let mut accepted: Vec<_> = (0..sky.len()).collect();
    let mut plane = Vector3::zeros();
    let mut normal = Matrix3::zeros();
    for _ in 0..4 {
        normal.fill(0.0);
        let mut target = Vector3::zeros();
        for i in &accepted {
            let (dx, dy, v) = sky[*i];
            let row = Vector3::new(1.0, dx, dy);
            normal += row * row.transpose();
            target += row * (v - pedestal);
        }
        plane = normal.lu().solve(&target)?;
        let residuals: Vec<_> = sky
            .iter()
            .map(|(dx, dy, v)| v - pedestal - plane.dot(&Vector3::new(1.0, *dx, *dy)))
            .collect();
        let center = median(&accepted.iter().map(|i| residuals[*i]).collect::<Vec<_>>());
        let sigma = 1.4826
            * median(
                &accepted
                    .iter()
                    .map(|i| (residuals[*i] - center).abs())
                    .collect::<Vec<_>>(),
            );
        let threshold = (3.5 * sigma).max(1e-10 * pedestal.abs().max(1.0));
        let next: Vec<_> = residuals
            .iter()
            .enumerate()
            .filter_map(|(i, r)| ((r - center).abs() <= threshold).then_some(i))
            .collect();
        if next.len() < MINIMUM_SKY_PIXELS {
            return None;
        }
        if next == accepted {
            break;
        }
        accepted = next;
    }
    let residuals: Vec<_> = accepted
        .iter()
        .map(|i| {
            let (dx, dy, v) = sky[*i];
            v - pedestal - plane.dot(&Vector3::new(1.0, dx, dy))
        })
        .collect();
    let center = median(&residuals);
    let noise = 1.4826
        * median(
            &residuals
                .iter()
                .map(|r| (r - center).abs())
                .collect::<Vec<_>>(),
        );
    Some(Sky {
        pedestal,
        plane,
        normal,
        noise,
        pixels: accepted.len(),
    })
}

/// Integral of `clamp(sqrt(r^2 - u^2), low, high)` for `u` from `a` to `b`,
/// with `-r <= a <= b <= r` and `0 <= low <= high`.
fn clamped_arc_integral(r: f64, a: f64, b: f64, low: f64, high: f64) -> f64 {
    // `sqrt(r^2 - u^2)` as a product: the difference of squares loses its
    // digits where `u` nears `r`, which is where a pixel's edge meets the
    // circle's.
    let chord = |u: f64| ((r - u) * (r + u)).max(0.0).sqrt();
    let arc = |u: f64| {
        let u = u.clamp(-r, r);
        let half = chord(u);
        0.5 * (u * half + r * r * u.atan2(half))
    };
    // The arc crosses a level `y` at `u = ±sqrt(r^2 - y^2)`.
    let mut cuts = vec![a, b];
    for level in [low, high] {
        if level < r {
            let u = chord(level);
            cuts.extend([-u, u].into_iter().filter(|u| *u > a && *u < b));
        }
    }
    cuts.sort_unstable_by(f64::total_cmp);
    cuts.windows(2)
        .map(|pair| {
            let (from, to) = (pair[0], pair[1]);
            let middle = 0.5 * (from + to);
            let height = chord(middle);
            if height <= low {
                low * (to - from)
            } else if high < r && height >= high {
                // Only a level below the circle's top is ever reached.
                high * (to - from)
            } else {
                arc(to) - arc(from)
            }
        })
        .sum()
}

/// Exact area of the part of the rectangle `[x0, x1] x [y0, y1]` inside the
/// circle of radius `r` about the origin.
fn circle_rectangle_overlap(r: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    let (a, b) = (x0.max(-r), x1.min(r));
    if a >= b || y0 >= r || y1 <= -r {
        return 0.0;
    }
    // At each `u` the circle spans `-s..s`, `s = sqrt(r^2 - u^2)`; the
    // rectangle keeps `clamp(s, y0, y1) - clamp(-s, y0, y1)` of it. Each
    // clamp splits into its nonnegative parts.
    let upper = |low: f64, high: f64| {
        if high <= 0.0 {
            // `s >= 0 >= high`: the clamp is `high` throughout.
            high * (b - a)
        } else if low >= 0.0 {
            clamped_arc_integral(r, a, b, low, high)
        } else {
            // `low < 0 < high`: `s` is never below `low`.
            clamped_arc_integral(r, a, b, 0.0, high)
        }
    };
    // clamp(-s, y0, y1) = -clamp(s, -y1, -y0).
    (upper(y0, y1) + upper(-y1, -y0)).clamp(0.0, (x1 - x0) * (y1 - y0))
}

/// The fraction of the pixel centred on `(px, py)` inside the circle of
/// radius `r` about `(x, y)`.
fn pixel_weight(px: f64, py: f64, x: f64, y: f64, r: f64) -> f64 {
    let (dx, dy) = (px - x, py - y);
    // Wholly inside or outside, by the pixel's farthest and nearest points.
    let far = (dx.abs() + 0.5).hypot(dy.abs() + 0.5);
    if far <= r {
        return 1.0;
    }
    let near = (dx.abs() - 0.5).max(0.0).hypot((dy.abs() - 0.5).max(0.0));
    if near >= r {
        return 0.0;
    }
    circle_rectangle_overlap(r, dx - 0.5, dx + 0.5, dy - 0.5, dy + 0.5)
}

fn check_map(length: Option<usize>, pixels: usize, name: &str) -> Result<()> {
    match length {
        Some(length) if length != pixels => Err(Error::Photometry(format!(
            "The {name} map has {length} values for an image of {pixels} pixels"
        ))),
        _ => Ok(()),
    }
}

/// Measure a circular aperture at each of `positions` on a linear mono
/// `image`, and return one [`ApertureMeasurement`] per position, in order.
///
/// Positions are `[x, y]`, zero-based with pixel centres on whole numbers,
/// and are measured exactly where given: nothing is detected, centred or
/// moved. The image must be one linear plane, such as a mono frame or the
/// green plane a caller extracted; nothing is converted from colour.
///
/// **Aperture.** Each pixel counts by the exact fraction of its area inside
/// the circle, so the sum follows a star smoothly across subpixel shifts.
/// Pixels off the image, masked or not finite are left out and flagged
/// `missing_coverage`; saturated pixels are summed and flagged `saturated`.
///
/// **Background.** The sky is a plane fitted to the annulus's pixels (those
/// whose centres lie between its radii), clipped at 3.5 robust sigma so a
/// neighbour's light does not raise it, and evaluated under each aperture
/// pixel, so a gradient across the star is removed. Masked, saturated and
/// non-finite annulus pixels are left out. A measurement is `unusable` when
/// fewer than twenty annulus pixels or less than half the annulus remain.
///
/// **Flux error.** The flux is `sum(w * (value - sky))` over the aperture's
/// pixels, `w` each pixel's weight. With `sky_noise` the annulus's robust
/// per-pixel scatter, `q = sum(w * [1, dx, dy])` over the aperture, and
/// `N = sum([1, dx, dy] * [1, dx, dy]^T)` over the fitted sky samples:
///
/// ```text
/// variance = sum(w^2) * sky_noise^2                    sky, read and dark noise
///          + sky_noise^2 * q^T * inverse(N) * q       error of the fitted sky
///          + max(sum(w^2 * (value - sky)), 0) / gain   source photons, with a gain
/// ```
///
/// The sky term propagates the plane coefficient covariance, including
/// slopes when masking or missing coverage makes the annulus asymmetric.
/// It reduces to `area^2 * sky_noise^2 / n_sky` for a centred symmetric
/// aperture and annulus. The last term is `flux / gain` but for the pixels
/// on the aperture's edge.
/// With an uncertainty map the first and last terms are replaced by
/// `sum(w^2 * uncertainty^2)`, the map being each pixel's total noise. This
/// assumes pixel noise is independent from pixel to pixel and, without a
/// map, that the aperture's sky noise is the annulus's. Demosaiced,
/// resampled or stacked pixels are correlated, and their true error is
/// larger than this one. Scintillation, flat-field error and aperture
/// losses are not modelled: the scatter of a constant check star across a
/// series is the test of these errors, not a part of them.
///
/// **Magnitude.** `-2.5 log10(flux)` with no zero point, for positive flux.
///
/// Errors only on input that cannot be measured at all: an image that is
/// not mono, a geometry whose radii are out of order, a map of the wrong
/// length, a position that is not finite, or a gain that is not positive.
pub fn aperture_photometry(
    image: &LinearImage,
    positions: &[[f64; 2]],
    options: &AperturePhotometryOptions,
    maps: &ApertureMaps<'_>,
) -> Result<Vec<ApertureMeasurement>> {
    if image.channels != 1 {
        return Err(invalid(
            "Aperture photometry measures one mono plane; extract the channel to measure first",
        ));
    }
    let pixels = image.width * image.height;
    if image.data.len() != pixels {
        return Err(invalid("The image's samples do not match its size"));
    }
    options.geometry.validate()?;
    if options.saturation.is_some_and(|level| !level.is_finite()) {
        return Err(invalid("The saturation level must be finite"));
    }
    if options
        .gain
        .is_some_and(|gain| !gain.is_finite() || gain <= 0.0)
    {
        return Err(invalid("The gain must be finite and positive"));
    }
    check_map(maps.mask.map(<[u8]>::len), pixels, "mask")?;
    check_map(maps.saturation.map(<[u8]>::len), pixels, "saturation")?;
    check_map(maps.uncertainty.map(<[f32]>::len), pixels, "uncertainty")?;
    if positions.iter().flatten().any(|value| !value.is_finite()) {
        return Err(invalid("Aperture positions must be finite"));
    }
    Ok(positions
        .iter()
        .map(|position| measure(image, *position, options, maps))
        .collect())
}

fn measure(
    image: &LinearImage,
    [x, y]: [f64; 2],
    options: &AperturePhotometryOptions,
    maps: &ApertureMaps<'_>,
) -> ApertureMeasurement {
    let geometry = &options.geometry;
    let radius = geometry.aperture_radius;
    let outer = geometry.annulus_outer;
    let mut flags = ApertureFlags::default();
    let mut result = ApertureMeasurement {
        x,
        y,
        flux: None,
        flux_error: None,
        magnitude: None,
        magnitude_error: None,
        background: None,
        background_noise: None,
        aperture_sum: None,
        aperture_area: 0.0,
        annulus_pixels: 0,
        peak: None,
        flags,
    };
    let saturated = |index: usize, value: f64| {
        options.saturation.is_some_and(|level| value >= level)
            || maps.saturation.is_some_and(|map| map[index] != 0)
    };

    // (weight, dx, dy, value, index) of each usable aperture pixel.
    let mut aperture = Vec::new();
    let mut sky = Vec::new();
    let mut annulus_area = 0_usize;
    // Every pixel the annulus's outer edge can reach, on the image or off.
    let reach = outer.max(radius + 1.0);
    let (left, right) = ((x - reach).floor() as i64, (x + reach).ceil() as i64);
    let (top, bottom) = ((y - reach).floor() as i64, (y + reach).ceil() as i64);
    for py in top..=bottom {
        for px in left..=right {
            let (dx, dy) = (px as f64 - x, py as f64 - y);
            let distance = dx.hypot(dy);
            let weight = pixel_weight(px as f64, py as f64, x, y, radius);
            let in_annulus = (geometry.annulus_inner..=outer).contains(&distance);
            if weight == 0.0 && !in_annulus {
                continue;
            }
            annulus_area += usize::from(in_annulus);
            let on_image =
                px >= 0 && py >= 0 && (px as usize) < image.width && (py as usize) < image.height;
            let index = if on_image {
                py as usize * image.width + px as usize
            } else {
                0
            };
            let value = if on_image {
                f64::from(image.data[index])
            } else {
                f64::NAN
            };
            let usable = value.is_finite() && !maps.mask.is_some_and(|mask| mask[index] != 0);
            if weight > 0.0 {
                if usable {
                    flags.saturated |= saturated(index, value);
                    aperture.push((weight, dx, dy, value, index));
                } else {
                    flags.missing_coverage = true;
                }
            }
            if in_annulus && usable && !saturated(index, value) {
                sky.push((dx, dy, value));
            }
        }
    }
    result.aperture_area = aperture.iter().map(|pixel| pixel.0).sum();
    if !aperture.is_empty() {
        result.aperture_sum = Some(aperture.iter().map(|pixel| pixel.0 * pixel.3).sum());
        result.peak = aperture.iter().map(|pixel| pixel.3).max_by(f64::total_cmp);
    }

    let enough_sky = sky.len() as f64 >= MINIMUM_ANNULUS_FRACTION * annulus_area as f64;
    let Some(sky) = enough_sky.then(|| fit_sky(&sky)).flatten() else {
        flags.unusable = true;
        result.flags = flags;
        return result;
    };
    if (sky.pixels as f64) < MINIMUM_ANNULUS_FRACTION * annulus_area as f64 {
        flags.unusable = true;
        result.flags = flags;
        return result;
    }
    result.background = Some(sky.at(0.0, 0.0));
    result.background_noise = Some(sky.noise);
    result.annulus_pixels = sky.pixels;
    if aperture.is_empty() {
        flags.unusable = true;
        result.flags = flags;
        return result;
    }

    let flux: f64 = aperture
        .iter()
        .map(|(weight, dx, dy, value, _)| weight * (value - sky.at(*dx, *dy)))
        .sum();
    // The aperture sums predictions of a three-coefficient sky plane. Its
    // error depends on the annulus geometry, including asymmetric masking,
    // and on the first moments of the covered aperture, not only its area.
    let moments = aperture
        .iter()
        .fold(Vector3::zeros(), |sum, (w, dx, dy, _, _)| {
            sum + Vector3::new(*w, w * dx, w * dy)
        });
    let sky_error = sky
        .normal
        .lu()
        .solve(&moments)
        .map(|solution| (moments.dot(&solution) * sky.noise * sky.noise).max(0.0));
    // The flux is a weighted sum of whole pixels' values, so each pixel's
    // variance enters by its weight squared.
    let variance = match maps.uncertainty {
        Some(map) => aperture
            .iter()
            .map(|(weight, _, _, _, index)| {
                let sigma = f64::from(map[*index]);
                (sigma.is_finite() && sigma >= 0.0).then_some(weight * weight * sigma * sigma)
            })
            .sum::<Option<f64>>(),
        None => {
            let weights: f64 = aperture.iter().map(|pixel| pixel.0 * pixel.0).sum();
            let photons = options.gain.map_or(0.0, |gain| {
                let source: f64 = aperture
                    .iter()
                    .map(|(weight, dx, dy, value, _)| weight * weight * (value - sky.at(*dx, *dy)))
                    .sum();
                source.max(0.0) / gain
            });
            Some(weights * sky.noise * sky.noise + photons)
        }
    };
    let error = variance
        .zip(sky_error)
        .map(|(variance, sky_error)| (variance + sky_error).sqrt());
    result.flux = Some(flux);
    result.flux_error = error;
    if flux > 0.0 {
        result.magnitude = Some(-2.5 * flux.log10());
        result.magnitude_error = error.map(|error| 2.5 / std::f64::consts::LN_10 * error / flux);
    }
    result.flags = flags;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn pixel_weights_are_exact_areas() {
        // A circle wholly inside one pixel, and one cutting it in half.
        let small = pixel_weight(0.0, 0.0, 0.1, -0.2, 0.25);
        assert!((small - PI * 0.0625).abs() < 1e-14, "{small}");
        // The pixel the circle's edge crosses on its axis: the arc's integral
        // across the pixel's height, less the pixel's inner edge.
        let edge = pixel_weight(10.0, 0.0, 0.0, 0.0, 10.0);
        let expected = 0.5 * 99.75_f64.sqrt() + 100.0 * 0.05_f64.asin() - 9.5;
        assert!((edge - expected).abs() < 1e-12, "{edge} against {expected}");
        // Weights over the grid sum to the circle's area at any subpixel centre.
        for (x, y, r) in [
            (0.0, 0.0, 3.0),
            (0.37, -0.21, 4.5),
            (0.5, 0.5, 2.0),
            (-0.49, 0.13, 0.7),
            // An edge a rounding error short of a pixel's.
            (118.9 - 119.0, 94.6 - 95.0, 0.6),
            (0.1, 0.4, 0.6),
            // A circle whose top touches a pixel's edge.
            (0.0, 0.0, 2.5),
            (0.5, 0.0, 3.5),
            (0.0, -0.5, 1.0),
            (0.25, 0.75, 7.3),
        ] {
            let mut sum = 0.0;
            for py in -12..=12 {
                for px in -12..=12 {
                    let weight = pixel_weight(f64::from(px), f64::from(py), x, y, r);
                    assert!((0.0..=1.0).contains(&weight));
                    sum += weight;
                }
            }
            assert!((sum - PI * r * r).abs() < 1e-13, "{x} {y} {r}: {sum}");
        }
    }

    #[test]
    fn a_pixel_weight_matches_a_fine_grid_count() {
        for (px, py, x, y, r) in [
            (3.0, 1.0, 0.3, 0.4, 3.0),
            (2.0, 2.0, 0.0, 0.0, 3.0),
            (0.0, 4.0, 0.2, -0.3, 4.4),
            (-3.0, -2.0, 0.45, 0.45, 3.6),
        ] {
            let n = 800;
            let mut inside = 0;
            for j in 0..n {
                for i in 0..n {
                    let u = px - 0.5 + (f64::from(i) + 0.5) / f64::from(n) - x;
                    let v = py - 0.5 + (f64::from(j) + 0.5) / f64::from(n) - y;
                    inside += i32::from(u * u + v * v <= r * r);
                }
            }
            let counted = f64::from(inside) / f64::from(n * n);
            let exact = pixel_weight(px, py, x, y, r);
            assert!((exact - counted).abs() < 2e-3, "{exact} against {counted}");
        }
    }
}
