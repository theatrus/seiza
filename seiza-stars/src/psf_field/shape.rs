//! A PSF stamp's width, elongation and wings, from an elliptical Moffat
//! profile with free β.

use std::f64::consts::PI;

use super::candidates::adaptive_moments;
use crate::psf_fitting::{LevenbergMarquardt, PSFFunction};

/// The shape of a PSF at one point, from an elliptical Moffat profile
/// `A·(1 + x′²/αₐ² + y′²/α_b²)^−β` fitted to its stamp.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PsfShape {
    /// Full width at half maximum in pixels: the mean of the major and minor
    /// axes' widths, as `seiza-stars` averages its other fitted widths.
    pub fwhm: f64,
    /// FWHM along the major axis, in pixels.
    pub fwhm_major: f64,
    /// FWHM along the minor axis, in pixels.
    pub fwhm_minor: f64,
    /// Ellipticity component `χ·cos 2φ`, where `χ = (a² − b²)/(a² + b²)` for
    /// the axes `a ≥ b` and `φ` is [`Self::angle`]. Positive along `x`.
    pub e1: f64,
    /// Ellipticity component `χ·sin 2φ`. Positive along the diagonal from
    /// `+x` towards `+y`.
    pub e2: f64,
    /// The major axis's direction in radians over `[0, π)`, from `+x`
    /// towards `+y`.
    pub angle: f64,
    /// Moffat β. Lower values mean stronger wings; a Gaussian is β → ∞, and
    /// the fit stops at [`MAXIMUM_BETA`].
    pub beta: f64,
}

/// The largest Moffat β a shape fit reports. A profile with no wings to
/// speak of, such as a Gaussian, reaches it.
pub const MAXIMUM_BETA: f64 = 25.0;

impl PsfShape {
    /// The ellipticity's magnitude, `sqrt(e1² + e2²)`.
    pub fn ellipticity(&self) -> f64 {
        self.e1.hypot(self.e2)
    }
}

/// An elliptical Moffat with free β. Parameters:
/// `[A, B, x0, y0, αx, αy, θ, β]`.
struct EllipticalMoffat;

impl PSFFunction for EllipticalMoffat {
    fn value(&self, x: f64, y: f64, params: &[f64]) -> f64 {
        let (cos, sin) = (params[6].cos(), params[6].sin());
        let (dx, dy) = (x - params[2], y - params[3]);
        let xp = dx * cos + dy * sin;
        let yp = -dx * sin + dy * cos;
        let d = 1.0 + xp * xp / (params[4] * params[4]) + yp * yp / (params[5] * params[5]);
        params[1] + params[0] * d.powf(-params[7])
    }

    fn gradient(&self, x: f64, y: f64, params: &[f64], grad: &mut [f64]) {
        let (a, ax, ay, theta, beta) = (params[0], params[4], params[5], params[6], params[7]);
        let (cos, sin) = (theta.cos(), theta.sin());
        let (dx, dy) = (x - params[2], y - params[3]);
        let xp = dx * cos + dy * sin;
        let yp = -dx * sin + dy * cos;
        let (ax2, ay2) = (ax * ax, ay * ay);
        let d = 1.0 + xp * xp / ax2 + yp * yp / ay2;
        let power = d.powf(-beta);
        let g = -beta * a * power / d;
        let (ddx, ddy) = (2.0 * xp / ax2, 2.0 * yp / ay2);
        grad[0] = power;
        grad[1] = 1.0;
        grad[2] = g * (-ddx * cos + ddy * sin);
        grad[3] = g * (-ddx * sin - ddy * cos);
        grad[4] = g * (-2.0 * xp * xp / (ax2 * ax));
        grad[5] = g * (-2.0 * yp * yp / (ay2 * ay));
        grad[6] = g * 2.0 * xp * yp * (1.0 / ax2 - 1.0 / ay2);
        grad[7] = -a * power * d.ln();
    }

    fn sigma_to_fwhm(&self, sigma: f64) -> f64 {
        sigma
    }
}

/// `2·sqrt(2^(1/β) − 1)`: a Moffat's FWHM over its α.
fn moffat_fwhm_factor(beta: f64) -> f64 {
    2.0 * (2.0_f64.powf(1.0 / beta) - 1.0).sqrt()
}

/// Fit the shape of a `size × size` stamp centred on its middle pixel.
pub(super) fn fit_shape(stamp: &[f64], size: usize) -> Option<PsfShape> {
    let peak = stamp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if peak.is_nan() || peak <= 0.0 {
        return None;
    }
    let centre = (size / 2) as f64;
    let values: Vec<f64> = stamp.iter().map(|v| v / peak).collect();
    let at = |x: isize, y: isize| {
        if x < 0 || y < 0 || x >= size as isize || y >= size as isize {
            0.0
        } else {
            values[y as usize * size + x as usize]
        }
    };
    let moments = adaptive_moments(at, centre, centre, centre, 0.0, 1.5)?;
    let (xx, xy, yy) = (moments.xx, moments.xy, moments.yy);
    let half_trace = 0.5 * (xx + yy);
    let spread = (0.25 * (xx - yy).powi(2) + xy * xy).sqrt();
    let (major, minor) = (
        (half_trace + spread).max(1e-6),
        (half_trace - spread).max(1e-6),
    );
    let theta0 = 0.5 * (2.0 * xy).atan2(xx - yy);
    let beta0 = 3.0;
    let alpha = |variance: f64| 2.354_820_045 * variance.sqrt() / moffat_fwhm_factor(beta0);

    let radius = centre;
    let mut positions = Vec::with_capacity(size * size);
    let mut samples = Vec::with_capacity(size * size);
    for v in 0..size {
        for u in 0..size {
            let (dx, dy) = (u as f64 - centre, v as f64 - centre);
            if dx * dx + dy * dy <= radius * radius {
                positions.push((dx, dy));
                samples.push(values[v * size + u]);
            }
        }
    }
    let initial = [
        1.0,
        0.0,
        moments.x - centre,
        moments.y - centre,
        alpha(major),
        alpha(minor),
        theta0,
        beta0,
    ];
    let lower = [0.0, -0.2, -2.0, -2.0, 0.1, 0.1, -2.0 * PI, 1.05];
    let upper = [
        10.0,
        0.2,
        2.0,
        2.0,
        size as f64,
        size as f64,
        2.0 * PI,
        MAXIMUM_BETA,
    ];
    let fitted = LevenbergMarquardt::default()
        .fit(
            &EllipticalMoffat,
            &positions,
            &samples,
            &initial,
            &lower,
            &upper,
        )
        .ok()?;
    let (ax, ay, theta, beta) = (fitted[4].abs(), fitted[5].abs(), fitted[6], fitted[7]);
    if !(ax.is_finite() && ay.is_finite() && beta.is_finite()) {
        return None;
    }
    let (a, b, angle) = if ax >= ay {
        (ax, ay, theta)
    } else {
        (ay, ax, theta + PI / 2.0)
    };
    let angle = angle.rem_euclid(PI);
    let factor = moffat_fwhm_factor(beta);
    let chi = (a * a - b * b) / (a * a + b * b);
    Some(PsfShape {
        fwhm: 0.5 * (a + b) * factor,
        fwhm_major: a * factor,
        fwhm_minor: b * factor,
        e1: chi * (2.0 * angle).cos(),
        e2: chi * (2.0 * angle).sin(),
        angle,
        beta,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moffat_stamp(size: usize, a: f64, b: f64, angle: f64, beta: f64) -> Vec<f64> {
        let centre = (size / 2) as f64;
        let (cos, sin) = (angle.cos(), angle.sin());
        (0..size * size)
            .map(|index| {
                let (dx, dy) = (
                    (index % size) as f64 - centre,
                    (index / size) as f64 - centre,
                );
                let xp = dx * cos + dy * sin;
                let yp = -dx * sin + dy * cos;
                (1.0 + xp * xp / (a * a) + yp * yp / (b * b)).powf(-beta)
            })
            .collect()
    }

    #[test]
    fn recovers_an_elliptical_moffat() {
        let (a, b, angle, beta) = (3.0, 2.2, 0.6, 2.8);
        let shape = fit_shape(&moffat_stamp(33, a, b, angle, beta), 33).unwrap();
        let factor = moffat_fwhm_factor(beta);
        assert!((shape.fwhm_major - a * factor).abs() < 1e-3, "{shape:?}");
        assert!((shape.fwhm_minor - b * factor).abs() < 1e-3, "{shape:?}");
        assert!((shape.beta - beta).abs() < 1e-3, "{shape:?}");
        assert!((shape.angle - angle).abs() < 1e-3, "{shape:?}");
        let chi = (a * a - b * b) / (a * a + b * b);
        assert!((shape.e1 - chi * (2.0 * angle).cos()).abs() < 1e-4);
        assert!((shape.e2 - chi * (2.0 * angle).sin()).abs() < 1e-4);
    }

    #[test]
    fn a_round_profile_has_no_ellipticity() {
        let shape = fit_shape(&moffat_stamp(33, 2.5, 2.5, 0.0, 4.0), 33).unwrap();
        assert!(shape.ellipticity() < 1e-4, "{shape:?}");
        assert!((shape.beta - 4.0).abs() < 0.01, "{shape:?}");
    }
}
