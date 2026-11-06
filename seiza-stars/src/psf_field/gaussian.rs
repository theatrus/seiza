//! Pixel-integrated round Gaussians: the target PSFs stars are rendered at
//! and deconvolution aims at.

/// FWHM over σ for a Gaussian, `2·sqrt(2 ln 2)`.
pub(super) const FWHM_PER_SIGMA: f64 = 2.354_820_045;

/// The error function, to 1.5 × 10⁻⁷ (Abramowitz and Stegun 7.1.26).
fn erf(x: f64) -> f64 {
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// The share of a unit Gaussian of `sigma` centred on `centre` that falls
/// in each pixel `from ..= to`, pixel `p` spanning `p ± 0.5`.
pub(super) fn pixel_profile(centre: f64, from: isize, to: isize, sigma: f64) -> Vec<f64> {
    let scale = 1.0 / (sigma * std::f64::consts::SQRT_2);
    (from..=to)
        .map(|p| {
            0.5 * (erf((p as f64 + 0.5 - centre) * scale) - erf((p as f64 - 0.5 - centre) * scale))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_profile_holds_the_gaussian_light() {
        let profile = pixel_profile(0.3, -20, 20, 1.7);
        assert!((profile.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        // Centred to the right of pixel 0, so pixel 1 holds more than −1.
        assert!(profile[21] > profile[19]);
        assert!(erf(0.0).abs() < 1e-9 && (erf(3.0) - 0.999_977_9).abs() < 1e-6);
    }
}
