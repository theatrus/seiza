/// PSF (Point Spread Function) fitting module
/// Implements Gaussian and Moffat PSF models with Levenberg-Marquardt fitting
/// Based on HocusFocus implementation by George Hilios
use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;

/// PSF fitting type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PSFType {
    /// No PSF fitting, use simple metrics only
    None,
    /// Gaussian PSF model
    Gaussian,
    /// Moffat PSF with beta=4.0
    Moffat4,
}

impl std::str::FromStr for PSFType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" => Ok(PSFType::None),
            "gaussian" => Ok(PSFType::Gaussian),
            "moffat" | "moffat4" | "moffat_4" => Ok(PSFType::Moffat4),
            _ => Err(format!("Unknown PSF type: {}", s)),
        }
    }
}

/// PSF model parameters after fitting
#[derive(Debug, Clone)]
pub struct PSFModel {
    /// PSF type used
    pub psf_type: PSFType,
    /// Amplitude (peak brightness above background)
    pub amplitude: f64,
    /// Background level
    pub background: f64,
    /// X offset from centroid
    pub x0: f64,
    /// Y offset from centroid
    pub y0: f64,
    /// Sigma in X direction
    pub sigma_x: f64,
    /// Sigma in Y direction
    pub sigma_y: f64,
    /// Rotation of the fitted model's X axis in radians. Use
    /// [`Self::major_axis_theta`] for an elongation direction.
    pub theta: f64,
    /// R-squared goodness of fit
    pub r_squared: f64,
    /// Root mean square error
    pub rmse: f64,
    /// Full Width Half Maximum
    pub fwhm: f64,
    /// Eccentricity (0 = circular, 1 = line)
    pub eccentricity: f64,
}

impl PSFModel {
    /// Orientation of the fitted ellipse's major axis in radians over
    /// `[0, π)`.
    ///
    /// [`Self::theta`] rotates the model's X axis. The major axis is instead
    /// the fitted X axis when `sigma_x >= sigma_y`, and the perpendicular Y
    /// axis otherwise. Consumers drawing elongation vectors or combining
    /// axial directions should use this normalized value.
    pub fn major_axis_theta(&self) -> f64 {
        let theta = if self.sigma_x >= self.sigma_y {
            self.theta
        } else {
            self.theta + PI / 2.0
        };
        theta.rem_euclid(PI)
    }

    /// Calculate FWHM from sigma values based on PSF type
    pub fn calculate_fwhm(&self) -> f64 {
        let avg_sigma = (self.sigma_x + self.sigma_y) / 2.0;
        match self.psf_type {
            PSFType::Gaussian => avg_sigma * 2.0 * (2.0 * 2.0_f64.ln()).sqrt(), // 2.354
            PSFType::Moffat4 => {
                // For Moffat with beta=4: FWHM = sigma * 2 * sqrt(2^(1/4) - 1)
                avg_sigma * 2.0 * (2.0_f64.powf(0.25) - 1.0).sqrt() // ≈ 1.1895
            }
            PSFType::None => 0.0,
        }
    }

    /// Total flux under the fitted model above its background, in the same
    /// units as `amplitude`.
    ///
    /// The integral of the fitted surface: `2π·A·σx·σy` for the Gaussian and
    /// `π·A·σx·σy/(β−1)` with β = 4 for the Moffat. Unlike a sum over a
    /// fixed box, it does not grow or shrink with seeing, so it compares one
    /// star's brightness between frames. Only meaningful for a star whose
    /// core was not clipped by saturation.
    pub fn integrated_flux(&self) -> f64 {
        let area = self.sigma_x.abs() * self.sigma_y.abs();
        match self.psf_type {
            PSFType::Gaussian => 2.0 * std::f64::consts::PI * self.amplitude * area,
            PSFType::Moffat4 => std::f64::consts::PI * self.amplitude * area / 3.0,
            PSFType::None => 0.0,
        }
    }

    /// Calculate eccentricity from sigma values
    pub fn calculate_eccentricity(&self) -> f64 {
        let a = self.sigma_x.max(self.sigma_y);
        let b = self.sigma_x.min(self.sigma_y);
        if a > 0.0 {
            (1.0 - (b / a).powi(2)).sqrt()
        } else {
            0.0
        }
    }
}

/// Trait for PSF models
pub trait PSFFunction: Send + Sync {
    /// Evaluate PSF at given position with parameters
    /// parameters: [A, B, x0, y0, sigma_x, sigma_y, theta]
    fn value(&self, x: f64, y: f64, params: &[f64]) -> f64;

    /// Calculate gradient (Jacobian) of PSF with respect to parameters
    fn gradient(&self, x: f64, y: f64, params: &[f64], grad: &mut [f64]);

    /// Convert sigma to FWHM for this PSF type
    fn sigma_to_fwhm(&self, sigma: f64) -> f64;
}

/// Gaussian PSF model
pub struct GaussianPSF;

impl PSFFunction for GaussianPSF {
    fn value(&self, x: f64, y: f64, params: &[f64]) -> f64 {
        let a = params[0]; // Amplitude
        let b = params[1]; // Background
        let x0 = params[2]; // X offset
        let y0 = params[3]; // Y offset
        let sigma_x = params[4];
        let sigma_y = params[5];
        let theta = params[6]; // Rotation angle

        // Rotate coordinates
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let dx = x - x0;
        let dy = y - y0;
        let xp = dx * cos_t + dy * sin_t;
        let yp = -dx * sin_t + dy * cos_t;

        // Gaussian function
        let arg = -(xp * xp / (2.0 * sigma_x * sigma_x) + yp * yp / (2.0 * sigma_y * sigma_y));
        b + a * arg.exp()
    }

    fn gradient(&self, x: f64, y: f64, params: &[f64], grad: &mut [f64]) {
        let a = params[0];
        let x0 = params[2];
        let y0 = params[3];
        let sigma_x = params[4];
        let sigma_y = params[5];
        let theta = params[6];

        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let dx = x - x0;
        let dy = y - y0;
        let xp = dx * cos_t + dy * sin_t;
        let yp = -dx * sin_t + dy * cos_t;

        let sx2 = sigma_x * sigma_x;
        let sy2 = sigma_y * sigma_y;
        let arg = -(xp * xp / (2.0 * sx2) + yp * yp / (2.0 * sy2));
        let exp_arg = arg.exp();

        // d/dA
        grad[0] = exp_arg;

        // d/dB
        grad[1] = 1.0;

        // d/dx0
        grad[2] = a * exp_arg * (xp * cos_t / sx2 - yp * sin_t / sy2);

        // d/dy0
        grad[3] = a * exp_arg * (xp * sin_t / sx2 + yp * cos_t / sy2);

        // d/dsigma_x
        grad[4] = a * exp_arg * xp * xp / (sx2 * sigma_x);

        // d/dsigma_y
        grad[5] = a * exp_arg * yp * yp / (sy2 * sigma_y);

        // d/dtheta: d(xp)/dθ = yp and d(yp)/dθ = −xp, so
        // d(arg)/dθ = −xp·yp·(1/σx² − 1/σy²).
        grad[6] = a * exp_arg * xp * yp * (1.0 / sy2 - 1.0 / sx2);
    }

    fn sigma_to_fwhm(&self, sigma: f64) -> f64 {
        sigma * 2.0 * (2.0 * 2.0_f64.ln()).sqrt()
    }
}

/// Moffat PSF model with beta=4.0
pub struct Moffat4PSF;

impl PSFFunction for Moffat4PSF {
    fn value(&self, x: f64, y: f64, params: &[f64]) -> f64 {
        let a = params[0]; // Amplitude
        let b = params[1]; // Background
        let x0 = params[2]; // X offset
        let y0 = params[3]; // Y offset
        let u = params[4]; // Sigma X
        let v = params[5]; // Sigma Y
        let theta = params[6]; // Rotation angle

        // Rotate coordinates
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let dx = x - x0;
        let dy = y - y0;
        let xp = dx * cos_t + dy * sin_t;
        let yp = -dx * sin_t + dy * cos_t;

        // Moffat function with beta=4
        let d = 1.0 + (xp * xp) / (u * u) + (yp * yp) / (v * v);
        b + a / d.powi(4)
    }

    fn gradient(&self, x: f64, y: f64, params: &[f64], grad: &mut [f64]) {
        let a = params[0];
        let x0 = params[2];
        let y0 = params[3];
        let u = params[4];
        let v = params[5];
        let theta = params[6];

        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let dx = x - x0;
        let dy = y - y0;
        let xp = dx * cos_t + dy * sin_t;
        let yp = -dx * sin_t + dy * cos_t;

        let u2 = u * u;
        let v2 = v * v;
        let u3 = u2 * u;
        let v3 = v2 * v;
        let xp2 = xp * xp;
        let yp2 = yp * yp;

        let d = 1.0 + xp2 / u2 + yp2 / v2;
        let beta = 4.0;

        // d/dA
        grad[0] = d.powf(-beta);

        // d/dB
        grad[1] = 1.0;

        // Common terms
        let factor = -a * beta * d.powf(-beta - 1.0);

        // d/dx0
        grad[2] = factor * ((2.0 * sin_t * yp / v2) - (2.0 * cos_t * xp / u2));

        // d/dy0
        grad[3] = factor * ((-2.0 * sin_t * xp / u2) - (2.0 * cos_t * yp / v2));

        // d/du (sigma_x)
        grad[4] = (2.0 * a * beta / u3) * xp2 * d.powf(-beta - 1.0);

        // d/dv (sigma_y)
        grad[5] = (2.0 * a * beta / v3) * yp2 * d.powf(-beta - 1.0);

        // d/dtheta
        grad[6] = factor * (2.0 * yp * xp * (1.0 / u2 - 1.0 / v2));
    }

    fn sigma_to_fwhm(&self, sigma: f64) -> f64 {
        // For Moffat with beta=4: FWHM = sigma * 2 * sqrt(2^(1/4) - 1)
        sigma * 2.0 * (2.0_f64.powf(0.25) - 1.0).sqrt()
    }
}

/// Bilinear interpolation for sub-pixel sampling
pub fn bilinear_sample(data: &[u16], width: usize, height: usize, x: f64, y: f64) -> f64 {
    // Clamp to image bounds
    let x = x.max(0.0).min((width - 1) as f64);
    let y = y.max(0.0).min((height - 1) as f64);

    // Get integer coordinates
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    // Get fractional parts
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    // Get pixel values
    let p00 = data[y0 * width + x0] as f64;
    let p10 = data[y0 * width + x1] as f64;
    let p01 = data[y1 * width + x0] as f64;
    let p11 = data[y1 * width + x1] as f64;

    // Bilinear interpolation
    let p0 = p00 * (1.0 - fx) + p10 * fx;
    let p1 = p01 * (1.0 - fx) + p11 * fx;
    p0 * (1.0 - fy) + p1 * fy
}

/// Extract sub-pixel sampled ROI around a star
pub fn extract_roi(
    data: &[u16],
    width: usize,
    height: usize,
    center_x: f64,
    center_y: f64,
    roi_size: usize,
    sample_spacing: f64,
) -> (Vec<(f64, f64)>, Vec<f64>) {
    let half_size = roi_size as f64 / 2.0;
    let mut positions = Vec::new();
    let mut values = Vec::new();

    let mut y = -half_size;
    while y <= half_size {
        let mut x = -half_size;
        while x <= half_size {
            let sample_x = center_x + x;
            let sample_y = center_y + y;

            if sample_x >= 0.0
                && sample_x < width as f64
                && sample_y >= 0.0
                && sample_y < height as f64
            {
                positions.push((x, y)); // Relative to centroid
                values.push(bilinear_sample(data, width, height, sample_x, sample_y));
            }

            x += sample_spacing;
        }
        y += sample_spacing;
    }

    (positions, values)
}

/// Simple Levenberg-Marquardt optimizer for PSF fitting
pub struct LevenbergMarquardt {
    max_iterations: usize,
    tolerance: f64,
    lambda: f64,
    lambda_factor: f64,
}

impl Default for LevenbergMarquardt {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            lambda: 0.01,
            lambda_factor: 10.0,
        }
    }
}

impl LevenbergMarquardt {
    /// Fit PSF model to data
    pub fn fit(
        &mut self,
        psf: &dyn PSFFunction,
        positions: &[(f64, f64)],
        values: &[f64],
        initial_params: &[f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
    ) -> Result<Vec<f64>, String> {
        let n_params = initial_params.len();
        let n_points = positions.len();

        if n_points < n_params {
            return Err("Not enough data points for fitting".to_string());
        }

        let mut params = initial_params.to_vec();
        let mut best_params = params.clone();
        let mut best_error = f64::MAX;

        let mut jacobian = DMatrix::<f64>::zeros(n_points, n_params);
        let mut residuals = DVector::<f64>::zeros(n_points);
        let mut gradient = vec![0.0; n_params];

        for _iter in 0..self.max_iterations {
            // Calculate residuals and Jacobian
            let mut current_error = 0.0;
            for (i, ((x, y), observed)) in positions.iter().zip(values.iter()).enumerate() {
                let predicted = psf.value(*x, *y, &params);
                let residual = observed - predicted;
                residuals[i] = residual;
                current_error += residual * residual;

                psf.gradient(*x, *y, &params, &mut gradient);
                for (j, &grad) in gradient.iter().enumerate() {
                    jacobian[(i, j)] = -grad; // Negative because residual = observed - predicted
                }
            }

            if current_error < best_error {
                best_error = current_error;
                best_params = params.clone();
            }

            // Check convergence
            if current_error < self.tolerance {
                break;
            }

            // LM update. The Jacobian here is d(residual)/d(param), so the
            // descent step solves (JᵀJ + λI)·δ = −Jᵀr; solving for +Jᵀr
            // walks uphill, every step gets rejected, and the "fit" returns
            // the initial guess untouched.
            let jt = jacobian.transpose();
            let jtj = &jt * &jacobian;
            let jtr = -(&jt * &residuals);

            loop {
                // Add lambda to diagonal (LM modification)
                let mut h = jtj.clone();
                for i in 0..n_params {
                    h[(i, i)] += self.lambda;
                }

                // Solve for delta
                match h.lu().solve(&jtr) {
                    Some(delta) => {
                        // Apply bounds
                        let mut new_params = params.clone();
                        for i in 0..n_params {
                            new_params[i] = (params[i] + delta[i])
                                .max(lower_bounds[i])
                                .min(upper_bounds[i]);
                        }

                        // Evaluate new error
                        let mut new_error = 0.0;
                        for ((x, y), observed) in positions.iter().zip(values.iter()) {
                            let predicted = psf.value(*x, *y, &new_params);
                            let residual = observed - predicted;
                            new_error += residual * residual;
                        }

                        if new_error < current_error {
                            // Accept update
                            params = new_params;
                            self.lambda /= self.lambda_factor;
                            break;
                        } else {
                            // Reject update, increase lambda
                            self.lambda *= self.lambda_factor;
                            if self.lambda > 1e10 {
                                return Ok(best_params);
                            }
                        }
                    }
                    None => {
                        // Singular matrix, increase lambda
                        self.lambda *= self.lambda_factor;
                        if self.lambda > 1e10 {
                            return Ok(best_params);
                        }
                    }
                }
            }
        }

        Ok(best_params)
    }
}

/// Where the elliptical fit starts: centre offset, axes and rotation.
#[derive(Debug, Clone, Copy)]
struct InitialShape {
    x0: f64,
    y0: f64,
    sigma_x: f64,
    sigma_y: f64,
    theta: f64,
}

impl InitialShape {
    /// The star's shape from adaptive second moments, with the model's X
    /// axis on the major axis.
    ///
    /// A round start leaves the fit no way to turn: at `sigma_x == sigma_y`
    /// the residual's derivative in theta is exactly zero, and a star
    /// elongated near 45° keeps both axis-aligned widths equal, so the fit
    /// settles on a round star (#505). Starting from the moments puts it
    /// on the right ellipse already.
    #[allow(clippy::too_many_arguments)]
    fn from_moments(
        data: &[u16],
        width: usize,
        height: usize,
        center_x: f64,
        center_y: f64,
        radius: f64,
        background: f64,
        initial_sigma: f64,
        psf: &dyn PSFFunction,
    ) -> Option<Self> {
        let value = |x: isize, y: isize| {
            if x < 0 || y < 0 || x as usize >= width || y as usize >= height {
                background
            } else {
                data[y as usize * width + x as usize] as f64
            }
        };
        let moments = crate::psf_field::adaptive_moments(
            value,
            center_x,
            center_y,
            radius,
            background,
            initial_sigma,
        )?;
        let half_trace = 0.5 * (moments.xx + moments.yy);
        let spread = (0.25 * (moments.xx - moments.yy).powi(2) + moments.xy.powi(2)).sqrt();
        let major = half_trace + spread;
        let minor = half_trace - spread;
        if !(major.is_finite() && minor > 0.0) {
            return None;
        }
        // The moments give a Gaussian's sigma; match the model's FWHM to it.
        let gaussian_fwhm = 2.0 * (2.0 * 2.0_f64.ln()).sqrt();
        let scale = gaussian_fwhm / psf.sigma_to_fwhm(1.0);
        Some(Self {
            x0: moments.x - center_x,
            y0: moments.y - center_y,
            sigma_x: major.sqrt() * scale,
            sigma_y: minor.sqrt() * scale,
            theta: 0.5 * (2.0 * moments.xy).atan2(moments.xx - moments.yy),
        })
    }
}

/// Type alias for residual map data (observed, fitted, residuals)
pub type ResidualMaps = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>);

/// PSF Fitter
pub struct PSFFitter {
    psf_type: PSFType,
    roi_size: usize,
    sample_spacing: f64,
}

impl PSFFitter {
    pub fn new(psf_type: PSFType) -> Self {
        Self {
            psf_type,
            roi_size: 32,        // Default ROI size
            sample_spacing: 0.5, // Sub-pixel sampling
        }
    }

    /// Fit PSF to a star
    #[allow(clippy::too_many_arguments)]
    pub fn fit_star(
        &self,
        data: &[u16],
        width: usize,
        height: usize,
        center_x: f64,
        center_y: f64,
        bbox_width: f64,
        bbox_height: f64,
        background: f64,
        peak_brightness: f64,
    ) -> Option<PSFModel> {
        if self.psf_type == PSFType::None {
            return None;
        }

        // Extract ROI with sub-pixel sampling
        let (positions, values) = extract_roi(
            data,
            width,
            height,
            center_x,
            center_y,
            self.roi_size,
            self.sample_spacing,
        );

        if positions.len() < 10 {
            return None; // Not enough points
        }

        // Set up PSF model
        let psf: Box<dyn PSFFunction> = match self.psf_type {
            PSFType::Gaussian => Box::new(GaussianPSF),
            PSFType::Moffat4 => Box::new(Moffat4PSF),
            PSFType::None => unreachable!(),
        };

        // Bounds
        let dx_limit = bbox_width / 8.0;
        let dy_limit = bbox_height / 8.0;
        let sigma_max = ((bbox_width * bbox_width + bbox_height * bbox_height).sqrt()) / 2.0;

        // Initial parameters: [A, B, x0, y0, sigma_x, sigma_y, theta]
        let start = InitialShape::from_moments(
            data,
            width,
            height,
            center_x,
            center_y,
            self.roi_size as f64 / 2.0,
            background,
            (bbox_width.max(bbox_height) / 3.0).max(1.0),
            &*psf,
        )
        .unwrap_or(InitialShape {
            x0: 0.0,
            y0: 0.0,
            sigma_x: bbox_width / 3.0,
            sigma_y: bbox_height / 3.0,
            theta: 0.0,
        });
        let initial_params = vec![
            peak_brightness - background, // Amplitude
            background,                   // Background
            start.x0.clamp(-dx_limit, dx_limit),
            start.y0.clamp(-dy_limit, dy_limit),
            start.sigma_x.clamp(0.1, sigma_max.max(0.1)),
            start.sigma_y.clamp(0.1, sigma_max.max(0.1)),
            start.theta,
        ];

        // Theta is periodic, so its bounds sit a half turn beyond any start
        // in (−π/2, π/2]: a star elongated near ±90° must be able to rotate
        // past it rather than stop against a bound.
        let lower_bounds = vec![
            0.0,             // A must be positive
            0.0,             // B must be positive
            -dx_limit,       // x0
            -dy_limit,       // y0
            0.1,             // sigma_x minimum
            0.1,             // sigma_y minimum
            -3.0 * PI / 2.0, // theta
        ];

        let upper_bounds = vec![
            2.0 * (peak_brightness - background), // A max
            peak_brightness,                      // B max
            dx_limit,                             // x0
            dy_limit,                             // y0
            sigma_max,                            // sigma_x max
            sigma_max,                            // sigma_y max
            3.0 * PI / 2.0,                       // theta
        ];

        // Fit the model
        let mut optimizer = LevenbergMarquardt::default();
        match optimizer.fit(
            &*psf,
            &positions,
            &values,
            &initial_params,
            &lower_bounds,
            &upper_bounds,
        ) {
            Ok(fitted_params) => {
                // Calculate goodness of fit
                let mut sum_squared_residuals = 0.0;
                let mut sum_squared_total = 0.0;
                let mean_value = values.iter().sum::<f64>() / values.len() as f64;

                for ((x, y), observed) in positions.iter().zip(values.iter()) {
                    let predicted = psf.value(*x, *y, &fitted_params);
                    sum_squared_residuals += (observed - predicted).powi(2);
                    sum_squared_total += (observed - mean_value).powi(2);
                }

                let r_squared = if sum_squared_total > 0.0 {
                    1.0 - sum_squared_residuals / sum_squared_total
                } else {
                    0.0
                };

                let rmse = (sum_squared_residuals / positions.len() as f64).sqrt();

                let mut model = PSFModel {
                    psf_type: self.psf_type,
                    amplitude: fitted_params[0],
                    background: fitted_params[1],
                    x0: fitted_params[2],
                    y0: fitted_params[3],
                    sigma_x: fitted_params[4].abs(),
                    sigma_y: fitted_params[5].abs(),
                    // An ellipse repeats every half turn.
                    theta: (fitted_params[6] + PI / 2.0).rem_euclid(PI) - PI / 2.0,
                    r_squared,
                    rmse,
                    fwhm: 0.0,
                    eccentricity: 0.0,
                };

                model.fwhm = model.calculate_fwhm();
                model.eccentricity = model.calculate_eccentricity();

                Some(model)
            }
            Err(_) => None,
        }
    }

    /// Generate residual map for visualization
    pub fn generate_residuals(
        &self,
        data: &[u16],
        width: usize,
        height: usize,
        center_x: f64,
        center_y: f64,
        model: &PSFModel,
    ) -> Option<ResidualMaps> {
        let roi_half = self.roi_size as f64 / 2.0;
        let mut observed = vec![vec![0.0; self.roi_size]; self.roi_size];
        let mut fitted = vec![vec![0.0; self.roi_size]; self.roi_size];
        let mut residuals = vec![vec![0.0; self.roi_size]; self.roi_size];

        let psf: Box<dyn PSFFunction> = match self.psf_type {
            PSFType::Gaussian => Box::new(GaussianPSF),
            PSFType::Moffat4 => Box::new(Moffat4PSF),
            PSFType::None => return None,
        };

        let params = vec![
            model.amplitude,
            model.background,
            model.x0,
            model.y0,
            model.sigma_x,
            model.sigma_y,
            model.theta,
        ];

        for i in 0..self.roi_size {
            for j in 0..self.roi_size {
                let rel_x = j as f64 - roi_half + 0.5;
                let rel_y = i as f64 - roi_half + 0.5;

                // Get observed value using bilinear interpolation
                let pixel_x = center_x + rel_x;
                let pixel_y = center_y + rel_y;

                if pixel_x >= 0.0
                    && pixel_x < width as f64
                    && pixel_y >= 0.0
                    && pixel_y < height as f64
                {
                    observed[i][j] = bilinear_sample(data, width, height, pixel_x, pixel_y);
                    fitted[i][j] = psf.value(rel_x, rel_y, &params);
                    residuals[i][j] = observed[i][j] - fitted[i][j];
                }
            }
        }

        Some((observed, fitted, residuals))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(sigma_x: f64, sigma_y: f64, theta: f64) -> PSFModel {
        PSFModel {
            psf_type: PSFType::Gaussian,
            amplitude: 1.0,
            background: 0.0,
            x0: 0.0,
            y0: 0.0,
            sigma_x,
            sigma_y,
            theta,
            r_squared: 1.0,
            rmse: 0.0,
            fwhm: 1.0,
            eccentricity: 0.5,
        }
    }

    /// Sum a model over a fine grid wide enough to hold its wings.
    fn numeric_flux(model: &PSFModel) -> f64 {
        let psf: Box<dyn PSFFunction> = match model.psf_type {
            PSFType::Gaussian => Box::new(GaussianPSF),
            PSFType::Moffat4 => Box::new(Moffat4PSF),
            PSFType::None => unreachable!(),
        };
        let params = [
            model.amplitude,
            0.0,
            0.0,
            0.0,
            model.sigma_x,
            model.sigma_y,
            model.theta,
        ];
        let step = 0.05;
        let half = 120.0;
        let steps = (2.0 * half / step) as usize;
        let mut sum = 0.0;
        for row in 0..steps {
            let y = -half + (row as f64 + 0.5) * step;
            for column in 0..steps {
                let x = -half + (column as f64 + 0.5) * step;
                sum += psf.value(x, y, &params);
            }
        }
        sum * step * step
    }

    #[test]
    fn integrated_flux_matches_the_fitted_surface() {
        for psf_type in [PSFType::Gaussian, PSFType::Moffat4] {
            let mut fitted = model(2.5, 1.6, 0.4);
            fitted.psf_type = psf_type;
            fitted.amplitude = 1_000.0;
            let numeric = numeric_flux(&fitted);
            let closed = fitted.integrated_flux();
            assert!(
                (closed - numeric).abs() / numeric < 1e-3,
                "{psf_type:?}: {closed} vs {numeric}"
            );
        }
        // One star seen through worse seeing: the same light spread wider.
        let mut sharp = model(1.5, 1.5, 0.0);
        sharp.psf_type = PSFType::Moffat4;
        sharp.amplitude = 4_000.0;
        let mut soft = sharp.clone();
        soft.sigma_x = 3.0;
        soft.sigma_y = 3.0;
        soft.amplitude = 1_000.0;
        assert!((sharp.integrated_flux() - soft.integrated_flux()).abs() < 1e-9);
    }

    #[test]
    fn major_axis_orientation_uses_the_wider_fitted_axis() {
        let x_major = model(3.0, 1.0, 0.2);
        assert!((x_major.major_axis_theta() - 0.2).abs() < 1e-12);

        let y_major = model(1.0, 3.0, 0.2);
        assert!((y_major.major_axis_theta() - (0.2 + PI / 2.0)).abs() < 1e-12);
    }

    #[test]
    fn major_axis_orientation_is_normalized_as_an_axial_direction() {
        let wrapped_positive = model(1.0, 3.0, 3.0 * PI / 4.0);
        assert!((wrapped_positive.major_axis_theta() - PI / 4.0).abs() < 1e-12);

        let wrapped_negative = model(3.0, 1.0, -PI / 4.0);
        assert!((wrapped_negative.major_axis_theta() - 3.0 * PI / 4.0).abs() < 1e-12);
        assert!((0.0..PI).contains(&wrapped_negative.major_axis_theta()));
    }

    #[test]
    fn gradients_match_finite_differences() {
        let params = [900.0, 100.0, 0.3, -0.2, 3.0, 1.5, 0.6];
        let psfs: [(&str, &dyn PSFFunction); 2] =
            [("gaussian", &GaussianPSF), ("moffat4", &Moffat4PSF)];
        for (name, psf) in psfs {
            for &(x, y) in &[(1.0, 2.0), (-2.5, 0.7), (0.4, -3.1)] {
                let mut grad = [0.0; 7];
                psf.gradient(x, y, &params, &mut grad);
                for i in 0..7 {
                    let h = 1e-6 * params[i].abs().max(1.0);
                    let (mut up, mut down) = (params, params);
                    up[i] += h;
                    down[i] -= h;
                    let numeric = (psf.value(x, y, &up) - psf.value(x, y, &down)) / (2.0 * h);
                    assert!(
                        (grad[i] - numeric).abs() <= 1e-5 * numeric.abs().max(1.0),
                        "{name} d/dp{i} at ({x}, {y}): analytic {} numeric {numeric}",
                        grad[i]
                    );
                }
            }
        }
    }

    /// One elliptical Gaussian star, σ = 3.0 × 1.5 px, its major axis at
    /// `angle`, on a flat sky with a little deterministic noise.
    fn elongated_star(angle: f64) -> (Vec<u16>, usize, f64, f64) {
        let size = 64;
        let (cx, cy) = (31.3, 32.6);
        let (cos, sin) = (angle.cos(), angle.sin());
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let data = (0..size * size)
            .map(|i| {
                let (dx, dy) = ((i % size) as f64 - cx, (i / size) as f64 - cy);
                let u = dx * cos + dy * sin;
                let v = -dx * sin + dy * cos;
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let noise = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                let star = 20_000.0 * (-0.5 * (u * u / 9.0 + v * v / 2.25)).exp();
                (1000.0 + star + 20.0 * noise).round() as u16
            })
            .collect();
        (data, size, cx, cy)
    }

    /// #505: the fit started round with theta 0, where the residual does not
    /// change with theta, so stars elongated near 45° came back round.
    #[test]
    fn recovers_elongation_at_every_angle() {
        let truth = (1.0_f64 - 0.25).sqrt();
        for psf_type in [PSFType::Gaussian, PSFType::Moffat4] {
            for degrees in [0.0_f64, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5] {
                let angle = degrees.to_radians();
                let (data, size, cx, cy) = elongated_star(angle);
                // Detection hands over a rounded centroid and a square box.
                let model = PSFFitter::new(psf_type)
                    .fit_star(
                        &data,
                        size,
                        size,
                        cx.round(),
                        cy.round(),
                        16.0,
                        16.0,
                        1000.0,
                        21_000.0,
                    )
                    .unwrap_or_else(|| panic!("{psf_type:?} at {degrees}° did not fit"));
                assert!(
                    (model.eccentricity - truth).abs() < 0.05,
                    "{psf_type:?} at {degrees}°: eccentricity {} (truth {truth})",
                    model.eccentricity
                );
                let fitted = model.major_axis_theta();
                let error = (fitted - angle + PI / 2.0).rem_euclid(PI) - PI / 2.0;
                assert!(
                    error.abs() < 3.0_f64.to_radians(),
                    "{psf_type:?} at {degrees}°: major axis at {}°",
                    fitted.to_degrees()
                );
                assert!((-PI / 2.0..PI / 2.0).contains(&model.theta));
            }
        }
    }
}
