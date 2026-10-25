//! Calibrated response inversion. No filter-name or bandwidth approximation.
use crate::{ContinuumPlan, ContinuumSources, Error, LinearImage, Result};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Responses in each exact prepared input's sample units per common flux unit.
/// The producer includes exposure, gain, throughput, extinction and line/doublet
/// assumptions in the calibration revision. Source identities must match plans.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EmissionFluxCalibration {
    pub revision: String,
    pub unit: String,
    pub lines: Vec<String>,
    pub sources: Vec<ContinuumSources>,
    /// Exact model snapshots used to calibrate residual zero levels. Source
    /// revisions alone do not bind a pedestal to the subtraction weights.
    #[serde(alias = "continuum_coefficients")]
    pub continuum_coefficients: Vec<Vec<f64>>,
    /// Row = narrowband image; column = physical line.
    #[serde(alias = "narrowband_response")]
    pub narrowband_response: Vec<Vec<f64>>,
    /// Outer index = subtraction plan; then reference index; then physical line.
    #[serde(alias = "reference_response")]
    pub reference_response: Vec<Vec<Vec<f64>>>,
    /// Independently calibrated additive zero levels, in each signed residual's
    /// sample units. Zero is an explicit declaration, never an inferred sky median.
    #[serde(alias = "residual_zero_levels")]
    pub residual_zero_levels: Vec<f64>,
    /// Optional covariance of the input residual vector, in residual sample units.
    /// Unknown noise is None, never a zero uncertainty claim.
    #[serde(alias = "residual_covariance")]
    pub residual_covariance: Option<Vec<Vec<f64>>>,
}

/// Auditable effective response and propagated residual-noise covariance.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmissionFluxReport {
    pub revision: String,
    pub unit: String,
    pub lines: Vec<String>,
    pub effective_response: Vec<Vec<f64>>,
    pub condition_number: f64,
    pub flux_covariance: Option<Vec<Vec<f64>>>,
    pub uncertainty_assumptions: String,
    pub calibration: EmissionFluxCalibration,
    pub plans: Vec<ContinuumPlan>,
}

fn invalid(detail: &str) -> Error {
    Error::Color(detail.into())
}

/// Solve M F = E - calibrated_zero_levels, where M[i,k] = NB[i,k] - sum_j a[i,j] BB[i,j,k].
/// Signed flux and common finite coverage are retained. This does not calibrate
/// unknown filters, remove unmodelled lines, or change a display recipe.
/// Calibrated inputs must share a PSF: explicitly matched plans must all declare
/// the same target FWHM; mixing matched and unmatched plans is refused.
pub fn solve_emission_flux(
    residuals: &[&LinearImage],
    plans: &[ContinuumPlan],
    calibration: &EmissionFluxCalibration,
) -> Result<(Vec<LinearImage>, EmissionFluxReport)> {
    let count = plans.len();
    if !(1..=3).contains(&count)
        || residuals.len() != count
        || calibration.lines.len() != count
        || calibration.sources.len() != count
        || calibration.continuum_coefficients.len() != count
        || calibration.narrowband_response.len() != count
        || calibration.reference_response.len() != count
        || calibration.residual_zero_levels.len() != count
        || calibration
            .residual_zero_levels
            .iter()
            .any(|v| !v.is_finite())
        || calibration.revision.trim().is_empty()
        || calibration.unit.trim().is_empty()
        || calibration.unit.len() > 48
        || !calibration.unit.bytes().all(|b| (32..=126).contains(&b))
        || calibration.lines.iter().any(|s| s.trim().is_empty())
        || calibration
            .lines
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != count
    {
        return Err(invalid(
            "Flux calibration needs one to three unique lines, a revision, common unit, explicit residual zero levels and exact source responses",
        ));
    }
    let (width, height) = (residuals[0].width, residuals[0].height);
    let target = plans[0].psf_matching.as_ref().map(|p| p.target_fwhm);
    let mut matrix = DMatrix::zeros(count, count);
    for (i, (plan, image)) in plans.iter().zip(residuals).enumerate() {
        crate::continuum::validate_plan(plan)?;
        if calibration.continuum_coefficients[i] != plan.coefficients {
            return Err(invalid(
                "Continuum coefficients changed; recalibrate residual zero levels and record the exact coefficient snapshots before exporting flux",
            ));
        }
        if plan.sources != calibration.sources[i]
            || image.width.checked_mul(image.height) != Some(image.data.len())
            || image.channels != 1
            || image.width != width
            || image.height != height
            || (plan.sources.width, plan.sources.height) != (width, height)
            || plan.psf_matching.as_ref().map(|p| p.target_fwhm) != target
            || calibration.narrowband_response[i].len() != count
            || calibration.reference_response[i].len() != plan.coefficients.len()
            || calibration.reference_response[i]
                .iter()
                .any(|row| row.len() != count)
            || calibration.narrowband_response[i]
                .iter()
                .chain(calibration.reference_response[i].iter().flatten())
                .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(invalid(
                "Flux responses must match exact plan sources, reference order, common grid and PSF target, with finite nonnegative responses",
            ));
        }
        for k in 0..count {
            matrix[(i, k)] = calibration.narrowband_response[i][k]
                - plan
                    .coefficients
                    .iter()
                    .enumerate()
                    .map(|(j, a)| a * calibration.reference_response[i][j][k])
                    .sum::<f64>();
        }
        if !matrix[(i, i)].is_finite()
            || matrix[(i, i)] <= 0.0
            || matrix[(i, i)] < calibration.narrowband_response[i][i] * 1e-4
        {
            return Err(invalid(
                "Continuum subtraction leaves a nonpositive response to the requested line",
            ));
        }
    }
    if matrix.iter().any(|v| !v.is_finite()) {
        return Err(invalid(
            "Effective line responses overflow finite calibration units",
        ));
    }
    // Row scaling removes arbitrary per-image gain units from conditioning.
    let mut scaled = matrix.clone();
    for i in 0..count {
        let scale = matrix[(i, i)];
        for k in 0..count {
            scaled[(i, k)] /= scale;
        }
    }
    if scaled.iter().any(|v| !v.is_finite()) {
        return Err(invalid("Scaled line responses are not finite"));
    }
    let scale = scaled.amax();
    let normalized = scaled / scale;
    let svd = normalized.clone().svd(false, false);
    let condition = svd.singular_values.max() / svd.singular_values.min();
    if !condition.is_finite() || condition > 1000.0 {
        return Err(invalid(
            "Line response matrix is rank deficient or too ill conditioned",
        ));
    }
    let normalized_inverse = normalized
        .clone()
        .try_inverse()
        .ok_or_else(|| invalid("Line response matrix is singular"))?;
    let identity = &normalized_inverse * &normalized;
    if (0..count).any(|i| {
        (0..count).any(|j| {
            !identity[(i, j)].is_finite() || (identity[(i, j)] - f64::from(i == j)).abs() > 1e-8
        })
    }) {
        return Err(invalid(
            "Line response inverse failed its numerical identity check",
        ));
    }
    // M = D * scale * normalized, so inverse columns carry reciprocal row units.
    let inverse = DMatrix::from_fn(count, count, |i, j| {
        normalized_inverse[(i, j)] / scale / matrix[(j, j)]
    });
    if inverse.iter().any(|v| !v.is_finite()) {
        return Err(invalid("Line response inverse is not finite"));
    }
    let covariance = if let Some(rows) = &calibration.residual_covariance {
        if rows.len() != count
            || rows
                .iter()
                .any(|r| r.len() != count || r.iter().any(|v| !v.is_finite()))
        {
            return Err(invalid(
                "Residual covariance must be a finite square matrix in residual units",
            ));
        }
        let mut cov = DMatrix::from_fn(count, count, |i, j| rows[i][j]);
        let scale = cov.amax();
        if (0..count).any(|i| {
            cov[(i, i)] < 0.0
                || (0..count).any(|j| (cov[(i, j)] - cov[(j, i)]).abs() > 1e-10 * scale)
        }) {
            return Err(invalid(
                "Residual covariance must be symmetric with nonnegative variances",
            ));
        }
        cov = &cov * 0.5 + cov.transpose() * 0.5;
        if scale > 0.0 && (&cov / scale).symmetric_eigen().eigenvalues.min() < -1e-10 {
            return Err(invalid("Residual covariance must be positive semidefinite"));
        }
        let mut propagated = &inverse * cov * inverse.transpose();
        if propagated.iter().any(|v| !v.is_finite()) {
            return Err(invalid("Propagated flux covariance overflows finite units"));
        }
        for i in 0..count {
            propagated[(i, i)] = propagated[(i, i)].max(0.0);
        }
        Some(propagated)
    } else {
        None
    };
    let mut output = vec![vec![f32::NAN; width * height]; count];
    for (i, plane) in output.iter_mut().enumerate() {
        for (p, sample) in plane.iter_mut().enumerate() {
            if residuals.iter().any(|image| !image.data[p].is_finite()) {
                continue;
            }
            let value = (0..count)
                .map(|j| {
                    inverse[(i, j)]
                        * (f64::from(residuals[j].data[p]) - calibration.residual_zero_levels[j])
                })
                .sum::<f64>();
            if !value.is_finite() || value.abs() > f64::from(f32::MAX) {
                return Err(invalid("Calibrated flux overflows float32 samples"));
            }
            *sample = value as f32;
        }
    }
    let images = output
        .into_iter()
        .map(|data| LinearImage::new(width, height, 1, data))
        .collect::<Result<Vec<_>>>()?;
    let rows = |matrix: &DMatrix<f64>| {
        (0..count)
            .map(|i| (0..count).map(|j| matrix[(i, j)]).collect())
            .collect()
    };
    let report=EmissionFluxReport {calibration:calibration.clone(),plans:plans.to_vec(),revision:calibration.revision.clone(),unit:calibration.unit.clone(),lines:calibration.lines.clone(),effective_response:rows(&matrix),condition_number:condition,flux_covariance:covariance.as_ref().map(rows),uncertainty_assumptions:"Propagates only supplied residual covariance. Calibration-response uncertainty, continuum-coefficient uncertainty, zero-level uncertainty, unmodelled lines and spatial PSF errors are not included; missing covariance means unknown uncertainty.".into()};
    Ok((images, report))
}
