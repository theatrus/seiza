//! Stellar continuum fitting and signed subtraction on aligned linear samples.
use crate::{Error, LinearImage, Result};
use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// The sample transfer of one prepared input, declared by its producer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContinuumTransfer {
    Linear,
    Display,
}

/// Identity and physical saturation level of the exact prepared samples.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuumSource {
    pub name: String,
    /// Changes when pixels, scale, background, registration or masks change.
    pub revision: String,
    pub transfer: ContinuumTransfer,
    /// Saturation in these samples' units, never an observed image maximum.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub saturation: Option<f64>,
    /// Exact producer-supplied clipping-mask revision when no scalar ceiling is known.
    #[serde(
        default,
        rename = "clippingRevision",
        alias = "clipping_revision",
        skip_serializing_if = "Option::is_none"
    )]
    pub clipping_revision: Option<String>,
}

/// Common grid and concrete sources. Reference order defines coefficient order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuumSources {
    pub width: usize,
    pub height: usize,
    pub narrowband: ContinuumSource,
    pub references: Vec<ContinuumSource>,
}

/// Local-background-subtracted stellar measurement on one linear plane.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuumPhotometry {
    pub x: f64,
    pub y: f64,
    pub flux: f64,
    /// Peak sample including its background, on the source's declared scale.
    pub peak: f64,
    /// Gaussian-equivalent profile width measured consistently across planes.
    pub fwhm: f64,
    pub snr: f64,
}

/// Measurements of one common star; reference order matches the sources.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuumStar {
    pub narrowband: ContinuumPhotometry,
    pub references: Vec<ContinuumPhotometry>,
}

/// Conservative automatic-fit controls. There is no image-pedestal fit.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct ContinuumFitOptions {
    #[serde(alias = "minimum_stars")]
    pub minimum_stars: usize,
    #[serde(alias = "max_stars")]
    pub max_stars: usize,
    #[serde(alias = "rejection_sigma")]
    pub rejection_sigma: f64,
    #[serde(alias = "max_iterations")]
    pub max_iterations: usize,
    #[serde(alias = "max_fractional_error")]
    pub max_fractional_error: f64,
    /// Both training and held-out stars must retain this fraction after clipping.
    #[serde(alias = "minimum_inlier_fraction")]
    pub minimum_inlier_fraction: f64,
    #[serde(alias = "max_condition_number")]
    pub max_condition_number: f64,
    #[serde(alias = "max_psf_ratio")]
    pub max_psf_ratio: f64,
    #[serde(alias = "max_centroid_offset")]
    pub max_centroid_offset: f64,
    #[serde(alias = "minimum_snr")]
    pub minimum_snr: f64,
    #[serde(alias = "saturation_fraction")]
    pub saturation_fraction: f64,
    #[serde(alias = "fallback_to_single")]
    pub fallback_to_single: bool,
}
impl Default for ContinuumFitOptions {
    fn default() -> Self {
        Self {
            minimum_stars: 12,
            max_stars: 400,
            rejection_sigma: 3.5,
            max_iterations: 8,
            max_fractional_error: 0.15,
            minimum_inlier_fraction: 0.7,
            max_condition_number: 1000.0,
            max_psf_ratio: 1.3,
            max_centroid_offset: 1.0,
            minimum_snr: 20.0,
            saturation_fraction: 0.8,
            fallback_to_single: true,
        }
    }
}

/// Aperture and sky-annulus radii in pixels on the common grid.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct ContinuumPhotometryOptions {
    #[serde(alias = "aperture_radius")]
    pub aperture_radius: f64,
    #[serde(alias = "annulus_inner")]
    pub annulus_inner: f64,
    #[serde(alias = "annulus_outer")]
    pub annulus_outer: f64,
    /// Maximum noise-adjusted absolute pixel residual / stellar flux at the
    /// 90th percentile of held-out stars. None disables this independent gate.
    #[serde(alias = "max_pixel_residual")]
    pub max_pixel_residual: Option<f64>,
    /// Explicit Gaussian broadening on copies, before measurement/subtraction.
    #[serde(alias = "psf_matching")]
    pub psf_matching: Option<ContinuumPsfOptions>,
}
impl Default for ContinuumPhotometryOptions {
    fn default() -> Self {
        Self {
            aperture_radius: 8.0,
            annulus_inner: 12.0,
            annulus_outer: 20.0,
            max_pixel_residual: Some(0.15),
            psf_matching: None,
        }
    }
}

/// Broaden all continuum planes to a common Gaussian-equivalent FWHM.
/// No deconvolution is performed. Real profile wings still require validation.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct ContinuumPsfOptions {
    /// None chooses 1.05 times the largest measured FWHM, in common-grid pixels.
    #[serde(alias = "target_fwhm")]
    pub target_fwhm: Option<f64>,
}

/// The exact deterministic preparation replayed when applying a fitted plan.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ContinuumPsfMatch {
    pub measured_stars: usize,
    pub measured_fwhm: Vec<f64>,
    pub target_fwhm: f64,
    pub kernel_sigma: Vec<f64>,
    pub kernel_radius: Vec<usize>,
    pub prepared_sources: ContinuumSources,
}

/// A rejected stellar measurement or an automatic-fit skip, not an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContinuumSkipReason {
    NoStars,
    UnknownSaturation,
    TooFewStars,
    InvalidMeasurement,
    Saturated,
    TooFaint,
    NotIsolated,
    MissingCoverage,
    PsfMismatch,
    RegistrationMismatch,
    RankDeficient,
    UnstableSampling,
    PoorFit,
    Outlier,
    SpatialResidual,
    BrightnessTrend,
    PixelResidual,
}

/// Resolved stellar residuals for inspection; validation stars are held out.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContinuumStarDiagnostic {
    pub x: f64,
    pub y: f64,
    pub flux: f64,
    pub used: bool,
    pub validation: bool,
    pub fractional_residual: Option<f64>,
    pub rejection: Option<ContinuumSkipReason>,
}

/// Fit quality on stars, with separate training and held-out errors.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContinuumDiagnostics {
    pub candidate_stars: usize,
    pub measured_stars: usize,
    /// Common stars passing saturation, signal, alignment and PSF checks.
    pub eligible_stars: usize,
    /// Eligible stars retained across the flux range for the bounded fit.
    pub sampled_stars: usize,
    /// RMS prediction difference between the bounded and full eligible fits.
    pub sampling_stability_rms: Option<f64>,
    /// Median reference stellar flux ratios, normalized to the first reference.
    pub reference_flux_ratios: Vec<f64>,
    pub used_stars: usize,
    pub validation_stars: usize,
    pub rejected: BTreeMap<ContinuumSkipReason, usize>,
    pub fractional_rms: Option<f64>,
    pub validation_fractional_rms: Option<f64>,
    pub condition_number: Option<f64>,
    /// Median fractional residual by quadrant: top-left, top-right,
    /// bottom-left, bottom-right. Quadrants with fewer than three stars are null.
    pub spatial_residuals: Vec<Option<f64>>,
    /// Median faint-star residual minus median bright-star residual.
    pub brightness_residual_difference: Option<f64>,
    pub stars: Vec<ContinuumStarDiagnostic>,
    /// Independent held-out profile validation, absent for premeasured fits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pixel_validation: Option<ContinuumPixelValidation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub psf_matching: Option<Box<ContinuumPsfMatch>>,
}

/// Noise-adjusted profile errors on held-out stars. Fractions use each
/// star's NB aperture flux; core and halo are signed and cannot cancel in L1.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContinuumPixelValidation {
    pub expected_stars: usize,
    pub stars: usize,
    pub p90_absolute_fraction: f64,
    pub maximum_absolute_fraction: f64,
    pub clipped_stars: usize,
    pub clipped_profiles: usize,
    pub clipped_max_absolute_fraction: Option<f64>,
    pub median_core_fraction: f64,
    pub median_halo_fraction: f64,
    pub limit: f64,
    pub accepted: bool,
}

/// How the coefficients were obtained. Manual plans have no fit diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContinuumOrigin {
    Fitted,
    SingleReferenceFallback,
    Manual,
}

/// A linear model tied to exact prepared source identities and scales.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ContinuumPlan {
    pub schema_version: u32,
    pub sources: ContinuumSources,
    pub coefficients: Vec<f64>,
    /// Empirical stellar reference ratios for scale-covariant emission projection.
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        alias = "reference_flux_ratios"
    )]
    pub reference_flux_ratios: Vec<f64>,
    pub origin: ContinuumOrigin,
    pub diagnostics: Option<ContinuumDiagnostics>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "psf_matching"
    )]
    pub psf_matching: Option<Box<ContinuumPsfMatch>>,
}

/// Automatic fitting never silently substitutes an unsubtracted image.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ContinuumFitOutcome {
    Fitted {
        plan: ContinuumPlan,
    },
    Skipped {
        reason: ContinuumSkipReason,
        detail: String,
        diagnostics: ContinuumDiagnostics,
    },
}

fn invalid(message: &str) -> Error {
    Error::Color(message.into())
}
fn validate_sources(sources: &ContinuumSources) -> Result<()> {
    if sources.width == 0
        || sources.height == 0
        || sources.width.checked_mul(sources.height).is_none()
        || sources.references.is_empty()
        || sources.references.len() > 3
    {
        return Err(invalid(
            "Continuum subtraction requires a nonempty grid and one to three reference planes",
        ));
    }
    let mut names = BTreeSet::new();
    for source in std::iter::once(&sources.narrowband).chain(&sources.references) {
        if source.transfer != ContinuumTransfer::Linear {
            return Err(invalid(
                "Continuum subtraction requires linear, star-containing input samples before stretching",
            ));
        }
        if source.name.trim().is_empty()
            || source.revision.trim().is_empty()
            || !names.insert(&source.name)
        {
            return Err(invalid(
                "Continuum sources require distinct names and nonempty prepared revisions",
            ));
        }
        if source
            .saturation
            .is_some_and(|v| !v.is_finite() || v <= 0.0)
            || source
                .clipping_revision
                .as_ref()
                .is_some_and(|v| v.trim().is_empty())
        {
            return Err(invalid(
                "Declare each continuum input's physical saturation level or an exact clipping-mask revision",
            ));
        }
    }
    Ok(())
}
fn validate_options(options: &ContinuumFitOptions) -> Result<()> {
    if options.minimum_stars < 8
        || options.max_stars < options.minimum_stars
        || options.max_stars > 4096
        || !(1..=32).contains(&options.max_iterations)
        || [
            options.rejection_sigma,
            options.max_fractional_error,
            options.max_condition_number,
            options.max_psf_ratio,
            options.max_centroid_offset,
            options.minimum_snr,
        ]
        .iter()
        .any(|x| !x.is_finite() || *x <= 0.0)
        || options.max_condition_number < 1.0
        || options.max_psf_ratio < 1.0
        || !options.minimum_inlier_fraction.is_finite()
        || !(0.5..=1.0).contains(&options.minimum_inlier_fraction)
        || !options.saturation_fraction.is_finite()
        || !(0.0..1.0).contains(&options.saturation_fraction)
        || options.saturation_fraction == 0.0
    {
        return Err(invalid(
            "Invalid continuum fit options: need at least eight stars, bounded sampling, positive finite limits and a saturation fraction in (0, 1)",
        ));
    }
    Ok(())
}
fn validate_images(
    narrowband: &LinearImage,
    references: &[&LinearImage],
    sources: &ContinuumSources,
) -> Result<()> {
    validate_sources(sources)?;
    if references.len() != sources.references.len() {
        return Err(invalid(
            "Continuum reference count does not match source identities",
        ));
    }
    for image in std::iter::once(narrowband).chain(references.iter().copied()) {
        if image.channels != 1
            || (image.width, image.height) != (sources.width, sources.height)
            || sources.width.checked_mul(sources.height) != Some(image.data.len())
        {
            return Err(invalid(
                "Continuum images must be mono planes on the declared common grid",
            ));
        }
    }
    Ok(())
}
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    sorted[(sorted.len() - 1) / 2] / 2.0 + sorted[sorted.len() / 2] / 2.0
}
fn rms(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |sum, x| sum.hypot(*x)) / (values.len() as f64).sqrt()
}
fn reject(diagnostics: &mut ContinuumDiagnostics, reason: ContinuumSkipReason) {
    *diagnostics.rejected.entry(reason).or_default() += 1;
}
fn skipped(
    reason: ContinuumSkipReason,
    detail: &str,
    diagnostics: ContinuumDiagnostics,
) -> ContinuumFitOutcome {
    ContinuumFitOutcome::Skipped {
        reason,
        detail: detail.into(),
        diagnostics,
    }
}
fn star_rejection(
    star: &ContinuumStar,
    sources: &ContinuumSources,
    options: &ContinuumFitOptions,
) -> Option<ContinuumSkipReason> {
    if star.references.len() != sources.references.len() {
        return Some(ContinuumSkipReason::InvalidMeasurement);
    }
    for (measurement, source) in std::iter::once((&star.narrowband, &sources.narrowband))
        .chain(star.references.iter().zip(&sources.references))
    {
        if [
            measurement.x,
            measurement.y,
            measurement.flux,
            measurement.peak,
            measurement.fwhm,
            measurement.snr,
        ]
        .iter()
        .any(|x| !x.is_finite())
            || measurement.x < 0.0
            || measurement.y < 0.0
            || measurement.x >= sources.width as f64
            || measurement.y >= sources.height as f64
            || measurement.flux <= 0.0
            || measurement.fwhm <= 0.0
        {
            return Some(ContinuumSkipReason::InvalidMeasurement);
        }
        if source
            .saturation
            .is_some_and(|level| measurement.peak >= level * options.saturation_fraction)
        {
            return Some(ContinuumSkipReason::Saturated);
        }
        if measurement.snr < options.minimum_snr {
            return Some(ContinuumSkipReason::TooFaint);
        }
    }
    for reference in &star.references {
        let ratio = reference.fwhm / star.narrowband.fwhm;
        if ratio.max(1.0 / ratio) > options.max_psf_ratio {
            return Some(ContinuumSkipReason::PsfMismatch);
        }
        if (reference.x - star.narrowband.x).hypot(reference.y - star.narrowband.y)
            > options.max_centroid_offset
        {
            return Some(ContinuumSkipReason::RegistrationMismatch);
        }
        if !(reference.flux / star.narrowband.flux).is_finite() {
            return Some(ContinuumSkipReason::InvalidMeasurement);
        }
    }
    None
}

// Column-scaled, through-origin nonnegative least squares. At most three
// references make exhaustive active-set selection bounded (seven systems).
fn nnls(matrix: &DMatrix<f64>, rows: &[usize]) -> Option<DVector<f64>> {
    let columns = matrix.ncols();
    let mut best = None;
    let mut best_error = f64::INFINITY;
    for mask in 1..(1_usize << columns) {
        let active: Vec<_> = (0..columns).filter(|c| mask & (1 << c) != 0).collect();
        let a = DMatrix::from_fn(rows.len(), active.len(), |r, c| {
            matrix[(rows[r], active[c])]
        });
        let gram = a.transpose() * &a;
        let b = a.transpose() * DVector::from_element(rows.len(), 1.0);
        let Some(cholesky) = gram.cholesky() else {
            continue;
        };
        let solution = cholesky.solve(&b);
        if solution.iter().any(|x| !x.is_finite() || *x < -1e-10) {
            continue;
        }
        let mut coefficients = DVector::zeros(columns);
        for (c, value) in active.iter().zip(solution.iter()) {
            coefficients[*c] = value.max(0.0);
        }
        let error = rows
            .iter()
            .map(|r| (1.0 - matrix.row(*r).dot(&coefficients.transpose())).powi(2))
            .sum::<f64>();
        if error < best_error {
            best = Some(coefficients);
            best_error = error;
        }
    }
    best
}
fn condition(matrix: &DMatrix<f64>, rows: &[usize]) -> Option<f64> {
    let a = DMatrix::from_fn(rows.len(), matrix.ncols(), |r, c| matrix[(rows[r], c)]);
    let eigen = (a.transpose() * a).symmetric_eigen().eigenvalues;
    let low = eigen.min();
    let high = eigen.max();
    (low > 1e-14 * high && high > 0.0).then(|| (high / low).sqrt())
}
struct Model {
    coefficients: Vec<f64>,
    training: Vec<usize>,
    validation: Vec<usize>,
    residuals: Vec<f64>,
    condition: f64,
    training_rms: f64,
    validation_rms: f64,
}
fn fit_model(
    stars: &[&ContinuumStar],
    columns: &[usize],
    options: &ContinuumFitOptions,
) -> Option<Model> {
    let count = stars.len();
    let training: Vec<_> = (0..count).filter(|i| i % 4 != 0).collect();
    let validation: Vec<_> = (0..count).filter(|i| i % 4 == 0).collect();
    let mut matrix = DMatrix::from_fn(count, columns.len(), |r, c| {
        stars[r].references[columns[c]].flux / stars[r].narrowband.flux
    });
    let scales: Vec<_> = (0..columns.len())
        .map(|c| {
            training
                .iter()
                .fold(0.0_f64, |n, r| n.hypot(matrix[(*r, c)]))
                / (training.len() as f64).sqrt()
        })
        .collect();
    if scales.iter().any(|x| !x.is_finite() || *x <= 0.0) {
        return None;
    }
    for c in 0..columns.len() {
        for r in 0..count {
            matrix[(r, c)] /= scales[c];
        }
    }
    let initial_condition = condition(&matrix, &training)?;
    if initial_condition > options.max_condition_number {
        return None;
    }
    let residuals_for = |coefficients: &DVector<f64>| {
        (0..count)
            .map(|r| 1.0 - matrix.row(r).dot(&coefficients.transpose()))
            .collect::<Vec<_>>()
    };
    let mut coefficients = nnls(&matrix, &training)?;
    let initial_residuals = residuals_for(&coefficients);
    let mut best_score = median(
        &training
            .iter()
            .map(|i| initial_residuals[*i].abs())
            .collect::<Vec<_>>(),
    );
    // Deterministic small-subset starts prevent a few high-leverage stars
    // from defining the initial curve used for sigma rejection.
    for trial in 0..64_usize {
        let mut rows = Vec::new();
        let mut state = (trial as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15);
        for _ in 0..columns.len() * 8 {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let row = training[(state as usize) % training.len()];
            if !rows.contains(&row) {
                rows.push(row);
            }
            if rows.len() == columns.len() {
                break;
            }
        }
        if rows.len() != columns.len() {
            continue;
        }
        if let Some(candidate) = nnls(&matrix, &rows) {
            let residuals = residuals_for(&candidate);
            let score = median(
                &training
                    .iter()
                    .map(|i| residuals[*i].abs())
                    .collect::<Vec<_>>(),
            );
            if score < best_score {
                coefficients = candidate;
                best_score = score;
            }
        }
    }
    let mut used = training.clone();
    for _ in 0..options.max_iterations {
        let residuals = residuals_for(&coefficients);
        let current: Vec<_> = used.iter().map(|i| residuals[*i]).collect();
        let center = median(&current);
        let sigma = 1.4826
            * median(
                &current
                    .iter()
                    .map(|r| (r - center).abs())
                    .collect::<Vec<_>>(),
            );
        let threshold = (options.rejection_sigma * sigma).max(1e-7);
        let next: Vec<_> = training
            .iter()
            .copied()
            .filter(|i| (residuals[*i] - center).abs() <= threshold)
            .collect();
        if next.len() < columns.len() + 3 {
            return None;
        }
        let next_coefficients = nnls(&matrix, &next)?;
        if next == used {
            coefficients = next_coefficients;
            break;
        }
        used = next;
        coefficients = next_coefficients;
    }
    let residuals = residuals_for(&coefficients);
    let current: Vec<_> = used.iter().map(|i| residuals[*i]).collect();
    let center = median(&current);
    let sigma = 1.4826
        * median(
            &current
                .iter()
                .map(|r| (r - center).abs())
                .collect::<Vec<_>>(),
        );
    let threshold = (options.rejection_sigma * sigma).max(1e-7);
    let validation_count = validation.len();
    let validation: Vec<_> = validation
        .into_iter()
        .filter(|i| (residuals[*i] - center).abs() <= threshold)
        .collect();
    if validation.len() < (options.minimum_stars / 4).max(2)
        || used.len() + validation.len() < options.minimum_stars
        || (used.len() as f64) < training.len() as f64 * options.minimum_inlier_fraction
        || (validation.len() as f64) < validation_count as f64 * options.minimum_inlier_fraction
    {
        return None;
    }
    let training_rms = rms(&used.iter().map(|i| residuals[*i]).collect::<Vec<_>>());
    let validation_rms = rms(&validation.iter().map(|i| residuals[*i]).collect::<Vec<_>>());
    let final_condition = condition(&matrix, &used)?;
    if training_rms.max(validation_rms) > options.max_fractional_error
        || final_condition > options.max_condition_number
    {
        return None;
    }
    let mut output = vec![0.0; stars[0].references.len()];
    for (c, column) in columns.iter().enumerate() {
        output[*column] = coefficients[c] / scales[c];
    }
    if output.iter().any(|x| !x.is_finite()) || output.iter().all(|x| *x <= 0.0) {
        return None;
    }
    Some(Model {
        coefficients: output,
        training: used,
        validation,
        residuals,
        condition: final_condition,
        training_rms,
        validation_rms,
    })
}

/// Fit common stellar aperture fluxes with a nonnegative through-origin model.
/// Images must be aligned linear, star-containing revisions. Every fourth good
/// star is held out; isolated-star photometry can instead be measured by
/// measure_and_fit_continuum. No catalog or filter throughput is inferred.
pub fn fit_continuum(
    stars: &[ContinuumStar],
    sources: &ContinuumSources,
    options: &ContinuumFitOptions,
) -> Result<ContinuumFitOutcome> {
    validate_sources(sources)?;
    if std::iter::once(&sources.narrowband)
        .chain(&sources.references)
        .any(|source| source.saturation.is_none() && source.clipping_revision.is_none())
    {
        return Err(invalid(
            "Automatic continuum fitting requires a physical saturation ceiling or producer clipping mask for every source",
        ));
    }
    validate_options(options)?;
    let mut diagnostics = ContinuumDiagnostics {
        candidate_stars: stars.len().min(options.max_stars),
        ..Default::default()
    };
    let mut good = Vec::new();
    let mut indices = Vec::new();
    for star in stars.iter().take(options.max_stars) {
        let rejection = star_rejection(star, sources, options);
        let index = diagnostics.stars.len();
        // Do not serialize nonfinite caller values in a diagnostic response.
        diagnostics.stars.push(ContinuumStarDiagnostic {
            x: if star.narrowband.x.is_finite() {
                star.narrowband.x
            } else {
                0.0
            },
            y: if star.narrowband.y.is_finite() {
                star.narrowband.y
            } else {
                0.0
            },
            flux: if star.narrowband.flux.is_finite() {
                star.narrowband.flux
            } else {
                0.0
            },
            used: false,
            validation: false,
            fractional_residual: None,
            rejection,
        });
        if let Some(reason) = rejection {
            reject(&mut diagnostics, reason);
        } else {
            good.push(star);
            indices.push(index);
        }
    }
    diagnostics.measured_stars = stars.len().min(options.max_stars);
    diagnostics.eligible_stars = good.len();
    diagnostics.sampled_stars = good.len();
    if !good.is_empty() {
        diagnostics.reference_flux_ratios = (0..sources.references.len())
            .map(|column| {
                median(
                    &good
                        .iter()
                        .map(|star| star.references[column].flux / star.references[0].flux)
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
    }
    if good.len() < options.minimum_stars {
        let reason = if stars.is_empty() {
            ContinuumSkipReason::NoStars
        } else if diagnostics
            .rejected
            .get(&ContinuumSkipReason::PsfMismatch)
            .copied()
            .unwrap_or(0)
            >= options.minimum_stars
        {
            ContinuumSkipReason::PsfMismatch
        } else if diagnostics
            .rejected
            .get(&ContinuumSkipReason::RegistrationMismatch)
            .copied()
            .unwrap_or(0)
            >= options.minimum_stars
        {
            ContinuumSkipReason::RegistrationMismatch
        } else {
            ContinuumSkipReason::TooFewStars
        };
        return Ok(skipped(
            reason,
            "Too few common isolated, unsaturated, PSF-compatible stars",
            diagnostics,
        ));
    }
    let columns: Vec<_> = (0..sources.references.len()).collect();
    let training: Vec<_> = (0..good.len()).filter(|i| i % 4 != 0).collect();
    let mut check = DMatrix::from_fn(good.len(), columns.len(), |r, c| {
        good[r].references[c].flux / good[r].narrowband.flux
    });
    for c in 0..columns.len() {
        let n = check
            .column(c)
            .iter()
            .fold(0.0_f64, |n, value| n.hypot(*value));
        if n > 0.0 && n.is_finite() {
            for r in 0..check.nrows() {
                check[(r, c)] /= n;
            }
        } else {
            return Ok(skipped(
                ContinuumSkipReason::RankDeficient,
                "Continuum reference fluxes cannot be scaled reliably",
                diagnostics,
            ));
        }
    }
    let identifiable =
        condition(&check, &training).is_some_and(|c| c <= options.max_condition_number);
    let mut origin = ContinuumOrigin::Fitted;
    let model = if identifiable {
        fit_model(&good, &columns, options)
    } else if options.fallback_to_single {
        origin = ContinuumOrigin::SingleReferenceFallback;
        columns
            .iter()
            .filter_map(|c| fit_model(&good, &[*c], options))
            .min_by(|a, b| a.validation_rms.total_cmp(&b.validation_rms))
    } else {
        None
    };
    let Some(model) = model else {
        return Ok(skipped(
            if identifiable {
                ContinuumSkipReason::PoorFit
            } else {
                ContinuumSkipReason::RankDeficient
            },
            "Stellar continuum fit failed rank, conditioning or held-out residual checks",
            diagnostics,
        ));
    };
    diagnostics.used_stars = model.training.len();
    diagnostics.validation_stars = model.validation.len();
    diagnostics.fractional_rms = Some(model.training_rms);
    diagnostics.validation_fractional_rms = Some(model.validation_rms);
    diagnostics.condition_number = Some(model.condition);
    let mut quadrants: [Vec<f64>; 4] = Default::default();
    let mut brightness = Vec::new();
    for (i, star) in good.iter().enumerate() {
        let used = model.training.contains(&i);
        let validation = model.validation.contains(&i);
        let entry = &mut diagnostics.stars[indices[i]];
        entry.used = used;
        entry.validation = validation;
        entry.fractional_residual = Some(model.residuals[i]);
        if !used && !validation {
            entry.rejection = Some(ContinuumSkipReason::Outlier);
            reject(&mut diagnostics, ContinuumSkipReason::Outlier);
            continue;
        }
        let quadrant = usize::from(star.narrowband.x >= sources.width as f64 / 2.0)
            + 2 * usize::from(star.narrowband.y >= sources.height as f64 / 2.0);
        quadrants[quadrant].push(model.residuals[i]);
        brightness.push((star.narrowband.flux, model.residuals[i]));
    }
    diagnostics.spatial_residuals = quadrants
        .iter()
        .map(|q| (q.len() >= 3).then(|| median(q)))
        .collect();
    brightness.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let part = (brightness.len() / 4).max(1);
    diagnostics.brightness_residual_difference = Some(
        median(&brightness[..part].iter().map(|v| v.1).collect::<Vec<_>>())
            - median(
                &brightness[brightness.len() - part..]
                    .iter()
                    .map(|v| v.1)
                    .collect::<Vec<_>>(),
            ),
    );
    if diagnostics
        .spatial_residuals
        .iter()
        .flatten()
        .any(|v| v.abs() > options.max_fractional_error)
    {
        return Ok(skipped(
            ContinuumSkipReason::SpatialResidual,
            "Stellar residuals vary across the field",
            diagnostics,
        ));
    }
    if diagnostics.brightness_residual_difference.unwrap().abs() > options.max_fractional_error {
        return Ok(skipped(
            ContinuumSkipReason::BrightnessTrend,
            "Stellar residuals vary with brightness",
            diagnostics,
        ));
    }
    Ok(ContinuumFitOutcome::Fitted {
        plan: ContinuumPlan {
            psf_matching: None,
            schema_version: 1,
            sources: sources.clone(),
            coefficients: model.coefficients,
            reference_flux_ratios: diagnostics.reference_flux_ratios.clone(),
            origin,
            diagnostics: Some(diagnostics),
        },
    })
}

/// Measure isolated aperture fluxes on the exact aligned linear planes, then
/// fit them. Empty positions requests Seiza detection on the narrowband plane.
/// Saturation is supplied in working units; image maxima are never used.
pub fn measure_and_fit_continuum(
    narrowband: &LinearImage,
    references: &[&LinearImage],
    sources: &ContinuumSources,
    positions: &[[f64; 2]],
    options: &ContinuumFitOptions,
    photometry: &ContinuumPhotometryOptions,
) -> Result<ContinuumFitOutcome> {
    measure_and_fit_inner(
        narrowband, references, sources, None, positions, options, photometry,
    )
}

/// Measure with producer-derived saturation/clipping masks on the exact grid.
/// One byte plane per input (NB then references), zero usable and one clipped.
/// Masks need exact revisions in sources; scalar saturation may then be absent.
/// Masks affect stellar eligibility, not subtraction samples or flux scales.
#[allow(clippy::too_many_arguments)]
pub fn measure_and_fit_continuum_masked(
    narrowband: &LinearImage,
    references: &[&LinearImage],
    sources: &ContinuumSources,
    clipping_masks: &[&[u8]],
    positions: &[[f64; 2]],
    options: &ContinuumFitOptions,
    photometry: &ContinuumPhotometryOptions,
) -> Result<ContinuumFitOutcome> {
    measure_and_fit_inner(
        narrowband,
        references,
        sources,
        Some(clipping_masks),
        positions,
        options,
        photometry,
    )
}

#[allow(clippy::too_many_arguments)]
fn measure_and_fit_inner(
    narrowband: &LinearImage,
    references: &[&LinearImage],
    sources: &ContinuumSources,
    clipping_masks: Option<&[&[u8]]>,
    positions: &[[f64; 2]],
    options: &ContinuumFitOptions,
    photometry: &ContinuumPhotometryOptions,
) -> Result<ContinuumFitOutcome> {
    validate_images(narrowband, references, sources)?;
    validate_options(options)?;
    let source_list: Vec<_> = std::iter::once(&sources.narrowband)
        .chain(&sources.references)
        .collect();
    match clipping_masks {
        Some(masks)
            if masks.len() == source_list.len()
                && masks.iter().all(|mask| {
                    mask.len() == sources.width * sources.height && mask.iter().all(|v| *v <= 1)
                })
                && source_list
                    .iter()
                    .all(|source| source.clipping_revision.is_some()) => {}
        None if source_list
            .iter()
            .all(|source| source.clipping_revision.is_none()) => {}
        _ => {
            return Err(invalid(
                "Continuum clipping masks need one binary plane and exact mask revision per source",
            ));
        }
    }
    if photometry
        .max_pixel_residual
        .is_some_and(|limit| !limit.is_finite() || limit <= 0.0 || limit > 1.0)
    {
        return Err(invalid(
            "Pixel residual limit must be greater than zero and at most one",
        ));
    }
    if ![
        photometry.aperture_radius,
        photometry.annulus_inner,
        photometry.annulus_outer,
    ]
    .iter()
    .all(|r| r.is_finite())
        || photometry.aperture_radius < 2.0
        || photometry.annulus_inner <= photometry.aperture_radius
        || photometry.annulus_outer <= photometry.annulus_inner
        || photometry.annulus_outer > 128.0
    {
        return Err(invalid(
            "Continuum aperture needs 2 <= radius < annulus inner < annulus outer <= 128 pixels",
        ));
    }
    let maximum_target = 32.0_f64
        .min(photometry.aperture_radius)
        .min(photometry.annulus_inner / 1.5);
    if photometry
        .psf_matching
        .as_ref()
        .and_then(|p| p.target_fwhm)
        .is_some_and(|target| !target.is_finite() || target <= 0.0 || target > maximum_target)
    {
        return Err(invalid(
            "PSF target must fit inside the aperture, with sky annulus inner radius at least 1.5 times the target FWHM (maximum 32 pixels)",
        ));
    }
    // Keep the search population independent of the usable-star budget.
    let max_candidates = 4096;
    let positions = if positions.is_empty() {
        let width = u32::try_from(sources.width)
            .map_err(|_| invalid("Continuum image is too wide for star detection"))?;
        let height = u32::try_from(sources.height)
            .map_err(|_| invalid("Continuum image is too tall for star detection"))?;
        seiza::detect_stars_luma_f32(
            &narrowband.data,
            width,
            height,
            &seiza::DetectConfig {
                sigma: 5.0,
                max_stars: max_candidates,
                ..Default::default()
            },
        )
        .iter()
        .map(|s| [s.x, s.y])
        .collect::<Vec<_>>()
    } else {
        positions.iter().take(max_candidates).copied().collect()
    };
    if photometry.psf_matching.is_some()
        && source_list
            .iter()
            .any(|source| source.saturation.is_none() && source.clipping_revision.is_none())
    {
        return Err(invalid(
            "Automatic continuum PSF matching requires a physical saturation ceiling or producer clipping mask for every original source",
        ));
    }
    if let Some(matching) = &photometry.psf_matching {
        let images: Vec<_> = std::iter::once(narrowband)
            .chain(references.iter().copied())
            .collect();
        let relaxed = ContinuumFitOptions {
            max_psf_ratio: f64::MAX,
            ..options.clone()
        };
        let mut widths = vec![Vec::new(); images.len()];
        for position in &positions {
            if positions.iter().any(|other| {
                other != position
                    && (other[0] - position[0]).hypot(other[1] - position[1])
                        < 2.0 * photometry.aperture_radius + 2.0
            }) {
                continue;
            }
            let measured: Option<Vec<_>> = images
                .iter()
                .zip(&source_list)
                .enumerate()
                .map(|(i, (image, source))| {
                    measure(
                        image,
                        *position,
                        source.saturation,
                        clipping_masks.map(|m| m[i]),
                        options,
                        photometry,
                    )
                    .ok()
                })
                .collect();
            let Some(measured) = measured else { continue };
            if star_rejection(
                &ContinuumStar {
                    narrowband: measured[0].clone(),
                    references: measured[1..].to_vec(),
                },
                sources,
                &relaxed,
            )
            .is_some()
            {
                continue;
            }
            for (samples, star) in widths.iter_mut().zip(measured) {
                samples.push(star.fwhm);
            }
        }
        if widths[0].len() < options.minimum_stars {
            return Ok(skipped(
                ContinuumSkipReason::TooFewStars,
                "Too few common unsaturated stars to measure PSF matching kernels",
                ContinuumDiagnostics::default(),
            ));
        }
        let measured_fwhm: Vec<_> = widths.iter().map(|w| median(w)).collect();
        let largest = measured_fwhm.iter().copied().fold(0.0, f64::max);
        let target = matching.target_fwhm.unwrap_or(largest * 1.05);
        if matching.target_fwhm.is_none() && target > maximum_target {
            return Ok(skipped(
                ContinuumSkipReason::PsfMismatch,
                "Measured PSFs are too broad for the current aperture/sky annulus; increase the photometry radii before matching",
                ContinuumDiagnostics::default(),
            ));
        }
        if !target.is_finite() || target < largest || target > maximum_target {
            return Err(invalid(&format!(
                "PSF target must be at least every measured FWHM ({largest:.2} pixels maximum) and at most {maximum_target:.2} pixels for this aperture/sky annulus"
            )));
        }
        let sigma: Vec<_> = measured_fwhm
            .iter()
            .map(|f| ((target * target - f * f).max(0.0)).sqrt() / 2.354820045)
            .collect();
        let radii: Vec<_> = sigma
            .iter()
            .map(|s| {
                if *s < 0.05 {
                    0
                } else {
                    (4.0 * s).ceil() as usize
                }
            })
            .collect();
        let prepared_sources = prepared_psf_sources(sources, &sigma, &radii)?;
        let report = ContinuumPsfMatch {
            measured_stars: widths[0].len(),
            measured_fwhm,
            target_fwhm: target,
            kernel_sigma: sigma,
            kernel_radius: radii,
            prepared_sources,
        };
        let prepared = apply_psf_preparation(&images, &report)?;
        let expanded_masks: Vec<_> = images
            .iter()
            .zip(&source_list)
            .enumerate()
            .map(|(i, (image, source))| {
                let mask: Vec<_> = image
                    .data
                    .iter()
                    .enumerate()
                    .map(|(p, v)| {
                        u8::from(
                            clipping_masks.is_some_and(|m| m[i][p] != 0)
                                || source.saturation.is_some_and(|s| {
                                    f64::from(*v) >= s * options.saturation_fraction
                                }),
                        )
                    })
                    .collect();
                dilate_square(
                    &mask,
                    sources.width,
                    sources.height,
                    report.kernel_radius[i],
                )
            })
            .collect();
        let masks: Vec<_> = expanded_masks.iter().map(Vec::as_slice).collect();
        let mut geometry = photometry.clone();
        geometry.psf_matching = None;
        let mut outcome = measure_and_fit_inner(
            &prepared[0],
            &prepared[1..].iter().collect::<Vec<_>>(),
            &report.prepared_sources,
            Some(&masks),
            &positions,
            options,
            &geometry,
        )?;
        if let ContinuumFitOutcome::Fitted { plan } = &mut outcome {
            plan.sources = sources.clone();
            plan.psf_matching = Some(Box::new(report.clone()));
        }
        match &mut outcome {
            ContinuumFitOutcome::Fitted { plan } => {
                plan.diagnostics.as_mut().unwrap().psf_matching = Some(Box::new(report))
            }
            ContinuumFitOutcome::Skipped { diagnostics, .. } => {
                diagnostics.psf_matching = Some(Box::new(report))
            }
        }
        return Ok(outcome);
    }
    let mut rejected = BTreeMap::new();
    let mut measurements = Vec::new();
    let mut measured_count = 0;
    let mut clipped_positions = Vec::new();
    for (index, position) in positions.iter().enumerate() {
        let measured = (|| {
            if !position.iter().all(|v| v.is_finite()) {
                return Err(ContinuumSkipReason::InvalidMeasurement);
            }
            let isolation = 2.0 * photometry.aperture_radius + 2.0;
            if positions.iter().enumerate().any(|(other, p)| {
                other != index && (p[0] - position[0]).hypot(p[1] - position[1]) < isolation
            }) {
                return Err(ContinuumSkipReason::NotIsolated);
            }
            let [x, y] = *position;
            let outer = photometry.annulus_outer;
            if x - outer >= 0.0
                && y - outer >= 0.0
                && x + outer < sources.width as f64
                && y + outer < sources.height as f64
                && std::iter::once(narrowband)
                    .chain(references.iter().copied())
                    .zip(&source_list)
                    .enumerate()
                    .any(|(i, (image, source))| {
                        aperture_has_clipping(
                            image,
                            *position,
                            source.saturation,
                            clipping_masks.map(|m| m[i]),
                            options.saturation_fraction,
                            photometry.aperture_radius,
                        )
                    })
            {
                clipped_positions.push(*position);
            }
            let narrowband = measure(
                narrowband,
                *position,
                sources.narrowband.saturation,
                clipping_masks.map(|masks| masks[0]),
                options,
                photometry,
            )?;
            let references = references
                .iter()
                .zip(&sources.references)
                .enumerate()
                .map(|(i, (image, source))| {
                    measure(
                        image,
                        *position,
                        source.saturation,
                        clipping_masks.map(|masks| masks[i + 1]),
                        options,
                        photometry,
                    )
                })
                .collect::<std::result::Result<Vec<_>, _>>()?;
            measured_count += 1;
            let star = ContinuumStar {
                narrowband,
                references,
            };
            if let Some(reason) = star_rejection(&star, sources, options) {
                return Err(reason);
            }
            Ok(star)
        })();
        match measured {
            Ok(star) => measurements.push(star),
            Err(reason) => {
                *rejected.entry(reason).or_insert(0_usize) += 1;
            }
        }
    }
    // Rejections do not spend the usable-star budget. Sample the whole
    // eligible brightness range, rather than stopping on the brightest stars.
    let eligible_count = measurements.len();
    measurements.sort_by(|a, b| {
        b.narrowband
            .flux
            .total_cmp(&a.narrowband.flux)
            .then_with(|| a.narrowband.x.total_cmp(&b.narrowband.x))
            .then_with(|| a.narrowband.y.total_cmp(&b.narrowband.y))
    });
    let sampled: Vec<_> = if measurements.len() > options.max_stars {
        let last = measurements.len() - 1;
        (0..options.max_stars)
            .map(|i| measurements[i * last / (options.max_stars - 1)].clone())
            .collect()
    } else {
        measurements.clone()
    };
    let mut outcome = fit_continuum(&sampled, sources, options)?;
    if eligible_count > sampled.len() && matches!(outcome, ContinuumFitOutcome::Fitted { .. }) {
        let audit_options = ContinuumFitOptions {
            max_stars: eligible_count,
            ..options.clone()
        };
        let audit = fit_continuum(&measurements, sources, &audit_options)?;
        match (&mut outcome, audit) {
            (ContinuumFitOutcome::Fitted { plan }, ContinuumFitOutcome::Fitted { plan: full }) => {
                let differences: Vec<_> = measurements
                    .iter()
                    .map(|star| {
                        star.references
                            .iter()
                            .zip(plan.coefficients.iter().zip(&full.coefficients))
                            .map(|(r, (a, b))| (a - b) * r.flux / star.narrowband.flux)
                            .sum::<f64>()
                    })
                    .collect();
                let difference = rms(&differences);
                plan.diagnostics.as_mut().unwrap().sampling_stability_rms = Some(difference);
                if !difference.is_finite() || difference > options.max_fractional_error * 0.5 {
                    outcome = skipped(
                        ContinuumSkipReason::UnstableSampling,
                        "Continuum predictions change across the bounded and full eligible star samples",
                        plan.diagnostics.clone().unwrap(),
                    );
                }
            }
            (
                _,
                ContinuumFitOutcome::Skipped {
                    reason,
                    detail,
                    diagnostics,
                },
            ) => {
                outcome = ContinuumFitOutcome::Skipped {
                    reason,
                    detail: format!("Broader eligible-star validation: {detail}"),
                    diagnostics,
                };
            }
            _ => unreachable!(),
        }
    }
    if let (Some(limit), ContinuumFitOutcome::Fitted { plan }) =
        (photometry.max_pixel_residual, &mut outcome)
    {
        let mut report =
            validate_pixel_profiles(narrowband, references, plan, options, photometry, limit);
        if !clipped_positions.is_empty() {
            // Clipped stars never train the model. Inspect their observed
            // residual profiles separately so a clean unsaturated sample cannot
            // conceal a visible saturated core/ring. No lost flux is inferred.
            let mut audit = plan.clone();
            audit.diagnostics.as_mut().unwrap().stars = clipped_positions
                .iter()
                .map(|p| ContinuumStarDiagnostic {
                    x: p[0],
                    y: p[1],
                    flux: 1.0,
                    used: false,
                    validation: true,
                    fractional_residual: None,
                    rejection: None,
                })
                .collect();
            let clipped =
                validate_pixel_profiles(narrowband, references, &audit, options, photometry, limit);
            report.clipped_stars = clipped_positions.len();
            report.clipped_profiles = clipped.stars;
            report.clipped_max_absolute_fraction =
                (clipped.stars > 0).then_some(clipped.maximum_absolute_fraction);
            report.accepted &= clipped.stars == clipped_positions.len()
                && clipped.maximum_absolute_fraction <= limit;
        }
        let accepted = report.accepted;
        plan.diagnostics.as_mut().unwrap().pixel_validation = Some(report);
        if !accepted {
            outcome = skipped(
                ContinuumSkipReason::PixelResidual,
                "Held-out or clipped stellar pixel profiles retain excessive cores or halos; match the PSFs or inspect the residual before using manual weights",
                plan.diagnostics.clone().unwrap(),
            );
        }
    }
    let diagnostics = match &mut outcome {
        ContinuumFitOutcome::Fitted { plan } => plan.diagnostics.as_mut().unwrap(),
        ContinuumFitOutcome::Skipped { diagnostics, .. } => diagnostics,
    };
    diagnostics.candidate_stars = positions.len();
    diagnostics.measured_stars = measured_count;
    diagnostics.eligible_stars = eligible_count;
    diagnostics.sampled_stars = sampled.len();
    for (reason, count) in rejected {
        *diagnostics.rejected.entry(reason).or_default() += count;
    }
    if let ContinuumFitOutcome::Skipped {
        reason,
        diagnostics,
        ..
    } = &mut outcome
        && matches!(
            *reason,
            ContinuumSkipReason::NoStars | ContinuumSkipReason::TooFewStars
        )
        && !positions.is_empty()
    {
        *reason = [
            ContinuumSkipReason::PsfMismatch,
            ContinuumSkipReason::RegistrationMismatch,
        ]
        .into_iter()
        .find(|r| diagnostics.rejected.get(r).copied().unwrap_or_default() >= options.minimum_stars)
        .unwrap_or(ContinuumSkipReason::TooFewStars);
    }
    Ok(outcome)
}
fn aperture_has_clipping(
    image: &LinearImage,
    position: [f64; 2],
    saturation: Option<f64>,
    mask: Option<&[u8]>,
    fraction: f64,
    radius: f64,
) -> bool {
    let [x, y] = position;
    for py in (y - radius).floor() as usize..=(y + radius).ceil() as usize {
        for px in (x - radius).floor() as usize..=(x + radius).ceil() as usize {
            if (px as f64 - x).hypot(py as f64 - y) > radius {
                continue;
            }
            let index = py * image.width + px;
            if mask.is_some_and(|m| m[index] != 0)
                || saturation.is_some_and(|s| f64::from(image.data[index]) >= s * fraction)
            {
                return true;
            }
        }
    }
    false
}

fn measure(
    image: &LinearImage,
    position: [f64; 2],
    saturation: Option<f64>,
    clipping_mask: Option<&[u8]>,
    options: &ContinuumFitOptions,
    geometry: &ContinuumPhotometryOptions,
) -> std::result::Result<ContinuumPhotometry, ContinuumSkipReason> {
    measure_profile(
        image,
        position,
        saturation,
        clipping_mask,
        options,
        geometry,
    )
    .map(|p| p.0)
}

type StellarProfile = (ContinuumPhotometry, Vec<(f64, f64, f64)>, f64);
fn measure_profile(
    image: &LinearImage,
    position: [f64; 2],
    saturation: Option<f64>,
    clipping_mask: Option<&[u8]>,
    options: &ContinuumFitOptions,
    geometry: &ContinuumPhotometryOptions,
) -> std::result::Result<StellarProfile, ContinuumSkipReason> {
    use ContinuumSkipReason::*;
    let [x, y] = position;
    let outer = geometry.annulus_outer;
    if x - outer < 0.0
        || y - outer < 0.0
        || x + outer >= image.width as f64
        || y + outer >= image.height as f64
    {
        return Err(MissingCoverage);
    }
    let bounds = (
        (x - outer).floor() as usize,
        (y - outer).floor() as usize,
        ((x + outer).ceil() as usize).min(image.width - 1),
        ((y + outer).ceil() as usize).min(image.height - 1),
    );
    let mut sky = Vec::new();
    let mut aperture = Vec::new();
    let mut sky_area = 0;
    for py in bounds.1..=bounds.3 {
        for px in bounds.0..=bounds.2 {
            let dx = px as f64 - x;
            let dy = py as f64 - y;
            let distance = dx.hypot(dy);
            let value = f64::from(image.data[py * image.width + px]);
            if distance <= geometry.aperture_radius {
                if !value.is_finite() {
                    return Err(MissingCoverage);
                }
                if saturation.is_some_and(|level| value >= level * options.saturation_fraction)
                    || clipping_mask.is_some_and(|mask| mask[py * image.width + px] != 0)
                {
                    return Err(Saturated);
                }
                aperture.push((dx, dy, value));
            } else if (geometry.annulus_inner..=outer).contains(&distance) {
                sky_area += 1;
                if value.is_finite() {
                    sky.push((dx, dy, value));
                }
            }
        }
    }
    if sky.len() < 20 || sky.len() * 4 < sky_area * 3 || aperture.len() < 4 {
        return Err(MissingCoverage);
    }
    let pedestal = median(&sky.iter().map(|s| s.2).collect::<Vec<_>>());
    let mut accepted: Vec<_> = (0..sky.len()).collect();
    let mut plane = Vector3::zeros();
    for _ in 0..4 {
        let mut normal = Matrix3::zeros();
        let mut target = Vector3::zeros();
        for i in &accepted {
            let (dx, dy, v) = sky[*i];
            let row = Vector3::new(1.0, dx, dy);
            normal += row * row.transpose();
            target += row * (v - pedestal);
        }
        plane = normal.lu().solve(&target).ok_or(MissingCoverage)?;
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
        if next.len() < 20 {
            return Err(MissingCoverage);
        }
        if next == accepted {
            break;
        }
        accepted = next;
    }
    let sky_residuals: Vec<_> = accepted
        .iter()
        .map(|i| {
            let (dx, dy, v) = sky[*i];
            v - pedestal - plane.dot(&Vector3::new(1.0, dx, dy))
        })
        .collect();
    let sky_center = median(&sky_residuals);
    let noise = 1.4826
        * median(
            &sky_residuals
                .iter()
                .map(|r| (r - sky_center).abs())
                .collect::<Vec<_>>(),
        );
    let signal: Vec<_> = aperture
        .iter()
        .map(|(dx, dy, v)| {
            (
                *dx,
                *dy,
                v - pedestal - plane.dot(&Vector3::new(1.0, *dx, *dy)),
            )
        })
        .collect();
    let flux = signal.iter().map(|s| s.2).sum::<f64>();
    let peak = signal.iter().map(|s| s.2).fold(f64::NEG_INFINITY, f64::max);
    let snr = (flux / (noise * (aperture.len() as f64).sqrt()).max(1e-100)).min(f64::MAX);
    if flux <= 0.0 || peak <= 0.0 || snr < options.minimum_snr {
        return Err(TooFaint);
    }
    let floor = (0.05 * peak).max(3.0 * noise);
    let core: Vec<_> = signal.iter().filter(|s| s.2 > floor).collect();
    let weight = core.iter().map(|s| s.2).sum::<f64>();
    if core.len() < 3 || weight <= 0.0 {
        return Err(TooFaint);
    }
    let cx = core.iter().map(|s| s.0 * s.2).sum::<f64>() / weight;
    let cy = core.iter().map(|s| s.1 * s.2).sum::<f64>() / weight;
    let variance = core
        .iter()
        .map(|s| ((s.0 - cx).powi(2) + (s.1 - cy).powi(2)) * s.2)
        .sum::<f64>()
        / (2.0 * weight);
    Ok((
        ContinuumPhotometry {
            x: x + cx,
            y: y + cy,
            flux,
            peak: aperture
                .iter()
                .map(|s| s.2)
                .fold(f64::NEG_INFINITY, f64::max),
            fwhm: 2.354820045 * variance.max(0.0).sqrt(),
            snr,
        },
        signal,
        noise,
    ))
}

fn validate_pixel_profiles(
    narrowband: &LinearImage,
    references: &[&LinearImage],
    plan: &ContinuumPlan,
    options: &ContinuumFitOptions,
    geometry: &ContinuumPhotometryOptions,
    limit: f64,
) -> ContinuumPixelValidation {
    let mut absolute = Vec::new();
    let mut cores = Vec::new();
    let mut halos = Vec::new();
    for star in plan
        .diagnostics
        .as_ref()
        .unwrap()
        .stars
        .iter()
        .filter(|s| s.validation && s.rejection.is_none())
    {
        let position = [star.x, star.y];
        let Ok((n, patch, noise)) =
            measure_profile(narrowband, position, None, None, options, geometry)
        else {
            continue;
        };
        let reference_profiles: Option<Vec<_>> = references
            .iter()
            .map(|image| measure_profile(image, position, None, None, options, geometry).ok())
            .collect();
        let Some(reference_profiles) = reference_profiles else {
            continue;
        };
        let sigma = (noise.powi(2)
            + reference_profiles
                .iter()
                .zip(&plan.coefficients)
                .map(|(p, a)| (a * p.2).powi(2))
                .sum::<f64>())
        .sqrt();
        let mut l1 = 0.0;
        let mut core = 0.0;
        let mut halo = 0.0;
        for (i, (dx, dy, value)) in patch.iter().enumerate() {
            let residual = value
                - reference_profiles
                    .iter()
                    .zip(&plan.coefficients)
                    .map(|(p, a)| a * p.1[i].2)
                    .sum::<f64>();
            l1 += (residual.abs() - 3.0 * sigma).max(0.0);
            if dx.hypot(*dy) <= n.fwhm * 0.5 {
                core += residual;
            } else {
                halo += residual;
            }
        }
        absolute.push(l1 / n.flux);
        cores.push(core / n.flux);
        halos.push(halo / n.flux);
    }
    absolute.sort_by(f64::total_cmp);
    let p90 = absolute
        .get((absolute.len().saturating_sub(1) * 9).div_ceil(10))
        .copied()
        .unwrap_or(1.0);
    let expected_stars = plan
        .diagnostics
        .as_ref()
        .unwrap()
        .stars
        .iter()
        .filter(|s| s.validation && s.rejection.is_none())
        .count();
    ContinuumPixelValidation {
        expected_stars,
        stars: absolute.len(),
        p90_absolute_fraction: p90,
        maximum_absolute_fraction: absolute.last().copied().unwrap_or(1.0),
        clipped_stars: 0,
        clipped_profiles: 0,
        clipped_max_absolute_fraction: None,
        median_core_fraction: if cores.is_empty() {
            0.0
        } else {
            median(&cores)
        },
        median_halo_fraction: if halos.is_empty() {
            0.0
        } else {
            median(&halos)
        },
        limit,
        accepted: absolute.len() >= (options.minimum_stars / 4).max(3)
            && absolute.len() as f64 >= expected_stars as f64 * options.minimum_inlier_fraction
            && p90 <= limit,
    }
}

fn prepared_psf_sources(
    sources: &ContinuumSources,
    sigma: &[f64],
    radii: &[usize],
) -> Result<ContinuumSources> {
    use sha2::{Digest, Sha256};
    let key = serde_json::to_vec(&(crate::OUTPUT_VERSION, sources, sigma, radii))
        .map_err(|e| invalid(&e.to_string()))?;
    let digest = Sha256::digest(key)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    let revision = format!("gaussian-psf:{digest}");
    let mut prepared = sources.clone();
    for (i, source) in std::iter::once(&mut prepared.narrowband)
        .chain(&mut prepared.references)
        .enumerate()
    {
        source.revision = format!("{}:{revision}:{i}", source.revision);
        source.clipping_revision = Some(format!("{revision}:clipping:{i}"));
        source.saturation = None;
    }
    Ok(prepared)
}

// Separable rolling box maximum: bounded O(pixels), independent of kernel area.
fn dilate_square(mask: &[u8], width: usize, height: usize, radius: usize) -> Vec<u8> {
    let mut horizontal = vec![0; mask.len()];
    let mut output = vec![0; mask.len()];
    for y in 0..height {
        let mut sum = 0usize;
        for x in 0..width {
            if x == 0 {
                sum = mask[y * width..y * width + (radius + 1).min(width)]
                    .iter()
                    .map(|v| usize::from(*v != 0))
                    .sum();
            } else {
                if x > radius {
                    sum -= usize::from(mask[y * width + x - radius - 1] != 0);
                }
                if x + radius < width {
                    sum += usize::from(mask[y * width + x + radius] != 0);
                }
            }
            horizontal[y * width + x] = u8::from(sum > 0);
        }
    }
    for x in 0..width {
        let mut sum = 0usize;
        for y in 0..height {
            if y == 0 {
                sum = (0..(radius + 1).min(height))
                    .map(|r| usize::from(horizontal[r * width + x] != 0))
                    .sum();
            } else {
                if y > radius {
                    sum -= usize::from(horizontal[(y - radius - 1) * width + x] != 0);
                }
                if y + radius < height {
                    sum += usize::from(horizontal[(y + radius) * width + x] != 0);
                }
            }
            output[y * width + x] = u8::from(sum > 0);
        }
    }
    output
}

fn apply_psf_preparation(
    images: &[&LinearImage],
    report: &ContinuumPsfMatch,
) -> Result<Vec<LinearImage>> {
    images
        .iter()
        .zip(report.kernel_sigma.iter().zip(&report.kernel_radius))
        .map(|(image, (sigma, radius))| {
            if *radius == 0 {
                return Ok((*image).clone());
            }
            let invalid: Vec<_> = image
                .data
                .iter()
                .map(|v| u8::from(!v.is_finite()))
                .collect();
            let invalid = dilate_square(&invalid, image.width, image.height, *radius);
            let finite: Vec<_> = image
                .data
                .iter()
                .map(|v| if v.is_finite() { *v } else { 0.0 })
                .collect();
            let mut data = seiza_imgproc::blur::gaussian_blur_f32(
                &finite,
                image.width,
                image.height,
                2 * radius + 1,
                *sigma,
                seiza_imgproc::BorderMode::Reflect101,
            );
            for (i, value) in data.iter_mut().enumerate() {
                let (x, y) = (i % image.width, i / image.width);
                if invalid[i] != 0
                    || x < *radius
                    || y < *radius
                    || x + radius >= image.width
                    || y + radius >= image.height
                {
                    *value = f32::NAN;
                }
            }
            LinearImage::new(image.width, image.height, 1, data)
        })
        .collect()
}

/// Subtract the plan on exactly its prepared linear revisions. Negative
/// residuals are retained; missing contributing coverage stays NaN. Coefficients
/// include the actual scales of these inputs. No clipping, pedestal, stretch,
/// luminance, line-throughput correction or implicit rescaling is applied.
pub fn subtract_continuum(
    narrowband: &LinearImage,
    references: &[&LinearImage],
    sources: &ContinuumSources,
    plan: &ContinuumPlan,
) -> Result<LinearImage> {
    validate_images(narrowband, references, sources)?;
    validate_plan(plan)?;
    if &plan.sources != sources {
        return Err(invalid(
            "Continuum plan does not match the prepared input revisions, grid or reference order",
        ));
    }
    if let Some(report) = &plan.psf_matching {
        let images: Vec<_> = std::iter::once(narrowband)
            .chain(references.iter().copied())
            .collect();
        let prepared = apply_psf_preparation(&images, report)?;
        let mut matched = plan.clone();
        matched.psf_matching = None;
        matched.sources = report.prepared_sources.clone();
        return subtract_continuum(
            &prepared[0],
            &prepared[1..].iter().collect::<Vec<_>>(),
            &matched.sources,
            &matched,
        );
    }
    let mut output = Vec::with_capacity(narrowband.data.len());
    for (i, n) in narrowband.data.iter().enumerate() {
        let mut value = f64::from(*n);
        let mut covered = n.is_finite();
        for (image, a) in references.iter().zip(&plan.coefficients) {
            if *a > 0.0 {
                let b = image.data[i];
                covered &= b.is_finite();
                value -= a * f64::from(b);
            }
        }
        if !covered {
            output.push(f32::NAN);
            continue;
        }
        if !value.is_finite() || value.abs() > f64::from(f32::MAX) {
            return Err(invalid(
                "Continuum subtraction overflows the output sample range",
            ));
        }
        output.push(value as f32);
    }
    LinearImage::new(narrowband.width, narrowband.height, 1, output)
}

pub(crate) fn validate_plan(plan: &ContinuumPlan) -> Result<()> {
    validate_sources(&plan.sources)?;
    if let Some(report) = &plan.psf_matching {
        let count = plan.sources.references.len() + 1;
        if report.kernel_sigma.len() != count
            || report.kernel_radius.len() != count
            || report.measured_fwhm.len() != count
            || !report.target_fwhm.is_finite()
            || report.target_fwhm <= 0.0
            || report.target_fwhm > 32.0
            || report
                .measured_fwhm
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.0 || *v > report.target_fwhm)
            || report
                .kernel_sigma
                .iter()
                .zip(&report.kernel_radius)
                .any(|(s, r)| {
                    !s.is_finite()
                        || *s < 0.0
                        || *s > 14.0
                        || *r
                            != if *s < 0.05 {
                                0
                            } else {
                                (4.0 * s).ceil() as usize
                            }
                })
            || report
                .measured_fwhm
                .iter()
                .zip(&report.kernel_sigma)
                .any(|(f, s)| {
                    let expected =
                        (report.target_fwhm.powi(2) - f.powi(2)).max(0.0).sqrt() / 2.354820045;
                    (s - expected).abs() > 1e-10 * expected.max(1.0)
                })
            || report.prepared_sources
                != prepared_psf_sources(&plan.sources, &report.kernel_sigma, &report.kernel_radius)?
        {
            return Err(invalid(
                "Continuum PSF preparation does not match the exact source revisions or bounded kernels",
            ));
        }
    }
    if plan.schema_version != 1 || plan.coefficients.len() != plan.sources.references.len() {
        return Err(invalid(
            "Continuum plan does not match its schema or reference order",
        ));
    }
    if plan.coefficients.iter().any(|a| !a.is_finite() || *a < 0.0) {
        return Err(invalid(
            "Continuum coefficients must be finite and nonnegative",
        ));
    }
    if (plan.origin == ContinuumOrigin::Manual) != plan.diagnostics.is_none() {
        return Err(invalid(
            "Manual continuum plans have no fit diagnostics; fitted plans must retain theirs",
        ));
    }
    if !plan.reference_flux_ratios.is_empty()
        && (plan.reference_flux_ratios.len() != plan.coefficients.len()
            || plan
                .reference_flux_ratios
                .iter()
                .any(|r| !r.is_finite() || *r <= 0.0))
    {
        return Err(invalid(
            "Continuum reference flux ratios must be positive, finite and match the reference count",
        ));
    }
    Ok(())
}
