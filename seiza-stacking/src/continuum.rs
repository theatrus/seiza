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
    pub saturation: f64,
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
}
impl Default for ContinuumPhotometryOptions {
    fn default() -> Self {
        Self {
            aperture_radius: 8.0,
            annulus_inner: 12.0,
            annulus_outer: 20.0,
        }
    }
}

/// A rejected stellar measurement or an automatic-fit skip, not an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContinuumSkipReason {
    NoStars,
    TooFewStars,
    InvalidMeasurement,
    Saturated,
    TooFaint,
    NotIsolated,
    MissingCoverage,
    PsfMismatch,
    RegistrationMismatch,
    RankDeficient,
    PoorFit,
    Outlier,
    SpatialResidual,
    BrightnessTrend,
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
    pub origin: ContinuumOrigin,
    pub diagnostics: Option<ContinuumDiagnostics>,
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
        if !source.saturation.is_finite() || source.saturation <= 0.0 {
            return Err(invalid(
                "Declare each continuum input's physical saturation level in its working units",
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
        if measurement.peak >= source.saturation * options.saturation_fraction {
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
            schema_version: 1,
            sources: sources.clone(),
            coefficients: model.coefficients,
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
    validate_images(narrowband, references, sources)?;
    validate_options(options)?;
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
    let max_candidates = options.max_stars.saturating_mul(4);
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
    let mut rejected = BTreeMap::new();
    let mut measurements = Vec::new();
    for (index, position) in positions.iter().take(options.max_stars).enumerate() {
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
            let narrowband = measure(
                narrowband,
                *position,
                sources.narrowband.saturation,
                options,
                photometry,
            )?;
            let references = references
                .iter()
                .zip(&sources.references)
                .map(|(image, source)| {
                    measure(image, *position, source.saturation, options, photometry)
                })
                .collect::<std::result::Result<Vec<_>, _>>()?;
            Ok(ContinuumStar {
                narrowband,
                references,
            })
        })();
        match measured {
            Ok(star) => measurements.push(star),
            Err(reason) => *rejected.entry(reason).or_insert(0_usize) += 1,
        }
    }
    let mut outcome = fit_continuum(&measurements, sources, options)?;
    let diagnostics = match &mut outcome {
        ContinuumFitOutcome::Fitted { plan } => plan.diagnostics.as_mut().unwrap(),
        ContinuumFitOutcome::Skipped { diagnostics, .. } => diagnostics,
    };
    diagnostics.candidate_stars = positions.len().min(options.max_stars);
    for (reason, count) in rejected {
        *diagnostics.rejected.entry(reason).or_default() += count;
    }
    if let ContinuumFitOutcome::Skipped { reason, .. } = &mut outcome
        && *reason == ContinuumSkipReason::NoStars
        && !positions.is_empty()
    {
        *reason = ContinuumSkipReason::TooFewStars;
    }
    Ok(outcome)
}
fn measure(
    image: &LinearImage,
    position: [f64; 2],
    saturation: f64,
    options: &ContinuumFitOptions,
    geometry: &ContinuumPhotometryOptions,
) -> std::result::Result<ContinuumPhotometry, ContinuumSkipReason> {
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
                if value >= saturation * options.saturation_fraction {
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
    Ok(ContinuumPhotometry {
        x: x + cx,
        y: y + cy,
        flux,
        peak: aperture
            .iter()
            .map(|s| s.2)
            .fold(f64::NEG_INFINITY, f64::max),
        fwhm: 2.354820045 * variance.max(0.0).sqrt(),
        snr,
    })
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
    if plan.schema_version != 1
        || &plan.sources != sources
        || plan.coefficients.len() != references.len()
    {
        return Err(invalid(
            "Continuum plan does not match the prepared input revisions, grid or reference order",
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
