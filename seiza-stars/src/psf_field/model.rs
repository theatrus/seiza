//! A mean PSF plus principal components whose coefficients vary as
//! polynomials across the field.

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rayon::prelude::*;

use super::candidates::median_f64;

/// Terms of a cubic in `(u, v)`, the highest degree a field takes:
/// `1, u, v, u², uv, v², u³, u²v, uv², v³`.
pub(super) const MAX_TERMS: usize = 10;

/// The terms of a polynomial of `degree` in two variables.
pub(super) fn term_count(degree: usize) -> usize {
    (degree + 1) * (degree + 2) / 2
}

fn terms(u: f64, v: f64, count: usize) -> [f64; MAX_TERMS] {
    let all = [
        1.0,
        u,
        v,
        u * u,
        u * v,
        v * v,
        u * u * u,
        u * u * v,
        u * v * v,
        v * v * v,
    ];
    let mut out = [0.0; MAX_TERMS];
    out[..count].copy_from_slice(&all[..count]);
    out
}

/// One star's normalised stamp, where it sits, and how much to trust it.
pub(super) struct StarStamp {
    /// Field coordinates, each in `[−1, 1]` across the image.
    pub(super) u: f64,
    pub(super) v: f64,
    /// `size × size` values at unit aperture flux.
    pub(super) values: Vec<f64>,
    /// The per-pixel noise of `values`, as a standard deviation.
    pub(super) noise: f64,
    /// Which pixels hold the star alone; `None` when all do. Pixels near a
    /// fainter neighbour are left out of every sum.
    pub(super) valid: Option<Vec<bool>>,
}

impl StarStamp {
    fn is_valid(&self, pixel: usize) -> bool {
        self.valid.as_ref().is_none_or(|valid| valid[pixel])
    }
}

/// One channel's fitted field.
#[derive(Clone, Debug)]
pub(super) struct ChannelModel {
    pub(super) size: usize,
    pub(super) mean: Vec<f64>,
    pub(super) components: Vec<Vec<f64>>,
    /// Polynomial coefficients for each component, `MAX_TERMS` each, unused
    /// high terms zero.
    pub(super) polynomials: Vec<[f64; MAX_TERMS]>,
    /// The polynomials' degree: 0 for a global PSF, else 1 to 3.
    pub(super) degree: usize,
}

impl ChannelModel {
    /// The model at field coordinates `(u, v)`, before normalisation.
    pub(super) fn raw_at(&self, u: f64, v: f64) -> Vec<f64> {
        let basis = terms(u, v, term_count(self.degree));
        let mut values = self.mean.clone();
        for (component, polynomial) in self.components.iter().zip(&self.polynomials) {
            let coefficient: f64 = basis.iter().zip(polynomial).map(|(t, p)| t * p).sum();
            for (value, c) in values.iter_mut().zip(component) {
                *value += coefficient * c;
            }
        }
        values
    }
}

/// How well the stars fit the model.
pub(super) struct FitQuality {
    /// Each star's `χ²` per pixel, against its background noise.
    pub(super) chi_squared: Vec<f64>,
    /// Each star's RMS residual within the core over the model's RMS there.
    pub(super) relative: Vec<f64>,
}

/// The highest degree a field's polynomials take.
pub(super) const MAX_DEGREE: usize = 3;

/// Folds the polynomial degree is cross-validated over.
const FOLDS: usize = 5;

/// Fit a mean and `components` principal components to `stamps`, with
/// coefficients that vary as polynomials of `degree`, or of the degree from
/// 1 to [`MAX_DEGREE`] that predicts held-out stars best when `None`.
pub(super) fn fit(
    stamps: &[&StarStamp],
    size: usize,
    components: usize,
    degree: Option<usize>,
) -> ChannelModel {
    let pixels = size * size;
    let weights = star_weights(stamps);
    let mut mean = vec![0.0; pixels];
    let mut totals = vec![0.0; pixels];
    for (stamp, weight) in stamps.iter().zip(&weights) {
        for (pixel, (sum, value)) in mean.iter_mut().zip(&stamp.values).enumerate() {
            if stamp.is_valid(pixel) {
                *sum += weight * value;
                totals[pixel] += weight;
            }
        }
    }
    for (value, total) in mean.iter_mut().zip(&totals) {
        *value = if *total > 0.0 { *value / total } else { 0.0 };
    }
    // A pixel most stars mask, as when their neighbours share an offset,
    // rests on too few stars to trust: it takes the mean of the supported
    // pixels at its radius, and moves no component.
    let all: f64 = weights.iter().sum();
    let supported: Vec<bool> = totals.iter().map(|t| *t >= 0.25 * all).collect();
    if supported.iter().any(|s| !s) {
        let centre = (size / 2) as f64;
        let ring = |pixel: usize| -> usize {
            ((pixel % size) as f64 - centre)
                .hypot((pixel / size) as f64 - centre)
                .round() as usize
        };
        let rings = size;
        let (mut sums, mut counts) = (vec![0.0; rings], vec![0usize; rings]);
        for pixel in 0..pixels {
            if supported[pixel] {
                sums[ring(pixel)] += mean[pixel];
                counts[ring(pixel)] += 1;
            }
        }
        for pixel in 0..pixels {
            if !supported[pixel] {
                let r = ring(pixel);
                mean[pixel] = if counts[r] > 0 {
                    sums[r] / counts[r] as f64
                } else {
                    0.0
                };
            }
        }
    }
    let usable = |stamp: &StarStamp, pixel: usize| supported[pixel] && stamp.is_valid(pixel);

    let components = components.min(stamps.len().saturating_sub(1));
    if components == 0 {
        return ChannelModel {
            size,
            mean,
            components: Vec::new(),
            polynomials: Vec::new(),
            degree: 0,
        };
    }

    // Weighted principal components through the stars' Gram matrix:
    // n × n, where n is far smaller than the stamp's pixel count.
    let count = stamps.len();
    let scaled: Vec<Vec<f64>> = stamps
        .iter()
        .zip(&weights)
        .map(|(stamp, weight)| {
            // A masked pixel takes the mean, so it moves no component.
            let root = weight.sqrt();
            stamp
                .values
                .iter()
                .zip(&mean)
                .enumerate()
                .map(|(pixel, (value, m))| {
                    if usable(stamp, pixel) {
                        root * (value - m)
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();
    let rows: Vec<Vec<f64>> = (0..count)
        .into_par_iter()
        .map(|i| {
            (0..count)
                .map(|j| {
                    if j < i {
                        0.0
                    } else {
                        scaled[i].iter().zip(&scaled[j]).map(|(a, b)| a * b).sum()
                    }
                })
                .collect()
        })
        .collect();
    let mut gram = DMatrix::<f64>::zeros(count, count);
    for i in 0..count {
        for j in i..count {
            gram[(i, j)] = rows[i][j];
            gram[(j, i)] = rows[i][j];
        }
    }
    let eigen = SymmetricEigen::new(gram);
    let mut order: Vec<usize> = (0..count).collect();
    order.sort_by(|&a, &b| eigen.eigenvalues[b].total_cmp(&eigen.eigenvalues[a]));

    let mut basis = Vec::with_capacity(components);
    for &index in order.iter().take(components) {
        if eigen.eigenvalues[index].is_nan() || eigen.eigenvalues[index] <= 0.0 {
            break;
        }
        let vector = eigen.eigenvectors.column(index);
        let mut component = vec![0.0; pixels];
        for (i, row) in scaled.iter().enumerate() {
            let coefficient = vector[i];
            for (value, d) in component.iter_mut().zip(row) {
                *value += coefficient * d;
            }
        }
        let norm = component.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 0.0 {
            component.iter_mut().for_each(|v| *v /= norm);
            basis.push(component);
        }
    }

    // Each star's coefficients, then a weighted polynomial through them.
    let coefficients: Vec<Vec<f64>> = stamps
        .iter()
        .map(|stamp| {
            basis
                .iter()
                .map(|component| {
                    stamp
                        .values
                        .iter()
                        .zip(&mean)
                        .zip(component)
                        .enumerate()
                        .filter(|(pixel, _)| usable(stamp, *pixel))
                        .map(|(_, ((value, m), c))| (value - m) * c)
                        .sum()
                })
                .collect()
        })
        .collect();
    let columns: Vec<Vec<f64>> = (0..basis.len())
        .map(|k| coefficients.iter().map(|row| row[k]).collect())
        .collect();
    let degree = degree.unwrap_or_else(|| choose_degree(stamps, &weights, &columns));
    let all: Vec<usize> = (0..stamps.len()).collect();
    let polynomials = columns
        .iter()
        .map(|values| fit_polynomial(stamps, &weights, values, term_count(degree), &all))
        .collect();
    ChannelModel {
        size,
        mean,
        components: basis,
        polynomials,
        degree,
    }
}

/// The degree, from 1 to [`MAX_DEGREE`], whose polynomials best predict the
/// coefficients of stars held out of their fit, over [`FOLDS`] folds. A
/// degree needs three stars per term; a higher one must beat the lower
/// ones by 2% to be chosen. A tilted sensor whose best focus crosses the
/// frame needs a cubic, and a flat field does with a plane.
fn choose_degree(stamps: &[&StarStamp], weights: &[f64], columns: &[Vec<f64>]) -> usize {
    let count = stamps.len();
    let mut errors = Vec::new();
    for degree in 1..=MAX_DEGREE {
        let terms = term_count(degree);
        if count < 3 * terms {
            break;
        }
        let mut error = 0.0;
        for fold in 0..FOLDS {
            let train: Vec<usize> = (0..count).filter(|index| index % FOLDS != fold).collect();
            for values in columns {
                let polynomial = fit_polynomial(stamps, weights, values, terms, &train);
                for index in (fold..count).step_by(FOLDS) {
                    let basis = terms_of(stamps[index], terms);
                    let predicted: f64 = basis.iter().zip(&polynomial).map(|(t, p)| t * p).sum();
                    error += weights[index] * (values[index] - predicted).powi(2);
                }
            }
        }
        errors.push(error);
    }
    let Some(best) = errors.iter().copied().reduce(f64::min) else {
        return 1;
    };
    errors
        .iter()
        .position(|&error| error <= 1.02 * best)
        .map_or(1, |index| index + 1)
}

/// Inverse-variance weights, capped at the upper quartile so the brightest
/// stars share one weight instead of outvoting the field.
fn star_weights(stamps: &[&StarStamp]) -> Vec<f64> {
    let raw: Vec<f64> = stamps
        .iter()
        .map(|stamp| 1.0 / stamp.noise.max(1e-12).powi(2))
        .collect();
    let mut sorted = raw.clone();
    sorted.sort_by(f64::total_cmp);
    let cap = sorted[(3 * sorted.len()) / 4];
    raw.iter().map(|w| w.min(cap)).collect()
}

/// The weighted least-squares polynomial of `terms` terms through
/// `values` at the positions of the stamps `indices` picks.
fn fit_polynomial(
    stamps: &[&StarStamp],
    weights: &[f64],
    values: &[f64],
    terms: usize,
    indices: &[usize],
) -> [f64; MAX_TERMS] {
    let mut normal = DMatrix::<f64>::zeros(terms, terms);
    let mut right = DVector::<f64>::zeros(terms);
    for &index in indices {
        let (stamp, weight, value) = (stamps[index], weights[index], values[index]);
        let basis = terms_of(stamp, terms);
        for a in 0..terms {
            right[a] += weight * basis[a] * value;
            for b in 0..terms {
                normal[(a, b)] += weight * basis[a] * basis[b];
            }
        }
    }
    let mut out = [0.0; MAX_TERMS];
    if let Some(solution) = normal.clone().cholesky().map(|c| c.solve(&right)) {
        out[..terms].copy_from_slice(solution.as_slice());
    } else if let Some(solution) = normal.lu().solve(&right) {
        out[..terms].copy_from_slice(solution.as_slice());
    }
    out
}

fn terms_of(stamp: &StarStamp, count: usize) -> [f64; MAX_TERMS] {
    terms(stamp.u, stamp.v, count)
}

/// Compare each star with the model at its position. `core` marks the
/// pixels the relative residual is measured over.
pub(super) fn quality(model: &ChannelModel, stamps: &[&StarStamp], core: &[bool]) -> FitQuality {
    let (chi_squared, relative) = stamps
        .par_iter()
        .map(|stamp| {
            let predicted = model.raw_at(stamp.u, stamp.v);
            let mut chi = 0.0;
            let mut count = 0usize;
            let (mut residual, mut signal) = (0.0, 0.0);
            for (pixel, ((value, model), &in_core)) in
                stamp.values.iter().zip(&predicted).zip(core).enumerate()
            {
                if !stamp.is_valid(pixel) {
                    continue;
                }
                count += 1;
                let difference = value - model;
                chi += (difference / stamp.noise.max(1e-12)).powi(2);
                if in_core {
                    residual += difference * difference;
                    signal += model * model;
                }
            }
            (
                chi / count.max(1) as f64,
                if signal > 0.0 {
                    (residual / signal).sqrt()
                } else {
                    f64::INFINITY
                },
            )
        })
        .unzip();
    FitQuality {
        chi_squared,
        relative,
    }
}

/// The stars whose core residual is far beyond the field's typical one:
/// more than three times the median and more than 10% of the model.
pub(super) fn outliers(relative: &[f64]) -> Vec<bool> {
    let mut sorted: Vec<f64> = relative.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return vec![true; relative.len()];
    }
    let typical = median_f64(&mut sorted);
    relative
        .iter()
        .map(|&value| !value.is_finite() || (value > 3.0 * typical && value > 0.1))
        .collect()
}
