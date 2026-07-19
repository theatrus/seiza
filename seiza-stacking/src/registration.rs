use crate::{Error, LinearImage, Result};
use seiza::{DetectBackend, DetectConfig, DetectedStar};

type ScoredTransform = (usize, f64, SimilarityTransform, Vec<(usize, usize)>);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimilarityTransform {
    pub scale: f64,
    pub rotation_radians: f64,
    pub translation_x: f64,
    pub translation_y: f64,
}

impl SimilarityTransform {
    pub const IDENTITY: Self = Self {
        scale: 1.0,
        rotation_radians: 0.0,
        translation_x: 0.0,
        translation_y: 0.0,
    };

    pub fn apply(self, x: f64, y: f64) -> (f64, f64) {
        let cosine = self.rotation_radians.cos() * self.scale;
        let sine = self.rotation_radians.sin() * self.scale;
        (
            cosine * x - sine * y + self.translation_x,
            sine * x + cosine * y + self.translation_y,
        )
    }

    pub fn inverse_apply(self, x: f64, y: f64) -> (f64, f64) {
        let x = x - self.translation_x;
        let y = y - self.translation_y;
        let cosine = self.rotation_radians.cos() / self.scale;
        let sine = self.rotation_radians.sin() / self.scale;
        (cosine * x + sine * y, -sine * x + cosine * y)
    }
}

#[derive(Clone, Debug)]
pub struct RegistrationOptions {
    pub detection_sigma: f32,
    pub maximum_stars: usize,
    pub triangle_stars: usize,
    pub descriptor_tolerance: f64,
    pub scale_tolerance: f64,
    pub match_tolerance_pixels: f64,
    pub minimum_matches: usize,
    pub maximum_candidates: usize,
}

impl Default for RegistrationOptions {
    fn default() -> Self {
        Self {
            detection_sigma: 4.0,
            maximum_stars: 100,
            triangle_stars: 24,
            descriptor_tolerance: 0.015,
            scale_tolerance: 0.08,
            match_tolerance_pixels: 2.5,
            minimum_matches: 6,
            maximum_candidates: 384,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RegistrationResult {
    pub transform: SimilarityTransform,
    pub matched_stars: usize,
    pub rms_error_pixels: f64,
}

#[derive(Clone, Debug)]
pub struct Registrar {
    width: usize,
    height: usize,
    reference_stars: Vec<DetectedStar>,
    reference_triangles: Vec<Triangle>,
    options: RegistrationOptions,
}

impl Registrar {
    pub fn new(reference: &LinearImage, options: RegistrationOptions) -> Result<Self> {
        let reference_stars = detect(reference, &options);
        if reference_stars.len() < options.minimum_matches.max(3) {
            return Err(Error::Registration(format!(
                "reference frame has only {} usable stars; need at least {}",
                reference_stars.len(),
                options.minimum_matches.max(3)
            )));
        }
        let reference_triangles = triangles(&reference_stars, options.triangle_stars);
        if reference_triangles.is_empty() {
            return Err(Error::Registration(
                "reference stars do not form a usable nondegenerate triangle".into(),
            ));
        }
        Ok(Self {
            width: reference.width,
            height: reference.height,
            reference_stars,
            reference_triangles,
            options,
        })
    }

    pub fn register(&self, source: &LinearImage) -> Result<RegistrationResult> {
        if source.width != self.width || source.height != self.height {
            return Err(Error::Registration(format!(
                "source frame is {}x{} but reference is {}x{}",
                source.width, source.height, self.width, self.height
            )));
        }
        let source_stars = detect(source, &self.options);
        if source_stars.len() < self.options.minimum_matches.max(3) {
            return Err(Error::Registration(format!(
                "source frame has only {} usable stars; need at least {}",
                source_stars.len(),
                self.options.minimum_matches.max(3)
            )));
        }
        let source_triangles = triangles(&source_stars, self.options.triangle_stars);
        let mut candidates = Vec::new();
        for source_triangle in &source_triangles {
            for reference_triangle in &self.reference_triangles {
                let error = (source_triangle.ratios[0] - reference_triangle.ratios[0]).abs()
                    + (source_triangle.ratios[1] - reference_triangle.ratios[1]).abs();
                if error > self.options.descriptor_tolerance * 2.0 {
                    continue;
                }
                if let Some(transform) = transform_from_triangles(
                    source_triangle,
                    reference_triangle,
                    &source_stars,
                    &self.reference_stars,
                ) && (transform.scale - 1.0).abs() <= self.options.scale_tolerance
                {
                    candidates.push((error, transform));
                }
            }
        }
        candidates.sort_by(|left, right| left.0.total_cmp(&right.0));
        candidates.truncate(self.options.maximum_candidates);

        let mut best: Option<ScoredTransform> = None;
        for (_, candidate) in candidates {
            let pairs = matched_pairs(
                candidate,
                &source_stars,
                &self.reference_stars,
                self.options.match_tolerance_pixels,
            );
            if pairs.len() < self.options.minimum_matches {
                continue;
            }
            let squared_error =
                pair_squared_error(candidate, &pairs, &source_stars, &self.reference_stars);
            let replace = best.as_ref().is_none_or(|(count, error, _, _)| {
                pairs.len() > *count || (pairs.len() == *count && squared_error < *error)
            });
            if replace {
                best = Some((pairs.len(), squared_error, candidate, pairs));
            }
        }
        let (_, _, mut transform, mut pairs) = best.ok_or_else(|| {
            Error::Registration("no star-triangle transform reached the match threshold".into())
        })?;

        // Refit against all inliers and rematch once to remove triangle noise.
        for _ in 0..2 {
            transform = fit_similarity(&pairs, &source_stars, &self.reference_stars)?;
            pairs = matched_pairs(
                transform,
                &source_stars,
                &self.reference_stars,
                self.options.match_tolerance_pixels,
            );
        }
        if pairs.len() < self.options.minimum_matches {
            return Err(Error::Registration(
                "refined transform lost too many matches".into(),
            ));
        }
        let rms = (pair_squared_error(transform, &pairs, &source_stars, &self.reference_stars)
            / pairs.len() as f64)
            .sqrt();
        Ok(RegistrationResult {
            transform,
            matched_stars: pairs.len(),
            rms_error_pixels: rms,
        })
    }
}

fn detect(image: &LinearImage, options: &RegistrationOptions) -> Vec<DetectedStar> {
    let mut luma = image.luminance();
    normalize_for_detection(&mut luma);
    let config = DetectConfig {
        backend: DetectBackend::F32,
        sigma: options.detection_sigma,
        max_stars: options.maximum_stars,
        ..DetectConfig::default()
    };
    seiza::detect_stars_luma_f32(&luma, image.width as u32, image.height as u32, &config)
}

fn normalize_for_detection(values: &mut [f32]) {
    let stride = (values.len() / 100_000).max(1);
    let mut sample = values
        .iter()
        .step_by(stride)
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if sample.is_empty() {
        values.fill(0.0);
        return;
    }
    sample.sort_unstable_by(f32::total_cmp);
    let low = sample[sample.len() / 100];
    let high = sample[sample.len() * 99 / 100].max(low + f32::EPSILON);
    for value in values {
        *value = if value.is_finite() {
            ((*value - low) / (high - low)).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
}

#[derive(Clone, Debug)]
struct Triangle {
    vertices: [usize; 3],
    ratios: [f64; 2],
}

fn triangles(stars: &[DetectedStar], maximum_stars: usize) -> Vec<Triangle> {
    let count = stars.len().min(maximum_stars);
    let mut output = Vec::new();
    for first in 0..count.saturating_sub(2) {
        for second in first + 1..count.saturating_sub(1) {
            for third in second + 1..count {
                let mut opposite = [
                    (distance(&stars[second], &stars[third]), first),
                    (distance(&stars[first], &stars[third]), second),
                    (distance(&stars[first], &stars[second]), third),
                ];
                opposite.sort_by(|left, right| left.0.total_cmp(&right.0));
                if opposite[2].0 < 8.0 || opposite[0].0 / opposite[2].0 < 0.12 {
                    continue;
                }
                output.push(Triangle {
                    vertices: [opposite[0].1, opposite[1].1, opposite[2].1],
                    ratios: [opposite[0].0 / opposite[2].0, opposite[1].0 / opposite[2].0],
                });
            }
        }
    }
    output
}

fn distance(left: &DetectedStar, right: &DetectedStar) -> f64 {
    (left.x - right.x).hypot(left.y - right.y)
}

fn transform_from_triangles(
    source: &Triangle,
    reference: &Triangle,
    source_stars: &[DetectedStar],
    reference_stars: &[DetectedStar],
) -> Option<SimilarityTransform> {
    let pairs = (0..3)
        .map(|index| (source.vertices[index], reference.vertices[index]))
        .collect::<Vec<_>>();
    fit_similarity(&pairs, source_stars, reference_stars).ok()
}

fn matched_pairs(
    transform: SimilarityTransform,
    source: &[DetectedStar],
    reference: &[DetectedStar],
    tolerance: f64,
) -> Vec<(usize, usize)> {
    let mut candidates = Vec::new();
    for (source_index, star) in source.iter().enumerate() {
        let (x, y) = transform.apply(star.x, star.y);
        if let Some((reference_index, distance)) = reference
            .iter()
            .enumerate()
            .map(|(index, reference)| (index, (x - reference.x).hypot(y - reference.y)))
            .filter(|(_, distance)| *distance <= tolerance)
            .min_by(|left, right| left.1.total_cmp(&right.1))
        {
            candidates.push((distance, source_index, reference_index));
        }
    }
    candidates.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut used_source = vec![false; source.len()];
    let mut used_reference = vec![false; reference.len()];
    let mut pairs = Vec::new();
    for (_, source_index, reference_index) in candidates {
        if !used_source[source_index] && !used_reference[reference_index] {
            used_source[source_index] = true;
            used_reference[reference_index] = true;
            pairs.push((source_index, reference_index));
        }
    }
    pairs
}

fn fit_similarity(
    pairs: &[(usize, usize)],
    source: &[DetectedStar],
    reference: &[DetectedStar],
) -> Result<SimilarityTransform> {
    if pairs.len() < 2 {
        return Err(Error::Registration("need at least two point pairs".into()));
    }
    let count = pairs.len() as f64;
    let source_center = pairs.iter().fold((0.0, 0.0), |sum, (s, _)| {
        (sum.0 + source[*s].x, sum.1 + source[*s].y)
    });
    let reference_center = pairs.iter().fold((0.0, 0.0), |sum, (_, r)| {
        (sum.0 + reference[*r].x, sum.1 + reference[*r].y)
    });
    let source_center = (source_center.0 / count, source_center.1 / count);
    let reference_center = (reference_center.0 / count, reference_center.1 / count);
    let mut numerator_a = 0.0;
    let mut numerator_b = 0.0;
    let mut denominator = 0.0;
    for (source_index, reference_index) in pairs {
        let sx = source[*source_index].x - source_center.0;
        let sy = source[*source_index].y - source_center.1;
        let rx = reference[*reference_index].x - reference_center.0;
        let ry = reference[*reference_index].y - reference_center.1;
        numerator_a += sx * rx + sy * ry;
        numerator_b += sx * ry - sy * rx;
        denominator += sx * sx + sy * sy;
    }
    if denominator <= f64::EPSILON {
        return Err(Error::Registration(
            "degenerate matched star geometry".into(),
        ));
    }
    let a = numerator_a / denominator;
    let b = numerator_b / denominator;
    let scale = a.hypot(b);
    if !scale.is_finite() || scale <= f64::EPSILON {
        return Err(Error::Registration("invalid similarity scale".into()));
    }
    Ok(SimilarityTransform {
        scale,
        rotation_radians: b.atan2(a),
        translation_x: reference_center.0 - a * source_center.0 + b * source_center.1,
        translation_y: reference_center.1 - b * source_center.0 - a * source_center.1,
    })
}

fn pair_squared_error(
    transform: SimilarityTransform,
    pairs: &[(usize, usize)],
    source: &[DetectedStar],
    reference: &[DetectedStar],
) -> f64 {
    pairs
        .iter()
        .map(|(source_index, reference_index)| {
            let (x, y) = transform.apply(source[*source_index].x, source[*source_index].y);
            (x - reference[*reference_index].x).powi(2)
                + (y - reference[*reference_index].y).powi(2)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_known_similarity_transform() {
        let source = [(1.0, 2.0), (7.0, 3.0), (4.0, 11.0), (13.0, 9.0)]
            .into_iter()
            .map(|(x, y)| DetectedStar {
                x,
                y,
                flux: 1.0,
                peak: 1.0,
                area: 3,
            })
            .collect::<Vec<_>>();
        let expected = SimilarityTransform {
            scale: 1.02,
            rotation_radians: 0.07,
            translation_x: 4.2,
            translation_y: -2.1,
        };
        let reference = source
            .iter()
            .map(|star| {
                let (x, y) = expected.apply(star.x, star.y);
                DetectedStar {
                    x,
                    y,
                    ..star.clone()
                }
            })
            .collect::<Vec<_>>();
        let pairs = (0..source.len())
            .map(|index| (index, index))
            .collect::<Vec<_>>();
        let actual = fit_similarity(&pairs, &source, &reference).unwrap();
        assert!((actual.scale - expected.scale).abs() < 1.0e-10);
        assert!((actual.rotation_radians - expected.rotation_radians).abs() < 1.0e-10);
        assert!((actual.translation_x - expected.translation_x).abs() < 1.0e-10);
        assert!((actual.translation_y - expected.translation_y).abs() < 1.0e-10);
    }
}
