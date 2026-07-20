use crate::{Error, LinearImage, Result};
use seiza::{DetectBackend, DetectConfig, DetectedStar};
use std::collections::HashMap;

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

    /// Pixel displacement produced by this transform at one source position.
    pub fn displacement_at(self, x: f64, y: f64) -> f64 {
        let (mapped_x, mapped_y) = self.apply(x, y);
        (mapped_x - x).hypot(mapped_y - y)
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
    /// Absolute floor for the maximum frame-to-reference displacement.
    pub maximum_drift_pixels: f64,
    /// Fraction of the reference frame's larger dimension used for the maximum
    /// displacement. The effective bound is the larger of this and the pixel
    /// floor.
    pub maximum_drift_fraction: f64,
    pub minimum_matches: usize,
    pub maximum_candidates: usize,
}

impl Default for RegistrationOptions {
    fn default() -> Self {
        Self {
            detection_sigma: 4.0,
            maximum_stars: 200,
            triangle_stars: 24,
            descriptor_tolerance: 0.015,
            scale_tolerance: 0.08,
            match_tolerance_pixels: 2.5,
            maximum_drift_pixels: Self::DEFAULT_MAXIMUM_DRIFT_PIXELS,
            maximum_drift_fraction: Self::DEFAULT_MAXIMUM_DRIFT_FRACTION,
            minimum_matches: 6,
            maximum_candidates: 384,
        }
    }
}

impl RegistrationOptions {
    pub const DEFAULT_MAXIMUM_DRIFT_PIXELS: f64 = 256.0;
    pub const DEFAULT_MAXIMUM_DRIFT_FRACTION: f64 = 0.15;

    pub fn effective_maximum_drift_pixels(&self, width: usize, height: usize) -> f64 {
        self.maximum_drift_pixels
            .max(width.max(height) as f64 * self.maximum_drift_fraction)
    }

    pub fn validate(&self) -> Result<()> {
        if !self.detection_sigma.is_finite()
            || self.detection_sigma <= 0.0
            || self.triangle_stars < 3
            || self.maximum_stars < self.minimum_matches.max(3)
            || self.minimum_matches < 3
            || self.maximum_candidates == 0
            || !self.descriptor_tolerance.is_finite()
            || self.descriptor_tolerance <= 0.0
            || !self.scale_tolerance.is_finite()
            || !(0.0..1.0).contains(&self.scale_tolerance)
            || !self.match_tolerance_pixels.is_finite()
            || self.match_tolerance_pixels <= 0.0
            || !self.maximum_drift_pixels.is_finite()
            || self.maximum_drift_pixels <= 0.0
            || !self.maximum_drift_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.maximum_drift_fraction)
        {
            return Err(Error::Registration("invalid registration options".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct RegistrationResult {
    pub transform: SimilarityTransform,
    pub matched_stars: usize,
    pub rms_error_pixels: f64,
    pub drift_pixels: f64,
}

#[derive(Clone, Debug)]
pub struct Registrar {
    width: usize,
    height: usize,
    reference_stars: Vec<DetectedStar>,
    reference_index: StarSpatialIndex,
    reference_triangles: Vec<Triangle>,
    maximum_drift_pixels: f64,
    options: RegistrationOptions,
}

impl Registrar {
    pub fn new(reference: &LinearImage, options: RegistrationOptions) -> Result<Self> {
        options.validate()?;
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
        let reference_index =
            StarSpatialIndex::new(&reference_stars, options.match_tolerance_pixels);
        let maximum_drift_pixels =
            options.effective_maximum_drift_pixels(reference.width, reference.height);
        Ok(Self {
            width: reference.width,
            height: reference.height,
            reference_stars,
            reference_index,
            reference_triangles,
            maximum_drift_pixels,
            options,
        })
    }

    pub fn register(&self, source: &LinearImage) -> Result<RegistrationResult> {
        let source_stars = detect(source, &self.options);
        if source_stars.len() < self.options.minimum_matches.max(3) {
            return Err(Error::Registration(format!(
                "source frame has only {} usable stars; need at least {}",
                source_stars.len(),
                self.options.minimum_matches.max(3)
            )));
        }
        let mut best: Option<ScoredTransform> = None;
        for candidate in translation_candidates(
            &source_stars,
            &self.reference_stars,
            &self.options,
            self.maximum_drift_pixels,
        ) {
            retain_scored_transform(
                &mut best,
                candidate,
                &source_stars,
                &self.reference_stars,
                &self.reference_index,
                &self.options,
            );
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
                    && transform.displacement_at(self.width as f64 * 0.5, self.height as f64 * 0.5)
                        <= self.maximum_drift_pixels
                {
                    candidates.push((error, transform));
                }
            }
        }
        candidates.sort_by(|left, right| left.0.total_cmp(&right.0));
        candidates.truncate(self.options.maximum_candidates);

        for (_, candidate) in candidates {
            retain_scored_transform(
                &mut best,
                candidate,
                &source_stars,
                &self.reference_stars,
                &self.reference_index,
                &self.options,
            );
        }
        let best = best.ok_or_else(|| {
            Error::Registration(format!(
                "no registration transform within the configured {:.1}px drift reached the match threshold",
                self.maximum_drift_pixels
            ))
        })?;
        self.refine_registration(best, &source_stars)
    }

    fn refine_registration(
        &self,
        (_, _, mut transform, mut pairs): ScoredTransform,
        source_stars: &[DetectedStar],
    ) -> Result<RegistrationResult> {
        // Refit against all inliers and rematch once to remove triangle noise.
        for _ in 0..2 {
            transform = fit_similarity(&pairs, source_stars, &self.reference_stars)?;
            pairs = matched_pairs(
                transform,
                source_stars,
                &self.reference_stars,
                &self.reference_index,
                self.options.match_tolerance_pixels,
            );
        }
        if pairs.len() < self.options.minimum_matches {
            return Err(Error::Registration(
                "refined transform lost too many matches".into(),
            ));
        }
        let drift = transform.displacement_at(self.width as f64 * 0.5, self.height as f64 * 0.5);
        if drift > self.maximum_drift_pixels {
            return Err(Error::Registration(format!(
                "refined transform drift {drift:.3}px exceeds the configured {:.3}px maximum",
                self.maximum_drift_pixels
            )));
        }
        let rms = (pair_squared_error(transform, &pairs, source_stars, &self.reference_stars)
            / pairs.len() as f64)
            .sqrt();
        Ok(RegistrationResult {
            transform,
            matched_stars: pairs.len(),
            rms_error_pixels: rms,
            drift_pixels: drift,
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct TranslationVote {
    count: usize,
    sum_x: f64,
    sum_y: f64,
}

/// Seed registration from the expected low-drift overlap before trying the
/// rank-sensitive bright-star triangles. This uses every retained detection,
/// so a cropped or noisy frame can still register when its common stars do not
/// land in both top-triangle subsets.
fn translation_candidates(
    source: &[DetectedStar],
    reference: &[DetectedStar],
    options: &RegistrationOptions,
    maximum_drift_pixels: f64,
) -> Vec<SimilarityTransform> {
    let bin_size = options.match_tolerance_pixels * 2.0;
    let mut votes = HashMap::<(i32, i32), TranslationVote>::new();
    for source in source {
        for reference in reference {
            let translation_x = reference.x - source.x;
            let translation_y = reference.y - source.y;
            if translation_x.hypot(translation_y) > maximum_drift_pixels {
                continue;
            }
            let key = (
                (translation_x / bin_size).round() as i32,
                (translation_y / bin_size).round() as i32,
            );
            let vote = votes.entry(key).or_default();
            vote.count += 1;
            vote.sum_x += translation_x;
            vote.sum_y += translation_y;
        }
    }
    let mut votes = votes
        .into_iter()
        .filter(|(_, vote)| vote.count >= options.minimum_matches)
        .collect::<Vec<_>>();
    votes.sort_unstable_by(|(left_key, left), (right_key, right)| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| left_key.cmp(right_key))
    });
    votes.truncate(options.maximum_candidates.min(64));
    votes
        .into_iter()
        .map(|(_, vote)| SimilarityTransform {
            translation_x: vote.sum_x / vote.count as f64,
            translation_y: vote.sum_y / vote.count as f64,
            ..SimilarityTransform::IDENTITY
        })
        .collect()
}

fn retain_scored_transform(
    best: &mut Option<ScoredTransform>,
    candidate: SimilarityTransform,
    source: &[DetectedStar],
    reference: &[DetectedStar],
    reference_index: &StarSpatialIndex,
    options: &RegistrationOptions,
) {
    let pairs = matched_pairs(
        candidate,
        source,
        reference,
        reference_index,
        options.match_tolerance_pixels,
    );
    if pairs.len() < options.minimum_matches {
        return;
    }
    let squared_error = pair_squared_error(candidate, &pairs, source, reference);
    let replace = best.as_ref().is_none_or(|(count, error, _, _)| {
        pairs.len() > *count || (pairs.len() == *count && squared_error < *error)
    });
    if replace {
        *best = Some((pairs.len(), squared_error, candidate, pairs));
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
    let (minimum, maximum) = values.iter().filter(|value| value.is_finite()).fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(minimum, maximum), value| (minimum.min(*value as f64), maximum.max(*value as f64)),
    );
    if !minimum.is_finite() || maximum <= minimum {
        values.fill(0.0);
        return;
    }
    let range = maximum - minimum;
    for value in values {
        *value = if value.is_finite() {
            ((*value as f64 - minimum) / range) as f32
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
    reference_index: &StarSpatialIndex,
    tolerance: f64,
) -> Vec<(usize, usize)> {
    let mut candidates = Vec::new();
    for (source_index, star) in source.iter().enumerate() {
        let (x, y) = transform.apply(star.x, star.y);
        if let Some((reference_index, distance_squared)) =
            reference_index.nearest_within(x, y, reference, tolerance)
        {
            candidates.push((distance_squared, source_index, reference_index));
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

#[derive(Clone, Debug)]
struct StarSpatialIndex {
    bin_size: f64,
    bins: HashMap<(i32, i32), Vec<usize>>,
}

impl StarSpatialIndex {
    fn new(stars: &[DetectedStar], bin_size: f64) -> Self {
        let mut bins = HashMap::<(i32, i32), Vec<usize>>::new();
        for (index, star) in stars.iter().enumerate() {
            bins.entry(Self::key(star.x, star.y, bin_size))
                .or_default()
                .push(index);
        }
        Self { bin_size, bins }
    }

    fn nearest_within(
        &self,
        x: f64,
        y: f64,
        stars: &[DetectedStar],
        tolerance: f64,
    ) -> Option<(usize, f64)> {
        let (bin_x, bin_y) = Self::key(x, y, self.bin_size);
        let maximum_squared = tolerance * tolerance;
        let mut best: Option<(usize, f64)> = None;
        for offset_y in -1..=1 {
            for offset_x in -1..=1 {
                let Some(indices) = self.bins.get(&(bin_x + offset_x, bin_y + offset_y)) else {
                    continue;
                };
                for &index in indices {
                    let star = &stars[index];
                    let distance_squared = (x - star.x).powi(2) + (y - star.y).powi(2);
                    if distance_squared > maximum_squared {
                        continue;
                    }
                    let replace = best.is_none_or(|(best_index, best_distance)| {
                        distance_squared < best_distance
                            || (distance_squared == best_distance && index < best_index)
                    });
                    if replace {
                        best = Some((index, distance_squared));
                    }
                }
            }
        }
        best
    }

    fn key(x: f64, y: f64, bin_size: f64) -> (i32, i32) {
        ((x / bin_size).floor() as i32, (y / bin_size).floor() as i32)
    }
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

    fn star(x: f64, y: f64) -> DetectedStar {
        DetectedStar {
            x,
            y,
            flux: 1.0,
            peak: 1.0,
            area: 3,
        }
    }

    #[test]
    fn fits_known_similarity_transform() {
        let source = [(1.0, 2.0), (7.0, 3.0), (4.0, 11.0), (13.0, 9.0)]
            .into_iter()
            .map(|(x, y)| star(x, y))
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

    #[test]
    fn detection_normalization_preserves_bright_sample_order() {
        let mut values = vec![0.0; 1_000];
        values.extend([1.0, 2.0, 100.0, f32::NAN]);
        normalize_for_detection(&mut values);

        assert_eq!(values[1_000], 0.01);
        assert_eq!(values[1_001], 0.02);
        assert_eq!(values[1_002], 1.0);
        assert_eq!(values[1_003], 0.0);
    }

    #[test]
    fn detection_normalization_handles_flat_images() {
        let mut values = [42.0, 42.0, f32::NAN];
        normalize_for_detection(&mut values);
        assert_eq!(values, [0.0; 3]);
    }

    #[test]
    fn low_drift_seed_uses_common_stars_beyond_the_triangle_subset() {
        let mut source = (0..24)
            .map(|index| star(1_000.0 + index as f64 * 31.0, 900.0 + index as f64 * 17.0))
            .collect::<Vec<_>>();
        let common = [
            (40.0, 30.0),
            (80.0, 35.0),
            (55.0, 65.0),
            (105.0, 72.0),
            (75.0, 110.0),
            (130.0, 125.0),
        ];
        source.extend(common.into_iter().map(|(x, y)| star(x, y)));

        let mut reference = (0..24)
            .map(|index| star(-1_000.0 - index as f64 * 29.0, -800.0 - index as f64 * 19.0))
            .collect::<Vec<_>>();
        reference.extend(common.into_iter().map(|(x, y)| star(x + 7.0, y - 4.0)));

        let options = RegistrationOptions {
            match_tolerance_pixels: 1.0,
            maximum_drift_pixels: 12.0,
            minimum_matches: common.len(),
            ..RegistrationOptions::default()
        };
        let candidates =
            translation_candidates(&source, &reference, &options, options.maximum_drift_pixels);
        let expected = candidates
            .iter()
            .find(|candidate| {
                (candidate.translation_x - 7.0).abs() < 1.0e-10
                    && (candidate.translation_y + 4.0).abs() < 1.0e-10
            })
            .expect("the lower-ranked common stars should seed registration");
        assert_eq!(
            matched_pairs(
                *expected,
                &source,
                &reference,
                &StarSpatialIndex::new(&reference, 1.0),
                1.0
            )
            .len(),
            common.len()
        );
    }

    #[test]
    fn low_drift_seed_honors_the_configured_search_bound() {
        let source = (0..6)
            .map(|index| star(index as f64 * 20.0, index as f64 * 7.0))
            .collect::<Vec<_>>();
        let reference = source
            .iter()
            .map(|source| star(source.x + 30.0, source.y))
            .collect::<Vec<_>>();
        let options = RegistrationOptions {
            maximum_drift_pixels: 10.0,
            minimum_matches: source.len(),
            ..RegistrationOptions::default()
        };
        assert!(
            translation_candidates(&source, &reference, &options, options.maximum_drift_pixels)
                .is_empty()
        );
    }

    #[test]
    fn registration_options_reject_invalid_drift_bounds() {
        for maximum_drift_pixels in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            let options = RegistrationOptions {
                maximum_drift_pixels,
                ..RegistrationOptions::default()
            };
            assert!(options.validate().is_err());
        }
        for maximum_drift_fraction in [-0.1, 1.1, f64::INFINITY, f64::NAN] {
            let options = RegistrationOptions {
                maximum_drift_fraction,
                ..RegistrationOptions::default()
            };
            assert!(options.validate().is_err());
        }
    }

    #[test]
    fn effective_drift_uses_the_larger_pixel_or_fractional_bound() {
        let options = RegistrationOptions::default();
        assert_eq!(options.effective_maximum_drift_pixels(1_000, 800), 256.0);
        assert_eq!(options.effective_maximum_drift_pixels(4_000, 3_000), 600.0);
    }

    #[test]
    fn spatial_index_checks_adjacent_and_negative_bins() {
        let reference = vec![star(0.0, 5.0), star(5.1, 5.0), star(20.0, 20.0)];
        let index = StarSpatialIndex::new(&reference, 2.5);

        assert_eq!(
            index.nearest_within(-1.0, 5.0, &reference, 2.5),
            Some((0, 1.0))
        );
        let (nearest, distance_squared) = index
            .nearest_within(2.7, 5.0, &reference, 2.5)
            .expect("the adjacent bin should be searched");
        assert_eq!(nearest, 1);
        assert!((distance_squared - 2.4_f64.powi(2)).abs() < 1.0e-12);
        assert_eq!(index.nearest_within(2.5, 20.0, &reference, 2.5), None);
    }
}
