use crate::{Error, LinearImage, Result};
use rayon::prelude::*;
use seiza::{DetectBackend, DetectConfig, DetectedStar};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

type ScoredTransform = (usize, f64, SimilarityTransform, Vec<(usize, usize)>);

/// A source-to-reference mapping combining uniform scale, rotation, and
/// translation.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct SimilarityTransform {
    /// Uniform scale factor.
    pub scale: f64,
    /// Rotation angle in radians.
    pub rotation_radians: f64,
    /// Horizontal translation in pixels.
    pub translation_x: f64,
    /// Vertical translation in pixels.
    pub translation_y: f64,
}

/// A source-to-reference affine mapping. Unlike [`SimilarityTransform`], this
/// can represent a parity flip or unequal axis scales, which are needed when
/// a solved image is resampled onto a canonical sky grid.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct AffineTransform {
    /// Linear source-to-reference coordinate transform.
    pub matrix: [[f64; 2]; 2],
    /// Horizontal translation in reference pixels.
    pub translation_x: f64,
    /// Vertical translation in reference pixels.
    pub translation_y: f64,
}

/// A rectangular region on the reference image's pixel grid.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct ReferenceRegion {
    /// Horizontal origin in reference pixels.
    pub x: usize,
    /// Vertical origin in reference pixels.
    pub y: usize,
    /// Region width in pixels.
    pub width: usize,
    /// Region height in pixels.
    pub height: usize,
}

impl SimilarityTransform {
    /// The transform that leaves every coordinate unchanged.
    pub const IDENTITY: Self = Self {
        scale: 1.0,
        rotation_radians: 0.0,
        translation_x: 0.0,
        translation_y: 0.0,
    };

    /// Check that this transform can be inverted for resampling.
    pub fn validate(self) -> Result<()> {
        if !self.scale.is_finite()
            || self.scale <= 0.0
            || !self.rotation_radians.is_finite()
            || !self.translation_x.is_finite()
            || !self.translation_y.is_finite()
        {
            return Err(Error::Registration(
                "resampling transform must be finite with a positive scale".into(),
            ));
        }
        Ok(())
    }

    /// Map a source coordinate forward to reference space.
    pub fn apply(self, x: f64, y: f64) -> (f64, f64) {
        let cosine = self.rotation_radians.cos() * self.scale;
        let sine = self.rotation_radians.sin() * self.scale;
        (
            cosine * x - sine * y + self.translation_x,
            sine * x + cosine * y + self.translation_y,
        )
    }

    /// Map a reference coordinate back to source space.
    pub fn inverse_apply(self, x: f64, y: f64) -> (f64, f64) {
        self.inverse_map()(x, y)
    }

    /// Compose this source-to-intermediate mapping with an
    /// intermediate-to-reference mapping. The returned transform applies
    /// `self` first and `next` second.
    pub fn then(self, next: Self) -> Self {
        let next_cosine = next.rotation_radians.cos() * next.scale;
        let next_sine = next.rotation_radians.sin() * next.scale;
        Self {
            scale: self.scale * next.scale,
            rotation_radians: self.rotation_radians + next.rotation_radians,
            translation_x: next_cosine * self.translation_x - next_sine * self.translation_y
                + next.translation_x,
            translation_y: next_sine * self.translation_x
                + next_cosine * self.translation_y
                + next.translation_y,
        }
    }

    /// The inverse mapping with its trigonometry evaluated once, for
    /// per-pixel loops.
    pub(crate) fn inverse_map(self) -> impl Fn(f64, f64) -> (f64, f64) {
        let cosine = self.rotation_radians.cos() / self.scale;
        let sine = self.rotation_radians.sin() / self.scale;
        let (translation_x, translation_y) = (self.translation_x, self.translation_y);
        move |x, y| {
            let x = x - translation_x;
            let y = y - translation_y;
            (cosine * x + sine * y, -sine * x + cosine * y)
        }
    }

    /// Express this similarity as a general affine transform.
    pub fn as_affine(self) -> AffineTransform {
        let cosine = self.rotation_radians.cos() * self.scale;
        let sine = self.rotation_radians.sin() * self.scale;
        AffineTransform {
            matrix: [[cosine, -sine], [sine, cosine]],
            translation_x: self.translation_x,
            translation_y: self.translation_y,
        }
    }

    /// Pixel displacement produced by this transform at one source position.
    pub fn displacement_at(self, x: f64, y: f64) -> f64 {
        let (mapped_x, mapped_y) = self.apply(x, y);
        (mapped_x - x).hypot(mapped_y - y)
    }
}

impl AffineTransform {
    /// The transform that leaves every coordinate unchanged.
    pub const IDENTITY: Self = Self {
        matrix: [[1.0, 0.0], [0.0, 1.0]],
        translation_x: 0.0,
        translation_y: 0.0,
    };

    /// Check that this transform is finite and invertible.
    pub fn validate(self) -> Result<()> {
        let determinant = self.determinant();
        if self
            .matrix
            .into_iter()
            .flatten()
            .any(|value| !value.is_finite())
            || !self.translation_x.is_finite()
            || !self.translation_y.is_finite()
            || !determinant.is_finite()
            || determinant.abs() <= 1.0e-15
        {
            return Err(Error::Registration(
                "affine resampling transform must be finite and invertible".into(),
            ));
        }
        Ok(())
    }

    /// Map a source coordinate forward to reference space.
    pub fn apply(self, x: f64, y: f64) -> (f64, f64) {
        (
            self.matrix[0][0].mul_add(x, self.matrix[0][1] * y) + self.translation_x,
            self.matrix[1][0].mul_add(x, self.matrix[1][1] * y) + self.translation_y,
        )
    }

    /// Map a reference coordinate back to source space.
    pub fn inverse_apply(self, x: f64, y: f64) -> (f64, f64) {
        self.inverse_map()(x, y)
    }

    /// Compose this source-to-intermediate mapping with an
    /// intermediate-to-reference mapping. The returned transform applies
    /// `self` first and `next` second.
    pub fn then(self, next: Self) -> Self {
        Self {
            matrix: multiply_2x2(next.matrix, self.matrix),
            translation_x: next.matrix[0][0]
                .mul_add(self.translation_x, next.matrix[0][1] * self.translation_y)
                + next.translation_x,
            translation_y: next.matrix[1][0]
                .mul_add(self.translation_x, next.matrix[1][1] * self.translation_y)
                + next.translation_y,
        }
    }

    fn determinant(self) -> f64 {
        self.matrix[0][0].mul_add(self.matrix[1][1], -self.matrix[0][1] * self.matrix[1][0])
    }

    fn inverse_map(self) -> impl Fn(f64, f64) -> (f64, f64) {
        let determinant = self.determinant();
        let inverse = [
            [
                self.matrix[1][1] / determinant,
                -self.matrix[0][1] / determinant,
            ],
            [
                -self.matrix[1][0] / determinant,
                self.matrix[0][0] / determinant,
            ],
        ];
        move |x, y| {
            let x = x - self.translation_x;
            let y = y - self.translation_y;
            (
                inverse[0][0].mul_add(x, inverse[0][1] * y),
                inverse[1][0].mul_add(x, inverse[1][1] * y),
            )
        }
    }
}

impl From<SimilarityTransform> for AffineTransform {
    fn from(value: SimilarityTransform) -> Self {
        value.as_affine()
    }
}

fn multiply_2x2(left: [[f64; 2]; 2], right: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [
            left[0][0].mul_add(right[0][0], left[0][1] * right[1][0]),
            left[0][0].mul_add(right[0][1], left[0][1] * right[1][1]),
        ],
        [
            left[1][0].mul_add(right[0][0], left[1][1] * right[1][0]),
            left[1][0].mul_add(right[0][1], left[1][1] * right[1][1]),
        ],
    ]
}

/// Tuning for star detection, triangle matching, and drift limits.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct RegistrationOptions {
    /// Detection threshold above the background, in noise sigma.
    pub detection_sigma: f32,
    /// Cap on the number of detected stars retained per frame.
    pub maximum_stars: usize,
    /// Number of brightest stars used to build matching triangles.
    pub triangle_stars: usize,
    /// Allowed difference between triangle shape descriptors for a match.
    pub descriptor_tolerance: f64,
    /// Allowed departure of a candidate transform's scale from unity.
    pub scale_tolerance: f64,
    /// Radius, in pixels, within which a mapped star counts as a match.
    pub match_tolerance_pixels: f64,
    /// Absolute floor for the maximum frame-to-reference displacement.
    pub maximum_drift_pixels: f64,
    /// Fraction of the reference frame's larger dimension used for the maximum
    /// displacement. The effective bound is the larger of this and the pixel
    /// floor.
    pub maximum_drift_fraction: f64,
    /// Fewest matched stars a transform must reach to be accepted.
    pub minimum_matches: usize,
    /// Cap on candidate transforms scored before choosing the best.
    pub maximum_candidates: usize,
    /// The geometry fitted after the similarity match. The default,
    /// [`RegistrationModel::Similarity`], is not serialized, so existing
    /// options, fingerprints and contexts keep their exact bytes.
    #[serde(default, skip_serializing_if = "RegistrationModel::is_similarity")]
    pub model: RegistrationModel,
}

/// The geometry registration fits to each frame.
///
/// Every model first finds a similarity transform (shift, rotation, scale)
/// from the brightest stars; that transform drives the drift, scale and
/// rotation gates. [`Self::Affine`] and [`Self::Quadratic`] then pair up to
/// 2 000 stars through it and fit a polynomial from reference to source
/// coordinates, with outliers clipped, which follows lens distortion that a
/// similarity cannot. On a 173 mm wide field after a meridian flip, where the
/// distortion turns against the sky, the residual on ~2 800 stars fell from
/// 0.46-0.70px (similarity) to 0.37 (affine) and 0.29-0.31 (quadratic),
/// against a 0.24px centroid-noise floor. A frame with too few paired stars
/// for its model keeps the similarity.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RegistrationModel {
    /// Shift, rotation and uniform scale.
    #[default]
    Similarity,
    /// A first-order polynomial: adds shear and unequal scale.
    Affine,
    /// A second-order polynomial: adds the field curvature of lens
    /// distortion and differential refraction.
    Quadratic,
}

impl RegistrationModel {
    /// Whether this is the default, which options leave unserialized.
    pub fn is_similarity(&self) -> bool {
        *self == Self::Similarity
    }

    /// The polynomial order fitted after the similarity, if any.
    fn polynomial_order(self) -> Option<u8> {
        match self {
            Self::Similarity => None,
            Self::Affine => Some(1),
            Self::Quadratic => Some(2),
        }
    }
}

/// A polynomial map from reference pixel coordinates to source pixel
/// coordinates, the direction resampling reads. Coordinates are centred on
/// the reference frame and scaled by half its larger dimension before the
/// terms are formed, so the coefficients stay well conditioned.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PolynomialWarp {
    order: u8,
    center_x: f64,
    center_y: f64,
    scale: f64,
    x: Vec<f64>,
    y: Vec<f64>,
}

impl PolynomialWarp {
    /// The polynomial order: 1 for affine, 2 for quadratic.
    pub fn order(&self) -> u8 {
        self.order
    }

    /// The source position of a reference pixel.
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let terms = Self::terms(
            self.order,
            (x - self.center_x) / self.scale,
            (y - self.center_y) / self.scale,
        );
        let mut source_x = 0.0;
        let mut source_y = 0.0;
        for ((term, cx), cy) in terms.iter().zip(&self.x).zip(&self.y) {
            source_x += term * cx;
            source_y += term * cy;
        }
        (source_x, source_y)
    }

    /// Check the order, coefficient counts, and that every value is finite.
    pub fn validate(&self) -> Result<()> {
        let terms = Self::term_count(self.order);
        if !(1..=2).contains(&self.order)
            || self.x.len() != terms
            || self.y.len() != terms
            || !(self.scale.is_finite() && self.scale > 0.0)
            || !self.center_x.is_finite()
            || !self.center_y.is_finite()
            || self.x.iter().chain(&self.y).any(|value| !value.is_finite())
        {
            return Err(Error::Registration(
                "polynomial warp must be order 1 or 2 with finite coefficients".into(),
            ));
        }
        Ok(())
    }

    fn term_count(order: u8) -> usize {
        match order {
            1 => 3,
            _ => 6,
        }
    }

    fn terms(order: u8, u: f64, v: f64) -> [f64; 6] {
        if order == 1 {
            [1.0, u, v, 0.0, 0.0, 0.0]
        } else {
            [1.0, u, v, u * u, u * v, v * v]
        }
    }

    /// Least-squares fit of `source = P(reference)` over point pairs, by the
    /// normal equations with partial pivoting. `None` when they are singular.
    pub(crate) fn fit(order: u8, width: usize, height: usize, pairs: &[PointPair]) -> Option<Self> {
        let center_x = width as f64 * 0.5;
        let center_y = height as f64 * 0.5;
        let scale = width.max(height) as f64 * 0.5;
        let count = Self::term_count(order);
        let mut normal = [[0.0_f64; 6]; 6];
        let mut right_x = [0.0_f64; 6];
        let mut right_y = [0.0_f64; 6];
        for &((reference_x, reference_y), (source_x, source_y)) in pairs {
            let terms = Self::terms(
                order,
                (reference_x - center_x) / scale,
                (reference_y - center_y) / scale,
            );
            for row in 0..count {
                for column in 0..count {
                    normal[row][column] += terms[row] * terms[column];
                }
                right_x[row] += terms[row] * source_x;
                right_y[row] += terms[row] * source_y;
            }
        }
        let x = solve(normal, right_x, count)?;
        let y = solve(normal, right_y, count)?;
        Some(Self {
            order,
            center_x,
            center_y,
            scale,
            x,
            y,
        })
    }
}

/// Solve the leading `count` x `count` system by Gaussian elimination with
/// partial pivoting.
fn solve(mut matrix: [[f64; 6]; 6], mut right: [f64; 6], count: usize) -> Option<Vec<f64>> {
    for column in 0..count {
        let pivot = (column..count)
            .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))?;
        if matrix[pivot][column].abs() < 1.0e-12 {
            return None;
        }
        matrix.swap(column, pivot);
        right.swap(column, pivot);
        let pivot_row = matrix[column];
        for row in column + 1..count {
            let factor = matrix[row][column] / pivot_row[column];
            for (value, &pivot) in matrix[row][column..count]
                .iter_mut()
                .zip(&pivot_row[column..count])
            {
                *value -= factor * pivot;
            }
            right[row] -= factor * right[column];
        }
    }
    let mut solution = vec![0.0; count];
    for row in (0..count).rev() {
        let mut sum = right[row];
        for k in row + 1..count {
            sum -= matrix[row][k] * solution[k];
        }
        solution[row] = sum / matrix[row][row];
    }
    solution
        .iter()
        .all(|value| value.is_finite())
        .then_some(solution)
}

/// A reference position and the source position of the same star.
type PointPair = ((f64, f64), (f64, f64));

/// Stars detected for a polynomial fit, against 200 for matching.
const DISTORTION_STARS: usize = 2_000;
/// How far a star may sit from its partner under the similarity transform
/// to be paired for the polynomial fit.
const DISTORTION_PAIR_PIXELS: f64 = 2.0;
/// Paired stars needed per polynomial term before a frame takes the
/// polynomial rather than its similarity.
const DISTORTION_PAIRS_PER_TERM: usize = 5;

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
            model: RegistrationModel::Similarity,
        }
    }
}

impl RegistrationOptions {
    /// Stars to detect: the matching set, or more for a polynomial fit.
    fn dense_star_count(&self) -> usize {
        if self.model.is_similarity() {
            self.maximum_stars
        } else {
            self.maximum_stars.max(DISTORTION_STARS)
        }
    }

    /// Default pixel floor for the maximum frame-to-reference displacement.
    pub const DEFAULT_MAXIMUM_DRIFT_PIXELS: f64 = 256.0;
    /// Default fraction of the larger frame dimension used for maximum drift.
    pub const DEFAULT_MAXIMUM_DRIFT_FRACTION: f64 = 0.15;

    /// Maximum drift for a frame of the given size: the larger of the pixel
    /// floor and the fractional bound.
    pub fn effective_maximum_drift_pixels(&self, width: usize, height: usize) -> f64 {
        self.maximum_drift_pixels
            .max(width.max(height) as f64 * self.maximum_drift_fraction)
    }

    /// Check that every option lies in its valid range.
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

/// The transform found for one frame, with its match quality.
#[derive(Clone, Debug)]
pub struct RegistrationResult {
    /// Fitted source-to-reference transform.
    pub transform: SimilarityTransform,
    /// Number of star pairs supporting the fit.
    pub matched_stars: usize,
    /// Root-mean-square residual of the matched pairs, in pixels.
    pub rms_error_pixels: f64,
    /// Displacement of the frame center under the transform, in pixels.
    pub drift_pixels: f64,
    /// The polynomial from reference to source coordinates fitted after the
    /// similarity, when [`RegistrationOptions::model`] asks for one and enough
    /// stars paired up. Resampling uses it in place of `transform`, and
    /// `matched_stars` and `rms_error_pixels` then describe its fit.
    pub warp: Option<PolynomialWarp>,
}

/// A reference frame's stars and triangles, reused to register many frames
/// against it.
#[derive(Clone, Debug)]
pub struct Registrar {
    width: usize,
    height: usize,
    reference_stars: Vec<DetectedStar>,
    reference_index: StarSpatialIndex,
    reference_triangles: Vec<Triangle>,
    /// Up to [`DISTORTION_STARS`] reference stars, brightest first, for a
    /// polynomial fit; empty for a similarity model.
    dense_stars: Vec<DetectedStar>,
    dense_index: StarSpatialIndex,
    maximum_drift_pixels: f64,
    options: RegistrationOptions,
}

impl Registrar {
    /// Detect stars in the reference frame and prepare the matching index.
    pub fn new(reference: &LinearImage, options: RegistrationOptions) -> Result<Self> {
        options.validate()?;
        // The detector sorts by brightness before truncating, so the first
        // `maximum_stars` of a dense detection are exactly the matching set.
        let dense_count = options.dense_star_count();
        let mut reference_stars = detect(reference, &options, dense_count);
        let dense_stars = if options.model.is_similarity() {
            Vec::new()
        } else {
            let dense = reference_stars.clone();
            reference_stars.truncate(options.maximum_stars);
            dense
        };
        let dense_index = StarSpatialIndex::new(&dense_stars, DISTORTION_PAIR_PIXELS);
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
            dense_stars,
            dense_index,
            maximum_drift_pixels,
            options,
        })
    }

    /// Positions of the reference frame's brightest stars, on its grid.
    pub(crate) fn reference_star_positions(&self) -> Vec<(f64, f64)> {
        self.reference_stars
            .iter()
            .map(|star| (star.x, star.y))
            .collect()
    }

    /// Find the best transform aligning `source` to the reference frame.
    pub fn register(&self, source: &LinearImage) -> Result<RegistrationResult> {
        let dense_source = detect(source, &self.options, self.options.dense_star_count());
        let source_stars = &dense_source[..dense_source.len().min(self.options.maximum_stars)];
        if source_stars.len() < self.options.minimum_matches.max(3) {
            return Err(Error::Registration(format!(
                "source frame has only {} usable stars; need at least {}",
                source_stars.len(),
                self.options.minimum_matches.max(3)
            )));
        }
        let mut best: Option<ScoredTransform> = None;
        for candidate in translation_candidates(
            source_stars,
            &self.reference_stars,
            &self.options,
            self.maximum_drift_pixels,
        ) {
            retain_scored_transform(
                &mut best,
                candidate,
                source_stars,
                &self.reference_stars,
                &self.reference_index,
                &self.options,
            );
        }

        let source_triangles = triangles(source_stars, self.options.triangle_stars);
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
                    source_stars,
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
                source_stars,
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
        let mut result = self.refine_registration(best, source_stars)?;
        if let Some(order) = self.options.model.polynomial_order()
            && let Some((warp, pairs, rms)) = self.fit_warp(order, result.transform, &dense_source)
        {
            result.warp = Some(warp);
            result.matched_stars = pairs;
            result.rms_error_pixels = rms;
        }
        Ok(result)
    }

    /// Pair the dense source stars with the dense reference stars through the
    /// similarity, then fit a polynomial from reference to source with three
    /// rounds of clipping at three times the median residual. Returns the
    /// warp, the pairs it kept, and their RMS residual, or `None` when too
    /// few stars pair up for the order.
    fn fit_warp(
        &self,
        order: u8,
        transform: SimilarityTransform,
        source_stars: &[DetectedStar],
    ) -> Option<(PolynomialWarp, usize, f64)> {
        let minimum = PolynomialWarp::term_count(order) * DISTORTION_PAIRS_PER_TERM;
        let mut pairs = source_stars
            .iter()
            .filter_map(|star| {
                let (x, y) = transform.apply(star.x, star.y);
                let (index, _) = self.dense_index.nearest_within(
                    x,
                    y,
                    &self.dense_stars,
                    DISTORTION_PAIR_PIXELS,
                )?;
                let reference = &self.dense_stars[index];
                Some(((reference.x, reference.y), (star.x, star.y)))
            })
            .collect::<Vec<_>>();
        if pairs.len() < minimum {
            return None;
        }
        let mut warp = None;
        for _ in 0..3 {
            let fitted = PolynomialWarp::fit(order, self.width, self.height, &pairs)?;
            let residual =
                |&((reference_x, reference_y), (source_x, source_y)): &((f64, f64), (f64, f64))| {
                    let (x, y) = fitted.apply(reference_x, reference_y);
                    (x - source_x).hypot(y - source_y)
                };
            let mut residuals = pairs.iter().map(residual).collect::<Vec<_>>();
            residuals.sort_by(f64::total_cmp);
            let limit = 3.0 * residuals[residuals.len() / 2] + 0.05;
            pairs.retain(|pair| residual(pair) <= limit);
            warp = Some(fitted);
            if pairs.len() < minimum {
                return None;
            }
        }
        let warp = PolynomialWarp::fit(order, self.width, self.height, &pairs).or(warp)?;
        let squared = pairs
            .iter()
            .map(|&((reference_x, reference_y), (source_x, source_y))| {
                let (x, y) = warp.apply(reference_x, reference_y);
                (x - source_x).powi(2) + (y - source_y).powi(2)
            })
            .sum::<f64>();
        let rms = (squared / pairs.len() as f64).sqrt();
        Some((warp, pairs.len(), rms))
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
            warp: None,
        })
    }
}

/// Resample a source image through a fitted source-to-reference transform.
/// Samples outside the source grid, or whose interpolation neighborhood is
/// not finite, are written as `NaN`.
pub fn resample_to_reference(
    source: &LinearImage,
    width: usize,
    height: usize,
    transform: SimilarityTransform,
) -> Result<LinearImage> {
    resample_region_to_reference(
        source,
        width,
        height,
        ReferenceRegion {
            x: 0,
            y: 0,
            width,
            height,
        },
        transform,
    )
}

/// Resample one bounded region of the reference grid without allocating the
/// full registered image. Output pixel `(0, 0)` corresponds to
/// `(region.x, region.y)` in reference coordinates. Samples outside the source
/// grid, or whose interpolation neighborhood is not finite, are written as
/// `NaN`.
pub fn resample_region_to_reference(
    source: &LinearImage,
    reference_width: usize,
    reference_height: usize,
    region: ReferenceRegion,
    transform: SimilarityTransform,
) -> Result<LinearImage> {
    resample_region_to_reference_with(
        source,
        reference_width,
        reference_height,
        region,
        transform,
        Interpolation::Bilinear,
    )
}

/// [`resample_region_to_reference`] with a choice of interpolation.
pub fn resample_region_to_reference_with(
    source: &LinearImage,
    reference_width: usize,
    reference_height: usize,
    region: ReferenceRegion,
    transform: SimilarityTransform,
    interpolation: Interpolation,
) -> Result<LinearImage> {
    transform.validate()?;
    resample_region_with_inverse(
        source,
        reference_width,
        reference_height,
        region,
        transform.inverse_map(),
        match interpolation {
            Interpolation::Bilinear => Sampling::Bilinear,
            Interpolation::Lanczos3 => Sampling::Lanczos3,
        },
    )
}

/// How registration resamples a frame onto the reference grid.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Interpolation {
    /// Bilinear: the four nearest samples. Fast, and smooths noise by an
    /// amount that depends on each frame's sub-pixel shift.
    #[default]
    Bilinear,
    /// Lanczos-3 over the 6x6 nearest samples, dropping the negative lobes
    /// where they would ring at a high-contrast edge, as PixInsight's
    /// clamping does. Sharper: on a 98-frame M45 stack it took FWHM from
    /// 2.63 to 2.47px, at about twice the resampling cost. Falls
    /// back to bilinear within three pixels of the source edge and where
    /// a tap is not finite.
    Lanczos3,
}

impl Interpolation {
    /// Whether this is the default, which options leave unserialized.
    pub fn is_bilinear(&self) -> bool {
        *self == Self::Bilinear
    }
}

/// Map a debayered Bayer frame onto the reference grid without interpolating
/// color: each output pixel takes, in the one channel it records, the sample
/// of the source photosite nearest its position, and leaves the other two
/// channels `NaN`. Debayering keeps each photosite's own sample exactly, so
/// these are the raw sensor values. Integrating many frames that land on
/// different photosites fills every channel without interpolation; this is
/// Bayer drizzle with a one-pixel drop at the reference scale.
pub fn resample_region_photosites(
    source: &LinearImage,
    reference_width: usize,
    reference_height: usize,
    region: ReferenceRegion,
    transform: SimilarityTransform,
    layout: crate::BayerLayout,
) -> Result<LinearImage> {
    transform.validate()?;
    if source.channels != 3 {
        return Err(Error::Registration(
            "photosite sampling needs a debayered three-channel source".into(),
        ));
    }
    resample_region_with_inverse(
        source,
        reference_width,
        reference_height,
        region,
        transform.inverse_map(),
        Sampling::NearestPhotosite(layout),
    )
}

/// The geometry carrying a frame onto the reference grid: the fitted
/// similarity, or the polynomial warp that refines it.
#[derive(Clone, Copy, Debug)]
pub(crate) enum FrameGeometry<'a> {
    Similarity(SimilarityTransform),
    Warp(&'a PolynomialWarp),
}

impl<'a> FrameGeometry<'a> {
    pub(crate) fn of(transform: SimilarityTransform, warp: Option<&'a PolynomialWarp>) -> Self {
        match warp {
            Some(warp) => Self::Warp(warp),
            None => Self::Similarity(transform),
        }
    }
}

/// Resample one reference-grid region through a frame's geometry.
pub(crate) fn resample_region_geometry(
    source: &LinearImage,
    reference_width: usize,
    reference_height: usize,
    region: ReferenceRegion,
    geometry: FrameGeometry<'_>,
    sampling: Sampling,
) -> Result<LinearImage> {
    if let Sampling::NearestPhotosite(_) = sampling
        && source.channels != 3
    {
        return Err(Error::Registration(
            "photosite sampling needs a debayered three-channel source".into(),
        ));
    }
    match geometry {
        FrameGeometry::Similarity(transform) => {
            transform.validate()?;
            resample_region_with_inverse(
                source,
                reference_width,
                reference_height,
                region,
                transform.inverse_map(),
                sampling,
            )
        }
        FrameGeometry::Warp(warp) => {
            warp.validate()?;
            resample_region_with_inverse(
                source,
                reference_width,
                reference_height,
                region,
                |x, y| warp.apply(x, y),
                sampling,
            )
        }
    }
}

/// Resample one reference-grid region through a polynomial warp from
/// [`RegistrationResult::warp`].
pub fn resample_region_warped(
    source: &LinearImage,
    reference_width: usize,
    reference_height: usize,
    region: ReferenceRegion,
    warp: &PolynomialWarp,
    interpolation: Interpolation,
) -> Result<LinearImage> {
    resample_region_geometry(
        source,
        reference_width,
        reference_height,
        region,
        FrameGeometry::Warp(warp),
        Sampling::from(interpolation),
    )
}

impl From<Interpolation> for Sampling {
    fn from(interpolation: Interpolation) -> Self {
        match interpolation {
            Interpolation::Bilinear => Self::Bilinear,
            Interpolation::Lanczos3 => Self::Lanczos3,
        }
    }
}

/// How a resampled sample is formed from the source.
#[derive(Clone, Copy)]
pub(crate) enum Sampling {
    /// Bilinear interpolation of every channel.
    Bilinear,
    /// Clamped Lanczos-3 interpolation of every channel.
    Lanczos3,
    /// The nearest source photosite, in its own channel only.
    NearestPhotosite(crate::BayerLayout),
}

/// Resample a source image through a general source-to-reference affine
/// transform. This supports parity flips as well as rotation and scale.
pub fn resample_to_reference_affine(
    source: &LinearImage,
    width: usize,
    height: usize,
    transform: AffineTransform,
) -> Result<LinearImage> {
    resample_region_to_reference_affine(
        source,
        width,
        height,
        ReferenceRegion {
            x: 0,
            y: 0,
            width,
            height,
        },
        transform,
    )
}

/// Resample one bounded reference-grid region through a general affine
/// transform without allocating the full registered image.
pub fn resample_region_to_reference_affine(
    source: &LinearImage,
    reference_width: usize,
    reference_height: usize,
    region: ReferenceRegion,
    transform: AffineTransform,
) -> Result<LinearImage> {
    transform.validate()?;
    resample_region_with_inverse(
        source,
        reference_width,
        reference_height,
        region,
        transform.inverse_map(),
        Sampling::Bilinear,
    )
}

fn resample_region_with_inverse(
    source: &LinearImage,
    reference_width: usize,
    reference_height: usize,
    region: ReferenceRegion,
    inverse: impl Fn(f64, f64) -> (f64, f64) + Sync,
    sampling: Sampling,
) -> Result<LinearImage> {
    if reference_width == 0 || reference_height == 0 {
        return Err(Error::Registration(
            "resampling reference dimensions must be non-zero".into(),
        ));
    }
    if region.width == 0 || region.height == 0 {
        return Err(Error::Registration(
            "resampling region dimensions must be non-zero".into(),
        ));
    }
    let region_right = region
        .x
        .checked_add(region.width)
        .ok_or_else(|| Error::Registration("resampling region dimensions overflow".into()))?;
    let region_bottom = region
        .y
        .checked_add(region.height)
        .ok_or_else(|| Error::Registration("resampling region dimensions overflow".into()))?;
    if region_right > reference_width || region_bottom > reference_height {
        return Err(Error::Registration(
            "resampling region exceeds the reference grid".into(),
        ));
    }
    let channels = source.channels;
    let sample_count = region
        .width
        .checked_mul(region.height)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or_else(|| Error::Registration("resampling output dimensions overflow".into()))?;
    let row_samples = region
        .width
        .checked_mul(channels)
        .ok_or_else(|| Error::Registration("resampling row dimensions overflow".into()))?;
    let mut data = vec![f32::NAN; sample_count];
    data.par_chunks_mut(row_samples)
        .enumerate()
        .for_each(|(y, output_row)| {
            resample_row(
                source,
                region.x,
                region.y + y,
                &inverse,
                sampling,
                output_row,
            );
        });
    LinearImage::new(region.width, region.height, channels, data)
}

/// One output row of [`resample_region_with_inverse`]: reference row
/// `reference_y` from column `first_x` on.
///
/// An RGB pixel's Lanczos taps run four wide, which baseline SSE2 holds in
/// one register; an AVX2 build of this loop measured only 1-4% faster, so
/// it has none.
fn resample_row<F>(
    source: &LinearImage,
    first_x: usize,
    reference_y: usize,
    inverse: &F,
    sampling: Sampling,
    output_row: &mut [f32],
) where
    F: Fn(f64, f64) -> (f64, f64),
{
    const COORDINATE_EPSILON: f64 = 1.0e-9;
    let channels = source.channels;
    let maximum_source_x = (source.width - 1) as f64;
    let maximum_source_y = (source.height - 1) as f64;
    for (x, output) in output_row.chunks_exact_mut(channels).enumerate() {
        let reference_x = first_x + x;
        let (source_x, source_y) = inverse(reference_x as f64, reference_y as f64);
        if source_x < -COORDINATE_EPSILON
            || source_y < -COORDINATE_EPSILON
            || source_x > maximum_source_x + COORDINATE_EPSILON
            || source_y > maximum_source_y + COORDINATE_EPSILON
        {
            continue;
        }
        // Exact quarter- and half-turns accumulate tiny trigonometric
        // error at the boundary. Clamp only coordinates already proven
        // to lie within the epsilon-expanded source grid.
        let source_x = source_x.clamp(0.0, maximum_source_x);
        let source_y = source_y.clamp(0.0, maximum_source_y);
        if let Sampling::NearestPhotosite(layout) = sampling {
            let nearest_x = ((source_x + 0.5) as usize).min(source.width - 1);
            let nearest_y = ((source_y + 0.5) as usize).min(source.height - 1);
            let channel = layout.channel_at(nearest_x, nearest_y);
            output[channel] =
                source.data[(nearest_y * source.width + nearest_x) * channels + channel];
            continue;
        }
        // Both coordinates are non-negative here, so truncation is
        // the floor, without a libm call on baseline x86-64.
        let x0 = source_x as usize;
        let y0 = source_y as usize;
        let x1 = (x0 + 1).min(source.width - 1);
        let y1 = (y0 + 1).min(source.height - 1);
        let tx = (source_x - x0 as f64) as f32;
        let ty = (source_y - y0 as f64) as f32;
        let bilinear = |channel: usize| {
            let sample =
                |x: usize, y: usize| source.data[(y * source.width + x) * channels + channel];
            let values = [
                sample(x0, y0),
                sample(x1, y0),
                sample(x0, y1),
                sample(x1, y1),
            ];
            values.iter().all(|value| value.is_finite()).then(|| {
                let top = values[0] * (1.0 - tx) + values[1] * tx;
                let bottom = values[2] * (1.0 - tx) + values[3] * tx;
                top * (1.0 - ty) + bottom * ty
            })
        };
        let lanczos_window = matches!(sampling, Sampling::Lanczos3)
            && x0 >= 2
            && y0 >= 2
            && x0 + 3 < source.width
            && y0 + 3 < source.height;
        if lanczos_window {
            let (weights_x, weights_y) = (lanczos3_weights(tx), lanczos3_weights(ty));
            // An RGB pixel's taps run as the lanes of one four-wide vector,
            // the fourth lane reading the next pixel's red and going unused.
            // That needs one sample past the window's last pixel, which only
            // the image's last pixel lacks.
            let last_tap = ((y0 + 3) * source.width + x0 + 3) * channels;
            if channels == 3 && last_tap + 4 <= source.data.len() {
                let mut rows = [[0.0_f32; 4]; 6];
                for (row, value) in rows.iter_mut().enumerate() {
                    let start = ((y0 + row - 2) * source.width + x0 - 2) * channels;
                    let window = &source.data[start..start + 19];
                    let taps: [[f32; 4]; 6] = std::array::from_fn(|column| {
                        let tap = &window[column * 3..column * 3 + 4];
                        [tap[0], tap[1], tap[2], tap[3]]
                    });
                    *value = clamped_lanczos_lanes(&weights_x, &taps);
                }
                // A non-finite tap carries through to the result.
                let values = clamped_lanczos_lanes(&weights_y, &rows);
                for ((channel, output_sample), &value) in output.iter_mut().enumerate().zip(&values)
                {
                    if value.is_finite() {
                        *output_sample = value;
                    } else if let Some(value) = bilinear(channel) {
                        *output_sample = value;
                    }
                }
                continue;
            }
            for (channel, output_sample) in output.iter_mut().enumerate() {
                let mut rows = [0.0_f32; 6];
                for (row, value) in rows.iter_mut().enumerate() {
                    let start = ((y0 + row - 2) * source.width + x0 - 2) * channels;
                    let taps: [f32; 6] = std::array::from_fn(|column| {
                        source.data[start + column * channels + channel]
                    });
                    *value = clamped_lanczos(&weights_x, &taps);
                }
                // A non-finite tap carries through to the result.
                let value = clamped_lanczos(&weights_y, &rows);
                if value.is_finite() {
                    *output_sample = value;
                } else if let Some(value) = bilinear(channel) {
                    *output_sample = value;
                }
            }
            continue;
        }
        for (channel, output_sample) in output.iter_mut().enumerate() {
            if let Some(value) = bilinear(channel) {
                *output_sample = value;
            }
        }
    }
}

/// The share of the positive lobes' contribution the negative lobes may
/// reach before a 1-D Lanczos pass drops them, as PixInsight's clamping
/// threshold does.
const LANCZOS_CLAMPING: f32 = 0.3;

/// One 1-D Lanczos-3 pass. Where the negative lobes would contribute more
/// than [`LANCZOS_CLAMPING`] of what the positive lobes do, the taps straddle
/// a high-contrast edge such as a bright star beside the sky, and the
/// negative lobes would ring; then only the positive lobes are used,
/// renormalized. Elsewhere, including at a star's peak, the full kernel
/// keeps its sharpness.
fn clamped_lanczos(weights: &[f32; 6], taps: &[f32; 6]) -> f32 {
    // Between two samples the Lanczos-3 taps at offsets -2..=3 always carry
    // the signs + - + + - +, so the lobes need no per-tap test.
    let positive =
        weights[0] * taps[0] + weights[2] * taps[2] + weights[3] * taps[3] + weights[5] * taps[5];
    let negative = weights[1] * taps[1] + weights[4] * taps[4];
    if negative.abs() > LANCZOS_CLAMPING * positive.abs() {
        positive / (weights[0] + weights[2] + weights[3] + weights[5])
    } else {
        positive + negative
    }
}

/// [`clamped_lanczos`] on four lanes at once: lane `k` of the result is
/// `clamped_lanczos(weights, taps[..][k])`, from the same operations in the
/// same order, so each lane matches it bit for bit. Both branches are formed
/// and one is selected per lane, which the compiler turns into vector
/// arithmetic and a blend.
#[inline(always)]
fn clamped_lanczos_lanes(weights: &[f32; 6], taps: &[[f32; 4]; 6]) -> [f32; 4] {
    let denominator = weights[0] + weights[2] + weights[3] + weights[5];
    let mut result = [0.0_f32; 4];
    for (lane, result) in result.iter_mut().enumerate() {
        let positive = weights[0] * taps[0][lane]
            + weights[2] * taps[2][lane]
            + weights[3] * taps[3][lane]
            + weights[5] * taps[5][lane];
        let negative = weights[1] * taps[1][lane] + weights[4] * taps[4][lane];
        let clamped = positive / denominator;
        let full = positive + negative;
        *result = if negative.abs() > LANCZOS_CLAMPING * positive.abs() {
            clamped
        } else {
            full
        };
    }
    result
}

/// Steps per pixel in the Lanczos-3 weight table.
const LANCZOS_STEPS: usize = 1024;

/// Normalized Lanczos-3 weights for the taps at offsets -2..=3 from the
/// sample a fraction `t` (0..1) to the left, from a table of 1/1024-pixel
/// steps.
fn lanczos3_weights(t: f32) -> [f32; 6] {
    static TABLE: std::sync::OnceLock<Vec<[f32; 6]>> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        (0..=LANCZOS_STEPS)
            .map(|step| {
                let t = step as f64 / LANCZOS_STEPS as f64;
                let mut weights = [0.0_f64; 6];
                for (tap, weight) in weights.iter_mut().enumerate() {
                    let x = (tap as f64 - 2.0 - t).abs();
                    *weight = if x < 1.0e-12 {
                        1.0
                    } else if x < 3.0 {
                        let pi_x = std::f64::consts::PI * x;
                        3.0 * pi_x.sin() * (pi_x / 3.0).sin() / (pi_x * pi_x)
                    } else {
                        0.0
                    };
                }
                let total: f64 = weights.iter().sum();
                weights.map(|weight| (weight / total) as f32)
            })
            .collect()
    });
    table[((t * LANCZOS_STEPS as f32 + 0.5) as usize).min(LANCZOS_STEPS)]
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

fn detect(
    image: &LinearImage,
    options: &RegistrationOptions,
    max_stars: usize,
) -> Vec<DetectedStar> {
    let mut luma = image.luminance();
    normalize_for_detection(&mut luma);
    let config = DetectConfig {
        backend: DetectBackend::F32,
        sigma: options.detection_sigma,
        max_stars,
        ..DetectConfig::default()
    };
    seiza::detect_stars_luma_f32(&luma, image.width as u32, image.height as u32, &config)
}

fn normalize_for_detection(values: &mut [f32]) {
    let (minimum, maximum) = values
        .par_iter()
        .filter(|value| value.is_finite())
        .map(|value| (*value as f64, *value as f64))
        .reduce(
            || (f64::INFINITY, f64::NEG_INFINITY),
            |left, right| (left.0.min(right.0), left.1.max(right.1)),
        );
    if !minimum.is_finite() || maximum <= minimum {
        values.par_iter_mut().for_each(|value| *value = 0.0);
        return;
    }
    let range = maximum - minimum;
    values.par_iter_mut().for_each(|value| {
        *value = if value.is_finite() {
            ((*value as f64 - minimum) / range) as f32
        } else {
            0.0
        };
    });
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

/// A star field for tests: 220 stars on a 512x384 grid, rendered where a
/// fixed quadratic distortion (up to about 3px at the corners) and a small
/// shift carry each reference position when `distorted` is set.
#[cfg(test)]
pub(crate) fn test_star_field(distorted: bool) -> LinearImage {
    let (width, height) = (512_usize, 384_usize);
    let mut state = 0x2545_f491_u32;
    let mut random = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        f64::from(state) / f64::from(u32::MAX)
    };
    let stars = (0..220)
        .map(|_| {
            (
                12.0 + random() * (width as f64 - 24.0),
                12.0 + random() * (height as f64 - 24.0),
                400.0 + random() * 4000.0,
            )
        })
        .collect::<Vec<_>>();
    let mut data = vec![100.0_f32; width * height];
    for &(x, y, flux) in &stars {
        let (x, y) = if distorted {
            test_distortion(x, y)
        } else {
            (x, y)
        };
        let (left, top) = ((x - 6.0).max(0.0) as usize, (y - 6.0).max(0.0) as usize);
        for py in top..((y + 7.0) as usize).min(height) {
            for px in left..((x + 7.0) as usize).min(width) {
                let (dx, dy) = (px as f64 - x, py as f64 - y);
                data[py * width + px] += (flux * (-(dx * dx + dy * dy) / 2.9).exp()) as f32;
            }
        }
    }
    LinearImage::new(width, height, 1, data).unwrap()
}

/// Where [`test_star_field`] puts the star whose reference position is
/// `(x, y)`.
#[cfg(test)]
pub(crate) fn test_distortion(x: f64, y: f64) -> (f64, f64) {
    let (u, v) = ((x - 256.0) / 256.0, (y - 192.0) / 256.0);
    (
        x + 2.0 * u * u - 1.0 * u * v + 0.6,
        y + 1.5 * v * v + 0.8 * u * v - 0.4,
    )
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

    fn lanczos(source: &LinearImage, transform: SimilarityTransform) -> LinearImage {
        resample_region_to_reference_with(
            source,
            source.width,
            source.height,
            ReferenceRegion {
                x: 0,
                y: 0,
                width: source.width,
                height: source.height,
            },
            transform,
            Interpolation::Lanczos3,
        )
        .unwrap()
    }

    fn shift(dx: f64, dy: f64) -> SimilarityTransform {
        SimilarityTransform {
            translation_x: dx,
            translation_y: dy,
            ..SimilarityTransform::IDENTITY
        }
    }

    #[test]
    fn lanczos_identity_and_whole_pixel_shifts_return_the_samples() {
        let (width, height) = (24, 20);
        let data = (0..width * height)
            .map(|index| ((index * 37) % 101) as f32 * 3.5 + 10.0)
            .collect::<Vec<_>>();
        let source = LinearImage::new(width, height, 1, data).unwrap();
        let same = lanczos(&source, SimilarityTransform::IDENTITY);
        let moved = lanczos(&source, shift(2.0, 1.0));
        for y in 0..height {
            for x in 0..width {
                let expected = source.data[y * width + x];
                assert!((same.data[y * width + x] - expected).abs() < 1.0e-3);
                if x >= 2 && y >= 1 {
                    let value = moved.data[y * width + x];
                    let from = source.data[(y - 1) * width + x - 2];
                    assert!((value - from).abs() < 1.0e-3, "({x}, {y})");
                }
            }
        }
    }

    #[test]
    fn lanczos_keeps_a_star_peak_better_than_bilinear_without_ringing() {
        let (width, height) = (32, 32);
        let star = |x: f64, y: f64| {
            let (dx, dy) = (x - 15.5, y - 16.0);
            100.0 + 5000.0 * (-(dx * dx + dy * dy) / (2.0 * 1.1 * 1.1)).exp()
        };
        let source = LinearImage::new(
            width,
            height,
            1,
            (0..width * height)
                .map(|index| star((index % width) as f64, (index / width) as f64) as f32)
                .collect(),
        )
        .unwrap();
        // Moving the frame half a pixel right puts the star's true centre on
        // pixel 16 of the output, where it peaks at 5100.
        let bilinear = resample_to_reference(&source, width, height, shift(0.5, 0.0)).unwrap();
        let sharp = lanczos(&source, shift(0.5, 0.0));
        let peak = 16 * width + 16;
        assert!(
            sharp.data[peak] > bilinear.data[peak] + 50.0,
            "lanczos {} against bilinear {}",
            sharp.data[peak],
            bilinear.data[peak]
        );
        assert!(sharp.data[peak] <= 5100.0 + 1.0e-3);
        // Clamping: no sample falls below the sky the star sits on.
        let finite = sharp.data.iter().filter(|value| value.is_finite());
        assert!(finite.clone().all(|&value| value >= 100.0 - 1.0e-3));
    }

    #[test]
    fn polynomial_warp_fit_recovers_a_quadratic_exactly() {
        let pairs = (0..10)
            .flat_map(|row| (0..12).map(move |column| (column as f64 * 40.0, row as f64 * 35.0)))
            .map(|(x, y)| ((x, y), test_distortion(x, y)))
            .collect::<Vec<_>>();
        let warp = PolynomialWarp::fit(2, 512, 384, &pairs).unwrap();
        warp.validate().unwrap();
        for (x, y) in [(0.0, 0.0), (511.0, 383.0), (100.5, 250.25)] {
            let (fitted_x, fitted_y) = warp.apply(x, y);
            let (true_x, true_y) = test_distortion(x, y);
            assert!((fitted_x - true_x).abs() < 1.0e-9 && (fitted_y - true_y).abs() < 1.0e-9);
        }
        // An affine fit of the same pairs cannot follow the curvature.
        let affine = PolynomialWarp::fit(1, 512, 384, &pairs).unwrap();
        let worst = pairs
            .iter()
            .map(|&((x, y), (true_x, true_y))| {
                let (fitted_x, fitted_y) = affine.apply(x, y);
                (fitted_x - true_x).hypot(fitted_y - true_y)
            })
            .fold(0.0, f64::max);
        assert!(worst > 0.3, "{worst}");
    }

    #[test]
    fn quadratic_registration_follows_a_distorted_field() {
        let reference = test_star_field(false);
        let source = test_star_field(true);
        let similar = Registrar::new(&reference, RegistrationOptions::default())
            .unwrap()
            .register(&source)
            .unwrap();
        assert!(similar.warp.is_none());
        let options = RegistrationOptions {
            model: RegistrationModel::Quadratic,
            ..RegistrationOptions::default()
        };
        let quadratic = Registrar::new(&reference, options)
            .unwrap()
            .register(&source)
            .unwrap();
        let warp = quadratic
            .warp
            .as_ref()
            .expect("enough stars pair for a quadratic");
        assert_eq!(warp.order(), 2);
        assert!(
            quadratic.rms_error_pixels < 0.1 && similar.rms_error_pixels > 0.3,
            "quadratic {} against similarity {}",
            quadratic.rms_error_pixels,
            similar.rms_error_pixels
        );
        for (x, y) in [(40.0, 40.0), (470.0, 340.0), (256.0, 192.0)] {
            let (fitted_x, fitted_y) = warp.apply(x, y);
            let (true_x, true_y) = test_distortion(x, y);
            assert!(
                (fitted_x - true_x).hypot(fitted_y - true_y) < 0.1,
                "({x}, {y}): fitted ({fitted_x}, {fitted_y}), true ({true_x}, {true_y})"
            );
        }
        // Too few stars for a quadratic keeps the similarity.
        let sparse = Registrar::new(
            &reference,
            RegistrationOptions {
                model: RegistrationModel::Quadratic,
                ..RegistrationOptions::default()
            },
        )
        .unwrap();
        assert!(sparse.fit_warp(2, quadratic.transform, &[]).is_none());
    }

    #[test]
    fn bilinear_resampling_uses_inverse_transform() {
        let source =
            LinearImage::new(4, 4, 1, (0..16).map(|value| value as f32).collect()).unwrap();
        let registered = resample_to_reference(
            &source,
            4,
            4,
            SimilarityTransform {
                translation_x: 1.0,
                ..SimilarityTransform::IDENTITY
            },
        )
        .unwrap();
        assert_eq!(registered.data[1], source.data[0]);
        assert!(registered.data[0].is_nan());
    }

    #[test]
    fn similarity_transform_composition_matches_sequential_application() {
        let first = SimilarityTransform {
            scale: 0.98,
            rotation_radians: 0.13,
            translation_x: 12.0,
            translation_y: -4.0,
        };
        let second = SimilarityTransform {
            scale: 1.02,
            rotation_radians: -0.07,
            translation_x: -3.0,
            translation_y: 9.0,
        };
        let intermediate = first.apply(42.0, 17.0);
        let sequential = second.apply(intermediate.0, intermediate.1);
        let composed = first.then(second).apply(42.0, 17.0);

        assert!((sequential.0 - composed.0).abs() < 1.0e-10);
        assert!((sequential.1 - composed.1).abs() < 1.0e-10);
    }

    #[test]
    fn affine_transform_composition_preserves_a_parity_flip() {
        let first = SimilarityTransform {
            scale: 0.98,
            rotation_radians: 0.13,
            translation_x: 12.0,
            translation_y: -4.0,
        }
        .as_affine();
        let second = AffineTransform {
            matrix: [[-1.0, 0.2], [0.0, 1.0]],
            translation_x: 40.0,
            translation_y: 3.0,
        };
        let intermediate = first.apply(42.0, 17.0);
        let sequential = second.apply(intermediate.0, intermediate.1);
        let composed = first.then(second);
        let actual = composed.apply(42.0, 17.0);

        assert!(
            composed.matrix[0][0] * composed.matrix[1][1]
                - composed.matrix[0][1] * composed.matrix[1][0]
                < 0.0
        );
        assert!((sequential.0 - actual.0).abs() < 1.0e-10);
        assert!((sequential.1 - actual.1).abs() < 1.0e-10);
        let restored = composed.inverse_apply(actual.0, actual.1);
        assert!((restored.0 - 42.0).abs() < 1.0e-10);
        assert!((restored.1 - 17.0).abs() < 1.0e-10);
    }

    #[test]
    fn affine_resampling_can_flip_an_image_horizontally() {
        let source = LinearImage::new(4, 2, 1, (0..8).map(|value| value as f32).collect()).unwrap();
        let flipped = resample_to_reference_affine(
            &source,
            4,
            2,
            AffineTransform {
                matrix: [[-1.0, 0.0], [0.0, 1.0]],
                translation_x: 3.0,
                translation_y: 0.0,
            },
        )
        .unwrap();

        assert_eq!(flipped.data, vec![3.0, 2.0, 1.0, 0.0, 7.0, 6.0, 5.0, 4.0]);
    }

    #[test]
    fn resampling_undoes_a_meridian_flip_about_the_image_center() {
        let source =
            LinearImage::new(4, 3, 1, (0..12).rev().map(|value| value as f32).collect()).unwrap();
        let registered = resample_to_reference(
            &source,
            4,
            3,
            SimilarityTransform {
                scale: 1.0,
                rotation_radians: std::f64::consts::PI,
                translation_x: 3.0,
                translation_y: 2.0,
            },
        )
        .unwrap();
        let expected = (0..12).map(|value| value as f32).collect::<Vec<_>>();
        assert_eq!(registered.data, expected);
    }

    #[test]
    fn identity_resampling_preserves_the_final_row_and_column() {
        let source =
            LinearImage::new(4, 4, 1, (0..16).map(|value| value as f32).collect()).unwrap();
        let registered =
            resample_to_reference(&source, 4, 4, SimilarityTransform::IDENTITY).unwrap();
        assert_eq!(registered.data, source.data);
    }

    #[test]
    fn resampling_places_a_cropped_source_on_the_reference_grid() {
        let source = LinearImage::new(3, 2, 1, (0..6).map(|value| value as f32).collect()).unwrap();
        let registered = resample_to_reference(
            &source,
            5,
            4,
            SimilarityTransform {
                translation_x: 1.0,
                translation_y: 1.0,
                ..SimilarityTransform::IDENTITY
            },
        )
        .unwrap();

        for y in 0..source.height {
            for x in 0..source.width {
                assert_eq!(
                    registered.data[(y + 1) * registered.width + x + 1],
                    source.data[y * source.width + x]
                );
            }
        }
        assert_eq!(
            registered
                .data
                .iter()
                .filter(|sample| sample.is_finite())
                .count(),
            source.sample_count()
        );
    }

    #[test]
    fn region_resampling_matches_the_same_full_reference_crop() {
        let source =
            LinearImage::new(6, 5, 3, (0..90).map(|value| value as f32).collect()).unwrap();
        let transform = SimilarityTransform {
            scale: 1.0,
            rotation_radians: 0.0,
            translation_x: 1.0,
            translation_y: -1.0,
        };
        let full = resample_to_reference(&source, 8, 7, transform).unwrap();
        let region = ReferenceRegion {
            x: 2,
            y: 1,
            width: 4,
            height: 3,
        };
        let cropped = resample_region_to_reference(&source, 8, 7, region, transform).unwrap();

        for y in 0..region.height {
            for x in 0..region.width {
                for channel in 0..source.channels {
                    let full_index =
                        ((region.y + y) * full.width + region.x + x) * source.channels + channel;
                    let crop_index = (y * region.width + x) * source.channels + channel;
                    assert_eq!(cropped.data[crop_index], full.data[full_index]);
                }
            }
        }
    }

    #[test]
    fn region_resampling_uses_absolute_reference_coordinates() {
        let source =
            LinearImage::new(5, 4, 1, (0..20).map(|value| value as f32).collect()).unwrap();
        let region = ReferenceRegion {
            x: 2,
            y: 1,
            width: 2,
            height: 2,
        };
        let cropped = resample_region_to_reference(
            &source,
            source.width,
            source.height,
            region,
            SimilarityTransform::IDENTITY,
        )
        .unwrap();

        assert_eq!(cropped.data, vec![7.0, 8.0, 12.0, 13.0]);
    }

    #[test]
    fn region_resampling_rejects_empty_or_out_of_bounds_regions() {
        let source = LinearImage::new(4, 4, 1, vec![0.0; 16]).unwrap();
        let empty = ReferenceRegion {
            x: 0,
            y: 0,
            width: 0,
            height: 1,
        };
        assert!(
            resample_region_to_reference(&source, 4, 4, empty, SimilarityTransform::IDENTITY)
                .is_err()
        );

        let outside = ReferenceRegion {
            x: 3,
            y: 3,
            width: 2,
            height: 2,
        };
        assert!(
            resample_region_to_reference(&source, 4, 4, outside, SimilarityTransform::IDENTITY)
                .is_err()
        );
    }

    #[test]
    fn resampling_rejects_invalid_output_and_transform() {
        let source = LinearImage::new(1, 1, 1, vec![1.0]).unwrap();
        assert!(resample_to_reference(&source, 0, 1, SimilarityTransform::IDENTITY).is_err());
        assert!(resample_to_reference(&source, 1, 0, SimilarityTransform::IDENTITY).is_err());
        assert!(
            resample_to_reference(
                &source,
                1,
                1,
                SimilarityTransform {
                    scale: 0.0,
                    ..SimilarityTransform::IDENTITY
                }
            )
            .is_err()
        );
        assert!(
            resample_to_reference(
                &source,
                1,
                1,
                SimilarityTransform {
                    translation_x: f64::NAN,
                    ..SimilarityTransform::IDENTITY
                }
            )
            .is_err()
        );
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

    /// The resampling row loop as it was before an RGB pixel's channels ran
    /// as vector lanes: each channel filtered on its own.
    fn resample_per_channel(
        source: &LinearImage,
        region: ReferenceRegion,
        inverse: impl Fn(f64, f64) -> (f64, f64),
        sampling: Sampling,
    ) -> Vec<f32> {
        const COORDINATE_EPSILON: f64 = 1.0e-9;
        let channels = source.channels;
        let mut data = vec![f32::NAN; region.width * region.height * channels];
        let maximum_source_x = (source.width - 1) as f64;
        let maximum_source_y = (source.height - 1) as f64;
        for (y, output_row) in data.chunks_mut(region.width * channels).enumerate() {
            for (x, output) in output_row.chunks_exact_mut(channels).enumerate() {
                let (source_x, source_y) = inverse((region.x + x) as f64, (region.y + y) as f64);
                if source_x < -COORDINATE_EPSILON
                    || source_y < -COORDINATE_EPSILON
                    || source_x > maximum_source_x + COORDINATE_EPSILON
                    || source_y > maximum_source_y + COORDINATE_EPSILON
                {
                    continue;
                }
                let source_x = source_x.clamp(0.0, maximum_source_x);
                let source_y = source_y.clamp(0.0, maximum_source_y);
                if let Sampling::NearestPhotosite(layout) = sampling {
                    let nearest_x = ((source_x + 0.5) as usize).min(source.width - 1);
                    let nearest_y = ((source_y + 0.5) as usize).min(source.height - 1);
                    let channel = layout.channel_at(nearest_x, nearest_y);
                    output[channel] =
                        source.data[(nearest_y * source.width + nearest_x) * channels + channel];
                    continue;
                }
                let x0 = source_x as usize;
                let y0 = source_y as usize;
                let x1 = (x0 + 1).min(source.width - 1);
                let y1 = (y0 + 1).min(source.height - 1);
                let tx = (source_x - x0 as f64) as f32;
                let ty = (source_y - y0 as f64) as f32;
                let bilinear = |channel: usize| {
                    let sample = |x: usize, y: usize| {
                        source.data[(y * source.width + x) * channels + channel]
                    };
                    let values = [
                        sample(x0, y0),
                        sample(x1, y0),
                        sample(x0, y1),
                        sample(x1, y1),
                    ];
                    values.iter().all(|value| value.is_finite()).then(|| {
                        let top = values[0] * (1.0 - tx) + values[1] * tx;
                        let bottom = values[2] * (1.0 - tx) + values[3] * tx;
                        top * (1.0 - ty) + bottom * ty
                    })
                };
                let lanczos_window = matches!(sampling, Sampling::Lanczos3)
                    && x0 >= 2
                    && y0 >= 2
                    && x0 + 3 < source.width
                    && y0 + 3 < source.height;
                if lanczos_window {
                    let (weights_x, weights_y) = (lanczos3_weights(tx), lanczos3_weights(ty));
                    for (channel, output_sample) in output.iter_mut().enumerate() {
                        let mut rows = [0.0_f32; 6];
                        for (row, value) in rows.iter_mut().enumerate() {
                            let start = ((y0 + row - 2) * source.width + x0 - 2) * channels;
                            let taps: [f32; 6] = std::array::from_fn(|column| {
                                source.data[start + column * channels + channel]
                            });
                            *value = clamped_lanczos(&weights_x, &taps);
                        }
                        let value = clamped_lanczos(&weights_y, &rows);
                        if value.is_finite() {
                            *output_sample = value;
                        } else if let Some(value) = bilinear(channel) {
                            *output_sample = value;
                        }
                    }
                    continue;
                }
                for (channel, output_sample) in output.iter_mut().enumerate() {
                    if let Some(value) = bilinear(channel) {
                        *output_sample = value;
                    }
                }
            }
        }
        data
    }

    #[test]
    fn resampling_matches_per_channel_filtering_bit_for_bit() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let warp = PolynomialWarp {
            order: 2,
            center_x: 15.0,
            center_y: 9.0,
            scale: 16.0,
            x: vec![15.3, 16.1, 0.4, 0.3, -0.2, 0.1],
            y: vec![8.6, -0.3, 15.8, 0.1, 0.25, -0.15],
        };
        let similarities = [
            SimilarityTransform::IDENTITY,
            shift(2.0, 1.0),
            shift(-0.37, 0.61),
            SimilarityTransform {
                scale: 1.03,
                rotation_radians: 0.31,
                translation_x: 1.3,
                translation_y: -0.7,
            },
            SimilarityTransform {
                scale: 1.0,
                rotation_radians: std::f64::consts::PI,
                translation_x: 30.0,
                translation_y: 16.0,
            },
        ];
        let layout = crate::BayerLayout {
            pattern: seiza_fits::BayerPattern::Grbg,
            x_offset: 1,
            y_offset: 0,
        };
        // Sizes too small for any Lanczos window, and ones whose last pixel
        // has a window ending on the image's last sample; noise with bright
        // spikes that trip the clamping, and non-finite samples.
        for (width, height) in [(1, 1), (3, 3), (6, 6), (7, 5), (9, 8), (31, 17)] {
            for channels in [1, 3] {
                let data = (0..width * height * channels)
                    .map(|_| match next() % 41 {
                        0 => f32::NAN,
                        1 => f32::INFINITY,
                        2 => -0.0,
                        3..=6 => 50.0 + (next() % 1000) as f32,
                        _ => (next() >> 40) as f32 / (1 << 24) as f32,
                    })
                    .collect::<Vec<_>>();
                let source = LinearImage::new(width, height, channels, data).unwrap();
                let regions = [
                    ReferenceRegion {
                        x: 0,
                        y: 0,
                        width,
                        height,
                    },
                    ReferenceRegion {
                        x: width / 3,
                        y: height / 2,
                        width: width - width / 3,
                        height: height - height / 2,
                    },
                ];
                let mut samplings = vec![Sampling::Lanczos3, Sampling::Bilinear];
                if channels == 3 {
                    samplings.push(Sampling::NearestPhotosite(layout));
                }
                for region in regions {
                    for &sampling in &samplings {
                        let check = |label: &str, got: LinearImage, expected: Vec<f32>| {
                            for (index, (a, b)) in got.data.iter().zip(&expected).enumerate() {
                                assert_eq!(
                                    a.to_bits(),
                                    b.to_bits(),
                                    "{label} {width}x{height}x{channels} {region:?} sample {index}"
                                );
                            }
                        };
                        for transform in similarities {
                            let got = resample_region_with_inverse(
                                &source,
                                width,
                                height,
                                region,
                                transform.inverse_map(),
                                sampling,
                            )
                            .unwrap();
                            let expected = resample_per_channel(
                                &source,
                                region,
                                transform.inverse_map(),
                                sampling,
                            );
                            check(&format!("{transform:?}"), got, expected);
                        }
                        let got = resample_region_with_inverse(
                            &source,
                            width,
                            height,
                            region,
                            |x, y| warp.apply(x, y),
                            sampling,
                        )
                        .unwrap();
                        let expected = resample_per_channel(
                            &source,
                            region,
                            |x, y| warp.apply(x, y),
                            sampling,
                        );
                        check("warp", got, expected);
                    }
                }
            }
        }
    }
}
