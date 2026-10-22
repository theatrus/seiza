use crate::binning::Scale;
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

    /// This transform between source and reference grids each `scale` times
    /// as fine, or `1 / b` for both binned by `b`: `C ∘ T ∘ C⁻¹` for the map
    /// `C(x) = scale · x + c`, `c = (scale − 1) / 2`, that carries a pixel
    /// centre to the finer grid. Scale and rotation stand, and the
    /// translation becomes `scale · t + (I − sR)·(c, c)`.
    pub(crate) fn rescaled(self, scale: f64) -> Self {
        let offset = (scale - 1.0) / 2.0;
        let cosine = self.rotation_radians.cos() * self.scale;
        let sine = self.rotation_radians.sin() * self.scale;
        Self {
            translation_x: scale * self.translation_x + offset - (cosine - sine) * offset,
            translation_y: scale * self.translation_y + offset - (sine + cosine) * offset,
            ..self
        }
    }
}

impl AffineTransform {
    /// The transform that leaves every coordinate unchanged.
    pub const IDENTITY: Self = Self {
        matrix: [[1.0, 0.0], [0.0, 1.0]],
        translation_x: 0.0,
        translation_y: 0.0,
    };

    /// The mapping onto a grid `scale` times as fine (drizzle at `scale`)
    /// or as coarse (binning by `1 / scale`) that covers the same area.
    /// Pixels count from 0 with centres on whole numbers, so a pixel's
    /// centre `x` lands at `scale · x + (scale − 1) / 2`: with a 2× drizzle,
    /// source pixel 0, spanning −0.5 to 0.5, covers output pixels 0 and 1,
    /// and its centre is at 0.5.
    pub fn scale_about_pixel_centres(scale: f64) -> Self {
        let offset = (scale - 1.0) / 2.0;
        Self {
            matrix: [[scale, 0.0], [0.0, scale]],
            translation_x: offset,
            translation_y: offset,
        }
    }

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
///
/// A lens-corrected warp must be serialized with a human-readable format
/// such as JSON, or through a saved stack context. Standalone compact binary
/// serialization is refused because its legacy layout cannot hold a lens.
#[derive(Clone, Debug, PartialEq)]
pub struct PolynomialWarp {
    order: u8,
    center_x: f64,
    center_y: f64,
    scale: f64,
    x: Vec<f64>,
    y: Vec<f64>,
    /// The lens distortion between the polynomial's undistorted source
    /// positions and the source's pixels, when an expanded canvas measured
    /// one.
    distortion: Option<crate::lens::LensDistortion>,
}

/// A warp as self-describing encodings carry it. A warp without distortion
/// reads and writes as it did before distortion existed; readers that
/// predate it refuse one with.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PolynomialWarpWire {
    order: u8,
    center_x: f64,
    center_y: f64,
    scale: f64,
    x: Vec<f64>,
    y: Vec<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    distortion: Option<crate::lens::LensDistortion>,
}

/// A warp as compact binary encodings carry it, which cannot tell an absent
/// field from the next value, so it holds no distortion: a context records
/// its canvas's lens once, and restores it to every warp.
#[derive(Deserialize, Serialize)]
struct PolynomialWarpCompact {
    order: u8,
    center_x: f64,
    center_y: f64,
    scale: f64,
    x: Vec<f64>,
    y: Vec<f64>,
}

impl Serialize for PolynomialWarp {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let Self {
            order,
            center_x,
            center_y,
            scale,
            x,
            y,
            distortion,
        } = self.clone();
        if serializer.is_human_readable() {
            PolynomialWarpWire {
                order,
                center_x,
                center_y,
                scale,
                x,
                y,
                distortion,
            }
            .serialize(serializer)
        } else {
            if distortion.is_some() {
                return Err(serde::ser::Error::custom(
                    "binary warp serialization cannot preserve lens distortion; use JSON or a stack context",
                ));
            }
            PolynomialWarpCompact {
                order,
                center_x,
                center_y,
                scale,
                x,
                y,
            }
            .serialize(serializer)
        }
    }
}

impl<'de> Deserialize<'de> for PolynomialWarp {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            let wire = PolynomialWarpWire::deserialize(deserializer)?;
            Ok(Self {
                order: wire.order,
                center_x: wire.center_x,
                center_y: wire.center_y,
                scale: wire.scale,
                x: wire.x,
                y: wire.y,
                distortion: wire.distortion,
            })
        } else {
            let compact = PolynomialWarpCompact::deserialize(deserializer)?;
            Ok(Self {
                order: compact.order,
                center_x: compact.center_x,
                center_y: compact.center_y,
                scale: compact.scale,
                x: compact.x,
                y: compact.y,
                distortion: None,
            })
        }
    }
}

impl PolynomialWarp {
    /// The polynomial order: 1 for affine, 2 for quadratic, 3 for cubic.
    pub fn order(&self) -> u8 {
        self.order
    }

    /// The same map on a reference grid whose origin lies `dx`, `dy` before
    /// this one's, as when the reference is a region of a larger image.
    pub(crate) fn translated(&self, dx: f64, dy: f64) -> Self {
        Self {
            center_x: self.center_x + dx,
            center_y: self.center_y + dy,
            ..self.clone()
        }
    }

    /// Apply `transform` to this map's source coordinates exactly. A lens
    /// must be attached only after composition in the undistorted plane.
    pub(crate) fn source_transformed(&self, transform: SimilarityTransform) -> Result<Self> {
        self.validate()?;
        transform.validate()?;
        if self.distortion.is_some() {
            return Err(Error::Registration(
                "cannot compose a panel projection with an attached lens".into(),
            ));
        }
        let cosine = transform.rotation_radians.cos() * transform.scale;
        let sine = transform.rotation_radians.sin() * transform.scale;
        let mut composed = self.clone();
        for ((x, y), (&a, &b)) in composed
            .x
            .iter_mut()
            .zip(&mut composed.y)
            .zip(self.x.iter().zip(&self.y))
        {
            *x = cosine * a - sine * b;
            *y = sine * a + cosine * b;
        }
        composed.x[0] += transform.translation_x;
        composed.y[0] += transform.translation_y;
        composed.validate()?;
        Ok(composed)
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
        match &self.distortion {
            Some(distortion) => distortion.distort(source_x, source_y),
            None => (source_x, source_y),
        }
    }

    /// This warp between grids each `scale` times as fine, as
    /// [`SimilarityTransform::rescaled`] carries a similarity: its centre
    /// moves to the finer grid and its scale stretches with it, its constant
    /// terms carry through the same map and the others stretch by `scale`,
    /// and its lens moves with it. Exact for every order.
    pub(crate) fn rescaled(&self, scale: f64) -> Self {
        let offset = (scale - 1.0) / 2.0;
        let carry = |coefficients: &[f64]| {
            coefficients
                .iter()
                .enumerate()
                .map(|(term, &value)| {
                    if term == 0 {
                        scale * value + offset
                    } else {
                        scale * value
                    }
                })
                .collect::<Vec<_>>()
        };
        Self {
            order: self.order,
            center_x: scale * self.center_x + offset,
            center_y: scale * self.center_y + offset,
            scale: scale * self.scale,
            x: carry(&self.x),
            y: carry(&self.y),
            distortion: self.distortion.map(|distortion| distortion.rescaled(scale)),
        }
    }

    /// This warp to a source grid `scale` times as fine, the reference grid
    /// as it is: the source side alone of [`Self::rescaled`], `C ∘ W`, its
    /// lens moving with the source.
    pub(crate) fn source_rescaled(&self, scale: f64) -> Self {
        let offset = (scale - 1.0) / 2.0;
        let carry = |coefficients: &[f64]| {
            coefficients
                .iter()
                .enumerate()
                .map(|(term, &value)| {
                    if term == 0 {
                        scale * value + offset
                    } else {
                        scale * value
                    }
                })
                .collect::<Vec<_>>()
        };
        Self {
            x: carry(&self.x),
            y: carry(&self.y),
            distortion: self.distortion.map(|distortion| distortion.rescaled(scale)),
            ..self.clone()
        }
    }

    /// This polynomial read as reaching undistorted source positions, which
    /// `distortion` carries onto the source's pixels.
    pub(crate) fn with_distortion(self, distortion: crate::lens::LensDistortion) -> Self {
        Self {
            distortion: Some(distortion),
            ..self
        }
    }

    /// A context saves this lens separately and attaches it after decoding
    /// the legacy warp payload. Refuse to discard any other distortion.
    pub(crate) fn for_context(mut self, lens: Option<crate::lens::LensDistortion>) -> Result<Self> {
        if self.distortion != lens {
            return Err(Error::Stack(
                "frame warp does not share the context's lens".into(),
            ));
        }
        self.distortion = None;
        Ok(self)
    }

    /// The first-order polynomial equal to the inverse of `transform`.
    pub(crate) fn from_similarity(transform: SimilarityTransform) -> Self {
        let inverse = transform.inverse_map();
        let (x0, y0) = inverse(0.0, 0.0);
        let (x1, y1) = inverse(1.0, 0.0);
        let (x2, y2) = inverse(0.0, 1.0);
        Self {
            order: 1,
            center_x: 0.0,
            center_y: 0.0,
            scale: 1.0,
            x: vec![x0, x1 - x0, x2 - x0],
            y: vec![y0, y1 - y0, y2 - y0],
            distortion: None,
        }
    }

    /// Check the order, coefficient counts, and that every value is finite.
    pub fn validate(&self) -> Result<()> {
        let terms = Self::term_count(self.order);
        if !(1..=3).contains(&self.order)
            || self.x.len() != terms
            || self.y.len() != terms
            || !(self.scale.is_finite() && self.scale > 0.0)
            || !self.center_x.is_finite()
            || !self.center_y.is_finite()
            || self.x.iter().chain(&self.y).any(|value| !value.is_finite())
        {
            return Err(Error::Registration(
                "polynomial warp must be order 1, 2 or 3 with finite coefficients".into(),
            ));
        }
        if let Some(distortion) = &self.distortion {
            distortion.validate()?;
        }
        Ok(())
    }

    fn term_count(order: u8) -> usize {
        match order {
            1 => 3,
            3 => 10,
            _ => 6,
        }
    }

    fn terms(order: u8, u: f64, v: f64) -> [f64; 10] {
        if order == 1 {
            [1.0, u, v, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        } else if order == 3 {
            [
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
            ]
        } else {
            [1.0, u, v, u * u, u * v, v * v, 0.0, 0.0, 0.0, 0.0]
        }
    }

    /// Least-squares fit of `source = P(reference)` over point pairs, by the
    /// normal equations with partial pivoting. `None` when they are singular.
    pub(crate) fn fit(order: u8, width: usize, height: usize, pairs: &[PointPair]) -> Option<Self> {
        let center_x = width as f64 * 0.5;
        let center_y = height as f64 * 0.5;
        let scale = width.max(height) as f64 * 0.5;
        let count = Self::term_count(order);
        let mut normal = [[0.0_f64; 10]; 10];
        let mut right_x = [0.0_f64; 10];
        let mut right_y = [0.0_f64; 10];
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
            distortion: None,
        })
    }
}

/// Solve the leading `count` x `count` system by Gaussian elimination with
/// partial pivoting.
fn solve(mut matrix: [[f64; 10]; 10], mut right: [f64; 10], count: usize) -> Option<Vec<f64>> {
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
pub(crate) type PointPair = ((f64, f64), (f64, f64));

/// Stars detected for a polynomial fit, against 200 for matching.
const DISTORTION_STARS: usize = 2_000;
/// How far a star may sit from its partner under the similarity transform
/// to be paired for the polynomial fit.
const DISTORTION_PAIR_PIXELS: f64 = 2.0;
/// [`DISTORTION_PAIR_PIXELS`] on a grid binned at `scale`, at least a pixel.
fn distortion_pair_pixels(scale: Scale) -> f64 {
    scale.floored_length(DISTORTION_PAIR_PIXELS, 1.0)
}

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
    pub(crate) fn dense_star_count(&self) -> usize {
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
    /// [`Self::reference_triangles`]' indices by their first ratio, and
    /// those ratios, for [`Self::register_similarity_stars`] to find the
    /// triangles a source triangle can match by binary search.
    triangles_by_ratio: Vec<usize>,
    first_ratios: Vec<f64>,
    /// Up to [`DISTORTION_STARS`] reference stars, brightest first, for a
    /// polynomial fit; empty for a similarity model.
    dense_stars: Vec<DetectedStar>,
    dense_index: StarSpatialIndex,
    maximum_drift_pixels: f64,
    /// In working pixels; see [`Self::at_scale`].
    options: RegistrationOptions,
    /// The bin factor of the frames it registers.
    scale: Scale,
}

impl Registrar {
    /// Detect stars in the reference frame and prepare the matching index.
    pub fn new(reference: &LinearImage, options: RegistrationOptions) -> Result<Self> {
        Self::at_scale(reference, &options, Scale::NATIVE)
    }

    /// [`Self::new`] for a reference binned at `scale`, with `options` in
    /// native pixels as a host writes them: they, and every pixel constant
    /// registration uses, are read on the binned grid.
    pub(crate) fn at_scale(
        reference: &LinearImage,
        options: &RegistrationOptions,
        scale: Scale,
    ) -> Result<Self> {
        options.validate()?;
        let options = scale.registration(options);
        let stars = detect(reference, &options, options.dense_star_count(), scale);
        Self::from_stars(reference.width, reference.height, stars, options, scale)
    }

    /// A registrar for a `width` by `height` reference whose stars were
    /// detected already, brightest first, as [`Self::detected_stars`] gives
    /// them back. `options` are in working pixels, as
    /// [`Scale::registration`] gives them, for frames binned at `scale`.
    pub(crate) fn from_stars(
        width: usize,
        height: usize,
        stars: Vec<DetectedStar>,
        options: RegistrationOptions,
        scale: Scale,
    ) -> Result<Self> {
        options.validate()?;
        // Saved panel detections enter here without passing through the
        // detector. Reject corrupt coordinates before indexing or fitting.
        if stars.iter().any(|star| {
            !star.x.is_finite()
                || !star.y.is_finite()
                || star.x < 0.0
                || star.y < 0.0
                || star.x >= width as f64
                || star.y >= height as f64
                || !star.flux.is_finite()
                || !star.peak.is_finite()
        }) {
            return Err(Error::Registration(
                "reference stars must be finite and lie within the reference image".into(),
            ));
        }
        // The detector sorts by brightness before truncating, so the first
        // `maximum_stars` of a dense detection are exactly the matching set.
        let mut reference_stars = stars;
        reference_stars.truncate(options.dense_star_count());
        let dense_stars = if options.model.is_similarity() {
            Vec::new()
        } else {
            let dense = reference_stars.clone();
            reference_stars.truncate(options.maximum_stars);
            dense
        };
        let dense_index = StarSpatialIndex::new(&dense_stars, distortion_pair_pixels(scale));
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
        let mut triangles_by_ratio = (0..reference_triangles.len()).collect::<Vec<_>>();
        triangles_by_ratio.sort_by(|&left, &right| {
            reference_triangles[left].ratios[0].total_cmp(&reference_triangles[right].ratios[0])
        });
        let first_ratios = triangles_by_ratio
            .iter()
            .map(|&index| reference_triangles[index].ratios[0])
            .collect();
        let reference_index =
            StarSpatialIndex::new(&reference_stars, options.match_tolerance_pixels);
        let maximum_drift_pixels = options.effective_maximum_drift_pixels(width, height);
        Ok(Self {
            width,
            height,
            reference_stars,
            reference_index,
            reference_triangles,
            triangles_by_ratio,
            first_ratios,
            dense_stars,
            dense_index,
            maximum_drift_pixels,
            options,
            scale,
        })
    }

    /// Every reference star detected, brightest first: what
    /// [`Self::from_stars`] rebuilds this registrar from.
    pub(crate) fn detected_stars(&self) -> &[DetectedStar] {
        if self.dense_stars.is_empty() {
            &self.reference_stars
        } else {
            &self.dense_stars
        }
    }

    /// Match on a panel creator's undistorted sensor plane instead of the
    /// projected canvas. The pixel scale and matching budget stay unchanged.
    pub(crate) fn remapped_stars(
        &self,
        width: usize,
        height: usize,
        map: impl Fn(f64, f64) -> (f64, f64),
    ) -> Result<Self> {
        let mut stars = Vec::new();
        for star in self.detected_stars() {
            let (x, y) = map(star.x, star.y);
            if !x.is_finite() || !y.is_finite() {
                return Err(Error::Registration(
                    "panel projection produced non-finite stars".into(),
                ));
            }
            if x >= 0.0 && y >= 0.0 && x < width as f64 && y < height as f64 {
                stars.push(DetectedStar { x, y, ..*star });
            }
        }
        let mut options = self.options.clone();
        options.model = RegistrationModel::Similarity;
        Self::from_stars(width, height, stars, options, self.scale)
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
        self.register_stars(&self.detect_source_stars(source))
    }

    /// Detect a source frame's stars once, to register it against several
    /// references with [`Self::register_stars`].
    pub(crate) fn detect_source_stars(&self, source: &LinearImage) -> Vec<DetectedStar> {
        detect(
            source,
            &self.options,
            self.options.dense_star_count(),
            self.scale,
        )
    }

    /// [`Self::register`] with the source's stars already detected.
    pub(crate) fn register_stars(
        &self,
        dense_source: &[DetectedStar],
    ) -> Result<RegistrationResult> {
        let mut result = self.register_similarity_stars(dense_source)?;
        if let Some((warp, pairs, rms)) = self.warp_stars(result.transform, dense_source) {
            result.warp = Some(warp);
            result.matched_stars = pairs;
            result.rms_error_pixels = rms;
        }
        Ok(result)
    }

    /// Refine a similarity found elsewhere: pair the source's brightest
    /// stars within the match tolerance of where `initial` carries them, and
    /// refit as [`Self::register_similarity_stars`] does, without searching.
    pub(crate) fn refine_similarity_stars(
        &self,
        initial: SimilarityTransform,
        dense_source: &[DetectedStar],
    ) -> Result<RegistrationResult> {
        let source_stars = &dense_source[..dense_source.len().min(self.options.maximum_stars)];
        let pairs = matched_pairs(
            initial,
            source_stars,
            &self.reference_stars,
            &self.reference_index,
            self.options.match_tolerance_pixels,
        );
        if pairs.len() < self.options.minimum_matches.max(3) {
            return Err(Error::Registration(format!(
                "only {} stars pair where the initial transform places them",
                pairs.len()
            )));
        }
        self.refine_registration((pairs.len(), 0.0, initial, pairs), source_stars)
    }

    /// The similarity [`Self::register_stars`] finds, before any warp.
    pub(crate) fn register_similarity_stars(
        &self,
        dense_source: &[DetectedStar],
    ) -> Result<RegistrationResult> {
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
        let candidates = self.triangle_candidates(&source_triangles, source_stars);

        for candidate in candidates {
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
        self.refine_registration(best, source_stars)
    }

    /// The best similarities by triangle ratio error, at most
    /// `maximum_candidates` of them, among pairs of a source and a reference
    /// triangle whose ratios agree within the tolerance and whose similarity
    /// is admissible: as a scan of every source triangle against every
    /// reference triangle, stably sorted by error, would give them.
    fn triangle_candidates(
        &self,
        source_triangles: &[Triangle],
        source_stars: &[DetectedStar],
    ) -> Vec<SimilarityTransform> {
        let tolerance = self.options.descriptor_tolerance * 2.0;
        let wanted = self.options.maximum_candidates;
        // Pairs within `limit`, in the scan's order. A pair's first ratios
        // differ by no more than its error, so only the reference triangles
        // in that window of first ratios are read; a margin keeps rounding
        // in the bounds from dropping a pair the full test admits.
        let pairs_within = |limit: f64| {
            let mut pairs = Vec::new();
            let mut window = Vec::new();
            for (source_index, source_triangle) in source_triangles.iter().enumerate() {
                let [first, second] = source_triangle.ratios;
                let low = self
                    .first_ratios
                    .partition_point(|&ratio| ratio < first - limit - 1.0e-9);
                let high = self
                    .first_ratios
                    .partition_point(|&ratio| ratio <= first + limit + 1.0e-9);
                window.clear();
                window.extend_from_slice(&self.triangles_by_ratio[low..high]);
                window.sort_unstable();
                for &reference_index in &window {
                    let reference = &self.reference_triangles[reference_index];
                    let error =
                        (first - reference.ratios[0]).abs() + (second - reference.ratios[1]).abs();
                    if error <= limit {
                        pairs.push((error, source_index, reference_index));
                    }
                }
            }
            pairs
        };
        // Fit the pairs best first, stably, and keep the admissible ones;
        // only those taken are fitted.
        let admissible = |mut pairs: Vec<(f64, usize, usize)>| {
            pairs.sort_by(|left, right| left.0.total_cmp(&right.0));
            pairs
                .into_iter()
                .filter_map(|(_, source_index, reference_index)| {
                    transform_from_triangles(
                        &source_triangles[source_index],
                        &self.reference_triangles[reference_index],
                        source_stars,
                        &self.reference_stars,
                    )
                    .filter(|transform| {
                        (transform.scale - 1.0).abs() <= self.options.scale_tolerance
                            && transform
                                .displacement_at(self.width as f64 * 0.5, self.height as f64 * 0.5)
                                <= self.maximum_drift_pixels
                    })
                })
                .take(wanted)
                .collect::<Vec<_>>()
        };
        // A quarter of the tolerance usually holds enough: about 830,000
        // pairs of 9,400 triangles each agree within the whole of it, and
        // only the best few hundred are wanted. When it does hold `wanted`
        // admissible pairs, every pair ranking above the last of them lies
        // within it too, so they are the whole scan's best.
        let narrow = admissible(pairs_within(tolerance * 0.25));
        if narrow.len() == wanted {
            return narrow;
        }
        admissible(pairs_within(tolerance))
    }

    /// The polynomial warp [`Self::register_stars`] fits around `transform`,
    /// with the pairs it kept and their RMS residual; `None` for a similarity
    /// model or when too few stars pair up.
    pub(crate) fn warp_stars(
        &self,
        transform: SimilarityTransform,
        dense_source: &[DetectedStar],
    ) -> Option<(PolynomialWarp, usize, f64)> {
        self.fit_warp(
            self.options.model.polynomial_order()?,
            transform,
            dense_source,
        )
    }

    /// Match through a geometric prediction instead of a similarity. At
    /// the edge of a wide tangent plane a similarity can miss stars by
    /// tens of pixels even when the camera's pose is already known.
    pub(crate) fn warp_stars_seeded(
        &self,
        seed: &PolynomialWarp,
        source_stars: &[DetectedStar],
    ) -> Option<(PolynomialWarp, usize, f64)> {
        // This seed already describes the canvas projection. Similarity
        // limits an unseeded fit; it must not erase known mosaic geometry.
        let order = self
            .options
            .model
            .polynomial_order()
            .unwrap_or(1)
            .max(seed.order());
        let pairing = distortion_pair_pixels(self.scale);
        let index = StarSpatialIndex::new(source_stars, pairing);
        let mut seen = std::collections::HashSet::new();
        let pairs = self
            .detected_stars()
            .iter()
            .filter_map(|reference| {
                let (x, y) = seed.apply(reference.x, reference.y);
                let (matched, _) = index.nearest_within(x, y, source_stars, pairing)?;
                if !seen.insert(matched) {
                    return None;
                }
                let source = &source_stars[matched];
                Some(((reference.x, reference.y), (source.x, source.y)))
            })
            .collect();
        self.fit_warp_pairs(order, pairs)
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
        let pairs = source_stars
            .iter()
            .filter_map(|star| {
                let (x, y) = transform.apply(star.x, star.y);
                let (index, _) = self.dense_index.nearest_within(
                    x,
                    y,
                    &self.dense_stars,
                    distortion_pair_pixels(self.scale),
                )?;
                let reference = &self.dense_stars[index];
                Some(((reference.x, reference.y), (star.x, star.y)))
            })
            .collect::<Vec<_>>();
        self.fit_warp_pairs(order, pairs)
    }

    fn fit_warp_pairs(
        &self,
        order: u8,
        mut pairs: Vec<PointPair>,
    ) -> Option<(PolynomialWarp, usize, f64)> {
        let minimum = PolynomialWarp::term_count(order) * DISTORTION_PAIRS_PER_TERM;
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
            let limit = 3.0 * residuals[residuals.len() / 2] + self.scale.length(0.05);
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

/// Refit a similarity found elsewhere against the stars of `reference`,
/// pairing each source star within the match tolerance of where `initial`
/// carries it. A star cut by the edge of either image has a biased centroid,
/// and a narrow overlap holds many, so stars within a few pixels of an edge
/// or of an undefined reference sample are left out. `None` when fewer than
/// the minimum matches pair up. Up to `stars` of the brightest on each side
/// take part. `options` are in working pixels, for images binned at `scale`.
pub(crate) fn refine_similarity(
    reference: &LinearImage,
    source: &LinearImage,
    source_stars: &[DetectedStar],
    initial: SimilarityTransform,
    options: &RegistrationOptions,
    stars: usize,
    scale: Scale,
) -> Option<RegistrationResult> {
    let margin = scale.centroid_margin(6);
    let defined = |image: &LinearImage, star: &DetectedStar| {
        let (cx, cy) = (star.x.round() as isize, star.y.round() as isize);
        (cy - margin..=cy + margin).all(|y| {
            (cx - margin..=cx + margin).all(|x| {
                x >= 0
                    && y >= 0
                    && (x as usize) < image.width
                    && (y as usize) < image.height
                    && image.data[(y as usize * image.width + x as usize) * image.channels]
                        .is_finite()
            })
        })
    };
    let reference_stars = detect(reference, options, stars, scale)
        .into_iter()
        .filter(|star| defined(reference, star))
        .collect::<Vec<_>>();
    let source_stars = source_stars[..source_stars.len().min(stars)]
        .iter()
        .filter(|star| defined(source, star))
        .cloned()
        .collect::<Vec<_>>();
    let source_stars = &source_stars[..];
    let index = StarSpatialIndex::new(&reference_stars, options.match_tolerance_pixels);
    let mut transform = initial;
    let mut pairs = Vec::new();
    for _ in 0..3 {
        pairs = matched_pairs(
            transform,
            source_stars,
            &reference_stars,
            &index,
            options.match_tolerance_pixels,
        );
        if pairs.len() < options.minimum_matches {
            return None;
        }
        transform = fit_similarity(&pairs, source_stars, &reference_stars).ok()?;
    }
    let rms = (pair_squared_error(transform, &pairs, source_stars, &reference_stars)
        / pairs.len() as f64)
        .sqrt();
    Some(RegistrationResult {
        transform,
        matched_stars: pairs.len(),
        rms_error_pixels: rms,
        drift_pixels: 0.0,
        warp: None,
    })
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
    /// clamping does. Only an edge well above the frame's noise counts, so
    /// sky noise is never clamped: that would bias a faint background by an
    /// amount that follows each frame's sub-pixel phase, and draw bands
    /// across a stack of rotated frames. Sharper: on a 98-frame M45 stack
    /// it took FWHM from 2.63 to 2.47px, at about twice the resampling
    /// cost (measured before the noise gate). Falls back to bilinear within
    /// three pixels of the source edge and where a tap is not finite.
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

/// [`resample_region_geometry`] of `source`, and of `mask`, its clip mask,
/// one sample a source pixel, bilinearly, as [`resample_region_geometry`]
/// resamples it: in one pass on the GPU, which finds every pixel's source
/// position once for both.
#[allow(clippy::too_many_arguments)]
pub(crate) fn resample_region_geometry_masked(
    source: &LinearImage,
    mask: Option<&LinearImage>,
    reference_width: usize,
    reference_height: usize,
    region: ReferenceRegion,
    geometry: FrameGeometry<'_>,
    sampling: Sampling,
) -> Result<(LinearImage, Option<LinearImage>)> {
    if let Sampling::NearestPhotosite(_) = sampling {
        let image = resample_region_geometry(
            source,
            reference_width,
            reference_height,
            region,
            geometry,
            sampling,
        )?;
        let mask = mask
            .map(|mask| {
                resample_region_geometry(
                    mask,
                    reference_width,
                    reference_height,
                    region,
                    geometry,
                    Sampling::Bilinear,
                )
            })
            .transpose()?;
        return Ok((image, mask));
    }
    match geometry {
        FrameGeometry::Similarity(transform) => {
            transform.validate()?;
            resample_region_with_inverse_masked(
                source,
                mask,
                reference_width,
                reference_height,
                region,
                transform.inverse_map(),
                sampling,
            )
        }
        FrameGeometry::Warp(warp) => {
            warp.validate()?;
            resample_region_with_inverse_masked(
                source,
                mask,
                reference_width,
                reference_height,
                region,
                |x, y| warp.apply(x, y),
                sampling,
            )
        }
    }
}

/// The smallest region of a `reference_width` by `reference_height` grid
/// holding every pixel [`resample_region_geometry`] can give a finite sample
/// for a `source_width` by `source_height` frame: those whose source position
/// lies on the frame. Every pixel outside it resamples to `NaN`, so resampling
/// only the region, and treating the rest as `NaN`, gives the same image.
///
/// A similarity carries the frame's corners there directly. A warp maps the
/// other way, reference to source, so points along the frame's edges are
/// carried back through it by Newton's method from where `similarity`, the
/// transform the warp refines, puts them. The region is then checked: when a
/// pixel just outside it reaches the frame, as only a warp that bends far
/// from the similarity could make one, or when an edge cannot be carried
/// back, the whole grid is returned.
///
/// The region holds the frame's own footprint. Far beyond the frame, where
/// it was never measured, a strong lens distortion can fold back onto the
/// frame, so that the grid would sample the frame a second time there; the
/// region leaves that out.
///
/// A region is never empty: a frame off the grid gets a one-pixel region,
/// which resamples to `NaN`.
pub(crate) fn footprint_region(
    source_width: usize,
    source_height: usize,
    reference_width: usize,
    reference_height: usize,
    geometry: FrameGeometry<'_>,
    similarity: SimilarityTransform,
) -> ReferenceRegion {
    let whole = ReferenceRegion {
        x: 0,
        y: 0,
        width: reference_width,
        height: reference_height,
    };
    if source_width == 0 || source_height == 0 || reference_width == 0 || reference_height == 0 {
        return whole;
    }
    let (right, bottom) = ((source_width - 1) as f64, (source_height - 1) as f64);
    let landed = match geometry {
        FrameGeometry::Similarity(transform) => [(0.0, 0.0), (right, 0.0), (0.0, bottom)]
            .into_iter()
            .chain([(right, bottom)])
            .map(|(x, y)| Some(transform.apply(x, y)))
            .collect::<Option<Vec<_>>>(),
        FrameGeometry::Warp(warp) => frame_edge(right, bottom)
            .map(|(x, y)| invert_warp(warp, (x, y), similarity.apply(x, y)))
            .collect::<Option<Vec<_>>>(),
    };
    let Some(landed) =
        landed.filter(|points| points.iter().all(|(x, y)| x.is_finite() && y.is_finite()))
    else {
        return whole;
    };
    let (mut left, mut top, mut far_right, mut far_bottom) = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for &(x, y) in &landed {
        left = left.min(x);
        top = top.min(y);
        far_right = far_right.max(x);
        far_bottom = far_bottom.max(y);
    }
    // Pixels whose centres lie within the frame's edges, and a margin for
    // the edge to bow between the points carried back.
    let clamp = |value: f64, limit: usize| value.clamp(0.0, limit as f64) as usize;
    let x = clamp(left.floor() - FOOTPRINT_MARGIN, reference_width - 1);
    let y = clamp(top.floor() - FOOTPRINT_MARGIN, reference_height - 1);
    let region = ReferenceRegion {
        x,
        y,
        width: clamp(far_right.ceil() + 1.0 + FOOTPRINT_MARGIN, reference_width).max(x + 1) - x,
        height: clamp(far_bottom.ceil() + 1.0 + FOOTPRINT_MARGIN, reference_height).max(y + 1) - y,
    };
    let reaches = |inverse: &dyn Fn(f64, f64) -> (f64, f64)| {
        ring(region, reference_width, reference_height).any(|(x, y)| {
            let (source_x, source_y) = inverse(x as f64, y as f64);
            within_source(source_x, source_y, right, bottom)
        })
    };
    let escapes = match geometry {
        FrameGeometry::Similarity(transform) => reaches(&transform.inverse_map()),
        FrameGeometry::Warp(warp) => reaches(&|x, y| warp.apply(x, y)),
    };
    if escapes { whole } else { region }
}

/// Pixels a frame's edge may bow outward past the bounds of the points of it
/// [`footprint_region`] carries onto the reference grid.
const FOOTPRINT_MARGIN: f64 = 2.0;

/// Points [`FOOTPRINT_EDGE_STEP`] pixels apart along the edge of a frame
/// whose far corner is `(right, bottom)`, the corners among them.
fn frame_edge(right: f64, bottom: f64) -> impl Iterator<Item = (f64, f64)> {
    let steps = |length: f64| ((length / FOOTPRINT_EDGE_STEP).ceil() as usize).max(1);
    let (across, down) = (steps(right), steps(bottom));
    let horizontal = (0..=across).flat_map(move |step| {
        let x = right * step as f64 / across as f64;
        [(x, 0.0), (x, bottom)]
    });
    let vertical = (1..down).flat_map(move |step| {
        let y = bottom * step as f64 / down as f64;
        [(0.0, y), (right, y)]
    });
    horizontal.chain(vertical)
}

/// The spacing of [`frame_edge`]'s points: a warp's edge bows far less than
/// [`FOOTPRINT_MARGIN`] between points this close.
const FOOTPRINT_EDGE_STEP: f64 = 16.0;

/// The reference pixels on the grid just outside `region`.
fn ring(
    region: ReferenceRegion,
    reference_width: usize,
    reference_height: usize,
) -> impl Iterator<Item = (usize, usize)> {
    let (left, top) = (region.x.checked_sub(1), region.y.checked_sub(1));
    let right = Some(region.x + region.width).filter(|&x| x < reference_width);
    let bottom = Some(region.y + region.height).filter(|&y| y < reference_height);
    let columns = region.x.saturating_sub(1)..(region.x + region.width + 1).min(reference_width);
    let rows = region.y..region.y + region.height;
    let across = [top, bottom]
        .into_iter()
        .flatten()
        .flat_map(move |y| columns.clone().map(move |x| (x, y)));
    let down = [left, right]
        .into_iter()
        .flatten()
        .flat_map(move |x| rows.clone().map(move |y| (x, y)));
    across.chain(down)
}

/// The reference position `warp` carries to source position `target`, by
/// Newton's method from `seed`, or `None` when it does not converge.
fn invert_warp(warp: &PolynomialWarp, target: (f64, f64), seed: (f64, f64)) -> Option<(f64, f64)> {
    let (mut x, mut y) = seed;
    for _ in 0..32 {
        let (u, v) = warp.apply(x, y);
        let (error_u, error_v) = (u - target.0, v - target.1);
        if !(error_u.is_finite() && error_v.is_finite()) {
            return None;
        }
        if error_u.hypot(error_v) < 1.0e-6 {
            return Some((x, y));
        }
        // The Jacobian, by central differences a pixel wide.
        let (right, left) = (warp.apply(x + 0.5, y), warp.apply(x - 0.5, y));
        let (below, above) = (warp.apply(x, y + 0.5), warp.apply(x, y - 0.5));
        let (du_dx, dv_dx) = (right.0 - left.0, right.1 - left.1);
        let (du_dy, dv_dy) = (below.0 - above.0, below.1 - above.1);
        let determinant = du_dx * dv_dy - du_dy * dv_dx;
        if !determinant.is_finite() || determinant.abs() < 1.0e-12 {
            return None;
        }
        x -= (dv_dy * error_u - du_dy * error_v) / determinant;
        y -= (du_dx * error_v - dv_dx * error_u) / determinant;
    }
    None
}

/// Whether resampling reads a frame whose far corner is
/// `(maximum_x, maximum_y)` at a source position: one on the frame, give or
/// take rounding, or one that is not a number.
#[inline(always)]
fn within_source(source_x: f64, source_y: f64, maximum_x: f64, maximum_y: f64) -> bool {
    const COORDINATE_EPSILON: f64 = 1.0e-9;
    !(source_x < -COORDINATE_EPSILON
        || source_y < -COORDINATE_EPSILON
        || source_x > maximum_x + COORDINATE_EPSILON
        || source_y > maximum_y + COORDINATE_EPSILON)
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

/// Where each channel of a frame samples the sky, relative to its pixel's
/// centre, in its own pixels: red, green and blue.
///
/// A [`crate::Demosaic::Superpixel`] pixel takes its red and blue from one
/// photosite each, half a sensor pixel from the 2x2 cell's centre
/// diagonally, and its green from the two greens either side of it, whose
/// mean lies at the centre. Binning on by the rest of the bin factor keeps
/// those offsets in sensor pixels, so in the frame's own they are
/// `±0.5 / bin_factor`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct ChannelSites(pub(crate) [(f64, f64); 3]);

impl ChannelSites {
    pub(crate) fn validate(self) -> Result<()> {
        if self
            .0
            .iter()
            .any(|&(x, y)| !x.is_finite() || !y.is_finite() || x.abs() > 0.5 || y.abs() > 0.5)
        {
            return Err(Error::Registration(
                "channel sites must be finite offsets within half a pixel".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn rescaled(self, scale: f64) -> Self {
        Self(self.0.map(|(x, y)| (x * scale, y * scale)))
    }

    /// The sites of a superpixel frame of `layout`, binned by `bin_factor`
    /// in all.
    pub(crate) fn superpixel(layout: crate::BayerLayout, bin_factor: u32) -> Self {
        let mut sites = [(0.0, 0.0); 3];
        for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let channel = layout.channel_at(x, y);
            if channel != 1 {
                sites[channel] = (
                    (x as f64 - 0.5) / f64::from(bin_factor),
                    (y as f64 - 0.5) / f64::from(bin_factor),
                );
            }
        }
        Self(sites)
    }
}

/// [`resample_region_geometry`] for a frame whose channels sample the sky
/// at `sites`: each output pixel takes every channel from where that
/// channel's samples put the pixel's centre, so its colours line up.
/// Without sites it is [`resample_region_geometry`] itself.
/// [`resample_region_sited`] of `source`, and of `mask`, its clip mask,
/// bilinearly at the pixels' own positions, as
/// [`resample_region_geometry_masked`] resamples it; with channel sites, the
/// mask is resampled apart.
#[allow(clippy::too_many_arguments)]
pub(crate) fn resample_region_sited_masked(
    source: &LinearImage,
    mask: Option<&LinearImage>,
    reference_width: usize,
    reference_height: usize,
    region: ReferenceRegion,
    geometry: FrameGeometry<'_>,
    sampling: Sampling,
    sites: Option<ChannelSites>,
) -> Result<(LinearImage, Option<LinearImage>)> {
    if sites.is_some() && source.channels == 3 {
        let image = resample_region_sited(
            source,
            reference_width,
            reference_height,
            region,
            geometry,
            sampling,
            sites,
        )?;
        let mask = mask
            .map(|mask| {
                resample_region_geometry(
                    mask,
                    reference_width,
                    reference_height,
                    region,
                    geometry,
                    Sampling::Bilinear,
                )
            })
            .transpose()?;
        return Ok((image, mask));
    }
    resample_region_geometry_masked(
        source,
        mask,
        reference_width,
        reference_height,
        region,
        geometry,
        sampling,
    )
}

pub(crate) fn resample_region_sited(
    source: &LinearImage,
    reference_width: usize,
    reference_height: usize,
    region: ReferenceRegion,
    geometry: FrameGeometry<'_>,
    sampling: Sampling,
    sites: Option<ChannelSites>,
) -> Result<LinearImage> {
    let Some(ChannelSites(sites)) = sites.filter(|_| source.channels == 3) else {
        return resample_region_geometry(
            source,
            reference_width,
            reference_height,
            region,
            geometry,
            sampling,
        );
    };
    if let Sampling::NearestPhotosite(_) = sampling {
        return Err(Error::Registration(
            "photosite sampling takes no channel sites".into(),
        ));
    }
    // A channel sampled at its site `d` holds the sky at `x + d` at pixel
    // `x`, so the sky at source position `p` is its sample at `p − d`.
    let channels = sites
        .iter()
        .enumerate()
        .map(|(channel, &(dx, dy))| {
            let source = source.channel(channel)?;
            let shifted = |(x, y): (f64, f64)| (x - dx, y - dy);
            match geometry {
                FrameGeometry::Similarity(transform) => {
                    transform.validate()?;
                    let inverse = transform.inverse_map();
                    resample_region_with_inverse(
                        &source,
                        reference_width,
                        reference_height,
                        region,
                        |x, y| shifted(inverse(x, y)),
                        sampling,
                    )
                }
                FrameGeometry::Warp(warp) => {
                    warp.validate()?;
                    resample_region_with_inverse(
                        &source,
                        reference_width,
                        reference_height,
                        region,
                        |x, y| shifted(warp.apply(x, y)),
                        sampling,
                    )
                }
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let mut data = vec![0.0_f32; region.width * region.height * 3];
    data.par_chunks_mut(3).enumerate().for_each(|(pixel, rgb)| {
        for (value, channel) in rgb.iter_mut().zip(&channels) {
            *value = channel.data[pixel];
        }
    });
    LinearImage::new(region.width, region.height, 3, data)
}

/// Which samples [`resample_region_sited`] would leave finite, sample by
/// sample, without interpolating: for counting what a frame covers when its
/// pixels are not wanted. A sample is finite exactly where its channel's
/// source position lies within the source, when every source sample is
/// finite and small enough that no interpolation overflows. `None` for a
/// source that is not, or for photosite sampling, which fills one channel
/// of each pixel.
pub(crate) fn resampled_coverage(
    source: &LinearImage,
    region: ReferenceRegion,
    geometry: FrameGeometry<'_>,
    sampling: Sampling,
    sites: Option<ChannelSites>,
) -> Result<Option<Vec<bool>>> {
    // Interpolation weights sum to one and Lanczos's clamped lobes stay
    // within a few times the taps, so a quarter of f32's range is safe.
    const LARGEST: f32 = f32::MAX / 16.0;
    if matches!(sampling, Sampling::NearestPhotosite(_))
        || !source
            .data
            .iter()
            .all(|value| value.is_finite() && value.abs() <= LARGEST)
    {
        return Ok(None);
    }
    let channels = source.channels;
    let sites = match sites.filter(|_| channels == 3) {
        Some(ChannelSites(sites)) => sites.to_vec(),
        None => vec![(0.0, 0.0); channels],
    };
    let inverse: Box<dyn Fn(f64, f64) -> (f64, f64) + Sync> = match geometry {
        FrameGeometry::Similarity(transform) => {
            transform.validate()?;
            Box::new(transform.inverse_map())
        }
        FrameGeometry::Warp(warp) => {
            warp.validate()?;
            Box::new(|x, y| warp.apply(x, y))
        }
    };
    let maximum_x = (source.width - 1) as f64;
    let maximum_y = (source.height - 1) as f64;
    let mut covered = vec![false; region.width * region.height * channels];
    covered
        .par_chunks_mut(region.width * channels)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, pixel) in row.chunks_exact_mut(channels).enumerate() {
                let (source_x, source_y) = inverse((region.x + x) as f64, (region.y + y) as f64);
                for (covered, &(dx, dy)) in pixel.iter_mut().zip(&sites) {
                    *covered = within_source(source_x - dx, source_y - dy, maximum_x, maximum_y);
                }
            }
        });
    Ok(Some(covered))
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
    resample_region_with_inverse_masked(
        source,
        None,
        reference_width,
        reference_height,
        region,
        inverse,
        sampling,
    )
    .map(|(image, _)| image)
}

/// [`resample_region_with_inverse`] of `source`, and of `mask` bilinearly,
/// in one pass on the GPU, or else one after the other.
fn resample_region_with_inverse_masked(
    source: &LinearImage,
    mask: Option<&LinearImage>,
    reference_width: usize,
    reference_height: usize,
    region: ReferenceRegion,
    inverse: impl Fn(f64, f64) -> (f64, f64) + Sync,
    sampling: Sampling,
) -> Result<(LinearImage, Option<LinearImage>)> {
    // The mask alone, as `resample_region_with_inverse` resamples it
    // bilinearly where the GPU's joint pass did not take it; the region is
    // the frame's, checked below.
    let mask_alone = |mask: &LinearImage| -> Result<LinearImage> {
        #[cfg(feature = "gpu")]
        if let Some((image, _)) = crate::gpu::resample_masked(
            mask,
            None,
            region,
            &inverse,
            crate::gpu::GpuSampling::Bilinear,
        ) {
            return Ok(image);
        }
        resample_region_on_cpu(
            mask,
            region,
            &inverse,
            Sampling::Bilinear,
            &vec![0.0; mask.channels],
        )
    };
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
    // Checked here, so both paths can size their output plainly.
    region
        .width
        .checked_mul(region.height)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or_else(|| Error::Registration("resampling output dimensions overflow".into()))?;
    let clamp_spans = match sampling {
        Sampling::Lanczos3 => lanczos_clamp_spans(source),
        _ => vec![0.0; channels],
    };
    #[cfg(feature = "gpu")]
    {
        let gpu_sampling = match sampling {
            Sampling::Lanczos3 => {
                let mut spans = [0.0_f32; 4];
                for (span, &value) in spans.iter_mut().zip(&clamp_spans) {
                    *span = value;
                }
                Some(crate::gpu::GpuSampling::Lanczos3 { clamp_spans: spans })
            }
            Sampling::Bilinear => Some(crate::gpu::GpuSampling::Bilinear),
            Sampling::NearestPhotosite(_) => None,
        };
        if let Some(gpu_sampling) = gpu_sampling
            && let Some((image, resampled_mask)) =
                crate::gpu::resample_masked(source, mask, region, &inverse, gpu_sampling)
        {
            let resampled_mask = match (resampled_mask, mask) {
                (Some(resampled), _) => Some(resampled),
                (None, Some(mask)) => Some(mask_alone(mask)?),
                (None, None) => None,
            };
            return Ok((image, resampled_mask));
        }
    }
    let image = resample_region_on_cpu(source, region, &inverse, sampling, &clamp_spans)?;
    Ok((image, mask.map(mask_alone).transpose()?))
}

/// [`resample_region_with_inverse`]'s CPU path, for a checked region.
pub(crate) fn resample_region_on_cpu(
    source: &LinearImage,
    region: ReferenceRegion,
    inverse: &(impl Fn(f64, f64) -> (f64, f64) + Sync),
    sampling: Sampling,
    clamp_spans: &[f32],
) -> Result<LinearImage> {
    let channels = source.channels;
    let row_samples = region.width * channels;
    let mut data = vec![f32::NAN; row_samples * region.height];
    data.par_chunks_mut(row_samples)
        .enumerate()
        .for_each(|(y, output_row)| {
            resample_row(
                source,
                region.x,
                region.y + y,
                inverse,
                sampling,
                clamp_spans,
                output_row,
            );
        });
    LinearImage::new(region.width, region.height, channels, data)
}

/// One output row of [`resample_region_with_inverse`]: reference row
/// `reference_y` from column `first_x` on. `clamp_spans` holds each
/// channel's [`lanczos_clamp_spans`].
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
    clamp_spans: &[f32],
    output_row: &mut [f32],
) where
    F: Fn(f64, f64) -> (f64, f64),
{
    let channels = source.channels;
    let maximum_source_x = (source.width - 1) as f64;
    let maximum_source_y = (source.height - 1) as f64;
    for (x, output) in output_row.chunks_exact_mut(channels).enumerate() {
        let reference_x = first_x + x;
        let (source_x, source_y) = inverse(reference_x as f64, reference_y as f64);
        if !within_source(source_x, source_y, maximum_source_x, maximum_source_y) {
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
                let spans = [
                    clamp_spans[0],
                    clamp_spans[1],
                    clamp_spans[2],
                    clamp_spans[0],
                ];
                let mut rows = [[0.0_f32; 4]; 6];
                for (row, value) in rows.iter_mut().enumerate() {
                    let start = ((y0 + row - 2) * source.width + x0 - 2) * channels;
                    let window = &source.data[start..start + 19];
                    let taps: [[f32; 4]; 6] = std::array::from_fn(|column| {
                        let tap = &window[column * 3..column * 3 + 4];
                        [tap[0], tap[1], tap[2], tap[3]]
                    });
                    *value = clamped_lanczos_lanes(&weights_x, &taps, &spans);
                }
                // A non-finite tap carries through to the result.
                let values = clamped_lanczos_lanes(&weights_y, &rows, &spans);
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
                    *value = clamped_lanczos(&weights_x, &taps, clamp_spans[channel]);
                }
                // A non-finite tap carries through to the result.
                let value = clamped_lanczos(&weights_y, &rows, clamp_spans[channel]);
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
pub(crate) const LANCZOS_CLAMPING: f32 = 0.3;

/// How many times a frame's photosite noise the taps of a 1-D Lanczos pass
/// must span, highest to lowest, before [`LANCZOS_CLAMPING`] may drop the
/// negative lobes.
///
/// The share test alone compares sample values with zero, so on a faint
/// calibrated background, a few times its noise above zero, it fires on
/// noise: on Sh2 140 OIII lights, 34 ADU of sky with 12.8 ADU of noise, it
/// clamped up to a fifth of the passes. Dropping the negative lobes there
/// is not neutral. It fires when the negative-lobe samples happen to be
/// high, so it raised the background by up to 4% and smoothed its noise,
/// both by an amount set by the sub-pixel phase. A frame rotated against
/// the reference sweeps that phase through a whole pixel every
/// `1 / sin(angle)` pixels, so a stack of frames rotated 0.3 degrees showed
/// bands about 190 pixels apart. Six samples of plain noise span more than
/// five times its standard deviation in about one pass in 200, and then the
/// share test must fire too, so noise is left alone. The second pass's taps
/// are the first pass's results, whose noise is 0.89 to 1 of the frame's
/// by phase, yet are held to the frame's own: a slightly stricter gate.
///
/// A star whose taps span less than this keeps its full kernel, and rings
/// as an unclamped Lanczos-3 does. Ringing is a fixed share of the star, so
/// unlike noise it does not average down across a stack: but for a star of
/// 2 px FWHM or wider it is under 1% of the peak, and for an undersampled
/// 1.5 px star about 2-4%, so a star that never clears five times the
/// noise in one frame rings by about a fifth of that frame's noise at
/// most. In a crowded field, a star to every 100-200 pixels, the stars
/// lift [`crate::snr::photosite_noise`] by a fifth to a half, the gate
/// sits nearer six or seven times the true noise, and that bound nearer
/// 0.3 of it. That is the ringing a sky bright enough to keep the share
/// test quiet gave every faint star before this gate existed, as it does in
/// PixInsight. Brighter stars span far more and are clamped as before.
const LANCZOS_CLAMPING_NOISE_SPAN: f32 = 5.0;

/// The span, per channel, that a 1-D Lanczos pass's taps must exceed before
/// it may clamp: [`LANCZOS_CLAMPING_NOISE_SPAN`] times the source's
/// [`crate::snr::photosite_noise`], which a demosaic does not lower, or
/// zero, clamping as the share test alone says, where the image is too
/// small to measure its noise.
pub(crate) fn lanczos_clamp_spans(source: &LinearImage) -> Vec<f32> {
    match crate::snr::photosite_noise(source) {
        Some(noise) => noise
            .into_iter()
            .map(|sigma| {
                if sigma.is_finite() {
                    LANCZOS_CLAMPING_NOISE_SPAN * sigma
                } else {
                    0.0
                }
            })
            .collect(),
        None => vec![0.0; source.channels],
    }
}

/// One 1-D Lanczos-3 pass. Where the taps span more than `noise_span` and
/// the negative lobes would contribute more than [`LANCZOS_CLAMPING`] of
/// what the positive lobes do, the taps straddle a high-contrast edge such
/// as a bright star beside the sky, and the negative lobes would ring; then
/// only the positive lobes are used, renormalized. Elsewhere, including at a
/// star's peak and on the noise of a faint sky, the full kernel keeps its
/// sharpness and its mean.
fn clamped_lanczos(weights: &[f32; 6], taps: &[f32; 6], noise_span: f32) -> f32 {
    // Between two samples the Lanczos-3 taps at offsets -2..=3 always carry
    // the signs + - + + - +, so the lobes need no per-tap test.
    let positive =
        weights[0] * taps[0] + weights[2] * taps[2] + weights[3] * taps[3] + weights[5] * taps[5];
    let negative = weights[1] * taps[1] + weights[4] * taps[4];
    // The span only once the share test fires, which most passes do not.
    if negative.abs() > LANCZOS_CLAMPING * positive.abs() && tap_span(taps) > noise_span {
        positive / (weights[0] + weights[2] + weights[3] + weights[5])
    } else {
        positive + negative
    }
}

/// The highest tap less the lowest. `max` and `min` pass over a NaN tap;
/// an infinite one makes the span infinite, but then the pass's result is
/// not finite either way, and the caller falls back to bilinear.
#[inline(always)]
fn tap_span(taps: &[f32; 6]) -> f32 {
    let mut highest = f32::NEG_INFINITY;
    let mut lowest = f32::INFINITY;
    for &tap in taps {
        highest = highest.max(tap);
        lowest = lowest.min(tap);
    }
    highest - lowest
}

/// [`clamped_lanczos`] on four lanes at once: lane `k` of the result is
/// `clamped_lanczos(weights, taps[..][k], noise_spans[k])`, from the same operations in the
/// same order, so each lane matches it bit for bit. Both branches are formed
/// and one is selected per lane, which the compiler turns into vector
/// arithmetic and a blend.
#[inline(always)]
fn clamped_lanczos_lanes(
    weights: &[f32; 6],
    taps: &[[f32; 4]; 6],
    noise_spans: &[f32; 4],
) -> [f32; 4] {
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
        let span = tap_span(&std::array::from_fn(|tap| taps[tap][lane]));
        *result = if span > noise_spans[lane] && negative.abs() > LANCZOS_CLAMPING * positive.abs()
        {
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
    lanczos3_table()[((t * LANCZOS_STEPS as f32 + 0.5) as usize).min(LANCZOS_STEPS)]
}

/// [`lanczos3_weights`]' table: the weights at every 1/1024-pixel step from
/// 0 to 1, which the GPU resampler reads too.
pub(crate) fn lanczos3_table() -> &'static [[f32; 6]] {
    static TABLE: std::sync::OnceLock<Vec<[f32; 6]>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
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
    })
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
    let translation = |pair: usize| {
        let (source, reference) = (
            &source[pair / reference.len()],
            &reference[pair % reference.len()],
        );
        (reference.x - source.x, reference.y - source.y)
    };
    // Each pair within the drift, by its bin, in the order the pairs are
    // scanned. Sorting these and summing each run of one bin gives every
    // bin the votes, and the sums in the order, a map filled in that scan
    // would: 800 stars against 800 fall in some 450,000 bins, and a map that
    // size missed the cache on nearly every vote.
    let mut binned = (0..source.len() * reference.len())
        .into_par_iter()
        .filter_map(|pair| {
            let (translation_x, translation_y) = translation(pair);
            if translation_x.hypot(translation_y) > maximum_drift_pixels {
                return None;
            }
            let key = (
                (translation_x / bin_size).round() as i32,
                (translation_y / bin_size).round() as i32,
            );
            Some((key, pair))
        })
        .collect::<Vec<_>>();
    binned.par_sort_unstable();
    let mut votes = binned
        .chunk_by(|left, right| left.0 == right.0)
        .filter(|run| run.len() >= options.minimum_matches)
        .map(|run| {
            let mut vote = TranslationVote::default();
            for &(_, pair) in run {
                let (translation_x, translation_y) = translation(pair);
                vote.count += 1;
                vote.sum_x += translation_x;
                vote.sum_y += translation_y;
            }
            (run[0].0, vote)
        })
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

/// Up to `max_stars` of an image's brightest stars, as registration
/// detects them in an image binned at `scale`.
pub(crate) fn detect_stars(
    image: &LinearImage,
    options: &RegistrationOptions,
    max_stars: usize,
    scale: Scale,
) -> Vec<DetectedStar> {
    detect(image, options, max_stars, scale)
}

fn detect(
    image: &LinearImage,
    options: &RegistrationOptions,
    max_stars: usize,
    scale: Scale,
) -> Vec<DetectedStar> {
    let config = scale.detection(DetectConfig {
        backend: DetectBackend::F32,
        sigma: options.detection_sigma,
        max_stars,
        ..DetectConfig::default()
    });
    #[cfg(feature = "gpu")]
    if let Some(stars) = crate::gpu::detect_stars(image, &config) {
        return stars;
    }
    let mut luma = image.luminance();
    normalize_for_detection(&mut luma);
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

    /// A frame large enough for the GPU: sky with a gradient and noise,
    /// 600 stars from faint to near saturation, some touching, a trail and
    /// a block of NaN; `channels` interleaved, each colour its own weight.
    #[cfg(feature = "gpu")]
    fn detection_field(width: usize, height: usize, channels: usize) -> LinearImage {
        let mut state = 0x2545_f491_4f6c_dd1d_u64 ^ channels as u64;
        let mut uniform = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        let stars: Vec<(f64, f64, f64, f64)> = (0..600)
            .map(|_| {
                (
                    uniform() * width as f64,
                    uniform() * height as f64,
                    10.0_f64.powf(1.3 + uniform() * 3.0),
                    0.8 + uniform() * 1.6,
                )
            })
            .collect();
        let mut data = vec![0.0_f32; width * height * channels];
        for (pixel, samples) in data.chunks_exact_mut(channels).enumerate() {
            let (x, y) = ((pixel % width) as f64, (pixel / width) as f64);
            let mut light = 0.0;
            for &(sx, sy, peak, sigma) in &stars {
                let (dx, dy) = (x - sx, y - sy);
                if dx.abs() < 9.0 && dy.abs() < 9.0 {
                    light += peak * (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
                }
            }
            if (y - 0.3 * x - 100.0).abs() < 1.0 {
                light += 300.0;
            }
            for (channel, sample) in samples.iter_mut().enumerate() {
                let noise = (uniform() + uniform() + uniform() - 1.5) * 12.0;
                let gain = 1.0 - 0.25 * channel as f64;
                *sample = (gain * (500.0 + 0.02 * x + light) + noise).min(60_000.0) as f32;
            }
        }
        for y in 700..712 {
            data[(y * width + 200) * channels..(y * width + 230) * channels].fill(f32::NAN);
        }
        LinearImage::new(width, height, channels, data).unwrap()
    }

    /// The CPU's detection, as `detect` makes it without a GPU.
    #[cfg(feature = "gpu")]
    fn detected_on_the_cpu(image: &LinearImage, config: &DetectConfig) -> Vec<DetectedStar> {
        let mut luma = image.luminance();
        normalize_for_detection(&mut luma);
        seiza::detect_stars_luma_f32(&luma, image.width as u32, image.height as u32, config)
    }

    /// The GPU detects the CPU's stars: bit for bit where normalizing the
    /// frame changes nothing, as on a frame already spanning 0 to 1; and,
    /// where it rounds in f32 against the CPU's f64, the same stars within a
    /// thousandth of a pixel but where a pixel on the threshold falls the
    /// other way.
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_detection_matches_the_cpu() {
        let Some(adapter) = crate::gpu::adapter_description() else {
            eprintln!("no GPU: GPU detection not compared");
            return;
        };
        eprintln!("comparing detection on {adapter}");
        let config = DetectConfig {
            backend: DetectBackend::F32,
            sigma: RegistrationOptions::default().detection_sigma,
            max_stars: 500,
            ..DetectConfig::default()
        };
        // Spanning exactly 0 to 1, so normalizing changes nothing.
        let mut unit = detection_field(1100, 1000, 1);
        let (low, high) = unit
            .data
            .iter()
            .filter(|value| value.is_finite())
            .fold((f32::MAX, f32::MIN), |(low, high), &v| {
                (low.min(v), high.max(v))
            });
        for value in &mut unit.data {
            *value = (*value - low) / (high - low);
        }
        unit.data[0] = 0.0;
        unit.data[1] = 1.0;
        let cpu = detected_on_the_cpu(&unit, &config);
        let gpu = crate::gpu::detect_stars(&unit, &config).expect("the GPU detects");
        assert!(cpu.len() > 300, "{} stars", cpu.len());
        assert_eq!(cpu.len(), gpu.len());
        for (a, b) in cpu.iter().zip(&gpu) {
            assert_eq!(
                (
                    a.x.to_bits(),
                    a.y.to_bits(),
                    a.flux.to_bits(),
                    a.peak.to_bits(),
                    a.area
                ),
                (
                    b.x.to_bits(),
                    b.y.to_bits(),
                    b.flux.to_bits(),
                    b.peak.to_bits(),
                    b.area
                ),
            );
        }
        for channels in [1, 3] {
            let image = detection_field(1101, 1003, channels);
            let cpu = detected_on_the_cpu(&image, &config);
            let gpu = crate::gpu::detect_stars(&image, &config).expect("the GPU detects");
            let (mut same, mut close, mut worst) = (0, 0, 0.0_f64);
            for star in &cpu {
                let nearest = gpu
                    .iter()
                    .map(|other| (other.x - star.x).hypot(other.y - star.y))
                    .fold(f64::INFINITY, f64::min);
                same += usize::from(nearest == 0.0);
                if nearest < 1e-3 {
                    close += 1;
                    worst = worst.max(nearest);
                }
            }
            eprintln!(
                "{channels} channel(s): {} CPU stars, {} GPU, {same} identical, {close} within \
                 a thousandth of a pixel (at most {worst:.1e})",
                cpu.len(),
                gpu.len()
            );
            assert!(cpu.len() > 300);
            assert!(cpu.len().abs_diff(gpu.len()) * 100 <= cpu.len());
            assert!(close * 100 >= cpu.len() * 98);
        }
    }

    fn star(x: f64, y: f64) -> DetectedStar {
        DetectedStar {
            x,
            y,
            flux: 1.0,
            peak: 1.0,
            area: 3,
        }
    }

    /// A 97x61 frame, a lens with `k1` and `k2` for it, and a quadratic
    /// warp through that lens that bends a few pixels from `transform`.
    fn lens_frame(
        transform: SimilarityTransform,
        k1: f64,
        k2: f64,
    ) -> (LinearImage, PolynomialWarp) {
        let (width, height) = (97, 61);
        let source = LinearImage::new(
            width,
            height,
            1,
            (0..width * height)
                .map(|index| 1.0 + (index % 13) as f32 * 0.25 + (index / width) as f32 * 0.01)
                .collect(),
        )
        .unwrap();
        let lens: crate::lens::LensDistortion = serde_json::from_value(serde_json::json!({
            "width": width,
            "height": height,
            "center_x": 51.5,
            "center_y": 27.0,
            "radius": 0.5 * (width as f64).hypot(height as f64),
            "k1": k1,
            "k2": k2,
        }))
        .unwrap();
        let inverse = transform.inverse_map();
        let pairs = (0..20)
            .flat_map(|row| (0..20).map(move |column| (column as f64 * 20.0, row as f64 * 15.0)))
            .map(|(x, y)| {
                let (u, v) = inverse(x, y);
                ((x, y), (u + 2.0e-4 * u * v, v - 3.0e-4 * u * u))
            })
            .collect::<Vec<_>>();
        let warp = PolynomialWarp::fit(2, 400, 300, &pairs)
            .unwrap()
            .with_distortion(lens);
        (source, warp)
    }

    /// Whether resampling `geometry` over `region` alone gives, there, what
    /// resampling the whole 400x300 grid gives, and `NaN` beyond it; and
    /// whether the whole grid has finite samples beyond the region.
    fn compare_footprint(
        source: &LinearImage,
        geometry: FrameGeometry<'_>,
        region: ReferenceRegion,
        sampling: Sampling,
    ) -> (bool, bool) {
        let (width, height) = (400, 300);
        let whole = resample_region_geometry(
            source,
            width,
            height,
            ReferenceRegion {
                x: 0,
                y: 0,
                width,
                height,
            },
            geometry,
            sampling,
        )
        .unwrap();
        let bounded =
            resample_region_geometry(source, width, height, region, geometry, sampling).unwrap();
        let placed = crate::image::Placed::on(&bounded, region, &whole);
        let (mut same, mut beyond) = (true, false);
        for (index, &sample) in whole.data.iter().enumerate() {
            let (x, y) = (index % width, index / width);
            let inside = placed.reaches(x, y, 1, 1);
            if inside {
                same &= sample.to_bits() == placed.sample(index).to_bits();
            } else {
                beyond |= sample.is_finite();
            }
        }
        (same, beyond)
    }

    /// Resampling a frame over its footprint region gives every sample the
    /// whole grid gives it, by similarity or by a quadratic warp through a
    /// lens, at any rotation, partly or wholly off the grid.
    #[test]
    fn footprint_holds_every_sample_a_frame_gives_the_grid() {
        let similarity =
            |rotation_radians: f64, translation_x: f64, translation_y: f64| SimilarityTransform {
                scale: 1.01,
                rotation_radians,
                translation_x,
                translation_y,
            };
        for (transform, on_grid) in [
            (similarity(0.0, 150.0, 120.0), true),
            (similarity(0.4, 180.0, 90.0), true),
            (similarity(std::f64::consts::PI, 260.0, 190.0), true),
            (similarity(-0.2, -40.0, 270.0), true),
            (similarity(0.1, 380.0, -30.0), true),
            (similarity(0.0, -500.0, 100.0), false),
        ] {
            let (source, warp) = lens_frame(transform, -0.03, 0.002);
            for geometry in [
                FrameGeometry::Similarity(transform),
                FrameGeometry::Warp(&warp),
            ] {
                let region = footprint_region(97, 61, 400, 300, geometry, transform);
                let small = region.width * region.height;
                if on_grid {
                    assert!(small < 400 * 300 / 4, "{transform:?}: {region:?}");
                } else {
                    assert_eq!(region.width, 1, "{transform:?}: {region:?}");
                }
                for sampling in [Sampling::Bilinear, Sampling::Lanczos3] {
                    let (same, beyond) = compare_footprint(&source, geometry, region, sampling);
                    assert!(same && !beyond, "{transform:?}: {region:?}");
                }
            }
        }
    }

    /// A warp Newton's method cannot carry back, here one that sends every
    /// pixel to one point of the frame, keeps the whole grid, which is then
    /// its footprint.
    #[test]
    fn footprint_of_a_warp_that_cannot_be_inverted_is_the_whole_grid() {
        let transform = SimilarityTransform {
            translation_x: 150.0,
            translation_y: 120.0,
            ..SimilarityTransform::IDENTITY
        };
        let (source, _) = lens_frame(transform, 0.0, 0.0);
        let warp = PolynomialWarp {
            order: 1,
            center_x: 0.0,
            center_y: 0.0,
            scale: 1.0,
            x: vec![5.0, 0.0, 0.0],
            y: vec![5.0, 0.0, 0.0],
            distortion: None,
        };
        let geometry = FrameGeometry::Warp(&warp);
        let region = footprint_region(97, 61, 400, 300, geometry, transform);
        assert_eq!(
            (region.x, region.y, region.width, region.height),
            (0, 0, 400, 300)
        );
        assert_eq!(
            compare_footprint(&source, geometry, region, Sampling::Bilinear),
            (true, false)
        );
    }

    /// Far past the frame, where it was never measured, a strong barrel
    /// distortion folds back onto the frame, so the whole grid samples the
    /// frame a second time, in a ring around it. The footprint is the frame
    /// itself, and leaves the ring out.
    #[test]
    fn footprint_leaves_out_where_a_lens_folds_back_far_from_the_frame() {
        let transform = SimilarityTransform {
            translation_x: 150.0,
            translation_y: 120.0,
            ..SimilarityTransform::IDENTITY
        };
        let (source, warp) = lens_frame(transform, -0.1, 0.0);
        let geometry = FrameGeometry::Warp(&warp);
        let region = footprint_region(97, 61, 400, 300, geometry, transform);
        // Wider than the frame, since barrel distortion draws its edges in.
        assert!(region.width < 130 && region.height < 90, "{region:?}");
        assert_eq!(
            compare_footprint(&source, geometry, region, Sampling::Bilinear),
            (true, true)
        );
    }

    #[test]
    fn geometric_seed_pairs_stars_beyond_the_similarity_search_radius() {
        let mut state = 721_u64;
        let mut random = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as f64 / (1_u64 << 31) as f64
        };
        let reference = (0..1200)
            .map(|_| star(30.0 + 900.0 * random(), 30.0 + 900.0 * random()))
            .collect::<Vec<_>>();
        for order in [2, 3] {
            let geometry = |x: f64, y: f64| {
                let cubic = if order == 3 { 1.0e-8 * x * x * y } else { 0.0 };
                (x + 0.02 * x + 0.00003 * x * y + cubic, y + 0.01 * x)
            };
            let source = reference
                .iter()
                .map(|s| {
                    let (x, y) = geometry(s.x, s.y);
                    star(x + 0.3, y - 0.2)
                })
                .collect::<Vec<_>>();
            let pairs = reference
                .iter()
                .map(|s| ((s.x, s.y), geometry(s.x, s.y)))
                .collect::<Vec<_>>();
            let seed = PolynomialWarp::fit(order, 1000, 1000, &pairs).unwrap();
            for model in [
                RegistrationModel::Similarity,
                RegistrationModel::Affine,
                RegistrationModel::Quadratic,
            ] {
                let registrar = Registrar::from_stars(
                    1000,
                    1000,
                    reference.clone(),
                    RegistrationOptions {
                        maximum_stars: 2000,
                        model,
                        ..RegistrationOptions::default()
                    },
                    Scale::NATIVE,
                )
                .unwrap();
                let old = registrar.warp_stars(SimilarityTransform::IDENTITY, &source);
                let (warp, count, rms) = registrar
                    .warp_stars_seeded(&seed, &source)
                    .unwrap_or_else(|| panic!("{model:?} discarded order-{order} geometry"));
                assert_eq!(warp.order(), order, "{model:?}");
                assert!(
                    count > 1100 && rms < 1e-8,
                    "{model:?}, order {order}: {count} pairs at {rms}"
                );
                assert!(old.is_none_or(|(_, n, r)| n < count / 2 || r > 0.25));
                for (x, y) in [(1.0, 2.0), (992.0, 998.0), (451.2, 179.3)] {
                    let expected = geometry(x, y);
                    let actual = warp.apply(x, y);
                    assert!(
                        (actual.0 - expected.0 - 0.3).hypot(actual.1 - expected.1 + 0.2) < 1e-8,
                        "{model:?}, order {order} at ({x}, {y})"
                    );
                }
            }
        }
    }

    #[test]
    fn polynomial_source_composition_preserves_every_order_and_rejects_a_lens() {
        let transform = SimilarityTransform {
            scale: 1.03,
            rotation_radians: 0.13,
            translation_x: 7.2,
            translation_y: -4.1,
        };
        for order in 1..=3 {
            let pairs = (0..8)
                .flat_map(|row| {
                    (0..8).map(move |column| {
                        let (x, y) = (column as f64 * 80.0, row as f64 * 80.0);
                        let quadratic = if order >= 2 { 0.00003 * x * y } else { 0.0 };
                        let cubic = if order == 3 { 1e-8 * x * x * y } else { 0.0 };
                        ((x, y), (x + quadratic + cubic, y + 0.02 * x))
                    })
                })
                .collect::<Vec<_>>();
            let warp = PolynomialWarp::fit(order, 640, 640, &pairs)
                .unwrap()
                .translated(1800.0, 2700.0);
            let composed = warp.source_transformed(transform).unwrap();
            assert_eq!(composed.order(), order);
            for (x, y) in [(1801.2, 2703.4), (2438.1, 3337.3), (2017.4, 3012.8)] {
                let (u, v) = warp.apply(x, y);
                let expected = transform.apply(u, v);
                let actual = composed.apply(x, y);
                assert!((actual.0 - expected.0).hypot(actual.1 - expected.1) < 1e-10);
            }
            assert!(
                warp.with_distortion(crate::lens::LensDistortion::none(640, 640))
                    .source_transformed(transform)
                    .is_err()
            );
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
    fn lens_warps_cannot_silently_lose_distortion_in_binary_serialization() {
        let plain = PolynomialWarp::from_similarity(shift(3.0, 5.0));
        let bytes = postcard::to_stdvec(&plain).unwrap();
        assert_eq!(
            postcard::from_bytes::<PolynomialWarp>(&bytes).unwrap(),
            plain
        );
        let lens = crate::lens::LensDistortion::none(160, 128);
        let warped = plain.clone().with_distortion(lens);
        assert!(postcard::to_stdvec(&warped).is_err());
        assert_eq!(
            serde_json::from_str::<PolynomialWarp>(&serde_json::to_string(&warped).unwrap())
                .unwrap(),
            warped,
        );
        assert!(warped.clone().for_context(None).is_err());
        assert_eq!(
            postcard::to_stdvec(&warped.for_context(Some(lens)).unwrap()).unwrap(),
            bytes
        );
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

    /// A faint calibrated sky: Gaussian noise of `sigma` about `sky`, as on
    /// the Sh2 140 OIII lights (34 ADU of sky, 12.8 ADU of noise), plus
    /// `extra` at each pixel, from the noise sequence `seed` names.
    fn noisy_sky(
        width: usize,
        height: usize,
        sky: f64,
        sigma: f64,
        seed: u64,
        extra: impl Fn(f64, f64) -> f64,
    ) -> LinearImage {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ seed.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        let mut uniform = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
        };
        let data = (0..width * height)
            .map(|index| {
                // Box-Muller.
                let gaussian =
                    (-2.0 * uniform().ln()).sqrt() * (2.0 * std::f64::consts::PI * uniform()).cos();
                let (x, y) = ((index % width) as f64, (index / width) as f64);
                (sky + sigma * gaussian + extra(x, y)) as f32
            })
            .collect();
        LinearImage::new(width, height, 1, data).unwrap()
    }

    /// A frame rotated 0.3 degrees against the reference: the sub-pixel
    /// phase it is sampled at sweeps a whole pixel every 1 / sin(0.3
    /// degrees), about 190 pixels, so anything that depends on the phase
    /// draws bands that far apart across a stack.
    const ROTATED: SimilarityTransform = SimilarityTransform {
        scale: 1.0,
        rotation_radians: 0.3 * std::f64::consts::PI / 180.0,
        translation_x: 0.37,
        translation_y: -0.21,
    };

    /// The mean of `channel` of `resampled`, a frame carried through
    /// `transform` onto a grid its own size, where both sub-pixel phases lie
    /// within a tenth of a pixel of a sample, and where both lie within a
    /// tenth of the midpoint.
    fn phase_means(
        resampled: &LinearImage,
        transform: SimilarityTransform,
        channel: usize,
    ) -> (f64, f64) {
        let inverse = transform.inverse_map();
        let (width, height) = (resampled.width, resampled.height);
        let (mut whole, mut half) = ((0.0, 0), (0.0, 0));
        for y in 0..height {
            for x in 0..width {
                let (source_x, source_y) = inverse(x as f64, y as f64);
                let inside = |value: f64, size: usize| value >= 3.0 && value < size as f64 - 4.0;
                if !inside(source_x, width) || !inside(source_y, height) {
                    continue;
                }
                let off_sample = |value: f64| {
                    let phase = value - value.floor();
                    phase.min(1.0 - phase)
                };
                let (phase_x, phase_y) = (off_sample(source_x), off_sample(source_y));
                let value =
                    f64::from(resampled.data[(y * width + x) * resampled.channels + channel]);
                if phase_x < 0.1 && phase_y < 0.1 {
                    whole = (whole.0 + value, whole.1 + 1);
                } else if phase_x > 0.4 && phase_y > 0.4 {
                    half = (half.0 + value, half.1 + 1);
                }
            }
        }
        assert!(whole.1 > 5000 && half.1 > 5000, "{whole:?} {half:?}");
        (whole.0 / whole.1 as f64, half.0 / half.1 as f64)
    }

    #[test]
    fn lanczos_leaves_a_faint_noisy_sky_unbiased_at_every_subpixel_phase() {
        // Clamping the noise of a sky only 2.7 sigma above zero fired when
        // the negative-lobe samples happened to be high, lifting the sky
        // about 0.1 sigma at half-pixel phases and leaving it at whole ones.
        let (sky, sigma) = (34.0, 12.8);
        let source = noisy_sky(512, 512, sky, sigma, 0, |_, _| 0.0);
        let (whole, half) = phase_means(&lanczos(&source, ROTATED), ROTATED, 0);
        // Each mean is good to about 0.01 sigma; the clamped noise put
        // 0.1 sigma between them.
        assert!(
            (half - whole).abs() < 0.04 * sigma,
            "sky {whole:.3} at whole-pixel phases against {half:.3} at half-pixel ones"
        );
        assert!((whole - sky).abs() < 0.04 * sigma, "{whole}");
    }

    #[test]
    fn lanczos_leaves_a_demosaiced_faint_sky_unbiased_at_every_subpixel_phase() {
        // A demosaic correlates neighbouring samples, so noise read from
        // neighbours comes out low: under a bilinear demosaic red and blue
        // read none at all, and the gate let the clamp loose on their noise
        // again. The gate reads the noise of each colour's own photosites.
        let (sky, sigma) = (34.0, 12.8);
        let layout = crate::BayerLayout {
            pattern: seiza_fits::BayerPattern::Rggb,
            x_offset: 0,
            y_offset: 0,
        };
        for demosaic in [
            crate::Demosaic::Bilinear,
            crate::Demosaic::Mhc,
            crate::Demosaic::Vng,
        ] {
            let source = noisy_sky(512, 512, sky, sigma, 1, |_, _| 0.0)
                .debayer_with(layout, demosaic)
                .unwrap();
            let noise = crate::snr::photosite_noise(&source).unwrap();
            for (channel, &noise) in noise.iter().enumerate() {
                assert!(
                    (f64::from(noise) / sigma - 1.0).abs() < 0.1,
                    "{demosaic:?} channel {channel} read noise {noise}"
                );
            }
            let resampled = lanczos(&source, ROTATED);
            for channel in 0..3 {
                let (whole, half) = phase_means(&resampled, ROTATED, channel);
                assert!(
                    (half - whole).abs() < 0.04 * sigma,
                    "{demosaic:?} channel {channel}: sky {whole:.3} at whole-pixel phases \
                     against {half:.3} at half-pixel ones"
                );
            }
        }
    }

    #[test]
    fn photosite_noise_reads_a_plain_frame_as_frame_noise_does() {
        let sigma = 12.8;
        let source = noisy_sky(256, 256, 34.0, sigma, 2, |x, y| 0.05 * x + 0.02 * y);
        let photosite = crate::snr::photosite_noise(&source).unwrap()[0];
        let neighbour = crate::frame_noise(&source).unwrap()[0];
        assert!(
            (f64::from(photosite) / sigma - 1.0).abs() < 0.08,
            "{photosite}"
        );
        assert!(
            (photosite / neighbour - 1.0).abs() < 0.08,
            "{photosite} {neighbour}"
        );
        assert_eq!(
            crate::snr::photosite_noise(&noisy_sky(40, 40, 34.0, sigma, 2, |_, _| 0.0)),
            None
        );
    }

    #[test]
    fn lanczos_still_clamps_a_bright_star_on_a_faint_noisy_sky() {
        // The noise gate must not let a bright star ring: unclamped, its
        // negative lobes would dig far below the sky.
        let (width, height, sky, sigma) = (96, 96, 34.0, 12.8);
        let source = noisy_sky(width, height, sky, sigma, 3, |x, y| {
            let (dx, dy) = (x - 47.5, y - 48.0);
            20_000.0 * (-(dx * dx + dy * dy) / (2.0 * 0.8 * 0.8)).exp()
        });
        assert!(lanczos_clamp_spans(&source)[0] > 0.0);
        let sharp = lanczos(&source, shift(0.5, 0.5));
        for y in 42..55 {
            for x in 42..55 {
                let value = sharp.data[y * width + x];
                assert!(
                    value > (sky - 5.0 * sigma) as f32,
                    "({x}, {y}) rang to {value}"
                );
            }
        }
        assert!(sharp.data[48 * width + 48] > 5_000.0);
    }

    #[test]
    fn lanczos_clamps_a_star_at_twenty_times_the_noise() {
        // Pins the gate's threshold: a 1.6 px FWHM star twenty times the
        // noise above a faint sky must still clamp where it falls under a
        // negative lobe. The 20 000 ADU star above would clamp under almost
        // any threshold, and the six-sigma one passes clamped or not.
        let (size, sky, sigma) = (96, 34.0, 12.8);
        let (peak, psf_sigma) = (20.0 * sigma, 1.6 / 2.3548);
        let source = noisy_sky(size, size, sky, sigma, 4, |x, y| {
            let (x, y) = (x - 48.0, y - 48.0);
            peak * (-(x * x + y * y) / (2.0 * psf_sigma * psf_sigma)).exp()
        });
        let span = lanczos_clamp_spans(&source)[0];
        assert!(span > 0.0);
        let weights = lanczos3_weights(0.5);
        let row = &source.data[48 * size..49 * size];
        // A pass from sample x0 reads x0 - 2 ..= x0 + 3, its negative lobes
        // at x0 - 1 and x0 + 2: the star's column under one or the other.
        for x0 in [49, 46] {
            let taps: [f32; 6] = std::array::from_fn(|tap| row[x0 - 2 + tap]);
            let positive = weights[0] * taps[0]
                + weights[2] * taps[2]
                + weights[3] * taps[3]
                + weights[5] * taps[5];
            let positive_only = positive / (weights[0] + weights[2] + weights[3] + weights[5]);
            let full = positive + weights[1] * taps[1] + weights[4] * taps[4];
            let value = clamped_lanczos(&weights, &taps, span);
            assert!(
                (value - positive_only).abs() < 1.0e-3 && (full - positive_only).abs() > 10.0,
                "pass from {x0} gave {value}: clamped {positive_only}, full {full}, \
                 taps {taps:?}, gate {span}"
            );
        }
    }

    #[test]
    fn a_star_too_faint_to_clamp_rings_little_even_stacked() {
        // A star peaking at six times the noise, which the gate mostly
        // leaves unclamped, in 128 dithered frames. Its ringing is the same
        // in every frame while the noise averages down, so check the stack:
        // around a 1.6 px FWHM star, sharper than most stacks see, no ring
        // about it dips under the sky by 1% of the peak. It measured 0.1%,
        // against 0.2% with no clamping at all.
        let (size, sky, sigma, frames) = (96, 34.0, 12.8, 128);
        let (peak, psf_sigma) = (6.0 * sigma, 1.6 / 2.3548);
        let mut state = 0x51_u64;
        let mut uniform = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        let mut stack = vec![0.0_f64; size * size];
        for frame in 0..frames {
            let (dx, dy) = (uniform(), uniform());
            let source = noisy_sky(size, size, sky, sigma, 100 + frame, |x, y| {
                let (x, y) = (x - 48.0 - dx, y - 48.0 - dy);
                peak * (-(x * x + y * y) / (2.0 * psf_sigma * psf_sigma)).exp()
            });
            assert!(lanczos_clamp_spans(&source)[0] > 0.0);
            let aligned = lanczos(&source, shift(-dx, -dy));
            for (total, &value) in stack.iter_mut().zip(&aligned.data) {
                *total += f64::from(value) / frames as f64;
            }
        }
        let ring = |inner: f64, outer: f64| {
            let (mut total, mut count) = (0.0, 0);
            for y in 8..size - 8 {
                for x in 8..size - 8 {
                    let radius = ((x as f64 - 48.0).powi(2) + (y as f64 - 48.0).powi(2)).sqrt();
                    if radius >= inner && radius < outer {
                        total += stack[y * size + x];
                        count += 1;
                    }
                }
            }
            total / count as f64
        };
        let background = ring(14.0, 30.0);
        let deepest = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
            .into_iter()
            .map(|inner| ring(inner, inner + 0.5) - background)
            .fold(f64::INFINITY, f64::min);
        assert!(
            deepest > -0.01 * peak,
            "the stacked ring dips {deepest:.3} ADU under the sky, against a {peak} ADU peak"
        );
        assert!((ring(0.0, 0.5) - background) > 0.8 * peak);
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

    /// A superpixel frame's red and blue sit half a sensor pixel off the
    /// cell's centre, and resampling from those sites gives every channel
    /// the same sky: on a plane, which bilinear interpolation follows
    /// exactly, the three channels agree everywhere both reach, through a
    /// shift and a half turn alike.
    #[test]
    fn resampling_from_channel_sites_lines_colours_up() {
        let rggb = crate::BayerLayout {
            pattern: seiza_fits::BayerPattern::Rggb,
            x_offset: 0,
            y_offset: 0,
        };
        let sites = ChannelSites::superpixel(rggb, 2);
        assert_eq!(sites.0, [(-0.25, -0.25), (0.0, 0.0), (0.25, 0.25)]);
        let gbrg = crate::BayerLayout {
            pattern: seiza_fits::BayerPattern::Gbrg,
            x_offset: 0,
            y_offset: 0,
        };
        assert_eq!(
            ChannelSites::superpixel(gbrg, 4).0,
            [(-0.125, 0.125), (0.0, 0.0), (0.125, -0.125)]
        );
        // A plane on the sensor, superpixelled: each colour holds the plane
        // where its photosite lies.
        let plane = |x: f64, y: f64| (100.0 + 3.0 * x - 2.0 * y) as f32;
        let (width, height) = (64, 48);
        let mosaic = (0..width * height)
            .map(|index| plane((index % width) as f64, (index / width) as f64))
            .collect();
        let frame = LinearImage::new(width, height, 1, mosaic)
            .unwrap()
            .debayer_with(rggb, crate::Demosaic::Superpixel)
            .unwrap();
        let (w, h) = (frame.width, frame.height);
        let region = ReferenceRegion {
            x: 0,
            y: 0,
            width: w,
            height: h,
        };
        for transform in [
            SimilarityTransform::IDENTITY,
            SimilarityTransform {
                translation_x: 1.3,
                translation_y: -0.6,
                ..SimilarityTransform::IDENTITY
            },
            SimilarityTransform {
                scale: 1.0,
                rotation_radians: std::f64::consts::PI,
                translation_x: (w - 1) as f64,
                translation_y: (h - 1) as f64,
            },
        ] {
            let sited = resample_region_sited(
                &frame,
                w,
                h,
                region,
                FrameGeometry::Similarity(transform),
                Sampling::Bilinear,
                Some(sites),
            )
            .unwrap();
            let unsited = resample_region_sited(
                &frame,
                w,
                h,
                region,
                FrameGeometry::Similarity(transform),
                Sampling::Bilinear,
                None,
            )
            .unwrap();
            let mut compared = 0;
            for (pixel, unsited) in sited
                .data
                .as_chunks::<3>()
                .0
                .iter()
                .zip(unsited.data.as_chunks::<3>().0.iter())
            {
                if pixel.iter().all(|value| value.is_finite()) {
                    assert!((pixel[0] - pixel[1]).abs() < 1e-3, "{pixel:?}");
                    assert!((pixel[2] - pixel[1]).abs() < 1e-3, "{pixel:?}");
                    // Green sits at the centre, where it was already.
                    assert_eq!(pixel[1], unsited[1]);
                    compared += 1;
                }
            }
            assert!(compared > w * h / 2, "{transform:?}");
            // Read from the pixel centres, red and blue differ by a sensor
            // pixel of the plane's slope in each axis.
            let first = &unsited.data[(h / 2 * w + w / 2) * 3..][..3];
            assert!((first[2] - first[0]).abs() > 0.5, "{first:?}");
        }
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
    /// as vector lanes: each channel filtered on its own, gated by its own
    /// entry of `clamp_spans`.
    fn resample_per_channel(
        source: &LinearImage,
        region: ReferenceRegion,
        inverse: impl Fn(f64, f64) -> (f64, f64),
        sampling: Sampling,
        clamp_spans: &[f32],
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
                            *value = clamped_lanczos(&weights_x, &taps, clamp_spans[channel]);
                        }
                        let value = clamped_lanczos(&weights_y, &rows, clamp_spans[channel]);
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
            distortion: None,
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
        // spikes that trip the clamping, and non-finite samples. The largest
        // is big enough to measure its noise, so the noise gate holds the
        // clamp back on the plain noise and lets it act on the spikes, with
        // a different span for each channel.
        for (width, height) in [(1, 1), (3, 3), (6, 6), (7, 5), (9, 8), (31, 17), (96, 80)] {
            for channels in [1, 3] {
                let data = (0..width * height * channels)
                    .enumerate()
                    .map(|(index, _)| match next() % 41 {
                        0 => f32::NAN,
                        1 => f32::INFINITY,
                        2 => -0.0,
                        3..=6 => 50.0 + (next() % 1000) as f32,
                        // A noise level of its own for each channel.
                        _ => {
                            (next() >> 40) as f32 / (1 << 24) as f32 * (1 + index % channels) as f32
                        }
                    })
                    .collect::<Vec<_>>();
                let source = LinearImage::new(width, height, channels, data).unwrap();
                let clamp_spans = lanczos_clamp_spans(&source);
                if width >= 96 {
                    assert!(
                        clamp_spans.iter().all(|&span| span > 0.0)
                            && clamp_spans.windows(2).all(|pair| pair[0] < pair[1]),
                        "{clamp_spans:?}"
                    );
                    // The gate must decide some samples, or this checks nothing.
                    let ungated = resample_per_channel(
                        &source,
                        ReferenceRegion {
                            x: 0,
                            y: 0,
                            width,
                            height,
                        },
                        shift(-0.37, 0.61).inverse_map(),
                        Sampling::Lanczos3,
                        &vec![0.0; channels],
                    );
                    let gated = resample_per_channel(
                        &source,
                        ReferenceRegion {
                            x: 0,
                            y: 0,
                            width,
                            height,
                        },
                        shift(-0.37, 0.61).inverse_map(),
                        Sampling::Lanczos3,
                        &clamp_spans,
                    );
                    assert!(
                        ungated
                            .iter()
                            .zip(&gated)
                            .any(|(a, b)| a.to_bits() != b.to_bits()),
                        "the noise gate changed nothing"
                    );
                }
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
                                &clamp_spans,
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
                            &clamp_spans,
                        );
                        check("warp", got, expected);
                    }
                }
            }
        }
    }

    /// The coverage told from geometry is the resampled image's finite
    /// samples exactly: by similarity and by warp, bilinear and Lanczos,
    /// with and without channel sites, mono and colour, at the frame's
    /// edges as well as within it; and there is none to tell for a source
    /// with a missing sample or for photosite sampling.
    #[test]
    fn resampled_coverage_is_the_resampled_finite_samples() {
        let (width, height) = (61, 47);
        let image = |channels: usize| {
            let data = (0..width * height * channels)
                .map(|index| 100.0 + ((index * 37) % 101) as f32)
                .collect::<Vec<_>>();
            LinearImage::new(width, height, channels, data).unwrap()
        };
        let transform = SimilarityTransform {
            translation_x: 23.375,
            translation_y: 11.125,
            rotation_radians: 0.41,
            scale: 1.03,
        };
        let warp = PolynomialWarp::from_similarity(SimilarityTransform {
            translation_x: -18.25,
            translation_y: -6.5,
            rotation_radians: -0.37,
            scale: 0.97,
        });
        let region = ReferenceRegion {
            x: 3,
            y: 2,
            width: 90,
            height: 80,
        };
        let sites = ChannelSites([(0.5, -0.5), (0.0, 0.0), (-0.5, 0.5)]);
        for channels in [1, 3] {
            let source = image(channels);
            for geometry in [
                FrameGeometry::Similarity(transform),
                FrameGeometry::Warp(&warp),
            ] {
                for sampling in [Sampling::Bilinear, Sampling::Lanczos3] {
                    for sited in [None, Some(sites)] {
                        let resampled = resample_region_sited(
                            &source, 100, 90, region, geometry, sampling, sited,
                        )
                        .unwrap();
                        let covered =
                            resampled_coverage(&source, region, geometry, sampling, sited)
                                .unwrap()
                                .unwrap();
                        let finite = resampled
                            .data
                            .iter()
                            .map(|value| value.is_finite())
                            .collect::<Vec<_>>();
                        assert!(finite.contains(&true) && finite.contains(&false));
                        assert_eq!(covered, finite);
                    }
                }
            }
        }
        let mut gap = image(3);
        gap.data[100] = f32::NAN;
        let geometry = FrameGeometry::Similarity(transform);
        assert!(
            resampled_coverage(&gap, region, geometry, Sampling::Bilinear, None)
                .unwrap()
                .is_none()
        );
        let layout = crate::BayerLayout {
            pattern: seiza_fits::BayerPattern::Rggb,
            x_offset: 0,
            y_offset: 0,
        };
        assert!(
            resampled_coverage(
                &image(3),
                region,
                geometry,
                Sampling::NearestPhotosite(layout),
                None
            )
            .unwrap()
            .is_none()
        );
    }

    /// The windowed, best-first search finds the same candidates, in the
    /// same order, as scanning every pair of triangles and fitting each,
    /// whether the narrow window holds enough or the whole tolerance is read,
    /// and on a lattice, whose congruent triangles tie, in the scan's order.
    #[test]
    fn triangle_candidates_match_a_scan_of_every_pair() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        let scattered = (0..80)
            .map(|_| star(20.0 + 960.0 * next(), 20.0 + 960.0 * next()))
            .collect::<Vec<_>>();
        let turn = SimilarityTransform {
            translation_x: 31.5,
            translation_y: -17.25,
            rotation_radians: 0.3,
            scale: 1.0,
        };
        let turned = scattered
            .iter()
            .map(|reference| {
                let (x, y) = turn.inverse_apply(reference.x, reference.y);
                star(x + 0.3 * (next() - 0.5), y + 0.3 * (next() - 0.5))
            })
            .collect::<Vec<_>>();
        let lattice = (0..48)
            .map(|index| {
                star(
                    100.0 + 110.0 * (index % 8) as f64,
                    120.0 + 130.0 * (index / 8) as f64,
                )
            })
            .collect::<Vec<_>>();
        let shifted = lattice
            .iter()
            .map(|reference| star(reference.x - 12.0, reference.y + 7.0))
            .collect::<Vec<_>>();
        for ((reference, source), maximum_candidates) in
            [(&scattered, &turned), (&lattice, &shifted)]
                .into_iter()
                .flat_map(|field| [64, 384, 1_000_000].map(|count| (field, count)))
        {
            let options = RegistrationOptions {
                triangle_stars: 30,
                maximum_candidates,
                ..RegistrationOptions::default()
            };
            let registrar =
                Registrar::from_stars(1000, 1000, reference.to_vec(), options, Scale::NATIVE)
                    .unwrap();
            let source_stars = &source[..source.len().min(registrar.options.maximum_stars)];
            let source_triangles = triangles(source_stars, registrar.options.triangle_stars);
            let mut scanned = Vec::new();
            for source_triangle in &source_triangles {
                for reference_triangle in &registrar.reference_triangles {
                    let error = (source_triangle.ratios[0] - reference_triangle.ratios[0]).abs()
                        + (source_triangle.ratios[1] - reference_triangle.ratios[1]).abs();
                    if error > registrar.options.descriptor_tolerance * 2.0 {
                        continue;
                    }
                    if let Some(transform) = transform_from_triangles(
                        source_triangle,
                        reference_triangle,
                        source_stars,
                        &registrar.reference_stars,
                    ) && (transform.scale - 1.0).abs() <= registrar.options.scale_tolerance
                        && transform.displacement_at(500.0, 500.0) <= registrar.maximum_drift_pixels
                    {
                        scanned.push((error, transform));
                    }
                }
            }
            scanned.sort_by(|left, right| left.0.total_cmp(&right.0));
            scanned.truncate(maximum_candidates);
            let expected = scanned
                .into_iter()
                .map(|(_, transform)| transform)
                .collect::<Vec<_>>();
            assert!(!expected.is_empty());
            assert_eq!(
                registrar.triangle_candidates(&source_triangles, source_stars),
                expected,
                "{maximum_candidates} candidates"
            );
        }
    }

    /// Votes summed by sorted runs pick the same translations, with the same
    /// means to the bit, as a map filled in the order the pairs are scanned:
    /// for a shifted copy of a field with stars added and lost, and with a
    /// drift limit that turns some pairs away.
    #[test]
    fn translation_candidates_match_votes_kept_in_a_map() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        let reference = (0..300)
            .map(|_| star(2000.0 * next(), 1500.0 * next()))
            .collect::<Vec<_>>();
        let mut source = Vec::new();
        for kept in reference.iter().take(260) {
            let (x, y) = (kept.x - 37.3 + 0.4 * next(), kept.y + 12.6 + 0.4 * next());
            source.push(star(x, y));
        }
        for _ in 0..40 {
            let (x, y) = (2000.0 * next(), 1500.0 * next());
            source.push(star(x, y));
        }
        let options = RegistrationOptions::default();
        for drift in [5000.0, 600.0] {
            let bin_size = options.match_tolerance_pixels * 2.0;
            let mut map = HashMap::<(i32, i32), TranslationVote>::new();
            for source in &source {
                for reference in &reference {
                    let translation_x = reference.x - source.x;
                    let translation_y = reference.y - source.y;
                    if translation_x.hypot(translation_y) > drift {
                        continue;
                    }
                    let vote = map
                        .entry((
                            (translation_x / bin_size).round() as i32,
                            (translation_y / bin_size).round() as i32,
                        ))
                        .or_default();
                    vote.count += 1;
                    vote.sum_x += translation_x;
                    vote.sum_y += translation_y;
                }
            }
            let mut votes = map
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
            let expected = votes
                .into_iter()
                .map(|(_, vote)| SimilarityTransform {
                    translation_x: vote.sum_x / vote.count as f64,
                    translation_y: vote.sum_y / vote.count as f64,
                    ..SimilarityTransform::IDENTITY
                })
                .collect::<Vec<_>>();
            assert!(expected.len() > 1);
            assert_eq!(
                translation_candidates(&source, &reference, &options, drift),
                expected
            );
        }
    }
}
