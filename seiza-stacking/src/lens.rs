//! A camera's radial lens distortion, measured where the panels of an
//! expanded mosaic overlap.
//!
//! One optical train images every panel, so one distortion bends them all,
//! and it bends opposite edges of two neighboring panels opposite ways: a
//! similarity fitted through their overlap matches its middle and misses its
//! ends. On a real three-panel mosaic shot through a 528mm astrograph onto a
//! full-frame sensor, similarities left 1.4 to 1.8px RMS between neighbors and
//! paired stars only across the middle third of each overlap. Undistorting
//! both sides by one shared radial model, about the optical axis rather than
//! the sensor's center, paired the whole overlap at 0.15px.
//!
//! Panels also lie on different tangent planes of the sky. Where the focal
//! length is known, each panel's undistorted pixels are placed on the
//! anchor's tangent plane by a rotation of the camera rather than a
//! similarity; on that mosaic, 2.2 degrees apart, the rotation took a further
//! 0.1px from the residual.

use crate::{Error, Result, SimilarityTransform};
use seiza::DetectedStar;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Radial distortion shared by the frames one camera and lens took. A star
/// an undistorted lens would image at `ideal` lies on the sensor at
/// `center + (ideal - center) * (1 + k1 q + k2 q²)`, where `q` is the squared
/// distance of `ideal` from `center` over `radius` squared.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LensDistortion {
    /// The sensor it was measured on.
    width: usize,
    height: usize,
    center_x: f64,
    center_y: f64,
    radius: f64,
    k1: f64,
    k2: f64,
}

impl LensDistortion {
    /// No distortion about the center of a `width` by `height` frame.
    pub(crate) fn none(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            center_x: (width as f64 - 1.0) * 0.5,
            center_y: (height as f64 - 1.0) * 0.5,
            radius: 0.5 * (width as f64).hypot(height as f64),
            k1: 0.0,
            k2: 0.0,
        }
    }

    /// The same distortion on a grid `scale` times as fine, or `1 / b` for
    /// the frame binned by `b`: its centre moves as a pixel centre does,
    /// `scale · x + (scale − 1) / 2`, its radius and frame stretch by
    /// `scale`, and its coefficients, which act on the squared distance over
    /// the squared radius, stand. The frame must divide evenly at a scale
    /// under 1.
    pub(crate) fn rescaled(&self, scale: f64) -> Self {
        let offset = (scale - 1.0) / 2.0;
        Self {
            width: (self.width as f64 * scale).round() as usize,
            height: (self.height as f64 * scale).round() as usize,
            center_x: scale * self.center_x + offset,
            center_y: scale * self.center_y + offset,
            radius: scale * self.radius,
            ..*self
        }
    }

    /// The sensor position of an undistorted frame position.
    pub(crate) fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        let (u, v) = (x - self.center_x, y - self.center_y);
        let factor = self.factor((u * u + v * v) / (self.radius * self.radius));
        (self.center_x + u * factor, self.center_y + v * factor)
    }

    /// The undistorted position of a sensor position: the inverse of
    /// [`Self::distort`], by fixed-point iteration, which converges at once
    /// for distortion of a few percent.
    pub(crate) fn undistort(&self, x: f64, y: f64) -> (f64, f64) {
        let (u, v) = (x - self.center_x, y - self.center_y);
        let (mut a, mut b) = (u, v);
        for _ in 0..30 {
            let factor = self.factor((a * a + b * b) / (self.radius * self.radius));
            let (next_a, next_b) = (u / factor, v / factor);
            let change = (next_a - a).abs() + (next_b - b).abs();
            (a, b) = (next_a, next_b);
            if change < 1.0e-10 {
                break;
            }
        }
        (self.center_x + a, self.center_y + b)
    }

    /// Stars with their positions undistorted.
    pub(crate) fn undistort_stars(&self, stars: &[DetectedStar]) -> Vec<DetectedStar> {
        stars
            .iter()
            .map(|star| {
                let (x, y) = self.undistort(star.x, star.y);
                DetectedStar { x, y, ..*star }
            })
            .collect()
    }

    fn factor(&self, q: f64) -> f64 {
        1.0 + self.k1 * q + self.k2 * q * q
    }

    /// The largest shift distortion makes within `radius` of the center.
    fn largest_shift(&self) -> f64 {
        (0..=64)
            .map(|step| {
                let q = (step as f64 / 64.0).powi(2);
                (self.factor(q) - 1.0).abs() * q.sqrt() * self.radius
            })
            .fold(0.0, f64::max)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        let values = [self.center_x, self.center_y, self.k1, self.k2];
        let sized = self.width > 0
            && self.height > 0
            && (self.radius - 0.5 * (self.width as f64).hypot(self.height as f64)).abs() < 1.0e-6;
        // Within the frame's circle the factor must stay positive, or the
        // distortion folds the image over itself.
        let folds = (0..=64).any(|step| self.factor((step as f64 / 64.0).powi(2)) <= 0.5);
        if values.iter().any(|value| !value.is_finite())
            || !(self.radius.is_finite() && self.radius > 0.0)
            || !sized
            || folds
        {
            return Err(Error::Registration(
                "lens distortion must be finite and must not fold the frame".into(),
            ));
        }
        Ok(())
    }
}

/// The lens distortion an expanded canvas was planned through, as measured
/// where its panels' first frames overlap.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ExpandedLens {
    /// The distortion's axis, in sensor pixels.
    pub center_x: f64,
    pub center_y: f64,
    /// The axis's offset from the sensor's center, in pixels.
    pub decenter_pixels: f64,
    /// Coefficients of `q` and `q²`, where `q` is a position's squared
    /// distance from the axis over the frame's half diagonal squared.
    pub k1: f64,
    pub k2: f64,
    /// The largest shift the distortion makes within the frame's half
    /// diagonal of its axis, in pixels.
    pub largest_shift_pixels: f64,
    /// RMS distance between the stars the panels' first frames pair where
    /// they overlap, through the lens, and fitted without it. `None` for a
    /// canvas restored from a context.
    pub overlap_rms_pixels: Option<f64>,
    pub undistorted_overlap_rms_pixels: Option<f64>,
    /// The focal length, in pixels, by which panels were placed on the
    /// anchor's tangent plane rather than by similarities, when the anchor's
    /// headers gave one.
    pub focal_length_pixels: Option<f64>,
}

impl ExpandedLens {
    pub(crate) fn new(distortion: &LensDistortion, fit: Option<&LensFit>) -> Self {
        let (width, height) = (distortion.width as f64, distortion.height as f64);
        Self {
            center_x: distortion.center_x,
            center_y: distortion.center_y,
            decenter_pixels: (distortion.center_x - (width - 1.0) * 0.5)
                .hypot(distortion.center_y - (height - 1.0) * 0.5),
            k1: distortion.k1,
            k2: distortion.k2,
            largest_shift_pixels: distortion.largest_shift(),
            overlap_rms_pixels: fit.map(|fit| fit.overall_rms),
            undistorted_overlap_rms_pixels: fit.map(|fit| fit.undistorted_rms),
            focal_length_pixels: fit.and_then(|fit| fit.projection).map(|p| p.focal),
        }
    }
}

/// Where a frame's undistorted pixels land on the anchor's undistorted grid.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Pose {
    Similarity(SimilarityTransform),
    /// The camera turned by `rotation` from the anchor's view, imaging with
    /// `focal` pixels of its own to the anchor's [`Projection::focal`].
    Rotation {
        rotation: [[f64; 3]; 3],
        focal: f64,
    },
}

/// The tangent-plane geometry shared by a camera's frames: its focal length
/// in pixels and the center every frame's view is taken about.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Projection {
    pub focal: f64,
    pub center_x: f64,
    pub center_y: f64,
}

impl Pose {
    /// The anchor-grid position of an undistorted frame position, or `None`
    /// behind the anchor's view.
    pub(crate) fn apply(
        &self,
        projection: Option<Projection>,
        x: f64,
        y: f64,
    ) -> Option<(f64, f64)> {
        match (*self, projection) {
            (Self::Similarity(transform), _) => Some(transform.apply(x, y)),
            (Self::Rotation { rotation, focal }, Some(p)) => {
                let ray = rotate(&rotation, [x - p.center_x, y - p.center_y, focal]);
                (ray[2] > 0.0).then(|| {
                    (
                        p.center_x + p.focal * ray[0] / ray[2],
                        p.center_y + p.focal * ray[1] / ray[2],
                    )
                })
            }
            (Self::Rotation { .. }, None) => None,
        }
    }

    /// The undistorted frame position of an anchor-grid position.
    #[cfg(test)]
    fn inverse(&self, projection: Option<Projection>, x: f64, y: f64) -> Option<(f64, f64)> {
        match (*self, projection) {
            (Self::Similarity(transform), _) => Some(transform.inverse_apply(x, y)),
            (Self::Rotation { rotation, focal }, Some(p)) => {
                let ray = rotate(
                    &transpose(&rotation),
                    [x - p.center_x, y - p.center_y, p.focal],
                );
                (ray[2] > 0.0).then(|| {
                    (
                        p.center_x + focal * ray[0] / ray[2],
                        p.center_y + focal * ray[1] / ray[2],
                    )
                })
            }
            (Self::Rotation { .. }, None) => None,
        }
    }
}

/// A frame taking part in the estimate: its brightest stars on its own
/// sensor, and a similarity placing them roughly on the anchor's grid.
pub(crate) struct LensFrame<'a> {
    pub stars: &'a [DetectedStar],
    pub placement: SimilarityTransform,
}

/// A distortion and the poses it was fitted with.
#[derive(Clone, Debug)]
pub(crate) struct LensFit {
    pub distortion: LensDistortion,
    pub projection: Option<Projection>,
    /// One per frame; the anchor's is the identity.
    pub poses: Vec<Pose>,
    /// Stars each frame paired with the others.
    pub pairs: Vec<usize>,
    /// Each frame's RMS distance from its partners, in anchor pixels.
    pub rms: Vec<f64>,
    /// RMS over every pair, and with the same pairs fitted undistorted.
    pub overall_rms: f64,
    pub undistorted_rms: f64,
}

impl LensFit {
    /// The same fit on a grid `scale` times as fine, or `1 / b` for frames
    /// binned by `b`, as [`LensDistortion::rescaled`] carries the
    /// distortion. A pixel centre moves to `scale · x + (scale − 1) / 2`;
    /// a similarity pose is carried through that map, and a camera rotation
    /// keeps its rotation with its focal length, like the projection's,
    /// stretched by `scale`. Distances stretch by `scale` too.
    pub(crate) fn rescaled(&self, scale: f64) -> Self {
        let offset = (scale - 1.0) / 2.0;
        Self {
            distortion: self.distortion.rescaled(scale),
            projection: self.projection.map(|projection| Projection {
                focal: projection.focal * scale,
                center_x: scale * projection.center_x + offset,
                center_y: scale * projection.center_y + offset,
            }),
            poses: self
                .poses
                .iter()
                .map(|pose| match *pose {
                    Pose::Similarity(transform) => Pose::Similarity(transform.rescaled(scale)),
                    Pose::Rotation { rotation, focal } => Pose::Rotation {
                        rotation,
                        focal: focal * scale,
                    },
                })
                .collect(),
            pairs: self.pairs.clone(),
            rms: self.rms.iter().map(|rms| rms * scale).collect(),
            overall_rms: self.overall_rms * scale,
            undistorted_rms: self.undistorted_rms * scale,
        }
    }
}

/// Stars each frame must pair with the others for its pose to be trusted.
const MINIMUM_PAIRS: usize = 30;

/// Pairing tolerances, widest first: each fit pairs the stars the previous
/// one brought within reach.
const PAIRING_PIXELS: [f64; 4] = [6.0, 4.0, 3.0, 2.0];

/// The distortion must take at least this fraction off the residual, and at
/// least [`MINIMUM_IMPROVEMENT_PIXELS`], or the frames keep their similarities.
///
/// Star noise alone, with nothing for a lens to explain, leaves the residual
/// within a fraction of a percent of the similarities': over the thousands
/// of pairs the overlaps give, a radial distortion's few terms cannot fit
/// noise. A sharp lens with mild distortion, though, takes only a fifth or so
/// off a residual that star centroids keep near 0.3 px: a Seestar S50 mosaic
/// went from 0.369 to 0.287 px, and its stack, corrected, fitted Gaia at
/// 0.33" rather than 0.62", with stars up to 5% tighter in its corners.
const MAXIMUM_RESIDUAL_RATIO: f64 = 0.9;
const MINIMUM_IMPROVEMENT_PIXELS: f64 = 0.05;

/// Bounds on a plausible distortion, as fractions of the frame's half
/// diagonal: how far it may move a star, and how far its axis may lie from
/// the sensor's center.
const MAXIMUM_SHIFT_FRACTION: f64 = 0.05;
const MAXIMUM_DECENTER_FRACTION: f64 = 0.25;

/// Fit one radial distortion and every frame's pose to the stars where the
/// frames overlap. `frames[0]` is the anchor. `None` unless every other frame
/// pairs enough stars and the distortion explains the residual a similarity
/// leaves.
pub(crate) fn estimate(
    frames: &[LensFrame<'_>],
    width: usize,
    height: usize,
    focal: Option<f64>,
) -> Option<LensFit> {
    if frames.len() < 2 {
        return None;
    }
    let projection = focal
        .filter(|f| f.is_finite() && *f > 0.0)
        .map(|focal| Projection {
            focal,
            center_x: (width as f64 - 1.0) * 0.5,
            center_y: (height as f64 - 1.0) * 0.5,
        });
    let model = Model::new(frames, width, height, projection);
    let mut params = model.initial();
    let lens_fixed = |index: usize| index < LENS_PARAMS;
    // Pair first through the similarities themselves, which registration
    // found where the frames overlap: a camera rotation standing in for one
    // differs from it by several pixels across a frame turned a few degrees
    // off the anchor's axis, enough to pair nothing. They pair only the middle
    // of each overlap; fit the poses alone there before the distortion is
    // let loose.
    let placed = frames
        .iter()
        .map(|frame| {
            frame
                .stars
                .iter()
                .map(|star| frame.placement.apply(star.x, star.y))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut pairs = pair_positions(&placed, PAIRING_PIXELS[0]);
    for _ in 0..2 {
        model.fit(&mut params, &pairs, &lens_fixed);
        pairs = model.clip(&params, pairs);
    }
    for tolerance in PAIRING_PIXELS {
        pairs = model.pair(&params, tolerance);
        for _ in 0..3 {
            model.fit(&mut params, &pairs, &|_| false);
            pairs = model.clip(&params, pairs);
        }
    }
    let (distortion, poses) = model.unpack(&params);
    distortion.validate().ok()?;
    let errors = model.errors(&params, &pairs);
    let overall_rms = rms(&errors);
    let mut baseline = model.initial();
    model.fit(&mut baseline, &pairs, &lens_fixed);
    let undistorted_rms = rms(&model.errors(&baseline, &pairs));

    let mut counts = vec![0; frames.len()];
    let mut squares = vec![0.0; frames.len()];
    for (pair, error) in pairs.iter().zip(&errors) {
        for frame in [pair.0, pair.2] {
            counts[frame] += 1;
            squares[frame] += error * error;
        }
    }
    let decenter = (distortion.center_x - (width as f64 - 1.0) * 0.5)
        .hypot(distortion.center_y - (height as f64 - 1.0) * 0.5);
    let plausible = counts[1..].iter().all(|&count| count >= MINIMUM_PAIRS)
        && overall_rms <= MAXIMUM_RESIDUAL_RATIO * undistorted_rms
        && undistorted_rms - overall_rms >= MINIMUM_IMPROVEMENT_PIXELS
        && distortion.largest_shift() <= MAXIMUM_SHIFT_FRACTION * distortion.radius
        && decenter <= MAXIMUM_DECENTER_FRACTION * distortion.radius;
    plausible.then(|| LensFit {
        distortion,
        projection,
        poses,
        rms: counts
            .iter()
            .zip(&squares)
            .map(|(&count, &square)| (square / count.max(1) as f64).sqrt())
            .collect(),
        pairs: counts,
        overall_rms,
        undistorted_rms,
    })
}

fn rms(errors: &[f64]) -> f64 {
    (errors.iter().map(|e| e * e).sum::<f64>() / errors.len().max(1) as f64).sqrt()
}

/// k1, k2, and the axis's offset from the sensor's center.
const LENS_PARAMS: usize = 4;
const POSE_PARAMS: usize = 4;

/// Two frames' stars that image one star of the sky: frame, star, frame, star.
type Pair = (usize, usize, usize, usize);

struct Model<'a> {
    frames: &'a [LensFrame<'a>],
    width: usize,
    height: usize,
    projection: Option<Projection>,
    /// Each frame's rotation the parameters turn further, from its
    /// similarity; empty without a projection.
    rotations: Vec<[[f64; 3]; 3]>,
}

impl<'a> Model<'a> {
    fn new(
        frames: &'a [LensFrame<'a>],
        width: usize,
        height: usize,
        projection: Option<Projection>,
    ) -> Self {
        let rotations = projection
            .map(|p| {
                frames
                    .iter()
                    .map(|frame| initial_rotation(frame.placement, p))
                    .collect()
            })
            .unwrap_or_default();
        Self {
            frames,
            width,
            height,
            projection,
            rotations,
        }
    }

    fn initial(&self) -> Vec<f64> {
        let mut params = vec![0.0; LENS_PARAMS];
        for frame in &self.frames[1..] {
            let t = frame.placement;
            params.extend(match self.projection {
                Some(_) => [0.0, 0.0, 0.0, t.scale.ln()],
                None => [
                    t.rotation_radians,
                    t.scale.ln(),
                    t.translation_x,
                    t.translation_y,
                ],
            });
        }
        params
    }

    fn unpack(&self, params: &[f64]) -> (LensDistortion, Vec<Pose>) {
        let mut distortion = LensDistortion::none(self.width, self.height);
        distortion.k1 = params[0];
        distortion.k2 = params[1];
        distortion.center_x += params[2];
        distortion.center_y += params[3];
        let mut poses = vec![Pose::Similarity(SimilarityTransform::IDENTITY)];
        for (index, p) in params[LENS_PARAMS..]
            .as_chunks::<POSE_PARAMS>()
            .0
            .iter()
            .enumerate()
        {
            poses.push(self.pose(index + 1, p));
        }
        if let Some(projection) = self.projection {
            poses[0] = Pose::Rotation {
                rotation: IDENTITY,
                focal: projection.focal,
            };
        }
        (distortion, poses)
    }

    fn pose(&self, frame: usize, p: &[f64]) -> Pose {
        match self.projection {
            Some(projection) => Pose::Rotation {
                rotation: multiply(&rodrigues([p[0], p[1], p[2]]), &self.rotations[frame]),
                focal: projection.focal * (-p[3]).exp(),
            },
            None => Pose::Similarity(SimilarityTransform {
                scale: p[1].exp(),
                rotation_radians: p[0],
                translation_x: p[2],
                translation_y: p[3],
            }),
        }
    }

    /// Every frame's undistorted stars.
    fn ideal(&self, distortion: &LensDistortion) -> Vec<Vec<(f64, f64)>> {
        self.frames
            .iter()
            .map(|frame| {
                frame
                    .stars
                    .iter()
                    .map(|star| distortion.undistort(star.x, star.y))
                    .collect()
            })
            .collect()
    }

    fn place(&self, pose: &Pose, ideal: &[(f64, f64)]) -> Vec<(f64, f64)> {
        ideal
            .iter()
            .map(|&(x, y)| {
                pose.apply(self.projection, x, y)
                    .unwrap_or((f64::NAN, f64::NAN))
            })
            .collect()
    }

    fn positions(&self, params: &[f64]) -> Vec<Vec<(f64, f64)>> {
        let (distortion, poses) = self.unpack(params);
        self.ideal(&distortion)
            .iter()
            .zip(&poses)
            .map(|(ideal, pose)| self.place(pose, ideal))
            .collect()
    }

    /// [`pair_positions`] where `params` place every frame's stars.
    fn pair(&self, params: &[f64], tolerance: f64) -> Vec<Pair> {
        pair_positions(&self.positions(params), tolerance)
    }

    fn errors(&self, params: &[f64], pairs: &[Pair]) -> Vec<f64> {
        let positions = self.positions(params);
        pairs
            .iter()
            .map(|&(i, a, j, b)| {
                let (p, q) = (positions[i][a], positions[j][b]);
                (p.0 - q.0).hypot(p.1 - q.1)
            })
            .collect()
    }

    /// Drop pairs further apart than three times the median, but never
    /// those within a third of a pixel.
    fn clip(&self, params: &[f64], pairs: Vec<Pair>) -> Vec<Pair> {
        let errors = self.errors(params, &pairs);
        let mut sorted = errors
            .iter()
            .copied()
            .filter(|e| e.is_finite())
            .collect::<Vec<_>>();
        if sorted.is_empty() {
            return pairs;
        }
        sorted.sort_by(f64::total_cmp);
        let limit = (3.0 * sorted[sorted.len() / 2]).max(0.3);
        pairs
            .into_iter()
            .zip(errors)
            .filter(|(_, error)| *error <= limit)
            .map(|(pair, _)| pair)
            .collect()
    }

    /// Levenberg-Marquardt over the parameters `fixed` leaves free, with a
    /// forward-difference Jacobian: a frame's pose moves only its own stars,
    /// so only the distortion's columns undistort every star again.
    fn fit(&self, params: &mut [f64], pairs: &[Pair], fixed: &dyn Fn(usize) -> bool) {
        if pairs.is_empty() {
            return;
        }
        let free = (0..params.len()).filter(|&k| !fixed(k)).collect::<Vec<_>>();
        let residuals = |positions: &[Vec<(f64, f64)>]| -> Vec<f64> {
            let mut out = Vec::with_capacity(pairs.len() * 2);
            for &(i, a, j, b) in pairs {
                let (p, q) = (positions[i][a], positions[j][b]);
                out.push(p.0 - q.0);
                out.push(p.1 - q.1);
            }
            out
        };
        let cost = |r: &[f64]| -> f64 {
            r.iter()
                .map(|v| if v.is_finite() { v * v } else { 1.0e12 })
                .sum()
        };
        let mut damping = 1.0e-3;
        let mut positions = self.positions(params);
        let mut current = residuals(&positions);
        let mut current_cost = cost(&current);
        for _ in 0..50 {
            let (distortion, _) = self.unpack(params);
            let ideal = self.ideal(&distortion);
            let mut jacobian = Vec::with_capacity(free.len());
            for &k in &free {
                let step = step_size(k, self.projection.is_some());
                let mut moved = params.to_vec();
                moved[k] += step;
                let shifted = if k < LENS_PARAMS {
                    self.positions(&moved)
                } else {
                    let frame = 1 + (k - LENS_PARAMS) / POSE_PARAMS;
                    let mut shifted = positions.clone();
                    let offset = LENS_PARAMS + (frame - 1) * POSE_PARAMS;
                    let pose = self.pose(frame, &moved[offset..offset + POSE_PARAMS]);
                    shifted[frame] = self.place(&pose, &ideal[frame]);
                    shifted
                };
                let column = residuals(&shifted)
                    .iter()
                    .zip(&current)
                    .map(|(after, before)| {
                        let d = (after - before) / step;
                        if d.is_finite() { d } else { 0.0 }
                    })
                    .collect::<Vec<_>>();
                jacobian.push(column);
            }
            let n = free.len();
            let mut normal = vec![vec![0.0; n]; n];
            let mut gradient = vec![0.0; n];
            for a in 0..n {
                for b in a..n {
                    let value = jacobian[a]
                        .iter()
                        .zip(&jacobian[b])
                        .map(|(x, y)| x * y)
                        .sum::<f64>();
                    normal[a][b] = value;
                    normal[b][a] = value;
                }
                gradient[a] = jacobian[a]
                    .iter()
                    .zip(&current)
                    .map(|(j, r)| if r.is_finite() { j * r } else { 0.0 })
                    .sum();
            }
            let mut improved = false;
            while damping < 1.0e10 {
                let mut damped = normal.clone();
                for (a, row) in damped.iter_mut().enumerate() {
                    row[a] += damping * normal[a][a].max(1.0e-12);
                }
                let Some(delta) = solve(damped, gradient.iter().map(|g| -g).collect()) else {
                    damping *= 10.0;
                    continue;
                };
                let mut trial = params.to_vec();
                for (&k, d) in free.iter().zip(&delta) {
                    trial[k] += d;
                }
                let trial_positions = self.positions(&trial);
                let trial_residuals = residuals(&trial_positions);
                let trial_cost = cost(&trial_residuals);
                if trial_cost < current_cost {
                    let gain = (current_cost - trial_cost) / current_cost.max(1.0e-300);
                    params.copy_from_slice(&trial);
                    positions = trial_positions;
                    current = trial_residuals;
                    current_cost = trial_cost;
                    damping = (damping / 3.0).max(1.0e-12);
                    improved = gain > 1.0e-10;
                    break;
                }
                damping *= 4.0;
            }
            if !improved {
                break;
            }
        }
    }
}

/// Unambiguous nearest neighbors within `tolerance` between every two
/// frames' `positions`, each star in at most one pair per frame pair.
fn pair_positions(positions: &[Vec<(f64, f64)>], tolerance: f64) -> Vec<Pair> {
    let cell = |(x, y): (f64, f64)| {
        (
            (x / tolerance).floor() as i64,
            (y / tolerance).floor() as i64,
        )
    };
    let mut pairs = Vec::new();
    for j in 1..positions.len() {
        let mut grid = HashMap::<(i64, i64), Vec<usize>>::new();
        for (index, &position) in positions[j].iter().enumerate() {
            if position.0.is_finite() && position.1.is_finite() {
                grid.entry(cell(position)).or_default().push(index);
            }
        }
        for i in 0..j {
            let mut best = HashMap::<usize, (f64, usize)>::new();
            for (a, &position) in positions[i].iter().enumerate() {
                if !(position.0.is_finite() && position.1.is_finite()) {
                    continue;
                }
                let (cx, cy) = cell(position);
                let mut near = Vec::new();
                for nx in cx - 1..=cx + 1 {
                    for ny in cy - 1..=cy + 1 {
                        for &b in grid.get(&(nx, ny)).into_iter().flatten() {
                            let (x, y) = positions[j][b];
                            let distance = (x - position.0).hypot(y - position.1);
                            if distance <= tolerance {
                                near.push((distance, b));
                            }
                        }
                    }
                }
                if let [(distance, b)] = near[..]
                    && best.get(&b).is_none_or(|&(d, _)| distance < d)
                {
                    best.insert(b, (distance, a));
                }
            }
            pairs.extend(best.into_iter().map(|(b, (_, a))| (i, a, j, b)));
        }
    }
    pairs.sort_unstable();
    pairs
}

/// A forward-difference step moving stars by around a thousandth of a pixel.
fn step_size(param: usize, rotation: bool) -> f64 {
    match param {
        0 | 1 => 1.0e-7,
        2 | 3 => 1.0e-3,
        _ => match ((param - LENS_PARAMS) % POSE_PARAMS, rotation) {
            (_, true) => 1.0e-8,
            (0 | 1, false) => 1.0e-7,
            _ => 1.0e-3,
        },
    }
}

/// The camera rotation that places a frame's center and roll where its
/// similarity onto the anchor's grid does.
fn initial_rotation(placement: SimilarityTransform, projection: Projection) -> [[f64; 3]; 3] {
    let (x, y) = placement.apply(projection.center_x, projection.center_y);
    let direction = normalize([
        x - projection.center_x,
        y - projection.center_y,
        projection.focal,
    ]);
    // The smallest turn taking the view axis onto `direction`.
    let axis = [-direction[1], direction[0], 0.0];
    let sine = axis[0].hypot(axis[1]);
    let align = if sine < 1.0e-15 {
        IDENTITY
    } else {
        let angle = sine.atan2(direction[2]);
        rodrigues([axis[0] / sine * angle, axis[1] / sine * angle, 0.0])
    };
    let (s, c) = placement.rotation_radians.sin_cos();
    multiply(&align, &[[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
}

const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / length, v[1] / length, v[2] / length]
}

/// The rotation by `|v|` radians about `v`.
fn rodrigues(v: [f64; 3]) -> [[f64; 3]; 3] {
    let angle = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if angle < 1.0e-300 {
        return IDENTITY;
    }
    let k = [v[0] / angle, v[1] / angle, v[2] / angle];
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    [
        [
            c + k[0] * k[0] * t,
            k[0] * k[1] * t - k[2] * s,
            k[0] * k[2] * t + k[1] * s,
        ],
        [
            k[1] * k[0] * t + k[2] * s,
            c + k[1] * k[1] * t,
            k[1] * k[2] * t - k[0] * s,
        ],
        [
            k[2] * k[0] * t - k[1] * s,
            k[2] * k[1] * t + k[0] * s,
            c + k[2] * k[2] * t,
        ],
    ]
}

fn multiply(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for (row, out_row) in out.iter_mut().enumerate() {
        for (column, value) in out_row.iter_mut().enumerate() {
            *value = (0..3).map(|k| a[row][k] * b[k][column]).sum();
        }
    }
    out
}

#[cfg(test)]
fn transpose(a: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for (row, out_row) in out.iter_mut().enumerate() {
        for (column, value) in out_row.iter_mut().enumerate() {
            *value = a[column][row];
        }
    }
    out
}

fn rotate(a: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

/// Solve a square system by Gaussian elimination with partial pivoting.
fn solve(mut matrix: Vec<Vec<f64>>, mut right: Vec<f64>) -> Option<Vec<f64>> {
    let n = right.len();
    for column in 0..n {
        let pivot = (column..n)
            .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))?;
        let magnitude = matrix[pivot][column].abs();
        if magnitude.is_nan() || magnitude <= 1.0e-300 {
            return None;
        }
        matrix.swap(column, pivot);
        right.swap(column, pivot);
        let pivot_row = matrix[column].clone();
        for row in column + 1..n {
            let factor = matrix[row][column] / pivot_row[column];
            for (value, pivot) in matrix[row][column..].iter_mut().zip(&pivot_row[column..]) {
                *value -= factor * pivot;
            }
            right[row] -= factor * right[column];
        }
    }
    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let sum = right[row]
            - (row + 1..n)
                .map(|k| matrix[row][k] * solution[k])
                .sum::<f64>();
        solution[row] = sum / matrix[row][row];
    }
    solution.iter().all(|v| v.is_finite()).then_some(solution)
}

/// The similarity carrying `from` onto `to` in the least-squares sense.
pub(crate) fn fit_similarity(
    from: &[(f64, f64)],
    to: &[(f64, f64)],
) -> Option<SimilarityTransform> {
    let n = from.len().min(to.len());
    if n < 2 {
        return None;
    }
    let mean = |points: &[(f64, f64)]| {
        let (x, y) = points[..n]
            .iter()
            .fold((0.0, 0.0), |acc, p| (acc.0 + p.0, acc.1 + p.1));
        (x / n as f64, y / n as f64)
    };
    let (fx, fy) = mean(from);
    let (tx, ty) = mean(to);
    let (mut a, mut b, mut norm) = (0.0, 0.0, 0.0);
    for (p, q) in from[..n].iter().zip(&to[..n]) {
        let (u, v) = (p.0 - fx, p.1 - fy);
        let (s, t) = (q.0 - tx, q.1 - ty);
        a += u * s + v * t;
        b += u * t - v * s;
        norm += u * u + v * v;
    }
    if norm.is_nan() || norm <= 0.0 {
        return None;
    }
    let (a, b) = (a / norm, b / norm);
    let scale = a.hypot(b);
    let rotation = b.atan2(a);
    let (sin, cos) = rotation.sin_cos();
    Some(SimilarityTransform {
        scale,
        rotation_radians: rotation,
        translation_x: tx - scale * (cos * fx - sin * fy),
        translation_y: ty - scale * (sin * fx + cos * fy),
    })
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
            area: 9,
        }
    }

    #[test]
    fn undistort_inverts_distort() {
        let mut lens = LensDistortion::none(2000, 1300);
        lens.k1 = 0.012;
        lens.k2 = -0.004;
        lens.center_x += 31.0;
        lens.center_y -= 17.0;
        lens.validate().unwrap();
        for (x, y) in [
            (0.0, 0.0),
            (1999.0, 1299.0),
            (1000.0, 650.0),
            (12.5, 1200.25),
        ] {
            let (u, v) = lens.distort(x, y);
            let (back_x, back_y) = lens.undistort(u, v);
            assert!((back_x - x).abs() < 1.0e-8 && (back_y - y).abs() < 1.0e-8);
        }
        assert!(lens.largest_shift() > 5.0);
    }

    /// A lens and the poses fitted with it carry between the sensor's grid
    /// and a binned one exactly: each maps a binned point where it maps the
    /// sensor's point, converted.
    #[test]
    fn a_lens_fit_carries_between_the_sensor_and_a_binned_grid() {
        let mut lens = LensDistortion::none(2000, 1300);
        lens.k1 = 0.012;
        lens.k2 = -0.004;
        lens.center_x += 31.0;
        lens.center_y -= 17.0;
        let projection = Projection {
            focal: 9000.0,
            center_x: 999.5,
            center_y: 649.5,
        };
        let fit = LensFit {
            distortion: lens,
            projection: Some(projection),
            poses: vec![
                Pose::Rotation {
                    rotation: IDENTITY,
                    focal: 9000.0,
                },
                Pose::Rotation {
                    rotation: rodrigues([0.01, -0.02, 0.003]),
                    focal: 9012.0,
                },
                Pose::Similarity(SimilarityTransform {
                    scale: 0.999,
                    rotation_radians: 3.1,
                    translation_x: 1834.0,
                    translation_y: 1199.5,
                }),
            ],
            pairs: vec![0, 40, 41],
            rms: vec![0.0, 0.2, 0.3],
            overall_rms: 0.25,
            undistorted_rms: 1.4,
        };
        for bin in [2.0, 4.0] {
            let offset = (bin - 1.0) / 2.0;
            let to_binned = |x: f64, y: f64| ((x - offset) / bin, (y - offset) / bin);
            let binned = fit.rescaled(1.0 / bin);
            binned.distortion.validate().unwrap();
            assert_eq!(
                (binned.distortion.width, binned.distortion.height),
                (2000 / bin as usize, 1300 / bin as usize)
            );
            let projection = binned.projection.unwrap();
            assert_eq!(
                (projection.center_x, projection.center_y),
                ((2000.0 / bin - 1.0) / 2.0, (1300.0 / bin - 1.0) / 2.0)
            );
            for (x, y) in [(0.0, 0.0), (1999.0, 1299.0), (731.5, 222.25)] {
                let (bx, by) = to_binned(x, y);
                let close = |(a, b): (f64, f64), (c, d): (f64, f64)| {
                    (bin * a + offset - c).abs() < 1e-7 && (bin * b + offset - d).abs() < 1e-7
                };
                assert!(close(binned.distortion.distort(bx, by), lens.distort(x, y)));
                for (native, coarse) in fit.poses.iter().zip(&binned.poses) {
                    let expected = native.apply(fit.projection, x, y).unwrap();
                    let found = coarse.apply(binned.projection, bx, by).unwrap();
                    assert!(close(found, expected), "{found:?} against {expected:?}");
                }
            }
            assert_eq!(binned.rms, vec![0.0, 0.2 / bin, 0.3 / bin]);
            let back = binned.rescaled(bin);
            assert!((back.distortion.center_x - lens.center_x).abs() < 1e-9);
            assert!((back.distortion.radius - lens.radius).abs() < 1e-9);
        }
    }

    #[test]
    fn folding_distortion_is_invalid() {
        let mut lens = LensDistortion::none(100, 100);
        lens.k1 = -0.8;
        assert!(lens.validate().is_err());
    }

    #[test]
    fn similarity_fit_recovers_a_transform() {
        let truth = SimilarityTransform {
            scale: 1.003,
            rotation_radians: 2.9,
            translation_x: 120.0,
            translation_y: -44.0,
        };
        let from = [(0.0, 0.0), (100.0, 5.0), (40.0, 90.0), (300.0, 200.0)];
        let to = from.map(|(x, y)| truth.apply(x, y));
        let fitted = fit_similarity(&from, &to).unwrap();
        for (p, q) in from.iter().zip(&to) {
            let (x, y) = fitted.apply(p.0, p.1);
            assert!((x - q.0).abs() < 1.0e-9 && (y - q.1).abs() < 1.0e-9);
        }
    }

    /// Three frames of one sky, stacked vertically with a fifth of each
    /// overlapping the next, the middle one turned half a revolution, all
    /// imaged through one decentered lens.
    fn mosaic(
        projection: Option<Projection>,
    ) -> (LensDistortion, Vec<Pose>, Vec<Vec<DetectedStar>>) {
        mosaic_with(projection, (0.012, 0.002), 0.1)
    }

    /// [`mosaic`] through a lens of radial terms `(k1, k2)`, its stars
    /// scattered uniformly over `jitter` pixels.
    fn mosaic_with(
        projection: Option<Projection>,
        (k1, k2): (f64, f64),
        jitter: f64,
    ) -> (LensDistortion, Vec<Pose>, Vec<Vec<DetectedStar>>) {
        let (width, height) = (2000, 1300);
        let mut lens = LensDistortion::none(width, height);
        lens.k1 = k1;
        lens.k2 = k2;
        lens.center_x += 40.0;
        lens.center_y += 60.0;
        let offsets = [0.0, 1040.0, 2080.0];
        let (cx, cy) = ((width - 1) as f64 * 0.5, (height - 1) as f64 * 0.5);
        let poses = offsets
            .iter()
            .enumerate()
            .map(|(index, &offset)| {
                let turn = if index == 1 {
                    std::f64::consts::PI
                } else {
                    0.0
                };
                let turned = SimilarityTransform {
                    scale: 1.0 + 0.0005 * index as f64,
                    rotation_radians: turn + 0.002 * index as f64,
                    translation_x: 0.0,
                    translation_y: 0.0,
                };
                // The frame's center lands `offset` below the anchor's.
                let (x, y) = turned.apply(cx, cy);
                let similarity = SimilarityTransform {
                    translation_x: cx + 3.0 * index as f64 - x,
                    translation_y: cy + offset - y,
                    ..turned
                };
                match projection {
                    Some(p) => Pose::Rotation {
                        rotation: initial_rotation(similarity, p),
                        focal: p.focal / similarity.scale,
                    },
                    None => Pose::Similarity(similarity),
                }
            })
            .collect::<Vec<_>>();
        // A deterministic scatter of sky stars across the whole mosaic.
        let mut state = 12345_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let sky = (0..9000)
            .map(|_| (next() * 2100.0 - 50.0, next() * 3500.0 - 50.0))
            .collect::<Vec<_>>();
        let frames = poses
            .iter()
            .map(|pose| {
                sky.iter()
                    .filter_map(|&(x, y)| {
                        let (u, v) = pose.inverse(projection, x, y)?;
                        let (u, v) = lens.distort(u, v);
                        (u >= 0.0 && v >= 0.0 && u < width as f64 && v < height as f64)
                            .then(|| star(u + (next() - 0.5) * jitter, v + (next() - 0.5) * jitter))
                    })
                    .collect()
            })
            .collect();
        (lens, poses, frames)
    }

    /// What a planner would find: each frame's similarity to the anchor
    /// fitted across the whole frame, distortion and all.
    fn planned<'a>(
        projection: Option<Projection>,
        lens: &LensDistortion,
        poses: &[Pose],
        stars: &'a [Vec<DetectedStar>],
    ) -> Vec<LensFrame<'a>> {
        poses
            .iter()
            .zip(stars)
            .map(|(pose, stars)| {
                let raw = stars.iter().map(|s| (s.x, s.y)).collect::<Vec<_>>();
                let sky = raw
                    .iter()
                    .map(|&(x, y)| {
                        let (u, v) = lens.undistort(x, y);
                        pose.apply(projection, u, v).unwrap()
                    })
                    .collect::<Vec<_>>();
                LensFrame {
                    stars,
                    placement: fit_similarity(&raw, &sky).unwrap(),
                }
            })
            .collect()
    }

    fn check(projection: Option<Projection>) {
        let (lens, poses, stars) = mosaic(projection);
        let (width, height) = (2000, 1300);
        let frames = planned(projection, &lens, &poses, &stars);
        let fit = estimate(&frames, width, height, projection.map(|p| p.focal)).unwrap();
        assert!(fit.overall_rms < 0.1, "{fit:?}");
        assert!(fit.undistorted_rms > 1.0, "{fit:?}");
        assert!(fit.pairs[1..].iter().all(|&n| n > 300), "{fit:?}");
        // The fitted model carries every frame's stars where the truth does.
        for ((pose, truth), frame) in fit.poses.iter().zip(&poses).zip(&stars) {
            for star in frame.iter().step_by(37) {
                let (u, v) = fit.distortion.undistort(star.x, star.y);
                let (x, y) = pose.apply(fit.projection, u, v).unwrap();
                let (tu, tv) = lens.undistort(star.x, star.y);
                let (tx, ty) = truth.apply(projection, tu, tv).unwrap();
                assert!((x - tx).hypot(y - ty) < 0.5, "{x},{y} vs {tx},{ty}");
            }
        }
    }

    #[test]
    fn estimate_recovers_a_decentered_lens_by_similarities() {
        check(None);
    }

    #[test]
    fn estimate_recovers_a_decentered_lens_on_tangent_planes() {
        check(Some(Projection {
            focal: 30_000.0,
            center_x: 999.5,
            center_y: 649.5,
        }));
    }

    /// A mild lens, its stars scattered as centroids scatter: the distortion
    /// takes a fifth off the residual, and is fitted, close to the truth.
    #[test]
    fn estimate_fits_a_mild_lens_behind_star_noise() {
        let (lens, poses, stars) = mosaic_with(None, (0.0012, 0.0), 0.6);
        let frames = planned(None, &lens, &poses, &stars);
        let fit = estimate(&frames, 2000, 1300, None).unwrap();
        let ratio = fit.overall_rms / fit.undistorted_rms;
        assert!(ratio > 0.75 && ratio < 0.85, "{fit:?}");
        assert!(fit.undistorted_rms - fit.overall_rms < 0.1, "{fit:?}");
        for (x, y) in [
            (0.0, 0.0),
            (1999.0, 0.0),
            (0.0, 1299.0),
            (1999.0, 1299.0),
            (1000.0, 650.0),
        ] {
            let (u, v) = fit.distortion.undistort(x, y);
            let (tu, tv) = lens.undistort(x, y);
            // The fit may absorb part of the distortion in the poses' scale,
            // so compare its departure from the frame's own similarity.
            let shift = (u - x).hypot(v - y) - (tu - x).hypot(tv - y);
            assert!(shift.abs() < 0.6, "({x}, {y}): {shift}");
        }
    }

    /// The same star noise with no lens behind it: nothing to fit.
    #[test]
    fn estimate_declines_star_noise_alone() {
        let (lens, poses, stars) = mosaic_with(None, (0.0, 0.0), 0.6);
        let frames = planned(None, &lens, &poses, &stars);
        assert!(estimate(&frames, 2000, 1300, None).is_none());
    }

    #[test]
    fn estimate_declines_frames_without_distortion() {
        let (_, poses, _) = mosaic(None);
        let (width, height) = (2000, 1300);
        let mut state = 7_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let sky = (0..6000)
            .map(|_| (next() * 2100.0 - 50.0, next() * 3500.0 - 50.0))
            .collect::<Vec<_>>();
        let stars = poses
            .iter()
            .map(|pose| {
                sky.iter()
                    .filter_map(|&(x, y)| {
                        let (u, v) = pose.inverse(None, x, y)?;
                        (u >= 0.0 && v >= 0.0 && u < width as f64 && v < height as f64)
                            .then(|| star(u, v))
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let frames = poses
            .iter()
            .zip(&stars)
            .map(|(pose, stars)| LensFrame {
                stars,
                placement: match pose {
                    Pose::Similarity(t) => *t,
                    Pose::Rotation { .. } => unreachable!(),
                },
            })
            .collect::<Vec<_>>();
        assert!(estimate(&frames, width, height, None).is_none());
    }
}
