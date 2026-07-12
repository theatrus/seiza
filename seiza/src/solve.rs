//! Near-field plate solving: match detected stars against catalog stars
//! around a hinted position and fit a [`crate::wcs::Wcs`].
//!
//! The algorithm, in order:
//!
//! 1. Project catalog stars around the hint onto a tangent plane.
//! 2. Form triangles from the brightest image and catalog stars and match
//!    them by side-ratio descriptors (scale/rotation/parity invariant).
//! 3. Each triangle correspondence proposes an affine pixel→tangent
//!    transform; implausible ones (wrong scale, strong shear) are dropped.
//! 4. Candidates are scored by how many detected stars land on catalog
//!    stars; the best transform seeds an iterative least-squares fit with
//!    re-matching and outlier clipping.
//! 5. The fitted affine is re-centred on the image and converted to a WCS.

use crate::catalog::{CatalogStar, StarCatalog, angular_separation_deg};
use crate::detect::DetectedStar;
use crate::wcs::Wcs;

/// Approximate knowledge required to seed a solve.
#[derive(Debug, Clone)]
pub struct SolveHint {
    /// Approximate field center, ICRS degrees (RA, Dec)
    pub center: (f64, f64),
    /// Search radius around the hinted center, degrees
    pub radius_deg: f64,
    /// Approximate pixel scale, arcseconds per pixel
    pub scale_arcsec_px: f64,
    /// Allowed relative scale error (e.g. 0.2 = ±20 %)
    pub scale_tolerance: f64,
}

/// A successful plate solution with quality metrics.
#[derive(Debug, Clone)]
pub struct Solution {
    pub wcs: Wcs,
    /// Number of detected stars matched to catalog stars
    pub matched_stars: usize,
    /// RMS of match residuals, arcseconds
    pub rms_arcsec: f64,
}

// Tuning constants. These are deliberately conservative; they can move
// into SolveHint if a use case needs it.
const N_IMAGE_TRIANGLE_STARS: usize = 24;
const N_CATALOG_TRIANGLE_STARS: usize = 100;
const N_VOTE_STARS: usize = 60;
const RATIO_TOLERANCE: f64 = 0.01;
const MIN_TRIANGLE_RATIO: f64 = 0.12;
const MAX_SHEAR: f64 = 0.06;
const MAX_CANDIDATES: usize = 4000;
const MATCH_TOLERANCE_PX: f64 = 4.0;
const MIN_INLIERS: usize = 6;
const REFINE_ITERATIONS: usize = 3;

/// Solve the field for an image of `(width, height)` given detected stars,
/// a catalog, and a hint.
pub fn solve(
    stars: &[DetectedStar],
    catalog: &dyn StarCatalog,
    hint: &SolveHint,
    dimensions: (u32, u32),
) -> Result<Solution, crate::Error> {
    if stars.len() < 4 {
        return Err(crate::Error::Solve(format!(
            "only {} stars detected; need at least 4",
            stars.len()
        )));
    }
    let scale_deg = hint.scale_arcsec_px / 3600.0;
    let (width, height) = (dimensions.0 as f64, dimensions.1 as f64);
    let fov_radius_deg = (width.hypot(height) / 2.0) * scale_deg * (1.0 + hint.scale_tolerance);

    // Catalog stars around the hint, on the tangent plane about the hint
    let search_radius = hint.radius_deg + fov_radius_deg;
    let area_ratio = (search_radius / fov_radius_deg.max(1e-6))
        .powi(2)
        .clamp(1.0, 16.0);
    let limit = (250.0 * area_ratio) as usize;
    let cone = catalog.cone_search(hint.center.0, hint.center.1, search_radius, limit);
    if cone.len() < 4 {
        return Err(crate::Error::Solve(format!(
            "only {} catalog stars within {search_radius:.2}° of the hint",
            cone.len()
        )));
    }
    let tangent = tangent_wcs(hint.center);
    let cat: Vec<(f64, f64, CatalogStar)> = cone
        .iter()
        .filter_map(|s| tangent.world_to_pixel(s.ra, s.dec).map(|(x, y)| (x, y, *s)))
        .collect();

    // Candidate transforms from triangle correspondences. Catalog
    // triangles are built per FOV-sized window sliding over the search
    // region — the brightest stars of the whole search cone mostly lie
    // outside the image, so a single global set would rarely overlap the
    // image's brightest stars. Image triangle stars are picked with
    // spatial diversity so a bright localized feature (a galaxy core, a
    // watermark strip) cannot monopolize the set.
    let img_pts = diverse_brightest(stars, dimensions, N_IMAGE_TRIANGLE_STARS);
    let img_tris = triangles(&img_pts);

    let step = fov_radius_deg.max(0.05);
    let n_steps = (hint.radius_deg / step).ceil() as i32;
    let mut windows: Vec<(f64, f64)> = Vec::new();
    for gy in -n_steps..=n_steps {
        for gx in -n_steps..=n_steps {
            let (ox, oy) = (gx as f64 * step, gy as f64 * step);
            if ox.hypot(oy) <= hint.radius_deg + 1e-9 {
                windows.push((ox, oy));
            }
        }
    }

    let mut candidates: Vec<(f64, Affine)> = Vec::new();
    for &(ox, oy) in &windows {
        let mut in_window: Vec<(f64, f64)> = cat
            .iter()
            .filter(|&&(x, y, _)| (x - ox).hypot(y - oy) <= fov_radius_deg)
            .map(|&(x, y, _)| (x, y))
            .collect();
        in_window.truncate(N_CATALOG_TRIANGLE_STARS);
        let mut cat_tris = triangles(&in_window);
        cat_tris.sort_by(|a, b| a.ratio_ba.total_cmp(&b.ratio_ba));

        for it in &img_tris {
            // Catalog triangles are sorted by ratio_ba: visit only the
            // tolerance window instead of the full cross product
            let start = cat_tris.partition_point(|t| t.ratio_ba < it.ratio_ba - RATIO_TOLERANCE);
            for ct in &cat_tris[start..] {
                if ct.ratio_ba > it.ratio_ba + RATIO_TOLERANCE {
                    break;
                }
                if (it.ratio_ca - ct.ratio_ca).abs() > RATIO_TOLERANCE {
                    continue;
                }
                let err = (it.ratio_ba - ct.ratio_ba)
                    .abs()
                    .max((it.ratio_ca - ct.ratio_ca).abs());
                // The tangent/pixel side length ratio must match the scale hint
                let tri_scale = ct.longest / it.longest;
                if (tri_scale / scale_deg - 1.0).abs() > hint.scale_tolerance {
                    continue;
                }
                let Some(affine) = Affine::from_three_points(&it.vertices, &ct.vertices) else {
                    continue;
                };
                if !affine.is_similarity_like(scale_deg, hint.scale_tolerance) {
                    continue;
                }
                candidates.push((err, affine));
            }
        }
    }
    if candidates.is_empty() {
        return Err(crate::Error::Solve(
            "no plausible triangle correspondences found".to_string(),
        ));
    }
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
    candidates.truncate(MAX_CANDIDATES);

    // Vote: which candidate lands the most detected stars on catalog stars?
    let vote_stars: Vec<(f64, f64)> = stars
        .iter()
        .take(N_VOTE_STARS)
        .map(|s| (s.x, s.y))
        .collect();
    let match_tol = MATCH_TOLERANCE_PX * scale_deg;
    let mut best: Option<(usize, Affine)> = None;
    for (_, affine) in &candidates {
        let inliers = count_inliers(affine, &vote_stars, &cat, match_tol);
        if best.as_ref().is_none_or(|(count, _)| inliers > *count) {
            best = Some((inliers, affine.clone()));
        }
    }
    let (votes, mut affine) = best.unwrap();
    if votes < MIN_INLIERS.min(vote_stars.len() * 2 / 3) {
        return Err(crate::Error::Solve(format!(
            "best candidate matched only {votes} of {} stars",
            vote_stars.len()
        )));
    }

    // Iterative least-squares refinement with re-matching
    let all_stars: Vec<(f64, f64)> = stars.iter().map(|s| (s.x, s.y)).collect();
    let mut pairs = Vec::new();
    for _ in 0..REFINE_ITERATIONS {
        pairs = match_pairs(&affine, &all_stars, &cat, match_tol);
        if pairs.len() < MIN_INLIERS {
            return Err(crate::Error::Solve(format!(
                "refinement collapsed to {} matches",
                pairs.len()
            )));
        }
        let fit_pairs: Vec<PointPair> = pairs
            .iter()
            .map(|&((px, py), ci)| ((px, py), (cat[ci].0, cat[ci].1)))
            .collect();
        affine = Affine::fit(&fit_pairs)
            .ok_or_else(|| crate::Error::Solve("degenerate least-squares system".to_string()))?;
    }

    // Re-centre: express the solution as a WCS about the image center
    let center_px = (width / 2.0, height / 2.0);
    let (cx, cy) = affine.apply(center_px.0, center_px.1);
    let crval = tangent.pixel_to_world(cx, cy);
    let recentre = tangent_wcs(crval);
    let final_pairs: Vec<PointPair> = pairs
        .iter()
        .filter_map(|&((px, py), cat_idx)| {
            let star = &cat[cat_idx].2;
            recentre
                .world_to_pixel(star.ra, star.dec)
                .map(|t| ((px, py), t))
        })
        .collect();
    let final_affine = Affine::fit(&final_pairs)
        .ok_or_else(|| crate::Error::Solve("degenerate final fit".to_string()))?;

    let crpix = final_affine
        .zero_point()
        .ok_or_else(|| crate::Error::Solve("solution transform is singular".to_string()))?;
    let wcs = Wcs {
        crval,
        crpix,
        cd: [
            [final_affine.a, final_affine.b],
            [final_affine.d, final_affine.e],
        ],
    };

    // Quality metrics against the final WCS
    let mut sum_sq = 0.0;
    for &((px, py), cat_idx) in &pairs {
        let star = &cat[cat_idx].2;
        let (ra, dec) = wcs.pixel_to_world(px, py);
        let sep = angular_separation_deg(ra, dec, star.ra, star.dec) * 3600.0;
        sum_sq += sep * sep;
    }
    let rms_arcsec = (sum_sq / pairs.len() as f64).sqrt();

    Ok(Solution {
        wcs,
        matched_stars: pairs.len(),
        rms_arcsec,
    })
}

/// The brightest `count` stars, but at most `count / 4 + 1` from any one
/// cell of a 4×4 grid over the image — bright non-stellar clutter tends to
/// cluster, real stars don't.
fn diverse_brightest(
    stars: &[DetectedStar],
    dimensions: (u32, u32),
    count: usize,
) -> Vec<(f64, f64)> {
    let per_cell_cap = count / 8 + 1;
    let (width, height) = (dimensions.0 as f64, dimensions.1 as f64);
    let mut per_cell = [0usize; 16];
    let mut picked = Vec::with_capacity(count);
    for star in stars {
        if picked.len() >= count {
            break;
        }
        let gx = ((star.x / width * 4.0) as usize).min(3);
        let gy = ((star.y / height * 4.0) as usize).min(3);
        let cell = gy * 4 + gx;
        if per_cell[cell] >= per_cell_cap {
            continue;
        }
        per_cell[cell] += 1;
        picked.push((star.x, star.y));
    }
    picked
}

/// A WCS whose pixel plane IS the tangent plane (degrees) about `center`.
fn tangent_wcs(center: (f64, f64)) -> Wcs {
    Wcs {
        crval: center,
        crpix: (0.0, 0.0),
        cd: [[1.0, 0.0], [0.0, 1.0]],
    }
}

#[derive(Debug, Clone)]
struct Triangle {
    /// Vertices ordered: opposite-longest, opposite-middle, opposite-shortest
    vertices: [(f64, f64); 3],
    ratio_ba: f64,
    ratio_ca: f64,
    longest: f64,
}

/// All non-degenerate triangles over a point set, with vertices in a
/// canonical order so correspondences map vertex-to-vertex.
fn triangles(points: &[(f64, f64)]) -> Vec<Triangle> {
    let mut result = Vec::new();
    let n = points.len();
    for i in 0..n {
        for j in i + 1..n {
            for k in j + 1..n {
                let p = [points[i], points[j], points[k]];
                // side s[v] is opposite vertex v
                let s = [dist(p[1], p[2]), dist(p[0], p[2]), dist(p[0], p[1])];
                let mut order = [0usize, 1, 2];
                order.sort_by(|&a, &b| s[b].total_cmp(&s[a]));
                let (a, b, c) = (s[order[0]], s[order[1]], s[order[2]]);
                if a <= 0.0 || c / a < MIN_TRIANGLE_RATIO {
                    continue;
                }
                result.push(Triangle {
                    vertices: [p[order[0]], p[order[1]], p[order[2]]],
                    ratio_ba: b / a,
                    ratio_ca: c / a,
                    longest: a,
                });
            }
        }
    }
    result
}

fn dist(p: (f64, f64), q: (f64, f64)) -> f64 {
    (p.0 - q.0).hypot(p.1 - q.1)
}

/// A correspondence between a source point and a target point
type PointPair = ((f64, f64), (f64, f64));

/// 6-parameter affine transform: (x, y) → (a x + b y + c, d x + e y + f)
#[derive(Debug, Clone)]
struct Affine {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
}

impl Affine {
    fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (
            self.a * x + self.b * y + self.c,
            self.d * x + self.e * y + self.f,
        )
    }

    /// Exact affine from three point correspondences.
    fn from_three_points(from: &[(f64, f64); 3], to: &[(f64, f64); 3]) -> Option<Self> {
        let m = [
            [from[0].0, from[0].1, 1.0],
            [from[1].0, from[1].1, 1.0],
            [from[2].0, from[2].1, 1.0],
        ];
        let abc = solve3(m, [to[0].0, to[1].0, to[2].0])?;
        let def = solve3(m, [to[0].1, to[1].1, to[2].1])?;
        Some(Self {
            a: abc[0],
            b: abc[1],
            c: abc[2],
            d: def[0],
            e: def[1],
            f: def[2],
        })
    }

    /// Least-squares affine over point pairs ((x, y), (u, v)).
    fn fit(pairs: &[PointPair]) -> Option<Self> {
        if pairs.len() < 3 {
            return None;
        }
        // Normal equations for design [x y 1]
        let (mut sxx, mut sxy, mut sx, mut syy, mut sy, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let (mut sxu, mut syu, mut su) = (0.0, 0.0, 0.0);
        let (mut sxv, mut syv, mut sv) = (0.0, 0.0, 0.0);
        for &((x, y), (u, v)) in pairs {
            sxx += x * x;
            sxy += x * y;
            sx += x;
            syy += y * y;
            sy += y;
            n += 1.0;
            sxu += x * u;
            syu += y * u;
            su += u;
            sxv += x * v;
            syv += y * v;
            sv += v;
        }
        let m = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]];
        let abc = solve3(m, [sxu, syu, su])?;
        let def = solve3(m, [sxv, syv, sv])?;
        Some(Self {
            a: abc[0],
            b: abc[1],
            c: abc[2],
            d: def[0],
            e: def[1],
            f: def[2],
        })
    }

    /// Whether the linear part is close to a (possibly reflected) similarity
    /// at a plausible scale.
    fn is_similarity_like(&self, expected_scale: f64, tolerance: f64) -> bool {
        // Column norms and their orthogonality measure shear directly
        let c1 = self.a.hypot(self.d);
        let c2 = self.b.hypot(self.e);
        if c1 <= 0.0 || c2 <= 0.0 {
            return false;
        }
        let ortho = (self.a * self.b + self.d * self.e).abs() / (c1 * c2);
        if ortho > MAX_SHEAR || (c1 / c2 - 1.0).abs() > MAX_SHEAR {
            return false;
        }
        let scale = (c1 * c2).sqrt();
        (scale / expected_scale - 1.0).abs() <= tolerance
    }

    /// The input point that maps to (0, 0).
    fn zero_point(&self) -> Option<(f64, f64)> {
        let det = self.a * self.e - self.b * self.d;
        if det.abs() < 1e-30 {
            return None;
        }
        Some((
            (self.b * self.f - self.e * self.c) / det,
            (self.d * self.c - self.a * self.f) / det,
        ))
    }
}

/// Solve a 3×3 linear system via Gaussian elimination with partial pivoting.
fn solve3(m: [[f64; 3]; 3], rhs: [f64; 3]) -> Option<[f64; 3]> {
    let mut aug = [[0.0f64; 4]; 3];
    for i in 0..3 {
        aug[i][..3].copy_from_slice(&m[i]);
        aug[i][3] = rhs[i];
    }
    for col in 0..3 {
        let pivot = (col..3).max_by(|&a, &b| aug[a][col].abs().total_cmp(&aug[b][col].abs()))?;
        if aug[pivot][col].abs() < 1e-30 {
            return None;
        }
        aug.swap(col, pivot);
        for row in 0..3 {
            if row == col {
                continue;
            }
            let factor = aug[row][col] / aug[col][col];
            let pivot_row = aug[col];
            for (k, value) in pivot_row.iter().enumerate().skip(col) {
                aug[row][k] -= factor * value;
            }
        }
    }
    Some([
        aug[0][3] / aug[0][0],
        aug[1][3] / aug[1][1],
        aug[2][3] / aug[2][2],
    ])
}

fn count_inliers(
    affine: &Affine,
    stars: &[(f64, f64)],
    cat: &[(f64, f64, CatalogStar)],
    tolerance: f64,
) -> usize {
    let tol_sq = tolerance * tolerance;
    stars
        .iter()
        .filter(|&&(x, y)| {
            let (u, v) = affine.apply(x, y);
            cat.iter()
                .any(|&(cx, cy, _)| (cx - u).powi(2) + (cy - v).powi(2) <= tol_sq)
        })
        .count()
}

/// Match each star to its nearest catalog star within tolerance; one catalog
/// star may serve at most one image star (greedy by distance).
fn match_pairs(
    affine: &Affine,
    stars: &[(f64, f64)],
    cat: &[(f64, f64, CatalogStar)],
    tolerance: f64,
) -> Vec<((f64, f64), usize)> {
    let tol_sq = tolerance * tolerance;
    let mut proposals: Vec<(f64, usize, usize)> = Vec::new();
    for (si, &(x, y)) in stars.iter().enumerate() {
        let (u, v) = affine.apply(x, y);
        let mut best: Option<(f64, usize)> = None;
        for (ci, &(cx, cy, _)) in cat.iter().enumerate() {
            let d = (cx - u).powi(2) + (cy - v).powi(2);
            if d <= tol_sq && best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, ci));
            }
        }
        if let Some((d, ci)) = best {
            proposals.push((d, si, ci));
        }
    }
    proposals.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut used_cat = vec![false; cat.len()];
    let mut used_star = vec![false; stars.len()];
    let mut pairs = Vec::new();
    for (_, si, ci) in proposals {
        if used_cat[ci] || used_star[si] {
            continue;
        }
        used_cat[ci] = true;
        used_star[si] = true;
        pairs.push((stars[si], ci));
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::MemoryCatalog;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// A synthetic sky region and the "detections" a given WCS would produce.
    fn synthetic_scene(
        truth: &Wcs,
        dims: (u32, u32),
        rng: &mut Lcg,
    ) -> (MemoryCatalog, Vec<DetectedStar>) {
        let mut catalog = Vec::new();
        let mut detected = Vec::new();

        // Catalog stars over a region comfortably containing the footprint
        for _ in 0..3000 {
            let ra = truth.crval.0 + (rng.next() - 0.5) * 12.0;
            let dec = (truth.crval.1 + (rng.next() - 0.5) * 12.0).clamp(-89.9, 89.9);
            let mag = (4.0 + rng.next() * 8.0) as f32;
            catalog.push(CatalogStar { ra, dec, mag });

            if let Some((x, y)) = truth.world_to_pixel(ra, dec)
                && x > 0.0
                && y > 0.0
                && x < dims.0 as f64
                && y < dims.1 as f64
            {
                if rng.next() < 0.3 {
                    continue; // undetected
                }
                detected.push(DetectedStar {
                    x: x + (rng.next() - 0.5) * 0.6,
                    y: y + (rng.next() - 0.5) * 0.6,
                    flux: 10f64.powf(-0.4 * mag as f64) * 1e6,
                    peak: 0.5,
                    area: 20,
                });
            }
        }
        // False detections
        for _ in 0..8 {
            detected.push(DetectedStar {
                x: rng.next() * dims.0 as f64,
                y: rng.next() * dims.1 as f64,
                flux: 10f64.powf(-0.4 * (5.0 + rng.next() * 6.0)) * 1e6,
                peak: 0.5,
                area: 15,
            });
        }
        detected.sort_by(|a, b| b.flux.total_cmp(&a.flux));
        (MemoryCatalog::new(catalog), detected)
    }

    fn assert_solution_matches(truth: &Wcs, solution: &Solution, dims: (u32, u32)) {
        assert!(solution.matched_stars >= 10, "{solution:?}");
        assert!(solution.rms_arcsec < 1.5, "{solution:?}");
        for &(x, y) in &[
            (0.0, 0.0),
            (dims.0 as f64, 0.0),
            (dims.0 as f64 / 2.0, dims.1 as f64 / 2.0),
            (0.0, dims.1 as f64),
        ] {
            let (ra_t, dec_t) = truth.pixel_to_world(x, y);
            let (ra_s, dec_s) = solution.wcs.pixel_to_world(x, y);
            let sep = angular_separation_deg(ra_t, dec_t, ra_s, dec_s) * 3600.0;
            assert!(sep < 3.0, "corner ({x}, {y}) off by {sep:.2}\"");
        }
    }

    #[test]
    fn solves_rotated_and_flipped_fields() {
        let dims = (4000u32, 3000u32);
        let mut seed = 7u64;
        for &rotation in &[0.0, 47.3, 211.0] {
            for &flipped in &[false, true] {
                seed += 1;
                let truth = Wcs::from_center_scale_rotation(
                    (150.5, 33.2),
                    (2000.0, 1500.0),
                    2.0,
                    rotation,
                    flipped,
                );
                let mut rng = Lcg(seed);
                let (catalog, detected) = synthetic_scene(&truth, dims, &mut rng);

                let hint = SolveHint {
                    center: (150.8, 33.0), // ~0.35° off
                    radius_deg: 1.5,
                    scale_arcsec_px: 2.2, // 10 % off
                    scale_tolerance: 0.25,
                };
                let solution = solve(&detected, &catalog, &hint, dims)
                    .unwrap_or_else(|e| panic!("rot {rotation} flip {flipped}: {e}"));
                assert_solution_matches(&truth, &solution, dims);
            }
        }
    }

    #[test]
    fn solves_near_the_pole() {
        let dims = (3000u32, 2000u32);
        let truth =
            Wcs::from_center_scale_rotation((100.0, 87.5), (1500.0, 1000.0), 3.0, 120.0, false);
        let mut rng = Lcg(42);
        let (catalog, detected) = synthetic_scene(&truth, dims, &mut rng);
        let hint = SolveHint {
            center: (99.0, 87.4),
            radius_deg: 2.0,
            scale_arcsec_px: 3.0,
            scale_tolerance: 0.2,
        };
        let solution = solve(&detected, &catalog, &hint, dims).unwrap();
        assert_solution_matches(&truth, &solution, dims);
    }

    #[test]
    fn fails_cleanly_on_the_wrong_sky_region() {
        let dims = (3000u32, 2000u32);
        let truth =
            Wcs::from_center_scale_rotation((150.5, 33.2), (1500.0, 1000.0), 2.0, 10.0, false);
        let mut rng = Lcg(9);
        let (catalog, detected) = synthetic_scene(&truth, dims, &mut rng);
        // Hint points 40° away — no overlap with the detections
        let hint = SolveHint {
            center: (190.0, 10.0),
            radius_deg: 1.0,
            scale_arcsec_px: 2.0,
            scale_tolerance: 0.2,
        };
        assert!(solve(&detected, &catalog, &hint, dims).is_err());
    }

    #[test]
    fn fails_cleanly_with_too_few_stars() {
        let catalog = MemoryCatalog::new(vec![]);
        let hint = SolveHint {
            center: (0.0, 0.0),
            radius_deg: 1.0,
            scale_arcsec_px: 1.0,
            scale_tolerance: 0.2,
        };
        assert!(solve(&[], &catalog, &hint, (100, 100)).is_err());
    }
}
