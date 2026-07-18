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
    /// SIP distortion polynomial order to fit (0 or 1 = linear solution
    /// only). Orders 2..=5 fit forward and inverse polynomials when enough
    /// matched stars support them and the fit reduces the residual.
    pub sip_order: u8,
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
// A center-window fast path must be stricter than the general hinted floor:
// if the mount hint is stale, a chance six-star alignment must not prevent
// the requested surrounding search. These are the same acceptance thresholds
// used when the blind solver verifies a whole-sky hypothesis.
const FAST_HINT_MIN_INLIERS: usize = 12;
const FAST_HINT_MAX_RMS_PX: f64 = 2.0;

/// Solve the field for an image of `(width, height)` given detected stars,
/// a catalog, and a hint.
pub fn solve(
    stars: &[DetectedStar],
    catalog: &dyn StarCatalog,
    hint: &SolveHint,
    dimensions: (u32, u32),
) -> Result<Solution, crate::Error> {
    // Mount coordinates are usually already inside the image. Try that one
    // FOV-sized window before constructing triangles for every window in the
    // fallback radius. A 2-degree search on a fine-scale frame can contain
    // dozens of windows and millions of catalog triangles.
    // When the requested radius is smaller than one window step, the fallback
    // already contains only the center window. Do not run it twice; blind
    // verification deliberately uses such tightly localized hints.
    let window_step_deg = fov_radius_deg(hint, dimensions).max(0.05);
    if hint.radius_deg >= window_step_deg {
        let center_hint = SolveHint {
            center: hint.center,
            radius_deg: 0.0,
            scale_arcsec_px: hint.scale_arcsec_px,
            scale_tolerance: hint.scale_tolerance,
            sip_order: hint.sip_order,
        };
        if let Ok(solution) = solve_search(stars, catalog, &center_hint, dimensions)
            && solution.matched_stars >= FAST_HINT_MIN_INLIERS
            && solution.rms_arcsec < FAST_HINT_MAX_RMS_PX * hint.scale_arcsec_px
        {
            return Ok(solution);
        }
    }

    match solve_search(stars, catalog, hint, dimensions) {
        Ok(solution) => Ok(solution),
        // Triangle matching assumes the list's flux ordering tracks real
        // brightness; when it fails, retry with the rank-robust quad
        // search that treats ordering as a prior only.
        Err(triangle_error) => {
            let tables = RankRobustTables::build(stars, dimensions);
            solve_rank_robust(stars, catalog, hint, dimensions, &tables, RR_PROBE_BUDGET).map_err(
                |robust_error| crate::Error::Solve(format!("{triangle_error}; {robust_error}")),
            )
        }
    }
}

/// [`solve`] for blind hypothesis verification: identical semantics, but
/// the caller decides whether this hypothesis has earned the rank-robust
/// fallback. A whole-sky search verifies hundreds of junk hypotheses per
/// image, so those pass `None` and fail at triangle cost; the few whose
/// position-only coarse score shows real catalog agreement pass prebuilt
/// tables and get the full search.
pub(crate) fn solve_for_blind_hypothesis(
    stars: &[DetectedStar],
    catalog: &dyn StarCatalog,
    hint: &SolveHint,
    dimensions: (u32, u32),
    tables: Option<&RankRobustTables>,
) -> Result<Solution, crate::Error> {
    match solve_search(stars, catalog, hint, dimensions) {
        Ok(solution) => Ok(solution),
        Err(triangle_error) => {
            let Some(tables) = tables else {
                return Err(triangle_error);
            };
            solve_rank_robust(stars, catalog, hint, dimensions, tables, RR_PROBE_BUDGET).map_err(
                |robust_error| crate::Error::Solve(format!("{triangle_error}; {robust_error}")),
            )
        }
    }
}

/// Run the triangle matcher across every FOV-sized window in `hint`.
///
/// [`solve`] first calls this with a zero radius for the common accurate-hint
/// case, then preserves the caller's full search radius as a fallback.
fn solve_search(
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
    let fov_radius_deg = fov_radius_deg(hint, dimensions);

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
    let grid = CatGrid::build(&cat, match_tol);
    let mut best: Option<(usize, Affine)> = None;
    for (_, affine) in &candidates {
        let inliers = count_inliers(affine, &vote_stars, &cat, &grid, match_tol);
        if best.as_ref().is_none_or(|(count, _)| inliers > *count) {
            best = Some((inliers, affine.clone()));
        }
    }
    let (votes, affine) = best.unwrap();
    if votes < MIN_INLIERS.min(vote_stars.len() * 2 / 3) {
        return Err(crate::Error::Solve(format!(
            "best candidate matched only {votes} of {} stars",
            vote_stars.len()
        )));
    }

    refine_to_solution(
        affine, stars, &cat, &grid, match_tol, &tangent, hint, dimensions,
    )
}

/// Refine one plausible pixel-to-tangent affine into a complete solution:
/// iterative re-matching with least-squares fits, re-centering about the
/// image center, quality metrics, and the optional SIP stage. Shared by
/// the triangle path and the rank-robust quad path.
#[allow(clippy::too_many_arguments)]
fn refine_to_solution(
    mut affine: Affine,
    stars: &[DetectedStar],
    cat: &[(f64, f64, CatalogStar)],
    grid: &CatGrid,
    match_tol: f64,
    tangent: &Wcs,
    hint: &SolveHint,
    dimensions: (u32, u32),
) -> Result<Solution, crate::Error> {
    let (width, height) = (dimensions.0 as f64, dimensions.1 as f64);
    // Iterative least-squares refinement with re-matching
    let all_stars: Vec<(f64, f64)> = stars.iter().map(|s| (s.x, s.y)).collect();
    let mut pairs = Vec::new();
    for _ in 0..REFINE_ITERATIONS {
        pairs = match_pairs(&affine, &all_stars, cat, grid, match_tol);
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
    let mut wcs = Wcs {
        crval,
        crpix,
        cd: [
            [final_affine.a, final_affine.b],
            [final_affine.d, final_affine.e],
        ],
        sip: None,
    };

    // Quality metrics against the final WCS
    let rms = |wcs: &Wcs| {
        let mut sum_sq = 0.0;
        for &((px, py), cat_idx) in &pairs {
            let star = &cat[cat_idx].2;
            let (ra, dec) = wcs.pixel_to_world(px, py);
            let sep = angular_separation_deg(ra, dec, star.ra, star.dec) * 3600.0;
            sum_sq += sep * sep;
        }
        (sum_sq / pairs.len() as f64).sqrt()
    };
    let mut rms_arcsec = rms(&wcs);

    let mut matched_stars = pairs.len();

    // Optional SIP distortion fit. Fit on the current inliers, then
    // re-match against the distorted model and fit again: stars whose
    // distortion pushed them past the linear matching tolerance only
    // become available once the polynomial predicts their positions.
    // Accepted only when the polynomial beats the linear solution over the
    // same matched set; the linear solution otherwise stands.
    if hint.sip_order >= 2 {
        let linear = wcs.clone();
        let mut fit_set: Vec<PointPair> = final_pairs.clone();
        for _ in 0..2 {
            let Some(candidate) = fit_sip(hint.sip_order, &wcs, &fit_set, dimensions) else {
                break;
            };
            let matched = match_sky_pairs(&candidate, stars, cat, &recentre, dimensions);
            if matched.len() < MIN_INLIERS || matched.len() < fit_set.len() {
                break;
            }
            let sum_sq_over = |wcs: &Wcs| {
                matched
                    .iter()
                    .map(|pair| {
                        let (ra, dec) = wcs.pixel_to_world(pair.pixel.0, pair.pixel.1);
                        let sep = angular_separation_deg(ra, dec, pair.sky.0, pair.sky.1) * 3600.0;
                        sep * sep
                    })
                    .sum::<f64>()
            };
            // In-sample residuals always shrink when parameters are added,
            // so compare degrees-of-freedom-corrected residuals: the
            // polynomial must explain more than its extra coefficients buy
            // for free, or a sparse noisy field would always "improve".
            let observations = 2.0 * matched.len() as f64;
            let candidate_parameters = candidate.sip.as_ref().map_or(6.0, |sip| {
                2.0 * crate::wcs::Sip::inverse_terms(sip.order).len() as f64
            });
            let candidate_sum_sq = sum_sq_over(&candidate);
            let reduced_candidate = candidate_sum_sq / (observations - candidate_parameters);
            let reduced_linear = sum_sq_over(&linear) / (observations - 6.0);
            if !reduced_candidate.is_finite() || reduced_candidate >= reduced_linear {
                break;
            }
            wcs = candidate;
            rms_arcsec = (candidate_sum_sq / matched.len() as f64).sqrt();
            matched_stars = matched.len();
            fit_set = matched
                .iter()
                .map(|pair| (pair.pixel, pair.tangent))
                .collect();
        }
    }

    Ok(Solution {
        wcs,
        matched_stars,
        rms_arcsec,
    })
}

// --------------------------------------------------------------------------
// Rank-robust fallback: catalog-seeded quads verified against a position
// hash over the full star list. Used when triangle matching fails, which
// happens when the input list's flux ordering does not track photometric
// brightness (see docs/design/rank-robust-matching.md). Brightness is used
// only as a search-ordering prior here, never as a correctness assumption.

/// Minimum stars for the fallback to be worth running.
const RR_MIN_STARS: usize = 30;
/// Stars (in given order, a soft prior) admitted to the pair table.
const RR_MAX_PAIR_STARS: usize = 2500;
/// Bright catalog window stars forming quads.
const RR_CATALOG_WINDOW: usize = 100;
/// Positional tolerance for quad-star probes, pixels.
const RR_PROBE_TOL_PX: f64 = 2.5;
/// Cheap-count acceptance floor before full refinement runs.
const RR_COUNT_MIN: usize = 10;
/// Total quads examined before giving up.
const RR_MAX_QUADS: usize = 400;
/// Probe budget per rank-robust search: bounds the worst case on an
/// unsolvable field to a few seconds of CPU, well under the cost of the
/// blind fallback that follows a failed hinted solve.
const RR_PROBE_BUDGET: usize = 200_000_000;

/// Star-list-side tables for the rank-robust search: the position hash
/// and the length-sorted pair table. They depend only on the detected
/// star list, so blind verification builds them once per image and
/// reuses them across every hypothesis.
pub(crate) struct RankRobustTables {
    pair_stars: Vec<(f64, f64)>,
    /// `(length_px, i, j)` sorted by length so a backbone length selects
    /// a candidate window by binary search.
    pairs: Vec<(f32, u32, u32)>,
    probe_grid: ImageGrid,
}

impl RankRobustTables {
    pub(crate) fn build(stars: &[DetectedStar], dimensions: (u32, u32)) -> Self {
        let probe_grid = ImageGrid::build(stars, RR_PROBE_TOL_PX);
        // Pair table over the (prior-ordered) star list.
        let pair_stars: Vec<(f64, f64)> = stars
            .iter()
            .take(RR_MAX_PAIR_STARS)
            .map(|s| (s.x, s.y))
            .collect();
        let diag_px = (dimensions.0 as f64).hypot(dimensions.1 as f64);
        let mut pairs: Vec<(f32, u32, u32)> = Vec::new();
        for i in 0..pair_stars.len() {
            for j in (i + 1)..pair_stars.len() {
                let length =
                    (pair_stars[i].0 - pair_stars[j].0).hypot(pair_stars[i].1 - pair_stars[j].1);
                if length > 8.0 && length <= diag_px {
                    pairs.push((length as f32, i as u32, j as u32));
                }
            }
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        Self {
            pair_stars,
            pairs,
            probe_grid,
        }
    }
}

/// Hinted solve that does not trust the star list's flux ordering.
fn solve_rank_robust(
    stars: &[DetectedStar],
    catalog: &dyn StarCatalog,
    hint: &SolveHint,
    dimensions: (u32, u32),
    tables: &RankRobustTables,
    probe_budget: usize,
) -> Result<Solution, crate::Error> {
    if stars.len() < RR_MIN_STARS {
        return Err(crate::Error::Solve(format!(
            "only {} stars; rank-robust matching needs at least {RR_MIN_STARS}",
            stars.len()
        )));
    }
    let scale_deg = hint.scale_arcsec_px / 3600.0;
    let fov_radius_deg = fov_radius_deg(hint, dimensions);
    let search_radius = hint.radius_deg + fov_radius_deg;
    let cone = catalog.cone_search(hint.center.0, hint.center.1, search_radius, 400);
    if cone.len() < 8 {
        return Err(crate::Error::Solve(format!(
            "only {} catalog stars within {search_radius:.2}\u{b0} of the hint",
            cone.len()
        )));
    }
    let tangent = tangent_wcs(hint.center);
    let cat: Vec<(f64, f64, CatalogStar)> = cone
        .iter()
        .filter_map(|s| tangent.world_to_pixel(s.ra, s.dec).map(|(x, y)| (x, y, *s)))
        .collect();
    let match_tol = MATCH_TOLERANCE_PX * scale_deg;
    let grid = CatGrid::build(&cat, match_tol);

    // Catalog quads: each bright window star anchors quads with three of
    // its four nearest bright neighbors. Catalog brightness is reliable,
    // so these quads are near-certainly present among the image stars.
    let window: Vec<(f64, f64)> = cat
        .iter()
        .take(RR_CATALOG_WINDOW)
        .map(|&(x, y, _)| (x, y))
        .collect();
    let mut quads: Vec<[(f64, f64); 4]> = Vec::new();
    for (anchor_index, &anchor) in window.iter().enumerate() {
        let mut neighbors: Vec<(f64, (f64, f64))> = window
            .iter()
            .enumerate()
            .filter(|&(other, _)| other != anchor_index)
            .map(|(_, &p)| ((p.0 - anchor.0).hypot(p.1 - anchor.1), p))
            .collect();
        neighbors.sort_by(|a, b| a.0.total_cmp(&b.0));
        let near: Vec<(f64, f64)> = neighbors.iter().take(4).map(|&(_, p)| p).collect();
        if near.len() < 3 {
            continue;
        }
        for skip in 0..near.len() {
            let mut quad = [anchor; 4];
            let mut slot = 1;
            for (index, &p) in near.iter().enumerate() {
                if index != skip && slot < 4 {
                    quad[slot] = p;
                    slot += 1;
                }
            }
            if slot == 4 {
                quads.push(quad);
            }
        }
    }
    quads.truncate(RR_MAX_QUADS);

    let RankRobustTables {
        pair_stars,
        pairs,
        probe_grid,
    } = tables;
    let scale_low = hint.scale_arcsec_px * (1.0 - hint.scale_tolerance.min(0.9));
    let scale_high = hint.scale_arcsec_px * (1.0 + hint.scale_tolerance);
    let mut probes_left = probe_budget;

    for quad in &quads {
        // Backbone: the quad's widest pair; the other two verify.
        let mut backbone = (0, 1);
        let mut widest = 0.0;
        for a in 0..4 {
            for b in (a + 1)..4 {
                let d = (quad[a].0 - quad[b].0).hypot(quad[a].1 - quad[b].1);
                if d > widest {
                    widest = d;
                    backbone = (a, b);
                }
            }
        }
        let p1 = quad[backbone.0];
        let p2 = quad[backbone.1];
        let others: Vec<(f64, f64)> = (0..4)
            .filter(|&k| k != backbone.0 && k != backbone.1)
            .map(|k| quad[k])
            .collect();
        let backbone_deg = widest;
        // Pixel-length window implied by the scale tolerance.
        let low = (backbone_deg * 3600.0 / scale_high - RR_PROBE_TOL_PX).max(8.0) as f32;
        let high = (backbone_deg * 3600.0 / scale_low + RR_PROBE_TOL_PX) as f32;
        let from = pairs.partition_point(|pair| pair.0 < low);
        let to = pairs.partition_point(|pair| pair.0 <= high);

        for &(_, i, j) in &pairs[from..to] {
            let image_i = pair_stars[i as usize];
            let image_j = pair_stars[j as usize];
            for (a_img, b_img) in [(image_i, image_j), (image_j, image_i)] {
                for mirrored in [false, true] {
                    probes_left = probes_left.saturating_sub(1);
                    if probes_left == 0 {
                        return Err(crate::Error::Solve(format!(
                            "rank-robust search exhausted its {probe_budget}-probe budget"
                        )));
                    }
                    let Some(transform) =
                        SimilarityTransform::from_pair(p1, p2, a_img, b_img, mirrored)
                    else {
                        continue;
                    };
                    // Both remaining quad stars must land on image stars.
                    if !others
                        .iter()
                        .all(|&p| probe_grid.hit(transform.apply(p), RR_PROBE_TOL_PX))
                    {
                        continue;
                    }
                    // Cheap census before paying for refinement: only
                    // window stars the transform places inside the frame
                    // can match, so the floor adapts when the hint offset
                    // leaves few of them in view.
                    probes_left = probes_left.saturating_sub(window.len());
                    let margin = RR_PROBE_TOL_PX * 1.5;
                    let (width, height) = (dimensions.0 as f64, dimensions.1 as f64);
                    let mut in_frame = 0usize;
                    let mut count = 0usize;
                    for &p in &window {
                        let t = transform.apply(p);
                        if t.0 >= -margin
                            && t.1 >= -margin
                            && t.0 < width + margin
                            && t.1 < height + margin
                        {
                            in_frame += 1;
                            if probe_grid.hit(t, margin) {
                                count += 1;
                            }
                        }
                    }
                    if count < RR_COUNT_MIN.min((in_frame * 2 / 3).max(4)) {
                        continue;
                    }
                    // Convert tangent->pixel similarity into the solver's
                    // pixel->tangent affine and hand off to the shared
                    // refinement machinery.
                    let inverse = Affine::from_three_points(
                        &[
                            transform.apply((0.0, 0.0)),
                            transform.apply((0.01, 0.0)),
                            transform.apply((0.0, 0.01)),
                        ],
                        &[(0.0, 0.0), (0.01, 0.0), (0.0, 0.01)],
                    );
                    let Some(inverse) = inverse else { continue };
                    if let Ok(solution) = refine_to_solution(
                        inverse, stars, &cat, &grid, match_tol, &tangent, hint, dimensions,
                    ) && solution.matched_stars >= FAST_HINT_MIN_INLIERS
                        && solution.rms_arcsec < FAST_HINT_MAX_RMS_PX * hint.scale_arcsec_px
                    {
                        return Ok(solution);
                    }
                }
            }
        }
    }
    Err(crate::Error::Solve(
        "rank-robust search found no verified transform".to_string(),
    ))
}

/// A tangent-degrees to pixel similarity (rotation + uniform scale +
/// translation, optionally mirrored), built from one point correspondence
/// pair via complex ratios.
struct SimilarityTransform {
    a: (f64, f64),
    b: (f64, f64),
    mirrored: bool,
}

impl SimilarityTransform {
    fn from_pair(
        p1: (f64, f64),
        p2: (f64, f64),
        i1: (f64, f64),
        i2: (f64, f64),
        mirrored: bool,
    ) -> Option<Self> {
        let source = if mirrored {
            complex_sub(conj(p2), conj(p1))
        } else {
            complex_sub(p2, p1)
        };
        let target = complex_sub(i2, i1);
        let a = complex_div(target, source)?;
        let anchor = if mirrored { conj(p1) } else { p1 };
        let b = complex_sub(i1, complex_mul(a, anchor));
        Some(Self { a, b, mirrored })
    }

    fn apply(&self, p: (f64, f64)) -> (f64, f64) {
        let z = if self.mirrored { conj(p) } else { p };
        let t = complex_mul(self.a, z);
        (t.0 + self.b.0, t.1 + self.b.1)
    }
}

fn conj(z: (f64, f64)) -> (f64, f64) {
    (z.0, -z.1)
}

fn complex_sub(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 - b.0, a.1 - b.1)
}

fn complex_mul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn complex_div(a: (f64, f64), b: (f64, f64)) -> Option<(f64, f64)> {
    let norm = b.0 * b.0 + b.1 * b.1;
    if norm < 1e-24 {
        return None;
    }
    Some((
        (a.0 * b.0 + a.1 * b.1) / norm,
        (a.1 * b.0 - a.0 * b.1) / norm,
    ))
}

/// Position hash over image stars for O(1) tolerance-circle probes.
struct ImageGrid {
    cell: f64,
    cells: rustc_hash::FxHashMap<(i64, i64), Vec<u32>>,
    positions: Vec<(f64, f64)>,
}

impl ImageGrid {
    fn build(stars: &[DetectedStar], tolerance: f64) -> Self {
        let cell = (tolerance * 2.0).max(1.0);
        let mut cells: rustc_hash::FxHashMap<(i64, i64), Vec<u32>> =
            rustc_hash::FxHashMap::default();
        let mut positions = Vec::with_capacity(stars.len());
        for (index, star) in stars.iter().enumerate() {
            positions.push((star.x, star.y));
            cells
                .entry((
                    (star.x / cell).floor() as i64,
                    (star.y / cell).floor() as i64,
                ))
                .or_default()
                .push(index as u32);
        }
        Self {
            cell,
            cells,
            positions,
        }
    }

    fn hit(&self, point: (f64, f64), tolerance: f64) -> bool {
        let cx = (point.0 / self.cell).floor() as i64;
        let cy = (point.1 / self.cell).floor() as i64;
        let tol_sq = tolerance * tolerance;
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(indices) = self.cells.get(&(cx + dx, cy + dy)) {
                    for &index in indices {
                        let p = self.positions[index as usize];
                        let d = (p.0 - point.0).powi(2) + (p.1 - point.1).powi(2);
                        if d <= tol_sq {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}

/// One matched star for SIP refinement: the detected pixel position, the
/// exact tangent-plane target about the solution's `crval`, and the sky
/// position for residual metrics.
struct SkyPair {
    pixel: (f64, f64),
    tangent: (f64, f64),
    sky: (f64, f64),
}

/// Match detected stars to catalog stars by projecting the catalog through
/// a (possibly distorted) WCS into pixel space.
fn match_sky_pairs(
    wcs: &Wcs,
    stars: &[DetectedStar],
    cat: &[(f64, f64, CatalogStar)],
    recentre: &Wcs,
    dimensions: (u32, u32),
) -> Vec<SkyPair> {
    let (width, height) = (dimensions.0 as f64, dimensions.1 as f64);
    let margin = 2.0 * MATCH_TOLERANCE_PX;
    let mut proposals: Vec<(f64, usize, usize)> = Vec::new();
    for (cat_index, &(_, _, star)) in cat.iter().enumerate() {
        let Some((sx, sy)) = wcs.world_to_pixel(star.ra, star.dec) else {
            continue;
        };
        if sx < -margin || sy < -margin || sx > width + margin || sy > height + margin {
            continue;
        }
        let nearest = stars
            .iter()
            .enumerate()
            .map(|(star_index, s)| ((s.x - sx).hypot(s.y - sy), star_index))
            .min_by(|a, b| a.0.total_cmp(&b.0));
        if let Some((distance, star_index)) = nearest
            && distance <= MATCH_TOLERANCE_PX
        {
            proposals.push((distance, star_index, cat_index));
        }
    }
    // Greedy mutual exclusion, closest matches first: without it, two
    // nearby catalog stars can both claim one detected star, inflating the
    // match count and double-weighting that star in the fit and residual.
    proposals.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut used_star = vec![false; stars.len()];
    let mut used_cat = vec![false; cat.len()];
    let mut result = Vec::new();
    for (_, star_index, cat_index) in proposals {
        if used_star[star_index] || used_cat[cat_index] {
            continue;
        }
        let star = &cat[cat_index].2;
        let Some(tangent) = recentre.world_to_pixel(star.ra, star.dec) else {
            continue;
        };
        used_star[star_index] = true;
        used_cat[cat_index] = true;
        result.push(SkyPair {
            pixel: (stars[star_index].x, stars[star_index].y),
            tangent,
            sky: (star.ra, star.dec),
        });
    }
    result
}

/// Fit SIP distortion to the matched pairs of one accepted linear solution
/// and return the complete distorted WCS. `pairs` maps pixel positions to
/// exact tangent-plane coordinates (degrees) about `wcs.crval`.
///
/// The fit solves the full polynomial including constant and linear terms:
/// the linear stage's CD matrix has already absorbed part of the average
/// distortion scale, so the residual field contains a linear component the
/// (order >= 2) SIP terms cannot express. The fitted constant and linear
/// parts are folded back into CRPIX and CD, leaving a pure SIP polynomial.
/// Returns `None` when the pair count cannot support even a quadratic fit
/// or the fitted distortion is implausibly large.
fn fit_sip(order: u8, wcs: &Wcs, pairs: &[PointPair], dimensions: (u32, u32)) -> Option<Wcs> {
    use crate::wcs::Sip;

    // Reduce the order until the matched pairs comfortably constrain the
    // coefficient count; each axis needs its own fit.
    let mut order = order.min(5);
    let (terms, sip_terms) = loop {
        if order < 2 {
            return None;
        }
        // The fit basis includes constant and linear terms; `sip_terms` is
        // the (p + q >= 2) subset retained as SIP coefficients.
        let terms = Sip::inverse_terms(order);
        let sip_terms = Sip::forward_terms(order);
        if pairs.len() >= terms.len() * 3 {
            break (terms, sip_terms);
        }
        order -= 1;
    };

    let det = wcs.cd[0][0] * wcs.cd[1][1] - wcs.cd[0][1] * wcs.cd[1][0];
    if det.abs() < 1e-18 {
        return None;
    }
    // Normalize pixel offsets so high-order monomials stay well conditioned.
    let (width, height) = (dimensions.0 as f64, dimensions.1 as f64);
    let norm = width.hypot(height).max(1.0);
    let scale = 1.0 / norm;

    let mut rows = Vec::with_capacity(pairs.len());
    let mut target_u = Vec::with_capacity(pairs.len());
    let mut target_v = Vec::with_capacity(pairs.len());
    for &((px, py), (xi, eta)) in pairs {
        let u = px - wcs.crpix.0;
        let v = py - wcs.crpix.1;
        // The undistorted offsets the linear CD matrix would need to land
        // exactly on the catalog star.
        let cap_u = (wcs.cd[1][1] * xi - wcs.cd[0][1] * eta) / det;
        let cap_v = (-wcs.cd[1][0] * xi + wcs.cd[0][0] * eta) / det;
        rows.push(monomials(&terms, u * scale, v * scale));
        target_u.push(cap_u * scale);
        target_v.push(cap_v * scale);
    }
    let unscale = |coefficients: &[f64], terms: &[(u8, u8)]| -> Option<Vec<f64>> {
        let scaled = coefficients
            .iter()
            .zip(terms)
            .map(|(value, &(p, q))| value * scale.powi(i32::from(p + q) - 1))
            .collect::<Vec<_>>();
        scaled
            .iter()
            .all(|value| value.is_finite())
            .then_some(scaled)
    };
    let w_u = unscale(&solve_least_squares(&rows, &target_u)?, &terms)?;
    let w_v = unscale(&solve_least_squares(&rows, &target_v)?, &terms)?;

    // Split the full fit T(u, v) = c + M*(u, v) + H(u, v): the affine part
    // becomes CRPIX/CD corrections, the higher-order part becomes SIP.
    let coefficient = |values: &[f64], term: (u8, u8)| {
        terms
            .iter()
            .position(|&t| t == term)
            .map_or(0.0, |index| values[index])
    };
    let c = (coefficient(&w_u, (0, 0)), coefficient(&w_v, (0, 0)));
    let m = [
        [coefficient(&w_u, (1, 0)), coefficient(&w_u, (0, 1))],
        [coefficient(&w_v, (1, 0)), coefficient(&w_v, (0, 1))],
    ];
    let m_det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    // The affine correction must stay a refinement of the linear stage,
    // including its off-diagonals: an unchecked shear or rotation drift is
    // as much a warp as a scale drift.
    if m_det.abs() < 1e-12
        || (m[0][0] - 1.0).abs() > 0.05
        || (m[1][1] - 1.0).abs() > 0.05
        || m[0][1].abs() > 0.05
        || m[1][0].abs() > 0.05
    {
        return None;
    }
    let m_inverse = [
        [m[1][1] / m_det, -m[0][1] / m_det],
        [-m[1][0] / m_det, m[0][0] / m_det],
    ];
    // T = c + M u + H(u) = M (u + M^-1 c) + H(u): with the reference pixel
    // shifted by -M^-1 c and CD scaled by M, the remaining distortion is
    // M^-1 H expressed about the new reference. The shift is subpixel, so
    // evaluating H about the old reference is within the fit noise.
    let shift = (
        m_inverse[0][0] * c.0 + m_inverse[0][1] * c.1,
        m_inverse[1][0] * c.0 + m_inverse[1][1] * c.1,
    );
    if shift.0.hypot(shift.1) > 8.0 * MATCH_TOLERANCE_PX {
        return None;
    }
    let crpix = (wcs.crpix.0 - shift.0, wcs.crpix.1 - shift.1);
    let cd = [
        [
            wcs.cd[0][0] * m[0][0] + wcs.cd[0][1] * m[1][0],
            wcs.cd[0][0] * m[0][1] + wcs.cd[0][1] * m[1][1],
        ],
        [
            wcs.cd[1][0] * m[0][0] + wcs.cd[1][1] * m[1][0],
            wcs.cd[1][0] * m[0][1] + wcs.cd[1][1] * m[1][1],
        ],
    ];
    let mut a = Vec::with_capacity(sip_terms.len());
    let mut b = Vec::with_capacity(sip_terms.len());
    for &term in &sip_terms {
        let h = (coefficient(&w_u, term), coefficient(&w_v, term));
        a.push(m_inverse[0][0] * h.0 + m_inverse[0][1] * h.1);
        b.push(m_inverse[1][0] * h.0 + m_inverse[1][1] * h.1);
    }
    let mut sip = Sip {
        order,
        a,
        b,
        ap: Vec::new(),
        bp: Vec::new(),
    };

    // A polynomial that "improves" matched stars by warping the frame is
    // worse than no polynomial: cap the correction at the corners.
    let max_correction = norm * 0.02;
    for (x, y) in [
        (0.0, 0.0),
        (width - 1.0, 0.0),
        (width - 1.0, height - 1.0),
        (0.0, height - 1.0),
    ] {
        let (f, g) = sip.forward(x - crpix.0, y - crpix.1);
        if !f.is_finite() || !g.is_finite() || f.abs() > max_correction || g.abs() > max_correction
        {
            return None;
        }
    }

    // Inverse polynomials fitted over a uniform grid of forward-distorted
    // offsets, so `world_to_pixel` works everywhere in the frame, not just
    // at the matched stars.
    let inverse_terms = Sip::inverse_terms(order);
    let mut inverse_rows = Vec::new();
    let mut inverse_u = Vec::new();
    let mut inverse_v = Vec::new();
    const GRID: usize = 12;
    for gy in 0..GRID {
        for gx in 0..GRID {
            let u = (width - 1.0) * gx as f64 / (GRID - 1) as f64 - crpix.0;
            let v = (height - 1.0) * gy as f64 / (GRID - 1) as f64 - crpix.1;
            let (f, g) = sip.forward(u, v);
            let cap_u = u + f;
            let cap_v = v + g;
            inverse_rows.push(monomials(&inverse_terms, cap_u * scale, cap_v * scale));
            inverse_u.push((u - cap_u) * scale);
            inverse_v.push((v - cap_v) * scale);
        }
    }
    sip.ap = unscale(
        &solve_least_squares(&inverse_rows, &inverse_u)?,
        &inverse_terms,
    )?;
    sip.bp = unscale(
        &solve_least_squares(&inverse_rows, &inverse_v)?,
        &inverse_terms,
    )?;
    // The inverse is an approximation; a solution whose world_to_pixel
    // cannot reproduce its own forward mapping is not worth keeping.
    for gy in 0..GRID {
        for gx in 0..GRID {
            let u = (width - 1.0) * gx as f64 / (GRID - 1) as f64 - crpix.0;
            let v = (height - 1.0) * gy as f64 / (GRID - 1) as f64 - crpix.1;
            let (f, g) = sip.forward(u, v);
            let (cap_u, cap_v) = (u + f, v + g);
            let (fi, gi) = sip.inverse(cap_u, cap_v);
            if (cap_u + fi - u).hypot(cap_v + gi - v) > 0.1 {
                return None;
            }
        }
    }
    Some(Wcs {
        crval: wcs.crval,
        crpix,
        cd,
        sip: Some(sip),
    })
}

fn monomials(terms: &[(u8, u8)], u: f64, v: f64) -> Vec<f64> {
    terms
        .iter()
        .map(|&(p, q)| u.powi(i32::from(p)) * v.powi(i32::from(q)))
        .collect()
}

/// Solve `rows * x = targets` in the least-squares sense via normal
/// equations with partial-pivot Gaussian elimination. The systems here are
/// tiny (at most 20 unknowns) and the inputs are pre-normalized.
fn solve_least_squares(rows: &[Vec<f64>], targets: &[f64]) -> Option<Vec<f64>> {
    let n = rows.first()?.len();
    let mut normal = vec![vec![0.0; n + 1]; n];
    for (row, &target) in rows.iter().zip(targets) {
        for i in 0..n {
            for j in 0..n {
                normal[i][j] += row[i] * row[j];
            }
            normal[i][n] += row[i] * target;
        }
    }
    for column in 0..n {
        let pivot = (column..n)
            .max_by(|&a, &b| normal[a][column].abs().total_cmp(&normal[b][column].abs()))?;
        if normal[pivot][column].abs() < 1e-12 {
            return None;
        }
        normal.swap(column, pivot);
        let (pivot_rows, elimination_rows) = normal.split_at_mut(column + 1);
        let pivot_row = &pivot_rows[column];
        for row in elimination_rows {
            let factor = row[column] / pivot_row[column];
            for (target, &source) in row[column..=n].iter_mut().zip(&pivot_row[column..=n]) {
                *target -= factor * source;
            }
        }
    }
    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let mut value = normal[row][n];
        for k in row + 1..n {
            value -= normal[row][k] * solution[k];
        }
        solution[row] = value / normal[row][row];
    }
    solution
        .iter()
        .all(|value| value.is_finite())
        .then_some(solution)
}

fn fov_radius_deg(hint: &SolveHint, dimensions: (u32, u32)) -> f64 {
    let scale_deg = hint.scale_arcsec_px / 3600.0;
    (dimensions.0 as f64).hypot(dimensions.1 as f64) / 2.0
        * scale_deg
        * (1.0 + hint.scale_tolerance)
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

/// Least-squares affine fit over point pairs, exposed for the blind
/// solver's hypothesis generation: returns (a, b, c, d, e, f) for
/// (x, y) → (a x + b y + c, d x + e y + f).
pub(crate) fn fit_affine(pairs: &[PointPair]) -> Option<(f64, f64, f64, f64, f64, f64)> {
    let affine = Affine::fit(pairs)?;
    Some((affine.a, affine.b, affine.c, affine.d, affine.e, affine.f))
}

/// A WCS whose pixel plane IS the tangent plane (degrees) about `center`.
fn tangent_wcs(center: (f64, f64)) -> Wcs {
    Wcs {
        crval: center,
        crpix: (0.0, 0.0),
        cd: [[1.0, 0.0], [0.0, 1.0]],
        sip: None,
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

/// Uniform grid over catalog tangent-plane positions, cell size = match
/// tolerance: any point within tolerance of a query lies in the query's
/// 3x3 cell neighborhood, so matching is O(1) per star instead of a scan.
struct CatGrid {
    cell: f64,
    map: rustc_hash::FxHashMap<(i32, i32), Vec<u32>>,
}

impl CatGrid {
    fn build(cat: &[(f64, f64, CatalogStar)], cell: f64) -> Self {
        let mut map: rustc_hash::FxHashMap<(i32, i32), Vec<u32>> = rustc_hash::FxHashMap::default();
        for (i, &(x, y, _)) in cat.iter().enumerate() {
            map.entry(((x / cell) as i32, (y / cell) as i32))
                .or_default()
                .push(i as u32);
        }
        Self { cell, map }
    }

    /// Indices of catalog entries in the 3x3 neighborhood of `(u, v)`.
    fn near(&self, u: f64, v: f64) -> impl Iterator<Item = u32> + '_ {
        let (cx, cy) = ((u / self.cell) as i32, (v / self.cell) as i32);
        (-1..=1).flat_map(move |dx| {
            (-1..=1).flat_map(move |dy| {
                self.map
                    .get(&(cx + dx, cy + dy))
                    .map(|v| v.iter().copied())
                    .into_iter()
                    .flatten()
            })
        })
    }
}

fn count_inliers(
    affine: &Affine,
    stars: &[(f64, f64)],
    cat: &[(f64, f64, CatalogStar)],
    grid: &CatGrid,
    tolerance: f64,
) -> usize {
    let tol_sq = tolerance * tolerance;
    stars
        .iter()
        .filter(|&&(x, y)| {
            let (u, v) = affine.apply(x, y);
            grid.near(u, v).any(|ci| {
                let (cx, cy, _) = cat[ci as usize];
                (cx - u).powi(2) + (cy - v).powi(2) <= tol_sq
            })
        })
        .count()
}

/// Match each star to its nearest catalog star within tolerance; one catalog
/// star may serve at most one image star (greedy by distance).
fn match_pairs(
    affine: &Affine,
    stars: &[(f64, f64)],
    cat: &[(f64, f64, CatalogStar)],
    grid: &CatGrid,
    tolerance: f64,
) -> Vec<((f64, f64), usize)> {
    let tol_sq = tolerance * tolerance;
    let mut proposals: Vec<(f64, usize, usize)> = Vec::new();
    for (si, &(x, y)) in stars.iter().enumerate() {
        let (u, v) = affine.apply(x, y);
        let mut best: Option<(f64, usize)> = None;
        for ci in grid.near(u, v) {
            let (cx, cy, _) = cat[ci as usize];
            let d = (cx - u).powi(2) + (cy - v).powi(2);
            if d <= tol_sq && best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, ci as usize));
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
    use crate::catalog::{MemoryCatalog, StarCatalog};
    use std::cell::Cell;

    struct CountingCatalog {
        inner: MemoryCatalog,
        searches: Cell<usize>,
    }

    impl CountingCatalog {
        fn new(inner: MemoryCatalog) -> Self {
            Self {
                inner,
                searches: Cell::new(0),
            }
        }
    }

    impl StarCatalog for CountingCatalog {
        fn cone_search(
            &self,
            ra: f64,
            dec: f64,
            radius_deg: f64,
            limit: usize,
        ) -> Vec<CatalogStar> {
            self.searches.set(self.searches.get() + 1);
            self.inner.cone_search(ra, dec, radius_deg, limit)
        }
    }

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
                    sip_order: 0,
                };
                let solution = solve(&detected, &catalog, &hint, dims)
                    .unwrap_or_else(|e| panic!("rot {rotation} flip {flipped}: {e}"));
                assert_solution_matches(&truth, &solution, dims);
            }
        }
    }

    #[test]
    fn sip_fit_reduces_distorted_field_residuals() {
        let dims = (4000u32, 3000u32);
        let truth =
            Wcs::from_center_scale_rotation((150.5, 33.2), (2000.0, 1500.0), 2.0, 30.0, false);
        let mut rng = Lcg(99);
        // Barrel distortion about the image center: ~4.7 px at the corners,
        // well inside the matcher's tolerance near the center and a strong
        // systematic residual for the polynomial to absorb. The scene is
        // deliberately noise-free so the residual is purely the distortion.
        let distort = |x: f64, y: f64| {
            let (dx, dy) = (x - 2000.0, y - 1500.0);
            let factor = 1.0 + 3e-10 * (dx * dx + dy * dy);
            (2000.0 + dx * factor, 1500.0 + dy * factor)
        };
        let mut catalog = Vec::new();
        let mut detected = Vec::new();
        for _ in 0..2500 {
            let ra = truth.crval.0 + (rng.next() - 0.5) * 6.0;
            let dec = (truth.crval.1 + (rng.next() - 0.5) * 6.0).clamp(-89.9, 89.9);
            let mag = (4.0 + rng.next() * 8.0) as f32;
            catalog.push(CatalogStar { ra, dec, mag });
            if let Some((x, y)) = truth.world_to_pixel(ra, dec)
                && (20.0..dims.0 as f64 - 20.0).contains(&x)
                && (20.0..dims.1 as f64 - 20.0).contains(&y)
            {
                let (xd, yd) = distort(x, y);
                detected.push(DetectedStar {
                    x: xd,
                    y: yd,
                    flux: 10f64.powf(-0.4 * mag as f64) * 1e6,
                    peak: 0.5,
                    area: 20,
                });
            }
        }
        detected.sort_by(|a, b| b.flux.total_cmp(&a.flux));
        let catalog = MemoryCatalog::new(catalog);
        let hint = |sip_order| SolveHint {
            center: (150.5, 33.2),
            radius_deg: 0.5,
            scale_arcsec_px: 2.0,
            scale_tolerance: 0.2,
            sip_order,
        };
        let linear = solve(&detected, &catalog, &hint(0), dims).unwrap();
        assert!(linear.wcs.sip.is_none());
        let solution = solve(&detected, &catalog, &hint(3), dims).unwrap();
        let sip = solution.wcs.sip.as_ref().expect("SIP polynomials fitted");
        assert!(sip.order >= 2, "{sip:?}");
        assert!(
            solution.rms_arcsec < linear.rms_arcsec * 0.6,
            "linear rms {:.2}\" vs SIP rms {:.2}\"",
            linear.rms_arcsec,
            solution.rms_arcsec
        );

        // The solution must map distorted pixels onto true sky positions,
        // and its inverse polynomials must map those positions back.
        for &(x, y) in &[(400.0, 300.0), (3600.0, 2700.0), (2000.0, 320.0)] {
            let (ra_t, dec_t) = truth.pixel_to_world(x, y);
            let (xd, yd) = distort(x, y);
            let (ra_s, dec_s) = solution.wcs.pixel_to_world(xd, yd);
            let sep = angular_separation_deg(ra_t, dec_t, ra_s, dec_s) * 3600.0;
            assert!(sep < 1.5, "({x}, {y}) off by {sep:.2}\"");
            let (xi, yi) = solution.wcs.world_to_pixel(ra_t, dec_t).unwrap();
            let miss = (xi - xd).hypot(yi - yd);
            assert!(miss < 1.0, "inverse missed by {miss:.2} px");
        }
    }

    #[test]
    fn rank_robust_fallback_solves_a_flux_shuffled_field() {
        let dims = (4000u32, 3000u32);
        let truth =
            Wcs::from_center_scale_rotation((150.5, 33.2), (2000.0, 1500.0), 2.0, 25.0, false);
        let mut rng = Lcg(77);
        // The Siril condition: the image list goes much deeper than the
        // catalog's bright stars, and its flux ordering is uncorrelated
        // with real brightness — the brightest-N subsets share almost no
        // members with the catalog's top stars.
        let mut catalog = Vec::new();
        let mut detected = Vec::new();
        for _ in 0..6000 {
            let ra = truth.crval.0 + (rng.next() - 0.5) * 8.0;
            let dec = (truth.crval.1 + (rng.next() - 0.5) * 8.0).clamp(-89.9, 89.9);
            let mag = (4.0 + rng.next() * 8.0) as f32;
            // The catalog is magnitude-limited; the image list goes much
            // deeper, so a shuffled ranking floods the brightest-N image
            // subsets with stars the catalog does not contain.
            if mag < 8.0 {
                catalog.push(CatalogStar { ra, dec, mag });
            }
            if let Some((x, y)) = truth.world_to_pixel(ra, dec)
                && x > 20.0
                && y > 20.0
                && x < dims.0 as f64 - 20.0
                && y < dims.1 as f64 - 20.0
            {
                detected.push(DetectedStar {
                    x,
                    y,
                    flux: rng.next() * 1e6, // shuffled: unrelated to mag
                    peak: 0.5,
                    area: 10,
                });
            }
        }
        detected.sort_by(|a, b| b.flux.total_cmp(&a.flux));
        assert!(detected.len() > 100, "scene too sparse: {}", detected.len());
        let catalog = CountingCatalog::new(MemoryCatalog::new(catalog));

        let hint = SolveHint {
            center: (150.6, 33.1),
            radius_deg: 0.5,
            scale_arcsec_px: 2.0,
            scale_tolerance: 0.2,
            sip_order: 0,
        };
        let solution = solve(&detected, &catalog, &hint, dims).unwrap();
        assert_solution_matches(&truth, &solution, dims);
        assert!(
            catalog.searches.get() >= 2,
            "the shuffled ordering must defeat the triangle stage and be \
             solved by the rank-robust fallback"
        );
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
            sip_order: 0,
        };
        let solution = solve(&detected, &catalog, &hint, dims).unwrap();
        assert_solution_matches(&truth, &solution, dims);
    }

    #[test]
    fn strong_center_solution_skips_the_radius_grid() {
        let dims = (4000u32, 3000u32);
        let truth =
            Wcs::from_center_scale_rotation((150.5, 33.2), (2000.0, 1500.0), 2.0, 31.0, false);
        let mut rng = Lcg(101);
        let (catalog, detected) = synthetic_scene(&truth, dims, &mut rng);
        let catalog = CountingCatalog::new(catalog);
        let hint = SolveHint {
            center: truth.crval,
            radius_deg: 2.0,
            scale_arcsec_px: 2.0,
            scale_tolerance: 0.2,
            sip_order: 0,
        };

        let solution = solve(&detected, &catalog, &hint, dims).unwrap();
        assert_solution_matches(&truth, &solution, dims);
        assert_eq!(
            catalog.searches.get(),
            1,
            "a strong center solution should not search the fallback grid"
        );
    }

    #[test]
    fn stale_center_falls_back_to_the_requested_radius() {
        let dims = (4000u32, 3000u32);
        let truth =
            Wcs::from_center_scale_rotation((150.5, 33.2), (2000.0, 1500.0), 1.0, 73.0, true);
        let mut rng = Lcg(202);
        let (catalog, detected) = synthetic_scene(&truth, dims, &mut rng);
        let catalog = CountingCatalog::new(catalog);
        let hint = SolveHint {
            // About 1.5 degrees from the true center: outside the center
            // window, but inside the requested 2-degree fallback radius.
            center: (152.3, 33.2),
            radius_deg: 2.0,
            scale_arcsec_px: 1.0,
            scale_tolerance: 0.2,
            sip_order: 0,
        };

        let solution = solve(&detected, &catalog, &hint, dims).unwrap();
        assert_solution_matches(&truth, &solution, dims);
        assert!(
            catalog.searches.get() >= 2,
            "a stale center should run the full-radius fallback"
        );
    }

    #[test]
    fn one_window_failure_is_not_retried() {
        let dims = (4000u32, 3000u32);
        let truth =
            Wcs::from_center_scale_rotation((150.5, 33.2), (2000.0, 1500.0), 2.0, 10.0, false);
        let mut rng = Lcg(303);
        let (catalog, detected) = synthetic_scene(&truth, dims, &mut rng);
        let catalog = CountingCatalog::new(catalog);
        let hint = SolveHint {
            center: (190.0, 10.0),
            // Below the ~1.67-degree FOV window step, so this search contains
            // no surrounding grid windows to defer to a second pass.
            radius_deg: 0.4,
            scale_arcsec_px: 2.0,
            scale_tolerance: 0.2,
            sip_order: 0,
        };

        assert!(solve(&detected, &catalog, &hint, dims).is_err());
        // One triangle-stage search plus one rank-robust fallback search:
        // the guarantee is that the triangle stage never retries the same
        // center, not that the fallback is skipped.
        assert_eq!(
            catalog.searches.get(),
            2,
            "one triangle search and one rank-robust fallback search"
        );
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
            sip_order: 0,
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
            sip_order: 0,
        };
        assert!(solve(&[], &catalog, &hint, (100, 100)).is_err());
    }
}

#[cfg(test)]
mod sip_fit {
    use super::*;

    #[test]
    fn fit_sip_recovers_an_exact_cubic_distortion() {
        let s = 2.0 / 3600.0;
        let wcs = Wcs {
            crval: (150.0, 33.0),
            crpix: (2000.0, 1500.0),
            cd: [[-s, 0.0], [0.0, -s]],
            sip: None,
        };
        let k = 3e-10;
        let mut pairs: Vec<PointPair> = Vec::new();
        for gy in 0..20 {
            for gx in 0..20 {
                let u = (gx as f64 / 19.0) * 3999.0 - 2000.0;
                let v = (gy as f64 / 19.0) * 2999.0 - 1500.0;
                // True offsets (u, v); measured (distorted) offsets:
                let factor = 1.0 + k * (u * u + v * v);
                let (ud, vd) = (u * factor, v * factor);
                let (xi, eta) = (-s * u, -s * v);
                pairs.push(((2000.0 + ud, 1500.0 + vd), (xi, eta)));
            }
        }
        let full = fit_sip(3, &wcs, &pairs, (4000, 3000)).expect("fit");
        assert!(full.sip.is_some());
        let mut worst: f64 = 0.0;
        for &((px, py), (xi, eta)) in &pairs {
            let true_wcs = Wcs {
                sip: None,
                ..wcs.clone()
            };
            let (ra, dec) = true_wcs.pixel_to_world(2000.0 + (-xi / s), 1500.0 + (-eta / s));
            let (ra2, dec2) = full.pixel_to_world(px, py);
            let sep = angular_separation_deg(ra, dec, ra2, dec2) * 3600.0;
            worst = worst.max(sep);
        }
        assert!(worst < 0.05, "worst {worst}");
    }
}
