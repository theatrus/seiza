//! Blind plate solving: no position hint, only a plausible pixel-scale
//! range.
//!
//! The approach is hypothesis-and-verify. A [`BlindIndex`] holds 4-star
//! patterns built from the catalog's bright stars: each pattern's six
//! pairwise distances, divided by the largest, form a 5-value descriptor
//! that is invariant to rotation, scale, translation, and parity. Image
//! patterns built the same way look up candidates in a quantized hash of
//! that descriptor; every candidate implies a field center and pixel
//! scale, and the strongest hypotheses are handed to the hinted solver
//! ([`crate::solve::solve`]) for verification and refinement.

use crate::catalog::{CatalogStar, StarCatalog};
use crate::detect::DetectedStar;
use crate::solve::{Solution, SolveHint, solve};
use crate::wcs::Wcs;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Debug, Clone)]
pub struct BlindParams {
    /// Pixel-scale search range, arcseconds per pixel
    pub min_scale_arcsec_px: f64,
    pub max_scale_arcsec_px: f64,
    /// Catalog stars brighter than this build the pattern index
    pub index_mag_limit: f32,
    /// Longest allowed pattern edge on the sky, degrees. Should be around
    /// half of the smallest field of view you expect to solve.
    pub max_pattern_deg: f64,
    /// How many hypotheses to verify before giving up
    pub max_hypotheses: usize,
    /// Minimum matched stars to accept a blind verification — chance
    /// alignments of a handful of stars are common across a whole-sky
    /// search, so this must be stricter than the hinted solver's floor
    pub min_matches: usize,
    /// Maximum accepted RMS residual, in pixels (scale-relative — an
    /// arcsecond cap would reject coarse-scale images out of hand)
    pub max_rms_px: f64,
}

impl Default for BlindParams {
    fn default() -> Self {
        Self {
            min_scale_arcsec_px: 0.3,
            max_scale_arcsec_px: 20.0,
            index_mag_limit: 12.5,
            max_pattern_deg: 6.0,
            max_hypotheses: 400,
            min_matches: 12,
            max_rms_px: 2.0,
        }
    }
}

/// Pattern tiers: disc radius on the sky (degrees) paired with the
/// magnitude cap that keeps a typical disc at ~15-20 stars. Patterns are
/// anchored at stars that are the brightest within their disc; a field
/// that fully contains a disc sees the identical star set the index saw.
const TIERS: &[(f64, f32)] = &[
    (6.0, 6.1),
    (3.0, 7.6),
    (1.5, 9.2),
    (0.75, 10.7),
    (0.4, 11.8),
];
/// Descriptor quantization bins per dimension.
const BINS: f64 = 128.0;
/// Probe the neighboring bin when a descriptor value is this close to a
/// bin edge (descriptor units). Measured image-vs-catalog descriptor
/// noise on real frames is below 0.0007; this is ~3x that.
const PROBE_EPSILON: f64 = 0.002;
/// Descriptor match tolerance in bins (probes adjacent bins).
const N_IMAGE_STARS: usize = 32;

struct Pattern {
    /// Tangent-plane coordinates (degrees) of the four stars about the
    /// pattern centroid, in canonical vertex order
    points: [(f64, f64); 4],
    /// Pattern centroid on the sky
    center: (f64, f64),
    /// Longest pairwise separation, degrees
    max_edge_deg: f64,
}

/// Coarse (RA, Dec, log-scale) vote bucket for a field hypothesis.
type HypothesisKey = (i64, i64, i64);
/// Vote count with the implied field center and pixel scale.
type Hypothesis = (u32, (f64, f64), f64);

/// A searchable pattern index over a catalog's bright stars, frozen into
/// sorted arrays: hash-free branchless binary search per lookup, cache
/// friendly, and directly serializable.
pub struct BlindIndex {
    /// Sorted, unique descriptor keys
    keys: Vec<u64>,
    /// `keys[i]`'s candidates live at `candidates[starts[i]..starts[i+1]]`
    starts: Vec<u32>,
    candidates: Vec<u32>,
    patterns: Vec<Pattern>,
}

impl BlindIndex {
    fn lookup(&self, key: u64) -> &[u32] {
        match self.keys.binary_search(&key) {
            Ok(i) => &self.candidates[self.starts[i] as usize..self.starts[i + 1] as usize],
            Err(_) => &[],
        }
    }
}

impl BlindIndex {
    /// Build the index: for each sky cell in a ladder of cell sizes, all
    /// 4-star combinations of the cell's brightest stars. Any field of
    /// view at least twice a tier's cell size in both axes fully contains
    /// some cell of that tier, and therefore all of that cell's quads —
    /// which are quads of *locally brightest* stars, exactly what the
    /// image side enumerates from its brightest detections.
    pub fn build(catalog: &dyn StarCatalog, params: &BlindParams) -> Self {
        let mut stars = catalog.all_brighter_than(params.index_mag_limit);
        stars.sort_unstable_by(|a, b| a.mag.total_cmp(&b.mag));
        // Unit vectors once per star: every radius test below becomes a
        // dot-product compare, with no per-pair trigonometry
        let units: Vec<[f64; 3]> = stars.iter().map(|s| unit_vector(s.ra, s.dec)).collect();

        let mut patterns = Vec::new();
        let mut entries: Vec<(u64, u32)> = Vec::new();
        let mut seen: FxHashSet<[u32; 4]> = FxHashSet::default();

        for &(radius, mag_cap) in TIERS {
            let n_tier = stars.partition_point(|s| s.mag <= mag_cap);
            let tier = &stars[..n_tier];
            let tier_units = &units[..n_tier];
            let cos_radius = radius.to_radians().cos();
            let mut grid: FxHashMap<(i32, i32), Vec<u32>> = FxHashMap::default();
            for (i, s) in tier.iter().enumerate() {
                grid.entry(cell_key(s.ra, s.dec, radius))
                    .or_default()
                    .push(i as u32);
            }

            // Anchors are independent: generate each anchor's quads in
            // parallel, then merge sequentially (dedup + bucket insert)
            let anchor_quads: Vec<Vec<([u32; 4], Pattern)>> = (0..n_tier)
                .into_par_iter()
                .map(|i| {
                    let star = &tier[i];
                    let unit = tier_units[i];
                    let mut members: Vec<u32> = Vec::new();
                    let (cx, cy) = cell_key(star.ra, star.dec, radius);
                    for dx in -1..=1 {
                        for dy in -1..=1 {
                            let Some(cell) = grid.get(&(cx + dx, cy + dy)) else {
                                continue;
                            };
                            for &j in cell {
                                if j as usize == i {
                                    continue;
                                }
                                let other = tier_units[j as usize];
                                let dot =
                                    unit[0] * other[0] + unit[1] * other[1] + unit[2] * other[2];
                                if dot >= cos_radius {
                                    if (j as usize) < i {
                                        // A brighter star owns this disc
                                        return Vec::new();
                                    }
                                    members.push(j);
                                }
                            }
                        }
                    }
                    if members.len() < 3 {
                        return Vec::new();
                    }
                    members.sort_unstable();
                    members.truncate(5);
                    members.push(i as u32);
                    members.sort_unstable();

                    let mut quads = Vec::new();
                    let m = members.len();
                    for a in 0..m {
                        for b in a + 1..m {
                            for c in b + 1..m {
                                for d in c + 1..m {
                                    let ids = [members[a], members[b], members[c], members[d]];
                                    let quad = [
                                        (tier_units[ids[0] as usize], &tier[ids[0] as usize]),
                                        (tier_units[ids[1] as usize], &tier[ids[1] as usize]),
                                        (tier_units[ids[2] as usize], &tier[ids[2] as usize]),
                                        (tier_units[ids[3] as usize], &tier[ids[3] as usize]),
                                    ];
                                    let Some(pattern) = catalog_pattern(&quad) else {
                                        continue;
                                    };
                                    if pattern.max_edge_deg > params.max_pattern_deg {
                                        continue;
                                    }
                                    quads.push((ids, pattern));
                                }
                            }
                        }
                    }
                    quads
                })
                .collect();

            for quads in anchor_quads {
                for (ids, pattern) in quads {
                    if !seen.insert(ids) {
                        continue;
                    }
                    entries.push((
                        descriptor_key(&descriptor(&pattern.points)),
                        patterns.len() as u32,
                    ));
                    patterns.push(pattern);
                }
            }
        }

        entries.par_sort_unstable();
        let mut keys = Vec::new();
        let mut starts = vec![0u32];
        let mut candidates = Vec::with_capacity(entries.len());
        for (key, index) in entries {
            if keys.last() != Some(&key) {
                keys.push(key);
                starts.push(candidates.len() as u32);
            }
            candidates.push(index);
            *starts.last_mut().unwrap() = candidates.len() as u32;
        }

        Self {
            keys,
            starts,
            candidates,
            patterns,
        }
    }

    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }
}

/// Solve with no position hint. `stars` must be sorted brightest-first
/// (as produced by [`crate::detect::detect_stars`]). Hypotheses are
/// verified in parallel across CPU cores.
pub fn solve_blind(
    stars: &[DetectedStar],
    catalog: &(dyn StarCatalog + Sync),
    index: &BlindIndex,
    params: &BlindParams,
    dimensions: (u32, u32),
) -> Result<Solution, crate::Error> {
    if stars.len() < 6 {
        return Err(crate::Error::Solve(format!(
            "only {} stars detected; need at least 6",
            stars.len()
        )));
    }

    // Image patterns: each of the brightest stars with 3-subsets of its
    // nearest bright neighbors, mirroring the index construction. The
    // catalog's bright stars are usually the *locally* brightest
    // detections even in processed images where galaxy cores or nebula
    // knots dominate globally — so pick with spatial uniformity (capped
    // per grid cell), exactly as the index density-caps its stars, then
    // try a ladder of subset sizes.
    let (width, height) = (dimensions.0 as f64, dimensions.1 as f64);
    let mut per_cell = FxHashMap::default();
    let all_picked: Vec<(f64, f64)> = stars
        .iter()
        .filter(|s| {
            let cell = (
                ((s.x / width * 6.0) as i32).min(5),
                ((s.y / height * 6.0) as i32).min(5),
            );
            let count = per_cell.entry(cell).or_insert(0u32);
            *count += 1;
            *count <= 2
        })
        .take(N_IMAGE_STARS)
        .map(|s| (s.x, s.y))
        .collect();
    let mut hypotheses: FxHashMap<HypothesisKey, Hypothesis> = FxHashMap::default();
    let mut seen_quads = FxHashSet::default();
    let mut stat_quads = 0u64;
    let mut stat_candidates = 0u64;
    let mut stat_scale_ok = 0u64;

    // All 4-combinations of the brightest picks, in a ladder of subset
    // sizes. Index patterns are built from a SPARSE bright tier (5-nearest
    // among mag <= index_mag_limit spans degrees), so matching image quads
    // are wide — combinations of the brightest detections, not
    // nearest-neighbor cliques among a dense detection list.
    for k in [8usize, 10, 12, 16, 20, 26, 32] {
        let picked = &all_picked[..k.min(all_picked.len())];
        let n = picked.len();
        for i in 0..n {
            for a in i + 1..n {
                for b in a + 1..n {
                    for c in b + 1..n {
                        if !seen_quads.insert([i, a, b, c]) {
                            continue;
                        }
                        let quad = [picked[i], picked[a], picked[b], picked[c]];
                        let Some((points, max_edge_px)) = canonical_quad(&quad) else {
                            continue;
                        };
                        stat_quads += 1;
                        let desc = descriptor(&points);
                        for key in descriptor_keys(&desc) {
                            let candidates = index.lookup(key);
                            stat_candidates += candidates.len() as u64;
                            for &candidate in candidates {
                                let pattern = &index.patterns[candidate as usize];
                                // Implied pixel scale from the size ratio
                                let scale = pattern.max_edge_deg * 3600.0 / max_edge_px;
                                if scale < params.min_scale_arcsec_px
                                    || scale > params.max_scale_arcsec_px
                                {
                                    continue;
                                }
                                stat_scale_ok += 1;
                                // The canonical vertex order gives a tentative
                                // correspondence: fit the affine and read off
                                // a precise implied field center
                                let Some(implied) = implied_center(
                                    &points,
                                    pattern,
                                    (dimensions.0 as f64 / 2.0, dimensions.1 as f64 / 2.0),
                                    scale,
                                ) else {
                                    continue;
                                };
                                // Coarse buckets so multiple correct quads
                                // merge votes and outrank one-off noise
                                let bucket = (
                                    (implied.0 * 2.0) as i64,
                                    (implied.1 * 2.0) as i64,
                                    (scale.ln() * 10.0) as i64,
                                );
                                let entry = hypotheses.entry(bucket).or_insert((0, implied, scale));
                                entry.0 += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    // Locally-windowed quads: dense fields bury a sky cell's brightest
    // stars far below the global brightest, so anchor a window at each
    // picked star across a ladder of radii and enumerate quads of the
    // window's brightest members. Windows whose anchor is not their own
    // brightest member are skipped (that region is covered by the window
    // anchored at the brighter star).
    let min_dim = width.min(height);
    let all_stars: Vec<(f64, f64)> = stars.iter().map(|s| (s.x, s.y)).collect();
    let mut seen_windows = FxHashSet::default();
    for anchor in 0..all_stars.len() {
        let (ax, ay) = all_stars[anchor];
        for divisor in [16.0, 11.0, 8.0, 5.6, 4.0, 2.8, 2.0] {
            let radius = min_dim / divisor;
            // The brightest five detections around this one — the anchor
            // itself need not make the cut: if a catalog anchor star went
            // undetected (star-shrink processing, saturation), windows
            // around its surviving neighbors still reproduce the rest of
            // its set, which the 6-member index patterns cover
            let mut window: Vec<usize> = Vec::new();
            for (j, &(x, y)) in all_stars.iter().enumerate() {
                if (x - ax).hypot(y - ay) <= radius {
                    window.push(j);
                    if window.len() == 5 {
                        break;
                    }
                }
            }
            if window.len() < 4 {
                continue;
            }
            let n = window.len();
            for a in 0..n {
                for b in a + 1..n {
                    for c in b + 1..n {
                        for d in c + 1..n {
                            let ids = [window[a], window[b], window[c], window[d]];
                            if !seen_windows.insert(ids) {
                                continue;
                            }
                            let quad = [
                                all_stars[ids[0]],
                                all_stars[ids[1]],
                                all_stars[ids[2]],
                                all_stars[ids[3]],
                            ];
                            let Some((points, max_edge_px)) = canonical_quad(&quad) else {
                                continue;
                            };
                            stat_quads += 1;
                            let desc = descriptor(&points);
                            for key in descriptor_keys(&desc) {
                                let candidates = index.lookup(key);
                                stat_candidates += candidates.len() as u64;
                                for &candidate in candidates {
                                    let pattern = &index.patterns[candidate as usize];
                                    let scale = pattern.max_edge_deg * 3600.0 / max_edge_px;
                                    if scale < params.min_scale_arcsec_px
                                        || scale > params.max_scale_arcsec_px
                                    {
                                        continue;
                                    }
                                    let Some(implied) = implied_center(
                                        &points,
                                        pattern,
                                        (width / 2.0, height / 2.0),
                                        scale,
                                    ) else {
                                        continue;
                                    };
                                    let bucket = (
                                        (implied.0 * 2.0) as i64,
                                        (implied.1 * 2.0) as i64,
                                        (scale.ln() * 10.0) as i64,
                                    );
                                    let entry =
                                        hypotheses.entry(bucket).or_insert((0, implied, scale));
                                    entry.0 += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if std::env::var("SEIZA_DEBUG").is_ok() {
        eprintln!(
            "blind-funnel: {stat_quads} image quads, {stat_candidates} candidates, \
             {stat_scale_ok} scale-ok"
        );
    }
    // Correct quads vote for the same field but their implied centers
    // straddle bucket boundaries; sum each bucket with its neighbors so
    // split votes recombine while uniform junk stays flat
    let mut smoothed: Vec<(HypothesisKey, u32, (f64, f64), f64)> = hypotheses
        .iter()
        .map(|(&key, &(_, implied, scale))| {
            let mut sum = 0;
            for dx in -1..=1i64 {
                for dy in -1..=1i64 {
                    for dz in -1..=1i64 {
                        if let Some(&(v, _, _)) =
                            hypotheses.get(&(key.0 + dx, key.1 + dy, key.2 + dz))
                        {
                            sum += v;
                        }
                    }
                }
            }
            (key, sum, implied, scale)
        })
        .collect();
    smoothed.sort_by_key(|entry| std::cmp::Reverse(entry.1));
    // Non-max suppression: a strong region floods all its neighbor
    // buckets with the same summed vote; verify each region once
    let mut taken: FxHashSet<HypothesisKey> = FxHashSet::default();
    let mut ranked: Vec<(u32, (f64, f64), f64)> = Vec::new();
    for (key, votes, implied, scale) in smoothed {
        if taken.contains(&key) {
            continue;
        }
        for dx in -2..=2i64 {
            for dy in -2..=2i64 {
                for dz in -1..=1i64 {
                    taken.insert((key.0 + dx, key.1 + dy, key.2 + dz));
                }
            }
        }
        ranked.push((votes, implied, scale));
    }
    if let Ok(truth) = std::env::var("SEIZA_DEBUG_TRUTH") {
        let parts: Vec<f64> = truth.split(',').filter_map(|v| v.parse().ok()).collect();
        if parts.len() == 2 {
            let near = ranked.iter().enumerate().find(|(_, h)| {
                crate::catalog::angular_separation_deg(h.1.0, h.1.1, parts[0], parts[1]) < 1.5
            });
            eprintln!(
                "blind-debug: {} hypotheses; nearest-to-truth rank {:?}",
                ranked.len(),
                near.map(|(rank, h)| (rank, h.0, h.1, h.2))
            );
        }
    }

    ranked.truncate(params.max_hypotheses);

    // Verify hypotheses in vote order, in parallel batches sized to the
    // core count — the true field usually ranks near the top, so small
    // batches keep the win cheap while parallelism absorbs the misses.
    // Blind acceptance is stricter than the hinted floor: across a
    // whole-sky search, chance alignments of a few stars are routine.
    let batch = rayon::current_num_threads().max(1);
    for chunk in ranked.chunks(batch) {
        let solution = chunk.par_iter().find_map_any(|&(_votes, center, scale)| {
            let hint = SolveHint {
                center,
                // The implied center is precise; a tight radius keeps
                // each verification fast
                radius_deg: 0.4,
                scale_arcsec_px: scale,
                scale_tolerance: 0.15,
            };
            solve(stars, catalog, &hint, dimensions).ok().filter(|s| {
                s.matched_stars >= params.min_matches && s.rms_arcsec < params.max_rms_px * scale
            })
        });
        if let Some(solution) = solution {
            return Ok(solution);
        }
    }

    Err(crate::Error::Solve(format!(
        "no hypothesis verified (tried {})",
        ranked.len()
    )))
}

/// Fit the quad correspondence and project the image center onto the sky.
/// The canonical vertex order gives the correspondence; wrong matches are
/// rejected by shear and scale disagreement.
fn implied_center(
    image_points: &[(f64, f64); 4],
    pattern: &Pattern,
    image_center: (f64, f64),
    scale_arcsec_px: f64,
) -> Option<(f64, f64)> {
    let pairs: Vec<((f64, f64), (f64, f64))> = image_points
        .iter()
        .copied()
        .zip(pattern.points.iter().copied())
        .collect();
    let (a, b, c, d, e, f) = crate::solve::fit_affine(&pairs)?;
    let c1 = a.hypot(d);
    let c2 = b.hypot(e);
    if c1 <= 0.0 || c2 <= 0.0 {
        return None;
    }
    let ortho = (a * b + d * e).abs() / (c1 * c2);
    if ortho > 0.08 || (c1 / c2 - 1.0).abs() > 0.08 {
        return None;
    }
    let fitted_scale = (c1 * c2).sqrt() * 3600.0;
    if (fitted_scale / scale_arcsec_px - 1.0).abs() > 0.08 {
        return None;
    }

    let tangent = Wcs {
        crval: pattern.center,
        crpix: (0.0, 0.0),
        cd: [[1.0, 0.0], [0.0, 1.0]],
    };
    let x = a * image_center.0 + b * image_center.1 + c;
    let y = d * image_center.0 + e * image_center.1 + f;
    Some(tangent.pixel_to_world(x, y))
}

/// Sky cell for a position: declination band index plus an RA bin whose
/// width is scaled by the band's cosine so cells stay roughly square.
fn cell_key(ra: f64, dec: f64, cell_deg: f64) -> (i32, i32) {
    let band = ((dec + 90.0) / cell_deg) as i32;
    let band_mid_dec = (band as f64 + 0.5) * cell_deg - 90.0;
    let ra_width = cell_deg / band_mid_dec.to_radians().cos().max(0.05);
    (band, (ra.rem_euclid(360.0) / ra_width) as i32)
}

/// Unit vector for an ICRS position.
fn unit_vector(ra: f64, dec: f64) -> [f64; 3] {
    let (sin_ra, cos_ra) = ra.to_radians().sin_cos();
    let (sin_dec, cos_dec) = dec.to_radians().sin_cos();
    [cos_dec * cos_ra, cos_dec * sin_ra, sin_dec]
}

/// Project a catalog quad to the tangent plane about its centroid and
/// canonicalize. Works entirely from precomputed unit vectors: the
/// centroid is the normalized vector sum, and the gnomonic projection is
/// a pair of dot products per star.
fn catalog_pattern(quad: &[([f64; 3], &CatalogStar)]) -> Option<Pattern> {
    let mut center = [0.0f64; 3];
    for (u, _) in quad {
        center[0] += u[0];
        center[1] += u[1];
        center[2] += u[2];
    }
    let norm = (center[0] * center[0] + center[1] * center[1] + center[2] * center[2]).sqrt();
    if norm <= 0.0 {
        return None;
    }
    for v in &mut center {
        *v /= norm;
    }
    // Local east/north tangent basis at the centroid
    let horizontal = center[0].hypot(center[1]);
    if horizontal < 1e-12 {
        // Degenerate exactly at a pole; fall back to an arbitrary basis
        return None;
    }
    let east = [-center[1] / horizontal, center[0] / horizontal, 0.0];
    let north = [-center[2] * east[1], center[2] * east[0], horizontal];

    let mut points = [(0.0, 0.0); 4];
    for (i, (u, _)) in quad.iter().enumerate() {
        let depth = u[0] * center[0] + u[1] * center[1] + u[2] * center[2];
        if depth <= 0.0 {
            return None;
        }
        let xi = (u[0] * east[0] + u[1] * east[1] + u[2] * east[2]) / depth;
        let eta = (u[0] * north[0] + u[1] * north[1] + u[2] * north[2]) / depth;
        points[i] = (xi.to_degrees(), eta.to_degrees());
    }
    let (points, max_edge_deg) = canonical_quad(&points)?;
    Some(Pattern {
        points,
        center: (
            center[1].atan2(center[0]).to_degrees().rem_euclid(360.0),
            center[2].asin().to_degrees(),
        ),
        max_edge_deg,
    })
}

/// Order quad vertices canonically (by total distance to the other three)
/// and return the longest edge.
fn canonical_quad(quad: &[(f64, f64); 4]) -> Option<([(f64, f64); 4], f64)> {
    let mut totals = [0.0f64; 4];
    let mut max_edge = 0.0f64;
    for i in 0..4 {
        for j in i + 1..4 {
            let d = (quad[i].0 - quad[j].0).hypot(quad[i].1 - quad[j].1);
            totals[i] += d;
            totals[j] += d;
            max_edge = max_edge.max(d);
        }
    }
    if max_edge <= 0.0 {
        return None;
    }
    let mut order = [0usize, 1, 2, 3];
    order.sort_by(|&a, &b| totals[a].total_cmp(&totals[b]));
    Some((
        [
            quad[order[0]],
            quad[order[1]],
            quad[order[2]],
            quad[order[3]],
        ],
        max_edge,
    ))
}

/// Five sorted edge-length ratios (each in (0, 1]): the pattern descriptor.
fn descriptor(points: &[(f64, f64); 4]) -> [f64; 5] {
    let mut edges = [0.0f64; 6];
    let mut k = 0;
    for i in 0..4 {
        for j in i + 1..4 {
            edges[k] = (points[i].0 - points[j].0).hypot(points[i].1 - points[j].1);
            k += 1;
        }
    }
    edges.sort_by(|a, b| a.total_cmp(b));
    let max = edges[5].max(1e-12);
    [
        edges[0] / max,
        edges[1] / max,
        edges[2] / max,
        edges[3] / max,
        edges[4] / max,
    ]
}

/// The home bucket key for a descriptor: floor-quantized bins packed
/// 8 bits per dimension (insertion side).
fn descriptor_key(desc: &[f64; 5]) -> u64 {
    let mut packed = 0u64;
    for &v in desc {
        packed = packed << 8 | ((v * BINS) as i64).clamp(0, 255) as u64;
    }
    packed
}

/// Query keys for a descriptor: the home bucket, plus the neighbor bin in
/// any dimension whose value lies within [`PROBE_EPSILON`] of a bin edge —
/// measurement noise can only flip a bin near its boundary, so distant
/// dimensions need no probing (typically 1-4 keys instead of 3^5).
fn descriptor_keys(desc: &[f64; 5]) -> Vec<u64> {
    let delta = PROBE_EPSILON * BINS;
    let mut options: [[i64; 2]; 5] = [[0, i64::MIN]; 5];
    let mut bins = [0i64; 5];
    for (dim, &v) in desc.iter().enumerate() {
        let scaled = v * BINS;
        let bin = scaled as i64;
        bins[dim] = bin;
        let frac = scaled - bin as f64;
        options[dim][0] = bin;
        if frac < delta && bin > 0 {
            options[dim][1] = bin - 1;
        } else if frac > 1.0 - delta && bin < 255 {
            options[dim][1] = bin + 1;
        }
    }
    let mut keys = Vec::with_capacity(4);
    let counts = options.map(|o| if o[1] == i64::MIN { 1usize } else { 2 });
    let total: usize = counts.iter().product();
    for combo in 0..total {
        let mut rest = combo;
        let mut packed = 0u64;
        for dim in 0..5 {
            let pick = rest % counts[dim];
            rest /= counts[dim];
            packed = packed << 8 | options[dim][pick].clamp(0, 255) as u64;
        }
        keys.push(packed);
    }
    keys
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::catalog::MemoryCatalog;

    pub(crate) struct Lcg(pub u64);
    impl Lcg {
        pub(crate) fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    pub(crate) fn whole_sky_catalog(rng: &mut Lcg) -> MemoryCatalog {
        // A bright "sky" plus a fainter background
        let mut stars = Vec::new();
        // Approximates the real sky's ~9000 stars brighter than mag 6.5
        for _ in 0..9000 {
            stars.push(CatalogStar {
                ra: rng.next() * 360.0,
                dec: (2.0 * rng.next() - 1.0).asin().to_degrees(),
                mag: 2.0 + rng.next() as f32 * 4.5, // bright: 2.0..6.5
            });
        }
        for _ in 0..40000 {
            stars.push(CatalogStar {
                ra: rng.next() * 360.0,
                dec: (2.0 * rng.next() - 1.0).asin().to_degrees(),
                mag: 6.5 + rng.next() as f32 * 5.5,
            });
        }
        MemoryCatalog::new(stars)
    }

    pub(crate) fn detections_for(
        truth: &Wcs,
        catalog: &MemoryCatalog,
        dims: (u32, u32),
        rng: &mut Lcg,
    ) -> Vec<DetectedStar> {
        let mut detected = Vec::new();
        for s in catalog.all_brighter_than(30.0) {
            if let Some((x, y)) = truth.world_to_pixel(s.ra, s.dec)
                && x > 0.0
                && y > 0.0
                && x < dims.0 as f64
                && y < dims.1 as f64
            {
                if rng.next() < 0.15 {
                    continue;
                }
                detected.push(DetectedStar {
                    x: x + (rng.next() - 0.5) * 0.6,
                    y: y + (rng.next() - 0.5) * 0.6,
                    flux: 10f64.powf(-0.4 * s.mag as f64) * 1e6,
                    peak: 0.5,
                    area: 20,
                });
            }
        }
        detected.sort_by(|a, b| b.flux.total_cmp(&a.flux));
        detected
    }

    #[test]
    fn blind_solves_an_unknown_field() {
        let mut rng = Lcg(99);
        let catalog = whole_sky_catalog(&mut rng);
        let dims = (4000u32, 3000u32);
        // A wide field somewhere on the sky, never disclosed to the solver
        let truth =
            Wcs::from_center_scale_rotation((212.4, -35.7), (2000.0, 1500.0), 6.0, 74.0, false);
        let detected = detections_for(&truth, &catalog, dims, &mut rng);
        assert!(
            detected.len() > 25,
            "test scene too sparse: {}",
            detected.len()
        );

        let params = BlindParams {
            min_scale_arcsec_px: 1.0,
            max_scale_arcsec_px: 15.0,
            ..Default::default()
        };
        let index = BlindIndex::build(&catalog, &params);
        assert!(index.pattern_count() > 1000);

        let solution = solve_blind(&detected, &catalog, &index, &params, dims).unwrap();
        let (ra, dec) = solution.wcs.pixel_to_world(2000.0, 1500.0);
        let sep = crate::catalog::angular_separation_deg(ra, dec, 212.4, -35.7);
        assert!(sep < 0.01, "center off by {sep}°");
        assert!((solution.wcs.scale_arcsec_per_px() - 6.0).abs() < 0.05);
    }

    #[test]
    fn blind_fails_cleanly_on_noise() {
        let mut rng = Lcg(7);
        let catalog = whole_sky_catalog(&mut rng);
        let params = BlindParams::default();
        let index = BlindIndex::build(&catalog, &params);

        // Pure noise detections
        let detected: Vec<DetectedStar> = (0..40)
            .map(|_| DetectedStar {
                x: rng.next() * 4000.0,
                y: rng.next() * 3000.0,
                flux: rng.next() * 1000.0,
                peak: 0.5,
                area: 12,
            })
            .collect();
        assert!(solve_blind(&detected, &catalog, &index, &params, (4000, 3000)).is_err());
    }
}

#[cfg(test)]
mod debug_tests {
    use crate::solve::SolveHint;
    use crate::wcs::Wcs;

    #[test]
    fn hinted_verify_works_at_this_scale() {
        let mut rng = super::tests::Lcg(99);
        let catalog = super::tests::whole_sky_catalog(&mut rng);
        let dims = (4000u32, 3000u32);
        let truth =
            Wcs::from_center_scale_rotation((212.4, -35.7), (2000.0, 1500.0), 6.0, 74.0, false);
        let detected = super::tests::detections_for(&truth, &catalog, dims, &mut rng);
        eprintln!("detections: {}", detected.len());

        let hint = SolveHint {
            center: (212.4, -35.7),
            radius_deg: 0.4,
            scale_arcsec_px: 6.0,
            scale_tolerance: 0.15,
        };
        let solution = crate::solve::solve(&detected, &catalog, &hint, dims);
        eprintln!("hinted with truth: {solution:?}");
        assert!(solution.is_ok());
    }
}
