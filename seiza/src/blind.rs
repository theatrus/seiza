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
use memmap2::{Mmap, MmapOptions};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::ops::Range;
use std::path::Path;

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
            min_scale_arcsec_px: 0.1,
            max_scale_arcsec_px: 20.0,
            index_mag_limit: 12.7,
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
    // Sub-degree fields; only fills from catalogs deeper than Tycho-2
    (0.2, 12.7),
    (0.1, 14.2),
    (0.06, 16.0),
];
/// Descriptor quantization bins per dimension.
const BINS: f64 = 128.0;
/// Probe the neighboring bin when a descriptor value is this close to a
/// bin edge (descriptor units). Measured image-vs-catalog descriptor
/// noise on real frames is below 0.0007; this is ~3x that.
const PROBE_EPSILON: f64 = 0.002;
/// Descriptor match tolerance in bins (probes adjacent bins).
const N_IMAGE_STARS: usize = 32;
/// Bound the temporary per-anchor vectors created by parallel index
/// generation. Deep Gaia tiers contain tens of millions of stars; collecting
/// one `Vec` per anchor for an entire tier can otherwise consume gigabytes
/// before any patterns are merged.
const INDEX_ANCHOR_BATCH: usize = 65_536;
const INDEX_MAGIC: &[u8; 8] = b"SEIZABI1";
const INDEX_HEADER_SIZE: usize = 64;
const PATTERN_RECORD_SIZE: usize = 11 * size_of::<f32>();
const SERIALIZE_BATCH: usize = 65_536;
const INDEX_TIER_SCHEMA: u32 = 1;

#[derive(Clone, Copy)]
struct Pattern {
    /// Tangent-plane coordinates (degrees) of the four stars about the
    /// pattern centroid, in canonical vertex order
    points: [(f32, f32); 4],
    /// Pattern centroid on the sky
    center: (f32, f32),
    /// Longest pairwise separation, degrees
    max_edge_deg: f32,
}

impl Pattern {
    fn points_f64(&self) -> [(f64, f64); 4] {
        self.points.map(|(x, y)| (x as f64, y as f64))
    }
}

/// Coarse (RA, Dec, log-scale) vote bucket for a field hypothesis.
type HypothesisKey = (i64, i64, i64);
/// Vote count with the implied field center and pixel scale.
type Hypothesis = (u32, (f64, f64), f64, Wcs);
type SmoothedHypothesis = (HypothesisKey, u32, (f64, f64), f64, Wcs);
type RankedHypothesis = (usize, u32, (f64, f64), f64, Wcs);

/// A searchable pattern index over a catalog's bright stars, frozen into
/// sorted arrays: hash-free branchless binary search per lookup, cache
/// friendly, and directly serializable.
pub struct BlindIndex {
    index_mag_limit: f32,
    max_pattern_deg: f32,
    storage: BlindIndexStorage,
}

enum BlindIndexStorage {
    Built {
        keys: Vec<u64>,
        starts: Vec<u32>,
        candidates: Vec<u32>,
        patterns: Vec<Pattern>,
    },
    Mapped(MappedIndex),
}

struct MappedIndex {
    map: Mmap,
    keys_offset: usize,
    starts_offset: usize,
    candidates_offset: usize,
    patterns_offset: usize,
    keys_len: usize,
    candidates_len: usize,
    patterns_len: usize,
}

impl BlindIndex {
    fn keys_len(&self) -> usize {
        match &self.storage {
            BlindIndexStorage::Built { keys, .. } => keys.len(),
            BlindIndexStorage::Mapped(index) => index.keys_len,
        }
    }

    fn candidates_len(&self) -> usize {
        match &self.storage {
            BlindIndexStorage::Built { candidates, .. } => candidates.len(),
            BlindIndexStorage::Mapped(index) => index.candidates_len,
        }
    }

    fn key_at(&self, index: usize) -> u64 {
        match &self.storage {
            BlindIndexStorage::Built { keys, .. } => keys[index],
            BlindIndexStorage::Mapped(mapped) => {
                read_u64(&mapped.map, mapped.keys_offset + index * size_of::<u64>())
            }
        }
    }

    fn start_at(&self, index: usize) -> u32 {
        match &self.storage {
            BlindIndexStorage::Built { starts, .. } => starts[index],
            BlindIndexStorage::Mapped(mapped) => {
                read_u32(&mapped.map, mapped.starts_offset + index * size_of::<u32>())
            }
        }
    }

    fn candidate_at(&self, index: usize) -> u32 {
        match &self.storage {
            BlindIndexStorage::Built { candidates, .. } => candidates[index],
            BlindIndexStorage::Mapped(mapped) => read_u32(
                &mapped.map,
                mapped.candidates_offset + index * size_of::<u32>(),
            ),
        }
    }

    fn pattern_at(&self, index: usize) -> Pattern {
        match &self.storage {
            BlindIndexStorage::Built { patterns, .. } => patterns[index],
            BlindIndexStorage::Mapped(mapped) => {
                let mut offset = mapped.patterns_offset + index * PATTERN_RECORD_SIZE;
                let mut points = [(0.0, 0.0); 4];
                for point in &mut points {
                    point.0 = read_f32(&mapped.map, offset);
                    point.1 = read_f32(&mapped.map, offset + 4);
                    offset += 8;
                }
                let center = (
                    read_f32(&mapped.map, offset),
                    read_f32(&mapped.map, offset + 4),
                );
                Pattern {
                    points,
                    center,
                    max_edge_deg: read_f32(&mapped.map, offset + 8),
                }
            }
        }
    }

    fn lookup(&self, key: u64) -> Range<usize> {
        let mut left = 0;
        let mut right = self.keys_len();
        while left < right {
            let middle = left + (right - left) / 2;
            match self.key_at(middle).cmp(&key) {
                std::cmp::Ordering::Less => left = middle + 1,
                std::cmp::Ordering::Greater => right = middle,
                std::cmp::Ordering::Equal => {
                    return self.start_at(middle) as usize..self.start_at(middle + 1) as usize;
                }
            }
        }
        0..0
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
            let tier_pattern_start = patterns.len();
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

            // Anchors are independent: generate bounded batches in parallel,
            // then merge sequentially (dedup + bucket insert). Batching keeps
            // deep Gaia tiers from retaining one temporary Vec per catalog
            // star for the duration of the whole tier.
            for anchor_start in (0..n_tier).step_by(INDEX_ANCHOR_BATCH) {
                let anchor_end = (anchor_start + INDEX_ANCHOR_BATCH).min(n_tier);
                let anchor_quads: Vec<Vec<([u32; 4], Pattern)>> = (anchor_start..anchor_end)
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
                                    let dot = unit[0] * other[0]
                                        + unit[1] * other[1]
                                        + unit[2] * other[2];
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
                        // The deep tiers dominate index size. Five total
                        // members still provide five alternative quads while
                        // avoiding the 15 combinations produced by six.
                        members.truncate(if mag_cap > 12.7 { 4 } else { 5 });
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
                                        if pattern.max_edge_deg as f64 > params.max_pattern_deg {
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
                            descriptor_key(&descriptor(&pattern.points_f64())),
                            patterns.len() as u32,
                        ));
                        patterns.push(pattern);
                    }
                }
            }
            if std::env::var("SEIZA_DEBUG").is_ok() {
                eprintln!(
                    "blind-index-tier: radius={radius:.2} mag<={mag_cap:.1} patterns={}",
                    patterns.len() - tier_pattern_start
                );
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
            index_mag_limit: params.index_mag_limit,
            max_pattern_deg: params.max_pattern_deg as f32,
            storage: BlindIndexStorage::Built {
                keys,
                starts,
                candidates,
                patterns,
            },
        }
    }

    /// Open a versioned blind-pattern index without copying its arrays into
    /// memory. The file remains memory-mapped for the lifetime of the index.
    pub fn open(path: &Path) -> Result<Self, crate::Error> {
        let file = File::open(path).map_err(|error| invalid_index(path, error))?;
        // SAFETY: the map is read-only and owned by the returned index. A
        // caller replacing the file while it is mapped has the same platform
        // constraints as the existing memory-mapped star catalog.
        let map =
            unsafe { MmapOptions::new().map(&file) }.map_err(|error| invalid_index(path, error))?;
        if map.len() < INDEX_HEADER_SIZE || &map[..8] != INDEX_MAGIC {
            return Err(invalid_index(path, "not a SEIZABI1 blind index"));
        }

        let keys_len = usize::try_from(read_u64(&map, 8))
            .map_err(|_| invalid_index(path, "key count does not fit this platform"))?;
        let candidates_len = usize::try_from(read_u64(&map, 16))
            .map_err(|_| invalid_index(path, "candidate count does not fit this platform"))?;
        let patterns_len = usize::try_from(read_u64(&map, 24))
            .map_err(|_| invalid_index(path, "pattern count does not fit this platform"))?;
        let index_mag_limit = read_f32(&map, 32);
        let max_pattern_deg = read_f32(&map, 36);
        if !index_mag_limit.is_finite() || !max_pattern_deg.is_finite() || max_pattern_deg <= 0.0 {
            return Err(invalid_index(path, "invalid build parameters"));
        }
        if read_u32(&map, 40) != BINS as u32 || read_u32(&map, 44) != INDEX_TIER_SCHEMA {
            return Err(invalid_index(path, "unsupported descriptor or tier schema"));
        }
        if candidates_len > u32::MAX as usize || patterns_len > u32::MAX as usize {
            return Err(invalid_index(path, "index arrays exceed the format limits"));
        }

        let Some((keys_offset, starts_offset, candidates_offset, patterns_offset, end)) =
            index_layout(keys_len, candidates_len, patterns_len)
        else {
            return Err(invalid_index(path, "index size overflows this platform"));
        };
        if end != map.len() {
            return Err(invalid_index(
                path,
                format!("file length is {}, expected {end}", map.len()),
            ));
        }

        let start_at =
            |index: usize| read_u32(&map, starts_offset + index * size_of::<u32>()) as usize;
        let mut previous = 0;
        for index in 0..=keys_len {
            let current = start_at(index);
            if current < previous || current > candidates_len {
                return Err(invalid_index(path, "candidate offsets are not monotonic"));
            }
            previous = current;
        }
        if start_at(0) != 0 || start_at(keys_len) != candidates_len {
            return Err(invalid_index(
                path,
                "candidate offsets do not span the array",
            ));
        }
        for index in 1..keys_len {
            let prior = read_u64(&map, keys_offset + (index - 1) * size_of::<u64>());
            let current = read_u64(&map, keys_offset + index * size_of::<u64>());
            if prior >= current {
                return Err(invalid_index(
                    path,
                    "descriptor keys are not strictly sorted",
                ));
            }
        }
        for index in 0..candidates_len {
            let candidate = read_u32(&map, candidates_offset + index * size_of::<u32>()) as usize;
            if candidate >= patterns_len {
                return Err(invalid_index(
                    path,
                    "candidate refers past the pattern array",
                ));
            }
        }

        Ok(Self {
            index_mag_limit,
            max_pattern_deg,
            storage: BlindIndexStorage::Mapped(MappedIndex {
                map,
                keys_offset,
                starts_offset,
                candidates_offset,
                patterns_offset,
                keys_len,
                candidates_len,
                patterns_len,
            }),
        })
    }

    /// Persist the index in the little-endian `SEIZABI1` format. The
    /// resulting file is suitable for memory-mapped reuse and CDN hosting.
    pub fn write_to(&self, path: &Path) -> io::Result<()> {
        let mut out = BufWriter::with_capacity(4 * 1024 * 1024, File::create(path)?);
        let mut header = [0u8; INDEX_HEADER_SIZE];
        header[..8].copy_from_slice(INDEX_MAGIC);
        header[8..16].copy_from_slice(&(self.keys_len() as u64).to_le_bytes());
        header[16..24].copy_from_slice(&(self.candidates_len() as u64).to_le_bytes());
        header[24..32].copy_from_slice(&(self.pattern_count() as u64).to_le_bytes());
        header[32..36].copy_from_slice(&self.index_mag_limit.to_le_bytes());
        header[36..40].copy_from_slice(&self.max_pattern_deg.to_le_bytes());
        header[40..44].copy_from_slice(&(BINS as u32).to_le_bytes());
        header[44..48].copy_from_slice(&INDEX_TIER_SCHEMA.to_le_bytes());
        out.write_all(&header)?;

        let mut buffer = Vec::with_capacity(SERIALIZE_BATCH * PATTERN_RECORD_SIZE);
        for start in (0..self.keys_len()).step_by(SERIALIZE_BATCH) {
            buffer.clear();
            for index in start..(start + SERIALIZE_BATCH).min(self.keys_len()) {
                buffer.extend_from_slice(&self.key_at(index).to_le_bytes());
            }
            out.write_all(&buffer)?;
        }
        for start in (0..=self.keys_len()).step_by(SERIALIZE_BATCH) {
            buffer.clear();
            for index in start..(start + SERIALIZE_BATCH).min(self.keys_len() + 1) {
                buffer.extend_from_slice(&self.start_at(index).to_le_bytes());
            }
            out.write_all(&buffer)?;
        }
        for start in (0..self.candidates_len()).step_by(SERIALIZE_BATCH) {
            buffer.clear();
            for index in start..(start + SERIALIZE_BATCH).min(self.candidates_len()) {
                buffer.extend_from_slice(&self.candidate_at(index).to_le_bytes());
            }
            out.write_all(&buffer)?;
        }
        for start in (0..self.pattern_count()).step_by(SERIALIZE_BATCH) {
            buffer.clear();
            for index in start..(start + SERIALIZE_BATCH).min(self.pattern_count()) {
                let pattern = self.pattern_at(index);
                for (x, y) in pattern.points {
                    buffer.extend_from_slice(&x.to_le_bytes());
                    buffer.extend_from_slice(&y.to_le_bytes());
                }
                buffer.extend_from_slice(&pattern.center.0.to_le_bytes());
                buffer.extend_from_slice(&pattern.center.1.to_le_bytes());
                buffer.extend_from_slice(&pattern.max_edge_deg.to_le_bytes());
            }
            out.write_all(&buffer)?;
        }
        out.flush()
    }

    pub fn pattern_count(&self) -> usize {
        match &self.storage {
            BlindIndexStorage::Built { patterns, .. } => patterns.len(),
            BlindIndexStorage::Mapped(index) => index.patterns_len,
        }
    }

    pub fn index_mag_limit(&self) -> f32 {
        self.index_mag_limit
    }

    pub fn max_pattern_deg(&self) -> f64 {
        self.max_pattern_deg as f64
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn index_layout(
    keys_len: usize,
    candidates_len: usize,
    patterns_len: usize,
) -> Option<(usize, usize, usize, usize, usize)> {
    let keys_offset = INDEX_HEADER_SIZE;
    let starts_offset = keys_offset.checked_add(keys_len.checked_mul(size_of::<u64>())?)?;
    let candidates_offset =
        starts_offset.checked_add(keys_len.checked_add(1)?.checked_mul(size_of::<u32>())?)?;
    let patterns_offset =
        candidates_offset.checked_add(candidates_len.checked_mul(size_of::<u32>())?)?;
    let end = patterns_offset.checked_add(patterns_len.checked_mul(PATTERN_RECORD_SIZE)?)?;
    Some((
        keys_offset,
        starts_offset,
        candidates_offset,
        patterns_offset,
        end,
    ))
}

fn invalid_index(path: &Path, message: impl std::fmt::Display) -> crate::Error {
    crate::Error::Catalog(format!("invalid blind index {}: {message}", path.display()))
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
                            for candidate_offset in candidates {
                                let candidate = index.candidate_at(candidate_offset);
                                let pattern = index.pattern_at(candidate as usize);
                                // Implied pixel scale from the size ratio
                                let scale = pattern.max_edge_deg as f64 * 3600.0 / max_edge_px;
                                if scale < params.min_scale_arcsec_px
                                    || scale > params.max_scale_arcsec_px
                                {
                                    continue;
                                }
                                stat_scale_ok += 1;
                                // The canonical vertex order gives a tentative
                                // correspondence: fit the affine and read off
                                // a precise implied field center
                                let Some((implied, coarse_wcs)) = implied_wcs(
                                    &points,
                                    &pattern,
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
                                let entry = hypotheses
                                    .entry(bucket)
                                    .or_insert((0, implied, scale, coarse_wcs));
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
                                for candidate_offset in candidates {
                                    let candidate = index.candidate_at(candidate_offset);
                                    let pattern = index.pattern_at(candidate as usize);
                                    let scale = pattern.max_edge_deg as f64 * 3600.0 / max_edge_px;
                                    if scale < params.min_scale_arcsec_px
                                        || scale > params.max_scale_arcsec_px
                                    {
                                        continue;
                                    }
                                    let Some((implied, coarse_wcs)) = implied_wcs(
                                        &points,
                                        &pattern,
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
                                    let entry = hypotheses
                                        .entry(bucket)
                                        .or_insert((0, implied, scale, coarse_wcs));
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
    let mut smoothed: Vec<SmoothedHypothesis> = hypotheses
        .iter()
        .map(|(&key, (_, implied, scale, coarse_wcs))| {
            let mut sum = 0;
            for dx in -1..=1i64 {
                for dy in -1..=1i64 {
                    for dz in -1..=1i64 {
                        if let Some((v, _, _, _)) =
                            hypotheses.get(&(key.0 + dx, key.1 + dy, key.2 + dz))
                        {
                            sum += *v;
                        }
                    }
                }
            }
            (key, sum, *implied, *scale, coarse_wcs.clone())
        })
        .collect();
    smoothed.sort_by_key(|entry| std::cmp::Reverse(entry.1));
    // Non-max suppression: a strong region floods all its neighbor
    // buckets with the same summed vote; verify each region once
    let mut taken: FxHashSet<HypothesisKey> = FxHashSet::default();
    let mut ranked: Vec<(u32, (f64, f64), f64, Wcs)> = Vec::new();
    for (key, votes, implied, scale, coarse_wcs) in smoothed {
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
        ranked.push((votes, implied, scale, coarse_wcs));
    }

    // Deep, small-field tiers often yield only one or two matching quads;
    // vote-only ordering buries those among whole-sky descriptor collisions.
    // Re-rank every multi-vote region (and at least the verification budget)
    // by cheaply projecting catalog stars through the quad's coarse WCS.
    let score_count = ranked
        .partition_point(|hypothesis| hypothesis.0 >= 2)
        .max(params.max_hypotheses)
        .min(ranked.len());
    let detection_grid = DetectionGrid::new(
        stars.iter().take(200),
        (width.min(height) / 250.0).clamp(8.0, 32.0),
    );
    let mut ranked: Vec<RankedHypothesis> = ranked
        .into_par_iter()
        .take(score_count)
        .map(|(votes, center, scale, coarse_wcs)| {
            let matches = coarse_match_count(
                &coarse_wcs,
                center,
                scale,
                dimensions,
                catalog,
                &detection_grid,
            );
            (matches, votes, center, scale, coarse_wcs)
        })
        .collect();
    ranked.par_sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));

    if let Ok(truth) = std::env::var("SEIZA_DEBUG_TRUTH") {
        let parts: Vec<f64> = truth.split(',').filter_map(|v| v.parse().ok()).collect();
        if parts.len() == 2 {
            let near = ranked.iter().enumerate().min_by(|(_, a), (_, b)| {
                let a_sep =
                    crate::catalog::angular_separation_deg(a.2.0, a.2.1, parts[0], parts[1]);
                let b_sep =
                    crate::catalog::angular_separation_deg(b.2.0, b.2.1, parts[0], parts[1]);
                a_sep.total_cmp(&b_sep)
            });
            eprintln!(
                "blind-debug: {} hypotheses; nearest-to-truth rank {:?}",
                ranked.len(),
                near.map(|(rank, h)| (rank, h.0, h.1, h.2, h.3))
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
        let solution = chunk.par_iter().find_map_any(
            |(_coarse_matches, _votes, center, scale, _coarse_wcs)| {
                let hint = SolveHint {
                    center: *center,
                    // The implied center is precise; a tight radius keeps
                    // each verification fast
                    radius_deg: 0.4,
                    scale_arcsec_px: *scale,
                    scale_tolerance: 0.15,
                };
                solve(stars, catalog, &hint, dimensions).ok().filter(|s| {
                    s.matched_stars >= params.min_matches
                        && s.rms_arcsec < params.max_rms_px * *scale
                })
            },
        );
        if let Some(solution) = solution {
            return Ok(solution);
        }
    }

    Err(crate::Error::Solve(format!(
        "no hypothesis verified (tried {})",
        ranked.len()
    )))
}

struct DetectionGrid {
    cell_size: f64,
    cells: FxHashMap<(i32, i32), Vec<(f64, f64)>>,
}

impl DetectionGrid {
    fn new<'a>(stars: impl Iterator<Item = &'a DetectedStar>, cell_size: f64) -> Self {
        let mut cells: FxHashMap<(i32, i32), Vec<(f64, f64)>> = FxHashMap::default();
        for star in stars {
            cells
                .entry((
                    (star.x / cell_size).floor() as i32,
                    (star.y / cell_size).floor() as i32,
                ))
                .or_default()
                .push((star.x, star.y));
        }
        Self { cell_size, cells }
    }

    fn contains_near(&self, x: f64, y: f64) -> bool {
        let cell = (
            (x / self.cell_size).floor() as i32,
            (y / self.cell_size).floor() as i32,
        );
        let tolerance_sq = self.cell_size * self.cell_size;
        (-1..=1).any(|dx| {
            (-1..=1).any(|dy| {
                self.cells
                    .get(&(cell.0 + dx, cell.1 + dy))
                    .is_some_and(|points| {
                        points
                            .iter()
                            .any(|&(px, py)| (px - x).powi(2) + (py - y).powi(2) <= tolerance_sq)
                    })
            })
        })
    }
}

/// Cheaply score a quad-derived WCS before invoking the triangle solver.
/// Correct deep-field quads project many bright catalog stars near image
/// detections; random descriptor collisions almost always project zero or one.
fn coarse_match_count(
    wcs: &Wcs,
    center: (f64, f64),
    scale_arcsec_px: f64,
    dimensions: (u32, u32),
    catalog: &(dyn StarCatalog + Sync),
    detections: &DetectionGrid,
) -> usize {
    let radius_deg =
        (dimensions.0 as f64).hypot(dimensions.1 as f64) * 0.5 * scale_arcsec_px / 3600.0 * 1.1;
    catalog
        .cone_search(center.0, center.1, radius_deg, 512)
        .into_iter()
        .filter_map(|star| wcs.world_to_pixel(star.ra, star.dec))
        .filter(|&(x, y)| {
            x >= 0.0
                && y >= 0.0
                && x < dimensions.0 as f64
                && y < dimensions.1 as f64
                && detections.contains_near(x, y)
        })
        .count()
}

/// Fit the quad correspondence and project the image center onto the sky.
/// The canonical vertex order gives the correspondence; wrong matches are
/// rejected by shear and scale disagreement.
fn implied_wcs(
    image_points: &[(f64, f64); 4],
    pattern: &Pattern,
    image_center: (f64, f64),
    scale_arcsec_px: f64,
) -> Option<((f64, f64), Wcs)> {
    let pairs: Vec<((f64, f64), (f64, f64))> = image_points
        .iter()
        .copied()
        .zip(pattern.points_f64())
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

    let det = a * e - b * d;
    if det.abs() < 1e-18 {
        return None;
    }
    let wcs = Wcs {
        crval: (pattern.center.0 as f64, pattern.center.1 as f64),
        // cd * (pixel - crpix) is the fitted affine tangent plane.
        crpix: ((b * f - e * c) / det, (d * c - a * f) / det),
        cd: [[a, b], [d, e]],
    };
    Some((wcs.pixel_to_world(image_center.0, image_center.1), wcs))
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
        points: points.map(|(x, y)| (x as f32, y as f32)),
        center: (
            center[1].atan2(center[0]).to_degrees().rem_euclid(360.0) as f32,
            center[2].asin().to_degrees() as f32,
        ),
        max_edge_deg: max_edge_deg as f32,
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

    #[test]
    fn blind_solves_a_deep_small_field() {
        let mut rng = Lcg(0x5E1A_D33F);
        let mut stars = whole_sky_catalog(&mut rng).all_brighter_than(30.0);
        let center: (f64, f64) = (202.43, 47.20);
        let cos_dec = center.1.to_radians().cos();

        // A Gaia-like dense patch whose brightest stars are all fainter than
        // the old G<=12.7 cutoff. The 0.33 x 0.28 degree image is also smaller
        // than the old 0.2-degree-radius tier expects.
        for _ in 0..600 {
            stars.push(CatalogStar {
                ra: center.0 + (rng.next() - 0.5) * 0.8 / cos_dec,
                dec: center.1 + (rng.next() - 0.5) * 0.8,
                mag: 12.8 + rng.next() as f32 * 3.2,
            });
        }
        let catalog = MemoryCatalog::new(stars);
        let dims = (1200, 1000);
        let truth = Wcs::from_center_scale_rotation(
            center,
            (dims.0 as f64 / 2.0, dims.1 as f64 / 2.0),
            1.0,
            17.0,
            false,
        );
        let detected = detections_for(&truth, &catalog, dims, &mut rng);
        assert!(
            detected.len() > 40,
            "test scene too sparse: {}",
            detected.len()
        );

        let params = BlindParams {
            min_scale_arcsec_px: 0.5,
            max_scale_arcsec_px: 2.0,
            index_mag_limit: 16.0,
            ..Default::default()
        };
        let built = BlindIndex::build(&catalog, &params);
        let dir = std::env::temp_dir().join(format!(
            "seiza-blind-index-{}-{}",
            std::process::id(),
            rng.0
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("blind-gaia16.idx");
        built.write_to(&path).unwrap();
        let index = BlindIndex::open(&path).unwrap();
        assert_eq!(index.pattern_count(), built.pattern_count());
        assert_eq!(index.index_mag_limit(), 16.0);
        assert_eq!(index.max_pattern_deg(), params.max_pattern_deg);
        let solution = solve_blind(&detected, &catalog, &index, &params, dims).unwrap();
        let (ra, dec) = solution
            .wcs
            .pixel_to_world(dims.0 as f64 / 2.0, dims.1 as f64 / 2.0);
        let separation = crate::catalog::angular_separation_deg(ra, dec, center.0, center.1);
        assert!(separation < 0.01, "center off by {separation} degrees");
        assert!((solution.wcs.scale_arcsec_per_px() - 1.0).abs() < 0.05);
        std::fs::remove_dir_all(dir).ok();
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
