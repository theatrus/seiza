//! `--sky-map`: a labelled chart of a solved image.
//!
//! The figure shows the image under a dark title bar and footer, with
//! constellation stick figures and names, IAU-named stars, and deep-sky
//! objects drawn through the solved WCS. Every position comes from a catalog:
//! the marks say where things are, not that they were detected.
//!
//! Gathering ([`collect`]) works in source-image pixels and is separate
//! from drawing ([`render`]) so either can be tested alone.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use ab_glyph::FontRef;
use anyhow::{Context, Result};
use image::{Rgb, RgbImage};
use rayon::prelude::*;
use seiza::Wcs;
use seiza::constellations::{self, ProjectedFigure};
use seiza::data_paths;
use seiza::objects::{ObjectCatalog, ObjectKind, ObjectQuery, ObjectSort};
use seiza::star_ids::{StarIdentifierCatalog, StarNameCatalog, StarNameKind};
use seiza_draw::{Fonts, Mask, draw_text, measure, wrap};

/// Default output width, pixels.
pub(crate) const DEFAULT_WIDTH: u32 = 2100;
pub(crate) const MIN_WIDTH: u32 = 640;
pub(crate) const MAX_WIDTH: u32 = 12_000;

/// IAU-named stars fainter than this are left unlabelled.
const STAR_MAG_LIMIT: f32 = 4.5;
/// Unnamed figure stars at least this bright get their Bayer designation.
const BAYER_MAG_LIMIT: f32 = 3.0;
const MAX_STARS: usize = 28;
const MAX_OBJECTS: usize = 14;

const DSO_KINDS: [ObjectKind; 8] = [
    ObjectKind::Galaxy,
    ObjectKind::OpenCluster,
    ObjectKind::GlobularCluster,
    ObjectKind::Nebula,
    ObjectKind::PlanetaryNebula,
    ObjectKind::HiiRegion,
    ObjectKind::SupernovaRemnant,
    ObjectKind::ClusterWithNebula,
];

/// Greek letters as the Bright Star Catalogue abbreviates them, and in full.
const GREEK_LETTERS: [(&str, &str); 24] = [
    ("Alp", "Alpha"),
    ("Bet", "Beta"),
    ("Gam", "Gamma"),
    ("Del", "Delta"),
    ("Eps", "Epsilon"),
    ("Zet", "Zeta"),
    ("Eta", "Eta"),
    ("The", "Theta"),
    ("Iot", "Iota"),
    ("Kap", "Kappa"),
    ("Lam", "Lambda"),
    ("Mu", "Mu"),
    ("Nu", "Nu"),
    ("Xi", "Xi"),
    ("Omi", "Omicron"),
    ("Pi", "Pi"),
    ("Rho", "Rho"),
    ("Sig", "Sigma"),
    ("Tau", "Tau"),
    ("Ups", "Upsilon"),
    ("Phi", "Phi"),
    ("Chi", "Chi"),
    ("Psi", "Psi"),
    ("Ome", "Omega"),
];

const BACKGROUND: Rgb<u8> = Rgb([13, 19, 27]);
const FIGURE_LINE: Rgb<u8> = Rgb([140, 185, 235]);
const FIGURE_NAME: Rgb<u8> = Rgb([182, 196, 214]);
const STAR_GOLD: Rgb<u8> = Rgb([255, 212, 120]);
const OBJECT_CYAN: Rgb<u8> = Rgb([96, 222, 245]);
const SHADOW: Rgb<u8> = Rgb([0, 0, 0]);
const BRAND: Rgb<u8> = Rgb([92, 205, 235]);
const TITLE: Rgb<u8> = Rgb([240, 244, 248]);
const MUTED: Rgb<u8> = Rgb([160, 170, 182]);
const DIM: Rgb<u8> = Rgb([118, 128, 140]);

/// Catalogs the sky map labels from; either may be missing.
#[derive(Default)]
pub(crate) struct SkyMapCatalogs {
    pub objects: Option<ObjectCatalog>,
    pub star_ids: Option<StarIdentifierCatalog>,
}

impl SkyMapCatalogs {
    /// Find `objects.bin` and the star-identifier sidecar next to `--data`
    /// (a directory, or the directory holding a star file) or in the
    /// standard locations. A missing catalog costs its labels, not the map.
    pub(crate) fn resolve(data: Option<&Path>, objects: Option<&Path>) -> Self {
        let dir = data.map(|path| {
            if path.is_dir() {
                path.to_path_buf()
            } else {
                path.parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| PathBuf::from("."))
            }
        });
        // Next to --data first, then the standard locations.
        let objects_path = match objects {
            Some(path) => data_paths::objects(Some(path)),
            None => data_paths::objects(dir.as_deref()).or_else(|_| data_paths::objects(None)),
        };
        let objects = match objects_path {
            Ok(path) => ObjectCatalog::open(&path)
                .map_err(|error| eprintln!("sky map: {}: {error}", path.display()))
                .ok(),
            Err(_) => {
                report_missing("objects.bin", "deep-sky objects", dir.as_deref());
                None
            }
        };
        let star_ids = match data_paths::star_identifiers(dir.as_deref())
            .or_else(|_| data_paths::star_identifiers(None))
        {
            Ok(path) => StarIdentifierCatalog::open(&path)
                .map_err(|error| eprintln!("sky map: {}: {error}", path.display()))
                .ok(),
            Err(_) => {
                report_missing("stars-lite-tycho2.ids.bin", "star names", dir.as_deref());
                None
            }
        };
        Self { objects, star_ids }
    }
}

/// Say which file is missing, where it was looked for, and how to get it.
fn report_missing(file: &str, labels: &str, dir: Option<&Path>) {
    let target = dir.map_or_else(|| "<dir>".to_string(), |dir| dir.display().to_string());
    eprintln!(
        "sky map: no {file} next to --data or in the standard catalog locations, so {labels} \
         are left out; `seiza download-data prebuilt --output {target} --file {file}` fetches it"
    );
}

/// Solve statistics for the footer.
pub(crate) struct SolveSummary {
    /// Shown in the title bar, normally the image file stem.
    pub name: String,
    pub matched_stars: usize,
    pub rms_arcsec: f64,
}

/// A labelled star in source-image pixels.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StarMark {
    pub name: String,
    pub x: f64,
    pub y: f64,
    pub mag: Option<f32>,
}

/// A labelled deep-sky object in source-image pixels.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ObjectMark {
    pub label: String,
    pub x: f64,
    pub y: f64,
    /// The ellipse to draw; see [`crate::object_outline`].
    pub semi_major_px: f64,
    pub semi_minor_px: f64,
    pub angle_deg: f64,
}

/// Everything the map labels, in source-image pixels.
#[derive(Debug, Default)]
pub(crate) struct Annotations {
    pub figures: Vec<ProjectedFigure>,
    pub stars: Vec<StarMark>,
    pub objects: Vec<ObjectMark>,
    /// Foreground limit, when asked for and the detections show one.
    pub sky_floor: Option<SkyFloor>,
    /// Marks the map would show but the floor left out.
    pub hidden: usize,
}

/// What `--sky-map-foreground` works from: detections a catalog star
/// confirms, which outline the sky in a photo with foreground.
pub(crate) struct Foreground {
    /// Image pixels.
    pub detections: Vec<(f64, f64)>,
    /// How far a detection may sit from its catalog star, pixels.
    pub tolerance: f64,
}

impl Foreground {
    /// A star the image shows is kept even below the floor: a bright star
    /// in haze just above the horizon is still sky.
    fn seen(&self, x: f64, y: f64) -> bool {
        self.detections
            .iter()
            .any(|&(dx, dy)| (dx - x).hypot(dy - y) <= self.tolerance)
    }
}

/// Detections with a catalog star within `tolerance` pixels. Bright edges
/// on a lit foreground are detected too; these are real stars.
pub(crate) fn confirmed_detections(
    detected: &[(f64, f64)],
    catalog: &[(f64, f64)],
    tolerance: f64,
) -> Vec<(f64, f64)> {
    let limit = tolerance * tolerance;
    detected
        .iter()
        .copied()
        .filter(|&(x, y)| {
            catalog
                .iter()
                .any(|&(cx, cy)| (cx - x).powi(2) + (cy - y).powi(2) <= limit)
        })
        .collect()
}

/// Where the sky ends in a photo with foreground: below the lowest
/// detected stars of a run of column bands (plus a margin) there is
/// ground, a tree, or a building, and catalog marks there only add clutter.
///
/// A band counts only when the strip under its lowest stars is too wide
/// and too empty to be chance for stars spread over the whole frame, and
/// only in a run of several such bands, so a frame with stars everywhere
/// has no floor.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SkyFloor {
    width: f64,
    /// Floor y at the centre of each equal-width column band.
    floors: Vec<f64>,
}

impl SkyFloor {
    const BANDS: usize = 20;
    /// Fewer detections than this cannot outline the sky.
    const MIN_STARS: usize = 60;
    /// Lowest stars of a band neighbourhood that may be chance matches on
    /// the foreground.
    const MAX_SKIPPED: usize = 2;
    /// An empty strip narrower than this fraction of the height is not
    /// foreground: detection thins out at the frame edge anyway.
    const MIN_STRIP: f64 = 0.1;
    /// How likely the strip may be for stars spread evenly down the band.
    const MAX_CHANCE: f64 = 1e-3;
    /// How much less likely by chance leaving out a low star must make
    /// the strip before that star counts as a match on the foreground.
    const SKIP_FACTOR: f64 = 100.0;
    /// Consecutive bands that must show the strip.
    const MIN_RUN: usize = 5;
    /// Space left under the lowest stars, as a fraction of the height.
    const MARGIN: f64 = 0.02;

    /// `None` unless the detections leave a clear strip along the bottom
    /// of several neighbouring bands; a telescope image has none.
    pub(crate) fn from_detections(detected: &[(f64, f64)], dims: (u32, u32)) -> Option<Self> {
        let (width, height) = (dims.0 as f64, dims.1 as f64);
        if detected.len() < Self::MIN_STARS || width <= 0.0 || height <= 0.0 {
            return None;
        }
        let mut bands = vec![Vec::new(); Self::BANDS];
        for &(x, y) in detected {
            if inside(dims, x, y) {
                let band = ((x / width * Self::BANDS as f64) as usize).min(Self::BANDS - 1);
                bands[band].push(y);
            }
        }
        // The top of the clear strip under each band and its neighbours,
        // when there is one. The neighbours keep one sparse band from
        // cutting into the sky; an end band borrows both from the inside,
        // so it weighs as many stars as any other. The lowest star or two
        // may be chance matches on the foreground, so the strip may hold
        // that many: the lowest star counts unless leaving it out makes
        // the strip far less likely by chance, as a lone match deep in the
        // ground does.
        let mut strips = (0..Self::BANDS)
            .map(|band| {
                let first = band.saturating_sub(1).min(Self::BANDS - 3);
                let mut ys = (first..first + 3)
                    .flat_map(|index| bands[index].iter().copied())
                    .collect::<Vec<_>>();
                ys.sort_by(|a, b| b.total_cmp(a));
                let candidates = (0..=Self::MAX_SKIPPED)
                    .take_while(|&below| below < ys.len())
                    .filter_map(|below| {
                        let strip = (height - ys[below]) / height;
                        (strip >= Self::MIN_STRIP)
                            .then(|| (binomial_cdf(below, ys.len(), strip), ys[below]))
                    })
                    .collect::<Vec<_>>();
                let least = candidates
                    .iter()
                    .map(|&(chance, _)| chance)
                    .fold(f64::INFINITY, f64::min);
                candidates
                    .into_iter()
                    .find(|&(chance, _)| {
                        chance < Self::MAX_CHANCE && chance <= least * Self::SKIP_FACTOR
                    })
                    .map(|(_, top)| top)
            })
            .collect::<Vec<_>>();
        let mut start = 0;
        while start < Self::BANDS {
            if strips[start].is_none() {
                start += 1;
                continue;
            }
            let end = (start..Self::BANDS)
                .find(|&index| strips[index].is_none())
                .unwrap_or(Self::BANDS);
            if end - start < Self::MIN_RUN {
                strips[start..end].fill(None);
            }
            start = end;
        }
        // Fill valleys among the bands that show a strip; a band without
        // one has no evidence of foreground, so no cut.
        let margin = height * Self::MARGIN;
        let known = strips
            .iter()
            .enumerate()
            .filter_map(|(band, strip)| strip.map(|y| (band, (y + margin).min(height))))
            .collect::<Vec<_>>();
        let mut filled = known.iter().map(|&(_, floor)| floor).collect::<Vec<_>>();
        fill_valleys(&mut filled);
        let mut floors = vec![height; Self::BANDS];
        for (&(band, _), floor) in known.iter().zip(filled) {
            floors[band] = floor;
        }
        if floors.iter().all(|&floor| floor >= height) {
            return None;
        }
        Some(Self { width, floors })
    }

    /// Floor y at image column `x`, interpolated between band centres.
    pub(crate) fn y_at(&self, x: f64) -> f64 {
        let band_width = self.width / self.floors.len() as f64;
        let position = (x / band_width - 0.5).clamp(0.0, (self.floors.len() - 1) as f64);
        let index = position.floor() as usize;
        let next = (index + 1).min(self.floors.len() - 1);
        let t = position - index as f64;
        self.floors[index] * (1.0 - t) + self.floors[next] * t
    }

    pub(crate) fn is_sky(&self, x: f64, y: f64) -> bool {
        y <= self.y_at(x)
    }
}

/// Few stars show low in a hazy sky, so a band's lowest star can sit far
/// above the horizon. A horizon rarely rises between two lower stretches,
/// so such a floor drops to the higher of the deepest floors on its two
/// sides. The ends are mirrored, so an end band is judged against its
/// neighbour like any other.
fn fill_valleys(floors: &mut [f64]) {
    let count = floors.len();
    if count < 2 {
        return;
    }
    let padded = std::iter::once(floors[1])
        .chain(floors.iter().copied())
        .chain(std::iter::once(floors[count - 2]))
        .collect::<Vec<_>>();
    let running_max = |values: &mut dyn Iterator<Item = f64>| {
        values
            .scan(f64::NEG_INFINITY, |high, floor| {
                *high = high.max(floor);
                Some(*high)
            })
            .collect::<Vec<_>>()
    };
    let left = running_max(&mut padded.iter().copied());
    let mut right = running_max(&mut padded.iter().rev().copied());
    right.reverse();
    for (index, floor) in floors.iter_mut().enumerate() {
        *floor = left[index + 1].min(right[index + 1]);
    }
}

/// P(X <= k) for X ~ Binomial(n, p): how likely at most `k` of `n` evenly
/// spread stars fall in a strip covering fraction `p` of the height.
fn binomial_cdf(k: usize, n: usize, p: f64) -> f64 {
    if p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return if k >= n { 1.0 } else { 0.0 };
    }
    let q = 1.0 - p;
    let mut term = q.powi(n as i32);
    let mut sum = term;
    for i in 0..k.min(n) {
        term *= (n - i) as f64 / (i + 1) as f64 * p / q;
        sum += term;
    }
    sum.min(1.0)
}

fn inside(dims: (u32, u32), x: f64, y: f64) -> bool {
    x >= 0.0 && y >= 0.0 && x < dims.0 as f64 && y < dims.1 as f64
}

/// The first `limit` marks `shown` accepts, and how many of the first
/// `limit` it turned away: the marks a map without the floor would show
/// but this one does not.
fn first_shown<T>(marks: Vec<T>, limit: usize, shown: impl Fn(&T) -> bool) -> (Vec<T>, usize) {
    let hidden = marks.iter().take(limit).filter(|mark| !shown(mark)).count();
    let kept = marks
        .into_iter()
        .filter(|mark| shown(mark))
        .take(limit)
        .collect();
    (kept, hidden)
}

/// Project the constellation figures and look up the named stars and
/// deep-sky objects in the field. With `foreground`, marks below the sky
/// it outlines are left out; see [`SkyFloor`].
pub(crate) fn collect(
    wcs: &Wcs,
    dims: (u32, u32),
    catalogs: &SkyMapCatalogs,
    foreground: Option<&Foreground>,
) -> Annotations {
    let sky_floor =
        foreground.and_then(|foreground| SkyFloor::from_detections(&foreground.detections, dims));
    let in_sky = |x: f64, y: f64| sky_floor.as_ref().is_none_or(|floor| floor.is_sky(x, y));
    let figures = constellations::project_figures(wcs, dims);
    let stars = catalogs
        .star_ids
        .as_ref()
        .map(|catalog| {
            named_stars(catalog, wcs, dims).unwrap_or_else(|error| {
                eprintln!("sky map: star names unavailable: {error}");
                Vec::new()
            })
        })
        .unwrap_or_default();
    let objects = catalogs
        .objects
        .as_ref()
        .map(|catalog| {
            deep_sky_objects(catalog, wcs, dims).unwrap_or_else(|error| {
                eprintln!("sky map: deep-sky objects unavailable: {error}");
                Vec::new()
            })
        })
        .unwrap_or_default();
    let (stars, hidden_stars) = first_shown(stars, MAX_STARS, |star| {
        in_sky(star.x, star.y) || foreground.is_some_and(|f| f.seen(star.x, star.y))
    });
    // An object centred outside the frame is judged where its extent
    // enters it.
    let (width, height) = (dims.0 as f64, dims.1 as f64);
    let (objects, hidden_objects) = first_shown(objects, MAX_OBJECTS, |object| {
        in_sky(
            object.x.clamp(0.0, width - 1.0),
            object.y.clamp(0.0, height - 1.0),
        )
    });
    Annotations {
        figures,
        stars,
        objects,
        sky_floor,
        hidden: hidden_stars + hidden_objects,
    }
}

/// Labelled stars in the image, brightest first: IAU proper names, then
/// bright figure stars without one by Bayer designation.
fn named_stars(
    catalog: &StarIdentifierCatalog,
    wcs: &Wcs,
    dims: (u32, u32),
) -> std::io::Result<Vec<StarMark>> {
    let center = wcs.pixel_to_world(dims.0 as f64 / 2.0, dims.1 as f64 / 2.0);
    let radius = crate::field_diagonal_deg(wcs, dims) / 2.0 + 0.5;
    let mut named = HashSet::new();
    let mut marks = Vec::new();
    let mut place = |name: String, stable_id: &str, ra: f64, dec: f64, mag: Option<f32>| {
        if named.contains(stable_id) {
            return;
        }
        named.insert(stable_id.to_string());
        if let Some((x, y)) = wcs.world_to_pixel(ra, dec)
            && inside(dims, x, y)
        {
            marks.push(StarMark { name, x, y, mag });
        }
    };
    for star in catalog.names_in_cone(
        center,
        radius,
        Some(StarNameCatalog::IauCatalogOfStarNames),
        Some(StarNameKind::ProperName),
    )? {
        if star.mag.is_some_and(|mag| mag <= STAR_MAG_LIMIT) {
            place(
                star.designation.to_string(),
                star.stable_id,
                star.ra,
                star.dec,
                star.mag,
            );
        }
    }
    let figure_stars = constellations::line_stars()
        .map(|star| format!("hr:{}", star.hr))
        .collect::<HashSet<_>>();
    for star in catalog.names_in_cone(
        center,
        radius,
        Some(StarNameCatalog::BrightStarCatalog),
        Some(StarNameKind::BayerFlamsteed),
    )? {
        if star.mag.is_some_and(|mag| mag <= BAYER_MAG_LIMIT)
            && figure_stars.contains(star.stable_id)
            && let Some(name) = bayer_name(star.designation)
        {
            place(name, star.stable_id, star.ra, star.dec, star.mag);
        }
    }
    marks.sort_by(|a, b| {
        a.mag
            .unwrap_or(f32::INFINITY)
            .total_cmp(&b.mag.unwrap_or(f32::INFINITY))
    });
    Ok(marks)
}

/// A Bayer designation with its Greek letter spelled out: "Gam Cas" and
/// "Gamma Cas" give "Gamma Cas", "Alp1 Cen" gives "Alpha1 Cen". `None` for
/// a Flamsteed number or anything else.
fn bayer_name(designation: &str) -> Option<String> {
    let (letter, constellation) = designation.trim().split_once(' ')?;
    let constellation = constellation.trim();
    constellations::constellation_name(constellation)?;
    let split = letter
        .find(|c: char| c.is_ascii_digit())
        .unwrap_or(letter.len());
    let (greek, component) = letter.split_at(split);
    if !component.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let (_, full) = GREEK_LETTERS.iter().find(|(short, full)| {
        greek.eq_ignore_ascii_case(short) || greek.eq_ignore_ascii_case(full)
    })?;
    Some(format!("{full}{component} {constellation}"))
}

/// Deep-sky objects whose extent reaches the image, most prominent first.
fn deep_sky_objects(
    catalog: &ObjectCatalog,
    wcs: &Wcs,
    dims: (u32, u32),
) -> Result<Vec<ObjectMark>, seiza::objects::ObjectQueryError> {
    let query = ObjectQuery {
        kinds: DSO_KINDS.to_vec(),
        sort: ObjectSort::Prominence,
        ..ObjectQuery::default()
    };
    Ok(catalog
        .query_footprint(wcs, dims, &query)?
        .into_iter()
        .map(|placed| {
            let object = &placed.object;
            let mut label = if object.common_name.is_empty() || object.common_name == object.name {
                object.name.clone()
            } else {
                format!("{} / {}", object.name, object.common_name)
            };
            if !inside(dims, placed.x, placed.y) {
                label.push_str(" (edge)");
            }
            let (semi_major_px, semi_minor_px, angle_deg) = crate::object_outline(&placed);
            ObjectMark {
                label,
                x: placed.x,
                y: placed.y,
                semi_major_px,
                semi_minor_px,
                angle_deg,
            }
        })
        .collect())
}

/// Label, draw and write a sky map.
pub(crate) fn write(
    path: &Path,
    photo: &RgbImage,
    wcs: &Wcs,
    catalogs: &SkyMapCatalogs,
    foreground: Option<&Foreground>,
    summary: &SolveSummary,
    width: u32,
) -> Result<()> {
    let annotations = collect(wcs, photo.dimensions(), catalogs, foreground);
    let map = render(photo, wcs, &annotations, summary, width)?;
    map.save(path)
        .with_context(|| format!("failed to write {}", path.display()))?;
    println!(
        "sky map written to {} ({} constellations, {} named stars, {} deep-sky objects)",
        path.display(),
        annotations.figures.len(),
        annotations.stars.len(),
        annotations.objects.len()
    );
    Ok(())
}

/// An axis-aligned box in output pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Rect {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

impl Rect {
    fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self {
            x0: x,
            y0: y,
            x1: x + width,
            y1: y + height,
        }
    }

    fn around(x: f64, y: f64, radius: f64) -> Self {
        Self::new(x - radius, y - radius, 2.0 * radius, 2.0 * radius)
    }

    fn overlaps(&self, other: &Rect) -> bool {
        self.x0 < other.x1 && other.x0 < self.x1 && self.y0 < other.y1 && other.y0 < self.y1
    }

    fn contains(&self, other: &Rect) -> bool {
        other.x0 >= self.x0 && other.y0 >= self.y0 && other.x1 <= self.x1 && other.y1 <= self.y1
    }

    fn nearest_point(&self, x: f64, y: f64) -> (f64, f64) {
        (x.clamp(self.x0, self.x1), y.clamp(self.y0, self.y1))
    }

    /// Shrunk by `by` on every side; a box too small for that collapses
    /// to its centre.
    fn inset(&self, by: f64) -> Rect {
        let (cx, cy) = ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0);
        Rect {
            x0: (self.x0 + by).min(cx),
            y0: (self.y0 + by).min(cy),
            x1: (self.x1 - by).max(cx),
            y1: (self.y1 - by).max(cy),
        }
    }
}

/// Greedy label placement: each label takes the first candidate spot
/// around its anchor that stays in bounds and clears everything placed
/// before it.
pub(crate) struct LabelPlacer {
    bounds: Rect,
    occupied: Vec<Rect>,
}

impl LabelPlacer {
    pub(crate) fn new(bounds: Rect) -> Self {
        Self {
            bounds,
            occupied: Vec::new(),
        }
    }

    /// Reserve an area (a marker) so later labels avoid it.
    pub(crate) fn reserve(&mut self, rect: Rect) {
        self.occupied.push(rect);
    }

    /// Place a `width` x `height` label near `(x, y)`, at least `gap`
    /// from it. Returns `None` when every candidate collides.
    pub(crate) fn place(
        &mut self,
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        gap: f64,
    ) -> Option<Rect> {
        let pad = height * 0.12;
        for distance in [gap, gap * 2.2, gap * 3.8] {
            let diagonal = distance * std::f64::consts::FRAC_1_SQRT_2;
            let candidates = [
                (x + diagonal, y - diagonal - height),
                (x + distance, y - height / 2.0),
                (x + diagonal, y + diagonal),
                (x - distance - width, y - height / 2.0),
                (x - diagonal - width, y - diagonal - height),
                (x - diagonal - width, y + diagonal),
                (x - width / 2.0, y - distance - height),
                (x - width / 2.0, y + distance),
            ];
            for (left, top) in candidates {
                let rect = Rect::new(left, top, width, height);
                let padded = Rect {
                    x0: rect.x0 - pad,
                    y0: rect.y0 - pad,
                    x1: rect.x1 + pad,
                    y1: rect.y1 + pad,
                };
                if self.bounds.contains(&rect)
                    && !self.occupied.iter().any(|other| other.overlaps(&padded))
                {
                    self.occupied.push(rect);
                    return Some(rect);
                }
            }
        }
        None
    }
}

/// Draw one line of text straight onto the canvas.
#[allow(clippy::too_many_arguments)]
fn draw_line(
    canvas: &mut RgbImage,
    font: &FontRef<'_>,
    size: f64,
    tracking: f64,
    (x, y): (f64, f64),
    color: Rgb<u8>,
    text: &str,
) {
    let (width, height) = measure(font, size, tracking, text);
    // Glyphs may reach a pixel or two past their advance.
    let slack = (size * 0.25).ceil() as i64;
    let origin = (x.floor() as i64 - slack, y.floor() as i64 - slack);
    let mut mask = Mask::at(
        origin,
        width.ceil() as usize + 2 * slack as usize,
        height.ceil() as usize + 2 * slack as usize,
    );
    draw_text(&mut mask, font, size, tracking, (x, y), text);
    mask.composite(canvas, color, 1.0);
}

/// Footer text: solve statistics, the caveat (with the foreground note
/// when the floor left something out), and the figure credit.
fn footer_text(
    wcs: &Wcs,
    summary: &SolveSummary,
    dims: (u32, u32),
    foreground_cut: bool,
) -> [String; 3] {
    let scale = wcs.scale_arcsec_per_px();
    let (ra, dec) = wcs.pixel_to_world(dims.0 as f64 / 2.0, dims.1 as f64 / 2.0);
    let mut stats = vec![format!("{} matched stars", summary.matched_stars)];
    if let Some(sip) = &wcs.sip {
        stats.push(format!("SIP order {}", sip.order));
    }
    stats.push(format!("{scale:.4} arcsec/px"));
    stats.push(format!("RMS {:.2} px", summary.rms_arcsec / scale));
    stats.push(format!("center RA {ra:.3}°  Dec {dec:+.3}°"));
    let caveat = if foreground_cut {
        "Catalog marks do not establish object detection. Marks below the lowest detected stars (foreground) are left out."
    } else {
        "Catalog marks do not establish object detection."
    };
    [
        stats.join("  |  "),
        caveat.to_string(),
        format!("Figures: {}", constellations::ATTRIBUTION),
    ]
}

/// Lay the image out under a title bar and over a footer, then draw the
/// figures, markers, and labels.
pub(crate) fn render(
    photo: &RgbImage,
    wcs: &Wcs,
    annotations: &Annotations,
    summary: &SolveSummary,
    width: u32,
) -> Result<RgbImage> {
    if !(MIN_WIDTH..=MAX_WIDTH).contains(&width) {
        anyhow::bail!("--sky-map-width must be between {MIN_WIDTH} and {MAX_WIDTH} pixels");
    }
    let fonts = Fonts::load()?;
    let u = width as f64 / 1400.0;
    let pad = (21.0 * u).round();
    let header = (100.0 * u).round();
    let (source_width, source_height) = photo.dimensions();
    let scale = (width as f64 - 2.0 * pad) / source_width as f64;
    let view_width = ((source_width as f64 * scale).round() as u32).max(1);
    let view_height = ((source_height as f64 * scale).round() as u32).max(1);
    let mut view = resize(photo, view_width, view_height);
    let foreground_cut =
        draw_annotations(&mut view, &fonts, annotations, u, scale) || annotations.hidden > 0;

    // The footer wraps any line wider than the image.
    let text_width = width as f64 - 2.0 * pad;
    let [stats, caveat, credit] = footer_text(wcs, summary, photo.dimensions(), foreground_cut);
    let footer_lines = [
        (&fonts.regular, 15.0 * u, TITLE, stats, 26.0 * u),
        (&fonts.regular, 13.5 * u, MUTED, caveat, 22.0 * u),
        (&fonts.regular, 11.5 * u, DIM, credit, 16.0 * u),
    ]
    .into_iter()
    .flat_map(|(font, size, color, text, advance)| {
        wrap(font, size, &text, text_width)
            .into_iter()
            .map(move |line| (font, size, color, line, advance))
    })
    .collect::<Vec<_>>();
    let footer_top = 14.0 * u;
    let footer =
        (footer_top + footer_lines.iter().map(|line| line.4).sum::<f64>() + 14.0 * u).round();
    let height = header as u32 + view_height + footer as u32;

    let mut canvas = RgbImage::from_pixel(width, height, BACKGROUND);
    image::imageops::replace(&mut canvas, &view, pad as i64, header as i64);
    let brand = format!("SEIZA  /  {}", summary.name);
    let projection = if wcs.sip.is_some() { "TAN/SIP" } else { "TAN" };
    let subtitle = format!(
        "Constellation figures, named stars and deep-sky positions from the solved {projection} WCS"
    );
    for (font, size, tracking, y, color, text) in [
        (
            &fonts.semibold,
            16.0 * u,
            0.6 * u,
            0.12,
            BRAND,
            brand.as_str(),
        ),
        (
            &fonts.semibold,
            26.0 * u,
            0.0,
            0.33,
            TITLE,
            "Catalog sky map",
        ),
        (
            &fonts.regular,
            14.0 * u,
            0.0,
            0.70,
            MUTED,
            subtitle.as_str(),
        ),
    ] {
        draw_line(
            &mut canvas,
            font,
            size,
            tracking,
            (pad, header * y),
            color,
            text,
        );
    }
    let mut y = header + view_height as f64 + footer_top;
    for (font, size, color, line, advance) in &footer_lines {
        draw_line(&mut canvas, font, *size, 0.0, (pad, y), *color, line);
        y += advance;
    }
    Ok(canvas)
}

/// Catmull-Rom resampling with rows in parallel: the filter
/// `image::imageops::resize` applies with `CatmullRom`, which runs on one
/// thread and was most of the map's cost for a phone frame.
fn resize(source: &RgbImage, width: u32, height: u32) -> RgbImage {
    let (source_width, source_height) = source.dimensions();
    let columns = resample_weights(source_width, width);
    let rows = resample_weights(source_height, height);
    let (in_stride, out_stride) = (source_width as usize * 3, width as usize * 3);
    // Every source row to the new width, then every output row from those.
    let mut wide = vec![0f32; source_height as usize * out_stride];
    wide.par_chunks_mut(out_stride)
        .zip(source.as_raw().par_chunks(in_stride))
        .for_each(|(out, row)| {
            for ((start, taps), pixel) in columns.iter().zip(out.chunks_exact_mut(3)) {
                for (offset, &weight) in taps.iter().enumerate() {
                    let input = &row[(start + offset) * 3..][..3];
                    for channel in 0..3 {
                        pixel[channel] += weight * input[channel] as f32;
                    }
                }
            }
        });
    let mut pixels = vec![0u8; height as usize * out_stride];
    pixels
        .par_chunks_mut(out_stride)
        .zip(rows.par_iter())
        .for_each(|(out, (start, taps))| {
            for (index, value) in out.iter_mut().enumerate() {
                let sum = taps
                    .iter()
                    .enumerate()
                    .map(|(offset, &weight)| weight * wide[(start + offset) * out_stride + index])
                    .sum::<f32>();
                *value = sum.round().clamp(0.0, 255.0) as u8;
            }
        });
    RgbImage::from_raw(width, height, pixels).expect("resampled buffer matches its size")
}

/// For each of `target` output samples, the first of `source` input samples
/// it reads and the normalized Catmull-Rom weights, widened when shrinking.
fn resample_weights(source: u32, target: u32) -> Vec<(usize, Vec<f32>)> {
    let ratio = source as f64 / target as f64;
    let spread = ratio.max(1.0);
    let support = 2.0 * spread;
    (0..target)
        .map(|index| {
            let center = (index as f64 + 0.5) * ratio;
            let left = ((center - support).floor().max(0.0) as usize).min(source as usize - 1);
            let right = ((center + support).ceil() as usize).clamp(left + 1, source as usize);
            let taps = (left..right)
                .map(|input| catmull_rom((input as f64 + 0.5 - center) / spread))
                .collect::<Vec<_>>();
            let total = taps.iter().sum::<f64>();
            (
                left,
                taps.iter().map(|weight| (weight / total) as f32).collect(),
            )
        })
        .collect()
}

fn catmull_rom(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        (1.5 * x - 2.5) * x * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}

/// Draw the figures, markers and labels onto the resized image. Returns
/// whether the sky floor dimmed a line or dropped a name.
fn draw_annotations(
    view: &mut RgbImage,
    fonts: &Fonts<'_>,
    annotations: &Annotations,
    u: f64,
    scale: f64,
) -> bool {
    let (width, height) = view.dimensions();
    let frame = Rect::new(0.0, 0.0, width as f64, height as f64);
    // Source pixel centres sit at integers; the view's pixel i spans
    // [i, i + 1].
    let to_view = |(x, y): (f64, f64)| ((x + 0.5) * scale, (y + 0.5) * scale);
    let mut cut = false;

    let mut lines = Mask::new(width, height);
    let mut gold = Mask::new(width, height);
    let mut cyan = Mask::new(width, height);
    let mut names = Mask::new(width, height);
    let mut placer = LabelPlacer::new(frame.inset(4.0 * u));

    // Constellation figures.
    for figure in &annotations.figures {
        for polyline in &figure.polylines {
            let points = polyline.iter().copied().map(to_view).collect::<Vec<_>>();
            lines.polyline(&points, 1.6 * u);
        }
    }
    if let Some(floor) = &annotations.sky_floor {
        cut |= lines.fade_below(30.0 * u, |x| (floor.y_at(x / scale - 0.5) + 0.5) * scale);
    }

    // Markers first, reserved so no label covers one.
    let star_radius = 5.5 * u;
    let stars = annotations
        .stars
        .iter()
        .map(|star| (star, to_view((star.x, star.y))))
        .collect::<Vec<_>>();
    for &(_, (x, y)) in &stars {
        gold.ring((x, y), star_radius, 1.5 * u);
        placer.reserve(Rect::around(x, y, star_radius + 2.0 * u));
    }
    let objects = annotations
        .objects
        .iter()
        .map(|object| {
            let (x, y) = to_view((object.x, object.y));
            (object, (x, y), object.semi_major_px * scale)
        })
        .collect::<Vec<_>>();
    for &(object, (x, y), major) in &objects {
        if major >= 7.0 * u {
            cyan.ellipse(
                (x, y),
                major,
                (object.semi_minor_px * scale).max(3.0 * u),
                object.angle_deg,
                1.5 * u,
            );
        } else {
            cyan.ring((x, y), 5.0 * u, 1.4 * u);
        }
        cyan.disc((x, y), 2.4 * u);
        if frame.contains(&Rect::around(x, y, 0.0)) {
            placer.reserve(Rect::around(x, y, (6.0 * u).max(major.min(9.0 * u))));
        }
    }

    // Labels, brightest stars first, then objects by prominence, then
    // constellation names.
    let label_size = 15.0 * u;
    let leader = |mask: &mut Mask, (x, y): (f64, f64), radius: f64, rect: &Rect| {
        let (tx, ty) = rect.nearest_point(x, y);
        let distance = (tx - x).hypot(ty - y);
        if distance > radius + 4.0 * u {
            let (ux, uy) = ((tx - x) / distance, (ty - y) / distance);
            mask.stroke(
                (x + ux * (radius + 1.5 * u), y + uy * (radius + 1.5 * u)),
                (tx - ux * 2.0 * u, ty - uy * 2.0 * u),
                1.1 * u,
            );
        }
    };
    for &(star, (x, y)) in &stars {
        let (w, h) = measure(&fonts.semibold, label_size, 0.0, &star.name);
        if let Some(rect) = placer.place(x, y, w, h, star_radius + 9.0 * u) {
            leader(&mut gold, (x, y), star_radius, &rect);
            draw_text(
                &mut gold,
                &fonts.semibold,
                label_size,
                0.0,
                (rect.x0, rect.y0),
                &star.name,
            );
        }
    }
    // Object labels anchor inside the frame, a little in from its edge.
    let anchors = frame.inset(8.0 * u);
    for &(object, (x, y), major) in &objects {
        let (w, h) = measure(&fonts.semibold, label_size, 0.0, &object.label);
        let (ax, ay) = anchors.nearest_point(x, y);
        let radius = if major >= 7.0 * u {
            major.min(18.0 * u)
        } else {
            5.0 * u
        };
        let gap = if (ax, ay) == (x, y) {
            radius + 9.0 * u
        } else {
            6.0 * u
        };
        if let Some(rect) = placer.place(ax, ay, w, h, gap) {
            if (ax, ay) == (x, y) {
                leader(&mut cyan, (x, y), 5.0 * u, &rect);
            }
            draw_text(
                &mut cyan,
                &fonts.semibold,
                label_size,
                0.0,
                (rect.x0, rect.y0),
                &object.label,
            );
        }
    }
    let name_size = 13.0 * u;
    let tracking = 1.2 * u;
    for figure in &annotations.figures {
        let Some(anchor) = figure.label else { continue };
        // A sliver of a figure at the frame edge does not earn a name.
        if figure.visible_length_px() * scale < 40.0 * u {
            continue;
        }
        if let Some(floor) = &annotations.sky_floor
            && !floor.is_sky(anchor.0, anchor.1)
        {
            cut = true;
            continue;
        }
        let text = figure.name.to_uppercase();
        let (w, h) = measure(&fonts.regular, name_size, tracking, &text);
        let (x, y) = to_view(anchor);
        if let Some(rect) = placer.place(x, y, w, h, 2.0 * u) {
            draw_text(
                &mut names,
                &fonts.regular,
                name_size,
                tracking,
                (rect.x0, rect.y0),
                &text,
            );
        }
    }

    // Dark halos under everything drawn on the photo, then the layers.
    let mut halo = Mask::new(width, height);
    for layer in [&gold, &cyan, &names] {
        halo.absorb(layer);
    }
    halo.dilated(1.6 * u).composite(view, SHADOW, 0.55);
    lines.composite(view, FIGURE_LINE, 0.72);
    names.composite(view, FIGURE_NAME, 0.92);
    cyan.composite(view, OBJECT_CYAN, 1.0);
    gold.composite(view, STAR_GOLD, 1.0);
    cut
}

#[cfg(test)]
mod tests {
    use super::*;
    use seiza::objects::SkyObject;
    use seiza::star_ids::StarIdentifierCatalogBuilder;

    fn cassiopeia() -> (Wcs, (u32, u32)) {
        let dims = (1600, 1200);
        let wcs = Wcs::from_center_scale_rotation((14.0, 60.0), (800.0, 600.0), 72.0, 0.0, false);
        (wcs, dims)
    }

    fn object(name: &str, common: &str, ra: f64, dec: f64, major: f32) -> SkyObject {
        SkyObject {
            kind: ObjectKind::OpenCluster,
            ra,
            dec,
            mag: Some(6.0),
            major_arcmin: Some(major),
            minor_arcmin: None,
            position_angle_deg: None,
            name: name.into(),
            common_name: common.into(),
            metadata: Default::default(),
        }
    }

    fn add_name(
        builder: &mut StarIdentifierCatalogBuilder,
        catalog: StarNameCatalog,
        kind: StarNameKind,
        designation: &str,
        hr: u32,
        mag: f32,
    ) {
        let star = constellations::line_star(hr).unwrap();
        builder
            .add_name(
                catalog,
                kind,
                designation,
                &format!("hr:{hr}"),
                "",
                star.ra,
                star.dec,
                Some(mag),
            )
            .unwrap();
    }

    fn catalogs(dir: &Path) -> SkyMapCatalogs {
        let path = dir.join("stars.ids.bin");
        let mut builder = StarIdentifierCatalogBuilder::new(2025.5, "test");
        add_name(
            &mut builder,
            StarNameCatalog::IauCatalogOfStarNames,
            StarNameKind::ProperName,
            "Schedar",
            168,
            2.24,
        );
        // The sidecar spells most Bayer letters two ways and adds the
        // Flamsteed number.
        for designation in ["Gam Cas", "Gamma Cas", "27 Cas"] {
            add_name(
                &mut builder,
                StarNameCatalog::BrightStarCatalog,
                StarNameKind::BayerFlamsteed,
                designation,
                264,
                2.47,
            );
        }
        // Too faint for a label.
        builder
            .add_name(
                StarNameCatalog::IauCatalogOfStarNames,
                StarNameKind::ProperName,
                "Faint",
                "hr:1",
                "",
                15.0,
                61.0,
                Some(5.5),
            )
            .unwrap();
        builder.write_to(&path).unwrap();
        SkyMapCatalogs {
            objects: Some(ObjectCatalog::new(vec![
                object("NGC 457", "Owl Cluster", 19.886, 58.291, 7.8),
                object("NGC 869", "", 34.744, 57.117, 14.4),
                object("NGC 9999", "", 200.0, -40.0, 5.0),
            ])),
            star_ids: Some(StarIdentifierCatalog::open(&path).unwrap()),
        }
    }

    #[test]
    fn collect_finds_figures_names_and_objects() {
        let dir = tempfile::tempdir().unwrap();
        let (wcs, dims) = cassiopeia();
        let annotations = collect(&wcs, dims, &catalogs(dir.path()), None);
        assert!(annotations.figures.iter().any(|f| f.abbr == "Cas"));
        let names = annotations
            .stars
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["Schedar", "Gamma Cas"]);
        let labels = annotations
            .objects
            .iter()
            .map(|o| o.label.as_str())
            .collect::<Vec<_>>();
        assert!(labels.contains(&"NGC 457 / Owl Cluster"));
        assert!(labels.contains(&"NGC 869"));
        assert!(!labels.iter().any(|label| label.starts_with("NGC 9999")));
        assert!(annotations.sky_floor.is_none());
        assert_eq!(annotations.hidden, 0);
    }

    #[test]
    fn collect_without_catalogs_still_draws_figures() {
        let (wcs, dims) = cassiopeia();
        let annotations = collect(&wcs, dims, &SkyMapCatalogs::default(), None);
        assert!(!annotations.figures.is_empty());
        assert!(annotations.stars.is_empty() && annotations.objects.is_empty());
    }

    #[test]
    fn bayer_names_spell_out_every_greek_letter() {
        assert_eq!(bayer_name("Gam Cas").as_deref(), Some("Gamma Cas"));
        assert_eq!(bayer_name("Gamma Cas").as_deref(), Some("Gamma Cas"));
        assert_eq!(bayer_name("Alp1 Cen").as_deref(), Some("Alpha1 Cen"));
        assert_eq!(bayer_name("Omi2 CMa").as_deref(), Some("Omicron2 CMa"));
        // The short letters are names in their own right.
        for (designation, name) in [
            ("Eta Cen", "Eta Cen"),
            ("Mu Vel", "Mu Vel"),
            ("Pi Pup", "Pi Pup"),
            ("Tau Pup", "Tau Pup"),
            ("Rho Pup", "Rho Pup"),
            ("Chi Car", "Chi Car"),
            ("Psi UMa", "Psi UMa"),
            ("Phi Sgr", "Phi Sgr"),
            ("Nu Pup", "Nu Pup"),
            ("Xi Pup", "Xi Pup"),
        ] {
            assert_eq!(
                bayer_name(designation).as_deref(),
                Some(name),
                "{designation}"
            );
        }
        assert_eq!(bayer_name("27 Cas"), None);
        assert_eq!(bayer_name("Gam Xyz"), None);
        assert_eq!(bayer_name("Gamma"), None);
        assert_eq!(bayer_name("Gam1a Cas"), None);
    }

    /// A Scorpius-Centaurus frame: bright Bayer figure stars rank with the
    /// proper names by magnitude, and short Greek names are labelled.
    #[test]
    fn bright_bayer_stars_outrank_faint_proper_names() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stars.ids.bin");
        let mut builder = StarIdentifierCatalogBuilder::new(2025.5, "test");
        // Eta Cen (HR 5440), Alpha Lup (HR 5469), Kappa Sco (HR 6580).
        for (designation, hr, mag) in [
            ("Eta Cen", 5440, 2.31),
            ("Alp Lup", 5469, 2.30),
            ("Alpha Lup", 5469, 2.30),
            ("Kap Sco", 6580, 2.41),
        ] {
            add_name(
                &mut builder,
                StarNameCatalog::BrightStarCatalog,
                StarNameKind::BayerFlamsteed,
                designation,
                hr,
                mag,
            );
        }
        // More faint proper names than the label budget, all near the
        // field centre.
        let center = constellations::line_star(5469).unwrap();
        for index in 0..40 {
            builder
                .add_name(
                    StarNameCatalog::IauCatalogOfStarNames,
                    StarNameKind::ProperName,
                    &format!("Faint {index}"),
                    &format!("hip:{}", 900_000 + index),
                    "",
                    center.ra + (index % 8) as f64 * 0.7 - 2.5,
                    center.dec + (index / 8) as f64 * 0.7 - 1.5,
                    Some(4.0 + index as f32 * 0.01),
                )
                .unwrap();
        }
        builder.write_to(&path).unwrap();
        let catalogs = SkyMapCatalogs {
            objects: None,
            star_ids: Some(StarIdentifierCatalog::open(&path).unwrap()),
        };
        let wcs = Wcs::from_center_scale_rotation(
            (center.ra, center.dec),
            (2000.0, 1500.0),
            90.0,
            0.0,
            false,
        );
        let annotations = collect(&wcs, (4000, 3000), &catalogs, None);
        let names = annotations
            .stars
            .iter()
            .map(|star| star.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names.len(), MAX_STARS);
        assert_eq!(&names[..3], ["Alpha Lup", "Eta Cen", "Kappa Sco"]);
    }

    #[test]
    fn render_lays_out_header_photo_and_footer() {
        let dir = tempfile::tempdir().unwrap();
        let (wcs, dims) = cassiopeia();
        let annotations = collect(&wcs, dims, &catalogs(dir.path()), None);
        let photo = RgbImage::from_pixel(dims.0, dims.1, Rgb([30, 30, 30]));
        let summary = SolveSummary {
            name: "synthetic".into(),
            matched_stars: 42,
            rms_arcsec: 90.0,
        };
        let map = render(&photo, &wcs, &annotations, &summary, 1400).unwrap();
        assert_eq!(map.width(), 1400);
        // 1358 px of photo at 1600:1200, plus header and footer.
        assert_eq!(map.height(), 100 + 1019 + 92);
        // The title bar and footer are background with text on them.
        assert_eq!(*map.get_pixel(1390, 5), BACKGROUND);
        let title_pixels = (0..100)
            .flat_map(|y| (0..700).map(move |x| (x, y)))
            .filter(|&(x, y)| *map.get_pixel(x, y) != BACKGROUND)
            .count();
        assert!(title_pixels > 500, "{title_pixels}");
        // The credit line is drawn at the bottom of the footer.
        let credit_pixels = (1119 + 60..1211)
            .flat_map(|y| (0..1400).map(move |x| (x, y)))
            .filter(|&(x, y)| *map.get_pixel(x, y) != BACKGROUND)
            .count();
        assert!(credit_pixels > 500, "{credit_pixels}");
        // Cassiopeia's Schedar-Gamma segment midpoint carries line colour.
        let schedar = constellations::line_star(168).unwrap();
        let gamma = constellations::line_star(264).unwrap();
        let a = wcs.world_to_pixel(schedar.ra, schedar.dec).unwrap();
        let b = wcs.world_to_pixel(gamma.ra, gamma.dec).unwrap();
        let scale = 1358.0 / 1600.0;
        let (mx, my) = (
            21.0 + ((a.0 + b.0) / 2.0 + 0.5) * scale,
            100.0 + ((a.1 + b.1) / 2.0 + 0.5) * scale,
        );
        // The brightest blue in a 3x3 neighbourhood (the line is anti-aliased).
        let pixel = (-1..=1)
            .flat_map(|dy| (-1..=1).map(move |dx| (dx, dy)))
            .map(|(dx, dy)| *map.get_pixel((mx as i64 + dx) as u32, (my as i64 + dy) as u32))
            .max_by_key(|pixel| pixel[2])
            .unwrap();
        assert!(pixel[2] > 150 && pixel[2] > pixel[0] + 40, "{pixel:?}");
        // The width is bounded.
        assert!(render(&photo, &wcs, &annotations, &summary, 100).is_err());
        // A frame far wider than tall still renders, labels and all.
        let strip = RgbImage::from_pixel(9000, 60, Rgb([30, 30, 30]));
        let annotations = Annotations {
            objects: vec![ObjectMark {
                label: "NGC 1".into(),
                x: 4500.0,
                y: 30.0,
                semi_major_px: 10.0,
                semi_minor_px: 10.0,
                angle_deg: 0.0,
            }],
            ..Annotations::default()
        };
        let map = render(&strip, &wcs, &annotations, &summary, 700).unwrap();
        assert_eq!(map.width(), 700);
    }

    #[test]
    fn footer_wraps_and_notes_the_foreground_only_when_cut() {
        let (wcs, dims) = cassiopeia();
        let summary = SolveSummary {
            name: "synthetic".into(),
            matched_stars: 42,
            rms_arcsec: 90.0,
        };
        let [stats, caveat, credit] = footer_text(&wcs, &summary, dims, false);
        assert!(stats.starts_with("42 matched stars"));
        assert!(!caveat.contains("foreground"));
        assert!(credit.contains(constellations::ATTRIBUTION));
        let [_, caveat, _] = footer_text(&wcs, &summary, dims, true);
        assert!(caveat.contains("(foreground) are left out"));
        let fonts = Fonts::load().unwrap();
        // The credit fits one line at the default size and wraps, losing
        // nothing, when narrower.
        assert_eq!(wrap(&fonts.regular, 11.5, &credit, 1358.0).len(), 1);
        let lines = wrap(&fonts.regular, 11.5, &credit, 300.0);
        assert!(lines.len() > 2, "{lines:?}");
        assert!(
            lines
                .iter()
                .all(|line| measure(&fonts.regular, 11.5, 0.0, line).0 <= 300.0)
        );
        assert_eq!(lines.join(" "), credit);
    }

    #[test]
    fn placer_avoids_overlaps_and_bounds() {
        let mut placer = LabelPlacer::new(Rect::new(0.0, 0.0, 200.0, 100.0));
        placer.reserve(Rect::around(100.0, 50.0, 4.0));
        let first = placer.place(100.0, 50.0, 60.0, 14.0, 10.0).unwrap();
        // First choice: up and to the right of the anchor.
        let d = 10.0 * std::f64::consts::FRAC_1_SQRT_2;
        assert_eq!(first, Rect::new(100.0 + d, 50.0 - d - 14.0, 60.0, 14.0));
        let second = placer.place(100.0, 50.0, 60.0, 14.0, 10.0).unwrap();
        assert!(!second.overlaps(&first));
        // Near the right edge the label flips left.
        let mut placer = LabelPlacer::new(Rect::new(0.0, 0.0, 200.0, 100.0));
        let edge = placer.place(190.0, 50.0, 60.0, 14.0, 10.0).unwrap();
        assert!(edge.x1 <= 180.0);
        // Nothing fits in a box smaller than the label.
        let mut placer = LabelPlacer::new(Rect::new(0.0, 0.0, 40.0, 10.0));
        assert!(placer.place(20.0, 5.0, 60.0, 14.0, 4.0).is_none());
        // Insetting a box too small for it leaves its centre.
        let inset = Rect::new(0.0, 0.0, 100.0, 10.0).inset(8.0);
        assert_eq!(inset, Rect::new(8.0, 5.0, 84.0, 0.0));
    }

    #[test]
    fn only_detections_near_catalog_stars_are_confirmed() {
        let detected = [(10.0, 10.0), (50.0, 50.0), (100.0, 100.0)];
        let catalog = [(11.0, 12.0), (100.0, 103.5), (300.0, 300.0)];
        assert_eq!(
            confirmed_detections(&detected, &catalog, 3.0),
            vec![(10.0, 10.0)]
        );
        assert_eq!(confirmed_detections(&detected, &catalog, 4.0).len(), 2);
    }

    /// A small deterministic generator, so the statistics tests are stable.
    struct Random(u64);

    impl Random {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn evenly_spread_stars_never_make_a_floor() {
        let dims = (4000, 3000);
        for count in [60, 100, 150, 300, 450, 600] {
            for seed in 1..=60u64 {
                let mut random = Random(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let stars = (0..count)
                    .map(|_| (random.next() * 4000.0, random.next() * 3000.0))
                    .collect::<Vec<_>>();
                assert!(
                    SkyFloor::from_detections(&stars, dims).is_none(),
                    "{count} stars, seed {seed}"
                );
            }
        }
    }

    /// A phone frame: stars down to a ragged horizon, lowest at the left
    /// edge, a rock in the middle and a tree on the right, plus a few
    /// chance matches on the ground.
    #[test]
    fn a_ragged_horizon_makes_a_floor_that_keeps_the_sky() {
        let dims = (4000, 3000);
        let horizon = |x: f64| {
            if x < 800.0 {
                2550.0 - x * 0.3
            } else if x < 2600.0 {
                2200.0 + 60.0 * (x / 150.0).sin()
            } else {
                2200.0 - (x - 2600.0) * 0.15
            }
        };
        for seed in 1..=20u64 {
            let mut random = Random(seed.wrapping_mul(0x2545_F491_4F6C_DD1D));
            let mut stars = Vec::new();
            while stars.len() < 450 {
                let (x, y) = (random.next() * 4000.0, random.next() * 3000.0);
                if y < horizon(x) {
                    stars.push((x, y));
                }
            }
            let sky = stars.clone();
            stars.extend([(900.0, 2900.0), (2100.0, 2700.0), (3300.0, 2950.0)]);
            let floor = SkyFloor::from_detections(&stars, dims).expect("a floor");
            // Every star of the real sky is kept, including those near the
            // horizon at the left edge, where it reaches lowest.
            for &(x, y) in &sky {
                assert!(
                    floor.is_sky(x, y),
                    "seed {seed}: ({x:.0}, {y:.0}) under the floor at {:.0}",
                    floor.y_at(x)
                );
            }
            // The ground is cut across the frame.
            for x in [100.0, 1500.0, 2000.0, 3900.0] {
                assert!(!floor.is_sky(x, 2950.0), "seed {seed}: x {x}");
            }
        }
    }

    #[test]
    fn sky_floor_follows_the_lowest_stars() {
        let dims = (2000, 1000);
        // Stars fill the top half on the left, the whole height on the right.
        let mut stars = Vec::new();
        for i in 0..400 {
            let x = (i * 37 % 2000) as f64;
            let y = (i * 53 % 1000) as f64;
            if x >= 1000.0 || y < 500.0 {
                stars.push((x, y));
            }
        }
        let floor = SkyFloor::from_detections(&stars, dims).unwrap();
        assert!(floor.y_at(200.0) < 600.0, "{}", floor.y_at(200.0));
        assert!(floor.y_at(1800.0) >= 950.0, "{}", floor.y_at(1800.0));
        assert!(floor.is_sky(200.0, 300.0) && !floor.is_sky(200.0, 800.0));
        // A sparse stretch between two low floors is filled, not cut.
        let mut valley = Vec::new();
        for i in 0..600 {
            let x = (i * 37 % 2000) as f64;
            let y = (i * 53 % 1000) as f64;
            let sparse = (800.0..1200.0).contains(&x);
            if (sparse && y < 200.0) || (!sparse && y < 700.0) {
                valley.push((x, y));
            }
        }
        let floor = SkyFloor::from_detections(&valley, dims).unwrap();
        assert!(floor.y_at(1000.0) > 650.0, "{}", floor.y_at(1000.0));
        assert!(!floor.is_sky(1000.0, 900.0));
        // A high edge band is lowered to its neighbour, like any other.
        let mut edge = Vec::new();
        for i in 0..600 {
            let x = (i * 37 % 2000) as f64;
            let y = (i * 53 % 1000) as f64;
            if (x < 100.0 && y < 300.0) || (x >= 100.0 && y < 700.0) {
                edge.push((x, y));
            }
        }
        let floor = SkyFloor::from_detections(&edge, dims).unwrap();
        assert!(floor.is_sky(20.0, 650.0), "{}", floor.y_at(20.0));
        // Stars everywhere: no floor. Too few stars: no floor.
        let everywhere = (0..2000)
            .map(|i| ((i * 37 % 2000) as f64, (i * 53 % 997) as f64))
            .collect::<Vec<_>>();
        assert!(SkyFloor::from_detections(&everywhere, dims).is_none());
        assert!(SkyFloor::from_detections(&stars[..20], dims).is_none());
        // A clear strip only a band or two wide is no floor.
        let notch = (0..800)
            .map(|i| ((i * 37 % 2000) as f64, (i * 53 % 1000) as f64))
            .filter(|&(x, y)| !(300.0..450.0).contains(&x) || y < 400.0)
            .collect::<Vec<_>>();
        assert!(SkyFloor::from_detections(&notch, dims).is_none());
        // Bands with no stars at all are no evidence either way.
        let left_only = (0..400)
            .map(|i| ((i * 37 % 800) as f64, (i * 53 % 1000) as f64))
            .collect::<Vec<_>>();
        assert!(SkyFloor::from_detections(&left_only, dims).is_none());
    }

    #[test]
    fn binomial_tail() {
        assert!((binomial_cdf(0, 10, 0.5) - 0.5f64.powi(10)).abs() < 1e-15);
        assert!((binomial_cdf(1, 3, 0.5) - 0.5).abs() < 1e-12);
        assert!((binomial_cdf(2, 2, 0.3) - 1.0).abs() < 1e-12);
        assert_eq!(binomial_cdf(0, 5, 0.0), 1.0);
        assert_eq!(binomial_cdf(0, 5, 1.0), 0.0);
    }

    #[test]
    fn foreground_marks_are_left_out_unless_the_image_shows_them() {
        let dir = tempfile::tempdir().unwrap();
        // Schedar near the bottom of the frame, in the foreground.
        let schedar = constellations::line_star(168).unwrap();
        let dims = (1600, 1200);
        let wcs = Wcs::from_center_scale_rotation(
            (schedar.ra, schedar.dec),
            (800.0, 1130.0),
            72.0,
            0.0,
            false,
        );
        let (sx, sy) = wcs.world_to_pixel(schedar.ra, schedar.dec).unwrap();
        // Detections fill the top 40%, plus a chance match on the ground.
        let mut detections = (0..300)
            .map(|i| ((i * 37 % 1600) as f64, (i * 53 % 480) as f64))
            .collect::<Vec<_>>();
        detections.push((sx + 30.0, 1190.0));
        let mut foreground = Foreground {
            detections,
            tolerance: 4.0,
        };
        let catalogs = catalogs(dir.path());
        let annotations = collect(&wcs, dims, &catalogs, Some(&foreground));
        let floor = annotations.sky_floor.as_ref().expect("a floor");
        assert!(!floor.is_sky(sx, sy));
        assert!(annotations.stars.iter().all(|star| star.name != "Schedar"));
        assert!(annotations.hidden > 0);
        assert!(!annotations.figures.is_empty(), "lines fade, not vanish");
        // Detected in haze just above the horizon: kept.
        foreground.detections.push((sx + 1.0, sy - 1.0));
        let annotations = collect(&wcs, dims, &catalogs, Some(&foreground));
        assert!(annotations.stars.iter().any(|star| star.name == "Schedar"));
    }

    #[test]
    fn objects_centred_below_the_frame_are_judged_at_its_edge() {
        let (wcs, dims) = cassiopeia();
        // Stars down to the bottom on the right half, a strip on the left.
        let detections = (0..800)
            .map(|i| ((i * 37 % 1600) as f64, (i * 53 % 1200) as f64))
            .filter(|&(x, y)| x >= 800.0 || y < 700.0)
            .collect::<Vec<_>>();
        let floor = SkyFloor::from_detections(&detections, dims).expect("a floor");
        assert!(floor.is_sky(1400.0, 1199.0) && !floor.is_sky(200.0, 1199.0));
        // An object centred below the right half, its extent in the frame.
        let below = wcs.pixel_to_world(1400.0, 1260.0);
        let catalogs = SkyMapCatalogs {
            objects: Some(ObjectCatalog::new(vec![object(
                "NGC 1", "", below.0, below.1, 300.0,
            )])),
            star_ids: None,
        };
        let foreground = Foreground {
            detections,
            tolerance: 4.0,
        };
        let annotations = collect(&wcs, dims, &catalogs, Some(&foreground));
        assert_eq!(annotations.objects.len(), 1, "{:?}", annotations.objects);
        assert_eq!(annotations.objects[0].label, "NGC 1 (edge)");
    }

    #[test]
    fn resize_matches_the_image_crate() {
        let mut random = Random(7);
        let source = RgbImage::from_fn(301, 157, |_, _| {
            Rgb([0, 1, 2].map(|_| (random.next() * 255.0) as u8))
        });
        for (width, height) in [(150, 78), (97, 51), (301, 157), (640, 330)] {
            let ours = resize(&source, width, height);
            let theirs = image::imageops::resize(
                &source,
                width,
                height,
                image::imageops::FilterType::CatmullRom,
            );
            let worst = ours
                .as_raw()
                .iter()
                .zip(theirs.as_raw())
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            assert!(worst <= 1, "{width}x{height}: off by {worst}");
        }
    }
}
