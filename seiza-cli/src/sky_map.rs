//! `--sky-map`: a labelled chart of a solved image.
//!
//! The figure shows the image under a dark title bar and footer, with
//! constellation stick figures and names, IAU-named stars, and deep-sky
//! objects drawn through the solved WCS. Every position comes from a catalog:
//! the marks say where things are, not that they were detected.
//!
//! Gathering ([`collect`]) works in source-image pixels and is separate
//! from drawing ([`render`]) so either can be tested alone.

use std::path::{Path, PathBuf};

use ab_glyph::{Font, FontRef, PxScale, ScaleFont, point};
use anyhow::{Context, Result};
use image::{Rgb, RgbImage};
use seiza::Wcs;
use seiza::constellations::{self, ProjectedFigure};
use seiza::data_paths;
use seiza::objects::{ObjectCatalog, ObjectKind, ObjectQuery, ObjectSort};
use seiza::star_ids::{StarIdentifierCatalog, StarNameCatalog, StarNameKind};

const REGULAR_TTF: &[u8] = include_bytes!("../fonts/Inter-Regular.ttf");
const SEMIBOLD_TTF: &[u8] = include_bytes!("../fonts/Inter-SemiBold.ttf");

/// Default output width, pixels.
pub(crate) const DEFAULT_WIDTH: u32 = 2100;
const MIN_WIDTH: u32 = 640;
const MAX_WIDTH: u32 = 12_000;

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
            Err(error) => {
                eprintln!("sky map: {error}; deep-sky objects are left out");
                None
            }
        };
        let star_ids = match data_paths::star_identifiers(dir.as_deref())
            .or_else(|_| data_paths::star_identifiers(None))
        {
            Ok(path) => StarIdentifierCatalog::open(&path)
                .map_err(|error| eprintln!("sky map: {}: {error}", path.display()))
                .ok(),
            Err(error) => {
                eprintln!("sky map: {error}; star names are left out");
                None
            }
        };
        Self { objects, star_ids }
    }
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
    pub semi_major_px: f64,
    pub semi_minor_px: f64,
    /// `None` draws a circle of the major axis.
    pub angle_deg: Option<f64>,
}

/// Everything the map labels, in source-image pixels.
#[derive(Debug, Default)]
pub(crate) struct Annotations {
    pub figures: Vec<ProjectedFigure>,
    pub stars: Vec<StarMark>,
    pub objects: Vec<ObjectMark>,
    /// Foreground limit, when the detections show one.
    pub sky_floor: Option<SkyFloor>,
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
/// detected star of each column band (plus a margin) there is ground, a
/// tree, or a building, and catalog marks there only add clutter.
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
    /// A band and its neighbours need this many stars to set a floor.
    const MIN_BAND_STARS: usize = 5;
    /// Lowest stars of a band neighbourhood ignored as possible chance
    /// matches on the foreground.
    const SKIPPED_LOWEST: usize = 2;

    /// `None` when stars reach the bottom of the frame everywhere, which is
    /// the normal case for a telescope image.
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
        let margin = height * 0.02;
        let floors = (0..Self::BANDS)
            .map(|band| {
                // A band and its neighbours, so one sparse band does not cut
                // into the sky. The third-lowest star, so a chance match or two
                // on the foreground do not lower the floor; too few stars
                // near means no evidence.
                let mut ys = (band.saturating_sub(1)..=(band + 1).min(Self::BANDS - 1))
                    .flat_map(|index| bands[index].iter().copied())
                    .collect::<Vec<_>>();
                if ys.len() < Self::MIN_BAND_STARS {
                    return height;
                }
                ys.sort_by(|a, b| b.total_cmp(a));
                (ys[Self::SKIPPED_LOWEST] + margin).min(height)
            })
            .collect::<Vec<_>>();
        // Few stars show low in a hazy sky, so a band's lowest star can sit
        // far above the horizon. A horizon rarely dips between two higher
        // stretches, so fill such valleys up to the lower of the highest
        // floors on either side.
        let mut floors = floors;
        let left = floors
            .iter()
            .scan(0.0f64, |high, &floor| {
                *high = high.max(floor);
                Some(*high)
            })
            .collect::<Vec<_>>();
        let mut right = floors
            .iter()
            .rev()
            .scan(0.0f64, |high, &floor| {
                *high = high.max(floor);
                Some(*high)
            })
            .collect::<Vec<_>>();
        right.reverse();
        for (index, floor) in floors.iter_mut().enumerate() {
            *floor = left[index].min(right[index]);
        }
        if floors.iter().all(|&floor| floor >= height * 0.95) {
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

fn inside(dims: (u32, u32), x: f64, y: f64) -> bool {
    x >= 0.0 && y >= 0.0 && x < dims.0 as f64 && y < dims.1 as f64
}

/// Project the constellation figures and look up the named stars and
/// deep-sky objects in the field. `detected` (image pixels) outlines the
/// sky in a photo with foreground; see [`SkyFloor`].
pub(crate) fn collect(
    wcs: &Wcs,
    dims: (u32, u32),
    catalogs: &SkyMapCatalogs,
    detected: &[(f64, f64)],
) -> Annotations {
    let sky_floor = SkyFloor::from_detections(detected, dims);
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
    Annotations {
        figures,
        stars: stars
            .into_iter()
            .filter(|star| in_sky(star.x, star.y))
            .collect(),
        objects: objects
            .into_iter()
            .filter(|object| in_sky(object.x.clamp(0.0, dims.0 as f64), object.y))
            .collect(),
        sky_floor,
    }
}

fn named_stars(
    catalog: &StarIdentifierCatalog,
    wcs: &Wcs,
    dims: (u32, u32),
) -> std::io::Result<Vec<StarMark>> {
    let center = wcs.pixel_to_world(dims.0 as f64 / 2.0, dims.1 as f64 / 2.0);
    let radius = wcs
        .footprint(dims.0, dims.1)
        .iter()
        .map(|&corner| constellations::separation_deg(center, corner))
        .fold(0.0, f64::max)
        + 0.5;
    let mut seen = std::collections::HashSet::new();
    let mut marks = Vec::new();
    let mut place = |name: &str, stable_id: &str, ra: f64, dec: f64, mag: Option<f32>| {
        if marks.len() >= MAX_STARS || seen.contains(stable_id) {
            return;
        }
        if let Some((x, y)) = wcs.world_to_pixel(ra, dec)
            && inside(dims, x, y)
        {
            seen.insert(stable_id.to_string());
            marks.push(StarMark {
                name: name.to_string(),
                x,
                y,
                mag,
            });
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
                star.designation,
                star.stable_id,
                star.ra,
                star.dec,
                star.mag,
            );
        }
    }
    // Bright figure stars without a proper name: label them "Gamma Cas".
    let figure_stars = constellations::line_stars()
        .map(|star| format!("hr:{}", star.hr))
        .collect::<std::collections::HashSet<_>>();
    let mut bayer = catalog
        .names_in_cone(
            center,
            radius,
            Some(StarNameCatalog::BrightStarCatalog),
            Some(StarNameKind::BayerFlamsteed),
        )?
        .into_iter()
        .filter(|star| {
            star.mag.is_some_and(|mag| mag <= BAYER_MAG_LIMIT)
                && figure_stars.contains(star.stable_id)
                && star
                    .designation
                    .split_whitespace()
                    .next()
                    .is_some_and(|word| word.len() > 3 && word.chars().all(char::is_alphabetic))
        })
        .collect::<Vec<_>>();
    // Prefer the spelled-out Greek letter ("Gamma Cas" over "Gam Cas").
    bayer.sort_by(|a, b| {
        a.mag
            .unwrap_or(f32::INFINITY)
            .total_cmp(&b.mag.unwrap_or(f32::INFINITY))
            .then_with(|| b.designation.len().cmp(&a.designation.len()))
    });
    for star in bayer {
        place(
            star.designation,
            star.stable_id,
            star.ra,
            star.dec,
            star.mag,
        );
    }
    marks.sort_by(|a, b| {
        a.mag
            .unwrap_or(f32::INFINITY)
            .total_cmp(&b.mag.unwrap_or(f32::INFINITY))
    });
    Ok(marks)
}

fn deep_sky_objects(
    catalog: &ObjectCatalog,
    wcs: &Wcs,
    dims: (u32, u32),
) -> Result<Vec<ObjectMark>, seiza::objects::ObjectQueryError> {
    let query = ObjectQuery {
        kinds: DSO_KINDS.to_vec(),
        sort: ObjectSort::Prominence,
        limit: Some(MAX_OBJECTS * 3),
        ..ObjectQuery::default()
    };
    let mut marks = Vec::new();
    for placed in catalog.query_footprint(wcs, dims, &query)? {
        if marks.len() >= MAX_OBJECTS {
            break;
        }
        let object = &placed.object;
        let mut label = if object.common_name.is_empty() || object.common_name == object.name {
            object.name.clone()
        } else {
            format!("{} / {}", object.name, object.common_name)
        };
        if !inside(dims, placed.x, placed.y) {
            label.push_str(" (edge)");
        }
        marks.push(ObjectMark {
            label,
            x: placed.x,
            y: placed.y,
            semi_major_px: placed.semi_major_px,
            semi_minor_px: placed.semi_minor_px,
            angle_deg: placed.angle_deg,
        });
    }
    Ok(marks)
}

/// Load, label, and write a sky map.
pub(crate) fn write(
    path: &Path,
    image: &RgbImage,
    wcs: &Wcs,
    catalogs: &SkyMapCatalogs,
    detected: &[(f64, f64)],
    summary: &SolveSummary,
    width: u32,
) -> Result<()> {
    let annotations = collect(wcs, image.dimensions(), catalogs, detected);
    let map = render(image, wcs, &annotations, summary, width)?;
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

/// Coverage mask for one colour layer, composited once so overlapping
/// strokes never double-blend.
struct Mask {
    width: usize,
    height: usize,
    coverage: Vec<f32>,
}

impl Mask {
    fn new(width: u32, height: u32) -> Self {
        Self {
            width: width as usize,
            height: height as usize,
            coverage: vec![0.0; width as usize * height as usize],
        }
    }

    fn set(&mut self, x: i64, y: i64, value: f32, clip: &Rect) {
        if (x as f64) < clip.x0
            || (y as f64) < clip.y0
            || (x as f64) >= clip.x1
            || (y as f64) >= clip.y1
            || x < 0
            || y < 0
            || x as usize >= self.width
            || y as usize >= self.height
        {
            return;
        }
        let cell = &mut self.coverage[y as usize * self.width + x as usize];
        *cell = cell.max(value.clamp(0.0, 1.0));
    }

    /// An anti-aliased line of `width` pixels.
    fn stroke(&mut self, p: (f64, f64), q: (f64, f64), width: f64, clip: &Rect) {
        let half = width / 2.0;
        let reach = half + 1.0;
        let x0 = (p.0.min(q.0) - reach).floor() as i64;
        let x1 = (p.0.max(q.0) + reach).ceil() as i64;
        let y0 = (p.1.min(q.1) - reach).floor() as i64;
        let y1 = (p.1.max(q.1) + reach).ceil() as i64;
        let (dx, dy) = (q.0 - p.0, q.1 - p.1);
        let length_sq = dx * dx + dy * dy;
        for y in y0.max(0)..=y1.min(self.height as i64 - 1) {
            for x in x0.max(0)..=x1.min(self.width as i64 - 1) {
                let (cx, cy) = (x as f64 + 0.5, y as f64 + 0.5);
                let t = if length_sq > 0.0 {
                    (((cx - p.0) * dx + (cy - p.1) * dy) / length_sq).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let distance = (cx - (p.0 + t * dx)).hypot(cy - (p.1 + t * dy));
                let value = (half + 0.5 - distance) as f32;
                if value > 0.0 {
                    self.set(x, y, value, clip);
                }
            }
        }
    }

    fn polyline(&mut self, points: &[(f64, f64)], width: f64, clip: &Rect) {
        for pair in points.windows(2) {
            self.stroke(pair[0], pair[1], width, clip);
        }
    }

    /// A ring of `radius` and line `width`; a radius of zero fills a disc.
    fn ring(&mut self, center: (f64, f64), radius: f64, width: f64, clip: &Rect) {
        let reach = radius + width / 2.0 + 1.0;
        for y in (center.1 - reach).floor() as i64..=(center.1 + reach).ceil() as i64 {
            for x in (center.0 - reach).floor() as i64..=(center.0 + reach).ceil() as i64 {
                let distance = (x as f64 + 0.5 - center.0).hypot(y as f64 + 0.5 - center.1);
                let value = (width / 2.0 + 0.5 - (distance - radius).abs()) as f32;
                if value > 0.0 {
                    self.set(x, y, value, clip);
                }
            }
        }
    }

    fn disc(&mut self, center: (f64, f64), radius: f64, clip: &Rect) {
        self.ring(center, radius / 2.0, radius, clip);
    }

    fn ellipse(
        &mut self,
        center: (f64, f64),
        semi_major: f64,
        semi_minor: f64,
        angle_deg: f64,
        width: f64,
        clip: &Rect,
    ) {
        let (sin_r, cos_r) = angle_deg.to_radians().sin_cos();
        let segments = ((semi_major * 0.5) as usize).clamp(48, 720);
        let points = (0..=segments)
            .map(|i| {
                let t = i as f64 / segments as f64 * std::f64::consts::TAU;
                let (lx, ly) = (semi_major * t.cos(), semi_minor * t.sin());
                (
                    center.0 + lx * cos_r - ly * sin_r,
                    center.1 + lx * sin_r + ly * cos_r,
                )
            })
            .collect::<Vec<_>>();
        self.polyline(&points, width, clip);
    }

    /// Fade coverage to nothing over `fade` pixels above `floor(x)`, inside
    /// `frame`.
    fn fade_below(&mut self, frame: &Rect, fade: f64, floor: impl Fn(f64) -> f64) {
        let (x0, x1) = (
            frame.x0.max(0.0) as usize,
            (frame.x1 as usize).min(self.width),
        );
        for x in x0..x1 {
            let limit = floor(x as f64 + 0.5);
            let start = ((limit - fade).max(0.0) as usize).min(self.height);
            for y in start..self.height {
                let factor = ((limit - (y as f64 + 0.5)) / fade).clamp(0.0, 1.0) as f32;
                self.coverage[y * self.width + x] *= factor;
            }
        }
    }

    /// Spread the mask by `radius` pixels with a soft edge, for a halo.
    fn dilated(&self, radius: f64) -> Mask {
        let r = radius.ceil() as i64;
        let mut out = Mask {
            width: self.width,
            height: self.height,
            coverage: vec![0.0; self.coverage.len()],
        };
        let offsets = (-r..=r)
            .flat_map(|dy| (-r..=r).map(move |dx| (dx, dy)))
            .filter_map(|(dx, dy)| {
                let weight = (radius + 0.5 - (dx as f64).hypot(dy as f64)).clamp(0.0, 1.0) as f32;
                (weight > 0.0).then_some((dx, dy, weight))
            })
            .collect::<Vec<_>>();
        for y in 0..self.height as i64 {
            for x in 0..self.width as i64 {
                let value = self.coverage[y as usize * self.width + x as usize];
                if value <= 0.0 {
                    continue;
                }
                for &(dx, dy, weight) in &offsets {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx as usize >= self.width || ny as usize >= self.height {
                        continue;
                    }
                    let cell = &mut out.coverage[ny as usize * self.width + nx as usize];
                    *cell = cell.max(value * weight);
                }
            }
        }
        out
    }

    fn composite(&self, canvas: &mut RgbImage, color: Rgb<u8>, alpha: f32) {
        for (index, &value) in self.coverage.iter().enumerate() {
            if value <= 0.0 {
                continue;
            }
            let a = value * alpha;
            let (x, y) = ((index % self.width) as u32, (index / self.width) as u32);
            let pixel = canvas.get_pixel_mut(x, y);
            for channel in 0..3 {
                let blended = pixel[channel] as f32 * (1.0 - a) + color[channel] as f32 * a;
                pixel[channel] = blended.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

struct Fonts<'a> {
    regular: FontRef<'a>,
    semibold: FontRef<'a>,
}

impl Fonts<'static> {
    fn load() -> Result<Self> {
        Ok(Self {
            regular: FontRef::try_from_slice(REGULAR_TTF).context("embedded Inter Regular")?,
            semibold: FontRef::try_from_slice(SEMIBOLD_TTF).context("embedded Inter SemiBold")?,
        })
    }
}

/// Width and line height of `text` at `size` pixels, with `tracking` extra
/// pixels between letters.
fn measure(font: &FontRef<'_>, size: f64, tracking: f64, text: &str) -> (f64, f64) {
    let scaled = font.as_scaled(PxScale::from(size as f32));
    let mut width = 0.0f64;
    let mut previous = None;
    for character in text.chars() {
        let id = scaled.glyph_id(character);
        if let Some(previous) = previous {
            width += scaled.kern(previous, id) as f64 + tracking;
        }
        width += scaled.h_advance(id) as f64;
        previous = Some(id);
    }
    (width, (scaled.ascent() - scaled.descent()) as f64)
}

/// Draw `text` with its top-left corner at `(x, y)` into a mask.
fn draw_text(
    mask: &mut Mask,
    font: &FontRef<'_>,
    size: f64,
    tracking: f64,
    (x, y): (f64, f64),
    text: &str,
) {
    let scale = PxScale::from(size as f32);
    let scaled = font.as_scaled(scale);
    let everywhere = Rect::new(0.0, 0.0, mask.width as f64, mask.height as f64);
    let baseline = y as f32 + scaled.ascent();
    let mut caret = x as f32;
    let mut previous = None;
    for character in text.chars() {
        let id = scaled.glyph_id(character);
        if let Some(previous) = previous {
            caret += scaled.kern(previous, id) + tracking as f32;
        }
        let glyph = id.with_scale_and_position(scale, point(caret, baseline));
        caret += scaled.h_advance(id);
        previous = Some(id);
        if let Some(outline) = font.outline_glyph(glyph) {
            let bounds = outline.px_bounds();
            outline.draw(|gx, gy, coverage| {
                mask.set(
                    bounds.min.x as i64 + gx as i64,
                    bounds.min.y as i64 + gy as i64,
                    coverage,
                    &everywhere,
                );
            });
        }
    }
}

/// Lay the image out under a title bar and over a footer, then draw the
/// figures, markers, and labels.
pub(crate) fn render(
    image: &RgbImage,
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
    let footer = (92.0 * u).round();
    let (source_width, source_height) = image.dimensions();
    let scale = (width as f64 - 2.0 * pad) / source_width as f64;
    let image_width = (source_width as f64 * scale).round() as u32;
    let image_height = ((source_height as f64 * scale).round() as u32).max(1);
    let height = header as u32 + image_height + footer as u32;

    let mut canvas = RgbImage::from_pixel(width, height, BACKGROUND);
    let photo = image::imageops::resize(
        image,
        image_width,
        image_height,
        image::imageops::FilterType::CatmullRom,
    );
    image::imageops::replace(&mut canvas, &photo, pad as i64, header as i64);
    let frame = Rect::new(pad, header, image_width as f64, image_height as f64);
    let to_canvas = |(x, y): (f64, f64)| (pad + x * scale, header + y * scale);

    let mut lines = Mask::new(width, height);
    let mut gold = Mask::new(width, height);
    let mut cyan = Mask::new(width, height);
    let mut names = Mask::new(width, height);
    let mut placer = LabelPlacer::new(Rect {
        x0: frame.x0 + 4.0 * u,
        y0: frame.y0 + 4.0 * u,
        x1: frame.x1 - 4.0 * u,
        y1: frame.y1 - 4.0 * u,
    });

    // Constellation figures.
    for figure in &annotations.figures {
        for polyline in &figure.polylines {
            let points = polyline.iter().copied().map(to_canvas).collect::<Vec<_>>();
            lines.polyline(&points, 1.6 * u, &frame);
        }
    }

    if let Some(floor) = &annotations.sky_floor {
        lines.fade_below(&frame, 30.0 * u, |cx| {
            header + floor.y_at((cx - pad) / scale) * scale
        });
    }

    // Markers first, reserved so no label covers one.
    let star_radius = 5.5 * u;
    let stars = annotations
        .stars
        .iter()
        .map(|star| (star, to_canvas((star.x, star.y))))
        .collect::<Vec<_>>();
    for &(_, (x, y)) in &stars {
        gold.ring((x, y), star_radius, 1.5 * u, &frame);
        placer.reserve(Rect::around(x, y, star_radius + 2.0 * u));
    }
    let objects = annotations
        .objects
        .iter()
        .map(|object| {
            let (x, y) = to_canvas((object.x, object.y));
            let major = object.semi_major_px * scale;
            let minor = match object.angle_deg {
                Some(_) => object.semi_minor_px * scale,
                None => major,
            };
            (object, (x, y), major, minor)
        })
        .collect::<Vec<_>>();
    for &(object, (x, y), major, minor) in &objects {
        if major >= 7.0 * u {
            cyan.ellipse(
                (x, y),
                major,
                minor.max(3.0 * u),
                object.angle_deg.unwrap_or(0.0),
                1.5 * u,
                &frame,
            );
        } else {
            cyan.ring((x, y), 5.0 * u, 1.4 * u, &frame);
        }
        cyan.disc((x, y), 2.4 * u, &frame);
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
                &frame,
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
    for &(object, (x, y), major, _) in &objects {
        let (w, h) = measure(&fonts.semibold, label_size, 0.0, &object.label);
        let (ax, ay) = frame.nearest_point(x, y);
        let (ax, ay) = (
            ax.clamp(frame.x0 + 8.0 * u, frame.x1 - 8.0 * u),
            ay.clamp(frame.y0 + 8.0 * u, frame.y1 - 8.0 * u),
        );
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
        let text = figure.name.to_uppercase();
        let (w, h) = measure(&fonts.regular, name_size, tracking, &text);
        // The sky anchor when it is in the frame, else the middle of the
        // longest visible piece.
        let anchor = figure.label.map(to_canvas).or_else(|| {
            figure
                .polylines
                .iter()
                .max_by(|a, b| polyline_length(a).total_cmp(&polyline_length(b)))
                .map(|line| to_canvas(line[line.len() / 2]))
        });
        let Some((x, y)) = anchor else { continue };
        if let Some(floor) = &annotations.sky_floor
            && !floor.is_sky((x - pad) / scale, (y - header) / scale)
        {
            continue;
        }
        // A sliver of a figure at the frame edge does not earn a name.
        if figure.visible_length_px() * scale < 40.0 * u {
            continue;
        }
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
        for (cell, &value) in halo.coverage.iter_mut().zip(&layer.coverage) {
            *cell = cell.max(value);
        }
    }
    halo.dilated(1.6 * u).composite(&mut canvas, SHADOW, 0.55);
    lines.composite(&mut canvas, FIGURE_LINE, 0.72);
    names.composite(&mut canvas, FIGURE_NAME, 0.92);
    cyan.composite(&mut canvas, OBJECT_CYAN, 1.0);
    gold.composite(&mut canvas, STAR_GOLD, 1.0);

    draw_chrome(
        &mut canvas,
        &fonts,
        u,
        pad,
        header,
        frame.y1,
        wcs,
        summary,
        image.dimensions(),
        annotations.sky_floor.is_some(),
    );
    Ok(canvas)
}

fn polyline_length(points: &[(f64, f64)]) -> f64 {
    points
        .windows(2)
        .map(|pair| (pair[1].0 - pair[0].0).hypot(pair[1].1 - pair[0].1))
        .sum()
}

#[allow(clippy::too_many_arguments)]
fn draw_chrome(
    canvas: &mut RgbImage,
    fonts: &Fonts<'_>,
    u: f64,
    pad: f64,
    header: f64,
    image_bottom: f64,
    wcs: &Wcs,
    summary: &SolveSummary,
    dims: (u32, u32),
    foreground: bool,
) {
    let (width, height) = canvas.dimensions();
    let text = |canvas: &mut RgbImage,
                font: &FontRef<'_>,
                size: f64,
                tracking: f64,
                at: (f64, f64),
                color: Rgb<u8>,
                content: &str| {
        let mut mask = Mask::new(width, height);
        draw_text(&mut mask, font, size, tracking, at, content);
        mask.composite(canvas, color, 1.0);
    };
    let brand = format!("SEIZA  /  {}", summary.name);
    text(
        canvas,
        &fonts.semibold,
        16.0 * u,
        0.6 * u,
        (pad, header * 0.12),
        BRAND,
        &brand,
    );
    text(
        canvas,
        &fonts.semibold,
        26.0 * u,
        0.0,
        (pad, header * 0.33),
        TITLE,
        "Catalog sky map",
    );
    let projection = if wcs.sip.is_some() { "TAN/SIP" } else { "TAN" };
    text(
        canvas,
        &fonts.regular,
        14.0 * u,
        0.0,
        (pad, header * 0.70),
        MUTED,
        &format!(
            "Constellation figures, named stars and deep-sky positions from the solved {projection} WCS"
        ),
    );

    let scale = wcs.scale_arcsec_per_px();
    let (ra, dec) = wcs.pixel_to_world(dims.0 as f64 / 2.0, dims.1 as f64 / 2.0);
    let mut stats = vec![format!("{} matched stars", summary.matched_stars)];
    if let Some(sip) = &wcs.sip {
        stats.push(format!("SIP order {}", sip.order));
    }
    stats.push(format!("{scale:.4} arcsec/px"));
    stats.push(format!("RMS {:.2} px", summary.rms_arcsec / scale));
    stats.push(format!("center RA {ra:.3}°  Dec {dec:+.3}°"));
    let mut y = image_bottom + 14.0 * u;
    text(
        canvas,
        &fonts.regular,
        15.0 * u,
        0.0,
        (pad, y),
        TITLE,
        &stats.join("  |  "),
    );
    y += 26.0 * u;
    text(
        canvas,
        &fonts.regular,
        13.5 * u,
        0.0,
        (pad, y),
        MUTED,
        if foreground {
            "Catalog marks do not establish object detection. Marks below the lowest detected stars (foreground) are left out."
        } else {
            "Catalog marks do not establish object detection."
        },
    );
    y += 22.0 * u;
    text(
        canvas,
        &fonts.regular,
        11.5 * u,
        0.0,
        (pad, y),
        DIM,
        &format!("Figures: {}", constellations::ATTRIBUTION),
    );
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

    fn catalogs(dir: &Path) -> SkyMapCatalogs {
        let path = dir.join("stars.ids.bin");
        let mut builder = StarIdentifierCatalogBuilder::new(2025.5, "test");
        let schedar = constellations::line_star(168).unwrap();
        let gamma = constellations::line_star(264).unwrap();
        builder
            .add_name(
                StarNameCatalog::IauCatalogOfStarNames,
                StarNameKind::ProperName,
                "Schedar",
                "hr:168",
                "",
                schedar.ra,
                schedar.dec,
                Some(2.24),
            )
            .unwrap();
        for designation in ["Gam Cas", "Gamma Cas", "27 Cas"] {
            builder
                .add_name(
                    StarNameCatalog::BrightStarCatalog,
                    StarNameKind::BayerFlamsteed,
                    designation,
                    "hr:264",
                    "",
                    gamma.ra,
                    gamma.dec,
                    Some(2.47),
                )
                .unwrap();
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
        let annotations = collect(&wcs, dims, &catalogs(dir.path()), &[]);
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
    }

    #[test]
    fn collect_without_catalogs_still_draws_figures() {
        let (wcs, dims) = cassiopeia();
        let annotations = collect(&wcs, dims, &SkyMapCatalogs::default(), &[]);
        assert!(!annotations.figures.is_empty());
        assert!(annotations.stars.is_empty() && annotations.objects.is_empty());
    }

    #[test]
    fn render_lays_out_header_photo_and_footer() {
        let dir = tempfile::tempdir().unwrap();
        let (wcs, dims) = cassiopeia();
        let annotations = collect(&wcs, dims, &catalogs(dir.path()), &[]);
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
        // Cassiopeia's Schedar-Gamma segment midpoint carries line colour.
        let schedar = constellations::line_star(168).unwrap();
        let gamma = constellations::line_star(264).unwrap();
        let a = wcs.world_to_pixel(schedar.ra, schedar.dec).unwrap();
        let b = wcs.world_to_pixel(gamma.ra, gamma.dec).unwrap();
        let scale = 1358.0 / 1600.0;
        let (mx, my) = (
            21.0 + (a.0 + b.0) / 2.0 * scale,
            100.0 + (a.1 + b.1) / 2.0 * scale,
        );
        // The brightest blue in a 3x3 neighbourhood (the line is anti-aliased).
        let pixel = (-1..=1)
            .flat_map(|dy| (-1..=1).map(move |dx| (dx, dy)))
            .map(|(dx, dy)| {
                *map.get_pixel(
                    (mx.round() as i64 + dx) as u32,
                    (my.round() as i64 + dy) as u32,
                )
            })
            .max_by_key(|pixel| pixel[2])
            .unwrap();
        assert!(pixel[2] > 150 && pixel[2] > pixel[0] + 40, "{pixel:?}");
        // The width is bounded.
        assert!(render(&photo, &wcs, &annotations, &summary, 100).is_err());
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
            if (sparse && y < 200.0) || (!sparse && y < 900.0) {
                valley.push((x, y));
            }
        }
        let floor = SkyFloor::from_detections(&valley, dims).unwrap();
        assert!(floor.y_at(1000.0) > 850.0, "{}", floor.y_at(1000.0));
        // Stars everywhere: no floor. Too few stars: no floor.
        let everywhere = (0..2000)
            .map(|i| ((i * 37 % 2000) as f64, (i * 53 % 997) as f64))
            .collect::<Vec<_>>();
        assert!(SkyFloor::from_detections(&everywhere, dims).is_none());
        assert!(SkyFloor::from_detections(&stars[..20], dims).is_none());
    }

    #[test]
    fn foreground_marks_are_left_out() {
        let dir = tempfile::tempdir().unwrap();
        let (wcs, dims) = cassiopeia();
        // Detections only in the top tenth of the frame.
        let detected = (0..200)
            .map(|i| ((i * 8) as f64, (i % 12) as f64 * 10.0))
            .collect::<Vec<_>>();
        let annotations = collect(&wcs, dims, &catalogs(dir.path()), &detected);
        assert!(annotations.sky_floor.is_some());
        assert!(annotations.stars.is_empty() && annotations.objects.is_empty());
        assert!(!annotations.figures.is_empty(), "lines fade, not vanish");
    }

    #[test]
    fn text_measures_and_draws() {
        let fonts = Fonts::load().unwrap();
        let (w, h) = measure(&fonts.semibold, 20.0, 0.0, "Polaris");
        assert!(w > 50.0 && w < 90.0, "{w}");
        assert!(h > 18.0 && h < 30.0, "{h}");
        let (tracked, _) = measure(&fonts.semibold, 20.0, 2.0, "Polaris");
        assert!((tracked - w - 12.0).abs() < 1e-3);
        let mut mask = Mask::new(120, 40);
        draw_text(&mut mask, &fonts.semibold, 20.0, 0.0, (2.0, 2.0), "Polaris");
        assert!(mask.coverage.iter().filter(|&&c| c > 0.5).count() > 100);
    }
}
