//! Constellation stick figures and their projection into a solved image.
//!
//! The figures come from the Constellation Lines dataset by Marc van der
//! Sluys (CC BY 4.0), embedded unmodified in
//! `data/constellation-lines/ConstellationLines.csv`. Each data line there is
//! one polyline of Bright Star Catalogue (HR) numbers. The positions of the
//! 697 stars those lines use are embedded alongside, derived once from the
//! Bright Star Catalogue (5th rev. ed., CDS V/50) at epoch 2025.5, so drawing
//! figures needs no catalog download.
//!
//! Anything that shows these figures must carry [`ATTRIBUTION`].
//!
//! [`project_figures`] turns the figures into pixel polylines through a
//! [`Wcs`]: it samples each segment along its great circle, drops the parts
//! behind the tangent plane, and clips the rest to the image so lines that
//! leave the frame reach its edge. It returns pixels only, leaving drawing
//! to the caller.

use std::collections::{BTreeSet, HashMap};
use std::sync::OnceLock;

use crate::Wcs;

/// Credit line the CC BY 4.0 licence of the line data requires: author,
/// copyright notice, source and licence link.
pub const ATTRIBUTION: &str = "Constellation Lines dataset, copyright (c) 2005-2023 Marc van der \
Sluys, hemel.waarnemen.com, DOI 10.5281/zenodo.10397192, CC BY 4.0 \
(https://creativecommons.org/licenses/by/4.0/).";

/// Upstream version of the embedded line data.
pub const DATASET_VERSION: &str = "1.3";

const LINES_CSV: &str = include_str!("../data/constellation-lines/ConstellationLines.csv");
const LINE_STARS_TSV: &str = include_str!("../data/constellation-lines/line-stars.tsv");

/// The 88 IAU constellations: abbreviation and Latin name.
pub const CONSTELLATION_NAMES: [(&str, &str); 88] = [
    ("And", "Andromeda"),
    ("Ant", "Antlia"),
    ("Aps", "Apus"),
    ("Aqr", "Aquarius"),
    ("Aql", "Aquila"),
    ("Ara", "Ara"),
    ("Ari", "Aries"),
    ("Aur", "Auriga"),
    ("Boo", "Boötes"),
    ("Cae", "Caelum"),
    ("Cam", "Camelopardalis"),
    ("Cnc", "Cancer"),
    ("CVn", "Canes Venatici"),
    ("CMa", "Canis Major"),
    ("CMi", "Canis Minor"),
    ("Cap", "Capricornus"),
    ("Car", "Carina"),
    ("Cas", "Cassiopeia"),
    ("Cen", "Centaurus"),
    ("Cep", "Cepheus"),
    ("Cet", "Cetus"),
    ("Cha", "Chamaeleon"),
    ("Cir", "Circinus"),
    ("Col", "Columba"),
    ("Com", "Coma Berenices"),
    ("CrA", "Corona Australis"),
    ("CrB", "Corona Borealis"),
    ("Crv", "Corvus"),
    ("Crt", "Crater"),
    ("Cru", "Crux"),
    ("Cyg", "Cygnus"),
    ("Del", "Delphinus"),
    ("Dor", "Dorado"),
    ("Dra", "Draco"),
    ("Equ", "Equuleus"),
    ("Eri", "Eridanus"),
    ("For", "Fornax"),
    ("Gem", "Gemini"),
    ("Gru", "Grus"),
    ("Her", "Hercules"),
    ("Hor", "Horologium"),
    ("Hya", "Hydra"),
    ("Hyi", "Hydrus"),
    ("Ind", "Indus"),
    ("Lac", "Lacerta"),
    ("Leo", "Leo"),
    ("LMi", "Leo Minor"),
    ("Lep", "Lepus"),
    ("Lib", "Libra"),
    ("Lup", "Lupus"),
    ("Lyn", "Lynx"),
    ("Lyr", "Lyra"),
    ("Men", "Mensa"),
    ("Mic", "Microscopium"),
    ("Mon", "Monoceros"),
    ("Mus", "Musca"),
    ("Nor", "Norma"),
    ("Oct", "Octans"),
    ("Oph", "Ophiuchus"),
    ("Ori", "Orion"),
    ("Pav", "Pavo"),
    ("Peg", "Pegasus"),
    ("Per", "Perseus"),
    ("Phe", "Phoenix"),
    ("Pic", "Pictor"),
    ("Psc", "Pisces"),
    ("PsA", "Piscis Austrinus"),
    ("Pup", "Puppis"),
    ("Pyx", "Pyxis"),
    ("Ret", "Reticulum"),
    ("Sge", "Sagitta"),
    ("Sgr", "Sagittarius"),
    ("Sco", "Scorpius"),
    ("Scl", "Sculptor"),
    ("Sct", "Scutum"),
    ("Ser", "Serpens"),
    ("Sex", "Sextans"),
    ("Tau", "Taurus"),
    ("Tel", "Telescopium"),
    ("Tri", "Triangulum"),
    ("TrA", "Triangulum Australe"),
    ("Tuc", "Tucana"),
    ("UMa", "Ursa Major"),
    ("UMi", "Ursa Minor"),
    ("Vel", "Vela"),
    ("Vir", "Virgo"),
    ("Vol", "Volans"),
    ("Vul", "Vulpecula"),
];

/// Latin name of a constellation from its IAU abbreviation (case-insensitive).
pub fn constellation_name(abbr: &str) -> Option<&'static str> {
    CONSTELLATION_NAMES
        .iter()
        .find(|(candidate, _)| candidate.eq_ignore_ascii_case(abbr))
        .map(|(_, name)| *name)
}

/// One star a figure line touches, at the embedded epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineStar {
    /// Bright Star Catalogue (Harvard Revised) number.
    pub hr: u32,
    /// ICRS degrees at [`LINE_STAR_EPOCH`].
    pub ra: f64,
    pub dec: f64,
    /// BSC visual magnitude.
    pub mag: f32,
}

/// Julian epoch of the embedded line-star positions.
pub const LINE_STAR_EPOCH: f64 = 2025.5;

/// One constellation's stick figure.
#[derive(Debug, Clone, PartialEq)]
pub struct Constellation {
    /// IAU abbreviation, e.g. `Cas`.
    pub abbr: &'static str,
    /// Latin name, e.g. `Cassiopeia`.
    pub name: &'static str,
    /// Unique star-to-star segments as HR pairs (lower number first).
    /// Retraced and repeated segments in the source appear once.
    pub segments: Vec<(u32, u32)>,
    /// Sky positions for the name (ICRS degrees): the normalized mean of
    /// the figure's star vectors. A figure in separate parts whose overall
    /// mean falls far from its own lines gets one position per part
    /// instead; Serpens is named at both its head (Caput) and tail (Cauda).
    pub labels: Vec<(f64, f64)>,
}

impl Constellation {
    /// Segment endpoints as ((RA, Dec), (RA, Dec)) in degrees. Segments
    /// naming a star without an embedded position are skipped; the embedded
    /// figures have none.
    pub fn segment_positions(&self) -> impl Iterator<Item = ((f64, f64), (f64, f64))> + '_ {
        self.segments.iter().filter_map(|&(a, b)| {
            let (a, b) = (line_star(a)?, line_star(b)?);
            Some(((a.ra, a.dec), (b.ra, b.dec)))
        })
    }

    /// HR numbers of every star the figure uses, ascending.
    pub fn stars(&self) -> Vec<u32> {
        self.segments
            .iter()
            .flat_map(|&(a, b)| [a, b])
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}

/// A malformed line in the embedded or supplied figure data.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("constellation data line {line}: {message}")]
pub struct ParseError {
    pub line: usize,
    pub message: String,
}

/// One polyline of the Constellation Lines CSV: the constellation it belongs
/// to and its HR numbers in drawing order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FigureLine {
    pub abbr: String,
    pub stars: Vec<u32>,
    /// One-based line number in the CSV, for error messages.
    pub line: usize,
}

/// Parse `ConstellationLines.csv`: a header line, then one polyline per
/// line as `abbr, count, hr, hr, ...`. A blank abbreviation continues the
/// previous constellation. Public for the data-regeneration example.
#[doc(hidden)]
pub fn parse_lines_csv(csv: &str) -> Result<Vec<FigureLine>, ParseError> {
    let mut lines = Vec::new();
    let mut previous: Option<String> = None;
    for (index, raw) in csv.lines().enumerate() {
        let number = index + 1;
        let error = |message: String| ParseError {
            line: number,
            message,
        };
        if index == 0 || raw.trim().is_empty() || raw.starts_with('#') {
            continue;
        }
        let fields = raw.split(',').map(str::trim).collect::<Vec<_>>();
        if fields.len() < 2 {
            return Err(error("expected an abbreviation and a star count".into()));
        }
        let abbr = if fields[0].is_empty() {
            previous
                .clone()
                .ok_or_else(|| error("continuation line before any constellation".into()))?
        } else {
            fields[0].to_string()
        };
        let count: usize = fields[1]
            .parse()
            .map_err(|_| error(format!("bad star count {:?}", fields[1])))?;
        let stars = fields[2..]
            .iter()
            .filter(|field| !field.is_empty())
            .map(|field| {
                field
                    .parse::<u32>()
                    .map_err(|_| error(format!("bad HR number {field:?}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if stars.len() != count {
            return Err(error(format!(
                "star count says {count} but the line lists {}",
                stars.len()
            )));
        }
        if stars.len() < 2 {
            return Err(error("a line needs at least two stars".into()));
        }
        previous = Some(abbr.clone());
        lines.push(FigureLine {
            abbr,
            stars,
            line: number,
        });
    }
    Ok(lines)
}

/// Parse the embedded line-star table: `hr<TAB>ra<TAB>dec<TAB>vmag` rows
/// after `#` comment lines. Each HR number may appear once.
#[doc(hidden)]
pub fn parse_line_stars_tsv(tsv: &str) -> Result<Vec<LineStar>, ParseError> {
    let mut stars = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (index, raw) in tsv.lines().enumerate() {
        let error = |message: String| ParseError {
            line: index + 1,
            message,
        };
        if raw.starts_with('#') || raw.trim().is_empty() {
            continue;
        }
        let fields = raw.split('\t').collect::<Vec<_>>();
        if fields.len() != 4 {
            return Err(error(format!("expected 4 fields, found {}", fields.len())));
        }
        let parse = |value: &str| {
            value
                .trim()
                .parse::<f64>()
                .ok()
                .filter(|value| value.is_finite())
                .ok_or_else(|| error(format!("bad number {value:?}")))
        };
        let hr = fields[0]
            .trim()
            .parse::<u32>()
            .map_err(|_| error(format!("bad HR number {:?}", fields[0])))?;
        let (ra, dec, mag) = (parse(fields[1])?, parse(fields[2])?, parse(fields[3])?);
        if !(0.0..360.0).contains(&ra) || !(-90.0..=90.0).contains(&dec) {
            return Err(error(format!("HR {hr} has an invalid position")));
        }
        if !seen.insert(hr) {
            return Err(error(format!("HR {hr} is listed twice")));
        }
        stars.push(LineStar {
            hr,
            ra,
            dec,
            mag: mag as f32,
        });
    }
    Ok(stars)
}

struct Figures {
    constellations: Vec<Constellation>,
    stars: HashMap<u32, LineStar>,
    /// The same stars in HR order, for stable iteration.
    sorted_stars: Vec<LineStar>,
}

fn loaded() -> &'static Figures {
    static FIGURES: OnceLock<Figures> = OnceLock::new();
    FIGURES.get_or_init(|| {
        build_figures(LINES_CSV, LINE_STARS_TSV).expect("embedded constellation data is valid")
    })
}

/// A figure whose overall mean lies farther than this from its own lines is
/// labelled per part.
const SPLIT_LABEL_DISTANCE_DEG: f64 = 5.0;

fn build_figures(lines_csv: &str, stars_tsv: &str) -> Result<Figures, ParseError> {
    let mut sorted_stars = parse_line_stars_tsv(stars_tsv)?;
    sorted_stars.sort_by_key(|star| star.hr);
    let stars = sorted_stars
        .iter()
        .map(|star| (star.hr, *star))
        .collect::<HashMap<_, _>>();
    // Per constellation: its unique segments and its source polylines.
    type Parsed = (&'static str, BTreeSet<(u32, u32)>, Vec<Vec<u32>>);
    let mut parsed: Vec<Parsed> = Vec::new();
    for line in parse_lines_csv(lines_csv)? {
        let error = |message: String| ParseError {
            line: line.line,
            message,
        };
        let (abbr, _) = CONSTELLATION_NAMES
            .iter()
            .find(|(abbr, _)| *abbr == line.abbr)
            .ok_or_else(|| error(format!("unknown constellation {:?}", line.abbr)))?;
        if let Some(hr) = line.stars.iter().find(|hr| !stars.contains_key(hr)) {
            return Err(error(format!("HR {hr} has no embedded position")));
        }
        let index = match parsed.iter().position(|(existing, ..)| existing == abbr) {
            Some(index) => index,
            None => {
                parsed.push((abbr, BTreeSet::new(), Vec::new()));
                parsed.len() - 1
            }
        };
        let (_, segments, polylines) = &mut parsed[index];
        for pair in line.stars.windows(2) {
            if pair[0] != pair[1] {
                segments.insert((pair[0].min(pair[1]), pair[0].max(pair[1])));
            }
        }
        polylines.push(line.stars);
    }
    let position = |hr: &u32| unit_vector(stars[hr].ra, stars[hr].dec);
    let mean = |members: &BTreeSet<u32>| {
        to_radec(members.iter().fold([0.0; 3], |sum, hr| {
            let v = position(hr);
            [sum[0] + v[0], sum[1] + v[1], sum[2] + v[2]]
        }))
    };
    let constellations = CONSTELLATION_NAMES
        .iter()
        .filter_map(|&(abbr, name)| {
            let (_, set, polylines) = parsed.iter().find(|(candidate, ..)| *candidate == abbr)?;
            let segments = set.iter().copied().collect::<Vec<_>>();
            let members = segments
                .iter()
                .flat_map(|&(a, b)| [a, b])
                .collect::<BTreeSet<_>>();
            let overall = mean(&members);
            let arcs = segments
                .iter()
                .map(|(a, b)| (position(a), position(b)))
                .collect::<Vec<_>>();
            let far = distance_to_arcs_deg(overall, &arcs) > SPLIT_LABEL_DISTANCE_DEG;
            let labels = if far && polylines.len() > 1 {
                // Name each part by the stars only it uses, so a line that
                // bridges into another part does not pull the name across.
                polylines
                    .iter()
                    .enumerate()
                    .filter_map(|(index, line)| {
                        let own = line
                            .iter()
                            .copied()
                            .filter(|hr| {
                                !polylines
                                    .iter()
                                    .enumerate()
                                    .any(|(other, line)| other != index && line.contains(hr))
                            })
                            .collect::<BTreeSet<_>>();
                        (!own.is_empty()).then(|| mean(&own))
                    })
                    .collect()
            } else {
                vec![overall]
            };
            Some(Constellation {
                abbr,
                name,
                segments,
                labels,
            })
        })
        .collect();
    Ok(Figures {
        constellations,
        stars,
        sorted_stars,
    })
}

/// Angular distance (degrees) from a sky position to the nearest of a set of
/// great-circle arcs given as unit-vector endpoints.
fn distance_to_arcs_deg(point: (f64, f64), arcs: &[([f64; 3], [f64; 3])]) -> f64 {
    let p = unit_vector(point.0, point.1);
    arcs.iter()
        .map(|&(a, b)| distance_to_arc(p, a, b))
        .fold(f64::INFINITY, f64::min)
        .to_degrees()
}

fn distance_to_arc(p: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let n = cross(a, b);
    let norm = dot(n, n).sqrt();
    let endpoints = angle(p, a).min(angle(p, b));
    if norm < 1e-12 {
        return endpoints;
    }
    let n = [n[0] / norm, n[1] / norm, n[2] / norm];
    let offset = dot(p, n);
    // Foot of the perpendicular on the great circle; on the arc when it
    // lies between the endpoints.
    let foot = [
        p[0] - offset * n[0],
        p[1] - offset * n[1],
        p[2] - offset * n[2],
    ];
    if dot(cross(a, foot), n) >= 0.0 && dot(cross(foot, b), n) >= 0.0 {
        offset.abs().clamp(0.0, 1.0).asin()
    } else {
        endpoints
    }
}

/// Every embedded constellation figure, in the order of
/// [`CONSTELLATION_NAMES`] (alphabetical by Latin name).
pub fn figures() -> &'static [Constellation] {
    &loaded().constellations
}

/// The figure for one constellation (case-insensitive abbreviation).
pub fn figure(abbr: &str) -> Option<&'static Constellation> {
    figures()
        .iter()
        .find(|figure| figure.abbr.eq_ignore_ascii_case(abbr))
}

/// Embedded position of a star used by the figures.
pub fn line_star(hr: u32) -> Option<&'static LineStar> {
    loaded().stars.get(&hr)
}

/// Every embedded line star, in HR order.
pub fn line_stars() -> impl Iterator<Item = &'static LineStar> {
    loaded().sorted_stars.iter()
}

/// A constellation figure projected into an image.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedFigure {
    pub abbr: &'static str,
    pub name: &'static str,
    /// Visible pieces as pixel polylines, each clipped to the image.
    pub polylines: Vec<Vec<(f64, f64)>>,
    /// Where to put the name: the [`Constellation::labels`] position in the
    /// image nearest its centre, or else the middle of the longest visible
    /// piece, so a narrow field crossed by one line still names it.
    pub label: Option<(f64, f64)>,
}

impl ProjectedFigure {
    /// Total drawn length in pixels.
    pub fn visible_length_px(&self) -> f64 {
        self.polylines
            .iter()
            .map(|line| polyline_length(line))
            .sum()
    }
}

/// Project every figure with a visible piece into an image of `dimensions`
/// through `wcs`.
///
/// Segments are sampled along their great circles, so long lines follow the
/// projection and partly visible ones run to the image edge. Samples behind
/// the tangent plane (`world_to_pixel` returns `None`) break a line. SIP
/// polynomials are only trusted near the image: farther out the linear TAN
/// position is used, so a distortion fit cannot fold distant sky back into
/// the frame.
pub fn project_figures(wcs: &Wcs, dimensions: (u32, u32)) -> Vec<ProjectedFigure> {
    let projector = Projector::new(wcs, dimensions);
    figures()
        .iter()
        .filter_map(|figure| {
            let polylines = figure
                .segment_positions()
                .flat_map(|(a, b)| projector.segment(a, b))
                .collect::<Vec<_>>();
            if polylines.is_empty() {
                return None;
            }
            let (cx, cy) = projector.center_px();
            let label = figure
                .labels
                .iter()
                .filter_map(|&(ra, dec)| projector.point(ra, dec))
                .filter(|&(x, y)| projector.inside(x, y))
                .min_by(|a, b| {
                    (a.0 - cx)
                        .hypot(a.1 - cy)
                        .total_cmp(&(b.0 - cx).hypot(b.1 - cy))
                })
                .or_else(|| {
                    polylines
                        .iter()
                        .max_by(|a, b| polyline_length(a).total_cmp(&polyline_length(b)))
                        .map(|line| point_along(line, polyline_length(line) / 2.0))
                });
            Some(ProjectedFigure {
                abbr: figure.abbr,
                name: figure.name,
                polylines,
                label,
            })
        })
        .collect()
}

/// Project one great-circle segment between two sky positions (degrees)
/// into clipped pixel polylines; see [`project_figures`].
#[cfg(test)]
fn project_segment(
    wcs: &Wcs,
    dimensions: (u32, u32),
    from: (f64, f64),
    to: (f64, f64),
) -> Vec<Vec<(f64, f64)>> {
    Projector::new(wcs, dimensions).segment(from, to)
}

struct Projector<'a> {
    wcs: &'a Wcs,
    linear: Wcs,
    width: f64,
    height: f64,
    center: [f64; 3],
    radius_rad: f64,
    step_rad: f64,
}

impl<'a> Projector<'a> {
    /// Beyond this fraction of the image size outside the frame, SIP is not
    /// trusted.
    const SIP_MARGIN: f64 = 0.25;
    /// Douglas-Peucker tolerance for the returned polylines.
    const SIMPLIFY_TOLERANCE_PX: f64 = 0.25;

    fn new(wcs: &'a Wcs, dimensions: (u32, u32)) -> Self {
        let (width, height) = (dimensions.0 as f64, dimensions.1 as f64);
        let (ra, dec) = wcs.pixel_to_world(width / 2.0, height / 2.0);
        let center = unit_vector(ra, dec);
        let radius_rad = wcs
            .footprint(dimensions.0, dimensions.1)
            .iter()
            .map(|&(ra, dec)| angle(center, unit_vector(ra, dec)))
            .fold(0.0, f64::max);
        let scale_rad = (wcs.scale_arcsec_per_px() / 3600.0).to_radians();
        Self {
            wcs,
            linear: Wcs {
                sip: None,
                ..wcs.clone()
            },
            width,
            height,
            center,
            radius_rad,
            // About 12 px between samples, kept between 0.02° and 1°.
            step_rad: (12.0 * scale_rad).clamp(0.02_f64.to_radians(), 1.0_f64.to_radians()),
        }
    }

    /// Pixel coordinates are zero-based centres, so the image spans
    /// -0.5 to size - 0.5.
    fn inside(&self, x: f64, y: f64) -> bool {
        x >= -0.5 && y >= -0.5 && x <= self.width - 0.5 && y <= self.height - 0.5
    }

    fn center_px(&self) -> (f64, f64) {
        ((self.width - 1.0) / 2.0, (self.height - 1.0) / 2.0)
    }

    fn point(&self, ra: f64, dec: f64) -> Option<(f64, f64)> {
        let linear = self.linear.world_to_pixel(ra, dec)?;
        if self.wcs.sip.is_none() {
            return Some(linear);
        }
        let (mx, my) = (
            self.width * Self::SIP_MARGIN,
            self.height * Self::SIP_MARGIN,
        );
        let near = linear.0 >= -mx
            && linear.1 >= -my
            && linear.0 <= self.width + mx
            && linear.1 <= self.height + my;
        if near {
            self.wcs.world_to_pixel(ra, dec)
        } else {
            Some(linear)
        }
    }

    fn segment(&self, from: (f64, f64), to: (f64, f64)) -> Vec<Vec<(f64, f64)>> {
        let (a, b) = (unit_vector(from.0, from.1), unit_vector(to.0, to.1));
        let length = angle(a, b);
        if !length.is_finite() || length <= 0.0 {
            return Vec::new();
        }
        // Every point of the arc is within half its length of the midpoint.
        let middle = slerp(a, b, length, 0.5);
        if angle(self.center, middle) > self.radius_rad + length / 2.0 + 1e-6 {
            return Vec::new();
        }
        let steps = ((length / self.step_rad).ceil() as usize).max(1);
        let samples = (0..=steps)
            .map(|step| {
                let (ra, dec) = to_radec(slerp(a, b, length, step as f64 / steps as f64));
                self.point(ra, dec)
            })
            .collect::<Vec<_>>();

        let mut polylines = Vec::new();
        let mut current: Vec<(f64, f64)> = Vec::new();
        for pair in samples.windows(2) {
            let clipped = match (pair[0], pair[1]) {
                (Some(p), Some(q)) => clip_segment(p, q, self.width, self.height),
                _ => None,
            };
            match clipped {
                Some((p, q)) => {
                    let joins = current.last().is_some_and(|last| {
                        (last.0 - p.0).abs() < 1e-6 && (last.1 - p.1).abs() < 1e-6
                    });
                    if !joins && !current.is_empty() {
                        polylines.push(std::mem::take(&mut current));
                    }
                    if current.is_empty() {
                        current.push(p);
                    }
                    current.push(q);
                }
                None => {
                    if !current.is_empty() {
                        polylines.push(std::mem::take(&mut current));
                    }
                }
            }
        }
        if !current.is_empty() {
            polylines.push(current);
        }
        polylines.retain(|line| polyline_length(line) > 0.5);
        // A TAN projection draws a great circle as a straight line, so most
        // samples are redundant; keep only what bends by a quarter pixel.
        polylines
            .into_iter()
            .map(|line| simplify(&line, Self::SIMPLIFY_TOLERANCE_PX))
            .collect()
    }
}

/// Liang-Barsky clip of the segment `p`-`q` to the pixel area of a
/// `width` x `height` image, `[-0.5, width - 0.5] x [-0.5, height - 0.5]`.
/// Non-finite input clips to nothing.
fn clip_segment(
    p: (f64, f64),
    q: (f64, f64),
    width: f64,
    height: f64,
) -> Option<((f64, f64), (f64, f64))> {
    if ![p.0, p.1, q.0, q.1].iter().all(|value| value.is_finite()) {
        return None;
    }
    // Shift so the pixel area starts at the origin, clip, shift back.
    let shift = |(x, y): (f64, f64), by: f64| (x + by, y + by);
    let (p, q) = (shift(p, 0.5), shift(q, 0.5));
    let (dx, dy) = (q.0 - p.0, q.1 - p.1);
    let (mut t0, mut t1) = (0.0_f64, 1.0_f64);
    for (denominator, numerator) in [
        (-dx, p.0),
        (dx, width - p.0),
        (-dy, p.1),
        (dy, height - p.1),
    ] {
        if denominator == 0.0 {
            if numerator < 0.0 {
                return None;
            }
            continue;
        }
        let t = numerator / denominator;
        if denominator < 0.0 {
            if t > t1 {
                return None;
            }
            t0 = t0.max(t);
        } else {
            if t < t0 {
                return None;
            }
            t1 = t1.min(t);
        }
    }
    (t0 <= t1).then_some((
        shift((p.0 + t0 * dx, p.1 + t0 * dy), -0.5),
        shift((p.0 + t1 * dx, p.1 + t1 * dy), -0.5),
    ))
}

fn polyline_length(line: &[(f64, f64)]) -> f64 {
    line.windows(2)
        .map(|pair| (pair[1].0 - pair[0].0).hypot(pair[1].1 - pair[0].1))
        .sum()
}

/// The point `distance` pixels along a polyline from its start.
fn point_along(line: &[(f64, f64)], distance: f64) -> (f64, f64) {
    let mut remaining = distance;
    for pair in line.windows(2) {
        let length = (pair[1].0 - pair[0].0).hypot(pair[1].1 - pair[0].1);
        if remaining <= length && length > 0.0 {
            let t = remaining / length;
            return (
                pair[0].0 + t * (pair[1].0 - pair[0].0),
                pair[0].1 + t * (pair[1].1 - pair[0].1),
            );
        }
        remaining -= length;
    }
    *line.last().expect("polylines have points")
}

/// Douglas-Peucker simplification keeping the endpoints.
fn simplify(line: &[(f64, f64)], tolerance: f64) -> Vec<(f64, f64)> {
    if line.len() < 3 {
        return line.to_vec();
    }
    let mut keep = vec![false; line.len()];
    keep[0] = true;
    keep[line.len() - 1] = true;
    let mut stack = vec![(0, line.len() - 1)];
    while let Some((first, last)) = stack.pop() {
        if last - first < 2 {
            continue;
        }
        let (a, b) = (line[first], line[last]);
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let length = dx.hypot(dy);
        let (index, distance) = (first + 1..last)
            .map(|index| {
                let p = line[index];
                let distance = if length > 0.0 {
                    ((p.0 - a.0) * dy - (p.1 - a.1) * dx).abs() / length
                } else {
                    (p.0 - a.0).hypot(p.1 - a.1)
                };
                (index, distance)
            })
            .max_by(|x, y| x.1.total_cmp(&y.1))
            .expect("at least one interior point");
        if distance > tolerance {
            keep[index] = true;
            stack.push((first, index));
            stack.push((index, last));
        }
    }
    line.iter()
        .zip(keep)
        .filter_map(|(point, kept)| kept.then_some(*point))
        .collect()
}

/// Angular separation between two sky positions, degrees.
pub fn separation_deg(a: (f64, f64), b: (f64, f64)) -> f64 {
    angle(unit_vector(a.0, a.1), unit_vector(b.0, b.1)).to_degrees()
}

fn unit_vector(ra: f64, dec: f64) -> [f64; 3] {
    let (sin_ra, cos_ra) = ra.to_radians().sin_cos();
    let (sin_dec, cos_dec) = dec.to_radians().sin_cos();
    [cos_dec * cos_ra, cos_dec * sin_ra, sin_dec]
}

fn to_radec(v: [f64; 3]) -> (f64, f64) {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let ra = v[1].atan2(v[0]).to_degrees().rem_euclid(360.0);
    let dec = (v[2] / norm).clamp(-1.0, 1.0).asin().to_degrees();
    (ra, dec)
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn angle(a: [f64; 3], b: [f64; 3]) -> f64 {
    let c = cross(a, b);
    dot(c, c).sqrt().atan2(dot(a, b))
}

fn slerp(a: [f64; 3], b: [f64; 3], length: f64, t: f64) -> [f64; 3] {
    let sin = length.sin();
    if sin.abs() < 1e-12 {
        return a;
    }
    let (wa, wb) = (((1.0 - t) * length).sin() / sin, (t * length).sin() / sin);
    [
        wa * a[0] + wb * b[0],
        wa * a[1] + wb * b[1],
        wa * a[2] + wb * b[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_data_covers_all_88_constellations() {
        let figures = figures();
        assert_eq!(figures.len(), 88);
        let names = figures.iter().map(|f| f.abbr).collect::<BTreeSet<_>>();
        assert_eq!(names.len(), 88);
        for (abbr, name) in CONSTELLATION_NAMES {
            let figure = figure(abbr).unwrap_or_else(|| panic!("{abbr} has no figure"));
            assert_eq!(figure.name, name);
            assert!(!figure.segments.is_empty(), "{abbr} has no segments");
        }
    }

    #[test]
    fn every_line_star_resolves_and_segments_are_short() {
        let csv = parse_lines_csv(LINES_CSV).unwrap();
        assert_eq!(csv.len(), 90, "upstream v1.3 has 90 polylines");
        let used = csv
            .iter()
            .flat_map(|line| line.stars.iter().copied())
            .collect::<BTreeSet<_>>();
        assert_eq!(used.len(), 697);
        for hr in &used {
            assert!(line_star(*hr).is_some(), "HR {hr} unresolved");
            assert!((1..=9110).contains(hr));
        }
        assert_eq!(line_stars().count(), used.len(), "no unused positions");
        for figure in figures() {
            for ((a, b), (from, to)) in figure.segments.iter().zip(figure.segment_positions()) {
                let length = separation_deg(from, to);
                assert!(
                    length > 0.05 && length < 40.0,
                    "{} HR {a}-{b} is {length:.1} degrees",
                    figure.abbr
                );
            }
        }
    }

    #[test]
    fn segments_are_deduplicated_and_retraced_lines_collapse() {
        // Andromeda's single source line retraces 165 and 337 several times.
        let and = figure("And").unwrap();
        let unique = and.segments.iter().collect::<BTreeSet<_>>();
        assert_eq!(unique.len(), and.segments.len());
        assert!(
            and.segments.len() < 19,
            "20 stars, retraced to fewer segments"
        );
        // Cassiopeia's W: Caph-Schedar-Gamma-Ruchbah-Segin.
        let cas = figure("Cas").unwrap();
        assert_eq!(
            cas.segments,
            vec![(21, 168), (168, 264), (264, 403), (403, 542)]
        );
        // Crux uses two source lines; Serpens two (Cauda and Caput).
        assert_eq!(figure("Cru").unwrap().segments.len(), 2);
        assert_eq!(figure("Ser").unwrap().segments.len(), 4 + 8);
    }

    #[test]
    fn known_star_positions_match() {
        // Polaris (HR 424) and Vega (HR 7001) near epoch 2025.5.
        let polaris = line_star(424).unwrap();
        assert!(separation_deg((polaris.ra, polaris.dec), (37.95, 89.264)) < 0.05);
        let vega = line_star(7001).unwrap();
        assert!(separation_deg((vega.ra, vega.dec), (279.2347, 38.7837)) < 0.01);
        assert!((vega.mag - 0.03).abs() < 0.1);
    }

    #[test]
    fn label_anchor_sits_among_the_figure() {
        let cas = figure("Cas").unwrap();
        let farthest = cas
            .stars()
            .into_iter()
            .map(|hr| {
                let star = line_star(hr).unwrap();
                separation_deg(cas.labels[0], (star.ra, star.dec))
            })
            .fold(0.0, f64::max);
        assert!(farthest < 10.0, "{farthest}");
        assert_eq!(cas.labels.len(), 1);
        assert_eq!(constellation_name("cas"), Some("Cassiopeia"));
        assert_eq!(constellation_name("Xyz"), None);
    }

    /// Every name position lies near its own figure's lines, and no more
    /// than a degree nearer any other figure's, so no name lands inside a
    /// neighbour. (The single mean put "SERPENS" 10.6° from its own lines
    /// and 1.8° from Ophiuchus'.)
    #[test]
    fn every_name_position_is_nearest_its_own_figure() {
        let arcs = |figure: &Constellation| {
            figure
                .segment_positions()
                .map(|(a, b)| (unit_vector(a.0, a.1), unit_vector(b.0, b.1)))
                .collect::<Vec<_>>()
        };
        let all = figures()
            .iter()
            .map(|f| (f.abbr, arcs(f)))
            .collect::<Vec<_>>();
        for figure in figures() {
            for &label in &figure.labels {
                let own = distance_to_arcs_deg(label, &arcs(figure));
                for (abbr, other) in &all {
                    if *abbr != figure.abbr {
                        let theirs = distance_to_arcs_deg(label, other);
                        assert!(
                            own <= theirs + 1.0,
                            "{} name at {label:?} is {own:.1}° from its lines but {theirs:.1}° from {abbr}",
                            figure.abbr
                        );
                    }
                }
            }
        }
        // Only Serpens, whose two parts lie either side of Ophiuchus, is
        // split; ring-shaped figures keep their single central name.
        let split = figures()
            .iter()
            .filter(|f| f.labels.len() > 1)
            .map(|f| f.abbr)
            .collect::<Vec<_>>();
        assert_eq!(split, ["Ser"]);
        // Serpens is named at both its head and its tail.
        let ser = figure("Ser").unwrap();
        assert_eq!(ser.labels.len(), 2);
        let oph_label = figure("Oph").unwrap().labels[0];
        for &label in &ser.labels {
            assert!(separation_deg(label, oph_label) > 10.0, "{label:?}");
        }
    }

    #[test]
    fn hand_built_figures_with_unknown_stars_do_not_panic() {
        let figure = Constellation {
            abbr: "Xyz",
            name: "Nowhere",
            segments: vec![(1, 2), (21, 168)],
            labels: Vec::new(),
        };
        assert_eq!(figure.segment_positions().count(), 1);
    }

    #[test]
    fn parser_rejects_bad_counts_and_handles_continuations() {
        let csv = "abr, nr, s01, s02, s03\nCru,  2, 4853, 4656,\n   ,  2, 4730, 4763,\n";
        let lines = parse_lines_csv(csv).unwrap();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[1].abbr, "Cru");
        assert_eq!(lines[1].stars, vec![4730, 4763]);
        let error = parse_lines_csv("abr, nr\nCas, 3, 1, 2,\n").unwrap_err();
        assert_eq!(error.line, 2);
        assert!(parse_lines_csv("abr, nr\n   , 2, 1, 2\n").is_err());
        assert!(parse_line_stars_tsv("# c\n1\t400.0\t0\t1\n").is_err());
        let twice = parse_line_stars_tsv("1\t10\t0\t1\n1\t11\t0\t1\n").unwrap_err();
        assert_eq!(twice.line, 2);
        // Errors name the real CSV line, past blank and comment lines.
        let error = parse_lines_csv("abr, nr\n\n# note\nCas, 3, 1, 2,\n").unwrap_err();
        assert_eq!(error.line, 4);
        assert_eq!(lines[0].line, 2);
    }

    #[test]
    fn clip_segment_cases() {
        // A 10-pixel image spans -0.5 to 9.5 in pixel-centre coordinates.
        assert_eq!(
            clip_segment((-10.0, 5.0), (20.0, 5.0), 10.0, 10.0),
            Some(((-0.5, 5.0), (9.5, 5.0)))
        );
        assert_eq!(clip_segment((-10.0, -5.0), (-1.0, 20.0), 10.0, 10.0), None);
        assert_eq!(
            clip_segment((2.0, 2.0), (3.0, 4.0), 10.0, 10.0),
            Some(((2.0, 2.0), (3.0, 4.0)))
        );
        let ((x0, y0), (x1, y1)) = clip_segment((5.0, 5.0), (15.0, 15.0), 10.0, 10.0).unwrap();
        assert_eq!((x0, y0), (5.0, 5.0));
        assert!((x1 - 9.5).abs() < 1e-12 && (y1 - 9.5).abs() < 1e-12);
        assert_eq!(clip_segment((f64::NAN, 1.0), (2.0, 2.0), 10.0, 10.0), None);
    }

    #[test]
    fn simplification_keeps_bends_and_drops_collinear_points() {
        let straight = (0..=10)
            .map(|i| (i as f64, 2.0 * i as f64))
            .collect::<Vec<_>>();
        assert_eq!(simplify(&straight, 0.25), vec![(0.0, 0.0), (10.0, 20.0)]);
        let bent = vec![(0.0, 0.0), (5.0, 0.1), (10.0, 0.0), (10.0, 10.0)];
        assert_eq!(
            simplify(&bent, 0.25),
            vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        );
        assert_eq!(
            point_along(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)], 15.0),
            (10.0, 5.0)
        );
    }

    fn cassiopeia_wcs() -> (Wcs, (u32, u32)) {
        // 40 degrees across at 0.02 deg/px, centered between Schedar and
        // Ruchbah, north up.
        let dims = (2000, 1500);
        let wcs = Wcs::from_center_scale_rotation((14.0, 60.0), (1000.0, 750.0), 72.0, 0.0, false);
        (wcs, dims)
    }

    #[test]
    fn projection_draws_cassiopeia_inside_and_clips_partial_lines() {
        let (wcs, dims) = cassiopeia_wcs();
        let projected = project_figures(&wcs, dims);
        let cas = projected.iter().find(|f| f.abbr == "Cas").unwrap();
        assert!(cas.label.is_some());
        // Without SIP every projected segment is a straight line: two points.
        assert!(
            projected
                .iter()
                .flat_map(|f| &f.polylines)
                .all(|line| line.len() == 2)
        );
        // Each W segment is fully inside: endpoints land on the stars.
        let caph = line_star(21).unwrap();
        let (cx, cy) = wcs.world_to_pixel(caph.ra, caph.dec).unwrap();
        assert!(
            cas.polylines
                .iter()
                .flat_map(|line| [line[0], *line.last().unwrap()])
                .any(|(x, y)| (x - cx).hypot(y - cy) < 0.5)
        );
        for figure in &projected {
            for line in &figure.polylines {
                for &(x, y) in line {
                    assert!(
                        (-0.5 - 1e-6..=dims.0 as f64 - 0.5 + 1e-6).contains(&x)
                            && (-0.5 - 1e-6..=dims.1 as f64 - 0.5 + 1e-6).contains(&y),
                        "{} point ({x}, {y}) outside the image",
                        figure.abbr
                    );
                }
            }
        }
        // Some figure is cut by the frame: a polyline ends on an edge.
        let on_edge = |(x, y): (f64, f64)| {
            (x + 0.5).abs() < 1e-6
                || (y + 0.5).abs() < 1e-6
                || (x - dims.0 as f64 + 0.5).abs() < 1e-6
                || (y - dims.1 as f64 + 0.5).abs() < 1e-6
        };
        assert!(
            projected
                .iter()
                .flat_map(|f| f.polylines.iter())
                .any(|line| on_edge(line[0]) || on_edge(*line.last().unwrap()))
        );
        // Far-away constellations are absent.
        assert!(projected.iter().all(|f| f.abbr != "Cru" && f.abbr != "Sco"));
    }

    #[test]
    fn a_narrow_field_crossed_by_a_line_still_gets_a_name() {
        // 1 degree around the middle of Orion's belt: the figure's own name
        // position is far outside, so the name goes on the visible line.
        let alnitak = line_star(1948).unwrap();
        let mintaka = line_star(1852).unwrap();
        let middle = (
            (alnitak.ra + mintaka.ra) / 2.0,
            (alnitak.dec + mintaka.dec) / 2.0,
        );
        let wcs = Wcs::from_center_scale_rotation(middle, (500.0, 500.0), 3.6, 0.0, false);
        let projected = project_figures(&wcs, (1000, 1000));
        let orion = projected.iter().find(|f| f.abbr == "Ori").expect("Orion");
        let (x, y) = orion.label.expect("named on its line");
        assert!((-0.5..=999.5).contains(&x) && (-0.5..=999.5).contains(&y));
        let on_line = orion.polylines.iter().any(|line| {
            line.windows(2).any(|pair| {
                let ((x0, y0), (x1, y1)) = (pair[0], pair[1]);
                let length = (x1 - x0).hypot(y1 - y0);
                ((x - x0) * (y1 - y0) - (y - y0) * (x1 - x0)).abs() / length < 1e-6
            })
        });
        assert!(on_line, "{x}, {y}");
    }

    #[test]
    fn segments_behind_the_tangent_plane_are_skipped() {
        let (wcs, dims) = cassiopeia_wcs();
        // From near the field center to the opposite side of the sky: only
        // the part in front of the tangent plane can be drawn, and it must
        // still be clipped to the image.
        let lines = project_segment(&wcs, dims, (14.0, 60.0), (194.0, -50.0));
        assert!(!lines.is_empty());
        for line in &lines {
            for &(x, y) in line {
                assert!((-0.5..=1999.5).contains(&x) && (-0.5..=1499.5).contains(&y));
            }
        }
        // Wholly behind: nothing.
        assert!(project_segment(&wcs, dims, (194.0, -40.0), (200.0, -45.0)).is_empty());
    }

    #[test]
    fn sip_is_not_trusted_far_outside_the_frame() {
        let (mut wcs, dims) = cassiopeia_wcs();
        // A wild inverse polynomial that would fold distant points inward.
        let mut sip = crate::Sip {
            order: 2,
            a: vec![0.0; 6],
            b: vec![0.0; 6],
            ap: vec![0.0; 6],
            bp: vec![0.0; 6],
        };
        sip.ap[3] = -2e-3;
        sip.bp[5] = -2e-3;
        wcs.sip = Some(sip);
        let projector = Projector::new(&wcs, dims);
        let far = wcs.pixel_to_world(-6000.0, 750.0);
        let (x, _) = projector.point(far.0, far.1).unwrap();
        assert!(x < -1000.0, "{x}");
    }
}
