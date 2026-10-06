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

/// Credit line the CC BY 4.0 licence of the line data requires.
pub const ATTRIBUTION: &str = "Constellation Lines dataset by Marc van der Sluys (2005-2023), \
hemel.waarnemen.com. DOI: 10.5281/zenodo.10397192. Licensed under CC BY 4.0.";

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
    /// Sky position for a name label: the normalized mean of the unit
    /// vectors of the figure's stars (ICRS degrees).
    pub label: (f64, f64),
}

impl Constellation {
    /// Segment endpoints as ((RA, Dec), (RA, Dec)) in degrees.
    pub fn segment_positions(&self) -> impl Iterator<Item = ((f64, f64), (f64, f64))> + '_ {
        self.segments.iter().map(|&(a, b)| {
            let a = line_star(a).expect("figure stars are checked when the data loads");
            let b = line_star(b).expect("figure stars are checked when the data loads");
            ((a.ra, a.dec), (b.ra, b.dec))
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
}

/// Parse `ConstellationLines.csv`: a header line, then one polyline per
/// line as `abbr, count, hr, hr, ...`. A blank abbreviation continues the
/// previous constellation.
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
        lines.push(FigureLine { abbr, stars });
    }
    Ok(lines)
}

/// Parse the embedded line-star table: `hr<TAB>ra<TAB>dec<TAB>vmag` rows
/// after `#` comment lines.
pub fn parse_line_stars_tsv(tsv: &str) -> Result<Vec<LineStar>, ParseError> {
    let mut stars = Vec::new();
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
}

fn loaded() -> &'static Figures {
    static FIGURES: OnceLock<Figures> = OnceLock::new();
    FIGURES.get_or_init(|| {
        build_figures(LINES_CSV, LINE_STARS_TSV).expect("embedded constellation data is valid")
    })
}

fn build_figures(lines_csv: &str, stars_tsv: &str) -> Result<Figures, ParseError> {
    let stars = parse_line_stars_tsv(stars_tsv)?
        .into_iter()
        .map(|star| (star.hr, star))
        .collect::<HashMap<_, _>>();
    let mut segments: Vec<(&'static str, BTreeSet<(u32, u32)>)> = Vec::new();
    for (index, line) in parse_lines_csv(lines_csv)?.into_iter().enumerate() {
        let error = |message: String| ParseError {
            line: index + 2,
            message,
        };
        let (abbr, _) = CONSTELLATION_NAMES
            .iter()
            .find(|(abbr, _)| *abbr == line.abbr)
            .ok_or_else(|| error(format!("unknown constellation {:?}", line.abbr)))?;
        if let Some(hr) = line.stars.iter().find(|hr| !stars.contains_key(hr)) {
            return Err(error(format!("HR {hr} has no embedded position")));
        }
        let entry = match segments.iter().position(|(existing, _)| existing == abbr) {
            Some(position) => &mut segments[position].1,
            None => {
                segments.push((abbr, BTreeSet::new()));
                &mut segments.last_mut().expect("just pushed").1
            }
        };
        for pair in line.stars.windows(2) {
            if pair[0] != pair[1] {
                entry.insert((pair[0].min(pair[1]), pair[0].max(pair[1])));
            }
        }
    }
    let constellations = CONSTELLATION_NAMES
        .iter()
        .filter_map(|&(abbr, name)| {
            let (_, set) = segments.iter().find(|(candidate, _)| *candidate == abbr)?;
            let segments = set.iter().copied().collect::<Vec<_>>();
            let members = segments
                .iter()
                .flat_map(|&(a, b)| [a, b])
                .collect::<BTreeSet<_>>();
            let sum = members.iter().fold([0.0; 3], |sum, hr| {
                let v = unit_vector(stars[hr].ra, stars[hr].dec);
                [sum[0] + v[0], sum[1] + v[1], sum[2] + v[2]]
            });
            Some(Constellation {
                abbr,
                name,
                segments,
                label: to_radec(sum),
            })
        })
        .collect();
    Ok(Figures {
        constellations,
        stars,
    })
}

/// Every embedded constellation figure, in IAU abbreviation order.
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

/// Every embedded line star, in no particular order.
pub fn line_stars() -> impl Iterator<Item = &'static LineStar> {
    loaded().stars.values()
}

/// A constellation figure projected into an image.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedFigure {
    pub abbr: &'static str,
    pub name: &'static str,
    /// Visible pieces as pixel polylines, each clipped to the image.
    pub polylines: Vec<Vec<(f64, f64)>>,
    /// Pixel position of [`Constellation::label`] when it lands in the image.
    pub label: Option<(f64, f64)>,
}

impl ProjectedFigure {
    /// Total drawn length in pixels.
    pub fn visible_length_px(&self) -> f64 {
        self.polylines
            .iter()
            .flat_map(|line| line.windows(2))
            .map(|pair| (pair[1].0 - pair[0].0).hypot(pair[1].1 - pair[0].1))
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
            let label = projector
                .point(figure.label.0, figure.label.1)
                .filter(|&(x, y)| projector.inside(x, y));
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
pub fn project_segment(
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

    fn inside(&self, x: f64, y: f64) -> bool {
        x >= 0.0 && y >= 0.0 && x <= self.width && y <= self.height
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
        polylines.retain(|line| {
            line.windows(2)
                .map(|pair| (pair[1].0 - pair[0].0).hypot(pair[1].1 - pair[0].1))
                .sum::<f64>()
                > 0.5
        });
        polylines
    }
}

/// Liang-Barsky clip of the segment `p`-`q` to `[0, width] x [0, height]`.
pub fn clip_segment(
    p: (f64, f64),
    q: (f64, f64),
    width: f64,
    height: f64,
) -> Option<((f64, f64), (f64, f64))> {
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
        (p.0 + t0 * dx, p.1 + t0 * dy),
        (p.0 + t1 * dx, p.1 + t1 * dy),
    ))
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

fn angle(a: [f64; 3], b: [f64; 3]) -> f64 {
    let cross = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let sin = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    let cos = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    sin.atan2(cos)
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
                separation_deg(cas.label, (star.ra, star.dec))
            })
            .fold(0.0, f64::max);
        assert!(farthest < 10.0, "{farthest}");
        assert_eq!(constellation_name("cas"), Some("Cassiopeia"));
        assert_eq!(constellation_name("Xyz"), None);
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
    }

    #[test]
    fn clip_segment_cases() {
        assert_eq!(
            clip_segment((-10.0, 5.0), (20.0, 5.0), 10.0, 10.0),
            Some(((0.0, 5.0), (10.0, 5.0)))
        );
        assert_eq!(clip_segment((-10.0, -5.0), (-1.0, 20.0), 10.0, 10.0), None);
        assert_eq!(
            clip_segment((2.0, 2.0), (3.0, 4.0), 10.0, 10.0),
            Some(((2.0, 2.0), (3.0, 4.0)))
        );
        let ((x0, y0), (x1, y1)) = clip_segment((5.0, 5.0), (15.0, 15.0), 10.0, 10.0).unwrap();
        assert_eq!((x0, y0), (5.0, 5.0));
        assert!((x1 - 10.0).abs() < 1e-12 && (y1 - 10.0).abs() < 1e-12);
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
                        (-1e-6..=dims.0 as f64 + 1e-6).contains(&x)
                            && (-1e-6..=dims.1 as f64 + 1e-6).contains(&y),
                        "{} point ({x}, {y}) outside the image",
                        figure.abbr
                    );
                }
            }
        }
        // Some figure is cut by the frame: a polyline ends on an edge.
        let on_edge = |(x, y): (f64, f64)| {
            x.abs() < 1e-6
                || y.abs() < 1e-6
                || (x - dims.0 as f64).abs() < 1e-6
                || (y - dims.1 as f64).abs() < 1e-6
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
    fn segments_behind_the_tangent_plane_are_skipped() {
        let (wcs, dims) = cassiopeia_wcs();
        // From near the field center to the opposite side of the sky: only
        // the part in front of the tangent plane can be drawn, and it must
        // still be clipped to the image.
        let lines = project_segment(&wcs, dims, (14.0, 60.0), (194.0, -50.0));
        assert!(!lines.is_empty());
        for line in &lines {
            for &(x, y) in line {
                assert!((0.0..=2000.0).contains(&x) && (0.0..=1500.0).contains(&y));
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
