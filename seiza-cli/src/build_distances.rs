//! Builds `object-distances.bin` from the catalogues `download-data
//! object-distances` fetches, matched to the objects in `objects.bin`.
//!
//! Each source gives some objects a distance. When several do, the source
//! earliest in [`Tier`] wins. An object that none measure may borrow the
//! distance of a galaxy it lies in, of a Galactic object that contains it
//! or lies inside it, or, for a dark nebula, of the nearest molecular-cloud
//! sightline.

use crate::build_data::designation_key;
use anyhow::{Context, Result, bail};
use seiza::catalog::distances::{
    DistanceBasis, DistanceEntry, DistanceMethod, DistanceSource, KindDistance,
    ObjectDistancesBuilder,
};
use seiza::objects::{ObjectCatalog, ObjectKind, SkyObject};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

/// Hubble constant for redshift distances, km/s/Mpc. Cosmicflows-4 sets its
/// distance scale with 75, so redshift distances sit on the same scale.
const HUBBLE_CONSTANT: f64 = 75.0;
const SPEED_OF_LIGHT_KM_S: f64 = 299_792.458;
/// Peculiar velocities blur a redshift distance by about this much, km/s.
const PECULIAR_VELOCITY: f64 = 300.0;
/// Below this redshift (cz = 300 km/s) peculiar motion swamps the Hubble flow.
const MIN_REDSHIFT: f64 = 0.001;
/// A galaxy lends its distance to the clusters and nebulae inside it only
/// when it lies beyond the Milky Way's nearest satellites, so a dwarf
/// spread across the Galactic bulge cannot claim bulge objects.
const MIN_HOST_DISTANCE_PC: f64 = 40_000.0;
const MAX_HOST_DISTANCE_PC: f64 = 5.0e6;
/// A dark nebula is seen in silhouette, so it only borrows the distance of
/// a neighbour closer than this.
const MAX_DARK_CLOUD_PC: f64 = 2000.0;
/// How far a dark nebula may lie from the cloud sightline it borrows from.
const NEARBY_CLOUD_DEG: f64 = 1.0;
/// Greatest separation between a source's position and the object its name
/// matches, beyond the object's own extent.
const NAME_MATCH_DEG: f64 = 1.0;
/// No galaxy in the catalog is nearer than the Sagittarius dwarf, 26 kpc.
const MIN_GALAXY_PC: f64 = 15_000.0;
/// Measurements published from this year on use Gaia-era data.
const GAIA_ERA: u32 = 2018;
/// Hunt & Reffert distances wider than this, as half the 16th–84th
/// percentile range over the median, come from parallaxes too small to
/// measure (NGC 2419 at 90 kpc reads as 230), so a later source decides.
const MAX_CLUSTER_SPREAD: f64 = 0.2;

const NOTICE: &str = "Distances to the objects in Seiza's object catalog. \
Released under the Open Database License (ODbL 1.0, \
https://opendatacommons.org/licenses/odbl/1-0/), as SIMBAD's terms require \
of a database built from it. Contains information from SIMBAD, operated at \
CDS, Strasbourg, France (Wenger et al. 2000, A&AS 143, 9), made available \
under the ODbL. This product used the VizieR catalogue access tool, CDS, \
Strasbourg, France (DOI 10.26093/cds/vizier). Each source below is cited \
with its own terms.";

const VIZIER_TERMS: &str = "VizieR: free for scientific use; cite the publication";

/// Sources in priority order.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum Tier {
    /// Open and globular clusters from Gaia DR3 member parallaxes.
    HuntReffert2024,
    /// Redshift-independent galaxy distances.
    Cosmicflows4,
    /// Planetary nebulae: central-star parallax combined with a statistical
    /// prior.
    ChornayWalton2021,
    /// Planetary nebulae: central-star parallax.
    GonzalezSantamaria2021,
    /// HII regions from the spectrophotometric distances of their stars.
    FosterBrunt2015,
    /// Supernova remnants.
    RanasingheLeahy2023,
    /// Molecular clouds named in the Star Formation Handbook sightlines.
    Zucker2020,
    /// Reflection nebulae from the parallax of the star that lights them.
    IlluminatingStar,
    /// Globular clusters.
    Harris1997,
    /// SIMBAD's distance measurements.
    Simbad,
    /// Planetary nebulae: statistical distances.
    StanghelliniHaywood2010,
    /// Lynds dark clouds: literature distances up to 1994.
    HiltonLahulla1995,
    /// HII regions: mostly kinematic distances.
    WiseHii2014,
    /// Galaxies: the Hubble flow from SIMBAD's redshifts.
    Redshift,
}

impl Tier {
    fn source_key(self) -> &'static str {
        match self {
            Self::HuntReffert2024 => "hunt-reffert-2024",
            Self::Cosmicflows4 => "cosmicflows-4",
            Self::ChornayWalton2021 => "chornay-walton-2021",
            Self::GonzalezSantamaria2021 => "gonzalez-santamaria-2021",
            Self::FosterBrunt2015 => "foster-brunt-2015",
            Self::RanasingheLeahy2023 => "ranasinghe-leahy-2023",
            Self::Zucker2020 => "zucker-2020",
            Self::Harris1997 => "harris-1997",
            Self::IlluminatingStar | Self::Simbad | Self::Redshift => "simbad",
            Self::StanghelliniHaywood2010 => "stanghellini-haywood-2010",
            Self::HiltonLahulla1995 => "hilton-lahulla-1995",
            Self::WiseHii2014 => "wise-hii-2014",
        }
    }

    /// The object kinds a source measures. A designation match to another
    /// kind is a different object sharing a number, or a mistyped one.
    fn accepts(self, kind: ObjectKind) -> bool {
        use ObjectKind::*;
        match self {
            Self::HuntReffert2024 => matches!(
                kind,
                OpenCluster
                    | GlobularCluster
                    | ClusterWithNebula
                    | Association
                    | HiiRegion
                    | Nebula
            ),
            Self::Cosmicflows4 | Self::Redshift => kind == Galaxy,
            Self::ChornayWalton2021
            | Self::GonzalezSantamaria2021
            | Self::StanghelliniHaywood2010 => matches!(kind, PlanetaryNebula | Nebula),
            Self::FosterBrunt2015 | Self::WiseHii2014 => {
                matches!(kind, HiiRegion | Nebula | ClusterWithNebula)
            }
            Self::RanasingheLeahy2023 => matches!(kind, SupernovaRemnant | Nebula),
            Self::Zucker2020 => matches!(
                kind,
                DarkNebula | Nebula | HiiRegion | ClusterWithNebula | SupernovaRemnant
            ),
            Self::IlluminatingStar => kind == Nebula,
            Self::Harris1997 => kind == GlobularCluster,
            Self::Simbad => measurable(kind),
            Self::HiltonLahulla1995 => kind == DarkNebula,
        }
    }
}

/// Kinds that get a distance at all. Stars are left to Gaia; OpenNGC's
/// `Other` holds asterisms and entries it could not classify, whose numbers
/// SIMBAD often gives to an unrelated galaxy.
fn measurable(kind: ObjectKind) -> bool {
    !matches!(
        kind,
        ObjectKind::Star | ObjectKind::DoubleStar | ObjectKind::Transient | ObjectKind::Other
    )
}

fn source_list() -> Vec<DistanceSource> {
    let source = |key: &str, citation: &str, licence: &str, url: &str| DistanceSource {
        key: key.into(),
        citation: citation.into(),
        licence: licence.into(),
        url: url.into(),
    };
    let vizier = |catalogue: &str| format!("https://cdsarc.cds.unistra.fr/viz-bin/cat/{catalogue}");
    vec![
        source(
            "simbad",
            "SIMBAD astronomical database, CDS, Strasbourg (Wenger et al. 2000, A&AS 143, 9): \
             distance measurements, redshifts and stellar parallaxes",
            "ODbL 1.0",
            "https://simbad.cds.unistra.fr/simbad/sim-tap",
        ),
        source(
            "hunt-reffert-2024",
            "Hunt & Reffert 2024, A&A 686, A42 (VizieR J/A+A/686/A42)",
            "CC BY 4.0",
            &vizier("J/A+A/686/A42"),
        ),
        source(
            "cosmicflows-4",
            "Tully et al. 2023, ApJ 944, 94, Cosmicflows-4 (VizieR J/ApJ/944/94)",
            "CC BY 4.0",
            &vizier("J/ApJ/944/94"),
        ),
        source(
            "chornay-walton-2021",
            "Chornay & Walton 2021, A&A 656, A110 (VizieR J/A+A/656/A110)",
            VIZIER_TERMS,
            &vizier("J/A+A/656/A110"),
        ),
        source(
            "gonzalez-santamaria-2021",
            "Gonzalez-Santamaria et al. 2021, A&A 656, A51 (VizieR J/A+A/656/A51)",
            VIZIER_TERMS,
            &vizier("J/A+A/656/A51"),
        ),
        source(
            "foster-brunt-2015",
            "Foster & Brunt 2015, AJ 150, 147 (VizieR J/AJ/150/147)",
            VIZIER_TERMS,
            &vizier("J/AJ/150/147"),
        ),
        source(
            "ranasinghe-leahy-2023",
            "Ranasinghe & Leahy 2023, ApJS 265, 53 (VizieR J/ApJS/265/53)",
            "CC BY 4.0",
            &vizier("J/ApJS/265/53"),
        ),
        source(
            "zucker-2020",
            "Zucker et al. 2020, A&A 633, A51 (VizieR J/A+A/633/A51)",
            VIZIER_TERMS,
            &vizier("J/A+A/633/A51"),
        ),
        source(
            "harris-1997",
            "Harris 1996 (rev. 1997), AJ 112, 1487 (VizieR VII/202)",
            VIZIER_TERMS,
            &vizier("VII/202"),
        ),
        source(
            "stanghellini-haywood-2010",
            "Stanghellini & Haywood 2010, ApJ 714, 1096 (VizieR J/ApJ/714/1096)",
            VIZIER_TERMS,
            &vizier("J/ApJ/714/1096"),
        ),
        source(
            "hilton-lahulla-1995",
            "Hilton & Lahulla 1995, A&AS 113, 325 (VizieR J/A+AS/113/325)",
            VIZIER_TERMS,
            &vizier("J/A+AS/113/325"),
        ),
        source(
            "wise-hii-2014",
            "Anderson et al. 2014, ApJS 212, 1, WISE catalog of Galactic HII regions \
             (VizieR J/ApJS/212/1)",
            VIZIER_TERMS,
            &vizier("J/ApJS/212/1"),
        ),
    ]
}

/// One source's distance for one object.
#[derive(Clone, Debug, PartialEq)]
struct Candidate {
    tier: Tier,
    distance_pc: f64,
    lower_pc: Option<f64>,
    upper_pc: Option<f64>,
    method: DistanceMethod,
    reference: Option<String>,
    /// How the source row found the object: 0 by its primary name, 1 by an
    /// alias, 2 by position.
    match_rank: u8,
    separation_deg: f64,
}

impl Candidate {
    fn new(tier: Tier, distance_pc: f64, method: DistanceMethod) -> Self {
        Self {
            tier,
            distance_pc,
            lower_pc: None,
            upper_pc: None,
            method,
            reference: None,
            match_rank: 2,
            separation_deg: 0.0,
        }
    }

    fn bounds(mut self, lower: Option<f64>, upper: Option<f64>) -> Self {
        let valid = |value: Option<f64>| value.filter(|value| value.is_finite() && *value > 0.0);
        self.lower_pc = valid(lower).filter(|lower| *lower <= self.distance_pc);
        self.upper_pc = valid(upper).filter(|upper| *upper >= self.distance_pc);
        self
    }

    fn reference(mut self, reference: &str) -> Self {
        let reference = reference.trim();
        self.reference = (!reference.is_empty()).then(|| reference.to_string());
        self
    }

    /// A value ± error, in parsecs.
    fn symmetric(self, error: Option<f64>) -> Self {
        let distance = self.distance_pc;
        self.bounds(
            error.map(|error| distance - error.abs()),
            error.map(|error| distance + error.abs()),
        )
    }

    /// Ordering within one tier: the best-matched row first.
    fn match_key(&self) -> (u8, u64) {
        (self.match_rank, (self.separation_deg * 1e9) as u64)
    }
}

/// Keep the better candidate for an object: earlier tier, then the closer
/// name match.
fn offer(best: &mut HashMap<usize, Candidate>, object: usize, candidate: Candidate) {
    match best.get(&object) {
        Some(current)
            if (current.tier, current.match_key()) <= (candidate.tier, candidate.match_key()) => {}
        _ => {
            best.insert(object, candidate);
        }
    }
}

/// Designation key with catalogue names spelled the way `objects.bin`
/// spells them: SIMBAD's `Cl Melotte 22` and `Barnard 33`, Hunt & Reffert's
/// `Melotte_22`, LEDA numbers as PGC.
fn catalog_key(value: &str) -> String {
    let compact: String = value
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .flat_map(|c| c.to_uppercase())
        .collect();
    for (long, short) in [
        ("CLMELOTTE", "MEL"),
        ("MELOTTE", "MEL"),
        ("CLCOLLINDER", "CR"),
        ("COLLINDER", "CR"),
        ("BARNARD", "B"),
        ("LEDA", "PGC"),
        ("MESSIER", "M"),
    ] {
        if let Some(rest) = compact.strip_prefix(long)
            && rest.starts_with(|c: char| c.is_ascii_digit())
        {
            return designation_key(&format!("{short}{rest}"));
        }
    }
    designation_key(&compact)
}

/// Objects by designation key, with whether the key is the primary name.
struct ObjectIndex<'a> {
    objects: &'a [SkyObject],
    keys: HashMap<String, Vec<(usize, bool)>>,
    by_kind: HashMap<u8, Vec<usize>>,
}

impl<'a> ObjectIndex<'a> {
    fn new(objects: &'a [SkyObject]) -> Self {
        let mut keys = HashMap::<String, Vec<(usize, bool)>>::new();
        let mut by_kind = HashMap::<u8, Vec<usize>>::new();
        for (index, object) in objects.iter().enumerate() {
            if !measurable(object.kind) {
                continue;
            }
            by_kind.entry(object.kind as u8).or_default().push(index);
            for (name, primary) in std::iter::once((&object.name, true))
                .chain(object.metadata.aliases.iter().map(|alias| (alias, false)))
            {
                let key = catalog_key(name);
                if key.is_empty() {
                    continue;
                }
                let entries = keys.entry(key).or_default();
                if !entries.iter().any(|(existing, _)| *existing == index) {
                    entries.push((index, primary));
                }
            }
        }
        Self {
            objects,
            keys,
            by_kind,
        }
    }

    /// Objects one of `names` designates, of a kind `tier` accepts, no
    /// farther from `position` than their extent plus [`NAME_MATCH_DEG`].
    /// Each comes with its match rank and separation.
    fn by_name<'n>(
        &self,
        names: impl IntoIterator<Item = &'n str>,
        position: Option<(f64, f64)>,
        tier: Tier,
    ) -> Vec<(usize, u8, f64)> {
        let mut found: Vec<(usize, u8, f64)> = Vec::new();
        for name in names {
            let Some(entries) = self.keys.get(&catalog_key(name)) else {
                continue;
            };
            for &(index, primary) in entries {
                let object = &self.objects[index];
                if !tier.accepts(object.kind) {
                    continue;
                }
                let separation = position.map_or(0.0, |(ra, dec)| {
                    separation_deg(ra, dec, object.ra, object.dec)
                });
                if separation > NAME_MATCH_DEG + radius_deg(object) {
                    continue;
                }
                let rank = if primary { 0 } else { 1 };
                match found.iter_mut().find(|(existing, ..)| *existing == index) {
                    Some(existing) if existing.1 > rank => existing.1 = rank,
                    Some(_) => {}
                    None => found.push((index, rank, separation)),
                }
            }
        }
        found
    }

    /// The nearest object of `kind` within `radius` degrees of a position,
    /// or whose extent covers it.
    fn nearest(&self, ra: f64, dec: f64, radius: f64, kind: ObjectKind) -> Option<(usize, f64)> {
        self.by_kind
            .get(&(kind as u8))?
            .iter()
            .filter_map(|&index| {
                let object = &self.objects[index];
                let separation = separation_deg(ra, dec, object.ra, object.dec);
                (separation <= radius.max(radius_deg(object))).then_some((index, separation))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
    }

    /// Objects of the kinds `tier` accepts.
    fn of_tier(&self, tier: Tier) -> impl Iterator<Item = usize> + '_ {
        let mut indices = self
            .by_kind
            .iter()
            .filter(|(_, indices)| tier.accepts(self.objects[indices[0]].kind))
            .flat_map(|(_, indices)| indices.iter().copied())
            .collect::<Vec<_>>();
        indices.sort_unstable();
        indices.into_iter()
    }
}

fn separation_deg(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
    let (ra1, dec1, ra2, dec2) = (
        ra1.to_radians(),
        dec1.to_radians(),
        ra2.to_radians(),
        dec2.to_radians(),
    );
    let haversine = ((dec2 - dec1) / 2.0).sin().powi(2)
        + dec1.cos() * dec2.cos() * ((ra2 - ra1) / 2.0).sin().powi(2);
    (2.0 * haversine.sqrt().min(1.0).asin()).to_degrees()
}

/// Half the catalogue major axis, degrees; zero when unknown.
fn radius_deg(object: &SkyObject) -> f64 {
    object
        .major_arcmin
        .map_or(0.0, |major| f64::from(major) / 120.0)
}

/// Area of an object's catalogue ellipse, square arcminutes.
fn area(object: &SkyObject) -> f64 {
    let major = f64::from(object.major_arcmin.unwrap_or(0.0));
    let minor = object.minor_arcmin.map_or(major, f64::from);
    major * minor
}

/// Whether `(ra, dec)` lies inside an object's catalogue ellipse, measured
/// on the gnomonic projection about its centre.
fn inside_ellipse(object: &SkyObject, ra: f64, dec: f64) -> bool {
    let Some(major) = object.major_arcmin.filter(|major| *major > 0.0) else {
        return false;
    };
    let minor = object
        .minor_arcmin
        .filter(|minor| *minor > 0.0)
        .unwrap_or(major);
    let (ra0, dec0) = (object.ra.to_radians(), object.dec.to_radians());
    let (ra, dec) = (ra.to_radians(), dec.to_radians());
    let cos_c = dec0.sin() * dec.sin() + dec0.cos() * dec.cos() * (ra - ra0).cos();
    if cos_c <= 0.0 {
        return false;
    }
    let xi = (dec.cos() * (ra - ra0).sin() / cos_c).to_degrees();
    let eta =
        ((dec0.cos() * dec.sin() - dec0.sin() * dec.cos() * (ra - ra0).cos()) / cos_c).to_degrees();
    // Position angle runs east of north; xi points east, eta north.
    let angle = f64::from(object.position_angle_deg.unwrap_or(0.0)).to_radians();
    let along = xi * angle.sin() + eta * angle.cos();
    let across = xi * angle.cos() - eta * angle.sin();
    let (a, b) = (f64::from(major) / 120.0, f64::from(minor) / 120.0);
    (along / a).powi(2) + (across / b).powi(2) <= 1.0
}

fn number(value: &str) -> Option<f64> {
    value
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
}

/// A VizieR `asu-tsv` table: comment lines start with `#`, then a header,
/// a units line, a dashed line and tab-separated rows.
struct VizierTable {
    columns: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl VizierTable {
    fn read(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("cannot read {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("{} is not a VizieR table", path.display()))
    }

    fn parse(text: &str) -> Result<Self> {
        let mut lines = text
            .lines()
            .filter(|line| !line.starts_with('#') && !line.trim().is_empty());
        let Some(header) = lines.next() else {
            bail!("no header line");
        };
        let columns = header
            .split('\t')
            .map(|column| column.trim().to_string())
            .collect::<Vec<_>>();
        let mut rows = Vec::new();
        let mut in_body = false;
        for line in lines {
            if !in_body {
                in_body = line.starts_with('-');
                continue;
            }
            rows.push(
                line.split('\t')
                    .map(|field| field.trim().to_string())
                    .collect(),
            );
        }
        if !in_body {
            bail!("no dashed line under the header");
        }
        Ok(Self { columns, rows })
    }

    fn column(&self, name: &str) -> Result<usize> {
        self.columns
            .iter()
            .position(|column| column == name)
            .with_context(|| format!("missing column {name}"))
    }

    /// Rows as the fields of `names`, in that order.
    fn select<const N: usize>(&self, names: [&str; N]) -> Result<Vec<[&str; N]>> {
        let indices = names.map(|name| self.column(name));
        let mut columns = [0; N];
        for (slot, index) in columns.iter_mut().zip(indices) {
            *slot = index?;
        }
        Ok(self
            .rows
            .iter()
            .map(|row| columns.map(|index| row.get(index).map_or("", String::as_str)))
            .collect())
    }
}

/// A CSV file from SIMBAD's TAP service, as rows of named fields.
fn read_tap_csv<const N: usize>(path: &Path, names: [&str; N]) -> Result<Vec<[String; N]>> {
    let mut reader =
        csv::Reader::from_path(path).with_context(|| format!("cannot read {}", path.display()))?;
    let headers = reader.headers()?.clone();
    let indices = names.map(|name| headers.iter().position(|header| header == name));
    let mut columns = [0; N];
    for ((slot, index), name) in columns.iter_mut().zip(indices).zip(names) {
        *slot = index.with_context(|| format!("{} has no column {name}", path.display()))?;
    }
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record.with_context(|| format!("{} is malformed", path.display()))?;
        rows.push(columns.map(|index| record.get(index).unwrap_or("").to_string()));
    }
    Ok(rows)
}

fn position(ra: &str, dec: &str) -> Option<(f64, f64)> {
    Some((number(ra)?, number(dec)?))
}

/// One SIMBAD distance measurement.
#[derive(Clone, Debug, PartialEq)]
struct SimbadMeasurement {
    distance_pc: f64,
    minus_pc: Option<f64>,
    plus_pc: Option<f64>,
    method: String,
    bibcode: String,
}

impl SimbadMeasurement {
    fn parse(
        dist: &str,
        unit: &str,
        minus: &str,
        plus: &str,
        method: &str,
        bibcode: &str,
    ) -> Option<Self> {
        let scale = match unit.trim() {
            "pc" => 1.0,
            "kpc" => 1e3,
            "Mpc" => 1e6,
            _ => return None,
        };
        let distance_pc = number(dist).filter(|value| *value > 0.0)? * scale;
        Some(Self {
            distance_pc,
            minus_pc: number(minus).map(|value| value.abs() * scale),
            plus_pc: number(plus).map(|value| value.abs() * scale),
            method: method.trim().to_string(),
            bibcode: bibcode.trim().to_string(),
        })
    }

    fn year(&self) -> Option<u32> {
        self.bibcode.get(..4)?.parse().ok()
    }

    fn is_redshift(&self) -> bool {
        simbad_method(&self.method) == DistanceMethod::Redshift
    }
}

/// SIMBAD's free-text method codes.
fn simbad_method(method: &str) -> DistanceMethod {
    match method.trim().to_ascii_lowercase().as_str() {
        "paral" | "plx" => DistanceMethod::Parallax,
        "st-l" | "sp-t" | "caiihk" | "sed-fit" => DistanceMethod::Photometric,
        "kin" => DistanceMethod::Kinematic,
        "t-f" | "sbf" | "t-rgb" | "trgb" | "t-rdb" | "cep" | "ceph" | "rrlyr" | "sn" | "pnlf"
        | "hb" | "bs" => DistanceMethod::StandardCandle,
        "redshift" | "flow" | "h0" => DistanceMethod::Redshift,
        "mem" | "group" | "grp" => DistanceMethod::Membership,
        _ => DistanceMethod::Unknown,
    }
}

/// One measurement from those SIMBAD lists for an object: Gaia-era ones
/// when there are any, for a galaxy those not from the redshift when there
/// are any, then the median of what is left (the lower one of an even
/// count), so a single outlier never decides. A galaxy ignores parallaxes
/// and anything nearer than [`MIN_GALAXY_PC`]: those belong to a star SIMBAD
/// identifies with it.
fn choose_simbad(measurements: &[SimbadMeasurement], galaxy: bool) -> Option<&SimbadMeasurement> {
    let mut pool: Vec<&SimbadMeasurement> = measurements
        .iter()
        .filter(|measurement| {
            !galaxy
                || (measurement.distance_pc >= MIN_GALAXY_PC
                    && simbad_method(&measurement.method) != DistanceMethod::Parallax)
        })
        .collect();
    if galaxy && pool.iter().any(|measurement| !measurement.is_redshift()) {
        pool.retain(|measurement| !measurement.is_redshift());
    }
    if pool
        .iter()
        .any(|measurement| measurement.year().is_some_and(|year| year >= GAIA_ERA))
    {
        pool.retain(|measurement| measurement.year().is_some_and(|year| year >= GAIA_ERA));
    }
    pool.sort_by(|a, b| {
        a.distance_pc
            .total_cmp(&b.distance_pc)
            .then_with(|| b.bibcode.cmp(&a.bibcode))
    });
    pool.get(pool.len().saturating_sub(1) / 2).copied()
}

/// Designations a Zucker et al. sightline name gives: catalogue numbers in
/// it (`L1228`, `LBN906`, `IC1396`, `S106`, `Ophiuchus_B44`), or the
/// object a popular cloud name means.
fn zucker_designations(name: &str) -> Vec<String> {
    let named = match name {
        "California" => Some("NGC 1499"),
        "Coalsack" => Some("C 99"),
        "North_America" => Some("NGC 7000"),
        "Lagoon" => Some("M 8"),
        "Rosette" => Some("Sh2-275"),
        "W3" => Some("IC 1795"),
        "W4" => Some("IC 1805"),
        "W5" => Some("IC 1848"),
        _ => None,
    };
    let mut designations = named.map(str::to_string).into_iter().collect::<Vec<_>>();
    for token in name.split('_') {
        if let Some(number) = token.strip_prefix("Sh2-") {
            designations.push(format!("Sh2-{number}"));
            continue;
        }
        let digits = token.trim_start_matches(|c: char| c.is_ascii_alphabetic());
        let prefix = &token[..token.len() - digits.len()];
        if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            continue;
        }
        let designation = match prefix {
            "L" => format!("LDN {digits}"),
            "S" => format!("Sh2-{digits}"),
            "B" | "LBN" | "IC" | "NGC" | "M" => format!("{prefix} {digits}"),
            _ => continue,
        };
        designations.push(designation);
    }
    designations
}

fn median(values: &mut [f64]) -> Option<f64> {
    values.sort_by(f64::total_cmp);
    values.get(values.len().saturating_sub(1) / 2).copied()
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    let position = fraction * (sorted.len() - 1) as f64;
    let (low, high) = (position.floor() as usize, position.ceil() as usize);
    sorted[low] + (sorted[high] - sorted[low]) * (position - low as f64)
}

/// Candidates from every source, keyed by object index.
struct Sources<'a> {
    index: &'a ObjectIndex<'a>,
    best: HashMap<usize, Candidate>,
    rows: BTreeMap<Tier, usize>,
}

impl Sources<'_> {
    fn add(&mut self, object: usize, candidate: Candidate) {
        *self.rows.entry(candidate.tier).or_default() += 1;
        offer(&mut self.best, object, candidate);
    }

    fn add_named<'n>(
        &mut self,
        names: impl IntoIterator<Item = &'n str>,
        position: Option<(f64, f64)>,
        candidate: Candidate,
    ) -> bool {
        let found = self.index.by_name(names, position, candidate.tier);
        for &(object, rank, separation) in &found {
            self.add(
                object,
                Candidate {
                    match_rank: rank,
                    separation_deg: separation,
                    ..candidate.clone()
                },
            );
        }
        !found.is_empty()
    }

    /// The nearest object of `kind` to `position` within `radius` degrees.
    fn add_nearest(
        &mut self,
        (ra, dec): (f64, f64),
        radius: f64,
        kind: ObjectKind,
        candidate: Candidate,
    ) {
        if let Some((object, separation)) = self.index.nearest(ra, dec, radius, kind) {
            self.add(
                object,
                Candidate {
                    separation_deg: separation,
                    ..candidate
                },
            );
        }
    }
}

fn hunt_reffert(sources: &mut Sources, input: &Path) -> Result<()> {
    let table = VizierTable::read(&input.join("hunt-reffert-2024.tsv"))?;
    for [name, all_names, kind, low, mid, high, ra, dec] in table.select([
        "Name", "AllNames", "Type", "dist16", "dist50", "dist84", "_RAJ2000", "_DEJ2000",
    ])? {
        let Some(distance) = number(mid).filter(|value| *value > 0.0) else {
            continue;
        };
        // r: rejected as a cluster.
        if kind == "r" {
            continue;
        }
        let (low, high) = (number(low), number(high));
        if let (Some(low), Some(high)) = (low, high)
            && (high - low) / (2.0 * distance) > MAX_CLUSTER_SPREAD
        {
            continue;
        }
        let candidate = Candidate::new(
            Tier::HuntReffert2024,
            distance,
            DistanceMethod::ClusterParallax,
        )
        .bounds(low, high)
        .reference("2024A&A...686A..42H");
        let name = name.replace('_', " ");
        let others = all_names
            .split(',')
            .map(|other| other.replace('_', " "))
            .filter(|other| *other != name)
            .collect::<Vec<_>>();
        // The main name first: another cluster may list it as a former
        // cross-match.
        if !sources.add_named([name.as_str()], position(ra, dec), candidate.clone()) {
            sources.add_named(
                others.iter().map(String::as_str),
                position(ra, dec),
                Candidate {
                    match_rank: 1,
                    ..candidate
                },
            );
        }
    }
    Ok(())
}

fn cosmicflows(sources: &mut Sources, input: &Path) -> Result<()> {
    let table = VizierTable::read(&input.join("cosmicflows-4.tsv"))?;
    for [pgc, modulus, error] in table.select(["PGC", "DM", "e_DM"])? {
        let (Some(modulus), Some(pgc)) = (number(modulus), number(pgc)) else {
            continue;
        };
        let parsecs = |modulus: f64| 10f64.powf(modulus / 5.0 + 1.0);
        let error = number(error).unwrap_or(0.0);
        let candidate = Candidate::new(
            Tier::Cosmicflows4,
            parsecs(modulus),
            DistanceMethod::StandardCandle,
        )
        .bounds(
            (error > 0.0).then(|| parsecs(modulus - error)),
            (error > 0.0).then(|| parsecs(modulus + error)),
        )
        .reference("2023ApJ...944...94T");
        sources.add_named([format!("PGC {pgc}").as_str()], None, candidate);
    }
    Ok(())
}

fn planetary_nebulae(sources: &mut Sources, input: &Path) -> Result<()> {
    let near = |radius_arcsec: f64| radius_arcsec / 3600.0;
    let table = VizierTable::read(&input.join("chornay-walton-2021.tsv"))?;
    for [png, name, mid, low, high, ra, dec] in table.select([
        "PNG", "Name", "rcomb", "b_rcomb", "B_rcomb", "pnRAdeg", "pnDEdeg",
    ])? {
        let Some(distance) = number(mid).filter(|value| *value > 0.0) else {
            continue;
        };
        let candidate = Candidate::new(Tier::ChornayWalton2021, distance, DistanceMethod::Parallax)
            .bounds(number(low), number(high))
            .reference("2021A&A...656A.110C");
        let png = format!("PN G{png}");
        if !sources.add_named([name, png.as_str()], position(ra, dec), candidate.clone())
            && let Some(position) = position(ra, dec)
        {
            sources.add_nearest(position, near(30.0), ObjectKind::PlanetaryNebula, candidate);
        }
    }

    let names = VizierTable::read(&input.join("gonzalez-santamaria-2021-names.tsv"))?;
    let names = names
        .select(["PNG", "OName", "_RAJ2000", "_DEJ2000"])?
        .into_iter()
        .map(|[png, name, ra, dec]| (png.to_string(), (name.to_string(), position(ra, dec))))
        .collect::<HashMap<_, _>>();
    let table = VizierTable::read(&input.join("gonzalez-santamaria-2021.tsv"))?;
    for [png, mid, low, high] in table.select(["PNG", "Dist", "b_Dist", "B_Dist"])? {
        let Some(distance) = number(mid).filter(|value| *value > 0.0) else {
            continue;
        };
        let candidate = Candidate::new(
            Tier::GonzalezSantamaria2021,
            distance,
            DistanceMethod::Parallax,
        )
        .bounds(number(low), number(high))
        .reference("2021A&A...656A..51G");
        let (name, star) = names
            .get(png)
            .map_or(("", None), |(name, star)| (name.as_str(), *star));
        if !sources.add_named([png, name], star, candidate.clone())
            && let Some(star) = star
        {
            sources.add_nearest(star, near(60.0), ObjectKind::PlanetaryNebula, candidate);
        }
    }

    let table = VizierTable::read(&input.join("stanghellini-haywood-2010.tsv"))?;
    for [png, mid, error, ra, dec] in table.select(["PNG", "d", "e_d", "_RA", "_DE"])? {
        let Some(distance) = number(mid).filter(|value| *value > 0.0) else {
            continue;
        };
        let candidate = Candidate::new(
            Tier::StanghelliniHaywood2010,
            distance * 1e3,
            DistanceMethod::Statistical,
        )
        .symmetric(number(error).map(|error| error * 1e3))
        .reference("2010ApJ...714.1096S");
        let png = format!("PN G{png}");
        if !sources.add_named([png.as_str()], position(ra, dec), candidate.clone())
            && let Some(position) = position(ra, dec)
        {
            sources.add_nearest(position, near(30.0), ObjectKind::PlanetaryNebula, candidate);
        }
    }
    Ok(())
}

fn hii_regions(sources: &mut Sources, input: &Path) -> Result<()> {
    let table = VizierTable::read(&input.join("foster-brunt-2015.tsv"))?;
    for [name, mid, error, simbad, ra, dec] in
        table.select(["HII", "r", "dr", "SimbadName", "_RAJ2000", "_DEJ2000"])?
    {
        let Some(distance) = number(mid).filter(|value| *value > 0.0) else {
            continue;
        };
        let candidate = Candidate::new(
            Tier::FosterBrunt2015,
            distance * 1e3,
            DistanceMethod::Photometric,
        )
        .symmetric(number(error).map(|error| error * 1e3))
        .reference("2015AJ....150..147F");
        sources.add_named([name, simbad], position(ra, dec), candidate);
    }

    // WISE regions match by position: the largest region whose circle
    // holds the object's centre, or whose centre the object's extent holds.
    let table = VizierTable::read(&input.join("wise-hii-2014.tsv"))?;
    let regions = table
        .select(["WISE", "Rad", "Dist", "_RAJ2000", "_DEJ2000"])?
        .into_iter()
        .filter_map(|[_, radius, distance, ra, dec]| {
            Some((
                position(ra, dec)?,
                number(radius).unwrap_or(30.0) / 3600.0,
                number(distance).filter(|value| *value > 0.0)? * 1e3,
            ))
        })
        .collect::<Vec<_>>();
    let index = sources.index;
    for position in index.of_tier(Tier::WiseHii2014) {
        let object = &index.objects[position];
        let best = regions
            .iter()
            .filter(|((ra, dec), radius, _)| {
                separation_deg(object.ra, object.dec, *ra, *dec) <= radius.max(radius_deg(object))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1));
        if let Some(&((ra, dec), _, distance)) = best {
            sources.add(
                position,
                Candidate {
                    separation_deg: separation_deg(object.ra, object.dec, ra, dec),
                    ..Candidate::new(Tier::WiseHii2014, distance, DistanceMethod::Kinematic)
                        .reference("2014ApJS..212....1A")
                },
            );
        }
    }
    Ok(())
}

fn supernova_remnants(sources: &mut Sources, input: &Path) -> Result<()> {
    let table = VizierTable::read(&input.join("ranasinghe-leahy-2023.tsv"))?;
    for [name, limit, distance, ra, dec] in
        table.select(["SNR", "l_X", "X", "_RAJ2000", "_DEJ2000"])?
    {
        // Limits are not distances.
        let Some(distance) = number(distance).filter(|value| *value > 0.0 && limit.is_empty())
        else {
            continue;
        };
        let candidate = Candidate::new(
            Tier::RanasingheLeahy2023,
            distance * 1e3,
            DistanceMethod::Compiled,
        )
        .reference("2023ApJS..265...53R");
        let name = format!("SNR {name}");
        if !sources.add_named([name.as_str()], position(ra, dec), candidate.clone())
            && let Some(position) = position(ra, dec)
        {
            sources.add_nearest(position, 0.1, ObjectKind::SupernovaRemnant, candidate);
        }
    }
    Ok(())
}

fn globular_clusters(sources: &mut Sources, input: &Path) -> Result<()> {
    let table = VizierTable::read(&input.join("harris-1997.tsv"))?;
    for [name, distance, ra, dec] in table.select(["ID", "Rsun", "_RAJ2000", "_DEJ2000"])? {
        let Some(distance) = number(distance).filter(|value| *value > 0.0) else {
            continue;
        };
        let candidate = Candidate::new(
            Tier::Harris1997,
            distance * 1e3,
            DistanceMethod::StandardCandle,
        )
        .reference("1996AJ....112.1487H");
        sources.add_named([name], position(ra, dec), candidate);
    }
    Ok(())
}

/// Zucker et al. sightlines, by cloud name. Returns every sightline for the
/// nearest-cloud fallback.
fn molecular_clouds(sources: &mut Sources, input: &Path) -> Result<Vec<(String, f64, f64, f64)>> {
    let table = VizierTable::read(&input.join("zucker-2020.tsv"))?;
    let sightlines = table
        .select(["Name", "d50", "_RAJ2000", "_DEJ2000"])?
        .into_iter()
        .filter_map(|[name, distance, ra, dec]| {
            let (ra, dec) = position(ra, dec)?;
            Some((
                name.to_string(),
                ra,
                dec,
                number(distance).filter(|value| *value > 0.0)?,
            ))
        })
        .collect::<Vec<_>>();
    let mut clouds = BTreeMap::<&str, Vec<&(String, f64, f64, f64)>>::new();
    for sightline in &sightlines {
        clouds.entry(&sightline.0).or_default().push(sightline);
    }
    for (name, lines) in clouds {
        let mut distances = lines.iter().map(|line| line.3).collect::<Vec<_>>();
        let Some(distance) = median(&mut distances) else {
            continue;
        };
        let (low, high) = (distances[0], distances[distances.len() - 1]);
        let candidate = Candidate::new(Tier::Zucker2020, distance, DistanceMethod::Extinction)
            .bounds(
                (lines.len() > 1).then_some(low),
                (lines.len() > 1).then_some(high),
            )
            .reference("2020A&A...633A..51Z");
        let designations = zucker_designations(name);
        let index = sources.index;
        // Sightlines sample a cloud's edges, so allow a few degrees.
        for (object, rank, _) in index.by_name(
            designations.iter().map(String::as_str),
            None,
            Tier::Zucker2020,
        ) {
            let target = &index.objects[object];
            if lines.iter().all(|line| {
                separation_deg(line.1, line.2, target.ra, target.dec) > 3.0 + radius_deg(target)
            }) {
                continue;
            }
            sources.add(
                object,
                Candidate {
                    match_rank: rank,
                    ..candidate.clone()
                },
            );
        }
    }

    let table = VizierTable::read(&input.join("hilton-lahulla-1995.tsv"))?;
    let mut lynds = BTreeMap::<String, Vec<f64>>::new();
    for [number_field, limit, distance, _] in table.select(["LDN", "n_Dist2", "Dist", "Refs"])? {
        if limit == "<" {
            continue;
        }
        if let Some(distance) = number(distance).filter(|value| *value > 0.0) {
            lynds
                .entry(number_field.to_string())
                .or_default()
                .push(distance);
        }
    }
    for (lynds_number, mut distances) in lynds {
        let Some(distance) = median(&mut distances) else {
            continue;
        };
        let (low, high) = (distances[0], distances[distances.len() - 1]);
        let several = distances.len() > 1;
        let candidate = Candidate::new(Tier::HiltonLahulla1995, distance, DistanceMethod::Compiled)
            .bounds(several.then_some(low), several.then_some(high))
            .reference("1995A&AS..113..325H");
        sources.add_named([format!("LDN {lynds_number}").as_str()], None, candidate);
    }
    Ok(sightlines)
}

fn reflection_nebulae(sources: &mut Sources, input: &Path) -> Result<()> {
    let parallaxes = read_tap_csv(
        &input.join("simbad-vdb-stars.csv"),
        ["id", "plx_value", "plx_err", "plx_bibcode"],
    )?
    .into_iter()
    .filter_map(|[id, parallax, error, bibcode]| {
        let (parallax, error) = (number(&parallax)?, number(&error)?);
        // A parallax under five times its error says little.
        (parallax > 0.0 && error > 0.0 && parallax >= 5.0 * error).then(|| {
            (
                id.split_whitespace().collect::<Vec<_>>().join(" "),
                (parallax, error, bibcode),
            )
        })
    })
    .collect::<HashMap<_, _>>();
    let table = VizierTable::read(&input.join("vdb-stars.tsv"))?;
    for [vdb, durchmusterung, hd] in table.select(["VdB", "DM", "HD"])? {
        let star = [
            (!hd.is_empty()).then(|| format!("HD {hd}")),
            (!durchmusterung.is_empty()).then(|| {
                durchmusterung
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
            }),
        ]
        .into_iter()
        .flatten()
        .find_map(|id| parallaxes.get(&id));
        let Some((parallax, error, bibcode)) = star else {
            continue;
        };
        let candidate = Candidate::new(
            Tier::IlluminatingStar,
            1000.0 / parallax,
            DistanceMethod::Parallax,
        )
        .bounds(
            Some(1000.0 / (parallax + error)),
            Some(1000.0 / (parallax - error)),
        )
        .reference(bibcode);
        sources.add_named([format!("vdB {vdb}").as_str()], None, candidate);
    }
    Ok(())
}

fn simbad(sources: &mut Sources, input: &Path) -> Result<()> {
    let path = input.join("simbad-distances.csv");
    let rows = read_tap_csv(
        &path,
        [
            "id",
            "oidref",
            "ra",
            "dec",
            "dist",
            "unit",
            "minus_err",
            "plus_err",
            "method",
            "bibcode",
        ],
    )?;
    struct Object {
        identifiers: Vec<String>,
        position: Option<(f64, f64)>,
        measurements: Vec<SimbadMeasurement>,
    }
    let mut objects = BTreeMap::<String, Object>::new();
    for [id, oid, ra, dec, dist, unit, minus, plus, method, bibcode] in rows {
        let object = objects.entry(oid).or_insert_with(|| Object {
            identifiers: Vec::new(),
            position: position(&ra, &dec),
            measurements: Vec::new(),
        });
        if !object.identifiers.contains(&id) {
            object.identifiers.push(id);
        }
        if let Some(measurement) =
            SimbadMeasurement::parse(&dist, &unit, &minus, &plus, &method, &bibcode)
            && !object.measurements.contains(&measurement)
        {
            object.measurements.push(measurement);
        }
    }
    let index = sources.index;
    for object in objects.values() {
        let found = index.by_name(
            object.identifiers.iter().map(String::as_str),
            object.position,
            Tier::Simbad,
        );
        for (position, rank, separation) in found {
            let galaxy = index.objects[position].kind == ObjectKind::Galaxy;
            let Some(chosen) = choose_simbad(&object.measurements, galaxy) else {
                continue;
            };
            let candidate = Candidate {
                match_rank: rank,
                separation_deg: separation,
                ..Candidate::new(
                    Tier::Simbad,
                    chosen.distance_pc,
                    simbad_method(&chosen.method),
                )
                .bounds(
                    chosen.minus_pc.map(|minus| chosen.distance_pc - minus),
                    chosen.plus_pc.map(|plus| chosen.distance_pc + plus),
                )
                .reference(&chosen.bibcode)
            };
            sources.add(position, candidate);
        }
    }
    Ok(())
}

fn redshifts(sources: &mut Sources, input: &Path) -> Result<()> {
    let rows = read_tap_csv(
        &input.join("simbad-redshifts.csv"),
        ["id", "rvz_redshift", "rvz_qual", "rvz_bibcode"],
    )?;
    for [id, redshift, quality, bibcode] in rows {
        // Quality E is SIMBAD's least trusted.
        let Some(redshift) =
            number(&redshift).filter(|redshift| *redshift >= MIN_REDSHIFT && quality != "E")
        else {
            continue;
        };
        let velocity = SPEED_OF_LIGHT_KM_S * redshift;
        let megaparsecs = |velocity: f64| velocity / HUBBLE_CONSTANT * 1e6;
        let candidate = Candidate::new(
            Tier::Redshift,
            megaparsecs(velocity),
            DistanceMethod::Redshift,
        )
        .bounds(
            Some(megaparsecs(velocity - PECULIAR_VELOCITY)),
            Some(megaparsecs(velocity + PECULIAR_VELOCITY)),
        )
        .reference(&bibcode);
        sources.add_named([id.as_str()], None, candidate);
    }
    Ok(())
}

/// What an object without its own distance borrows.
#[derive(Clone, Debug, PartialEq)]
struct Borrowed {
    from: usize,
    basis: DistanceBasis,
}

/// Kinds that may borrow from a Galactic object that holds them or that
/// they hold. Clusters are left out: an unrelated cluster behind a nebula
/// would take the nebula's depth.
fn borrows_from_neighbour(kind: ObjectKind) -> bool {
    matches!(
        kind,
        ObjectKind::Nebula
            | ObjectKind::DarkNebula
            | ObjectKind::HiiRegion
            | ObjectKind::ClusterWithNebula
            | ObjectKind::SupernovaRemnant
    )
}

fn lends_to_neighbour(kind: ObjectKind) -> bool {
    matches!(
        kind,
        ObjectKind::OpenCluster
            | ObjectKind::ClusterWithNebula
            | ObjectKind::HiiRegion
            | ObjectKind::Nebula
            | ObjectKind::Association
            | ObjectKind::DarkNebula
            | ObjectKind::SupernovaRemnant
    )
}

/// The measured objects others may borrow a distance from.
struct Lenders {
    /// Galaxies between [`MIN_HOST_DISTANCE_PC`] and
    /// [`MAX_HOST_DISTANCE_PC`] with a distance not from their redshift.
    hosts: Vec<usize>,
    /// Galactic objects of kinds that lend to neighbours.
    neighbours: Vec<usize>,
}

impl Lenders {
    fn new(objects: &[SkyObject], measured: &HashMap<usize, Candidate>) -> Self {
        let mut hosts = Vec::new();
        let mut neighbours = Vec::new();
        for (&index, candidate) in measured {
            let object = &objects[index];
            if object.kind == ObjectKind::Galaxy {
                if candidate.method != DistanceMethod::Redshift
                    && (MIN_HOST_DISTANCE_PC..=MAX_HOST_DISTANCE_PC)
                        .contains(&candidate.distance_pc)
                {
                    hosts.push(index);
                }
            } else if lends_to_neighbour(object.kind) {
                neighbours.push(index);
            }
        }
        hosts.sort_unstable();
        neighbours.sort_unstable();
        Self { hosts, neighbours }
    }

    /// The object `object` borrows a distance from, if any: the smallest
    /// galaxy whose ellipse holds it, else the smallest Galactic object
    /// whose extent holds it, else the nearest one inside its own extent. A
    /// supernova remnant only pairs with another remnant; a dark nebula only
    /// with a neighbour nearer than [`MAX_DARK_CLOUD_PC`].
    fn borrow_for(
        &self,
        object: &SkyObject,
        objects: &[SkyObject],
        measured: &HashMap<usize, Candidate>,
    ) -> Option<Borrowed> {
        if object.kind == ObjectKind::Galaxy || !measurable(object.kind) {
            return None;
        }
        let host = self
            .hosts
            .iter()
            .filter(|&&index| inside_ellipse(&objects[index], object.ra, object.dec))
            .min_by(|&&a, &&b| area(&objects[a]).total_cmp(&area(&objects[b])));
        if let Some(&from) = host {
            return Some(Borrowed {
                from,
                basis: DistanceBasis::HostGalaxy,
            });
        }
        if !borrows_from_neighbour(object.kind) {
            return None;
        }
        let mut container: Option<(usize, f64)> = None;
        let mut inside: Option<(usize, f64)> = None;
        for &index in &self.neighbours {
            let lender = &objects[index];
            if (object.kind == ObjectKind::SupernovaRemnant)
                != (lender.kind == ObjectKind::SupernovaRemnant)
                || (object.kind == ObjectKind::DarkNebula
                    && measured[&index].distance_pc > MAX_DARK_CLOUD_PC)
                || lender.metadata.id == object.metadata.id
            {
                continue;
            }
            let separation = separation_deg(object.ra, object.dec, lender.ra, lender.dec);
            if separation <= radius_deg(lender) {
                let size = area(lender);
                if container.is_none_or(|(_, best_size)| size < best_size) {
                    container = Some((index, size));
                }
            } else if separation <= radius_deg(object)
                && inside.is_none_or(|(_, best_separation)| separation < best_separation)
            {
                inside = Some((index, separation));
            }
        }
        container.or(inside).map(|(from, _)| Borrowed {
            from,
            basis: DistanceBasis::Contained,
        })
    }
}

/// The nearest cloud sightline within [`NEARBY_CLOUD_DEG`] of a dark nebula.
fn nearby_cloud<'s>(
    object: &SkyObject,
    sightlines: &'s [(String, f64, f64, f64)],
) -> Option<&'s (String, f64, f64, f64)> {
    if object.kind != ObjectKind::DarkNebula {
        return None;
    }
    sightlines
        .iter()
        .map(|line| (separation_deg(object.ra, object.dec, line.1, line.2), line))
        .filter(|(separation, _)| *separation <= NEARBY_CLOUD_DEG)
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, line)| line)
}

/// Build `object-distances.bin` from the files in `input` for the objects
/// in the catalog at `objects`.
pub fn build_object_distances(input: &Path, objects: &Path, output: &Path) -> Result<()> {
    let catalog = ObjectCatalog::open(objects)
        .with_context(|| format!("failed to open {}", objects.display()))?;
    let objects = catalog.read_all()?;
    let index = ObjectIndex::new(&objects);
    let mut sources = Sources {
        index: &index,
        best: HashMap::new(),
        rows: BTreeMap::new(),
    };
    hunt_reffert(&mut sources, input)?;
    cosmicflows(&mut sources, input)?;
    planetary_nebulae(&mut sources, input)?;
    hii_regions(&mut sources, input)?;
    supernova_remnants(&mut sources, input)?;
    globular_clusters(&mut sources, input)?;
    let sightlines = molecular_clouds(&mut sources, input)?;
    reflection_nebulae(&mut sources, input)?;
    simbad(&mut sources, input)?;
    redshifts(&mut sources, input)?;
    let measured = sources.best;

    let mut builder = ObjectDistancesBuilder::new(NOTICE);
    let mut source_index = HashMap::new();
    for source in source_list() {
        let key = source.key.clone();
        source_index.insert(key, builder.add_source(source)?);
    }
    let source_of = |tier: Tier| source_index[tier.source_key()];

    let lenders = Lenders::new(&objects, &measured);
    let mut by_kind = BTreeMap::<&str, [usize; 5]>::new();
    for (position, object) in objects.iter().enumerate() {
        if !measurable(object.kind) {
            continue;
        }
        let counts = by_kind.entry(object.kind.as_str()).or_default();
        let (candidate, basis, via) = if let Some(candidate) = measured.get(&position) {
            (candidate.clone(), DistanceBasis::Measured, None)
        } else if let Some(borrowed) = lenders.borrow_for(object, &objects, &measured) {
            (
                measured[&borrowed.from].clone(),
                borrowed.basis,
                Some(objects[borrowed.from].metadata.id.clone()),
            )
        } else if let Some((name, _, _, distance)) = nearby_cloud(object, &sightlines) {
            (
                Candidate::new(Tier::Zucker2020, *distance, DistanceMethod::Extinction)
                    .reference("2020A&A...633A..51Z"),
                DistanceBasis::NearbyCloud,
                Some(format!("zucker-2020:{name}")),
            )
        } else {
            counts[4] += 1;
            continue;
        };
        counts[basis as usize] += 1;
        builder.add(DistanceEntry {
            object_id: object.metadata.id.clone(),
            distance_pc: candidate.distance_pc,
            lower_pc: candidate.lower_pc,
            upper_pc: candidate.upper_pc,
            source: source_of(candidate.tier),
            method: candidate.method,
            basis,
            reference: candidate.reference,
            via,
        });
    }

    let mut measured_by_kind = BTreeMap::<u8, (ObjectKind, Vec<f64>)>::new();
    for (&position, candidate) in &measured {
        let kind = objects[position].kind;
        measured_by_kind
            .entry(kind as u8)
            .or_insert_with(|| (kind, Vec::new()))
            .1
            .push(candidate.distance_pc);
    }
    for (kind, mut distances) in measured_by_kind.into_values() {
        if distances.len() < 5 {
            continue;
        }
        distances.sort_by(f64::total_cmp);
        builder.set_kind_default(KindDistance {
            kind,
            objects: distances.len() as u32,
            distance_pc: percentile(&distances, 0.5),
            lower_pc: percentile(&distances, 0.16),
            upper_pc: percentile(&distances, 0.84),
        });
    }

    let entries = builder.len();
    builder.write_to(output)?;
    println!("{entries} object distances written to {}", output.display());
    println!(
        "{:<18} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "kind", "measured", "contained", "host", "cloud", "none"
    );
    for (kind, counts) in &by_kind {
        println!(
            "{kind:<18} {:>9} {:>9} {:>9} {:>9} {:>9}",
            counts[0], counts[1], counts[2], counts[3], counts[4]
        );
    }
    let mut chosen = BTreeMap::<Tier, usize>::new();
    for candidate in measured.values() {
        *chosen.entry(candidate.tier).or_default() += 1;
    }
    println!("{:<24} {:>9} {:>9}", "source", "matches", "chosen");
    for (tier, rows) in &sources.rows {
        println!(
            "{:<24} {:>9} {:>9}",
            format!("{tier:?}"),
            rows,
            chosen.get(tier).copied().unwrap_or(0)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use seiza::objects::ObjectMetadata;

    fn object(name: &str, kind: ObjectKind, ra: f64, dec: f64, major: Option<f32>) -> SkyObject {
        SkyObject {
            kind,
            ra,
            dec,
            mag: None,
            major_arcmin: major,
            minor_arcmin: None,
            position_angle_deg: None,
            name: name.into(),
            common_name: String::new(),
            metadata: ObjectMetadata {
                id: format!("test:{}", name.replace(' ', "")),
                ..ObjectMetadata::default()
            },
        }
    }

    fn measurement(distance_pc: f64, method: &str, bibcode: &str) -> SimbadMeasurement {
        SimbadMeasurement {
            distance_pc,
            minus_pc: None,
            plus_pc: None,
            method: method.into(),
            bibcode: bibcode.into(),
        }
    }

    #[test]
    fn catalog_keys_spell_names_like_the_object_catalog() {
        assert_eq!(catalog_key("Cl Melotte   22"), catalog_key("Mel 22"));
        assert_eq!(catalog_key("Melotte_22"), catalog_key("Mel 22"));
        assert_eq!(catalog_key("Barnard  33"), catalog_key("B 33"));
        assert_eq!(catalog_key("LEDA    2557"), catalog_key("PGC 2557"));
        assert_eq!(catalog_key("SH  2-220"), catalog_key("Sh2-220"));
        assert_eq!(catalog_key("M  42"), catalog_key("M 42"));
        assert_eq!(catalog_key("NGC_2632"), catalog_key("NGC 2632"));
        assert_ne!(catalog_key("Barnard 33"), catalog_key("Barnard 333"));
        assert_ne!(catalog_key("LBN 974"), catalog_key("LDN 974"));
    }

    #[test]
    fn earlier_tiers_win_then_closer_name_matches() {
        let mut best = HashMap::new();
        let simbad = Candidate::new(Tier::Simbad, 488.0, DistanceMethod::Unknown);
        offer(&mut best, 7, simbad.clone());
        let zucker = Candidate::new(Tier::Zucker2020, 433.0, DistanceMethod::Extinction);
        offer(&mut best, 7, zucker.clone());
        assert_eq!(best[&7], zucker);
        // A later tier never replaces an earlier one.
        offer(
            &mut best,
            7,
            Candidate::new(Tier::WiseHii2014, 900.0, DistanceMethod::Kinematic),
        );
        assert_eq!(best[&7], zucker);
        // Within a tier, a primary-name match beats an alias match.
        let alias = Candidate {
            match_rank: 1,
            ..Candidate::new(
                Tier::HuntReffert2024,
                2144.0,
                DistanceMethod::ClusterParallax,
            )
        };
        offer(&mut best, 8, alias);
        let primary = Candidate {
            match_rank: 0,
            ..Candidate::new(
                Tier::HuntReffert2024,
                3983.0,
                DistanceMethod::ClusterParallax,
            )
        };
        offer(&mut best, 8, primary.clone());
        assert_eq!(best[&8], primary);
        assert!(Tier::HuntReffert2024 < Tier::Simbad && Tier::Simbad < Tier::Redshift);
    }

    #[test]
    fn simbad_choice_prefers_gaia_era_median_and_skips_redshifts_for_galaxies() {
        let rows = [
            measurement(500.0, "kin", "2006ApJ...653.1226Q"),
            measurement(458.0, "", "2018MNRAS.473..849D"),
            measurement(433.0, "plx", "2020A&A...633A..51Z"),
            measurement(488.0, "", "2017AstBu..72..257L"),
        ];
        // Two Gaia-era values; the lower median is the smaller.
        assert_eq!(choose_simbad(&rows, false).unwrap().distance_pc, 433.0);
        let old = [
            measurement(2000.0, "", "1973PASP...85..579T"),
            measurement(1800.0, "", "1990ApJ...100..100X"),
            measurement(2200.0, "", "1991ApJ...100..100X"),
        ];
        assert_eq!(choose_simbad(&old, false).unwrap().distance_pc, 2000.0);
        let galaxy = [
            measurement(7.7e5, "redshift", "2018MNRAS.479.4136K"),
            measurement(7.61e5, "", "2021ApJ...920...84L"),
            measurement(7.9e5, "T-F", "2011ApJS..192....6L"),
        ];
        assert_eq!(choose_simbad(&galaxy, true).unwrap().distance_pc, 7.61e5);
        assert_eq!(choose_simbad(&galaxy, false).unwrap().distance_pc, 7.61e5);
        // A star's parallax under a galaxy's name is not the galaxy's.
        let star = [
            measurement(287.0, "paral", "2020yCat.1350....0G"),
            measurement(5.0e3, "", "2021yCat.0000....0X"),
        ];
        assert!(choose_simbad(&star, true).is_none());
        assert_eq!(choose_simbad(&star, false).unwrap().distance_pc, 287.0);
        let only_redshift = [measurement(7.2e6, "Redshift", "2019MNRAS.000..000X")];
        assert_eq!(
            choose_simbad(&only_redshift, true).unwrap().distance_pc,
            7.2e6
        );
        assert!(choose_simbad(&[], false).is_none());
        assert_eq!(simbad_method("paral   "), DistanceMethod::Parallax);
        assert_eq!(simbad_method("T-RGB"), DistanceMethod::StandardCandle);
        assert_eq!(simbad_method("?"), DistanceMethod::Unknown);
    }

    #[test]
    fn simbad_measurements_convert_units_and_errors() {
        let measurement =
            SimbadMeasurement::parse("0.135", "kpc ", "-0.004", "0.004", "paral", "2021X").unwrap();
        assert!((measurement.distance_pc - 135.0).abs() < 1e-9);
        assert_eq!(measurement.minus_pc, Some(4.0));
        assert!(SimbadMeasurement::parse("1", "ly", "", "", "", "").is_none());
        assert!(SimbadMeasurement::parse("0", "pc", "", "", "", "").is_none());
    }

    #[test]
    fn zucker_names_give_designations() {
        assert_eq!(zucker_designations("L1228D"), Vec::<String>::new());
        assert_eq!(zucker_designations("L1228"), ["LDN 1228"]);
        assert_eq!(zucker_designations("Ophiuchus_B44"), ["B 44"]);
        assert_eq!(zucker_designations("Ophiuchus_L1688"), ["LDN 1688"]);
        assert_eq!(zucker_designations("Mon_OB1_NGC2264"), ["NGC 2264"]);
        assert_eq!(zucker_designations("S106"), ["Sh2-106"]);
        assert_eq!(zucker_designations("Sh2-231"), ["Sh2-231"]);
        assert_eq!(zucker_designations("IC1396"), ["IC 1396"]);
        assert_eq!(zucker_designations("California"), ["NGC 1499"]);
        assert!(zucker_designations("Taurus").is_empty());
        assert!(zucker_designations("CB28").is_empty());
    }

    #[test]
    fn vizier_tables_parse_header_units_and_rows() {
        let text = "#RESOURCE=yCat\n#Column\tName\n\nName\tdist50\n \tpc\n----\t------\nNGC_2244\t1415.2\nM45\t\n";
        let table = VizierTable::parse(text).unwrap();
        let rows = table.select(["dist50", "Name"]).unwrap();
        assert_eq!(rows, [["1415.2", "NGC_2244"], ["", "M45"]]);
        assert!(table.select(["missing"]).is_err());
        assert!(VizierTable::parse("#only comments\n").is_err());
    }

    #[test]
    fn ellipses_hold_points_along_their_position_angle() {
        // 60' × 20' ellipse with its major axis pointing east.
        let mut galaxy = object("Host", ObjectKind::Galaxy, 10.0, 41.0, Some(60.0));
        galaxy.minor_arcmin = Some(20.0);
        galaxy.position_angle_deg = Some(90.0);
        let east = 0.4 / 41f64.to_radians().cos();
        assert!(inside_ellipse(&galaxy, 10.0 + east, 41.0));
        assert!(!inside_ellipse(&galaxy, 10.0, 41.4));
        assert!(inside_ellipse(&galaxy, 10.0, 41.1));
        galaxy.major_arcmin = None;
        assert!(!inside_ellipse(&galaxy, 10.0, 41.0));
    }

    #[test]
    fn neighbours_lend_distances_by_containment_and_host_galaxy() {
        let mut lmc = object("LMC", ObjectKind::Galaxy, 80.9, -69.75, Some(645.0));
        lmc.minor_arcmin = Some(550.0);
        let objects = vec![
            lmc,
            object("NGC 2070", ObjectKind::HiiRegion, 84.66, -69.1, Some(40.0)),
            object(
                "Cygnus Loop",
                ObjectKind::SupernovaRemnant,
                312.75,
                30.67,
                Some(230.0),
            ),
            object(
                "NGC 6960",
                ObjectKind::SupernovaRemnant,
                311.43,
                30.72,
                Some(70.0),
            ),
            object("IC 434", ObjectKind::Nebula, 85.25, -2.4, Some(60.0)),
            object("B 33", ObjectKind::DarkNebula, 85.25, -2.46, Some(6.0)),
            object("Far HII", ObjectKind::HiiRegion, 300.0, 35.0, Some(120.0)),
            object("LDN 1", ObjectKind::DarkNebula, 300.1, 35.1, Some(10.0)),
            object("NGC 7000", ObjectKind::Nebula, 314.7, 44.3, Some(120.0)),
            object("Lone", ObjectKind::OpenCluster, 314.8, 44.4, None),
            object("Vega", ObjectKind::Star, 279.2, 38.8, None),
        ];
        let mut measured = HashMap::new();
        measured.insert(
            0,
            Candidate::new(Tier::Cosmicflows4, 49_600.0, DistanceMethod::StandardCandle),
        );
        measured.insert(
            2,
            Candidate::new(Tier::RanasingheLeahy2023, 735.0, DistanceMethod::Compiled),
        );
        measured.insert(
            4,
            Candidate::new(Tier::Simbad, 400.0, DistanceMethod::Parallax),
        );
        measured.insert(
            6,
            Candidate::new(Tier::WiseHii2014, 5000.0, DistanceMethod::Kinematic),
        );
        measured.insert(
            8,
            Candidate::new(Tier::Simbad, 800.0, DistanceMethod::Parallax),
        );
        let lenders = Lenders::new(&objects, &measured);
        let borrow = |index: usize| lenders.borrow_for(&objects[index], &objects, &measured);

        assert_eq!(
            borrow(1),
            Some(Borrowed {
                from: 0,
                basis: DistanceBasis::HostGalaxy
            })
        );
        assert_eq!(
            borrow(3),
            Some(Borrowed {
                from: 2,
                basis: DistanceBasis::Contained
            })
        );
        assert_eq!(
            borrow(5),
            Some(Borrowed {
                from: 4,
                basis: DistanceBasis::Contained
            })
        );
        // A dark cloud does not take a distant region's depth.
        assert_eq!(borrow(7), None);
        // Clusters and stars never borrow from a neighbour.
        assert_eq!(borrow(9), None);
        assert_eq!(borrow(10), None);
        // Nor does a galaxy.
        assert_eq!(borrow(0), None);
    }

    #[test]
    fn dark_nebulae_take_the_nearest_cloud_sightline_within_a_degree() {
        let sightlines = vec![
            ("Taurus".to_string(), 68.0, 26.0, 140.0),
            ("Perseus".to_string(), 55.0, 31.5, 290.0),
        ];
        let near = object("LDN 1495", ObjectKind::DarkNebula, 68.5, 26.5, Some(30.0));
        assert_eq!(nearby_cloud(&near, &sightlines).unwrap().0, "Taurus");
        let far = object("LDN 9", ObjectKind::DarkNebula, 120.0, 0.0, Some(30.0));
        assert!(nearby_cloud(&far, &sightlines).is_none());
        let nebula = object("NGC 1555", ObjectKind::Nebula, 68.0, 26.0, None);
        assert!(nearby_cloud(&nebula, &sightlines).is_none());
    }

    #[test]
    fn name_matches_respect_kind_and_position() {
        let mut objects = vec![
            object("NGC 2244", ObjectKind::OpenCluster, 97.98, 4.94, Some(24.0)),
            object("PGC 2557", ObjectKind::Galaxy, 10.68, 41.27, Some(190.0)),
        ];
        objects[0].metadata.aliases = vec!["Mel 50".into()];
        let index = ObjectIndex::new(&objects);
        let found = index.by_name(["NGC_2244"], Some((97.98, 4.9)), Tier::HuntReffert2024);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].1, 0);
        assert_eq!(index.by_name(["Cl Melotte 50"], None, Tier::Simbad)[0].1, 1);
        // Too far from the source's position.
        assert!(
            index
                .by_name(["NGC 2244"], Some((120.0, 4.9)), Tier::HuntReffert2024)
                .is_empty()
        );
        // Cosmicflows-4 only measures galaxies.
        assert!(
            index
                .by_name(["NGC 2244"], None, Tier::Cosmicflows4)
                .is_empty()
        );
        assert_eq!(
            index.by_name(["LEDA 2557"], None, Tier::Cosmicflows4).len(),
            1
        );
    }
}
