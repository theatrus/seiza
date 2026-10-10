//! Distances to the objects in the object catalog, for placing a nebula,
//! cluster or galaxy at its depth among catalogue stars.
//!
//! Format `SEIZADS1` (little-endian). `str` below is a u16 byte length
//! followed by that many bytes of UTF-8.
//!
//! ```text
//! magic          [u8; 8]  = b"SEIZADS1"
//! entry_count    u32
//! source_count   u8
//! reserved       [u8; 3]      zero
//! strings_len    u32
//! catalog        [u8; 32]     fingerprint of the object catalog the IDs
//!                             belong to (ObjectCatalog::fingerprint)
//! notice         str          licence and attribution for the whole file
//! sources        source_count × { key: str, citation: str, licence: str, url: str }
//! entries        entry_count × 28 bytes, sorted by object ID:
//!                { id: u32, distance_pc: f32, lower_pc: f32, upper_pc: f32,
//!                  reference: u32, via: u32, source: u8, method: u8,
//!                  basis: u8, reserved: u8 }
//! strings        strings_len bytes of str records
//! ```
//!
//! Entries are keyed by the object catalog's stable ID
//! ([`crate::objects::ObjectMetadata::id`], e.g. `openngc:NGC1976`). The
//! `id`, `reference` and `via` fields are byte offsets into `strings`,
//! `u32::MAX` for none. `source` indexes `sources`. Bounds are NaN when the
//! source gives none; otherwise they are the source's own interval, a
//! 16th–84th percentile range or a value ± its error. `reference` names the
//! measurement's publication, usually a bibcode; `via` is what an entry
//! borrows its distance from (see [`DistanceBasis`]).
//!
//! Object IDs do not survive a rebuild of the object catalog, so a file
//! only answers for the catalog it was built from: [`ObjectDistances::open`]
//! compares fingerprints and refuses any other with [`CatalogMismatch`].
//!
//! [`ObjectDistances::object_at_pixel`] finds the object at a pixel of a
//! solved image with its distance. An object with no entry falls back to
//! [`typical_distance`] for its kind, which lives in code, not in the file.

use crate::objects::{
    ObjectCatalog, ObjectKind, ObjectQuery, ObjectQueryError, PlacedObject, SkyObject,
};
use crate::wcs::Wcs;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"SEIZADS1";
const ENTRY_SIZE: usize = 28;
const NONE: u32 = u32::MAX;
/// An object whose semi-major axis spans less than this fraction of the
/// image diagonal is a speck at that scale, such as a faint background
/// galaxy: [`ObjectDistances::object_at_pixel`] takes it only when nothing
/// larger holds the pixel.
const SPECK_FRACTION: f64 = 0.01;

/// How a distance was measured.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum DistanceMethod {
    /// Not stated by the source.
    Unknown = 0,
    /// Trigonometric parallax of the object, or of the star that lights or
    /// ionizes it.
    Parallax = 1,
    /// Parallaxes of a cluster's member stars.
    ClusterParallax = 2,
    /// Spectroscopic or photometric parallax, isochrone or main-sequence
    /// fitting of associated stars.
    Photometric = 3,
    /// Radial velocity and a Galactic rotation curve.
    Kinematic = 4,
    /// The distance at which extinction rises along the line of sight.
    Extinction = 5,
    /// A statistical relation, such as a planetary nebula's surface
    /// brightness against its radius.
    Statistical = 6,
    /// Cepheids, the tip of the red giant branch, RR Lyrae, the horizontal
    /// branch, supernovae, Tully–Fisher, the fundamental plane or surface
    /// brightness fluctuations.
    StandardCandle = 7,
    /// The Hubble flow, from the redshift.
    Redshift = 8,
    /// A compilation's adopted value, which mixes methods.
    Compiled = 9,
    /// Membership of a group or association whose distance is known.
    Membership = 10,
}

impl DistanceMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Parallax => "parallax",
            Self::ClusterParallax => "cluster-parallax",
            Self::Photometric => "photometric",
            Self::Kinematic => "kinematic",
            Self::Extinction => "extinction",
            Self::Statistical => "statistical",
            Self::StandardCandle => "standard-candle",
            Self::Redshift => "redshift",
            Self::Compiled => "compiled",
            Self::Membership => "membership",
        }
    }

    /// Values a newer file may add read as [`Self::Unknown`].
    pub fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Parallax,
            2 => Self::ClusterParallax,
            3 => Self::Photometric,
            4 => Self::Kinematic,
            5 => Self::Extinction,
            6 => Self::Statistical,
            7 => Self::StandardCandle,
            8 => Self::Redshift,
            9 => Self::Compiled,
            10 => Self::Membership,
            _ => Self::Unknown,
        }
    }
}

/// Whether a distance belongs to the object itself or to a neighbour.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum DistanceBasis {
    /// Measured for this object.
    Measured = 0,
    /// Borrowed from a Galactic object, named by `via`, that contains this
    /// one or lies inside it: a filament of a supernova remnant, a nebula
    /// around a cluster.
    Contained = 1,
    /// The distance of the galaxy, named by `via`, that this object lies in:
    /// a cluster in the Large Magellanic Cloud or M31.
    HostGalaxy = 2,
    /// The nearest molecular-cloud sightline, named by `via`, for a dark
    /// nebula.
    NearbyCloud = 3,
    /// The typical distance of the object's kind ([`typical_distance`]);
    /// never stored, only returned by [`ObjectDistances::estimate`].
    KindDefault = 4,
}

impl DistanceBasis {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Measured => "measured",
            Self::Contained => "contained",
            Self::HostGalaxy => "host-galaxy",
            Self::NearbyCloud => "nearby-cloud",
            Self::KindDefault => "kind-default",
        }
    }

    fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Measured),
            1 => Some(Self::Contained),
            2 => Some(Self::HostGalaxy),
            3 => Some(Self::NearbyCloud),
            _ => None,
        }
    }
}

/// A catalogue the file's distances come from.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DistanceSource {
    /// Short, stable key, e.g. `hunt-reffert-2024`.
    pub key: String,
    /// Authors, year, journal and catalogue ID, for attribution.
    pub citation: String,
    /// The terms the data is used under.
    pub licence: String,
    /// Where the data was fetched from.
    pub url: String,
}

/// A distance for one object.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObjectDistance<'a> {
    /// The object catalog's stable ID.
    pub object_id: &'a str,
    pub distance_pc: f64,
    pub lower_pc: Option<f64>,
    pub upper_pc: Option<f64>,
    /// `None` only for a [`DistanceBasis::KindDefault`] estimate.
    pub source: Option<&'a DistanceSource>,
    pub method: DistanceMethod,
    pub basis: DistanceBasis,
    /// The publication of the measurement, usually a bibcode.
    pub reference: Option<&'a str>,
    /// What a borrowed distance comes from: an object catalog ID, or for
    /// [`DistanceBasis::NearbyCloud`] a cloud sightline such as
    /// `zucker-2020:Taurus`.
    pub via: Option<&'a str>,
}

/// The typical distance of one object kind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KindDistance {
    pub kind: ObjectKind,
    /// Median of the measured distances.
    pub distance_pc: f64,
    /// 16th percentile.
    pub lower_pc: f64,
    /// 84th percentile.
    pub upper_pc: f64,
}

/// A rough distance for an object of `kind` that no source measures: the
/// median and 16th–84th percentiles of the measured distances of that kind
/// in the first published distance file (October 2026). `build-data
/// object-distances` prints the same figures for a new build. `None` for
/// stars, transients and unclassified objects.
pub fn typical_distance(kind: ObjectKind) -> Option<KindDistance> {
    let (distance_pc, lower_pc, upper_pc) = match kind {
        ObjectKind::Galaxy => (1.67e8, 7.8e7, 2.9e8),
        ObjectKind::OpenCluster => (1780.0, 910.0, 3130.0),
        ObjectKind::GlobularCluster => (8300.0, 5300.0, 14_200.0),
        ObjectKind::Nebula => (1140.0, 290.0, 3100.0),
        ObjectKind::PlanetaryNebula => (2630.0, 1200.0, 4610.0),
        ObjectKind::HiiRegion => (2690.0, 1020.0, 5090.0),
        ObjectKind::SupernovaRemnant => (4500.0, 1820.0, 9550.0),
        ObjectKind::DarkNebula => (200.0, 140.0, 590.0),
        ObjectKind::ClusterWithNebula => (1820.0, 490.0, 3120.0),
        ObjectKind::Association => (1400.0, 860.0, 2000.0),
        ObjectKind::Star | ObjectKind::DoubleStar | ObjectKind::Other | ObjectKind::Transient => {
            return None;
        }
    };
    Some(KindDistance {
        kind,
        distance_pc,
        lower_pc,
        upper_pc,
    })
}

/// A distance file opened against an object catalog it was not built from.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error(
    "the object distance file was built for object catalog {expected}, but this \
     object catalog is {found}; object IDs change between catalog builds, so use \
     the distance file published with this objects.bin or rebuild it with \
     `seiza build-data object-distances`"
)]
pub struct CatalogMismatch {
    /// Fingerprint recorded in the distance file, as hex.
    pub expected: String,
    /// Fingerprint of the catalog it was opened with, as hex.
    pub found: String,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// One distance for [`ObjectDistancesBuilder::add`].
#[derive(Clone, Debug, PartialEq)]
pub struct DistanceEntry {
    pub object_id: String,
    pub distance_pc: f64,
    pub lower_pc: Option<f64>,
    pub upper_pc: Option<f64>,
    /// Index returned by [`ObjectDistancesBuilder::add_source`].
    pub source: u8,
    pub method: DistanceMethod,
    pub basis: DistanceBasis,
    pub reference: Option<String>,
    pub via: Option<String>,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn pack_bound(value: Option<f64>) -> f32 {
    value
        .filter(|value| value.is_finite() && *value > 0.0)
        .map_or(f32::NAN, |value| value as f32)
}

fn unpack_bound(value: f32) -> Option<f64> {
    value.is_finite().then_some(f64::from(value))
}

fn write_str(out: &mut impl Write, value: &str) -> io::Result<()> {
    let length =
        u16::try_from(value.len()).map_err(|_| invalid(format!("string too long: {value}")))?;
    out.write_all(&length.to_le_bytes())?;
    out.write_all(value.as_bytes())
}

/// Accumulates distances in memory, then writes a `SEIZADS1` file.
#[derive(Debug, Default)]
pub struct ObjectDistancesBuilder {
    notice: String,
    catalog: [u8; 32],
    sources: Vec<DistanceSource>,
    entries: Vec<DistanceEntry>,
}

impl ObjectDistancesBuilder {
    /// `notice` states the licence and attribution for the whole file;
    /// `catalog` is [`ObjectCatalog::fingerprint`] of the catalog whose IDs
    /// the entries use.
    pub fn new(notice: &str, catalog: [u8; 32]) -> Self {
        Self {
            notice: notice.to_string(),
            catalog,
            ..Self::default()
        }
    }

    /// Register a source and return the index entries refer to it by.
    pub fn add_source(&mut self, source: DistanceSource) -> io::Result<u8> {
        let index = u8::try_from(self.sources.len())
            .ok()
            .filter(|&index| index < u8::MAX)
            .ok_or_else(|| invalid("a distance file holds at most 255 sources"))?;
        self.sources.push(source);
        Ok(index)
    }

    pub fn add(&mut self, entry: DistanceEntry) {
        self.entries.push(entry);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Write the file. Fails on a duplicate object ID, a distance that is
    /// not finite and positive, the basis reserved for estimates, or an
    /// unregistered source.
    pub fn write_to(mut self, path: &Path) -> io::Result<()> {
        self.entries
            .sort_by(|a, b| a.object_id.as_bytes().cmp(b.object_id.as_bytes()));
        if let Some(pair) = self
            .entries
            .windows(2)
            .find(|pair| pair[0].object_id == pair[1].object_id)
        {
            return Err(invalid(format!(
                "duplicate distance for {}",
                pair[0].object_id
            )));
        }
        let entry_count =
            u32::try_from(self.entries.len()).map_err(|_| invalid("too many distance entries"))?;

        let mut strings = Vec::new();
        let mut offsets = HashMap::<String, u32>::new();
        let mut intern = |value: &str| -> io::Result<u32> {
            if let Some(&offset) = offsets.get(value) {
                return Ok(offset);
            }
            let offset = u32::try_from(strings.len())
                .ok()
                .filter(|&offset| offset != NONE)
                .ok_or_else(|| invalid("distance string table too large"))?;
            write_str(&mut strings, value)?;
            offsets.insert(value.to_string(), offset);
            Ok(offset)
        };
        let mut records = Vec::with_capacity(self.entries.len() * ENTRY_SIZE);
        for entry in &self.entries {
            if !(entry.distance_pc.is_finite() && entry.distance_pc > 0.0) {
                return Err(invalid(format!(
                    "distance for {} is not positive",
                    entry.object_id
                )));
            }
            if usize::from(entry.source) >= self.sources.len() {
                return Err(invalid(format!(
                    "distance for {} names an unknown source",
                    entry.object_id
                )));
            }
            if entry.basis == DistanceBasis::KindDefault {
                return Err(invalid("kind defaults are not stored as entries"));
            }
            let id = intern(&entry.object_id)?;
            let reference = entry.reference.as_deref().map_or(Ok(NONE), &mut intern)?;
            let via = entry.via.as_deref().map_or(Ok(NONE), &mut intern)?;
            records.extend_from_slice(&id.to_le_bytes());
            records.extend_from_slice(&(entry.distance_pc as f32).to_le_bytes());
            records.extend_from_slice(&pack_bound(entry.lower_pc).to_le_bytes());
            records.extend_from_slice(&pack_bound(entry.upper_pc).to_le_bytes());
            records.extend_from_slice(&reference.to_le_bytes());
            records.extend_from_slice(&via.to_le_bytes());
            records.extend_from_slice(&[entry.source, entry.method as u8, entry.basis as u8, 0]);
        }
        let strings_len =
            u32::try_from(strings.len()).map_err(|_| invalid("distance string table too large"))?;

        let mut out = BufWriter::new(File::create(path)?);
        out.write_all(MAGIC)?;
        out.write_all(&entry_count.to_le_bytes())?;
        out.write_all(&[self.sources.len() as u8, 0, 0, 0])?;
        out.write_all(&strings_len.to_le_bytes())?;
        out.write_all(&self.catalog)?;
        write_str(&mut out, &self.notice)?;
        for source in &self.sources {
            for field in [&source.key, &source.citation, &source.licence, &source.url] {
                write_str(&mut out, field)?;
            }
        }
        out.write_all(&records)?;
        out.write_all(&strings)?;
        out.flush()
    }
}

#[derive(Clone, Copy, Debug)]
struct RawEntry {
    id: u32,
    distance: f32,
    lower: f32,
    upper: f32,
    reference: u32,
    via: u32,
    source: u8,
    method: u8,
    basis: DistanceBasis,
}

/// A `SEIZADS1` distance file, read into memory.
#[derive(Debug)]
pub struct ObjectDistances {
    notice: String,
    catalog: [u8; 32],
    sources: Vec<DistanceSource>,
    entries: Vec<RawEntry>,
    strings: Vec<u8>,
}

/// Reads fields front to back, failing on truncation.
struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, len: usize) -> io::Result<&'a [u8]> {
        let end = self
            .at
            .checked_add(len)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| invalid("distance file is truncated"))?;
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }

    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> io::Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn str(&mut self) -> io::Result<String> {
        let length = u16::from_le_bytes(self.take(2)?.try_into().unwrap());
        std::str::from_utf8(self.take(usize::from(length))?)
            .map(str::to_string)
            .map_err(|_| invalid("distance file string is not UTF-8"))
    }
}

/// The string at `offset` in a table, if it is in bounds and UTF-8.
fn table_str(strings: &[u8], offset: u32) -> Option<&str> {
    let start = usize::try_from(offset).ok()?;
    let length = u16::from_le_bytes(strings.get(start..start + 2)?.try_into().ok()?);
    std::str::from_utf8(strings.get(start + 2..start + 2 + usize::from(length))?).ok()
}

impl ObjectDistances {
    /// Open a distance file for use with `catalog`. Fails with
    /// [`io::ErrorKind::InvalidData`] wrapping a [`CatalogMismatch`] when the
    /// file was built from another object catalog: none of its lookups could
    /// be trusted.
    pub fn open(path: &Path, catalog: &ObjectCatalog) -> io::Result<Self> {
        let distances = Self::open_unpaired(path)?;
        distances.check_catalog(catalog)?;
        Ok(distances)
    }

    /// Open a distance file without pairing it to an object catalog, to
    /// inspect it. Lookups only mean something for the catalog whose
    /// fingerprint [`Self::catalog_fingerprint`] returns.
    pub fn open_unpaired(path: &Path) -> io::Result<Self> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    /// Parse and check a whole file, unpaired: every string in bounds, IDs
    /// sorted and unique, distances positive, sources registered.
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let mut cursor = Cursor { bytes, at: 0 };
        if cursor.take(8).ok() != Some(&MAGIC[..]) {
            return Err(invalid("not a SEIZADS1 object distance file"));
        }
        let entry_count = cursor.u32()? as usize;
        let source_count = cursor.u8()?;
        cursor.take(3)?;
        let strings_len = cursor.u32()? as usize;
        let catalog: [u8; 32] = cursor.take(32)?.try_into().unwrap();
        let notice = cursor.str()?;
        let mut sources = Vec::with_capacity(usize::from(source_count));
        for _ in 0..source_count {
            sources.push(DistanceSource {
                key: cursor.str()?,
                citation: cursor.str()?,
                licence: cursor.str()?,
                url: cursor.str()?,
            });
        }
        let records = cursor.take(
            entry_count
                .checked_mul(ENTRY_SIZE)
                .ok_or_else(|| invalid("distance file entry count overflows"))?,
        )?;
        let strings = cursor.take(strings_len)?.to_vec();
        if cursor.at != bytes.len() {
            return Err(invalid("distance file has trailing bytes"));
        }

        let mut entries = Vec::with_capacity(entry_count);
        let mut previous: Option<&str> = None;
        for record in records.chunks_exact(ENTRY_SIZE) {
            let mut fields = Cursor {
                bytes: record,
                at: 0,
            };
            let id = fields.u32()?;
            let distance = fields.f32()?;
            let lower = fields.f32()?;
            let upper = fields.f32()?;
            let reference = fields.u32()?;
            let via = fields.u32()?;
            let (source, method, basis) = (record[24], record[25], record[26]);
            let object_id = table_str(&strings, id)
                .ok_or_else(|| invalid("distance entry ID lies outside the string table"))?;
            if previous.is_some_and(|previous| previous.as_bytes() >= object_id.as_bytes()) {
                return Err(invalid("distance entries are not sorted by unique ID"));
            }
            previous = Some(object_id);
            for offset in [reference, via] {
                if offset != NONE && table_str(&strings, offset).is_none() {
                    return Err(invalid("distance entry string lies outside the table"));
                }
            }
            if !(distance.is_finite() && distance > 0.0) {
                return Err(invalid(format!("distance for {object_id} is not positive")));
            }
            if source >= source_count {
                return Err(invalid(format!(
                    "distance for {object_id} names an unknown source"
                )));
            }
            let basis = DistanceBasis::from_u8(basis)
                .ok_or_else(|| invalid(format!("distance for {object_id} has an unknown basis")))?;
            entries.push(RawEntry {
                id,
                distance,
                lower,
                upper,
                reference,
                via,
                source,
                method,
                basis,
            });
        }
        Ok(Self {
            notice,
            catalog,
            sources,
            entries,
            strings,
        })
    }

    /// [`ObjectCatalog::fingerprint`] of the catalog the file was built for.
    pub fn catalog_fingerprint(&self) -> [u8; 32] {
        self.catalog
    }

    /// Fail unless `catalog` is the one the file was built for, with
    /// [`io::ErrorKind::InvalidData`] wrapping a [`CatalogMismatch`].
    pub fn check_catalog(&self, catalog: &ObjectCatalog) -> io::Result<()> {
        let found = catalog.fingerprint()?;
        if found == self.catalog {
            return Ok(());
        }
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            CatalogMismatch {
                expected: hex(&self.catalog),
                found: hex(&found),
            },
        ))
    }

    /// Licence and attribution for the whole file.
    pub fn notice(&self) -> &str {
        &self.notice
    }

    pub fn sources(&self) -> &[DistanceSource] {
        &self.sources
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn string(&self, offset: u32) -> Option<&str> {
        (offset != NONE)
            .then(|| table_str(&self.strings, offset))
            .flatten()
    }

    fn resolve(&self, entry: &RawEntry) -> ObjectDistance<'_> {
        ObjectDistance {
            object_id: self.string(entry.id).unwrap_or_default(),
            distance_pc: f64::from(entry.distance),
            lower_pc: unpack_bound(entry.lower),
            upper_pc: unpack_bound(entry.upper),
            source: self.sources.get(usize::from(entry.source)),
            method: DistanceMethod::from_u8(entry.method),
            basis: entry.basis,
            reference: self.string(entry.reference),
            via: self.string(entry.via),
        }
    }

    /// The distance stored for an object catalog ID.
    pub fn get(&self, object_id: &str) -> Option<ObjectDistance<'_>> {
        self.entries
            .binary_search_by(|entry| {
                self.string(entry.id)
                    .unwrap_or_default()
                    .as_bytes()
                    .cmp(object_id.as_bytes())
            })
            .ok()
            .map(|index| self.resolve(&self.entries[index]))
    }

    /// The distance stored for an object, under its ID or any of its
    /// alternate IDs.
    pub fn for_object(&self, object: &SkyObject) -> Option<ObjectDistance<'_>> {
        std::iter::once(&object.metadata.id)
            .chain(&object.metadata.alternate_ids)
            .filter(|id| !id.is_empty())
            .find_map(|id| self.get(id))
    }

    /// [`Self::for_object`], or else [`typical_distance`] for the object's
    /// kind, marked [`DistanceBasis::KindDefault`]. `None` for stars and the
    /// other kinds that have no typical distance.
    pub fn estimate<'a>(&'a self, object: &'a SkyObject) -> Option<ObjectDistance<'a>> {
        self.for_object(object).or_else(|| {
            let default = typical_distance(object.kind)?;
            Some(ObjectDistance {
                object_id: &object.metadata.id,
                distance_pc: default.distance_pc,
                lower_pc: Some(default.lower_pc),
                upper_pc: Some(default.upper_pc),
                source: None,
                method: DistanceMethod::Unknown,
                basis: DistanceBasis::KindDefault,
                reference: None,
                via: None,
            })
        })
    }

    /// Every entry, sorted by object ID.
    pub fn iter(&self) -> impl Iterator<Item = ObjectDistance<'_>> {
        self.entries.iter().map(|entry| self.resolve(entry))
    }

    /// The catalogued object at a pixel of a solved image of size
    /// `dimensions`, such as the frame centre or the point a camera
    /// focused on, with its distance.
    ///
    /// Of the objects in the frame with a distance ([`Self::estimate`]),
    /// those whose catalogue ellipse holds the pixel come first, and of
    /// them the smallest: the most specific object there, so a pixel on the
    /// Horsehead finds Barnard 33 rather than IC 434. An object whose
    /// semi-major axis spans under 1% of the image diagonal, such as a faint
    /// galaxy behind a nebula, counts only when nothing larger holds the
    /// pixel. When no ellipse holds it, the object whose edge (or centre,
    /// for an object with no catalogued size) lies nearest wins, if within
    /// `search_radius_px`.
    ///
    /// `catalog` must be the catalog the file was built for, or this fails
    /// with [`ObjectQueryError::Catalog`] carrying the [`CatalogMismatch`]
    /// message.
    pub fn object_at_pixel<'a>(
        &'a self,
        catalog: &ObjectCatalog,
        wcs: &Wcs,
        dimensions: (u32, u32),
        pixel: (f64, f64),
        search_radius_px: f64,
    ) -> Result<Option<ObjectAtPixel<'a>>, ObjectQueryError> {
        self.check_catalog(catalog)
            .map_err(|error| ObjectQueryError::Catalog(error.to_string()))?;
        let diagonal = f64::from(dimensions.0).hypot(f64::from(dimensions.1));
        let mut best: Option<((bool, bool, f64), ObjectAtPixel<'a>)> = None;
        for placed in catalog.query_footprint(wcs, dimensions, &ObjectQuery::default())? {
            if self.estimate(&placed.object).is_none() {
                continue;
            }
            let offset_px = edge_offset(&placed, pixel);
            let inside = placed.semi_major_px > 0.0 && offset_px == 0.0;
            if !inside && offset_px > search_radius_px {
                continue;
            }
            let speck = placed.semi_major_px < SPECK_FRACTION * diagonal;
            let area = placed.semi_major_px * placed.semi_minor_px;
            // Lower sorts first: inside, then not a speck, then the smaller
            // ellipse inside or the nearer edge outside.
            let key = (
                !inside,
                inside && speck,
                if inside { area } else { offset_px },
            );
            if best.as_ref().is_none_or(|(current, _)| key < *current) {
                best = Some((
                    key,
                    ObjectAtPixel {
                        placed,
                        inside,
                        offset_px,
                        distances: self,
                    },
                ));
            }
        }
        Ok(best.map(|(_, found)| found))
    }
}

/// The object [`ObjectDistances::object_at_pixel`] finds.
#[derive(Clone, Debug)]
pub struct ObjectAtPixel<'a> {
    /// The object with its position and ellipse in the image.
    pub placed: PlacedObject,
    /// Whether the pixel lies inside the object's catalogue ellipse.
    pub inside: bool,
    /// Pixels from the pixel to the ellipse's edge, or to the centre of an
    /// object with no catalogued size; zero inside.
    pub offset_px: f64,
    distances: &'a ObjectDistances,
}

impl ObjectAtPixel<'_> {
    /// The object's distance: stored, or typical for its kind.
    pub fn distance(&self) -> Option<ObjectDistance<'_>> {
        self.distances.estimate(&self.placed.object)
    }
}

/// Pixels from `pixel` to the edge of a placed object's ellipse along the
/// line to its centre; zero inside, the distance to the centre for an
/// object with no size. An ellipse with no known orientation counts as the
/// circle of its major axis.
fn edge_offset(placed: &PlacedObject, (x, y): (f64, f64)) -> f64 {
    let (dx, dy) = (x - placed.x, y - placed.y);
    let distance = dx.hypot(dy);
    if placed.semi_major_px <= 0.0 {
        return distance;
    }
    let (a, b, angle) = match placed.angle_deg {
        Some(angle) => (
            placed.semi_major_px,
            placed.semi_minor_px.max(f64::MIN_POSITIVE),
            angle.to_radians(),
        ),
        None => (placed.semi_major_px, placed.semi_major_px, 0.0),
    };
    let (sin, cos) = angle.sin_cos();
    let along = dx * cos + dy * sin;
    let across = -dx * sin + dy * cos;
    let radius = ((along / a).powi(2) + (across / b).powi(2)).sqrt();
    if radius <= 1.0 {
        0.0
    } else {
        distance * (1.0 - 1.0 / radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::ObjectMetadata;

    fn source(key: &str) -> DistanceSource {
        DistanceSource {
            key: key.into(),
            citation: format!("{key} citation"),
            licence: "CC BY 4.0".into(),
            url: format!("https://example.test/{key}"),
        }
    }

    fn entry(id: &str, distance_pc: f64, source: u8) -> DistanceEntry {
        DistanceEntry {
            object_id: id.into(),
            distance_pc,
            lower_pc: None,
            upper_pc: None,
            source,
            method: DistanceMethod::Parallax,
            basis: DistanceBasis::Measured,
            reference: None,
            via: None,
        }
    }

    fn object(id: &str, kind: ObjectKind, alternate_ids: &[&str]) -> SkyObject {
        SkyObject {
            kind,
            ra: 0.0,
            dec: 0.0,
            mag: None,
            major_arcmin: None,
            minor_arcmin: None,
            position_angle_deg: None,
            name: id.into(),
            common_name: String::new(),
            metadata: ObjectMetadata {
                id: id.into(),
                alternate_ids: alternate_ids.iter().map(|id| id.to_string()).collect(),
                ..ObjectMetadata::default()
            },
        }
    }

    /// Orion around the Horsehead: IC 434 (60'), Barnard 33 (6') inside it,
    /// a 0.5' background galaxy on the Horsehead, and M 42 to the north.
    fn orion() -> Vec<SkyObject> {
        let place = |mut object: SkyObject, ra: f64, dec: f64, major: f32| {
            object.ra = ra;
            object.dec = dec;
            object.major_arcmin = Some(major);
            object
        };
        vec![
            place(
                object("openngc:IC434", ObjectKind::Nebula, &[]),
                85.25,
                -2.4,
                60.0,
            ),
            place(
                object("openngc:B33", ObjectKind::DarkNebula, &[]),
                85.25,
                -2.46,
                6.0,
            ),
            place(
                object("test:PGC1", ObjectKind::Galaxy, &[]),
                85.255,
                -2.46,
                0.5,
            ),
            place(
                object("openngc:NGC1976", ObjectKind::ClusterWithNebula, &[]),
                83.82,
                -5.39,
                66.0,
            ),
            object("iau-csn:Alnitak", ObjectKind::Star, &[]),
        ]
    }

    /// A v4 object catalog file of `objects`.
    fn catalog(directory: &Path, name: &str, objects: Vec<SkyObject>) -> ObjectCatalog {
        let path = directory.join(name);
        ObjectCatalog::new(objects).write_to(&path).unwrap();
        ObjectCatalog::open(&path).unwrap()
    }

    fn sample(fingerprint: [u8; 32]) -> ObjectDistancesBuilder {
        let mut builder = ObjectDistancesBuilder::new("ODbL 1.0; test notice", fingerprint);
        let clusters = builder.add_source(source("clusters")).unwrap();
        let simbad = builder.add_source(source("simbad")).unwrap();
        builder.add(DistanceEntry {
            lower_pc: Some(134.8),
            upper_pc: Some(134.9),
            method: DistanceMethod::ClusterParallax,
            reference: Some("2024A&A...686A..42H".into()),
            ..entry("openngc:MEL22", 134.84, clusters)
        });
        builder.add(DistanceEntry {
            lower_pc: Some(418.0),
            upper_pc: None,
            reference: Some("2020A&A...633A..51Z".into()),
            ..entry("openngc:NGC1976", 433.0, simbad)
        });
        builder.add(DistanceEntry {
            basis: DistanceBasis::Contained,
            via: Some("openngc:IC434".into()),
            reference: Some("2020A&A...633A..51Z".into()),
            ..entry("openngc:B33", 400.0, simbad)
        });
        builder.add(entry("openngc:IC434", 400.0, simbad));
        builder
    }

    #[test]
    fn distances_round_trip_through_a_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("object-distances.bin");
        let builder = sample([7; 32]);
        assert_eq!(builder.len(), 4);
        builder.write_to(&path).unwrap();
        let distances = ObjectDistances::open_unpaired(&path).unwrap();
        assert_eq!(distances.len(), 4);
        assert_eq!(distances.catalog_fingerprint(), [7; 32]);
        assert_eq!(distances.notice(), "ODbL 1.0; test notice");
        assert_eq!(
            distances
                .sources()
                .iter()
                .map(|source| source.key.as_str())
                .collect::<Vec<_>>(),
            ["clusters", "simbad"]
        );
        assert_eq!(distances.sources()[1], source("simbad"));

        let pleiades = distances.get("openngc:MEL22").unwrap();
        assert!((pleiades.distance_pc - 134.84).abs() < 1e-3);
        assert!((pleiades.lower_pc.unwrap() - 134.8).abs() < 1e-3);
        assert!((pleiades.upper_pc.unwrap() - 134.9).abs() < 1e-3);
        assert_eq!(pleiades.source.unwrap().key, "clusters");
        assert_eq!(pleiades.method, DistanceMethod::ClusterParallax);
        assert_eq!(pleiades.basis, DistanceBasis::Measured);
        assert_eq!(pleiades.reference, Some("2024A&A...686A..42H"));
        assert_eq!(pleiades.via, None);

        let orion = distances.get("openngc:NGC1976").unwrap();
        assert_eq!((orion.lower_pc, orion.upper_pc), (Some(418.0), None));
        let horsehead = distances.get("openngc:B33").unwrap();
        assert_eq!(horsehead.basis, DistanceBasis::Contained);
        assert_eq!(horsehead.via, Some("openngc:IC434"));
        // Interned: both entries share one reference string.
        assert_eq!(horsehead.reference, orion.reference);

        assert!(distances.get("openngc:NGC1977").is_none());
        assert!(distances.get("").is_none());
        let ids = distances
            .iter()
            .map(|distance| distance.object_id)
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            [
                "openngc:B33",
                "openngc:IC434",
                "openngc:MEL22",
                "openngc:NGC1976"
            ]
        );
    }

    #[test]
    fn lookup_by_object_tries_alternate_ids_then_the_kind_default() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("object-distances.bin");
        sample([0; 32]).write_to(&path).unwrap();
        let distances = ObjectDistances::open_unpaired(&path).unwrap();

        let merged = object(
            "openngc:IC433",
            ObjectKind::Nebula,
            &["vizier:VII/9:LBN954", "openngc:NGC1976"],
        );
        let found = distances.for_object(&merged).unwrap();
        assert_eq!(found.object_id, "openngc:NGC1976");
        assert_eq!(distances.estimate(&merged), Some(found));

        let cloud = object("vizier:VII/7A:LDN1", ObjectKind::DarkNebula, &[]);
        assert!(distances.for_object(&cloud).is_none());
        let estimate = distances.estimate(&cloud).unwrap();
        let typical = typical_distance(ObjectKind::DarkNebula).unwrap();
        assert_eq!(estimate.basis, DistanceBasis::KindDefault);
        assert_eq!(estimate.object_id, "vizier:VII/7A:LDN1");
        assert_eq!(
            (estimate.distance_pc, estimate.lower_pc, estimate.upper_pc),
            (
                typical.distance_pc,
                Some(typical.lower_pc),
                Some(typical.upper_pc)
            )
        );
        assert!(estimate.source.is_none());

        let star = object("iau-csn:Vega", ObjectKind::Star, &[]);
        assert!(distances.estimate(&star).is_none());
    }

    #[test]
    fn typical_distances_are_ordered_and_skip_stars() {
        for kind in [
            ObjectKind::Galaxy,
            ObjectKind::OpenCluster,
            ObjectKind::GlobularCluster,
            ObjectKind::Nebula,
            ObjectKind::PlanetaryNebula,
            ObjectKind::HiiRegion,
            ObjectKind::SupernovaRemnant,
            ObjectKind::DarkNebula,
            ObjectKind::ClusterWithNebula,
            ObjectKind::Association,
        ] {
            let typical = typical_distance(kind).unwrap();
            assert_eq!(typical.kind, kind);
            assert!(
                typical.lower_pc < typical.distance_pc && typical.distance_pc < typical.upper_pc
            );
        }
        for kind in [
            ObjectKind::Star,
            ObjectKind::DoubleStar,
            ObjectKind::Other,
            ObjectKind::Transient,
        ] {
            assert!(typical_distance(kind).is_none());
        }
    }

    #[test]
    fn opening_with_another_object_catalog_fails_plainly() {
        let directory = tempfile::tempdir().unwrap();
        let built_for = catalog(directory.path(), "objects.bin", orion());
        let path = directory.path().join("object-distances.bin");
        sample(built_for.fingerprint().unwrap())
            .write_to(&path)
            .unwrap();
        let distances = ObjectDistances::open(&path, &built_for).unwrap();
        assert_eq!(distances.get("openngc:B33").unwrap().distance_pc, 400.0);

        // A rebuilt catalog: one object moved, so the IDs may mean other things.
        let mut rebuilt = orion();
        rebuilt[0].ra += 0.01;
        let rebuilt = catalog(directory.path(), "objects-rebuilt.bin", rebuilt);
        let error = ObjectDistances::open(&path, &rebuilt).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        let mismatch = error
            .get_ref()
            .and_then(|inner| inner.downcast_ref::<CatalogMismatch>())
            .unwrap();
        assert_eq!(mismatch.expected, hex(&built_for.fingerprint().unwrap()));
        assert_eq!(mismatch.found, hex(&rebuilt.fingerprint().unwrap()));
        assert!(error.to_string().contains("rebuild it"), "{error}");

        // A pixel lookup against the wrong catalog refuses too.
        let unpaired = ObjectDistances::open_unpaired(&path).unwrap();
        let wcs = Wcs::from_center_scale_rotation((85.25, -2.46), (500.0, 500.0), 3.0, 0.0, false);
        let refused = unpaired
            .object_at_pixel(&rebuilt, &wcs, (1000, 1000), (500.0, 500.0), 20.0)
            .unwrap_err();
        assert!(
            matches!(refused, ObjectQueryError::Catalog(ref message) if message.contains("built for"))
        );
    }

    #[test]
    fn the_object_at_a_pixel_is_the_smallest_one_holding_it() {
        let directory = tempfile::tempdir().unwrap();
        let objects = catalog(directory.path(), "objects.bin", orion());
        let path = directory.path().join("object-distances.bin");
        sample(objects.fingerprint().unwrap())
            .write_to(&path)
            .unwrap();
        let distances = ObjectDistances::open(&path, &objects).unwrap();
        // 3"/px, 1000 × 1000 px: a 50' field on the Horsehead.
        let wcs = Wcs::from_center_scale_rotation((85.25, -2.46), (500.0, 500.0), 3.0, 0.0, false);
        let at = |pixel: (f64, f64), radius: f64| {
            distances
                .object_at_pixel(&objects, &wcs, (1000, 1000), pixel, radius)
                .unwrap()
        };

        // On the Horsehead: Barnard 33 rather than IC 434 around it, and not
        // the 0.5' background galaxy, a speck in a 50' field.
        let found = at((500.0, 500.0), 0.0).unwrap();
        assert_eq!(found.placed.object.metadata.id, "openngc:B33");
        assert!(found.inside);
        assert_eq!(found.offset_px, 0.0);
        let distance = found.distance().unwrap();
        assert_eq!(distance.distance_pc, 400.0);
        assert_eq!(distance.basis, DistanceBasis::Contained);

        // 10' north of the Horsehead: inside IC 434 only.
        let north = wcs.world_to_pixel(85.25, -2.46 + 10.0 / 60.0).unwrap();
        let found = at(north, 0.0).unwrap();
        assert_eq!(found.placed.object.metadata.id, "openngc:IC434");
        assert_eq!(found.distance().unwrap().basis, DistanceBasis::Measured);

        // In a 5' field the galaxy is no speck, so a pixel on it finds it,
        // with the typical galaxy distance.
        let narrow =
            Wcs::from_center_scale_rotation((85.255, -2.46), (50.0, 50.0), 3.0, 0.0, false);
        let found = distances
            .object_at_pixel(&objects, &narrow, (100, 100), (50.0, 50.0), 0.0)
            .unwrap()
            .unwrap();
        assert_eq!(found.placed.object.metadata.id, "test:PGC1");
        assert_eq!(found.distance().unwrap().basis, DistanceBasis::KindDefault);

        // Off every ellipse, the nearest edge within the search radius wins.
        let outside = wcs.world_to_pixel(85.25, -2.4 - 31.0 / 60.0).unwrap();
        assert!(at(outside, 5.0).is_none());
        let found = at(outside, 40.0).unwrap();
        assert_eq!(found.placed.object.metadata.id, "openngc:IC434");
        assert!(!found.inside);
        assert!((found.offset_px - 20.0).abs() < 1.0, "{}", found.offset_px);
    }

    #[test]
    fn edge_offsets_follow_the_ellipse_orientation() {
        let placed = |angle_deg: Option<f64>| PlacedObject {
            object: object("test:E", ObjectKind::Galaxy, &[]),
            x: 100.0,
            y: 100.0,
            semi_major_px: 20.0,
            semi_minor_px: 5.0,
            angle_deg,
        };
        // Major axis along +x.
        assert_eq!(edge_offset(&placed(Some(0.0)), (115.0, 100.0)), 0.0);
        assert!((edge_offset(&placed(Some(0.0)), (100.0, 115.0)) - 10.0).abs() < 1e-9);
        // Turned to +y.
        assert_eq!(edge_offset(&placed(Some(90.0)), (100.0, 115.0)), 0.0);
        assert!((edge_offset(&placed(Some(90.0)), (115.0, 100.0)) - 10.0).abs() < 1e-9);
        // No orientation: the circle of the major axis.
        assert_eq!(edge_offset(&placed(None), (100.0, 115.0)), 0.0);
        // No size: the distance to the centre.
        let point = PlacedObject {
            semi_major_px: 0.0,
            semi_minor_px: 0.0,
            ..placed(None)
        };
        assert_eq!(edge_offset(&point, (103.0, 104.0)), 5.0);
    }

    #[test]
    fn builder_rejects_duplicates_bad_distances_and_unknown_sources() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("bad.bin");
        let mut builder = sample([0; 32]);
        builder.add(entry("openngc:MEL22", 136.0, 0));
        assert!(builder.write_to(&path).is_err());

        for bad in [
            entry("a", 0.0, 0),
            entry("a", f64::NAN, 0),
            entry("a", 10.0, 7),
            DistanceEntry {
                basis: DistanceBasis::KindDefault,
                ..entry("a", 10.0, 0)
            },
        ] {
            let mut builder = sample([0; 32]);
            builder.add(bad);
            assert!(builder.write_to(&path).is_err());
        }
    }

    #[test]
    fn reader_rejects_damaged_files() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("object-distances.bin");
        sample([0; 32]).write_to(&path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert!(ObjectDistances::from_bytes(&bytes).is_ok());

        let mut wrong_magic = bytes.clone();
        wrong_magic[7] = b'9';
        assert!(ObjectDistances::from_bytes(&wrong_magic).is_err());
        assert!(ObjectDistances::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(ObjectDistances::from_bytes(&trailing).is_err());
        assert!(ObjectDistances::open_unpaired(&directory.path().join("missing")).is_err());
    }
}
