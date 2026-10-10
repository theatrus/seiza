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
//! default_count  u8
//! reserved       u16          zero
//! strings_len    u32
//! notice         str          licence and attribution for the whole file
//! sources        source_count × { key: str, citation: str, licence: str, url: str }
//! defaults       default_count × { kind: u8, objects: u32, distance_pc: f32,
//!                lower_pc: f32, upper_pc: f32 }
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
//! `defaults` holds, per object kind, the median and 16th–84th percentile
//! range of that kind's measured distances in the file:
//! [`ObjectDistances::estimate`] falls back to it for an object with no entry.

use crate::objects::{ObjectKind, SkyObject};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"SEIZADS1";
const ENTRY_SIZE: usize = 28;
const DEFAULT_SIZE: usize = 17;
const NONE: u32 = u32::MAX;

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
    /// The typical distance of the object's kind; never stored, only
    /// returned by [`ObjectDistances::estimate`].
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
    /// How many measured distances the figures come from.
    pub objects: u32,
    /// Median of the measured distances.
    pub distance_pc: f64,
    /// 16th percentile.
    pub lower_pc: f64,
    /// 84th percentile.
    pub upper_pc: f64,
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
    sources: Vec<DistanceSource>,
    defaults: Vec<KindDistance>,
    entries: Vec<DistanceEntry>,
}

impl ObjectDistancesBuilder {
    /// `notice` states the licence and attribution for the whole file.
    pub fn new(notice: &str) -> Self {
        Self {
            notice: notice.to_string(),
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

    pub fn set_kind_default(&mut self, default: KindDistance) {
        self.defaults
            .retain(|existing| existing.kind != default.kind);
        self.defaults.push(default);
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
        let default_count =
            u8::try_from(self.defaults.len()).map_err(|_| invalid("too many kind defaults"))?;

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
        out.write_all(&[self.sources.len() as u8, default_count, 0, 0])?;
        out.write_all(&strings_len.to_le_bytes())?;
        write_str(&mut out, &self.notice)?;
        for source in &self.sources {
            for field in [&source.key, &source.citation, &source.licence, &source.url] {
                write_str(&mut out, field)?;
            }
        }
        for default in &self.defaults {
            out.write_all(&[default.kind as u8])?;
            out.write_all(&default.objects.to_le_bytes())?;
            for value in [default.distance_pc, default.lower_pc, default.upper_pc] {
                out.write_all(&(value as f32).to_le_bytes())?;
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
    sources: Vec<DistanceSource>,
    defaults: Vec<KindDistance>,
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
    pub fn open(path: &Path) -> io::Result<Self> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    /// Parse and check a whole file: every string in bounds, IDs sorted and
    /// unique, distances positive, sources registered.
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let mut cursor = Cursor { bytes, at: 0 };
        if cursor.take(8).ok() != Some(&MAGIC[..]) {
            return Err(invalid("not a SEIZADS1 object distance file"));
        }
        let entry_count = cursor.u32()? as usize;
        let source_count = cursor.u8()?;
        let default_count = cursor.u8()?;
        cursor.take(2)?;
        let strings_len = cursor.u32()? as usize;
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
        let mut defaults = Vec::with_capacity(usize::from(default_count));
        for _ in 0..default_count {
            let record = cursor.take(DEFAULT_SIZE)?;
            let mut fields = Cursor {
                bytes: record,
                at: 1,
            };
            defaults.push(KindDistance {
                kind: ObjectKind::from_u8(record[0]),
                objects: fields.u32()?,
                distance_pc: f64::from(fields.f32()?),
                lower_pc: f64::from(fields.f32()?),
                upper_pc: f64::from(fields.f32()?),
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
            sources,
            defaults,
            entries,
            strings,
        })
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

    /// The typical distance of an object kind, when the file has one.
    pub fn kind_default(&self, kind: ObjectKind) -> Option<&KindDistance> {
        self.defaults.iter().find(|default| default.kind == kind)
    }

    pub fn kind_defaults(&self) -> &[KindDistance] {
        &self.defaults
    }

    /// [`Self::for_object`], or else the typical distance of the object's
    /// kind with its 16th–84th percentile range. `None` for kinds the file
    /// measures nothing of, such as stars.
    pub fn estimate<'a>(&'a self, object: &'a SkyObject) -> Option<ObjectDistance<'a>> {
        self.for_object(object).or_else(|| {
            let default = self.kind_default(object.kind)?;
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

    fn sample() -> ObjectDistancesBuilder {
        let mut builder = ObjectDistancesBuilder::new("ODbL 1.0; test notice");
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
            via: Some("openngc:NGC1976".into()),
            reference: Some("2020A&A...633A..51Z".into()),
            ..entry("vizier:VII/220A:B33", 433.0, simbad)
        });
        builder.set_kind_default(KindDistance {
            kind: ObjectKind::DarkNebula,
            objects: 12,
            distance_pc: 300.0,
            lower_pc: 150.0,
            upper_pc: 900.0,
        });
        builder
    }

    #[test]
    fn distances_round_trip_through_a_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("object-distances.bin");
        let builder = sample();
        assert_eq!(builder.len(), 3);
        builder.write_to(&path).unwrap();
        let distances = ObjectDistances::open(&path).unwrap();
        assert_eq!(distances.len(), 3);
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
        let horsehead = distances.get("vizier:VII/220A:B33").unwrap();
        assert_eq!(horsehead.basis, DistanceBasis::Contained);
        assert_eq!(horsehead.via, Some("openngc:NGC1976"));
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
            ["openngc:MEL22", "openngc:NGC1976", "vizier:VII/220A:B33"]
        );
        let default = distances.kind_default(ObjectKind::DarkNebula).unwrap();
        assert_eq!((default.objects, default.distance_pc), (12, 300.0));
        assert!(distances.kind_default(ObjectKind::Galaxy).is_none());
    }

    #[test]
    fn lookup_by_object_tries_alternate_ids_then_the_kind_default() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("object-distances.bin");
        sample().write_to(&path).unwrap();
        let distances = ObjectDistances::open(&path).unwrap();

        let merged = object(
            "openngc:IC434",
            ObjectKind::Nebula,
            &["vizier:VII/9:LBN954", "openngc:NGC1976"],
        );
        let found = distances.for_object(&merged).unwrap();
        assert_eq!(found.object_id, "openngc:NGC1976");
        assert_eq!(distances.estimate(&merged), Some(found));

        let cloud = object("vizier:VII/7A:LDN1", ObjectKind::DarkNebula, &[]);
        assert!(distances.for_object(&cloud).is_none());
        let estimate = distances.estimate(&cloud).unwrap();
        assert_eq!(estimate.basis, DistanceBasis::KindDefault);
        assert_eq!(estimate.object_id, "vizier:VII/7A:LDN1");
        assert_eq!(
            (estimate.distance_pc, estimate.lower_pc, estimate.upper_pc),
            (300.0, Some(150.0), Some(900.0))
        );
        assert!(estimate.source.is_none());

        let star = object("iau-csn:Vega", ObjectKind::Star, &[]);
        assert!(distances.estimate(&star).is_none());
    }

    #[test]
    fn builder_rejects_duplicates_bad_distances_and_unknown_sources() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("bad.bin");
        let mut builder = sample();
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
            let mut builder = sample();
            builder.add(bad);
            assert!(builder.write_to(&path).is_err());
        }
    }

    #[test]
    fn reader_rejects_damaged_files() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("object-distances.bin");
        sample().write_to(&path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert!(ObjectDistances::from_bytes(&bytes).is_ok());

        let mut wrong_magic = bytes.clone();
        wrong_magic[7] = b'9';
        assert!(ObjectDistances::from_bytes(&wrong_magic).is_err());
        assert!(ObjectDistances::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(ObjectDistances::from_bytes(&trailing).is_err());
        assert!(ObjectDistances::open(&directory.path().join("missing")).is_err());
    }
}
