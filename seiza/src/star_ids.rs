//! Offline lookup of source-qualified stellar catalog identifiers.
//!
//! `SEIZASI1` is an optional sidecar to the solver-oriented star tile files.
//! It keeps identifier lookup out of the hot astrometric scan while allowing
//! applications to resolve designations such as `TYC 5949-2777-1`,
//! `RR Lyr`, or `STF 2382 AB` without an online catalog service.
//!
//! ```text
//! magic         [u8; 8] = b"SEIZASI1"
//! numeric_count u64
//! name_count    u64
//! string_bytes  u64
//! epoch         f64 Julian year for stored coordinates
//! attribution   u16 byte length + UTF-8 bytes
//! padding       to an 8-byte boundary
//! numeric       sorted fixed-width 24-byte records:
//!                 namespace u8, reserved[3], value u64,
//!                 ra u32, dec u32, magnitude u16, reserved u16
//! names         sorted fixed-width 40-byte records containing string-table
//!                 ranges, catalog/kind, coordinates, and optional magnitude
//! strings       deduplicated UTF-8 bytes used by name records
//! ```

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::str::FromStr;

const MAGIC: &[u8; 8] = b"SEIZASI1";
const HEADER_FIXED_SIZE: usize = 42;
const NUMERIC_RECORD_SIZE: usize = 24;
const NAME_RECORD_SIZE: usize = 40;
const MAG_OFFSET: f32 = 3.0;
const MAX_PACKED_MAG: f32 = u16::MAX as f32 / 1000.0 - MAG_OFFSET;

/// Namespace of a source-qualified stellar catalog identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum StarIdNamespace {
    Tycho2 = 1,
    Hipparcos = 2,
    HenryDraper = 3,
    HenryDraperExtension = 4,
    HarvardRevised = 5,
    GaiaDr3 = 6,
    Sao = 7,
    Fk5 = 8,
}

impl StarIdNamespace {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Tycho2 => "Tycho-2",
            Self::Hipparcos => "Hipparcos",
            Self::HenryDraper => "Henry Draper",
            Self::HenryDraperExtension => "Henry Draper Extension",
            Self::HarvardRevised => "Harvard Revised",
            Self::GaiaDr3 => "Gaia DR3",
            Self::Sao => "SAO",
            Self::Fk5 => "FK5",
        }
    }

    fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Tycho2),
            2 => Some(Self::Hipparcos),
            3 => Some(Self::HenryDraper),
            4 => Some(Self::HenryDraperExtension),
            5 => Some(Self::HarvardRevised),
            6 => Some(Self::GaiaDr3),
            7 => Some(Self::Sao),
            8 => Some(Self::Fk5),
            _ => None,
        }
    }
}

/// A canonical stellar catalog designation supported by the identifier index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StarIdentifier {
    Tycho2 {
        region: u16,
        number: u16,
        component: u8,
    },
    Hipparcos(u32),
    HenryDraper(u32),
    HenryDraperExtension(u32),
    HarvardRevised(u32),
    GaiaDr3(u64),
    Sao(u32),
    Fk5(u32),
}

impl StarIdentifier {
    pub fn namespace(self) -> StarIdNamespace {
        match self {
            Self::Tycho2 { .. } => StarIdNamespace::Tycho2,
            Self::Hipparcos(_) => StarIdNamespace::Hipparcos,
            Self::HenryDraper(_) => StarIdNamespace::HenryDraper,
            Self::HenryDraperExtension(_) => StarIdNamespace::HenryDraperExtension,
            Self::HarvardRevised(_) => StarIdNamespace::HarvardRevised,
            Self::GaiaDr3(_) => StarIdNamespace::GaiaDr3,
            Self::Sao(_) => StarIdNamespace::Sao,
            Self::Fk5(_) => StarIdNamespace::Fk5,
        }
    }

    /// Stable source-qualified form suitable for APIs and persisted metadata.
    pub fn stable_id(self) -> String {
        match self {
            Self::Tycho2 {
                region,
                number,
                component,
            } => format!("tycho2:{region}-{number}-{component}"),
            Self::Hipparcos(number) => format!("hip:{number}"),
            Self::HenryDraper(number) => format!("hd:{number}"),
            Self::HenryDraperExtension(number) => format!("hde:{number}"),
            Self::HarvardRevised(number) => format!("hr:{number}"),
            Self::GaiaDr3(number) => format!("gaia-dr3:{number}"),
            Self::Sao(number) => format!("sao:{number}"),
            Self::Fk5(number) => format!("fk5:{number}"),
        }
    }

    fn encoded(self) -> (u8, u64) {
        let value = match self {
            Self::Tycho2 {
                region,
                number,
                component,
            } => ((region as u64) << 17) | ((number as u64) << 3) | component as u64,
            Self::Hipparcos(number)
            | Self::HenryDraper(number)
            | Self::HenryDraperExtension(number)
            | Self::HarvardRevised(number)
            | Self::Sao(number)
            | Self::Fk5(number) => number as u64,
            Self::GaiaDr3(number) => number,
        };
        (self.namespace() as u8, value)
    }

    fn from_encoded(namespace: u8, value: u64) -> Option<Self> {
        match StarIdNamespace::from_u8(namespace)? {
            StarIdNamespace::Tycho2 => {
                let component = (value & 0x7) as u8;
                let number = ((value >> 3) & 0x3fff) as u16;
                let region = ((value >> 17) & 0x3fff) as u16;
                (value >> 31 == 0 && region > 0 && number > 0 && (1..=4).contains(&component))
                    .then_some(Self::Tycho2 {
                        region,
                        number,
                        component,
                    })
            }
            StarIdNamespace::Hipparcos => u32::try_from(value)
                .ok()
                .filter(|value| *value > 0)
                .map(Self::Hipparcos),
            StarIdNamespace::HenryDraper => u32::try_from(value)
                .ok()
                .filter(|value| *value > 0)
                .map(Self::HenryDraper),
            StarIdNamespace::HenryDraperExtension => u32::try_from(value)
                .ok()
                .filter(|value| *value > 0)
                .map(Self::HenryDraperExtension),
            StarIdNamespace::HarvardRevised => u32::try_from(value)
                .ok()
                .filter(|value| *value > 0)
                .map(Self::HarvardRevised),
            StarIdNamespace::GaiaDr3 => (value > 0).then_some(Self::GaiaDr3(value)),
            StarIdNamespace::Sao => u32::try_from(value)
                .ok()
                .filter(|value| *value > 0)
                .map(Self::Sao),
            StarIdNamespace::Fk5 => u32::try_from(value)
                .ok()
                .filter(|value| *value > 0)
                .map(Self::Fk5),
        }
    }
}

impl fmt::Display for StarIdentifier {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Tycho2 {
                region,
                number,
                component,
            } => write!(formatter, "TYC {region}-{number}-{component}"),
            Self::Hipparcos(number) => write!(formatter, "HIP {number}"),
            Self::HenryDraper(number) => write!(formatter, "HD {number}"),
            Self::HenryDraperExtension(number) => write!(formatter, "HDE {number}"),
            Self::HarvardRevised(number) => write!(formatter, "HR {number}"),
            Self::GaiaDr3(number) => write!(formatter, "Gaia DR3 {number}"),
            Self::Sao(number) => write!(formatter, "SAO {number}"),
            Self::Fk5(number) => write!(formatter, "FK5 {number}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "invalid stellar catalog identifier {input:?}; expected TYC A-B-C, HIP N, HD N, HDE N, HR N, Gaia DR3 N, SAO N, or FK5 N"
)]
pub struct ParseStarIdentifierError {
    input: String,
}

impl FromStr for StarIdentifier {
    type Err = ParseStarIdentifierError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let upper = input.trim().to_ascii_uppercase();
        let invalid = || ParseStarIdentifierError {
            input: input.to_string(),
        };

        if let Some(rest) = upper
            .strip_prefix("TYCHO2")
            .or_else(|| upper.strip_prefix("TYCHO-2"))
            .or_else(|| upper.strip_prefix("TYC"))
        {
            if !rest
                .chars()
                .all(|character| character.is_ascii_digit() || " -:".contains(character))
            {
                return Err(invalid());
            }
            let values = numeric_parts(rest);
            if values.len() != 3 {
                return Err(invalid());
            }
            let (region, number, component) = (
                values[0].parse::<u16>().map_err(|_| invalid())?,
                values[1].parse::<u16>().map_err(|_| invalid())?,
                values[2].parse::<u8>().map_err(|_| invalid())?,
            );
            if region == 0
                || region > 0x3fff
                || number == 0
                || number > 0x3fff
                || !(1..=4).contains(&component)
            {
                return Err(invalid());
            }
            return Ok(Self::Tycho2 {
                region,
                number,
                component,
            });
        }

        if let Some(rest) = upper
            .strip_prefix("GAIA DR3")
            .or_else(|| upper.strip_prefix("GAIA-DR3"))
            .or_else(|| upper.strip_prefix("GAIADR3"))
        {
            let value = single_number::<u64>(rest).ok_or_else(invalid)?;
            return (value > 0)
                .then_some(Self::GaiaDr3(value))
                .ok_or_else(invalid);
        }

        for (prefix, constructor) in [
            ("HIPPARCOS", Self::Hipparcos as fn(u32) -> Self),
            ("HIP", Self::Hipparcos),
            ("HDE", Self::HenryDraperExtension),
            ("HD", Self::HenryDraper),
            ("HR", Self::HarvardRevised),
            ("SAO", Self::Sao),
            ("FK5", Self::Fk5),
        ] {
            if let Some(rest) = upper.strip_prefix(prefix) {
                let value = single_number::<u32>(rest).ok_or_else(invalid)?;
                return (value > 0)
                    .then_some(constructor(value))
                    .ok_or_else(invalid);
            }
        }

        Err(invalid())
    }
}

fn numeric_parts(value: &str) -> Vec<&str> {
    value
        .split(|character: char| !character.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .collect()
}

fn single_number<T: FromStr>(value: &str) -> Option<T> {
    let value = value
        .trim()
        .strip_prefix(':')
        .or_else(|| value.trim().strip_prefix('-'))
        .unwrap_or(value.trim())
        .trim();
    (!value.is_empty() && value.chars().all(|character| character.is_ascii_digit()))
        .then(|| value.parse().ok())
        .flatten()
}

/// Source catalog for a textual stellar designation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum StarNameCatalog {
    IauCatalogOfStarNames = 1,
    BrightStarCatalog = 2,
    GeneralCatalogOfVariableStars = 3,
    WashingtonDoubleStar = 4,
}

impl StarNameCatalog {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::IauCatalogOfStarNames => "IAU Catalog of Star Names",
            Self::BrightStarCatalog => "Bright Star Catalogue",
            Self::GeneralCatalogOfVariableStars => "GCVS",
            Self::WashingtonDoubleStar => "Washington Double Star Catalog",
        }
    }

    fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::IauCatalogOfStarNames),
            2 => Some(Self::BrightStarCatalog),
            3 => Some(Self::GeneralCatalogOfVariableStars),
            4 => Some(Self::WashingtonDoubleStar),
            _ => None,
        }
    }
}

/// Semantic role of a textual stellar designation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum StarNameKind {
    ProperName = 1,
    BayerFlamsteed = 2,
    VariableStar = 3,
    DoubleStar = 4,
}

impl StarNameKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ProperName => "proper-name",
            Self::BayerFlamsteed => "bayer-flamsteed",
            Self::VariableStar => "variable-star",
            Self::DoubleStar => "double-star",
        }
    }

    fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::ProperName),
            2 => Some(Self::BayerFlamsteed),
            3 => Some(Self::VariableStar),
            4 => Some(Self::DoubleStar),
            _ => None,
        }
    }
}

/// Normalize a human-facing stellar designation for case-, spacing-, and
/// punctuation-insensitive exact lookup and prefix completion.
pub fn normalize_star_name(value: &str) -> String {
    let characters = value.chars().collect::<Vec<_>>();
    let mut normalized = String::new();
    for (index, character) in characters.iter().copied().enumerate() {
        if character.is_alphanumeric() {
            normalized.extend(character.to_uppercase());
        } else if character == '+' {
            normalized.push(character);
        } else if character == '-' && index > 0 && index + 1 < characters.len() {
            let left = characters[index - 1].to_ascii_uppercase();
            let right = characters[index + 1].to_ascii_uppercase();
            let numeric_sign = left.is_ascii_digit() && right.is_ascii_digit();
            let component_separator = ('A'..='D').contains(&left) && ('A'..='D').contains(&right);
            if numeric_sign || component_separator {
                normalized.push(character);
            }
        }
    }
    normalized
}

/// One exact identifier match from a [`StarIdentifierCatalog`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IdentifiedStar {
    pub identifier: StarIdentifier,
    pub ra: f64,
    pub dec: f64,
    pub mag: f32,
}

/// One textual designation match borrowed from a memory-mapped catalog.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NamedStar<'a> {
    pub designation: &'a str,
    pub stable_id: &'a str,
    pub catalog: StarNameCatalog,
    pub kind: StarNameKind,
    /// Catalog-specific concise metadata: variability type or components.
    pub detail: &'a str,
    pub ra: f64,
    pub dec: f64,
    pub mag: Option<f32>,
}

/// Exact stellar lookup result from either the compact numeric index or the
/// textual designation index.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StarLookupMatch<'a> {
    Identifier(IdentifiedStar),
    Name(NamedStar<'a>),
}

#[derive(Debug, Clone, Copy)]
struct PackedEntry {
    namespace: u8,
    value: u64,
    ra: u32,
    dec: u32,
    mag: u16,
}

impl PackedEntry {
    fn key(self) -> (u8, u64) {
        (self.namespace, self.value)
    }

    fn sort_key(self) -> (u8, u64, u32, u32, u16) {
        (self.namespace, self.value, self.ra, self.dec, self.mag)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NameEntry {
    key: String,
    designation: String,
    stable_id: String,
    detail: String,
    catalog: StarNameCatalog,
    kind: StarNameKind,
    ra: u32,
    dec: u32,
    mag: u16,
}

impl NameEntry {
    fn sort_key(
        &self,
    ) -> (
        &str,
        &str,
        &str,
        StarNameCatalog,
        StarNameKind,
        u32,
        u32,
        u16,
    ) {
        (
            &self.key,
            &self.designation,
            &self.stable_id,
            self.catalog,
            self.kind,
            self.ra,
            self.dec,
            self.mag,
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct PackedNameEntry {
    catalog: u8,
    kind: u8,
    key_offset: u32,
    key_len: u16,
    designation_len: u16,
    designation_offset: u32,
    stable_id_offset: u32,
    stable_id_len: u16,
    detail_len: u16,
    detail_offset: u32,
    ra: u32,
    dec: u32,
    mag: u16,
}

#[derive(Default)]
struct StringTable {
    bytes: Vec<u8>,
    entries: HashMap<String, (u32, u16)>,
}

impl StringTable {
    fn intern(&mut self, value: &str) -> io::Result<(u32, u16)> {
        if let Some(&range) = self.entries.get(value) {
            return Ok(range);
        }
        let end = self
            .bytes
            .len()
            .checked_add(value.len())
            .ok_or_else(|| invalid_input("star name string table is too large"))?;
        if end > u32::MAX as usize {
            return Err(invalid_input("star name string table exceeds 4 GiB"));
        }
        let offset = u32::try_from(self.bytes.len())
            .map_err(|_| invalid_input("star name string table exceeds 4 GiB"))?;
        let len = u16::try_from(value.len())
            .map_err(|_| invalid_input("one star name string exceeds 65535 bytes"))?;
        self.bytes.extend_from_slice(value.as_bytes());
        self.entries.insert(value.to_string(), (offset, len));
        Ok((offset, len))
    }
}

/// Builds a compact exact-identifier sidecar without changing star tiles.
pub struct StarIdentifierCatalogBuilder {
    epoch: f64,
    attribution: String,
    entries: Vec<PackedEntry>,
    names: Vec<NameEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StarIdentifierCatalogStats {
    pub numeric_entries: usize,
    pub name_entries: usize,
    pub string_bytes: usize,
}

impl StarIdentifierCatalogBuilder {
    pub fn new(epoch: f64, attribution: &str) -> Self {
        Self {
            epoch,
            attribution: attribution.to_string(),
            entries: Vec::new(),
            names: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len() + self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.names.is_empty()
    }

    pub fn numeric_len(&self) -> usize {
        self.entries.len()
    }

    pub fn name_len(&self) -> usize {
        self.names.len()
    }

    pub fn add(
        &mut self,
        identifier: StarIdentifier,
        ra: f64,
        dec: f64,
        mag: f32,
    ) -> io::Result<()> {
        if !ra.is_finite() || !dec.is_finite() || !mag.is_finite() || !(-90.0..=90.0).contains(&dec)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "star identifier entry has invalid coordinates or magnitude",
            ));
        }
        if !(-MAG_OFFSET..=MAX_PACKED_MAG).contains(&mag) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "star identifier magnitude is outside the packed range",
            ));
        }
        let (namespace, value) = identifier.encoded();
        if StarIdentifier::from_encoded(namespace, value) != Some(identifier) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "stellar catalog identifier is outside its namespace range",
            ));
        }
        self.entries.push(PackedEntry {
            namespace,
            value,
            ra: pack_ra(ra),
            dec: pack_dec(dec),
            mag: pack_mag(mag),
        });
        Ok(())
    }

    /// Add a textual catalog designation. `detail` is a concise
    /// catalog-specific value such as a GCVS variability type or WDS component
    /// label. Magnitude may be absent in the source catalog.
    #[allow(clippy::too_many_arguments)]
    pub fn add_name(
        &mut self,
        catalog: StarNameCatalog,
        kind: StarNameKind,
        designation: &str,
        stable_id: &str,
        detail: &str,
        ra: f64,
        dec: f64,
        mag: Option<f32>,
    ) -> io::Result<()> {
        let designation = designation.trim();
        let stable_id = stable_id.trim();
        let key = normalize_star_name(designation);
        if key.is_empty() || stable_id.is_empty() {
            return Err(invalid_input(
                "star name designation and stable ID must not be empty",
            ));
        }
        if !ra.is_finite() || !dec.is_finite() || !(-90.0..=90.0).contains(&dec) {
            return Err(invalid_input("star name entry has invalid coordinates"));
        }
        if mag.is_some_and(|value| {
            !value.is_finite() || !(-MAG_OFFSET..MAX_PACKED_MAG).contains(&value)
        }) {
            return Err(invalid_input(
                "star name magnitude is outside the packed range",
            ));
        }
        self.names.push(NameEntry {
            key,
            designation: designation.to_string(),
            stable_id: stable_id.to_string(),
            detail: detail.trim().to_string(),
            catalog,
            kind,
            ra: pack_ra(ra),
            dec: pack_dec(dec),
            mag: mag.map(pack_mag).unwrap_or(u16::MAX),
        });
        Ok(())
    }

    pub fn write_to(mut self, path: &Path) -> io::Result<StarIdentifierCatalogStats> {
        if !self.epoch.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "star identifier epoch must be finite",
            ));
        }
        let attribution = self.attribution.as_bytes();
        let attribution_len = u16::try_from(attribution.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "star identifier attribution exceeds 65535 bytes",
            )
        })?;
        self.entries.sort_by_key(|entry| entry.sort_key());
        self.entries.dedup_by_key(|entry| entry.sort_key());
        self.names
            .sort_by(|left, right| left.sort_key().cmp(&right.sort_key()));
        self.names.dedup();

        let mut strings = StringTable::default();
        let mut packed_names = Vec::with_capacity(self.names.len());
        for name in &self.names {
            let (key_offset, key_len) = strings.intern(&name.key)?;
            let (designation_offset, designation_len) = strings.intern(&name.designation)?;
            let (stable_id_offset, stable_id_len) = strings.intern(&name.stable_id)?;
            let (detail_offset, detail_len) = strings.intern(&name.detail)?;
            packed_names.push(PackedNameEntry {
                catalog: name.catalog as u8,
                kind: name.kind as u8,
                key_offset,
                key_len,
                designation_len,
                designation_offset,
                stable_id_offset,
                stable_id_len,
                detail_len,
                detail_offset,
                ra: name.ra,
                dec: name.dec,
                mag: name.mag,
            });
        }
        let numeric_entries = self.entries.len();
        let name_entries = self.names.len();
        let string_bytes = strings.bytes.len();

        let mut output = BufWriter::new(File::create(path)?);
        output.write_all(MAGIC)?;
        output.write_all(&(numeric_entries as u64).to_le_bytes())?;
        output.write_all(&(name_entries as u64).to_le_bytes())?;
        output.write_all(&(string_bytes as u64).to_le_bytes())?;
        output.write_all(&self.epoch.to_le_bytes())?;
        output.write_all(&attribution_len.to_le_bytes())?;
        output.write_all(attribution)?;
        let header_len = HEADER_FIXED_SIZE + attribution.len();
        let padding = header_len.next_multiple_of(8) - header_len;
        output.write_all(&vec![0; padding])?;

        for entry in self.entries {
            output.write_all(&[entry.namespace, 0, 0, 0])?;
            output.write_all(&entry.value.to_le_bytes())?;
            output.write_all(&entry.ra.to_le_bytes())?;
            output.write_all(&entry.dec.to_le_bytes())?;
            output.write_all(&entry.mag.to_le_bytes())?;
            output.write_all(&[0, 0])?;
        }
        for entry in packed_names {
            output.write_all(&[entry.catalog, entry.kind, 0, 0])?;
            output.write_all(&entry.key_offset.to_le_bytes())?;
            output.write_all(&entry.key_len.to_le_bytes())?;
            output.write_all(&entry.designation_len.to_le_bytes())?;
            output.write_all(&entry.designation_offset.to_le_bytes())?;
            output.write_all(&entry.stable_id_offset.to_le_bytes())?;
            output.write_all(&entry.stable_id_len.to_le_bytes())?;
            output.write_all(&entry.detail_len.to_le_bytes())?;
            output.write_all(&entry.detail_offset.to_le_bytes())?;
            output.write_all(&entry.ra.to_le_bytes())?;
            output.write_all(&entry.dec.to_le_bytes())?;
            output.write_all(&entry.mag.to_le_bytes())?;
            output.write_all(&[0, 0])?;
        }
        output.write_all(&strings.bytes)?;
        output.flush()?;
        Ok(StarIdentifierCatalogStats {
            numeric_entries,
            name_entries,
            string_bytes,
        })
    }
}

/// Read-only, memory-mapped exact lookup over stellar catalog identifiers.
pub struct StarIdentifierCatalog {
    map: memmap2::Mmap,
    epoch: f64,
    attribution: String,
    numeric_offset: usize,
    numeric_count: usize,
    names_offset: usize,
    name_count: usize,
    strings_offset: usize,
    string_bytes: usize,
}

impl StarIdentifierCatalog {
    /// Memory-map a stellar identifier sidecar and validate its header and
    /// section bounds. Opening does not scan the numeric records, textual
    /// records, or string table; use [`Self::validate`] when accepting an
    /// untrusted catalog and full-file integrity checking is required.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        // Safety: the file is opened read-only; concurrent truncation has the
        // same constraints as the existing memory-mapped star tile catalog.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        if map.len() < HEADER_FIXED_SIZE || &map[..8] != MAGIC {
            return Err(invalid_data("not a seiza star identifier catalog"));
        }
        let numeric_count = read_count(&map[8..16], "numeric identifier")?;
        let name_count = read_count(&map[16..24], "star name")?;
        let string_bytes = read_count(&map[24..32], "star name string byte")?;
        let epoch = f64::from_le_bytes(map[32..40].try_into().unwrap());
        if !epoch.is_finite() {
            return Err(invalid_data("star identifier epoch is not finite"));
        }
        let attribution_len = u16::from_le_bytes(map[40..42].try_into().unwrap()) as usize;
        let attribution_end = HEADER_FIXED_SIZE
            .checked_add(attribution_len)
            .ok_or_else(|| invalid_data("invalid star identifier header"))?;
        let numeric_offset = attribution_end.next_multiple_of(8);
        let numeric_bytes = numeric_count
            .checked_mul(NUMERIC_RECORD_SIZE)
            .ok_or_else(|| invalid_data("star identifier catalog is too large"))?;
        let names_offset = numeric_offset
            .checked_add(numeric_bytes)
            .ok_or_else(|| invalid_data("star identifier catalog is too large"))?;
        let names_bytes = name_count
            .checked_mul(NAME_RECORD_SIZE)
            .ok_or_else(|| invalid_data("star name catalog is too large"))?;
        let strings_offset = names_offset
            .checked_add(names_bytes)
            .ok_or_else(|| invalid_data("star name catalog is too large"))?;
        let expected_len = strings_offset
            .checked_add(string_bytes)
            .ok_or_else(|| invalid_data("star identifier catalog is too large"))?;
        if attribution_end > map.len() || expected_len != map.len() {
            return Err(invalid_data("star identifier file length is inconsistent"));
        }
        let attribution = std::str::from_utf8(&map[HEADER_FIXED_SIZE..attribution_end])
            .map_err(|_| invalid_data("star identifier attribution is not UTF-8"))?
            .to_string();

        Ok(Self {
            map,
            epoch,
            attribution,
            numeric_offset,
            numeric_count,
            names_offset,
            name_count,
            strings_offset,
            string_bytes,
        })
    }

    /// Exhaustively validate every record and the complete UTF-8 string
    /// table. This intentionally touches the whole memory mapping and is
    /// separate from [`Self::open`] so normal startup remains demand-paged.
    pub fn validate(&self) -> io::Result<()> {
        std::str::from_utf8(&self.map[self.strings_offset..])
            .map_err(|_| invalid_data("star name string table is not UTF-8"))?;

        let mut previous = None;
        for index in 0..self.numeric_count {
            let entry = self.packed_entry(index);
            if StarIdentifier::from_encoded(entry.namespace, entry.value).is_none() {
                return Err(invalid_data("invalid star identifier record"));
            }
            if previous.is_some_and(|key| key > entry.sort_key()) {
                return Err(invalid_data("star identifier records are not sorted"));
            }
            previous = Some(entry.sort_key());
        }
        let mut previous_key: Option<&str> = None;
        for index in 0..self.name_count {
            let entry = self.packed_name_entry(index);
            if StarNameCatalog::from_u8(entry.catalog).is_none()
                || StarNameKind::from_u8(entry.kind).is_none()
            {
                return Err(invalid_data("invalid star name catalog or kind"));
            }
            let key = self.name_string(entry.key_offset, entry.key_len)?;
            let designation = self.name_string(entry.designation_offset, entry.designation_len)?;
            let stable_id = self.name_string(entry.stable_id_offset, entry.stable_id_len)?;
            self.name_string(entry.detail_offset, entry.detail_len)?;
            if key.is_empty() || designation.is_empty() || stable_id.is_empty() {
                return Err(invalid_data("star name record contains an empty key or ID"));
            }
            if previous_key.is_some_and(|previous| previous > key) {
                return Err(invalid_data("star name records are not sorted"));
            }
            previous_key = Some(key);
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.numeric_count + self.name_count
    }

    pub fn is_empty(&self) -> bool {
        self.numeric_count == 0 && self.name_count == 0
    }

    pub fn numeric_len(&self) -> usize {
        self.numeric_count
    }

    pub fn name_len(&self) -> usize {
        self.name_count
    }

    pub fn attribution(&self) -> &str {
        &self.attribution
    }

    /// Julian year of the coordinates stored in this catalog.
    pub fn epoch(&self) -> f64 {
        self.epoch
    }

    /// Return every entry matching an exact identifier. Multiple results are
    /// valid for catalog identifiers that refer to unresolved components.
    pub fn lookup(&self, identifier: StarIdentifier) -> Vec<IdentifiedStar> {
        let target = identifier.encoded();
        let mut low = 0usize;
        let mut high = self.numeric_count;
        while low < high {
            let middle = low + (high - low) / 2;
            if self.packed_entry(middle).key() < target {
                low = middle + 1;
            } else {
                high = middle;
            }
        }

        let mut matches = Vec::new();
        let mut index = low;
        while index < self.numeric_count {
            let entry = self.packed_entry(index);
            if entry.key() != target {
                break;
            }
            matches.push(IdentifiedStar {
                identifier,
                ra: unpack_ra(entry.ra),
                dec: unpack_dec(entry.dec),
                mag: unpack_mag(entry.mag),
            });
            index += 1;
        }
        matches
    }

    pub fn lookup_str(
        &self,
        identifier: &str,
    ) -> Result<Vec<IdentifiedStar>, ParseStarIdentifierError> {
        Ok(self.lookup(identifier.parse()?))
    }

    /// Resolve a human-facing designation such as `Vega`, `RR Lyr`,
    /// `WDS 18367+3849`, or `STF 2382 AB`.
    pub fn lookup_name(&self, designation: &str) -> io::Result<Vec<NamedStar<'_>>> {
        let key = normalize_star_name(designation);
        if key.is_empty() {
            return Ok(Vec::new());
        }
        let mut index = self.lower_bound_name(&key)?;
        let mut matches = Vec::new();
        while index < self.name_count {
            let entry = self.packed_name_entry(index);
            if self.name_key(entry)? != key {
                break;
            }
            matches.push(self.named_star(entry)?);
            index += 1;
        }
        Ok(matches)
    }

    /// Prefix completion for interactive name search. Results stay in
    /// normalized lexical order and may contain more than one catalog record
    /// for the same displayed designation.
    pub fn search_names(&self, prefix: &str, limit: usize) -> io::Result<Vec<NamedStar<'_>>> {
        let prefix = normalize_star_name(prefix);
        if prefix.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let mut index = self.lower_bound_name(&prefix)?;
        let mut matches = Vec::new();
        while index < self.name_count && matches.len() < limit {
            let entry = self.packed_name_entry(index);
            if !self.name_key(entry)?.starts_with(&prefix) {
                break;
            }
            matches.push(self.named_star(entry)?);
            index += 1;
        }
        Ok(matches)
    }

    /// Resolve either a typed numeric identifier or a textual designation.
    pub fn lookup_query(&self, query: &str) -> io::Result<Vec<StarLookupMatch<'_>>> {
        if let Ok(identifier) = query.parse::<StarIdentifier>() {
            return Ok(self
                .lookup(identifier)
                .into_iter()
                .map(StarLookupMatch::Identifier)
                .collect());
        }
        Ok(self
            .lookup_name(query)?
            .into_iter()
            .map(StarLookupMatch::Name)
            .collect())
    }

    fn lower_bound_name(&self, key: &str) -> io::Result<usize> {
        let mut low = 0usize;
        let mut high = self.name_count;
        while low < high {
            let middle = low + (high - low) / 2;
            let entry = self.packed_name_entry(middle);
            if self.name_key(entry)? < key {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        Ok(low)
    }

    fn named_star(&self, entry: PackedNameEntry) -> io::Result<NamedStar<'_>> {
        let designation = self.name_string(entry.designation_offset, entry.designation_len)?;
        let stable_id = self.name_string(entry.stable_id_offset, entry.stable_id_len)?;
        if designation.is_empty() || stable_id.is_empty() {
            return Err(invalid_data(
                "star name record contains an empty designation or ID",
            ));
        }
        Ok(NamedStar {
            designation,
            stable_id,
            catalog: StarNameCatalog::from_u8(entry.catalog)
                .ok_or_else(|| invalid_data("invalid star name catalog"))?,
            kind: StarNameKind::from_u8(entry.kind)
                .ok_or_else(|| invalid_data("invalid star name kind"))?,
            detail: self.name_string(entry.detail_offset, entry.detail_len)?,
            ra: unpack_ra(entry.ra),
            dec: unpack_dec(entry.dec),
            mag: (entry.mag != u16::MAX).then(|| unpack_mag(entry.mag)),
        })
    }

    fn name_key(&self, entry: PackedNameEntry) -> io::Result<&str> {
        let key = self.name_string(entry.key_offset, entry.key_len)?;
        if key.is_empty() {
            return Err(invalid_data("star name record contains an empty key"));
        }
        Ok(key)
    }

    fn name_string(&self, offset: u32, len: u16) -> io::Result<&str> {
        let start = offset as usize;
        let end = start
            .checked_add(len as usize)
            .ok_or_else(|| invalid_data("invalid star name string range"))?;
        if end > self.string_bytes {
            return Err(invalid_data("star name string range is out of bounds"));
        }
        std::str::from_utf8(&self.map[self.strings_offset + start..self.strings_offset + end])
            .map_err(|_| invalid_data("star name string is not UTF-8"))
    }

    fn packed_name_entry(&self, index: usize) -> PackedNameEntry {
        let start = self.names_offset + index * NAME_RECORD_SIZE;
        let record = &self.map[start..start + NAME_RECORD_SIZE];
        PackedNameEntry {
            catalog: record[0],
            kind: record[1],
            key_offset: u32::from_le_bytes(record[4..8].try_into().unwrap()),
            key_len: u16::from_le_bytes(record[8..10].try_into().unwrap()),
            designation_len: u16::from_le_bytes(record[10..12].try_into().unwrap()),
            designation_offset: u32::from_le_bytes(record[12..16].try_into().unwrap()),
            stable_id_offset: u32::from_le_bytes(record[16..20].try_into().unwrap()),
            stable_id_len: u16::from_le_bytes(record[20..22].try_into().unwrap()),
            detail_len: u16::from_le_bytes(record[22..24].try_into().unwrap()),
            detail_offset: u32::from_le_bytes(record[24..28].try_into().unwrap()),
            ra: u32::from_le_bytes(record[28..32].try_into().unwrap()),
            dec: u32::from_le_bytes(record[32..36].try_into().unwrap()),
            mag: u16::from_le_bytes(record[36..38].try_into().unwrap()),
        }
    }

    fn packed_entry(&self, index: usize) -> PackedEntry {
        let start = self.numeric_offset + index * NUMERIC_RECORD_SIZE;
        let record = &self.map[start..start + NUMERIC_RECORD_SIZE];
        PackedEntry {
            namespace: record[0],
            value: u64::from_le_bytes(record[4..12].try_into().unwrap()),
            ra: u32::from_le_bytes(record[12..16].try_into().unwrap()),
            dec: u32::from_le_bytes(record[16..20].try_into().unwrap()),
            mag: u16::from_le_bytes(record[20..22].try_into().unwrap()),
        }
    }
}

fn pack_ra(ra: f64) -> u32 {
    (ra.rem_euclid(360.0) / 360.0 * u32::MAX as f64) as u32
}

fn unpack_ra(value: u32) -> f64 {
    value as f64 / u32::MAX as f64 * 360.0
}

fn pack_dec(dec: f64) -> u32 {
    ((dec + 90.0) / 180.0 * u32::MAX as f64) as u32
}

fn unpack_dec(value: u32) -> f64 {
    value as f64 / u32::MAX as f64 * 180.0 - 90.0
}

fn pack_mag(mag: f32) -> u16 {
    ((mag + MAG_OFFSET) * 1000.0).round() as u16
}

fn unpack_mag(value: u16) -> f32 {
    value as f32 / 1000.0 - MAG_OFFSET
}

fn read_count(bytes: &[u8], label: &'static str) -> io::Result<usize> {
    usize::try_from(u64::from_le_bytes(bytes.try_into().unwrap())).map_err(|_| {
        invalid_data(match label {
            "numeric identifier" => "numeric identifier count does not fit this platform",
            "star name" => "star name count does not fit this platform",
            _ => "star name string table does not fit this platform",
        })
    })
}

fn invalid_input(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn invalid_data(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_formats_supported_designations() {
        let tyc: StarIdentifier = "tyc 5949-2777-1".parse().unwrap();
        assert_eq!(tyc.to_string(), "TYC 5949-2777-1");
        assert_eq!(tyc.stable_id(), "tycho2:5949-2777-1");
        assert_eq!("HIP:32349".parse(), Ok(StarIdentifier::Hipparcos(32349)));
        assert_eq!(
            "Gaia-DR3 123456789".parse(),
            Ok(StarIdentifier::GaiaDr3(123456789))
        );
        assert_eq!(
            "hde:225301".parse(),
            Ok(StarIdentifier::HenryDraperExtension(225301))
        );
        assert_eq!("tycho2:5949-2777-1".parse(), Ok(tyc));
        assert_eq!("FK5 1007".parse(), Ok(StarIdentifier::Fk5(1007)));
        assert_eq!(normalize_star_name("STF 2382 AB"), "STF2382AB");
        assert_eq!(normalize_star_name("WDS 18367+3849"), "WDS18367+3849");
        assert_eq!(normalize_star_name("WDS 18367-3849"), "WDS18367-3849");
        assert_eq!(normalize_star_name("RR-Lyr"), "RRLYR");
        assert_eq!(normalize_star_name("STF 1 AB-C"), "STF1AB-C");
        assert!("TYC 1-2".parse::<StarIdentifier>().is_err());
        assert!("TYC 1-2-8".parse::<StarIdentifier>().is_err());
        assert!("HIP abc32349".parse::<StarIdentifier>().is_err());
        assert!("unknown 42".parse::<StarIdentifier>().is_err());
    }

    #[test]
    fn round_trips_and_preserves_duplicate_component_matches() {
        let dir =
            std::env::temp_dir().join(format!("seiza-star-identifiers-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("stars.ids.bin");
        let mut builder = StarIdentifierCatalogBuilder::new(2025.5, "test source");
        builder
            .add(
                StarIdentifier::Tycho2 {
                    region: 5949,
                    number: 2777,
                    component: 1,
                },
                101.28854,
                -16.71314,
                -1.088,
            )
            .unwrap();
        builder
            .add(
                StarIdentifier::Hipparcos(32349),
                101.28854,
                -16.71314,
                -1.088,
            )
            .unwrap();
        builder
            .add(StarIdentifier::Hipparcos(32349), 101.28860, -16.71300, 7.0)
            .unwrap();
        assert!(
            builder
                .add(
                    StarIdentifier::Tycho2 {
                        region: 1,
                        number: 2,
                        component: 8,
                    },
                    1.0,
                    2.0,
                    3.0,
                )
                .is_err()
        );
        builder
            .add_name(
                StarNameCatalog::GeneralCatalogOfVariableStars,
                StarNameKind::VariableStar,
                "RR Lyr",
                "gcvs:RR-LYR",
                "RRAB",
                291.3663,
                42.7844,
                Some(7.06),
            )
            .unwrap();
        builder
            .add_name(
                StarNameCatalog::WashingtonDoubleStar,
                StarNameKind::DoubleStar,
                "STF 2382 AB",
                "wds:18367+3849:STF2382:AB",
                "AB",
                279.2347,
                38.7837,
                None,
            )
            .unwrap();
        builder
            .add_name(
                StarNameCatalog::IauCatalogOfStarNames,
                StarNameKind::ProperName,
                "Vega",
                "iau-csn:Vega",
                "",
                279.2347,
                38.7837,
                Some(0.03),
            )
            .unwrap();
        builder.write_to(&path).unwrap();

        let catalog = StarIdentifierCatalog::open(&path).unwrap();
        catalog.validate().unwrap();
        assert_eq!(catalog.len(), 6);
        assert_eq!(catalog.numeric_len(), 3);
        assert_eq!(catalog.name_len(), 3);
        assert_eq!(catalog.epoch(), 2025.5);
        assert_eq!(catalog.attribution(), "test source");
        let tyc = catalog.lookup_str("TYC 5949-2777-1").unwrap();
        assert_eq!(tyc.len(), 1);
        assert!((tyc[0].ra - 101.28854).abs() < 1e-5);
        assert!((tyc[0].mag - -1.088).abs() < 0.002);
        assert_eq!(catalog.lookup(StarIdentifier::Hipparcos(32349)).len(), 2);
        assert!(
            catalog
                .lookup(StarIdentifier::HenryDraper(48915))
                .is_empty()
        );
        let variable = catalog.lookup_name("rr-lyr").unwrap();
        assert_eq!(variable.len(), 1);
        assert_eq!(variable[0].designation, "RR Lyr");
        assert_eq!(variable[0].detail, "RRAB");
        assert_eq!(variable[0].kind, StarNameKind::VariableStar);
        let double = catalog.lookup_name("stf2382ab").unwrap();
        assert_eq!(double.len(), 1);
        assert_eq!(double[0].mag, None);
        let completion = catalog.search_names("v", 5).unwrap();
        assert_eq!(completion.len(), 1);
        assert_eq!(completion[0].designation, "Vega");
        assert!(matches!(
            catalog.lookup_query("Vega").unwrap().as_slice(),
            [StarLookupMatch::Name(NamedStar {
                designation: "Vega",
                ..
            })]
        ));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn open_defers_record_and_string_validation_until_touched() {
        let dir = std::env::temp_dir().join(format!(
            "seiza-star-identifiers-lazy-open-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let valid_path = dir.join("valid.ids.bin");
        let mut builder = StarIdentifierCatalogBuilder::new(2025.5, "test source");
        builder
            .add_name(
                StarNameCatalog::GeneralCatalogOfVariableStars,
                StarNameKind::VariableStar,
                "RR Lyr",
                "gcvs:RRLYR",
                "RRAB",
                291.3663,
                42.7844,
                Some(7.06),
            )
            .unwrap();
        builder.write_to(&valid_path).unwrap();

        let valid = std::fs::read(&valid_path).unwrap();
        let numeric_count = u64::from_le_bytes(valid[8..16].try_into().unwrap()) as usize;
        let name_count = u64::from_le_bytes(valid[16..24].try_into().unwrap()) as usize;
        let attribution_len = u16::from_le_bytes(valid[40..42].try_into().unwrap()) as usize;
        let numeric_offset = (HEADER_FIXED_SIZE + attribution_len).next_multiple_of(8);
        let names_offset = numeric_offset + numeric_count * NUMERIC_RECORD_SIZE;
        let strings_offset = names_offset + name_count * NAME_RECORD_SIZE;

        let invalid_record_path = dir.join("invalid-record.ids.bin");
        let mut invalid_record = valid.clone();
        invalid_record[names_offset] = u8::MAX;
        std::fs::write(&invalid_record_path, invalid_record).unwrap();
        let catalog = StarIdentifierCatalog::open(&invalid_record_path).unwrap();
        assert!(catalog.validate().is_err());
        assert!(catalog.lookup_name("RR Lyr").is_err());

        let invalid_string_path = dir.join("invalid-string.ids.bin");
        let mut invalid_string = valid;
        invalid_string[strings_offset] = u8::MAX;
        std::fs::write(&invalid_string_path, invalid_string).unwrap();
        let catalog = StarIdentifierCatalog::open(&invalid_string_path).unwrap();
        assert!(catalog.validate().is_err());
        assert!(catalog.lookup_name("RR Lyr").is_err());

        std::fs::remove_dir_all(&dir).ok();
    }
}
