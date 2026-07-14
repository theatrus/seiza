//! Offline lookup of source-qualified stellar catalog identifiers.
//!
//! `SEIZASI1` is an optional sidecar to the solver-oriented star tile files.
//! It keeps identifier lookup out of the hot astrometric scan while allowing
//! applications to resolve designations such as `TYC 5949-2777-1` or
//! `HIP 32349` without an online catalog service.
//!
//! ```text
//! magic         [u8; 8] = b"SEIZASI1"
//! entry_count   u64
//! epoch          f64 Julian year for stored coordinates
//! attribution   u16 byte length + UTF-8 bytes
//! padding       to an 8-byte boundary
//! entries       sorted fixed-width 24-byte records:
//!                 namespace u8, reserved[3], value u64,
//!                 ra u32, dec u32, magnitude u16, reserved u16
//! ```

use std::fmt;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::str::FromStr;

const MAGIC: &[u8; 8] = b"SEIZASI1";
const RECORD_SIZE: usize = 24;
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
            | Self::Sao(number) => number as u64,
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
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "invalid stellar catalog identifier {input:?}; expected TYC A-B-C, HIP N, HD N, HDE N, HR N, Gaia DR3 N, or SAO N"
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

/// One exact identifier match from a [`StarIdentifierCatalog`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IdentifiedStar {
    pub identifier: StarIdentifier,
    pub ra: f64,
    pub dec: f64,
    pub mag: f32,
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

/// Builds a compact exact-identifier sidecar without changing star tiles.
pub struct StarIdentifierCatalogBuilder {
    epoch: f64,
    attribution: String,
    entries: Vec<PackedEntry>,
}

impl StarIdentifierCatalogBuilder {
    pub fn new(epoch: f64, attribution: &str) -> Self {
        Self {
            epoch,
            attribution: attribution.to_string(),
            entries: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
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

    pub fn write_to(mut self, path: &Path) -> io::Result<()> {
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

        let mut output = BufWriter::new(File::create(path)?);
        output.write_all(MAGIC)?;
        output.write_all(&(self.entries.len() as u64).to_le_bytes())?;
        output.write_all(&self.epoch.to_le_bytes())?;
        output.write_all(&attribution_len.to_le_bytes())?;
        output.write_all(attribution)?;
        let header_len = 26 + attribution.len();
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
        output.flush()
    }
}

/// Read-only, memory-mapped exact lookup over stellar catalog identifiers.
pub struct StarIdentifierCatalog {
    map: memmap2::Mmap,
    epoch: f64,
    attribution: String,
    entries_offset: usize,
    entry_count: usize,
}

impl StarIdentifierCatalog {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        // Safety: the file is opened read-only; concurrent truncation has the
        // same constraints as the existing memory-mapped star tile catalog.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        if map.len() < 26 || &map[..8] != MAGIC {
            return Err(invalid_data("not a seiza star identifier catalog"));
        }
        let entry_count_u64 = u64::from_le_bytes(map[8..16].try_into().unwrap());
        let entry_count = usize::try_from(entry_count_u64)
            .map_err(|_| invalid_data("star identifier count does not fit this platform"))?;
        let epoch = f64::from_le_bytes(map[16..24].try_into().unwrap());
        if !epoch.is_finite() {
            return Err(invalid_data("star identifier epoch is not finite"));
        }
        let attribution_len = u16::from_le_bytes(map[24..26].try_into().unwrap()) as usize;
        let attribution_end = 26usize
            .checked_add(attribution_len)
            .ok_or_else(|| invalid_data("invalid star identifier header"))?;
        let entries_offset = attribution_end.next_multiple_of(8);
        let entries_bytes = entry_count
            .checked_mul(RECORD_SIZE)
            .ok_or_else(|| invalid_data("star identifier catalog is too large"))?;
        let expected_len = entries_offset
            .checked_add(entries_bytes)
            .ok_or_else(|| invalid_data("star identifier catalog is too large"))?;
        if attribution_end > map.len() || expected_len != map.len() {
            return Err(invalid_data("star identifier file length is inconsistent"));
        }
        let attribution = std::str::from_utf8(&map[26..attribution_end])
            .map_err(|_| invalid_data("star identifier attribution is not UTF-8"))?
            .to_string();

        let catalog = Self {
            map,
            epoch,
            attribution,
            entries_offset,
            entry_count,
        };
        let mut previous = None;
        for index in 0..catalog.entry_count {
            let entry = catalog.packed_entry(index);
            if StarIdentifier::from_encoded(entry.namespace, entry.value).is_none() {
                return Err(invalid_data("invalid star identifier record"));
            }
            if previous.is_some_and(|key| key > entry.sort_key()) {
                return Err(invalid_data("star identifier records are not sorted"));
            }
            previous = Some(entry.sort_key());
        }
        Ok(catalog)
    }

    pub fn len(&self) -> usize {
        self.entry_count
    }

    pub fn is_empty(&self) -> bool {
        self.entry_count == 0
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
        let mut high = self.entry_count;
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
        while index < self.entry_count {
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

    fn packed_entry(&self, index: usize) -> PackedEntry {
        let start = self.entries_offset + index * RECORD_SIZE;
        let record = &self.map[start..start + RECORD_SIZE];
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
        builder.write_to(&path).unwrap();

        let catalog = StarIdentifierCatalog::open(&path).unwrap();
        assert_eq!(catalog.len(), 3);
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

        std::fs::remove_dir_all(&dir).ok();
    }
}
