//! Memory-mapped `SEIZAOB3` object catalog storage.
//!
//! The public object API lives in `objects.rs`. This module owns the fixed
//! binary layout, its writer, and demand-paged readers.

use crate::objects::{ObjectMetadata, ObjectNameMatch, SkyObject};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

pub(crate) const MAGIC: &[u8; 8] = b"SEIZAOB3";
const HEADER_SIZE: usize = 104;
const RECORD_SIZE: usize = 84;
const TILE_INDEX_SIZE: usize = 8;
const CANDIDATE_SIZE: usize = 4;
const NAME_INDEX_SIZE: usize = 16;
const STRING_REF_SIZE: usize = 6;
const DEFAULT_BANDS: u32 = 180;

#[derive(Debug, Clone, Copy)]
struct StringRef {
    offset: u32,
    len: u16,
}

impl StringRef {
    const EMPTY: Self = Self { offset: 0, len: 0 };
}

#[derive(Debug, Clone, Copy)]
struct ListRef {
    start: u32,
    count: u16,
}

#[derive(Debug)]
struct PackedRecord {
    kind: u8,
    ra: f64,
    dec: f64,
    mag: f32,
    major: f32,
    minor: f32,
    position_angle: f32,
    strings: [StringRef; 4],
    lists: [ListRef; 4],
}

#[derive(Debug)]
struct BuildName {
    key: String,
    designation: StringRef,
    object_index: u32,
}

#[derive(Debug, Clone, Copy)]
struct PackedName {
    key: StringRef,
    designation: StringRef,
    object_index: u32,
}

#[derive(Debug, Default)]
struct StringTable {
    bytes: Vec<u8>,
    refs: HashMap<String, StringRef>,
}

impl StringTable {
    fn intern(&mut self, value: &str) -> io::Result<StringRef> {
        if value.is_empty() {
            return Ok(StringRef::EMPTY);
        }
        if let Some(reference) = self.refs.get(value) {
            return Ok(*reference);
        }
        let len = u16::try_from(value.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "object catalog string exceeds 65535 bytes",
            )
        })?;
        let offset = u32::try_from(self.bytes.len()).map_err(|_| catalog_too_large())?;
        self.bytes.extend_from_slice(value.as_bytes());
        let reference = StringRef { offset, len };
        self.refs.insert(value.to_string(), reference);
        Ok(reference)
    }
}

#[derive(Debug, Clone)]
struct Grid {
    n_bands: u32,
    bins: Vec<u32>,
    offsets: Vec<u32>,
}

impl Grid {
    fn new(n_bands: u32) -> Self {
        let band_height = 180.0 / n_bands as f64;
        let mut bins = Vec::with_capacity(n_bands as usize);
        let mut offsets = Vec::with_capacity(n_bands as usize);
        let mut total = 0u32;
        for band in 0..n_bands {
            let dec_mid = -90.0 + (band as f64 + 0.5) * band_height;
            let circumference = 360.0 * dec_mid.to_radians().cos().max(1e-6);
            let count = (circumference / band_height).ceil().max(1.0) as u32;
            offsets.push(total);
            bins.push(count);
            total += count;
        }
        Self {
            n_bands,
            bins,
            offsets,
        }
    }

    fn n_tiles(&self) -> u32 {
        self.offsets[self.n_bands as usize - 1] + self.bins[self.n_bands as usize - 1]
    }

    fn band_of(&self, dec: f64) -> u32 {
        let band = ((dec + 90.0) / 180.0 * self.n_bands as f64) as i64;
        band.clamp(0, self.n_bands as i64 - 1) as u32
    }

    fn tile_of(&self, ra: f64, dec: f64) -> u32 {
        let band = self.band_of(dec);
        let count = self.bins[band as usize];
        let bin = ((ra.rem_euclid(360.0) / 360.0 * count as f64) as u32).min(count - 1);
        self.offsets[band as usize] + bin
    }

    /// Return a conservative set of tiles intersecting a spherical cap.
    fn cone_tiles(&self, ra: f64, dec: f64, radius_deg: f64) -> Vec<u32> {
        if radius_deg >= 180.0 {
            return (0..self.n_tiles()).collect();
        }
        let dec_lo = (dec - radius_deg).max(-90.0);
        let dec_hi = (dec + radius_deg).min(90.0);
        let band_lo = self.band_of(dec_lo);
        let band_hi = self.band_of(dec_hi);
        let mut tiles = Vec::new();

        // Maximum longitude displacement anywhere in a spherical cap. The
        // previous radius/cos(dec-band-edge) approximation is not conservative
        // near the poles: for example, a 9-degree cap at Dec -80 reaches more
        // than 50 degrees in RA. When the cap contains a pole, every longitude
        // is possible; otherwise this is the exact tangent longitude.
        let ra_radius = if dec.abs() + radius_deg >= 90.0 {
            180.0
        } else {
            (radius_deg.to_radians().sin() / dec.to_radians().cos())
                .clamp(-1.0, 1.0)
                .asin()
                .abs()
                .to_degrees()
        };

        for band in band_lo..=band_hi {
            let count = self.bins[band as usize];
            let offset = self.offsets[band as usize];
            if ra_radius >= 180.0 {
                tiles.extend(offset..offset + count);
                continue;
            }
            let bin_width = 360.0 / count as f64;
            let start = ((ra - ra_radius).rem_euclid(360.0) / bin_width) as u32 % count;
            let span = (2.0 * ra_radius / bin_width).ceil() as u32 + 1;
            if span >= count {
                tiles.extend(offset..offset + count);
            } else {
                for index in 0..span {
                    tiles.push(offset + (start + index) % count);
                }
            }
        }
        tiles
    }
}

/// Stable canonical-record order used by v4 for spatial page locality. The
/// tile index still carries overlap entries for large objects; this is only the
/// primary-center order of the single canonical record.
pub(crate) fn spatial_order(objects: &[SkyObject]) -> Vec<usize> {
    let grid = Grid::new(DEFAULT_BANDS);
    let mut order = (0..objects.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        let tile = |object: &SkyObject| {
            if object.ra.is_finite()
                && object.dec.is_finite()
                && (-90.0..=90.0).contains(&object.dec)
            {
                grid.tile_of(object.ra, object.dec)
            } else {
                u32::MAX
            }
        };
        tile(&objects[left])
            .cmp(&tile(&objects[right]))
            .then_with(|| objects[left].metadata.id.cmp(&objects[right].metadata.id))
            .then(left.cmp(&right))
    });
    order
}

#[derive(Debug)]
struct Layout {
    records_offset: usize,
    tile_index_offset: usize,
    candidates_offset: usize,
    names_offset: usize,
    list_refs_offset: usize,
    strings_offset: usize,
    file_size: usize,
}

impl Layout {
    fn calculate(
        record_count: usize,
        tile_count: usize,
        candidate_count: usize,
        name_count: usize,
        list_ref_count: usize,
        string_bytes: usize,
    ) -> io::Result<Self> {
        let records_offset = HEADER_SIZE;
        let tile_index_offset = section_end(records_offset, record_count, RECORD_SIZE)?;
        let candidates_offset = section_end(tile_index_offset, tile_count, TILE_INDEX_SIZE)?;
        let names_offset = section_end(candidates_offset, candidate_count, CANDIDATE_SIZE)?;
        let list_refs_offset = section_end(names_offset, name_count, NAME_INDEX_SIZE)?;
        let strings_offset = section_end(list_refs_offset, list_ref_count, STRING_REF_SIZE)?;
        let file_size = strings_offset
            .checked_add(string_bytes)
            .ok_or_else(catalog_too_large)?;
        Ok(Self {
            records_offset,
            tile_index_offset,
            candidates_offset,
            names_offset,
            list_refs_offset,
            strings_offset,
            file_size,
        })
    }
}

fn section_end(offset: usize, count: usize, size: usize) -> io::Result<usize> {
    let bytes = count.checked_mul(size).ok_or_else(catalog_too_large)?;
    offset
        .checked_add(bytes)
        .ok_or_else(catalog_too_large)
        .map(|end| end.next_multiple_of(8))
}

/// Write a new memory-mapped object catalog.
pub(crate) fn write(path: &Path, objects: &[SkyObject]) -> io::Result<()> {
    let bytes = encode(objects)?;
    let mut output = BufWriter::new(File::create(path)?);
    output.write_all(&bytes)?;
    output.flush()
}

/// Encode the complete v3 catalog. V4 embeds this proven demand-paged query
/// representation as one independently versioned hot section.
pub(crate) fn encode(objects: &[SkyObject]) -> io::Result<Vec<u8>> {
    let record_count = u32::try_from(objects.len()).map_err(|_| catalog_too_large())?;
    let grid = Grid::new(DEFAULT_BANDS);
    let mut strings = StringTable::default();
    let mut list_refs = Vec::<StringRef>::new();
    let mut records = Vec::with_capacity(objects.len());
    let mut tiles = vec![Vec::<u32>::new(); grid.n_tiles() as usize];
    let mut build_names = Vec::<BuildName>::new();

    for (index, object) in objects.iter().enumerate() {
        let object_index = u32::try_from(index).map_err(|_| catalog_too_large())?;
        let scalar_strings = [
            strings.intern(&object.name)?,
            strings.intern(&object.common_name)?,
            strings.intern(&object.metadata.id)?,
            strings.intern(&object.metadata.source)?,
        ];
        let lists = [
            pack_list(&mut strings, &mut list_refs, &object.metadata.aliases)?,
            pack_list(&mut strings, &mut list_refs, &object.metadata.parent_ids)?,
            pack_list(&mut strings, &mut list_refs, &object.metadata.alternate_ids)?,
            pack_list(
                &mut strings,
                &mut list_refs,
                &object.metadata.alternate_sources,
            )?,
        ];
        records.push(PackedRecord {
            kind: object.kind as u8,
            ra: object.ra,
            dec: object.dec,
            mag: object.mag.unwrap_or(f32::NAN),
            major: object.major_arcmin.unwrap_or(f32::NAN),
            minor: object.minor_arcmin.unwrap_or(f32::NAN),
            position_angle: object.position_angle_deg.unwrap_or(f32::NAN),
            strings: scalar_strings,
            lists,
        });

        let mut object_tiles = if object.ra.is_finite()
            && object.dec.is_finite()
            && (-90.0..=90.0).contains(&object.dec)
        {
            let radius = object
                .major_arcmin
                .filter(|value| value.is_finite() && *value > 0.0)
                .map(|value| value as f64 / 120.0)
                .unwrap_or(0.0);
            if radius > 0.0 {
                grid.cone_tiles(object.ra, object.dec, radius.min(180.0))
            } else {
                vec![grid.tile_of(object.ra, object.dec)]
            }
        } else {
            // Preserve invalid records for explicit validation without making
            // normal open scan catalog contents.
            vec![0]
        };
        object_tiles.sort_unstable();
        object_tiles.dedup();
        for tile in object_tiles {
            tiles[tile as usize].push(object_index);
        }

        let mut seen_keys = HashSet::new();
        for designation in object_designations(object) {
            let key = normalize_name(designation);
            if key.is_empty() || !seen_keys.insert(key.clone()) {
                continue;
            }
            build_names.push(BuildName {
                key,
                designation: strings.intern(designation)?,
                object_index,
            });
        }
    }

    build_names.sort_by(|left, right| {
        left.key
            .cmp(&right.key)
            .then(left.object_index.cmp(&right.object_index))
    });
    let mut names = Vec::with_capacity(build_names.len());
    for name in build_names {
        names.push(PackedName {
            key: strings.intern(&name.key)?,
            designation: name.designation,
            object_index: name.object_index,
        });
    }

    let candidate_count_usize = tiles.iter().try_fold(0usize, |total, tile| {
        total.checked_add(tile.len()).ok_or_else(catalog_too_large)
    })?;
    let candidate_count = u32::try_from(candidate_count_usize).map_err(|_| catalog_too_large())?;
    let name_count = u32::try_from(names.len()).map_err(|_| catalog_too_large())?;
    let list_ref_count = u32::try_from(list_refs.len()).map_err(|_| catalog_too_large())?;
    let string_bytes = strings.bytes.len();
    let layout = Layout::calculate(
        objects.len(),
        tiles.len(),
        candidate_count_usize,
        names.len(),
        list_refs.len(),
        string_bytes,
    )?;

    let mut output = Vec::with_capacity(layout.file_size);
    output.write_all(MAGIC)?;
    output.write_all(&(HEADER_SIZE as u32).to_le_bytes())?;
    output.write_all(&DEFAULT_BANDS.to_le_bytes())?;
    output.write_all(&record_count.to_le_bytes())?;
    output.write_all(&grid.n_tiles().to_le_bytes())?;
    output.write_all(&candidate_count.to_le_bytes())?;
    output.write_all(&name_count.to_le_bytes())?;
    output.write_all(&list_ref_count.to_le_bytes())?;
    output.write_all(&0u32.to_le_bytes())?;
    output.write_all(&(string_bytes as u64).to_le_bytes())?;
    for offset in [
        layout.records_offset,
        layout.tile_index_offset,
        layout.candidates_offset,
        layout.names_offset,
        layout.list_refs_offset,
        layout.strings_offset,
        layout.file_size,
    ] {
        output.write_all(&(offset as u64).to_le_bytes())?;
    }

    for record in &records {
        write_record(&mut output, record)?;
    }
    write_padding(
        &mut output,
        HEADER_SIZE + records.len() * RECORD_SIZE,
        layout.tile_index_offset,
    )?;

    let mut candidate_start = 0u32;
    for tile in &tiles {
        let count = u32::try_from(tile.len()).map_err(|_| catalog_too_large())?;
        output.write_all(&candidate_start.to_le_bytes())?;
        output.write_all(&count.to_le_bytes())?;
        candidate_start = candidate_start
            .checked_add(count)
            .ok_or_else(catalog_too_large)?;
    }
    write_padding(
        &mut output,
        layout.tile_index_offset + tiles.len() * TILE_INDEX_SIZE,
        layout.candidates_offset,
    )?;
    for tile in &tiles {
        for object_index in tile {
            output.write_all(&object_index.to_le_bytes())?;
        }
    }
    write_padding(
        &mut output,
        layout.candidates_offset + candidate_count_usize * CANDIDATE_SIZE,
        layout.names_offset,
    )?;
    for name in &names {
        write_string_ref(&mut output, name.key)?;
        write_string_ref(&mut output, name.designation)?;
        output.write_all(&name.object_index.to_le_bytes())?;
    }
    write_padding(
        &mut output,
        layout.names_offset + names.len() * NAME_INDEX_SIZE,
        layout.list_refs_offset,
    )?;
    for reference in &list_refs {
        write_string_ref(&mut output, *reference)?;
    }
    write_padding(
        &mut output,
        layout.list_refs_offset + list_refs.len() * STRING_REF_SIZE,
        layout.strings_offset,
    )?;
    output.write_all(&strings.bytes)?;
    if output.len() != layout.file_size {
        return Err(catalog_too_large());
    }
    Ok(output)
}

fn pack_list(
    strings: &mut StringTable,
    list_refs: &mut Vec<StringRef>,
    values: &[String],
) -> io::Result<ListRef> {
    let start = u32::try_from(list_refs.len()).map_err(|_| catalog_too_large())?;
    let count = u16::try_from(values.len()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "object catalog string list exceeds 65535 entries",
        )
    })?;
    for value in values {
        list_refs.push(strings.intern(value)?);
    }
    Ok(ListRef { start, count })
}

fn object_designations(object: &SkyObject) -> impl Iterator<Item = &str> {
    std::iter::once(object.name.as_str())
        .chain(std::iter::once(object.common_name.as_str()))
        .chain(std::iter::once(object.metadata.id.as_str()))
        .chain(object.metadata.aliases.iter().map(String::as_str))
        .chain(object.metadata.alternate_ids.iter().map(String::as_str))
}

fn write_record(output: &mut impl Write, record: &PackedRecord) -> io::Result<()> {
    output.write_all(&[record.kind, 0, 0, 0])?;
    output.write_all(&record.ra.to_le_bytes())?;
    output.write_all(&record.dec.to_le_bytes())?;
    output.write_all(&record.mag.to_le_bytes())?;
    output.write_all(&record.major.to_le_bytes())?;
    output.write_all(&record.minor.to_le_bytes())?;
    output.write_all(&record.position_angle.to_le_bytes())?;
    for reference in record.strings {
        write_string_ref(output, reference)?;
    }
    for list in record.lists {
        output.write_all(&list.start.to_le_bytes())?;
        output.write_all(&list.count.to_le_bytes())?;
    }
    Ok(())
}

fn write_string_ref(output: &mut impl Write, reference: StringRef) -> io::Result<()> {
    output.write_all(&reference.offset.to_le_bytes())?;
    output.write_all(&reference.len.to_le_bytes())
}

fn write_padding(output: &mut impl Write, current: usize, target: usize) -> io::Result<()> {
    if target < current {
        return Err(catalog_too_large());
    }
    output.write_all(&vec![0; target - current])
}

/// Read-only mmap view. Opening validates only the header and section bounds.
pub(crate) struct MappedObjectCatalog {
    map: Arc<memmap2::Mmap>,
    base: usize,
    section_len: usize,
    grid: Grid,
    record_count: usize,
    tile_count: usize,
    candidate_count: usize,
    name_count: usize,
    list_ref_count: usize,
    layout: Layout,
}

impl std::fmt::Debug for MappedObjectCatalog {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MappedObjectCatalog")
            .field("record_count", &self.record_count)
            .field("tile_count", &self.tile_count)
            .field("candidate_count", &self.candidate_count)
            .field("name_count", &self.name_count)
            .finish_non_exhaustive()
    }
}

impl MappedObjectCatalog {
    pub(crate) fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        // Safety: the catalog is mapped read-only and retained for the lifetime
        // of every borrowed byte slice.
        let map = Arc::new(unsafe { memmap2::Mmap::map(&file)? });
        let section_len = map.len();
        Self::from_mmap(map, 0, section_len)
    }

    pub(crate) fn from_mmap(
        map: Arc<memmap2::Mmap>,
        base: usize,
        section_len: usize,
    ) -> io::Result<Self> {
        if base > map.len() || section_len > map.len() - base {
            return Err(invalid_data("v3 object catalog section is out of bounds"));
        }
        let bytes = &map[base..base + section_len];
        if bytes.len() < HEADER_SIZE || &bytes[..8] != MAGIC {
            return Err(invalid_data("not a seiza v3 object catalog"));
        }
        let header_size = read_u32_at(bytes, 8)? as usize;
        let n_bands = read_u32_at(bytes, 12)?;
        if header_size != HEADER_SIZE || n_bands == 0 || n_bands > 4096 {
            return Err(invalid_data("invalid object catalog header"));
        }
        let record_count = read_count(bytes, 16, "object")?;
        let tile_count = read_count(bytes, 20, "tile")?;
        let candidate_count = read_count(bytes, 24, "tile candidate")?;
        let name_count = read_count(bytes, 28, "name")?;
        let list_ref_count = read_count(bytes, 32, "list string")?;
        let string_bytes = read_u64_at(bytes, 40)?;
        let string_bytes = usize::try_from(string_bytes)
            .map_err(|_| invalid_data("object string table is too large"))?;
        let layout = Layout::calculate(
            record_count,
            tile_count,
            candidate_count,
            name_count,
            list_ref_count,
            string_bytes,
        )?;
        let stored_offsets = [
            read_u64_at(bytes, 48)?,
            read_u64_at(bytes, 56)?,
            read_u64_at(bytes, 64)?,
            read_u64_at(bytes, 72)?,
            read_u64_at(bytes, 80)?,
            read_u64_at(bytes, 88)?,
            read_u64_at(bytes, 96)?,
        ];
        let expected_offsets = [
            layout.records_offset,
            layout.tile_index_offset,
            layout.candidates_offset,
            layout.names_offset,
            layout.list_refs_offset,
            layout.strings_offset,
            layout.file_size,
        ];
        if stored_offsets
            .iter()
            .zip(expected_offsets)
            .any(|(&stored, expected)| usize::try_from(stored).ok() != Some(expected))
            || layout.file_size != bytes.len()
        {
            return Err(invalid_data(
                "object catalog section bounds are inconsistent",
            ));
        }
        let grid = Grid::new(n_bands);
        if grid.n_tiles() as usize != tile_count {
            return Err(invalid_data("object catalog tile count is inconsistent"));
        }
        Ok(Self {
            map,
            base,
            section_len,
            grid,
            record_count,
            tile_count,
            candidate_count,
            name_count,
            list_ref_count,
            layout,
        })
    }

    fn bytes(&self) -> &[u8] {
        &self.map[self.base..self.base + self.section_len]
    }

    pub(crate) fn len(&self) -> usize {
        self.record_count
    }

    pub(crate) fn object(&self, index: usize) -> io::Result<SkyObject> {
        if index >= self.record_count {
            return Err(invalid_data("object record index is out of bounds"));
        }
        let start = self.layout.records_offset + index * RECORD_SIZE;
        let record = &self.bytes()[start..start + RECORD_SIZE];
        if record[1..4] != [0, 0, 0] {
            return Err(invalid_data("object record reserved bytes are nonzero"));
        }
        let scalar = |offset| self.string_from_ref(read_string_ref(record, offset));
        Ok(SkyObject {
            kind: crate::objects::ObjectKind::from_u8(record[0]),
            ra: f64::from_le_bytes(record[4..12].try_into().unwrap()),
            dec: f64::from_le_bytes(record[12..20].try_into().unwrap()),
            mag: optional_f32(record, 20),
            major_arcmin: optional_f32(record, 24),
            minor_arcmin: optional_f32(record, 28),
            position_angle_deg: optional_f32(record, 32),
            name: scalar(36)?.to_string(),
            common_name: scalar(42)?.to_string(),
            metadata: ObjectMetadata {
                id: scalar(48)?.to_string(),
                source: scalar(54)?.to_string(),
                aliases: self.string_list(read_list_ref(record, 60))?,
                parent_ids: self.string_list(read_list_ref(record, 66))?,
                alternate_ids: self.string_list(read_list_ref(record, 72))?,
                alternate_sources: self.string_list(read_list_ref(record, 78))?,
            },
        })
    }

    pub(crate) fn read_all(&self) -> io::Result<Vec<SkyObject>> {
        (0..self.record_count)
            .map(|index| self.object(index))
            .collect()
    }

    /// Return record indices from tiles covering a query cap. Candidate lists
    /// include every tile touched by each object's conservative extent.
    pub(crate) fn candidates(&self, ra: f64, dec: f64, radius_deg: f64) -> io::Result<Vec<u32>> {
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        for tile in self.grid.cone_tiles(ra, dec, radius_deg) {
            let (start, count) = self.tile_range(tile as usize)?;
            for candidate_index in start..start + count {
                let offset = self.layout.candidates_offset + candidate_index * CANDIDATE_SIZE;
                let object_index = read_u32_at(self.bytes(), offset)?;
                let index = object_index as usize;
                if index >= self.record_count {
                    return Err(invalid_data("object tile candidate is out of bounds"));
                }
                if seen.insert(object_index) {
                    result.push(object_index);
                }
            }
        }
        Ok(result)
    }

    pub(crate) fn lookup_name(&self, designation: &str) -> io::Result<Vec<ObjectNameMatch>> {
        let key = normalize_name(designation);
        if key.is_empty() {
            return Ok(Vec::new());
        }
        let mut index = self.lower_bound_name(&key)?;
        let mut matches = Vec::new();
        while index < self.name_count {
            let entry = self.name_entry(index)?;
            if self.string_from_ref(entry.key)? != key {
                break;
            }
            matches.push(ObjectNameMatch {
                object: self.object(entry.object_index as usize)?,
                matched_name: self.string_from_ref(entry.designation)?.to_string(),
            });
            index += 1;
        }
        Ok(matches)
    }

    /// Resolve exact-name index entries to canonical ordinals without scanning
    /// the object-core section. Used by v4 cold-detail lookup.
    pub(crate) fn lookup_name_indices(&self, designation: &str) -> io::Result<Vec<u32>> {
        let key = normalize_name(designation);
        if key.is_empty() {
            return Ok(Vec::new());
        }
        let mut index = self.lower_bound_name(&key)?;
        let mut matches = Vec::new();
        while index < self.name_count {
            let entry = self.name_entry(index)?;
            if self.string_from_ref(entry.key)? != key {
                break;
            }
            if matches.last().copied() != Some(entry.object_index) {
                matches.push(entry.object_index);
            }
            index += 1;
        }
        Ok(matches)
    }

    pub(crate) fn search_names(
        &self,
        prefix: &str,
        limit: usize,
    ) -> io::Result<Vec<ObjectNameMatch>> {
        let key = normalize_name(prefix);
        if key.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let mut index = self.lower_bound_name(&key)?;
        let mut matches = Vec::new();
        while index < self.name_count && matches.len() < limit {
            let entry = self.name_entry(index)?;
            if !self.string_from_ref(entry.key)?.starts_with(&key) {
                break;
            }
            matches.push(ObjectNameMatch {
                object: self.object(entry.object_index as usize)?,
                matched_name: self.string_from_ref(entry.designation)?.to_string(),
            });
            index += 1;
        }
        Ok(matches)
    }

    /// Exhaustive file validation, intentionally separate from open.
    pub(crate) fn validate(&self) -> io::Result<()> {
        std::str::from_utf8(&self.bytes()[self.layout.strings_offset..])
            .map_err(|_| invalid_data("object string table is not UTF-8"))?;

        for index in 0..self.list_ref_count {
            self.string_from_ref(self.list_string_ref(index)?)?;
        }
        for index in 0..self.record_count {
            let start = self.layout.records_offset + index * RECORD_SIZE;
            if self.bytes()[start] > 13 {
                return Err(invalid_data("object record kind is invalid"));
            }
            self.object(index)?;
        }

        let mut referenced = vec![false; self.record_count];
        let mut previous_end = 0usize;
        for tile in 0..self.tile_count {
            let (start, count) = self.tile_range(tile)?;
            if start != previous_end || start + count > self.candidate_count {
                return Err(invalid_data(
                    "object tile candidate ranges are inconsistent",
                ));
            }
            let mut previous_object = None;
            for candidate in start..start + count {
                let offset = self.layout.candidates_offset + candidate * CANDIDATE_SIZE;
                let object = read_u32_at(self.bytes(), offset)? as usize;
                if object >= self.record_count
                    || previous_object.is_some_and(|previous| previous >= object)
                {
                    return Err(invalid_data(
                        "object tile candidates are invalid or unsorted",
                    ));
                }
                referenced[object] = true;
                previous_object = Some(object);
            }
            previous_end = start + count;
        }
        if previous_end != self.candidate_count || referenced.iter().any(|value| !*value) {
            return Err(invalid_data(
                "object tile index does not cover every record",
            ));
        }

        let mut previous_key: Option<&str> = None;
        for index in 0..self.name_count {
            let entry = self.name_entry(index)?;
            let key = self.string_from_ref(entry.key)?;
            let designation = self.string_from_ref(entry.designation)?;
            if key.is_empty()
                || designation.is_empty()
                || entry.object_index as usize >= self.record_count
                || previous_key.is_some_and(|previous| previous > key)
            {
                return Err(invalid_data("object name index is invalid or unsorted"));
            }
            previous_key = Some(key);
        }
        Ok(())
    }

    fn tile_range(&self, tile: usize) -> io::Result<(usize, usize)> {
        if tile >= self.tile_count {
            return Err(invalid_data("object tile index is out of bounds"));
        }
        let offset = self.layout.tile_index_offset + tile * TILE_INDEX_SIZE;
        let start = read_u32_at(self.bytes(), offset)? as usize;
        let count = read_u32_at(self.bytes(), offset + 4)? as usize;
        if start > self.candidate_count || count > self.candidate_count - start {
            return Err(invalid_data("object tile candidate range is out of bounds"));
        }
        Ok((start, count))
    }

    fn string_list(&self, list: ListRef) -> io::Result<Vec<String>> {
        let start = list.start as usize;
        let count = list.count as usize;
        if start > self.list_ref_count || count > self.list_ref_count - start {
            return Err(invalid_data("object string list is out of bounds"));
        }
        (start..start + count)
            .map(|index| {
                self.string_from_ref(self.list_string_ref(index)?)
                    .map(str::to_string)
            })
            .collect()
    }

    fn list_string_ref(&self, index: usize) -> io::Result<StringRef> {
        if index >= self.list_ref_count {
            return Err(invalid_data("object list string index is out of bounds"));
        }
        let offset = self.layout.list_refs_offset + index * STRING_REF_SIZE;
        Ok(read_string_ref(self.bytes(), offset))
    }

    fn string_from_ref(&self, reference: StringRef) -> io::Result<&str> {
        let start = reference.offset as usize;
        let count = reference.len as usize;
        let string_bytes = self.section_len - self.layout.strings_offset;
        if start > string_bytes || count > string_bytes - start {
            return Err(invalid_data("object string reference is out of bounds"));
        }
        let start = self.layout.strings_offset + start;
        std::str::from_utf8(&self.bytes()[start..start + count])
            .map_err(|_| invalid_data("object string is not UTF-8"))
    }

    fn name_entry(&self, index: usize) -> io::Result<PackedName> {
        if index >= self.name_count {
            return Err(invalid_data("object name index is out of bounds"));
        }
        let offset = self.layout.names_offset + index * NAME_INDEX_SIZE;
        Ok(PackedName {
            key: read_string_ref(self.bytes(), offset),
            designation: read_string_ref(self.bytes(), offset + 6),
            object_index: read_u32_at(self.bytes(), offset + 12)?,
        })
    }

    fn lower_bound_name(&self, key: &str) -> io::Result<usize> {
        let mut low = 0usize;
        let mut high = self.name_count;
        while low < high {
            let middle = low + (high - low) / 2;
            let entry = self.name_entry(middle)?;
            if self.string_from_ref(entry.key)? < key {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        Ok(low)
    }
}

fn read_count(bytes: &[u8], offset: usize, label: &str) -> io::Result<usize> {
    usize::try_from(read_u32_at(bytes, offset)?)
        .map_err(|_| invalid_data_owned(format!("{label} count is too large")))
}

fn read_u32_at(bytes: &[u8], offset: usize) -> io::Result<u32> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| invalid_data("object catalog offset overflow"))?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| invalid_data("truncated object catalog"))?;
    Ok(u32::from_le_bytes(slice.try_into().unwrap()))
}

fn read_u64_at(bytes: &[u8], offset: usize) -> io::Result<u64> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| invalid_data("object catalog offset overflow"))?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| invalid_data("truncated object catalog"))?;
    Ok(u64::from_le_bytes(slice.try_into().unwrap()))
}

fn read_string_ref(bytes: &[u8], offset: usize) -> StringRef {
    StringRef {
        offset: u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()),
        len: u16::from_le_bytes(bytes[offset + 4..offset + 6].try_into().unwrap()),
    }
}

fn read_list_ref(bytes: &[u8], offset: usize) -> ListRef {
    ListRef {
        start: u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()),
        count: u16::from_le_bytes(bytes[offset + 4..offset + 6].try_into().unwrap()),
    }
}

fn optional_f32(bytes: &[u8], offset: usize) -> Option<f32> {
    let value = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    (!value.is_nan()).then_some(value)
}

pub(crate) fn normalize_name(value: &str) -> String {
    let compact: String = value
        .chars()
        .filter(|character| character.is_alphanumeric())
        .flat_map(char::to_uppercase)
        .collect();
    if compact.is_empty() {
        return compact;
    }
    let number_start = if compact.starts_with("SH2") {
        3
    } else {
        compact
            .char_indices()
            .find_map(|(index, character)| character.is_ascii_digit().then_some(index))
            .unwrap_or(compact.len())
    };
    if number_start == compact.len() {
        return compact;
    }
    let number_end = compact[number_start..]
        .char_indices()
        .find_map(|(index, character)| {
            (!character.is_ascii_digit()).then_some(number_start + index)
        })
        .unwrap_or(compact.len());
    let number = compact[number_start..number_end].trim_start_matches('0');
    let number = if number.is_empty() { "0" } else { number };
    format!(
        "{}{}{}",
        &compact[..number_start],
        number,
        &compact[number_end..]
    )
}

fn catalog_too_large() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "object catalog is too large for the v3 format",
    )
}

fn invalid_data(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn invalid_data_owned(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
