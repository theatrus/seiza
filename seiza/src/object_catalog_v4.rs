//! Extensible, sectioned v4 object-catalog container.
//!
//! The canonical query representation is an embedded v3 section so its mmap
//! behavior and mature indices remain unchanged. Source-qualified details and
//! provenance are independently versioned sections. New optional sections can
//! be added without changing the container envelope.

use crate::object_catalog_v3;
use crate::objects::{
    GeometryData, GeometryQuality, GeometryRole, ObjectCatalogCapabilities, ObjectCatalogData,
    ObjectCatalogProvenance, ObjectContour, ObjectDetails, ObjectGeometry, ObjectNameMatch,
    ObjectRelation, ObjectSelection, ObjectSourceRecord, SkyObject,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

pub(crate) const MAGIC: &[u8; 8] = b"SEIZAOB\0";
const CONTAINER_MAJOR: u16 = 1;
const CONTAINER_MINOR: u16 = 0;
const HEADER_SIZE: usize = 64;
const DIRECTORY_ENTRY_SIZE: usize = 96;
const FLAG_REQUIRED: u32 = 1;
const DETAIL_INDEX_STRIDE: u32 = 16;

const CANONICAL_V3: [u8; 16] = *b"CANONICAL_V3\0\0\0\0";
const DETAIL_INDEX: [u8; 16] = *b"DETAIL_INDEX\0\0\0\0";
const DETAIL_DATA: [u8; 16] = *b"DETAIL_DATA\0\0\0\0\0";
const PROVENANCE_JSON: [u8; 16] = *b"PROVENANCE_JSON\0";
const CAPABILITIES: [u8; 16] = *b"CAPABILITIES\0\0\0\0";
const OUTLINE_VERTICES: [u8; 16] = *b"OUTLINE_VERTICES";

const CAP_SOURCE_RECORDS: u64 = 1 << 0;
const CAP_RELATIONS: u64 = 1 << 1;
const CAP_SELECTIONS: u64 = 1 << 2;
const CAP_ELLIPSES: u64 = 1 << 3;
const CAP_OUTLINES: u64 = 1 << 4;
const CAP_PROVENANCE: u64 = 1 << 5;

#[derive(Clone, Debug)]
struct SectionEntry {
    kind: [u8; 16],
    schema_major: u16,
    schema_minor: u16,
    flags: u32,
    instance: u32,
    offset: usize,
    len: usize,
    record_count: u64,
    record_stride: u32,
    checksum: [u8; 32],
}

#[derive(Debug)]
struct BuildSection {
    kind: [u8; 16],
    schema_major: u16,
    schema_minor: u16,
    flags: u32,
    instance: u32,
    record_count: u64,
    record_stride: u32,
    bytes: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DiskObjectDetails {
    canonical_id: String,
    source_records: Vec<ObjectSourceRecord>,
    relations: Vec<ObjectRelation>,
    selections: Vec<ObjectSelection>,
    geometries: Vec<DiskObjectGeometry>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DiskObjectGeometry {
    id: String,
    source_record_id: String,
    role: GeometryRole,
    quality: GeometryQuality,
    method: String,
    evidence: String,
    data: DiskGeometryData,
}

#[derive(Debug, Serialize, Deserialize)]
enum DiskGeometryData {
    Point {
        ra_deg: f64,
        dec_deg: f64,
    },
    Ellipse {
        center_ra_deg: f64,
        center_dec_deg: f64,
        major_arcmin: f32,
        minor_arcmin: Option<f32>,
        position_angle_deg: Option<f32>,
    },
    OutlineSet {
        level: Option<String>,
        contours: Vec<DiskContour>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct DiskContour {
    closed: bool,
    vertex_start: u64,
    vertex_count: u32,
}

impl DiskObjectDetails {
    fn pack(detail: &ObjectDetails, vertices: &mut Vec<u8>) -> io::Result<Self> {
        let geometries = detail
            .geometries
            .iter()
            .map(|geometry| DiskObjectGeometry::pack(geometry, vertices))
            .collect::<io::Result<_>>()?;
        Ok(Self {
            canonical_id: detail.canonical_id.clone(),
            source_records: detail.source_records.clone(),
            relations: detail.relations.clone(),
            selections: detail.selections.clone(),
            geometries,
        })
    }

    fn unpack(self, vertex_bytes: &[u8]) -> io::Result<ObjectDetails> {
        let geometries = self
            .geometries
            .into_iter()
            .map(|geometry| geometry.unpack(vertex_bytes))
            .collect::<io::Result<_>>()?;
        Ok(ObjectDetails {
            canonical_id: self.canonical_id,
            source_records: self.source_records,
            relations: self.relations,
            selections: self.selections,
            geometries,
        })
    }
}

impl DiskObjectGeometry {
    fn pack(geometry: &ObjectGeometry, vertices: &mut Vec<u8>) -> io::Result<Self> {
        let data = match &geometry.data {
            GeometryData::Point { ra_deg, dec_deg } => DiskGeometryData::Point {
                ra_deg: *ra_deg,
                dec_deg: *dec_deg,
            },
            GeometryData::Ellipse {
                center_ra_deg,
                center_dec_deg,
                major_arcmin,
                minor_arcmin,
                position_angle_deg,
            } => DiskGeometryData::Ellipse {
                center_ra_deg: *center_ra_deg,
                center_dec_deg: *center_dec_deg,
                major_arcmin: *major_arcmin,
                minor_arcmin: *minor_arcmin,
                position_angle_deg: *position_angle_deg,
            },
            GeometryData::OutlineSet { level, contours } => {
                let mut packed = Vec::with_capacity(contours.len());
                for contour in contours {
                    let vertex_start =
                        u64::try_from(vertices.len() / 16).map_err(|_| catalog_too_large())?;
                    let vertex_count =
                        u32::try_from(contour.vertices.len()).map_err(|_| catalog_too_large())?;
                    for &(ra, dec) in &contour.vertices {
                        vertices.extend_from_slice(&ra.to_le_bytes());
                        vertices.extend_from_slice(&dec.to_le_bytes());
                    }
                    packed.push(DiskContour {
                        closed: contour.closed,
                        vertex_start,
                        vertex_count,
                    });
                }
                DiskGeometryData::OutlineSet {
                    level: level.clone(),
                    contours: packed,
                }
            }
        };
        Ok(Self {
            id: geometry.id.clone(),
            source_record_id: geometry.source_record_id.clone(),
            role: geometry.role,
            quality: geometry.quality,
            method: geometry.method.clone(),
            evidence: geometry.evidence.clone(),
            data,
        })
    }

    fn unpack(self, vertex_bytes: &[u8]) -> io::Result<ObjectGeometry> {
        let data = match self.data {
            DiskGeometryData::Point { ra_deg, dec_deg } => GeometryData::Point { ra_deg, dec_deg },
            DiskGeometryData::Ellipse {
                center_ra_deg,
                center_dec_deg,
                major_arcmin,
                minor_arcmin,
                position_angle_deg,
            } => GeometryData::Ellipse {
                center_ra_deg,
                center_dec_deg,
                major_arcmin,
                minor_arcmin,
                position_angle_deg,
            },
            DiskGeometryData::OutlineSet { level, contours } => {
                let mut unpacked = Vec::with_capacity(contours.len());
                for contour in contours {
                    let start = usize::try_from(contour.vertex_start)
                        .ok()
                        .and_then(|value| value.checked_mul(16))
                        .ok_or_else(|| invalid_data("outline vertex offset overflow"))?;
                    let len = contour.vertex_count as usize;
                    let byte_len = len
                        .checked_mul(16)
                        .ok_or_else(|| invalid_data("outline vertex length overflow"))?;
                    if start > vertex_bytes.len() || byte_len > vertex_bytes.len() - start {
                        return Err(invalid_data("outline vertex range is out of bounds"));
                    }
                    let mut vertices = Vec::with_capacity(len);
                    for vertex in vertex_bytes[start..start + byte_len].chunks_exact(16) {
                        vertices.push((
                            f64::from_le_bytes(vertex[..8].try_into().unwrap()),
                            f64::from_le_bytes(vertex[8..].try_into().unwrap()),
                        ));
                    }
                    unpacked.push(ObjectContour {
                        closed: contour.closed,
                        vertices,
                    });
                }
                GeometryData::OutlineSet {
                    level,
                    contours: unpacked,
                }
            }
        };
        Ok(ObjectGeometry {
            id: self.id,
            source_record_id: self.source_record_id,
            role: self.role,
            quality: self.quality,
            method: self.method,
            evidence: self.evidence,
            data,
        })
    }
}

impl BuildSection {
    fn required(kind: [u8; 16], schema_major: u16, bytes: Vec<u8>) -> Self {
        Self {
            kind,
            schema_major,
            schema_minor: 0,
            flags: FLAG_REQUIRED,
            instance: 0,
            record_count: 0,
            record_stride: 0,
            bytes,
        }
    }

    fn optional(kind: [u8; 16], schema_major: u16, bytes: Vec<u8>) -> Self {
        Self {
            flags: 0,
            ..Self::required(kind, schema_major, bytes)
        }
    }
}

pub(crate) fn write(path: &Path, data: &ObjectCatalogData) -> io::Result<()> {
    let sections = build_sections(data)?;
    write_sections(path, sections)
}

fn build_sections(data: &ObjectCatalogData) -> io::Result<Vec<BuildSection>> {
    if !data.details.is_empty() && data.details.len() != data.objects.len() {
        return Err(invalid_input(
            "object details must be empty or ordinal-aligned with canonical objects",
        ));
    }
    let synthesized;
    let details = if data.details.is_empty() {
        synthesized = data
            .objects
            .iter()
            .enumerate()
            .map(|(index, object)| {
                let record_id = if object.metadata.id.is_empty() {
                    format!("legacy:ordinal:{index}")
                } else {
                    object.metadata.id.clone()
                };
                ObjectDetails::from_canonical_with_record_id(object, record_id)
            })
            .collect::<Vec<_>>();
        synthesized.as_slice()
    } else {
        data.details.as_slice()
    };
    for (object, detail) in data.objects.iter().zip(details) {
        if detail.canonical_id != object.metadata.id {
            return Err(invalid_input(
                "object detail canonical IDs must match canonical object order",
            ));
        }
    }

    let order = object_catalog_v3::spatial_order(&data.objects);
    let canonical_objects = order
        .iter()
        .map(|&index| data.objects[index].clone())
        .collect::<Vec<_>>();
    let ordered_details = order
        .iter()
        .map(|&index| &details[index])
        .collect::<Vec<_>>();
    let canonical = object_catalog_v3::encode(&canonical_objects)?;
    let mut detail_index = Vec::with_capacity(details.len() * DETAIL_INDEX_STRIDE as usize);
    let mut detail_data = Vec::new();
    let mut outline_vertices = Vec::new();
    for detail in &ordered_details {
        let offset = u64::try_from(detail_data.len()).map_err(|_| catalog_too_large())?;
        let disk_detail = DiskObjectDetails::pack(detail, &mut outline_vertices)?;
        let encoded = postcard::to_stdvec(&disk_detail).map_err(postcard_input_error)?;
        let len = u32::try_from(encoded.len()).map_err(|_| catalog_too_large())?;
        detail_index.extend_from_slice(&offset.to_le_bytes());
        detail_index.extend_from_slice(&len.to_le_bytes());
        detail_index.extend_from_slice(&0u32.to_le_bytes());
        detail_data.extend_from_slice(&encoded);
    }
    let provenance = serde_json::to_vec(&data.provenance).map_err(json_error)?;
    let capability_bits = capability_bits(details, true);

    let mut canonical_section = BuildSection::required(CANONICAL_V3, 3, canonical);
    canonical_section.record_count = data.objects.len() as u64;
    let mut detail_index_section = BuildSection::optional(DETAIL_INDEX, 1, detail_index);
    detail_index_section.record_count = details.len() as u64;
    detail_index_section.record_stride = DETAIL_INDEX_STRIDE;
    let mut detail_data_section = BuildSection::optional(DETAIL_DATA, 1, detail_data);
    detail_data_section.record_count = details.len() as u64;
    let mut provenance_section = BuildSection::optional(PROVENANCE_JSON, 1, provenance);
    provenance_section.record_count = 1;
    let mut capabilities_section =
        BuildSection::optional(CAPABILITIES, 1, capability_bits.to_le_bytes().to_vec());
    capabilities_section.record_count = 1;
    capabilities_section.record_stride = 8;
    let mut outline_section = BuildSection::optional(OUTLINE_VERTICES, 1, outline_vertices);
    outline_section.record_count = (outline_section.bytes.len() / 16) as u64;
    outline_section.record_stride = 16;

    Ok(vec![
        canonical_section,
        detail_index_section,
        detail_data_section,
        provenance_section,
        capabilities_section,
        outline_section,
    ])
}

fn capability_bits(details: &[ObjectDetails], provenance: bool) -> u64 {
    let mut bits = if provenance { CAP_PROVENANCE } else { 0 };
    for detail in details {
        if !detail.source_records.is_empty() {
            bits |= CAP_SOURCE_RECORDS;
        }
        if !detail.relations.is_empty() {
            bits |= CAP_RELATIONS;
        }
        if !detail.selections.is_empty() {
            bits |= CAP_SELECTIONS;
        }
        for geometry in &detail.geometries {
            match geometry.data {
                GeometryData::Ellipse { .. } => bits |= CAP_ELLIPSES,
                GeometryData::OutlineSet { .. } => bits |= CAP_OUTLINES,
                GeometryData::Point { .. } => {}
            }
        }
    }
    bits
}

fn write_sections(path: &Path, sections: Vec<BuildSection>) -> io::Result<()> {
    let section_count = u32::try_from(sections.len()).map_err(|_| catalog_too_large())?;
    let directory_offset = HEADER_SIZE;
    let directory_len = sections
        .len()
        .checked_mul(DIRECTORY_ENTRY_SIZE)
        .ok_or_else(catalog_too_large)?;
    let mut next_offset = align8(
        directory_offset
            .checked_add(directory_len)
            .ok_or_else(catalog_too_large)?,
    );
    let mut entries = Vec::with_capacity(sections.len());
    for section in &sections {
        let offset = next_offset;
        let len = section.bytes.len();
        next_offset = align8(offset.checked_add(len).ok_or_else(catalog_too_large)?);
        entries.push(SectionEntry {
            kind: section.kind,
            schema_major: section.schema_major,
            schema_minor: section.schema_minor,
            flags: section.flags,
            instance: section.instance,
            offset,
            len,
            record_count: section.record_count,
            record_stride: section.record_stride,
            checksum: Sha256::digest(&section.bytes).into(),
        });
    }
    let file_size = next_offset;

    let mut output = BufWriter::new(File::create(path)?);
    output.write_all(MAGIC)?;
    output.write_all(&CONTAINER_MAJOR.to_le_bytes())?;
    output.write_all(&CONTAINER_MINOR.to_le_bytes())?;
    output.write_all(&(HEADER_SIZE as u32).to_le_bytes())?;
    output.write_all(&(directory_offset as u64).to_le_bytes())?;
    output.write_all(&section_count.to_le_bytes())?;
    output.write_all(&(DIRECTORY_ENTRY_SIZE as u32).to_le_bytes())?;
    output.write_all(&(file_size as u64).to_le_bytes())?;
    output.write_all(&0u64.to_le_bytes())?;
    output.write_all(&[0u8; 16])?;
    for entry in &entries {
        write_directory_entry(&mut output, entry)?;
    }
    let directory_end = directory_offset + directory_len;
    write_padding(&mut output, directory_end, align8(directory_end))?;
    let mut position = align8(directory_end);
    for (entry, section) in entries.iter().zip(sections) {
        write_padding(&mut output, position, entry.offset)?;
        output.write_all(&section.bytes)?;
        position = entry.offset + entry.len;
        let aligned = align8(position);
        write_padding(&mut output, position, aligned)?;
        position = aligned;
    }
    if position != file_size {
        return Err(catalog_too_large());
    }
    output.flush()
}

fn write_directory_entry(output: &mut impl Write, entry: &SectionEntry) -> io::Result<()> {
    output.write_all(&entry.kind)?;
    output.write_all(&entry.schema_major.to_le_bytes())?;
    output.write_all(&entry.schema_minor.to_le_bytes())?;
    output.write_all(&entry.flags.to_le_bytes())?;
    output.write_all(&entry.instance.to_le_bytes())?;
    output.write_all(&0u32.to_le_bytes())?;
    output.write_all(&(entry.offset as u64).to_le_bytes())?;
    output.write_all(&(entry.len as u64).to_le_bytes())?;
    output.write_all(&entry.record_count.to_le_bytes())?;
    output.write_all(&entry.record_stride.to_le_bytes())?;
    output.write_all(&0u32.to_le_bytes())?;
    output.write_all(&entry.checksum)
}

pub(crate) struct MappedObjectCatalog {
    map: Arc<memmap2::Mmap>,
    canonical: object_catalog_v3::MappedObjectCatalog,
    sections: Vec<SectionEntry>,
    detail_index: Option<usize>,
    detail_data: Option<usize>,
    provenance_json: Option<usize>,
    outline_vertices: Option<usize>,
    capability_bits: u64,
    unknown_optional_sections: usize,
}

impl std::fmt::Debug for MappedObjectCatalog {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MappedObjectCatalogV4")
            .field("objects", &self.canonical.len())
            .field("sections", &self.sections.len())
            .field("unknown_optional_sections", &self.unknown_optional_sections)
            .finish_non_exhaustive()
    }
}

impl MappedObjectCatalog {
    pub(crate) fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        // Safety: the file is mapped read-only and the mapping is retained by
        // both the container and its embedded canonical view.
        let map = Arc::new(unsafe { memmap2::Mmap::map(&file)? });
        if map.len() < HEADER_SIZE || &map[..8] != MAGIC {
            return Err(invalid_data("not a seiza v4 object catalog"));
        }
        let major = read_u16(&map, 8)?;
        let _minor = read_u16(&map, 10)?;
        let header_size = read_u32(&map, 12)? as usize;
        let directory_offset = read_usize_u64(&map, 16, "directory offset")?;
        let section_count = read_u32(&map, 24)? as usize;
        let entry_size = read_u32(&map, 28)? as usize;
        let file_size = read_usize_u64(&map, 32, "file size")?;
        if major != CONTAINER_MAJOR
            || header_size != HEADER_SIZE
            || directory_offset != HEADER_SIZE
            || entry_size != DIRECTORY_ENTRY_SIZE
            || file_size != map.len()
            || map[40..64].iter().any(|byte| *byte != 0)
        {
            return Err(invalid_data(
                "unsupported or invalid object container header",
            ));
        }
        let directory_len = section_count
            .checked_mul(entry_size)
            .ok_or_else(|| invalid_data("object section directory is too large"))?;
        let directory_end = directory_offset
            .checked_add(directory_len)
            .ok_or_else(|| invalid_data("object section directory overflow"))?;
        if directory_end > map.len() {
            return Err(invalid_data("truncated object section directory"));
        }

        let mut sections = Vec::with_capacity(section_count);
        let mut seen = HashSet::new();
        let mut unknown_optional_sections = 0usize;
        for index in 0..section_count {
            let offset = directory_offset + index * entry_size;
            let entry = read_directory_entry(&map, offset)?;
            if !seen.insert((entry.kind, entry.instance)) {
                return Err(invalid_data("duplicate object section instance"));
            }
            let known = is_known_section(entry.kind);
            if !known && entry.flags & FLAG_REQUIRED != 0 {
                return Err(invalid_data(
                    "object catalog requires an unknown section capability",
                ));
            }
            if !known {
                unknown_optional_sections += 1;
            } else {
                validate_known_schema(&entry)?;
            }
            sections.push(entry);
        }
        let data_start = align8(directory_end);
        let mut ranges = sections
            .iter()
            .map(|entry| {
                let end = entry
                    .offset
                    .checked_add(entry.len)
                    .ok_or_else(|| invalid_data("object section range overflow"))?;
                if entry.offset < data_start || end > map.len() || entry.offset % 8 != 0 {
                    return Err(invalid_data("object section is out of bounds or unaligned"));
                }
                Ok((entry.offset, end))
            })
            .collect::<io::Result<Vec<_>>>()?;
        ranges.sort_unstable();
        if ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(invalid_data("object catalog sections overlap"));
        }

        let canonical_index = unique_section(&sections, CANONICAL_V3)?
            .ok_or_else(|| invalid_data("object catalog has no canonical query section"))?;
        let canonical_entry = &sections[canonical_index];
        if canonical_entry.flags & FLAG_REQUIRED == 0 {
            return Err(invalid_data("canonical object section must be required"));
        }
        let canonical = object_catalog_v3::MappedObjectCatalog::from_mmap(
            Arc::clone(&map),
            canonical_entry.offset,
            canonical_entry.len,
        )?;
        if canonical_entry.record_count != canonical.len() as u64 {
            return Err(invalid_data("canonical object count is inconsistent"));
        }

        let detail_index = unique_section(&sections, DETAIL_INDEX)?;
        let detail_data = unique_section(&sections, DETAIL_DATA)?;
        if detail_index.is_some() != detail_data.is_some() {
            return Err(invalid_data("object detail sections are incomplete"));
        }
        if let Some(index) = detail_index {
            let entry = &sections[index];
            if entry.record_count != canonical.len() as u64
                || entry.record_stride != DETAIL_INDEX_STRIDE
                || entry.len != canonical.len() * DETAIL_INDEX_STRIDE as usize
            {
                return Err(invalid_data("object detail index shape is inconsistent"));
            }
        }
        let provenance_json = unique_section(&sections, PROVENANCE_JSON)?;
        let outline_vertices = unique_section(&sections, OUTLINE_VERTICES)?;
        if let Some(index) = outline_vertices {
            let entry = &sections[index];
            if entry.record_stride != 16
                || entry.record_count > usize::MAX as u64
                || entry.len != entry.record_count as usize * 16
            {
                return Err(invalid_data("outline vertex section shape is inconsistent"));
            }
        }
        let capability_bits = if let Some(index) = unique_section(&sections, CAPABILITIES)? {
            let entry = &sections[index];
            if entry.len != 8 || entry.record_count != 1 || entry.record_stride != 8 {
                return Err(invalid_data("object capabilities section is invalid"));
            }
            read_u64(&map, entry.offset)?
        } else {
            0
        };

        Ok(Self {
            map,
            canonical,
            sections,
            detail_index,
            detail_data,
            provenance_json,
            outline_vertices,
            capability_bits,
            unknown_optional_sections,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.canonical.len()
    }

    pub(crate) fn object(&self, index: usize) -> io::Result<SkyObject> {
        self.canonical.object(index)
    }

    pub(crate) fn read_all(&self) -> io::Result<Vec<SkyObject>> {
        self.canonical.read_all()
    }

    pub(crate) fn candidates(&self, ra: f64, dec: f64, radius_deg: f64) -> io::Result<Vec<u32>> {
        self.canonical.candidates(ra, dec, radius_deg)
    }

    pub(crate) fn lookup_name(&self, designation: &str) -> io::Result<Vec<ObjectNameMatch>> {
        self.canonical.lookup_name(designation)
    }

    pub(crate) fn search_names(
        &self,
        prefix: &str,
        limit: usize,
    ) -> io::Result<Vec<ObjectNameMatch>> {
        self.canonical.search_names(prefix, limit)
    }

    pub(crate) fn details(&self, index: usize) -> io::Result<Option<ObjectDetails>> {
        let Some(detail) = self.disk_details(index)? else {
            return Ok(None);
        };
        let vertex_bytes = self
            .outline_vertices
            .map(|section| {
                let entry = &self.sections[section];
                &self.map[entry.offset..entry.offset + entry.len]
            })
            .unwrap_or(&[]);
        detail.unpack(vertex_bytes).map(Some)
    }

    fn disk_details(&self, index: usize) -> io::Result<Option<DiskObjectDetails>> {
        let (Some(index_section), Some(data_section)) = (self.detail_index, self.detail_data)
        else {
            return Ok(None);
        };
        if index >= self.len() {
            return Err(invalid_data("object detail index is out of bounds"));
        }
        let index_entry = &self.sections[index_section];
        let data_entry = &self.sections[data_section];
        let record_offset = index_entry.offset + index * DETAIL_INDEX_STRIDE as usize;
        let offset = read_usize_u64(&self.map, record_offset, "object detail offset")?;
        let len = read_u32(&self.map, record_offset + 8)? as usize;
        if self.map[record_offset + 12..record_offset + 16]
            .iter()
            .any(|byte| *byte != 0)
            || offset > data_entry.len
            || len > data_entry.len - offset
        {
            return Err(invalid_data("object detail range is invalid"));
        }
        let start = data_entry.offset + offset;
        postcard::from_bytes(&self.map[start..start + len])
            .map(Some)
            .map_err(postcard_data_error)
    }

    pub(crate) fn details_by_id(&self, canonical_id: &str) -> io::Result<Option<ObjectDetails>> {
        let Some(index) = self.index_by_id(canonical_id)? else {
            return Ok(None);
        };
        self.details(index)
    }

    pub(crate) fn source_records_by_id(
        &self,
        canonical_id: &str,
    ) -> io::Result<Option<Vec<ObjectSourceRecord>>> {
        let Some(index) = self.index_by_id(canonical_id)? else {
            return Ok(None);
        };
        Ok(self
            .disk_details(index)?
            .map(|details| details.source_records))
    }

    fn index_by_id(&self, canonical_id: &str) -> io::Result<Option<usize>> {
        for index in self.canonical.lookup_name_indices(canonical_id)? {
            if self.object(index as usize)?.metadata.id == canonical_id {
                return Ok(Some(index as usize));
            }
        }
        Ok(None)
    }

    pub(crate) fn provenance(&self) -> io::Result<Option<ObjectCatalogProvenance>> {
        let Some(index) = self.provenance_json else {
            return Ok(None);
        };
        let entry = &self.sections[index];
        serde_json::from_slice(&self.map[entry.offset..entry.offset + entry.len])
            .map(Some)
            .map_err(json_data_error)
    }

    pub(crate) fn capabilities(&self) -> ObjectCatalogCapabilities {
        ObjectCatalogCapabilities {
            source_records: self.capability_bits & CAP_SOURCE_RECORDS != 0,
            relations: self.capability_bits & CAP_RELATIONS != 0,
            selections: self.capability_bits & CAP_SELECTIONS != 0,
            ellipses: self.capability_bits & CAP_ELLIPSES != 0,
            outlines: self.capability_bits & CAP_OUTLINES != 0,
            provenance: self.capability_bits & CAP_PROVENANCE != 0,
            unknown_optional_sections: self.unknown_optional_sections,
        }
    }

    /// Exhaustive validation. Normal open deliberately does not hash or parse
    /// cold sections.
    pub(crate) fn validate(&self) -> io::Result<()> {
        for section in &self.sections {
            let bytes = &self.map[section.offset..section.offset + section.len];
            let checksum: [u8; 32] = Sha256::digest(bytes).into();
            if checksum != section.checksum {
                return Err(invalid_data("object section checksum mismatch"));
            }
        }
        self.canonical.validate()?;

        let mut canonical_ids = HashSet::new();
        let mut source_record_ids = HashSet::new();
        let mut geometry_ids = HashSet::new();
        for index in 0..self.len() {
            let object = self.object(index)?;
            if !object.metadata.id.is_empty() && !canonical_ids.insert(object.metadata.id.clone()) {
                return Err(invalid_data("duplicate canonical object ID"));
            }
            let Some(detail) = self.details(index)? else {
                continue;
            };
            if detail.canonical_id != object.metadata.id {
                return Err(invalid_data("object detail canonical ID mismatch"));
            }
            let local_sources = detail
                .source_records
                .iter()
                .map(|record| record.id.as_str())
                .collect::<HashSet<_>>();
            for record in &detail.source_records {
                if record.id.is_empty() || !source_record_ids.insert(record.id.clone()) {
                    return Err(invalid_data("duplicate or empty source-record ID"));
                }
            }
            for geometry in &detail.geometries {
                if geometry.id.is_empty()
                    || !geometry_ids.insert(geometry.id.clone())
                    || !local_sources.contains(geometry.source_record_id.as_str())
                {
                    return Err(invalid_data("invalid object geometry reference"));
                }
                validate_geometry(&geometry.data)?;
            }
            for relation in &detail.relations {
                if !local_sources.contains(relation.source_record_id.as_str()) {
                    return Err(invalid_data("invalid object relation source reference"));
                }
            }
            for selection in &detail.selections {
                if selection
                    .source_record_id
                    .as_deref()
                    .is_some_and(|id| !local_sources.contains(id))
                    || selection.geometry_id.as_deref().is_some_and(|id| {
                        !detail.geometries.iter().any(|geometry| geometry.id == id)
                    })
                {
                    return Err(invalid_data("invalid object selection reference"));
                }
            }
        }
        self.provenance()?;
        Ok(())
    }
}

fn validate_geometry(data: &GeometryData) -> io::Result<()> {
    let valid_position =
        |ra: f64, dec: f64| ra.is_finite() && dec.is_finite() && (-90.0..=90.0).contains(&dec);
    match data {
        GeometryData::Point { ra_deg, dec_deg } => {
            if !valid_position(*ra_deg, *dec_deg) {
                return Err(invalid_data("point geometry has invalid coordinates"));
            }
        }
        GeometryData::Ellipse {
            center_ra_deg,
            center_dec_deg,
            major_arcmin,
            minor_arcmin,
            position_angle_deg,
        } => {
            if !valid_position(*center_ra_deg, *center_dec_deg)
                || !major_arcmin.is_finite()
                || *major_arcmin <= 0.0
                || minor_arcmin.is_some_and(|value| !value.is_finite() || value <= 0.0)
                || position_angle_deg.is_some_and(|value| !value.is_finite())
            {
                return Err(invalid_data("ellipse geometry is invalid"));
            }
        }
        GeometryData::OutlineSet { contours, .. } => {
            if contours.is_empty()
                || contours.iter().any(|contour| {
                    contour.vertices.len() < 2
                        || contour
                            .vertices
                            .iter()
                            .any(|&(ra, dec)| !valid_position(ra, dec))
                })
            {
                return Err(invalid_data("outline geometry is invalid"));
            }
        }
    }
    Ok(())
}

fn read_directory_entry(bytes: &[u8], offset: usize) -> io::Result<SectionEntry> {
    let entry = bytes
        .get(offset..offset + DIRECTORY_ENTRY_SIZE)
        .ok_or_else(|| invalid_data("truncated object section directory entry"))?;
    if entry[24..28].iter().any(|byte| *byte != 0) || entry[60..64].iter().any(|byte| *byte != 0) {
        return Err(invalid_data("object section reserved fields are nonzero"));
    }
    Ok(SectionEntry {
        kind: entry[..16].try_into().unwrap(),
        schema_major: u16::from_le_bytes(entry[16..18].try_into().unwrap()),
        schema_minor: u16::from_le_bytes(entry[18..20].try_into().unwrap()),
        flags: u32::from_le_bytes(entry[20..24].try_into().unwrap()),
        instance: u32::from_le_bytes(entry[28..32].try_into().unwrap()),
        offset: usize::try_from(u64::from_le_bytes(entry[32..40].try_into().unwrap()))
            .map_err(|_| invalid_data("object section offset is too large"))?,
        len: usize::try_from(u64::from_le_bytes(entry[40..48].try_into().unwrap()))
            .map_err(|_| invalid_data("object section length is too large"))?,
        record_count: u64::from_le_bytes(entry[48..56].try_into().unwrap()),
        record_stride: u32::from_le_bytes(entry[56..60].try_into().unwrap()),
        checksum: entry[64..96].try_into().unwrap(),
    })
}

fn validate_known_schema(entry: &SectionEntry) -> io::Result<()> {
    let expected = if entry.kind == CANONICAL_V3 {
        3
    } else if matches!(
        entry.kind,
        DETAIL_INDEX | DETAIL_DATA | PROVENANCE_JSON | CAPABILITIES | OUTLINE_VERTICES
    ) {
        1
    } else {
        return Ok(());
    };
    if entry.schema_major != expected {
        return Err(invalid_data("unsupported object section schema"));
    }
    Ok(())
}

fn is_known_section(kind: [u8; 16]) -> bool {
    matches!(
        kind,
        CANONICAL_V3
            | DETAIL_INDEX
            | DETAIL_DATA
            | PROVENANCE_JSON
            | CAPABILITIES
            | OUTLINE_VERTICES
    )
}

fn unique_section(sections: &[SectionEntry], kind: [u8; 16]) -> io::Result<Option<usize>> {
    let mut matches = sections
        .iter()
        .enumerate()
        .filter(|(_, section)| section.kind == kind);
    let first = matches.next().map(|(index, _)| index);
    if matches.next().is_some() {
        return Err(invalid_data(
            "multiple instances of a singleton object section",
        ));
    }
    Ok(first)
}

fn read_u16(bytes: &[u8], offset: usize) -> io::Result<u16> {
    let value = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| invalid_data("truncated object catalog"))?;
    Ok(u16::from_le_bytes(value.try_into().unwrap()))
}

fn read_u32(bytes: &[u8], offset: usize) -> io::Result<u32> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| invalid_data("truncated object catalog"))?;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}

fn read_u64(bytes: &[u8], offset: usize) -> io::Result<u64> {
    let value = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| invalid_data("truncated object catalog"))?;
    Ok(u64::from_le_bytes(value.try_into().unwrap()))
}

fn read_usize_u64(bytes: &[u8], offset: usize, label: &str) -> io::Result<usize> {
    usize::try_from(read_u64(bytes, offset)?)
        .map_err(|_| invalid_data_owned(format!("object {label} is too large")))
}

fn align8(value: usize) -> usize {
    value.next_multiple_of(8)
}

fn write_padding(output: &mut impl Write, current: usize, target: usize) -> io::Result<()> {
    if target < current {
        return Err(catalog_too_large());
    }
    output.write_all(&vec![0; target - current])
}

fn catalog_too_large() -> io::Error {
    invalid_input("object catalog is too large for the v4 container")
}

fn invalid_input(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn invalid_data(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn invalid_data_owned(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn json_error(error: serde_json::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, error)
}

fn json_data_error(error: serde_json::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}

fn postcard_input_error(error: postcard::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, error)
}

fn postcard_data_error(error: postcard::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::{ObjectKind, ObjectMetadata};

    fn object() -> SkyObject {
        SkyObject {
            kind: ObjectKind::Nebula,
            ra: 338.0509,
            dec: 40.591,
            mag: None,
            major_arcmin: Some(75.0),
            minor_arcmin: Some(20.0),
            position_angle_deg: None,
            name: "LBN 437".into(),
            common_name: String::new(),
            metadata: ObjectMetadata {
                id: "vizier:VII/9:LBN437".into(),
                source: "VizieR VII/9/catalog".into(),
                aliases: vec!["DG 187".into()],
                ..ObjectMetadata::default()
            },
        }
    }

    #[test]
    fn v4_round_trip_is_mmap_backed_and_validates() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("objects.bin");
        let data = ObjectCatalogData::from_objects(vec![object()]);
        write(&path, &data).unwrap();

        let catalog = MappedObjectCatalog::open(&path).unwrap();
        assert_eq!(catalog.len(), 1);
        assert_eq!(catalog.lookup_name("DG187").unwrap().len(), 1);
        assert_eq!(catalog.details(0).unwrap().unwrap().source_records.len(), 1);
        assert!(catalog.capabilities().source_records);
        assert!(catalog.capabilities().ellipses);
        catalog.validate().unwrap();
    }

    #[test]
    fn unknown_optional_sections_are_skipped_but_required_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let optional = dir.path().join("optional.bin");
        let required = dir.path().join("required.bin");
        let data = ObjectCatalogData::from_objects(vec![object()]);
        let mut sections = build_sections(&data).unwrap();
        let unknown = *b"FUTURE_SECTION\0\0";
        sections.push(BuildSection::optional(unknown, 1, vec![1, 2, 3]));
        write_sections(&optional, sections).unwrap();
        let catalog = MappedObjectCatalog::open(&optional).unwrap();
        assert_eq!(catalog.capabilities().unknown_optional_sections, 1);
        catalog.validate().unwrap();

        let mut sections = build_sections(&data).unwrap();
        sections.push(BuildSection::required(unknown, 1, vec![1, 2, 3]));
        write_sections(&required, sections).unwrap();
        assert!(MappedObjectCatalog::open(&required).is_err());
    }
}
