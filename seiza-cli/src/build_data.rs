//! Catalog data builders.

use anyhow::{Context, Result, bail};
use seiza::catalog::TileSetBuilder;
use seiza::star_ids::{
    StarIdentifier, StarIdentifierCatalogBuilder, StarNameCatalog, StarNameKind,
    normalize_star_name,
};
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Build a star tile file from the Tycho-2 catalogue (CDS I/259).
///
/// Expects the distribution files `tyc2.dat.NN[.gz]` in `input`, plus
/// `suppl_1.dat[.gz]` — the supplement holds most stars brighter than
/// magnitude ~2 (Sirius is not in the main catalogue). Mean positions
/// (ICRS, epoch J2000; supplement epoch J1991.25) are proper-motion
/// corrected to `epoch`; entries without a mean position fall back to the
/// observed position.
pub fn build_tycho2(
    input: &Path,
    output: &Path,
    identifier_index: Option<&Path>,
    identifier_sources: Option<&Path>,
    epoch: f64,
    max_mag: f32,
) -> Result<()> {
    if identifier_index.is_some_and(|path| path == output) {
        bail!("--identifier-index must differ from --output");
    }
    if identifier_sources.is_some() && identifier_index.is_none() {
        bail!("--identifier-sources requires --identifier-index");
    }
    if !epoch.is_finite() || !max_mag.is_finite() {
        bail!("epoch and magnitude limit must be finite");
    }
    let mut parts: Vec<_> = std::fs::read_dir(input)
        .with_context(|| format!("cannot read {}", input.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("tyc2.dat."))
        })
        .collect();
    parts.sort();
    if parts.is_empty() {
        bail!("no tyc2.dat.* files found in {}", input.display());
    }

    // 45 declination bands ≈ 4° tiles: right-sized for the ~2.5M star
    // lite tier
    let mut builder = TileSetBuilder::new(
        45,
        epoch,
        "Tycho-2 (Hog et al. 2000, CDS I/259) incl. supplement-1; free for scientific use",
    );
    let mut skipped_no_mag = 0u64;
    let mut too_faint = 0u64;
    let mut identifiers = identifier_index.map(|_| {
        let attribution = if identifier_sources.is_some() {
            "Tycho-2 I/259; Bright Star Catalogue V/50; GCVS B/gcvs; Washington Double Star B/wds; IAU Catalog of Star Names"
        } else {
            "Tycho-2 (Hog et al. 2000, CDS I/259); TYC identifiers and catalog-supplied HIP cross-identifications"
        };
        StarIdentifierCatalogBuilder::new(
            epoch,
            attribution,
        )
    });

    for part in &parts {
        let file =
            std::fs::File::open(part).with_context(|| format!("cannot open {}", part.display()))?;
        let reader: Box<dyn std::io::Read> = if part.extension().is_some_and(|ext| ext == "gz") {
            Box::new(flate2::read::GzDecoder::new(file))
        } else {
            Box::new(file)
        };

        for line in BufReader::new(reader).lines() {
            let line = line?;
            let Some(star) = parse_tycho2_line(&line, epoch) else {
                skipped_no_mag += 1;
                continue;
            };
            if star.mag > max_mag {
                too_faint += 1;
                continue;
            }
            builder.add(star.ra, star.dec, star.mag);
            add_tycho_identifiers(&mut identifiers, star)?;
        }
    }

    let mut suppl_count = 0u64;
    for name in ["suppl_1.dat.gz", "suppl_1.dat"] {
        let path = input.join(name);
        if !path.exists() {
            continue;
        }
        let file = std::fs::File::open(&path)
            .with_context(|| format!("cannot open {}", path.display()))?;
        let reader: Box<dyn std::io::Read> = if name.ends_with(".gz") {
            Box::new(flate2::read::GzDecoder::new(file))
        } else {
            Box::new(file)
        };
        for line in BufReader::new(reader).lines() {
            let line = line?;
            let Some(star) = parse_tycho2_suppl_line(&line, epoch) else {
                skipped_no_mag += 1;
                continue;
            };
            if star.mag > max_mag {
                too_faint += 1;
                continue;
            }
            builder.add(star.ra, star.dec, star.mag);
            add_tycho_identifiers(&mut identifiers, star)?;
            suppl_count += 1;
        }
        break;
    }
    if suppl_count == 0 {
        eprintln!(
            "warning: no supplement-1 stars ingested — the brightest stars \
             (including Sirius) will be missing; run download-data tycho2"
        );
    }

    let count = builder.star_count();
    builder.write_to(output)?;
    if let (Some(sources), Some(identifiers)) = (identifier_sources, identifiers.as_mut()) {
        let stats = add_stellar_identifier_sources(identifiers, sources, epoch)?;
        println!(
            "added {} Bright Star aliases, {} GCVS variables, {} WDS designations, and {} IAU names",
            stats.bright, stats.variables, stats.doubles, stats.proper_names
        );
    }
    if let (Some(path), Some(identifiers)) = (identifier_index, identifiers) {
        let stats = identifiers.write_to(path)?;
        println!(
            "{} numeric identifiers and {} names written to {}",
            stats.numeric_entries,
            stats.name_entries,
            path.display()
        );
    }
    println!(
        "{} stars written to {} (epoch {epoch}, {} unusable records skipped, {} fainter than {max_mag})",
        count,
        output.display(),
        skipped_no_mag,
        too_faint
    );
    Ok(())
}

fn assemble_object_catalog_data(
    input: &Path,
    curation_dir: Option<&Path>,
    mut objects: Vec<seiza::objects::SkyObject>,
    source_records: Vec<seiza::objects::ObjectSourceRecord>,
) -> Result<seiza::objects::ObjectCatalogData> {
    use seiza::objects::{
        GeometryData, GeometryQuality, GeometryRole, ObjectCatalogData, ObjectCatalogProvenance,
        ObjectDetails, ObjectFacet, ObjectGeometry, ObjectProvenanceFile, ObjectProvenanceSource,
        ObjectRelation, ObjectRelationKind, ObjectSelection,
    };

    let mut id_to_objects = std::collections::HashMap::<String, Vec<usize>>::new();
    let mut primary_id_to_object = std::collections::HashMap::<String, usize>::new();
    for (index, object) in objects.iter().enumerate() {
        if primary_id_to_object
            .insert(object.metadata.id.clone(), index)
            .is_some()
        {
            bail!("duplicate canonical object ID: {}", object.metadata.id);
        }
        for id in std::iter::once(&object.metadata.id).chain(&object.metadata.alternate_ids) {
            let entries = id_to_objects.entry(id.clone()).or_default();
            if !entries.contains(&index) {
                entries.push(index);
            }
        }
    }
    let mut details = objects
        .iter()
        .map(|object| ObjectDetails {
            canonical_id: object.metadata.id.clone(),
            ..ObjectDetails::default()
        })
        .collect::<Vec<_>>();
    let mut seen_source_ids = std::collections::HashSet::new();
    for record in source_records {
        if record.id.is_empty() || !seen_source_ids.insert(record.id.clone()) {
            bail!("duplicate or empty source-record ID: {}", record.id);
        }
        let exact = primary_id_to_object.get(&record.id).copied();
        let candidates = id_to_objects.get(&record.id).cloned().unwrap_or_default();
        let index = exact
            .or_else(|| (candidates.len() == 1).then_some(candidates[0]))
            .with_context(|| {
                format!(
                    "source record {} has no unambiguous canonical object",
                    record.id
                )
            })?;
        let geometry = if let Some(major_arcmin) = record.object.major_arcmin {
            ObjectGeometry {
                id: format!("{}#catalog-ellipse", record.id),
                source_record_id: record.id.clone(),
                role: GeometryRole::CatalogExtent,
                quality: GeometryQuality::Catalog,
                method: String::new(),
                evidence: String::new(),
                data: GeometryData::Ellipse {
                    center_ra_deg: record.object.ra,
                    center_dec_deg: record.object.dec,
                    major_arcmin,
                    minor_arcmin: record.object.minor_arcmin,
                    position_angle_deg: record.object.position_angle_deg,
                },
            }
        } else {
            ObjectGeometry {
                id: format!("{}#catalog-point", record.id),
                source_record_id: record.id.clone(),
                role: GeometryRole::CatalogExtent,
                quality: GeometryQuality::Catalog,
                method: String::new(),
                evidence: String::new(),
                data: GeometryData::Point {
                    ra_deg: record.object.ra,
                    dec_deg: record.object.dec,
                },
            }
        };
        details[index].geometries.push(geometry);
        details[index].source_records.push(record);
    }

    for (object, detail) in objects.iter_mut().zip(&mut details) {
        if detail.source_records.is_empty() {
            bail!(
                "canonical object {} has no source records",
                object.metadata.id
            );
        }
        let canonical_id = object.metadata.id.clone();
        let identity_index = detail
            .source_records
            .iter()
            .position(|record| record.id == canonical_id)
            .unwrap_or(0);
        let identity = detail.source_records[identity_index].clone();
        let geometry_source = detail
            .source_records
            .iter()
            .position(|record| record.object.major_arcmin.is_some())
            .unwrap_or(identity_index);
        let photometry_source = detail
            .source_records
            .iter()
            .position(|record| record.object.mag.is_some());

        let mut canonical = identity.object.clone();
        canonical.metadata.id = canonical_id.clone();
        canonical.metadata.source = identity.source.clone();
        canonical.metadata.aliases.clear();
        canonical.metadata.parent_ids.clear();
        canonical.metadata.alternate_ids.clear();
        canonical.metadata.alternate_sources.clear();
        if let Some(record) = detail.source_records.get(geometry_source) {
            canonical.major_arcmin = record.object.major_arcmin;
            canonical.minor_arcmin = record.object.minor_arcmin;
            canonical.position_angle_deg = record.object.position_angle_deg;
        }
        canonical.mag = photometry_source.and_then(|index| detail.source_records[index].object.mag);
        for record in &detail.source_records {
            if record.object.kind != seiza::objects::ObjectKind::Other
                && canonical.kind == seiza::objects::ObjectKind::Other
            {
                canonical.kind = record.object.kind;
            }
            if canonical.common_name.is_empty() && !record.object.common_name.is_empty() {
                canonical.common_name = record.object.common_name.clone();
            }
            add_alias(&mut canonical, record.object.name.clone());
            for alias in &record.object.metadata.aliases {
                add_alias(&mut canonical, alias.clone());
            }
            if record.id != canonical_id {
                add_unique(&mut canonical.metadata.alternate_ids, record.id.clone());
                detail.relations.push(ObjectRelation {
                    kind: ObjectRelationKind::SameAs,
                    target_id: record.id.clone(),
                    source_record_id: record.id.clone(),
                    note: String::new(),
                });
            }
            for id in &record.object.metadata.alternate_ids {
                if id != &canonical_id {
                    add_unique(&mut canonical.metadata.alternate_ids, id.clone());
                }
            }
            if record.source != canonical.metadata.source {
                add_unique(
                    &mut canonical.metadata.alternate_sources,
                    record.source.clone(),
                );
            }
            for source in &record.object.metadata.alternate_sources {
                if source != &canonical.metadata.source {
                    add_unique(&mut canonical.metadata.alternate_sources, source.clone());
                }
            }
            for parent in &record.object.metadata.parent_ids {
                add_unique(&mut canonical.metadata.parent_ids, parent.clone());
                detail.relations.push(ObjectRelation {
                    kind: ObjectRelationKind::ComponentOf,
                    target_id: parent.clone(),
                    source_record_id: record.id.clone(),
                    note: String::new(),
                });
            }
        }
        let identity_id = identity.id.clone();
        detail.selections.extend([
            ObjectSelection {
                facet: ObjectFacet::PreferredIdentity,
                source_record_id: Some(identity_id.clone()),
                geometry_id: None,
                reason: String::new(),
            },
            ObjectSelection {
                facet: ObjectFacet::PreferredPosition,
                source_record_id: Some(identity_id.clone()),
                geometry_id: None,
                reason: String::new(),
            },
            ObjectSelection {
                facet: ObjectFacet::PreferredClassification,
                source_record_id: Some(identity_id),
                geometry_id: None,
                reason: String::new(),
            },
        ]);
        if let Some(index) = photometry_source {
            detail.selections.push(ObjectSelection {
                facet: ObjectFacet::PreferredPhotometry,
                source_record_id: Some(detail.source_records[index].id.clone()),
                geometry_id: None,
                reason: String::new(),
            });
        }
        let geometry_record_id = detail.source_records[geometry_source].id.clone();
        let geometry_id = detail
            .geometries
            .iter()
            .find(|geometry| geometry.source_record_id == geometry_record_id)
            .map(|geometry| geometry.id.clone());
        detail.selections.push(ObjectSelection {
            facet: ObjectFacet::PreferredGeometry,
            source_record_id: Some(geometry_record_id),
            geometry_id,
            reason: String::new(),
        });
        *object = canonical;
    }

    let mut provenance = ObjectCatalogProvenance {
        build_policy: "v4 source records; explicit identity links; atomic geometry selection"
            .into(),
        ..ObjectCatalogProvenance::default()
    };
    for descriptor in OBJECT_SOURCE_DESCRIPTORS {
        let mut files = Vec::new();
        for name in descriptor.files {
            let path = input.join(name);
            if path.exists() {
                let (bytes, sha256) = file_digest(&path)?;
                files.push(ObjectProvenanceFile {
                    path: (*name).into(),
                    bytes,
                    sha256,
                });
            }
        }
        if !files.is_empty() {
            provenance.sources.push(ObjectProvenanceSource {
                name: descriptor.label.into(),
                reference_url: descriptor.reference_url.into(),
                revision: String::new(),
                files,
            });
        }
    }

    let mut data = ObjectCatalogData {
        objects,
        details,
        provenance,
    };
    if let Some(curation_dir) = curation_dir {
        apply_object_curation(input, curation_dir, &mut data)?;
    }
    Ok(data)
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CurationMetadataFile {
    repository: String,
    #[serde(default)]
    commit: Option<String>,
    schema_version: u32,
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CuratedObjectFile {
    schema_version: u32,
    id: String,
    target_id: String,
    #[serde(default)]
    notes: String,
    #[serde(default)]
    evidence: Vec<CuratedEvidence>,
    #[serde(default)]
    geometries: Vec<CuratedGeometry>,
    #[serde(default)]
    outlines: Vec<CuratedOutline>,
    #[serde(default)]
    relations: Vec<CuratedRelation>,
    #[serde(default)]
    selections: Vec<CuratedSelection>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CuratedEvidence {
    #[serde(default)]
    kind: String,
    #[serde(default)]
    citation: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    notes: String,
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CuratedGeometry {
    id: String,
    #[serde(rename = "type")]
    geometry_type: String,
    center_ra_deg: f64,
    center_dec_deg: f64,
    major_arcmin: f32,
    #[serde(default)]
    minor_arcmin: Option<f32>,
    #[serde(default)]
    position_angle_deg: Option<f32>,
    #[serde(default = "default_fallback_geometry_role")]
    role: String,
    #[serde(default = "default_estimated_quality")]
    quality: String,
    #[serde(default)]
    method: String,
    #[serde(default)]
    evidence: Vec<CuratedEvidence>,
    #[serde(default)]
    notes: String,
    #[serde(default)]
    preferred: bool,
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CuratedSelection {
    facet: String,
    #[serde(default)]
    source_record_id: Option<String>,
    #[serde(default)]
    geometry_id: Option<String>,
    #[serde(default)]
    notes: String,
    #[serde(default)]
    evidence: Vec<CuratedEvidence>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CuratedRelation {
    kind: String,
    related_id: String,
    #[serde(default)]
    source_record_id: Option<String>,
    #[serde(default)]
    notes: String,
    #[serde(default)]
    evidence: Vec<CuratedEvidence>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CuratedOutline {
    file: String,
    #[serde(default)]
    source_record_id: Option<String>,
    #[serde(default = "default_brightness_geometry_role")]
    role: String,
    #[serde(default = "default_catalog_quality")]
    quality: String,
    #[serde(default = "default_openngc_outline_method")]
    method: String,
    #[serde(default)]
    preferred: bool,
    #[serde(default)]
    evidence: Vec<CuratedEvidence>,
    #[serde(default)]
    notes: String,
}

fn default_fallback_geometry_role() -> String {
    "fallback-extent".into()
}

fn default_brightness_geometry_role() -> String {
    "brightness-level".into()
}

fn default_estimated_quality() -> String {
    "estimated".into()
}

fn default_catalog_quality() -> String {
    "catalog".into()
}

fn default_openngc_outline_method() -> String {
    "OpenNGC hand-drawn DSS2 contour".into()
}

fn apply_object_curation(
    input: &Path,
    curation_dir: &Path,
    data: &mut seiza::objects::ObjectCatalogData,
) -> Result<()> {
    use seiza::objects::{
        GeometryData, ObjectCurationRevision, ObjectFacet, ObjectGeometry, ObjectProperty,
        ObjectProvenanceFile, ObjectProvenanceSource, ObjectRelation, ObjectSelection,
        ObjectSourceRecord,
    };

    let metadata_path = curation_dir.join("curation.json");
    let metadata: CurationMetadataFile =
        serde_json::from_reader(std::fs::File::open(&metadata_path).with_context(|| {
            format!(
                "missing pinned curation metadata {}",
                metadata_path.display()
            )
        })?)?;
    if metadata.repository.trim().is_empty() {
        bail!("curation.json must record repository and schema_version");
    }
    if metadata.schema_version != 2 {
        bail!(
            "unsupported curation schema {}; expected per-object schema 2",
            metadata.schema_version
        );
    }
    let commit = resolve_curation_commit(curation_dir, metadata.commit.as_deref())?;

    let documents = read_curation_documents(curation_dir, metadata.schema_version)?;
    let mut curated_facets = std::collections::HashSet::<(String, String)>::new();
    let mut correction_ids = std::collections::HashSet::<String>::new();
    let mut outline_files = std::collections::HashSet::<String>::new();
    for document in documents {
        let target_id = document.target_id.as_str();
        let index = resolve_canonical_index(data, target_id)?;

        for row in document.geometries {
            if row.geometry_type != "ellipse" {
                bail!(
                    "curation {} geometry {} has unsupported type {}",
                    document.id,
                    row.id,
                    row.geometry_type
                );
            }
            if !correction_ids.insert(row.id.clone()) {
                bail!("duplicate curation correction ID {}", row.id);
            }
            validate_curated_ellipse(&document.id, &row)?;
            let role = parse_geometry_role(&row.role)?;
            let quality = parse_geometry_quality(&row.quality)?;
            let source_record_id = format!("curation:{}", row.id);
            if data.details[index]
                .source_records
                .iter()
                .any(|record| record.id == source_record_id)
            {
                bail!("duplicate curation correction ID {}", row.id);
            }
            let geometry_id = format!("{source_record_id}#ellipse");
            let notes = combine_curation_notes(&document.notes, &row.notes);
            let evidence = combine_curation_evidence(&document.evidence, &row.evidence);
            let mut corrected = data.objects[index].clone();
            corrected.ra = row.center_ra_deg;
            corrected.dec = row.center_dec_deg;
            corrected.major_arcmin = Some(row.major_arcmin);
            corrected.minor_arcmin = row.minor_arcmin;
            corrected.position_angle_deg = row.position_angle_deg;
            corrected.metadata.id = source_record_id.clone();
            corrected.metadata.source = "Seiza catalog curation".into();
            let mut properties = Vec::new();
            for (name, value) in [
                ("note", notes.as_str()),
                ("method", row.method.as_str()),
                ("evidence", evidence.as_str()),
            ] {
                if !value.is_empty() {
                    properties.push(ObjectProperty {
                        name: name.into(),
                        value: value.into(),
                        unit: None,
                    });
                }
            }
            data.details[index].source_records.push(ObjectSourceRecord {
                id: source_record_id.clone(),
                source: "Seiza catalog curation".into(),
                object: corrected,
                properties,
            });
            let geometry = ObjectGeometry {
                id: geometry_id.clone(),
                source_record_id: source_record_id.clone(),
                role,
                quality,
                method: row.method,
                evidence,
                data: GeometryData::Ellipse {
                    center_ra_deg: row.center_ra_deg,
                    center_dec_deg: row.center_dec_deg,
                    major_arcmin: row.major_arcmin,
                    minor_arcmin: row.minor_arcmin,
                    position_angle_deg: row.position_angle_deg,
                },
            };
            data.details[index].geometries.push(geometry.clone());
            if row.preferred {
                register_curated_facet(&mut curated_facets, target_id, "preferred-geometry")?;
                replace_selection(
                    &mut data.details[index].selections,
                    ObjectSelection {
                        facet: ObjectFacet::PreferredGeometry,
                        source_record_id: Some(source_record_id),
                        geometry_id: Some(geometry_id),
                        reason: preferred_curation_reason(&document.notes, &row.notes),
                    },
                );
                apply_geometry_to_hot_object(&mut data.objects[index], &geometry)?;
            }
        }

        for row in document.outlines {
            if !outline_files.insert(row.file.clone()) {
                bail!("OpenNGC outline {} is mapped more than once", row.file);
            }
            let source_record_id = row.source_record_id.clone().unwrap_or_else(|| {
                data.details[index]
                    .source_records
                    .iter()
                    .find(|record| record.source == "OpenNGC")
                    .map(|record| record.id.clone())
                    .unwrap_or_default()
            });
            if source_record_id.is_empty()
                || !data.details[index]
                    .source_records
                    .iter()
                    .any(|record| record.id == source_record_id)
            {
                bail!(
                    "OpenNGC outline {} has no valid source record for {}",
                    row.file,
                    target_id
                );
            }
            let path = [
                input.join("outlines").join("objects").join(&row.file),
                input.join("outlines").join(&row.file),
            ]
            .into_iter()
            .find(|path| path.exists())
            .with_context(|| format!("OpenNGC outline {} is missing", row.file))?;
            let contours = parse_openngc_outline(&std::fs::read_to_string(&path)?)?;
            let (bytes, sha256) = file_digest(&path)?;
            let outline_source = if let Some(source) = data
                .provenance
                .sources
                .iter_mut()
                .find(|source| source.name == "OpenNGC outlines")
            {
                source
            } else {
                data.provenance.sources.push(ObjectProvenanceSource {
                    name: "OpenNGC outlines".into(),
                    reference_url: "https://github.com/mattiaverga/OpenNGC/tree/master/outlines"
                        .into(),
                    revision: String::new(),
                    files: Vec::new(),
                });
                data.provenance.sources.last_mut().unwrap()
            };
            if !outline_source
                .files
                .iter()
                .any(|file| file.path == format!("outlines/objects/{}", row.file))
            {
                outline_source.files.push(ObjectProvenanceFile {
                    path: format!("outlines/objects/{}", row.file),
                    bytes,
                    sha256,
                });
            }
            let level = row
                .file
                .rsplit_once("_lv")
                .and_then(|(_, suffix)| suffix.strip_suffix(".txt"))
                .map(|level| format!("level-{level}"));
            let geometry_id = format!("openngc-outline:{}", row.file);
            let narrative = curation_narrative(
                &document.notes,
                &row.notes,
                &document.evidence,
                &row.evidence,
            );
            let geometry = ObjectGeometry {
                id: geometry_id.clone(),
                source_record_id: source_record_id.clone(),
                role: parse_geometry_role(&row.role)?,
                quality: parse_geometry_quality(&row.quality)?,
                method: row.method,
                evidence: if narrative.is_empty() {
                    format!("outlines/objects/{}", row.file)
                } else {
                    narrative
                },
                data: GeometryData::OutlineSet { level, contours },
            };
            data.details[index].geometries.push(geometry.clone());
            if row.preferred {
                register_curated_facet(&mut curated_facets, target_id, "preferred-geometry")?;
                replace_selection(
                    &mut data.details[index].selections,
                    ObjectSelection {
                        facet: ObjectFacet::PreferredGeometry,
                        source_record_id: Some(source_record_id),
                        geometry_id: Some(geometry_id),
                        reason: preferred_curation_reason(&document.notes, &row.notes),
                    },
                );
                apply_geometry_to_hot_object(&mut data.objects[index], &geometry)?;
            }
        }

        for row in document.relations {
            let source_record_id = row.source_record_id.unwrap_or_else(|| {
                data.details[index]
                    .selections
                    .iter()
                    .find(|selection| selection.facet == ObjectFacet::PreferredIdentity)
                    .and_then(|selection| selection.source_record_id.clone())
                    .unwrap_or_default()
            });
            if !data.details[index]
                .source_records
                .iter()
                .any(|record| record.id == source_record_id)
            {
                bail!("relation for {target_id} has an invalid source record");
            }
            data.details[index].relations.push(ObjectRelation {
                kind: parse_relation_kind(&row.kind)?,
                target_id: row.related_id,
                source_record_id,
                note: curation_narrative(
                    &document.notes,
                    &row.notes,
                    &document.evidence,
                    &row.evidence,
                ),
            });
        }

        for row in document.selections {
            let facet = parse_object_facet(&row.facet)?;
            register_curated_facet(&mut curated_facets, target_id, &row.facet)?;
            let selection = ObjectSelection {
                facet,
                source_record_id: row.source_record_id,
                geometry_id: row.geometry_id,
                reason: curation_narrative(
                    &document.notes,
                    &row.notes,
                    &document.evidence,
                    &row.evidence,
                ),
            };
            apply_selected_facet(&mut data.objects[index], &data.details[index], &selection)?;
            replace_selection(&mut data.details[index].selections, selection);
        }
    }

    let mut curation_files = Vec::new();
    for path in recursive_files(curation_dir)? {
        let relative = path.strip_prefix(curation_dir).unwrap_or(&path);
        let (bytes, sha256) = file_digest(&path)?;
        curation_files.push(ObjectProvenanceFile {
            path: relative.to_string_lossy().replace('\\', "/"),
            bytes,
            sha256,
        });
    }
    curation_files.sort_by(|left, right| left.path.cmp(&right.path));
    data.provenance.curation = Some(ObjectCurationRevision {
        repository: metadata.repository,
        commit,
        schema_version: metadata.schema_version.to_string(),
        files: curation_files,
    });
    Ok(())
}

fn resolve_curation_commit(curation_dir: &Path, declared: Option<&str>) -> Result<String> {
    if curation_dir.join(".git").exists() {
        let head = std::process::Command::new("git")
            .args(["-C"])
            .arg(curation_dir)
            .args(["rev-parse", "HEAD"])
            .output()
            .context("cannot inspect curation Git revision")?;
        if !head.status.success() {
            bail!("cannot resolve curation Git revision");
        }
        let head = String::from_utf8(head.stdout)?.trim().to_string();
        if declared.is_some_and(|declared| declared != head) {
            bail!("curation.json commit does not match checkout HEAD {head}");
        }
        let status = std::process::Command::new("git")
            .args(["-C"])
            .arg(curation_dir)
            .args(["status", "--porcelain", "--untracked-files=all"])
            .output()
            .context("cannot inspect curation Git status")?;
        if !status.status.success() || !status.stdout.is_empty() {
            bail!("curation checkout must be clean before building");
        }
        return Ok(head);
    }
    let snapshot_revision = curation_dir.join(".seiza-revision");
    if snapshot_revision.exists() {
        let commit = std::fs::read_to_string(&snapshot_revision)
            .with_context(|| format!("cannot read {}", snapshot_revision.display()))?;
        let commit = commit.trim();
        if declared.is_some_and(|declared| declared != commit) {
            bail!("curation.json commit does not match downloaded snapshot {commit}");
        }
        if !commit.is_empty() {
            return Ok(commit.to_string());
        }
    }
    declared
        .filter(|commit| !commit.trim().is_empty())
        .map(str::to_string)
        .context("non-Git curation snapshots must declare commit in curation.json")
}

fn read_curation_documents(
    curation_dir: &Path,
    schema_version: u32,
) -> Result<Vec<CuratedObjectFile>> {
    let objects_dir = curation_dir.join("objects");
    if !objects_dir.is_dir() {
        bail!(
            "curation schema {schema_version} requires an objects directory at {}",
            objects_dir.display()
        );
    }
    let mut document_ids = std::collections::HashSet::new();
    let mut target_ids = std::collections::HashSet::new();
    let mut documents = Vec::new();
    for path in recursive_files(&objects_dir)? {
        if path.extension().and_then(|extension| extension.to_str()) != Some("toml") {
            bail!("unexpected non-TOML curation file {}", path.display());
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("cannot read curation document {}", path.display()))?;
        let document: CuratedObjectFile = toml::from_str(&text)
            .with_context(|| format!("invalid curation document {}", path.display()))?;
        if document.schema_version != schema_version {
            bail!(
                "curation document {} uses schema {}, expected {}",
                path.display(),
                document.schema_version,
                schema_version
            );
        }
        validate_curation_document(&path, &document)?;
        if !document_ids.insert(document.id.clone()) {
            bail!("duplicate curation document ID {}", document.id);
        }
        if !target_ids.insert(document.target_id.clone()) {
            bail!(
                "multiple curation documents target {}; keep one file per object",
                document.target_id
            );
        }
        documents.push(document);
    }
    Ok(documents)
}

fn validate_curation_document(path: &Path, document: &CuratedObjectFile) -> Result<()> {
    let file_id = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("");
    if document.id.is_empty()
        || !document
            .id
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        || !document
            .id
            .bytes()
            .next()
            .is_some_and(|byte| byte.is_ascii_alphanumeric())
    {
        bail!(
            "curation document {} has invalid ID {}; use lowercase letters, digits, and hyphens",
            path.display(),
            document.id
        );
    }
    if file_id != document.id {
        bail!(
            "curation document {} must be named {}.toml",
            path.display(),
            document.id
        );
    }
    if document.target_id.trim().is_empty() {
        bail!(
            "curation document {} has an empty target_id",
            path.display()
        );
    }
    validate_curation_evidence(path, "document", &document.evidence)?;
    for geometry in &document.geometries {
        if geometry.id.trim().is_empty() {
            bail!(
                "curation document {} has an empty geometry ID",
                path.display()
            );
        }
        validate_curation_evidence(path, "geometry", &geometry.evidence)?;
    }
    for outline in &document.outlines {
        if outline.file.trim().is_empty()
            || outline.file.contains(['/', '\\'])
            || !outline.file.ends_with(".txt")
        {
            bail!(
                "curation document {} has invalid outline file {}",
                path.display(),
                outline.file
            );
        }
        validate_curation_evidence(path, "outline", &outline.evidence)?;
    }
    for relation in &document.relations {
        validate_curation_evidence(path, "relation", &relation.evidence)?;
    }
    for selection in &document.selections {
        validate_curation_evidence(path, "selection", &selection.evidence)?;
    }
    Ok(())
}

fn validate_curation_evidence(
    path: &Path,
    owner: &str,
    evidence: &[CuratedEvidence],
) -> Result<()> {
    if evidence.iter().any(|item| {
        item.kind.trim().is_empty()
            && item.citation.trim().is_empty()
            && item.url.trim().is_empty()
            && item.notes.trim().is_empty()
    }) {
        bail!(
            "curation document {} has empty {owner} evidence",
            path.display()
        );
    }
    Ok(())
}

fn validate_curated_ellipse(document_id: &str, geometry: &CuratedGeometry) -> Result<()> {
    if !(0.0..360.0).contains(&geometry.center_ra_deg)
        || !(-90.0..=90.0).contains(&geometry.center_dec_deg)
        || !geometry.major_arcmin.is_finite()
        || geometry.major_arcmin <= 0.0
        || geometry
            .minor_arcmin
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        || geometry
            .position_angle_deg
            .is_some_and(|value| !value.is_finite() || !(0.0..180.0).contains(&value))
    {
        bail!(
            "curation {document_id} geometry {} has invalid ellipse coordinates or dimensions",
            geometry.id
        );
    }
    Ok(())
}

fn combine_curation_notes(document: &str, item: &str) -> String {
    [document.trim(), item.trim()]
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn combine_curation_evidence(document: &[CuratedEvidence], item: &[CuratedEvidence]) -> String {
    let mut evidence = Vec::new();
    for entry in document.iter().chain(item) {
        let entry = format_curation_evidence(entry);
        if !evidence.contains(&entry) {
            evidence.push(entry);
        }
    }
    evidence.join("\n")
}

fn format_curation_evidence(evidence: &CuratedEvidence) -> String {
    let mut heading = [evidence.kind.trim(), evidence.citation.trim()]
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(": ");
    if !evidence.url.trim().is_empty() {
        if !heading.is_empty() {
            heading.push_str(": ");
        }
        heading.push_str(evidence.url.trim());
    }
    match (heading.is_empty(), evidence.notes.trim().is_empty()) {
        (false, false) => format!("{heading} — {}", evidence.notes.trim()),
        (false, true) => heading,
        (true, false) => evidence.notes.trim().to_string(),
        (true, true) => String::new(),
    }
}

fn preferred_curation_reason(document: &str, item: &str) -> String {
    if item.trim().is_empty() {
        document.trim().to_string()
    } else {
        item.trim().to_string()
    }
}

fn curation_narrative(
    document_notes: &str,
    item_notes: &str,
    document_evidence: &[CuratedEvidence],
    item_evidence: &[CuratedEvidence],
) -> String {
    let notes = combine_curation_notes(document_notes, item_notes);
    let evidence = combine_curation_evidence(document_evidence, item_evidence);
    match (notes.is_empty(), evidence.is_empty()) {
        (false, false) => format!("{notes}\n\nEvidence:\n{evidence}"),
        (false, true) => notes,
        (true, false) => format!("Evidence:\n{evidence}"),
        (true, true) => String::new(),
    }
}

fn resolve_canonical_index(
    data: &seiza::objects::ObjectCatalogData,
    target_id: &str,
) -> Result<usize> {
    let exact = data
        .objects
        .iter()
        .enumerate()
        .filter(|(_, object)| object.metadata.id == target_id)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if exact.len() == 1 {
        return Ok(exact[0]);
    }
    if exact.len() > 1 {
        bail!("curation target {target_id} has duplicate canonical primary IDs");
    }
    let matches = data
        .objects
        .iter()
        .enumerate()
        .filter(|(_, object)| {
            object
                .metadata
                .alternate_ids
                .iter()
                .any(|id| id == target_id)
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        bail!(
            "curation target {target_id} resolves to {} canonical objects",
            matches.len()
        );
    }
    Ok(matches[0])
}

fn register_curated_facet(
    seen: &mut std::collections::HashSet<(String, String)>,
    target_id: &str,
    facet: &str,
) -> Result<()> {
    let key = (target_id.to_string(), facet.to_ascii_lowercase());
    if !seen.insert(key) {
        bail!("multiple curated selections for {target_id} facet {facet}");
    }
    Ok(())
}

fn replace_selection(
    selections: &mut Vec<seiza::objects::ObjectSelection>,
    selection: seiza::objects::ObjectSelection,
) {
    selections.retain(|existing| existing.facet != selection.facet);
    selections.push(selection);
}

fn apply_selected_facet(
    object: &mut seiza::objects::SkyObject,
    detail: &seiza::objects::ObjectDetails,
    selection: &seiza::objects::ObjectSelection,
) -> Result<()> {
    use seiza::objects::ObjectFacet;
    let source = selection.source_record_id.as_deref().map(|id| {
        detail
            .source_records
            .iter()
            .find(|record| record.id == id)
            .with_context(|| format!("selection references unknown source record {id}"))
    });
    let source = source.transpose()?;
    match selection.facet {
        ObjectFacet::PreferredIdentity => {
            let source = source.context("preferred identity requires source_record_id")?;
            let canonical_id = object.metadata.id.clone();
            object.name = source.object.name.clone();
            object.common_name = source.object.common_name.clone();
            object.metadata.source = source.source.clone();
            object.metadata.id = canonical_id;
        }
        ObjectFacet::PreferredPosition => {
            let source = source.context("preferred position requires source_record_id")?;
            object.ra = source.object.ra;
            object.dec = source.object.dec;
        }
        ObjectFacet::PreferredPhotometry => {
            let source = source.context("preferred photometry requires source_record_id")?;
            object.mag = source.object.mag;
        }
        ObjectFacet::PreferredClassification => {
            let source = source.context("preferred classification requires source_record_id")?;
            object.kind = source.object.kind;
        }
        ObjectFacet::PreferredGeometry => {
            let geometry_id = selection
                .geometry_id
                .as_deref()
                .context("preferred geometry requires geometry_id")?;
            let geometry = detail
                .geometries
                .iter()
                .find(|geometry| geometry.id == geometry_id)
                .with_context(|| format!("selection references unknown geometry {geometry_id}"))?;
            apply_geometry_to_hot_object(object, geometry)?;
        }
    }
    Ok(())
}

fn apply_geometry_to_hot_object(
    object: &mut seiza::objects::SkyObject,
    geometry: &seiza::objects::ObjectGeometry,
) -> Result<()> {
    match &geometry.data {
        seiza::objects::GeometryData::Point { ra_deg, dec_deg } => {
            object.ra = *ra_deg;
            object.dec = *dec_deg;
            object.major_arcmin = None;
            object.minor_arcmin = None;
            object.position_angle_deg = None;
        }
        seiza::objects::GeometryData::Ellipse {
            center_ra_deg,
            center_dec_deg,
            major_arcmin,
            minor_arcmin,
            position_angle_deg,
        } => {
            object.ra = *center_ra_deg;
            object.dec = *center_dec_deg;
            object.major_arcmin = Some(*major_arcmin);
            object.minor_arcmin = *minor_arcmin;
            object.position_angle_deg = *position_angle_deg;
        }
        seiza::objects::GeometryData::OutlineSet { contours, .. } => {
            let (ra, dec, radius_deg) = outline_bounding_cap(contours)?;
            object.ra = ra;
            object.dec = dec;
            object.major_arcmin = Some((radius_deg * 120.0) as f32);
            object.minor_arcmin = None;
            object.position_angle_deg = None;
        }
    }
    Ok(())
}

fn outline_bounding_cap(contours: &[seiza::objects::ObjectContour]) -> Result<(f64, f64, f64)> {
    let vertices = contours
        .iter()
        .flat_map(|contour| contour.vertices.iter().copied())
        .collect::<Vec<_>>();
    if vertices.is_empty() {
        bail!("outline contains no vertices");
    }
    let mut sum = [0.0f64; 3];
    for (ra, dec) in &vertices {
        let ra = ra.to_radians();
        let dec = dec.to_radians();
        sum[0] += dec.cos() * ra.cos();
        sum[1] += dec.cos() * ra.sin();
        sum[2] += dec.sin();
    }
    let norm = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
    if norm <= 1e-12 {
        bail!("outline has a degenerate spherical center");
    }
    let center = [sum[0] / norm, sum[1] / norm, sum[2] / norm];
    let ra = center[1].atan2(center[0]).to_degrees().rem_euclid(360.0);
    let dec = center[2].asin().to_degrees();
    let radius = vertices
        .iter()
        .map(|&(vertex_ra, vertex_dec)| {
            let vertex_ra = vertex_ra.to_radians();
            let vertex_dec = vertex_dec.to_radians();
            let dot = center[0] * vertex_dec.cos() * vertex_ra.cos()
                + center[1] * vertex_dec.cos() * vertex_ra.sin()
                + center[2] * vertex_dec.sin();
            dot.clamp(-1.0, 1.0).acos().to_degrees()
        })
        .fold(0.0, f64::max);
    Ok((ra, dec, radius))
}

fn parse_openngc_outline(content: &str) -> Result<Vec<seiza::objects::ObjectContour>> {
    let mut contours = Vec::new();
    let mut vertices = Vec::new();
    for line in content.lines().skip(1) {
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() < 4 || fields[0] != "line" {
            continue;
        }
        let ra = fields[2]
            .parse::<f64>()
            .with_context(|| format!("invalid OpenNGC outline RA in {line}"))?;
        let dec = fields[3]
            .parse::<f64>()
            .with_context(|| format!("invalid OpenNGC outline Dec in {line}"))?;
        vertices.push((ra, dec));
        if fields[1] == "*" {
            if vertices.len() < 2 {
                bail!("OpenNGC outline contour has fewer than two vertices");
            }
            contours.push(seiza::objects::ObjectContour {
                closed: true,
                vertices: std::mem::take(&mut vertices),
            });
        }
    }
    if !vertices.is_empty() {
        if vertices.len() < 2 {
            bail!("OpenNGC outline contour has fewer than two vertices");
        }
        contours.push(seiza::objects::ObjectContour {
            closed: false,
            vertices,
        });
    }
    if contours.is_empty() {
        bail!("OpenNGC outline has no contours");
    }
    Ok(contours)
}

fn parse_geometry_role(value: &str) -> Result<seiza::objects::GeometryRole> {
    use seiza::objects::GeometryRole;
    match value.trim().to_ascii_lowercase().as_str() {
        "catalog-extent" => Ok(GeometryRole::CatalogExtent),
        "preferred-render" => Ok(GeometryRole::PreferredRender),
        "fallback-extent" => Ok(GeometryRole::FallbackExtent),
        "brightness-level" => Ok(GeometryRole::BrightnessLevel),
        "component" => Ok(GeometryRole::Component),
        _ => bail!("unknown geometry role {value}"),
    }
}

fn parse_geometry_quality(value: &str) -> Result<seiza::objects::GeometryQuality> {
    use seiza::objects::GeometryQuality;
    match value.trim().to_ascii_lowercase().as_str() {
        "catalog" => Ok(GeometryQuality::Catalog),
        "curated" => Ok(GeometryQuality::Curated),
        "estimated" => Ok(GeometryQuality::Estimated),
        "derived" => Ok(GeometryQuality::Derived),
        _ => bail!("unknown geometry quality {value}"),
    }
}

fn parse_object_facet(value: &str) -> Result<seiza::objects::ObjectFacet> {
    use seiza::objects::ObjectFacet;
    match value.trim().to_ascii_lowercase().as_str() {
        "preferred-identity" => Ok(ObjectFacet::PreferredIdentity),
        "preferred-position" => Ok(ObjectFacet::PreferredPosition),
        "preferred-geometry" => Ok(ObjectFacet::PreferredGeometry),
        "preferred-photometry" => Ok(ObjectFacet::PreferredPhotometry),
        "preferred-classification" => Ok(ObjectFacet::PreferredClassification),
        _ => bail!("unknown object selection facet {value}"),
    }
}

fn parse_relation_kind(value: &str) -> Result<seiza::objects::ObjectRelationKind> {
    use seiza::objects::ObjectRelationKind;
    match value.trim().to_ascii_lowercase().as_str() {
        "same-as" => Ok(ObjectRelationKind::SameAs),
        "component-of" => Ok(ObjectRelationKind::ComponentOf),
        "parent-of" => Ok(ObjectRelationKind::ParentOf),
        "duplicate-of" => Ok(ObjectRelationKind::DuplicateOf),
        "catalog-alias" => Ok(ObjectRelationKind::CatalogAlias),
        _ => bail!("unknown object relation kind {value}"),
    }
}

fn recursive_files(root: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();
    let mut directories = vec![root.to_path_buf()];
    while let Some(directory) = directories.pop() {
        for entry in std::fs::read_dir(&directory)
            .with_context(|| format!("cannot read curation directory {}", directory.display()))?
        {
            let path = entry?.path();
            if path.is_dir() {
                if path.file_name().and_then(|name| name.to_str()) != Some(".git") {
                    directories.push(path);
                }
            } else if path.is_file()
                && path.file_name().and_then(|name| name.to_str()) != Some(".seiza-revision")
            {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

fn add_tycho_identifiers(
    builder: &mut Option<StarIdentifierCatalogBuilder>,
    star: ParsedTychoStar,
) -> Result<()> {
    let Some(builder) = builder else {
        return Ok(());
    };
    builder.add(star.tyc, star.ra, star.dec, star.mag)?;
    if let Some(hip) = star.hip {
        builder.add(hip, star.ra, star.dec, star.mag)?;
    }
    Ok(())
}

#[derive(Debug, Default)]
struct StellarIdentifierStats {
    bright: usize,
    variables: usize,
    doubles: usize,
    proper_names: usize,
}

type BrightStarPositions = std::collections::HashMap<u32, (f64, f64, Option<f32>)>;

fn add_stellar_identifier_sources(
    builder: &mut StarIdentifierCatalogBuilder,
    input: &Path,
    epoch: f64,
) -> Result<StellarIdentifierStats> {
    for name in ["bsc-identifiers.tsv", "gcvs.tsv", "wds.tsv", "IAU-CSN.txt"] {
        if !input.join(name).exists() {
            bail!(
                "{} is missing {}; run download-data star-identifiers first",
                input.display(),
                name
            );
        }
    }

    let mut stats = StellarIdentifierStats::default();
    let bright_positions = add_bright_star_identifiers(
        builder,
        &input.join("bsc-identifiers.tsv"),
        epoch,
        &mut stats,
    )?;
    add_gcvs_identifiers(builder, &input.join("gcvs.tsv"), epoch, &mut stats)?;
    add_wds_identifiers(builder, &input.join("wds.tsv"), epoch, &mut stats)?;

    let file = std::fs::File::open(input.join("IAU-CSN.txt"))?;
    for line in BufReader::new(file).lines() {
        let line = line?;
        let Some(star) = parse_iau_csn_line(&line) else {
            continue;
        };
        let hr = iau_csn_hr(&line);
        let (ra, dec, mag) = hr
            .and_then(|number| bright_positions.get(&number).copied())
            .unwrap_or((star.ra, star.dec, star.mag));
        let stable_id = hr
            .map(|number| format!("hr:{number}"))
            .unwrap_or_else(|| star.metadata.id.clone());
        builder.add_name(
            StarNameCatalog::IauCatalogOfStarNames,
            StarNameKind::ProperName,
            &star.name,
            &stable_id,
            "",
            ra,
            dec,
            mag,
        )?;
        stats.proper_names += 1;
    }
    Ok(stats)
}

fn add_bright_star_identifiers(
    builder: &mut StarIdentifierCatalogBuilder,
    path: &Path,
    epoch: f64,
    stats: &mut StellarIdentifierStats,
) -> Result<BrightStarPositions> {
    let mut positions = std::collections::HashMap::new();
    let file = std::fs::File::open(path)?;
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.starts_with('#') {
            continue;
        }
        let raw_fields = line.split('\t').collect::<Vec<_>>();
        let fields = raw_fields
            .iter()
            .map(|field| field.trim())
            .collect::<Vec<_>>();
        if fields.len() < 13 {
            continue;
        }
        let (Ok(ra), Ok(dec), Ok(hr), Ok(mag)) = (
            fields[0].parse::<f64>(),
            fields[1].parse::<f64>(),
            fields[2].parse::<u32>(),
            fields[10].parse::<f32>(),
        ) else {
            continue;
        };
        let (ra, dec) = propagate_catalog_position(
            ra,
            dec,
            fields[11].parse().ok(),
            fields[12].parse().ok(),
            1.0,
            epoch,
        );
        let stable_id = format!("hr:{hr}");
        positions.insert(hr, (ra, dec, Some(mag)));
        for identifier in [
            Some(StarIdentifier::HarvardRevised(hr)),
            fields[4]
                .parse::<u32>()
                .ok()
                .map(StarIdentifier::HenryDraper),
            fields[5].parse::<u32>().ok().map(StarIdentifier::Sao),
            fields[6].parse::<u32>().ok().map(StarIdentifier::Fk5),
        ]
        .into_iter()
        .flatten()
        {
            builder.add(identifier, ra, dec, mag)?;
            stats.bright += 1;
        }

        for bayer_flamsteed in bright_star_designations(raw_fields[3]) {
            builder.add_name(
                StarNameCatalog::BrightStarCatalog,
                StarNameKind::BayerFlamsteed,
                &bayer_flamsteed,
                &stable_id,
                "",
                ra,
                dec,
                Some(mag),
            )?;
            stats.bright += 1;
        }
        if !fields[7].is_empty() {
            let components = fields[8];
            let designation = collapse_whitespace(&format!("ADS {} {}", fields[7], components));
            builder.add_name(
                StarNameCatalog::BrightStarCatalog,
                StarNameKind::DoubleStar,
                &designation,
                &stable_id,
                components,
                ra,
                dec,
                Some(mag),
            )?;
            stats.bright += 1;
        }
        if !fields[9].is_empty() && !fields[9].eq_ignore_ascii_case("Var?") {
            builder.add_name(
                StarNameCatalog::BrightStarCatalog,
                StarNameKind::VariableStar,
                fields[9],
                &stable_id,
                "",
                ra,
                dec,
                Some(mag),
            )?;
            stats.bright += 1;
        }
    }
    Ok(positions)
}

fn add_gcvs_identifiers(
    builder: &mut StarIdentifierCatalogBuilder,
    path: &Path,
    epoch: f64,
    stats: &mut StellarIdentifierStats,
) -> Result<()> {
    let file = std::fs::File::open(path)?;
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.starts_with('#') {
            continue;
        }
        let fields = line.split('\t').map(str::trim).collect::<Vec<_>>();
        if fields.len() < 14 || !fields[13].is_empty() {
            continue;
        }
        let (Ok(ra), Ok(dec)) = (fields[0].parse::<f64>(), fields[1].parse::<f64>()) else {
            continue;
        };
        let designation = collapse_whitespace(fields[2]);
        if designation.is_empty() {
            continue;
        }
        let (ra, dec) = propagate_catalog_position(
            ra,
            dec,
            fields[10].parse().ok(),
            fields[11].parse().ok(),
            1.0,
            epoch,
        );
        let stable_id = format!("gcvs:{}", stable_name_fragment(&designation));
        let detail = gcvs_detail(&fields);
        builder.add_name(
            StarNameCatalog::GeneralCatalogOfVariableStars,
            StarNameKind::VariableStar,
            &designation,
            &stable_id,
            &detail,
            ra,
            dec,
            parse_identifier_mag(fields[4]),
        )?;
        stats.variables += 1;
    }
    Ok(())
}

fn add_wds_identifiers(
    builder: &mut StarIdentifierCatalogBuilder,
    path: &Path,
    epoch: f64,
    stats: &mut StellarIdentifierStats,
) -> Result<()> {
    let file = std::fs::File::open(path)?;
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.starts_with('#') {
            continue;
        }
        let fields = line.split('\t').map(str::trim).collect::<Vec<_>>();
        if fields.len() < 11 {
            continue;
        }
        let (Ok(ra), Ok(dec)) = (fields[0].parse::<f64>(), fields[1].parse::<f64>()) else {
            continue;
        };
        if fields[2].is_empty() || fields[3].is_empty() {
            continue;
        }
        let (ra, dec) = propagate_catalog_position(
            ra,
            dec,
            fields[9].parse().ok(),
            fields[10].parse().ok(),
            0.001,
            epoch,
        );
        let discoverer = collapse_whitespace(fields[3]);
        let components = if fields[4].is_empty() {
            "AB"
        } else {
            fields[4]
        };
        let stable_id = format!(
            "wds:{}:{}:{}",
            fields[2],
            stable_name_fragment(&discoverer),
            stable_wds_component(components)
        );
        let mag = parse_identifier_mag(fields[5]);
        let detail = wds_detail(&fields, components);
        for designation in [
            format!("WDS {}", fields[2]),
            collapse_whitespace(&format!("{discoverer} {components}")),
        ] {
            builder.add_name(
                StarNameCatalog::WashingtonDoubleStar,
                StarNameKind::DoubleStar,
                &designation,
                &stable_id,
                &detail,
                ra,
                dec,
                mag,
            )?;
            stats.doubles += 1;
        }
    }
    Ok(())
}

fn propagate_catalog_position(
    ra: f64,
    dec: f64,
    pm_ra: Option<f64>,
    pm_dec: Option<f64>,
    arcsec_per_unit: f64,
    epoch: f64,
) -> (f64, f64) {
    let dt = epoch - 2000.0;
    let cos_dec = dec.to_radians().cos().max(1e-6);
    let ra =
        (ra + pm_ra.unwrap_or(0.0) * arcsec_per_unit * dt / 3600.0 / cos_dec).rem_euclid(360.0);
    let dec = (dec + pm_dec.unwrap_or(0.0) * arcsec_per_unit * dt / 3600.0).clamp(-90.0, 90.0);
    (ra, dec)
}

fn collapse_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn bright_star_designations(value: &str) -> Vec<String> {
    let mut names = Vec::new();
    let Some(flamsteed) = value.get(0..3).map(str::trim) else {
        let name = collapse_whitespace(value);
        return (!name.is_empty()).then_some(name).into_iter().collect();
    };
    let greek = value.get(3..6).map(str::trim).unwrap_or("");
    let component = value.get(6..7).map(str::trim).unwrap_or("");
    let constellation = value.get(7..10).map(str::trim).unwrap_or("");
    if constellation.is_empty() {
        return names;
    }
    if !flamsteed.is_empty() {
        names.push(format!("{flamsteed} {constellation}"));
    }
    if !greek.is_empty() {
        let abbreviation = format!("{greek}{component} {constellation}");
        names.push(abbreviation);
        if let Some((full, symbol)) = greek_name(greek) {
            names.push(format!("{full}{component} {constellation}"));
            names.push(format!("{symbol}{component} {constellation}"));
        }
    }
    names.sort();
    names.dedup();
    names
}

fn greek_name(abbreviation: &str) -> Option<(&'static str, &'static str)> {
    match abbreviation {
        "Alp" => Some(("Alpha", "α")),
        "Bet" => Some(("Beta", "β")),
        "Gam" => Some(("Gamma", "γ")),
        "Del" => Some(("Delta", "δ")),
        "Eps" => Some(("Epsilon", "ε")),
        "Zet" => Some(("Zeta", "ζ")),
        "Eta" => Some(("Eta", "η")),
        "The" => Some(("Theta", "θ")),
        "Iot" => Some(("Iota", "ι")),
        "Kap" => Some(("Kappa", "κ")),
        "Lam" => Some(("Lambda", "λ")),
        "Mu" => Some(("Mu", "μ")),
        "Nu" => Some(("Nu", "ν")),
        "Xi" => Some(("Xi", "ξ")),
        "Omi" => Some(("Omicron", "ο")),
        "Pi" => Some(("Pi", "π")),
        "Rho" => Some(("Rho", "ρ")),
        "Sig" => Some(("Sigma", "σ")),
        "Tau" => Some(("Tau", "τ")),
        "Ups" => Some(("Upsilon", "υ")),
        "Phi" => Some(("Phi", "φ")),
        "Chi" => Some(("Chi", "χ")),
        "Psi" => Some(("Psi", "ψ")),
        "Ome" => Some(("Omega", "ω")),
        _ => None,
    }
}

fn iau_csn_hr(line: &str) -> Option<u32> {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    tokens
        .windows(2)
        .find(|pair| pair[0] == "HR")
        .and_then(|pair| pair[1].parse().ok())
}

fn stable_name_fragment(value: &str) -> String {
    normalize_star_name(value)
}

fn stable_wds_component(value: &str) -> String {
    let mut result = String::new();
    for byte in value.to_ascii_uppercase().bytes() {
        if byte.is_ascii_alphanumeric() || byte == b'-' {
            result.push(byte as char);
        } else {
            result.push_str(&format!("%{byte:02X}"));
        }
    }
    result
}

fn parse_identifier_mag(value: &str) -> Option<f32> {
    value
        .parse::<f32>()
        .ok()
        .filter(|magnitude| (-3.0..62.0).contains(magnitude))
}

fn gcvs_detail(fields: &[&str]) -> String {
    let mut parts = Vec::new();
    if !fields[3].is_empty() {
        parts.push(fields[3].to_string());
    }
    let minimum = parse_identifier_mag(fields[6]);
    let band = if fields[7].is_empty() {
        fields[8]
    } else {
        fields[7]
    };
    if fields[5].contains('(') {
        if let Some(amplitude) = minimum {
            parts.push(format!("amplitude={amplitude:.3}{band}"));
        }
    } else if let (Some(maximum), Some(minimum)) = (parse_identifier_mag(fields[4]), minimum) {
        parts.push(format!(
            "range={maximum:.3}-{}{minimum:.3}{band}",
            fields[5]
        ));
    }
    if let Ok(period) = fields[9].parse::<f64>()
        && period.is_finite()
        && period > 0.0
    {
        parts.push(format!("period={period}d"));
    }
    parts.join("; ")
}

fn wds_detail(fields: &[&str], components: &str) -> String {
    let mut parts = vec![components.to_string()];
    if let Ok(separation) = fields[8].parse::<f32>() {
        parts.push(format!("sep={separation}arcsec"));
    }
    if let Ok(position_angle) = fields[7].parse::<u16>() {
        parts.push(format!("pa={position_angle}deg"));
    }
    if let Some(secondary_mag) = parse_identifier_mag(fields[6]) {
        parts.push(format!("mag2={secondary_mag:.2}"));
    }
    parts.join("; ")
}

/// Build a transient catalog from the Rochester "Latest Supernovae"
/// active list. Each row becomes a Transient object whose common name
/// carries the type, latest magnitude, discovery date, and host.
pub fn build_transients(input: &Path, output: &Path) -> Result<()> {
    use seiza::objects::{ObjectCatalog, ObjectKind, ObjectMetadata, SkyObject};

    let path = input.join("snactive.html");
    // The page contains Latin-1 discoverer names; decode lossily
    let bytes = std::fs::read(&path).with_context(|| {
        format!(
            "cannot read {}; run download-data transients",
            path.display()
        )
    })?;
    let content = String::from_utf8_lossy(&bytes);

    let mut objects = Vec::new();
    for row in content.split("<tr>").skip(1) {
        let row = row.split("</tr>").next().unwrap_or("");
        let cells: Vec<String> = row
            .split("</td>")
            .map(|cell| {
                // Strip tags within the cell
                let mut text = String::new();
                let mut in_tag = false;
                for c in cell.chars() {
                    match c {
                        '<' => in_tag = true,
                        '>' => in_tag = false,
                        c if !in_tag => text.push(c),
                        _ => {}
                    }
                }
                text.trim().to_string()
            })
            .collect();
        if cells.len() < 12 {
            continue;
        }

        let designation = &cells[0];
        if designation.is_empty() {
            continue;
        }
        let (Some(ra), Some(dec)) = (
            parse_sexagesimal(&cells[2]).map(|h| h * 15.0),
            parse_sexagesimal(&cells[3]),
        ) else {
            continue;
        };
        let host = &cells[1];
        let mag: Option<f32> = cells[5].trim_end_matches('*').parse().ok();
        let sn_type = &cells[7];
        let discovered = &cells[11];

        let name = if designation.starts_with("AT") || designation.starts_with("SN") {
            designation.clone()
        } else {
            format!("SN {designation}")
        };
        let mut details = Vec::new();
        if !sn_type.is_empty() && sn_type != "unk" {
            details.push(format!("type {sn_type}"));
        }
        if !discovered.is_empty() {
            details.push(format!("disc. {discovered}"));
        }
        if !host.is_empty() && host != "none" {
            details.push(format!("in {host}"));
        }

        objects.push(SkyObject {
            kind: ObjectKind::Transient,
            ra,
            dec,
            mag,
            major_arcmin: None,
            minor_arcmin: None,
            position_angle_deg: None,
            name,
            common_name: details.join(", "),
            metadata: ObjectMetadata {
                id: format!("rochester:{designation}"),
                source: "Rochester Latest Supernovae".to_string(),
                aliases: Vec::new(),
                parent_ids: Vec::new(),
                alternate_ids: Vec::new(),
                alternate_sources: Vec::new(),
            },
        });
    }

    if objects.is_empty() {
        bail!("no transients parsed from {}", path.display());
    }
    let catalog = ObjectCatalog::new(objects);
    let count = catalog.len();
    catalog.write_to(output)?;
    println!("{count} transients written to {}", output.display());
    Ok(())
}

/// Cheap spatial hash for deduplicating objects by position.
struct PositionDedup {
    cells: std::collections::HashMap<(i32, i32), Vec<(f64, f64)>>,
}

impl PositionDedup {
    fn new() -> Self {
        Self {
            cells: std::collections::HashMap::new(),
        }
    }

    fn cell(ra: f64, dec: f64) -> (i32, i32) {
        ((ra * 10.0) as i32, (dec * 10.0) as i32)
    }

    fn insert(&mut self, ra: f64, dec: f64) {
        self.cells
            .entry(Self::cell(ra, dec))
            .or_default()
            .push((ra, dec));
    }

    fn near(&self, ra: f64, dec: f64, radius_deg: f64) -> bool {
        let (cx, cy) = Self::cell(ra, dec);
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(points) = self.cells.get(&(cx + dx, cy + dy))
                    && points.iter().any(|&(r, d)| {
                        seiza::catalog::angular_separation_deg(ra, dec, r, d) <= radius_deg
                    })
                {
                    return true;
                }
            }
        }
        false
    }
}

#[derive(Debug, Clone, Copy)]
struct ParsedTychoStar {
    ra: f64,
    dec: f64,
    mag: f32,
    tyc: StarIdentifier,
    hip: Option<StarIdentifier>,
}

/// Parse one fixed-width Tycho-2 record at `epoch`, retaining its TYC and
/// optional Hipparcos identifiers for the offline lookup sidecar.
fn parse_tycho2_line(line: &str, epoch: f64) -> Option<ParsedTychoStar> {
    // Byte ranges from the CDS ReadMe are 1-indexed inclusive
    let field =
        |from: usize, to: usize| -> &str { line.get(from - 1..to).map(str::trim).unwrap_or("") };

    // VT magnitude, falling back to BT
    let mag: f32 = field(124, 129)
        .parse()
        .or_else(|_| field(111, 116).parse())
        .ok()?;
    let tyc = StarIdentifier::Tycho2 {
        region: field(1, 4).parse().ok()?,
        number: field(6, 10).parse().ok()?,
        component: field(12, 12).parse().ok()?,
    };
    let hip = field(143, 148)
        .parse::<u32>()
        .ok()
        .map(StarIdentifier::Hipparcos);

    // Mean position (may be absent when pflag is X), else observed position
    let (ra, dec) =
        if let (Ok(ra), Ok(dec)) = (field(16, 27).parse::<f64>(), field(29, 40).parse::<f64>()) {
            let dt = epoch - 2000.0;
            // mas/yr; pmRA includes cos(dec)
            let pm_ra: f64 = field(42, 48).parse().unwrap_or(0.0);
            let pm_dec: f64 = field(50, 56).parse().unwrap_or(0.0);
            let cos_dec = dec.to_radians().cos().max(1e-6);
            (
                (ra + pm_ra * dt / 3_600_000.0 / cos_dec).rem_euclid(360.0),
                (dec + pm_dec * dt / 3_600_000.0).clamp(-90.0, 90.0),
            )
        } else {
            (
                field(153, 164).parse::<f64>().ok()?,
                field(166, 177).parse::<f64>().ok()?,
            )
        };
    Some(ParsedTychoStar {
        ra,
        dec,
        mag,
        tyc,
        hip,
    })
}

/// Build a star tile file from an ASTAP `.1476` star database directory
/// (e.g. D80, Gaia DR3). The format is documented in ASTAP's
/// unit_star_database.pas: each of the 1476 sky-area files has a 110-byte
/// header (text description, record size in the final byte) followed by
/// 5-byte records; `FF FF FF` section headers carry the dec high byte
/// (offset +128) and the section magnitude ((byte - 16) / 10).
pub fn build_astap(
    input: &Path,
    output: &Path,
    epoch: f64,
    max_mag: f32,
    bands: u32,
) -> Result<()> {
    let mut parts: Vec<_> = std::fs::read_dir(input)
        .with_context(|| format!("cannot read {}", input.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "1476"))
        .collect();
    parts.sort();
    if parts.is_empty() {
        bail!("no .1476 files found in {}", input.display());
    }

    let mut builder = TileSetBuilder::new(
        bands,
        epoch,
        "Gaia DR3 via the ASTAP star database (ESA Gaia DPAC; CC BY-SA 3.0 IGO attribution)",
    );
    let mut too_faint = 0u64;

    for part in &parts {
        let mut reader = BufReader::with_capacity(
            1 << 20,
            std::fs::File::open(part).with_context(|| format!("cannot open {}", part.display()))?,
        );
        let mut header = [0u8; 110];
        std::io::Read::read_exact(&mut reader, &mut header)
            .with_context(|| format!("{} is too short for a header", part.display()))?;
        let record_size = header[109];
        if record_size != 5 {
            bail!(
                "{}: unsupported record size {record_size} (only the 5-byte \
                 1476 format is supported)",
                part.display()
            );
        }

        let mut record = [0u8; 5];
        let mut dec9: i32 = 0;
        let mut mag: f32 = 0.0;
        loop {
            match std::io::Read::read_exact(&mut reader, &mut record) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            if record[0] == 0xFF && record[1] == 0xFF && record[2] == 0xFF {
                dec9 = record[3] as i32 - 128;
                mag = (record[4] as f32 - 16.0) / 10.0;
                continue;
            }
            if mag > max_mag {
                too_faint += 1;
                continue;
            }
            let ra = (record[0] as f64 + record[1] as f64 * 256.0 + record[2] as f64 * 65536.0)
                * 360.0
                / ((1u32 << 24) - 1) as f64;
            let dec_int = record[3] as i32 + record[4] as i32 * 256 + dec9 * 65536;
            let dec = dec_int as f64 * 90.0 / ((128 * 65536) - 1) as f64;
            builder.add(ra, dec, mag);
        }
    }

    let count = builder.star_count();
    builder.write_to(output)?;
    println!(
        "{} stars written to {} (epoch {epoch}, {} fainter than {max_mag})",
        count,
        output.display(),
        too_faint
    );
    Ok(())
}

/// Parse one supplement-1/2 record: positions are ICRS at epoch J1991.25.
fn parse_tycho2_suppl_line(line: &str, epoch: f64) -> Option<ParsedTychoStar> {
    let field =
        |from: usize, to: usize| -> &str { line.get(from - 1..to).map(str::trim).unwrap_or("") };

    let mag: f32 = field(97, 102)
        .parse()
        .or_else(|_| field(84, 89).parse())
        .ok()?;
    let tyc = StarIdentifier::Tycho2 {
        region: field(1, 4).parse().ok()?,
        number: field(6, 10).parse().ok()?,
        component: field(12, 12).parse().ok()?,
    };
    let hip = field(116, 121)
        .parse::<u32>()
        .ok()
        .map(StarIdentifier::Hipparcos);
    let ra = field(16, 27).parse::<f64>().ok()?;
    let dec = field(29, 40).parse::<f64>().ok()?;

    let dt = epoch - 1991.25;
    let pm_ra: f64 = field(42, 48).parse().unwrap_or(0.0);
    let pm_dec: f64 = field(50, 56).parse().unwrap_or(0.0);
    let cos_dec = dec.to_radians().cos().max(1e-6);
    let ra = (ra + pm_ra * dt / 3_600_000.0 / cos_dec).rem_euclid(360.0);
    let dec = (dec + pm_dec * dt / 3_600_000.0).clamp(-90.0, 90.0);
    Some(ParsedTychoStar {
        ra,
        dec,
        mag,
        tyc,
        hip,
    })
}

/// Build an object catalog from OpenNGC, selected VizieR tables, and the IAU
/// star-name list, whichever are present in `input`.
pub fn build_objects(
    input: &Path,
    output: &Path,
    source_manifest: Option<&Path>,
    curation_dir: Option<&Path>,
) -> Result<()> {
    use seiza::objects::{
        ObjectCatalog, ObjectKind, ObjectMetadata, ObjectProperty, ObjectSourceRecord, SkyObject,
    };

    let mut objects = Vec::new();
    let mut source_records = Vec::new();
    let mut sources = 0;

    for name in ["NGC.csv", "addendum.csv"] {
        let path = input.join(name);
        if !path.exists() {
            continue;
        }
        sources += 1;
        let content = std::fs::read_to_string(&path)?;
        let mut lines = content.lines();
        let headers = lines
            .next()
            .map(|line| line.split(';').map(str::trim).collect::<Vec<_>>())
            .unwrap_or_default();
        for line in lines {
            if let Some(object) = parse_openngc_line(line) {
                let properties = headers
                    .iter()
                    .zip(line.split(';'))
                    .filter_map(|(name, value)| {
                        let value = value.trim();
                        (!name.is_empty() && !value.is_empty()).then(|| ObjectProperty {
                            name: (*name).to_string(),
                            value: value.to_string(),
                            unit: None,
                        })
                    })
                    .collect();
                source_records.push(ObjectSourceRecord {
                    id: object.metadata.id.clone(),
                    source: object.metadata.source.clone(),
                    object: object.clone(),
                    properties,
                });
                objects.push(object);
            }
        }
    }

    // Every later source is checked against explicit designations and stable
    // IDs already ingested. Positional proximity remains only a fallback for
    // catalogs that do not provide a usable cross-identification.
    let mut identity_index = ObjectIdentityIndex::new(&objects);
    let mut merge_stats = ObjectMergeStats::default();

    for (file, prefix, kind, source, id_prefix) in [
        (
            "sh2.tsv",
            "Sh2-",
            ObjectKind::HiiRegion,
            "VizieR VII/20/catalog",
            "vizier:VII/20:Sh2-",
        ),
        (
            "barnard.tsv",
            "B",
            ObjectKind::DarkNebula,
            "VizieR VII/220A/barnard",
            "vizier:VII/220A:B",
        ),
    ] {
        let path = input.join(file);
        if !path.exists() {
            continue;
        }
        sources += 1;
        let content = std::fs::read_to_string(&path)?;
        for line in content.lines() {
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split('\t').map(str::trim).collect();
            if fields.len() < 3 {
                continue;
            }
            let (Ok(ra), Ok(dec)) = (fields[0].parse::<f64>(), fields[1].parse::<f64>()) else {
                continue; // column header / units / separator rows
            };
            let Ok(number) = fields[2].parse::<u32>() else {
                continue;
            };
            let diam: Option<f32> = fields.get(3).and_then(|d| d.parse().ok());
            let object = SkyObject {
                kind,
                ra,
                dec,
                mag: None,
                major_arcmin: diam.filter(|d| *d > 0.0),
                minor_arcmin: None,
                position_angle_deg: None,
                name: format!("{prefix}{number}"),
                common_name: String::new(),
                metadata: ObjectMetadata {
                    id: format!("{id_prefix}{number}"),
                    source: source.to_string(),
                    aliases: Vec::new(),
                    parent_ids: Vec::new(),
                    alternate_ids: Vec::new(),
                    alternate_sources: Vec::new(),
                },
            };
            identity_index.ingest(
                &mut objects,
                &mut source_records,
                object,
                catalog_properties(
                    &fields,
                    &["ra_deg", "dec_deg", "catalog_number", "diameter_arcmin"],
                ),
                &mut merge_stats,
            );
        }
    }

    // Generic VizieR TSV sources: (file, parse into SkyObject)
    let mut grid_dedup = PositionDedup::new();

    for (file, kind, prefix, source, id_prefix) in [
        (
            "ugc.tsv",
            ObjectKind::Galaxy,
            "UGC ",
            "VizieR VII/26D/catalog",
            "vizier:VII/26D:UGC",
        ),
        (
            "ldn.tsv",
            ObjectKind::DarkNebula,
            "LDN ",
            "VizieR VII/7A/ldn",
            "vizier:VII/7A:LDN",
        ),
        (
            "vdb.tsv",
            ObjectKind::Nebula,
            "vdB ",
            "VizieR VII/21/catalog",
            "vizier:VII/21:VdB",
        ),
    ] {
        let path = input.join(file);
        if !path.exists() {
            continue;
        }
        sources += 1;
        let content = std::fs::read_to_string(&path)?;
        for line in content.lines() {
            let fields: Vec<&str> = line.split('\t').map(str::trim).collect();
            if line.starts_with('#') || fields.len() < 3 {
                continue;
            }
            let (Ok(ra), Ok(dec)) = (fields[0].parse::<f64>(), fields[1].parse::<f64>()) else {
                continue;
            };
            let Ok(number) = fields[2].parse::<u32>() else {
                continue;
            };
            let suffix = if file == "ugc.tsv" {
                fields.get(3).copied().unwrap_or("")
            } else {
                ""
            };
            let (major, minor, pa) = match file {
                "ugc.tsv" => (
                    fields.get(4).and_then(|v| v.parse::<f32>().ok()),
                    fields.get(5).and_then(|v| v.parse::<f32>().ok()),
                    fields.get(6).and_then(|v| v.parse::<f32>().ok()),
                ),
                // LDN publishes an area in square degrees
                "ldn.tsv" => (
                    fields
                        .get(3)
                        .and_then(|v| v.parse::<f64>().ok())
                        .map(|area| (2.0 * (area / std::f64::consts::PI).sqrt() * 60.0) as f32),
                    None,
                    None,
                ),
                // vdB publishes a max radius in arcminutes
                _ => (
                    fields
                        .get(3)
                        .and_then(|v| v.parse::<f32>().ok())
                        .map(|r| r * 2.0),
                    None,
                    None,
                ),
            };
            let object = SkyObject {
                kind,
                ra,
                dec,
                mag: None,
                major_arcmin: major.filter(|v| *v > 0.0),
                minor_arcmin: minor.filter(|v| *v > 0.0),
                position_angle_deg: pa,
                name: format!("{prefix}{number}{suffix}"),
                common_name: String::new(),
                metadata: ObjectMetadata {
                    id: format!("{id_prefix}{number}{suffix}"),
                    source: source.to_string(),
                    aliases: Vec::new(),
                    parent_ids: Vec::new(),
                    alternate_ids: Vec::new(),
                    alternate_sources: Vec::new(),
                },
            };
            identity_index.ingest(
                &mut objects,
                &mut source_records,
                object,
                catalog_properties(
                    &fields,
                    match file {
                        "ugc.tsv" => &[
                            "ra_deg",
                            "dec_deg",
                            "catalog_number",
                            "suffix",
                            "major_arcmin",
                            "minor_arcmin",
                            "position_angle_deg",
                        ][..],
                        "ldn.tsv" => &["ra_deg", "dec_deg", "catalog_number", "area_sq_deg"][..],
                        _ => &["ra_deg", "dec_deg", "catalog_number", "max_radius_arcmin"][..],
                    },
                ),
                &mut merge_stats,
            );
        }
    }

    let csn = input.join("IAU-CSN.txt");
    if csn.exists() {
        sources += 1;
        let content = std::fs::read_to_string(&csn)?;
        for line in content.lines() {
            if let Some(object) = parse_iau_csn_line(line) {
                identity_index.ingest(
                    &mut objects,
                    &mut source_records,
                    object,
                    vec![ObjectProperty {
                        name: "raw_record".into(),
                        value: line.into(),
                        unit: None,
                    }],
                    &mut merge_stats,
                );
            }
        }
    }

    // PGC/HyperLEDA galaxies: keep those with D25 >= 0.4 arcmin, dedup
    // against galaxies already present from NGC/IC/UGC
    for o in &objects {
        if o.kind == ObjectKind::Galaxy {
            grid_dedup.insert(o.ra, o.dec);
        }
    }
    let pgc = input.join("pgc.tsv");
    if pgc.exists() {
        sources += 1;
        let content = std::fs::read_to_string(&pgc)?;
        for line in content.lines() {
            let fields: Vec<&str> = line.split('\t').map(str::trim).collect();
            if line.starts_with('#') || fields.len() < 4 {
                continue;
            }
            let (Ok(ra), Ok(dec)) = (fields[0].parse::<f64>(), fields[1].parse::<f64>()) else {
                continue;
            };
            let Ok(number) = fields[2].parse::<u32>() else {
                continue;
            };
            // logD25 is log10 of the diameter in 0.1-arcmin units
            let Some(major) = fields
                .get(3)
                .and_then(|v| v.parse::<f64>().ok())
                .map(|log_d| 10f64.powf(log_d) * 0.1)
            else {
                continue;
            };
            if major < 0.4 {
                continue;
            }
            let minor = fields
                .get(4)
                .and_then(|v| v.parse::<f64>().ok())
                .map(|log_r| major / 10f64.powf(log_r));
            let object = SkyObject {
                kind: ObjectKind::Galaxy,
                ra,
                dec,
                mag: None,
                major_arcmin: Some(major as f32),
                minor_arcmin: minor.map(|m| m as f32),
                position_angle_deg: fields.get(5).and_then(|v| v.parse().ok()),
                name: format!("PGC {number}"),
                common_name: String::new(),
                metadata: ObjectMetadata {
                    id: format!("vizier:VII/237:PGC{number}"),
                    source: "VizieR VII/237/pgc".to_string(),
                    aliases: Vec::new(),
                    parent_ids: Vec::new(),
                    alternate_ids: Vec::new(),
                    alternate_sources: Vec::new(),
                },
            };
            let matches = identity_index.matching_indices(&objects, &object);
            if matches.is_empty() && grid_dedup.near(ra, dec, 30.0 / 3600.0) {
                continue;
            }
            source_records.push(ObjectSourceRecord {
                id: object.metadata.id.clone(),
                source: object.metadata.source.clone(),
                object: object.clone(),
                properties: catalog_properties(
                    &fields,
                    &[
                        "ra_deg",
                        "dec_deg",
                        "pgc",
                        "log_d25",
                        "log_r25",
                        "position_angle_deg",
                    ],
                ),
            });
            identity_index.merge_or_add_with_matches(
                &mut objects,
                object,
                matches,
                &mut merge_stats,
            );
        }
    }

    // Bright Star Catalogue: HD-numbered naked-eye stars. IAU-named stars
    // are already present, so skip BSC entries landing on one.
    for o in &objects {
        if o.kind == ObjectKind::Star {
            grid_dedup.insert(o.ra, o.dec);
        }
    }
    let bsc = input.join("bsc.tsv");
    if bsc.exists() {
        sources += 1;
        let content = std::fs::read_to_string(&bsc)?;
        for line in content.lines() {
            let fields: Vec<&str> = line.split('\t').map(str::trim).collect();
            if line.starts_with('#') || fields.len() < 3 {
                continue;
            }
            let (Ok(ra), Ok(dec)) = (fields[0].parse::<f64>(), fields[1].parse::<f64>()) else {
                continue;
            };
            let Ok(hd) = fields[2].parse::<u32>() else {
                continue;
            };
            if grid_dedup.near(ra, dec, 120.0 / 3600.0) {
                continue;
            }
            let bayer = fields.get(3).copied().unwrap_or("").trim().to_string();
            let object = SkyObject {
                kind: ObjectKind::Star,
                ra,
                dec,
                mag: fields.get(4).and_then(|v| v.parse().ok()),
                major_arcmin: None,
                minor_arcmin: None,
                position_angle_deg: None,
                name: format!("HD {hd}"),
                common_name: bayer,
                metadata: ObjectMetadata {
                    id: format!("vizier:V/50:HD{hd}"),
                    source: "VizieR V/50/catalog".to_string(),
                    aliases: Vec::new(),
                    parent_ids: Vec::new(),
                    alternate_ids: Vec::new(),
                    alternate_sources: Vec::new(),
                },
            };
            identity_index.ingest(
                &mut objects,
                &mut source_records,
                object,
                catalog_properties(
                    &fields,
                    &["ra_deg", "dec_deg", "hd", "name", "visual_magnitude"],
                ),
                &mut merge_stats,
            );
        }
    }

    // Green's Galactic supernova remnants: whole-remnant ellipses that
    // complement the NGC/IC filament entries; skip any that OpenNGC
    // already carries as an SNR at the same position (e.g. the Crab)
    let mut snr_dedup = PositionDedup::new();
    for o in &objects {
        if o.kind == ObjectKind::SupernovaRemnant {
            snr_dedup.insert(o.ra, o.dec);
        }
    }
    let snr = input.join("snr.tsv");
    if snr.exists() {
        sources += 1;
        let content = std::fs::read_to_string(&snr)?;
        for line in content.lines() {
            let fields: Vec<&str> = line.split('\t').map(str::trim).collect();
            if line.starts_with('#') || fields.len() < 3 {
                continue;
            }
            let (Ok(ra), Ok(dec)) = (fields[0].parse::<f64>(), fields[1].parse::<f64>()) else {
                continue;
            };
            let designation = fields[2];
            if !designation.starts_with('G') || snr_dedup.near(ra, dec, 120.0 / 3600.0) {
                continue;
            }
            let object = SkyObject {
                kind: ObjectKind::SupernovaRemnant,
                ra,
                dec,
                mag: None,
                major_arcmin: fields.get(3).and_then(|v| v.parse().ok()),
                minor_arcmin: fields.get(4).and_then(|v| v.parse().ok()),
                position_angle_deg: None,
                name: format!("SNR {designation}"),
                common_name: fields.get(5).unwrap_or(&"").to_string(),
                metadata: ObjectMetadata {
                    id: format!("vizier:VII/284:{designation}"),
                    source: "VizieR VII/284/snrs".to_string(),
                    aliases: Vec::new(),
                    parent_ids: Vec::new(),
                    alternate_ids: Vec::new(),
                    alternate_sources: Vec::new(),
                },
            };
            identity_index.ingest(
                &mut objects,
                &mut source_records,
                object,
                catalog_properties(
                    &fields,
                    &[
                        "ra_deg",
                        "dec_deg",
                        "snr",
                        "major_arcmin",
                        "minor_arcmin",
                        "names",
                    ],
                ),
                &mut merge_stats,
            );
        }
    }

    // Galactic Wolf-Rayet stars. Bright ones with an IAU name or a
    // Bright Star Catalogue entry are already present, so skip those
    // positions; the WR number becomes the primary designation.
    let wr = input.join("wr.tsv");
    if wr.exists() {
        sources += 1;
        let content = std::fs::read_to_string(&wr)?;
        for line in content.lines() {
            let fields: Vec<&str> = line.split('\t').map(str::trim).collect();
            if line.starts_with('#') || fields.len() < 3 {
                continue;
            }
            let (Ok(ra), Ok(dec)) = (fields[0].parse::<f64>(), fields[1].parse::<f64>()) else {
                continue;
            };
            let number = fields[2];
            if number.is_empty()
                || !number.starts_with(|c: char| c.is_ascii_digit())
                || grid_dedup.near(ra, dec, 30.0 / 3600.0)
            {
                continue;
            }
            let common_name = [3usize, 4, 5]
                .iter()
                .filter_map(|&i| fields.get(i).copied())
                .find(|v| !v.is_empty())
                .unwrap_or("")
                .to_string();
            let object = SkyObject {
                kind: ObjectKind::Star,
                ra,
                dec,
                mag: None,
                major_arcmin: None,
                minor_arcmin: None,
                position_angle_deg: None,
                name: format!("WR {number}"),
                common_name,
                metadata: ObjectMetadata {
                    id: format!("vizier:III/215:WR{number}"),
                    source: "VizieR III/215/table13".to_string(),
                    aliases: Vec::new(),
                    parent_ids: Vec::new(),
                    alternate_ids: Vec::new(),
                    alternate_sources: Vec::new(),
                },
            };
            identity_index.ingest(
                &mut objects,
                &mut source_records,
                object,
                catalog_properties(
                    &fields,
                    &["ra_deg", "dec_deg", "wr", "name", "gcvs", "other_name"],
                ),
                &mut merge_stats,
            );
        }
    }

    // Bright-nebula catalogs contain explicit cross-identifications. Merge
    // only on those identifiers: the LBN documentation notes that distinct
    // regions can intentionally share an identical center, so positional
    // deduplication would destroy real sub-objects.
    for (file, parser) in [
        (
            "ced.tsv",
            parse_cederblad_line as fn(&str) -> Option<SkyObject>,
        ),
        ("lbn.tsv", parse_lbn_line as fn(&str) -> Option<SkyObject>),
    ] {
        let path = input.join(file);
        if !path.exists() {
            continue;
        }
        sources += 1;
        let content = std::fs::read_to_string(&path)?;
        for line in content.lines() {
            if let Some(object) = parser(line) {
                let fields = line.split('\t').map(str::trim).collect::<Vec<_>>();
                let names = if file == "ced.tsv" {
                    &[
                        "ra_deg",
                        "dec_deg",
                        "ced",
                        "suffix",
                        "name",
                        "major_arcmin",
                        "minor_arcmin",
                        "class",
                        "spectrum",
                    ][..]
                } else {
                    &[
                        "ra_deg",
                        "dec_deg",
                        "lbn",
                        "major_arcmin",
                        "minor_arcmin",
                        "name",
                        "id",
                    ][..]
                };
                identity_index.ingest(
                    &mut objects,
                    &mut source_records,
                    object,
                    catalog_properties(&fields, names),
                    &mut merge_stats,
                );
            }
        }
    }

    if sources == 0 {
        bail!(
            "no catalog sources found in {} (expected NGC.csv, sh2.tsv, \
             barnard.tsv, IAU-CSN.txt); run download-data objects first",
            input.display()
        );
    }

    let data = assemble_object_catalog_data(input, curation_dir, objects, source_records)?;
    let audit = audit_object_metadata(&data.objects)?;
    let catalog = ObjectCatalog::from_data(data)?;
    let count = catalog.len();
    catalog.write_to(output)?;
    if let Some(manifest) = source_manifest {
        let provenance = catalog.provenance()?;
        write_object_source_manifest(
            input,
            output,
            manifest,
            catalog.objects(),
            audit,
            merge_stats,
            provenance.as_ref(),
        )?;
    }
    println!(
        "object metadata: {} aliases, {} alternate IDs, {} alternate sources, {} parent links ({} unresolved)",
        audit.aliases,
        audit.alternate_ids,
        audit.alternate_sources,
        audit.parent_links,
        audit.unresolved_parent_links,
    );
    println!(
        "{count} objects from {sources} source files written to {} ({} new identity-indexed records, {} explicit cross-catalog merges, {} ambiguous cross-identifications retained separately)",
        output.display(),
        merge_stats.added,
        merge_stats.merged,
        merge_stats.ambiguous,
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, Default)]
struct ObjectMergeStats {
    added: usize,
    merged: usize,
    ambiguous: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct ObjectMetadataAudit {
    aliases: usize,
    alternate_ids: usize,
    alternate_sources: usize,
    parent_links: usize,
    unresolved_parent_links: usize,
}

fn audit_object_metadata(objects: &[seiza::objects::SkyObject]) -> Result<ObjectMetadataAudit> {
    let mut primary_ids = std::collections::HashSet::with_capacity(objects.len());
    let mut audit = ObjectMetadataAudit::default();
    for object in objects {
        if object.metadata.id.is_empty() {
            bail!("object '{}' has no stable ID", object.name);
        }
        if !primary_ids.insert(object.metadata.id.as_str()) {
            bail!("duplicate primary object ID: {}", object.metadata.id);
        }
        audit.aliases += object.metadata.aliases.len();
        audit.alternate_ids += object.metadata.alternate_ids.len();
        audit.alternate_sources += object.metadata.alternate_sources.len();
        audit.parent_links += object.metadata.parent_ids.len();
    }
    for object in objects {
        audit.unresolved_parent_links += object
            .metadata
            .parent_ids
            .iter()
            .filter(|parent| !primary_ids.contains(parent.as_str()))
            .count();
    }
    Ok(audit)
}

struct ObjectIdentityIndex {
    by_designation: std::collections::HashMap<String, Vec<usize>>,
}

impl ObjectIdentityIndex {
    fn new(objects: &[seiza::objects::SkyObject]) -> Self {
        let mut index = Self {
            by_designation: std::collections::HashMap::new(),
        };
        for (object_index, object) in objects.iter().enumerate() {
            index.register(object_index, object);
        }
        index
    }

    fn register(&mut self, object_index: usize, object: &seiza::objects::SkyObject) {
        for designation in identity_designations(object) {
            let key = designation_key(designation);
            if key.is_empty() {
                continue;
            }
            let entries = self.by_designation.entry(key).or_default();
            if !entries.contains(&object_index) {
                entries.push(object_index);
            }
        }
    }

    fn merge_or_add(
        &mut self,
        objects: &mut Vec<seiza::objects::SkyObject>,
        incoming: seiza::objects::SkyObject,
        stats: &mut ObjectMergeStats,
    ) {
        let matches = self.matching_indices(objects, &incoming);
        self.merge_or_add_with_matches(objects, incoming, matches, stats);
    }

    fn ingest(
        &mut self,
        objects: &mut Vec<seiza::objects::SkyObject>,
        source_records: &mut Vec<seiza::objects::ObjectSourceRecord>,
        incoming: seiza::objects::SkyObject,
        properties: Vec<seiza::objects::ObjectProperty>,
        stats: &mut ObjectMergeStats,
    ) {
        source_records.push(seiza::objects::ObjectSourceRecord {
            id: incoming.metadata.id.clone(),
            source: incoming.metadata.source.clone(),
            object: incoming.clone(),
            properties,
        });
        self.merge_or_add(objects, incoming, stats);
    }

    fn matching_indices(
        &self,
        objects: &[seiza::objects::SkyObject],
        incoming: &seiza::objects::SkyObject,
    ) -> Vec<usize> {
        let mut matches = Vec::new();
        for designation in identity_designations(incoming) {
            if let Some(indices) = self.by_designation.get(&designation_key(designation)) {
                matches.extend(indices.iter().copied().filter(|&index| {
                    !source_contributes(&objects[index], &incoming.metadata.source)
                }));
            }
        }
        matches.sort_unstable();
        matches.dedup();
        matches
    }

    fn merge_or_add_with_matches(
        &mut self,
        objects: &mut Vec<seiza::objects::SkyObject>,
        incoming: seiza::objects::SkyObject,
        matches: Vec<usize>,
        stats: &mut ObjectMergeStats,
    ) {
        if matches.len() == 1 {
            let object_index = matches[0];
            merge_catalog_record(&mut objects[object_index], incoming);
            self.register(object_index, &objects[object_index]);
            stats.merged += 1;
        } else {
            if matches.len() > 1 {
                stats.ambiguous += 1;
            }
            let object_index = objects.len();
            objects.push(incoming);
            self.register(object_index, &objects[object_index]);
            stats.added += 1;
        }
    }
}

fn identity_designations(object: &seiza::objects::SkyObject) -> impl Iterator<Item = &str> {
    std::iter::once(object.name.as_str())
        .chain(object.metadata.aliases.iter().map(String::as_str))
        .chain(std::iter::once(object.metadata.id.as_str()))
        .chain(object.metadata.alternate_ids.iter().map(String::as_str))
        .filter(|value| !value.is_empty())
}

fn source_contributes(object: &seiza::objects::SkyObject, source: &str) -> bool {
    object.metadata.source == source
        || object
            .metadata
            .alternate_sources
            .iter()
            .any(|existing| existing == source)
}

fn merge_catalog_record(
    target: &mut seiza::objects::SkyObject,
    incoming: seiza::objects::SkyObject,
) {
    if target.kind == seiza::objects::ObjectKind::Other {
        target.kind = incoming.kind;
    }
    target.mag = target.mag.or(incoming.mag);
    target.major_arcmin = target.major_arcmin.or(incoming.major_arcmin);
    target.minor_arcmin = target.minor_arcmin.or(incoming.minor_arcmin);
    target.position_angle_deg = target.position_angle_deg.or(incoming.position_angle_deg);
    if target.common_name.is_empty() {
        target.common_name = incoming.common_name;
    }

    add_alias(target, incoming.name);
    for alias in incoming.metadata.aliases {
        add_alias(target, alias);
    }
    if incoming.metadata.id != target.metadata.id {
        add_unique(&mut target.metadata.alternate_ids, incoming.metadata.id);
    }
    for id in incoming.metadata.alternate_ids {
        if id != target.metadata.id {
            add_unique(&mut target.metadata.alternate_ids, id);
        }
    }
    if incoming.metadata.source != target.metadata.source {
        add_unique(
            &mut target.metadata.alternate_sources,
            incoming.metadata.source,
        );
    }
    for source in incoming.metadata.alternate_sources {
        if source != target.metadata.source {
            add_unique(&mut target.metadata.alternate_sources, source);
        }
    }
    for parent in incoming.metadata.parent_ids {
        add_unique(&mut target.metadata.parent_ids, parent);
    }
}

fn add_alias(object: &mut seiza::objects::SkyObject, alias: String) {
    let key = designation_key(&alias);
    if key.is_empty()
        || designation_key(&object.name) == key
        || object
            .metadata
            .aliases
            .iter()
            .any(|existing| designation_key(existing) == key)
    {
        return;
    }
    object.metadata.aliases.push(alias);
}

fn add_unique(values: &mut Vec<String>, value: String) {
    if !value.is_empty() && !values.contains(&value) {
        values.push(value);
    }
}

fn catalog_properties(fields: &[&str], names: &[&str]) -> Vec<seiza::objects::ObjectProperty> {
    names
        .iter()
        .zip(fields)
        .filter_map(|(name, value)| {
            let value = value.trim();
            (!value.is_empty()).then(|| seiza::objects::ObjectProperty {
                name: (*name).into(),
                value: value.into(),
                unit: None,
            })
        })
        .collect()
}

fn designation_key(value: &str) -> String {
    let compact: String = value
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
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
            .find_map(|(index, c)| c.is_ascii_digit().then_some(index))
            .unwrap_or(compact.len())
    };
    if number_start == compact.len() {
        return compact;
    }
    let number_end = compact[number_start..]
        .char_indices()
        .find_map(|(index, c)| (!c.is_ascii_digit()).then_some(number_start + index))
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

fn catalog_aliases(value: &str) -> Vec<String> {
    let mut aliases: Vec<String> = Vec::new();
    for value in value.split([',', ';', '|']) {
        let value = value.split_whitespace().collect::<Vec<_>>().join(" ");
        if value.is_empty() {
            continue;
        }
        let compact: String = value
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .flat_map(char::to_uppercase)
            .collect();
        let canonical = if let Some(number) = compact.strip_prefix("SH2") {
            format!("Sh2-{}", normalized_number(number))
        } else if let Some(number) = compact.strip_prefix("CED") {
            format!("Ced {}", normalized_number(number))
        } else if let Some(number) = compact.strip_prefix("LBN") {
            format!("LBN {}", normalized_number(number))
        } else {
            value
        };
        if !aliases
            .iter()
            .any(|existing| designation_key(existing) == designation_key(&canonical))
        {
            aliases.push(canonical);
        }
    }
    aliases
}

fn lbn_cross_aliases(value: &str) -> Vec<String> {
    catalog_aliases(value)
        .into_iter()
        .map(|alias| {
            let compact: String = alias
                .chars()
                .filter(|c| c.is_ascii_alphanumeric())
                .flat_map(char::to_uppercase)
                .collect();
            if let Some(number) = compact.strip_prefix('S')
                && number.starts_with(|c: char| c.is_ascii_digit())
            {
                format!("Sh2-{}", normalized_number(number))
            } else if let Some(number) = compact.strip_prefix('C')
                && number.starts_with(|c: char| c.is_ascii_digit())
            {
                format!("Ced {}", normalized_number(number))
            } else {
                alias
            }
        })
        .collect()
}

fn prefixed_catalog_aliases(value: &str, prefix: &str) -> Vec<String> {
    value
        .split([',', ';'])
        .flat_map(|value| {
            let value = value.trim();
            if value.is_empty() {
                Vec::new()
            } else if value
                .to_ascii_uppercase()
                .starts_with(&prefix.to_ascii_uppercase())
            {
                catalog_aliases(value)
            } else {
                catalog_aliases(&format!("{prefix} {value}"))
            }
        })
        .collect()
}

fn normalized_number(value: &str) -> String {
    let number_end = value
        .char_indices()
        .find_map(|(index, c)| (!c.is_ascii_digit()).then_some(index))
        .unwrap_or(value.len());
    let number = value[..number_end].trim_start_matches('0');
    let number = if number.is_empty() { "0" } else { number };
    format!("{number}{}", &value[number_end..])
}

fn stable_id_for_alias(alias: &str) -> Option<String> {
    let key = designation_key(alias);
    for (prefix, namespace) in [
        ("NGC", "openngc:NGC"),
        ("IC", "openngc:IC"),
        ("SH2", "vizier:VII/20:Sh2-"),
        ("CED", "vizier:VII/231:Ced"),
        ("LBN", "vizier:VII/9:LBN"),
        ("UGC", "vizier:VII/26D:UGC"),
        ("PGC", "vizier:VII/237:PGC"),
        ("HD", "vizier:V/50:HD"),
        ("WR", "vizier:III/215:WR"),
        ("M", "messier:M"),
        ("B", "vizier:VII/220A:B"),
        ("C", "caldwell:C"),
    ] {
        if let Some(value) = key.strip_prefix(prefix)
            && value.starts_with(|c: char| c.is_ascii_digit())
        {
            return Some(format!("{namespace}{value}"));
        }
    }
    None
}

fn parse_lbn_line(line: &str) -> Option<seiza::objects::SkyObject> {
    use seiza::objects::{ObjectKind, ObjectMetadata, SkyObject};

    if line.starts_with('#') {
        return None;
    }
    let fields: Vec<&str> = line.split('\t').map(str::trim).collect();
    if fields.len() < 6 {
        return None;
    }
    let (ra, dec, number) = (
        fields[0].parse().ok()?,
        fields[1].parse().ok()?,
        fields[2].parse::<u32>().ok()?,
    );
    let aliases = lbn_cross_aliases(fields[5]);
    let primary_id = format!("vizier:VII/9:LBN{number}");
    let mut alternate_ids = Vec::new();
    for alias in &aliases {
        if let Some(id) = stable_id_for_alias(alias)
            && id != primary_id
        {
            add_unique(&mut alternate_ids, id);
        }
    }
    Some(SkyObject {
        kind: ObjectKind::Nebula,
        ra,
        dec,
        mag: None,
        major_arcmin: fields[3].parse().ok().filter(|value: &f32| *value > 0.0),
        minor_arcmin: fields[4].parse().ok().filter(|value: &f32| *value > 0.0),
        position_angle_deg: None,
        name: format!("LBN {number}"),
        common_name: String::new(),
        metadata: ObjectMetadata {
            id: primary_id,
            source: "VizieR VII/9/catalog".to_string(),
            aliases,
            parent_ids: Vec::new(),
            alternate_ids,
            alternate_sources: Vec::new(),
        },
    })
}

fn parse_cederblad_line(line: &str) -> Option<seiza::objects::SkyObject> {
    use seiza::objects::{ObjectKind, ObjectMetadata, SkyObject};

    if line.starts_with('#') {
        return None;
    }
    let fields: Vec<&str> = line.split('\t').map(str::trim).collect();
    if fields.len() < 9 {
        return None;
    }
    let (ra, dec, number) = (
        fields[0].parse().ok()?,
        fields[1].parse().ok()?,
        fields[2].parse::<u32>().ok()?,
    );
    let suffix = fields[3];
    let class = fields[7];
    let spectrum = fields[8];
    let kind = if class.starts_with('A') {
        ObjectKind::ClusterWithNebula
    } else if spectrum.contains(['E', 'e']) {
        ObjectKind::HiiRegion
    } else {
        ObjectKind::Nebula
    };
    let aliases = catalog_aliases(fields[4]);
    let primary_id = format!("vizier:VII/231:Ced{number}{suffix}");
    let mut alternate_ids = Vec::new();
    for alias in &aliases {
        if let Some(id) = stable_id_for_alias(alias)
            && id != primary_id
        {
            add_unique(&mut alternate_ids, id);
        }
    }
    Some(SkyObject {
        kind,
        ra,
        dec,
        mag: None,
        major_arcmin: fields[5].parse().ok().filter(|value: &f32| *value > 0.0),
        minor_arcmin: fields[6].parse().ok().filter(|value: &f32| *value > 0.0),
        position_angle_deg: None,
        name: format!("Ced {number}{suffix}"),
        common_name: String::new(),
        metadata: ObjectMetadata {
            id: primary_id,
            source: "VizieR VII/231/catalog".to_string(),
            aliases,
            parent_ids: Vec::new(),
            alternate_ids,
            alternate_sources: Vec::new(),
        },
    })
}

/// One `;`-separated OpenNGC row. Skips duplicates and non-existent entries.
fn parse_openngc_line(line: &str) -> Option<seiza::objects::SkyObject> {
    use seiza::objects::{ObjectKind, ObjectMetadata, SkyObject};

    let fields: Vec<&str> = line.split(';').collect();
    if fields.len() < 30 {
        return None;
    }
    let kind = match fields[1] {
        "G" | "GPair" | "GTrpl" | "GGroup" => ObjectKind::Galaxy,
        "OCl" => ObjectKind::OpenCluster,
        "GCl" => ObjectKind::GlobularCluster,
        "PN" => ObjectKind::PlanetaryNebula,
        "HII" => ObjectKind::HiiRegion,
        "SNR" => ObjectKind::SupernovaRemnant,
        "DrkN" => ObjectKind::DarkNebula,
        "Neb" | "EmN" | "RfN" => ObjectKind::Nebula,
        "Cl+N" => ObjectKind::ClusterWithNebula,
        // Bare star entries are catalog-number noise next to the IAU
        // named-star list (e.g. IC 1318 is typed as the star gamma Cyg)
        "*" | "**" => return None,
        "*Ass" => ObjectKind::Association,
        "Dup" | "NonEx" => return None,
        _ => ObjectKind::Other,
    };

    // RA "HH:MM:SS.ss", Dec "+DD:MM:SS.s"
    let ra = parse_sexagesimal(fields[2])? * 15.0;
    let dec = parse_sexagesimal(fields[3])?;

    // Prefer the Messier designation, prettify the NGC/IC name, and retain
    // the OpenNGC designation as both a stable ID and an alias when needed.
    let raw_name = fields[0];
    let catalog_name = if let Some(rest) = raw_name.strip_prefix("NGC") {
        format!("NGC {}", rest.trim_start_matches('0'))
    } else if let Some(rest) = raw_name.strip_prefix("IC") {
        format!("IC {}", rest.trim_start_matches('0'))
    } else {
        raw_name.to_string()
    };
    let name = match fields.get(23).map(|m| m.trim_start_matches('0')) {
        Some(m) if !m.is_empty() => format!("M {m}"),
        _ => catalog_name.clone(),
    };
    let primary_id = format!("openngc:{}", designation_key(&catalog_name));
    let mut aliases: Vec<String> = (catalog_name != name)
        .then_some(catalog_name.clone())
        .into_iter()
        .collect();
    for alias in prefixed_catalog_aliases(fields.get(24).copied().unwrap_or(""), "NGC")
        .into_iter()
        .chain(prefixed_catalog_aliases(
            fields.get(25).copied().unwrap_or(""),
            "IC",
        ))
        .chain(catalog_aliases(fields.get(27).copied().unwrap_or("")))
    {
        if designation_key(&alias) != designation_key(&name)
            && !aliases
                .iter()
                .any(|existing| designation_key(existing) == designation_key(&alias))
        {
            aliases.push(alias);
        }
    }
    let mut alternate_ids = Vec::new();
    for designation in std::iter::once(&name).chain(&aliases) {
        if let Some(id) = stable_id_for_alias(designation)
            && id != primary_id
        {
            add_unique(&mut alternate_ids, id);
        }
    }
    let common_name = fields
        .get(28)
        .and_then(|names| names.split(',').next())
        .unwrap_or("")
        .trim()
        .to_string();

    Some(SkyObject {
        kind,
        ra,
        dec,
        mag: fields[9].parse().ok().or_else(|| fields[8].parse().ok()),
        major_arcmin: fields[5].parse().ok().filter(|v: &f32| *v > 0.0),
        minor_arcmin: fields[6].parse().ok().filter(|v: &f32| *v > 0.0),
        position_angle_deg: fields[7].parse().ok(),
        name,
        common_name,
        metadata: ObjectMetadata {
            id: primary_id,
            source: "OpenNGC".to_string(),
            aliases,
            parent_ids: Vec::new(),
            alternate_ids,
            alternate_sources: Vec::new(),
        },
    })
}

/// "HH:MM:SS.ss" or "+DD:MM:SS.s" to a float in the leading unit.
fn parse_sexagesimal(value: &str) -> Option<f64> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let negative = value.starts_with('-');
    let parts: Vec<&str> = value.trim_start_matches(['-', '+']).split(':').collect();
    let mut total = 0.0;
    let mut scale = 1.0;
    for part in parts {
        total += part.parse::<f64>().ok()? * scale;
        scale /= 60.0;
    }
    Some(if negative { -total } else { total })
}

/// One line of the IAU-CSN list. The ASCII name occupies the first 18
/// bytes; RA/Dec (J2000, degrees) are anchored by the date column.
fn parse_iau_csn_line(line: &str) -> Option<seiza::objects::SkyObject> {
    use seiza::objects::{ObjectKind, ObjectMetadata, SkyObject};

    if line.starts_with('#') || line.len() < 40 || !line.is_ascii() && line.get(..18).is_none() {
        return None;
    }
    let name = line.get(..18)?.trim();
    if name.is_empty() {
        return None;
    }
    let tokens: Vec<&str> = line.split_whitespace().collect();
    let date_index = tokens
        .iter()
        .position(|t| t.len() == 10 && t.as_bytes()[4] == b'-' && t.as_bytes()[7] == b'-')?;
    if date_index < 6 {
        return None;
    }
    let ra: f64 = tokens[date_index - 2].parse().ok()?;
    let dec: f64 = tokens[date_index - 1].parse().ok()?;
    let mag: Option<f32> = tokens[date_index - 6].parse().ok();

    Some(SkyObject {
        kind: ObjectKind::Star,
        ra,
        dec,
        mag,
        major_arcmin: None,
        minor_arcmin: None,
        position_angle_deg: None,
        name: name.to_string(),
        common_name: name.to_string(),
        metadata: ObjectMetadata {
            id: format!("iau-csn:{name}"),
            source: "IAU Catalog of Star Names".to_string(),
            aliases: Vec::new(),
            parent_ids: Vec::new(),
            alternate_ids: Vec::new(),
            alternate_sources: Vec::new(),
        },
    })
}

struct ObjectSourceDescriptor {
    label: &'static str,
    reference_url: &'static str,
    files: &'static [&'static str],
}

const OBJECT_SOURCE_DESCRIPTORS: &[ObjectSourceDescriptor] = &[
    ObjectSourceDescriptor {
        label: "OpenNGC",
        reference_url: "https://github.com/mattiaverga/OpenNGC",
        files: &["NGC.csv", "addendum.csv"],
    },
    ObjectSourceDescriptor {
        label: "VizieR VII/20/catalog",
        reference_url: "https://cdsarc.cds.unistra.fr/viz-bin/cat/VII/20",
        files: &["sh2.tsv"],
    },
    ObjectSourceDescriptor {
        label: "VizieR VII/220A/barnard",
        reference_url: "https://cdsarc.cds.unistra.fr/viz-bin/cat/VII/220A",
        files: &["barnard.tsv"],
    },
    ObjectSourceDescriptor {
        label: "VizieR VII/26D/catalog",
        reference_url: "https://cdsarc.cds.unistra.fr/viz-bin/cat/VII/26D",
        files: &["ugc.tsv"],
    },
    ObjectSourceDescriptor {
        label: "VizieR VII/7A/ldn",
        reference_url: "https://cdsarc.cds.unistra.fr/viz-bin/cat/VII/7A",
        files: &["ldn.tsv"],
    },
    ObjectSourceDescriptor {
        label: "VizieR VII/21/catalog",
        reference_url: "https://cdsarc.cds.unistra.fr/viz-bin/cat/VII/21",
        files: &["vdb.tsv"],
    },
    ObjectSourceDescriptor {
        label: "VizieR VII/231/catalog",
        reference_url: "https://cdsarc.cds.unistra.fr/viz-bin/cat/VII/231",
        files: &["ced.tsv"],
    },
    ObjectSourceDescriptor {
        label: "VizieR VII/9/catalog",
        reference_url: "https://cdsarc.cds.unistra.fr/viz-bin/cat/VII/9",
        files: &["lbn.tsv"],
    },
    ObjectSourceDescriptor {
        label: "VizieR V/50/catalog",
        reference_url: "https://cdsarc.cds.unistra.fr/viz-bin/cat/V/50",
        files: &["bsc.tsv"],
    },
    ObjectSourceDescriptor {
        label: "VizieR VII/237/pgc",
        reference_url: "https://cdsarc.cds.unistra.fr/viz-bin/cat/VII/237",
        files: &["pgc.tsv"],
    },
    ObjectSourceDescriptor {
        label: "VizieR VII/284/snrs",
        reference_url: "https://cdsarc.cds.unistra.fr/viz-bin/cat/VII/284",
        files: &["snr.tsv"],
    },
    ObjectSourceDescriptor {
        label: "VizieR III/215/table13",
        reference_url: "https://cdsarc.cds.unistra.fr/viz-bin/cat/III/215",
        files: &["wr.tsv"],
    },
    ObjectSourceDescriptor {
        label: "IAU Catalog of Star Names",
        reference_url: "https://www.iau.org/public/themes/naming_stars/",
        files: &["IAU-CSN.txt"],
    },
];

fn write_object_source_manifest(
    input: &Path,
    output: &Path,
    manifest: &Path,
    objects: &[seiza::objects::SkyObject],
    audit: ObjectMetadataAudit,
    merge_stats: ObjectMergeStats,
    provenance: Option<&seiza::objects::ObjectCatalogProvenance>,
) -> Result<()> {
    let mut counts = std::collections::BTreeMap::<&str, usize>::new();
    for object in objects {
        let mut sources = vec![object.metadata.source.as_str()];
        sources.extend(object.metadata.alternate_sources.iter().map(String::as_str));
        sources.sort_unstable();
        sources.dedup();
        for source in sources {
            *counts.entry(source).or_default() += 1;
        }
    }

    let mut sources = Vec::new();
    for descriptor in OBJECT_SOURCE_DESCRIPTORS {
        let mut files = Vec::new();
        for name in descriptor.files {
            let path = input.join(name);
            if !path.exists() {
                continue;
            }
            let (bytes, sha256) = file_digest(&path)?;
            files.push(serde_json::json!({
                "name": name,
                "bytes": bytes,
                "sha256": sha256,
            }));
        }
        if files.is_empty() {
            continue;
        }
        sources.push(serde_json::json!({
            "label": descriptor.label,
            "reference_url": descriptor.reference_url,
            "contributing_objects": counts.get(descriptor.label).copied().unwrap_or(0),
            "files": files,
        }));
    }

    let (bytes, sha256) = file_digest(output)?;
    let document = serde_json::json!({
        "format": "SEIZAOB-v4-container-1",
        "artifact": {
            "name": output.file_name().unwrap_or_default().to_string_lossy(),
            "objects": objects.len(),
            "bytes": bytes,
            "sha256": sha256,
            "metadata": {
                "aliases": audit.aliases,
                "alternate_ids": audit.alternate_ids,
                "alternate_sources": audit.alternate_sources,
                "parent_links": audit.parent_links,
                "unresolved_parent_links": audit.unresolved_parent_links,
            },
            "identity_ingest": {
                "new_records": merge_stats.added,
                "cross_catalog_merges": merge_stats.merged,
                "ambiguous_cross_identifications": merge_stats.ambiguous,
            },
        },
        "sources": sources,
        "curation": provenance.and_then(|provenance| provenance.curation.as_ref()),
        "acknowledgements": [
            "This product includes data retrieved through the VizieR catalogue access tool, CDS, Strasbourg, France.",
            "Catalog publications and source-specific usage terms are linked by each source entry."
        ],
    });
    let mut json = serde_json::to_string_pretty(&document)?;
    json.push('\n');
    std::fs::write(manifest, json)?;
    println!("object source manifest written to {}", manifest.display());
    Ok(())
}

fn file_digest(path: &Path) -> Result<(u64, String)> {
    use sha2::Digest;
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let bytes = file.metadata()?.len();
    let mut hasher = sha2::Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let hash = hasher.finalize();
    let sha256 = hash.iter().map(|byte| format!("{byte:02x}")).collect();
    Ok((bytes, sha256))
}

/// Build star tiles from Gaia DR3 TAP CSV chunks (download-data gaia).
/// Positions are epoch J2016.0; proper motions are applied to `epoch`.
pub fn build_gaia(input: &Path, output: &Path, epoch: f64, max_mag: f32, bands: u32) -> Result<()> {
    let mut parts: Vec<_> = std::fs::read_dir(input)
        .with_context(|| format!("cannot read {}", input.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("gaia-") && n.ends_with(".csv"))
        })
        .collect();
    parts.sort();
    if parts.is_empty() {
        bail!(
            "no gaia-*.csv files in {}; run download-data gaia first",
            input.display()
        );
    }

    let mut builder = TileSetBuilder::new(
        bands,
        epoch,
        "Gaia DR3 (ESA/Gaia/DPAC, CC BY-SA 3.0 IGO); G magnitudes",
    );
    let mut too_faint = 0u64;
    let dt = epoch - 2016.0;

    for part in &parts {
        let file =
            std::fs::File::open(part).with_context(|| format!("cannot open {}", part.display()))?;
        for line in BufReader::new(file).lines().skip(1) {
            let line = line?;
            let mut fields = line.split(',');
            let (Some(ra), Some(dec), pmra, pmdec, Some(mag)) = (
                fields.next().and_then(|v| v.parse::<f64>().ok()),
                fields.next().and_then(|v| v.parse::<f64>().ok()),
                fields.next().and_then(|v| v.parse::<f64>().ok()),
                fields.next().and_then(|v| v.parse::<f64>().ok()),
                fields.next().and_then(|v| v.parse::<f32>().ok()),
            ) else {
                continue;
            };
            if mag > max_mag {
                too_faint += 1;
                continue;
            }
            let cos_dec = dec.to_radians().cos().max(1e-6);
            let ra = (ra + pmra.unwrap_or(0.0) * dt / 3_600_000.0 / cos_dec).rem_euclid(360.0);
            let dec = (dec + pmdec.unwrap_or(0.0) * dt / 3_600_000.0).clamp(-90.0, 90.0);
            builder.add(ra, dec, mag);
        }
    }

    let count = builder.star_count();
    builder.write_to(output)?;
    println!(
        "{count} stars written to {} (epoch {epoch}, {too_faint} fainter than {max_mag})",
        output.display()
    );
    Ok(())
}

/// Write or roll forward a complete bundle manifest for hosting.
pub fn build_manifest(
    dir: &Path,
    base_manifest: Option<&Path>,
    version: &str,
    output: &Path,
) -> Result<()> {
    use seiza_download::{BundleManifest, ManifestFile};
    use std::collections::BTreeMap;

    let content_addressed = if version.starts_with("catalog-bundle-v4-") {
        true
    } else if version.starts_with("catalog-bundle-v2-") {
        false
    } else {
        bail!(
            "unsupported bundle version {version}; expected catalog-bundle-v2-* or catalog-bundle-v4-*"
        );
    };

    let mut files = if let Some(path) = base_manifest {
        let bytes = std::fs::read(path)
            .with_context(|| format!("cannot read base manifest {}", path.display()))?;
        let manifest = BundleManifest::parse(&bytes)
            .with_context(|| format!("invalid base manifest {}", path.display()))?;
        manifest
            .files
            .into_iter()
            .map(|file| (file.name.clone(), file))
            .collect::<BTreeMap<_, _>>()
    } else {
        BTreeMap::new()
    };

    let mut entries: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .is_some_and(|ext| ext == "bin" || ext == "idx")
        })
        .collect();
    entries.sort();
    if entries.is_empty() && files.is_empty() {
        bail!("no .bin or .idx data files in {}", dir.display());
    }

    let replacement_count = entries.len();
    for path in &entries {
        let (bytes, hash_hex) = file_digest(path)?;
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .context("bundle artifact filename is not UTF-8")?
            .to_string();
        let key = content_addressed.then(|| format!("artifacts/{hash_hex}/{name}"));
        files.insert(
            name.clone(),
            ManifestFile {
                name: name.clone(),
                key,
                bytes,
                sha256: hash_hex.clone(),
            },
        );
        println!("  {name}: {bytes} bytes, sha256 {hash_hex}");
    }

    // A previous v2 or v4 manifest may be rolled into either output
    // generation. V4 always rekeys retained artifacts by content hash; v2
    // deliberately retains the flat layout required by released readers.
    for file in files.values_mut() {
        file.key = content_addressed.then(|| format!("artifacts/{}/{}", file.sha256, file.name));
    }
    let manifest = BundleManifest {
        version: version.to_string(),
        files: files.into_values().collect(),
    };
    manifest
        .validate()
        .context("generated bundle is incomplete")?;
    let mut encoded = serde_json::to_vec_pretty(&manifest)?;
    encoded.push(b'\n');
    std::fs::write(output, encoded)?;
    println!(
        "manifest written to {} ({} replaced, {} retained)",
        output.display(),
        replacement_count,
        manifest.files.len().saturating_sub(replacement_count)
    );
    Ok(())
}

/// Comets (MPC CometEls.txt) and bright numbered asteroids (MPCORB) into
/// a minor-body element set for time-dependent matching.
pub fn build_minor_bodies(input: &Path, output: &Path, max_h: f32) -> Result<()> {
    use seiza::minor_bodies::{MinorBodyCatalog, julian_date};

    let mut bodies = Vec::new();

    // JPL SBDB carries every catalogued comet with apparition-specific
    // elements (historic acquisition dates need the elements from THAT
    // apparition); MPC CometEls is the fallback for fresh discoveries
    let sbdb = input.join("sbdb-comets.json");
    if sbdb.exists() {
        let parsed: serde_json::Value = serde_json::from_reader(std::fs::File::open(&sbdb)?)?;
        for row in parsed["data"]
            .as_array()
            .map(|d| d.as_slice())
            .unwrap_or(&[])
        {
            if let Some(body) = parse_sbdb_comet(row) {
                bodies.push(body);
            }
        }
    }
    let sbdb_names: std::collections::HashSet<String> =
        bodies.iter().map(|b| b.name.clone()).collect();

    let comets = input.join("CometEls.txt");
    if comets.exists() {
        let content = std::fs::read_to_string(&comets)?;
        for line in content.lines() {
            if let Some(body) = parse_comet_line(line)
                && !sbdb_names.contains(&body.name)
            {
                bodies.push(body);
            }
        }
    }
    let comet_count = bodies.len();

    let mpcorb = input.join("MPCORB.DAT.gz");
    if mpcorb.exists() {
        let file = std::fs::File::open(&mpcorb)?;
        let reader = std::io::BufReader::new(flate2::read::GzDecoder::new(file));
        use std::io::BufRead;
        let mut in_data = false;
        for line in reader.lines() {
            let line = line?;
            if !in_data {
                if line.starts_with("----------") {
                    in_data = true;
                }
                continue;
            }
            if let Some(body) = parse_mpcorb_line(&line, max_h) {
                bodies.push(body);
            }
        }
    }

    if bodies.is_empty() {
        bail!(
            "no minor bodies parsed from {} (expected CometEls.txt and/or MPCORB.DAT.gz); \
             run download-data mpc first",
            input.display()
        );
    }
    let catalog = MinorBodyCatalog::new(bodies);
    catalog.write_to(output)?;
    println!(
        "{} minor bodies ({} comets, {} asteroids H <= {max_h}) written to {}",
        catalog.len(),
        comet_count,
        catalog.len() - comet_count,
        output.display()
    );

    // A couple of headline entries as a sanity check
    for body in catalog.bodies().iter().take(3) {
        if let Some((ra, dec, mag, _)) =
            MinorBodyCatalog::position_at(body, julian_date(2026, 7, 13.0))
        {
            println!("  {}: now near ({ra:.2}, {dec:.2}) V~{mag:.1}", body.name);
        }
    }
    Ok(())
}

/// One JPL SBDB comet row: [full_name, epoch, q, e, i, om, w, tp, M1, K1].
fn parse_sbdb_comet(row: &serde_json::Value) -> Option<seiza::minor_bodies::MinorBody> {
    use seiza::minor_bodies::{MinorBody, MinorBodyKind};
    let text = |i: usize| row.get(i).and_then(|v| v.as_str()).map(str::trim);
    let number = |i: usize| text(i).and_then(|v| v.parse::<f64>().ok());
    let name = text(0)?.to_string();
    if name.is_empty() {
        return None;
    }
    Some(MinorBody {
        kind: MinorBodyKind::Comet,
        name,
        epoch_jd: number(7)?, // perihelion time tp (TDB)
        q_or_a: number(2)?,
        eccentricity: number(3)?,
        inclination_deg: number(4)?,
        node_deg: number(5)?,
        arg_perihelion_deg: number(6)?,
        mean_anomaly_deg: 0.0,
        h_mag: number(8).unwrap_or(12.0) as f32,
        slope: number(9).unwrap_or(4.0) as f32,
    })
}

/// One fixed-width MPC CometEls.txt record.
fn parse_comet_line(line: &str) -> Option<seiza::minor_bodies::MinorBody> {
    use seiza::minor_bodies::{MinorBody, MinorBodyKind, julian_date};
    if line.len() < 103 {
        return None;
    }
    let field = |a: usize, b: usize| line.get(a - 1..b).map(str::trim).unwrap_or("");
    let year: i32 = field(15, 18).parse().ok()?;
    let month: u32 = field(20, 21).parse().ok()?;
    let day: f64 = field(23, 29).parse().ok()?;
    let q: f64 = field(31, 39).parse().ok()?;
    let e: f64 = field(41, 49).parse().ok()?;
    let arg_peri: f64 = field(52, 59).parse().ok()?;
    let node: f64 = field(62, 69).parse().ok()?;
    let incl: f64 = field(72, 79).parse().ok()?;
    let m1: f32 = field(92, 95).parse().unwrap_or(12.0);
    let k1: f32 = field(97, 100).parse().unwrap_or(4.0);
    let name = line
        .get(102..158)
        .or_else(|| line.get(102..))
        .map(str::trim)
        .unwrap_or("")
        .to_string();
    if name.is_empty() {
        return None;
    }
    Some(MinorBody {
        kind: MinorBodyKind::Comet,
        name,
        epoch_jd: julian_date(year, month, day),
        q_or_a: q,
        eccentricity: e,
        inclination_deg: incl,
        node_deg: node,
        arg_perihelion_deg: arg_peri,
        mean_anomaly_deg: 0.0,
        h_mag: m1,
        slope: k1,
    })
}

/// One fixed-width MPCORB record; numbered asteroids up to `max_h` only.
fn parse_mpcorb_line(line: &str, max_h: f32) -> Option<seiza::minor_bodies::MinorBody> {
    use seiza::minor_bodies::{MinorBody, MinorBodyKind};
    if line.len() < 104 {
        return None;
    }
    let field = |a: usize, b: usize| line.get(a - 1..b).map(str::trim).unwrap_or("");
    // Numbered objects pack the number in columns 1-5 (base-62 first char
    // above 99999); provisional-only objects use a different packing that
    // includes letters past column 5 — skip those
    let packed = field(1, 7);
    let number = unpack_asteroid_number(packed)?;
    let h: f32 = field(9, 13).parse().ok()?;
    if h > max_h {
        return None;
    }
    let g: f32 = field(15, 19).parse().unwrap_or(0.15);
    let epoch_jd = unpack_epoch(field(21, 25))?;
    let mean_anomaly: f64 = field(27, 35).parse().ok()?;
    let arg_peri: f64 = field(38, 46).parse().ok()?;
    let node: f64 = field(49, 57).parse().ok()?;
    let incl: f64 = field(60, 68).parse().ok()?;
    let e: f64 = field(71, 79).parse().ok()?;
    let a: f64 = field(93, 103).parse().ok()?;

    // Readable designation ("1 Ceres") lives at columns 167+
    let name = match line
        .get(166..194)
        .or_else(|| line.get(166..))
        .map(str::trim)
    {
        Some(designation) if !designation.is_empty() => match designation.split_once(' ') {
            Some((num, rest)) if num.chars().all(|c| c.is_ascii_digit()) => {
                format!("({num}) {}", rest.trim())
            }
            _ => designation.to_string(),
        },
        _ => format!("({number})"),
    };

    Some(MinorBody {
        kind: MinorBodyKind::Asteroid,
        name,
        epoch_jd,
        q_or_a: a,
        eccentricity: e,
        inclination_deg: incl,
        node_deg: node,
        arg_perihelion_deg: arg_peri,
        mean_anomaly_deg: mean_anomaly,
        h_mag: h,
        slope: g,
    })
}

/// MPC packed asteroid number: "00001" -> 1, "A0000" -> 100000, ...
fn unpack_asteroid_number(packed: &str) -> Option<u32> {
    if packed.is_empty() || packed.len() > 5 {
        return None;
    }
    let mut chars = packed.chars();
    let first = chars.next()?;
    let rest: String = chars.collect();
    let head = if first.is_ascii_digit() {
        first.to_digit(10)?
    } else if first.is_ascii_uppercase() {
        first as u32 - 'A' as u32 + 10
    } else if first.is_ascii_lowercase() {
        first as u32 - 'a' as u32 + 36
    } else {
        return None;
    };
    let tail: u32 = if rest.is_empty() {
        0
    } else {
        rest.parse().ok()?
    };
    Some(head * 10u32.pow(rest.len() as u32) + tail)
}

/// MPC packed epoch: "K239D" -> JD of 2023-09-13.0 TT.
fn unpack_epoch(packed: &str) -> Option<f64> {
    use seiza::minor_bodies::julian_date;
    let bytes = packed.as_bytes();
    if bytes.len() != 5 {
        return None;
    }
    let century = match bytes[0] {
        b'I' => 1800,
        b'J' => 1900,
        b'K' => 2000,
        _ => return None,
    };
    let year: i32 = packed.get(1..3)?.parse::<i32>().ok()? + century;
    let code = |b: u8| -> Option<u32> {
        match b {
            b'1'..=b'9' => Some((b - b'0') as u32),
            b'A'..=b'V' => Some((b - b'A') as u32 + 10),
            _ => None,
        }
    };
    let month = code(bytes[3])?;
    let day = code(bytes[4])?;
    Some(julian_date(year, month, day as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_complete_bundle_inputs(dir: &Path, marker: &str) {
        for name in seiza_download::REQUIRED_BUNDLE_FILES {
            std::fs::write(dir.join(name), format!("{marker}:{name}")).unwrap();
        }
    }

    fn object(name: &str, id: &str, source: &str) -> seiza::objects::SkyObject {
        seiza::objects::SkyObject {
            kind: seiza::objects::ObjectKind::Nebula,
            ra: 312.5,
            dec: 44.3,
            mag: None,
            major_arcmin: None,
            minor_arcmin: None,
            position_angle_deg: None,
            name: name.to_string(),
            common_name: String::new(),
            metadata: seiza::objects::ObjectMetadata {
                id: id.to_string(),
                source: source.to_string(),
                aliases: Vec::new(),
                parent_ids: Vec::new(),
                alternate_ids: Vec::new(),
                alternate_sources: Vec::new(),
            },
        }
    }

    #[test]
    fn v4_manifest_builder_emits_content_addressed_complete_bundle() {
        let dir = tempfile::tempdir().unwrap();
        write_complete_bundle_inputs(dir.path(), "initial");
        let output = dir.path().join("manifest.json");
        build_manifest(dir.path(), None, "catalog-bundle-v4-test", &output).unwrap();

        let manifest =
            seiza_download::BundleManifest::parse(&std::fs::read(output).unwrap()).unwrap();
        assert_eq!(
            manifest.files.len(),
            seiza_download::REQUIRED_BUNDLE_FILES.len()
        );
        for file in manifest.files {
            assert_eq!(
                file.key.as_deref(),
                Some(format!("artifacts/{}/{}", file.sha256, file.name).as_str())
            );
        }
    }

    #[test]
    fn one_dynamic_update_rolls_forward_v2_and_v4_bundles() {
        let full = tempfile::tempdir().unwrap();
        write_complete_bundle_inputs(full.path(), "initial");
        let base = full.path().join("base-v2.json");
        build_manifest(full.path(), None, "catalog-bundle-v2-base", &base).unwrap();
        let original =
            seiza_download::BundleManifest::parse(&std::fs::read(&base).unwrap()).unwrap();

        let delta = tempfile::tempdir().unwrap();
        std::fs::write(delta.path().join("transients.bin"), b"new transients").unwrap();
        std::fs::write(delta.path().join("minor-bodies.bin"), b"new minor bodies").unwrap();

        let v2 = delta.path().join("next-v2.json");
        let v4 = delta.path().join("next-v4.json");
        build_manifest(delta.path(), Some(&base), "catalog-bundle-v2-next", &v2).unwrap();
        build_manifest(delta.path(), Some(&base), "catalog-bundle-v4-next", &v4).unwrap();

        let v2 = seiza_download::BundleManifest::parse(&std::fs::read(v2).unwrap()).unwrap();
        let v4 = seiza_download::BundleManifest::parse(&std::fs::read(v4).unwrap()).unwrap();
        let original_objects = original
            .files
            .iter()
            .find(|file| file.name == "objects.bin")
            .unwrap();
        let original_transients = original
            .files
            .iter()
            .find(|file| file.name == "transients.bin")
            .unwrap();
        for manifest in [&v2, &v4] {
            assert_eq!(
                manifest
                    .files
                    .iter()
                    .find(|file| file.name == "objects.bin")
                    .unwrap()
                    .sha256,
                original_objects.sha256
            );
            assert_ne!(
                manifest
                    .files
                    .iter()
                    .find(|file| file.name == "transients.bin")
                    .unwrap()
                    .sha256,
                original_transients.sha256
            );
        }
        assert!(v2.files.iter().all(|file| file.key.is_none()));
        assert!(v4.files.iter().all(|file| file.key.is_some()));
    }

    #[test]
    fn normalizes_catalog_designations_for_identity_matching() {
        assert_eq!(designation_key("NGC 0224"), "NGC224");
        assert_eq!(designation_key("Sh2-001"), "SH21");
        assert_eq!(designation_key("Ced 055b"), "CED55B");

        assert_eq!(
            catalog_aliases("NGC 7000; Sh2-117, Ced 55, C 20"),
            vec!["NGC 7000", "Sh2-117", "Ced 55", "C 20"]
        );
        assert_eq!(
            lbn_cross_aliases("NGC 7000; S 117, C 55"),
            vec!["NGC 7000", "Sh2-117", "Ced 55"]
        );
        assert_eq!(stable_id_for_alias("M 031").as_deref(), Some("messier:M31"));
    }

    #[test]
    fn parses_lbn_cross_identifiers_and_extent() {
        let object = parse_lbn_line("312.5\t44.3\t331\t120\t90\tNGC 7000; C 20\t1").unwrap();
        assert_eq!(object.name, "LBN 331");
        assert_eq!(object.metadata.id, "vizier:VII/9:LBN331");
        assert_eq!(object.major_arcmin, Some(120.0));
        assert_eq!(object.minor_arcmin, Some(90.0));
        assert_eq!(object.metadata.aliases, vec!["NGC 7000", "Ced 20"]);
        assert_eq!(
            object.metadata.alternate_ids,
            vec!["openngc:NGC7000", "vizier:VII/231:Ced20"]
        );
    }

    #[test]
    fn parses_cederblad_suffix_cross_identifiers_and_kind() {
        let object =
            parse_cederblad_line("83.8\t-5.4\t55\tb\tNGC 1976, M 42\t30\t20\tE\tE").unwrap();
        assert_eq!(object.name, "Ced 55b");
        assert_eq!(object.kind, seiza::objects::ObjectKind::HiiRegion);
        assert_eq!(object.metadata.id, "vizier:VII/231:Ced55b");
        assert_eq!(
            object.metadata.alternate_ids,
            vec!["openngc:NGC1976", "messier:M42"]
        );
    }

    #[test]
    fn openngc_retains_catalog_identifiers_as_aliases_and_ids() {
        let mut fields = vec![""; 30];
        fields[0] = "NGC0224";
        fields[1] = "G";
        fields[2] = "00:42:44.3";
        fields[3] = "+41:16:09";
        fields[5] = "177.8";
        fields[6] = "69.7";
        fields[8] = "4.36";
        fields[23] = "031";
        fields[25] = "10";
        fields[27] = "PGC 2557, UGC 454, C 23";
        fields[28] = "Andromeda Galaxy";
        let object = parse_openngc_line(&fields.join(";")).unwrap();

        assert_eq!(object.name, "M 31");
        assert_eq!(object.metadata.id, "openngc:NGC224");
        assert_eq!(
            object.metadata.aliases,
            vec!["NGC 224", "IC 10", "PGC 2557", "UGC 454", "C 23"]
        );
        assert_eq!(
            object.metadata.alternate_ids,
            vec![
                "messier:M31",
                "openngc:IC10",
                "vizier:VII/237:PGC2557",
                "vizier:VII/26D:UGC454",
                "caldwell:C23"
            ]
        );
    }

    #[test]
    fn explicit_cross_identifiers_merge_and_preserve_provenance() {
        let mut objects = vec![object("NGC 7000", "openngc:NGC7000", "OpenNGC")];
        let mut incoming = object("LBN 331", "vizier:VII/9:LBN331", "VizieR VII/9/catalog");
        incoming.metadata.aliases.push("NGC 7000".to_string());
        incoming.major_arcmin = Some(120.0);
        let mut index = ObjectIdentityIndex::new(&objects);
        let mut stats = ObjectMergeStats::default();

        index.merge_or_add(&mut objects, incoming, &mut stats);

        assert_eq!(objects.len(), 1);
        assert_eq!(stats.merged, 1);
        assert_eq!(objects[0].major_arcmin, Some(120.0));
        assert_eq!(objects[0].metadata.aliases, vec!["LBN 331"]);
        assert_eq!(
            objects[0].metadata.alternate_ids,
            vec!["vizier:VII/9:LBN331"]
        );
        assert_eq!(
            objects[0].metadata.alternate_sources,
            vec!["VizieR VII/9/catalog"]
        );
    }

    #[test]
    fn explicit_ugc_identity_merges_despite_zero_padding() {
        let mut existing = object("M 31", "openngc:NGC224", "OpenNGC");
        existing.metadata.aliases.push("UGC 00454".to_string());
        existing
            .metadata
            .alternate_ids
            .push("vizier:VII/26D:UGC454".to_string());
        let incoming = object("UGC 454", "vizier:VII/26D:UGC454", "VizieR VII/26D/catalog");
        let mut objects = vec![existing];
        let mut index = ObjectIdentityIndex::new(&objects);
        let mut stats = ObjectMergeStats::default();

        index.merge_or_add(&mut objects, incoming, &mut stats);

        assert_eq!(objects.len(), 1);
        assert_eq!(stats.merged, 1);
        assert_eq!(objects[0].metadata.alternate_sources.len(), 1);
        assert_eq!(
            objects[0].metadata.alternate_sources[0],
            "VizieR VII/26D/catalog"
        );
    }

    #[test]
    fn identical_centers_do_not_merge_without_a_cross_identifier() {
        let mut objects = vec![object("LBN 1", "vizier:VII/9:LBN1", "VizieR VII/9/catalog")];
        let incoming = object("LBN 2", "vizier:VII/9:LBN2", "VizieR VII/9/catalog");
        let mut index = ObjectIdentityIndex::new(&objects);
        let mut stats = ObjectMergeStats::default();

        index.merge_or_add(&mut objects, incoming, &mut stats);

        assert_eq!(objects.len(), 2);
        assert_eq!(stats.added, 1);
        assert_eq!(stats.merged, 0);
    }

    #[test]
    fn repeated_cross_identifier_does_not_collapse_rows_from_one_catalog() {
        let mut objects = vec![object("NGC 7000", "openngc:NGC7000", "OpenNGC")];
        let mut first = object("LBN 1", "vizier:VII/9:LBN1", "VizieR VII/9/catalog");
        first.metadata.aliases.push("NGC 7000".to_string());
        let mut second = object("LBN 2", "vizier:VII/9:LBN2", "VizieR VII/9/catalog");
        second.metadata.aliases.push("NGC 7000".to_string());
        let mut index = ObjectIdentityIndex::new(&objects);
        let mut stats = ObjectMergeStats::default();

        index.merge_or_add(&mut objects, first, &mut stats);
        index.merge_or_add(&mut objects, second, &mut stats);

        assert_eq!(objects.len(), 2);
        assert_eq!(stats.merged, 1);
        assert_eq!(stats.added, 1);
        assert_eq!(objects[1].name, "LBN 2");
    }

    #[test]
    fn metadata_audit_rejects_duplicate_primary_ids() {
        let objects = vec![
            object("LBN 1", "vizier:VII/9:LBN1", "VizieR VII/9/catalog"),
            object("Other", "vizier:VII/9:LBN1", "test"),
        ];
        let error = audit_object_metadata(&objects).unwrap_err();
        assert!(error.to_string().contains("duplicate primary object ID"));
    }

    // A real Tycho-2 record (TYC 1-1-1)
    const SAMPLE: &str = "0001 00008 1| |  2.31750494|  2.23184345|  -16.3|   -9.0| 68| 73| 1.7| 1.8|1958.89|1951.94| 4|1.0|1.0|0.9|1.0|12.146|0.158|12.146|0.223|999| |         |  2.31754222|  2.23186444|1.67|1.54| 88.0|100.8| |-0.2";

    #[test]
    fn parses_a_real_record() {
        let star = parse_tycho2_line(SAMPLE, 2000.0).unwrap();
        assert!((star.ra - 2.31750494).abs() < 1e-8);
        assert!((star.dec - 2.23184345).abs() < 1e-8);
        assert!((star.mag - 12.146).abs() < 1e-3);
        assert_eq!(
            star.tyc,
            StarIdentifier::Tycho2 {
                region: 1,
                number: 8,
                component: 1,
            }
        );
        assert_eq!(star.hip, None);
    }

    #[test]
    fn applies_proper_motion() {
        let star = parse_tycho2_line(SAMPLE, 2025.0).unwrap();
        // pmRA = -16.3 mas/yr over 25 years ≈ -0.41" of RA*cos(dec)
        let d_ra_arcsec = (star.ra - 2.31750494) * 3600.0 * star.dec.to_radians().cos();
        assert!((d_ra_arcsec - -0.4075).abs() < 0.01, "{d_ra_arcsec}");
        let d_dec_arcsec = (star.dec - 2.23184345) * 3600.0;
        assert!((d_dec_arcsec - -0.225).abs() < 0.01, "{d_dec_arcsec}");
    }

    // The real supplement-1 record for Sirius (TYC 5949-2777-1)
    const SIRIUS: &str = "5949 02777 1|H|101.28854105|-16.71314306| -546.0|-1223.1|  1.2|  1.0|  1.3|  1.2|H|      |     |-1.088|0.002|999| | 32349 ";

    #[test]
    fn parses_a_supplement_record_with_proper_motion() {
        let star = parse_tycho2_suppl_line(SIRIUS, 1991.25).unwrap();
        assert!((star.ra - 101.28854105).abs() < 1e-8);
        assert!((star.dec - -16.71314306).abs() < 1e-8);
        assert!((star.mag - -1.088).abs() < 1e-3);
        assert_eq!(
            star.tyc,
            StarIdentifier::Tycho2 {
                region: 5949,
                number: 2777,
                component: 1,
            }
        );
        assert_eq!(star.hip, Some(StarIdentifier::Hipparcos(32349)));

        // Sirius moves fast: ~-546 mas/yr (RA*cos dec), -1223.1 mas/yr (Dec)
        let star = parse_tycho2_suppl_line(SIRIUS, 2025.5).unwrap();
        let dt = 2025.5 - 1991.25;
        let d_dec_arcsec = (star.dec - -16.71314306) * 3600.0;
        assert!((d_dec_arcsec - -1.2231 * dt).abs() < 0.01);
        assert!(star.ra < 101.28854105); // moving in -RA
    }

    #[test]
    fn expands_bright_star_bayer_and_flamsteed_designations() {
        assert_eq!(
            bright_star_designations("  3Alp Lyr"),
            vec!["3 Lyr", "Alp Lyr", "Alpha Lyr", "α Lyr"]
        );
        assert_eq!(bright_star_designations(" 33    Psc"), vec!["33 Psc"]);
        assert_eq!(
            bright_star_designations("   Alp1Cen"),
            vec!["Alp1 Cen", "Alpha1 Cen", "α1 Cen"]
        );
        assert_eq!(
            iau_csn_hr("Vega              Vega              HR 7001      alf"),
            Some(7001)
        );
        assert_eq!(stable_wds_component("A,Ia"), "A%2CIA");
        assert_eq!(stable_wds_component("AB-C"), "AB-C");
        assert_eq!(
            gcvs_detail(&[
                "", "", "", "ROT", "10.0", "(", "0.25", "", "V", "2.5", "", "", "", "",
            ]),
            "ROT; amplitude=0.250V; period=2.5d"
        );
    }

    #[test]
    fn builds_bright_variable_double_and_proper_name_indexes() {
        let dir =
            std::env::temp_dir().join(format!("seiza-stellar-source-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("bsc-identifiers.tsv"),
            "279.234583\t+38.783611\t7001\t  3Alp Lyr\t172167\t 67174\t 699\t11510\t  \tAlp Lyr  \t 0.03\t 0.202\t 0.286\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("gcvs.tsv"),
            "291.3662917\t+42.7843611\tRR Lyr    \tRRAB      \t 7.060\t\t8.120\t\tV\t0.56686776\t-0.110\t-0.196\t2000.000\t\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("wds.tsv"),
            "281.0847473\t+39.6701126\t18443+3940\tSTF2382\tAB   \t 5.150\t 6.10\t123\t2.50\t  11\t  61\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("IAU-CSN.txt"),
            "Vega              Vega              HR 7001      alf   α     Lyr _    18369+3846  0.03  V  91262 172167 279.234735  38.783689 2016-06-30 \n",
        )
        .unwrap();

        let mut builder = StarIdentifierCatalogBuilder::new(2025.5, "test catalogs");
        let stats = add_stellar_identifier_sources(&mut builder, &dir, 2025.5).unwrap();
        assert_eq!(stats.bright, 10);
        assert_eq!(stats.variables, 1);
        assert_eq!(stats.doubles, 2);
        assert_eq!(stats.proper_names, 1);
        let path = dir.join("stars.ids.bin");
        let written = builder.write_to(&path).unwrap();
        assert_eq!(written.numeric_entries, 4);
        assert_eq!(written.name_entries, 10);

        let catalog = seiza::star_ids::StarIdentifierCatalog::open(&path).unwrap();
        let hr = catalog.lookup(StarIdentifier::HarvardRevised(7001));
        let vega = catalog.lookup_name("Vega").unwrap();
        assert_eq!(hr.len(), 1);
        assert_eq!(vega.len(), 1);
        assert_eq!(vega[0].stable_id, "hr:7001");
        assert!((vega[0].ra - hr[0].ra).abs() < 1e-8);
        assert_eq!(catalog.lookup_name("Alpha Lyr").unwrap().len(), 1);
        assert_eq!(
            catalog.lookup_name("RR Lyr").unwrap()[0].detail,
            "RRAB; range=7.060-8.120V; period=0.56686776d"
        );
        assert_eq!(
            catalog.lookup_name("STF 2382 AB").unwrap()[0].detail,
            "AB; sep=2.5arcsec; pa=123deg; mag2=6.10"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn astap_record_decoding_matches_the_documented_sirius_example() {
        // From unit_star_database.pas: RA bytes C3 06 48, DEC bytes D7 39
        // with section dec9 = -24 (0xE8)
        let ra = (0xC3 as f64 + 0x06 as f64 * 256.0 + 0x48 as f64 * 65536.0) * 360.0
            / ((1u32 << 24) - 1) as f64;
        assert!((ra - 101.2871).abs() < 0.001, "{ra}");
        let dec_int = 0xD7_i32 + 0x39_i32 * 256 + (-24) * 65536;
        let dec = dec_int as f64 * 90.0 / ((128 * 65536) - 1) as f64;
        assert!((dec - -16.71614).abs() < 0.0001, "{dec}");
    }

    #[test]
    fn astap_builder_reads_a_synthetic_area_file() {
        let dir = std::env::temp_dir().join(format!("seiza-astap-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let mut data = vec![b' '; 110];
        data[..9].copy_from_slice(b"TEST FILE");
        data[109] = 5;
        // Section: dec9 = -24, magnitude byte 6 => -1.0
        data.extend_from_slice(&[0xFF, 0xFF, 0xFF, (-24i32 + 128) as u8, 6]);
        // Sirius
        data.extend_from_slice(&[0xC3, 0x06, 0x48, 0xD7, 0x39]);
        // Fainter section at dec9 = 0, magnitude byte 116 => 10.0
        data.extend_from_slice(&[0xFF, 0xFF, 0xFF, 128, 116]);
        data.extend_from_slice(&[0x00, 0x00, 0x80, 0x00, 0x40]); // ra=180°, dec small
        std::fs::write(dir.join("test_0101.1476"), &data).unwrap();

        let out = dir.join("out.bin");
        build_astap(&dir, &out, 2025.0, 21.0, 45).unwrap();

        let catalog = seiza::catalog::TileCatalog::open(&out).unwrap();
        assert_eq!(catalog.star_count(), 2);
        use seiza::catalog::StarCatalog;
        let sirius = catalog.cone_search(101.287, -16.716, 0.01, 5);
        assert_eq!(sirius.len(), 1);
        assert!((sirius[0].mag - -1.0).abs() < 0.01);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_magnitude_free_records() {
        let mut broken = SAMPLE.to_string();
        broken.replace_range(110..135, &" ".repeat(25));
        assert!(parse_tycho2_line(&broken, 2000.0).is_none());
    }

    #[test]
    fn canonical_geometry_is_selected_as_an_atomic_source_tuple() {
        use seiza::objects::{ObjectProperty, ObjectSourceRecord};

        let mut first = object("Catalog A", "catalog:a", "Catalog A");
        first.major_arcmin = Some(10.0);
        first.metadata.alternate_ids.push("catalog:b".into());
        let mut second = object("Catalog B", "catalog:b", "Catalog B");
        second.major_arcmin = Some(20.0);
        second.minor_arcmin = Some(5.0);
        second.position_angle_deg = Some(30.0);
        let records = [first.clone(), second.clone()]
            .into_iter()
            .map(|object| ObjectSourceRecord {
                id: object.metadata.id.clone(),
                source: object.metadata.source.clone(),
                object,
                properties: Vec::<ObjectProperty>::new(),
            })
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let data = assemble_object_catalog_data(dir.path(), None, vec![first], records).unwrap();

        assert_eq!(data.objects[0].major_arcmin, Some(10.0));
        assert_eq!(data.objects[0].minor_arcmin, None);
        assert_eq!(data.objects[0].position_angle_deg, None);
        assert_eq!(data.details[0].source_records.len(), 2);
        assert!(data.details[0].geometries.iter().any(|geometry| matches!(
            geometry.data,
            seiza::objects::GeometryData::Ellipse {
                major_arcmin: 20.0,
                minor_arcmin: Some(5.0),
                position_angle_deg: Some(30.0),
                ..
            }
        )));
    }

    #[test]
    fn external_curation_preserves_lbn_measurement_and_selects_correction() {
        use seiza::objects::{ObjectCatalog, ObjectSourceRecord};

        let input = tempfile::tempdir().unwrap();
        let curation = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(input.path().join("outlines/objects")).unwrap();
        std::fs::write(
            input.path().join("outlines/objects/LBN437_lv1.txt"),
            "Object\tCont_Flag\tRAJ2000\tDEJ2000\tX\tY\tLabel_Flag\tInfo\n\
line\t+\t338.0\t40.5\t0\t0\tfalse\t\n\
line\t*\t338.1\t40.6\t0\t0\tfalse\t\n",
        )
        .unwrap();
        std::fs::create_dir_all(curation.path().join("objects")).unwrap();
        std::fs::write(
            curation.path().join("curation.json"),
            r#"{"repository":"https://github.com/example/seiza-catalog-curation","commit":"0123456789abcdef","schema_version":2}"#,
        )
        .unwrap();
        std::fs::write(
            curation.path().join("objects/lbn-437.toml"),
            r#"schema_version = 2
id = "lbn-437"
target_id = "vizier:VII/9:LBN437"
notes = "The catalog ellipse has no position angle."

[[evidence]]
kind = "catalog"
citation = "VizieR VII/9"

[[geometries]]
id = "lbn437-pa"
type = "ellipse"
center_ra_deg = 338.0509
center_dec_deg = 40.5910
major_arcmin = 75
minor_arcmin = 20
position_angle_deg = 90
role = "fallback-extent"
quality = "estimated"
method = "WCS image review"
notes = "Rotate the major axis east-west."
preferred = true

[[geometries.evidence]]
kind = "image-review"
notes = "Two WCS-solved images"

[[outlines]]
file = "LBN437_lv1.txt"
source_record_id = "vizier:VII/9:LBN437"
notes = "Keep the reviewed contour as a candidate."

[[outlines.evidence]]
kind = "upstream-source"
url = "https://example.test/LBN437_lv1.txt"

[[relations]]
kind = "catalog-alias"
related_id = "DG 187"
source_record_id = "vizier:VII/9:LBN437"
notes = "Retain the historical designation."

[[selections]]
facet = "preferred-classification"
source_record_id = "vizier:VII/9:LBN437"
notes = "Use the catalog classification."
"#,
        )
        .unwrap();
        let mut lbn = object("LBN 437", "vizier:VII/9:LBN437", "VizieR VII/9/catalog");
        lbn.ra = 338.0509;
        lbn.dec = 40.5910;
        lbn.major_arcmin = Some(75.0);
        lbn.minor_arcmin = Some(20.0);
        let record = ObjectSourceRecord {
            id: lbn.metadata.id.clone(),
            source: lbn.metadata.source.clone(),
            object: lbn.clone(),
            properties: Vec::new(),
        };
        let data = assemble_object_catalog_data(
            input.path(),
            Some(curation.path()),
            vec![lbn],
            vec![record],
        )
        .unwrap();
        assert_eq!(data.objects[0].position_angle_deg, Some(90.0));
        assert_eq!(data.details[0].source_records.len(), 2);
        assert!(data.details[0].geometries.iter().any(|geometry| matches!(
            geometry.data,
            seiza::objects::GeometryData::Ellipse {
                position_angle_deg: None,
                ..
            }
        )));
        assert_eq!(
            data.provenance.curation.as_ref().unwrap().commit,
            "0123456789abcdef"
        );

        let path = input.path().join("objects.bin");
        ObjectCatalog::from_data(data)
            .unwrap()
            .write_to(&path)
            .unwrap();
        let mapped = ObjectCatalog::open(&path).unwrap();
        mapped.validate().unwrap();
        let details = mapped
            .object_details("vizier:VII/9:LBN437")
            .unwrap()
            .unwrap();
        assert_eq!(details.geometries.len(), 3);
        assert_eq!(details.relations.len(), 1);
        assert!(details.selections.iter().any(|selection| {
            selection.facet == seiza::objects::ObjectFacet::PreferredClassification
                && selection.source_record_id.as_deref() == Some("vizier:VII/9:LBN437")
        }));
        let curated = details
            .geometries
            .iter()
            .find(|geometry| geometry.id == "curation:lbn437-pa#ellipse")
            .unwrap();
        assert_eq!(
            curated.evidence,
            "catalog: VizieR VII/9\nimage-review — Two WCS-solved images"
        );
        let outline = details
            .geometries
            .iter()
            .find(|geometry| geometry.id == "openngc-outline:LBN437_lv1.txt")
            .unwrap();
        assert!(
            outline
                .evidence
                .contains("Keep the reviewed contour as a candidate.")
        );
        assert!(outline.evidence.contains("upstream-source"));
    }

    #[test]
    fn per_object_curation_rejects_unknown_fields_and_duplicate_targets() {
        let curation = tempfile::tempdir().unwrap();
        let objects = curation.path().join("objects");
        std::fs::create_dir_all(&objects).unwrap();
        let first = objects.join("first.toml");
        std::fs::write(
            &first,
            "schema_version = 2\nid = \"first\"\ntarget_id = \"catalog:A\"\nsurprise = true\n",
        )
        .unwrap();
        let error = read_curation_documents(curation.path(), 2).unwrap_err();
        assert!(error.to_string().contains("invalid curation document"));

        std::fs::write(
            &first,
            "schema_version = 2\nid = \"first\"\ntarget_id = \"catalog:A\"\n",
        )
        .unwrap();
        std::fs::write(
            objects.join("second.toml"),
            "schema_version = 2\nid = \"second\"\ntarget_id = \"catalog:A\"\n",
        )
        .unwrap();
        let error = read_curation_documents(curation.path(), 2).unwrap_err();
        assert!(error.to_string().contains("keep one file per object"));
    }

    #[test]
    fn per_object_curation_rejects_outline_paths_on_every_platform() {
        let curation = tempfile::tempdir().unwrap();
        let objects = curation.path().join("objects");
        std::fs::create_dir_all(&objects).unwrap();
        std::fs::write(
            objects.join("unsafe.toml"),
            "schema_version = 2\nid = \"unsafe\"\ntarget_id = \"catalog:A\"\n\n\
             [[outlines]]\nfile = \"nested\\\\outline.txt\"\n",
        )
        .unwrap();

        let error = read_curation_documents(curation.path(), 2).unwrap_err();
        assert!(error.to_string().contains("invalid outline file"));
    }

    #[test]
    fn curation_provenance_excludes_git_metadata() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join(".git/objects")).unwrap();
        std::fs::create_dir_all(root.path().join("objects")).unwrap();
        std::fs::write(root.path().join(".git/HEAD"), "ref: refs/heads/main\n").unwrap();
        std::fs::write(root.path().join(".git/objects/ignored"), "ignored").unwrap();
        std::fs::write(root.path().join(".seiza-revision"), "0123456789abcdef\n").unwrap();
        std::fs::write(root.path().join("objects/kept.toml"), "kept").unwrap();

        let files = recursive_files(root.path()).unwrap();
        assert_eq!(files, vec![root.path().join("objects/kept.toml")]);
    }

    #[test]
    fn parses_disconnected_openngc_contours() {
        let contours = parse_openngc_outline(
            "Object\tCont_Flag\tRAJ2000\tDEJ2000\tX\tY\tLabel_Flag\tInfo\n\
line\t+\t10.0\t20.0\t0\t0\tfalse\t\n\
line\t*\t11.0\t21.0\t0\t0\tfalse\t\n\
line\t+\t30.0\t40.0\t0\t0\tfalse\t\n\
line\t*\t31.0\t41.0\t0\t0\tfalse\t\n",
        )
        .unwrap();
        assert_eq!(contours.len(), 2);
        assert!(contours.iter().all(|contour| contour.closed));
        assert_eq!(contours[1].vertices, vec![(30.0, 40.0), (31.0, 41.0)]);
    }
}
