# Extensible object catalog container v4

Status: implemented

## Decision summary

Object catalog v4 will replace the single flattened object record with a
provenance-preserving canonical object plus all contributing catalog records.
It will also replace the version-numbered monolithic wire layout with a stable,
sectioned container whose sections are independently versioned.

The intent is not to claim that a binary format can never change. A new
container major version remains available if the container envelope itself is
wrong. Adding a catalog, a measurement, an index, a relationship, or a geometry
type must not require an objects v5.

Production corrections and catalog-selection decisions will be maintained in
a separate GitHub repository. Seiza contains the schemas, builder, reader, and
small test fixtures, but no production correction rows. A generated catalog
contains the compiled canonical choices and enough provenance to reproduce
them from the external curation revision.

## Why v3 is not sufficient

`SEIZAOB3` is deliberately compact and demand-paged, but each 84-byte record is
one flattened `SkyObject`. It can retain alternate identifiers and source
labels, but not the complete source records that contributed to the object.

The current builder can also combine optional measurements independently. That
can create a geometry tuple that appeared in no source, such as taking a major
axis from one catalog and a minor axis from another. Geometry must instead be
an atomic, source-qualified observation.

The fixed v3 header and record layout also mean that adding a new field or
geometry representation changes the entire wire format. V4 moves that
evolution boundary down to independently versioned sections and payloads.

## Goals

- Preserve every source-qualified catalog identity and its supplied metadata.
- Nominate one preferred identity while allowing coherent facets such as
  position, geometry, and photometry to select different source records.
- Never synthesize a geometry tuple by merging its components independently.
- Represent ellipses, irregular outlines, and future geometry types without
  changing the core object record.
- Keep ordinary name and viewport queries demand-paged and bounded in memory.
- Allow an older reader to skip unknown optional sections safely.
- Keep exhaustive validation explicit rather than touching the complete mmap
  during normal open.
- Make every generated catalog reproducible from pinned upstream and curation
  revisions.

## Identities, source records, and canonical selection

A source record is an immutable normalized representation of one upstream
catalog row. It retains the source-qualified designation, aliases, coordinates,
classification, measurements, geometry, notes, and upstream provenance.

A canonical object groups source records that explicit catalog identifiers or
curated relationships establish as the same astronomical object. Positional
proximity alone does not create identity.

Relations are typed rather than flattened into aliases:

```text
same-as
component-of
parent-of
duplicate-of
catalog-alias
```

Each canonical object nominates one preferred identity record for display and
stable lookup. Selection records can separately nominate coherent facets:

```text
preferred-identity
preferred-position
preferred-geometry
preferred-photometry
preferred-classification
```

Selectors point to complete source records or measurement groups. In
particular, preferred geometry selects one complete geometry record including
its source center, dimensions, and orientation.

## Geometry model

Every geometry has a common descriptor and a type-specific payload. The common
descriptor contains:

```text
geometry ID
canonical object ordinal
source-record reference
semantic role
geometry type ID and payload schema version
payload offset and length
generic spherical bounding cap
quality and provenance flags
```

The bounding cap is derived indexing data. It allows spatial lookup and a
conservative fallback even when a reader does not understand the detailed
geometry payload.

Initial geometry payloads are:

- `Point`: a position with no measured extent.
- `Ellipse`: center, major axis, minor axis, and optional position angle.
- `OutlineSet`: one or more spherical contour paths, with an optional level or
  semantic label.

An asymmetric ellipse with an unknown position angle remains a valid catalog
measurement, but is not a fully oriented render geometry. A circle does not
need an orientation. Renderers must not silently replace an unknown position
angle with zero.

### OpenNGC geometry

OpenNGC supplies two different kinds of geometry and they must remain distinct:

1. `NGC.csv` and `addendum.csv` contain catalog ellipses expressed as major
   axis, minor axis, and position angle when available.
2. The [`outlines`](https://github.com/mattiaverga/OpenNGC/tree/master/outlines)
   directory contains RA/Dec contour paths at as many as three image-brightness
   levels.

The outlines were hand-drawn against DSS2 imagery in Aladin and simplified as
line strings. A file can contain multiple disconnected contours, and its name
can be only indicative because a contour may include adjacent objects. They are
therefore stored as `OutlineSet`, not promoted to exact physical polygons or
converted into an ellipse. See OpenNGC's
[`metodology.txt`](https://github.com/mattiaverga/OpenNGC/blob/master/outlines/metodology.txt)
and [`shape.py`](https://github.com/mattiaverga/OpenNGC/blob/master/outlines/shape.py).

All outline levels are preserved as candidate geometries. An explicit curation
record maps an outline to a canonical identity and decides whether one level is
preferred for rendering.

### LBN 437 initial curation decision

The VizieR VII/9 LBN record supplies:

```text
identity:             vizier:VII/9:LBN437
center:               338.0509 deg, +40.5910 deg ICRS
major axis:           75 arcmin
minor axis:           20 arcmin
position angle:       unknown
alternate identity:  DG 187
```

The [VII/9 catalog](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=VII%2F9)
contains maximum and minimum dimensions but no position-angle column. Two
independently WCS-solved images inspected during the investigation show that
the current zero-degree rendering is perpendicular to the cataloged feature.
The practical fallback correction is:

```text
geometry:             ellipse
center:               338.0509 deg, +40.5910 deg ICRS
major axis:           75 arcmin
minor axis:           20 arcmin
position angle:       90 deg east of north
role:                 fallback-extent
quality:              estimated
method:               WCS-aligned image review
```

This is an image-derived orientation, not a missing value recovered from the
LBN catalog. It must retain that distinction in provenance.

The full 75-by-20 arcminute catalog extent must also remain distinct from the
much smaller molecular head/core. Scientific observations describe the head as
about 1.6 by 0.4 parsecs, roughly 15 by 4 arcminutes at 360 parsecs, and describe
the overall morphology as curved or comma-shaped rather than a clean ellipse.
See [Soam et al.](https://arxiv.org/abs/1304.1618). A reviewed outline can later
become the preferred render geometry while the ellipse remains a conservative
fallback and indexing extent.

## External curation repository

The proposed source of truth is a separate repository such as
`seiza-catalog-curation`. It stores only Seiza-authored corrections,
relationships, source-outline mappings, and selection decisions. It does not
fork complete upstream catalogs.

Suggested layout:

```text
schema/
  v1/
catalogs/
  identities.csv
  geometry.csv
  selections.csv
  openngc-outlines.csv
outlines/
  <curated contour files when needed>
evidence/
  README.md
```

The root `curation.json` records `repository` and `schema_version`. A normal Git
checkout derives the exact commit from `HEAD` and must be clean; an exported
snapshot additionally records `commit` in that file. The initial CSV contracts
are:

- `geometry.csv`: `correction_id`, `target_id`, complete ellipse center/axes/
  angle, role, quality, method, evidence, note, and `preferred`.
- `selections.csv`: target, facet, source-record ID, geometry ID, and reason.
- `identities.csv`: target, typed relation, related ID, source-record ID, and
  note.
- `openngc-outlines.csv`: upstream filename, target, source-record ID, role,
  preferred flag, evidence, and note.

Missing optional tables are empty. Duplicate corrections, conflicting curated
facet selections, unresolved targets, dangling source/geometry references, and
dirty Git checkouts fail the build.

Every production correction has a stable correction ID, target
source-qualified ID, evidence reference, method, explanation, and quality or
confidence. Curation CI checks:

- schema and unique-key validity;
- that referenced upstream identifiers exist;
- coordinate, axis, and position-angle ranges;
- contour structure and closure flags;
- selection conflicts and dangling references;
- required evidence and attribution fields.

The Seiza object builder accepts local `--input` and `--curation-dir`
inputs and performs no network access. `seiza-sources` owns fetching and caching
both upstream archives and a pinned curation snapshot. Release builds record
the curation repository URL, exact commit, schema version, and file checksums in
the generated provenance section and bundle manifest.

The compiled preferred values live in the generated database so it remains
fast and fully offline. The correction tables themselves are not hard-coded or
maintained in the Seiza repository. A separate runtime curation overlay remains
possible later, but is not required unless applications need to replace or
disable corrections without rebuilding the catalog.

## Stable sectioned container

V4 introduces a stable container magic rather than encoding the object schema
generation in the magic itself:

```text
[u8; 8] magic = "SEIZAOB\0"
u16     container major
u16     container minor
u32     header size
u64     section-directory offset
u32     section count
u32     section-entry size
u64     exact file size
...     reserved envelope fields
```

The implemented envelope freezes a 64-byte header and 96-byte directory
entries. Directory entries carry a 16-byte section ID, section schema version,
required/optional flags, instance ID, byte range, record count/stride, and a
SHA-256 checksum. Sections and fixed records are eight-byte aligned.

A logical section-directory entry contains:

```text
section type ID
section schema major and minor
required/optional and encoding flags
section instance ID
offset and byte length
record count and fixed stride, when applicable
section checksum
```

All offsets and lengths are bounds-checked with checked arithmetic. Sections
are aligned for direct typed access where their schema permits it.

### Implemented initial sections

The first v4 writer embeds the proven v3 canonical query layout as one
required, independently versioned hot section. This keeps the existing
spatial/name index and mmap behavior while the container evolves around it; it
does not make the v3 file envelope part of the v4 envelope contract.

Source records, relations, selections, and geometry descriptors are encoded in
a compact Serde/Postcard payload per canonical ordinal. A fixed detail
index provides direct offset/length lookup. Outline vertices are excluded from
those documents and stored as packed `(f64 RA, f64 Dec)` records, so source
metadata lookup does not page contour coordinates. Future columnar detail
sections can be added alongside this representation without changing the
container major.

| Section | Access | Purpose |
| --- | --- | --- |
| `CANONICAL_V3` | hot, required | Canonical core, strings, spatial tiles, and sorted name index |
| `DETAIL_INDEX` | warm | Canonical ordinal to detail byte range |
| `DETAIL_DATA` | cold | Compact typed source records, relations, selections, and geometry descriptors |
| `OUTLINE_VERTICES` | cold | Packed spherical contour vertices |
| `CAPABILITIES` | hot | Presence bits for records, relations, selections, ellipses, outlines, and provenance |
| `PROVENANCE_JSON` | cold | Source revisions, hashes, curation revision, and build policy |

The object core remains intentionally small. New columns are separate sections
keyed by the dense canonical object ordinal rather than additions to one
ever-growing record.

### Compatibility rules

- A reader accepts the same container major when all unknown required sections
  are absent.
- Unknown optional sections are skipped without being paged in.
- A section's backward-compatible additions increment its minor version.
- An incompatible change increments only that section's major version.
- Adding a new optional section or geometry payload type does not change the
  container major or create an objects v5.
- The container major changes only when the header, section directory, object
  ordinal model, or other envelope-level invariant becomes incompatible.

The public API exposes catalog capabilities so callers can distinguish, for
example, ellipse-only data from catalogs containing detailed outlines.

## Open, mmap, and validation behavior

`ObjectCatalog::open` maps the file and reads only the fixed header, section
directory, capabilities word, and the small headers needed to construct known
hot-section views.
It validates magic, versions, alignment, exact file length, non-overlapping
section bounds, and required capabilities. It does not scan records, strings,
outline vertices, source metadata, or section checksums.

Queries validate references as they are touched. Canonical name and viewport
queries use only the core and relevant indices. Detail lookup first uses the
name index to obtain the canonical ordinal, then pages only its 16-byte detail
index entry and length-delimited document. `catalog_records` does not touch
packed outline vertices; requesting full geometry pages only referenced vertex
ranges.

`ObjectCatalog::validate` and `seiza catalog validate` explicitly scan all
known sections, verify checksums and UTF-8, validate cross-section references,
and check semantic invariants. Unknown optional sections can still have their
stored byte checksum verified without understanding their payload.

## API shape

Existing fast APIs continue to return the preferred canonical `SkyObject`:

```text
lookup_name
search_names
query_region
```

New detail APIs expose provenance without penalizing normal overlay queries:

```text
object_details(canonical_id)
catalog_records(canonical_id)
geometries(canonical_id)
relations(canonical_id)
capabilities()
```

The CLI should display the selected record for each facet and support an
`--all-sources` form for comparison and audit work.

## Hosted transition

V4 is published only in the complete `/data/v4/` bundle. Previously released
paths are immutable compatibility contracts: `/data/` retains `SEIZAOB1`, the
historical standalone `/data/v3/` URL remains reserved for v0.4.0, and the
complete `/data/v2/` bundle retains `SEIZAOB3` for v0.4.1/v0.5. Old readers
therefore never encounter v4 bytes at a URL they already know.

The v4 manifest uses `catalog-bundle-v4-*` and requires each artifact key to
be `artifacts/<sha256>/<name>`. The public `/data/v4/manifest.json` pointer is
published last; content-addressed artifact keys and archived manifests are
never overwritten. See [Hosted catalog bundles](catalog-bundles.md) for the
complete S3 and cache contract.

## Build and publication flow

1. Download and cache upstream catalogs and OpenNGC outlines independently of
   the builder.
2. Fetch a curation snapshot pinned by commit and checksum.
3. Parse every upstream row into an immutable source record.
4. Build identity groups from explicit identifiers and curated relations.
5. Apply deterministic selection policy and curated nominations.
6. Derive conservative geometry bounds and on-disk query indices.
7. Write the sectioned container and provenance.
8. Run exhaustive validation, deterministic rebuild comparison, and release
   profiling for file size, open cost, RSS, and representative queries.
9. Upload the database under its content-addressed `/data/v4/` artifact key.
10. Publish the archived complete manifest and then the current manifest
    pointer, leaving every older compatibility path untouched.

## Acceptance criteria

- LBN 437 retains its original 75-by-20 arcminute unoriented LBN measurement
  and a separate estimated 90-degree fallback geometry with provenance.
- An eventual LBN 437 outline can become preferred without changing the
  container or ellipse-section schema.
- Sharpless and LBN records for the same object remain separately inspectable;
  no hybrid major/minor-axis tuple is created.
- OpenNGC ellipses, multiple contour levels, and disconnected contours remain
  distinct source geometries.
- A reader can open a file containing an unknown optional test section and
  reject an unknown required section with a capability error.
- Normal open does not scan records or verify whole-section checksums.
- Viewport and exact-name queries remain bounded-memory and demand-paged.
- Canonical records are stored in deterministic primary-center tile order for
  spatial page locality; large objects are referenced by every overlapping
  tile without duplicating their records.
- The same pinned inputs produce byte-identical output and a manifest that
  identifies every upstream and curation revision.
