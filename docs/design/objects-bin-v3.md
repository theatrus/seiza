# Object catalog binary format v3

Status: implemented

## Purpose

`SEIZAOB3` adds identity, hierarchy, provenance, and indexed point/ellipse
storage while making normal catalog use demand-paged. Opening a catalog no
longer creates one `SkyObject` plus several heap allocations for every record.
Viewport and name queries touch only relevant mmap pages and materialize owned
`SkyObject` values only for returned candidates.

The cost moves to disk: fixed records, shared metadata, spatial candidate
lists, and the normalized name index make the current 315,434-object bundle
about 60 MB. Measured release-process peak RSS is about 3.7 MB for an exact
name lookup and 5.8 MB for a 3-degree cone, compared with about 110.7 MB for
the previous eager-decoded implementation.

## Wire format

All integers and floats are little-endian. Every section begins at an 8-byte
boundary. The 104-byte header contains:

```text
[u8; 8] magic = "SEIZAOB3"
u32     header size = 104
u32     declination-band count
u32     object-record count
u32     sky-tile count
u32     tile-candidate count
u32     normalized-name count
u32     list-string-reference count
u32     reserved
u64     UTF-8 string-table bytes
u64     record-section offset
u64     tile-index offset
u64     tile-candidate offset
u64     name-index offset
u64     list-string-reference offset
u64     string-table offset
u64     exact file size
```

Each 84-byte object record contains the kind, f64 ICRS RA/Dec, four optional
f32 measurements (NaN means unknown), four six-byte scalar string references,
and four six-byte list ranges. Scalar references point into one shared UTF-8
table. List ranges point into the packed six-byte string-reference section.

The tile index contains `{ start: u32, count: u32 }` ranges into a packed u32
record-number array. The grid uses equal-height declination bands and fewer RA
bins toward the poles. A point object occurs in its center tile. An extended
object occurs in every tile conservatively intersected by its major-axis
radius, so a bounded query can still find an object whose center is outside the
viewport. Query results are deduplicated before exact spherical intersection.

The sorted name index uses 16-byte entries:

```text
string-ref normalized key
string-ref original matching designation
u32        object record number
```

It indexes primary and common names, aliases, stable IDs, and alternate IDs.
Normalization removes punctuation/spacing and applies Unicode uppercase, so
`M31`, `M 31`, and `m-31` share a key.

## Open, query, and validation behavior

`ObjectCatalog::open` validates only magic, counts, calculated section bounds,
tile geometry, and exact file length. It does not scan records, strings, tile
candidates, or names. `query_region`, `lookup_name`, and `search_names` validate
every reference they touch and return owned results.

`ObjectCatalog::validate` intentionally touches the complete file: UTF-8,
record references and semantics, unique stable IDs, tile ranges and coverage,
candidate ordering, and name-index ordering. The CLI auto-detects v3 with
`seiza catalog validate --data objects.bin`.

`ObjectCatalog::objects` remains available for compatibility, but explicitly
materializes and caches the complete mmap catalog. `read_all` is its fallible,
owned equivalent. Interactive and overlay callers should use indexed queries.

`ObjectCatalog::write_to` writes v3. `write_v1_to` remains for controlled
compatibility exports, and the reader remains backward compatible with the
deployed v1 format.
