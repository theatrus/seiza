# Object catalog binary format v2

Status: implemented

## Purpose

`SEIZAOB2` adds stable identity, aliases, hierarchy, and provenance to the
compact object catalog. These fields let applications merge offline VizieR,
OpenNGC, and curated data without treating a display name as a database key.
They also let a cataloged sub-region refer to one or more containing objects.

Detailed visible-feature geometry remains a separate future `features.bin`
layer. `objects.bin` continues to describe astronomical catalog objects with a
point or ellipse.

## Wire format

All integers and floats are little-endian. The file starts with:

```text
8 bytes   magic = "SEIZAOB2"
u32       object_count
```

Each object is a length-delimited record:

```text
u32       record_bytes
u8        kind
f64       ICRS right ascension, degrees
f64       ICRS declination, degrees
f32       visual magnitude; NaN when unknown
f32       major axis, arcminutes; 0 when unknown
f32       minor axis, arcminutes; 0 when unknown
f32       position angle east of north; NaN when unknown
string    display designation
string    common name
string    stable source-qualified ID
string    source/provenance label
strings   alternate designations
strings   stable IDs of containing parent objects
strings   stable IDs assigned by other source catalogs
strings   other contributing source labels
...       future fields may be appended inside record_bytes
```

A `string` is a u16 UTF-8 byte length followed by those bytes. A `strings`
value is a u16 item count followed by that many strings. Writers reject values
that exceed those limits instead of silently truncating them.

The record length is the compatibility boundary: a v2 reader parses the known
prefix and skips any unknown tail. A safety limit rejects individual records
larger than 16 MiB.

## Identity conventions

IDs are opaque to the library. Dataset builders use source-qualified values,
for example:

```text
openngc:NGC224
vizier:VII/20:Sh2-101
vizier:VII/237:PGC2557
iau-csn:Sirius
```

The source field records the contributing dataset or VizieR table. Aliases are
alternate catalog designations, not informal feature labels. Parent IDs may be
empty and may contain more than one value when a region participates in
multiple useful containment relationships. When records from multiple catalogs
are confidently cross-identified, the retained object carries the other stable
IDs and contributing sources in dedicated v2 fields instead of producing
duplicate viewport hits.

## Compatibility and rollout

`ObjectCatalog::write_to` writes v2. `ObjectCatalog::open` reads both
`SEIZAOB1` and `SEIZAOB2`; v1 records receive empty metadata. A controlled
`write_v1_to` compatibility export is available, but necessarily drops all v2
metadata.

Publishing a v2 hosted catalog therefore requires a reader upgrade, while the
new reader can continue using every previously published v1 catalog.
