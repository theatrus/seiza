# Coordinate-only catalog lookup and feature overlays

Status: indexed coordinate/name object API and CLI implemented; feature-overlay
data model proposed

The provenance-preserving object and geometry model is specified in the
[extensible object catalog v4 proposal](objects-bin-v4.md).

## Goal

When an application already knows an image's sky bounds, identify the catalog
objects expected in that field without detecting stars or plate solving the
pixels. A cone is sufficient for a candidate list. Ordered sky corners plus
image dimensions can later support approximate pixel placement without asking
seiza to solve the image.

This is catalog association, not pixel recognition. The API deliberately calls
its ranking `predicted_prominence`: an object can be geometrically present but
invisible in a particular exposure, bandpass, stretch, or sky condition.

## API and CLI

The library exposes `SkyRegion`, `ObjectQuery`, `ObjectHit`, and
`ObjectCatalog::query_region`. Regions are either sky cones or convex spherical
polygons in boundary order. Polygon containment uses unit vectors and
great-circle edges, so footprints crossing RA 0 degrees work normally.

Object centers are tested exactly. An object's catalog extent is conservatively
treated as a circle using its major-axis radius for boundary intersection; the
original major/minor axes and position angle remain on `SkyObject`.

```shell
seiza catalog objects \
  --data objects.bin \
  --ra 10.6848 --dec 41.2691 --radius 3 \
  --sort prominence --format json

seiza catalog objects \
  --data objects.bin \
  --corner 8.91,42.14 --corner 12.47,42.02 \
  --corner 12.31,40.35 --corner 9.02,40.46 \
  --kind galaxy,nebula --min-size 2 --format csv
```

Table, JSON, and CSV outputs distinguish center matches from extent-only
matches. Filters cover object type, magnitude, angular size, common names, and
result count.

For interactive use, load one `ObjectCatalog` per process and reuse it while
the viewport pans or zooms. Debounce viewport queries in the UI and discard
stale results when newer bounds arrive. Object catalog v4 memory-maps a
sectioned container and uses an embedded sky-tile candidate index, so a
viewport query decodes only records near the requested bounds. Canonical
records are stored in primary-center tile order for page locality. Large
objects are listed in every tile
touched by their conservative major-axis extent, preserving extent-only hits.

The same file contains a sorted normalized index over primary/common names,
aliases, stable IDs, and alternate IDs. Exact and prefix queries are available
through `ObjectCatalog::lookup_name`, `ObjectCatalog::search_names`, and:

```shell
seiza catalog object --data objects.bin "M 31"
seiza catalog object --data objects.bin "openngc:NGC224"
seiza catalog object --data objects.bin "andro" --prefix --limit 10
seiza catalog object --data objects.bin "openngc:NGC224" --all-sources --format json
```

Normal open checks header and section bounds only. `ObjectCatalog::validate` or
`seiza catalog validate --data objects.bin` deliberately scans all sections,
checksums, records, strings, details, and indices when full integrity
validation is needed. Deployed `SEIZAOB1` and `SEIZAOB3` files remain readable;
v1 requires heap decoding while v3 remains mmap-backed.

## Which catalog should supply sub-objects?

There is no single authoritative all-sky catalog of visually named structures
such as the Cygnus Wall, pillars, dust lanes, or spiral-arm segments. These
need three deliberately separate sources.

### Published astronomical regions: VizieR

[VizieR](https://vizier.cds.unistra.fr/) is the primary source for published
region catalogs. The existing object builder already ingests Sharpless
([VII/20](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=VII%2F20)),
Lynds dark nebulae (VII/7A), Barnard, and van den Bergh reflection nebulae
([VII/21](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=VII%2F21)).

The default object builder now includes the two highest-value additions for
broad visible nebulosity:

- Lynds' bright-nebula catalog
  ([LBN, VII/9](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=VII%2F9));
- Cederblad bright diffuse Galactic nebulae
  ([VII/231](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=VII%2F231));

Logical follow-on catalogs are RCW southern H-alpha regions and other
published regional catalogs selected from VizieR. Galaxy-specific H II-region
catalogs, such as PHANGS products, fit better as optional packs than as part of
the all-sky default.

These describe physical/cataloged regions. Most supply a center and sometimes
a diameter, brightness class, or morphology, but rarely a faithful visual
outline. LBN's `ID` field says whether a brightness region belongs to a more
extensive complex, but does not identify a containing catalog record, so the
builder does not invent a `parent_id` from it.

### Identifiers and hierarchy: SIMBAD enrichment

[SIMBAD](https://simbad.cds.unistra.fr/simbad/) supplies cross-identifiers,
object types, and parent/child hierarchy links. It is useful for deduplication
and for connecting a region to a containing nebula, galaxy, or cluster.
SIMBAD explicitly describes itself as a database rather than a catalog, so it
should not become the base bulk dataset. Its ODbL terms also mean derived
bundles need explicit data licensing and attribution separate from seiza's
Apache-2.0 source code.

### Informal visible structures: a curated seiza feature layer

Features such as the Cygnus Wall need a small curated overlay with provenance,
not an invented scientific designation in `objects.bin`. A future
`features.bin` should contain:

```text
id, display_name, aliases
parent_object_ids[]
kind (ridge, wall, pillar, dust-lane, arm, knot, shell, other)
geometry (point, ellipse, polygon, or MOC)
source_reference, source_url, confidence
bandpass_hint, min_fov_deg, max_fov_deg, display_priority
```

Polygons are suitable for hand-reviewed outlines. For large or complicated
coverage, the IVOA Multi-Order Coverage representation used by
[Aladin](https://aladin.cds.unistra.fr/java/FAQ.html) is a natural later
extension. Every informal feature must carry provenance and confidence so the
UI can distinguish catalog facts from editorial annotations.

## Rollout sequence

1. Done: use object metadata for source-qualified IDs, aliases,
   parent links, and provenance while retaining v1 read compatibility.
2. Done: add LBN and Cederblad through the existing VizieR build pipeline, with
   explicit cross-identifier merging and no positional deduplication. LBN can
   intentionally assign the same center to distinct regions.
3. Done: add mmap-native fixed records plus on-disk sky and normalized-name
   indices for bounded-memory viewport, autocomplete, and direct retrieval.
4. Next: define a source-controlled `features.json` schema and seed a small
   reviewed set of genuinely useful named structures.
5. Later: add optional image-evidence scoring inside predicted feature regions.
   This verifies visibility but still does not require a global plate solve
   when the supplied bounds and orientation are trusted.
