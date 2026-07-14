# Offline stellar identifier lookup

Status: `SEIZASI1` sidecar; Tycho-2, Bright Star, GCVS, WDS, and IAU-name
importers; exact and prefix APIs; and CLI implemented

## Goal

Resolve a designation such as `TYC 5949-2777-1`, `HIP 32349`, `Vega`,
`RR Lyr`, or `STF 2382 AB` to catalog coordinates. This is offline catalog
lookup: it does not inspect an image and does not run a plate solver.

The identifier data stays in an optional sidecar. Solver-oriented
`SEIZAST2` tiles remain compact and unchanged, and applications that need
identity lookup can memory-map the sidecar once and reuse it.

```shell
seiza download-data star-identifiers --output raw/star-identifiers

seiza build-data tycho2 \
  --input raw/tycho2 \
  --output stars-lite-tycho2.bin \
  --identifier-index stars-lite-tycho2.ids.bin \
  --identifier-sources raw/star-identifiers

seiza catalog star \
  --data stars-lite-tycho2.ids.bin \
  "TYC 5949-2777-1" --format json

seiza catalog star \
  --data stars-lite-tycho2.ids.bin \
  "HIP 32349"

seiza catalog star --data stars-lite-tycho2.ids.bin "RR Lyr"
seiza catalog star --data stars-lite-tycho2.ids.bin "STF 2382 AB"
seiza catalog star --data stars-lite-tycho2.ids.bin "RR L" --prefix --limit 10
```

The library entry points are `StarIdentifierCatalog::lookup` for typed numeric
IDs, `lookup_name` for exact textual designations, `lookup_query` for either,
and `search_names` for interactive prefix completion. All are binary searches
over sorted memory-mapped records. They return lists because a catalog
identifier can be associated with multiple components and a WDS coordinate
identifier intentionally represents multiple discoverer/component rows.

`StarIdentifierCatalog::open` validates only the header and section bounds; it
does not scan the mapped records or string table. Text lookups validate the
O(log n + k) records and strings they touch, where `k` is the result count.
Call `validate` explicitly when a
complete integrity check of an untrusted sidecar is required; that operation
intentionally reads the whole mapping.

## `SEIZASI1` format

```text
8 bytes   magic = "SEIZASI1"
8 bytes   numeric record count (u64 LE)
8 bytes   textual-name record count (u64 LE)
8 bytes   string-table byte count (u64 LE)
8 bytes   coordinate epoch as a Julian year (f64 LE)
2 bytes   attribution byte length (u16 LE)
N bytes   UTF-8 attribution
0-7 bytes zero padding to an 8-byte boundary

24-byte numeric records, sorted by namespace then numeric value:
  1 byte  namespace
  3 bytes reserved
  8 bytes numeric identifier value (u64 LE)
  4 bytes RA packed over [0, 360) (u32 LE)
  4 bytes Dec packed over [-90, 90] (u32 LE)
  2 bytes magnitude, millimagnitudes with a +3 offset (u16 LE)
  2 bytes reserved

40-byte textual records, sorted by normalized lookup key:
  1 byte  source catalog
  1 byte  semantic kind (proper, Bayer/Flamsteed, variable, or double)
  2 bytes reserved
  ranges  normalized key, display designation, stable ID, and detail
  4 bytes RA packed over [0, 360)
  4 bytes Dec packed over [-90, 90]
  2 bytes optional packed magnitude; 65535 means unavailable
  2 bytes reserved

UTF-8 string table shared by all textual records
```

TYC's three numeric parts are packed into the numeric value. Namespaces are
typed and source-qualified; for example, `hip:32349` cannot collide with
`hd:32349`. The first version defines TYC, HIP, HD/HDE, HR, Gaia DR3, SAO,
and FK5 numeric namespaces. Text keys are Unicode case-normalized, insensitive
to ordinary spaces and punctuation, and preserve the signs in WDS coordinate
IDs and meaningful WDS component separators.

## Initial data

[Tycho-2 I/259](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/I/259?format=html&tex=true)
is the right first source because every row supplies its three-part TYC
designation and the catalog already carries a HIP cross-identification where
one exists. The builder retains both identifiers at the requested output
epoch and magnitude limit. This produces:

- one TYC entry for every retained Tycho-2 or supplement-1 row;
- an additional HIP entry when that row contains a Hipparcos number;
- more than one result when a supplied cross-identification applies to
  multiple components.

The canonical display form follows the Tycho-2 documentation:
`TYC region-running-component`. Stable API IDs use
`tycho2:region-running-component` and `hip:number`.

A release build over the complete Tycho-2 main catalog and supplement plus the
complete downloaded name sources, with the default magnitude limit of 13,
retained 2,544,004 solver stars and wrote 2,698,290 numeric identifiers plus
387,099 textual names after exact-record deduplication. The importers accepted
40,543 Bright Star aliases,
60,619 valid GCVS variables, 315,826 WDS coordinate/discoverer designations,
and 451 IAU proper names. The solver tiles remained 25,473,888 bytes; the
enriched sidecar was 100,435,815 bytes, including variable ranges/periods and
double-star geometry. On the development machine the combined build took 2.14
seconds and a standalone named lookup took 0.02 seconds.

Bright Star proper motions propagate HR/HD/SAO/FK5, Bayer/Flamsteed, ADS, and
variable aliases to the requested output epoch. IAU names link to their HR
record when available, so `Vega`, `Alpha Lyr`, and `HR 7001` return the same
propagated position and stable ID. GCVS and WDS positions are also propagated
when their source rows supply proper motion; otherwise their J2000 position is
retained.

## Catalog roadmap

The useful order is based on what an offline astrophotography UI is likely to
display, not on catalog age.

1. **Done — TYC + HIP: default identity layer.** They arrive together
   from Tycho-2, cover the lite solver tier, and add little build complexity.
2. **Done — bright-star identities and names.** The
   [Bright Star Catalogue](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/V/50?format=html&tex=true)
   supplies HR, HD, SAO, FK5, Bayer/Flamsteed, ADS, and variable identifiers;
   the IAU Catalog of Star Names adds proper names.
3. **Done — WDS double-star identities.** The
   [Washington Double Star Catalog](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/B/wds?format=html&tex=true)
   adds coordinate IDs, discoverer designations, component labels, latest
   separation/position angle, and component magnitudes. Multiple results for
   one WDS coordinate ID are intentional.
4. **Done — GCVS variable-star identities.** The
   [General Catalogue of Variable Stars](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/B/gcvs?format=html&tex=true)
   adds variable-star designations, variability types, magnitude ranges, and
   periods when published.
5. **Next — Gaia DR3: optional deep identity layer.** Preserve `source_id` in
   a Gaia build when requested. The ID must stay release-qualified as
   `gaia-dr3`; a full G<=15 sidecar is large, so it should not be silently
   added to the normal solver download.
6. **Later — full HD/HDE and workflow-specific 2MASS/TIC packs.** The
   [Henry Draper catalogue III/135A](https://cdsarc.cds.unistra.fr/viz-bin/cat/III/135A)
   extends beyond the bright subset, while
   [2MASS II/246](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/II/246?format=html&tex=true)
   is valuable for infrared work but its hundreds of millions of point
   sources make it unsuitable for the default bundle. TESS Input Catalog IDs
   similarly make sense for exoplanet workflows rather than general solving.

SAO and older astrometric catalogs are useful as accepted aliases when a
cross-match supplies them. They should not become separate default position
layers: Tycho-2, Hipparcos, and Gaia provide the modern astrometry.

## Next API layer

The remaining useful addition is a spatial identified-star query that returns
identifiers and names inside an already-known cone or image footprint. The
current exact and prefix indexes deliberately remain independent of solver
tiles so a small application can ship only TYC/HIP while an interactive atlas
opts into the richer name sources.
