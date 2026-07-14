# Offline stellar identifier lookup

Status: `SEIZASI1` sidecar, Tycho-2/Hipparcos importer, library API, and CLI
implemented

## Goal

Resolve a catalog designation such as `TYC 5949-2777-1` or `HIP 32349` to
the star coordinates already selected for a seiza star dataset. This is an
exact offline catalog lookup: it does not inspect an image and does not run a
plate solver.

The identifier data stays in an optional sidecar. Solver-oriented
`SEIZAST2` tiles remain compact and unchanged, and applications that need
identity lookup can memory-map the sidecar once and reuse it.

```shell
seiza build-data tycho2 \
  --input raw/tycho2 \
  --output stars-lite-tycho2.bin \
  --identifier-index stars-lite-tycho2.ids.bin

seiza catalog star \
  --data stars-lite-tycho2.ids.bin \
  "TYC 5949-2777-1" --format json

seiza catalog star \
  --data stars-lite-tycho2.ids.bin \
  "HIP 32349"
```

The library entry point is `StarIdentifierCatalog::lookup`. Exact lookup is a
binary search over sorted, fixed-width records. It returns a list rather than
one record because a catalog identifier can be associated with multiple
resolved components.

## `SEIZASI1` format

```text
8 bytes   magic = "SEIZASI1"
8 bytes   entry count (u64 LE)
8 bytes   coordinate epoch as a Julian year (f64 LE)
2 bytes   attribution byte length (u16 LE)
N bytes   UTF-8 attribution
0-7 bytes zero padding to an 8-byte boundary

24-byte records, sorted by namespace then numeric value:
  1 byte  namespace
  3 bytes reserved
  8 bytes numeric identifier value (u64 LE)
  4 bytes RA packed over [0, 360) (u32 LE)
  4 bytes Dec packed over [-90, 90] (u32 LE)
  2 bytes magnitude, millimagnitudes with a +3 offset (u16 LE)
  2 bytes reserved
```

TYC's three numeric parts are packed into the numeric value. Namespaces are
typed and source-qualified; for example, `hip:32349` cannot collide with
`hd:32349`. The first version defines TYC, HIP, HD/HDE, HR, Gaia DR3, and SAO
namespaces. Reserved bytes allow later flags or record variants without
changing the solver catalog.

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

A release build over the complete 2,539,913-row main catalog and 17,588-row
supplement, with the default magnitude limit of 13, retained 2,544,004 stars
and wrote 2,667,952 identifier entries. The solver tiles were 25,473,888 bytes
and the sidecar was 64,030,976 bytes. On the development machine the combined
build took 1.63 seconds, and a standalone exact CLI lookup measured below
the timer's 0.01-second resolution.

## Catalog roadmap

The useful order is based on what an offline astrophotography UI is likely to
display, not on catalog age.

1. **TYC + HIP: default bright-star identity layer.** They arrive together
   from Tycho-2, cover the lite solver tier, and add little build complexity.
2. **Gaia DR3: optional deep identity layer.** Preserve `source_id` in a Gaia
   build when requested. The ID must stay release-qualified as `gaia-dr3`:
   Gaia documents `source_id` as unique within a release but not stable across
   releases. A full G<=15 sidecar is large, so it should not be silently added
   to the normal solver download.
3. **HD/HDE + HR and curated names: bright-star aliases.**
   [Hipparcos I/239](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/I/239?format=html&tex=true)
   supplies useful cross-identifiers, while the
   [Henry Draper catalogue III/135A](https://cdsarc.cds.unistra.fr/viz-bin/cat/III/135A)
   provides the broader HD/HDE set. Bayer, Flamsteed, and IAU common names are
   strings rather than numeric namespaces and belong in a later normalized
   name/alias table.
4. **WDS: optional double-star pack.** The
   [Washington Double Star Catalog](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/B/wds?format=html&tex=true)
   adds discoverer designations and explicit component labels. Its regularly
   updated nature argues for a separately versioned pack, not baking it into
   every solver file.
5. **GCVS: optional variable-star pack.** The
   [General Catalogue of Variable Stars](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/B/gcvs?format=html&tex=true)
   adds variable-star designations and variability types useful to observing
   and annotation applications.
6. **2MASS and TIC: workflow-specific packs.**
   [2MASS II/246](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/II/246?format=html&tex=true)
   is valuable for infrared work but its hundreds of millions of point
   sources make it unsuitable for the default bundle. TESS Input Catalog IDs
   similarly make sense for exoplanet workflows rather than general solving.

SAO and older astrometric catalogs are useful as accepted aliases when a
cross-match supplies them. They should not become separate default position
layers: Tycho-2, Hipparcos, and Gaia provide the modern astrometry.

## Next API layers

Exact numeric lookup is deliberately first. Two later indexes can remain
independent of it:

- a normalized string alias index for Bayer, Flamsteed, discoverer, variable,
  and common names, with prefix search for interactive autocomplete;
- a spatial identified-star query for returning identifiers inside an
  already-known cone or image footprint.

Keeping these separate lets a small application ship only TYC/HIP exact lookup
while an interactive atlas opts into richer and larger name packs.
