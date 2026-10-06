# Constellation lines

Stick figures for the 88 IAU constellations, embedded in the `seiza` crate by
`seiza::constellations`.

## `ConstellationLines.csv`

Constellation Lines dataset by Marc van der Sluys (2005-2023),
hemel.waarnemen.com. DOI: 10.5281/zenodo.10397192. Licensed under CC BY 4.0.

- Source: <https://github.com/MarcvdSluys/ConstellationLines>, tag `v1.3`
  (commit `ebb779f`), fetched from raw.githubusercontent.com.
- Unmodified. SHA-256
  `4a723a979c2fe9ea2984a7515b4caab85ff2160adebfe25ac9694e2a928a301d`.
- Licence: CC BY 4.0, full text in [`LICENSE`](LICENSE), copied unmodified
  from the same tag. The upstream readme and `CITATION.cff` also give CC BY
  4.0. The comment header of the upstream `ConstellationLines.dat` still names
  CC BY-SA 4.0; that file is not vendored here, and the repository's licence
  file, readme and citation metadata all say CC BY 4.0.

Each data line is one polyline of Bright Star Catalogue (HR) numbers:
`abbr, count, hr, hr, ...`. A blank abbreviation continues the previous
constellation (Crux), and Serpens has a line for each of its two parts. Lines
may retrace themselves; `seiza` joins consecutive stars and keeps each
segment once.

## `line-stars.tsv`

Positions and V magnitudes of the 697 stars the lines use, so figures draw
without any catalog download. Derived from the Bright Star Catalogue, 5th
Revised Ed. (Hoffleit & Warren 1991; CDS catalogue V/50, via VizieR, CDS,
Strasbourg): J2000 positions propagated with the BSC proper motions to epoch
2025.5, as stored in the `hr:` entries of seiza's star-identifier sidecar
(`stars-lite-tycho2.ids.bin`). Regenerate it with

```
cargo run -p seiza --example constellation_line_stars -- \
    /path/to/stars-lite-tycho2.ids.bin > seiza/data/constellation-lines/line-stars.tsv
```

The constellation names in `seiza::constellations::CONSTELLATION_NAMES` are
the IAU abbreviations and Latin names.

Anything that shows these figures must credit the dataset as above;
`seiza::constellations::ATTRIBUTION` holds the line, and `seiza solve
--sky-map` prints it on the map.
