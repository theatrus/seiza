# seiza-cli

The `seiza` command-line tool: star detection, hinted and blind plate
solving, and star/object dataset management for astrophotography. The
library lives in the [seiza](https://crates.io/crates/seiza) crate.

```
cargo install seiza-cli
```

## Solving

```
# Hinted: approximate center (or FITS RA/DEC headers) plus pixel scale
seiza solve image.jpg --data stars.bin --ra 324.8 --dec 57.5 --scale 2.8
seiza solve light.fits --data stars.bin --scale 1.45

# Blind: no position, just a plausible scale range
seiza solve-blind image.jpg --data stars.bin --min-scale 0.5 --max-scale 15

# Annotate detections or list objects in a solved field
seiza detect image.jpg --annotate out.png
seiza solve image.jpg --data stars.bin ... --objects objects.bin

# Query objects when sky bounds are already known; no image or solve needed
seiza catalog objects --data objects.bin --ra 10.6848 --dec 41.2691 --radius 3
seiza catalog objects --data objects.bin \
  --corner 8.91,42.14 --corner 12.47,42.02 \
  --corner 12.31,40.35 --corner 9.02,40.46 \
  --sort prominence --format json
```

`catalog objects` accepts a cone or a convex polygon whose vertices are in
boundary order. It can filter by object kind, magnitude, angular size, and
common-name availability; results can be emitted as a table, JSON, or CSV.
JSON and CSV include stable IDs, source provenance, aliases, and parent IDs
when the catalog provides them. The prominence score is a catalog-based
prediction, not proof that the object is visible in the image pixels.

## Use with N.I.N.A.

seiza speaks ASTAP's CLI contract: set N.I.N.A.'s plate solver to ASTAP
and point the ASTAP path at the seiza binary (a copy named `astap.exe`
also works). Provide a star catalog via the `SEIZA_STAR_DATA`
environment variable or a `stars-*.bin` next to the executable —
`seiza download-data prebuilt` fetches one. Hinted and blind-slot
solving both work; see the repository's `docs/design/astap-mode.md`.

## Datasets

The quickest route is the prebuilt, SHA-256-verified sets hosted at
downloads.seiza.fyi (Tycho-2 lite, Gaia DR3 G≤15, the unified object
catalog, and a nightly-refreshed transient list):

```
seiza download-data prebuilt --output data
seiza download-data prebuilt --output data --file objects.bin --file transients.bin
```

Building from primary sources stays supported for custom depths, epochs,
or tile granularity — note the Gaia TAP download alone can take many
hours:

```
seiza download-data tycho2 --output raw/tycho2
seiza build-data tycho2 --input raw/tycho2 --output stars-lite.bin

seiza download-data gaia --output raw/gaia        # Gaia DR3 via TAP, resumable
seiza build-data gaia --input raw/gaia --output stars-gaia.bin

seiza download-data objects --output raw/objects
seiza build-data objects --input raw/objects --output objects.bin
seiza download-data transients --output raw/transients
seiza build-data transients --input raw/transients --output transients.bin
```

FITS files are read natively (see
[seiza-fits](https://crates.io/crates/seiza-fits)) with automatic
autostretch before detection, and RA/DEC hints taken from headers.

## License

Apache-2.0
