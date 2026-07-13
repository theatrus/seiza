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
```

## Datasets

Download primary catalogs and build the compact tile formats:

```
seiza download-data tycho2 --output raw/tycho2
seiza build-data tycho2 --input raw/tycho2 --output stars-lite.bin

seiza download-data gaia --output raw/gaia        # Gaia DR3 via TAP, resumable
seiza build-data gaia --input raw/gaia --output stars-gaia.bin

seiza download-data openngc objects transients ...
seiza build-data objects --input raw --output objects.bin
seiza build-data transients --input raw/transients --output transients.bin
```

FITS files are read natively (see
[seiza-fits](https://crates.io/crates/seiza-fits)) with automatic
autostretch before detection, and RA/DEC hints taken from headers.

## License

Apache-2.0
