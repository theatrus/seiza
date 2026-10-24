# Catalogs and data

[Documentation](README.md) · [Download and query performance](performance.md#catalogs-and-downloads)

## Install

```sh
seiza setup
```

Choose a preset for hinted solving, blind solving, or all published data. Each
preset includes object search, Solar System objects, active transients, and at
least one solving catalog. Seiza verifies each versioned download with SHA-256.

For an explicit portable directory:

```sh
seiza download-data prebuilt --output data
```

Seiza installs a consistent set of files from the versioned
[bundle manifest](https://downloads.seiza.fyi/data/v4/manifest.json) and
decompresses downloads into files it can memory-map. The standard bundle
contains the following files. Sizes and counts are approximate; consult the
manifest for the downloaded version.

| File | Contents | Uncompressed size |
|---|---|---:|
| `stars-lite-tycho2.bin` | 2.5 million Tycho-2 stars | 25 MB |
| `stars-gaia.bin` | 36.7 million Gaia DR3 stars, G ≤ 15 | 367 MB |
| `stars-deep-gaia17.bin` | 154.1 million Gaia DR3 stars, G ≤ 17 | 1.54 GB |
| `blind-gaia16.idx` | Prebuilt whole-sky pattern index, G ≤ 16 | 1.63 GB |
| `stars-lite-tycho2.ids.bin` | 2.7 million numeric identifiers, 387,000 names | 100 MB |
| `objects.bin` | About 315,000 objects with indexed names and geometry | 241 MB |
| `minor-bodies.bin` | Comets and asteroids | 16 MB |
| `transients.bin` | Active supernovae/novae, refreshed nightly | 2 MB |

The G ≤ 17 star catalog and G ≤ 16 index support small, fine-scale fields where
the G ≤ 15 catalog contains too few bright stars to match.
Optional distance data adds distances, uncertainty, methods, and sources for
about 180,000 objects.

Install the optional Gaia photometry catalog for offline
[photometric color calibration](image-processing.md#photometric-color-calibration):

```sh
seiza setup --preset solver-lite --gaia-photometry
# Or download just this artifact into a portable directory:
seiza download-data prebuilt --output data --file stars-gaia-photometry.bin
```

For fainter fields, choose the `blind-deep-gaia20` preset. It installs
`stars-deep-gaia20.bin` (about 9 GB) and uses the G ≤ 16 blind index.
The `all` preset includes the G ≤ 20 catalog, photometry, object distances,
and star distances. A plain `download-data prebuilt` installs the standard
bundle in the table. To add distance data to a smaller setup preset, use
`--object-distances` or `--star-distances`.

## Locations

`--data` and `--index` accept explicit files or directories. Without them, the
resolver checks configured overrides, files beside the program, and standard
setup locations. A directory selects the deepest star catalog present.

- `SEIZA_CATALOG_DIR`: default setup and application catalog directory.
- `SEIZA_STAR_DATA`: explicit solving-star catalog override.
- `SEIZA_BLIND_INDEX`: explicit blind-index override.

The all-users Windows installer sets `SEIZA_CATALOG_DIR` to
`%ProgramData%\Seiza\catalogs`. Rust hosts can use `seiza::data_paths` for the
same discovery rules. See [data-path details](design/data-paths.md).

## Query and validate

```sh
seiza catalog object --data data "Andromeda Galaxy"
seiza catalog objects --data data --ra 10.6848 --dec 41.2691 --radius 3 --format json
seiza catalog star --data data "TYC 5949-2777-1" --format json
seiza catalog distance "M 42"
seiza catalog object --data data "M 42" --all-sources
seiza catalog validate --data data/objects.bin
```

Object catalogs combine OpenNGC, Sharpless, Barnard, UGC, LDN/LBN, Cederblad,
vdB, PGC, Galactic supernova remnants, Wolf-Rayet stars, planetary nebulae,
named/HD stars, and transients. Star identifiers include TYC/HIP/HR/HD/SAO/FK5,
IAU and Bayer/Flamsteed names, GCVS variables, and WDS double stars.

The [v4 object container](design/objects-bin-v4.md) stores contributing records,
relations, source geometry (including OpenNGC contours), and build provenance.
Legacy `SEIZAOB1` and `SEIZAOB3` files remain readable. Normal opens avoid a full
scan. Run `catalog validate` to check the entire file for integrity errors.

## Embed downloads

Use [`seiza-download`](../seiza-download/README.md) to install verified catalogs
from an application. Enable `seiza`'s non-default `downloads` feature to call
the download API through `seiza`:

```rust,no_run
let manager = seiza::downloads::CatalogManager::builder().build()?;
let files = manager
    .ensure(&seiza::downloads::CatalogSet::solver_lite()
        .with(seiza::downloads::Dataset::Objects))
    .await?;
let stars = seiza::catalog::TileCatalog::open(
    files.path(seiza::downloads::Dataset::StarsLiteTycho2)?,
)?;
```

## Custom builds

Use prebuilt data unless you need a different depth, epoch, or upstream source.
Raw acquisition lives in [`seiza-sources`](../seiza-sources/README.md), separate
from runtime downloads. Gaia archive downloads can take hours.

```sh
seiza download-data gaia --output raw/gaia --max-mag 17 --chunks 3072
seiza build-data gaia --input raw/gaia --output stars-deep.bin --max-mag 17
seiza build-blind-index --data stars-deep.bin --output blind-gaia16.idx --index-mag-limit 16
```

See the [dataset CLI reference](../seiza-cli/README.md#datasets),
[bundle design](design/catalog-bundles.md), and
[star identifiers](design/star-identifiers.md).
