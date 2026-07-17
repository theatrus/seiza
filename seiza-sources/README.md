# seiza-sources

`seiza-sources` asynchronously acquires the upstream astronomy distributions
used to build custom Seiza catalogs: Tycho-2, Gaia DR3 TAP queries, OpenNGC,
selected VizieR catalogs, stellar identifiers, transients, MPC, and JPL SBDB.
The OpenNGC acquisition includes both database CSVs and the hand-drawn contour
files under `outlines/objects`; the object builder associates outlines only
through explicit curation mappings.

```rust,no_run
use seiza_sources::SourceDownloader;

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let sources = SourceDownloader::new()?;
sources.download_objects("raw/objects").await?;
sources
    .download_curation(
        "theatrus/seiza-catalog-curation",
        "0123456789abcdef0123456789abcdef01234567",
        "raw/curation",
    )
    .await?;
sources.download_gaia("raw/gaia", 15.0, 768).await?;
# Ok(())
# }
```

Downloads stream into partial files and are moved into place only after their
available integrity checks pass. Gaia chunks are resumable across runs and are
retried asynchronously. A caller-provided event reporter can drive logs or a
progress UI without library output on stdout.

This crate is for catalog-building inputs. Applications that need ready-to-open
Seiza `.bin` and `.idx` files should use
[`seiza-download`](https://crates.io/crates/seiza-download), which provides the
versioned runtime bundle cache.
