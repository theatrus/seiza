# seiza-download

`seiza-download` asynchronously installs published Seiza catalog bundles into
a shared, content-addressed cache. It verifies new downloads while streaming
them and returns paths that can be opened directly by the memory-mapped readers
in [`seiza`](https://crates.io/crates/seiza).

```rust,no_run
use seiza_download::{CatalogManager, CatalogSet, Dataset};

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let manager = CatalogManager::builder().build()?;
let catalogs = manager.ensure(&CatalogSet::solver_lite()).await?;

let stars = seiza::catalog::TileCatalog::open(catalogs.path(Dataset::StarsLiteTycho2)?)?;
let objects = seiza::objects::ObjectCatalog::open(catalogs.path(Dataset::Objects)?)?;
# let _ = (stars, objects);
# Ok(())
# }
```

The caller supplies the async runtime; this crate never starts one. Network
access occurs only when `ensure` is explicitly awaited. Normal cache hits check
the manifest's file size without hashing or paging through large catalogs.
Call `CatalogBundle::verify` for an exhaustive SHA-256 check.

When a manifest offers a zstd transport, the manager verifies the encoded
stream, decompresses it directly into the cache temporary file, and verifies
the canonical uncompressed bytes before atomically installing them. The cache
contains only the uncompressed file expected by mmap readers. Older readers
ignore the optional encoding metadata and continue downloading the retained
uncompressed artifact.

The default cache follows the platform cache directory and can be overridden
with `SEIZA_CACHE_DIR` or `CatalogManagerBuilder::cache_dir`. The default
selection is the small Tycho-2 solver catalog together with the object, Solar
System, and transient annotation data; every other named set includes the same
annotation data, and `CatalogSet::all()` downloads the complete standard
bundle. A bare star catalog remains available through `CatalogSet::dataset`. The optional Gaia G≤20 deep catalog is about
9 GB and remains opt-in: request `Dataset::StarsDeepGaia20` explicitly, or use
`CatalogSet::blind_deep_gaia20()` to pair it with the G≤16 blind index.

Raw Gaia, VizieR, MPC, and other builder inputs intentionally live in
[`seiza-sources`](https://crates.io/crates/seiza-sources).
