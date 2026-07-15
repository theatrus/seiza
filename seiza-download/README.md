# seiza-download

`seiza-download` asynchronously installs published Seiza catalog bundles into
a shared, content-addressed cache. It verifies new downloads while streaming
them and returns paths that can be opened directly by the memory-mapped readers
in [`seiza`](https://crates.io/crates/seiza).

```rust,no_run
use seiza_download::{CatalogManager, CatalogSet, Dataset};

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let manager = CatalogManager::builder().build()?;
let catalogs = manager
    .ensure(&CatalogSet::solver_lite().with(Dataset::Objects))
    .await?;

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

The default cache follows the platform cache directory and can be overridden
with `SEIZA_CACHE_DIR` or `CatalogManagerBuilder::cache_dir`. The default
selection is the small Tycho-2 solver catalog; downloading the complete bundle
requires `CatalogSet::all()`.

Raw Gaia, VizieR, MPC, and other builder inputs intentionally live in
[`seiza-sources`](https://crates.io/crates/seiza-sources).
