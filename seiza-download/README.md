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
The Gaia DR3 photometry catalog for colour calibration (about 460 MB) is
opt-in the same way: `CatalogSet::gaia_photometry()`. So are the distances to
deep-sky objects in `objects.bin` (about 10 MB): `CatalogSet::object_distances()`.

Learned models are published the same way, under their own manifest:
`ModelManager::ensure("noise-v1-d33")` fetches a model's card and the graphs
it names into the same kind of verified, content-addressed cache, through zstd
where the manifest offers it, and returns their paths.
`ModelManager::ensure_latest("denoise")` fetches the model the manifest's
`latest` names for a task. `DownloadEvent` serializes with its kind under
`event`, so a host can hand progress on as it is.
[`seiza-model`](https://crates.io/crates/seiza-model)'s `ModelStore::download`
installs them. A hosted model can list extras that installing it never
fetches, such as the training weights its graphs were exported from.

Raw Gaia, VizieR, MPC, and other builder inputs intentionally live in
[`seiza-sources`](https://crates.io/crates/seiza-sources).

`OnnxRuntimeInstaller::new(directory)` selects the tested CPU runtime for
Windows x64, Linux x64 or Apple Silicon. `installed()` verifies an existing
install without network access; `install(progress, cancel)` repairs it or
downloads the pinned archives from PSF Guard's mirror, then Microsoft. Every
archive is checked by size and SHA-256. Windows also receives the app-local
VC++ libraries from a separately pinned Seiza CLI package, so a system-wide
redistributable is unnecessary. The returned `InstalledOnnxRuntime` has the
version and absolute library path. No call loads the runtime or selects a
backend. Hosts that bundle DirectML should continue to prefer that library.

Installation writes a fresh directory and atomically switches `current.json`.
Failure or cancellation leaves the previous install intact. Old directories
are retained because another process may have their libraries open; hosts can
remove unused directories when their applications are stopped. Transfer uses
30-second connection and 60-second inactivity timeouts. Receipt checks and
extraction run on the calling thread, so desktop hosts call off the UI thread.
The runtime pins are in `src/onnxruntime.json`; a regression test holds them to
the package builder's `packaging/onnxruntime.env`.
