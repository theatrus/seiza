# FITS tile integrity and window work

Two independent problems were reproduced through the native Rust API:

- A GZIP_2 tile with two samples plus one trailing decompressed byte was
  accepted. Unshuffling computed an integer sample width, discarded the
  remainder, and only then checked the size. The decoder now checks that
  every pixel has a complete sample before unshuffling. The regression covers
  two-, four-, and eight-byte samples, including the decompression-size guard.
- A one-pixel compressed window walked every tile, allocating start/extent
  vectors for each before rejecting unrelated tiles. It now computes the
  intersecting tile-grid rectangle and visits its table rows directly.
  Only the retained three planes and the first slice of higher axes are read.

The raw failing regression logs are included. The allocation regression used
100,000 valid NOCOMPRESS tiles, all pointing to the same two-byte heap sample:
the original read made 200,098 allocations for one pixel; the fixed read makes
100. The new integration
test bounds the allocation count rather than using a flaky wall-clock limit.

## Release measurements

Linux x86_64, rustc 1.99.1 (58cf8f9c3 2026-10-13), external release-profile
binary, five reads per file. Every GZIP_2 tile describes a single Float32
sample equal to 42.0, sharing its compressed heap array. The requested pixel
is in the last tile; all tile descriptors are valid. The fixture is created
before timing, and every result is checked against 42.0.

| Tile count | Before median | After median |
| --- | ---: | ---: |
| 1,000 | 42.9 µs | 17.1 µs |
| 100,000 | 2.41 ms | 23.6 µs |
| 1,000,000 | 30.5 ms | 22.3 µs |

Raw samples are in `before.json` and `after.json`. These synthetic files
isolate tile-index traversal, not throughput on typical acquisition frames.
The repeated-read samples include file opening and mapping; they are not
cold-cache I/O measurements. The before run used the original traversal;
the after run also includes the shuffled-length validation fix. This speedup
applies to the mapped path: fallback reads still copy the table into memory.

To reproduce against either revision with dependencies cached:

```sh
python3 docs/benchmarks/fits-tile-windows/measure.py /tmp/tile-results.json
```

## Correctness and output versions

`gzip_2_lights_read_whole_keep_their_output` and
`gzip_2_lights_read_in_windows_keep_their_output` use non-square tiles with
partial edges and windows that cross tile boundaries. They reproduce the
existing MONO_DEFAULT stack summary without changing its recorded values.
Both passed on the original implementation. Temporarily refusing GZIP_2
samples made the whole-image case fail; temporarily skipping narrow regions
made the window case fail. Both mutations were removed.

Native tests additionally compare mapped and unmapped windows of four-axis
images: tiles spanning multiple planes and higher slices, partial edge tiles,
tiles larger than the image, clipped windows, and empty windows. Expected
samples come directly from coordinates, independently of tile enumeration.
All stacking/stretch/pipeline output-version checks remain unchanged; no
OUTPUT_VERSION or recorded summary changes are needed.

## Final validation

On the branch based on PR #446 (the batch compile-memory fix):

- Full workspace, all features: 2,267 passed, 19 ignored.
- Python runtime suite, freshly built with maturin in an isolated Python 3.12
  environment: 1,637 passed, 21 skipped.
- All 77 stacking/stretch/pipeline output-version checks pass unchanged.
- FITS and image-I/O tests pass in debug and release builds.
- Workspace and Python formatting and Clippy pass; Rust 1.89 workspace
  all-targets/all-features locked checking and binding parity pass.

The Python test tools were installed into a temporary environment, resolving
this environment's previous inability to run the runtime binding suite.
