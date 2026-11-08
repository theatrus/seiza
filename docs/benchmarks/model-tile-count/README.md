# Model tile geometry

`Tiling::tiles` previously materialized both axes' tile origins merely to count
them. It now counts each padded axis arithmetically, with checked padding and
multiplication. On Linux x86_64 with rustc 1.99.1, counting the same 1,000,001-tile
grid changed from three allocation/reallocation calls requesting 24,000,008 bytes
to zero allocations. The count is unchanged. This measures requested allocation
bytes, not peak RSS or inference throughput; it does not allocate an image.

Reproduce against either revision after fetching its dependencies:

```sh
python3 docs/benchmarks/model-tile-count/measure.py /tmp/tile-count.txt
```

The probe calls the public native API and asserts the expected count. Its global
allocator measures only the call, excluding printing and runtime initialization.

The included regression logs show the old margin arithmetic panicking in debug
and accepting an invalid margin in release. Model cards previously accepted the
same invalid geometry. The shared core now refuses invalid cards on open, before
runtime loading or store replacement; both C and Python exercise installation
against an already installed valid model. Direct native calls also refuse empty
axes, unrepresentable padding, and overflowing tile counts.

A range test compares the new padding and count with the previous padding formula
and materialized origins across tile sizes, grid multiples, overlaps, and partial
strides. Valid inference is unchanged; the existing model parity fixtures and
pixelwise inference tests cover the shared tiling loop.

Validation against main at `7cfe303`: 2,367 Rust tests passed (19 ignored),
1,698 Python tests passed (24 skipped, including three needing optional ONNX
Runtime), and all 79 stacking/stretch/pipeline output-version checks passed
unchanged. Focused model tests passed in debug and release. Workspace/Python
formatting and Clippy, the Rust 1.92 all-targets/all-features/locked check, and
binding parity passed. The checked-in reproduction script reproduced zero
allocations after the fix.
