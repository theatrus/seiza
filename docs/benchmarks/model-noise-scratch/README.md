# Native model noise scratch

The model's image-noise estimator previously allocated three f64 scratch buffers
for every block. Rayon jobs now retain and clear their scratch across blocks.
Constant-noise mode also fills the existing noise map in place after finding its
finite median, instead of allocating another image-sized output.

Before is main at `7cfe303`; after is this change.

Measured on Linux x86_64, rustc 1.99.1, release build, a deterministic 4096 × 4096
image with a gradient, a bright region, white noise and missing pixels, 32-pixel
noise blocks, and nine measured repetitions after warm-up per case:

| Mode | Threads | Before median | After median | Before allocation calls | After median allocation calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Image | 1 | 222.512 ms | 213.043 ms | 49,217 | 71 |
| Constant | 1 | 290.974 ms | 267.292 ms | 49,241 | 94 |
| Image | 8 | 45.263 ms | 43.693 ms | 49,217 | 2,984 |
| Constant | 8 | 121.332 ms | 96.250 ms | 49,241 | 2,923 |

Peak RSS across the four cases fell from 276,928 KiB to 210,216 KiB (24.1%).
This is preprocessing only, not end-to-end model inference; no ONNX Runtime is
loaded. Timing differences for image-noise mode are small and should not be
read as a stable throughput gain. The constant-map case removes a 64 MiB output
allocation at this size. Allocation requests include all preprocessing stages,
not only scratch; eight-thread counts vary with Rayon's job splitting.

Timing and allocation measurements use separately compiled executables. Atomic
allocation counting is compiled out for timing/RSS runs; otherwise its contention
would bias parallel timing. Raw CSVs and `/usr/bin/time -v` reports are included.
All full-output checksums agree before/after, across repetitions and thread
counts, in both noise modes. Existing training-harness parity fixtures also pass
unchanged.

To reproduce against each revision (fetch Cargo dependencies first):

```sh
python3 docs/benchmarks/model-noise-scratch/measure.py /tmp/timing.csv \
  --side 4096 --target-dir /tmp/seiza-noise-target --resources /tmp/resources.txt
python3 docs/benchmarks/model-noise-scratch/measure.py /tmp/allocation.csv \
  --side 4096 --target-dir /tmp/seiza-noise-target --count-allocations
```

The probe compiles the actual private noise/sky/stats source files by path and
reuses the public native reading types, so it isolates the kernel without adding
a public API just for benchmarking. Compilation and image/pool construction are
excluded from per-call timings. Image construction and every case are included
in the process RSS measurement. Each measured call hashes its whole output.

The new native test runs changing block coverage, NaNs and infinities through
both modes on one, two and eight threads, comparing every output's bits. It
passed on unchanged code. Temporarily omitting the reused values buffer's clear
made it fail at pixel 128; the mutation was removed. No model output-version or
binding changes are needed: this is private shared-core allocation reuse.

Validation against `7cfe303`: 2,363 Rust tests passed (19 ignored), 1,692
Python tests passed (24 skipped, including three requiring optional ONNX Runtime),
and all 79 stacking/stretch/pipeline output-version checks passed unchanged.
Focused model tests passed in debug and release, including the training-harness
parity fixtures. Workspace/Python formatting and Clippy, Rust 1.92
all-targets/all-features/locked checking, and binding parity passed. Native
runtime-dependent tests follow their existing availability guards; this
benchmark and the new regression need no inference runtime.
