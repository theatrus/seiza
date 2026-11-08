# Streaming FITS headers

A header reader previously accumulated raw 2,880-byte blocks until `END`,
then parsed all of them. The reader now parses each block immediately,
retaining decoded cards and a single raw block. Long header strings and
commentary remain supported across blocks; there is no new size limit.
Memory still scales with actual decoded metadata, which the caller receives.

Measured with rustc 1.99.1 on Linux x86_64, using an unoptimized external
binary that calls `seiza_fits::read_header`. The valid fixture contains
32,768 blocks with three valued cards and otherwise blank padding, followed
by an END block: 94,374,720 bytes. The unfinished fixture omits the END block.
These deliberately large headers isolate raw-buffer retention; they are not
representative of typical acquisition headers.

| Fixture | Before peak RSS | After peak RSS | Result |
| --- | ---: | ---: | --- |
| Valid | 94,208 KiB | 2,176 KiB | Same three decoded cards |
| Unfinished | 94,336 KiB | 2,176 KiB | Same header-runs-past-EOF error |

Raw `/usr/bin/time -v` reports are included. The script also checks the decoded
cards and expected EOF error. Single-run timings in those
reports are diagnostic, not throughput benchmarks. To repeat on each revision:

```sh
python3 docs/benchmarks/fits-header-streaming/measure.py /tmp/header-results
```

The native tests cover primary and extension headers, continuation strings
across blocks, commentary, pixel offsets, and 4,096 padding blocks without
retaining a fixture of that size in memory. The stacking output fixture
`a_stack_with_multiblock_fits_headers_keeps_its_output` was first recorded
against unchanged code, reusing the existing COSMETIC summary. A temporary
refusal to read the second block made it fail; the mutation was removed.

Validation was repeated against main at `573da05`, including the newly merged
GZIP_2 and learned-model code. The native FITS and image-I/O tests pass in debug
and release builds. Workspace/Python formatting and Clippy, Rust 1.92
all-targets/all-features checking, and binding parity pass. The full workspace suite passes
2,359 tests (19 ignored), and Python passes 1,692 (24 skipped, including three
that require the optional ONNX Runtime library). All 78 stacking/stretch/pipeline
output-version checks pass unchanged. No public API, binding, or
output-version changes are needed: both bindings use the shared FITS reader.
