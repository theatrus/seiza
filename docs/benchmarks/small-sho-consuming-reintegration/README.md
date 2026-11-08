# Release the live accumulator before final reintegration

`LiveStacker::finish_reintegration` consumes a completed live stack and releases its online accumulator before replay allocates its buffers. It calls the existing reintegration implementation with the same options, progress callback and per-frame diagnostics. The CLI uses it for ordinary `stack --reintegrate`; Rust, C and Python hosts can opt into it when they have finished adding frames. Borrowed reintegration remains available for stacks that will continue. Drizzled reintegration retains its existing allocation behavior.

On the CPU, the released accumulator holds 16 bytes per live image sample, or 20 with inverse-noise weighting, plus any clipping fractions. For this dataset's 26,091,648 weighted mono samples, the accumulator alone occupies 497.7 MiB. Reference images, calibration, provenance and cached frames remain available to replay. Ownership ensures cleanup on success, failure and cancellation. The C call clears the accepted handle; Python marks the stack finished after argument validation. Snapshots retain diagnostics, reference headers and input-path protections.

## Measurement

Both controlled binaries contain #454, #455 and #456, held constant on main `7cfe303`. Only the after binary contains this change. This PR itself is independent of those PRs and based on main; a separate final release build on main is checked for output equality. Source and binary identities are in `identity.json`.

The workload uses thirty original MonkeyJelly ASI2600MM frames corresponding to `small_sho`: ten H/O/S frames each, 6248 × 4176 mono, 16-bit, 180 seconds, 1,565,994,240 input bytes. Each workflow stacks three channels and makes the SHO composite. Options include local-background normalization, Lanczos-3, inverse-noise weighting, quadratic registration, automatic defect repair and final reintegration. No calibration masters are supplied. See `inputs.json` for inventory and SHA-256 hashes.

Release builds with locked dependencies, Rust 1.99.1, Linux KVM, ten exposed vCPUs and 31 GiB RAM. Eight Rayon threads, two preparation workers, 1 GiB pipeline budget, CPU only. One full warmup per binary precedes five alternating pairs. No builds or other quality runs occurred during these timed workflows. Hashing and report analysis are outside timed intervals. Results use a warm filesystem cache on this machine.

| Complete workflow | Before median (range) | After median (range) |
| --- | --- | --- |
| Elapsed seconds | 22.545 (21.883–23.609) | 22.617 (21.620–23.637) |
| User CPU seconds | 114.02 (113.52–115.74) | 114.96 (113.09–115.14) |
| Kernel CPU seconds | 14.94 (14.58–15.10) | 15.12 (14.53–15.41) |
| Maximum stage RSS, KiB | 1,900,968 (1,861,020–1,922,484) | 1,797,144 (1,744,012–1,891,788) |

Runtime is essentially unchanged: median elapsed time is 0.32% higher, with mixed pair results. The observed overall RSS median is lower, but its ranges overlap; these runs do not establish a reliable percentage reduction in the whole-workflow peak. Stage RSS is the maximum of individual process peaks, not summed concurrent memory.

Separate H-channel profiles sample `/proc/PID/status` every 10 ms. The first `reintegrate ...` progress line starts the replay phase; the first `wrote ...` line ends it. These four runs are excluded from all timing summaries.

| Profile order | Binary | Preparation peak, KiB | Reintegration peak, KiB |
| --- | --- | --- | --- |
| 0 | Before | 1,638,932 | 1,803,132 |
| 1 | After | 1,832,600 | 1,363,204 |
| 2 | After | 1,783,420 | 1,314,900 |
| 3 | Before | 1,670,216 | 1,801,900 |

Reintegration peaks fall from about 1.72 GiB to 1.25–1.30 GiB: 430–476 MiB less in the adjacent before/after comparisons, or a 25.7% reduction between the two medians. Preparation peaks vary and now dominate the after runs. This is evidence for lower memory during final replay on this H-channel workload, rather than a universal reduction in process maximum RSS. Sampling and scheduling can miss short peaks; the phase boundaries follow progress output rather than exact function entry and exit. Raw timestamped samples, actual commands and logs are under `profile/`.

## Output and lifecycle checks

All twelve complete workflows, including warmups, produce byte-identical H, O, S and SHO FITS files, including headers and floating-point representations: 156,549,888 image samples per workflow. Their complete admission records match: all thirty frames accepted, with identical registration, normalization, defect and sample diagnostics. `quality-summary.json` and `full-results.jsonl` retain the evidence.

Separate before/after runs request a clipping map and an expanding canvas saved for continuation. The H images, requested clipping maps and entire saved contexts are byte-identical (`requested-quality.json`, `requested-records.json`). These runs are excluded from timing. A release build directly on main, without #454/#455/#456, is checked separately under `final-main-quality/`.

Native tests compare borrowed and consuming replay to the bit across equal/inverse-noise weighting, no/local-background normalization and retained/uncached frames, including image, variance, coverage, rejection counts, weights, diagnostics and progress events. They check scratch cleanup on success and cancellation. C tests check retained/uncached equality, diagnostics and null handles after success/failure. Python tests check all eight mode combinations, reports, input protections, FITS headers, failure lifecycle and validation before consumption.

The original 79 stacking/stretch/pipeline output-version cases passed before the implementation. `a_reintegration_and_its_drizzle_keep_their_output` now also checks the consuming call against its existing summary. A new weighted local-background replay case was recorded using the original main implementation before wiring it to the consuming call. Adding 1 to replay's output samples made both relevant cases fail; the mutations were undone. All 80 final output-version cases pass. No existing summary, tolerance or output version changed. Baseline, mutation and final logs are under `output-version/`.

Formatting, workspace/Python Clippy with warnings denied, Rust 1.92 compatibility, **2,366 Rust tests** (19 ignored), binding parity, maturin build and **1,815 Python tests** (4 skipped) pass. Validation results and commands are retained under `validation/`. The full Rust suite uses `RUST_TEST_THREADS=1` because this environment previously exhibited intermittent `Text file busy` failures when executing concurrently created external-tool fixture scripts (documented in #456). All tests remain enabled, including internal pipeline concurrency tests. Localhost access is enabled for mock-server tests. Astropy is installed for Python FITS tests.

## Reproduction

Build the control from main `7cfe303` plus commits `c5a3bd1c3d841b401f60f67bae5a23a5b9030ee3` (#454), `442b3a40a2a1e91f4e288bce5ae42057184bb25d` (#455) and `facb5f76b7e1642b99d4fa054bfa0345b9625f89` (#456). Build the after binary from the same source plus this PR. Use `cargo build --release --locked -p seiza-cli`; place the binaries as `seiza-before` and `seiza-after` in a fresh work directory. Stage the input manifest's raw originals with the same filenames, then run:

```sh
SEIZA_SHO_WORK=/tmp/seiza-sho-reintegration-reproduce \
SEIZA_SHO_INPUTS=/path/to/raw/inputs \
python3 docs/benchmarks/small-sho-consuming-reintegration/benchmark.py
```

Run `profile.py` with the same environment to repeat the separate four H-channel profiles. The packaged drivers reconstruct the recorded command templates; the benchmark driver was checked against all eight before/after channel templates. `records.jsonl` holds commands, GNU time resources and logs for all twelve workflows. `summary.json` retains each measured value. Large image files are excluded from Git.
