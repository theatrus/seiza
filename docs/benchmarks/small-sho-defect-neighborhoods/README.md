# Faster CPU defect detection on small_sho

The complete 30-light small SHO workflow takes **29.923 → 26.961 seconds**
(medians of five runs per version), **9.9% less elapsed time**. Every measured
pair improved. All H/O/S stacks and the combined linear SHO FITS are
**byte-for-byte identical**, including headers and missing-pixel coverage.

## Change and workload

Automatic defect detection examines each pixel's same-plane neighbours. Its
common inner ring previously checked each neighbour's coordinates separately.
For an interior pixel, `gather` now checks the ring's boundary once and computes
its eight indices from horizontal and vertical strides. The sample order,
finite-value filtering, medians, noise estimates, thresholds, and defect repair
are unchanged. Border pixels and outer rings retain the original lookup.

Both binaries are release builds from `7cfe303`, using Rust 1.99.1 and locked
dependencies. The after binary changes only this private lookup. No public API,
binding, output-version, quality setting, or worker count changes.
[Binary identities](identity.json) and [input fingerprints](inputs.json) are
retained. Runtime code for this path is shared by Rust, C, and Python hosts.

Inputs are ten each of H, O, and S, 180-second ASI2600MM lights, 6248 × 4176,
from `/mnt/astro/unsorted/2022-10-27-ASIRedCat/MonkeyJelly`. Their 30 names
match `/mnt/astro/small_sho/pixinsight_wbpp`'s registered dataset. The raw files
were staged locally in `/tmp/seiza-sho-benchmark/inputs`; source files were
not modified. This is the same dataset used in the preceding comparison.

The machine is a 10-vCPU Linux VM. Each workflow uses eight Rayon threads,
two preparation workers, a 1024 MiB pipeline/reintegration budget, and no GPU.
It includes automatic reference selection, light-derived defect detection,
local-background normalization, Lanczos-3 resampling, inverse-noise weights,
quadratic registration, and three-pass reintegration. Each filter uses the
same H_0001 alignment grid. Cheap reports and FITS writing are timed. Linear
SHO composition uses palette SHO with no additional normalization, registration,
stretching, or denoising. No calibration masters are supplied.

A supplied master dark normally disables automatic defect learning; this
benchmark does not establish the same gain for that workflow or for GPU
screening of defects.

## Measurements

One complete warm-up per binary preceded the measurements. Runs alternate
order: after/before, before/after, after/before, before/after, after/before.
Two pairs were added to the initial three to check VM timing variation; all
five pairs are retained. Builds and image-quality analysis did not run during
these measurements. Input caches were warm. FITS hashing occurs outside the
timed intervals.

| Measurement | Before | After |
| --- | ---: | ---: |
| Full workflow, median elapsed | 29.923 s | 26.961 s |
| Full workflow, elapsed range | 29.030–31.041 s | 26.718–28.836 s |
| Full workflow, median user CPU time | 166.920 s | 148.440 s |
| H stack, median elapsed | 9.927 s | 8.737 s |
| O stack, median elapsed | 9.573 s | 8.807 s |
| S stack, median elapsed | 9.448 s | 8.870 s |
| SHO composition, median elapsed | 0.732 s | 0.707 s |
| Per-workflow peak RSS, median | 1.968 GiB | 1.842 GiB |
| Per-workflow peak RSS, range | 1.807–2.016 GiB | 1.776–1.888 GiB |

Full-workflow medians are taken over complete workflows, not sums of stage
medians. User CPU time sums across threads and subprocesses; its median falls
**11.1%**. Peak RSS is the largest subprocess peak within each serial workflow.
Its observed median is lower, but the ranges overlap and this change does not
reduce buffer storage; **no established memory saving is claimed**.

| Pair | Before | After | Elapsed reduction |
| --- | ---: | ---: | ---: |
| 1 | 29.030 s | 26.940 s | 7.20% |
| 2 | 31.041 s | 28.836 s | 7.10% |
| 3 | 29.923 s | 26.718 s | 10.71% |
| 4 | 30.162 s | 26.961 s | 10.61% |
| 5 | 29.068 s | 28.285 s | 2.70% |

The median paired reduction is 7.20%; comparing the two overall medians gives
9.90%. These are measurements of this CPU workload, not a universal stacking
speedup. [Raw runs](results.jsonl), [summary](summary.json), and the `runs/`
commands, logs, and GNU time resources preserve every repetition.

## Output and correctness

All twelve workflows, including warm-ups, admit all 30 lights and reject no
whole frames: 360 processed lights in total. Complete admission records,
reference identities, normalization references, registration diagnostics,
weights, and defect counts match exactly. Defect counts remain H=3718, O=6254,
S=688.

Each of the four output files has one SHA-256 across all twelve workflows.
That verifies all **156,549,888 H/O/S/SHO samples per workflow**, including NaN
bits, as well as the FITS headers. The hashes are recorded in `summary.json` and
every raw result. There is no pixel difference to assess with a noise or PSF
proxy on these outputs.

The new native test compares gathered samples bit-for-bit with an independent
coordinate-based lookup, across borders and interiors, mono/RGB, Bayer spacing,
inner/outer rings, NaNs, infinities, and signed zero. The existing
`a_stack_repairing_learned_defects_keeps_its_output` fixture covers the modified
mode. A planted +1000 neighbour-value mutation made that fixture fail; the
mutation was removed. All **79 output-version cases** pass unchanged before
and after (stacking 74, stretch 3, pipeline 2). No output-version bump is needed.
Baseline, final, mutation, and native-oracle logs are in `validation/`.

## Reproduction

[benchmark.py](benchmark.py) reconstructs the measured commands from the four
`commands/` templates. Its reconstruction was checked against all eight recorded
before/after stage commands. On this machine, the full scientific outputs and
binaries remain under `/tmp/seiza-sho-defect-benchmark`.

To repeat, build the base and changed revisions with
`cargo build --release --locked -p seiza-cli`, save the binaries as
`seiza-before` and `seiza-after` in a new work folder, and stage the fingerprinted
raw files locally. Then run:

```sh
SEIZA_SHO_WORK=/tmp/new-sho-run SEIZA_SHO_INPUTS=/path/to/staged/lights \
  python3 docs/benchmarks/small-sho-defect-neighborhoods/benchmark.py
```

The driver refuses existing per-run record directories. It retains the last
scientific FITS files for each version and hashes every repetition before they
are overwritten. It requires GNU time; its other dependencies are Python's
standard library and the two release binaries.

The full repository checks also passed: workspace and Python formatting and
Clippy, Rust 1.92 locked all-target/all-feature checking, binding parity,
**2,363 Rust tests** (19 ignored), and **1,692 Python tests** (24 skipped).
The C header is unchanged. Commands, exit codes, and full logs are retained
in `validation/`.

The final rebuilt release has identical executable code and production source
but a different build ID and symbol-name metadata after the test-only edit.
[Section comparison](binary-sections.json) records this check. An additional
full workflow with that final binary reproduces all four FITS hashes and the
admission records; [its quality check](final-build-quality.json) is excluded
from the timing summary.
