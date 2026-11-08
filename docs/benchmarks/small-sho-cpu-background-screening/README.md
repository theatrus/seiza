# CPU background screening on small SHO

Two exact shortcuts reduce work on ordinary background pixels:

- Float star detection checks blocks of 32 samples with a reduction before locating the first positive sample with the existing scalar search. It preserves the seed order and flood-fill arithmetic.
- Mono defect detection screens each interior row against the existing eight-neighbour extrema test, then runs the original detailed examination on candidates. It rules out a pixel only when the entire neighbourhood is finite and the existing test would return immediately. Borders and non-finite neighbourhoods take the full path. CFA and RGB defect detection retain their existing path.

Both changes are internal to the native core and apply to its Rust, C and Python callers. The statistical thresholds and output versions are unchanged.

## Controlled measurement

This measures the additional gain **after both #454 and #455**. Both binaries contain those changes, held constant on main `7cfe303`; only the after binary contains this PR. This PR itself is independently based on main and includes neither earlier PR. The final main-based release build is checked separately. Do not multiply percentages from separate benchmark rounds.

The workload is the thirty original MonkeyJelly ASI2600MM frames corresponding to `small_sho`: ten each H/O/S, 6248 × 4176 mono, 16-bit, 180-second exposures, 1,565,994,240 bytes total. Each workflow stacks all three channels and makes a SHO composite, using local-background normalization, Lanczos-3, inverse-noise weighting, quadratic registration, automatic defect repair and final reintegration. Calibration masters are not supplied. Input inventory and SHA-256 checksums are in `inputs.json`.

Release builds with locked dependencies, Rust 1.99.1, Linux KVM, ten exposed vCPUs and 31 GiB RAM; eight Rayon threads, two preparation workers, 1 GiB pipeline budget, CPU only. After one complete warmup per binary, five pairs alternate after/before and before/after. No builds or other quality runs occurred during measurement. Hashing and report analysis are outside timed intervals. These are warm-cache results on this machine.

| Complete-workflow metric | Before median (range) | After median (range) |
| --- | --- | --- |
| Elapsed seconds | 24.847 (24.546–25.161) | 22.894 (22.730–23.420) |
| User CPU seconds | 134.53 (134.08–137.76) | 121.02 (120.55–122.56) |
| Kernel CPU seconds | 14.64 (14.60–14.88) | 14.52 (14.47–14.79) |
| Minor page faults | 7,162,587 (7,129,525–7,178,856) | 7,166,643 (7,150,666–7,176,696) |
| Maximum stage RSS, KiB | 1,820,548 (1,783,816–1,849,040) | 1,793,016 (1,759,204–1,803,440) |

Median elapsed time fell **7.86%**, with every pair faster. Median user CPU fell **10.04%**. Peak RSS ranges overlap; this is not evidence of an established peak-memory reduction. Stage peaks are individual process maxima, not summed concurrent memory. The defect screen uses task-local scratch of one byte per image column.

## Output checks

Every one of the twelve measured/warmup workflows produced identical complete H, O, S and SHO FITS SHA-256 hashes, including headers and floating-point representations: 156,549,888 image samples per workflow. All thirty lights were accepted, with identical complete frame diagnostics, normalization, registration, sample decisions and defect reports (3,718 H, 6,254 O, 688 S defect pixels). See `full-results.jsonl`.

Separate runs request a clipping map, then an expanding canvas saved for later continuation. The requested maps, H images and entire saved contexts were byte-identical. Successful commands, logs and complete file hashes are recorded under `requested/`. These checks are excluded from timing. The final build directly on main, without #454/#455, is also checked against all four image hashes and admission records under `final-main-quality/`.

The new float-search test compares every start position across block boundaries, signed zero, NaNs, infinities and the smallest positive subnormal. The existing dense-versus-sparse detector test compares complete stars to the bit. The new mono-screen test compares flags with the original finite-neighbour gather and extrema check over ragged widths, borders, non-finite samples, extreme values and threshold boundaries. Existing defect fixtures verify learned hot/dead pixels, stars and repair.

All 79 stacking/stretch/pipeline output-version cases pass with their existing summaries and versions unchanged. `a_mono_stack_with_default_options_keeps_its_output` covers float CPU detection; `a_stack_repairing_learned_defects_keeps_its_output` covers the mono CPU defect path. Their 192 × 160 fixtures are below the GPU cutoff. Temporarily disabling float positive-pixel detection made the former fail; dropping mono defect candidates made the latter fail. Both mutations were undone. Baselines, failures and final logs are in `output-version/`.

Full validation passed: formatting, workspace/Python Clippy with warnings denied, Rust 1.92 compatibility, **2,364 Rust tests** (19 ignored), binding parity, maturin build and **1,803 Python tests** (4 skipped). Astropy is installed so FITS-dependent tests run. Commands, output and exit codes are retained in `validation/`.

Validation environment: the sandbox initially denied a localhost mock-server bind. With localhost access, the parallel external-tool fixture tests intermittently failed to execute freshly written scripts (`Text file busy`, reproduced in focused repeats). The final Rust suite uses `RUST_TEST_THREADS=1`; every test remains enabled, including tests that exercise their own thread pools and pipeline workers. Original failures and diagnostics are retained in `validation/`. The temporary diagnostic assertion was restored; the external-tool source is unchanged.

## Exploratory measurements

The float-search change alone was measured in five alternating pairs: median 24.909 → 24.466 seconds (1.78%), with every pair faster and identical outputs. `float-search-only/` retains those results and binary hashes. The headline table measures both changes together directly against the common baseline; it does not add their percentages.

A mono-specific resampling specialization was discarded after five pairs: median 24.743 → 24.832 seconds and essentially unchanged user CPU. Its summary is retained in `rejected-resampling-summary.json`; none of that candidate's code is included here.

## Reproduction and raw evidence

`identity.json` records compiler, CPU, source and binary identities. `summary.json` retains all five measured values per metric. `records.jsonl` contains each actual command, GNU time resource record and log, including warmups (iteration 0). The full FITS and context files remain outside the repository.

Build the control from main `7cfe303` plus commits `c5a3bd1c3d841b401f60f67bae5a23a5b9030ee3` (#454) and `442b3a40a2a1e91f4e288bce5ae42057184bb25d` (#455), and the after build from the same source plus this PR. Use `cargo build --release --locked -p seiza-cli` and put the two binaries at `seiza-before` and `seiza-after` under a fresh work directory. Stage the raw originals with the manifest's names, then run:

```sh
SEIZA_SHO_WORK=/tmp/seiza-sho-screen-reproduce \
SEIZA_SHO_INPUTS=/path/to/raw/inputs \
python3 docs/benchmarks/small-sho-cpu-background-screening/benchmark.py
```

The packaged driver was checked to reconstruct all eight measured command templates exactly. It retains warmups, alternates order, hashes outside timed intervals and records admission. Preserve machine/thread limits, dependency lock and workload when comparing results.
