# Skip unused saturation tracking in batch stacks

The CLI previously detected, registered and accumulated raw saturation masks even when it never exported them. `StackOptions::track_clipping` now lets a caller skip that work. It defaults to true in Rust, C (`trackClipping`) and Python (`track_clipping`). The CLI disables it only when neither `--clipped` nor `--save-canvas` is requested. Saved contexts retain clipping history for later colour processing.

## Controlled small SHO measurement

This measures the **additional** improvement after PR #454. Both measured binaries contain its defect-neighbourhood change (commit `c5a3bd1c3d841b401f60f67bae5a23a5b9030ee3`), held constant on main `7cfe303`; only the after binary contains this PR. This PR itself is independently based on main and does not include #454. Do not combine independent PR percentages to infer an end-to-end result.

Thirty original MonkeyJelly ASI2600MM frames, ten each H/O/S, 6248 × 4176, 180 seconds, 1,565,994,240 input bytes. Their names match the registered `small_sho` dataset. Each workflow stacks all three channels and creates the SHO composite, with local-background normalization, Lanczos-3, inverse-noise weighting, quadratic registration and automatic defect repair. No calibration masters are supplied. See `inputs.json` for inventory and checksums.

Release builds, locked dependencies, Rust 1.99.1, Linux KVM, ten exposed vCPUs, 31 GiB RAM; eight Rayon threads, two pipeline workers, 1 GiB pipeline budget, CPU only. One complete warmup per binary, followed by five alternating paired workflows. No builds or quality analysis ran concurrently with timed workflows. Hashing and report inspection are outside timed intervals. These are warm-cache results on this machine, not a universal speedup guarantee.

| Metric | Before median (range) | After median (range) |
| --- | --- | --- |
| Complete elapsed seconds | 26.689 (26.574–27.160) | 23.087 (22.636–23.302) |
| User CPU seconds | 147.44 (146.48–148.14) | 126.22 (124.92–127.42) |
| Kernel CPU seconds | 15.90 (15.83–16.17) | 14.33 (14.06–14.40) |
| Minor page faults | 8,699,374 (8,669,802–8,710,189) | 7,152,741 (7,147,079–7,175,281) |
| Maximum stage RSS, KiB | 1,900,972 (1,880,200–1,963,400) | 1,847,724 (1,829,240–1,918,508) |

Median elapsed time fell **13.50%**; every paired workflow was faster. Median user CPU fell 14.39%. Peak RSS ranges overlap: these measurements do not establish a peak-memory reduction. Stage peaks are process maxima, not summed concurrent memory.

## Output and compatibility checks

All twelve measured/warmup workflows produced identical SHA-256 hashes for every complete H, O, S and SHO FITS file, including headers and floating-point representations: 156,549,888 image samples per workflow. All thirty lights were accepted each time, with identical complete frame diagnostics, registration, normalization, sample rejection and defect reports (H: 3,718; O: 6,254; S: 688 repaired defect pixels). `full-results.jsonl` records hashes and admission details.

Separate runs with `--clipped` produced byte-identical clipping maps and H images. Separate expanding-canvas runs with **only** `--save-canvas` produced byte-identical H images and entire saved contexts. Reopening the after context confirmed a nonempty clipping map with maximum 1.0. These checks are excluded from timing; successful commands/logs and hashes are under `requested/`.

A final release build of this PR directly on main, without #454, also reproduced all four FITS hashes and frame diagnostics. Its independent quality record is under `final-main-quality/`; its timing is not part of the controlled performance comparison.

Native tests cover the constructor, sequential and pipelined pushes, image/variance bit equality, coverage equality, settings serialization, fingerprints, context reopen and continued integration. C and Python tests exercise their own native bindings. Default true is omitted from serialized options, preserving old settings/fingerprints/context bytes; false is explicitly serialized.

All 79 stacking/stretch/pipeline output-version tests pass with existing summaries and versions unchanged. The large saturated-stack case now checks enabled/disabled tracking with Bilinear/Lanczos-3 and CPU/default GPU selection. Additional image summaries were recorded on unchanged code first. Disabling clipping capture with a planted mutation made the original clipping-map case fail; the mutation was undone. Logs are under `output-version/`. No GPU device was available, so default selection exercised its CPU fallback.

Full validation passed: formatting, workspace and Python Clippy with warnings denied, Rust 1.92 compatibility, **2,364 Rust tests** (19 ignored), binding parity, Python build/stub checks and **1,805 Python tests** (4 skipped). Astropy was installed in the temporary Python environment so FITS-dependent tests actually ran. Commands, outputs and exit codes are under `validation/`.

## Reproduction and artifacts

`identity.json` records measured binary hashes, compiler and CPU identity. `summary.json` contains the five measured values for each metric. `records/` contains every command, resource measurement and log, including warmups (iteration 0).

Build the before revision with #454, then the same source plus this change using `cargo build --release --locked -p seiza-cli`; save the binaries as `seiza-before` and `seiza-after` under a fresh work directory. Stage the originals in a separate input directory and run:

```sh
SEIZA_SHO_WORK=/tmp/seiza-sho-clipping-reproduce \
SEIZA_SHO_INPUTS=/path/to/raw/inputs \
python3 docs/benchmarks/small-sho-clipping-tracking/benchmark.py
```

The packaged driver reconstructs all eight measured before/after command templates exactly (checked while packaging). Outputs stay outside the repository. Large input images, output FITS and saved contexts are not committed. The driver retains warmups, alternates order, hashes outside timed intervals and records frame admission. Preserve the same source versions, dependency lock, machine/thread limits and workload when comparing results.
