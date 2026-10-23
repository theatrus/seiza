# Windows NVIDIA stacking validation

PR #344, commit `085eb5fd61ab03e82b829e7e51fae592fbf415bb`, tested on
2026-10-22. No GPU implementation changes were needed.

## Machine and data

- Windows 11 Pro, build 26200; Rust 1.99.1, MSVC x64, release build.
- Threadripper PRO 9975WX, 32 cores / 64 threads, 383 GiB RAM.
- GeForce RTX 5090, 32 GiB VRAM, NVIDIA driver 610.88, WDDM.
- 200 Pleiades green-filter lights, ZWO ASI6200MM Pro, 9576×6388,
  16-bit mono, 60-second exposures, gain 100. The first 200 files in
  acquisition order from `O:/_ByTelescope/Ultracat/_Source/2026/Pleiades`,
  nights 2026-09-22, 2026-09-23 and 2026-09-25, excluding `Rejects`.
- Copied 24,470,208,000 bytes to local NVMe before timing. Output and
  reintegration scratch also used local NVMe. No calibration masters.

Every run used the same ordered input list, `--reference first`,
`--reintegrate`, `--report`, and otherwise the CLI defaults, including
Lanczos-3, quadratic registration, local-background normalization,
inverse-noise weighting, automatic defect learning and the 4096 MiB
pipeline memory budget. Timings include process startup, reading, defect
learning, stacking, reintegration, writing and provenance hashing.

## Results

| Mode | Wall time | Process CPU time |
|---|---:|---:|
| CPU, `SEIZA_GPU=off` | 55.1–56.7 s | 2445–2456 CPU-s |
| Vulkan, `WGPU_BACKEND=vulkan` | 42.4–44.2 s | 588–599 CPU-s |
| Direct3D 12 with DXC | 42.0 s | 604 CPU-s |

Two CPU runs, three Vulkan runs and one DXC run. The first Vulkan run
and the DXC run enabled GPU tracing; the repeats did not. No compilation
ran during these measurements. An earlier FXC trace
overlapped helper compilation, so its end-to-end timing is excluded.

Vulkan is the backend selected with no environment overrides on this
machine. It reduces end-to-end time by about 20–25% and process CPU time
by about 76%. The system was also driving the desktop.

All modes integrated the same 192 frames and rejected the same eight:
six for normalization gains outside the configured range, two for
registration. Traces show 200 GPU detection calls, 198 resampling calls,
16 defect screens, 191 admission calls, a moments readback, and 355 GPU
reintegration bands. This exercises the discrete GPU staging path.

Against the CPU stack, the GPU stacks have RMS differences of
0.00978–0.00980 times the CPU stack's noise (3.565 ADU). Finite-pixel masks
agree. Of 61,171,488 pixels, 25–27 differ by more than one noise sigma and
926–947 by more than half a sigma. The largest difference is 242 ADU,
67.9 noise sigma, at pixel (941, 752), on a steep bright feature (the
CPU's 3×3 neighborhood spans 4658–36153 ADU). The small RMS does not mean
there are no larger local differences. This real set is mono; VNG was
checked with the synthetic RGB/Bayer tests below.

The two CPU stacks are bit-identical to each other, as are all three
Vulkan stacks.

## Windows setup and reproduction

Install rustup with the MSVC toolchain and Visual Studio C++ Build Tools,
including the Windows SDK. Cargo uses the repository's pinned 1.99.1
toolchain. CUDA is unnecessary. The installed NVIDIA driver supplies
Vulkan; no extra runtime DLL was required for the default backend.

```powershell
cargo build --release -p seiza-cli -p seiza-cabi --locked
$lights = Get-ChildItem '<local data directory>' -Filter '*.fits' |
    Sort-Object Name | Select-Object -ExpandProperty FullName
$env:WGPU_BACKEND = 'vulkan'
Remove-Item Env:SEIZA_GPU -ErrorAction SilentlyContinue
& .\target\release\seiza.exe stack @lights --reference first --reintegrate `
    --output '<GPU output.fits>' --scratch-directory '<local scratch>' `
    --report '<GPU report.json>'
# Current PR head uses --no-gpu, replacing SEIZA_GPU=off from the first run.
# Repeat with --no-gpu and separate CPU output, scratch and report paths.
```

Direct3D 12 also works. Without `dxcompiler.dll` discoverable, wgpu uses
FXC and compiling all pipelines took 14.6 seconds in the real stack
process (20.1 seconds in the first regression process). Using the Windows
SDK's signed DXC 1.9.2602.17, with its x64 directory on the process PATH
and `WGPU_DX12_COMPILER=dxc`, reduced initialization to 1.8 seconds.
The default Vulkan path avoids this additional DLL requirement.

## Verification

- Both `output_version` suites pass unchanged on CPU and Direct3D 12:
  33 stacking cases and three stretch cases. The stacking suite also
  passes with DXC. GPU traces confirm the large mono and Bayer cases
  actually execute kernels.
- All ten GPU comparison tests pass on Vulkan (the default), Direct3D 12
  with FXC, and Direct3D 12 with DXC. They cover resampling, joint mask
  resampling, bands, defects, detection, VNG, normalization, residency,
  accumulation and WGSL validation. Adapter logs explicitly identify the
  RTX 5090 and backend; none skipped for a missing GPU.
- The built Windows C ABI DLL opened a real 61 MP reference, accepted two
  more FITS lights, read back the GPU moments and reported three accepted
  frames. The generated header is unchanged.
- Binding parity passes with Python's UTF-8 mode enabled. The CLI passes
  the repository's static CRT check.

The local worktree's ignored `target/gpu-validation/` directory holds the
source manifest, hardware details, benchmark script, all output images,
reports, comparison utility, C ABI smoke script, traces and test logs.
Linux NVIDIA validation remains outstanding.

## Parallelism and CPU-budget follow-up

Refreshed to PR head `0f1a54777498648f98c5cdce9a7eac867fb597c5`, which
replaces the `SEIZA_GPU*` environment controls with request options and
CLI flags. The following runs use that head, the same 200 local lights,
the same full-stack options, and Vulkan. GPU utilization and total VRAM
use were sampled every 250 ms with `nvidia-smi`; VRAM includes the desktop.

| One complete 200-frame stack | CPU threads | Prep workers | GPU calls | Wall | CPU time | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| CPU only | 16 | 6 | — | 103.0 s | 1571 CPU-s | — |
| GPU, defaults | 64 | 6 | 3 | 44.0 s | 603 CPU-s | 9.0 GiB |
| More work in flight | 64 | 12 | 6 | 44.0 s | 610 CPU-s | 14.0 GiB |
| More work in flight | 64 | 24 | 12 | 44.7 s | 617 CPU-s | 21.7 GiB |
| GPU, smaller CPU pool | 16 | 6 | 3 | 42.0 s | 270 CPU-s | 9.0 GiB |
| GPU, smaller pool and more workers | 16 | 12 | 6 | 40.9 s | 293 CPU-s | 13.9 GiB |

The matched 16-thread comparison is about 2.5× faster on the GPU. Raising
frame concurrency mainly raises memory consumption; it does not saturate
this GPU. Mean GPU utilization stays about 20–25%, even with 22 GiB of
VRAM occupied. The default run spends about 31 seconds in the live pass
and nine in final integration. Its band traces spend considerably longer
in host-side upload/staging than in the submit-and-wait span. These are
CPU wall timings, not GPU timestamp measurements of transfers or kernels.
Occupying more VRAM does not by itself remove the host processing,
transfer, readback and scratch-file costs.

All full-stack concurrency settings admitted the same 192 frames, rejected
the same eight and produced bit-identical GPU images, including when the
CPU thread budget changed.

Four independent 50-frame jobs, each using 16 Rayon threads, six preparation
workers and three GPU calls, completed together in **33.8 seconds**, with
**355.6 CPU-s** summed over the four processes and **25.9 GiB** peak VRAM.
The four CPU-only counterparts, with the same aggregate 64-thread budget,
took **42.5 seconds** and **2230 CPU-s**. GPU utilization averaged 25.1%
and peaked at 56%; larger occupancy did not imply sustained full compute.

These subset jobs returned **191 admitted / nine rejected** frames in
total, versus 192/eight for the complete stack. They have separate
references, normalization, defect learning, admission histories and
rejection statistics. Their images are useful independent stacks, but
averaging them would not reproduce the complete stack. Parallel targets
or filters can use this process-level concurrency directly. Preserving a
single stack's behavior requires retaining its global admission and
rejection decisions; overlapping spatial-band transfers and integration
is a more suitable direction than merging independently rejected means.

Two 100-frame GPU jobs, with **eight Rayon threads each** (16 total),
completed in **34.6 seconds**, used **229.9 CPU-s** in total, and peaked
at **15.7 GiB VRAM**. They also admitted 191 frames and rejected nine.
This is nearly the four-job throughput with a quarter of its CPU thread
budget and considerably less memory. For independent stacks it is the
more economical tested configuration.

A trial parallelized the copies from discrete GPU readback buffers.
Although the first 16-thread run fell from 42.0 to 39.9 seconds, the
controlled repeat was **40.94 seconds / 244 CPU-s** for the existing path
and **40.65 seconds / 269 CPU-s** for the trial. Output pixels were
bit-identical. The small repeat gain did not justify the extra CPU work;
the trial was reverted. No readback-copy change from that experiment remains.

Keeping all 200 resampled mono images resident would require about 49 GB
for their pixels alone, before masks, accumulators and scratch buffers.
This card cannot hold that whole working set. Better residency needs a
bounded, tiled design rather than simply keeping every frame.

For one complete stack, a 16-thread pool and the default six preparation
workers / three GPU calls are sufficient in these measurements. The
12-worker / six-call run was not consistently faster than the repeated
default. The controls can be reproduced explicitly as follows:

```powershell
$env:RAYON_NUM_THREADS = '16'
& .\target\release\seiza.exe stack @lights --reference first --reintegrate `
    --workers 6 --gpu-concurrency 3 --pipeline-memory-mib 4096 `
    --output '<GPU output.fits>' --scratch-directory '<local scratch>' `
    --report '<GPU report.json>'
# Add --no-gpu for the matched CPU-only comparison.
# Add --gpu-trace to inspect individual GPU calls.
```

The local parallel-subset benchmark is:

```powershell
.\target\gpu-tuning\subsets.ps1 -Name another-two-job-run -Parts 2 -TotalThreads 16
```

It writes separate subset stacks; it does not combine them.

The ignored `target/gpu-tuning/` directory contains the refreshed scripts,
exact arguments, phase timings, telemetry, per-subset results and output
comparisons. Both output-version suites pass unchanged on the refreshed
head. A planted corruption of discrete GPU readback made both large mono
and Bayer output cases fail and was removed before further measurement.

## Buffered bands and sparse clipping

The local implementation based on `0f1a547` now separates band submission
from collection, with two reusable buffer sets and one mapped upload per
set on discrete GPUs. It submits the next band before collecting the
previous band's results. Discrete GPUs may use input bands up to 512 MiB,
within half the configured band memory and both device binding limits;
CPU integration and mapped shared-memory GPUs keep the 128 MiB cap.
Every band still sees every accepted frame in its original order. On this
data the larger bands reduce 355 submissions to 88.

The larger gain comes from removing full clipping-mask transfers. A
discrete GPU receives a bounded list of nonzero raw-mask samples and
expands it there. The resampling shader returns only finite positive mask
contributions. The CPU adds each pixel's contribution in admission order,
with exactly the dense path's arithmetic. Dense inputs use the existing
upload; an output list that overflows is resampled as a dense mask.
Negative zeros, fractional masks and non-finite input samples are retained
in the sparse upload, and no overflowing contribution is silently lost.

Three baseline and three modified runs used the same 200 local frames,
16 Rayon threads, six default preparation workers, three GPU calls,
Vulkan, reintegration and tracing. No build or other benchmark ran during
these timings.

| Median of three runs | PR baseline | Buffered bands + sparse clipping |
|---|---:|---:|
| Whole stack | 41.00 s | 33.26 s |
| Reference to final estimation | 29.79 s | 22.22 s |
| Final band integration | 7.92 s | 7.25 s |
| Process CPU time | 248.7 CPU-s | 234.1 CPU-s |
| Peak host working set | 8.09 GiB | 6.13 GiB |
| Peak total VRAM, including desktop | 9.09 GiB | 8.12 GiB |
| Mean sampled GPU utilization | 23.6% | 19.2% |

Baseline wall times were 43.19, 40.81 and 41.00 seconds; modified runs
were 33.26, 33.27 and 32.45 seconds: **18.9% less median elapsed time**,
or 23.3% more throughput. All three modified images are bit-identical to
the baseline across all **61,171,488 pixels**, and all runs admit the same
192 frames and reject the same eight. The sample rejection total is also
unchanged at 60,666,768.

Each modified run used 197 sparse masks, without a dense fallback. Those
masks uploaded **4,227,168 bytes** and read back **413,139,732 bytes** in
total. The previous two full-float transfers needed 96,406,265,088 bytes
for those same masks. A typical frame uploads about 23 KB and reads back
2 MiB, rather than 244,685,952 bytes in each direction. The bounded
readback transfers its capacity, not only its roughly 5,000 contributions.

GPU busy percentage falls even while throughput improves: both host
transfers and GPU mask writes have been removed. It is not evidence of
insufficient input data. The image itself stays resident through GPU
normalization and admission, but its full CPU readback still serves the
normalization fit and the reintegration scratch cache. Removing that
remaining readback requires compact fit samples and a different cache
strategy; this change does not claim to eliminate it.

Band-only pipelining gave modest gains; the live preparation phase accounts
for most of the measured improvement. Parallel packing of the mapped
uploads was tested and removed: it did not improve the integration phase.

The output-version coverage now includes multiple GPU bands, a shorter
tail, and a full-size raw-clipping map. Planting a change in the small-band
GPU mean made the first case fail; zeroing the GPU-resampled clipping mask
made the clipping case fail. Both mutations were undone. The original
summaries and output versions remain unchanged; the new clipping summary
was recorded on the unmodified clipping implementation before the sparse
path was introduced. Additional comparisons cover every compact mask
entry, empty masks, dense overflow, fractional and non-finite masks,
cropped regions, buffer reuse, cancellation, source errors and CPU retry
after partial results or while draining the final band.

The ignored `target/gpu-tuning/` directory holds the exact run arguments,
logs, telemetry, `resident-clip-summary.jsonl`, bit comparisons and the
`band-results.py` aggregation script. No public API was added: both hosts
receive the change through the shared core, and binding parity passes.

CPU mask planning and coordinate-grid generation now precede admission to
the GPU-call limiter, so Rayon work cannot hold a permit while executing
another frame's task that needs one. The final ordering check took
**33.28 seconds / 229.6 CPU-s**, with the same 192/eight frame decisions.
The one-thread-pool comparison also completes with four frame tasks and
one GPU-call permit.

With the transfer bottleneck reduced, more preparation concurrency is
useful on this machine. Twelve workers, six GPU calls and a 16 GiB pipeline
memory budget, still using 16 Rayon threads, took **31.06 and 31.53 seconds**
and 253.3/256.8 CPU-s. Both images remain bit-identical to the baseline,
with the same frame decisions and sample rejection count. The first run
peaked at 12.19 GiB total VRAM and 11.03 GiB host working set. This is the
faster tested configuration, about 24% less elapsed time than the original
41.00-second median; defaults remain a lower-memory option at 33.3 seconds.

```powershell
$env:RAYON_NUM_THREADS = '16'
& .\target\release\seiza.exe stack @lights --reference first --reintegrate `
    --workers 12 --gpu-concurrency 6 --pipeline-memory-mib 16384 `
    --output '<GPU output.fits>' --scratch-directory '<local scratch>' `
    --report '<GPU report.json>'
```

Validation on Windows includes the stacking suite (468 unit tests, one
integration test, 34 output-version cases and its doc test, with five
pre-existing ignored unit tests), the stretch suite's three output cases,
and 14 GPU comparisons each on Vulkan and DX12 with SDK DXC. Both output
versions stay unchanged. Clippy for all stacking targets and features,
the build of all stacking targets without the GPU feature, workspace and
Python formatting, binding parity and `git diff --check` pass. The release
CLI and C ABI DLL build; the DLL's native smoke test accepts three full-size
FITS frames. Building it leaves the generated header unchanged.
