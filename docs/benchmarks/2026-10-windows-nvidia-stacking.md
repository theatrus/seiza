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

## Profiling frame preparation and reusing uploads

Further profiling starts from `a245500`, after the buffered-band and
sparse-clipping changes. With tracing enabled, the pipeline now reports
read/decode time, raw clipping and calibration, and the accepted frame's
registration, resampling, coverage/photometry, normalization fit, cache
write, normalization application and weighting/finish. Spans print with
three decimal places in milliseconds. These remain overlapping
host elapsed times, including queue waits and Rayon scheduling; they are
not device timestamps or additive CPU times.

In the first instrumented 12-worker/six-call run, the median accepted
frame spends 279 ms registering (including detection), 237 ms resampling,
167 ms writing its cache and 70 ms weighting/finishing. In comparison,
coverage/photometry and normalization fitting take 4.7 and 8.9 ms. This
points toward transfers and cache traffic before a rewrite of the fit.

Discrete-GPU frame uploads now use reusable mapped staging buffers from
the existing 1 GiB pool, submitting their copies immediately. The previous
path used `queue.write_buffer` for each full frame. Buffer mappings wait
for any earlier use before a staging buffer is filled again. Shared-memory
adapters retain their direct mapped uploads. The change neither raises
the pool limit nor adds public options.

Three alternating before/after pairs used all 200 frames, 16 Rayon
threads, 12 preparation workers, six GPU calls and a 16 GiB pipeline
budget. Tracing was disabled, and no build, test or other benchmark ran
alongside them. The baseline and modified executables were saved separately.

| Median of three untraced runs | `a245500` | Reusable frame uploads |
|---|---:|---:|
| Whole stack | 31.84 s | 28.82 s |
| Reference to final estimation | 19.84 s | 18.49 s |
| Process CPU time | 255.3 CPU-s | 265.0 CPU-s |
| Peak host working set | 11.05 GiB | 10.73 GiB |
| Peak total VRAM, including desktop | 12.43 GiB | 12.15 GiB |
| Mean sampled GPU utilization | 17.8% | 16.0% |

Baseline runs took 31.84, 32.41 and 30.23 seconds; modified runs took
30.47, 28.52 and 28.82 seconds. This is **9.5% less median elapsed time**
with **3.8% more median process CPU time**. The GPU busy percentage again
does not rise with throughput. Individual phase medians need not sum to
the median whole-stack time.

A trial mapping each immutable scratch file once instead of reopening
and reading its bands was removed. The complete stack took 34.20 seconds,
with final integration alone taking 12.69 seconds and a reported 47.47 GiB
working set, compared with roughly 7–9 seconds integrating through the
bounded reader. Mapping moved file-backed pages into the process's working
set; it did not make this Windows workload faster. Its pixels still matched.

The ignored `target/gpu-tuning/` directory contains the paired runs,
`profile-paired-summary.jsonl`, the stage summaries and the saved executables.
`profile-stages.py` aggregates host spans, and `profile-pairs.py` aggregates
whole-run timings and telemetry. The normal CLI invocation above reproduces
the configuration; omit `--gpu-trace` when timing it.

Reducing GPU concurrency to three, keeping 12 workers and 16 Rayon threads,
took 27.05, 28.94 and 28.00 seconds before the startup change below. A single
32-thread/six-call trial took 27.49 seconds but 373.9 CPU-s. More CPU threads
and more GPU calls did not consistently improve the balance.

### Device time and initial residency

Tracing now also reports compute-pass timestamp queries, when supported,
separately from the host spans. Timestamp readbacks share the existing
output wait. Normalization stays asynchronous: its timestamps travel with
the resident frame and are read at admission. The old host label `kernel`
is replaced by queue/readback labels so it cannot be mistaken for device
compute time. An adapter without timestamp support retains host profiling.

The following traced Vulkan run uses the original 16-by-16 resampling
workgroups, 16 Rayon threads, 12 preparation workers and three GPU calls:

| Device compute stage | Calls | Median per call | Sum of intervals |
|---|---:|---:|---:|
| Defect screen | 16 | 0.475 ms | 7.6 ms |
| Star detection, five dispatches | 200 | 1.957 ms | 398.3 ms |
| Resampling | 198 | 1.068 ms | 220.9 ms |
| Normalization | 191 | 0.353 ms | 74.5 ms |
| Admission and live integration | 191 | 2.433 ms | 500.1 ms |
| Final band integration | 88 | 2.118 ms | 186.8 ms |

These measured compute intervals sum to 1.39 seconds in a 27.65-second
run. They omit transfers, clears and mask scatter, and are not an occupancy
measurement or an additive breakdown of elapsed time. The host packing of
a 512 MiB band alone has a median near 27 ms. The frame dispatch has about
239,600 workgroups and each full band 2,731: there is ample pixel-level
parallelism, while the queue is fed intermittently. A band's frame reduction
retains frame order; changing it for more parallelism would change its
floating-point arithmetic.

A 32-by-8 resampling-workgroup trial measured 1.037 ms per call and
209.2 ms in total, against 1.068 ms and 216.2 ms in the first device-timed
16-by-16 run. The roughly 7 ms difference over 200 frames does not establish
an end-to-end gain or an optimum for other GPUs. The existing shape remains.

The timestamp counts exposed a startup transfer: only 179 of 191 incoming
accepted frames normalized on the GPU, matching the 12 preparation workers
that began before the lazy GPU accumulator existed. Offering the accumulator
before preparation keeps all 191 there, avoiding another 12 full-frame
uploads, about 2.94 GB. Its first untraced run took 26.04 seconds, with
identical final pixels. Offering it after workers started was also tested
and removed: two runs took 30.63 seconds traced and 30.00 untraced.

The new `a_pipelined_stack_large_enough_for_the_gpu_keeps_its_output` case
uses the existing large mono summaries, recorded with lazy startup before
the change. Corrupting only the GPU's initial mean makes its live summary
fail; that mutation was undone. Both this case and the original large mono
case run with CPU, GPU and traced GPU options and cover repeated bands and
a short tail. Existing summaries and output versions are unchanged.
Local-background pipeline startup has its own new live and reintegrated
summaries, first recorded with lazy startup and verified on CPU, GPU and
traced GPU. A planted initial-mean corruption fails that case too.

### Combined branch after the shared-memory band update

Upstream `2ac5d30` was incorporated after this work was validated. It raises
shared-memory GPUs' band cap to the same 512 MiB already used on this
NVIDIA adapter; its Metal and Iris Xe measurements are in the design notes.
Three final untraced runs on the combined code, using 16 Rayon threads,
12 preparation workers, three GPU calls and a 16 GiB pipeline budget, took
**28.07, 28.47 and 28.01 seconds**. The median is **28.07 seconds / 264.75
CPU-s**, against the earlier 31.84-second / 255.27 CPU-s baseline with six
GPU calls: **11.9% less elapsed time and 3.7% more process CPU time**, with
the lower concurrency included. Median peaks are 9.24 GiB host working set
and 11.04 GiB total VRAM. The 26.04-second prototype result above is a
single faster run, not the final median.

All three final outputs are bit-identical across all 61,171,488 pixels.
Every frame's admission decision is unchanged (192 accepted, eight rejected),
as is the reintegration's 60,666,768 rejected samples. The ignored
`profile-merged-final-summary.jsonl` and pixel comparisons record this check.

```powershell
$env:RAYON_NUM_THREADS = '16'
& .\target\release\seiza.exe stack @lights --reference first --reintegrate `
    --workers 12 --gpu-concurrency 3 --pipeline-memory-mib 16384 `
    --output '<GPU output.fits>' --scratch-directory '<local scratch>' `
    --report '<GPU report.json>'
```

The final checks include 468 passing stacking unit tests (five existing
ignores), 36 stacking and three stretch output cases, and all four large
GPU output cases on DX12/DXC, including traced global and local-background
pipeline startup. Stacking Clippy, the CPU-only build, workspace and Python
formatting, binding parity, release CLI/C ABI builds and the three-frame
native C ABI smoke test pass. The generated C header stays unchanged.

## Direct band reads and compact live-only readback

Cache bands now stream decoded, normalized rows straight into disjoint
parts of the GPU's mapped upload buffer. They no longer allocate full CPU
input bands and then serialize a second packing copy. Two GPU buffer sets
still overlap submission and collection. Mapping pairs of frames keeps
view offsets aligned even when a frame has an odd number of samples.
The CPU integration path keeps its existing read-ahead bands; a source
error or cancellation still returns without a GPU-to-CPU retry, and a GPU
failure still retries with fresh outputs and counts.

A live-only stack on a discrete GPU can also leave a full-grid, unsited
resampled image on the device. It gathers exactly the CPU normalization
and photometry's sample positions, including the fallback fits, and counts
finite samples on the GPU with integer reductions. The original CPU fits
run on those unchanged values, in their original order. This is not an
approximate GPU fit. One recent reference plan is shared between workers;
dense plans are rejected once and remembered, and the extra shaders compile
only when compact readback is actually used. Shared-memory adapters,
partial footprints, channel-sited and photosite frames keep the original
readback. The change adds no public option or binding surface.

On the same 200 mono 61 MP FITS files, with 16 Rayon threads, 12 workers,
three GPU calls and a 16 GiB pipeline budget, the final untraced runs gave:

| Median of three runs | Direct band reads, full readback control | Retained implementation |
|---|---:|---:|
| Live-only elapsed | 16.03 s | **13.46 s** |
| Live-only process CPU | 178.30 CPU-s | **164.19 CPU-s** |
| Live-only peak host working set | 7.80 GiB | **6.70 GiB** |
| With reintegration, elapsed | 27.39 s | 28.24 s |
| With reintegration, process CPU | 259.34 CPU-s | 260.33 CPU-s |

Live-only elapsed falls **16.0%**, process CPU **7.9%**, and median peak
working set **14.1%**. The final live-only times were 13.76, 13.30 and
13.46 s. The controls were 16.03, 15.98 and 16.14 s. A traced final run
reports **16,232,116 bytes instead of 244,685,952 bytes** per resampled
image, a **93.4% reduction** in that image's readback, including the finite
count. Other transfers such as initial uploads and clip-mask packets remain.

There is no repeatable elapsed improvement for the full reintegration run.
The final range is 27.48–28.84 s, against the control's 26.42–28.09 s;
the previous PR head's median was 28.07 s. Reintegration still needs the
full unnormalized cache payload, and that path retains the existing full
readback and cache writer. A prototype gathering compact samples and then
reading the full cache payload took about 29.2 s. Sharing one mapped
readback between fitting and cache writing reduced process CPU but gave a
28.04 s median in paired runs, against the control's 27.39 s. A raw-byte
cache-write trial also established no gain. Those cache changes were removed.

The control and final measurements above include direct band reads. That
change alone did not establish an end-to-end speedup here: its first
untraced and traced reintegrations took 27.37 and 27.31 s. It removes the
extra input-band storage and serialized packing step without changing band
boundaries or arithmetic. The compact readback is where the repeatable
live-only improvement comes from.

All twelve control and retained FITS files were checked against their
reported SHA-256 hashes and are byte-identical within each mode, including
every one of the 61,171,488 output pixels. Every frame decision and every
reported diagnostic is identical, with 192 accepted and eight rejected
frames; the reintegrations reject the same 60,666,768 samples. The ignored
`selected-final-summary.jsonl` records the individual runs and checks.

Both output versions and all existing summaries are unchanged. The large
global and local-background pipeline cases cover direct band reads; two
new live-only cases use their already-recorded live summaries and cover
compact readback. Corrupting gathered values fails both new cases, and a
planted band payload corruption fails the cached local-background case.
All planted changes were undone. Additional unit checks compare complete
CPU fits with gathered mono/RGB samples, test finite counts and sample
bits on the GPU, and exercise odd-sized bands, cancellation and rollback.

Local validation passes the complete stacking and stretch test suites:
470 and 60 unit tests, respectively, their integration/doc tests, and 38
stacking plus three stretch output cases (existing ignores stay ignored).
All six large GPU output cases also pass on DX12/DXC. Stacking Clippy with
all targets/features and warnings denied, the CPU-only all-targets check,
workspace/Python formatting, binding parity, release CLI/C ABI builds and
the three-frame native C ABI smoke test pass. The generated C header is
unchanged.
