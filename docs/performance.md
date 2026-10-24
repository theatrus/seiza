# Performance

[Documentation](README.md) · [Project overview](../README.md)

These are recorded release-build measurements, not timing guarantees. Hardware,
input dimensions, catalog depth, file cache, CPU threads, GPU backend, and output
options affect results. MP means megapixels; CPU-s is total CPU time across
threads, not elapsed time. Times for a complete command must not be compared to
kernel-only times.

[Operation coverage](#operation-coverage) · [Solving](#plate-solving) ·
[Stacking](#stacking) · [I/O and display](#image-io-and-display) ·
[Background](#background-correction) · [Catalogs](#catalogs-and-downloads) ·
[Reproduction](#reproduce-and-compare)

## Operation coverage

This table lists measurements for each operation family. **Unmeasured** means
the repository documentation has no separate timing for that operation.
Pipeline totals include several operations; use an isolated benchmark to
measure an individual function.

| Operation | Recorded timing or memory evidence | Limits / main cost controls |
|---|---|---|
| Star detection | Included in solve timings; compact detector peak memory 179 MiB for a 61 MP FITS frame, versus 590 MiB previously | No standalone current detector timing; image size, sampling backend, thresholds |
| Star measurement and tilt | Unmeasured | Detection count, measurement mode, field grid |
| Hinted solve | 0.04–0.23 s, 0.10 s median, 17 FITS frames | i5-1340P, October 2026; whole CLI command |
| Blind solve | 0.04–0.16 s, 0.09 s median, 16 FITS frames | Same laptop; prebuilt index; scale range and field affect search |
| SIP fit | Included in the linear/SIP comparison below | Standalone fit unmeasured; order, matches, field distortion |
| WCS transforms and FITS cards | Unmeasured separately | Number of points, projection, SIP order |
| Catalog name/cone/footprint/distance queries | Unmeasured separately | Catalog depth, query area, OS page cache |
| Catalog installation and validation | July complete bundle: 3.93 GB unpacked, 2.90 GB compressed; 3.74 s local validation/decompression | Download/network time and exhaustive catalog validation unmeasured |
| Overlay and sky-map rendering | Unmeasured separately | Output size, label/object count; not included in every solve command |
| FITS decode | 16.5 ms, 125.2 MB peak RSS, warm local 61 MP frame | macOS profiling, median of five single-decode processes; uncompressed input |
| XISF/raster decode and image writing | Unmeasured separately | Codec/compression, bit depth, channels, disk |
| Statistics and display stretch | ~8 ms statistics, ~25 ms legacy stretch on 26 MP | Legacy desktop observation; no standalone times for GHS, layered, or ordered stacks |
| Calibration application, matching, grading, response/dust analysis | Unmeasured separately | Input dimensions, master count, sample/feature count; application is included in calibrated stack runs |
| Bias/dark/flat master construction | Unmeasured separately | Number/size of frames, rejection, storage, bounded tile workspace |
| Registration and resampling | Included in whole-stack times; per-pass profiles linked below | CPU/GPU, transform model, kernel, native/binned grid |
| Normalization, weighting, live accumulation | Included in whole-stack times | No standalone portable benchmark; canvas, channels, normalization model |
| Completed-stack rejection/reintegration | Included in the `--reintegrate` stack timings | Cache/disk bandwidth, bands, CPU/GPU, rejection settings |
| Drizzle | 43 s for 62 × 26 MP mono frames at 2×; 159 s for 126 × 26 MP color frames at 1× | i5-1340P; whole commands, including reintegration, not drizzle-only kernels |
| Mosaic planning, sky orientation, context save/resume | Unmeasured separately | Output area, overlap, frame/context size and disk |
| Background fitting/correction | ~0.7 s complete 26 MP command, including corrected/model FITS and JSON | Warm local smoke test; fit-only and correction-only timings unmeasured |
| Color composition/cropping | Unmeasured separately | Registration, output area, channel count, palette, crop search |
| Photometric color calibration | Unmeasured separately | Offline catalog versus network, number of usable stars, photometry and output |
| Deconvolution | Unmeasured; corpus study measures quality only | Dimensions, channels, PSF width, iteration count; four iterations by default |
| Satellite prediction | Unmeasured | Catalog size, exposure duration, sampling, element-cache/network availability |
| Parallax preparation, tour planning, rendering and encoding | Unmeasured | Image/output size, stars, depth layers, frame count, codec and encoder |
| Image-processing primitives | Unmeasured | Dimensions, channels, kernel/filter parameters |

Sources and measurement scopes follow. Unmeasured operations need a dedicated
benchmark before a numerical speed claim can be made.

## Plate solving

Solve timings below include process startup, image loading, star detection,
catalog access, solving, and result output.

On a laptop (Intel Core i5-1340P, Fedora, release builds, October 2026), with
`stars-deep-gaia17.bin` and `blind-gaia16.idx`, three runs of each image:

- Blind solves of 16 real FITS frames, from 2 to 61 megapixels, took
  **0.04–0.16 seconds, 0.09 s median**.
- Hinted solves of 17 frames took **0.04–0.23 seconds, 0.10 s median**.
  Checking a 26 MP frame's own header WCS takes 0.07–0.10 s.
- Ten processed AstroBin JPEGs of about 22 megapixels solved blind in
  0.19–0.59 s, 0.35 s median; decoding the JPEG is most of that.

On our Windows 11 test machine in July 2026 (Intel Core i7-12700H, release
builds, no GPU), before the October solver work:

- A real hinted FITS solve usually finished in **0.2-0.4 seconds**.
- Blind solving with a prebuilt index took **0.53 seconds median** across 13
  real FITS frames ranging from 26 to 61 megapixels.
- Compact u8 detection cut peak detector memory from **1.08 GiB to 543 MiB**
  on a 94 MP JPEG, and from **590 MiB to 179 MiB** on a 61 MP FITS frame.

The [scientific regression corpus](../regression/README.md) checks astrometry
and calibrated stacks against known results, tests pinned camera images, and
records runtime and peak memory.

### Compared with Asterion

[Asterion](https://github.com/asterion-solver/asterion) comparison on the laptop
above: blind solves, seven runs of each program interleaved, each with its own
catalogs (Asterion 0.1.8 with gaia-500 and tycho2):

| Image | seiza, linear | seiza, SIP order 3 | Asterion |
|---|---:|---:|---:|
| M45, 26 MP colour raw | 0.06–0.08 s | 0.06–0.08 s | 0.07–0.16 s |
| IC 5070, 61 MP mono | 0.09–0.11 s | 0.08–0.11 s | 0.10–0.17 s |
| M31, 8 MP colour raw | 0.03–0.05 s | 0.03–0.05 s | 0.06–0.09 s |
| NGC 253, 2 MP mono | 0.03 s | 0.03–0.04 s | 0.07–0.22 s |
| `elephant`, 22 MP AstroBin JPEG | 0.30–0.32 s | 0.30–0.32 s | 0.31–0.35 s |

Asterion chooses a SIP polynomial order by default. Seiza defaults to a linear
fit; use `--sip-order` to request SIP. Linear and SIP solves had similar wall
times on these telescope frames.

Phone-photo results varied with the crop. Asterion solved a cropped 24 mm frame
in 0.19 s; Seiza required about 5 s. Asterion's defaults did not solve the full
phone frames, which Seiza solved blind in about 0.4 s.

### Compared with ASTAP

[ASTAP](https://www.hnsky.org/astap.htm) comparison from July 2026, before the
October solver work: repeated real-FITS runs using the catalog recommended for
each solver. Results depend on the field and search strategy.

| Workload | seiza | ASTAP | Result |
|---|---:|---:|---|
| Accurate hint, 26 MP narrow field | 0.25-0.27 s | 0.27-0.29 s | Roughly tied; seiza 9-11% faster |
| Accurate hint, 61 MP wide field | 0.42 s | 0.63 s | seiza 1.5x faster |
| Position blind, 26 MP narrow field | 1.61 s | 31.28 s | seiza 19x faster |
| Position blind, 61 MP wide field | 0.65 s | 1.09 s | seiza 1.7x faster |

The 61 MP position-blind row was rerun after the blind-pipeline improvements
(three runs each: seiza 0.62-0.65 s, ASTAP 1.08-1.16 s).

On a separate set of 25 heavily processed JPEGs, seiza solved all 25 with a
hint and all 25 position-blind. ASTAP solved 13 and 12 respectively. Among
images both programs solved, seiza was **3.5x faster hinted** and **6.5x faster
position-blind** by median wall-time ratio. Seiza also solved all 25 with no
position or scale hint in 0.90 seconds median. These processed web images test
a different workload from ASTAP's normal FITS use.

Runs used the normal OS file cache and included complete command-line wall time.
See the
[FITS comparison](benchmarks/2026-07-astap-comparison.md) and
[blind/detection follow-up](benchmarks/2026-07-blind-pipeline-priorities.md)
for the images, catalogs, repetitions, correctness checks, and caveats.

## Stacking

Whole `seiza stack --reintegrate` runs in October 2026, wall time and CPU
time:

| Data | Machine | CPU only | With the GPU |
|---|---|---:|---:|
| Pleiades mono, 200 × 61 MP | M4 Max (Metal) | 68.9–71.0 s, 1057–1097 CPU-s | 12.4–12.5 s, 111–112 CPU-s |
| Pleiades mono, 200 × 61 MP | i5-1340P, Iris Xe (Vulkan, Linux) | 273–276 s, 3944–3962 CPU-s | 90.2–90.5 s, 616–620 CPU-s |
| Pleiades mono, 200 × 61 MP | Threadripper PRO 9975WX, RTX 5090 (Vulkan, Windows), 16 threads | 103 s, 1571 CPU-s | 27.1 s, 208 CPU-s |
| Bode's Galaxy L, 12 × 26 MP mono | i7-12700H, Iris Xe (Vulkan, Windows) | 8.09 s, 111 CPU-s | 4.79 s, 16 CPU-s |
| Bode's Galaxy L, 12 × 26 MP mono | i7-12700H, Arc A770M (Vulkan, Windows) | 8.09 s, 111 CPU-s | 3.83 s, 14 CPU-s |
| M45 colour, 40 × 26 MP | M4 Max (Metal) | 17–18 s | 5.1–5.3 s, 30–31 CPU-s |
| M45 colour, 40 × 26 MP | i5-1340P, Iris Xe (Vulkan, Linux) | 62 s, 811 CPU-s | 38.8–38.9 s, 158 CPU-s |

For the M4 Max Pleiades stack, GPU acceleration reduced total CPU time to about
a tenth of the CPU-only run. Peak memory fell from 9.6–9.8 GiB to 6.8–7.0 GiB.
On the i5-1340P, peak memory fell from 8.9–9.2 GiB to 3.5–3.6 GiB; shared-memory
bandwidth limits the Iris Xe's throughput.

The Threadripper CPU-only stack finished in 55–57 s when allowed all 64 threads.
The GPU run used 12 preparation workers and a 16 GiB pipeline budget. See the
[Windows NVIDIA report](benchmarks/2026-10-windows-nvidia-stacking.md) for the
protocol and output checks.

The [GPU design report](design/gpu-stacking.md) includes per-pass equivalence,
kernel timings, CPU/GPU memory, and backend-specific limitations. Those kernel
times exclude parts of a complete stack command. For example, its 26 MP
resampling measurements are 4–7 ms on M4 Max GPU versus 97 ms CPU, and 29 ms on
Iris Xe GPU versus 186 ms CPU; these are not per-frame pipeline latency.

[Binned stacking experiments](design/binned-stacking.md#7-measurements) also record
CPU costs and memory tradeoffs. Native decode/calibration and registration work
limit the speedup, even when a smaller canvas cuts integration memory sharply.

### Drizzle

The [CLI stacking observations](../seiza-cli/README.md#defaults-and-quality-options)
record these October 2026 whole-command times on an i5-1340P laptop (16 threads,
32 GB RAM). GPU settings and repetition counts are not specified there:

| Input | Mode | Wall time | Output check |
|---|---|---:|---|
| M45, 126 × 26 MP color, no calibration | Default stack | 77 s | 2.7 px FWHM |
| Same M45 inputs | `--reintegrate` | 122 s | Completed-stack rejection |
| Same M45 inputs | `--drizzle 1` | 159 s | 2.69 px FWHM, SNR 356 |
| H-alpha, 62 × 26 MP mono | `--drizzle 2` | 43 s | 1.78 px FWHM in reference pixels |

These are a different dataset and protocol from the CPU/GPU comparison above;
they do not isolate the cost of drizzle or imply the same throughput at every
drizzle scale. Mosaic planning, reprojection, and context save/resume have no
standalone timing recorded.

## Image I/O and display

[Streaming FITS profiling](design/fits-streaming-io.md#release-profiling) used
optimized `io_profile` examples, local disk, and macOS `/usr/bin/time -l`.
Median warm-cache load and peak RSS across five single-decode processes:

| Input | Streamed decode | Peak RSS |
|---|---:|---:|
| 61 MP mono, 122 MB | 16.5 ms | 125.2 MB |
| 8 MP RGGB, 16 MB | 2.4 ms | 19.5 MB |
| 2 MP mono, 4 MB | 0.7 ms | 7.0 MB |

The 61 MP whole-file-buffer baseline took 25.8 ms and 247.6 MB. Streaming
allocates the final typed image plus a bounded conversion buffer, avoiding a
second full raw-file copy. This does not establish compressed FITS, XISF,
raster-decoder, or image-writer performance.

The [FITS crate README](../seiza-fits/README.md) separately records approximate
26 MP desktop costs: ~75 ms loading, ~8 ms statistics, and ~25 ms stretching to
display range. It does not identify the host or repetitions. These older
observations are lower-confidence than the profiling table and apply to the
legacy statistics/Auto-MTF display path, not every `seiza-stretch` model.

## Background correction

[Initial real-image validation](design/background-extraction.md#initial-real-image-validation)
ran the release CLI on local 6248 × 4176 mono H-alpha and three-plane SHO stacks.
Both commands took about **0.7 s** with warm files, writing a corrected FITS,
a full model FITS, and JSON diagnostics. These are developer-machine smoke
tests without a specified hardware/repetition protocol, not portable benchmarks
or isolated fit/apply timings. The mono fit accepted 87 of 96 windows; RGB
accepted 77 of 96 shared windows.

Fitting retains compact samples and coefficients. In-place correction does not
require a full-size model allocation; rendering the model image does. Polynomial
selection and opt-in RBF have different fitting costs.

## Catalogs and downloads

The [July 2026 compression report](benchmarks/2026-07-catalog-zstd.md) records a
complete eight-artifact bundle of **3,927,799,223 bytes unpacked** and
**2,900,245,820 bytes with zstd level 22**, saving 26.2% of transfer bytes.
Sequential local validation/decompression took **3.74 s** on the development Mac.
Installation also includes network transfer and disk writes. Artifact versions
and preset selection change the download size. Clients store the unpacked files
for memory mapping.

Object, star, and identifier catalogs use memory mapping and indices so normal
queries page in relevant records instead of scanning every entry. No current
standalone name/cone/footprint query latency is recorded. The
[cone-search optimization report](benchmarks/2026-07-cone-search-optimization.md)
measures changes within solve workloads, not a universal query-time guarantee.

## Reproduce and compare

Build outside the timed run, then use the
[scientific regression harness](../regression/README.md):

```sh
cargo build --release --locked -p seiza-cli --bin seiza --example regression_fixtures
python3 regression/run.py --hosted --output /tmp/seiza-main.json
python3 regression/run.py --hosted --binary /path/to/candidate/seiza \
  --baseline /tmp/seiza-main.json --output /tmp/seiza-candidate.json
```

The default protocol discards one warmup and measures three repetitions with
two Rayon threads. Reports retain per-sample wall time, quality, input/binary
hashes, and environment metadata. GNU time supplies peak RSS and CPU time on
Linux; macOS/Windows reports have no comparable RSS value. Use a quiet host,
matching fixtures, harness, thread count, build profile, and feature settings.
Alternate base/candidate runs for optimization claims.

The harness measures a new worker or stack process for each run, including
startup, loading, and lazy index construction. Measure a persistent worker
separately to determine its latency after startup. The harness excludes
compilation, downloads, fixture creation, and output verification from its times.
CI fails on incorrect scientific results and records performance for review.
A baseline comparison on the same machine can also reject runtime or memory
regressions.

For isolated FITS load profiling:

```sh
cargo build --release --locked -p seiza-fits --example io_profile
target/release/examples/io_profile stream image.fits 50 --load-only
```

When adding a timing, record the input dimensions/channels, hardware, OS,
revision, build/features, catalogs, cache state, repetitions, outputs, and whether
it measures a kernel, an API call, or a complete process. Keep the correctness
check alongside the speed measurement. The
[deconvolution corpus](benchmarks/2026-07-deconvolution-corpus.md), for example,
is a quality comparison and supplies no runtime benchmark.
