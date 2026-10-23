# GPU stacking

Status: defect learning's screen, VNG debayering, star detection,
resampling, the live accumulator and the final integration's band passes
run on the GPU, behind `seiza-stacking`'s default `gpu` feature. More stages are follow-on work
(see the end).

## What runs where

`seiza-stacking` reaches the GPU through [wgpu](https://wgpu.rs): Metal on
macOS, Vulkan on Linux, and Direct3D 12 or Vulkan on Windows, chosen at run
time. It opens the device once, the first time a frame large enough to gain
from it arrives. These stages run there:

- **Defect learning's screen** (`defects::frame_deviations`): the first
  test of every sample of a light the defect map is learned from, against
  the extremes of its eight same-plane neighbours. Nearly every sample
  passes it; the GPU flags the rest in a bitmask, the test loosened by a
  hundred-thousandth so it flags every sample the CPU's exact test keeps,
  and the CPU examines only those, the exact test first. The deviations,
  and so the map, are bit for bit those of examining every sample.
- **VNG debayering** (`image::LinearImage::debayer_with`): every pixel two
  or more from each edge of a Bayer mosaic, with the channel balance the
  CPU measures. The two-pixel border, which takes bilinear estimates, stays
  on the CPU, and so do MHC, bilinear and superpixel debayering.
- **Star detection** (`registration::detect`): the luminance, normalized
  to 0..=1 as each pass reads it, each 64x64 tile's median and robust sigma
  by exact radix selection (an even tile's upper middle sample is its lower
  one again, or the least above it, not a second selection), and the
  threshold, which writes the pixels that pass in row-major order. A mono
  frame's luminance is read where the frame lies, not copied. `seiza::detect_stars_from_excess` gathers them into
  stars on the CPU: the same flood fill as detection over the whole excess
  image, over those pixels alone, which gives the same stars to the bit.
- **Registration resampling** (`registration::resample_region_with_inverse`):
  clamped Lanczos-3 and bilinear, mono or RGB, over the whole reference grid
  or a footprint of it. Every frame of a live stack and every frame a final
  pass prepares goes through it, and its clip mask with it: one pass finds
  each pixel's source position once, resamples the frame there and the
  mask bilinearly, to the bit as two passes would. For clipping statistics,
  sparse input masks are sent as index/value pairs and expanded on the
  GPU, then only the resampled mask's nonzero contributions are returned.
  The CPU accumulates those in frame order, preserving the dense path's
  exact sums. Both lists are bounded; a dense mask uses the original upload
  and, on output overflow, is resampled again as a dense mask. This avoids
  full mask transfers in either direction for sparse clipping. The CPU
  builds its mask packet and coordinate grid before taking a GPU-call
  permit, so that Rayon work cannot hold a permit while it runs another
  frame's task that needs one. Bayer drizzle's
  nearest-photosite sampling stays on the CPU.
- **The live accumulator** (`stack::Accumulator`): a live stack's moments
  move to the GPU with its first frame of a million samples or more, and
  each frame is classified against them (the admission gate) and integrated
  there, delta-sigma rejection and the weighted Welford update included, in
  one submission: the host turns the minimum integrated fraction into the
  count of accepted samples it takes, and the integrating pass acts only if
  classification reached it. The
  CPU's copies are freed. [`LiveStacker::view`], snapshots and saved
  contexts read the moments back, once a frame, the first time they are
  asked for, into a copy the next frame drops; the borrowed view stays valid
  for as long as the borrow, since only `&mut` methods integrate.
- **Final integration** (`batch::bands`): the three passes of leave-one-out
  rejection over one band of rows of every frame, with weights, the trimmed
  first estimate and the Student-t limits. Two reusable buffer sets separate
  submission from collection: the CPU stages and submits the next band
  before collecting the previous one's outputs. On discrete GPUs each set
  has one mapped upload buffer rather than a separate staging write for
  each frame. Frame order and rejection
  arithmetic stay the same. A band that records sample fates for a drizzle
  is integrated on the CPU. A GPU failure retries the whole integration on
  the CPU with fresh outputs and frame counts; progress starts again at the
  first band. Cancellation and source read errors return without a retry.

The band pipeline holds at most two submissions in flight. CPU integration
retains the 128 MiB input band cap. GPUs, discrete or sharing the CPU's
memory, use up to 512 MiB per band to amortize reads and submissions, within
half the configured band memory and the device's input and output binding
limits. On the Iris Xe that took the Pleiades stack's final integration
from 25 s to 17.5 s; on the M4 Max it stayed at 2.3 s. The CPU reader,
both GPU sets and their staging buffers use those same rows; output
buffers also scale with that row count. `GpuOptions::concurrency`
continues to limit frame preparation calls, independently of these two
band buffer sets. Band traces report host upload/staging, time in flight
while other work runs, the remaining wait and copying outputs. These spans
overlap across bands and are not GPU timestamp measurements. Mask traces
also report actual sparse upload bytes and bounded readback bytes.

The resampled image stays on the GPU through normalization and admission.
A full CPU image copy still serves normalization fitting and the scratch
cache used for reintegration. Removing that copy needs compact fit samples
and a different cache strategy; sparse clipping does not remove it.

Everything else (decoding, calibration, registration fits, normalization
fits) stays on the CPU for now.

The GPU takes a frame only when it holds at least a million samples; below
that its fixed cost, a few milliseconds, outweighs the gain, and the small
synthetic frames of the unit tests and most `output_version` cases resample
on the CPU. The decision is made by the source frame, not the region asked
for, so a footprint holds exactly what resampling the whole grid would.

A frame's samples are read from the GPU where they already are when they
can be: VNG's output, its border the CPU fills written in, stays on the GPU
for star detection and resampling to read, and a frame detection uploads,
as a mono one, is read again there by resampling. Each thread keeps at most
one such frame, known by where its samples lie, how many there are and a
fingerprint of 1024 of them, and gives it back when the frame is prepared.
Resampling's output stays on the GPU too, while a live stack's moments are
there: the CPU fits the frame's normalization from its copy, the GPU
applies it where the frame lies (one gain and offset per channel fused as
the CPU's `mul_add`, bit for bit; a tile grid interpolated with the CPU's
own weights, within a few units in the last place), and the accumulator
reads it there. The CPU's copy is dropped once the frame is normalized, so
a frame waiting to be integrated holds one copy, not two; a frame the GPU
cannot integrate after all comes back from it, normalized. The debayered
frame still comes back to the CPU, which measures its noise for weighting.

## How close it stays to the CPU

The GPU works in single precision; the CPU evaluates source positions in
f64 and accumulates moments in f64.

- **Positions.** The host evaluates the frame's geometry (similarity, warp,
  lens) in f64 on a grid every 16 reference pixels, anchored on the
  reference grid itself, and stores each node's source position as whole
  pixels and the fraction past them. The shader interpolates positions
  relative to one node's whole pixels, so a pixel's fraction is good to
  about a millionth of a pixel however far it lies from the origin. The
  first version interpolated plain f32 positions; at 6000 pixels from the
  origin those carry 2-3 ten-thousandths of a pixel of rounding, enough to
  pick the neighbouring 1/1024-pixel weight step a few percent of the time,
  and on a bright star's flank that step is tens of ADU.
- **Weights and clamping.** The same 1025-step Lanczos-3 table as the CPU,
  the same clamping rule, and the same bilinear fallback. NaN is tested by
  its bits, since shader compilers may assume no float is NaN.
- **Debayering.** The same gradients, threshold and colour differences in
  f32, as the CPU's are. The mosaic is balanced to green in a pass of its
  own: balanced as the kernel read each pixel's window, the product was
  fused into the gradients' differences, which flipped a few directions on
  stars' flanks (0.6 of the noise at most, on a few pixels in a million).
  Balanced first, a frame matches the CPU's to 6e-6 of the noise RMS, no
  sample beyond a thousandth of it. The window is written out cell by cell
  so it stays in registers; indexed through tables, as on the CPU, the
  kernel ran five times slower.
- **Star detection.** The tiles' medians and robust sigmas are exact, and
  so is the threshold: on a frame already spanning 0 to 1, which
  normalizing leaves alone, the GPU finds the CPU's stars bit for bit, on
  Metal and on Vulkan. Normalizing in f32, where the CPU works in f64,
  moves a sample by a unit in its last place, and a pixel on the threshold
  can fall the other way; on a calibrated Sh2 140 frame of 200 stars, the
  stars that keep every pixel lie within 2e-4 of a pixel of the CPU's, and
  up to five a frame lose or gain one and move further.
- **Integration.** The CPU stores moments in f32 between f64 updates; the
  GPU does the updates in f32 too. Rejection decisions agree at all but a
  handful of samples that lie on a threshold.

On the Sh2 140 Hα regression set (45 frames, 6248x4176), the stack with the
GPU against the stack without it, both reintegrated:

| | |
|---|---:|
| RMS difference | 0.0045 of the stack's noise (0.004 on the sky) |
| Pixels beyond 0.1 / 0.5 / 1 noise | 2181 / 674 / 12 of 26.1 M |
| Largest difference | 6.6 noise, on a bright star's flank |
| Coverage | identical |

The band passes alone, GPU against CPU on the same resampled frames, leave
84% of samples bit-identical and 3 pixels beyond 0.1 noise.

These differences sit well inside `tests/output_version.rs`'s tolerance:
`a_stack_large_enough_for_the_gpu_keeps_its_output` stacks and reintegrates
eight 1280x960 frames, which reach the resampler, the accumulator and the
band kernel, and `a_debayered_stack_large_enough_for_the_gpu_keeps_its_output`
stacks six RGGB frames of that size, which reach the VNG kernel, against
summaries recorded on the CPU. Both pass on the GPU too, so the GPU path
keeps `OUTPUT_VERSION`. A planted change in the band kernel or the VNG
threshold fails them, as does a gross change in the resampler; a sub-pixel
change in the resampler is below what a summary sees, and the
GPU-against-CPU unit tests in `gpu.rs` and `image.rs` catch that instead.
The VNG kernel balancing as it read the window, above, moved the colour
case's green maximum by 1.3e-5 and failed it. Those tests, and the cases,
run on the CPU alone where no GPU is present, as on CI.

On the 40-frame M45 OSC set (6248x4176 RGGB, `--reintegrate`), the stack
with the GPU's VNG against the CPU's, everything else on the GPU in both:
0.0013 of the noise RMS on the sky in each channel. Under 170 pixels a
channel differ by more than 0.1 of the noise, all on bright stars, where a
star's centroid on a frame moves a hair with the demosaic and its
registration with it.

## Controls and fallbacks

- Build without the `gpu` feature (`default-features = false`) for a
  CPU-only crate without wgpu.
- Each request carries a `GpuOptions`, which says which passes may run on
  the GPU: `StackOptions::gpu` for a live stack and its reintegration,
  `BatchStackOptions::gpu` for `integrate_registered_frames_banded` (and to
  narrow a reintegration's rejection passes), and
  `DefectDetectionOptions::gpu` for defect learning. The C ABI takes it as
  the `gpu` object of the options or request JSON, Python as
  `seiza.GpuOptions`, and the CLI's `stack` as `--no-gpu` and
  `--cpu-pass`. `enabled: false` keeps everything on the CPU and never
  opens the device; `accumulator`, `debayer`, `detection`, `defects`,
  `resampling` and `integration` keep one pass each.
- `concurrency` sets how many frame calls (debayering, detection,
  resampling, screening) run on the GPU at once, three by default: a frame
  waiting for one holds no GPU buffer.
- `trace` prints every GPU call's stages (grid, upload, kernel, copy out)
  to stderr, and the time opening the device took (12 ms on the M4 Max,
  46 ms on an Iris Xe), to see where the time goes on a machine.
- The options decide where the arithmetic runs, which moves results by
  rounding at most, so a stack's configuration fingerprint leaves them out;
  its context keeps them. `WGPU_BACKEND` chooses the backend, as wgpu reads
  it.
- No adapter, a software adapter, or a frame too large for a GPU buffer
  keeps the work on the CPU. A GPU error (device lost, validation) is
  caught, logged once, and turns the GPU off for the rest of the process;
  the frame or band it hit is done on the CPU. A live stack whose moments
  are on the GPU reads them back and carries on on the CPU; if the device is
  lost and they cannot be read, `snapshot`, `into_snapshot` and
  `save_context` fail saying so rather than return a wrong stack. Nothing
  panics on the GPU's account.
- Frame-sized buffers return to a pool of at most 1 GiB: making a new one
  maps and clears fresh memory, which took longer than the upload.
- An integrated GPU (Apple Silicon, Intel Iris Xe) reads and writes buffers
  the CPU maps directly; a discrete one goes through staging copies. A
  frame goes in and comes out on the core that asked for it: frames are
  prepared on several cores at once, and copying each on every core made
  the call wait behind other frames' work in the pool (0.5 s longer on the
  M4 Max's M45 stack, 70 CPU-s more on the Iris Xe's, for no time saved).
  A live stack's moments, which nothing else is reading beside, come back
  on every core, since one core alone reads write-combined memory several
  times slower. Output goes straight into its vector rather than over
  zeros, which cost as much as the copy.
- Each call waits for its own submission, not for the queue to empty:
  waiting for the queue held every preparing thread behind every other's
  kernels, and cost the Iris Xe's M45 stack a second.

## Measurements

The 45-frame Sh2 140 Hα stack with `--reintegrate` (seiza CLI, release):

| Machine | CPU only | With the GPU |
|---|---:|---:|
| M4 Max, 16 cores, 40-core GPU (Metal) | 9.5 s, 136 CPU-s | 4.6 s, 34 CPU-s |

The twelve frames of 2026-09-30, with `--reintegrate`:

| Machine | CPU only | With the GPU |
|---|---:|---:|
| i9-13900H, Iris Xe (Vulkan, Fedora 44) | 8.3-8.9 s, 112 CPU-s | 5.0 s, 33 CPU-s |

Both machines were shared with other work, so wall times are noisy; CPU
time is the steadier measure. The resampling kernel alone, per 26 MP frame:
4-7 ms on the M4 Max against 97 ms on its CPU, 29 ms on the Iris Xe against
186 ms on its CPU.

With the accumulator on the GPU, the M4 Max's live pass integrates a
26 MP frame in about 11 ms (7.4 ms to upload and classify, 3.7 ms to
integrate) against 30-40 ms on its CPU. The 45-frame live stack matches
the CPU's to 0.00015 of the noise RMS, no pixel beyond 0.5 noise. The live
pass is then bound by preparing frames: with 12 workers rather than the
default it takes 19 ms a frame against 28.

The 40-frame M45 OSC set with `--reintegrate` on the M4 Max, VNG on the GPU
against VNG on the CPU, everything else on the GPU in both: 6.9-7.2 s and
54-56 CPU-s against 7.8-7.9 s and 74-77 CPU-s. The VNG kernel takes 4-7 ms
a 26 MP frame on an idle GPU, against about 14 ms (225 CPU-ms) on the
CPU's 16 cores; debayering fell from 28% of the workers' busy time to 2%.
With each call waiting for its own submission and copying on its own core,
6.7-7.2 s and 51-53 CPU-s.

Learning the defect map from 16 of the M45 lights took 1.1 s of that on the
M4 Max, before any frame was prepared; with the GPU's screen it takes
0.65 s, most of it reading and calibrating the lights, and the run 10
CPU-s less (39 against 49). On the Iris Xe, 1.7 s against 3.2 s, and 200
CPU-s against 228. The stacks are byte-identical either way.

The same set on an i5-1340P with its Iris Xe (Vulkan, Linux): 41-42 s and
240 CPU-s, against 62 s and 811 CPU-s on the CPU alone. There the GPU is
the bottleneck, busy for most of the run with about half a second of
kernels a frame, so VNG on the GPU saves CPU time (271 CPU-s against 299)
but not wall time.

200 mono 61 MP Pleiades frames with `--reference first --reintegrate`, as
on Windows below, copied to local disk, before (0f1a547) and after
double-buffered bands and compact clip masks (a245500), and with 512 MiB
bands on these shared-memory GPUs as well. Medians; every image is
byte-identical to the others on the same machine:

| Machine | 0f1a547 | a245500 | 512 MiB bands | CPU only |
|---|---:|---:|---:|---:|
| M4 Max (Metal) | 18.7 s, 138 CPU-s | 13.9 s, 124 CPU-s | 14.6 s, 119 CPU-s | 77.8 s, 1160 CPU-s |
| i5-1340P, Iris Xe (Vulkan) | 142 s, 950 CPU-s | 105 s, 828 CPU-s | 96.8 s, 735 CPU-s | |

The M4 Max's final integration fell from 4.0 s to 2.3 s and its live pass
from 13.4 s to 10.4 s; the Iris Xe's from 54 s to 25 s, then 17.5 s with
512 MiB bands, and from 83.6 s to 75.8 s. Peak memory fell from 12.7 GiB to
10.7 GiB and from 7.0 GiB to 5.5-6.1 GiB. Uploading the clip mask
sparsely on these GPUs too saved nothing measurable on either and cost
4-23 CPU-s, so it stays on discrete GPUs, which save the copy across the
bus. The Mac shared its cores with other work, so its wall times are
noisy.

Windows NVIDIA validation is recorded separately in
[`2026-10-windows-nvidia-stacking.md`](../benchmarks/2026-10-windows-nvidia-stacking.md):
200 mono 61 MP Pleiades frames on a Threadripper PRO 9975WX and RTX 5090,
through Vulkan and Direct3D 12, including the discrete GPU staging path.

## The Iris Xe's limit

On the i5-1340P's Iris Xe, M45 takes 41 s: 5 s choosing the reference,
learning defects and preparing it, 23 s for the live pass (about 600 ms a
frame), 12 s reintegrating. Each stage's kernel is 4-20 times the M4 Max's,
and a mapped buffer costs more besides: the driver maps it afresh each time,
and the first write to each 4 KB page faults, about 36 ms of a 47 ms copy of
128 MB, against 3 ms for the whole copy on the M4 Max. Uploading through
`queue.write_buffer` instead was slower, 49 s against 41.

The live pass there is bound by what the CPU and the GPU share, the
memory: a GPU call is in flight 98% of it, the one thread integrating
frames spends 80% of it in its classify-and-integrate call, and none of
these moved it:

| | live pass |
|---|---:|
| as it stands | 23.2-23.5 s |
| VNG on the CPU | 22.9-23.5 s (38 CPU-s more) |
| VNG in workgroup tiles (17% faster kernel) | within 0.5 s |
| classify and integrate in one submission | 23.2 s |
| each frame staged on the GPU by its worker | 23.4-23.7 s |
| 4 to 12 workers | 40.6-42.5 s for the run |

Staging frames took a third off the integrating thread's calls, and it
then waited as long for the next frame. What would move it is fewer bytes
crossing memory: a frame that goes up once and stays (calibrated,
debayered, resampled and integrated where it lies, coming back only as the
luminance star detection needs and the normalized frame a reintegration
keeps). A discrete GPU, where each crossing is a PCIe transfer, would gain
the same way.

## Follow-on work

- **Frames that stay on the GPU** from calibration to integration, above.
  A debayered frame already does from VNG to integration. What still
  crosses is the debayered frame back (for its noise), the registered frame
  back (for the normalization fit, photometry and the cache), and the clip
  mask both ways.
- **Speculative integration.** One pass that classifies and integrates
  into a second copy of the moments, kept when the frame is admitted: in
  the `gpu-spike` experiments 44 ms a frame against 76 ms for separate
  passes on the Iris Xe, for twice the moments' memory. One submission now
  does both passes; the second copy is what remains, and on the Iris Xe
  the passes are not what holds the live pass.
- **Tiled kernels for Intel.** Workgroup-shared tiles cut the Iris Xe's
  VNG kernel from 92-97 to 78-81 ms and its resampling kernel by 28% in
  the `gpu-spike` experiments, and slow the M4 Max's three and eight times;
  neither moved the Iris Xe's run, for the reason above.
- **NVIDIA on Linux** remains untested. Windows Vulkan and Direct3D 12
  validation, including the discrete GPU staging path, is recorded above.
