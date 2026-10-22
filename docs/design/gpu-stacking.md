# GPU stacking

Status: VNG debayering, resampling, the live accumulator and the final
integration's band passes run on the GPU, behind `seiza-stacking`'s default
`gpu` feature. More stages are follow-on work (see the end).

## What runs where

`seiza-stacking` reaches the GPU through [wgpu](https://wgpu.rs): Metal on
macOS, Vulkan on Linux, and Direct3D 12 or Vulkan on Windows, chosen at run
time. It opens the device once, the first time a frame large enough to gain
from it arrives. These stages run there:

- **VNG debayering** (`image::LinearImage::debayer_with`): every pixel two
  or more from each edge of a Bayer mosaic, with the channel balance the
  CPU measures. The two-pixel border, which takes bilinear estimates, stays
  on the CPU, and so do MHC, bilinear and superpixel debayering.
- **Registration resampling** (`registration::resample_region_with_inverse`):
  clamped Lanczos-3 and bilinear, mono or RGB, over the whole reference grid
  or a footprint of it. Every frame of a live stack and every frame a final
  pass prepares goes through it, and so does the clipped-pixel mask. Bayer
  drizzle's nearest-photosite sampling stays on the CPU.
- **The live accumulator** (`stack::Accumulator`): a live stack's moments
  move to the GPU with its first frame of a million samples or more, and
  each frame is classified against them (the admission gate) and integrated
  there, delta-sigma rejection and the weighted Welford update included. The
  CPU's copies are freed. [`LiveStacker::view`], snapshots and saved
  contexts read the moments back, once a frame, the first time they are
  asked for, into a copy the next frame drops; the borrowed view stays valid
  for as long as the borrow, since only `&mut` methods integrate.
- **Final integration** (`batch::bands`): the three passes of leave-one-out
  rejection over one band of rows of every frame, with weights, the trimmed
  first estimate and the Student-t limits. A band that records sample fates
  for a drizzle is integrated on the CPU.

Everything else (decoding, calibration, defect detection, star detection,
registration fits, normalization fits) stays on the CPU for now.

The GPU takes a frame only when it holds at least a million samples; below
that its fixed cost, a few milliseconds, outweighs the gain, and the small
synthetic frames of the unit tests and most `output_version` cases resample
on the CPU. The decision is made by the source frame, not the region asked
for, so a footprint holds exactly what resampling the whole grid would.

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
- `SEIZA_GPU_ACCUMULATOR=off` keeps the live accumulator on the CPU, and
  `SEIZA_GPU_DEBAYER=off` VNG debayering.
- `SEIZA_GPU_TRACE=1` prints every GPU call's stages (grid, upload,
  kernel, copy out) to stderr, to see where the time goes on a machine.
- `SEIZA_GPU=off` (or `0`, `cpu`), set before the first large frame, keeps
  everything on the CPU. `SEIZA_GPU_BANDS=off` keeps the final integration
  there while resampling stays on the GPU. `WGPU_BACKEND` chooses the
  backend, as wgpu reads it.
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
  the CPU maps directly; a discrete one goes through staging copies. Mapped
  buffers are filled and copied out on every core, because one core reading
  write-combined memory is several times slower, and output goes straight
  into its vector rather than over zeros, which cost as much as the copy.

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

## Follow-on work

- **Speculative integration.** One pass that classifies and integrates
  into a second copy of the moments, kept when the frame is admitted: in
  the `gpu-spike` experiments 44 ms a frame against 76 ms for separate
  passes on the Iris Xe, for twice the moments' memory.
- **Workers.** The pipeline's default worker count follows a memory budget
  set for CPU preparation; with the GPU doing the heavy passes, more
  workers pay.
- **A tiled resampling kernel for Intel.** Workgroup-shared source tiles cut
  the Iris Xe's kernel from 29 to 21 ms and slow the M4 Max eightfold, so
  the kernel would be chosen by adapter.
- **Calibration, debayering and normalization on the GPU**, which would let
  a frame go up once, raw, and come back only as its registered result.
