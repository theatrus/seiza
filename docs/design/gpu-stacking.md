# GPU stacking

Status: resampling and the final integration's band passes run on the GPU,
behind `seiza-stacking`'s default `gpu` feature. More stages are follow-on
work (see the end).

## What runs where

`seiza-stacking` reaches the GPU through [wgpu](https://wgpu.rs): Metal on
macOS, Vulkan on Linux, and Direct3D 12 or Vulkan on Windows, chosen at run
time. It opens the device once, the first time a frame large enough to gain
from it arrives. Two stages run there:

- **Registration resampling** (`registration::resample_region_with_inverse`):
  clamped Lanczos-3 and bilinear, mono or RGB, over the whole reference grid
  or a footprint of it. Every frame of a live stack and every frame a final
  pass prepares goes through it, and so does the clipped-pixel mask. Bayer
  drizzle's nearest-photosite sampling stays on the CPU.
- **Final integration** (`batch::bands`): the three passes of leave-one-out
  rejection over one band of rows of every frame, with weights, the trimmed
  first estimate and the Student-t limits. A band that records sample fates
  for a drizzle is integrated on the CPU.

Everything else (decoding, calibration, debayering, star detection,
registration fits, normalization fits, the live accumulator) stays on the
CPU for now.

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
eight 1280x960 frames, which reach both GPU stages, against summaries
recorded on the CPU, and passes on the GPU too, so the GPU path keeps
`OUTPUT_VERSION`. A planted change in the band kernel fails it, as does a
gross change in the resampler; a sub-pixel change in the resampler is below
what a summary sees, and the GPU-against-CPU unit tests in `gpu.rs` catch
that instead. Those tests, and the case, run on the CPU alone where no GPU
is present, as on CI.

## Controls and fallbacks

- Build without the `gpu` feature (`default-features = false`) for a
  CPU-only crate without wgpu.
- `SEIZA_GPU=off` (or `0`, `cpu`), set before the first large frame, keeps
  everything on the CPU. `SEIZA_GPU_BANDS=off` keeps the final integration
  there while resampling stays on the GPU. `WGPU_BACKEND` chooses the
  backend, as wgpu reads it.
- No adapter, a software adapter, or a frame too large for a GPU buffer
  keeps the work on the CPU. A GPU error (device lost, validation) is
  caught, logged once, and turns the GPU off for the rest of the process;
  the frame or band it hit is done on the CPU. Nothing panics on the GPU's
  account.
- An integrated GPU (Apple Silicon, Intel Iris Xe) reads and writes buffers
  the CPU maps directly; a discrete one goes through staging copies. Mapped
  output is copied out on every core, because one core reading
  write-combined memory is several times slower.

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

## Follow-on work

Measured in the `gpu-spike` experiments, not yet in the crate:

- **The live accumulator on the GPU.** Resampling, normalization,
  delta-sigma classification and the weighted Welford update in one pass,
  on stack state that stays on the GPU in two buffers: the pass writes the
  state after the frame, and admitting the frame swaps them. 11.8 ms a
  frame on the M4 Max and 44 ms on the Iris Xe, against 141 ms and 277 ms
  on their CPUs. `LiveStacker::view` borrows the CPU's state today, so this
  needs a readback when a host asks for it.
- **A tiled resampling kernel for Intel.** Workgroup-shared source tiles cut
  the Iris Xe's kernel from 29 to 21 ms and slow the M4 Max eightfold, so
  the kernel would be chosen by adapter.
- **Calibration, debayering and normalization on the GPU**, which would let
  a frame go up once, raw, and come back only as its registered result.
