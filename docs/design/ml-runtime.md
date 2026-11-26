# Learned model runtime

Status: a first build works through the CLI, the C ABI and Python (see
[Work order](#work-order)). Every figure from it was measured on one Mac. The
NVIDIA and Intel measurements come when those machines are arranged.

## Short answer

The training harness runs our models in Python. This plan moves that into a
new crate, `seiza-model`, reached from the CLI, the C ABI and Python.

One binding runs every model on every platform: ONNX Runtime, through the
`ort` crate. On Apple hardware it drives Core ML, so models still run on the
Neural Engine and the GPU. This replaces the two backends of the
[restoration plan](ml-restoration-training.md#backends), which had a second,
native Core ML binding and a second export format for it. Measured on the
same model, ONNX Runtime's Core ML provider ran the whole graph with no CPU
fallback, as fast as native Core ML and with the same error, so the second
backend would buy nothing.

A model ships as one dynamic ONNX graph per precision plus a model card.
Seiza's Rust code does everything around the graph: sky, noise input, tiling,
blending, the output bound and provenance. The backend only runs tiles.

A 61 MP frame should take about 15 seconds on an M4 Max's GPU, the default,
with the current 41M-parameter model, or about 16 on its Neural Engine. Those
are estimates from measured 26 MP runs.

[Every shipped model works from the image alone](ml-restoration-training.md#short-answer),
so a host passes an image and nothing else, and `seiza stack` writes nothing
new. Which noise input ships is not settled, but the choice costs the runtime
nothing (see [Noise input](#noise-input)).

## What the models need

The harness has moved past the restoration plan, which sized this for
1–3M-parameter networks. A runtime has to honour what the models are now.

| | Restoration plan | Now (the D-series noise runs) |
|---|---|---|
| Network | width 16–32, four levels, 1–3M parameters | width 64 or 96, five levels, 41M or 93M parameters |
| Compute | 0.2 MFLOP per pixel | 1.76 MFLOP per pixel at width 64 |
| 61 MP frame | 15 TFLOP | 544 TFLOP at tile 512; 216 at tile 1024 |
| Weights | not stated | 166 MB f32, 83 MB f16 at width 64; 187 MB f16 at width 96 |
| Noise plane | `asinh(σ / σ̄)` | the same, and a run's card must say so |
| Sky | one background | `flat` runs take out a smooth sky first |

Each run's reading of its inputs is in its `run.json`, and a model given the
wrong reading fails badly, so all of it moves to the model card:
`noise_map`, `noise_plane`, `sky`, and constants such as `SKY_BLOCK` (32),
`FLAT_BLOCK` (256) and `IMAGE_MAP_SCALE` (1.13).

Three things the restoration plan describes do not yet exist anywhere.

- **The output bound.** "The noise model's output is bounded by its input"
  was tried in a script and recorded, but the trainer has no such code, so
  no run has been measured with it. The Rust port is its first real
  implementation, and the harness needs the same bound or its gates do not
  judge what ships.
- **A model that needs only the image.** The trainer can now make both
  image-only inputs, but every run on record trained and measured on maps
  built from the stack's variance.
- **A Core ML export.** The harness writes ONNX only.

Only mono models exist, so RGB runs a mono model on each channel. The noise
model will change again, and PSF and star models need other inputs, so a
model's shape is its card's business, not the code's. See
[Model shapes](#model-shapes).

## Measurements

Model: D22, width 64, five levels, 41.5M parameters. Image: a 4-frame
Elephant's Trunk Hα half, 6248 × 4176. Reference: PyTorch f32. Errors are in
units of the image's median noise, σ̄, on the residual the network returns.
The machine is an M4 Max.

The machine was shared. Another job held the GPU and a corpus build held
the CPU, so **GPU and CPU times are upper bounds**. The Neural Engine was
free. Errors do not depend on load.

| Path | Precision | 512 tile | 1024 tile | 26 MP whole | Sky RMS | All RMS | Largest |
|---|---|---|---|---|---|---|---|
| PyTorch on Metal (today's harness) | f32 | 0.24 s | | | under 1e-6 | 1e-6 | 4e-4 |
| ONNX Runtime CPU | f32 | 0.35 s | | | under 1e-6 | 1e-6 | 4e-4 |
| Core ML, GPU | f32 | 0.15–0.24 s | 0.54 s | 30.7 s | 4e-7 | 1e-6 | 4e-4 |
| Core ML, GPU | f16 | 0.12 s | 0.44 s | 25.1 s | 0.0007 | 0.0028 | 2.2 |
| **Core ML, Neural Engine** | f16 | **0.030 s** | **0.13 s** | **6.7 s** | 0.0143 | 0.0164 | 1.4 |

The largest errors sit on bright stars, whose residuals reach 1,000σ̄ and
more. The Neural Engine's mean error on the sky is −0.0005σ̄.

**ONNX Runtime reaches the Neural Engine.** Given an f16 graph and
`CPUAndNeuralEngine`, its Core ML provider took 0.030 s a tile, against
0.031 s through native Core ML, with the same 0.0149σ̄ error on the same
tile. It ran with CPU fallback disabled, so every node was on Core ML. An
f32 graph with the same setting quietly runs on the CPU, at 0.51 s, so the
f16 graph is what opens the Neural Engine.

**One graph serves every tile size.** A graph with named height and width
axes, fixed when a session opens by ONNX Runtime's dimension override, ran
at both sizes at full speed. The file is the same size as a static one.

**Tile size is the cheapest speed-up there is.** Margins cost the same
pixels at any tile size, so a larger tile wastes less:

| Tile | Stride | Useful share | Tiles, 61 MP |
|---|---|---|---|
| 512 | 224 | 19% | 1,176 |
| 768 | 480 | 39% | 260 |
| 1024 | 736 | 52% | 117 |
| 1536 | 1248 | 66% | 40 |

Going from 512 to 1024 took the whole image from 81.5 s to 30.7 s on the
GPU, and from 15.6 s to 6.7 s on the Neural Engine. It changed nothing that
matters: outputs differ by 7e-8σ̄ RMS on the GPU and 1e-8σ̄ on the Neural
Engine.

**f16 makes no seams.** Two tile grids 368 pixels apart, the crops' own
edges left out, differ by 7e-9σ̄ RMS and at most 1e-6σ̄ on the Neural Engine.
On the GPU at f32 they differ by 1.6e-7σ̄. The Neural Engine's error is
large beside that but does not depend on where a tile falls.

**The largest model still fits.** Width 96 (93M parameters) ran on the Neural
Engine at 0.32 s per 1024 tile and on the GPU at f16 at 0.86 s, which puts a
61 MP frame near 38 seconds.

**Core ML's model cache ignores the dimension override.** With
`ModelCacheDirectory` set, a session opened at 1024 loaded the cached 512
graph in no time, then failed at its first run with "Error executing model".
The cache directory has to carry the tile size itself.

**From Rust.** `seiza model denoise` ran D22 on the same image, reading the
constant map, with Microsoft's 1.31.1 macOS library. The reference is the
harness on the same graph through ONNX Runtime's CPU provider. Times cover the
whole run, the noise input and sky included.

| Backend | Precision | Tile | 26 MP whole | Sky RMS | Sky mean | Largest |
|---|---|---|---|---|---|---|
| CPU | f32 | 512 | 165 s | 1e-8 | 0 | 1.2e-5 |
| GPU | f32 | 1024 | 6.4 s | 4.9e-7 | 0 | 0.0011 |
| GPU | f16 | 1024 | 5.9 s | 0.0004 | −5e-6 | 0.81 |
| Neural Engine | f16 | 1024 | 6.7 s | 0.0147 | −0.0013 | 1.06 |
| Neural Engine | f16 | 512 | 15.6 s | 0.0147 | −0.0013 | 1.06 |

The Rust path reproduces the harness, and the Neural Engine's time matches
Python's. The 1024-tile times alternate the three paths over four rounds with
nothing else on the GPU; one GPU f32 round took 7.9 s. Run one after the other
an hour or two earlier, the GPU took 9–10 s, for a reason not recorded, and
shared it took the 25–31 s above, while the Neural Engine's time did not move.
So an idle GPU on this chip beats the Neural Engine at either precision, with
about a 35th of its sky error at f16. The Neural Engine's one advantage is
that other work on the GPU does not slow it. Its mean error here is closer to
its budget than the −0.0005σ̄ above. D24 on another stacker's master gave the
same picture: 5.9 s and 0.0004σ̄ on the GPU, 6.7 s and 0.0153σ̄ on the Neural
Engine. Decision 6 follows from this.

**Not yet measured:** anything on NVIDIA or Intel; any Neural Engine older
than the M4's; a loaded-versus-idle GPU comparison; any smaller GPU. This one
has 40 cores, and a base M4 has a quarter of that beside a Neural Engine of
the same size, so the Neural Engine may well be the faster one there.

## Decisions proposed

1. **One binding: `ort`, with `load-dynamic`.** Take `objc2-core-ml`, the
   `.mlpackage` export and the `coreml` feature out of the restoration plan.
   None of them was built, so no code changes. Native Core ML stays
   available later, as a second `TileRunner`, if a host cannot carry a
   33 MB library. This keeps the decision that Core ML runs models on Apple
   hardware and changes only how Seiza reaches it.
2. **Two graphs per model, f32 and f16, with dynamic height and width.**
   A host fetches the one its backend needs. Safetensors weights stay in the
   store, because nothing at run time reads them.
3. **Tile size belongs to the backend, not the export.** Default to 1024 on
   a GPU or the Neural Engine, and 512 where memory is short. Whole-image
   parity already holds across sizes.
4. **The Neural Engine has its own parity budget.** Decided. The restoration
   plan allows 0.01σ for every accelerated backend. The GPU passes at f16 and
   the Neural Engine does not, at 0.0143σ̄ on the sky. The Neural Engine's
   budget is sky RMS under 0.02σ̄ and mean under 0.002σ̄, with seams as for
   any backend. With 15% of the noise kept, an error that size adds about
   0.5% to the noise left, in quadrature. A model still earns the Neural
   Engine by passing its own parity run.
5. **Pin `ort` to `=2.0.0-rc.15` and raise the Rust floor to 1.92.** Decided.
   There is no stable 2.0 yet, so the pin is exact, and Dependabot needs an
   ignore rule. With no `api-N` feature, `ort` asks only for ONNX Runtime 1.17
   or later.
6. **The GPU at f32 is the default on Apple hardware.** Decided. With nothing
   else on it, it took 6.4 s against the Neural Engine's 6.7 s, and its result
   is the CPU's to 4.9e-7σ̄ on the sky and 0.0011σ̄ on the brightest star. f16
   would save 0.5 s for about 800 times that error on the sky, so it is for
   the Neural Engine, which needs it, or for `--precision f16`.
   `--backend neural-engine` leaves the GPU free and keeps its time when the
   GPU is busy, and `detect` picks it only for a card that leaves the GPU out.
   The price is memory: a 26 MP run's peak footprint at a 1024 tile was 3.9 GB
   at f32, 2.5 GB at f16 and 1.1 GB on the Neural Engine. On an 8 GB Mac that
   may call for `--tile 512`, which has not been tried there.

## The crate

`seiza-model` depends on neither `seiza-stacking` nor `seiza-stars`
for the noise model. It takes a borrowed image and makes everything else a
model reads from it.

```text
seiza-model/src/
  lib.rs       the public names and OUTPUT_VERSION
  model.rs     NoiseModel, Denoised, ImageView
  backend.rs   Backend, BackendOptions, detection, library search
  session.rs   the ONNX Runtime session
  bundle.rs    ModelBundle, ModelCard, verification
  store.rs     ModelStore: installed models
  tiles.rs     tile starts, window weights, blending
  noise.rs     the noise input: image_sigma, or its median, as the card says
  sky.rs       smooth sky, background and noise scales, the network's input
  bands.rs     NoiseBands
  parity.rs    tests against the harness's recorded answers
```

`TileRunner` is the seam between Seiza and any runtime. It stays private to the
crate, so it can grow named tensors when a PSF or star model needs several
inputs:

```rust
pub(crate) trait TileRunner {
    fn run(&mut self, planes: &[f32], side: usize, residual: &mut [f32]) -> Result<(), ModelError>;
}
```

Tests drive tiling and blending through a closure, so most of the crate
builds and tests with no ONNX Runtime at all.

### Backends

```rust
pub enum Backend {
    CoreMl(CoreMlUnits), // NeuralEngine, Gpu or Cpu
    TensorRt,
    TensorRtRtx,
    Cuda,
    DirectMl,
    WebGpu,
    Cpu,
}
```

`Backend` has `FromStr` and `Display` for `--backend`. `Backend::detect`
picks the best one the machine can run and the model's card allows.
`Backend::available` reports each one with the reason it cannot run, which
is what a user needs when a GPU goes unused.

| Hardware | Order tried | Graph |
|---|---|---|
| Apple Silicon | GPU, Neural Engine, CPU | f32, or f16 on the Neural Engine |
| NVIDIA | decided by the measurements; see [Work order](#work-order) | f16 or f32 |
| Windows with a DirectML-enabled runtime | DirectML, then CPU | f32 |
| Other Intel and AMD GPUs | WebGPU on explicit selection, CPU by default | f32 |
| Anything else | CPU | f32 |

DirectML uses the high-performance DXGI adapter by default. An explicit
adapter index selects a particular GPU. It runs with sequential execution
and memory patterns disabled. The bundled CPU runtime is still usable on
machines without a DirectML-enabled runtime. WebGPU remains experimental.

Finding the library follows the [catalog path rules](data-paths.md): an
explicit path, then `SEIZA_ONNXRUNTIME`, then beside the executable, then the
shared data directory, then the system. A set variable that points at
nothing is an error, never a fall-through. ONNX Runtime can be loaded once
per process, so detection holds its result for the process's life.

A compile cache sits in the platform cache directory, keyed on the model
hash, tile, precision, backend and ONNX Runtime version. The measured failure
above is why the tile is in the key. A first TensorRT build can take minutes,
so compiling is a progress phase a host can show.

### The tile loop

The rules are the ones in
[Tiling and parity](ml-restoration-training.md#tiling-and-parity), which the
Rust port must keep exactly: tiles start on the model's grid, weight is zero
inside the margin, and a raised cosine crosses the rest of the overlap.

Three things change from the harness.

- **Memory.** The harness holds f64 sums and weights for the whole image,
  about 1 GB at 61 MP. Rust blends in row bands, so peak memory stays a few
  tiles' worth, as the restoration plan requires.
- **Overlap of work.** A Neural Engine tile takes 124 ms, so preparing and
  blending tiles on the CPU must run beside it, not between tiles. One
  bounded queue on each side of the runner does this.
- **Threads.** ONNX Runtime's CPU pool takes its size from the caller's
  rayon pool, so a host that limits Seiza limits this too.

A cancel is checked between tiles and also reaches `RunOptions::terminate`,
because one CPU tile on a slow machine can take many seconds.

### Noise input

The graph takes two inputs: the image and a noise plane. A noise map gives,
at each pixel, how much noise that pixel should hold: its standard deviation
σ. The graph reads it as `asinh(σ / σ̄)`, and the map's median σ̄ sets the
units of both planes.

Every run so far reads a map built from a stack's variance and coverage. A
shipped model cannot, because a linear FITS file from any stacker is all it
gets. So a host passes an image, `seiza stack` writes nothing new, and a step
after stacking leaves nothing out of date.

Two inputs are candidates, and the choice is not settled:

| Candidate | Card's `noise_map` | What fills the plane |
|---|---|---|
| Blind | `constant` | the median of `image_sigma` everywhere: the level, and nothing of where it changes |
| Image map | `image` | `image_sigma`: second differences in 32-pixel blocks, plus a term growing with the signal |

Both come from `image_sigma`, so the runtime does the same work for either.
`noise.rs` ports it, and the card says whether the map or its median fills
the plane. Step 2 need not wait for the choice. A later model that drops the
plane altogether changes only the card's named inputs.

D20, trained on the stack's map, was measured unchanged on the rebuilt
validation sets, reading each input in turn. Its error against matched halves:

| Map read | 4-frame halves | 8-frame halves |
|---|---|---|
| The stack's, as trained | 1.709 | 1.184 |
| Image map | 1.760 | 1.181 |
| Constant | 1.761 | 1.192 |

The constant does as well as the image map, so the model barely uses where
the map changes, and a host's image is enough even for a model trained with
the stack's map.

Still open:

- **Models trained blind.** D27 and D28 are queued. No model trained on
  either input has faced the gates.
- **The worst pairs.** With the constant, a 4-frame pair's error rose 1.4% at
  the median, but 22% on the worst. Noise level does not explain it: on nine
  pairs from both ends and the middle, the stack's level was 0.85–1.31 times
  the image's, and the gap did not follow the change in error. The cause is
  unknown.
- **Other targets and stackers.** Ten of the twelve validation sets are
  nights of one target, and all twelve are Seiza's stacks. On 24 train sets'
  4-frame halves, the image map read 1.15–1.31 times the noise of the halves'
  difference on the sky, steadier than the stack's modelled map at 1.19–1.61.
  The table below is one target from another stacker, which is not enough.
  The gates need other targets and stacks from other software.
- **Drizzled stacks.** Second differences read correlated noise low, and
  drizzle correlates it. The table's 2× drizzle read 1.63 times under its real
  noise, and the model took out far less. The level needs a correction for
  correlated noise, or the model a way to find it.

D24 read the constant map on three stacks, with every amount at 1. Noise is
measured inside empty 64-pixel blocks, each block's own plane taken out, so a
gradient cannot count as noise. A star's change is at its peak, for stars over
200 times the noise.

| Stack | Real noise over the level read | Noise left at 1 / 2 / 4 / 8 px | Star peaks, median |
|---|---|---|---|
| Seiza, 4 frames, Elephant's Trunk Hα | 0.93 | 3.9 / 5.5 / 9.6 / 17% | −3.1%, 5% of them past −18.5% |
| Another stacker's master, NGC 7331 L | 0.94 | 2.9 / 3.2 / 4.5 / 7.2% | +0.46% |
| The same stacker's 2× drizzle | 1.63 | 44 / 52 / 75 / 81% | −0.27% |

So a stack from another stacker needs nothing Seiza's does not, and its noise
level reads as truly as Seiza's own. At 0.7 and 0.9, about 30% of the noise
stays at every scale on the first two.

A variant that reads Seiza's own records may sit beside the image-only model,
for Seiza's stacks. Nothing here builds it. If one earns its place,
`modelled_sigma` needs only its block grid and one gain, about 25,000 numbers
for a 26 MP frame, which fits a small FITS extension.

### Sharpening

`SharpenModel` runs the PSF model: an image brought from its own PSF to a
narrower one. It takes the image and two numbers, the share of their width
stars are brought to and the share asked of everything that is not a star
(1 leaves things as they are; the sheets ask for 0.46 and 0.5). Everything
else the network reads is made from the image by fixed steps, which the
bundle's card names:

| What the network reads | How it is made from the image |
|---|---|
| The image, `asinh((x − b) / σ̄)` | The background `b` and median noise `σ̄` are the noise model's scales: the medians of every 17th pixel of the image and of its noise map. |
| The noise map, `asinh(σ / σ̄)` | `image_sigma`, as the noise model reads it, with its raise for spread noise. |
| The stars' ratio, FWHM(target) / FWHM(image) | The PSF field (`seiza-stars`' `PsfField`, with the card's settings; one PSF for the whole image when too few stars allow a field that varies) is sampled at a grid of nodes, both edges of each axis and at most 256 px apart. A node's width is a round Gaussian's, fitted to its stamp's core. Its ratio is the share asked, never under the card's floor in pixels over that width and never over 1, and the map is bilinear between nodes. |
| The rest's ratio | The same, from the second share. A model trained with one strength has no such plane. |
| The point sources, `asinh(p / σ̄)` | The stars the measuring detector finds on the image unsmoothed, with each of the card's structure removals in turn, and the clipped cores it turns away (found as the peaks of 16 px blocks at the image's clip level), each laid down at its flux and sub-pixel place as the stamp of the node nearest it. |
| The PSF, as a vector at every pixel of the coarsest grid | Each node's stamp, cut or padded to 129 px, is read as two planes (over its peak, and its asinh, which shows the wings) by the stamp encoder, once. The nodes' vectors are interpolated bilinearly to the middle of every coarsest pixel of each tile, so the output does not depend on where the tiles fall. |

The residual is added to the asinh of the image in σ̄ and taken back through
sinh, as the card says, and a pixel that is not finite stays NaN. A colour
image's channels run one by one, each with a PSF field, point sources and
scales of its own. An image with too few stars for any PSF is refused with
`ModelError::PsfField`, which a host can tell from every other refusal.

A sharpening bundle holds two graphs where a noise bundle holds one, because
one graph of planes and a stamp cannot follow a PSF that changes over a
tile: `encoder-f32.onnx`, stamp planes to a vector, and `body-f32.onnx` and
`body-f16.onnx`, planes and a vector map to the residual, with the body's
height and width free and the map's a tile over the model's grid. The
encoder runs on ONNX Runtime's CPU provider whatever runs the body: it is
small, it runs once for each node, and its vectors are then the same on
every backend. The card is a noise card with `task` `sharpen`, the sky read
as one background and no hole guard (a card that asks for either is
refused), the planes above, and a `sharpen` section: the encoder's file,
hash, tensors, stamp side and vector length; the vector map's tensor and
dimensions; how the residual is applied; the floor in pixels; the sharpest
share the run was trained to be asked for; the shares a run takes when its
caller names none; the PSF field's settings, the nodes' spacing and the core
a width is fitted to; the detector's settings and the clipped cores'; and,
where the model has one, the scaled pass below.
The other options `seiza-stars` takes for the two are set in `seiza-model`,
none left to its defaults. The sharpening
trainer's `bundle-sharpen` writes the bundle, and holds each
single-precision graph to
PyTorch on a random input. No sharpening model is published, and
`seiza model sharpen` fetches none: it runs a bundle named with `--model` or
installed from its directory.

`sharpen.points.structure_removals` names how the detector takes nebulosity
and gradients away before it looks for stars, and it runs once with each, in
order. One removal does not do: `filtered`, the detector's default, takes
bright stars on bright, structured nebula away with the nebula, and a star
missing from the point sources is sharpened as nebula and ringed dark;
`atrous`, the à trous wavelet residual, keeps those stars. The first
removal's stars come first, then the clipped cores, found against those
stars alone; a star a later removal finds is added only where it lies beyond
the card's core radius of every point before it, the cores among them, so it
never stands in for a saturated star's core. A card that names none detects
with `filtered` alone, as a card did before they were named, and its output
is unchanged. A removal named twice, or an empty list, is refused.

`SharpenOptions::rolloff` holds back the finest part of the change where it
lies under the noise, after the model: the harness's `rolloff_step`.
`SharpenOptions::hold_noise` then holds back its broad part, between 4 and
16 px, where that lies under so many times the noise the image's noise has in
the band were it white: the harness's `noise_hold_step`. A run trained on
pairs of many fields, asked 0.35, still raised the noise between 8 and 16 px on
the independent halves of NGC 6820 (held out) by 1.17, 1.24 and 1.27 in the
faintest, middle and brightest thirds of its cloud, for shared structure 1.31,
1.37 and 1.20, where the goal raises it by 1.17, 1.18 and 1.16 for 1.32, 1.34
and 1.17; held at 0.5, the noise there is 1.11, 1.16 and 1.17 and the structure
1.29, 1.35 and 1.20. The fixture holds both steps to the harness's arithmetic
on a known change. Both are 0 unless asked, and with both 0 the output is the
model's. The harness's `clarity_step` is not ported.

**The scaled pass.** A model trained mostly on stars 2 to 5 px across does
not narrow the rims of an image whose stars are 11.5 px across, whatever it
is asked: on the Elephant's Trunk sheet, asked harder at full size, a rim
grew taller rather than narrower, and the cloud mottled. Binned 3 × 3, where
its stars are 3.9 px across, the same model narrows them. A card's
`sharpen.scaled` names the pass the harness's sheets run: `stars_px`, the
stars' width an image is binned toward; `blend`, a share of the stars'
width; and `rest`, the share asked of what is not a star on the binned
image. Where the median of the nodes' widths over `stars_px`, rounded to a
whole number (a half to the even one, as the harness rounds), is 2 or more,
the image is binned by it, each block's finite pixels averaged, and
sharpened again, with a noise map, scales, PSF field, point sources and
vectors measured from the binned image as from any image. The binned run is
asked for the stars' ratio the full size asks, binned and held to the floor
in the binned image's pixels, and for the rest at `rest`, or at
`SharpenOptions::binned_rest`. Its change is brought back up by Keys' cubic
(a = −½), each binned pixel at the middle of its block. The result takes
everything finer than a Gaussian of `blend` times the stars' width from
that change and everything coarser from the full-size run's:
`image + G(full-size change) + up − G(up)`. The stars keep the full-size
change: brought back up from the binned grid by a separable cubic, a star's
change comes out square, with a dark moat where the cubic's negative lobes
fall, which a second-moment ellipticity cannot see. The point sources are
rendered as the network reads them, and where their light over the noise
map is at least `stars` times the noise (the card's `scaled.stars`, 2 when
a card does not say; 0 blends at the stars too) the full-size output is
kept whole, the blend fading in below that: `share · output + (1 − share) ·
blended`, with the share that light over `stars` times the noise, held to 1.
On the Elephant's Trunk, the share of a star's half-peak ring that follows
cos 4θ is 0.017 at full size, 0.037 to 0.040 with the blend at the stars,
and 0.017 with them kept. The roll-off then runs as before. Where the binned image has too few stars for a PSF, the channel is
sharpened at full size alone, and `SharpenedChannel::binned` says 1;
`SharpenOptions::full_size_only` holds a run to full size whatever the card
says. The trainer's `bundle-sharpen` writes the pass at 3.5 px, 0.5 and
0.35, the Trunk's best, with `stars` at 2, and with it asks 0.46 and 0.45 at
full size unless told (`--no-scaled` leaves it out). A card without it runs as before, so
the runtime's output for such a card is unchanged.

**Against the harness.** The harness here is the sharpening trainer's
Python inference, under `training`, and the fixtures are recorded with it.
The port is held to it piece by
piece on a synthetic star field with no runtime (the field's nodes, stamps
and widths, the point sources, both ratio maps, the point-source map, a
tile's vector map and the roll-off, and with a scaled pass, on a second field
of wider stars, the binned image, its scales and field, both binned ratio
maps, the stars' share, and a known change brought up and put together by
scale), and a tiny
model's whole output through ONNX Runtime agrees with PyTorch's to 1.3e-4σ̄
at most, and with the scaled pass to 9.8e-5σ̄. The harness renders
the point sources with a Fourier transform and the runtime lays each stamp
down directly; the two agree to rounding. The harness measures the PSF field
and detects the stars through this workspace's own code, so those agree
exactly.

On a 1536 px crop of the Elephant's Trunk Hα sheet, with a trained model that is not published, of
width 96 (94M parameters), the runtime against the harness on PyTorch, both
asked for stars at 0.46 and the rest at 0.5, in the image's noise σ̄:

| Backend | Largest difference | RMS | On the sky (within 5σ̄ of the background), RMS | Time |
|---|---|---|---|---|
| CPU, f32, tile 512 | 0.020 | 4.3e-5 | 5.6e-7 | 19.5 s |
| Core ML GPU, f32, tile 1024 | 0.017 | 4.1e-5 | | 1.7 s |
| Core ML GPU, f16, tile 1024 | 68 | 0.14 | 3.4e-4 | 1.3 to 1.9 s |
| Neural Engine, f16, tile 1024 | 242 | 0.64 | | 2.4 s |

The largest f32 difference is at a star's core 20,400σ̄ high, a millionth of
its value: single-precision rounding through the sinh. The crop holds too
few stars for a field that varies, so it ran with one PSF. Times are for the
whole crop on an Apple-silicon Mac, measurement and all, after the model is open.

The whole 26 MP sheet image (6248 × 4176) has a field that varies, measured
from 273 stars and sampled at 18 × 26 nodes, with 278 point sources. There
the Core ML GPU at f32 differs from the harness by 0.024σ̄ at most and
2.2e-5σ̄ RMS, in 18.7 s, and the CPU by 0.028σ̄ and 2.3e-5σ̄, in 259 s. With
a roll-off of 1 the RMS is 6.3e-5σ̄ on both.

With the scaled pass, a later model of the same width, its card written by
`bundle-sharpen` with the pass and asked what the card asks (0.46 and 0.45 at
full size, the rest at 0.35 binned), bins that image 3 × 3: its stars are
11.5 px across, and binned it is 2083 × 1392, with a field of 188 stars
sampled at 7 × 10 nodes. With the stars keeping the full-size change, the
Core ML GPU at f32 differs from the sheet's own output by 0.018σ̄ at most
and 1.2e-5σ̄ RMS, in 23.0 s against 18.0 s at full size alone, and the CPU
by 0.025σ̄ and 2.2e-5σ̄, in 281 s against 257 s. Its stars read 0.017 on
the cos 4θ measure, as the sheet's do.

Two later models of width 128 (167M parameters), their cards naming both
structure removals and the scaled pass, agree as well on that whole image,
which lists 287 point sources with both removals, asked 0.46 and 0.45. At f32
the first differs from the harness by 0.025σ̄ at most and 2.4e-5σ̄ RMS on the
Core ML GPU and by 0.045σ̄ and 2.9e-5σ̄ on the CPU; the second by 0.028σ̄ and
2.2e-5σ̄, and by 0.025σ̄ and 2.3e-5σ̄. The scaled pass gives the same numbers,
and with a roll-off of 1 the RMS is 1.3e-4σ̄ for the first and 3.2e-5σ̄ for the
second (2026-11-25).

**Half precision loses the stars' cores.** The residual is added in asinh,
so an error in it is a relative error in the pixel: the f16 graph is right
to 3.4e-4σ̄ on the sky, and wrong by 0.1 to 0.2% RMS, 0.6% at most, of every
pixel over 30σ̄, which is tens of σ̄ at a bright core. A sharpening bundle's
card should list f32 backends alone under `parity` until an f16 graph is
measured and found good enough, so the Neural Engine, which runs only f16,
is not chosen unasked.

**Memory.** A run holds the planes whole, five per pixel for a model with two
strengths, as the harness does, with its copy of the image and the noise map.
Beside them it holds an f64 plane twice: the point-source map while it is
rendered (with the f32 map made from it), and the blend's sums while the tiles
run (with the residual made from them). That is ten f32 planes' worth, 2.4 GB
beside the image and the result for a 61 MP frame. A roll-off runs once the
planes are dropped, beside the image's copy, the noise map, the residual and
the output: six f32 planes of its own, and a blur's f64 plane and the f32 one
it makes, thirteen f32 planes' worth with the run's four, or 3.2 GB. Cutting
each tile's planes as it runs would bring the first down to the point-source
map and the noise map, and letting the roll-off's steps go as each is used
would bring the second down by several planes. The scaled pass runs the
binned image after the full size's planes are dropped: its planes are a
ninth of those at 3 × 3 and a quarter at 2 × 2. Putting the two together
then holds the full-size output, the change brought up, the full-size change
and two blurs, with a blur's f64 plane while each is made.

## What hosts see

Following the [Rust-first rules](../../AGENTS.md#the-rust-api-comes-first).
The restoration plan's [native API](ml-restoration-training.md#native-api)
stands, with `Backend` as above.

**Rust.** PSF Guard calls these directly.

```rust
let options = BackendOptions::default();
let backend = Backend::detect(&bundle, &options)?;
let mut model = NoiseModel::open(&bundle, backend, &options)?;
let image = ImageView::new(&pixels, width, height, channels)?;
let result = model.denoise(image, &DenoiseOptions::default())?;
let kept = result.blend(&NoiseBands { high: 0.7, low: 0.9, scale_px: 5.0 });
```

`denoise` makes the noise input the card names from the image. `Denoised`
keeps the residual, so a slider for the noise kept costs a blend and no
inference.

**CLI.**

```text
seiza model denoise stack.fits -o clean.fits --high-amount 0.7 --low-amount 0.9
seiza model denoise stack.fits -o clean.fits --backend cpu --tile 512
seiza model list | install [<name|dir>] | verify | remove <name>
seiza model backends      # which backends run here, and why the others do not
```

Everything learned sits under `seiza model`, running and managing alike, so
it reads apart from the classical commands: `seiza deconvolve` beside
`seiza model sharpen`. `seiza process` fits less well, because `stretch`,
`background` and `deconvolve` already process images at the top level and
`seiza pipeline` already exists.

`--high-amount` and `--low-amount` are the harness's names, so sheets and the
CLI speak alike. `seiza model sharpen` takes `--stars` and `--rest`, the two
shares, each the model's own when left out, `--rolloff`, `--hold-noise`, and,
for a card with a scaled pass, `--binned-rest` and `--full-size-only`. Its C twin is
`seiza_sharpen_model_sharpen`, which writes into the caller's buffer and
answers with what ran, and its Python twin `SharpenModel.sharpen`. From a
file to a file, as the CLI runs it, they are `RestorationImage::sharpen` and
`FileSharpening::write_linear`, `seiza_sharpen_model_sharpen_file` and
`SharpenModel.sharpen_file`. Output is 32-bit FITS, written through `seiza-image-io`.

**C ABI.** The apps hold a model and a result, and run files as one call:

```c
SeizaNoiseModel *seiza_noise_model_open(const char *bundle, const char *backend, char **error_out);
SeizaDenoised *seiza_noise_model_denoise(SeizaNoiseModel *, const float *image, size_t width,
    size_t height, size_t channels, const char *options_json,
    SeizaRestoreProgressCallback, const SeizaCancelSignal *, void *context, char **error_out);
bool seiza_denoised_blend(const SeizaDenoised *, float high, float low, float scale_px, float *out, char **error_out);
char *seiza_noise_model_denoise_file_json(const char *request_json, /* progress, cancel */ ...);
char *seiza_model_backends_json(char **error_out);
const char *seiza_model_output_version(void);
```

The JSON file call is only a request and answer over the native API, as
[rule 2](../../AGENTS.md#the-rust-api-comes-first) allows.

**Python.** It wraps the native types, never the JSON call.

```python
model = seiza.NoiseModel("noise-v1", backend="auto")
result = model.denoise(image, progress=report, cancel=flag)
clean = result.blend(high=0.7, low=0.9, scale_px=5.0)
seiza.model_backends()
```

The GIL is released for the run. `NoiseModel` and `Denoised` join
`[core_functions] objects` in `.github/binding-parity.toml`, and
`OUTPUT_VERSION` joins the rules in [AGENTS.md](../../AGENTS.md#output-versions-revise-and-test-every-time).

Every restored file records the model hash, backend, precision, tile and
options. The backend recorded is the one requested: Core ML can place work
on another unit without saying so, and an f32 graph on the Neural Engine
setting does exactly that. Hosts key cached results on `OUTPUT_VERSION`, the
model hash, the backend and the precision, as the restoration plan says. Tile
size stays out of the key, since it does not change the output.

## Getting the runtime onto machines

`load-dynamic` means no build links ONNX Runtime, so crates.io builds and the
Rust 1.92 check need nothing native, and `seiza.exe` keeps its static CRT. Each
channel supplies the library its own way.

The packages carry Microsoft's own build, pinned once in
`packaging/onnxruntime.env`, which CI reads too. Microsoft's GitHub repository
has returned 404 since about 2026-11-18, so the archives come first from PSF
Guard's release `onnxruntime-1.31.1`, which holds the same files byte for byte,
and from Microsoft second, each checked against the pinned hash. Its archives
are mostly debug symbols, and the libraries themselves are small:

| Archive | Library | Compressed | Notes |
|---|---|---|---|
| macOS arm64 | 44.8 MB | 12.6 MB | Reports CPU, Core ML and WebGPU, and ran every Rust figure above. Ad-hoc signed, so a signed app re-signs it. |
| Linux x86-64 | 29.3 MB | 11.2 MB | Asks for glibc 2.28 and GLIBCXX 3.4.22, older than our Ubuntu 22.04 build. |
| Windows x86-64 | 17.3 MB | 6.3 MB | Carries an Authenticode signature. Imports `MSVCP140` and `VCRUNTIME140`. |

The Windows library imports `MSVCP140`, `MSVCP140_1`, `VCRUNTIME140` and
`VCRUNTIME140_1`, the Visual C++ runtime that `seiza.exe` avoids, so without
them it will not load on a clean machine. The zip and the MSI carry those four
beside the binary, taken from the build machine's redistributable, since a zip
has no installer to run one, and Windows looks beside the executable first.
`packaging/windows/stage-vc-runtime.ps1` reads the imports and stops the build
if they outgrow that list. CI's `model-runtime` job runs a tiny model on all
three systems, and the `Packaged runtime` workflow installs each package on a
machine with nothing else on it, a Windows container among them. A package
ships ONNX Runtime's `LICENSE` and `ThirdPartyNotices.txt`.

**Telemetry.** Microsoft's own builds report to Microsoft on every system,
unless told not to. Each event held the process's name, the model's file name,
the machine, the system and a device identifier, in a store under
`Microsoft/DeveloperTools/.onnxruntime` that an uploader sends on. `seiza-model`
turns this off, since nobody who runs Seiza on their own images has agreed to
it. ONNX Runtime's API is not enough: it stops only non-essential events, and
an event may go out before it can be called. `ORT_DISABLE_TELEMETRY=1`, read as
the library starts, stops the uploader and the identifier being made at all,
so `seiza-model` sets it before loading the library unless the variable is
already set, and calls the API as well for a process that loaded ONNX Runtime
first. A run on a Mac with the variable left the store untouched, and without
it added 24 KB.

| Channel | Library | Notes |
|---|---|---|
| Homebrew | `depends_on "onnxruntime"` | Built with Core ML on macOS; 1.31.1 today. The formula lives in the tap, and gains this with the first release carrying `seiza-model`. |
| Python wheel | `pip install seiza[ml]` brings `onnxruntime` | Seiza finds the package's library, under `onnxruntime/capi`, without importing it. Wheels exist for Python 3.11 to 3.14. NVIDIA users take `onnxruntime-gpu`. Our wheels stay abi3 and unlinked. |
| Mac app | signed dylib in the bundle | Or `ort`'s static macOS build, if the app prefers one binary. |
| Windows CLI zip and MSI | `onnxruntime.dll` beside the binary, with the Visual C++ runtime | CPU in the default archive. An explicit DirectML-enabled runtime adds GPU execution. |
| Windows desktop app | The DirectML-enabled ONNX Runtime bundled by Windows App SDK | DirectML or CPU. |
| NVIDIA | a separate download | CUDA and TensorRT are too large for the installers, and their licences need reading before Seiza redistributes them. |
| deb and rpm | private copy in `/usr/lib/seiza` | So Seiza does not depend on what each distribution carries. The search looks in `../lib/seiza` from the binary. |
| `cargo install` | `seiza setup` fetches it | Into the shared data directory, verified through `seiza-download`, which the search covers. Not built yet. |
| PSF Guard | its own choice, through `BackendOptions` | It links `seiza-model` like any crate. |

`ort`'s prebuilt list has no Intel Mac build, and our macOS wheel is
universal2. An Intel Mac gets `BackendUnavailable` unless a library is found.

### Model bundles

A bundle holds `model-f32.onnx`, `model-f16.onnx` and `card.json`, verified
and installed through `seiza-download`, as catalogues are, under
`/models/v1/`. A host can ship a bundle instead.

`/models/v1/manifest.json` lists the published models, each with the files
that install it and the extras hosted beside it, in the form a
[catalog bundle](catalog-bundles.md)'s files and transports take:

```json
{
  "version": "model-bundle-v1-2026-11-10",
  "latest": { "denoise": "noise-v1-d33" },
  "models": [
    {
      "name": "noise-v1-d33",
      "files": [
        { "name": "card.json", "key": "artifacts/<sha256>/card.json", "bytes": 995, "sha256": "<sha256>" },
        { "name": "model-f32.onnx", "key": "artifacts/<sha256>/model-f32.onnx", "bytes": 373285843, "sha256": "<sha256>" },
        { "name": "model-f16.onnx", "key": "artifacts/<sha256>/model-f16.onnx", "bytes": 186650316, "sha256": "<sha256>" }
      ],
      "extras": [
        { "name": "weights.pt", "key": "artifacts/<sha256>/weights.pt", "bytes": 373288961, "sha256": "<sha256>" }
      ],
      "transports": [
        { "name": "model-f32.onnx", "encoding": "zstd", "key": "artifacts/<encoded-sha256>/model-f32.onnx.zst", "bytes": 346396783, "sha256": "<encoded-sha256>" }
      ]
    }
  ]
}
```

A model's name is its card's and its directory once installed. Its `files`
are `card.json` and the graphs the card names; `extras`, here the PyTorch
weights the graphs were exported from, are never fetched by an install. Every
key is content-addressed, so no published artifact is replaced, and the
zstd transports are verified before and after decoding, as a catalog's are.

`latest` names, for each task as cards spell it, the model to install when
none is named: the newest that has passed its gates. It is a pointer, as the
catalogs' `manifest.json` is to their current bundle, rather than an order
among `models` or a date to compare, so publishing a model never makes it
the default by accident, and taking one back is a one-line change. Each name
it holds must be offered, and the install refuses a card whose task is not
the one it was named for. A manifest without it still reads; it then names
no default.

`ModelStore::download` installs a model by name (`Published::Named`) or the
latest for a task (`Published::Latest`). The CLI's `seiza model install
[NAME]`, `seiza setup --noise-model`, the C ABI's `seiza_model_download_json`
(a null name for the latest) and Python's `download_model` (no name for the
latest) all call it. It fetches the files into the download cache and installs
them whole. It refuses a card that names another model or differs from its
manifest entry, and graphs that differ from the card. The store hard-links the
cached files where it can, so a model takes its space once. Each step is
reported as a `DownloadEvent`, which the C ABI hands its callback as JSON and
Python its `progress` as a dict, its kind under `event`. An install fetches
both precisions for now, about 520 MB for a model of 93 million weights;
fetching only the one a backend needs is still to come.

`seiza model denoise` with no `--model` runs the one installed noise model
and touches no network. With none installed, it fetches the manifest from the
server, installs the latest noise model with its progress shown, and runs.
If the manifest cannot be reached, it stops with an error that names
`seiza model install`; it never falls back to a cached manifest or another
model. A newer published model is not fetched over an installed one: `seiza
model install` brings it, and while more than one is installed `--model`
chooses. A model named with `--model` always wins and is never fetched.

Publishing follows the catalogs' order: upload each artifact, uncompressed
and as zstd, to its content-addressed key with an immutable cache lifetime;
check every hosted object's size and hash, encoded and decoded; keep the
manifest under `/models/v1/manifests/<version>.json`; and write
`/models/v1/manifest.json`, with a short cache lifetime, last.

The card is written from a list of allowed fields, never by copying
`run.json`, which holds machine paths and the training configuration in
full. It gives the contract, the graph's axes, the grid and reach, the tile
defaults, the trained ranges, each artefact's hash, and which backends the
model has passed parity on. A build that cannot honour a card's contract
refuses with `ModelContract`.

A bundle runs only after its hashes match. `--model PATH` is the one way to
run an unlisted file, and the user asks for it by name.

### Model shapes

Nothing in the runtime knows a model's shape. The card says it, and the graph
is held to the card when it opens.

| What can change | Where it comes from |
|---|---|
| Depth and width | The graph. Reach, ramp and tile multiple are the card's `tiling`. |
| Which planes go in, in what order | `inputs`: each tensor by name, with its planes. `["image"]` is a model with no noise plane. |
| How many tensors | `inputs` again, so `image` and `noise` may be two tensors or one. |
| Tile size | `dimensions` names the free axes. None means a graph fixed at `tiling.tile`, and any other tile is refused. |
| Tensor types | The graph itself, for each tensor, whatever the file's precision. An f16 file may read and write f32 at either end. |

A graph that differs from its card is refused at open, with what differs:
planes, tile, an unknown or unnamed tensor, a type that is neither f32 nor
f16. Run on, such a graph would at best stop at its first tile, and at worst
keep part of what the network wrote. The small graphs in
`seiza-model/tests/fixtures/shapes` hold each of these, in CI, against answers
worked out by hand.

A new kind of input is one `InputPlane` variant and one arm of `network_input`.
A build that does not know a plane refuses the card and names the plane. The
noise reading, as opposed to the shape, is `reading`: a change to how the noise
level is read needs a field there, so that a card can say which rule its model
expects.

Three things stay fixed until a model needs them, because each depends on what
that model's planes mean, which cannot be known first:

- **A graph writes one plane.** A colour model, or one that writes stars and
  the rest apart, is refused.
- **A model reads the image once.** So a colour image still runs channel by
  channel, each with scales of its own.
- **`task` is `denoise`.** Star and PSF models are new tasks, with a run of
  their own, on the same session and tiling.

## Testing

- **Fixtures.** The bundle step writes a small model, a few hundred
  kilobytes, with input tiles and a whole image through tiling. It is
  committed, and it drives CI on all three systems.
- **Pure Rust parity.** Sky, `image_sigma`, scales, window weights and bands match
  the harness's outputs on fixture planes.
- **`output_version`.** A case on the fixture model on ONNX Runtime's CPU
  provider, under the rules in AGENTS.md, with a planted change seen to fail.
  CPU kernels differ between x86 and ARM, so its tolerance must be set from
  all three CI systems.
- **Backend parity, off CI.** GitHub's macOS runners have no Neural Engine or
  usable GPU, so the restoration plan's "Core ML parity runs on macOS CI"
  would only test the CPU. Real parity runs on a Mac and on the NVIDIA machine
  before each model release, and its results go in the card.
- **The harness as a host.** The strongest test is the harness itself:
  `evaluate-noise` and the contact sheets run through `seiza.NoiseModel`, so
  the gates, seam maps and sheets judge the code that ships. Then `tiles.py`
  and the Rust tiling cannot drift apart unnoticed.

## Work order

Each step is one PR unless it says otherwise. Where a step adds a public
function, its C export and Python twin land in the same PR, as
[AGENTS.md](../../AGENTS.md#the-core-the-c-abi-and-python-move-together)
requires, so earlier steps keep their items `pub(crate)` until step 4.

Steps 0 to 4 are built, with model delivery's local commands: `seiza-model`,
`seiza model`, its C ABI and Python, and `bundle-noise` in `training/`. It
matches the harness on a real image on a Mac, and a tiny model runs through ONNX
Runtime in CI. The output bound is still not built, and no shipped channel
carries the library yet. Step 6 is built: `seiza model install [NAME]`,
`/models/v1/`, where `noise-v1-d33` is published as the latest noise model,
`seiza setup --noise-model`, and a first `seiza model denoise` that fetches it.

| # | Step | Done when |
|---|---|---|
| 0 | **Spike.** A throwaway Rust binary on `ort` rc.15 with `load-dynamic`: the CPU provider on macOS, Windows and Linux, and Core ML on a Mac, with parity. Repeat the GPU rows on an idle Mac. | A tile runs from Rust on all three systems, and the Mac's numbers hold from Rust. |
| 1 | **Bundle export** in `training/`. Dynamic f32 and f16 graphs, `card.json` from the allow-list, fixtures, the output bound in the harness. | `bundle --run NAME` makes a bundle that passes a parity check in Python. |
| 2 | **`seiza-model` core**, no ONNX Runtime. Tiling, sky, `image_sigma`, scales, bands, the bound, `ModelCard`, `OUTPUT_VERSION`. | Rust matches the harness on fixtures through a closure runner. |
| 3 | **Backends.** `TileRunner` on `ort`, detection, library search, compile cache, cancel. `seiza model backends`, with its C and Python twins. | Fixture parity on the CPU provider in CI on all three systems; Core ML on a Mac. |
| 4 | **`denoise` end to end.** `NoiseModel`, `Denoised`, CLI, C ABI, Python, parity entries, provenance. Marked experimental. | `seiza model denoise` matches the harness on a real image within budget, on every backend measured so far. |
| 5 | **Harness on the runtime.** `evaluate-noise` and the sheets through `seiza.NoiseModel`. | One run's measures and seam maps agree with the PyTorch path. |
| 6 | **Model delivery.** `seiza model install`, `/models/v1/`, `seiza setup`. | A clean machine installs a model and denoises. |
| 7 | **Packaging**, one PR per channel, each with a smoke test on an installed package. | Each row of the table above runs a tile after install. |
| 8 | **PSF and star models**, as they pass their gates, with what they read made from the image: a PSF stamp from its PSF field, star and halo maps from its plate solution. The sharpening runtime is built (see Sharpening): `SharpenModel`, `seiza model sharpen` and its C and Python twins. | `seiza model sharpen` and `seiza model separate-stars` on the same runner. |
| 9 | **NVIDIA and Intel**, when the machines are arranged. Measure CUDA, TensorRT, TensorRT-RTX and CPU on NVIDIA, and WebGPU and CPU on Intel Xe, with parity. | The backend table is filled in and NVIDIA's shipped provider is chosen. |

Step 0 gates the rest only where it says so. Steps 1 and 2 can start at once
and run side by side. Nothing here waits for the choice of noise input.
Parity compares code, not models, so it can run D22 with a card that says
`constant`, as `evaluate-noise --noise-map constant` does. Only a release
needs a model that has passed its gates on an image-only input.

Step 9 can come any time after step 3. Until then NVIDIA and Intel machines
run the CPU provider, and step 7's NVIDIA download waits for it.

## Risks

- **`ort` is a release candidate.** Its API has changed between candidates.
  The pin holds it still, and only `session.rs` and `backend.rs` touch it.
- **The Neural Engine's error may differ by generation.** Only the M4's is
  measured. The GPU is the default, so this reaches only a run that asks for
  the Neural Engine, or a card that leaves the GPU out. The GPU's own error
  is measured on the same chip, at a 25th of its budget.
- **Core ML can fall back silently.** The run still succeeds, only slower,
  and the provenance names the unit requested.
- **ONNX Runtime versions.** `ort` rc.15 asks for at least 1.17. It has run
  against 1.31.1 from pip and from Microsoft's archive, and not yet against
  Homebrew's.
- **GPU memory.** One 96-channel f32 plane at tile 1024 is 0.4 GB, so a
  tile's activations should come to a few gigabytes; steps 0 and 9 measure it.
  `BackendOptions` takes a tile size, and an allocation failure
  retries at 512 before it refuses.
- **Stacking and inference share the GPU.** A host running both should run
  them in turn. Nothing here shares a device between wgpu and ONNX Runtime.
