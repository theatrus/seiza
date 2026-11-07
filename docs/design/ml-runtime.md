# Learned model runtime

Status: proposed. Nothing here is built yet. The measurements are from
Python through ONNX Runtime and Core ML, on one Mac; a Rust spike and the
NVIDIA and Intel measurements come first (see [Work order](#work-order)).

## Short answer

The training harness runs our models in Python. This plan moves that into a
new crate, `seiza-restore`, reached from the CLI, the C ABI and Python.

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

A 61 MP frame should take about 15 seconds on an M4 Max's Neural Engine with
the current 41M-parameter model. That is an estimate from a measured 26 MP
run.

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

Only mono models exist. RGB runs a mono model on each channel until an RGB
model is trained. PSF and star models need several inputs, so the runtime
takes named tensors from the start, not one image in and one image out.

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

**Not yet measured:** anything on NVIDIA or Intel; anything from Rust; any
Neural Engine older than the M4's; a loaded-versus-idle GPU comparison.

## Decisions proposed

1. **One binding: `ort`, with `load-dynamic`.** Drop `objc2-core-ml`, the
   `.mlpackage` export and the `coreml` feature. Native Core ML stays
   available later, as a second `TileRunner`, if a host cannot carry a
   33 MB library. This keeps the decision that Core ML runs models on Apple
   hardware and changes only how Seiza reaches it.
2. **Two graphs per model, f32 and f16, with dynamic height and width.**
   A host fetches the one its backend needs. Safetensors weights stay in the
   store, because nothing at run time reads them.
3. **Tile size belongs to the backend, not the export.** Default to 1024 on
   a GPU or the Neural Engine, and 512 where memory is short. Whole-image
   parity already holds across sizes.
4. **Change the Neural Engine's parity budget.** The restoration plan
   allows 0.01σ for every accelerated backend. The GPU passes at f16 and the
   Neural Engine does not, at 0.0143σ̄ on the sky. Proposed for the Neural
   Engine: sky RMS under 0.02σ̄, mean under 0.002σ̄, seams as for any
   backend. With 15% of the noise kept, an error that size adds about 0.5%
   to the noise left, in quadrature. A model still earns the Neural Engine by
   passing its own parity run.
5. **Pin `ort` to `=2.0.0-rc.13`.** It is the newest release that fits our
   Rust 1.89 floor; rc.15 needs 1.92. There is no stable 2.0 yet. Dependabot
   needs an ignore rule, or the floor moves.

## The crate

`seiza-restore` depends on neither `seiza-stacking` nor `seiza-stars`
for the noise model. It takes a borrowed image and makes everything else a
model reads from it.

```text
seiza-restore/src/
  lib.rs       NoiseModel, Denoised, RestoreError, OUTPUT_VERSION
  backend.rs   Backend, BackendOptions, detection, library search
  session.rs   TileRunner, the ONNX Runtime session, the compile cache
  bundle.rs    ModelBundle, ModelCard, verification
  tiles.rs     tile starts, window weights, streaming blend
  noise.rs     the noise input: image_sigma, or its median, as the card says
  sky.rs       smooth sky, background and noise scales, the network's input
  bands.rs     NoiseBands
```

`TileRunner` is the seam between Seiza and any runtime:

```rust
pub trait TileRunner {
    /// Run one tile of named input tensors, filling the named outputs.
    fn run(&mut self, inputs: &[Tensor<'_>], outputs: &mut [TensorMut<'_>]) -> Result<(), RestoreError>;
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
| Apple Silicon | Neural Engine, GPU, CPU | f16, then f32 for CPU |
| NVIDIA | decided by the measurements; see [Work order](#work-order) | f16 or f32 |
| Intel and AMD GPUs | WebGPU if measured worthwhile, else CPU | f32, or f16 if parity holds |
| Anything else | CPU | f32 |

DirectML is left out. Microsoft keeps it in maintenance, with new work going
to Windows ML. ONNX Runtime's WebGPU provider is still marked experimental,
but it is the one provider that would cover Intel Xe, AMD and NVIDIA on
Windows and Linux, which matches what the [stacker](gpu-stacking.md) does
through wgpu. `ort`'s prebuilt list has it for both.

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
  Second differences read correlated noise low, and each stacker's resampling
  correlates it differently, so the level may differ on other stacks. The
  gates need other targets and stacks from other software.

A variant that reads Seiza's own records may sit beside the image-only model,
for Seiza's stacks. Nothing here builds it. If one earns its place,
`modelled_sigma` needs only its block grid and one gain, about 25,000 numbers
for a 26 MP frame, which fits a small FITS extension.

## What hosts see

Following the [Rust-first rules](../../AGENTS.md#the-rust-api-comes-first).
The restoration plan's [native API](ml-restoration-training.md#native-api)
stands, with `Backend` as above.

**Rust.** PSF Guard calls these directly.

```rust
let backend = Backend::detect(&BackendOptions::default())?;
let model = NoiseModel::open(&bundle, &backend)?;
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
seiza model list | install <name> | verify | remove <name>
seiza model backends      # which backends run here, and why the others do not
```

Everything learned sits under `seiza model`, running and managing alike, so
it reads apart from the classical commands: `seiza deconvolve` beside
`seiza model sharpen`. `seiza process` fits less well, because `stretch`,
`background` and `deconvolve` already process images at the top level and
`seiza pipeline` already exists.

`--high-amount` and `--low-amount` are the harness's names, so sheets and the
CLI speak alike. Output is 32-bit FITS, written through `seiza-image-io`.

**C ABI.** The apps hold a model and a result, and run files as one call:

```c
SeizaNoiseModel *seiza_noise_model_open(const char *bundle, const char *backend, char **error_out);
SeizaDenoised *seiza_noise_model_denoise(SeizaNoiseModel *, const float *image, size_t width,
    size_t height, size_t channels, const char *options_json,
    SeizaRestoreProgressCallback, const SeizaCancelSignal *, void *context, char **error_out);
bool seiza_denoised_blend(const SeizaDenoised *, float high, float low, float scale_px, float *out, char **error_out);
char *seiza_noise_model_denoise_file_json(const char *request_json, /* progress, cancel */ ...);
char *seiza_restore_backends_json(char **error_out);
const char *seiza_restore_output_version(void);
```

The JSON file call is only a request and answer over the native API, as
[rule 2](../../AGENTS.md#the-rust-api-comes-first) allows.

**Python.** It wraps the native types, never the JSON call.

```python
model = seiza.NoiseModel("noise-v1", backend="auto")
result = model.denoise(image, progress=report, cancel=flag)
clean = result.blend(high=0.7, low=0.9, scale_px=5.0)
seiza.restore_backends()
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
Rust 1.89 check need nothing native, and the Windows build keeps its static
CRT. Each channel supplies the library its own way.

| Channel | Library | Notes |
|---|---|---|
| Homebrew | `depends_on "onnxruntime"` | Built with Core ML on macOS; 1.31.1 today, against the 1.28 `ort` wraps, so the spike must check this. |
| Python wheel | an `ml` extra needing `onnxruntime` | The macOS wheel carries `libonnxruntime.1.31.1.dylib`, 33 MB, with Core ML. NVIDIA users take `onnxruntime-gpu`. Our wheels stay abi3 and unlinked. |
| Mac app | signed dylib in the bundle | Or `ort`'s static macOS build, if the app prefers one binary. |
| Windows MSI and app | `onnxruntime.dll` beside the binary | CPU, plus WebGPU if measured worthwhile. |
| NVIDIA | a separate download | CUDA and TensorRT are too large for the installers, and their licences need reading before Seiza redistributes them. |
| deb and rpm | private copy under Seiza's lib directory | So Seiza does not depend on what each distribution carries. |
| PSF Guard | its own choice, through `BackendOptions` | It links `seiza-restore` like any crate. |

`ort`'s prebuilt list has no Intel Mac build, and our macOS wheel is
universal2. An Intel Mac gets `BackendUnavailable` unless a library is found.

### Model bundles

A bundle holds `model-f32.onnx`, `model-f16.onnx` and `card.json`, verified
and installed through `seiza-download`, as catalogues are, under a new
`/models/v1/` path. `seiza model install` fetches only the precision the
backend needs. A host can ship a bundle instead.

The card is written from a list of allowed fields, never by copying
`run.json`, which holds machine paths and the training configuration in
full. It gives the contract, the graph's axes, the grid and reach, the tile
defaults, the trained ranges, each artefact's hash, and which backends the
model has passed parity on. A build that cannot honour a card's contract
refuses with `ModelContract`.

A bundle runs only after its hashes match. `--model PATH` is the one way to
run an unlisted file, and the user asks for it by name.

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

| # | Step | Done when |
|---|---|---|
| 0 | **Spike and measure.** A throwaway Rust binary on `ort` rc.13 with `load-dynamic`, run on macOS, Windows and Linux. Measure CUDA, TensorRT, TensorRT-RTX and CPU on NVIDIA, and WebGPU and CPU on Intel Xe, with parity. Repeat the GPU rows on an idle Mac. | The backend table is filled in and NVIDIA's shipped provider is chosen. |
| 1 | **Bundle export** in `training/`. Dynamic f32 and f16 graphs, `card.json` from the allow-list, fixtures, the output bound in the harness. | `bundle --run NAME` makes a bundle that passes a parity check in Python. |
| 2 | **`seiza-restore` core**, no ONNX Runtime. Tiling, sky, `image_sigma`, scales, bands, the bound, `ModelCard`, `OUTPUT_VERSION`. | Rust matches the harness on fixtures through a closure runner. |
| 3 | **Backends.** `TileRunner` on `ort`, detection, library search, compile cache, cancel. `seiza model backends`, with its C and Python twins. | Fixture parity on the CPU provider in CI on all three systems; Core ML on a Mac. |
| 4 | **`denoise` end to end.** `NoiseModel`, `Denoised`, CLI, C ABI, Python, parity entries, provenance. Marked experimental. | `seiza model denoise` matches the harness on a real image within budget, on every backend from step 0. |
| 5 | **Harness on the runtime.** `evaluate-noise` and the sheets through `seiza.NoiseModel`. | One run's measures and seam maps agree with the PyTorch path. |
| 6 | **Model delivery.** `seiza model install`, `/models/v1/`, `seiza setup`. | A clean machine installs a model and denoises. |
| 7 | **Packaging**, one PR per channel, each with a smoke test on an installed package. | Each row of the table above runs a tile after install. |
| 8 | **PSF and star models**, as they pass their gates, with what they read made from the image: a PSF stamp from its PSF field, star and halo maps from its plate solution. | `seiza model sharpen` and `seiza model separate-stars` on the same runner. |

Step 0 gates the rest only where it says so. Steps 1 and 2 can start at once
and run side by side. Nothing here waits for the choice of noise input.
Parity compares code, not models, so it can run D22 with a card that says
`constant`, as `evaluate-noise --noise-map constant` does. Only a release
needs a model that has passed its gates on an image-only input.

## Risks

- **`ort` is a release candidate.** Its API has changed between candidates.
  The pin holds it still, and only `session.rs` and `backend.rs` touch it.
- **The Neural Engine's error may differ by generation.** Only the M4's is
  measured. Parity runs on what we own, and a card can leave the Neural
  Engine out, so an unmeasured chip costs speed, not correctness, if `detect`
  falls back to the GPU on chips the card does not list.
- **Core ML can fall back silently.** The run still succeeds, only slower,
  and the provenance names the unit requested.
- **ONNX Runtime versions.** Homebrew and pip are three releases ahead of
  what `ort` rc.13 wraps. The C API is meant to stay compatible; the spike
  checks it.
- **GPU memory.** One 96-channel f32 plane at tile 1024 is 0.4 GB, so a
  tile's activations should come to a few gigabytes; step 0 measures it.
  `BackendOptions` takes a tile size, and an allocation failure
  retries at 512 before it refuses.
- **Stacking and inference share the GPU.** A host running both should run
  them in turn. Nothing here shares a device between wgpu and ONNX Runtime.
