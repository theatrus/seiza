# Distributed stacking

Status: phase 1 built; paused before phase 2 (see **Phases**)

## Goal

Stack one target, or one mosaic, on many machines and over many nights, and
keep the result current as new frames arrive. Seiza owns every step, from raw
calibration frames to the assembled mosaic. Work runs on the machine that holds
the frames, and only small results that describe themselves are moved. After
a restart, an extension or a new contributor, only the steps whose inputs
changed run again. The results' own header cards show which steps those are.

The same model serves three settings:

- one person's machines, coordinated by PSF Guard;
- any scheduler that can run a command;
- a casual collaboration that runs for months or years. People drop partial
  stacks into a shared folder, and anyone can build the result from it. No
  server has to stay up, and nobody has to keep their frames.

How it works today:

- One process stacks a channel. `LiveStacker` registers every frame to one
  reference, or to an expanded canvas planned from every frame on that
  machine. Reintegration then rejects samples across all of the frames.
- The written stack is the mean image only. `StackSnapshot` holds variance,
  coverage and rejection counts, and the accumulator holds a weight sum, but
  none of them reach the file.
- PSF Guard's remote workers take a whole channel per task. Masters are built
  on the server. No task splits a stack, and nothing merges two.
- PSF Guard resumes a stack only when the new frame list extends the old one
  at the end. A manifest written beside the live-stack context proves this.

## Move statistics, not pictures

A weighted mean is a ratio of two sums. Suppose each piece of work returns
Σwx and Σw for every pixel, plus the measurements that put it on a common
scale. Combining those pieces then gives the same mean as one stack of all
the frames. A stack written as a mean image alone has lost Σw. The combiner
must then guess one weight per file, and it gets coverage edges, rejected
trails and vignetted corners wrong.

Rejection is the part that does not add up across pieces. Leave-one-out
clipping needs every sample at a pixel at once. But what rejection removes —
satellites, aircraft, cosmic rays, one bad frame — belongs to one moment.
Rejecting within a session, where there are dozens of frames, removes them.
Across sessions, the remaining job is to catch a fault that affects a whole
session, such as a bad flat or a misregistration. That check can use the
variance each session reports, rather than estimate it from three or four
samples.

## Terms

- **Session**: frames of one target taken through one optical train and filter
  and calibrated by one recipe, usually in one night. In other words, the
  lights one `FileCalibrationSession` calibrates.
- **Grid**: a TAN projection with a pixel scale, cut into tiles of a fixed
  size. A grid is a value with a definition, and that definition's hash
  identifies it. A grid never changes once made: a session outside it adds
  tiles, but never moves the projection. A project grid is fitted to one
  project's first footprints. A sky grid is one cell of a fixed tiling of
  the whole sky.
- **Project**: a grid, the bands it stacks, its combine options, and its
  decisions about which partial stacks to include. A store holds any number
  of projects, and a session can belong to several of them.
- **Partial stack**: one session's frames integrated on one tile. It holds the
  mean, weight and coverage over the window of the tile the session covers,
  plus measurements and header cards. It is the unit that machines and
  people exchange, and the unit a run resumes from.
- **Tile stack**: every partial stack on one tile, combined.
- **Mosaic**: a grid's tile stacks assembled into one image.
- **Store**: where results live, each under the key of the step that made
  it. A directory, a synced folder or a bucket.

## Pipeline

| Stage | Where it runs | Reads | Writes | Built on |
|---|---|---|---|---|
| 1. Masters | Machine holding the calibration frames | Raw bias, dark, flat | Master files | `build_master_from_fits`, the calibration library |
| 2. Register | Machine holding the session's lights | Lights, masters | Live-stack context and footprint | `LiveStacker`, SIP solving |
| 3. Grid | Any machine, once per project | The first sessions' footprints | Grid definition | New |
| 4. Replay and measure | Machine holding the session's lights | Lights, register output, grid | One partial stack per tile the session reaches | `reintegrate`, `snr.rs`, `psf_fitting`, `color_calibration` |
| 5. Background | Any machine | Sky cells, the project's baseline | One plane per partial stack | `batch/mosaic.rs` plane fit |
| 6. Tile stack | Any machine, one task per tile | A tile's partial stacks and their planes | Tile stack | New combiner |
| 7. Assemble | Any machine | A grid's tile stacks | Mosaic file with WCS and credits | `seiza-image-io` writers |

Stages 1, 2 and 4 need the frames, so they run where the frames are. Nothing
after stage 4 does. That is the line a collaboration crosses: a contributor
runs stages 1 to 4 on their own machine and shares the partial stacks.

### 2. Register

The live pass runs as it does today. It calibrates, detects stars, and
registers each frame to the session's reference, then normalizes, weights and
accumulates. That gives a deep image on the reference grid. Solving it with
SIP, using the frame headers as the hint, maps the reference to the sky and
gives the session's footprint.

Nothing here depends on a grid, so every project that includes the session
reuses it. The live-stack context stays with the frames. While a session is
still being shot, new frames are appended to it, as PSF Guard appends them
today.

### 3. Grid

A project grid is chosen from the footprints of the sessions registered when
the project starts, or written into the project by hand. It is one TAN
projection, one pixel scale and one tile size, and it is then fixed. A later
session that falls outside it adds tiles. It never moves the projection, so
tiles already stacked keep their keys. A sky grid needs no step: its tiles
follow from its definition.

### 4. Replay and measure

Reintegration already reads every frame again and maps it through the
transform recorded in the ledger. Here it maps each frame through three
steps: frame to reference, reference to sky, and sky to tile. So each frame is
resampled once, straight onto every tile it touches, and leave-one-out
rejection runs per tile within the session. A session's footprint covers a
few tiles, and the tile size bounds the memory each accumulator needs. Each
partial stack keeps only the window of its tile that the session covers, so
a session's partial stacks together are about the size of its footprint.

Distortion comes from the session's SIP solve, not from a lens model fitted
to the overlaps of one canvas (`lens.rs`). Before per-session placement
replaces the expanded canvas, it has to pair stars across real mosaic
overlaps as tightly as the canvas does. `lens.rs` records 0.15 px.

Measurement runs while the frames and partial stacks are still on the
machine.

- **Weight.** The weight of a pixel is the inverse variance of its mean. It
  has two factors, and both are needed:
  - The accumulator's per-sample frame-weight sum, W. It has sharp structure:
    dither edges, holes left by rejected trails, and the edges of the canvas.
  - The noise of a frame of weight 1, σ₁. It varies smoothly: corners
    brightened by the flat are noisier, and so is a sky gradient. It is
    measured on a mesh of cells over the stack with the second-difference
    estimator `snr.rs` uses, converted per cell using that cell's W, and
    interpolated.

  The stored weight is W / σ₁². With scalar frame weights alone, vignetting
  does not show at all.
- **Photometric zero point.** It comes from aperture photometry of Gaia stars,
  with a colour term fitted in BP − RP, as `color_calibration` already does
  for RGB. A broadband partial stack then needs no other file to reach a
  common scale. Narrowband is harder; see the open questions.
- **PSF.** A Moffat fit per cell gives a grid of PSF models. `psf_fitting`
  reads `u16` samples today and needs an `f32` path.
- **Sky cells.** These are star-masked background means and coverage on a
  coarse grid: the same cells `batch/mosaic.rs` fits planes through. They
  are what stage 5 reads instead of pixels.
- **Band.** Each channel is assigned one of the project's bands, from the
  frames' `FILTER` card and the project's list of names for each band.

### 5. Background

Today the mosaic background fit reads every input once more before its
passes. Here it reads only the sky cells of every partial stack on the grid.
That is a few megabytes, not the frames. It solves one additive plane per
partial stack so that overlapping cells agree, then tapers what is left, as
`batch/mosaic.rs` does ([registered mosaic backgrounds](mosaic-backgrounds.md)).
Sessions of the same field overlap completely, so the same solve also takes
out night-to-night differences in sky glow. The result is a small file of its
own. The partial stacks are never rewritten.

A full solve moves every plane a little whenever one partial stack joins, and
then every tile has to be stacked again. So a project keeps a baseline:

- The baseline is a full solve over the partial stacks the project held when
  it was made. The project records its key.
- Partial stacks added later are solved together, with the baseline's planes
  held fixed. Only the tiles that hold a later partial stack are stacked
  again.
- A new baseline is made on purpose, when someone asks for one. Every tile
  is then stacked again once.

Both solves are pure functions of their inputs. Anyone who runs them on the
same partial stacks and baseline gets the same planes, whatever order the
partial stacks arrived in.

### 6. Tile stack

For each pixel, partial stack j contributes a value xⱼ, already scaled by its
zero point and corrected by its plane, and a weight wⱼ. The tile stack is
Σwⱼxⱼ / Σwⱼ, and its weight is Σwⱼ.

Weighting modes:

- **Inverse variance** (the default) is best for faint, background-limited
  signal. It pays no attention to resolution.
- **Point source** multiplies each weight by 1 / FWHMⱼ² from that partial
  stack's PSF. A session with poor seeing then counts for less.
- **Proper coaddition** ([Zackay & Ofek 2017][zo]) combines the tile in
  Fourier space, from each partial stack's pixels, zero point, noise and PSF.
  Its result is optimal for any measurement when the background dominates
  the noise. It assumes white noise, a PSF that does not change across the
  tile, and full coverage. Resampling correlates noise, and mosaic edges
  break the coverage assumption. So it waits until the first two modes are
  proven, and it applies only to tiles that every partial stack covers.

Because a tile stack is a sum, leaving one partial stack out is exact and cheap:
remove its term from both sums. Each partial stack gets a consistency score
against all the others. The score measures background offset, gradient, and
the dipoles a misregistration leaves around stars. A partial stack that fails
is flagged and left out of the tile stack. Its pixels are never quietly clipped.

Rejecting single pixels across partial stacks is off by default. Two sessions
with different seeing disagree at every star core, by more than their
background noise allows. A test that used only background noise would clip
stars. When enabled, it covers only pixels whose variance is dominated by
background, and it uses each partial stack's own across-frame variance, which
includes source noise.

Partial stacks made by different Seiza versions combine in one tile. Each is
an input that carries its own measurements, and the tile stack's key records
each one's key.

### 7. Assemble

Assembly writes a grid's tile stacks as one FITS or XISF image with WCS. A
`CREDITS` table goes with it: each contributor, their sessions, their
exposure, and their share of the summed weight. Each tile stack carries the
same table for its own tile.

## Partial stack files

The primary image is the weighted mean, so any viewer opens a partial stack
as an ordinary stack. Further images carry the rest.

| Image | Contents |
|---|---|
| Primary | Weighted mean, in the session's physical units |
| `WEIGHT` | Inverse variance per sample, W / σ₁² |
| `COVER` | Frames integrated per sample, `u16`, or `f32` where a deep tile passes 65535 |
| `REJECT` | Rejected samples per sample, `u16` or `f32` likewise (optional) |
| `VARIANCE` | Variance of one frame of weight 1 about the mean, from the stack's own frames (optional) |
| `PSFGRID` | One PSF model per cell |
| `SKYCELLS` | Star-masked background means and coverage per cell |

Tile stacks and mosaics add `CREDITS`.

A partial stack carries only the cards listed below. Observer names, site
coordinates and file paths from the frames are left out unless a contributor
chooses to include them. Partial stacks are meant to be shared, and capture
software writes the observer's location into every frame.

`seiza-image-io::save_images` writes the images to one file, `f32` or
unsigned 16-bit: in FITS the mean as the primary HDU and the rest as image
extensions named by `EXTNAME`, in XISF each as an image with that `id`.
`read_astronomy_named` reads one back by its name.

## Header cards

The cards make a file describe itself. A host can read only the header,
which is cheap and decodes no pixels, and learn what the file holds, what
made it, whether it is current, and whether it can be extended. Tile stacks
and mosaics carry the same identity cards, with partial stack keys as their
inputs.

Identity and resumption:

| Card | Value | Meaning |
|---|---|---|
| `SEIZAPST` | `'PARTIAL'`, `'TILE'` or `'MOSAIC'` | What the file is, as `SEIZAMST` names a master |
| `SEIZAPVR` | integer | Schema version of these cards |
| `SEIZASTK` | e.g. `'0.51.0'` | `seiza_stacking::OUTPUT_VERSION` when the frames were integrated |
| `SEIZAPIP` | e.g. `'0.1.0'` | `seiza_pipeline::OUTPUT_VERSION` when the file was written |
| `STEPKEY` | SHA-256 | Key of the step that made the file (see below) |
| `INPUTSHA` | SHA-256 | The ordered inputs: frame content hashes for a partial stack, partial stack keys for a tile |
| `NINPUTS` | integer | How many inputs `INPUTSHA` covers, whether accepted or rejected |
| `NCOMBINE` | integer | Inputs integrated (an existing card) |
| `CALIBSHA` | SHA-256 | The calibration masters' content |
| `OPTSSHA` | SHA-256 | The options that change the pixels |
| `HOSTSHA` | SHA-256 | A fingerprint the host supplies for its own decisions, such as PSF Guard's scoring settings; Seiza only compares it |
| `ESTIMATR` | `'ADDITIVE'` or `'CLIPPED'` | Whether frames can be appended, or the file can only be combined |
| `CONTEXT` | SHA-256 | The live-stack context that can extend an `ADDITIVE` file, when one was kept |
| `CONTRIB` | string | Who contributed the session: a handle the contributor picks |
| `WGHTSRC` | `'STACK'` or `'MEASURED'` | Whether `WEIGHT` came from the accumulator, or was built from measured noise for a stack made by other software |

Grid:

| Card | Meaning |
|---|---|
| WCS cards | The tile's projection, SIP included. `CRPIX` is the internal value plus 1 |
| `GRIDSHA` | SHA-256 of the grid definition |
| `TILEX`, `TILEY` | The tile's place on the grid |
| `WINDOWX`, `WINDOWY` | Where the file's window starts within the tile |
| `NOISECOR` | How much resampling correlated the noise |

Measurements, with *n* the channel:

| Card | Meaning |
|---|---|
| `BAND`*n* | The project band of channel *n* |
| `NOISE`*n* | σ₁, the median over the tile |
| `SKY`*n* | Background level |
| `ZEROPT`*n*, `ZPERR`*n*, `ZPSTARS` | Zero point against Gaia, its error, and the number of stars behind it |
| `PSFFWHM`, `PSFBETA`, `PSFELONG` | Median PSF: FWHM in arcseconds, Moffat β, elongation |
| `EXPTOTAL` | Exposure integrated, in seconds |
| `DATE-BEG`, `DATE-END` | First and last frame |
| `FILTER`, `INSTRUME`, `TELESCOP` | Carried over from the frames when they all agree |

A key is the SHA-256 of a step's inputs, options and `OUTPUT_VERSION`:

- A partial stack's key covers the content hashes of its frames, its
  masters' keys, the grid, the options and the version.
- A tile stack's key covers its partial stacks' keys, the measurements it
  used, the background planes and the combine options.

A change therefore reaches exactly the steps downstream of it. Identity is
the SHA-256 of a file's bytes, not its path, size and modification time,
because two machines can hold the same frame at different paths. A host
caches each hash by path, size and modification time, so each file is hashed
once.

## Resuming

A step is done when a file carrying its `STEPKEY` exists and its checks
pass. To resume, a coordinator rebuilds the plan, computes every key from
the inputs, reads the headers in the store, and runs only the steps that
are missing. The writers already publish files atomically, so a half-written
file never carries a key.

- **A worker dies during a step.** Its lease lapses and the step runs again
  somewhere else.
- **The coordinator restarts.** PSF Guard's task board lives in memory. A
  restarted coordinator recovers every finished step from the headers, so
  the board itself never needs to be saved.
- **More frames arrive for a session already stacked.** For an `ADDITIVE`
  file, hash the first `NINPUTS` entries of the new input list. If that
  equals `INPUTSHA`, and `CALIBSHA`, `OPTSSHA`, `HOSTSHA` and `SEIZASTK` all
  match, reopen the `CONTEXT` and push only the new frames. This is the
  proof PSF Guard's `ResumeManifest` makes, carried in the file itself. A
  `CLIPPED` file means reintegrating that one session again, from the
  reopened context's ledger when one was kept and from scratch otherwise.

  The step over the first `NINPUTS` frames has a key of its own, which a
  store finds without an index: computing the keys of the new list's
  prefixes, longest first, finds the step to extend. Phase 1 does this with
  `SessionOptions::keep_context`, which keeps each session's context in the
  store beside its partial stack.
- **A new session arrives.** Only its partial stacks are new. They are
  solved against the baseline, and only the tiles they reach are combined
  again, from the partial stacks already stored. That costs little, because
  a tile holds a handful of them.
- **A contributor's frames are gone.** Nothing after stage 4 needs them.
  Their partial stacks are the lasting record, and every later step reads
  only those.
- **A contributor withdraws.** The project records a decision to leave out
  their partial stacks. The tiles they reached are stacked again without
  them, which gives what those tiles would have been had the partial stacks
  never arrived.
- **Combine options change.** Tile keys change; partial stack keys do not.
- **A second project covers the same sky.** If it uses the same grid, every
  session the two projects share is already stacked: its partial stacks
  have the same keys. Only the second project's tile stacks are new.
- **Seiza changes an `OUTPUT_VERSION`.** If it is stacking's, every partial
  stack, and everything after it, runs again. If it is `seiza-pipeline`'s, that
  reruns the partial stacks too, because `seiza-pipeline` writes them (see the
  open questions).

`INPUTSHA` hashes inputs only, never decisions. Whether a frame was accepted
is an output of the step, and a host extending the list does not know it in
advance.

## Measuring again

A partial stack outlives its frames, but measurement keeps improving. A
remeasure step reads only a partial stack's pixels and writes a small
measurement file. Its key covers the partial stack's key and the version of
the measuring code. When that version changes, the plan schedules a
remeasure for every partial stack, and tile stacks use the new
measurements. A better zero-point fit then reaches a session whose frames
were deleted years before. The pixels are never rewritten, and the cards
keep the measurements made when the file was written. This needs
measurement to have a version of its own (see the open questions).

## Who runs what

Seiza owns:

- the plan: the steps, their keys and their dependencies;
- every step;
- the file format and its cards;
- the rules a store follows;
- a local executor.

The CLI runs a whole plan on one machine with the same steps a distributed
run uses, so a distributed result can be checked against a local one.

Later, PSF Guard's board and worker protocol move into Seiza, as the
calibration library did (#403), so that the CLI, the apps and PSF Guard share
one worker.

The API follows AGENTS.md:

- Grids, projects, partial stacks, steps and the plan are native types with
  methods.
- The C ABI may run planning as one JSON call.
- Python binds the native types.

## Interop and submission

Three interfaces carry the work, and each can be used without the other
two: stores move results, plans and steps say what runs, and a board hands
work out.

### Stores: what moves

Every step writes its outputs under its key. A store can be a local
directory, a NAS, a synced folder or a bucket, as long as these rules hold:

- A file is written once and never changed. A key already present is never
  overwritten.
- Files are written under a hidden temporary name and renamed into place, so
  a sync tool never copies half a file under a real name.
- There is no shared index. A store's contents are read from its files'
  headers, so two machines never edit the same file.
- What does change, such as a project's decisions or claims on work, is
  written as new small files, one per entry. Old entries are never edited.
- Copies that sync tools make of a conflicting file are ignored.

Merging two stores is then copying across whatever keys one lacks. Nothing
conflicts and nothing runs again. A laptop that worked offline, a second
coordinator and a USB drive all join the same way.

```text
store/
  projects/ngc7000/project.json
  projects/ngc7000/decisions/2026-11-02T211403Z-3f1a.json
  steps/9f/9f2c…/partial-3-1.fits
  steps/9f/9f2c…/partial-4-1.fits
  claims/9f2c….<handle>
```

Other software can read what a store holds. The primary image of every
partial stack, tile stack and mosaic is an ordinary mean image, and `WEIGHT`
and `COVER` are plain image extensions.

### Plans and steps: what runs

A plan is a JSON document computed from the project, an inventory and the
versions in use. The same inputs always give the same plan, so anyone can
compute it again and check its keys. Each step lists:

- its key, its kind, and the keys or frame hashes it reads;
- what it needs: which machine holds its frames, the memory it expects,
  whether it uses a GPU, and the versions it requires;
- the outputs it writes.

Running one step has one contract:

```text
seiza pipeline step plan.json <key> --store <store> [--locations map.json]
```

- If the key is already in the store, it does nothing.
- It reads frames from local paths first, checked by hash, and everything
  else from the store.
- It publishes its outputs atomically and exits with a status.

Anything that can run a command can therefore drive Seiza: PSF Guard's
board, a cluster scheduler, cron, or a person at a terminal. Hosts can also
run steps in-process: the apps through the C ABI, Python through the native
types, and Rust hosts through `Plan` and `Step::execute`.

```text
seiza pipeline inventory <dirs> --out frames.json
seiza pipeline plan      <project> --inventory frames.json --store <store> --out plan.json
seiza pipeline run       plan.json --store <store>
seiza pipeline step      plan.json <key> --store <store>
```

An inventory lists sessions and their frames. `seiza pipeline inventory`
makes one by reading FITS and XISF headers, grouping frames into sessions
with the calibration library, and hashing them. PSF Guard makes the same
inventory from its database. A frame is named by its content hash, with
every place it is known to be:

```json
{"sha256": "41ab…", "size": 52183040,
 "locations": [{"holder": "remote-site", "path": "D:/NINA/NGC7000/…"}]}
```

### The board: how work is handed out

PSF Guard's protocol stays as it is. Workers claim with HTTP long polls and
hold 60-second leases that progress reports renew. A lapsed lease puts the
task back in the queue, and outputs are uploaded with their SHA-256 and
checked on arrival. What changes is that tasks become steps instead of whole
channels, and placement matches what a step needs against what a worker
declares:

| Step | Needs | Runs on |
|---|---|---|
| Masters, register, replay | The machine holding the session's frames | That machine, or the server after an upload |
| Background | Sky cells | Any machine, usually the coordinator |
| Tile stack | The tile's partial stacks | Any machine; preferably one that already has them |
| Assemble | The tile stacks | Any machine |

Workers already declare the kinds of work they take
(`--work stack|rc-astro|…`). They also need to declare:

- **Holdings:** which sessions' frames they hold, so a remote site's steps
  run there instead of uploading its frames.
- **Versions:** the stacking and pipeline versions they run. A worker on
  another version would make different keys, so it takes only steps whose
  keys match its versions.

A worker keeps the inputs it fetched between tasks, by hash. Today it
deletes its scratch after each task. PSF Guard's placement policies
(`workers_first`, `workers_only`, `this_server`) and its preference for tasks
a person started carry over, per project and per kind of step.

Without a board, claims are files. A machine about to run a step writes
`claims/<key>.<handle>`, and others skip steps with a recent claim. Two
machines that run one step anyway waste time but produce nothing wrong,
because steps are idempotent.

### Submitting work

| Submitted | Typical case | What runs afterwards | Fidelity |
|---|---|---|---|
| Frames, left in place | Your own rigs, with a worker beside the frames | Everything | Full |
| Frames, uploaded | A capture PC with no worker, through PSF Guard's ingest | Everything, on the server or other workers | Full, at the cost of the upload |
| A session still being shot | Frames as they arrive | Appended to the live stack; partial stacks when the session ends | Full |
| Partial stacks made with Seiza | A contributor who stacked on their own machine | Background onwards | Full; checked by measurement, not recomputed |
| A finished stack | A PixInsight or Siril master light, or a Seiza stack made outside the project | Measured into a partial stack, then background onwards | Reduced |
| A store | An offline laptop, another coordinator | Nothing; the files are copied across | Whatever its files hold |

A finished stack becomes a partial stack in four steps:

- Seiza measures its noise, PSF, Gaia zero point and sky cells, as it does
  for its own partial stacks.
- A Seiza stack keeps its weight map. For a stack from other software, Seiza
  builds `WEIGHT` from the measured noise mesh, treats coverage as full, and
  writes `WGHTSRC = 'MEASURED'`.
- `INPUTSHA` is the hash of the submitted file.
- The stack is resampled onto the project grid, and `NOISECOR` records what
  that cost.

The combiner weights such a partial stack correctly, but cannot see its
dither edges or rejected trails.

### Trust

A store keeps two classes of partial stack:

- **Derived:** made by a step the store's own machines could run again. Its
  key was computed from inputs they can see, and it matched.
- **Submitted:** made elsewhere. Its key is the submitter's claim, because
  nobody else holds its frames.

A submitted partial stack is checked by what can be measured:

- its cards follow the schema;
- a fresh plate solve of its mean agrees with its WCS;
- its zero point and PSF measure as its cards say;
- its consistency score against the tile's other partial stacks passes.

A project chooses whether to accept submitted partial stacks, and whether to
require a signature from a contributor key it lists. The signature sits in a
file beside the partial stack. A FITS card value holds 68 characters, and an
Ed25519 signature is 128 hex digits.

## Collaborations

A casual collaboration is a project in a shared store, with several people
adding to it over months or years. Nobody runs a server, and nobody has to
keep their frames.

**Starting.** Someone writes the project file and puts it in the store. The
bands map the filter names people's capture software writes onto the bands
the project stacks.

```json
{"name": "NGC 7000 wide field",
 "grid": {"kind": "project", "center": [314.7, 44.3], "scaleArcsec": 1.2, "tilePixels": 4096},
 "bands": {"Ha": ["Ha", "H-alpha", "Halpha 7nm"], "OIII": ["OIII", "O-III"]},
 "combine": {"weighting": "inverseVariance"},
 "curators": ["<handle>"],
 "acceptSubmitted": true}
```

**Contributing.** A member runs one command on their own frames:

```text
seiza pipeline contribute <store>/projects/ngc7000/project.json <frames…> --as <handle> --out <outbox>
```

It runs masters, register, replay and measure on their machine, and writes
partial stacks with `CONTRIB` set. The member copies the outbox into the
shared store, or lets their sync tool do it. Their frames never leave their
machine. A member who stacks in other software submits that stack instead,
and `contribute` measures it into a reduced-fidelity partial stack.

**Building.** Anyone runs `seiza pipeline run` against the shared store. The
plan finds the partial stacks that no tile stack includes yet, solves their
backgrounds against the baseline, and stacks the tiles they reach. Two
members who build at the same time do the same work twice, but write the
same keys.

**Curating.** Decisions are small files in the project's `decisions/`, one
per decision: leave a partial stack out, take it back, make a new baseline,
or name a release. Each says who made it, when and why:

```json
{"by": "<handle>", "at": "2026-11-02T21:14:03Z",
 "exclude": ["9f2c…"], "reason": "Gradient dipoles around stars; flat mismatch"}
```

A project's state is its file, its decisions read in time order, and the
partial stacks in the store. Anyone who reads the same state builds the same
keys. Only curators' decisions count, and the project file lists the
curators.

**Flags.** A partial stack that fails its consistency score is left out
automatically and shown to the curators, who can take it back.

**Withdrawing.** A member who wants out asks a curator to record a decision
leaving out their partial stacks, or deletes the partial stacks from the
store. Either way, the tiles they reached are stacked again without them.

**Credit.** Every tile stack and mosaic carries `CREDITS`: each contributor,
their sessions, their exposure, and their share of the summed weight.

**Releases.** A curator names a release by recording the project's state
key. Anyone can build that release again from the same store, even after
newer partial stacks have joined.

**Years later.** The partial stacks are the archive.

- Partial stacks from older Seiza versions stay valid inputs. Each carries
  its own measurements and its schema version, and readers keep reading old
  schemas.
- Better measurement reaches them through remeasure, without their frames.
- A new Seiza version that changes how tiles are combined stacks every tile
  again. That costs reading the partial stacks, not stacking any frames.

Seiza can promise that the same state gives the same keys. It cannot promise
that the same keys give the same bytes on every machine: a build on another
CPU, or on a GPU, can differ in the last bits. The store keeps the copy that
arrived first and never overwrites it.

## The `seiza-pipeline` crate

The pipeline lives in a new crate, `seiza-pipeline`. It holds grids, projects,
partial stack files and their cards, the steps, the plan and the local
executor. Those are not per-frame work, and they change output on a schedule
of their own, so they get a crate and an output version of their own rather
than a corner of `seiza-stacking`. The new crate depends on `seiza-stacking`,
`seiza-calibration` and `seiza-image-io`, and later on `seiza` for solving and
Gaia photometry; none of them depends on it.

What stays where:

- `seiza-stacking` keeps the per-frame work: calibration, registration,
  normalization, weighting and rejection. It gains whatever the session
  step needs to call: replaying a ledger onto tiles, an accumulator exposing
  its weight sum, and merging accumulators on one grid.
- Image formats stay in `seiza-fits` and `seiza-xisf`, which gain
  multi-image files.
- `seiza-pipeline` owns what is new: the files and cards, the measurements put
  together into a partial stack, the background solve over sky cells, the
  tile combine, assembly and the plan.

`seiza-pipeline` writes pixels, so it has an `OUTPUT_VERSION` and an
`output_version` test of its own, under the same rules as stacking and
stretching. `SEIZASTK` records the stacking version that wrote a partial
stack's pixels, and a new card, `SEIZAPIP`, records the pipeline version. Both
go into every key. The crate also gets a C export for each public
function, Python twins, and an entry in `.github/binding-parity.toml`, as
AGENTS.md requires.

## Phases

1. **Partial stack files on today's grid.**
   - The `seiza-pipeline` crate, with its `OUTPUT_VERSION` and test.
   - Multi-image FITS and XISF.
   - The partial stack type and its cards.
   - Merging partial stacks on one reference grid.
   - The local executor, splitting a channel by session.
   - PSF Guard's session task and resuming from cards.

   This needs no plate solving: every worker registers against the same
   reference frame, so normalization and weights agree. It must show that
   the result differs from a single-machine stack only at pixels rejected
   differently.

   Done in Seiza. Hosts reach the same steps through the C ABI, Python and
   the CLI. PSF Guard has an opt-in method, off by default, that stacks each
   session of a channel apart on its own machine and combines them. It waits
   for the Seiza release that ships `seiza-pipeline`. Its worker task for one
   session is deferred.
   - `LiveStacker::open_fits_sessions_with_reference_image` stacks one
     session against the shared reference, its lights calibrated by their own
     recipe, and `LiveStacker::add_session_lights` lets a reopened session
     stack take the lights its session gained.
   - `StackSnapshot::weight_sum` keeps Σw. Until phase 2 measures σ₁,
     `WEIGHT` holds W itself, which is all that combining on one grid needs,
     and `VARIANCE` the variance of a frame of weight 1 the stack measured.
   - `SessionStep` plans a session's key from its files' contents and runs
     it, and `run_channel` runs a channel against a `Store`, doing only what
     the store lacks. With `keep_context`, a session that gains frames is
     extended from its stored context, as **Resuming** describes.
   - The C ABI, Python and `seiza pipeline run` reach them.

   The partition invariant holds to `f32` rounding, and combining is held to
   its output by `seiza-pipeline`'s `output_version` test. The resume
   invariant holds to the bit: a session extended from its context, live or
   reintegrated, gives the partial stack stacking every frame at once does.

   Left from phase 1, for when work resumes:
   - A digest cache. Every plan still reads and hashes each light to make
     its session's key.
   - Pruning a store: nothing yet removes partial stacks and contexts that
     no plan names, in Seiza's store or in PSF Guard's cache.
   - A worker task per session, so that remote workers take sessions rather
     than whole channels, and a session's stack that can stop part way.
2. **Measurements.** The weight map, the PSF grid with an `f32` fitter, Gaia
   zero points for mono filters, and sky cells.
3. **Project grids.**
   - Register, grid and replay as separate steps.
   - Replaying frames onto tiles, and tile stacks.
   - The background solve on sky cells, with a baseline.
   - Assembly, with credits.

   This replaces the expanded canvas for distributed runs once it pairs
   overlaps as tightly.
4. **The whole plan and interop.**
   - Masters become steps too, and every stage gets keys.
   - The store rules and the plan JSON.
   - The `inventory`, `plan`, `run` and `step` commands.
   - PSF Guard runs every kind of step. Workers declare holdings and
     versions, and cache their inputs.
5. **Collaborations.**
   - The `contribute` command.
   - Project decisions, and claims as files.
   - Submitted and derived partial stacks, and signatures.
   - Finished stacks as partial stacks.
   - Remeasure, credits and releases.
6. **Sky grids and better weighting.**
   - One or more fixed tilings of the whole sky, so that projects covering
     the same sky share partial stacks.
   - Point-source and proper coaddition weighting.
   - The worker protocol moves into Seiza.

From phase 1 onwards, the store and the plan handle more than one project.
A project grid is the only kind of grid until phase 6, but projects are
never assumed to be alone in a store.

## Invariants to test

Each new output gets an output-version case before it lands. A planted change
in that mode must make the case fail.

- **Partition.** Frames split into sessions and combined by inverse variance,
  without rejection across sessions, match one weighted stack of all the
  frames on the same grid, to `f32` rounding.
- **Merge.** Merging `ADDITIVE` partial stacks equals stacking their union
  without rejection, checked against `integrate_registered_frames`.
- **Resume.** Writing a file, reading its cards and extending it is
  bit-identical to never stopping, as `save_context` is today.
- **Leave one out.** Removing a term from the sums equals combining without
  that partial stack.
- **Tiles.** Neighbouring tiles agree where their margins overlap, and
  assembling tiles equals combining the grid untiled.
- **Weight.** A synthetic vignetted flat with dither edges gives the combined
  noise the weight map predicts.
- **Placement.** A real mosaic placed by per-session SIP solves pairs overlap
  stars as tightly as the expanded canvas does.
- **Order.** The same partial stacks and baseline give the same planes and
  tile stacks, whatever order the partial stacks arrived in.
- **Withdrawal.** Leaving a contributor's partial stacks out gives the tiles
  that would exist had they never been added.
- **Stores.** Copying two stores together and then building gives the same
  keys as building each and then copying the results together.
- **Mixed versions.** A tile combines partial stacks written by older
  schemas and versions.

## Open questions

- **Which sky tilings to offer.** This stays open by design. Candidates
  include TAN cells indexed by HEALPix, or bands of declination as in survey
  projection cells. Each needs a tile size and a pixel scale, and sessions
  range from under 1 to over 3 arcseconds per pixel, so there may be one
  tiling per resolution. Whatever is chosen has to meet three conditions:
  - Every machine computes the same cells and tiles from the definition
    alone.
  - Several tilings can sit side by side in one store, told apart by
    `GRIDSHA`.
  - A project can name any of them, or a project grid.
- **Versions.** Should each stage carry its own version, so that a change to
  the combiner does not stack every session again? `seiza-pipeline` could keep
  separate versions for the session steps, measurement and the tile steps,
  with the crate's `OUTPUT_VERSION` staying the floor. Remeasure needs at
  least the measurement version.
- **Bands from colour cameras.** A broadband OSC frame gives R, G and B. A
  dual-band filter on an OSC camera puts Hα mostly in red and OIII in green
  and blue, but not cleanly. How a project maps those channels onto bands is
  open.
- **Identity.** Handles are enough for a casual project. Should a project
  that cares about attribution use contributor keys from the start?
- **Narrowband zero points.** Gaia's broadband photometry predicts the
  continuum in a 3 nm filter poorly. Relative gains from star photometry in
  overlaps may have to set the scale, with Gaia only as a check.
- **Drizzle onto tiles.** The expanded canvas refuses Bayer drizzle and
  drizzled reintegration today.
- **Precision.** Partial stacks are stored as `f32` and the combiner sums in
  `f64`. Tiles are combined again from stored partial stacks rather than by
  adding and subtracting terms in place, so error does not accumulate.
- **Keeping the ledger.** Should a session keep its frame ledger, so that
  rejection across all frames can run again later, at the cost of keeping
  the frames?
- **Size.** A partial stack is about two and a half times the size of a mean
  image (mean and weight in `f32`, coverage in `u16`). `seiza-fits` reads
  tile-compressed FITS. Writing it would halve the size of `COVER` and
  `REJECT`.

[zo]: https://ui.adsabs.harvard.edu/abs/2017ApJ...836..188Z/abstract
