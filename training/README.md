# Training corpus

The passes here turn telescope data into the training corpus for Seiza's
learned restoration models, as the [design note](../docs/design/ml-restoration-training.md#real-data)
lays out. Every image passes through the `seiza` package built from this
checkout, so the corpus is made by the stacker users run. Code and
configuration are committed; the store, its data and any weights are not.

## Set up

```sh
uv venv .venv && uv pip install --python .venv/bin/python maturin
(cd seiza-py && VIRTUAL_ENV=../.venv ../.venv/bin/maturin develop --release)
uv pip install --python .venv/bin/python -e 'training[test]'
(cd training && ../.venv/bin/python -m pytest)
```

Rebuild `seiza-py` whenever the stacker changes. Every product records the
`seiza` version and `STACKING_OUTPUT_VERSION` it was made with.

## Passes

Each pass reads what earlier passes recorded in the manifest and can be
rerun alone; `--set ID` limits one to a set, and `--limit N` to N sets.
`measure` and `build` each read every frame, so `process` runs them one
set at a time, and the build reads the frames the measurement just read
from the page cache rather than over the network again.

```sh
cd training
python -m seiza_training --store STORE crawl --corpus CORPUS --library LIBRARY
python -m seiza_training --store STORE calibrate
python -m seiza_training --store STORE stacks --psf-guard STACKS
python -m seiza_training --store STORE split --hold-out TELESCOPE
python -m seiza_training --store STORE process   # measure, then build, set by set
python -m seiza_training --store STORE review
python -m seiza_training --store STORE status
```

1. **crawl** reads every header under `CORPUS` and `LIBRARY` through
   Seiza, and groups the lights into sets: one target, filter, camera,
   gain, offset, exposure, binning and night on one telescope. The corpus
   keeps lights as `<telescope>/_Source/<year>/<target>/<night>/LIGHT/`,
   and masters made by other software under `<telescope>/_Process`; the
   library keeps calibration masters as `<database>/calibration-masters/`.
   A file that has not changed is not read again. Headers that disagree,
   such as two exposure cards, are flagged, never resolved silently.
2. **calibrate** chooses each set's bias, dark and flat from the library
   with `plan_calibration`, which must suit every light of the set. A set
   without a flat, or without both a dark and a bias, is left out with the
   reason.
3. **stacks** copies in the finished stacks PSF Guard made from the corpus:
   one target, filter and exposure over all the nights it was imaged, under
   `STACKS/<database>/stack-previews/<job>/group-<n>.fits`. Each PSF Guard
   database is a telescope's scheduler database, found by the targets its
   jobs name in `CORPUS/<telescope>/_Scheduler/schedulerdb.sqlite`. The
   file names the scheduler recorded for a stack's frames find them in the
   corpus, and so the sets they came from. A stack copied at its current
   revision is not copied again.
4. **measure** measures every frame of each calibratable set with
   `analyze_project`. A frame that lost most of its stars, whose stars are
   elongated well beyond the set's, or that could not be measured stays out
   of the set's accepted frames. The measurements decide the recipes: noise
   pairs need 40 accepted frames, and a seeing split needs a quarter of the
   frames to be at least 25% sharper than the worst quarter. One frame is
   plate solved, when catalogs are installed.
5. **split** gives every set its train, validation or test group. Sets of
   one object, or whose fields overlap on the sky, share a group, so
   neighbouring sky never crosses a split. A group keeps the split it was
   first given. A finished stack joins the group of the sets its frames
   came from, or else of the sets of its object. `--hold-out` marks a
   telescope's sets and stacks `held-out`: never trained on, and scored
   apart.
6. **build** calibrates and registers each set's accepted frames once, with
   the CLI's default stacking options, and keeps the registered frames in
   the scratch directory while it integrates every product from them with
   `LiveStacker.reintegrate(frames=...)`. All of a set's products share one
   grid and one normalization. A frame the stacker had to brighten by half
   again over the set's median gain lost its light to cloud, and is left
   out of every product. When registration turns away more frames for
   turning past its rotation limit than it admits, the camera turned during
   the night and the reference fell in the smaller orientation, so the set
   is registered again on a reference from the larger one.
7. **review** writes `STORE/review/index.html` and one page per built set,
   with its stacks at one stretch, its measurements and its pair checks.
   Record a decision with
   `python -m seiza_training --store STORE decide SUBJECT KIND accept|reject --reviewer NAME`,
   where `KIND` is `set` or `starless`.

## Products

Per set, under `STORE/products/<set id>/`:

| Directory | Contents |
|---|---|
| `deep/` | every clear admitted frame |
| `noise/nSS-dD/{a,b}/` | noise pair: A of `SS` frames drawn one from each stretch of the night, B of all the others; three draws `D` of 4, 8, 16 and 32 frames, at most a fifth of the set |
| `seeing/{best,worst}/` | the best and worst quarter of the frames by FWHM, at matched counts |

A finished stack's image is under `STORE/products/psf-guard/<database>/<job>/<group>/stack/`,
as `image.npy` alone: PSF Guard keeps no variance or coverage beside it.

Each stack directory holds `image.npy` and `variance.npy` (float32),
`coverage.npy` and `rejected.npy` (uint16), as `(height, width)` or
`(channels, height, width)`, so a loader memory-maps them and reads patches
at random. The deep stack and the seeing quarters are also written as
`stack.fits`, with the reference's WCS.

A noise pair whose halves' mean FWHM differs by more than 10%, or whose
backgrounds differ by more than a quarter of the noise of their difference,
is kept but marked `rejected`, with the reason and the measurements.

## The manifest

`STORE/manifest.sqlite` holds:

- `frames`: every file read, with its header fields, flags and set;
- `sets`: each set's calibration plan, measurement summary, recipes, sky
  group, split and build state;
- `measurements`: every measured frame;
- `products`: every product, with its recipe, frames, files, statistics and
  the versions it was made with;
- `reviews`: review decisions, with reviewer and date;
- `passes`: each pass run, with its arguments, versions and summary.

## Time and space

On an M4 Max reading over the network, measuring takes about 1 s a frame
for a 26 MP sensor and 4 s for a 61 MP one; it is bound by the processor.
Building a 26 MP set of 94 frames, with its frames already in the page
cache, took 19 s: 4 s to register and the rest for the deep stack and 9
noise pairs. A set needs about 4 bytes per pixel per frame of scratch while
it builds, 17 GB for 160 frames of 26 MP, and its products about 6 GB for
94 frames, most of them noise pairs. `build` refuses a set that the scratch
or the store's disk cannot hold.

The store can be on a network share. The manifest keeps a rollback journal
rather than WAL, whose shared-memory index needs a local disk, and a build
writes only finished products there. Its registered frames, written once and
read for every product, go to `--scratch`, which should be on local disk;
by default it is `~/.cache/seiza-training/scratch`. On ZFS, 1 MB records
and zstd suit the store: zstd keeps a stack in 58% of its size (the float
planes in about 80%, the 16-bit counts in 13%), and lz4 in 77%.
