"""Train, evaluate and export the noise model on the corpus's noise pairs.

A pair's A half is the input and its B half, every other frame of the
night, the target. B's noise is independent of A's, so an L2 loss against
it converges to the clean image as Noise2Noise has it, and B is deeper
than A, so the target adds little noise of its own.

The network sees `asinh((x − b) / σ̄)` and the noise map `σ / σ̄`, where `b`
is the image's median and `σ̄` its median noise, both of the whole image so
that no tile has statistics of its own. It returns a residual in units of
`σ̄`; the loss is taken in linear units, scaled by `σ̄`, with bright star
cores weighted down, since the halves' seeing differs a little there.
With `sky` "flat", the network reads the image less its smooth sky
(`smooth_sky`) instead: one background for the whole image leaves a
gradient's far side well above it, where a model reads the sky as signal and
leaves its coarse noise, or adds to it. The residual is added to the image
as it was, so the sky taken out comes back.

Training reads overlapping tiles from a tile bank: windows cut once from
each pair on an overlapping grid and cached on local disk, from which each
step takes tiles at random offsets. The bank sits in the local scratch
directory, beside the registered frames. Evaluation runs whole images through
the same overlapping-tile inference that ships, and measures the noise of
`f(A) − B` against that of `A − B`: at the pixel scale, which is the grain,
and binned by the band scale, which is the low band.

The dark-noise pairs (`dark_noise`) have a target far less noisy than a B
half, a stack of half the night's frames, whose noise the input does not
share, so on them a model's error is measured nearly directly. Their noise
alone, on a flat sky, shows whether a model makes structure where there is
none. They can also make up a share of every batch.

Against them a model's signal transfer is measured too: how much of the
signal at each scale it keeps in faint and mid regions. The input and the
target share their signal but no noise, so `⟨f(x)·t⟩ / ⟨x·t⟩` within a
band holds no noise bias: 1 keeps the signal at that scale whole, and less
than 1 smooths it away.

A few frames cannot reject every hot pixel or cosmic ray. Training can add
synthetic ones to each input tile, resampled as registration resamples a
frame, while the target keeps none, so the network learns to remove them;
the empty sky, with outliers planted in it, measures how much of them a
model leaves.

Training can also change both halves of a pair alike, which keeps their
noise independent, so the network meets images the pairs do not hold:
tiles shifted by a sub-pixel offset through Lanczos-3, or sharpened by an
unsharp mask, the noise map carried through the filter; a sky plane under
the input; and a share of every batch from the pairs binned 2 × 2, whose
stars are half as many pixels wide. Models trained without them lost faint
stars in binned and sharpened images.

Taking out all the noise a model finds gives the least error against a
clean target but looks smooth and flat; an image keeps some. `NoiseBands`
takes out a share of the grain and of the wider mottle apart, as the
runtime's per-band amounts do, and a run can be measured with them.

Threads build batches ahead while the GPU trains, each batch from a
generator seeded by its step, and on CUDA the network runs channels-last.
Training can run the network in bfloat16 with the loss in float32;
evaluation and export stay in float32. Evaluation measures several pairs
at once, their GPU calls taking turns.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import random
import threading
import time
import warnings
import zlib
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import dark_noise
from .build import BAND_SCALE_PX, binned
from .manifest import Manifest, loads, now
from .store import Store, robust_sigma, versions
from .tiles import infer
from .unet import ResidualUNet, reach, receptive_field

WINDOW = 320
WINDOWS_PER_HALF = 40
# Of a pair binned 2 × 2, a quarter of the pixels, a quarter as many.
BINNED_WINDOWS = WINDOWS_PER_HALF // 4
# Pixels this many σ̄ above the background are star cores, weighted down.
CORE_SIGMA = 50.0


def pairs(
    manifest: Manifest, split: str, sets: set[str] | None = None, *, kind: str = "noise-a"
) -> list[dict[str, Any]]:
    """The built pairs of one split's sets: each A product of `kind` with
    its B, the noise pairs' or, with `deep-half-a`, the deep stacks'
    halves."""
    rows = manifest.rows(
        "SELECT p.*, s.split, s.name FROM products p JOIN sets s ON s.id = p.set_id "
        "WHERE p.kind = ? AND p.status = 'built' AND s.split = ? ORDER BY p.id",
        (kind, split),
    )
    found = []
    for row in rows:
        if sets is not None and row["set_id"] not in sets:
            continue
        b = manifest.rows("SELECT * FROM products WHERE id = ?", (row["id"].removesuffix("/a") + "/b",))
        if b and b[0]["status"] == "built":
            found.append({"a": dict(row), "b": dict(b[0]), "set": row["name"]})
    return found


NOISE_MAPS = ("raw", "model", "image", "constant")
# The side of the blocks the modelled map takes the sky's variance over.
SKY_BLOCK = 32
# The median of a χ² with one degree of freedom over its mean: a second
# difference's square, of white noise, has it.
CHI2_1_MEDIAN = 0.4549
# The image map fits its signal term on the blocks whose median is this
# many times its sky noise above the background, when there are at least
# `IMAGE_GAIN_BLOCKS` of them.
IMAGE_GAIN_LEVEL = 5.0
IMAGE_GAIN_BLOCKS = 20
# The image map's level against the modelled map's: the median, over 24 of
# the train sets' 4-frame A halves, of the modelled map's median noise over
# the image map's before this factor (measured 2026-11-06, spread
# 0.99–1.31). It puts the image map on the level that models trained with
# the stack's map know. Against the noise of the halves' difference on the
# sky, the image map read 1.15–1.31 times it, the modelled map 1.19–1.61.
IMAGE_MAP_SCALE = 1.13
# How much neighbouring pixels share their noise: the second-difference
# noise of the image binned 2 × 2, times 2, over the image's own, on the
# sky; white noise gives 1. Seiza's train halves read 1.12–1.45, at 4 frames,
# 8 and a night's depth alike, with this median (2026-11-06, 16 of each).
# Another stacker's stacks read the same, 1.22–1.38, but its drizzled ones
# 2.45–2.58: their noise is spread over neighbours, so second differences
# see far less of it than there is, and a model told that much leaves a
# fifth of the finest noise as specks. The image map is raised by as much
# as an image's correlation exceeds this.
NOISE_CORRELATION = 1.24
# How the image map is made, named in every bank made with it, so a bank
# cut before a change to `image_sigma` is never read after it. 2: raised
# for spread noise.
IMAGE_MAP_REVISION = 2


def map_part(noise_map: str) -> str:
    """The noise map in a bank's name: a map made from the image names how
    it was made, as well as its kind."""
    if noise_map in ("image", "constant"):
        return f"map-{noise_map}-{IMAGE_MAP_REVISION}-{IMAGE_MAP_SCALE:g}-{NOISE_CORRELATION:g}"
    return f"map-{noise_map}"


def noise_correlation(image: np.ndarray) -> float:
    """`NOISE_CORRELATION`'s measure of one image, on the pixels within 3
    of its robust spreads of its median; 1 where too few are left."""
    values = image.astype(np.float64)
    level = float(np.nanmedian(values[::3, ::3]))
    spread = 1.4826 * float(np.nanmedian(np.abs(values[::3, ::3] - level)))
    sky = np.isfinite(values) & (np.abs(values - level) < 3 * spread)

    def noise(plane: np.ndarray, mask: np.ndarray) -> float:
        across = (plane[:, :-2] - 2.0 * plane[:, 1:-1] + plane[:, 2:])[mask[:, 1:-1]]
        down = (plane[:-2, :] - 2.0 * plane[1:-1, :] + plane[2:, :])[mask[1:-1, :]]
        found = np.concatenate([across[np.isfinite(across)], down[np.isfinite(down)]])
        if found.size < 1000:
            return float("nan")
        return 1.4826 * float(np.median(np.abs(found - np.median(found)))) / math.sqrt(6.0)

    height, width = values.shape[0] // 2 * 2, values.shape[1] // 2 * 2

    def half(plane: np.ndarray) -> np.ndarray:
        return plane[:height, :width].reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))

    whole, binned = noise(values, sky), noise(half(values), half(sky.astype(np.float64)) == 1)
    ratio = 2 * binned / whole
    return ratio if np.isfinite(ratio) and whole > 0 else 1.0


def input_sigma(kind: str, image: np.ndarray, sigma: np.ndarray | None, coverage: np.ndarray | None) -> np.ndarray:
    """The noise map a network reads, of `kind` (`NOISE_MAPS`): the stack's
    own per-pixel map, the map `modelled_sigma` makes from it, the map
    `image_sigma` makes from the image alone, which needs neither the
    stack's variance nor its coverage, or that map's median everywhere,
    which tells the network the image's noise and nothing of where it
    changes."""
    if kind == "raw":
        return sigma
    if kind == "model":
        return modelled_sigma(image, sigma, coverage)
    if kind == "image":
        return image_sigma(image)
    if kind == "constant":
        mapped = image_sigma(image)
        return np.where(np.isfinite(mapped), np.nanmedian(mapped), np.nan).astype(np.float32)
    raise ValueError(f"a noise map is one of {NOISE_MAPS}, not {kind!r}")


def image_sigma(image: np.ndarray) -> np.ndarray:
    """A noise map from the image alone, for stacks that carry no variance:
    those made by other stackers, or changed after stacking. It has the
    modelled map's shape. The sky's variance is that of the image's second
    differences along each axis, as Seiza's `snr` module measures noise: a
    planar gradient cancels in them, and as medians over `SKY_BLOCK`-pixel
    blocks the stars drop out. Spread back smoothly, the blocks keep
    vignetting and thinly covered edges. A term growing with the signal is
    fitted on the blocks a bright nebula fills, so stars and nebulae stay
    noisier; it is taken out of those blocks' own variance first, so the
    sky is not counted twice. Where the noise is shared between neighbours
    more than in Seiza's stacks, as in a drizzled stack, the map is raised
    by as much (`noise_correlation`). NaN where the image is."""
    height, width = image.shape
    values = image.astype(np.float64)
    rows, columns = max(1, height // SKY_BLOCK), max(1, width // SKY_BLOCK)

    def block_medians(plane: np.ndarray) -> np.ndarray:
        if plane.shape[0] < rows * SKY_BLOCK or plane.shape[1] < columns * SKY_BLOCK:
            plane = np.pad(plane, ((0, max(0, rows * SKY_BLOCK - plane.shape[0])),
                                   (0, max(0, columns * SKY_BLOCK - plane.shape[1]))), constant_values=np.nan)
        blocks = plane[: rows * SKY_BLOCK, : columns * SKY_BLOCK].reshape(rows, SKY_BLOCK, columns, SKY_BLOCK)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(blocks, axis=(1, 3))

    across = np.full_like(values, np.nan)
    across[:, 1:-1] = (values[:, :-2] - 2.0 * values[:, 1:-1] + values[:, 2:]) ** 2
    down = np.full_like(values, np.nan)
    down[1:-1, :] = (values[:-2, :] - 2.0 * values[1:-1, :] + values[2:, :]) ** 2
    # A second difference of white noise has six times its variance.
    variance_blocks = 0.5 * (block_medians(across) + block_medians(down)) / (6.0 * CHI2_1_MEDIAN)
    background = float(np.nanmedian(values[::7, ::7])) if np.isfinite(values).any() else 0.0
    signal_blocks = block_medians(values) - background
    finite = np.isfinite(variance_blocks) & np.isfinite(signal_blocks)
    sky_variance = float(np.median(variance_blocks[finite])) if finite.any() else 1.0
    gain = 0.0
    nebula = finite & (signal_blocks > IMAGE_GAIN_LEVEL * np.sqrt(sky_variance))
    if nebula.sum() >= IMAGE_GAIN_BLOCKS:
        floor = float(np.median(variance_blocks[finite & ~nebula])) if (finite & ~nebula).any() else sky_variance
        gain = max(0.0, float(np.median((variance_blocks[nebula] - floor) / signal_blocks[nebula])))
    sky_blocks = variance_blocks - gain * np.clip(np.nan_to_num(signal_blocks, nan=0.0), 0.0, None)
    sky_blocks = np.where(np.isfinite(sky_blocks), sky_blocks, sky_variance)
    sky_blocks = np.maximum(sky_blocks, 0.25 * sky_variance)
    sky = upsample(sky_blocks, height, width, SKY_BLOCK).astype(np.float64)
    signal = np.clip(np.nan_to_num(values - background, nan=0.0), 0.0, None)
    raised = max(1.0, noise_correlation(image) / NOISE_CORRELATION)
    mapped = IMAGE_MAP_SCALE * raised * np.sqrt(sky + gain * signal)
    return np.where(np.isfinite(image), mapped, np.nan).astype(np.float32)


def load_half(
    store: Store, product: dict[str, Any], *, noise_map: bool = True, kind: str = "raw"
) -> tuple[np.ndarray, np.ndarray | None]:
    """A stack's image, NaN where nothing covers it, and its noise map
    when `noise_map`: `sqrt(variance / coverage)` per pixel, or the map of
    `kind` (`input_sigma`) made from it or from the image."""
    files = loads(product["files"])
    image = np.load(store.root / files["image"]).astype(np.float32)
    coverage = np.load(store.root / files["coverage"])
    image[coverage == 0] = np.nan
    if not noise_map:
        return image, None
    variance = np.load(store.root / files["variance"]).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.where(coverage > 0, np.sqrt(variance / coverage), np.nan).astype(np.float32)
    return image, input_sigma(kind, image, sigma, coverage)


def extended(grid: np.ndarray) -> np.ndarray:
    """A grid with one more value on each side, carried on along the slope
    from the edge's neighbour, or repeated along a side one value long."""
    for axis in (0, 1):
        ends = [(0, 0), (0, 0)]
        ends[axis] = (1, 1)
        if grid.shape[axis] > 1:
            grid = np.pad(grid, ends, mode="reflect", reflect_type="odd")
        else:
            grid = np.pad(grid, ends, mode="edge")
    return grid


def upsample(grid: np.ndarray, height: int, width: int, block: int, *, extrapolate: bool = False) -> np.ndarray:
    """A grid of block values, one per `block` × `block` pixels, spread
    bilinearly over `height` × `width` pixels: past the outer blocks'
    centres held at theirs, or with `extrapolate` carried on along the
    slope from the next blocks in."""
    offset = 0
    if extrapolate:
        grid, offset = extended(grid), 1
    rows, columns = grid.shape
    ys = np.clip((np.arange(height, dtype=np.float32) + 0.5) / block - 0.5 + offset, 0, rows - 1)
    xs = np.clip((np.arange(width, dtype=np.float32) + 0.5) / block - 0.5 + offset, 0, columns - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, rows - 1), np.minimum(x0 + 1, columns - 1)
    wy, wx = (ys - y0)[:, None], (xs - x0)[None, :]
    top = grid[y0][:, x0] * (1 - wx) + grid[y0][:, x1] * wx
    bottom = grid[y1][:, x0] * (1 - wx) + grid[y1][:, x1] * wx
    return (top * (1 - wy) + bottom * wy).astype(np.float32)


def modelled_sigma(image: np.ndarray, sigma: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """A noise map without a short stack's scatter. A stack's per-pixel
    variance comes from its few frames at that pixel, so with four frames
    it scatters by a factor of several. The model keeps its level: the
    sky's variance, as medians over `SKY_BLOCK`-pixel blocks spread back
    smoothly, so vignetting and thinly covered edges stay; plus a term
    growing with the image's own signal, fitted where the signal holds most
    of the variance, so stars and bright nebulae stay noisier. The median of
    a variance from `n` samples runs low by the Wilson–Hilferty factor,
    which is divided out."""
    height, width = image.shape
    variance = sigma.astype(np.float64) ** 2
    rows, columns = max(1, height // SKY_BLOCK), max(1, width // SKY_BLOCK)
    blocks = variance[: rows * SKY_BLOCK, : columns * SKY_BLOCK].reshape(rows, SKY_BLOCK, columns, SKY_BLOCK)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sky_blocks = np.nanmedian(blocks, axis=(1, 3))
    covered = coverage[coverage > 0]
    frames = float(np.median(covered)) if covered.size else 1.0
    if frames > 1:
        sky_blocks /= (1.0 - 2.0 / (9.0 * (frames - 1.0))) ** 3
    sky_blocks = np.where(np.isfinite(sky_blocks), sky_blocks, np.nanmedian(sky_blocks))
    sky = upsample(sky_blocks, height, width, SKY_BLOCK).astype(np.float64)
    signal = np.clip(np.nan_to_num(image - np.nanmedian(image), nan=0.0), 0.0, None)
    bright = np.isfinite(variance) & (signal > 20.0 * np.sqrt(np.nanmedian(sky_blocks)))
    gain = 0.0
    if bright.sum() > 100:
        gain = max(0.0, float(np.median((variance[bright] - sky[bright]) / signal[bright])))
    modelled = np.sqrt(sky + gain * signal).astype(np.float32)
    return np.where(coverage > 0, modelled, np.nan).astype(np.float32)


def scales(image: np.ndarray, sigma: np.ndarray) -> tuple[float, float]:
    """An image's background `b` and median noise `σ̄`."""
    finite = np.isfinite(image) & np.isfinite(sigma)
    return float(np.median(image[finite][::17])), float(np.median(sigma[finite][::17]))


# `matched` fits a gain on the pixels whose box-blurred B is `MATCH_LEVEL`
# σ̄ (of A) above its background, grown 3 pixels to take in the stars'
# wings, when there are at least `MATCH_PIXELS` of them.
MATCH_LEVEL = 10.0
MATCH_PIXELS = 1000
# Its background planes leave out, in each of a few rounds, the pixels
# either half puts this many times its spread from its plane.
MATCH_CLIP = 4.0
MATCH_ROUNDS = 3


def matched(image_a: np.ndarray, image_b: np.ndarray, noise: float) -> tuple[np.ndarray, float]:
    """B scaled and shifted onto A's flux and background, and the gain.

    The halves of a pair are stacks of different frames, so their
    normalization can leave B's bright flux a few percent above or below
    A's: among the train pairs, from 0.58 to 1.42 times it, a tenth of
    them more than 2.5% off. A model cannot tell one pair's error from the
    next, so it learns their mean and dims what it denoises.

    Each half's background is a plane fitted by least squares over the
    same faint pixels, those where B blurred widely is below its median:
    a mean over shared pixels, which neither half's noise moves (a median
    over a nebula moves with the noise, which the halves do not share).
    A pixel either half puts `MATCH_CLIP` times its spread (from the
    median absolute deviation) from its plane is left out of both, and the
    planes fitted again: a short stack keeps hot pixels its rejection
    missed, which would lift A's plane and with it matched B's sky.
    The gain is then the ratio of the halves' flux above those planes,
    box-blurred, over the bright pixels, where whole stars count whatever
    the seeing of each half. (A least-squares gain would not do: on sky,
    B's blurred noise has nothing to do with A, and drags the slope
    down.) Pixels count only where every neighbour is covered."""
    both = np.isfinite(image_a) & np.isfinite(image_b)
    smooth_a = box_blur(np.where(both, image_a, 0.0).astype(np.float32), 2)
    smooth_b = box_blur(np.where(both, image_b, 0.0).astype(np.float32), 2)
    wide_b = box_blur(np.where(both, image_b, 0.0).astype(np.float32), 8)
    inner = box_blur(both.astype(np.float32), 2) > 0.999
    inner_wide = box_blur(both.astype(np.float32), 8) > 0.999
    height, width = image_a.shape
    across, down = np.arange(width) / width, np.arange(height) / height
    rows, columns = np.nonzero(inner_wide[::4, ::4])
    rows, columns = rows * 4, columns * 4
    faint = wide_b[rows, columns] <= np.median(wide_b[rows, columns])
    rows, columns = rows[faint], columns[faint]
    design = np.stack([np.ones(len(rows)), across[columns], down[rows]], axis=1)
    values = [image[rows, columns].astype(np.float64) for image in (image_a, image_b)]
    keep = np.ones(len(rows), dtype=bool)
    for round in range(MATCH_ROUNDS):
        fits = [np.linalg.lstsq(design[keep], value[keep], rcond=None)[0] for value in values]
        if round == MATCH_ROUNDS - 1:
            break
        keep = np.ones(len(rows), dtype=bool)
        for value, fit in zip(values, fits):
            residual = value - design @ fit
            spread = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
            if spread > 0:
                keep &= np.abs(residual) <= MATCH_CLIP * spread
    plane_a, plane_b = ((offset + along * across[None, :] + up * down[:, None]).astype(np.float32)
                        for offset, along, up in fits)
    signal_a, signal_b = smooth_a - plane_a, smooth_b - plane_b
    bright = inner & (box_max((inner & (signal_b > MATCH_LEVEL * noise)).astype(np.float32), 3) > 0)
    gain = float(signal_a[bright].sum(dtype=np.float64) / signal_b[bright].sum(dtype=np.float64)) \
        if bright.sum() >= MATCH_PIXELS else 1.0
    return (gain * (image_b - plane_b) + plane_a).astype(np.float32), gain


# The bands of B's brightness, in σ̄ above A's background, where
# `agreement` compares the halves' flux: the stars' wings and their cores.
AGREEMENT_BANDS = {"wings": (200.0, 1000.0), "cores": (1000.0, np.inf)}


def agreement(image_a: np.ndarray, image_b: np.ndarray, noise: float) -> dict[str, float | None]:
    """A's flux over B's in each of `AGREEMENT_BANDS`, B already `matched`,
    or None where fewer than 50 pixels are that bright. Matching leaves an
    error of normalization at 1 in both; haze in one half (its light spread
    into halos about the stars, their cores dimmed) or a change of seeing
    moves flux between the cores and the wings, which no gain puts back."""
    both = np.isfinite(image_a) & np.isfinite(image_b)
    background = float(np.median(image_a[both][::17]))
    level = (image_b - background) / noise
    found = {}
    for band, (low, high) in AGREEMENT_BANDS.items():
        mask = both & (level >= low) & (level < high)
        found[band] = float(np.sum(image_a[mask] - background, dtype=np.float64)
                            / np.sum(image_b[mask] - background, dtype=np.float64)) if mask.sum() >= 50 else None
    return found


def disagree(found: dict[str, float | None], limit: float) -> bool:
    """Whether any band of an `agreement` is more than `limit` from 1."""
    return any(value is not None and abs(value - 1) > limit for value in found.values())


# How the network reads the noise map: in σ̄ as it is, or through asinh, as
# it reads the image. Read as it is, a bright star's map reaches 50σ̄ and
# more, and one 41M model trained 160k steps took a star whose map rose to
# 55σ̄ for a cue to add tens of thousands of σ̄ in a square around it,
# though at 50σ̄ in the other half of the same pair it did nothing.
NOISE_PLANES = ("linear", "asinh")


def network_input(
    image: np.ndarray, sigma: np.ndarray, background: float, noise: float, noise_plane: str = "linear"
) -> np.ndarray:
    """The two planes the network reads, the noise map as `noise_plane`
    says; NaN becomes the background."""
    x = np.arcsinh(np.nan_to_num((image - background) / noise, nan=0.0))
    s = np.nan_to_num(sigma / noise, nan=1.0)
    if noise_plane == "asinh":
        s = np.arcsinh(s)
    return np.stack([x, s]).astype(np.float32)


# How the network reads the sky: less the one background of the whole
# image, or less the image's smooth sky first.
SKIES = ("global", "flat")
# The side of the blocks whose medians make the smooth sky.
FLAT_BLOCK = 256


def smooth_sky(image: np.ndarray) -> np.ndarray:
    """The slow change of an image's sky, less its own median, so that the
    image less it keeps its level: medians over `FLAT_BLOCK`-pixel blocks
    (one block where the image is smaller), a 3 × 3 median over those
    against a star or a knot that fills a block, spread back bilinearly and
    carried on past the outer blocks' centres, as a gradient is."""
    height, width = image.shape
    block = min(FLAT_BLOCK, height, width)
    rows, columns = height // block, width // block
    blocks = image[: rows * block, : columns * block].reshape(rows, block, columns, block)
    with warnings.catch_warnings():
        # A block wholly outside the coverage is NaN.
        warnings.simplefilter("ignore", RuntimeWarning)
        grid = np.nanmedian(blocks, axis=(1, 3))
    grid = np.where(np.isfinite(grid), grid, np.nanmedian(grid))
    # Carried on past the edge, so that a gradient's edge blocks keep their
    # values through the median.
    padded = extended(grid)
    grid = np.median(np.stack([padded[1 + dy:1 + dy + rows, 1 + dx:1 + dx + columns]
                               for dy in (-1, 0, 1) for dx in (-1, 0, 1)]), axis=0)
    return upsample(grid - float(np.median(grid)), height, width, block, extrapolate=True)


def seen(image: np.ndarray, sky: str = "global") -> np.ndarray:
    """The image the network reads: with `sky` "flat", less its smooth sky."""
    return image - smooth_sky(image) if sky == "flat" else image


def view(
    image: np.ndarray, sigma: np.ndarray, *, sky: str = "global", noise_plane: str = "linear"
) -> tuple[np.ndarray, float, float]:
    """The planes the network reads of a whole image, as `sky` and
    `noise_plane` say, with the background and median noise they are in
    units of; the residual it returns is added to the image as it is."""
    looked = seen(image, sky)
    background, noise = scales(looked, sigma)
    return network_input(looked, sigma, background, noise, noise_plane), background, noise


def flattened(image_a: np.ndarray, sigma_a: np.ndarray, image_b: np.ndarray,
              sky: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A pair's halves as the network reads them: with `sky` "flat", both
    less A's smooth sky, so the target moves with the input."""
    if sky != "flat":
        return image_a, sigma_a, image_b
    smooth = smooth_sky(image_a)
    return image_a - smooth, sigma_a, image_b - smooth


def binned_pair(image_a: np.ndarray, sigma_a: np.ndarray,
                image_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A pair's halves binned 2 × 2, each pixel the mean of its block, and
    A's noise map that mean's noise; NaN where any of a block is."""
    height, width = image_a.shape[0] // 2 * 2, image_a.shape[1] // 2 * 2

    def mean(plane: np.ndarray) -> np.ndarray:
        return plane[:height, :width].reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3), dtype=np.float32)

    return mean(image_a), np.sqrt(mean(sigma_a.astype(np.float32) ** 2)) / 2, mean(image_b)


def banked_pair(
    store: Store, pair: dict[str, Any], *, noise_map: str, weight_map: str | None, match_flux: bool,
    agree: float | None, holes: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None:
    """One pair's A image, its noise map and B image as `bank` cuts them,
    with A's weight map in a weighted bank; None when the matched halves
    `disagree` by more than `agree`."""
    weighted = weight_map is not None
    if weighted:
        image_a, weight_a = load_half(store, pair["a"], kind=weight_map)
        sigma_a = None
    else:
        image_a, sigma_a = load_half(store, pair["a"], kind=noise_map)
        weight_a = None
    image_b, _ = load_half(store, pair["b"], noise_map=False)
    if holes:
        from .holes import HoleGuard, guarded_pair

        image_a, image_b = guarded_pair(image_a, image_b, HoleGuard())
        if not weighted and noise_map in ("image", "constant"):
            # As the runtime makes it, from the image the network reads.
            sigma_a = input_sigma(noise_map, image_a, None, None)
    if weighted:
        sigma_a = input_sigma(noise_map, image_a, None, None)
    if match_flux:
        noise = scales(image_a, sigma_a if weight_a is None else weight_a)[1]
        image_b, _ = matched(image_a, image_b, noise)
        if agree is not None and disagree(agreement(image_a, image_b, noise), agree):
            return None
    return image_a, sigma_a, image_b, weight_a


# How often a bank being cut records how far it has come, in seconds.
CHECKPOINT_SECONDS = 60.0


def in_order(function: Any, items: list[Any], workers: int) -> Any:
    """`function` of each of `items`, in their order: with `workers` above 1,
    that many at once in processes of their own, at most twice as many
    ahead of the one yielded."""
    if workers <= 1:
        yield from map(function, items)
        return
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending: deque = deque()
        for item in items:
            pending.append(pool.submit(function, item))
            if len(pending) > 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def bank(
    store: Store, chosen: list[dict[str, Any]], *, seed: int = 0, noise_map: str = "raw", match_flux: bool = False,
    agree: float | None = None, sky: str = "global", binned_windows: int = 0, weight_map: str | None = None,
    holes: bool = False, workers: int = 1,
) -> dict[str, Any]:
    """Windows of every pair: A's image and noise map, and B's image, with
    `match_flux` first `matched` to A's. With `agree` too, a pair whose
    matched halves `disagree` by more than that is left out; the ids left
    out are under "dropped". With `sky` "flat", both halves are less A's
    smooth sky (`flattened`). With `binned_windows`, up to that many
    windows of each pair `binned_pair` 2 × 2, its sky taken out after, are
    cut in the same pass into a bank of their own, under "binned". With
    `weight_map` (`WEIGHT_MAPS`), A's map of that kind is banked too, under
    "weight", for the loss to weigh by and the outliers to be scaled in,
    and the halves are matched and judged by it. With `holes`, each pair
    is `guarded_pair` first, as the runtime's hole guard has a network read
    it. Cached by the pairs, the seed, the kinds of noise map, the matching,
    the agreement, the sky and the guard asked for. `workers` pairs are
    read and prepared at once, which leaves the bank as it would be. A cut
    records how far it has come every `CHECKPOINT_SECONDS`, and one stopped
    part way goes on from there when asked again, to the same bank."""
    if agree is not None and not match_flux:
        raise ValueError("agreement is judged on matched halves: ask for match_flux too")
    if sky not in SKIES:
        raise ValueError(f"sky is one of {SKIES}, not {sky!r}")
    if weight_map is not None and (weight_map not in WEIGHT_MAPS or noise_map not in ("image", "constant")):
        raise ValueError(f"a weight map, one of {WEIGHT_MAPS}, is for networks that read a map made from the "
                         f"image alone, not {noise_map!r}")
    parts = [p["a"]["id"] for p in chosen] + [seed, WINDOW, WINDOWS_PER_HALF]
    if noise_map != "raw":
        parts.append(map_part(noise_map))
    if weight_map is not None:
        parts.append(f"weight-{weight_map}")
    if match_flux:
        # 2: the background planes leave out hot pixels.
        parts.append("match-flux-2")
    if agree is not None:
        parts.append(f"agree-{agree:g}")
    if sky != "global":
        parts.append(f"sky-{sky}")
    if holes:
        from .holes import FLOOR_PERCENTILE, HoleGuard

        guard = HoleGuard()
        parts.append(f"holes-{guard.depth:g}-{guard.margin}-{guard.fill_radius}-{FLOOR_PERCENTILE:g}-{SKY_BLOCK}")

    def named(parts: list[Any]) -> str:
        return "noise-" + hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16]

    weighted = weight_map is not None

    def windows(progress: dict[str, Any] | None) -> tuple[Windows, Windows | None]:
        whole = Windows(store, named(parts), sources=len(chosen), seed=seed, weighted=weighted,
                        resume=progress and progress.get("whole"))
        binned = Windows(store, named([*parts, "binned-2", binned_windows]), sources=len(chosen), seed=seed,
                         per_source=binned_windows, weighted=weighted,
                         resume=progress and progress.get("binned")) if binned_windows else None
        return whole, binned

    checkpoint = store.scratch / "tiles" / named(parts) / "progress.json"
    progress = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
    whole, binned = windows(progress)
    cutting = [bank for bank in (whole, binned) if bank is not None and not bank.done]
    if progress and not all(bank.resumed for bank in cutting):
        # A cut whose files are gone starts again.
        progress = None
        whole, binned = windows(None)
    record = whole.directory / "dropped.json"
    if cutting:
        start = progress["pairs"] if progress else 0
        dropped = list(progress["dropped"]) if progress else []
        prepare = functools.partial(banked_pair, store, noise_map=noise_map, weight_map=weight_map,
                                    match_flux=match_flux, agree=agree, holes=holes)
        rest = chosen[start:]
        recorded = time.monotonic()
        for done, (pair, prepared) in enumerate(zip(rest, in_order(prepare, rest, workers)), start=start + 1):
            if prepared is None:
                dropped.append(pair["a"]["id"])
            else:
                image_a, sigma_a, image_b, weight_a = prepared
                if not whole.done:
                    whole.add(*flattened(image_a, sigma_a, image_b, sky), weight_a=weight_a)
                if binned is not None and not binned.done:
                    binned_a, binned_sigma, binned_b = binned_pair(image_a, sigma_a, image_b)
                    if noise_map in ("image", "constant"):
                        # What the image map makes of a binned image, as it
                        # would of one binned before it came.
                        binned_sigma = input_sigma(noise_map, binned_a, None, None)
                    binned_weight = binned_pair(image_a, weight_a, image_b)[1] if weighted else None
                    binned.add(*flattened(binned_a, binned_sigma, binned_b, sky), weight_a=binned_weight)
            if time.monotonic() - recorded >= CHECKPOINT_SECONDS:
                state = {"pairs": done, "dropped": dropped,
                         "whole": None if whole.done else whole.state(),
                         "binned": None if binned is None or binned.done else binned.state()}
                # Written whole and then moved, so a stop part way through
                # leaves the last record as it was.
                checkpoint.with_suffix(".tmp").write_text(json.dumps(state))
                checkpoint.with_suffix(".tmp").replace(checkpoint)
                recorded = time.monotonic()
        # Before the banks are marked done.
        record.write_text(json.dumps(dropped))
    banked = whole.finish()
    banked["dropped"] = json.loads(record.read_text()) if record.exists() else []
    if binned is not None:
        banked["binned"] = binned.finish()
    checkpoint.unlink(missing_ok=True)
    return banked


# The A halves of the deep stacks' pairs: a night's, and many nights'.
DEEP_KINDS = ("deep-half-a", "multi-night-half-a")
# Those a model trains on. A multi-night half holds whole nights, whose
# seeing differs, so its target's stars are wider or narrower than its
# input's and would teach a model to change them; it is only measured.
DEEP_TRAINING_KINDS = ("deep-half-a",)


def deep_pairs(
    manifest: Manifest, split: str, sets: set[str] | None = None, *, kinds: tuple[str, ...] = DEEP_KINDS
) -> list[dict[str, Any]]:
    """The pairs of halves of one split's deep stacks, of `kinds`: far less
    noisy than the noise pairs, with signal far further above their
    noise."""
    return [pair for kind in kinds for pair in pairs(manifest, split, sets, kind=kind)]


def dark_noise_bank(
    store: Store, manifest: Manifest, *, sets: set[str] | None = None, seed: int = 0, noise_map: str = "raw",
    sky: str = "global",
) -> dict[str, np.ndarray]:
    """Windows of the train split's dark-noise pairs: the input, its noise
    map and the deep stack it was made from, as `cut_windows` cuts them,
    with `sky` "flat" both less the input's smooth sky."""
    chosen = dark_noise.pairs(manifest, "train", sets)
    if not chosen:
        raise RuntimeError("no built dark-noise pairs in the train split")
    parts = ["dark-noise"] + [p["input"]["id"] for p in chosen] + [seed, WINDOW, WINDOWS_PER_HALF,
                                                                   map_part(noise_map)]
    if sky != "global":
        parts.append(f"sky-{sky}")
    key = hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16]

    def images():
        for pair in chosen:
            image, sigma = load_half(store, pair["input"], kind=noise_map)
            target, _ = load_half(store, pair["target"], noise_map=False)
            yield flattened(image, sigma, target, sky)

    return cut_windows(store, f"dark-noise-{key}", images(), sources=len(chosen), seed=seed)


BANK_PLANES = ("a", "sigma", "b")
# The plane a bank cut with a weight map holds as well: A's map of that kind.
WEIGHT_PLANE = "weight"
# The maps a network that reads one made from the image alone may still be
# trained against: its loss weighs each pixel by, and its outliers are
# scaled in, the stack's own noise, which only training has.
WEIGHT_MAPS = ("raw", "model")


class Windows:
    """A bank being cut, as `cut_windows` cuts it, one source at a time, so
    that one pass over the pairs can fill more than one: at most
    `per_source` windows of each, `sources` in all. `done` says the bank
    was finished before, and then `add` is not called. A `weighted` bank
    holds A's weight map as well. Given a `state` it recorded, a cut stopped
    part way goes on from there; `resumed` says it did."""

    def __init__(self, store: Store, name: str, *, sources: int, seed: int = 0,
                 per_source: int = WINDOWS_PER_HALF, weighted: bool = False,
                 resume: dict[str, Any] | None = None) -> None:
        self.directory = store.scratch / "tiles" / name
        self.done = (self.directory / "done.json").exists()
        self.sources, self.per_source = sources, per_source
        self.rng = random.Random(seed)
        self.background: list[float] = []
        self.noise: list[float] = []
        self.count = 0
        self.capacity = max(1, sources * per_source)
        self.resumed = False
        if self.done:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        files = [self.directory / f"{plane}.npy" for plane in BANK_PLANES + ((WEIGHT_PLANE,) if weighted else ())]
        self.resumed = resume is not None and all(file.exists() for file in files)
        self.planes = {
            file.stem: np.lib.format.open_memmap(file, mode="r+" if self.resumed else "w+", dtype=np.float32,
                                                 shape=(self.capacity, WINDOW, WINDOW))
            for file in files
        }
        if self.resumed:
            self.count = resume["count"]
            self.background, self.noise = list(resume["background"]), list(resume["noise"])
            version, internal, gauss = resume["rng"]
            self.rng.setstate((version, tuple(internal), gauss))

    def state(self) -> dict[str, Any]:
        """How far the cut has come, for a cut that goes on from here, with
        the windows cut so far on disk first."""
        for plane in self.planes.values():
            plane.flush()
        version, internal, gauss = self.rng.getstate()
        return {"count": self.count, "background": self.background, "noise": self.noise,
                "rng": [version, list(internal), gauss]}

    def add(self, image_a: np.ndarray, sigma_a: np.ndarray, image_b: np.ndarray,
            weight_a: np.ndarray | None = None) -> None:
        """Cut the windows of one `(image, noise map, target)`, and of A's
        weight map in a `weighted` bank. The background and median noise
        are the noise map's, the network's units."""
        b, s = scales(image_a, sigma_a)
        height, width = image_a.shape
        step = WINDOW // 2
        grid = [(y, x) for y in range(0, height - WINDOW, step) for x in range(0, width - WINDOW, step)]
        self.rng.shuffle(grid)
        kept = 0
        for y, x in grid:
            window_a = image_a[y:y + WINDOW, x:x + WINDOW]
            window_b = image_b[y:y + WINDOW, x:x + WINDOW]
            if not (np.isfinite(window_a).all() and np.isfinite(window_b).all()):
                continue
            if self.count == self.capacity:
                break
            self.planes["a"][self.count] = window_a
            self.planes["sigma"][self.count] = sigma_a[y:y + WINDOW, x:x + WINDOW]
            self.planes["b"][self.count] = window_b
            if WEIGHT_PLANE in self.planes:
                self.planes[WEIGHT_PLANE][self.count] = weight_a[y:y + WINDOW, x:x + WINDOW]
            self.background.append(b)
            self.noise.append(s)
            self.count += 1
            kept += 1
            if kept == self.per_source:
                break

    def finish(self) -> dict[str, np.ndarray]:
        """The bank's planes, as `open_bank` reads them, marked done."""
        done = self.directory / "done.json"
        if self.done:
            return open_bank(self.directory, json.loads(done.read_text())["windows"])
        for plane in self.planes.values():
            plane.flush()
        del self.planes
        np.save(self.directory / "background.npy", np.array(self.background, dtype=np.float32))
        np.save(self.directory / "noise.npy", np.array(self.noise, dtype=np.float32))
        done.write_text(json.dumps({"windows": self.count, "sources": self.sources}))
        self.done = True
        return open_bank(self.directory, self.count)


def cut_windows(store: Store, name: str, images: Any, *, sources: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Windows of each `(image, noise map, target)` that `images` yields, at
    most `sources` of them, on an overlapping grid and wholly inside both
    images' coverage, with each source's background and median noise.

    The bank is written window by window into memory-mapped `.npy` files in
    the local scratch directory under `name`, and read back the same way, so
    neither building nor training holds it in memory. `done.json` marks a
    finished bank; one without it is built again."""
    windows = Windows(store, name, sources=sources, seed=seed)
    if not windows.done:
        for image_a, sigma_a, image_b in images:
            windows.add(image_a, sigma_a, image_b)
    return windows.finish()


def open_bank(directory: Path, count: int) -> dict[str, np.ndarray]:
    """A finished bank's planes, memory-mapped and cut to its windows, its
    weight map's among them where it was cut with one."""
    planes = BANK_PLANES + ((WEIGHT_PLANE,) if (directory / f"{WEIGHT_PLANE}.npy").exists() else ())
    banked = {plane: np.load(directory / f"{plane}.npy", mmap_mode="r")[:count] for plane in planes}
    banked["background"] = np.load(directory / "background.npy")
    banked["noise"] = np.load(directory / "noise.npy")
    return banked


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def structure_odds(banked: dict[str, np.ndarray], factor: float) -> np.ndarray | None:
    """How often to draw each window: `1 + factor` times the share of its
    target's pixels over 20σ̄, normalised. Stars and nebulae fill a small
    share of the sky, so drawn uniformly the network rarely meets them."""
    if factor <= 0:
        return None
    count = len(banked["b"])
    share = np.empty(count)
    for start in range(0, count, 256):
        # A chunk at a time, since the bank is memory-mapped from disk.
        stop = min(count, start + 256)
        level = (np.asarray(banked["b"][start:stop]) - banked["background"][start:stop, None, None]) / banked["noise"][start:stop, None, None]
        share[start:stop] = (level > 20.0).mean(axis=(1, 2))
    odds = 1.0 + factor * share
    return odds / odds.sum()


# Outliers per this many pixels: `count` in `outliers` is per 256 × 256.
OUTLIER_AREA = 256 * 256
# What an outlier's amplitude is a multiple of: the image's typical noise
# σ̄, or the noise at the pixel itself.
OUTLIER_SCALES = ("typical", "local")
# The faintest hot pixel `outliers` plants, in σ̄. Near it a hot pixel looks
# like a faint star, so a model trained on many learns to take out some of
# the faint stars' flux too.
OUTLIER_FLOOR = 5.0


def lanczos3(offset: float) -> np.ndarray:
    """Lanczos-3 weights of the six pixels from two before to three after
    a point `offset` (0 to 1) past a pixel, summing to 1."""
    weights = np.sinc(np.arange(-2, 4) - offset) * np.sinc((np.arange(-2, 4) - offset) / 3)
    return weights / weights.sum()


def outliers(shape: tuple[int, int], rng: np.random.Generator, count: float, *,
             floor: float = OUTLIER_FLOOR) -> np.ndarray:
    """Outliers a few frames fail to reject, in units of σ̄, `count` per
    256 × 256 pixels on average: hot pixels (`floor` to 200σ̄), cold ones
    (−5 to −30σ̄) and cosmic-ray tracks 3 to 15 pixels long (twice `floor`
    to 300σ̄ along them). Each point is spread over the pixels around it as
    Lanczos-3 registration spreads a frame's pixel, at a random sub-pixel
    offset."""
    height, width = shape
    plane = np.zeros((height + 6, width + 6), dtype=np.float32)
    for _ in range(rng.poisson(count * height * width / OUTLIER_AREA)):
        y, x = rng.uniform(0, height), rng.uniform(0, width)
        kind = rng.random()
        if kind < 0.6:
            points, amplitude = [(y, x)], math.exp(rng.uniform(math.log(floor), math.log(200)))
        elif kind < 0.75:
            points, amplitude = [(y, x)], -math.exp(rng.uniform(math.log(5), math.log(30)))
        else:
            length, angle = rng.uniform(3, 15), rng.uniform(0, math.pi)
            steps = np.arange(int(length) + 1)
            points = list(zip(y + steps * math.sin(angle), x + steps * math.cos(angle)))
            amplitude = math.exp(rng.uniform(math.log(2 * floor), math.log(300)))
        for py, px in points:
            if not (0 <= py < height and 0 <= px < width):
                continue
            row, column = math.floor(py), math.floor(px)
            # The padding of 3 puts the tap two pixels before at row + 1.
            plane[row + 1:row + 7, column + 1:column + 7] += amplitude * np.outer(lanczos3(py - row), lanczos3(px - column))
    return plane[3:-3, 3:-3]


# How deep the spots `dark_spots` plants are, in σ̄, drawn log-uniformly:
# from the deepest cold pixel `outliers` plants, which a model learns to
# take out, to as deep as a stack's worst seen (−3,500σ̄, drizzled).
DARK_SPOT_DEPTHS = (30.0, 5000.0)
# The most pixels in one dark spot.
DARK_SPOT_PIXELS = 16


def dark_spots(shape: tuple[int, int], rng: np.random.Generator, count: float) -> np.ndarray:
    """Dark spots a stack holds in every frame, which no rejection takes
    out: hot pixels a dark subtracts too much of, dead ones a missing or
    poor flat leaves. In units of σ̄, `count` per 256 × 256 pixels on
    average. Each is a cluster of 1 to `DARK_SPOT_PIXELS` pixels grown from
    one, each 30 to 100% of the spot's depth (`DARK_SPOT_DEPTHS`); half are
    spread as Lanczos-3 registration spreads a frame's pixel, half left
    sharp, as drizzle leaves them. They are the image's, so a model should
    keep them as they are: one that fills them overshoots, and turns a
    hole into a bright spike ringed by speckle."""
    height, width = shape
    plane = np.zeros((height + 6, width + 6), dtype=np.float32)
    steps = ((0, 1), (0, -1), (1, 0), (-1, 0))
    for _ in range(rng.poisson(count * height * width / OUTLIER_AREA)):
        depth = math.exp(rng.uniform(math.log(DARK_SPOT_DEPTHS[0]), math.log(DARK_SPOT_DEPTHS[1])))
        size = int(math.exp(rng.uniform(0.0, math.log(DARK_SPOT_PIXELS + 1))))
        cells = [(int(rng.integers(0, height)), int(rng.integers(0, width)))]
        while len(cells) < size:
            y, x = cells[int(rng.integers(0, len(cells)))]
            dy, dx = steps[int(rng.integers(0, 4))]
            if (y + dy, x + dx) not in cells:
                cells.append((y + dy, x + dx))
        spread = rng.random() < 0.5
        offset_y, offset_x = rng.uniform(0, 1, 2)
        weights = np.outer(lanczos3(offset_y), lanczos3(offset_x)) if spread else None
        for y, x in cells:
            if not (0 <= y < height and 0 <= x < width):
                continue
            amplitude = -depth * rng.uniform(0.3, 1.0)
            if spread:
                plane[y + 1:y + 7, x + 1:x + 7] += amplitude * weights
            else:
                plane[y + 3, x + 3] += amplitude
    return plane[3:-3, 3:-3]


# A shifted or sharpened tile is filtered from this many pixels more on
# each side.
FILTER_MARGIN = 8
# The unsharp masks `sharpened` draws from: its Gaussian's width in pixels,
# and the share of the image less its blur it adds, drawn log-uniformly.
SHARPEN_WIDTHS = (0.7, 2.0)
SHARPEN_AMOUNTS = (0.2, 1.0)
# The boxes `spread_out` draws from, their width in pixels: a 2x drizzle
# drops each frame's pixel on a box about 2 output pixels wide, less with a
# smaller drop, a 3x one on about 3.
SPREAD_WIDTHS = (1.5, 3.0)
# The share of tiles given a sky plane, and its least offset and tilt in σ̄.
GRADIENT_SHARE = 0.5
GRADIENT_LEAST = 0.5


def taps(plane: np.ndarray, weights: np.ndarray, lead: int, start: int, size: int, axis: int) -> np.ndarray:
    """`size` pixels of `plane` along `axis` from `start`, each the sum of
    `weights` over the pixels from `lead` before it."""
    moved = np.moveaxis(plane, axis, 0)
    first = start - lead
    out = sum(weight * moved[first + i:first + i + size] for i, weight in enumerate(weights))
    return np.moveaxis(out, 0, axis)


def separable(plane: np.ndarray, weights: np.ndarray, lead: int, start: int, size: int) -> np.ndarray:
    """The `size` × `size` pixels of `plane` from (`start`, `start`), each
    filtered by `weights` down and across its last two axes."""
    down = taps(plane, weights[0], lead, start, size, -2)
    return taps(down, weights[1], lead, start, size, -1)


def through(planes: tuple[np.ndarray, ...], variance: np.ndarray, weights: tuple[np.ndarray, np.ndarray],
            margin: int, size: int) -> tuple[tuple[np.ndarray, ...], np.ndarray, float]:
    """`planes` through the separable `weights`, down and across from two
    pixels before, and their noise's `variance` (one map, or several
    stacked on a first axis) through the squared weights, with the share of
    white noise's variance that is left."""
    rows, columns = weights
    moved = tuple(separable(plane, weights, 2, margin, size) for plane in planes)
    squared = (rows**2, columns**2)
    return moved, separable(variance, squared, 2, margin, size), float(squared[0].sum() * squared[1].sum())


def shifted(planes: tuple[np.ndarray, ...], variance: np.ndarray, rng: np.random.Generator, margin: int,
            size: int) -> tuple[tuple[np.ndarray, ...], np.ndarray, float]:
    """`planes` moved by one sub-pixel offset through Lanczos-3, as
    registration resamples a frame, `through` its weights."""
    rows, columns = (lanczos3(offset) for offset in rng.random(2))
    return through(planes, variance, (rows, columns), margin, size)


def box_weights(width: float, offset: float) -> np.ndarray:
    """The share of a box `width` pixels wide (at most 3), centred `offset`
    (0 to 1) past a pixel, that falls on each of the five pixels from two
    before it to two after."""
    centres = np.arange(-2, 3, dtype=np.float64)
    low, high = offset - width / 2, offset + width / 2
    overlap = np.clip(np.minimum(centres + 0.5, high) - np.maximum(centres - 0.5, low), 0.0, None)
    return overlap / overlap.sum()


def spread_out(planes: tuple[np.ndarray, ...], variance: np.ndarray, rng: np.random.Generator, margin: int,
               size: int) -> tuple[tuple[np.ndarray, ...], np.ndarray, float]:
    """`planes` spread over a box `SPREAD_WIDTHS` wide at a random
    sub-pixel offset, as drizzle spreads a frame's pixel over the output
    pixels its drop covers, `through` the box's weights. Drizzle spreads
    the noise far more than registration does, and models that never saw
    it have thrown pixels thousands of σ̄ on a drizzle's edges."""
    rows, columns = (box_weights(rng.uniform(*SPREAD_WIDTHS), offset) for offset in rng.random(2))
    return through(planes, variance, (rows, columns), margin, size)


def sharpened(planes: tuple[np.ndarray, ...], variance: np.ndarray, rng: np.random.Generator, margin: int,
              size: int) -> tuple[tuple[np.ndarray, ...], np.ndarray, float]:
    """`planes` through one unsharp mask, `(1 + k) x − k G * x` for a
    Gaussian `G` and an amount `k` drawn from `SHARPEN_WIDTHS` and
    `SHARPEN_AMOUNTS`, and their noise's `variance` through the mask's
    squared weights, with the share of white noise's variance it makes."""
    width = rng.uniform(*SHARPEN_WIDTHS)
    amount = math.exp(rng.uniform(*np.log(SHARPEN_AMOUNTS)))
    reach_px = math.ceil(3 * width)
    gaussian = np.exp(-0.5 * (np.arange(-reach_px, reach_px + 1) / width) ** 2)
    gaussian /= gaussian.sum()
    inner = slice(margin, margin + size)
    moved = tuple((1 + amount) * plane[inner, inner] - amount * separable(plane, (gaussian, gaussian), reach_px,
                                                                          margin, size)
                  for plane in planes)
    # The mask's square is (1 + k)² − 2k(1 + k)G(0) at the centre and k²G²
    # throughout.
    centre = (1 + amount) ** 2 - 2 * amount * (1 + amount) * gaussian[reach_px] ** 2
    squared = gaussian**2
    carried = centre * variance[..., inner, inner] + amount**2 * separable(variance, (squared, squared),
                                                                            reach_px, margin, size)
    return moved, carried, float(centre + amount**2 * squared.sum() ** 2)


def sky_plane(size: int, rng: np.random.Generator, largest: float) -> np.ndarray:
    """A sky plane over a tile, in σ̄: an offset of either sign, and a tilt
    end to end across the tile in a random direction, each drawn
    log-uniformly from `GRADIENT_LEAST` to `largest` and half of it."""
    offset = math.copysign(math.exp(rng.uniform(math.log(GRADIENT_LEAST), math.log(largest))), rng.random() - 0.5)
    tilt = math.exp(rng.uniform(math.log(GRADIENT_LEAST), math.log(max(GRADIENT_LEAST, largest / 2))))
    angle = rng.uniform(0, 2 * math.pi)
    across = (np.arange(size, dtype=np.float32) - (size - 1) / 2) / size
    return (offset + tilt * (math.cos(angle) * across[None, :] + math.sin(angle) * across[:, None])).astype(np.float32)


def batch_from(
    banked: dict[str, np.ndarray],
    size: int,
    tile: int,
    rng: np.random.Generator,
    *,
    map_jitter: float = 0.0,
    odds: np.ndarray | None = None,
    outlier_count: float = 0.0,
    outlier_floor: float = OUTLIER_FLOOR,
    outlier_scale: str = "typical",
    noise_plane: str = "linear",
    resample: float = 0.0,
    sharpen: float = 0.0,
    gradient: float = 0.0,
    dark_spot_count: float = 0.0,
    spread: float = 0.0,
) -> tuple[torch.Tensor, ...]:
    """`size` tiles at random windows and offsets, each flipped and turned
    one of eight ways: the network's input (its noise map as `noise_plane`
    says), the target and A, in units of σ̄ above the background, and A's
    noise map in units of σ̄. With
    `map_jitter`, the noise map the network reads is scaled by a factor
    drawn log-uniformly from `exp(±map_jitter)` per tile, so the network
    learns to judge the noise from the image, the map telling it only
    where the noise is higher or lower. With `outlier_count`, A gains
    `outliers` from `outlier_floor` up and the target does not; the noise
    map, which a modelled map would not change, stays as it was. Their
    amplitudes are in σ̄, or with `outlier_scale` "local" in the noise at
    each pixel: on a bright nebula, whose noise is many times σ̄, an
    outlier of 10σ̄ is a bump of ordinary noise, and a model taught to take
    those out shaves the noise's peaks and dims what is bright.

    The halves are changed alike, after the outliers, which keeps their
    noise independent: with `dark_spot_count`, both gain the same
    `dark_spots` that many per 256 × 256 pixels, so the target keeps them;
    `resample` of the tiles are `shifted`, `sharpen` of them `sharpened`
    and `spread` of them `spread_out`, the noise map and σ̄ following, and
    with
    `gradient`, `GRADIENT_SHARE` of them gain a `sky_plane` up to that many
    σ̄ under the input. The target would gain it too, but the loss, which
    takes the input from its estimate, would not see it there, and the
    loss's weights are kept on the target's own brightness.

    A bank cut with a weight map (see `bank`) gives the noise map returned
    for the loss, and the local noise the outliers are scaled in, from that
    map; the network still reads its own."""
    count = len(banked["a"])
    picks = rng.choice(count, size, p=odds) if odds is not None else rng.integers(0, count, size)
    margin = FILTER_MARGIN if resample or sharpen or spread else 0
    span = tile + 2 * margin
    weighted = WEIGHT_PLANE in banked
    inputs, targets, references, sigmas = [], [], [], []
    for index in picks:
        y, x = rng.integers(0, WINDOW - span + 1, 2)
        a = banked["a"][index, y:y + span, x:x + span]
        sigma = banked["sigma"][index, y:y + span, x:x + span]
        b = banked["b"][index, y:y + span, x:x + span]
        background, noise = float(banked["background"][index]), float(banked["noise"][index])
        turns, flip = int(rng.integers(0, 4)), bool(rng.integers(0, 2))
        held = [banked[WEIGHT_PLANE][index, y:y + span, x:x + span]] if weighted else []
        planes = np.stack([a, sigma, b, *held])
        planes = np.rot90(planes, turns, axes=(1, 2))
        if flip:
            planes = planes[:, :, ::-1]
        a, sigma, b = planes[:3]
        weight = planes[3] if weighted else sigma
        if outlier_count:
            scale = np.where(np.isfinite(weight), weight, noise) if outlier_scale == "local" else noise
            a = a + outliers(a.shape, rng, outlier_count, floor=outlier_floor) * scale
        if dark_spot_count:
            spots = dark_spots(a.shape, rng, dark_spot_count) * noise
            a, b = a + spots, b + spots
        if margin:
            choice = rng.random()
            change = (shifted if choice < resample else sharpened if choice < resample + sharpen
                      else spread_out if choice < resample + sharpen + spread else None)
            if change is None:
                inner = slice(margin, margin + tile)
                a, sigma, b, weight = a[inner, inner], sigma[inner, inner], b[inner, inner], weight[inner, inner]
            else:
                maps = (sigma, weight) if weighted else (sigma,)
                variance = np.stack([np.where(np.isfinite(plane), plane, noise) ** 2 for plane in maps])
                (a, b), variance, share = change((a, b), variance, rng, margin, tile)
                a, b = a.astype(np.float32), b.astype(np.float32)
                carried = np.sqrt(np.clip(variance, 0.0, None)).astype(np.float32)
                sigma, weight = carried if weighted else (carried[0], carried[0])
                noise *= math.sqrt(share)
        stated = sigma * float(np.exp(rng.uniform(-map_jitter, map_jitter))) if map_jitter else sigma
        shown = a
        if gradient and rng.random() < GRADIENT_SHARE:
            shown = a + noise * sky_plane(tile, rng, gradient)
        inputs.append(network_input(shown, stated, background, noise, noise_plane))
        targets.append(np.stack([(b - background) / noise]))
        references.append(np.stack([(a - background) / noise]))
        sigmas.append(np.stack([np.nan_to_num(weight / noise, nan=1.0)]))
    return tuple(
        torch.from_numpy(np.ascontiguousarray(np.stack(values)))
        for values in (inputs, targets, references, sigmas)
    )


class Prefetch:
    """`make(step)` for steps `start` + 1, `start` + 2, …, built up to
    `depth` ahead on `threads` threads and handed out in step order. A batch
    depends only on its step, so the batches are those calling `make` in
    the loop would give, whatever the threads, and a run resumed after
    `start` meets the batches it would have met. NumPy releases the
    interpreter lock for the copies and the arithmetic, which is most of a
    batch."""

    def __init__(self, make: Callable[[int], Any], depth: int, threads: int = 1, start: int = 0) -> None:
        self.make = make
        self.pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="batch")
        self.pending: deque[Future] = deque()
        self.step = start
        for _ in range(max(depth, threads)):
            self.submit()

    def submit(self) -> None:
        self.step += 1
        self.pending.append(self.pool.submit(self.make, self.step))

    def get(self) -> Any:
        """The next step's batch, or the error that building it raised."""
        item = self.pending.popleft().result()
        self.submit()
        return item

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)


def step_generator(seed: int, step: int) -> np.random.Generator:
    """The random generator of one training step's batch."""
    return np.random.default_rng([seed, step])


# Whole-image evaluation runs this many pairs at once; their GPU calls take
# turns under `GPU`, and their loading and NumPy work overlap.
EVALUATION_THREADS = 4
GPU = threading.Lock()


PRECISIONS = ("fp32", "bf16")
LOSS_WEIGHTS = ("core", "variance")


def loss_of(
    model: ResidualUNet,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    references: torch.Tensor,
    sigmas: torch.Tensor,
    weighting: str = "core",
    huber: float = 0.0,
    huber_above: float = 0.0,
    scales: tuple[int, ...] = (),
    scale_weight: float = 1.0,
) -> torch.Tensor:
    """L2 of `A + residual` against the target, both in units of σ̄ above
    the background. `core` weights star cores down by their brightness;
    `variance` weights every pixel by the inverse of A's noise variance
    there, so a pixel counts by how well A knew it.

    With `scales`, the weighted error is also averaged over k × k blocks
    for each k there, times k, and its mean square added, `scale_weight`
    times. A pixel's noise lies mostly at the finest scale, so an error
    there costs the per-pixel loss far more than noise left over 8 or 16
    pixels, which a model then leaves, and which shows as mottle once the
    grain is gone. Times k, the blocks' noise weighs as the pixels' does,
    so every scale's error counts alike. A block average of the target is
    still unbiased, so the clean image is still what the loss converges
    on. With `huber`, each block's term is robust in turn, so a star core
    or an outlier cannot rule it; the error is not clipped before the
    average, which would let a model meet a block's mean with errors of
    alternating sign, each past the clip: one so trained striped star cores
    and bright nebulae column by column and dug rings around stars. With
    `huber_above`, a block holding any pixel whose target is that bright is
    left out of the scale terms, as a star core's seeing differs between
    the halves; mottle shows on the sky and faint signal.

    With `huber`, each pixel's weighted error is squared only within
    `huber` of zero and grows linearly beyond it. Noise stays in the
    squared part, which a noise-pair target needs to converge on the clean
    image; a star core whose seeing differs between the halves, or a pixel
    only A failed to reject, cannot rule the loss.

    With `huber_above` as well, only pixels whose target is that many σ̄
    above the background are robust. A pixel only A failed to reject sits
    over sky in the target, so it keeps the squared loss, and the network
    learns to remove it; a star core, whose error is mostly the halves'
    seeing, stays robust."""
    estimate = references + model(inputs)
    if weighting == "variance":
        weight = 1.0 / sigmas.clamp(min=0.25) ** 2
    else:
        weight = 1.0 / (1.0 + (targets.abs() / CORE_SIGMA) ** 2)
    error = (estimate - targets) * weight.sqrt()
    if huber > 0:
        robust = torch.nn.functional.huber_loss(error, torch.zeros_like(error), delta=huber, reduction="none") * 2.0
        if huber_above > 0:
            robust = torch.where(targets.abs() > huber_above, robust, error**2)
        loss = robust.mean()
    else:
        loss = (error**2).mean()
    bright = (targets.abs() > huber_above).float() if huber > 0 and huber_above > 0 else None
    for k in scales:
        block = k * torch.nn.functional.avg_pool2d(error, k)
        if huber > 0:
            term = torch.nn.functional.huber_loss(block, torch.zeros_like(block), delta=huber, reduction="none") * 2.0
        else:
            term = block**2
        if bright is not None:
            term = torch.where(torch.nn.functional.max_pool2d(bright, k) > 0, torch.zeros_like(term), term)
        loss = loss + scale_weight * term.mean()
    return loss


class Rewind:
    """Puts a run's model and optimizer back where they were two log windows
    before when a window's mean loss is not finite, or is more than `factor`
    times the lowest so far; training then goes on with the steps' own
    batches, so it does not meet the same ones again. With no normalisation,
    weights that blow up can kill the network for good: every channel of one
    block shut, the output a constant and the loss back at the input's own,
    the gradient too small ever to bring it back.

    Each rewind also takes `damping` off the learning rate for the rest of
    the run (`rate`, the share of the schedule's the run now takes): a run
    put back at the same rate can blow up again within a window, as one at
    160,000 steps did. More than `most` rewinds within `within` steps and
    the run stops; a long run may meet more spikes in all than a short one."""

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, factor: float, most: int = 6,
                 within: int = 20_000, damping: float = 0.8):
        self.model, self.optimizer, self.factor, self.most = model, optimizer, factor, most
        self.within, self.damping = within, damping
        self.lowest = math.inf
        self.count = 0
        self.rate = 1.0
        self.recent: list[int] = []
        self.saved: list[tuple[int, dict, dict]] = []

    def after(self, step: int, mean: float) -> int | None:
        """At the end of the window to `step`, with its mean loss: keep the
        weights if the window was sound, or go back. Returns the step gone
        back to, if it went back."""
        if math.isfinite(mean) and mean <= self.factor * self.lowest:
            self.lowest = min(self.lowest, mean)
            self.saved = [*self.saved, (step, offloaded(self.model.state_dict()),
                                        offloaded(self.optimizer.state_dict()))][-2:]
            return None
        if not self.saved:
            raise RuntimeError(f"the loss was {mean:.4g} at step {step}, before any sound window to go back to")
        self.count += 1
        self.recent = [*(back for back in self.recent if back > step - self.within), step]
        if len(self.recent) > self.most:
            raise RuntimeError(f"the loss was {mean:.4g} at step {step}, after {self.most} rewinds "
                               f"within {self.within} steps")
        self.rate *= self.damping
        back, model_state, optimizer_state = self.saved[0]
        self.model.load_state_dict(model_state)
        # A copy: on the host the optimizer would keep the saved tensors as
        # its own, and step them on.
        self.optimizer.load_state_dict(offloaded(optimizer_state))
        self.saved = self.saved[:1]
        return back

    def state(self) -> dict[str, Any]:
        """What a run's checkpoint keeps of the guard, so a resumed run
        rewinds as the run it continues would have."""
        return {"lowest": self.lowest, "count": self.count, "rate": self.rate, "recent": list(self.recent),
                "saved": list(self.saved)}

    def restore(self, state: dict[str, Any]) -> None:
        """Put the guard back as `state` recorded it."""
        self.lowest, self.count, self.rate = state["lowest"], state["count"], state["rate"]
        self.recent, self.saved = list(state["recent"]), [tuple(entry) for entry in state["saved"]]


def offloaded(state: Any) -> Any:
    """A copy of a state dict in host memory."""
    if isinstance(state, torch.Tensor):
        return state.detach().to("cpu", copy=True)
    if isinstance(state, dict):
        return {key: offloaded(value) for key, value in state.items()}
    if isinstance(state, (list, tuple)):
        return type(state)(offloaded(value) for value in state)
    return state


def train(
    store: Store,
    manifest: Manifest,
    *,
    name: str,
    steps: int,
    width: int = 16,
    levels: int = 4,
    tile: int = 256,
    batch: int = 16,
    learning_rate: float = 2e-4,
    sets: set[str] | None = None,
    seed: int = 0,
    log_every: int = 250,
    extra: dict[str, np.ndarray] | None = None,
    extra_fraction: float = 0.0,
    extra_label: str | None = None,
    loss_weight: str = "core",
    map_jitter: float = 0.0,
    validation_sets: set[str] | None = None,
    noise_map: str = "raw",
    huber: float = 0.0,
    huber_above: float = 0.0,
    structure_sampling: float = 0.0,
    outlier_count: float = 0.0,
    outlier_floor: float = OUTLIER_FLOOR,
    outlier_scale: str = "typical",
    noise_plane: str = "linear",
    clip_grad: float = 0.0,
    match_flux: bool = False,
    agree: float | None = None,
    precision: str = "fp32",
    prefetch: int = 8,
    batch_threads: int = 4,
    scale_loss: tuple[int, ...] = (),
    scale_weight: float = 1.0,
    rewind: float = 0.0,
    sky: str = "global",
    resample: float = 0.0,
    sharpen: float = 0.0,
    gradient: float = 0.0,
    bin_fraction: float = 0.0,
    dark_spot_count: float = 0.0,
    spread: float = 0.0,
    weight_map: str | None = None,
    holes: bool = False,
    bank_workers: int = 1,
) -> dict[str, Any]:
    """Train a noise model on the train split's pairs, evaluate it on the
    validation split's, export it to ONNX and record the run.

    `scale_loss` adds the error averaged over blocks of each of its sides,
    `scale_weight` times each, to the loss (see `loss_of`). `rewind` goes
    back two log windows when one's loss is more than that many times the
    lowest, each time at a lower learning rate (see `Rewind`; 0: never),
    and the history records where.

    The network reads the sky as `sky` says (see `SKIES`), in training,
    evaluation and the sheets. `resample`, `sharpen`, `spread`, `gradient`
    and `dark_spot_count` change both halves of the tiles alike (see
    `batch_from`), and `bin_fraction` of every batch comes from the pairs
    binned 2 × 2, `BINNED_WINDOWS` of each cut with the bank. The record
    holds the trained model's `stability` on spread noise. With
    `weight_map`, a network that reads a `noise_map` made from the image
    alone is trained against the stack's own noise all the same: the loss
    weighs by, and the outliers are scaled in, A's map of that kind, and
    the halves are matched and judged by it (see `bank`). With `holes`, the
    network trains and is measured as the runtime's hole guard runs it: it
    never sees an image's holes, and they are put back after (see
    `guarded_pair`). `bank_workers` pairs are prepared at once when the bank
    is cut.

    `extra` is a second bank from `cut_windows`, of other targets for
    their inputs; it makes up `extra_fraction` of every batch, and the run
    records it under `extra_label`. `outlier_count` adds that many
    synthetic outliers per 256 × 256 pixels to every input tile, the hot
    ones from `outlier_floor` up, in units `outlier_scale` names (see
    `batch_from`). The network reads the noise map as `noise_plane` says
    (see `NOISE_PLANES`). `clip_grad` scales each step's gradient
    down to that norm where it is longer, so one wild batch cannot throw
    the weights far, and records the norms it saw in the history.
    `match_flux` trains on B halves `matched` to their A halves' flux, and
    `agree` leaves out the pairs whose halves then `disagree` by more than
    that; the run's own validation measures against B as it is, as every
    run's does.
    `precision` "bf16" runs the network in bfloat16 under autocast.
    `prefetch` batches are built ahead on `batch_threads` threads; 0 builds
    each in turn. Each step's batch draws from `step_generator(seed, step)`,
    so it is the same either way. (Runs before this drew every batch from
    one generator in turn.)"""
    if precision not in PRECISIONS:
        raise ValueError(f"precision is one of {PRECISIONS}, not {precision!r}")
    if outlier_scale not in OUTLIER_SCALES:
        raise ValueError(f"outlier_scale is one of {OUTLIER_SCALES}, not {outlier_scale!r}")
    if noise_plane not in NOISE_PLANES:
        raise ValueError(f"noise_plane is one of {NOISE_PLANES}, not {noise_plane!r}")
    if sky not in SKIES:
        raise ValueError(f"sky is one of {SKIES}, not {sky!r}")
    if resample < 0 or sharpen < 0 or spread < 0 or resample + sharpen + spread > 1:
        raise ValueError(f"the shares of tiles shifted ({resample}), sharpened ({sharpen}) and spread ({spread}) "
                         "add up to at most 1")
    if (resample or sharpen or spread) and tile + 2 * FILTER_MARGIN > WINDOW:
        raise ValueError(f"a shifted, sharpened or spread tile reads {FILTER_MARGIN} pixels more on each side: "
                         f"the tile, {tile}, is at most {WINDOW - 2 * FILTER_MARGIN}")
    if any(k < 2 or tile % k for k in scale_loss):
        raise ValueError(f"each of the loss's scales {scale_loss} must be at least 2 and divide the tile, {tile}")
    if extra is not None and extra_fraction and not len(extra["a"]):
        raise RuntimeError(f"the {extra_label} bank holds no window wholly inside its pairs' coverage")
    torch.manual_seed(seed)
    chosen = pairs(manifest, "train", sets)
    if not chosen:
        raise RuntimeError("no built noise pairs in the train split")
    started = time.monotonic()
    banked = bank(store, chosen, seed=seed, noise_map=noise_map, match_flux=match_flux, agree=agree, sky=sky,
                  binned_windows=BINNED_WINDOWS if bin_fraction else 0, weight_map=weight_map, holes=holes,
                  workers=bank_workers)
    if bin_fraction and not len(banked["binned"]["a"]):
        raise RuntimeError("the binned bank holds no window wholly inside its pairs' coverage")
    bank_seconds = time.monotonic() - started
    where = device()
    cuda = where.type == "cuda"
    layout = torch.channels_last if cuda else torch.contiguous_format
    model = ResidualUNet(width=width, levels=levels).to(where, memory_format=layout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=steps, pct_start=0.05)
    guard = Rewind(model, optimizer, rewind) if rewind else None
    history = []
    extra_count = round(batch * extra_fraction) if extra is not None else 0
    binned_count = round(batch * bin_fraction)
    odds = structure_odds(banked, structure_sampling)
    extra_odds = structure_odds(extra, structure_sampling) if extra is not None else None
    binned_odds = structure_odds(banked["binned"], structure_sampling) if binned_count else None
    changes = {"map_jitter": map_jitter, "outlier_count": outlier_count, "outlier_floor": outlier_floor,
               "outlier_scale": outlier_scale, "noise_plane": noise_plane, "resample": resample,
               "sharpen": sharpen, "gradient": gradient, "dark_spot_count": dark_spot_count, "spread": spread}

    def assemble(step: int) -> tuple[torch.Tensor, ...]:
        rng = step_generator(seed, step)
        parts = [batch_from(banked, batch - extra_count - binned_count, tile, rng, odds=odds, **changes)]
        if extra_count:
            parts.append(batch_from(extra, extra_count, tile, rng, odds=extra_odds, **changes))
        if binned_count:
            parts.append(batch_from(banked["binned"], binned_count, tile, rng, odds=binned_odds, **changes))
        tensors = tuple(torch.cat(values) for values in zip(*parts))
        # Pinned, a batch copies to the GPU while it still trains.
        return tuple(t.pin_memory() for t in tensors) if cuda else tensors

    source = Prefetch(assemble, prefetch, batch_threads) if prefetch else None
    started = time.monotonic()
    # Summed on the device, so a step never waits for the GPU to finish.
    running = torch.zeros((), device=where)
    # Steps in the log window so far: the last is shorter when the steps
    # are not a whole number of windows.
    counted = 0
    # With clipping, the gradient's norm before it: the sum and the largest.
    norms = torch.zeros(2, device=where)
    try:
        for step in range(1, steps + 1):
            tensors = source.get() if source else assemble(step)
            inputs, targets, references, sigmas = (t.to(where, non_blocking=True) for t in tensors)
            inputs = inputs.contiguous(memory_format=layout)
            with torch.autocast(where.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                loss = loss_of(model, inputs, targets, references, sigmas, loss_weight, huber, huber_above,
                               scale_loss, scale_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                norms[0] += norm
                norms[1] = torch.maximum(norms[1], norm)
            optimizer.step()
            schedule.step()
            if guard and guard.rate != 1.0:
                # The schedule sets each step's rate afresh, so this does
                # not compound.
                for group in optimizer.param_groups:
                    group["lr"] *= guard.rate
            running += loss.detach().float()
            counted += 1
            if step % log_every == 0 or step == steps:
                mean = float(running) / counted
                entry = {"step": step, "loss": mean, "seconds": round(time.monotonic() - started, 1)}
                noted = ""
                if clip_grad:
                    entry["grad_norm_mean"], entry["grad_norm_max"] = float(norms[0]) / counted, float(norms[1])
                    noted = f" gradient {entry['grad_norm_mean']:.3g} (largest {entry['grad_norm_max']:.3g})"
                    norms.zero_()
                back = guard.after(step, mean) if guard else None
                if back is not None:
                    entry["rewound_to"] = back
                    noted += f", back to step {back}"
                history.append(entry)
                print(f"train {name}: step {step} loss {mean:.4f}{noted} ({time.monotonic() - started:.0f} s)",
                      flush=True)
                running.zero_()
                counted = 0
    finally:
        if source:
            source.close()
    train_seconds = time.monotonic() - started

    directory = store.root / "models" / "noise" / name
    directory.mkdir(parents=True, exist_ok=True)
    model = model.cpu().to(memory_format=torch.contiguous_format).eval()
    torch.save(model.state_dict(), directory / "weights.pt")
    inference = tiling(levels)
    validation = evaluate(store, manifest, model, split="validation", **inference, sets=validation_sets,
                          noise_map=noise_map, noise_plane=noise_plane, sky=sky, holes=holes)
    validation["dark_noise"] = evaluate_dark_noise(store, manifest, model, split="validation", **inference,
                                                   sets=validation_sets, noise_map=noise_map,
                                                   noise_plane=noise_plane, sky=sky, holes=holes)
    steady = stability(model, inference, sky=sky, noise_plane=noise_plane)
    print(f"train {name}: largest step on spread noise {steady['largest']:.0f}σ̄"
          f"{'' if steady['stable'] else ', a blow-up'}", flush=True)
    exported = export(model, directory / "model.onnx", tile=512)
    record = {
        "name": name,
        "created": now(),
        "config": {"width": width, "levels": levels, "tile": tile, "batch": batch, "steps": steps,
                   "learning_rate": learning_rate, "seed": seed,
                   "train_pairs": len(chosen) - len(banked["dropped"]), "pairs_dropped": banked["dropped"],
                   "windows": int(len(banked["a"])), "core_sigma": CORE_SIGMA,
                   "loss_weight": loss_weight, "map_jitter": map_jitter, "noise_map": noise_map,
                   "weight_map": weight_map, "holes": holes,
                   "huber": huber, "huber_above": huber_above, "structure_sampling": structure_sampling,
                   "outliers": outlier_count, "outlier_floor": outlier_floor,
                   "outlier_scale": outlier_scale, "noise_plane": noise_plane, "clip_grad": clip_grad,
                   "match_flux": match_flux, "agree": agree, "precision": precision, "prefetch": prefetch,
                   "batch_threads": batch_threads, "batch_generator": "per-step",
                   "scale_loss": list(scale_loss), "scale_weight": scale_weight if scale_loss else None,
                   "rewind": rewind, "rewinds": guard.count if guard else 0,
                   "rewind_rate": guard.rate if guard else 1.0, "sky": sky, "resample": resample,
                   "sharpen": sharpen, "gradient": gradient, "bin_fraction": bin_fraction,
                   "dark_spots": dark_spot_count, "spread": spread,
                   "binned_windows": int(len(banked["binned"]["a"])) if bin_fraction else None,
                   "extra": {"label": extra_label, "fraction": extra_fraction, "windows": int(len(extra["a"]))}
                   if extra is not None else None},
        "parameters": sum(p.numel() for p in model.parameters()),
        "receptive_field_px": receptive_field(levels),
        "inference": inference,
        "device": str(where),
        "seconds": {"bank": round(bank_seconds, 1), "train": round(train_seconds, 1)},
        "history": history,
        "validation": validation,
        "stability": steady,
        "export": exported,
        "versions": versions(),
    }
    (directory / "run.json").write_text(json.dumps(record, indent=2))
    return record


def denoiser(model: ResidualUNet, background: float, noise: float, batch_device: torch.device | None = None):
    """A function from network-input tiles to the residual in linear units."""
    where = batch_device or torch.device("cpu")
    with GPU:
        # Under the lock, so a thread moving the model never meets another's
        # forward pass.
        model = model.to(where)

    def run(tiles: np.ndarray) -> np.ndarray:
        with GPU, torch.no_grad():
            residual = model(torch.from_numpy(tiles).to(where)).cpu().numpy()
        return residual * noise

    return run


@dataclass(frozen=True)
class NoiseBands:
    """How much of the noise a model finds is taken out, as the runtime's
    `NoiseBands`. The removed noise `r = x − f(x)` is split by a Gaussian of
    `scale_px` into the low band, the mottle, and the high band, the grain
    left over; `high` and `low` of each are taken out. Ones take out all
    of it. Keeping a share `k` of white noise leaves an image as noisy as a
    stack `1/k²` as deep."""

    high: float = 1.0
    low: float = 1.0
    scale_px: float = 5.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.high <= 1.0 and 0.0 <= self.low <= 1.0 and self.scale_px > 0):
            raise ValueError(f"amounts run from 0 to 1 and the scale is positive: {self}")

    @property
    def whole(self) -> bool:
        return self.high == 1.0 and self.low == 1.0

    def __str__(self) -> str:
        return f"high {self.high:g} low {self.low:g} at {self.scale_px:g} px"

    def apply(self, image: np.ndarray, residual: np.ndarray) -> np.ndarray:
        """`image` with these shares of the model's `residual`, `f(x) − x`,
        added: `x − a_high·r_high − a_low·r_low`."""
        if self.high == self.low:
            return image + self.high * residual
        return image + self.high * residual + (self.low - self.high) * gaussian_blur(residual, self.scale_px)


def measure(estimate: np.ndarray, a: np.ndarray, b: np.ndarray, background: float, noise: float) -> dict[str, float]:
    """How close `estimate` comes to B, against how close A came, and
    `signal_kept`'s measures of the signal it keeps, B's noise being
    independent of A's."""
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(estimate)
    before = np.where(valid, a - b, np.nan)
    after = np.where(valid, estimate - b, np.nan)
    floor_high = robust_sigma(np.where(valid, b, np.nan)) or 0.0
    result = {}
    for band, transform in (("high", lambda image: image), ("low", lambda image: binned(image, BAND_SCALE_PX))):
        sigma_before = robust_sigma(transform(before))
        sigma_after = robust_sigma(transform(after))
        result[f"{band}_before"] = sigma_before
        result[f"{band}_after"] = sigma_after
        result[f"{band}_ratio"] = sigma_after / sigma_before if sigma_before else None
    result["noise_floor_high"] = floor_high
    excess_before = result["high_before"] ** 2 - floor_high**2
    excess_after = result["high_after"] ** 2 - floor_high**2
    result["high_removed"] = 1.0 - excess_after / excess_before if excess_before > 0 else None
    # Over every pixel, stars included: what blurring or ringing would cost.
    result["mse_before"] = float(np.nanmean((before / noise) ** 2))
    result["mse_after"] = float(np.nanmean((after / noise) ** 2))
    sky = valid & (np.abs(b - background) < 3 * noise)
    result["sky_bias"] = float(np.median((estimate - a)[sky]) / noise)
    # Error against B by B's brightness. On sky, the worst 0.1% of pixels
    # are mostly what a short stack could not reject — hot pixels, cosmic
    # rays, trails — and are reported apart from the rest.
    level = (b - background) / noise
    squared = np.where(valid, ((estimate - b) / noise) ** 2, np.nan)
    sky_errors = np.sort(squared[sky])
    keep = len(sky_errors) - max(1, len(sky_errors) // 1000)
    result["sky_mse"] = float(np.mean(sky_errors))
    result["sky_mse_trimmed"] = float(np.mean(sky_errors[:keep]))
    result["sky_outlier_share"] = float(np.sum(sky_errors[keep:]) / np.sum(sky_errors))
    for region, low, high in (("faint", 3, 20), ("mid", 20, 200), ("core", 200, np.inf)):
        mask = valid & (level >= low) & (level < high)
        result[f"{region}_mse"] = float(np.mean(squared[mask])) if mask.any() else None
    bright = valid & (b - background > 20 * noise)
    signal = float(np.sum((a - background)[bright]))
    result["bright_flux_change"] = float(np.sum((estimate - a)[bright]) / signal) if signal else None
    result.update(signal_kept(estimate, a, b, valid, background, noise))
    return result


def evaluate(
    store: Store,
    manifest: Manifest,
    model: ResidualUNet,
    *,
    split: str,
    tile: int = 512,
    overlap: int = 136,
    margin: int = 52,
    multiple: int | None = None,
    limit: int | None = None,
    sets: set[str] | None = None,
    noise_map: str = "raw",
    noise_plane: str = "linear",
    bands: NoiseBands = NoiseBands(),
    match_flux: bool = False,
    threads: int = EVALUATION_THREADS,
    sky: str = "global",
    holes: bool = False,
) -> dict[str, Any]:
    """Denoise each of a split's A halves, whole, through overlapping
    tiles that start on the model's grid, or on multiples of `multiple`
    when given, take out `bands` of what the model finds, and measure it
    against its B half. The deep stacks' halves are measured the same way, one
    against the other, under the size `deep-half`, or `multi-night` for an
    object stacked over its nights: they show the model at a night's whole
    depth and beyond, far less noisy than the noise pairs. The
    network reads the noise map it was trained with, as `noise_plane`
    says, and the sky as `sky` does; every run is measured
    in the same units, the per-pixel map's median noise, so runs with
    different maps compare. With `match_flux`, each B is first `matched`
    to its A, so a model that keeps A's flux is not marked down for the
    halves' own difference, and each pair's gain and `agreement` are
    recorded with it. With `holes`, the network never sees an image's
    holes, and they are put back after, as the runtime's hole guard runs
    it. `threads` pairs are measured at once."""
    from .holes import HoleGuard, guarded, guarded_residual

    guard = HoleGuard() if holes else None
    model = model.eval()
    where = device()

    def one(pair: dict[str, Any]) -> dict[str, Any]:
        image_a, sigma_a = load_half(store, pair["a"])
        image_b, _ = load_half(store, pair["b"], noise_map=False)
        background, noise = scales(image_a, sigma_a)
        gain, found = None, {}
        if match_flux:
            image_b, gain = matched(image_a, image_b, noise)
            found = {f"flux_{band}": value for band, value in agreement(image_a, image_b, noise).items()}
        coverage = np.load(store.root / loads(pair["a"]["files"])["coverage"])
        sigma_a = input_sigma(noise_map, image_a, sigma_a, coverage)
        settings = {"tile": tile, "overlap": overlap, "margin": margin, "multiple": multiple or model.multiple}

        def denoise(image: np.ndarray, sigma: np.ndarray) -> np.ndarray:
            shown, mask, mapped = guarded(image, guard) if guard else (image, None, None)
            if mask is not None and mask.any() and noise_map in ("image", "constant"):
                sigma = input_sigma(noise_map, shown, None, None)
            planes, image_background, image_noise = view(shown, sigma, sky=sky, noise_plane=noise_plane)
            found = infer(denoiser(model, image_background, image_noise, where), planes, **settings)[0]
            if mask is not None:
                found = guarded_residual(image, shown, found, mask, mapped, guard)
            return bands.apply(image, found)

        estimate = denoise(image_a, sigma_a)
        recipe = loads(pair["a"]["recipe"])
        size = recipe.get("size") or recipe["recipe"]
        left = {}
        # A noise pair's halves are independent, so B denoised too shows
        # the noise left at each scale; a deep stack's halves share the
        # night's calibration.
        if isinstance(size, int):
            alone_b, sigma_b = load_half(store, pair["b"])
            sigma_b = input_sigma(noise_map, alone_b, sigma_b,
                                  np.load(store.root / loads(pair["b"]["files"])["coverage"]))
            left = noise_left(estimate, denoise(alone_b, sigma_b), image_a, alone_b, noise)
        return {"pair": pair["a"]["id"], "set": pair["set"], "size": size,
                "flux_gain": gain, **found, **measure(estimate, image_a, image_b, background, noise), **left}

    chosen = pairs(manifest, split, sets) + deep_pairs(manifest, split, sets)
    with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="evaluate") as pool:
        per_pair = list(pool.map(one, chosen[:limit]))
    by_size: dict[Any, list[dict[str, Any]]] = {}
    for row in per_pair:
        by_size.setdefault(row["size"], []).append(row)

    def mean(rows, key):
        values = [row[key] for row in rows if row.get(key) is not None]
        return float(np.mean(values)) if values else None

    keys = ("high_ratio", "low_ratio", "high_removed", "mse_before", "mse_after", "sky_bias", "bright_flux_change",
            "sky_mse", "sky_mse_trimmed", "sky_outlier_share", "faint_mse", "mid_mse", "core_mse",
            *(f"transfer_{region}_{band}" for region in ("faint", "mid") for band in TRANSFER_BANDS),
            "peak_flux_kept_faint", "peak_flux_kept_mid", "peak_height_kept_faint", "peak_height_kept_mid",
            "star_fwhm_px", *(f"star_fwhm_{kind}_{level}" for kind in ("ratio", "target") for level in STAR_LEVELS),
            *(f"noise_left_{scale}px" for scale in MOTTLE_SCALES))
    return {
        "split": split,
        "bands": asdict(bands),
        "match_flux": match_flux,
        "pairs": len(per_pair),
        # Frame counts first, then the deep halves.
        "by_size": {str(size): {key: mean(rows, key) for key in keys}
                    for size, rows in sorted(by_size.items(), key=lambda item: (isinstance(item[0], str), item[0]))},
        "pairs_detail": per_pair,
    }


# Signal-transfer bands: the image box-blurred at the first radius less the
# image box-blurred at the second, radius 0 being the image itself.
TRANSFER_BANDS = {"fine": (0, 1), "medium": (1, 4), "coarse": (4, 12)}
# The side of the blocks whose spread gives the shared signal's error.
TRANSFER_BLOCK = 64
# Outliers per 256 × 256 pixels planted on the empty sky.
OUTLIER_TEST_COUNT = 4.0
# Peak photometry: a box of this radius, less the mean of the ring from
# two pixels beyond it out to the second radius; peaks are the highest
# within the third.
PEAK_APERTURE, PEAK_ANNULUS, PEAK_SEPARATION = 3, 10, 5
# Star widths: stars the highest within this radius on the target, by their
# height above the ring around them, in units of the median noise.
STAR_SEPARATION = 7
STAR_LEVELS = {"faint": (20, 200), "bright": (200, 5000)}
STARS_PER_LEVEL = 1500


def box_filter(image: np.ndarray, radius: int, pool: Callable) -> np.ndarray:
    """`pool` over the (2·radius + 1)² box about each pixel, the edge pixels
    repeated outward, on the training device: a measure's whole-image
    filters cost little there, and most of an evaluation on the CPU."""
    with GPU:
        pixels = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None].to(device())
        padded = torch.nn.functional.pad(pixels, (radius, radius, radius, radius), mode="replicate")
        return pool(padded, 2 * radius + 1, stride=1)[0, 0].cpu().numpy()


def box_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """The mean over the (2·radius + 1)² box about each pixel, the edge
    pixels repeated outward. `image` must be finite."""
    if radius == 0:
        return image
    return box_filter(image, radius, torch.nn.functional.avg_pool2d)


def box_max(image: np.ndarray, radius: int) -> np.ndarray:
    """The largest value in the (2·radius + 1)² box about each pixel."""
    return box_filter(image, radius, torch.nn.functional.max_pool2d)


def gaussian_blur(image: np.ndarray, sigma_px: float) -> np.ndarray:
    """A Gaussian blur of scale `sigma_px`, cut at three of it, the edge
    pixels repeated outward, on the training device. `image` must be
    finite."""
    radius = max(1, math.ceil(3 * sigma_px))
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (offsets / sigma_px) ** 2)
    kernel /= kernel.sum()
    with GPU:
        pixels = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None].to(device())
        padded = torch.nn.functional.pad(pixels, (radius, radius, radius, radius), mode="replicate")
        kernel = kernel.to(pixels.device)
        rows = torch.nn.functional.conv2d(padded, kernel.view(1, 1, 1, -1))
        return torch.nn.functional.conv2d(rows, kernel.view(1, 1, -1, 1))[0, 0].cpu().numpy()


def photometry(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """At each pixel, the flux in the `PEAK_APERTURE` box above the mean of
    the ring around it, and that mean. `image` must be finite."""
    area = (2 * PEAK_APERTURE + 1) ** 2
    hole = (2 * PEAK_APERTURE + 5) ** 2
    outer = (2 * PEAK_ANNULUS + 1) ** 2
    ring = (box_blur(image, PEAK_ANNULUS) * outer - box_blur(image, PEAK_APERTURE + 2) * hole) / (outer - hole)
    return box_blur(image, PEAK_APERTURE) * area - ring * area, ring


def against_truth(estimate: np.ndarray, image: np.ndarray, target: np.ndarray, floor: np.ndarray,
                  background: float, noise: float) -> dict[str, float | None]:
    """How close `estimate` comes to a clean `target`, against how close
    the input came, in units of `noise`: the mean squared error over all
    pixels and by the target's level, the error's noise at the pixel and
    band scales, and the bias on the sky. `floor` is the target's own
    noise variance, which no estimate can remove; it is given on the sky,
    as in star cores it holds the frames' differing seeing too.

    With them, `signal_kept`'s measures of the signal the estimate keeps."""
    valid = np.isfinite(image) & np.isfinite(target) & np.isfinite(estimate)
    before = np.where(valid, (image - target) / noise, np.nan)
    after = np.where(valid, (estimate - target) / noise, np.nan)
    result: dict[str, float | None] = {
        "mse_before": float(np.nanmean(before**2)),
        "mse_after": float(np.nanmean(after**2)),
    }
    for band, transform in (("high", lambda x: x), ("low", lambda x: binned(x, BAND_SCALE_PX))):
        sigma_before, sigma_after = robust_sigma(transform(before)), robust_sigma(transform(after))
        result[f"{band}_ratio"] = sigma_after / sigma_before if sigma_before else None
    level = (target - background) / noise
    sky = valid & (np.abs(level) < 3)
    result["sky_bias"] = float(np.median(after[sky])) if sky.any() else None
    result["sky_mse_floor"] = float(np.nanmean(floor[sky] / noise**2)) if sky.any() else None
    for region, low, high in (("sky", -3, 3), ("faint", 3, 20), ("mid", 20, 200), ("core", 200, np.inf)):
        mask = valid & (level >= low) & (level < high)
        result[f"{region}_mse_before"] = float(np.mean(before[mask] ** 2)) if mask.any() else None
        result[f"{region}_mse_after"] = float(np.mean(after[mask] ** 2)) if mask.any() else None
    bright = valid & (level > 20)
    signal = float(np.sum((target - background)[bright]))
    result["bright_flux_change"] = float(np.sum((estimate - target)[bright]) / signal) if signal else None
    result.update(signal_kept(estimate, image, target, valid, background, noise))
    return result


def signal_kept(estimate: np.ndarray, image: np.ndarray, target: np.ndarray, valid: np.ndarray,
                background: float, noise: float) -> dict[str, float | int | None]:
    """How much of the input's signal `estimate` keeps, against a `target`
    whose noise is independent of the input's: a clean target, or a B half.

    `transfer_<region>_<band>` is the share of the signal in each of
    `TRANSFER_BANDS` that the estimate keeps where the target, smoothed,
    is faint or mid: `⟨e·t⟩ / ⟨x·t⟩` over the band-passed estimate, input
    and target. The target's noise is independent of both, so it cancels.
    It is None where the band holds too little signal to measure: an
    oversampled image's pixel scale in faint regions, mostly.

    `peak_flux_kept_<region>` is the share of the input's flux the
    estimate keeps in small apertures on the target's faint or mid peaks,
    stars and knots by their height above the ring around them, which a
    region's transfer, ruled by its extended signal, hardly sees.
    `peak_height_kept_<region>` is the share of their height, the 3 × 3
    mean above the ring, that it keeps: a peak softened within its
    aperture keeps its flux and loses height."""
    result: dict[str, float | int | None] = {}

    def filled(x: np.ndarray) -> np.ndarray:
        return np.where(valid, x - background, 0.0).astype(np.float32)

    widest = max(outer for _, outer in TRANSFER_BANDS.values())
    inside = box_blur(valid.astype(np.float32), widest) > 0.999
    smooth = box_blur(filled(target), 2) / noise
    regions = {"faint": inside & (smooth >= 3) & (smooth < 20), "mid": inside & (smooth >= 20) & (smooth < 200)}
    height, width = valid.shape
    blocks = (np.arange(height)[:, None] // TRANSFER_BLOCK) * (width // TRANSFER_BLOCK + 1) \
        + np.arange(width)[None, :] // TRANSFER_BLOCK
    for band, (inner, outer) in TRANSFER_BANDS.items():
        e, x, t = (box_blur(filled(v), inner) - box_blur(filled(v), outer) for v in (estimate, image, target))
        for region, mask in regions.items():
            ids = blocks[mask]
            shared = np.bincount(ids, weights=(x[mask] * t[mask]).astype(np.float64))
            kept = float(np.sum(e[mask] * t[mask], dtype=np.float64))
            # The shared signal's error from the spread of its blocks' sums,
            # which overstates it where the signal varies; a band with no
            # signal measurably above that has no transfer to give.
            used = shared[np.bincount(ids) > 0]
            error = math.sqrt(len(used) * float(np.var(used))) if len(used) > 1 else math.inf
            total = float(shared.sum())
            result[f"transfer_{region}_{band}"] = kept / total if total > 3 * error else None

    # Peaks, faint stars and knots, are chosen on the target and measured
    # on the input and the estimate, whose noise the choice does not favour.
    sharp = box_blur(filled(target), 1)
    _, around = photometry(filled(target))
    height_above = (sharp - around) / noise
    peaks = inside & (sharp == box_max(sharp, PEAK_SEPARATION))
    flux_e, ring_e = photometry(filled(estimate))
    flux_x, ring_x = photometry(filled(image))
    height_e = box_blur(filled(estimate), 1) - ring_e
    height_x = box_blur(filled(image), 1) - ring_x
    for region, low, high in (("faint", 3, 20), ("mid", 20, 200)):
        chosen = peaks & (height_above >= low) & (height_above < high)
        result[f"peaks_{region}"] = int(chosen.sum())
        for measured, kept, given in (("flux", flux_e, flux_x), ("height", height_e, height_x)):
            total = float(np.sum(given[chosen], dtype=np.float64))
            result[f"peak_{measured}_kept_{region}"] = (float(np.sum(kept[chosen], dtype=np.float64)) / total
                                                        if total > 0 else None)

    stars = inside & (sharp == box_max(sharp, STAR_SEPARATION))
    result.update(star_widths(filled(estimate), filled(image), filled(target), stars, height_above))
    return result


def star_widths(estimate: np.ndarray, image: np.ndarray, target: np.ndarray, stars: np.ndarray,
                height_above: np.ndarray) -> dict[str, float | int | None]:
    """How an estimate changes star widths: at `stars` chosen on the target,
    the FWHM from each star's second moment about the target's centroid,
    within a radius of one and a half of the field's typical FWHM and above
    the median of the ring outside it, in the estimate, the input and the
    target alike. All three are background-subtracted and finite.

    `star_fwhm_ratio_<level>` is the median over stars of the estimate's
    FWHM over the input's: 1 keeps stars as sharp as they came, more
    softens them. `star_fwhm_target_<level>` is the estimate's over the
    target's. The input's noise widens neither on average, since it adds
    nothing to a moment's expected sums."""
    result: dict[str, float | int | None] = {}
    picked = []
    for low, high in STAR_LEVELS.values():
        found = np.flatnonzero(stars & (height_above >= low) & (height_above < high))
        # At most STARS_PER_LEVEL, spread over the field.
        picked.append(found[np.linspace(0, len(found) - 1, min(len(found), STARS_PER_LEVEL)).astype(int)]
                      if len(found) else found)
    ys, xs = np.unravel_index(np.concatenate(picked), stars.shape)
    if len(ys) < 10:
        return result

    def stamps(image: np.ndarray, radius: int, outer: int) -> np.ndarray:
        padded = np.pad(image, outer)
        windows = np.lib.stride_tricks.sliding_window_view(padded, (2 * outer + 1, 2 * outer + 1))
        return windows[ys, xs].astype(np.float64)

    def widths(stack: np.ndarray, radius: int, outer: int, centre: tuple[np.ndarray, np.ndarray] | None):
        offsets = np.arange(-outer, outer + 1, dtype=np.float64)
        dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
        distance = np.hypot(dy, dx)
        ring = (distance > radius + 1) & (distance <= outer)
        light = stack - np.median(stack[:, ring], axis=1)[:, None, None]
        aperture = distance <= radius
        weights = np.where(aperture, light, 0.0)
        total = weights.sum(axis=(1, 2))
        if centre is None:
            with np.errstate(divide="ignore", invalid="ignore"):
                centre = ((weights * dy).sum(axis=(1, 2)) / total, (weights * dx).sum(axis=(1, 2)) / total)
        cy, cx = (np.clip(np.nan_to_num(c), -1.0, 1.0)[:, None, None] for c in centre)
        spread = (weights * ((dy - cy) ** 2 + (dx - cx) ** 2)).sum(axis=(1, 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            # ⟨r²⟩ = 2σ² for a Gaussian, whose FWHM is 2.3548σ.
            fwhm = 2.3548 * np.sqrt(np.clip(spread / total, 0.0, None) / 2.0)
        return np.where(total > 0, fwhm, np.nan), centre

    # The field's typical width sets the radius, found from the target in
    # two steps, so a coarsely sampled field and a finely sampled one both
    # take their stars whole.
    radius = 6
    for _ in range(2):
        outer = radius + 2 + max(3, radius // 2)
        typical = float(np.nanmedian(widths(stamps(target, radius, outer), radius, outer, None)[0]))
        radius = int(np.clip(round(1.5 * typical), 3, 30)) if np.isfinite(typical) else radius
    outer = radius + 2 + max(3, radius // 2)
    fwhm_t, centre = widths(stamps(target, radius, outer), radius, outer, None)
    fwhm_e, _ = widths(stamps(estimate, radius, outer), radius, outer, centre)
    fwhm_x, _ = widths(stamps(image, radius, outer), radius, outer, centre)
    level = height_above[ys, xs]
    result["star_fwhm_px"] = float(np.nanmedian(fwhm_t))
    for name, (low, high) in STAR_LEVELS.items():
        chosen = (level >= low) & (level < high) & np.isfinite(fwhm_t) & np.isfinite(fwhm_e) & np.isfinite(fwhm_x) \
            & (fwhm_x > 0) & (fwhm_t > 0)
        result[f"stars_{name}"] = int(chosen.sum())
        enough = chosen.sum() >= 10
        result[f"star_fwhm_ratio_{name}"] = float(np.median(fwhm_e[chosen] / fwhm_x[chosen])) if enough else None
        result[f"star_fwhm_target_{name}"] = float(np.median(fwhm_e[chosen] / fwhm_t[chosen])) if enough else None
    return result


# The scales, in pixels, at which `noise_left` smooths.
MOTTLE_SCALES = (4, 8, 16)


def noise_left(
    estimate_a: np.ndarray, estimate_b: np.ndarray, image_a: np.ndarray, image_b: np.ndarray, noise: float
) -> dict[str, float | None]:
    """The share of a noise pair's noise a model left at each of
    `MOTTLE_SCALES`, on sky with nothing brighter than the noise within 8
    pixels. A − B holds only the halves' noise, and f(A) − f(B) only what
    the model left of it, signal and gradients cancelling in both; each is
    Gaussian-smoothed over that sky at the scale, and the share is the
    spread of the one over the other's. A model that takes out the grain
    but leaves the noise at 8 or 16 pixels leaves mottle. On a flat sky of
    noise alone, the first models left 3–7% of the coarse noise; on a real
    one, 60–90% at those scales, since there faint signal looks like it."""
    valid = np.isfinite(image_a) & np.isfinite(image_b) & np.isfinite(estimate_a) & np.isfinite(estimate_b)
    both = np.where(valid, (image_a + image_b) / 2, 0.0)
    background = float(np.median(both[valid][::7])) if valid.any() else 0.0
    level = gaussian_blur(np.where(valid, both - background, 0.0).astype(np.float32), 2.0) / noise
    sky = valid & ~(box_max((level > 1.0).astype(np.float32), 8) > 0)
    noise_alone = np.where(sky, image_a - image_b, 0.0).astype(np.float32)
    noise_kept = np.where(sky, estimate_a - estimate_b, 0.0).astype(np.float32)
    result: dict[str, float | None] = {}
    for scale in MOTTLE_SCALES:
        weight = gaussian_blur(sky.astype(np.float32), scale)
        # The blur repeats the image's edge outward, which would spread one
        # pixel's noise over the kernel, so the edges are left out.
        reach = 3 * scale
        middle = (slice(reach, -reach), slice(reach, -reach))
        inside = np.zeros_like(sky)
        inside[middle] = sky[middle] & (weight[middle] > 0.5)
        if not inside.any():
            result[f"noise_left_{scale}px"] = None
            continue
        spread = [float(np.sqrt(np.mean((gaussian_blur(x, scale)[inside] / weight[inside]) ** 2)))
                  for x in (noise_kept, noise_alone)]
        result[f"noise_left_{scale}px"] = spread[0] / spread[1] if spread[1] else None
    return result


def on_empty_sky(output: np.ndarray, image: np.ndarray, background: float, noise: float) -> dict[str, float | None]:
    """What a model made of noise on a flat sky: the noise left at the
    pixel and band scales, as a share of the input's, and the brightest
    binned feature of the output in units of the input's binned noise,
    which a model that invents structure raises."""
    valid = np.isfinite(output) & np.isfinite(image)
    before = np.where(valid, image - background, np.nan)
    after = np.where(valid, output - background, np.nan)
    result: dict[str, float | None] = {}
    for band, transform in (("high", lambda x: x), ("low", lambda x: binned(x, BAND_SCALE_PX))):
        sigma_before, sigma_after = robust_sigma(transform(before)), robust_sigma(transform(after))
        result[f"{band}_ratio"] = sigma_after / sigma_before if sigma_before else None
    low_before = robust_sigma(binned(before, BAND_SCALE_PX)) or float("nan")
    low_after = binned(after, BAND_SCALE_PX)
    low_after = low_after[np.isfinite(low_after)]
    result["peak_low_sigma"] = float(np.percentile(np.abs(low_after), 99.99) / low_before) if low_after.size else None
    result["bias"] = float(np.nanmedian(after) / noise)
    return result


def evaluate_dark_noise(
    store: Store,
    manifest: Manifest,
    model: ResidualUNet,
    *,
    split: str,
    tile: int = 512,
    overlap: int = 136,
    margin: int = 52,
    multiple: int | None = None,
    sets: set[str] | None = None,
    noise_map: str = "raw",
    noise_plane: str = "linear",
    bands: NoiseBands = NoiseBands(),
    threads: int = EVALUATION_THREADS,
    sky: str = "global",
    holes: bool = False,
) -> dict[str, Any]:
    """Denoise each of a split's dark-noise inputs and measure it against
    its clean target (`truth`), and denoise its noise alone on a flat sky
    (`empty_sky`), once as it is and once with `outliers` planted in it.
    Inference, noise plane, sky, units, `bands`, threads and the hole
    guard are `evaluate`'s."""
    from .holes import HoleGuard, guarded, guarded_residual

    guard = HoleGuard() if holes else None
    model = model.eval()
    where = device()

    def kept(image: np.ndarray, found: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """What `found` finds in `image`, with the guard's holes kept from it
        and put back after, as the runtime runs a model."""
        if guard is None:
            return found(image)
        shown, mask, mapped = guarded(image, guard)
        return guarded_residual(image, shown, found(shown), mask, mapped, guard)

    def one(pair: dict[str, Any]) -> dict[str, Any]:
        image, sigma = load_half(store, pair["input"])
        target, _ = load_half(store, pair["target"], noise_map=False)
        target_files = loads(pair["target"]["files"])
        target_coverage = np.load(store.root / target_files["coverage"])
        with np.errstate(divide="ignore", invalid="ignore"):
            floor = np.load(store.root / target_files["variance"]) / target_coverage
        background, noise = scales(image, sigma)
        coverage = np.load(store.root / loads(pair["input"]["files"])["coverage"])
        settings = {"tile": tile, "overlap": overlap, "margin": margin, "multiple": multiple or model.multiple}

        def found(shown: np.ndarray) -> np.ndarray:
            stated = input_sigma(noise_map, shown, sigma, coverage)
            planes, input_background, input_noise = view(shown, stated, sky=sky, noise_plane=noise_plane)
            return infer(denoiser(model, input_background, input_noise, where), planes, **settings)[0]

        estimate = bands.apply(image, kept(image, found))
        truth = against_truth(estimate, image, target, floor, background, noise)
        del estimate

        files = loads(pair["input"]["files"])
        alone = np.load(store.root / files["noise"]).astype(np.float32)
        alone_coverage = np.load(store.root / files["noise_coverage"])
        alone_variance = np.load(store.root / files["noise_variance"]).astype(np.float32)
        blank = np.where(alone_coverage > 0, background + alone, np.nan).astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            sky_sigma = np.where(alone_coverage > 0, np.sqrt(alone_variance / alone_coverage), np.nan).astype(np.float32)
        sky_sigma = input_sigma(noise_map, blank, sky_sigma, alone_coverage)
        _, sky_background, sky_noise = view(blank, sky_sigma, sky=sky, noise_plane=noise_plane)
        sky_run = denoiser(model, sky_background, sky_noise, where)

        def on_sky(shown: np.ndarray) -> np.ndarray:
            # At the sky's own scales, so the outliers planted below are all
            # that differs between the two.
            return infer(sky_run, network_input(seen(shown, sky), sky_sigma, sky_background, sky_noise, noise_plane),
                         **settings)[0]

        output = bands.apply(blank, kept(blank, on_sky))
        empty = on_empty_sky(output, blank, background, noise)
        # The same outliers for every model, planted on the empty sky: the
        # share of their energy the output keeps.
        planted = outliers(blank.shape, np.random.default_rng(zlib.crc32(pair["input"]["id"].encode())),
                           OUTLIER_TEST_COUNT) * sky_noise
        planted[~np.isfinite(blank)] = 0.0
        spoiled = blank + planted
        left = bands.apply(spoiled, kept(spoiled, on_sky)) - output
        empty["outliers_left"] = float(np.nansum(left**2) / np.sum(planted**2))
        return {"pair": pair["input"]["id"], "set": pair["set"],
                "size": loads(pair["input"]["recipe"]).get("size"),
                **{f"truth_{key}": value for key, value in truth.items()},
                **{f"empty_{key}": value for key, value in empty.items()}}

    with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="evaluate") as pool:
        rows = list(pool.map(one, dark_noise.pairs(manifest, split, sets)))

    def mean(key):
        values = [row[key] for row in rows if row.get(key) is not None]
        return float(np.mean(values)) if values else None

    keys = sorted({key for row in rows for key in row if key.startswith(("truth_", "empty_"))})
    return {"split": split, "bands": asdict(bands), "pairs": len(rows), "mean": {key: mean(key) for key in keys},
            "pairs_detail": rows}


def export(model: ResidualUNet, path: Path, *, tile: int) -> dict[str, Any]:
    """Write the model as ONNX at a static tile shape, and hold ONNX
    Runtime's CPU output to PyTorch's on a random tile."""
    import onnxruntime

    model = model.cpu().eval()
    sample = torch.randn(1, 2, tile, tile)
    torch.onnx.export(model, (sample,), str(path), input_names=["planes"], output_names=["residual"],
                      opset_version=17, dynamo=False)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    expected = model(sample).detach().numpy()
    got = session.run(None, {"planes": sample.numpy()})[0]
    scale = float(np.abs(expected).max()) or 1.0
    return {"path": str(path), "opset": 17, "tile": tile, "max_relative_error": float(np.abs(got - expected).max() / scale),
            "bytes": path.stat().st_size}


def load_run(store: Store, name: str) -> tuple[dict[str, Any], ResidualUNet]:
    """A recorded run's record and its model, on the CPU."""
    directory = store.root / "models" / "noise" / name
    record = json.loads((directory / "run.json").read_text())
    model = ResidualUNet(width=record["config"]["width"], levels=record["config"]["levels"])
    model.load_state_dict(torch.load(directory / "weights.pt", map_location="cpu"))
    # Tiling follows from the network's depth: a run recorded with older
    # settings is run with these.
    record["inference"] = tiling(record["config"]["levels"])
    return record, model.eval()


def tiling(levels: int) -> dict[str, int]:
    """How a model of `levels` runs over an image: tiles of 512, a margin of
    its reach, which reproduces whole-image inference, a cosine over 32 more
    pixels, and every tile starting on its grid."""
    margin = reach(levels)
    return {"tile": 512, "overlap": 2 * margin + 32, "margin": margin, "multiple": 2 ** (levels - 1)}


def evaluate_saved(
    store: Store, manifest: Manifest, name: str, *, split: str, limit: int | None = None, sets: set[str] | None = None,
    bands: NoiseBands = NoiseBands(), match_flux: bool = False, noise_map: str | None = None,
    holes: bool | None = None,
) -> dict[str, Any]:
    """Measure a recorded run's model on `split`, taking out `bands` of
    what it finds, and add it to the record: under the split's name, then,
    when `bands` keep some noise, theirs, with `match_flux` "matched
    flux", and with `noise_map` the kind of map the network reads in place
    of the one it trained with, "<kind> map". The hole guard runs as the
    run trained, or as `holes` says, which a label then names."""
    directory = store.root / "models" / "noise" / name
    record, model = load_run(store, name)
    config = record["config"]
    read = {"noise_map": noise_map or config.get("noise_map", "raw"),
            "noise_plane": config.get("noise_plane", "linear"), "sky": config.get("sky", "global")}
    guarded = bool(config.get("holes")) if holes is None else holes
    result = evaluate(store, manifest, model, split=split, limit=limit, sets=sets, **read, bands=bands,
                      match_flux=match_flux, holes=guarded, **record["inference"])
    result["dark_noise"] = evaluate_dark_noise(store, manifest, model, split=split, sets=sets, **read, bands=bands,
                                               holes=guarded, **record["inference"])
    label = ", ".join([split] + ([] if bands.whole else [str(bands)]) + (["matched flux"] if match_flux else [])
                      + ([f"{noise_map} map"] if noise_map else [])
                      + ([f"holes {'kept' if guarded else 'seen'}"] if holes is not None else []))
    record.setdefault("evaluations", {})[label] = result
    (directory / "run.json").write_text(json.dumps(record, indent=2))
    return {key: value for key, value in result.items() if key != "pairs_detail"} | {
        "dark_noise": {key: value for key, value in result["dark_noise"].items() if key != "pairs_detail"}
    }


# The stability probe's noise: spread over boxes of these sides in pixels,
# at these multiples of the noise the network is told.
STABILITY_SPREADS = (1, 2, 3, 4)
STABILITY_AMOUNTS = (1.0, 1.5, 2.0, 3.0, 5.0)
# A step past this many σ̄ is a blow-up: the noise itself reaches about
# 5σ̄ at its own level and 25σ̄ at five times it, and stable models' steps
# stay near that.
STABILITY_LIMIT = 100.0


def stability(model: ResidualUNet, inference: dict[str, int], *, sky: str = "global", noise_plane: str = "linear",
              size: int = 1024, seed: int = 7) -> dict[str, Any]:
    """How far a model throws a pixel on synthetic sky: flat, with Gaussian
    noise spread over k × k boxes (`STABILITY_SPREADS`; a 2x drizzle
    spreads it over about 2) at multiples of the noise the network is told
    (`STABILITY_AMOUNTS`), the largest step in σ̄ for each, and whether all
    stay within `STABILITY_LIMIT`. The halves' measures never show it: some
    models that measure as well as any throw pixels thousands of σ̄ on
    spread noise, as on a drizzle's edges."""
    white = np.random.default_rng(seed).standard_normal((size + 3, size + 3)).astype(np.float32)
    where = device()
    level, told = 100.0, 1.0
    steps = {}
    for side in STABILITY_SPREADS:
        field = sum(white[dy:dy + size, dx:dx + size] for dy in range(side) for dx in range(side))
        field = field / field.std()
        for amount in STABILITY_AMOUNTS:
            image = (level + amount * told * field).astype(np.float32)
            planes, background, noise = view(image, np.full_like(image, told), sky=sky, noise_plane=noise_plane)
            residual = infer(denoiser(model, background, noise, where), planes, **inference)[0]
            steps[f"{side}px x{amount:g}"] = float(np.abs(residual).max() / noise)
    largest = max(steps.values())
    return {"steps": steps, "largest": largest, "stable": largest <= STABILITY_LIMIT}


def stability_saved(store: Store, name: str) -> dict[str, Any]:
    """A recorded run's `stability`, added to its record."""
    directory = store.root / "models" / "noise" / name
    record, model = load_run(store, name)
    config = record["config"]
    record["stability"] = stability(model, record["inference"], sky=config.get("sky", "global"),
                                    noise_plane=config.get("noise_plane", "linear"))
    (directory / "run.json").write_text(json.dumps(record, indent=2))
    return record["stability"]
