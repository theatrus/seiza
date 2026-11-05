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


NOISE_MAPS = ("raw", "model")
# The side of the blocks the modelled map takes the sky's variance over.
SKY_BLOCK = 32


def load_half(
    store: Store, product: dict[str, Any], *, noise_map: bool = True, modelled: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    """A stack's image, NaN where nothing covers it, and its noise map
    when `noise_map`: `sqrt(variance / coverage)` per pixel, or, when
    `modelled`, the map `modelled_sigma` makes from it."""
    files = loads(product["files"])
    image = np.load(store.root / files["image"]).astype(np.float32)
    coverage = np.load(store.root / files["coverage"])
    image[coverage == 0] = np.nan
    if not noise_map:
        return image, None
    variance = np.load(store.root / files["variance"]).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.where(coverage > 0, np.sqrt(variance / coverage), np.nan).astype(np.float32)
    if modelled:
        sigma = modelled_sigma(image, sigma, coverage)
    return image, sigma


def upsample(grid: np.ndarray, height: int, width: int, block: int) -> np.ndarray:
    """A grid of block values, one per `block` × `block` pixels, spread
    bilinearly over `height` × `width` pixels."""
    rows, columns = grid.shape
    ys = np.clip((np.arange(height, dtype=np.float32) + 0.5) / block - 0.5, 0, rows - 1)
    xs = np.clip((np.arange(width, dtype=np.float32) + 0.5) / block - 0.5, 0, columns - 1)
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


def bank(
    store: Store, chosen: list[dict[str, Any]], *, seed: int = 0, noise_map: str = "raw", match_flux: bool = False,
    agree: float | None = None,
) -> dict[str, Any]:
    """Windows of every pair: A's image and noise map, and B's image, with
    `match_flux` first `matched` to A's. With `agree` too, a pair whose
    matched halves `disagree` by more than that is left out; the ids left
    out are under "dropped". Cached by the pairs, the seed, the kind of
    noise map, the matching and the agreement asked for."""
    if agree is not None and not match_flux:
        raise ValueError("agreement is judged on matched halves: ask for match_flux too")
    parts = [p["a"]["id"] for p in chosen] + [seed, WINDOW, WINDOWS_PER_HALF]
    if noise_map != "raw":
        parts.append(f"map-{noise_map}")
    if match_flux:
        # 2: the background planes leave out hot pixels.
        parts.append("match-flux-2")
    if agree is not None:
        parts.append(f"agree-{agree:g}")
    key = hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16]
    record = store.scratch / "tiles" / f"noise-{key}" / "dropped.json"

    def halves():
        dropped = []
        for pair in chosen:
            image_a, sigma_a = load_half(store, pair["a"], modelled=noise_map == "model")
            image_b, _ = load_half(store, pair["b"], noise_map=False)
            if match_flux:
                noise = scales(image_a, sigma_a)[1]
                image_b, _ = matched(image_a, image_b, noise)
                if agree is not None and disagree(agreement(image_a, image_b, noise), agree):
                    dropped.append(pair["a"]["id"])
                    continue
            yield image_a, sigma_a, image_b
        # Before cut_windows marks the bank done.
        record.write_text(json.dumps(dropped))

    banked = cut_windows(store, f"noise-{key}", halves(), sources=len(chosen), seed=seed)
    banked["dropped"] = json.loads(record.read_text()) if record.exists() else []
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
    store: Store, manifest: Manifest, *, sets: set[str] | None = None, seed: int = 0, noise_map: str = "raw"
) -> dict[str, np.ndarray]:
    """Windows of the train split's dark-noise pairs: the input, its noise
    map and the deep stack it was made from, as `cut_windows` cuts them."""
    chosen = dark_noise.pairs(manifest, "train", sets)
    if not chosen:
        raise RuntimeError("no built dark-noise pairs in the train split")
    parts = ["dark-noise"] + [p["input"]["id"] for p in chosen] + [seed, WINDOW, WINDOWS_PER_HALF, noise_map]
    key = hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16]

    def images():
        for pair in chosen:
            image, sigma = load_half(store, pair["input"], modelled=noise_map == "model")
            target, _ = load_half(store, pair["target"], noise_map=False)
            yield image, sigma, target

    return cut_windows(store, f"dark-noise-{key}", images(), sources=len(chosen), seed=seed)


BANK_PLANES = ("a", "sigma", "b")


def cut_windows(store: Store, name: str, images: Any, *, sources: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Windows of each `(image, noise map, target)` that `images` yields, at
    most `sources` of them, on an overlapping grid and wholly inside both
    images' coverage, with each source's background and median noise.

    The bank is written window by window into memory-mapped `.npy` files in
    the local scratch directory under `name`, and read back the same way, so
    neither building nor training holds it in memory. `done.json` marks a
    finished bank; one without it is built again."""
    directory = store.scratch / "tiles" / name
    done = directory / "done.json"
    if done.exists():
        return open_bank(directory, json.loads(done.read_text())["windows"])
    directory.mkdir(parents=True, exist_ok=True)
    capacity = max(1, sources * WINDOWS_PER_HALF)
    planes = {
        plane: np.lib.format.open_memmap(directory / f"{plane}.npy", mode="w+", dtype=np.float32,
                                         shape=(capacity, WINDOW, WINDOW))
        for plane in BANK_PLANES
    }
    rng = random.Random(seed)
    background, noise = [], []
    count = 0
    for image_a, sigma_a, image_b in images:
        b, s = scales(image_a, sigma_a)
        height, width = image_a.shape
        step = WINDOW // 2
        grid = [(y, x) for y in range(0, height - WINDOW, step) for x in range(0, width - WINDOW, step)]
        rng.shuffle(grid)
        kept = 0
        for y, x in grid:
            window_a = image_a[y:y + WINDOW, x:x + WINDOW]
            window_b = image_b[y:y + WINDOW, x:x + WINDOW]
            if not (np.isfinite(window_a).all() and np.isfinite(window_b).all()):
                continue
            if count == capacity:
                break
            planes["a"][count] = window_a
            planes["sigma"][count] = sigma_a[y:y + WINDOW, x:x + WINDOW]
            planes["b"][count] = window_b
            background.append(b)
            noise.append(s)
            count += 1
            kept += 1
            if kept == WINDOWS_PER_HALF:
                break
    for plane in planes.values():
        plane.flush()
    del planes
    np.save(directory / "background.npy", np.array(background, dtype=np.float32))
    np.save(directory / "noise.npy", np.array(noise, dtype=np.float32))
    done.write_text(json.dumps({"windows": count, "sources": sources}))
    return open_bank(directory, count)


def open_bank(directory: Path, count: int) -> dict[str, np.ndarray]:
    """A finished bank's planes, memory-mapped and cut to its windows."""
    banked = {plane: np.load(directory / f"{plane}.npy", mmap_mode="r")[:count] for plane in BANK_PLANES}
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
    those out shaves the noise's peaks and dims what is bright."""
    count = len(banked["a"])
    picks = rng.choice(count, size, p=odds) if odds is not None else rng.integers(0, count, size)
    inputs, targets, references, sigmas = [], [], [], []
    for index in picks:
        y, x = rng.integers(0, WINDOW - tile + 1, 2)
        a = banked["a"][index, y:y + tile, x:x + tile]
        sigma = banked["sigma"][index, y:y + tile, x:x + tile]
        b = banked["b"][index, y:y + tile, x:x + tile]
        background, noise = float(banked["background"][index]), float(banked["noise"][index])
        turns, flip = int(rng.integers(0, 4)), bool(rng.integers(0, 2))
        planes = np.stack([a, sigma, b])
        planes = np.rot90(planes, turns, axes=(1, 2))
        if flip:
            planes = planes[:, :, ::-1]
        a, sigma, b = planes
        if outlier_count:
            scale = np.where(np.isfinite(sigma), sigma, noise) if outlier_scale == "local" else noise
            a = a + outliers(a.shape, rng, outlier_count, floor=outlier_floor) * scale
        stated = sigma * float(np.exp(rng.uniform(-map_jitter, map_jitter))) if map_jitter else sigma
        inputs.append(network_input(a, stated, background, noise, noise_plane))
        targets.append(np.stack([(b - background) / noise]))
        references.append(np.stack([(a - background) / noise]))
        sigmas.append(np.stack([np.nan_to_num(sigma / noise, nan=1.0)]))
    return tuple(
        torch.from_numpy(np.ascontiguousarray(np.stack(values)))
        for values in (inputs, targets, references, sigmas)
    )


class Prefetch:
    """`make(step)` for steps 1, 2, …, built up to `depth` ahead on `threads`
    threads and handed out in step order. A batch depends only on its step,
    so the batches are those calling `make` in the loop would give, whatever
    the threads. NumPy releases the interpreter lock for the copies and the
    arithmetic, which is most of a batch."""

    def __init__(self, make: Callable[[int], Any], depth: int, threads: int = 1) -> None:
        self.make = make
        self.pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="batch")
        self.pending: deque[Future] = deque()
        self.step = 0
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
) -> torch.Tensor:
    """L2 of `A + residual` against the target, both in units of σ̄ above
    the background. `core` weights star cores down by their brightness;
    `variance` weights every pixel by the inverse of A's noise variance
    there, so a pixel counts by how well A knew it.

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
    if huber > 0:
        error = (estimate - targets) * weight.sqrt()
        robust = torch.nn.functional.huber_loss(error, torch.zeros_like(error), delta=huber, reduction="none") * 2.0
        if huber_above > 0:
            robust = torch.where(targets.abs() > huber_above, robust, error**2)
        return robust.mean()
    return (weight * (estimate - targets) ** 2).mean()


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
) -> dict[str, Any]:
    """Train a noise model on the train split's pairs, evaluate it on the
    validation split's, export it to ONNX and record the run.

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
    if extra is not None and extra_fraction and not len(extra["a"]):
        raise RuntimeError(f"the {extra_label} bank holds no window wholly inside its pairs' coverage")
    torch.manual_seed(seed)
    chosen = pairs(manifest, "train", sets)
    if not chosen:
        raise RuntimeError("no built noise pairs in the train split")
    started = time.monotonic()
    banked = bank(store, chosen, seed=seed, noise_map=noise_map, match_flux=match_flux, agree=agree)
    bank_seconds = time.monotonic() - started
    where = device()
    cuda = where.type == "cuda"
    layout = torch.channels_last if cuda else torch.contiguous_format
    model = ResidualUNet(width=width, levels=levels).to(where, memory_format=layout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=steps, pct_start=0.05)
    history = []
    extra_count = round(batch * extra_fraction) if extra is not None else 0
    odds = structure_odds(banked, structure_sampling)
    extra_odds = structure_odds(extra, structure_sampling) if extra is not None else None

    def assemble(step: int) -> tuple[torch.Tensor, ...]:
        rng = step_generator(seed, step)
        parts = [batch_from(banked, batch - extra_count, tile, rng, map_jitter=map_jitter, odds=odds,
                            outlier_count=outlier_count, outlier_floor=outlier_floor,
                            outlier_scale=outlier_scale, noise_plane=noise_plane)]
        if extra_count:
            parts.append(batch_from(extra, extra_count, tile, rng, map_jitter=map_jitter, odds=extra_odds,
                                    outlier_count=outlier_count, outlier_floor=outlier_floor,
                                    outlier_scale=outlier_scale, noise_plane=noise_plane))
        tensors = tuple(torch.cat(values) for values in zip(*parts))
        # Pinned, a batch copies to the GPU while it still trains.
        return tuple(t.pin_memory() for t in tensors) if cuda else tensors

    source = Prefetch(assemble, prefetch, batch_threads) if prefetch else None
    started = time.monotonic()
    # Summed on the device, so a step never waits for the GPU to finish.
    running = torch.zeros((), device=where)
    # With clipping, the gradient's norm before it: the sum and the largest.
    norms = torch.zeros(2, device=where)
    try:
        for step in range(1, steps + 1):
            tensors = source.get() if source else assemble(step)
            inputs, targets, references, sigmas = (t.to(where, non_blocking=True) for t in tensors)
            inputs = inputs.contiguous(memory_format=layout)
            with torch.autocast(where.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                loss = loss_of(model, inputs, targets, references, sigmas, loss_weight, huber, huber_above)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                norms[0] += norm
                norms[1] = torch.maximum(norms[1], norm)
            optimizer.step()
            schedule.step()
            running += loss.detach().float()
            if step % log_every == 0 or step == steps:
                mean = float(running) / log_every
                entry = {"step": step, "loss": mean, "seconds": round(time.monotonic() - started, 1)}
                gradient = ""
                if clip_grad:
                    entry["grad_norm_mean"], entry["grad_norm_max"] = float(norms[0]) / log_every, float(norms[1])
                    gradient = f" gradient {entry['grad_norm_mean']:.3g} (largest {entry['grad_norm_max']:.3g})"
                    norms.zero_()
                history.append(entry)
                print(f"train {name}: step {step} loss {mean:.4f}{gradient} ({time.monotonic() - started:.0f} s)",
                      flush=True)
                running.zero_()
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
                          noise_map=noise_map, noise_plane=noise_plane)
    validation["dark_noise"] = evaluate_dark_noise(store, manifest, model, split="validation", **inference,
                                                   sets=validation_sets, noise_map=noise_map,
                                                   noise_plane=noise_plane)
    exported = export(model, directory / "model.onnx", tile=512)
    record = {
        "name": name,
        "created": now(),
        "config": {"width": width, "levels": levels, "tile": tile, "batch": batch, "steps": steps,
                   "learning_rate": learning_rate, "seed": seed,
                   "train_pairs": len(chosen) - len(banked["dropped"]), "pairs_dropped": banked["dropped"],
                   "windows": int(len(banked["a"])), "core_sigma": CORE_SIGMA,
                   "loss_weight": loss_weight, "map_jitter": map_jitter, "noise_map": noise_map,
                   "huber": huber, "huber_above": huber_above, "structure_sampling": structure_sampling,
                   "outliers": outlier_count, "outlier_floor": outlier_floor,
                   "outlier_scale": outlier_scale, "noise_plane": noise_plane, "clip_grad": clip_grad,
                   "match_flux": match_flux, "agree": agree, "precision": precision, "prefetch": prefetch,
                   "batch_threads": batch_threads, "batch_generator": "per-step",
                   "extra": {"label": extra_label, "fraction": extra_fraction, "windows": int(len(extra["a"]))}
                   if extra is not None else None},
        "parameters": sum(p.numel() for p in model.parameters()),
        "receptive_field_px": receptive_field(levels),
        "inference": inference,
        "device": str(where),
        "seconds": {"bank": round(bank_seconds, 1), "train": round(train_seconds, 1)},
        "history": history,
        "validation": validation,
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
) -> dict[str, Any]:
    """Denoise each of a split's A halves, whole, through overlapping
    tiles that start on the model's grid, or on multiples of `multiple`
    when given, take out `bands` of what the model finds, and measure it
    against its B half. The deep stacks' halves are measured the same way, one
    against the other, under the size `deep-half`, or `multi-night` for an
    object stacked over its nights: they show the model at a night's whole
    depth and beyond, far less noisy than the noise pairs. The
    network reads the noise map it was trained with, as `noise_plane`
    says; every run is measured
    in the same units, the per-pixel map's median noise, so runs with
    different maps compare. With `match_flux`, each B is first `matched`
    to its A, so a model that keeps A's flux is not marked down for the
    halves' own difference, and each pair's gain and `agreement` are
    recorded with it. `threads` pairs are measured at once."""
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
        if noise_map == "model":
            coverage = np.load(store.root / loads(pair["a"]["files"])["coverage"])
            sigma_a = modelled_sigma(image_a, sigma_a, coverage)
        input_background, input_noise = scales(image_a, sigma_a)
        planes = network_input(image_a, sigma_a, input_background, input_noise, noise_plane)
        residual = infer(denoiser(model, input_background, input_noise, where), planes,
                         tile=tile, overlap=overlap, margin=margin,
                         multiple=multiple or model.multiple)[0]
        estimate = bands.apply(image_a, residual)
        recipe = loads(pair["a"]["recipe"])
        return {"pair": pair["a"]["id"], "set": pair["set"], "size": recipe.get("size") or recipe["recipe"],
                "flux_gain": gain, **found, **measure(estimate, image_a, image_b, background, noise)}

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
            "star_fwhm_px", *(f"star_fwhm_{kind}_{level}" for kind in ("ratio", "target") for level in STAR_LEVELS))
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
) -> dict[str, Any]:
    """Denoise each of a split's dark-noise inputs and measure it against
    its clean target (`truth`), and denoise its noise alone on a flat sky
    (`empty_sky`), once as it is and once with `outliers` planted in it.
    Inference, noise plane, units, `bands` and threads are `evaluate`'s."""
    model = model.eval()
    where = device()

    def one(pair: dict[str, Any]) -> dict[str, Any]:
        image, sigma = load_half(store, pair["input"])
        target, _ = load_half(store, pair["target"], noise_map=False)
        target_files = loads(pair["target"]["files"])
        target_coverage = np.load(store.root / target_files["coverage"])
        with np.errstate(divide="ignore", invalid="ignore"):
            floor = np.load(store.root / target_files["variance"]) / target_coverage
        background, noise = scales(image, sigma)
        coverage = np.load(store.root / loads(pair["input"]["files"])["coverage"])
        stated = modelled_sigma(image, sigma, coverage) if noise_map == "model" else sigma
        input_background, input_noise = scales(image, stated)
        run = denoiser(model, input_background, input_noise, where)
        estimate = bands.apply(image, infer(run, network_input(image, stated, input_background, input_noise,
                                                               noise_plane),
                                            tile=tile, overlap=overlap, margin=margin,
                                            multiple=multiple or model.multiple)[0])
        truth = against_truth(estimate, image, target, floor, background, noise)
        del estimate

        files = loads(pair["input"]["files"])
        alone = np.load(store.root / files["noise"]).astype(np.float32)
        alone_coverage = np.load(store.root / files["noise_coverage"])
        alone_variance = np.load(store.root / files["noise_variance"]).astype(np.float32)
        sky = np.where(alone_coverage > 0, background + alone, np.nan).astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            sky_sigma = np.where(alone_coverage > 0, np.sqrt(alone_variance / alone_coverage), np.nan).astype(np.float32)
        if noise_map == "model":
            sky_sigma = modelled_sigma(sky, sky_sigma, alone_coverage)
        sky_background, sky_noise = scales(sky, sky_sigma)
        sky_run = denoiser(model, sky_background, sky_noise, where)
        output = bands.apply(sky, infer(sky_run, network_input(sky, sky_sigma, sky_background, sky_noise, noise_plane),
                                        tile=tile, overlap=overlap, margin=margin,
                                        multiple=multiple or model.multiple)[0])
        empty = on_empty_sky(output, sky, background, noise)
        # The same outliers for every model, planted on the empty sky: the
        # share of their energy the output keeps.
        planted = outliers(sky.shape, np.random.default_rng(zlib.crc32(pair["input"]["id"].encode())),
                           OUTLIER_TEST_COUNT) * sky_noise
        planted[~np.isfinite(sky)] = 0.0
        spoiled = sky + planted
        left = bands.apply(spoiled, infer(sky_run, network_input(spoiled, sky_sigma, sky_background, sky_noise,
                                                                 noise_plane),
                                          tile=tile, overlap=overlap, margin=margin,
                                          multiple=multiple or model.multiple)[0]) - output
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
    bands: NoiseBands = NoiseBands(), match_flux: bool = False,
) -> dict[str, Any]:
    """Measure a recorded run's model on `split`, taking out `bands` of
    what it finds, and add it to the record: under the split's name, then,
    when `bands` keep some noise, theirs, and with `match_flux` "matched
    flux"."""
    directory = store.root / "models" / "noise" / name
    record, model = load_run(store, name)
    config = record["config"]
    read = {"noise_map": config.get("noise_map", "raw"), "noise_plane": config.get("noise_plane", "linear")}
    result = evaluate(store, manifest, model, split=split, limit=limit, sets=sets, **read, bands=bands,
                      match_flux=match_flux, **record["inference"])
    result["dark_noise"] = evaluate_dark_noise(store, manifest, model, split=split, sets=sets, **read, bands=bands,
                                               **record["inference"])
    label = ", ".join([split] + ([] if bands.whole else [str(bands)]) + (["matched flux"] if match_flux else []))
    record.setdefault("evaluations", {})[label] = result
    (directory / "run.json").write_text(json.dumps(record, indent=2))
    return {key: value for key, value in result.items() if key != "pairs_detail"} | {
        "dark_noise": {key: value for key, value in result["dark_noise"].items() if key != "pairs_detail"}
    }
