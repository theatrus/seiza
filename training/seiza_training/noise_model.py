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
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
import warnings
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import dark_noise
from .build import BAND_SCALE_PX, binned
from .manifest import Manifest, loads, now
from .store import Store, robust_sigma, versions
from .tiles import infer
from .unet import ResidualUNet, receptive_field

WINDOW = 320
WINDOWS_PER_HALF = 40
# Pixels this many σ̄ above the background are star cores, weighted down.
CORE_SIGMA = 50.0


def pairs(manifest: Manifest, split: str, sets: set[str] | None = None) -> list[dict[str, Any]]:
    """The built noise pairs of one split's sets: each A product with its B."""
    rows = manifest.rows(
        "SELECT p.*, s.split, s.name FROM products p JOIN sets s ON s.id = p.set_id "
        "WHERE p.kind = 'noise-a' AND p.status = 'built' AND s.split = ? ORDER BY p.id",
        (split,),
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


def network_input(image: np.ndarray, sigma: np.ndarray, background: float, noise: float) -> np.ndarray:
    """The two planes the network reads; NaN becomes the background."""
    x = np.arcsinh(np.nan_to_num((image - background) / noise, nan=0.0))
    s = np.nan_to_num(sigma / noise, nan=1.0)
    return np.stack([x, s]).astype(np.float32)


def bank(store: Store, chosen: list[dict[str, Any]], *, seed: int = 0, noise_map: str = "raw") -> dict[str, np.ndarray]:
    """Windows of every pair: A's image and noise map, and B's image.
    Cached by the pairs, the seed and the kind of noise map."""
    parts = [p["a"]["id"] for p in chosen] + [seed, WINDOW, WINDOWS_PER_HALF]
    if noise_map != "raw":
        parts.append(f"map-{noise_map}")
    key = hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16]

    def halves():
        for pair in chosen:
            image_a, sigma_a = load_half(store, pair["a"], modelled=noise_map == "model")
            image_b, _ = load_half(store, pair["b"], noise_map=False)
            yield image_a, sigma_a, image_b

    return cut_windows(store, f"noise-{key}", halves(), sources=len(chosen), seed=seed)


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


def lanczos3(offset: float) -> np.ndarray:
    """Lanczos-3 weights of the six pixels from two before to three after
    a point `offset` (0 to 1) past a pixel, summing to 1."""
    weights = np.sinc(np.arange(-2, 4) - offset) * np.sinc((np.arange(-2, 4) - offset) / 3)
    return weights / weights.sum()


def outliers(shape: tuple[int, int], rng: np.random.Generator, count: float) -> np.ndarray:
    """Outliers a few frames fail to reject, in units of σ̄, `count` per
    256 × 256 pixels on average: hot pixels (5 to 200σ̄), cold ones (−5 to
    −30σ̄) and cosmic-ray tracks 3 to 15 pixels long (10 to 300σ̄ along
    them). Each point is spread over the pixels around it as Lanczos-3
    registration spreads a frame's pixel, at a random sub-pixel offset."""
    height, width = shape
    plane = np.zeros((height + 6, width + 6), dtype=np.float32)
    for _ in range(rng.poisson(count * height * width / OUTLIER_AREA)):
        y, x = rng.uniform(0, height), rng.uniform(0, width)
        kind = rng.random()
        if kind < 0.6:
            points, amplitude = [(y, x)], math.exp(rng.uniform(math.log(5), math.log(200)))
        elif kind < 0.75:
            points, amplitude = [(y, x)], -math.exp(rng.uniform(math.log(5), math.log(30)))
        else:
            length, angle = rng.uniform(3, 15), rng.uniform(0, math.pi)
            steps = np.arange(int(length) + 1)
            points = list(zip(y + steps * math.sin(angle), x + steps * math.cos(angle)))
            amplitude = math.exp(rng.uniform(math.log(10), math.log(300)))
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
) -> tuple[torch.Tensor, ...]:
    """`size` tiles at random windows and offsets, each flipped and turned
    one of eight ways: the network's input, the target and A, in units of
    σ̄ above the background, and A's noise map in units of σ̄. With
    `map_jitter`, the noise map the network reads is scaled by a factor
    drawn log-uniformly from `exp(±map_jitter)` per tile, so the network
    learns to judge the noise from the image, the map telling it only
    where the noise is higher or lower. With `outlier_count`, A gains
    `outliers` and the target does not; the noise map, which a modelled
    map would not change, stays as it was."""
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
            a = a + outliers(a.shape, rng, outlier_count) * noise
        stated = sigma * float(np.exp(rng.uniform(-map_jitter, map_jitter))) if map_jitter else sigma
        inputs.append(network_input(a, stated, background, noise))
        targets.append(np.stack([(b - background) / noise]))
        references.append(np.stack([(a - background) / noise]))
        sigmas.append(np.stack([np.nan_to_num(sigma / noise, nan=1.0)]))
    return tuple(
        torch.from_numpy(np.ascontiguousarray(np.stack(values)))
        for values in (inputs, targets, references, sigmas)
    )


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
) -> dict[str, Any]:
    """Train a noise model on the train split's pairs, evaluate it on the
    validation split's, export it to ONNX and record the run.

    `extra` is a second bank from `cut_windows`, of other targets for
    their inputs; it makes up `extra_fraction` of every batch, and the run
    records it under `extra_label`. `outlier_count` adds that many
    synthetic outliers per 256 × 256 pixels to every input tile."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    chosen = pairs(manifest, "train", sets)
    if not chosen:
        raise RuntimeError("no built noise pairs in the train split")
    started = time.monotonic()
    banked = bank(store, chosen, seed=seed, noise_map=noise_map)
    bank_seconds = time.monotonic() - started
    where = device()
    model = ResidualUNet(width=width, levels=levels).to(where)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=steps, pct_start=0.05)
    history = []
    started = time.monotonic()
    running = 0.0
    extra_count = round(batch * extra_fraction) if extra is not None else 0
    odds = structure_odds(banked, structure_sampling)
    extra_odds = structure_odds(extra, structure_sampling) if extra is not None else None
    for step in range(1, steps + 1):
        parts = [batch_from(banked, batch - extra_count, tile, rng, map_jitter=map_jitter, odds=odds,
                            outlier_count=outlier_count)]
        if extra_count:
            parts.append(batch_from(extra, extra_count, tile, rng, map_jitter=map_jitter, odds=extra_odds,
                                    outlier_count=outlier_count))
        inputs, targets, references, sigmas = (torch.cat(values).to(where) for values in zip(*parts))
        loss = loss_of(model, inputs, targets, references, sigmas, loss_weight, huber, huber_above)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        running += float(loss.detach())
        if step % log_every == 0 or step == steps:
            history.append({"step": step, "loss": running / log_every, "seconds": round(time.monotonic() - started, 1)})
            print(f"train {name}: step {step} loss {running / log_every:.4f} ({time.monotonic() - started:.0f} s)", flush=True)
            running = 0.0
    train_seconds = time.monotonic() - started

    directory = store.root / "models" / "noise" / name
    directory.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    torch.save(model.state_dict(), directory / "weights.pt")
    # A margin of half the receptive field reproduces whole-image inference;
    # the cosine ramps over 32 more pixels.
    margin = math.ceil(receptive_field(levels) / 2)
    overlap = 2 * margin + 32
    validation = evaluate(store, manifest, model, split="validation", tile=512, overlap=overlap, margin=margin,
                          sets=validation_sets, noise_map=noise_map)
    validation["dark_noise"] = evaluate_dark_noise(store, manifest, model, split="validation", tile=512,
                                                   overlap=overlap, margin=margin, sets=validation_sets,
                                                   noise_map=noise_map)
    exported = export(model, directory / "model.onnx", tile=512)
    record = {
        "name": name,
        "created": now(),
        "config": {"width": width, "levels": levels, "tile": tile, "batch": batch, "steps": steps,
                   "learning_rate": learning_rate, "seed": seed, "train_pairs": len(chosen),
                   "windows": int(len(banked["a"])), "core_sigma": CORE_SIGMA,
                   "loss_weight": loss_weight, "map_jitter": map_jitter, "noise_map": noise_map,
                   "huber": huber, "huber_above": huber_above, "structure_sampling": structure_sampling,
                   "outliers": outlier_count,
                   "extra": {"label": extra_label, "fraction": extra_fraction, "windows": int(len(extra["a"]))}
                   if extra is not None else None},
        "parameters": sum(p.numel() for p in model.parameters()),
        "receptive_field_px": receptive_field(levels),
        "inference": {"tile": 512, "overlap": overlap, "margin": margin},
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
    model = model.to(where)

    def run(tiles: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            residual = model(torch.from_numpy(tiles).to(where)).cpu().numpy()
        return residual * noise

    return run


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
    limit: int | None = None,
    sets: set[str] | None = None,
    noise_map: str = "raw",
) -> dict[str, Any]:
    """Denoise each of a split's A halves, whole, through overlapping
    tiles, and measure it against its B half. The network reads the noise
    map it was trained with; every run is measured in the same units, the
    per-pixel map's median noise, so runs with different maps compare."""
    model = model.eval()
    where = device()
    per_pair = []
    for pair in pairs(manifest, split, sets)[:limit]:
        image_a, sigma_a = load_half(store, pair["a"])
        image_b, _ = load_half(store, pair["b"], noise_map=False)
        background, noise = scales(image_a, sigma_a)
        if noise_map == "model":
            coverage = np.load(store.root / loads(pair["a"]["files"])["coverage"])
            sigma_a = modelled_sigma(image_a, sigma_a, coverage)
        input_background, input_noise = scales(image_a, sigma_a)
        planes = network_input(image_a, sigma_a, input_background, input_noise)
        residual = infer(denoiser(model, input_background, input_noise, where), planes,
                         tile=tile, overlap=overlap, margin=margin)[0]
        estimate = image_a + residual
        recipe = loads(pair["a"]["recipe"])
        per_pair.append({"pair": pair["a"]["id"], "set": pair["set"], "size": recipe.get("size"),
                         **measure(estimate, image_a, image_b, background, noise)})
    by_size: dict[Any, list[dict[str, Any]]] = {}
    for row in per_pair:
        by_size.setdefault(row["size"], []).append(row)

    def mean(rows, key):
        values = [row[key] for row in rows if row[key] is not None]
        return float(np.mean(values)) if values else None

    keys = ("high_ratio", "low_ratio", "high_removed", "mse_before", "mse_after", "sky_bias", "bright_flux_change",
            "sky_mse", "sky_mse_trimmed", "sky_outlier_share", "faint_mse", "mid_mse", "core_mse",
            *(f"transfer_{region}_{band}" for region in ("faint", "mid") for band in TRANSFER_BANDS),
            "peak_flux_kept_faint", "peak_flux_kept_mid", "peak_height_kept_faint", "peak_height_kept_mid")
    return {
        "split": split,
        "pairs": len(per_pair),
        "by_size": {str(size): {key: mean(rows, key) for key in keys} for size, rows in sorted(by_size.items())},
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


def box_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """The mean over the (2·radius + 1)² box about each pixel, the edge
    pixels repeated outward. `image` must be finite."""
    if radius == 0:
        return image
    side = 2 * radius + 1
    blurred = np.pad(image.astype(np.float64), radius, mode="edge")
    for axis in (0, 1):
        total = np.insert(np.cumsum(blurred, axis=axis), 0, 0.0, axis=axis)
        blurred = (total[side:] - total[:-side]) / side if axis == 0 else (total[:, side:] - total[:, :-side]) / side
    return blurred.astype(np.float32)


def box_max(image: np.ndarray, radius: int) -> np.ndarray:
    """The largest value in the (2·radius + 1)² box about each pixel."""
    height, width = image.shape
    padded = np.pad(image, radius, mode="edge")
    rows = padded[:height]
    for shift in range(1, 2 * radius + 1):
        rows = np.maximum(rows, padded[shift:shift + height])
    largest = rows[:, :width]
    for shift in range(1, 2 * radius + 1):
        largest = np.maximum(largest, rows[:, shift:shift + width])
    return largest


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
    sets: set[str] | None = None,
    noise_map: str = "raw",
) -> dict[str, Any]:
    """Denoise each of a split's dark-noise inputs and measure it against
    its clean target (`truth`), and denoise its noise alone on a flat sky
    (`empty_sky`), once as it is and once with `outliers` planted in it.
    Inference and units are `evaluate`'s."""
    model = model.eval()
    where = device()
    rows = []
    for pair in dark_noise.pairs(manifest, split, sets):
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
        estimate = image + infer(run, network_input(image, stated, input_background, input_noise),
                                 tile=tile, overlap=overlap, margin=margin)[0]
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
        output = sky + infer(denoiser(model, sky_background, sky_noise, where),
                             network_input(sky, sky_sigma, sky_background, sky_noise),
                             tile=tile, overlap=overlap, margin=margin)[0]
        empty = on_empty_sky(output, sky, background, noise)
        # The same outliers for every model, planted on the empty sky: the
        # share of their energy the output keeps.
        planted = outliers(sky.shape, np.random.default_rng(zlib.crc32(pair["input"]["id"].encode())),
                           OUTLIER_TEST_COUNT) * sky_noise
        planted[~np.isfinite(sky)] = 0.0
        spoiled = sky + planted
        left = spoiled - output + infer(denoiser(model, sky_background, sky_noise, where),
                                        network_input(spoiled, sky_sigma, sky_background, sky_noise),
                                        tile=tile, overlap=overlap, margin=margin)[0]
        empty["outliers_left"] = float(np.nansum(left**2) / np.sum(planted**2))
        rows.append({"pair": pair["input"]["id"], "set": pair["set"],
                     "size": loads(pair["input"]["recipe"]).get("size"),
                     **{f"truth_{key}": value for key, value in truth.items()},
                     **{f"empty_{key}": value for key, value in empty.items()}})

    def mean(key):
        values = [row[key] for row in rows if row.get(key) is not None]
        return float(np.mean(values)) if values else None

    keys = sorted({key for row in rows for key in row if key.startswith(("truth_", "empty_"))})
    return {"split": split, "pairs": len(rows), "mean": {key: mean(key) for key in keys}, "pairs_detail": rows}


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


def evaluate_saved(
    store: Store, manifest: Manifest, name: str, *, split: str, limit: int | None = None, sets: set[str] | None = None
) -> dict[str, Any]:
    """Measure a recorded run's model on `split`, and add it to the record."""
    directory = store.root / "models" / "noise" / name
    record = json.loads((directory / "run.json").read_text())
    config = record["config"]
    model = ResidualUNet(width=config["width"], levels=config["levels"])
    model.load_state_dict(torch.load(directory / "weights.pt", map_location="cpu"))
    result = evaluate(store, manifest, model, split=split, limit=limit, sets=sets,
                      noise_map=config.get("noise_map", "raw"), **record["inference"])
    result["dark_noise"] = evaluate_dark_noise(store, manifest, model, split=split, sets=sets,
                                               noise_map=config.get("noise_map", "raw"), **record["inference"])
    record.setdefault("evaluations", {})[split] = result
    (directory / "run.json").write_text(json.dumps(record, indent=2))
    return {key: value for key, value in result.items() if key != "pairs_detail"} | {
        "dark_noise": {key: value for key, value in result["dark_noise"].items() if key != "pairs_detail"}
    }
