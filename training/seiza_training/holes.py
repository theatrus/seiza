"""Keep an image's holes away from a network.

A stack can hold pixels far below its sky that every frame holds, so no
rejection takes them out: hot pixels a dark took too much of, dead ones a
missing flat leaves. A network that meets one fills it and overshoots, and
turns the hole into a bright spike ringed by speckle. Whatever made a hole,
the model is not the place to decide it is a fault. So a model whose card
asks for it (`reading.holes`) never sees them: pixels deeper below the sky's
floor than any noise reaches are filled from their surroundings before the
network runs, and put back after, with the lobes registration spread round
them. Training with planted dark spots instead cost a fifth of the noise
taken out at four frames.

The runtime's `holes` module does the same, held to this one by the parity
fixtures, so the arithmetic is kept plain: order statistics over blocks and
boxes, and the image map, each easy to write again in Rust.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .noise_model import SKY_BLOCK, image_sigma, upsample

# Each block's floor is this percentile of its pixels, read between order
# statistics as numpy reads it by default.
FLOOR_PERCENTILE = 20.0


@dataclass(frozen=True)
class HoleGuard:
    """How a card's `reading.holes` keeps the holes away from a network."""

    # How far below the sky's floor a pixel is a hole, in the image map's
    # noise. No noise reaches it, and the cold outliers training plants stop
    # at 30σ̄, so a shallower one still reaches the network to be taken out.
    depth: float = 30.0
    # How far the mask reaches past a hole, in pixels: Lanczos-3
    # registration spreads a pixel this far.
    margin: int = 3
    # A masked pixel is filled with the median of the unmasked pixels in the
    # box this far round it: a star or a knot in the box barely moves it.
    fill_radius: int = 12
    # Within the mask, the input is put back where it differs from the
    # network's output by more than this many of the image map's noise: the
    # hole and its lobes, while the margin's ordinary noise keeps the
    # network's output.
    restore: float = 5.0

    def card(self) -> dict[str, Any]:
        return asdict(self)


def sky_floor(image: np.ndarray, block: int = SKY_BLOCK) -> np.ndarray:
    """The sky's floor under each pixel: the lowest of a block's
    `FLOOR_PERCENTILE` and its eight neighbours', spread bilinearly, and
    never above the image's background. The blocks are `image_sigma`'s; one
    a large star or a nebula's edge lifts is passed over for a neighbour on
    the sky, and a dark lane within a nebula wider than them, which stays
    above the sky, is not a hole."""
    height, width = image.shape
    rows, columns = max(1, height // block), max(1, width // block)
    plane = image.astype(np.float64)
    if height < rows * block or width < columns * block:
        plane = np.pad(plane, ((0, max(0, rows * block - height)), (0, max(0, columns * block - width))),
                       constant_values=np.nan)
    blocks = plane[:rows * block, :columns * block].reshape(rows, block, columns, block)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(rows, columns, block * block)
    finite = np.isfinite(blocks).any(axis=2)
    low = np.full((rows, columns), np.inf)
    if finite.any():
        low[finite] = np.nanpercentile(blocks[finite], FLOOR_PERCENTILE, axis=1)
    around = np.pad(low, 1, constant_values=np.inf)
    floor = np.min([around[1 + dy:1 + dy + rows, 1 + dx:1 + dx + columns]
                    for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=0)
    # The background as `image_sigma` measures it.
    sampled = plane[:height:7, :width:7]
    background = float(np.nanmedian(sampled)) if np.isfinite(sampled).any() else 0.0
    return np.minimum(upsample(np.where(np.isfinite(floor), floor, background), height, width, block), background)


def grown(mask: np.ndarray, radius: int) -> np.ndarray:
    """`mask` widened by `radius` pixels in every direction, to a square."""
    out = mask.copy()
    for axis in (0, 1):
        base = out.copy()
        length = base.shape[axis]
        for step in range(1, min(radius, length - 1) + 1):
            ahead = [slice(None), slice(None)]
            behind = [slice(None), slice(None)]
            ahead[axis], behind[axis] = slice(step, None), slice(None, -step)
            out[tuple(ahead)] |= base[tuple(behind)]
            out[tuple(behind)] |= base[tuple(ahead)]
    return out


def hole_mask(image: np.ndarray, guard: HoleGuard, noise_map: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """The pixels the network must not see: those deeper than `guard.depth`
    of `noise_map` below `floor`, widened by `guard.margin`. Where the map
    holds no noise, nothing is a hole."""
    finite = np.isfinite(image)
    noise = noise_map.astype(np.float64)
    with np.errstate(invalid="ignore"):
        deep = finite & (noise > 0) & (floor - image.astype(np.float64) > guard.depth * noise)
    if not deep.any():
        return deep
    return grown(deep, guard.margin) & finite


def filled(image: np.ndarray, mask: np.ndarray, guard: HoleGuard, floor: np.ndarray) -> np.ndarray:
    """`image` with each masked pixel the median of the unmasked ones within
    `guard.fill_radius`, cut at the image's edges, or the floor where there
    are none."""
    keep = np.isfinite(image) & ~mask
    out = image.astype(np.float32).copy()
    height, width = image.shape
    radius = guard.fill_radius
    for y, x in zip(*np.nonzero(mask)):
        rows = slice(max(0, y - radius), min(height, y + radius + 1))
        columns = slice(max(0, x - radius), min(width, x + radius + 1))
        around = image[rows, columns][keep[rows, columns]].astype(np.float64)
        out[y, x] = np.median(around) if around.size else floor[y, x]
    return out


def guarded(image: np.ndarray, guard: HoleGuard,
            block: int = SKY_BLOCK) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What the network reads in place of `image`, the mask of what it does
    not see, and the image map the restore is measured in. The image itself
    comes back when it holds no hole."""
    noise_map = image_sigma(image)
    floor = sky_floor(image, block)
    mask = hole_mask(image, guard, noise_map, floor)
    if not mask.any():
        return image, mask, noise_map
    return filled(image, mask, guard, floor), mask, noise_map


def guarded_pair(image_a: np.ndarray, image_b: np.ndarray, guard: HoleGuard,
                 block: int = SKY_BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """A pair as a model trains on it under the guard: A, the input, with
    its holes filled as the runtime fills them, and B, the target, filled
    where either half holds one. The network never sees a hole, as it never
    will, and the target never asks it for one it could not see; where A
    holds a hole, both halves hold their surroundings' median, which the
    network learns to leave be. The runtime puts the holes back after."""
    def mask_and_floor(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        floor = sky_floor(image, block)
        return hole_mask(image, guard, image_sigma(image), floor), floor

    mask_a, floor_a = mask_and_floor(image_a)
    mask_b, floor_b = mask_and_floor(image_b)
    either = mask_a | mask_b
    read_a = filled(image_a, mask_a, guard, floor_a) if mask_a.any() else image_a
    read_b = filled(image_b, either, guard, floor_b) if either.any() else image_b
    return read_a, read_b


def restored(image: np.ndarray, output: np.ndarray, mask: np.ndarray, noise_map: np.ndarray,
             guard: HoleGuard) -> np.ndarray:
    """`output`, with the input put back where the mask holds it and it
    differs from `output` by more than `guard.restore` of the map's noise."""
    with np.errstate(invalid="ignore"):
        back = mask & (np.abs(image.astype(np.float64) - output) > guard.restore * noise_map.astype(np.float64))
    return np.where(back, image, output).astype(np.float32)


def guarded_residual(image: np.ndarray, read: np.ndarray, found: np.ndarray, mask: np.ndarray,
                     noise_map: np.ndarray, guard: HoleGuard) -> np.ndarray:
    """What a network `found` in `read`, the guarded image, as a residual of
    `image`: the network's own outside the mask, and within it what takes
    `image` to the restored output. Shares of it are taken out as of any
    residual (`NoiseBands`)."""
    if not mask.any():
        return found
    output = restored(image, read + found, mask, noise_map, guard)
    return np.where(mask, output - image, found).astype(np.float32)
