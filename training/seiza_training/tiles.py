"""Overlapping tiles: where they start, and how their outputs blend.

Every run cuts images into tiles, never feeding a model a whole image, so
memory and the exported graph's shape stay fixed whatever the image's size.
Within `margin` of a tile's edge, an output saw past the edge, into the
convolutions' zero padding, so it takes no weight there unless that edge is
the image's own, where whole-image inference would pad too. Past the
margin, a tile's weight rises as a raised cosine across the rest of the
overlap. Neighbouring tiles therefore overlap by at least twice the margin:
a margin of half the receptive field reproduces whole-image inference.

A network that halves the image `levels - 1` times samples it on a grid of
`2 ** (levels - 1)` pixels, so what it makes of a pixel depends on where
the tile holding it starts, unless every tile starts on that grid. Moved a
pixel, a trained noise model's tile changed by up to 0.8σ; moved 16, by
float error. Given the model's `multiple`, tiles start only on its grid:
the stride is cut to a multiple of it, and the image is padded by
reflection at its far edges until the last tile, flush with them, starts on
it too.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def origins(length: int, tile: int, overlap: int) -> list[int]:
    """Tile starts along one axis of `length` pixels: a stride of
    `tile - overlap`, with the last tile flush with the far edge."""
    if length <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)
    return starts


def window(tile: int, overlap: int, margin: int, *, first: bool = False, last: bool = False) -> np.ndarray:
    """A tile's blending weight along one axis: zero within `margin` of an
    inner edge, a raised cosine across the rest of the overlap, flat between.
    `first` and `last` mark edges that are the image's own."""
    ramp_length = overlap - 2 * margin
    if ramp_length < 1:
        raise ValueError(f"an overlap of {overlap} leaves no ramp past two margins of {margin}")
    ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(ramp_length, dtype=np.float32) + 0.5) / ramp_length)
    rising = np.concatenate([np.zeros(margin, dtype=np.float32), ramp])
    weight = np.ones(tile, dtype=np.float32)
    if not first:
        weight[: len(rising)] = rising
    if not last:
        weight[tile - len(rising):] = np.minimum(weight[tile - len(rising):], rising[::-1])
    return weight


def infer(
    run: Callable[[np.ndarray], np.ndarray],
    planes: np.ndarray,
    *,
    tile: int,
    overlap: int,
    margin: int,
    batch: int = 8,
    outputs: int = 1,
    multiple: int = 1,
) -> np.ndarray:
    """Run `run` over overlapping tiles of `planes`, (channels, height,
    width), and blend its (outputs, tile, tile) results into one image.
    `run` takes and returns a batch: (n, channels, tile, tile) to (n,
    outputs, tile, tile). Every tile starts on a multiple of `multiple`. An
    image smaller than a tile, or whose last tile would start off that grid,
    is padded by reflection, and the padding is cut away again."""
    if tile % multiple:
        raise ValueError(f"a tile of {tile} is not a multiple of {multiple}")
    # A stride cut to the grid widens the overlap, never narrows it.
    stride = (tile - overlap) // multiple * multiple
    if stride < 1:
        raise ValueError(f"an overlap of {overlap} leaves no stride of {multiple} in a tile of {tile}")
    overlap = tile - stride
    channels, height, width = planes.shape

    def padding(length: int) -> int:
        return tile - length if length <= tile else -(length - tile) % multiple

    pad_y, pad_x = padding(height), padding(width)
    if pad_y or pad_x:
        planes = np.pad(planes, ((0, 0), (0, pad_y), (0, pad_x)), mode="reflect")
    _, full_height, full_width = planes.shape
    total = np.zeros((outputs, full_height, full_width), dtype=np.float64)
    weights = np.zeros((full_height, full_width), dtype=np.float64)
    rows, columns = origins(full_height, tile, overlap), origins(full_width, tile, overlap)
    starts = [(y, x) for y in rows for x in columns]
    for begin in range(0, len(starts), batch):
        chunk = starts[begin:begin + batch]
        tiles = np.stack([planes[:, y:y + tile, x:x + tile] for y, x in chunk])
        results = run(tiles)
        for (y, x), result in zip(chunk, results):
            weight = np.outer(
                window(tile, overlap, margin, first=y == rows[0], last=y == rows[-1]),
                window(tile, overlap, margin, first=x == columns[0], last=x == columns[-1]),
            )
            total[:, y:y + tile, x:x + tile] += result * weight
            weights[y:y + tile, x:x + tile] += weight
    blended = total / np.maximum(weights, 1e-12)
    return blended[:, :height, :width].astype(np.float32)
