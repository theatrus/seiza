"""Write the harness's hole guard on a small image holding holes, so the
crate's unit tests hold its Rust to the Python a card's guard was chosen
with.

Run from a training environment, with `training/` from the
`noise-model-training` branch until it merges:

    PYTHONPATH=training python seiza-model/tests/fixtures/make_hole_fixtures.py

`holes.f32` is the image, row-major little-endian floats. `holes.json` holds
the floor at sampled pixels, the mask whole, and what the network reads and
the residual it comes back as at every masked pixel, for a stand-in network
that takes every pixel to `LEVEL`.
"""

import json
from pathlib import Path

import numpy as np

from seiza_training.holes import HoleGuard, guarded, guarded_residual, sky_floor
from seiza_training.noise_model import IMAGE_MAP_SCALE, SKY_BLOCK

HERE = Path(__file__).parent
WIDTH, HEIGHT = 230, 170
STRIDE = 11
LEVEL = 100.0


def lanczos3(offset: float) -> np.ndarray:
    taps = np.arange(-2, 4) - offset
    weights = np.sinc(taps) * np.sinc(taps / 3)
    return weights / weights.sum()


def image() -> np.ndarray:
    """A gradient, a few stars, a corner nothing covers, two sharp holes as a
    drizzle leaves them, one registration spread, and a dip a network takes
    out itself."""
    rng = np.random.default_rng(9)
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    signal = 100.0 + 0.04 * x + 0.02 * y
    for cy, cx in rng.uniform((0, 0), (HEIGHT, WIDTH), (12, 2)):
        signal += 4000.0 * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * 1.6**2))
    plane = (signal + rng.normal(0.0, 2.0, (HEIGHT, WIDTH))).astype(np.float32)
    plane[:10, :14] = np.nan
    plane[40, 60] -= 2500.0
    plane[41, 60:62] -= 1800.0
    plane[2, 120] -= 900.0
    plane[120:126, 170:176] += (np.outer(lanczos3(0.3), lanczos3(0.6)) * -1200.0).astype(np.float32)
    plane[150, 30] -= 40.0
    return plane


def main() -> None:
    plane = image()
    plane.astype("<f4").tofile(HERE / "holes.f32")
    guard = HoleGuard()
    read, mask, noise_map = guarded(plane, guard)
    found = np.where(np.isfinite(read), np.float32(LEVEL) - read, np.nan).astype(np.float32)
    residual = guarded_residual(plane, read, found, mask, noise_map, guard)
    masked = np.flatnonzero(mask)
    floor = sky_floor(plane)
    expected = {
        "width": WIDTH,
        "height": HEIGHT,
        "stride": STRIDE,
        "sky_block": SKY_BLOCK,
        "image_map_scale": IMAGE_MAP_SCALE,
        "guard": guard.card(),
        "level": LEVEL,
        "floor": [float(value) for value in floor[::STRIDE, ::STRIDE].ravel()],
        "mask": masked.tolist(),
        "read": [float(read.flat[index]) for index in masked],
        "residual": [float(residual.flat[index]) for index in masked],
    }
    (HERE / "holes.json").write_text(json.dumps(expected, indent=0) + "\n")


if __name__ == "__main__":
    main()
