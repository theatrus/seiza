"""Write the harness's noise map on a small image whose noise is spread over
neighbours, as a drizzled stack's is, so the crate's unit tests hold its
raise of the map (`reading.noise_correlation`) to the Python a model was
trained with.

Run from a training environment, with `training/` from the
`noise-model-training` branch until it merges:

    PYTHONPATH=training python seiza-model/tests/fixtures/make_spread_fixtures.py

`spread.f32` is the image, row-major little-endian floats. `spread.json`
holds the image's correlation, and the raised image map at sampled pixels.
"""

import json
from pathlib import Path

import numpy as np

from seiza_training.noise_model import (IMAGE_MAP_SCALE, NOISE_CORRELATION, SKY_BLOCK, input_sigma,
                                        noise_correlation)

HERE = Path(__file__).parent
WIDTH, HEIGHT = 240, 200
STRIDE = 13


def image() -> np.ndarray:
    """A gradient, a few stars and a corner nothing covers, under white noise
    each pixel of which is spread over a 2 × 2 box."""
    rng = np.random.default_rng(4)
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    signal = 100.0 + 0.03 * x + 0.02 * y
    for cy, cx in rng.uniform((0, 0), (HEIGHT, WIDTH), (10, 2)):
        signal += 3000.0 * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * 2.0**2))
    noise = rng.normal(0.0, 2.0, (HEIGHT + 1, WIDTH + 1))
    spread = (noise[:-1, :-1] + noise[1:, :-1] + noise[:-1, 1:] + noise[1:, 1:]) / 2
    plane = (signal + spread).astype(np.float32)
    plane[:8, :10] = np.nan
    return plane


def main() -> None:
    plane = image()
    plane.astype("<f4").tofile(HERE / "spread.f32")
    mapped = input_sigma("image", plane, None, None)
    expected = {
        "width": WIDTH,
        "height": HEIGHT,
        "stride": STRIDE,
        "sky_block": SKY_BLOCK,
        "image_map_scale": IMAGE_MAP_SCALE,
        "noise_correlation": NOISE_CORRELATION,
        "correlation": noise_correlation(plane),
        "image_sigma": [None if np.isnan(value) else float(value)
                        for value in mapped[::STRIDE, ::STRIDE].astype(np.float64).ravel()],
    }
    (HERE / "spread.json").write_text(json.dumps(expected, indent=0) + "\n")


if __name__ == "__main__":
    main()
