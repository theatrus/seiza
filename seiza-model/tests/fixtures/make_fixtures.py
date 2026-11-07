"""Write the harness's answers on a small synthetic image, so the crate's unit
tests hold its Rust to the Python a model was trained with.

Run from a training environment, with `training/` from the
`noise-model-training` branch until it merges:

    PYTHONPATH=training python seiza-model/tests/fixtures/make_fixtures.py

`image.f32` is the image, row-major little-endian floats. `expected.json` holds
each function's answer at sampled pixels, which keeps the fixture small.
"""

import json
from pathlib import Path

import numpy as np

import onnxruntime
import torch

from seiza_training import noise_model
from seiza_training.bundle import card, export_graph, sha256_of
from seiza_training.noise_model import NoiseBands, input_sigma, network_input, scales, seen, smooth_sky
from seiza_training.tiles import infer, origins, window
from seiza_training.unet import ResidualUNet, reach

HERE = Path(__file__).parent
WIDTH, HEIGHT = 300, 230
# Off every grid, so that reflection pads the tiles and blocks leave a remainder.
# The real sky block is larger than this image would use well.
FLAT_BLOCK = 64
STRIDE = 13
TILING = {"tile": 64, "overlap": 36, "margin": 8, "multiple": 16}
# The tiny model the runtime's integration tests run: small enough to commit.
TINY_WIDTH, TINY_LEVELS, TINY_RAMP, TINY_TILE = 4, 2, 8, 64


def sampled(plane: np.ndarray) -> list:
    """Every `STRIDE`th pixel of a plane, NaN as null."""
    values = plane[::STRIDE, ::STRIDE].astype(np.float64).ravel()
    return [None if np.isnan(value) else float(value) for value in values]


def image() -> np.ndarray:
    """A gradient, stars, a nebula on the right, a noisier strip on the left
    and a corner nothing covers."""
    rng = np.random.default_rng(5)
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    nebula = np.where(x > 150, 2000.0, 0.0)
    signal = 100.0 + 0.05 * x + 0.03 * y + nebula
    for cy, cx in rng.uniform((0, 0), (HEIGHT, WIDTH), (30, 2)):
        signal += 3000.0 * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * 1.5**2))
    sigma = np.sqrt(np.where(x < 32, 4.0, 1.0) * 4.0 + 0.05 * nebula)
    plane = (signal + rng.normal(0.0, 1.0, (HEIGHT, WIDTH)) * sigma).astype(np.float32)
    plane[:12, :12] = np.nan
    return plane


def local_filter(tiles: np.ndarray) -> np.ndarray:
    """A stand-in for a network that looks past a pixel, so that tile edges
    show unless the margins keep them out: the first plane less a quarter of
    its neighbours, zero past the tile, plus a tenth of the second plane."""
    first = tiles[:, 0]
    padded = np.pad(first, ((0, 0), (1, 1), (1, 1)))
    around = padded[:, :-2, 1:-1] + padded[:, 2:, 1:-1] + padded[:, 1:-1, :-2] + padded[:, 1:-1, 2:]
    return (first - 0.25 * around + 0.1 * tiles[:, 1])[:, None].astype(np.float32)


def tiny_model(planes: np.ndarray, noise: float) -> dict:
    """A tiny bundle of random weights, and what the harness makes of the
    image through its f32 graph on ONNX Runtime's CPU provider."""
    torch.manual_seed(5)
    model = ResidualUNet(width=TINY_WIDTH, levels=TINY_LEVELS).eval()
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    directory = HERE / "tiny"
    directory.mkdir(exist_ok=True)
    graphs = []
    for precision, dtype in (("f32", torch.float32), ("f16", torch.float16)):
        path = directory / f"model-{precision}.onnx"
        export_graph(model.to(dtype), path, dtype)
        graphs.append({"precision": precision, "file": path.name, "sha256": sha256_of(path)})
    margin = reach(TINY_LEVELS)
    tiling = {"tile": TINY_TILE, "overlap": 2 * margin + TINY_RAMP, "margin": margin, "multiple": 2 ** (TINY_LEVELS - 1)}
    record = {"config": {"noise_plane": "asinh", "sky": "flat"}, "inference": tiling}
    written = card("noise-tiny", record, "constant", graphs, ["cpu"])
    written["reading"]["flat_block"] = FLAT_BLOCK
    (directory / "card.json").write_text(json.dumps(written, indent=2) + "\n")

    options = onnxruntime.SessionOptions()
    options.add_free_dimension_override_by_name("height", TINY_TILE)
    options.add_free_dimension_override_by_name("width", TINY_TILE)
    session = onnxruntime.InferenceSession(str(directory / "model-f32.onnx"), options,
                                           providers=["CPUExecutionProvider"])

    def run(tiles: np.ndarray) -> np.ndarray:
        return np.concatenate([session.run(None, {"planes": tile[None]})[0] for tile in tiles]) * noise

    residual = infer(run, planes, **tiling, batch=1)[0]
    return {"width": WIDTH, "height": HEIGHT, "stride": STRIDE, "noise": float(noise), "residual": sampled(residual)}


def main() -> None:
    noise_model.FLAT_BLOCK = FLAT_BLOCK
    plane = image()
    plane.astype("<f4").tofile(HERE / "image.f32")

    mapped = input_sigma("image", plane, None, None)
    constant = input_sigma("constant", plane, None, None)
    sky = smooth_sky(plane)
    looked = seen(plane, "flat")
    background, noise = scales(looked, constant)
    planes = network_input(looked, constant, background, noise, "asinh")
    blended = infer(local_filter, planes, **TILING, batch=1)[0]
    residual = blended * np.float32(noise)
    expected = {
        "width": WIDTH,
        "height": HEIGHT,
        "stride": STRIDE,
        "flat_block": FLAT_BLOCK,
        "image_map_scale": noise_model.IMAGE_MAP_SCALE,
        "image_sigma": sampled(mapped),
        "constant": float(np.nanmedian(constant)),
        "smooth_sky": sampled(sky),
        "background": float(background),
        "noise": float(noise),
        "plane_image": sampled(planes[0]),
        "plane_noise": sampled(planes[1]),
        "origins": origins(200, 64, 48),
        "window": window(64, 48, 8, first=False, last=False).astype(np.float64).tolist(),
        "blended": sampled(blended),
        "blend_whole": sampled(NoiseBands().apply(plane, residual)),
        "blend_kept": sampled(NoiseBands(high=0.7, low=0.9, scale_px=5.0).apply(plane, residual)),
    }
    (HERE / "expected.json").write_text(json.dumps(expected, indent=0) + "\n")
    (HERE / "tiny-expected.json").write_text(json.dumps(tiny_model(planes, noise), indent=0) + "\n")


if __name__ == "__main__":
    main()
