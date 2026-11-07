"""Write a trained noise model as a bundle Seiza's runtime opens.

A bundle holds one graph per precision and a card. A graph's height and width
stay free, so one file serves every tile size: the runtime fixes them when it
opens the graph. It is traced on a small sample, since a half-precision trace
at a full tile takes the CPU far longer.

The card is written from the fields below, never by copying `run.json`, which
holds machine paths and the whole training configuration. It names everything
that must match training, because a model given the wrong reading fails badly.

A shipped model takes an image and nothing else, so a card names only noise
inputs made from the image. A model trained on the stack's map can read one in
its place, which `evaluate-noise --noise-map` measures first.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .noise_model import FLAT_BLOCK, IMAGE_MAP_SCALE, SKY_BLOCK, load_run
from .store import Store

CARD_FORMAT = 1
# What a card may name: the rest need a stack's own records.
IMAGE_ONLY = ("constant", "image")
SAMPLE_SIDE = 64
AXES = {"planes": {2: "height", 3: "width"}, "residual": {2: "height", 3: "width"}}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def export_graph(model: Any, path: Path, dtype: Any) -> None:
    """Write `model` as ONNX with its height and width free."""
    import torch

    sample = torch.zeros(1, 2, SAMPLE_SIDE, SAMPLE_SIDE, dtype=dtype)
    torch.onnx.export(model, (sample,), str(path), input_names=["planes"], output_names=["residual"],
                      opset_version=17, dynamo=False, dynamic_axes=AXES)


def card(name: str, record: dict[str, Any], noise_map: str, graphs: list[dict[str, str]],
         parity: list[str]) -> dict[str, Any]:
    """A bundle's card, from the fields it may hold."""
    config, inference = record["config"], record["inference"]
    return {
        "format": CARD_FORMAT,
        "name": name,
        "task": "denoise",
        "graphs": graphs,
        "input": "planes",
        "input_channels": 2,
        "output": "residual",
        "dimensions": ["height", "width"],
        "reading": {
            "noise_map": noise_map,
            "noise_plane": config.get("noise_plane", "linear"),
            "sky": config.get("sky", "global"),
            "sky_block": SKY_BLOCK,
            "flat_block": FLAT_BLOCK,
            "image_map_scale": IMAGE_MAP_SCALE,
        },
        "tiling": {
            "tile": inference["tile"],
            "margin": inference["margin"],
            "ramp": inference["overlap"] - 2 * inference["margin"],
            "multiple": inference["multiple"],
        },
        "parity": parity,
    }


def write_bundle(store: Store, run: str, directory: Path, *, name: str, noise_map: str | None = None,
                 parity: tuple[str, ...] = ()) -> dict[str, Any]:
    """Write `run` to `directory` as the bundle `name`, its network reading
    `noise_map`, or the map it trained with when that is image-only."""
    import torch

    record, model = load_run(store, run)
    chosen = noise_map or record["config"].get("noise_map", "raw")
    if chosen not in IMAGE_ONLY:
        raise ValueError(f"a shipped model reads one of {IMAGE_ONLY}, and {run} trained with {chosen!r}: "
                         "say which it reads instead")
    directory.mkdir(parents=True, exist_ok=True)
    graphs = []
    for precision, dtype in (("f32", torch.float32), ("f16", torch.float16)):
        path = directory / f"model-{precision}.onnx"
        export_graph(model.to(dtype), path, dtype)
        graphs.append({"precision": precision, "file": path.name, "sha256": sha256_of(path)})
    written = card(name, record, chosen, graphs, list(parity))
    (directory / "card.json").write_text(json.dumps(written, indent=2) + "\n")
    return written
