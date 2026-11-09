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

What the runtime does around the network is the card's too: with
`reading.holes`, it keeps the image's holes from the network (`holes`), and a
card without it runs the network on the image as it is.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from .holes import HoleGuard
from .noise_model import FLAT_BLOCK, IMAGE_MAP_SCALE, NOISE_CORRELATION, SKY_BLOCK, load_run
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


# The backends a card may name, as seiza-model's `Backend` spells them: a
# name it does not know would leave a card no backend may run.
BACKENDS = ("neural-engine", "coreml-gpu", "coreml-cpu", "tensorrt", "tensorrt-rtx", "cuda", "webgpu", "cpu")


def card(name: str, record: dict[str, Any], noise_map: str, graphs: list[dict[str, str]],
         parity: list[str], holes: HoleGuard | None = None) -> dict[str, Any]:
    """A bundle's card, from the fields it may hold."""
    unknown = [backend for backend in parity if backend not in BACKENDS]
    if unknown:
        raise ValueError(f"parity names backends the runtime knows, one of {BACKENDS}, not {unknown}")
    config, inference = record["config"], record["inference"]
    reading = {
        "noise_map": noise_map,
        "noise_plane": config.get("noise_plane", "linear"),
        "sky": config.get("sky", "global"),
        "sky_block": SKY_BLOCK,
        "flat_block": FLAT_BLOCK,
        "image_map_scale": IMAGE_MAP_SCALE,
        # The image map is raised by as much as an image's correlation
        # exceeds this, as `image_sigma` does.
        "noise_correlation": NOISE_CORRELATION,
    }
    if holes is not None:
        reading["holes"] = holes.card()
    return {
        "format": CARD_FORMAT,
        "name": name,
        "task": "denoise",
        "graphs": graphs,
        "inputs": [{"tensor": "planes", "planes": ["image", "noise"]}],
        "output": "residual",
        "dimensions": ["height", "width"],
        "reading": reading,
        "tiling": {
            "tile": inference["tile"],
            "margin": inference["margin"],
            "ramp": inference["overlap"] - 2 * inference["margin"],
            "multiple": inference["multiple"],
        },
        "parity": parity,
    }


def write_bundle(store: Store, run: str, directory: Path, *, name: str, noise_map: str | None = None,
                 parity: tuple[str, ...] = (), holes: bool | None = None) -> dict[str, Any]:
    """Write `run` to `directory` as the bundle `name`, its network reading
    `noise_map`, or the map it trained with when that is image-only, and
    the runtime keeping holes from it as the run trained, or as `holes`
    says."""
    import torch

    record, model = load_run(store, run)
    trained = bool(record["config"].get("holes"))
    guard = trained if holes is None else holes
    if guard != trained:
        print(f"bundle {name}: the runtime {'keeps holes from' if guard else 'shows holes to'} a network trained "
              f"{'with' if trained else 'without'} the guard", file=sys.stderr)
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
    written = card(name, record, chosen, graphs, list(parity), HoleGuard() if guard else None)
    (directory / "card.json").write_text(json.dumps(written, indent=2) + "\n")
    return written
