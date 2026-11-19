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


# The PSF model's bundle.
#
# A sharpening bundle holds two graphs: the stamp encoder, run once for each
# node of the image's PSF field, and the body, run on every tile with the
# nodes' vectors interpolated to its bottleneck's pixels. Its card is a noise
# card's with `task` "sharpen" and a `sharpen` section, which names
# everything the runtime makes from the image for it: the PSF field and how
# it is measured, the point sources, the stamp the encoder reads, and how the
# residual is applied. The `reading` is the image map's, as `image_sigma`
# makes it, and the planes are named as the run's are, with dashes.


def sharpen_card(name: str, record: dict[str, Any], graphs: list[dict[str, str]], encoder: dict[str, Any],
                 parity: list[str], strengths: tuple[float, float] | None = None,
                 scaled: bool = True) -> dict[str, Any]:
    """A sharpening bundle's card, from the fields it may hold: `graphs`
    are the body's, one per precision, and `encoder` the encoder's file,
    hash and vector length. With `scaled`, the runtime also sharpens an
    image whose stars are far wider than `SCALE_TO` binned toward it, as
    the sheets' scaled pass does, and a model with the rest's plane asks
    `SCALED_FULL` at full size unless told."""
    from . import psf_fields
    from .multi_night import STACK_DETECTION
    from .noise_model import IMAGE_MAP_REVISION
    from .sharpen_bank import STAMP, WIDTH_CORE
    from .sharpen_model import (BODY_AXES, BODY_INPUTS, BODY_OUTPUT, ENCODER_INPUT, ENCODER_OUTPUT, PLANES, STAMP_PLANES,
                                WINGS, settings_from)
    from .sharpen_sheets import (FULL_NONSTELLAR, FULL_STRENGTH, SCALE_BLEND, SCALE_REST, SCALE_STARS, SCALE_TO,
                                 SCALED_FULL)

    unknown = [backend for backend in parity if backend not in BACKENDS]
    if unknown:
        raise ValueError(f"parity names backends the runtime knows, one of {BACKENDS}, not {unknown}")
    config, inference = record["config"], record["inference"]
    settings = settings_from(config["settings"])
    planes = tuple(config.get("planes", PLANES))
    if planes != settings.planes:
        raise ValueError(f"the run reads {planes}, and its settings say {settings.planes}")
    if IMAGE_MAP_REVISION != 2:
        raise ValueError(f"the runtime makes the image map of revision 2, not {IMAGE_MAP_REVISION}")
    own = (SCALED_FULL if scaled else FULL_NONSTELLAR) if settings.nonstellar else (FULL_STRENGTH, FULL_STRENGTH)
    stars, rest = strengths or own
    # The scaled pass: the stars' width an image is binned toward, the
    # share of its stars' width (as a Gaussian's σ) under which its change
    # is the binned run's, the rest's ask on the binned image, and the
    # star light, in pixel noise, from which a star keeps the full-size
    # change whole (`SCALE_STARS`).
    binned = ({"scaled": {"stars_px": SCALE_TO, "blend": SCALE_BLEND, "rest": SCALE_REST, "stars": SCALE_STARS}}
              if scaled else {})
    return {
        "format": CARD_FORMAT,
        "name": name,
        "task": "sharpen",
        "graphs": graphs,
        "inputs": [{"tensor": BODY_INPUTS[0], "planes": [plane.replace("_", "-") for plane in planes]}],
        "output": BODY_OUTPUT,
        "dimensions": list(BODY_AXES[BODY_INPUTS[0]].values()),
        "reading": {
            "noise_map": "image",
            "noise_plane": "asinh",
            "sky": "global",
            "sky_block": SKY_BLOCK,
            "flat_block": FLAT_BLOCK,
            "image_map_scale": IMAGE_MAP_SCALE,
            "noise_correlation": NOISE_CORRELATION,
        },
        "tiling": {
            "tile": inference["tile"],
            "margin": inference["margin"],
            "ramp": inference["overlap"] - 2 * inference["margin"],
            "multiple": inference["multiple"],
        },
        "parity": parity,
        "sharpen": {
            "encoder": {
                "file": encoder["file"],
                "sha256": encoder["sha256"],
                "input": ENCODER_INPUT,
                "output": ENCODER_OUTPUT,
                "planes": list(config.get("stamp_planes", STAMP_PLANES)),
                "side": STAMP,
                "wings": config.get("wings", WINGS),
                "features": encoder["features"],
            },
            "vector": {"tensor": BODY_INPUTS[1], "dimensions": list(BODY_AXES[BODY_INPUTS[1]].values())},
            "residual": settings.residual,
            # The narrowest a star is asked to be, in pixels, and the
            # sharpest ask the run's known-kernel windows were drawn at.
            "floor_px": settings.floor_px,
            "sharpest": 1.0 / max(settings.kernel_ratios),
            "strengths": {"stars": stars, "rest": rest},
            "psf": {**psf_fields.MEASURE_OPTIONS, "grid_px": psf_fields.GRID_PX, "width_core": WIDTH_CORE},
            "points": {
                "noise_reduction_radius": STACK_DETECTION["noise_reduction_radius"],
                "clip_peaks": psf_fields.CLIP_PEAKS,
                "peak_block": psf_fields.PEAK_BLOCK,
                "core_radius": psf_fields.CORE_RADIUS,
            },
            **binned,
        },
    }


def write_sharpen_bundle(store: Store, run: str, directory: Path, *, name: str, parity: tuple[str, ...] = (),
                         strengths: tuple[float, float] | None = None, scaled: bool = True) -> dict[str, Any]:
    """Write the sharpening `run` to `directory` as the bundle `name`: its
    encoder and its body (`sharpen_model.export_parts`), each held to
    PyTorch, and its card. `strengths` are the shares of their width stars
    and the rest are brought to when a caller names none; `scaled` gives
    the card the scaled pass (`sharpen_card`)."""
    from .sharpen_model import export_parts, load_run as load_sharpen_run

    record, model = load_sharpen_run(store, run)
    written = export_parts(model, directory)
    worst = max([written["encoder"]["max_relative_error"],
                 *(entry["max_relative_error"] for entry in written["body"] if "max_relative_error" in entry)])
    if worst > 1e-3:
        raise ValueError(f"an exported graph of {run} differs from PyTorch by {worst:.3g} of its output")
    graphs = [{"precision": entry["precision"], "file": entry["file"], "sha256": sha256_of(directory / entry["file"])}
              for entry in written["body"]]
    encoder = {"file": written["encoder"]["file"], "sha256": sha256_of(directory / written["encoder"]["file"]),
               "features": written["features"]}
    made = sharpen_card(name, record, graphs, encoder, list(parity), strengths, scaled)
    (directory / "card.json").write_text(json.dumps(made, indent=2) + "\n")
    return {"card": made, "export": written}
