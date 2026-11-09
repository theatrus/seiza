"""Denoised planes for a sharpening pair bank, so its pairs can pass their
noise on (`sharpen_model.Settings.pass_noise`).

A seeing pair's halves each hold noise of their own, and a model trained
to make one from the other learns to take noise out. `clean` runs a noise
model over every window of a pair bank, input and target, each from the
window alone as the noise model reads any image, and keeps the results
beside the bank's arrays (`CLEAN_PLANES`, and `clean.json`). The trainer
then asks for the target's denoised plus the noise found in the input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .holes import HoleGuard, guarded, guarded_residual
from .noise_model import denoiser, device, input_sigma, load_run, view
from .sharpen_bank import open_bank
from .sharpen_model import CLEAN_PLANES
from .store import Store
from .tiles import infer

# The noise maps a network may read here: those made from the image alone.
IMAGE_MAPS = ("image", "constant")


def denoised(record: dict[str, Any], model: Any, image: np.ndarray, where: Any) -> np.ndarray:
    """`image` with all the noise the run finds taken out, a window as one
    tile and a larger image as the run tiles any, under the hole guard
    where the run trained under it."""
    config = record["config"]
    kind = config.get("noise_map", "raw")
    if kind not in IMAGE_MAPS:
        raise ValueError(f"{record['name']} reads a {kind} noise map; a window has only its image ({IMAGE_MAPS})")
    guard = HoleGuard() if config.get("holes") else None
    shown, mask, mapped = guarded(image, guard) if guard else (image, None, None)
    sigma = input_sigma(kind, shown, None, None)
    planes, background, noise = view(shown, sigma, sky=config.get("sky", "global"),
                                     noise_plane=config.get("noise_plane", "linear"))
    grid = record["inference"]["multiple"]
    if max(image.shape) <= record["inference"]["tile"] and image.shape[0] == image.shape[1]:
        # A window is the one tile, whole.
        inference = {"tile": image.shape[0], "overlap": grid, "margin": 0, "multiple": grid}
    else:
        inference = record["inference"]
    found = infer(denoiser(model, background, noise, where), planes, **inference)[0]
    if mask is not None:
        found = guarded_residual(image, shown, found, mask, mapped, guard)
    return (image + found).astype(np.float32)


def clean(store: Store, noise_run: str, directory: Path) -> dict[str, Any]:
    """Make a pair bank's denoised planes with `noise_run` of `store`; one
    made before with the same run is left."""
    directory = Path(directory)
    marker = directory / "clean.json"
    if marker.exists() and json.loads(marker.read_text()).get("noise_run") == noise_run:
        return {**json.loads(marker.read_text()), "kept": True}
    banked = open_bank(directory)
    if not banked["done"]["pairs"]:
        raise ValueError(f"{directory} is not a pair bank")
    record, model = load_run(store, noise_run)
    where = device()
    count = len(banked["info"])
    marker.unlink(missing_ok=True)
    for name, source in zip(CLEAN_PLANES, ("input", "target")):
        out = np.lib.format.open_memmap(directory / f"{name}.npy", mode="w+", dtype=np.float32,
                                        shape=banked[source].shape)
        for window in range(count):
            out[window] = denoised(record, model, np.asarray(banked[source][window], dtype=np.float32), where)
        out.flush()
    found = {"noise_run": noise_run, "windows": count, "parameters": record.get("parameters")}
    marker.write_text(json.dumps(found))
    return found
