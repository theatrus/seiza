"""A known-kernel bank from images sharper than any stack: mosaics from a
space telescope's public archive.

A stack's own windows, blurred, teach a model to give the stack back, and
nothing in them is sharper than the stack. An image from above the
atmosphere, brought down to a few of its pixels a pixel, is real sky whose
nebulae, dust and galaxies are sharp to the pixel: blurred by the PSFs of
real telescopes (`sharpen_model.Settings.real_kernels`), it is what a
ground-based stack of that sky would be, with the truth beside it.

`cut` reads drizzled FITS mosaics (the science image and its weight), and
for each of `bins` brings one down by that factor, through a round Gaussian
of half the factor's width and then every `bin`-th pixel, so no star is
squared by its pixels as a block mean squares it. It measures the result's
PSF field and point sources as the PSF pass measures a stack's, makes its
noise map from the image alone, and cuts windows wholly inside what the
mosaic covers well, into a bank in the known-kernel banks' layout
(`sharpen_bank.layout`), which `train-sharpen --more-kernel-bank` draws
from.
"""

from __future__ import annotations

import functools
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import psf_fields
from .noise_model import WINDOW, image_sigma, in_order, scales
from .sharpen_bank import (BANK_VERSION, STAMP, Bank, fitted_stamp, layout, saturation, stamp_width, summarize,
                           window_corners)

# A mosaic's pixels count where its weight is at least this share of its
# deep parts' (its 90th percentile): where few exposures overlap, it keeps
# their cosmic rays, which would be taught as sky.
WEIGHT_SHARE = 0.5
# What is left of them is found by its sharpness (`cosmic_rays`).
RAY_SIGMA = 6.0
RAY_SHARPNESS = 0.6
# A pixel brought down is kept where this share of its weight fell on
# pixels the mosaic holds: a cosmic ray's few pixels are filled from
# about them, a gap is not.
HELD_SHARE = 0.25
# Windows of each mosaic at each factor.
WINDOWS = 120
# The Gaussian a mosaic is smoothed by before every `bin`-th pixel is
# taken, as a share of the factor, in σ.
PREFILTER = 0.5
# Without stars enough for a field, the PSF is a round Gaussian of this
# FWHM in the mosaic's own pixels, widened as the prefilter widens it.
NATIVE_FWHM = 2.4
# A window may miss this share of its pixels (cosmic rays, a chip gap's
# end): required whole, over half the Hubble mosaics gave no window and
# the rest a few. What is missing is filled from about it (`filled_holes`)
# and marked saturated, which keeps it out of the loss.
HOLES = 0.02
# The Gaussians, in σ px, the holes are filled by, narrowest first.
FILLS = (1.5, 4.0, 12.0)


def filled_holes(image: np.ndarray) -> np.ndarray:
    """`image` with what is not finite filled by the mean of what is held
    about it, by the narrowest of `FILLS` that reaches held pixels, and by
    the image's median where none does."""
    from .sharpen_model import blurred_images

    holes = ~np.isfinite(image)
    if not holes.any():
        return image
    out = image.copy()
    held = np.where(holes, 0.0, image).astype(np.float32)
    weight = (~holes).astype(np.float32)
    for sigma in FILLS:
        left = ~np.isfinite(out)
        if not left.any():
            break
        pad = int(4 * sigma) + 2
        values, share = (blurred_images(plane, [sigma**2 * np.eye(2)], pad=pad)[0] for plane in (held, weight))
        reached = left & (share > 0.05)
        out[reached] = values[reached] / share[reached]
    out[~np.isfinite(out)] = float(np.median(image[~holes])) if (~holes).any() else 0.0
    return out


def brought_down(image: np.ndarray, factor: int) -> np.ndarray:
    """`image` at one pixel in `factor`: smoothed by a round Gaussian of
    `PREFILTER` × `factor` px, in strips to bound memory, NaN kept NaN where
    most of a pixel's weight was, then every `factor`-th pixel."""
    from .sharpen_model import blurred_images

    variance = (PREFILTER * factor) ** 2
    finite = np.isfinite(image)
    filled = np.where(finite, image, 0.0).astype(np.float32)
    weight = finite.astype(np.float32)
    rows = []
    strip, pad = 2048, int(4 * PREFILTER * factor) + 2
    start = factor // 2
    for top in range(0, image.shape[0], strip):
        low, high = max(0, top - pad), min(image.shape[0], top + strip + pad)
        values, share = (blurred_images(plane[low:high], [variance * np.eye(2)], pad=pad)[0]
                         for plane in (filled, weight))
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(share > HELD_SHARE, values / share, np.nan)
        rows.append(out[top - low:top - low + min(strip, image.shape[0] - top)])
    return np.concatenate(rows)[start::factor, start::factor].astype(np.float32)


class MeasuredField:
    """An image's PSF as `sharpen_bank` reads a recorded field's: the field
    measured from it, or, where its stars are too few, one round Gaussian."""

    def __init__(self, image: np.ndarray, fallback_fwhm: float) -> None:
        self.table, self.level, _ = psf_fields.detections(image)
        self.field, self.status, _ = psf_fields.measure_field(image)
        sigma = fallback_fwhm / 2.3548
        middle = (STAMP - 1) / 2
        down, across = np.mgrid[0:STAMP, 0:STAMP] - middle
        round_stamp = np.exp(-(across**2 + down**2) / (2 * sigma**2))
        self.fallback = (round_stamp / round_stamp.sum()).astype(np.float32)
        self.fallback_fwhm = fallback_fwhm

    def stamp_at(self, x: float, y: float) -> np.ndarray:
        if self.field is None:
            return self.fallback
        return np.asarray(self.field.stamp_at(float(x), float(y)), dtype=np.float32)

    def shape_at(self, x: float, y: float) -> dict[str, float]:
        shape = None if self.field is None else self.field.shape_at(float(x), float(y))
        if shape is None:
            return {"fwhm": self.fallback_fwhm if self.field is None else float("nan"), "e1": 0.0, "e2": 0.0,
                    "beta": float("nan")}
        return {name: float(getattr(shape, name)) for name in ("fwhm", "e1", "e2", "beta")}


def cosmic_rays(image: np.ndarray, strip: int = 1024) -> np.ndarray:
    """Where a mosaic holds what no star is: a pixel far above its four
    neighbours' mean, by `RAY_SIGMA` of the image's noise, and by more than
    `RAY_SHARPNESS` of its height above the ring two pixels out. A star's
    middle stands 0.4 of that above its neighbours at this sampling; a
    cosmic ray's hit or track, all of it. Grown by a pixel."""
    from .sharpen_bank import grown

    out = np.zeros(image.shape, dtype=bool)
    ring = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if max(abs(dy), abs(dx)) == 2]
    for top in range(0, image.shape[0], strip):
        low, high = max(0, top - 2), min(image.shape[0], top + strip + 2)
        held = np.isfinite(image[low:high])
        if not held.any():
            continue
        part = np.nan_to_num(image[low:high], nan=0.0)
        padded = np.pad(part, 2, mode="edge")
        height, width = part.shape

        def moved(dy: int, dx: int) -> np.ndarray:
            return padded[2 + dy:2 + dy + height, 2 + dx:2 + dx + width]

        near = (moved(-1, 0) + moved(1, 0) + moved(0, -1) + moved(0, 1)) / 4
        far = np.median(np.stack([moved(dy, dx) for dy, dx in ring[::2]]), axis=0)
        above = part - near
        sample = above[held][::7]
        noise = 1.4826 * float(np.median(np.abs(sample - np.median(sample)))) + 1e-12
        with np.errstate(divide="ignore", invalid="ignore"):
            found = held & (above > RAY_SIGMA * noise) & (above > RAY_SHARPNESS * (part - far))
        out[top:top + min(strip, image.shape[0] - top)] = found[top - low:top - low + min(strip,
                                                                                          image.shape[0] - top)]
    return grown(out, 1)


def loaded(path: Path) -> np.ndarray:
    """A mosaic's science image, NaN where its weight is thin or none and
    at its cosmic rays."""
    from astropy.io import fits

    with fits.open(path, memmap=True) as file:
        image = np.asarray(file["SCI"].data, dtype=np.float32)
        weight = np.asarray(file["WHT"].data, dtype=np.float32) if "WHT" in file else None
    covered = np.isfinite(image) & (image != 0)
    if weight is not None and (weight > 0).any():
        sample = weight[::8, ::8]
        covered &= weight >= WEIGHT_SHARE * float(np.percentile(sample[sample > 0], 90))
    image = np.where(covered, image, np.nan)
    image[cosmic_rays(image)] = np.nan
    return image


def prepared(entry: dict[str, Any], root: Path, factor: int, seed: int, per_source: int) -> dict[str, Any]:
    """One mosaic at one factor, as `Bank.add` takes a source's windows."""
    name = f"external/{entry['path']}/b{factor}"
    record: dict[str, Any] = {"input": name, "input_kind": "external", "set": entry.get("target", entry["path"]),
                              "split": "train", "telescope": entry.get("telescope", "external"),
                              "pixel_scale_arcsec": None, "record": {"bin": factor, "filters": entry.get("filters")}}
    image = brought_down(loaded(root / entry["path"]), factor)
    field = MeasuredField(image, float(np.hypot(NATIVE_FWHM, 2.3548 * PREFILTER * factor) / factor))
    record["input_psf"] = {"status": field.status, "clip_level": field.level, "detections": len(field.table)}
    finite = np.isfinite(image)
    corners = window_corners(finite, random.Random(f"{seed}/{name}"), per_source, HOLES)
    if not corners:
        return {"status": "dropped", "record": {**record, "reason": "no-window"}}
    saturated = saturation(image, field.level) | ~finite
    image = filled_holes(image)
    sigma = image_sigma(image)
    background, noise = scales(image, np.where(finite, sigma, np.nan))
    names, _, columns = layout(False)
    planes = {"image": image, "sigma": sigma, "saturated": saturated}
    cut_planes = {plane: np.stack([planes[plane][y:y + WINDOW, x:x + WINDOW] for y, x in corners]).astype(names[plane])
                  for plane in planes}
    stamps, info = [], np.zeros((len(corners), len(columns)), dtype=np.float32)
    for row, (y, x) in enumerate(corners):
        centre = (x + (WINDOW - 1) / 2, y + (WINDOW - 1) / 2)
        stamp, _ = fitted_stamp(field.stamp_at(*centre))
        shape = field.shape_at(*centre)
        stamps.append(stamp)
        values = {"y": y, "x": x, "background": background, "noise": noise,
                  "saturated": float(cut_planes["saturated"][row].mean()), **shape, "width": stamp_width(stamp)}
        info[row] = [values.get(column, 0.0) for column in columns]
    record["stamp_lost"], record["background"], record["noise"] = 0.0, background, noise
    return {"status": "cut", "record": record, "planes": cut_planes, "stamps": {"stamp": np.stack(stamps)},
            "info": info, "points": field.table}


def attempted(root: str, seed: int, per_source: int, job: tuple[dict[str, Any], int]) -> dict[str, Any]:
    """`prepared` in a process of its own; an error is a dropped source,
    so one bad mosaic does not stop the cut."""
    entry, factor = job
    try:
        return prepared(entry, Path(root), factor, seed, per_source)
    except Exception as error:  # noqa: BLE001
        return {"status": "dropped", "record": {"input": entry["path"], "reason": str(error)}}


def cut(root: Path, work: Path, *, bins: tuple[int, ...] = (2, 3), per_source: int = WINDOWS, seed: int = 0,
        limit: int | None = None, workers: int = 1, telescope: str = "external") -> dict[str, Any]:
    """Cut the bank of the mosaics `root`/index.json lists (each entry's
    `path` under `root`), each at each of `bins`, under `work`; one cut
    before from the same list is opened."""
    from .sharpen_bank import open_bank

    index = json.loads((root / "index.json").read_text())
    entries = [{**entry, "telescope": entry.get("telescope", telescope)} for _, entry in sorted(index.items())][:limit]
    parts = [BANK_VERSION, "external", seed, WINDOW, STAMP, per_source, list(bins), PREFILTER, HOLES,
             [entry["path"] for entry in entries]]
    name = "sharpen-external-train-" + hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16]
    directory = work / name
    bank = Bank(directory, pairs=False, capacity=len(entries) * len(bins) * per_source)
    if not bank.done:
        dropped = []
        began = time.monotonic()
        jobs = [(entry, factor) for entry in entries for factor in bins]
        prepare = functools.partial(attempted, str(root), seed, per_source)
        for done, ((entry, factor), result) in enumerate(zip(jobs, in_order(prepare, jobs, workers)), start=1):
            if result["status"] == "dropped":
                dropped.append(result["record"])
            else:
                bank.add(result)
            print(json.dumps({"bank": name, "done": done, "of": len(jobs), "input": entry["path"],
                              "bin": factor, "status": result["status"],
                              "reason": result["record"].get("reason"), "windows": bank.count,
                              "elapsed": round(time.monotonic() - began)}), file=sys.stderr, flush=True)
        bank.finish({"label": "external", "split": "train", "seed": seed, "per_source": per_source,
                     "dropped": dropped, "summary": summarize(bank, False)})
    return open_bank(directory)
