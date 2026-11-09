"""The sharpening bank: windows of the corpus's stacks, cut once onto local
disk, from which the PSF model's trainer draws its tiles, as the noise
model's trainer draws from its bank (`noise_model.bank`).

Two kinds of bank are cut, each split's apart, so a validation set's
windows never reach a train bank:

- **Known-kernel windows** (`deep`), of the deep stacks, a night's and
  many nights': the image, the stack's noise map (`sqrt(variance /
  coverage)`) and a saturation mask, with the local PSF stamp and shape at
  each window's centre. The trainer blurs them by a kernel K, and renders
  the target at the stack's own PSF or a narrower one, so only the sharp
  windows are stored.
- **Pairs** (`seeing`, and any a caller supplies): an input and a target
  on one grid. A seeing pair's worst half is the input and its best half
  the target. Each window holds both images, the input's noise map made
  from its image alone (`image_sigma`, of the whole image, as the runtime
  makes it), both stacks' own noise maps, for the loss to weigh by, and a
  saturation mask over either; with the stamp and shape of both halves'
  PSF at its centre and the ratio FWHM(target) / FWHM(input) there, the
  value the ratio map takes, both as the field's Moffat fit has it and as
  the stamps' own widths do (`stamp_width`), since the fit fails on the
  sharpest stamps. Each pair keeps the median offset of its stars'
  centroids, target from input: halves on one grid differ by as much as
  their PSFs differ in symmetry, which a stamp centred on its centroid
  does not show. A caller's pairs carry a record of their own, kept with
  their windows.

What a network reads must be made from the image alone: the PSF stamp
(from the input's PSF field), the noise map (`input_map`) and the
point-source map, drawn from the input's detections (each source's
`points`) until the Gaia pass gives a catalogue's. The stacks' own noise
maps and coverage are for training only: loss weights and matching.

A seeing pair is taken only when its halves differ in the PSF alone. One
the multi-night pass judged (`seeing_check`) is taken as the manifest
records it, and its target is put on its input's flux and sky by the gain
and the planes the check found: a trainer must match them, as
normalization brightens a frame as its seeing worsens. One made before
the check, a night's split among them, is judged here the same way, from
its stacks and the stars the PSF pass found on them, and the verdict kept
with the bank, not in the manifest.

The windows are cut from raw stacks. The sharpener is to train on the
noise model's output later; those windows will be cut from the denoised
stacks, beside these.

Saturated cores are masked from every loss on real images. The stacks
keep no saturation plane, so the mask is where either image reaches its
clip level, the level at which its brightest stars' cores pile up
(`psf_fields.clip_level`), grown by `SATURATION_GROW`.

A bank is written into memory-mapped `.npy` files under a work directory
on local disk, and read back the same way. Each source's windows are drawn
at random from an overlapping grid, wholly inside both images' coverage,
by a generator seeded by the bank's seed and the source's input, so
`workers` sources are read and prepared at once, in processes of their
own, and the bank is the same whatever their number. A cut records how far
it has come every `CHECKPOINT_SECONDS`, and one stopped part way goes on
from there when asked again, to the same bank.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import psf_fields
from .manifest import Manifest, loads
from .noise_model import CHECKPOINT_SECONDS, WINDOW, image_sigma, in_order, scales
from .store import Store

# Raised with the layout of a bank or what goes into it, so a bank cut
# before is cut again. 2: stamp widths, and pairs kept whatever their
# centroids' offset short of `ALIGNMENT_PX`.
BANK_VERSION = 2
# The side of the stamps a bank keeps: a field's stamp is seven times its
# FWHM, so this holds whole the stamps of fields up to 18.4 px, the widest
# the corpus's worst seeing halves measure (a C925 night's, 14.9–18.4 px
# across its field, 2026-11-08). A smaller stamp is padded with zeros; a
# larger one is cut to it, its sum put back to one, and the share it lost
# recorded.
STAMP = 129
# Windows of each deep stack, as the noise bank takes of each pair.
DEEP_WINDOWS = 40
# Windows of each seeing pair: a few dozen pairs to the deep stacks' many
# hundred, so each gives more.
PAIR_WINDOWS = 160
# Pixels a saturated core's mask is grown by: the pixels next to a clipped
# core are lifted by it, from the frames whose cores clipped.
SATURATION_GROW = 2
# A pair whose stars move between its halves by more than this, as the
# median of the offsets of the stars found in both, is not on one grid.
# Halves on one grid still move their stars' centroids where their PSFs
# differ in shape: a lopsided PSF pulls its centroid from its peak, and a
# sharper one pulls it less. The C925's seeing halves, and deconvolutions
# of its stacks, move them by up to a pixel, all one way (2026-11-08), so
# only a larger move is taken for a fault; each pair's offset is kept with
# it.
ALIGNMENT_PX = 1.5
# `stamp_width` fits a round Gaussian to the stamp's pixels from this share
# of its peak up: the core, where a Moffat's wings do not yet pull it wide.
WIDTH_CORE = 0.05
# The Gaussian widths, σ in pixels, it tries before refining the best.
WIDTH_SIGMAS = np.geomspace(0.25, 30.0, 300)

PAIR_PLANES = {"input": np.float32, "target": np.float32, "input_sigma": np.float32, "input_map": np.float32,
               "target_sigma": np.float32, "saturated": np.uint8}
DEEP_PLANES = {"image": np.float32, "sigma": np.float32, "saturated": np.uint8}
PAIR_STAMPS = ("input_stamp", "target_stamp")
DEEP_STAMPS = ("stamp",)
# Each window's row of `info`: its source's index in the bank, its corner,
# the background and median noise the network's units are (of the map it
# reads), the share of it saturated, and the PSF's shape at its centre:
# the field's Moffat fit (NaN where it fails, as on stamps sharper than
# about 2 px) and the stamp's own width (`stamp_width`), which never fails,
# with each one's ratio, target over input.
PAIR_INFO = ("source", "y", "x", "background", "noise", "saturated", "input_fwhm", "input_e1", "input_e2",
             "input_beta", "target_fwhm", "target_e1", "target_e2", "target_beta", "ratio", "input_width",
             "target_width", "width_ratio")
DEEP_INFO = ("source", "y", "x", "background", "noise", "saturated", "fwhm", "e1", "e2", "beta", "width")


def layout(pairs: bool) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    """A bank's planes, stamps and info columns."""
    return (PAIR_PLANES, PAIR_STAMPS, PAIR_INFO) if pairs else (DEEP_PLANES, DEEP_STAMPS, DEEP_INFO)


def deep_sources(manifest: Manifest, split: str, sets: set[str] | None = None) -> list[dict[str, Any]]:
    """The known-kernel bank's sources of one split: every deep stack, a
    night's and many nights', of a set calibrated with a flat."""
    return [{"input": row, "record": {}} for row in
            psf_fields.eligible(manifest, kinds=("deep", "multi-night"), splits=(split,), sets=sets)]


def seeing_pairs(manifest: Manifest, split: str, sets: set[str] | None = None) -> tuple[list[dict[str, Any]],
                                                                                         list[dict[str, Any]]]:
    """The seeing pairs of one split, as sources of a pair bank, worst half
    the input and best the target, and those left out with the reason. A
    pair the manifest records as judged is taken as it records it, its
    gain with it; one made before the check is judged when it is cut."""
    from .multi_night import screened

    found, left = [], []
    for best in psf_fields.eligible(manifest, kinds=("seeing-best", "multi-night-seeing-best"), splits=(split,),
                                    sets=sets):
        worst = psf_fields.annotated(manifest, [best["id"].removesuffix("/best") + "/worst"])
        recipe = loads(best["recipe"], {})
        name = {"pair": best["id"].removesuffix("/best"), "kind": best["kind"].removesuffix("-best")}
        if not worst or worst[0]["status"] != "built":
            left.append({**name, "reason": "worst-not-built"})
            continue
        record = {**name, "object": recipe.get("object"), "frames_each": recipe.get("frames_each"),
                  "best_fwhm_px": recipe.get("best_fwhm_px"), "worst_fwhm_px": recipe.get("worst_fwhm_px")}
        check = recipe.get("check")
        if check is not None:
            found.append({"input": worst[0], "target": best, "match": True, "gain": check.get("flux_gain"),
                          "judge": False, "record": {**record, "judged": "manifest", "check": check}})
            continue
        spread = (recipe["worst_fwhm_px"] - recipe["best_fwhm_px"]) / recipe["best_fwhm_px"]
        if not screened(4 * recipe["frames_each"], spread):
            left.append({**name, "reason": "seeing-spread"})
            continue
        found.append({"input": worst[0], "target": best, "match": True, "gain": None, "judge": True,
                      "record": {**record, "judged": "bank"}})
    rejected = manifest.rows(
        "SELECT p.id, p.reason FROM products p JOIN sets s ON s.id = p.set_id WHERE p.kind IN "
        "('seeing-best', 'multi-night-seeing-best') AND p.status = 'rejected' AND s.split = ? "
        "AND s.calibration_status = 'ready' ORDER BY p.id", (split,))
    left += [{"pair": row["id"].removesuffix("/best"), "reason": f"manifest:{row['reason']}"} for row in rejected]
    return found, left


# A binned bank's stacks are binned so their stars come to about this many
# pixels across, by no less than 2 and no more than `BIN_MOST`.
BINNED_PX = 1.6
BIN_MOST = 6


def binned_sources(store: Store, manifest: Manifest, split: str, sets: set[str] | None = None) -> list[dict[str, Any]]:
    """The known-kernel bank's sources again, each to be binned (its
    record's `bin`) until its stars are about `BINNED_PX` across: real sky
    whose stars and detail are as sharp, in pixels, as the sharpest any
    image holds. A stack's own windows, blurred, give inputs no sharper
    than itself, and the model met no input as sharp as a well-sampled
    stack asked for its floor; asked on one, it over-reached, raising
    bright cores 3.4 times where 2.4 keeps their flux and digging the
    difference out of a ring about them (2026-11-09)."""
    found = []
    for source in deep_sources(manifest, split, sets):
        record = psf_fields.recorded(store, source["input"]["id"])
        width = ((record or {}).get("fwhm_px") or {}).get("median")
        if not width:
            continue
        found.append({**source, "record": {"bin": int(min(BIN_MOST, max(2, round(width / BINNED_PX))))}})
    return found


def binned(plane: np.ndarray, factor: int, how: str = "mean") -> np.ndarray:
    """`plane` in blocks of `factor` pixels a side, cut to whole blocks:
    their mean, their root mean square over `factor` (a mean's noise, of a
    noise map), or whether any is set."""
    rows, columns = plane.shape[0] // factor, plane.shape[1] // factor
    blocks = plane[:rows * factor, :columns * factor].reshape(rows, factor, columns, factor)
    if how == "any":
        return blocks.any(axis=(1, 3))
    if how == "noise":
        return (np.sqrt(np.mean(blocks.astype(np.float64) ** 2, axis=(1, 3))) / factor).astype(np.float32)
    return blocks.mean(axis=(1, 3)).astype(np.float32)


def binned_stamp(stamp: np.ndarray, factor: int) -> np.ndarray:
    """A PSF stamp as a `factor`-binned image holds it, a star at a binned
    pixel's middle: blocks of `factor` about the stamp's middle, moved half
    a pixel first where `factor` is even, summing to one."""
    from .sharpen_model import shifted

    size = stamp.shape[0]
    middle = (size - 1) // 2
    if factor % 2 == 0:
        stamp = shifted(stamp, 0.5, 0.5, pad=4)
        first = middle - factor // 2 + 1
    else:
        first = middle - (factor - 1) // 2
    reach = min(first // factor, (size - first - factor) // factor)
    start = first - reach * factor
    side = 2 * reach + 1
    out = stamp[start:start + side * factor, start:start + side * factor].reshape(side, factor, side, factor)
    out = out.sum(axis=(1, 3)).astype(np.float64)
    return (out / out.sum()).astype(np.float32)


class BinnedField:
    """A PSF field as a `factor`-binned image holds it: positions in binned
    pixels, stamps and widths binned, point sources moved and scaled."""

    def __init__(self, field: psf_fields.Field, factor: int, image: np.ndarray) -> None:
        self.field, self.factor, self.record, self.usable = field, factor, field.record, field.usable
        self.image = image

    def full(self, value: float) -> float:
        return (value + 0.5) * self.factor - 0.5

    def stamp_at(self, x: float, y: float) -> np.ndarray:
        return binned_stamp(self.field.stamp_at(self.full(x), self.full(y)), self.factor)

    def shape_at(self, x: float, y: float) -> dict[str, float]:
        shape = dict(self.field.shape_at(self.full(x), self.full(y)))
        for name in ("fwhm", "fwhm_major", "fwhm_minor"):
            if name in shape:
                shape[name] = shape[name] / self.factor
        return shape

    def points(self) -> np.ndarray:
        columns = psf_fields.DETECTION_COLUMNS
        points = self.field.points().copy()
        points[:, 0] = (points[:, 0] + 0.5) / self.factor - 0.5
        points[:, 1] = (points[:, 1] + 0.5) / self.factor - 0.5
        points[:, columns.index("flux")] /= self.factor**2
        points[:, columns.index("fwhm")] /= self.factor
        points[:, columns.index("peak")] = psf_fields.raw_peaks(self.image, points[:, 0], points[:, 1])
        return points


class Detected:
    """A detection from a PSF record, with what `seeing_check` reads of a
    measured star."""

    __slots__ = ("x", "y", "flux", "fwhm", "eccentricity", "saturated")

    def __init__(self, row: np.ndarray) -> None:
        columns = dict(zip(psf_fields.DETECTION_COLUMNS, row.tolist()))
        self.x, self.y, self.flux, self.fwhm = columns["x"], columns["y"], columns["flux"], columns["fwhm"]
        self.eccentricity = None if math.isnan(columns["eccentricity"]) else columns["eccentricity"]
        self.saturated = bool(columns["saturated"])


def grown(mask: np.ndarray, radius: int) -> np.ndarray:
    """`mask` grown by `radius` pixels, square, cut at the edges."""
    out = mask.copy()
    for axis in (0, 1):
        spread = out.copy()
        for shift in range(1, radius + 1):
            forward = [slice(None)] * 2
            back = [slice(None)] * 2
            forward[axis], back[axis] = slice(shift, None), slice(None, -shift)
            spread[tuple(forward)] |= out[tuple(back)]
            spread[tuple(back)] |= out[tuple(forward)]
        out = spread
    return out


def saturation(image: np.ndarray, level: float | None) -> np.ndarray:
    """Where `image` reaches its clip level, grown by `SATURATION_GROW`;
    nowhere when it has none."""
    if level is None:
        return np.zeros(image.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        return grown(image >= level, SATURATION_GROW)


def fitted_stamp(stamp: np.ndarray) -> tuple[np.ndarray, float]:
    """A stamp as `STAMP` pixels a side, about its centre, and the share of
    its sum that a cut lost."""
    size = stamp.shape[0]
    if size <= STAMP:
        out = np.zeros((STAMP, STAMP), dtype=np.float32)
        pad = (STAMP - size) // 2
        out[pad:pad + size, pad:pad + size] = stamp
        return out, 0.0
    cut = (size - STAMP) // 2
    inner = stamp[cut:cut + STAMP, cut:cut + STAMP].astype(np.float64)
    total = float(stamp.sum(dtype=np.float64))
    return (inner / inner.sum()).astype(np.float32), 1.0 - float(inner.sum()) / total


def stamp_width(stamp: np.ndarray) -> float:
    """A stamp's FWHM as a round Gaussian, centred on the stamp's middle,
    fitted by least squares to its core (`WIDTH_CORE`): one measure for
    every stamp, sharp or wide, where the field's Moffat fit fails on the
    sharpest. NaN for a stamp with no positive peak."""
    peak = float(np.nanmax(stamp)) if np.isfinite(stamp).any() else float("nan")
    if not peak > 0:
        return float("nan")
    rows, columns = np.nonzero(stamp >= WIDTH_CORE * peak)
    middle = (stamp.shape[0] - 1) / 2
    squared = (columns - middle) ** 2 + (rows - middle) ** 2
    values = stamp[rows, columns].astype(np.float64)
    models = np.exp(-squared[None, :] / (2 * WIDTH_SIGMAS[:, None] ** 2))
    amplitude = (models @ values) / np.einsum("ij,ij->i", models, models)
    residual = np.sum((values[None, :] - amplitude[:, None] * models) ** 2, axis=1)
    best = int(np.clip(np.argmin(residual), 1, len(WIDTH_SIGMAS) - 2))
    # A parabola through the best three, in log σ, as the grid is.
    low, middle_value, high = residual[best - 1:best + 2]
    bend = low - 2 * middle_value + high
    step = float(np.log(WIDTH_SIGMAS[1] / WIDTH_SIGMAS[0]))
    shift = 0.5 * (low - high) / bend if bend > 0 else 0.0
    return float(2.3548 * WIDTH_SIGMAS[best] * math.exp(float(np.clip(shift, -1, 1)) * step))


def load(store: Store, product: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None]:
    """A product's image, NaN where nothing covers it, and its stack's noise
    map, `sqrt(variance / coverage)`, or None where it keeps no variance."""
    files = loads(product["files"])
    image = psf_fields.load_image(store, product)
    if "variance" not in files or "coverage" not in files:
        return image, None
    variance = np.load(store.root / files["variance"]).astype(np.float32, copy=False)
    coverage = np.load(store.root / files["coverage"])
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.where(coverage > 0, np.sqrt(variance / coverage), np.nan).astype(np.float32)
    return image, sigma


def window_corners(finite: np.ndarray, rng: random.Random, count: int) -> list[tuple[int, int]]:
    """Up to `count` window corners from the overlapping grid, at random,
    each window wholly inside `finite`."""
    height, width = finite.shape
    step = WINDOW // 2
    grid = [(y, x) for y in range(0, height - WINDOW, step) for x in range(0, width - WINDOW, step)]
    rng.shuffle(grid)
    # How many pixels of each window are not finite, from a summed table.
    table = np.zeros((height + 1, width + 1), dtype=np.int64)
    table[1:, 1:] = np.cumsum(np.cumsum(~finite, axis=0), axis=1)
    chosen = []
    for y, x in grid:
        if table[y + WINDOW, x + WINDOW] - table[y, x + WINDOW] - table[y + WINDOW, x] + table[y, x] == 0:
            chosen.append((y, x))
            if len(chosen) == count:
                break
    return chosen


def centre_psf(field: psf_fields.Field, y: int, x: int) -> tuple[np.ndarray, float, dict[str, float]]:
    """A window's PSF stamp at its centre, as `STAMP` pixels a side, the
    share a cut lost, and its shape there."""
    centre = (x + (WINDOW - 1) / 2, y + (WINDOW - 1) / 2)
    stamp, lost = fitted_stamp(field.stamp_at(*centre))
    return stamp, lost, field.shape_at(*centre)


def psf_summary(field: psf_fields.Field) -> dict[str, Any]:
    """What a source's record keeps of a PSF record."""
    record = field.record
    return {"status": record["status"], "stamp_size": record.get("stamp_size"), "fwhm_px": record.get("fwhm_px"),
            "warnings": record.get("warnings"), "clip_level": record.get("clip_level"),
            "detections": record.get("detections")}


def alignment(input_points: np.ndarray, target_points: np.ndarray) -> dict[str, Any]:
    """The median offset of the target's stars from the input's, of those
    found in both within twice `ALIGNMENT_PX`, so that a move past it is
    measured, and how many. With none found in both, the offset is None:
    a pair cannot be judged, and is kept."""
    from .multi_night import within

    targets = [Detected(row) for row in target_points]
    near = within(targets, 2 * ALIGNMENT_PX)
    offsets = []
    for row in input_points:
        found = near(float(row[0]), float(row[1]))
        if found:
            other = min(found, key=lambda star: (star.x - row[0]) ** 2 + (star.y - row[1]) ** 2)
            offsets.append((other.x - float(row[0]), other.y - float(row[1])))
    if not offsets:
        return {"matched": 0, "offset_px": None}
    return {"matched": len(offsets), "offset_px": [statistics.median(dx for dx, _ in offsets),
                                                   statistics.median(dy for _, dy in offsets)]}


def prepared(store_root: str, scratch: str | None, seed: int, per_source: int,
             source: dict[str, Any]) -> dict[str, Any]:
    """One source's windows, as `Bank.add` takes them, or why it was left
    out: read, judged, matched and cut in a process of its own."""
    store = Store(store_root, scratch)
    pairs = "target" in source
    record: dict[str, Any] = {"input": source["input"]["id"], "input_kind": source["input"]["kind"],
                              "set": source["input"]["set_id"], "split": source["input"].get("split"),
                              "telescope": source["input"].get("telescope"),
                              "pixel_scale_arcsec": source["input"].get("pixel_scale_arcsec"),
                              **({"target": source["target"]["id"], "target_kind": source["target"]["kind"]}
                                 if pairs else {}),
                              "record": source.get("record", {})}

    def left(reason: str, **more: Any) -> dict[str, Any]:
        return {"status": "dropped", "record": {**record, "reason": reason, **more}}

    input_field = psf_fields.Field(store, source["input"]["id"])
    if not input_field.usable:
        return left("psf-refused")
    image, sigma = load(store, source["input"])
    if sigma is None:
        return left("no-variance")
    level = input_field.record.get("clip_level")
    saturated = saturation(image, level)
    factor = int(source.get("record", {}).get("bin", 1))
    if factor > 1 and not pairs:
        image, sigma, saturated = binned(image, factor), binned(sigma, factor, "noise"), binned(saturated, factor,
                                                                                               "any")
        input_field = BinnedField(input_field, factor, image)
    record["input_psf"] = psf_summary(input_field)
    planes: dict[str, np.ndarray] = {}
    if pairs:
        target_field = psf_fields.Field(store, source["target"]["id"])
        if not target_field.usable:
            return left("psf-refused")
        target, target_sigma = load(store, source["target"])
        if target.shape != image.shape:
            return left("grid-mismatch", shapes=[list(image.shape), list(target.shape)])
        target[~np.isfinite(image)] = np.nan
        record["target_psf"] = psf_summary(target_field)
        record["alignment"] = alignment(input_field.detected(), target_field.detected())
        offset = record["alignment"]["offset_px"]
        if offset is not None and math.hypot(*offset) > ALIGNMENT_PX:
            return left("misaligned")
        gain = source.get("gain")
        if source.get("judge"):
            from .multi_night import seeing_check

            status, reason, check = seeing_check(
                (target, np.isfinite(target).astype(np.uint8), [Detected(row) for row in target_field.detected()]),
                (image, np.isfinite(image).astype(np.uint8), [Detected(row) for row in input_field.detected()]))
            record["record"] = {**record["record"], "check": check}
            if status != "built":
                return left(reason, check=check)
            gain = check.get("flux_gain")
        saturated |= saturation(target, target_field.record.get("clip_level"))
        if source.get("match"):
            from .multi_night import block_skies, plane_image

            gain = 1.0 if gain is None else float(gain)
            covered = np.isfinite(image) & np.isfinite(target)
            _, _, _, skies = block_skies(target, image, covered)
            if skies is None:
                return left("background-unmatched")
            plane_target, plane_input = (plane_image(coefficients, image.shape) for coefficients in skies)
            target = (target - plane_target) / gain + plane_input
            if target_sigma is not None:
                target_sigma = target_sigma / gain
            change = skies[0] - skies[1]
            record["match"] = {"gain": gain, "background_offset": float(change[0] + 0.5 * change[1] + 0.5 * change[2]),
                               "planes": [list(map(float, coefficients)) for coefficients in skies]}
        input_map = image_sigma(image)
        record["target_sigma"] = target_sigma is not None
        finite = np.isfinite(image) & np.isfinite(target) & np.isfinite(sigma) & np.isfinite(input_map)
        if target_sigma is not None:
            finite &= np.isfinite(target_sigma)
        planes = {"input": image, "target": target, "input_sigma": sigma, "input_map": input_map,
                  "target_sigma": target_sigma if target_sigma is not None else np.full_like(image, np.nan),
                  "saturated": saturated}
        background, noise = scales(image, input_map)
    else:
        finite = np.isfinite(image) & np.isfinite(sigma)
        planes = {"image": image, "sigma": sigma, "saturated": saturated}
        background, noise = scales(image, sigma)
    rng = random.Random(f"{seed}/{source['input']['id']}")
    corners = window_corners(finite, rng, per_source)
    if not corners:
        return left("no-window")
    names, stamp_names, columns = layout(pairs)
    cut = {name: np.stack([plane[y:y + WINDOW, x:x + WINDOW] for y, x in corners]).astype(names[name])
           for name, plane in planes.items()}
    stamps: dict[str, list[np.ndarray]] = {name: [] for name in stamp_names}
    info = np.zeros((len(corners), len(columns)), dtype=np.float32)
    lost = 0.0
    for row, (y, x) in enumerate(corners):
        values = {"y": y, "x": x, "background": background, "noise": noise,
                  "saturated": float(cut["saturated"][row].mean())}
        fields = (("input", input_field), ("target", target_field)) if pairs else (("", input_field),)
        for prefix, field in fields:
            stamp, share, shape = centre_psf(field, y, x)
            stamps[f"{prefix}_stamp" if prefix else "stamp"].append(stamp)
            lost = max(lost, share)
            for name in ("fwhm", "e1", "e2", "beta"):
                values[f"{prefix}_{name}" if prefix else name] = shape[name]
            values[f"{prefix}_width" if prefix else "width"] = stamp_width(stamp)
        if pairs:
            values["ratio"] = values["target_fwhm"] / values["input_fwhm"]
            values["width_ratio"] = values["target_width"] / values["input_width"]
        info[row] = [values.get(column, 0.0) for column in columns]
    record["stamp_lost"] = lost
    record["background"], record["noise"] = background, noise
    return {"status": "cut", "record": record, "planes": cut,
            "stamps": {name: np.stack(stack) for name, stack in stamps.items()}, "info": info,
            "points": input_field.points()}


class Bank:
    """A sharpening bank being cut, one source at a time: at most
    `capacity` windows. `done` says it was finished before. Given a `state`
    it recorded, a cut stopped part way goes on from there; `resumed` says
    it did."""

    def __init__(self, directory: Path, *, pairs: bool, capacity: int, resume: dict[str, Any] | None = None) -> None:
        self.directory = directory
        self.pairs = pairs
        self.done = (directory / "done.json").exists()
        self.count = 0
        self.sources: list[dict[str, Any]] = []
        self.resumed = False
        if self.done:
            return
        (directory / "points").mkdir(parents=True, exist_ok=True)
        planes, stamps, info = layout(pairs)
        shapes = {**{name: ((capacity, WINDOW, WINDOW), dtype) for name, dtype in planes.items()},
                  **{name: ((capacity, STAMP, STAMP), np.float32) for name in stamps},
                  "info": ((capacity, len(info)), np.float32)}
        self.resumed = resume is not None and all((directory / f"{name}.npy").exists() for name in shapes)
        self.arrays = {name: np.lib.format.open_memmap(directory / f"{name}.npy", mode="r+" if self.resumed else "w+",
                                                       dtype=dtype, shape=shape)
                       for name, (shape, dtype) in shapes.items()}
        if self.resumed:
            self.count = resume["count"]
            self.sources = list(resume["sources"])

    def add(self, cut: dict[str, Any]) -> None:
        """Write one source's windows, as `prepared` cuts them."""
        index, count = len(self.sources), len(cut["info"])
        span = slice(self.count, self.count + count)
        for name, values in (*cut["planes"].items(), *cut["stamps"].items()):
            self.arrays[name][span] = values
        info = cut["info"].copy()
        info[:, 0] = index
        self.arrays["info"][span] = info
        np.save(self.directory / "points" / f"{index}.npy", cut["points"])
        self.sources.append({**cut["record"], "windows": [self.count, self.count + count]})
        self.count += count

    def state(self) -> dict[str, Any]:
        """How far the cut has come, with the windows so far on disk first."""
        for array in self.arrays.values():
            array.flush()
        return {"count": self.count, "sources": self.sources}

    def finish(self, summary: dict[str, Any]) -> dict[str, Any]:
        """The bank, as `open_bank` reads it, marked done with `summary`."""
        if not self.done:
            for array in self.arrays.values():
                array.flush()
            del self.arrays
            planes, stamps, info = layout(self.pairs)
            (self.directory / "sources.json").write_text(json.dumps(self.sources, indent=1))
            (self.directory / "done.json").write_text(json.dumps({
                "bank_version": BANK_VERSION, "windows": self.count, "sources": len(self.sources),
                "pairs": self.pairs, "planes": list(planes), "stamps": list(stamps), "info": list(info),
                "window": WINDOW, "stamp": STAMP, **summary}, indent=1))
            self.done = True
        return open_bank(self.directory)


def open_bank(directory: Path) -> dict[str, Any]:
    """A finished bank: its planes, stamps and `info`, memory-mapped and cut
    to its windows; its sources' records under "sources", each with the
    span of its windows; and its record under "done". A source's points
    are `points/<index>.npy`, in `psf_fields.DETECTION_COLUMNS`."""
    done = json.loads((directory / "done.json").read_text())
    count = done["windows"]
    banked: dict[str, Any] = {name: np.load(directory / f"{name}.npy", mmap_mode="r")[:count]
                              for name in (*done["planes"], *done["stamps"], "info")}
    banked["sources"] = json.loads((directory / "sources.json").read_text())
    banked["done"] = done
    banked["directory"] = directory
    return banked


def cut(
    store: Store, sources: list[dict[str, Any]], *, label: str, split: str, work: Path, seed: int = 0,
    per_source: int = DEEP_WINDOWS, workers: int = 1,
) -> dict[str, Any]:
    """Cut a bank of `sources` under `work`, named by `label`, `split` and a
    hash of what went into it. A source is a deep stack (`{"input": row}`)
    or a pair (`{"input": row, "target": row}`), all of one kind; both may
    carry a `record`, kept with the source's windows. A pair asks with
    `match` for its target to be put on its input's flux, by `gain`, and
    sky, by the planes `seeing_check` fits, and with `judge` for
    `seeing_check` to judge it first, its gain then the check's. A finished
    bank is opened, not cut again."""
    if not sources:
        raise ValueError(f"no sources for the {label} bank of the {split} split")
    pairs = "target" in sources[0]
    if any(("target" in source) != pairs for source in sources):
        raise ValueError("a bank holds deep stacks or pairs, not both")
    parts = [BANK_VERSION, label, split, seed, WINDOW, STAMP, per_source, SATURATION_GROW, ALIGNMENT_PX] + [
        [source["input"]["id"], source.get("target", {}).get("id"), source.get("match", False), source.get("gain"),
         source.get("judge", False), source.get("record", {})] for source in sources]
    name = f"sharpen-{label}-{split}-" + hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]
    directory = work / name
    checkpoint = directory / "progress.json"
    progress = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
    capacity = len(sources) * per_source
    bank = Bank(directory, pairs=pairs, capacity=capacity, resume=progress and progress["bank"])
    if progress and not bank.done and not bank.resumed:
        # A cut whose files are gone starts again.
        progress = None
        bank = Bank(directory, pairs=pairs, capacity=capacity)
    if not bank.done:
        start = progress["sources"] if progress else 0
        dropped = list(progress["dropped"]) if progress else []
        prepare = functools.partial(prepared, str(store.root), str(store.scratch), seed, per_source)
        rest = sources[start:]
        recorded = began = time.monotonic()
        for done, (source, result) in enumerate(zip(rest, in_order(prepare, rest, workers)), start=start + 1):
            if result["status"] == "dropped":
                dropped.append(result["record"])
            else:
                bank.add(result)
            print(json.dumps({"bank": name, "done": done, "of": len(sources), "input": source["input"]["id"],
                              "status": result["status"], "reason": result["record"].get("reason"),
                              "windows": bank.count, "elapsed": round(time.monotonic() - began)}),
                  file=sys.stderr, flush=True)
            if time.monotonic() - recorded >= CHECKPOINT_SECONDS:
                state = {"sources": done, "dropped": dropped, "bank": bank.state()}
                # Written whole and then moved, so a stop part way through
                # leaves the last record as it was.
                checkpoint.with_suffix(".tmp").write_text(json.dumps(state))
                checkpoint.with_suffix(".tmp").replace(checkpoint)
                recorded = time.monotonic()
        bank.finish({"label": label, "split": split, "seed": seed, "per_source": per_source, "dropped": dropped,
                     "summary": summarize(bank, pairs)})
    checkpoint.unlink(missing_ok=True)
    return open_bank(directory)


def summarize(bank: Bank, pairs: bool) -> dict[str, Any]:
    """A bank's windows by telescope, and the spread of their PSF widths
    (and, of pairs, ratios)."""
    info = np.asarray(bank.arrays["info"][:bank.count]) if hasattr(bank, "arrays") else np.zeros((0, 1))
    columns = layout(pairs)[2]
    found: dict[str, Any] = {"windows": bank.count, "sources": len(bank.sources), "telescopes": {}}
    for source in bank.sources:
        start, end = source["windows"]
        found["telescopes"][source["telescope"]] = found["telescopes"].get(source["telescope"], 0) + end - start

    def spread(column: str) -> list[float] | None:
        values = info[:, columns.index(column)] if len(info) else np.zeros(0)
        values = values[np.isfinite(values)]
        return [round(float(np.percentile(values, q)), 3) for q in (0, 5, 50, 95, 100)] if values.size else None

    for column in (("input_fwhm", "target_fwhm", "ratio", "input_width", "target_width", "width_ratio") if pairs
                   else ("fwhm", "width")) + ("saturated",):
        found[column] = spread(column)
    return found
