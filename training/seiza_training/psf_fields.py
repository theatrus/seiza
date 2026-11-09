"""The PSF pass: each source product's PSF field, measured once and kept
beside its stack, for the sharpening bank and its trainer.

A product's field is measured by `seiza.PsfField` on its image, NaN where
nothing covers it, with no catalogue (the Gaia pass comes later). A field
that has too few stars to vary is measured again as one global PSF, and
recorded as such; one with too few even for that is recorded as refused.
`PsfField` cannot be saved and opened again, so what the bank needs of it
is sampled on a grid of nodes at most `GRID_PX` apart, as Seiza's remaining
kernel samples it: the stamp at each node (each summing to one, so a
weighted mean of four is a stamp too) and its shape (FWHM, ellipticity
components, Moffat β). Its polynomial is of degree three at most across the
whole field, so between nodes this far apart it is all but linear.

Beside the field, each record keeps:

- the stars the field was fitted on (core and wing stars);
- every star the measuring detector finds on the stack, unsmoothed, as the
  multi-night pass measures a seeing pair's halves (`stack_psf`), the
  stars `seeing_check` judges a pair by, and the clipped cores the
  detector turns away: the point-source list a network may read, made
  from the image alone, until the Gaia pass gives one from the catalogue;
- the level at which the stack's brightest stars clip, found as
  `PsfField` finds it (`clip_level`), from the image's own peaks: the
  bank's saturation mask;
- the median FWHM the product's frames were measured at, smoothed, for
  comparison.

Records go under `STORE/psf/<product id>/`: `field.npz` with the arrays and
`field.json`, written last, which marks a product done. The manifest is
only read. A pass stopped part way goes on with the products not yet done;
`workers` products are measured at once, in processes of their own.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import Manifest, loads
from .store import Store, versions

# Raised with the records' layout, so records made before are made again.
PSF_PASS_VERSION = 1

# The most the field's nodes are apart, in pixels, as for Seiza's
# `RemainingKernel`.
GRID_PX = 256

# The products the sharpening bank cuts from: deep stacks (a night's, and
# many nights'), for pairs blurred by a known kernel at training time, and
# both halves of every seeing pair, a night's and many nights'.
SOURCE_KINDS = ("deep", "multi-night", "seeing-best", "seeing-worst", "multi-night-seeing-best",
                "multi-night-seeing-worst")
# The splits a model trains and is chosen on; test and held-out stay out.
SPLITS = ("train", "validation")

# `PsfField.measure`'s options, as the pass uses them: its defaults, with no
# catalogue. They are recorded with every field.
MEASURE_OPTIONS = {"components": 3, "minimum_stars": 30, "maximum_stars": 400, "minimum_snr": 30.0,
                   "detection_sigma": 5.0, "isolation_magnitudes": 5.0}

# The columns of a record's detections, as `points` returns them: position,
# flux and 3 × 3 peak in the image's units (the detector reads 16-bit
# counts), FWHM in pixels, signal to noise, eccentricity (NaN where not
# fitted), whether the detector called the star saturated, whether its
# peak reaches the stack's clip level, and whether the detector found it.
# The detector turns away the stars whose cores clip, the brightest, so
# their cores are added from the image (`clipped_cores`), at their
# brightest pixel, with their peak alone.
DETECTION_COLUMNS = ("x", "y", "flux", "peak", "fwhm", "snr", "eccentricity", "saturated", "clipped", "detected")
# The columns of the field's own stars: role is 0 for a core star, 1 for a
# saturated star that gave the wings.
STAR_COLUMNS = ("x", "y", "flux", "snr", "fwhm", "role")
# The shape columns sampled at each node.
SHAPE_COLUMNS = ("fwhm", "fwhm_major", "fwhm_minor", "e1", "e2", "beta", "angle")

# `clip_level` reads the peaks of this many of the brightest objects, as
# `PsfField` reads its 300 brightest stars'.
CLIP_PEAKS = 300
# An object's peak is the largest value of a block of this side that is
# the largest of the 3 × 3 blocks about it: each star once, however bright,
# where the detector, which turns clipped stars away, would miss the
# brightest.
PEAK_BLOCK = 16
# A clipped core within this many pixels of a brighter one, or of a star
# the detector found, is the same star.
CORE_RADIUS = 24.0


def eligible(
    manifest: Manifest, *, kinds: Iterable[str] = SOURCE_KINDS, splits: Iterable[str] = SPLITS,
    sets: set[str] | None = None,
) -> list[dict[str, Any]]:
    """The built products of `kinds` in sets calibrated with a flat
    (`calibration_status` 'ready', never 'ready:no-flat': a flat-less set's
    vignetting and dust change its stars across the field) of `splits`,
    with their set's split, in id order."""
    kinds, splits = tuple(kinds), tuple(splits)
    rows = manifest.rows(
        f"{PRODUCTS} WHERE p.status = 'built' AND s.calibration_status = 'ready' "
        f"AND p.kind IN ({', '.join('?' * len(kinds))}) AND s.split IN ({', '.join('?' * len(splits))}) "
        f"ORDER BY p.id",
        (*kinds, *splits),
    )
    return [dict(row) for row in rows if sets is None or row["set_id"] in sets]


# Products with what the pass and the bank record of their sets.
PRODUCTS = ("SELECT p.*, s.split, s.calibration_status, s.telescope, s.pixel_scale_arcsec "
            "FROM products p JOIN sets s ON s.id = p.set_id")


def annotated(manifest: Manifest, product_ids: Iterable[str]) -> list[dict[str, Any]]:
    """The products of `product_ids`, in their order, with their sets'
    split, calibration, telescope and scale, as `eligible` returns them; an
    id with no product, or no set, is left out. For products a caller
    chooses, such as the parts of the pairs it supplies a bank."""
    found = []
    for product_id in product_ids:
        rows = manifest.rows(f"{PRODUCTS} WHERE p.id = ?", (product_id,))
        if rows:
            found.append(dict(rows[0]))
    return found


def directory(store: Store, product_id: str) -> Path:
    """Where a product's PSF record lives."""
    return store.root / "psf" / product_id


def recorded(store: Store, product_id: str) -> dict[str, Any] | None:
    """A product's PSF record, or None where the pass has not made one (or
    made it under another layout)."""
    path = directory(store, product_id) / "field.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    return record if record.get("pass_version") == PSF_PASS_VERSION else None


def frame_fwhm(manifest: Manifest, product: dict[str, Any]) -> float | None:
    """The median FWHM the product's frames were measured at, or None where
    none of them was."""
    frames = loads(product.get("frames"), [])
    found = []
    for start in range(0, len(frames), 500):
        chunk = frames[start:start + 500]
        found += [row["fwhm_px"] for row in manifest.rows(
            f"SELECT fwhm_px FROM measurements WHERE path IN ({', '.join('?' * len(chunk))}) AND fwhm_px IS NOT NULL",
            chunk)]
    return statistics.median(found) if found else None


def load_image(store: Store, product: dict[str, Any]) -> np.ndarray:
    """A product's image as float32, NaN where its coverage is 0; a product
    with no coverage plane (an image made from another) is NaN only where
    it is not finite already."""
    files = loads(product["files"])
    image = np.load(store.root / files["image"]).astype(np.float32, copy=False)
    if image.ndim == 3:
        image = image[0]
    if "coverage" in files:
        coverage = np.load(store.root / files["coverage"])
        image[(coverage if coverage.ndim == 2 else coverage[0]) == 0] = np.nan
    return image


def clip_level(peaks: np.ndarray) -> float | None:
    """The level at which a stack's brightest stars clip, as `PsfField`
    finds it: three or more of the 3 × 3 peaks within 2% of one another, at
    90% of the brightest or above, and 0.8 times their median, from which
    stars count as saturated. A stack's clipped cores share one level
    closely, as unclipped stars do not. None where no such cluster is."""
    values = np.sort(np.asarray(peaks, dtype=np.float64)[np.isfinite(peaks)])[::-1]
    if values.size == 0 or values[0] <= 0:
        return None
    for start, top in enumerate(values):
        if top < 0.9 * values[0]:
            break
        # The values run down, so those within 2% of this one follow it.
        clustered = int(np.sum(values[start:] >= 0.98 * top))
        if clustered >= 3:
            return float(0.8 * np.median(values[start:start + clustered]))
    return None


def raw_peaks(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """The largest finite value in the 3 × 3 pixels about each position."""
    height, width = image.shape
    columns = np.clip(np.rint(xs).astype(int), 0, width - 1)
    rows = np.clip(np.rint(ys).astype(int), 0, height - 1)
    peaks = np.full(len(columns), -np.inf)
    for down in (-1, 0, 1):
        for across in (-1, 0, 1):
            values = image[np.clip(rows + down, 0, height - 1), np.clip(columns + across, 0, width - 1)]
            peaks = np.fmax(peaks, np.where(np.isfinite(values), values, -np.inf))
    return peaks


def block_peaks(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The image's objects' peaks, brightest first, and where they are
    (`PEAK_BLOCK`): a clipped core spread over several blocks gives each
    of them, all at its clip level."""
    height, width = image.shape
    rows, columns = height // PEAK_BLOCK, width // PEAK_BLOCK
    blocks = image[:rows * PEAK_BLOCK, :columns * PEAK_BLOCK].reshape(rows, PEAK_BLOCK, columns, PEAK_BLOCK)
    filled = np.where(np.isfinite(blocks), blocks, -np.inf).transpose(0, 2, 1, 3).reshape(rows, columns, -1)
    where = filled.argmax(axis=2)
    maxima = np.take_along_axis(filled, where[..., None], axis=2)[..., 0]
    padded = np.pad(maxima, 1, constant_values=-np.inf)
    around = np.max([padded[1 + down:1 + down + rows, 1 + across:1 + across + columns]
                     for down in (-1, 0, 1) for across in (-1, 0, 1)], axis=0)
    kept = np.isfinite(maxima) & (maxima >= around)
    block_rows, block_columns = np.nonzero(kept)
    order = np.argsort(-maxima[kept], kind="stable")
    offsets = where[kept][order]
    return (maxima[kept][order].astype(np.float64),
            (block_columns[order] * PEAK_BLOCK + offsets % PEAK_BLOCK).astype(np.float64),
            (block_rows[order] * PEAK_BLOCK + offsets // PEAK_BLOCK).astype(np.float64))


def detections(image: np.ndarray) -> tuple[np.ndarray, float | None, str | None]:
    """Every star the measuring detector finds on a stack, unsmoothed, as
    `stack_psf` measures one, and the clipped cores it turns away, in
    `DETECTION_COLUMNS`; the clip level, from the image's peaks; and the
    detector's error where it failed."""
    import seiza

    from .multi_night import STACK_DETECTION, scaled_counts

    peaks, peak_x, peak_y = block_peaks(image)
    level = clip_level(peaks[:CLIP_PEAKS])
    counts, low, high = scaled_counts(image)
    error = None
    try:
        found = seiza.detect_measured_stars(counts, **STACK_DETECTION).stars
    except Exception as failure:  # noqa: BLE001 - recorded, not fatal
        found, error = [], str(failure)
    # Counts above the 0.01st percentile, in the image's units.
    scale = (high - low) / 65535
    table = np.array([[star.x, star.y, star.flux * scale, 0.0, star.fwhm, star.snr,
                       np.nan if star.eccentricity is None else star.eccentricity, float(star.saturated), 0.0, 1.0]
                      for star in found], dtype=np.float64).reshape(-1, len(DETECTION_COLUMNS))
    if len(table):
        table[:, 3] = raw_peaks(image, table[:, 0], table[:, 1])
    if level is not None:
        table[:, 8] = table[:, 3] >= level
        cores: list[list[float]] = []
        for value, x, y in zip(peaks, peak_x, peak_y):
            if value < level:
                break
            known = np.concatenate([table[:, :2], np.array(cores).reshape(-1, len(DETECTION_COLUMNS))[:, :2]])
            if not len(known) or np.min((known[:, 0] - x) ** 2 + (known[:, 1] - y) ** 2) > CORE_RADIUS**2:
                cores.append([x, y, np.nan, value, np.nan, np.nan, np.nan, 1.0, 1.0, 0.0])
        if cores:
            table = np.concatenate([table, np.array(cores)])
    return table.astype(np.float32), level, error


def grid_nodes(length: int) -> np.ndarray:
    """Node positions along one side of `length` pixels: both edges, at
    most `GRID_PX` apart."""
    return np.linspace(0.0, length - 1.0, max(2, math.ceil((length - 1) / GRID_PX) + 1))


def measure_field(image: np.ndarray) -> tuple[Any | None, str, str | None]:
    """The image's PSF field and how it was had: "measured", "global" when
    too few stars let it vary, or "refused" (no field) with the error."""
    import seiza

    try:
        return seiza.PsfField.measure(image, **MEASURE_OPTIONS), "measured", None
    except seiza.TooFewStarsError as error:
        refusal = str(error)
    try:
        return seiza.PsfField.measure(image, **MEASURE_OPTIONS, sparse="global"), "global", refusal
    except seiza.TooFewStarsError as error:
        return None, "refused", str(error)


def measure(store: Store, product: dict[str, Any]) -> dict[str, Any]:
    """Measure one product's PSF field and keep its record (see the module
    notes); returns the record."""
    started = time.monotonic()
    image = load_image(store, product)
    height, width = image.shape
    loaded = time.monotonic()
    table, level, detection_error = detections(image)
    detected = time.monotonic()
    field, status, error = measure_field(image)
    fitted = time.monotonic()
    record: dict[str, Any] = {
        "pass_version": PSF_PASS_VERSION,
        "product": product["id"],
        "kind": product["kind"],
        "set": product["set_id"],
        "split": product.get("split"),
        "telescope": product.get("telescope"),
        "pixel_scale_arcsec": product.get("pixel_scale_arcsec"),
        "frame_fwhm_px": product.get("frame_fwhm_px"),
        "shape": [height, width],
        "status": status,
        "error": error,
        "options": MEASURE_OPTIONS,
        "grid_px": GRID_PX,
        "detections": len(table),
        "detection_error": detection_error,
        "clip_level": level,
        "versions": versions(),
    }
    arrays: dict[str, np.ndarray] = {"detections": table}
    if field is not None:
        xs, ys = grid_nodes(width), grid_nodes(height)
        stamps = np.stack([np.stack([field.stamp_at(float(x), float(y)) for x in xs]) for y in ys]).astype(np.float32)
        shapes = np.full((len(ys), len(xs), len(SHAPE_COLUMNS)), np.nan, dtype=np.float32)
        for row, y in enumerate(ys):
            for column, x in enumerate(xs):
                shape = field.shape_at(float(x), float(y))
                if shape is not None:
                    shapes[row, column] = [getattr(shape, name) for name in SHAPE_COLUMNS]
        stars = np.array([[star.x, star.y, star.flux, star.snr, star.fwhm, float(star.role == "wing")]
                          for star in field.stars], dtype=np.float32).reshape(-1, len(STAR_COLUMNS))
        arrays.update(xs=xs.astype(np.float32), ys=ys.astype(np.float32), stamps=stamps, shapes=shapes, stars=stars)
        fwhm = shapes[..., 0][np.isfinite(shapes[..., 0])]
        ellipticity = np.hypot(shapes[..., 3], shapes[..., 4])
        centre = field.shape_at((width - 1) / 2, (height - 1) / 2)
        sums = stamps.sum(axis=(2, 3))
        record.update({
            "stamp_size": field.stamp_size,
            "component_count": field.component_count,
            "star_count": field.star_count,
            "core_stars": int(np.sum(stars[:, 5] == 0)),
            "wing_stars": int(np.sum(stars[:, 5] == 1)),
            "is_global": field.is_global,
            "warnings": list(field.warnings),
            "channel_fits": [{"reduced_chi_squared": fit.reduced_chi_squared,
                              "relative_residual": fit.relative_residual, "degree": fit.degree,
                              "wing_stars": fit.wing_stars,
                              "wing_overlap": None if fit.wing_overlap is None else list(fit.wing_overlap)}
                             for fit in field.channel_fits],
            "nodes": [len(ys), len(xs)],
            "fwhm_px": {"centre": None if centre is None else centre.fwhm,
                        "min": float(fwhm.min()) if fwhm.size else None,
                        "median": float(np.median(fwhm)) if fwhm.size else None,
                        "max": float(fwhm.max()) if fwhm.size else None},
            "ellipticity_max": float(np.nanmax(ellipticity)) if np.isfinite(ellipticity).any() else None,
            "unshaped_nodes": int(np.sum(~np.isfinite(shapes[..., 0]))),
            "stamp_sums": [float(sums.min()), float(sums.max())],
        })
    record["seconds"] = {"load": round(loaded - started, 1), "detect": round(detected - loaded, 1),
                         "field": round(fitted - detected, 1), "total": round(time.monotonic() - started, 1)}
    target = directory(store, product["id"])
    target.mkdir(parents=True, exist_ok=True)
    # Written whole and then moved, the JSON last: a record is there only
    # once all of it is.
    with (target / "field.npz.tmp").open("wb") as file:
        np.savez(file, **arrays)
    (target / "field.npz.tmp").replace(target / "field.npz")
    (target / "field.json.tmp").write_text(json.dumps(record, indent=1))
    (target / "field.json.tmp").replace(target / "field.json")
    return record


def measured(store_root: str, scratch: str | None, product: dict[str, Any]) -> dict[str, Any]:
    """`measure` in a process of its own, which opens the store by path;
    an error is returned as a record, not raised, so one bad product does
    not stop the pass."""
    store = Store(store_root, scratch)
    try:
        return measure(store, product)
    except Exception as error:  # noqa: BLE001 - reported in the summary
        return {"product": product["id"], "status": "failed", "error": f"{type(error).__name__}: {error}"}


def run(manifest: Manifest, store: Store, products: list[dict[str, Any]], *, workers: int = 4,
        again: bool = False, limit: int | None = None) -> dict[str, Any]:
    """Measure the PSF field of each of `products` not yet recorded (all of
    them with `again`), `workers` at once; at most `limit` of them. Returns
    each status's count, the products that failed, and the spread of the
    fields' FWHM."""
    import functools

    from .noise_model import in_order

    todo = [product for product in products if again or recorded(store, product["id"]) is None]
    skipped = len(products) - len(todo)
    todo = todo[:limit]
    for product in todo:
        product["frame_fwhm_px"] = frame_fwhm(manifest, product)
    task = functools.partial(measured, str(store.root), str(store.scratch))
    summary: dict[str, Any] = {"products": len(products), "skipped": skipped, "status": {}, "failed": []}
    started = time.monotonic()
    for done, record in enumerate(in_order(task, todo, workers), start=1):
        summary["status"][record["status"]] = summary["status"].get(record["status"], 0) + 1
        if record["status"] == "failed":
            summary["failed"].append({"product": record["product"], "error": record["error"]})
        fwhm = (record.get("fwhm_px") or {}).get("centre")
        print(json.dumps({"done": done, "of": len(todo), "product": record["product"], "status": record["status"],
                          "fwhm_centre": None if fwhm is None else round(fwhm, 2),
                          "seconds": record.get("seconds", {}).get("total"),
                          "elapsed": round(time.monotonic() - started)}), file=sys.stderr, flush=True)
    summary["fields"] = field_summary(store, products)
    return summary


def field_summary(store: Store, products: list[dict[str, Any]]) -> dict[str, Any]:
    """The recorded fields of `products` by kind: how many were measured,
    global and refused, and the range of their centres' FWHM and of the
    frames' measured FWHM."""
    found: dict[str, dict[str, Any]] = {}
    for product in products:
        record = recorded(store, product["id"])
        if record is None:
            continue
        kind = found.setdefault(f"{record['kind']}:{record.get('split')}", {"status": {}, "centre": [], "frames": []})
        kind["status"][record["status"]] = kind["status"].get(record["status"], 0) + 1
        centre = (record.get("fwhm_px") or {}).get("centre")
        if centre is not None:
            kind["centre"].append(centre)
        if record.get("frame_fwhm_px") is not None:
            kind["frames"].append(record["frame_fwhm_px"])
    return {name: {"status": kind["status"],
                   "fwhm_centre_px": [round(min(kind["centre"]), 2), round(statistics.median(kind["centre"]), 2),
                                      round(max(kind["centre"]), 2)] if kind["centre"] else None,
                   "frame_fwhm_px": [round(min(kind["frames"]), 2), round(statistics.median(kind["frames"]), 2),
                                     round(max(kind["frames"]), 2)] if kind["frames"] else None}
            for name, kind in sorted(found.items())}


class Field:
    """A product's recorded PSF field, sampled anywhere by weighing its
    four nearest nodes bilinearly."""

    def __init__(self, store: Store, product_id: str) -> None:
        self.record = recorded(store, product_id)
        if self.record is None:
            raise FileNotFoundError(f"no PSF record for {product_id}: run the psf-fields pass first")
        with np.load(directory(store, product_id) / "field.npz") as arrays:
            self.arrays = {name: arrays[name] for name in arrays.files}

    @property
    def usable(self) -> bool:
        """Whether the record holds a field, measured or global."""
        return "stamps" in self.arrays

    def weights(self, x: float, y: float) -> list[tuple[int, int, float]]:
        """The nodes about (`x`, `y`) and their bilinear weights."""
        found = []
        for axis, value in (("ys", y), ("xs", x)):
            nodes = self.arrays[axis]
            low = int(np.clip(np.searchsorted(nodes, value, side="right") - 1, 0, len(nodes) - 2))
            share = float(np.clip((value - nodes[low]) / (nodes[low + 1] - nodes[low]), 0.0, 1.0))
            found.append((low, share))
        (row, down), (column, across) = found
        return [(row, column, (1 - down) * (1 - across)), (row, column + 1, (1 - down) * across),
                (row + 1, column, down * (1 - across)), (row + 1, column + 1, down * across)]

    def stamp_at(self, x: float, y: float) -> np.ndarray:
        """The local PSF stamp, summing to one."""
        stamps = self.arrays["stamps"]
        return sum(weight * stamps[row, column] for row, column, weight in self.weights(x, y)).astype(np.float32)

    def shape_at(self, x: float, y: float) -> dict[str, float]:
        """The local shape (`SHAPE_COLUMNS`), from the nodes about the point
        whose shape was fitted; NaN where none was."""
        shapes = self.arrays["shapes"]
        nodes = [(shapes[row, column], weight) for row, column, weight in self.weights(x, y)
                 if np.isfinite(shapes[row, column, 0])]
        total = sum(weight for _, weight in nodes)
        if not nodes or total <= 0:
            return {name: float("nan") for name in SHAPE_COLUMNS}
        mean = sum(weight * shape for shape, weight in nodes) / total
        return {name: float(value) for name, value in zip(SHAPE_COLUMNS, mean)}

    def points(self) -> np.ndarray:
        """The stack's point sources, in `DETECTION_COLUMNS`: the detector's
        stars and the clipped cores it turned away."""
        return self.arrays["detections"]

    def detected(self) -> np.ndarray:
        """The detector's stars alone, as `stack_psf` finds them."""
        points = self.points()
        return points[points[:, DETECTION_COLUMNS.index("detected")] == 1]
