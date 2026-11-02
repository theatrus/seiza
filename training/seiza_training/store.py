"""Where products go, and how they are written.

Each plane is a `.npy` file, so a training loader memory-maps it and reads
patches at random without loading whole images. Coverage and rejection
counts are stored as 16-bit integers. Stacks that are solved or reviewed
are also written as FITS by Seiza.
"""

from __future__ import annotations

import os
import shutil
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

import seiza

# Registered frames are written once and read for every product, so they
# belong on local disk even when the store is on a network share.
SCRATCH = Path.home() / ".cache" / "seiza-training" / "scratch"


class Store:
    """The corpus store: the manifest, products and review pages, and the
    scratch directory where a build keeps its registered frames."""

    def __init__(self, root: str | Path, scratch: str | Path | None = None) -> None:
        self.root = Path(root)
        self.products = self.root / "products"
        self.review = self.root / "review"
        self.scratch = Path(scratch) if scratch is not None else SCRATCH
        for directory in (self.products, self.review):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.sqlite"

    def set_directory(self, set_id: str) -> Path:
        return self.products / set_id

    def relative(self, path: Path) -> str:
        return str(path.relative_to(self.root))

    def check_space(self, scratch_bytes: int, product_bytes: int) -> None:
        """Refuse a build whose frames or products would not fit, counting
        both against one disk when the scratch and the store share it."""
        self.scratch.mkdir(parents=True, exist_ok=True)
        if os.stat(self.scratch).st_dev == os.stat(self.root).st_dev:
            needs = [(self.root, scratch_bytes + product_bytes)]
        else:
            needs = [(self.scratch, scratch_bytes), (self.root, product_bytes)]
        for directory, need in needs:
            free = shutil.disk_usage(directory).free
            if free < need:
                raise RuntimeError(
                    f"needs about {need / 1e9:.1f} GB in {directory} and {free / 1e9:.1f} GB is free"
                )


def plane(array: np.ndarray) -> np.ndarray:
    """A snapshot plane as (height, width) or (channels, height, width)."""
    if array.ndim == 3:
        return np.ascontiguousarray(np.moveaxis(array, -1, 0))
    return np.ascontiguousarray(array)


def counts16(array: np.ndarray) -> np.ndarray:
    return plane(np.minimum(array, np.iinfo(np.uint16).max).astype(np.uint16))


def robust_sigma(image: np.ndarray) -> float | None:
    """The noise of `image` from neighbouring differences, which star and
    nebula structure barely move: 1.4826 MAD of the differences over √2."""
    data = image if image.ndim == 2 else image[..., 0]
    difference = (data[:, 1:] - data[:, :-1])[::3, ::3]
    difference = difference[np.isfinite(difference)]
    if difference.size == 0:
        return None
    median = np.median(difference)
    return float(1.4826 * np.median(np.abs(difference - median)) / np.sqrt(2.0))


def stack_stats(snapshot: Any) -> dict[str, Any]:
    image = snapshot.image
    finite = image[np.isfinite(image)]
    coverage = snapshot.coverage
    return {
        "accepted_frames": snapshot.accepted_frames,
        "shape": list(image.shape),
        "coverage_max": int(coverage.max()),
        "coverage_median": float(np.median(coverage)),
        "median": float(np.median(finite)) if finite.size else None,
        "noise": robust_sigma(image),
        "rejected_fraction": float(snapshot.rejected_samples.sum())
        / max(1.0, float(coverage.sum() + snapshot.rejected_samples.sum())),
    }


def save_stack(directory: Path, snapshot: Any, *, fits: bool) -> dict[str, str]:
    """Write a stack snapshot's planes into `directory`; also its FITS,
    with the reference's WCS, when `fits`."""
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, array in (
        ("image", plane(snapshot.image.astype(np.float32, copy=False))),
        ("variance", plane(snapshot.variance.astype(np.float32, copy=False))),
        ("coverage", counts16(snapshot.coverage)),
        ("rejected", counts16(snapshot.rejected_samples)),
    ):
        path = directory / f"{name}.npy"
        np.save(path, array)
        files[name] = path.name
    if fits:
        path = directory / "stack.fits"
        path.unlink(missing_ok=True)
        snapshot.write_fits(path)
        files["fits"] = path.name
    return files


WINDOW = 1024


def read_image(path: str | Path, width: int, height: int) -> np.ndarray:
    """A whole FITS or XISF image, as (height, width) or (planes, height,
    width) float32, read through Seiza a window at a time: a window's side
    is at most 1024 pixels."""
    image: np.ndarray | None = None
    for top in range(0, height, WINDOW):
        for left in range(0, width, WINDOW):
            window = seiza.read_image_window(path, left, top, WINDOW, WINDOW)
            values = window.values
            if image is None:
                image = np.empty((values.shape[0], height, width), dtype=np.float32)
            image[:, window.y:window.y + window.height, window.x:window.x + window.width] = values
    assert image is not None
    return image[0] if image.shape[0] == 1 else image


def versions() -> dict[str, str]:
    return {
        "seiza": seiza.__version__,
        "stacking_output": seiza.STACKING_OUTPUT_VERSION,
        "stretch_output": seiza.STRETCH_OUTPUT_VERSION,
    }


def write_png(path: Path, pixels: np.ndarray) -> None:
    """Write 8-bit grey (h, w) or RGB (h, w, 3) pixels as a PNG."""
    pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
    height, width = pixels.shape[:2]
    colour = 2 if pixels.ndim == 3 else 0
    rows = pixels.reshape(height, -1)
    raw = b"".join(b"\x00" + row.tobytes() for row in rows)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, colour, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
