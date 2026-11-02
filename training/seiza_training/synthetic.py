"""A small synthetic corpus laid out as the telescope corpus is: one night of
lights whose seeing worsens as the night goes on, and bias and flat masters
that Seiza builds into a calibration library. The tests run every pass on
it, and it tries the passes without telescope data. Writing it needs
astropy, from the `test` extra.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import seiza

WIDTH, HEIGHT = 512, 384
FRAMES = 45
START = datetime(2026, 1, 10, 4, 0, 0, tzinfo=timezone.utc)
LIGHTS = Path("telescopes") / "Test Scope" / "_Source" / "2026" / "Test Nebula"


def common_header(kind: str, when: datetime, exposure: float):
    from astropy.io import fits

    header = fits.Header()
    header["IMAGETYP"] = kind
    header["EXPTIME"] = exposure
    header["EXPOSURE"] = exposure
    header["DATE-OBS"] = when.strftime("%Y-%m-%dT%H:%M:%S.000")
    header["GAIN"] = 100
    header["OFFSET"] = 30
    header["INSTRUME"] = "Test Camera"
    header["TELESCOP"] = "Test Scope"
    header["XPIXSZ"] = 3.76
    header["FOCALLEN"] = 400.0
    header["CCD-TEMP"] = -10.0
    header["XBINNING"] = 1
    header["YBINNING"] = 1
    return header


def write(path: Path, data: np.ndarray, header) -> None:
    from astropy.io import fits

    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(np.clip(data, 0, 65535).astype(np.uint16), header=header).writeto(path)


def star_field(rng: np.random.Generator, sigma: float, shift: tuple[float, float], angle_deg: float = 0.0) -> np.ndarray:
    """The same 80 stars at Gaussian width `sigma`, shifted and turned, on
    a flat sky with read noise."""
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    image = np.full((HEIGHT, WIDTH), 500.0, dtype=np.float32)
    stars = np.random.default_rng(11)
    turn = np.radians(angle_deg)
    centre_x, centre_y = (WIDTH - 1) / 2, (HEIGHT - 1) / 2
    for _ in range(80):
        x, y = stars.uniform(8, WIDTH - 8) - centre_x, stars.uniform(8, HEIGHT - 8) - centre_y
        x, y = x * np.cos(turn) - y * np.sin(turn), x * np.sin(turn) + y * np.cos(turn)
        x, y = x + centre_x + shift[0], y + centre_y + shift[1]
        flux = stars.uniform(10_000, 150_000)
        image += flux / (2 * np.pi * sigma**2) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    return image + rng.normal(0.0, 8.0, image.shape)


def build_corpus(root: Path) -> Path:
    """Write the corpus under `root`: `telescopes/` for `--corpus`,
    `library/` for `--library`, and an empty `catalogs/` for
    `--catalog-dir`. Returns `root`."""
    rng = np.random.default_rng(5)
    lights = root / LIGHTS / "2026-01-09" / "LIGHT"
    for index in range(FRAMES):
        when = START + timedelta(minutes=2 * index)
        header = common_header("LIGHT", when, 60.0)
        header["FILTER"] = "L"
        header["OBJECT"] = "Test Nebula"
        header["RA"] = 83.8
        header["DEC"] = -5.4
        # Seeing worsens through the night.
        sigma = 1.6 + 2.0 * index / (FRAMES - 1)
        shift = (rng.uniform(-3, 3), rng.uniform(-3, 3))
        write(lights / f"light_{index:03d}.fits", star_field(rng, sigma, shift) + 30.0, header)

    raw = root / "raw"
    bias_paths, flat_paths = [], []
    for index in range(5):
        header = common_header("BIAS", START - timedelta(hours=2, minutes=index), 0.0)
        path = raw / f"bias_{index}.fits"
        write(path, 30.0 + rng.normal(0, 3, (HEIGHT, WIDTH)), header)
        bias_paths.append(path)
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    vignette = 1.0 - 0.2 * (((xx - WIDTH / 2) / WIDTH) ** 2 + ((yy - HEIGHT / 2) / HEIGHT) ** 2)
    for index in range(5):
        header = common_header("FLAT", START - timedelta(hours=1, minutes=index), 1.0)
        header["FILTER"] = "L"
        path = raw / f"flat_{index}.fits"
        write(path, 30.0 + 20_000.0 * vignette + rng.normal(0, 50, (HEIGHT, WIDTH)), header)
        flat_paths.append(path)
    masters = root / "library" / "db-test" / "calibration-masters"
    masters.mkdir(parents=True)
    seiza.build_bias(bias_paths, masters / "bias-test.fits")
    seiza.build_flat(flat_paths, masters / "flat-test.fits", bias=masters / "bias-test.fits")
    (root / "catalogs").mkdir()
    return root
