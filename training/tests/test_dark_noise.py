"""The dark-noise pass's calibration follows the stacker's handling of
masters, and its gate measures what a master dark adds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from seiza_training.dark_noise import Calibration, master_excess  # noqa: E402

WIDTH, HEIGHT = 64, 48


def write(path: Path, data: np.ndarray, **cards) -> str:
    header = fits.Header()
    for key, value in cards.items():
        header[key] = value
    fits.PrimaryHDU(data, header=header).writeto(path)
    return str(path)


def planes():
    rng = np.random.default_rng(5)
    bias = (300.0 + rng.normal(0, 2, (HEIGHT, WIDTH))).astype(np.float32)
    current = rng.exponential(0.5, (HEIGHT, WIDTH)).astype(np.float32)
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    flat = (1.2 - 0.3 * ((xx - WIDTH / 2) ** 2 + (yy - HEIGHT / 2) ** 2) / (WIDTH / 2) ** 2).astype(np.float32)
    raw = (bias + 4 * current + 20 * flat + rng.normal(0, 3, (HEIGHT, WIDTH))).astype(np.float32)
    return bias, current, flat, raw


def test_a_bias_subtracted_dark_is_scaled_to_the_frame_exposure(tmp_path):
    bias, current, flat, raw = planes()
    masters = {
        "bias": write(tmp_path / "bias.fits", bias, SEIZAMST="BIAS"),
        "dark": write(tmp_path / "dark.fits", current, SEIZAMST="DARK", BIASSUB=True, EXPTIME=75.0, NCOMBINE=2),
        "flat": write(tmp_path / "flat.fits", flat, SEIZAMST="FLAT", BIASSUB=True, FLATNORM=True),
    }
    light = write(tmp_path / "light.fits", raw, EXPTIME=300.0)
    calibrated = Calibration(masters, WIDTH, HEIGHT).apply(light, 300.0)
    expected = (raw - bias - 4.0 * current) / (flat / np.median(flat[::7, ::7]))
    np.testing.assert_allclose(calibrated, expected, rtol=1e-5, atol=1e-4)


def test_a_dark_holding_its_bias_loses_it_before_scaling(tmp_path):
    bias, current, flat, raw = planes()
    masters = {
        "bias": write(tmp_path / "bias.fits", bias, SEIZAMST="BIAS"),
        "dark": write(tmp_path / "dark.fits", bias + current, SEIZAMST="DARK", EXPTIME=75.0),
        "flat": write(tmp_path / "flat.fits", flat + bias, SEIZAMST="FLAT"),
    }
    light = write(tmp_path / "light.fits", raw, EXPTIME=300.0)
    calibrated = Calibration(masters, WIDTH, HEIGHT).apply(light, 300.0)
    # A raw flat loses the bias too, then is normalized by its median.
    expected = (raw - bias - 4.0 * current) / (flat / np.median(flat[::7, ::7]))
    np.testing.assert_allclose(calibrated, expected, rtol=1e-4, atol=1e-3)


def test_the_gate_measures_a_scaled_master_against_one_frame(tmp_path):
    dark = write(tmp_path / "dark.fits", np.zeros((HEIGHT, WIDTH), np.float32), SEIZAMST="DARK", BIASSUB=True,
                 EXPTIME=75.0, NCOMBINE=2)
    assert master_excess({"dark": dark}, 300.0) == pytest.approx(4.0 / np.sqrt(2.0))
    assert master_excess({"dark": dark}, 75.0) == pytest.approx(1.0 / np.sqrt(2.0))
    assert master_excess({}, 300.0) is None
