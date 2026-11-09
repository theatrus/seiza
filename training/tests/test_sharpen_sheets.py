"""The PSF model's sheets: the ratio the network is asked for over an
image, from its field's widths."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

from seiza_training import sharpen_sheets  # noqa: E402


def test_the_ratio_brings_each_star_to_its_share_never_below_the_floor_nor_above_its_width():
    field = SimpleNamespace(arrays={"ys": np.array([0.0, 99.0]), "xs": np.array([0.0, 49.5, 99.0])})
    widths = np.array([[4.0, 2.5, np.nan], [1.2, 6.0, 3.0]])
    ratio = sharpen_sheets.ratio_map(widths, field, (100, 100), 0.5, 1.5)
    # At the nodes: half the width, but 1.5 px at least and never wider; a
    # node without a width takes the median's.
    assert ratio[0, 0] == pytest.approx(0.5)
    assert ratio[0, 50] == pytest.approx(1.5 / 2.5, abs=0.01)
    assert ratio[0, 99] == pytest.approx(0.5)
    assert ratio[99, 0] == pytest.approx(1.0)
    # Between them, bilinear.
    assert ratio[49, 0] == pytest.approx(0.5 + (1.0 - 0.5) * 49 / 99, abs=1e-5)
    assert ratio.shape == (100, 100) and ratio.dtype == np.float32


def test_noise_gain_reads_noise_left_taken_out_or_sharpened():
    rng = np.random.default_rng(0)
    image = (100.0 + rng.normal(0, 2.0, (512, 512))).astype(np.float32)
    assert sharpen_sheets.noise_gain(image, image) == {"pixel": 1.0, "blocks": 1.0}
    calm = (100.0 + 0.5 * (image - 100.0)).astype(np.float32)
    found = sharpen_sheets.noise_gain(image, calm)
    assert found["pixel"] == pytest.approx(0.5, rel=0.05) and found["blocks"] == pytest.approx(0.5, rel=0.05)
    # Away from its stars, the bright third's structure, sharpened or not.
    image[:256] += 50.0 + 5.0 * np.sin(np.arange(512) / 3.0)[None, :]
    points = np.array([[100.0, 100.0] + [0.0] * 8], dtype=np.float32)
    assert sharpen_sheets.noise_gain(image, image, points, 3.0)["detail"] == 1.0
    sharper = image.copy()
    sharper[:256] += 5.0 * np.sin(np.arange(512) / 3.0)[None, :]
    assert sharpen_sheets.noise_gain(image, sharper, points, 3.0)["detail"] > 1.3


def test_reblur_gives_back_the_input_of_a_sharpening_that_only_undid_blur():
    from seiza_training import sharpen_model as sm

    rng = np.random.default_rng(1)
    side = 256
    sharp = np.full((side, side), 100.0, np.float32)
    down, across = np.mgrid[0:side, 0:side]
    for x, y in rng.uniform(20, side - 20, (30, 2)):
        sharp += 5000.0 * np.exp(-((across - x) ** 2 + (down - y) ** 2) / (2 * 1.5**2)).astype(np.float32)
    blurred = sm.blurred_images(sharp, [(2.0**2 - 1.5**2) * np.eye(2)])[0]
    cells = sharpen_sheets.CELLS

    def big(plane):
        return np.tile(plane, (cells[1], cells[0]))

    runner = SimpleNamespace(noise_map=big(np.ones((side, side), np.float32)), background=100.0)
    valid = big(np.ones((side, side), dtype=bool))
    cell = {"cell": [0, 0], "ratio": 0.75, "width_px": 2.0 * sm.FWHM_PER_SIGMA}
    assert sharpen_sheets.reblur(big(blurred), big(sharp), runner, valid, cell) < 1.0
    # Twice the stars' light where none was: no blur of the input hid it.
    assert sharpen_sheets.reblur(big(blurred), big(100.0 + 2 * (sharp - 100.0)), runner, valid, cell) > 20.0
    assert sharpen_sheets.reblur(big(blurred), big(sharp), runner, valid, {**cell, "ratio": 1.02}) is None
