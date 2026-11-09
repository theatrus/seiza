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

    runner = SimpleNamespace(noise_map=big(np.ones((side, side), np.float32)), background=100.0, noise=1.0,
                             measure_noise=1.0)
    valid = big(np.ones((side, side), dtype=bool))
    cell = {"cell": [0, 0], "ratio": 0.75, "width_px": 2.0 * sm.FWHM_PER_SIGMA}
    assert sharpen_sheets.reblur(big(blurred), big(sharp), runner, valid, cell) < 1.0
    # Twice the stars' light where none was: no blur of the input hid it.
    assert sharpen_sheets.reblur(big(blurred), big(100.0 + 2 * (sharp - 100.0)), runner, valid, cell) > 20.0
    assert sharpen_sheets.reblur(big(blurred), big(sharp), runner, valid, {**cell, "ratio": 1.02}) is None


def test_a_star_s_ellipticity_is_its_light_s_and_roundness_counts_the_stars_pinched():
    from seiza_training import psf_fields

    side = 124
    down, across = np.mgrid[0:side, 0:side]

    def star(x, y, sx, sy):
        return 5000.0 * np.exp(-((across - x) ** 2 / (2 * sx**2) + (down - y) ** 2 / (2 * sy**2)))

    places = [(30.3, 30.6), (92.2, 32.1), (31.5, 92.4), (93.0, 93.7)]
    round_image = (100.0 + sum(star(x, y, 2.0, 2.0) for x, y in places)).astype(np.float32)
    pinched = (100.0 + sum(star(x, y, 2.0, 1.4) for x, y in places)).astype(np.float32)
    assert sharpen_sheets.ellipticity(round_image, 30.3, 30.6, 9.0) == pytest.approx(0.0, abs=0.01)
    assert sharpen_sheets.ellipticity(pinched, 30.3, 30.6, 9.0) == pytest.approx(0.3, abs=0.02)
    columns = psf_fields.DETECTION_COLUMNS
    points = np.zeros((4, len(columns)), dtype=np.float32)
    points[:, :2] = places
    points[:, columns.index("flux")], points[:, columns.index("snr")] = 1e5, 200.0
    points[:, columns.index("detected")] = 1.0
    runner = SimpleNamespace(points=points, widths=np.array([[4.7]]))
    valid = np.ones((side, side), dtype=bool)
    found = sharpen_sheets.roundness(round_image, pinched, runner, valid)
    assert found["stars"] == 4 and found["worse_by_0_05"] == 1.0
    assert found["input"][1] == pytest.approx(0.0, abs=0.01) and found["output"][1] == pytest.approx(0.3, abs=0.02)
    assert sharpen_sheets.roundness(round_image, round_image, runner, valid)["worse_by_0_05"] == 0.0


def test_edges_read_a_steepened_edge_and_noise_sharpened_along_it():
    from seiza_training import psf_fields

    rng = np.random.default_rng(3)
    side = 512
    down, across = np.mgrid[0:side, 0:side]

    def scene(steep):
        return (100.0 + 60.0 / (1.0 + np.exp(-(across - 256 + 0.1 * down) / steep))).astype(np.float32)

    noise = rng.normal(0, 1.0, (side, side)).astype(np.float32)
    image = scene(6.0) + noise
    points = np.zeros((0, len(psf_fields.DETECTION_COLUMNS)), dtype=np.float32)
    runner = SimpleNamespace(points=points, widths=np.array([[3.0]]), measure_noise=1.0)
    same = sharpen_sheets.edges(image, image, runner)
    assert same["gain"] == 1.0 and same["noise"] == 1.0 and same["cells"] > 500
    sharper = sharpen_sheets.edges(image, scene(3.0) + noise, runner)
    assert sharper["gain"] > 1.3 and sharper["noise"] == pytest.approx(1.0, abs=0.1)
    speckled = sharpen_sheets.edges(image, scene(6.0) + 2.0 * noise, runner)
    assert speckled["gain"] == pytest.approx(1.0, abs=0.1) and speckled["noise"] > 1.6
