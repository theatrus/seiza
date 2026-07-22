from __future__ import annotations

import numpy as np
import pytest

import seiza


def star_field(width: int = 64, height: int = 48) -> np.ndarray:
    """Deterministic uint8 test image: noise floor plus a few bright blobs."""
    rng = np.random.default_rng(7)
    image = rng.integers(5, 20, size=(height, width), dtype=np.uint8)
    for cy, cx in [(12, 16), (30, 40), (20, 55)]:
        yy, xx = np.mgrid[0:height, 0:width]
        blob = 200.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 6.0)
        image = np.clip(image + blob.astype(np.uint16), 0, 255).astype(np.uint8)
    return image


def test_gaussian_blur_uint8_preserves_dtype_and_mass() -> None:
    image = star_field()
    blurred = seiza.gaussian_blur(image, 1.0)
    assert blurred.dtype == np.uint8
    assert blurred.shape == image.shape
    # Blur must smooth: the peak drops, the neighborhood mean survives.
    assert blurred.max() < image.max()
    assert abs(int(blurred.mean()) - int(image.mean())) <= 1


def test_gaussian_blur_float32_matches_kernel_convolution() -> None:
    image = np.zeros((9, 9), dtype=np.float32)
    image[4, 4] = 1.0
    blurred = seiza.gaussian_blur(image, 1.0, ksize=3)
    assert blurred.dtype == np.float32
    # Separable response to an impulse is the outer product of the
    # normalized 3-tap sigma-1 kernel.
    taps = np.exp(-np.array([-1.0, 0.0, 1.0]) ** 2 / 2.0)
    taps /= taps.sum()
    assert blurred[4, 4] == pytest.approx(taps[1] * taps[1], rel=1e-6)
    assert blurred[4, 3] == pytest.approx(taps[1] * taps[0], rel=1e-6)
    assert blurred[3, 3] == pytest.approx(taps[0] * taps[0], rel=1e-6)


def test_gaussian_blur_rejects_even_ksize_and_other_dtypes() -> None:
    image = star_field()
    with pytest.raises(ValueError):
        seiza.gaussian_blur(image, 1.0, ksize=4)
    with pytest.raises(ValueError):
        seiza.gaussian_blur(image.astype(np.float64), 1.0)


def test_median_blur3_removes_salt_noise() -> None:
    image = np.full((16, 16), 10, dtype=np.uint8)
    image[8, 8] = 255
    cleaned = seiza.median_blur3(image)
    assert cleaned[8, 8] == 10


def test_canny_finds_a_step_edge() -> None:
    image = np.zeros((16, 32), dtype=np.uint8)
    image[:, 16:] = 200
    edges = seiza.canny(image, 10, 80)
    assert edges.dtype == np.uint8
    edge_cols = np.unique(np.nonzero(edges)[1])
    assert edge_cols.size > 0
    assert set(edge_cols.tolist()) <= {15, 16}


def test_otsu_separates_bimodal_image() -> None:
    image = np.concatenate(
        [np.full(500, 50, dtype=np.uint8), np.full(500, 200, dtype=np.uint8)]
    ).reshape(20, 50)
    threshold = seiza.otsu_threshold(image)
    assert 50 <= threshold < 200
    binary = seiza.otsu_binary(image)
    assert int((binary == 255).sum()) == 500


def test_erode_and_dilate_roundtrip_a_square() -> None:
    image = np.zeros((16, 16), dtype=np.uint8)
    image[4:12, 4:12] = 255
    eroded = seiza.erode(image, shape="rect")
    assert int((eroded == 255).sum()) == 36
    reopened = seiza.dilate(eroded, shape="rect")
    assert np.array_equal(reopened, image)


def test_find_contours_and_area() -> None:
    image = np.zeros((16, 16), dtype=np.uint8)
    image[4:9, 4:9] = 255  # 5x5 square
    contours = seiza.find_contours(image)
    assert len(contours) == 1
    contour = contours[0]
    assert contour.dtype == np.int32
    assert contour.shape[1] == 2
    # cv::contourArea semantics: polygon over pixel centers, so 4x4 = 16.
    assert seiza.contour_area(contour) == pytest.approx(16.0)


def test_dt_filter_smooths_but_keeps_edges() -> None:
    image = np.full((8, 32), 10.0, dtype=np.float32)
    image[:, 16:] = 200.0
    image += np.where((np.indices(image.shape).sum(axis=0) % 2) == 0, 0.5, -0.5).astype(
        np.float32
    )
    smoothed = seiza.dt_filter(image, image, 8.0, 30.0)
    assert smoothed.dtype == np.float32
    assert abs(float(smoothed[4, 4]) - 10.0) < 0.4
    assert float(smoothed[4, 15]) < 30.0
    assert float(smoothed[4, 16]) > 180.0


def test_remove_structures_keeps_peak_drops_gradient() -> None:
    height = width = 32
    yy, xx = np.mgrid[0:height, 0:width]
    image = (xx + yy).astype(np.float64) * 20.0
    image[16, 16] += 400.0
    residual = seiza.remove_structures(image, layers=3)
    assert residual.dtype == np.float64
    peak = residual[16, 16]
    background = abs(residual[8, 8])
    assert peak > 100.0
    assert peak > 4.0 * background
    atrous = seiza.remove_structures(image, layers=3, method="atrous")
    assert atrous.shape == image.shape
