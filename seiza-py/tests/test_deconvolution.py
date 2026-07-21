from __future__ import annotations

import numpy as np
import pytest

import seiza


def gaussian_star(size: int = 41, fwhm: float = 2.8) -> np.ndarray:
    axis = np.arange(size, dtype=np.float32) - size // 2
    x, y = np.meshgrid(axis, axis)
    sigma = np.float32(fwhm / 2.35482)
    star = np.exp(-0.5 * (x * x + y * y) / (sigma * sigma))
    return np.asarray(star / star.sum(), dtype=np.float32)


def test_deconvolution_restores_mono_and_preserves_flux() -> None:
    image = gaussian_star()
    restored = seiza.deconvolve(image, psf_fwhm=2.8)

    assert restored.dtype == np.float32
    assert restored.shape == image.shape
    assert restored[20, 20] > image[20, 20]
    assert float(restored.sum()) == pytest.approx(float(image.sum()), abs=1.0e-5)


def test_deconvolution_preserves_rgb_layout() -> None:
    red = gaussian_star(31)
    image = np.stack(
        [red, np.full_like(red, 0.25), np.full_like(red, 0.5)], axis=-1
    )
    restored = seiza.deconvolve(image, psf_fwhm=2.8)

    assert restored[15, 15, 0] > image[15, 15, 0]
    np.testing.assert_array_equal(restored[..., 1], image[..., 1])
    np.testing.assert_array_equal(restored[..., 2], image[..., 2])


def test_deconvolution_rejects_noncontiguous_and_invalid_inputs() -> None:
    image = gaussian_star()
    with pytest.raises(ValueError, match="C-contiguous"):
        seiza.deconvolve(image[:, ::2], psf_fwhm=2.8)
    with pytest.raises(seiza.EngineError, match="amount"):
        seiza.deconvolve(image, psf_fwhm=2.8, amount=1.1)
