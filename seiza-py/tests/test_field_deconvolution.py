from __future__ import annotations

import numpy as np
import pytest

import seiza


def star_field(
    width: int = 240,
    height: int = 200,
    fwhm_left: float = 3.2,
    fwhm_right: float = 4.2,
    seed: int = 3,
) -> np.ndarray:
    """Gaussian stars on a grid whose FWHM grows from left to right."""
    rng = np.random.default_rng(seed)
    image = np.full((height, width), 100.0)
    yy, xx = np.mgrid[-12:13, -12:13]
    for cy in range(20, height - 20, 24):
        for cx in range(20, width - 20, 24):
            fx, fy = rng.uniform(-0.5, 0.5, 2)
            fwhm = fwhm_left + (fwhm_right - fwhm_left) * cx / (width - 1)
            sigma = fwhm / 2.35482
            flux = rng.uniform(2.0e4, 8.0e4)
            star = np.exp(-0.5 * ((xx - fx) ** 2 + (yy - fy) ** 2) / sigma**2)
            image[cy - 12 : cy + 13, cx - 12 : cx + 13] += flux * star / (
                2 * np.pi * sigma**2
            )
    image += rng.normal(0.0, 2.0, image.shape)
    return np.ascontiguousarray(image, dtype=np.float32)


def measure(image: np.ndarray) -> seiza.PsfField:
    return seiza.PsfField.measure(image, sparse="global")


def test_remaining_kernel_describes_the_blur_to_the_target() -> None:
    image = star_field()
    field = measure(image)
    kernel = seiza.RemainingKernel(field, target_fwhm=2.2, node_spacing=64)

    assert (kernel.width, kernel.height, kernel.channels) == (240, 200, 1)
    assert kernel.node_spacing == 64
    assert kernel.node_count == 5 * 5
    assert kernel.kernel_size == field.stamp_size
    assert kernel.target_fwhm_range() == pytest.approx((2.2, 2.2))
    low, high = kernel.psf_fwhm_range()
    assert 3.0 < low < high < 4.5

    local = kernel.kernel_at(120.0, 100.0)
    assert local.shape == (kernel.kernel_size, kernel.kernel_size)
    assert float(local.sum()) == pytest.approx(1.0, abs=1e-4)

    flat = np.full_like(image, 7.0)
    # Flat to a few tenths of a percent, least well in the corners.
    np.testing.assert_allclose(kernel.blur(flat), flat, rtol=1e-2)
    with pytest.raises(ValueError):
        kernel.blur(np.zeros((10, 10), dtype=np.float32))
    with pytest.raises(ValueError):
        kernel.psf_fwhm_range(channel=1)


def test_targets_are_chosen_one_way() -> None:
    field = measure(star_field())
    best = seiza.RemainingKernel(field, node_spacing=64)
    low, high = best.target_fwhm_range()
    assert low == pytest.approx(high)
    assert low == pytest.approx(best.psf_fwhm_range()[0])
    ratio = seiza.RemainingKernel(field, target_ratio=0.6, node_spacing=64)
    assert ratio.target_fwhm_range()[1] > ratio.target_fwhm_range()[0]
    with pytest.raises(ValueError):
        seiza.RemainingKernel(field, target_fwhm=2.0, target_ratio=0.5)
    with pytest.raises(ValueError):
        seiza.RemainingKernel(field, target_fwhm=1.0)
    with pytest.raises(ValueError):
        seiza.RemainingKernel(field, node_spacing=8)


def test_deconvolution_sharpens_and_reblurs_to_its_input() -> None:
    image = star_field()
    field = measure(image)
    kernel = seiza.RemainingKernel(field, target_fwhm=2.2, node_spacing=64)
    restored = seiza.deconvolve_field(
        image, kernel, iterations=30, amount=1.0, noise_fraction=0.0, max_correction=3.0
    )
    assert restored.shape == image.shape and restored.dtype == np.float32
    assert float(restored.sum()) == pytest.approx(float(image.sum()), rel=1e-4)
    before = field.shape_at(120.0, 100.0)
    after = measure(restored).shape_at(120.0, 100.0)
    assert after.fwhm < before.fwhm - 0.6

    noise = np.full_like(image, 2.0)
    consistency = seiza.reblur_consistency(restored, image, noise, kernel)
    channel = consistency.channels[0]
    assert channel.samples == image.size
    assert len(channel.band_power) == 8
    assert channel.spectral_flatness is not None
    # The input offered as its own sharpened image does not reblur to itself.
    soft = seiza.reblur_consistency(image, image, noise, kernel).channels[0]
    assert soft.reduced_chi_squared > channel.reduced_chi_squared


def test_masked_deconvolution_keeps_missing_samples() -> None:
    image = star_field()
    kernel = seiza.RemainingKernel(measure(image), target_fwhm=2.2, node_spacing=64)
    image[:, :3] = np.nan
    with pytest.raises(seiza.EngineError):
        seiza.deconvolve_field(image, kernel)
    restored = seiza.deconvolve_field(image, kernel, masked=True)
    assert np.isnan(restored[:, :3]).all()
    assert np.isfinite(restored[:, 3:]).all()
