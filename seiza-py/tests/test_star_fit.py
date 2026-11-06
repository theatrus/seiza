from __future__ import annotations

import numpy as np
import pytest

import seiza


def scene(
    width: int = 300, height: int = 270, seed: int = 3, sigma_noise: float = 2.0
) -> tuple[np.ndarray, list[tuple[float, float, float]], list[float]]:
    """Gaussian stars of FWHM 3 on a grid over a sloping sky, with their list."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    image = 100.0 + 0.04 * xx - 0.02 * yy
    sigma = 3.0 / 2.35482
    stars, fluxes = [], []
    for cy in range(24, height - 24, 30):
        for cx in range(24, width - 24, 30):
            x, y = cx + rng.uniform(-0.5, 0.5), cy + rng.uniform(-0.5, 0.5)
            flux = rng.uniform(5.0e3, 6.0e4)
            image = image + flux / (2 * np.pi * sigma**2) * np.exp(
                -0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / sigma**2
            )
            stars.append((x, y, -2.5 * np.log10(flux)))
            fluxes.append(flux)
    image = image + rng.normal(0.0, sigma_noise, image.shape)
    return np.ascontiguousarray(image, dtype=np.float32), stars, fluxes


def test_fits_removes_and_renders_stars() -> None:
    image, stars, fluxes = scene()
    noise = np.full_like(image, 2.0)
    field = seiza.PsfField.measure(image)
    fit = seiza.StarFit.fit(image, field, stars, noise=noise)

    assert (fit.width, fit.height, fit.channels) == (300, 270, 1)
    assert len(fit.stars) == len(stars)
    for star, truth in zip(fit.stars, fluxes):
        assert star.flux[0] == pytest.approx(truth, rel=0.03)
        assert star.flux_error[0] > 0.0
        assert not star.saturated

    model = fit.model()
    removed = fit.subtract(image)
    np.testing.assert_allclose(removed + model, image, atol=0.05)
    # Away from the cores, where the PSF model's percent-level error
    # shows, what is left is the sky and its noise.
    yy, xx = np.mgrid[0:270, 0:300]
    sky = 100.0 + 0.04 * xx - 0.02 * yy
    clear = np.ones(image.shape, dtype=bool)
    for x, y, _ in stars:
        clear &= np.hypot(xx - x, yy - y) > 6.0
    assert float(np.std((removed - sky)[clear])) < 2.4
    assert float(np.std((image - sky)[~clear])) > 10 * float(np.std((removed - sky)[~clear]))

    rendered = fit.render(2.0)
    assert float(rendered.sum()) == pytest.approx(sum(fluxes), rel=0.03)
    sharpened = fit.sharpen(image, 2.0)
    np.testing.assert_allclose(sharpened, removed + rendered, atol=0.05)
    after = seiza.PsfField.measure(sharpened, sparse="global")
    assert after.shape_at(150.0, 135.0).fwhm == pytest.approx(2.0, abs=0.25)


def test_refined_positions_follow_the_image() -> None:
    image, stars, _ = scene(seed=5)
    field = seiza.PsfField.measure(image)
    offset = [(x + 0.4, y - 0.3, m) for x, y, m in stars]
    fit = seiza.StarFit.fit(image, field, offset, positions="refined")
    for star, truth in zip(fit.stars, stars):
        assert star.x == pytest.approx(truth[0], abs=0.08)
        assert star.y == pytest.approx(truth[1], abs=0.08)
    with pytest.raises(ValueError):
        seiza.StarFit.fit(image, field, offset, positions="loose")


def test_rejects_mismatched_images() -> None:
    image, stars, _ = scene()
    field = seiza.PsfField.measure(image)
    with pytest.raises(ValueError):
        seiza.StarFit.fit(image[:100], field, stars)
    fit = seiza.StarFit.fit(image, field, stars)
    with pytest.raises(ValueError):
        fit.render(0.0)
    with pytest.raises(ValueError):
        fit.subtract(image[:100].copy())
