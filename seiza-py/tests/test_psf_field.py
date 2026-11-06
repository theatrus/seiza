from __future__ import annotations

import numpy as np
import pytest

import seiza


def star_field(
    width: int = 420,
    height: int = 360,
    spacing: int = 32,
    fwhm_left: float = 2.8,
    fwhm_right: float = 3.6,
    seed: int = 4,
) -> np.ndarray:
    """Gaussian stars on a grid whose FWHM grows from left to right."""
    rng = np.random.default_rng(seed)
    image = np.full((height, width), 100.0)
    yy, xx = np.mgrid[-10:11, -10:11]
    for cy in range(24, height - 24, spacing):
        for cx in range(24, width - 24, spacing):
            fx, fy = rng.uniform(-0.5, 0.5, 2)
            fwhm = fwhm_left + (fwhm_right - fwhm_left) * cx / (width - 1)
            sigma = fwhm / 2.35482
            flux = rng.uniform(2.0e4, 8.0e4)
            star = np.exp(-0.5 * ((xx - fx) ** 2 + (yy - fy) ** 2) / sigma**2)
            image[cy - 10 : cy + 11, cx - 10 : cx + 11] += flux * star / (
                2 * np.pi * sigma**2
            )
    image += rng.normal(0.0, 2.0, image.shape)
    return np.ascontiguousarray(image, dtype=np.float32)


def test_measures_a_varying_field() -> None:
    image = star_field()
    field = seiza.PsfField.measure(image)

    assert (field.width, field.height, field.channels) == (420, 360, 1)
    assert field.stamp_size == 33
    assert field.star_count >= 30
    assert field.component_count == 3
    assert not field.is_global
    assert field.warnings == []
    assert all(star.role == "core" for star in field.stars)
    assert field.channel_fits[0].relative_residual < 0.1
    assert field.channel_fits[0].wing_overlap is None

    stamp = field.stamp_at(210.0, 180.0)
    assert stamp.shape == (33, 33)
    assert stamp.dtype == np.float32
    assert float(stamp.sum()) == pytest.approx(1.0, abs=1e-5)

    left = field.shape_at(40.0, 180.0)
    right = field.shape_at(380.0, 180.0)
    assert left is not None and right is not None
    assert left.fwhm == pytest.approx(2.9, abs=0.2)
    assert right.fwhm == pytest.approx(3.5, abs=0.2)
    assert abs(left.e1) < 0.05 and abs(left.e2) < 0.05
    assert left.ellipticity == pytest.approx(np.hypot(left.e1, left.e2))
    with pytest.raises(ValueError):
        field.shape_at(40.0, 180.0, channel=1)


def test_rgb_channels_have_their_own_field() -> None:
    narrow = star_field(seed=7)
    wide = star_field(fwhm_left=3.8, fwhm_right=4.6, seed=7)
    image = np.ascontiguousarray(np.stack([narrow, narrow, wide], axis=-1))
    field = seiza.PsfField.measure(image)

    assert field.channels == 3
    assert field.stamp_at(210.0, 180.0).shape == (33, 33, 3)
    red = field.shape_at(210.0, 180.0, channel=0)
    blue = field.shape_at(210.0, 180.0, channel=2)
    assert red is not None and blue is not None
    assert blue.fwhm > red.fwhm + 0.7


def test_sparse_fields_raise_unless_a_global_psf_is_accepted() -> None:
    image = star_field(200, 160)
    with pytest.raises(seiza.TooFewStarsError):
        seiza.PsfField.measure(image)
    assert issubclass(seiza.TooFewStarsError, seiza.EngineError)

    field = seiza.PsfField.measure(image, sparse="global")
    assert field.is_global
    assert field.component_count == 0
    assert len(field.warnings) == 1
    np.testing.assert_array_equal(field.stamp_at(10.0, 10.0), field.stamp_at(190.0, 150.0))
    with pytest.raises(ValueError):
        seiza.PsfField.measure(image, sparse="sometimes")


def test_catalogue_neighbours_disqualify_stars() -> None:
    image = star_field()
    plain = seiza.PsfField.measure(image)
    # A faint catalogue companion beside every star that gave a stamp: none
    # is isolated any more.
    catalogue = [(star.x, star.y, 10.0) for star in plain.stars]
    catalogue += [(star.x + 5.0, star.y, 12.0) for star in plain.stars]
    with pytest.raises(seiza.TooFewStarsError):
        seiza.PsfField.measure(image, catalogue=catalogue)
    # Companions six magnitudes down do not count.
    catalogue = [(star.x, star.y, 10.0) for star in plain.stars]
    catalogue += [(star.x + 5.0, star.y, 16.0) for star in plain.stars]
    field = seiza.PsfField.measure(image, catalogue=catalogue)
    assert field.star_count == plain.star_count


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        seiza.PsfField.measure(np.zeros((10, 10, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        seiza.PsfField.measure(star_field(), stamp_size=32)
