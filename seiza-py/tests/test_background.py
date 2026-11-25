from __future__ import annotations

import numpy as np
import pytest

import seiza


def gradient(height: int = 72, width: int = 96) -> np.ndarray:
    y, x = np.mgrid[-1.0:1.0:complex(height), -1.0:1.0:complex(width)]
    return np.asarray(0.2 + 0.08 * x - 0.04 * y, dtype=np.float32)


def test_background_model_corrects_without_requiring_a_model_image() -> None:
    image = gradient()
    image[20:23, 30:33] += 0.7
    model = seiza.fit_background(
        image,
        degree=1,
        samples_per_axis=9,
        sample_radius=2,
    )
    assert model.width == image.shape[1]
    assert model.height == image.shape[0]
    assert model.channels == 1
    assert model.diagnostics["accepted_samples"] > 10
    corrected = model.correct(image)
    assert corrected.dtype == np.float32
    assert corrected.shape == image.shape
    assert corrected[36, 3] == pytest.approx(corrected[36, -4], abs=0.003)
    background = model.render()
    assert background.shape == image.shape
    assert background[36, 3] == pytest.approx(image[36, 3], abs=0.003)


def test_background_model_accepts_a_structure_mask_and_preserves_nan() -> None:
    image = gradient(80, 80)
    mask = np.zeros(image.shape, dtype=np.bool_)
    image[20:60, 20:60] += 0.4
    mask[20:60, 20:60] = True
    image[0, 0] = np.nan
    model = seiza.fit_background(
        image,
        mask=mask,
        degree=1,
        samples_per_axis=9,
        sample_radius=2,
    )
    corrected = model.correct(image, mode="subtract")
    assert np.isnan(corrected[0, 0])
    assert all(not mask[y, x] for x, y, *_ in model.samples())


def test_background_model_rejects_unknown_correction_modes() -> None:
    image = gradient(48, 64)
    model = seiza.fit_background(image, degree=1, sample_radius=2)
    with pytest.raises(ValueError, match="subtract or divide"):
        model.correct(image, mode="future")


def test_background_model_exposes_automatic_selection_and_strength() -> None:
    height, width = 84, 112
    y, x = np.mgrid[-1.0:1.0:complex(height), -1.0:1.0:complex(width)]
    image = np.asarray(
        0.3
        + 0.04 * x
        - 0.025 * y
        + 0.055 * np.sin(np.pi * x) * np.cos(np.pi * y / 2.0),
        dtype=np.float32,
    )
    model = seiza.fit_background(
        image,
        model="automatic",
        degree=2,
        rbf_smoothing=0.002,
        allow_radial_basis=True,
        minimum_improvement=0.08,
        samples_per_axis=14,
        sample_radius=1,
        search_steps=0,
    )
    assert model.model_kind == "radial_basis"
    assert model.diagnostics["model_selection"]["selected"] == "radial_basis"
    assert model.diagnostics["model_selection"]["candidates"][-1]["model"] == "radial_basis"
    unchanged = model.correct(image, strength=0.0)
    assert np.array_equal(unchanged, image)
    full = model.correct(image, strength=1.0)
    assert np.std(full) < np.std(image) * 0.1


def test_channels_fitted_apart_share_one_surface():
    y, x = np.mgrid[0:64, 0:96].astype(np.float32)
    u, v = x / 96 - 0.5, y / 64 - 0.5
    channels = [
        np.full((64, 96), 100.0, dtype=np.float32),
        (100.0 + 8.0 * u).astype(np.float32),
        (100.0 + 4.0 * u + 30.0 * (u * u + v * v)).astype(np.float32),
    ]
    models = [seiza.fit_background(channel, model="automatic", search_steps=0) for channel in channels]
    shared = seiza.select_shared_model(models)
    assert shared is not None
    assert shared["model"] in ("polynomial", "radial_basis")
    refit = [seiza.fit_background(channel, search_steps=0, **shared) for channel in channels]
    assert len({model.model_kind for model in refit}) == 1

    fixed = [seiza.fit_background(channel, model="polynomial", degree=1) for channel in channels]
    assert seiza.select_shared_model(fixed) is None


@pytest.mark.parametrize("mode", ["subtract", "divide"])
def test_a_corrected_rectangle_equals_that_rectangle_of_the_whole_image(mode: str) -> None:
    image = np.stack([gradient(), gradient() * 1.5, gradient() + 0.1], axis=-1)
    image = np.ascontiguousarray(image, dtype=np.float32)
    model = seiza.fit_background(image, degree=1, samples_per_axis=9, sample_radius=2)
    whole = model.correct(image, mode=mode, strength=0.7)
    x, y, width, height = 17, 9, 40, 23
    tile = np.ascontiguousarray(image[y : y + height, x : x + width])
    before = tile.copy()
    corrected = model.correct_region(tile, x, y, mode=mode, strength=0.7)
    assert corrected.dtype == np.float32
    assert corrected.shape == tile.shape
    assert np.array_equal(corrected, whole[y : y + height, x : x + width])
    assert np.array_equal(tile, before)


def test_a_rectangle_outside_the_fitted_image_is_refused() -> None:
    image = gradient()
    model = seiza.fit_background(image, degree=1, samples_per_axis=9, sample_radius=2)
    tile = np.ascontiguousarray(image[:16, :32])
    with pytest.raises(seiza.EngineError, match="outside the fitted"):
        model.correct_region(tile, image.shape[1] - 31, 0)
    with pytest.raises(seiza.EngineError, match="outside the fitted"):
        model.correct_region(tile, 0, image.shape[0] - 15)
    with pytest.raises(ValueError, match="channel count"):
        model.correct_region(np.zeros((4, 4, 3), dtype=np.float32), 0, 0)
