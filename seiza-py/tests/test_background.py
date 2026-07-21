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
