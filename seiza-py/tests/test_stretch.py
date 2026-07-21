from __future__ import annotations

import numpy as np
import pytest

import seiza


def test_percentile_asinh_returns_display_referred_f32() -> None:
    image = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
    stretched = seiza.stretch(
        image,
        model="percentile-asinh",
        black_percentile=0.01,
        white_percentile=0.995,
        strength=10.0,
    )
    assert stretched.dtype == np.float32
    assert stretched.shape == image.shape
    assert stretched[0, 0] == 0.0
    assert stretched[-1, -1] == 1.0
    assert stretched[5, 5] > 0.5


def test_luminance_preserving_stretch_keeps_rgb_ratios() -> None:
    image = np.array([[[0.2, 0.4, 0.1]]], dtype=np.float32)
    stretched = seiza.stretch(
        image,
        model="asinh",
        black=0.0,
        white=1.0,
        strength=10.0,
        color_strategy="luminance-preserving",
    )
    assert stretched[0, 0, 0] / stretched[0, 0, 1] == pytest.approx(0.5)
    assert stretched[0, 0, 2] / stretched[0, 0, 1] == pytest.approx(0.25)


def test_manual_ghs_exposes_the_reference_parameters() -> None:
    image = np.array([[0.0, 0.25, 1.0]], dtype=np.float32)
    stretched = seiza.stretch(
        image,
        model="ghs",
        stretch_factor=np.log1p(10.0),
        local_intensity=0.0,
        symmetry_point=0.0,
        protect_shadows=0.0,
        protect_highlights=1.0,
    )
    expected = (1.0 - np.exp(-2.5)) / (1.0 - np.exp(-10.0))
    assert stretched[0, 0] == 0.0
    assert stretched[0, 1] == pytest.approx(expected, abs=1.0e-6)
    assert stretched[0, 2] == 1.0


def test_stretch_rejects_unknown_models() -> None:
    with pytest.raises(ValueError, match="model must be"):
        seiza.stretch(np.zeros((2, 2), dtype=np.float32), model="future-auto")
