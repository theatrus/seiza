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


def test_median_anchor_places_the_median_on_the_target() -> None:
    rng = np.random.default_rng(7)
    image = (0.05 + 0.01 * rng.standard_normal((64, 64))).astype(np.float32)
    image[::9, ::9] = 0.9
    stretched = seiza.stretch(image, model="median-anchor", target_median=0.3)
    assert float(np.median(stretched)) == pytest.approx(0.3, abs=0.02)
    # The default target differs from auto-mtf's: 0.25.
    default = seiza.stretch(image, model="median-anchor")
    assert float(np.median(default)) == pytest.approx(0.25, abs=0.02)
    # No black clipping keeps the darkest pixel at zero.
    unclipped = seiza.stretch(image, model="median-anchor", black_point_sigma=None)
    assert float(unclipped.min()) == pytest.approx(0.0, abs=1.0e-6)
    with pytest.raises(seiza.EngineError):
        seiza.stretch(image, model="median-anchor", curves_boost=2.0)


def test_layered_places_the_background_and_keeps_rgb_in_range() -> None:
    rng = np.random.default_rng(11)
    mono = (0.05 + 0.005 * rng.standard_normal((96, 128))).astype(np.float32)
    mono[::13, ::17] = 0.9
    stretched = seiza.stretch(mono, model="layered", target_background=0.2)
    assert stretched.shape == mono.shape
    assert float(np.median(stretched)) == pytest.approx(0.2, abs=0.02)
    # With no contrast recovery or compression it is the plain curve.
    anchor = seiza.stretch(
        mono, model="median-anchor", target_median=0.2, black_point_sigma=3.75
    )
    np.testing.assert_array_equal(stretched, anchor)

    rgb = np.stack([mono * 0.9, mono, mono * 1.1], axis=-1).astype(np.float32)
    rgb[3, 4] = np.nan
    lifted = seiza.stretch(
        rgb,
        model="layered",
        contrast_recovery=0.6,
        dynamic_range_compression=0.5,
        scale_separation=32,
        saturation_amount=0.4,
        saturation_boost=0.3,
        lightness_mask=False,
    )
    assert lifted.shape == rgb.shape
    assert lifted.dtype == np.float32
    assert float(lifted.min()) >= 0.0 and float(lifted.max()) <= 1.0
    assert np.all(lifted[3, 4] == 0.0)
    with pytest.raises(seiza.EngineError):
        seiza.stretch(rgb, model="layered", color_strategy="unlinked")
    with pytest.raises(seiza.EngineError):
        seiza.stretch(mono, model="layered", scale_separation=4)


def test_stretch_rejects_unknown_models() -> None:
    with pytest.raises(ValueError, match="model must be"):
        seiza.stretch(np.zeros((2, 2), dtype=np.float32), model="future-auto")
