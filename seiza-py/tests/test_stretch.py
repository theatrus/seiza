from __future__ import annotations

import json

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


def test_ht_rescales_where_mtf_shifts() -> None:
    image = np.linspace(0.0, 1.0, 11, dtype=np.float32)[None]
    levels = {"shadows": 0.2, "midtone": 0.25, "highlights": 0.8}
    full = {"shadows": 0.0, "midtone": 0.25, "highlights": 1.0}

    shifted = seiza.stretch(image, model="mtf", **levels)
    unclipped = seiza.stretch(image, model="mtf", **full)
    np.testing.assert_allclose(shifted, unclipped, atol=1.0e-6)

    x = np.clip((image.astype(np.float64) - 0.2) / 0.6, 0.0, 1.0)
    expected = 0.75 * x / (0.5 * x + 0.25)
    rescaled = seiza.stretch(image, model="ht", **levels)
    np.testing.assert_allclose(rescaled, expected, atol=1.0e-6)
    np.testing.assert_allclose(
        seiza.stretch(image, model="ht", **full), unclipped, atol=1.0e-6
    )
    with pytest.raises(seiza.EngineError, match="HT shadows and highlights"):
        seiza.stretch(image, model="ht", shadows=0.8, highlights=0.2)


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


def test_a_stretch_resolved_on_a_reference_applies_to_other_images() -> None:
    rng = np.random.default_rng(13)
    reference = (0.05 + 0.005 * rng.standard_normal((64, 80, 3))).astype(np.float32)
    brighter = reference + np.float32(0.02)
    for options in (
        {"model": "auto-mtf"},
        {"model": "percentile-asinh", "color_strategy": "unlinked"},
        {"model": "median-anchor", "curves_boost": 0.4, "hdr_amount": 0.5},
        {"model": "layered", "contrast_recovery": 0.5, "scale_separation": 16},
        {"model": "ht", "shadows": 0.02, "midtone": 0.1, "highlights": 0.9},
        {"model": "ghs", "stretch_factor": 3.0},
    ):
        resolved = seiza.resolve_stretch(reference, **options)
        assert resolved.channels == 3
        own = seiza.stretch(reference, **options)
        np.testing.assert_array_equal(resolved.apply(reference), own)
        stored = json.loads(json.dumps(resolved.to_dict()))
        again = seiza.ResolvedStretch.from_dict(stored)
        assert again.to_dict() == resolved.to_dict()
        np.testing.assert_array_equal(again.apply(brighter), resolved.apply(brighter))
    # The reference's curve shows the difference that each image's own hides.
    resolved = seiza.resolve_stretch(reference, model="auto-mtf")
    assert "ResolvedStretch(channels=3" in repr(resolved)
    level = float(np.median(resolved.apply(reference)))
    assert float(np.median(resolved.apply(brighter))) > level + 0.1
    hidden = seiza.stretch(brighter, model="auto-mtf")
    assert float(np.median(hidden)) == pytest.approx(level, abs=0.02)
    # Another size is fine; another channel count is not.
    assert resolved.apply(reference[:10, :20].copy()).shape == (10, 20, 3)
    with pytest.raises(seiza.EngineError, match="channel layout"):
        resolved.apply(reference[..., 0].copy())
    with pytest.raises(ValueError, match="model must be"):
        seiza.resolve_stretch(reference, model="future-auto")
    with pytest.raises(ValueError, match="out of range"):
        seiza.ResolvedStretch.from_dict(
            {
                "channels": 1,
                "color_strategy": "linked",
                "curves": [{"type": "linear", "black": 1.0, "white": float("nan")}],
            }
        )


def test_a_plans_dict_says_what_its_curves_are() -> None:
    plan = seiza.resolve_stretch(
        np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
    ).to_dict()
    assert set(plan) == {"channels", "color_strategy", "curves"}
    assert plan["curves"][0]["type"] == "asinh"


def test_stretch_rejects_unknown_models() -> None:
    with pytest.raises(ValueError, match="model must be"):
        seiza.stretch(np.zeros((2, 2), dtype=np.float32), model="future-auto")
