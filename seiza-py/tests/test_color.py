from __future__ import annotations

import numpy as np
import pytest

import seiza


def mono(value: float) -> np.ndarray:
    return np.array([[value]], dtype=np.float32)


def test_rgb_combines_mono_channels_without_normalization() -> None:
    rgb = seiza.combine_rgb(
        mono(0.2), mono(0.4), mono(0.6), normalization="none"
    )
    np.testing.assert_allclose(rgb, [[[0.2, 0.4, 0.6]]])


def test_lrgb_replaces_linear_luminance_and_preserves_chromaticity() -> None:
    rgb = seiza.combine_lrgb(
        mono(0.8),
        mono(0.2),
        mono(0.4),
        mono(0.1),
        normalization="none",
    )
    luminance = (
        0.2126 * rgb[0, 0, 0]
        + 0.7152 * rgb[0, 0, 1]
        + 0.0722 * rgb[0, 0, 2]
    )
    assert luminance == pytest.approx(0.8)
    assert rgb[0, 0, 0] / rgb[0, 0, 1] == pytest.approx(0.5)


def test_super_rgb_targets_channel_sum_and_preserves_chromaticity() -> None:
    rgb = seiza.combine_rgb(
        mono(0.2),
        mono(0.4),
        mono(0.1),
        luminance_mode="super",
        normalization="none",
    )
    luminance = (
        0.2126 * rgb[0, 0, 0]
        + 0.7152 * rgb[0, 0, 1]
        + 0.0722 * rgb[0, 0, 2]
    )
    assert luminance == pytest.approx(0.7)
    assert rgb[0, 0, 0] / rgb[0, 0, 1] == pytest.approx(0.5)
    assert rgb[0, 0, 2] / rgb[0, 0, 1] == pytest.approx(0.25)


def test_rgb_rejects_unknown_luminance_mode() -> None:
    with pytest.raises(ValueError, match="'native' or 'super'"):
        seiza.combine_rgb(
            mono(0.2), mono(0.4), mono(0.1), luminance_mode="sum"
        )


def test_super_lrgb_adds_all_channels_and_preserves_chromaticity() -> None:
    rgb = seiza.combine_lrgb(
        mono(0.8),
        mono(0.2),
        mono(0.4),
        mono(0.1),
        luminance_mode="super",
        normalization="none",
    )
    luminance = (
        0.2126 * rgb[0, 0, 0]
        + 0.7152 * rgb[0, 0, 1]
        + 0.0722 * rgb[0, 0, 2]
    )
    assert luminance == pytest.approx(1.5)
    assert rgb[0, 0, 0] / rgb[0, 0, 1] == pytest.approx(0.5)
    assert rgb[0, 0, 2] / rgb[0, 0, 1] == pytest.approx(0.25)


def test_super_lrgb_rejects_replace_weight() -> None:
    with pytest.raises(ValueError, match="only applies"):
        seiza.combine_lrgb(
            mono(0.8),
            mono(0.2),
            mono(0.4),
            mono(0.1),
            luminance_mode="super",
            luminance_weight=0.5,
            normalization="none",
        )


@pytest.mark.parametrize(
    ("palette", "expected"),
    [
        ("sho", [0.4, 0.2, 0.3]),
        ("hso", [0.2, 0.4, 0.3]),
        ("hoo", [0.2, 0.3, 0.3]),
    ],
)
def test_narrowband_direct_palettes(palette: str, expected: list[float]) -> None:
    rgb = seiza.combine_narrowband(
        mono(0.2),
        mono(0.3),
        mono(0.4) if palette != "hoo" else None,
        palette=palette,
        normalization="none",
    )
    np.testing.assert_allclose(rgb, [[expected]])


def test_three_filter_palette_requires_sii() -> None:
    with pytest.raises(ValueError, match="requires an SII"):
        seiza.combine_narrowband(
            mono(0.2), mono(0.3), palette="foraxx-sho", normalization="none"
        )


def test_foraxx_without_normalization_rejects_sensor_units() -> None:
    with pytest.raises(seiza.EngineError, match=r"finite samples in \[0, 1\]"):
        seiza.combine_narrowband(
            mono(1_000.0),
            mono(900.0),
            palette="foraxx-hoo",
            normalization="none",
        )
