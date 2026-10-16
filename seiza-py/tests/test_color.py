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


def bordered(blank_rows: int) -> np.ndarray:
    """A 128x128 frame blank across its top `blank_rows` rows."""
    values = np.full((128, 128), 0.25, dtype=np.float32)
    values[:blank_rows, :] = np.nan
    return values


def test_crop_trims_the_registration_border() -> None:
    rgb = seiza.combine_rgb(
        bordered(0),
        bordered(3),
        bordered(5),
        normalization="none",
        crop="inscribed",
    )
    assert rgb.shape == (123, 128, 3)
    assert np.isfinite(rgb).all()


def test_uncropped_composition_keeps_the_blank_border() -> None:
    rgb = seiza.combine_rgb(
        bordered(0), bordered(3), bordered(5), normalization="none"
    )
    assert rgb.shape == (128, 128, 3)
    assert np.isnan(rgb[0, 0, 0])


def test_crop_report_flags_a_channel_far_from_the_others() -> None:
    report = seiza.crop_report(
        {"red": bordered(0), "green": bordered(2), "blue": bordered(80)},
        crop="inscribed",
    )
    assert report["region"] == (0, 80, 128, 48)
    assert report["grid"] == (128, 128)
    assert report["retained_fraction"] == pytest.approx(48 / 128)
    flagged = [
        channel for channel in report["channels"] if channel["off_center"]
    ]
    assert [channel["name"] for channel in flagged] == ["blue"]
    assert flagged[0]["center_offset_pixels"] > 32.0


def test_crop_report_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match="crop must be"):
        seiza.crop_report({"red": bordered(0)}, crop="tightest")


def wcs_header() -> dict[str, object]:
    return {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 64.5,
        "CRPIX2": 64.5,
        "CRVAL1": 83.8,
        "CRVAL2": -5.4,
        "CD1_1": -0.0003,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.0003,
        "OBJECT": "M42",
    }


def test_composition_writes_its_output_with_the_reference_wcs(tmp_path) -> None:
    fits = pytest.importorskip("astropy.io.fits")
    output = tmp_path / "rgb.fits"
    rgb = seiza.combine_rgb(
        bordered(0),
        bordered(3),
        bordered(5),
        normalization="none",
        crop="inscribed",
        output=output,
        header=wcs_header(),
    )
    with fits.open(output) as written:
        header = written[0].header
        assert header["SEIZACLR"] == "RGB"
        assert header["SEIZATRF"] == "LINEAR"
        # The crop moves the reference pixel by its origin.
        assert header["CRPIX2"] == pytest.approx(64.5 - 5)
        assert header["CRVAL1"] == pytest.approx(83.8)
        assert written[0].data.shape == (3, 123, 128)
        np.testing.assert_allclose(np.moveaxis(written[0].data, 0, -1), rgb)


def test_composition_writes_xisf_by_its_extension(tmp_path) -> None:
    output = tmp_path / "hoo.xisf"
    seiza.combine_narrowband(
        mono(0.2), mono(0.3), palette="hoo", normalization="none", output=output
    )
    assert output.read_bytes().startswith(b"XISF0100")


def test_header_needs_an_output() -> None:
    with pytest.raises(ValueError, match="header only applies with output"):
        seiza.combine_rgb(
            mono(0.2), mono(0.4), mono(0.6), normalization="none", header={}
        )


def test_narrowband_matrix_mixes_filters_by_weight(tmp_path) -> None:
    fits = pytest.importorskip("astropy.io.fits")
    output = tmp_path / "matrix.fits"
    rgb = seiza.combine_narrowband_matrix(
        mono(0.2),
        mono(0.6),
        mono(0.4),
        matrix={
            "red": {"sii": 0.5, "ha": 0.5},
            "green": {"ha": 1.0},
            "blue": {"oiii": 1.0},
        },
        normalization="none",
        output=output,
    )
    np.testing.assert_allclose(rgb, [[[0.3, 0.2, 0.6]]])
    with fits.open(output) as written:
        assert written[0].header["SEIZACLR"] == "NARROWBAND-MATRIX"


def test_narrowband_matrix_refuses_unknown_names_and_a_missing_sii() -> None:
    with pytest.raises(ValueError, match="matrix channels"):
        seiza.combine_narrowband_matrix(
            mono(0.2), mono(0.3), matrix={"cyan": {"ha": 1.0}}
        )
    with pytest.raises(ValueError, match="matrix filters"):
        seiza.combine_narrowband_matrix(
            mono(0.2), mono(0.3), matrix={"red": {"hb": 1.0}}
        )
    with pytest.raises(seiza.EngineError, match="SII"):
        seiza.combine_narrowband_matrix(
            mono(0.2),
            mono(0.3),
            matrix={"red": {"sii": 1.0}, "green": {"ha": 1.0}},
            normalization="none",
        )
