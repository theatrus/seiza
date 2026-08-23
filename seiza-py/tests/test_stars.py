"""The measurement detector and tilt analysis through the wheel API."""

import numpy as np
import pytest

import seiza


def synthetic_field(height=256, width=320, soft_top_left=False):
    rng = np.random.default_rng(11)
    image = rng.normal(600.0, 6.0, size=(height, width))
    yy, xx = np.mgrid[:height, :width]
    for row in range(3):
        for col in range(3):
            for index in range(4):
                y = (row + 0.5) * height / 3 + (index - 1.5) * 11.0
                x = (col + 0.5) * width / 3 + (index - 1.5) * 13.0
                # Soft enough to measure, sharp enough that the
                # synthetic-gaussian-wary validator still accepts it.
                sigma = 2.1 if soft_top_left and row == 0 and col == 0 else 1.6
                image += 9000.0 * np.exp(
                    -((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2)
                )
    return np.clip(image, 0, 65535).astype(np.uint16)


def test_detects_and_measures_stars():
    result = seiza.detect_measured_stars(synthetic_field(), psf_type="gaussian")
    assert len(result.stars) >= 9
    assert result.average_hfr > 0
    star = result.stars[0]
    assert star.hfr > 0 and star.fwhm > 0
    assert star.eccentricity is not None, "PSF was fitted"
    assert 0.0 <= star.theta < np.pi, "major-axis orientation is normalized"


def test_tilt_analysis_sees_the_soft_corner():
    result = seiza.detect_measured_stars(
        synthetic_field(soft_top_left=True), psf_type="gaussian"
    )
    cells, summary = seiza.tilt_analysis(result)
    assert len(cells) == 9
    top_left = next(c for c in cells if c.row == 0 and c.col == 0)
    center = next(c for c in cells if c.row == 1 and c.col == 1)
    assert top_left.median_hfr > center.median_hfr
    assert summary.worst_corner == "top-left"
    assert summary.tilt_percent > 0


def test_triangle_tilt_analysis_exposes_native_geometry_and_confidence():
    result = seiza.detect_measured_stars(synthetic_field(), psf_type="gaussian")
    summary = seiza.triangle_tilt_analysis(result, angle_degrees=-30)

    assert summary.angle_degrees == pytest.approx(330.0)
    assert summary.inner_radius_pixels == pytest.approx(
        0.25 * np.hypot(result.width / 2, result.height / 2)
    )
    assert summary.outer_radius_pixels == pytest.approx(
        0.5 * min(result.width, result.height)
    )
    assert summary.minimum_stars_per_region == 3
    assert [sector.sector for sector in summary.sectors] == [1, 2, 3]
    assert [sector.axis_angle_degrees for sector in summary.sectors] == pytest.approx(
        [330.0, 90.0, 210.0]
    )
    for sector in summary.sectors:
        assert (sector.median_hfr is not None) == (sector.star_count > 0)

    expected_ready = all(
        sector.star_count >= summary.minimum_stars_per_region
        for sector in summary.sectors
    )
    assert summary.ready == expected_ready
    assert (summary.tilt_percent is not None) == expected_ready
    assert (summary.best_sector is not None) == expected_ready
    assert (summary.worst_sector is not None) == expected_ready


def test_triangle_tilt_analysis_rejects_a_non_finite_angle():
    result = seiza.detect_measured_stars(synthetic_field(), psf_type="none")
    with pytest.raises(ValueError, match="triangle angle must be finite"):
        seiza.triangle_tilt_analysis(result, angle_degrees=float("inf"))


def test_a_bad_knob_is_an_error_not_a_default():
    with pytest.raises(ValueError, match="psf_type"):
        seiza.detect_measured_stars(synthetic_field(), psf_type="parabolic")
