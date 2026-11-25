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


def test_multi_frame_tilt_combines_the_frames():
    frames = [
        seiza.detect_measured_stars(
            synthetic_field(soft_top_left=True), psf_type="gaussian"
        )
        for _ in range(3)
    ]
    tilt = seiza.multi_frame_tilt_analysis(frames, triangle_angle_degrees=-30)

    assert (tilt.width, tilt.height) == (frames[0].width, frames[0].height)
    assert len(tilt.frames) == 3
    assert tilt.frames[0].star_count == len(frames[0].stars)
    assert tilt.frames[0].summary.worst_corner == "top-left"
    assert len(tilt.cells) == 9
    top_left = tilt.cells[0]
    assert (top_left.row, top_left.col) == (0, 0)
    assert top_left.star_count == sum(
        cell.star_count
        for frame in frames
        for cell in seiza.tilt_analysis(frame)[0]
        if (cell.row, cell.col) == (0, 0)
    )
    assert top_left.relative_hfr.frame_count == 3
    assert top_left.relative_hfr.median > 1.0
    assert (
        top_left.relative_hfr.lower_quartile
        <= top_left.relative_hfr.median
        <= top_left.relative_hfr.upper_quartile
    )
    assert top_left.median_hfr == pytest.approx(
        top_left.relative_hfr.median * tilt.median_frame_hfr
    )
    assert tilt.summary.worst_corner == "top-left"
    assert tilt.worst_corner_frame_count == 3
    assert tilt.frame_tilt_percent.frame_count == 3
    assert tilt.triangle.summary.angle_degrees == pytest.approx(330.0)
    assert len(tilt.triangle.sector_relative_hfr) == 3
    assert tilt.frames[0].triangle is not None

    # The soft top-left corner is a plane softening toward the top left.
    plane = tilt.focal_plane
    assert 270.0 <= plane.tilt_angle_degrees < 360.0
    assert plane.tilt_percent > 0
    assert 0.0 <= plane.tilt_r_squared <= plane.r_squared <= 1.0
    assert plane.radial_percent == pytest.approx(100 * plane.radial)
    assert tilt.cells[4].elongation_from_radial_degrees is None
    for cell in tilt.cells:
        angle = cell.elongation_from_radial_degrees
        assert angle is None or 0.0 <= angle <= 90.0


def test_multi_frame_tilt_refuses_mixed_sensors_and_no_frames():
    small = seiza.detect_measured_stars(
        synthetic_field(height=200, width=300), psf_type="none"
    )
    large = seiza.detect_measured_stars(synthetic_field(), psf_type="none")
    with pytest.raises(ValueError, match="frame 1 is 320x256"):
        seiza.multi_frame_tilt_analysis([small, large])
    with pytest.raises(ValueError, match="at least one frame"):
        seiza.multi_frame_tilt_analysis([])
    assert seiza.multi_frame_tilt_analysis([large]).triangle is None


def test_a_bad_knob_is_an_error_not_a_default():
    with pytest.raises(ValueError, match="psf_type"):
        seiza.detect_measured_stars(synthetic_field(), psf_type="parabolic")


def test_a_target_star_count_is_accepted():
    image = synthetic_field()
    once = seiza.detect_measured_stars(image)
    retried = seiza.detect_measured_stars(image, target_star_count=10_000)
    # No pass can reach the target, so the best one comes back.
    assert len(retried.stars) >= len(once.stars)


def test_a_fits_file_measures_like_its_array(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    image = synthetic_field()
    path = tmp_path / "light.fits"
    header = fits.Header()
    header["FOCALLEN"] = 2000.0
    header["XPIXSZ"] = 3.76
    fits.writeto(path, image, header)

    from_file = seiza.detect_measured_stars_file(path, psf_type="gaussian")
    from_array = seiza.detect_measured_stars(
        image, psf_type="gaussian", focal_length_mm=2000.0, pixel_size_um=3.76
    )
    assert (from_file.width, from_file.height) == (image.shape[1], image.shape[0])
    assert len(from_file.stars) == len(from_array.stars)
    assert from_file.average_hfr == pytest.approx(from_array.average_hfr)

    raster = tmp_path / "frame.png"
    raster.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        seiza.detect_measured_stars_file(raster)


def test_pixel_scale_advice():
    assert seiza.pixel_scale_arcsec(518.0, 3.76) == pytest.approx(1.497, abs=0.01)
    assert seiza.pixel_scale_arcsec(0.0, 3.76) is None
    assert seiza.classify_pixel_scale(1.5) == "widefield"
    assert seiza.classify_pixel_scale(1.0) == "standard"
    assert seiza.classify_pixel_scale(0.36) == "longfocal"
    assert seiza.recommend_detection_binning(6.2) == 2
    assert seiza.recommend_detection_binning(-1.0) == 1
