"""Gaia photometric colour calibration through the Python bindings."""

import numpy as np
import pytest

import seiza


def synthetic_field():
    """A camera recording a solar-coloured star at R/G 0.6 and B/G 1.4."""
    width, height = 1200, 1000
    wcs = seiza.Wcs.from_center_scale_rotation((56.75, 24.1), (600.0, 500.0), 3.0, 10.0)
    rng = np.random.default_rng(3)
    image = np.empty((height, width, 3), np.float32)
    for channel, sky in enumerate((120.0, 100.0, 90.0)):
        image[..., channel] = sky + rng.normal(0.0, 1.0, (height, width))
    yy, xx = np.mgrid[:21, :21]
    gaia = []
    for index in range(140):
        x = 30.0 + (index * 7919) % 1140 + (index % 7) * 0.13
        y = 30.0 + (index * 6271) % 940 + (index % 5) * 0.17
        bp_rp = 0.1 + ((index * 37) % 160) / 100.0
        g = 9.0 + ((index * 13) % 40) / 10.0
        ra, dec = wcs.pixel_to_world(x, y)
        gaia.append({"ra": ra, "dec": dec, "g": g, "bp": g + 0.3, "rp": g + 0.3 - bp_rp, "ruwe": 1.0})
        flux = 2.0e5 * 10 ** (-0.4 * (g - 9.0))
        delta = bp_rp - 0.82
        ratios = (0.6 * 10 ** (0.2 * delta), 1.0, 1.4 * 10 ** (-0.32 * delta))
        x0, y0 = int(x) - 10, int(y) - 10
        profile = np.exp(-((xx + x0 - x) ** 2 + (yy + y0 - y) ** 2) / 4.5) / (np.pi * 4.5)
        ys, xs = slice(max(y0, 0), min(y0 + 21, height)), slice(max(x0, 0), min(x0 + 21, width))
        crop = profile[ys.start - y0 : ys.stop - y0, xs.start - x0 : xs.stop - x0]
        for channel, ratio in enumerate(ratios):
            image[ys, xs, channel] += (flux * ratio * crop).astype(np.float32)
    return image, wcs, gaia


def test_calibrate_color_recovers_the_camera_response():
    image, wcs, gaia = synthetic_field()
    calibration = seiza.calibrate_color(image, wcs, gaia)
    r, g, b = calibration.gains
    assert g == 1.0
    assert abs(r - 1 / 0.6) < 0.03
    assert abs(b - 1 / 1.4) < 0.02
    assert calibration.red_fit["stars"] >= 20
    calibrated = calibration.apply(image)
    assert calibrated.shape == image.shape
    np.testing.assert_allclose(
        calibrated[0, 0], image[0, 0] * np.array(calibration.gains) + np.array(calibration.offsets),
        rtol=1e-5,
    )


def test_wcs_reads_back_from_its_own_header_cards():
    wcs = seiza.Wcs.from_center_scale_rotation((56.75, 24.1), (600.0, 500.0), 3.0, 10.0)
    again = seiza.Wcs.from_header(wcs.fits_header_cards())
    assert again is not None
    assert again.pixel_to_world(10.0, 20.0) == pytest.approx(wcs.pixel_to_world(10.0, 20.0))
    assert seiza.Wcs.from_header({"CTYPE1": "RA---SIN"}) is None


def test_calibrate_color_refuses_a_mono_image_and_too_few_stars():
    image, wcs, gaia = synthetic_field()
    with pytest.raises(ValueError):
        seiza.calibrate_color(image[..., 0].copy(), wcs, gaia)
    with pytest.raises(ValueError):
        seiza.calibrate_color(image, wcs, gaia[:5])


def test_calibrate_color_takes_a_bp_rp_colour_as_the_offline_catalog_gives():
    image, wcs, gaia = synthetic_field()
    offline = [
        {"ra": row["ra"], "dec": row["dec"], "pmra": None, "pmdec": None,
         "g": row["g"], "bp_rp": row["bp"] - row["rp"]}
        for row in gaia
    ]
    online = seiza.calibrate_color(image, wcs, gaia).gains
    assert seiza.calibrate_color(image, wcs, offline).gains == pytest.approx(online)


def test_calibrate_color_reports_its_stars_and_takes_the_fit_options():
    image, wcs, gaia = synthetic_field()
    calibration = seiza.calibrate_color(image, wcs, gaia)
    measurements = calibration.measurements
    assert len(measurements) == calibration.stars_measured
    star = measurements[0]
    assert set(star) == {"x", "y", "g", "bp_rp", "flux", "used"}
    assert len(star["flux"]) == 3 and len(star["used"]) == 2

    with pytest.raises(ValueError):
        seiza.calibrate_color(image, wcs, gaia, minimum_stars=500)
    narrow = seiza.calibrate_color(
        image,
        wcs,
        gaia,
        fit_bp_rp_range=(0.3, 1.4),
        rejection_sigma=2.5,
        minimum_stars=5,
        calibrator_max_g=10.0,
    )
    assert 0 < len(narrow.measurements) < len(measurements)
    assert all(star["g"] <= 10.0 for star in narrow.measurements)
    for star in narrow.measurements:
        if not 0.3 - 1e-6 <= star["bp_rp"] <= 1.4 + 1e-6:
            assert star["used"] == (False, False)


def test_color_calibration_apply_refuses_a_mono_image():
    image, wcs, gaia = synthetic_field()
    calibration = seiza.calibrate_color(image, wcs, gaia)
    with pytest.raises(ValueError):
        calibration.apply(image[..., 0].copy())


def test_gaia_photometry_catalog_cone_reports_a_missing_catalog(tmp_path):
    with pytest.raises(ValueError):
        seiza.gaia_photometry_catalog_cone(tmp_path / "absent.bin", 56.75, 24.1, 1.0)


AUTO_STRETCH = {
    "model": {"type": "auto-mtf", "target_median": 0.25, "shadows_clip": -2.8},
    "color_strategy": "linked",
    "max_analysis_samples": 200_000,
}


def turned_and_cropped(wcs):
    """A solved frame, and the mapping that turns it half round and crops it
    at (40, 60)."""
    to_field = ((-1.0, 0.0), (0.0, -1.0))
    frame = wcs.transformed(to_field, 1199.0, 999.0)
    return frame, (to_field, 1199.0 - 40.0, 999.0 - 60.0)


def test_wcs_transformed_carries_a_solution_through_a_turn_and_a_crop():
    wcs = seiza.Wcs.from_center_scale_rotation((56.75, 24.1), (600.0, 500.0), 3.0, 10.0)
    frame, (matrix, tx, ty) = turned_and_cropped(wcs)
    field = frame.transformed(matrix, tx, ty)
    ra, dec = frame.pixel_to_world(100.0, 200.0)
    x, y = field.world_to_pixel(ra, dec)
    assert (x, y) == pytest.approx((1199.0 - 100.0 - 40.0, 999.0 - 200.0 - 60.0))
    with pytest.raises(ValueError):
        wcs.transformed(((0.0, 0.0), (0.0, 0.0)))


def test_calibrate_and_stretch_recovers_the_gains_and_stretches():
    image, wcs, gaia = synthetic_field()
    result = seiza.calibrate_and_stretch(image, wcs, stretch=AUTO_STRETCH, gaia=gaia)
    outcome = result.calibration
    assert outcome.status == "calibrated" and outcome.reason is None
    r, g, b = outcome.calibration.gains
    assert abs(r - 1 / 0.6) < 0.03 and abs(b - 1 / 1.4) < 0.03
    assert result.image.shape == image.shape
    assert 0.0 <= result.image.min() and result.image.max() <= 1.0
    assert len(result.stretch_plans) == 1
    assert result.linked_stages == []

    # A per-channel stage is linked once calibration has run.
    relinked = seiza.calibrate_and_stretch(
        image, wcs, stretch={**AUTO_STRETCH, "color_strategy": "unlinked"}, gaia=gaia
    )
    assert relinked.linked_stages == [0]
    np.testing.assert_array_equal(relinked.image, result.image)

    calibrated, alone = seiza.calibrate_color_image(image, wcs, gaia=gaia)
    assert alone.status == "calibrated"
    assert alone.calibration.gains == pytest.approx(outcome.calibration.gains)
    assert not np.array_equal(calibrated, image)


def test_calibrate_and_stretch_still_stretches_when_it_cannot_calibrate(tmp_path):
    image, wcs, gaia = synthetic_field()
    plain = seiza.calibrate_and_stretch(image, None, stretch=[AUTO_STRETCH], gaia=gaia)
    assert plain.calibration.status == "skipped"
    assert plain.calibration.reason == "no_solution"
    assert plain.calibration.calibration is None
    assert 0.0 <= plain.image.min() and plain.image.max() <= 1.0

    for kwargs, reason in [
        ({"catalog": tmp_path}, "no_catalog"),
        ({"gaia": []}, "no_stars"),
        ({"gaia": gaia[:6]}, "too_few_stars"),
    ]:
        unchanged, outcome = seiza.calibrate_color_image(image, wcs, **kwargs)
        assert outcome.reason == reason, outcome.detail
        np.testing.assert_array_equal(unchanged, image)
    with pytest.raises(ValueError):
        seiza.calibrate_and_stretch(image, wcs, stretch={"model": "nonsense"}, gaia=gaia)
