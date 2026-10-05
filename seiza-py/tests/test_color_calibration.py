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


def test_gaia_photometry_catalog_cone_reports_a_missing_catalog(tmp_path):
    with pytest.raises(ValueError):
        seiza.gaia_photometry_catalog_cone(tmp_path / "absent.bin", 56.75, 24.1, 1.0)
