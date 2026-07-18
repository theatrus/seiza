"""End-to-end binding tests using a synthetic sky.

A synthetic catalog is projected through a known WCS to produce image star
positions; solving must recover that WCS. This mirrors the Rust crate's own
synthetic solve tests without needing any catalog data files.
"""

import math
import random

import numpy as np
import pytest

import seiza

CENTER = (150.0, 35.0)
SCALE = 2.5  # arcsec/px
WIDTH, HEIGHT = 1600, 1200


def synthetic_field(rotation_deg=20.0, n_stars=120, seed=7):
    """Return (catalog, star_list, truth_wcs) for a synthetic sky."""
    rng = random.Random(seed)
    truth = seiza.Wcs.from_center_scale_rotation(
        CENTER,
        ((WIDTH - 1) / 2.0, (HEIGHT - 1) / 2.0),
        SCALE,
        rotation_deg,
    )
    catalog_stars = []
    image_stars = []
    for index in range(n_stars):
        x = rng.uniform(30.0, WIDTH - 30.0)
        y = rng.uniform(30.0, HEIGHT - 30.0)
        ra, dec = truth.pixel_to_world(x, y)
        mag = 5.0 + 8.0 * index / n_stars
        catalog_stars.append((ra, dec, mag))
        image_stars.append(seiza.Star(x=x, y=y, flux=10.0 ** (-0.4 * mag) * 1e6))
    # Field stars outside the image keep the catalog honest.
    for _ in range(200):
        catalog_stars.append(
            (
                CENTER[0] + rng.uniform(-3.0, 3.0),
                CENTER[1] + rng.uniform(-3.0, 3.0),
                rng.uniform(9.0, 13.0),
            )
        )
    return seiza.StarCatalog.from_stars(catalog_stars), image_stars, truth


def test_hinted_solve_recovers_center_scale_and_rotation():
    catalog, stars, truth = synthetic_field()
    solution = seiza.solve(
        stars,
        catalog,
        WIDTH,
        HEIGHT,
        ra=CENTER[0] + 0.05,
        dec=CENTER[1] - 0.05,
        scale_arcsec_px=SCALE * 1.05,
    )
    assert solution.matched_stars >= 20
    assert solution.rms_arcsec < 0.5
    ra, dec = solution.center
    assert math.isclose(ra, CENTER[0], abs_tol=1e-3)
    assert math.isclose(dec, CENTER[1], abs_tol=1e-3)
    assert math.isclose(solution.scale_arcsec_px, SCALE, rel_tol=1e-3)
    assert math.isclose(solution.rotation_deg, 20.0, abs_tol=0.1)
    assert not solution.flipped


def test_solve_accepts_plain_tuples_in_any_order():
    catalog, stars, _ = synthetic_field()
    tuples = [(s.x, s.y, s.flux) for s in stars]
    random.Random(3).shuffle(tuples)  # binding must sort brightest-first
    solution = seiza.solve(
        tuples,
        catalog,
        WIDTH,
        HEIGHT,
        ra=CENTER[0],
        dec=CENTER[1],
        scale_arcsec_px=SCALE,
    )
    assert solution.matched_stars >= 20


def test_fits_header_cards_round_trip_through_wcs():
    catalog, stars, _ = synthetic_field()
    solution = seiza.solve(
        stars,
        catalog,
        WIDTH,
        HEIGHT,
        ra=CENTER[0],
        dec=CENTER[1],
        scale_arcsec_px=SCALE,
    )
    cards = solution.fits_header_cards()
    assert cards["CTYPE1"] == "RA---TAN"
    # FITS CRPIX is 1-indexed
    assert math.isclose(cards["CRPIX1"], solution.wcs.crpix[0] + 1.0)
    det = cards["CD1_1"] * cards["CD2_2"] - cards["CD1_2"] * cards["CD2_1"]
    assert math.isclose(abs(det) ** 0.5 * 3600.0, SCALE, rel_tol=1e-3)

    text = solution.fits_header_text()
    assert len(text) % 80 == 0
    assert text.rstrip().endswith("END")
    assert "CRVAL1" in text


def test_detect_finds_synthetic_stars_f32_and_u8():
    rng = np.random.default_rng(11)
    image = rng.normal(100.0, 2.0, size=(400, 600)).astype(np.float32)
    positions = [(100.5, 200.5), (300.25, 150.75), (50.0, 500.0), (350.0, 40.0)]
    yy, xx = np.mgrid[0:400, 0:600]
    for y, x in positions:
        image += 3000.0 * np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / 4.0)).astype(
            np.float32
        )

    stars = seiza.detect(image)
    assert len(stars) >= len(positions)
    for y, x in positions:
        assert any(abs(s.x - x) < 1.0 and abs(s.y - y) < 1.0 for s in stars[:8])

    u8 = np.clip(image / image.max() * 255.0, 0, 255).astype(np.uint8)
    stars_u8 = seiza.detect(u8)
    assert len(stars_u8) >= len(positions)


def test_wcs_transform_round_trip():
    wcs = seiza.Wcs.from_center_scale_rotation((10.0, -45.0), (500.0, 400.0), 1.5, 33.0)
    ra, dec = wcs.pixel_to_world(123.0, 456.0)
    x, y = wcs.world_to_pixel(ra, dec)
    assert math.isclose(x, 123.0, abs_tol=1e-6)
    assert math.isclose(y, 456.0, abs_tol=1e-6)
    assert len(wcs.footprint(1000, 800)) == 4


def test_solve_failure_raises():
    catalog = seiza.StarCatalog.from_stars([(10.0, 10.0, 5.0)])
    with pytest.raises(RuntimeError):
        seiza.solve(
            [(10.0, 10.0, 100.0), (20.0, 20.0, 90.0)],
            catalog,
            WIDTH,
            HEIGHT,
            ra=10.0,
            dec=10.0,
            scale_arcsec_px=2.0,
        )


def test_star_catalog_cone_search_and_metadata():
    catalog = seiza.StarCatalog.from_stars(
        [(10.0, 10.0, 5.0), (10.1, 10.1, 6.0), (200.0, -30.0, 4.0)]
    )
    assert catalog.star_count() == 3
    nearby = catalog.cone_search(10.0, 10.0, 1.0)
    assert len(nearby) == 2
    assert nearby[0][2] == pytest.approx(5.0)


def test_sip_fit_on_distorted_field_and_header_output():
    rng = random.Random(21)
    truth = seiza.Wcs.from_center_scale_rotation(
        CENTER, ((WIDTH - 1) / 2.0, (HEIGHT - 1) / 2.0), SCALE, 10.0
    )

    def distort(x, y):
        dx, dy = x - WIDTH / 2.0, y - HEIGHT / 2.0
        factor = 1.0 + 2e-9 * (dx * dx + dy * dy)
        return WIDTH / 2.0 + dx * factor, HEIGHT / 2.0 + dy * factor

    catalog_stars = []
    image_stars = []
    for index in range(400):
        x = rng.uniform(20.0, WIDTH - 20.0)
        y = rng.uniform(20.0, HEIGHT - 20.0)
        ra, dec = truth.pixel_to_world(x, y)
        mag = 5.0 + 8.0 * index / 400
        catalog_stars.append((ra, dec, mag))
        xd, yd = distort(x, y)
        image_stars.append((xd, yd, 10.0 ** (-0.4 * mag) * 1e6))
    catalog = seiza.StarCatalog.from_stars(catalog_stars)

    kwargs = dict(ra=CENTER[0], dec=CENTER[1], scale_arcsec_px=SCALE)
    linear = seiza.solve(image_stars, catalog, WIDTH, HEIGHT, **kwargs)
    assert linear.wcs.sip_order is None
    solution = seiza.solve(image_stars, catalog, WIDTH, HEIGHT, sip_order=3, **kwargs)
    assert solution.wcs.sip_order is not None and solution.wcs.sip_order >= 2
    assert solution.rms_arcsec < linear.rms_arcsec

    coefficients = solution.wcs.sip_coefficients()
    assert set(coefficients) == {"A", "B", "AP", "BP"}
    assert (2, 0) in coefficients["A"]

    cards = solution.fits_header_cards()
    assert cards["CTYPE1"] == "RA---TAN-SIP"
    assert cards["A_ORDER"] == solution.wcs.sip_order
    assert "A_2_0" in cards and "BP_0_0" in cards

    text = solution.fits_header_text()
    assert len(text) % 80 == 0
    assert "A_ORDER" in text and "AP_0_0" in text

    # Distorted pixels land on true sky positions and round-trip through
    # the inverse polynomials.
    x, y = 200.0, 900.0
    ra_t, dec_t = truth.pixel_to_world(x, y)
    xd, yd = distort(x, y)
    ra_s, dec_s = solution.wcs.pixel_to_world(xd, yd)
    assert abs(ra_s - ra_t) * 3600.0 < 2.0
    assert abs(dec_s - dec_t) * 3600.0 < 2.0
    xi, yi = solution.wcs.world_to_pixel(ra_t, dec_t)
    assert math.hypot(xi - xd, yi - yd) < 1.0
