"""Cross-validate the emitted SIP header against astropy/wcslib.

wcslib is the same library Siril and most FITS consumers parse WCS with, so
agreement here is the definition of "our header is valid": an independent
implementation must reproduce our transforms from the keywords alone.

Skipped when astropy is not installed.
"""

import math
import random

import numpy as np
import pytest

import seiza

fits = pytest.importorskip("astropy.io.fits")
astropy_wcs = pytest.importorskip("astropy.wcs")

CENTER, SCALE, W, H = (150.0, 35.0), 2.5, 1600, 1200


def distort(x, y):
    dx, dy = x - W / 2, y - H / 2
    factor = 1.0 + 2e-9 * (dx * dx + dy * dy)
    return W / 2 + dx * factor, H / 2 + dy * factor


def solve_distorted_field():
    truth = seiza.Wcs.from_center_scale_rotation(
        CENTER, ((W - 1) / 2, (H - 1) / 2), SCALE, 10.0
    )
    rng = random.Random(21)
    catalog_stars, image_stars = [], []
    for index in range(400):
        x, y = rng.uniform(20, W - 20), rng.uniform(20, H - 20)
        ra, dec = truth.pixel_to_world(x, y)
        mag = 5.0 + 8.0 * index / 400
        catalog_stars.append((ra, dec, mag))
        xd, yd = distort(x, y)
        image_stars.append((xd, yd, 10 ** (-0.4 * mag) * 1e6))
    solution = seiza.solve(
        image_stars,
        seiza.StarCatalog.from_stars(catalog_stars),
        W,
        H,
        ra=CENTER[0],
        dec=CENTER[1],
        scale_arcsec_px=SCALE,
        sip_order=3,
    )
    assert solution.wcs.sip_order == 3
    return truth, solution


def wcslib_from_cards(cards):
    header = fits.Header()
    for key, value in cards.items():
        header[key] = value
    header["NAXIS"], header["NAXIS1"], header["NAXIS2"] = 2, W, H
    return astropy_wcs.WCS(header)


def test_emitted_sip_header_is_wcslib_valid_and_carries_distortion():
    truth, solution = solve_distorted_field()
    w = wcslib_from_cards(solution.fits_header_cards())
    assert w.sip is not None, "astropy did not recognize the SIP keywords"
    grid = [(x, y) for x in np.linspace(5, W - 5, 9) for y in np.linspace(5, H - 5, 7)]

    # wcslib must reproduce our forward transform from the keywords alone.
    for x, y in grid:
        ra_s, dec_s = solution.wcs.pixel_to_world(x, y)
        ra_a, dec_a = w.all_pix2world([[x, y]], 0)[0]
        separation = (
            math.hypot((ra_s - ra_a) * math.cos(math.radians(dec_s)), dec_s - dec_a)
            * 3600.0
        )
        assert separation < 1e-6, f"({x}, {y}) disagrees by {separation}\""

    # The SIP terms must carry real correction relative to the linear part —
    # the synthetic distortion reaches ~1.9 px inside this grid.
    linear = fits.Header()
    for key, value in solution.fits_header_cards().items():
        if not key.startswith(("A_", "B_", "AP_", "BP_")):
            linear[key] = value.replace("-SIP", "") if isinstance(value, str) else value
    wl = astropy_wcs.WCS(linear)
    carried = max(
        math.hypot(*(wl.all_world2pix(w.all_pix2world([[x, y]], 0), 0)[0] - (x, y)))
        for x, y in grid
    )
    assert carried > 1.5

    # Reading our header, wcslib must land distorted pixels on true sky.
    for x, y in [(100, 100), (1500, 1100), (800, 100), (100, 1100)]:
        ra_t, dec_t = truth.pixel_to_world(x, y)
        ra_a, dec_a = w.all_pix2world([list(distort(x, y))], 0)[0]
        separation = (
            math.hypot((ra_a - ra_t) * math.cos(math.radians(dec_t)), dec_a - dec_t)
            * 3600.0
        )
        assert separation < 0.5, f"({x}, {y}) off truth by {separation}\""

    # Our AP/BP inverse must agree with wcslib's exact iterative inverse.
    for x, y in grid:
        sky = w.all_pix2world([[x, y]], 0)
        xi, yi = w.all_world2pix(sky, 0, tolerance=1e-8)[0]
        xa, ya = solution.wcs.world_to_pixel(sky[0][0], sky[0][1])
        assert math.hypot(xa - xi, ya - yi) < 0.1
