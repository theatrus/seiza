"""The starless model's clean skies: stars fitted with a stack's own PSF field
and taken out, the field's miss, the photon noise, and the mask."""

from types import SimpleNamespace

import numpy as np
import pytest

seiza = pytest.importorskip("seiza")

from seiza_training import starless_sky as ss  # noqa: E402


def moffat(size, x, y, fwhm, beta=3.0):
    """A Moffat star of unit flux centred at (x, y) on a size × size grid."""
    alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    yy, xx = np.mgrid[0:size[0], 0:size[1]].astype(np.float64)
    profile = (1.0 + ((xx - x) ** 2 + (yy - y) ** 2) / alpha**2) ** -beta
    return profile / profile.sum()


def test_a_cut_moved_to_a_star_s_centre_and_back_is_the_cut():
    rng = np.random.default_rng(1)
    plane = rng.standard_normal((80, 80))
    star = moffat((80, 80), 40.3, 39.6, 4.0) * 1000.0
    plane += star
    cut = ss.centred(plane, 40.3, 39.6, 10)
    # The star sits at the cut's middle once moved there...
    assert np.unravel_index(np.argmax(cut), cut.shape) == (10, 10)
    alone = ss.centred(star, 40.3, 39.6, 10)
    yy, xx = np.mgrid[0:21, 0:21]
    assert (alone * xx).sum() / alone.sum() == pytest.approx(10.0, abs=0.005)
    assert (alone * yy).sum() / alone.sum() == pytest.approx(10.0, abs=0.005)
    # ...and `placed` is its inverse: back where the plane had it.
    back = ss.placed(cut, 40.3, 39.6)
    original = plane[40 - 10:40 + 11, 40 - 10:40 + 11]
    assert np.abs(back - original)[3:-3, 3:-3].max() < 0.05 * star.max()
    assert ss.centred(plane, 3.0, 40.0, 10) is None


def test_the_detector_s_stars_the_catalogue_lacks_are_kept_with_their_flux():
    columns = ("x", "y", "flux", "peak")
    found = np.array([[10.0, 10.0, 5.0, 1.0], [50.0, 50.0, 7.0, 1.0], [52.0, 51.0, np.nan, 1.0],
                      [90.0, 20.0, -1.0, 1.0]])
    listed = [(11.0, 11.0, 15.0)]
    assert ss.faint_detections(found, columns, listed) == [(50.0, 50.0, 7.0)]


def test_the_magnitude_scale_comes_from_the_stars_measured_well():
    stars = [SimpleNamespace(magnitude=m, flux=[10 ** (-0.4 * (m - 25.0))], flux_error=[1e-6 * 10 ** (-0.4 * (m - 25))],
                             saturated=False) for m in np.linspace(10, 16, 30)]
    slope, zero = ss.magnitude_scale(stars)
    assert slope == pytest.approx(1.0, abs=1e-6) and zero == pytest.approx(-25.0, abs=1e-4)
    assert ss.magnitude_scale(stars[:5]) is None


def test_the_photon_term_is_how_the_leftovers_scatter_grows_with_the_stars_light():
    rng = np.random.default_rng(2)
    shape = (200, 200)
    model = np.zeros(shape)
    model[50:150, 50:150] = np.linspace(100, 1000, 100)[None, :]
    noise = np.ones(shape)
    sky = rng.standard_normal(shape) * np.sqrt(1.0 + 0.3 * model)
    assert ss.photon_term(sky, model, noise) == pytest.approx(0.3, rel=0.1)
    assert ss.photon_term(sky, np.zeros(shape), noise) == 0.0


def test_a_core_left_reads_light_and_holes_in_units_of_the_mean_s_noise():
    rng = np.random.default_rng(3)
    plane = rng.standard_normal((100, 100))
    noise = np.ones((100, 100))
    assert abs(ss.core_left(plane, noise, 50.0, 50.0, 4.0)) < 4.0
    bump = plane + moffat((100, 100), 50.0, 50.0, 4.0) * 400.0
    assert ss.core_left(bump, noise, 50.0, 50.0, 4.0) > 10.0
    hole = plane - moffat((100, 100), 50.0, 50.0, 4.0) * 400.0
    assert ss.core_left(hole, noise, 50.0, 50.0, 4.0) < -10.0
    assert ss.core_left(plane, noise, 2.0, 50.0, 4.0) is None


def test_a_clean_sky_takes_out_the_stars_and_leaves_the_nebula_and_its_noise():
    rng = np.random.default_rng(4)
    size = (640, 640)
    yy, xx = np.mgrid[0:size[0], 0:size[1]].astype(np.float64)
    nebula = 100.0 + 40.0 * np.exp(-((xx - 300) ** 2 + (yy - 340) ** 2) / (2 * 120.0**2))
    noise_level = 2.0
    stars, light = [], np.zeros(size)
    for _ in range(220):
        x, y = rng.uniform(20, size[1] - 20), rng.uniform(20, size[0] - 20)
        flux = 10 ** rng.uniform(2.2, 4.6)
        light += flux * moffat(size, x, y, 4.0)
        stars.append((x, y, 25.0 - 2.5 * np.log10(flux)))
    image = (nebula + light + rng.standard_normal(size) * noise_level).astype(np.float32)
    noise = np.full(size, noise_level, np.float32)
    made = ss.clean_sky(image, stars, noise=noise)
    sky, mask = made["sky"], made["mask"]
    truth = nebula
    inner = (slice(24, -24), slice(24, -24))
    kept = ~mask[inner]
    left = (sky - truth)[inner][kept]
    # What is left where it is kept is the noise, and no star's light.
    assert abs(float(np.mean(left))) < 0.1 * noise_level
    assert float(np.std(left)) < 1.25 * noise_level
    assert made["summary"]["masked_share"] < 0.25
    assert made["stars"].shape == (len(stars), len(ss.STAR_COLUMNS))
    # Every star the fit took out is within a fraction of a pixel of its place.
    table = made["stars"]
    assert np.median(np.hypot(table[:, 0] - np.array([s[0] for s in stars]),
                              table[:, 1] - np.array([s[1] for s in stars]))) < 0.3
