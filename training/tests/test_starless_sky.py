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
    g, f = ss.photon_terms(sky, model, noise)
    assert g == pytest.approx(0.3, rel=0.15) and f < 0.01
    assert ss.photon_terms(sky, np.zeros(shape), noise) == (0.0, 0.0)


def test_the_shape_s_miss_grows_with_the_light_and_is_not_taken_for_photon_noise():
    rng = np.random.default_rng(12)
    shape = (300, 300)
    model = np.zeros(shape)
    model[20:280, 20:280] = np.geomspace(100, 30000, 260)[None, :]
    noise = np.ones(shape)
    # Photon noise of 0.3 a unit of light, and a miss of 2% of it.
    sky = rng.standard_normal(shape) * np.sqrt(1.0 + 0.3 * model + (0.02 * model) ** 2)
    g, f = ss.photon_terms(sky, model, noise)
    assert g == pytest.approx(0.3, rel=0.3)
    assert f == pytest.approx(0.02, rel=0.1)
    # The miss alone is not read as photon noise.
    g, f = ss.photon_terms(rng.standard_normal(shape) * np.sqrt(1.0 + (0.02 * model) ** 2), model, noise)
    assert g < 0.05 and f == pytest.approx(0.02, rel=0.1)


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


def test_a_dipole_reads_a_star_taken_out_a_fraction_of_a_pixel_off_its_place():
    rng = np.random.default_rng(31)
    noise = np.ones((100, 100))
    plane = rng.standard_normal((100, 100))
    star = moffat((100, 100), 50.3, 49.8, 3.0) * 1000.0
    # Taken out where it is, nothing leans; taken out 0.3 px off, in x or in
    # y, the core leans far past the limit, though its mean barely moves.
    assert ss.core_dipole(plane + star - moffat((100, 100), 50.3, 49.8, 3.0) * 1000.0, noise, 50.3, 49.8, 3.0) < 4.0
    for dx, dy in ((0.3, 0.0), (0.0, 0.3)):
        off = plane + star - moffat((100, 100), 50.3 + dx, 49.8 + dy, 3.0) * 1000.0
        assert ss.core_dipole(off, noise, 50.3, 49.8, 3.0) > 3 * ss.CORE_LIMIT
        assert abs(ss.core_left(off, noise, 50.3, 49.8, 3.0)) < ss.CORE_LIMIT
    # On noise alone it is the length of two unit normals: its square's mean
    # is 2.
    values = [ss.core_dipole(rng.standard_normal((30, 30)), np.ones((30, 30)), 15.0, 15.0, 3.0) for _ in range(400)]
    assert np.mean(np.square(values)) == pytest.approx(2.0, rel=0.2)
    assert ss.core_dipole(plane, noise, 1.0, 50.0, 3.0) is None


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


def test_a_star_held_off_its_place_leaves_a_dipole_the_mask_covers():
    rng = np.random.default_rng(14)
    size = (400, 400)
    noise_level = 2.0
    places, light = [], np.zeros(size)
    for _ in range(60):
        x, y = rng.uniform(30, 370, 2)
        flux = 10 ** rng.uniform(3.0, 3.6)
        light += flux * moffat(size, x, y, 4.0)
        places.append((x, y, 25.0 - 2.5 * np.log10(flux)))
    image = (100.0 + light + rng.standard_normal(size) * noise_level).astype(np.float32)
    # The first ten are listed 0.6 px off and held there, as a catalogue's
    # places are where they are not refined; the rest are refined.
    listed = [(x + 0.6, y, m) if index < 10 else (x, y, m) for index, (x, y, m) in enumerate(places)]
    made = ss.clean_sky(image, listed, noise=np.full(size, noise_level, np.float32),
                        refine=[index >= 10 for index in range(60)])
    # (The table holds the refined stars first, then those held.)
    table = made["stars"]
    dipole = 1 << ss.REASONS.index("dipole")
    held = np.array([int(np.argmin(np.hypot(table[:, 0] - x, table[:, 1] - y))) for x, y, _ in listed[:10]])
    assert np.allclose(table[held, 0], [x for x, _, _ in listed[:10]], atol=1e-3)
    reasons = table[:, ss.STAR_COLUMNS.index("reasons")].astype(int)
    assert all(reasons[held] & dipole)
    # A star fitted well is not taken for a dipole: those the rest lean are
    # blends, poor fits too.
    poor = 1 << ss.REASONS.index("poor fit")
    assert all(reasons[index] & poor for index in range(len(table)) if index not in held and reasons[index] & dipole)
    for x, y, _ in listed[:10]:
        assert made["mask"][int(round(y)), int(round(x))]


def test_a_sparse_field_is_measured_with_one_psf_for_all():
    rng = np.random.default_rng(17)
    size = (300, 300)
    image = 100.0 + rng.standard_normal(size) * 2.0
    for x, y in ((60.3, 70.1), (220.6, 90.4), (150.2, 230.7), (80.9, 200.2), (240.1, 240.5)):
        image += 5000.0 * moffat(size, x, y, 4.0)
    image = image.astype(np.float32)
    with pytest.raises(seiza.TooFewStarsError):
        seiza.PsfField.measure(image, saturation=60000.0)
    field = ss.measured_field(image, 60000.0)
    assert field.shape_at(150.0, 150.0).fwhm == pytest.approx(4.0, rel=0.15)
