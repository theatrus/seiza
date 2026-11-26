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


def test_a_glow_is_followed_ring_by_ring_as_far_as_it_shows():
    rng = np.random.default_rng(23)
    size = (700, 700)
    yy, xx = np.mgrid[0:size[0], 0:size[1]]
    radius = np.hypot(xx - 300.4, yy - 310.2)
    sky = rng.standard_normal(size)
    glow = 5.0 / (1.0 + radius**2 / 400.0)
    # A glow of 5 / (1 + r² / 400) stands over itself 1.55 times further out
    # by 0.3 of the noise to about 57 px.
    assert 50.0 <= ss.glow_reach(sky + glow, 300.4, 310.2, 0.3, 10.0) < 75.0
    # A hole as well...
    assert 50.0 <= ss.glow_reach(sky - glow, 300.4, 310.2, 0.3, 10.0) < 75.0
    # ...but not the noise alone, nor a glow that ends before the rings begin.
    assert ss.glow_reach(sky, 300.4, 310.2, 0.3, 10.0) == 0.0
    assert ss.glow_reach(sky + glow, 300.4, 310.2, 0.3, 90.0) == 0.0
    # Light past a quiet ring is not the star's: a nebula's shell at 150 px.
    shell = 3.0 * ((radius > 150) & (radius < 175))
    assert 50.0 <= ss.glow_reach(sky + glow + shell, 300.4, 310.2, 0.3, 10.0) < 75.0
    # Nor is a nebula beside the star, ten times the noise and as wide as the
    # glow reaches, nor a slope across it: they stand over the rings further
    # out on one side only.
    beside = 10.0 * np.exp(-((xx - 380.0) ** 2 + (yy - 310.0) ** 2) / (2 * 60.0**2))
    assert ss.glow_reach(sky + beside, 300.4, 310.2, 0.3, 10.0) == 0.0
    assert ss.glow_reach(sky + 0.02 * (xx - 300.0), 300.4, 310.2, 0.3, 10.0) == 0.0


def wide_winged_stars(rng, size, fwhm, beta, flux_range, clipped_flux=None, spacing=110, clipped_beta=None):
    """Bright stars of wide Moffat wings on a grid `spacing` px apart from
    (110, 110), and one far brighter at its first place when `clipped_flux`
    is given, its wings of `clipped_beta` if given."""
    stars, light = [], np.zeros(size)
    for y in range(110, size[0] - 100, spacing):
        for x in range(110, size[1] - 100, spacing):
            if clipped_flux is not None and (x, y) == (110, 110):
                continue
            px, py = x + rng.uniform(-0.5, 0.5), y + rng.uniform(-0.5, 0.5)
            flux = 10 ** rng.uniform(*flux_range)
            light += flux * moffat(size, px, py, fwhm, beta)
            stars.append((px, py, 25.0 - 2.5 * np.log10(flux)))
    if clipped_flux is not None:
        light += clipped_flux * moffat(size, 110.3, 110.6, fwhm, clipped_beta or beta)
        stars.append((110.3, 110.6, 25.0 - 2.5 * np.log10(clipped_flux)))
    return stars, light


def test_the_fit_s_estimate_carries_bright_stars_on_past_their_stamps():
    rng = np.random.default_rng(21)
    size = (900, 900)
    # Moffat wings of β 2.5: of a 3 px star's light, 1% lies past the field's
    # 33 px stamps, 10 to 30 times the noise a pixel at their edge.
    stars, light = wide_winged_stars(rng, size, 3.0, 2.5, (5.5, 6.0))
    image = (100.0 + light + rng.standard_normal(size)).astype(np.float32)
    noise = np.ones(size, np.float32)
    truth = ss.clean_sky(image, stars, noise=noise)
    estimate = ss.clean_sky(image, stars, noise=noise, wings=True)
    assert truth["summary"]["stamp_px"] == 33 and truth["summary"]["winged"] == 0
    assert estimate["summary"]["winged"] == len(stars)
    yy, xx = np.mgrid[0:size[0], 0:size[1]]

    def left(sky, low, high, square=None):
        """The median over the stars of the mean left about each, from `low`
        to `high` px, over its flux; within or past its stamp's square."""
        found = []
        for x, y, magnitude in stars:
            near = slice(int(y) - high, int(y) + high + 1), slice(int(x) - high, int(x) + high + 1)
            distance = np.hypot(xx[near] - x, yy[near] - y)
            ring = (distance >= low) & (distance < high)
            if square is not None:
                within = np.maximum(np.abs(xx[near] - round(x)), np.abs(yy[near] - round(y))) <= 16
                ring &= within if square else ~within
            found.append(float(np.mean(sky[near][ring] - 100.0)) / 10 ** (-0.4 * (magnitude - 25.0)))
        return float(np.median(found))

    # A clean sky, a truth, keeps what the fit measured: past the stamps, the
    # wing is left, and only the mask...
    past = left(truth["sky"], 18, 50)
    assert past > 0
    # ...but the estimate takes the wing out, most of it, and leaves no square:
    # just within the stamp's edge and just past it, at the same radii, what
    # is left is alike.
    assert abs(left(estimate["sky"], 18, 50)) < past / 4
    edge_step = left(truth["sky"], 15, 22, True) - left(truth["sky"], 15, 22, False)
    assert abs(left(estimate["sky"], 15, 22, True) - left(estimate["sky"], 15, 22, False)) < abs(edge_step) / 3
    # What the estimate took out more is in its model, nothing else changed.
    assert np.allclose((estimate["sky"] + estimate["model"])[200:700, 200:700],
                       (truth["sky"] + truth["model"])[200:700, 200:700], atol=0.05)
    # Within the stamps' trusted ring (0.6 of their 16 px half side) the two
    # are the same; past the stamps the estimate's model follows the stars'
    # own wings, where the fit's falls to nothing.
    follows, falls = [], []
    for x, y, magnitude in stars:
        flux = 10 ** (-0.4 * (magnitude - 25.0))
        near = slice(int(round(y)) - 30, int(round(y)) + 31), slice(int(round(x)) - 30, int(round(x)) + 31)
        distance = np.hypot(xx[near] - x, yy[near] - y)
        core = distance <= 8.0
        assert np.allclose(estimate["model"][near][core], truth["model"][near][core], rtol=1e-4, atol=1e-3)
        ring = (distance >= 18.0) & (distance < 25.0)
        wing = flux * moffat(size, x, y, 3.0, 2.5)[near][ring].mean()
        follows.append(estimate["model"][near][ring].mean() / wing)
        falls.append(truth["model"][near][ring].mean() / wing)
    assert 0.8 < np.median(follows) < 1.2
    assert np.median(falls) < 0.3



def test_a_star_s_stamp_is_drawn_as_the_fit_draws_it():
    rng = np.random.default_rng(21)
    size = (900, 900)
    stars, light = wide_winged_stars(rng, size, 3.0, 2.5, (5.5, 6.0))
    image = (100.0 + light + rng.standard_normal(size)).astype(np.float32)
    field = ss.measured_field(image, None)
    for x, y, magnitude in stars[10:14]:
        fit = seiza.StarFit.fit(image, field, [(x, y, magnitude)], noise=np.ones(size, np.float32))
        star = fit.stars[0]
        column, row, stamp = ss.fit_stamp(field, star.x, star.y)
        half = stamp.shape[0] // 2
        drawn = fit.model()[row - half:row + half + 1, column - half:column + half + 1]
        np.testing.assert_allclose(star.flux[0] * stamp, drawn, atol=1e-6 * float(drawn.max()))


def test_a_bright_star_s_glow_is_masked_as_far_as_it_shows():
    rng = np.random.default_rng(22)
    size = (900, 900)
    # The clipped star's wings are wide (β 1.4), the others' narrower (β 2),
    # and fainter, under half the noise past their discs.
    stars, light = wide_winged_stars(rng, size, 2.0, 2.0, (4.7, 5.0), clipped_flux=3e6, clipped_beta=1.4)
    clip = 30000.0
    image = np.minimum(100.0 + light + rng.standard_normal(size), clip).astype(np.float32)
    made = ss.clean_sky(image, stars, noise=np.ones(size, np.float32), clip=clip)
    table = made["stars"]
    reasons = table[:, ss.STAR_COLUMNS.index("reasons")].astype(int)
    glow = 1 << ss.REASONS.index("glow")
    index = int(np.argmin(np.hypot(table[:, 0] - 110.3, table[:, 1] - 110.6)))
    assert table[index, ss.STAR_COLUMNS.index("clipped")] == 1.0
    assert reasons[index] & glow
    # Its disc alone, (3 + 2 log10(peak / noise)) FWHM, ends near 30 px; its
    # wing, 8 times the noise at 50 px and half of it at 140 px, shows to
    # 140 px or so.
    for distance in (60, 100):
        assert made["mask"][110, 110 + distance] and made["mask"][110 + distance, 110]
    assert not made["mask"][110 + 200, 110 + 200 - 55]
    # The others glow hardly at all.
    others = [i for i in range(len(table)) if i != index]
    assert sum(bool(reasons[i] & glow) for i in others) <= len(others) // 10
