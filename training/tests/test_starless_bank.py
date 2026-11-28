"""The starless model's known-truth composites: stars put back into a clean
sky, their truth, and the catalogue fit of them."""

import math

import numpy as np
import pytest

from seiza_training import starless_bank as bank
from seiza_training import starless_sky as ss
from seiza_training import starless_stars as stars


def clean_sky_case(rng, size=(400, 400), count=150, faintest=19.0):
    """A smooth sky with noise, and a clean-sky star table as the pass
    leaves one: magnitudes, and fluxes on a scale of 1 and zero 25."""
    yy, xx = np.mgrid[0:size[0], 0:size[1]].astype(np.float64)
    sky = 100.0 + 20.0 * np.exp(-((xx - 200) ** 2 + (yy - 180) ** 2) / (2 * 90.0**2))
    noise_level = 2.0
    sky = (sky + rng.standard_normal(size) * noise_level).astype(np.float32)
    magnitudes = rng.uniform(13.0, faintest, count)
    flux = 10 ** (-0.4 * (magnitudes - 25.0))
    table = np.zeros((count, len(ss.STAR_COLUMNS)))
    col = {name: index for index, name in enumerate(ss.STAR_COLUMNS)}
    table[:, col["x"]], table[:, col["y"]] = rng.uniform(0, size[1], count), rng.uniform(0, size[0], count)
    table[:, col["magnitude"]], table[:, col["flux"]] = magnitudes, flux
    table[:, col["flux_error"]] = flux / 1000.0
    summary = {"fwhm_px": 4.0, "photon_term": 0.5, "clip_level": 30000.0}
    return sky, table, summary, np.full(size, noise_level, np.float32)


def test_a_composite_s_layers_add_back_and_its_stars_are_the_stack_s_kind(monkeypatch):
    # Without galaxies, the starless truth is the clean sky.
    monkeypatch.setattr(bank, "GALAXY_DENSITY", (0.0, 0.0))
    rng = np.random.default_rng(5)
    sky, table, summary, noise = clean_sky_case(rng)
    telescope = bank.Telescope(stamp=stars.moffat_stamp(bank.WIDE_STAMP, 4.0, 3.0), fwhm=4.0)
    made = bank.composite(sky, table, summary, telescope, rng, noise)
    # The truths add back to the input, and the starless one is the sky.
    np.testing.assert_allclose(made["starless"] + made["stars"], made["input"], atol=1e-3)
    np.testing.assert_array_equal(made["starless"], sky)
    count = len(made["placed"])
    assert bank.DENSITY[0] * len(table) - 1 <= count <= bank.DENSITY[1] * len(table) + 1
    # The stars' light is what was put back, at the stack's magnitude scale.
    put = made["placed"][:, 3]
    np.testing.assert_allclose(put, 10 ** (-0.4 * (made["placed"][:, 2] - 25.0)), rtol=1e-9)
    # All of a star far from the edges is drawn; the light is the stars'
    # (less what falls off the edges, plus the few halos).
    star = int(np.argmin(np.abs(made["placed"][:, 0] - 200) + np.abs(made["placed"][:, 1] - 200)))
    lone = bank.Telescope(stamp=stars.moffat_stamp(bank.WIDE_STAMP, 4.0, 3.0), fwhm=4.0)
    alone = np.zeros_like(made["light"], dtype=np.float64)
    x, y, _, flux = made["placed"][star]
    stamp = lone.stamp_for(x, y, flux, float(np.median(noise)))
    bank.add_star(alone, stamp, x, y, flux)
    # (the shift by a fraction of a pixel loses a sliver of the stamp's edge)
    assert alone.sum() == pytest.approx(flux * stamp.sum(), rel=1e-3)
    assert 0.8 * put.sum() < made["light"].sum() < 1.2 * put.sum()
    # Nothing over the clip level, and the stars' photon noise where they
    # are bright.
    assert np.nanmax(made["input"]) <= summary["clip_level"]
    bright = made["light"] > 500.0
    shot = (made["stars"] - made["light"])[bright]
    unclipped = made["input"][bright] < summary["clip_level"]
    ratio = np.var(shot[unclipped]) / np.mean(summary["photon_term"] * made["light"][bright][unclipped])
    assert ratio == pytest.approx(1.0, rel=0.25)


def one_bright_star_case(rng):
    """Faint stars and one far brighter, so its light stands alone."""
    sky, table, summary, noise = clean_sky_case(rng, size=(500, 500), count=300, faintest=19.0)
    col = {name: index for index, name in enumerate(ss.STAR_COLUMNS)}
    table[:, col["magnitude"]] = np.clip(table[:, col["magnitude"]], 17.0, None)
    # Several bright entries, so a draw of the table always takes one.
    table[:5, col["magnitude"]] = 9.0
    table[:, col["flux"]] = 10 ** (-0.4 * (table[:, col["magnitude"]] - 25.0))
    table[:, col["flux_error"]] = table[:, col["flux"]] / 1000.0
    return sky, table, summary, noise


def annulus(made, star, inner, outer):
    x, y = star[0], star[1]
    yy, xx = np.mgrid[0:made["light"].shape[0], 0:made["light"].shape[1]]
    distance = np.hypot(xx - x, yy - y)
    return float(made["light"][(distance >= inner) & (distance < outer)].mean())


def redrawn(made, telescope, noise, *, without=()):
    """The composite's stars drawn again from its own list, with their
    plain profiles alone (no halos), each on the stamp `composite` picks,
    leaving out those `without` lists."""
    light = np.zeros(made["light"].shape, dtype=np.float64)
    floor = float(np.nanmedian(noise))
    for index, (x, y, _, flux) in enumerate(made["placed"]):
        if index not in without:
            bank.add_star(light, telescope.stamp_for(x, y, flux, floor), x, y, flux)
    return light


def test_a_sharp_star_moved_within_its_pixel_leaves_no_cross_and_keeps_its_light():
    # A core narrower than its sampling, as a well-focused pupil's: moved by
    # Fourier transform it rings along its row and column far past its
    # light; moved as the fit draws it, by Lanczos-3, it does not.
    stamp = stars.moffat_stamp(101, 1.2, 3.0)
    light = np.zeros((201, 201))
    bank.add_star(light, stamp, 100.3, 99.6, 1e6)
    assert light.sum() == pytest.approx(1e6 * stamp.sum(), rel=1e-6)
    fourier = np.zeros((201, 201))
    fourier[50:151, 50:151] = 1e6 * ss.placed(stamp, 100.3, 99.6)
    yy, xx = np.mgrid[0:201, 0:201]
    distance = np.hypot(xx - 100.3, yy - 99.6)
    along = ((np.abs(yy - 100) <= 1) | (np.abs(xx - 100) <= 1)) & (distance > 20) & (distance < 45)
    off = ~along & (distance > 20) & (distance < 45)

    def cross(plane):
        """The most light along the row and column 20-45 px out, over the
        most off them at those distances (the Moffat's own wing)."""
        return np.abs(plane[along]).max() / np.abs(plane[off]).max()

    assert cross(fourier) > 5.0
    assert cross(light) < 1.5


def test_the_brightest_stars_carry_halos_wider_than_their_stamps():
    plain = stars.moffat_stamp(bank.WIDE_STAMP, 4.0, 3.0)
    telescope = bank.Telescope(stamp=plain, fwhm=4.0)
    sky, table, summary, noise = one_bright_star_case(np.random.default_rng(6))
    made = bank.composite(sky, table, summary, telescope, np.random.default_rng(16), noise)
    halos = made["light"] - redrawn(made, telescope, noise)
    bright = made["placed"][np.argmin(made["placed"][:, 2])]
    share = halos.sum() / made["placed"][made["placed"][:, 2] <= bright[2] + 0.5, 3].sum()
    # The halos hold their share of the bright stars' light...
    assert bank.HALO_LIGHT[0] * 0.8 < share < (bank.HALO_LIGHT[1] + bank.RING_LIGHT[1]) * 1.2
    # ...far past the core (30 to 40 px, 8 to 10 FWHM), where the star's own
    # profile has next to none.
    own = bright[3] * float(np.mean([plain[bank.WIDE_STAMP // 2, bank.WIDE_STAMP // 2 + r] for r in range(30, 40)]))
    assert annulus({"light": halos}, bright, 30, 40) > 20 * own


def test_a_bright_star_is_drawn_as_far_as_its_light_reaches(monkeypatch):
    monkeypatch.setattr(bank, "HALO_STARS", 0.0)
    plain = stars.moffat_stamp(bank.WIDE_STAMP, 4.0, 1.5)
    telescope = bank.Telescope(stamp=plain, fwhm=4.0)
    sky, table, summary, noise = one_bright_star_case(np.random.default_rng(9))
    made = bank.composite(sky, table, summary, telescope, np.random.default_rng(19), noise)
    np.testing.assert_allclose(made["light"], redrawn(made, telescope, noise), atol=1e-2)
    bright = made["placed"][int(np.argmin(made["placed"][:, 2]))]
    others = redrawn(made, telescope, noise, without={index for index in range(len(made["placed"]))
                                                      if made["placed"][index, 2] <= bright[2] + 0.5})
    # Its wings reach 20 to 26 px out (5 to 6.5 FWHM) as the profile has them
    # there, far past where a faint star's stamp ends.
    expected = bright[3] * float(np.mean([plain[bank.WIDE_STAMP // 2, bank.WIDE_STAMP // 2 + r] for r in range(20, 26)]))
    assert annulus({"light": made["light"] - others}, bright, 20, 26) == pytest.approx(expected, rel=0.3)
    faint = telescope.stamp_for(0.0, 0.0, 0.1, float(np.median(noise)))
    assert faint.shape[0] < 2 * 20 + 1


def test_a_star_s_stamp_reaches_as_far_as_its_light_and_ends_round():
    plain = stars.moffat_stamp(bank.WIDE_STAMP, 4.0, 1.5)
    telescope = bank.Telescope(stamp=plain, fwhm=4.0)
    floor = 2.0
    for flux in (10.0, 1e4, 1e7):
        stamp = telescope.stamp_for(10.0, 10.0, flux, floor)
        reach = telescope.reach(flux, floor)
        half = stamp.shape[0] // 2
        distance = bank.radii(stamp.shape[0])
        # Out to its reach the stamp is the profile's, untouched...
        inside = distance <= min(reach, half / bank.FADE_BAND) - 1
        np.testing.assert_allclose(stamp[inside], bank.cropped(plain, stamp.shape[0])[inside])
        # ...past its reach the profile's light is under the fade...
        if half < bank.WIDE_STAMP // 2:
            middle = bank.WIDE_STAMP // 2
            assert flux * plain[middle, middle + int(math.ceil(reach)) + 1] < bank.FADE * floor
        # ...and it ends in a circle: nothing at its half side or past it.
        assert np.all(stamp[distance >= half] == 0.0)
        assert stamp[half, 0] == 0.0 and stamp[0, 0] == 0.0
    # The brighter, the further.
    assert telescope.reach(1e3, floor) < telescope.reach(1e6, floor)
    assert telescope.reach(1e-3, floor) == bank.CORE_REACH * 4.0


class StampField:
    """A stand-in for a PSF field: one stamp everywhere."""

    def __init__(self, stamp):
        self.stamp = stamp
        self.arrays = {"stamps": stamp[None, None]}

    def stamp_at(self, x, y):
        return self.stamp


def test_a_field_s_stamp_is_carried_on_by_a_wing_matched_where_it_ends():
    rng = np.random.default_rng(21)
    truth = stars.moffat_stamp(bank.WIDE_STAMP, 6.0, 2.5)
    # The field's stamp: the middle 65 px, with noise near its edges.
    small = bank.cropped(truth, 65) + rng.normal(0.0, 2e-6, (65, 65)) * (bank.radii(65) > 20)
    telescope = bank.Telescope(field=StampField(small), fwhm=6.0, wing_beta=2.5)
    stamp = telescope.stamp_for(0.0, 0.0, 1e8, 1.0)
    assert stamp.shape[0] > 65
    middle = stamp.shape[0] // 2
    along = stamp[middle, middle:]
    profile = truth[bank.WIDE_STAMP // 2, bank.WIDE_STAMP // 2:]
    # The field's own stamp in its trusted middle; past its edge a wing with
    # the profile's light, carried on smoothly.
    np.testing.assert_allclose(along[:15], profile[:15], rtol=1e-3)
    np.testing.assert_allclose(along[35:60], profile[35:60], rtol=0.15)
    # A faint star's stamp stays inside the field's own.
    assert telescope.stamp_for(0.0, 0.0, 1.0, 1.0).shape[0] <= 65


def test_bright_cores_clip_where_the_stack_clips():
    rng = np.random.default_rng(10)
    sky, table, summary, noise = one_bright_star_case(rng)
    summary = {**summary, "clip_level": 2000.0}
    made = bank.composite(sky, table, summary, bank.Telescope(stamp=stars.moffat_stamp(bank.WIDE_STAMP, 4.0, 3.0), fwhm=4.0),
                          rng, noise)
    assert np.nanmax(made["input"]) == pytest.approx(2000.0)
    assert np.sum(made["input"] == np.float32(2000.0)) >= 4
    # The stars truth is what the image shows of them, clipped too.
    np.testing.assert_allclose(made["starless"] + made["stars"], made["input"], atol=1e-3)


def test_a_composite_is_fitted_at_the_places_its_stars_were_put():
    rng = np.random.default_rng(7)
    sky, table, summary, noise = clean_sky_case(rng, size=(500, 500), count=260, faintest=17.0)
    telescope = bank.Telescope(stamp=stars.moffat_stamp(bank.WIDE_STAMP, 4.0, 3.0), fwhm=4.0)
    made = bank.composite(sky, table, summary, telescope, rng, noise)
    fit = bank.fitted(made, noise, summary["clip_level"], 4.0)
    assert fit["stars"].shape[0] == len(made["placed"]) + len(made["listed"])
    kept = ~fit["mask"][20:-20, 20:-20]
    left = (fit["sky"] - made["starless"])[20:-20, 20:-20][kept]
    # The fit takes most of the stars out: what is left is near the noise.
    assert abs(float(np.nanmean(left))) < 0.5
    assert float(np.nanstd(left)) < 1.6 * float(np.median(noise))


def test_a_points_plane_marks_each_star_by_its_light_share():
    placed = np.array([[10.2, 20.7, 12.0, 0.0], [30.0, 5.0, 14.5, 0.0], [-3.0, 5.0, 10.0, 0.0]])
    plane = bank.points_plane(placed, (40, 40))
    assert plane[21, 10] == pytest.approx(10 ** (-0.4 * 2.0))
    assert plane[5, 30] == pytest.approx(10 ** (-0.4 * 4.5))
    assert np.count_nonzero(plane) == 2


def test_a_bank_holds_windows_whose_truth_adds_back_and_whose_mask_is_the_clean_sky_s(tmp_path, monkeypatch):
    import json

    from seiza_training.store import Store

    monkeypatch.setattr(bank, "PUPIL_SHARE", 1.0)
    monkeypatch.setattr(bank, "GALAXY_DENSITY", (0.0, 0.0))
    rng = np.random.default_rng(11)
    sky, table, summary, noise = clean_sky_case(rng, size=(500, 500), count=260, faintest=17.0)
    mask = np.zeros(sky.shape, bool)
    mask[0:120, 0:120] = True
    store = Store(tmp_path / "store", tmp_path / "scratch")
    here = ss.directory(store, "abc/deep")
    here.mkdir(parents=True)
    np.save(here / "sky.npy", sky)
    np.save(here / "mask.npy", mask)
    np.savez_compressed(here / "stars.npz", stars=table, columns=np.array(ss.STAR_COLUMNS))
    (here / "sky.json").write_text(json.dumps({**summary, "version": ss.CLEAN_SKY_VERSION}))
    done = bank.cut(store, [{"id": "abc/deep"}], work=tmp_path / "banks", draws=2, windows=3, side=128, seed=4,
                    workers=1)
    assert done["windows"] == 6 and done["telescopes"] == {"field": 0, "pupil": 6} and not done["dropped"]
    directory = tmp_path / "banks" / done["name"]
    planes = {name: np.load(directory / f"{name}.npy", mmap_mode="r")[:done["windows"]] for name in bank.PLANES}
    records = json.loads((directory / "sources.json").read_text())
    for index, record in enumerate(records):
        top, left = record["top"], record["left"]
        # The starless truth is the clean sky there, and the window's valid
        # plane is the clean sky's mask's complement.
        np.testing.assert_allclose(planes["starless"][index], sky[top:top + 128, left:left + 128], atol=1e-4)
        np.testing.assert_array_equal(planes["valid"][index].astype(bool), ~mask[top:top + 128, left:left + 128])
        assert planes["valid"][index].mean() >= bank.VALID_SHARE
    # The two draws are different composites of one sky.
    assert not np.array_equal(planes["input"][0], planes["input"][3])
    # A finished bank is opened, not cut again.
    assert bank.cut(store, [{"id": "abc/deep"}], work=tmp_path / "banks", draws=2, windows=3, side=128, seed=4,
                    workers=1) == done


def test_a_pupil_telescope_has_the_stack_s_width():
    rng = np.random.default_rng(8)
    for _ in range(3):
        telescope = bank.pupil_telescope(rng, 4.0)
        stamp = telescope.stamp
        assert stamp.shape == (bank.WIDE_STAMP, bank.WIDE_STAMP) and stamp.sum() == pytest.approx(1.0)
        middle = bank.WIDE_STAMP // 2
        row = stamp[middle]
        above = np.nonzero(row >= row.max() / 2)[0]
        assert 2.5 <= above[-1] - above[0] + 1 <= 6.5


def vaned_telescope(vanes=4, hexagon=False):
    """A pupil telescope of known vanes, and its PSF without them."""
    pupil = stars.Pupil(obstruction=0.3, vanes=vanes, vane_width=0.02, angle=20.0, hexagon=hexagon)
    optics = stars.Optics(diffraction_px=1.5, seeing_fwhm=2.5, pupil=pupil)
    bare = stars.Optics(diffraction_px=1.5, seeing_fwhm=2.5, pupil=stars.Pupil(obstruction=0.3))
    return (bank.Telescope(stamp=stars.pupil_psf(optics, bank.WIDE_STAMP), fwhm=3.0,
                           plain=stars.pupil_psf(bare, bank.WIDE_STAMP)), pupil)


def test_a_pupil_s_spikes_are_its_stamp_less_the_same_optics_without_vanes():
    telescope, pupil = vaned_telescope()
    drawn, spikes = telescope.stamps_for(0.0, 0.0, 1e7, 1.0)
    assert spikes.shape == drawn.shape
    directions = stars.spike_directions(pupil)
    # The spikes hold the light along the spikes and next to none between...
    seen = stars.spikes_seen(spikes, directions, 15.0, 60.0)
    assert seen["on"] > 20 * abs(seen["between"])
    # ...and the stamp less them throws none.
    rest = stars.spikes_seen(drawn - spikes, directions, 15.0, 60.0)
    assert rest["on"] < 1.5 * rest["between"]
    # The spikes are the vanes' alone: the obstruction's rings stay with the
    # star (4 to 12 px out, the spikes hold 1% of the light; with the
    # obstruction's rings, 17%).
    middle = drawn.shape[0] // 2
    yy, xx = np.mgrid[0:drawn.shape[0], 0:drawn.shape[1]]
    near = (np.hypot(xx - middle, yy - middle) >= 4) & (np.hypot(xx - middle, yy - middle) < 12)
    assert abs(spikes[near].sum()) < 0.05 * drawn[near].sum()
    # A telescope without vanes, as the stack's own field, throws none.
    assert bank.Telescope(stamp=telescope.stamp, fwhm=3.0).stamps_for(0.0, 0.0, 1e7, 1.0)[1] is None
    # And a drawn pupil telescope keeps its PSF without vanes where it has vanes.
    rng = np.random.default_rng(31)
    for _ in range(6):
        drawn_telescope = bank.pupil_telescope(rng, 3.0)
        has_vanes = not drawn_telescope.description.startswith("pupil vanes 0 ")
        assert (drawn_telescope.plain is not None) == has_vanes
        if has_vanes:
            thrown = drawn_telescope.stamp - drawn_telescope.plain
            assert abs(thrown[near].sum()) < 0.05 * drawn_telescope.stamp[near].sum()


def test_a_composite_keeps_its_halos_and_spikes_light_apart():
    telescope, pupil = vaned_telescope()
    sky, table, summary, noise = one_bright_star_case(np.random.default_rng(6))
    made = bank.composite(sky, table, summary, telescope, np.random.default_rng(16), noise)
    # The halos' light is what the stars hold past their own profiles.
    np.testing.assert_allclose(made["halos"], made["light"] - redrawn(made, telescope, noise),
                               atol=1e-6 * float(made["light"].max()))
    assert made["halos"].sum() > 0
    # The spikes' light runs along the brightest star's spikes.
    bright = made["placed"][np.argmin(made["placed"][:, 2])]
    half = 80
    row, column = int(round(bright[1])), int(round(bright[0]))
    if half <= row < sky.shape[0] - half and half <= column < sky.shape[1] - half:
        cut = made["spikes"][row - half:row + half + 1, column - half:column + half + 1]
        seen = stars.spikes_seen(cut, stars.spike_directions(pupil), 15.0, 60.0)
        assert seen["on"] > 10 * abs(seen["between"])
    assert np.abs(made["spikes"]).sum() > 0
    # A telescope without vanes leaves the spikes' plane empty.
    plain = bank.Telescope(stamp=stars.moffat_stamp(bank.WIDE_STAMP, 4.0, 3.0), fwhm=4.0)
    assert not bank.composite(sky, table, summary, plain, np.random.default_rng(16), noise)["spikes"].any()


def test_a_made_galaxy_is_of_unit_light_centred_and_as_long_as_asked():
    stamp = stars.galaxy_stamp(61, 5.0, 1.0, 0.3, 0.3, 0.9, 30.0)
    assert stamp.sum() == pytest.approx(1.0) and stamp.min() >= 0.0
    yy, xx = np.mgrid[0:61, 0:61].astype(np.float64) - 30.0
    assert float((stamp * xx).sum()) == pytest.approx(0.0, abs=1e-6)
    assert float((stamp * yy).sum()) == pytest.approx(0.0, abs=1e-6)
    # Its long axis lies at the angle asked, and it is as flattened.
    cov = np.array([[(stamp * xx * xx).sum(), (stamp * xx * yy).sum()],
                    [(stamp * xx * yy).sum(), (stamp * yy * yy).sum()]])
    values, vectors = np.linalg.eigh(cov)
    angle = math.degrees(math.atan2(vectors[1, 1], vectors[0, 1])) % 180.0
    assert angle == pytest.approx(30.0, abs=3.0)
    assert math.sqrt(values[0] / values[1]) < 0.6
    # A galaxy under a pixel keeps its light, sampled finer.
    assert stars.galaxy_stamp(15, 0.3, 0.1, 0.5, 0.5, 0.8, 0.0).sum() == pytest.approx(1.0)


def test_a_composite_keeps_its_galaxies_in_the_starless_truth(monkeypatch):
    monkeypatch.setattr(bank, "GALAXY_DENSITY", (1500.0, 1500.0))
    rng = np.random.default_rng(21)
    sky, table, summary, noise = clean_sky_case(rng, count=150)
    telescope = bank.Telescope(stamp=stars.moffat_stamp(bank.WIDE_STAMP, 4.0, 3.0), fwhm=4.0)
    made = bank.composite(sky, table, summary, telescope, rng, noise)
    galaxies = made["galaxies"].astype(np.float64)
    assert galaxies.sum() > 0 and galaxies.min() >= 0.0
    # The starless truth is the sky with the galaxies and their photon
    # noise; the stars layer, the composite less it, is the stars' alone.
    unclipped = made["input"] < summary["clip_level"]
    left = (made["starless"] - sky - galaxies)[unclipped]
    assert abs(float(left.mean())) < 0.05
    bright = (galaxies > 200.0) & unclipped & (made["light"] < 1e-3)
    if bright.sum() > 100:
        ratio = np.var((made["starless"] - sky - galaxies)[bright]) / np.mean(summary["photon_term"] * galaxies[bright])
        assert ratio == pytest.approx(1.0, rel=0.3)
    np.testing.assert_allclose(made["starless"] + made["stars"], made["input"], atol=1e-3)
    away = (made["light"] < 1e-6) & unclipped
    assert float(np.abs(made["stars"][away]).max()) < 1e-5 * float(np.abs(made["input"][away]).max()) + 1e-3
    # Some of them carry a catalogue source, fitted and marked as a star.
    count = int(round(400 * 400 / 1e6 * 1500.0))
    assert 0.15 * count < len(made["listed"]) < 0.45 * count
    assert bank.catalogued(made).shape == (len(made["placed"]) + len(made["listed"]), 4)
    plane = bank.points_plane(bank.catalogued(made), sky.shape)
    x, y = made["listed"][0, :2]
    assert plane[int(round(y)), int(round(x))] > 0


def test_a_galaxy_s_sources_are_left_and_a_star_before_it_is_not():
    galaxies = np.array([[100.0, 100.0, 40.0, 20.0, 0.0]])
    x = np.array([100.0, 120.0, 120.0, 100.0, 104.0, 130.0, 300.0])
    y = np.array([100.0, 100.0, 100.0, 110.0, 100.0, 100.0, 300.0])
    chi2 = np.array([50.0, 0.9, 8.0, 0.95, 0.9, np.nan, 30.0])
    # The nucleus, a poorly fitted source and one at the middle are the
    # galaxy's; a point-like one off the middle is a star before it; one
    # outside every galaxy is a star.
    np.testing.assert_array_equal(ss.galaxy_sources(galaxies, x, y, chi2),
                                  [True, False, True, False, True, True, False])
    out = ss.within_galaxies(galaxies, x, y)
    assert out[0] == 0.0 and out[1] == pytest.approx(0.5) and out[3] == pytest.approx(0.5) and np.isinf(out[6])
    mask = ss.galaxy_mask(galaxies, (200, 200))
    assert mask.sum() == pytest.approx(math.pi * 40.0 * 20.0, rel=0.03)
    assert mask[100, 139] and not mask[100, 141] and mask[119, 100] and not mask[121, 100]
    assert not ss.galaxy_mask(np.zeros((0, 5)), (20, 20)).any()
