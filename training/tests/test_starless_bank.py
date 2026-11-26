"""The starless model's known-truth composites: stars put back into a clean
sky, their truth, and the catalogue fit of them."""

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


def test_a_composite_s_layers_add_back_and_its_stars_are_the_stack_s_kind():
    rng = np.random.default_rng(5)
    sky, table, summary, noise = clean_sky_case(rng)
    telescope = bank.Telescope(stamp=stars.moffat_stamp(bank.WIDE_STAMP, 4.0, 3.0))
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
    lone = bank.Telescope(stamp=stars.moffat_stamp(bank.WIDE_STAMP, 4.0, 3.0))
    alone = np.zeros_like(made["light"], dtype=np.float64)
    x, y, _, flux = made["placed"][star]
    bank.add_star(alone, lone.stamp_at(x, y, bank.NARROW_STAMP), x, y, flux)
    # (the shift by a fraction of a pixel loses a sliver of the stamp's edge)
    assert alone.sum() == pytest.approx(flux * lone.stamp_at(x, y, bank.NARROW_STAMP).sum(), rel=1e-3)
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


def redrawn(made, telescope, noise, *, narrow_only=False):
    """The composite's stars drawn again from its own list, with their
    plain profiles alone (no halos), each on the stamp `composite` picks, or
    all on the narrow one."""
    light = np.zeros(made["light"].shape, dtype=np.float64)
    floor = float(np.nanmedian(noise))
    for x, y, _, flux in made["placed"]:
        narrow = telescope.stamp_at(x, y, bank.NARROW_STAMP)
        wide = flux * float(narrow.max()) > bank.WIDE_PEAK * floor and not narrow_only
        bank.add_star(light, telescope.stamp_at(x, y, bank.WIDE_STAMP) if wide else narrow, x, y, flux)
    return light


def test_the_brightest_stars_carry_halos_wider_than_their_stamps():
    plain = stars.moffat_stamp(bank.WIDE_STAMP, 4.0, 3.0)
    telescope = bank.Telescope(stamp=plain)
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


def test_a_bright_star_is_drawn_on_the_wide_stamp_past_the_narrow_one(monkeypatch):
    monkeypatch.setattr(bank, "HALO_STARS", 0.0)
    plain = stars.moffat_stamp(bank.WIDE_STAMP, 4.0, 1.5)
    telescope = bank.Telescope(stamp=plain)
    sky, table, summary, noise = one_bright_star_case(np.random.default_rng(9))
    made = bank.composite(sky, table, summary, telescope, np.random.default_rng(19), noise)
    np.testing.assert_allclose(made["light"], redrawn(made, telescope, noise), atol=1e-2)
    beyond = made["light"] - redrawn(made, telescope, noise, narrow_only=True)
    bright = made["placed"][np.argmin(made["placed"][:, 2])]
    # Its wings reach past the narrow stamp's half-width, as the profile has
    # them there.
    expected = bright[3] * float(np.mean([plain[bank.WIDE_STAMP // 2, bank.WIDE_STAMP // 2 + r] for r in range(20, 26)]))
    assert annulus({"light": beyond}, bright, 20, 26) == pytest.approx(expected, rel=0.3)


def test_bright_cores_clip_where_the_stack_clips():
    rng = np.random.default_rng(10)
    sky, table, summary, noise = one_bright_star_case(rng)
    summary = {**summary, "clip_level": 2000.0}
    made = bank.composite(sky, table, summary, bank.Telescope(stamp=stars.moffat_stamp(bank.WIDE_STAMP, 4.0, 3.0)),
                          rng, noise)
    assert np.nanmax(made["input"]) == pytest.approx(2000.0)
    assert np.sum(made["input"] == np.float32(2000.0)) >= 4
    # The stars truth is what the image shows of them, clipped too.
    np.testing.assert_allclose(made["starless"] + made["stars"], made["input"], atol=1e-3)


def test_a_composite_is_fitted_at_the_places_its_stars_were_put():
    rng = np.random.default_rng(7)
    sky, table, summary, noise = clean_sky_case(rng, size=(500, 500), count=260, faintest=17.0)
    telescope = bank.Telescope(stamp=stars.moffat_stamp(bank.WIDE_STAMP, 4.0, 3.0))
    made = bank.composite(sky, table, summary, telescope, rng, noise)
    fit = bank.fitted(made, noise, summary["clip_level"], 4.0)
    assert fit["stars"].shape[0] == len(made["placed"])
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
