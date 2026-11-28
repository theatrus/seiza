"""The starless model's trainer: its two ways, the tiles it reads, its loss
and its baseline, and that a short run beats the baseline."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from seiza_training import starless_bank as bank  # noqa: E402
from seiza_training import starless_model as sm  # noqa: E402
from seiza_training import starless_sky as ss  # noqa: E402
from seiza_training import starless_stars as stars  # noqa: E402
from seiza_training.store import Store  # noqa: E402
from seiza_training.unet import ResidualUNet  # noqa: E402


@pytest.fixture(scope="module")
def small_bank(tmp_path_factory):
    """A bank of composites of one synthetic clean sky, pupil telescopes."""
    tmp_path = tmp_path_factory.mktemp("starless")
    rng = np.random.default_rng(12)
    size = (512, 512)
    yy, xx = np.mgrid[0:size[0], 0:size[1]].astype(np.float64)
    sky = (100.0 + 30.0 * np.exp(-((xx - 250) ** 2 + (yy - 260) ** 2) / (2 * 110.0**2))
           + rng.standard_normal(size) * 2.0).astype(np.float32)
    count = 300
    table = np.zeros((count, len(ss.STAR_COLUMNS)))
    col = {name: index for index, name in enumerate(ss.STAR_COLUMNS)}
    magnitudes = rng.uniform(13.0, 17.0, count)
    table[:, col["magnitude"]], table[:, col["flux"]] = magnitudes, 10 ** (-0.4 * (magnitudes - 25.0))
    table[:, col["flux_error"]] = table[:, col["flux"]] / 1000.0
    store = Store(tmp_path / "store", tmp_path / "scratch")
    here = ss.directory(store, "syn/deep")
    here.mkdir(parents=True)
    np.save(here / "sky.npy", sky)
    np.save(here / "mask.npy", np.zeros(size, bool))
    np.savez_compressed(here / "stars.npz", stars=table, columns=np.array(ss.STAR_COLUMNS))
    (here / "sky.json").write_text(json.dumps({"fwhm_px": 4.0, "photon_term": 0.5, "clip_level": 40000.0,
                                               "version": ss.CLEAN_SKY_VERSION}))
    original = bank.PUPIL_SHARE
    bank.PUPIL_SHARE = 1.0
    try:
        done = bank.cut(store, [{"id": "syn/deep"}], work=tmp_path / "banks", draws=2, windows=4, side=128,
                        seed=1, workers=1)
    finally:
        bank.PUPIL_SHARE = original
    return store, tmp_path / "banks" / done["name"]


def test_compression_and_its_inverse_meet():
    plane = np.array([[-5.0, 100.0, 1e5]], np.float32)
    back = sm.released(sm.compressed(plane, 10.0, 2.0), 10.0, 2.0)
    np.testing.assert_allclose(back, plane, rtol=1e-5)


@pytest.mark.parametrize("way, start", [("fit", "fit_starless"), ("image", "input")])
def test_a_run_starts_as_its_baseline(small_bank, way, start):
    _, directory = small_bank
    found = sm.Bank(directory)
    settings = sm.Settings(way=way, tile=64, width=8, levels=3)
    model = ResidualUNet(inputs=len(sm.READS[way]), outputs=1, width=8, levels=3)
    batch = sm.batch_from(found, np.array([0, 1]), settings, np.random.default_rng(2))
    assert batch["planes"].shape == (2, len(sm.READS[way]), 64, 64)
    run_loss, base_loss, _ = sm.loss_of(model, batch)
    assert float(run_loss) == pytest.approx(float(base_loss))
    planes = {name: np.asarray(found.planes[name][0, :64, :64]) for name in found.planes}
    # The fit way starts at the fit's starless estimate, the image way at the
    # image.
    np.testing.assert_allclose(sm.starless(model, planes, settings), planes[start], rtol=1e-4, atol=1e-2)


def test_the_stars_model_is_read_about_zero_and_the_images_about_the_sky(small_bank):
    _, directory = small_bank
    found = sm.Bank(directory)
    planes = {name: np.asarray(found.planes[name][0]) for name in found.planes}
    read, level, scale = sm.network_reads(planes, "fit")
    by_name = dict(zip(sm.READS["fit"], read))
    starless = np.abs(planes["input"] - planes["starless"]) < planes["noise"]
    # Where no star is, the image and the fit's estimate read near zero, and
    # so does the stars model, which holds no sky.
    for name in ("input", "fit_starless", "fit_model"):
        assert abs(float(np.median(by_name[name][starless]))) < 0.2, name
    assert level > 50.0 * scale / sm.ASINH


def test_the_loss_weighs_where_stars_were_and_skips_what_is_not_valid(small_bank):
    _, directory = small_bank
    found = sm.Bank(directory)
    example = sm.example(found, 0, sm.Settings(way="image", tile=128, width=8, levels=3), np.random.default_rng(3))
    weight = example["weight"]
    assert set(np.unique(weight)) <= {0.0, 1.0, sm.STAR_WEIGHT}
    assert (weight == sm.STAR_WEIGHT).any() and (weight == 1.0).any()


def test_a_short_run_beats_its_baseline(small_bank, tmp_path):
    store, directory = small_bank
    record = sm.train(store, name="tiny", bank=directory, validation_bank=directory,
                      settings=sm.Settings(way="image", tile=64, width=8, levels=3), steps=300, batch=8,
                      learning_rate=3e-3, log_every=100, checkpoint_every=300, validation_batches=2,
                      where=torch.device("cpu"))
    last = record["history"][-1]
    assert last["validation"] < 0.8 * last["validation_baseline"]
    assert (sm.run_directory(store, "tiny") / "weights.pt").exists()
    with pytest.raises(ValueError):
        sm.Settings(way="both")
    # Its evaluation: closer to the truth than its baseline where stars
    # were, and a sheet beside the numbers.
    found = sm.evaluate(store, "tiny", directory, count=8, sheet_rows=2, where=torch.device("cpu"))
    run, base = found["measures"]["all/run"]["median"], found["measures"]["all/baseline"]["median"]
    assert run["stars_rms"] < base["stars_rms"] and run["light_left"] < base["light_left"]
    assert run["stars_stretched"] < base["stars_stretched"]
    assert found["measures"]["pupil/run"] and "field/run" not in found["measures"]
    evaluation = sm.run_directory(store, "tiny") / "evaluation"
    assert (evaluation / f"{directory.name}.json").exists() and (evaluation / f"{directory.name}.png").exists()


def with_a_halo(planes):
    """A window's planes with a halo of 20 times the noise at its middle
    added to its input and its halos."""
    planes = dict(planes)
    side = planes["input"].shape[0]
    yy, xx = np.mgrid[0:side, 0:side] - (side - 1) / 2.0
    halo = (20.0 * float(np.median(planes["noise"])) * np.exp(-(xx**2 + yy**2) / (2 * 12.0**2))).astype(np.float32)
    planes["input"] = planes["input"] + halo
    planes["halos"] = planes["halos"] + halo
    return planes


def test_a_layered_run_starts_with_no_halos_or_spikes_and_its_layers_add_back(small_bank):
    _, directory = small_bank
    found = sm.Bank(directory)
    settings = sm.Settings(way="fit", tile=64, width=8, levels=3, layers=True)
    assert settings.outputs == 1 + len(sm.LAYERS) and sm.Settings().outputs == 1
    model = ResidualUNet(inputs=len(sm.READS["fit"]), outputs=settings.outputs, width=8, levels=3)
    batch = sm.batch_from(found, np.array([0, 1]), settings, np.random.default_rng(2))
    assert batch["spikes"].shape == batch["halos"].shape == (2, 64, 64)
    run_loss, base_loss, parts = sm.loss_of(model, batch)
    assert float(run_loss) == pytest.approx(float(base_loss))
    assert set(parts) == {f"{name}{part}" for name in ("starless", *sm.LAYERS) for part in ("", "_baseline")}
    planes = with_a_halo({name: np.asarray(found.planes[name][0, :64, :64]) for name in found.planes})
    layers = sm.layers(model, planes, settings)
    assert not layers["halos"].any() and not layers["spikes"].any()
    np.testing.assert_allclose(layers["starless"], planes["fit_starless"], rtol=1e-4, atol=1e-2)
    # Given the halo, the four add back to the input, and the share of it
    # kept moves it from the stars to the starless image, the three a user
    # takes still adding back.
    layers["halos"] = planes["halos"]
    layers["stars"] = planes["input"] - layers["starless"] - layers["halos"] - layers["spikes"]
    for share in (0.0, 0.3, 1.0):
        taken = sm.kept(layers, share)
        np.testing.assert_allclose(taken["starless"] + taken["stars"] + taken["spikes"], planes["input"], atol=1e-2)
        np.testing.assert_allclose(taken["starless"] - layers["starless"], share * planes["halos"], atol=1e-3)
    with pytest.raises(ValueError):
        sm.kept(layers, 1.5)


def test_the_spike_penalty_is_a_setting_that_defaults_to_every_part_s(small_bank):
    _, directory = small_bank
    found = sm.Bank(directory)
    plain = sm.Settings(way="image", tile=64, width=8, levels=3, layers=True)
    eased = sm.Settings(way="image", tile=64, width=8, levels=3, layers=True, spike_quadratic=0.25,
                        spike_weight=4.0)
    torch.manual_seed(3)
    model = ResidualUNet(inputs=len(sm.READS["image"]), outputs=plain.outputs, width=8, levels=3)
    torch.nn.init.normal_(model.last.weight, std=0.05)
    batch = sm.batch_from(found, np.array([0, 1]), plain, np.random.default_rng(2))
    with torch.no_grad():
        out = model(batch["planes"])
        run, _, parts = sm.loss_of(model, batch, plain)
        weight, total = batch["weight"], batch["weight"].sum()

        def charbonnier(estimate, truth, quadratic):
            return float((weight * (torch.sqrt((estimate - truth) ** 2 + quadratic**2) - quadratic
                                    + sm.CHARBONNIER)).sum() / total)

        # By default, the spikes are penalised as the starless image and the
        # halos are, and weigh as the halos do; without settings, alike.
        assert float(parts["spikes"]) == pytest.approx(charbonnier(out[:, 2], batch["spikes"], sm.CHARBONNIER))
        assert float(run) == pytest.approx(float(parts["starless"]) + sm.LAYER_WEIGHT
                                           * (float(parts["halos"]) + float(parts["spikes"])))
        assert float(sm.loss_of(model, batch)[0]) == pytest.approx(float(run))
        # Eased, only the spikes' part changes: quadratic to 0.25, from the
        # same floor (a perfect estimate pays the usual epsilon), weighed four
        # times.
        run_eased, _, parts_eased = sm.loss_of(model, batch, eased)
        assert float(parts_eased["starless"]) == pytest.approx(float(parts["starless"]))
        assert float(parts_eased["halos"]) == pytest.approx(float(parts["halos"]))
        assert float(parts_eased["spikes"]) == pytest.approx(charbonnier(out[:, 2], batch["spikes"], 0.25))
        assert float(parts_eased["spikes"]) < float(parts["spikes"])
        assert charbonnier(batch["spikes"], batch["spikes"], 0.25) == pytest.approx(
            float((weight * sm.CHARBONNIER).sum() / total))
        assert float(run_eased) == pytest.approx(float(parts["starless"]) + sm.LAYER_WEIGHT * float(parts["halos"])
                                                 + 4.0 * float(parts_eased["spikes"]))
    with pytest.raises(ValueError):
        sm.Settings(layers=True, spike_quadratic=0.0)
    # A run kept before the setting reads as the default penalty.
    kept_before = {"way": "fit", "tile": 256, "width": 32, "levels": 5, "layers": True}
    assert sm.Settings(**kept_before) == sm.Settings(layers=True)


def test_a_share_of_each_batch_can_be_drawn_from_the_windows_with_spikes(small_bank):
    _, directory = small_bank
    found = sm.Bank(directory)
    # The windows with spikes are those whose truth tops the noise over
    # enough pixels.
    over = [int((np.asarray(found.planes["spikes"][index]) > np.asarray(found.planes["noise"][index])).sum())
            for index in range(found.count)]
    for pixels in (1, sm.SPIKY_PIXELS):
        np.testing.assert_array_equal(sm.spiky_windows(found, pixels),
                                      [index for index, count in enumerate(over) if count >= pixels])
    # The windows with bright stars and no spikes: none of their spike
    # truth over the noise, their stars' light peaking high enough.
    plain = [index for index, count in enumerate(over) if not count
             and float((np.asarray(found.planes["input"][index]) - np.asarray(found.planes["starless"][index])).max())
             >= 5.0 * float(np.median(found.planes["noise"][index]))]
    np.testing.assert_array_equal(sm.spikeless_windows(found, 5.0), plain)
    # With no share, a batch is drawn as before; with a quarter, four of
    # sixteen come from the windows with spikes, and with another eighth,
    # two from those without.
    spiky, spikeless = np.array([3, 7]), np.array([11, 13])
    before = np.random.default_rng(5).integers(0, 100, 16)
    np.testing.assert_array_equal(sm.drawn(100, [(spiky, 0.0)], 16, np.random.default_rng(5)), before)
    np.testing.assert_array_equal(sm.drawn(100, [], 16, np.random.default_rng(5)), before)
    picked = sm.drawn(100, [(spiky, 0.25)], 16, np.random.default_rng(5))
    assert len(picked) == 16 and set(picked[-4:]) <= {3, 7}
    picked = sm.drawn(100, [(spiky, 0.25), (spikeless, 0.125)], 16, np.random.default_rng(5))
    assert len(picked) == 16 and set(picked[-6:-2]) <= {3, 7} and set(picked[-2:]) <= {11, 13}
    np.testing.assert_array_equal(sm.drawn(100, [(np.zeros(0, np.int64), 0.25)], 16, np.random.default_rng(5)),
                                  before)
    with pytest.raises(ValueError):
        sm.drawn(100, [(spiky, 0.75), (spikeless, 0.5)], 16, np.random.default_rng(5))
    store, _ = small_bank
    with pytest.raises(ValueError):
        sm.train(store, name="no-layers", bank=directory, validation_bank=None,
                 settings=sm.Settings(way="image", tile=64, width=8, levels=3), steps=1, spike_share=0.25)
    with pytest.raises(ValueError):
        sm.train(store, name="too-much", bank=directory, validation_bank=None,
                 settings=sm.Settings(way="image", tile=64, width=8, levels=3, layers=True), steps=1,
                 spike_share=0.75, spikeless_share=0.5)


def test_a_tile_can_be_read_about_a_level_moved_from_its_own(small_bank):
    _, directory = small_bank
    found = sm.Bank(directory)
    planes = {name: np.asarray(found.planes[name][0, :64, :64], dtype=np.float32) for name in found.planes}
    _, level, scale = sm.network_reads(planes, "image")
    assert level == pytest.approx(float(np.nanmedian(planes["input"])))
    # Read about a level and noise given, the input is read less that level
    # in that noise, and the noise map as its ratio to it.
    given = (level - 20.0 * scale, 1.3 * scale)
    given_read, at, by = sm.network_reads(planes, "image", *given)
    assert (at, by) == pytest.approx(given)
    np.testing.assert_allclose(given_read[0], sm.compressed(planes["input"], *given), rtol=1e-6)
    np.testing.assert_allclose(given_read[1], planes["noise"] / given[1], rtol=1e-6)
    # The moves: most below the tile's own level, some near it, a few above,
    # none past 60 noise; the noise a little either side.
    rng = np.random.default_rng(4)
    moves = np.array([sm.jittered(0.0, 1.0, rng) for _ in range(4000)])
    assert 0.3 < np.mean(moves[:, 0] < -1.5) < 0.6 and np.mean(moves[:, 0] > 1.0) > 0.1
    assert moves[:, 0].min() >= -60.0 and moves[:, 0].max() <= 12.0
    assert 0.8 < np.median(moves[:, 1]) < 1.2
    # A run with jitter draws its tiles so; one without, as before.
    plain = sm.Settings(way="image", tile=64, width=8, levels=3)
    moved = sm.Settings(way="image", tile=64, width=8, levels=3, level_jitter=True)
    a = sm.example(found, 0, plain, np.random.default_rng(9))
    b = sm.example(found, 0, moved, np.random.default_rng(9))
    assert not np.allclose(a["planes"][0], b["planes"][0])
    # Its layers come back the same at any level given to a model that
    # does nothing (a residual of zero), as `layers` releases about it.
    model = ResidualUNet(inputs=len(sm.READS["image"]), outputs=moved.outputs, width=8, levels=3)
    own = sm.layers(model, planes, moved)
    other = sm.layers(model, planes, moved, level=level - 30.0 * scale, scale=scale)
    np.testing.assert_allclose(other["starless"], own["starless"], rtol=1e-4, atol=1e-3 * scale)


def test_a_short_layered_run_finds_the_spikes(small_bank):
    store, directory = small_bank
    record = sm.train(store, name="tiny-layers", bank=directory, validation_bank=directory,
                      settings=sm.Settings(way="image", tile=64, width=8, levels=3, layers=True), steps=300,
                      batch=8, learning_rate=3e-3, log_every=100, checkpoint_every=300, validation_batches=2,
                      where=torch.device("cpu"))
    last = record["history"][-1]
    assert last["validation"] < 0.8 * last["validation_baseline"]
    assert last["validation_spikes"] < 0.9 * last["validation_spikes_baseline"]
    assert record["config"]["layers"] == list(sm.LAYERS)
    found = sm.evaluate(store, "tiny-layers", directory, count=8, sheet_rows=2, where=torch.device("cpu"))
    run = found["measures"]["all/run"]["median"]
    assert run["spikes_found"] > 0.0 and "halos_rms" in run
    assert "spikes_rms" not in found["measures"]["all/baseline"]["median"]
    # Its sheet shows the halos and spikes beside the bank's: ten columns.
    from PIL import Image

    with Image.open(sm.run_directory(store, "tiny-layers") / "evaluation" / f"{directory.name}.png") as sheet:
        assert sheet.width == 10 * 128 + 9 * 6


def test_the_layer_measures_read_the_truth_as_perfect_and_none_as_nothing_found(small_bank):
    _, directory = small_bank
    found = sm.Bank(directory)
    # A window whose spikes top the noise somewhere.
    index = next(index for index in range(found.count)
                 if (found.planes["spikes"][index] > found.planes["noise"][index]).any())
    planes = with_a_halo({name: np.asarray(found.planes[name][index]) for name in found.planes})
    perfect = sm.layer_measures({name: planes[name] for name in sm.LAYERS}, planes)
    for name in sm.LAYERS:
        assert perfect[f"{name}_rms"] == 0.0 and perfect[f"{name}_stray"] == 0.0
        assert perfect[f"{name}_found"] == pytest.approx(1.0)
    none = sm.layer_measures({name: np.zeros_like(planes[name]) for name in sm.LAYERS}, planes)
    assert none["halos_found"] == 0.0 and none["halos_rms"] > 1.0 and none["halos_stray"] < 0.1
    # The halo given as spikes is found by neither, and strays from the
    # spikes.
    swapped = sm.layer_measures({"halos": np.zeros_like(planes["halos"]),
                                 "spikes": planes["spikes"] + planes["halos"]}, planes)
    assert swapped["halos_found"] == 0.0 and swapped["spikes_stray"] > 0.1


def test_the_measures_read_the_truth_as_perfect_and_the_input_as_all_the_stars_left(small_bank):
    _, directory = small_bank
    found = sm.Bank(directory)
    planes = {name: np.asarray(found.planes[name][1]) for name in found.planes}
    perfect = sm.measures(planes["starless"], planes)
    assert perfect["stars_rms"] == 0.0 and perfect["sky_rms"] == 0.0 and perfect["light_left"] == 0.0
    assert perfect["stars_stretched"] == 0.0 and perfect["sky_stretched"] == 0.0
    untouched = sm.measures(planes["input"], planes)
    assert untouched["light_left"] == pytest.approx(1.0) and untouched["cores_mean"] > 10.0
    # A hole: the truth less the stars' light reads as a negative core.
    holed = sm.measures(planes["starless"] - (planes["input"] - planes["starless"]), planes)
    assert holed["cores_mean"] < -10.0 and holed["light_left"] == pytest.approx(-1.0)
    # The truth keeps all its noise where stars were; a smooth fill none.
    assert perfect["noise_kept"] == pytest.approx(1.0)
    smooth = sm.box_mean(planes["starless"], 15)
    assert sm.measures(smooth, planes)["noise_kept"] < 0.4


def test_a_box_mean_is_the_mean_over_the_box_within_the_plane():
    plane = np.random.default_rng(3).standard_normal((20, 30))
    found = sm.box_mean(plane, 5)
    for row, column in ((0, 0), (10, 15), (19, 29), (2, 27)):
        box = plane[max(0, row - 2):row + 3, max(0, column - 2):column + 3]
        assert found[row, column] == pytest.approx(box.mean())


def sky_with_a_smooth_fill(rng, rho=0.2):
    """A sky of correlated noise (neighbours correlating by `rho`), 2 a
    pixel, on a slope, with a disc of 40 px smoothed over as a network fills
    a star."""
    size = (300, 300)
    noise = sm.grained(size, rho, rho, rng) * 2.0
    yy, xx = np.mgrid[0:size[0], 0:size[1]]
    sky = 100.0 + 0.01 * xx + noise
    disc = np.hypot(xx - 150, yy - 150) < 40
    smooth = np.where(disc, 100.0 + 0.01 * xx, sky)
    return sky.astype(np.float32), smooth.astype(np.float32), disc, np.full(size, 2.0, np.float32)


def test_a_smooth_fill_gets_noise_of_the_image_s_grain_and_the_layers_still_add_back():
    rng = np.random.default_rng(4)
    sky, smooth, disc, noise = sky_with_a_smooth_fill(rng)
    image = sky + 50.0 * np.exp(-((np.mgrid[0:300, 0:300][1] - 150) ** 2
                                  + (np.mgrid[0:300, 0:300][0] - 150) ** 2) / (2 * 6.0**2))
    out, stars = sm.filled(image, smooth, seed=1)
    np.testing.assert_allclose(out + stars, image, atol=1e-3)
    # Within the disc, noise as the sky's, of its grain...
    inner = np.hypot(*(np.mgrid[0:300, 0:300] - 150)) < 30
    drawn = (out - smooth).astype(np.float64)
    assert np.std(drawn[inner]) == pytest.approx(2.0, rel=0.15)
    across = np.corrcoef(drawn[inner][:-1], drawn[inner][1:])[0, 1]
    assert across == pytest.approx(0.2, abs=0.08)
    # ...and away from it, where nothing is lacking, nothing drawn.
    away = np.hypot(*(np.mgrid[0:300, 0:300] - 150)) > 60
    np.testing.assert_array_equal(out[away], smooth[away])
    # The same each time for one seed.
    np.testing.assert_array_equal(sm.filled(image, smooth, seed=1)[0], out)
    # A grain read from white noise is near none.
    white = rng.standard_normal((200, 200))
    assert max(sm.grain(white, np.ones(white.shape, bool))) < 0.05
