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
    run_loss, base_loss = sm.loss_of(model, batch)
    assert float(run_loss) == pytest.approx(float(base_loss))
    planes = {name: np.asarray(found.planes[name][0, :64, :64]) for name in found.planes}
    # The fit way starts at the fit's starless estimate, the image way at the
    # image.
    np.testing.assert_allclose(sm.starless(model, planes, settings), planes[start], rtol=1e-4, atol=1e-2)


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
