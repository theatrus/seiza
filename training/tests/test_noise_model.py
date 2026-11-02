"""Overlapping-tile inference, and a noise model trained end to end on the
synthetic corpus's noise pairs."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from seiza_training import tiles  # noqa: E402
from seiza_training.__main__ import main  # noqa: E402
from seiza_training.manifest import Manifest  # noqa: E402
from seiza_training.synthetic import build_corpus  # noqa: E402
from seiza_training.unet import ResidualUNet, receptive_field  # noqa: E402


def test_tiles_cover_the_image_and_blend_back_to_it():
    assert tiles.origins(1000, 256, 64) == [0, 192, 384, 576, 744]
    assert tiles.origins(200, 256, 64) == [0]
    identity = tiles.infer(lambda batch: batch[:, :1], np.random.default_rng(1).normal(size=(2, 300, 410)).astype(np.float32),
                           tile=128, overlap=32, margin=8)
    assert identity.shape == (1, 300, 410)


def test_overlapping_tiles_match_the_whole_image():
    torch.manual_seed(2)
    model = ResidualUNet(width=4, levels=3).eval()
    torch.nn.init.normal_(model.last.weight, std=0.1)
    planes = np.random.default_rng(3).normal(size=(2, 192, 256)).astype(np.float32)
    with torch.no_grad():
        whole = model(torch.from_numpy(planes)[None]).numpy()[0]
        run = lambda batch: model(torch.from_numpy(batch)).numpy()  # noqa: E731
        field = receptive_field(3)
        tiled = tiles.infer(run, planes, tile=96, overlap=field + 3, margin=(field + 1) // 2)
    np.testing.assert_allclose(tiled, whole, atol=1e-5 * np.abs(whole).max())


def test_a_noise_model_trains_evaluates_and_exports(tmp_path, capsys):
    corpus = build_corpus(tmp_path / "corpus")
    store = tmp_path / "store"

    def run(*arguments):
        assert main(["--store", str(store), "--scratch", str(tmp_path / "scratch"), *arguments]) == 0

    for arguments in (
        ("crawl", "--corpus", str(corpus / "telescopes"), "--library", str(corpus / "library")),
        ("calibrate",), ("split",), ("process", "--catalog-dir", str(corpus / "catalogs")),
    ):
        run(*arguments)
    with Manifest(store / "manifest.sqlite") as manifest:
        with manifest.transaction():
            manifest.db.execute("UPDATE sets SET split = 'train'")
    capsys.readouterr()
    run("train-noise", "--name", "smoke", "--steps", "6", "--width", "4", "--levels", "3", "--tile", "64", "--batch", "2")
    record = json.loads((store / "models" / "noise" / "smoke" / "run.json").read_text())
    assert record["config"]["train_pairs"] == 6
    assert record["export"]["max_relative_error"] < 1e-4
    assert record["inference"]["overlap"] >= receptive_field(3) // 2
    capsys.readouterr()
    run("evaluate-noise", "--name", "smoke", "--split", "train", "--limit", "2")
    measured = json.loads(capsys.readouterr().out)
    assert measured["pairs"] == 2
    size = next(iter(measured["by_size"].values()))
    assert size["high_ratio"] is not None and abs(size["sky_bias"]) < 1.0


def test_the_modelled_noise_map_keeps_the_level_and_drops_the_scatter():
    from seiza_training.noise_model import modelled_sigma

    rng = np.random.default_rng(4)
    height, width, frames, true_sigma = 256, 320, 4, 2.0
    # A per-pixel variance from four frames scatters as chi-square with
    # three degrees of freedom; the map is its square root over coverage.
    variance = true_sigma**2 * frames * rng.chisquare(frames - 1, (height, width)) / (frames - 1)
    coverage = np.full((height, width), frames)
    image = rng.normal(100.0, true_sigma, (height, width)).astype(np.float32)
    raw = np.sqrt(variance / coverage).astype(np.float32)
    modelled = modelled_sigma(image, raw, coverage)
    assert np.percentile(raw, 99) / np.percentile(raw, 1) > 5, "the per-pixel map scatters"
    assert np.percentile(modelled, 99) / np.percentile(modelled, 1) < 1.2, "the modelled one does not"
    assert abs(np.median(modelled) / true_sigma - 1.0) < 0.08, "and keeps the level"
