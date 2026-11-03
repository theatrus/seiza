"""Overlapping-tile inference, a noise model trained end to end on the
synthetic corpus's noise pairs, the synthetic outliers training adds and
the signal transfer the clean targets measure."""

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


def test_an_outlier_is_spread_as_registration_spreads_a_pixel():
    from seiza_training.noise_model import lanczos3, outliers

    np.testing.assert_allclose(lanczos3(0.0), [0, 0, 1, 0, 0, 0], atol=1e-12)
    assert abs(lanczos3(0.37).sum() - 1.0) < 1e-12
    assert not outliers((512, 512), np.random.default_rng(5), 0.0).any()
    plane = outliers((512, 512), np.random.default_rng(5), 8.0)
    assert plane.max() > 5 and plane.min() < 0
    assert (np.abs(plane) > 0.5).mean() < 0.01, "outliers are sparse"


def test_outliers_reach_the_input_and_not_the_target():
    from seiza_training.noise_model import WINDOW, batch_from

    flat = np.full((2, WINDOW, WINDOW), 100.0, dtype=np.float32)
    banked = {"a": flat, "sigma": np.full_like(flat, 2.0), "b": flat,
              "background": np.array([100.0, 100.0], dtype=np.float32),
              "noise": np.array([2.0, 2.0], dtype=np.float32)}
    inputs, targets, references, _ = batch_from(banked, 4, 128, np.random.default_rng(6), outlier_count=40)
    assert not targets.any()
    assert references.abs().max() > 5
    assert torch.equal(inputs[:, 0] != 0, references[:, 0] != 0)
    _, _, references, _ = batch_from(banked, 4, 128, np.random.default_rng(6))
    assert not references.any()


def test_signal_transfer_keeps_the_input_whole_and_falls_as_an_estimate_smooths():
    from seiza_training.noise_model import against_truth, box_blur

    rng = np.random.default_rng(7)
    field = box_blur(rng.normal(size=(300, 300)).astype(np.float32), 1)
    structure = 8.0 + 4.0 * field / field.std()
    image = 100.0 + structure + rng.normal(0.0, 1.0, field.shape).astype(np.float32)
    target = 100.0 + structure + rng.normal(0.0, 0.5, field.shape).astype(np.float32)
    floor = np.full(field.shape, 0.25, dtype=np.float32)

    def transfer(estimate):
        return against_truth(estimate, image, target, floor, 100.0, 1.0)

    assert transfer(image)["transfer_faint_fine"] == pytest.approx(1.0)
    assert transfer(100.0 + structure)["transfer_faint_medium"] == pytest.approx(1.0, abs=0.05)
    blurred = transfer(box_blur(image, 2))
    assert blurred["transfer_faint_fine"] < 0.3 and 0.3 < blurred["transfer_faint_medium"] < 0.9
    assert transfer(np.full_like(image, 100.0))["transfer_faint_medium"] == pytest.approx(0.0)

    # A faint field smooth at the pixel scale has no fine signal to keep.
    smooth = rng.normal(size=field.shape).astype(np.float32)
    for _ in range(3):
        smooth = box_blur(smooth, 6)
    structure = 8.0 + 0.5 * smooth / smooth.std()
    image = 100.0 + structure + rng.normal(0.0, 1.0, field.shape).astype(np.float32)
    target = 100.0 + structure + rng.normal(0.0, 0.5, field.shape).astype(np.float32)
    assert transfer(image)["transfer_faint_fine"] is None
    assert transfer(image)["transfer_faint_coarse"] == pytest.approx(1.0)


def test_peak_flux_shows_faint_stars_an_estimate_drops():
    from seiza_training.noise_model import against_truth, box_blur

    rng = np.random.default_rng(9)
    stars = np.zeros((300, 300), dtype=np.float32)
    offsets = np.arange(-3, 4)
    psf = np.exp(-(offsets[:, None] ** 2 + offsets[None, :] ** 2) / (2 * 1.5**2))
    for y, x in rng.integers(20, 280, (40, 2)):
        stars[y - 3:y + 4, x - 3:x + 4] += 8.0 * psf
    image = 100.0 + stars + rng.normal(0.0, 1.0, stars.shape).astype(np.float32)
    target = 100.0 + stars + rng.normal(0.0, 0.5, stars.shape).astype(np.float32)
    floor = np.full(stars.shape, 0.25, dtype=np.float32)

    def measured(estimate):
        return against_truth(estimate, image, target, floor, 100.0, 1.0)

    assert measured(image)["peaks_faint"] >= 30
    assert measured(image)["peak_flux_kept_faint"] == pytest.approx(1.0)
    assert measured(100.0 + 0.5 * stars)["peak_flux_kept_faint"] == pytest.approx(0.5, abs=0.05)
    assert measured(np.full_like(image, 100.0))["peak_flux_kept_faint"] == pytest.approx(0.0)
    # Softened within the aperture, a peak keeps most of its flux and
    # loses height.
    softened = measured(box_blur(image, 1))
    assert softened["peak_flux_kept_faint"] > 0.9
    assert softened["peak_height_kept_faint"] < softened["peak_flux_kept_faint"] - 0.1
