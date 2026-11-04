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
            # The deepest noise pair, whose edges are covered, stands in for
            # a deep stack's halves.
            deepest = manifest.rows("SELECT id FROM products WHERE kind = 'noise-a' "
                                    "ORDER BY json_array_length(frames) DESC, id LIMIT 1")[0]["id"].removesuffix("/a")
            for half in ("a", "b"):
                row = dict(manifest.rows("SELECT * FROM products WHERE id = ?", (f"{deepest}/{half}",))[0])
                row.update(id=f"{row['set_id']}/deep-half/{half}", kind=f"deep-half-{half}",
                           recipe=json.dumps({"recipe": "deep-half", "half": half}))
                manifest.db.execute(f"INSERT INTO products ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                                    tuple(row.values()))
    capsys.readouterr()
    run("train-noise", "--name", "smoke", "--steps", "6", "--width", "4", "--levels", "3", "--tile", "64", "--batch", "2",
        "--precision", "bf16", "--prefetch", "2", "--batch-threads", "2", "--deep-fraction", "0.5",
        "--outliers", "2", "--outlier-floor", "10", "--clip-grad", "1", "--match-flux", "--agree", "0.9",
        "--outlier-scale", "local")
    record = json.loads((store / "models" / "noise" / "smoke" / "run.json").read_text())
    assert record["config"]["train_pairs"] == 6
    assert record["config"]["extra"]["label"] == "deep" and record["config"]["extra"]["windows"] > 0
    assert record["config"]["precision"] == "bf16"
    assert record["config"]["outliers"] == 2 and record["config"]["outlier_floor"] == 10
    assert record["config"]["clip_grad"] == 1 and record["config"]["match_flux"]
    assert record["config"]["agree"] == 0.9 and record["config"]["pairs_dropped"] == []
    assert record["config"]["outlier_scale"] == "local"
    assert all(entry["grad_norm_max"] > 0 and entry["grad_norm_mean"] > 0 for entry in record["history"])
    assert record["export"]["max_relative_error"] < 1e-4
    assert record["inference"]["overlap"] >= receptive_field(3) // 2
    capsys.readouterr()
    run("evaluate-noise", "--name", "smoke", "--split", "train")
    measured = json.loads(capsys.readouterr().out)
    assert measured["pairs"] == 7
    size = next(iter(measured["by_size"].values()))
    assert size["high_ratio"] is not None and abs(size["sky_bias"]) < 1.0
    assert list(measured["by_size"])[-1] == "deep-half"
    assert measured["by_size"]["deep-half"]["high_ratio"] is not None
    run("evaluate-noise", "--name", "smoke", "--split", "train", "--limit", "2", "--high-amount", "0.7",
        "--low-amount", "0.9")
    kept = json.loads(capsys.readouterr().out)
    assert kept["bands"] == {"high": 0.7, "low": 0.9, "scale_px": 5.0}
    run("evaluate-noise", "--name", "smoke", "--split", "train", "--limit", "2", "--match-flux")
    assert json.loads(capsys.readouterr().out)["match_flux"]
    evaluations = json.loads((store / "models" / "noise" / "smoke" / "run.json").read_text())["evaluations"]
    assert set(evaluations) == {"train", "train, high 0.7 low 0.9 at 5 px", "train, matched flux"}
    assert all(row["flux_gain"] is not None for row in evaluations["train, matched flux"]["pairs_detail"])
    assert all(row["flux_gain"] is None for row in evaluations["train"]["pairs_detail"])

    # Asked for halves that agree to within nothing, the bank leaves every
    # pair out, and says which.
    from seiza_training.noise_model import bank, pairs
    from seiza_training.store import Store

    with Manifest(store / "manifest.sqlite") as manifest:
        chosen = pairs(manifest, "train")
    strict = bank(Store(store, tmp_path / "scratch"), chosen, match_flux=True, agree=1e-9)
    assert len(strict["a"]) == 0 and sorted(strict["dropped"]) == sorted(p["a"]["id"] for p in chosen)
    with pytest.raises(ValueError, match="match_flux"):
        bank(Store(store, tmp_path / "scratch"), chosen, agree=0.1)

    # Contact sheets: one against a deeper reference, one showing the noise
    # taken out, and one whose input is not built.
    pytest.importorskip("PIL")
    with Manifest(store / "manifest.sqlite") as manifest:
        half = manifest.rows("SELECT * FROM products WHERE kind = 'noise-a' ORDER BY id LIMIT 1")[0]
        deep = manifest.rows("SELECT id FROM products WHERE kind = 'deep' AND set_id = ?", (half["set_id"],))[0]["id"]
    crops = [{"x": 8, "y": 8, "label": "corner"}, {"x": 10_000, "y": 20, "label": "past the edge"}]
    (store / "sheets").mkdir()
    (store / "sheets" / "noise.json").write_text(json.dumps({"sheets": [
        {"name": "half", "title": "A half", "input": half["id"], "reference": deep, "side": 48, "crops": crops},
        {"name": "deep", "title": "The deep stack", "input": deep, "side": 48, "crops": crops},
        {"name": "gone", "title": "Not built", "input": "nothing/deep", "side": 48, "crops": crops},
    ]}))
    run("sheet-noise", "--name", "smoke")
    sheets = json.loads(capsys.readouterr().out)
    assert set(sheets["made"]) == {"half", "deep"} and sheets["missing"] == ["gone"]
    png = (store / sheets["made"]["deep"]).read_bytes()
    assert png.startswith(b"\x89PNG")
    width, height = int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")
    assert width == 7 * 48 + 8 * 12 and height > 2 * 48
    assert json.loads((store / "models" / "noise" / "smoke" / "run.json").read_text())["sheets"] == sheets["made"]


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


def test_matching_puts_b_on_a_s_background_and_flux():
    from seiza_training.noise_model import matched

    rng = np.random.default_rng(3)
    y, x = np.mgrid[:256, :256]
    sky = np.zeros((256, 256))
    for cy, cx, height in rng.uniform((0, 0, 50), (256, 256, 3000), (60, 3)):
        sky += height * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * 2.0**2))
    sky += 40 * np.exp(-((y - 128) ** 2 + (x - 90) ** 2) / (2 * 30.0**2))
    a = (100 + sky + rng.normal(0, 2, sky.shape)).astype(np.float32)
    # B's normalization 3% low, its background lower and tilted.
    b = (95 + 0.02 * x + 0.97 * sky + rng.normal(0, 0.5, sky.shape)).astype(np.float32)
    a[:, :10] = np.nan
    fixed, gain = matched(a, b, 2.0)
    assert abs(gain * 0.97 - 1) < 0.003
    assert abs(np.mean((fixed - 100 - sky)[np.isfinite(a)])) < 0.05
    bright = sky > 40
    assert abs(np.nansum(fixed[bright] - 100) / np.sum(sky[bright]) - 1) < 0.005
    # As noisy as A, as a deep stack's other half is: the gain holds.
    noisy = (95 + 0.97 * sky + rng.normal(0, 2, sky.shape)).astype(np.float32)
    assert abs(matched(a, noisy, 2.0)[1] * 0.97 - 1) < 0.003
    # An empty sky gives no gain to fit, and only the background moves.
    fixed, gain = matched(np.full((64, 64), 10, np.float32), np.full((64, 64), 7, np.float32), 1.0)
    assert gain == 1.0 and np.allclose(fixed, 10)


def test_haze_in_one_half_shows_as_disagreement_and_a_gain_does_not():
    from seiza_training.noise_model import agreement, disagree, matched

    rng = np.random.default_rng(4)
    y, x = np.mgrid[:256, :256]
    stars, halos = np.zeros((256, 256)), np.zeros((256, 256))
    for cy, cx, height in rng.uniform((0, 0, 3000), (256, 256, 20000), (25, 3)):
        r2 = (y - cy) ** 2 + (x - cx) ** 2
        stars += height * np.exp(-r2 / (2 * 2.0**2))
        halos += height * 0.05 * np.exp(-r2 / (2 * 6.0**2))
    b = (100 + stars + rng.normal(0, 0.5, stars.shape)).astype(np.float32)
    # Only out of gain: matched, the halves agree.
    a = (100 + 1.04 * stars + rng.normal(0, 2, stars.shape)).astype(np.float32)
    found = agreement(a, matched(a, b, 2.0)[0], 2.0)
    assert not disagree(found, 0.02), found
    # Haze in A: its cores dimmed and their light spread wide.
    hazy = (100 + 0.95 * stars + halos + rng.normal(0, 2, stars.shape)).astype(np.float32)
    found = agreement(hazy, matched(hazy, b, 2.0)[0], 2.0)
    assert disagree(found, 0.05) and found["wings"] > 1 > found["cores"], found


def test_noise_bands_take_out_the_grain_and_the_mottle_apart():
    from seiza_training.noise_model import NoiseBands, gaussian_blur

    rng = np.random.default_rng(10)
    grain = rng.normal(size=(256, 256)).astype(np.float32)
    mottle = rng.normal(size=(256, 256)).astype(np.float32)
    for _ in range(3):
        mottle = gaussian_blur(mottle, 8.0)
    mottle *= 1.0 / mottle.std()
    image = (100.0 + grain + mottle).astype(np.float32)
    residual = 100.0 - image  # a model that finds all of the noise

    assert np.allclose(NoiseBands().apply(image, residual), 100.0, atol=1e-4)
    kept = NoiseBands(high=0.7, low=0.7).apply(image, residual) - 100.0
    assert kept.std() == pytest.approx(0.3 * (grain + mottle).std(), rel=1e-4)

    def left(bands):
        return bands.apply(image, residual) - 100.0

    # The grain alone kept: little of the mottle comes back.
    only_grain = left(NoiseBands(high=0.0, low=1.0))
    assert np.corrcoef(only_grain.ravel(), grain.ravel())[0, 1] > 0.95
    assert abs(np.corrcoef(only_grain.ravel(), mottle.ravel())[0, 1]) < 0.1
    # The mottle alone kept: little of the grain comes back.
    only_mottle = left(NoiseBands(high=1.0, low=0.0))
    assert np.corrcoef(only_mottle.ravel(), mottle.ravel())[0, 1] > 0.95
    with pytest.raises(ValueError):
        NoiseBands(high=1.5)


def test_prefetched_batches_are_the_ones_built_in_turn_and_errors_come_through():
    import time

    from seiza_training.noise_model import Prefetch, step_generator

    def make(step):
        # Uneven work, so threads finish out of order.
        time.sleep(0.002 * (step % 3))
        return step, step_generator(7, step).integers(0, 1000, 3)

    direct = [make(step) for step in range(1, 21)]
    for threads in (1, 4):
        source = Prefetch(make, 3, threads)
        fetched = [source.get() for _ in range(20)]
        source.close()
        assert [step for step, _ in fetched] == list(range(1, 21))
        assert all(np.array_equal(a, b) for (_, a), (_, b) in zip(direct, fetched))

    def failing(step):
        raise RuntimeError("bank gone")

    source = Prefetch(failing, 2, 2)
    with pytest.raises(RuntimeError, match="bank gone"):
        source.get()
    source.close()


def test_an_outlier_is_spread_as_registration_spreads_a_pixel():
    from seiza_training.noise_model import lanczos3, outliers

    np.testing.assert_allclose(lanczos3(0.0), [0, 0, 1, 0, 0, 0], atol=1e-12)
    assert abs(lanczos3(0.37).sum() - 1.0) < 1e-12
    assert not outliers((512, 512), np.random.default_rng(5), 0.0).any()
    plane = outliers((512, 512), np.random.default_rng(5), 8.0)
    assert plane.max() > 5 and plane.min() < 0
    assert (np.abs(plane) > 0.5).mean() < 0.01, "outliers are sparse"
    # The same draws from a higher floor: each hot pixel and track brighter,
    # and each pixel's Lanczos weights sum to 1.
    higher = outliers((512, 512), np.random.default_rng(5), 8.0, floor=20.0)
    assert np.array_equal(higher != 0, plane != 0)
    assert higher.sum() > plane.sum() + 8 * 15


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
    # Scaled to the noise at each pixel, an outlier where the noise is ten
    # times higher is ten times brighter; scaled to σ̄, it is not.
    banked["sigma"] = np.where(np.arange(WINDOW) < WINDOW // 2, 2.0, 20.0)[None, None, :] * np.ones_like(flat)
    for scale, low, high in (("local", 5.0, np.inf), ("typical", 0.5, 2.0)):
        _, _, references, sigmas = batch_from(banked, 1, WINDOW, np.random.default_rng(7), outlier_count=400,
                                              outlier_scale=scale)
        # The tile is turned and flipped; its noise map shows where the
        # noisier half went.
        planted, noisier = references[0, 0].abs(), sigmas[0, 0] > 5
        ratio = float(planted[noisier].max() / planted[~noisier].max())
        assert low < ratio < high, (scale, ratio)


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


def test_star_widths_show_an_estimate_that_softens_stars():
    from seiza_training.noise_model import against_truth, gaussian_blur

    rng = np.random.default_rng(12)
    stars = np.zeros((400, 400), dtype=np.float32)
    yy, xx = np.mgrid[0:400, 0:400]
    for y, x in rng.uniform(20, 380, (120, 2)):
        stars += rng.uniform(40, 400) * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.5**2))
    image = 100.0 + stars + rng.normal(0.0, 1.0, stars.shape).astype(np.float32)
    target = 100.0 + stars + rng.normal(0.0, 0.3, stars.shape).astype(np.float32)
    floor = np.full(stars.shape, 0.09, dtype=np.float32)

    def measured(estimate):
        return against_truth(estimate, image, target, floor, 100.0, 1.0)

    kept = measured(image)
    assert kept["star_fwhm_ratio_faint"] == pytest.approx(1.0)
    assert kept["star_fwhm_px"] == pytest.approx(2.3548 * 1.5, rel=0.1)
    # A clean estimate is as sharp as the noisy input, on average.
    assert measured(100.0 + stars)["star_fwhm_ratio_faint"] == pytest.approx(1.0, abs=0.03)
    softened = measured(gaussian_blur(image, 1.0))
    assert softened["star_fwhm_ratio_faint"] > 1.1 and softened["star_fwhm_target_faint"] > 1.1


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
