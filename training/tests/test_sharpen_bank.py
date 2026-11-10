"""The PSF pass and the sharpening bank: fields measured once and read
back between their nodes, seeing pairs judged and matched, windows that
hold what the trainer needs, and a cut stopped part way that goes on to
the same bank."""

import json
import math
import sqlite3

import numpy as np
import pytest

seiza = pytest.importorskip("seiza")
pytest.importorskip("torch")
if not hasattr(seiza, "PsfField"):
    pytest.skip("this seiza has no PsfField", allow_module_level=True)

from seiza_training import psf_fields, sharpen_bank  # noqa: E402
from seiza_training.__main__ import main  # noqa: E402
from seiza_training.manifest import Manifest, dumps  # noqa: E402
from seiza_training.noise_model import WINDOW  # noqa: E402
from seiza_training.store import Store  # noqa: E402
from seiza_training.synthetic import build_corpus  # noqa: E402

SIDE = 3 * WINDOW
FRAMES = 16


def field(sigma: float, *, seed: int, gain: float = 1.0, sky: float = 500.0, stars: int = 12,
          stretch: float = 1.0, clip: float | None = None, left_sigma: float | None = None,
          shift: float = 0.0) -> np.ndarray:
    """A stack's image: `stars` × `stars` stars on a jittered grid at
    Gaussian width `sigma` (`left_sigma` in the left third), `stretch`
    times as wide as high, moved `shift` pixels down, through `gain`, on a
    flat sky with noise of 2; three far brighter than the rest are clipped
    at `clip`."""
    down, across = np.mgrid[0:SIDE, 0:SIDE].astype(np.float32)
    image = np.full((SIDE, SIDE), sky, dtype=np.float32)
    place = np.random.default_rng(3)
    for row in range(stars):
        for column in range(stars):
            x = (column + 0.5) * SIDE / stars + place.uniform(-4, 4)
            y = (row + 0.5) * SIDE / stars + place.uniform(-4, 4) + shift
            flux = place.uniform(20_000, 200_000)
            if (row, column) in ((0, 0), (0, 5), (5, 0)):
                flux = 4_000_000
            width = left_sigma if left_sigma is not None and x < SIDE / 3 else sigma
            wide, high = width * stretch, width / stretch
            image += gain * flux / (2 * np.pi * wide * high) * np.exp(
                -((across - x) ** 2 / (2 * wide**2) + (down - y) ** 2 / (2 * high**2)))
    image += np.random.default_rng(seed).normal(0.0, 2.0, image.shape).astype(np.float32)
    if clip is not None:
        image = np.minimum(image, clip)
    return image


def product(store: Store, product_id: str, image: np.ndarray, *, kind: str = "deep", split: str = "train",
            uncovered: slice | None = None, variance: bool = True) -> dict:
    """Write a product's planes into the store, as the build does, and
    return its row: a stack of `FRAMES` frames whose noise is 2, nothing
    covering the columns `uncovered`."""
    directory = store.products / product_id
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "image.npy", image.astype(np.float32))
    files = {"image": f"products/{product_id}/image.npy"}
    if variance:
        coverage = np.full(image.shape, FRAMES, dtype=np.uint16)
        if uncovered is not None:
            coverage[:, uncovered] = 0
        np.save(directory / "coverage.npy", coverage)
        np.save(directory / "variance.npy", np.full(image.shape, 4.0 * FRAMES, dtype=np.float32))
        files.update(coverage=f"products/{product_id}/coverage.npy", variance=f"products/{product_id}/variance.npy")
    return {"id": product_id, "kind": kind, "set_id": product_id.split("/")[0], "split": split,
            "telescope": "Test Scope", "pixel_scale_arcsec": 1.9, "files": json.dumps(files), "frames": "[]"}


@pytest.fixture()
def store(tmp_path):
    return Store(tmp_path / "store", tmp_path / "scratch")


def test_the_clip_level_needs_a_pile_up_as_the_field_s_does():
    assert psf_fields.clip_level(np.array([100.0, 60.0, 40.0, 30.0])) is None
    # Bright stars spread over a few percent are not a clip level.
    assert psf_fields.clip_level(np.array([100.0, 97.0, 94.0, 91.0, 88.0])) is None
    assert psf_fields.clip_level(np.array([70_000.0, 69_800.0, 69_500.0, 40_000.0])) == pytest.approx(0.8 * 69_800)
    # A hot pixel above the clipped cores does not hide them.
    assert psf_fields.clip_level(np.array([67_200.0, 62_090.0, 61_600.0, 61_470.0, 50_000.0])) == pytest.approx(
        0.8 * 61_600)


def test_a_field_is_recorded_on_a_grid_and_read_back_between_its_nodes(store):
    row = product(store, "a/deep", field(1.5, seed=1, left_sigma=2.2, clip=30_000.0))
    record = psf_fields.measure(store, row)
    assert record["status"] == "measured" and not record["is_global"] and record["star_count"] >= 30
    assert record == psf_fields.recorded(store, "a/deep")
    # The nodes span the image at most `GRID_PX` apart.
    assert record["nodes"] == [math.ceil((SIDE - 1) / psf_fields.GRID_PX) + 1] * 2
    assert record["stamp_sums"] == pytest.approx([1.0, 1.0], abs=1e-4)
    # The right of the field is 1.5 px wide, the left 2.2.
    loaded = psf_fields.Field(store, "a/deep")
    assert loaded.shape_at(0.85 * SIDE, SIDE / 2)["fwhm"] == pytest.approx(2.3548 * 1.5, rel=0.1)
    assert loaded.shape_at(0.1 * SIDE, SIDE / 2)["fwhm"] == pytest.approx(2.3548 * 2.2, rel=0.1)
    # At a node the stamp is the node's; between nodes a weighted mean of
    # the four about it, still summing to one.
    xs, ys = loaded.arrays["xs"], loaded.arrays["ys"]
    assert np.array_equal(loaded.stamp_at(float(xs[1]), float(ys[2])), loaded.arrays["stamps"][2, 1])
    between = loaded.stamp_at(float(xs[1] + xs[2]) / 2, float(ys[2]))
    assert np.allclose(between, (loaded.arrays["stamps"][2, 1] + loaded.arrays["stamps"][2, 2]) / 2, atol=1e-7)
    assert float(between.sum()) == pytest.approx(1.0, abs=1e-5)
    # The detector finds the stars, and the three clipped ones mark the
    # clip level.
    points = loaded.points()
    assert len(points) == pytest.approx(144, abs=6)
    assert record["clip_level"] == pytest.approx(0.8 * 30_000.0, rel=0.01)
    clipped = points[points[:, psf_fields.DETECTION_COLUMNS.index("clipped")] == 1]
    assert len(clipped) == 3


def test_a_field_with_too_few_stars_is_one_psf_or_refused(store):
    few = psf_fields.measure(store, product(store, "few/deep", field(1.5, seed=1, stars=4)))
    assert few["status"] == "global" and few["is_global"] and "stars" in few["error"].lower()
    assert few["stamp_sums"] == pytest.approx([1.0, 1.0], abs=1e-4)
    empty = product(store, "none/deep", np.random.default_rng(1).normal(500.0, 2.0, (SIDE, SIDE)).astype(np.float32))
    none = psf_fields.measure(store, empty)
    assert none["status"] == "refused" and not psf_fields.Field(store, "none/deep").usable
    left = sharpen_bank.prepared(str(store.root), str(store.scratch), 0, 4, {"input": empty, "record": {}})
    assert left["status"] == "dropped" and left["record"]["reason"] == "psf-refused"


def seeing_pair(store: Store, name: str, **worst) -> dict:
    """A seeing pair's halves in the store, measured, as a bank source: the
    best at Gaussian width 1.5, the worst at 2.2 changed by `worst`, its
    gain and sky those of the halves' normalization, and the edge columns
    of the worst uncovered."""
    best = product(store, f"{name}/seeing/best", field(1.5, seed=1, clip=30_000.0), kind="seeing-best")
    worst = product(store, f"{name}/seeing/worst", field(**{"sigma": 2.2, "seed": 2, "clip": 30_000.0, **worst}),
                    kind="seeing-worst", uncovered=slice(0, 40))
    for row in (best, worst):
        psf_fields.measure(store, row)
    return {"input": worst, "target": best, "match": True, "gain": None, "judge": True, "record": {"pair": name}}


def test_a_seeing_pair_is_judged_matched_and_cut_with_both_psfs(store, tmp_path):
    source = seeing_pair(store, "s", gain=0.9, sky=503.0)
    banked = sharpen_bank.cut(store, [source], label="seeing", split="train", work=tmp_path / "work", per_source=6)
    assert banked["done"]["windows"] == 6 and banked["done"]["dropped"] == []
    record = banked["sources"][0]
    assert record["record"]["pair"] == "s" and record["record"]["check"]["fwhm_ratio"] == pytest.approx(2.2 / 1.5,
                                                                                                      rel=0.06)
    # The target is put on the input's flux and sky: the check's gain, the
    # best half's flux over the worst's, divides it.
    assert record["match"]["gain"] == pytest.approx(1 / 0.9, rel=0.01)
    assert record["match"]["background_offset"] == pytest.approx(-3.0, abs=0.3)
    assert record["alignment"]["matched"] > 100 and math.hypot(*record["alignment"]["offset_px"]) < 0.05
    info = banked["info"]
    column = {name: index for index, name in enumerate(sharpen_bank.PAIR_INFO)}
    assert np.allclose(info[:, column["ratio"]], 1.5 / 2.2, rtol=0.08)
    assert np.allclose(info[:, column["target_fwhm"]], 2.3548 * 1.5, rtol=0.1)
    # No window reaches the uncovered edge.
    assert (info[:, column["x"]] >= 40).all()
    for stamps in ("input_stamp", "target_stamp"):
        assert banked[stamps].shape[1:] == (sharpen_bank.STAMP, sharpen_bank.STAMP)
        assert np.allclose(banked[stamps].sum(axis=(1, 2)), 1.0, atol=1e-4)
    # Matched, the halves' skies and stars' flux agree, over the windows
    # with no saturated core.
    sky = banked["input"] < 520
    assert float(np.median(banked["target"][sky] - banked["input"][sky])) == pytest.approx(0.0, abs=0.2)
    clear = ~banked["saturated"].any(axis=(1, 2))
    assert clear.sum() >= 2
    ratio = (banked["target"][clear] - 503.0).sum() / (banked["input"][clear] - 503.0).sum()
    assert float(ratio) == pytest.approx(1.0, abs=0.03)
    # The maps: the stacks' own noise, the image's, and the saturated cores
    # of either.
    assert np.allclose(banked["input_sigma"], 2.0) and np.allclose(banked["target_sigma"], 2.0 * 0.9, rtol=0.01)
    assert float(np.median(banked["input_map"])) == pytest.approx(2.0 * 1.13, rel=0.15)
    assert banked["saturated"].dtype == np.uint8
    assert ((banked["input"] >= 0.8 * 30_000) <= (banked["saturated"] == 1)).all()
    assert len(np.load(banked["directory"] / "points" / "0.npy")) == pytest.approx(144, abs=6)
    # The stamps' own widths give the ratio too.
    assert np.allclose(info[:, column["target_width"]], 2.3548 * 1.5, rtol=0.1)
    assert np.allclose(info[:, column["width_ratio"]], 1.5 / 2.2, rtol=0.08)


@pytest.mark.parametrize("fwhm", [1.2, 1.8, 3.0, 8.0, 16.0])
def test_a_stamp_s_width_is_its_core_s_whether_sharp_or_wide(fwhm):
    middle = (sharpen_bank.STAMP - 1) / 2
    down, across = np.mgrid[0:sharpen_bank.STAMP, 0:sharpen_bank.STAMP]
    squared = (across - middle) ** 2 + (down - middle) ** 2
    gaussian = np.exp(-squared / (2 * (fwhm / 2.3548) ** 2))
    assert sharpen_bank.stamp_width(gaussian / gaussian.sum()) == pytest.approx(fwhm, rel=0.02)
    # A Moffat's wings, of β 2.5, move it little: most where a stamp is
    # sharp enough that its core is a few pixels.
    alpha = fwhm / (2 * math.sqrt(2 ** (1 / 2.5) - 1))
    moffat = (1 + squared / alpha**2) ** -2.5
    assert sharpen_bank.stamp_width(moffat / moffat.sum()) == pytest.approx(fwhm, rel=0.15)
    assert math.isnan(sharpen_bank.stamp_width(np.zeros((5, 5))))


@pytest.mark.parametrize(("change", "reason"), [({"stretch": 1.3}, "psf-shape-mismatch"),
                                                ({"sigma": 1.55}, "seeing-too-close")])
def test_a_seeing_pair_that_differs_in_more_than_its_psf_is_left_out(store, tmp_path, change, reason):
    source = seeing_pair(store, "w", **change)
    banked = sharpen_bank.cut(store, [source, seeing_pair(store, "s")], label="seeing", split="train",
                              work=tmp_path / "work", per_source=2)
    assert [left["reason"] for left in banked["done"]["dropped"]] == [reason]
    assert [record["record"]["pair"] for record in banked["sources"]] == ["s"]


def test_a_caller_s_pairs_keep_their_record_and_need_no_variance(store, tmp_path):
    source = product(store, "c/deep", field(2.2, seed=2))
    target = product(store, "c/made/one", field(1.5, seed=1), kind="made", variance=False)
    for row in (source, target):
        psf_fields.measure(store, row)
    banked = sharpen_bank.cut(store, [{"input": source, "target": target, "record": {"setting": "one", "n": 1}}],
                              label="extra", split="train", work=tmp_path / "work", per_source=3)
    assert banked["sources"][0]["record"] == {"setting": "one", "n": 1}
    assert banked["sources"][0]["target_sigma"] is False and np.isnan(banked["target_sigma"]).all()
    assert "match" not in banked["sources"][0]
    assert np.isfinite(banked["target"]).all() and len(banked["info"]) == 3


def test_a_pair_s_centroids_may_move_a_little_but_not_off_its_grid(store, tmp_path):
    source = product(store, "m/deep", field(2.2, seed=2))
    near = product(store, "m/made/near", field(1.5, seed=1, shift=0.6), kind="made", variance=False)
    far = product(store, "m/made/far", field(1.5, seed=1, shift=2.5), kind="made", variance=False)
    for row in (source, near, far):
        psf_fields.measure(store, row)
    banked = sharpen_bank.cut(store, [{"input": source, "target": target, "record": {}} for target in (near, far)],
                              label="extra", split="train", work=tmp_path / "work", per_source=2)
    # A lopsided PSF moves its centroid as far, and a pair keeps its offset.
    assert banked["sources"][0]["alignment"]["offset_px"][1] == pytest.approx(0.6, abs=0.05)
    assert [left["reason"] for left in banked["done"]["dropped"]] == ["misaligned"]


def test_a_sharpening_bank_cut_stopped_part_way_goes_on_to_the_same_bank(store, tmp_path, monkeypatch):
    sources = []
    for index in range(4):
        row = product(store, f"d{index}/deep", field(1.5 + 0.2 * index, seed=index), uncovered=slice(0, 30 * index))
        psf_fields.measure(store, row)
        sources.append({"input": row, "record": {}})
    sources.append({"input": product(store, "none/deep", np.random.default_rng(1).normal(
        500.0, 2.0, (SIDE, SIDE)).astype(np.float32)), "record": {}})
    psf_fields.measure(store, sources[-1]["input"])

    def cut(work, workers=1):
        return sharpen_bank.cut(store, sources, label="deep", split="train", work=tmp_path / work, per_source=5,
                                workers=workers)

    alone = cut("alone")
    assert alone["done"]["windows"] == 20 and alone["done"]["sources"] == 4
    assert [left["reason"] for left in alone["done"]["dropped"]] == ["psf-refused"]
    assert np.allclose(alone["stamp"].sum(axis=(1, 2)), 1.0, atol=1e-4)
    # Read and prepared in processes of their own, the bank is the same.
    shared = cut("shared", workers=2)
    prepare, calls = sharpen_bank.prepared, []

    def stopping(*arguments, **options):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return prepare(*arguments, **options)

    monkeypatch.setattr(sharpen_bank, "CHECKPOINT_SECONDS", 0.0)
    monkeypatch.setattr(sharpen_bank, "prepared", stopping)
    with pytest.raises(KeyboardInterrupt):
        cut("stopped")
    monkeypatch.setattr(sharpen_bank, "prepared", prepare)
    resumed = cut("stopped")
    assert len(calls) == 3, "the two sources cut before the stop are not cut again"
    for banked in (shared, resumed):
        for plane in ("image", "sigma", "saturated", "stamp", "info"):
            assert np.array_equal(alone[plane], banked[plane]), plane
        assert banked["sources"] == alone["sources"]
        assert banked["done"]["dropped"] == alone["done"]["dropped"]
        for index in range(4):
            name = f"points/{index}.npy"
            assert np.array_equal(np.load(alone["directory"] / name), np.load(banked["directory"] / name))
    assert not list((tmp_path / "stopped").rglob("progress.json")), "a finished cut leaves no record behind"
    # Finished, it is opened as it was, not cut again.
    monkeypatch.setattr(sharpen_bank, "prepared", stopping)
    assert np.array_equal(cut("stopped")["image"], alone["image"]) and len(calls) == 3


def test_the_passes_measure_and_cut_the_corpus_reading_its_manifest_alone(tmp_path, capsys):
    corpus = build_corpus(tmp_path / "corpus", nights=2, vignetted=True)
    store = tmp_path / "store"

    def run(*arguments):
        assert main(["--store", str(store), "--scratch", str(tmp_path / "scratch"), *arguments]) == 0
        return json.loads(capsys.readouterr().out)

    run("crawl", "--corpus", str(corpus / "telescopes"), "--library", str(corpus / "library"))
    for arguments in (("calibrate",), ("split",), ("process", "--catalog-dir", str(corpus / "catalogs")),
                      ("multi-night", "--seeing")):
        run(*arguments)
    with Manifest(store / "manifest.sqlite") as manifest, manifest.transaction():
        nights = [row["id"] for row in manifest.rows("SELECT id FROM sets ORDER BY name")]
        # Both nights train, as an object's nights share their split; the
        # multi-night pair is the first night's, as it was judged.
        manifest.db.execute("UPDATE sets SET split = 'train'")
    database = store / "manifest.sqlite"
    before = (database.read_bytes(), database.stat().st_mtime_ns)

    measured = run("psf-fields", "--workers", "2")
    kinds = {"deep": 2, "multi-night": 1, "seeing-best": 2, "seeing-worst": 2, "multi-night-seeing-best": 1,
             "multi-night-seeing-worst": 1}
    assert measured["products"] == sum(kinds.values()) and measured["skipped"] == 0
    assert set(measured["status"]) <= {"measured", "global"} and sum(measured["status"].values()) == 9
    assert {name.split(":")[0] for name in measured["fields"]} == set(kinds)
    # Measured again, each product is skipped.
    assert run("psf-fields")["skipped"] == sum(kinds.values())

    banks = run("cut-sharpen", "--workers", "2", "--deep-windows", "2", "--pair-windows", "2", "--split", "train")
    deep, seeing = banks["deep:train"], banks["seeing:train"]
    assert deep["sources"] == 3 and deep["windows"] > 0 and deep["dropped"] == []
    assert seeing["left"] == []
    found = {record["record"]["pair"]: record["record"]["judged"] for record in seeing_sources(seeing)}
    pair = next(name for name in found if name.startswith("multi-night/"))
    assert found.pop(pair) == "manifest" and set(found.values()) <= {"bank"}
    # A night's pairs carry no check, so the bank judges them.
    assert set(found) | {left["record"]["pair"] for left in seeing["dropped"]} == {
        f"{night}/seeing" for night in nights}
    # Neither pass wrote to the manifest.
    assert (database.read_bytes(), database.stat().st_mtime_ns) == before
    with pytest.raises(sqlite3.OperationalError):
        Manifest.read_only(database).db.execute("UPDATE sets SET split = 'test'")
    # A pair the manifest marks rejected is left out, with its reason.
    with Manifest(database) as manifest, manifest.transaction():
        manifest.db.execute("UPDATE products SET status = 'rejected', reason = 'psf-shape-mismatch' "
                            "WHERE kind LIKE 'multi-night-seeing-%'")
    again = run("cut-sharpen", "--bank", "seeing", "--split", "train", "--pair-windows", "2")["seeing:train"]
    assert {"pair": pair, "reason": "manifest:psf-shape-mismatch"} in again["left"]
    assert dumps(again) != dumps(seeing)


def seeing_sources(banked: dict) -> list[dict]:
    """A bank's source records, from its directory."""
    from pathlib import Path

    return json.loads((Path(banked["directory"]) / "sources.json").read_text())


def test_external_windows_may_miss_a_few_pixels_which_are_filled():
    import random

    from seiza_training import sharpen_external
    from seiza_training.noise_model import WINDOW
    from seiza_training.sharpen_bank import window_corners

    rng = np.random.default_rng(0)
    image = (10.0 + rng.standard_normal((WINDOW + 200, WINDOW + 200))).astype(np.float32)
    image[250:253, 250:254] = np.nan
    finite = np.isfinite(image)
    assert window_corners(finite, random.Random(0), 5) == []
    assert window_corners(finite, random.Random(0), 5, sharpen_external.HOLES)
    filled = sharpen_external.filled_holes(image)
    assert np.isfinite(filled).all()
    assert np.array_equal(filled[finite], image[finite])
    assert abs(float(filled[~finite].mean()) - 10.0) < 1.0
