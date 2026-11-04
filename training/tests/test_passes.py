"""Every pass, end to end, on the small synthetic corpus."""

import json
from datetime import timedelta

import numpy as np
import pytest

from seiza_training.__main__ import main
from seiza_training.manifest import Manifest, loads
from seiza_training.synthetic import FRAMES, LIGHTS, START, build_corpus, common_header, star_field, write


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return build_corpus(tmp_path_factory.mktemp("corpus"))


def run(store, *arguments):
    assert main(["--store", str(store), "--scratch", str(store.parent / "scratch"), *arguments]) == 0


def test_every_pass_builds_and_records_a_set(corpus, tmp_path, capsys):
    store = tmp_path / "store"
    run(store, "crawl", "--corpus", str(corpus / "telescopes"), "--library", str(corpus / "library"))
    with Manifest(store / "manifest.sqlite") as manifest:
        sets = manifest.rows("SELECT * FROM sets")
        assert len(sets) == 1
        assert sets[0]["frames"] == FRAMES
        assert sets[0]["filter"] == "L"
        masters = manifest.rows("SELECT role FROM frames WHERE origin = 'library' AND is_master = 1")
        assert sorted(row["role"] for row in masters) == ["bias", "flat"]
        set_id = sets[0]["id"]

    run(store, "calibrate")
    run(store, "measure", "--catalog-dir", str(corpus / "catalogs"))
    run(store, "split")
    with Manifest(store / "manifest.sqlite") as manifest:
        row = manifest.rows("SELECT * FROM sets")[0]
        assert row["calibration_status"] == "ready", row["calibration"]
        calibration = loads(row["calibration"])
        assert calibration["masters"]["flat"].endswith("flat-test.fits")
        assert calibration["masters"]["bias"].endswith("bias-test.fits")
        measurement = loads(row["measurement"])
        assert measurement["accepted"] == FRAMES, measurement
        assert set(measurement["recipes"]) == {"deep", "noise-pairs", "seeing-split"}
        assert row["split"] in ("train", "validation", "test")

    run(store, "build")
    run(store, "review")
    capsys.readouterr()
    with Manifest(store / "manifest.sqlite") as manifest:
        row = manifest.rows("SELECT * FROM sets")[0]
        assert row["build_status"] == "built", row["build_reason"]
        products = {product["id"]: product for product in manifest.rows("SELECT * FROM products")}
        kinds = sorted(product["kind"] for product in products.values())
        # Sizes 4 and 8 fit a fifth of 45 frames, three draws each.
        assert kinds.count("noise-a") == kinds.count("noise-b") == 6
        assert kinds.count("deep") == kinds.count("seeing-best") == kinds.count("seeing-worst") == 1
        assert len(kinds) == 15

        deep = products[f"{set_id}/deep"]
        admitted = loads(deep["frames"])
        assert len(admitted) == FRAMES
        shape = np.load(store / loads(deep["files"])["image"], mmap_mode="r").shape
        for product in products.values():
            files = loads(product["files"])
            image = np.load(store / files["image"], mmap_mode="r")
            assert image.shape == shape, product["id"]
            coverage = np.load(store / files["coverage"], mmap_mode="r")
            assert coverage.dtype == np.uint16
            assert int(coverage.max()) <= len(loads(product["frames"]))

        a = products[f"{set_id}/noise/n08-d0/a"]
        b = products[f"{set_id}/noise/n08-d0/b"]
        frames_a, frames_b = loads(a["frames"]), loads(b["frames"])
        assert len(frames_a) == 8 and len(frames_b) == FRAMES - 8
        assert not set(frames_a) & set(frames_b)
        assert "fits" not in loads(a["files"]) and "fits" not in loads(b["files"])
        check = loads(a["recipe"])["check"]
        assert check["difference_sigma"] > 0

        best = loads(products[f"{set_id}/seeing/best"]["recipe"])
        assert best["frames_each"] == FRAMES // 4
        assert best["worst_fwhm_px"] > 1.25 * best["best_fwhm_px"]

    assert not any((store.parent / "scratch").iterdir()), "registered frames are deleted"
    page = store / "review" / set_id / "index.html"
    assert page.exists()
    assert (store / "review" / set_id / "deep.png").exists()
    assert set_id in (store / "review" / "index.html").read_text()

    run(store, "decide", set_id, "set", "accept", "--reviewer", "test")
    run(store, "status")
    status = json.loads(capsys.readouterr().out)
    assert status["reviews"] == {"set:accept": 1}
    assert status["build"] == {"built": 1}


def test_process_measures_and_builds_set_by_set(corpus, tmp_path, capsys):
    store = tmp_path / "store"
    run(store, "crawl", "--corpus", str(corpus / "telescopes"), "--library", str(corpus / "library"))
    run(store, "calibrate")
    run(store, "split", "--hold-out", "Test Scope")
    run(store, "process", "--catalog-dir", str(corpus / "catalogs"))
    capsys.readouterr()
    run(store, "process")
    assert json.loads(capsys.readouterr().out)["built"] == 0, "a built set is not built again"
    with Manifest(store / "manifest.sqlite") as manifest:
        row = manifest.rows("SELECT * FROM sets")[0]
        assert row["split"] == "held-out"
        assert row["measurement_status"] == "measured"
        assert row["build_status"] == "built", row["build_reason"]
        assert manifest.rows("SELECT COUNT(*) FROM products WHERE kind = 'deep'")[0][0] == 1


def test_a_set_built_before_it_qualified_gets_its_noise_pairs_alone(corpus, tmp_path, capsys, monkeypatch):
    from seiza_training import measure

    store = tmp_path / "store"
    run(store, "crawl", "--corpus", str(corpus / "telescopes"), "--library", str(corpus / "library"))
    run(store, "calibrate")
    run(store, "split")
    # Measured while pairs asked for more frames than the set has.
    monkeypatch.setattr(measure, "NOISE_PAIR_FRAMES", FRAMES + 1)
    run(store, "process", "--catalog-dir", str(corpus / "catalogs"))
    with Manifest(store / "manifest.sqlite") as manifest:
        assert not manifest.rows("SELECT 1 FROM products WHERE kind = 'noise-a'")
        before = {row["id"]: dict(row) for row in manifest.rows("SELECT * FROM products")}
    monkeypatch.undo()
    capsys.readouterr()
    run(store, "recipes", "--dry-run")
    assert json.loads(capsys.readouterr().out)["gained"] == {"noise-pairs": 1}
    run(store, "recipes")
    capsys.readouterr()
    run(store, "build", "--pairs-only")
    assert json.loads(capsys.readouterr().out)["built"] == 1
    run(store, "build", "--pairs-only")
    assert json.loads(capsys.readouterr().out)["built"] == 0, "a set with pairs is passed by"
    with Manifest(store / "manifest.sqlite") as manifest:
        after = {row["id"]: dict(row) for row in manifest.rows("SELECT * FROM products")}
        row = manifest.rows("SELECT * FROM sets")[0]
    # Sizes 4 and 8, three draws each; the deep stack and the seeing split
    # as they were.
    assert sorted(after[key]["kind"] for key in after.keys() - before.keys()).count("noise-a") == 6
    assert all(after[key] == value for key, value in before.items())
    assert row["build_status"] == "built" and "pairs_later" in loads(row["build_reason"])


def test_a_turned_camera_registers_on_its_larger_orientation(corpus, tmp_path):
    # Ten sharp frames turned 15 degrees, which the reference choice favours,
    # and 35 softer frames as the camera was for most of the night.
    rng = np.random.default_rng(9)
    lights = tmp_path / LIGHTS / "2026-01-11" / "LIGHT"
    for index in range(FRAMES):
        header = common_header("LIGHT", START + timedelta(days=2, minutes=2 * index), 60.0)
        header["FILTER"] = "L"
        header["RA"] = 83.8
        header["DEC"] = -5.4
        turned = index < 10
        sigma = 1.6 if turned else 2.2
        shift = (rng.uniform(-3, 3), rng.uniform(-3, 3))
        image = star_field(rng, sigma, shift, 15.0 if turned else 0.0) + 30.0
        write(lights / f"light_{index:03d}.fits", image, header)
    store = tmp_path / "store"
    run(store, "crawl", "--corpus", str(tmp_path / "telescopes"), "--library", str(corpus / "library"))
    run(store, "calibrate")
    run(store, "split")
    run(store, "process", "--catalog-dir", str(corpus / "catalogs"))
    with Manifest(store / "manifest.sqlite") as manifest:
        row = manifest.rows("SELECT * FROM sets")[0]
        assert row["build_status"] == "built", row["build_reason"]
        summary = loads(row["build_reason"])
        assert summary["reference_retried"]["rotated_away"] >= 30
        assert int(summary["reference"].rsplit("_", 1)[1].split(".")[0]) >= 10
        assert summary["admitted"] >= 30
