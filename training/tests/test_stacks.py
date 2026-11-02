"""The stacks pass on the synthetic corpus, with a PSF Guard stacks folder
and a scheduler database laid out as PSF Guard and the scheduler keep them."""

import json
import sqlite3

import numpy as np
import pytest
from astropy.io import fits

from seiza_training.__main__ import main
from seiza_training.manifest import Manifest, loads
from seiza_training.synthetic import FRAMES, LIGHTS, build_corpus


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = build_corpus(tmp_path_factory.mktemp("corpus"))
    scheduler = root / "telescopes" / "Test Scope" / "_Scheduler"
    scheduler.mkdir()
    db = sqlite3.connect(scheduler / "schedulerdb.sqlite")
    db.execute("CREATE TABLE target (Id INTEGER PRIMARY KEY, name TEXT)")
    db.execute("CREATE TABLE acquiredimage (Id INTEGER PRIMARY KEY, metadata TEXT)")
    db.execute("INSERT INTO target VALUES (7, 'Test Nebula')")
    for index in range(FRAMES):
        name = f"D:\\pictures\\Test Nebula\\2026-01-09\\LIGHT\\light_{index:03d}.fits"
        db.execute("INSERT INTO acquiredimage VALUES (?, ?)", (100 + index, json.dumps({"FileName": name})))
    db.commit()
    db.close()

    rng = np.random.default_rng(3)
    image = rng.normal(500.0, 5.0, (60, 80)).astype(np.float32)
    job = root / "psf-guard" / "db-test" / "stack-previews" / ("ab" * 32)
    job.mkdir(parents=True)
    fits.PrimaryHDU(image).writeto(job / "group-0.fits")
    (job / "manifest.json").write_text(json.dumps({
        "job_id": "ab" * 32, "artifact_revision": "r1", "stacking_version": "0.48.1",
        "method": {"normalization": "local_background"},
        "groups": [
            {"index": 0, "state": "ready", "target_id": 7, "target_name": "Test Nebula",
             "filter_name": "L", "exposure_group": {"max_seconds": 60.0}, "accepted_frames": 40,
             "total_exposure_seconds": 2400.0,
             "frames": [{"image_id": 100 + index, "disposition": "accepted"} for index in range(40)]
             + [{"image_id": 999, "disposition": "accepted"}]},
            {"index": 1, "state": "failed", "target_id": 7, "target_name": "Test Nebula", "frames": []},
        ],
    }))
    # A database whose targets no scheduler under the corpus names.
    other = root / "psf-guard" / "db-other" / "stack-previews" / ("cd" * 32)
    other.mkdir(parents=True)
    (other / "manifest.json").write_text(json.dumps({
        "job_id": "cd" * 32, "artifact_revision": "r1",
        "groups": [{"index": 0, "state": "ready", "target_id": 3, "target_name": "Elsewhere", "frames": []}],
    }))
    fits.PrimaryHDU(image).writeto(other / "group-0.fits")
    return root, image


def run(store, *arguments):
    assert main(["--store", str(store), "--scratch", str(store.parent / "scratch"), *arguments]) == 0


def test_a_psf_guard_stack_is_copied_and_joins_its_sets_split(corpus, tmp_path, capsys):
    root, image = corpus
    store = tmp_path / "store"
    run(store, "crawl", "--corpus", str(root / "telescopes"), "--library", str(root / "library"))
    run(store, "stacks", "--psf-guard", str(root / "psf-guard"))
    run(store, "split")
    with Manifest(store / "manifest.sqlite") as manifest:
        set_row = manifest.rows("SELECT * FROM sets")[0]
        stacks = manifest.rows("SELECT * FROM stacks")
        assert len(stacks) == 1, "the failed group and the unmatched database are left out"
        stack = stacks[0]
        assert stack["id"] == f"psf-guard/db-test/{'ab' * 32}/0"
        assert (stack["telescope"], stack["target"], stack["status"]) == ("Test Scope", "Test Nebula", "copied")
        assert loads(stack["sets"]) == [set_row["id"]]
        assert (stack["sky_group"], stack["split"]) == (set_row["sky_group"], set_row["split"])
        product = manifest.rows("SELECT * FROM products WHERE set_id = ?", (stack["id"],))[0]
        assert product["kind"] == "stack"
        assert len(loads(product["frames"])) == 40, "image 999 has no file in the corpus"
        assert all("/" + str(LIGHTS) + "/" in path for path in loads(product["frames"]))
        np.testing.assert_array_equal(np.load(store / loads(product["files"])["image"]), image)

    capsys.readouterr()
    run(store, "stacks")
    summary = json.loads(capsys.readouterr().out)
    assert (summary["copied"], summary["kept"]) == (0, 1), "the folder is remembered and a copy kept"
    assert summary["databases"] == {"db-other": None, "db-test": "Test Scope"}


def test_a_held_out_telescopes_stacks_are_held_out(corpus, tmp_path):
    root, _ = corpus
    store = tmp_path / "store"
    run(store, "crawl", "--corpus", str(root / "telescopes"), "--library", str(root / "library"))
    run(store, "stacks", "--psf-guard", str(root / "psf-guard"))
    run(store, "split", "--hold-out", "Test Scope")
    with Manifest(store / "manifest.sqlite") as manifest:
        assert manifest.rows("SELECT split FROM stacks")[0][0] == "held-out"
