"""Multi-night deep stacks: every night of one object on one reference,
read through a source map, and halves that share no night."""

import json
import shutil

import numpy as np
import pytest

pytest.importorskip("astropy")

from seiza_training.__main__ import main  # noqa: E402
from seiza_training.manifest import Manifest, loads  # noqa: E402
from seiza_training.multi_night import deal, object_name  # noqa: E402
from seiza_training.synthetic import build_corpus  # noqa: E402


def test_an_object_is_its_sets_less_the_night_and_nights_are_dealt_evenly():
    assert object_name("Ultracat/California Nebula/2025-11-03/HA-300s-g100-bin1") == (
        "Ultracat/California Nebula/HA-300s-g100-bin1")
    a, b = deal({"n1": 70, "n2": 45, "n3": 40, "n4": 30})
    assert sorted(a + b) == ["n1", "n2", "n3", "n4"] and not set(a) & set(b)
    assert {tuple(a), tuple(b)} == {("n1", "n4"), ("n2", "n3")}


def test_every_night_of_an_object_stacks_onto_one_reference(tmp_path, capsys):
    corpus = build_corpus(tmp_path / "corpus", nights=2)
    store = tmp_path / "store"

    def run(*arguments):
        assert main(["--store", str(store), "--scratch", str(tmp_path / "scratch"), *arguments]) == 0
        return json.loads(capsys.readouterr().out)

    run("crawl", "--corpus", str(corpus / "telescopes"), "--library", str(corpus / "library"))
    for arguments in (("calibrate",), ("split",), ("process", "--catalog-dir", str(corpus / "catalogs"))):
        run(*arguments)
    listed = run("multi-night", "--list")["objects"]
    assert [(row["object"], row["nights"]) for row in listed] == [("Test Scope/Test Nebula/L-60s-g100-bin1", 2)]
    assert listed[0]["hours"] == pytest.approx(listed[0]["frames"] * 60 / 3600)

    # Stacked on a machine that mounts the corpus elsewhere.
    moved = tmp_path / "moved"
    shutil.move(corpus, moved)
    stacked = run("multi-night", "--min-hours", "0", "--source-map", f"{corpus}={moved}")["stacked"]
    summary = stacked["Test Scope/Test Nebula/L-60s-g100-bin1"]
    assert len(summary["nights"]) == 2 and summary["frames"] == sum(summary["nights"].values())

    with Manifest(store / "manifest.sqlite") as manifest:
        rows = {row["kind"]: dict(row) for row in manifest.rows("SELECT * FROM products WHERE id LIKE 'multi-night/%'")}
        nights = {row["id"]: [frame["path"] for frame in manifest.rows(
            "SELECT path FROM measurements WHERE set_id = ?", (row["id"],))] for row in manifest.rows("SELECT id FROM sets")}
    assert set(rows) == {"multi-night", "multi-night-half-a", "multi-night-half-b",
                         "multi-night-seeing-best", "multi-night-seeing-worst"}
    best, worst = loads(rows["multi-night-seeing-best"]["frames"]), loads(rows["multi-night-seeing-worst"]["frames"])
    seeing = loads(rows["multi-night-seeing-best"]["recipe"])
    assert len(best) == len(worst) == seeing["frames_each"] > 0 and not set(best) & set(worst)
    assert seeing["best_fwhm_px"] < seeing["worst_fwhm_px"]
    assert set(best) | set(worst) <= set(loads(rows["multi-night"]["frames"]))
    assert loads(rows["multi-night-seeing-best"]["stats"])["stack_fwhm_px"] < \
        loads(rows["multi-night-seeing-worst"]["stats"])["stack_fwhm_px"]
    whole = loads(rows["multi-night"]["frames"])
    a, b = loads(rows["multi-night-half-a"]["frames"]), loads(rows["multi-night-half-b"]["frames"])
    assert sorted(a + b) == sorted(whole) and not set(a) & set(b)
    assert all(path.startswith(str(corpus)) for path in whole), "the products record the manifest's paths"
    # Each half holds whole nights.
    for half in (a, b):
        assert sum(set(half) >= set(paths) for paths in nights.values()) == 1

    files = loads(rows["multi-night"]["files"])
    coverage = np.load(store / files["coverage"])
    variance = np.load(store / files["variance"])
    assert coverage.max() <= len(whole) and coverage.max() > max(len(a), len(b))
    assert np.isfinite(variance[coverage > 0]).all() and (variance[coverage > 0] > 0).mean() > 0.99
    assert loads(rows["multi-night"]["recipe"])["nights"] == sorted(nights)

    # Already stacked: left alone.
    assert run("multi-night", "--min-hours", "0", "--source-map", f"{corpus}={moved}")["skipped"] == [
        "Test Scope/Test Nebula/L-60s-g100-bin1"]
