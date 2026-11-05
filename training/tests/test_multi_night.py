"""Multi-night deep stacks: every night of one object on one reference,
read through a source map, halves that share no night, and seeing pairs
judged on whatever else moves with seeing."""

import json
import shutil

import numpy as np
import pytest

import seiza

pytest.importorskip("astropy")

from seiza_training.__main__ import main  # noqa: E402
from seiza_training.manifest import Manifest, dumps, loads  # noqa: E402
from seiza_training.multi_night import (  # noqa: E402
    STACK_DETECTION,
    deal,
    gives_seeing_split,
    object_name,
    seeing_check,
)
from seiza_training.synthetic import build_corpus  # noqa: E402

HEIGHT, WIDTH = 432, 576


def test_an_object_is_its_sets_less_the_night_and_nights_are_dealt_evenly():
    assert object_name("Ultracat/California Nebula/2025-11-03/HA-300s-g100-bin1") == (
        "Ultracat/California Nebula/HA-300s-g100-bin1")
    a, b = deal({"n1": 70, "n2": 45, "n3": 40, "n4": 30})
    assert sorted(a + b) == ["n1", "n2", "n3", "n4"] and not set(a) & set(b)
    assert {tuple(a), tuple(b)} == {("n1", "n4"), ("n2", "n3")}


def half(sigma: float, *, seed: int, gain: float = 1.0, sky: float = 500.0, tilt: float = 0.0,
         stretch: float = 1.0, left_sigma: float | None = None, stars: bool = True):
    """A stack's image, coverage and measured stars: 108 stars on a jittered
    grid at Gaussian width `sigma` (`left_sigma` in the left third of the
    field), `stretch` times as wide as high, and a nebula wider than any
    seeing, all through `gain`, on a sky that climbs `tilt` across."""
    down, across = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    image = sky + tilt * across / WIDTH + gain * 300.0 * np.exp(
        -((across - 300) ** 2 + (down - 200) ** 2) / (2 * 70.0**2))
    place = np.random.default_rng(3)
    for row in range(9):
        for column in range(12):
            x = (column + 0.5) * WIDTH / 12 + place.uniform(-4, 4)
            y = (row + 0.5) * HEIGHT / 9 + place.uniform(-4, 4)
            flux = gain * place.uniform(20_000, 200_000)
            width = left_sigma if left_sigma is not None and x < WIDTH / 3 else sigma
            wide, high = width * stretch, width / stretch
            if stars:
                image = image + flux / (2 * np.pi * wide * high) * np.exp(
                    -((across - x) ** 2 / (2 * wide**2) + (down - y) ** 2 / (2 * high**2)))
    image = (image + np.random.default_rng(seed).normal(0.0, 2.0, image.shape)).astype(np.float32)
    found = seiza.detect_measured_stars(np.clip(image, 0, 65535).astype(np.uint16), **STACK_DETECTION)
    return image, np.ones(image.shape, dtype=np.uint16), found.stars


# How the worst half differs from the best, beyond its seeing (Gaussian
# width 2.2 against 1.5), and what the check makes of it.
CHANGES = {
    "the psf alone": ({}, None),
    "the same seeing": ({"sigma": 1.55}, "seeing-too-close"),
    "focus moved on one side": ({"left_sigma": 1.3}, "psf-field-mismatch"),
    "wind": ({"stretch": 1.3}, "psf-shape-mismatch"),
    "haze": ({"gain": 0.9}, "flux-mismatch"),
    "a brighter sky": ({"sky": 503.0}, "background-mismatch"),
    "a gradient": ({"tilt": 6.0}, "background-mismatch"),
    "no stars": ({"stars": False}, "psf-unmeasured"),
}


@pytest.fixture(scope="module")
def best_half():
    return half(1.5, seed=1)


@pytest.mark.parametrize(("change", "reason"), list(CHANGES.values()), ids=list(CHANGES))
def test_a_seeing_pair_is_kept_only_where_its_halves_differ_in_the_psf_alone(best_half, change, reason):
    status, found, stats = seeing_check(best_half, half(**{"sigma": 2.2, "seed": 2, **change}))
    assert (status, found) == ("built" if reason is None else "rejected", reason), stats
    if reason is None:
        # Measured unsmoothed, the stacks' widths keep their true ratio.
        assert stats["fwhm_ratio"] == pytest.approx(2.2 / 1.5, rel=0.06)
        assert stats["star_flux_ratio"] == pytest.approx(1.0, abs=0.01)
        assert stats["block_flux_ratio"] == pytest.approx(1.0, abs=0.01)
        assert all(ratio > 1.25 for row in stats["fwhm_ratio_field"] for ratio in row)


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
