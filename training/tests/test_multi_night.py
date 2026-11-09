"""Multi-night deep stacks: every night of one object on one reference,
read through a source map, halves that share no night, and seeing pairs
judged on whatever else moves with seeing."""

import json
import shutil

import numpy as np
import pytest

pytest.importorskip("astropy")

from seiza_training.__main__ import main  # noqa: E402
from seiza_training.manifest import Manifest, dumps, loads  # noqa: E402
from seiza_training.multi_night import (  # noqa: E402
    deal,
    gives_seeing_split,
    object_name,
    seeing_check,
    stack_psf,
)
from seiza_training.synthetic import build_corpus  # noqa: E402

HEIGHT, WIDTH = 432, 576


def test_an_object_is_its_sets_less_the_night_and_nights_are_dealt_evenly():
    assert object_name("Ultracat/California Nebula/2025-11-03/HA-300s-g100-bin1") == (
        "Ultracat/California Nebula/HA-300s-g100-bin1")
    a, b = deal({"n1": 70, "n2": 45, "n3": 40, "n4": 30})
    assert sorted(a + b) == ["n1", "n2", "n3", "n4"] and not set(a) & set(b)
    assert {tuple(a), tuple(b)} == {("n1", "n4"), ("n2", "n3")}


def test_an_objects_frames_screen_it_for_a_seeing_pair():
    sharp, soft = [4.0] * 16, [4.4] * 16
    # 32 frames whose worst quarter is 10% wider than their best.
    assert gives_seeing_split(sharp + soft)
    assert not gives_seeing_split(sharp + soft[1:])
    assert not gives_seeing_split(sharp + [4.3] * 16)


def half(sigma: float, *, seed: int, gain: float = 1.0, sky: float = 500.0, tilt: float = 0.0,
         stretch: float = 1.0, left_sigma: float | None = None, halo: float = 0.0, nebula: float = 1.0,
         glow: float = 0.0, stars: bool = True):
    """A stack's image, coverage and measured stars: 108 stars on a jittered
    grid at Gaussian width `sigma` (`left_sigma` in the left third of the
    field), `stretch` times as wide as high, with `halo` of their light
    spread 20 pixels wide, and a nebula wider than any seeing, `nebula`
    times as bright, all through `gain`, on a sky that climbs `tilt`
    across and has a `glow` 150 pixels wide at one corner."""
    down, across = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    image = sky + tilt * across / WIDTH + gain * nebula * 300.0 * np.exp(
        -((across - 300) ** 2 + (down - 200) ** 2) / (2 * 70.0**2))
    image = image + glow * np.exp(-(across**2 + down**2) / (2 * 150.0**2))
    place = np.random.default_rng(3)
    for row in range(9):
        for column in range(12):
            x = (column + 0.5) * WIDTH / 12 + place.uniform(-4, 4)
            y = (row + 0.5) * HEIGHT / 9 + place.uniform(-4, 4)
            flux = gain * place.uniform(20_000, 200_000)
            width = left_sigma if left_sigma is not None and x < WIDTH / 3 else sigma
            wide, high = width * stretch, width / stretch
            if stars:
                distance = (across - x) ** 2, (down - y) ** 2
                image = image + (1 - halo) * flux / (2 * np.pi * wide * high) * np.exp(
                    -(distance[0] / (2 * wide**2) + distance[1] / (2 * high**2)))
                if halo:
                    image = image + halo * flux / (2 * np.pi * 20.0**2) * np.exp(
                        -(distance[0] + distance[1]) / (2 * 20.0**2))
    image = (image + np.random.default_rng(seed).normal(0.0, 2.0, image.shape)).astype(np.float32)
    return image, np.ones(image.shape, dtype=np.uint16), stack_psf(image)[0]


# How the worst half differs from the best, beyond its seeing (Gaussian
# width 2.2 against 1.5), and what the check makes of it.
CHANGES = {
    "the psf alone": ({}, None),
    "a gain on all its light": ({"gain": 0.9}, None),
    "the same seeing": ({"sigma": 1.55}, "seeing-too-close"),
    "focus moved on one side": ({"left_sigma": 1.3}, "psf-field-mismatch"),
    "wind": ({"stretch": 1.3}, "psf-shape-mismatch"),
    "light other than the stars'": ({"nebula": 0.8}, "flux-mismatch"),
    "haze, which spreads the stars' light": ({"halo": 0.2}, "flux-mismatch"),
    "a brighter sky": ({"sky": 503.0}, None),
    "a gradient": ({"tilt": 6.0}, None),
    "a glow in one corner": ({"glow": 6.0}, "background-mismatch"),
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
        assert all(ratio > 1.25 for row in stats["fwhm_ratio_field"] for ratio in row)
        # A gain and a sky plane are the trainer's to match, and recorded.
        assert stats["background_offset"] == pytest.approx(500.0 - change.get("sky", 500.0) - change.get(
            "tilt", 0.0) / 2, abs=0.5)
        assert stats["background_tilt"] == pytest.approx(change.get("tilt", 0.0) / 2, abs=0.5)
        gain = 1 / change.get("gain", 1.0)
        assert stats["flux_gain"] == stats["star_flux_ratio"] == pytest.approx(gain, rel=0.01)
        # A median moves a little as blur smooths a nebula.
        assert stats["extended_flux_ratio"] == pytest.approx(gain, rel=0.025)


def test_a_stack_is_measured_whatever_a_few_pixels_far_below_its_sky(best_half):
    image = best_half[0].copy()
    image[5, 100:118] = -21_809.0
    image[0] = np.nan
    stars, psf = stack_psf(image)
    assert len(stars) == pytest.approx(len(best_half[2]), abs=3) and len(stars) > 80
    assert psf["stack_fwhm_px"] == pytest.approx(2.3548 * 1.5, rel=0.06)


def test_every_night_of_an_object_stacks_onto_one_reference(tmp_path, capsys):
    corpus = build_corpus(tmp_path / "corpus", nights=2, vignetted=True)
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
    # Seeing worsens through each night, so the quarters of both differ.
    assert listed[0]["seeing_split"] and listed[0]["seeing_spread"] > 0.25

    # Stacked on a machine that mounts the corpus elsewhere: too shallow
    # for a deep stack, but its seeing gives a pair.
    moved = tmp_path / "moved"
    shutil.move(corpus, moved)
    assert run("multi-night", "--source-map", f"{corpus}={moved}")["stacked"] == {}
    stacked = run("multi-night", "--seeing", "--source-map", f"{corpus}={moved}")["stacked"]
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
    # The pair differs in its PSF alone, measured on its stacks, and each
    # half records its frames' widths.
    assert summary["seeing"]["status"] == "built"
    for kind in ("multi-night-seeing-best", "multi-night-seeing-worst"):
        assert (rows[kind]["status"], rows[kind]["reason"]) == ("built", None)
        assert len(loads(rows[kind]["stats"])["frame_fwhm_px"]) == seeing["frames_each"]
    assert max(loads(rows["multi-night-seeing-best"]["stats"])["frame_fwhm_px"]) <= \
        min(loads(rows["multi-night-seeing-worst"]["stats"])["frame_fwhm_px"])
    check = seeing["check"]
    assert check["fwhm_ratio"] > 1.25 and check["stars"] >= 30 and seeing["spread"] > 0.25
    assert check["background_residual"] < 0.25 * check["difference_sigma"]
    whole = loads(rows["multi-night"]["frames"])
    a, b = loads(rows["multi-night-half-a"]["frames"]), loads(rows["multi-night-half-b"]["frames"])
    assert sorted(a + b) == sorted(whole) and not set(a) & set(b)
    assert all(path.startswith(str(corpus)) for path in whole), "the products record the manifest's paths"
    # Each half holds whole nights.
    for dealt in (a, b):
        assert sum(set(dealt) >= set(paths) for paths in nights.values()) == 1

    files = loads(rows["multi-night"]["files"])
    coverage = np.load(store / files["coverage"])
    variance = np.load(store / files["variance"])
    assert coverage.max() <= len(whole) and coverage.max() > max(len(a), len(b))
    assert np.isfinite(variance[coverage > 0]).all() and (variance[coverage > 0] > 0).mean() > 0.99
    assert loads(rows["multi-night"]["recipe"])["nights"] == sorted(nights)

    # Already stacked: left alone.
    assert run("multi-night", "--min-hours", "0", "--source-map", f"{corpus}={moved}")["skipped"] == [
        "Test Scope/Test Nebula/L-60s-g100-bin1"]

    # Judged again from its files, the pair is judged alike; recorded as if
    # its frames' quarters had differed by 5%, it is rejected.
    name = "Test Scope/Test Nebula/L-60s-g100-bin1"
    assert run("multi-night", "--check-seeing")["judged"][name]["status"] == "built"
    with Manifest(store / "manifest.sqlite") as manifest, manifest.transaction():
        for kind in ("multi-night-seeing-best", "multi-night-seeing-worst"):
            manifest.update("products", "id", rows[kind]["id"], {"recipe": dumps(
                {**loads(rows[kind]["recipe"]), "worst_fwhm_px": 1.05 * seeing["best_fwhm_px"]})})
    assert run("multi-night", "--check-seeing", "--object", name)["judged"][name] == {
        "status": "rejected", "reason": "seeing-spread", "fwhm_ratio": check["fwhm_ratio"]}
    with Manifest(store / "manifest.sqlite") as manifest:
        assert {row["status"] for row in manifest.rows(
            "SELECT status FROM products WHERE kind LIKE 'multi-night-seeing-%'")} == {"rejected"}
