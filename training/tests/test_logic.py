import random
import struct
import zlib

import numpy as np
import pytest

from seiza_training import build, measure, sky, splits
from seiza_training.manifest import Manifest
from seiza_training.store import write_png


def test_header_positions_read_degrees_or_sexagesimal():
    assert sky.header_position({"RA": 149.2, "DEC": 69.5}) == (149.2, 69.5)
    ra, dec = sky.header_position({"OBJCTRA": "09 56 15", "OBJCTDEC": "-69 12 41"})
    assert ra == pytest.approx(149.0625)
    assert dec == pytest.approx(-69.211389, abs=1e-6)
    assert sky.header_position({}) == (None, None)


def test_separation_and_field():
    assert sky.separation_deg(10.0, 0.0, 11.0, 0.0) == pytest.approx(1.0)
    assert sky.separation_deg(359.5, 0.0, 0.5, 0.0) == pytest.approx(1.0)
    scale = sky.pixel_scale_arcsec(3.76, 749.0)
    assert scale == pytest.approx(1.0355, abs=1e-3)
    assert sky.field_diagonal_deg(scale, 6248, 4176) == pytest.approx(2.16, abs=0.01)


def test_objects_and_overlapping_fields_share_a_group():
    def row(identity, target, ra, dec, field=1.0):
        return {"id": identity, "target": target, "ra_deg": ra, "dec_deg": dec,
                "field_deg": field, "width": 3000, "height": 2000}

    sets = [
        row("a", "NGC7000", 314.0, 44.0),
        row("b", "NGC 7000", 100.0, -20.0),  # same object, whatever the header says
        row("c", "Pelican", 314.3, 44.2),  # overlaps a
        row("g", "Sadr", 315.2, 44.6),  # within c's corner circle, but no sky shared
        row("d", "Heart and Soul Nebula Panel 1", 38.0, 61.0),
        row("e", "Heart and Soul Nebula Panel 3", 45.0, 61.0),
        row("f", "M 31", 10.7, 41.3),
    ]
    group = splits.sky_groups(sets)
    assert group["a"] == group["b"] == group["c"]
    assert group["d"] == group["e"]
    assert len({group["a"], group["d"], group["f"], group["g"]}) == 4


def test_splits_are_stable_held_out_and_sticky(tmp_path):
    with Manifest(tmp_path / "manifest.sqlite") as manifest:
        with manifest.transaction():
            for index in range(40):
                manifest.upsert(
                    "sets",
                    {
                        "id": f"s{index:02d}",
                        "name": f"set {index}",
                        "telescope": "Hold" if index < 3 else "Main",
                        "target": f"Object {index}",
                        "ra_deg": index * 9.0,
                        "dec_deg": 0.0,
                        "field_deg": 1.0,
                        "width": 3000,
                        "height": 2000,
                        "frames": 10,
                    },
                    key="id",
                )
        summary = splits.assign_splits(manifest, hold_out="Hold")
        assert summary["hold_out"] == "Hold"
        rows = {row["id"]: row["split"] for row in manifest.rows("SELECT id, split FROM sets")}
        assert all(rows[f"s{index:02d}"] == "held-out" for index in range(3))
        assert set(rows.values()) <= {*splits.SPLITS, "held-out"}
        assert "train" in rows.values()
        # A rerun, and a set added to an existing group, keep their splits.
        with manifest.transaction():
            manifest.upsert(
                "sets",
                {"id": "s99", "name": "late", "telescope": "Main", "target": "Object 10",
                 "ra_deg": 300.0, "dec_deg": 50.0, "field_deg": 1.0, "width": 3000,
                 "height": 2000, "frames": 5},
                key="id",
            )
        splits.assign_splits(manifest)
        again = {row["id"]: row["split"] for row in manifest.rows("SELECT id, split FROM sets")}
        assert all(again[key] == value for key, value in rows.items())
        assert again["s99"] == rows["s10"]
        with pytest.raises(ValueError, match="holds out"):
            splits.assign_splits(manifest, hold_out="Main")


def test_pairs_are_disjoint_interleaved_and_seeded():
    ordered = [f"f{index:03d}" for index in range(100)]
    pairs = build.pair_draws(ordered, "set")
    assert {(pair["size"], pair["draw"]) for pair in pairs} == {
        (size, draw) for size in (4, 8, 16) for draw in range(3)
    }
    for pair in pairs:
        assert not set(pair["a"]) & set(pair["b"])
        assert sorted(pair["a"] + pair["b"]) == ordered
        positions = sorted(ordered.index(path) for path in pair["a"])
        stretch = len(ordered) / pair["size"]
        assert all(int(k * stretch) <= p < int((k + 1) * stretch) + 1 for k, p in enumerate(positions))
    assert build.pair_draws(ordered, "set") == pairs
    assert build.pair_draws(ordered, "other") != pairs
    assert build.pair_draws(ordered[:19], "set") == []


def test_seeing_split_takes_matched_quarters():
    fwhm = {f"f{index}": 2.0 + 0.1 * index for index in range(10)}
    best, worst = build.seeing_split(fwhm)
    assert best == ["f0", "f1"]
    assert worst == ["f8", "f9"]


def test_frames_are_judged_and_sets_qualify():
    rng = random.Random(3)
    frames = [
        {"path": f"f{index}", "status": "measured", "stars": 200, "fwhm_px": 2.0 + index * 0.02,
         "eccentricity": 0.3, "noise": 10.0, "flags": []}
        for index in range(48)
    ]
    frames[5]["stars"] = 40
    frames[6]["eccentricity"] = 0.8
    frames[7]["status"] = "failed"
    rng.shuffle(frames)
    measure.judge(frames)
    flags = {frame["path"]: frame["flags"] for frame in frames}
    assert flags["f5"] == ["cloud"]
    assert flags["f6"] == ["elongated"]
    assert flags["f7"] == ["measurement-failed"]
    summary = measure.summarize(frames)
    assert summary["accepted"] == 45
    assert summary["recipes"] == ["noise-pairs", "seeing-split", "deep"]
    assert summary["seeing_spread"] > 0.25


def test_png_is_well_formed(tmp_path):
    pixels = (np.arange(12, dtype=np.uint8).reshape(3, 4) * 20)
    path = tmp_path / "grey.png"
    write_png(path, pixels)
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    length = struct.unpack(">I", data[8:12])[0]
    width, height = struct.unpack(">II", data[16:24])
    assert (length, width, height) == (13, 4, 3)
    start = data.index(b"IDAT") + 4
    size = struct.unpack(">I", data[start - 8:start - 4])[0]
    raw = zlib.decompress(data[start:start + size])
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(3, 5)
    assert (rows[:, 0] == 0).all()
    np.testing.assert_array_equal(rows[:, 1:], pixels)


def test_reviews_record_one_decision_each(tmp_path):
    with Manifest(tmp_path / "manifest.sqlite") as manifest:
        manifest.record_review("s1", "set", "accept", "reviewer")
        manifest.record_review("s1", "set", "reject", "reviewer", "trails")
        row = manifest.review("s1", "set")
        assert (row["decision"], row["note"]) == ("reject", "trails")
        with pytest.raises(ValueError):
            manifest.record_review("s1", "set", "maybe", "reviewer")


def test_whole_images_are_read_a_window_at_a_time(tmp_path):
    from astropy.io import fits

    from seiza_training.store import read_image

    rng = np.random.default_rng(1)
    mono = rng.normal(100.0, 5.0, (1100, 2300)).astype(np.float32)
    fits.writeto(tmp_path / "mono.fits", mono)
    np.testing.assert_array_equal(read_image(tmp_path / "mono.fits", 2300, 1100), mono)
    colour = rng.normal(100.0, 5.0, (3, 1030, 1025)).astype(np.float32)
    fits.writeto(tmp_path / "colour.fits", colour)
    np.testing.assert_array_equal(read_image(tmp_path / "colour.fits", 1025, 1030), colour)


def test_a_build_needs_room_for_its_frames_and_products_together(tmp_path):
    import shutil

    from seiza_training.store import Store

    store = Store(tmp_path / "store", tmp_path / "scratch")
    free = shutil.disk_usage(tmp_path).free
    store.check_space(free // 3, free // 3)
    # The scratch and the store share a disk here, so each need fits alone
    # and the two together do not.
    with pytest.raises(RuntimeError, match="GB is free"):
        store.check_space(free // 2 + 2**30, free // 2 + 2**30)
