"""Flat sessions and their dust, as the C ABI's seiza_flat_* and seiza_dust_*
calls offer them: the same synthetic flats as seiza-cabi's tests, and the
same results."""

import numpy as np
import pytest
import seiza

W, H, BIN = 512, 384, 7
LEVEL, PEDESTAL = 30_000.0, 300.0


def flat(disks, noise, seed):
    """A binned flat in raw ADU: a soft vignetting falloff, dust disks
    (x, y, radius, depth) in plane pixels, and seeded noise."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = W / 2.0, H / 2.0
    value = 1.0 - 0.02 * (np.hypot(x - cx, y - cy) / np.hypot(cx, cy)) ** 2
    for dx, dy, radius, depth in disks:
        value = np.where(np.hypot(x - dx, y - dy) <= radius, value * (1.0 - depth), value)
    rng = np.random.default_rng(seed)
    noisy = value * (1.0 + noise * rng.standard_normal((H, W)))
    return (noisy * LEVEL + PEDESTAL).astype(np.float32)


def session(disks, seed):
    return [flat(disks, 0.0015, seed + frame) for frame in range(6)]


def response(flats):
    built = seiza.flat_build_response([f - PEDESTAL for f in flats], BIN)
    assert built is not None
    assert built["frames"] == 6 and built["bin"] == BIN
    assert built["plane"].shape == (H, W) and built["plane"].dtype == np.float32
    return built["plane"]


def test_a_session_finds_its_dust_and_each_flat_is_checked_against_it():
    flats = session([(200.0, 150.0, 40.0, 0.004)], 20)
    found = seiza.dust_find_session(flats, BIN, PEDESTAL)
    assert found["frames"] == 6
    disks = [f for f in found["features"] if f["shape"] == "disk"]
    assert len(disks) == 1, found["features"]
    assert disks[0]["seen_in"]["frames"] == 6

    first = seiza.dust_frame(flats[0], PEDESTAL, found)
    assert 0 in first["seen"]

    matches = seiza.dust_match_features(found["features"], found["features"])
    assert all(pair["earlier"] == pair["later"] for pair in matches)

    # A single plane's dust, and options as a dict.
    single = seiza.dust_find(flats[0] - PEDESTAL, BIN, {"edge_sigma": 3.0})
    assert "features" in single


def test_two_sessions_compare_and_a_change_map_carries_a_flat_across():
    earlier = response(session([(200.0, 150.0, 40.0, 0.004)], 40))
    later = response(session([(200.0, 150.0, 40.0, 0.004), (360.0, 250.0, 30.0, 0.006)], 60))

    comparison = seiza.flat_compare(earlier, later, BIN)
    for key in ("broad", "dust", "sigma", "spots"):
        assert key in comparison
    changes = seiza.dust_find_changes(earlier, later, BIN)
    assert any(change["kind"] == "new" for change in changes["changes"])

    change = seiza.flat_change_map(earlier, later)
    assert change.shape == (H, W)
    image = np.full((70, 70), 1000.0, np.float32)
    seiza.flat_apply_change_map(image, change, BIN, True)
    assert np.isfinite(image).all()

    fit = seiza.flat_fit_change(later, earlier, change, "before")
    assert "outcome" in fit
    choice = seiza.flat_choose_response(later, [earlier, later], alike=[(0, 1)])
    assert "winners" in choice and "pairs" in choice

    background = seiza.flat_lights_background([f - PEDESTAL for f in session([], 80)], 0.0, BIN)
    assert background is None or background.shape == (H, W)


def test_planes_bin_round_trip_through_the_cache_and_verdicts_combine():
    assert seiza.flat_bin_factor(6000, 4000) >= 1
    binned = seiza.flat_bin_frame(np.full((20, 30), 2.0, np.float32), 3)
    assert binned.shape == (6, 10) and (binned == 2.0).all()

    # Zero is the format's mark for a missing sample, so start above it.
    plane = (np.arange(1, 13, dtype=np.float32) / 8.0).reshape(3, 4)
    decoded = seiza.flat_decode_plane(seiza.flat_encode_plane(plane))
    assert decoded.shape == (3, 4)
    assert np.allclose(decoded, plane, atol=1e-4)
    assert seiza.flat_decode_plane(b"\x01\x02") is None

    assert seiza.flat_combine_verdicts(["before", "undecided"]) == "before"

    with pytest.raises(ValueError, match="unknown field"):
        seiza.dust_find(np.ones((64, 64), np.float32), 1, {"edge_sigmaa": 3})
    with pytest.raises(ValueError, match="reference_side"):
        seiza.flat_fit_change(plane, plane, plane, "sideways")


def test_constellation_figures_and_their_projection_carry_the_attribution():
    data = seiza.constellation_figures()
    assert "CC BY 4.0" in data["attribution"]
    assert len(data["figures"]) == 88 and len(data["stars"]) == 697
    orion = next(figure for figure in data["figures"] if figure["abbr"] == "Ori")
    assert orion["name"] == "Orion" and orion["segments"]

    # A wide field on Orion's belt shows Orion, inside the frame.
    wcs = seiza.Wcs.from_center_scale_rotation((83.8, -1.2), (999.5, 749.5), 60.0)
    projected = seiza.project_constellations(wcs, 2000, 1500)
    assert "CC BY 4.0" in projected["attribution"]
    orion = next(figure for figure in projected["figures"] if figure["abbr"] == "Ori")
    points = [point for line in orion["polylines"] for point in line]
    assert points and all(-0.5 <= x <= 2000.5 and -0.5 <= y <= 1500.5 for x, y in points)
    with pytest.raises(ValueError):
        seiza.project_constellations(wcs, 0, 1500)
