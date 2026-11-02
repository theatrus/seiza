import numpy as np
import pytest

import seiza

# The fixtures are written as FITS through astropy; where it has no wheel
# yet, such as on a newly released Python, these tests skip.
fits = pytest.importorskip("astropy.io.fits")


def star_field(tmp_path, sessions=2, per_session=3):
    """Lights of one star field a few pixels apart, each session's carrying
    a bias of its own, and that bias as a master."""
    rng = np.random.default_rng(7)
    height, width, margin = 96, 128, 10
    sky = np.full((height + 2 * margin, width + 2 * margin), 100.0)
    yy, xx = np.mgrid[: sky.shape[0], : sky.shape[1]]
    for _ in range(60):
        x, y = rng.uniform(5, sky.shape[1] - 5), rng.uniform(5, sky.shape[0] - 5)
        sky += rng.uniform(200, 3000) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 3.2)
    plan = []
    for session in range(sessions):
        level = 50.0 * (session + 1)
        bias = tmp_path / f"bias-{session}.fits"
        fits.writeto(bias, np.full((height, width), level, np.float32))
        lights = []
        for index in range(per_session):
            frame = session * per_session + index
            ox, oy = margin - 3 + frame % 7, margin - 2 + (frame * 3) % 5
            data = sky[oy : oy + height, ox : ox + width] + level
            data += rng.normal(0, 3 + frame % 3, data.shape)
            light = tmp_path / f"light-{session}-{index}.fits"
            fits.writeto(light, data.astype(np.float32))
            lights.append(light)
        plan.append({"lights": lights, "calibration": {"bias": bias}})
    return plan


def test_a_channel_runs_resumes_and_combines(tmp_path):
    sessions = star_field(tmp_path)
    reference = {
        "image": sessions[0]["lights"][0],
        "light": sessions[0]["lights"][0],
        "calibration": sessions[0]["calibration"],
    }
    options = {
        "stack": {"normalization": {"mode": "none"}, "rejection": {"mode": "none"}},
        "contributor": "tester",
    }
    plan = {"reference": reference, "sessions": sessions, "options": options}
    store = tmp_path / "store"
    first = seiza.run_channel(plan, store)
    assert not first["tile"]["reused"]
    assert first["tile"]["identity"]["kind"] == "tile"
    assert first["tile"]["identity"]["combined"] == 6
    assert [session["reused"] for session in first["sessions"]] == [False, False]
    again = seiza.run_channel(plan, store)
    assert again["tile"]["reused"]
    assert again["tile"]["identity"] == first["tile"]["identity"]

    # One session's step has the key the channel stored it under.
    step = seiza.SessionStep.plan(reference, sessions[1], options)
    assert step.key == first["sessions"][1]["key"]
    assert len(step.frames) == 3 and len(step.key) == 64
    assert step.lights == sessions[1]["lights"]
    outcome = step.run_into(store)
    assert outcome["reused"] and outcome["path"] == first["sessions"][1]["path"]

    identity = seiza.PartialStack.read_identity(outcome["path"])
    assert identity["kind"] == "partial"
    assert identity["step_key"] == step.key
    assert identity["contributor"] == "tester"

    parts = [seiza.PartialStack.open(session["path"]) for session in first["sessions"]]
    part = parts[0]
    assert part.mean.shape == (96, 128)
    assert part.weight.shape == part.coverage.shape == part.mean.shape
    assert part.rejected is not None and part.variance is not None
    assert part.coverage.max() == 3
    tile = seiza.PartialStack.combine(list(reversed(parts)))
    assert tile.identity == first["tile"]["identity"]
    from_files = seiza.PartialStack.combine_files(
        [session["path"] for session in first["sessions"]])
    np.testing.assert_array_equal(from_files.mean, tile.mean)
    np.testing.assert_array_equal(tile.coverage, parts[0].coverage + parts[1].coverage)
    covered = tile.coverage > 0
    weighted = (parts[0].mean * parts[0].weight + parts[1].mean * parts[1].weight)
    both = (parts[0].coverage > 0) & (parts[1].coverage > 0)
    np.testing.assert_allclose(
        tile.mean[both], (weighted / (parts[0].weight + parts[1].weight))[both], rtol=1e-5
    )
    assert np.isfinite(tile.mean[covered]).all()
    # Each session's bias was taken off: the sky sits near 100.
    assert np.nanmedian(tile.mean) == pytest.approx(100, abs=5)

    written = tmp_path / "tile.xisf"
    tile.write(written)
    reread = seiza.PartialStack.open(written)
    assert reread.identity == tile.identity
    np.testing.assert_array_equal(reread.coverage, tile.coverage)

    fresh = seiza.SessionStep.plan(reference, sessions[0], {"stack": options["stack"]})
    stack, rejected = fresh.run()
    assert stack.identity["kind"] == "partial" and rejected == []
    with pytest.raises(seiza.StackError):
        seiza.PartialStack.read_identity(tmp_path / "missing.fits")


def test_a_session_that_gains_lights_is_extended(tmp_path):
    sessions = star_field(tmp_path, sessions=1, per_session=4)
    reference = {
        "image": sessions[0]["lights"][0],
        "light": sessions[0]["lights"][0],
        "calibration": sessions[0]["calibration"],
    }
    options = {
        "stack": {"normalization": {"mode": "none"}, "rejection": {"mode": "none"}},
        "keep_context": True,
    }
    early = {"lights": sessions[0]["lights"][:2], "calibration": sessions[0]["calibration"]}
    store = tmp_path / "store"
    first = seiza.SessionStep.plan(reference, early, options).run_into(store)
    assert first["extended_from"] is None
    later = seiza.SessionStep.plan(reference, sessions[0], options).run_into(store)
    assert later["extended_from"] == first["key"]
    assert not later["reused"]
    extended = seiza.PartialStack.open(later["path"])
    assert extended.identity["combined"] == 4
    assert extended.identity["context"] is not None
    whole, _ = seiza.SessionStep.plan(reference, sessions[0], options).run()
    np.testing.assert_array_equal(extended.mean, whole.mean)
    np.testing.assert_array_equal(extended.weight, whole.weight)
