"""The PSF Guard fixture also crosses the Python binding unchanged."""
import json
from pathlib import Path

import pytest
import seiza

FIXTURES = Path(__file__).resolve().parents[2] / "seiza-calibration/tests/fixtures/flat-groups"


def request():
    return json.loads((FIXTURES / "request.json").read_text())


def test_psf_guard_groups_events_and_undecided_placement():
    expected = json.loads((FIXTURES / "expected.json").read_text())
    actual = seiza.flat_group_plan(**request())
    assert actual["algorithm_version"] == 1
    for name in ("families", "boundaries", "events"):
        assert actual[name] == expected[name]
    for index, name in enumerate(("before", "during", "after")):
        assert actual["lights"][index]["events"][0]["placement"] == expected["placements"][0][name]
    assert actual["lights"][1]["events"][0]["map_source"] == ["b", "c"]
    assert actual["lights"][0]["automatic_family"]["family"] == 0


def test_empty_request_and_raw_imaging_sessions():
    assert seiza.flat_group_plan()["families"] == []
    actual = seiza.flat_group_plan(
        imaging_lights=[(3, "p", 4 * 86400), (1, "p", 86400), (2, "p", 87000)],
        profile_rigs=[("p", ["rig_with_underscores"])],
    )
    assert len(actual["imaging"]) == 2
    assert actual["imaging"][0]["lights"] == [1, 2]
    assert actual["imaging"][0]["rigs"] == ["rig_with_underscores"]


def test_invalid_records_and_limits_are_errors():
    value = request()
    value["limits"][0]["limits"]["dust"] = -1
    with pytest.raises(ValueError, match="positive finite"):
        seiza.flat_group_plan(**value)
    with pytest.raises(ValueError, match="unknown field"):
        seiza.flat_group_plan(responses=[{"misspelled": 1}])
    value = request()
    value["lights"][0]["at"] = 2**63 - 1
    with pytest.raises(ValueError, match="capture time"):
        seiza.flat_group_plan(**value)
    value = request()
    value["lights"][0]["compatible_families"] = ["gone"]
    with pytest.raises(ValueError, match="unknown compatible family"):
        seiza.flat_group_plan(**value)


def test_orphaned_override_is_kept_with_its_reason():
    value = request()
    value["group_overrides"] = [{
        "anchor_frame": "gone", "session_key": "gone", "family_key": "rig-ha-sensor",
        "rig_uuid": "rig", "choice": "split", "label": "old choice", "chosen_at": 1,
    }]
    actual = seiza.flat_group_plan(**value)
    assert actual["overrides"][0]["status"]["status"] == "orphaned"
    assert actual["families"][0]["groups"] == [[0, 1], [2]]
