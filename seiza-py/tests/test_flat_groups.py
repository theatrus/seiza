"""The PSF Guard fixture also crosses the Python binding unchanged."""
import json
from pathlib import Path

import pytest
import seiza

FIXTURES = Path(__file__).resolve().parents[2] / "seiza-calibration/tests/fixtures/flat-groups"


def request():
    return json.loads((FIXTURES / "request.json").read_text())


def plan(value):
    """The book a request's rows make, and where each of its lights stands."""
    lights = value.pop("lights", [])
    groups = seiza.FlatGroups(value)
    return groups, [groups.place_light(light) for light in lights]


def test_psf_guard_groups_events_and_undecided_placement():
    expected = json.loads((FIXTURES / "expected.json").read_text())
    groups, lights = plan(request())
    assert groups.families == expected["families"]
    assert groups.boundaries == expected["boundaries"]
    assert groups.events == expected["events"]
    for index, name in enumerate(("before", "during", "after")):
        assert lights[index]["events"][0]["placement"] == expected["placements"][0][name]
    assert lights[1]["events"][0]["map_source"] == ["b", "c"]
    assert lights[0]["automatic_family"]["family"] == 0


def test_empty_rows_and_raw_imaging_sessions():
    assert seiza.FlatGroups({}).families == []
    groups = seiza.FlatGroups(dict(
        imaging_lights=[(3, "p", 4 * 86400), (1, "p", 86400), (2, "p", 87000)],
        profile_rigs=[("p", ["rig_with_underscores"])],
    ))
    assert len(groups.imaging) == 2
    assert groups.imaging[0]["lights"] == [1, 2]
    assert groups.imaging[0]["rigs"] == ["rig_with_underscores"]


def test_invalid_records_and_limits_are_errors():
    value = request()
    value["limits"][0]["limits"]["dust"] = -1
    with pytest.raises(ValueError, match="positive finite"):
        plan(value)
    with pytest.raises(ValueError, match="unknown field"):
        seiza.FlatGroups(dict(responses=[{"misspelled": 1}]))
    value = request()
    value["lights"][0]["at"] = 2**63 - 1
    with pytest.raises(ValueError, match="capture time"):
        plan(value)
    value = request()
    value["lights"][0]["compatible_families"] = ["gone"]
    with pytest.raises(ValueError, match="unknown compatible family"):
        plan(value)


def test_orphaned_override_is_kept_with_its_reason():
    value = request()
    value["group_overrides"] = [{
        "anchor_frame": "gone", "session_key": "gone", "family_key": "rig-ha-sensor",
        "rig_uuid": "rig", "choice": "split", "label": "old choice", "chosen_at": 1,
    }]
    groups, _ = plan(value)
    assert groups.overrides[0]["status"]["status"] == "orphaned"
    assert groups.families[0]["groups"] == [[0, 1], [2]]


def test_the_book_reports_its_jobs_and_angle_model():
    groups, _ = plan(request())
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in groups.pairs_to_compare())
    assert isinstance(groups.outliers, list) and isinstance(groups.spanning, list)
    assert isinstance(groups.angle_terms, dict) and isinstance(groups.epochs, dict)
