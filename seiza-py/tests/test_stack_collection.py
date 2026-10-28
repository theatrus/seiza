import copy
import json
import pytest
import seiza

REPORT = json.loads('''{"schema_version":1,"stacking_version":"test","output_version":"test","stage":"online","width":2,"height":1,"channels":1,"accepted_frames":2,"rejected_frames":1,"support":[{"supported_pixels":2,"total_pixels":2,"support_minimum":1.0,"support_maximum":2.0,"support_mean":1.5,"rejected_samples":1}],"support_map":{"width":2,"height":1,"channels":1,"values":[2,1],"quantity":"accepted_observations"},"rejection_map":{"width":2,"height":1,"channels":1,"values":[0,1],"quantity":"rejected_observations"},"context":null,"final_samples":null,"provenance":null,"warnings":[] }''')

def collection():
    return {"schema_version": 1, "title": "SHO </script> &", "stacks": [{"label": "S", "report": copy.deepcopy(REPORT)}]}

def test_collection_roundtrip_and_missing_context():
    data = collection()
    html = seiza.stack_collection_html(data)
    embedded = html.split('<script id="data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert json.loads(embedded) == data
    assert "SHO </script>" not in html
    assert "<iframe" not in html

@pytest.mark.parametrize("change", [
    lambda c: c.update(schema_version=99),
    lambda c: c.update(stacks=[]),
    lambda c: c["stacks"].append({"label": " s ", "report": REPORT}),
    lambda c: c["stacks"][0]["report"]["support_map"].update(values=[]),
    lambda c: c["stacks"][0]["report"].update(channels=0),
])
def test_collection_rejects_invalid_data(change):
    data = collection()
    change(data)
    with pytest.raises(ValueError):
        seiza.stack_collection_html(data)


def test_collection_preserves_rgb_components_different_grids_and_drizzle_units():
    data = collection()
    rgb = copy.deepcopy(REPORT)
    rgb.update(width=1, height=1, channels=3)
    rgb["support"] = [dict(rgb["support"][0], total_pixels=1, supported_pixels=1) for _ in range(3)]
    for key in ["support_map", "rejection_map"]:
        rgb[key].update(width=1, height=1, channels=3, values=[1, 1, 1])
    drizzle = copy.deepcopy(REPORT)
    drizzle.update(stage="drizzled", rejection_map=None)
    drizzle["support_map"]["quantity"] = "drop_area_times_frame_weight"
    drizzle["support"][0]["rejected_samples"] = None
    data["stacks"] += [{"label": "RGB", "report": rgb}, {"label": "Drizzle", "report": drizzle}]
    html = seiza.stack_collection_html(data)
    embedded = html.split('<script id="data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert json.loads(embedded) == data
    drizzle["support_map"]["quantity"] = "accepted_observations"
    with pytest.raises(ValueError):
        seiza.stack_collection_html(data)


def test_report_levels_skip_reads_preserve_result_and_validate(tmp_path):
    missing = tmp_path / "missing.fits"
    assert seiza.prepare_stack_report(level="none", inputs=[missing]) is None
    with pytest.raises(ValueError, match="level"):
        seiza.prepare_stack_report(REPORT, level="unexpected")
    with pytest.raises(ValueError, match="required"):
        seiza.prepare_stack_report(level="cheap")
    with pytest.raises(ValueError, match="inputs"):
        seiza.prepare_stack_report(REPORT, level="rich")
    cheap = seiza.prepare_stack_report(REPORT, level="cheap", inputs=[missing])
    assert cheap["provenance"]["reporting"] == {"level": "cheap", "analysis": None}
    rich = seiza.prepare_stack_report(REPORT, level="rich", inputs=[missing])
    analysis = rich["provenance"]["reporting"]["analysis"]
    assert analysis["frames"][0]["measurement_status"] == "failed"
    assert analysis["frames"][0]["error"]
    assert analysis["timings"]["total_seconds"] >= 0
    for key in ["support", "support_map", "rejection_map", "accepted_frames", "final_samples"]:
        assert rich[key] == cheap[key] == REPORT[key]
    assert seiza.prepare_stack_report(rich, level="cheap")["provenance"]["reporting"]["analysis"] is None
    assert REPORT["provenance"] is None
    html = seiza.stack_report_html(rich)
    assert "Image quality & integration depth" in html and "__CHART_SCRIPT__" not in html
