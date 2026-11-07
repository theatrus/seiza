import json

import pytest

pytest.importorskip("torch")

from seiza_training import bundle  # noqa: E402


def test_a_card_holds_only_what_the_runtime_reads():
    record = {
        "config": {"noise_plane": "asinh", "sky": "flat", "width": 64, "manifest": "/private/machine/path"},
        "inference": {"tile": 512, "overlap": 276, "margin": 122, "multiple": 16},
    }
    graphs = [{"precision": "f32", "file": "model-f32.onnx", "sha256": "00"}]
    written = bundle.card("noise-test", record, "constant", graphs, ["cpu"])
    assert "/private/machine/path" not in json.dumps(written), "a machine's paths stay out of a card"
    assert written["tiling"] == {"tile": 512, "margin": 122, "ramp": 32, "multiple": 16}
    assert written["reading"]["noise_map"] == "constant"
    assert written["reading"]["sky"] == "flat"
    assert written["parity"] == ["cpu"]
    assert written["inputs"] == [{"tensor": "planes", "planes": ["image", "noise"]}]


def test_a_model_trained_on_the_stack_s_map_must_say_what_it_reads(monkeypatch, tmp_path):
    monkeypatch.setattr(bundle, "load_run", lambda store, run: ({"config": {"noise_map": "model"}}, None))
    with pytest.raises(ValueError, match="say which it reads"):
        bundle.write_bundle(None, "R1", tmp_path, name="noise-test")
    assert not list(tmp_path.iterdir()), "nothing is written"
