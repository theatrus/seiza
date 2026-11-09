import json

import pytest

pytest.importorskip("torch")

from seiza_training import bundle  # noqa: E402
from seiza_training.holes import HoleGuard  # noqa: E402


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
    assert written["reading"]["noise_correlation"] == 1.24, "the runtime raises the map for spread noise as training did"
    assert written["parity"] == ["cpu"]
    assert written["inputs"] == [{"tensor": "planes", "planes": ["image", "noise"]}]
    assert "holes" not in written["reading"], "a card without the guard runs the network on the image as it is"
    guarded = bundle.card("noise-test", record, "constant", graphs, ["cpu"], HoleGuard())
    assert guarded["reading"]["holes"] == {"depth": 30.0, "margin": 3, "fill_radius": 12, "restore": 5.0}


def test_a_model_trained_on_the_stack_s_map_must_say_what_it_reads(monkeypatch, tmp_path):
    monkeypatch.setattr(bundle, "load_run", lambda store, run: ({"config": {"noise_map": "model"}}, None))
    with pytest.raises(ValueError, match="say which it reads"):
        bundle.write_bundle(None, "R1", tmp_path, name="noise-test")
    assert not list(tmp_path.iterdir()), "nothing is written"


def test_a_bundle_keeps_holes_from_its_network_as_its_run_trained(monkeypatch, tmp_path):
    import torch

    record = {
        "config": {"noise_map": "constant", "noise_plane": "asinh", "sky": "global", "holes": True},
        "inference": {"tile": 512, "overlap": 276, "margin": 122, "multiple": 16},
    }
    monkeypatch.setattr(bundle, "load_run", lambda store, run: (record, torch.nn.Identity()))
    monkeypatch.setattr(bundle, "export_graph", lambda model, path, dtype: path.write_bytes(b"graph"))
    written = bundle.write_bundle(None, "R1", tmp_path / "trained", name="noise-test")
    assert written["reading"]["holes"]["depth"] == 30.0, "a run trained under the guard is shipped with it"
    told = bundle.write_bundle(None, "R1", tmp_path / "told", name="noise-test", holes=False)
    assert "holes" not in told["reading"], "asked, the card leaves the guard out"


def test_a_card_names_only_backends_the_runtime_knows():
    record = {"config": {}, "inference": {"tile": 512, "overlap": 276, "margin": 122, "multiple": 16}}
    graphs = [{"precision": "f32", "file": "model-f32.onnx", "sha256": "00"}]
    with pytest.raises(ValueError, match="coreml_gpu"):
        bundle.card("noise-test", record, "constant", graphs, ["coreml_gpu"])
