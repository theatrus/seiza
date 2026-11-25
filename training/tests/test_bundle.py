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


SHARPEN_RECORD = {
    "config": {"width": 4, "levels": 3, "planes": ["image", "noise", "ratio", "nonstellar_ratio", "points"],
               "stamp_planes": ["peak", "wings"], "wings": 0.001, "choice": {"kernel": "/private/machine/path"},
               "settings": {"nonstellar": True, "kernel_ratios": [1.1, 4.0], "floor_px": 1.5, "residual": "asinh"}},
    "inference": {"tile": 512, "overlap": 276, "margin": 122, "multiple": 16},
}


def test_a_sharpening_card_names_what_the_runtime_makes_from_the_image():
    graphs = [{"precision": "f32", "file": "body-f32.onnx", "sha256": "00"}]
    encoder = {"file": "encoder-f32.onnx", "sha256": "11", "features": 64}
    written = bundle.sharpen_card("sharpen-test", SHARPEN_RECORD, graphs, encoder, ["cpu"])
    assert "/private/machine/path" not in json.dumps(written), "a machine's paths stay out of a card"
    assert written["task"] == "sharpen"
    assert written["inputs"] == [{"tensor": "planes",
                                  "planes": ["image", "noise", "ratio", "nonstellar-ratio", "points"]}]
    assert written["dimensions"] == ["height", "width"]
    assert written["reading"]["noise_map"] == "image", "the noise map is made from the image alone"
    assert written["tiling"] == {"tile": 512, "margin": 122, "ramp": 32, "multiple": 16}
    told = written["sharpen"]
    assert told["encoder"] == {"file": "encoder-f32.onnx", "sha256": "11", "input": "stamp", "output": "vector",
                               "planes": ["peak", "wings"], "side": 129, "wings": 0.001, "features": 64}
    assert told["vector"] == {"tensor": "vector", "dimensions": ["map_height", "map_width"]}
    assert told["residual"] == "asinh" and told["floor_px"] == 1.5 and told["sharpest"] == 0.25
    assert told["strengths"] == {"stars": 0.46, "rest": 0.45}, "the scaled pass's full-size ask, unless told"
    assert told["scaled"] == {"stars_px": 3.5, "blend": 0.5, "rest": 0.35, "stars": 2.0}
    assert told["psf"]["grid_px"] == 256 and told["psf"]["minimum_snr"] == 30.0
    assert told["points"] == {"noise_reduction_radius": 0, "structure_removals": ["filtered", "atrous"],
                              "clip_peaks": 300, "peak_block": 16, "core_radius": 24.0}
    asked = bundle.sharpen_card("sharpen-test", SHARPEN_RECORD, graphs, encoder, [], (0.6, 0.7))
    assert asked["sharpen"]["strengths"] == {"stars": 0.6, "rest": 0.7}
    whole = bundle.sharpen_card("sharpen-test", SHARPEN_RECORD, graphs, encoder, [], scaled=False)["sharpen"]
    assert "scaled" not in whole, "a card without the scaled pass says nothing of it"
    assert whole["strengths"] == {"stars": 0.46, "rest": 0.5}, "the sheets' full-size ask, unless told"


def test_a_sharpening_bundle_s_graphs_match_the_model(monkeypatch, tmp_path):
    import numpy as np
    import torch

    onnxruntime = pytest.importorskip("onnxruntime")
    from seiza_training import sharpen_model

    torch.manual_seed(3)
    model = sharpen_model.build_model(4, 3, sharpen_model.PLANES_NONSTELLAR).eval()
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    monkeypatch.setattr(sharpen_model, "load_run", lambda store, run: (SHARPEN_RECORD, model))
    written = bundle.write_sharpen_bundle(None, "S1", tmp_path, name="sharpen-test")
    assert {entry["precision"] for entry in written["export"]["body"]} == {"f32", "f16"}
    assert written["export"]["encoder"]["max_relative_error"] < 1e-4
    assert written["export"]["body"][0]["max_relative_error"] < 1e-4
    card = json.loads((tmp_path / "card.json").read_text())
    assert card == written["card"]
    for graph in [*card["graphs"], card["sharpen"]["encoder"]]:
        assert bundle.sha256_of(tmp_path / graph["file"]) == graph["sha256"]

    # The two graphs together are the model on one stamp: a map that holds
    # one vector everywhere is that vector.
    planes = torch.randn(1, 5, 64, 48)
    stamp = torch.rand(1, 2, 129, 129)
    encoder = onnxruntime.InferenceSession(str(tmp_path / "encoder-f32.onnx"), providers=["CPUExecutionProvider"])
    body = onnxruntime.InferenceSession(str(tmp_path / "body-f32.onnx"), providers=["CPUExecutionProvider"])
    vector = encoder.run(None, {"stamp": stamp.numpy()})[0]
    assert vector.shape == (1, 64, 1, 1)
    spread = np.ascontiguousarray(np.broadcast_to(vector, (1, 64, 16, 12)))
    got = body.run(None, {"planes": planes.numpy(), "vector": spread})[0]
    with torch.no_grad():
        expected = model.float()(planes, stamp).numpy()
    assert np.abs(got - expected).max() <= 1e-4 * np.abs(expected).max()


def test_a_card_names_only_backends_the_runtime_knows():
    record = {"config": {}, "inference": {"tile": 512, "overlap": 276, "margin": 122, "multiple": 16}}
    graphs = [{"precision": "f32", "file": "model-f32.onnx", "sha256": "00"}]
    with pytest.raises(ValueError, match="coreml_gpu"):
        bundle.card("noise-test", record, "constant", graphs, ["coreml_gpu"])
