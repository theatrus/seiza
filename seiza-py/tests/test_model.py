from __future__ import annotations

import importlib.util
import json
import shutil
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import seiza

FIXTURES = Path(__file__).parents[2] / "seiza-model/tests/fixtures"
TINY = FIXTURES / "tiny"
needs_runtime = pytest.mark.skipif(
    "SEIZA_ONNXRUNTIME" not in os.environ and importlib.util.find_spec("onnxruntime") is None,
    reason="neither SEIZA_ONNXRUNTIME nor the onnxruntime package gives a library",
)


def fixture_image() -> np.ndarray:
    return np.fromfile(FIXTURES / "image.f32", dtype="<f4").reshape(230, 300)


def test_a_model_installs_verifies_lists_and_goes(tmp_path: Path) -> None:
    store = tmp_path / "store"
    assert seiza.installed_models(store) == []

    installed = seiza.install_model(TINY, store)
    assert installed.name == "noise-tiny"
    assert installed.noise_map == "constant"
    assert installed.precisions == ["f32", "f16"]
    assert [model.name for model in seiza.installed_models(store)] == ["noise-tiny"]

    seiza.verify_model("noise-tiny", store)
    seiza.verify_model(None, store)
    seiza.remove_model("noise-tiny", store)
    with pytest.raises(seiza.EngineError, match="noise-tiny"):
        seiza.remove_model("noise-tiny", store)


@pytest.mark.parametrize(
    "field,value",
    [
        ("margin", sys.maxsize + 1),
        ("ramp", 2 * sys.maxsize + 1),
        ("tile", 0),
        ("tile", 63),
        ("tile", 16),
        ("multiple", 0),
    ],
)
def test_invalid_tiling_cannot_replace_an_installed_model(tmp_path: Path, field: str, value: int) -> None:
    store = tmp_path / "store"
    seiza.install_model(TINY, store)
    source = tmp_path / "source"
    shutil.copytree(TINY, source)
    path = source / "card.json"
    card = json.loads(path.read_text())
    card["tiling"][field] = value
    path.write_text(json.dumps(card))

    with pytest.raises(seiza.EngineError, match="tiling"):
        seiza.install_model(source, store)
    seiza.verify_model("noise-tiny", store)
    assert [model.name for model in seiza.installed_models(store)] == ["noise-tiny"]


def test_a_changed_graph_is_refused(tmp_path: Path) -> None:
    store = tmp_path / "store"
    installed = seiza.install_model(TINY, store)
    with (Path(installed.path) / "model-f32.onnx").open("ab") as graph:
        graph.write(b"\0")
    with pytest.raises(seiza.EngineError, match="does not match its card"):
        seiza.verify_model("noise-tiny", store)


def test_backends_answer_with_or_without_a_library() -> None:
    statuses = seiza.model_backends()
    assert any(status.backend == "cpu" for status in statuses)
    assert all(isinstance(status.automatic, bool) for status in statuses)


def test_the_onnxruntime_package_s_library_is_found(tmp_path: Path) -> None:
    capi = tmp_path / "onnxruntime" / "capi"
    capi.mkdir(parents=True)
    (tmp_path / "onnxruntime" / "__init__.py").write_text("")
    name = "onnxruntime.dll" if sys.platform == "win32" else "libonnxruntime.so.1.31.1"
    (capi / name).write_bytes(b"not a library")
    (capi / "libonnxruntime_providers_shared.so").write_bytes(b"not this one")
    environment = {key: value for key, value in os.environ.items() if key != "SEIZA_ONNXRUNTIME"}
    environment["PYTHONPATH"] = str(tmp_path)

    # In a process of its own, since a process loads ONNX Runtime once.
    script = "import seiza; print(seiza.model_backends()[0].unavailable)"
    found = subprocess.run([sys.executable, "-c", script], env=environment, capture_output=True, text=True, check=True)
    assert name in found.stdout

    environment["SEIZA_ONNXRUNTIME"] = str(tmp_path / "chosen")
    (tmp_path / "chosen").write_bytes(b"not a library")
    found = subprocess.run([sys.executable, "-c", script], env=environment, capture_output=True, text=True, check=True)
    assert "chosen" in found.stdout and name not in found.stdout


def test_a_bad_backend_or_precision_is_refused() -> None:
    with pytest.raises(ValueError, match="metal"):
        seiza.NoiseModel(str(TINY), "metal")
    with pytest.raises(ValueError, match="f8"):
        seiza.NoiseModel(str(TINY), "cpu", precision="f8")


def test_the_output_version_is_a_version() -> None:
    assert seiza.MODEL_OUTPUT_VERSION.count(".") == 2


@needs_runtime
def test_a_tiny_model_runs_and_blends() -> None:
    image = fixture_image()
    model = seiza.NoiseModel(str(TINY), "cpu")
    assert model.backend == "cpu"
    assert model.tile == 64
    assert seiza.model_runtime().version.startswith("1.")

    seen: list[tuple[int, int, int, int]] = []
    result = model.denoise(image, progress=lambda *step: seen.append(step))
    assert seen[-1][2] == seen[-1][3] and seen[-1][:2] == (0, 1)

    report = result.report
    assert (report.model, report.backend, report.precision) == ("noise-tiny", "cpu", "f32")
    assert report.noise_map == "constant"
    assert report.channels[0].noise > 0

    residual = result.residual()
    assert residual.shape == image.shape
    whole = result.blend()
    assert whole.dtype == np.float32
    np.testing.assert_array_equal(whole[50:60, 50:60], (image + residual)[50:60, 50:60])
    assert np.isnan(whole[0, 0])
    kept = result.blend(high=0.0, low=0.0)
    np.testing.assert_array_equal(kept, image)
    with pytest.raises(ValueError, match="amounts"):
        result.blend(high=2.0)
    with pytest.raises(IndexError):
        result.residual(1)


@needs_runtime
def test_colour_channels_run_alone_and_come_back_in_place() -> None:
    image = fixture_image()
    model = seiza.NoiseModel(str(TINY), "cpu")
    mono = model.denoise(image).blend()
    colour = np.ascontiguousarray(np.stack([image, image * 2, image], axis=-1))
    blended = model.denoise(colour).blend()
    assert blended.shape == colour.shape
    np.testing.assert_array_equal(blended[..., 0], mono)
    np.testing.assert_array_equal(blended[..., 2], mono)


@needs_runtime
def test_a_run_stops_on_cancel_and_raises_what_a_callback_raises() -> None:
    image = fixture_image()
    model = seiza.NoiseModel(str(TINY), "cpu")
    with pytest.raises(seiza.EngineError, match="cancelled"):
        model.denoise(image, cancel=lambda: True)

    def broken(*_: int) -> None:
        raise KeyError("from the callback")

    with pytest.raises(KeyError, match="from the callback"):
        model.denoise(image, progress=broken)
