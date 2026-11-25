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
    assert installed.holes is None, "the tiny model's card has no hole guard"
    assert len(installed.card_sha256) == 64
    assert installed.precisions == ["f32", "f16"]
    assert [model.name for model in seiza.installed_models(store)] == ["noise-tiny"]

    seiza.verify_model("noise-tiny", store)
    seiza.verify_model(None, store)
    seiza.remove_model("noise-tiny", store)
    with pytest.raises(seiza.EngineError, match="noise-tiny"):
        seiza.remove_model("noise-tiny", store)


def test_a_hosted_model_downloads_and_installs(tmp_path: Path) -> None:
    import functools
    import hashlib
    import http.server
    import threading

    hosted = tmp_path / "hosted"
    files = []
    for name in ("card.json", "model-f32.onnx", "model-f16.onnx"):
        data = (TINY / name).read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        key = f"artifacts/{sha256}/{name}"
        (hosted / key).parent.mkdir(parents=True)
        (hosted / key).write_bytes(data)
        files.append({"name": name, "key": key, "bytes": len(data), "sha256": sha256})
    manifest = {
        "version": "model-bundle-v1-test",
        "latest": {"denoise": "noise-tiny"},
        "models": [{"name": "noise-tiny", "files": files}],
    }
    (hosted / "manifest.json").write_text(json.dumps(manifest))

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(hosted))
    handler.log_message = lambda *_: None  # type: ignore[attr-defined]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        store, cache = tmp_path / "store", tmp_path / "cache"
        assert seiza.model_catalog(base_url=url, cache_dir=cache) == manifest
        events: list[dict] = []
        # No name: the latest noise model the manifest names.
        installed = seiza.download_model(store=store, base_url=url, cache_dir=cache, progress=events.append)
        assert installed.name == "noise-tiny"
        assert installed.precisions == ["f32", "f16"]
        assert installed.card_sha256 == hashlib.sha256((TINY / "card.json").read_bytes()).hexdigest()
        kinds = [event["event"] for event in events]
        assert kinds[0] == "fetching-manifest" and kinds[-1] == "install-complete"
        assert any(
            event["event"] == "download-started" and event["name"] == "model-f32.onnx" for event in events
        ), events
        seiza.verify_model("noise-tiny", store)
        assert [model.name for model in seiza.installed_models(store)] == ["noise-tiny"]
        assert seiza.download_model("noise-tiny", store, base_url=url, cache_dir=cache).name == "noise-tiny"
        with pytest.raises(seiza.EngineError, match="noise-tiny"):
            seiza.download_model("noise-other", store, base_url=url, cache_dir=cache)

        def refuse(event: dict) -> None:
            raise RuntimeError(f"stop at {event['event']}")

        with pytest.raises(RuntimeError, match="stop at fetching-manifest"):
            seiza.download_model("noise-tiny", store, base_url=url, cache_dir=cache, progress=refuse)
    finally:
        server.shutdown()


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


def test_a_card_s_hole_guard_is_listed_with_its_settings(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(TINY, source)
    path = source / "card.json"
    card = json.loads(path.read_text())
    card["reading"]["holes"] = {"depth": 30.0, "margin": 3, "fill_radius": 12, "restore": 5.0}
    path.write_text(json.dumps(card))

    holes = seiza.install_model(source, tmp_path / "store").holes
    assert isinstance(holes, seiza.HoleGuard)
    assert (holes.depth, holes.margin, holes.fill_radius, holes.restore) == (30.0, 3, 12, 5.0)
    listed = seiza.installed_models(tmp_path / "store")[0].holes
    assert listed == holes and hash(listed) == hash(holes), "a cache keyed on the guard finds it again"


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
    assert report.holes is None
    assert len(report.card_sha256) == 64 and report.card_sha256 != report.sha256
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
def test_a_file_run_writes_linear_or_comparison_output(tmp_path: Path) -> None:
    model = seiza.NoiseModel(str(TINY), "cpu")
    source = TINY.parent / "tiny-input.fits"
    original = tmp_path / "original.fits"
    restored = tmp_path / "restored.fits"
    linear = tmp_path / "linear.fits"
    preview = model.denoise_file(source, center=True)
    report = preview.write_comparison(original, restored)
    assert report.model == "noise-tiny"
    assert original.read_bytes() != restored.read_bytes()
    with pytest.raises(seiza.EngineError, match="separate model runs"):
        preview.write_linear(linear)
    assert model.denoise_file(source).write_linear(linear).backend == "cpu"
    assert linear.is_file()


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


@needs_runtime
def test_file_region_preview_stays_linear_and_rejects_full_export(tmp_path: Path) -> None:
    model = seiza.NoiseModel(str(TINY), "cpu")
    source = TINY.parent / "tiny-input.fits"
    result = model.denoise_file(source, region=(10, 10, 100, 80))
    output = tmp_path / "preview.fits"
    assert result.write_preview_linear(output, high=0.0, low=0.0).model == "noise-tiny"
    assert output.is_file()
    # Read through Seiza's source-window API: no display stretch or quantization.
    crop = seiza.read_image_window(output, 0, 0, 100, 80)
    expected = seiza.read_image_window(source, 10, 10, 100, 80)
    np.testing.assert_array_equal(crop.values, expected.values)
    with pytest.raises(seiza.EngineError, match="separate model runs"):
        result.write_linear(tmp_path / "full.fits")
    with pytest.raises(ValueError, match="center or region"):
        model.denoise_file(source, center=True, region=(0, 0, 10, 10))
    for region in [(0, 0, 0, 5), (127, 0, 2, 1), (0, 95, 1, 2)]:
        with pytest.raises(seiza.EngineError, match="preview region"):
            model.denoise_file(source, region=region)


@needs_runtime
def test_region_preview_context_preserves_the_rest_of_the_frame(tmp_path: Path) -> None:
    model = seiza.NoiseModel(str(TINY), "cpu")
    source = TINY.parent / "tiny-input.fits"
    result = model.denoise_file(source, region=(10, 10, 100, 80))
    output = tmp_path / "context.fits"
    result.write_preview_in_context(source, output, high=0.0, low=0.0)
    expected = seiza.read_image_window(source, 0, 0, 128, 96).values
    np.testing.assert_array_equal(seiza.read_image_window(output, 0, 0, 128, 96).values, expected)
    result.write_preview_in_context(source, output)
    actual = seiza.read_image_window(output, 0, 0, 128, 96).values
    # Planar source windows: preserve all samples outside the requested crop.
    a = actual.reshape(96, 128)
    b = expected.reshape(96, 128)
    np.testing.assert_array_equal(a[:10], b[:10])
    np.testing.assert_array_equal(a[90:], b[90:])
    np.testing.assert_array_equal(a[:, :10], b[:, :10])
    np.testing.assert_array_equal(a[:, 110:], b[:, 110:])
    assert np.any(a[10:90, 10:110] != b[10:90, 10:110])
    assert seiza.read_image_header(output)["headers"]["MDPREV"] is True


SHARPEN = FIXTURES / "sharpen"


def star_field() -> np.ndarray:
    return np.fromfile(SHARPEN / "image.f32", dtype="<f4").reshape(250, 330)


def test_a_sharpening_model_installs_with_its_encoder_and_lists_what_it_asks(tmp_path: Path) -> None:
    installed = seiza.install_model(SHARPEN / "tiny", store=tmp_path)
    assert installed.task == "sharpen"
    assert (installed.path / "encoder-f32.onnx").is_file()
    assert installed.sharpen is not None
    assert (installed.sharpen.stars, installed.sharpen.rest) == (0.46, 0.5)
    assert installed.sharpen.floor_px == 1.5 and installed.sharpen.sharpest == 0.25
    assert (installed.sharpen.scaled_px, installed.sharpen.binned_rest) == (None, None), "no scaled pass"
    seiza.verify_model("sharpen-tiny", store=tmp_path)
    noise = seiza.install_model(TINY, store=tmp_path)
    assert noise.sharpen is None, "a noise model asks for no shares"
    assert [model.name for model in seiza.installed_models(store=tmp_path)] == ["noise-tiny", "sharpen-tiny"]


def test_verifying_with_no_name_checks_every_installed_model(tmp_path: Path) -> None:
    seiza.verify_model(store=tmp_path / "empty")
    installed = seiza.install_model(SHARPEN / "tiny", store=tmp_path)
    seiza.verify_model(store=tmp_path)
    seiza.install_model(TINY, store=tmp_path)
    seiza.verify_model(store=tmp_path)

    # The sharpening model is checked too, with no name given.
    (installed.path / "encoder-f32.onnx").write_bytes(b"other")
    with pytest.raises(seiza.EngineError, match="encoder-f32.onnx"):
        seiza.verify_model(store=tmp_path)
    seiza.verify_model("noise-tiny", store=tmp_path)


@needs_runtime
def test_a_tiny_sharpening_model_matches_the_harness() -> None:
    expected = json.loads((SHARPEN / "expected.json").read_text())
    image = star_field()
    model = seiza.SharpenModel(str(SHARPEN / "tiny"), "cpu")
    assert (model.backend, model.tile) == ("cpu", 128)
    seen: list[tuple[int, int, int, int]] = []
    result = model.sharpen(image, *expected["strengths"], progress=lambda *step: seen.append(step))
    assert seen and seen[-1][2] == seen[-1][3]
    assert result.image.shape == image.shape and result.image.dtype == np.float32
    assert np.isnan(result.image[-1, 0]), "a pixel with no data stays without"
    report = result.report
    assert (report.model, report.backend, report.precision) == ("sharpen-tiny", "cpu", "f32")
    assert (report.stars, report.rest, report.rolloff) == (pytest.approx(0.46), 0.5, 0.0)
    read = report.channels[0]
    assert (read.psf_rows, read.psf_columns) == (len(expected["ys"]), len(expected["xs"]))
    assert read.psf_stars == expected["stars"] and not read.psf_global
    assert read.points == len(expected["points"])
    stride = expected["stride"]
    want = np.array([np.nan if value is None else value for value in expected["output"]], dtype=np.float32)
    got = result.image[::stride, ::stride].ravel()
    assert np.nanmax(np.abs(got - want)) < 2e-3 * expected["noise"]

    own = model.sharpen(image)
    assert (own.report.stars, own.report.rest) == (pytest.approx(0.46), 0.5), "the card's own shares, left out"
    rolled = model.sharpen(image, rolloff=expected["rolloff"])
    want = np.array([np.nan if value is None else value for value in expected["rolled"]], dtype=np.float32)
    assert np.nanmax(np.abs(rolled.image[::stride, ::stride].ravel() - want)) < 2e-3 * expected["noise"]


@needs_runtime
def test_a_scaled_pass_bins_a_wide_starred_image_as_the_harness_does(tmp_path: Path) -> None:
    expected = json.loads((SHARPEN / "expected.json").read_text())["scaled"]
    image = np.fromfile(SHARPEN / "scaled.f32", dtype="<f4").reshape(expected["height"], expected["width"])
    bundle = tmp_path / "scaled"
    shutil.copytree(SHARPEN / "tiny", bundle)
    card = json.loads((bundle / "card.json").read_text())
    card["sharpen"]["scaled"] = expected["scaled"]
    (bundle / "card.json").write_text(json.dumps(card))
    installed = seiza.install_model(bundle, store=tmp_path / "store")
    assert installed.sharpen is not None
    assert (installed.sharpen.scaled_px, installed.sharpen.binned_rest) == (3.5, 0.35)

    model = seiza.SharpenModel(str(bundle), "cpu")
    result = model.sharpen(image, *expected["strengths"])
    assert result.report.channels[0].binned == expected["factor"]
    assert result.report.binned_rest == pytest.approx(0.35) and not result.report.full_size_only
    stride = expected["stride"]
    want = np.array([np.nan if value is None else value for value in expected["output"]], dtype=np.float32)
    assert np.nanmax(np.abs(result.image[::stride, ::stride].ravel() - want)) < 2e-3 * expected["noise"]

    held = model.sharpen(image, *expected["strengths"], full_size_only=True, binned_rest=0.5)
    assert held.report.channels[0].binned == 1 and held.report.full_size_only
    assert held.report.binned_rest is None, "nothing binned to ask of"
    plain = seiza.SharpenModel(str(SHARPEN / "tiny"), "cpu").sharpen(image, *expected["strengths"])
    assert np.array_equal(held.image, plain.image, equal_nan=True), "held to full size, the plain card's run"
    asked = model.sharpen(image, *expected["strengths"], binned_rest=0.6)
    assert asked.report.binned_rest == pytest.approx(0.6)
    with pytest.raises(ValueError, match="share"):
        model.sharpen(image, binned_rest=0.0)


def write_fits(path: Path, image: np.ndarray) -> None:
    """A minimal single-precision FITS image."""
    cards = ["SIMPLE  =                    T", "BITPIX  =                  -32", "NAXIS   =                    2",
             f"NAXIS1  = {image.shape[1]:>20}", f"NAXIS2  = {image.shape[0]:>20}", "END"]
    header = "".join(card.ljust(80) for card in cards).ljust(2880).encode("ascii")
    data = image.astype(">f4").tobytes()
    path.write_bytes(header + data + b"\0" * (-len(data) % 2880))


@needs_runtime
def test_a_file_is_sharpened_to_a_file_that_says_what_ran(tmp_path: Path) -> None:
    image = star_field()
    source, output = tmp_path / "stars.fits", tmp_path / "sharp.fits"
    write_fits(source, image)
    model = seiza.SharpenModel(str(SHARPEN / "tiny"), "cpu")
    report = model.sharpen_file(source, output, 0.6, rolloff=0.5)
    assert (report.model, report.stars, report.rest, report.rolloff) == ("sharpen-tiny", pytest.approx(0.6), 0.5, 0.5)
    header = output.read_bytes()[:5760].decode("ascii", "replace")
    assert "SEIZAMDL" in header and "MDSTARS" in header and "MDREST" in header
    written = np.frombuffer(output.read_bytes(), dtype=">f4", offset=header.index("END     ") // 2880 * 2880 + 2880,
                            count=image.size).reshape(image.shape)
    direct = model.sharpen(image, 0.6, rolloff=0.5).image
    assert np.array_equal(written, direct, equal_nan=True)
    with pytest.raises(seiza.EngineError):
        model.sharpen_file(tmp_path / "missing.fits", output)


@needs_runtime
def test_a_sharpening_refuses_a_bad_share_a_bare_image_and_a_noise_model() -> None:
    model = seiza.SharpenModel(str(SHARPEN / "tiny"), "cpu")
    with pytest.raises(ValueError, match="share"):
        model.sharpen(star_field(), stars=1.5)
    rng = np.random.default_rng(3)
    bare = (100.0 + rng.normal(0.0, 2.0, (160, 200))).astype(np.float32)
    with pytest.raises(seiza.TooFewStarsError):
        model.sharpen(bare)
    with pytest.raises(seiza.EngineError, match="sharpening"):
        seiza.SharpenModel(str(TINY), "cpu")
    with pytest.raises(seiza.EngineError, match="noise model"):
        seiza.NoiseModel(str(SHARPEN / "tiny"), "cpu")
