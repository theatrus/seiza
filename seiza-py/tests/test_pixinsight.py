"""PixInsight MLDenoise bindings, exercised against a stand-in PixInsight.

The stand-in reads ``-r=script,request,result`` as PixInsight runs a
script, copies the exchange image to the output, and writes the result the
real script writes, so these tests need no PixInsight and no model. Set
``SEIZA_MLDENOISE_MODEL`` to an ``.xmlm`` file to also run the installed
PixInsight.
"""

import os
import stat
import struct
import sys

import numpy as np
import pytest
import seiza



def fake(expected=None):
    """A stand-in PixInsight that checks the request holds `expected`."""
    return f"""#!/usr/bin/env python3
import json, shutil, sys
run = next(a for a in sys.argv if a.startswith('-r='))[3:].split(',')
open(run[3], 'w').write('started')
request = json.load(open(run[1]))
for key, value in {expected or {}!r}.items():
    assert request[key] == value, (key, request[key])
shutil.copy(request['input'], request['output'])
json.dump({{'ok': True, 'pixinsightVersion': '1.9.6', 'seconds': 1.25}}, open(run[2], 'w'))
"""


FAILING = """#!/usr/bin/env python3
import json, sys
run = next(a for a in sys.argv if a.startswith('-r='))[3:].split(',')
json.dump({'ok': False, 'error': 'Error: MLDenoise did not run',
           'log': '*** Failed to load the selected model.'}, open(run[2], 'w'))
"""

HEADER = (
    '<?xml version="1.0" encoding="UTF-8"?><xmlm version="1.0" '
    'xmlns="http://www.pixinsight.com/xmlm"><CreationTime>2026-10-05T22:31:37Z'
    "</CreationTime><Process>MLDenoise</Process><Version>V3</Version>"
    '<Model role="mono" sha256="aa">mono.onnx</Model>'
    '<Model role="weights" sha256="bb">weights.bin</Model></xmlm>'
)

runs_stand_in = pytest.mark.skipif(
    not sys.platform.startswith("linux") and sys.platform != "darwin",
    reason="the stand-in PixInsight is a Python script run as a program",
)


def executable(directory, name, text):
    path = directory / name
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "MLDenoise.xmlm"
    data = HEADER.encode()
    path.write_bytes(b"XMLM0100" + struct.pack("<II", len(data), 0) + data + b"\0" * 32)
    return path


def test_the_model_header_gives_its_routes_and_weights(model):
    header = seiza.pixinsight_ml_denoise_model(model)
    assert header.process == "MLDenoise"
    assert header.format_version == "V3"
    assert header.created == "2026-10-05T22:31:37Z"
    assert header.mono and not header.rgb
    assert header.weights_sha256 == "bb"
    assert len(header.header_sha256) == 64
    with pytest.raises(ValueError, match="not a PixInsight model file"):
        seiza.pixinsight_ml_denoise_model(__file__)
    with pytest.raises(FileNotFoundError):
        seiza.pixinsight_ml_denoise_model(model.parent / "missing.xmlm")


def test_options_outside_pixinsights_ranges_are_refused(model):
    image = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="amount"):
        seiza.pixinsight_ml_denoise_image(image, model, amount=1.5, executable="/nonexistent")
    with pytest.raises(ValueError, match="tile overlap"):
        seiza.pixinsight_ml_denoise_image(image, model, overlap_percent=60, executable="/nonexistent")
    with pytest.raises(ValueError, match="precision"):
        seiza.pixinsight_ml_denoise_image(image, model, precision="full", executable="/nonexistent")
    support = seiza.MlDenoiseLocalSupport(smoothness=9)
    with pytest.raises(ValueError, match="smoothness"):
        seiza.pixinsight_ml_denoise_image(image, model, local_support=support, executable="/nonexistent")
    with pytest.raises(ValueError, match="timeout"):
        seiza.pixinsight_ml_denoise_image(image, model, timeout_seconds=0, executable="/nonexistent")
    with pytest.raises(FileNotFoundError):
        seiza.pixinsight_ml_denoise_image(image, model.parent / "missing.xmlm", executable="/nonexistent")


@runs_stand_in
def test_an_image_returns_in_its_own_units(tmp_path, model):
    expected = {
        "amount": 0.5,
        "overlapPercent": 30,
        "precision": "high",
        "localSupport": {"clipLow": -1.0, "background": 0.3, "smoothness": 3},
    }
    stand_in = executable(tmp_path, "PixInsight", fake(expected))
    image = np.array([[-12.0, np.nan, 250.5], [1000.0, 40000.0, 65000.0]], dtype=np.float32)
    support = seiza.MlDenoiseLocalSupport(clip_low=-1.0, background=0.3, smoothness=3)
    denoised = seiza.pixinsight_ml_denoise_image(
        image,
        model,
        amount=0.5,
        overlap_percent=30,
        precision="high",
        local_support=support,
        executable=stand_in,
    )
    assert denoised.image.shape == image.shape
    # NaN marks an uncovered pixel, and comes back as NaN.
    np.testing.assert_allclose(denoised.image, image, rtol=1e-6, atol=1e-2, equal_nan=True)
    assert denoised.pixinsight_version == "1.9.6"
    assert denoised.seconds == 1.25
    assert denoised.model.weights_sha256 == "bb"
    # An RGB image needs a model with an RGB route; this one has none.
    with pytest.raises(ValueError, match="cannot denoise a 3-channel image"):
        seiza.pixinsight_ml_denoise_image(np.ones((2, 2, 3), dtype=np.float32), model, executable=stand_in)


@runs_stand_in
def test_a_file_is_denoised_with_the_requested_options(tmp_path, model):
    from astropy.io import fits

    stand_in = executable(tmp_path, "PixInsight", fake({"amount": 0.75, "localSupport": None}))
    source = tmp_path / "light.fits"
    data = np.arange(12, dtype=np.float32).reshape(3, 4) * 100.0
    fits.PrimaryHDU(data, header=fits.Header({"OBJECT": "M31"})).writeto(source)
    output = tmp_path / "denoised.fits"
    run = seiza.pixinsight_ml_denoise_file(source, output, model, amount=0.75, executable=stand_in)
    assert run.output == output
    assert run.pixinsight_version == "1.9.6"
    with fits.open(output) as written:
        assert written[0].header["OBJECT"] == "M31"
        np.testing.assert_allclose(written[0].data, data, atol=1e-2)


@runs_stand_in
def test_a_failed_run_explains_what_pixinsight_said(tmp_path, model):
    failing = executable(tmp_path, "PixInsight", FAILING)
    with pytest.raises(RuntimeError, match="Failed to load the selected model"):
        seiza.pixinsight_ml_denoise_image(np.ones((4, 4), dtype=np.float32), model, executable=failing)


@pytest.mark.skipif(
    not os.environ.get("SEIZA_MLDENOISE_MODEL") or seiza.pixinsight_locate() is None,
    reason="needs PixInsight and SEIZA_MLDENOISE_MODEL",
)
def test_the_installed_pixinsight_denoises_an_image():
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:256, 0:256]
    star = 30000.0 * np.exp(-((x - 128.0) ** 2 + (y - 128.0) ** 2) / 8.0)
    image = (1000.0 + star + rng.normal(0.0, 50.0, (256, 256))).astype(np.float32)
    denoised = seiza.pixinsight_ml_denoise_image(image, os.environ["SEIZA_MLDENOISE_MODEL"], amount=1.0)
    corner = slice(0, 64), slice(0, 64)
    assert denoised.image[corner].std() < image[corner].std() * 0.8
    assert abs(float(denoised.image[corner].mean()) - 1000.0) < 50.0
