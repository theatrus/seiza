"""Project analysis stays a thin, schema-preserving core binding."""
import pytest
import seiza


def test_analysis_rejected_frame_and_portable_html(tmp_path):
    report = seiza.analyze_project([tmp_path / "missing.fits"])
    assert report["schema_version"] == 1
    assert report["algorithm_version"] == 1
    assert report["groups"] == []
    frame = report["frames"][0]
    assert frame["status"] == "rejected"
    assert frame["error"]
    assert "capture_time" in frame
    assert "registration_rms_pixels" in frame
    report["warnings"].append("</script><img src=x onerror=alert(1)>")
    html = seiza.analysis_html(report)
    assert "Seiza project analysis" in html
    assert "<img src=x" not in html


@pytest.mark.parametrize("options", [{"samples": 0}, {"samples": 2097153}, {"max_frames": 0}, {"filter": " "}])
def test_analysis_validates_before_reading(options):
    with pytest.raises(ValueError):
        seiza.analyze_project(["missing.fits"], **options)


def test_analysis_empty_input():
    with pytest.raises(ValueError, match="no FITS/XISF"):
        seiza.analyze_project([])


def test_analysis_successful_project(tmp_path):
    # Independent fixture writer, as in test_stacking.py; production reads use
    # seiza-image-io. Different numeric scales must preserve physical photometry.
    fits = pytest.importorskip("astropy.io.fits")
    import numpy as np
    yy, xx = np.mgrid[:400, :400]
    scene = np.full((400, 400), 1000.0)
    for i in range(36):
        x = 35 + (i % 6) * 58 + (i * 17) % 13
        y = 35 + (i // 6) * 58 + (i * 11) % 17
        scene += (1800 + (i * 79) % 2000) * np.exp(-((xx-x)**2+(yy-y)**2)/18)
    rng = np.random.default_rng(901)
    for i in range(4):
        header = fits.Header({"FILTER": "Ha", "EXPTIME": 60.0})
        scale = 0.01 if i % 2 else 1.0
        fits.writeto(tmp_path / f"{i}.fits", ((scene+rng.normal(0,10,scene.shape))*scale).astype("float32"), header)
    report = seiza.analyze_project([tmp_path])
    group = report["groups"][0]
    assert group["accepted_frames"] == 4
    assert group["integration_seconds"] == 240
    assert group["common_patches"] == 100
    assert group["curve"][0]["noise"] is None
    assert group["curve"][1]["noise"] > 0
    assert report["frames"][1]["gain"] == pytest.approx(100, rel=0.03)
    assert "Frame health" in seiza.analysis_html(report)
