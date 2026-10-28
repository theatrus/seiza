"""Project analysis stays a thin, schema-preserving core binding."""
import pytest
import seiza


def test_analysis_rejected_frame_and_portable_html(tmp_path):
    report = seiza.analyze_project([tmp_path / "missing.fits"])
    assert report["schema_version"] == 3
    assert report["algorithm_version"] == 3
    assert report["groups"] == []
    frame = report["frames"][0]
    assert frame["measurement_status"] == "failed"
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
    assert group["depth_frames"] == 4
    assert group["integration_seconds"] == 240
    assert group["common_patches"] == 100
    assert group["curve"][0]["noise"] is None
    assert group["curve"][1]["noise"] > 0
    assert report["frames"][1]["gain"] == pytest.approx(100, rel=0.03)
    assert "Frame health" in seiza.analysis_html(report)

    for metric, frame_key in [
        ("star_count", "star_count"), ("hfr", "median_hfr"),
        ("fwhm", "median_fwhm"), ("eccentricity", "median_eccentricity"),
        ("noise", "noise"), ("background", "background"),
        ("star_snr", "median_star_snr"),
    ]:
        values = [f[frame_key] for f in report["frames"] if f[frame_key] is not None]
        assert group["median_" + metric] == pytest.approx(float(np.median(values)))
    assert group["median_normalized_noise"] == pytest.approx(
        float(np.median([f["noise"] * f["gain"] for f in report["frames"]])))
    snr = [f["median_star_snr"] for f in report["frames"]]
    assert min(snr) > 0
    assert max(snr) / min(snr) < 1.1  # Dimensionless despite 100x stored-unit change.

    metrics = seiza.analyze_project([tmp_path], metrics_only=True)
    assert metrics["options"]["metrics_only"] is True
    assert metrics["groups"][0]["measured_frames"] == 4
    assert metrics["groups"][0]["depth_frames"] == 0
    assert metrics["groups"][0]["reference"] is None
    assert metrics["groups"][0]["median_normalized_noise"] is None
    for a, b in zip(report["frames"], metrics["frames"]):
        assert b["measurement_status"] == "measured"
        assert b["depth_status"] == "not_requested"
        for key in ("star_count", "median_hfr", "median_eccentricity", "median_star_snr", "noise", "background"):
            assert a[key] == b[key]
    assert metrics["timings"]["registration_seconds"] == 0
    assert "Depth analysis not requested" in seiza.analysis_html(metrics)

@pytest.mark.parametrize("kind,components", [("cfa", ["R", "G_red", "G_blue", "B"]), ("rgb", ["R", "G", "B"])])
def test_color_component_schema_and_sampling(tmp_path, kind, components):
    fits = pytest.importorskip("astropy.io.fits")
    import numpy as np
    path = tmp_path / "osc.fits"
    header = fits.Header({"FILTER": "OSC", "EXPTIME": 60.0})
    if kind == "cfa":
        header["BAYERPAT"] = "RGGB"
        header["XBAYROFF"] = 1
        image = np.full((80, 80), 1000, dtype="float32")
    else:
        image = np.stack([np.full((80, 80), 1000 * i, dtype="float32") for i in (1, 2, 3)])
    fits.writeto(path, image, header)
    report = seiza.analyze_project([path], metrics_only=True)
    assert [g["component"] for g in report["groups"]] == components
    assert all(g["image_kind"] == kind for g in report["groups"])
    assert all(g["measured_frames"] == 1 for g in report["groups"])
    for frame in report["frames"]:
        assert frame["measurement_status"] == "measured"
        assert frame["depth_status"] == "not_requested"
        assert frame["sampling"]["source_width"] == 80
        assert frame["sampling"]["spacing_pixels"] == (2 if kind == "cfa" else 1)
    if kind == "cfa":
        assert report["frames"][0]["sampling"]["offset_pixels"] == [1, 0]
        assert report["frames"][0]["sampling"]["bayer_offsets"] == [1, 0]
    else:
        assert [f["background"] for f in report["frames"]] == [1000, 2000, 3000]
    assert "Filter / component" in seiza.analysis_html(report)
