"""Reading a frame's role from its header, and planning which calibration
frames suit a set of lights: the calls behind seiza_probe_frame_json and
seiza_calibration_plan_json in the C ABI."""

from pathlib import Path

import numpy as np
import pytest

import seiza

fits = pytest.importorskip("astropy.io.fits")


def write_frame(path, **cards):
    header = fits.Header()
    for key, value in cards.items():
        header[key.replace("_", "-")] = value
    fits.writeto(path, np.full((8, 10), 100.0, dtype=np.float32), header, overwrite=True)
    return path


def signature(exposure_seconds=60.0, camera_temp_c=-10.0, captured_at_unix=1000):
    return seiza.FrameSignature(
        camera="ASI2600MM",
        width=100,
        height=80,
        channels=1,
        exposure_seconds=exposure_seconds,
        camera_temp_c=camera_temp_c,
        captured_at_unix=captured_at_unix,
    )


LIGHT = seiza.CalibrationRecord("light.fits", signature(), role="light")


def test_a_probe_reads_role_master_flag_signature_and_state(tmp_path):
    path = write_frame(
        tmp_path / "master-dark.fits",
        IMAGETYP="Master Dark Frame",
        INSTRUME="ASI2600MM",
        GAIN=100,
        EXPTIME=60.0,
        DATE_OBS="2026-01-02T03:04:05",
        BIASSUB=True,
    )

    probe = seiza.probe_frame(path)

    assert probe.path == path
    assert probe.format == "FITS"
    assert probe.raw_image_type == "Master Dark Frame"
    assert probe.role == "dark"
    assert probe.is_master
    assert probe.signature.camera == "ASI2600MM"
    assert probe.signature.gain == 100
    assert probe.signature.exposure_seconds == 60.0
    assert (probe.signature.width, probe.signature.height) == (10, 8)
    assert probe.signature.captured_at_unix is not None
    assert probe.calibration_state.bias_subtracted
    assert not probe.calibration_state.dark_subtracted
    assert not probe.calibration_state.flat_normalized


def test_a_probe_refuses_what_is_not_fits_or_xisf(tmp_path):
    with pytest.raises(seiza.StackError, match="not a FITS or XISF path"):
        seiza.probe_frame(tmp_path / "frame.png")
    with pytest.raises(seiza.StackError, match="failed to read FITS header"):
        seiza.probe_frame(tmp_path / "missing.fits")


def test_darks_are_ordered_by_time_and_each_exclusion_says_why():
    plan = seiza.plan_calibration(
        "dark",
        LIGHT,
        [
            seiza.CalibrationRecord(
                "dark-later.fits", signature(camera_temp_c=-9.8, captured_at_unix=1100), role="dark"
            ),
            seiza.CalibrationRecord(
                "dark-nearest.fits", signature(camera_temp_c=-10.1, captured_at_unix=1005), role="dark"
            ),
            seiza.CalibrationRecord(
                "wrong-exposure.fits", signature(exposure_seconds=120.0, captured_at_unix=1001), role="dark"
            ),
            seiza.CalibrationRecord("a-flat.fits", signature(), role="flat"),
        ],
    )

    assert plan.kind == "dark"
    assert plan.minimum == 2
    assert plan.ready
    assert plan.matched_paths == [Path("dark-nearest.fits"), Path("dark-later.fits")]
    assert plan.selected_paths == plan.matched_paths
    assert plan.excluded == [
        (Path("wrong-exposure.fits"), "exposure-mismatch"),
        (Path("a-flat.fits"), "role-mismatch"),
    ]


def test_a_bias_lets_darks_of_another_exposure_match():
    dark = seiza.CalibrationRecord("dark-120.fits", signature(exposure_seconds=120.0), role="dark")

    unscaled = seiza.plan_calibration("dark", LIGHT, [dark], minimum=1)
    assert not unscaled.ready
    assert unscaled.excluded == [(Path("dark-120.fits"), "exposure-mismatch")]

    scaled = seiza.plan_calibration("dark", LIGHT, [dark], minimum=1, bias_available=True)
    assert scaled.ready
    assert scaled.selected_paths == [Path("dark-120.fits")]


def test_flats_must_suit_every_light_they_serve():
    ha = seiza.CalibrationRecord("ha.fits", seiza.FrameSignature(camera="ASI2600MM", filter="Ha"), role="light")
    oiii = seiza.CalibrationRecord("oiii.fits", seiza.FrameSignature(camera="ASI2600MM", filter="OIII"), role="light")
    flat = seiza.CalibrationRecord("ha-flat.fits", seiza.FrameSignature(camera="ASI2600MM", filter="Ha"), role="flat")

    alone = seiza.plan_calibration("flat", ha, [flat], minimum=1)
    assert alone.selected_paths == [Path("ha-flat.fits")]

    both = seiza.plan_calibration("flat", ha, [flat], references=[ha, oiii], minimum=1)
    assert not both.ready
    assert both.excluded == [(Path("ha-flat.fits"), "optics-mismatch")]


def test_probes_plan_as_records(tmp_path):
    light = seiza.probe_frame(
        write_frame(tmp_path / "light.fits", IMAGETYP="LIGHT", INSTRUME="ASI2600MM", EXPTIME=60.0)
    )
    darks = [
        seiza.probe_frame(
            write_frame(tmp_path / f"dark-{index}.fits", IMAGETYP="DARK", INSTRUME="ASI2600MM", EXPTIME=60.0)
        )
        for index in range(2)
    ]
    bias = seiza.probe_frame(write_frame(tmp_path / "bias.fits", IMAGETYP="BIAS", INSTRUME="ASI2600MM"))

    plan = seiza.plan_calibration("dark", light, [*darks, bias])

    assert plan.ready
    assert sorted(plan.selected_paths) == sorted(dark.path for dark in darks)
    assert plan.excluded == [(bias.path, "role-mismatch")]


def test_bad_plans_are_refused():
    dark = seiza.CalibrationRecord("dark.fits", signature(), role="dark")
    with pytest.raises(ValueError, match="kind"):
        seiza.plan_calibration("light", LIGHT, [dark])
    with pytest.raises(ValueError, match="role light"):
        seiza.plan_calibration("dark", dark, [])
    with pytest.raises(ValueError, match="rotation tolerance"):
        seiza.plan_calibration("dark", LIGHT, [dark], tolerances=seiza.MatchTolerances(rotation_deg=float("nan")))
    with pytest.raises(ValueError, match="duplicate calibration candidate"):
        seiza.plan_calibration("dark", LIGHT, [dark, dark])
    with pytest.raises(ValueError, match="include the primary"):
        seiza.plan_calibration("dark", LIGHT, [dark], references=[seiza.CalibrationRecord("other.fits", signature(), role="light")])
    with pytest.raises(ValueError, match="frame role"):
        seiza.CalibrationRecord("dark.fits", signature(), role="darkish")
    with pytest.raises(TypeError, match="CalibrationRecord or FrameProbe"):
        seiza.plan_calibration("dark", LIGHT, ["dark.fits"])


def test_a_record_keeps_what_it_was_given():
    record = seiza.CalibrationRecord("dark.fits", signature())
    assert record.path == Path("dark.fits")
    assert record.role == "unknown"
    assert record.signature.camera == "ASI2600MM"


def test_a_light_is_planned_flats_at_the_angle_the_host_measured():
    def record(path, role, rotation_deg):
        signature = seiza.FrameSignature(
            camera="ASI2600MM", width=100, height=80, channels=1, rotation_deg=rotation_deg
        )
        return seiza.CalibrationRecord(path, signature, role=role)

    light = record("light.fits", "light", 113.8)
    flat = record("flat.fits", "flat", 3.2)

    def plan(rotation):
        tolerances = seiza.MatchTolerances(light_rotation=rotation)
        return seiza.plan_calibration("flat", light, [flat], minimum=1, tolerances=tolerances)

    assert plan("recorded").selected_paths == []
    assert [str(path) for path in plan(3.5).selected_paths] == ["flat.fits"]
    assert [str(path) for path in plan("unknown").selected_paths] == ["flat.fits"]
