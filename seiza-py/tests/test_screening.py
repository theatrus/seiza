"""Judging calibration frames before a master is built, and residual flat
responses: the calls behind seiza_calibration_measure_*_json,
seiza_calibration_screen_*_json and the residual flat calls in the C ABI."""

import numpy as np
import pytest

import seiza

fits = pytest.importorskip("astropy.io.fits")


def write(path, image):
    fits.writeto(path, np.asarray(image, dtype=np.float32), overwrite=True)
    return path


def noise(seed, shape):
    return np.random.default_rng(seed).uniform(-0.5, 0.5, shape).astype(np.float32)


def test_darks_are_measured_from_files_and_screened_as_a_set(tmp_path):
    paths = [
        write(tmp_path / f"dark-{index}.fits", 500.0 + (40.0 if index == 3 else 0.0) + 8.0 * noise(index, (64, 64)))
        for index in range(4)
    ]
    darks = [seiza.measure_dark(path, hot_sigma=10.0) for path in paths]
    first = darks[0]
    assert (first.width, first.height) == (64, 64)
    assert 499.0 < first.level < 502.0
    assert first.noise > 0.0
    assert first.hot_pixels == 0
    assert first.algorithm_version >= 1
    for name in ("sigma", "row_banding", "column_banding", "hot_threshold"):
        assert np.isfinite(getattr(first, name))

    verdicts = seiza.screen_darks(darks)
    assert [verdict.grade for verdict in verdicts[:3]] == ["ok", "ok", "ok"]
    assert verdicts[3].grade == "poor"
    finding = verdicts[3].findings[0]
    assert (finding.code, finding.grade) == ("level_excess", "poor")
    assert finding.value >= finding.limit

    lenient = seiza.screen_darks(darks, level_excess=(1.0e9, 2.0e9))
    assert all(finding.code != "level_excess" for finding in lenient[3].findings)
    assert seiza.screen_bias(darks)[3].findings[0].code == "level"

    # A measurement keeps as JSON and judges the same when read back.
    stored = [seiza.DarkMeasurement.from_json(dark.to_json()) for dark in darks]
    assert [verdict.grade for verdict in seiza.screen_darks(stored)] == [
        verdict.grade for verdict in verdicts
    ]


def test_flats_are_measured_from_files_and_screened_as_a_set(tmp_path):
    paths = [write(tmp_path / f"flat-{index}.fits", 30000.0 + 100.0 * noise(10 + index, (64, 64))) for index in range(3)]
    paths.append(write(tmp_path / "clipped.fits", np.full((64, 64), 65535.0)))
    flats = [seiza.measure_flat(path, full_scale=65535.0) for path in paths]
    flat = flats[0]
    assert flat.channel_levels == [pytest.approx(flat.level, rel=1e-3)]
    assert flat.clipped_fraction == 0.0
    assert flats[3].clipped_fraction == 1.0
    profile = flat.illumination
    assert profile.response.shape == (profile.rows, profile.columns)
    assert flat.feature_count == len(flat.features)
    assert flat.detection_depth > 0.0
    assert seiza.FlatMeasurement.from_json(flat.to_json()).level == flat.level

    report = seiza.screen_flats(flats)
    assert report.frames[0].grade == "ok"
    assert report.frames[3].grade == "poor"
    assert report.frames[3].findings[0].code == "clipped"
    assert report.master.grade in ("ok", "suspect", "poor")
    assert report.steady_features >= 0
    assert report.changing_features >= 0
    assert report.carried_features >= 0

    # Frames of different sizes cannot form one set.
    small = seiza.measure_flat(write(tmp_path / "small.fits", 30000.0 + 100.0 * noise(20, (48, 48))))
    with pytest.raises(ValueError, match="cannot form one set"):
        seiza.screen_flats([flat, small])
    with pytest.raises(ValueError, match="steady_presence"):
        seiza.screen_flats(flats, steady_presence=0.2)
    with pytest.raises(seiza.StackError):
        seiza.measure_dark(tmp_path / "missing.fits")


def test_a_wide_shallow_disk_is_a_large_feature_with_a_distance(tmp_path):
    # Filter dust: 0.5% deep and 80 pixels across, far below the compact
    # threshold, on a vignetted flat.
    height, width = 256, 384
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = width / 2.0, height / 2.0
    response = 30000.0 * (1.0 - 0.25 * ((x - cx) ** 2 + (y - cy) ** 2) / (cx * cx + cy * cy))
    image = response * (1.0 + 0.01 * np.random.default_rng(4).standard_normal((height, width)))
    image = np.where(np.hypot(x - 150.0, y - 120.0) <= 40.0, image * 0.995, image)
    path = write(tmp_path / "filter-dust.fits", image)

    flat = seiza.measure_flat(path, focal_ratio=4.9, pixel_size_um=3.76)
    disks = [feature for feature in flat.features if feature.shape == "disk"]
    assert len(disks) == 1, flat.features
    disk = disks[0]
    assert abs(disk.x - 150.0) < 3.0 and abs(disk.radius - 40.0) < 4.0
    # 80 pixels at f/4.9 and 3.76 µm: about 1.5 mm in front of the sensor.
    assert disk.distance_mm == pytest.approx(80.0 * 4.9 * 3.76 / 1000.0, rel=0.15)
    assert disk.hole is None
    assert disk.seen_in is None, "seen_in is only for dust found on a session's stack"
    assert isinstance(flat.vignetting, list)
    assert seiza.FlatMeasurement.from_json(flat.to_json()).features[0].shape == "disk"

    # Without the optics there is no distance; an obstruction alone is refused.
    assert seiza.measure_flat(path).features[0].distance_mm is None
    with pytest.raises(ValueError, match="focal ratio"):
        seiza.measure_flat(path, obstruction=0.35)


def residual_crop(size, offset, shadow):
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    background = 1000.0 + 1.5 * x + 0.75 * y + offset
    radius = np.hypot(x - 15.5, y - 15.5)
    response = np.where(shadow & (radius >= 6.0) & (radius <= 9.0), 0.8, 1.0)
    star = np.where((np.arange(size)[None, :] + int(offset)) % size == np.arange(size)[:, None], 2000.0, 0.0)
    return (background * response + star).astype(np.float32)


def test_a_residual_flat_response_is_built_checked_and_applied():
    crops = [residual_crop(32, index * 3.0, True) for index in range(7)]
    response, diagnostics = seiza.build_residual_flat(crops, smoothing_sigma=0.8, edge_feather_fraction=0.0)
    assert response.shape == (32, 32)
    assert response.dtype == np.float32
    assert diagnostics.sample_count == 7
    assert diagnostics.corrected_samples > 40
    assert diagnostics.total_samples == 32 * 32
    assert diagnostics.largest_connected_pixels > 20
    assert diagnostics.minimum_response >= 1.0 / 1.2 - 1e-6
    assert diagnostics.maximum_applied_gain <= 1.2 + 1e-5
    assert diagnostics.algorithm_version >= 1
    seiza.validate_residual_flat_response(response)

    light = residual_crop(40, 0.0, False)
    corrected = seiza.apply_residual_flat_response(response, light, 4, 4)
    assert corrected[4 + 8, 4 + 15] > light[4 + 8, 4 + 15]
    np.testing.assert_array_equal(corrected[:4], light[:4])

    with pytest.raises(ValueError, match="extends past"):
        seiza.apply_residual_flat_response(response, light, 20, 20)
    bad = response.copy()
    bad[0, 0] = 1.5
    with pytest.raises(ValueError, match="in \\(0, 1\\]"):
        seiza.validate_residual_flat_response(bad)
    plain = [residual_crop(32, index * 3.0, False) for index in range(7)]
    with pytest.raises(ValueError):
        seiza.build_residual_flat(plain)
