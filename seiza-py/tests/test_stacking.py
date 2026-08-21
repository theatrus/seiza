"""Native Python coverage for batch, live, and calibration stacking APIs."""

import numpy as np
import pytest
from astropy.io import fits

import seiza


def synthetic_star_field(height=128, width=160):
    rng = np.random.default_rng(29)
    image = rng.normal(100.0, 1.5, size=(height, width)).astype(np.float32)
    yy, xx = np.mgrid[:height, :width]
    positions = [
        (16.4, 19.7),
        (28.1, 71.3),
        (34.8, 132.2),
        (49.7, 43.1),
        (58.3, 103.4),
        (70.2, 22.8),
        (76.5, 82.7),
        (87.8, 143.1),
        (96.2, 54.4),
        (104.1, 116.8),
        (113.0, 31.2),
        (118.4, 91.5),
    ]
    for index, (y, x) in enumerate(positions):
        amplitude = 900.0 + index * 130.0
        image += amplitude * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 3.2).astype(
            np.float32
        )
    return image


def no_adjustment_options():
    return seiza.StackOptions(normalization="none", rejection="none")


def test_stack_options_reject_unknown_modes():
    with pytest.raises(ValueError, match="normalization"):
        seiza.StackOptions(normalization="mystery")
    with pytest.raises(ValueError, match="rejection"):
        seiza.StackOptions(rejection="mystery")
    with pytest.raises(ValueError, match="delta-sigma"):
        seiza.StackOptions(rejection_warmup=1)


def test_live_stacker_accepts_numpy_and_returns_owned_snapshot():
    image = synthetic_star_field()
    stacker = seiza.LiveStacker.from_array(image, options=no_adjustment_options())

    disposition = stacker.push(image.copy())
    assert disposition.accepted
    assert disposition.matched_stars >= 6
    assert disposition.registration_rms_pixels < 0.1
    assert stacker.accepted_frames == 2

    snapshot = stacker.snapshot()
    assert snapshot.image.shape == image.shape
    assert snapshot.variance.shape == image.shape
    assert snapshot.coverage.shape == image.shape
    assert snapshot.rejected_samples.shape == image.shape
    assert snapshot.image.dtype == np.float32
    assert snapshot.coverage.dtype == np.uint32
    np.testing.assert_allclose(snapshot.image, image, rtol=0.0, atol=1.0e-3)
    assert np.all(snapshot.coverage == 2)

    # Returned arrays are copies and cannot alter the live accumulator.
    snapshot.image[:] = 0.0
    assert np.any(stacker.snapshot().image != 0.0)

    final = stacker.finish()
    assert final.accepted_frames == 2
    with pytest.raises(RuntimeError, match="finished"):
        stacker.snapshot()


def test_live_stacker_context_reopens_and_continues(tmp_path):
    image = synthetic_star_field()
    stacker = seiza.LiveStacker.from_array(image, options=no_adjustment_options())
    assert stacker.push(image).accepted
    context = tmp_path / "live.seiza-stack"
    stacker.save_context(context)

    resumed = seiza.LiveStacker.open_context(context)
    assert resumed.accepted_frames == 2
    assert resumed.push(image).accepted
    final = resumed.finish()
    assert final.accepted_frames == 3
    np.testing.assert_allclose(final.image, image, rtol=0.0, atol=1.0e-3)
    assert np.all(final.coverage == 3)


def test_live_stacker_accepts_and_registers_meridian_flipped_frame():
    image = synthetic_star_field()
    flipped = np.ascontiguousarray(np.rot90(image, 2))
    stacker = seiza.LiveStacker.from_array(image, options=no_adjustment_options())

    disposition = stacker.push(flipped)

    assert disposition.accepted
    assert disposition.matched_stars >= 6
    assert abs(abs(disposition.rotation_degrees) - 180.0) < 1.0
    assert disposition.registration_rms_pixels < 0.5
    snapshot = stacker.finish()
    assert snapshot.accepted_frames == 2
    np.testing.assert_allclose(snapshot.image, image, rtol=0.0, atol=1.0)


def test_batch_fits_stack_writes_linear_output_and_diagnostics(tmp_path):
    image = synthetic_star_field()
    first = tmp_path / "light-001.fits"
    second = tmp_path / "light-002.fits"
    output = tmp_path / "stack.fits"
    fits.writeto(first, image, overwrite=True)
    fits.writeto(second, image, overwrite=True)

    result = seiza.stack_fits(
        [first, second], output, options=no_adjustment_options()
    )

    assert result.output == output
    assert result.accepted_frames == 2
    assert result.rejected_frames == 0
    assert len(result.frames) == 1
    assert result.frames[0].accepted
    with fits.open(output) as hdus:
        assert hdus[0].header["STACKCNT"] == 2
        assert hdus[0].data.dtype.kind == "f"
        np.testing.assert_allclose(hdus[0].data, image, rtol=0.0, atol=1.0e-3)

    with pytest.raises(ValueError, match="requires dark"):
        seiza.stack_fits(
            [first, second],
            tmp_path / "unused.fits",
            options=no_adjustment_options(),
            dark_exposure_seconds=60.0,
        )
    with pytest.raises(ValueError, match="duplicate input"):
        seiza.stack_fits(
            [first, first],
            tmp_path / "duplicate.fits",
            options=no_adjustment_options(),
        )


def test_batch_fits_stack_registers_meridian_flip_without_large_fixtures(tmp_path):
    image = synthetic_star_field()
    first = tmp_path / "before-flip.fits"
    second = tmp_path / "after-flip.fits"
    output = tmp_path / "stack.fits"
    fits.writeto(
        first,
        image,
        header=fits.Header({"PIERSIDE": "EAST"}),
        overwrite=True,
    )
    fits.writeto(
        second,
        np.ascontiguousarray(np.rot90(image, 2)),
        header=fits.Header({"PIERSIDE": "WEST"}),
        overwrite=True,
    )

    result = seiza.stack_fits(
        [first, second], output, options=no_adjustment_options()
    )

    assert result.accepted_frames == 2
    assert result.rejected_frames == 0
    assert result.frames[0].accepted
    assert abs(abs(result.frames[0].rotation_degrees) - 180.0) < 1.0
    with fits.open(output) as hdus:
        assert hdus[0].header["STACKCNT"] == 2
        assert "PIERSIDE" not in hdus[0].header
        np.testing.assert_allclose(hdus[0].data, image, rtol=0.0, atol=1.0)


def test_batch_fits_stack_preserves_three_plane_color(tmp_path):
    image = synthetic_star_field()
    rgb = np.stack(
        [image, image * 0.65 + 25.0, image * 0.35 + 50.0], axis=0
    ).astype(np.float32)
    first = tmp_path / "rgb-before-flip.fits"
    second = tmp_path / "rgb-after-flip.fits"
    output = tmp_path / "rgb-stack.fits"
    fits.writeto(first, rgb, overwrite=True)
    fits.writeto(second, np.ascontiguousarray(rgb[:, ::-1, ::-1]), overwrite=True)

    result = seiza.stack_fits(
        [first, second], output, options=no_adjustment_options()
    )

    assert result.channels == 3
    assert result.accepted_frames == 2
    assert result.frames[0].accepted
    with fits.open(output) as hdus:
        assert hdus[0].data.shape == rgb.shape
        np.testing.assert_allclose(hdus[0].data, rgb, rtol=0.0, atol=1.0)


def test_fits_live_stacker_protects_inputs_from_duplicates_and_output(tmp_path):
    image = synthetic_star_field()
    first = tmp_path / "light-001.fits"
    second = tmp_path / "light-002.fits"
    output = tmp_path / "stack.fits"
    fits.writeto(first, image, overwrite=True)
    fits.writeto(second, image, overwrite=True)

    stacker = seiza.LiveStacker(first, options=no_adjustment_options())
    assert stacker.push_fits(second).accepted
    context = tmp_path / "live.seiza-stack"
    stacker.save_context(context)
    stacker = seiza.LiveStacker.open_context(context)
    with pytest.raises(ValueError, match="already been used"):
        stacker.push_fits(second)
    with pytest.raises(ValueError, match="must not refer"):
        stacker.finish(first)
    assert stacker.accepted_frames == 2
    assert stacker.finish(output).accepted_frames == 2
    assert output.exists()


def test_pipelined_push_matches_pushing_one_at_a_time(tmp_path):
    image = synthetic_star_field()
    paths = []
    for index in range(4):
        path = tmp_path / f"light-{index:03d}.fits"
        fits.writeto(path, image, overwrite=True)
        paths.append(path)

    one_at_a_time = seiza.LiveStacker(paths[0], options=no_adjustment_options())
    for path in paths[1:]:
        one_at_a_time.push_fits(path)
    expected = one_at_a_time.snapshot()

    pipelined = seiza.LiveStacker(paths[0], options=no_adjustment_options())
    dispositions, report = pipelined.push_fits_pipelined(paths[1:], workers=3)
    actual = pipelined.snapshot()

    assert len(dispositions) == 3
    assert report.integrated == expected.accepted_frames - 1
    assert report.failed == 0
    np.testing.assert_array_equal(actual.image, expected.image)
    np.testing.assert_array_equal(actual.coverage, expected.coverage)


def test_pipelined_push_reports_a_bad_path_in_place(tmp_path):
    image = synthetic_star_field()
    first = tmp_path / "light-001.fits"
    second = tmp_path / "light-002.fits"
    broken = tmp_path / "broken.fits"
    fits.writeto(first, image, overwrite=True)
    fits.writeto(second, image, overwrite=True)
    broken.write_bytes(b"not a fits file")

    stacker = seiza.LiveStacker(first, options=no_adjustment_options())
    # A repeat of the reference, an unreadable file, and a usable frame. None
    # of them raises; the run carries on and the summary counts the trouble.
    dispositions, report = stacker.push_fits_pipelined(
        [first, broken, second], workers=2
    )

    assert [d.accepted for d in dispositions] == [False, False, True]
    assert "already been used" in dispositions[0].reason
    assert report.failed == 2
    assert report.integrated == 1


def test_build_bias_stops_when_the_cancel_predicate_says_so(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"bias-{index:03d}.fits"
        fits.writeto(path, np.full((16, 20), 100.0, dtype=np.float32), overwrite=True)
        paths.append(path)
    output = tmp_path / "master-bias.fits"
    checks = []

    def cancel():
        checks.append(len(checks))
        return len(checks) > 1

    with pytest.raises(seiza.StackError, match="cancelled"):
        seiza.build_bias(paths, output, cancel=cancel)

    assert not output.exists()


def test_build_bias_reports_an_exception_raised_by_the_cancel_predicate(tmp_path):
    paths = []
    for index in range(2):
        path = tmp_path / f"bias-{index:03d}.fits"
        fits.writeto(path, np.full((16, 20), 100.0, dtype=np.float32), overwrite=True)
        paths.append(path)
    output = tmp_path / "master-bias.fits"

    def cancel():
        raise LookupError("no job registry")

    with pytest.raises(LookupError, match="no job registry"):
        seiza.build_bias(paths, output, cancel=cancel)


def test_build_bias_writes_master_metadata_and_statistics(tmp_path):
    first = tmp_path / "bias-001.fits"
    second = tmp_path / "bias-002.fits"
    output = tmp_path / "master-bias.fits"
    fits.writeto(first, np.full((16, 20), 100.0, dtype=np.float32), overwrite=True)
    fits.writeto(second, np.full((16, 20), 102.0, dtype=np.float32), overwrite=True)

    result = seiza.build_bias([first, second], output)

    assert result.kind == "bias"
    assert result.input_frames == 2
    assert result.rejected_samples == 0
    assert result.input_statistics == [(320, 0), (320, 0)]
    with fits.open(output) as hdus:
        assert hdus[0].header["SEIZAMST"] == "BIAS"
        assert hdus[0].header["NCOMBINE"] == 2
        np.testing.assert_allclose(hdus[0].data, 101.0)


def test_compatible_calibration_names_what_a_light_could_accept():
    """The warn-don't-fail question, asked from Python.

    A stack with no masters keeps both lists empty; the describe functions
    carry the same field-and-both-readings text the Rust surface produces.
    """
    stacker = seiza.LiveStacker.from_array(
        synthetic_star_field(), options=no_adjustment_options()
    )
    light = seiza.FrameSignature(rotation_deg=101.93)
    kept, dropped = stacker.compatible_calibration(light)
    assert kept == []
    assert dropped == []

    flat = seiza.FrameSignature(rotation_deg=104.24)
    reason = seiza.describe_optics_mismatch(light, flat)
    assert "101.93" in reason
    assert "104.24" in reason
    assert "deg apart" in reason

    # The tolerance parameter is honored: wide enough, nothing to describe as
    # a rotation gap, and the match itself flips.
    wide = seiza.MatchTolerances(rotation_deg=5.0)
    assert seiza.optics_match(light, flat, wide)
