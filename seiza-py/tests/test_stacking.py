"""Native Python coverage for batch, live, and calibration stacking APIs."""

import shutil
import pathlib
import tempfile

import numpy as np
import pytest

import seiza

# The fixtures are written as FITS through astropy; where it has no wheel
# yet, such as on a newly released Python, these tests skip.
fits = pytest.importorskip("astropy.io.fits")


def synthetic_star_field(height=128, width=160, noise=1.5):
    rng = np.random.default_rng(29)
    image = rng.normal(100.0, noise, size=(height, width)).astype(np.float32)
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


def test_dark_calibrated_flat_readout_policy_and_context(tmp_path):
    def write(name, data, **cards):
        header = fits.Header()
        for key, value in cards.items():
            header[key] = value
        path = tmp_path / name
        fits.writeto(path, data, header, overwrite=True)
        return path

    light = write("light.fits", synthetic_star_field(), GAIN=100, OFFSET=50, EXPTIME=60)
    next_light = write("next.fits", synthetic_star_field(), GAIN=100, OFFSET=50, EXPTIME=60)
    dark = write("dark.fits", np.ones((128, 160), np.float32), SEIZAMST="DARK", GAIN=100, OFFSET=50, EXPTIME=60)
    flat = write("flat.fits", np.ones((128, 160), np.float32), SEIZAMST="FLAT", GAIN=0, OFFSET=10, DARKSUB=True, BIASSUB=True, FLATNORM=True)
    assert not seiza.MatchTolerances().allow_flat_gain_offset_mismatch
    with pytest.raises(seiza.StackError, match="master flat"):
        seiza.LiveStacker(light, dark=dark, flat=flat, options=no_adjustment_options())
    gate = seiza.MatchTolerances(allow_flat_gain_offset_mismatch=True)
    options = seiza.StackOptions(normalization="none", rejection="none", match_tolerances=gate)
    stacker = seiza.LiveStacker(light, dark=dark, flat=flat, options=options)
    assert stacker.push_fits(next_light).accepted
    context = tmp_path / "flat.seiza-stack"
    stacker.save_context(context)
    restored = seiza.LiveStacker.open_context(context)
    assert restored.match_tolerances().allow_flat_gain_offset_mismatch
    assert restored.reintegrate().accepted_frames == 2
    restored.set_match_tolerances()
    signature = seiza.FrameSignature(width=160, height=128, channels=1, gain=100, offset=50, exposure_seconds=60)
    with pytest.raises(seiza.StackError, match="master flat"):
        restored.validate_calibration(signature)
    bad_dark = write("bad-dark.fits", np.ones((128, 160), np.float32), SEIZAMST="DARK", GAIN=0, OFFSET=10, EXPTIME=60)
    with pytest.raises(seiza.StackError, match="master dark"):
        seiza.LiveStacker(light, dark=bad_dark, flat=flat, options=options)
    uncalibrated = write("normalized-only.fits", np.ones((128, 160), np.float32), SEIZAMST="FLAT", GAIN=0, OFFSET=10, FLATNORM=True)
    with pytest.raises(seiza.StackError, match="master flat"):
        seiza.LiveStacker(light, flat=uncalibrated, options=options)


def test_stack_options_reject_unknown_modes():
    with pytest.raises(ValueError, match="normalization"):
        seiza.StackOptions(normalization="mystery")
    with pytest.raises(ValueError, match="rejection"):
        seiza.StackOptions(rejection="mystery")
    with pytest.raises(ValueError, match="delta-sigma"):
        seiza.StackOptions(rejection_warmup=1)
    with pytest.raises(ValueError, match="demosaic"):
        seiza.StackOptions(demosaic="mystery")


def test_stack_options_choose_the_demosaic():
    assert seiza.StackOptions().demosaic == "vng"
    for method in ["vng", "mhc", "bilinear"]:
        assert seiza.StackOptions(demosaic=method).demosaic == method
    options = seiza.StackOptions(demosaic="superpixel", bin_factor=2)
    assert options.demosaic == "superpixel"
    # A superpixel bins each 2x2 cell, so it needs an even bin factor.
    with pytest.raises(ValueError, match="superpixel"):
        seiza.StackOptions(demosaic="superpixel")


def test_gpu_options_say_which_passes_may_run_on_the_gpu():
    everywhere = seiza.GpuOptions()
    assert everywhere.enabled and everywhere.debayer and everywhere.integration
    assert (everywhere.concurrency, everywhere.trace) == (3, False)
    assert seiza.StackOptions().gpu == everywhere
    narrowed = seiza.GpuOptions(detection=False, concurrency=1)
    options = seiza.StackOptions(gpu=narrowed)
    assert options.gpu == narrowed
    assert not options.gpu.detection and options.gpu.resampling
    assert "detection=False" in repr(narrowed)
    assert not seiza.GpuOptions(enabled=False).enabled
    with pytest.raises(ValueError, match="concurrency"):
        seiza.GpuOptions(concurrency=0)
    assert everywhere.memory_bytes is None
    capped = seiza.GpuOptions(memory_bytes=8 << 30)
    assert capped.memory_bytes == 8 << 30
    assert "memory_bytes=8589934592" in repr(capped)
    assert seiza.StackOptions(gpu=capped).gpu == capped
    with pytest.raises(ValueError, match="memory_bytes"):
        seiza.GpuOptions(memory_bytes=0)


def test_a_stack_on_the_cpu_keeps_its_gpu_options_in_a_context(tmp_path):
    image = synthetic_star_field()
    cpu = seiza.GpuOptions(enabled=False)
    options = seiza.StackOptions(normalization="none", rejection="none", gpu=cpu)
    stacker = seiza.LiveStacker.from_array(image, options=options)
    everywhere = seiza.LiveStacker.from_array(image, options=no_adjustment_options())
    assert stacker.push(image.copy()).accepted
    # Where the arithmetic runs is no part of the configuration.
    assert stacker.state().configuration_fingerprint == (
        everywhere.state().configuration_fingerprint
    )
    stacker.save_context(tmp_path / "cpu.seiza-stack")
    reopened = seiza.LiveStacker.open_context(tmp_path / "cpu.seiza-stack")
    assert reopened.state().configuration_fingerprint == (
        stacker.state().configuration_fingerprint
    )


def test_a_context_reopens_under_other_gpu_options(tmp_path):
    image = synthetic_star_field()
    stacker = seiza.LiveStacker.from_array(image, options=no_adjustment_options())
    assert stacker.push(image.copy()).accepted
    context = tmp_path / "everywhere.seiza-stack"
    stacker.save_context(context)
    on_the_cpu = seiza.LiveStacker.open_context(
        context, gpu=seiza.GpuOptions(enabled=False)
    )
    assert on_the_cpu.accepted_frames == stacker.accepted_frames
    assert on_the_cpu.state().configuration_fingerprint == (
        stacker.state().configuration_fingerprint
    )
    assert on_the_cpu.push(image.copy()).accepted
    assert on_the_cpu.accepted_frames == stacker.accepted_frames + 1
    # gpu=None keeps the options the stack was saved with.
    kept = seiza.LiveStacker.open_context(context, gpu=None)
    assert kept.state().configuration_fingerprint == (
        stacker.state().configuration_fingerprint
    )
    # Both stacks hold the same frames, so their means agree.
    stacker.push(image.copy())
    assert np.allclose(
        on_the_cpu.snapshot().image, stacker.snapshot().image, rtol=1e-5, equal_nan=True
    )


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


def test_channels_stack_onto_a_shared_reference_image(tmp_path):
    # A larger, brighter image sets the grid but adds none of its pixels.
    common = tmp_path / "common.fits"
    fits.writeto(common, synthetic_star_field(height=144, width=176) * 2.0)
    image = synthetic_star_field()
    lights = [tmp_path / "red-001.fits", tmp_path / "red-002.fits"]
    for light in lights:
        fits.writeto(light, image)
    options = no_adjustment_options()

    output = tmp_path / "red.fits"
    result = seiza.stack_fits(lights, output, options=options, reference_image=common)
    assert result.accepted_frames == 2
    assert len(result.frames) == 2
    assert (result.height, result.width) == (144, 176)
    with fits.open(output) as hdus:
        assert abs(hdus[0].data[70, 70] - image[70, 70]) < 1.0
    with pytest.raises(ValueError, match="reference image"):
        seiza.stack_fits(lights, common, options=options, reference_image=common)
    with pytest.raises(ValueError, match="duplicate"):
        seiza.stack_fits(
            lights, output, options=options, bias=common, reference_image=common
        )

    stacker = seiza.LiveStacker.open_fits_with_reference_image(
        common, lights[0], options=options
    )
    assert not stacker.reference_integrated
    assert stacker.accepted_frames == 0
    for light in lights:
        assert stacker.push_fits(light).accepted
    assert stacker.accepted_frames == 2
    with pytest.raises(ValueError, match="must not refer"):
        stacker.finish(common)
    final = stacker.finish(tmp_path / "live.fits")
    assert final.image.shape == (144, 176)
    assert seiza.LiveStacker.from_array(image).reference_integrated


def test_auto_reference_falls_back_to_a_light_that_aligns(tmp_path):
    common = tmp_path / "common.fits"
    fits.writeto(common, synthetic_star_field(height=144, width=176))
    # The quietest light scores best, but it is turned a quarter turn and
    # cannot align to the common image.
    turned = tmp_path / "red-turned.fits"
    fits.writeto(
        turned, np.ascontiguousarray(np.rot90(synthetic_star_field(noise=0.5)))
    )
    lights = [tmp_path / "red-001.fits", turned, tmp_path / "red-002.fits"]
    for light in lights[::2]:
        fits.writeto(light, synthetic_star_field())
    options = no_adjustment_options()

    choice = seiza.choose_reference(lights)
    assert (choice.reference_index, choice.reference_path) == (1, turned)
    assert choice.ranking[0] == 1
    assert sorted(choice.ranking) == [0, 1, 2]
    assert all(score.stars > 0 for score in choice.scores)
    with pytest.raises(ValueError, match="at least one"):
        seiza.choose_reference([])

    result = seiza.stack_fits(
        lights,
        tmp_path / "red.fits",
        options=options,
        reference_image=common,
        reference="auto",
    )
    assert result.reference_light in lights[::2]
    assert result.accepted_frames == 2
    assert len(result.frames) == 3
    assert result.frames[0].source == result.reference_light
    assert not next(frame for frame in result.frames if frame.source == turned)
    # Asked for the first light, the stack tries no other.
    with pytest.raises(seiza.StackError, match="could not be aligned"):
        seiza.stack_fits(
            [turned, lights[0]],
            tmp_path / "first.fits",
            options=options,
            reference_image=common,
        )
    with pytest.raises(ValueError, match="first"):
        seiza.stack_fits(lights, tmp_path / "unused.fits", reference="best")

    # Without a reference image, the best light becomes the stack's reference.
    result = seiza.stack_fits(
        lights[::2], tmp_path / "auto.fits", options=options, reference="auto"
    )
    assert result.reference_light in lights[::2]
    assert [frame.source for frame in result.frames] == [
        light for light in lights[::2] if light != result.reference_light
    ]

    stacker, selection = seiza.LiveStacker.open_fits_with_reference_image_candidates(
        common, [turned, lights[0]], options=options
    )
    assert (selection.index, selection.path) == (1, lights[0])
    [(index, path, reason)] = selection.skipped
    assert (index, path) == (0, turned)
    assert reason
    assert not stacker.reference_integrated
    assert stacker.accepted_frames == 0
    assert stacker.push_fits(selection.path).accepted
    with pytest.raises(ValueError, match="at least one"):
        seiza.LiveStacker.open_fits_with_reference_image_candidates(common, [])


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


def test_build_flat_masks_stars_in_sky_flats_in_a_chosen_scratch_directory(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    height, width = 80, 96
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    grain = (((yy * width + xx) * 37 + 11) % 17) * 0.002 - 0.016
    paths = []
    for frame in range(6):
        star = 8.0 * np.exp(-((xx - (20.0 + frame * 11.0)) ** 2 + (yy - 40.0) ** 2) / 8.0)
        illumination = 1000.0 + frame * 300.0
        path = tmp_path / f"sky-flat-{frame}.fits"
        header = fits.Header({"IMAGETYP": "FLAT"})
        data = (illumination * (1.0 + grain + star) + 200.0).astype(np.float32)
        fits.writeto(path, data, header)
        paths.append(path)

    masking = seiza.FlatStarMasking(detection_sigma=5.0)
    assert masking.halo_radius_factor == 3.0
    assert masking.minimum_clean_samples == 2
    result = seiza.build_flat(
        paths,
        tmp_path / "master-flat.fits",
        flat_star_masking=masking,
        scratch_directory=scratch,
    )
    assert result.rejection_method == "MEDIAN_MAD"
    assert result.masked_samples > 0
    assert sum(result.input_masked_samples) == result.masked_samples
    assert len(result.input_masked_samples) == len(result.input_statistics)
    statistics = result.flat_star_masking
    assert statistics["masked_samples"] == result.masked_samples
    assert statistics["options"].detection_sigma == 5.0
    assert (
        result.accepted_samples + result.rejected_samples + result.masked_samples
        == width * height * len(paths)
    )
    assert list(scratch.iterdir()) == []

    with pytest.raises(seiza.StackError):
        seiza.build_flat(
            paths, tmp_path / "unwritten.fits", scratch_directory=tmp_path / "missing"
        )
    plain = seiza.build_flat(paths, tmp_path / "plain-flat.fits")
    assert plain.flat_star_masking is None
    assert plain.masked_samples == 0


def test_build_dark_screens_darks_that_caught_stray_light(tmp_path):
    yy, xx = np.mgrid[:32, :32].astype(np.float32)
    noise = (((yy * 32 + xx) * 37 + 11) % 17) * 0.6 - 4.8
    falloff = 1.0 - 0.06 * ((xx - 15.5) ** 2 + (yy - 15.5) ** 2) / (2.0 * 15.5 * 15.5)
    paths = []
    for index, leak in enumerate([0.0, 0.5, 0.0, 40.0, 400.0]):
        path = tmp_path / f"dark-{index}.fits"
        fits.writeto(path, (503.0 + noise + leak * falloff).astype(np.float32))
        paths.append(path)

    result = seiza.build_dark(paths, tmp_path / "master-dark.fits", dark_level_screening=1.0)
    assert result.input_frames == 3
    assert result.rejection_method == "LEAVE_ONE_OUT"
    unscreened = seiza.build_dark(paths, tmp_path / "unscreened.fits")
    assert unscreened.input_frames == 5
    with pytest.raises(seiza.StackError, match="level screening"):
        seiza.build_dark(paths, tmp_path / "bad.fits", dark_level_screening=0.0)


def test_prepared_stack_reports_noise_weights_and_mappings(tmp_path):
    image = synthetic_star_field()
    prepared = tmp_path / "prepared.fits"
    fits.writeto(prepared, image, fits.Header({"OBJECT": "M 31"}))
    options = seiza.StackOptions(normalization="none", rejection="none", weighting="inverse-noise-variance")
    stacker = seiza.LiveStacker.open_prepared_fits(prepared, options=options)
    assert stacker.accepted_frames == 1

    reference_noise = stacker.reference_noise
    assert len(reference_noise) == 1 and reference_noise[0] > 0
    assert seiza.frame_noise(image) == pytest.approx(reference_noise)
    assert seiza.frame_noise(np.ones((2, 2), np.float32)) is None
    assert stacker.reference_mapping["schema_version"] == 1

    disposition = stacker.push(image)
    assert disposition.accepted
    assert len(disposition.noise) == 1
    assert disposition.weight == pytest.approx([1.0], abs=1e-3)
    assert disposition.mapping["schema_version"] == 1
    assert "transform" in disposition.mapping
    with pytest.raises(seiza.StackError, match="push_linear"):
        stacker.push_fits(prepared)

    raw = tmp_path / "raw.fits"
    fits.writeto(raw, image, fits.Header({"BAYERPAT": "RGGB"}))
    with pytest.raises(seiza.StackError, match="Bayer"):
        seiza.LiveStacker.open_prepared_fits(raw)


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


def test_validate_calibration_honors_the_callers_tolerances(tmp_path):
    """A flat from 3.2 degrees, chosen on purpose for a light at 113.8.

    Under the default two-degree gate the light is refused, and the refusal
    names the gate; asked with the gate the host chose the flat under, it
    passes. The yes-or-no answer agrees with compatible_calibration.
    """
    reference = tmp_path / "reference.fits"
    flat = tmp_path / "master-flat.fits"
    header = fits.Header()
    header["IMAGETYP"] = "LIGHT"
    header["ROTATANG"] = 3.2
    fits.writeto(reference, synthetic_star_field(), header, overwrite=True)
    header = fits.Header()
    header["SEIZAMST"] = "FLAT"
    header["ROTATANG"] = 3.2
    fits.writeto(flat, np.ones((128, 160), dtype=np.float32), header, overwrite=True)
    stacker = seiza.LiveStacker(reference, flat=flat, options=no_adjustment_options())

    light = seiza.FrameSignature(width=160, height=128, channels=1, rotation_deg=113.8)
    with pytest.raises(seiza.StackError, match="tolerance 2.00"):
        stacker.validate_calibration(light)
    kept, dropped = stacker.compatible_calibration(light)
    assert kept == []
    assert len(dropped) == 1

    wide = seiza.MatchTolerances(rotation_deg=180.0)
    assert stacker.validate_calibration(light, wide) is None
    kept, dropped = stacker.compatible_calibration(light, wide)
    assert kept == ["flat"]
    assert dropped == []

    # Tighter than the defaults refuses what they accept.
    near = seiza.FrameSignature(width=160, height=128, channels=1, rotation_deg=4.7)
    assert stacker.validate_calibration(near) is None
    with pytest.raises(seiza.StackError, match="tolerance 1.00"):
        stacker.validate_calibration(near, seiza.MatchTolerances(rotation_deg=1.0))

    # A tolerance no comparison can use is refused, not read as "anything".
    for unusable in (float("nan"), float("inf"), -1.0, 360.0):
        tolerances = seiza.MatchTolerances(rotation_deg=unusable)
        with pytest.raises(seiza.StackError, match="rotation_deg"):
            stacker.validate_calibration(light, tolerances)
        with pytest.raises(seiza.StackError, match="rotation_deg"):
            stacker.compatible_calibration(light, tolerances)


def test_a_stack_admits_lights_under_its_own_gate(tmp_path):
    """The stack's gate: set when it opens or later, and kept per frame."""
    flat = tmp_path / "master-flat.fits"
    header = fits.Header()
    header["SEIZAMST"] = "FLAT"
    header["ROTATANG"] = 3.2
    fits.writeto(flat, np.ones((128, 160), dtype=np.float32), header, overwrite=True)

    def light(name, rotation_deg):
        path = tmp_path / name
        header = fits.Header()
        header["IMAGETYP"] = "LIGHT"
        header["ROTATANG"] = rotation_deg
        fits.writeto(path, synthetic_star_field(), header, overwrite=True)
        return path

    reference = light("reference.fits", 3.2)
    wide = seiza.MatchTolerances(rotation_deg=180.0)
    stacker = seiza.LiveStacker(reference, flat=flat, options=no_adjustment_options())
    assert stacker.match_tolerances().rotation_deg == 2.0
    assert not stacker.push_fits(light("turned-1.fits", 113.8)).accepted

    stacker.set_match_tolerances(wide)
    assert stacker.match_tolerances().rotation_deg == 180.0
    signature = seiza.FrameSignature(width=160, height=128, channels=1, rotation_deg=113.8)
    assert stacker.validate_calibration(signature) is None
    assert stacker.push_fits(light("turned-2.fits", 113.8)).accepted
    stacker.set_match_tolerances()
    assert stacker.match_tolerances().rotation_deg == 2.0
    assert not stacker.push_fits(light("turned-3.fits", 113.8)).accepted
    with pytest.raises(seiza.StackError, match="finite"):
        stacker.set_match_tolerances(seiza.MatchTolerances(rotation_deg=float("inf")))

    # A reference is judged under the gate in the options it opens with.
    turned = light("turned-reference.fits", 113.8)
    with pytest.raises(seiza.StackError, match="tolerance 2.00"):
        seiza.LiveStacker(turned, flat=flat, options=no_adjustment_options())
    options = seiza.StackOptions(normalization="none", rejection="none", match_tolerances=wide)
    assert options.match_tolerances.rotation_deg == 180.0
    gated = seiza.LiveStacker(turned, flat=flat, options=options)
    assert gated.match_tolerances().rotation_deg == 180.0


def test_reintegrate_and_drizzle_reject_a_reference_trail(tmp_path):
    clean = synthetic_star_field()
    paths = []
    for index in range(6):
        frame = clean + np.float32(index % 3) * 0.2
        if index == 0:
            frame = frame.copy()
            frame[60, 10:150] += 5000.0
        path = tmp_path / f"light-{index:03d}.fits"
        fits.writeto(path, frame, overwrite=True)
        paths.append(path)
    stacker = seiza.LiveStacker(paths[0], options=no_adjustment_options())
    for path in paths[1:]:
        assert stacker.push_fits(path).accepted

    assert stacker.snapshot().image[60, 80] - clean[60, 80] > 500.0
    replayed = stacker.reintegrate(scratch_directory=tmp_path)
    assert abs(replayed.image[60, 80] - clean[60, 80]) < 2.0

    integrated, drizzled = stacker.reintegrate_drizzled(scale=2)
    assert drizzled.scale == 2
    assert drizzled.image.shape == (2 * clean.shape[0], 2 * clean.shape[1])
    assert drizzled.weight.shape == drizzled.image.shape
    assert drizzled.accepted_frames == 6
    # Reference row 60 covers drizzled rows 120 and 121.
    assert np.all(np.abs(drizzled.image[120:122, 160] - clean[60, 80]) < 50.0)
    np.testing.assert_array_equal(integrated.image, replayed.image)

    output = tmp_path / "drizzled.fits"
    drizzled.write_fits(output)
    with fits.open(output) as hdus:
        assert hdus[0].data.shape == drizzled.image.shape
        assert hdus[0].header["DRIZSCL"] == 2
    with pytest.raises(ValueError, match="must not refer"):
        drizzled.write_fits(paths[0])
    with pytest.raises(ValueError, match="scale"):
        stacker.reintegrate_drizzled(scale=5)


@pytest.mark.parametrize("bin_factor", [1, 2])
def test_reintegration_rejects_unusable_explicit_scratch(tmp_path, bin_factor):
    path = tmp_path / "light.fits"
    fits.writeto(path, synthetic_star_field())
    stacker = seiza.LiveStacker(
        path, options=seiza.StackOptions(bin_factor=bin_factor, normalization="none")
    )
    for scratch in [tmp_path / "missing", path]:
        with pytest.raises(seiza.StackError, match="scratch directory"):
            stacker.reintegrate(scratch_directory=scratch)
        with pytest.raises(seiza.StackError, match="scratch directory"):
            stacker.reintegrate_full_resolution(scratch_directory=scratch)
    assert stacker.accepted_frames == 1
    assert not (tmp_path / "missing").exists()


def test_retained_frames_reintegrate_identically(tmp_path):
    clean = synthetic_star_field()
    paths = []
    for index in range(6):
        frame = clean + np.float32(index % 3) * 0.2
        if index == 0:
            frame = frame.copy()
            frame[60, 10:150] += 5000.0
        path = tmp_path / f"light-{index:03d}.fits"
        fits.writeto(path, frame, overwrite=True)
        paths.append(path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    def stack(retain):
        stacker = seiza.LiveStacker(paths[0], options=no_adjustment_options())
        if retain:
            stacker.retain_frames_for_reintegration(scratch_directory=scratch)
        for path in paths[1:]:
            assert stacker.push_fits(path).accepted
        return stacker

    retained = stack(True)
    assert len(list(scratch.iterdir())) == 1
    fresh = stack(False)
    np.testing.assert_array_equal(
        retained.reintegrate().image, fresh.reintegrate().image
    )
    _, retained_drizzle = retained.reintegrate_drizzled(scale=2)
    _, fresh_drizzle = fresh.reintegrate_drizzled(scale=2)
    np.testing.assert_array_equal(retained_drizzle.image, fresh_drizzle.image)
    np.testing.assert_array_equal(retained_drizzle.weight, fresh_drizzle.weight)
    retained.finish()
    assert list(scratch.iterdir()) == []


def overlapping_panels(tmp_path, width=192, height=160, step=112):
    """Three panels across one star field, each overlapping the next by
    80 pixels; the first and third share no sky."""
    rng = np.random.default_rng(41)
    sky_width = width + 2 * step
    sky = rng.normal(100.0, 1.5, size=(height, sky_width)).astype(np.float32)
    yy, xx = np.mgrid[:height, :sky_width]
    for _ in range(140):
        x, y = rng.uniform(8, sky_width - 8), rng.uniform(8, height - 8)
        sky += rng.uniform(400.0, 4000.0) * np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / 3.2
        ).astype(np.float32)
    paths = []
    for panel in range(3):
        path = tmp_path / f"panel-{panel}.fits"
        fits.writeto(path, sky[:, panel * step : panel * step + width])
        paths.append(path)
    return paths


def test_stack_options_choose_a_registration_model():
    assert seiza.StackOptions().registration_model == "similarity"
    options = seiza.StackOptions(registration_model="quadratic")
    assert options.registration_model == "quadratic"
    with pytest.raises(ValueError, match="registration_model"):
        seiza.StackOptions(registration_model="thin-plate")


@pytest.mark.parametrize("normalization", ["none", "global", "local-background"])
def test_shared_mosaic_canvas_keeps_channel_photometry(tmp_path, normalization):
    paths = overlapping_panels(tmp_path)
    options = seiza.StackOptions(
        normalization=normalization, rejection="none", minimum_overlap=0.1,
        gpu=seiza.GpuOptions(enabled=False),
    )
    source = seiza.LiveStacker.open_fits_expanding(paths, options=options)
    channels = []
    for index, path in enumerate(paths):
        other = tmp_path / f"oiii-{index}.fits"
        fits.writeto(other, fits.getdata(path) * 0.25)
        channels.append(other)
    # No integration is needed to reuse an in-memory plan. Start on the far
    # panel and defer the disconnected near panel until the middle joins.
    first = source.open_fits_on_canvas([paths[2], paths[0], paths[1]], options=options)
    second = source.open_fits_on_canvas(
        [channels[2], channels[0], channels[1]], options=options
    )
    assert source.accepted_frames == first.accepted_frames == second.accepted_frames == 0
    assert not second.reference_integrated
    for path in paths:
        assert first.push_fits(path).accepted
    for path in channels:
        assert second.push_fits(path).accepted
    a, b = first.snapshot(), second.snapshot()
    assert (a.width, a.height) == (b.width, b.height)
    valid = np.isfinite(a.image) & np.isfinite(b.image)
    assert valid.mean() > 0.9
    np.testing.assert_allclose(b.image[valid], a.image[valid] * 0.25, rtol=0.005, atol=0.05)
    partial = source.open_fits_on_canvas([channels[2], channels[1]], options=options)
    for path in channels[1:]:
        assert partial.push_fits(path).accepted
    sparse = partial.snapshot()
    assert sparse.image.shape == a.image.shape
    assert 0.5 < np.isfinite(sparse.image).mean() < 0.9
    common = np.isfinite(sparse.image) & np.isfinite(b.image)
    # Fewer exposures change the pixel averages, but star positions stay on
    # the same grid throughout the shared coverage.
    assert np.corrcoef(sparse.image[common], b.image[common])[0, 1] > 0.995
    context = tmp_path / "oiii.seiza-stack"
    second.save_context(context)
    restored = seiza.LiveStacker.open_context(context)
    third = restored.open_fits_on_canvas([paths[2], paths[0], paths[1]], options=options)
    for path in paths:
        assert third.push_fits(path).accepted
    c = third.snapshot()
    assert (c.width, c.height) == (a.width, a.height)
    np.testing.assert_allclose(c.image, a.image, rtol=0.005, atol=0.05, equal_nan=True)


def test_shared_mosaic_canvas_validates_source_and_cancel(tmp_path):
    paths = overlapping_panels(tmp_path)
    options = seiza.StackOptions(normalization="none", gpu=seiza.GpuOptions(enabled=False))
    ordinary = seiza.LiveStacker(paths[0], options=options)
    with pytest.raises(seiza.StackError, match="expanded stack"):
        ordinary.open_fits_on_canvas(paths, options=options)
    canvas = seiza.LiveStacker.open_fits_expanding(paths, options=options)
    with pytest.raises(ValueError, match="at least one light"):
        canvas.open_fits_on_canvas([], options=options)
    with pytest.raises(seiza.StackError, match="bin factor"):
        canvas.open_fits_on_canvas(paths, options=seiza.StackOptions(bin_factor=2))
    with pytest.raises(seiza.StackError, match="cancel"):
        canvas.open_fits_on_canvas(paths, options=options, cancel=lambda: True)
    def interrupt():
        raise KeyboardInterrupt("shared canvas interrupted")
    with pytest.raises(KeyboardInterrupt, match="shared canvas interrupted"):
        canvas.open_fits_on_canvas(paths, options=options, cancel=interrupt)
    assert canvas.accepted_frames == 0


def test_shared_mosaic_canvas_uses_each_filters_calibration(tmp_path):
    paths = overlapping_panels(tmp_path)
    channels = []
    for index, path in enumerate(paths):
        data = fits.getdata(path, memmap=False)
        fits.writeto(path, data, fits.Header({"FILTER": "Ha"}), overwrite=True)
        other = tmp_path / f"oiii-{index}.fits"
        fits.writeto(other, data * 0.25, fits.Header({"FILTER": "OIII"}))
        channels.append(other)
    flats = []
    for name in ("Ha", "OIII"):
        flat = tmp_path / f"flat-{name}.fits"
        fits.writeto(flat, np.ones((160, 192), dtype=np.float32),
                     fits.Header({"SEIZAMST": "FLAT", "FILTER": name, "DARKSUB": True}))
        flats.append(flat)
    options = seiza.StackOptions(normalization="none", gpu=seiza.GpuOptions(enabled=False))
    source = seiza.LiveStacker.open_fits_expanding(paths, options=options, flat=flats[0])
    with pytest.raises(seiza.StackError, match="filter"):
        source.open_fits_on_canvas(channels, options=options, flat=flats[0])
    channel = source.open_fits_on_canvas(channels, options=options, flat=flats[1])
    for path in channels:
        assert channel.push_fits(path).accepted
    assert channel.accepted_frames == 3


@pytest.mark.parametrize("normalization", ["none", "global", "local-background"])
def test_joint_canvas_preserves_each_filters_unique_coverage(tmp_path, normalization):
    paths = overlapping_panels(tmp_path)
    other = []
    for i, path in enumerate(paths[1:]):
        out = tmp_path / f"oiii-{i}.fits"
        fits.writeto(out, fits.getdata(path) * 0.01, fits.Header({"FILTER": "OIII"}))
        other.append(out)
    flat = tmp_path / "oiii-flat.fits"
    fits.writeto(flat, np.ones((160, 192), np.float32),
                 fits.Header({"SEIZAMST": "FLAT", "FILTER": "OIII", "DARKSUB": True}))
    groups = [seiza.CanvasChannel(paths[:2]), seiza.CanvasChannel(other, flat=flat)]
    assert groups[1].paths == other and groups[1].flat == flat
    assert groups[0].bias is groups[0].dark is groups[0].dark_exposure_seconds is None
    options = seiza.StackOptions(normalization=normalization, rejection="none",
                                minimum_overlap=0.1, gpu=seiza.GpuOptions(enabled=False))
    first = seiza.LiveStacker.open_fits_joint_canvas(groups, options=options)
    assert first.accepted_frames == 0
    for path in paths[:2]:
        assert first.push_fits(path).accepted
    context = tmp_path / "union.seiza-stack"
    first.save_context(context)
    restored = seiza.LiveStacker.open_context(context)
    second = restored.open_fits_on_canvas(other, flat=flat, options=options)
    for path in other:
        assert second.push_fits(path).accepted
    a, b = first.snapshot().image, second.snapshot().image
    assert a.shape == b.shape and a.shape[1] >= 416
    va, vb = np.isfinite(a), np.isfinite(b)
    assert (va & ~vb).mean() > 0.2 and (vb & ~va).mean() > 0.2
    common = va & vb
    assert np.corrcoef(a[common], b[common])[0, 1] > 0.995
    assert 0.009 < b[common].mean() / a[common].mean() < 0.011
    # The selected channel can be any group, with its own flat and brightness.
    selected = seiza.LiveStacker.open_fits_joint_canvas(groups, 1, options=options)
    for path in other:
        assert selected.push_fits(path).accepted
    np.testing.assert_allclose(selected.snapshot().image, b, rtol=0.005, atol=0.005, equal_nan=True)


def test_joint_canvas_validation_and_cancellation(tmp_path):
    paths = overlapping_panels(tmp_path)
    options = seiza.StackOptions(normalization="none", gpu=seiza.GpuOptions(enabled=False))
    groups = [seiza.CanvasChannel(paths[:2]), seiza.CanvasChannel(paths[2:])]
    for invalid, message in [([], "index"), ([seiza.CanvasChannel([])], "light"),
                             ([groups[0], groups[0]], "unique"),
                             ([seiza.CanvasChannel(paths, dark_exposure_seconds=10)], "dark")]:
        with pytest.raises(seiza.StackError, match=message):
            seiza.LiveStacker.open_fits_joint_canvas(invalid, options=options)
    with pytest.raises(seiza.StackError, match="index"):
        seiza.LiveStacker.open_fits_joint_canvas(groups, 2, options=options)
    with pytest.raises(ValueError, match="positive"):
        seiza.LiveStacker.open_fits_joint_canvas(groups, max_canvas_megasamples=0)
    with pytest.raises(seiza.StackError, match="cancel"):
        seiza.LiveStacker.open_fits_joint_canvas(groups, options=options, cancel=lambda: True)
    def interrupt():
        raise KeyboardInterrupt("joint interrupted")
    with pytest.raises(KeyboardInterrupt, match="joint interrupted"):
        seiza.LiveStacker.open_fits_joint_canvas(groups, options=options, cancel=interrupt)
    flat = tmp_path / "wrong-flat.fits"
    fits.writeto(flat, np.ones((160, 192), np.float32),
                 fits.Header({"SEIZAMST": "FLAT", "FILTER": "wrong", "DARKSUB": True}))
    for path in paths:
        data = fits.getdata(path, memmap=False)
        fits.writeto(path, data, fits.Header({"FILTER": "Ha"}), overwrite=True)
    with pytest.raises(seiza.StackError, match="filter"):
        seiza.LiveStacker.open_fits_joint_canvas(
            [groups[0], seiza.CanvasChannel(paths[2:], flat=flat)], options=options)


def test_joint_canvas_keeps_omitted_group_notes(tmp_path):
    paths = overlapping_panels(tmp_path)
    blank = tmp_path / "blank.fits"
    fits.writeto(blank, np.ones((160, 192), np.float32))
    options = seiza.StackOptions(normalization="none", gpu=seiza.GpuOptions(enabled=False))
    stacker = seiza.LiveStacker.open_fits_joint_canvas(
        [seiza.CanvasChannel(paths[:2]), seiza.CanvasChannel([blank])], options=options)
    assert stacker.expansion_planning_notes
    channel = stacker.open_fits_on_canvas(paths[:2], options=options)
    assert channel.expansion_planning_notes == stacker.expansion_planning_notes


def test_expanding_stack_holds_every_panel(tmp_path):
    paths = overlapping_panels(tmp_path)
    options = seiza.StackOptions(
        normalization="none",
        rejection="none",
        minimum_overlap=0.1,
        registration_model="quadratic",
    )
    # Anchored on the last panel, the first joins only through the middle.
    order = [paths[2], paths[0], paths[1]]
    stacker = seiza.LiveStacker.open_fits_expanding(order, options=options)
    assert not stacker.reference_integrated
    for path in order:
        assert stacker.push_fits(path).accepted
    snapshot = stacker.snapshot()
    assert snapshot.accepted_frames == 3
    assert 416 <= snapshot.width <= 420
    assert 160 <= snapshot.height <= 164
    assert np.isfinite(snapshot.image).mean() > 0.95
    # Undistorted panels leave the lens nothing to explain.
    assert stacker.expanded_lens is None
    assert isinstance(stacker.expansion_planning_notes, list)
    ordinary = seiza.LiveStacker(paths[0], options=options)
    assert ordinary.expanded_lens is None
    assert ordinary.expansion_planning_notes == []
    # A frame costs its own footprint, not the canvas, which is more than
    # twice a panel's width.
    memory = stacker.pipeline_memory
    frame = ordinary.pipeline_memory
    assert set(memory) == {"worker_bytes", "integration_bytes", "sequential_bytes"}
    assert frame["integration_bytes"] == 4 * 192 * 160
    assert frame["worker_bytes"] < memory["worker_bytes"]
    assert memory["integration_bytes"] < 4 * snapshot.width * snapshot.height


def test_expanding_stack_notes_lights_that_did_not_join(tmp_path):
    paths = overlapping_panels(tmp_path)
    again = tmp_path / "panel-1-again.fits"
    shutil.copy(paths[1], again)
    # Neighbours share 80 of their 192 columns, under half.
    options = seiza.StackOptions(
        normalization="none", rejection="none", minimum_overlap=0.5
    )
    order = [paths[1], paths[0], again, paths[2]]
    stacker = seiza.LiveStacker.open_fits_expanding(order, options=options)
    note = stacker.expansion_planning_notes[0]
    assert note.startswith(
        "2 of 4 lights did not join the canvas, which holds 1 panel,"
    ), note
    assert "they matched a panel but overlapped the canvas by at most 0.4" in note
    assert "below the minimum overlap of 0.5" in note
    assert not stacker.push_fits(paths[0]).accepted


def test_expanding_stack_defaults_to_the_expanded_minimum_overlap(tmp_path):
    # Neighbours share 0.42 of their footprints: under the 0.60 a fixed
    # reference needs, over the expanded canvas's 0.05.
    paths = overlapping_panels(tmp_path)
    order = [paths[1], paths[0], paths[2]]
    for options in (None, seiza.StackOptions(normalization="none")):
        stacker = seiza.LiveStacker.open_fits_expanding(order, options=options)
        assert all(stacker.push_fits(path).accepted for path in order)
    # Options that set the fixed reference's minimum keep it.
    with pytest.raises(seiza.StackError, match="panel overlap 0.4"):
        seiza.LiveStacker.open_fits_expanding(
            order, options=seiza.StackOptions(minimum_overlap=0.6)
        )


def lens_panels(tmp_path, width=600, height=400, step=300):
    """Three panels stacked down one star field, a quarter of each
    overlapping the next, through a radial lens about an axis 30 pixels off
    the sensor's centre. Returns the paths and the axis."""
    rng = np.random.default_rng(7)
    axis = ((width - 1) / 2 + 24.0, (height - 1) / 2 - 18.0)
    radius = 0.5 * np.hypot(width, height)
    sky_height = 2 * step + height
    stars = np.column_stack(
        [
            rng.uniform(-10, width + 10, 4000),
            rng.uniform(-10, sky_height + 10, 4000),
            0.02 + 0.0005 / (0.01 + rng.uniform(size=4000)) ** 1.5,
        ]
    )
    yy, xx = np.mgrid[:height, :width]
    paths = []
    for panel in range(3):
        u, v = stars[:, 0] - axis[0], stars[:, 1] - panel * step - axis[1]
        q = (u * u + v * v) / radius**2
        factor = 1.0 + 0.02 * q + 0.004 * q * q
        x, y = axis[0] + u * factor, axis[1] + v * factor
        image = np.full((height, width), 0.01, np.float32)
        for sx, sy, peak in zip(x, y, stars[:, 2]):
            if -5 < sx < width + 5 and -5 < sy < height + 5:
                x0, y0 = int(sx) - 4, int(sy) - 4
                patch = (
                    slice(max(y0, 0), max(y0 + 9, 0)),
                    slice(max(x0, 0), max(x0 + 9, 0)),
                )
                d2 = (xx[patch] - sx) ** 2 + (yy[patch] - sy) ** 2
                image[patch] += (peak * np.exp(-d2 / 2.9)).astype(np.float32)
        image += rng.normal(0.0, 0.0005, image.shape).astype(np.float32)
        path = tmp_path / f"lens-{panel}.fits"
        fits.writeto(path, image)
        paths.append(path)
    return paths, axis


@pytest.mark.parametrize("model", ["similarity", "quadratic"])
@pytest.mark.parametrize("bin_factor", [1, 2])
def test_expanding_stack_measures_the_lens_across_panels(tmp_path, model, bin_factor):
    paths, axis = lens_panels(tmp_path)
    options = seiza.StackOptions(
        normalization="none", minimum_overlap=0.1, registration_model=model,
        bin_factor=bin_factor,
    )
    stacker = seiza.LiveStacker.open_fits_expanding(paths, options=options)
    lens = stacker.expanded_lens
    assert lens is not None
    assert np.hypot(lens["center_x"] - axis[0], lens["center_y"] - axis[1]) < 8.0
    assert lens["overlap_rms_pixels"] < 0.2 < lens["undistorted_overlap_rms_pixels"]
    assert lens["focal_length_pixels"] is None
    assert stacker.push_fits(paths[0]).accepted
    context = tmp_path / "projected-mosaic.seiza-stack"
    stacker.save_context(context)
    resumed = seiza.LiveStacker.open_context(context)
    # Planning residual diagnostics are transient; the geometric model persists.
    for key in ("center_x", "center_y", "k1", "k2", "focal_length_pixels"):
        assert resumed.expanded_lens[key] == lens[key]
    for path in paths[1:]:
        planned = stacker.push_fits(path)
        continued = resumed.push_fits(path)
        assert planned.accepted, planned.reason
        assert continued.accepted, continued.reason
        assert continued.registration_rms_pixels < 0.3


def test_shared_mosaic_canvas_preserves_binned_lens_geometry(tmp_path):
    paths, _ = lens_panels(tmp_path)
    options = seiza.StackOptions(
        normalization="none", minimum_overlap=0.1, registration_model="quadratic",
        bin_factor=2, gpu=seiza.GpuOptions(enabled=False),
    )
    source = seiza.LiveStacker.open_fits_expanding(paths, options=options)
    assert source.expanded_lens is not None
    channel = source.open_fits_on_canvas(paths, options=options)
    assert channel.expanded_lens == source.expanded_lens
    for path in paths:
        result = channel.push_fits(path)
        assert result.accepted, result.reason
        assert result.registration_rms_pixels < 0.3
        assert source.push_fits(path).accepted
    assert channel.snapshot().image.shape == source.snapshot().image.shape


def test_expanding_stack_refuses_bad_inputs_and_honours_cancel(tmp_path):
    paths = overlapping_panels(tmp_path)
    options = seiza.StackOptions(minimum_overlap=0.1)
    with pytest.raises(ValueError, match="at least one light"):
        seiza.LiveStacker.open_fits_expanding([], options=options)
    with pytest.raises(ValueError, match="max_canvas_megasamples"):
        seiza.LiveStacker.open_fits_expanding(
            paths, options=options, max_canvas_megasamples=0
        )
    with pytest.raises(ValueError, match="duplicate"):
        seiza.LiveStacker.open_fits_expanding(paths, options=options, flat=paths[1])
    with pytest.raises(seiza.StackError, match="could not join"):
        seiza.LiveStacker.open_fits_expanding([paths[0], paths[2]], options=options)
    with pytest.raises(seiza.StackError, match="cancel"):
        seiza.LiveStacker.open_fits_expanding(
            paths, options=options, cancel=lambda: True
        )

    def interrupt():
        raise KeyError("stop planning")

    with pytest.raises(KeyError, match="stop planning"):
        seiza.LiveStacker.open_fits_expanding(paths, options=options, cancel=interrupt)


def lights_with_a_header(tmp_path, count=3):
    image = synthetic_star_field()
    header = fits.Header()
    header["IMAGETYP"] = "LIGHT"
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = 83.8
    header["CRVAL2"] = -5.4
    header["CRPIX1"] = 80.5
    header["CRPIX2"] = 64.5
    header["CDELT1"] = -0.0003
    header["CDELT2"] = 0.0003
    paths = []
    for index in range(count):
        path = tmp_path / f"light-{index:03d}.fits"
        fits.writeto(path, image, header, overwrite=True)
        paths.append(path)
    return image, paths


def test_snapshots_write_with_the_reference_headers(tmp_path):
    image, paths = lights_with_a_header(tmp_path)
    stacker = seiza.LiveStacker(paths[0], options=no_adjustment_options())
    assert stacker.push_fits(paths[1]).accepted

    snapshot_output = tmp_path / "snapshot.fits"
    stacker.snapshot().write_fits(snapshot_output)
    running_output = tmp_path / "running.fits"
    stacker.write_fits(running_output)
    reintegrated_output = tmp_path / "reintegrated.fits"
    stacker.reintegrate().write_fits(reintegrated_output)

    for output in (snapshot_output, running_output, reintegrated_output):
        with fits.open(output) as hdus:
            assert hdus[0].header["CRVAL1"] == 83.8
            assert hdus[0].data.shape == image.shape
            np.testing.assert_allclose(hdus[0].data, image, rtol=0.0, atol=1.0e-2)

    # Writing the running stack leaves it open for more frames.
    assert stacker.push_fits(paths[2]).accepted
    assert stacker.accepted_frames == 3

    for write in (stacker.snapshot().write_fits, stacker.write_fits):
        with pytest.raises(ValueError, match="must not refer"):
            write(paths[1])
    finished = stacker.finish()
    with pytest.raises(ValueError, match="must not refer"):
        finished.write_fits(paths[0])
    finished_output = tmp_path / "finished.fits"
    finished.write_fits(finished_output)
    with fits.open(finished_output) as hdus:
        assert hdus[0].header["CRVAL1"] == 83.8


def test_binned_stack_reintegrates_at_full_resolution(tmp_path):
    image, paths = lights_with_a_header(tmp_path)
    assert seiza.StackOptions().bin_factor == 1
    with pytest.raises(ValueError, match="bin factor"):
        seiza.StackOptions(bin_factor=5)
    options = seiza.StackOptions(normalization="none", rejection="none", bin_factor=2)
    assert options.bin_factor == 2
    stacker = seiza.LiveStacker(paths[0], options=options)
    assert stacker.bin_factor == 2
    assert stacker.reference_mapping["bin_factor"] == 2
    for path in paths[1:]:
        disposition = stacker.push_fits(path)
        assert disposition.accepted
        assert disposition.mapping["bin_factor"] == 2

    # The live stack is on the binned grid, its WCS with it.
    binned_output = tmp_path / "binned.fits"
    stacker.snapshot().write_fits(binned_output)
    with fits.open(binned_output) as hdus:
        assert hdus[0].data.shape == (64, 80)
        assert hdus[0].header["CRPIX1"] == 40.5
        assert hdus[0].header["CDELT1"] == pytest.approx(-0.0006)
        assert hdus[0].header["SEIZABIN"] == 2

    full, refined, rms = stacker.reintegrate_full_resolution()
    assert full.image.shape == image.shape
    assert refined == 2
    assert rms is not None and 0.0 <= rms < 1.0
    np.testing.assert_allclose(full.image, image, rtol=0.0, atol=1.0e-2)
    full_output = tmp_path / "full.fits"
    full.write_fits(full_output)
    with fits.open(full_output) as hdus:
        assert hdus[0].header["CRPIX1"] == pytest.approx(80.5)
        assert hdus[0].header["CDELT1"] == pytest.approx(-0.0003)
        assert "SEIZABIN" not in hdus[0].header

    unrefined, refined, _ = stacker.reintegrate_full_resolution(refine=False)
    assert refined == 0
    assert unrefined.image.shape == image.shape
    # A drizzle by the bin factor drops each frame's sensor pixels back onto
    # the sensor's grid.
    _, drizzled = stacker.reintegrate_drizzled(scale=2)
    assert drizzled.image.shape == image.shape
    # A depth reading says what the stack was binned by.
    depth = stacker.measure_depth()
    if depth is not None:
        assert depth.bin_factor == 2


def test_live_stacker_state_describes_the_stack(tmp_path):
    _, paths = lights_with_a_header(tmp_path, count=2)
    stacker = seiza.LiveStacker(paths[0], options=no_adjustment_options())
    assert stacker.push_fits(paths[1]).accepted

    state = stacker.state()
    assert len(state.configuration_fingerprint) == 64
    assert (state.width, state.height, state.channels) == (160, 128, 1)
    assert (state.accepted_frames, state.rejected_frames) == (2, 0)
    assert state.input_mode == "calibrate-and-prepare"
    # Rust's canonical paths can carry a Windows extended-length prefix.
    assert len(state.input_paths) == len(paths)
    assert all(
        actual.samefile(expected) for actual, expected in zip(state.input_paths, paths)
    )
    assert state.reference_integrated
    assert state.reference_image_path is None
    assert state.reintegration_unavailable is None
    assert state.reference_frame.role == "light"
    assert not state.reference_frame.is_master
    assert state.reference_frame.signature.width == 160
    assert not state.reference_frame.calibration_state.bias_subtracted

    from_array = seiza.LiveStacker.from_array(
        synthetic_star_field(), options=no_adjustment_options()
    ).state()
    assert from_array.input_mode == "prepared-only"
    assert from_array.input_paths == []
    assert from_array.reference_frame.role == "unknown"


def test_masters_can_be_replaced_part_way_through(tmp_path):
    _, paths = lights_with_a_header(tmp_path, count=2)
    biases = []
    for index in range(2):
        path = tmp_path / f"bias-{index:03d}.fits"
        fits.writeto(path, np.full((128, 160), 10.0, dtype=np.float32), overwrite=True)
        biases.append(path)
    bias = tmp_path / "master-bias.fits"
    seiza.build_bias(biases, bias)

    stacker = seiza.LiveStacker(paths[0], options=no_adjustment_options())
    before = stacker.state().configuration_fingerprint
    stacker.set_calibration_fits(bias=bias)
    after = stacker.state()
    assert after.configuration_fingerprint != before
    assert any(path.samefile(bias) for path in after.input_paths)
    assert stacker.push_fits(paths[1]).accepted

    stacker.set_calibration_fits()
    assert stacker.state().configuration_fingerprint == before

    with pytest.raises(ValueError, match="dark_exposure_seconds requires dark"):
        stacker.set_calibration_fits(dark_exposure_seconds=30.0)
    with pytest.raises(seiza.StackError):
        stacker.set_calibration_fits(bias=tmp_path / "missing.fits")
    from_array = seiza.LiveStacker.from_array(
        synthetic_star_field(), options=no_adjustment_options()
    )
    with pytest.raises(seiza.StackError):
        from_array.set_calibration_fits(bias=bias)


def test_flat_defect_suppression_replaces_a_hot_pixel(tmp_path):
    width, height = 16, 16
    rng = np.random.default_rng(7)
    paths = []
    for index in range(2):
        flat = (200.0 + rng.random((height, width))).astype(np.float32)
        flat[7, 9] = 5000.0
        path = tmp_path / f"flat-{index}.fits"
        fits.writeto(path, flat, overwrite=True)
        paths.append(path)

    plain = seiza.build_flat(paths, tmp_path / "plain.fits")
    assert plain.defect_pixels_replaced == 0
    assert plain.skipped_inputs == []

    suppressed = seiza.build_flat(
        paths, tmp_path / "suppressed.fits", defect_suppression=True
    )
    assert suppressed.defect_pixels_replaced >= 1
    with fits.open(tmp_path / "suppressed.fits") as hdus:
        assert abs(hdus[0].data[7, 9] - 1.0) < 0.1

    with pytest.raises(ValueError, match="defect_suppression=True"):
        seiza.build_flat(paths, tmp_path / "unused.fits", defect_sigma_low=5.0)
    with pytest.raises(ValueError, match="positive finite"):
        seiza.build_flat(
            paths,
            tmp_path / "unused.fits",
            defect_suppression=True,
            defect_sigma_high=-1.0,
        )


def test_a_master_reports_the_inputs_it_set_aside(tmp_path):
    paths = []
    for name, rotation in [("first", 10.0), ("stray", 90.0), ("third", 10.0)]:
        header = fits.Header()
        header["IMAGETYP"] = "FLAT"
        header["FILTER"] = "R"
        header["ROTATANG"] = rotation
        path = tmp_path / f"flat-{name}.fits"
        fits.writeto(path, np.full((8, 8), 100.0, dtype=np.float32), header, overwrite=True)
        paths.append(path)

    result = seiza.build_flat(paths, tmp_path / "master-flat.fits")

    assert result.input_frames == 2
    assert len(result.input_statistics) == 2
    assert [path for path, _ in result.skipped_inputs] == [paths[1]]
    assert result.input_paths == [paths[0], paths[2]]
    assert "optical" in result.skipped_inputs[0][1]


def test_registered_frames_integrate_with_leave_one_out_rejection(tmp_path):
    rng = np.random.default_rng(7)
    paths = []
    for index in range(5):
        frame = (10.0 + rng.uniform(-0.5, 0.5, (16, 16))).astype(np.float32)
        if index == 2:
            frame[2, 8] = 1000.0
            frame[0, 0] = np.nan
        path = tmp_path / f"registered-{index}.fits"
        fits.writeto(path, frame)
        paths.append(path)

    stack, frames = seiza.integrate_registered_frames(paths)
    assert stack.accepted_frames == 5
    assert stack.image[2, 8] < 11.0
    assert [frame.path for frame in frames] == paths
    assert frames[2].finite_samples == 16 * 16 - 1
    assert frames[2].integrated_samples < frames[2].finite_samples
    assert frames[0].finite_samples == 16 * 16

    output = tmp_path / "stack.fits"
    stack.write_fits(output)
    assert output.exists()
    with pytest.raises(ValueError, match="stack input"):
        stack.write_fits(paths[0])

    weighted, _ = seiza.integrate_registered_frames(
        paths, frame_weights=[[1.0]] * 5, minimum_sigma=1e-3, sigma_low=4.0, sigma_high=4.0
    )
    assert weighted.image.shape == (16, 16)
    small = tmp_path / "small.fits"
    fits.writeto(small, np.full((8, 8), 10.0, dtype=np.float32))
    with pytest.raises(seiza.StackError):
        seiza.integrate_registered_frames([paths[0], paths[1], small])

    # A chosen scratch directory gives the same stack and is left empty; one
    # that does not exist is refused.
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    kept, _ = seiza.integrate_registered_frames(paths, scratch_directory=scratch)
    assert np.array_equal(kept.image, stack.image, equal_nan=True)
    assert list(scratch.iterdir()) == []
    with pytest.raises(seiza.StackError, match="scratch directory"):
        seiza.integrate_registered_frames(paths, scratch_directory=tmp_path / "missing")
    # On the CPU alone, the same stack: these frames are too small for the GPU.
    on_cpu, _ = seiza.integrate_registered_frames(
        paths, gpu=seiza.GpuOptions(enabled=False)
    )
    assert np.array_equal(on_cpu.image, stack.image, equal_nan=True)

    def stop():
        raise KeyError("stop")

    with pytest.raises(KeyError):
        seiza.integrate_registered_frames(paths, cancel=stop)


def test_a_stack_reads_a_light_at_the_angle_the_host_measured(tmp_path):
    flat = tmp_path / "master-flat.fits"
    header = fits.Header()
    header["SEIZAMST"] = "FLAT"
    header["ROTATANG"] = 3.2
    fits.writeto(flat, np.ones((128, 160), dtype=np.float32), header, overwrite=True)
    reference = tmp_path / "reference.fits"
    header = fits.Header()
    header["IMAGETYP"] = "LIGHT"
    header["ROTATANG"] = 3.2
    fits.writeto(reference, synthetic_star_field(), header, overwrite=True)
    stacker = seiza.LiveStacker(reference, flat=flat, options=no_adjustment_options())
    assert stacker.match_tolerances().light_rotation == "recorded"

    # The rotator reads 113.8 degrees; the host measured the light at 3.5.
    light = seiza.FrameSignature(width=160, height=128, channels=1, rotation_deg=113.8)
    with pytest.raises(seiza.StackError, match="113.8"):
        stacker.validate_calibration(light)
    stacker.set_match_tolerances(seiza.MatchTolerances(light_rotation=3.5))
    assert stacker.match_tolerances().light_rotation == 3.5
    assert stacker.validate_calibration(light) is None
    stacker.set_match_tolerances(seiza.MatchTolerances(light_rotation="unknown"))
    assert stacker.match_tolerances().light_rotation == "unknown"
    assert stacker.validate_calibration(light) is None
    stacker.set_match_tolerances()
    assert stacker.match_tolerances().light_rotation == "recorded"

    with pytest.raises(ValueError, match="light_rotation"):
        seiza.MatchTolerances(light_rotation="sideways")
    with pytest.raises(ValueError, match="finite"):
        seiza.MatchTolerances(light_rotation=float("nan"))


def dithered_lights(tmp_path, hot):
    """Six integer dithers of one star field, with hot pixels fixed on the
    sensor and fresh noise in each."""
    rng = np.random.default_rng(3)
    field = synthetic_star_field(height=144, width=176, noise=0.0)
    paths = []
    dithers = [(0, 0), (5, -3), (-4, 4), (3, 6), (-6, -5), (7, 2)]
    for index, (dy, dx) in enumerate(dithers):
        light = field[8 - dy : 136 - dy, 8 - dx : 168 - dx].copy()
        light += rng.normal(0.0, 1.5, light.shape).astype(np.float32)
        for y, x in hot:
            light[y, x] += 5000.0
        path = tmp_path / f"light-{index}.fits"
        fits.writeto(path, light.astype(np.float32))
        paths.append(path)
    return paths


def test_defects_are_learned_from_dithered_lights_and_repaired(tmp_path):
    hot = [(40, 50), (90, 120)]
    lights = dithered_lights(tmp_path, hot)
    defects = seiza.detect_defects(lights)
    assert defects.pixels == sorted(y * 160 + x for y, x in hot)
    assert (defects.width, defects.height, defects.frames, len(defects)) == (
        160,
        128,
        6,
        2,
    )
    # Steady hot pixels: their excess is learned, as a dark would hold it.
    assert all(abs(excess - 5000.0) < 50.0 for excess in defects.excess)
    assert defects.exposure_seconds is None

    options = seiza.StackOptions(normalization="none", rejection="none", defects=defects)
    assert options.defects.pixels == defects.pixels
    reference = seiza.LiveStacker(lights[0], options=options).snapshot()
    for y, x in hot:
        assert reference.image[y, x] < 120.0

    # Without a dark, stack_fits learns the map itself.
    seiza.stack_fits(lights, tmp_path / "auto.fits", options=no_adjustment_options())
    seiza.stack_fits(
        lights, tmp_path / "off.fits", options=no_adjustment_options(), defects="off"
    )
    with fits.open(tmp_path / "auto.fits") as auto, fits.open(tmp_path / "off.fits") as off:
        for y, x in hot:
            assert auto[0].data[y, x] < 120.0
            assert off[0].data[y, x] > 500.0
    # The map is learned under the stack's GPU options.
    cpu_options = seiza.StackOptions(
        normalization="none", rejection="none", gpu=seiza.GpuOptions(enabled=False)
    )
    seiza.stack_fits(lights, tmp_path / "auto-cpu.fits", options=cpu_options)
    with fits.open(tmp_path / "auto-cpu.fits") as on_cpu:
        for y, x in hot:
            assert on_cpu[0].data[y, x] < 120.0
    with pytest.raises(ValueError, match="defects"):
        seiza.stack_fits(lights, tmp_path / "unused.fits", defects="maybe")

    built = seiza.DefectMap(160, 128, [5, 1, 5])
    assert (built.pixels, built.frames) == ([1, 5], 0)
    with pytest.raises(ValueError, match="inside the frame"):
        seiza.DefectMap(4, 4, [16])
    with pytest.raises(seiza.StackError, match="persistence"):
        seiza.detect_defects(lights, persistence=2.0)
    on_cpu = seiza.detect_defects(lights, gpu=seiza.GpuOptions(enabled=False))
    assert (on_cpu.pixels, on_cpu.excess) == (defects.pixels, defects.excess)


@pytest.mark.parametrize("bin_factor", [2, 4])
def test_superpixel_mapping_channel_sites_survive_context(tmp_path, bin_factor):
    image = synthetic_star_field(height=256, width=320)
    paths = [tmp_path / f"sites-{i}.fits" for i in range(2)]
    for path in paths:
        fits.writeto(path, image, header=fits.Header({"BAYERPAT": "RGGB"}))
    options = seiza.StackOptions(demosaic="superpixel", bin_factor=bin_factor,
                                 normalization="none", rejection="none")
    stacker = seiza.LiveStacker(paths[0], options=options)
    d = 0.5 / bin_factor
    sites = [[-d, -d], [0.0, 0.0], [d, d]]
    assert stacker.reference_mapping["channel_sites"] == sites
    disposition = stacker.push_fits(paths[1])
    assert disposition.accepted
    assert disposition.mapping["channel_sites"] == sites
    context = tmp_path / "sites.seiza"
    stacker.save_context(context)
    reopened = seiza.LiveStacker.open_context(context)
    assert reopened.reference_mapping["channel_sites"] == sites


def test_live_stacker_records_where_frames_clipped(tmp_path):
    # 16-bit frames clip at 65535, which their encoding declares.
    image = np.clip(synthetic_star_field(), 0, 65535).astype(np.uint16)
    clipped = image.copy()
    clipped[118:120, 91:93] = 65535
    first = tmp_path / "light-001.fits"
    second = tmp_path / "light-002.fits"
    fits.writeto(first, image, overwrite=True)
    fits.writeto(second, clipped, overwrite=True)

    stacker = seiza.LiveStacker(first, options=no_adjustment_options())
    assert stacker.clipped_fraction() is None
    assert stacker.push_fits(second).accepted
    fraction = stacker.clipped_fraction()
    assert fraction.shape == image.shape and fraction.dtype == np.float32
    # One of the two observations clipped at the core; none in the sky.
    assert 0.3 <= fraction[118:120, 91:93].min() and fraction.max() <= 0.5
    assert fraction[:20, :20].max() == 0.0
    assert seiza.LiveStacker.from_array(synthetic_star_field()).clipped_fraction() is None


def test_registered_mosaic_matches_backgrounds_and_keeps_shared_nebula(tmp_path):
    yy, xx = np.mgrid[:96, :160]
    sky = (20.0 + 100.0 * np.exp(-((xx - 78) ** 2 + (yy - 48) ** 2) / 180.0)).astype(np.float32)
    paths = []
    for index in range(4):
        panel = index // 2
        image = np.full(sky.shape, np.nan, dtype=np.float32)
        left = panel * 48
        image[:, left:left + 112] = sky[:, left:left + 112] + panel * 18.0
        image[10, 70] = np.nan
        path = tmp_path / f"panel-{index}.fits"
        fits.writeto(path, image)
        paths.append(path)
    result, _ = seiza.integrate_registered_frames(
        paths, mosaic_panels=[0, 0, 1, 1], gpu=seiza.GpuOptions(enabled=False)
    )
    image = result.image
    keep = np.isfinite(image)
    basis = np.column_stack((np.ones(keep.sum()), xx[keep], yy[keep]))
    difference = image[keep] - sky[keep]
    common_plane, *_ = np.linalg.lstsq(basis, difference, rcond=None)
    assert np.sqrt(np.mean((difference - basis @ common_plane) ** 2)) < 0.1
    assert np.isnan(image[10, 70])
    assert result.coverage[10, 70] == 0
    assert result.coverage[30, 80] == 4
    with pytest.raises(seiza.StackError, match="one panel ID"):
        seiza.integrate_registered_frames(paths, mosaic_panels=[0])


def _integrate_registered_mosaic_fixture(images, ids=None):
    with tempfile.TemporaryDirectory() as folder:
        paths = []
        for i, image in enumerate(images):
            path = pathlib.Path(folder) / f"{i}.fits"
            fits.writeto(path, np.moveaxis(image, -1, 0) if image.ndim == 3 else image)
            paths.append(path)
        result, _ = seiza.integrate_registered_frames(
            paths, mosaic_panels=ids, gpu=seiza.GpuOptions(enabled=False)
        )
        return result.image.copy()


def test_a_valid_eight_pixel_overlap_is_background_matched():
    a = np.full((128, 248), np.nan, np.float32)
    b = a.copy()
    a[:, :128] = 100
    b[:, 120:] = 118
    result = _integrate_registered_mosaic_fixture([a, b], [0, 1])
    # Remove a possible common linear sky slope at the coverage boundary.
    discontinuity = float(result[64, 120] - 2 * result[64, 119] + result[64, 118])
    assert abs(discontinuity) < 0.1, f"uncorrected boundary step: {discontinuity} ADU"


@pytest.mark.parametrize("missing_blue", [False, True])
def test_rgb_channels_can_have_independent_missing_samples(missing_blue):
    images = []
    for panel in range(2):
        image = np.full((96, 160, 3), np.nan, np.float32)
        left = panel * 48
        image[:, left : left + 112] = 100 + panel * 18
        image[:, 1::2, 0] = np.nan
        image[:, ::2, 1] = np.nan
        if missing_blue:
            image[:, :, 2] = np.nan
        images.append(image)
    ordinary = _integrate_registered_mosaic_fixture(images)
    corrected = _integrate_registered_mosaic_fixture(images, [0, 1])
    change = float(np.nanmax(abs(corrected - ordinary)))
    assert change > 1, f"the 18 ADU panel mismatch was left untouched: change={change}"
    assert np.array_equal(np.isnan(corrected), np.isnan(ordinary))
    y, x = np.mgrid[:96, :160]
    for c in range(2 if missing_blue else 3):
        valid = np.isfinite(corrected[:, :, c])
        basis = np.column_stack([np.ones(valid.sum()), x[valid], y[valid]])
        values = corrected[:, :, c][valid]
        plane = np.linalg.lstsq(basis, values, rcond=None)[0]
        assert np.sqrt(np.mean((values - basis @ plane) ** 2)) < 0.1


def test_shared_stellar_signal_does_not_create_a_background_halo():
    y, x = np.mgrid[:192, :256]
    star = (1000 * np.exp(-((x - 103) ** 2 + (y - 82) ** 2) / 8)).astype(np.float32)
    results = []
    for signal in [False, True]:
        images = []
        for panel in range(2):
            image = (100 + panel * 1000 * x / 256 + (star if signal else 0)).astype(np.float32)
            image[~((x >= panel * 96) & (x < panel * 96 + 160))] = np.nan
            images.append(image)
        results.append(_integrate_registered_mosaic_fixture(images, [0, 1]))
    error = results[1] - results[0] - star
    radius_squared = (x - 103) ** 2 + (y - 82) ** 2
    outside = np.isfinite(error) & (radius_squared > 40**2)
    basis = np.column_stack([np.ones(outside.sum()), x[outside], y[outside]])
    plane = np.linalg.lstsq(basis, error[outside], rcond=None)[0]
    residual = error - (plane[0] + plane[1] * x + plane[2] * y)
    ring = np.isfinite(residual) & (radius_squared > 6**2) & (radius_squared < 30**2)
    halo = float(np.max(abs(residual[ring])))
    assert halo < 1, f"identical shared star created {halo:.3f} ADU of background structure"


def test_a_rejected_cosmic_ray_cannot_spread_into_the_background():
    outputs = []
    for hit in [False, True]:
        images = []
        for index in range(6):
            panel = index // 3
            image = np.full((192, 256), np.nan, np.float32)
            image[:, panel * 96 : panel * 96 + 160] = 100 + panel * 18
            if hit and index == 0:
                image[85, 97] += 3000
            images.append(image)
        outputs.append(_integrate_registered_mosaic_fixture(images, [0, 0, 0, 1, 1, 1]))
    error = outputs[1] - outputs[0]
    # These pixels lie outside the overlap and never saw the cosmic ray.
    halo = float(np.max(abs(error[70:102, 80:96])))
    assert halo < 0.1, f"rejected cosmic ray left a {halo:.3f} ADU background halo"
