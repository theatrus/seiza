"""Native Python coverage for batch, live, and calibration stacking APIs."""

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


def test_expanding_stack_measures_the_lens_across_panels(tmp_path):
    paths, axis = lens_panels(tmp_path)
    options = seiza.StackOptions(
        normalization="none", minimum_overlap=0.1, registration_model="quadratic"
    )
    stacker = seiza.LiveStacker.open_fits_expanding(paths, options=options)
    lens = stacker.expanded_lens
    assert lens is not None
    assert np.hypot(lens["center_x"] - axis[0], lens["center_y"] - axis[1]) < 8.0
    assert lens["overlap_rms_pixels"] < 0.2 < lens["undistorted_overlap_rms_pixels"]
    assert lens["focal_length_pixels"] is None
    for path in paths:
        assert stacker.push_fits(path).accepted


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


def test_live_stacker_state_describes_the_stack(tmp_path):
    _, paths = lights_with_a_header(tmp_path, count=2)
    stacker = seiza.LiveStacker(paths[0], options=no_adjustment_options())
    assert stacker.push_fits(paths[1]).accepted

    state = stacker.state()
    assert len(state.configuration_fingerprint) == 64
    assert (state.width, state.height, state.channels) == (160, 128, 1)
    assert (state.accepted_frames, state.rejected_frames) == (2, 0)
    assert state.input_mode == "calibrate-and-prepare"
    assert state.input_paths == paths
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
    assert bias in after.input_paths
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
