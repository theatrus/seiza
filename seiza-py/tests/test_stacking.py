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


def test_fits_live_stacker_protects_inputs_from_duplicates_and_output(tmp_path):
    image = synthetic_star_field()
    first = tmp_path / "light-001.fits"
    second = tmp_path / "light-002.fits"
    output = tmp_path / "stack.fits"
    fits.writeto(first, image, overwrite=True)
    fits.writeto(second, image, overwrite=True)

    stacker = seiza.LiveStacker(first, options=no_adjustment_options())
    assert stacker.push_fits(second).accepted
    with pytest.raises(ValueError, match="already been used"):
        stacker.push_fits(second)
    with pytest.raises(ValueError, match="must not refer"):
        stacker.finish(first)
    assert stacker.accepted_frames == 2
    assert stacker.finish(output).accepted_frames == 2
    assert output.exists()


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
