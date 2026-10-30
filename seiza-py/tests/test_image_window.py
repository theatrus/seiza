"""read_image_window reads a rectangle of a file's samples in physical units."""

import numpy as np
import pytest
import seiza


def write_fits(path, data, bitpix, cards=()):
    """A primary-HDU FITS file of `data` (planes, height, width or height,
    width), stored big-endian as `bitpix`, with extra `cards`."""
    axes = list(reversed(data.shape))
    header = ["SIMPLE  =                    T", f"BITPIX  = {bitpix:>20}", f"NAXIS   = {len(axes):>20}"]
    header += [f"NAXIS{index + 1:<3}= {length:>20}" for index, length in enumerate(axes)]
    header += [f"{keyword:<8}= {value:>20}" for keyword, value in cards]
    header.append("END")
    text = "".join(f"{card:<80}" for card in header).encode("ascii")
    text += b" " * (-len(text) % 2880)
    dtype = {16: ">i2", -32: ">f4"}[bitpix]
    payload = data.astype(dtype).tobytes()
    payload += b"\0" * (-len(payload) % 2880)
    path.write_bytes(text + payload)


def test_a_window_holds_the_physical_samples_it_covers(tmp_path):
    stored = (np.arange(9 * 7).reshape(7, 9) * 37 % 3000 - 1500).astype(np.int16)
    path = tmp_path / "frame.fits"
    write_fits(path, stored, 16, [("BZERO", "32768")])

    window = seiza.read_image_window(path, 2, 3, 4, 2)
    assert (window.x, window.y, window.width, window.height) == (2, 3, 4, 2)
    assert (window.image_width, window.image_height, window.planes) == (9, 7, 1)
    assert window.values.dtype == np.float32
    assert window.values.shape == (1, 2, 4)
    np.testing.assert_array_equal(window.values[0], stored[3:5, 2:6].astype(np.float64) + 32768)


def test_an_rgb_cube_gives_each_plane(tmp_path):
    cube = np.arange(3 * 5 * 6, dtype=np.float32).reshape(3, 5, 6) / 4
    path = tmp_path / "rgb.fits"
    write_fits(path, cube, -32)

    window = seiza.read_image_window(path, 1, 1, 3, 3)
    assert window.values.shape == (3, 3, 3)
    np.testing.assert_array_equal(window.values, cube[:, 1:4, 1:4])


def test_a_window_is_clipped_and_one_off_the_image_is_empty(tmp_path):
    path = tmp_path / "small.fits"
    write_fits(path, np.ones((4, 4), dtype=np.float32), -32)

    clipped = seiza.read_image_window(path, 3, 2, 10, 10)
    assert (clipped.x, clipped.y, clipped.width, clipped.height) == (3, 2, 1, 2)
    assert clipped.values.shape == (1, 2, 1)

    outside = seiza.read_image_window(path, 9, 9, 2, 2)
    assert (outside.width, outside.height) == (0, 0)
    assert outside.values.shape == (1, 0, 0)


def test_too_large_a_window_and_a_missing_file_raise(tmp_path):
    path = tmp_path / "small.fits"
    write_fits(path, np.zeros((4, 4), dtype=np.float32), -32)
    with pytest.raises(ValueError):
        seiza.read_image_window(path, 0, 0, 1025, 1)
    with pytest.raises(ValueError):
        seiza.read_image_window(tmp_path / "missing.fits", 0, 0, 1, 1)


def write_normalized_xisf(path, samples):
    """A monolithic XISF file of the gray float32 `samples` (height, width),
    declared normalized as PixInsight writes it: bounds="0:1"."""
    height, width = samples.shape
    data = samples.astype("<f4").tobytes().hex()
    header = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<xisf version="1.0" xmlns="http://www.pixinsight.com/xisf">'
        f'<Image geometry="{width}:{height}:1" sampleFormat="Float32" bounds="0:1" '
        f'colorSpace="Gray" location="inline:hex">{data}</Image></xisf>'
    ).encode("utf-8")
    path.write_bytes(b"XISF0100" + len(header).to_bytes(4, "little") + bytes(4) + header)


def test_a_normalized_xisf_reads_as_stored_or_on_the_scale_asked_for(tmp_path):
    samples = (np.arange(12, dtype=np.float32) / 16).reshape(3, 4)
    path = tmp_path / "master.xisf"
    write_normalized_xisf(path, samples)

    stored = seiza.read_image_window(path, 1, 1, 2, 2)
    np.testing.assert_array_equal(stored.values[0], samples[1:3, 1:3])

    scaled = seiza.read_image_window(path, 1, 1, 2, 2, normalized_full_scale=65535.0)
    np.testing.assert_array_equal(scaled.values[0], samples[1:3, 1:3] * np.float32(65535))

    # A FITS file declares no bounds, so the scale leaves it alone.
    fits = tmp_path / "frame.fits"
    write_fits(fits, samples, -32)
    unscaled = seiza.read_image_window(fits, 1, 1, 2, 2, normalized_full_scale=65535.0)
    np.testing.assert_array_equal(unscaled.values[0], samples[1:3, 1:3])

    for scale in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="normalized_full_scale"):
            seiza.read_image_window(path, 0, 0, 1, 1, normalized_full_scale=scale)
