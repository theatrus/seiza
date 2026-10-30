"""read_image_header reads a FITS or XISF file's header without its pixels."""

import numpy as np
import pytest
import seiza


def test_a_header_is_read_with_its_commentary(tmp_path):
    cards = [
        "SIMPLE  =                    T",
        "BITPIX  =                   16",
        "NAXIS   =                    2",
        "NAXIS1  =                    1",
        "NAXIS2  =                    1",
        "EXPTIME =                 30.0",
        "OBJECT  = 'M42     '",
        "HISTORY Calibrated with a master dark",
        "COMMENT Written by a test",
        "END",
    ]
    text = "".join(f"{card:<80}" for card in cards).encode("ascii")
    path = tmp_path / "light.fits"
    path.write_bytes(text + b" " * (-len(text) % 2880) + b"\0" * 2880)

    header = seiza.read_image_header(path)
    assert header["headers"]["EXPTIME"] == 30.0
    assert header["headers"]["OBJECT"] == "M42"
    assert header["headers"]["NAXIS1"] == 1
    assert header["history"] == ["Calibrated with a master dark"]
    assert header["comments"] == ["Written by a test"]

    xisf = tmp_path / "master.xisf"
    write_gray_xisf(xisf)
    header = seiza.read_image_header(xisf)
    assert header["headers"]["NAXIS1"] == 2
    assert header["history"] == [] and header["comments"] == []

    with pytest.raises(ValueError):
        seiza.read_image_header(tmp_path / "missing.fits")


def write_gray_xisf(path):
    """A monolithic 2x2 gray float32 XISF file, its samples inline."""
    data = np.zeros(4, dtype="<f4").tobytes().hex()
    header = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<xisf version="1.0" xmlns="http://www.pixinsight.com/xisf">'
        f'<Image geometry="2:2:1" sampleFormat="Float32" colorSpace="Gray" '
        f'location="inline:hex">{data}</Image></xisf>'
    ).encode("utf-8")
    path.write_bytes(b"XISF0100" + len(header).to_bytes(4, "little") + bytes(4) + header)
