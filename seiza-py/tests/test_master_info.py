"""read_master_info reads what a stacked image records of its integration."""

import numpy as np
import pytest
import seiza


def write_fits(path, cards, history=()):
    header = ["SIMPLE  =                    T", "BITPIX  =                  -32", "NAXIS   =                    2"]
    header += ["NAXIS1  =                    1", "NAXIS2  =                    1"]
    header += [f"{keyword:<8}= {value:>20}" for keyword, value in cards]
    header += [f"HISTORY {line}" for line in history]
    header.append("END")
    text = "".join(f"{card:<80}" for card in header).encode("ascii")
    text += b" " * (-len(text) % 2880)
    path.write_bytes(text + np.zeros(720, dtype=">f4").tobytes())


def test_a_master_says_where_each_value_came_from(tmp_path):
    path = tmp_path / "master.fits"
    write_fits(
        path,
        [("EXPTIME", "300.0"), ("FILTER", "'Ha      '")],
        ["Integration with PixInsight 1.8.9-2", "ImageIntegration.numberOfImages: 38"],
    )
    info = seiza.read_master_info(path)

    def pair(recorded):
        return recorded.value, recorded.source

    assert pair(info.total_exposure_seconds) == (11400.0, "EXPTIME × frames")
    assert pair(info.frames) == (38, "ImageIntegration.numberOfImages")
    assert pair(info.sub_exposure_seconds) == (300.0, "EXPTIME")
    assert pair(info.filter) == ("Ha", "FILTER")
    assert pair(info.integrator) == ("PixInsight 1.8.9-2", "Integration with PixInsight")
    assert info.drizzle_scale is None
    assert info.notes == []
    assert "38" in repr(info.frames)


def test_records_that_contradict_each_other_are_noted(tmp_path):
    path = tmp_path / "siril.fits"
    write_fits(
        path,
        [("STACKCNT", "20"), ("EXPTIME", "180.0"), ("LIVETIME", "1800.0")],
    )
    info = seiza.read_master_info(str(path))
    assert info.total_exposure_seconds.value == 1800.0
    assert info.total_exposure_seconds.source == "LIVETIME"
    assert len(info.notes) == 1
    with pytest.raises(ValueError):
        seiza.read_master_info(tmp_path / "missing.fits")
