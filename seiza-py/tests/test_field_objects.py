"""Catalogued objects in a solved field, and when an image was taken."""

import numpy as np
import pytest

import seiza


def test_read_capture_time_reads_fits_headers(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    dated = tmp_path / "dated.fits"
    header = fits.Header()
    header["DATE-OBS"] = "2025-07-20T12:34:56.5"
    fits.writeto(dated, np.zeros((4, 4), dtype=np.float32), header)
    assert seiza.read_capture_time(dated) == "2025-07-20T12:34:56.5"

    undated = tmp_path / "undated.fits"
    fits.writeto(undated, np.zeros((4, 4), dtype=np.float32))
    assert seiza.read_capture_time(undated) is None

    with pytest.raises(ValueError):
        seiza.read_capture_time(tmp_path / "missing.fits")


def test_field_objects_say_why_a_catalog_gave_nothing(tmp_path):
    wcs = seiza.Wcs.from_center_scale_rotation((10.0, 20.0), (50.0, 50.0), 3.6)
    found = seiza.field_objects(wcs, 100, 100, catalog_dir=tmp_path)
    assert isinstance(found, seiza.FieldObjects)
    assert found.objects == []
    assert "objects.bin" in found.object_catalog_error
    assert "transients.bin" in found.transient_catalog_error
    assert found.minor_body_error is not None

    dated = seiza.field_objects(
        wcs, 100, 100, catalog_dir=tmp_path, capture_time="2025-07-20T12:00:00Z"
    )
    assert "minor-bodies.bin" in dated.minor_body_error
