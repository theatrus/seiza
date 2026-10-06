"""EXIF metadata from phone photos through the Python bindings."""

import struct

import seiza


def jpeg_with_focal_length(focal_35mm):
    """A minimal JPEG: SOI, an APP1 EXIF segment, EOI."""
    tiff = b"MM\x00\x2a" + struct.pack(">I", 8)
    # IFD0: one entry pointing at the Exif sub-IFD, which starts at offset 26.
    tiff += struct.pack(">H", 1) + struct.pack(">HHII", 0x8769, 4, 1, 26) + struct.pack(">I", 0)
    # Exif IFD: FocalLengthIn35mmFilm (SHORT), value left-justified.
    tiff += struct.pack(">H", 1) + struct.pack(">HHIHH", 0xA405, 3, 1, focal_35mm, 0)
    tiff += struct.pack(">I", 0)
    payload = b"Exif\x00\x00" + tiff
    return b"\xff\xd8" + b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload + b"\xff\xd9"


def test_focal_length_gives_a_scale_hint_with_a_wide_fallback(tmp_path):
    path = tmp_path / "phone.jpg"
    path.write_bytes(jpeg_with_focal_length(24))
    metadata = seiza.read_photo_metadata(path, 4032, 3024)
    assert metadata["focal_length_35mm"] == 24.0
    assert abs(metadata["scale_hint"]["nominal_arcsec_per_pixel"] - 73.8) < 0.1
    (narrow_min, narrow_max), (wide_min, wide_max) = metadata["scale_ranges"]
    assert narrow_min < 73.8 < narrow_max
    assert wide_min == 0.1 and wide_max == narrow_max


def test_files_without_exif_give_empty_metadata(tmp_path):
    path = tmp_path / "frame.fits"
    path.write_bytes(b"SIMPLE  =                    T")
    metadata = seiza.read_photo_metadata(path)
    assert metadata["capture_time_utc"] is None
    assert metadata["warnings"] == []
    assert "scale_ranges" not in metadata
    assert seiza.read_photo_metadata(path, 100, 100)["scale_ranges"] == [[0.1, 20.0]]
