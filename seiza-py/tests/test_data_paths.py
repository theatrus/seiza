"""Catalog path resolution through the bindings.

Opening real catalogs needs data files, so these tests exercise the
resolution wiring through its error paths: directories with nothing
usable and missing files must raise FileNotFoundError with the same
messages the Rust library produces.
"""

import pytest

import seiza


def test_open_directory_without_a_catalog_raises_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="star catalog"):
        seiza.StarCatalog.open(tmp_path)


def test_open_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        seiza.StarCatalog.open(tmp_path / "nope.bin")


def test_blind_index_directory_without_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="blind index"):
        seiza.BlindIndex.open(tmp_path)


def test_blind_index_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        seiza.BlindIndex.open(tmp_path / "nope.idx")
