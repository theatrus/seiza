"""Offline checks for the real-data benchmark's integrity/parity gates."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def benchmark():
    path = Path(__file__).parents[1] / "examples/benchmark_aperture_real_data.py"
    spec = importlib.util.spec_from_file_location("real_photometry_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cached_file_is_verified_even_when_download_is_enabled(benchmark, tmp_path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"original")
    entry = dict(file=path.name, bytes=8, sha256=hashlib.sha256(b"original").hexdigest())
    assert benchmark.verified_file(tmp_path, entry, "unused", False) == path
    path.write_bytes(b"modified")  # Same size; checksum must catch it.
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        benchmark.verified_file(tmp_path, entry, "unused", True)
    assert path.read_bytes() == b"modified"  # Do not silently replace existing data.


def test_comparison_checks_values_and_input_identity_but_not_timing(benchmark):
    before = dict(dataset="fixture/v1", geometry={}, fields=["flux", "unusable"], frames=[],
                  rows=[dict(file="a.fit", stars=1, layout="mono", strides=[8, 4],
                             values=[[1.25, False]], ms=2.0)])
    after = copy.deepcopy(before)
    after["rows"][0]["ms"] = 4.0
    assert benchmark.compare(before, after)["speedups"][0]["speedup"] == 0.5
    for values in ([[1.26, False]], [[1.25, True]]):
        changed = copy.deepcopy(after)
        changed["rows"][0]["values"] = values
        with pytest.raises(ValueError, match="differs in values"):
            benchmark.compare(before, changed)
    after["dataset"] = "fixture/v2"
    with pytest.raises(ValueError, match="inputs differ: dataset"):
        benchmark.compare(before, after)
