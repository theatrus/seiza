"""Shared runtime discovery and cancellation never need network access."""
import pytest
import seiza


def test_missing_runtime_is_reported_without_creating_files(tmp_path):
    directory = tmp_path / "runtime"
    assert seiza.installed_onnxruntime(directory) is None
    assert not directory.exists()


def test_cancelled_install_never_downloads_or_creates_files(tmp_path):
    directory = tmp_path / "runtime"
    with pytest.raises(RuntimeError, match="cancelled"):
        seiza.install_onnxruntime(directory, cancel=lambda: True)
    assert not directory.exists()


def test_cancellation_callback_exception_is_preserved(tmp_path):
    def cancel():
        raise ValueError("stop from Python")

    with pytest.raises(ValueError, match="stop from Python"):
        seiza.install_onnxruntime(tmp_path / "runtime", cancel=cancel)


def test_invalid_receipt_is_not_treated_as_an_install(tmp_path):
    (tmp_path / "current.json").write_text("not json")
    assert seiza.installed_onnxruntime(tmp_path) is None
