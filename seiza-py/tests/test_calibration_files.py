"""The shared file coordinator, through Python's native class rather than JSON."""
from pathlib import Path
import numpy as np
import pytest
import seiza


@pytest.fixture
def files(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    library = tmp_path / "library"
    library.mkdir()
    def write(path, kind, gain, level):
        header = fits.Header({"IMAGETYP": kind, "INSTRUME": "TestCam", "GAIN": gain,
            "OFFSET": 50, "XBINNING": 1, "YBINNING": 1,
            "EXPTIME": 0.01 if kind == "BIAS" else 30.0,
            "DATE-OBS": "2026-10-20T20:00:00Z"})
        data = level + (np.arange(4096).reshape(64, 64) % 7 - 3)
        fits.writeto(path, data.astype(np.float32), header, overwrite=True)
    lights = []
    for i, gain in enumerate([100, 200, 100]):
        path = tmp_path / f"light-{i}.fits"
        write(path, "LIGHT", gain, 2000)
        lights.append(path)
    for gain in [100, 200]:
        for i in range(3):
            write(library / f"bias-{gain}-{i}.fits", "BIAS", gain, gain + 400)
    return lights, library, tmp_path / "cache", write


def test_native_class_analyzes_prepares_reuses_and_checks_sources(files):
    lights, library, cache, write = files
    events = []
    job = seiza.CalibrationFiles(lights, cache, library=library)
    review = job.analyze(progress=events.append)
    assert review["snapshot"]["assignments"] == [0, 1, 0]
    assert review["file_sessions"] == []
    assert review["missing"] == [False, False, False]
    assert "evidence" in {e["phase"] for e in events}
    built = job.prepare(review["review_key"], progress=events.append)
    assert len(built["file_sessions"]) == 2
    assert "masters" in {e["phase"] for e in events}
    seiza.CalibrationFiles.verify(built["sources"])
    masters = [Path(s["bias"]) for s in built["file_sessions"]]
    stamps = [p.stat().st_mtime_ns for p in masters]
    assert job.prepare(review["review_key"])["file_sessions"] == built["file_sessions"]
    assert [p.stat().st_mtime_ns for p in masters] == stamps
    changed = seiza.CalibrationFiles(lights, cache, library=library,
        rules=seiza.CalibrationRules({"use_darks": False}))
    with pytest.raises(ValueError, match="review changed"):
        changed.prepare(review["review_key"])
    write(lights[0], "LIGHT", 100, 3100)
    with pytest.raises(ValueError, match="review changed"):
        job.prepare(review["review_key"])
    with pytest.raises(ValueError, match="review changed"):
        seiza.CalibrationFiles.verify(built["sources"])


def test_provided_files_and_cancel_callback_errors(files):
    lights, library, cache, _ = files
    job = seiza.CalibrationFiles(lights, cache, provided=list(library.iterdir()),
        dark_association="one_set")
    with pytest.raises(ValueError, match="cancelled"):
        job.analyze(cancel=lambda: True)
    def broken(_):
        raise RuntimeError("progress broke")
    with pytest.raises(RuntimeError, match="progress broke"):
        job.analyze(progress=broken)
    with pytest.raises(KeyboardInterrupt):
        job.analyze(cancel=lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert job.analyze()["missing"] == [False, False, False]
    with pytest.raises(ValueError, match="Unknown incomplete-light"):
        seiza.CalibrationFiles(lights, cache, incomplete="guess")
