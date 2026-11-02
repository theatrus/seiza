"""Pass 1: read every header and group the lights into sets.

The telescope corpus keeps raw lights by night, under
`<telescope>/_Source/<year>/<target>/<night>/LIGHT`, and masters made by
other software under `<telescope>/_Process`. The calibration library keeps
masters under `<library>/<database>/calibration-masters`, and a calibration
folder keeps raw darks, biases and flats, in any layout. Every header is
read through Seiza (`probe_frame` and `read_image_header`), never parsed
here, and a file whose size and modification time have not changed since
the last crawl is not read again.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import seiza

from . import sky
from .manifest import Manifest, dumps, loads

IMAGE_SUFFIXES = {".fits", ".fit", ".fts", ".xisf"}

# Two exposure cards that disagree by more than this fraction are flagged.
EXPOSURE_AGREEMENT = 0.005


@dataclass(frozen=True)
class Candidate:
    """A file the crawl found, with what its path says about it."""

    path: Path
    origin: str
    telescope: str | None
    target: str | None
    night: str | None


def image_files(directory: Path) -> Iterator[Path]:
    """Every image file below `directory`, skipping hidden entries."""
    for root, folders, files in os.walk(directory):
        folders[:] = sorted(folder for folder in folders if not folder.startswith("."))
        for name in sorted(files):
            if not name.startswith(".") and Path(name).suffix.lower() in IMAGE_SUFFIXES:
                yield Path(root) / name


def corpus_candidates(corpus: Path) -> Iterator[Candidate]:
    """The lights under each telescope's `_Source`, and the masters under
    its `_Process`."""
    for telescope in sorted(entry for entry in corpus.iterdir() if entry.is_dir()):
        if telescope.name.startswith((".", "_")):
            continue
        source = telescope / "_Source"
        if source.is_dir():
            for path in image_files(source):
                parts = path.relative_to(source).parts
                # <year>/<target>/<night>/LIGHT/<file>
                if len(parts) >= 5 and parts[-2].upper() == "LIGHT":
                    yield Candidate(path, "source", telescope.name, parts[1], parts[2])
                else:
                    yield Candidate(path, "source-unsorted", telescope.name, None, None)
        process = telescope / "_Process"
        if process.is_dir():
            for path in image_files(process):
                target = process_target(path.relative_to(process))
                yield Candidate(path, "process", telescope.name, target, None)


def process_target(relative: Path) -> str | None:
    """The target a `_Process` folder names, without its trailing date:
    `Done/NGC 7000 - 20250812/...` gives `NGC 7000`."""
    for part in relative.parts[:-1]:
        if part in ("Done", "InProgress", "Stacked", "master"):
            continue
        return re.sub(r"\s*-\s*\d{8}$", "", part).strip() or None
    return None


def library_candidates(library: Path) -> Iterator[Candidate]:
    """Calibration masters under `<library>/<database>/calibration-masters`."""
    for database in sorted(entry for entry in library.iterdir() if entry.is_dir()):
        masters = database / "calibration-masters"
        if masters.is_dir():
            for path in image_files(masters):
                yield Candidate(path, "library", None, None, None)


def calibration_candidates(calibration: Path) -> Iterator[Candidate]:
    """Raw calibration frames anywhere below `calibration`. Their headers
    say what each is, and their camera settings, not their folders, match
    them to lights."""
    for path in image_files(calibration):
        yield Candidate(path, "calibration", None, None, None)


def number(value: Any) -> float | None:
    """A header card as a float, when it is a number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def frame_row(candidate: Candidate, stat: os.stat_result) -> dict[str, Any]:
    """Read one file's header into a manifest row."""
    row: dict[str, Any] = {
        "path": str(candidate.path),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "telescope": candidate.telescope,
        "origin": candidate.origin,
        "target": candidate.target,
        "night": candidate.night,
        "error": None,
    }
    flags: list[str] = []
    try:
        probe = seiza.probe_frame(candidate.path)
        header = seiza.read_image_header(candidate.path)["headers"]
    # A header Seiza cannot read (seiza.StackError is a RuntimeError) is
    # recorded as unreadable, never fatal to the crawl.
    except (ValueError, OSError, RuntimeError) as error:
        row.update(flags=dumps(["unreadable"]), error=str(error))
        return row
    signature = probe.signature
    state = probe.calibration_state
    ra, dec = sky.header_position(header)
    row.update(
        role=probe.role,
        is_master=int(probe.is_master),
        raw_image_type=probe.raw_image_type,
        header_telescope=signature.telescope,
        camera=signature.camera,
        filter=signature.filter,
        gain=signature.gain,
        offset=signature.offset,
        readout_mode=signature.readout_mode,
        binning_x=signature.binning_x,
        binning_y=signature.binning_y,
        width=signature.width,
        height=signature.height,
        channels=signature.channels,
        bayer_pattern=signature.bayer_pattern,
        exposure_seconds=signature.exposure_seconds,
        camera_temp_c=signature.camera_temp_c,
        captured_at_unix=signature.captured_at_unix,
        focal_length_mm=signature.focal_length_mm,
        pixel_size_um=number(header.get("XPIXSZ")),
        rotation_deg=signature.rotation_deg,
        ra_deg=ra,
        dec_deg=dec,
        object=header.get("OBJECT") if isinstance(header.get("OBJECT"), str) else None,
        airmass=number(header.get("AIRMASS")),
        pier_side=header.get("PIERSIDE") if isinstance(header.get("PIERSIDE"), str) else None,
        saturation=number(header.get("SATURATE")),
        software=header.get("SWCREATE") if isinstance(header.get("SWCREATE"), str) else None,
        calibration_state=dumps(
            {
                "bias_subtracted": state.bias_subtracted,
                "dark_subtracted": state.dark_subtracted,
                "flat_normalized": state.flat_normalized,
            }
        ),
    )
    exposure, exptime = number(header.get("EXPOSURE")), number(header.get("EXPTIME"))
    if exposure is not None and exptime is not None:
        if abs(exposure - exptime) > EXPOSURE_AGREEMENT * max(abs(exposure), abs(exptime)):
            flags.append(f"exposure-disagrees:EXPOSURE={exposure:g},EXPTIME={exptime:g}")
    if candidate.origin == "source" and probe.role != "light":
        flags.append(f"role-is-{probe.role}")
    if signature.captured_at_unix is None:
        flags.append("no-capture-time")
    if candidate.origin == "source" and not signature.filter:
        flags.append("no-filter")
    if signature.gain is None:
        flags.append("no-gain")
    row["flags"] = dumps(flags)
    return row


def set_key(row: Any) -> tuple:
    """What makes two lights one set."""
    exposure = row["exposure_seconds"]
    return (
        row["telescope"],
        row["target"],
        row["night"],
        row["filter"] or "",
        row["camera"] or "",
        row["gain"],
        row["offset"],
        None if exposure is None else round(exposure, 2),
        row["binning_x"] or 1,
        row["width"],
        row["height"],
        row["channels"],
    )


def set_identity(key: tuple) -> tuple[str, str]:
    """A stable id and a readable name for the set with `key`."""
    telescope, target, night, filter_, _camera, gain, _offset, exposure, binning, *_ = key
    digest = hashlib.sha256(repr(key).encode()).hexdigest()[:12]
    exposure_text = "unknown" if exposure is None else f"{exposure:g}s"
    name = f"{telescope}/{target}/{night}/{filter_ or 'nofilter'}-{exposure_text}-g{gain}-bin{binning}"
    return digest, name


def spread(values: list[float]) -> float:
    return max(values) - min(values) if values else 0.0


def group_sets(manifest: Manifest) -> dict[str, int]:
    """Group the readable lights into sets, and write one row per set."""
    lights = manifest.rows(
        "SELECT * FROM frames WHERE origin = 'source' AND role = 'light' AND error IS NULL"
    )
    groups: dict[tuple, list[Any]] = defaultdict(list)
    for row in lights:
        groups[set_key(row)].append(row)
    counts: dict[str, int] = {}
    with manifest.transaction():
        manifest.db.execute("UPDATE frames SET set_id = NULL")
        for key, rows in groups.items():
            set_id, name = set_identity(key)
            telescope, target, night, filter_, camera, gain, offset, exposure, binning, width, height, channels = key
            positions = [(row["ra_deg"], row["dec_deg"]) for row in rows if row["ra_deg"] is not None]
            ra = dec = None
            if positions:
                # The middle frame's pointing: a mean wraps badly at RA 0.
                ra, dec = sorted(positions)[len(positions) // 2]
            first = rows[0]
            scale = sky.pixel_scale_arcsec(first["pixel_size_um"], first["focal_length_mm"], binning)
            flags = sorted({flag for row in rows for flag in loads(row["flags"], [])})
            rotations = [row["rotation_deg"] for row in rows if row["rotation_deg"] is not None]
            if spread(rotations) > 1.0:
                flags.append(f"mixed-rotation:{spread(rotations):.1f}deg")
            temperatures = [row["camera_temp_c"] for row in rows if row["camera_temp_c"] is not None]
            if spread(temperatures) > 2.0:
                flags.append(f"mixed-temperature:{spread(temperatures):.1f}C")
            manifest.upsert(
                "sets",
                {
                    "id": set_id,
                    "name": name,
                    "telescope": telescope,
                    "target": target,
                    "filter": filter_ or None,
                    "camera": camera or None,
                    "gain": gain,
                    "offset": offset,
                    "exposure_seconds": exposure,
                    "binning": binning,
                    "width": width,
                    "height": height,
                    "channels": channels,
                    "bayer_pattern": first["bayer_pattern"],
                    "night": night,
                    "frames": len(rows),
                    "ra_deg": ra,
                    "dec_deg": dec,
                    "pixel_scale_arcsec": scale,
                    "field_deg": sky.field_diagonal_deg(scale, width, height),
                    "flags": dumps(flags),
                },
                key="id",
            )
            manifest.db.executemany(
                "UPDATE frames SET set_id = ? WHERE path = ?",
                [(set_id, row["path"]) for row in rows],
            )
            counts[set_id] = len(rows)
        # A set whose lights all went away is no longer a set.
        manifest.db.execute(
            "DELETE FROM sets WHERE id NOT IN (SELECT DISTINCT set_id FROM frames WHERE set_id IS NOT NULL)"
        )
    return counts


def crawl(
    manifest: Manifest,
    corpus: Path | None,
    library: Path | None,
    *,
    calibration: Path | None = None,
    workers: int = 16,
    progress: bool = True,
) -> dict[str, Any]:
    """Read every new or changed header under `corpus`, `library` and
    `calibration`, then regroup the sets."""
    candidates: list[Candidate] = []
    if corpus is not None:
        candidates.extend(corpus_candidates(corpus))
    if library is not None:
        candidates.extend(library_candidates(library))
    if calibration is not None:
        candidates.extend(calibration_candidates(calibration))
    known = manifest.frame_stamps()
    pending: list[tuple[Candidate, os.stat_result]] = []
    present = set()
    for candidate in candidates:
        path = str(candidate.path)
        present.add(path)
        try:
            stat = candidate.path.stat()
        except OSError:
            continue
        if known.get(path) != (stat.st_size, stat.st_mtime):
            pending.append((candidate, stat))
    read = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = pool.map(lambda item: frame_row(*item), pending)
        batch = []
        for row in rows:
            batch.append(row)
            read += 1
            if len(batch) >= 500:
                with manifest.transaction():
                    for item in batch:
                        manifest.upsert("frames", item)
                batch.clear()
                if progress:
                    print(f"crawl: read {read} of {len(pending)} headers", file=sys.stderr)
        with manifest.transaction():
            for item in batch:
                manifest.upsert("frames", item)
    # Files that are gone, under the roots this crawl covered.
    roots = [str(root) for root in (corpus, library, calibration) if root is not None]
    gone = [
        path for path in known if path not in present and any(path.startswith(root) for root in roots)
    ]
    with manifest.transaction():
        manifest.db.executemany("DELETE FROM frames WHERE path = ?", [(path,) for path in gone])
    counts = group_sets(manifest)
    by_origin = {
        f"{row['origin']}:{row['role']}": row["count"]
        for row in manifest.rows(
            "SELECT origin, role, COUNT(*) AS count FROM frames GROUP BY origin, role"
        )
    }
    return {
        "found": len(candidates),
        "read": read,
        "removed": len(gone),
        "sets": len(counts),
        "unreadable": manifest.rows("SELECT COUNT(*) AS n FROM frames WHERE error IS NOT NULL")[0]["n"],
        "by_origin": by_origin,
    }
