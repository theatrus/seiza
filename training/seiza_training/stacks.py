"""Take in the finished stacks PSF Guard has made from the corpus.

PSF Guard keeps each stack job under
`<stacks>/<database>/stack-previews/<job>/`: a `manifest.json`, and a
`group-<n>.fits` for each group it stacked, one target, filter and exposure
over every night it was imaged. Each ready group whose FITS is present
becomes a row of `stacks`, and its image is copied into the store as a
product of kind `stack`, so later passes read it as they read the corpus's
own stacks. Its frames are not stacked again.

Each PSF Guard database is one telescope's scheduler database: the one under
the corpus whose `_Scheduler/schedulerdb.sqlite` holds the targets its jobs
name. A group lists its frames by scheduler image id; the file names the
scheduler recorded for them find the corpus frames, and so the sets those
frames belong to. The split pass puts a stack in its sets' sky group.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np

import seiza

from .manifest import Manifest, dumps, now
from .store import Store, read_image, robust_sigma, versions

ORIGIN = "psf-guard"


def scheduler_records(corpus: Path) -> dict[str, dict[str, Any]]:
    """Each telescope's scheduler targets, as (id, name) pairs, and its
    images' file names by image id. The database is copied first, so
    reading it never holds a lock that the scheduler would meet."""
    records = {}
    for path in sorted(corpus.glob("*/_Scheduler/schedulerdb.sqlite")):
        telescope = path.parent.parent.name
        with tempfile.TemporaryDirectory() as scratch:
            copy = Path(scratch) / "schedulerdb.sqlite"
            shutil.copyfile(path, copy)
            db = sqlite3.connect(copy)
            try:
                targets = {(row[0], row[1]) for row in db.execute("SELECT Id, name FROM target")}
                files = {}
                for image_id, metadata in db.execute("SELECT Id, metadata FROM acquiredimage"):
                    try:
                        name = json.loads(metadata or "{}").get("FileName")
                    except ValueError:
                        continue
                    if name:
                        # The scheduler records the path on the rig, often Windows.
                        files[image_id] = PureWindowsPath(name).name
            except sqlite3.Error as error:
                print(f"stacks: {path}: {error}", file=sys.stderr)
                continue
            finally:
                db.close()
        records[telescope] = {"targets": targets, "files": files}
    return records


def read_jobs(database: Path) -> list[tuple[Path, dict[str, Any]]]:
    jobs = []
    for path in sorted(database.glob("stack-previews/*/manifest.json")):
        try:
            jobs.append((path.parent, json.loads(path.read_text())))
        except (OSError, ValueError) as error:
            print(f"stacks: {path}: {error}", file=sys.stderr)
    return jobs


def database_telescope(jobs: list[tuple[Path, dict[str, Any]]], records: dict[str, dict[str, Any]]) -> str | None:
    """The telescope whose scheduler names most of the jobs' targets."""
    named = {
        (group["target_id"], group["target_name"])
        for _, job in jobs
        for group in job.get("groups", [])
        if group.get("target_id") is not None
    }
    scores = {telescope: len(named & record["targets"]) for telescope, record in records.items()}
    best = max(scores, key=scores.get, default=None)
    return best if best is not None and scores[best] > 0 else None


def corpus_frames(manifest: Manifest, telescope: str) -> dict[str, tuple[str, str | None]]:
    """A telescope's corpus lights by file name: their paths and sets."""
    return {
        Path(row["path"]).name: (row["path"], row["set_id"])
        for row in manifest.rows(
            "SELECT path, set_id FROM frames WHERE telescope = ? AND role = 'light'", (telescope,)
        )
    }


def copy_stack(store: Store, stack_id: str, path: Path) -> tuple[dict[str, str], dict[str, Any], tuple[int, int, int]]:
    """Copy a stack's FITS into the store as a plane."""
    signature = seiza.probe_frame(path).signature
    width, height, channels = signature.width, signature.height, signature.channels or 1
    image = read_image(path, width, height).astype(np.float32, copy=False)
    directory = store.products / stack_id / "stack"
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "image.npy", np.ascontiguousarray(image))
    finite = image[np.isfinite(image)]
    stats = {
        "shape": list(image.shape),
        "median": float(np.median(finite)) if finite.size else None,
        "noise": robust_sigma(image if image.ndim == 2 else image[0]),
    }
    return {"image": store.relative(directory / "image.npy")}, stats, (width, height, channels)


def import_stacks(
    manifest: Manifest,
    store: Store,
    stacks_root: Path,
    corpus: Path,
    *,
    again: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Record and copy every ready group stack under `stacks_root`."""
    records = scheduler_records(corpus)
    counts = {"copied": 0, "kept": 0, "failed": 0, "databases": {}}
    for database in sorted(path for path in stacks_root.iterdir() if path.is_dir()):
        jobs = read_jobs(database)
        telescope = database_telescope(jobs, records)
        counts["databases"][database.name] = telescope
        if telescope is None:
            print(f"stacks: {database.name}: no scheduler under the corpus names its targets", file=sys.stderr)
            continue
        files = records[telescope]["files"]
        frames = corpus_frames(manifest, telescope)
        for job_directory, job in jobs:
            for group in job.get("groups", []):
                if limit is not None and counts["copied"] >= limit:
                    return counts
                fits = job_directory / f"group-{group['index']}.fits"
                if group.get("state") != "ready" or not fits.is_file():
                    continue
                stack_id = f"{ORIGIN}/{database.name}/{job['job_id']}/{group['index']}"
                existing = manifest.rows("SELECT revision, status FROM stacks WHERE id = ?", (stack_id,))
                if (
                    existing and not again and existing[0]["status"] == "copied"
                    and existing[0]["revision"] == job.get("artifact_revision")
                ):
                    counts["kept"] += 1
                    continue
                image_ids = [frame["image_id"] for frame in group.get("frames", []) if frame.get("disposition") != "rejected"]
                matched = [frames[files[image]] for image in image_ids if files.get(image) in frames]
                sets = sorted({set_id for _, set_id in matched if set_id})
                exposure = group.get("exposure_group") or {}
                row = {
                    "id": stack_id,
                    "origin": ORIGIN,
                    "path": str(fits),
                    "revision": job.get("artifact_revision"),
                    "telescope": telescope,
                    "target": group.get("target_name"),
                    "filter": group.get("filter_name"),
                    "exposure_seconds": exposure.get("max_seconds"),
                    "frames": group.get("accepted_frames"),
                    "total_exposure_seconds": group.get("total_exposure_seconds"),
                    "method": dumps(job.get("method", {})),
                    "stacking_version": job.get("stacking_version"),
                    "sets": dumps(sets),
                    "created": now(),
                }
                try:
                    written, stats, (width, height, channels) = copy_stack(store, stack_id, fits)
                except (ValueError, OSError, RuntimeError) as error:
                    with manifest.transaction():
                        manifest.upsert("stacks", {**row, "status": "failed", "reason": str(error)}, key="id")
                    counts["failed"] += 1
                    print(f"stacks: {stack_id}: {error}", file=sys.stderr)
                    continue
                with manifest.transaction():
                    manifest.upsert(
                        "stacks",
                        {**row, "width": width, "height": height, "channels": channels,
                         "status": "copied", "reason": None},
                        key="id",
                    )
                    manifest.upsert(
                        "products",
                        {
                            "id": f"{stack_id}/stack",
                            "set_id": stack_id,
                            "kind": "stack",
                            "recipe": dumps({
                                "recipe": "stack", "origin": ORIGIN, "source": str(fits),
                                "frames_found_in_corpus": len(matched), "frames_listed": len(image_ids),
                            }),
                            "frames": dumps([path for path, _ in matched]),
                            "files": dumps(written),
                            "stats": dumps(stats),
                            "versions": dumps({**versions(), "made_with": job.get("stacking_version")}),
                            "created": now(),
                            "status": "built",
                            "reason": None,
                        },
                        key="id",
                    )
                counts["copied"] += 1
                print(
                    f"stacks: {stack_id}: {group.get('target_name')} {group.get('filter_name')}, "
                    f"{len(matched)} of {len(image_ids)} frames in the corpus",
                    file=sys.stderr,
                )
    return counts
