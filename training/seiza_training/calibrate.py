"""Pass 2: choose calibration masters for every set.

Seiza's `plan_calibration` applies the same matching policy the desktop
apps and PSF Guard use, to the masters the crawl found in the library. The
plan must suit every light of the set. A set needs a flat, and a dark or a
bias; a set without one is left out, with the reason recorded.

The plan orders the masters that suit by time, nearest first, as frames to
combine; it does not judge a master's own noise. A master's noise is the
same in every light, so a bias or dark is chosen by it: the nearest whose
noise, scaled to the lights' exposure, is at most `MASTER_NOISE` of one
frame's read noise, or else the quietest. A master of `n` frames scaled by
`s` adds `s / √n`: two 75-second darks scaled to 300 seconds add 2.8 times
a frame's read noise to every light. A flat is taken nearest in time, as
dust moves.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import seiza

from .manifest import Manifest, dumps

KINDS = ("bias", "dark", "flat")
# The read noise a bias or dark master may add, as a share of one frame's.
MASTER_NOISE = 0.2
_counts: dict[str, tuple[float | None, float | None]] = {}


def master_noise(path: str, kind: str, exposure: float | None) -> float | None:
    """The read noise the master at `path` adds to a light of `exposure`
    seconds, as a share of one frame's: its scale over the square root of
    the frames it combines. `None` when its header lacks either."""
    if path not in _counts:
        header = seiza.read_image_header(path)["headers"]
        _counts[path] = (header.get("NCOMBINE"), header.get("EXPTIME"))
    frames, seconds = _counts[path]
    if not frames:
        return None
    scale = 1.0
    if kind == "dark":
        if not seconds or not exposure:
            return None
        scale = exposure / seconds
    return scale / frames ** 0.5


def quietest_near(paths: list[str], kind: str, exposure: float | None) -> tuple[str, float | None]:
    """Of `paths`, nearest first, the first quiet enough, or the quietest."""
    noises = [(path, master_noise(path, kind, exposure)) for path in paths]
    for path, noise in noises:
        if noise is not None and noise <= MASTER_NOISE:
            return path, noise
    known = [(noise, index) for index, (_, noise) in enumerate(noises) if noise is not None]
    if not known:
        return noises[0]
    return noises[min(known)[1]]


def signature(row: Any) -> seiza.FrameSignature:
    """A frame's signature, rebuilt from its manifest row."""
    return seiza.FrameSignature(
        camera=row["camera"],
        telescope=row["header_telescope"],
        width=row["width"],
        height=row["height"],
        channels=row["channels"],
        binning_x=row["binning_x"],
        binning_y=row["binning_y"],
        gain=row["gain"],
        offset=row["offset"],
        readout_mode=row["readout_mode"],
        bayer_pattern=row["bayer_pattern"],
        filter=row["filter"],
        focal_length_mm=row["focal_length_mm"],
        rotation_deg=row["rotation_deg"],
        exposure_seconds=row["exposure_seconds"],
        camera_temp_c=row["camera_temp_c"],
        captured_at_unix=row["captured_at_unix"],
    )


def record(row: Any, role: str) -> seiza.CalibrationRecord:
    return seiza.CalibrationRecord(row["path"], signature(row), role=role)


def plan_set(lights: list[Any], masters: dict[str, list[Any]]) -> dict[str, Any]:
    """The masters for one set's `lights`, and why the others were left out."""
    references = [record(row, "light") for row in lights]
    # The middle light stands for the set; every light must suit the plan.
    reference = references[len(references) // 2]
    chosen: dict[str, str | None] = {}
    plans: dict[str, Any] = {}
    for kind in KINDS:
        candidates = [
            record(row, kind)
            for row in masters.get(kind, [])
            if row["camera"] == lights[0]["camera"]
        ]
        if not candidates:
            chosen[kind] = None
            plans[kind] = {"ready": False, "candidates": 0, "excluded": {}}
            continue
        plan = seiza.plan_calibration(
            kind,
            reference,
            candidates,
            references=references,
            minimum=1,
            bias_available=chosen.get("bias") is not None,
        )
        plans[kind] = {
            "ready": plan.ready,
            "candidates": len(candidates),
            "matched": len(plan.matched_paths),
            "excluded": dict(Counter(reason for _, reason in plan.excluded)),
        }
        if not plan.ready:
            chosen[kind] = None
        elif kind == "flat":
            chosen[kind] = str(plan.selected_paths[0])
        else:
            matched = [str(path) for path in plan.matched_paths]
            chosen[kind], noise = quietest_near(matched, kind, lights[0]["exposure_seconds"])
            plans[kind]["noise"] = round(noise, 4) if noise is not None else None
            nearest = master_noise(matched[0], kind, lights[0]["exposure_seconds"])
            plans[kind]["nearest_noise"] = round(nearest, 4) if nearest is not None else None
    if chosen["flat"] is None:
        status = "excluded:no-flat"
    elif chosen["dark"] is None and chosen["bias"] is None:
        status = "excluded:no-dark-or-bias"
    else:
        status = "ready"
    return {"status": status, "masters": chosen, "plans": plans}


def calibrate(manifest: Manifest, *, only: set[str] | None = None) -> dict[str, Any]:
    """Plan calibration for every set, or the sets `only` names."""
    masters: dict[str, list[Any]] = {}
    for row in manifest.rows(
        "SELECT * FROM frames WHERE origin = 'library' AND is_master = 1 AND error IS NULL"
    ):
        masters.setdefault(row["role"], []).append(row)
    statuses: Counter[str] = Counter()
    for set_row in manifest.rows("SELECT id FROM sets ORDER BY name"):
        if only is not None and set_row["id"] not in only:
            continue
        lights = manifest.set_frames(set_row["id"])
        result = plan_set(lights, masters)
        statuses[result["status"]] += 1
        with manifest.transaction():
            manifest.update(
                "sets",
                "id",
                set_row["id"],
                {"calibration": dumps(result), "calibration_status": result["status"]},
            )
    return {"masters": {kind: len(rows) for kind, rows in masters.items()}, "sets": dict(statuses)}
