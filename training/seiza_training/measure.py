"""Pass 3: measure every frame of the sets that can be calibrated.

`analyze_project` measures each frame's stars, background and noise without
stacking. A frame that lost most of its stars to cloud, whose stars are
elongated well beyond the set's, or that could not be measured is flagged
and kept out of the set's accepted frames. The measurements then decide
which recipes a set qualifies for. One frame per set is plate solved, for
its scale and WCS.
"""

from __future__ import annotations

import statistics
import sys
from collections import Counter
from typing import Any

import seiza

from .calibrate import CALIBRATABLE_SQL
from .manifest import Manifest, dumps, loads

# A frame with fewer stars than this fraction of the set's median lost them
# to cloud. Worse seeing also costs stars, so only a collapse counts here;
# the build pass judges transparency by each frame's photometric gain.
CLOUD_FRACTION = 0.25
# A frame this much more eccentric than the set's median trailed or was
# knocked.
ELONGATION_EXCESS = 0.2
ELONGATION_FLOOR = 0.6

# Recipe requirements (see the design note's "Building it"). A pair's A is at
# most a fifth of the set, so 20 frames give pairs of 4 against the other
# 16, and 40 those of 8 too.
NOISE_PAIR_FRAMES = 20
SEEING_SPLIT_SPREAD = 0.25
SEEING_SPLIT_FRAMES = 32


def quartile_means(values: list[float]) -> tuple[float, float] | None:
    """The mean of the lowest and of the highest quarter of `values`."""
    ordered = sorted(values)
    quarter = len(ordered) // 4
    if quarter == 0:
        return None
    return statistics.fmean(ordered[:quarter]), statistics.fmean(ordered[-quarter:])


def judge(frames: list[dict[str, Any]]) -> None:
    """Flag each measured frame of one set against the set's medians."""
    measured = [frame for frame in frames if frame["status"] == "measured"]
    if not measured:
        return
    stars = statistics.median(frame["stars"] or 0 for frame in measured)
    eccentricities = [frame["eccentricity"] for frame in measured if frame["eccentricity"] is not None]
    eccentricity = statistics.median(eccentricities) if eccentricities else None
    for frame in frames:
        flags = frame["flags"]
        if frame["status"] != "measured":
            flags.append("measurement-failed")
            continue
        if (frame["stars"] or 0) < CLOUD_FRACTION * stars:
            flags.append("cloud")
        if (
            eccentricity is not None
            and frame["eccentricity"] is not None
            and frame["eccentricity"] > max(ELONGATION_FLOOR, eccentricity + ELONGATION_EXCESS)
        ):
            flags.append("elongated")
        if frame["fwhm_px"] is None:
            flags.append("no-fwhm")


def summarize(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """The set's accepted frames, their spread of seeing, and the recipes
    they qualify for."""
    accepted = [frame for frame in frames if not frame["flags"]]
    fwhm = [frame["fwhm_px"] for frame in accepted]
    summary: dict[str, Any] = {
        "measured": sum(frame["status"] == "measured" for frame in frames),
        "accepted": len(accepted),
        "flagged": dict(Counter(flag for frame in frames for flag in frame["flags"])),
    }
    if fwhm:
        summary["median_fwhm_px"] = statistics.median(fwhm)
        summary["median_noise"] = statistics.median(
            frame["noise"] for frame in accepted if frame["noise"] is not None
        ) if any(frame["noise"] is not None for frame in accepted) else None
    quartiles = quartile_means(fwhm)
    if quartiles is not None:
        best, worst = quartiles
        summary["best_quartile_fwhm_px"] = best
        summary["worst_quartile_fwhm_px"] = worst
        summary["seeing_spread"] = (worst - best) / best if best > 0 else None
    summary["recipes"] = recipes_for(summary)
    return summary


def recipes_for(summary: dict[str, Any]) -> list[str]:
    """The recipes a set's measurement summary qualifies it for."""
    recipes = []
    if summary["accepted"] >= NOISE_PAIR_FRAMES:
        recipes.append("noise-pairs")
    if (
        summary["accepted"] >= SEEING_SPLIT_FRAMES
        and summary.get("seeing_spread") is not None
        and summary["seeing_spread"] >= SEEING_SPLIT_SPREAD
    ):
        recipes.append("seeing-split")
    if summary["accepted"]:
        recipes.append("deep")
    return recipes


def refresh_recipes(manifest: Manifest, *, dry_run: bool = False) -> dict[str, Any]:
    """Decide every measured set's recipes again from its recorded summary,
    as `recipes_for` decides them now, without measuring its frames again:
    after a recipe's requirement changes. Returns how many sets changed
    and the recipes they gained and lost; with `dry_run`, records
    nothing."""
    gained: Counter[str] = Counter()
    lost: Counter[str] = Counter()
    changed = 0
    rows = manifest.rows("SELECT id, measurement, recipes FROM sets WHERE measurement_status = 'measured'")
    with manifest.transaction():
        for row in rows:
            summary = loads(row["measurement"], {})
            old, new = loads(row["recipes"], []), recipes_for(summary)
            if set(new) == set(old):
                continue
            changed += 1
            gained.update(set(new) - set(old))
            lost.update(set(old) - set(new))
            if not dry_run:
                summary["recipes"] = new
                manifest.update("sets", "id", row["id"], {"measurement": dumps(summary), "recipes": dumps(new)})
    return {"changed": changed, "gained": dict(gained), "lost": dict(lost), "dry_run": dry_run}


def solve_one(frames: list[dict[str, Any]], catalog_dir: str | None) -> dict[str, Any]:
    """Plate solve the accepted frame of median FWHM."""
    accepted = sorted(
        (frame for frame in frames if not frame["flags"]), key=lambda frame: frame["fwhm_px"]
    )
    if not accepted:
        return {"status": "no-accepted-frame"}
    frame = accepted[len(accepted) // 2]
    try:
        solution = seiza.solve_image(frame["path"], catalog_dir=catalog_dir, time_limit=60.0)
    except Exception as error:  # noqa: BLE001 - any refusal is recorded, never fatal
        return {"status": "unsolved", "path": frame["path"], "error": str(error)}
    return {
        "status": "solved",
        "path": frame["path"],
        "ra_deg": solution.ra,
        "dec_deg": solution.dec,
        "scale_arcsec_px": solution.scale_arcsec_px,
        "rotation_deg": solution.rotation_deg,
        "flipped": solution.flipped,
        "matched_stars": solution.matched_stars,
        "rms_arcsec": solution.rms_arcsec,
        "wcs": solution.fits_header_cards(),
    }


def measure_set(manifest: Manifest, set_row: Any, catalog_dir: str | None) -> dict[str, Any]:
    """Measure one set's lights and record the result."""
    lights = manifest.set_frames(set_row["id"])
    report = seiza.analyze_project([row["path"] for row in lights], metrics_only=True)
    by_path = {record["path"]: record for record in report["frames"]}
    frames = []
    for row in lights:
        record = by_path.get(row["path"])
        if record is None:
            frames.append({"path": row["path"], "status": "missing", "flags": [], "detail": None,
                           "fwhm_px": None, "hfr_px": None, "eccentricity": None, "stars": None,
                           "background": None, "noise": None, "star_snr": None})
            continue
        frames.append(
            {
                "path": row["path"],
                "status": record["measurement_status"],
                "fwhm_px": record.get("median_fwhm"),
                "hfr_px": record.get("median_hfr"),
                "eccentricity": record.get("median_eccentricity"),
                "stars": record.get("star_count"),
                "background": record.get("background"),
                "noise": record.get("noise"),
                "star_snr": record.get("median_star_snr"),
                "flags": [],
                "detail": {"error": record.get("error"), "warnings": record.get("warnings", [])},
            }
        )
    judge(frames)
    summary = summarize(frames)
    summary["solve"] = solve_one(frames, catalog_dir)
    with manifest.transaction():
        manifest.db.execute("DELETE FROM measurements WHERE set_id = ?", (set_row["id"],))
        for frame in frames:
            manifest.upsert(
                "measurements",
                {
                    "path": frame["path"],
                    "set_id": set_row["id"],
                    "status": frame["status"],
                    "fwhm_px": frame["fwhm_px"],
                    "hfr_px": frame["hfr_px"],
                    "eccentricity": frame["eccentricity"],
                    "stars": frame["stars"],
                    "background": frame["background"],
                    "noise": frame["noise"],
                    "star_snr": frame["star_snr"],
                    "flags": dumps(frame["flags"]),
                    "detail": dumps(frame["detail"]),
                },
            )
        manifest.update(
            "sets",
            "id",
            set_row["id"],
            {
                "measurement": dumps(summary),
                "measurement_status": "measured",
                "recipes": dumps(summary["recipes"]),
            },
        )
    return summary


def measure(
    manifest: Manifest,
    *,
    only: set[str] | None = None,
    again: bool = False,
    catalog_dir: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Measure the calibratable sets not measured yet (all of them with
    `again`), or the sets `only` names."""
    rows = manifest.rows(
        f"SELECT * FROM sets WHERE {CALIBRATABLE_SQL} ORDER BY name"
    )
    done = 0
    outcomes: Counter[str] = Counter()
    for set_row in rows:
        if only is not None and set_row["id"] not in only:
            continue
        if set_row["measurement_status"] == "measured" and not again:
            continue
        if limit is not None and done >= limit:
            break
        summary = measure_recorded(manifest, set_row, catalog_dir)
        if summary is None:
            outcomes["failed"] += 1
            continue
        done += 1
        for recipe in summary["recipes"]:
            outcomes[recipe] += 1
    return {"measured": done, "outcomes": dict(outcomes)}


def measure_recorded(manifest: Manifest, set_row: Any, catalog_dir: str | None) -> dict[str, Any] | None:
    """Measure one set, recording a failure rather than raising it."""
    try:
        summary = measure_set(manifest, set_row, catalog_dir)
    except Exception as error:  # noqa: BLE001 - one bad set must not stop the pass
        with manifest.transaction():
            manifest.update(
                "sets", "id", set_row["id"],
                {"measurement_status": "failed", "measurement": dumps({"error": str(error)})},
            )
        print(f"measure: {set_row['name']}: {error}", file=sys.stderr)
        return None
    print(
        f"measure: {set_row['name']}: {summary['accepted']} of {set_row['frames']} accepted, "
        f"recipes {', '.join(summary['recipes']) or 'none'}, solve {summary['solve']['status']}",
        file=sys.stderr,
    )
    return summary


def accepted_frames(manifest: Manifest, set_id: str) -> list[Any]:
    """A set's accepted measured frames, in capture order."""
    return [
        row
        for row in manifest.rows(
            "SELECT m.*, f.captured_at_unix FROM measurements m JOIN frames f ON f.path = m.path "
            "WHERE m.set_id = ? ORDER BY f.captured_at_unix, m.path",
            (set_id,),
        )
        if not loads(row["flags"], [])
    ]
