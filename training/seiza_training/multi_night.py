"""Multi-night deep stacks: every accepted frame of one object, through one
filter on one telescope, over all its measured nights, stacked onto one
reference with the variance and coverage every product carries, and the
two halves of it that share no night.

A finished stack copied in from PSF Guard holds only its image, so the
noise model cannot be run on it with a noise map of its own, nor split.
These stacks are the corpus's own. One stacker registers every night's
frames onto the deepest night's reference and keeps them for
reintegration, calibrating each night with that night's masters
(`LiveStacker.set_calibration_fits` switches between them). Every product
is then reintegrated from those frames, so all share one grid and one
normalization:

- `multi-night/{key}/stack` (kind `multi-night`), every admitted frame: a
  reference far deeper than any night's, for the reference night's
  products, which share its grid;
- `multi-night/{key}/half/a` and `/b` (kinds `multi-night-half-a` and
  `-b`), the nights dealt between them so their frame counts come out
  even: a noise pair at a depth no night reaches, whose noise is
  independent since the halves share no frame;
- `multi-night/{key}/seeing/best` and `/worst` (kinds
  `multi-night-seeing-best` and `-worst`), the best and the worst quarter
  of every night's frames by measured FWHM, at matched counts: PSF pairs
  deeper, and further apart in seeing, than a night's split.

An object is its sets' name less the night,
`telescope/target/FILTER-exposure-gain-binning`; `key` is a hash of it.
The products are recorded under the reference night's set, so they take
its split; every night of an object shares its sky group, and so its
split.

`source_map` rewrites the start of frame and master paths, for a machine
that mounts the corpus elsewhere; the products record the manifest's paths.
"""

from __future__ import annotations

import hashlib
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import seiza

from .build import mean_fwhm, record_product, seeing_split, stack_options, transparency_outliers
from .manifest import Manifest, loads
from .measure import accepted_frames
from .store import Store, save_stack, stack_stats, versions

# Objects with less than this many hours of accepted frames are left alone
# unless named.
MINIMUM_HOURS = 10.0


def object_name(set_name: str) -> str:
    """A set's object: its name less the night."""
    parts = set_name.split("/")
    return "/".join(parts[:2] + parts[3:])


def object_key(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:12]


def objects(manifest: Manifest) -> dict[str, list[dict[str, Any]]]:
    """Every object with two or more measured, calibratable nights: its
    sets, in night order."""
    found: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest.rows(
        "SELECT * FROM sets WHERE calibration_status = 'ready' AND measurement_status = 'measured' "
        "AND split IS NOT NULL ORDER BY name"
    ):
        found[object_name(row["name"])].append(dict(row))
    return {name: sets for name, sets in found.items() if len(sets) > 1}


def survey(manifest: Manifest) -> list[dict[str, Any]]:
    """Each object's nights, accepted frames and hours, deepest first."""
    rows = []
    for name, sets in objects(manifest).items():
        frames = [len(accepted_frames(manifest, row["id"])) for row in sets]
        hours = sum(count * (row["exposure_seconds"] or 0.0) for count, row in zip(frames, sets)) / 3600
        rows.append({"object": name, "key": object_key(name), "split": sets[0]["split"], "nights": len(sets),
                     "frames": sum(frames), "hours": round(hours, 2)})
    return sorted(rows, key=lambda row: -row["hours"])


def deal(counts: dict[str, int]) -> tuple[list[str], list[str]]:
    """Nights into two halves of near-even frame counts: the largest
    first, each to the half with fewer frames so far."""
    halves: tuple[list[str], list[str]] = ([], [])
    totals = [0, 0]
    for night in sorted(counts, key=lambda night: (-counts[night], night)):
        smaller = 0 if totals[0] <= totals[1] else 1
        halves[smaller].append(night)
        totals[smaller] += counts[night]
    return halves


def run(
    manifest: Manifest,
    store: Store,
    *,
    names: set[str] | None = None,
    minimum_hours: float = MINIMUM_HOURS,
    source_map: dict[str, str] | None = None,
    again: bool = False,
    limit: int | None = None,
    keep_scratch: bool = False,
) -> dict[str, Any]:
    """Stack the named objects, or every one with `minimum_hours`, deepest
    first; an object already stacked is left unless `again`."""
    chosen = [row for row in survey(manifest)
              if (row["object"] in names if names is not None else row["hours"] >= minimum_hours)]
    if names is not None and len(chosen) < len(names):
        missing = sorted(names - {row["object"] for row in chosen})
        raise SystemExit(f"no object with two measured nights is named {missing}")
    sets = objects(manifest)
    summary: dict[str, Any] = {"stacked": {}, "skipped": []}
    for row in chosen[:limit]:
        stack_id = f"multi-night/{row['key']}/stack"
        if not again and manifest.rows("SELECT 1 FROM products WHERE id = ? AND status = 'built'", (stack_id,)):
            summary["skipped"].append(row["object"])
            continue
        summary["stacked"][row["object"]] = stack_object(
            manifest, store, row["object"], sets[row["object"]], source_map=source_map or {},
            keep_scratch=keep_scratch,
        )
    return summary


def stack_object(
    manifest: Manifest,
    store: Store,
    name: str,
    sets: list[dict[str, Any]],
    *,
    source_map: dict[str, str],
    keep_scratch: bool = False,
) -> dict[str, Any]:
    """Register every night of one object once, and integrate its stack and
    halves."""
    started = time.monotonic()
    key = object_key(name)

    def local(path: str) -> str:
        for prefix, replacement in source_map.items():
            if path.startswith(prefix):
                return replacement + path[len(prefix):]
        return path

    measured = {row["id"]: accepted_frames(manifest, row["id"]) for row in sets}
    nights = {night: [frame["path"] for frame in frames] for night, frames in measured.items()}
    fwhm = {frame["path"]: frame["fwhm_px"] for frames in measured.values() for frame in frames}
    masters = {row["id"]: loads(row["calibration"], {}).get("masters", {}) for row in sets}
    deepest = max(nights, key=lambda night: (len(nights[night]), night))
    # The deepest night's own reference when it was built, else Seiza's choice.
    built = manifest.rows("SELECT versions FROM products WHERE id = ? AND status = 'built'", (f"{deepest}/deep",))
    reference = loads(built[0]["versions"]).get("reference") if built else None
    if reference is None:
        reference = str(seiza.choose_reference([local(path) for path in nights[deepest]]).reference_path)
        reference = next(path for path in nights[deepest] if local(path) == reference)
    original = {local(path): path for paths in nights.values() for path in paths}

    def calibration(night: str) -> dict[str, str | None]:
        return {kind: local(path) if path else None for kind, path in
                ((kind, masters[night].get(kind)) for kind in ("bias", "dark", "flat"))}

    width, height = sets[0]["width"] or 0, sets[0]["height"] or 0
    frames = sum(len(paths) for paths in nights.values())
    samples = width * height * (sets[0]["channels"] or 1)
    store.check_space(samples * 4 * frames * 17 // 16, samples * 3 * 12)
    defects = None
    if masters[deepest].get("dark") is None:
        # As a night's build does without a dark: the sensor's defects, from
        # the deepest night.
        defects = seiza.detect_defects([local(path) for path in nights[deepest]],
                                       bias=calibration(deepest)["bias"], flat=calibration(deepest)["flat"])
    options = stack_options(defects)
    scratch = store.scratch / "multi-night" / key
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    directory = store.products / "multi-night" / key
    summary: dict[str, Any] = {"key": key, "reference": reference, "nights": {}}
    stacker = None
    try:
        stacker = seiza.LiveStacker(local(reference), options=options, **calibration(deepest))
        stacker.retain_frames_for_reintegration(scratch_directory=scratch)
        gains: dict[str, dict[str, float]] = {night: {} for night in nights}
        if stacker.reference_integrated:
            gains[deepest][reference] = 1.0
        turned_away: dict[str, str | None] = {}
        for night in [deepest, *sorted(night for night in nights if night != deepest)]:
            if night != deepest:
                stacker.set_calibration_fits(**calibration(night))
            dispositions, _ = stacker.push_fits_pipelined(
                [local(path) for path in nights[night] if path != reference])
            for disposition in dispositions:
                path = original[str(disposition.source)]
                if disposition.accepted:
                    gains[night][path] = disposition.normalization_mean_gain or 1.0
                else:
                    turned_away[path] = disposition.reason
        # Haze is judged within a night: nights differ in sky as a whole.
        dimmed = {path: gain for night in nights for path, gain in transparency_outliers(gains[night]).items()}
        admitted = {night: [path for path in nights[night] if path in gains[night] and path not in dimmed]
                    for night in nights}
        counts = {night: len(paths) for night, paths in admitted.items() if paths}
        if len(counts) < 2:
            raise RuntimeError(f"{len(counts)} nights registered and clear")
        run_versions = {
            **versions(),
            "options": {"normalization": options.normalization, "registration_model": options.registration_model,
                        "weighting": options.weighting, "interpolation": options.interpolation,
                        "defects": defects is not None},
            "masters": masters,
            "reference": reference,
        }
        recipe = {"recipe": "multi-night", "object": name, "nights": sorted(counts)}
        everything = [path for night in sorted(counts) for path in admitted[night]]
        stack = stacker.reintegrate(frames=[local(path) for path in everything])
        files = save_stack(directory / "stack", stack, fits=True)
        record_product(manifest, store, deepest, f"multi-night/{key}/stack", "multi-night", recipe, everything,
                       directory / "stack", files,
                       {**stack_stats(stack), "turned_away": turned_away, "dimmed": dimmed}, run_versions)
        del stack
        for half, half_nights in zip("ab", deal(counts)):
            paths = [path for night in sorted(half_nights) for path in admitted[night]]
            snapshot = stacker.reintegrate(frames=[local(path) for path in paths])
            files = save_stack(directory / f"half-{half}", snapshot, fits=False)
            record_product(manifest, store, deepest, f"multi-night/{key}/half/{half}", f"multi-night-half-{half}",
                           {**recipe, "half": half, "nights": sorted(half_nights)}, paths,
                           directory / f"half-{half}", files, stack_stats(snapshot), run_versions)
            summary[f"half_{half}"] = {"nights": len(half_nights), "frames": len(paths)}
            del snapshot
        # The best and worst quarters by seeing over every night, at matched
        # counts: PSF pairs deeper, and further apart, than one night gives.
        in_stack = {path: fwhm[path] for path in everything if fwhm.get(path) is not None}
        best, worst = seeing_split(in_stack)
        seeing = {**recipe, "recipe": "multi-night-seeing", "frames_each": len(best),
                  "best_fwhm_px": mean_fwhm(best, in_stack), "worst_fwhm_px": mean_fwhm(worst, in_stack)}
        for half, frames in (("best", best), ("worst", worst)):
            snapshot = stacker.reintegrate(frames=[local(path) for path in frames])
            half_directory = directory / "seeing" / half
            files = save_stack(half_directory, snapshot, fits=True)
            stats = stack_stats(snapshot)
            try:
                measured_stack = seiza.detect_measured_stars_file(half_directory / files["fits"])
                stats["stack_stars"] = len(measured_stack.stars)
                # With no star found, the average width reads 0.
                stats["stack_fwhm_px"] = measured_stack.average_fwhm if measured_stack.stars else None
            except Exception as error:  # noqa: BLE001 - recorded, not fatal
                stats["stack_fwhm_error"] = str(error)
            record_product(manifest, store, deepest, f"multi-night/{key}/seeing/{half}",
                           f"multi-night-seeing-{half}", {**seeing, "nights": sorted({
                               night for night, paths in admitted.items() if set(paths) & set(frames)})},
                           frames, half_directory, files, stats, run_versions)
            summary[f"seeing_{half}"] = {"frames": len(frames), "fwhm_px": stats.get("stack_fwhm_px")}
            del snapshot
        summary["nights"] = counts
        summary["frames"] = len(everything)
        summary["turned_away"] = len(turned_away)
        summary["dimmed"] = len(dimmed)
    finally:
        # The stacker removes its own frames when freed.
        del stacker
        if not keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
    summary["seconds"] = round(time.monotonic() - started, 1)
    return summary
