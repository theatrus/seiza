"""Pass 5: calibrate and register each set once, then build its products.

Every accepted frame is calibrated, registered and resampled onto one
reference, with the options a user's stack has by default, and kept in a
scratch directory on local disk. Every product is then integrated from
those same frames by `LiveStacker.reintegrate(frames=...)`, so all of a
set's stacks share one grid and one normalization, and registration runs
once per set rather than once per stack. The scratch frames are deleted
when the set is done.

Products:

- the deep stack, of every admitted frame;
- noise pairs: an A stack of 4, 8, 16 or 32 frames (at most a fifth of the
  set), drawn one from each stretch of the night so that seeing, gradients
  and airmass match, and a B stack of every other frame; three draws each;
- a seeing split: the best and the worst quarter of the frames by measured
  FWHM, at matched counts.
"""

from __future__ import annotations

import random
import shutil
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

import seiza

from .manifest import Manifest, dumps, loads, now
from .measure import accepted_frames
from .store import Store, robust_sigma, save_stack, stack_stats, versions

PAIR_SIZES = (4, 8, 16, 32)
# A frame the stacker had to brighten by this much more than the set's
# median gain lost that much of its light to cloud or haze; it stays out of
# every product.
DIMMED_GAIN = 1.5
PAIR_DRAWS = 3
# A pair whose halves' mean FWHM differs by more than this fraction, or
# whose backgrounds differ by more than this many sigma of their
# difference, is kept but marked rejected.
PAIR_FWHM_TOLERANCE = 0.10
PAIR_BACKGROUND_TOLERANCE = 0.25
# The scale that divides the noise model's high and low bands, in pixels.
BAND_SCALE_PX = 5


def stack_options(defects: Any | None) -> seiza.StackOptions:
    """The options a user's stack has by default (the CLI's)."""
    return seiza.StackOptions(
        normalization="local-background",
        registration_model="quadratic",
        weighting="inverse-noise-variance",
        interpolation="lanczos3",
        defects=defects,
    )


def transparency_outliers(gains: dict[str, float]) -> dict[str, float]:
    """The frames whose photometric gain, the factor normalization scaled
    them by to match the reference, exceeds the median's by `DIMMED_GAIN`."""
    if not gains:
        return {}
    median = statistics.median(gains.values())
    return {path: gain for path, gain in gains.items() if gain > DIMMED_GAIN * median}


def interleaved(ordered: list[str], size: int, rng: random.Random) -> list[str]:
    """`size` of the time-ordered frames, one drawn from each of `size`
    equal stretches of the night."""
    count = len(ordered)
    return [ordered[rng.randrange(k * count // size, (k + 1) * count // size)] for k in range(size)]


def pair_draws(ordered: list[str], set_id: str) -> list[dict[str, Any]]:
    """Every noise pair a set of `ordered` frames yields."""
    pairs = []
    for size in PAIR_SIZES:
        if size > len(ordered) // 5:
            continue
        for draw in range(PAIR_DRAWS):
            seed = f"{set_id}/{size}/{draw}"
            chosen = interleaved(ordered, size, random.Random(seed))
            taken = set(chosen)
            rest = [path for path in ordered if path not in taken]
            pairs.append({"size": size, "draw": draw, "seed": seed, "a": chosen, "b": rest})
    return pairs


def seeing_split(fwhm: dict[str, float]) -> tuple[list[str], list[str]]:
    """The best and the worst quarter of the frames by FWHM."""
    ordered = sorted(fwhm, key=lambda path: (fwhm[path], path))
    quarter = len(ordered) // 4
    return ordered[:quarter], ordered[-quarter:]


def mean_fwhm(paths: list[str], fwhm: dict[str, float]) -> float:
    return statistics.fmean(fwhm[path] for path in paths)


def binned(image: np.ndarray, scale: int) -> np.ndarray:
    """`image` averaged over `scale` × `scale` blocks, ignoring NaN."""
    height = image.shape[0] // scale * scale
    width = image.shape[1] // scale * scale
    blocks = image[:height, :width].reshape(height // scale, scale, width // scale, scale)
    with warnings.catch_warnings():
        # A block wholly outside both halves' coverage is NaN.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(blocks, axis=(1, 3))


def pair_check(a: Any, b: Any, fwhm_a: float, fwhm_b: float) -> tuple[str, str | None, dict[str, Any]]:
    """Whether a pair's halves saw the same sky alike."""
    both = (a.coverage > 0) & (b.coverage > 0)
    difference = np.where(both, a.image - b.image, np.nan).astype(np.float32)
    finite = difference[np.isfinite(difference)][::7]
    offset = float(np.median(finite)) if finite.size else float("nan")
    sigma = robust_sigma(difference)
    with np.errstate(divide="ignore", invalid="ignore"):
        predicted_a = np.nanmedian(np.where(both, a.variance / a.coverage, np.nan)[::5, ::5])
        predicted_b = np.nanmedian(np.where(both, b.variance / b.coverage, np.nan)[::5, ::5])
    stats = {
        "fwhm_a_px": fwhm_a,
        "fwhm_b_px": fwhm_b,
        "background_offset": offset,
        "difference_sigma": sigma,
        "predicted_difference_sigma": float(np.sqrt(predicted_a + predicted_b)),
        # Neighbouring differences see only the grain. Binned by the noise
        # model's band scale, white noise falls by the scale and correlated
        # noise by less, so the two together measure the low band.
        "binned_scale_px": BAND_SCALE_PX,
        "difference_sigma_binned": robust_sigma(binned(difference, BAND_SCALE_PX)),
    }
    if abs(fwhm_a - fwhm_b) > PAIR_FWHM_TOLERANCE * fwhm_b:
        return "rejected", "seeing-mismatch", stats
    if sigma is None or abs(offset) > PAIR_BACKGROUND_TOLERANCE * sigma:
        return "rejected", "background-mismatch", stats
    return "built", None, stats


def record_product(
    manifest: Manifest,
    store: Store,
    set_id: str,
    product_id: str,
    kind: str,
    recipe: dict[str, Any],
    frames: list[str],
    directory: Path,
    files: dict[str, str],
    stats: dict[str, Any],
    run_versions: dict[str, Any],
    status: str = "built",
    reason: str | None = None,
) -> None:
    with manifest.transaction():
        manifest.upsert(
            "products",
            {
                "id": product_id,
                "set_id": set_id,
                "kind": kind,
                "recipe": dumps(recipe),
                "frames": dumps(frames),
                "files": dumps({name: store.relative(directory / file) for name, file in files.items()}),
                "stats": dumps(stats),
                "versions": dumps(run_versions),
                "created": now(),
                "status": status,
                "reason": reason,
            },
            key="id",
        )


def register(
    paths: list[str], reference: str, options: Any, masters: dict[str, Any], scratch: Path
) -> tuple[Any, list[Any]]:
    """Calibrate and register `paths` onto `reference`, keeping each admitted
    frame in a fresh `scratch` directory. Returns the stacker and each
    pushed frame's disposition."""
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    stacker = seiza.LiveStacker(
        reference,
        options=options,
        bias=masters.get("bias"),
        dark=masters.get("dark"),
        flat=masters.get("flat"),
    )
    stacker.retain_frames_for_reintegration(scratch_directory=scratch)
    dispositions, _ = stacker.push_fits_pipelined([path for path in paths if path != reference])
    return stacker, dispositions


def rotated_away(dispositions: list[Any]) -> list[str]:
    """The frames registration turned away for turning beyond its limit.
    A disposition gives only its reason's text, so that is what is read."""
    return [
        str(disposition.source)
        for disposition in dispositions
        if not disposition.accepted and "rotation deviation" in (disposition.reason or "")
    ]


def estimated_bytes(set_row: Any, frames: int, pairs: int, seeing: bool) -> tuple[int, int]:
    """Disk a set's build needs: in scratch, its registered frames; in the
    store, its products' planes (image and variance in float, counts in 16
    bits)."""
    samples = (set_row["width"] or 0) * (set_row["height"] or 0) * (set_row["channels"] or 1)
    stacks = 1 + 2 * pairs + (2 if seeing else 0)
    return samples * 4 * frames * 17 // 16, samples * stacks * 12


def build_set(
    manifest: Manifest, store: Store, set_row: Any, *, keep_scratch: bool = False, pairs_only: bool = False
) -> dict[str, Any]:
    """Build one set's products, or with `pairs_only` only its noise pairs,
    for a set built before it qualified for them: its deep stack and
    seeing split, which other products are made from, stay as they are."""
    set_id = set_row["id"]
    started = time.monotonic()
    calibration = loads(set_row["calibration"], {})
    masters = calibration.get("masters", {})
    recipes = set(loads(set_row["recipes"], []))
    measured = accepted_frames(manifest, set_id)
    paths = [row["path"] for row in measured]
    fwhm = {row["path"]: row["fwhm_px"] for row in measured}
    if len(paths) < 2:
        raise RuntimeError("fewer than two accepted frames")

    pair_count = sum(
        PAIR_DRAWS for size in PAIR_SIZES if size <= len(paths) // 5
    ) if "noise-pairs" in recipes else 0
    store.check_space(*estimated_bytes(set_row, len(paths), pair_count, "seeing-split" in recipes))

    defects = None
    if masters.get("dark") is None:
        # As the CLI's --defects auto does without a dark.
        defects = seiza.detect_defects(paths, bias=masters.get("bias"), flat=masters.get("flat"))
    options = stack_options(defects)
    reference = str(seiza.choose_reference(paths).reference_path)
    scratch = store.scratch / set_id
    directory = store.set_directory(set_id)
    summary: dict[str, Any] = {}
    try:
        stacker, dispositions = register(paths, reference, options, masters, scratch)
        rotated = rotated_away(dispositions)
        if len(rotated) > 1 + sum(disposition.accepted for disposition in dispositions):
            # The camera turned during the night, and the reference fell in
            # the smaller of its orientations: register on the larger one.
            del stacker
            summary["reference_retried"] = {"first": reference, "rotated_away": len(rotated)}
            reference = str(seiza.choose_reference(rotated).reference_path)
            stacker, dispositions = register(paths, reference, options, masters, scratch)
        summary["reference"] = reference
        run_versions = {
            **versions(),
            "options": {
                "normalization": options.normalization,
                "registration_model": options.registration_model,
                "weighting": options.weighting,
                "interpolation": options.interpolation,
                "defects": defects is not None,
            },
            "masters": masters,
            "reference": reference,
        }
        gains = {reference: 1.0} if stacker.reference_integrated else {}
        gains.update(
            (str(disposition.source), disposition.normalization_mean_gain or 1.0)
            for disposition in dispositions
            if disposition.accepted
        )
        turned_away = {
            str(disposition.source): disposition.reason
            for disposition in dispositions
            if not disposition.accepted
        }
        dimmed = transparency_outliers(gains)
        admitted = [path for path in gains if path not in dimmed]
        summary["admitted"] = len(admitted)
        summary["turned_away"] = turned_away
        summary["dimmed"] = dimmed
        registered = time.monotonic()
        if len(admitted) < 2:
            raise RuntimeError(f"{len(admitted)} frames registered and clear")

        if not pairs_only:
            deep = stacker.reintegrate(frames=admitted)
            files = save_stack(directory / "deep", deep, fits=True)
            record_product(
                manifest, store, set_id, f"{set_id}/deep", "deep", {"recipe": "deep"},
                admitted, directory / "deep", files,
                {**stack_stats(deep), "turned_away": turned_away, "dimmed": dimmed}, run_versions,
            )
            del deep

        # Time order, for drawing interleaved subsets.
        captured = {row["path"]: row["captured_at_unix"] or 0 for row in measured}
        ordered = sorted(admitted, key=lambda path: (captured.get(path, 0), path))
        built_pairs = rejected_pairs = 0
        if "noise-pairs" in recipes:
            for pair in pair_draws(ordered, set_id):
                name = f"n{pair['size']:02d}-d{pair['draw']}"
                a = stacker.reintegrate(frames=pair["a"])
                b = stacker.reintegrate(frames=pair["b"])
                status, reason, check = pair_check(
                    a, b, mean_fwhm(pair["a"], fwhm), mean_fwhm(pair["b"], fwhm)
                )
                recipe = {
                    "recipe": "noise-pair",
                    "pair": f"{set_id}/noise/{name}",
                    "size": pair["size"],
                    "draw": pair["draw"],
                    "seed": pair["seed"],
                    "check": check,
                }
                for half, snapshot, frames in (("a", a, pair["a"]), ("b", b, pair["b"])):
                    half_directory = directory / "noise" / name / half
                    files = save_stack(half_directory, snapshot, fits=False)
                    record_product(
                        manifest, store, set_id, f"{set_id}/noise/{name}/{half}", f"noise-{half}",
                        recipe, frames, half_directory, files, stack_stats(snapshot), run_versions,
                        status, reason,
                    )
                built_pairs += status == "built"
                rejected_pairs += status != "built"
                del a, b
        summary["pairs"] = {"built": built_pairs, "rejected": rejected_pairs}

        if "seeing-split" in recipes and not pairs_only:
            in_stack = {path: fwhm[path] for path in admitted if fwhm.get(path) is not None}
            best, worst = seeing_split(in_stack)
            recipe = {
                "recipe": "seeing-split",
                "frames_each": len(best),
                "best_fwhm_px": mean_fwhm(best, in_stack),
                "worst_fwhm_px": mean_fwhm(worst, in_stack),
            }
            for half, frames in (("best", best), ("worst", worst)):
                snapshot = stacker.reintegrate(frames=frames)
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
                record_product(
                    manifest, store, set_id, f"{set_id}/seeing/{half}", f"seeing-{half}",
                    recipe, frames, half_directory, files, stats, run_versions,
                )
                del snapshot
            summary["seeing"] = recipe
        summary["seconds"] = {
            "register": round(registered - started, 1),
            "total": round(time.monotonic() - started, 1),
        }
        # Dropping the stacker deletes the frames it kept.
        del stacker
    finally:
        if not keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
    return summary


def build(
    manifest: Manifest,
    store: Store,
    *,
    only: set[str] | None = None,
    again: bool = False,
    limit: int | None = None,
    keep_scratch: bool = False,
    pairs_only: bool = False,
) -> dict[str, Any]:
    """Build the products of every measured set with a split, or the sets
    `only` names. With `pairs_only`, instead the noise pairs of each built
    set that qualifies for them and has none (see `build_set`)."""
    rows = manifest.rows(
        "SELECT * FROM sets WHERE calibration_status = 'ready' AND measurement_status = 'measured' "
        "AND split IS NOT NULL ORDER BY name"
    )
    built = failed = 0
    for set_row in rows:
        if only is not None and set_row["id"] not in only:
            continue
        if pairs_only:
            if set_row["build_status"] != "built" or "noise-pairs" not in loads(set_row["recipes"], []):
                continue
            if manifest.rows("SELECT 1 FROM products WHERE set_id = ? AND kind = 'noise-a' LIMIT 1",
                             (set_row["id"],)):
                continue
        elif set_row["build_status"] == "built" and not again:
            continue
        if limit is not None and built + failed >= limit:
            break
        if "deep" not in loads(set_row["recipes"], []):
            continue
        if build_recorded(manifest, store, set_row, keep_scratch=keep_scratch, pairs_only=pairs_only) is None:
            failed += 1
        else:
            built += 1
    return {"built": built, "failed": failed}


def build_recorded(
    manifest: Manifest, store: Store, set_row: Any, *, keep_scratch: bool = False, pairs_only: bool = False
) -> dict[str, Any] | None:
    """Build one set, recording a failure rather than raising it. Building
    `pairs_only` adds to the set's build record, and a failure leaves the
    set built, as its other products are."""
    print(f"build: {set_row['name']} ({set_row['split']})", file=sys.stderr)
    earlier = loads(set_row["build_reason"], {}) if pairs_only else {}
    try:
        summary = build_set(manifest, store, set_row, keep_scratch=keep_scratch, pairs_only=pairs_only)
    except Exception as error:  # noqa: BLE001 - one bad set must not stop the pass
        with manifest.transaction():
            manifest.update(
                "sets", "id", set_row["id"],
                {"build_reason": dumps({**earlier, "pairs_later": {"error": str(error)}})} if pairs_only
                else {"build_status": "failed", "build_reason": str(error)},
            )
        print(f"build: {set_row['name']}: {error}", file=sys.stderr)
        return None
    with manifest.transaction():
        manifest.update(
            "sets", "id", set_row["id"],
            {"build_status": "built",
             "build_reason": dumps({**earlier, "pairs_later": summary} if pairs_only else summary)},
        )
    print(
        f"build: {set_row['name']}: {summary['admitted']} frames, pairs {summary['pairs']}, "
        f"{summary['seconds']}",
        file=sys.stderr,
    )
    return summary


def process(
    manifest: Manifest,
    store: Store,
    *,
    only: set[str] | None = None,
    limit: int | None = None,
    catalog_dir: str | None = None,
    keep_scratch: bool = False,
) -> dict[str, Any]:
    """Measure, then build, each calibratable set with a split, one set at
    a time, so the build reads the frames the measurement just read from the
    page cache rather than from the network again."""
    from .measure import measure_recorded

    rows = manifest.rows(
        "SELECT * FROM sets WHERE calibration_status = 'ready' AND split IS NOT NULL "
        "AND (build_status IS NULL OR build_status != 'built') ORDER BY name"
    )
    counts = {"measured": 0, "built": 0, "failed": 0, "skipped": 0}
    for set_row in rows:
        if only is not None and set_row["id"] not in only:
            continue
        if limit is not None and counts["built"] + counts["failed"] >= limit:
            break
        if set_row["measurement_status"] != "measured":
            if measure_recorded(manifest, set_row, catalog_dir) is None:
                counts["failed"] += 1
                continue
            counts["measured"] += 1
            set_row = manifest.rows("SELECT * FROM sets WHERE id = ?", (set_row["id"],))[0]
        if "deep" not in loads(set_row["recipes"], []):
            counts["skipped"] += 1
            continue
        if build_recorded(manifest, store, set_row, keep_scratch=keep_scratch) is None:
            counts["failed"] += 1
        else:
            counts["built"] += 1
    return counts
