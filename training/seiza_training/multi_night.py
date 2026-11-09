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
  of all its nights' frames by measured FWHM, at matched counts: PSF pairs
  deeper, and further apart in seeing, than a night's split. An object
  whose frames' quarters differ by less than `SEEING_SCREEN_SPREAD` gets
  none. Each half's PSF is measured on its stack, and the pair is judged
  by `seeing_check`: one that fails is kept but marked rejected, with the
  reason.

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
import math
import shutil
import statistics
import time
import warnings
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

import seiza

from .build import (
    PAIR_BACKGROUND_TOLERANCE,
    mean_fwhm,
    record_product,
    seeing_split,
    stack_options,
    transparency_outliers,
)
from .manifest import Manifest, dumps, loads
from .measure import SEEING_SPLIT_FRAMES, SEEING_SPLIT_SPREAD, accepted_frames, quartile_means
from .store import Store, plane, robust_sigma, save_stack, stack_stats, versions

# Objects with less than this many hours of accepted frames are left alone
# unless named, or, with `seeing`, unless their nights give a seeing split.
MINIMUM_HOURS = 10.0

# An object's frames are worth a seeing pair when there are a night's
# split's worth of them (`SEEING_SPLIT_FRAMES`) and their worst quarter's
# mean FWHM exceeds their best's by this much. The frames' widths were
# measured smoothed (see `STACK_DETECTION`), which shrinks a narrow star's
# excess width most: the first six multi-night pairs' frames' quarters
# differ by 3.5% to 12% on 1.2–1.5″/px astrographs and by 32% and 48% on
# the C925, and their stacks by 12% to 28%, and 43% and 69%. So the frames
# only screen; the stacks, measured unsmoothed, decide.
SEEING_SCREEN_SPREAD = 0.10

# A seeing pair teaches sharpening only where its halves differ in the PSF
# alone. Whatever moves with seeing (airmass, focus, wind, haze) moves
# something else too, so `seeing_check` measures each half's stack on the
# stars found in both, within `SEEING_MATCH_PX` of each other, and marks a
# pair rejected unless:
# - `SEEING_STARS` stars are measured in both, unsaturated and fitted;
# - the worst half's median FWHM exceeds the best's by as much as a
#   seeing split needs (`SEEING_SPLIT_SPREAD`), and in none of the
#   field's `SEEING_FIELD` × `SEEING_FIELD` tiles with `SEEING_FIELD_STARS`
#   stars is it the sharper: a focus or tilt that changed between nights;
# - its stars' axis ratio, minor over major, is no more than
#   `SEEING_SHAPE_TOLERANCE` below the best half's (wind, tracking). Seeing
#   is a round blur, which makes stars rounder. (Eccentricity moves most
#   where stars are nearly round: 0.22 and 0.32 are axis ratios of 0.98
#   and 0.95.);
# - the halves' flux differs by one gain, within `SEEING_FLUX_TOLERANCE`:
#   the stars', in apertures of `SEEING_APERTURE` times the worst half's
#   FWHM about the `SEEING_FLUX_STARS` brightest with no other within two
#   apertures, and the extended light's, the medians of the
#   `SEEING_BLOCK_PX` blocks the best half puts `SEEING_BRIGHT` times its
#   noise above its sky (a median leaves out the stars). A gain alone is
#   recorded, not rejected, for a trainer to match: it is normalization's,
#   whose photometric gain, measured in a 5-pixel aperture that loses more
#   of a wider star, brightens a frame as its seeing worsens. It brightened
#   one C925 object's worst quarter 1.7 times, and its noise with it;
# - their skies differ by one plane: on the fainter half of the blocks,
#   their medians less a plane fitted to each half's differ, in nine of
#   ten blocks, by no more than a noise pair's backgrounds may
#   (`PAIR_BACKGROUND_TOLERANCE` of the noise of the halves' difference):
#   a glow in one corner, or a gradient no plane follows. The planes'
#   difference, at the centre and at the furthest corner, is recorded, as
#   the gain is, for a trainer to match: normalization leaves a blurred
#   frame's sky a little low, its star wings lifting the medians it
#   matches.
SEEING_MATCH_PX = 2.0
SEEING_STARS = 30
SEEING_FIELD = 3
SEEING_FIELD_STARS = 10
SEEING_SHAPE_TOLERANCE = 0.05
SEEING_FLUX_TOLERANCE = 0.05
SEEING_APERTURE = 3.0
SEEING_FLUX_STARS = 200
SEEING_BLOCK_PX = 64
SEEING_BRIGHT = 3.0
# A median over fewer stars or blocks than this judges nothing.
SEEING_FLUX_LEAST = 10
# The measuring detector smooths an image before it measures its stars,
# as a single frame needs, and that widens narrow stars most: Gaussian
# stars 3.5 and 5.9 pixels wide measure 5.5 and 7.0, so a ratio of 1.67
# reads 1.29. A stack is deep enough to be measured as it is (3.5 and 5.7).
STACK_DETECTION = {"noise_reduction_radius": 0}


def object_name(set_name: str) -> str:
    """A set's object: its name less the night."""
    parts = set_name.split("/")
    return "/".join(parts[:2] + parts[3:])


def object_key(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:12]


def objects(manifest: Manifest) -> dict[str, list[dict[str, Any]]]:
    """Every object with two or more measured nights calibrated with a
    flat: its sets, in night order."""
    found: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest.rows(
        "SELECT * FROM sets WHERE calibration_status = 'ready' AND measurement_status = 'measured' "
        "AND split IS NOT NULL ORDER BY name"
    ):
        found[object_name(row["name"])].append(dict(row))
    return {name: sets for name, sets in found.items() if len(sets) > 1}


def seeing_spread(fwhm: list[float]) -> float | None:
    """How much wider the worst quarter of `fwhm` is than the best, on
    their means, as a night's measurement judges it."""
    quartiles = quartile_means(fwhm)
    if quartiles is None or quartiles[0] <= 0:
        return None
    return (quartiles[1] - quartiles[0]) / quartiles[0]


def screened(frames: int, spread: float | None) -> bool:
    """Whether `frames` frames whose quarters' spread of seeing is
    `spread` are worth a seeing pair (`SEEING_SCREEN_SPREAD`)."""
    return frames >= SEEING_SPLIT_FRAMES and spread is not None and spread >= SEEING_SCREEN_SPREAD


def gives_seeing_split(fwhm: list[float]) -> bool:
    """Whether frames of these FWHMs are worth a seeing pair."""
    return screened(len(fwhm), seeing_spread(fwhm))


def survey(manifest: Manifest) -> list[dict[str, Any]]:
    """Each object's nights, accepted frames and hours, and the spread of
    seeing over all its nights, deepest first."""
    rows = []
    for name, sets in objects(manifest).items():
        frames = [accepted_frames(manifest, row["id"]) for row in sets]
        counts = [len(night) for night in frames]
        hours = sum(count * (row["exposure_seconds"] or 0.0) for count, row in zip(counts, sets)) / 3600
        fwhm = [frame["fwhm_px"] for night in frames for frame in night if frame["fwhm_px"] is not None]
        spread = seeing_spread(fwhm)
        rows.append({"object": name, "key": object_key(name), "split": sets[0]["split"], "nights": len(sets),
                     "frames": sum(counts), "hours": round(hours, 2),
                     "seeing_spread": None if spread is None else round(spread, 3),
                     "seeing_split": gives_seeing_split(fwhm)})
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


def usable(stars: list[Any]) -> list[Any]:
    """The stars measured with a fitted PSF, unsaturated."""
    return [star for star in stars
            if not star.saturated and star.eccentricity is not None and math.isfinite(star.fwhm) and star.fwhm > 0]


def within(stars: list[Any], radius: float) -> Callable[[float, float], list[Any]]:
    """A lookup of the `stars` within `radius` of a point."""
    cells: dict[tuple[int, int], list[Any]] = defaultdict(list)
    for star in stars:
        cells[(math.floor(star.x / radius), math.floor(star.y / radius))].append(star)

    def near(x: float, y: float) -> list[Any]:
        column, row = math.floor(x / radius), math.floor(y / radius)
        return [star for across in (-1, 0, 1) for down in (-1, 0, 1)
                for star in cells.get((column + across, row + down), ())
                if (star.x - x) ** 2 + (star.y - y) ** 2 <= radius**2]

    return near


def matched_stars(best: list[Any], worst: list[Any]) -> list[tuple[Any, Any]]:
    """Each usable star of the best half with the nearest usable star of
    the worst within `SEEING_MATCH_PX`: the same star, on the shared grid."""
    near = within(usable(worst), SEEING_MATCH_PX)
    pairs = []
    for star in usable(best):
        found = near(star.x, star.y)
        if found:
            pairs.append((star, min(found, key=lambda other: (other.x - star.x) ** 2 + (other.y - star.y) ** 2)))
    return pairs


def aperture_flux(image: np.ndarray, covered: np.ndarray, x: float, y: float, radius: float) -> float | None:
    """The flux within `radius` of (`x`, `y`) above the median of the ring
    out to 1.5 `radius`, or None where the ring leaves the image or its
    coverage. Both halves count the same pixels about one position."""
    outer = math.ceil(1.5 * radius)
    column, row = round(x), round(y)
    if min(column, row) < outer or row + outer >= image.shape[0] or column + outer >= image.shape[1]:
        return None
    window = np.s_[row - outer:row + outer + 1, column - outer:column + outer + 1]
    if not covered[window].all():
        return None
    stamp = np.asarray(image[window], dtype=np.float64)
    down, across = np.mgrid[-outer:outer + 1, -outer:outer + 1]
    distance = np.hypot(across + column - x, down + row - y)
    sky = np.median(stamp[(distance > radius) & (distance <= 1.5 * radius)])
    return float((stamp[distance <= radius] - sky).sum())


def blocks(values: np.ndarray, size: int) -> np.ndarray:
    """`values` cut into `size` × `size` blocks, as (rows, columns, size²);
    a part block at the right or bottom edge is left out."""
    rows, columns = values.shape[0] // size, values.shape[1] // size
    return (np.asarray(values[:rows * size, :columns * size])
            .reshape(rows, size, columns, size).swapaxes(1, 2).reshape(rows, columns, size * size))


def sky_planes(medians: list[np.ndarray], faint: np.ndarray) -> list[np.ndarray] | None:
    """Planes (offset, along, down), across the image as 0 to 1, through
    each half's `faint` block medians by least squares, fitted again
    without the blocks either half puts four times its spread (from the
    median absolute deviation) from its plane, as `noise_model.matched`
    fits its backgrounds. None with fewer than three faint blocks."""
    rows, columns = np.nonzero(faint)
    if len(rows) < 3:
        return None
    design = np.stack([np.ones(len(rows)), (columns + 0.5) / faint.shape[1], (rows + 0.5) / faint.shape[0]], axis=1)
    values = [median[rows, columns].astype(np.float64) for median in medians]

    def fit(keep: np.ndarray) -> list[np.ndarray]:
        return [np.linalg.lstsq(design[keep], value[keep], rcond=None)[0] for value in values]

    keep = np.ones(len(rows), dtype=bool)
    for value, coefficients in zip(values, fit(keep)):
        residual = value - design @ coefficients
        spread = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        if spread > 0:
            keep &= np.abs(residual) <= 4.0 * spread
    return fit(keep) if keep.sum() >= 3 else None


def seeing_check(
    best: tuple[np.ndarray, np.ndarray, list[Any]], worst: tuple[np.ndarray, np.ndarray, list[Any]]
) -> tuple[str, str | None, dict[str, Any]]:
    """Whether a seeing pair's halves differ in their PSF alone (see
    `SEEING_STARS`). Each half is its image and coverage, (height, width)
    or (channels, height, width) as a stack's files hold them, of which
    the first channel is judged, and the stars measured on its stack."""
    (image_best, coverage_best, stars_best), (image_worst, coverage_worst, stars_worst) = best, worst
    if image_best.ndim == 3:
        image_best, coverage_best, image_worst, coverage_worst = (
            planes[0] for planes in (image_best, coverage_best, image_worst, coverage_worst))
    pairs = matched_stars(stars_best, stars_worst)
    stats: dict[str, Any] = {"stars": len(pairs)}
    if len(pairs) < SEEING_STARS:
        return "rejected", "psf-unmeasured", stats

    fwhm_best = statistics.median(star.fwhm for star, _ in pairs)
    fwhm_worst = statistics.median(star.fwhm for _, star in pairs)
    height, width = image_best.shape
    tiles: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    for star, other in pairs:
        tile = (min(SEEING_FIELD - 1, int(SEEING_FIELD * star.y / height)),
                min(SEEING_FIELD - 1, int(SEEING_FIELD * star.x / width)))
        tiles[tile].append((star.fwhm, other.fwhm))
    field = [[statistics.median(w for _, w in tiles[tile]) / statistics.median(b for b, _ in tiles[tile])
              if len(tiles[tile]) >= SEEING_FIELD_STARS else None
              for tile in ((row, column) for column in range(SEEING_FIELD))] for row in range(SEEING_FIELD)]
    eccentricity = [statistics.median(star.eccentricity for star in half) for half in zip(*pairs)]
    stats.update({
        "fwhm_best_px": fwhm_best,
        "fwhm_worst_px": fwhm_worst,
        "fwhm_ratio": fwhm_worst / fwhm_best,
        "fwhm_ratio_field": field,
        "eccentricity_best": eccentricity[0],
        "eccentricity_worst": eccentricity[1],
        "axis_ratio_best": math.sqrt(1 - min(1.0, eccentricity[0]) ** 2),
        "axis_ratio_worst": math.sqrt(1 - min(1.0, eccentricity[1]) ** 2),
    })

    covered = (np.asarray(coverage_best) > 0) & (np.asarray(coverage_worst) > 0)
    radius = SEEING_APERTURE * fwhm_worst
    crowd = within([*stars_best, *stars_worst], 2.0 * radius)
    ratios = []
    for star, _ in sorted(pairs, key=lambda pair: -pair[0].flux):
        # The star itself, in either half, is no neighbour.
        if any((other.x - star.x) ** 2 + (other.y - star.y) ** 2 > SEEING_MATCH_PX**2
               for other in crowd(star.x, star.y)):
            continue
        fluxes = [aperture_flux(image, covered, star.x, star.y, radius) for image in (image_best, image_worst)]
        if fluxes[0] is None or fluxes[1] is None or min(fluxes) <= 0:
            continue
        ratios.append(fluxes[0] / fluxes[1])
        if len(ratios) == SEEING_FLUX_STARS:
            break
    stats["flux_stars"] = len(ratios)
    stats["star_flux_ratio"] = float(np.median(ratios)) if len(ratios) >= SEEING_FLUX_LEAST else None

    with np.errstate(invalid="ignore"):
        sigma = robust_sigma(np.where(covered, image_best - image_worst, np.nan).astype(np.float32))
        noise_best = robust_sigma(np.where(covered, image_best, np.nan).astype(np.float32))
    full = blocks(covered, SEEING_BLOCK_PX).all(axis=2)
    with warnings.catch_warnings():
        # A block outside the coverage may hold NaN; it is never used.
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = [np.median(blocks(image, SEEING_BLOCK_PX), axis=2) for image in (image_best, image_worst)]
    stats["difference_sigma"] = sigma
    planes = None
    if full.any():
        faint = full & (medians[0] <= np.median(medians[0][full]))
        planes = sky_planes(medians, faint)
    if planes is not None:
        change = planes[0] - planes[1]
        stats["background_offset"] = float(change[0] + 0.5 * change[1] + 0.5 * change[2])
        # The furthest a corner's difference strays from the centre's.
        stats["background_tilt"] = float(0.5 * (abs(change[1]) + abs(change[2])))
        rows, columns = np.indices(full.shape)
        signal = [median - (c[0] + c[1] * (columns + 0.5) / full.shape[1] + c[2] * (rows + 0.5) / full.shape[0])
                  for median, c in zip(medians, planes)]
        stats["background_residual"] = float(np.percentile(np.abs(signal[0][faint] - signal[1][faint]), 90))
        bright = full & (signal[0] >= SEEING_BRIGHT * (noise_best or np.inf)) & (signal[1] > 0)
        stats["extended_blocks"] = int(bright.sum())
        stats["extended_flux_ratio"] = float(np.median(signal[0][bright] / signal[1][bright])) \
            if bright.sum() >= SEEING_FLUX_LEAST else None
    # The one gain that takes the worst half's flux onto the best's, which
    # a trainer applies before comparing them.
    stats["flux_gain"] = stats["star_flux_ratio"] if stats["star_flux_ratio"] is not None \
        else stats.get("extended_flux_ratio")

    if fwhm_worst - fwhm_best < SEEING_SPLIT_SPREAD * fwhm_best:
        return "rejected", "seeing-too-close", stats
    if any(ratio is not None and ratio < 1 for row in field for ratio in row):
        return "rejected", "psf-field-mismatch", stats
    if stats["axis_ratio_best"] - stats["axis_ratio_worst"] > SEEING_SHAPE_TOLERANCE:
        return "rejected", "psf-shape-mismatch", stats
    if stats["star_flux_ratio"] is not None and stats.get("extended_flux_ratio") is not None and \
            abs(stats["extended_flux_ratio"] / stats["star_flux_ratio"] - 1) > SEEING_FLUX_TOLERANCE:
        return "rejected", "flux-mismatch", stats
    if sigma is None or planes is None or stats["background_residual"] > PAIR_BACKGROUND_TOLERANCE * sigma:
        return "rejected", "background-mismatch", stats
    return "built", None, stats


def counts(image: np.ndarray) -> np.ndarray:
    """A stack's first channel as the 16-bit counts the measuring detector
    reads: linear from its 0.01st percentile to its brightest pixel, NaN
    as its median. A few pixels far below the sky, which an edge of low
    coverage can leave, would otherwise take the whole range: one C925
    stack's 18 pixels below -100, down to -21,809, hid its 448 stars."""
    values = image if image.ndim == 2 else image[0]
    sample = np.asarray(values[::7, ::7])
    sample = sample[np.isfinite(sample)]
    low, high = float(np.percentile(sample, 0.01)), float(np.nanmax(values))
    with np.errstate(invalid="ignore"):
        scaled = (np.where(np.isfinite(values), values, np.median(sample)) - low) * (65535 / max(high - low, 1e-6))
    return np.clip(scaled, 0, 65535).astype(np.uint16)


def stack_psf(image: np.ndarray) -> tuple[list[Any], dict[str, Any]]:
    """The stars measured on a stack, as `counts`, unsmoothed
    (`STACK_DETECTION`), and its PSF's width and shape; no stars, and the
    error, where measuring failed."""
    try:
        found = seiza.detect_measured_stars(counts(image), **STACK_DETECTION)
    except Exception as error:  # noqa: BLE001 - recorded, not fatal
        return [], {"stack_fwhm_error": str(error)}
    fitted = usable(found.stars)
    return found.stars, {
        "stack_stars": len(found.stars),
        # With no star found, the average width reads 0.
        "stack_fwhm_px": found.average_fwhm if found.stars else None,
        "stack_eccentricity": statistics.median(star.eccentricity for star in fitted) if fitted else None,
    }


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
    seeing: bool = False,
) -> dict[str, Any]:
    """Stack the named objects, or every one with `minimum_hours` and, with
    `seeing`, every one whose nights give a seeing split however few its
    hours, deepest first; an object already stacked is left unless
    `again`."""
    chosen = [row for row in survey(manifest)
              if (row["object"] in names if names is not None
                  else row["hours"] >= minimum_hours or (seeing and row["seeing_split"]))]
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
        in_stack = {path: fwhm[path] for path in everything if fwhm.get(path) is not None}
        summary["seeing"] = split_seeing(manifest, store, stacker, key, deepest, recipe, admitted, in_stack, local,
                                         directory / "seeing", run_versions)
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


def split_seeing(
    manifest: Manifest,
    store: Store,
    stacker: Any,
    key: str,
    owner: str,
    recipe: dict[str, Any],
    admitted: dict[str, list[str]],
    fwhm: dict[str, float],
    local: Callable[[str], str],
    directory: Path,
    run_versions: dict[str, Any],
) -> dict[str, Any]:
    """Integrate, measure and judge one object's seeing pair from the
    frames `stacker` keeps: the best and the worst quarter of all its
    nights' frames by FWHM, at matched counts and so at matched depth, an
    object's frames sharing one exposure. An object whose frames do not
    pass the screen (`screened`) gets none, and a pair it was given before
    is marked rejected."""
    ids = {half: f"multi-night/{key}/seeing/{half}" for half in ("best", "worst")}
    spread = seeing_spread(list(fwhm.values()))
    if not gives_seeing_split(list(fwhm.values())):
        with manifest.transaction():
            for product_id in ids.values():
                manifest.update("products", "id", product_id, {"status": "rejected", "reason": "seeing-spread"})
        return {"skipped": "seeing-spread", "frames": len(fwhm), "spread": spread}
    best, worst = seeing_split(fwhm)
    halves = {"best": best, "worst": worst}
    measured: dict[str, tuple[np.ndarray, np.ndarray, list[Any]]] = {}
    files: dict[str, dict[str, str]] = {}
    stats: dict[str, dict[str, Any]] = {}
    for half, frames in halves.items():
        snapshot = stacker.reintegrate(frames=[local(path) for path in frames])
        files[half] = save_stack(directory / half, snapshot, fits=True)
        image = plane(snapshot.image)
        stars, psf = stack_psf(image)
        stats[half] = {**stack_stats(snapshot), **psf, "frame_fwhm_px": [fwhm[path] for path in frames]}
        measured[half] = (image, plane(snapshot.coverage), stars)
        del snapshot
    status, reason, check = seeing_check(measured["best"], measured["worst"])
    del measured
    seeing = {**recipe, "recipe": "multi-night-seeing", "frames_each": len(best),
              "best_fwhm_px": mean_fwhm(best, fwhm), "worst_fwhm_px": mean_fwhm(worst, fwhm), "spread": spread,
              "check": check}
    for half, frames in halves.items():
        nights = sorted(night for night, paths in admitted.items() if set(paths) & set(frames))
        record_product(manifest, store, owner, ids[half], f"multi-night-seeing-{half}", {**seeing, "nights": nights},
                       frames, directory / half, files[half], stats[half], run_versions, status, reason)
    return {"status": status, "reason": reason, "frames_each": len(best), "spread": spread,
            "fwhm_ratio": check.get("fwhm_ratio")}


def check_seeing(manifest: Manifest, store: Store, *, names: set[str] | None = None) -> dict[str, Any]:
    """Judge again, from their files, the seeing pairs recorded for the
    named objects, or for all, as `split_seeing` judges a new one: a pair
    made before the checks were, or after they change. Returns each
    object's status and reason."""
    judged = {}
    for best in manifest.rows("SELECT * FROM products WHERE kind = 'multi-night-seeing-best' ORDER BY id"):
        recipe = loads(best["recipe"])
        if names is not None and recipe["object"] not in names:
            continue
        rows = {"best": best,
                "worst": manifest.rows("SELECT * FROM products WHERE id = ?",
                                       (best["id"].removesuffix("/best") + "/worst",))[0]}
        measured, stats = {}, {}
        for half, row in rows.items():
            files = loads(row["files"])
            image = np.load(store.root / files["image"], mmap_mode="r")
            stars, stats[half] = stack_psf(image)
            measured[half] = (image, np.load(store.root / files["coverage"], mmap_mode="r"), stars)
        status, reason, check = seeing_check(measured["best"], measured["worst"])
        # Each half is a quarter of the frames: 8 or more from 32 or more.
        spread = (recipe["worst_fwhm_px"] - recipe["best_fwhm_px"]) / recipe["best_fwhm_px"]
        if not screened(4 * recipe["frames_each"], spread):
            status, reason = "rejected", "seeing-spread"
        with manifest.transaction():
            for half, row in rows.items():
                manifest.update("products", "id", row["id"], {
                    "recipe": dumps({**loads(row["recipe"]), "spread": spread, "check": check}),
                    "stats": dumps({**loads(row["stats"], {}), **stats[half]}),
                    "status": status,
                    "reason": reason,
                })
        judged[recipe["object"]] = {"status": status, "reason": reason, "fwhm_ratio": check.get("fwhm_ratio")}
    return judged
