"""Noise pairs with a known clean image, from dark frames.

A noise pair's two halves both carry noise, so its target is right only
on average. Here the target is a set's deep stack, and the input is the
deep stack plus the noise a short stack of the same night would add: the
noise of real dark frames, taken with the lights' camera settings, and the
photon noise those frames lack.

For each of `size` lights, the A half of one of the set's noise pairs:

- one dark, matched to the lights by camera, gain, offset, binning, size,
  exposure and temperature, is calibrated as the stacker calibrated the
  lights, with the set's own masters, the dark's scaled by exposure as the
  stacker scales it, and less its median, which normalization would
  remove;
- photon noise is drawn in the frame's own grid, white, scaled for the
  flat's vignetting. Its variance on the sky is the light's own noise less
  the dark's, each measured as the stacker weighs frames
  (`seiza.frame_noise`); where the deep stack has signal above the sky, the
  variance grows with it, at the gain the sky implies;
- both are resampled by the light's own registration
  (`FrameDisposition.resample`), so the noise carries the correlation
  between neighbouring pixels that resampling gives a real stack, and
  scaled by the light's normalization gain.

The frames are then integrated by `seiza.integrate_registered_frames`, with
the lights' weights: the three-pass rejection a reintegrated half has, so
a hot pixel or cosmic ray the darks hold is rejected as it would be in the
lights. Put through the same steps, the lights themselves give back their
stacked half to within its background fit.

So the input holds the hot pixels, read noise, random telegraph noise,
amplifier glow and cosmic rays of the camera that took the lights. The
deep stack keeps its own noise, a stack of every frame of the night, so
the pairs use the smaller sizes, whose noise is many times the deep's.

Each product also keeps the noise of the same frames with sky photon noise
only: added to a flat sky, it is an image with nothing in it, on which a
model must not make structure.
"""

from __future__ import annotations

import hashlib
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

import seiza

from .build import BAND_SCALE_PX, binned, record_product, stack_options
from .manifest import Manifest, dumps, loads
from .store import Store, counts16, read_image, robust_sigma, versions

# The A halves whose frames lend their registration, by size.
SIZES = (8, 16)
# A dark's exposure within this share of the lights', and its sensor
# within this many degrees of the lights' median.
EXPOSURE_TOLERANCE = 0.01
TEMPERATURE_TOLERANCE_C = 2.5
# A set whose master dark, scaled to the lights' exposure, adds more read
# noise than this share of one frame's is left out: its own noise, the same
# in every light, would rule the pair's.
MASTER_EXCESS_LIMIT = 0.5


def master_excess(masters: dict[str, str], exposure: float) -> float | None:
    """The read noise a set's master dark adds to every light, as a share
    of one frame's: its scale to the lights' exposure over the square root
    of the frames it combines. `None` when its header says too little."""
    if not masters.get("dark"):
        return None
    header = seiza.read_image_header(masters["dark"])["headers"]
    frames, dark_exposure = header.get("NCOMBINE"), header.get("EXPTIME")
    if not frames or not dark_exposure:
        return None
    return exposure / float(dark_exposure) / float(frames) ** 0.5


def light_temperature(manifest: Manifest, set_id: str) -> float | None:
    """The median sensor temperature of a set's lights."""
    temperatures = [
        row["camera_temp_c"] for row in manifest.set_frames(set_id) if row["camera_temp_c"] is not None
    ]
    return statistics.median(temperatures) if temperatures else None


def matching_darks(manifest: Manifest, set_row: Any) -> list[str]:
    """The raw darks taken with a set's camera settings, at its exposure
    and temperature, in path order."""
    temperature = light_temperature(manifest, set_row["id"])
    exposure = set_row["exposure_seconds"]
    if temperature is None or not exposure:
        return []
    rows = manifest.rows(
        "SELECT path, camera_temp_c, exposure_seconds, readout_mode FROM frames "
        "WHERE origin = 'calibration' AND role = 'dark' AND is_master = 0 AND error IS NULL "
        "AND camera = ? AND gain IS ? AND \"offset\" IS ? AND binning_x = ? AND width = ? AND height = ? "
        "ORDER BY path",
        (set_row["camera"], set_row["gain"], set_row["offset"], set_row["binning"], set_row["width"],
         set_row["height"]),
    )
    return [
        row["path"]
        for row in rows
        if row["exposure_seconds"] is not None
        and abs(row["exposure_seconds"] - exposure) <= EXPOSURE_TOLERANCE * exposure
        and row["camera_temp_c"] is not None
        and abs(row["camera_temp_c"] - temperature) <= TEMPERATURE_TOLERANCE_C
    ]


class Calibration:
    """A set's masters, applied as the stacker applies them
    (`CalibrationMasters`): less the bias, less the dark's signal scaled by
    exposure when the bias is known, over the flat's response. A dark
    without `BIASSUB` holds the bias, which comes out of it first; a flat
    with none of `BIASSUB`, `DARKSUB` or `FLATNORM` is raw, and loses the
    bias too. The response is the flat over its median of positive
    values."""

    def __init__(self, masters: dict[str, str], width: int, height: int):
        self.width, self.height = width, height
        bias = read_image(masters["bias"], width, height) if masters.get("bias") else None
        self.bias = bias if bias is not None else np.zeros((height, width), dtype=np.float32)
        self.dark = np.zeros((height, width), dtype=np.float32)
        self.dark_exposure = None
        if masters.get("dark"):
            header = seiza.read_image_header(masters["dark"])["headers"]
            self.dark = read_image(masters["dark"], width, height)
            subtracted = bool(header.get("BIASSUB"))
            if not subtracted and bias is not None:
                self.dark = self.dark - bias
            if (subtracted or bias is not None) and header.get("EXPTIME"):
                self.dark_exposure = float(header["EXPTIME"])
        self.flat = np.ones((height, width), dtype=np.float32)
        if masters.get("flat"):
            header = seiza.read_image_header(masters["flat"])["headers"]
            flat = read_image(masters["flat"], width, height)
            if bias is not None and not any(header.get(key) for key in ("BIASSUB", "DARKSUB", "FLATNORM")):
                flat = flat - bias
            positive = flat[::7, ::7]
            positive = positive[np.isfinite(positive) & (positive > 0)]
            self.flat = (flat / np.median(positive)).astype(np.float32)
            self.flat[~(self.flat > 1.0e-6)] = np.nan

    def apply(self, path: str, exposure: float) -> np.ndarray:
        """A raw frame of `exposure` seconds, calibrated."""
        scale = exposure / self.dark_exposure if self.dark_exposure else 1.0
        raw = read_image(path, self.width, self.height)
        return ((raw - self.bias - scale * self.dark) / self.flat).astype(np.float32)


def write_frame(path: Path, image: np.ndarray) -> None:
    """A registered frame as a 32-bit float FITS image, for
    `integrate_registered_frames` to read back."""
    from astropy.io import fits

    fits.PrimaryHDU(image.astype(np.float32, copy=False)).writeto(path)


def integrate(paths: list[Path], weights: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """The weighted, rejected stack of the frames at `paths`: its mean (NaN
    where nothing covers it), the variance of a frame of weight 1, its
    coverage and how many samples were rejected."""
    snapshot, _ = seiza.integrate_registered_frames(
        [str(path) for path in paths], frame_weights=[[weight] for weight in weights]
    )
    shape = (snapshot.height, snapshot.width)
    coverage = np.asarray(snapshot.coverage).reshape(shape)
    mean = np.asarray(snapshot.image, dtype=np.float32).reshape(shape)
    mean = np.where(coverage > 0, mean, np.nan).astype(np.float32)
    variance = np.asarray(snapshot.variance, dtype=np.float32).reshape(shape)
    return mean, variance, coverage, int(np.asarray(snapshot.rejected_samples).sum())


def sky_mask(image: np.ndarray, noise: float) -> np.ndarray:
    """Pixels within two noise of the image's median."""
    return np.isfinite(image) & (np.abs(image - np.nanmedian(image[::7, ::7])) < 2.0 * noise)


def check(noise: np.ndarray, a: np.ndarray, b: np.ndarray, deep: np.ndarray, a_frames: int, b_frames: int) -> dict[str, Any]:
    """The synthetic noise against the real A half's, on the sky, at the
    pixel scale and binned by the band scale. `A − B` holds A's noise and
    B's, and B's is A's scaled by their frame counts, so A's alone is
    `var(A − B) / (1 + a_frames / b_frames)`."""
    difference = a - b
    share = 1.0 + a_frames / b_frames
    sky = sky_mask(deep, (robust_sigma(difference) or 0.0) / np.sqrt(share))
    sky &= np.isfinite(noise) & np.isfinite(difference)
    real = np.where(sky, difference, np.nan)
    synthetic = np.where(sky, noise, np.nan)
    pixel_real = (robust_sigma(real) or float("nan")) / np.sqrt(share)
    pixel_synthetic = robust_sigma(synthetic) or float("nan")
    band_real = float(np.nanstd(binned(real, BAND_SCALE_PX))) / np.sqrt(share)
    band_synthetic = float(np.nanstd(binned(synthetic, BAND_SCALE_PX)))
    return {
        "pixel_real": round(float(pixel_real), 4),
        "pixel_synthetic": round(float(pixel_synthetic), 4),
        "pixel_ratio": round(float(pixel_synthetic / pixel_real), 4),
        "band_real": round(band_real, 4),
        "band_synthetic": round(band_synthetic, 4),
        "band_ratio": round(band_synthetic / band_real, 4),
        "sky_share": round(float(sky.mean()), 3),
    }


def noise_stacks(
    store: Store,
    set_row: Any,
    masters: dict[str, str],
    reference: str,
    lights: list[str],
    darks: list[str],
    deep: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """The noise `lights` would have had from `darks`, one each, registered
    onto `reference`: integrated with the photon noise of `deep`'s signal
    (`scene`) and with the sky's alone (`sky`), each as `integrate` gives
    it, and what each frame contributed (`frames`)."""
    stacker = seiza.LiveStacker(
        reference, options=stack_options(None),
        bias=masters.get("bias"), dark=masters.get("dark"), flat=masters.get("flat"),
    )
    dispositions, _ = stacker.push_fits_pipelined([path for path in lights if path != reference])
    frames: list[tuple[str, Any]] = [(reference, None)] if reference in lights else []
    frames += [(str(d.source), d) for d in dispositions if d.accepted]
    del stacker

    width, height = set_row["width"], set_row["height"]
    exposure = float(set_row["exposure_seconds"])
    calibration = Calibration(masters, width, height)
    signal = np.clip(np.nan_to_num(deep - np.nanmedian(deep[::7, ::7]), nan=0.0), 0.0, None)
    with tempfile.TemporaryDirectory(dir=store.scratch, prefix="dark-noise-") as scratch:
        scene_paths, sky_paths, weights = [], [], []
        records = []
        for index, ((light, disposition), dark) in enumerate(zip(frames, darks)):
            dark_image = calibration.apply(dark, exposure)
            dark_image -= np.nanmedian(dark_image[::3, ::3])
            light_image = calibration.apply(light, exposure)
            sky = float(np.nanmedian(light_image[::3, ::3]))
            light_noise = seiza.frame_noise(light_image)[0]
            dark_noise = seiza.frame_noise(dark_image)[0]
            del light_image
            photon = max(light_noise ** 2 - dark_noise ** 2, 0.0)
            gain, weight = 1.0, 1.0
            if disposition is not None:
                gain = disposition.normalization_mean_gain or 1.0
                weight = disposition.weight[0] if disposition.weight else 1.0
            white = rng.standard_normal((height, width), dtype=np.float32) / np.sqrt(calibration.flat)
            dark_image = np.nan_to_num(dark_image, nan=0.0)
            white = np.nan_to_num(white, nan=0.0).astype(np.float32)
            if disposition is not None:
                dark_image = disposition.resample(dark_image)
                white = disposition.resample(white)
            # The photon variance per pixel: the sky's, growing with the
            # deep stack's signal in this frame's units at the gain the sky
            # implies.
            growth = signal / (gain * sky) if sky > 0 else 0.0
            scene_paths.append(Path(scratch) / f"scene-{index:03d}.fits")
            write_frame(scene_paths[-1], gain * (dark_image + np.sqrt(photon * (1.0 + growth)) * white))
            sky_paths.append(Path(scratch) / f"sky-{index:03d}.fits")
            write_frame(sky_paths[-1], gain * (dark_image + np.sqrt(photon) * white))
            weights.append(weight)
            records.append({
                "light": light, "dark": dark, "gain": round(gain, 5), "weight": round(weight, 5),
                "sky": round(sky, 3), "light_noise": round(light_noise, 4), "dark_noise": round(dark_noise, 4),
                "electrons_per_unit": round(sky / photon, 4) if photon > 0 else None,
            })
            del dark_image, white
        return {"scene": integrate(scene_paths, weights), "sky": integrate(sky_paths, weights), "frames": records}


def synthesize(
    manifest: Manifest, store: Store, set_row: Any, size: int, *, again: bool = False
) -> dict[str, Any] | None:
    """One set's dark-noise pair for its A half of `size` frames; `None`
    when the set lacks a half of that size, a sound dark master or enough
    darks."""
    set_id = set_row["id"]
    product_id = f"{set_id}/dark-noise/n{size:02d}"
    existing = manifest.rows("SELECT status FROM products WHERE id = ?", (product_id,))
    if existing and existing[0]["status"] == "built" and not again:
        return None
    masters = loads(set_row["calibration"], {}).get("masters", {})
    excess = master_excess(masters, float(set_row["exposure_seconds"]))
    if excess is None or excess > MASTER_EXCESS_LIMIT:
        return None
    halves = manifest.rows(
        "SELECT * FROM products WHERE id = ? AND status = 'built'", (f"{set_id}/noise/n{size:02d}-d0/a",)
    )
    deep_rows = manifest.rows("SELECT * FROM products WHERE id = ? AND status = 'built'", (f"{set_id}/deep",))
    if not halves or not deep_rows:
        return None
    half, deep_row = dict(halves[0]), dict(deep_rows[0])
    darks = matching_darks(manifest, set_row)
    if len(darks) < size:
        return None
    started = time.monotonic()
    seed = int(hashlib.sha256(product_id.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    chosen = [darks[index] for index in sorted(rng.choice(len(darks), size, replace=False))]

    reference = loads(deep_row["versions"])["reference"]
    lights = loads(half["frames"])
    deep_files = loads(deep_row["files"])
    deep = np.load(store.root / deep_files["image"]).astype(np.float32)
    deep_coverage = np.load(store.root / deep_files["coverage"])
    deep[deep_coverage == 0] = np.nan
    deep_variance = np.load(store.root / deep_files["variance"]).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        deep_sigma2 = np.where(deep_coverage > 0, deep_variance / deep_coverage, np.nan).astype(np.float32)

    stacks = noise_stacks(store, set_row, masters, reference, lights, chosen, deep, rng)
    noise, variance, coverage, scene_rejected = stacks["scene"]
    gate, gate_variance, gate_coverage, sky_rejected = stacks["sky"]
    records = stacks["frames"]
    covered = (coverage > 0) & (deep_coverage > 0)
    image = np.where(covered, deep + noise, np.nan).astype(np.float32)
    # The input's noise is the synthetic noise and the deep stack's own;
    # `variance / coverage` is the noise map, as for a stacked half.
    total = np.where(covered, variance + coverage * np.nan_to_num(deep_sigma2), 0.0).astype(np.float32)

    a_image = np.load(store.root / loads(half["files"])["image"]).astype(np.float32)
    b_row = manifest.rows("SELECT * FROM products WHERE id = ?", (half["id"].removesuffix("/a") + "/b",))[0]
    b_image = np.load(store.root / loads(b_row["files"])["image"]).astype(np.float32)
    checked = check(np.where(covered, noise, np.nan), a_image, b_image, deep, len(lights), len(loads(b_row["frames"])))
    del a_image, b_image

    directory = store.set_directory(set_id) / "dark-noise" / f"n{size:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, array in (
        ("image", image), ("variance", total), ("coverage", counts16(np.where(covered, coverage, 0))),
        ("noise", gate), ("noise_variance", gate_variance), ("noise_coverage", counts16(gate_coverage)),
    ):
        np.save(directory / f"{name}.npy", array)
        files[name] = f"{name}.npy"
    gains = [r["electrons_per_unit"] for r in records if r["electrons_per_unit"] is not None]
    recipe = {
        "recipe": "dark-noise",
        "size": size,
        "seed": seed,
        "target": deep_row["id"],
        "half": half["id"],
        "darks_matched": len(darks),
        "master_excess": round(excess, 4),
        "frames": records,
    }
    stats = {
        "check": checked,
        "electrons_per_unit": round(statistics.median(gains), 4) if gains else None,
        "rejected": {"scene": scene_rejected, "noise": sky_rejected},
        "seconds": round(time.monotonic() - started, 1),
    }
    record_product(
        manifest, store, set_id, product_id, "dark-noise", recipe, [r["light"] for r in records],
        directory, files, stats, {**versions(), "masters": masters, "reference": reference},
    )
    return {"product": product_id, **stats}


def run(
    manifest: Manifest,
    store: Store,
    *,
    sets: set[str] | None = None,
    splits: tuple[str, ...] = ("train", "validation"),
    sizes: tuple[int, ...] = SIZES,
    again: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Make the dark-noise pairs of every built set in `splits`, or of the
    sets `sets` names."""
    rows = manifest.rows(
        f"SELECT * FROM sets WHERE build_status = 'built' AND split IN ({','.join('?' * len(splits))}) ORDER BY name",
        splits,
    )
    made = skipped = failed = 0
    for set_row in rows:
        if sets is not None and set_row["id"] not in sets:
            continue
        for size in sizes:
            if limit is not None and made >= limit:
                return {"made": made, "skipped": skipped, "failed": failed}
            try:
                result = synthesize(manifest, store, set_row, size, again=again)
            except Exception as error:  # noqa: BLE001 - one bad set must not stop the pass
                failed += 1
                print(f"dark-noise: {set_row['name']} n{size:02d}: {error}", file=sys.stderr)
                continue
            if result is None:
                skipped += 1
                continue
            made += 1
            print(f"dark-noise: {set_row['name']} n{size:02d}: {dumps(result)}", file=sys.stderr, flush=True)
    return {"made": made, "skipped": skipped, "failed": failed}


def pairs(manifest: Manifest, split: str, sets: set[str] | None = None) -> list[dict[str, Any]]:
    """The built dark-noise products of one split's sets, each with its
    target."""
    rows = manifest.rows(
        "SELECT p.*, s.name FROM products p JOIN sets s ON s.id = p.set_id "
        "WHERE p.kind = 'dark-noise' AND p.status = 'built' AND s.split = ? ORDER BY p.id",
        (split,),
    )
    found = []
    for row in rows:
        if sets is not None and row["set_id"] not in sets:
            continue
        target = manifest.rows("SELECT * FROM products WHERE id = ?", (loads(row["recipe"])["target"],))
        if target:
            found.append({"input": dict(row), "target": dict(target[0]), "set": row["name"]})
    return found
