"""Train and evaluate the PSF model on the sharpening banks.

The model maps an image whose PSF is P to the same sky through a round
target PSF T, told the ratio FWHM(T) / FWHM(P) at each pixel and the local
PSF's stamp. It learns from the banks `sharpen_bank` cuts:

- **Known-kernel pairs**, from the deep stacks' windows. An elliptical
  Gaussian K is drawn, so that P_in = P_ref ∗ K is 1.1 to 2 times as wide
  as the window's own PSF, P_ref. The input is the window blurred by K,
  with fresh white noise at a drawn share of the stack's own: the inputs
  will be the noise model's output, which keeps a little noise. The target
  is the window brought to a round T. Second moments add under
  convolution, so blurring the window by the Gaussian whose covariance is
  Σ_T − Σ_ref, with Σ_T round and at least Σ_ref's largest eigenvalue,
  leaves a PSF whose second moments are round. A stamp's second moments
  are its wings' as much as its core's, so T is often wider than P_ref:
  by 9% at the median of the train bank's stamps. The ratio is FWHM(T) /
  FWHM(P_in), both measured on the stamps as the bank measures stamp
  widths (`stamp_width`), and the network reads P_in's stamp, never
  P_ref's. A share of them are identity windows: K a point, the ratio 1,
  and the target the input itself, which is the design's identity loss.
- **Pairs** on one grid: the seeing bank's worst half as input and its
  best half as target, and any pairs a caller supplies in that bank's
  format. The network reads the input's stamp and the bank's ratio, from
  the stamps' widths (which every window has) or from the field's Moffat
  fit (which fails below about 2 px). The target is moved so that its
  stars' light-weighted centres meet the input's, so the model is never
  taught to move stars: halves on one grid differ in their stars' centres
  as much as their PSFs differ in symmetry, which a stamp centred on its
  star does not show. Windows whose target is wider than their input are
  left out.

Windows of either kind whose stamps are out of focus, ringed or flat-topped,
are left out, and so is every window of a source where more than a quarter
are.

What the network reads is made from the image alone, as it will be at
inference: `asinh((x − b) / σ̄)` of the image, `asinh(σ / σ̄)` of a noise
map made from the image (`image_sigma`), the ratio, and a point-source map,
`asinh` of the input's detections rendered with the local stamp, left
empty a tenth of the time, as an image with no catalogue and few
detections will be. `b` and `σ̄` are the whole image's: a pair's from its
bank; a known-kernel window's from the window, since its blur and noise
are made here. The stamp is read as two planes, the stamp over its peak,
and its asinh, which shows the wings. The stacks' own noise maps weigh the
loss and nothing else. The output is a residual added to the asinh of the
input in σ̄, which the network reads, and taken back through sinh: on the
sky a residual in σ̄, as the noise model's, and at a star's core nearly a
log, so a core thousands of σ̄ from its target is in reach (`RESIDUALS`).

The loss, in linear units:

- Charbonnier error of output against target, in units of the target's own
  noise, which the stack's map gives (the input stack's where a supplied
  pair keeps none, and the image's own where one frame alone covers the
  sky), read no higher than `noise_cap` times its tile's median, with
  clipped cores masked out, and the band at a window's edge that its blur
  or shift read past;
- the same on the error's gradient, the error averaged over 1, 2 and 4
  pixel blocks, each times its side, so every scale's noise weighs alike;
- Charbonnier error of each point source's flux within three times the
  input's FWHM, in units of that flux's noise, counted once for each pixel
  of its aperture, as a pixel's error is.

The second and third weigh a tenth each (see `LossSettings`).

Reblur consistency, `|R ∗ f(y) − y|²` on real images without a target,
and a centroid-shift loss are left for later: both would be terms of
`loss_terms`, the first on its own batches.

Augmentation is the eight flips and turns, alike for the image, every map,
the apertures and the stamp, and exposure scaling: a known-kernel window's
signal is scaled about its background by up to `exposure` either way
before the fresh noise is added, so the same noise meets a brighter or
fainter sky. A pair's noise is its stacks' own, which a scale would move
off its maps, so pairs are not scaled. There are no sub-pixel shifts.

Batches are drawn as the noise trainer's are: each from a generator seeded
by its step, so threads that build them ahead change nothing and a run
resumed from a checkpoint is the run it continues. Within a bank, no
object (a telescope, target and filter, over all its nights) takes more
than `object_cap` of the draws: an object stacked over many nights is in
the deep bank as each night's stack and as their multi-night stack, which
share their sky.

Inference (`sharpen`) runs the network over an image's overlapping tiles,
each conditioned on the stamp at its centre and its own point-source map,
which contact sheets and the bundle will call; evaluation runs it on the
validation banks' windows.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import psf_fields
from .noise_model import Prefetch, Rewind, device, image_sigma, offloaded, step_generator, tiling
from .sharpen_bank import SATURATION_GROW, STAMP, grown, open_bank, stamp_width
from .store import Store, versions
from .tiles import infer
from .unet import SharpeningUNet, reach, receptive_field

# The planes the network reads, in order, and the stamp's.
PLANES = ("image", "noise", "ratio", "points")
# The planes a model with separate strengths for stars and for the rest
# reads: the ratio asked of each.
PLANES_NONSTELLAR = ("image", "noise", "ratio", "nonstellar_ratio", "points")
STAMP_PLANES = ("peak", "wings")
# A known-kernel window's mask of its stars, `S / (S + STAR_MASK_NOISE σ)`
# of its detections rendered with its stamp, `S`, and its noise, `σ`: the
# stars' own share of each pixel, where they stand above the noise.
STAR_MASK_NOISE = 3.0
# A deep stack's clipped cores are flat plateaus; blurred, they became soft
# bumps no camera makes, and runs trained on them dug dark rings 4 to 5
# widths about real clipped stars, whose plateaus end steeply, down to
# 19σ̄ below the sky (2026-11-09). So each clipped core is filled first:
# the window's stamp at the core, scaled to the unclipped pixels out to
# `FILL_REACH` widths, wherever it is brighter; the blurred input and the
# target are then clipped where the stack clipped, as a camera clips after
# the sky has blurred the star.
FILL_REACH = 4.0
# A pair bank's denoised planes, which `sharpen_clean` makes beside it.
CLEAN_PLANES = ("input_clean", "target_clean")
# A known-kernel window's passed noise is blurred by up to this many pixels.
NOISE_BLUR = 0.6
# The core's place is sought this far, in pixels, either way of its
# plateau's middle, at these steps.
FILL_SHIFT = 0.6
FILL_STEP = 0.1
# The stamp's second plane is `asinh(s / (peak · WINGS)) / asinh(1 / WINGS)`:
# its wings down to a ten-thousandth of the peak still show.
WINGS = 1e-3
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
SOURCES = ("kernel", "seeing", "pairs")
# How a pair's target is moved, so the model is never taught to move stars:
# by the shift of its stars' light-weighted centres, by the median shift of
# their detections' centroids, or not at all. On the C925's lopsided worst
# halves a star's peak and its light's centre move apart, by up to 2 px,
# and a detection's centroid falls between them, nearer the peak the
# fainter the star (2026-11-09). The light's centre is the same for every
# star, and the one point that convolution with a PSF centred on its own
# light keeps.
CENTROID_OFFSETS = ("light", "detection", "none")
# A pair's light offset is measured on its windows' isolated stars, each
# within this many input widths, less the median of the annulus out to
# half as far again ...
LIGHT_RADIUS = 2.5
# ... and fitted as a plane over the field where it finds this many stars,
# as their median where it finds `LIGHT_LEAST`, and taken from its
# detections' median where it finds fewer.
LIGHT_PLANE_STARS = 30
LIGHT_LEAST = 8
# The version of the light offsets a bank keeps beside its arrays.
LIGHT_VERSION = 1
RATIO_SOURCES = ("stamp", "fit")
# How the network's residual meets the input: added to it in σ̄, as the
# noise model's is, or added to the asinh of it, which the network reads,
# and taken back through sinh. On the sky the two agree; at a star's core,
# where a sharpened target differs from its input by thousands of σ̄,
# a linear residual starting from zero never got there: two smoke runs
# (2026-11-09) gave cores residuals under 1σ̄ where their targets asked
# 700σ̄ at the median, and sharpened nothing.
RESIDUALS = ("asinh", "linear")
# The bank's info column each ratio source reads.
RATIO_COLUMNS = {"stamp": "width_ratio", "fit": "ratio"}
# A stamp is out of focus where its azimuthal profile rises again within
# two widths of its middle (a ring), or where the radius at which it falls
# to 90% of its centre is more than `FLAT_TOP` of the radius at which it
# falls to half (a flat top: a Gaussian reads 0.2 to 0.35 sampled, a disc
# near 1). Over the PSF fields' nodes, ringed and flat stamps read 0.6 to
# 0.9, and stamps in focus 0.2 to 0.45 at any width, 2 px or 13 (2026-11-09).
# A source with more than `OUT_OF_FOCUS_SHARE` of its windows out of focus
# is left out whole: one Ultracat object's worst halves, with rings over
# half their field, and a few deep stacks whose every corner is out.
FLAT_TOP = 0.6
OUT_OF_FOCUS_SHARE = 0.25
# A known-kernel window's draw of K stops within this share of the width
# ratio asked.
RATIO_TOLERANCE = 0.01
KERNEL_ITERATIONS = 8
# The canvas a stamp is blurred on, so a wide kernel does not wrap.
STAMP_CANVAS = 256
# Windows are blurred and shifted with this much reflected around them.
WINDOW_PAD = 64
# A blur reads this many of its σ past a pixel: the band at a window's edge
# it reads past, and how far a clipped core's error spreads.
BLUR_REACH = 3.0
# A pair's target is moved by its offset, which reads past the window's
# edge by a few pixels.
SHIFT_BAND = 4
# The noise below which a pixel's error is not weighed further up, in σ̄.
SIGMA_FLOOR = 0.25
# Point sources whose flux the loss weighs: a detection, unclipped, at
# least this signal to noise; at most `APERTURES` of them a tile, brightest
# first, none whose aperture meets another's or a masked pixel.
FLUX_SNR = 10.0
APERTURES = 32
APERTURE_FWHM = 3.0
# Validation batches draw from generators seeded apart from the training
# steps'.
VALIDATION_STREAM = 1 << 20
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class Settings:
    """How windows become examples: the tile, the known-kernel draws, the
    pairs' handling and the point-source map."""

    tile: int = 256
    # FWHM(P_in) / FWHM(P_ref), drawn uniformly between these.
    kernel_ratios: tuple[float, float] = (1.1, 2.0)
    # K's ellipticity, 1 − minor/major, drawn up to this, at a random angle.
    kernel_ellipticity: float = 0.3
    # The fresh noise's share of the deep stack's own, drawn up to this.
    kernel_noise: float = 0.5
    # The share of known-kernel windows that are identity windows.
    identity: float = 0.1
    # A known-kernel window's signal is scaled by up to this either way.
    exposure: float = 2.0
    # The narrowest target, as a stamp width.
    floor_px: float = 1.5
    # No window asks for a target wider than this times its input; a
    # known-kernel window's K is drawn wide enough to keep it.
    max_ratio: float = 1.0
    min_ratio: float = 0.0
    points_empty: float = 0.1
    centroid_offset: str = "light"
    ratio_source: str = "stamp"
    residual: str = "asinh"
    # Separate strengths for stars and for the rest (`PLANES_NONSTELLAR`):
    # a known-kernel window asks each for a ratio drawn between the
    # sharpest its stack allows and 1, the same for both `equal_share` of
    # the time; a pair asks both for its own.
    nonstellar: bool = False
    equal_share: float = 0.3
    # Fill a known-kernel window's clipped cores before its blur, and clip
    # input and target after it (`FILL_REACH`).
    fill_clipped: bool = False
    # Targets carry their input's noise, so the model sharpens what is not
    # noise and passes the noise on: a known-kernel window's fresh noise,
    # shared a little between neighbours as a stack's is, is added to its
    # target too, and a pair whose bank keeps denoised planes
    # (`CLEAN_PLANES`) is asked for its target's denoised plus its input's
    # own noise. Without it the first runs took out 94% of a wide-starred
    # stack's pixel noise where the other sharpener takes out none
    # (2026-11-09): a pair's halves hold noise of their own, and the
    # likeliest target of an input's noise is none.
    pass_noise: bool = False
    # The share of known-kernel windows blurred by a real PSF, another
    # window's stamp, in place of a Gaussian: the wings, lopsided cores and
    # tilted shapes that telescopes and seeing make, which are what an
    # image is to be deconvolved of. A donor whose blur falls outside
    # `kernel_ratios` gives way to a Gaussian.
    real_kernels: float = 0.0
    # Make a known-kernel window's target PSF round about its middle, not
    # only in its second moments: a blur can even out a PSF's widths but
    # not a cross of four arms, whose widths are alike already, and runs
    # that took a star's round halo off left its cross standing, twice as
    # plain as the other sharpener leaves it (2026-11-09).
    round_target: bool = False

    @property
    def planes(self) -> tuple[str, ...]:
        return PLANES_NONSTELLAR if self.nonstellar else PLANES

    def __post_init__(self) -> None:
        # Runs before the light offset recorded the detections' as "shift".
        if self.centroid_offset == "shift":
            object.__setattr__(self, "centroid_offset", "detection")
        if self.residual not in RESIDUALS:
            raise ValueError(f"residual is one of {RESIDUALS}, not {self.residual!r}")
        if self.centroid_offset not in CENTROID_OFFSETS:
            raise ValueError(f"centroid_offset is one of {CENTROID_OFFSETS}, not {self.centroid_offset!r}")
        if self.ratio_source not in RATIO_SOURCES:
            raise ValueError(f"ratio_source is one of {RATIO_SOURCES}, not {self.ratio_source!r}")
        low, high = self.kernel_ratios
        if not 1.0 < low <= high:
            raise ValueError(f"kernel ratios run from above 1 up: {self.kernel_ratios}")
        if self.tile % 8:
            raise ValueError(f"a tile of {self.tile} is not a multiple of 8")


@dataclass(frozen=True)
class LossSettings:
    """The loss's terms: Charbonnier's ε, in units of the target's noise;
    the most the target's noise is read as, in multiples of its tile's
    median; the gradient term's weight and block sides; the flux term's
    weight.

    A stack's noise map at a star's core holds its frames' scatter with
    the seeing, thousands of σ̄ where the sky's is one, and errors read in
    it leave a blurred star all but free: a first run (2026-11-09) learnt
    to denoise its pairs and never to sharpen. Capped at `noise_cap`, the
    map still weighs the sky and the faint signal, and a core counts.

    The stars are a two-hundredth of a tile's pixels, and the targets' own
    noise, which no model can predict, fills the rest; a term that adds
    noise drowns them. Fitting one batch of known-kernel tiles, the cores
    never moved with the gradient term at half weight; at a tenth they
    reached four fifths of their targets in 300 steps."""

    epsilon: float = 0.1
    noise_cap: float = 10.0
    gradient_weight: float = 0.1
    gradient_scales: tuple[int, ...] = (1, 2, 4)
    flux_weight: float = 0.1


# Banks.


def find_bank(work: Path, label: str, split: str) -> Path:
    """The finished bank of `label` and `split` under `work`, the newest
    where several are."""
    found = sorted((path for path in work.glob(f"sharpen-{label}-{split}-*") if (path / "done.json").exists()),
                   key=lambda path: (path / "done.json").stat().st_mtime)
    if not found:
        raise FileNotFoundError(f"no finished {label} bank of the {split} split under {work}: run cut-sharpen")
    return found[-1]


def object_name(set_name: str) -> str:
    """A set's object, its name less its night: telescope, target and
    filter, as the multi-night pass names an object."""
    return "/".join(part for part in set_name.split("/") if not DATE.match(part))


def capped_odds(groups: np.ndarray, cap: float) -> np.ndarray:
    """How often to draw each window, given its object's index: every
    window alike, except that no object takes more than `cap` of the draws
    (or an even share, where fewer objects than `1 / cap` leave no other
    way), the rest going to the others by their windows."""
    count = len(groups)
    if count == 0:
        return np.zeros(0)
    _, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    cap = max(cap, 1.0 / len(sizes)) if cap > 0 else 1.0
    shares = sizes / count
    capped = np.zeros(len(sizes), dtype=bool)
    while True:
        over = ~capped & (shares > cap * (1 + 1e-12))
        if not over.any():
            break
        capped |= over
        rest = (1.0 - cap * capped.sum()) / sizes[~capped].sum() if (~capped).any() else 0.0
        shares = np.where(capped, cap, sizes * rest)
    odds = shares[inverse] / sizes[inverse]
    return odds / odds.sum()


class Bank:
    """A sharpening bank as the trainer draws from it: its arrays, the
    windows it keeps, how often each is drawn, and why the rest are left
    out. A pair bank keeps the windows whose ratio is known and within the
    settings' bounds, of the pairs not `excluded`. With `focus_check`, any
    bank leaves out the windows whose stamps are out of focus, and every
    window of a source where more than `OUT_OF_FOCUS_SHARE` are. Any bank
    leaves out the sources whose input is of a kind in `left_out_kinds`.
    `set_names` names each set, so its object is known; without it a
    source's object is its record's, or its set. With `multi_night_share`,
    a known-kernel bank's stacks of many nights take that share of its
    draws, the nights' own stacks the rest."""

    def __init__(self, directory: Path, settings: Settings, *, excluded: tuple[str, ...] = (), object_cap: float = 0.0,
                 set_names: dict[str, str] | None = None, weight: float = 1.0, focus_check: bool = True,
                 left_out_kinds: tuple[str, ...] = (), multi_night_share: float = 0.0) -> None:
        self.directory = Path(directory)
        self.arrays = open_bank(self.directory)
        done = self.arrays["done"]
        self.label, self.pairs, self.weight = done["label"], bool(done["pairs"]), weight
        self.side = int(done["window"])
        self.columns = {name: index for index, name in enumerate(done["info"])}
        self.info = np.asarray(self.arrays["info"])
        self.sources = self.arrays["sources"]
        self.objects = [self.object_of(source, set_names) for source in self.sources]
        source_of = self.info[:, 0].astype(int)
        keep = np.ones(len(self.info), dtype=bool)
        self.left: dict[str, int] = {}

        def leave(mask: np.ndarray, reason: str) -> None:
            dropped = keep & mask
            if dropped.any():
                self.left[reason] = int(dropped.sum())
            keep[dropped] = False

        if self.pairs:
            excluded_sources = np.array([pair_id(source) in excluded or source["input"] in excluded
                                         or source.get("target") in excluded for source in self.sources])
            leave(excluded_sources[source_of], "excluded")
        kinds = np.array([source.get("input_kind") for source in self.sources])
        for kind in left_out_kinds:
            leave((kinds == kind)[source_of], f"kind:{kind}")
        if self.pairs:
            ratio = self.info[:, self.columns[RATIO_COLUMNS[settings.ratio_source]]]
            leave(~np.isfinite(ratio), "ratio-unknown")
            with np.errstate(invalid="ignore"):
                leave(ratio > settings.max_ratio, "ratio-above")
                leave(ratio < settings.min_ratio, "ratio-below")
        if focus_check:
            stamps = (("input_stamp", "input_width"), ("target_stamp", "target_width")) if self.pairs \
                else (("stamp", "width"),)
            out = np.zeros(len(self.info), dtype=bool)
            for name, width in stamps:
                out |= out_of_focus(self.arrays[name], self.info[:, self.columns[width]])
            counts = np.bincount(source_of, minlength=len(self.sources))
            share = np.bincount(source_of, weights=out, minlength=len(self.sources)) / np.maximum(counts, 1)
            leave((share > OUT_OF_FOCUS_SHARE)[source_of], "out-of-focus-source")
            leave(out, "out-of-focus")
        self.kept = np.nonzero(keep)[0]
        if not len(self.kept):
            raise ValueError(f"the {self.label} bank at {self.directory} keeps no window")
        names = np.array(self.objects)
        self.odds = capped_odds(names[source_of[self.kept]], object_cap)
        deepest = (kinds == "multi-night")[source_of[self.kept]]
        if multi_night_share > 0 and not self.pairs and deepest.any() and (~deepest).any():
            self.odds = np.where(deepest, self.odds * multi_night_share / self.odds[deepest].sum(),
                                 self.odds * (1 - multi_night_share) / self.odds[~deepest].sum())
        self.multi_night_share = float(self.odds[deepest].sum())
        self.cumulative = np.cumsum(self.odds)
        self.object_cap = object_cap
        self.loaded: dict[int, np.ndarray] = {}
        self.light = light_offsets(self) if self.pairs and settings.centroid_offset == "light" else {}
        self.clean: dict[str, Any] = {}
        if self.pairs and (self.directory / "clean.json").exists():
            self.clean = {name: np.load(self.directory / f"{name}.npy", mmap_mode="r") for name in CLEAN_PLANES}
            self.clean["record"] = json.loads((self.directory / "clean.json").read_text())

    @staticmethod
    def object_of(source: dict[str, Any], set_names: dict[str, str] | None) -> str:
        if set_names and source.get("set") in set_names:
            return object_name(set_names[source["set"]])
        return (source.get("record") or {}).get("object") or source.get("set") or source["input"]

    def draw(self, rng: np.random.Generator) -> int:
        """A window, drawn by the bank's odds."""
        return int(self.kept[min(int(np.searchsorted(self.cumulative, rng.random() * self.cumulative[-1])),
                                 len(self.kept) - 1)])

    def row(self, window: int) -> dict[str, float]:
        return {name: float(self.info[window, index]) for name, index in self.columns.items()}

    def points(self, source: int) -> np.ndarray:
        """A source's point sources, in `psf_fields.DETECTION_COLUMNS`, in
        its image's coordinates."""
        if source not in self.loaded:
            self.loaded[source] = np.load(self.directory / "points" / f"{source}.npy")
        return self.loaded[source]

    def offset(self, source: int) -> tuple[float, float] | None:
        """A pair's median detection offset, target from input, (x, y)."""
        found = (self.sources[source].get("alignment") or {}).get("offset_px")
        return None if found is None else (float(found[0]), float(found[1]))

    def target_offset(self, window: int, how: str) -> tuple[float, float] | None:
        """How far a pair's target stars sit from its input's at a window's
        middle, (x, y), measured as `how` says (`CENTROID_OFFSETS`): the
        light offset where its pair has one, else the detections'."""
        if how == "none":
            return None
        source = int(self.info[window, 0])
        fit = self.light.get(source) if how == "light" else None
        if fit is None:
            return self.offset(source)
        middle = (self.side - 1) / 2
        across = (self.info[window, self.columns["x"]] + middle - fit["centre"][0]) / 1000
        down = (self.info[window, self.columns["y"]] + middle - fit["centre"][1]) / 1000
        return tuple(float(c[0] + c[1] * across + c[2] * down) for c in (fit["dx"], fit["dy"]))

    def summary(self) -> dict[str, Any]:
        """What a run records of the bank."""
        kept_objects = {self.objects[int(self.info[window, 0])] for window in self.kept}
        source_of = self.info[self.kept, 0].astype(int)
        shares: dict[str, float] = {}
        for window_source, odds in zip(source_of, self.odds):
            name = self.objects[window_source]
            shares[name] = shares.get(name, 0.0) + float(odds)
        found = {"directory": str(self.directory), "label": self.label, "windows": int(len(self.info)),
                 "kept": int(len(self.kept)), "left": self.left, "objects": len(kept_objects),
                 "largest_object_share": round(max(shares.values()), 4), "object_cap": self.object_cap,
                 "weight": self.weight, "multi_night_share": round(self.multi_night_share, 4)}
        if self.clean:
            found["clean"] = self.clean["record"]
        if self.light:
            kinds = [fit["kind"] for fit in self.light.values()]
            found["light_offsets"] = {kind: kinds.count(kind) for kind in sorted(set(kinds))}
            found["light_offsets"]["detection"] = len(self.sources) - len(kinds)
        return found


def pair_id(source: dict[str, Any]) -> str:
    """A pair's id: its record's, as the seeing bank names it, or its
    input's."""
    return (source.get("record") or {}).get("pair") or source["input"]


def stamp_profiles(stamps: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """Each stamp's azimuthal mean profile about its middle, in whole-pixel
    radii, for a stack of stamps of one side."""
    side = stamps.shape[-1]
    middle = (side - 1) / 2
    down, across = np.mgrid[0:side, 0:side] - middle
    radii = np.rint(np.hypot(across, down)).astype(int).ravel()
    spread = np.zeros((side * side, radii.max() + 1), dtype=np.float32)
    spread[np.arange(side * side), radii] = 1.0
    spread /= spread.sum(axis=0)
    return np.concatenate([np.asarray(stamps[start:start + chunk], dtype=np.float32).reshape(-1, side * side) @ spread
                           for start in range(0, len(stamps), chunk)])


def falling_to(profiles: np.ndarray, level: float) -> np.ndarray:
    """The radius at which each profile first falls below `level` of its
    centre, between whole-pixel radii; NaN where it never does."""
    centre = profiles[:, 0]
    below = profiles[:, 1:] < level * centre[:, None]
    index = below.argmax(axis=1) + 1
    rows = np.arange(len(profiles))
    before, after = profiles[rows, index - 1], profiles[rows, index]
    with np.errstate(divide="ignore", invalid="ignore"):
        radius = (index - 1) + (before - level * centre) / (before - after)
    return np.where(below.any(axis=1) & (centre > 0), radius, np.nan)


def out_of_focus(stamps: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """Which stamps are out of focus: a profile rising again within two
    `widths` of the middle, a flat top (`FLAT_TOP`), or no positive centre."""
    profiles = stamp_profiles(stamps)
    reach = np.maximum(2, np.ceil(2 * np.nan_to_num(np.asarray(widths, dtype=np.float64), nan=1.0))).astype(int)
    radii = np.arange(1, profiles.shape[1])
    rising = np.where(radii[None, :] <= reach[:, None], profiles[:, 1:], -np.inf).max(axis=1) > profiles[:, 0]
    with np.errstate(invalid="ignore"):
        flat = falling_to(profiles, 0.9) / falling_to(profiles, 0.5) > FLAT_TOP
    return rising | flat | ~(profiles[:, 0] > 0)


def light_centre(plane: np.ndarray, x: float, y: float, radius: float) -> tuple[float, float] | None:
    """A star's light-weighted centre within `radius` of (`x`, `y`), less
    the median of the annulus out to 1.5 `radius`; None where either holds
    a pixel not finite or the light is not positive."""
    reach = int(math.ceil(1.5 * radius)) + 1
    column, row = int(round(x)), int(round(y))
    if row - reach < 0 or column - reach < 0 or row + reach >= plane.shape[0] or column + reach >= plane.shape[1]:
        return None
    values = plane[row - reach:row + reach + 1, column - reach:column + reach + 1].astype(np.float64)
    down, across = np.mgrid[-reach:reach + 1, -reach:reach + 1]
    distance = np.hypot(across + column - x, down + row - y)
    inside, annulus = distance <= radius, (distance > radius) & (distance <= 1.5 * radius)
    if not (np.isfinite(values[inside]).all() and np.isfinite(values[annulus]).all()):
        return None
    light = np.where(inside, values - float(np.median(values[annulus])), 0.0)
    total = light.sum()
    if not total > 0:
        return None
    return float(column + (light * across).sum() / total), float(row + (light * down).sum() / total)


def peak_centre(plane: np.ndarray, x: float, y: float) -> tuple[float, float] | None:
    """A star's peak: its brightest pixel within 2 px of (`x`, `y`), moved
    by a parabola through each axis' three pixels."""
    column, row = int(round(x)), int(round(y))
    if row < 3 or column < 3 or row + 3 >= plane.shape[0] or column + 3 >= plane.shape[1]:
        return None
    box = plane[row - 2:row + 3, column - 2:column + 3]
    if not np.isfinite(box).all():
        return None
    down, across = np.unravel_index(int(np.argmax(box)), box.shape)
    row, column = row - 2 + int(down), column - 2 + int(across)
    near = plane[row - 1:row + 2, column - 1:column + 2].astype(np.float64)
    found = []
    for low, middle, high in ((near[1, 0], near[1, 1], near[1, 2]), (near[0, 1], near[1, 1], near[2, 1])):
        bend = low - 2 * middle + high
        found.append(float(np.clip(0.5 * (low - high) / bend, -0.5, 0.5)) if bend < 0 else 0.0)
    return column + found[0], row + found[1]


def fitted_offsets(stars: np.ndarray) -> dict[str, Any]:
    """A pair's light offsets from its stars' rows (x, y, dx, dy): a plane
    over the field in each axis, clipped at three robust σ twice, where
    there are `LIGHT_PLANE_STARS`, else the medians; its slopes per 1000 px
    about the stars' middle."""
    centre = np.median(stars[:, :2], axis=0)
    if len(stars) < LIGHT_PLANE_STARS:
        return {"kind": "median", "stars": len(stars), "centre": centre.tolist(),
                "dx": [float(np.median(stars[:, 2])), 0.0, 0.0], "dy": [float(np.median(stars[:, 3])), 0.0, 0.0]}
    design = np.column_stack([np.ones(len(stars)), (stars[:, 0] - centre[0]) / 1000, (stars[:, 1] - centre[1]) / 1000])
    found: dict[str, Any] = {"kind": "plane", "stars": len(stars), "centre": centre.tolist()}
    for axis, name in ((2, "dx"), (3, "dy")):
        values = stars[:, axis]
        keep = np.ones(len(values), dtype=bool)
        for _ in range(3):
            coefficients, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
            residual = values - design @ coefficients
            scatter = 1.4826 * float(np.median(np.abs(residual[keep] - np.median(residual[keep]))))
            keep = np.abs(residual) <= 3 * max(scatter, 1e-3)
        found[name] = [float(value) for value in coefficients]
    return found


def light_offsets(bank: Bank) -> dict[int, dict[str, Any]]:
    """Each pair's light offset, target from input (`fitted_offsets`), from
    the isolated stars of all its windows, measured by `light_centre` in
    both within `LIGHT_RADIUS` input widths; kept beside the bank's arrays
    once measured. A pair with fewer than `LIGHT_LEAST` stars has none."""
    path = bank.directory / f"light-offsets-{LIGHT_VERSION}.json"
    if path.exists():
        return {int(source): fit for source, fit in json.loads(path.read_text()).items()}
    rows: dict[int, list[tuple[float, float, float, float]]] = {}
    for window in range(len(bank.info)):
        row = bank.row(window)
        source, width = int(row["source"]), row["input_width"]
        if not width > 0:
            continue
        radius = LIGHT_RADIUS * width
        image = np.asarray(bank.arrays["input"][window], dtype=np.float32)
        target = np.asarray(bank.arrays["target"][window], dtype=np.float32)
        valid = np.isfinite(image) & np.isfinite(target) & ~np.asarray(bank.arrays["saturated"][window]).astype(bool)
        stars = measured_stars(window_points(bank, source, row), np.ones(image.shape, dtype=bool), valid,
                               int(math.ceil(1.5 * radius)) + 1)
        for x, y in stars[:, :2]:
            before, after = light_centre(image, float(x), float(y), radius), light_centre(target, float(x), float(y),
                                                                                           radius)
            if before is not None and after is not None:
                rows.setdefault(source, []).append((float(x) + row["x"], float(y) + row["y"], after[0] - before[0],
                                                    after[1] - before[1]))
    found = {source: fitted_offsets(np.array(stars)) for source, stars in rows.items() if len(stars) >= LIGHT_LEAST}
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(found))
    partial.replace(path)
    return found


def set_names(manifest_path: Path | None) -> dict[str, str]:
    """Each set's name, from the manifest opened for reading alone; none
    where there is no manifest."""
    if manifest_path is None or not Path(manifest_path).exists():
        return {}
    from .manifest import Manifest

    with Manifest.read_only(manifest_path) as manifest:
        return {row["id"]: row["name"] for row in manifest.rows("SELECT id, name FROM sets")}


@dataclass(frozen=True)
class BankChoice:
    """Which banks a run trains and is measured on, and how it reads them:
    every path, the supplied banks' weights, the pairs left out, the cap on
    an object's draws, the manifest whose set names give the objects, and
    whether windows out of focus are left out. A run records it, so a
    resumed run opens its banks as it did."""

    kernel: str
    seeing: tuple[str, ...]
    validation_kernel: str | None
    validation_seeing: tuple[str, ...]
    pairs: tuple[tuple[str, float], ...] = ()
    excluded: tuple[str, ...] = ()
    object_cap: float = 0.03
    manifest: str | None = None
    focus_check: bool = True
    left_out_kinds: tuple[str, ...] = ()
    multi_night_share: float = 0.0
    # Further known-kernel banks, such as a binned one, each with its
    # weight against the first's 1 in the known-kernel share of the mix.
    more_kernels: tuple[tuple[str, float], ...] = ()

    def options(self) -> dict[str, Any]:
        names = set_names(Path(self.manifest) if self.manifest else None)
        return {"excluded": self.excluded, "object_cap": self.object_cap, "set_names": names,
                "focus_check": self.focus_check, "left_out_kinds": self.left_out_kinds,
                "multi_night_share": self.multi_night_share}

    def train(self, settings: Settings) -> dict[str, list[Bank]]:
        """The train banks, by kind."""
        options = self.options()
        found = {"kernel": [Bank(Path(self.kernel), settings, **options),
                            *(Bank(Path(path), settings, **options, weight=weight)
                              for path, weight in self.more_kernels)],
                 "seeing": [Bank(Path(path), settings, **options) for path in self.seeing]}
        if self.pairs:
            found["pairs"] = [Bank(Path(path), settings, **options, weight=weight) for path, weight in self.pairs]
        return found

    def validation(self, settings: Settings) -> dict[str, list[Bank]]:
        """The validation banks, by kind."""
        options = self.options()
        found: dict[str, list[Bank]] = {}
        if self.validation_kernel:
            found["kernel"] = [Bank(Path(self.validation_kernel), settings, **options)]
        if self.validation_seeing:
            found["seeing"] = [Bank(Path(path), settings, **options) for path in self.validation_seeing]
        return found

    @classmethod
    def recorded(cls, values: dict[str, Any]) -> BankChoice:
        return cls(**{key: (tuple(tuple(item) if isinstance(item, list) else item for item in value)
                            if isinstance(value, list) else value) for key, value in values.items()})


# Known-kernel maths.


def moments(stamp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A stamp's centroid, (x, y) from its middle, and its second moments
    about it, as a covariance in (x, y)."""
    values = stamp.astype(np.float64)
    total = values.sum()
    middle = (stamp.shape[0] - 1) / 2
    down, across = np.mgrid[0:stamp.shape[0], 0:stamp.shape[1]] - middle
    x, y = (values * across).sum() / total, (values * down).sum() / total
    dx, dy = across - x, down - y
    covariance = np.array([[(values * dx * dx).sum(), (values * dx * dy).sum()],
                           [(values * dx * dy).sum(), (values * dy * dy).sum()]]) / total
    return np.array([x, y]), covariance


def kernel_covariance(sigma: float, ellipticity: float, angle: float) -> np.ndarray:
    """The covariance, (x, y), of an elliptical Gaussian `sigma` along its
    major axis, `1 − ellipticity` times it along its minor, the major axis
    `angle` from x."""
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return rotation @ np.diag([sigma**2, (sigma * (1.0 - ellipticity)) ** 2]) @ rotation.T


def fourier_gaussian(shape: tuple[int, int], covariance: np.ndarray) -> np.ndarray:
    """The transform, on `rfft2`'s frequencies, of a Gaussian of
    `covariance` in (x, y): `exp(−2π² kᵀΣk)`. A degenerate covariance, flat
    along one axis, blurs along the other alone."""
    ky = np.fft.fftfreq(shape[0])[:, None]
    kx = np.fft.rfftfreq(shape[1])[None, :]
    (xx, xy), (_, yy) = covariance
    return np.exp(-2.0 * math.pi**2 * (xx * kx**2 + 2.0 * xy * kx * ky + yy * ky**2))


def fourier_shift(shape: tuple[int, int], dx: float, dy: float) -> np.ndarray:
    """The transform that moves an image's content by (`dx`, `dy`)."""
    ky = np.fft.fftfreq(shape[0])[:, None]
    kx = np.fft.rfftfreq(shape[1])[None, :]
    return np.exp(-2j * math.pi * (kx * dx + ky * dy))


def filtered(image: np.ndarray, transforms: list[np.ndarray | None], pad: int,
             mode: str = "reflect") -> list[np.ndarray]:
    """`image` through each of `transforms` (None: as it is), from one
    transform of it, padded by `pad` (reflected, or zeros with `mode`
    "constant") so a filter's reach past its edge reads something like it."""
    padded = np.pad(image.astype(np.float32), pad, mode=mode)
    spectrum = np.fft.rfft2(padded)
    found = []
    for transform in transforms:
        if transform is None:
            found.append(image.astype(np.float32))
            continue
        out = np.fft.irfft2(spectrum * transform, s=padded.shape)
        found.append(out[pad:pad + image.shape[0], pad:pad + image.shape[1]].astype(np.float32))
    return found


def fourier_stamp(shape: tuple[int, int], stamp: np.ndarray) -> np.ndarray:
    """The transform, on `rfft2`'s frequencies, of a blur by `stamp`, its
    middle the blur's origin, summing to one."""
    side = stamp.shape[0]
    canvas = np.zeros(shape, dtype=np.float64)
    canvas[:side, :side] = stamp / stamp.sum(dtype=np.float64)
    return np.fft.rfft2(np.roll(canvas, (-(side // 2), -(side // 2)), axis=(0, 1)))


def blur_transform(shape: tuple[int, int], blur: np.ndarray) -> np.ndarray:
    """A blur's transform: of a Gaussian, given its 2 × 2 covariance, or of
    a stamp."""
    return fourier_gaussian(shape, blur) if blur.shape == (2, 2) else fourier_stamp(shape, blur)


def blurred_images(image: np.ndarray, covariances: list[np.ndarray | None], pad: int = WINDOW_PAD) -> list[np.ndarray]:
    """`image` blurred by each of `covariances` (None: as it is), a
    Gaussian's covariance or a stamp (`blur_transform`), reflected past its
    edges."""
    shape = (image.shape[0] + 2 * pad, image.shape[1] + 2 * pad)
    return filtered(image, [None if covariance is None else blur_transform(shape, covariance)
                            for covariance in covariances], pad)


def shifted(image: np.ndarray, dx: float, dy: float, pad: int = 16) -> np.ndarray:
    """`image` with its content moved by (`dx`, `dy`) pixels, through its
    transform, reflected past its edges."""
    shape = (image.shape[0] + 2 * pad, image.shape[1] + 2 * pad)
    return filtered(image, [fourier_shift(shape, dx, dy)], pad)[0]


def blurred_stamp(stamp: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """A stamp blurred by a Gaussian of `covariance`, on a canvas wide
    enough not to wrap, cut back to its side and summing to one again."""
    pad = (STAMP_CANVAS - stamp.shape[0]) // 2
    out = filtered(stamp, [blur_transform((stamp.shape[0] + 2 * pad,) * 2, covariance)], pad, mode="constant")[0]
    total = float(out.sum(dtype=np.float64))
    return (out / total).astype(np.float32) if total > 0 else out


def positive(covariance: np.ndarray) -> np.ndarray:
    """A symmetric matrix with its negative eigenvalues, which only rounding
    leaves, set to zero."""
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    return (vectors * np.clip(values, 0.0, None)) @ vectors.T


# A rounding filter's gain is held to this at most, where the PSF's
# transform is all but nothing, and its transform's own floor.
ROUNDING_GAIN = 2.0
ROUNDING_FLOOR = 1e-3
# The azimuthal profile is weighed over this many pixels of radius at the
# middle, out to `ROUNDING_REACH`.
ROUNDING_RING = 0.2
ROUNDING_REACH = 40
# The rounding filter reads this far past a pixel, at most, for the loss.
ROUNDING_BAND = 12


def azimuthal(stamp: np.ndarray) -> np.ndarray:
    """A stamp averaged about its middle: at each radius, the mean of its
    pixels weighed by a Gaussian in radius, `ROUNDING_RING` px wide at the
    middle and a tenth of the radius wider, so that a ring always holds
    pixels at every angle; read back at every pixel's own radius and
    summing to what it did. Past `ROUNDING_REACH` px, where little is, the
    weights are whole rings."""
    middle = (stamp.shape[0] - 1) / 2
    down, across = np.mgrid[0:stamp.shape[0], 0:stamp.shape[1]] - middle
    radius = np.hypot(across, down).ravel()
    values = stamp.astype(np.float64).ravel()
    near = radius <= ROUNDING_REACH + 4
    centres = np.arange(0.0, ROUNDING_REACH + 0.01, 0.25)
    spread = ROUNDING_RING + 0.1 * centres
    weights = np.exp(-0.5 * ((radius[near][None, :] - centres[:, None]) / spread[:, None]) ** 2)
    profile = (weights @ values[near]) / weights.sum(axis=1)
    rings = np.rint(radius).astype(int)
    counts = np.maximum(np.bincount(rings), 1)
    far_values = np.bincount(rings, values) / counts
    far_radii = np.arange(len(far_values), dtype=np.float64)
    far = far_radii > ROUNDING_REACH
    out = np.interp(radius, np.concatenate([centres, far_radii[far]]),
                    np.concatenate([profile, far_values[far]])).reshape(stamp.shape)
    # The core is left as it is: its few pixels sit at few radii, and a
    # mean over radius would sharpen or blur it. From one width out to two
    # the mean takes over.
    width = stamp_width(stamp)
    if width > 0:
        share = np.clip(radius.reshape(stamp.shape) / width - 1.0, 0.0, 1.0)
        out = share * out + (1.0 - share) * stamp
    total = out.sum()
    return (out * (float(stamp.sum(dtype=np.float64)) / total)).astype(np.float32) if total > 0 else stamp


def rounding(stamp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The filter, as a stamp to blur by, that takes a PSF `stamp` to its
    azimuthal mean, and that mean: the ratio of their transforms, held to
    `ROUNDING_GAIN`, and one where the PSF's own is under `ROUNDING_FLOOR`
    of its largest."""
    side = stamp.shape[0]
    pad = (STAMP_CANVAS - side) // 2
    canvas = (side + 2 * pad,) * 2
    round_stamp = azimuthal(stamp)
    before = np.fft.rfft2(np.pad(stamp.astype(np.float64), pad))
    after = np.fft.rfft2(np.pad(round_stamp.astype(np.float64), pad))
    size = np.abs(before)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(size > ROUNDING_FLOOR * size.max(), after / before, 1.0)
    gain = np.abs(ratio)
    ratio = np.where(gain > ROUNDING_GAIN, ratio * (ROUNDING_GAIN / np.maximum(gain, 1e-12)), ratio)
    kernel = np.fft.fftshift(np.fft.irfft2(ratio, s=canvas))
    centre = canvas[0] // 2
    half = side // 2
    kernel = kernel[centre - half:centre + half + 1, centre - half:centre + half + 1]
    return (kernel / kernel.sum()).astype(np.float32), round_stamp


def rounded(stamp: np.ndarray, floor_px: float) -> tuple[np.ndarray, np.ndarray, float]:
    """The blur that brings a stamp to a PSF round in its second moments,
    `Σ_T − Σ_ref` with `Σ_T` round at the stamp's largest eigenvalue, or as
    much wider as keeps its width at `floor_px` or more; the blurred stamp;
    and its width."""
    _, covariance = moments(stamp)
    variance = float(np.linalg.eigvalsh(covariance)[-1])
    for _ in range(KERNEL_ITERATIONS):
        delta = positive(variance * np.eye(2) - covariance)
        target = blurred_stamp(stamp, delta)
        width = stamp_width(target)
        if width >= floor_px:
            break
        # Widths a pixel or two across add in quadrature only roughly, so
        # each step asks a little more than they say.
        variance += 1.1 * ((floor_px / FWHM_PER_SIGMA) ** 2 - (width / FWHM_PER_SIGMA) ** 2) + 1e-3
    return delta, target, width


def widened(stamp: np.ndarray, reference: float, asked: float, ellipticity: float,
            angle: float) -> tuple[np.ndarray, np.ndarray, float]:
    """An elliptical Gaussian K, of `ellipticity` at `angle`, that makes
    `stamp ∗ K` `asked` times as wide as the stamp's width `reference`, as
    `stamp_width` measures both; its covariance, the blurred stamp and its
    width. Its size starts from Gaussians' widths adding in quadrature and
    is refined until the ratio is within `RATIO_TOLERANCE`."""
    sigma = reference * math.sqrt(asked**2 - 1.0) / (FWHM_PER_SIGMA * math.sqrt(1.0 - ellipticity))
    for _ in range(KERNEL_ITERATIONS):
        covariance = kernel_covariance(sigma, ellipticity, angle)
        blurred = blurred_stamp(stamp, covariance)
        width = stamp_width(blurred)
        got = width / reference
        if abs(got - asked) <= RATIO_TOLERANCE * asked:
            break
        sigma *= float(np.clip(math.sqrt((asked**2 - 1.0) / max(got**2 - 1.0, 1e-6)), 0.5, 2.0))
    return covariance, blurred, width


@dataclass
class Kernel:
    """A known-kernel window's draw: K's covariance (zero for an identity
    window), the target's blur `Σ_T − Σ_ref`, the stamp the network reads
    (P_in), the ratio FWHM(T) / FWHM(P_in), the stamps' widths, and the
    widening asked, FWHM(P_in) / FWHM(P_ref)."""

    covariance: np.ndarray
    target: np.ndarray
    stamp: np.ndarray
    ratio: float
    widths: dict[str, float]
    asked: float
    # A real PSF that blurs in K's place, where one does.
    donor: np.ndarray | None = None
    # The filter that makes the target's PSF round about its middle
    # (`rounding`), where the settings ask it.
    rounding: np.ndarray | None = None

    @property
    def identity(self) -> bool:
        return self.donor is None and not self.covariance.any()

    @property
    def blur(self) -> np.ndarray:
        """What blurs the window into its input: the donor, or K."""
        return self.covariance if self.donor is None else self.donor

    def reach(self) -> int:
        """How far the input's or the target's blur reads, in pixels."""
        largest = max(float(np.linalg.eigvalsh(self.covariance)[-1]), float(np.linalg.eigvalsh(self.target)[-1]))
        found = int(math.ceil(BLUR_REACH * math.sqrt(max(largest, 0.0))))
        if self.donor is not None:
            # A real PSF's wings run far past its core: three of its widths.
            found = max(found, int(math.ceil(3 * stamp_width(self.donor))))
        return min(found, WINDOW_PAD)


def donor_kernel(stamp: np.ndarray, donor: np.ndarray, settings: Settings) -> Kernel | None:
    """A known-kernel window's draw with a real PSF, `donor`, for K: None
    where it widens the stamp by less or more than `kernel_ratios` allow,
    or leaves the target wider than `max_ratio` of the input."""
    reference = stamp_width(stamp)
    blurred = blurred_stamp(stamp, donor)
    width = stamp_width(blurred)
    delta, target, target_width = rounded(stamp, settings.floor_px)
    made = None
    if settings.round_target:
        made, target = rounding(target)
        target_width = stamp_width(target)
    asked = width / reference
    low, high = settings.kernel_ratios
    if not low <= asked <= high or target_width > settings.max_ratio * width:
        return None
    return Kernel(np.zeros((2, 2)), delta, blurred, target_width / width,
                  {"reference": reference, "input": width, "target": target_width}, asked, donor, made)


def known_kernel(stamp: np.ndarray, rng: np.random.Generator, settings: Settings, *, identity: bool = False) -> Kernel:
    """Draw a known-kernel window's K for its PSF stamp, and the target's
    blur (see the module notes). The widening is drawn from the settings'
    `kernel_ratios`, from no less than keeps FWHM(T) / FWHM(P_in) within
    `max_ratio`: T is wider than P_ref where the stamp's second moments are
    elliptical."""
    reference = stamp_width(stamp)
    if identity:
        zero = np.zeros((2, 2))
        return Kernel(zero, zero, stamp.astype(np.float32), 1.0,
                      {"reference": reference, "input": reference, "target": reference}, 1.0)
    delta, target, target_width = rounded(stamp, settings.floor_px)
    made = None
    if settings.round_target:
        made, target = rounding(target)
        target_width = stamp_width(target)
    low, high = settings.kernel_ratios
    low = max(low, target_width / (reference * settings.max_ratio))
    asked = float(rng.uniform(low, max(low, high)))
    ellipticity = float(rng.uniform(0.0, settings.kernel_ellipticity))
    angle = float(rng.uniform(0.0, math.pi))
    covariance, blurred, width = widened(stamp, reference, asked, ellipticity, angle)
    return Kernel(covariance, delta, blurred, target_width / width,
                  {"reference": reference, "input": width, "target": target_width}, asked, None, made)


# Examples.


@dataclass
class Example:
    """One window as the network meets it, before it is cut to a tile: the
    input and target in the image's units, the noise map the network reads
    (made from the input alone) and the target's own noise for the loss,
    where the loss may look, the stamp and ratio the network reads, the
    input's point sources in the window's coordinates, the background and
    median noise the network's units are, and the input PSF's width."""

    image: np.ndarray
    target: np.ndarray
    noise_map: np.ndarray
    sigma: np.ndarray
    valid: np.ndarray
    stamp: np.ndarray
    ratio: float
    points: np.ndarray
    background: float
    noise: float
    width: float
    kind: str
    source: int
    window: int
    asked: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    # The ratio asked of what is not a star, where it is not `ratio`.
    nonstellar_ratio: float | None = None

    def ratios(self, settings: Settings) -> float | tuple[float, float]:
        """The ratio planes' values the settings' network reads."""
        if not settings.nonstellar:
            return self.ratio
        return self.ratio, self.ratio if self.nonstellar_ratio is None else self.nonstellar_ratio


def stack_noise(sigma: np.ndarray, image: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """A stack's own noise map, where it knows the noise: none where one
    frame alone covers the sky, whose variance is zero, and two C925 deep
    stacks have such patches. There the map made from the image alone
    (`fallback`, or `image_sigma` of `image`) stands in."""
    known = np.isfinite(sigma) & (sigma > 0)
    if known.all():
        return sigma.astype(np.float32)
    stand_in = image_sigma(image) if fallback is None else fallback
    return np.where(known, sigma, stand_in).astype(np.float32)


def edge_band(shape: tuple[int, int], band: int) -> np.ndarray:
    """True but within `band` pixels of the edges."""
    inside = np.zeros(shape, dtype=bool)
    if 2 * band < min(shape):
        inside[band:shape[0] - band, band:shape[1] - band] = True
    return inside


def window_points(bank: Bank, source: int, row: dict[str, float]) -> np.ndarray:
    """A source's point sources in a window's coordinates."""
    points = bank.points(source).copy()
    points[:, 0] -= row["x"]
    points[:, 1] -= row["y"]
    return points


def kernel_example(bank: Bank, window: int, rng: np.random.Generator, settings: Settings, *,
                   identity: bool | None = None, scaled: bool = True) -> Example:
    """A deep stack's window as a known-kernel pair (see the module notes);
    an identity window by the settings' share, or as `identity` says. With
    `scaled`, its signal is scaled by up to `exposure` either way."""
    row = bank.row(window)
    source = int(row["source"])
    image = np.asarray(bank.arrays["image"][window], dtype=np.float32)
    sigma = stack_noise(np.asarray(bank.arrays["sigma"][window], dtype=np.float32), image)
    saturated = np.asarray(bank.arrays["saturated"][window]).astype(bool)
    stamp = np.asarray(bank.arrays["stamp"][window], dtype=np.float32)
    background = row["background"]
    if identity is None:
        identity = bool(rng.random() < settings.identity)
    kernel = None
    if not identity and settings.real_kernels > 0 and rng.random() < settings.real_kernels:
        kernel = donor_kernel(stamp, np.asarray(bank.arrays["stamp"][bank.draw(rng)], dtype=np.float32), settings)
    if kernel is None:
        kernel = known_kernel(stamp, rng, settings, identity=identity)
    scale = math.exp(rng.uniform(-1.0, 1.0) * math.log(settings.exposure)) if scaled and settings.exposure > 1 else 1.0
    share = float(rng.uniform(0.0, settings.kernel_noise))
    signal = background + scale * (image - background)
    points = window_points(bank, source, row)
    points[:, 2] *= scale
    points[:, 3] = background + scale * (points[:, 3] - background)
    level = (bank.sources[source].get("input_psf") or {}).get("clip_level")
    clip = background + scale * (level - background) if settings.fill_clipped and level is not None else None
    if clip is not None and not identity and saturated.any():
        signal = filled(signal, saturated, points, stamp, row["width"], background)
    ratio, nonstellar_ratio, band = kernel.ratio, None, kernel.reach()
    if identity:
        blurred, target = signal, None
    elif settings.nonstellar:
        # Stars and the rest each asked for their own share of the input's
        # width, no sharper than the stack allows and nearer that than 1
        # (a quarter of the way up at the median): the target is the two
        # blurs of the window, each where its part of the light is.
        def drawn() -> float:
            return kernel.ratio + (1.0 - kernel.ratio) * float(rng.random()) ** 2

        ratio = drawn()
        nonstellar_ratio = ratio if rng.random() < settings.equal_share else drawn()
        blurs = [target_blur(stamp, kernel, value) for value in (ratio, nonstellar_ratio)]
        blurred, stars, rest = blurred_images(signal, [kernel.blur, *blurs])
        rendered = point_map(points, stamp, image.shape, background)
        with np.errstate(divide="ignore", invalid="ignore"):
            mask = np.clip(np.nan_to_num(rendered / (rendered + STAR_MASK_NOISE * scale * sigma)), 0.0, 1.0)
        target = (mask * stars + (1.0 - mask) * rest).astype(np.float32)
        band = max(band, *(blur_reach(blur) for blur in blurs))
    else:
        blurred, target = blurred_images(signal, [kernel.blur, kernel.target])
    fresh = rng.standard_normal(image.shape).astype(np.float32)
    if settings.pass_noise:
        # Shared with its neighbours by a blur up to `NOISE_BLUR` px, at the
        # spread it had.
        spread_px = float(rng.uniform(0.0, NOISE_BLUR))
        fresh = blurred_images(fresh, [spread_px**2 * np.eye(2)], pad=8)[0]
        fresh /= max(float(fresh.std()), 1e-6)
    fresh = fresh * (share * sigma)
    shown = blurred + fresh
    if target is not None and kernel.rounding is not None:
        target = blurred_images(target, [kernel.rounding])[0]
        band = max(band, ROUNDING_BAND)
    if target is None:
        target, target_sigma = shown, math.hypot(scale, share) * sigma
    else:
        target_sigma = scale * sigma
        if settings.pass_noise:
            target = target + fresh
    # The clipped cores stay out of the loss, but not the pixels about them:
    # input and target are blurs of the same window there, and agree.
    valid = edge_band(image.shape, band) & ~saturated
    if clip is not None:
        shown, target = np.minimum(shown, clip), np.minimum(target, clip)
        valid &= ~grown((shown >= clip) | (target >= clip), SATURATION_GROW)
    noise_map = image_sigma(shown)
    noise = float(np.median(noise_map[np.isfinite(noise_map)]))
    return Example(shown, target, noise_map, target_sigma.astype(np.float32), valid, kernel.stamp, ratio,
                   points, background, noise, kernel.widths["input"], "identity" if identity else "kernel", source,
                   window, kernel.asked, {"scale": scale, "noise_share": share, "widths": kernel.widths,
                                          "kernel": kernel.covariance.tolist(), "sharpest": kernel.ratio,
                                          "real": kernel.donor is not None},
                   nonstellar_ratio)


def filled(image: np.ndarray, saturated: np.ndarray, points: np.ndarray, stamp: np.ndarray, width: float,
           background: float) -> np.ndarray:
    """`image` with each clipped core put back: the stamp at the middle of
    the core's clipped pixels, scaled to the unclipped pixels within
    `FILL_REACH` widths, less the median of those out to half as far again,
    wherever it is brighter than what the stack kept."""
    columns = psf_fields.DETECTION_COLUMNS
    cores = points[(points[:, columns.index("clipped")] == 1) | (points[:, columns.index("saturated")] == 1)]
    out = image.astype(np.float32).copy()
    reach = int(math.ceil(1.5 * FILL_REACH * width)) + 2
    height, side = image.shape
    for star in cores:
        column, row = int(round(float(star[0]))), int(round(float(star[1])))
        top, bottom = max(0, row - reach), min(height, row + reach + 1)
        left, right = max(0, column - reach), min(side, column + reach + 1)
        if bottom <= top or right <= left:
            continue
        box = (slice(top, bottom), slice(left, right))
        down, across = np.mgrid[top:bottom, left:right]
        near = saturated[box] & (np.hypot(across - star[0], down - star[1]) <= FILL_REACH * width)
        if not near.any():
            continue
        # The middle of the plateau the detector found, where a flat top
        # gives it no peak.
        x, y = float(across[near].mean()), float(down[near].mean())
        distance = np.hypot(across - x, down - y)
        values = out[box]
        finite = np.isfinite(values)
        ring = ~saturated[box] & finite & (distance <= FILL_REACH * width)
        sky = ~saturated[box] & finite & (distance > FILL_REACH * width) & (distance <= 1.5 * FILL_REACH * width)
        if ring.sum() < 8 or sky.sum() < 8:
            continue
        level = float(np.median(values[sky]))
        # The place within `FILL_SHIFT` of the plateau's middle whose stamp,
        # scaled, best fits the unclipped ring: a sharp core's steep sides
        # change much with a fraction of a pixel.
        best = None
        for dy in np.arange(-FILL_SHIFT, FILL_SHIFT + 1e-6, FILL_STEP):
            for dx in np.arange(-FILL_SHIFT, FILL_SHIFT + 1e-6, FILL_STEP):
                model = placed(stamp, values.shape, x - left + dx, y - top + dy)
                weight = float((model[ring] ** 2).sum())
                if not weight > 0:
                    continue
                scale = float((model[ring] * (values[ring] - level)).sum()) / weight
                misfit = float(((values[ring] - level - scale * model[ring]) ** 2).sum())
                if best is None or misfit < best[0]:
                    best = (misfit, scale, model)
        if best is None or not best[1] > 0:
            continue
        _, amplitude, model = best
        core = saturated[box] & finite
        values[core] = np.maximum(values[core], level + amplitude * model[core])
        out[box] = values
    return out


def placed(stamp: np.ndarray, shape: tuple[int, int], x: float, y: float) -> np.ndarray:
    """The stamp with its middle at (`x`, `y`) of a region of `shape`, cut
    to it, moved the last fraction of a pixel through its transform, which
    keeps a sharp stamp's width as a split over four pixels does not."""
    column, row = int(round(x)), int(round(y))
    half = stamp.shape[0] // 2
    canvas = np.zeros(shape, dtype=np.float32)
    top, left = row - half, column - half
    rows = slice(max(0, top), min(shape[0], top + stamp.shape[0]))
    columns = slice(max(0, left), min(shape[1], left + stamp.shape[1]))
    canvas[rows, columns] = stamp[rows.start - top:rows.stop - top, columns.start - left:columns.stop - left]
    return shifted(canvas, x - column, y - row)


def blur_reach(covariance: np.ndarray) -> int:
    """How far a Gaussian blur of `covariance` reads, in pixels."""
    return int(math.ceil(BLUR_REACH * math.sqrt(max(float(np.linalg.eigvalsh(covariance)[-1]), 0.0))))


def target_blur(stamp: np.ndarray, kernel: Kernel, ratio: float) -> np.ndarray:
    """The target's blur for a known-kernel window asked for `ratio` of its
    input's width, no less than the sharpest its stack allows
    (`kernel.ratio`): the blur that rounds the stamp, and as much more round
    Gaussian as brings the rounded stamp to that width."""
    if ratio <= kernel.ratio * (1 + RATIO_TOLERANCE):
        return kernel.target
    extra, _, _ = widened(blurred_stamp(stamp, kernel.target), kernel.widths["target"], ratio / kernel.ratio, 0.0, 0.0)
    return kernel.target + extra


def pair_example(bank: Bank, window: int, settings: Settings) -> Example:
    """A pair's window: the input and the target, moved back by the pair's
    offset at the window as the settings measure it, the input's image-only
    noise map and stamp, and the bank's ratio. The loss weighs by the target
    stack's noise, or the input stack's where the bank keeps none for its
    target."""
    row = bank.row(window)
    source = int(row["source"])
    planes = {name: np.asarray(bank.arrays[name][window]) for name in
              ("input", "target", "input_sigma", "input_map", "target_sigma")}
    target = planes["target"].astype(np.float32)
    band = 0
    offset = bank.target_offset(window, settings.centroid_offset)
    passed = settings.pass_noise and bool(bank.clean)
    if passed:
        target = np.asarray(bank.clean["target_clean"][window], dtype=np.float32)
    if offset is not None:
        target = shifted(target, -offset[0], -offset[1])
        band = SHIFT_BAND
    if passed:
        # The target's denoised, and the noise the denoiser found in the
        # input: what the model is to leave where it is.
        target = target + (planes["input"].astype(np.float32)
                           - np.asarray(bank.clean["input_clean"][window], dtype=np.float32))
    sigma = stack_noise(planes["target_sigma"], planes["target"],
                        stack_noise(planes["input_sigma"], planes["input"], planes["input_map"]))
    saturated = np.asarray(bank.arrays["saturated"][window]).astype(bool)
    valid = edge_band(target.shape, band) & ~saturated & np.isfinite(sigma)
    return Example(planes["input"].astype(np.float32), target, planes["input_map"].astype(np.float32),
                   sigma.astype(np.float32), valid, np.asarray(bank.arrays["input_stamp"][window], dtype=np.float32),
                   row[RATIO_COLUMNS[settings.ratio_source]], window_points(bank, source, row), row["background"],
                   row["noise"], row["input_width"], "pairs", source, window, None,
                   {"offset": offset, "target_width": row["target_width"]})


def example_from(bank: Bank, window: int, rng: np.random.Generator, settings: Settings) -> Example:
    return pair_example(bank, window, settings) if bank.pairs else kernel_example(bank, window, rng, settings)


# What the network reads.


def stamp_planes(stamp: np.ndarray) -> np.ndarray:
    """The planes the network reads of a stamp (`STAMP_PLANES`): the stamp
    over its peak, and its asinh, which shows the wings."""
    peak = float(stamp.max())
    scaled = stamp / peak if peak > 0 else np.zeros_like(stamp)
    return np.stack([scaled, np.arcsinh(scaled / WINGS) / math.asinh(1.0 / WINGS)]).astype(np.float32)


def point_fluxes(points: np.ndarray, stamp: np.ndarray, background: float) -> np.ndarray:
    """Each point source's flux: the detector's, or, for a clipped core it
    turned away, its peak over the stamp's, which a clipped peak makes too
    low but places where it is."""
    columns = psf_fields.DETECTION_COLUMNS
    flux = points[:, columns.index("flux")].astype(np.float64)
    peak = points[:, columns.index("peak")].astype(np.float64)
    estimate = (peak - background) / max(float(stamp.max()), 1e-12)
    return np.where(np.isfinite(flux), flux, np.clip(estimate, 0.0, None))


def point_map(points: np.ndarray, stamp: np.ndarray, shape: tuple[int, int], background: float) -> np.ndarray:
    """The point sources rendered with `stamp` over a region of `shape`
    whose corner is the points' origin, each at its flux and sub-pixel
    place, in the image's units above the background."""
    pad = stamp.shape[0] // 2
    canvas_shape = (shape[0] + 2 * pad, shape[1] + 2 * pad)
    canvas = np.zeros(canvas_shape, dtype=np.float64)
    if len(points):
        flux = point_fluxes(points, stamp, background)
        x, y = points[:, 0] + pad, points[:, 1] + pad
        inside = (x >= 0) & (y >= 0) & (x < canvas_shape[1] - 1) & (y < canvas_shape[0] - 1) & np.isfinite(flux)
        x, y, flux = x[inside], y[inside], flux[inside]
        left, top = np.floor(x).astype(int), np.floor(y).astype(int)
        across, down = x - left, y - top
        for dy, dx, share in ((0, 0, (1 - down) * (1 - across)), (0, 1, (1 - down) * across),
                              (1, 0, down * (1 - across)), (1, 1, down * across)):
            np.add.at(canvas, (top + dy, left + dx), flux * share)
    if not canvas.any():
        return np.zeros(shape, dtype=np.float32)
    kernel = np.zeros(canvas_shape, dtype=np.float64)
    kernel[:stamp.shape[0], :stamp.shape[1]] = stamp
    kernel = np.roll(kernel, (-pad, -pad), axis=(0, 1))
    rendered = np.fft.irfft2(np.fft.rfft2(canvas) * np.fft.rfft2(kernel), s=canvas_shape)
    return rendered[pad:pad + shape[0], pad:pad + shape[1]].astype(np.float32)


def network_planes(image: np.ndarray, noise_map: np.ndarray, ratio: Any, points: np.ndarray,
                   background: float, noise: float) -> np.ndarray:
    """The planes the network reads (`PLANES`, or `PLANES_NONSTELLAR` given
    a pair of ratios, the stars' and the rest's): the image and the
    point-source map as `asinh` in units of the median noise above the
    background, the noise map as `asinh(σ / σ̄)`, and each ratio, a number
    or a map; NaN becomes the background."""
    x = np.arcsinh(np.nan_to_num((image - background) / noise, nan=0.0))
    s = np.arcsinh(np.nan_to_num(noise_map / noise, nan=1.0))
    ratios = ratio if isinstance(ratio, tuple) else (ratio,)
    r = [np.broadcast_to(np.asarray(value, dtype=np.float32), image.shape) for value in ratios]
    p = np.arcsinh(points / noise)
    return np.stack([x, s, *r, p]).astype(np.float32)


def apertures(points: np.ndarray, valid: np.ndarray, radius: float, limit: int = APERTURES) -> np.ndarray:
    """A label for each pixel within `radius` of a point source whose flux
    the loss weighs: unclipped detections at `FLUX_SNR` or more, brightest
    first, each wholly inside the region and on valid pixels and meeting no
    other's aperture; 0 elsewhere."""
    columns = psf_fields.DETECTION_COLUMNS
    labels = np.zeros(valid.shape, dtype=np.int64)
    if not len(points) or not radius > 0:
        return labels
    usable = ((points[:, columns.index("detected")] == 1) & (points[:, columns.index("clipped")] == 0)
              & (points[:, columns.index("saturated")] == 0) & np.isfinite(points[:, columns.index("flux")])
              & (points[:, columns.index("snr")] >= FLUX_SNR))
    chosen = points[usable]
    chosen = chosen[np.argsort(-chosen[:, columns.index("flux")], kind="stable")]
    reach_px = int(math.ceil(radius))
    offsets = np.arange(-reach_px, reach_px + 1)
    disk = (offsets[:, None] ** 2 + offsets[None, :] ** 2) <= radius**2
    height, width = valid.shape
    count = 0
    for x, y in chosen[:, :2]:
        column, row = int(round(float(x))), int(round(float(y)))
        if row - reach_px < 0 or column - reach_px < 0 or row + reach_px >= height or column + reach_px >= width:
            continue
        box = (slice(row - reach_px, row + reach_px + 1), slice(column - reach_px, column + reach_px + 1))
        if not valid[box][disk].all() or labels[box][disk].any():
            continue
        count += 1
        labels[box][disk] = count
        if count == limit:
            break
    return labels


def arrays_of(example: Example, rng: np.random.Generator | None, settings: Settings) -> dict[str, np.ndarray]:
    """An example as a training tile: cut at a random place in its window,
    its point-source map rendered (empty by the settings' share), its
    apertures labelled, and everything flipped and turned one of eight ways
    alike, stamp included. Without `rng` the whole window, as it is, with
    its points."""
    side = example.image.shape[0]
    tile = min(settings.tile, side) if rng is not None else side
    top, left = (int(value) for value in rng.integers(0, side - tile + 1, 2)) if rng is not None else (0, 0)
    cut = (slice(top, top + tile), slice(left, left + tile))
    points = example.points.copy()
    points[:, 0] -= left
    points[:, 1] -= top
    empty = rng is not None and rng.random() < settings.points_empty
    rendered = np.zeros((tile, tile), np.float32) if empty else point_map(points, example.stamp, (tile, tile),
                                                                          example.background)
    planes = network_planes(example.image[cut], example.noise_map[cut], example.ratios(settings), rendered,
                            example.background, example.noise)
    valid = example.valid[cut]
    found = {
        "planes": planes,
        "stamp": stamp_planes(example.stamp),
        "reference": ((example.image[cut] - example.background) / example.noise)[None],
        "target": ((example.target[cut] - example.background) / example.noise)[None],
        "sigma": (example.sigma[cut] / example.noise)[None],
        "valid": valid[None].astype(np.float32),
        "labels": apertures(points, valid, APERTURE_FWHM * example.width),
    }
    if rng is not None:
        turns, flip = int(rng.integers(0, 4)), bool(rng.integers(0, 2))
        for name, values in found.items():
            values = np.rot90(values, turns, axes=(-2, -1))
            found[name] = np.ascontiguousarray(values[..., ::-1] if flip else values)
    return found


def batch_from(banks: list[Bank], shares: np.ndarray, size: int, rng: np.random.Generator,
               settings: Settings) -> dict[str, torch.Tensor]:
    """`size` tiles, each from a bank drawn by `shares` and a window drawn
    by the bank's odds, as tensors: the planes, the stamps, the input and
    target in σ̄ above the background, the target's noise in σ̄, where the
    loss may look, and the apertures."""
    cumulative = np.cumsum(shares)
    examples = []
    for _ in range(size):
        bank = banks[min(int(np.searchsorted(cumulative, rng.random() * cumulative[-1])), len(banks) - 1)]
        examples.append(arrays_of(example_from(bank, bank.draw(rng), rng, settings), rng, settings))
    return {name: torch.from_numpy(np.stack([example[name] for example in examples])) for name in examples[0]}


# Batches built in worker processes. Threads build a batch no faster past
# four: blurs, maps and fills hold the interpreter's lock between their
# NumPy calls, so on the box four threads gave 11 batches a second and 32
# gave 7, while the network waited (2026-11-09). Each process opens the
# banks once and builds whole batches; a batch depends on its step alone,
# so they are the batches the loop would build itself.
WORKER: dict[str, Any] = {}


def worker_start(choice: BankChoice, settings: Settings, mix: tuple[float, float, float], size: int,
                 seed: int) -> None:
    """A worker's banks, opened once."""
    torch.set_num_threads(1)
    sources, shares = mixture(choice.train(settings), mix)
    WORKER.update(sources=sources, shares=shares, size=size, seed=seed, settings=settings)


def worker_batch(step: int) -> dict[str, np.ndarray]:
    """A step's batch as arrays, its masks and labels as bytes where they
    fit, to cross to the training process."""
    tensors = batch_from(WORKER["sources"], WORKER["shares"], WORKER["size"], step_generator(WORKER["seed"], step),
                         WORKER["settings"])
    arrays = {name: value.numpy() for name, value in tensors.items()}
    arrays["valid"] = arrays["valid"].astype(np.uint8)
    if arrays["labels"].max(initial=0) < 256:
        arrays["labels"] = arrays["labels"].astype(np.uint8)
    return arrays


class Workers:
    """Steps `start` + 1, `start` + 2, …'s batches, built up to `depth`
    ahead in `processes` worker processes and handed out in step order, as
    `Prefetch` hands out its threads' (`worker_batch`)."""

    def __init__(self, choice: BankChoice, settings: Settings, mix: tuple[float, float, float], size: int,
                 seed: int, *, depth: int, processes: int, start: int = 0) -> None:
        import multiprocessing
        import os
        from collections import deque
        from concurrent.futures import ProcessPoolExecutor

        # Each worker is one thread of NumPy and Torch: twelve of them
        # should not each start a pool of their own.
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(variable, "1")
        self.pool = ProcessPoolExecutor(processes, mp_context=multiprocessing.get_context("spawn"),
                                        initializer=worker_start, initargs=(choice, settings, mix, size, seed))
        self.pending: deque[Any] = deque()
        self.step = start
        for _ in range(max(depth, processes)):
            self.submit()

    def submit(self) -> None:
        self.step += 1
        self.pending.append(self.pool.submit(worker_batch, self.step))

    def get(self) -> dict[str, torch.Tensor]:
        """The next step's batch as the loop builds it, or the error that
        building it raised."""
        arrays = self.pending.popleft().result()
        self.submit()
        arrays["valid"] = arrays["valid"].astype(np.float32)
        arrays["labels"] = arrays["labels"].astype(np.int64)
        return {name: torch.from_numpy(value) for name, value in arrays.items()}

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


# The loss.


def applied(reference: Any, residual: Any, mode: str) -> Any:
    """The output, in σ̄ above the background, of an input `reference` in
    the same units and the network's `residual`, as `mode` says
    (`RESIDUALS`): for tensors or arrays alike."""
    if mode == "linear":
        return reference + residual
    if isinstance(reference, torch.Tensor):
        return torch.sinh(torch.asinh(reference) + residual)
    return np.sinh(np.arcsinh(reference) + residual)


def charbonnier(values: torch.Tensor, epsilon: float) -> torch.Tensor:
    return torch.sqrt(values**2 + epsilon**2) - epsilon


def loss_terms(residual: torch.Tensor, batch: dict[str, torch.Tensor], settings: LossSettings,
               mode: str = "asinh") -> dict[str, torch.Tensor]:
    """The loss's terms for a batch's residuals (see the module notes) and
    their weighted sum, `total`, the residual applied as `mode` says
    (`applied`). Errors are in units of the target's noise
    and set to zero where the loss may not look before any is averaged,
    so a clipped core cannot reach a block's mean.

    Reblur consistency on real images without a target, and a centroid
    shift at the point sources, would join these."""
    output = applied(batch["reference"], residual.float(), mode)
    valid = batch["valid"]
    sigma = batch["sigma"].clamp(min=SIGMA_FLOOR)
    if settings.noise_cap > 0:
        typical = sigma.flatten(1).median(dim=1).values.view(-1, 1, 1, 1)
        sigma = torch.minimum(sigma, settings.noise_cap * typical)
    error = (output - batch["target"]) / sigma * valid
    epsilon = settings.epsilon
    pixel = (charbonnier(error, epsilon) * valid).sum() / valid.sum().clamp(min=1.0)
    terms = []
    for scale in settings.gradient_scales:
        if scale == 1:
            blocks, inside = error, valid
        else:
            blocks = scale * torch.nn.functional.avg_pool2d(error, scale)
            inside = -torch.nn.functional.max_pool2d(-valid, scale)
        across = blocks[..., :, 1:] - blocks[..., :, :-1]
        down = blocks[..., 1:, :] - blocks[..., :-1, :]
        inside_across = inside[..., :, 1:] * inside[..., :, :-1]
        inside_down = inside[..., 1:, :] * inside[..., :-1, :]
        weight = (inside_across.sum() + inside_down.sum()).clamp(min=1.0)
        terms.append(((charbonnier(across, epsilon) * inside_across).sum()
                      + (charbonnier(down, epsilon) * inside_down).sum()) / weight)
    gradient = torch.stack(terms).mean() if terms else torch.zeros((), device=residual.device)
    labels = batch["labels"].reshape(len(residual), -1)
    sums = torch.zeros(len(residual), APERTURES + 1, device=residual.device)
    variances, areas = torch.zeros_like(sums), torch.zeros_like(sums)
    sums.scatter_add_(1, labels, (output - batch["target"]).reshape(len(residual), -1))
    variances.scatter_add_(1, labels, (sigma**2).reshape(len(residual), -1))
    areas.scatter_add_(1, labels, torch.ones_like(sigma).reshape(len(residual), -1))
    scores = sums[:, 1:] / variances[:, 1:].clamp(min=1e-12).sqrt()
    # Each star's error counts once for each pixel of its aperture, over
    # the valid pixels, as a pixel's does: as a mean over the stars, a
    # tenth of it pulled each aperture's pixels a hundred times harder
    # than the pixel term pulls its own, and no star was ever sharpened.
    flux = (charbonnier(scores, epsilon) * areas[:, 1:]).sum() / valid.sum().clamp(min=1.0)
    total = pixel + settings.gradient_weight * gradient + settings.flux_weight * flux
    return {"total": total, "pixel": pixel, "gradient": gradient, "flux": flux}


# Training.


def build_model(width: int, levels: int, planes: tuple[str, ...] = PLANES) -> SharpeningUNet:
    return SharpeningUNet(inputs=len(planes), width=width, levels=levels, stamp_inputs=len(STAMP_PLANES))


def run_directory(store: Store, name: str) -> Path:
    return store.root / "models" / "sharpen" / name


def mixture(train_banks: dict[str, list[Bank]], mix: tuple[float, float, float]) -> tuple[list[Bank], np.ndarray]:
    """The banks a batch draws from and each one's share: the known-kernel
    bank's, the seeing banks' and the supplied banks' shares of `mix`, the
    supplied share shared among their banks by their weights and folded into
    the seeing share where none is given."""
    kernel, seeing, supplied = mix
    if not train_banks.get("pairs"):
        seeing, supplied = seeing + supplied, 0.0
    banks, shares = [], []
    for kind, share in (("kernel", kernel), ("seeing", seeing), ("pairs", supplied)):
        group = train_banks.get(kind) or []
        total = sum(bank.weight for bank in group)
        for bank in group:
            if share > 0:
                banks.append(bank)
                shares.append(share * bank.weight / total)
    if not banks:
        raise ValueError("no bank to draw from")
    shares_array = np.array(shares, dtype=np.float64)
    return banks, shares_array / shares_array.sum()


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Write a checkpoint whole, then move it into place."""
    torch.save(state, path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)


def train(
    store: Store,
    *,
    name: str,
    choice: BankChoice,
    steps: int,
    width: int = 32,
    levels: int = 4,
    batch: int = 8,
    learning_rate: float = 2e-4,
    seed: int = 0,
    log_every: int = 250,
    checkpoint_every: int = 1000,
    settings: Settings = Settings(),
    loss: LossSettings = LossSettings(),
    mix: tuple[float, float, float] = (0.6, 0.25, 0.15),
    clip_grad: float = 1.0,
    rewind: float = 2.0,
    precision: str = "fp32",
    validation_batches: int = 4,
    validation_windows: int = 200,
    resume: bool = False,
    stop_after: int | None = None,
    prefetch: int = 8,
    batch_threads: int = 4,
    workers: int = 0,
    where: torch.device | None = None,
) -> dict[str, Any]:
    """Train a PSF model on the banks `choice` names, the known-kernel,
    seeing and supplied banks mixed by `mix`, record it under
    `STORE/models/sharpen/<name>/` and measure it on the validation banks.

    Every `log_every` steps the window's mean loss is logged, with the
    loss on `validation_batches` fixed batches from the validation banks.
    `rewind` goes back two log windows when one's loss is more than that
    many times the lowest (see `Rewind`; 0: never), and `clip_grad` limits
    each step's gradient norm (0: never). Every `checkpoint_every` steps,
    at the end of a log window, the run is saved whole; with `resume`, a
    run goes on from its last checkpoint and is the run it would have been
    unstopped, as each batch depends on its step alone (`resume_run` reads
    the configuration the run recorded). `stop_after` stops a run with a
    checkpoint after that step, to be resumed later. At the end the
    weights, the ONNX graph and the record (`run.json`) are written, with
    `evaluate`'s measures on `validation_windows` windows of each
    validation bank. `prefetch` batches are built ahead on `batch_threads`
    threads, or with `workers`, in that many worker processes (`Workers`),
    which leaves them as they are."""
    if precision not in ("fp32", "bf16"):
        raise ValueError(f"precision is fp32 or bf16, not {precision!r}")
    if checkpoint_every % log_every:
        raise ValueError(f"checkpoints ({checkpoint_every}) fall at the end of log windows ({log_every})")
    if settings.tile % 2 ** (levels - 1):
        raise ValueError(f"a tile of {settings.tile} is not a multiple of {2 ** (levels - 1)}")
    directory = run_directory(store, name)
    checkpoint_path = directory / "checkpoint.pt"
    # Everything that decides the run, which a resumed run must match.
    call = json.loads(json.dumps({
        "steps": steps, "width": width, "levels": levels, "batch": batch, "learning_rate": learning_rate,
        "seed": seed, "log_every": log_every, "checkpoint_every": checkpoint_every, "settings": asdict(settings),
        "loss": asdict(loss), "mix": list(mix), "clip_grad": clip_grad, "rewind": rewind, "precision": precision,
        "validation_batches": validation_batches, "validation_windows": validation_windows,
        "choice": asdict(choice)}))
    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"{name} has no checkpoint to go on from")
        recorded = json.loads((directory / "config.json").read_text())["call"]
        changed = sorted(key for key in set(recorded) | set(call) if recorded.get(key) != call.get(key))
        if changed:
            raise ValueError(f"{name} was recorded with another {', '.join(changed)}: resume it as it was")
    elif checkpoint_path.exists() or (directory / "run.json").exists():
        raise FileExistsError(f"a run named {name} is already recorded: resume it, or name another")
    banks, validation_banks = choice.train(settings), choice.validation(settings)
    config = {**call, "planes": list(settings.planes), "stamp_planes": list(STAMP_PLANES), "wings": WINGS,
              "banks": {kind: [bank.summary() for bank in group] for kind, group in banks.items()},
              "validation_banks": {kind: [bank.summary() for bank in group]
                                   for kind, group in validation_banks.items()}}
    if not resume:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(json.dumps({"call": call, **config}, indent=2))
    torch.manual_seed(seed)
    where = where or device()
    cuda = where.type == "cuda"
    model = build_model(width, levels, settings.planes).to(where)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=steps, pct_start=0.05)
    guard = Rewind(model, optimizer, rewind) if rewind else None
    history: list[dict[str, Any]] = []
    start, seconds = 0, 0.0
    if resume:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        if guard is not None:
            guard.restore(state["rewind"])
        history, start, seconds = state["history"], state["step"], state["seconds"]
        print(f"train {name}: resumed after step {start}", flush=True)
    sources, shares = mixture(banks, mix)
    checks = []
    if validation_batches and validation_banks:
        checked, checked_shares = mixture(validation_banks, mix)
        checks = [batch_from(checked, checked_shares, batch, np.random.default_rng([seed, VALIDATION_STREAM, index]),
                             settings) for index in range(validation_batches)]

    def assemble(step: int) -> dict[str, torch.Tensor]:
        tensors = batch_from(sources, shares, batch, step_generator(seed, step), settings)
        return {key: value.pin_memory() for key, value in tensors.items()} if cuda else tensors

    def validation_loss() -> dict[str, float]:
        totals: dict[str, float] = {}
        with torch.no_grad():
            for tensors in checks:
                moved = {key: value.to(where) for key, value in tensors.items()}
                terms = loss_terms(model(moved["planes"], moved["stamp"]), moved, loss, settings.residual)
                for key, value in terms.items():
                    totals[key] = totals.get(key, 0.0) + float(value) / len(checks)
        return totals

    last = min(steps, stop_after) if stop_after else steps
    if workers:
        source: Any = Workers(choice, settings, mix, batch, seed, depth=max(prefetch, 2 * workers), processes=workers,
                              start=start)
    else:
        source = Prefetch(assemble, prefetch, batch_threads, start=start) if prefetch else None
    began = time.monotonic() - seconds
    running = torch.zeros(4, device=where)
    norms = torch.zeros(2, device=where)
    try:
        for step in range(start + 1, last + 1):
            tensors = source.get() if source else assemble(step)
            moved = {key: value.to(where, non_blocking=True) for key, value in tensors.items()}
            with torch.autocast(where.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                residual = model(moved["planes"], moved["stamp"])
            terms = loss_terms(residual, moved, loss, settings.residual)
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            if clip_grad:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                norms[0] += norm
                norms[1] = torch.maximum(norms[1], norm)
            optimizer.step()
            schedule.step()
            if guard and guard.rate != 1.0:
                for group in optimizer.param_groups:
                    group["lr"] *= guard.rate
            running += torch.stack([terms[key].detach().float() for key in ("total", "pixel", "gradient", "flux")])
            if step % log_every == 0 or step == last:
                window = log_every if step % log_every == 0 else step % log_every
                means = (running / window).tolist()
                entry = {"step": step, "loss": means[0], "pixel": means[1], "gradient": means[2], "flux": means[3],
                         "seconds": round(time.monotonic() - began, 1)}
                noted = ""
                if clip_grad:
                    entry["grad_norm_mean"], entry["grad_norm_max"] = float(norms[0]) / window, float(norms[1])
                    noted = f" gradient {entry['grad_norm_mean']:.3g} (largest {entry['grad_norm_max']:.3g})"
                    norms.zero_()
                if checks:
                    entry["validation"] = validation_loss()
                    noted += f", validation {entry['validation']['total']:.4f}"
                back = guard.after(step, means[0]) if guard else None
                if back is not None:
                    entry["rewound_to"] = back
                    noted += f", back to step {back}"
                history.append(entry)
                print(f"train {name}: step {step} loss {means[0]:.4f}{noted} ({entry['seconds']:.0f} s)", flush=True)
                running.zero_()
                if step % checkpoint_every == 0 or step == last:
                    save_checkpoint(checkpoint_path, {
                        "step": step, "model": offloaded(model.state_dict()),
                        "optimizer": offloaded(optimizer.state_dict()), "schedule": schedule.state_dict(),
                        "rewind": guard.state() if guard else None, "history": history,
                        "seconds": time.monotonic() - began})
    finally:
        if source:
            source.close()
    if last < steps:
        return {"name": name, "stopped_after": last, "history": history}
    model = model.cpu().eval()
    torch.save(model.state_dict(), directory / "weights.pt")
    train_seconds = time.monotonic() - began
    validation = evaluate(model, validation_banks, settings=settings, limit=validation_windows, seed=seed,
                          where=where) if validation_windows else None
    exported = export(model, directory / "model.onnx", tile=256)
    record = {
        "name": name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {**config, "prefetch": prefetch, "batch_threads": batch_threads, "workers": workers},
        "parameters": sum(p.numel() for p in model.parameters()),
        "receptive_field_px": receptive_field(levels),
        "inference": tiling(levels),
        "device": str(where),
        "seconds": {"train": round(train_seconds, 1)},
        "history": history,
        "validation": validation,
        "export": exported,
        "versions": versions(),
    }
    (directory / "run.json").write_text(json.dumps(record, indent=2))
    checkpoint_path.unlink(missing_ok=True)
    return record


def resume_run(store: Store, name: str, *, stop_after: int | None = None, prefetch: int = 8,
               batch_threads: int = 4, workers: int = 0, where: torch.device | None = None) -> dict[str, Any]:
    """Go on with a run from its last checkpoint, as its `config.json`
    records it, on `where` (default: the training device), which should be
    the device it ran on for the run to be the same."""
    call = json.loads((run_directory(store, name) / "config.json").read_text())["call"]
    arguments = {key: value for key, value in call.items() if key not in ("settings", "loss", "choice", "mix")}
    return train(store, name=name, choice=BankChoice.recorded(call["choice"]), settings=settings_from(call["settings"]),
                 loss=LossSettings(**{key: tuple(value) if isinstance(value, list) else value
                                      for key, value in call["loss"].items()}),
                 mix=tuple(call["mix"]), resume=True, stop_after=stop_after, prefetch=prefetch,
                 batch_threads=batch_threads, workers=workers, where=where, **arguments)


def export(model: SharpeningUNet, path: Path, *, tile: int) -> dict[str, Any]:
    """Write the model as ONNX at a static tile shape, reading the planes
    and the stamp, and hold ONNX Runtime's CPU output to PyTorch's on a
    random tile."""
    import onnxruntime

    model = model.cpu().eval()
    planes = torch.randn(1, model.first[0].in_channels, tile, tile)
    stamp = torch.rand(1, len(STAMP_PLANES), STAMP, STAMP)
    torch.onnx.export(model, (planes, stamp), str(path), input_names=["planes", "stamp"], output_names=["residual"],
                      opset_version=17, dynamo=False)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        expected = model(planes, stamp).numpy()
    got = session.run(None, {"planes": planes.numpy(), "stamp": stamp.numpy()})[0]
    scale = float(np.abs(expected).max()) or 1.0
    return {"path": str(path), "opset": 17, "tile": tile,
            "max_relative_error": float(np.abs(got - expected).max() / scale), "bytes": path.stat().st_size}


# Inference.


def node_features(model: SharpeningUNet, stamps: np.ndarray, where: torch.device, chunk: int = 64) -> np.ndarray:
    """The stamp encoder's vector of each of a PSF field's node stamps,
    (rows, columns, features)."""
    rows, columns = stamps.shape[:2]
    flat = np.stack([stamp_planes(stamp) for stamp in stamps.reshape(rows * columns, *stamps.shape[2:])])
    with torch.no_grad():
        found = [model.encoder(torch.from_numpy(flat[start:start + chunk]).to(where)).cpu().numpy()
                 for start in range(0, len(flat), chunk)]
    return np.concatenate(found)[:, :, 0, 0].reshape(rows, columns, -1)


def interpolating(nodes: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """The weights, (positions, nodes), that interpolate values at `nodes`
    linearly at `positions`, held at the ends."""
    return np.stack([np.interp(positions, nodes, column) for column in np.eye(len(nodes))], axis=1)


def sampled_features(features: np.ndarray, ys: np.ndarray, xs: np.ndarray, top: int, left: int, side: int,
                     multiple: int) -> np.ndarray:
    """The node vectors interpolated bilinearly at the middles of a tile's
    bottleneck pixels, (features, side / multiple, side / multiple), for a
    tile whose corner is at (`left`, `top`) in the nodes' coordinates."""
    middles = (np.arange(side // multiple) + 0.5) * multiple - 0.5
    return np.einsum("ri,ijf,cj->frc", interpolating(ys, top + middles), features,
                     interpolating(xs, left + middles)).astype(np.float32)


def field_point_map(points: np.ndarray, ys: np.ndarray, xs: np.ndarray, stamps: np.ndarray, shape: tuple[int, int],
                    background: float) -> np.ndarray:
    """The point sources rendered over an image of `shape`, each with the
    stamp of the PSF field's node nearest it (`point_map`)."""
    out = np.zeros(shape, dtype=np.float32)
    if not len(points):
        return out
    pad = stamps.shape[-1] // 2 + 1
    nearest_row = np.abs(points[:, 1:2] - ys[None, :]).argmin(axis=1)
    nearest_column = np.abs(points[:, 0:1] - xs[None, :]).argmin(axis=1)
    for row, column in sorted(set(zip(nearest_row.tolist(), nearest_column.tolist()))):
        chosen = points[(nearest_row == row) & (nearest_column == column)]
        top = max(0, int(math.floor(float(chosen[:, 1].min()))) - pad)
        left = max(0, int(math.floor(float(chosen[:, 0].min()))) - pad)
        bottom = min(shape[0], int(math.ceil(float(chosen[:, 1].max()))) + pad + 1)
        right = min(shape[1], int(math.ceil(float(chosen[:, 0].max()))) + pad + 1)
        if bottom <= top or right <= left:
            continue
        local = chosen.copy()
        local[:, 0] -= left
        local[:, 1] -= top
        out[top:bottom, left:right] += point_map(local, stamps[row, column], (bottom - top, right - left), background)
    return out


def sharpen(model: SharpeningUNet, image: np.ndarray, stamp_at: Any, *, ratio: float | np.ndarray,
            points: np.ndarray | None = None, noise_map: np.ndarray | None = None, background: float | None = None,
            noise: float | None = None, inference: dict[str, int] | None = None, residual: str = "asinh",
            where: torch.device | None = None,
            grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Sharpen a whole image: its overlapping tiles, each conditioned on
    `stamp_at(x, y)` at its centre and with its point sources rendered by
    that stamp, blended as the noise model's are. Given a PSF field's
    `grid`, (node rows, node columns, node stamps of `STAMP` pixels) in the
    image's coordinates, `stamp_at` is not read: the nodes' vectors are
    interpolated to every bottleneck pixel (`SharpeningUNet.mapped`) and
    the point sources rendered once with their nearest node's stamp, so
    the output does not depend on where the tiles fall. `ratio` is a
    number or a map. The noise map is made from the image unless given,
    and the background and median noise from the two unless given.
    `points` are the image's point sources in
    `psf_fields.DETECTION_COLUMNS` (none: an empty map). The tiles'
    residuals are blended, then applied as `residual` says (`applied`).
    NaN stays NaN."""
    noise_map = image_sigma(image) if noise_map is None else noise_map
    finite = np.isfinite(image) & np.isfinite(noise_map)
    if background is None:
        background = float(np.median(image[finite][::17]))
    if noise is None:
        noise = float(np.median(noise_map[finite][::17]))
    settings = inference or tiling(model.levels)
    tile = settings["tile"]
    where = where or torch.device("cpu")
    model = model.to(where).eval()
    found = np.zeros((0, len(psf_fields.DETECTION_COLUMNS)), np.float32) if points is None else points
    height, width = image.shape
    if grid is not None:
        ys, xs, stamps = grid
        features = node_features(model, stamps, where)
        planes = network_planes(image, noise_map, ratio, field_point_map(found, ys, xs, stamps, image.shape,
                                                                         background), background, noise)

        def mapped(tiles: np.ndarray, origins: list[tuple[int, int]]) -> np.ndarray:
            vectors = np.stack([sampled_features(features, ys, xs, top, left, tile, model.multiple)
                                for top, left in origins])
            with torch.no_grad():
                return model.mapped(torch.from_numpy(np.ascontiguousarray(tiles)).to(where),
                                    torch.from_numpy(vectors).to(where)).cpu().numpy()

        found_residual = infer(mapped, planes, tile=tile, overlap=settings["overlap"], margin=settings["margin"],
                               multiple=settings.get("multiple", model.multiple), origins_too=True)[0]
        output = background + noise * applied(np.nan_to_num((image - background) / noise), found_residual, residual)
        return np.where(np.isfinite(image), output, np.nan).astype(np.float32)
    planes = network_planes(image, noise_map, ratio, np.zeros(image.shape, np.float32), background, noise)[:-1]

    def run(tiles: np.ndarray, origins: list[tuple[int, int]]) -> np.ndarray:
        stamps, rendered = [], []
        for top, left in origins:
            stamp = np.asarray(stamp_at(min(left + (tile - 1) / 2, width - 1), min(top + (tile - 1) / 2, height - 1)),
                               dtype=np.float32)
            stamps.append(stamp_planes(stamp))
            local = found.copy()
            local[:, 0] -= left
            local[:, 1] -= top
            rendered.append(np.arcsinh(point_map(local, stamp, (tile, tile), background) / noise))
        batch = np.concatenate([tiles, np.stack(rendered)[:, None]], axis=1)
        with torch.no_grad():
            return model(torch.from_numpy(np.ascontiguousarray(batch)).to(where),
                         torch.from_numpy(np.stack(stamps)).to(where)).cpu().numpy()

    found_residual = infer(run, planes, tile=tile, overlap=settings["overlap"], margin=settings["margin"],
                           multiple=settings.get("multiple", model.multiple), origins_too=True)[0]
    output = background + noise * applied(np.nan_to_num((image - background) / noise), found_residual, residual)
    return np.where(np.isfinite(image), output, np.nan).astype(np.float32)


# Evaluation.

# Stars measured for their width, flux and rings: unclipped detections at
# this signal to noise or more, with no neighbour brighter than
# `NEIGHBOUR_SHARE` of their flux within the box measured.
MEASURE_SNR = 20.0
NEIGHBOUR_SHARE = 0.1
# Rings are looked for about stars whose peak is this many σ̄ above the
# background.
RING_PEAK = 50.0


def empirical_stamp(image: np.ndarray, stars: np.ndarray, side: int) -> np.ndarray | None:
    """The mean of the stars' boxes of `side` pixels, each less the median
    of its border, moved through its transform so the star sits on the
    box's middle, and scaled to sum to one: the stars' own stamp, which
    `stamp_width` measures as it measures the field's."""
    half = side // 2
    pad = 4
    found = []
    for x, y in stars[:, :2]:
        column, row = int(round(float(x))), int(round(float(y)))
        box = image[row - half - pad:row + half + pad + 1, column - half - pad:column + half + pad + 1]
        if box.shape != (side + 2 * pad, side + 2 * pad) or not np.isfinite(box).all():
            continue
        border = np.concatenate([box[:2].ravel(), box[-2:].ravel(), box[:, :2].ravel(), box[:, -2:].ravel()])
        box = box - float(np.median(border))
        moved = filtered(box, [fourier_shift(box.shape, column - float(x), row - float(y))], 0)[0]
        cut = moved[pad:pad + side, pad:pad + side].astype(np.float64)
        total = cut.sum()
        if total > 0:
            found.append(cut / total)
    return np.mean(found, axis=0) if found else None


# An empirical stamp's width is fitted within this many input widths of its
# middle, past the core's 5% level of any profile with β of 2.5 or more,
# so a noisy pixel far out cannot join the core.
WIDTH_REACH = 1.75


def measured_width(stamp: np.ndarray | None, width: float) -> float:
    """An empirical stamp's width, as `stamp_width` measures the field's,
    fitted within `WIDTH_REACH` of the input's `width`; NaN where its peak
    is not at its middle, as when the stars are lost."""
    if stamp is None:
        return float("nan")
    middle = (stamp.shape[0] - 1) / 2
    down, across = np.mgrid[0:stamp.shape[0], 0:stamp.shape[1]] - middle
    distance = np.hypot(across, down)
    peak = np.unravel_index(int(np.argmax(stamp)), stamp.shape)
    if distance[peak] > max(2.0, 0.5 * width):
        return float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        return stamp_width(np.where(distance <= WIDTH_REACH * width, stamp, 0.0))


def measured_stars(points: np.ndarray, inner: np.ndarray, valid: np.ndarray, radius: int) -> np.ndarray:
    """The point sources a measure within `radius` of them may use:
    unclipped detections at `MEASURE_SNR` or more, whose box of `radius`
    lies on the inner, valid region, with no neighbour within it brighter
    than `NEIGHBOUR_SHARE` of their flux (a clipped core always is)."""
    columns = psf_fields.DETECTION_COLUMNS
    flux = points[:, columns.index("flux")]
    height, width = inner.shape
    with np.errstate(invalid="ignore"):
        column, row = np.rint(points[:, 0]), np.rint(points[:, 1])
        usable = ((points[:, columns.index("detected")] == 1) & (points[:, columns.index("clipped")] == 0)
                  & (points[:, columns.index("saturated")] == 0) & (points[:, columns.index("snr")] >= MEASURE_SNR)
                  & np.isfinite(flux) & (column - radius >= 0) & (row - radius >= 0)
                  & (column + radius < width) & (row + radius < height))
    kept = []
    for index in np.nonzero(usable)[0]:
        x, y = float(points[index, 0]), float(points[index, 1])
        column, row = int(round(x)), int(round(y))
        if row - radius < 0 or column - radius < 0 or row + radius >= height or column + radius >= width:
            continue
        box = (slice(row - radius, row + radius + 1), slice(column - radius, column + radius + 1))
        if not (inner[box].all() and valid[box].all()):
            continue
        near = (np.hypot(points[:, 0] - x, points[:, 1] - y) <= radius) & ~(
            np.nan_to_num(flux, nan=np.inf) <= NEIGHBOUR_SHARE * flux[index])
        near[index] = False
        if near.any():
            continue
        kept.append(index)
    return points[kept]


def ring_and_flux(image: np.ndarray, reference: np.ndarray, x: float, y: float, width: float,
                  noise: float) -> dict[str, float]:
    """About one star: the flux within `APERTURE_FWHM` widths above the
    background of a thin annulus just outside it, of `image` and of
    `reference`, and the deepest dip of `image` below that background in a
    ring 1 to 3 widths out, in σ̄, as the lowest of its azimuthal means a
    pixel wide and as its lowest pixel."""
    radius = APERTURE_FWHM * width
    reach_px = int(math.ceil(radius + 3))
    column, row = int(round(x)), int(round(y))
    box = (slice(row - reach_px, row + reach_px + 1), slice(column - reach_px, column + reach_px + 1))
    offsets = np.arange(-reach_px, reach_px + 1)
    distance = np.hypot(offsets[:, None] + row - y, offsets[None, :] + column - x)
    annulus = (distance > radius) & (distance <= radius + 3)
    aperture = distance <= radius
    found: dict[str, float] = {}
    for label, plane in (("", image), ("reference_", reference)):
        values = plane[box]
        level = float(np.median(values[annulus]))
        found[f"{label}flux"] = float((values[aperture] - level).sum())
        if label:
            continue
        ring = (distance >= width) & (distance <= 3 * width)
        means = [float(values[ring & (np.abs(distance - r) < 0.5)].mean()) for r in np.arange(math.ceil(width),
                                                                                                  3 * width + 0.5)
                 if (ring & (np.abs(distance - r) < 0.5)).any()]
        found["ring_profile"] = (min(means) - level) / noise if means else float("nan")
        found["ring_pixel"] = float(values[ring].min() - level) / noise
    return found


def window_measures(example: Example, output: np.ndarray, band: int,
                    noise_cap: float = LossSettings.noise_cap) -> dict[str, Any]:
    """One window's measures: the error before and after against the target
    in its own noise, read as the loss reads it (`noise_cap`), and in σ̄;
    and, at its stars, the widths of the input's, the output's and the
    target's own stamps, the flux the output keeps of the input's and of
    the target's, and the rings about bright stars."""
    inner = edge_band(example.image.shape, band)
    look = inner & example.valid
    sigma = np.maximum(example.sigma, SIGMA_FLOOR * example.noise)
    if noise_cap > 0:
        sigma = np.minimum(sigma, noise_cap * float(np.median(sigma)))
    before = (example.image - example.target) / sigma
    after = (output - example.target) / sigma
    found: dict[str, Any] = {
        "kind": example.kind, "source": example.source, "window": example.window, "ratio": example.ratio,
        "asked": example.asked,
        "error_before": float(np.sqrt(np.mean(before[look] ** 2))) if look.any() else None,
        "error_after": float(np.sqrt(np.mean(after[look] ** 2))) if look.any() else None,
        "mse_before": float(np.mean(((example.image - example.target) / example.noise)[look] ** 2)) if look.any()
        else None,
        "mse_after": float(np.mean(((output - example.target) / example.noise)[look] ** 2)) if look.any() else None,
    }
    if example.kind == "identity":
        found["change"] = float(np.sqrt(np.mean(((output - example.image) / example.noise)[look] ** 2)))
    side = int(min(65, max(9, 2 * math.ceil(2.5 * example.width) + 1)))
    stars = measured_stars(example.points, inner, example.valid, side // 2 + 5)
    found["stars"] = int(len(stars))
    if len(stars):
        widths = {}
        for label, plane in (("input", example.image), ("output", output), ("target", example.target)):
            widths[label] = measured_width(empirical_stamp(plane, stars, side), example.width)
        found["widths"] = widths
        found["achieved"] = widths["output"] / widths["input"]
        found["target_achieved"] = widths["target"] / widths["input"]
    stars = measured_stars(example.points, inner, example.valid, int(math.ceil(APERTURE_FWHM * example.width)) + 4)
    if len(stars):
        rows = [ring_and_flux(output, example.image, float(x), float(y), example.width, example.noise)
                for x, y in stars[:, :2]]
        targets = [ring_and_flux(example.target, example.image, float(x), float(y), example.width, example.noise)
                   for x, y in stars[:, :2]]
        kept = [row["flux"] / row["reference_flux"] for row in rows if row["reference_flux"] > 0]
        against = [row["flux"] / target["flux"] for row, target in zip(rows, targets) if target["flux"] > 0]
        found["flux_kept"] = kept
        found["flux_against_target"] = against
        columns = psf_fields.DETECTION_COLUMNS
        bright = (stars[:, columns.index("peak")] - example.background) / example.noise >= RING_PEAK
        found["rings"] = [{"output": row["ring_profile"], "output_pixel": row["ring_pixel"],
                           "target": target["ring_profile"]}
                          for row, target, chosen in zip(rows, targets, bright) if chosen]
    found["moved"] = star_moves(example, output, inner)
    return found


def star_moves(example: Example, output: np.ndarray, inner: np.ndarray) -> list[dict[str, Any]]:
    """How far each isolated star moved from the input to the output, (x,
    y): its light-weighted centre within `LIGHT_RADIUS` input widths, its
    peak, and how far the target's light sits from the input's."""
    radius = LIGHT_RADIUS * example.width
    stars = measured_stars(example.points, inner, example.valid, int(math.ceil(1.5 * radius)) + 1)
    moves = []
    for x, y in stars[:, :2]:
        x, y = float(x), float(y)
        light = [light_centre(plane, x, y, radius) for plane in (example.image, output, example.target)]
        peak = [peak_centre(plane, x, y) for plane in (example.image, output)]
        if light[0] is None or light[1] is None:
            continue
        moves.append({"light": [light[1][0] - light[0][0], light[1][1] - light[0][1]],
                      "peak": [peak[1][0] - peak[0][0], peak[1][1] - peak[0][1]]
                      if peak[0] is not None and peak[1] is not None else None,
                      "target_light": [light[2][0] - light[0][0], light[2][1] - light[0][1]]
                      if light[2] is not None else None})
    return moves


def summarized(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """A bank's windows' measures, pooled."""

    def mean(key: str) -> float | None:
        values = [row[key] for row in rows if row.get(key) is not None]
        return float(np.mean(values)) if values else None

    def spread(values: list[float]) -> list[float] | None:
        values = [value for value in values if np.isfinite(value)]
        return [round(float(np.percentile(values, q)), 4) for q in (5, 50, 95)] if values else None

    found: dict[str, Any] = {"windows": len(rows), "error_before": mean("error_before"),
                             "error_after": mean("error_after"), "mse_before": mean("mse_before"),
                             "mse_after": mean("mse_after")}
    if found["error_before"]:
        found["error_ratio"] = found["error_after"] / found["error_before"]
    measured = [row for row in rows if row.get("achieved") is not None and np.isfinite(row["achieved"])]
    found["windows_with_stars"] = len(measured)
    if measured:
        asked = [row["achieved"] / row["ratio"] for row in measured]
        found["achieved_over_asked"] = spread(asked)
        found["achieved_within_10pc"] = float(np.mean([abs(value - 1) <= 0.1 for value in asked]))
        found["target_over_asked"] = spread([row["target_achieved"] / row["ratio"] for row in measured])
    kept = [value for row in rows for value in row.get("flux_kept", [])]
    if kept:
        found["flux_kept"] = spread(kept)
        found["flux_kept_within_1pc"] = float(np.mean([abs(value - 1) <= 0.01 for value in kept]))
        found["flux_against_target"] = spread([value for row in rows for value in row.get("flux_against_target", [])])
        found["stars"] = len(kept)
    rings = [ring for row in rows for ring in row.get("rings", [])]
    if rings:
        found["ring_output"] = spread([ring["output"] for ring in rings])
        found["ring_output_worst"] = float(np.nanmin([ring["output"] for ring in rings]))
        found["ring_output_pixel"] = spread([ring["output_pixel"] for ring in rings])
        found["ring_target"] = spread([ring["target"] for ring in rings])
        found["rings_deeper_than_1"] = float(np.mean([ring["output"] < -1.0 for ring in rings]))
        found["bright_stars"] = len(rings)
    changes = [row["change"] for row in rows if row.get("change") is not None]
    if changes:
        found["identity_change"] = spread(changes)
    moves = [move for row in rows for move in row.get("moved", [])]
    for key in ("light", "peak", "target_light"):
        shifts = np.array([move[key] for move in moves if move.get(key) is not None])
        if len(shifts):
            found[f"moved_{key}_px"] = spread(list(np.hypot(shifts[:, 0], shifts[:, 1])))
            found[f"moved_{key}_median"] = [round(float(np.median(shifts[:, axis])), 4) for axis in (0, 1)]
    if moves:
        found["moved_stars"] = len(moves)
    return found


def chosen_windows(bank: Bank, limit: int | None) -> np.ndarray:
    """Up to `limit` of a bank's kept windows, spread evenly over it, so
    each source has its share."""
    if limit is None or limit >= len(bank.kept):
        return bank.kept
    return bank.kept[np.unique(np.linspace(0, len(bank.kept) - 1, limit).round().astype(int))]


def evaluate(model: SharpeningUNet, banks: dict[str, list[Bank]], *, settings: Settings = Settings(),
             limit: int | None = 200, seed: int = 0, identity_share: float = 0.25,
             where: torch.device | None = None) -> dict[str, Any]:
    """Measure a model on up to `limit` windows of each bank, whole, through
    `sharpen`: the known-kernel windows each with a K drawn for it alone
    (seeded by the window), without exposure scaling, and `identity_share`
    of them again as identity windows; the pairs as training reads them. A
    window's measures are taken inside the network's reach of its edge and
    the band its blur or shift read past (see `window_measures`)."""
    where = where or device()
    model = model.eval()
    found: dict[str, Any] = {"limit": limit, "seed": seed}
    for kind, group in banks.items():
        for bank in group:
            rows, identity_rows = [], []
            windows = chosen_windows(bank, limit)
            for order, window in enumerate(windows):
                rng = np.random.default_rng([seed, int(window)])
                if bank.pairs:
                    examples = [pair_example(bank, int(window), settings)]
                else:
                    examples = [kernel_example(bank, int(window), rng, settings, identity=False, scaled=False)]
                    if order < identity_share * len(windows):
                        examples.append(kernel_example(bank, int(window), rng, settings, identity=True, scaled=False))
                for example in examples:
                    # A window is one tile: whole-image inference on it.
                    inference = {**tiling(model.levels), "tile": example.image.shape[0]}
                    output = sharpen(model, example.image, lambda x, y, stamp=example.stamp: stamp,
                                     ratio=example.ratios(settings), points=example.points,
                                     noise_map=example.noise_map,
                                     background=example.background, noise=example.noise, inference=inference,
                                     residual=settings.residual, where=where)
                    # The band its blur or shift read past is out of
                    # `valid` already.
                    measures = window_measures(example, output, reach(model.levels))
                    (identity_rows if example.kind == "identity" else rows).append(measures)
            entry = {"bank": bank.summary(), **summarized(rows)}
            if identity_rows:
                entry["identity"] = summarized(identity_rows)
            entry["windows_detail"] = rows + identity_rows
            found[f"{kind}:{bank.label}"] = entry
    return found


def load_run(store: Store, name: str) -> tuple[dict[str, Any], SharpeningUNet]:
    """A recorded run's record and its model, on the CPU."""
    directory = run_directory(store, name)
    record = json.loads((directory / "run.json").read_text())
    model = build_model(record["config"]["width"], record["config"]["levels"],
                        tuple(record["config"].get("planes", PLANES)))
    model.load_state_dict(torch.load(directory / "weights.pt", map_location="cpu"))
    record["inference"] = tiling(record["config"]["levels"])
    return record, model.eval()


def evaluate_saved(store: Store, name: str, banks: dict[str, list[Bank]], *, limit: int | None, seed: int = 0,
                   label: str | None = None) -> dict[str, Any]:
    """Measure a recorded run's model on `banks` and add the measures to
    its record, under `label` (default: the banks' labels and the limit).
    Returns them without the per-window detail."""
    directory = run_directory(store, name)
    record, model = load_run(store, name)
    settings = settings_from(record["config"]["settings"])
    result = evaluate(model, banks, settings=settings, limit=limit, seed=seed)
    label = label or ", ".join([*(f"{kind}:{bank.label}" for kind, group in banks.items() for bank in group),
                                f"limit {limit}"])
    record.setdefault("evaluations", {})[label] = result
    (directory / "run.json").write_text(json.dumps(record, indent=2))
    return {key: ({inner: value for inner, value in entry.items() if inner != "windows_detail"}
                  if isinstance(entry, dict) else entry) for key, entry in result.items()}


def settings_from(recorded: dict[str, Any]) -> Settings:
    """A run's settings, as its record keeps them."""
    return replace(Settings(), **{key: tuple(value) if isinstance(value, list) else value
                                  for key, value in recorded.items()})
