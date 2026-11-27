"""Known-truth composites for the starless model, and their bank
(`docs/design/ml-star-removal.md`).

A composite is a stack's clean sky (`starless_sky`) with stars put back:

- as many as the clean-sky pass took out of it, up to half again or down to
  half, at random places, their magnitudes drawn from those it took out and
  their light from the stack's own magnitude scale, so the field's crowding
  and brightnesses are its own;
- all of one telescope's shape: the stack's own PSF field (its recorded or
  measured stamps), or a pupil's (`starless_stars`) of random vanes,
  obstruction and spikes, blurred to the stack's stars' width;
- the brightest with halos: wide wings holding a share of their light, and
  some an off-centre reflection ring;
- clipped where the stack clips, and with their own photon noise, at the
  stack's photon term.

Its truth: the starless image is the clean sky, the stars layer the
composite less it (the stars' light as the image shows it, clipped, with
their photon noise), so the two add back to the input exactly. The sky
noise stays with the starless image.

So that a network can be trained to heal what the catalogue fit leaves, each
composite is also fitted as the runtime fits an image: its stars, listed at
the places they were put (as a catalogue lists them), fitted with the PSF
field measured on the composite itself and taken out (`starless_sky
.clean_sky`). The fit's starless estimate and stars model are kept beside
the truth, and a plane marking where the stars are, so both ways the plan
compares can be trained from one bank: a network that heals the fit's
estimate, and one that reads the image and the places alone.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import starless_sky
from . import starless_stars as stars
from .store import Store

# Raised with the bank's layout or what it holds, so a bank of another is cut
# anew. 3: the fit's estimate carries bright stars past their stamps, fainter
# made halos and rings, the halos' and spikes' own light, on clean skies of
# version 3. 4: stars are moved into place by Lanczos-3, not by Fourier
# transform (`add_star`).
BANK_VERSION = 4
# A bank's planes, a window each: the composite; its starless truth; the
# catalogue fit's starless estimate and its stars' model; the noise map a
# network reads, made from the input alone (`image_sigma`); where the loss
# may look (finite, and outside the clean sky's mask); the stars' places; and
# of the stars' light, noise-free, what their halos and rings hold and what
# their diffraction spikes do (a pupil's PSF less the same optics' without
# vanes or hexagon; 0 for a stack's own field, of refractors and SCTs with no
# spider), so that halos can be kept in part and spikes given a layer of
# their own: the stars layer less these is the stars' cores and wings.
PLANES = ("input", "starless", "fit_starless", "fit_model", "noise", "valid", "points", "halos", "spikes")
# A window is kept where at least this share of it is valid.
VALID_SHARE = 0.8
# Stacks are cut this many times the workers ahead of the one being stored,
# so one large stack does not leave the other workers idle.
AHEAD = 8

# A composite holds between these shares of the stars the clean-sky pass
# took out of its stack.
DENSITY = (0.5, 1.5)
# The share of composites whose stars are a pupil's, with spikes, rather
# than the stack's own PSF field.
PUPIL_SHARE = 0.5
# A pupil composite's vanes, as `starless_stars.Pupil.vanes` counts them
# (radial bars: 4 is two crossed vanes), and how often each is drawn.
VANES = (0, 2, 3, 4)
# A star's stamp reaches as far as its light stays above `FADE` of the
# noise (`Telescope.reach`), and fades out in a circle from there to
# `FADE_BAND` times as far, so no star ends in a square edge however bright;
# it reaches `CORE_REACH` star widths at least and is at most `WIDE_STAMP`
# px across.
FADE = 0.05
FADE_BAND = 1.33
CORE_REACH = 2.0
WIDE_STAMP = 255
# A field's stamps end a few star widths out, their edges noisy: past
# `FIELD_TRUSTED` of a stamp's half side it gives way to a Moffat wing
# (`winged`), its β drawn from `WING_BETA` for each composite and its level
# matched to the stamp's light there, which runs on as far as the star's
# light reaches.
FIELD_TRUSTED = 0.6
WING_BETA = (1.8, 3.0)
# The brightest share of stars carry halos: wings holding between these
# shares of their light, drawn evenly in their logarithm, wide `HALO_WIDTH`
# star widths, and a reflection ring for `RING_SHARE` of them. Bright stars
# on our stacks leave 1 to 2.5% of their light past 20 px (2026-11-26), so
# halos are drawn from a half percent up, the faint ones as common as the
# strong.
HALO_STARS = 0.005
HALO_LIGHT = (0.005, 0.15)
HALO_WIDTH = (3.0, 8.0)
RING_SHARE = 0.3
RING_LIGHT = (0.002, 0.02)
RING_RADIUS = (6.0, 20.0)


def log_uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    """A draw between `bounds`, even in its logarithm."""
    return float(math.exp(rng.uniform(math.log(bounds[0]), math.log(bounds[1]))))


def radii(size: int) -> np.ndarray:
    """Each pixel's distance from a size × size grid's middle."""
    half = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64) - half
    return np.hypot(xx, yy)


def rounded(stamp: np.ndarray, inner: float = 1.0 / FADE_BAND) -> np.ndarray:
    """`stamp` faded by a raised cosine from `inner` of its half side to
    nothing at its half side, so that it ends in a circle."""
    half = (stamp.shape[0] - 1) / 2.0
    if half <= 0:
        return stamp
    t = np.clip((radii(stamp.shape[0]) / half - inner) / (1.0 - inner), 0.0, 1.0)
    return stamp * (0.5 * (1.0 + np.cos(np.pi * t)))


def cropped(stamp: np.ndarray, size: int) -> np.ndarray:
    """The middle size × size of `stamp` (both odd), padded with zeros where
    `stamp` is smaller."""
    full = stamp.shape[0]
    if size == full:
        return stamp
    if size < full:
        start = (full - size) // 2
        return stamp[start:start + size, start:start + size]
    out = np.zeros((size, size), dtype=stamp.dtype)
    start = (size - full) // 2
    out[start:start + full, start:start + full] = stamp
    return out


def outer_light(stamp: np.ndarray) -> np.ndarray:
    """For each whole radius R from a stamp's middle, the most light (by
    size) of any of its pixels R px out or further."""
    ring = np.floor(radii(stamp.shape[0])).astype(int)
    most = np.zeros(ring.max() + 1)
    np.maximum.at(most, ring.ravel(), np.abs(stamp).ravel())
    return np.maximum.accumulate(most[::-1])[::-1]


def winged(source: np.ndarray, size: int, fwhm: float, beta: float) -> np.ndarray:
    """A field's stamp on a size × size grid, carried on by a Moffat wing of
    `fwhm` and `beta`: the stamp to `FIELD_TRUSTED` of its half side, the
    wing from nine tenths of it, blended between, the wing's level matched
    to the stamp's mean light in a ring where the blend starts (none where
    that is not above zero)."""
    half = source.shape[0] // 2
    placed = cropped(np.asarray(source, dtype=np.float64), size)
    distance = radii(size)
    start, end = FIELD_TRUSTED * half, 0.9 * half
    alpha = fwhm / (2.0 * math.sqrt(2.0 ** (1.0 / beta) - 1.0))
    moffat = (1.0 + (distance / alpha) ** 2) ** -beta
    ring = np.abs(distance - start) <= 1.0
    level = max(0.0, float(placed[ring].mean())) / float(moffat[ring].mean()) if ring.any() else 0.0
    t = np.clip((distance - start) / (end - start), 0.0, 1.0)
    keep = 0.5 * (1.0 + np.cos(np.pi * t))
    return keep * placed + (1.0 - keep) * level * moffat


@dataclass
class Telescope:
    """The shape every star of a composite takes: the stack's own field
    (`field`), carried on by a wing of `wing_beta`, or a pupil's PSF
    (`stamp`, `WIDE_STAMP` across), at the stack's star width `fwhm`."""

    field: Any | None = None
    stamp: np.ndarray | None = None
    description: str = ""
    fwhm: float = 3.0
    wing_beta: float = 2.5
    # A pupil's PSF without its vanes or hexagon, where it has them: the
    # stamp less this is the spikes.
    plain: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.field is None:
            envelope = np.asarray(self.stamp, dtype=np.float64)
        else:
            stamps = getattr(self.field, "arrays", {}).get("stamps")
            # The most light any of the field's stamps holds, carried on: a
            # star's reach is judged by the widest of them.
            envelope = winged(np.abs(np.asarray(stamps, dtype=np.float64)).max(axis=(0, 1)) if stamps is not None
                              else np.asarray(self.field.stamp_at(0.0, 0.0), dtype=np.float64),
                              WIDE_STAMP, self.fwhm, self.wing_beta)
        self.outer = outer_light(envelope)

    def reach(self, flux: float, floor: float) -> float:
        """How far out, in px, a star of `flux` stays above `FADE` × `floor`
        (the noise): `CORE_REACH` star widths at least."""
        above = np.nonzero(flux * self.outer >= FADE * floor)[0]
        found = float(above[-1] + 1) if above.size else 0.0
        return max(found, CORE_REACH * self.fwhm)

    def stamp_for(self, x: float, y: float, flux: float, floor: float) -> np.ndarray:
        """The stamp a star of `flux` at (x, y) is drawn on over the noise
        `floor`: as far as it reaches and `FADE_BAND` times on, faded out in
        a circle (`rounded`), its light per unit of `flux`."""
        return self.stamps_for(x, y, flux, floor)[0]

    def stamps_for(self, x: float, y: float, flux: float, floor: float) -> tuple[np.ndarray, np.ndarray | None]:
        """`stamp_for`'s stamp, and its diffraction spikes drawn alike (the
        stamp less `plain`), None where the telescope throws none."""
        outer = min(WIDE_STAMP // 2, int(math.ceil(self.reach(flux, floor) * FADE_BAND)))
        size = 2 * outer + 1
        if self.field is None:
            source = np.asarray(self.stamp, dtype=np.float64)
        else:
            source = np.asarray(self.field.stamp_at(x, y), dtype=np.float64)
            if outer > FIELD_TRUSTED * (source.shape[0] // 2):
                source = winged(source, size, self.fwhm, self.wing_beta)
        drawn = rounded(cropped(source, size))
        if self.plain is None:
            return drawn, None
        return drawn, rounded(cropped(source - np.asarray(self.plain, dtype=np.float64), size))


def magnitude_light(table: np.ndarray) -> tuple[float, float] | None:
    """The instrumental magnitude as `slope` × magnitude + `zero`, from a
    clean-sky star table's unclipped, kept stars; None with too few."""
    col = {name: index for index, name in enumerate(starless_sky.STAR_COLUMNS)}
    good = (table[:, col["flux"]] > 0) & (table[:, col["clipped"]] == 0) & (table[:, col["reasons"]] == 0) \
        & (table[:, col["flux"]] > starless_sky.SCALE_SNR * table[:, col["flux_error"]])
    if good.sum() < starless_sky.SCALE_STARS:
        return None
    slope, zero = np.polyfit(table[good, col["magnitude"]], -2.5 * np.log10(table[good, col["flux"]]), 1)
    return float(slope), float(zero)


def pupil_telescope(rng: np.random.Generator, fwhm: float) -> Telescope:
    """A telescope of random vanes and obstruction, its stars `fwhm` px wide
    as the stack's are (the seeing blur makes up the width diffraction does
    not)."""
    vanes = int(rng.choice(VANES))
    pupil = stars.Pupil(obstruction=float(rng.uniform(0.15, 0.45)) if vanes or rng.random() < 0.5 else 0.0,
                        vanes=vanes, vane_width=float(rng.uniform(0.006, 0.025)),
                        angle=float(rng.uniform(0.0, 180.0)))
    diffraction = float(rng.uniform(0.5, max(0.6, 0.6 * fwhm)))
    seeing = max(0.8, math.sqrt(max(fwhm**2 - (1.03 * diffraction) ** 2, 0.64)))
    optics = stars.Optics(diffraction_px=diffraction, broad=bool(rng.random() < 0.5), seeing_fwhm=seeing,
                          pupil=pupil)
    stamp = stars.pupil_psf(optics, WIDE_STAMP)
    plain = None
    if pupil.vanes or pupil.hexagon:
        bare = stars.Pupil(obstruction=pupil.obstruction)
        plain = stars.pupil_psf(stars.Optics(diffraction_px=optics.diffraction_px, broad=optics.broad,
                                             seeing_fwhm=optics.seeing_fwhm, seeing_beta=optics.seeing_beta,
                                             pupil=bare), WIDE_STAMP)
    return Telescope(stamp=stamp, fwhm=fwhm, plain=plain,
                     description=f"pupil vanes {vanes} obstruction {pupil.obstruction:.2f} "
                                 f"diffraction {diffraction:.2f} px seeing {seeing:.2f} px")


def add_star(light: np.ndarray, stamp: np.ndarray, x: float, y: float, flux: float) -> None:
    """Add `flux` × `stamp`, centred at (x, y), to `light`, clipped at its
    edges: moved within its pixel by Lanczos-3, as the fit draws a star
    (`starless_sky.lanczos_shifted`). A shift by Fourier transform rings
    along the row and the column through a core narrower than its
    sampling, a dotted cross no telescope makes, many times the noise
    beside a bright star."""
    half = stamp.shape[0] // 2
    column, row = int(round(x)), int(round(y))
    moved = starless_sky.lanczos_shifted(stamp, x - column, y - row) * flux
    height, width = light.shape
    top, left = row - half, column - half
    a0, b0 = max(0, top), max(0, left)
    a1, b1 = min(height, top + stamp.shape[0]), min(width, left + stamp.shape[1])
    if a0 >= a1 or b0 >= b1:
        return
    light[a0:a1, b0:b1] += moved[a0 - top:a1 - top, b0 - left:b1 - left]


def composite(sky: np.ndarray, table: np.ndarray, summary: dict[str, Any], telescope: Telescope,
              rng: np.random.Generator, noise: np.ndarray) -> dict[str, Any]:
    """`sky` with stars put back (the module's list): the input, both
    truths, the stars' noise-free light and, of it, the halos' and rings'
    (`halos`) and the diffraction spikes' (`spikes`), and the stars as (x, y,
    magnitude, flux) in the order they were put."""
    height, width = sky.shape
    col = {name: index for index, name in enumerate(starless_sky.STAR_COLUMNS)}
    scale = magnitude_light(table)
    if scale is None:
        raise ValueError("too few well-measured stars to scale magnitudes to light")
    slope, zero = scale
    magnitudes = table[np.isfinite(table[:, col["magnitude"]]), col["magnitude"]]
    count = int(round(len(magnitudes) * rng.uniform(*DENSITY)))
    drawn = rng.choice(magnitudes, size=count) + rng.normal(0.0, 0.15, size=count)
    finite = np.isfinite(sky)
    light = np.zeros((height, width), dtype=np.float64)
    # Of that light, the halos' and rings', and the spikes'.
    halo_light = np.zeros((height, width), dtype=np.float64)
    spike_light = np.zeros((height, width), dtype=np.float64)
    fwhm = float(summary.get("fwhm_px") or 3.0)
    floor = float(np.nanmedian(noise))
    placed = []
    # The brightest `HALO_STARS` of them, by count, carry halos (none at 0).
    halos = int(round(HALO_STARS * count))
    halo_cut = float(np.sort(drawn)[halos - 1]) if halos else -np.inf
    for magnitude in drawn:
        x, y = rng.uniform(0, width - 1), rng.uniform(0, height - 1)
        if not finite[int(round(y)), int(round(x))]:
            continue
        flux = 10 ** (-0.4 * (slope * magnitude + zero))
        stamp, spikes = telescope.stamps_for(x, y, flux, floor)
        add_star(light, stamp, x, y, flux)
        if spikes is not None:
            add_star(spike_light, spikes, x, y, flux)
        if magnitude <= halo_cut:
            size = 2 * int(min(height, width, 6 * HALO_WIDTH[1] * fwhm) // 2) - 1
            share = log_uniform(rng, HALO_LIGHT)
            halo = rounded(stars.halo_wings(size, rng.uniform(*HALO_WIDTH) * fwhm, rng.uniform(1.2, 2.0)))
            for target in (light, halo_light):
                add_star(target, halo, x, y, flux * share)
            if rng.random() < RING_SHARE:
                radius = rng.uniform(*RING_RADIUS) * fwhm
                ring_size = 2 * int(min(height, width, 2.6 * radius + 20) // 2) - 1
                ring = rounded(stars.reflection_ring(ring_size, radius, radius / rng.uniform(4.0, 8.0),
                                                     x=rng.normal(0, 0.05 * radius), y=rng.normal(0, 0.05 * radius)))
                ring_flux = flux * log_uniform(rng, RING_LIGHT)
                for target in (light, halo_light):
                    add_star(target, ring, x, y, ring_flux)
        placed.append((x, y, float(magnitude), flux))
    photon = float(summary.get("photon_term") or 0.0)
    shot = rng.standard_normal((height, width)) * np.sqrt(photon * np.clip(light, 0, None))
    image = sky.astype(np.float64) + light + shot
    clip = summary.get("clip_level")
    if clip:
        image = np.minimum(image, float(clip))
    image = np.where(finite, image, np.nan).astype(np.float32)
    starless = sky.astype(np.float32)
    return {"input": image, "starless": starless, "stars": (image - starless).astype(np.float32),
            "light": light.astype(np.float32), "halos": halo_light.astype(np.float32),
            "spikes": spike_light.astype(np.float32), "placed": np.array(placed, dtype=np.float64).reshape(-1, 4)}


def fitted(made: dict[str, Any], noise: np.ndarray, clip: float | None, fwhm: float) -> dict[str, Any]:
    """The catalogue fit of a composite, as the runtime fits an image: its
    stars at the places they were put, with their places refined where their
    signal to noise reaches `starless_sky.REFINE_SNR` (a star's light over
    the noise of a Gaussian's effective area, 4π σ², for its `fwhm`), with
    the field measured on it and bright stars carried on past its stamps,
    so no square edge is left in the estimate (`starless_sky.clean_sky`,
    `wings`)."""
    listed = [(float(x), float(y), float(m)) for x, y, m, _ in made["placed"]]
    fluxes = made["placed"][:, 3] if listed else np.zeros(0)
    footprint = float(np.nanmedian(noise)) * math.sqrt(4.0 * math.pi) * fwhm / 2.3548
    refine = [bool(flux >= starless_sky.REFINE_SNR * footprint) for flux in fluxes]
    return starless_sky.clean_sky(made["input"], listed, noise=noise, clip=clip, refine=refine, wings=True)


def points_plane(placed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """A plane marking each star's place with its magnitude's light share:
    1 at the brightest, falling with magnitude, as the sharpening model's
    point plane marks sources."""
    plane = np.zeros(shape, dtype=np.float32)
    if not len(placed):
        return plane
    brightest = float(placed[:, 2].min())
    for x, y, magnitude, _ in placed:
        row, column = int(round(y)), int(round(x))
        if 0 <= row < shape[0] and 0 <= column < shape[1]:
            plane[row, column] = max(plane[row, column], 10 ** (-0.4 * (magnitude - brightest)))
    return plane


def window_corners(valid: np.ndarray, side: int, count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Up to `count` windows' top left corners, drawn from a grid half a
    window apart, each `VALID_SHARE` valid at least."""
    height, width = valid.shape
    corners = [(top, left) for top in range(0, height - side + 1, side // 2)
               for left in range(0, width - side + 1, side // 2)
               if valid[top:top + side, left:left + side].mean() >= VALID_SHARE]
    if not corners:
        return []
    chosen = rng.choice(len(corners), size=min(count, len(corners)), replace=False)
    return [corners[index] for index in sorted(chosen)]


def source_telescope(store: Store, product: dict[str, Any], rng: np.random.Generator, fwhm: float) -> Telescope:
    """A composite's telescope: a pupil's (`PUPIL_SHARE` of the time) or the
    stack's own PSF field, recorded or measured on the stack."""
    if rng.random() < PUPIL_SHARE:
        return pupil_telescope(rng, fwhm)
    from . import psf_fields

    try:
        field = psf_fields.Field(store, product["id"])
        if not field.usable:
            raise FileNotFoundError(product["id"])
    except (FileNotFoundError, KeyError):
        field = psf_fields.Field.measured_from(psf_fields.load_image(store, product))
    beta = float(rng.uniform(*WING_BETA))
    return Telescope(field=field, fwhm=fwhm, wing_beta=beta, description=f"the stack's own PSF field, wing β {beta:.2f}")


def prepared(store_root: str, scratch: str | None, product: dict[str, Any], *, draws: int, windows: int,
             side: int, seed: int) -> dict[str, Any]:
    """`draws` composites of one cleaned product, each fitted, and up to
    `windows` windows of each: their planes (`PLANES`) and records. In a
    process of its own; an error comes back as the product's record."""
    from .noise_model import image_sigma

    store = Store(store_root, scratch)
    try:
        here = starless_sky.directory(store, product["id"])
        sky = np.load(here / "sky.npy")
        mask = np.load(here / "mask.npy")
        table = np.load(here / "stars.npz")["stars"]
        summary = json.loads((here / "sky.json").read_text())
        fwhm = float(summary.get("fwhm_px") or 3.0)
        clip = summary.get("clip_level")
        found = []
        noise_sky = image_sigma(sky).astype(np.float32)
        for draw in range(draws):
            rng = np.random.default_rng([seed, int(hashlib.sha256(product["id"].encode()).hexdigest()[:8], 16), draw])
            telescope = source_telescope(store, product, rng, fwhm)
            made = composite(sky, table, summary, telescope, rng, noise_sky)
            noise = image_sigma(made["input"]).astype(np.float32)
            fit = fitted(made, noise, clip, fwhm)
            # Only the clean sky's own mask is kept out of the loss, where its
            # truth is not to be trusted; where the fit fails is where healing
            # is learned.
            valid = np.isfinite(made["input"]) & np.isfinite(sky) & ~mask
            plane = points_plane(made["placed"], sky.shape)
            full = {"input": made["input"], "starless": made["starless"], "fit_starless": fit["sky"],
                    "fit_model": fit["model"], "noise": noise, "valid": valid.astype(np.uint8), "points": plane,
                    "halos": made["halos"], "spikes": made["spikes"]}
            for top, left in window_corners(valid, side, windows, rng):
                cut = (slice(top, top + side), slice(left, left + side))
                found.append({"planes": {name: np.nan_to_num(full[name][cut]) for name in PLANES},
                              "record": {"product": product["id"], "draw": draw, "top": top, "left": left,
                                         "telescope": telescope.description, "fwhm_px": fwhm,
                                         "stars": int(len(made["placed"])), "telescope_kind":
                                             "pupil" if telescope.field is None else "field"}})
        return {"status": "cut", "product": product["id"], "windows": found}
    except Exception as error:  # noqa: BLE001 - reported in the summary
        return {"status": "dropped", "product": product["id"], "error": f"{type(error).__name__}: {error}",
                "windows": []}


def cut(store: Store, products: list[dict[str, Any]], *, work: Path, draws: int = 2, windows: int = 8,
        side: int = 512, seed: int = 0, workers: int = 2) -> dict[str, Any]:
    """Cut a bank of composites of `products` (cleaned by the clean-sky
    pass) under `work`, named by a hash of what went into it: `draws`
    composites a product, up to `windows` windows of `side` a composite.
    Windows go into memory-mapped planes; `sources.json` names each
    window's product, draw, place and telescope, and `done.json`, written
    last, the count. A finished bank is not cut again."""
    parts = [BANK_VERSION, seed, draws, windows, side, [product["id"] for product in products]]
    name = "starless-" + hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16]
    directory = work / name
    if (directory / "done.json").exists():
        return json.loads((directory / "done.json").read_text())
    directory.mkdir(parents=True, exist_ok=True)
    capacity = max(1, len(products) * draws * windows)
    arrays = {plane: np.lib.format.open_memmap(directory / f"{plane}.npy", mode="w+",
                                               dtype=np.uint8 if plane == "valid" else np.float32,
                                               shape=(capacity, side, side)) for plane in PLANES}
    records, dropped, count = [], [], 0
    task = functools.partial(prepared, str(store.root), str(store.scratch), draws=draws, windows=windows, side=side,
                             seed=seed)
    from .noise_model import in_order

    began = time.monotonic()
    for done, result in enumerate(in_order(task, products, workers, ahead=AHEAD), start=1):
        if result["status"] != "cut":
            dropped.append({"product": result["product"], "error": result.get("error")})
        for window in result["windows"]:
            for plane in PLANES:
                arrays[plane][count] = window["planes"][plane]
            records.append(window["record"])
            count += 1
        print(json.dumps({"bank": name, "done": done, "of": len(products), "product": result["product"],
                          "status": result["status"], "windows": count, "error": result.get("error"),
                          "elapsed": round(time.monotonic() - began)}), file=sys.stderr, flush=True)
    for array in arrays.values():
        array.flush()
    (directory / "sources.json").write_text(json.dumps(records))
    summary = {"version": BANK_VERSION, "name": name, "windows": count, "capacity": capacity, "side": side,
               "draws": draws, "per_draw": windows, "seed": seed, "products": len(products), "dropped": dropped,
               "planes": list(PLANES),
               "telescopes": {kind: sum(1 for record in records if record["telescope_kind"] == kind)
                              for kind in ("field", "pupil")}}
    (directory / "done.json").write_text(json.dumps(summary, indent=2))
    return summary
