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

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import starless_sky
from . import starless_stars as stars

# A composite holds between these shares of the stars the clean-sky pass
# took out of its stack.
DENSITY = (0.5, 1.5)
# The share of composites whose stars are a pupil's, with spikes, rather
# than the stack's own PSF field.
PUPIL_SHARE = 0.5
# A pupil composite's vanes, as `starless_stars.Pupil.vanes` counts them
# (radial bars: 4 is two crossed vanes), and how often each is drawn.
VANES = (0, 2, 3, 4)
# Stars brighter than this many times the noise at their peak are drawn on
# a stamp `WIDE_STAMP` px across; the rest on `NARROW_STAMP`.
WIDE_PEAK = 200.0
WIDE_STAMP = 255
NARROW_STAMP = 31
# The brightest share of stars carry halos: wings holding between these
# shares of their light, wide `HALO_WIDTH` star widths, and a reflection
# ring for `RING_SHARE` of them.
HALO_STARS = 0.005
HALO_LIGHT = (0.03, 0.15)
HALO_WIDTH = (3.0, 8.0)
RING_SHARE = 0.3
RING_LIGHT = (0.005, 0.03)
RING_RADIUS = (6.0, 20.0)


@dataclass
class Telescope:
    """The shape every star of a composite takes: the stack's own field
    (`field`), or a pupil's PSF (`stamp`, wide) at the stack's width."""

    field: Any | None = None
    stamp: np.ndarray | None = None
    description: str = ""

    def stamp_at(self, x: float, y: float, size: int) -> np.ndarray:
        """A stamp of `size` (odd) for a star at (x, y), summing to what of
        the star's light it holds."""
        source = self.field.stamp_at(x, y) if self.field is not None else self.stamp
        source = np.asarray(source, dtype=np.float64)
        full = source.shape[0]
        if size >= full:
            return source
        start = (full - size) // 2
        return source[start:start + size, start:start + size]


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
    return Telescope(stamp=stamp, description=f"pupil vanes {vanes} obstruction {pupil.obstruction:.2f} "
                                              f"diffraction {diffraction:.2f} px seeing {seeing:.2f} px")


def add_star(light: np.ndarray, stamp: np.ndarray, x: float, y: float, flux: float) -> None:
    """Add `flux` × `stamp`, centred at (x, y), to `light`, clipped at its
    edges."""
    half = stamp.shape[0] // 2
    column, row = int(round(x)), int(round(y))
    moved = starless_sky.placed(stamp, x, y) * flux
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
    truths, the stars' noise-free light, and the stars as (x, y, magnitude,
    flux) in the order they were put."""
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
        narrow = telescope.stamp_at(x, y, NARROW_STAMP)
        peak = flux * float(narrow.max())
        stamp = telescope.stamp_at(x, y, WIDE_STAMP) if peak > WIDE_PEAK * floor else narrow
        add_star(light, stamp, x, y, flux)
        if magnitude <= halo_cut:
            size = 2 * int(min(height, width, 6 * HALO_WIDTH[1] * fwhm) // 2) - 1
            share = rng.uniform(*HALO_LIGHT)
            add_star(light, stars.halo_wings(size, rng.uniform(*HALO_WIDTH) * fwhm, rng.uniform(1.2, 2.0)),
                     x, y, flux * share)
            if rng.random() < RING_SHARE:
                radius = rng.uniform(*RING_RADIUS) * fwhm
                ring_size = 2 * int(min(height, width, 2.6 * radius + 20) // 2) - 1
                ring = stars.reflection_ring(ring_size, radius, radius / rng.uniform(4.0, 8.0),
                                             x=rng.normal(0, 0.05 * radius), y=rng.normal(0, 0.05 * radius))
                add_star(light, ring, x, y, flux * rng.uniform(*RING_LIGHT))
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
            "light": light.astype(np.float32), "placed": np.array(placed, dtype=np.float64).reshape(-1, 4)}


def fitted(made: dict[str, Any], noise: np.ndarray, clip: float | None, fwhm: float) -> dict[str, Any]:
    """The catalogue fit of a composite, as the runtime fits an image: its
    stars at the places they were put, with their places refined where their
    signal to noise reaches `starless_sky.REFINE_SNR` (a star's light over
    the noise of a Gaussian's effective area, 4π σ², for its `fwhm`), with
    the field measured on it (`starless_sky.clean_sky`)."""
    listed = [(float(x), float(y), float(m)) for x, y, m, _ in made["placed"]]
    fluxes = made["placed"][:, 3] if listed else np.zeros(0)
    footprint = float(np.nanmedian(noise)) * math.sqrt(4.0 * math.pi) * fwhm / 2.3548
    refine = [bool(flux >= starless_sky.REFINE_SNR * footprint) for flux in fluxes]
    return starless_sky.clean_sky(made["input"], listed, noise=noise, clip=clip, refine=refine)


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
