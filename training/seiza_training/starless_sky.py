"""The clean-sky pass for the starless model (`docs/design/ml-star-removal.md`):
each stack's stars fitted with its own PSF field at their places and taken
out, leaving its sky and nebula with their noise, and a mask of where what is
left cannot be trusted. Stars are then put back into these skies with known
light, so both of the model's layers have a known truth.

For a product:

1. Its stars. The stack is solved from its own stars with its set's centre
   and scale as the prior (`seiza.solve_with_prior`), and the stars of a
   catalogue over it listed (Seiza's deep Gaia catalogue, or the one named,
   such as the one to G 20); with them, the stars the PSF pass's detector
   finds that the catalogue lacks, fainter ones, given magnitudes on the
   catalogue's scale.
2. The fit. Its PSF field measured on it, with its clip level
   (`seiza.PsfField`), every star fitted with it at once, positions refined
   (`seiza.StarFit`), and subtracted.
3. The field's systematic miss. The field's stamps end at a square and are
   measured from the image, so they miss a little of every star's light. What
   bright, unclipped, isolated stars leave, each moved to its own centre and
   over its flux, is averaged and taken out of every star in proportion to
   its flux (`PSF_MISS_*`). Past that cut nothing more is taken out of a
   clean sky: what the stars leave past the stamps (33 px, or 7 FWHM) grows
   with their brightness and, in a crowded field, the faint stars on a
   bright one's wing take some of it for their own light, so no one profile
   by flux reads it (2026-11-26: on a dense stack of 2.3 px stars, such a
   profile held -15% of a star's light). Bright stars' glows are masked
   instead (step 5). Where the fit's estimate is wanted rather than a
   truth, as the bank's fit planes are and a runtime's fit is, each bright
   star is carried on past its stamp by a Moffat wing (`WING_*`), so its
   square edge is not left in the estimate.
4. The noise each star is judged by: the image's noise map, with the star's
   own photon noise, whose scale is found from how the leftovers' scatter
   grows with the stars' light (`photon_term`). Judged by the sky's noise
   alone, a well taken out bright star fails by its photon noise.
5. The mask: about each star that is clipped, fitted poorly, or left far
   over or under its noise at its core, a disc that grows with the star's
   brightness (`MASK_*`); and about a clipped or very bright star, as far
   as what it leaves, read ring by ring, stays over half the noise
   (`GLOW_*`): what a stack's stars leave past the stamps grows with their
   brightness, which no one profile by flux takes out. On a deep stack of 8.6 px stars, a tenth of a
   percent of each star's light is left after the correction; the stars
   fainter than G 13 leave nothing the stack's two independent halves agree
   on more than they agree on the sky beside them, and the brighter ones
   leave light in both, which the mask covers (2026-11-25).

Records go under `STORE/starless/<product id>/`: `sky.npy` (float32, NaN
where the image is NaN), `mask.npy` (bool, True where left out), `stars.npz`
(the fitted stars, `STAR_COLUMNS`), and `sky.json` with the summary, written
last, which marks a product done. The manifest is only read. A pass stopped
part way goes on with the products not yet done.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import Manifest
from .store import Store, versions

# Raised with the records' layout or what they hold, so older ones are made
# again. 3: bright stars' glows masked (`GLOW_*`).
CLEAN_SKY_VERSION = 3

# The products the pass cleans, as the sharpening bank's known-kernel
# windows come from.
SOURCE_KINDS = ("deep", "multi-night")
SPLITS = ("train", "validation")

# The fitted stars' columns: position, catalogue (or scaled) magnitude, flux
# and its error, background, reduced chi-squared, clipped, the core's mean
# left in units of its noise (`core_left`), and why the star was masked
# (`REASONS`, a bit each; 0 for a star kept).
STAR_COLUMNS = ("x", "y", "magnitude", "flux", "flux_error", "background", "chi2", "clipped", "core_left",
                "reasons")
REASONS = ("clipped", "poor fit", "hole", "left", "dipole", "glow")

# A detector's star within this many pixels of a catalogue star is that star.
SAME_STAR_PX = 3.0
# Only the stars a first fit at their catalogue places measures at this
# signal to noise or more are taken out. A deeper catalogue than the stack
# lists stars under its noise: on a deep stack of 8.6 px stars, the 9,000
# stars from G 17 to G 20 fitted and taken out raised the share masked from
# 3.4% to 8.2%, for stars the detector found 3 of (2026-11-25). Those left
# are part of the sky's noise to a model, as to the detector.
TAKE_OUT_SNR = 3.0
# Stars measured at this signal to noise or more have their places refined;
# fainter ones are fitted at the catalogue's. (From 50, a dense field kept
# hundreds of dipoles: stars a fraction of a pixel off the catalogue.)
REFINE_SNR = 5.0
# The detector's stars are given magnitudes from the catalogue's stars
# measured at more than this signal to noise, of which at least
# `SCALE_STARS` are needed.
SCALE_SNR = 20.0
SCALE_STARS = 20

# The PSF's miss is read from stars this many times over the noise in flux,
# unclipped, with no other star within `PSF_MISS_ISOLATION` FWHM, on a cut
# `PSF_MISS_CUT` FWHM across, from `PSF_MISS_STARS` of them at least.
PSF_MISS_FLUX = 2000.0
PSF_MISS_ISOLATION = 4.0
PSF_MISS_CUT = 3.0
PSF_MISS_STARS = 10

# About a clipped star, or one over `GLOW_FLUX` times its noise in flux, what
# the fit leaves is read in rings from its disc's edge out to `GLOW_REACH` px,
# `GLOW_BINS` of them evenly spread in the logarithm of the radius, each cut
# into `GLOW_SECTORS` sectors. A ring glows where, in all but one of its
# sectors at most, the median stands over `GLOW_LIMIT` of the noise above the
# same sector's median `GLOW_AHEAD` rings further out (or under it, for a
# hole): a star's glow is round, but a nebula about it slopes, and stands over
# the ring further out on one side only. While its rings glow, one after the
# other, its disc reaches on, the reason "glow" where it had none. On a stack
# of 2.3 px stars, the light bright stars leave past the 33 px stamps read 2
# to 3 millionths of their flux a pixel at 16 to 24 px, tens of times the
# noise for the brightest.
GLOW_FLUX = 20000.0
GLOW_LIMIT = 0.3
GLOW_BINS = 16
GLOW_SECTORS = 8
GLOW_AHEAD = 3
GLOW_REACH = 256

# For the fit's estimate (`wings=True`), each bright star's stamp is kept to
# `WING_TRUSTED` of its half side and blended, by a raised cosine to
# `WING_BLENDED` of it, into a Moffat wing of the field's FWHM and β there (β
# held within `WING_BETA`), matched to the stamp's mean light on the ring where
# the blend starts; the wing carries on past the stamp's square as far as its
# light stays over `WING_FADE` of the noise (`WING_REACH` px at most), faded
# round over the last quarter of that. A field's stamps are measured, so their
# outer parts hold its noise and any pedestal, which a bright star's flux
# raises into a square of noise about it; the wing has neither, nor an edge.
WING_TRUSTED = 0.6
WING_BLENDED = 0.9
WING_FADE = 0.05
WING_BETA = (1.5, 4.0)
WING_REACH = 512

# The photon term is read from pixels where the stars' model is over this
# many times the sky's noise, from `PHOTON_PIXELS` of them at least, in
# `PHOTON_BINS` bins of the model's light of `PHOTON_BIN_PIXELS` each at
# least.
PHOTON_MODEL = 50.0
PHOTON_PIXELS = 1000
PHOTON_BINS = 12
PHOTON_BIN_PIXELS = 50
# The median of a chi-squared variable of one degree of freedom.
CHI2_MEDIAN = 0.454936

# A star is masked when its fit's reduced chi-squared is over `POOR_CHI2`,
# or the mean left in its core (half a FWHM) is beyond `CORE_LIMIT` of that
# mean's noise; read against a ring `RING_FWHM` FWHM out.
POOR_CHI2 = 3.0
CORE_LIMIT = 5.0
RING_FWHM = 3.0
# A masked star's disc: `MASK_BASE` FWHM, and `MASK_PER_DECADE` more for each
# decade its peak stands over its noise.
MASK_BASE = 3.0
MASK_PER_DECADE = 2.0


def faint_detections(detections: np.ndarray, columns: Sequence[str],
                     listed: Sequence[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """The detector's stars no catalogue star is within `SAME_STAR_PX` of,
    as (x, y, flux)."""
    index = {name: column for column, name in enumerate(columns)}
    known = np.array([(x, y) for x, y, _ in listed], dtype=np.float64).reshape(-1, 2)
    found = []
    for row in np.asarray(detections, dtype=np.float64).reshape(-1, len(columns)):
        x, y, flux = row[index["x"]], row[index["y"]], row[index["flux"]]
        if not (np.isfinite(flux) and flux > 0):
            continue
        if known.size and np.min(np.hypot(known[:, 0] - x, known[:, 1] - y)) < SAME_STAR_PX:
            continue
        found.append((float(x), float(y), float(flux)))
    return found


def magnitude_scale(fitted: Sequence[Any]) -> tuple[float, float] | None:
    """The instrumental magnitude, −2.5 log10 of the flux, as `slope` × the
    catalogue's magnitude + `zero`, from the stars measured well, or None
    where too few are."""
    good = [st for st in fitted if np.isfinite(st.flux[0]) and st.flux[0] > 0 and not st.saturated
            and st.flux[0] > SCALE_SNR * st.flux_error[0]]
    if len(good) < SCALE_STARS:
        return None
    slope, zero = np.polyfit([st.magnitude for st in good], [-2.5 * math.log10(st.flux[0]) for st in good], 1)
    return float(slope), float(zero)


def centred(plane: np.ndarray, x: float, y: float, half: int) -> np.ndarray | None:
    """The (2 `half` + 1)-pixel square about (x, y), moved by a Fourier shift
    so that (x, y) is its middle; None where it would leave the plane."""
    column, row = int(round(x)), int(round(y))
    pad = half + 2
    height, width = plane.shape
    if not (pad <= column < width - pad and pad <= row < height - pad):
        return None
    part = np.nan_to_num(plane[row - pad:row + pad + 1, column - pad:column + pad + 1].astype(np.float64))
    fy = np.fft.fftfreq(part.shape[0])[:, None]
    fx = np.fft.fftfreq(part.shape[1])[None, :]
    moved = np.real(np.fft.ifft2(np.fft.fft2(part) * np.exp(2j * np.pi * (fx * (x - column) + fy * (y - row)))))
    return moved[2:-2, 2:-2]


def placed(cut: np.ndarray, x: float, y: float) -> np.ndarray:
    """`cut`, centred at its middle, moved so its middle is at (x, y) within
    the pixel (x, y) rounds to: `centred`'s inverse."""
    part = np.pad(cut, 2)
    fy = np.fft.fftfreq(part.shape[0])[:, None]
    fx = np.fft.fftfreq(part.shape[1])[None, :]
    shift = np.exp(-2j * np.pi * (fx * (x - round(x)) + fy * (y - round(y))))
    return np.real(np.fft.ifft2(np.fft.fft2(part) * shift))[2:-2, 2:-2]


def local_noise(noise: np.ndarray, x: float, y: float, reach: int = 8) -> float:
    column, row = int(round(x)), int(round(y))
    part = noise[max(0, row - reach):row + reach + 1, max(0, column - reach):column + reach + 1]
    return float(np.nanmedian(part)) if part.size else float("nan")


def psf_miss(sky: np.ndarray, fitted: Sequence[Any], noise: np.ndarray, fwhm: float) -> np.ndarray | None:
    """What the field's PSF misses of a star, a pixel's worth of its flux, as
    a cut `PSF_MISS_CUT` FWHM across, centred: the median of what bright,
    unclipped, isolated stars leave in `sky`, each moved to its own centre,
    less its cut's edge, over its flux. None where too few stars give it."""
    half = int(math.ceil(PSF_MISS_CUT * fwhm)) // 2
    positions = np.array([(st.x, st.y) for st in fitted], dtype=np.float64).reshape(-1, 2)
    cuts = []
    for index, st in enumerate(fitted):
        flux = st.flux[0]
        if st.saturated or not (flux > 0) or not (flux >= PSF_MISS_FLUX * local_noise(noise, st.x, st.y)):
            continue
        near = np.hypot(*(positions - positions[index]).T)
        if np.sum(near < PSF_MISS_ISOLATION * fwhm) > 1:
            continue
        cut = centred(sky, st.x, st.y, half)
        if cut is None:
            continue
        edge = np.concatenate([cut[0], cut[-1], cut[:, 0], cut[:, -1]])
        cuts.append((cut - np.median(edge)) / flux)
    if len(cuts) < PSF_MISS_STARS:
        return None
    return np.median(np.array(cuts), axis=0)


def without_miss(sky: np.ndarray, fitted: Sequence[Any], miss: np.ndarray) -> np.ndarray:
    """`sky` with `miss` taken out of every unclipped star by its flux."""
    corrected = sky.astype(np.float64).copy()
    half = miss.shape[0] // 2
    height, width = sky.shape
    for st in fitted:
        if st.saturated or not (st.flux[0] > 0):
            continue
        column, row = int(round(st.x)), int(round(st.y))
        if not (half <= column < width - half and half <= row < height - half):
            continue
        corrected[row - half:row + half + 1, column - half:column + half + 1] -= st.flux[0] * placed(miss, st.x, st.y)
    return corrected.astype(np.float32)


def moffat_alpha(fwhm: float, beta: float) -> float:
    """A Moffat profile's α for its FWHM and β."""
    return fwhm / (2.0 * math.sqrt(2.0 ** (1.0 / beta) - 1.0))


def lanczos_shifted(stamp: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """`stamp` (odd, centred) moved by (`dx`, `dy`) of a pixel, each within
    half a pixel, by Lanczos-3, zero past its edge: as `seiza.StarFit` draws
    a star (an output sample u is the stamp at u − dx)."""
    size = stamp.shape[0]

    def weights(t: float) -> np.ndarray:
        taps = t - (np.arange(6) - 2.0)
        found = np.where(np.abs(taps) < 3.0, np.sinc(taps) * np.sinc(taps / 3.0), 0.0)
        return found / found.sum()

    offset_x, tx = (-1, 1.0 - dx) if dx > 0 else (0, -dx)
    offset_y, ty = (-1, 1.0 - dy) if dy > 0 else (0, -dy)
    wx, wy = weights(tx), weights(ty)
    padded = np.pad(np.asarray(stamp, dtype=np.float64), 3)
    rows = sum(wx[k] * padded[:, offset_x + k + 1:offset_x + k + 1 + size] for k in range(6))
    return sum(wy[k] * rows[offset_y + k + 1:offset_y + k + 1 + size, :] for k in range(6))


def fit_stamp(field: Any, x: float, y: float) -> tuple[int, int, np.ndarray]:
    """A star's stamp at (x, y) as `seiza.StarFit` draws it, of unit flux:
    the pixel its middle lands on, (column, row), and its samples, its first
    channel's for a colour field."""
    column, row = int(round(x)), int(round(y))
    stamp = np.asarray(field.stamp_at(x, y), dtype=np.float64)
    stamp = stamp[..., 0] if stamp.ndim == 3 else stamp
    return column, row, lanczos_shifted(stamp, x - column, y - row)


def winged(sky: np.ndarray, model: np.ndarray, fitted: Sequence[Any], field: Any,
           noise: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """`sky` with each bright star of `fitted` drawn with a smooth wing from
    its stamp's trusted ring out (`WING_*`) in place of the stamp's outer
    part, and taken out to where that wing fades; `model` with the change put
    in; and how many stars were."""
    sky = sky.astype(np.float64)
    model = model.astype(np.float64)
    height, width = sky.shape
    half = field.stamp_size // 2
    start, end = WING_TRUSTED * half, WING_BLENDED * half
    offsets = np.arange(-half, half + 1)

    def mono(stamp: Any) -> np.ndarray:
        stamp = np.asarray(stamp, dtype=np.float64)
        return stamp[..., 0] if stamp.ndim == 3 else stamp

    # Stars too faint for their stamp's outer part to show anywhere are passed
    # over before their own stamps are read.
    centre = mono(field.stamp_at((width - 1) / 2, (height - 1) / 2))
    ring_at_middle = np.abs(np.hypot(offsets[None, :], offsets[:, None]) - start) <= 1.0
    typical = max(float(np.max(np.abs(centre[ring_at_middle]))), 1e-12)
    # A wing reaches past the trusted ring only where the star's light on it
    # times the Moffat's value at the ring's middle over the Moffat's mean on
    # it tops the floor. That ratio is at most ((start + 1) / start)^(2β), the
    # ring holding no pixel past start + 1, so a star short of the floor by
    # more is passed over before its shape is read: reading it is the slow
    # step, and most stars stop here.
    margin = ((start + 1.0) / start) ** (2.0 * WING_BETA[1])
    count = 0
    for st in fitted:
        flux = st.flux[0]
        floor = WING_FADE * local_noise(noise, st.x, st.y)
        if not (flux > 0) or not (flux * 4.0 * typical > floor):
            continue
        column, row, stamp = fit_stamp(field, st.x, st.y)
        local = np.hypot(offsets[None, :] + column - st.x, offsets[:, None] + row - st.y)
        ring = np.abs(local - start) <= 1.0
        light = max(0.0, float(stamp[ring].mean()))
        if not (flux * light * margin > floor):
            continue
        shape = field.shape_at(st.x, st.y)
        if shape is None or not np.isfinite(shape.fwhm) or not np.isfinite(shape.beta):
            continue
        beta = float(np.clip(shape.beta, *WING_BETA))
        alpha = moffat_alpha(float(shape.fwhm), beta)
        moffat_local = (1.0 + local**2 / alpha**2) ** -beta
        level = light / float(moffat_local[ring].mean())
        if not (level > 0):
            continue
        over = flux * level / floor
        if over <= 1.0:
            continue
        reach = min(alpha * math.sqrt(over ** (1.0 / beta) - 1.0), WING_REACH)
        if reach <= start:
            continue
        span = max(int(math.ceil(reach)), half)
        top, bottom = max(0, row - span), min(height, row + span + 1)
        left, right = max(0, column - span), min(width, column + span + 1)
        if top >= bottom or left >= right:
            continue
        rows = np.arange(top, bottom)[:, None]
        columns = np.arange(left, right)[None, :]
        distance = np.hypot(columns - st.x, rows - st.y)
        fade = np.clip((reach - distance) / (0.25 * reach), 0.0, 1.0)
        wing = level * (1.0 + distance**2 / alpha**2) ** -beta * (0.5 - 0.5 * np.cos(np.pi * fade))
        blend = np.clip((distance - start) / (end - start), 0.0, 1.0)
        keep = 0.5 * (1.0 + np.cos(np.pi * blend))
        # The stamp on the patch, nothing past its square.
        drawn = np.zeros(distance.shape)
        a0, b0 = max(top, row - half), max(left, column - half)
        a1, b1 = min(bottom, row + half + 1), min(right, column + half + 1)
        drawn[a0 - top:a1 - top, b0 - left:b1 - left] = stamp[a0 - row + half:a1 - row + half,
                                                              b0 - column + half:b1 - column + half]
        change = flux * ((keep * drawn + (1.0 - keep) * wing) - drawn)
        sky[top:bottom, left:right] -= change
        model[top:bottom, left:right] += change
        count += 1
    return sky.astype(np.float32), model.astype(np.float32), count


def glow_reach(sky: np.ndarray, x: float, y: float, floor: float, start: float) -> float:
    """How far what is left about (x, y) glows (`GLOW_*`): the outer edge of
    the last of the rings from `start` px on that glow, each after the one
    before; 0 where the first does not."""
    if not (start > 0) or start >= GLOW_REACH:
        return 0.0
    ratio = (GLOW_REACH / start) ** (1.0 / GLOW_BINS)
    edges = start * ratio ** np.arange(GLOW_BINS + GLOW_AHEAD + 1)
    height, width = sky.shape
    column, row = int(round(x)), int(round(y))
    found = 0.0
    # The first ring alone is read first, on a small cut: most stars stop
    # there.
    for rings in (1, GLOW_BINS):
        span = int(math.ceil(edges[rings + GLOW_AHEAD]))
        top, bottom = max(0, row - span), min(height, row + span + 1)
        left, right = max(0, column - span), min(width, column + span + 1)
        if top >= bottom or left >= right:
            return 0.0
        part = np.asarray(sky[top:bottom, left:right], dtype=np.float64)
        dx = np.arange(left, right)[None, :] - x
        dy = np.arange(top, bottom)[:, None] - y
        ring = np.digitize(np.hypot(dx, dy), edges) - 1
        sector = ((np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * GLOW_SECTORS).astype(int) % GLOW_SECTORS
        inside = (ring >= 0) & (ring < rings + GLOW_AHEAD) & np.isfinite(part)
        keys = ring[inside] * GLOW_SECTORS + sector[inside]
        values = part[inside]
        order = np.argsort(keys, kind="stable")
        keys, values = keys[order], values[order]
        medians = np.full((rings + GLOW_AHEAD) * GLOW_SECTORS, np.nan)
        bounds = np.searchsorted(keys, np.arange(medians.size + 1))
        for key in range(medians.size):
            if bounds[key + 1] > bounds[key]:
                medians[key] = np.median(values[bounds[key]:bounds[key + 1]])
        medians = medians.reshape(rings + GLOW_AHEAD, GLOW_SECTORS)
        found = 0.0
        for index in range(rings):
            excess = medians[index] - medians[index + GLOW_AHEAD]
            over, under = np.sum(excess > floor), np.sum(excess < -floor)
            if max(over, under) < GLOW_SECTORS - 1:
                return found
            found = float(edges[index + 1])
    return found


def photon_terms(sky: np.ndarray, model: np.ndarray, noise: np.ndarray) -> tuple[float, float]:
    """How the leftovers' variance grows with the stars' light m in a pixel:
    as g·m, the stars' photon noise (an inverse gain), and as (f·m)², the
    share f of their light the field's shape misses, which grows with the
    light itself. Fitted, both at least 0, to the variance in bins of m;
    (0, 0) where too few bright pixels give them. A leftover is judged
    against the photon noise alone: the shape's miss is what the mask is
    for."""
    level = float(np.nanmedian(sky))
    floor = float(np.nanmedian(noise))
    bright = np.isfinite(sky) & np.isfinite(noise) & (model > PHOTON_MODEL * floor)
    if bright.sum() < PHOTON_PIXELS:
        return 0.0, 0.0
    light = model[bright].astype(np.float64)
    squares = (sky[bright].astype(np.float64) - level) ** 2
    floors = noise[bright].astype(np.float64) ** 2
    edges = np.geomspace(light.min(), light.max() * (1 + 1e-9), PHOTON_BINS + 1)
    found = []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (light >= low) & (light < high)
        if inside.sum() >= PHOTON_BIN_PIXELS:
            # A squared normal deviate's median is `CHI2_MEDIAN` of its mean,
            # so a bin's median square is scaled to a variance before the
            # sky's is taken off.
            variance = float(np.median(squares[inside])) / CHI2_MEDIAN - float(np.median(floors[inside]))
            found.append((float(np.median(light[inside])), variance))
    if not found:
        return 0.0, 0.0
    m, v = (np.array(column) for column in zip(*found))
    # Each bin weighs by its variance's own size, so the faint bins, where
    # the photon noise shows, count as much as the bright.
    weight = 1.0 / np.maximum(np.abs(v), floor**2)

    def fitted(columns: list[np.ndarray]) -> np.ndarray:
        design = np.stack(columns, axis=1) * weight[:, None]
        return np.linalg.lstsq(design, v * weight, rcond=None)[0]

    g, f2 = fitted([m, m**2])
    if g < 0:
        g, f2 = 0.0, max(float(fitted([m**2])[0]), 0.0)
    elif f2 < 0:
        g, f2 = max(float(fitted([m])[0]), 0.0), 0.0
    return float(g), math.sqrt(float(f2))


def core_left(plane: np.ndarray, noise: np.ndarray, x: float, y: float, fwhm: float) -> float | None:
    """The mean of `plane` within half a FWHM of (x, y), less its median on a
    ring `RING_FWHM` FWHM out, in units of that mean's noise; None where the
    ring would leave the plane."""
    ring = RING_FWHM * fwhm
    reach = int(math.ceil(ring + 2))
    column, row = int(round(x)), int(round(y))
    height, width = plane.shape
    if not (reach <= column < width - reach and reach <= row < height - reach):
        return None
    yy, xx = np.mgrid[-reach:reach + 1, -reach:reach + 1]
    distance = np.hypot(xx - (x - column), yy - (y - row))
    part = plane[row - reach:row + reach + 1, column - reach:column + reach + 1].astype(np.float64)
    level = np.nanmedian(part[(distance > ring - 2) & (distance <= ring)])
    core = distance <= 0.5 * fwhm
    spread = np.nanmean(noise[row - reach:row + reach + 1, column - reach:column + reach + 1][core])
    value = np.nanmean(part[core] - level) / (spread / math.sqrt(core.sum()))
    return float(value) if np.isfinite(value) else None


def core_dipole(plane: np.ndarray, noise: np.ndarray, x: float, y: float, fwhm: float) -> float | None:
    """How far what is left within a FWHM of (x, y) leans to one side: its
    first moment about the star in units of that moment's noise. A star
    fitted a fraction of a pixel off its place leaves a dipole, light on one
    side and a dip on the other, which a core's mean does not see."""
    reach = int(math.ceil(fwhm + 2))
    column, row = int(round(x)), int(round(y))
    height, width = plane.shape
    if not (reach <= column < width - reach and reach <= row < height - reach):
        return None
    yy, xx = np.mgrid[-reach:reach + 1, -reach:reach + 1].astype(np.float64)
    dx, dy = xx - (x - column), yy - (y - row)
    distance = np.hypot(dx, dy)
    inside = (distance <= fwhm) & (distance > 0)
    part = plane[row - reach:row + reach + 1, column - reach:column + reach + 1].astype(np.float64)
    spread = float(np.nanmean(noise[row - reach:row + reach + 1, column - reach:column + reach + 1][inside]))
    level = float(np.nanmedian(part[inside]))
    found = []
    for weight in (dx / np.where(distance > 0, distance, 1.0), dy / np.where(distance > 0, distance, 1.0)):
        w = weight[inside]
        found.append(float(np.nansum((part[inside] - level) * w)) / (spread * math.sqrt(float(np.sum(w * w)))))
    value = math.hypot(*found)
    return value if np.isfinite(value) else None


def measured_field(image: np.ndarray, clip: float | None) -> Any:
    """The image's PSF field, varying over it; one PSF for all where too
    few isolated stars give one that varies."""
    import seiza

    image = np.ascontiguousarray(image, dtype=np.float32)
    try:
        return seiza.PsfField.measure(image, saturation=clip)
    except seiza.TooFewStarsError:
        return seiza.PsfField.measure(image, saturation=clip, sparse="global")


def clean_sky(image: np.ndarray, stars: Sequence[tuple[float, float, float]], *, noise: np.ndarray,
              clip: float | None = None, refine: Sequence[bool] | None = None,
              wings: bool = False) -> dict[str, Any]:
    """`image` with `stars`, (x, y, magnitude), fitted with the PSF field
    measured on it and taken out (steps 2 to 5 of the module's): the clean
    sky, the stars' model, the mask, the stars' table (`STAR_COLUMNS`) and a
    summary. With `refine`, only the stars it marks have their positions
    refined, first, and the rest are fitted where they are listed, on what
    the first leave. With `wings`, for the fit's estimate rather than a
    truth, bright stars are carried on past their stamps (`WING_*`)."""
    import seiza

    image = np.ascontiguousarray(image, dtype=np.float32)
    noise = np.ascontiguousarray(noise, dtype=np.float32)
    height, width = image.shape
    field = measured_field(image, clip)
    refine = [True] * len(stars) if refine is None else list(refine)
    groups = [([star for star, moved in zip(stars, refine) if moved], "refined"),
              ([star for star, moved in zip(stars, refine) if not moved], "fixed")]
    sky, model, all_stars = image, np.zeros_like(image), []
    for listed, positions in groups:
        if not listed:
            continue
        fit = seiza.StarFit.fit(np.ascontiguousarray(sky, dtype=np.float32), field, listed, noise=noise,
                                saturation=clip, positions=positions)
        model = model + fit.model().astype(np.float32)
        sky = fit.subtract(np.ascontiguousarray(sky, dtype=np.float32)).astype(np.float32)
        all_stars += list(fit.stars)
    shape = field.shape_at((width - 1) / 2, (height - 1) / 2)
    fwhm = float(shape.fwhm) if shape is not None else float("nan")
    fitted = [st for st in all_stars if np.isfinite(st.flux[0]) and st.flux[0] > 0]
    miss = psf_miss(sky, fitted, noise, fwhm) if np.isfinite(fwhm) else None
    if miss is not None:
        sky = without_miss(sky, fitted, miss)
    carried = 0
    if wings:
        sky, model, carried = winged(sky, model, fitted, field, noise)
    inverse_gain, shape_error = photon_terms(sky, model, noise)
    judged = np.sqrt(noise.astype(np.float64) ** 2 + inverse_gain * np.clip(model, 0, None))

    mask = np.zeros((height, width), dtype=bool)
    rows, reasons = [], Counter()
    for st in all_stars:
        why = 0
        flux = st.flux[0]
        left = core_left(sky, judged, st.x, st.y, fwhm) if np.isfinite(fwhm) and flux > 0 else None
        column, row = int(round(st.x)), int(round(st.y))
        peak = float(np.nanmax(model[max(0, row - 1):row + 2, max(0, column - 1):column + 2])) \
            if 0 <= column < width and 0 <= row < height else 0.0
        clipped = bool(st.saturated) or (clip is not None and peak >= 0.8 * clip)
        if clipped:
            why |= 1
        if np.isfinite(st.reduced_chi_squared) and st.reduced_chi_squared > POOR_CHI2:
            why |= 2
        if left is not None and left < -CORE_LIMIT:
            why |= 4
        if left is not None and left > CORE_LIMIT:
            why |= 8
        lean = core_dipole(sky, judged, st.x, st.y, fwhm) if np.isfinite(fwhm) and flux > 0 else None
        if lean is not None and lean > CORE_LIMIT:
            why |= 16
        over = max(peak / max(local_noise(judged, st.x, st.y), 1e-12), 1.0)
        reach = (MASK_BASE + MASK_PER_DECADE * math.log10(over)) * fwhm if np.isfinite(fwhm) else 0.0
        if np.isfinite(fwhm) and (clipped or flux >= GLOW_FLUX * local_noise(noise, st.x, st.y)):
            glow = glow_reach(sky, st.x, st.y, GLOW_LIMIT * local_noise(noise, st.x, st.y), reach)
            if glow > reach:
                why |= 32
                reach = glow
        if why:
            for bit, reason in enumerate(REASONS):
                if why & (1 << bit):
                    reasons[reason] += 1
            top, bottom = max(0, int(st.y - reach)), min(height, int(st.y + reach) + 1)
            left_edge, right_edge = max(0, int(st.x - reach)), min(width, int(st.x + reach) + 1)
            yy, xx = np.ogrid[top:bottom, left_edge:right_edge]
            mask[top:bottom, left_edge:right_edge] |= (yy - st.y) ** 2 + (xx - st.x) ** 2 <= reach**2
        rows.append((st.x, st.y, st.magnitude, flux, st.flux_error[0], st.background[0], st.reduced_chi_squared,
                     float(clipped), np.nan if left is None else left, float(why)))
    sky[~np.isfinite(image)] = np.nan
    table = np.array(rows, dtype=np.float64).reshape(-1, len(STAR_COLUMNS))
    kept = table[:, STAR_COLUMNS.index("reasons")] == 0
    summary = {
        "stars": len(rows), "fitted": len(fitted), "masked_stars": int((~kept).sum()), "reasons": dict(reasons),
        "masked_share": round(float(mask.mean()), 4), "fwhm_px": round(fwhm, 3),
        "psf_miss": None if miss is None else round(float(miss.sum()), 5),
        "stamp_px": int(field.stamp_size),
        "winged": carried,
        "photon_term": round(inverse_gain, 5),
        "shape_error": round(shape_error, 5),
        "clip_level": clip,
    }
    return {"sky": sky, "model": model, "mask": mask, "stars": table, "summary": summary}


def listed_stars(image: np.ndarray, centre: tuple[float, float], scale_arcsec_px: float,
                 catalog_path: str | None = None) -> tuple[list[tuple[float, float, float]], dict[str, Any]]:
    """The catalogue's stars over `image` as (x, y, magnitude), from a solve
    of its own stars about `centre` (degrees) at `scale_arcsec_px`, and what
    the solve found."""
    import seiza

    height, width = image.shape
    detected = seiza.detect(np.nan_to_num(image), max_stars=800)
    solution = seiza.solve_with_prior(detected, seiza.StarCatalog.open(), width, height,
                                      {"center": centre, "scale_arcsec_px": scale_arcsec_px}, sip_order=3)
    catalog = seiza.StarCatalog.open(catalog_path)
    radius = 0.5 * math.hypot(width, height) * scale_arcsec_px / 3600.0 * 1.05
    stars = []
    for ra, dec, magnitude in catalog.cone_search(solution.ra, solution.dec, radius, limit=5_000_000):
        pixel = solution.wcs.world_to_pixel(ra, dec)
        if pixel is not None and 0 <= pixel[0] < width and 0 <= pixel[1] < height:
            stars.append((float(pixel[0]), float(pixel[1]), float(magnitude)))
    solved = {"matched": solution.matched_stars, "rms_px": round(solution.rms_arcsec / scale_arcsec_px, 3),
              "ra": solution.ra, "dec": solution.dec}
    return stars, solved


# Products with their sets' place and scale.
PRODUCTS = ("SELECT p.*, s.split, s.calibration_status, s.telescope, s.pixel_scale_arcsec, s.ra_deg, s.dec_deg "
            "FROM products p JOIN sets s ON s.id = p.set_id")


def eligible(manifest: Manifest, *, kinds: Iterable[str] = SOURCE_KINDS, splits: Iterable[str] = SPLITS,
             sets: set[str] | None = None) -> list[dict[str, Any]]:
    """The built products of `kinds` in sets calibrated with a flat, of
    `splits`, in id order, as the PSF pass chooses its own."""
    kinds, splits = tuple(kinds), tuple(splits)
    rows = manifest.rows(
        f"{PRODUCTS} WHERE p.status = 'built' AND s.calibration_status = 'ready' "
        f"AND p.kind IN ({', '.join('?' * len(kinds))}) AND s.split IN ({', '.join('?' * len(splits))}) "
        f"ORDER BY p.id", (*kinds, *splits))
    return [dict(row) for row in rows if sets is None or row["set_id"] in sets]


def annotated(manifest: Manifest, product_ids: Iterable[str]) -> list[dict[str, Any]]:
    """The products of `product_ids`, in their order, as `eligible` gives
    them; an id with no product is left out."""
    found = []
    for product_id in product_ids:
        rows = manifest.rows(f"{PRODUCTS} WHERE p.id = ?", (product_id,))
        if rows:
            found.append(dict(rows[0]))
    return found


def directory(store: Store, product_id: str) -> Path:
    """Where a product's clean sky lives."""
    return store.root / "starless" / product_id


def recorded(store: Store, product_id: str) -> dict[str, Any] | None:
    """A product's clean-sky summary, or None where the pass has not made
    one of this version."""
    path = directory(store, product_id) / "sky.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    return record if record.get("version") == CLEAN_SKY_VERSION else None


def measure(store: Store, product: dict[str, Any], catalog_path: str | None = None) -> dict[str, Any]:
    """Clean one product's sky and keep it (the module's steps 1 to 5)."""
    from . import psf_fields
    from .noise_model import image_sigma

    started = time.monotonic()
    image = psf_fields.load_image(store, product)
    noise = image_sigma(image).astype(np.float32)
    listed, solved = listed_stars(image, (product["ra_deg"], product["dec_deg"]), product["pixel_scale_arcsec"],
                                  catalog_path)
    import seiza

    found, clip, _ = psf_fields.detections(image)
    extra = faint_detections(found, psf_fields.DETECTION_COLUMNS, listed)
    # A first fit at the catalogue's places: which stars the stack shows, and
    # the flux scale for the detector's stars.
    first = seiza.StarFit.fit(np.ascontiguousarray(image), measured_field(image, clip), listed,
                              noise=noise, saturation=clip, positions="fixed")
    measured_snr = {star: (np.inf if st.saturated else st.flux[0] / st.flux_error[0])
                    for star, st in zip(listed, first.stars)
                    if st.saturated or (np.isfinite(st.flux[0]) and st.flux_error[0] > 0)}
    shown = [star for star in listed if measured_snr.get(star, 0.0) >= TAKE_OUT_SNR]
    stars = list(shown)
    scale = magnitude_scale(first.stars)
    if extra and scale is not None:
        slope, zero = scale
        stars += [(x, y, (-2.5 * math.log10(flux) - zero) / slope) for x, y, flux in extra]
    # Bright stars' places are refined; a faint one's would wander onto the
    # noise, so it is fitted where the catalogue puts it (the detector's
    # stars, with no catalogue place, are refined).
    refine = [measured_snr.get(star, np.inf) >= REFINE_SNR for star in stars]
    made = clean_sky(image, stars, noise=noise, clip=clip, refine=refine)
    target = directory(store, product["id"])
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "sky.npy", made["sky"])
    np.save(target / "mask.npy", made["mask"])
    np.savez_compressed(target / "stars.npz", stars=made["stars"], columns=np.array(STAR_COLUMNS))
    record = {"version": CLEAN_SKY_VERSION, "product": product["id"], "catalogue": catalog_path or "default",
              "catalogue_stars": len(listed), "shown": len(shown), "detected_only": len(stars) - len(shown),
              "solve": solved,
              **made["summary"], "seconds": round(time.monotonic() - started, 1), "versions": versions()}
    (target / "sky.json").write_text(json.dumps(record, indent=2))
    return record


def measured(store_root: str, scratch: str | None, product: dict[str, Any], catalog_path: str | None) -> dict:
    """`measure` in a process of its own; an error comes back as a record."""
    store = Store(store_root, scratch)
    try:
        return measure(store, product, catalog_path)
    except Exception as error:  # noqa: BLE001 - reported in the summary
        return {"product": product["id"], "status": "failed", "error": f"{type(error).__name__}: {error}"}


def run(store: Store, products: list[dict[str, Any]], *, catalog_path: str | None = None, workers: int = 2,
        again: bool = False, limit: int | None = None) -> dict[str, Any]:
    """Clean the sky of each of `products` not yet cleaned (all with
    `again`), `workers` at once, at most `limit` of them."""
    import functools

    from .noise_model import in_order

    todo = [product for product in products if again or recorded(store, product["id"]) is None][:limit]
    task = functools.partial(measured, str(store.root), str(store.scratch), catalog_path=catalog_path)
    summary: dict[str, Any] = {"products": len(products), "done": 0, "failed": []}
    started = time.monotonic()
    for count, record in enumerate(in_order(task, todo, workers), start=1):
        if record.get("status") == "failed":
            summary["failed"].append(record)
        else:
            summary["done"] += 1
        print(json.dumps({"done": count, "of": len(todo), "product": record["product"],
                          "masked_share": record.get("masked_share"), "stars": record.get("stars"),
                          "error": record.get("error"), "elapsed": round(time.monotonic() - started)}),
              file=sys.stderr, flush=True)
    return summary
