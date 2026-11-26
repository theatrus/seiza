"""Stars for the starless model's known-truth composites
(`docs/design/ml-star-removal.md`): PSFs made from a telescope's pupil, with
the diffraction spikes its vanes throw, and the halos bright stars carry.

A star's image is the squared modulus of its pupil's Fourier transform
(`pupil_psf`): a clear disc less its central obstruction and the vanes that
hold it, or a hexagon with a strut as JWST's segmented mirror. Each vane
throws a pair of spikes square to it, so two crossed vanes give four, three
give six, and a hexagon's edges six more with the strut's two. The pattern
is summed over the filter's wavelengths (a broad filter smears the spikes'
fringes and the rings into a continuum, a narrow one keeps them), made four
times finer than the camera's pixels and averaged down to them, then blurred
by seeing and tracking (a Moffat profile of the FWHM asked). Its scale is
set by how many of the camera's pixels the diffraction limit λ/D spans.

A bright star's halo reaches far past its core: wide wings (`halo_wings`,
a Moffat profile of a low β, holding a share of the star's light), and the
rings a filter's or a corrector's reflections throw (`reflection_ring`, an
annulus of a given radius and width, a little off the star where the
reflection is not centred), each a stamp of unit light to be scaled.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

# The PSF is made this many times finer than the pixels, then averaged down;
# odd, so the middle pixel's centre is a fine sample, where the pattern's
# middle is put.
OVERSAMPLE = 5
# The broad filters' wavelengths, relative to their middle, summed over.
BROAD = (0.8, 0.9, 1.0, 1.1, 1.2)
NARROW = (1.0,)


@dataclass(frozen=True)
class Pupil:
    """A telescope's pupil: `obstruction` the central obstruction's share
    of the aperture's diameter; `vanes` the number of straight vanes from
    the middle out (2 crossed vanes are 4 here), at `angle` degrees from x
    and evenly spread, each `vane_width` of the diameter wide; `hexagon`
    for a hexagonal aperture (JWST's), whose edges throw spikes of their
    own."""

    obstruction: float = 0.0
    vanes: int = 0
    vane_width: float = 0.01
    angle: float = 0.0
    hexagon: bool = False

    def mask(self, size: int, diameter: float) -> np.ndarray:
        """The pupil on a size × size grid, `diameter` samples across."""
        centre = (size - 1) / 2.0
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float64) - centre
        radius = diameter / 2.0
        if self.hexagon:
            # A flat-topped regular hexagon of circumradius `radius`.
            inside = np.ones((size, size), bool)
            for k in range(6):
                theta = math.radians(60.0 * k + 30.0)
                inside &= xx * math.cos(theta) + yy * math.sin(theta) <= radius * math.cos(math.radians(30.0))
        else:
            inside = np.hypot(xx, yy) <= radius
        inside &= np.hypot(xx, yy) >= self.obstruction * radius
        half_width = self.vane_width * diameter / 2.0
        for k in range(self.vanes):
            theta = math.radians(self.angle + 360.0 * k / self.vanes)
            along = xx * math.cos(theta) + yy * math.sin(theta)
            across = -xx * math.sin(theta) + yy * math.cos(theta)
            inside &= ~((along >= 0) & (np.abs(across) <= half_width))
        return inside.astype(np.float64)


@dataclass(frozen=True)
class Optics:
    """How a pupil is seen: `diffraction_px` the camera's pixels the
    diffraction limit λ/D spans at the filter's middle; `broad` for a broad
    filter's spread of wavelengths; `seeing_fwhm` the seeing and tracking
    blur in pixels, a Moffat profile of `seeing_beta`."""

    diffraction_px: float = 0.7
    broad: bool = False
    seeing_fwhm: float = 2.5
    seeing_beta: float = 3.0
    pupil: Pupil = field(default_factory=Pupil)


def moffat_stamp(size: int, fwhm: float, beta: float, x: float = 0.0, y: float = 0.0) -> np.ndarray:
    """A Moffat profile of unit light on a size × size grid, centred (x, y)
    from its middle."""
    alpha = fwhm / (2.0 * math.sqrt(2.0 ** (1.0 / beta) - 1.0))
    centre = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64) - centre
    profile = (1.0 + ((xx - x) ** 2 + (yy - y) ** 2) / alpha**2) ** -beta
    return profile / profile.sum()


def convolved(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`a` convolved with `b` (same size, centred), by Fourier transform."""
    fa = np.fft.rfft2(np.fft.ifftshift(a))
    fb = np.fft.rfft2(np.fft.ifftshift(b))
    return np.fft.fftshift(np.fft.irfft2(fa * fb, s=a.shape))


def pupil_psf(optics: Optics, size: int) -> np.ndarray:
    """The PSF of `optics` on a size × size stamp of the camera's pixels
    (size odd), of unit light, centred on its middle pixel."""
    if size % 2 == 0:
        raise ValueError(f"a stamp is an odd number of pixels across, not {size}")
    fine = size * OVERSAMPLE
    # The transform's grid: twice the fine stamp, so the pattern does not
    # wrap onto itself.
    grid = 2 * fine + 1
    total = np.zeros((fine, fine))
    for share in (BROAD if optics.broad else NARROW):
        # λ/D spans `diffraction_px` × share pixels, so OVERSAMPLE × that
        # fine samples; a pupil of D samples on a grid of N gives λ/D = N/D
        # samples.
        per_limit = optics.diffraction_px * share * OVERSAMPLE
        diameter = grid / per_limit
        if diameter > grid:
            raise ValueError("the diffraction limit is under a fine sample: raise diffraction_px")
        pupil = optics.pupil.mask(grid, diameter)
        image = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))) ** 2
        # The fine stamp with the pattern's middle (the grid's, which is odd)
        # at its own middle sample, the middle pixel's centre.
        start = grid // 2 - (fine - 1) // 2
        total += image[start:start + fine, start:start + fine] / image.sum()
    pixels = total.reshape(size, OVERSAMPLE, size, OVERSAMPLE).mean(axis=(1, 3))
    blurred = convolved(pixels, moffat_stamp(size, optics.seeing_fwhm, optics.seeing_beta))
    blurred = np.clip(blurred, 0.0, None)
    return blurred / blurred.sum()


def halo_wings(size: int, fwhm: float, beta: float = 1.5) -> np.ndarray:
    """A bright star's wide wings: a Moffat profile of a low β, of unit
    light, to be scaled by the share of the star's light they hold."""
    return moffat_stamp(size, fwhm, beta)


def reflection_ring(size: int, radius: float, width: float, x: float = 0.0, y: float = 0.0) -> np.ndarray:
    """A reflection's ring of unit light: an annulus of `radius` and a
    Gaussian cross-section of `width` (FWHM), centred (x, y) from the
    stamp's middle, as a reflection off a filter or a corrector throws."""
    centre = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64) - centre
    distance = np.hypot(xx - x, yy - y)
    ring = np.exp(-4.0 * math.log(2.0) * ((distance - radius) / width) ** 2)
    return ring / ring.sum()


def spike_directions(pupil: Pupil) -> list[float]:
    """The directions, in degrees from x in [0, 180), of the spikes a
    pupil's vanes throw: square to each vane, a vane and its opposite
    giving one line."""
    found = set()
    for k in range(pupil.vanes):
        theta = (pupil.angle + 360.0 * k / pupil.vanes + 90.0) % 180.0
        found.add(round(theta, 6))
    if pupil.hexagon:
        for k in range(3):
            found.add(round((60.0 * k + 90.0) % 180.0, 6))
    return sorted(found)


def light_along(stamp: np.ndarray, angle: float, inner: float, outer: float, width: float = 1.5) -> float:
    """The mean of `stamp` within `width` pixels of the line through its
    middle at `angle` degrees, between `inner` and `outer` from it."""
    size = stamp.shape[0]
    centre = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64) - centre
    theta = math.radians(angle)
    along = xx * math.cos(theta) + yy * math.sin(theta)
    across = -xx * math.sin(theta) + yy * math.cos(theta)
    chosen = (np.abs(across) <= width) & (np.abs(along) >= inner) & (np.abs(along) <= outer)
    return float(stamp[chosen].mean())


def spikes_seen(stamp: np.ndarray, directions: Sequence[float], inner: float, outer: float) -> dict[str, float]:
    """The light along a stamp's spikes against the light along lines
    between them, at the same distances."""
    on = [light_along(stamp, angle, inner, outer) for angle in directions]
    between = [light_along(stamp, (angle + 180.0 / (2 * max(len(directions), 1))) % 180.0, inner, outer)
               for angle in directions]
    return {"on": float(np.mean(on)), "between": float(np.mean(between))}
