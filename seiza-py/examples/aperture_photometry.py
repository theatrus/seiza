"""Measure a variable star against comparison stars on one mono plane.

Forced aperture photometry: every star is measured where you say it is, on a
linear plane such as a mono sub or the green channel of a debayered Seestar
sub. Run it on each sub of a series and plot the differential magnitude
against time.

    python aperture_photometry.py            # a synthetic field
"""
import math

import numpy as np

import seiza


def differential_magnitude(plane, target, comparisons, *, aperture_radius=6.0,
                           annulus_inner=10.0, annulus_outer=16.0, **options):
    """Target minus the comparison ensemble, in magnitudes, with its error.

    `plane` is a linear float32 (height, width) array; `target` and each of
    `comparisons` an (x, y) position, zero-based. Returns (None, None) when a
    star cannot be used: saturated, off the frame, masked or not above zero.
    `options` go to seiza.aperture_photometry: saturation=, gain=, mask=, ...
    """
    stars = seiza.aperture_photometry(
        plane, [target, *comparisons], aperture_radius=aperture_radius,
        annulus_inner=annulus_inner, annulus_outer=annulus_outer, **options)
    if any(star.flags or star.magnitude is None for star in stars):
        return None, None
    target, comparisons = stars[0], stars[1:]
    # The ensemble is the comparisons' summed flux, and its error theirs in quadrature.
    ensemble = sum(star.flux for star in comparisons)
    ensemble_error = math.sqrt(sum(star.flux_error ** 2 for star in comparisons))
    magnitude = target.magnitude + 2.5 * math.log10(ensemble)
    error = math.hypot(target.magnitude_error,
                       2.5 / math.log(10) * ensemble_error / ensemble)
    return magnitude, error


def synthetic_sub(target_flux, rng):
    """A 160 x 120 sub: sky with a gradient, a target and two comparisons."""
    y, x = np.mgrid[:120, :160]
    plane = 200.0 + 0.3 * x - 0.1 * y
    for (sx, sy), flux in zip(POSITIONS, [target_flux, 40_000.0, 25_000.0]):
        plane += flux / (2 * np.pi * 4.0) * np.exp(-((x - sx) ** 2 + (y - sy) ** 2) / 8.0)
    return rng.poisson(plane).astype(np.float32)


POSITIONS = [(80.3, 60.7), (40.6, 30.2), (120.1, 85.5)]

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    for index in range(8):
        # The target dims by half a magnitude and comes back.
        dip = 0.5 * math.sin(math.pi * index / 7)
        sub = synthetic_sub(30_000.0 * 10 ** (-0.4 * dip), rng)
        # An RGB sub would pass its green plane: rgb[..., 1].
        magnitude, error = differential_magnitude(
            sub, POSITIONS[0], POSITIONS[1:], saturation=65535.0, gain=1.0)
        print(f"sub {index}: {magnitude:+.3f} +/- {error:.3f} mag")
