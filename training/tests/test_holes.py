"""The hole guard: what it masks, what it fills the mask with, and what it
puts back after a network has run."""

import numpy as np
import pytest

pytest.importorskip("torch")

from seiza_training.holes import (HoleGuard, filled, grown, guarded, guarded_pair, guarded_residual,  # noqa: E402
                                  hole_mask, restored, sky_floor)
from seiza_training.noise_model import image_sigma  # noqa: E402

GUARD = HoleGuard()


def sky(height: int = 256, width: int = 320, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:height, :width]
    return (100.0 + 0.02 * x + 0.01 * y + rng.normal(0.0, 2.0, (height, width))).astype(np.float32)


def lanczos3(offset: float) -> np.ndarray:
    taps = np.arange(-2, 4) - offset
    weights = np.sinc(taps) * np.sinc(taps / 3)
    return weights / weights.sum()


def test_growth_cuts_at_the_edges():
    point = np.zeros((9, 9), dtype=bool)
    point[0, 4] = True
    assert grown(point, 2).sum() == 3 * 5
    assert not grown(np.zeros((4, 4), dtype=bool), 3).any()


def test_a_hole_and_its_lobes_are_masked_filled_and_put_back():
    image = sky()
    image[NaN_ROWS] = np.nan
    # A sharp hole, as a drizzle leaves one, and one registration spread.
    image[60, 80] -= 3000.0
    image[61, 80] -= 2000.0
    spread = np.outer(lanczos3(0.4), lanczos3(0.7)) * -1500.0
    image[150:156, 200:206] += spread.astype(np.float32)
    read, mask, noise_map = guarded(image, GUARD)
    assert mask[60, 80] and mask[61, 80] and mask[152, 202]
    assert mask[60 - 3, 80 + 3] and not mask[60 - 4, 80]
    assert mask.sum() < 2 * 11 * 12, "only the holes and their margins"
    assert np.isnan(read[NaN_ROWS]).all() and not mask[NaN_ROWS].any()
    assert abs(read[60, 80] - 101.6) < 3 and abs(read[152, 202] - 103.0) < 3, "filled near the sky"
    assert np.array_equal(read[~mask], image[~mask], equal_nan=True), "nothing else changes"

    # A network that takes the noise out entirely, as the sky's smooth trend.
    y, x = np.mgrid[:image.shape[0], :image.shape[1]]
    smooth = (100.0 + 0.02 * x + 0.01 * y).astype(np.float32)
    found = np.where(np.isfinite(read), smooth - read, np.nan).astype(np.float32)
    residual = guarded_residual(image, read, found, mask, noise_map, GUARD)
    output = image + residual
    assert output[60, 80] == image[60, 80] and output[61, 80] == image[61, 80], "the hole comes back as it was"
    lobe = np.unravel_index(np.argmax(np.where(mask, image, -np.inf)), image.shape)
    assert output[lobe] == image[lobe] and image[lobe] - smooth[lobe] > 100, "and its bright lobe"
    quiet = mask & (np.abs(image - smooth) < 3.0)
    assert quiet.sum() > 50 and np.allclose(output[quiet], smooth[quiet]), "the margin's noise is taken out"
    outside = np.isfinite(image) & ~mask
    assert np.array_equal(residual[outside], found[outside]), "outside the mask, the network's own"


NaN_ROWS = (slice(0, 10), slice(0, 40))


def test_ordinary_noise_stars_and_edges_hold_no_hole():
    image = sky(seed=8)
    height, width = image.shape
    y, x = np.mgrid[:height, :width]
    # A star wider than a block, its core high above the sky.
    image += (60000.0 * np.exp(-((y - 128) ** 2 + (x - 100) ** 2) / (2 * 14.0**2))).astype(np.float32)
    # A bright nebula with a sharp edge, and a dark lane in it that stays
    # above the sky.
    image += np.where(x > 220, 3000.0, 0.0).astype(np.float32)
    image[:, 260:263] -= 2800.0
    read, mask, _ = guarded(image, GUARD)
    assert not mask.any()
    assert read is image, "an image with no hole is read as it is"


def test_the_floor_passes_over_a_lifted_block_and_a_shallow_dip_is_not_a_hole():
    image = sky(seed=11)
    image[64:96, 64:96] += 5000.0
    floor = sky_floor(image)
    assert abs(floor[80, 80] - floor[80, 160]) < 10, "the bright block's neighbours hold the floor"
    noise_map = image_sigma(image)
    image[200, 250] -= 25 * float(np.nanmedian(noise_map))
    assert not hole_mask(image, GUARD, noise_map, sky_floor(image)).any(), "25σ̄ is left to the network"
    image[200, 250] -= 40 * float(np.nanmedian(noise_map))
    assert hole_mask(image, GUARD, noise_map, sky_floor(image))[200, 250]


def test_a_star_beside_a_hole_does_not_lift_its_fill():
    image = sky(seed=5)
    y, x = np.mgrid[:image.shape[0], :image.shape[1]]
    image += (4000.0 * np.exp(-((y - 100) ** 2 + (x - 108) ** 2) / (2 * 1.6**2))).astype(np.float32)
    mask = np.zeros(image.shape, dtype=bool)
    mask[97:104, 97:104] = True
    floor = sky_floor(image)
    read = filled(image, mask, GUARD, floor)
    assert np.abs(read[mask] - 101.5).max() < 3, "the sky's level, not a plateau lit by the star"


def test_a_map_without_noise_holds_no_hole():
    image = np.full((96, 96), 100.0, dtype=np.float32)
    image[40, 40] = 0.0
    assert not hole_mask(image, GUARD, np.zeros_like(image), sky_floor(image)).any()


def test_a_pair_trains_with_the_input_s_holes_filled_and_the_target_s_too():
    image_a, image_b = sky(seed=21), sky(seed=22)
    image_a[60, 80] -= 3000.0
    image_b[60, 80] -= 3000.0
    image_b[150, 200] -= 2500.0
    read_a, read_b = guarded_pair(image_a, image_b, GUARD)
    assert abs(read_a[60, 80] - 101.6) < 3 and abs(read_b[60, 80] - 101.6) < 3, "a hole both hold"
    assert read_a[150, 200] == image_a[150, 200], "the input keeps what is no hole in it"
    assert abs(read_b[150, 200] - 103.5) < 3, "the target asks for no hole the input does not show"
    untouched = np.ones(image_a.shape, dtype=bool)
    untouched[55:66, 75:86] = untouched[145:156, 195:206] = False
    assert np.array_equal(read_a[untouched], image_a[untouched])
    assert np.array_equal(read_b[untouched], image_b[untouched])


def test_putting_back_needs_a_difference_past_the_threshold():
    image = np.full((4, 4), 10.0, dtype=np.float32)
    output = image.copy()
    output[0, 0], output[1, 1] = 14.0, 16.0
    mask = np.ones((4, 4), dtype=bool)
    back = restored(image, output, mask, np.ones((4, 4), dtype=np.float32), HoleGuard(restore=5.0))
    assert back[0, 0] == 14.0 and back[1, 1] == 10.0
