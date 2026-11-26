"""Stars for the starless model's composites: pupil PSFs with their spikes,
and bright stars' halos."""

import math

import numpy as np
import pytest

from seiza_training import starless_stars as stars


def moments(stamp):
    size = stamp.shape[0]
    centre = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64) - centre
    total = stamp.sum()
    x, y = (stamp * xx).sum() / total, (stamp * yy).sum() / total
    xx2 = (stamp * (xx - x) ** 2).sum() / total
    yy2 = (stamp * (yy - y) ** 2).sum() / total
    xy = (stamp * (xx - x) * (yy - y)).sum() / total
    return x, y, math.hypot(xx2 - yy2, 2 * xy) / (xx2 + yy2)


def test_a_clear_pupil_makes_a_round_star_of_unit_light_at_the_middle():
    psf = stars.pupil_psf(stars.Optics(diffraction_px=0.8, seeing_fwhm=2.5), 63)
    assert psf.sum() == pytest.approx(1.0) and psf.min() >= 0.0
    x, y, ellipticity = moments(psf)
    assert abs(x) < 0.02 and abs(y) < 0.02
    assert ellipticity < 0.01
    assert np.unravel_index(np.argmax(psf), psf.shape) == (31, 31)
    with pytest.raises(ValueError):
        stars.pupil_psf(stars.Optics(), 64)


@pytest.mark.parametrize("vanes, angle, expected", [(4, 0.0, [0.0, 90.0]), (4, 30.0, [30.0, 120.0]),
                                                    (3, 0.0, [30.0, 90.0, 150.0])])
def test_vanes_throw_spikes_square_to_them_and_none_between(vanes, angle, expected):
    pupil = stars.Pupil(obstruction=0.3, vanes=vanes, vane_width=0.02, angle=angle)
    assert stars.spike_directions(pupil) == pytest.approx(expected)
    psf = stars.pupil_psf(stars.Optics(diffraction_px=1.2, seeing_fwhm=1.5, pupil=pupil), 101)
    seen = stars.spikes_seen(psf, stars.spike_directions(pupil), 12, 45)
    assert seen["on"] > 3.0 * seen["between"]
    clear = stars.pupil_psf(stars.Optics(diffraction_px=1.2, seeing_fwhm=1.5,
                                         pupil=stars.Pupil(obstruction=0.3)), 101)
    plain = stars.spikes_seen(clear, stars.spike_directions(pupil), 12, 45)
    assert plain["on"] == pytest.approx(plain["between"], rel=0.3)


def test_a_hexagon_with_a_strut_throws_six_spikes_and_two():
    pupil = stars.Pupil(obstruction=0.1, vanes=2, vane_width=0.02, angle=90.0, hexagon=True)
    directions = stars.spike_directions(pupil)
    assert directions == pytest.approx([0.0, 30.0, 90.0, 150.0])
    psf = stars.pupil_psf(stars.Optics(diffraction_px=1.5, seeing_fwhm=1.2, pupil=pupil), 101)
    seen = stars.spikes_seen(psf, directions, 12, 45)
    assert seen["on"] > 3.0 * seen["between"]


def test_a_broad_filter_smears_the_rings_a_narrow_one_keeps():
    optics = dict(diffraction_px=2.5, seeing_fwhm=0.6, pupil=stars.Pupil(obstruction=0.35))
    narrow = stars.pupil_psf(stars.Optics(**optics), 81)
    broad = stars.pupil_psf(stars.Optics(broad=True, **optics), 81)
    profile = lambda stamp: stamp[40, 40:]  # noqa: E731
    # The narrow filter's rings dip deeper between them than the broad's.
    def contrast(row):
        segment = row[3:20]
        return (segment.max() - segment.min()) / segment.max()
    assert contrast(profile(narrow)) > contrast(profile(broad))


def test_halos_are_stamps_of_unit_light_where_asked():
    wings = stars.halo_wings(201, 20.0)
    assert wings.sum() == pytest.approx(1.0)
    ring = stars.reflection_ring(201, 60.0, 8.0, x=5.0)
    assert ring.sum() == pytest.approx(1.0)
    centre = 100
    assert ring[centre, centre + 65] > 10 * ring[centre, centre + 30]
    # Off the star by 5 px: its peaks along the row 65 px out on one side
    # and 55 on the other.
    row = ring[centre]
    assert int(np.argmax(row[centre:])) == 65 and int(np.argmax(row[:centre][::-1])) + 1 == 55
