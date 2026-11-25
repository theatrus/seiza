"""Wcs array transforms, and reproject between two plate solutions."""

import numpy as np
import pytest
import seiza


def sky(ra, dec):
    """A smooth sky brightness, so any grid can be checked against it."""
    return 1000.0 + 300.0 * np.sin((ra - 150.0) * 20.0) * np.cos((dec - 30.0) * 15.0)


def render(wcs, width, height):
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    ra, dec = wcs.pixel_to_world_array(x, y)
    return sky(ra, dec).astype(np.float32)


def test_array_transforms_match_one_point_at_a_time():
    wcs = seiza.Wcs.from_center_scale_rotation((150.0, 30.0), (99.5, 49.5), 2.0, 20.0, True)
    x = np.array([[0.0, 99.5], [180.25, 17.0]])
    y = np.array([[0.0, 49.5], [12.5, 90.0]])
    ra, dec = wcs.pixel_to_world_array(x, y)
    assert ra.shape == dec.shape == (2, 2)
    for index in np.ndindex(x.shape):
        assert (ra[index], dec[index]) == wcs.pixel_to_world(x[index], y[index])

    back_x, back_y = wcs.world_to_pixel_array(ra, dec)
    np.testing.assert_allclose(back_x, x, atol=1e-6)
    np.testing.assert_allclose(back_y, y, atol=1e-6)

    # The far side of the sky has no pixel.
    far_x, far_y = wcs.world_to_pixel_array(np.array([330.0]), np.array([-30.0]))
    assert np.isnan(far_x[0]) and np.isnan(far_y[0])

    with pytest.raises(ValueError, match="share a shape"):
        wcs.pixel_to_world_array(np.zeros(3), np.zeros(4))


def test_a_rotated_image_lands_on_a_north_up_grid():
    source_wcs = seiza.Wcs.from_center_scale_rotation(
        (150.0, 30.0), (199.5, 149.5), 2.0, 37.0, True
    )
    source = render(source_wcs, 400, 300)
    grid = seiza.Wcs.from_center_scale_rotation((150.0, 30.0), (99.5, 99.5), 3.0, 0.0, False)

    image, coverage = seiza.reproject(source, source_wcs, grid, 200, 200, interpolation="lanczos3")
    assert image.shape == coverage.shape == (200, 200)
    assert image.dtype == coverage.dtype == np.float32
    inside = coverage == 1.0
    assert inside.sum() > 25_000
    np.testing.assert_allclose(image[inside], render(grid, 200, 200)[inside], atol=2.0)
    # The grid's corners lie off the rotated source.
    assert coverage[0, 0] == 0.0 and np.isnan(image[0, 0])


def test_an_rgb_image_keeps_its_channels():
    wcs = seiza.Wcs.from_center_scale_rotation((150.0, 30.0), (31.5, 23.5), 2.0, 0.0, False)
    rgb = np.stack([np.full((48, 64), value, np.float32) for value in (1.0, 2.0, 3.0)], axis=-1)
    image, coverage = seiza.reproject(rgb, wcs, wcs, 64, 48)
    assert image.shape == (48, 64, 3)
    np.testing.assert_allclose(image[10, 10], [1.0, 2.0, 3.0], atol=1e-5)
    assert np.all(coverage == 1.0)


def test_a_coarser_grid_averages_rather_than_aliases():
    wcs = seiza.Wcs.from_center_scale_rotation((150.0, 30.0), (255.5, 255.5), 1.0, 0.0, False)
    checkerboard = (np.indices((512, 512)).sum(axis=0) % 2 * 2.0).astype(np.float32)
    grid = seiza.Wcs.from_center_scale_rotation((150.0, 30.0), (31.5, 31.5), 7.3, 0.0, False)

    smooth, _ = seiza.reproject(checkerboard, wcs, grid, 64, 64)
    assert np.all(np.abs(smooth[8:56, 8:56] - 1.0) < 0.2)
    point, _ = seiza.reproject(checkerboard, wcs, grid, 64, 64, antialias=False)
    assert np.ptp(point[8:56, 8:56]) > 1.0


def test_bad_requests_raise():
    wcs = seiza.Wcs.from_center_scale_rotation((150.0, 30.0), (9.5, 9.5), 2.0, 0.0, False)
    image = np.ones((20, 20), np.float32)
    with pytest.raises(ValueError, match="interpolation"):
        seiza.reproject(image, wcs, wcs, 20, 20, interpolation="cubic")
    with pytest.raises(ValueError):
        seiza.reproject(image, wcs, wcs, 0, 20)
