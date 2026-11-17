"""Forced aperture photometry, and its agreement with photutils.

photutils is the established implementation: its exact circular apertures
must give the sums and areas Seiza gives, on whole apertures, on edges and
under masks, and its sigma-clipped annulus the same fluxes to within the
error Seiza reports. Those cases are skipped when photutils is not installed.
"""

import importlib.util
import math
import pathlib

import numpy as np
import pytest

import seiza

WIDTH, HEIGHT = 120, 96
GEOMETRY = dict(aperture_radius=5.0, annulus_inner=9.0, annulus_outer=15.0)


def field(stars, *, sigma=1.7, sky=lambda x, y: 150.0 + 0.4 * x - 0.25 * y):
    y, x = np.mgrid[:HEIGHT, :WIDTH].astype(float)
    data = sky(x, y)
    for (sx, sy), flux in stars:
        data = data + flux / (2 * np.pi * sigma**2) * np.exp(
            -((x - sx) ** 2 + (y - sy) ** 2) / (2 * sigma**2))
    return data


def enclosed(radius, sigma=1.7):
    return 1 - math.exp(-radius**2 / (2 * sigma**2))


STARS = [((40.3, 30.7), 8000.0), ((85.0, 60.0), 1500.0), ((20.6, 70.2), 300.0)]
POSITIONS = [position for position, _ in STARS]


def test_target_and_comparisons_are_measured_where_given_on_a_gradient():
    image = field(STARS).astype(np.float32)
    results = seiza.aperture_photometry(image, POSITIONS, **GEOMETRY)
    assert len(results) == len(STARS)
    for result, ((x, y), flux) in zip(results, STARS):
        assert (result.x, result.y) == (x, y)
        assert not result.flags
        assert result.flux == pytest.approx(flux * enclosed(5.0), rel=2e-3)
        assert result.background == pytest.approx(150.0 + 0.4 * x - 0.25 * y, abs=0.01)
        assert result.magnitude == pytest.approx(-2.5 * math.log10(result.flux))
        assert result.aperture_area == pytest.approx(math.pi * 25)
        # The sky is taken off under each pixel, not as one level times the
        # area, which on a gradient is a little different.
        assert result.aperture_sum == pytest.approx(
            result.flux + result.background * result.aperture_area, abs=0.5)
    assert "ApertureMeasurement(x=40.3, y=30.7, flux=" in repr(results[0])
    # Positions also come as an (n, 2) array.
    again = seiza.aperture_photometry(image, np.array(POSITIONS), **GEOMETRY)
    assert [r.flux for r in again] == [r.flux for r in results]


def test_a_green_plane_view_is_measured_without_conversion():
    green = field(STARS).astype(np.float32)
    rgb = np.stack([green * 0.5, green, green * 2.0], axis=-1)
    plane = rgb[..., 1]
    assert not plane.flags.c_contiguous
    expected = seiza.aperture_photometry(green, POSITIONS, **GEOMETRY)
    measured = seiza.aperture_photometry(plane, POSITIONS, **GEOMETRY)
    assert [m.flux for m in measured] == [e.flux for e in expected]
    # A colour array is refused, not averaged or picked from.
    with pytest.raises(ValueError, match="mono plane"):
        seiza.aperture_photometry(rgb, POSITIONS, **GEOMETRY)
    with pytest.raises(TypeError):
        seiza.aperture_photometry(green.astype(np.float64), POSITIONS, **GEOMETRY)


def test_flags_cover_saturation_masks_edges_and_unusable_positions():
    image = np.minimum(field(STARS), 400.0).astype(np.float32)
    positions = [*POSITIONS, (2.0, 50.0), (-60.0, 10.0)]
    results = seiza.aperture_photometry(image, positions, saturation=400.0, **GEOMETRY)
    assert len(results) == 5
    bright, medium, faint, edge, off = results
    assert bright.flags.saturated and bright.flux is not None and bright.peak == 400.0
    assert not medium.flags and not faint.flags
    assert edge.flags.missing_coverage and not edge.flags.unusable
    assert edge.aperture_area < math.pi * 25 - 5
    assert off.flags.unusable and off.flags.missing_coverage
    assert (off.flux, off.flux_error, off.magnitude, off.background) == (None,) * 4
    assert repr(off.flags) == "ApertureFlags(saturated=False, missing_coverage=True, unusable=True)"

    mask = np.zeros(image.shape, bool)
    mask[60, 86] = True
    clipped = np.zeros(image.shape, np.uint8)
    clipped[70, 21] = 1
    masked = seiza.aperture_photometry(image, POSITIONS, mask=mask,
                                       saturation_mask=clipped, **GEOMETRY)
    assert masked[1].flags.missing_coverage and not masked[1].flags.saturated
    assert masked[1].aperture_area == pytest.approx(math.pi * 25 - 1)
    assert masked[2].flags.saturated and masked[2].flux == faint.flux
    assert not masked[0].flags

    with pytest.raises(ValueError, match="shape"):
        seiza.aperture_photometry(image, POSITIONS, mask=mask[:-1], **GEOMETRY)
    with pytest.raises(ValueError, match="bool or uint8"):
        seiza.aperture_photometry(image, POSITIONS, mask=mask.astype(float), **GEOMETRY)
    with pytest.raises(ValueError, match="aperture"):
        seiza.aperture_photometry(image, POSITIONS, aperture_radius=5, annulus_inner=4,
                                  annulus_outer=9)
    with pytest.raises(ValueError):
        seiza.aperture_photometry(image, [(math.nan, 3.0)], **GEOMETRY)
    with pytest.raises(ValueError, match="positions"):
        seiza.aperture_photometry(image, [(1.0, 2.0, 3.0)], **GEOMETRY)
    with pytest.raises(TypeError):
        seiza.aperture_photometry(image, POSITIONS)


def test_faint_flux_keeps_its_sign_and_errors_follow_the_noise():
    rng = np.random.default_rng(7)
    empty = [(60.0, 48.0)]
    fluxes, errors = [], []
    for _ in range(300):
        image = (field([], sky=lambda x, y: 100.0 + 0 * x) +
                 rng.normal(0, 4.0, (HEIGHT, WIDTH))).astype(np.float32)
        result = seiza.aperture_photometry(image, empty, **GEOMETRY)[0]
        fluxes.append(result.flux)
        errors.append(result.flux_error)
        if result.flux <= 0:
            assert result.magnitude is None and result.magnitude_error is None
        else:
            assert result.magnitude_error == pytest.approx(
                2.5 / math.log(10) * result.flux_error / result.flux)
    assert 100 < sum(flux < 0 for flux in fluxes) < 200
    assert np.std(fluxes) == pytest.approx(np.mean(errors), rel=0.15)

    # A gain adds the source's photons; an uncertainty map replaces both.
    image = field(STARS).astype(np.float32)
    sky_only = seiza.aperture_photometry(image, POSITIONS[:1], **GEOMETRY)[0]
    with_gain = seiza.aperture_photometry(image, POSITIONS[:1], gain=2.0, **GEOMETRY)[0]
    assert with_gain.flux == sky_only.flux
    assert with_gain.flux_error**2 == pytest.approx(
        sky_only.flux_error**2 + with_gain.flux / 2.0, rel=1e-2)
    uncertainty = np.full(image.shape, 3.0, np.float32)
    mapped = seiza.aperture_photometry(image, POSITIONS[:1], gain=2.0,
                                       uncertainty=uncertainty, **GEOMETRY)[0]
    sky_term = mapped.aperture_area**2 * mapped.background_noise**2 / mapped.annulus_pixels
    # Each pixel's variance by its weight squared: a little under the area.
    assert 9.0 * (mapped.aperture_area - 8) < mapped.flux_error**2 - sky_term < 9.0 * mapped.aperture_area
    with pytest.raises(ValueError, match="gain"):
        seiza.aperture_photometry(image, POSITIONS, gain=0.0, **GEOMETRY)


def test_the_example_measures_a_dip_against_its_comparisons():
    path = pathlib.Path(__file__).parents[1] / "examples/aperture_photometry.py"
    spec = importlib.util.spec_from_file_location("seiza_aperture_photometry_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    rng = np.random.default_rng(3)
    target, comparisons = example.POSITIONS[0], example.POSITIONS[1:]
    measure = lambda flux: example.differential_magnitude(
        example.synthetic_sub(flux, rng), target, comparisons, gain=1.0)
    bright, bright_error = measure(30_000.0)
    faint, _ = measure(30_000.0 * 10 ** (-0.4 * 0.5))
    assert bright == pytest.approx(2.5 * math.log10(65_000 / 30_000), abs=4 * bright_error)
    assert faint - bright == pytest.approx(0.5, abs=0.06)
    assert 0.005 < bright_error < 0.03
    # A saturated comparison gives no number.
    assert example.differential_magnitude(
        example.synthetic_sub(30_000.0, rng), target, comparisons, saturation=100.0) == (None, None)


# --- against photutils ---------------------------------------------------


def photutils_aperture():
    return pytest.importorskip("photutils.aperture")


def test_aperture_sums_and_areas_match_photutils_exact_apertures():
    aperture = photutils_aperture()
    rng = np.random.default_rng(11)
    image = (field(STARS) + rng.normal(0, 2.0, (HEIGHT, WIDTH))).astype(np.float32)
    mask = rng.random(image.shape) < 0.03
    # Subpixel shifts, small and large radii, and apertures over two edges.
    positions = [(40.0 + 0.1 * i, 30.0 + 0.07 * i) for i in range(10)]
    positions += [(1.3, 50.2), (118.9, 94.6), (60.5, 0.4), (85.0, 60.0)]
    for radius in (0.6, 2.5, 5.0, 11.3):
        geometry = dict(aperture_radius=radius, annulus_inner=radius + 4,
                        annulus_outer=radius + 9)
        apertures = aperture.CircularAperture(positions, radius)
        for use_mask in (None, mask):
            table = aperture.aperture_photometry(
                image.astype(float), apertures, method="exact", mask=use_mask)
            areas = apertures.area_overlap(image, mask=use_mask, method="exact")
            results = seiza.aperture_photometry(image, positions, mask=use_mask, **geometry)
            for result, expected, area in zip(results, table["aperture_sum"], areas):
                assert result.aperture_area == pytest.approx(area, rel=1e-10, abs=1e-10)
                assert result.aperture_sum == pytest.approx(expected, rel=1e-9, abs=1e-7)
                whole = math.isclose(area, math.pi * radius**2, rel_tol=1e-9)
                assert result.flags.missing_coverage == (not whole)


def test_fluxes_and_errors_agree_with_a_photutils_annulus_reduction():
    aperture = photutils_aperture()
    stats = pytest.importorskip("astropy.stats")
    rng = np.random.default_rng(23)
    sigma = 3.0
    image = (field(STARS) + rng.normal(0, sigma, (HEIGHT, WIDTH))).astype(np.float32)
    error = np.full(image.shape, sigma, np.float32)
    apertures = aperture.CircularAperture(POSITIONS, GEOMETRY["aperture_radius"])
    annuli = aperture.CircularAnnulus(
        POSITIONS, GEOMETRY["annulus_inner"], GEOMETRY["annulus_outer"])
    table = aperture.aperture_photometry(
        image.astype(float), apertures, error=error.astype(float), method="exact")
    sky = aperture.ApertureStats(
        image.astype(float), annuli, sigma_clip=stats.SigmaClip(sigma=3.0, maxiters=10))
    results = seiza.aperture_photometry(image, POSITIONS, uncertainty=error, **GEOMETRY)
    for result, row, background, scatter in zip(results, table, sky.median, sky.std):
        expected = row["aperture_sum"] - background * apertures.area
        # Two sky estimators on the same pixels: a clipped plane and a
        # clipped median. They differ by less than the error either reports.
        assert abs(result.flux - expected) < result.flux_error
        assert result.background == pytest.approx(background, abs=4 * sigma / math.sqrt(400))
        # The annulus's scatter is about the fitted plane; a plain standard
        # deviation also holds the gradient across the annulus.
        assert result.background_noise == pytest.approx(sigma, rel=0.2)
        assert scatter > result.background_noise
        # On this nearly symmetric annulus the plane covariance reduces
        # approximately to the mean-sky term. photutils supplies only the
        # aperture error, each pixel's variance by its
        # weight squared; Seiza adds the sky fit's.
        sky_term = result.aperture_area**2 * result.background_noise**2 / result.annulus_pixels
        assert result.flux_error**2 == pytest.approx(
            row["aperture_sum_err"] ** 2 + sky_term, rel=0.01)
    # Without a map, the annulus's own scatter stands in for it.
    plain = seiza.aperture_photometry(image, POSITIONS, **GEOMETRY)
    for result, mapped in zip(plain, results):
        assert result.flux == mapped.flux
        assert result.flux_error == pytest.approx(mapped.flux_error, rel=0.15)


def test_asymmetric_annulus_uncertainty_and_post_clip_coverage():
    rng = np.random.default_rng(109)
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    dx, dy = x - 60, y - 48
    radius = np.hypot(dx, dy)
    mask = (dx < -1) & (radius >= 10)
    fluxes, errors = [], []
    for _ in range(400):
        image = (100 + rng.normal(0, 3, (HEIGHT, WIDTH))).astype(np.float32)
        result = seiza.aperture_photometry(
            image, [(60, 48)], aperture_radius=8, annulus_inner=10,
            annulus_outer=14, mask=mask)[0]
        assert not result.flags
        fluxes.append(result.flux)
        errors.append(result.flux_error)
    assert np.std(fluxes) == pytest.approx(np.mean(errors), rel=0.15)

    annulus = (radius >= 10) & (radius <= 16)
    mask = annulus & (dx < -2)
    contamination = annulus & ~mask & (x % 5 == 0)
    image = (100 + rng.normal(0, 1, (HEIGHT, WIDTH))).astype(np.float32)
    image[contamination] += 1000
    assert np.count_nonzero(annulus & ~mask) > np.count_nonzero(annulus) / 2
    result = seiza.aperture_photometry(
        image, [(60, 48)], aperture_radius=6, annulus_inner=10,
        annulus_outer=16, mask=mask)[0]
    assert result.flags.unusable
    assert result.flux is None and result.flux_error is None
