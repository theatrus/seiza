import copy

import numpy as np
import pytest
import seiza


def sources(width=256, height=256, count=1):
    def source(name):
        return dict(name=name, revision=f"{name}:linear-v1", transfer="linear", saturation=65535)
    return dict(width=width, height=height, narrowband=source("ha"),
                references=[source(n) for n in ["red", "green", "blue"][:count]])


def stars():
    def measurement(flux, i):
        return dict(x=25 + i % 7 * 30, y=25 + i // 7 * 20,
                    flux=flux, peak=flux / 20 + 100, fwhm=3, snr=100)
    return [dict(narrowband=measurement(0.35 * (1000 + i * 100), i),
                 references=[measurement(1000 + i * 100, i)]) for i in range(40)]


def test_fit_returns_core_plan_and_skips_unreliable_stars():
    context = sources()
    outcome = seiza.fit_continuum(stars(), context, options={"minimum_stars": 12})
    assert outcome["status"] == "fitted"
    plan = outcome["plan"]
    assert plan["sources"] == context
    assert plan["coefficients"] == pytest.approx([0.35])
    assert plan["diagnostics"]["validationStars"] >= 3
    assert len(plan["diagnostics"]["spatialResiduals"]) == 4
    bad = stars()
    for star in bad:
        star["references"][0]["fwhm"] = 10
    skipped = seiza.fit_continuum(bad, context)
    assert skipped["status"] == "skipped"
    assert skipped["reason"] == "psf_mismatch"
    with pytest.raises(ValueError):
        seiza.fit_continuum(stars(), context, options={"typo": 1})


def test_subtraction_preserves_signed_samples_coverage_and_inputs():
    context = sources(4, 1, 2)
    plan = dict(schemaVersion=1, sources=copy.deepcopy(context), coefficients=[0.35, 0],
                origin="manual", diagnostics=None)
    n = np.array([[0.2, 0.1, 0.5, np.nan]], dtype=np.float32)
    r = np.array([[0.5, 0.3, 0.4, 1]], dtype=np.float32)
    before = n.copy()
    result = seiza.subtract_continuum(n, [r, np.full_like(r, np.nan)], context, plan)
    np.testing.assert_allclose(result, [[0.025, -0.005, 0.36, np.nan]], atol=1e-7)
    np.testing.assert_array_equal(n, before)
    assert not np.shares_memory(result, n)
    context["narrowband"]["revision"] = "changed"
    with pytest.raises(ValueError, match="revisions"):
        seiza.subtract_continuum(n, [r, r], context, plan)
    context["narrowband"]["transfer"] = "display"
    with pytest.raises(ValueError, match="linear"):
        seiza.subtract_continuum(n, [r, r], context, plan)


def test_automatic_and_supplied_star_photometry_agree_across_gradients():
    yy, xx = np.mgrid[:256, :256]
    continuum = np.zeros((256, 256), dtype=np.float64)
    positions = [(35 + x * 60, 35 + y * 60) for y in range(4) for x in range(4)]
    for i, (x, y) in enumerate(positions):
        continuum += (1000 + 50 * i) * np.exp(-((xx-x)**2 + (yy-y)**2) / (2 * 1.4**2))
    r = np.ascontiguousarray(300 + 0.2 * xx - 0.1 * yy + continuum, dtype=np.float32)
    n = np.ascontiguousarray(100 - 0.1 * xx + 0.15 * yy + 0.35 * continuum, dtype=np.float32)
    context = sources()
    for locations in [positions, None]:
        outcome = seiza.measure_and_fit_continuum(n, [r], context, positions=locations,
            fit_options={"minimum_stars": 12}, photometry_options={"aperture_radius": 8})
        assert outcome["status"] == "fitted", outcome
        assert outcome["plan"]["coefficients"] == pytest.approx([0.35], abs=2e-5)
        result = seiza.subtract_continuum(n, [r], context, outcome["plan"])
        assert result.dtype == np.float32
    with pytest.raises(ValueError, match="mono"):
        seiza.measure_and_fit_continuum(np.repeat(n[..., None], 3, axis=2), [r], context)
    with pytest.raises(ValueError, match="contiguous"):
        seiza.measure_and_fit_continuum(n[:, ::-1], [r], context)
    empty = seiza.measure_and_fit_continuum(np.zeros_like(n), [r], context)
    assert empty["status"] == "skipped"
    assert empty["reason"] == "no_stars"
