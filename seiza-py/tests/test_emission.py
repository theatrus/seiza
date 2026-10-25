import numpy as np
import pytest
import seiza


def test_projection_structure_and_calibrated_addition_use_core_units():
    source = lambda n: dict(name=n, revision='v1', transfer='linear')
    plan = dict(schemaVersion=1, sources=dict(width=1, height=1, narrowband=source('ha'), references=[source('red')]),
                coefficients=[0.2], origin='manual', diagnostics=None)
    line = dict(plan=plan, reference_channels=[0], amount=0.25)
    base = np.array([[[0.2, 0.3, 0.4]]], dtype=np.float32)
    residual = np.array([[0.04]], dtype=np.float32)
    np.testing.assert_allclose(seiza.blend_emission_rgb(base, [residual], [line]), [[[0.25, 0.3, 0.4]]], atol=1e-6)
    before = residual.copy()
    seiza.blend_emission_rgb(base, [residual], [line], negative='positive_contribution')
    np.testing.assert_array_equal(residual, before)
    np.testing.assert_allclose(seiza.blend_display_structure(base, base * 2, 0.5), base * 1.5, atol=1e-6)
    np.testing.assert_allclose(seiza.add_rgb_contribution(base, base, gains=[2, 1, 0.5]), base * [3, 2, 1.5], atol=1e-6)
    with pytest.raises(ValueError):
        seiza.blend_display_structure(base, base * 2, 1.5)
    with pytest.raises(ValueError):
        seiza.add_rgb_contribution(base, base, gains=[0, 1, 1])


def test_green_reduction_is_explicit_display_only_and_keeps_input():
    image = np.array([[[.2,.8,.3],[.4,.4,.4]]], dtype=np.float32)
    original = image.copy()
    result = seiza.reduce_green_excess(image, .5)
    np.testing.assert_allclose(result, [[[.2,.55,.3],[.4,.4,.4]]], rtol=1e-6)
    np.testing.assert_array_equal(image, original)
    with pytest.raises(ValueError):
        seiza.reduce_green_excess(image, 2)


def test_calibrated_flux_uses_exact_responses_and_unknown_covariance():
    source = lambda name: dict(name=name, revision='exact-v1', transfer='linear')
    sources = dict(width=2, height=1, narrowband=source('ha'), references=[source('red')])
    plan = dict(schemaVersion=1, sources=sources, coefficients=[.2], origin='manual', diagnostics=None)
    calibration = dict(revision='lab-v1', unit='flux/pixel', lines=['ha'], sources=[sources],
        narrowband_response=[[1.]], reference_response=[[[1.]]], continuum_coefficients=[[.2]], residual_zero_levels=[0.], residual_covariance=[[4.]])
    residual = np.array([[4., -4.]], dtype=np.float32)
    images, report = seiza.solve_emission_flux([residual], [plan], calibration)
    np.testing.assert_allclose(images[0], [[5., -5.]])
    assert report['fluxCovariance'] == [[6.25]]
    assert report['unit'] == 'flux/pixel'
    calibration['residual_covariance'] = None
    assert seiza.solve_emission_flux([residual], [plan], calibration)[1]['fluxCovariance'] is None


def test_linked_star_reduction_shrinks_only_the_screen_star_layer():
    nebula = np.tile(np.array([.1,.2,.3], dtype=np.float32), (3,3,1))
    starless = np.full((3,3,3), .2, dtype=np.float32)
    levels = np.array([[.05,.2,.05],[.2,.8,.2],[.05,.2,.05]], dtype=np.float32)
    stars = levels[...,None] * np.array([1,.5,.25], dtype=np.float32)
    original = np.ascontiguousarray(1-(1-starless)*(1-stars))
    result = seiza.screen_rgb_stars_reduced(nebula, original, starless, .5)
    np.testing.assert_allclose(result[1,1], [.55,.4,.3875], rtol=1e-6)
    np.testing.assert_array_equal(nebula[1,1], np.array([.1,.2,.3], dtype=np.float32))
    with pytest.raises(ValueError):
        seiza.screen_rgb_stars_reduced(nebula, original, starless, -1)
