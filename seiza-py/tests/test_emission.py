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
