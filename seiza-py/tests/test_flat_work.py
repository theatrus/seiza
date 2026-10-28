"""PSF Guard can drive the shared stages from cached records and array kernels."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import seiza


def library():
    path = Path(__file__).resolve().parents[2] / 'seiza-calibration/tests/fixtures/library/psf-guard.json'
    result = copy.deepcopy(json.loads(path.read_text())['cases'][12]['request'])
    result['grouping']['lights'] = []
    result['grouping']['responses'][0]['rotation'] = 86.0
    result['grouping']['comparisons'][0]['dust'] = 0.03
    for frame in result['frames']:
        if frame['frame_uuid'].startswith('a-'):
            frame['rotation'] = 86.0
    first = result['lights'][0]
    result['lights'] = [dict(first, path=f'/absent/light-{i}.fits', timestamp=345600+i*60,
                            rotator_position=89.0) for i in range(12)]
    return result


def test_cached_metadata_questions_feedback_and_no_read_reuse():
    source = library()
    contexts = [dict(light_index=i, id=100+i, target='M42', pedestal_adu=123.0) for i in range(12)]
    plan = seiza.calibration_flat_work_plan(library=source, light_context=contexts)
    assert len(plan['families']) == 1
    question = plan['families'][0]
    assert len(question['sample_lights']) == 8
    assert question['sample_lights'][0] == 100 and question['sample_lights'][-1] == 111
    assert question['pedestal_adu'] == 123.0
    assert len(question['candidates']) == 2
    # Unread/insufficient backgrounds choose nothing; no host tie policy needed.
    result = seiza.calibration_flat_work_finish(kind='family', question=question,
                                              choice=dict(winners=[], pairs=[]), lights=3)
    assert result['kind'] == 'family' and result['row']['chosen'] is None
    source['grouping'] = plan['grouping']
    source['grouping']['family_evidence'].append(result['row'])
    change = plan['changes'][0]
    row = seiza.calibration_flat_work_finish(dict(kind='change', question=change, fits=[]))['row']
    assert row['verdict'] == 'undecided'
    source['grouping']['evidence'].append(row)
    reused = seiza.calibration_flat_work_plan(dict(library=source, light_context=contexts))
    assert reused['families'][0]['answered']
    assert reused['families'][0]['sample_lights'] == []
    assert reused['changes'][0]['answered']
    with pytest.raises(ValueError):
        seiza.calibration_flat_work_plan(library=source, light_context=[dict(light_index=1000)])
    with pytest.raises(ValueError):
        seiza.calibration_flat_work_finish(kind='family', question=question,
                                          choice=dict(winners=[99], pairs=[]), lights=3)


def test_dust_evidence_map_has_the_same_array_surface_as_other_kernels():
    before = np.ones((64, 64), dtype=np.float32)
    after = before.copy()
    after[24:40, 24:40] = .9
    actual = seiza.flat_evidence_change_map(before, after)
    assert actual is not None
    expected = seiza.flat_change_map(before, after)
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match='bin'):
        seiza.flat_evidence_change_map(before, after, bin=0)


def test_cached_response_pedestal_removal_uses_the_native_kernel():
    plane = np.array([[1.0, .8]], dtype=np.float32)
    corrected = seiza.flat_response_without_pedestal(plane, level=100.0, pedestal=20.0)
    np.testing.assert_allclose(corrected, [[1.0, .75]], atol=1e-7)
    np.testing.assert_array_equal(plane, [[1.0, np.float32(.8)]])
