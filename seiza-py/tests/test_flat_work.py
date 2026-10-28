"""A host drives the shared flat-work stages from cached records and array kernels."""
import copy

import numpy as np
import pytest
import seiza

from calibration_host import cases, groups_of, resolved, rules_of


def library():
    result = copy.deepcopy(cases('psf-guard.json')[12]['request'])
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


def flat_work(source, grouping=None):
    """The family and change questions of the source's lights, as a host
    asks them: imaging sessions from its own lights, each light's evidence
    record from its catalog row, its header from the file it keeps."""
    rules = rules_of(source)
    lights = {100 + i: light for i, light in enumerate(source['lights'])}
    by_path = {light['path']: light for light in lights.values()}
    if grouping is None:
        grouping = copy.deepcopy(source['grouping'])
        grouping['imaging_lights'] = [[id, light['rig_uuid'], light['timestamp']] for id, light in lights.items()]
        grouping['profile_rigs'] = [[rig, [rig]] for rig in sorted({l['rig_uuid'] for l in lights.values()})]
    nights = seiza.CalibrationNights(grouping.get('night_boundaries', []), frames=source['frames'], rules=rules)
    groups = groups_of(grouping, rules, nights)
    calibration = seiza.CalibrationLibrary(source['frames'], groups=groups)

    def lights_of(session, rig=None):
        return [dict(path=light['path'], at=light['timestamp'], angle=light.get('rotator_position'),
                     filter=light.get('filter') or '', target='M42')
                for light in (lights[id] for id in session['lights'] if id in lights)
                if rig is None or light['rig_uuid'] == rig]

    rigs = sorted({rig for session in groups.imaging for rig in session['rigs']})
    families = [question for rig in rigs for question in seiza.calibration_family_questions(
        calibration, rules, rig, lambda session, rig=rig: lights_of(session, rig), by_path.get)]
    needed = set(map(tuple, groups.sessions_needing_evidence()))
    changes = []
    for event, session, key in groups.sessions_in_windows():
        rig = groups.events[event]['rig_uuid']
        question = seiza.calibration_change_question(groups, event, session, lights_of(groups.imaging[session], rig))
        if question is not None:
            changes.append((question, (event, session) not in needed))
    return families, changes, grouping


def test_cached_metadata_questions_feedback_and_no_read_reuse():
    source = library()
    families, changes, grouping = flat_work(source)
    assert len(families) == 1
    question = families[0]
    ids = {light['path']: 100 + i for i, light in enumerate(source['lights'])}
    sample = [ids[light['path']] for light in question.sample()]
    assert len(sample) == 8 and sample[0] == 100 and sample[-1] == 111
    assert len(question.candidates) == 2
    assert question.representatives() is not None
    assert question.retention_key()['key'] == question.key
    # Unread/insufficient backgrounds choose nothing; no host tie policy needed.
    row = question.row(dict(winners=[], pairs=[]), 3)
    assert row['chosen'] is None
    grouping['family_evidence'] = list(grouping.get('family_evidence', [])) + [row]
    change, answered = changes[0]
    assert not answered
    evidence = change.row([])
    assert evidence['verdict'] == 'undecided'
    grouping['evidence'] = list(grouping.get('evidence', [])) + [evidence]
    families, changes, _ = flat_work(source, grouping)
    assert families[0].answered
    assert families[0].sample() == []
    assert changes[0][1]
    with pytest.raises(ValueError, match='exceed the question'):
        question.row(dict(winners=[99], pairs=[]), 3)


def test_family_questions_raise_what_the_host_callbacks_raise():
    source = library()
    _, _, grouping = flat_work(source)
    rules = rules_of(source)
    nights = seiza.CalibrationNights(frames=source['frames'], rules=rules)
    calibration = seiza.CalibrationLibrary(source['frames'], groups=groups_of(grouping, rules, nights))
    rig = calibration.groups.imaging[0]['rigs'][0]

    def lights_of(session):
        raise LookupError('the catalog is gone')

    with pytest.raises(LookupError, match='catalog'):
        seiza.calibration_family_questions(calibration, rules, rig, lights_of, lambda path: None)


def test_change_filters_fit_the_lights_against_the_reference_and_map():
    source = library()
    _, changes, _ = flat_work(source)
    change, _ = changes[0]
    if not change.filters:
        pytest.skip('the fixture change has no filter with enough lights')
    flat = change.filters[0]
    before, after = flat.map_responses()
    assert before['session_key'] != after['session_key']
    plane = np.ones((32, 32), dtype=np.float32)
    shifted = plane.copy()
    shifted[8:16, 8:16] = .9
    change_map = flat.change_map(plane, shifted)
    if change_map is not None:
        fit = flat.fit([plane] * 4, plane, change_map, pedestal=10.0)
        assert fit is None or fit['filter'] == flat.filter


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


def test_path_scoped_manual_choices_keep_identical_basenames_separate():
    source = library()
    source['lights'] = [dict(source['lights'][0], path='/first/shared.fits'),
                        dict(source['lights'][0], path='/second/shared.fits')]
    _, _, grouping = flat_work(source)
    rules = rules_of(source)
    nights = seiza.CalibrationNights(frames=source['frames'], rules=rules)
    event = seiza.FlatGroups(resolved(grouping, rules, nights)).events[0]
    source['grouping'] = grouping
    source['grouping']['light_choices'] = [dict(light_id=0, file_name='/first/shared.fits',
        filter='Ha', at=source['lights'][0]['timestamp'], row=dict(light_guid='first',
        change_key=event['key'], rig_uuid=event['rig_uuid'], change_start=event['start'],
        change_end=event['end'], choice='before', group_frame=None, group_session_key=None,
        label='before', chosen_at=1))]
    from calibration_host import plan
    selected = plan(source)['selections']
    assert selected[0].flat_choices
    assert not selected[1].flat_choices
