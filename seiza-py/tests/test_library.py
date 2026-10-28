"""Reference outputs captured from PSF Guard, through the public Python API."""
import copy
import json
from pathlib import Path

import pytest
import seiza

FIXTURES = Path(__file__).resolve().parents[2] / 'seiza-calibration/tests/fixtures/library'


def prepare_grouping(grouping):
    grouping = copy.deepcopy(grouping)
    grouping['limits'] = [
        {'rig_uuid': rig, 'limits': {'measured': True, 'dust': 0.01, 'broad': 0.02,
                                    'window': 86400, 'rotation': 2.0}}
        for rig in sorted({r['rig_uuid'] for r in grouping.get('responses', [])})
    ]
    return seiza.FlatGroups(grouping)


@pytest.mark.parametrize('case', json.loads((FIXTURES / 'recipes.json').read_text())['cases'], ids=lambda c: c['name'])
def test_resolved_recipes_determine_sessions_and_exposure(case):
    result = seiza.calibration_library_plan(**case['request'])
    assert result['assignments'] == case['assignments']
    assert len(result['sessions']) == len(case['recipes'])
    for session, expected in zip(result['sessions'], case['recipes']):
        recipe = next(r for r in session['recipes'] if r['kind'] == case['kind'])
        for key in ('action', 'inputs', 'exposure_seconds'):
            assert recipe[key] == expected[key]
        assert recipe['requires_source_verification'] is False


def test_recipe_exposure_survives_native_master_building(tmp_path):
    import numpy as np
    fits = pytest.importorskip("astropy.io.fits")

    case = json.loads((FIXTURES / 'recipes.json').read_text())['cases'][1]
    request = copy.deepcopy(case['request'])
    for frame in request['frames']:
        path = tmp_path / (frame['frame_uuid'] + '.fits')
        frame['source_path'] = str(path)
        # Only raw recipe inputs exist; an unused catalog master needs no I/O.
        if not frame['is_master']:
            fits.writeto(path, np.full((64, 64), 300.0, dtype=np.float32),
                         fits.Header({'IMAGETYP': 'DARK', 'EXPTIME': 300.0}))
    plan = seiza.calibration_library_plan(**request)
    recipe = next(r for r in plan['sessions'][0]['recipes'] if r['kind'] == 'dark')
    paths = {f['frame_uuid']: f['source_path'] for f in request['frames']}
    output = tmp_path / 'master.fits'
    seiza.build_dark([paths[id] for id in recipe['inputs']], output,
                     exposure_seconds=recipe['exposure_seconds'])
    assert fits.getheader(output)['EXPTIME'] == 300.0


@pytest.mark.parametrize('case', json.loads((FIXTURES / 'psf-guard.json').read_text())['cases'], ids=lambda c: c['name'])
def test_psf_guard_selection_reference(case):
    result = seiza.calibration_library_plan(**case['request'])
    for key in ('bias', 'dark', 'dark_flat', 'flat', 'kept_out', 'tolerances', 'flat_basis', 'flat_family'):
        assert result['selections'][0][key] == case['expected'][key]
    assert result['assignments'] == [0, 0]
    reversed_request = copy.deepcopy(case['request'])
    reversed_request['frames'].reverse()
    assert seiza.calibration_library_plan(**reversed_request)['sessions'] == result['sessions']
    cached_request = copy.deepcopy(case['request'])
    book = prepare_grouping(cached_request.pop('grouping', {}))
    assert seiza.calibration_library_plan(**cached_request, prepared_grouping=book) == result


def test_prepared_grouping_rejects_conflicting_rows_and_limits():
    request = copy.deepcopy(json.loads((FIXTURES / 'psf-guard.json').read_text())['cases'][13]['request'])
    book = prepare_grouping(request.pop('grouping'))
    with pytest.raises(ValueError, match='either'):
        seiza.calibration_library_plan(**request, grouping={'epoch_samples': [{'rig_uuid': 'r', 'at': 1, 'offset': 2}]}, prepared_grouping=book)
    request['global_settings'] = [{'key': 'flat_split_dust_pct', 'value': 5.0}]
    with pytest.raises(ValueError, match='limits'):
        seiza.calibration_library_plan(**request, prepared_grouping=book)


@pytest.mark.parametrize('case', json.loads((FIXTURES / 'grading.json').read_text())['cases'], ids=lambda c: c['name'])
def test_psf_guard_grades_and_overrides(case):
    result = seiza.calibration_library_plan(**case['request'])
    for key in ('bias', 'dark', 'dark_flat', 'flat', 'kept_out'):
        assert sorted(result['selections'][0][key]) == case['expected'][key]
    for grade, expected in zip(result['grades'], case['grades']):
        assert grade['frame_uuid'] == expected['frame_uuid']
        assert grade['judged']['grade'] == expected['grade']
        assert grade['judged']['code'] == expected['code']
        assert grade['kept_out'] == expected['kept_out']
    request = copy.deepcopy(case['request'])
    request['regrade'] = False
    ungraded = seiza.calibration_library_plan(**request)
    assert all(g['judged'] is None for g in ungraded['grades'])
    for key in ('bias', 'dark', 'dark_flat', 'flat', 'kept_out'):
        assert sorted(ungraded['selections'][0][key]) == case['ungraded'][key]
    if case['use_anyway']:
        request['regrade'] = True
        request['quality'][-1]['override_choice'] = 'use'
        result = seiza.calibration_library_plan(**request)
        for key in ('bias', 'dark', 'dark_flat', 'flat', 'kept_out'):
            assert sorted(result['selections'][0][key]) == case['use_anyway'][key]


def test_stale_failed_and_missing_measurements_stay_distinct():
    request = copy.deepcopy(json.loads((FIXTURES / 'grading.json').read_text())['cases'][0]['request'])
    request['quality'][0]['source_fingerprint'] = 'changed'
    request['quality'][1]['measurement'] = None
    request['quality'][1]['measure_error'] = 'cannot read'
    request['quality'] = request['quality'][:2]
    result = seiza.calibration_library_plan(**request)
    assert [g['measurement_state'] for g in result['grades'][:3]] == ['stale', 'failed', 'unmeasured']
    assert all(g['judged'] is None for g in result['grades'][:3])


def test_library_schema_refuses_duplicate_identities_and_unknown_fields():
    with pytest.raises(ValueError, match='unique'):
        seiza.calibration_library_plan(frames=[{'frame_uuid': 'a'}, {'frame_uuid': 'a'}])
    with pytest.raises(ValueError):
        seiza.calibration_library_plan(lights=[{'unknown': True}])
    with pytest.raises(ValueError, match='duplicate setting'):
        seiza.calibration_library_plan(global_settings=[{'key': 'use_darks', 'value': True}] * 2)


def flat_only_request():
    case = json.loads((FIXTURES / 'psf-guard.json').read_text())['cases'][9]
    request = copy.deepcopy(case['request'])
    for frame in request['frames']:
        frame['exposure_s'] = 2.0
        frame['camera_temp'] = -10.0
    request['lights'] = [copy.deepcopy(request['lights'][0]) for _ in range(4)]
    for light, path in zip(request['lights'], ('/d.fits', '/a.fits', '/c.fits', '/b.fits')):
        light['path'] = path
    return request


def test_flat_only_recipes_require_a_reliable_pedestal_and_reuse_a_pin():
    request = flat_only_request()
    result = seiza.calibration_library_plan(**request)
    session = result['sessions'][0]
    application = session['application']
    assert application['flat_use'] == 'needs_pedestal_fit'
    assert application['sample_lights'] == ['/a.fits', '/c.fits', '/d.fits']
    key = session['key']
    request['pedestal_evidence'] = [{'session_key': key, 'estimates': [298, 300, 302]}]
    application = seiza.calibration_library_plan(**request)['sessions'][0]['application']
    assert (application['flat_use'], application['pedestal']) == ('fitted_pedestal', 300)
    request['pedestal_evidence'][0]['estimates'] = [100, 300, 700]
    assert seiza.calibration_library_plan(**request)['sessions'][0]['application']['flat_use'] == 'withheld'
    request['pedestal_evidence'][0]['pinned'] = 123
    assert seiza.calibration_library_plan(**request)['sessions'][0]['application']['pedestal'] == 123
    request['force_flat_without_pedestal'] = True
    assert seiza.calibration_library_plan(**request)['sessions'][0]['application']['flat_use'] == 'forced_without_pedestal'


def test_master_failures_skip_declared_dependents():
    request = flat_only_request()
    for index, kind in enumerate(('bias', 'bias', 'dark', 'dark', 'dark_flat', 'dark_flat')):
        frame = copy.deepcopy(request['frames'][0])
        frame.update(id=100 + index, frame_uuid=f'{kind}-{index}', kind=kind,
                     exposure_s=300.0 if kind == 'dark' else 2.0,
                     source_path=f'/{kind}-{index}.fits', source_fingerprint=f'{kind}-{index}')
        request['frames'].append(frame)
    result = seiza.calibration_library_plan(**request)
    session = result['sessions'][0]
    recipes = {r['kind']: r for r in session['recipes']}
    assert recipes['flat']['dependencies'] == ['bias', 'dark_flat']
    assert recipes['dark']['dark_level_screening'] is True
    assert recipes['flat']['defect_suppression'] is True
    assert session['application']['flat_use'] == 'with_masters'
    request['master_failures'] = [{'session_key': session['key'], 'kind': 'bias', 'reason': 'read error'}]
    session = seiza.calibration_library_plan(**request)['sessions'][0]
    assert [r['action'] for r in session['recipes']] == ['failed', 'blocked', 'blocked', 'blocked']
    assert session['application']['flat_use'] == 'unavailable'


def test_only_verified_candidates_receive_psf_guards_master_input_cap():
    request = json.loads((FIXTURES / 'psf-guard.json').read_text())['cases'][1]['request']
    base = request['frames'][2]
    request['frames'] = []
    for index in range(100):
        frame = copy.deepcopy(base)
        frame.update(id=index, frame_uuid=f'b{index}', source_path=f'/b{index}.fits',
                     source_fingerprint=f'fp{index}', source_verified=True)
        request['frames'].append(frame)
    result = seiza.calibration_library_plan(**request)
    recipe = result['sessions'][0]['recipes'][0]
    assert len(recipe['inputs']) == 64
    assert recipe['input_limit'] == 64
    assert recipe['requires_source_verification'] is False
    for frame in request['frames']:
        frame['source_verified'] = False
    result = seiza.calibration_library_plan(**request)
    assert result['sessions'][0]['recipes'][0]['requires_source_verification'] is True


def test_legacy_dark_screening_uses_psf_guards_current_grade_or_override_rule():
    request = copy.deepcopy(json.loads((FIXTURES / 'psf-guard.json').read_text())['cases'][0]['request'])
    request['quality'] = [
        {'frame_uuid': f['frame_uuid'], 'source_fingerprint': f['source_fingerprint'],
         'algorithm_version': 1, 'grade': 'ok'} for f in request['frames']
    ]
    dark = lambda: seiza.calibration_library_plan(**request)['sessions'][0]['recipes'][2]
    assert dark()['dark_level_screening'] is False
    request['quality'][2]['source_fingerprint'] = 'stale'
    assert dark()['dark_level_screening'] is True
    request['quality'][2]['override_choice'] = 'use'
    assert dark()['dark_level_screening'] is False


def test_shared_master_resolution_and_retry_policy():
    import json
    from pathlib import Path
    base = Path(__file__).resolve().parents[2] / 'seiza-calibration/tests/fixtures/library'
    for case in json.loads((base / 'flat-retries.json').read_text())['cases']:
        result = seiza.calibration_flat_retry(case['request'])
        for field in ['action', 'inputs', 'rebuild_first']:
            assert result[field] == case[field], case['name']
    for case in json.loads((base / 'recipes.json').read_text())['cases']:
        source = case['request']
        request = dict(frames=source['frames'], lights=source['lights'],
                       global_settings=source.get('global_settings', []))
        plan = seiza.calibration_master_plan(request)
        assert len(plan['recipes']) == 4
        cohort = seiza.calibration_master_cohort(dict(kind=case['kind'], frames=source['frames']))
        assert set(cohort) == {'kept', 'dropped'}
    with pytest.raises(ValueError):
        seiza.calibration_flat_retry({'candidates': [], 'first': {'used': ['missing']}})


def test_selected_master_plan_reuses_library_decisions():
    import json
    from pathlib import Path
    fixture = Path(__file__).resolve().parents[2] / 'seiza-calibration/tests/fixtures/library/psf-guard.json'
    for case in json.loads(fixture.read_text())['cases']:
        source = case['request']
        result = seiza.calibration_library_plan(**source)
        rows = {f['frame_uuid']: f for f in source['frames']}
        for i, selection in enumerate(result['selections']):
            request = dict(frames=[rows[key] for kind in ['bias', 'dark_flat', 'dark', 'flat'] for key in selection[kind]],
                           lights=[source['lights'][i]], global_settings=source.get('global_settings', []), rigs=source.get('rigs', []),
                           flat_basis=selection['flat_basis'], flat_family=selection['flat_family'], flat_held_back=selection['flat_held_back'])
            plan = seiza.calibration_master_plan(request)
            expected = result['sessions'][result['assignments'][i]]
            assert plan['key'] == expected['key'], case['name']
            assert plan['recipes'] == expected['recipes'], case['name']


def test_master_flat_stability_measures_transient_features(tmp_path):
    import numpy as np
    # Standalone FITS writer keeps this policy oracle independent of optional Astropy.
    def write(path, data):
        cards = ["SIMPLE  =                    T", "BITPIX  =                  -32", "NAXIS   =                    2",
                 "NAXIS1  =                  128", "NAXIS2  =                  128", "END"]
        header = ''.join(card.ljust(80) for card in cards).encode('ascii')
        header += b' ' * (-len(header) % 2880)
        pixels = np.asarray(data, dtype='>f4').tobytes()
        path.write_bytes(header + pixels + b'\0' * (-len(pixels) % 2880))
    master = np.ones((128, 128), dtype=np.float32)
    master[56:72, 56:72] = 0.5
    master_path, clean_path = tmp_path / 'master.fits', tmp_path / 'clean.fits'
    write(master_path, master)
    write(clean_path, np.ones_like(master))
    result = seiza.master_flat_stability({'master': str(master_path), 'inputs': [
        {'path': str(master_path)}, {'path': str(clean_path)}, {'path': str(tmp_path / 'missing.fits')}]})
    assert result['algorithm_version'] == 2
    assert result['feature_pixels'] >= 16
    assert result['disagreement'][0] == 0
    assert result['disagreement'][1] >= result['drift_threshold']
    assert result['disagreement'][2] == result['drift_threshold']
    assert result['errors'][0][0] == 2


def test_supplied_sets_and_response_preparation_share_core_rules():
    frames = [dict(frame_uuid=f"flat-{i}", rig_uuid="r", kind="flat",
                   source_path=f"/flat-{i}.fits", source_fingerprint=f"stamp-{i}",
                   source_verified=True, captured_at=1_700_000_000 + (i // 20) * 259200,
                   width=512, height=512, channels=1, gain=100, offset=20,
                   filter="Ha", rotation=0.0, exposure_s=30.0) for i in range(80)]
    prepared = seiza.calibration_flat_response_plan(frames=frames)
    assert len(prepared['plans']) == 4
    assert all(len(p['sample_frames']) == 16 for p in prepared['plans'])
    assert prepared['unplanned'] == []
    assert len(prepared['grades']) == 80
    assert seiza.calibration_flat_response_plan(dict(frames=frames)) == prepared
    assert seiza.calibration_flat_response_plan(frames=frames, flat_association="one_set")['plans'] == []
    light = dict(path="/light.fits", rig_uuid="r", width=512, height=512, channels=1,
                 gain=100, offset=20, filter="Ha", rotator_position=0.0,
                 exposure_s=30.0, timestamp=1_700_000_000)
    result = seiza.calibration_library_plan(frames=frames, lights=[light], flat_association="one_set")
    session = result['sessions'][0]
    flat = next(r for r in session['recipes'] if r['kind'] == 'flat')
    assert len(flat['inputs']) == 80
    assert flat['input_limit'] == 80
    assert session['flat_stability']['try_next_session'] is False
    direct = seiza.calibration_master_plan(frames=frames, lights=[light], flat_association="one_set")
    assert direct['recipes'] == session['recipes']
    frames[0]['gain'] = 200
    with pytest.raises(ValueError, match='compatible'):
        seiza.calibration_library_plan(frames=frames, lights=[light], flat_association="one_set")
    with pytest.raises(ValueError):
        seiza.calibration_library_plan(flat_association="invalid")
