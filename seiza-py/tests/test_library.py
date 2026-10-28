"""Reference outputs captured from PSF Guard, through the native Python API."""
import copy

import pytest
import seiza

from calibration_host import FIXTURES, KINDS, cases, groups_of, ids, plan, retry, rules_of, selection_record


@pytest.mark.parametrize('case', cases('recipes.json'), ids=lambda c: c['name'])
def test_resolved_recipes_determine_sessions_and_exposure(case):
    result = plan(case['request'])
    assert result['assignments'] == case['assignments']
    assert len(result['sessions']) == len(case['recipes'])
    for session, expected in zip(result['sessions'], case['recipes']):
        recipe = session.recipe(case['kind'])
        for key in ('action', 'inputs', 'exposure_seconds'):
            assert recipe[key] == expected[key]
        assert recipe['requires_source_verification'] is False


def test_recipe_exposure_survives_native_master_building(tmp_path):
    import numpy as np
    fits = pytest.importorskip("astropy.io.fits")

    request = copy.deepcopy(cases('recipes.json')[1]['request'])
    for frame in request['frames']:
        path = tmp_path / (frame['frame_uuid'] + '.fits')
        frame['source_path'] = str(path)
        # Only raw recipe inputs exist; an unused catalog master needs no I/O.
        if not frame['is_master']:
            fits.writeto(path, np.full((64, 64), 300.0, dtype=np.float32),
                         fits.Header({'IMAGETYP': 'DARK', 'EXPTIME': 300.0}))
    recipe = plan(request)['sessions'][0].recipe('dark')
    paths = {f['frame_uuid']: f['source_path'] for f in request['frames']}
    output = tmp_path / 'master.fits'
    seiza.build_dark([paths[id] for id in recipe['inputs']], output,
                     exposure_seconds=recipe['exposure_seconds'])
    assert fits.getheader(output)['EXPTIME'] == 300.0


@pytest.mark.parametrize('case', cases('psf-guard.json'), ids=lambda c: c['name'])
def test_psf_guard_selection_reference(case):
    result = plan(case['request'])
    selected = selection_record(result['selections'][0])
    for key in ('bias', 'dark', 'dark_flat', 'flat', 'kept_out', 'tolerances', 'flat_basis', 'flat_family'):
        assert selected[key] == case['expected'][key]
    assert result['assignments'] == [0, 0]
    sessions = [session.to_dict() for session in result['sessions']]
    reversed_request = copy.deepcopy(case['request'])
    reversed_request['frames'].reverse()
    assert [s.to_dict() for s in plan(reversed_request)['sessions']] == sessions
    # A host's cached book selects as the one built from the same rows.
    rules = result['rules']
    nights = seiza.CalibrationNights(case['request'].get('grouping', {}).get('night_boundaries', []),
                                     frames=case['request']['frames'], rules=rules)
    book = groups_of(case['request'].get('grouping', {}), rules, nights)
    cached = plan(case['request'], groups=book)
    assert [s.to_dict() for s in cached['sessions']] == sessions
    assert [selection_record(s) for s in cached['selections']] == [selection_record(s) for s in result['selections']]


def test_a_book_groups_each_rig_by_its_own_limits():
    request = copy.deepcopy(cases('psf-guard.json')[13]['request'])
    request['global_settings'] = [{'key': 'flat_split_dust_pct', 'value': 5.0}]
    rules = rules_of(request)
    groups = plan(request)['library'].groups
    for row in request['grouping']['responses']:
        assert groups.limits(row['rig_uuid']) == rules.for_rig(row['rig_uuid']).grouping_limits()
    assert rules.for_rig(request['grouping']['responses'][0]['rig_uuid']).grouping_limits()['dust'] == 0.05


@pytest.mark.parametrize('case', cases('grading.json'), ids=lambda c: c['name'])
def test_psf_guard_grades_and_overrides(case):
    result = plan(case['request'])
    for key in ('bias', 'dark', 'dark_flat', 'flat', 'kept_out'):
        assert sorted(selection_record(result['selections'][0])[key]) == case['expected'][key]
    for grade, expected in zip(result['grades'], case['grades']):
        assert grade['frame_uuid'] == expected['frame_uuid']
        assert grade['judged']['grade'] == expected['grade']
        assert grade['judged']['code'] == expected['code']
        assert grade['kept_out'] == expected['kept_out']
    request = copy.deepcopy(case['request'])
    request['regrade'] = False
    ungraded = plan(request)
    assert all(g['judged'] is None for g in ungraded['grades'])
    for key in ('bias', 'dark', 'dark_flat', 'flat', 'kept_out'):
        assert sorted(selection_record(ungraded['selections'][0])[key]) == case['ungraded'][key]
    if case['use_anyway']:
        request['regrade'] = True
        request['quality'][-1]['override_choice'] = 'use'
        result = plan(request)
        for key in ('bias', 'dark', 'dark_flat', 'flat', 'kept_out'):
            assert sorted(selection_record(result['selections'][0])[key]) == case['use_anyway'][key]


def test_stale_failed_and_missing_measurements_stay_distinct():
    request = copy.deepcopy(cases('grading.json')[0]['request'])
    request['quality'][0]['source_fingerprint'] = 'changed'
    request['quality'][1]['measurement'] = None
    request['quality'][1]['measure_error'] = 'cannot read'
    request['quality'] = request['quality'][:2]
    result = plan(request)
    assert [g['measurement_state'] for g in result['grades'][:3]] == ['stale', 'failed', 'unmeasured']
    assert all(g['judged'] is None for g in result['grades'][:3])


def test_records_refuse_duplicate_identities_and_unknown_fields():
    with pytest.raises(ValueError, match='unique'):
        seiza.CalibrationQuality([{'frame_uuid': 'a'}, {'frame_uuid': 'a'}])
    with pytest.raises(ValueError):
        seiza.CalibrationLibrary([{'frame_uuid': 'a', 'unknown': True}])
    library = seiza.CalibrationLibrary([])
    with pytest.raises(ValueError):
        library.select({'unknown': True}, seiza.CalibrationRules().global_rules())
    with pytest.raises(ValueError, match='duplicate setting'):
        rules_of({'global_settings': [{'key': 'use_darks', 'value': True}] * 2})


def flat_only_request():
    request = copy.deepcopy(cases('psf-guard.json')[9]['request'])
    for frame in request['frames']:
        frame['exposure_s'] = 2.0
        frame['camera_temp'] = -10.0
    request['lights'] = [copy.deepcopy(request['lights'][0]) for _ in range(4)]
    for light, path in zip(request['lights'], ('/d.fits', '/a.fits', '/c.fits', '/b.fits')):
        light['path'] = path
    return request


def test_flat_only_recipes_require_a_reliable_pedestal_and_reuse_a_pin():
    request = flat_only_request()
    session = plan(request)['sessions'][0]
    application = session.application
    assert application['flat_use'] == 'needs_pedestal_fit'
    assert application['sample_lights'] == ['/a.fits', '/c.fits', '/d.fits']
    session.accept_pedestal([298, 300, 302])
    assert (session.application['flat_use'], session.application['pedestal']) == ('fitted_pedestal', 300)
    key = session.key
    request['pedestal_evidence'] = [{'session_key': key, 'estimates': [100, 300, 700]}]
    assert plan(request)['sessions'][0].application['flat_use'] == 'withheld'
    request['pedestal_evidence'][0]['pinned'] = 123
    assert plan(request)['sessions'][0].application['pedestal'] == 123
    request['force_flat_without_pedestal'] = True
    assert plan(request)['sessions'][0].application['flat_use'] == 'forced_without_pedestal'
    with pytest.raises(ValueError, match='at most three'):
        plan(flat_only_request())['sessions'][0].accept_pedestal([1, 2, 3, 4])


def test_master_failures_skip_declared_dependents():
    request = flat_only_request()
    for index, kind in enumerate(('bias', 'bias', 'dark', 'dark', 'dark_flat', 'dark_flat')):
        frame = copy.deepcopy(request['frames'][0])
        frame.update(id=100 + index, frame_uuid=f'{kind}-{index}', kind=kind,
                     exposure_s=300.0 if kind == 'dark' else 2.0,
                     source_path=f'/{kind}-{index}.fits', source_fingerprint=f'{kind}-{index}')
        request['frames'].append(frame)
    session = plan(request)['sessions'][0]
    recipes = {r['kind']: r for r in session.recipes}
    assert recipes['flat']['dependencies'] == ['bias', 'dark_flat']
    assert recipes['dark']['dark_level_screening'] is True
    assert recipes['flat']['defect_suppression'] is True
    assert session.application['flat_use'] == 'with_masters'
    session.record_failure('bias', 'read error')
    session.plan_flat_use(request['lights'])
    assert [r['action'] for r in session.recipes] == ['failed', 'blocked', 'blocked', 'blocked']
    assert session.recipe('flat')['failure'] == 'skipped because the bias master failed to build'
    assert session.application['flat_use'] == 'unavailable'


def test_only_verified_candidates_receive_psf_guards_master_input_cap():
    request = copy.deepcopy(cases('psf-guard.json')[1]['request'])
    base = request['frames'][2]
    request['frames'] = []
    for index in range(100):
        frame = copy.deepcopy(base)
        frame.update(id=index, frame_uuid=f'b{index}', source_path=f'/b{index}.fits',
                     source_fingerprint=f'fp{index}', source_verified=True)
        request['frames'].append(frame)
    recipe = plan(request)['sessions'][0].recipes[0]
    assert len(recipe['inputs']) == 64
    assert recipe['input_limit'] == 64
    assert recipe['requires_source_verification'] is False
    for frame in request['frames']:
        frame['source_verified'] = False
    assert plan(request)['sessions'][0].recipes[0]['requires_source_verification'] is True


def test_legacy_dark_screening_uses_psf_guards_current_grade_or_override_rule():
    request = copy.deepcopy(cases('psf-guard.json')[0]['request'])
    request['quality'] = [
        {'frame_uuid': f['frame_uuid'], 'source_fingerprint': f['source_fingerprint'],
         'algorithm_version': 1, 'grade': 'ok'} for f in request['frames']
    ]
    dark = lambda: plan(request)['sessions'][0].recipes[2]
    assert dark()['dark_level_screening'] is False
    request['quality'][2]['source_fingerprint'] = 'stale'
    assert dark()['dark_level_screening'] is True
    request['quality'][2]['override_choice'] = 'use'
    assert dark()['dark_level_screening'] is False


@pytest.mark.parametrize('case', cases('flat-retries.json'), ids=lambda c: c['name'])
def test_flat_retry_follows_psf_guards_decisions(case):
    result = retry(case['request'])
    for field in ['action', 'inputs', 'rebuild_first']:
        assert result[field] == case[field], case['name']


def test_master_sessions_and_cohorts_of_chosen_frames():
    for case in cases('recipes.json'):
        source = case['request']
        rules = rules_of(source)
        light = source['lights'][0]
        selection = seiza.CalibrationSelection(source['frames'], rules.for_light(light))
        session = seiza.calibration_resolve_session(selection, seiza.CalibrationQuality())
        assert len(session.recipes) == 4
        cohort = seiza.calibration_coherent_cohort(case['kind'], source['frames'], rules.global_rules())
        assert set(cohort) == {'kept', 'dropped'}
        assert all(frame['kind'] == case['kind'] for frame in cohort['kept'])
    candidates = cases('flat-retries.json')[0]['request']['candidates']
    with pytest.raises(ValueError):
        seiza.FlatRetry({'used': ['missing']}, candidates, seiza.CalibrationRules().global_rules())


def test_a_selection_of_chosen_frames_resolves_as_the_library_did():
    for case in cases('psf-guard.json'):
        source = case['request']
        result = plan(source)
        rows = {f['frame_uuid']: f for f in source['frames']}
        for i, selection in enumerate(result['selections']):
            frames = [rows[key] for kind in ('bias', 'dark_flat', 'dark', 'flat') for key in ids(selection.of_kind(kind))]
            chosen = seiza.CalibrationSelection(
                frames, result['rules'].for_light(source['lights'][i]), flat_basis=selection.flat_basis,
                flat_family=selection.flat_family, flat_held_back=selection.flat_held_back,
            )
            session = seiza.calibration_resolve_session(chosen, seiza.CalibrationQuality())
            expected = result['sessions'][result['assignments'][i]]
            assert session.key == expected.key, case['name']
            assert session.recipes == expected.recipes, case['name']


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
    result = seiza.master_flat_stability(master=str(master_path), inputs=[
        {'path': master_path}, {'path': str(clean_path)}, {'path': str(tmp_path / 'missing.fits')}])
    assert result['algorithm_version'] == 2
    assert result['feature_pixels'] >= 16
    assert result['disagreement'][0] == 0
    assert result['disagreement'][1] >= result['drift_threshold']
    assert result['disagreement'][2] == result['drift_threshold']
    assert result['errors'][0][0] == 2


def test_supplied_sets_and_flat_sessions_share_core_rules():
    frames = [dict(frame_uuid=f"flat-{i}", rig_uuid="r", kind="flat",
                   source_path=f"/flat-{i}.fits", source_fingerprint=f"stamp-{i}",
                   source_verified=True, captured_at=1_700_000_000 + (i // 20) * 259200,
                   width=512, height=512, channels=1, gain=100, offset=20,
                   filter="Ha", rotation=0.0, exposure_s=30.0) for i in range(80)]
    rules = seiza.CalibrationRules()
    nights = seiza.CalibrationNights(frames=frames, rules=rules)
    sessions = [s for s in seiza.calibration_flat_sessions(frames, nights, rules) if s.measurable()]
    assert len(sessions) == 4
    assert all(len(s.combined()) == 16 for s in sessions)
    pedestals = seiza.PedestalTable(frames)
    rows = [s.pending_row(pedestals.for_session(s)) for s in sessions]
    assert all(row['bin'] is None and row['error'] is None for row in rows)
    assert len({row['response_key'] for row in rows}) == 4
    light = dict(path="/light.fits", rig_uuid="r", width=512, height=512, channels=1,
                 gain=100, offset=20, filter="Ha", rotator_position=0.0,
                 exposure_s=30.0, timestamp=1_700_000_000)
    session = plan(dict(frames=frames, lights=[light], flat_association="one_set"))['sessions'][0]
    flat = session.recipe('flat')
    assert len(flat['inputs']) == 80
    assert flat['input_limit'] == 80
    assert session.flat_stability['try_next_session'] is False
    chosen = seiza.CalibrationSelection(frames, rules.for_light(light), flat_association="one_set")
    direct = seiza.calibration_resolve_session(chosen, seiza.CalibrationQuality())
    assert direct.recipes == session.recipes
    frames[0]['gain'] = 200
    with pytest.raises(ValueError, match='compatible'):
        plan(dict(frames=frames, lights=[light], flat_association="one_set"))
    with pytest.raises(ValueError):
        plan(dict(flat_association="invalid"))


def test_measured_rows_carry_the_response_and_what_would_not_read():
    import numpy as np
    frames = [dict(frame_uuid=f"flat-{i}", rig_uuid="r", kind="flat", source_fingerprint=f"s{i}",
                   captured_at=1_700_000_000 + i, width=64, height=64, filter="Ha", rotation=0.0)
              for i in range(8)]
    rules = seiza.CalibrationRules()
    plan_ = seiza.calibration_flat_sessions(frames, seiza.CalibrationNights(), rules)[0]
    planes = [np.full((16, 16), 1000.0 + i, dtype=np.float32) for i in range(8)]
    response = seiza.flat_build_response(planes, 4)
    row = plan_.measured_row(20.0, 8, 4, response)
    assert row['frames'] == 8 and row['bin'] == 4 and row['error'] is None
    assert row['pedestal_adu'] == 20.0 and row['width'] == 16
    unread = [{'frame_uuid': 'flat-0', 'kind': 'missing', 'reason': 'not on disk'}]
    failed = plan_.measured_row(None, 1, 4, None, unread)
    assert failed['error'].startswith('1 of 8 flats could not be read')


def test_settings_follow_the_shared_table():
    schema = seiza.calibration_settings_schema()
    keys = [field["key"] for field in schema]
    assert "rotation_tolerance_deg" in keys and len(keys) == len(set(keys))
    ok = seiza.calibration_setting_check("rotation_tolerance_deg", 3.0)
    assert ok == {
        "key": "rotation_tolerance_deg",
        "valid": True,
        "error": None,
        "text": "3°",
        "default": False,
    }
    wide = seiza.calibration_setting_check("rotation_tolerance_deg", 1000.0)
    assert not wide["valid"] and "Flat rotation" in wide["error"]
    unknown = seiza.calibration_setting_check("from_the_future", True)
    assert not unknown["valid"] and "unknown setting" in unknown["error"]
    rules = seiza.CalibrationRules({'dark_reach_days': 90}, {'rig-a': {'name': 'SpaceCat', 'values': {'dark_reach_days': 400}}})
    rig = rules.for_rig('rig-a')
    assert rig.value('dark_reach_days') == 400
    assert (rig.source('dark_reach_days'), rig.set_by('dark_reach_days')) == ('rig', 'rig')
    assert rules.for_rig('other').source('dark_reach_days') == 'global'
    assert rig.set_by('complete_dark_frames') is None
    assert rig.mode('check_no_flat') == 'flag'
    assert rig.rig == {'rig_uuid': 'rig-a', 'name': 'SpaceCat'}
    assert rules.rig_sets('rig-a', 'dark_reach_days') and not rules.rig_sets('rig-a', 'use_darks')
    assert [note['key'] for note in rig.notes()] == ['dark_reach_days']
    assert rules.digest is not None and seiza.CalibrationRules().digest is None
    with pytest.raises(ValueError, match='is not a setting'):
        rig.value('from_the_future')
    with pytest.raises(ValueError, match='not a light check'):
        rig.mode('use_darks')
