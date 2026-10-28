"""The decisions a host that builds its own masters calls, and the checks a
native host is held to: the same as the C ABI's requests."""
import pytest
import seiza


def frame(i, kind='flat', at=0, **fields):
    return dict(dict(
        id=i, frame_uuid=f'{kind}-{i}', rig_uuid='rig', kind=kind,
        source_path=f'/{kind}-{i}.fits', source_fingerprint=f'stamp-{i}',
        source_verified=True, captured_at=1_700_000_000 + at, camera='Test Camera',
        telescope='Test Scope', width=512, height=512, channels=1, binning_x=1,
        binning_y=1, gain=100, offset=20, exposure_s=30.0, camera_temp=-10.0,
        filter='Ha', rotation=0.0,
    ), **fields)


def test_pedestal_decisions():
    assert seiza.calibration_pedestal_from_fits([100.0, 101.0, 102.0]) == 101.0
    with pytest.raises(ValueError, match='disagree'):
        seiza.calibration_pedestal_from_fits([100.0, 900.0])
    with pytest.raises(ValueError, match='usable fit'):
        seiza.calibration_pedestal_from_fits([])
    assert seiza.calibration_pedestal_hint({'camera': 'ZWO ASI2600MM Pro', 'offset': 50}) == 500.0
    assert seiza.calibration_pedestal_hint({'camera': 'Other', 'offset': 50}) is None
    assert seiza.calibration_pedestal_sample_lights(['/d', '/a', '/c', '/b']) == ['/a', '/c', '/d']
    assert not seiza.calibration_pedestal_moved(1.0, 1.2)
    assert seiza.calibration_pedestal_moved(1.0, 2.0)
    assert seiza.calibration_pedestal_moved(None, 1.0)


def test_master_decisions():
    rules = seiza.CalibrationRules().global_rules()
    external = frame(9, is_master=True)
    raw = [frame(0), frame(1, at=60)]
    adopted = seiza.calibration_adopted_master('flat', [external] + raw, rules)
    assert adopted['frame_uuid'] == 'flat-9'
    assert seiza.calibration_adopted_master('flat', [external] + raw, rules, held_back=['flat-9']) is None
    darks = [frame(0, 'dark'), frame(1, 'dark', at=60)]
    assert seiza.calibration_dark_level_screening('dark', darks, seiza.CalibrationQuality())
    assert not seiza.calibration_dark_level_screening('flat', raw, seiza.CalibrationQuality())
    assert seiza.calibration_flat_set_label(raw).startswith('2 Ha flats from ')
    key = seiza.calibration_family_key(raw[0])
    assert len(key) == 32 and key == seiza.calibration_family_key(raw[1])
    finding = dict(code='level_excess', grade='poor', value=2.0, limit=1.0)
    assert seiza.calibration_describe_finding(finding, 'dark')


def test_a_native_host_is_refused_what_the_c_abi_is_refused():
    with pytest.raises(ValueError, match='nonempty and unique'):
        seiza.CalibrationLibrary([frame(0), frame(0)])
    with pytest.raises(ValueError, match='finite'):
        seiza.CalibrationLibrary([frame(0, exposure_s=float('nan'))])
    with pytest.raises(ValueError, match='night shifts'):
        seiza.CalibrationNights([dict(rig_uuid='rig', frame_uuid='f', seconds=43200)] * 2)
    library = seiza.CalibrationLibrary([frame(0)], rules=seiza.CalibrationRules())
    library.check_limits(seiza.CalibrationRules())
    rules = seiza.CalibrationRules().global_rules()
    with pytest.raises(ValueError, match='positive'):
        library.select(dict(path='/l.fits', rig_uuid='rig', width=0), rules)


def test_a_retry_takes_its_policy_from_the_session_or_from_the_host_not_both():
    flats = [frame(0), frame(1, at=60)]
    rules = seiza.CalibrationRules().global_rules()
    chosen = seiza.CalibrationSelection(flats, rules)
    session = seiza.calibration_resolve_session(chosen, seiza.CalibrationQuality())
    first = dict(used=['flat-0', 'flat-1'])
    assert seiza.FlatRetry(first, flats, rules, session=session).next()['action'] == 'publish_first'
    with pytest.raises(ValueError, match='not both'):
        seiza.FlatRetry(first, flats, rules, session=session, held_back=['flat-0'])
