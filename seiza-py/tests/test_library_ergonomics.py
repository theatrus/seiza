"""Host-facing contracts: strict settings, retries on a session, cached pixels."""
import json
from pathlib import Path

import numpy as np
import pytest
import seiza

FIXTURES = Path(__file__).parents[2] / 'seiza-calibration/tests/fixtures/library'


@pytest.mark.parametrize('settings, message', [
    ({'external_masterz': 'ignore'}, "unknown setting 'external_masterz'"),
    ({'external_masters': 'nonsense'}, 'external_masters'),
    ({'use_darks': 'false'}, 'must be true or false'),
])
def test_settings_are_refused_the_same_for_the_server_and_a_rig(settings, message):
    with pytest.raises(ValueError, match=message):
        seiza.CalibrationRules(settings)
    with pytest.raises(ValueError, match=f'rig a: .*{message}'):
        seiza.CalibrationRules(rigs={'a': {'values': settings}})


def session_and_rules(frames, **selection):
    rules = seiza.CalibrationRules().for_light({})
    policy = selection.pop('flat_stability', None)
    chosen = seiza.CalibrationSelection(frames, rules, **selection)
    return seiza.calibration_resolve_session(chosen, seiza.CalibrationQuality(), flat_stability=policy), rules


def test_retry_takes_the_sessions_policy_and_held_back_flats():
    request = json.loads((FIXTURES / 'flat-retries.json').read_text())['cases'][0]['request']
    session, rules = session_and_rules(request['candidates'], flat_stability={'enabled': False},
                                       flat_held_back=request['held_back'])
    assert session.flat_held_back == request['held_back']
    retry = lambda: seiza.FlatRetry(request['first'], request['candidates'], rules,
                                    held_back=session.flat_held_back, policy=session.flat_stability)
    assert retry().next()['action'] == 'publish_first'
    # Missing flags use the documented true defaults, including enabled.
    session, rules = session_and_rules(request['candidates'], flat_stability={'trim_unstable': False},
                                       flat_held_back=request['held_back'])
    flat_retry = retry()
    decision = flat_retry.next()
    assert decision['action'] == 'try_alternate'
    assert decision['inputs'] == ['f4', 'f5']  # excludes f6 from the session
    flat_retry.alternate(dict(used=['f4', 'f5'], disagreement=[0, 0], drift_threshold=16))
    assert flat_retry.next()['action'] == 'publish_alternate'
    with pytest.raises(ValueError, match='original/held-back session'):
        flat_retry.alternate(dict(used=['f0', 'f1'], disagreement=[0, 0], drift_threshold=16))
    # A refused answer leaves the retry where it was.
    assert flat_retry.next()['action'] == 'publish_alternate'


@pytest.mark.parametrize('mode', ['mono', 'rgb', 'cfa'])
def test_cached_pixels_equal_file_measurement_with_calibration(tmp_path, mode):
    fits = pytest.importorskip('astropy.io.fits')
    master = np.full((128, 128, 3) if mode == 'rgb' else (128, 128), 1000., dtype=np.float32)
    master[56:72, 56:72] = 500.
    clean = np.full_like(master, 1000.)
    bias = np.full_like(master, 50.)
    dark = np.full_like(master, 70.)  # includes 50 bias and 20 dark current
    sources = [master + 70, clean + 70, clean + 70]
    metadata = [{'exposure_seconds': 2.}, {'exposure_seconds': 2.}, {'skipped': True}]
    bayer = {'pattern': 'RGGB', 'x_offset': 1, 'y_offset': 1} if mode == 'cfa' else None
    buffered = seiza.master_flat_stability_images(
        master, sources, bias=bias, dark=dark, dark_exposure_seconds=2.,
        input_metadata=metadata, bayer=bayer,
    )
    assert buffered['disagreement'][0] == buffered['disagreement'][2] == 0
    assert buffered['disagreement'][1] >= buffered['drift_threshold']
    assert buffered['feature_pixels'] >= 16

    def write(name, image, kind):
        header = fits.Header({'IMAGETYP': kind, 'EXPTIME': 2.})
        if bayer:
            header['BAYERPAT'] = 'RGGB'
            header['XBAYROFF'] = header['YBAYROFF'] = 1
        path = tmp_path / name
        fits.writeto(path, np.moveaxis(image, -1, 0) if mode == 'rgb' else image, header)
        return str(path)

    request = dict(master=write('master.fits', master, 'FLAT'),
                   bias=write('bias.fits', bias, 'BIAS'), dark=write('dark.fits', dark, 'DARK'),
                   dark_exposure_seconds=2., inputs=[])
    for i, (image, record) in enumerate(zip(sources, metadata)):
        request['inputs'].append(dict(path=write(f'{i}.fits', image, 'FLAT'), **record))
    assert seiza.master_flat_stability(**request) == buffered


def test_cached_pixel_validation_and_indexed_errors():
    master = np.ones((128, 128), dtype=np.float32)
    master[56:72, 56:72] = .5
    result = seiza.master_flat_stability_images(master, [np.ones((64, 64), dtype=np.float32)])
    assert result['errors'][0][0] == 0
    assert result['disagreement'] == [result['drift_threshold']]
    with pytest.raises(ValueError, match='aligned'):
        seiza.master_flat_stability_images(master, [master], input_metadata=[{}, {}])
    with pytest.raises(ValueError, match='C-contiguous'):
        seiza.master_flat_stability_images(master.T, [master])
    with pytest.raises(ValueError, match='Bayer'):
        seiza.master_flat_stability_images(master, [master], bayer={'pattern': 'INVALID'})
    with pytest.raises(ValueError, match='positive'):
        seiza.master_flat_stability_images(master, [master], input_metadata=[{'exposure_seconds': -1}])
    with pytest.raises(ValueError, match='requires a dark'):
        seiza.master_flat_stability_images(master, [master], dark_exposure_seconds=2.)


@pytest.mark.parametrize('scenario', ['trim', 'alternate', 'withhold', 'memo', 'cancel'])
def test_complete_host_loop_cleans_up_and_publishes_core_choice(scenario):
    import importlib.util
    import sys
    path = Path(__file__).parents[1] / 'examples/calibration_retry.py'
    spec = importlib.util.spec_from_file_location('seiza_calibration_retry_example', path)
    example = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = example
    spec.loader.exec_module(example)
    request = json.loads((FIXTURES / 'flat-retries.json').read_text())['cases'][0]['request']
    policy = {'retain_unstable': False, 'trim_unstable': False, 'try_next_session': False} if scenario == 'withhold' else {}
    for frame in request['candidates']:
        frame.update(camera='test', width=128, height=128, channels=1, exposure_s=2.)
    if scenario == 'memo':
        policy = {'enabled': False}
    session, rules = session_and_rules(request['candidates'], flat_held_back=request['held_back'],
                                       flat_stability=policy)
    built, discarded, published = [], [], []

    def build(ids, allow_memo):
        if built and scenario == 'cancel':
            raise InterruptedError('cancelled')
        if built and scenario == 'alternate' and list(ids) == ['f0', 'f1', 'f2']:
            return None
        feedback = dict(request['first']) if not built else dict(used=list(ids), disagreement=[0] * len(ids), drift_threshold=16)
        feedback['memo'] = scenario == 'memo' and allow_memo
        attempt = example.Attempt((list(ids), allow_memo), feedback)
        built.append(attempt)
        return attempt

    def publish(attempt, note):
        published.append(attempt)

    if scenario == 'cancel':
        with pytest.raises(InterruptedError):
            example.execute_flat(session, rules, request['candidates'], build, publish, discarded.append)
        assert discarded == built and not published
        return
    chosen = example.execute_flat(session, rules, request['candidates'], build, publish, discarded.append)
    if scenario == 'withhold':
        assert chosen is None and discarded == built and not published
    else:
        assert published == [chosen]
        assert discarded == [attempt for attempt in built if attempt is not chosen]
        expected_ids = {'trim': ['f0', 'f1', 'f2'], 'alternate': ['f4', 'f5'], 'memo': request['first']['used']}[scenario]
        assert chosen.feedback['used'] == expected_ids
        if scenario == 'memo':
            assert chosen.artifact[1] is False
