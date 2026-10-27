"""Host-facing contracts: strict controls, session forwarding and cached pixels."""
import json
from pathlib import Path

import numpy as np
import pytest
import seiza

FIXTURES = Path(__file__).parents[2] / 'seiza-calibration/tests/fixtures/library'


@pytest.mark.parametrize('settings, message', [
    ([{'key': 'external_masterz', 'value': 'ignore'}], "unknown setting 'external_masterz'"),
    ([{'key': 'external_masters', 'value': 'nonsense'}], 'external_masters'),
    ([{'key': 'use_darks', 'value': 'false'}], 'must be true or false'),
    ([{'key': 'use_darks', 'value': True}] * 2, "duplicate setting key 'use_darks'"),
])
def test_settings_fail_consistently_across_endpoints(settings, message):
    for call in [
        lambda: seiza.calibration_library_plan(global_settings=settings),
        lambda: seiza.calibration_master_plan(lights=[{}], global_settings=settings),
        lambda: seiza.calibration_master_cohort(settings=settings),
        lambda: seiza.calibration_flat_retry(settings=settings),
        lambda: seiza.calibration_library_plan(rigs=[{'rig_uuid': 'a', 'settings': settings}]),
    ]:
        with pytest.raises(ValueError, match=message):
            call()


def test_keyword_and_mapping_calls_agree_and_refuse_ambiguous_overrides():
    source = json.loads((FIXTURES / 'recipes.json').read_text())['cases'][0]['request']
    request = {k: source[k] for k in ['frames', 'lights']}
    assert seiza.calibration_master_plan(request) == seiza.calibration_master_plan(**request)
    cohort = dict(frames=request['frames'], kind='bias')
    assert seiza.calibration_master_cohort(cohort) == seiza.calibration_master_cohort(**cohort)
    with pytest.raises(ValueError, match='either a request mapping or keyword fields'):
        seiza.calibration_master_plan(request, flat_star_masking=True)


def test_retry_uses_returned_session_and_successful_feedback_needs_no_extra_flag():
    request = json.loads((FIXTURES / 'flat-retries.json').read_text())['cases'][0]['request']
    session = seiza.calibration_master_plan(
        frames=request['candidates'], lights=[{}], flat_stability={'enabled': False},
        flat_held_back=request['held_back'],
    )
    retry = dict(session=session, candidates=request['candidates'], first=request['first'])
    assert seiza.calibration_flat_retry(**retry)['action'] == 'publish_first'
    with pytest.raises(ValueError, match='conflict'):
        seiza.calibration_flat_retry(**retry, policy={'enabled': True})
    session['flat_stability'] = {'trim_unstable': False}
    # Missing flags use the documented true defaults, including enabled.
    result = seiza.calibration_flat_retry(**retry)
    assert result['action'] == 'try_alternate'
    assert result['inputs'] == ['f4', 'f5']  # excludes f6 from the session
    retry['alternate'] = dict(used=['f4', 'f5'], disagreement=[0, 0], drift_threshold=16)
    assert seiza.calibration_flat_retry(**retry)['action'] == 'publish_alternate'
    assert seiza.calibration_flat_retry(retry) == seiza.calibration_flat_retry(**retry)


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
    assert seiza.master_flat_stability(request) == buffered


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
    session = seiza.calibration_master_plan(
        frames=request['candidates'], lights=[{}],
        flat_held_back=request['held_back'], flat_stability=policy,
    )
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
            example.execute_flat(session, request['candidates'], build, publish, discarded.append)
        assert discarded == built and not published
        return
    chosen = example.execute_flat(session, request['candidates'], build, publish, discarded.append)
    if scenario == 'withhold':
        assert chosen is None and discarded == built and not published
    else:
        assert published == [chosen]
        assert discarded == [attempt for attempt in built if attempt is not chosen]
        expected_ids = {'trim': ['f0', 'f1', 'f2'], 'alternate': ['f4', 'f5'], 'memo': request['first']['used']}[scenario]
        assert chosen.feedback['used'] == expected_ids
        if scenario == 'memo':
            assert chosen.artifact[1] is False
