"""A Python host of the native calibration library, for the tests.

The shared fixtures under seiza-calibration/tests/fixtures/library are the
records a host keeps: frames, lights, quality rows, setting rows, flat
grouping rows and execution feedback. A host turns them into the native
objects and calls them as a Rust host does; this is that host, written once
so every test drives the same steps.
"""
import copy
import json
from pathlib import Path

import seiza

FIXTURES = Path(__file__).resolve().parents[2] / 'seiza-calibration/tests/fixtures/library'
KINDS = ('bias', 'dark', 'dark_flat', 'flat')
BUILD_ORDER = ('bias', 'dark_flat', 'dark', 'flat')


def cases(name):
    return json.loads((FIXTURES / name).read_text())['cases']


def layer(rows):
    """A stored layer from setting rows, each key once."""
    values = {}
    for row in rows:
        if row['key'] in values:
            raise ValueError(f"duplicate setting key '{row['key']}'")
        values[row['key']] = row['value']
    return values


def rules_of(request):
    rigs = {rig['rig_uuid']: {'name': rig.get('name'), 'values': layer(rig.get('settings', []))}
            for rig in request.get('rigs', [])}
    return seiza.CalibrationRules(layer(request.get('global_settings', [])), rigs)


def ids(frames):
    return [frame['frame_uuid'] for frame in frames]


def resolved(grouping, rules, nights):
    """Stored grouping rows with each rig's limits from its settings and
    every night split as grading splits it."""
    grouping = copy.deepcopy(grouping)
    rigs = sorted({row['rig_uuid'] for row in grouping.get('responses', [])})
    grouping['limits'] = [{'rig_uuid': rig, 'limits': rules.for_rig(rig).grouping_limits()} for rig in rigs]
    grouping['night_boundaries'] = nights.boundaries()
    return grouping


def groups_of(grouping, rules, nights):
    """The grouping book of stored rows ([`resolved`])."""
    return seiza.FlatGroups(resolved(grouping, rules, nights))


def plan(request, groups=None):
    """Grade, select for every light, resolve and dedupe sessions, and
    apply the request's execution feedback, as a host does."""
    rules = rules_of(request)
    grouping = request.get('grouping', {})
    nights = seiza.CalibrationNights(
        list(grouping.get('night_boundaries', [])) + list(request.get('night_shifts', [])),
        frames=request.get('frames', []), rules=rules,
    )
    quality = seiza.CalibrationQuality(request.get('quality', []))
    judged = quality.regrade(request.get('frames', []), rules, nights) if request.get('regrade') else None
    frames = quality.keep_out(request.get('frames', []), judged)
    if groups is None:
        groups = groups_of(grouping, rules, nights)
    library = seiza.CalibrationLibrary(frames, groups=groups,
                                       available_responses=request.get('available_responses', []))
    associations = dict(flat_association=request.get('flat_association', 'automatic'),
                        dark_association=request.get('dark_association', 'automatic'))
    library.check_associations(rules, **associations)
    lights = request.get('lights', [])
    selections, sessions, assignments, keys = [], [], [], {}
    for light in lights:
        selection = library.select(light, rules.for_light(light), **associations)
        session = seiza.calibration_resolve_session(
            selection, quality, flat_star_masking=request.get('flat_star_masking', False),
            flat_stability=request.get('flat_stability'),
        )
        # The first light whose masters share a key stands for every light they serve.
        if session.key not in keys:
            keys[session.key] = len(sessions)
            sessions.append(session)
        assignments.append(keys[session.key])
        selections.append(selection)
    without = 'force' if request.get('force_flat_without_pedestal') else 'fit_pedestal'
    for index, session in enumerate(sessions):
        for kind in BUILD_ORDER:
            for failure in request.get('master_failures', []):
                if failure['session_key'] == session.key and failure['kind'] == kind:
                    session.record_failure(kind, failure['reason'])
        own = [light for light, assigned in zip(lights, assignments) if assigned == index]
        session.plan_flat_use(own, without_masters=without)
        for evidence in request.get('pedestal_evidence', []):
            if evidence['session_key'] == session.key:
                session.accept_pedestal(evidence.get('estimates', []), evidence.get('pinned'))
    grades = [dict(frame_uuid=frame['frame_uuid'],
                   judged=judged.get(frame['frame_uuid']) if judged is not None else None,
                   kept_out=frame.get('kept_out'),
                   measurement_state=quality.measurement_state(frame)) for frame in frames]
    return dict(rules=rules, quality=quality, library=library, selections=selections,
                sessions=sessions, assignments=assignments, grades=grades)


def selection_record(selection):
    """What the tests compare of a selection: frame UUIDs and its notes."""
    record = {kind: ids(selection.of_kind(kind)) for kind in KINDS}
    record.update(kept_out=ids(selection.kept_out), tolerances=selection.match_tolerances(),
                  flat_basis=selection.flat_basis, flat_family=selection.flat_family,
                  flat_held_back=selection.flat_held_back, flat_choices=selection.flat_choices)
    return record


def retry(request):
    """A flat retry fixture's next step, through FlatRetry."""
    rules = seiza.CalibrationRules(layer(request.get('settings', []))).global_rules()
    flat_retry = seiza.FlatRetry(request['first'], request['candidates'], rules,
                                 held_back=request.get('held_back', []), policy=request.get('policy'))
    if request.get('trimmed_attempted') or request.get('trimmed') is not None:
        flat_retry.trimmed(request.get('trimmed'))
    if request.get('alternate_attempted') or request.get('alternate') is not None:
        flat_retry.alternate(request.get('alternate'))
    return flat_retry.next()
