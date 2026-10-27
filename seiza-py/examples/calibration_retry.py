"""A complete flat execution loop for a host with its own builder and cache.

Build receives ordered candidate IDs and whether a cache memo is allowed.
Return None for a failed/unavailable retry. Exceptions abort the operation;
finally releases every unpublished attempt, including on cancellation.
"""
from dataclasses import dataclass
from typing import Callable, Sequence

import seiza


@dataclass
class Attempt:
    artifact: object
    feedback: seiza.FlatAttempt


def execute_flat(
    session: seiza.CalibrationSession,
    candidates: Sequence[seiza.CalibrationFrame],
    build: Callable[[Sequence[str], bool], Attempt | None],
    publish: Callable[[Attempt, str | None], None],
    discard: Callable[[Attempt], None],
) -> Attempt | None:
    """Execute an integrate recipe; adoption and dependencies precede this call.

    The host verifies sources, builds/measures pixels, and atomically publishes.
    All trimming, alternate-session and retention choices come from Seiza.
    A published master's provenance must record its final accepted inputs.
    """
    recipe = next(r for r in session['recipes'] if r['kind'] == 'flat')
    if recipe['action'] != 'integrate':
        raise ValueError('execute_flat needs an integrate recipe')
    first = build(recipe['inputs'], True)
    if first is None:
        return None
    attempts = [first]
    retained: Attempt | None = None
    by_stage = {'first': first}
    request: seiza.CalibrationFlatRetryRequest = dict(
        session=session, candidates=candidates, first=first.feedback,
    )
    try:
        while True:
            decision = seiza.calibration_flat_retry(request)
            action = decision['action']
            if action in ('try_trimmed', 'try_alternate'):
                stage = 'trimmed' if action == 'try_trimmed' else 'alternate'
                attempt = build(decision['inputs'], True)
                if attempt is None:
                    if stage == 'trimmed':
                        request['trimmed_attempted'] = True
                    else:
                        request['alternate_attempted'] = True
                else:
                    attempts.append(attempt)
                    by_stage[stage] = attempt
                    # Successful feedback implies attempted. A failed retry
                    # instead needs the explicit flag above.
                    if stage == 'trimmed':
                        request['trimmed'] = attempt.feedback
                    else:
                        request['alternate'] = attempt.feedback
                continue
            if action == 'withhold':
                return None
            stage = {'publish_first': 'first', 'publish_trimmed': 'trimmed',
                     'publish_alternate': 'alternate'}[action]
            chosen = by_stage[stage]
            if action == 'publish_first' and decision['rebuild_first']:
                rebuilt = build(first.feedback['used'], False)
                if rebuilt is None:
                    return None
                chosen = rebuilt
                attempts.append(chosen)
                if chosen.feedback.get('memo', False):
                    raise ValueError('forced rebuild returned a memo instead of master pixels')
            publish(chosen, decision['note'])
            retained = chosen
            return retained
    finally:
        for attempt in attempts:
            if attempt is not retained:
                discard(attempt)
