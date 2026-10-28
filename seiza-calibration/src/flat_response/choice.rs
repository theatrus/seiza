//! Which of several flat responses a light's own sky background follows.
//!
//! A rig's flats at several rotator angles, or from either side of a dust
//! change, are candidates for one light. Its sky background, the median of a
//! few of its lights ([`super::evidence::background`]), was shot through one
//! of those light paths and carries that path's dust. Every pair of
//! candidates is told apart by its compact difference, the pair's
//! [`super::metric::change_map`], and the background is fitted against that
//! difference with placebos ([`super::evidence::fit`]): a weight near 0 says
//! the background follows the first of the pair, near 1 the second, and the
//! placebos say whether the weight stands clear of what the sky's own
//! structure does at places where nothing differs. Each pair is fitted from
//! both of its candidates, and decides only when both fits agree, so the
//! order the candidates come in changes nothing.
//!
//! A candidate wins when it beats every candidate it can be told apart from
//! and loses to none. Two candidates whose responses hold nothing compact
//! apart cannot be told apart, and need not be: either corrects the light
//! the same way. A pair the background could not be weighed against at all
//! (no data over the shadows, planes too small) is not alike: nothing is
//! known of it, and neither candidate wins past it. Over a nebula the fits
//! usually stay undecided, and nothing wins; the host then falls back on
//! what it knew before, such as the recorded angle, and should say so.
//!
//! The pairwise fit, not a residual per candidate, is what decides. A
//! residual measures the sky's structure as much as the flat's, and over a
//! nebula the smallest one is the candidate whose dust happens to sit
//! under the faintest part of it. A pair's fit asks only about the pixels
//! where the two candidates differ, and its placebos measure the sky there.

use super::evidence::{self, BackgroundRatio, Fit, FitOutcome, Side, Verdict};
use super::metric::{self, Plane};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Bump when equal backgrounds and candidates may choose differently.
pub const CHOICE_ALGORITHM_VERSION: u32 = 1;

/// What the background said of one pair of candidates.
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum PairOutcome {
    /// The two hold nothing compact apart, or the host measured them
    /// alike: they cannot be told apart, and correct a light alike.
    Alike,
    /// The background could not be weighed against their difference.
    /// Nothing is known: neither may win past the other.
    Unfitted,
    /// The fit, `second` over `first`, weighed from `first`'s side: a
    /// weight of 0 follows `first`, 1 follows `second`. Its verdict is
    /// `Before` for `first` and `After` for `second` only when the fit
    /// from `second`'s side agrees; otherwise undecided.
    Fit(Fit),
}

/// How the background fared between two candidates.
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PairFit {
    /// The first candidate of the pair, by index.
    pub first: usize,
    /// The second candidate of the pair, by index; `first < second`.
    pub second: usize,
    /// What the background said of them.
    pub outcome: PairOutcome,
}

impl PairFit {
    /// The fit, when the background could be weighed against the pair.
    pub fn fit(&self) -> Option<Fit> {
        match self.outcome {
            PairOutcome::Fit(fit) => Some(fit),
            _ => None,
        }
    }

    /// The candidate the fit decided for, when it decided.
    pub fn favours(&self) -> Option<usize> {
        match self.fit()?.verdict {
            Verdict::Before => Some(self.first),
            Verdict::After => Some(self.second),
            Verdict::Undecided => None,
        }
    }

    /// Whether the two cannot be told apart: alike, not merely unfitted.
    pub fn alike(&self) -> bool {
        self.outcome == PairOutcome::Alike
    }
}

/// Which candidates a background follows.
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ResponseChoice {
    /// Candidates that beat every candidate they can be told apart from and
    /// lose to none, in index order. More than one only when they are alike
    /// with each other. Empty when nothing stands clear: the background
    /// could not tell, as over a nebula, a pair could not be fitted, or
    /// every candidate is alike. A lone candidate wins unopposed.
    pub winners: Vec<usize>,
    /// Every pair's outcome, in index order.
    pub pairs: Vec<PairFit>,
}

impl ResponseChoice {
    /// The one candidate the background follows, when exactly one wins.
    pub fn chosen(&self) -> Option<usize> {
        match self.winners.as_slice() {
            [only] => Some(*only),
            _ => None,
        }
    }

    /// Whether every pair was alike: nothing compact tells the candidates
    /// apart, so any of them corrects the light the same way. A pair that
    /// could not be fitted is not alike.
    pub fn all_alike(&self) -> bool {
        !self.pairs.is_empty() && self.pairs.iter().all(PairFit::alike)
    }

    /// Whether some pair could not be fitted at all.
    pub fn any_unfitted(&self) -> bool {
        self.pairs
            .iter()
            .any(|pair| pair.outcome == PairOutcome::Unfitted)
    }

    /// The pair of `a` and `b`, either way round.
    pub fn pair(&self, a: usize, b: usize) -> Option<&PairFit> {
        let (first, second) = (a.min(b), a.max(b));
        self.pairs
            .iter()
            .find(|pair| pair.first == first && pair.second == second)
    }
}

/// Fit a background against every pair of candidate responses and say
/// which candidates it follows.
///
/// `background` is a light background with its pedestal removed
/// ([`super::evidence::background`]); `candidates` are flat responses with
/// theirs removed ([`super::metric::FlatResponse::without_pedestal`]), all
/// on the background's grid. A candidate of another size is never chosen,
/// and its pairs are unfitted.
///
/// `alike(a, b)` says which pairs the host already measured alike, flat
/// against flat, by its own limits: they are not fitted, and count as
/// candidates that cannot be told apart. Two responses alike within noise
/// would otherwise still leave a few noise spots in their difference, and a
/// fit to those decides nothing.
pub fn choose_response(
    background: &Plane,
    candidates: &[Plane],
    alike: impl Fn(usize, usize) -> bool,
) -> ResponseChoice {
    let usable = |plane: &Plane| {
        plane.width == background.width
            && plane.height == background.height
            && plane.data.len() == plane.width * plane.height
    };
    // The background over each candidate, once: every pair it is in
    // weighs the same residual.
    let ratios: Vec<Option<BackgroundRatio>> = candidates
        .par_iter()
        .map(|candidate| {
            usable(candidate)
                .then(|| evidence::background_ratio(candidate, background))
                .flatten()
        })
        .collect();
    // Each pair is fitted on its own, so the pairs run side by side; `alike`
    // is asked first, here, so the host's closure need not be shared.
    let to_fit: Vec<(usize, usize, bool)> = (0..candidates.len())
        .flat_map(|first| (first + 1..candidates.len()).map(move |second| (first, second)))
        .map(|(first, second)| (first, second, alike(first, second)))
        .collect();
    let pairs: Vec<PairFit> = to_fit
        .into_par_iter()
        .map(|(first, second, alike)| {
            let outcome = if alike {
                PairOutcome::Alike
            } else {
                match (&ratios[first], &ratios[second]) {
                    (Some(from_first), Some(from_second)) => pair_outcome(
                        &candidates[first],
                        &candidates[second],
                        from_first,
                        from_second,
                    ),
                    _ => PairOutcome::Unfitted,
                }
            };
            PairFit {
                first,
                second,
                outcome,
            }
        })
        .collect();
    let lone = candidates.len() == 1 && usable(&candidates[0]);
    let winners = (0..candidates.len())
        .filter(|candidate| usable(&candidates[*candidate]))
        .filter(|candidate| {
            let mine: Vec<&PairFit> = pairs
                .iter()
                .filter(|pair| pair.first == *candidate || pair.second == *candidate)
                .collect();
            if mine.is_empty() {
                return lone;
            }
            let beats = mine
                .iter()
                .filter(|pair| pair.favours() == Some(*candidate))
                .count();
            let holds = mine
                .iter()
                .all(|pair| pair.alike() || pair.favours() == Some(*candidate));
            holds && beats > 0
        })
        .collect();
    ResponseChoice { winners, pairs }
}

/// The background against the difference between two candidates, fitted
/// from each side: `second` over `first` weighed from `first`, and its
/// inverse weighed from `second`. A side is taken only when both agree.
fn pair_outcome(
    first: &Plane,
    second: &Plane,
    from_first: &BackgroundRatio,
    from_second: &BackgroundRatio,
) -> PairOutcome {
    let Some(map) = metric::change_map(first, second) else {
        return PairOutcome::Unfitted;
    };
    // The same difference seen from `second`: first over second.
    let inverse = Plane {
        width: map.width,
        height: map.height,
        data: map
            .data
            .iter()
            .map(|value| {
                if *value > 0.0 && value.is_finite() {
                    value.recip()
                } else {
                    1.0
                }
            })
            .collect(),
    };
    let forward = evidence::fit_ratio(from_first, Side::Before, &map);
    let backward = evidence::fit_ratio(from_second, Side::Before, &inverse);
    match (forward, backward) {
        (FitOutcome::NoShadows, FitOutcome::NoShadows) => PairOutcome::Alike,
        (FitOutcome::Unfitted, _) | (_, FitOutcome::Unfitted) => PairOutcome::Unfitted,
        (FitOutcome::Fit(forward), FitOutcome::Fit(backward)) => {
            let verdict = match (forward.verdict, backward.verdict) {
                // From `first`: the lights lack the change, so follow
                // `first`; from `second`, they carry the inverse change.
                (Verdict::Before, Verdict::After) => Verdict::Before,
                (Verdict::After, Verdict::Before) => Verdict::After,
                _ => Verdict::Undecided,
            };
            PairOutcome::Fit(Fit { verdict, ..forward })
        }
        // One side sees shadows the other does not: nothing to agree on.
        (FitOutcome::Fit(fit), _) => PairOutcome::Fit(Fit {
            verdict: Verdict::Undecided,
            ..fit
        }),
        (_, FitOutcome::Fit(fit)) => PairOutcome::Fit(Fit {
            carried: 1.0 - fit.carried,
            verdict: Verdict::Undecided,
            ..fit
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat_response::metric::tests::{noise, synthetic};

    /// Lights shot through `flat`, with nebula-like structure of `nebula`.
    fn lights(flat: &Plane, nebula: f32, seed: u64) -> Vec<Plane> {
        (0..5)
            .map(|frame| Plane {
                width: flat.width,
                height: flat.height,
                data: flat
                    .data
                    .iter()
                    .enumerate()
                    .map(|(index, f)| {
                        let (x, y) = (index % flat.width, index / flat.width);
                        let structure = nebula * noise(x / 6, y / 6, seed + 99);
                        1000.0 * f * (1.0 + structure) * (1.0 + 0.002 * noise(x, y, seed + frame))
                    })
                    .collect(),
            })
            .collect()
    }

    /// One rig's flats at three angles: the shadows of the optics ahead of
    /// the rotator land elsewhere at each. `again` is the first angle on
    /// another night, the same dust with fresh noise.
    fn angles() -> (Plane, Plane, Plane, Plane) {
        let first = synthetic(240, 160, &[(60.0, 90.0, 14.0, 0.016)], 0.0, &[], 0.0003, 1);
        let second = synthetic(240, 160, &[(170.0, 40.0, 14.0, 0.016)], 0.0, &[], 0.0003, 2);
        let third = synthetic(240, 160, &[(120.0, 130.0, 12.0, 0.02)], 0.0, &[], 0.0003, 3);
        let again = synthetic(240, 160, &[(60.0, 90.0, 14.0, 0.016)], 0.0, &[], 0.0003, 4);
        (first, second, third, again)
    }

    #[test]
    fn a_smooth_sky_follows_the_flats_it_was_shot_through() {
        let (first, second, third, _) = angles();
        let candidates = [first.clone(), second.clone(), third.clone()];
        for (truth, expected) in [(&first, 0), (&second, 1), (&third, 2)] {
            let background =
                evidence::background(&lights(truth, 0.0, 7), 0.0, 10).expect("background");
            let choice = choose_response(&background, &candidates, |_, _| false);
            assert_eq!(choice.chosen(), Some(expected), "{choice:?}");
            assert_eq!(choice.pairs.len(), 3);
        }
    }

    #[test]
    fn candidates_alike_win_together() {
        let (first, second, _, again) = angles();
        let candidates = [first.clone(), second, again];
        let background = evidence::background(&lights(&first, 0.0, 11), 0.0, 10).unwrap();
        let choice = choose_response(&background, &candidates, |a, b| (a, b) == (0, 2));
        assert_eq!(choice.winners, vec![0, 2], "{choice:?}");
        assert!(choice.pair(0, 2).unwrap().alike());
        assert_eq!(choice.chosen(), None);
    }

    #[test]
    fn a_nebula_chooses_nothing_rather_than_the_wrong_flats() {
        let (first, second, third, _) = angles();
        let candidates = [first.clone(), second, third];
        for seed in 0..10u64 {
            let background =
                evidence::background(&lights(&first, 0.1, 100 * seed), 0.0, 10).unwrap();
            let choice = choose_response(&background, &candidates, |_, _| false);
            assert!(
                choice.winners.is_empty() || choice.winners == vec![0],
                "{seed}: {choice:?}"
            );
        }
    }

    #[test]
    fn a_lone_candidate_wins_unopposed_and_another_size_never_does() {
        let (first, _, _, _) = angles();
        let background = evidence::background(&lights(&first, 0.0, 3), 0.0, 10).unwrap();
        assert_eq!(
            choose_response(&background, std::slice::from_ref(&first), |_, _| false).winners,
            vec![0]
        );
        // A candidate of another size cannot be weighed: its pair is
        // unfitted, and neither wins past it.
        let small = synthetic(120, 80, &[], 0.0, &[], 0.0003, 5);
        let choice = choose_response(&background, &[small, first], |_, _| false);
        assert!(choice.winners.is_empty(), "{choice:?}");
        assert!(choice.any_unfitted());
        assert!(!choice.all_alike());
    }

    #[test]
    fn candidates_measured_alike_are_not_fitted_and_none_wins_alone() {
        let (first, _, _, again) = angles();
        let background = evidence::background(&lights(&first, 0.0, 5), 0.0, 10).unwrap();
        let choice = choose_response(&background, &[first.clone(), again.clone()], |_, _| true);
        assert!(choice.all_alike(), "{choice:?}");
        assert!(choice.winners.is_empty());
        // Not known alike: noise in their difference decides nothing, and
        // never against the flats the lights were shot through.
        let choice = choose_response(&background, &[first, again], |_, _| false);
        assert!(
            choice.winners.is_empty() || choice.winners == vec![0],
            "{choice:?}"
        );
    }
}
