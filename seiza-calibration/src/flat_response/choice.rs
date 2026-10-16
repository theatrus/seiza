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
//! structure does at places where nothing differs.
//!
//! A candidate wins when it beats every candidate it can be told apart from
//! and loses to none. Two candidates whose responses hold nothing compact
//! apart cannot be told apart, and need not be: either corrects the light
//! the same way. Over a nebula the fits usually stay undecided, and nothing
//! wins; the host then falls back on what it knew before, such as the
//! recorded angle, and should say so.
//!
//! The pairwise fit, not a residual per candidate, is what decides. A
//! residual measures the sky's structure as much as the flat's, and over a
//! nebula the smallest one is the candidate whose dust happens to sit
//! under the faintest part of it. A pair's fit asks only about the pixels
//! where the two candidates differ, and its placebos measure the sky there.

use super::evidence::{self, Fit, Side, Verdict};
use super::metric::{self, Plane};
use serde::{Deserialize, Serialize};

/// Bump when equal backgrounds and candidates may choose differently.
pub const CHOICE_ALGORITHM_VERSION: u32 = 1;

/// How the background fared between two candidates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PairFit {
    /// The two candidates, by index, `first < second`.
    pub first: usize,
    pub second: usize,
    /// The background fitted against the pair's difference, `second` over
    /// `first`: a weight of 0 follows `first`, 1 follows `second`. `None`
    /// when the two hold nothing compact apart, so they cannot be told
    /// apart, and correct a light alike.
    pub fit: Option<Fit>,
}

impl PairFit {
    /// The candidate the fit decided for, when it decided.
    pub fn favours(&self) -> Option<usize> {
        match self.fit?.verdict {
            Verdict::Before => Some(self.first),
            Verdict::After => Some(self.second),
            Verdict::Undecided => None,
        }
    }

    /// Whether the pair could not be told apart at all.
    pub fn alike(&self) -> bool {
        self.fit.is_none()
    }
}

/// Which candidates a background follows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ResponseChoice {
    /// Candidates that beat every candidate they can be told apart from and
    /// lose to none, in index order. More than one only when they cannot be
    /// told apart from each other. Empty when nothing stands clear: the
    /// background could not tell, as over a nebula, or every candidate is
    /// alike. A lone candidate wins unopposed.
    pub winners: Vec<usize>,
    /// Every pair's fit, in index order.
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
    /// apart, so any of them corrects the light the same way.
    pub fn all_alike(&self) -> bool {
        !self.pairs.is_empty() && self.pairs.iter().all(PairFit::alike)
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
/// on the background's grid. A candidate of another size is never chosen.
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
    let usable =
        |plane: &Plane| plane.width == background.width && plane.height == background.height;
    let mut pairs = Vec::new();
    for first in 0..candidates.len() {
        for second in first + 1..candidates.len() {
            if !usable(&candidates[first]) || !usable(&candidates[second]) {
                continue;
            }
            let fit = if alike(first, second) {
                None
            } else {
                pair_fit(background, &candidates[first], &candidates[second])
            };
            pairs.push(PairFit { first, second, fit });
        }
    }
    let winners = (0..candidates.len())
        .filter(|candidate| usable(&candidates[*candidate]))
        .filter(|candidate| {
            let mine: Vec<&PairFit> = pairs
                .iter()
                .filter(|pair| pair.first == *candidate || pair.second == *candidate)
                .collect();
            if mine.is_empty() {
                // Unopposed: the only usable candidate.
                return candidates.iter().filter(|plane| usable(plane)).count() == 1;
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

/// The background against the difference between two candidates, `second`
/// over `first`, weighed from `first`'s side.
fn pair_fit(background: &Plane, first: &Plane, second: &Plane) -> Option<Fit> {
    let map = metric::change_map(first, second)?;
    evidence::fit(background, first, Side::Before, &map)
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
        let small = synthetic(120, 80, &[], 0.0, &[], 0.0003, 5);
        let choice = choose_response(&background, &[small, first], |_, _| false);
        assert_eq!(choice.winners, vec![1], "{choice:?}");
        assert!(choice.pairs.is_empty());
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
