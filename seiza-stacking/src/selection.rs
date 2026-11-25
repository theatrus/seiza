//! Choosing which lights a stack keeps, and how much each counts.
//!
//! A night's lights are not equal. Cloud and haze dim the stars, a gust or
//! a guiding error stretches them, and focus or seeing spread them out. A
//! stack that takes every light equally lets the worst of them soften and
//! elongate its stars; one that drops too many gives up depth.
//!
//! [`select_frames`] works from the scores [`crate::choose_reference`]
//! already gives every light. A light is turned away only when it is an
//! outlier against the batch's own lights: too little starlight, too few
//! stars, stars too elongated or too spread out. Each kept light then gets a
//! weight factor, the reference's PSF area over its own, which
//! [`crate::LiveStacker::set_frame_weight_factors`] multiplies into its
//! inverse-noise weight. Noise variance times PSF area is the matched-filter
//! weight that gives a stack's point sources the highest signal to noise.

use crate::ReferenceScore;

/// When [`select_frames`] turns a light away. Every limit is relative to the
/// batch's scored lights: a median, and for shapes a robust spread (1.4826
/// times the median absolute deviation) above it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameSelectionOptions {
    /// Lights whose [`ReferenceScore::score`] is below this fraction of the
    /// median are turned away: cloud, haze, or stars so spread out that
    /// their starlight barely rises above the sky.
    pub minimum_relative_score: f32,
    /// Lights with fewer than this fraction of the median star count are
    /// turned away: cloud over part of the frame.
    pub minimum_relative_stars: f32,
    /// Lights whose [`ReferenceScore::eccentricity`] is more than this many robust
    /// spreads above the batch median are turned away: wind, a guiding
    /// error or trailing.
    pub eccentricity_sigma: f32,
    /// The smallest eccentricity margin above the median that turns a light
    /// away, however tightly the batch agrees: measurement scatter alone
    /// moves a light's median by about this much.
    pub minimum_eccentricity_margin: f32,
    /// An absolute eccentricity no kept light may exceed, if any.
    pub maximum_eccentricity: Option<f32>,
    /// Lights whose [`ReferenceScore::psf_area`] is more than this many
    /// robust spreads above the batch median are turned away: lost focus,
    /// a burst of poor seeing, a trail.
    pub psf_area_sigma: f32,
    /// The smallest PSF-area margin above the median, as a fraction of it,
    /// that turns a light away.
    pub minimum_psf_area_margin: f32,
    /// Smallest weight factor a kept light receives.
    pub minimum_weight_factor: f32,
    /// Largest weight factor a kept light receives.
    pub maximum_weight_factor: f32,
}

impl Default for FrameSelectionOptions {
    fn default() -> Self {
        Self {
            minimum_relative_score: 0.5,
            minimum_relative_stars: 0.5,
            eccentricity_sigma: 3.0,
            minimum_eccentricity_margin: 0.03,
            maximum_eccentricity: None,
            psf_area_sigma: 3.0,
            minimum_psf_area_margin: 0.1,
            minimum_weight_factor: 0.25,
            maximum_weight_factor: 4.0,
        }
    }
}

impl FrameSelectionOptions {
    /// Check that every fraction, spread and bound is usable.
    pub fn validate(&self) -> crate::Result<()> {
        let fraction = |value: f32| value.is_finite() && (0.0..=1.0).contains(&value);
        let positive = |value: f32| value.is_finite() && value > 0.0;
        let non_negative = |value: f32| value.is_finite() && value >= 0.0;
        if !(fraction(self.minimum_relative_score)
            && fraction(self.minimum_relative_stars)
            && positive(self.eccentricity_sigma)
            && non_negative(self.minimum_eccentricity_margin)
            && self
                .maximum_eccentricity
                .is_none_or(|limit| limit.is_finite() && (0.0..=1.0).contains(&limit))
            && positive(self.psf_area_sigma)
            && non_negative(self.minimum_psf_area_margin)
            && positive(self.minimum_weight_factor)
            && self.minimum_weight_factor <= 1.0
            && self.maximum_weight_factor.is_finite()
            && self.maximum_weight_factor >= 1.0)
        {
            return Err(crate::Error::Stack(
                "invalid frame selection options".into(),
            ));
        }
        Ok(())
    }
}

/// Why [`select_frames`] turned a light away.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FrameRejection {
    /// The light could not be read or scored.
    Unscored,
    /// Its starlight over the sky, [`ReferenceScore::score`], fell below
    /// `limit`.
    LowSignal { score: f32, limit: f32 },
    /// It held fewer than `limit` stars.
    FewStars { stars: usize, limit: f32 },
    /// Its stars' shared stretch, [`ReferenceScore::eccentricity`], was
    /// above `limit`.
    Elongated { eccentricity: f32, limit: f32 },
    /// Its stars' light was spread over more than `limit` pixels.
    Soft { psf_area: f32, limit: f32 },
}

impl FrameRejection {
    /// The reason as one word, as [`FrameRejectionKind`] spells it.
    pub fn kind(&self) -> FrameRejectionKind {
        match self {
            Self::Unscored => FrameRejectionKind::Unscored,
            Self::LowSignal { .. } => FrameRejectionKind::LowSignal,
            Self::FewStars { .. } => FrameRejectionKind::FewStars,
            Self::Elongated { .. } => FrameRejectionKind::Elongated,
            Self::Soft { .. } => FrameRejectionKind::Soft,
        }
    }

    /// The measured value that failed, if the light was scored.
    pub fn value(&self) -> Option<f32> {
        match *self {
            Self::Unscored => None,
            Self::LowSignal { score, .. } => Some(score),
            Self::FewStars { stars, .. } => Some(stars as f32),
            Self::Elongated { eccentricity, .. } => Some(eccentricity),
            Self::Soft { psf_area, .. } => Some(psf_area),
        }
    }

    /// The limit it failed, if the light was scored.
    pub fn limit(&self) -> Option<f32> {
        match *self {
            Self::Unscored => None,
            Self::LowSignal { limit, .. }
            | Self::FewStars { limit, .. }
            | Self::Elongated { limit, .. }
            | Self::Soft { limit, .. } => Some(limit),
        }
    }
}

impl std::fmt::Display for FrameRejection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Unscored => formatter.write_str("could not be read or scored"),
            Self::LowSignal { score, limit } => {
                write!(formatter, "star signal {score:.1} is below {limit:.1}")
            }
            Self::FewStars { stars, limit } => {
                write!(formatter, "{stars} stars, fewer than {limit:.0}")
            }
            Self::Elongated {
                eccentricity,
                limit,
            } => write!(
                formatter,
                "stars elongated: eccentricity {eccentricity:.3} is above {limit:.3}"
            ),
            Self::Soft { psf_area, limit } => write!(
                formatter,
                "stars spread out: PSF area {psf_area:.1}px is above {limit:.1}px"
            ),
        }
    }
}

/// [`FrameRejection`]'s reason without its numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FrameRejectionKind {
    Unscored,
    LowSignal,
    FewStars,
    Elongated,
    Soft,
}

impl FrameRejectionKind {
    const NAMES: [(Self, &'static str); 5] = [
        (Self::Unscored, "unscored"),
        (Self::LowSignal, "low-signal"),
        (Self::FewStars, "few-stars"),
        (Self::Elongated, "elongated"),
        (Self::Soft, "soft"),
    ];
}

impl std::fmt::Display for FrameRejectionKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = Self::NAMES
            .iter()
            .find(|(kind, _)| kind == self)
            .map(|(_, name)| *name)
            .expect("every kind is named");
        formatter.write_str(name)
    }
}

impl std::str::FromStr for FrameRejectionKind {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Self::NAMES
            .iter()
            .find(|(_, known)| *known == name)
            .map(|(kind, _)| *kind)
            .ok_or_else(|| format!("unknown frame rejection {name:?}"))
    }
}

/// Which lights a stack keeps, and how each kept light's stars compare.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameSelection {
    /// One verdict per light, in the order the scores were given: `Ok` with
    /// its PSF area for a kept light.
    pub verdicts: Vec<Result<f32, FrameRejection>>,
    /// The batch medians the limits are drawn from, over the scored lights.
    pub median_score: f32,
    pub median_stars: f32,
    pub median_psf_area: f32,
    pub median_eccentricity: Option<f32>,
    /// The limits a light had to stay within.
    pub score_limit: f32,
    pub stars_limit: f32,
    pub psf_area_limit: f32,
    pub eccentricity_limit: Option<f32>,
    /// The point-source signal to noise of a stack of the kept lights
    /// weighted by [`Self::weight_factor`] and their noise, over that of
    /// every scored light stacked with equal weight. Above 1 when the
    /// selection gains.
    pub expected_snr_gain: f64,
    weight_bounds: (f32, f32),
}

impl FrameSelection {
    /// The indices of the kept lights.
    pub fn kept(&self) -> impl Iterator<Item = usize> + '_ {
        self.verdicts
            .iter()
            .enumerate()
            .filter_map(|(index, verdict)| verdict.is_ok().then_some(index))
    }

    /// The best reference among the kept lights, as
    /// [`crate::rank_references`] ranks them, or `None` if none is kept.
    pub fn reference(&self, scores: &[Option<ReferenceScore>]) -> Option<usize> {
        let kept = scores
            .iter()
            .enumerate()
            .map(|(index, score)| {
                score
                    .clone()
                    .filter(|_| self.verdicts.get(index).is_some_and(Result::is_ok))
            })
            .collect::<Vec<_>>();
        crate::rank_references(&kept).first().copied()
    }

    /// Light `frame`'s weight factor against light `reference`: the
    /// reference's PSF area over its own, within the options' bounds, or
    /// `None` unless both are kept. The reference's factor is 1.
    pub fn weight_factor(&self, frame: usize, reference: usize) -> Option<f32> {
        let own = *self.verdicts.get(frame)?.as_ref().ok()?;
        let reference = *self.verdicts.get(reference)?.as_ref().ok()?;
        let (low, high) = self.weight_bounds;
        Some((reference / own).clamp(low, high))
    }
}

/// The median and robust spread of `values`.
fn centre_and_spread(values: &mut [f32]) -> Option<(f32, f32)> {
    let centre = seiza_stats::median_in_place(&mut values.to_vec())?;
    let mut deviations = values
        .iter()
        .map(|value| (value - centre).abs())
        .collect::<Vec<_>>();
    Some((
        centre,
        1.4826 * seiza_stats::median_in_place(&mut deviations)?,
    ))
}

/// Decide which lights to keep from their scores, which
/// [`crate::choose_reference`] gives in path order. A light is turned away
/// for the first limit it fails, in the order of [`FrameRejection`].
pub fn select_frames(
    scores: &[Option<ReferenceScore>],
    options: &FrameSelectionOptions,
) -> crate::Result<FrameSelection> {
    options.validate()?;
    let scored = scores.iter().flatten().collect::<Vec<_>>();
    if scored.is_empty() {
        return Err(crate::Error::Stack(
            "no light could be scored to select from".into(),
        ));
    }
    let (median_score, _) =
        centre_and_spread(&mut scored.iter().map(|s| s.score).collect::<Vec<_>>())
            .expect("scored lights");
    let (median_stars, _) =
        centre_and_spread(&mut scored.iter().map(|s| s.stars as f32).collect::<Vec<_>>())
            .expect("scored lights");
    let (median_psf_area, psf_area_spread) =
        centre_and_spread(&mut scored.iter().map(|s| s.psf_area).collect::<Vec<_>>())
            .expect("scored lights");
    let shaped = scored
        .iter()
        .filter_map(|s| s.eccentricity)
        .collect::<Vec<_>>();
    let eccentricity = (shaped.len() >= 3)
        .then(|| centre_and_spread(&mut shaped.clone()))
        .flatten();
    let score_limit = options.minimum_relative_score * median_score;
    let stars_limit = options.minimum_relative_stars * median_stars;
    let psf_area_limit = median_psf_area
        + (options.psf_area_sigma * psf_area_spread)
            .max(options.minimum_psf_area_margin * median_psf_area);
    let eccentricity_limit = eccentricity.map(|(centre, spread)| {
        let relative =
            centre + (options.eccentricity_sigma * spread).max(options.minimum_eccentricity_margin);
        options
            .maximum_eccentricity
            .map_or(relative, |limit| relative.min(limit))
    });
    let verdicts = scores
        .iter()
        .map(|score| {
            let Some(score) = score else {
                return Err(FrameRejection::Unscored);
            };
            if score.score < score_limit {
                return Err(FrameRejection::LowSignal {
                    score: score.score,
                    limit: score_limit,
                });
            }
            if (score.stars as f32) < stars_limit {
                return Err(FrameRejection::FewStars {
                    stars: score.stars,
                    limit: stars_limit,
                });
            }
            if let (Some(value), Some(limit)) = (score.eccentricity, eccentricity_limit)
                && value > limit
            {
                return Err(FrameRejection::Elongated {
                    eccentricity: value,
                    limit,
                });
            }
            if score.psf_area > psf_area_limit {
                return Err(FrameRejection::Soft {
                    psf_area: score.psf_area,
                    limit: psf_area_limit,
                });
            }
            Ok(score.psf_area)
        })
        .collect::<Vec<_>>();

    // Each light's point-source signal to noise is its score. Normalized to
    // one brightness, a stack weighted by score squared reaches
    // sqrt(sum s²); every light at equal weight reaches n / sqrt(sum 1/s²).
    let kept_snr = scores
        .iter()
        .zip(&verdicts)
        .filter(|(_, verdict)| verdict.is_ok())
        .filter_map(|(score, _)| score.as_ref())
        .map(|score| f64::from(score.score).powi(2))
        .sum::<f64>()
        .sqrt();
    let positive = scored
        .iter()
        .map(|s| f64::from(s.score))
        .filter(|s| *s > 0.0)
        .collect::<Vec<_>>();
    let equal_snr = positive.len() as f64 / positive.iter().map(|s| s.powi(-2)).sum::<f64>().sqrt();
    let expected_snr_gain = if equal_snr > 0.0 {
        kept_snr / equal_snr
    } else {
        1.0
    };
    Ok(FrameSelection {
        verdicts,
        median_score,
        median_stars,
        median_psf_area,
        median_eccentricity: eccentricity.map(|(centre, _)| centre),
        score_limit,
        stars_limit,
        psf_area_limit,
        eccentricity_limit,
        expected_snr_gain,
        weight_bounds: (options.minimum_weight_factor, options.maximum_weight_factor),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn score(score: f32, stars: usize, psf_area: f32, eccentricity: f32) -> Option<ReferenceScore> {
        Some(ReferenceScore {
            stars,
            median_star_area: psf_area,
            background: 0.1,
            background_variation: 0.01,
            score,
            psf_area,
            eccentricity: Some(eccentricity),
        })
    }

    /// Twenty steady lights with small scatter, then one of each kind of bad
    /// light.
    fn batch() -> Vec<Option<ReferenceScore>> {
        let mut scores = (0..20)
            .map(|index| {
                let jitter = (index as f32 * 0.37).sin();
                score(
                    40.0 + 2.0 * jitter,
                    1000 + (30.0 * jitter) as usize,
                    6.0 + 0.2 * jitter,
                    0.30 + 0.01 * jitter,
                )
            })
            .collect::<Vec<_>>();
        scores.push(score(15.0, 900, 6.1, 0.30)); // 20: cloud dims the stars
        scores.push(score(38.0, 300, 6.0, 0.30)); // 21: cloud over part of it
        scores.push(score(36.0, 980, 6.2, 0.45)); // 22: wind
        scores.push(score(30.0, 990, 9.5, 0.31)); // 23: lost focus
        scores.push(None); // 24: unreadable
        scores
    }

    #[test]
    fn turns_away_only_the_outliers() {
        let selection = select_frames(&batch(), &FrameSelectionOptions::default()).unwrap();
        assert_eq!(
            selection.kept().collect::<Vec<_>>(),
            (0..20).collect::<Vec<_>>()
        );
        let kinds = selection.verdicts[20..]
            .iter()
            .map(|verdict| verdict.unwrap_err().kind())
            .collect::<Vec<_>>();
        assert_eq!(
            kinds,
            [
                FrameRejectionKind::LowSignal,
                FrameRejectionKind::FewStars,
                FrameRejectionKind::Elongated,
                FrameRejectionKind::Soft,
                FrameRejectionKind::Unscored,
            ]
        );
        let elongated = selection.verdicts[22].unwrap_err();
        assert_eq!(elongated.value(), Some(0.45));
        assert!(elongated.limit().unwrap() < 0.45);
        assert!(
            selection.expected_snr_gain > 1.0,
            "{}",
            selection.expected_snr_gain
        );
    }

    #[test]
    fn the_eccentricity_margin_spares_a_tight_batch() {
        // Every light within 0.005 of the median: the robust spread is tiny,
        // and without the margin 3 sigma would turn away measurement noise.
        let scores = (0..10)
            .map(|index| score(40.0, 1000, 6.0, 0.30 + 0.001 * index as f32))
            .collect::<Vec<_>>();
        let selection = select_frames(&scores, &FrameSelectionOptions::default()).unwrap();
        assert_eq!(selection.kept().count(), 10);
        let absolute = FrameSelectionOptions {
            maximum_eccentricity: Some(0.305),
            ..FrameSelectionOptions::default()
        };
        let selection = select_frames(&scores, &absolute).unwrap();
        assert_eq!(selection.kept().count(), 6);
    }

    #[test]
    fn weight_factors_follow_the_psf_area_against_the_reference() {
        let scores = vec![
            score(40.0, 1000, 6.0, 0.3),
            score(40.0, 1000, 12.0, 0.3),
            score(40.0, 1000, 3.0, 0.3),
            score(40.0, 1000, 6.0, 0.3),
            score(40.0, 1000, 6.0, 0.3),
        ];
        let wide = FrameSelectionOptions {
            psf_area_sigma: 100.0,
            minimum_psf_area_margin: 10.0,
            ..FrameSelectionOptions::default()
        };
        let selection = select_frames(&scores, &wide).unwrap();
        assert_eq!(selection.weight_factor(0, 0), Some(1.0));
        assert_eq!(selection.weight_factor(1, 0), Some(0.5));
        assert_eq!(selection.weight_factor(2, 0), Some(2.0));
        let tight = FrameSelectionOptions {
            maximum_weight_factor: 1.5,
            ..wide
        };
        let selection = select_frames(&scores, &tight).unwrap();
        assert_eq!(selection.weight_factor(2, 0), Some(1.5));
    }

    #[test]
    fn the_reference_is_the_best_kept_light() {
        let mut scores = batch();
        // The sharpest, brightest light is elongated: it cannot be the reference.
        scores[22] = score(80.0, 1100, 5.0, 0.6);
        let selection = select_frames(&scores, &FrameSelectionOptions::default()).unwrap();
        let reference = selection.reference(&scores).unwrap();
        assert_ne!(reference, 22);
        assert!(selection.verdicts[reference].is_ok());
        assert_eq!(selection.weight_factor(22, reference), None);
    }

    #[test]
    fn refuses_nothing_to_select_and_bad_options() {
        assert!(select_frames(&[None, None], &FrameSelectionOptions::default()).is_err());
        let bad = FrameSelectionOptions {
            minimum_weight_factor: 2.0,
            ..FrameSelectionOptions::default()
        };
        assert!(select_frames(&batch(), &bad).is_err());
    }

    #[test]
    fn rejection_kinds_read_back_from_their_names() {
        for (kind, name) in FrameRejectionKind::NAMES {
            assert_eq!(kind.to_string(), name);
            assert_eq!(name.parse::<FrameRejectionKind>(), Ok(kind));
        }
        assert!("cloudy".parse::<FrameRejectionKind>().is_err());
    }
}
