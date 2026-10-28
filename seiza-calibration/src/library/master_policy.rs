//! PSF Guard's flat master retries, independent of files and catalogs.
//! Adapted from theatrus/psf-guard 835961cb, calibration.rs (Apache-2.0).
use super::*;

/// Which retries a flat master may take when its frames disagree about
/// spots that come and go. The default is PSF Guard's: leave a minority
/// out, else take the next session, else keep the set and say so.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FlatRetryPolicy {
    /// Judge a flat set at all; off, the first build always serves.
    pub enabled: bool,
    /// Rebuild without a minority of frames that disagree.
    pub trim_unstable: bool,
    /// Take the next flat session when the set as a whole disagrees.
    pub try_next_session: bool,
    /// Keep an unstable set nothing replaces, with a note, rather than
    /// withhold the flat.
    pub retain_unstable: bool,
}
impl Default for FlatRetryPolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            trim_unstable: true,
            try_next_session: true,
            retain_unstable: true,
        }
    }
}

/// One flat master build, as its judgement needs it: the frames the
/// integrator combined and how far each disagreed with the master.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FlatAttempt {
    /// The frames combined, by UUID, in the order they were offered.
    pub used: Vec<String>,
    /// For each of `used`, the master's feature pixels it disagrees at;
    /// empty when the stability was not measured.
    pub disagreement: Vec<usize>,
    /// The disagreement at which a frame drifts: it was looking through a
    /// different sensor window.
    pub drift_threshold: usize,
    /// The master's feature pixels the frames were compared at.
    pub feature_pixels: usize,
    /// A remembered judgement of a set once found unstable: its record
    /// remains, its file does not, so it never serves as it is.
    pub memo: bool,
    /// What a note calls the set; "20 G flats from 2026-09-10" when absent.
    /// Never affects the decision.
    pub label: Option<String>,
}
impl FlatAttempt {
    /// The frames whose disagreement reaches the threshold.
    pub fn drifting(&self) -> Vec<&str> {
        self.used
            .iter()
            .zip(&self.disagreement)
            .filter(|(_, n)| **n >= self.drift_threshold)
            .map(|(id, _)| id.as_str())
            .collect()
    }
    /// Whether this build makes an honest master: nothing drifts and it
    /// has a file.
    pub fn stable(&self) -> bool {
        self.drifting().is_empty() && !self.memo
    }
}

/// What a flat master does next.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FlatRetryAction {
    /// Publish the first build: it is stable, or nothing replaces it.
    PublishFirst,
    /// Build again from [`FlatRetryDecision::inputs`], the first set less
    /// the frames that drift.
    TryTrimmed,
    /// Publish the trimmed build.
    PublishTrimmed,
    /// Build the next flat session, [`FlatRetryDecision::inputs`].
    TryAlternate,
    /// Publish the next session's build.
    PublishAlternate,
    /// Publish no flat master: the set is unstable and the policy keeps
    /// none such.
    Withhold,
}

/// A flat retry's next step.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FlatRetryDecision {
    pub action: FlatRetryAction,
    /// The frames to build, by UUID, for a `Try` action; empty otherwise.
    pub inputs: Vec<String>,
    /// The first build is a memo: should it serve, build it again for real
    /// first.
    pub rebuild_first: bool,
    /// What the master's card says about the choice.
    pub note: Option<String>,
}
impl FlatRetryDecision {
    /// The candidates [`FlatRetryDecision::inputs`] names, in its order.
    pub fn frames<'a>(&self, candidates: &'a [CalibrationFrame]) -> Vec<&'a CalibrationFrame> {
        self.inputs
            .iter()
            .filter_map(|id| candidates.iter().find(|f| &f.frame_uuid == id))
            .collect()
    }
}

/// A flat master's way to a set that agrees with itself, decided one step
/// at a time from what each build showed. A minority of drifting frames is
/// left out; a set that disagrees as a whole gives way to the next flat
/// session; when nothing better builds, the first set serves with a note.
/// The host builds, measures, stages and publishes; this decides which.
///
/// ```text
/// let mut retry = FlatRetry::new(first, &flats, &held_back, &rules, policy)?;
/// loop {
///     let step = retry.next();
///     match step.action {
///         FlatRetryAction::TryTrimmed => retry.trimmed(build(step.frames(&flats)))?,
///         FlatRetryAction::TryAlternate => retry.alternate(build(step.frames(&flats)))?,
///         _ => return publish(step),
///     }
/// }
/// ```
#[derive(Clone, Debug)]
pub struct FlatRetry<'a> {
    first: FlatAttempt,
    candidates: &'a [CalibrationFrame],
    held_back: &'a [String],
    rules: &'a Rules,
    policy: FlatRetryPolicy,
    /// Tried: the build, or `None` when it made no master.
    trimmed: Option<Option<FlatAttempt>>,
    alternate: Option<Option<FlatAttempt>>,
}
impl<'a> FlatRetry<'a> {
    /// The retry of a flat master first built as `first` from `candidates`,
    /// the light's flats nearest first. `held_back` frames, from the far
    /// side of a measured change, never stand in for it. `rules` give the
    /// flat session window and the coherence of the next set.
    pub fn new(
        first: FlatAttempt,
        candidates: &'a [CalibrationFrame],
        held_back: &'a [String],
        rules: &'a Rules,
        policy: FlatRetryPolicy,
    ) -> crate::Result<Self> {
        if first.used.is_empty() || candidates.iter().any(|f| f.kind != CalibrationKind::Flat) {
            return Err(invalid("flat feedback needs a nonempty first flat cohort"));
        }
        let retry = Self {
            first,
            candidates,
            held_back,
            rules,
            policy,
            trimmed: None,
            alternate: None,
        };
        retry.check(&retry.first)?;
        Ok(retry)
    }

    /// Record the trimmed build, or `None` when it made no master.
    pub fn trimmed(&mut self, attempt: Option<FlatAttempt>) -> crate::Result<()> {
        if let Some(attempt) = &attempt {
            self.check(attempt)?;
            let drifting = self.first.drifting();
            if attempt
                .used
                .iter()
                .any(|id| !self.first.used.contains(id) || drifting.contains(&id.as_str()))
            {
                return Err(invalid(
                    "trim feedback contains an input outside the stable first cohort",
                ));
            }
        }
        self.trimmed = Some(attempt);
        Ok(())
    }

    /// Record the next session's build, or `None` when it made no master.
    pub fn alternate(&mut self, attempt: Option<FlatAttempt>) -> crate::Result<()> {
        if let Some(attempt) = &attempt {
            self.check(attempt)?;
            let eligible = self.next_session_frames();
            if attempt
                .used
                .iter()
                .any(|id| !eligible.iter().any(|f| &f.frame_uuid == id))
            {
                return Err(invalid(
                    "alternate feedback contains an input in the original/held-back session",
                ));
            }
        }
        self.alternate = Some(attempt);
        Ok(())
    }

    /// The next step.
    pub fn next(&self) -> FlatRetryDecision {
        let first = &self.first;
        let drifting = first.drifting();
        let decide = |action, inputs, note| FlatRetryDecision {
            action,
            inputs,
            rebuild_first: first.memo,
            note,
        };
        if !self.policy.enabled || drifting.is_empty() {
            return decide(FlatRetryAction::PublishFirst, Vec::new(), None);
        }
        let used = first.used.len();
        let pixels = first.feature_pixels;
        let stable = used.saturating_sub(drifting.len());
        // A minority of odd frames: the rest of this set still make an
        // honest master. Only this set's frames are offered, so the trim
        // cannot slide onto another session while the card speaks of
        // leaving frames out.
        let trim = self.policy.trim_unstable && stable >= MIN_MASTER_FRAMES && stable * 2 >= used;
        if trim {
            match &self.trimmed {
                None => {
                    let rest = first
                        .used
                        .iter()
                        .filter(|id| !drifting.contains(&id.as_str()))
                        .cloned()
                        .collect();
                    return decide(FlatRetryAction::TryTrimmed, rest, None);
                }
                Some(Some(trimmed)) if trimmed.stable() => {
                    let note = format!(
                        "Left out {} of {used} flats that disagreed with the rest of the set at \
                         {pixels} pixels ({SPOTS})",
                        drifting.len()
                    );
                    return decide(FlatRetryAction::PublishTrimmed, Vec::new(), Some(note));
                }
                Some(_) => {}
            }
        }
        // The set disagrees with itself: the next set takes its place when
        // there is one, as one master takes it: the nearest coherent
        // cluster of what remains, given nearest first.
        if self.policy.try_next_session {
            match &self.alternate {
                None => {
                    let remaining: Vec<CalibrationFrame> =
                        self.next_session_frames().into_iter().cloned().collect();
                    let cohort = coherent_cohort(CalibrationKind::Flat, &remaining, self.rules);
                    if cohort.kept.len() >= MIN_MASTER_FRAMES {
                        let inputs = cohort.kept.into_iter().map(|f| f.frame_uuid).collect();
                        return decide(FlatRetryAction::TryAlternate, inputs, None);
                    }
                }
                Some(Some(alternate)) if alternate.stable() => {
                    let note = format!(
                        "The nearest {} disagree with each other at {pixels} pixels ({SPOTS}); \
                         used {} instead",
                        self.label(first),
                        self.label(alternate)
                    );
                    return decide(FlatRetryAction::PublishAlternate, Vec::new(), Some(note));
                }
                Some(_) => {}
            }
        }
        let (action, disposition) = if self.policy.retain_unstable {
            (
                FlatRetryAction::PublishFirst,
                "Built from an unstable flat set",
            )
        } else {
            (FlatRetryAction::Withhold, "Withheld an unstable flat set")
        };
        let note = format!(
            "{disposition}: {} of {used} frames disagree with the master at {pixels} pixels \
             ({SPOTS}). Retake these flats",
            drifting.len()
        );
        decide(action, Vec::new(), Some(note))
    }

    /// Refuse feedback that names frames twice, frames that are not
    /// candidates, or disagreement that does not line up with them.
    fn check(&self, attempt: &FlatAttempt) -> crate::Result<()> {
        let mut seen = BTreeSet::new();
        if attempt.used.len() < MIN_MASTER_FRAMES
            || (!attempt.disagreement.is_empty()
                && (attempt.disagreement.len() != attempt.used.len()
                    || attempt.drift_threshold == 0))
            || attempt.used.iter().any(|id| {
                !self.candidates.iter().any(|f| &f.frame_uuid == id) || !seen.insert(id)
            })
        {
            return Err(invalid(
                "flat feedback needs unique candidate inputs and aligned disagreement",
            ));
        }
        Ok(())
    }

    /// The candidates the next session may come from: raw frames, not held
    /// back, and not shot within one flat session of the first set. The
    /// frames the clusterer dropped from the first set's run saw the same
    /// window, so they go with it.
    fn next_session_frames(&self) -> Vec<&'a CalibrationFrame> {
        let window = i64::try_from(self.rules.flat_session_seconds).unwrap_or(i64::MAX);
        let set: Vec<&CalibrationFrame> = self.rows(&self.first.used);
        let first = set.iter().filter_map(|frame| frame.captured_at).min();
        let last = set.iter().filter_map(|frame| frame.captured_at).max();
        let same_session = |candidate: &CalibrationFrame| {
            set.iter().any(|frame| frame.frame_uuid == candidate.frame_uuid)
                || match (first, last, candidate.captured_at) {
                    (Some(first), Some(last), Some(at)) => {
                        at >= first.saturating_sub(window) && at <= last.saturating_add(window)
                    }
                    _ => false,
                }
        };
        self.candidates
            .iter()
            .filter(|f| {
                !f.is_master && !self.held_back.contains(&f.frame_uuid) && !same_session(f)
            })
            .collect()
    }

    fn rows(&self, used: &[String]) -> Vec<&'a CalibrationFrame> {
        used.iter()
            .filter_map(|id| self.candidates.iter().find(|f| &f.frame_uuid == id))
            .collect()
    }

    fn label(&self, attempt: &FlatAttempt) -> String {
        attempt.label.clone().unwrap_or_else(|| {
            let rows: Vec<CalibrationFrame> =
                self.rows(&attempt.used).into_iter().cloned().collect();
            flat_set_label(&rows)
        })
    }
}

/// "20 G flats from 2026-09-10", for a note about one flat set.
pub fn flat_set_label(frames: &[CalibrationFrame]) -> String {
    let filter = frames
        .iter()
        .find_map(|frame| frame.filter.as_deref())
        .map(|name| format!("{name} "))
        .unwrap_or_default();
    let day = |timestamp: i64| {
        chrono::DateTime::from_timestamp(timestamp, 0)
            .map(|at| at.format("%Y-%m-%d").to_string())
            .unwrap_or_else(|| timestamp.to_string())
    };
    let first = frames.iter().filter_map(|frame| frame.captured_at).min();
    let last = frames.iter().filter_map(|frame| frame.captured_at).max();
    let when = match (first, last) {
        (Some(first), Some(last)) if day(first) != day(last) => {
            format!(" from {} to {}", day(first), day(last))
        }
        (Some(first), _) => format!(" from {}", day(first)),
        _ => String::new(),
    };
    format!("{} {filter}flats{when}", frames.len())
}

const SPOTS: &str = "spots that come and go across the run: dew or frost on the sensor window?";
