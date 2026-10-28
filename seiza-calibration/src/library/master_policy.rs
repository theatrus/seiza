//! PSF Guard's remaining master decisions, independent of files and catalogs.
//! Adapted from theatrus/psf-guard 835961cb, calibration.rs (Apache-2.0).
use super::*;

/// Resolve already selected candidates after host verification/remapping.
/// Candidate order is significant. No matching, scanning or pixel I/O occurs.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MasterPlanRequest {
    pub frames: Vec<CalibrationFrame>,
    pub lights: Vec<Light>,
    pub quality: Vec<QualityRecord>,
    pub global_settings: Vec<Setting>,
    pub rigs: Vec<RigSettings>,
    pub flat_basis: FlatBasis,
    pub flat_family: Option<FlatFamily>,
    pub flat_held_back: Vec<String>,
    pub flat_star_masking: bool,
    pub flat_stability: FlatRetryPolicy,
    pub master_failures: Vec<MasterFailure>,
    pub pedestal_evidence: Vec<PedestalEvidence>,
    pub force_flat_without_pedestal: bool,
    pub flat_association: Association,
    pub dark_association: Association,
}

fn setting_values(rows: Vec<Setting>, scope: &str) -> crate::Result<rules::RuleValues> {
    let mut result = rules::RuleValues::new();
    for row in rows {
        let spec = rules::fields()
            .iter()
            .find(|spec| spec.key == row.key)
            .ok_or_else(|| invalid(format!("{scope}: unknown setting '{}'", row.key)))?;
        rules::validate(spec, &row.value)
            .map_err(|error| invalid(format!("{scope}.{}: {error}", row.key)))?;
        if result.insert(row.key.clone(), row.value).is_some() {
            return Err(invalid(format!(
                "{scope}: duplicate setting key '{}'",
                row.key
            )));
        }
    }
    Ok(result)
}

pub(super) fn rule_book(global: Vec<Setting>, rigs: Vec<RigSettings>) -> crate::Result<RuleBook> {
    let mut layers = BTreeMap::new();
    for rig in rigs {
        let values = setting_values(rig.settings, &format!("rigs[{}].settings", rig.rig_uuid))?;
        if layers
            .insert(
                rig.rig_uuid,
                rules::RigLayer {
                    name: rig.name,
                    values,
                },
            )
            .is_some()
        {
            return Err(invalid("duplicate rig settings"));
        }
    }
    Ok(RuleBook::new(
        setting_values(global, "global_settings")?,
        layers,
    ))
}
/// Produce the same recipes/identity as library planning from selected rows.
/// Hosts must replan after changed verification, grades, options or sources.
pub fn plan_masters(request: MasterPlanRequest) -> crate::Result<Session> {
    let light = request
        .lights
        .first()
        .ok_or_else(|| invalid("master planning needs a light"))?;
    let (rules, quality) = check_request(
        &request.frames,
        &request.lights,
        request.quality,
        request.global_settings,
        request.rigs,
    )?;
    supplied::validate(
        &request.frames,
        &rules,
        request.flat_association,
        request.dark_association,
    )?;
    let mut selected = CalibrationSelection {
        rules: rules.for_light(light),
        flat_basis: request.flat_basis,
        flat_family: request.flat_family,
        flat_held_back: request.flat_held_back,
        ..Default::default()
    };
    for frame in request.frames {
        selected.of_kind_mut(frame.kind).push(frame);
    }
    if request.flat_association == Association::OneSet {
        selected.one_set.insert(CalibrationKind::Flat);
    }
    if request.dark_association == Association::OneSet {
        selected.one_set.insert(CalibrationKind::Dark);
    }
    let options = SessionOptions {
        flat_star_masking: request.flat_star_masking,
        flat_stability: request.flat_stability,
    };
    let mut sessions = vec![resolve_session(&selected, &quality, &options)];
    finalize(
        &mut sessions,
        &vec![0; request.lights.len()],
        &request.lights,
        &request.master_failures,
        &request.pedestal_evidence,
        request.force_flat_without_pedestal,
    )?;
    Ok(sessions.remove(0))
}

/// Coherence alone, for previews and build retries; accepts caller-ordered rows.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MasterCohortRequest {
    pub kind: CalibrationKind,
    pub frames: Vec<CalibrationFrame>,
    pub settings: Vec<Setting>,
}
#[derive(Debug, Serialize)]
pub struct MasterCohort {
    pub kept: Vec<String>,
    pub dropped: Vec<String>,
}
/// Select a complete/coherent cohort using the same policy as master recipes.
/// Settings are validated. Metadata remains permissive for catalog previews;
/// plan_masters validates executable input identities and source records.
pub fn master_cohort(request: MasterCohortRequest) -> crate::Result<MasterCohort> {
    let values = setting_values(request.settings, "settings")?;
    let rules = RuleBook::new(values, BTreeMap::new()).global();
    let subset = coherent_cohort(request.kind, &request.frames, &rules);
    Ok(MasterCohort {
        kept: subset.kept.into_iter().map(|f| f.frame_uuid).collect(),
        dropped: subset.dropped.into_iter().map(|f| f.frame_uuid).collect(),
    })
}

/// Defaults preserve PSF Guard's trim, alternate-session and honest fallback.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FlatRetryPolicy {
    pub enabled: bool,
    pub trim_unstable: bool,
    pub try_next_session: bool,
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
/// Accepted input UUIDs and measured disagreement in identical order.
/// A missing stability measurement is represented by empty disagreement.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FlatAttempt {
    pub used: Vec<String>,
    pub disagreement: Vec<usize>,
    pub drift_threshold: usize,
    pub feature_pixels: usize,
    pub memo: bool,
    /// Optional host label for explanations; never affects the decision.
    pub label: Option<String>,
}
impl FlatAttempt {
    fn drifting(&self) -> Vec<&String> {
        self.used
            .iter()
            .zip(&self.disagreement)
            .filter(|(_, n)| **n >= self.drift_threshold)
            .map(|(id, _)| id)
            .collect()
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FlatRetryRequest {
    /// Pass the returned session directly to retain its policy, held-back rows
    /// and resolved time window. Conflicting explicit overrides are rejected.
    pub session: Option<Session>,
    pub candidates: Vec<CalibrationFrame>,
    pub held_back: Vec<String>,
    pub settings: Vec<Setting>,
    pub policy: Option<FlatRetryPolicy>,
    pub first: FlatAttempt,
    /// Set true when the retry failed/yielded no master. Successful feedback
    /// implies attempted, so callers need not duplicate that state.
    pub trimmed_attempted: bool,
    pub trimmed: Option<FlatAttempt>,
    pub alternate_attempted: bool,
    pub alternate: Option<FlatAttempt>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FlatRetryAction {
    PublishFirst,
    TryTrimmed,
    PublishTrimmed,
    TryAlternate,
    PublishAlternate,
    Withhold,
}
#[derive(Debug, Serialize)]
pub struct FlatRetryDecision {
    pub action: FlatRetryAction,
    pub inputs: Vec<String>,
    pub rebuild_first: bool,
    pub note: Option<String>,
}
/// Whether a flat set whose `drifting` of `used` frames disagree with the
/// master still holds a stable majority that makes an honest master alone.
pub fn trims_to_majority(used: usize, drifting: usize) -> bool {
    let stable = used.saturating_sub(drifting);
    stable >= MIN_MASTER_FRAMES && stable * 2 >= used
}

/// The candidates shot within one flat session of `set`: the set itself
/// and whatever the clusterer left out of it from the same run, within
/// `window_seconds` of its first and last frames. The next set never comes
/// from these: they saw the same window.
pub fn same_session_as(
    set: &[CalibrationFrame],
    candidates: &[CalibrationFrame],
    window_seconds: u64,
) -> Vec<CalibrationFrame> {
    let window = i64::try_from(window_seconds).unwrap_or(i64::MAX);
    let first = set.iter().filter_map(|frame| frame.captured_at).min();
    let last = set.iter().filter_map(|frame| frame.captured_at).max();
    candidates
        .iter()
        .filter(|candidate| {
            set.iter()
                .any(|frame| frame.frame_uuid == candidate.frame_uuid)
                || match (first, last, candidate.captured_at) {
                    (Some(first), Some(last), Some(at)) => {
                        at >= first.saturating_sub(window) && at <= last.saturating_add(window)
                    }
                    _ => false,
                }
        })
        .cloned()
        .collect()
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

/// The note for a master built after leaving out the frames that disagreed.
pub fn trimmed_note(left_out: usize, used: usize, feature_pixels: usize) -> String {
    format!(
        "Left out {left_out} of {used} flats that disagreed with the rest of the set at \
         {feature_pixels} pixels ({SPOTS})"
    )
}

/// The note for a master built from the next set instead of the nearest.
pub fn alternate_note(first: &str, other: &str, feature_pixels: usize) -> String {
    format!(
        "The nearest {first} disagree with each other at {feature_pixels} pixels ({SPOTS}); \
         used {other} instead"
    )
}

/// The note for the nearest set kept, or withheld, though unstable.
pub fn unstable_note(
    drifting: usize,
    used: usize,
    feature_pixels: usize,
    withheld: bool,
) -> String {
    let disposition = if withheld {
        "Withheld an unstable flat set"
    } else {
        "Built from an unstable flat set"
    };
    format!(
        "{disposition}: {drifting} of {used} frames disagree with the master at \
         {feature_pixels} pixels ({SPOTS}). Retake these flats"
    )
}

/// Decide the next flat retry/publication from immutable execution feedback.
/// Hosts retain staging, cancellation, cache publication and pixel measurement.
pub fn flat_retry(request: FlatRetryRequest) -> crate::Result<FlatRetryDecision> {
    let mut rules = (*RuleBook::new(
        setting_values(request.settings.clone(), "settings")?,
        BTreeMap::new(),
    )
    .global())
    .clone();
    let default_policy = FlatRetryPolicy::default();
    let (policy, held_back, window) = if let Some(session) = &request.session {
        session.tolerances.validate()?;
        if request
            .policy
            .as_ref()
            .is_some_and(|p| p != &session.flat_stability)
            || (!request.held_back.is_empty() && request.held_back != session.flat_held_back)
            || (!request.settings.is_empty()
                && rules.flat_session_seconds != session.tolerances.flat_session_seconds)
        {
            return Err(invalid(
                "flat retry overrides conflict with the supplied session",
            ));
        }
        // An alternate set is clustered as the session's own flats were.
        rules.flat_session_seconds = session.tolerances.flat_session_seconds;
        rules.rotation_deg = session.tolerances.rotation_deg;
        rules.focal_length_mm = session.tolerances.focal_length_mm;
        rules.dark_temperature_c = session.tolerances.dark_temperature_c;
        (
            &session.flat_stability,
            &session.flat_held_back,
            session.tolerances.flat_session_seconds,
        )
    } else {
        (
            request.policy.as_ref().unwrap_or(&default_policy),
            &request.held_back,
            rules.flat_session_seconds,
        )
    };
    check_request(&request.candidates, &[], Vec::new(), Vec::new(), Vec::new())?;
    let ids: BTreeSet<_> = request.candidates.iter().map(|f| &f.frame_uuid).collect();
    for attempt in std::iter::once(&request.first)
        .chain(request.trimmed.iter())
        .chain(request.alternate.iter())
    {
        let mut seen = BTreeSet::new();
        if attempt.used.len() < MIN_MASTER_FRAMES
            || (!attempt.disagreement.is_empty()
                && (attempt.disagreement.len() != attempt.used.len()
                    || attempt.drift_threshold == 0))
            || attempt
                .used
                .iter()
                .any(|id| !ids.contains(id) || !seen.insert(id))
        {
            return Err(invalid(
                "flat feedback needs unique candidate inputs and aligned disagreement",
            ));
        }
    }
    let trimmed_attempted = request.trimmed_attempted || request.trimmed.is_some();
    let alternate_attempted = request.alternate_attempted || request.alternate.is_some();
    if request.first.used.is_empty()
        || request
            .candidates
            .iter()
            .any(|f| f.kind != CalibrationKind::Flat)
    {
        return Err(invalid("flat feedback needs a nonempty first flat cohort"));
    }
    let drifting = request.first.drifting();
    if request.trimmed.as_ref().is_some_and(|attempt| {
        attempt
            .used
            .iter()
            .any(|id| !request.first.used.contains(id) || drifting.contains(&id))
    }) {
        return Err(invalid(
            "trim feedback contains an input outside the stable first cohort",
        ));
    }
    let first = &request.first;
    let drifting = first.drifting();
    let mut result = FlatRetryDecision {
        action: FlatRetryAction::PublishFirst,
        inputs: Vec::new(),
        rebuild_first: first.memo,
        note: None,
    };
    if !policy.enabled || drifting.is_empty() {
        return Ok(result);
    }
    let used = first.used.len();
    let trim = policy.trim_unstable && trims_to_majority(used, drifting.len());
    if trim && !trimmed_attempted {
        result.action = FlatRetryAction::TryTrimmed;
        result.inputs = first
            .used
            .iter()
            .filter(|id| !drifting.contains(id))
            .cloned()
            .collect();
        return Ok(result);
    }
    if trim
        && let Some(attempt) = &request.trimmed
        && attempt.drifting().is_empty()
        && !attempt.memo
    {
        result.action = FlatRetryAction::PublishTrimmed;
        result.note = Some(trimmed_note(drifting.len(), used, first.feature_pixels));
        return Ok(result);
    }
    let rows_of = |used: &[String]| -> Vec<CalibrationFrame> {
        used.iter()
            .filter_map(|id| request.candidates.iter().find(|f| &f.frame_uuid == id))
            .cloned()
            .collect()
    };
    let first_rows = rows_of(&first.used);
    let same_session: BTreeSet<String> = same_session_as(&first_rows, &request.candidates, window)
        .into_iter()
        .map(|f| f.frame_uuid)
        .collect();
    let eligible_alternate = |f: &CalibrationFrame| {
        !f.is_master && !held_back.contains(&f.frame_uuid) && !same_session.contains(&f.frame_uuid)
    };
    let alternate_ids: BTreeSet<_> = request
        .candidates
        .iter()
        .filter(|f| eligible_alternate(f))
        .map(|f| &f.frame_uuid)
        .collect();
    if request.alternate.as_ref().is_some_and(|attempt| {
        attempt.used.is_empty() || attempt.used.iter().any(|id| !alternate_ids.contains(id))
    }) {
        return Err(invalid(
            "alternate feedback contains an input in the original/held-back session",
        ));
    }
    if policy.try_next_session && !alternate_attempted {
        // The next session out, as one master takes it: the nearest
        // cluster of the remaining candidates, given nearest first, not
        // every flat outside the first window.
        let remaining: Vec<CalibrationFrame> = request
            .candidates
            .iter()
            .filter(|f| eligible_alternate(f))
            .cloned()
            .collect();
        let cohort = coherent_cohort(CalibrationKind::Flat, &remaining, &rules);
        if cohort.kept.len() >= MIN_MASTER_FRAMES {
            result.action = FlatRetryAction::TryAlternate;
            result.inputs = cohort.kept.into_iter().map(|f| f.frame_uuid).collect();
            return Ok(result);
        }
    }
    if policy.try_next_session
        && let Some(attempt) = &request.alternate
        && attempt.drifting().is_empty()
        && !attempt.memo
    {
        let label = |attempt: &FlatAttempt| {
            attempt
                .label
                .clone()
                .unwrap_or_else(|| flat_set_label(&rows_of(&attempt.used)))
        };
        result.action = FlatRetryAction::PublishAlternate;
        result.note = Some(alternate_note(
            &label(first),
            &label(attempt),
            first.feature_pixels,
        ));
        return Ok(result);
    }
    result.action = if policy.retain_unstable {
        FlatRetryAction::PublishFirst
    } else {
        FlatRetryAction::Withhold
    };
    result.note = Some(unstable_note(
        drifting.len(),
        used,
        first.feature_pixels,
        result.action == FlatRetryAction::Withhold,
    ));
    Ok(result)
}
