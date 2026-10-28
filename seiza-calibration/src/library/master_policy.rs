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
    let stable = used - drifting.len();
    let spots = "spots that come and go across the run: dew or frost on the sensor window?";
    let trim = policy.trim_unstable && stable >= MIN_MASTER_FRAMES && stable * 2 >= used;
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
        result.note = Some(format!(
            "Left out {} of {used} flats that disagreed with the rest of the set at {} pixels ({spots})",
            drifting.len(),
            first.feature_pixels
        ));
        return Ok(result);
    }
    let first_rows: Vec<_> = request
        .candidates
        .iter()
        .filter(|f| first.used.contains(&f.frame_uuid))
        .collect();
    let start = first_rows.iter().filter_map(|f| f.captured_at).min();
    let end = first_rows.iter().filter_map(|f| f.captured_at).max();
    let window =
        i64::try_from(window).map_err(|_| invalid("flat retry session window is too large"))?;
    let eligible_alternate = |f: &CalibrationFrame| {
        !f.is_master
            && !held_back.contains(&f.frame_uuid)
            && !first.used.contains(&f.frame_uuid)
            && !matches!((start, end, f.captured_at), (Some(a), Some(b), Some(t)) if t >= a.saturating_sub(window) && t <= b.saturating_add(window))
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
                .unwrap_or_else(|| format!("{} flats", attempt.used.len()))
        };
        result.action = FlatRetryAction::PublishAlternate;
        result.note = Some(format!(
            "The nearest {} disagree with each other at {} pixels ({spots}); used {} instead",
            label(first),
            first.feature_pixels,
            label(attempt)
        ));
        return Ok(result);
    }
    result.action = if policy.retain_unstable {
        FlatRetryAction::PublishFirst
    } else {
        FlatRetryAction::Withhold
    };
    let disposition = if result.action == FlatRetryAction::Withhold {
        "Withheld an unstable flat set"
    } else {
        "Built from an unstable flat set"
    };
    result.note = Some(format!(
        "{disposition}: {} of {used} frames disagree with the master at {} pixels ({spots}). Retake these flats",
        drifting.len(),
        first.feature_pixels
    ));
    Ok(result)
}
