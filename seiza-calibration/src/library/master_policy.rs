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
fn validate_frames(frames: &[CalibrationFrame]) -> crate::Result<()> {
    let mut ids = BTreeSet::new();
    for frame in frames {
        if frame.frame_uuid.is_empty() || !ids.insert(&frame.frame_uuid) {
            return Err(invalid(
                "calibration frame UUIDs must be nonempty and unique",
            ));
        }
        validate_values(&frame_signature(frame).seiza)?;
        if frame.source_verified
            && (frame.source_path.as_os_str().is_empty() || frame.source_fingerprint.is_empty())
        {
            return Err(invalid("verified sources need a path and fingerprint"));
        }
    }
    Ok(())
}

/// Produce the same recipes/identity as library planning from selected rows.
/// Hosts must replan after changed verification, grades, options or sources.
pub fn plan_masters(request: MasterPlanRequest) -> crate::Result<Session> {
    validate_frames(&request.frames)?;
    let light = request
        .lights
        .first()
        .ok_or_else(|| invalid("master planning needs a light"))?;
    for light in &request.lights {
        validate_values(&light_signature(light).seiza)?;
    }
    let rules = rule_book(request.global_settings, request.rigs)?.for_light(light);
    let mut selected = CalibrationSelection {
        rules,
        flat_basis: request.flat_basis,
        flat_family: request.flat_family,
        flat_held_back: request.flat_held_back,
        ..Default::default()
    };
    let ids: BTreeSet<_> = request
        .frames
        .iter()
        .map(|f| f.frame_uuid.clone())
        .collect();
    let mut quality = BTreeMap::new();
    for row in request.quality {
        if !ids.contains(&row.frame_uuid) || quality.insert(row.frame_uuid.clone(), row).is_some() {
            return Err(invalid("quality rows must name unique selected frames"));
        }
    }
    for frame in request.frames {
        match frame.kind {
            CalibrationKind::Bias => &mut selected.bias,
            CalibrationKind::Dark => &mut selected.dark,
            CalibrationKind::DarkFlat => &mut selected.dark_flat,
            CalibrationKind::Flat => &mut selected.flat,
        }
        .push(frame);
    }
    for frames in [
        &mut selected.bias,
        &mut selected.dark,
        &mut selected.dark_flat,
        &mut selected.flat,
    ] {
        if frames.iter().all(|f| f.source_verified) {
            frames.truncate(MAX_MASTER_FRAMES);
        }
    }
    let session = resolved_session(
        &selected,
        &quality,
        &HashMap::new(),
        request.flat_star_masking,
        request.flat_stability,
    );
    let mut sessions = vec![session];
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

pub(super) fn resolved_session(
    selected: &CalibrationSelection,
    quality: &BTreeMap<String, QualityRecord>,
    judged: &HashMap<String, Judged>,
    masking: bool,
    stability: FlatRetryPolicy,
) -> Session {
    let mut recipes = recipes(selected);
    let frames: Vec<_> = selected
        .bias
        .iter()
        .chain(&selected.dark)
        .chain(&selected.dark_flat)
        .chain(&selected.flat)
        .cloned()
        .collect();
    for recipe in &mut recipes {
        recipe.flat_star_masking = recipe.kind == CalibrationKind::Flat && masking;
        recipe.dark_level_screening = matches!(
            recipe.kind,
            CalibrationKind::Dark | CalibrationKind::DarkFlat
        ) && recipe.inputs.iter().any(|id| {
            !judged.contains_key(id)
                && quality.get(id).is_none_or(|q| {
                    q.override_choice.is_none()
                        && !(q.grade.is_some() && selected_frame_current(id, q, &frames))
                })
        });
    }
    Session {
        key: session_key(selected, &recipes, &stability),
        tolerances: selected.match_tolerances(),
        recipes,
        application: Application::default(),
        flat_stability: stability,
        flat_held_back: selected.flat_held_back.clone(),
    }
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
    let rules = RuleBook::new(values, BTreeMap::new()).for_rig("");
    let subset = master_subset_report(request.kind, &request.frames, &rules);
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
    let rules = RuleBook::new(
        setting_values(request.settings.clone(), "settings")?,
        BTreeMap::new(),
    )
    .for_rig("");
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
    validate_frames(&request.candidates)?;
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
        !held_back.contains(&f.frame_uuid)
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
        result.action = FlatRetryAction::TryAlternate;
        result.inputs = request
            .candidates
            .iter()
            .filter(|f| eligible_alternate(f))
            .map(|f| f.frame_uuid.clone())
            .collect();
        return Ok(result);
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
