//! The C ABI's master planning, cohort and flat retry calls, on the native
//! session, cohort and retry decisions.
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

/// A request's setting rows as a layer: each key once.
fn layer(rows: Vec<Setting>, scope: &str) -> crate::Result<rules::RuleValues> {
    let mut result = rules::RuleValues::new();
    for row in rows {
        if result.insert(row.key.clone(), row.value).is_some() {
            return Err(invalid(format!(
                "{scope}: duplicate setting key '{}'",
                row.key
            )));
        }
    }
    Ok(result)
}

/// One request layer, checked the way a settings panel saves it.
fn setting_values(rows: Vec<Setting>, scope: &str) -> crate::Result<rules::RuleValues> {
    let values = layer(rows, scope)?;
    rules::RuleBook::validated(values.clone(), BTreeMap::new())
        .map_err(|error| setting_error(error, scope))?;
    Ok(values)
}

fn setting_error(error: rules::SettingError, scope: &str) -> crate::Error {
    match error.problem {
        rules::SettingProblem::Unknown => {
            invalid(format!("{scope}: unknown setting '{}'", error.key))
        }
        rules::SettingProblem::Invalid(problem) => {
            invalid(format!("{scope}.{}: {problem}", error.key))
        }
    }
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
    let associations = Associations {
        flat: request.flat_association,
        dark: request.dark_association,
    };
    associations.check(&request.frames, &rules)?;
    let selected = CalibrationSelection {
        flat_basis: request.flat_basis,
        flat_family: request.flat_family,
        flat_held_back: request.flat_held_back,
        ..CalibrationSelection::of_frames(request.frames, rules.for_light(light), associations)
    };
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
/// Decide the next flat retry/publication from immutable execution feedback,
/// with [`FlatRetry`]. Hosts retain staging, cancellation, cache publication
/// and pixel measurement.
pub fn flat_retry(request: FlatRetryRequest) -> crate::Result<FlatRetryDecision> {
    let mut rules = (*RuleBook::new(
        setting_values(request.settings.clone(), "settings")?,
        BTreeMap::new(),
    )
    .global())
    .clone();
    let (policy, held_back) = if let Some(session) = &request.session {
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
        (session.flat_stability.clone(), &session.flat_held_back)
    } else {
        (request.policy.clone().unwrap_or_default(), &request.held_back)
    };
    check_request(&request.candidates, &[], Vec::new(), Vec::new(), Vec::new())?;
    let mut retry = FlatRetry::new(
        request.first,
        &request.candidates,
        held_back,
        &rules,
        policy,
    )?;
    // A successful build implies it was tried.
    if request.trimmed_attempted || request.trimmed.is_some() {
        retry.trimmed(request.trimmed)?;
    }
    if request.alternate_attempted || request.alternate.is_some() {
        retry.alternate(request.alternate)?;
    }
    Ok(retry.next())
}
