// Extracted policy from theatrus/psf-guard 79f5e31, calibration.rs:
// resolve_masters_with_recipe and estimate_flat_pedestal. Apache-2.0.
use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MasterFailure {
    pub session_key: String,
    pub kind: CalibrationKind,
    pub reason: String,
}
/// Successful per-light fits from fit_flat_pedestal, in the returned sample set.
/// A pinned value reproduces a previous stack and wins over fresh estimates.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PedestalEvidence {
    pub session_key: String,
    pub estimates: Vec<f32>,
    pub pinned: Option<f32>,
}
#[derive(Clone, Copy, Debug, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FlatUse {
    #[default]
    Unavailable,
    WithMasters,
    NeedsPedestalFit,
    FittedPedestal,
    ForcedWithoutPedestal,
    Withheld,
}
#[derive(Debug, Default, Serialize)]
pub struct Application {
    pub flat_use: FlatUse,
    pub pedestal: Option<f32>,
    pub sample_lights: Vec<PathBuf>,
    pub pedestal_hint: Option<f64>,
}

pub(super) fn recipes(selected: &CalibrationSelection) -> Vec<MasterRecipe> {
    let mut recipes: Vec<MasterRecipe> = Vec::new();
    for (kind, frames) in [
        (CalibrationKind::Bias, &selected.bias),
        (CalibrationKind::DarkFlat, &selected.dark_flat),
        (CalibrationKind::Dark, &selected.dark),
        (CalibrationKind::Flat, &selected.flat),
    ] {
        let masters: Vec<_> = frames
            .iter()
            .filter(|f| {
                f.is_master
                    && !(kind == CalibrationKind::Flat
                        && selected.flat_held_back.contains(&f.frame_uuid))
            })
            .collect();
        let raw: Vec<_> = frames.iter().filter(|f| !f.is_master).cloned().collect();
        let subset = master_subset_report(kind, &raw, &selected.rules);
        let adopt = match selected.rules.external_masters {
            ExternalMasterPolicy::Prefer => masters.first().copied(),
            ExternalMasterPolicy::Fallback if subset.kept.len() < MIN_MASTER_FRAMES => {
                masters.first().copied()
            }
            _ => None,
        };
        let (action, inputs) = if let Some(master) = adopt {
            (RecipeAction::External, vec![master.frame_uuid.clone()])
        } else if subset.kept.len() >= MIN_MASTER_FRAMES {
            (
                RecipeAction::Integrate,
                subset.kept.iter().map(|f| f.frame_uuid.clone()).collect(),
            )
        } else {
            (RecipeAction::Unavailable, Vec::new())
        };
        let dependencies = if matches!(action, RecipeAction::Integrate) {
            recipes
                .iter()
                .filter(|r| {
                    available(&r.action)
                        && match kind {
                            CalibrationKind::Bias => false,
                            CalibrationKind::Dark | CalibrationKind::DarkFlat => {
                                r.kind == CalibrationKind::Bias
                            }
                            CalibrationKind::Flat => {
                                matches!(r.kind, CalibrationKind::Bias | CalibrationKind::DarkFlat)
                            }
                        }
                })
                .map(|r| r.kind)
                .collect()
        } else {
            Vec::new()
        };
        recipes.push(MasterRecipe {
            kind,
            action,
            inputs,
            dependencies,
            dropped: subset
                .dropped
                .iter()
                .map(|f| f.frame_uuid.clone())
                .collect(),
            input_limit: MAX_MASTER_FRAMES,
            requires_source_verification: frames.iter().any(|f| !f.source_verified),
            failure: None,
            exposure_seconds: frames.first().and_then(|f| f.exposure_s),
            defect_suppression: kind == CalibrationKind::Flat,
            flat_star_masking: false,
            dark_level_screening: false,
        });
    }
    recipes
}
fn available(action: &RecipeAction) -> bool {
    matches!(action, RecipeAction::External | RecipeAction::Integrate)
}
fn header_pedestal_hint(light: &Light) -> Option<f64> {
    let offset = light.offset.filter(|value| *value > 0)?;
    let camera = light.camera.as_deref()?.to_ascii_uppercase();
    (camera.contains("ZWO") || camera.contains("ASI")).then_some(offset as f64 * 10.0)
}
fn accepted_pedestal(evidence: &PedestalEvidence, hint: Option<f64>) -> Option<f32> {
    if let Some(pinned) = evidence.pinned {
        return Some(pinned);
    }
    let mut estimates = evidence.estimates.clone();
    if estimates.is_empty() {
        return None;
    }
    estimates.sort_by(f32::total_cmp);
    let pedestal = estimates[estimates.len() / 2];
    let spread = estimates[estimates.len() - 1] - estimates[0];
    if spread > (pedestal * 0.25).max(64.0) || !(0.0..=13_107.0).contains(&pedestal) {
        return None;
    }
    if let Some(hint) = hint {
        let hint = hint as f32;
        if pedestal < hint / 3.0 || pedestal > hint * 3.0 {
            return None;
        }
    }
    Some(pedestal)
}
pub(super) fn finalize(
    sessions: &mut [Session],
    assignments: &[usize],
    lights: &[Light],
    failures: &[MasterFailure],
    evidence: &[PedestalEvidence],
    force: bool,
) -> crate::Result<()> {
    let keys: BTreeSet<_> = sessions.iter().map(|s| s.key.as_str()).collect();
    let mut failed = BTreeSet::new();
    for row in failures {
        if !keys.contains(row.session_key.as_str()) || !failed.insert((&row.session_key, row.kind))
        {
            return Err(invalid(
                "master failures must name unique session/kind recipes",
            ));
        }
    }
    let mut fits = BTreeSet::new();
    for row in evidence {
        if !keys.contains(row.session_key.as_str())
            || !fits.insert(&row.session_key)
            || row.estimates.len() > 3
            || row.estimates.iter().any(|v| !v.is_finite())
            || row.pinned.is_some_and(|v| !v.is_finite() || v < 0.0)
        {
            return Err(invalid(
                "pedestal evidence needs a unique session, at most three finite estimates and a nonnegative finite pinned value",
            ));
        }
    }
    for (index, session) in sessions.iter_mut().enumerate() {
        let mut unavailable = BTreeSet::new();
        for recipe in &mut session.recipes {
            if let Some(row) = failures
                .iter()
                .find(|row| row.session_key == session.key && row.kind == recipe.kind)
            {
                if !available(&recipe.action) {
                    return Err(invalid(
                        "an unavailable recipe cannot report a build failure",
                    ));
                }
                recipe.action = RecipeAction::Failed;
                recipe.failure = Some(row.reason.clone());
            } else if recipe
                .dependencies
                .iter()
                .any(|kind| unavailable.contains(kind))
            {
                recipe.action = RecipeAction::Blocked;
                recipe.failure =
                    Some("skipped because a declared dependency failed to build".into());
            }
            if matches!(recipe.action, RecipeAction::Failed | RecipeAction::Blocked) {
                unavailable.insert(recipe.kind);
            }
        }
        let serves = |kind| {
            session
                .recipes
                .iter()
                .any(|r| r.kind == kind && available(&r.action))
        };
        if !serves(CalibrationKind::Flat) {
            continue;
        }
        if serves(CalibrationKind::Bias) || serves(CalibrationKind::Dark) {
            session.application.flat_use = FlatUse::WithMasters;
            continue;
        }
        if force {
            session.application.flat_use = FlatUse::ForcedWithoutPedestal;
            continue;
        }
        let mut paths: Vec<_> = lights
            .iter()
            .zip(assignments)
            .filter(|(_, a)| **a == index)
            .map(|(l, _)| l.path.clone())
            .collect();
        paths.sort();
        paths.dedup();
        session.application.sample_lights = match paths.len() {
            0 => Vec::new(),
            1 => vec![paths[0].clone()],
            2 => paths.clone(),
            len => vec![
                paths[0].clone(),
                paths[len / 2].clone(),
                paths[len - 1].clone(),
            ],
        };
        session.application.pedestal_hint = lights
            .iter()
            .zip(assignments)
            .find(|(_, a)| **a == index)
            .and_then(|(l, _)| header_pedestal_hint(l));
        if let Some(evidence) = evidence.iter().find(|row| row.session_key == session.key) {
            let pedestal = accepted_pedestal(evidence, session.application.pedestal_hint);
            session.application.pedestal = pedestal;
            session.application.flat_use = if pedestal.is_some() {
                FlatUse::FittedPedestal
            } else {
                FlatUse::Withheld
            };
        } else {
            session.application.flat_use = FlatUse::NeedsPedestalFit;
        }
    }
    Ok(())
}
