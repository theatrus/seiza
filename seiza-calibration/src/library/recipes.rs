//! A light's master recipes, the session they make, and what execution
//! feedback does to them.
//!
//! Extracted from theatrus/psf-guard at 79f5e31: resolve_masters_with_recipe
//! and estimate_flat_pedestal. Apache-2.0.
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
impl PedestalEvidence {
    fn validate(&self) -> crate::Result<()> {
        if self.estimates.len() > 3
            || self.estimates.iter().any(|v| !v.is_finite())
            || self.pinned.is_some_and(|v| !v.is_finite() || v < 0.0)
        {
            return Err(invalid(
                "pedestal evidence needs at most three finite estimates and a nonnegative finite pinned value",
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Application {
    pub flat_use: FlatUse,
    pub pedestal: Option<f32>,
    pub sample_lights: Vec<PathBuf>,
    pub pedestal_hint: Option<f64>,
}

/// Builder options every recipe of a session takes.
#[derive(Clone, Debug, Default)]
pub struct SessionOptions {
    /// Mask stars in master flats (flats from twilight or the night sky).
    pub flat_star_masking: bool,
    pub flat_stability: FlatRetryPolicy,
}

/// The session a light's verified selection builds: each kind's recipe,
/// the complete gate, and an identity that covers what the masters are
/// made from and how. A candidate list longer than an automatic master
/// takes is cut to its nearest [`MAX_MASTER_FRAMES`] once every candidate
/// is verified; an unverified selection is a preview, not an executable
/// recipe, and the host verifies and resolves it again.
pub fn resolve_session(
    selected: &CalibrationSelection,
    quality: &QualityIndex,
    options: &SessionOptions,
) -> Session {
    let mut capped = None;
    for kind in CalibrationKind::ALL {
        let frames = selected.of_kind(kind);
        if !selected.one_set.contains(&kind)
            && frames.len() > MAX_MASTER_FRAMES
            && frames.iter().all(|f| f.source_verified)
        {
            capped
                .get_or_insert_with(|| selected.clone())
                .of_kind_mut(kind)
                .truncate(MAX_MASTER_FRAMES);
        }
    }
    let selected = capped.as_ref().unwrap_or(selected);
    let mut recipes = recipes(selected);
    for recipe in &mut recipes {
        if recipe.action != RecipeAction::Integrate {
            continue;
        }
        recipe.flat_star_masking =
            recipe.kind == CalibrationKind::Flat && options.flat_star_masking;
        // The integrator's own stray-light screen is for darks the library
        // has not graded yet. Once every dark is graded or chosen, the grades
        // have kept the poor ones out, and the screen would only overrule a
        // person who chose to use one anyway.
        recipe.dark_level_screening = matches!(
            recipe.kind,
            CalibrationKind::Dark | CalibrationKind::DarkFlat
        ) && recipe.inputs.iter().any(|id| {
            selected
                .frames()
                .find(|frame| &frame.frame_uuid == id)
                .is_none_or(|frame| !quality.judged(frame))
        });
    }
    let mut stability = options.flat_stability.clone();
    if selected.one_set.contains(&CalibrationKind::Flat) {
        stability.try_next_session = false;
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

/// A session's identity: what each master is made from (its inputs as a
/// set, whichever light's nearest frame anchors them), how it is built, the
/// gate and the settings that chose it. Lights whose masters agree share a
/// session; the first of them stands for the rest. Failure and pedestal
/// feedback refer back to this key, so neither participates in it.
fn session_key(
    selected: &CalibrationSelection,
    recipes: &[MasterRecipe],
    stability: &FlatRetryPolicy,
) -> String {
    let fingerprints: HashMap<_, _> = selected
        .frames()
        .map(|frame| (frame.frame_uuid.as_str(), &frame.source_fingerprint))
        .collect();
    let integrates_flats = recipes
        .iter()
        .any(|r| r.kind == CalibrationKind::Flat && r.action == RecipeAction::Integrate);
    let resolved: Vec<_> = recipes
        .iter()
        .map(|recipe| {
            let mut inputs: Vec<_> = recipe
                .inputs
                .iter()
                .map(|id| (id, fingerprints[id.as_str()]))
                .collect();
            inputs.sort();
            (
                recipe.kind,
                &recipe.action,
                inputs,
                &recipe.dependencies,
                recipe.defect_suppression,
                recipe.flat_star_masking,
                recipe.dark_level_screening,
                recipe.requires_source_verification,
            )
        })
        .collect();
    // Retries and held-back flats act only on flats integrated here.
    let flats = integrates_flats.then(|| {
        (
            stability,
            selected.flat_held_back.iter().collect::<BTreeSet<_>>(),
        )
    });
    let identity = (
        POLICY_ALGORITHM_VERSION,
        resolved,
        selected.match_tolerances(),
        selected.rules.matching_digest(),
        selected.flat_basis.correction(),
        flats,
    );
    hex_digest(&serde_json::to_string(&identity).expect("validated session identity"))
}

pub(super) fn recipes(selected: &CalibrationSelection) -> Vec<MasterRecipe> {
    let mut recipes: Vec<MasterRecipe> = Vec::new();
    for kind in [
        CalibrationKind::Bias,
        CalibrationKind::DarkFlat,
        CalibrationKind::Dark,
        CalibrationKind::Flat,
    ] {
        let frames = selected.of_kind(kind);
        let masters: Vec<_> = frames
            .iter()
            .filter(|f| {
                f.is_master
                    && !(kind == CalibrationKind::Flat
                        && selected.flat_held_back.contains(&f.frame_uuid))
            })
            .collect();
        let raw: Vec<_> = frames.iter().filter(|f| !f.is_master).cloned().collect();
        let subset = if selected.one_set.contains(&kind) {
            Cohort {
                kept: raw.clone(),
                dropped: Vec::new(),
            }
        } else {
            coherent_cohort(kind, &raw, &selected.rules)
        };
        let adopt = match selected.rules.external_masters {
            ExternalMasterPolicy::Prefer => masters.first().copied(),
            ExternalMasterPolicy::Fallback if subset.kept.len() < MIN_MASTER_FRAMES => {
                masters.first().copied()
            }
            _ => None,
        };
        let (action, inputs, exposure_seconds) = if let Some(master) = adopt {
            (
                RecipeAction::External,
                vec![master.frame_uuid.clone()],
                master.exposure_s,
            )
        } else if subset.kept.len() >= MIN_MASTER_FRAMES {
            (
                RecipeAction::Integrate,
                subset.kept.iter().map(|f| f.frame_uuid.clone()).collect(),
                subset.kept.first().and_then(|f| f.exposure_s),
            )
        } else {
            (RecipeAction::Unavailable, Vec::new(), None)
        };
        let dependencies = if action == RecipeAction::Integrate {
            recipes
                .iter()
                .filter(|r| {
                    r.serves()
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
            coherent_inputs: subset.kept.iter().map(|f| f.frame_uuid.clone()).collect(),
            input_limit: if selected.one_set.contains(&kind) {
                raw.len().max(MAX_MASTER_FRAMES)
            } else {
                MAX_MASTER_FRAMES
            },
            requires_source_verification: frames.iter().any(|f| !f.source_verified),
            failure: None,
            exposure_seconds,
            defect_suppression: kind == CalibrationKind::Flat,
            flat_star_masking: false,
            dark_level_screening: false,
        });
    }
    recipes
}

impl MasterRecipe {
    /// Whether this recipe yields a master: one adopted or integrated.
    pub fn serves(&self) -> bool {
        matches!(
            self.action,
            RecipeAction::External | RecipeAction::Integrate
        )
    }
}

impl Session {
    /// The recipe for one kind.
    pub fn recipe(&self, kind: CalibrationKind) -> Option<&MasterRecipe> {
        self.recipes.iter().find(|recipe| recipe.kind == kind)
    }

    /// Record that `kind`'s master failed to build. Every master that
    /// depends on it is blocked rather than built without it: a flat built
    /// without its failed bias bakes the bias pedestal into its
    /// normalization and miscorrects every light, which is worse than no
    /// flat at all.
    pub fn record_failure(
        &mut self,
        kind: CalibrationKind,
        reason: impl Into<String>,
    ) -> crate::Result<()> {
        let recipe = self
            .recipes
            .iter_mut()
            .find(|recipe| recipe.kind == kind)
            .filter(|recipe| recipe.action != RecipeAction::Unavailable)
            .ok_or_else(|| invalid("an unavailable recipe cannot report a build failure"))?;
        recipe.action = RecipeAction::Failed;
        recipe.failure = Some(reason.into());
        let mut unavailable = BTreeSet::new();
        for recipe in &mut self.recipes {
            if recipe.serves()
                && let Some(kind) = recipe
                    .dependencies
                    .iter()
                    .find(|kind| unavailable.contains(*kind))
            {
                recipe.action = RecipeAction::Blocked;
                let kind = match kind {
                    CalibrationKind::Bias => "bias",
                    CalibrationKind::DarkFlat => "dark-flat",
                    CalibrationKind::Dark => "dark",
                    CalibrationKind::Flat => "flat",
                };
                recipe.failure = Some(format!("skipped because the {kind} master failed to build"));
            }
            if matches!(recipe.action, RecipeAction::Failed | RecipeAction::Blocked) {
                unavailable.insert(recipe.kind);
            }
        }
        Ok(())
    }

    fn serves(&self, kind: CalibrationKind) -> bool {
        self.recipe(kind).is_some_and(MasterRecipe::serves)
    }

    /// How the flat applies to `lights`, the session's own, once every
    /// master has built or failed. A flat is divided only into a light whose
    /// pedestal is removed: with neither a bias nor a dark master the
    /// division amplifies the light's pedestal by 1/vignette at the frame
    /// edges, so the session needs a pedestal fitted from sample lights
    /// first, unless `force` applies the flat regardless.
    pub fn plan_flat_use(&mut self, lights: &[&Light], force: bool) {
        self.application = Application::default();
        if !self.serves(CalibrationKind::Flat) {
            return;
        }
        if self.serves(CalibrationKind::Bias) || self.serves(CalibrationKind::Dark) {
            self.application.flat_use = FlatUse::WithMasters;
            return;
        }
        if force {
            self.application.flat_use = FlatUse::ForcedWithoutPedestal;
            return;
        }
        let mut paths: Vec<_> = lights.iter().map(|l| l.path.clone()).collect();
        paths.sort();
        paths.dedup();
        self.application.sample_lights = match paths.len() {
            0..=2 => paths,
            len => vec![
                paths[0].clone(),
                paths[len / 2].clone(),
                paths[len - 1].clone(),
            ],
        };
        self.application.pedestal_hint = lights.first().and_then(|l| header_pedestal_hint(l));
        self.application.flat_use = FlatUse::NeedsPedestalFit;
    }

    /// Accept or refuse the pedestal fitted from the sample lights
    /// [`Session::plan_flat_use`] asked for. A pinned value reproduces a
    /// previous stack. Fits that disagree, or a pedestal far from the
    /// camera's recorded offset, withhold the flat.
    pub fn accept_pedestal(&mut self, evidence: &PedestalEvidence) -> crate::Result<()> {
        evidence.validate()?;
        if self.application.flat_use != FlatUse::NeedsPedestalFit {
            return Ok(());
        }
        let pedestal = accepted_pedestal(evidence, self.application.pedestal_hint);
        self.application.pedestal = pedestal;
        self.application.flat_use = if pedestal.is_some() {
            FlatUse::FittedPedestal
        } else {
            FlatUse::Withheld
        };
        Ok(())
    }
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

/// Apply a request's execution feedback to its sessions: failures, then
/// each session's flat use for its own lights, then pedestal fits.
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
        row.validate()?;
        if !keys.contains(row.session_key.as_str()) || !fits.insert(&row.session_key) {
            return Err(invalid("pedestal evidence needs a unique session"));
        }
    }
    for (index, session) in sessions.iter_mut().enumerate() {
        // In build order, so a failure blocks what was built after it.
        for kind in [
            CalibrationKind::Bias,
            CalibrationKind::DarkFlat,
            CalibrationKind::Dark,
            CalibrationKind::Flat,
        ] {
            if let Some(row) = failures
                .iter()
                .find(|row| row.session_key == session.key && row.kind == kind)
            {
                session.record_failure(kind, row.reason.clone())?;
            }
        }
        let own: Vec<&Light> = lights
            .iter()
            .zip(assignments)
            .filter(|(_, a)| **a == index)
            .map(|(l, _)| l)
            .collect();
        session.plan_flat_use(&own, force);
        if let Some(row) = evidence.iter().find(|row| row.session_key == session.key) {
            session.accept_pedestal(row)?;
        }
    }
    Ok(())
}
