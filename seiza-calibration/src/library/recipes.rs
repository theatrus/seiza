//! A light's master recipes, the session they make, and what execution
//! feedback does to them.
//!
//! Extracted from theatrus/psf-guard at 79f5e31: resolve_masters_with_recipe
//! and estimate_flat_pedestal. Apache-2.0.
use super::*;

/// How a session's flat master applies to its lights.
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlatUse {
    /// No flat master serves.
    #[default]
    Unavailable,
    /// After a bias or dark master removed the pedestal.
    WithMasters,
    /// With neither master: a pedestal must be fitted from sample lights
    /// first ([`Application::sample_lights`]).
    NeedsPedestalFit,
    /// After subtracting the pedestal fitted from the lights.
    FittedPedestal,
    /// With neither master nor a pedestal, because the host forced it:
    /// vignetted edges may brighten.
    ForcedWithoutPedestal,
    /// Not at all: no reliable pedestal could be fitted.
    Withheld,
}
words!(FlatUse {
    Unavailable => "unavailable",
    WithMasters => "with_masters",
    NeedsPedestalFit => "needs_pedestal_fit",
    FittedPedestal => "fitted_pedestal",
    ForcedWithoutPedestal => "forced_without_pedestal",
    Withheld => "withheld",
});
impl FlatUse {
    /// How a flat applies to a light, before any pedestal fit: only into a
    /// light whose pedestal a bias or dark master removed. With neither,
    /// the division amplifies the pedestal by 1/vignette at the frame
    /// edges, the vignette comes out inverted, and that is worse than no
    /// flat at all; so the pedestal is fitted from the lights first, unless
    /// `without_masters` forces the flat regardless.
    pub fn decide(flat: bool, pedestal_removed: bool, without_masters: FlatWithoutMasters) -> Self {
        match (flat, pedestal_removed, without_masters) {
            (false, _, _) => Self::Unavailable,
            (true, true, _) => Self::WithMasters,
            (true, false, FlatWithoutMasters::Force) => Self::ForcedWithoutPedestal,
            (true, false, FlatWithoutMasters::FitPedestal) => Self::NeedsPedestalFit,
        }
    }
    /// How a flat that needed a pedestal fit applies, given the pedestal
    /// accepted: after subtracting it, or not at all.
    pub fn after_fit(pedestal: Option<f32>) -> Self {
        if pedestal.is_some() {
            Self::FittedPedestal
        } else {
            Self::Withheld
        }
    }
}

/// What a flat does for a light no bias or dark master removes the
/// pedestal from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlatWithoutMasters {
    /// Fit the pedestal from the lights, and withhold the flat when no
    /// reliable one fits.
    #[default]
    FitPedestal,
    /// Apply the flat anyway: the host forced calibration on.
    Force,
}
words!(FlatWithoutMasters {
    FitPedestal => "fit_pedestal",
    Force => "force",
});

/// How a session's flat applies, once its masters are built.
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Application {
    pub flat_use: FlatUse,
    /// The pedestal subtracted before the flat, in ADU, for
    /// [`FlatUse::FittedPedestal`].
    pub pedestal: Option<f32>,
    /// The lights to fit the pedestal from, for
    /// [`FlatUse::NeedsPedestalFit`]: [`pedestal_sample_lights`].
    pub sample_lights: Vec<PathBuf>,
    /// The pedestal the camera's recorded offset predicts: [`pedestal_hint`].
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
    let capped = selected.needs_cap().then(|| {
        let mut capped = selected.clone();
        capped.cap_verified();
        capped
    });
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

/// The external master a light takes for `kind` from its candidates,
/// nearest first, used as it is and never integrated with raw frames: the
/// nearest one under `prefer`, the nearest one when the raw frames cannot
/// make a master under `fallback`, none under `ignore`. A flat master from
/// the far side of a measured change (`held_back`) never stands in.
pub fn adopted_master<'a>(
    kind: CalibrationKind,
    frames: &'a [CalibrationFrame],
    held_back: &[String],
    rules: &Rules,
) -> Option<&'a CalibrationFrame> {
    adopted_with(kind, frames, held_back, rules.external_masters, || {
        let raw: Vec<CalibrationFrame> = frames.iter().filter(|f| !f.is_master).cloned().collect();
        coherent_cohort(kind, &raw, rules).kept.len()
    })
}

fn adopted_with<'a>(
    kind: CalibrationKind,
    frames: &'a [CalibrationFrame],
    held_back: &[String],
    policy: ExternalMasterPolicy,
    coherent_raw: impl FnOnce() -> usize,
) -> Option<&'a CalibrationFrame> {
    let mut masters = frames.iter().filter(|f| {
        f.is_master && !(kind == CalibrationKind::Flat && held_back.contains(&f.frame_uuid))
    });
    match policy {
        ExternalMasterPolicy::Prefer => masters.next(),
        ExternalMasterPolicy::Fallback if coherent_raw() < MIN_MASTER_FRAMES => masters.next(),
        _ => None,
    }
}

/// Whether the integrator screens `inputs` for stray light by their level.
/// The screen is for darks the library has not graded yet: once every one
/// is graded or chosen, the grades have kept the poor ones out, and the
/// screen would only overrule a person who chose to use one anyway.
pub fn dark_level_screening<'a>(
    kind: CalibrationKind,
    inputs: impl IntoIterator<Item = &'a CalibrationFrame>,
    quality: &QualityIndex,
) -> bool {
    matches!(kind, CalibrationKind::Dark | CalibrationKind::DarkFlat)
        && inputs.into_iter().any(|frame| !quality.judged(frame))
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
        let raw: Vec<_> = frames.iter().filter(|f| !f.is_master).cloned().collect();
        let subset = if selected.one_set.contains(&kind) {
            // One supplied set serves every light in capture order, so its
            // first frame does not depend on which light or host asks.
            let mut kept = raw.clone();
            kept.sort_by(|a, b| {
                (a.captured_at, a.id, &a.frame_uuid).cmp(&(b.captured_at, b.id, &b.frame_uuid))
            });
            Cohort {
                kept,
                dropped: Vec::new(),
            }
        } else {
            coherent_cohort(kind, &raw, &selected.rules)
        };
        let adopt = adopted_with(
            kind,
            frames,
            &selected.flat_held_back,
            selected.rules.external_masters,
            || subset.kept.len(),
        );
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
                .filter(|r| r.serves() && kind.dependencies().contains(&r.kind))
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
                recipe.failure = Some(format!(
                    "skipped because the {} master failed to build",
                    kind.label()
                ));
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
    pub fn plan_flat_use<'a>(
        &mut self,
        lights: impl IntoIterator<Item = &'a Light>,
        without_masters: FlatWithoutMasters,
    ) {
        self.application = Application {
            flat_use: FlatUse::decide(
                self.serves(CalibrationKind::Flat),
                self.serves(CalibrationKind::Bias) || self.serves(CalibrationKind::Dark),
                without_masters,
            ),
            ..Application::default()
        };
        if self.application.flat_use == FlatUse::NeedsPedestalFit {
            let lights: Vec<&Light> = lights.into_iter().collect();
            let paths: Vec<_> = lights.iter().map(|l| l.path.clone()).collect();
            self.application.sample_lights = pedestal_sample_lights(&paths);
            self.application.pedestal_hint = lights.first().and_then(|l| pedestal_hint(l));
        }
    }

    /// Accept or refuse the pedestal fitted from the sample lights
    /// [`Session::plan_flat_use`] asked for: `estimates`, one per light
    /// that fitted, at most three. A `pinned` value reproduces a previous
    /// stack and wins. Fits that disagree, or a pedestal far from the
    /// camera's recorded offset, withhold the flat
    /// ([`pedestal_from_fits`]).
    pub fn accept_pedestal(&mut self, estimates: &[f32], pinned: Option<f32>) -> crate::Result<()> {
        if estimates.len() > 3
            || estimates.iter().any(|v| !v.is_finite())
            || pinned.is_some_and(|v| !v.is_finite() || v < 0.0)
        {
            return Err(invalid(
                "pedestal evidence needs at most three finite estimates and a nonnegative finite pinned value",
            ));
        }
        if self.application.flat_use != FlatUse::NeedsPedestalFit {
            return Ok(());
        }
        let pedestal =
            pinned.or_else(|| pedestal_from_fits(estimates, self.application.pedestal_hint).ok());
        self.application.pedestal = pedestal;
        self.application.flat_use = FlatUse::after_fit(pedestal);
        Ok(())
    }
}

/// The pedestal a camera's recorded offset predicts, in ADU, where its
/// driver is known to record it: a ZWO camera's offset times ten.
pub fn pedestal_hint(light: &Light) -> Option<f64> {
    let offset = light.offset.filter(|value| *value > 0)?;
    let camera = light.camera.as_deref()?.to_ascii_uppercase();
    (camera.contains("ZWO") || camera.contains("ASI")).then_some(offset as f64 * 10.0)
}

/// The lights a flat pedestal is fitted from: by sorted path, so the
/// estimate does not depend on the caller's light order, the first, middle
/// and last.
pub fn pedestal_sample_lights(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut sorted = paths.to_vec();
    sorted.sort();
    sorted.dedup();
    match sorted.len() {
        0..=2 => sorted,
        len => vec![
            sorted[0].clone(),
            sorted[len / 2].clone(),
            sorted[len - 1].clone(),
        ],
    }
}

/// The most fitted pedestal a flat is applied after, in ADU: 20% of the
/// 16-bit scale.
pub const MAX_PEDESTAL_ADU: f32 = 13_107.0;

/// Why fitted pedestals were refused.
#[derive(Clone, Debug, PartialEq)]
pub enum PedestalRefusal {
    /// No light produced a usable fit.
    NoFit,
    /// The per-light estimates disagree.
    Disagree(Vec<f32>),
    /// Outside 0 to [`MAX_PEDESTAL_ADU`].
    OutOfRange(f32),
    /// More than 3x either way from the camera's recorded offset: more
    /// likely a sky gradient than a pedestal.
    ContradictsOffset { pedestal: f32, hint: f64 },
}
impl std::error::Error for PedestalRefusal {}
impl std::fmt::Display for PedestalRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoFit => write!(f, "no light produced a usable fit"),
            Self::Disagree(estimates) => {
                write!(f, "per-light estimates disagree ({estimates:?} ADU)")
            }
            Self::OutOfRange(pedestal) => {
                write!(f, "{pedestal:.1} ADU is outside the sane range")
            }
            Self::ContradictsOffset { pedestal, hint } => write!(
                f,
                "{pedestal:.1} ADU contradicts the camera's recorded offset (~{hint:.0} ADU)"
            ),
        }
    }
}

/// The pedestal under a session's lights from the fits of its sample
/// lights: their median, when they agree, lie in range, and do not
/// contradict the camera's recorded offset.
pub fn pedestal_from_fits(estimates: &[f32], hint: Option<f64>) -> Result<f32, PedestalRefusal> {
    let mut estimates = estimates.to_vec();
    if estimates.is_empty() {
        return Err(PedestalRefusal::NoFit);
    }
    estimates.sort_by(f32::total_cmp);
    let pedestal = estimates[estimates.len() / 2];
    let spread = estimates[estimates.len() - 1] - estimates[0];
    if spread > (pedestal * 0.25).max(64.0) {
        return Err(PedestalRefusal::Disagree(estimates));
    }
    if !(0.0..=MAX_PEDESTAL_ADU).contains(&pedestal) {
        return Err(PedestalRefusal::OutOfRange(pedestal));
    }
    if let Some(hint) = hint {
        let h = hint as f32;
        if pedestal < h / 3.0 || pedestal > h * 3.0 {
            return Err(PedestalRefusal::ContradictsOffset { pedestal, hint });
        }
    }
    Ok(pedestal)
}
