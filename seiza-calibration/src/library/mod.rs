//! PSF Guard's library grading and selection policy, detached from its catalog.
//! Hosts supply immutable rows and current measurements, then execute recipes
//! through the stacker's master builders. No files or database are opened here.
mod flat_notes;
mod grading;
mod recipes;
mod rules;
pub use flat_notes::{
    AlikeRangeNote, ChoiceNote, EventNote, FlatBasis, FlatCorrection, FlatFamily, GroupNote,
};
pub use recipes::{Application, FlatUse, MasterFailure, PedestalEvidence};
use recipes::{finalize, recipes};
pub use rules::{ChoiceOption, Effect, FieldKind, FieldSpec, Group, RuleNote, RuleSource};
mod selection;
use crate::{
    BiasScreening, DarkMeasurement, DarkScreening, Finding, FindingCode, FlatMeasurement,
    FrameSignature, FrameVerdict, Grade, MatchTolerances,
};
use flat_notes as flat_groups;
use rules::{RuleBook, Rules};
use selection::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
const MIN_MASTER_FRAMES: usize = 2;
const MAX_MASTER_FRAMES: usize = 64;
const POLICY_ALGORITHM_VERSION: u32 = 2;
const DEFAULT_DARK_REACH_DAYS: f64 = 183.0;
const DEFAULT_COMPLETE_DARK_FRAMES: usize = 10;
const DEFAULT_FLAT_MAX_AGE_DAYS: f64 = 60.0;
const DARK_SESSION_SECONDS: u64 = 86400;
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationKind {
    #[default]
    Bias,
    Dark,
    DarkFlat,
    Flat,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidDirection {
    Forward,
    Backward,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DarkChoice {
    #[default]
    Complete,
    Nearest,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExternalMasterPolicy {
    #[default]
    Prefer,
    Fallback,
    Ignore,
}
impl ExternalMasterPolicy {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "prefer" => Some(Self::Prefer),
            "fallback" => Some(Self::Fallback),
            "ignore" => Some(Self::Ignore),
            _ => None,
        }
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CalibrationFrame {
    pub id: i64,
    pub frame_uuid: String,
    pub rig_uuid: String,
    pub kind: CalibrationKind,
    pub source_path: PathBuf,
    pub source_fingerprint: String,
    pub captured_at: Option<i64>,
    pub telescope: Option<String>,
    pub camera: Option<String>,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub channels: Option<i64>,
    pub binning_x: Option<i64>,
    pub binning_y: Option<i64>,
    pub gain: Option<i64>,
    pub offset: Option<i64>,
    pub readout_mode: Option<i64>,
    /// The readout mode's name when the header spelled it that way; see
    /// A named readout mode, matched independently of its numeric index.
    pub readout_mode_name: Option<String>,
    /// An integrated master from other software rather than a raw frame.
    /// Matched leniently (its producer strips most settings) and used as a
    /// master as-is instead of being integrated.
    pub is_master: bool,
    pub bayer_pattern: Option<String>,
    pub exposure_s: Option<f64>,
    pub camera_temp: Option<f64>,
    pub filter: Option<String>,
    pub focal_length_mm: Option<f64>,
    /// Rotator angle in degrees at capture, when the rig recorded one.
    pub rotation: Option<f64>,
    /// User-set validity boundary; absent means both directions.
    pub valid_direction: Option<ValidDirection>,
    /// Why this frame stays out of masters and exports: a person kept it
    /// out ("by hand"), or its grade is poor and this is the main finding.
    /// A frame not graded yet keeps the stray-light mark of the older dark
    /// check. Absent when the frame serves.
    pub kept_out: Option<String>,
    /// Set after the current file (or a basename-remapped file) has been
    /// checked against this catalog row's hard settings.
    pub source_verified: bool,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Light {
    pub path: PathBuf,
    pub rig_uuid: String,
    pub camera: Option<String>,
    pub telescope: Option<String>,
    pub bayer_pattern: Option<String>,
    pub filter: Option<String>,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub channels: Option<i64>,
    pub binning_x: Option<i64>,
    pub binning_y: Option<i64>,
    pub gain: Option<i64>,
    pub offset: Option<i64>,
    pub readout_mode: Option<i64>,
    pub readout_mode_name: Option<String>,
    pub focal_length_mm: Option<f64>,
    pub rotator_position: Option<f64>,
    pub exposure_s: Option<f64>,
    pub camera_temp: Option<f64>,
    pub timestamp: Option<i64>,
}
type FrameMeta = Light;

#[derive(Clone, Debug, Default)]
struct CalibrationSelection {
    bias: Vec<CalibrationFrame>,
    dark: Vec<CalibrationFrame>,
    dark_flat: Vec<CalibrationFrame>,
    flat: Vec<CalibrationFrame>,
    kept_out: Vec<CalibrationFrame>,
    library: LibraryKinds,
    rules: Arc<Rules>,
    flat_basis: flat_groups::FlatBasis,
    flat_held_back: Vec<String>,
    flat_family: Option<flat_groups::FlatFamily>,
    flat_choices: Vec<flat_groups::ChoiceNote>,
}
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct LibraryKinds {
    pub bias: bool,
    pub dark: bool,
    pub flat: bool,
    pub unused_dark: bool,
}
impl LibraryKinds {
    fn note(&mut self, kind: CalibrationKind) {
        match kind {
            CalibrationKind::Bias => self.bias = true,
            CalibrationKind::Dark => self.dark = true,
            CalibrationKind::Flat => self.flat = true,
            CalibrationKind::DarkFlat => {}
        }
    }
}
impl CalibrationSelection {
    /// The tolerances a stack holds this light to against its masters, its
    /// gate: its rig's, with the rotation widened just enough for flats
    /// taken from the nearest other angle (see [`flat_groups`]). Its
    /// calibration session carries it.
    fn match_tolerances(&self) -> crate::MatchTolerances {
        let mut tolerances = self.rules.tolerances();
        if let Some(widened) = self.flat_basis.rotation_tolerance() {
            tolerances.rotation_deg = tolerances.rotation_deg.max(widened);
        }
        if let Some(family) = &self.flat_family {
            tolerances.light_rotation = family.light_rotation;
        }
        tolerances
    }
}
struct LibrarySnapshot {
    frames: Vec<CalibrationFrame>,
    flat_book: Arc<crate::flat_response::groups::Book>,
    available_responses: BTreeSet<String>,
}
impl LibrarySnapshot {
    fn can_correct(&self, before: &str, after: &str) -> bool {
        self.available_responses.contains(before) && self.available_responses.contains(after)
    }
}
struct MasterSubset {
    kept: Vec<CalibrationFrame>,
    dropped: Vec<CalibrationFrame>,
}
struct Signature {
    seiza: FrameSignature,
    readout_mode_name: Option<String>,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityOverride {
    Use,
    Exclude,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StoredMeasurement {
    /// A bias, dark or dark-flat.
    Dark(DarkMeasurement),
    /// A flat.
    Flat(FlatRecord),
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlatRecord {
    #[serde(flatten)]
    pub measurement: FlatMeasurement,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optics: Option<crate::dust::Optics>,
}
impl std::ops::Deref for FlatRecord {
    type Target = FlatMeasurement;
    fn deref(&self) -> &FlatMeasurement {
        &self.measurement
    }
}
struct Measured {
    frame: CalibrationFrame,
    signature: Signature,
    measurement: StoredMeasurement,
}
#[derive(Clone, Debug, Serialize)]
pub struct Judged {
    pub grade: Grade,
    pub code: Option<FindingCode>,
    pub reason: Option<String>,
    pub findings: Vec<Finding>,
    pub stray_light: Option<String>,
}
#[derive(PartialEq, Eq)]
struct FlatSessionKey {
    rig: String,
    night: String,
    filter: Option<String>,
    width: usize,
    height: usize,
}
struct Nights {
    shifts: BTreeMap<(String, String), i64>,
}
impl Nights {
    fn night_of_frame(&self, rig: &str, frame: &str, at: i64) -> String {
        let shift = self
            .shifts
            .get(&(rig.into(), frame.into()))
            .copied()
            .unwrap_or(43200);
        ((at as i128 - shift as i128).div_euclid(86400)).to_string()
    }
}
fn session_key(selected: &CalibrationSelection, recipes: &[MasterRecipe]) -> String {
    let frames: HashMap<_, _> = selected
        .bias
        .iter()
        .chain(&selected.dark)
        .chain(&selected.dark_flat)
        .chain(&selected.flat)
        .map(|frame| (frame.frame_uuid.as_str(), frame))
        .collect();
    let resolved: Vec<_> = recipes
        .iter()
        .map(|recipe| {
            // Input order is meaningful: the first input anchors the master's
            // metadata/exposure. Unused candidates and exclusions are not inputs.
            let inputs: Vec<_> = recipe
                .inputs
                .iter()
                .map(|id| (id, &frames[id.as_str()].source_fingerprint))
                .collect();
            (
                recipe.kind,
                &recipe.action,
                inputs,
                &recipe.dependencies,
                recipe.exposure_seconds,
                recipe.defect_suppression,
                recipe.flat_star_masking,
                recipe.dark_level_screening,
                recipe.requires_source_verification,
            )
        })
        .collect();
    // Resolve adoption, builder options and the complete gate before deduping.
    // Failure and pedestal evidence refer back to this base key, so neither
    // finalization outcome participates in it.
    let identity = (
        POLICY_ALGORITHM_VERSION,
        resolved,
        selected.match_tolerances(),
        selected.rules.matching_digest(),
        selected.flat_basis.correction(),
    );
    hex_digest(&serde_json::to_string(&identity).expect("validated session identity"))
}
fn hex_digest(value: &str) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(value.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// One measured raw row. A stale algorithm/fingerprint is not regraded.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct QualityRecord {
    pub frame_uuid: String,
    pub source_fingerprint: Option<String>,
    pub algorithm_version: Option<u32>,
    pub measurement: Option<StoredMeasurement>,
    pub measure_error: Option<String>,
    pub grade: Option<String>,
    pub reason: Option<String>,
    pub stray_light: Option<String>,
    pub override_choice: Option<QualityOverride>,
}
/// Setting identifiers and values remain opaque in both bindings.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Setting {
    pub key: String,
    pub value: serde_json::Value,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RigSettings {
    pub rig_uuid: String,
    pub name: Option<String>,
    pub settings: Vec<Setting>,
}
/// Host-resolved local-noon shift, unwrapped (12h minus UTC offset).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NightShift {
    pub rig_uuid: String,
    pub frame_uuid: String,
    pub seconds: i64,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Request {
    pub frames: Vec<CalibrationFrame>,
    pub lights: Vec<Light>,
    pub quality: Vec<QualityRecord>,
    pub global_settings: Vec<Setting>,
    pub rigs: Vec<RigSettings>,
    pub grouping: crate::flat_response::groups::Request,
    pub available_responses: Vec<String>,
    pub night_shifts: Vec<NightShift>,
    pub regrade: bool,
    pub master_failures: Vec<MasterFailure>,
    pub pedestal_evidence: Vec<PedestalEvidence>,
    pub force_flat_without_pedestal: bool,
    pub flat_star_masking: bool,
}
#[derive(Debug, Serialize)]
pub struct Selection {
    pub bias: Vec<String>,
    pub dark: Vec<String>,
    pub dark_flat: Vec<String>,
    pub flat: Vec<String>,
    pub kept_out: Vec<String>,
    pub library: LibraryKinds,
    pub tolerances: MatchTolerances,
    pub flat_basis: flat_groups::FlatBasis,
    pub flat_held_back: Vec<String>,
    pub flat_family: Option<flat_groups::FlatFamily>,
    pub flat_choices: Vec<flat_groups::ChoiceNote>,
    pub rule_notes: Vec<rules::RuleNote>,
}
#[derive(Debug, Serialize)]
pub struct FrameGrade {
    pub frame_uuid: String,
    pub judged: Option<Judged>,
    pub kept_out: Option<String>,
    pub measurement_state: String,
    pub measure_error: Option<String>,
}
#[derive(Debug, Serialize)]
pub struct Snapshot {
    pub algorithm_version: u32,
    pub selections: Vec<Selection>,
    pub grades: Vec<FrameGrade>,
    pub sessions: Vec<Session>,
    pub assignments: Vec<usize>,
    pub settings_digest: Option<String>,
    pub settings_schema: Vec<rules::FieldSpec>,
}
#[derive(Debug, Serialize)]
pub struct Session {
    pub key: String,
    pub tolerances: MatchTolerances,
    pub recipes: Vec<MasterRecipe>,
    pub application: Application,
}
#[derive(Debug, Serialize)]
pub struct MasterRecipe {
    pub kind: CalibrationKind,
    pub action: RecipeAction,
    pub inputs: Vec<String>,
    pub dependencies: Vec<CalibrationKind>,
    pub dropped: Vec<String>,
    pub input_limit: usize,
    pub requires_source_verification: bool,
    pub failure: Option<String>,
    pub exposure_seconds: Option<f64>,
    pub defect_suppression: bool,
    pub flat_star_masking: bool,
    pub dark_level_screening: bool,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecipeAction {
    External,
    Integrate,
    Unavailable,
    Failed,
    Blocked,
}

fn invalid(message: impl Into<String>) -> crate::Error {
    crate::Error::CalibrationLibrary(message.into())
}

/// Run PSF Guard's grading and selection against an immutable library snapshot.
/// Selections and assignments retain light order; sessions reuse identical recipes.
/// Recipes still require host source verification and the native pixel builders.
pub fn evaluate(mut request: Request) -> crate::Result<Snapshot> {
    if !request.grouping.lights.is_empty() {
        return Err(invalid(
            "library lights belong in lights, not grouping.lights",
        ));
    }
    let mut ids = BTreeSet::new();
    for frame in &request.frames {
        if frame.frame_uuid.is_empty() || !ids.insert(&frame.frame_uuid) {
            return Err(invalid(
                "calibration frame UUIDs must be nonempty and unique",
            ));
        }
        if frame.source_verified
            && (frame.source_path.as_os_str().is_empty() || frame.source_fingerprint.is_empty())
        {
            return Err(invalid("verified sources need a path and fingerprint"));
        }
        validate_values(&frame_signature(frame).seiza)?;
    }
    for light in &request.lights {
        validate_values(&light_signature(light).seiza)?;
    }
    let values = |settings: Vec<Setting>| -> crate::Result<rules::RuleValues> {
        let mut values = rules::RuleValues::new();
        for setting in settings {
            if values.insert(setting.key, setting.value).is_some() {
                return Err(invalid("duplicate setting key"));
            }
        }
        Ok(values)
    };
    let mut rigs = BTreeMap::new();
    for rig in request.rigs {
        if rigs
            .insert(
                rig.rig_uuid,
                rules::RigLayer {
                    name: rig.name,
                    values: values(rig.settings)?,
                },
            )
            .is_some()
        {
            return Err(invalid("duplicate rig settings"));
        }
    }
    let rules = RuleBook::new(values(request.global_settings)?, rigs);
    // The same resolved rules grade, group and select. A separate grouping
    // request cannot silently give its rig another threshold.
    let group_rigs: BTreeSet<_> = request
        .grouping
        .responses
        .iter()
        .map(|r| r.rig_uuid.clone())
        .collect();
    request.grouping.limits = group_rigs
        .into_iter()
        .map(|rig_uuid| {
            let rule = rules.for_rig(&rig_uuid);
            crate::flat_response::groups::RigLimits {
                rig_uuid,
                limits: flat_groups::limits_of(&rule),
            }
        })
        .collect();
    let mut shifts: BTreeMap<_, _> = request
        .grouping
        .night_boundaries
        .iter()
        .map(|s| ((s.rig_uuid.clone(), s.frame_uuid.clone()), s.seconds))
        .collect();
    let mut explicit = BTreeSet::new();
    for shift in request.night_shifts {
        if !(-86400..=129600).contains(&shift.seconds)
            || !explicit.insert((shift.rig_uuid.clone(), shift.frame_uuid.clone()))
        {
            return Err(invalid(
                "night shifts must be unique and within one day of noon UTC",
            ));
        }
        shifts.insert((shift.rig_uuid, shift.frame_uuid), shift.seconds);
    }
    for frame in &request.frames {
        if let Some(offset) = rules.for_rig(&frame.rig_uuid).night_offset_seconds {
            shifts.insert(
                (frame.rig_uuid.clone(), frame.frame_uuid.clone()),
                43200 - offset,
            );
        }
    }
    request.grouping.night_boundaries = shifts
        .iter()
        .map(
            |((rig_uuid, frame_uuid), seconds)| crate::flat_response::groups::NightBoundary {
                rig_uuid: rig_uuid.clone(),
                frame_uuid: frame_uuid.clone(),
                seconds: *seconds,
            },
        )
        .collect();
    let nights = Nights { shifts };
    let mut quality = BTreeMap::new();
    for row in request.quality {
        if !ids.contains(&row.frame_uuid) || quality.insert(row.frame_uuid.clone(), row).is_some() {
            return Err(invalid("quality rows must name unique calibration frames"));
        }
    }
    // SQL load_measured orders oldest first, id breaking equal capture times.
    let mut oldest = request.frames.clone();
    oldest.sort_by(|a, b| {
        (a.captured_at, a.id, &a.frame_uuid).cmp(&(b.captured_at, b.id, &b.frame_uuid))
    });
    let current = |frame: &CalibrationFrame, row: &QualityRecord| {
        !frame.is_master
            && row.algorithm_version
                == Some(if frame.kind == CalibrationKind::Flat {
                    crate::FLAT_QUALITY_ALGORITHM_VERSION
                } else {
                    crate::FRAME_QUALITY_ALGORITHM_VERSION
                })
            && row.source_fingerprint.as_deref() == Some(&frame.source_fingerprint)
    };
    let measured: Vec<_> = oldest
        .iter()
        .filter_map(|frame| {
            let row = quality.get(&frame.frame_uuid)?;
            if !current(frame, row) {
                return None;
            }
            let measurement = row.measurement.clone()?;
            if matches!(&measurement, StoredMeasurement::Flat(_))
                != (frame.kind == CalibrationKind::Flat)
            {
                return None;
            }
            Some(Measured {
                frame: frame.clone(),
                signature: frame_signature(frame),
                measurement,
            })
        })
        .collect();
    let mut judged = HashMap::new();
    if request.regrade {
        grading::judge_darks(&measured, &mut judged, &rules);
        grading::judge_flats(&measured, &nights, &mut judged, &rules);
    }
    let mut grades = Vec::new();
    for frame in &mut request.frames {
        let row = quality.get(&frame.frame_uuid);
        let grade = judged.get(&frame.frame_uuid);
        let grade_name = grade
            .map(|g| g.grade.as_str())
            .or_else(|| row.and_then(|r| r.grade.as_deref()));
        let reason = grade
            .and_then(|g| g.reason.clone())
            .or_else(|| row.and_then(|r| r.reason.clone()));
        let stray = grade
            .and_then(|g| g.stray_light.clone())
            .or_else(|| row.and_then(|r| r.stray_light.clone()));
        // A supplied legacy kept-out reason represents the catalog's old
        // stray-light mark when no quality row exists.
        frame.kept_out = grading::kept_out_reason(
            stray.or_else(|| frame.kept_out.clone()),
            grade_name,
            reason,
            row.and_then(|r| r.override_choice),
        );
        let state = if frame.is_master {
            "external"
        } else {
            match row {
                None => "unmeasured",
                Some(r) if !current(frame, r) => "stale",
                Some(r) if r.measurement.is_some() => "current",
                Some(r) if r.measure_error.is_some() => "failed",
                _ => "unmeasured",
            }
        };
        grades.push(FrameGrade {
            frame_uuid: frame.frame_uuid.clone(),
            judged: grade.cloned(),
            kept_out: frame.kept_out.clone(),
            measurement_state: state.into(),
            measure_error: row.and_then(|r| r.measure_error.clone()),
        });
    }
    request.frames.sort_by(|a, b| {
        b.captured_at
            .cmp(&a.captured_at)
            .then(b.id.cmp(&a.id))
            .then(b.frame_uuid.cmp(&a.frame_uuid))
    });
    let library = LibrarySnapshot {
        frames: request.frames,
        flat_book: Arc::new(crate::flat_response::groups::build_book(request.grouping)?),
        available_responses: request.available_responses.into_iter().collect(),
    };
    let mut selections = Vec::new();
    let mut sessions = Vec::new();
    let mut assignments = Vec::new();
    let mut keys = BTreeMap::new();
    for light in &request.lights {
        let rule = rules.for_light(light);
        let choices = library.light_choices(light, &rule);
        let mut selected = library.select_choosing(light, rule, choices);
        // PSF Guard verifies/remaps all candidates and caps each kind before
        // coherence and external-master adoption. An unverified preview is
        // not an executable recipe: the host must verify and replan first.
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
        let mut recipes = recipes(&selected);
        for recipe in &mut recipes {
            recipe.flat_star_masking =
                recipe.kind == CalibrationKind::Flat && request.flat_star_masking;
            recipe.dark_level_screening = matches!(
                recipe.kind,
                CalibrationKind::Dark | CalibrationKind::DarkFlat
            ) && recipe.inputs.iter().any(|id| {
                !judged.contains_key(id)
                    && quality.get(id).is_none_or(|q| {
                        q.override_choice.is_none()
                            && !(q.grade.is_some()
                                && selected_frame_current(id, q, &library.frames))
                    })
            });
        }
        let key = session_key(&selected, &recipes);
        let index = match keys.get(&key) {
            Some(index) => *index,
            None => {
                let index = sessions.len();
                keys.insert(key.clone(), index);
                sessions.push(Session {
                    key,
                    tolerances: selected.match_tolerances(),
                    recipes,
                    application: Application::default(),
                });
                index
            }
        };
        assignments.push(index);
        let names =
            |frames: &[CalibrationFrame]| frames.iter().map(|f| f.frame_uuid.clone()).collect();
        selections.push(Selection {
            bias: names(&selected.bias),
            dark: names(&selected.dark),
            dark_flat: names(&selected.dark_flat),
            flat: names(&selected.flat),
            kept_out: names(&selected.kept_out),
            library: selected.library,
            tolerances: selected.match_tolerances(),
            flat_basis: selected.flat_basis.clone(),
            flat_held_back: selected.flat_held_back.clone(),
            flat_family: selected.flat_family.clone(),
            flat_choices: selected.flat_choices.clone(),
            rule_notes: selected.rules.notes(),
        });
    }
    finalize(
        &mut sessions,
        &assignments,
        &request.lights,
        &request.master_failures,
        &request.pedestal_evidence,
        request.force_flat_without_pedestal,
    )?;
    Ok(Snapshot {
        algorithm_version: POLICY_ALGORITHM_VERSION,
        selections,
        grades,
        sessions,
        assignments,
        settings_digest: rules.digest(),
        settings_schema: rules::fields().to_vec(),
    })
}

fn selected_frame_current(id: &str, row: &QualityRecord, frames: &[CalibrationFrame]) -> bool {
    frames
        .iter()
        .find(|f| f.frame_uuid == id)
        .is_some_and(|frame| {
            row.algorithm_version
                == Some(if frame.kind == CalibrationKind::Flat {
                    crate::FLAT_QUALITY_ALGORITHM_VERSION
                } else {
                    crate::FRAME_QUALITY_ALGORITHM_VERSION
                })
                && row.source_fingerprint.as_deref() == Some(&frame.source_fingerprint)
        })
}

fn validate_values(s: &FrameSignature) -> crate::Result<()> {
    if [
        s.focal_length_mm,
        s.rotation_deg,
        s.exposure_seconds,
        s.camera_temp_c,
    ]
    .into_iter()
    .flatten()
    .any(|v| !v.is_finite())
    {
        return Err(invalid("frame metadata must be finite"));
    }
    if [s.width, s.height, s.channels, s.binning_x, s.binning_y]
        .into_iter()
        .flatten()
        .any(|v| v <= 0)
    {
        return Err(invalid(
            "known frame dimensions and binning must be positive",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
