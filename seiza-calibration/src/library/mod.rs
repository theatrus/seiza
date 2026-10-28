//! Calibration library policy, extracted from PSF Guard and shared by every
//! host: grading calibration frames against their peers, choosing each
//! light's frames, and resolving the masters they build. Nothing here opens
//! files or a database.
//!
//! Native Rust hosts hold the model directly: a [`Library`] of catalog rows
//! selects for each light ([`Library::select`]), [`grade`] judges stored
//! measurements, and [`resolve_session`] turns a verified selection into its
//! master recipes. C and Python submit one [`Request`] to [`evaluate`], which
//! runs the same calls.
mod flat_notes;
mod flat_work;
mod grading;
mod master_policy;
mod recipes;
mod response_plan;
pub mod rules;
mod selection;
mod supplied;

pub use flat_notes::{
    AlikeRangeNote, ChoiceNote, EventNote, FlatBasis, FlatCorrection, FlatFamily, GroupNote,
    correction_key, describe_spots, describe_what, edge_words, family_label, place_light,
    vignetting_leads,
};
pub use flat_work::{
    ChangeFilter, ChangeQuestion, EvidenceLight, FamilyQuestion, FlatChangeFilter,
    FlatChangeQuestion, FlatCorrectionPlan, FlatFamilyCandidate, FlatFamilyQuestion,
    FlatLightContext, FlatWorkAnswer, FlatWorkPlans, FlatWorkRequest, FlatWorkResult,
    change_question, family_questions, finish_flat_work, plan_flat_work,
};
pub use grading::{describe_finding, grade, kept_out_reason};
pub use master_policy::{
    FlatAttempt, FlatRetryAction, FlatRetryDecision, FlatRetryPolicy, FlatRetryRequest,
    MasterCohort, MasterCohortRequest, MasterPlanRequest, alternate_note, flat_retry,
    flat_set_label, master_cohort, plan_masters, same_session_as, trimmed_note, trims_to_majority,
    unstable_note,
};
pub use recipes::{
    Application, FlatUse, MAX_PEDESTAL_ADU, MasterFailure, PedestalEvidence, PedestalRefusal,
    SessionOptions, accept_pedestal, adopted_master, dark_level_screening, master_dependencies,
    pedestal_hint, pedestal_sample_lights, resolve_session,
};
pub use response_plan::{
    FlatResponsePlan, FlatResponsePlans, PedestalTable, SessionPlan, family_key, focal_classes,
    pedestal_moved, plan_flat_responses, plan_sessions,
};
pub use rules::{
    ChoiceOption, Effect, FieldKind, FieldSpec, Group, RigLayer, RuleBook, RuleNote, RuleSource,
    RuleValues, Rules,
};
pub use selection::{Cohort, coherent_cohort};
pub use supplied::Association;

/// The tests a light and a candidate frame pass to match, as selection
/// applies them. Hosts that explain a refusal ask the same questions.
pub mod matching {
    pub use super::selection::{
        Signature, dark_within_reach, exposure_matches, flat_matches, frame_pair_matches,
        frame_signature, light_signature, readout_names_agree, readout_names_equal, rotation_gap,
        sensor_matches, signature_sensor_consistent, signature_sensor_matches, temperature_admits,
        temperature_matches, validity_admits,
    };
}

use crate::flat_response::groups;
use crate::{
    BiasScreening, DarkMeasurement, DarkScreening, Finding, FindingCode, FlatMeasurement,
    FrameSignature, FrameVerdict, Grade, MatchTolerances,
};
use flat_notes as flat_groups;
use recipes::finalize;
use selection::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

/// The fewest frames a master integrates.
pub const MIN_MASTER_FRAMES: usize = 2;
/// The most frames of a kind an automatic master takes, nearest first.
pub const MAX_MASTER_FRAMES: usize = 64;
/// The version of session identity: what a session key covers and how.
pub const POLICY_ALGORITHM_VERSION: u32 = 5;
pub const DEFAULT_DARK_REACH_DAYS: f64 = 183.0;
pub const DEFAULT_COMPLETE_DARK_FRAMES: usize = 10;
/// How far a light's nearest flats may be, in days, before it counts as one
/// that cannot be calibrated, when no setting says otherwise.
pub const DEFAULT_FLAT_MAX_AGE_DAYS: f64 = 60.0;
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
impl CalibrationKind {
    pub const ALL: [CalibrationKind; 4] = [Self::Bias, Self::Dark, Self::DarkFlat, Self::Flat];
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bias => "bias",
            Self::Dark => "dark",
            Self::DarkFlat => "dark_flat",
            Self::Flat => "flat",
        }
    }
    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == value)
    }
}

/// Which lights a calibration frame may serve, relative to its own capture
/// time: marked around an optical change, so a frame never serves imaging
/// on the other side of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidDirection {
    /// Only lights captured at or after this frame.
    Forward,
    /// Only lights captured at or before this frame.
    Backward,
}
impl ValidDirection {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Backward => "backward",
        }
    }
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "forward" => Some(Self::Forward),
            "backward" => Some(Self::Backward),
            _ => None,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DarkChoice {
    #[default]
    Complete,
    Nearest,
}
impl DarkChoice {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::Nearest => "nearest",
        }
    }
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "complete" => Some(Self::Complete),
            "nearest" => Some(Self::Nearest),
            _ => None,
        }
    }
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
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Prefer => "prefer",
            Self::Fallback => "fallback",
            Self::Ignore => "ignore",
        }
    }
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "prefer" => Some(Self::Prefer),
            "fallback" => Some(Self::Fallback),
            "ignore" => Some(Self::Ignore),
            _ => None,
        }
    }
}

/// One catalog row: a raw calibration frame, or a master another program
/// integrated.
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
    /// The readout mode's name when the header spelled it that way, matched
    /// independently of its numeric index.
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

/// The frames one light matched, per kind, nearest first, and why.
#[derive(Clone, Debug, Default)]
pub struct CalibrationSelection {
    pub bias: Vec<CalibrationFrame>,
    pub dark: Vec<CalibrationFrame>,
    pub dark_flat: Vec<CalibrationFrame>,
    pub flat: Vec<CalibrationFrame>,
    /// Frames that matched but stay out: graded poor, kept out by hand, or
    /// marked for stray light before grading.
    pub kept_out: Vec<CalibrationFrame>,
    /// Which kinds the library holds for this light's sensor at all.
    pub library: LibraryKinds,
    /// The rules the light was matched with. Whatever reduces or judges the
    /// selection afterwards reads them here, so one light is never matched
    /// by one rule and built or checked by another.
    pub rules: Arc<Rules>,
    /// Why the flats are the ones they are: the nearest, or chosen around a
    /// measured change.
    pub flat_basis: flat_groups::FlatBasis,
    /// Flats from the far side of a measured change that stay in `flat`
    /// only so the light's stacks keep their keys, by frame UUID: too far
    /// for its own master, and refused by every fallback.
    pub flat_held_back: Vec<String>,
    /// The response family the flats come from, when measured grouping
    /// chose one.
    pub flat_family: Option<flat_groups::FlatFamily>,
    /// A person's choices for this light's flats: those in force, and
    /// those that found nothing to apply to.
    pub flat_choices: Vec<flat_groups::ChoiceNote>,
    /// Kinds taken as one supplied set rather than chosen by time.
    pub one_set: BTreeSet<CalibrationKind>,
}
/// Which kinds of usable frame the library holds for one light's sensor,
/// whatever their optics, exposure, age or validity mark. A light only
/// lacks what its library could have given it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct LibraryKinds {
    pub bias: bool,
    pub dark: bool,
    pub flat: bool,
    /// Darks for this sensor that the light's rig does without
    /// ([`Rules::use_darks`]): the library still holds something for it.
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
    /// taken from the nearest other angle. Its calibration session carries it.
    pub fn match_tolerances(&self) -> MatchTolerances {
        let mut tolerances = self.rules.tolerances();
        if let Some(widened) = self.flat_basis.rotation_tolerance() {
            tolerances.rotation_deg = tolerances.rotation_deg.max(widened);
        }
        if let Some(family) = &self.flat_family {
            tolerances.light_rotation = family.light_rotation;
        }
        tolerances
    }
    pub fn is_empty(&self) -> bool {
        self.bias.is_empty()
            && self.dark.is_empty()
            && self.dark_flat.is_empty()
            && self.flat.is_empty()
    }
    pub fn of_kind(&self, kind: CalibrationKind) -> &[CalibrationFrame] {
        match kind {
            CalibrationKind::Bias => &self.bias,
            CalibrationKind::Dark => &self.dark,
            CalibrationKind::DarkFlat => &self.dark_flat,
            CalibrationKind::Flat => &self.flat,
        }
    }
    fn of_kind_mut(&mut self, kind: CalibrationKind) -> &mut Vec<CalibrationFrame> {
        match kind {
            CalibrationKind::Bias => &mut self.bias,
            CalibrationKind::Dark => &mut self.dark,
            CalibrationKind::DarkFlat => &mut self.dark_flat,
            CalibrationKind::Flat => &mut self.flat,
        }
    }
    /// Every selected frame, bias, dark, dark-flat then flat.
    pub fn frames(&self) -> impl Iterator<Item = &CalibrationFrame> {
        self.bias
            .iter()
            .chain(&self.dark)
            .chain(&self.dark_flat)
            .chain(&self.flat)
    }
}

/// Every frame of a library, held once so many lights select against it
/// without copying it: the catalog rows with their kept-out reasons decided,
/// the flat grouping book, and the correction responses the host can read.
#[derive(Debug, Default)]
pub struct Library {
    /// Newest first, as matching walks them.
    frames: Vec<CalibrationFrame>,
    flat_book: Arc<groups::Book>,
    available_responses: BTreeSet<String>,
}
impl Library {
    /// A library of `frames`. Their order does not matter: matching walks
    /// them newest first, the id and then the UUID breaking a tie. A blank
    /// readout mode name reads as none.
    pub fn new(
        mut frames: Vec<CalibrationFrame>,
        flat_book: Arc<groups::Book>,
        available_responses: impl IntoIterator<Item = String>,
    ) -> Self {
        for frame in &mut frames {
            if frame
                .readout_mode_name
                .as_deref()
                .is_some_and(|name| name.trim().is_empty())
            {
                frame.readout_mode_name = None;
            }
        }
        frames.sort_by(|a, b| {
            b.captured_at
                .cmp(&a.captured_at)
                .then(b.id.cmp(&a.id))
                .then(b.frame_uuid.cmp(&a.frame_uuid))
        });
        Self {
            frames,
            flat_book,
            available_responses: available_responses.into_iter().collect(),
        }
    }
    /// The frames, newest first.
    pub fn frames(&self) -> &[CalibrationFrame] {
        &self.frames
    }
    pub fn flat_book(&self) -> &Arc<groups::Book> {
        &self.flat_book
    }
    fn can_correct(&self, before: &str, after: &str) -> bool {
        self.available_responses.contains(before) && self.available_responses.contains(after)
    }
    /// What a stack would match to `light` by `rules`: each kind's
    /// candidates nearest first, and the account of its flats.
    pub fn select(&self, light: &Light, rules: Arc<Rules>) -> CalibrationSelection {
        let choices = self.light_choices(light, &rules);
        self.select_choosing(light, rules, choices)
    }
    /// [`Library::select`], with a supplied set taken whole for a kind
    /// under [`Association::OneSet`]. Check the set first with
    /// [`Library::check_associations`].
    pub fn select_associated(
        &self,
        light: &Light,
        rules: Arc<Rules>,
        flat: Association,
        dark: Association,
    ) -> CalibrationSelection {
        let mut selected = self.select(light, rules);
        supplied::apply(self, light, &mut selected, flat, dark);
        selected
    }
    /// Whether the library's frames can stand as one supplied set for each
    /// kind taken whole.
    pub fn check_associations(
        &self,
        rules: &RuleBook,
        flat: Association,
        dark: Association,
    ) -> crate::Result<()> {
        supplied::validate(&self.frames, rules, flat, dark)
    }
}

/// The night a frame belongs to: where its rig's nights split.
pub trait NightSplit {
    /// Seconds past midnight UTC at which this frame's night begins, as
    /// `12h - UTC offset` unwrapped: 43200 at Greenwich.
    fn boundary(&self, rig_uuid: &str, frame_uuid: &str) -> i64;
    /// The night a frame shot at `at` belongs to.
    fn night_of_frame(&self, rig_uuid: &str, frame_uuid: &str, at: i64) -> String {
        night_of(at, self.boundary(rig_uuid, frame_uuid))
    }
}
/// The date a night starting at `boundary` seconds past midnight UTC calls
/// a moment, as `YYYY-MM-DD`: the frames of one night share it, the dusk's
/// and the dawn's alike.
pub fn night_of(timestamp: i64, boundary: i64) -> String {
    chrono::DateTime::from_timestamp(timestamp.saturating_sub(boundary), 0)
        .map(|when| when.date_naive().to_string())
        .unwrap_or_else(|| "unknown".into())
}
/// Per-frame night boundaries given as records; an unlisted frame's night
/// begins at noon UTC.
#[derive(Clone, Debug, Default)]
pub struct NightShifts {
    shifts: BTreeMap<(String, String), i64>,
}
impl NightShifts {
    pub fn new(rows: impl IntoIterator<Item = groups::NightBoundary>) -> Self {
        Self {
            shifts: rows
                .into_iter()
                .map(|row| ((row.rig_uuid, row.frame_uuid), row.seconds))
                .collect(),
        }
    }
    fn rows(&self) -> Vec<groups::NightBoundary> {
        self.shifts
            .iter()
            .map(|((rig_uuid, frame_uuid), seconds)| groups::NightBoundary {
                rig_uuid: rig_uuid.clone(),
                frame_uuid: frame_uuid.clone(),
                seconds: *seconds,
            })
            .collect()
    }
}
impl NightSplit for NightShifts {
    fn boundary(&self, rig_uuid: &str, frame_uuid: &str) -> i64 {
        self.shifts
            .get(&(rig_uuid.to_string(), frame_uuid.to_string()))
            .copied()
            .unwrap_or(43200)
    }
}

/// The stored quality rows of a library, by frame.
#[derive(Clone, Debug, Default)]
pub struct QualityIndex {
    rows: BTreeMap<String, QualityRecord>,
    /// Frames graded in this call rather than from a stored grade.
    fresh: HashSet<String>,
}
impl QualityIndex {
    /// Index `rows`, one per frame.
    pub fn new(rows: impl IntoIterator<Item = QualityRecord>) -> crate::Result<Self> {
        let mut index = BTreeMap::new();
        for row in rows {
            if index.insert(row.frame_uuid.clone(), row).is_some() {
                return Err(invalid("quality rows must name unique calibration frames"));
            }
        }
        Ok(Self {
            rows: index,
            fresh: HashSet::new(),
        })
    }
    pub fn get(&self, frame_uuid: &str) -> Option<&QualityRecord> {
        self.rows.get(frame_uuid)
    }
    /// Whether `row` was measured from `frame` as it stands, by this build's
    /// measurement.
    pub fn current(frame: &CalibrationFrame, row: &QualityRecord) -> bool {
        !frame.is_master
            && row.algorithm_version
                == Some(if frame.kind == CalibrationKind::Flat {
                    crate::FLAT_QUALITY_ALGORITHM_VERSION
                } else {
                    crate::FRAME_QUALITY_ALGORITHM_VERSION
                })
            && row.source_fingerprint.as_deref() == Some(&frame.source_fingerprint)
    }
    /// Whether a person chose for `frame`, or it carries a grade made from
    /// its current file: its grade already keeps it out if it is poor.
    pub fn judged(&self, frame: &CalibrationFrame) -> bool {
        self.fresh.contains(&frame.frame_uuid)
            || self.rows.get(&frame.frame_uuid).is_some_and(|row| {
                row.override_choice.is_some() || (row.grade.is_some() && Self::current(frame, row))
            })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityOverride {
    Use,
    Exclude,
}
impl QualityOverride {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Use => "use",
            Self::Exclude => "exclude",
        }
    }
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "use" => Some(Self::Use),
            "exclude" => Some(Self::Exclude),
            _ => None,
        }
    }
}
/// What was measured on one frame, as stored.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StoredMeasurement {
    /// A bias, dark or dark-flat.
    Dark(DarkMeasurement),
    /// A flat.
    Flat(FlatRecord),
}
/// A flat's measurement, with the light cone its header recorded so its
/// session's dust, found later on the stack, can be placed too. Stored as
/// the measurement's own fields plus `optics`.
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
impl std::ops::DerefMut for FlatRecord {
    fn deref_mut(&mut self) -> &mut FlatMeasurement {
        &mut self.measurement
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct Judged {
    pub grade: Grade,
    pub code: Option<FindingCode>,
    pub reason: Option<String>,
    pub findings: Vec<Finding>,
    pub stray_light: Option<String>,
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
pub type NightShift = groups::NightBoundary;
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Request {
    pub frames: Vec<CalibrationFrame>,
    pub lights: Vec<Light>,
    pub quality: Vec<QualityRecord>,
    pub global_settings: Vec<Setting>,
    pub rigs: Vec<RigSettings>,
    pub grouping: groups::Request,
    pub available_responses: Vec<String>,
    pub night_shifts: Vec<NightShift>,
    pub regrade: bool,
    pub master_failures: Vec<MasterFailure>,
    pub pedestal_evidence: Vec<PedestalEvidence>,
    pub force_flat_without_pedestal: bool,
    pub flat_star_masking: bool,
    pub flat_stability: FlatRetryPolicy,
    /// Automatic association, or one explicit supplied set for every compatible light.
    pub flat_association: Association,
    pub dark_association: Association,
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
impl Selection {
    fn of(selected: &CalibrationSelection) -> Self {
        let names =
            |frames: &[CalibrationFrame]| frames.iter().map(|f| f.frame_uuid.clone()).collect();
        Self {
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
        }
    }
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
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Session {
    pub key: String,
    pub tolerances: MatchTolerances,
    pub recipes: Vec<MasterRecipe>,
    pub application: Application,
    pub flat_stability: FlatRetryPolicy,
    #[serde(default)]
    pub flat_held_back: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MasterRecipe {
    pub kind: CalibrationKind,
    pub action: RecipeAction,
    pub inputs: Vec<String>,
    pub dependencies: Vec<CalibrationKind>,
    pub dropped: Vec<String>,
    /// Coherent raw cohort, including a singleton that cannot yet integrate.
    pub coherent_inputs: Vec<String>,
    pub input_limit: usize,
    pub requires_source_verification: bool,
    pub failure: Option<String>,
    pub exposure_seconds: Option<f64>,
    pub defect_suppression: bool,
    pub flat_star_masking: bool,
    pub dark_level_screening: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

fn hex_digest(value: &str) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(value.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Run grading and selection against an immutable library snapshot.
/// Selections and assignments retain light order; sessions reuse identical recipes.
/// Recipes still require host source verification and the native pixel builders.
pub fn evaluate(request: Request) -> crate::Result<Snapshot> {
    evaluate_inner(request, None)
}

/// Select against the host's immutable cached grouping book. No grouping rows
/// may also be supplied. The host invalidates the book when its source rows,
/// night boundaries or settings change; resolved grouping limits are checked.
pub fn evaluate_with_book(request: Request, book: Arc<groups::Book>) -> crate::Result<Snapshot> {
    if !request.grouping.is_empty() {
        return Err(invalid(
            "supply either a prepared grouping book or grouping rows",
        ));
    }
    evaluate_inner(request, Some(book))
}

/// A request's settings, frames and quality, checked once for every
/// planning call that takes them.
struct Checked {
    rules: RuleBook,
    nights: NightShifts,
    quality: QualityIndex,
}

/// Refuse what no host should send: duplicate or empty frame identities,
/// verified rows without a source, non-finite or nonpositive metadata,
/// unknown or invalid settings, and quality rows for frames not supplied.
fn check_request(
    frames: &[CalibrationFrame],
    lights: &[Light],
    quality: Vec<QualityRecord>,
    global: Vec<Setting>,
    rigs: Vec<RigSettings>,
) -> crate::Result<(RuleBook, QualityIndex)> {
    let mut ids = BTreeSet::new();
    for frame in frames {
        if frame.frame_uuid.is_empty() || !ids.insert(frame.frame_uuid.as_str()) {
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
    for light in lights {
        validate_values(&light_signature(light).seiza)?;
    }
    if let Some(row) = quality
        .iter()
        .find(|row| !ids.contains(row.frame_uuid.as_str()))
    {
        return Err(invalid(format!(
            "quality row {} names no supplied calibration frame",
            row.frame_uuid
        )));
    }
    let rules = master_policy::rule_book(global, rigs)?;
    Ok((rules, QualityIndex::new(quality)?))
}

fn evaluate_inner(
    mut request: Request,
    prepared: Option<Arc<groups::Book>>,
) -> crate::Result<Snapshot> {
    if !request.grouping.lights.is_empty() {
        return Err(invalid(
            "library lights belong in lights, not grouping.lights",
        ));
    }
    // Light evidence is kept per imaging session: without the sessions it
    // was measured for, it would place nothing, silently.
    let grouping = &request.grouping;
    if (!grouping.evidence.is_empty() || !grouping.family_evidence.is_empty())
        && grouping.imaging.is_empty()
        && grouping.imaging_lights.is_empty()
    {
        return Err(invalid(
            "light evidence needs the imaging sessions it was measured for: submit the grouping plan_flat_work returns",
        ));
    }
    let checked = checked(&mut request)?;
    if let Some(book) = &prepared {
        for session in &book.sessions {
            let rig = &session.row.rig_uuid;
            if book.limits(rig) != Some(checked.rules.for_rig(rig).grouping_limits()) {
                return Err(invalid(
                    "prepared grouping limits differ from the library settings",
                ));
            }
        }
    }
    let Checked {
        rules,
        nights,
        mut quality,
    } = checked;
    // Only current, fingerprint-matching measurements are regraded; stale,
    // failed and absent ones keep their state and never become a grade.
    let judged = if request.regrade {
        let measured = request.frames.iter().filter_map(|frame| {
            let row = quality.get(&frame.frame_uuid)?;
            QualityIndex::current(frame, row).then_some((frame, row.measurement.as_ref()?))
        });
        grade(measured, &rules, &nights)
    } else {
        HashMap::new()
    };
    quality.fresh = judged.keys().cloned().collect();
    let grades = apply_grades(&mut request.frames, &quality, &judged);
    let flat_book = match prepared {
        Some(book) => book,
        None => Arc::new(groups::prepare(std::mem::take(&mut request.grouping))?),
    };
    let library = Library::new(request.frames, flat_book, request.available_responses);
    library.check_associations(&rules, request.flat_association, request.dark_association)?;
    let options = SessionOptions {
        flat_star_masking: request.flat_star_masking,
        flat_stability: request.flat_stability,
    };
    let mut selections = Vec::new();
    let mut sessions: Vec<Session> = Vec::new();
    let mut assignments = Vec::new();
    let mut keys = BTreeMap::new();
    for light in &request.lights {
        let selected = library.select_associated(
            light,
            rules.for_light(light),
            request.flat_association,
            request.dark_association,
        );
        let session = resolve_session(&selected, &quality, &options);
        // The first light whose masters share a key stands for every light
        // they serve: its candidates' order anchors the masters.
        let index = *keys.entry(session.key.clone()).or_insert_with(|| {
            sessions.push(session);
            sessions.len() - 1
        });
        assignments.push(index);
        selections.push(Selection::of(&selected));
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

/// Check a request and resolve the settings and night boundaries its
/// grouping, grading and selection share. The grouping takes each rig's
/// limits from the same settings, so a separate grouping request cannot
/// silently give its rig another threshold.
fn checked(request: &mut Request) -> crate::Result<Checked> {
    let (rules, quality) = check_request(
        &request.frames,
        &request.lights,
        std::mem::take(&mut request.quality),
        std::mem::take(&mut request.global_settings),
        std::mem::take(&mut request.rigs),
    )?;
    let group_rigs: BTreeSet<_> = request
        .grouping
        .responses
        .iter()
        .map(|r| r.rig_uuid.clone())
        .collect();
    request.grouping.limits = group_rigs
        .into_iter()
        .map(|rig_uuid| groups::RigLimits {
            limits: rules.for_rig(&rig_uuid).grouping_limits(),
            rig_uuid,
        })
        .collect();
    let nights = resolve_nights(request, &rules)?;
    Ok(Checked {
        rules,
        nights,
        quality,
    })
}

/// Each frame's kept-out reason from its quality row and any fresh grade,
/// and its grade as the snapshot reports it, in frame order.
fn apply_grades(
    frames: &mut [CalibrationFrame],
    quality: &QualityIndex,
    judged: &HashMap<String, Judged>,
) -> Vec<FrameGrade> {
    let mut grades = Vec::with_capacity(frames.len());
    for frame in frames {
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
        frame.kept_out = kept_out_reason(
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
                Some(r) if !QualityIndex::current(frame, r) => "stale",
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
    grades
}

/// Merge a request's night boundaries: the grouping's, then explicit
/// shifts, then each rig's `night_zone` setting. The grouping then holds
/// the merged set, so grouping and grading split nights alike.
fn resolve_nights(request: &mut Request, rules: &RuleBook) -> crate::Result<NightShifts> {
    let mut rows: BTreeMap<_, _> = std::mem::take(&mut request.grouping.night_boundaries)
        .into_iter()
        .map(|s| ((s.rig_uuid.clone(), s.frame_uuid.clone()), s))
        .collect();
    let mut explicit = BTreeSet::new();
    for shift in std::mem::take(&mut request.night_shifts) {
        if !(-86400..=129600).contains(&shift.seconds)
            || !explicit.insert((shift.rig_uuid.clone(), shift.frame_uuid.clone()))
        {
            return Err(invalid(
                "night shifts must be unique and within one day of noon UTC",
            ));
        }
        rows.insert((shift.rig_uuid.clone(), shift.frame_uuid.clone()), shift);
    }
    for frame in &request.frames {
        if let Some(offset) = rules.for_rig(&frame.rig_uuid).night_offset_seconds {
            rows.insert(
                (frame.rig_uuid.clone(), frame.frame_uuid.clone()),
                groups::NightBoundary {
                    rig_uuid: frame.rig_uuid.clone(),
                    frame_uuid: frame.frame_uuid.clone(),
                    seconds: 43200 - offset,
                },
            );
        }
    }
    let nights = NightShifts::new(rows.into_values());
    request.grouping.night_boundaries = nights.rows();
    Ok(nights)
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
