//! Calibration library policy, extracted from PSF Guard and shared by every
//! host: grading calibration frames against their peers, choosing each
//! light's frames, and resolving the masters they build. Nothing here opens
//! files or a database.
//!
//! This is the API. Rust hosts call it as their own code, and Python binds
//! it: a [`rules::RuleBook`] holds the settings, a [`QualityIndex`] the
//! stored measurements ([`QualityIndex::regrade`] judges them afresh), a
//! [`Library`] of catalog rows selects for each light
//! ([`Library::select`]), [`resolve_session`] turns a verified selection
//! into its master recipes, and the [`Session`] takes what building them
//! showed. [`FlatRetry`] decides a flat master's retries, and
//! [`plan_flat_sessions`], [`family_questions`] and [`change_question`] the
//! measuring of flat grouping.
//!
//! The C ABI runs whole workflows as one JSON call each instead, built on
//! these: `library::json`, compiled with the `json` feature. Nothing else
//! uses it.
mod flat_notes;
mod flat_work;
mod grading;
#[cfg(feature = "json")]
pub mod json;
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
    ChangeFilter, ChangeQuestion, EvidenceLight, FamilyQuestion, Representatives, RetentionKey,
    change_question, family_questions,
};
pub use grading::{KeptOut, describe_finding, grade};
pub use master_policy::{
    FlatAttempt, FlatRetry, FlatRetryAction, FlatRetryDecision, FlatRetryPolicy, flat_set_label,
};
pub use recipes::{
    Application, FlatUse, FlatWithoutMasters, MAX_PEDESTAL_ADU, PedestalRefusal, SessionOptions,
    adopted_master, dark_level_screening, pedestal_from_fits, pedestal_hint,
    pedestal_sample_lights, resolve_session,
};
pub use response_plan::{
    FlatSessionPlan, PedestalLevel, PedestalTable, ResponseRead, family_key, focal_classes,
    pedestal_moved, plan_flat_sessions,
};
use rules::{DarkChoice, ExternalMasterPolicy, RuleBook, Rules};
pub use selection::{Cohort, coherent_cohort};
pub use supplied::{Association, Associations};

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
/// How far a dark may be from a light, in days, when no setting says.
pub const DEFAULT_DARK_REACH_DAYS: f64 = 183.0;
/// The darks of one night that make a complete set, when no setting says.
pub const DEFAULT_COMPLETE_DARK_FRAMES: usize = 10;
/// How far a light's nearest flats may be, in days, before it counts as one
/// that cannot be calibrated, when no setting says otherwise.
pub const DEFAULT_FLAT_MAX_AGE_DAYS: f64 = 60.0;
const DARK_SESSION_SECONDS: u64 = 86400;

#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
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
words!(CalibrationKind {
    Bias => "bias",
    Dark => "dark",
    DarkFlat => "dark_flat",
    Flat => "flat",
});
impl CalibrationKind {
    /// The kind in words: "bias", "dark", "dark-flat" or "flat".
    pub fn label(self) -> &'static str {
        match self {
            Self::DarkFlat => "dark-flat",
            other => other.as_str(),
        }
    }
    /// The masters a master of this kind is built on: a dark or dark-flat
    /// on its bias, a flat on its bias and dark-flat. One whose dependency
    /// failed is skipped rather than built without it.
    pub fn dependencies(self) -> &'static [CalibrationKind] {
        match self {
            Self::Bias => &[],
            Self::Dark | Self::DarkFlat => &[Self::Bias],
            Self::Flat => &[Self::Bias, Self::DarkFlat],
        }
    }
    /// The version of the measurement a stored grade of this kind must have
    /// been made with to be current.
    pub fn measurement_version(self) -> u32 {
        match self {
            Self::Flat => crate::FLAT_QUALITY_ALGORITHM_VERSION,
            _ => crate::FRAME_QUALITY_ALGORITHM_VERSION,
        }
    }
}

/// Which lights a calibration frame may serve, relative to its own capture
/// time: marked around an optical change, so a frame never serves imaging
/// on the other side of it.
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidDirection {
    /// Only lights captured at or after this frame.
    Forward,
    /// Only lights captured at or before this frame.
    Backward,
}
words!(ValidDirection {
    Forward => "forward",
    Backward => "backward",
});

/// One catalog row: a raw calibration frame, or a master another program
/// integrated.
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
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
/// A light as matching reads it: its file and header.
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Light {
    pub path: PathBuf,
    /// The library rig it was shot with, by the key rigs are filed under.
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
    /// Rotator angle in degrees at capture.
    pub rotator_position: Option<f64>,
    pub exposure_s: Option<f64>,
    /// Sensor temperature, °C.
    pub camera_temp: Option<f64>,
    /// Capture time, Unix seconds.
    pub timestamp: Option<i64>,
}

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
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
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
    /// The selection of exactly `frames`, each under its kind, in the order
    /// given, matched by `rules`: candidates a host chose or verified
    /// itself rather than ones [`Library::select`] found. The kinds
    /// `associations` takes whole are one supplied set. Check them with
    /// [`Library::check_associations`].
    pub fn of_frames(
        frames: impl IntoIterator<Item = CalibrationFrame>,
        rules: Arc<Rules>,
        associations: Associations,
    ) -> Self {
        let mut selected = Self {
            rules,
            one_set: associations.one_set(),
            ..Self::default()
        };
        for frame in frames {
            selected.of_kind_mut(frame.kind).push(frame);
        }
        selected
    }

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
    /// Cut each kind whose candidates are all verified to its nearest
    /// [`MAX_MASTER_FRAMES`], as a master takes them; a kind taken as one
    /// supplied set keeps every frame. An unverified kind is a preview and
    /// keeps its candidates until the host verifies and selects again.
    pub fn cap_verified(&mut self) {
        for kind in CalibrationKind::ALL {
            if self.capped_kind(kind) {
                self.of_kind_mut(kind).truncate(MAX_MASTER_FRAMES);
            }
        }
    }

    /// Whether [`CalibrationSelection::cap_verified`] would cut anything.
    pub(super) fn needs_cap(&self) -> bool {
        CalibrationKind::ALL
            .into_iter()
            .any(|kind| self.capped_kind(kind))
    }

    fn capped_kind(&self, kind: CalibrationKind) -> bool {
        let frames = self.of_kind(kind);
        !self.one_set.contains(&kind)
            && frames.len() > MAX_MASTER_FRAMES
            && frames.iter().all(|f| f.source_verified)
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
impl CalibrationFrame {
    /// Refuse a row no host should send: an empty UUID, a verified source
    /// without a path and fingerprint, or metadata that is not finite or
    /// dimensions and binning that are not positive.
    pub fn check(&self) -> crate::Result<()> {
        if self.frame_uuid.is_empty() {
            return Err(invalid(
                "calibration frame UUIDs must be nonempty and unique",
            ));
        }
        if self.source_verified
            && (self.source_path.as_os_str().is_empty() || self.source_fingerprint.is_empty())
        {
            return Err(invalid("verified sources need a path and fingerprint"));
        }
        validate_values(&frame_signature(self).seiza)
    }
}

impl Light {
    /// Refuse metadata that is not finite, or dimensions and binning that
    /// are not positive.
    pub fn check(&self) -> crate::Result<()> {
        validate_values(&light_signature(self).seiza)
    }
}

/// Refuse frames that repeat a UUID, or one that fails
/// [`CalibrationFrame::check`].
fn check_frames(frames: &[CalibrationFrame]) -> crate::Result<()> {
    let mut ids = BTreeSet::new();
    for frame in frames {
        frame.check()?;
        if !ids.insert(frame.frame_uuid.as_str()) {
            return Err(invalid(
                "calibration frame UUIDs must be nonempty and unique",
            ));
        }
    }
    Ok(())
}

/// Refuse a grouping book whose rig limits differ from the rules'.
fn check_book_limits(book: &groups::Book, rules: &RuleBook) -> crate::Result<()> {
    for session in &book.sessions {
        let rig = &session.row.rig_uuid;
        if book.limits(rig) != Some(rules.for_rig(rig).grouping_limits()) {
            return Err(invalid(
                "prepared grouping limits differ from the library settings",
            ));
        }
    }
    Ok(())
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

impl Library {
    /// [`Library::new`] for rows a host has not checked itself: refuses
    /// repeated or empty UUIDs and rows that fail [`CalibrationFrame::check`].
    pub fn checked(
        frames: Vec<CalibrationFrame>,
        flat_book: Arc<groups::Book>,
        available_responses: impl IntoIterator<Item = String>,
    ) -> crate::Result<Self> {
        check_frames(&frames)?;
        Ok(Self::new(frames, flat_book, available_responses))
    }

    /// Refuse a grouping book grouped by other limits than `rules` give its
    /// rigs: a book a host cached across a change of settings.
    pub fn check_limits(&self, rules: &RuleBook) -> crate::Result<()> {
        check_book_limits(&self.flat_book, rules)
    }

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
        associations: Associations,
    ) -> CalibrationSelection {
        let mut selected = self.select(light, rules);
        supplied::apply(self, light, &mut selected, associations);
        selected
    }
    /// Whether the library's frames can stand as one supplied set for each
    /// kind taken whole.
    pub fn check_associations(
        &self,
        rules: &RuleBook,
        associations: Associations,
    ) -> crate::Result<()> {
        associations.check(&self.frames, rules)
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
    /// The boundaries of `rows`, refusing two for one frame or a shift more
    /// than a day from noon UTC.
    pub fn checked(rows: impl IntoIterator<Item = groups::NightBoundary>) -> crate::Result<Self> {
        let rows: Vec<groups::NightBoundary> = rows.into_iter().collect();
        let mut seen = BTreeSet::new();
        for row in &rows {
            if !(-86400..=129600).contains(&row.seconds)
                || !seen.insert((row.rig_uuid.as_str(), row.frame_uuid.as_str()))
            {
                return Err(invalid(
                    "night shifts must be unique and within one day of noon UTC",
                ));
            }
        }
        Ok(Self::new(rows))
    }
    /// The boundaries of `rows`, one per frame; a later row for a frame
    /// replaces an earlier one.
    pub fn new(rows: impl IntoIterator<Item = groups::NightBoundary>) -> Self {
        Self {
            shifts: rows
                .into_iter()
                .map(|row| ((row.rig_uuid, row.frame_uuid), row.seconds))
                .collect(),
        }
    }
    /// These boundaries with each rig's `night_zone` setting in force over
    /// them for its `frames`: a rig set to a UTC offset splits its nights at
    /// local noon there, whatever its frames were given.
    pub fn with_zones(mut self, frames: &[CalibrationFrame], rules: &RuleBook) -> Self {
        for frame in frames {
            if let Some(offset) = rules.for_rig(&frame.rig_uuid).night_offset_seconds {
                self.shifts.insert(
                    (frame.rig_uuid.clone(), frame.frame_uuid.clone()),
                    43200 - offset,
                );
            }
        }
        self
    }
    /// The boundaries held, one per frame.
    pub fn boundaries(&self) -> Vec<groups::NightBoundary> {
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
            && row.algorithm_version == Some(frame.kind.measurement_version())
            && row.source_fingerprint.as_deref() == Some(&frame.source_fingerprint)
    }
    /// Grade afresh every frame of `frames` whose stored measurement is
    /// current, by `rules` and `nights`, and count those frames judged from
    /// now on. Stale, failed and absent measurements keep their state and
    /// never become a grade.
    pub fn regrade(
        &mut self,
        frames: &[CalibrationFrame],
        rules: &RuleBook,
        nights: &dyn NightSplit,
    ) -> HashMap<String, Judged> {
        let measured = frames.iter().filter_map(|frame| {
            let row = self.rows.get(&frame.frame_uuid)?;
            Self::current(frame, row).then_some((frame, row.measurement.as_ref()?))
        });
        let judged = grade(measured, rules, nights);
        self.fresh.extend(judged.keys().cloned());
        judged
    }

    /// Why `frame` stays out of masters and exports, from a fresh grade in
    /// `judged` or else its stored row; for a frame with neither, the
    /// stray-light mark it carries. `frame.kept_out` is read as that mark,
    /// which an older build left in the catalog: give frames as the catalog
    /// stores them, not ones this has already decided for.
    pub fn kept_out(&self, frame: &CalibrationFrame, judged: Option<&Judged>) -> Option<String> {
        let row = self.rows.get(&frame.frame_uuid);
        KeptOut {
            choice: row.and_then(|r| r.override_choice),
            grade: judged
                .map(|g| g.grade)
                .or_else(|| row.and_then(|r| r.grade)),
            grade_reason: judged
                .and_then(|g| g.reason.as_deref())
                .or_else(|| row.and_then(|r| r.reason.as_deref())),
            stray_light: judged
                .and_then(|g| g.stray_light.as_deref())
                .or_else(|| row.and_then(|r| r.stray_light.as_deref()))
                .or(frame.kept_out.as_deref()),
        }
        .reason()
    }

    /// Set every frame's kept-out reason ([`QualityIndex::kept_out`]), once,
    /// on frames as the catalog stores them.
    pub fn keep_out(&self, frames: &mut [CalibrationFrame], judged: &HashMap<String, Judged>) {
        for frame in frames {
            frame.kept_out = self.kept_out(frame, judged.get(&frame.frame_uuid));
        }
    }

    /// What is known of `frame`'s measurement.
    pub fn measurement_state(&self, frame: &CalibrationFrame) -> MeasurementState {
        if frame.is_master {
            return MeasurementState::External;
        }
        match self.rows.get(&frame.frame_uuid) {
            None => MeasurementState::Unmeasured,
            Some(row) if !Self::current(frame, row) => MeasurementState::Stale,
            Some(row) if row.measurement.is_some() => MeasurementState::Current,
            Some(row) if row.measure_error.is_some() => MeasurementState::Failed,
            _ => MeasurementState::Unmeasured,
        }
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

#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityOverride {
    Use,
    Exclude,
}
words!(QualityOverride {
    Use => "use",
    Exclude => "exclude",
});
/// What was measured on one frame, as stored.
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
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
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
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
/// A frame's grade against its peers, as [`grade`] judges it.
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize)]
pub struct Judged {
    pub grade: Grade,
    /// The finding that decided the grade.
    pub code: Option<FindingCode>,
    /// That finding in words.
    pub reason: Option<String>,
    pub findings: Vec<Finding>,
    /// The stray-light mark, for a dark that caught light.
    pub stray_light: Option<String>,
}

/// What is known of a frame's measurement.
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementState {
    /// A master another program integrated: never measured.
    External,
    /// No measurement stored.
    Unmeasured,
    /// Measured from another file, or by another build's measurement.
    Stale,
    /// Measured from the frame as it stands.
    Current,
    /// The measurement failed.
    Failed,
}
words!(MeasurementState {
    External => "external",
    Unmeasured => "unmeasured",
    Stale => "stale",
    Current => "current",
    Failed => "failed",
});

/// A frame's stored quality: its measurement, grade, and a person's choice.
/// A stale algorithm/fingerprint is not regraded.
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct QualityRecord {
    pub frame_uuid: String,
    /// The fingerprint of the file the measurement was made from.
    pub source_fingerprint: Option<String>,
    /// The measurement's version ([`CalibrationKind::measurement_version`]).
    pub algorithm_version: Option<u32>,
    pub measurement: Option<StoredMeasurement>,
    /// Why the measurement failed.
    pub measure_error: Option<String>,
    /// The stored grade and its reason.
    pub grade: Option<Grade>,
    pub reason: Option<String>,
    pub stray_light: Option<String>,
    /// A person's choice to use or exclude the frame.
    pub override_choice: Option<QualityOverride>,
}
/// The masters one light's selection builds, and how its flat applies:
/// [`resolve_session`]. Lights whose sessions share a key calibrate
/// identically.
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Session {
    /// What the masters are made from and how ([`POLICY_ALGORITHM_VERSION`]).
    pub key: String,
    /// The gate a stack holds the light to against its masters.
    pub tolerances: MatchTolerances,
    /// Bias, dark-flat, dark and flat, in build order.
    pub recipes: Vec<MasterRecipe>,
    pub application: Application,
    pub flat_stability: FlatRetryPolicy,
    #[serde(default)]
    pub flat_held_back: Vec<String>,
}
/// How one master of a session is made.
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MasterRecipe {
    pub kind: CalibrationKind,
    pub action: RecipeAction,
    /// The frames it integrates or adopts, by UUID, nearest first.
    pub inputs: Vec<String>,
    /// The masters it is built on ([`CalibrationKind::dependencies`]).
    pub dependencies: Vec<CalibrationKind>,
    /// Candidates left out of its cohort.
    pub dropped: Vec<String>,
    /// Coherent raw cohort, including a singleton that cannot yet integrate.
    pub coherent_inputs: Vec<String>,
    /// The most frames an automatic master takes ([`MAX_MASTER_FRAMES`]).
    pub input_limit: usize,
    /// Its inputs must be checked against their files before it builds.
    pub requires_source_verification: bool,
    /// Why it failed or was blocked.
    pub failure: Option<String>,
    /// The exposure the master stands for: a dark's, scaled to the light.
    pub exposure_seconds: Option<f64>,
    /// Repair hot and cold pixels while integrating.
    pub defect_suppression: bool,
    /// Mask stars in a master flat.
    pub flat_star_masking: bool,
    /// Screen darks for stray light while integrating: for darks not graded
    /// yet ([`dark_level_screening`]).
    pub dark_level_screening: bool,
}
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecipeAction {
    /// Adopt an external master as it is.
    External,
    /// Integrate the raw inputs.
    Integrate,
    /// Nothing to build: no usable frames of this kind.
    Unavailable,
    /// The build failed ([`MasterRecipe::failure`] says why).
    Failed,
    /// Not built, because a master it depends on failed.
    Blocked,
}
words!(RecipeAction {
    External => "external",
    Integrate => "integrate",
    Unavailable => "unavailable",
    Failed => "failed",
    Blocked => "blocked",
});

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

#[cfg(test)]
mod tests;
