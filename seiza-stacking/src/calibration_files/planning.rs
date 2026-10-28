//! Desktop orchestration of Seiza's native objects. JSON is only the Swift
//! presentation boundary; grading, selection and master decisions stay native.
use super::*;
use seiza_calibration::flat_response::groups;
use seiza_calibration::library::rules::{self, RigLayer, RuleBook};
use std::sync::Arc;

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Setting {
    pub key: String,
    pub value: Value,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RigSettings {
    pub rig_uuid: String,
    pub settings: Vec<Setting>,
}
pub(super) fn rules(settings: &[Setting], rigs: &[RigSettings]) -> Result<RuleBook> {
    let values = |settings: &[Setting]| -> Result<BTreeMap<String, Value>> {
        let mut values = BTreeMap::new();
        for setting in settings {
            if values
                .insert(setting.key.clone(), setting.value.clone())
                .is_some()
            {
                return Err(format!("Duplicate calibration setting: {}", setting.key));
            }
        }
        Ok(values)
    };
    let mut layers = BTreeMap::new();
    for rig in rigs {
        if layers
            .insert(
                rig.rig_uuid.clone(),
                RigLayer {
                    name: None,
                    values: values(&rig.settings)?,
                },
            )
            .is_some()
        {
            return Err("Duplicate calibration rig settings".into());
        }
    }
    RuleBook::validated(values(settings)?, layers).map_err(error)
}

#[derive(Clone)]
pub(super) struct Planner {
    pub frames: Vec<policy::CalibrationFrame>,
    pub lights: Vec<policy::Light>,
    pub quality: policy::QualityIndex,
    pub judged: std::collections::HashMap<String, policy::Judged>,
    pub rules: RuleBook,
    pub nights: policy::NightShifts,
    pub grouping: groups::Request,
    pub available_responses: Vec<String>,
    pub options: policy::SessionOptions,
    pub associations: policy::Associations,
}
impl Planner {
    pub fn new(request: &Request, rows: &[Cached], lights: &[Cached]) -> Result<Self> {
        let rules = match &request.rules {
            Some(rules) => rules.clone(),
            None => rules(&request.settings, &request.rigs)?,
        };
        let mut frames = rows.iter().map(|r| r.frame.clone()).collect::<Vec<_>>();
        for frame in &frames {
            frame.check().map_err(error)?;
        }
        let mut quality =
            policy::QualityIndex::new(rows.iter().map(|r| r.quality.clone())).map_err(error)?;
        let nights = policy::NightShifts::default().with_zones(&frames, &rules);
        let judged = quality.regrade(&frames, &rules, &nights);
        quality.keep_out(&mut frames, &judged);
        for light in lights {
            light.light.check().map_err(error)?;
        }
        let planner = Self {
            frames,
            lights: lights.iter().map(|r| r.light.clone()).collect(),
            quality,
            judged,
            rules,
            nights,
            grouping: groups::Request::default(),
            available_responses: Vec::new(),
            options: policy::SessionOptions {
                flat_star_masking: request.flat_star_masking,
                flat_stability: request.flat_stability.clone(),
            },
            associations: policy::Associations {
                flat: request.flat_association,
                dark: request.dark_association,
            },
        };
        planner.library()?;
        Ok(planner)
    }
    pub fn library(&self) -> Result<policy::Library> {
        let library = policy::Library::checked(
            self.frames.clone(),
            Arc::new(groups::prepare(self.grouping.clone()).map_err(error)?),
            self.available_responses.clone(),
        )
        .map_err(error)?;
        library.check_limits(&self.rules).map_err(error)?;
        library
            .check_associations(&self.rules, self.associations)
            .map_err(error)?;
        Ok(library)
    }
    pub fn select(
        &self,
        library: &policy::Library,
        light: &policy::Light,
    ) -> policy::CalibrationSelection {
        library.select_associated(light, self.rules.for_light(light), self.associations)
    }
    pub fn snapshot(&self) -> Result<Snapshot> {
        let library = self.library()?;
        let mut sessions: Vec<policy::Session> = Vec::new();
        let mut assignments = Vec::new();
        let mut selections = Vec::new();
        let mut keys = BTreeMap::new();
        for light in &self.lights {
            let mut selection = self.select(&library, light);
            selection.cap_verified();
            let session = policy::resolve_session(&selection, &self.quality, &self.options);
            let index = *keys.entry(session.key.clone()).or_insert_with(|| {
                sessions.push(session);
                sessions.len() - 1
            });
            assignments.push(index);
            selections.push(Selection::from(&selection));
        }
        for (index, session) in sessions.iter_mut().enumerate() {
            session.plan_flat_use(
                self.lights
                    .iter()
                    .zip(&assignments)
                    .filter(|(_, a)| **a == index)
                    .map(|(l, _)| l),
                policy::FlatWithoutMasters::FitPedestal,
            );
        }
        Ok(Snapshot {
            algorithm_version: policy::POLICY_ALGORITHM_VERSION,
            selections,
            sessions,
            assignments,
            settings_digest: self.rules.digest(),
        })
    }
    pub fn evidence_lights(
        &self,
        session: &groups::ImagingSession,
        rig: &str,
        lights: &[Cached],
    ) -> Vec<policy::EvidenceLight> {
        session
            .lights
            .iter()
            .filter_map(|id| usize::try_from(*id).ok().and_then(|id| lights.get(id)))
            .filter(|l| l.light.rig_uuid == rig)
            .filter_map(|l| {
                Some(policy::EvidenceLight {
                    path: l.source.path.clone(),
                    at: l.light.timestamp?,
                    angle: l.light.rotator_position,
                    filter: l.light.filter.clone().unwrap_or_default(),
                    target: l.target.clone().unwrap_or_default(),
                })
            })
            .collect()
    }
}

/// Native selection and master sessions, with assignments in light order.
// Swift review records: identifiers and descriptions from native selections.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Debug, Serialize)]
pub struct Snapshot {
    /// Version of the native planning records.
    pub algorithm_version: u32,
    /// Per-light native selection records in original order.
    pub selections: Vec<Selection>,
    /// Unique master sessions or measured flat sessions.
    pub sessions: Vec<policy::Session>,
    /// Index of each light's session in sessions.
    pub assignments: Vec<usize>,
    /// Native digest of resolved settings.
    pub settings_digest: Option<String>,
}
/// The identifiers and native explanations of one light's selected calibration.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Debug, Serialize)]
pub struct Selection {
    /// Selected bias frame identifiers.
    pub bias: Vec<String>,
    /// Selected dark frame identifiers.
    pub dark: Vec<String>,
    /// Selected dark-flat frame identifiers.
    pub dark_flat: Vec<String>,
    /// Selected flat frame identifiers.
    pub flat: Vec<String>,
    /// Compatible frames excluded by native quality/validity policy.
    pub kept_out: Vec<String>,
    /// Kinds the native library could provide for this light.
    pub library: policy::LibraryKinds,
    /// The complete native physical matching gate.
    pub tolerances: seiza_calibration::MatchTolerances,
    /// Native explanation, placement and any required correction.
    pub flat_basis: policy::FlatBasis,
    /// Far-side flats unavailable to native alternate retries.
    pub flat_held_back: Vec<String>,
    /// Native family choice and its supporting basis.
    pub flat_family: Option<policy::FlatFamily>,
    /// Applied and orphaned personal commands.
    pub flat_choices: Vec<policy::ChoiceNote>,
    /// Resolved per-light rule explanations.
    pub rule_notes: Vec<rules::RuleNote>,
}
impl From<&policy::CalibrationSelection> for Selection {
    fn from(s: &policy::CalibrationSelection) -> Self {
        let ids = |v: &[policy::CalibrationFrame]| v.iter().map(|f| f.frame_uuid.clone()).collect();
        Self {
            bias: ids(&s.bias),
            dark: ids(&s.dark),
            dark_flat: ids(&s.dark_flat),
            flat: ids(&s.flat),
            kept_out: ids(&s.kept_out),
            library: s.library,
            tolerances: s.match_tolerances(),
            flat_basis: s.flat_basis.clone(),
            flat_held_back: s.flat_held_back.clone(),
            flat_family: s.flat_family.clone(),
            flat_choices: s.flat_choices.clone(),
            rule_notes: s.rules.notes(),
        }
    }
}
/// Measured native grouping visible to file-based hosts.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Debug, Serialize)]
pub struct GroupingReview {
    /// Version of the native planning records.
    pub algorithm_version: u32,
    /// Imaging sessions formed from acquisition identities and times.
    pub imaging: Vec<groups::ImagingSession>,
    /// Unique master sessions or measured flat sessions.
    pub sessions: Vec<groups::Session>,
    /// Measured native response families.
    pub families: Vec<groups::Family>,
    /// Measured or manually chosen response boundaries.
    pub boundaries: Vec<groups::Boundary>,
    /// Optical changes resolved by the native grouping book.
    pub events: Vec<groups::Event>,
    /// Applied/orphaned native group edge commands.
    pub overrides: Vec<groups::OverrideState>,
}
pub(super) fn grouping(book: &groups::Book) -> GroupingReview {
    GroupingReview {
        algorithm_version: groups::ALGORITHM_VERSION,
        imaging: book.imaging.clone(),
        sessions: book.sessions.clone(),
        families: book.families.clone(),
        boundaries: book.boundaries.clone(),
        events: book.events.clone(),
        overrides: book.overrides.clone(),
    }
}
