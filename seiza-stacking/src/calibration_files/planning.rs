//! Desktop orchestration of Seiza's native objects: grading, selection and
//! master decisions stay in `seiza_calibration::library`.
use super::*;
use seiza_calibration::flat_response::groups;
use seiza_calibration::library::rules::RuleBook;
use std::sync::Arc;

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
    pub fn new(
        rules: &RuleBook,
        options: &PreparationOptions,
        rows: &[Cached],
        lights: &[Cached],
    ) -> Result<Self> {
        let rules = rules.clone();
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
            options: options.masters.clone(),
            associations: options.associations,
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
            selections.push(selection.summary());
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
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Debug, Serialize)]
pub struct Snapshot {
    /// Version of the native planning records.
    pub algorithm_version: u32,
    /// Per-light native selection records in original order.
    pub selections: Vec<policy::SelectionSummary>,
    /// Unique master sessions or measured flat sessions.
    pub sessions: Vec<policy::Session>,
    /// Index of each light's session in sessions.
    pub assignments: Vec<usize>,
    /// Native digest of resolved settings.
    pub settings_digest: Option<String>,
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
