//! Measured flat families, change events, and light assignments.
//!
//! Extracted from PSF Guard rather than reconstructed from metadata matching.
//! Hosts build a [`Book`] from their stored rows ([`prepare`]) and ask it
//! about each light ([`Book::place_light`]). The C ABI's one-call plan is
//! `library::json::flat_group_plan`. Neither interface reads a catalog, files, or pixels.
//! Hosts supply the response/comparison/evidence measurements and the families
//! that passed their hard sensor/optics gates. This is one stage of calibration
//! selection, not a replacement for grading, dark selection, or master recipes.

pub(crate) mod lights;
pub(crate) mod model;
pub(crate) mod overrides;
mod rows;

use super::{choice, evidence, metric};
use serde::{Deserialize, Serialize};

use std::collections::{BTreeSet, HashMap};

/// Version the host includes when caching grouping snapshots and assignments.
pub const ALGORITHM_VERSION: u32 = 1;

pub use lights::{
    ANGLE_PRIOR_DEG, AngleEpochs, EPOCH_JUMP_DEG, Epoch, FAMILY_EVIDENCE_VERSION, FamilyBasis,
    FamilyChoice, half, wrap,
};
pub use model::{
    AlikeRange, AngleSpan, AngleTerm, Book, Boundary, Event, Family, HandEdge, ImagingSession,
    Limits, Placement, ROTATION_CLUSTER_DEG, Session, canonical_filter, imaging_sessions,
};
pub use overrides::{
    GROUP_CHOICE_KEY, GroupChoice, GroupOverrideRow, GroupPick, LightChoice, LightChoiceEntry,
    LightChoiceRow, OverrideState, OverrideStatus, ResolvedChoices,
};
pub use rows::{
    ComparisonRow, EvidenceFit, EvidenceRow, FamilyEvidenceRow, FamilyPairRow, FileStamp,
    ResponseRow, UnreadFlat, UnreadKind,
};

/// Limits for one rig; every rig with usable responses must have an entry.
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RigLimits {
    pub rig_uuid: String,
    pub limits: Limits,
}

/// A rig's sky-minus-mechanical angle, modulo a half turn, at a Unix time.
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EpochSample {
    pub rig_uuid: String,
    pub at: i64,
    pub offset: f64,
}

/// Local-noon shift for one flat, unwrapped (12h minus UTC offset). Unlisted flats use
/// noon UTC. The stable frame identity, rather than a database id, names it.
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NightBoundary {
    pub rig_uuid: String,
    pub frame_uuid: String,
    pub seconds: i64,
}

/// One light to place. `compatible_families` names families that already
/// passed the host's hard sensor/optics gates; empty means no candidates.
/// A missing capture time leaves its event placements unknown.
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LightQuery {
    pub file_name: String,
    pub rig_uuid: String,
    pub filter: Option<String>,
    pub at: Option<i64>,
    pub angle: Option<f64>,
    pub compatible_families: Vec<String>,
    pub rotation_tolerance: f64,
}

/// A complete, immutable snapshot of inputs for the grouping stage. Vectors
/// of keyed records avoid converting user identifiers when bindings translate
/// field names. Omitted collections are empty; unknown fields are refused.
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Request {
    pub responses: Vec<ResponseRow>,
    pub comparisons: Vec<ComparisonRow>,
    pub evidence: Vec<EvidenceRow>,
    pub imaging: Vec<ImagingSession>,
    /// Alternative to `imaging`: `(id, profile, captured_at)` source rows.
    pub imaging_lights: Vec<(i64, Option<String>, i64)>,
    /// `(profile, rigs)` bindings used when grouping `imaging_lights`.
    pub profile_rigs: Vec<(Option<String>, Vec<String>)>,
    pub limits: Vec<RigLimits>,
    pub group_overrides: Vec<GroupOverrideRow>,
    pub night_boundaries: Vec<NightBoundary>,
    pub epoch_samples: Vec<EpochSample>,
    pub family_evidence: Vec<FamilyEvidenceRow>,
    pub light_choices: Vec<LightChoiceEntry>,
    pub lights: Vec<LightQuery>,
}

impl Request {
    /// Whether the request holds no rows at all.
    pub fn is_empty(&self) -> bool {
        let Self {
            responses,
            comparisons,
            evidence,
            imaging,
            imaging_lights,
            profile_rigs,
            limits,
            group_overrides,
            night_boundaries,
            epoch_samples,
            family_evidence,
            light_choices,
            lights,
        } = self;
        responses.is_empty()
            && comparisons.is_empty()
            && evidence.is_empty()
            && imaging.is_empty()
            && imaging_lights.is_empty()
            && profile_rigs.is_empty()
            && limits.is_empty()
            && group_overrides.is_empty()
            && night_boundaries.is_empty()
            && epoch_samples.is_empty()
            && family_evidence.is_empty()
            && light_choices.is_empty()
            && lights.is_empty()
    }
}

/// An automatic event placement and the response pair from which a correction
/// map can be measured. Personal choices are returned separately on the light.
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, Serialize)]
pub struct EventPlacement {
    pub event: usize,
    pub placement: Option<Placement>,
    pub map_source: Option<(String, String)>,
    pub evidence_reference: Option<usize>,
}

/// Inputs for a host's response-choice pixel pass. Candidate and representative
/// indices refer to this snapshot. Reuse `key` on the resulting family evidence
/// row; changes to the candidates or their measurements invalidate old evidence.
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, Serialize)]
pub struct FamilyProbe {
    pub key: String,
    pub candidates: Vec<usize>,
    pub representatives: Vec<Option<usize>>,
}

/// Results in request order. Automatic results deliberately remain distinct
/// from resolved personal choices, as in the native book: the selector can
/// explain both and enforce its hard calibration gates before using a choice.
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, Serialize)]
pub struct LightResult {
    pub automatic_family: Option<FamilyChoice>,
    pub family_probe: Option<FamilyProbe>,
    pub choices: ResolvedChoices,
    pub events: Vec<EventPlacement>,
}

/// Whether a Unix time is within the range grouping's arithmetic supports.
fn safe_time(at: i64) -> bool {
    (-(1_i64 << 50)..=(1_i64 << 50)).contains(&at)
}

impl LightQuery {
    /// Refuse a capture time outside the supported range, a non-finite
    /// angle, or a negative or non-finite rotation tolerance.
    pub fn check(&self) -> crate::Result<()> {
        if self.at.is_some_and(|at| !safe_time(at)) {
            return Err(invalid("capture time is outside the supported range"));
        }
        if !self.rotation_tolerance.is_finite()
            || self.rotation_tolerance < 0.0
            || self.angle.is_some_and(|angle| !angle.is_finite())
        {
            return Err(invalid(
                "light angles and nonnegative rotation tolerances must be finite",
            ));
        }
        Ok(())
    }
}

impl Book {
    /// Where one light stands in this book: its automatic family among
    /// `light.compatible_families`, the pixel question that would settle a
    /// close call, a person's resolved choices, and its placement against
    /// each change of its rig and filter. Missing measurements stay missing:
    /// no metadata-only verdict is invented.
    pub fn place_light(&self, light: &LightQuery) -> crate::Result<LightResult> {
        light.check()?;
        let compatible = light
            .compatible_families
            .iter()
            .map(|key| {
                let index = self
                    .families
                    .iter()
                    .position(|family| family.key == *key)
                    .ok_or_else(|| invalid(format!("unknown compatible family {key:?}")))?;
                let family = &self.families[index];
                if family.rig_uuid != light.rig_uuid
                    || family.filter.as_deref().map(model::canonical_filter)
                        != light.filter.as_deref().map(model::canonical_filter)
                {
                    return Err(invalid(format!(
                        "compatible family {key:?} has another rig or filter"
                    )));
                }
                Ok(index)
            })
            .collect::<crate::Result<Vec<_>>>()?;
        let automatic_family = self.choose_family(
            &light.rig_uuid,
            light.filter.as_deref(),
            light.at,
            light.angle,
            &compatible,
            light.rotation_tolerance,
        );
        let family_probe = light.at.zip(light.angle).map(|(at, angle)| {
            let candidates: Vec<_> = self
                .family_candidates(&light.rig_uuid, Some(at), angle, &compatible)
                .into_iter()
                .map(|(family, _)| family)
                .collect();
            let key = self.family_evidence_key(&candidates, &light.rig_uuid, at, angle);
            let representatives = candidates
                .iter()
                .map(|family| self.family_representative(*family, &light.rig_uuid, at, angle))
                .collect();
            FamilyProbe {
                key,
                candidates,
                representatives,
            }
        });
        let choices = self.light_choices(
            &light.file_name,
            light.filter.as_deref(),
            &light.rig_uuid,
            light.at,
        );
        let events = self
            .events
            .iter()
            .enumerate()
            .filter(|(_, event)| {
                event.rig_uuid == light.rig_uuid && event.applies_to(light.filter.as_deref())
            })
            .map(|(index, event)| EventPlacement {
                event: index,
                placement: light.at.map(|at| self.place(event, at)),
                map_source: self.map_source(index, light.filter.as_deref()),
                evidence_reference: light
                    .at
                    .zip(light.filter.as_deref())
                    .and_then(|(at, filter)| self.evidence_reference(event, filter, at)),
            })
            .collect();
        Ok(LightResult {
            automatic_family,
            family_probe,
            choices,
            events,
        })
    }
}

/// Assemble the same measured book for grouping and library selection.
pub fn prepare(request: Request) -> crate::Result<Book> {
    validate(&request)?;
    let imaging = if request.imaging_lights.is_empty() {
        request.imaging
    } else {
        model::imaging_sessions(request.imaging_lights, |profile| {
            request
                .profile_rigs
                .iter()
                .find(|(key, _)| key.as_deref() == profile)
                .map(|(_, rigs)| rigs.clone())
                .unwrap_or_default()
        })
    };
    let limits: HashMap<_, _> = request
        .limits
        .into_iter()
        .map(|row| (row.rig_uuid, row.limits))
        .collect();
    let nights: HashMap<_, _> = request
        .night_boundaries
        .into_iter()
        .map(|row| ((row.rig_uuid, row.frame_uuid), row.seconds))
        .collect();
    let mut book = Book::build_with_overrides(
        request.responses,
        request.comparisons,
        request.evidence,
        imaging,
        |rig| limits[rig],
        request.group_overrides,
        |rig, frame| {
            nights
                .get(&(rig.to_owned(), frame.to_owned()))
                .copied()
                .unwrap_or(model::DEFAULT_NIGHT_BOUNDARY)
        },
    );
    book.epochs = AngleEpochs::from_samples(
        request
            .epoch_samples
            .into_iter()
            .map(|row| (row.rig_uuid, row.at, row.offset))
            .collect(),
    );
    for row in request.family_evidence {
        book.family_evidence
            .entry(row.session_key.clone())
            .or_default()
            .push(row);
    }
    for row in request.light_choices {
        book.light_choices
            .entry(row.file_name.clone())
            .or_default()
            .push(row);
    }
    Ok(book)
}

fn invalid(message: impl Into<String>) -> crate::Error {
    crate::Error::FlatGroups(message.into())
}

fn validate(request: &Request) -> crate::Result<()> {
    if !request.imaging.is_empty() && !request.imaging_lights.is_empty() {
        return Err(invalid(
            "supply imaging sessions or imaging_lights, not both",
        ));
    }
    let mut profiles = BTreeSet::new();
    for (profile, _) in &request.profile_rigs {
        if !profiles.insert(profile) {
            return Err(invalid("duplicate profile_rigs binding"));
        }
    }
    let mut light_ids = BTreeSet::new();
    for (id, _, _) in &request.imaging_lights {
        if !light_ids.insert(id) {
            return Err(invalid("duplicate imaging light id"));
        }
    }
    let mut rigs = BTreeSet::new();
    for row in &request.limits {
        let l = row.limits;
        if !rigs.insert(row.rig_uuid.as_str()) {
            return Err(invalid("duplicate rig limits"));
        }
        if !l.dust.is_finite()
            || l.dust <= 0.0
            || !l.broad.is_finite()
            || l.broad <= 0.0
            || !l.rotation.is_finite()
            || l.rotation < 0.0
            || l.window < 0
            || l.window > 1_000_000_000_000
        {
            return Err(invalid(
                "limits need positive finite dust/broad, nonnegative rotation, and window in 0..=1000000000000 seconds",
            ));
        }
    }
    let mut responses = BTreeSet::new();
    let mut frames = BTreeSet::new();
    for row in &request.responses {
        if row.error.is_some() || row.captured_first.is_none() || row.captured_last.is_none() {
            continue;
        }
        if !rigs.contains(row.rig_uuid.as_str()) {
            return Err(invalid("missing rig limits"));
        }
        if !responses.insert(&row.response_key) {
            return Err(invalid("duplicate response key"));
        }
        for frame in &row.frame_uuids {
            if !frames.insert(frame) {
                return Err(invalid("a flat belongs to multiple responses"));
            }
        }
        if row.captured_first > row.captured_last {
            return Err(invalid("response ends before it starts"));
        }
    }
    // Leave ample headroom for the policy's timestamp differences and windows.
    // This accepts Unix seconds millions of years from the present, and refuses
    // integer extrema before subtraction can overflow in either build profile.
    let times = request
        .responses
        .iter()
        .flat_map(|row| [row.captured_first, row.captured_last])
        .flatten()
        .chain(request.imaging.iter().flat_map(|row| [row.start, row.end]))
        .chain(request.imaging_lights.iter().map(|(_, _, at)| *at))
        .chain(request.epoch_samples.iter().map(|row| row.at))
        .chain(
            request
                .family_evidence
                .iter()
                .flat_map(|row| [row.start, row.end]),
        )
        .chain(request.light_choices.iter().filter_map(|row| row.at))
        .chain(
            request
                .light_choices
                .iter()
                .flat_map(|row| [row.row.change_start, row.row.change_end])
                .flatten(),
        );
    if times.into_iter().any(|at| !safe_time(at)) {
        return Err(invalid("capture time is outside the supported range"));
    }
    for row in &request.lights {
        row.check()?;
    }
    let mut nights = BTreeSet::new();
    for row in &request.night_boundaries {
        if !(-86400..=129600).contains(&row.seconds)
            || !nights.insert((&row.rig_uuid, &row.frame_uuid))
        {
            return Err(invalid(
                "night boundaries need unique rig/frame keys and noon shifts within one day of noon UTC",
            ));
        }
    }
    Ok(())
}
