//! The C ABI's flat work calls: the family and change questions of a
//! library request's lights, on the native stages.
use super::*;
use crate::flat_response::{choice, evidence};
use crate::library::flat_work::{FamilyRowOf, angle_gap, family_row};
use std::path::Path;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FlatLightContext {
    pub light_index: usize,
    pub id: Option<i64>,
    pub profile: Option<String>,
    pub target: String,
    pub excluded: bool,
    /// Optional cached host pedestal. None uses the library's bias level at
    /// the light's gain and offset.
    pub pedestal_adu: Option<f64>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FlatWorkRequest {
    pub library: Request,
    pub light_context: Vec<FlatLightContext>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlatFamilyCandidate {
    pub key: String,
    pub response: Option<groups::ResponseRow>,
    pub angle_gap: Option<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlatFamilyQuestion {
    pub session_key: String,
    pub rig_uuid: String,
    pub filter: String,
    pub target: String,
    pub angle: f64,
    pub angle_low: f64,
    pub angle_high: f64,
    pub start: i64,
    pub end: i64,
    pub key: String,
    pub answered: bool,
    /// IDs refer to FlatLightContext, defaulting to the library light's index.
    pub sample_lights: Vec<i64>,
    pub pedestal_adu: Option<f64>,
    pub candidates: Vec<FlatFamilyCandidate>,
    pub alike: Vec<(usize, usize)>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlatChangeFilter {
    pub filter: String,
    pub sample_lights: Vec<i64>,
    pub targets: Vec<String>,
    pub pedestal_adu: Option<f64>,
    pub reference: groups::ResponseRow,
    pub reference_side: evidence::Side,
    pub before: groups::ResponseRow,
    pub after: groups::ResponseRow,
    pub vignetting_changes: Vec<crate::dust::VignettingChange>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlatChangeQuestion {
    pub session_key: String,
    pub event_key: String,
    pub rig_uuid: String,
    pub answered: bool,
    pub filters: Vec<FlatChangeFilter>,
}
#[derive(Clone, Debug, Serialize)]
pub struct FlatCorrectionPlan {
    pub light_indices: Vec<usize>,
    pub correction: FlatCorrection,
    pub before: groups::ResponseRow,
    pub after: groups::ResponseRow,
    pub bin: usize,
    pub divide: bool,
}
#[derive(Debug, Serialize)]
pub struct FlatWorkPlans {
    pub algorithm_version: u32,
    /// Feed evidence answers back into this resolved request. All manual rows
    /// and catalog imaging sessions are preserved.
    pub grouping: groups::Request,
    pub families: Vec<FlatFamilyQuestion>,
    pub changes: Vec<FlatChangeQuestion>,
    pub corrections: Vec<FlatCorrectionPlan>,
    pub overrides: Vec<groups::OverrideState>,
}

/// A request light with the context the evidence stages read.
struct WorkLight<'a> {
    meta: &'a Light,
    context: FlatLightContext,
    id: i64,
}

/// Plan light-background evidence, correction inputs and resolved manual
/// choices from a library request whose lights carry their metadata, so no
/// file is read. Return the answers to `grouping.family_evidence` /
/// `grouping.evidence` of the returned grouping and plan again.
pub fn plan_flat_work(mut request: FlatWorkRequest) -> crate::Result<FlatWorkPlans> {
    let mut context = BTreeMap::new();
    for row in request.light_context {
        if row.light_index >= request.library.lights.len()
            || row.pedestal_adu.is_some_and(|v| !v.is_finite())
            || context.insert(row.light_index, row).is_some()
        {
            return Err(invalid(
                "light contexts must name unique existing light indices and finite pedestals",
            ));
        }
    }
    let all_lights = std::mem::take(&mut request.library.lights);
    for light in &all_lights {
        validate_values(&light_signature(light).seiza)?;
    }
    let mut ids = BTreeSet::new();
    let mut lights = Vec::new();
    for (index, meta) in all_lights.iter().enumerate() {
        let context = context.remove(&index).unwrap_or(FlatLightContext {
            light_index: index,
            ..Default::default()
        });
        let id = context
            .id
            .unwrap_or(i64::try_from(index).map_err(|_| invalid("too many lights"))?);
        if !ids.insert(id) {
            return Err(invalid("light context IDs must be unique"));
        }
        lights.push(WorkLight { meta, context, id });
    }
    // Imaging sessions from the lights themselves, unless the host gave its
    // catalog's.
    let mut grouping = std::mem::take(&mut request.library.grouping);
    if grouping.imaging.is_empty() && grouping.imaging_lights.is_empty() {
        let mut profiles: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for light in &lights {
            let Some(at) = light.meta.timestamp.filter(|_| !light.context.excluded) else {
                continue;
            };
            let profile = light
                .context
                .profile
                .clone()
                .unwrap_or_else(|| light.meta.rig_uuid.clone());
            profiles
                .entry(profile.clone())
                .or_default()
                .insert(light.meta.rig_uuid.clone());
            grouping.imaging_lights.push((light.id, Some(profile), at));
        }
        grouping.profile_rigs = profiles
            .into_iter()
            .map(|(p, r)| (Some(p), r.into_iter().collect()))
            .collect();
    }
    let mut library_request = request.library;
    library_request.grouping = grouping;
    let Checked {
        rules,
        nights,
        mut quality,
    } = checked(&mut library_request)?;
    let judged = if library_request.regrade {
        let measured = library_request.frames.iter().filter_map(|frame| {
            let row = quality.get(&frame.frame_uuid)?;
            QualityIndex::current(frame, row).then_some((frame, row.measurement.as_ref()?))
        });
        grade(measured, &rules, &nights)
    } else {
        HashMap::new()
    };
    quality.fresh = judged.keys().cloned().collect();
    apply_grades(&mut library_request.frames, &quality, &judged);
    let pedestals = response_plan::PedestalTable::new(&library_request.frames, &quality);
    let grouping = library_request.grouping.clone();
    let book = Arc::new(groups::prepare(grouping.clone())?);
    let library = Library::new(
        std::mem::take(&mut library_request.frames),
        Arc::clone(&book),
        library_request.available_responses.iter().cloned(),
    );
    let by_id: BTreeMap<i64, &WorkLight> = lights.iter().map(|l| (l.id, l)).collect();
    let by_path: BTreeMap<&Path, &WorkLight> =
        lights.iter().map(|l| (l.meta.path.as_path(), l)).collect();
    let evidence_lights = |session: &groups::ImagingSession, rig: &str| -> Vec<EvidenceLight> {
        session
            .lights
            .iter()
            .filter_map(|id| by_id.get(id))
            .filter(|l| !l.context.excluded && l.meta.rig_uuid == rig)
            .filter_map(|l| {
                Some(EvidenceLight {
                    path: l.meta.path.clone(),
                    at: l.meta.timestamp?,
                    angle: l.meta.rotator_position.filter(|a| a.is_finite()),
                    filter: l.meta.filter.clone().unwrap_or_default(),
                    target: l.context.target.clone(),
                })
            })
            .collect()
    };
    let id_of = |path: &Path| by_path.get(path).map(|l| l.id);
    let pedestal_of = |path: &Path| {
        let light = by_path.get(path)?;
        light.context.pedestal_adu.or_else(|| {
            pedestals.pedestal_for(
                &light.meta.rig_uuid,
                light.meta.gain,
                light.meta.offset,
                None,
            )
        })
    };
    let mut families = Vec::new();
    let mut changes = Vec::new();
    if library_request.flat_association != Association::OneSet {
        let rigs: BTreeSet<String> = book
            .imaging
            .iter()
            .flat_map(|session| session.rigs.iter().cloned())
            .collect();
        for rig in &rigs {
            let questions = family_questions(
                &library,
                &rules,
                rig,
                &mut |session| evidence_lights(session, rig),
                &mut |path| {
                    by_path
                        .get(path)
                        .map(|l| l.meta.clone())
                        .unwrap_or_default()
                },
            );
            for question in questions {
                let representatives = question.representatives(&book);
                let candidates: Vec<FlatFamilyCandidate> = question
                    .candidates
                    .iter()
                    .enumerate()
                    .map(|(position, family)| {
                        let gap = angle_gap(&book, *family, question.angle);
                        FlatFamilyCandidate {
                            key: book.families[*family].key.clone(),
                            response: representatives
                                .as_ref()
                                .map(|reps| book.sessions[reps[position]].row.clone()),
                            angle_gap: gap.is_finite().then_some(gap),
                        }
                    })
                    .collect();
                let mut alike = Vec::new();
                if let Some(reps) = &representatives {
                    for a in 0..reps.len() {
                        for b in a + 1..reps.len() {
                            if book.measured_alike(&[reps[a], reps[b]]) {
                                alike.push((a, b));
                            }
                        }
                    }
                }
                let sample = question.sample();
                families.push(FlatFamilyQuestion {
                    session_key: book.imaging[question.session].key.clone(),
                    rig_uuid: question.rig.clone(),
                    filter: question.filter.clone(),
                    target: question.target.clone(),
                    angle: question.angle,
                    angle_low: question.low,
                    angle_high: question.high,
                    start: question.start,
                    end: question.end,
                    key: question.key.clone(),
                    answered: question.answered,
                    sample_lights: if representatives.is_some() {
                        sample.iter().filter_map(|l| id_of(&l.path)).collect()
                    } else {
                        Vec::new()
                    },
                    pedestal_adu: sample.first().and_then(|l| pedestal_of(&l.path)),
                    candidates,
                    alike,
                });
            }
        }
        let needed: BTreeSet<_> = book.sessions_needing_evidence().into_iter().collect();
        for (event_index, session_index, key) in book.sessions_in_windows() {
            let event = &book.events[event_index];
            let session = &book.imaging[session_index];
            let answered = !needed.contains(&(event_index, session_index));
            let Some(question) = change_question(
                &book,
                event_index,
                session_index,
                evidence_lights(session, &event.rig_uuid),
            ) else {
                continue;
            };
            let filters = question
                .filters
                .iter()
                .map(|filter| {
                    let boundary = &book.boundaries[filter.map];
                    let (before, after) = filter.map_responses(&book);
                    let mut targets: Vec<String> =
                        filter.lights.iter().map(|l| l.target.clone()).collect();
                    targets.sort();
                    targets.dedup();
                    FlatChangeFilter {
                        filter: filter.filter.clone(),
                        sample_lights: if answered {
                            Vec::new()
                        } else {
                            filter
                                .lights
                                .iter()
                                .filter_map(|l| id_of(&l.path))
                                .collect()
                        },
                        targets,
                        pedestal_adu: filter.lights.first().and_then(|l| pedestal_of(&l.path)),
                        reference: book.sessions[filter.reference].row.clone(),
                        reference_side: filter.reference_side,
                        before: before.clone(),
                        after: after.clone(),
                        vignetting_changes: boundary.vignetting_changes.clone(),
                    }
                })
                .collect();
            changes.push(FlatChangeQuestion {
                session_key: session.key.clone(),
                event_key: key,
                rig_uuid: event.rig_uuid.clone(),
                answered,
                filters,
            });
        }
    }
    let mut corrections: Vec<FlatCorrectionPlan> = Vec::new();
    for (index, light) in all_lights.iter().enumerate() {
        let selection = library.select_associated(
            light,
            rules.for_light(light),
            library_request.flat_association,
            library_request.dark_association,
        );
        let Some(correction) = selection.flat_basis.correction() else {
            continue;
        };
        if let Some(existing) = corrections
            .iter_mut()
            .find(|c| c.correction.key == correction.key)
        {
            existing.light_indices.push(index);
            continue;
        }
        let (Some(before), Some(after)) = (
            book.session_by_response(&correction.before_response),
            book.session_by_response(&correction.after_response),
        ) else {
            return Err(invalid("correction response no longer exists"));
        };
        let before = book.sessions[before].row.clone();
        let after = book.sessions[after].row.clone();
        corrections.push(FlatCorrectionPlan {
            light_indices: vec![index],
            correction: correction.clone(),
            bin: before.bin.unwrap_or(1),
            divide: correction.light_side == evidence::Side::Before,
            before,
            after,
        });
    }
    let overrides = groups::evaluate(grouping.clone())?.overrides;
    Ok(FlatWorkPlans {
        algorithm_version: 2,
        grouping,
        families,
        changes,
        corrections,
        overrides,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FlatWorkAnswer {
    Family {
        question: FlatFamilyQuestion,
        choice: choice::ResponseChoice,
        lights: usize,
    },
    Change {
        question: FlatChangeQuestion,
        fits: Vec<groups::EvidenceFit>,
    },
}
#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FlatWorkResult {
    Family { row: groups::FamilyEvidenceRow },
    Change { row: groups::EvidenceRow },
}
/// The row a host keeps for a question it answered with the native
/// kernels: the family stage's tie and missing-fit decisions, or a
/// change's combined verdict.
pub fn finish_flat_work(answer: FlatWorkAnswer) -> crate::Result<FlatWorkResult> {
    match answer {
        FlatWorkAnswer::Family {
            question: q,
            choice,
            lights,
        } => {
            let n = q.candidates.len();
            let mut pairs = BTreeSet::new();
            let expected_pairs = n
                .checked_mul(n.saturating_sub(1))
                .and_then(|v| v.checked_div(2));
            if lights == 0
                || (lights < evidence::MIN_LIGHTS && !choice.winners.is_empty())
                || lights > evidence::MAX_LIGHTS
                || n < 2
                || (choice.pairs.is_empty() && !choice.winners.is_empty())
                || (!choice.pairs.is_empty() && expected_pairs != Some(choice.pairs.len()))
                || choice
                    .pairs
                    .iter()
                    .any(|p| !pairs.insert((p.first, p.second)))
                || choice.winners.iter().any(|i| *i >= n)
                || choice
                    .pairs
                    .iter()
                    .any(|p| p.first >= n || p.second >= n || p.first >= p.second)
            {
                return Err(invalid(
                    "family feedback indices or light count exceed the question",
                ));
            }
            let keys: Vec<String> = q.candidates.iter().map(|c| c.key.clone()).collect();
            let gaps: Vec<f64> = q
                .candidates
                .iter()
                .map(|c| c.angle_gap.unwrap_or(f64::INFINITY))
                .collect();
            let row = family_row(
                FamilyRowOf {
                    session_key: &q.session_key,
                    filter: &q.filter,
                    target: &q.target,
                    angle: q.angle,
                    low: q.angle_low,
                    high: q.angle_high,
                    start: q.start,
                    end: q.end,
                    key: &q.key,
                },
                &keys,
                &gaps,
                &choice,
                lights,
            );
            Ok(FlatWorkResult::Family { row })
        }
        FlatWorkAnswer::Change { question, fits } => {
            let mut filters = BTreeSet::new();
            if fits.iter().any(|fit| {
                fit.lights > evidence::MAX_LIGHTS
                    || !fit.carried.is_finite()
                    || fit.spread.is_some_and(|v| !v.is_finite())
                    || !question.filters.iter().any(|q| q.filter == fit.filter)
                    || !filters.insert(fit.filter.clone())
            }) {
                return Err(invalid(
                    "change fits must uniquely name question filters with finite measurements and bounded light counts",
                ));
            }
            let verdict = evidence::combine(&fits.iter().map(|f| f.verdict).collect::<Vec<_>>());
            Ok(FlatWorkResult::Change {
                row: groups::EvidenceRow {
                    session_key: question.session_key,
                    event_key: question.event_key,
                    verdict,
                    fits,
                },
            })
        }
    }
}
