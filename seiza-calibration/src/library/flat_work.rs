//! PSF Guard's evidence pass as pure plans and result decisions. Hosts supply
//! cached catalog records and measured pixels; this module never opens files.
use super::*;
use crate::flat_response::{choice, evidence, groups};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FlatLightContext {
    pub light_index: usize,
    pub id: Option<i64>,
    pub profile: Option<String>,
    pub target: String,
    pub excluded: bool,
    /// Optional cached host pedestal. None uses the shared bias selection.
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
fn picked<T: Clone>(values: &[T]) -> Vec<T> {
    if values.len() <= evidence::MAX_LIGHTS {
        values.to_vec()
    } else {
        (0..evidence::MAX_LIGHTS)
            .map(|i| values[i * (values.len() - 1) / (evidence::MAX_LIGHTS - 1)].clone())
            .collect()
    }
}
struct WorkLight<'a> {
    meta: &'a Light,
    context: FlatLightContext,
    id: i64,
    pedestal: Option<f64>,
}
fn angle_groups<'a>(
    mut lights: Vec<&'a WorkLight<'a>>,
    tolerance: f64,
) -> Vec<Vec<&'a WorkLight<'a>>> {
    lights.sort_by(|a, b| {
        a.meta
            .rotator_position
            .unwrap_or(f64::NAN)
            .rem_euclid(360.0)
            .total_cmp(
                &b.meta
                    .rotator_position
                    .unwrap_or(f64::NAN)
                    .rem_euclid(360.0),
            )
            .then(a.meta.timestamp.cmp(&b.meta.timestamp))
            .then(a.id.cmp(&b.id))
    });
    let mut out: Vec<Vec<&WorkLight>> = Vec::new();
    for light in lights {
        let joins=out.last().and_then(|g|g.last()).is_some_and(|last|matches!((last.meta.rotator_position,light.meta.rotator_position),(Some(a),Some(b)) if flat_notes::lights::wrap(b-a).abs()<=tolerance));
        if joins {
            out.last_mut().unwrap().push(light);
        } else {
            out.push(vec![light]);
        }
    }
    out
}
/// Plan light-background evidence, correction inputs and resolved manual choices.
/// The complete metadata/quality/grouping snapshot is reusable without any files.
/// Return answers to `grouping.family_evidence` / `grouping.evidence` and replan.
pub fn plan_flat_work(mut request: FlatWorkRequest) -> crate::Result<FlatWorkPlans> {
    let snapshot = evaluate(request.library.clone())?;
    let rules = master_policy::rule_book(
        request.library.global_settings.clone(),
        request.library.rigs.clone(),
    )?;
    resolve_nights(&mut request.library, &rules)?;
    let mut grouping = request.library.grouping.clone();
    grouping.limits = grouping
        .responses
        .iter()
        .map(|r| r.rig_uuid.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|rig_uuid| groups::RigLimits {
            limits: rules.for_rig(&rig_uuid).grouping_limits(),
            rig_uuid,
        })
        .collect();
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
    let quality: BTreeMap<_, _> = request
        .library
        .quality
        .iter()
        .map(|r| (r.frame_uuid.as_str(), r))
        .collect();
    let mut ids = BTreeSet::new();
    let mut lights = Vec::new();
    for (index, meta) in request.library.lights.iter().enumerate() {
        let c = context.remove(&index).unwrap_or(FlatLightContext {
            light_index: index,
            ..Default::default()
        });
        let id =
            c.id.unwrap_or(i64::try_from(index).map_err(|_| invalid("too many lights"))?);
        if !ids.insert(id) {
            return Err(invalid("light context IDs must be unique"));
        }
        let pedestal = c.pedestal_adu.or_else(|| {
            let mut levels: Vec<_> = snapshot.selections[index]
                .bias
                .iter()
                .filter_map(|id| match quality.get(id.as_str())?.measurement.as_ref()? {
                    StoredMeasurement::Dark(m) => Some(m.level as f64),
                    _ => None,
                })
                .filter(|v| v.is_finite())
                .collect();
            levels.sort_by(f64::total_cmp);
            levels.get(levels.len() / 2).copied()
        });
        lights.push(WorkLight {
            meta,
            context: c,
            id,
            pedestal,
        });
    }
    if grouping.imaging.is_empty() && grouping.imaging_lights.is_empty() {
        let mut profiles: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for light in &lights {
            if light.context.excluded {
                continue;
            }
            let Some(at) = light.meta.timestamp else {
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
    let book = Arc::new(groups::prepare(grouping.clone())?);
    let mut frames = request.library.frames.clone();
    for (frame, grade) in frames.iter_mut().zip(&snapshot.grades) {
        frame.kept_out = grade.kept_out.clone();
    }
    let library = Library::new(
        frames,
        Arc::clone(&book),
        request.library.available_responses.iter().cloned(),
    );
    let eligible = |session: &groups::ImagingSession, rig: &str| {
        lights
            .iter()
            .filter(|l| {
                !l.context.excluded && l.meta.rig_uuid == rig && session.lights.contains(&l.id)
            })
            .collect::<Vec<_>>()
    };
    let mut families = Vec::new();
    let mut changes = Vec::new();
    if request.library.flat_association != Association::OneSet {
        for session in &book.imaging {
            let measured: Vec<_> = session
                .rigs
                .iter()
                .filter(|r| book.limits(r).is_some_and(|l| l.measured))
                .collect();
            let [rig] = measured.as_slice() else {
                continue;
            };
            let mut by_target: BTreeMap<(String, String, String), Vec<&WorkLight>> =
                BTreeMap::new();
            for light in eligible(session, rig) {
                // Never mix sensor modes in one background. IDs, paths and time
                // are omitted from this physical partition.
                let m = light.meta;
                let sensor = serde_json::to_string(&(
                    m.width,
                    m.height,
                    m.channels,
                    m.binning_x,
                    m.binning_y,
                    m.gain,
                    m.offset,
                    m.readout_mode,
                    &m.readout_mode_name,
                    &m.bayer_pattern,
                    m.focal_length_mm,
                ))
                .map_err(|e| invalid(e.to_string()))?;
                by_target
                    .entry((
                        groups::model::canonical_filter(m.filter.as_deref().unwrap_or_default()),
                        light.context.target.clone(),
                        sensor,
                    ))
                    .or_default()
                    .push(light);
            }
            for ((filter, target, _), members) in by_target {
                let tolerance = rules
                    .for_rig(rig)
                    .rotation_deg
                    .min(groups::ROTATION_CLUSTER_DEG);
                for group in angle_groups(members, tolerance) {
                    if group.len() < evidence::MIN_LIGHTS
                        || group.iter().any(|l| {
                            l.meta.rotator_position.is_none() || l.meta.timestamp.is_none()
                        })
                    {
                        continue;
                    }
                    let low = group[0].meta.rotator_position.unwrap();
                    let high = group.last().unwrap().meta.rotator_position.unwrap();
                    let mut angles: Vec<_> = group
                        .iter()
                        .map(|l| l.meta.rotator_position.unwrap())
                        .collect();
                    angles.sort_by(f64::total_cmp);
                    let angle = angles[angles.len() / 2];
                    let compatible = library
                        .flat_family_compatible(group[0].meta, &rules.for_light(group[0].meta));
                    let mut asked = BTreeSet::new();
                    for light in &group {
                        let candidates = book.family_candidates(
                            rig,
                            light.meta.timestamp,
                            light.meta.rotator_position.unwrap(),
                            &compatible,
                        );
                        if candidates.len() >= 2 {
                            asked.extend(candidates.into_iter().map(|(i, _)| i));
                        }
                    }
                    if asked.len() < 2 {
                        continue;
                    }
                    let indices: Vec<_> = asked.into_iter().collect();
                    let key = book.family_evidence_key(&indices, rig, session.start, angle);
                    let answered = book
                        .family_evidence
                        .get(&session.key)
                        .into_iter()
                        .flatten()
                        .any(|r| {
                            r.filter == filter
                                && r.target == target
                                && r.angle_key() == format!("{low:.1}")
                                && book.family_evidence_current(r, rig, session.start)
                        });
                    let representatives: Vec<_> = indices
                        .iter()
                        .map(|i| book.family_representative(*i, rig, session.start, angle))
                        .collect();
                    let candidates = indices
                        .iter()
                        .zip(&representatives)
                        .map(|(i, r)| {
                            let gap = book.families[*i]
                                .angles
                                .iter()
                                .map(|span| span.gap_to(angle))
                                .fold(f64::INFINITY, f64::min);
                            FlatFamilyCandidate {
                                key: book.families[*i].key.clone(),
                                response: r.map(|r| book.sessions[r].row.clone()),
                                angle_gap: gap.is_finite().then_some(gap),
                            }
                        })
                        .collect();
                    let mut alike = Vec::new();
                    for a in 0..indices.len() {
                        for b in a + 1..indices.len() {
                            if let (Some(a_rep), Some(b_rep)) =
                                (representatives[a], representatives[b])
                                && book.measured_alike(&[a_rep, b_rep])
                            {
                                alike.push((a, b));
                            }
                        }
                    }
                    families.push(FlatFamilyQuestion {
                        session_key: session.key.clone(),
                        rig_uuid: (*rig).clone(),
                        filter: filter.clone(),
                        target: target.clone(),
                        angle,
                        angle_low: low,
                        angle_high: high,
                        start: group.iter().filter_map(|l| l.meta.timestamp).min().unwrap(),
                        end: group.iter().filter_map(|l| l.meta.timestamp).max().unwrap(),
                        key,
                        answered,
                        sample_lights: if answered {
                            Vec::new()
                        } else {
                            picked(&group).iter().map(|l| l.id).collect()
                        },
                        pedestal_adu: group[0].pedestal,
                        candidates,
                        alike,
                    });
                }
            }
        }
        let needed: BTreeSet<_> = book.sessions_needing_evidence().into_iter().collect();
        for (event_index, session_index, key) in book.sessions_in_windows() {
            let event = &book.events[event_index];
            let session = &book.imaging[session_index];
            let answered = !needed.contains(&(event_index, session_index));
            let mut by_filter: BTreeMap<String, Vec<&WorkLight>> = BTreeMap::new();
            for light in eligible(session, &event.rig_uuid) {
                by_filter
                    .entry(light.meta.filter.clone().unwrap_or_default())
                    .or_default()
                    .push(light);
            }
            let mut filters = Vec::new();
            for (filter, group) in by_filter {
                if group.len() < evidence::MIN_LIGHTS || !event.applies_to(Some(&filter)) {
                    continue;
                }
                let Some(reference) = book.evidence_reference(event, &filter, session.start) else {
                    continue;
                };
                let Some(boundary) = book
                    .map_member(event_index, Some(&filter))
                    .map(|i| &book.boundaries[i])
                else {
                    continue;
                };
                let reference = &book.sessions[reference];
                let Some(reference_side) = event.side_of(reference.first) else {
                    continue;
                };
                let samples = picked(&group);
                let targets = samples
                    .iter()
                    .map(|l| l.context.target.clone())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                filters.push(FlatChangeFilter {
                    filter,
                    sample_lights: if answered {
                        Vec::new()
                    } else {
                        samples.iter().map(|l| l.id).collect()
                    },
                    targets,
                    pedestal_adu: group[0].pedestal,
                    reference: reference.row.clone(),
                    reference_side,
                    before: book.sessions[boundary.before].row.clone(),
                    after: book.sessions[boundary.after].row.clone(),
                    vignetting_changes: boundary.vignetting_changes.clone(),
                });
            }
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
    for (index, selection) in snapshot.selections.iter().enumerate() {
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
        algorithm_version: 1,
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
/// Finalize native kernel feedback, including PSF Guard's tie, missing-fit and
/// aggregate verdict policy. The caller stores the returned row unchanged.
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
            let chosen = match choice.winners.as_slice() {
                [] => None,
                [only] => Some(*only),
                _ if choice.any_unfitted() => None,
                several => several.iter().copied().min_by(|a, b| {
                    q.candidates[*a]
                        .angle_gap
                        .unwrap_or(f64::INFINITY)
                        .total_cmp(&q.candidates[*b].angle_gap.unwrap_or(f64::INFINITY))
                }),
            };
            let key_of = |i: usize| q.candidates[i].key.clone();
            let row = groups::FamilyEvidenceRow {
                session_key: q.session_key,
                filter: q.filter,
                target: q.target.clone(),
                angle: q.angle,
                angle_low: q.angle_low,
                angle_high: q.angle_high,
                start: q.start,
                end: q.end,
                key: q.key,
                candidates: (0..n).map(key_of).collect(),
                chosen: chosen.map(key_of),
                lights,
                targets: vec![q.target],
                pairs: choice
                    .pairs
                    .iter()
                    .map(|p| groups::FamilyPairRow {
                        first: key_of(p.first),
                        second: key_of(p.second),
                        outcome: match p.outcome {
                            choice::PairOutcome::Alike => "alike",
                            choice::PairOutcome::Unfitted => "unfitted",
                            choice::PairOutcome::Fit(_) => "fit",
                        }
                        .into(),
                        carried: p.fit().map(|f| f.carried),
                        spread: p
                            .fit()
                            .and_then(|f| f.spread.is_finite().then_some(f.spread)),
                        verdict: p.fit().map(|f| f.verdict),
                    })
                    .collect(),
            };
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
