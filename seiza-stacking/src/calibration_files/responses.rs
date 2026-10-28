//! Pixel I/O and caches only. Session preparation, pair requests and every
//! grouping/association decision are supplied by Seiza.
use super::*;
use seiza_calibration::flat_response::{groups, metric};

#[derive(Serialize, Deserialize)]
struct Measured {
    plane: Source,
    bin: usize,
    level: f32,
    noise: f32,
    frames: usize,
    width: usize,
    height: usize,
}
fn atomic_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let mut file = tempfile::NamedTempFile::new_in(path.parent().unwrap()).map_err(error)?;
    file.write_all(bytes).map_err(error)?;
    file.as_file().sync_all().map_err(error)?;
    file.persist(path).map_err(error)?;
    Ok(())
}
fn measure(
    plan: &policy::FlatSessionPlan,
    rows: &BTreeMap<String, &Cached>,
    cache: &Path,
    cancel: &CancelSignal,
) -> Result<Measured> {
    let key = digest(format!("response-1-{}", plan.response_key()));
    let memo = cache.join(format!("response-{key}.json"));
    if let Ok(data) = std::fs::read(&memo)
        && let Ok(value) = serde_json::from_slice::<Measured>(&data)
        && verify(std::slice::from_ref(&value.plane)).is_ok()
        && metric::decode_plane(&std::fs::read(&value.plane.path).map_err(error)?).is_some()
    {
        return Ok(value);
    }
    let mut binned = Vec::new();
    let mut bin = None;
    for sample in plan.combined() {
        let id = &sample.frame_uuid;
        check(cancel)?;
        let row = rows
            .get(id)
            .ok_or("Response sample is missing from the reviewed library")?;
        verify(&[row.source.clone(), row.processed_source.clone()])?;
        let frame = FitsFrame::open(&row.processed).map_err(error)?;
        let factor =
            metric::bin_factor(frame.image.width, frame.image.height, frame.bayer.is_some());
        if bin.is_some_and(|b| b != factor) {
            return Err("Response sample dimensions changed".into());
        }
        bin = Some(factor);
        binned.push(metric::bin_frame(
            &frame.image.data,
            frame.image.width,
            frame.image.height,
            frame.image.channels,
            factor,
        ));
    }
    check(cancel)?;
    let response = metric::build_response(&binned, bin.unwrap_or(1))
        .ok_or("Flat response could not be measured")?;
    let plane = cache.join(format!("response-{key}.bin"));
    atomic_bytes(&plane, &metric::encode_plane(&response.plane))?;
    let measured = Measured {
        plane: stamp(&plane)?,
        bin: response.bin,
        level: response.level,
        noise: response.noise,
        frames: response.frames,
        width: response.plane.width,
        height: response.plane.height,
    };
    atomic_json(&memo, &measured)?;
    Ok(measured)
}
fn corrected(row: &groups::ResponseRow, measured: &Measured) -> Result<metric::Plane> {
    verify(std::slice::from_ref(&measured.plane))?;
    let plane = metric::decode_plane(&std::fs::read(&measured.plane.path).map_err(error)?)
        .ok_or("Invalid response cache")?;
    Ok(metric::FlatResponse {
        plane,
        bin: measured.bin,
        level: measured.level,
        noise: measured.noise,
        frames: measured.frames,
    }
    .without_pedestal(row.pedestal_adu.unwrap_or(0.0) as f32))
}
pub(super) fn prepare(
    planner: &Planner,
    rows: &[Cached],
    cache: &Path,
    cancel: &CancelSignal,
    warnings: &mut Vec<String>,
) -> Result<Prepared> {
    let plans = if planner.associations.flat == policy::Association::OneSet {
        Vec::new()
    } else {
        policy::plan_flat_sessions(&planner.frames, &planner.nights, &planner.rules)
    };
    let unplanned = planner
        .frames
        .iter()
        .filter(|frame| {
            planner.associations.flat != policy::Association::OneSet
                && frame.kind == Kind::Flat
                && !frame.is_master
                && frame.kept_out.is_none()
                && frame.captured_at.is_none()
        })
        .count();
    if unplanned > 0 {
        warnings.push(format!("{unplanned} flats have no capture time; measured temporal grouping is unavailable for those files."));
    }
    let pedestals = policy::PedestalTable::new(&planner.frames, &planner.quality);
    let by_id = rows
        .iter()
        .map(|r| (r.frame.frame_uuid.clone(), r))
        .collect();
    let mut grouping = planner.grouping.clone();
    grouping.responses.clear();
    grouping.comparisons.clear();
    grouping.limits = plans
        .iter()
        .map(|p| p.rig_uuid.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|rig_uuid| groups::RigLimits {
            limits: planner.rules.for_rig(&rig_uuid).grouping_limits(),
            rig_uuid,
        })
        .collect();
    grouping.night_boundaries = planner.nights.boundaries();
    let mut measured = BTreeMap::new();
    let mut sources = Vec::new();
    for plan in plans {
        if !plan.measurable()
            || !planner
                .rules
                .for_rig(&plan.rig_uuid)
                .grouping_limits()
                .measured
        {
            continue;
        }
        check(cancel)?;
        let pedestal = pedestals.for_session(&plan);
        let mut row = plan.pending_row(pedestal);
        match measure(&plan, &by_id, cache, cancel) {
            Ok(value) => {
                let response = metric::FlatResponse {
                    plane: metric::decode_plane(&std::fs::read(&value.plane.path).map_err(error)?)
                        .ok_or("Invalid response cache")?,
                    bin: value.bin,
                    level: value.level,
                    noise: value.noise,
                    frames: value.frames,
                };
                row = plan.measured_row(
                    pedestal,
                    policy::ResponseRead {
                        read: value.frames,
                        bin: value.bin,
                        response: Some(&response),
                        unread: Vec::new(),
                    },
                );
                sources.push(value.plane.clone());
                measured.insert(row.response_key.clone(), value);
            }
            Err(e) => {
                check(cancel)?;
                warnings.push(format!("Flat response {}: {e}", row.session_key));
                row.error = Some(e);
            }
        }
        grouping.responses.push(row);
    }
    let mut tried = BTreeSet::new();
    loop {
        check(cancel)?;
        let book = groups::prepare(grouping.clone()).map_err(error)?;
        let mut pending = Vec::new();
        for (a, b) in book.pairs_to_compare() {
            let (a, b) = (&book.sessions[a].row, &book.sessions[b].row);
            if tried.insert((a.response_key.clone(), b.response_key.clone())) {
                pending.push((a.clone(), b.clone()));
            }
        }
        if pending.is_empty() {
            break;
        }
        for (a, b) in pending {
            check(cancel)?;
            let (Some(ma), Some(mb)) =
                (measured.get(&a.response_key), measured.get(&b.response_key))
            else {
                continue;
            };
            let key = digest(
                serde_json::to_vec(&(
                    metric::COMPARE_ALGORITHM_VERSION,
                    &ma.plane,
                    &mb.plane,
                    a.pedestal_adu,
                    b.pedestal_adu,
                ))
                .map_err(error)?,
            );
            let memo = cache.join(format!("comparison-{key}.json"));
            let comparison: Option<metric::Comparison> = std::fs::read(&memo)
                .ok()
                .and_then(|data| serde_json::from_slice(&data).ok());
            let comparison = match comparison {
                Some(c) => Some(c),
                None => {
                    let left = corrected(&a, ma)?;
                    let right = corrected(&b, mb)?;
                    let result = metric::compare(&left, &right, ma.bin);
                    if let Some(c) = &result {
                        atomic_json(&memo, c)?;
                    }
                    result
                }
            };
            if let Some(c) = comparison {
                grouping.comparisons.push(groups::ComparisonRow {
                    earlier_key: a.response_key,
                    later_key: b.response_key,
                    broad: f64::from(c.broad),
                    dust: f64::from(c.dust),
                    sigma: f64::from(c.sigma),
                    spots: c.spots,
                    broad_x: f64::from(c.broad_x),
                    broad_y: f64::from(c.broad_y),
                    vignetting: f64::from(c.vignetting),
                    vignetting_changes: c.vignetting_changes,
                });
            } else {
                warnings.push(format!("Flat sessions {} and {} could not be compared; their relationship remains unmeasured.",a.session_key,b.session_key));
            }
        }
    }
    Ok(Prepared {
        grouping,
        sources,
        measured,
    })
}

pub(super) struct Prepared {
    pub grouping: groups::Request,
    pub sources: Vec<Source>,
    measured: BTreeMap<String, Measured>,
}
fn read_binned(light: &Cached, cache: &Path, cancel: &CancelSignal) -> Result<metric::Plane> {
    check(cancel)?;
    verify(&[light.source.clone(), light.processed_source.clone()])?;
    let key = digest(
        serde_json::to_vec(&(
            "light-bin-1",
            &light.source,
            &light.processed_source,
            metric::RESPONSE_ALGORITHM_VERSION,
            metric::MAX_SIDE,
        ))
        .map_err(error)?,
    );
    let memo = cache.join(format!("light-bin-{key}.json"));
    if let Ok(data) = std::fs::read(&memo)
        && let Ok(source) = serde_json::from_slice::<Source>(&data)
        && verify(std::slice::from_ref(&source)).is_ok()
        && let Some(plane) = std::fs::read(&source.path)
            .ok()
            .and_then(|bytes| metric::decode_plane(&bytes))
    {
        return Ok(plane);
    }
    let frame = FitsFrame::open(&light.processed).map_err(error)?;
    let bin = metric::bin_factor(frame.image.width, frame.image.height, frame.bayer.is_some());
    let plane = metric::bin_frame(
        &frame.image.data,
        frame.image.width,
        frame.image.height,
        frame.image.channels,
        bin,
    );
    check(cancel)?;
    verify(&[light.source.clone(), light.processed_source.clone()])?;
    let path = cache.join(format!("light-bin-{key}.bin"));
    atomic_bytes(&path, &metric::encode_plane(&plane))?;
    atomic_json(&memo, &stamp(&path)?)?;
    Ok(plane)
}
fn sample_paths(
    samples: &[policy::EvidenceLight],
    lights: &[Cached],
    cache: &Path,
    cancel: &CancelSignal,
) -> Result<Vec<metric::Plane>> {
    samples
        .iter()
        .map(|sample| {
            read_binned(
                lights
                    .iter()
                    .find(|l| l.source.path == sample.path)
                    .ok_or("Missing evidence light")?,
                cache,
                cancel,
            )
        })
        .collect()
}
/// A native correction with exact response dependencies and affected lights.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Debug, Serialize)]
pub struct FlatCorrectionPlan {
    /// Affected indices in the original light order.
    pub light_indices: Vec<usize>,
    /// Native correction identity and direction.
    pub correction: policy::FlatCorrection,
    /// Exact before-side response dependency.
    pub before: groups::ResponseRow,
    /// Exact after-side response dependency.
    pub after: groups::ResponseRow,
    /// Response sampling bin factor.
    pub bin: usize,
    /// Whether the full change map divides the source flat.
    pub divide: bool,
}
fn corrections(planner: &Planner) -> Result<Vec<FlatCorrectionPlan>> {
    let library = planner.library()?;
    let book = library.flat_book();
    let mut out: Vec<FlatCorrectionPlan> = Vec::new();
    for (index, light) in planner.lights.iter().enumerate() {
        let selected = planner.select(&library, light);
        let Some(correction) = selected.flat_basis.correction() else {
            continue;
        };
        if let Some(existing) = out.iter_mut().find(|c| c.correction.key == correction.key) {
            existing.light_indices.push(index);
            continue;
        }
        let before = book
            .sessions
            .iter()
            .find(|s| s.row.response_key == correction.before_response)
            .ok_or("Missing before correction response")?
            .row
            .clone();
        let after = book
            .sessions
            .iter()
            .find(|s| s.row.response_key == correction.after_response)
            .ok_or("Missing after correction response")?
            .row
            .clone();
        out.push(FlatCorrectionPlan {
            light_indices: vec![index],
            correction: correction.clone(),
            bin: before.bin.ok_or("Missing correction binning")?,
            before,
            after,
            divide: correction.light_side
                == seiza_calibration::flat_response::evidence::Side::Before,
        });
    }
    Ok(out)
}
impl Prepared {
    fn plane(&self, row: &groups::ResponseRow) -> Result<metric::Plane> {
        corrected(
            row,
            self.measured
                .get(&row.response_key)
                .ok_or("A required response plane is unavailable")?,
        )
    }
    fn evidence_key(
        &self,
        question: &impl Serialize,
        ids: &[i64],
        lights: &[Cached],
    ) -> Result<String> {
        let sources = ids
            .iter()
            .map(|id| {
                usize::try_from(*id)
                    .ok()
                    .and_then(|i| lights.get(i))
                    .map(|l| (&l.source, &l.processed_source))
                    .ok_or_else(|| "Missing evidence source".to_owned())
            })
            .collect::<Result<Vec<_>>>()?;
        digest_key(&(
            "flat-evidence-1",
            question,
            sources,
            &self.sources,
            seiza_calibration::flat_response::evidence::EVIDENCE_ALGORITHM_VERSION,
            seiza_calibration::flat_response::choice::CHOICE_ALGORITHM_VERSION,
        ))
    }
    pub(super) fn connect(
        &mut self,
        planner: &mut Planner,
        lights: &[Cached],
        cache: &Path,
        cancel: &CancelSignal,
        warnings: &mut Vec<String>,
    ) -> Result<FlatWorkReview> {
        planner.grouping = self.grouping.clone();
        planner.available_responses = self.measured.keys().cloned().collect();
        // Mac owns the acquisition identities; Seiza forms imaging sessions.
        planner.grouping.imaging_lights = lights
            .iter()
            .enumerate()
            .filter_map(|(id, l)| {
                Some((
                    id as i64,
                    Some(l.light.rig_uuid.clone()),
                    l.light.timestamp?,
                ))
            })
            .collect();
        planner.grouping.profile_rigs = lights
            .iter()
            .map(|l| l.light.rig_uuid.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|rig| (Some(rig.clone()), vec![rig]))
            .collect();
        if planner.associations.flat != policy::Association::OneSet {
            let library = planner.library()?;
            let book = library.flat_book();
            let pedestals = policy::PedestalTable::new(&planner.frames, &planner.quality);
            for (event, session) in book.sessions_needing_evidence() {
                check(cancel)?;
                let Some(question) = policy::change_question(
                    book,
                    event,
                    session,
                    planner.evidence_lights(
                        &book.imaging[session],
                        &book.events[event].rig_uuid,
                        lights,
                    ),
                ) else {
                    continue;
                };
                let ids = question
                    .filters
                    .iter()
                    .flat_map(|f| f.lights.iter())
                    .filter_map(|l| {
                        lights
                            .iter()
                            .position(|c| c.source.path == l.path)
                            .map(|i| i as i64)
                    })
                    .collect::<Vec<_>>();
                let memo = cache.join(format!(
                    "change-evidence-{}.json",
                    self.evidence_key(
                        &(
                            "native-change-2",
                            book.evidence_key(&book.events[event], &book.imaging[session])
                        ),
                        &ids,
                        lights
                    )?
                ));
                let cached = std::fs::read(&memo)
                    .ok()
                    .and_then(|bytes| serde_json::from_slice::<groups::EvidenceRow>(&bytes).ok())
                    .filter(|r| {
                        r.session_key == book.imaging[session].key
                            && r.event_key
                                == book.evidence_key(&book.events[event], &book.imaging[session])
                    });
                let row = if let Some(row) = cached {
                    row
                } else {
                    let mut fits = Vec::new();
                    for filter in &question.filters {
                        check(cancel)?;
                        let (before, after) = filter.map_responses(book);
                        let Some(map) =
                            filter.change_map(book, &self.plane(before)?, &self.plane(after)?)
                        else {
                            continue;
                        };
                        let binned = sample_paths(&filter.lights, lights, cache, cancel)?;
                        let pedestal = filter
                            .lights
                            .first()
                            .and_then(|l| lights.iter().find(|c| c.source.path == l.path))
                            .and_then(|l| {
                                pedestals.light_pedestal(
                                    &l.light.rig_uuid,
                                    l.light.gain,
                                    l.light.offset,
                                )
                            });
                        if let Some(fit) = filter.fit(
                            book,
                            binned,
                            &self.plane(&book.sessions[filter.reference].row)?,
                            &map,
                            pedestal,
                        ) {
                            fits.push(fit);
                        }
                    }
                    let row = question.row(book, fits).map_err(error)?;
                    atomic_json(&memo, &row)?;
                    row
                };
                self.sources.push(stamp(&memo)?);
                planner.grouping.evidence.push(row);
            }
            // Rebuild the immutable book with change answers before asking
            // family questions. Their keys/representatives are core decisions.
            let library = planner.library()?;
            let book = library.flat_book();
            let rigs = book
                .imaging
                .iter()
                .flat_map(|s| s.rigs.iter().cloned())
                .collect::<BTreeSet<_>>();
            for rig in rigs {
                let questions = policy::family_questions(
                    &library,
                    &planner.rules,
                    &rig,
                    |session| planner.evidence_lights(session, &rig, lights),
                    |path| {
                        lights
                            .iter()
                            .find(|l| l.source.path == path)
                            .map(|l| l.light.clone())
                    },
                );
                for question in questions {
                    check(cancel)?;
                    if question.answered {
                        continue;
                    }
                    let sampled = question.sample().into_iter().cloned().collect::<Vec<_>>();
                    let ids = sampled
                        .iter()
                        .filter_map(|l| {
                            lights
                                .iter()
                                .position(|c| c.source.path == l.path)
                                .map(|i| i as i64)
                        })
                        .collect::<Vec<_>>();
                    let memo = cache.join(format!(
                        "family-evidence-{}.json",
                        self.evidence_key(
                            &(
                                "native-family-2",
                                &question.key,
                                &book.imaging[question.session].key,
                                &question.filter,
                                &question.target,
                                question.low,
                                question.high
                            ),
                            &ids,
                            lights
                        )?
                    ));
                    let cached = std::fs::read(&memo)
                        .ok()
                        .and_then(|b| serde_json::from_slice::<groups::FamilyEvidenceRow>(&b).ok())
                        .filter(|row| {
                            policy::RetentionKey::of(row) == question.retention_key(book)
                        });
                    let row = if let Some(row) = cached {
                        row
                    } else {
                        let row = if let Some(reps) = question.representatives(book) {
                            let planes = reps
                                .sessions()
                                .iter()
                                .map(|i| {
                                    let row = &book.sessions[*i].row;
                                    if self.measured.contains_key(&row.response_key) {
                                        self.plane(row).map(Some)
                                    } else {
                                        Ok(None)
                                    }
                                })
                                .collect::<Result<Vec<_>>>()?;
                            let binned = sample_paths(&sampled, lights, cache, cancel)?;
                            let pedestal = sampled
                                .first()
                                .and_then(|l| lights.iter().find(|c| c.source.path == l.path))
                                .and_then(|l| {
                                    pedestals.light_pedestal(
                                        &l.light.rig_uuid,
                                        l.light.gain,
                                        l.light.offset,
                                    )
                                });
                            question.decide(book, &reps, binned, planes, pedestal)
                        } else {
                            warnings.push("Flat family evidence has no measured representative; its pixel choice remains unknown.".into());
                            question.row(book, &Default::default(), 0).map_err(error)?
                        };
                        atomic_json(&memo, &row)?;
                        row
                    };
                    self.sources.push(stamp(&memo)?);
                    planner.grouping.family_evidence.push(row);
                }
            }
        }
        self.grouping = planner.grouping.clone();
        let library = planner.library()?;
        let book = library.flat_book();
        for state in &book.overrides {
            if let groups::OverrideStatus::Orphaned { reason } = &state.status {
                warnings.push(format!("Saved flat group choice: {reason}"));
            }
        }
        Ok(FlatWorkReview {
            algorithm_version: 2,
            grouping: planner.grouping.clone(),
            corrections: corrections(planner)?,
            overrides: book.overrides.clone(),
            group_edges: self.group_edges()?,
        })
    }
    pub(super) fn correct_master(
        &self,
        flat: &Built,
        planner: &Planner,
        _lights: &[Cached],
        cache: &Path,
        cancel: &CancelSignal,
        warnings: &mut Vec<String>,
    ) -> Result<Built> {
        let corrections = corrections(planner)?;
        if corrections.is_empty() {
            return Ok(flat.clone());
        }
        if corrections.len() != 1 || corrections[0].light_indices.len() != planner.lights.len() {
            return Err(
                "The executed flat correction no longer shares one calibration session".into(),
            );
        }
        let plan = &corrections[0];
        let before = self
            .measured
            .get(&plan.before.response_key)
            .ok_or("The correction's before response is missing")?;
        let after = self
            .measured
            .get(&plan.after.response_key)
            .ok_or("The correction's after response is missing")?;
        let sources = [
            flat.source.clone(),
            before.plane.clone(),
            after.plane.clone(),
        ];
        verify(&sources)?;
        check(cancel)?;
        let key = digest_key(&(
            &sources,
            &plan.correction,
            plan.bin,
            metric::CHANGE_MAP_ALGORITHM_VERSION,
            crate::OUTPUT_VERSION,
        ))?;
        let path = cache.join(format!("corrected-flat-{key}.fits"));
        let memo = path.with_extension("json");
        if let Ok(data) = std::fs::read(&memo)
            && let Ok(built) = serde_json::from_slice::<Built>(&data)
            && built.source.path == path
            && verify(std::slice::from_ref(&built.source)).is_ok()
        {
            return Ok(built);
        }
        let map = metric::change_map(&self.plane(&plan.before)?, &self.plane(&plan.after)?)
            .ok_or("The required correction map could not be measured")?;
        let mut frame = FitsFrame::open(&flat.source.path).map_err(error)?;
        metric::apply_change_map(
            &mut frame.image.data,
            frame.image.width,
            frame.image.channels,
            &map,
            plan.bin,
            plan.divide,
        );
        check(cancel)?;
        verify(&sources)?;
        // Preserve the source master's acquisition/optical metadata and flags.
        publish_frame(&path, &frame)?;
        let note = format!(
            "Applied Seiza's flat correction {} across the measured change.",
            plan.correction.key
        );
        warnings.push(note.clone());
        let mut built = flat.clone();
        built.source = stamp(&path)?;
        built.warnings.push(note);
        atomic_json(&memo, &built)?;
        Ok(built)
    }
    pub(super) fn group_edges(&self) -> Result<Vec<GroupEdge>> {
        let book = groups::prepare(self.grouping.clone()).map_err(error)?;
        let mut edges = Vec::new();
        for family in &book.families {
            for pair in family.sessions.windows(2) {
                let before = &book.sessions[pair[0]].row;
                let after = &book.sessions[pair[1]].row;
                let label = format!(
                    "{} · {} · {} → {}",
                    family.filter.as_deref().unwrap_or("Unknown filter"),
                    family
                        .rotation
                        .map(|a| format!("{a:.1}°"))
                        .unwrap_or_else(|| "Unknown angle".into()),
                    before.night,
                    after.night
                );
                let row = groups::GroupOverrideRow {
                    anchor_frame: after
                        .frame_uuids
                        .first()
                        .cloned()
                        .ok_or("Flat session has no anchor")?,
                    session_key: after.session_key.clone(),
                    family_key: after.family_key.clone(),
                    rig_uuid: after.rig_uuid.clone(),
                    choice: None,
                    before_frame: before.frame_uuids.first().cloned(),
                    before_session_key: Some(before.session_key.clone()),
                    label: label.clone(),
                    chosen_at: 0,
                    night: Some(after.night.clone()),
                    before_night: Some(before.night.clone()),
                    rotation: after.rotation,
                    before_rotation: before.rotation,
                };
                edges.push(GroupEdge {
                    id: after.session_key.clone(),
                    label,
                    row,
                });
            }
        }
        Ok(edges)
    }
    pub(super) fn light_options(&self, lights: &[Cached]) -> Result<Vec<LightOptions>> {
        let book = groups::prepare(self.grouping.clone()).map_err(error)?;
        Ok(lights
            .iter()
            .map(|light| {
                let changes = book
                    .events_for(&light.light.rig_uuid, light.light.filter.as_deref())
                    .map(|event| ChangeOption {
                        key: event.key.clone(),
                        start: event.start,
                        end: event.end,
                        rig_uuid: event.rig_uuid.clone(),
                    })
                    .collect();
                let mut groups = Vec::new();
                for family in book
                    .families
                    .iter()
                    .filter(|f| f.rig_uuid == light.light.rig_uuid)
                {
                    for (index, group) in family.groups.iter().enumerate() {
                        if let Some(row) = group.first().map(|i| &book.sessions[*i].row) {
                            groups.push(GroupOption {
                                frame: row.frame_uuids.first().cloned(),
                                session_key: row.session_key.clone(),
                                label: format!(
                                    "{} · {:.1}° · group {}",
                                    family.filter.as_deref().unwrap_or("Unknown filter"),
                                    family.rotation.unwrap_or(0.0),
                                    index + 1
                                ),
                            });
                        }
                    }
                }
                LightOptions {
                    path: light.source.path.clone(),
                    fingerprint: light.source.fingerprint.clone(),
                    guid: light.frame.frame_uuid.clone(),
                    rig_uuid: light.light.rig_uuid.clone(),
                    changes,
                    groups,
                }
            })
            .collect())
    }
}
fn digest_key(value: &impl Serialize) -> Result<String> {
    Ok(digest(serde_json::to_vec(value).map_err(error)?))
}

/// Evidence and corrections, with the exact native rows used for selection.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Debug, Serialize)]
pub struct FlatWorkReview {
    /// Version of the native planning records.
    pub algorithm_version: u32,
    /// The measured native grouping used for this review.
    pub grouping: groups::Request,
    /// Corrections that must be applied before stacking.
    pub corrections: Vec<FlatCorrectionPlan>,
    /// Applied/orphaned native group edge commands.
    pub overrides: Vec<groups::OverrideState>,
    /// Native edge rows a host can offer to split/join.
    pub group_edges: Vec<GroupEdge>,
}
/// A native edge on which a host can offer split/join commands.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Debug, Serialize)]
pub struct GroupEdge {
    /// Stable native identity.
    pub id: String,
    /// Human-readable native capture/group label.
    pub label: String,
    /// The unchanged native command row, resolved by Seiza.
    pub row: groups::GroupOverrideRow,
}
/// Applicable native choices for one source, bound to its reviewed identity.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Debug, Serialize)]
pub struct LightOptions {
    /// Canonical source path.
    pub path: PathBuf,
    /// Metadata identity captured when this file was inspected.
    pub fingerprint: String,
    /// Stable identity of this reviewed light.
    pub guid: String,
    /// Native camera/telescope rig identity.
    pub rig_uuid: String,
    /// Applicable native change commands.
    pub changes: Vec<ChangeOption>,
    /// Measured native groups available for review.
    pub groups: Vec<GroupOption>,
}
/// A measured optical change to which a personal side choice can refer.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Debug, Serialize)]
pub struct ChangeOption {
    /// Stable native event or evidence identity.
    pub key: String,
    /// First boundary time, Unix seconds.
    pub start: i64,
    /// Last boundary time, Unix seconds.
    pub end: i64,
    /// Native camera/telescope rig identity.
    pub rig_uuid: String,
}
/// A native measured group a personal choice can name.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Debug, Serialize)]
pub struct GroupOption {
    /// A flat frame anchoring this native group.
    pub frame: Option<String>,
    /// Native session identity anchoring this choice.
    pub session_key: String,
    /// Human-readable native capture/group label.
    pub label: String,
}
