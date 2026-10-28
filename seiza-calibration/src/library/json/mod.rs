//! The C ABI's calibration library calls: one JSON request runs a whole
//! workflow and returns one answer, so the C ABI needs few exports.
//!
//! Each call only checks its request and runs the native API on it
//! ([`Library`], [`grade`], [`resolve_session`], ...). Rust hosts call the
//! native API themselves, and Python binds it: neither uses these.
use super::*;

mod flat_work;
mod masters;
mod responses;

pub use flat_work::*;
pub use masters::*;
pub use responses::*;

/// A master that failed to build, by its session's key.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MasterFailure {
    pub session_key: String,
    pub kind: CalibrationKind,
    pub reason: String,
}
/// Successful per-light fits from fit_flat_pedestal, in the returned sample
/// set, by session key ([`Session::accept_pedestal`]). A pinned value
/// reproduces a previous stack and wins over fresh estimates.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PedestalEvidence {
    pub session_key: String,
    pub estimates: Vec<f32>,
    pub pinned: Option<f32>,
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
impl Request {
    fn associations(&self) -> Associations {
        Associations {
            flat: self.flat_association,
            dark: self.dark_association,
        }
    }
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
    pub measurement_state: MeasurementState,
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
    let rules = masters::rule_book(global, rigs)?;
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
    let judged = if request.regrade {
        quality.regrade(&request.frames, &rules, &nights)
    } else {
        HashMap::new()
    };
    let grades = apply_grades(&mut request.frames, &quality, &judged);
    let flat_book = match prepared {
        Some(book) => book,
        None => Arc::new(groups::prepare(std::mem::take(&mut request.grouping))?),
    };
    let associations = request.associations();
    let library = Library::new(request.frames, flat_book, request.available_responses);
    library.check_associations(&rules, associations)?;
    let options = SessionOptions {
        flat_star_masking: request.flat_star_masking,
        flat_stability: request.flat_stability,
    };
    let mut selections = Vec::new();
    let mut sessions: Vec<Session> = Vec::new();
    let mut assignments = Vec::new();
    let mut keys = BTreeMap::new();
    for light in &request.lights {
        let selected =
            library.select_associated(light, rules.for_light(light), associations);
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
    quality.keep_out(frames, judged);
    frames
        .iter()
        .map(|frame| FrameGrade {
            frame_uuid: frame.frame_uuid.clone(),
            judged: judged.get(&frame.frame_uuid).cloned(),
            kept_out: frame.kept_out.clone(),
            measurement_state: quality.measurement_state(frame),
            measure_error: quality
                .get(&frame.frame_uuid)
                .and_then(|r| r.measure_error.clone()),
        })
        .collect()
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
    request.grouping.night_boundaries = nights.boundaries();
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

/// Apply a request's execution feedback to its sessions: failures, then
/// each session's flat use for its own lights, then pedestal fits.
fn finalize(
    sessions: &mut [Session],
    assignments: &[usize],
    lights: &[Light],
    failures: &[MasterFailure],
    evidence: &[PedestalEvidence],
    force: bool,
) -> crate::Result<()> {
    let without_masters = if force {
        FlatWithoutMasters::Force
    } else {
        FlatWithoutMasters::FitPedestal
    };
    let keys: BTreeSet<_> = sessions.iter().map(|s| s.key.as_str()).collect();
    let mut failed = BTreeSet::new();
    for row in failures {
        if !keys.contains(row.session_key.as_str()) || !failed.insert((&row.session_key, row.kind))
        {
            return Err(invalid(
                "master failures must name unique session/kind recipes",
            ));
        }
    }
    let mut fits = BTreeSet::new();
    for row in evidence {
        if !keys.contains(row.session_key.as_str()) || !fits.insert(&row.session_key) {
            return Err(invalid("pedestal evidence needs a unique session"));
        }
    }
    for (index, session) in sessions.iter_mut().enumerate() {
        // In build order, so a failure blocks what was built after it.
        for kind in [
            CalibrationKind::Bias,
            CalibrationKind::DarkFlat,
            CalibrationKind::Dark,
            CalibrationKind::Flat,
        ] {
            if let Some(row) = failures
                .iter()
                .find(|row| row.session_key == session.key && row.kind == kind)
            {
                session.record_failure(kind, row.reason.clone())?;
            }
        }
        let own = lights
            .iter()
            .zip(assignments)
            .filter(|(_, a)| **a == index)
            .map(|(l, _)| l);
        session.plan_flat_use(own, without_masters);
        if let Some(row) = evidence.iter().find(|row| row.session_key == session.key) {
            session.accept_pedestal(&row.estimates, row.pinned)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
