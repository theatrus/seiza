//! The light-evidence stages of flat grouping: which response family an
//! imaging session's lights follow where their angle points at several,
//! and which side of a change lights shot during it stand on. The planning
//! opens no file. Hosts give each imaging session's lights, read the
//! headers and pixels these ask for, and keep the rows they return.
//!
//! Extracted from theatrus/psf-guard at 79f5e31d, `calibration/flat_groups/family_job.rs`
//! and `job.rs`. Copyright the PSF Guard contributors; Apache-2.0.
use super::*;
use crate::flat_response::metric::Plane;
use crate::flat_response::{choice, evidence, groups};
use std::path::Path;

/// One light of an imaging session, as the evidence stages ask it.
#[derive(Clone, Debug)]
pub struct EvidenceLight {
    pub path: PathBuf,
    pub at: i64,
    /// Its mechanical angle, when the capture recorded one.
    pub angle: Option<f64>,
    /// As captured; compared canonically.
    pub filter: String,
    pub target: String,
}

/// At most [`evidence::MAX_LIGHTS`] of `values`, evenly spaced, the first
/// and last always among them.
fn picked<T>(values: &[T]) -> Vec<&T> {
    if values.len() <= evidence::MAX_LIGHTS {
        values.iter().collect()
    } else {
        (0..evidence::MAX_LIGHTS)
            .map(|i| &values[i * (values.len() - 1) / (evidence::MAX_LIGHTS - 1)])
            .collect()
    }
}

/// Groups of lights within `tolerance` of each other's angle, around the
/// circle, each in angle order.
fn angle_groups(mut lights: Vec<EvidenceLight>, tolerance: f64) -> Vec<Vec<EvidenceLight>> {
    lights.sort_by(|a, b| {
        a.angle
            .unwrap_or(f64::NAN)
            .rem_euclid(360.0)
            .total_cmp(&b.angle.unwrap_or(f64::NAN).rem_euclid(360.0))
            .then(a.at.cmp(&b.at))
    });
    let mut groups: Vec<Vec<EvidenceLight>> = Vec::new();
    for light in lights {
        let joins = groups.last().and_then(|group| group.last()).is_some_and(|last| {
            matches!((last.angle, light.angle), (Some(a), Some(b)) if flat_groups::lights::wrap(b - a).abs() <= tolerance)
        });
        if joins {
            groups.last_mut().expect("joined").push(light);
        } else {
            groups.push(vec![light]);
        }
    }
    groups
}

/// One group of an imaging session's lights the family stage asks: one
/// filter, one target, one recorded angle, with two or more candidate
/// families. One whose answer is kept under its current key needs nothing
/// read.
#[derive(Clone, Debug)]
pub struct FamilyQuestion {
    /// Its imaging session, by index in the book.
    pub session: usize,
    pub rig: String,
    /// Canonical.
    pub filter: String,
    pub target: String,
    /// In angle order; empty when answered.
    pub lights: Vec<EvidenceLight>,
    /// Candidate families, by index in the book; empty when answered.
    pub candidates: Vec<usize>,
    /// The lights' middle, lowest and highest recorded angles.
    pub angle: f64,
    pub low: f64,
    pub high: f64,
    pub start: i64,
    pub end: i64,
    pub key: String,
    pub answered: bool,
}

/// The family questions of one rig's imaging sessions: per session, its
/// lights by filter and target ([`evidence::MIN_LIGHTS`] or more), then by
/// angle within the rig's rotation tolerance. `lights_of` gives a
/// session's lights; `header` reads one light's header, asked only for the
/// first of a group that needs one: to fill angles its capture did not
/// record, and for the sensor and optics of a question not yet answered.
pub fn family_questions(
    library: &Library,
    rules: &RuleBook,
    rig: &str,
    lights_of: &mut dyn FnMut(&groups::ImagingSession) -> Vec<EvidenceLight>,
    header: &mut dyn FnMut(&Path) -> Light,
) -> Vec<FamilyQuestion> {
    let book = library.flat_book();
    let mut out = Vec::new();
    let tolerance = rules
        .for_rig(rig)
        .rotation_deg
        .min(groups::ROTATION_CLUSTER_DEG);
    for (index, session) in book.imaging.iter().enumerate() {
        let measured: Vec<&String> = session
            .rigs
            .iter()
            .filter(|rig| book.limits(rig).is_some_and(|limits| limits.measured))
            .collect();
        let [only] = measured.as_slice() else {
            continue;
        };
        if only.as_str() != rig {
            continue;
        }
        let mut by_target: BTreeMap<(String, String), Vec<EvidenceLight>> = BTreeMap::new();
        for light in lights_of(session) {
            by_target
                .entry((light.filter.clone(), light.target.clone()))
                .or_default()
                .push(light);
        }
        let kept = book.family_evidence.get(&session.key);
        for ((filter, target), mut lights) in by_target {
            if lights.len() < evidence::MIN_LIGHTS {
                continue;
            }
            // Lights the capture recorded no angle for: their header's.
            if lights.iter().any(|light| light.angle.is_none()) {
                let meta = header(&lights[0].path);
                for light in lights.iter_mut().filter(|light| light.angle.is_none()) {
                    light.angle = meta.rotator_position;
                }
            }
            let canonical = groups::model::canonical_filter(&filter);
            for group in angle_groups(lights, tolerance) {
                let angles: Vec<f64> = group.iter().filter_map(|light| light.angle).collect();
                if group.len() < evidence::MIN_LIGHTS || angles.len() != group.len() {
                    continue;
                }
                let mut sorted = angles.clone();
                sorted.sort_by(f64::total_cmp);
                let (low, high, angle) = (
                    group[0].angle.unwrap_or_default(),
                    group[group.len() - 1].angle.unwrap_or_default(),
                    sorted[sorted.len() / 2],
                );
                let (start, end) = (
                    group.iter().map(|light| light.at).min().unwrap_or_default(),
                    group.iter().map(|light| light.at).max().unwrap_or_default(),
                );
                let place = |row: &groups::FamilyEvidenceRow| {
                    row.filter == canonical
                        && row.target == target
                        && row.angle_key() == format!("{low:.1}")
                };
                // Kept and current: nothing to read, not even a header.
                if let Some(row) = kept
                    .into_iter()
                    .flatten()
                    .find(|row| place(row) && book.family_evidence_current(row, rig, session.start))
                {
                    out.push(FamilyQuestion {
                        session: index,
                        rig: rig.to_string(),
                        filter: canonical.clone(),
                        target: target.clone(),
                        lights: Vec::new(),
                        candidates: Vec::new(),
                        angle: row.angle,
                        low: row.angle_low,
                        high: row.angle_high,
                        start: row.start,
                        end: row.end,
                        key: row.key.clone(),
                        answered: true,
                    });
                    continue;
                }
                // The first light's header: its sensor, optics and rig.
                let mut meta = header(&group[0].path);
                if meta.rig_uuid != rig {
                    continue;
                }
                meta.filter = Some(filter.clone());
                let compatible = library.flat_family_compatible(&meta, &rules.for_light(&meta));
                if compatible.len() < 2 {
                    continue;
                }
                let mut asked: BTreeSet<usize> = BTreeSet::new();
                for light in &group {
                    let candidates = book.family_candidates(
                        rig,
                        Some(light.at),
                        light.angle.unwrap_or(angle),
                        &compatible,
                    );
                    if candidates.len() >= 2 {
                        asked.extend(candidates.iter().map(|(family, _)| *family));
                    }
                }
                if asked.len() < 2 {
                    continue;
                }
                let candidates: Vec<usize> = asked.into_iter().collect();
                let key = book.family_evidence_key(&candidates, rig, session.start, angle);
                out.push(FamilyQuestion {
                    session: index,
                    rig: rig.to_string(),
                    filter: canonical.clone(),
                    target: target.clone(),
                    lights: group,
                    candidates,
                    angle,
                    low,
                    high,
                    start,
                    end,
                    key,
                    answered: false,
                });
            }
        }
    }
    out
}

impl FamilyQuestion {
    /// Where its row is kept: the session, filter, target, angle and key a
    /// store retains it under.
    pub fn retention_key(&self, book: &groups::Book) -> (String, String, String, String, String) {
        (
            book.imaging[self.session].key.clone(),
            self.filter.clone(),
            self.target.clone(),
            format!("{:.1}", self.low),
            self.key.clone(),
        )
    }

    /// The flat session standing for each candidate, nearest the lights in
    /// time; `None` when one has none, and the question cannot be weighed.
    pub fn representatives(&self, book: &groups::Book) -> Option<Vec<usize>> {
        let start = book.imaging[self.session].start;
        let representatives: Vec<usize> = self
            .candidates
            .iter()
            .filter_map(|family| book.family_representative(*family, &self.rig, start, self.angle))
            .collect();
        (representatives.len() == self.candidates.len()).then_some(representatives)
    }

    /// The lights to read: at most [`evidence::MAX_LIGHTS`], evenly spaced.
    pub fn sample(&self) -> Vec<&EvidenceLight> {
        picked(&self.lights)
    }

    /// Answer from the sampled lights' binned planes and each
    /// representative's response plane, pedestal out (`None` when not on
    /// disk: an empty plane, never weighed, so nothing wins past it). The
    /// lights' background is fitted against every pair of candidates.
    pub fn decide(
        &self,
        book: &groups::Book,
        representatives: &[usize],
        binned: Vec<Plane>,
        planes: Vec<Option<Plane>>,
        pedestal: f32,
    ) -> groups::FamilyEvidenceRow {
        let planes: Vec<Plane> = planes
            .into_iter()
            .map(|plane| plane.unwrap_or_else(|| Plane::filled(0, 0, 1.0)))
            .collect();
        let (width, height) = binned
            .first()
            .map_or((0, 0), |plane| (plane.width, plane.height));
        let binned: Vec<Plane> = binned
            .into_iter()
            .filter(|plane| plane.width == width && plane.height == height)
            .collect();
        let bin = book.sessions[representatives[0]].row.bin.unwrap_or(1);
        let picked = match evidence::background(&binned, pedestal, bin) {
            Some(background) => choice::choose_response(&background, &planes, |a, b| {
                book.measured_alike(&[representatives[a], representatives[b]])
            }),
            None => choice::ResponseChoice::default(),
        };
        self.row(book, &picked, binned.len())
    }

    /// The row a choice among this question's candidates makes. Several
    /// winners alike with each other: the nearest of them to the lights'
    /// angle stands for them; a pair the background could not be weighed
    /// against leaves nothing to choose among.
    pub fn row(
        &self,
        book: &groups::Book,
        picked: &choice::ResponseChoice,
        lights: usize,
    ) -> groups::FamilyEvidenceRow {
        let gaps: Vec<f64> = self
            .candidates
            .iter()
            .map(|family| angle_gap(book, *family, self.angle))
            .collect();
        let keys: Vec<String> = self
            .candidates
            .iter()
            .map(|family| book.families[*family].key.clone())
            .collect();
        family_row(
            FamilyRowOf {
                session_key: &book.imaging[self.session].key,
                filter: &self.filter,
                target: &self.target,
                angle: self.angle,
                low: self.low,
                high: self.high,
                start: self.start,
                end: self.end,
                key: &self.key,
            },
            &keys,
            &gaps,
            picked,
            lights,
        )
    }
}

fn angle_gap(book: &groups::Book, family: usize, angle: f64) -> f64 {
    book.families[family]
        .angles
        .iter()
        .map(|span| span.gap_to(angle))
        .fold(f64::INFINITY, f64::min)
}

struct FamilyRowOf<'a> {
    session_key: &'a str,
    filter: &'a str,
    target: &'a str,
    angle: f64,
    low: f64,
    high: f64,
    start: i64,
    end: i64,
    key: &'a str,
}

fn family_row(
    of: FamilyRowOf<'_>,
    keys: &[String],
    gaps: &[f64],
    picked: &choice::ResponseChoice,
    lights: usize,
) -> groups::FamilyEvidenceRow {
    let chosen = match picked.winners.as_slice() {
        [] => None,
        [only] => Some(*only),
        _ if picked.any_unfitted() => None,
        several => several
            .iter()
            .copied()
            .min_by(|a, b| gaps[*a].total_cmp(&gaps[*b])),
    };
    groups::FamilyEvidenceRow {
        session_key: of.session_key.to_string(),
        filter: of.filter.to_string(),
        target: of.target.to_string(),
        angle: of.angle,
        angle_low: of.low,
        angle_high: of.high,
        start: of.start,
        end: of.end,
        key: of.key.to_string(),
        candidates: keys.to_vec(),
        chosen: chosen.map(|i| keys[i].clone()),
        lights,
        targets: vec![of.target.to_string()],
        pairs: picked
            .pairs
            .iter()
            .map(|pair| groups::FamilyPairRow {
                first: keys[pair.first].clone(),
                second: keys[pair.second].clone(),
                outcome: match pair.outcome {
                    choice::PairOutcome::Alike => "alike",
                    choice::PairOutcome::Unfitted => "unfitted",
                    choice::PairOutcome::Fit(_) => "fit",
                }
                .into(),
                carried: pair.fit().map(|fit| fit.carried),
                spread: pair
                    .fit()
                    .and_then(|fit| fit.spread.is_finite().then_some(fit.spread)),
                verdict: pair.fit().map(|fit| fit.verdict),
            })
            .collect(),
    }
}

/// The lights of one imaging session shot during a change, by filter:
/// those that ask which side of it they stand on.
#[derive(Clone, Debug)]
pub struct ChangeQuestion {
    pub event: usize,
    pub session: usize,
    pub filters: Vec<ChangeFilter>,
}

/// One filter's lights of a [`ChangeQuestion`], with the flat session of
/// a known side their background is fitted against.
#[derive(Clone, Debug)]
pub struct ChangeFilter {
    /// As captured.
    pub filter: String,
    /// The lights to read: at most [`evidence::MAX_LIGHTS`].
    pub lights: Vec<EvidenceLight>,
    /// The nearest flat session of this filter on either side, any angle.
    pub reference: usize,
    pub reference_side: evidence::Side,
    /// The boundary whose change map the fit reads.
    pub map: usize,
}

/// What one imaging session's `lights` can say about a change shot during
/// it: per filter the change applies to, with [`evidence::MIN_LIGHTS`] or
/// more lights, a nearest reference session of known side and a change
/// map. `None` when the session has no lights to read, so a host tries
/// again once it finds them.
pub fn change_question(
    book: &groups::Book,
    event_index: usize,
    session_index: usize,
    lights: Vec<EvidenceLight>,
) -> Option<ChangeQuestion> {
    let event = &book.events[event_index];
    let session = &book.imaging[session_index];
    if lights.is_empty() {
        return None;
    }
    let mut by_filter: BTreeMap<String, Vec<EvidenceLight>> = BTreeMap::new();
    for light in lights {
        by_filter
            .entry(light.filter.clone())
            .or_default()
            .push(light);
    }
    let mut filters = Vec::new();
    for (filter, lights) in by_filter {
        if !event.applies_to(Some(&filter)) || lights.len() < evidence::MIN_LIGHTS {
            continue;
        }
        let Some(reference) = book.evidence_reference(event, &filter, session.start) else {
            continue;
        };
        let Some(reference_side) = event.side_of(book.sessions[reference].first) else {
            continue;
        };
        let Some(map) = book.map_member(event_index, Some(&filter)) else {
            continue;
        };
        filters.push(ChangeFilter {
            lights: picked(&lights).into_iter().cloned().collect(),
            filter,
            reference,
            reference_side,
            map,
        });
    }
    Some(ChangeQuestion {
        event: event_index,
        session: session_index,
        filters,
    })
}

impl ChangeFilter {
    /// The responses the change map is measured between: before, then after.
    pub fn map_responses<'a>(
        &self,
        book: &'a groups::Book,
    ) -> (&'a groups::ResponseRow, &'a groups::ResponseRow) {
        let boundary = &book.boundaries[self.map];
        (
            &book.sessions[boundary.before].row,
            &book.sessions[boundary.after].row,
        )
    }

    /// The change map the lights' evidence is fitted against, from the two
    /// response planes [`ChangeFilter::map_responses`] names, pedestal out:
    /// the change of dust alone, its vignetting changes masked.
    pub fn change_map(&self, book: &groups::Book, before: &Plane, after: &Plane) -> Option<Plane> {
        let boundary = &book.boundaries[self.map];
        let (row, _) = self.map_responses(book);
        evidence::change_map(
            before,
            after,
            &boundary.vignetting_changes,
            row.bin.unwrap_or(1),
        )
    }

    /// Fit the lights' background, from their binned planes at the map's
    /// size, against the reference session's plane and the change map.
    pub fn fit(
        &self,
        book: &groups::Book,
        binned: Vec<Plane>,
        reference: &Plane,
        map: &Plane,
        pedestal: f32,
    ) -> Option<groups::EvidenceFit> {
        let binned: Vec<Plane> = binned
            .into_iter()
            .filter(|plane| plane.width == map.width && plane.height == map.height)
            .collect();
        let bin = book.sessions[self.reference].row.bin.unwrap_or(1);
        let background = evidence::background(&binned, pedestal, bin)?;
        let fit = evidence::fit(&background, reference, self.reference_side, map)?;
        let mut targets: Vec<String> = self.lights.iter().map(|l| l.target.clone()).collect();
        targets.sort();
        targets.dedup();
        Some(groups::EvidenceFit {
            filter: self.filter.clone(),
            targets,
            lights: binned.len(),
            carried: fit.carried,
            spread: fit.spread.is_finite().then_some(fit.spread),
            verdict: fit.verdict,
        })
    }
}

impl ChangeQuestion {
    /// The row its filters' fits make. Lights found but nothing to fit are
    /// kept as undecided, so they are not read again until a map or
    /// reference changes the key.
    pub fn row(&self, book: &groups::Book, fits: Vec<groups::EvidenceFit>) -> groups::EvidenceRow {
        let event = &book.events[self.event];
        let session = &book.imaging[self.session];
        let verdicts: Vec<evidence::Verdict> = fits.iter().map(|fit| fit.verdict).collect();
        groups::EvidenceRow {
            session_key: session.key.clone(),
            event_key: book.evidence_key(event, session),
            verdict: evidence::combine(&verdicts),
            fits,
        }
    }
}

// Records for hosts that submit a library request rather than holding a
// [`Library`]: the same stages over its lights.

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
