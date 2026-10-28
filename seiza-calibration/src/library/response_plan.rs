//! Raw-flat response sessions, extracted from PSF Guard's file job. The core
//! chooses sessions and bounded response samples; hosts read/cache the pixels.
use super::*;
use crate::flat_response::{groups, metric};
#[derive(Clone, Debug, Serialize)]
pub struct FlatResponsePlan {
    pub row: groups::ResponseRow,
    pub sample_frames: Vec<String>,
}
#[derive(Debug, Serialize)]
pub struct FlatResponsePlans {
    pub algorithm_version: u32,
    pub plans: Vec<FlatResponsePlan>,
    pub limits: Vec<groups::RigLimits>,
    pub night_boundaries: Vec<groups::NightBoundary>,
    /// Eligible raw flats without a capture time cannot establish night/change evidence.
    pub unplanned: Vec<String>,
    pub grades: Vec<FrameGrade>,
}
fn key(value: impl Serialize) -> String {
    hex_digest(&serde_json::to_string(&value).expect("finite response identity"))
}
/// Plan graded raw-flat sessions and response sampling, without opening files.
/// Uses the same global/rig settings, grades and night boundaries as evaluate.
/// Hosts fill each row's measurement fields with metric::build_response, compare
/// the resulting planes, and pass the rows back to evaluate in grouping.
pub fn plan_flat_responses(mut request: Request) -> crate::Result<FlatResponsePlans> {
    request.lights.clear();
    request.grouping = groups::Request {
        night_boundaries: std::mem::take(&mut request.grouping.night_boundaries),
        ..Default::default()
    };
    let snapshot = evaluate(request.clone())?;
    let rules = master_policy::rule_book(request.global_settings.clone(), request.rigs.clone())?;
    let nights = resolve_nights(&mut request, &rules)?;
    let grades: BTreeMap<_, _> = snapshot
        .grades
        .iter()
        .map(|g| (g.frame_uuid.as_str(), g))
        .collect();
    let mut flats: Vec<_> = request
        .frames
        .iter()
        .filter(|f| {
            f.kind == CalibrationKind::Flat
                && !f.is_master
                && grades[&f.frame_uuid.as_str()].kept_out.is_none()
        })
        .collect();
    flats.sort_by_key(|f| (f.captured_at, f.id, &f.frame_uuid));
    let unplanned = flats
        .iter()
        .filter(|f| request.flat_association != Association::OneSet && f.captured_at.is_none())
        .map(|f| f.frame_uuid.clone())
        .collect();
    let mut plans: Vec<FlatResponsePlan> = Vec::new();
    if request.flat_association != Association::OneSet {
        let focal = focal_classes(&flats, &rules);
        for f in flats.iter().filter(|f| f.captured_at.is_some()) {
            let night = nights.night_of_frame(&f.rig_uuid, &f.frame_uuid, f.captured_at.unwrap());
            let family = key((
                &f.rig_uuid,
                f.filter.as_deref().map(flat_notes::model::canonical_filter),
                focal[&f.frame_uuid],
                f.width,
                f.height,
                f.channels,
                f.binning_x,
                f.binning_y,
                f.gain,
                f.offset,
                f.readout_mode,
                f.readout_mode_name
                    .as_deref()
                    .map(|s| s.trim().to_ascii_lowercase()),
                f.bayer_pattern.as_deref().map(str::to_ascii_uppercase),
            ));
            let joins = |p: &&mut FlatResponsePlan| {
                p.row.family_key == family
                    && p.row.night == night
                    && match (p.row.rotation, f.rotation) {
                        (None, None) => true,
                        (Some(a), Some(b)) => {
                            let d = (a - b).rem_euclid(360.0);
                            d.min(360.0 - d) <= groups::ROTATION_CLUSTER_DEG
                        }
                        _ => false,
                    }
            };
            if let Some(plan) = plans.iter_mut().find(joins) {
                plan.row.frame_uuids.push(f.frame_uuid.clone());
                plan.row.captured_last = f.captured_at;
            } else {
                plans.push(FlatResponsePlan {
                    row: groups::ResponseRow {
                        response_key: String::new(),
                        session_key: key((&family, &night, &f.frame_uuid)),
                        rig_uuid: f.rig_uuid.clone(),
                        family_key: family,
                        night,
                        filter: f.filter.clone(),
                        rotation: f.rotation,
                        frame_uuids: vec![f.frame_uuid.clone()],
                        frames: 0,
                        captured_first: f.captured_at,
                        captured_last: f.captured_at,
                        level_adu: None,
                        pedestal_adu: None,
                        noise: None,
                        width: None,
                        height: None,
                        bin: None,
                        algorithm_version: metric::RESPONSE_ALGORITHM_VERSION,
                        error: None,
                        unread: Vec::new(),
                    },
                    sample_frames: Vec::new(),
                });
            }
        }
    }
    let frames: BTreeMap<_, _> = request
        .frames
        .iter()
        .map(|f| (f.frame_uuid.as_str(), f))
        .collect();
    for plan in &mut plans {
        let n = plan.row.frame_uuids.len();
        plan.sample_frames = if n <= metric::MAX_FRAMES {
            plan.row.frame_uuids.clone()
        } else {
            (0..metric::MAX_FRAMES)
                .map(|i| plan.row.frame_uuids[i * (n - 1) / (metric::MAX_FRAMES - 1)].clone())
                .collect()
        };
        let inputs: Vec<_> = plan
            .sample_frames
            .iter()
            .map(|id| (id, &frames[id.as_str()].source_fingerprint))
            .collect();
        plan.row.response_key = key((metric::RESPONSE_ALGORITHM_VERSION, metric::MAX_SIDE, inputs));
    }
    // Resolve pedestal evidence with the shared matcher rather than assuming
    // every bias/dark-flat at one gain belongs to the response's sensor mode.
    request.lights = plans
        .iter()
        .map(|p| as_light(frames[p.row.frame_uuids[0].as_str()]))
        .collect();
    request.flat_association = Association::Automatic;
    request.dark_association = Association::Automatic;
    let selected = evaluate(request.clone())?;
    let quality: BTreeMap<_, _> = request
        .quality
        .iter()
        .map(|q| (q.frame_uuid.as_str(), q))
        .collect();
    for (plan, selection) in plans.iter_mut().zip(&selected.selections) {
        let ids = if selection.dark_flat.is_empty() {
            &selection.bias
        } else {
            &selection.dark_flat
        };
        let mut levels: Vec<f64> = ids
            .iter()
            .filter_map(|id| match quality.get(id.as_str())?.measurement.as_ref()? {
                StoredMeasurement::Dark(m) => Some(m.level as f64),
                _ => None,
            })
            .filter(|v| v.is_finite())
            .collect();
        levels.sort_by(f64::total_cmp);
        plan.row.pedestal_adu = levels.get(levels.len() / 2).copied();
    }
    let rigs: BTreeSet<_> = plans.iter().map(|p| p.row.rig_uuid.clone()).collect();
    let limits = rigs
        .into_iter()
        .map(|rig_uuid| groups::RigLimits {
            limits: flat_notes::limits_of(&rules.for_rig(&rig_uuid)),
            rig_uuid,
        })
        .collect();
    Ok(FlatResponsePlans {
        algorithm_version: 1,
        plans,
        limits,
        night_boundaries: request.grouping.night_boundaries,
        unplanned,
        grades: snapshot.grades,
    })
}
fn as_light(f: &CalibrationFrame) -> Light {
    Light {
        path: f.source_path.clone(),
        rig_uuid: f.rig_uuid.clone(),
        camera: f.camera.clone(),
        telescope: f.telescope.clone(),
        bayer_pattern: f.bayer_pattern.clone(),
        filter: f.filter.clone(),
        width: f.width,
        height: f.height,
        channels: f.channels,
        binning_x: f.binning_x,
        binning_y: f.binning_y,
        gain: f.gain,
        offset: f.offset,
        readout_mode: f.readout_mode,
        readout_mode_name: f.readout_mode_name.clone(),
        focal_length_mm: f.focal_length_mm,
        rotator_position: f.rotation,
        exposure_s: f.exposure_s,
        camera_temp: f.camera_temp,
        timestamp: f.captured_at,
    }
}
// Chained focal classes preserve PSF Guard's handling of reducers and absent metadata.
fn focal_classes(flats: &[&CalibrationFrame], rules: &RuleBook) -> HashMap<String, Option<f64>> {
    let mut out = HashMap::new();
    let rigs: BTreeSet<_> = flats.iter().map(|f| f.rig_uuid.as_str()).collect();
    for rig in rigs {
        let tolerance = rules.for_rig(rig).coherence_tolerances().focal_length_mm;
        let mut known: Vec<f64> = flats
            .iter()
            .filter(|f| f.rig_uuid == rig)
            .filter_map(|f| f.focal_length_mm)
            .collect();
        known.sort_by(f64::total_cmp);
        known.dedup();
        let classes: Vec<_> = known
            .iter()
            .enumerate()
            .filter(|(i, x)| *i == 0 || **x - known[i - 1] > tolerance)
            .map(|(_, x)| *x)
            .collect();
        for f in flats.iter().filter(|f| f.rig_uuid == rig) {
            let class = match f.focal_length_mm {
                Some(mm) => classes.iter().rev().find(|start| **start <= mm).copied(),
                None if classes.len() == 1 => Some(classes[0]),
                None => None,
            };
            out.insert(f.frame_uuid.clone(), class);
        }
    }
    out
}
