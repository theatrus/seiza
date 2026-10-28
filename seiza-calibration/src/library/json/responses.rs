//! The C ABI's flat response planning call, on [`plan_flat_sessions`].
use super::*;

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

/// The measurable flat sessions of a library request, for hosts that
/// submit records rather than holding a [`Library`]: each with the row to
/// fill once its sampled flats are read, under the identities PSF Guard
/// keeps its responses by. Rigs grouped by time are not measured, and a
/// supplied set ([`Association::OneSet`]) needs no sessions.
pub fn plan_flat_responses(mut request: Request) -> crate::Result<FlatResponsePlans> {
    request.lights.clear();
    request.master_failures.clear();
    request.pedestal_evidence.clear();
    request.grouping = groups::Request {
        night_boundaries: std::mem::take(&mut request.grouping.night_boundaries),
        ..Default::default()
    };
    let Checked {
        rules,
        nights,
        mut quality,
    } = checked(&mut request)?;
    let judged = if request.regrade {
        quality.regrade(&request.frames, &rules, &nights)
    } else {
        HashMap::new()
    };
    let grades = apply_grades(&mut request.frames, &quality, &judged);
    let one_set = request.flat_association == Association::OneSet;
    let unplanned = request
        .frames
        .iter()
        .filter(|f| {
            !one_set
                && f.kind == CalibrationKind::Flat
                && !f.is_master
                && f.kept_out.is_none()
                && f.captured_at.is_none()
        })
        .map(|f| f.frame_uuid.clone())
        .collect();
    let pedestals = PedestalTable::new(&request.frames, &quality);
    let plans: Vec<FlatResponsePlan> = if one_set {
        Vec::new()
    } else {
        plan_flat_sessions(&request.frames, &nights, &rules)
            .into_iter()
            .filter(|plan| {
                plan.measurable() && rules.for_rig(&plan.rig_uuid).grouping_limits().measured
            })
            .map(|plan| {
                let row = plan.pending_row(pedestals.for_session(&plan));
                FlatResponsePlan {
                    sample_frames: plan
                        .combined()
                        .iter()
                        .map(|f| f.frame_uuid.clone())
                        .collect(),
                    row,
                }
            })
            .collect()
    };
    let rigs: BTreeSet<_> = plans.iter().map(|p| p.row.rig_uuid.clone()).collect();
    let limits = rigs
        .into_iter()
        .map(|rig_uuid| groups::RigLimits {
            limits: rules.for_rig(&rig_uuid).grouping_limits(),
            rig_uuid,
        })
        .collect();
    Ok(FlatResponsePlans {
        algorithm_version: 2,
        plans,
        limits,
        night_boundaries: request.grouping.night_boundaries,
        unplanned,
        grades,
    })
}
