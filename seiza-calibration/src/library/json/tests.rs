use super::*;

#[test]
fn selections_and_master_cohorts_match_psf_guard_reference() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/library/psf-guard.json"
    ))
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let request: Request = serde_json::from_value(case["request"].clone()).unwrap();
        let result = evaluate(request.clone()).unwrap();
        let actual = serde_json::to_value(&result).unwrap();
        let expected = &case["expected"];
        for key in [
            "bias",
            "dark",
            "dark_flat",
            "flat",
            "kept_out",
            "tolerances",
            "flat_basis",
            "flat_family",
        ] {
            assert_eq!(
                actual["selections"][0][key], expected[key],
                "{}: {key}",
                case["name"]
            );
        }
        assert_eq!(result.assignments, vec![0, 0]);
        let rules = RuleBook::new(
            request
                .global_settings
                .into_iter()
                .map(|s| (s.key, s.value))
                .collect(),
            BTreeMap::new(),
        )
        .global();
        for (kind, key) in [
            (CalibrationKind::Bias, "bias"),
            (CalibrationKind::Dark, "dark"),
            (CalibrationKind::DarkFlat, "dark_flat"),
            (CalibrationKind::Flat, "flat"),
        ] {
            let frames: Vec<_> = expected[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| {
                    request
                        .frames
                        .iter()
                        .find(|f| f.frame_uuid == id.as_str().unwrap())
                        .unwrap()
                        .clone()
                })
                .collect();
            assert_eq!(
                coherent_master_subset(kind, &frames, &rules).len() as u64,
                expected["master_counts"][key].as_u64().unwrap(),
                "{}: {key}",
                case["name"]
            );
        }
    }
}

#[test]
fn grades_and_eligibility_match_psf_guards_captured_measurements() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/library/grading.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let mut request: Request = serde_json::from_value(case["request"].clone()).unwrap();
        request.regrade = false;
        let ungraded = serde_json::to_value(evaluate(request.clone()).unwrap()).unwrap();
        for key in ["bias", "dark", "dark_flat", "flat", "kept_out"] {
            let mut actual = ungraded["selections"][0][key].as_array().unwrap().clone();
            actual.sort_by_key(|v| v.as_str().unwrap().to_owned());
            assert_eq!(
                serde_json::json!(actual),
                case["ungraded"][key],
                "{}: {key}",
                case["name"]
            );
        }
        request.regrade = true;
        let result = evaluate(request.clone()).unwrap();
        for expected in case["grades"].as_array().unwrap() {
            let grade = result
                .grades
                .iter()
                .find(|g| g.frame_uuid == expected["frame_uuid"].as_str().unwrap())
                .unwrap();
            let judged = serde_json::to_value(grade.judged.as_ref().unwrap()).unwrap();
            assert_eq!(judged["grade"], expected["grade"]);
            assert_eq!(judged["code"], expected["code"]);
            assert_eq!(serde_json::json!(grade.kept_out), expected["kept_out"]);
        }
        if !case["use_anyway"].is_null() {
            request.quality.last_mut().unwrap().override_choice = Some(QualityOverride::Use);
            let result = evaluate(request).unwrap();
            assert!(result.selections[0].kept_out.is_empty());
            assert_eq!(result.selections[0].dark.len(), 4);
        }
    }
}

fn supplied_frame(id: usize, kind: CalibrationKind, night: i64) -> CalibrationFrame {
    CalibrationFrame {
        id: id as i64,
        frame_uuid: format!("supplied-{kind:?}-{id}"),
        rig_uuid: "supplied-rig".into(),
        kind,
        source_path: format!("/supplied-{id}.fits").into(),
        source_fingerprint: format!("stamp-{id}"),
        source_verified: true,
        captured_at: Some(1_700_000_000 + night * 86400),
        camera: Some("Test Camera".into()),
        telescope: Some("Test Scope".into()),
        width: Some(512),
        height: Some(512),
        channels: Some(1),
        binning_x: Some(1),
        binning_y: Some(1),
        gain: Some(100),
        offset: Some(20),
        exposure_s: Some(30.0),
        camera_temp: Some(-10.0),
        filter: Some("Ha".into()),
        rotation: Some(0.0),
        ..Default::default()
    }
}
fn supplied_light() -> Light {
    let f = supplied_frame(0, CalibrationKind::Flat, 0);
    Light {
        path: "/light.fits".into(),
        rig_uuid: f.rig_uuid,
        camera: f.camera,
        telescope: f.telescope,
        width: f.width,
        height: f.height,
        channels: f.channels,
        binning_x: f.binning_x,
        binning_y: f.binning_y,
        gain: f.gain,
        offset: f.offset,
        exposure_s: f.exposure_s,
        camera_temp: f.camera_temp,
        filter: f.filter,
        rotator_position: f.rotation,
        timestamp: f.captured_at,
        ..Default::default()
    }
}
#[test]
fn a_selection_reports_the_verified_frames_its_masters_take() {
    let frames: Vec<_> = (0..80)
        .map(|i| supplied_frame(i, CalibrationKind::Bias, 0))
        .collect();
    let request = Request {
        frames,
        lights: vec![supplied_light()],
        ..Default::default()
    };
    let result = evaluate(request.clone()).unwrap();
    assert_eq!(result.selections[0].bias.len(), MAX_MASTER_FRAMES);
    // A preview of unverified frames keeps every candidate.
    let mut unverified = request;
    unverified.frames[0].source_verified = false;
    let result = evaluate(unverified).unwrap();
    assert_eq!(result.selections[0].bias.len(), 80);
}

#[test]
fn supplied_one_set_keeps_all_eligible_frames_across_nights_without_the_auto_cap() {
    for kind in [CalibrationKind::Flat, CalibrationKind::Dark] {
        let frames: Vec<_> = (0..80)
            .map(|i| supplied_frame(i, kind, (i / 2) as i64 * 2))
            .collect();
        let mut request = Request {
            frames,
            lights: vec![supplied_light()],
            ..Default::default()
        };
        let automatic = evaluate(request.clone()).unwrap();
        let before = automatic.sessions[0]
            .recipes
            .iter()
            .find(|r| r.kind == kind)
            .unwrap();
        assert!(before.inputs.len() < 80);
        if kind == CalibrationKind::Flat {
            request.flat_association = Association::OneSet;
        } else {
            request.dark_association = Association::OneSet;
        }
        let explicit = evaluate(request).unwrap();
        let after = explicit.sessions[0]
            .recipes
            .iter()
            .find(|r| r.kind == kind)
            .unwrap();
        assert_eq!(after.action, RecipeAction::Integrate);
        assert_eq!(after.inputs.len(), 80);
        assert_eq!(after.input_limit, 80);
        assert_eq!(after.inputs[0], format!("supplied-{kind:?}-0"));
        assert_eq!(after.inputs[79], format!("supplied-{kind:?}-79"));
        assert_ne!(automatic.sessions[0].key, explicit.sessions[0].key);
        if kind == CalibrationKind::Flat {
            assert!(!explicit.sessions[0].flat_stability.try_next_session);
        }
    }
}
#[test]
fn one_supplied_set_rejects_incompatible_sources_and_does_not_relax_light_gates() {
    let a = supplied_frame(0, CalibrationKind::Dark, 0);
    let mut b = supplied_frame(1, CalibrationKind::Dark, 10);
    b.gain = Some(200);
    let mut request = Request {
        frames: vec![a, b],
        lights: vec![supplied_light()],
        dark_association: Association::OneSet,
        ..Default::default()
    };
    assert!(
        evaluate(request.clone())
            .unwrap_err()
            .to_string()
            .contains("cannot form one compatible")
    );
    request.frames[1].gain = Some(100);
    request.frames[1].exposure_s = Some(60.0);
    assert!(evaluate(request.clone()).is_err());
    request.frames[1].exposure_s = Some(30.0);
    request.lights[0].gain = Some(300);
    let report = evaluate(request).unwrap();
    assert!(report.selections[0].dark.is_empty());
    assert!(report.selections[0].library.dark);
}
#[test]
fn one_supplied_set_preserves_grades_manual_choices_and_excludes_poor_inputs() {
    let frames = (0..3)
        .map(|i| supplied_frame(i, CalibrationKind::Flat, i as i64 * 3))
        .collect::<Vec<_>>();
    let quality = vec![QualityRecord {
        frame_uuid: frames[1].frame_uuid.clone(),
        grade: Some(crate::Grade::Poor),
        reason: Some("bad flat".into()),
        ..Default::default()
    }];
    let mut request = Request {
        frames,
        quality,
        lights: vec![supplied_light()],
        flat_association: Association::OneSet,
        ..Default::default()
    };
    let result = evaluate(request.clone()).unwrap();
    assert_eq!(
        result.sessions[0]
            .recipes
            .iter()
            .find(|r| r.kind == CalibrationKind::Flat)
            .unwrap()
            .inputs
            .len(),
        2
    );
    request.quality[0].override_choice = Some(QualityOverride::Use);
    assert_eq!(
        evaluate(request).unwrap().sessions[0]
            .recipes
            .iter()
            .find(|r| r.kind == CalibrationKind::Flat)
            .unwrap()
            .inputs
            .len(),
        3
    );
}
#[test]
fn raw_flat_response_plans_group_nights_modes_and_bound_samples() {
    let mut frames = (0..40)
        .map(|i| supplied_frame(i, CalibrationKind::Flat, (i / 20) as i64 * 3))
        .collect::<Vec<_>>();
    // PSF Guard's family key: a readout mode's name splits sessions, its
    // index does not.
    let mut index_only = supplied_frame(42, CalibrationKind::Flat, 0);
    index_only.readout_mode = Some(2);
    frames.push(index_only);
    let mut other = supplied_frame(40, CalibrationKind::Flat, 0);
    other.readout_mode_name = Some("High Gain Mode".into());
    frames.push(other);
    let mut other = supplied_frame(43, CalibrationKind::Flat, 0);
    other.readout_mode_name = Some("High Gain Mode".into());
    frames.push(other);
    // A session of one flat cannot make a response.
    let mut alone = supplied_frame(44, CalibrationKind::Flat, 0);
    alone.readout_mode_name = Some("Low Noise".into());
    frames.push(alone);
    let mut untimed = supplied_frame(41, CalibrationKind::Flat, 0);
    untimed.captured_at = None;
    frames.push(untimed);
    let request = Request {
        frames,
        ..Default::default()
    };
    let result = plan_flat_responses(request.clone()).unwrap();
    assert_eq!(result.plans.len(), 3);
    assert_eq!(result.unplanned, vec!["supplied-Flat-41"]);
    let large = result
        .plans
        .iter()
        .filter(|p| p.row.frame_uuids.len() >= 20)
        .collect::<Vec<_>>();
    assert_eq!(large.len(), 2);
    assert!(large.iter().any(|p| p.row.frame_uuids.len() == 21));
    for plan in large {
        assert_eq!(
            plan.sample_frames.len(),
            crate::flat_response::metric::MAX_FRAMES
        );
        assert_eq!(plan.sample_frames.first(), plan.row.frame_uuids.first());
        assert_eq!(plan.sample_frames.last(), plan.row.frame_uuids.last());
    }
    assert_eq!(result.limits.len(), 1);
    let mut reversed = request;
    reversed.frames.reverse();
    assert_eq!(
        serde_json::to_value(result.plans).unwrap(),
        serde_json::to_value(plan_flat_responses(reversed).unwrap().plans).unwrap()
    );
}

#[test]
fn response_preparation_preserves_explicit_night_boundaries_and_one_set_needs_no_dates() {
    let mut frame = supplied_frame(1, CalibrationKind::Flat, 0);
    frame.captured_at = Some(1_700_000_000);
    let mut second = supplied_frame(2, CalibrationKind::Flat, 0);
    second.captured_at = Some(1_700_000_060);
    let mut request = Request {
        frames: vec![frame, second],
        ..Default::default()
    };
    let shift = |frame: &str, seconds| crate::flat_response::groups::NightBoundary {
        rig_uuid: "supplied-rig".into(),
        frame_uuid: frame.into(),
        seconds,
    };
    request.grouping.night_boundaries =
        vec![shift("supplied-Flat-1", 0), shift("supplied-Flat-2", 0)];
    let first = plan_flat_responses(request.clone()).unwrap();
    assert_eq!(first.night_boundaries[0].seconds, 0);
    request.night_shifts = vec![
        shift("supplied-Flat-1", 86400),
        shift("supplied-Flat-2", 86400),
    ];
    let second = plan_flat_responses(request.clone()).unwrap();
    assert_ne!(first.plans[0].row.night, second.plans[0].row.night);
    request.frames[0].captured_at = None;
    request.frames[1].captured_at = None;
    request.flat_association = Association::OneSet;
    let explicit = plan_flat_responses(request).unwrap();
    assert!(explicit.plans.is_empty());
    assert!(explicit.unplanned.is_empty());
}

fn work_fixture() -> FlatWorkRequest {
    let cases: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/library/psf-guard.json"
    ))
    .unwrap();
    let mut library: Request =
        serde_json::from_value(cases["cases"][12]["request"].clone()).unwrap();
    library.grouping.lights.clear();
    library.grouping.responses[0].rotation = Some(86.0);
    library.grouping.comparisons[0].dust = 0.03;
    for frame in &mut library.frames {
        if frame.frame_uuid.starts_with("a-") {
            frame.rotation = Some(86.0);
        }
    }
    let mut light = library.lights[0].clone();
    light.rotator_position = Some(89.0);
    library.lights = (0..12)
        .map(|i| {
            let mut l = light.clone();
            l.path = format!("/does-not-exist/light-{i}.fits").into();
            l.timestamp = Some(345600 + i * 60);
            l
        })
        .collect();
    let light_context = (0..12)
        .map(|i| FlatLightContext {
            light_index: i,
            id: Some(100 + i as i64),
            target: "M42".into(),
            pedestal_adu: Some(123.0),
            ..Default::default()
        })
        .collect();
    FlatWorkRequest {
        library,
        light_context,
    }
}
#[test]
fn cached_metadata_drives_family_and_change_questions_without_files() {
    let request = work_fixture();
    let plan = plan_flat_work(request.clone()).unwrap();
    assert_eq!(plan.families.len(), 1, "{plan:?}");
    assert_eq!(plan.changes.len(), 1, "{plan:?}");
    let q = &plan.families[0];
    assert_eq!(q.candidates.len(), 2);
    assert_eq!(q.sample_lights.len(), 8);
    assert_eq!(q.sample_lights.first(), Some(&100));
    assert_eq!(q.sample_lights.last(), Some(&111));
    assert_eq!(q.pedestal_adu, Some(123.0));
    assert_eq!(plan.changes[0].filters.len(), 1);
    assert_eq!(plan.changes[0].filters[0].sample_lights.len(), 8);
    let answer = finish_flat_work(FlatWorkAnswer::Family {
        question: q.clone(),
        choice: crate::flat_response::choice::ResponseChoice::default(),
        lights: 8,
    })
    .unwrap();
    let FlatWorkResult::Family { row } = answer else {
        panic!("family row")
    };
    assert_eq!(row.chosen, None);
    let mut repeat = request.clone();
    repeat.library.grouping = plan.grouping.clone();
    repeat.library.grouping.family_evidence.push(row);
    let change = finish_flat_work(FlatWorkAnswer::Change {
        question: plan.changes[0].clone(),
        fits: vec![crate::flat_response::groups::EvidenceFit {
            filter: "Ha".into(),
            targets: vec!["M42".into()],
            lights: 8,
            carried: 0.0,
            spread: Some(0.01),
            verdict: crate::flat_response::evidence::Verdict::Before,
        }],
    })
    .unwrap();
    let FlatWorkResult::Change { row } = change else {
        panic!("change row")
    };
    assert_eq!(row.verdict, crate::flat_response::evidence::Verdict::Before);
    repeat.library.grouping.evidence.push(row);
    let cached = plan_flat_work(repeat).unwrap();
    assert!(cached.families[0].answered);
    assert!(cached.families[0].sample_lights.is_empty());
    assert!(cached.changes[0].answered);
    assert!(cached.changes[0].filters[0].sample_lights.is_empty());
    let mut excluded = request;
    for context in &mut excluded.light_context {
        context.excluded = true;
    }
    let empty = plan_flat_work(excluded).unwrap();
    assert!(empty.families.is_empty());
    assert!(empty.changes.is_empty());
}
#[test]
fn evidence_questions_keep_targets_modes_and_personal_group_choices() {
    let mut request = work_fixture();
    for i in 0..3 {
        request.light_context[i].target = "M31".into();
    }
    let plan = plan_flat_work(request.clone()).unwrap();
    assert_eq!(plan.families.len(), 2);
    for question in plan.families {
        assert!(
            question
                .sample_lights
                .iter()
                .all(|id| if question.target == "M31" {
                    *id < 103
                } else {
                    *id >= 103
                })
        );
    }
    // A manual split is resolved in the core, not interpreted by the caller.
    let row = &request.library.grouping.responses[2];
    request
        .library
        .grouping
        .group_overrides
        .push(crate::flat_response::groups::GroupOverrideRow {
            anchor_frame: row.frame_uuids[0].clone(),
            session_key: row.session_key.clone(),
            family_key: row.family_key.clone(),
            rig_uuid: row.rig_uuid.clone(),
            choice: Some(crate::flat_response::groups::GroupChoice::Split),
            before_frame: None,
            before_session_key: None,
            label: "manual edge".into(),
            chosen_at: 1,
            night: Some(row.night.clone()),
            before_night: None,
            rotation: row.rotation,
            before_rotation: None,
        });
    let manual = plan_flat_work(request).unwrap();
    assert_eq!(manual.overrides.len(), 1);
    assert!(manual.overrides[0].applied());
    assert_eq!(
        manual.grouping.group_overrides[0].choice,
        Some(crate::flat_response::groups::GroupChoice::Split)
    );
}
#[test]
fn work_feedback_refuses_invalid_indices_duplicate_filters_and_empty_pixel_reads() {
    use crate::flat_response::{choice, evidence, groups};
    let plan = plan_flat_work(work_fixture()).unwrap();
    let q = plan.families[0].clone();
    assert!(
        finish_flat_work(FlatWorkAnswer::Family {
            question: q.clone(),
            choice: choice::ResponseChoice {
                winners: vec![99],
                pairs: vec![]
            },
            lights: 8
        })
        .is_err()
    );
    assert!(
        finish_flat_work(FlatWorkAnswer::Family {
            question: q,
            choice: Default::default(),
            lights: 0
        })
        .is_err()
    );
    let fit = groups::EvidenceFit {
        filter: "Ha".into(),
        targets: vec![],
        lights: 8,
        carried: 0.0,
        spread: None,
        verdict: evidence::Verdict::Before,
    };
    assert!(
        finish_flat_work(FlatWorkAnswer::Change {
            question: plan.changes[0].clone(),
            fits: vec![fit.clone(), fit]
        })
        .is_err()
    );
}

#[test]
fn correction_plans_keep_response_dependencies_side_and_shared_identity() {
    let mut request = work_fixture();
    let mut other = Vec::new();
    for source in request.library.grouping.responses.iter().skip(1) {
        let mut row = source.clone();
        row.response_key = format!("oiii-{}", row.response_key);
        row.session_key = format!("oiii-{}", row.session_key);
        row.family_key = "oiii-sensor".into();
        row.filter = Some("OIII".into());
        row.frame_uuids = row
            .frame_uuids
            .iter()
            .map(|id| format!("oiii-{id}"))
            .collect();
        other.push(row);
    }
    let spot=serde_json::from_value(serde_json::json!({"kind":"new","darker":true,"x":500.0,"y":300.0,"radius":30.0,"area":2827.0,"peak":0.03})).unwrap();
    request.library.grouping.comparisons[2].spots = vec![spot];
    let mut compare = request.library.grouping.comparisons[2].clone();
    compare.earlier_key = "oiii-b".into();
    compare.later_key = "oiii-c".into();
    request.library.grouping.comparisons.push(compare);
    request.library.grouping.responses.extend(other);
    request
        .library
        .frames
        .retain(|f| !f.frame_uuid.starts_with("b-"));
    request.library.available_responses = request
        .library
        .grouping
        .responses
        .iter()
        .map(|r| r.response_key.clone())
        .collect();
    for light in &mut request.library.lights {
        light.timestamp = Some(172800);
        light.rotator_position = Some(90.0);
    }
    let plan = plan_flat_work(request).unwrap();
    assert_eq!(plan.corrections.len(), 1, "{plan:?}");
    let correction = &plan.corrections[0];
    assert_eq!(correction.light_indices.len(), 12);
    assert_eq!(correction.before.response_key, "b");
    assert_eq!(correction.after.response_key, "c");
    assert!(correction.divide);
    assert_eq!(correction.bin, 10);
    assert_eq!(
        correction.correction.light_side,
        crate::flat_response::evidence::Side::Before
    );
}
#[test]
fn psf_guard_family_ties_and_unfitted_pairs_are_resolved_in_core() {
    use crate::flat_response::{choice, evidence};
    let mut q = plan_flat_work(work_fixture()).unwrap().families.remove(0);
    q.candidates[0].angle_gap = Some(2.0);
    q.candidates[1].angle_gap = Some(1.0);
    let mut third = q.candidates[0].clone();
    third.key = "third".into();
    q.candidates.push(third);
    let fit = choice::PairOutcome::Fit(evidence::Fit {
        carried: 0.0,
        spread: 0.01,
        verdict: evidence::Verdict::Before,
    });
    let choice = choice::ResponseChoice {
        winners: vec![0, 1],
        pairs: vec![
            choice::PairFit {
                first: 0,
                second: 1,
                outcome: choice::PairOutcome::Alike,
            },
            choice::PairFit {
                first: 0,
                second: 2,
                outcome: fit,
            },
            choice::PairFit {
                first: 1,
                second: 2,
                outcome: fit,
            },
        ],
    };
    let answer = finish_flat_work(FlatWorkAnswer::Family {
        question: q.clone(),
        choice: choice.clone(),
        lights: 8,
    })
    .unwrap();
    let FlatWorkResult::Family { row } = answer else {
        panic!()
    };
    assert_eq!(row.chosen, Some(q.candidates[1].key.clone()));
    let mut unknown = choice;
    unknown.pairs[0].outcome = choice::PairOutcome::Unfitted;
    let answer = finish_flat_work(FlatWorkAnswer::Family {
        question: q,
        choice: unknown,
        lights: 8,
    })
    .unwrap();
    let FlatWorkResult::Family { row } = answer else {
        panic!()
    };
    assert_eq!(row.chosen, None);
}

#[test]
fn path_scoped_light_choices_do_not_leak_between_identical_basenames() {
    use crate::flat_response::groups;
    let mut request = work_fixture().library;
    request.lights.truncate(2);
    request.lights[0].path = "/first/shared.fits".into();
    request.lights[1] = request.lights[0].clone();
    request.lights[1].path = "/second/shared.fits".into();
    let book = groups::prepare(
        plan_flat_work(FlatWorkRequest {
            library: request.clone(),
            ..Default::default()
        })
        .unwrap()
        .grouping,
    )
    .unwrap();
    let event = &book.events[0];
    request.grouping.light_choices = vec![groups::LightChoiceEntry {
        light_id: 0,
        file_name: "/first/shared.fits".into(),
        filter: Some("Ha".into()),
        at: request.lights[0].timestamp,
        row: groups::LightChoiceRow {
            light_guid: "first".into(),
            change_key: event.key.clone(),
            rig_uuid: event.rig_uuid.clone(),
            change_start: Some(event.start),
            change_end: Some(event.end),
            choice: Some(groups::LightChoice::Before),
            group_frame: None,
            group_session_key: None,
            label: "before".into(),
            chosen_at: 1,
        },
    }];
    let result = evaluate(request).unwrap();
    assert!(!result.selections[0].flat_choices.is_empty());
    assert!(result.selections[1].flat_choices.is_empty());
}
