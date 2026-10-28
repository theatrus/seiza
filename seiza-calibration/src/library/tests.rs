use super::*;
// Reference regression from PSF Guard 79f5e31.
#[test]
fn a_master_never_mixes_readout_mode_names() {
    let dark = |id: i64, name: Option<&str>| CalibrationFrame {
        id,
        frame_uuid: format!("d{id}"),
        rig_uuid: "r".into(),
        kind: CalibrationKind::Dark,
        source_path: format!("/dark-{id}.fits").into(),
        source_fingerprint: "x".into(),
        captured_at: Some(1_000_000_000 + id),
        telescope: None,
        camera: Some("Camera".into()),
        width: Some(3000),
        height: Some(2000),
        channels: Some(1),
        binning_x: Some(1),
        binning_y: Some(1),
        gain: Some(100),
        offset: Some(20),
        readout_mode: None,
        readout_mode_name: name.map(str::to_string),
        is_master: false,
        bayer_pattern: None,
        exposure_s: Some(300.0),
        camera_temp: Some(-10.0),
        filter: None,
        focal_length_mm: None,
        rotation: None,
        valid_direction: None,
        kept_out: None,
        source_verified: false,
    };
    // Five darks in one mode and two strays in another: the two strays
    // are set aside rather than averaged into a master whose read noise
    // and bias level describe neither mode.
    let frames: Vec<CalibrationFrame> = (0..5)
        .map(|id| dark(id, Some("Extend Fullwell 2CMS")))
        .chain((5..7).map(|id| dark(id, Some("High Gain Mode"))))
        .collect();
    let subset = coherent_master_subset(CalibrationKind::Dark, &frames, &Rules::default());
    assert_eq!(subset.len(), 5);
    assert!(
        subset
            .iter()
            .all(|frame| frame.readout_mode_name.as_deref() == Some("Extend Fullwell 2CMS"))
    );

    // A frame that never named its mode is tolerated in either set: two
    // calibration frames are only apart when both say something different.
    let frames: Vec<CalibrationFrame> = (0..4)
        .map(|id| dark(id, Some("Extend Fullwell 2CMS")))
        .chain(std::iter::once(dark(4, None)))
        .collect();
    assert_eq!(
        coherent_master_subset(CalibrationKind::Dark, &frames, &Rules::default()).len(),
        5
    );

    // Nearest-first, like Seiza's session rule: two High Gain darks from
    // last night outrank twenty Extend Fullwell darks from last month
    // when the light names no mode — and the twenty are another mode,
    // not frames "the integrator would not accept", so none is reported
    // as set aside.
    let frames: Vec<CalibrationFrame> = (0..2)
        .map(|id| dark(id, Some("High Gain Mode")))
        .chain((2..22).map(|id| dark(id, Some("Extend Fullwell 2CMS"))))
        .collect();
    let report = master_subset_report(CalibrationKind::Dark, &frames, &Rules::default());
    assert_eq!(report.kept.len(), 2);
    assert!(report.dropped.is_empty());

    // A lone nearest stray does not make a master; the next mode does.
    let frames: Vec<CalibrationFrame> = std::iter::once(dark(0, Some("High Gain Mode")))
        .chain((1..6).map(|id| dark(id, Some("Extend Fullwell 2CMS"))))
        .collect();
    let report = master_subset_report(CalibrationKind::Dark, &frames, &Rules::default());
    assert_eq!(report.kept.len(), 5);
    assert!(
        report
            .kept
            .iter()
            .all(|frame| frame.readout_mode_name.as_deref() == Some("Extend Fullwell 2CMS"))
    );
}

#[test]
fn selections_and_master_cohorts_match_psf_guard_reference() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/library/psf-guard.json")).unwrap();
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
        serde_json::from_str(include_str!("../../tests/fixtures/library/grading.json")).unwrap();
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
        grade: Some("poor".into()),
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
    let mut other = supplied_frame(40, CalibrationKind::Flat, 0);
    other.readout_mode = Some(2);
    frames.push(other);
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
        .filter(|p| p.row.frame_uuids.len() == 20)
        .collect::<Vec<_>>();
    assert_eq!(large.len(), 2);
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
