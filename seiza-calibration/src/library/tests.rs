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
