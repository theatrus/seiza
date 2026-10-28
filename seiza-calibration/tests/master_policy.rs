use seiza_calibration::library::{self, FlatRetryRequest, MasterPlanRequest, Request};
use serde_json::Value;

#[test]
fn flat_retry_preserves_psf_guard_decisions_and_optional_host_choices() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/library/flat-retries.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let result = serde_json::to_value(
            library::flat_retry(serde_json::from_value(case["request"].clone()).unwrap()).unwrap(),
        )
        .unwrap();
        for field in ["action", "inputs", "rebuild_first"] {
            assert_eq!(result[field], case[field], "{} {field}", case["name"]);
        }
        if case["name"] == "publish_stable_alternate" {
            assert!(
                result["note"]
                    .as_str()
                    .unwrap()
                    .contains("four initial flats")
            );
            assert!(
                result["note"]
                    .as_str()
                    .unwrap()
                    .contains("two alternate flats")
            );
        }
    }
}

#[test]
fn verified_selected_planning_matches_library_recipes_options_and_identity() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/library/recipes.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let request: Request = serde_json::from_value(case["request"].clone()).unwrap();
        let result = library::evaluate(request.clone()).unwrap();
        for (index, selection) in result.selections.iter().enumerate() {
            let mut frames = Vec::new();
            for ids in [
                &selection.bias,
                &selection.dark_flat,
                &selection.dark,
                &selection.flat,
            ] {
                frames.extend(ids.iter().map(|id| {
                    request
                        .frames
                        .iter()
                        .find(|f| &f.frame_uuid == id)
                        .unwrap()
                        .clone()
                }));
            }
            let selected = library::plan_masters(MasterPlanRequest {
                frames,
                lights: vec![request.lights[index].clone()],
                global_settings: request.global_settings.clone(),
                rigs: request.rigs.clone(),
                flat_basis: selection.flat_basis.clone(),
                flat_family: selection.flat_family.clone(),
                flat_held_back: selection.flat_held_back.clone(),
                ..Default::default()
            })
            .unwrap();
            let expected = &result.sessions[result.assignments[index]];
            assert_eq!(selected.key, expected.key, "{}", case["name"]);
            assert_eq!(
                serde_json::to_value(&selected.recipes).unwrap(),
                serde_json::to_value(&expected.recipes).unwrap()
            );
        }
    }
}

#[test]
fn recipe_respects_cached_grades_overrides_and_star_masking() {
    let mut request = MasterPlanRequest {
        lights: vec![Default::default()],
        frames: (0..3)
            .map(|i| library::CalibrationFrame {
                frame_uuid: format!("d{i}"),
                source_fingerprint: format!("fp{i}"),
                id: i,
                captured_at: Some(i),
                exposure_s: Some(300.0),
                camera_temp: Some(-10.0),
                kind: library::CalibrationKind::Dark,
                camera: Some("Camera".into()),
                width: Some(64),
                height: Some(64),
                channels: Some(1),
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    };
    let screening = |request: MasterPlanRequest| {
        library::plan_masters(request).unwrap().recipes[2].dark_level_screening
    };
    assert!(
        screening(request.clone()),
        "{:?}",
        library::plan_masters(request.clone()).unwrap()
    );
    request.quality = request
        .frames
        .iter()
        .map(|f| library::QualityRecord {
            frame_uuid: f.frame_uuid.clone(),
            source_fingerprint: Some(f.source_fingerprint.clone()),
            algorithm_version: Some(seiza_calibration::FRAME_QUALITY_ALGORITHM_VERSION),
            grade: Some("ok".into()),
            ..Default::default()
        })
        .collect();
    assert!(!screening(request.clone()));
    request.quality[0].source_fingerprint = Some("stale".into());
    assert!(screening(request.clone()));
    request.quality[0].override_choice = Some(library::QualityOverride::Use);
    assert!(!screening(request.clone()));
    request
        .frames
        .iter_mut()
        .for_each(|f| f.kind = library::CalibrationKind::Flat);
    request.quality.clear();
    let a = library::plan_masters(request.clone()).unwrap();
    request.flat_star_masking = true;
    let b = library::plan_masters(request).unwrap();
    assert!(!a.recipes[3].flat_star_masking);
    assert!(b.recipes[3].flat_star_masking);
    assert_ne!(a.key, b.key);
}

#[test]
fn invalid_retry_feedback_is_rejected() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/library/flat-retries.json")).unwrap();
    let request: FlatRetryRequest =
        serde_json::from_value(fixture["cases"][0]["request"].clone()).unwrap();
    let mut invalid = request.clone();
    invalid.first.disagreement.pop();
    assert!(library::flat_retry(invalid).is_err());
    let mut invalid = request.clone();
    invalid.first.used[0] = "unknown".into();
    assert!(library::flat_retry(invalid).is_err());
    let mut invalid = request.clone();
    invalid.trimmed_attempted = true;
    invalid.trimmed = Some(library::FlatAttempt::default());
    assert!(library::flat_retry(invalid).is_err());
    let mut invalid = request.clone();
    invalid.trimmed_attempted = true;
    invalid.alternate_attempted = true;
    invalid.alternate = Some(library::FlatAttempt {
        used: vec!["f0".into(), "f1".into()],
        ..Default::default()
    });
    assert!(library::flat_retry(invalid).is_err());
    let mut invalid = request;
    invalid.first.drift_threshold = 0;
    assert!(library::flat_retry(invalid).is_err());
}

#[test]
fn every_policy_endpoint_validates_settings_with_context() {
    use library::{MasterCohortRequest, RigSettings, Setting};
    for (rows, expected) in [
        (
            vec![("external_masterz", serde_json::json!("ignore"))],
            "unknown setting 'external_masterz'",
        ),
        (
            vec![("external_masters", serde_json::json!("nonsense"))],
            "external_masters",
        ),
        (vec![("use_darks", serde_json::json!("false"))], "use_darks"),
        (
            vec![("rotation_tolerance_deg", serde_json::json!(-1))],
            "rotation_tolerance_deg",
        ),
        (
            vec![
                ("use_darks", serde_json::json!(false)),
                ("use_darks", serde_json::json!(true)),
            ],
            "duplicate setting key 'use_darks'",
        ),
    ] {
        let settings: Vec<_> = rows
            .into_iter()
            .map(|(key, value)| Setting {
                key: key.into(),
                value,
            })
            .collect();
        let errors = [
            library::evaluate(Request {
                global_settings: settings.clone(),
                ..Default::default()
            })
            .unwrap_err()
            .to_string(),
            library::plan_masters(MasterPlanRequest {
                lights: vec![Default::default()],
                global_settings: settings.clone(),
                ..Default::default()
            })
            .unwrap_err()
            .to_string(),
            library::master_cohort(MasterCohortRequest {
                settings: settings.clone(),
                ..Default::default()
            })
            .unwrap_err()
            .to_string(),
            library::flat_retry(FlatRetryRequest {
                settings: settings.clone(),
                ..Default::default()
            })
            .unwrap_err()
            .to_string(),
            library::evaluate(Request {
                rigs: vec![RigSettings {
                    rig_uuid: "camera-a".into(),
                    settings,
                    ..Default::default()
                }],
                ..Default::default()
            })
            .unwrap_err()
            .to_string(),
        ];
        for error in &errors {
            assert!(error.contains(expected), "{error}");
        }
        assert!(errors[4].contains("rigs[camera-a].settings"));
    }
}

#[test]
fn returned_session_carries_retry_policy_window_and_exclusions() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/library/flat-retries.json")).unwrap();
    let original: FlatRetryRequest =
        serde_json::from_value(fixture["cases"][0]["request"].clone()).unwrap();
    let mut plan = MasterPlanRequest {
        lights: vec![Default::default()],
        frames: original.candidates.clone(),
        flat_held_back: original.held_back.clone(),
        ..Default::default()
    };
    plan.flat_stability.enabled = false;
    let session = library::plan_masters(plan.clone()).unwrap();
    let mut without_exclusions = plan.clone();
    without_exclusions.flat_held_back.clear();
    assert_ne!(
        session.key,
        library::plan_masters(without_exclusions).unwrap().key
    );
    // A JSON round trip is exactly how C/Python hosts forward the result.
    let session = serde_json::from_value(serde_json::to_value(session).unwrap()).unwrap();
    let mut retry = FlatRetryRequest {
        session: Some(session),
        held_back: vec![],
        ..original.clone()
    };
    assert_eq!(
        library::flat_retry(retry.clone()).unwrap().action,
        library::FlatRetryAction::PublishFirst
    );
    retry.policy = Some(Default::default());
    assert!(
        library::flat_retry(retry)
            .unwrap_err()
            .to_string()
            .contains("conflict")
    );

    plan.flat_stability.enabled = true;
    plan.flat_stability.trim_unstable = false;
    plan.global_settings = vec![library::Setting {
        key: "flat_session_hours".into(),
        value: serde_json::json!(48),
    }];
    let mut retry = FlatRetryRequest {
        session: Some(library::plan_masters(plan).unwrap()),
        held_back: vec![],
        ..original
    };
    assert!(
        library::flat_retry(retry.clone())
            .unwrap()
            .inputs
            .is_empty(),
        "the rig's 48-hour window excludes next-day flats"
    );
    retry
        .session
        .as_mut()
        .unwrap()
        .tolerances
        .flat_session_seconds = 86400;
    let result = library::flat_retry(retry.clone()).unwrap();
    assert_eq!(
        result.inputs,
        ["f4", "f5"],
        "held-back f6 travels with the session"
    );
    retry.settings = vec![library::Setting {
        key: "flat_session_hours".into(),
        value: serde_json::json!(48),
    }];
    assert!(
        library::flat_retry(retry)
            .unwrap_err()
            .to_string()
            .contains("conflict")
    );
}

#[test]
fn successful_retry_feedback_implies_attempted_but_failed_retry_needs_a_flag() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/library/flat-retries.json")).unwrap();
    let case = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["action"] == "publish_trimmed")
        .unwrap();
    let mut request: FlatRetryRequest = serde_json::from_value(case["request"].clone()).unwrap();
    request.trimmed_attempted = false;
    assert_eq!(
        library::flat_retry(request.clone()).unwrap().action,
        library::FlatRetryAction::PublishTrimmed
    );
    request.trimmed = None;
    assert_eq!(
        library::flat_retry(request.clone()).unwrap().action,
        library::FlatRetryAction::TryTrimmed
    );
    request.trimmed_attempted = true;
    assert_eq!(
        library::flat_retry(request).unwrap().action,
        library::FlatRetryAction::TryAlternate
    );
}

#[test]
fn an_alternate_flat_set_is_the_next_session_alone_without_masters() {
    let flat = |id: &str, at: i64, master: bool| library::CalibrationFrame {
        frame_uuid: id.into(),
        kind: library::CalibrationKind::Flat,
        captured_at: Some(at),
        camera: Some("Camera".into()),
        filter: Some("Ha".into()),
        is_master: master,
        ..Default::default()
    };
    // The first set drifted entirely; two later sessions each match the
    // light, a dust state apart, and an external master is on offer too.
    let candidates = vec![
        flat("a0", 0, false),
        flat("a1", 1, false),
        flat("m", 50_000, true),
        flat("b0", 100_000, false),
        flat("b1", 100_001, false),
        flat("c0", 400_000, false),
        flat("c1", 400_001, false),
    ];
    let request = FlatRetryRequest {
        candidates,
        first: library::FlatAttempt {
            used: vec!["a0".into(), "a1".into()],
            disagreement: vec![30, 30],
            drift_threshold: 16,
            feature_pixels: 80,
            ..Default::default()
        },
        ..Default::default()
    };
    let decision = library::flat_retry(request).unwrap();
    assert_eq!(decision.action, library::FlatRetryAction::TryAlternate);
    assert_eq!(decision.inputs, ["b0", "b1"]);
}
