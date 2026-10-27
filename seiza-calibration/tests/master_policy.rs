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
