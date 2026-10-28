use seiza_calibration::library::{Request, evaluate};
use serde_json::Value;
use std::sync::Arc;

#[test]
fn cached_grouping_preserves_library_results_and_rejects_conflicting_inputs() {
    use seiza_calibration::{flat_response::groups, library};
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/library/psf-guard.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let mut request: Request = serde_json::from_value(case["request"].clone()).unwrap();
        let expected = serde_json::to_value(evaluate(request.clone()).unwrap()).unwrap();
        let mut grouping = std::mem::take(&mut request.grouping);
        grouping.limits = grouping
            .responses
            .iter()
            .map(|r| r.rig_uuid.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .map(|rig_uuid| groups::RigLimits {
                rig_uuid,
                limits: groups::Limits {
                    measured: true,
                    dust: 0.01,
                    broad: 0.02,
                    window: 86400,
                    rotation: 2.0,
                },
            })
            .collect();
        let book = Arc::new(groups::prepare(grouping).unwrap());
        let result = library::evaluate_with_book(request.clone(), Arc::clone(&book)).unwrap();
        assert_eq!(
            serde_json::to_value(result).unwrap(),
            expected,
            "{}",
            case["name"]
        );
        if !book.sessions.is_empty() {
            let mut changed = request.clone();
            changed.global_settings.push(library::Setting {
                key: "flat_split_dust_pct".into(),
                value: 5.0.into(),
            });
            assert!(library::evaluate_with_book(changed, Arc::clone(&book)).is_err());
        }
        request.grouping.epoch_samples.push(groups::EpochSample {
            rig_uuid: "r".into(),
            at: 1,
            offset: 2.0,
        });
        assert!(library::evaluate_with_book(request, book).is_err());
    }
}

#[test]
fn resolved_recipes_determine_sessions_and_exposure() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/library/recipes.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let request: Request = serde_json::from_value(case["request"].clone()).unwrap();
        let result = serde_json::to_value(evaluate(request).unwrap()).unwrap();
        assert_eq!(
            result["assignments"], case["assignments"],
            "{}",
            case["name"]
        );
        assert_eq!(
            result["sessions"].as_array().unwrap().len(),
            case["recipes"].as_array().unwrap().len()
        );
        for (session, expected) in result["sessions"]
            .as_array()
            .unwrap()
            .iter()
            .zip(case["recipes"].as_array().unwrap())
        {
            let recipe = session["recipes"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["kind"] == case["kind"])
                .unwrap();
            for key in ["action", "inputs", "exposure_seconds"] {
                assert_eq!(recipe[key], expected[key], "{}: {key}", case["name"]);
            }
            assert_eq!(recipe["requires_source_verification"], false);
        }
    }
}

#[test]
fn identity_tracks_effective_inputs_options_and_gate() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/library/recipes.json")).unwrap();
    let mut request = fixture["cases"][0]["request"].clone();
    let key = |request: &Value| {
        evaluate(serde_json::from_value(request.clone()).unwrap())
            .unwrap()
            .sessions
            .remove(0)
            .key
    };
    let original = key(&request);
    // The large raw cohort is unused under external/prefer.
    request["frames"][0]["source_fingerprint"] = "unused change".into();
    assert_eq!(key(&request), original);
    request["frames"].as_array_mut().unwrap().reverse();
    assert_eq!(key(&request), original);
    request["frames"][1]["source_fingerprint"] = "adopted source changed".into();
    assert_ne!(key(&request), original);

    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/library/psf-guard.json")).unwrap();
    let request = fixture["cases"][9]["request"].clone();
    let original = key(&request);
    let mut changed = request.clone();
    changed["flat_star_masking"] = true.into();
    assert_ne!(key(&changed), original);
    changed = request.clone();
    changed["global_settings"] = serde_json::json!([{"key":"rotation_tolerance_deg","value":3.0}]);
    assert_ne!(key(&changed), original);
    changed = request.clone();
    changed["master_failures"] =
        serde_json::json!([{"session_key":original,"kind":"flat","reason":"failed read"}]);
    assert_eq!(key(&changed), original);
    changed = request;
    changed["pedestal_evidence"] = serde_json::json!([{"session_key":original,"pinned":300.0}]);
    assert_eq!(key(&changed), original);
}

#[test]
fn lights_whose_masters_take_the_same_frames_share_one_session() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/library/recipes.json")).unwrap();
    let mut request = fixture["cases"][1]["request"].clone();
    let mut a = request["lights"][0].clone();
    a["timestamp"] = 20.into();
    let mut b = a.clone();
    b["timestamp"] = 21.into();
    request["lights"] = serde_json::json!([a, b, a]);
    let result = evaluate(serde_json::from_value(request).unwrap()).unwrap();
    // Each light's nearest frame differs, but the master takes the same two
    // frames for both: one master, anchored as the first light orders them.
    assert_eq!(result.selections[0].dark, ["f1", "f2", "f0"]);
    assert_eq!(result.selections[1].dark, ["f2", "f1", "f0"]);
    assert_eq!(result.assignments, [0, 0, 0]);
    assert_eq!(result.sessions.len(), 1);
    assert_eq!(result.sessions[0].recipes[2].inputs, ["f1", "f2"]);
}
