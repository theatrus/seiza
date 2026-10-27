use seiza_calibration::flat_response::groups::{self, Request};
use serde_json::{Value, json};

fn request() -> Request {
    serde_json::from_str(include_str!("fixtures/flat-groups/request.json")).unwrap()
}

#[test]
fn the_psf_guard_fixture_keeps_its_groups_changes_and_uncertainty() {
    let expected: Value =
        serde_json::from_str(include_str!("fixtures/flat-groups/expected.json")).unwrap();
    let snapshot = groups::evaluate(request()).unwrap();
    let actual = serde_json::to_value(&snapshot).unwrap();
    for field in ["families", "boundaries", "events"] {
        assert_eq!(actual[field], expected[field], "{field}");
    }
    for (index, name) in ["before", "during", "after"].iter().enumerate() {
        assert_eq!(
            actual["lights"][index]["events"][0]["placement"],
            expected["placements"][0][name]
        );
    }
    assert_eq!(
        snapshot.lights[0].automatic_family.as_ref().unwrap().family,
        0
    );
    assert_eq!(
        snapshot.lights[1].events[0].map_source,
        Some(("b".into(), "c".into()))
    );
}

#[test]
fn incomplete_measurements_do_not_claim_angle_equivalence() {
    let mut request = request();
    request.comparisons.clear();
    request.lights.clear();
    let snapshot = groups::evaluate(request).unwrap();
    assert_eq!(snapshot.families.len(), 2);
    assert!(snapshot.families.iter().all(|f| f.alike_ranges.is_empty()));
}

#[test]
fn stale_choices_remain_orphaned_and_do_not_change_automatic_results() {
    let mut request = request();
    request.group_overrides = serde_json::from_value(json!([{
        "anchor_frame":"gone", "session_key":"gone", "family_key":"rig-ha-sensor",
        "rig_uuid":"rig", "choice":"split", "label":"old choice", "chosen_at":1
    }]))
    .unwrap();
    let snapshot = groups::evaluate(request).unwrap();
    assert!(matches!(
        snapshot.overrides[0].status,
        groups::OverrideStatus::Orphaned { .. }
    ));
    assert_eq!(snapshot.families[0].groups, vec![vec![0, 1], vec![2]]);
}

#[test]
fn a_join_override_changes_groups_and_a_reset_restores_the_measurement() {
    let mut r = request();
    let choice = json!({
        "anchor_frame":"c-0", "session_key":"s-c", "family_key":"rig-ha-sensor",
        "rig_uuid":"rig", "choice":"join", "before_frame":"b-0",
        "before_session_key":"s-b", "label":"join", "chosen_at":1
    });
    r.group_overrides = serde_json::from_value(json!([choice])).unwrap();
    let actual = groups::evaluate(r.clone()).unwrap();
    assert_eq!(actual.families[0].groups, vec![vec![0, 1, 2]]);
    assert!(matches!(
        actual.overrides[0].status,
        groups::OverrideStatus::Applied { agrees: false }
    ));
    r.group_overrides[0].choice = None;
    let reset = groups::evaluate(r).unwrap();
    assert_eq!(reset.families[0].groups, vec![vec![0, 1], vec![2]]);
}

#[test]
fn a_light_choice_resolves_separately_from_its_automatic_uncertainty() {
    let mut r = request();
    let baseline = groups::evaluate(r.clone()).unwrap();
    let event = &baseline.events[0];
    r.light_choices = serde_json::from_value(json!([{
        "light_id":7, "file_name":"light-1.fits", "filter":"Ha", "at":4*86400,
        "row": {"light_guid":"guid", "change_key":event.key, "rig_uuid":"rig",
            "change_start":event.start, "change_end":event.end, "choice":"before",
            "label":"before dust moved", "chosen_at":7}
    }]))
    .unwrap();
    let actual = groups::evaluate(r.clone()).unwrap();
    assert_eq!(actual.lights[1].choices.sides.len(), 1);
    assert_eq!(actual.lights[1].choices.sides[0].0, 0);
    assert!(matches!(
        actual.lights[1].events[0].placement,
        Some(groups::Placement::Undetermined { .. })
    ));
    let key = &actual.lights[1].family_probe.as_ref().unwrap().key;
    assert_eq!(key.len(), 24);
    r.light_choices[0].at = Some(4 * 86400 + 3);
    assert!(
        groups::evaluate(r).unwrap().lights[1]
            .choices
            .sides
            .is_empty()
    );
}

#[test]
fn raw_imaging_rows_group_by_profile_and_three_hour_gaps() {
    let mut request = request();
    request.imaging_lights = vec![
        (3, Some("p".into()), 4 * 86400),
        (1, Some("p".into()), 86400),
        (2, Some("p".into()), 87000),
        (4, Some("q".into()), 87000),
    ];
    request.profile_rigs = vec![(Some("p".into()), vec!["rig".into()])];
    let snapshot = groups::evaluate(request).unwrap();
    assert_eq!(snapshot.imaging.len(), 3);
    assert_eq!(snapshot.imaging[0].lights, vec![1, 2]);
    assert_eq!(snapshot.imaging[0].rigs, vec!["rig"]);
    assert_eq!(snapshot.imaging[2].profile.as_deref(), Some("q"));
    assert_eq!(snapshot.sessions_needing_evidence, vec![(0, 1)]);
}

#[test]
fn malformed_requests_fail_before_running_the_policy() {
    let mut r = request();
    r.responses.push(r.responses[0].clone());
    assert!(
        groups::evaluate(r)
            .unwrap_err()
            .to_string()
            .contains("duplicate response")
    );
    let mut r = request();
    r.limits.clear();
    assert!(
        groups::evaluate(r)
            .unwrap_err()
            .to_string()
            .contains("missing rig limits")
    );
    let mut r = request();
    r.lights[0].at = Some(i64::MAX);
    assert!(
        groups::evaluate(r)
            .unwrap_err()
            .to_string()
            .contains("capture time")
    );
    let mut r = request();
    r.limits[0].limits.dust = f64::NAN;
    assert!(
        groups::evaluate(r)
            .unwrap_err()
            .to_string()
            .contains("finite")
    );
    let mut r = request();
    r.lights[0].compatible_families = vec!["absent".into()];
    assert!(
        groups::evaluate(r)
            .unwrap_err()
            .to_string()
            .contains("unknown compatible family")
    );
    let mut r = request();
    r.lights[0].rig_uuid = "other".into();
    assert!(
        groups::evaluate(r)
            .unwrap_err()
            .to_string()
            .contains("another rig or filter")
    );
    assert!(serde_json::from_value::<Request>(json!({"limitts":[]})).is_err());
}
