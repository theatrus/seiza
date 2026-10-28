// The native library API as a Rust host calls it, without the C ABI's
// JSON calls.
use seiza_calibration::flat_response::groups::{self, NightBoundary};
use seiza_calibration::library::rules::{RigLayer, RuleBook, RuleValues};
use seiza_calibration::library::{
    Application, Association, Associations, CalibrationFrame, CalibrationKind,
    CalibrationSelection, FlatAttempt, FlatRetry, FlatRetryAction, FlatRetryPolicy, Library, Light,
    NightShifts, Session, ValidDirection, plan_flat_sessions,
};
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;

const DAY: i64 = 86400;

fn frame(id: usize, kind: CalibrationKind, at: i64) -> CalibrationFrame {
    CalibrationFrame {
        id: id as i64,
        frame_uuid: format!("{kind:?}-{id}"),
        rig_uuid: "rig".into(),
        kind,
        source_path: format!("/{kind:?}-{id}.fits").into(),
        source_fingerprint: format!("stamp-{id}"),
        source_verified: true,
        captured_at: Some(1_700_000_000 + at),
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

fn light(at: i64) -> Light {
    let f = frame(0, CalibrationKind::Flat, at);
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

fn library(frames: Vec<CalibrationFrame>) -> Library {
    Library::new(frames, Arc::default(), Vec::new())
}

#[test]
fn a_blank_readout_mode_name_names_no_mode_wherever_frames_come_from() {
    let mut frames: Vec<_> = (0..4)
        .map(|i| frame(i, CalibrationKind::Dark, 60 * i as i64))
        .collect();
    frames[0].readout_mode_name = Some("High Gain".into());
    frames[1].readout_mode_name = Some("High Gain".into());
    frames[2].readout_mode_name = Some(" ".into());
    frames[3].readout_mode_name = Some(String::new());
    let rules = RuleBook::default().global();
    // Chosen frames, as a host gives them, cluster as the library's do.
    let selected = CalibrationSelection::of_frames(
        frames.clone(),
        Arc::clone(&rules),
        Associations::default(),
    );
    let cohort =
        seiza_calibration::library::coherent_cohort(CalibrationKind::Dark, &selected.dark, &rules);
    assert_eq!(
        cohort.kept.len(),
        4,
        "blank names ride along with the named mode"
    );
    let from_library = library(frames);
    let cohort = seiza_calibration::library::coherent_cohort(
        CalibrationKind::Dark,
        from_library.frames(),
        &rules,
    );
    assert_eq!(cohort.kept.len(), 4);
}

#[test]
fn a_flat_retry_for_a_session_keeps_its_policy() {
    let flats: Vec<_> = [0, 60, 10 * DAY, 10 * DAY + 60]
        .into_iter()
        .enumerate()
        .map(|(i, at)| frame(i, CalibrationKind::Flat, at))
        .collect();
    let first = FlatAttempt {
        used: vec![flats[0].frame_uuid.clone(), flats[1].frame_uuid.clone()],
        disagreement: vec![10, 10],
        drift_threshold: 5,
        feature_pixels: 40,
        ..Default::default()
    };
    let rules = RuleBook::default().global();
    let automatic = FlatRetry::new(
        first.clone(),
        &flats,
        &[],
        &rules,
        FlatRetryPolicy::default(),
    )
    .unwrap()
    .next();
    assert_eq!(automatic.action, FlatRetryAction::TryAlternate);
    // A supplied set's session never takes another session's flats.
    let session = Session {
        key: "one-set".into(),
        tolerances: Default::default(),
        recipes: Vec::new(),
        application: Application::default(),
        flat_stability: FlatRetryPolicy {
            try_next_session: false,
            ..FlatRetryPolicy::default()
        },
        flat_held_back: Vec::new(),
    };
    let decision = FlatRetry::for_session(first, &flats, &session, &rules)
        .unwrap()
        .next();
    assert_eq!(decision.action, FlatRetryAction::PublishFirst);
}

#[test]
fn a_supplied_set_keeps_to_the_lights_side_of_a_marked_change_and_reports_once() {
    let mut flats: Vec<_> = (0..4)
        .map(|i| frame(i, CalibrationKind::Flat, i as i64 * DAY))
        .collect();
    // The last flat serves only lights shot after it.
    flats[3].valid_direction = Some(ValidDirection::Forward);
    let mut dark_flat = frame(10, CalibrationKind::DarkFlat, 0);
    dark_flat.kept_out = Some("by hand".into());
    let mut frames = flats;
    frames.push(dark_flat);
    let library = library(frames);
    let associations = Associations {
        flat: Association::OneSet,
        dark: Association::Automatic,
    };
    let selected = library.select_associated(
        &light(DAY + 600),
        RuleBook::default().global(),
        associations,
    );
    let uuids: Vec<_> = selected
        .flat
        .iter()
        .map(|f| f.frame_uuid.as_str())
        .collect();
    assert_eq!(uuids.len(), 3, "{uuids:?}");
    assert!(!uuids.contains(&"Flat-3"));
    let kept: Vec<_> = selected
        .kept_out
        .iter()
        .map(|f| f.frame_uuid.as_str())
        .collect();
    assert_eq!(kept, ["DarkFlat-10"]);
}

#[test]
fn flat_sessions_do_not_depend_on_the_order_frames_are_given() {
    let flats: Vec<_> = (0..4)
        .map(|i| {
            let mut f = frame(i, CalibrationKind::Flat, 0);
            f.id = 0;
            f
        })
        .collect();
    let nights = NightShifts::new(Vec::<NightBoundary>::new());
    let rules = RuleBook::default();
    let keys = |frames: &[CalibrationFrame]| {
        plan_flat_sessions(frames, &nights, &rules)
            .into_iter()
            .map(|plan| plan.session_key)
            .collect::<Vec<_>>()
    };
    let mut reversed = flats.clone();
    reversed.reverse();
    assert_eq!(keys(&flats), keys(&reversed));
}

#[test]
fn grading_settings_differ_only_when_what_they_resolve_to_does() {
    let values = |pairs: &[(&str, serde_json::Value)]| -> RuleValues {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    };
    let book = |global: RuleValues, rig: RuleValues| {
        RuleBook::new(
            global,
            BTreeMap::from([(
                "rig".to_string(),
                RigLayer {
                    name: None,
                    values: rig,
                },
            )]),
        )
    };
    let base = book(
        values(&[("rotation_tolerance_deg", json!(2))]),
        RuleValues::new(),
    );
    let same = book(
        values(&[("rotation_tolerance_deg", json!(2.0))]),
        values(&[("rotation_tolerance_deg", json!(2))]),
    );
    assert!(!base.grades_differ(&same));
    let unreadable = book(
        values(&[
            ("rotation_tolerance_deg", json!(2)),
            ("grade_bias_level", json!("high")),
        ]),
        RuleValues::new(),
    );
    assert!(!base.grades_differ(&unreadable));
    let wider = book(
        values(&[("rotation_tolerance_deg", json!(4))]),
        RuleValues::new(),
    );
    assert!(base.grades_differ(&wider));
}

#[test]
fn a_native_host_is_refused_what_the_c_abi_is_refused() {
    let twin = vec![
        frame(0, CalibrationKind::Bias, 0),
        frame(0, CalibrationKind::Bias, 0),
    ];
    assert!(Library::checked(twin, Arc::default(), Vec::new()).is_err());
    let mut unknown = frame(1, CalibrationKind::Bias, 0);
    unknown.exposure_s = Some(f64::NAN);
    assert!(Library::checked(vec![unknown], Arc::default(), Vec::new()).is_err());
    let mut dim = light(0);
    dim.width = Some(0);
    assert!(dim.check().is_err());
    let boundary = |seconds| NightBoundary {
        rig_uuid: "rig".into(),
        frame_uuid: "f".into(),
        seconds,
    };
    assert!(NightShifts::checked([boundary(43200), boundary(40000)]).is_err());
    assert!(NightShifts::checked([boundary(200_000)]).is_err());
    assert!(NightShifts::checked([boundary(43200)]).is_ok());
    let book = Arc::new(groups::Book::default());
    assert!(
        Library::new(Vec::new(), book, Vec::new())
            .check_limits(&RuleBook::default())
            .is_ok()
    );
}
