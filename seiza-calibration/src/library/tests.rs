use super::*;
use supplied::{Association, Associations};
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
        dark_subtracted: false,
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
    let report = coherent_cohort(CalibrationKind::Dark, &frames, &Rules::default());
    assert_eq!(report.kept.len(), 2);
    assert!(report.dropped.is_empty());

    // A lone nearest stray does not make a master; the next mode does.
    let frames: Vec<CalibrationFrame> = std::iter::once(dark(0, Some("High Gain Mode")))
        .chain((1..6).map(|id| dark(id, Some("Extend Fullwell 2CMS"))))
        .collect();
    let report = coherent_cohort(CalibrationKind::Dark, &frames, &Rules::default());
    assert_eq!(report.kept.len(), 5);
    assert!(
        report
            .kept
            .iter()
            .all(|frame| frame.readout_mode_name.as_deref() == Some("Extend Fullwell 2CMS"))
    );
}

// theatrus/psf-guard#1021: a flat master integrated at gain 0 from flats and
// flat darks of its own, for lights shot at gain 100.
#[test]
fn an_external_flat_master_at_another_gain_serves_only_when_allowed() {
    let frame = |id: i64, kind: CalibrationKind, is_master: bool| CalibrationFrame {
        id,
        dark_subtracted: is_master,
        frame_uuid: format!("f{id}"),
        rig_uuid: "r".into(),
        kind,
        source_path: format!("/frame-{id}.fits").into(),
        source_fingerprint: "x".into(),
        captured_at: Some(1_000_000_000 + id),
        camera: Some("ASI6200MM Pro".into()),
        telescope: Some("SCA260".into()),
        width: Some(9576),
        height: Some(6388),
        channels: Some(1),
        binning_x: Some(1),
        binning_y: Some(1),
        gain: Some(0),
        offset: Some(0),
        is_master,
        exposure_s: Some(3.9),
        filter: Some("Ha".into()),
        ..Default::default()
    };
    let light = Light {
        rig_uuid: "r".into(),
        camera: Some("ASI6200MM Pro".into()),
        telescope: Some("SCA260".into()),
        filter: Some("Ha".into()),
        width: Some(9576),
        height: Some(6388),
        channels: Some(1),
        binning_x: Some(1),
        binning_y: Some(1),
        gain: Some(100),
        offset: Some(80),
        exposure_s: Some(3.9),
        timestamp: Some(1_000_050_000),
        ..Default::default()
    };
    let master = frame(1, CalibrationKind::Flat, true);
    let raw_flat = frame(2, CalibrationKind::Flat, false);
    let dark_master = frame(3, CalibrationKind::Dark, true);
    let library = Library::new(
        vec![master.clone(), raw_flat.clone(), dark_master.clone()],
        Default::default(),
        [],
    );
    let ids = |frames: &[CalibrationFrame]| -> Vec<i64> { frames.iter().map(|f| f.id).collect() };

    let strict = Arc::new(Rules::default());
    assert!(!strict.admits_sensor(&light, &master));
    assert_eq!(strict.matching_digest(), None);
    let selected = library.select(&light, Arc::clone(&strict));
    assert!(selected.flat.is_empty());
    assert!(!selected.match_tolerances().allow_flat_gain_offset_mismatch);

    let allowed = RuleBook::new(
        rules::RuleValues::from([(
            "allow_flat_gain_offset_mismatch".to_string(),
            serde_json::json!(true),
        )]),
        Default::default(),
    )
    .for_rig("r");
    assert!(allowed.admits_sensor(&light, &master));
    // Only the master's gain and offset are let go: its camera still counts.
    let other_camera = CalibrationFrame {
        camera: Some("ASI2600MM Pro".into()),
        ..master.clone()
    };
    assert!(!allowed.admits_sensor(&light, &other_camera));
    // A master that does not record its dark removed keeps its offset, and
    // the stack's gate would refuse it: selection does too.
    let not_dark_subtracted = CalibrationFrame {
        dark_subtracted: false,
        ..master.clone()
    };
    assert!(!allowed.admits_sensor(&light, &not_dark_subtracted));
    // Raw flats are integrated with the light's own settings, and darks
    // subtract the light's offset: both still match exactly.
    assert!(!allowed.admits_sensor(&light, &raw_flat));
    assert!(!allowed.admits_sensor(&light, &dark_master));

    let selected = library.select(&light, Arc::clone(&allowed));
    assert_eq!(ids(&selected.flat), vec![1]);
    assert!(selected.dark.is_empty());
    // The stack gate that judges the light against the master lets it go
    // too, and the stack's identity changes with the setting.
    assert!(selected.match_tolerances().allow_flat_gain_offset_mismatch);
    assert!(allowed.matching_digest().is_some());

    // A set supplied whole, here the master alone, is taken on the same
    // terms.
    let one_set = Associations {
        flat: Association::OneSet,
        ..Default::default()
    };
    let supplied = Library::new(vec![master.clone()], Default::default(), []);
    let associated = supplied.select_associated(&light, Arc::clone(&allowed), one_set);
    assert_eq!(ids(&associated.flat), vec![1]);
    let associated = supplied.select_associated(&light, Arc::clone(&strict), one_set);
    assert!(associated.flat.is_empty());

    // The setting reaches a rig through its own layer too.
    let rig_only = RuleBook::new(
        Default::default(),
        std::collections::BTreeMap::from([(
            "r".to_string(),
            rules::RigLayer {
                name: Some("SCA260 · ASI6200MM Pro".into()),
                values: rules::RuleValues::from([(
                    "allow_flat_gain_offset_mismatch".to_string(),
                    serde_json::json!(true),
                )]),
            },
        )]),
    );
    assert!(rig_only.for_rig("r").admits_sensor(&light, &master));
    assert!(!rig_only.for_rig("another").admits_sensor(&light, &master));
}

// Dark-flats calibrate the raw flats a master is integrated from, so they
// pair with the nearest raw flat, never with an external master nearer the
// light at another exposure, gain or offset.
#[test]
fn dark_flats_pair_with_the_raw_flats_not_a_nearer_master() {
    let frame = |id: i64, kind: CalibrationKind, gain: i64, exposure: f64| CalibrationFrame {
        id,
        frame_uuid: format!("f{id}"),
        rig_uuid: "r".into(),
        kind,
        source_path: format!("/frame-{id}.fits").into(),
        source_fingerprint: "x".into(),
        captured_at: Some(1_000_000_000 + id),
        camera: Some("Camera".into()),
        width: Some(3000),
        height: Some(2000),
        channels: Some(1),
        binning_x: Some(1),
        binning_y: Some(1),
        gain: Some(gain),
        offset: Some(if gain == 0 { 0 } else { 80 }),
        exposure_s: Some(exposure),
        camera_temp: Some(-10.0),
        filter: Some("Ha".into()),
        ..Default::default()
    };
    // The master, shot nearest the light, at gain 0 and 3.9 s.
    let master = CalibrationFrame {
        is_master: true,
        dark_subtracted: true,
        captured_at: Some(1_000_000_100),
        ..frame(1, CalibrationKind::Flat, 0, 3.9)
    };
    let mut frames = vec![master];
    // Raw flats at the light's gain, 2 s, with their dark-flats, and the
    // master's own gain-0, 3.9 s dark-flats.
    frames.extend((10..13).map(|id| frame(id, CalibrationKind::Flat, 100, 2.0)));
    frames.extend((20..23).map(|id| frame(id, CalibrationKind::DarkFlat, 100, 2.0)));
    frames.extend((30..33).map(|id| frame(id, CalibrationKind::DarkFlat, 0, 3.9)));
    let library = Library::new(frames, Default::default(), []);
    let light = Light {
        rig_uuid: "r".into(),
        camera: Some("Camera".into()),
        filter: Some("Ha".into()),
        width: Some(3000),
        height: Some(2000),
        channels: Some(1),
        binning_x: Some(1),
        binning_y: Some(1),
        gain: Some(100),
        offset: Some(80),
        exposure_s: Some(300.0),
        camera_temp: Some(-10.0),
        timestamp: Some(1_000_000_200),
        ..Default::default()
    };
    let allowed = RuleBook::new(
        rules::RuleValues::from([(
            "allow_flat_gain_offset_mismatch".to_string(),
            serde_json::json!(true),
        )]),
        Default::default(),
    )
    .for_rig("r");
    let selected = library.select(&light, allowed);
    assert_eq!(
        selected.flat.first().map(|f| f.id),
        Some(1),
        "the master is nearest"
    );
    let mut dark_flats: Vec<i64> = selected.dark_flat.iter().map(|f| f.id).collect();
    dark_flats.sort_unstable();
    assert_eq!(dark_flats, vec![20, 21, 22]);
}
