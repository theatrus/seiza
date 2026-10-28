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
