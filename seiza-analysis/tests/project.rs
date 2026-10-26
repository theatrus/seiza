use seiza_analysis::{AnalysisOptions, analyze};
use seiza_fits::{F32ImageData, HeaderValue, WriteHeaderCard};
use std::path::Path;

fn write_frame(
    path: &Path,
    seed: u64,
    shift: (f64, f64),
    throughput: f64,
    filter: Option<&str>,
    size: usize,
) {
    let mut state = seed.wrapping_mul(0xd1342543de82ef95);
    let mut uniform = || {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
    };
    let mut data: Vec<f32> = (0..size * size)
        .map(|_| {
            let noise = (-2.0 * uniform().ln()).sqrt() * (std::f64::consts::TAU * uniform()).cos();
            (1000.0 + noise * 10.0) as f32
        })
        .collect();
    for i in 0..36 {
        let x = 35.0 + (i % 6) as f64 * 58.0 + ((i * 17) % 13) as f64 + shift.0;
        let y = 35.0 + (i / 6) as f64 * 58.0 + ((i * 11) % 17) as f64 + shift.1;
        let amplitude = 1800.0 + (i * 79 % 2000) as f64;
        for yy in (y as usize).saturating_sub(14)..(y as usize + 15).min(size) {
            for xx in (x as usize).saturating_sub(14)..(x as usize + 15).min(size) {
                let r2 = (xx as f64 - x).powi(2) + (yy as f64 - y).powi(2);
                data[yy * size + xx] += (amplitude * (-r2 / 18.0).exp()) as f32;
            }
        }
    }
    for v in &mut data {
        *v *= throughput as f32;
    }
    let mut headers = vec![
        WriteHeaderCard::new("EXPTIME", HeaderValue::Float(60.0)),
        WriteHeaderCard::new(
            "DATE-OBS",
            HeaderValue::String("2026-01-02T03:04:05".into()),
        ),
    ];
    if let Some(filter) = filter {
        headers.push(WriteHeaderCard::new(
            "FILTER",
            HeaderValue::String(filter.into()),
        ));
    }
    seiza_image_io::save_f32(
        path,
        size,
        size,
        F32ImageData::Mono(&data),
        &seiza_image_io::SaveOptions {
            headers: &headers,
            ..Default::default()
        },
    )
    .unwrap();
}

#[test]
fn independent_noise_falls_with_depth_and_flux_units_survive_scaling() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..12 {
        write_frame(
            &dir.path().join(format!("{i:02}.fits")),
            i,
            (0.0, 0.0),
            if i % 2 == 0 { 1.0 } else { 0.001 },
            Some("Ha"),
            400,
        );
    }
    let options = AnalysisOptions {
        samples: 65536,
        ..Default::default()
    };
    let report = analyze(&[dir.path().to_path_buf()], &options).unwrap();
    assert_eq!(report.groups.len(), 1, "{:#?}", report.frames);
    let group = &report.groups[0];
    assert_eq!(group.accepted_frames, 12, "{:#?}", report.frames);
    for row in report.frames.iter().skip(1) {
        assert_eq!(
            row.normalization.as_deref(),
            Some("stellar_flux"),
            "{row:?}"
        );
    }
    let gain = report.frames[1].gain.unwrap();
    assert!((gain / 1000.0 - 1.0).abs() < 0.03, "gain {gain}");
    assert!(group.curve[0].noise.is_none());
    assert!(group.curve[2].noise.is_none());
    let first = group.curve[1].noise.unwrap();
    let last = group.curve[11].noise.unwrap();
    assert!((first - 10.0 / 2f64.sqrt()).abs() < 1.0, "{first}");
    assert!(
        (last / first - (2f64 / 12.0).sqrt()).abs() < 0.08,
        "{first} -> {last}"
    );
    assert_eq!(group.integration_seconds, 720.0);
    assert_eq!(group.unknown_exposures, 0);
    assert_eq!(
        report.frames[0].capture_time.as_deref(),
        Some("2026-01-02T03:04:05")
    );
}

#[test]
fn discovery_filters_xisf_deduplication_and_failed_frames() {
    let dir = tempfile::tempdir().unwrap();
    write_frame(
        &dir.path().join("a.fits"),
        1,
        (0.0, 0.0),
        1.0,
        Some("Ha"),
        400,
    );
    write_frame(
        &dir.path().join("b.xisf"),
        2,
        (3.0, -2.0),
        0.8,
        Some("ha"),
        400,
    );
    write_frame(
        &dir.path().join("c.fits"),
        3,
        (0.0, 0.0),
        1.0,
        Some("OIII"),
        400,
    );
    std::fs::write(dir.path().join("broken.fits"), b"bad").unwrap();
    let list = dir.path().join("frames.txt");
    std::fs::write(
        &list,
        "# relative paths\nb.xisf\na.fits\na.fits\nbroken.fits\nc.fits\n",
    )
    .unwrap();
    let report = analyze(
        &[list],
        &AnalysisOptions {
            filter: Some("HA".into()),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(report.frames.len(), 4);
    assert_eq!(report.groups.len(), 1);
    assert_eq!(report.groups[0].accepted_frames, 2, "{:#?}", report.frames);
    assert!(report.frames[1].registration_rms_pixels.unwrap() < 0.2);
    assert_eq!(report.frames[2].status, "rejected");
    assert_eq!(report.frames[3].status, "excluded");
    assert!(report.frames[2].error.is_some());
}

#[test]
fn invalid_requests_and_unreadable_inputs_are_distinct() {
    assert!(analyze(&[], &AnalysisOptions::default()).is_err());
    assert!(
        analyze(
            &["missing.fits".into()],
            &AnalysisOptions {
                samples: 0,
                ..Default::default()
            }
        )
        .is_err()
    );
    let report = analyze(&["missing.fits".into()], &AnalysisOptions::default()).unwrap();
    assert_eq!(report.frames[0].status, "rejected");
    assert!(report.groups.is_empty());
}

#[test]
fn common_footprint_excludes_every_frames_invalid_patches() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..4 {
        let path = dir.path().join(format!("{i}.fits"));
        write_frame(&path, i, (0.0, 0.0), 1.0, None, 400);
        if i == 0 || i == 3 {
            let raw = seiza_image_io::read_astronomy(&path).unwrap();
            let headers: Vec<_> = raw
                .image
                .headers
                .iter()
                .filter(|(k, _)| matches!(k.as_str(), "EXPTIME" | "DATE-OBS"))
                .map(|(k, v)| WriteHeaderCard::new(k, v.clone()))
                .collect();
            let mut data = raw.image.into_physical_f32();
            data[200 * 400 + if i == 0 { 200 } else { 232 }] = f32::NAN;
            seiza_image_io::save_f32(
                &path,
                400,
                400,
                F32ImageData::Mono(&data),
                &seiza_image_io::SaveOptions {
                    headers: &headers,
                    ..Default::default()
                },
            )
            .unwrap();
        }
    }
    let report = analyze(&[dir.path().to_path_buf()], &AnalysisOptions::default()).unwrap();
    let group = &report.groups[0];
    assert_eq!(group.accepted_frames, 4, "{:#?}", report.frames);
    assert_eq!(group.candidate_patches, 100);
    assert_eq!(group.common_patches, 98);
    assert!(group.curve[1].noise.is_some());
    assert!(group.curve[3].noise.is_some());
    assert!(group.filter.is_none());
    assert!(!group.warnings.is_empty());
}

#[test]
fn small_color_bayer_and_mismatched_inputs_are_reported() {
    let dir = tempfile::tempdir().unwrap();
    write_frame(
        &dir.path().join("a.fits"),
        0,
        (0.0, 0.0),
        1.0,
        Some("Ha"),
        400,
    );
    write_frame(
        &dir.path().join("b.fits"),
        1,
        (0.0, 0.0),
        1.0,
        Some("Ha"),
        384,
    );
    let filter = WriteHeaderCard::new("FILTER", HeaderValue::String("Ha".into()));
    let bayer = WriteHeaderCard::new("BAYERPAT", HeaderValue::String("RGGB".into()));
    for (name, size, rgb, headers) in [
        ("color.fits", 40, true, vec![filter.clone()]),
        ("bayer.fits", 40, false, vec![filter.clone(), bayer]),
        ("tiny.fits", 1, false, vec![filter]),
    ] {
        let data = vec![0.0; size * size * if rgb { 3 } else { 1 }];
        seiza_image_io::save_f32(
            &dir.path().join(name),
            size,
            size,
            if rgb {
                F32ImageData::RgbInterleaved(&data)
            } else {
                F32ImageData::Mono(&data)
            },
            &seiza_image_io::SaveOptions {
                headers: &headers,
                ..Default::default()
            },
        )
        .unwrap();
    }
    let report = analyze(&[dir.path().to_path_buf()], &AnalysisOptions::default()).unwrap();
    assert_eq!(report.frames.len(), 5);
    assert_eq!(
        report
            .frames
            .iter()
            .filter(|f| f.status == "rejected")
            .count(),
        4
    );
    assert_eq!(report.groups[0].accepted_frames, 1);
    assert!(report.groups[0].curve[0].noise.is_none());
}
