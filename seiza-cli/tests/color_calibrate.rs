use seiza_fits::{F32ImageData, FitsImage, HeaderValue, WriteHeaderCard, write_f32_image};
use std::fmt::Write as _;
use std::process::Command;

/// A camera that records a solar-coloured star at R/G 0.6 and B/G 1.4:
/// calibration should find gains of 1/0.6 and 1/1.4, from a Gaia CSV
/// supplied offline.
#[test]
fn color_calibrate_cli_recovers_channel_gains_from_a_gaia_csv() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("rgb.fits");
    let output = directory.path().join("calibrated.fits");
    let report = directory.path().join("report.json");
    let gaia = directory.path().join("gaia.csv");
    let (width, height) = (1200, 1000);
    let wcs =
        seiza::Wcs::from_center_scale_rotation((56.75, 24.1), (600.0, 500.0), 3.0, 10.0, false);
    let sky = [120.0_f32, 100.0, 90.0];
    let mut data = (0..width * height * 3)
        .map(|index| sky[index % 3] + ((index * 7919) % 13) as f32 * 0.3)
        .collect::<Vec<_>>();
    let mut csv =
        "ra,dec,pmra,pmdec,phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag,ruwe\n".to_owned();
    for index in 0..140 {
        let x = 30.0 + ((index * 7919) % 1140) as f64 + ((index % 7) as f64) * 0.13;
        let y = 30.0 + ((index * 6271) % 940) as f64 + ((index % 5) as f64) * 0.17;
        let bp_rp = 0.1 + ((index * 37) % 160) as f64 / 100.0;
        let g = 9.0 + ((index * 13) % 40) as f64 / 10.0;
        let (ra, dec) = wcs.pixel_to_world(x, y);
        writeln!(
            csv,
            "{ra},{dec},0,0,{g},{},{},1.0",
            g + 0.3,
            g + 0.3 - bp_rp
        )
        .unwrap();
        let flux = 2.0e5 * 10f64.powf(-0.4 * (g - 9.0));
        let delta = bp_rp - 0.82;
        let ratios = [
            0.6 * 10f64.powf(0.2 * delta),
            1.0,
            1.4 * 10f64.powf(-0.32 * delta),
        ];
        for py in (y as usize).saturating_sub(10)..(y as usize + 11).min(height) {
            for px in (x as usize).saturating_sub(10)..(x as usize + 11).min(width) {
                let r2 = (px as f64 - x).powi(2) + (py as f64 - y).powi(2);
                let profile = (-r2 / 4.5).exp() / (std::f64::consts::PI * 4.5);
                for (channel, ratio) in ratios.iter().enumerate() {
                    data[(py * width + px) * 3 + channel] += (flux * ratio * profile) as f32;
                }
            }
        }
    }
    std::fs::write(&gaia, csv).unwrap();
    let cards = wcs
        .fits_header_cards()
        .into_iter()
        .map(|(key, value)| {
            let value = match value {
                seiza::FitsCardValue::Text(text) => HeaderValue::String(text.into()),
                seiza::FitsCardValue::Integer(number) => HeaderValue::Integer(i64::from(number)),
                seiza::FitsCardValue::Number(number) => HeaderValue::Float(number),
            };
            WriteHeaderCard::new(&key, value)
        })
        .collect::<Vec<_>>();
    write_f32_image(
        &input,
        width,
        height,
        F32ImageData::RgbInterleaved(&data),
        &cards,
    )
    .unwrap();

    let run = |args: &[&str]| {
        let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
            .args(args)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    };
    let check_gains = |report: &std::path::Path| {
        let report: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(report).unwrap()).unwrap();
        let gains = report["gains"].as_array().unwrap();
        let gain = |channel: usize| gains[channel].as_f64().unwrap();
        assert!((gain(0) - 1.0 / 0.6).abs() < 0.03, "{gains:?}");
        assert!((gain(2) - 1.0 / 1.4).abs() < 0.02, "{gains:?}");
    };
    run(&[
        "color-calibrate",
        input.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
        "--report",
        report.to_str().unwrap(),
        "--gaia-csv",
        gaia.to_str().unwrap(),
    ]);
    check_gains(&report);

    // The same stars as an offline photometry catalog give the same gains.
    let chunks = directory.path().join("chunks");
    std::fs::create_dir(&chunks).unwrap();
    std::fs::copy(&gaia, chunks.join("gaiaphot-0000.csv")).unwrap();
    let catalog = directory.path().join("stars-gaia-photometry.bin");
    run(&[
        "build-data",
        "gaia-photometry",
        "--input",
        chunks.to_str().unwrap(),
        "--output",
        catalog.to_str().unwrap(),
    ]);
    let offline_report = directory.path().join("offline.json");
    run(&[
        "color-calibrate",
        input.to_str().unwrap(),
        "--output",
        directory.path().join("offline.fits").to_str().unwrap(),
        "--report",
        offline_report.to_str().unwrap(),
        "--gaia-catalog",
        catalog.to_str().unwrap(),
    ]);
    check_gains(&offline_report);
    let written = FitsImage::open(&output).unwrap();
    let header = |key: &str| {
        written
            .headers
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value.clone())
    };
    assert_eq!(
        header("COLORCAL").and_then(|value| value.as_str().map(str::to_owned)),
        Some("Gaia DR3 BP-RP".to_owned())
    );
    assert!(
        header("CRVAL1").is_some(),
        "the astrometric solution is kept"
    );
}
