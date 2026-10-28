//! Deterministic OSC fixtures and repeatable analysis timing, not camera data.
//! cargo run --release -p seiza-analysis --example osc_preview -- NEW_DIRECTORY
use seiza_analysis::{AnalysisOptions, analyze, render_html};
use seiza_fits::{F32ImageData, HeaderValue, WriteHeaderCard};
use std::path::PathBuf;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("provide a new output directory")?,
    );
    std::fs::create_dir(&root)?;
    let side = 800;
    for kind in ["cfa", "rgb"] {
        let dir = root.join(kind);
        std::fs::create_dir(&dir)?;
        for frame in 0..6 {
            let mut state = 90210u64 + frame * 999;
            let mut uniform = || {
                state = state.wrapping_add(0x9e3779b97f4a7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                z ^= z >> 31;
                ((z >> 11) as f64 + 1.) / ((1u64 << 53) as f64 + 1.)
            };
            let channels = if kind == "cfa" { 1 } else { 3 };
            let mut pixels = vec![0.; side * side * channels];
            let stars = (0..64)
                .map(|i| {
                    (
                        45. + (i % 8) as f64 * 95. + ((i * 17) % 13) as f64,
                        45. + (i / 8) as f64 * 95. + ((i * 11) % 17) as f64,
                        1800. + ((i * 79) % 2000) as f64,
                    )
                })
                .collect::<Vec<_>>();
            for y in 0..side {
                for x in 0..side {
                    let light = stars
                        .iter()
                        .map(|&(cx, cy, a)| {
                            let d = (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2);
                            if d < 1800. { a * (-d / 72.).exp() } else { 0. }
                        })
                        .sum::<f64>();
                    for channel in 0..channels {
                        let c = if kind == "cfa" {
                            [[0, 1], [1, 2]][y % 2][x % 2]
                        } else {
                            channel
                        };
                        let noise = (-2. * uniform().ln()).sqrt()
                            * (std::f64::consts::TAU * uniform()).cos();
                        pixels[(y * side + x) * channels + channel] = (1000.
                            + c as f64 * 200.
                            + light * [0.8, 1., 0.65][c]
                            + noise * [6., 10., 14.][c])
                            as f32;
                    }
                }
            }
            let mut headers = vec![
                WriteHeaderCard::new("FILTER", HeaderValue::String("OSC fixture".into())),
                WriteHeaderCard::new("EXPTIME", HeaderValue::Float(60.)),
                WriteHeaderCard::new(
                    "DATE-OBS",
                    HeaderValue::String(format!("2026-01-02T03:{:02}:00", frame)),
                ),
            ];
            if kind == "cfa" {
                headers.push(WriteHeaderCard::new(
                    "BAYERPAT",
                    HeaderValue::String("RGGB".into()),
                ));
            }
            seiza_image_io::save_f32(
                &dir.join(format!("{frame:02}.fits")),
                side,
                side,
                if channels == 1 {
                    F32ImageData::Mono(&pixels)
                } else {
                    F32ImageData::RgbInterleaved(&pixels)
                },
                &seiza_image_io::SaveOptions {
                    headers: &headers,
                    ..Default::default()
                },
            )?;
        }
    }
    let options = AnalysisOptions {
        samples: 32768,
        ..Default::default()
    };
    // One unrecorded warm-up per layout, followed by alternating measured runs.
    for kind in ["cfa", "rgb"] {
        analyze(&[root.join(kind)], &options)?;
    }
    let mut runs = Vec::new();
    for repeat in 0..3 {
        for kind in if repeat % 2 == 0 {
            ["cfa", "rgb"]
        } else {
            ["rgb", "cfa"]
        } {
            let report = analyze(&[root.join(kind)], &options)?;
            assert_eq!(report.groups.len(), if kind == "cfa" { 4 } else { 3 });
            assert!(
                report
                    .groups
                    .iter()
                    .all(|g| g.depth_frames == 6 && g.curve[5].noise.is_some()),
                "{:?}",
                report.frames
            );
            runs.push(serde_json::json!({"kind":kind,"repeat":repeat,"timings":report.timings,"source_count":6,"component_records":report.frames.len(),"width":side,"height":side}));
            std::fs::write(
                root.join(format!("{kind}.json")),
                serde_json::to_vec_pretty(&report)?,
            )?;
            std::fs::write(root.join(format!("{kind}.html")), render_html(&report)?)?;
            eprintln!(
                "{kind} repeat {repeat}: {:.3}s",
                report.timings.total_seconds
            );
        }
    }
    std::fs::write(root.join("timings.json"), serde_json::to_vec_pretty(&runs)?)?;
    Ok(())
}
