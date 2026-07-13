//! Integration suite against real camera images hosted at
//! downloads.seiza.fyi. Ignored by default (network + ~250 MB download,
//! cached under target/): run with
//!
//! ```text
//! cargo test -p seiza-cli --test hosted_integration -- --ignored
//! ```

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

const BASE: &str = "https://downloads.seiza.fyi";
/// (file, ra, dec, arcsec/px, blind). Mirrors testdata/solutions.json;
/// kept inline so expectations are versioned with the code. Images
/// needing a deeper star catalog than the hosted Tycho-2 lite are
/// hinted-only or listed with `blind: false`.
const CASES: &[(&str, f64, f64, f64, bool)] = &[
    ("m31-composite.jpg", 10.61678, 41.29111, 2.5818, true),
    ("wr134.jpg", 302.81594, 35.83233, 2.5796, true),
    ("sh2-101-tulip.jpg", 300.06815, 35.45363, 1.0288, true),
    ("ic5070-ha-61mp-mono.fit", 313.10989, 43.91495, 1.2585, true),
    (
        "m31-asi585mc-rggb-osc.fits",
        10.66661,
        41.26876,
        1.3581,
        false,
    ),
];

fn cache_dir() -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("seiza-testdata");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn fetch(path: &str, dest: &Path) {
    if dest.exists() {
        return;
    }
    let url = format!("{BASE}/{path}");
    let response = ureq::get(&url).call().unwrap_or_else(|e| {
        panic!("failed to fetch {url}: {e}");
    });
    let mut bytes = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut bytes)
        .unwrap_or_else(|e| panic!("failed to read {url}: {e}"));
    let partial = dest.with_extension("part");
    std::fs::write(&partial, &bytes).unwrap();
    std::fs::rename(&partial, dest).unwrap();
}

fn parse_solution(output: &str) -> Option<(f64, f64, f64)> {
    // "  center     : ... (10.61678°, 41.29114°)" and "  pixel scale: 2.5818\"/px"
    let center_line = output.lines().find(|l| l.contains("center"))?;
    let coords = center_line.rsplit_once('(')?.1.trim_end_matches(')');
    let (ra, dec) = coords.split_once(',')?;
    let ra: f64 = ra.trim().trim_end_matches('°').parse().ok()?;
    let dec: f64 = dec
        .trim()
        .trim_end_matches(')')
        .trim_end_matches('°')
        .parse()
        .ok()?;
    let scale_line = output.lines().find(|l| l.contains("pixel scale"))?;
    let scale: f64 = scale_line
        .split(':')
        .nth(1)?
        .trim()
        .split('"')
        .next()?
        .parse()
        .ok()?;
    Some((ra, dec, scale))
}

fn angular_separation_deg(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
    let (d1, d2) = (dec1.to_radians(), dec2.to_radians());
    let dra = (ra2 - ra1).to_radians();
    (d1.sin() * d2.sin() + d1.cos() * d2.cos() * dra.cos())
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

#[test]
#[ignore = "network: downloads real test images from downloads.seiza.fyi"]
fn hosted_images_solve_to_known_solutions() {
    let dir = cache_dir();
    let stars = dir.join("stars-lite-tycho2.bin");
    fetch("data/stars-lite-tycho2.bin", &stars);

    let seiza = env!("CARGO_BIN_EXE_seiza");
    let mut failures = Vec::new();

    for &(file, ra, dec, scale, blind) in CASES {
        let image = dir.join(file);
        fetch(&format!("testdata/{file}"), &image);

        let output = if blind {
            Command::new(seiza)
                .args(["solve-blind"])
                .arg(&image)
                .args(["--data"])
                .arg(&stars)
                .args(["--min-scale", "0.3", "--max-scale", "20"])
                .output()
        } else {
            Command::new(seiza)
                .args(["solve"])
                .arg(&image)
                .args(["--data"])
                .arg(&stars)
                .args([
                    "--ra",
                    &ra.to_string(),
                    "--dec",
                    &dec.to_string(),
                    "--radius",
                    "2",
                    "--scale",
                    &scale.to_string(),
                    "--scale-tolerance",
                    "0.2",
                ])
                .output()
        }
        .expect("failed to run seiza");

        let text = String::from_utf8_lossy(&output.stdout).to_string()
            + &String::from_utf8_lossy(&output.stderr);
        let Some((got_ra, got_dec, got_scale)) = parse_solution(&text) else {
            failures.push(format!("{file}: no solution\n{text}"));
            continue;
        };
        let separation = angular_separation_deg(ra, dec, got_ra, got_dec);
        let scale_error = (got_scale / scale - 1.0).abs();
        if separation > 0.02 || scale_error > 0.005 {
            failures.push(format!(
                "{file}: center off by {separation:.4} deg, scale error {:.2}%",
                scale_error * 100.0
            ));
        }
    }

    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
