//! Integration suite against real camera images hosted at
//! downloads.seiza.fyi. Ignored by default (network + ~350 MB download,
//! cached under target/): run with
//!
//! ```text
//! cargo test -p seiza-cli --test hosted_integration -- --ignored
//! ```

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

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

#[test]
#[ignore = "network: downloads one FITS image and the hosted Tycho-2 lite catalog"]
fn hosted_worker_solves_fits_path_twice_in_one_process() {
    let dir = cache_dir();
    let stars = dir.join("stars-lite-tycho2.bin");
    let image = dir.join("m31-asi585mc-rggb-osc.fits");
    fetch("data/stars-lite-tycho2.bin", &stars);
    fetch("testdata/m31-asi585mc-rggb-osc.fits", &image);

    let mut child = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args(["worker", "--data"])
        .arg(&stars)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to start seiza worker");

    let solve_params = serde_json::json!({
        "imagePath": image,
        "mode": "hinted",
        "hint": {
            "centerRaDeg": 10.66661,
            "centerDecDeg": 41.26876,
            "radiusDeg": 2.0,
            "scaleArcsecPerPixel": 1.3581,
            "scaleTolerance": 0.2
        }
    });
    let requests = [
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": { "protocolVersion": 1, "clientName": "hosted-test" }
        }),
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "solve",
            "params": solve_params
        }),
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "solve",
            "params": solve_params
        }),
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "shutdown"
        }),
    ];

    {
        let stdin = child.stdin.as_mut().expect("worker stdin was not piped");
        for request in requests {
            serde_json::to_writer(&mut *stdin, &request).unwrap();
            stdin.write_all(b"\n").unwrap();
        }
    }
    drop(child.stdin.take());

    let output = child
        .wait_with_output()
        .expect("failed to wait for seiza worker");
    assert!(
        output.status.success(),
        "worker failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let responses: Vec<serde_json::Value> = String::from_utf8(output.stdout)
        .expect("worker output was not UTF-8")
        .lines()
        .map(|line| serde_json::from_str(line).expect("worker emitted invalid JSON"))
        .collect();
    assert_eq!(responses.len(), 4);

    let initialized = &responses[0]["result"];
    assert_eq!(initialized["protocolVersion"], 1);
    assert!(initialized["catalog"]["starCount"].as_u64().unwrap() > 0);

    for response in &responses[1..=2] {
        assert!(response.get("error").is_none(), "solve failed: {response}");
        let result = &response["result"];
        let ra = result["center"]["raDeg"].as_f64().unwrap();
        let dec = result["center"]["decDeg"].as_f64().unwrap();
        let scale = result["pixelScaleArcsecPerPixel"].as_f64().unwrap();
        assert!(angular_separation_deg(10.66661, 41.26876, ra, dec) < 0.02);
        assert!((scale / 1.3581 - 1.0).abs() < 0.005);
        assert_eq!(result["wcs"]["pixelOrigin"], 1);
    }

    assert_eq!(responses[3]["result"]["shutdown"], true);
}

#[test]
#[ignore = "network: requires SEIZA_SERVER_TEST_URL and downloads one FITS image"]
fn hosted_remote_worker_uses_seiza_server_native_api() {
    let dir = cache_dir();
    let image = dir.join("m31-asi585mc-rggb-osc.fits");
    fetch("testdata/m31-asi585mc-rggb-osc.fits", &image);
    let Ok(server_url) = std::env::var("SEIZA_SERVER_TEST_URL") else {
        eprintln!("skipping: set SEIZA_SERVER_TEST_URL to a configured seiza-server");
        return;
    };
    let mut command = Command::new(env!("CARGO_BIN_EXE_seiza"));
    command.args(["worker", "--server", &server_url]);
    if let Ok(token) = std::env::var("SEIZA_SERVER_TOKEN") {
        command.args(["--server-token", &token]);
    }
    let mut worker = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to start remote-backed worker");
    let requests = [
        serde_json::json!({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": { "protocolVersion": 1, "clientName": "hosted-remote-test" }
        }),
        serde_json::json!({
            "jsonrpc": "2.0", "id": 2, "method": "solve",
            "params": {
                "imagePath": image,
                "mode": "hinted",
                "hint": {
                    "centerRaDeg": 10.66661,
                    "centerDecDeg": 41.26876,
                    "radiusDeg": 2.0,
                    "scaleArcsecPerPixel": 1.3581,
                    "scaleTolerance": 0.2
                }
            }
        }),
        serde_json::json!({ "jsonrpc": "2.0", "id": 3, "method": "shutdown" }),
    ];
    {
        let stdin = worker.stdin.as_mut().unwrap();
        for request in requests {
            serde_json::to_writer(&mut *stdin, &request).unwrap();
            stdin.write_all(b"\n").unwrap();
        }
    }
    drop(worker.stdin.take());
    let output = worker.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let responses: Vec<serde_json::Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(responses[0]["result"]["backend"], "remote");
    let result = &responses[1]["result"];
    let ra = result["center"]["raDeg"].as_f64().unwrap();
    let dec = result["center"]["decDeg"].as_f64().unwrap();
    assert!(angular_separation_deg(10.66661, 41.26876, ra, dec) < 0.02);
    assert_eq!(result["transfer"]["encoding"], "png-gray8");
    eprintln!(
        "remote payload: {} bytes from {}-byte FITS",
        result["transfer"]["encodedBytes"].as_u64().unwrap(),
        std::fs::metadata(&image).unwrap().len()
    );
    assert!(
        result["transfer"]["encodedBytes"].as_u64().unwrap()
            < std::fs::metadata(&image).unwrap().len()
    );
}

#[test]
#[ignore = "network: downloads and validates the hosted stellar identifier sidecar"]
fn hosted_star_identifiers_are_downloadable_and_queryable() {
    let dir = cache_dir().join("catalog-bundle-v2");
    std::fs::create_dir_all(&dir).unwrap();
    let seiza = env!("CARGO_BIN_EXE_seiza");

    let download = Command::new(seiza)
        .args(["download-data", "prebuilt", "--output"])
        .arg(&dir)
        .args(["--file", "stars-lite-tycho2.ids.bin"])
        .output()
        .expect("failed to run seiza download-data prebuilt");
    assert!(
        download.status.success(),
        "hosted sidecar download failed:\n{}{}",
        String::from_utf8_lossy(&download.stdout),
        String::from_utf8_lossy(&download.stderr)
    );

    let identifiers = dir.join("stars-lite-tycho2.ids.bin");
    let validate = Command::new(seiza)
        .args(["catalog", "validate", "--data"])
        .arg(&identifiers)
        .output()
        .expect("failed to validate hosted stellar identifier sidecar");
    let validation_text = String::from_utf8_lossy(&validate.stdout).to_string()
        + &String::from_utf8_lossy(&validate.stderr);
    assert!(validate.status.success(), "{validation_text}");
    assert!(
        validation_text.contains("2698290 numeric identifiers, 387099 names"),
        "unexpected hosted sidecar contents:\n{validation_text}"
    );

    let query = Command::new(seiza)
        .args(["catalog", "star", "--data"])
        .arg(&identifiers)
        .args(["RR Lyr", "--format", "json"])
        .output()
        .expect("failed to query hosted stellar identifier sidecar");
    let query_text = String::from_utf8_lossy(&query.stdout).to_string()
        + &String::from_utf8_lossy(&query.stderr);
    assert!(query.status.success(), "{query_text}");
    assert!(
        query_text.contains("\"stable_id\": \"gcvs:RRLYR\""),
        "RR Lyr was not resolved through the hosted sidecar:\n{query_text}"
    );
}
