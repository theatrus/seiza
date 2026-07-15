use seiza::catalog::TileSetBuilder;
use serde_json::Value;
use std::io::{Read, Write};
use std::process::{Command, Stdio};

fn test_catalog() -> (std::path::PathBuf, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!(
        "seiza-worker-protocol-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let catalog = dir.join("stars.bin");
    let mut builder = TileSetBuilder::new(2, 2025.5, "worker protocol test");
    builder.add(10.0, 20.0, 5.0);
    builder.write_to(&catalog).unwrap();
    (dir, catalog)
}

fn run_worker(catalog: &std::path::Path, requests: &[Value]) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args(["worker", "--data", catalog.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    {
        let stdin = child.stdin.as_mut().unwrap();
        for request in requests {
            serde_json::to_writer(&mut *stdin, request).unwrap();
            stdin.write_all(b"\n").unwrap();
        }
    }
    drop(child.stdin.take());
    child.wait_with_output().unwrap()
}

#[test]
fn worker_initializes_once_and_handles_multiple_requests() {
    let (dir, catalog) = test_catalog();
    let output = run_worker(
        &catalog,
        &[
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": 1,
                    "clientName": "integration-test",
                    "clientVersion": "1"
                }
            }),
            serde_json::json!({ "jsonrpc": "2.0", "id": 2, "method": "ping" }),
            serde_json::json!({ "jsonrpc": "2.0", "id": 3, "method": "ping" }),
            serde_json::json!({ "jsonrpc": "2.0", "id": 4, "method": "shutdown" }),
        ],
    );

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let responses: Vec<Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(responses.len(), 4);
    assert_eq!(responses[0]["result"]["protocolVersion"], 1);
    assert_eq!(responses[0]["result"]["catalog"]["starCount"], 1);
    assert_eq!(responses[1]["id"], 2);
    assert_eq!(responses[2]["id"], 3);
    assert_eq!(responses[3]["result"]["shutdown"], true);

    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn eof_cleanly_ends_a_one_shot_worker() {
    let (dir, catalog) = test_catalog();
    let output = run_worker(
        &catalog,
        &[
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": "init",
                "method": "initialize",
                "params": { "protocolVersion": 1 }
            }),
            serde_json::json!({ "jsonrpc": "2.0", "id": "ping", "method": "ping" }),
        ],
    );

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8(output.stdout).unwrap().lines().count(), 2);

    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn worker_adapts_json_rpc_to_the_seiza_server_native_api() {
    let dir = std::env::temp_dir().join(format!(
        "seiza-server-adapter-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let image_path = dir.join("field.png");
    image::GrayImage::from_pixel(64, 48, image::Luma([12]))
        .save(&image_path)
        .unwrap();

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let base_url = format!("http://127.0.0.1:{port}");
    let server = std::thread::spawn(move || {
        for step in 0..3 {
            let (mut stream, _) = listener.accept().unwrap();
            let request = read_http_request(&mut stream);
            assert!(
                request
                    .headers
                    .contains("authorization: bearer test-secret")
            );
            let response = match step {
                0 => {
                    assert_eq!(request.request_line, "GET /api/v1/health HTTP/1.1");
                    serde_json::json!({ "status": "ready", "solver_ready": true })
                }
                1 => {
                    assert_eq!(request.request_line, "POST /api/v1/solves HTTP/1.1");
                    assert!(
                        request
                            .headers
                            .contains("content-type: multipart/form-data; boundary=")
                    );
                    assert!(
                        request
                            .body
                            .windows(8)
                            .any(|window| window == b"\x89PNG\r\n\x1a\n")
                    );
                    let multipart = String::from_utf8_lossy(&request.body);
                    assert!(multipart.contains("name=\"options\""));
                    assert!(multipart.contains("\"center_ra_deg\":10.0"));
                    assert!(multipart.contains("filename=\"nina-solve.png\""));
                    serde_json::json!({
                        "id": "job-opaque-id",
                        "status": "queued",
                        "solution": null,
                        "error": null
                    })
                }
                2 => {
                    assert_eq!(
                        request.request_line,
                        "GET /api/v1/solves/job-opaque-id HTTP/1.1"
                    );
                    serde_json::json!({
                        "id": "job-opaque-id",
                        "status": "succeeded",
                        "solution": {
                            "center_ra_deg": 10.0,
                            "center_dec_deg": 20.0,
                            "pixel_scale_arcsec_per_pixel": 1.5,
                            "matched_stars": 24,
                            "rms_arcsec": 0.4,
                            "image_width": 64,
                            "image_height": 48,
                            "wcs": {
                                "crval": [10.0, 20.0],
                                "crpix": [31.5, 23.5],
                                "cd": [[-0.0004166667, 0.0], [0.0, 0.0004166667]]
                            }
                        },
                        "error": null
                    })
                }
                _ => unreachable!(),
            };
            write_json_response(&mut stream, &response);
        }
    });

    let mut child = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args([
            "worker",
            "--server",
            &base_url,
            "--server-token",
            "test-secret",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    {
        let stdin = child.stdin.as_mut().unwrap();
        for request in [
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": { "protocolVersion": 1 }
            }),
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "solve",
                "params": {
                    "imagePath": image_path,
                    "mode": "hinted",
                    "hint": {
                        "centerRaDeg": 10.0,
                        "centerDecDeg": 20.0,
                        "radiusDeg": 2.0,
                        "scaleArcsecPerPixel": 1.5
                    }
                }
            }),
            serde_json::json!({ "jsonrpc": "2.0", "id": 3, "method": "shutdown" }),
        ] {
            serde_json::to_writer(&mut *stdin, &request).unwrap();
            stdin.write_all(b"\n").unwrap();
        }
    }
    drop(child.stdin.take());
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let responses: Vec<Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(responses[0]["result"]["backend"], "remote");
    assert_eq!(
        responses[0]["result"]["capabilities"]["remoteImageEncoding"],
        "png-gray8"
    );
    assert_eq!(responses[1]["result"]["center"]["raDeg"], 10.0);
    assert_eq!(responses[1]["result"]["wcs"]["pixelOrigin"], 1);
    assert_eq!(responses[1]["result"]["wcs"]["crpix"][0], 32.5);
    assert_eq!(responses[1]["result"]["transfer"]["encoding"], "png-gray8");
    assert_eq!(responses[2]["result"]["shutdown"], true);

    server.join().unwrap();
    std::fs::remove_dir_all(dir).ok();
}

struct HttpRequest {
    request_line: String,
    headers: String,
    body: Vec<u8>,
}

fn read_http_request(stream: &mut std::net::TcpStream) -> HttpRequest {
    let mut bytes = Vec::new();
    let header_end = loop {
        let mut buffer = [0u8; 4096];
        let read = stream.read(&mut buffer).unwrap();
        assert!(read > 0, "connection closed before HTTP headers");
        bytes.extend_from_slice(&buffer[..read]);
        if let Some(offset) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            break offset + 4;
        }
    };
    let header_text = String::from_utf8(bytes[..header_end].to_vec()).unwrap();
    let content_length = header_text
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().unwrap())
        })
        .unwrap_or(0);
    while bytes.len() - header_end < content_length {
        let mut buffer = [0u8; 4096];
        let read = stream.read(&mut buffer).unwrap();
        assert!(read > 0, "connection closed before HTTP body");
        bytes.extend_from_slice(&buffer[..read]);
    }
    HttpRequest {
        request_line: header_text.lines().next().unwrap().to_string(),
        headers: header_text.to_ascii_lowercase(),
        body: bytes[header_end..header_end + content_length].to_vec(),
    }
}

fn write_json_response(stream: &mut std::net::TcpStream, value: &Value) {
    let body = serde_json::to_vec(value).unwrap();
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .unwrap();
    stream.write_all(&body).unwrap();
    stream.flush().unwrap();
}
