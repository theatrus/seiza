use seiza::catalog::TileSetBuilder;
use serde_json::Value;
use std::io::Write;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

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

struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        self.0.kill().ok();
        self.0.wait().ok();
    }
}

#[test]
fn worker_can_initialize_against_an_authenticated_remote_server() {
    let (dir, catalog) = test_catalog();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let base_url = format!("http://127.0.0.1:{port}");

    let server = Command::new(env!("CARGO_BIN_EXE_seiza-server"))
        .args(["--listen", &format!("127.0.0.1:{port}"), "--data"])
        .arg(&catalog)
        .args(["--token", "test-secret"])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let _server = ChildGuard(server);

    let started = Instant::now();
    loop {
        match ureq::get(&format!("{base_url}/v1/status"))
            .set("Authorization", "Bearer test-secret")
            .call()
        {
            Ok(_) => break,
            Err(_) if started.elapsed() < Duration::from_secs(10) => {
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(error) => panic!("seiza-server did not start: {error}"),
        }
    }

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
            serde_json::json!({ "jsonrpc": "2.0", "id": 2, "method": "shutdown" }),
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
        "gray8-zstd"
    );
    assert_eq!(responses[0]["result"]["catalog"]["starCount"], 1);
    assert_eq!(responses[1]["result"]["shutdown"], true);

    std::fs::remove_dir_all(dir).ok();
}
