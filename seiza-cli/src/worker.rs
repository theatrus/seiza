//! Persistent plate-solving worker over newline-delimited JSON-RPC 2.0.
//!
//! The worker keeps the star catalog and optional blind index open while it
//! reads requests from stdin. Each input line is one JSON-RPC request and
//! each output line is one response. Closing stdin cleanly ends the worker,
//! so the same protocol supports both long-lived and one-shot clients.

use anyhow::{Context, Result};
use seiza::blind::{BlindIndex, BlindParams, solve_blind};
use seiza::catalog::TileCatalog;
use seiza::solve::{Solution, SolveHint, solve};
use seiza::{DetectConfig, Wcs, detect_stars};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::{self, BufRead, BufReader, BufWriter, Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tiny_http::{Header, Method, Response, Server, StatusCode};

const JSON_RPC_VERSION: &str = "2.0";
const PROTOCOL_VERSION: u32 = 1;
const MAX_REQUEST_BYTES: usize = 1024 * 1024;
const MAX_REMOTE_REQUEST_BYTES: u64 = 128 * 1024 * 1024;
const REMOTE_SCHEMA_VERSION: u32 = 1;
const REMOTE_MAGIC: &[u8; 8] = b"SEIZA\0I1";
const REMOTE_CONTENT_TYPE: &str = "application/vnd.seiza.solve-image+zstd";

const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;
const NOT_INITIALIZED: i64 = -32002;
const SOLVE_FAILED: i64 = -32010;

/// Run a local or remote-backed worker on the process standard streams.
pub fn run(
    data_path: Option<&Path>,
    index_path: Option<&Path>,
    server: Option<&str>,
    server_token: Option<&str>,
) -> Result<()> {
    let service = match (data_path, server) {
        (Some(data_path), None) => WorkerService::local(data_path, index_path)?,
        (None, Some(server)) => WorkerService::remote(server, server_token)?,
        _ => {
            anyhow::bail!("provide either --data for local solving or --server for remote solving")
        }
    };
    let stdin = io::stdin();
    let stdout = io::stdout();
    run_loop(
        BufReader::new(stdin.lock()),
        BufWriter::new(stdout.lock()),
        service,
    )
}

#[derive(Debug, Deserialize)]
struct RpcRequest {
    jsonrpc: String,
    #[serde(default)]
    id: Value,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Serialize)]
struct RpcResponse {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<RpcError>,
}

impl RpcResponse {
    fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: JSON_RPC_VERSION,
            id,
            result: Some(result),
            error: None,
        }
    }

    fn error(id: Value, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: JSON_RPC_VERSION,
            id,
            result: None,
            error: Some(RpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }

    fn error_with_data(id: Value, code: i64, message: impl Into<String>, data: Value) -> Self {
        Self {
            jsonrpc: JSON_RPC_VERSION,
            id,
            result: None,
            error: Some(RpcError {
                code,
                message: message.into(),
                data: Some(data),
            }),
        }
    }
}

#[derive(Debug, Serialize)]
struct RpcError {
    code: i64,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

struct HandledResponse {
    response: RpcResponse,
    shutdown: bool,
}

trait RpcService {
    fn handle(&mut self, request: RpcRequest) -> HandledResponse;
}

fn run_loop<R, W, S>(mut reader: R, mut writer: W, mut service: S) -> Result<()>
where
    R: BufRead,
    W: Write,
    S: RpcService,
{
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }

        let handled = if line.len() > MAX_REQUEST_BYTES {
            HandledResponse {
                response: RpcResponse::error(
                    Value::Null,
                    INVALID_REQUEST,
                    format!("request exceeds {MAX_REQUEST_BYTES} bytes"),
                ),
                shutdown: false,
            }
        } else {
            match serde_json::from_str::<RpcRequest>(&line) {
                Ok(request) => service.handle(request),
                Err(error) => HandledResponse {
                    response: RpcResponse::error(
                        Value::Null,
                        PARSE_ERROR,
                        format!("invalid JSON: {error}"),
                    ),
                    shutdown: false,
                },
            }
        };

        serde_json::to_writer(&mut writer, &handled.response)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        if handled.shutdown {
            break;
        }
    }
    Ok(())
}

struct LocalSolver {
    catalog: TileCatalog,
    index: Option<BlindIndex>,
}

enum WorkerBackend {
    Local(Box<LocalSolver>),
    Remote(RemoteSolver),
}

struct WorkerService {
    backend: WorkerBackend,
    initialized: bool,
}

impl WorkerService {
    fn local(data_path: &Path, index_path: Option<&Path>) -> Result<Self> {
        Ok(Self {
            backend: WorkerBackend::Local(Box::new(LocalSolver::open(data_path, index_path)?)),
            initialized: false,
        })
    }

    fn remote(server: &str, token: Option<&str>) -> Result<Self> {
        Ok(Self {
            backend: WorkerBackend::Remote(RemoteSolver::new(server, token)?),
            initialized: false,
        })
    }

    fn initialize(&mut self, id: Value, params: Value) -> RpcResponse {
        let params: InitializeParams = match serde_json::from_value(params) {
            Ok(params) => params,
            Err(error) => {
                return RpcResponse::error(
                    id,
                    INVALID_PARAMS,
                    format!("invalid initialize parameters: {error}"),
                );
            }
        };
        if params.protocol_version != PROTOCOL_VERSION {
            return RpcResponse::error_with_data(
                id,
                INVALID_PARAMS,
                format!(
                    "unsupported protocol version {}; expected {PROTOCOL_VERSION}",
                    params.protocol_version
                ),
                json!({ "supportedProtocolVersions": [PROTOCOL_VERSION] }),
            );
        }

        let backend = match &mut self.backend {
            WorkerBackend::Local(local) => local.status(),
            WorkerBackend::Remote(remote) => match remote.status() {
                Ok(status) => status,
                Err(error) => {
                    return RpcResponse::error(
                        id,
                        SOLVE_FAILED,
                        format!("failed to initialize remote solver: {error:#}"),
                    );
                }
            },
        };
        self.initialized = true;
        RpcResponse::success(
            id,
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "server": backend.server,
                "client": {
                    "name": params.client_name,
                    "version": params.client_version
                },
                "capabilities": {
                    "solveModes": ["hinted", "blind"],
                    "imageInputs": ["fits_path"],
                    "remoteImageEncoding": backend.remote_image_encoding,
                    "maxConcurrentRequests": 1
                },
                "catalog": backend.catalog,
                "backend": backend.kind
            }),
        )
    }

    fn solve(&mut self, id: Value, params: Value) -> RpcResponse {
        if !self.initialized {
            return RpcResponse::error(
                id,
                NOT_INITIALIZED,
                "worker must be initialized before solving",
            );
        }
        let params: SolveParams = match serde_json::from_value(params) {
            Ok(params) => params,
            Err(error) => {
                return RpcResponse::error(
                    id,
                    INVALID_PARAMS,
                    format!("invalid solve parameters: {error}"),
                );
            }
        };
        if let Err(error) = params.validate() {
            return RpcResponse::error(id, INVALID_PARAMS, error);
        }

        let result = match &mut self.backend {
            WorkerBackend::Local(local) => local.solve_path(&params),
            WorkerBackend::Remote(remote) => remote.solve_path(&params),
        };
        match result {
            Ok(result) => match serde_json::to_value(result) {
                Ok(result) => RpcResponse::success(id, result),
                Err(error) => RpcResponse::error(
                    id,
                    SOLVE_FAILED,
                    format!("failed to serialize solution: {error}"),
                ),
            },
            Err(error) => RpcResponse::error(id, SOLVE_FAILED, format!("{error:#}")),
        }
    }
}

impl LocalSolver {
    fn open(data_path: &Path, index_path: Option<&Path>) -> Result<Self> {
        let catalog = TileCatalog::open(data_path)
            .with_context(|| format!("failed to open star catalog {}", data_path.display()))?;
        let index = index_path
            .map(|path| {
                BlindIndex::open(path)
                    .map_err(anyhow::Error::from)
                    .with_context(|| format!("failed to open blind index {}", path.display()))
            })
            .transpose()?;

        if let Some(index) = &index {
            warn_on_index_catalog_mismatch(index, &catalog);
        }

        Ok(Self { catalog, index })
    }

    fn status(&self) -> BackendStatus {
        BackendStatus {
            kind: "local".to_string(),
            server: ServerIdentity {
                name: "seiza".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            catalog: CatalogStatus {
                star_count: self.catalog.star_count(),
                blind_index_loaded: self.index.is_some(),
                blind_index_pattern_count: self.index.as_ref().map(BlindIndex::pattern_count),
                blind_index_magnitude_limit: self.index.as_ref().map(BlindIndex::index_mag_limit),
            },
            remote_image_encoding: None,
        }
    }

    fn solve_path(&mut self, params: &SolveParams) -> Result<WorkerSolveResult> {
        let total_started = Instant::now();

        let load_started = Instant::now();
        let image = crate::load_image(&params.image_path)?;
        let load_elapsed = load_started.elapsed();
        self.solve_image(&image, params, load_elapsed, Duration::ZERO, total_started)
    }

    fn solve_image(
        &mut self,
        image: &image::DynamicImage,
        params: &SolveParams,
        load_elapsed: Duration,
        transport_elapsed: Duration,
        total_started: Instant,
    ) -> Result<WorkerSolveResult> {
        let dimensions = (image.width(), image.height());

        let detect_started = Instant::now();
        let stars = detect_stars(
            image,
            &DetectConfig {
                sigma: params.detection.sigma,
                ignore_border: params.detection.ignore_border,
                max_stars: params.detection.max_stars,
                ..Default::default()
            },
        );
        let detect_elapsed = detect_started.elapsed();

        let mut index_elapsed = Duration::ZERO;
        let solve_started = Instant::now();
        let solution = match params.mode {
            SolveMode::Hinted => {
                let hint = params.hint.as_ref().expect("validated hinted parameters");
                solve(
                    &stars,
                    &self.catalog,
                    &SolveHint {
                        center: (hint.center_ra_deg, hint.center_dec_deg),
                        radius_deg: hint.radius_deg,
                        scale_arcsec_px: hint.scale_arcsec_per_pixel,
                        scale_tolerance: hint.scale_tolerance,
                    },
                    dimensions,
                )
                .map_err(anyhow::Error::from)?
            }
            SolveMode::Blind => {
                let blind = params.blind.clone().unwrap_or_default();
                let mut blind_params = BlindParams {
                    min_scale_arcsec_px: blind.min_scale_arcsec_per_pixel,
                    max_scale_arcsec_px: blind.max_scale_arcsec_per_pixel,
                    index_mag_limit: blind.index_magnitude_limit,
                    max_hypotheses: blind.max_hypotheses,
                    max_coarse_hypotheses: blind.max_coarse_hypotheses,
                    ..Default::default()
                };

                let index_started = Instant::now();
                if self.index.is_none() {
                    self.index = Some(BlindIndex::build(&self.catalog, &blind_params));
                }
                index_elapsed = index_started.elapsed();
                let index = self.index.as_ref().expect("blind index initialized");
                blind_params.index_mag_limit = index.index_mag_limit();
                blind_params.max_pattern_deg = index.max_pattern_deg();

                solve_blind(&stars, &self.catalog, index, &blind_params, dimensions)
                    .map_err(anyhow::Error::from)?
            }
        };
        let solve_elapsed = solve_started.elapsed().saturating_sub(index_elapsed);

        Ok(solution_result(
            params.mode,
            &solution,
            dimensions,
            SolveTimings {
                load_ms: milliseconds(load_elapsed),
                encode_ms: 0.0,
                transport_ms: milliseconds(transport_elapsed),
                detect_ms: milliseconds(detect_elapsed),
                index_ms: milliseconds(index_elapsed),
                solve_ms: milliseconds(solve_elapsed),
                total_ms: milliseconds(total_started.elapsed()),
            },
        ))
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct BackendStatus {
    kind: String,
    server: ServerIdentity,
    catalog: CatalogStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    remote_image_encoding: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ServerIdentity {
    name: String,
    version: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct CatalogStatus {
    star_count: u64,
    blind_index_loaded: bool,
    blind_index_pattern_count: Option<usize>,
    blind_index_magnitude_limit: Option<f32>,
}

struct RemoteSolver {
    base_url: String,
    token: Option<String>,
    agent: ureq::Agent,
}

impl RemoteSolver {
    fn new(base_url: &str, token: Option<&str>) -> Result<Self> {
        let base_url = base_url.trim_end_matches('/');
        if !(base_url.starts_with("http://") || base_url.starts_with("https://")) {
            anyhow::bail!("remote server URL must start with http:// or https://");
        }
        Ok(Self {
            base_url: base_url.to_string(),
            token: token.map(str::to_string),
            agent: ureq::AgentBuilder::new().build(),
        })
    }

    fn status(&self) -> Result<BackendStatus> {
        let response = self.call(self.agent.get(&format!("{}/v1/status", self.base_url)))?;
        response
            .into_json()
            .context("invalid status response from seiza-server")
    }

    fn solve_path(&self, params: &SolveParams) -> Result<WorkerSolveResult> {
        let total_started = Instant::now();
        let load_started = Instant::now();
        let image = crate::load_image(&params.image_path)?.into_luma8();
        let load_elapsed = load_started.elapsed();

        let encode_started = Instant::now();
        let body = encode_remote_request(&image, params)?;
        let encode_elapsed = encode_started.elapsed();

        let transport_started = Instant::now();
        let request = self
            .agent
            .post(&format!("{}/v1/solve", self.base_url))
            .set("Content-Type", REMOTE_CONTENT_TYPE);
        let response = self.call_with_body(request, &body)?;
        let mut result: WorkerSolveResult = response
            .into_json()
            .context("invalid solve response from seiza-server")?;
        let transport_elapsed = transport_started.elapsed();

        result.timings.load_ms = milliseconds(load_elapsed);
        result.timings.encode_ms = milliseconds(encode_elapsed);
        result.timings.transport_ms = milliseconds(transport_elapsed);
        result.timings.total_ms = milliseconds(total_started.elapsed());
        result.transfer = Some(RemoteTransfer {
            encoding: "gray8-zstd".to_string(),
            uncompressed_bytes: image.as_raw().len(),
            encoded_bytes: body.len(),
        });
        Ok(result)
    }

    fn call(&self, mut request: ureq::Request) -> Result<ureq::Response> {
        if let Some(token) = &self.token {
            request = request.set("Authorization", &format!("Bearer {token}"));
        }
        map_remote_response(request.call())
    }

    fn call_with_body(&self, mut request: ureq::Request, body: &[u8]) -> Result<ureq::Response> {
        if let Some(token) = &self.token {
            request = request.set("Authorization", &format!("Bearer {token}"));
        }
        map_remote_response(request.send_bytes(body))
    }
}

fn map_remote_response(
    response: std::result::Result<ureq::Response, ureq::Error>,
) -> Result<ureq::Response> {
    match response {
        Ok(response) => Ok(response),
        Err(ureq::Error::Status(code, response)) => {
            let message = response.into_string().unwrap_or_default();
            anyhow::bail!("seiza-server returned HTTP {code}: {message}")
        }
        Err(error) => Err(anyhow::Error::new(error).context("seiza-server request failed")),
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct RemoteSolveRequest {
    schema_version: u32,
    mode: SolveMode,
    hint: Option<HintParams>,
    blind: Option<BlindSolveParams>,
    detection: DetectionParams,
    image: RemoteImageMetadata,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct RemoteImageMetadata {
    width: u32,
    height: u32,
    encoding: String,
    uncompressed_bytes: usize,
}

fn encode_remote_request(image: &image::GrayImage, params: &SolveParams) -> Result<Vec<u8>> {
    let metadata = RemoteSolveRequest {
        schema_version: REMOTE_SCHEMA_VERSION,
        mode: params.mode,
        hint: params.hint.clone(),
        blind: params.blind.clone(),
        detection: params.detection.clone(),
        image: RemoteImageMetadata {
            width: image.width(),
            height: image.height(),
            encoding: "gray8-zstd".to_string(),
            uncompressed_bytes: image.as_raw().len(),
        },
    };
    let metadata = serde_json::to_vec(&metadata)?;
    let metadata_len = u32::try_from(metadata.len()).context("remote metadata is too large")?;
    let compressed = zstd::stream::encode_all(Cursor::new(image.as_raw()), 3)
        .context("failed to compress remote image")?;

    let mut body = Vec::with_capacity(REMOTE_MAGIC.len() + 4 + metadata.len() + compressed.len());
    body.extend_from_slice(REMOTE_MAGIC);
    body.extend_from_slice(&metadata_len.to_le_bytes());
    body.extend_from_slice(&metadata);
    body.extend_from_slice(&compressed);
    Ok(body)
}

fn decode_remote_request(body: &[u8]) -> Result<(RemoteSolveRequest, image::GrayImage)> {
    if body.len() < REMOTE_MAGIC.len() + 4 || &body[..REMOTE_MAGIC.len()] != REMOTE_MAGIC {
        anyhow::bail!("invalid Seiza remote image envelope");
    }
    let metadata_offset = REMOTE_MAGIC.len() + 4;
    let metadata_len = u32::from_le_bytes(
        body[REMOTE_MAGIC.len()..metadata_offset]
            .try_into()
            .expect("four-byte metadata length"),
    ) as usize;
    let image_offset = metadata_offset
        .checked_add(metadata_len)
        .filter(|offset| *offset <= body.len())
        .context("remote metadata length exceeds request body")?;
    let metadata: RemoteSolveRequest = serde_json::from_slice(&body[metadata_offset..image_offset])
        .context("invalid remote solve metadata")?;
    if metadata.schema_version != REMOTE_SCHEMA_VERSION {
        anyhow::bail!(
            "unsupported remote schema version {}; expected {REMOTE_SCHEMA_VERSION}",
            metadata.schema_version
        );
    }
    if metadata.image.encoding != "gray8-zstd" {
        anyhow::bail!(
            "unsupported remote image encoding {}",
            metadata.image.encoding
        );
    }
    let expected = (metadata.image.width as usize)
        .checked_mul(metadata.image.height as usize)
        .context("remote image dimensions overflow")?;
    if expected == 0 || expected > MAX_REMOTE_REQUEST_BYTES as usize {
        anyhow::bail!("remote image dimensions are empty or too large");
    }
    if metadata.image.uncompressed_bytes != expected {
        anyhow::bail!("remote image byte count does not match its dimensions");
    }
    let mut pixels = Vec::with_capacity(expected);
    zstd::stream::read::Decoder::new(&body[image_offset..])
        .context("invalid Zstandard image stream")?
        .take(expected as u64 + 1)
        .read_to_end(&mut pixels)
        .context("failed to decompress remote image")?;
    if pixels.len() != expected {
        anyhow::bail!("decompressed image size does not match its dimensions");
    }
    let image = image::GrayImage::from_raw(metadata.image.width, metadata.image.height, pixels)
        .context("failed to construct remote image")?;
    Ok((metadata, image))
}

/// Run the warm HTTP service used by remote-backed workers.
pub fn run_server(
    listen: &str,
    data_path: &Path,
    index_path: Option<&Path>,
    token: Option<&str>,
) -> Result<()> {
    let mut solver = LocalSolver::open(data_path, index_path)?;
    let server = Server::http(listen)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .with_context(|| format!("failed to listen on {listen}"))?;
    eprintln!("seiza-server listening on http://{listen}");

    for mut request in server.incoming_requests() {
        if !authorized(&request, token) {
            request.respond(text_response(StatusCode(401), "unauthorized"))?;
            continue;
        }
        match (request.method(), request.url()) {
            (&Method::Get, "/v1/status") => {
                let mut status = solver.status();
                status.kind = "remote".to_string();
                status.server.name = "seiza-server".to_string();
                status.remote_image_encoding = Some("gray8-zstd".to_string());
                request.respond(json_response(StatusCode(200), &status)?)?;
            }
            (&Method::Post, "/v1/solve") => {
                let is_expected_type = request.headers().iter().any(|header| {
                    header.field.equiv("Content-Type")
                        && header.value.as_str().starts_with(REMOTE_CONTENT_TYPE)
                });
                if !is_expected_type {
                    request.respond(text_response(StatusCode(415), "unsupported content type"))?;
                    continue;
                }
                let mut body = Vec::new();
                request
                    .as_reader()
                    .take(MAX_REMOTE_REQUEST_BYTES + 1)
                    .read_to_end(&mut body)?;
                if body.len() as u64 > MAX_REMOTE_REQUEST_BYTES {
                    request.respond(text_response(StatusCode(413), "request body too large"))?;
                    continue;
                }
                let response = (|| -> Result<WorkerSolveResult> {
                    let (metadata, image) = decode_remote_request(&body)?;
                    let params = SolveParams {
                        image_path: PathBuf::from("<remote>"),
                        mode: metadata.mode,
                        hint: metadata.hint,
                        blind: metadata.blind,
                        detection: metadata.detection,
                    };
                    params.validate().map_err(anyhow::Error::msg)?;
                    solver.solve_image(
                        &image::DynamicImage::ImageLuma8(image),
                        &params,
                        Duration::ZERO,
                        Duration::ZERO,
                        Instant::now(),
                    )
                })();
                match response {
                    Ok(result) => request.respond(json_response(StatusCode(200), &result)?)?,
                    Err(error) => {
                        request.respond(text_response(StatusCode(422), &format!("{error:#}")))?
                    }
                }
            }
            _ => request.respond(text_response(StatusCode(404), "not found"))?,
        }
    }
    Ok(())
}

fn authorized(request: &tiny_http::Request, token: Option<&str>) -> bool {
    let Some(token) = token else { return true };
    request.headers().iter().any(|header| {
        header.field.equiv("Authorization") && header.value.as_str() == format!("Bearer {token}")
    })
}

fn json_response<T: Serialize>(status: StatusCode, value: &T) -> Result<Response<Cursor<Vec<u8>>>> {
    let body = serde_json::to_vec(value)?;
    let content_type = Header::from_bytes("Content-Type", "application/json")
        .map_err(|_| anyhow::anyhow!("invalid JSON content-type header"))?;
    Ok(Response::from_data(body)
        .with_status_code(status)
        .with_header(content_type))
}

fn text_response(status: StatusCode, value: &str) -> Response<Cursor<Vec<u8>>> {
    Response::from_string(value).with_status_code(status)
}

impl RpcService for WorkerService {
    fn handle(&mut self, request: RpcRequest) -> HandledResponse {
        let id = request.id;
        if request.jsonrpc != JSON_RPC_VERSION {
            return HandledResponse {
                response: RpcResponse::error(
                    id,
                    INVALID_REQUEST,
                    format!("jsonrpc must be {JSON_RPC_VERSION}"),
                ),
                shutdown: false,
            };
        }

        match request.method.as_str() {
            "initialize" => HandledResponse {
                response: self.initialize(id, request.params),
                shutdown: false,
            },
            "solve" => HandledResponse {
                response: self.solve(id, request.params),
                shutdown: false,
            },
            "ping" => HandledResponse {
                response: RpcResponse::success(id, json!({ "protocolVersion": PROTOCOL_VERSION })),
                shutdown: false,
            },
            "shutdown" => HandledResponse {
                response: RpcResponse::success(id, json!({ "shutdown": true })),
                shutdown: true,
            },
            method => HandledResponse {
                response: RpcResponse::error(
                    id,
                    METHOD_NOT_FOUND,
                    format!("unknown method {method}"),
                ),
                shutdown: false,
            },
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InitializeParams {
    protocol_version: u32,
    #[serde(default)]
    client_name: Option<String>,
    #[serde(default)]
    client_version: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SolveParams {
    image_path: PathBuf,
    mode: SolveMode,
    #[serde(default)]
    hint: Option<HintParams>,
    #[serde(default)]
    blind: Option<BlindSolveParams>,
    #[serde(default)]
    detection: DetectionParams,
}

impl SolveParams {
    fn validate(&self) -> std::result::Result<(), String> {
        if self.image_path.as_os_str().is_empty() {
            return Err("imagePath must not be empty".to_string());
        }
        self.detection.validate()?;
        match self.mode {
            SolveMode::Hinted => self
                .hint
                .as_ref()
                .ok_or_else(|| "hint is required for hinted solves".to_string())?
                .validate(),
            SolveMode::Blind => self.blind.clone().unwrap_or_default().validate(),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum SolveMode {
    Hinted,
    Blind,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct HintParams {
    center_ra_deg: f64,
    center_dec_deg: f64,
    radius_deg: f64,
    scale_arcsec_per_pixel: f64,
    #[serde(default = "default_scale_tolerance")]
    scale_tolerance: f64,
}

impl HintParams {
    fn validate(&self) -> std::result::Result<(), String> {
        finite("hint.centerRaDeg", self.center_ra_deg)?;
        finite("hint.centerDecDeg", self.center_dec_deg)?;
        finite("hint.radiusDeg", self.radius_deg)?;
        finite("hint.scaleArcsecPerPixel", self.scale_arcsec_per_pixel)?;
        finite("hint.scaleTolerance", self.scale_tolerance)?;
        if !(-90.0..=90.0).contains(&self.center_dec_deg) {
            return Err("hint.centerDecDeg must be between -90 and 90".to_string());
        }
        if self.radius_deg <= 0.0 || self.radius_deg > 180.0 {
            return Err("hint.radiusDeg must be greater than 0 and at most 180".to_string());
        }
        if self.scale_arcsec_per_pixel <= 0.0 {
            return Err("hint.scaleArcsecPerPixel must be greater than 0".to_string());
        }
        if self.scale_tolerance <= 0.0 {
            return Err("hint.scaleTolerance must be greater than 0".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase")]
struct BlindSolveParams {
    min_scale_arcsec_per_pixel: f64,
    max_scale_arcsec_per_pixel: f64,
    index_magnitude_limit: f32,
    max_hypotheses: usize,
    max_coarse_hypotheses: usize,
}

impl Default for BlindSolveParams {
    fn default() -> Self {
        Self {
            min_scale_arcsec_per_pixel: 0.1,
            max_scale_arcsec_per_pixel: 20.0,
            index_magnitude_limit: 12.7,
            max_hypotheses: 400,
            max_coarse_hypotheses: 20_000,
        }
    }
}

impl BlindSolveParams {
    fn validate(&self) -> std::result::Result<(), String> {
        finite(
            "blind.minScaleArcsecPerPixel",
            self.min_scale_arcsec_per_pixel,
        )?;
        finite(
            "blind.maxScaleArcsecPerPixel",
            self.max_scale_arcsec_per_pixel,
        )?;
        if self.min_scale_arcsec_per_pixel <= 0.0
            || self.max_scale_arcsec_per_pixel <= self.min_scale_arcsec_per_pixel
        {
            return Err(
                "blind scale range must be positive and maxScaleArcsecPerPixel must exceed minScaleArcsecPerPixel"
                    .to_string(),
            );
        }
        if self.max_hypotheses == 0 {
            return Err("blind.maxHypotheses must be greater than 0".to_string());
        }
        if self.max_coarse_hypotheses < self.max_hypotheses {
            return Err("blind.maxCoarseHypotheses must be at least maxHypotheses".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase")]
struct DetectionParams {
    sigma: f32,
    ignore_border: u32,
    max_stars: usize,
}

impl Default for DetectionParams {
    fn default() -> Self {
        Self {
            sigma: 4.0,
            ignore_border: 0,
            max_stars: 500,
        }
    }
}

impl DetectionParams {
    fn validate(&self) -> std::result::Result<(), String> {
        if !self.sigma.is_finite() || self.sigma <= 0.0 {
            return Err("detection.sigma must be finite and greater than 0".to_string());
        }
        if self.max_stars < 4 || self.max_stars > 10_000 {
            return Err("detection.maxStars must be between 4 and 10000".to_string());
        }
        Ok(())
    }
}

fn default_scale_tolerance() -> f64 {
    0.2
}

fn finite(name: &str, value: f64) -> std::result::Result<(), String> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(format!("{name} must be finite"))
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct WorkerSolveResult {
    schema_version: u32,
    mode: SolveMode,
    image: SolvedImage,
    center: SkyCoordinate,
    pixel_scale_arcsec_per_pixel: f64,
    rotation_deg: f64,
    parity: String,
    radius_deg: f64,
    matched_stars: usize,
    rms_arcsec: f64,
    wcs: WorkerWcs,
    footprint: Vec<SkyCoordinate>,
    timings: SolveTimings,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    transfer: Option<RemoteTransfer>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct RemoteTransfer {
    encoding: String,
    uncompressed_bytes: usize,
    encoded_bytes: usize,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct SolvedImage {
    width: u32,
    height: u32,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct SkyCoordinate {
    ra_deg: f64,
    dec_deg: f64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct WorkerWcs {
    projection: String,
    pixel_origin: u8,
    crval: [f64; 2],
    crpix: [f64; 2],
    cd: [[f64; 2]; 2],
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct SolveTimings {
    load_ms: f64,
    encode_ms: f64,
    transport_ms: f64,
    detect_ms: f64,
    index_ms: f64,
    solve_ms: f64,
    total_ms: f64,
}

fn solution_result(
    mode: SolveMode,
    solution: &Solution,
    dimensions: (u32, u32),
    timings: SolveTimings,
) -> WorkerSolveResult {
    let wcs = &solution.wcs;
    let center = wcs.pixel_to_world(dimensions.0 as f64 / 2.0, dimensions.1 as f64 / 2.0);
    let determinant = wcs.cd[0][0] * wcs.cd[1][1] - wcs.cd[0][1] * wcs.cd[1][0];
    let rotation_deg = north_rotation_deg(wcs);
    let scale = wcs.scale_arcsec_per_px();
    let radius_deg =
        ((dimensions.0 as f64 * scale).hypot(dimensions.1 as f64 * scale) / 2.0) / 3600.0;
    let footprint = wcs
        .footprint(dimensions.0, dimensions.1)
        .into_iter()
        .map(|(ra_deg, dec_deg)| SkyCoordinate { ra_deg, dec_deg })
        .collect();

    WorkerSolveResult {
        schema_version: PROTOCOL_VERSION,
        mode,
        image: SolvedImage {
            width: dimensions.0,
            height: dimensions.1,
        },
        center: SkyCoordinate {
            ra_deg: center.0,
            dec_deg: center.1,
        },
        pixel_scale_arcsec_per_pixel: scale,
        rotation_deg,
        parity: if determinant > 0.0 {
            "normal".to_string()
        } else {
            "mirrored".to_string()
        },
        radius_deg,
        matched_stars: solution.matched_stars,
        rms_arcsec: solution.rms_arcsec,
        wcs: WorkerWcs {
            projection: "TAN".to_string(),
            pixel_origin: 1,
            crval: [wcs.crval.0, wcs.crval.1],
            // The library uses zero-based pixel centers. The wire contract
            // uses FITS-standard one-based CRPIX values.
            crpix: [wcs.crpix.0 + 1.0, wcs.crpix.1 + 1.0],
            cd: wcs.cd,
        },
        footprint,
        timings,
        transfer: None,
    }
}

fn north_rotation_deg(wcs: &Wcs) -> f64 {
    wcs.world_to_pixel(wcs.crval.0, wcs.crval.1 + 0.01)
        .map(|(x, y)| {
            (x - wcs.crpix.0)
                .atan2(-(y - wcs.crpix.1))
                .to_degrees()
                .rem_euclid(360.0)
        })
        .unwrap_or(f64::NAN)
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn warn_on_index_catalog_mismatch(index: &BlindIndex, catalog: &TileCatalog) {
    let built_from = index.source_star_count();
    let runtime = catalog.star_count();
    if built_from > 0 && runtime > 0 && built_from.max(runtime) > 2 * built_from.min(runtime) {
        eprintln!(
            "warning: blind index was built from a {built_from}-star catalog but solving \
             against {runtime} stars; deep-tier hypotheses may never verify"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    struct TestService {
        calls: usize,
    }

    impl RpcService for TestService {
        fn handle(&mut self, request: RpcRequest) -> HandledResponse {
            self.calls += 1;
            let shutdown = request.method == "shutdown";
            HandledResponse {
                response: RpcResponse::success(
                    request.id,
                    json!({ "method": request.method, "call": self.calls }),
                ),
                shutdown,
            }
        }
    }

    #[test]
    fn loop_handles_multiple_requests_until_eof() {
        let input = concat!(
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}\n",
            "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}\n"
        );
        let mut output = Vec::new();
        run_loop(
            Cursor::new(input.as_bytes()),
            &mut output,
            TestService { calls: 0 },
        )
        .unwrap();

        let responses: Vec<Value> = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0]["id"], 1);
        assert_eq!(responses[0]["result"]["call"], 1);
        assert_eq!(responses[1]["id"], 2);
        assert_eq!(responses[1]["result"]["call"], 2);
    }

    #[test]
    fn one_request_then_eof_is_a_valid_one_shot_session() {
        let input = "{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"ping\"}\n";
        let mut output = Vec::new();
        run_loop(
            Cursor::new(input.as_bytes()),
            &mut output,
            TestService { calls: 0 },
        )
        .unwrap();
        let response: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(response["id"], 7);
        assert_eq!(response["result"]["call"], 1);
    }

    #[test]
    fn malformed_json_returns_parse_error_and_keeps_reading() {
        let input = concat!(
            "not json\n",
            "{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"ping\"}\n"
        );
        let mut output = Vec::new();
        run_loop(
            Cursor::new(input.as_bytes()),
            &mut output,
            TestService { calls: 0 },
        )
        .unwrap();
        let responses: Vec<Value> = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(responses[0]["error"]["code"], PARSE_ERROR);
        assert_eq!(responses[1]["id"], 9);
    }

    #[test]
    fn shutdown_stops_before_later_requests() {
        let input = concat!(
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"shutdown\"}\n",
            "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}\n"
        );
        let mut output = Vec::new();
        run_loop(
            Cursor::new(input.as_bytes()),
            &mut output,
            TestService { calls: 0 },
        )
        .unwrap();
        assert_eq!(String::from_utf8(output).unwrap().lines().count(), 1);
    }

    #[test]
    fn solve_validation_requires_a_hint_for_hinted_mode() {
        let params: SolveParams = serde_json::from_value(json!({
            "imagePath": "image.fits",
            "mode": "hinted"
        }))
        .unwrap();
        assert_eq!(
            params.validate().unwrap_err(),
            "hint is required for hinted solves"
        );
    }

    #[test]
    fn result_uses_fits_one_based_crpix() {
        let solution = Solution {
            wcs: Wcs {
                crval: (123.0, -20.0),
                crpix: (1999.0, 1499.0),
                cd: [[-0.001, 0.0], [0.0, 0.001]],
            },
            matched_stars: 42,
            rms_arcsec: 0.5,
        };
        let result = solution_result(
            SolveMode::Hinted,
            &solution,
            (4000, 3000),
            SolveTimings {
                load_ms: 1.0,
                encode_ms: 0.0,
                transport_ms: 0.0,
                detect_ms: 2.0,
                index_ms: 0.0,
                solve_ms: 3.0,
                total_ms: 6.0,
            },
        );
        assert_eq!(result.wcs.pixel_origin, 1);
        assert_eq!(result.wcs.crpix, [2000.0, 1500.0]);
        assert_eq!(result.matched_stars, 42);
    }

    #[test]
    fn remote_envelope_round_trips_compact_gray_pixels() {
        let image = image::GrayImage::from_pixel(1024, 768, image::Luma([7]));
        let params = SolveParams {
            image_path: PathBuf::from("image.fits"),
            mode: SolveMode::Hinted,
            hint: Some(HintParams {
                center_ra_deg: 10.0,
                center_dec_deg: 20.0,
                radius_deg: 2.0,
                scale_arcsec_per_pixel: 1.5,
                scale_tolerance: 0.2,
            }),
            blind: None,
            detection: DetectionParams::default(),
        };

        let encoded = encode_remote_request(&image, &params).unwrap();
        assert!(encoded.len() < image.as_raw().len() / 10);
        let (metadata, decoded) = decode_remote_request(&encoded).unwrap();
        assert_eq!(metadata.image.encoding, "gray8-zstd");
        assert_eq!(decoded.dimensions(), image.dimensions());
        assert_eq!(decoded.as_raw(), image.as_raw());
    }
}
