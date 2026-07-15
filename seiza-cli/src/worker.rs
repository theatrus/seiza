//! Persistent plate-solving worker over newline-delimited JSON-RPC 2.0.
//!
//! The worker keeps the star catalog and optional blind index open while it
//! reads requests from stdin. Each input line is one JSON-RPC request and
//! each output line is one response. Closing stdin cleanly ends the worker,
//! so the same protocol supports both long-lived and one-shot clients.

use anyhow::{Context, Result};
use image::ImageEncoder;
use seiza::blind::{BlindIndex, BlindParams, solve_blind};
use seiza::catalog::TileCatalog;
use seiza::solve::{Solution, SolveHint, solve};
use seiza::{DetectBackend, DetectConfig, Wcs};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use ureq::unversioned::multipart::{Form, Part};

const JSON_RPC_VERSION: &str = "2.0";
const PROTOCOL_VERSION: u32 = 1;
const MAX_REQUEST_BYTES: usize = 1024 * 1024;
const SERVER_POLL_INTERVAL: Duration = Duration::from_millis(250);
const SERVER_STATUS_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_ERROR_BODY_BYTES: u64 = 64 * 1024;

const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;
const NOT_INITIALIZED: i64 = -32002;
const SOLVE_FAILED: i64 = -32010;

/// Run a local or remote-backed worker on the process standard streams.
pub(crate) struct WorkerOptions<'a> {
    pub(crate) data_path: Option<&'a Path>,
    pub(crate) index_path: Option<&'a Path>,
    pub(crate) server: Option<&'a str>,
    pub(crate) server_token: Option<&'a str>,
    pub(crate) server_upload: ServerUploadFormat,
    pub(crate) server_timeout: Duration,
    pub(crate) detection_backend: DetectBackend,
    pub(crate) detection_fallback: crate::DetectionFallback,
    pub(crate) detection_fallback_hypotheses: usize,
}

pub(crate) fn run(options: WorkerOptions<'_>) -> Result<()> {
    let WorkerOptions {
        data_path,
        index_path,
        server,
        server_token,
        server_upload,
        server_timeout,
        detection_backend,
        detection_fallback,
        detection_fallback_hypotheses,
    } = options;
    let service = match (data_path, server) {
        (Some(data_path), None) => WorkerService::local(
            data_path,
            index_path,
            detection_backend,
            detection_fallback,
            detection_fallback_hypotheses,
        )?,
        (None, Some(server)) => {
            WorkerService::remote(server, server_token, server_upload, server_timeout)?
        }
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

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub(crate) enum ServerUploadFormat {
    /// MTF-stretched lossless 8-bit grayscale PNG
    Png,
    /// Original FITS bytes, including headers and source bit depth
    Fits,
}

#[derive(Debug, Deserialize)]
struct RpcRequest {
    jsonrpc: String,
    #[serde(default)]
    id: RequestId,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Default)]
enum RequestId {
    #[default]
    Missing,
    Present(Value),
}

impl RequestId {
    fn is_present(&self) -> bool {
        matches!(self, Self::Present(_))
    }

    fn is_valid(&self) -> bool {
        matches!(
            self,
            Self::Missing | Self::Present(Value::Null | Value::String(_) | Value::Number(_))
        )
    }

    fn into_value(self) -> Value {
        match self {
            Self::Missing => Value::Null,
            Self::Present(value) => value,
        }
    }
}

impl<'de> Deserialize<'de> for RequestId {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Value::deserialize(deserializer).map(Self::Present)
    }
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
    loop {
        let (handled, respond) = match read_request_line(&mut reader)? {
            RequestLine::Eof => break,
            RequestLine::TooLong => (
                HandledResponse {
                    response: RpcResponse::error(
                        Value::Null,
                        INVALID_REQUEST,
                        format!("request exceeds {MAX_REQUEST_BYTES} bytes"),
                    ),
                    shutdown: false,
                },
                true,
            ),
            RequestLine::Line(line) if line.iter().all(u8::is_ascii_whitespace) => continue,
            RequestLine::Line(line) => match serde_json::from_slice::<Value>(&line) {
                Err(error) => (
                    HandledResponse {
                        response: RpcResponse::error(
                            Value::Null,
                            PARSE_ERROR,
                            format!("invalid JSON: {error}"),
                        ),
                        shutdown: false,
                    },
                    true,
                ),
                Ok(value) => match serde_json::from_value::<RpcRequest>(value) {
                    Err(error) => (
                        HandledResponse {
                            response: RpcResponse::error(
                                Value::Null,
                                INVALID_REQUEST,
                                format!("invalid JSON-RPC request: {error}"),
                            ),
                            shutdown: false,
                        },
                        true,
                    ),
                    Ok(request) if !request.id.is_valid() => (
                        HandledResponse {
                            response: RpcResponse::error(
                                Value::Null,
                                INVALID_REQUEST,
                                "request id must be a string, number, or null",
                            ),
                            shutdown: false,
                        },
                        true,
                    ),
                    Ok(request) => {
                        let respond = request.id.is_present();
                        (service.handle(request), respond)
                    }
                },
            },
        };

        if respond {
            serde_json::to_writer(&mut writer, &handled.response)?;
            writer.write_all(b"\n")?;
            writer.flush()?;
        }
        if handled.shutdown {
            break;
        }
    }
    Ok(())
}

enum RequestLine {
    Eof,
    Line(Vec<u8>),
    TooLong,
}

/// Read and drain exactly one newline-delimited request without ever retaining
/// more than the protocol limit. `BufRead::read_line` cannot enforce this: it
/// allocates the complete line before the caller can inspect its length.
fn read_request_line<R: BufRead>(reader: &mut R) -> io::Result<RequestLine> {
    let mut line = Vec::with_capacity(8 * 1024);
    let mut too_long = false;

    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Ok(if too_long {
                RequestLine::TooLong
            } else if line.is_empty() {
                RequestLine::Eof
            } else {
                RequestLine::Line(line)
            });
        }

        let newline = available.iter().position(|byte| *byte == b'\n');
        let consumed = newline.map_or(available.len(), |offset| offset + 1);
        let payload_len = newline.unwrap_or(consumed);
        if !too_long {
            if line.len().saturating_add(payload_len) > MAX_REQUEST_BYTES {
                too_long = true;
                line.clear();
            } else {
                line.extend_from_slice(&available[..payload_len]);
            }
        }
        reader.consume(consumed);

        if newline.is_some() {
            return Ok(if too_long {
                RequestLine::TooLong
            } else {
                RequestLine::Line(line)
            });
        }
    }
}

struct LocalSolver {
    catalog: TileCatalog,
    index: Option<BlindIndex>,
    detection_backend: DetectBackend,
    detection_fallback: crate::DetectionFallback,
    detection_fallback_hypotheses: usize,
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
    fn local(
        data_path: &Path,
        index_path: Option<&Path>,
        detection_backend: DetectBackend,
        detection_fallback: crate::DetectionFallback,
        detection_fallback_hypotheses: usize,
    ) -> Result<Self> {
        Ok(Self {
            backend: WorkerBackend::Local(Box::new(LocalSolver::open(
                data_path,
                index_path,
                detection_backend,
                detection_fallback,
                detection_fallback_hypotheses,
            )?)),
            initialized: false,
        })
    }

    fn remote(
        server: &str,
        token: Option<&str>,
        upload_format: ServerUploadFormat,
        timeout: Duration,
    ) -> Result<Self> {
        Ok(Self {
            backend: WorkerBackend::Remote(RemoteSolver::new(
                server,
                token,
                upload_format,
                timeout,
            )?),
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
                    "imageInputs": ["image_path"],
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
    fn open(
        data_path: &Path,
        index_path: Option<&Path>,
        detection_backend: DetectBackend,
        detection_fallback: crate::DetectionFallback,
        detection_fallback_hypotheses: usize,
    ) -> Result<Self> {
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

        Ok(Self {
            catalog,
            index,
            detection_backend,
            detection_fallback,
            detection_fallback_hypotheses,
        })
    }

    fn status(&self) -> BackendStatus {
        BackendStatus {
            kind: "local".to_string(),
            server: ServerIdentity {
                name: "seiza".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            catalog: CatalogStatus {
                star_count: Some(self.catalog.star_count()),
                blind_index_loaded: Some(self.index.is_some()),
                blind_index_pattern_count: self.index.as_ref().map(BlindIndex::pattern_count),
                blind_index_magnitude_limit: self.index.as_ref().map(BlindIndex::index_mag_limit),
            },
            remote_image_encoding: None,
        }
    }

    fn solve_path(&mut self, params: &SolveParams) -> Result<WorkerSolveResult> {
        let total_started = Instant::now();

        let load_started = Instant::now();
        let image = crate::load_image(&params.image_path, self.detection_backend)?;
        let load_elapsed = load_started.elapsed();
        let dimensions = image.dimensions();

        let detect_started = Instant::now();
        let config = DetectConfig {
            backend: self.detection_backend,
            sigma: params.detection.sigma,
            ignore_border: params.detection.ignore_border,
            max_stars: params.detection.max_stars,
            ..Default::default()
        };
        let can_retry_f32 = self.detection_fallback == crate::DetectionFallback::F32
            && crate::auto_can_retry_f32(&params.image_path, &image, self.detection_backend);
        let mut invocation = crate::SolveInvocation::new(
            &params.image_path,
            &image,
            config,
            self.detection_fallback,
        );
        let detect_elapsed = detect_started.elapsed();

        let mut index_elapsed = Duration::ZERO;
        let solution = match params.mode {
            SolveMode::Hinted => {
                let hint = params.hint.as_ref().expect("validated hinted parameters");
                let hint = SolveHint {
                    center: (hint.center_ra_deg, hint.center_dec_deg),
                    radius_deg: hint.radius_deg,
                    scale_arcsec_px: hint.scale_arcsec_per_pixel,
                    scale_tolerance: hint.scale_tolerance,
                };
                invocation.solve(|stars| solve(stars, &self.catalog, &hint, dimensions))?
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

                invocation.solve_with_pass(|stars, pass| {
                    let attempt_params = crate::blind_params_for_detection_pass(
                        &blind_params,
                        can_retry_f32,
                        self.detection_fallback_hypotheses,
                        pass,
                    );
                    solve_blind(stars, &self.catalog, index, &attempt_params, dimensions)
                })?
            }
        };
        let solve_elapsed = total_started
            .elapsed()
            .saturating_sub(load_elapsed)
            .saturating_sub(detect_elapsed)
            .saturating_sub(index_elapsed);

        Ok(solution_result(
            params.mode,
            &solution,
            dimensions,
            SolveTimings {
                load_ms: milliseconds(load_elapsed),
                encode_ms: 0.0,
                transport_ms: 0.0,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    star_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    blind_index_loaded: Option<bool>,
    blind_index_pattern_count: Option<usize>,
    blind_index_magnitude_limit: Option<f32>,
}

struct RemoteSolver {
    base_url: String,
    token: Option<String>,
    agent: ureq::Agent,
    upload_format: ServerUploadFormat,
    timeout: Duration,
}

impl RemoteSolver {
    fn new(
        base_url: &str,
        token: Option<&str>,
        upload_format: ServerUploadFormat,
        timeout: Duration,
    ) -> Result<Self> {
        let base_url = base_url.trim_end_matches('/');
        if !(base_url.starts_with("http://") || base_url.starts_with("https://")) {
            anyhow::bail!("remote server URL must start with http:// or https://");
        }
        if timeout.is_zero() {
            anyhow::bail!("remote server timeout must be greater than zero");
        }
        Ok(Self {
            base_url: base_url.to_string(),
            token: token.map(str::to_string),
            agent: ureq::Agent::config_builder()
                .http_status_as_error(false)
                .build()
                .new_agent(),
            upload_format,
            timeout,
        })
    }

    fn status(&self) -> Result<BackendStatus> {
        let timeout = self.timeout.min(SERVER_STATUS_TIMEOUT);
        let mut response = self.get(&format!("{}/api/v1/health", self.base_url), timeout)?;
        let health: ServerHealth = response
            .body_mut()
            .read_json()
            .context("invalid health response from seiza-server")?;
        if !health.solver_ready {
            anyhow::bail!(
                "seiza-server is {} but its solver is not ready",
                health.status
            );
        }
        Ok(BackendStatus {
            kind: "remote".to_string(),
            server: ServerIdentity {
                name: "seiza-server".to_string(),
                version: health
                    .versions
                    .and_then(|versions| versions.seiza_server)
                    .unwrap_or_else(|| "api-v1".to_string()),
            },
            catalog: CatalogStatus {
                star_count: None,
                blind_index_loaded: None,
                blind_index_pattern_count: None,
                blind_index_magnitude_limit: None,
            },
            remote_image_encoding: Some(self.upload_format.encoding().to_string()),
        })
    }

    fn solve_path(&self, params: &SolveParams) -> Result<WorkerSolveResult> {
        let total_started = Instant::now();
        let load_started = Instant::now();
        let source = match self.upload_format {
            ServerUploadFormat::Png => RemoteUploadSource::Image(
                crate::load_image(&params.image_path, DetectBackend::U8)?.to_luma8(),
            ),
            ServerUploadFormat::Fits => RemoteUploadSource::Fits {
                path: if crate::is_fits_path(&params.image_path) {
                    params.image_path.clone()
                } else {
                    anyhow::bail!("--server-upload fits requires a FITS image path");
                },
                bytes: usize::try_from(
                    std::fs::metadata(&params.image_path)
                        .with_context(|| {
                            format!(
                                "failed to inspect {} for upload",
                                params.image_path.display()
                            )
                        })?
                        .len(),
                )
                .context("FITS upload is too large for this platform")?,
            },
        };
        let load_elapsed = load_started.elapsed();

        let encode_started = Instant::now();
        let upload = match source {
            RemoteUploadSource::Image(image) => RemoteUpload {
                data: RemoteUploadData::Bytes(encode_png(&image)?),
                filename: "nina-solve.png",
                content_type: "image/png",
                encoding: self.upload_format.encoding(),
                uncompressed_bytes: image.as_raw().len(),
                encoded_bytes: 0,
            },
            RemoteUploadSource::Fits { path, bytes } => RemoteUpload {
                data: RemoteUploadData::File(path),
                filename: "nina-solve.fits",
                content_type: "application/fits",
                encoding: self.upload_format.encoding(),
                uncompressed_bytes: bytes,
                encoded_bytes: bytes,
            },
        };
        let upload = upload.with_encoded_size();
        let options = server_options(params);
        let options = serde_json::to_string(&options)?;
        let encode_elapsed = encode_started.elapsed();

        let transport_started = Instant::now();
        let form = upload.form(&options)?;
        let mut response = self.post_form(
            &format!("{}/api/v1/solves", self.base_url),
            self.remaining(transport_started)?,
            form,
        )?;
        let mut job: ServerJob = response
            .body_mut()
            .read_json()
            .context("invalid job response from seiza-server")?;
        loop {
            match job.status {
                ServerJobStatus::Queued | ServerJobStatus::Solving => {
                    let remaining = self.remaining(transport_started)?;
                    std::thread::sleep(SERVER_POLL_INTERVAL.min(remaining));
                    let mut response = self.get(
                        &format!("{}/api/v1/solves/{}", self.base_url, job.id),
                        self.remaining(transport_started)?,
                    )?;
                    job = response
                        .body_mut()
                        .read_json()
                        .context("invalid polling response from seiza-server")?;
                }
                ServerJobStatus::Failed => {
                    anyhow::bail!(
                        "seiza-server solve failed: {}",
                        job.error.as_deref().unwrap_or("unknown server error")
                    );
                }
                ServerJobStatus::Succeeded => break,
            }
        }
        let solution = job
            .solution
            .context("seiza-server reported success without a solution")?;
        let transport_elapsed = transport_started.elapsed();
        let mut result = solution.into_worker_result(params.mode);
        result.timings = SolveTimings {
            load_ms: milliseconds(load_elapsed),
            encode_ms: milliseconds(encode_elapsed),
            transport_ms: milliseconds(transport_elapsed),
            detect_ms: 0.0,
            index_ms: 0.0,
            solve_ms: 0.0,
            total_ms: milliseconds(total_started.elapsed()),
        };
        result.transfer = Some(RemoteTransfer {
            encoding: upload.encoding.to_string(),
            uncompressed_bytes: upload.uncompressed_bytes,
            encoded_bytes: upload.encoded_bytes,
        });
        Ok(result)
    }

    fn get(&self, url: &str, timeout: Duration) -> Result<HttpResponse> {
        let mut request = self.agent.get(url);
        if let Some(token) = &self.token {
            request = request.header("Authorization", &format!("Bearer {token}"));
        }
        let request = request.config().timeout_global(Some(timeout)).build();
        map_remote_response(request.call())
    }

    fn post_form(&self, url: &str, timeout: Duration, form: Form<'_>) -> Result<HttpResponse> {
        let mut request = self.agent.post(url);
        if let Some(token) = &self.token {
            request = request.header("Authorization", &format!("Bearer {token}"));
        }
        let request = request.config().timeout_global(Some(timeout)).build();
        map_remote_response(request.send(form))
    }

    fn remaining(&self, started: Instant) -> Result<Duration> {
        let remaining = self.timeout.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            anyhow::bail!(
                "seiza-server solve timed out after {:.1} seconds",
                self.timeout.as_secs_f64()
            );
        }
        Ok(remaining)
    }
}

impl ServerUploadFormat {
    fn encoding(self) -> &'static str {
        match self {
            Self::Png => "png-gray8",
            Self::Fits => "fits",
        }
    }
}

enum RemoteUploadSource {
    Image(image::GrayImage),
    Fits { path: PathBuf, bytes: usize },
}

enum RemoteUploadData {
    Bytes(Vec<u8>),
    File(PathBuf),
}

struct RemoteUpload {
    data: RemoteUploadData,
    filename: &'static str,
    content_type: &'static str,
    encoding: &'static str,
    uncompressed_bytes: usize,
    encoded_bytes: usize,
}

impl RemoteUpload {
    fn with_encoded_size(mut self) -> Self {
        if let RemoteUploadData::Bytes(bytes) = &self.data {
            self.encoded_bytes = bytes.len();
        }
        self
    }

    fn form<'a>(&'a self, options: &'a str) -> Result<Form<'a>> {
        let part = match &self.data {
            RemoteUploadData::Bytes(bytes) => Part::bytes(bytes),
            RemoteUploadData::File(path) => Part::file(path)
                .with_context(|| format!("failed to open {} for upload", path.display()))?,
        }
        .file_name(self.filename)
        .mime_str(self.content_type)?;
        Ok(Form::new().text("options", options).part("file", part))
    }
}

type HttpResponse = ureq::http::Response<ureq::Body>;

fn map_remote_response(
    response: std::result::Result<HttpResponse, ureq::Error>,
) -> Result<HttpResponse> {
    let mut response = response
        .map_err(|error| anyhow::Error::new(error).context("seiza-server request failed"))?;
    if !response.status().is_success() {
        let status = response.status();
        let message = response
            .body_mut()
            .with_config()
            .limit(MAX_ERROR_BODY_BYTES)
            .read_to_string()
            .unwrap_or_else(|error| format!("failed to read error response: {error}"));
        anyhow::bail!("seiza-server returned HTTP {status}: {message}");
    }
    Ok(response)
}

#[derive(Debug, Deserialize)]
struct ServerHealth {
    status: String,
    solver_ready: bool,
    #[serde(default)]
    versions: Option<ServerVersions>,
}

#[derive(Debug, Deserialize)]
struct ServerVersions {
    #[serde(default)]
    seiza_server: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ServerJob {
    id: String,
    status: ServerJobStatus,
    solution: Option<ServerSolution>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ServerJobStatus {
    Queued,
    Solving,
    Succeeded,
    Failed,
}

#[derive(Debug, Deserialize)]
struct ServerSolution {
    center_ra_deg: f64,
    center_dec_deg: f64,
    pixel_scale_arcsec_per_pixel: f64,
    matched_stars: usize,
    rms_arcsec: f64,
    image_width: u32,
    image_height: u32,
    wcs: ServerWcs,
}

impl ServerSolution {
    fn into_worker_result(self, mode: SolveMode) -> WorkerSolveResult {
        let solution = Solution {
            wcs: Wcs {
                crval: (self.wcs.crval[0], self.wcs.crval[1]),
                crpix: (self.wcs.crpix[0], self.wcs.crpix[1]),
                cd: self.wcs.cd,
            },
            matched_stars: self.matched_stars,
            rms_arcsec: self.rms_arcsec,
        };
        let mut result = solution_result(
            mode,
            &solution,
            (self.image_width, self.image_height),
            SolveTimings::default(),
        );
        result.center = SkyCoordinate {
            ra_deg: self.center_ra_deg,
            dec_deg: self.center_dec_deg,
        };
        result.pixel_scale_arcsec_per_pixel = self.pixel_scale_arcsec_per_pixel;
        result
    }
}

#[derive(Debug, Deserialize)]
struct ServerWcs {
    crval: [f64; 2],
    crpix: [f64; 2],
    cd: [[f64; 2]; 2],
}

#[derive(Debug, Serialize)]
struct ServerSolveOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    center_ra_deg: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    center_dec_deg: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    radius_deg: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    scale_arcsec_per_pixel: Option<f64>,
    scale_tolerance: f64,
    min_scale_arcsec_per_pixel: f64,
    max_scale_arcsec_per_pixel: f64,
    sigma: f32,
    ignore_border: u32,
    max_stars: usize,
}

fn server_options(params: &SolveParams) -> ServerSolveOptions {
    // A client may retain both option objects while switching modes. Only the
    // selected mode may influence the server request, or a nominally blind
    // remote request would silently become hinted while local mode stays blind.
    let hint = match params.mode {
        SolveMode::Hinted => params.hint.as_ref(),
        SolveMode::Blind => None,
    };
    let blind = params.blind.clone().unwrap_or_default();
    ServerSolveOptions {
        center_ra_deg: hint.map(|hint| hint.center_ra_deg),
        center_dec_deg: hint.map(|hint| hint.center_dec_deg),
        radius_deg: hint.map(|hint| hint.radius_deg),
        scale_arcsec_per_pixel: hint.map(|hint| hint.scale_arcsec_per_pixel),
        scale_tolerance: hint.map_or_else(default_scale_tolerance, |hint| hint.scale_tolerance),
        min_scale_arcsec_per_pixel: blind.min_scale_arcsec_per_pixel,
        max_scale_arcsec_per_pixel: blind.max_scale_arcsec_per_pixel,
        sigma: params.detection.sigma,
        ignore_border: params.detection.ignore_border,
        max_stars: params.detection.max_stars,
    }
}

fn encode_png(image: &image::GrayImage) -> Result<Vec<u8>> {
    let mut png = Vec::new();
    image::codecs::png::PngEncoder::new(&mut png)
        .write_image(
            image.as_raw(),
            image.width(),
            image.height(),
            image::ExtendedColorType::L8,
        )
        .context("failed to encode compact PNG for seiza-server")?;
    Ok(png)
}

impl RpcService for WorkerService {
    fn handle(&mut self, request: RpcRequest) -> HandledResponse {
        let id = request.id.into_value();
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
        if !(0.0..=360.0).contains(&self.center_ra_deg) {
            return Err("hint.centerRaDeg must be between 0 and 360".to_string());
        }
        if !(-90.0..=90.0).contains(&self.center_dec_deg) {
            return Err("hint.centerDecDeg must be between -90 and 90".to_string());
        }
        if self.radius_deg <= 0.0 || self.radius_deg > 180.0 {
            return Err("hint.radiusDeg must be greater than 0 and at most 180".to_string());
        }
        if self.scale_arcsec_per_pixel <= 0.0 {
            return Err("hint.scaleArcsecPerPixel must be greater than 0".to_string());
        }
        if !(0.01..=1.0).contains(&self.scale_tolerance) {
            return Err("hint.scaleTolerance must be between 0.01 and 1.0".to_string());
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
        if !self.index_magnitude_limit.is_finite() {
            return Err("blind.indexMagnitudeLimit must be finite".to_string());
        }
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

#[derive(Debug, Default, Deserialize, Serialize)]
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
    use std::io::{Cursor, Read};

    struct TestService {
        calls: usize,
    }

    impl RpcService for TestService {
        fn handle(&mut self, request: RpcRequest) -> HandledResponse {
            self.calls += 1;
            let shutdown = request.method == "shutdown";
            HandledResponse {
                response: RpcResponse::success(
                    request.id.into_value(),
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
    fn notification_executes_without_writing_a_response() {
        let input = concat!(
            "{\"jsonrpc\":\"2.0\",\"method\":\"ping\"}\n",
            "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}\n"
        );
        let mut output = Vec::new();
        run_loop(
            Cursor::new(input.as_bytes()),
            &mut output,
            TestService { calls: 0 },
        )
        .unwrap();
        let response: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(response["id"], 2);
        assert_eq!(response["result"]["call"], 2);
    }

    #[test]
    fn compound_request_id_is_rejected() {
        let input = "{\"jsonrpc\":\"2.0\",\"id\":{},\"method\":\"ping\"}\n";
        let mut output = Vec::new();
        run_loop(
            Cursor::new(input.as_bytes()),
            &mut output,
            TestService { calls: 0 },
        )
        .unwrap();
        let response: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(response["id"], Value::Null);
        assert_eq!(response["error"]["code"], INVALID_REQUEST);
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
    fn valid_json_with_no_method_is_an_invalid_request() {
        let mut output = Vec::new();
        run_loop(Cursor::new(b"{}\n"), &mut output, TestService { calls: 0 }).unwrap();
        let response: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(response["error"]["code"], INVALID_REQUEST);
    }

    #[test]
    fn oversized_request_is_drained_before_the_next_request() {
        let mut input = vec![b' '; MAX_REQUEST_BYTES + 1];
        input.extend_from_slice(b"\n{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"ping\"}\n");
        let mut output = Vec::new();
        run_loop(Cursor::new(input), &mut output, TestService { calls: 0 }).unwrap();
        let responses: Vec<Value> = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0]["error"]["code"], INVALID_REQUEST);
        assert_eq!(responses[1]["id"], 9);
    }

    #[test]
    fn invalid_utf8_returns_parse_error_and_keeps_reading() {
        let input = b"\xff\n{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"ping\"}\n";
        let mut output = Vec::new();
        run_loop(Cursor::new(input), &mut output, TestService { calls: 0 }).unwrap();
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
    fn blind_server_options_ignore_a_stale_hint() {
        let params: SolveParams = serde_json::from_value(json!({
            "imagePath": "image.fits",
            "mode": "blind",
            "hint": {
                "centerRaDeg": 10.0,
                "centerDecDeg": 20.0,
                "radiusDeg": 2.0,
                "scaleArcsecPerPixel": 1.5
            }
        }))
        .unwrap();
        params.validate().unwrap();
        let options = server_options(&params);
        assert!(options.center_ra_deg.is_none());
        assert!(options.center_dec_deg.is_none());
        assert!(options.radius_deg.is_none());
        assert!(options.scale_arcsec_per_pixel.is_none());
    }

    #[test]
    fn expired_remote_deadline_is_reported() {
        let solver = RemoteSolver::new(
            "http://127.0.0.1:1",
            None,
            ServerUploadFormat::Png,
            Duration::from_secs(1),
        )
        .unwrap();
        let started = Instant::now().checked_sub(Duration::from_secs(2)).unwrap();
        assert!(
            solver
                .remaining(started)
                .unwrap_err()
                .to_string()
                .contains("timed out")
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
    fn remote_multipart_contains_compact_png_and_server_options() {
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

        let png = encode_png(&image).unwrap();
        assert!(png.len() < image.as_raw().len() / 10);
        let upload = RemoteUpload {
            encoded_bytes: png.len(),
            data: RemoteUploadData::Bytes(png),
            filename: "nina-solve.png",
            content_type: "image/png",
            encoding: "png-gray8",
            uncompressed_bytes: image.as_raw().len(),
        };
        let options = serde_json::to_string(&server_options(&params)).unwrap();
        let mut form = upload.form(&options).unwrap();
        let boundary = form.boundary().to_string();
        let mut body = Vec::new();
        form.read_to_end(&mut body).unwrap();
        assert!(body.windows(8).any(|window| window == b"\x89PNG\r\n\x1a\n"));
        let text = String::from_utf8_lossy(&body);
        assert!(text.starts_with(&format!("--{boundary}\r\n")));
        assert!(text.contains("name=\"options\""));
        assert!(text.contains("\"center_ra_deg\":10.0"));
        assert!(text.contains("name=\"file\"; filename=\"nina-solve.png\""));
    }
}
