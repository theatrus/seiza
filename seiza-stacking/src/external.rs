//! Driving external image-processing CLIs on stacked images.
//!
//! The first supported tool family is RC-Astro's `rc-astro` multi-tool
//! (BlurXTerminator `bxt`, StarXTerminator `sxt`, NoiseXTerminator `nxt`).
//! The CLI publishes a machine-readable contract: `rc-astro <tool> --json`
//! prints a schema document naming every parameter with its flag, type,
//! range, and default, plus the license state; a processing run under
//! `--json` emits NDJSON events (progress, per-file save status, warnings,
//! errors) on stdout. Parameter names and even flags change between CLI
//! builds (`--stars` vs `--difference`, `--nsr` vs `--nsd`), so everything
//! here resolves flags from the live schema instead of hard-coding them —
//! the same approach RC-Astro's own integration guide asks hosts to take.
//!
//! A tool consumes an image file and writes one or more image files:
//! [`RcAstroCli::run_on_file`] works at that level, and
//! [`RcAstroCli::process_image`] round-trips a [`LinearImage`] through a
//! temporary 32-bit-float FITS. StarXTerminator's optional stars output
//! (the original minus the starless result) comes back as a sidecar image
//! so a caller can stretch starless and stars independently.

use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use seiza_fits::HeaderValue;

use crate::cancel::CancelSignal;
use crate::fits::{FitsFrame, write_processed_image_fits_f32};
use crate::image::LinearImage;
use crate::{Error, Result};

/// The subcommands `rc-astro` offers for image processing.
pub const RC_ASTRO_TOOLS: [&str; 3] = ["bxt", "sxt", "nxt"];

/// The exchange bit depth for every run. Linear stacked data loses precision
/// below 32-bit float, and the tools accept it on every platform.
const EXCHANGE_DEPTH: &str = "32F";

/// A handle on an installed `rc-astro` executable.
#[derive(Debug, Clone)]
pub struct RcAstroCli {
    executable: PathBuf,
    host: String,
}

/// One tool's live contract, read from `rc-astro <tool> --json`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExternalToolSchema {
    /// Contract version of the document (v3 through v6 are known).
    pub schema_version: u32,
    /// The CLI's own version, e.g. "2.6.6". Part of any cache key: a CLI
    /// upgrade can change the output for identical inputs.
    pub cli_version: String,
    /// The subcommand, e.g. "sxt".
    pub key: String,
    /// Human name, e.g. "RC-Astro StarXTerminator".
    pub name: String,
    /// The neural-network model generation the run will use. Cache-key
    /// material for the same reason as `cli_version`.
    pub ml_version: Option<i64>,
    /// Whether the product reported a valid license. An unlicensed run
    /// would fail with exit code 77; refusing early gives a better message.
    pub licensed: bool,
    /// The license line as the CLI printed it, for display.
    pub license_message: Option<String>,
    /// Every parameter the document describes, GUI-only ones included.
    pub parameters: Vec<ExternalToolParameter>,
}

/// One schema parameter.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExternalToolParameter {
    /// Stable-ish name used to request a value, e.g. "stars".
    pub name: String,
    /// The CLI flag, absent for GUI-only parameters.
    pub flag: Option<String>,
    /// Display label, e.g. "Generate Star Image".
    pub label: String,
    /// Display description.
    pub description: String,
    /// Type, default, and range.
    pub kind: ExternalParameterKind,
}

/// A parameter's type with its default and range.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExternalParameterKind {
    /// A float with an inclusive range.
    Float {
        /// Value used when the parameter is not requested.
        default: f64,
        /// Smallest accepted value.
        min: f64,
        /// Largest accepted value.
        max: f64,
    },
    /// An on/off switch. The flag is emitted only when the value is true.
    Bool {
        /// Value used when the parameter is not requested.
        default: bool,
    },
    /// An integer with an inclusive range.
    Int {
        /// Value used when the parameter is not requested.
        default: i64,
        /// Smallest accepted value.
        min: i64,
        /// Largest accepted value.
        max: i64,
    },
}

/// A requested parameter value, matched against the schema by name.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum ExternalParameterValue {
    /// For [`ExternalParameterKind::Bool`] parameters.
    Bool(bool),
    /// For [`ExternalParameterKind::Int`] parameters.
    Int(i64),
    /// For [`ExternalParameterKind::Float`] parameters.
    Float(f64),
}

/// What to run: a tool, its parameter values, and the compute device.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExternalToolRequest {
    /// The subcommand, e.g. "sxt". Must match the schema's `key`.
    pub tool: String,
    /// Parameter values by schema name. Anything absent keeps the tool's
    /// default. An unknown or GUI-only name is refused before spawning.
    pub parameters: Vec<(String, ExternalParameterValue)>,
    /// Compute device: "auto", "cpu", "gpu", or "gpuN". None uses the
    /// tool's saved default.
    pub device: Option<String>,
}

/// What a run produced.
#[derive(Debug, Clone)]
pub struct ExternalToolRun {
    /// The requested output file.
    pub primary: PathBuf,
    /// Extra files the tool wrote beside the output — StarXTerminator's
    /// stars image when "stars" was requested.
    pub sidecars: Vec<PathBuf>,
    /// The compute device the tool reported using, e.g. "cpu".
    pub device: Option<String>,
    /// Warning messages from the event stream.
    pub warnings: Vec<String>,
}

/// A [`LinearImage`] round trip through a tool.
#[derive(Debug, Clone)]
pub struct ProcessedStackImage {
    /// The processed image, same dimensions and channel count as the input.
    pub image: LinearImage,
    /// StarXTerminator's stars image when one was produced.
    pub stars: Option<LinearImage>,
    /// The compute device the tool reported using.
    pub device: Option<String>,
    /// Warning messages from the event stream.
    pub warnings: Vec<String>,
}

fn external_error(tool: &str, message: impl Into<String>) -> Error {
    Error::ExternalTool {
        tool: tool.to_string(),
        message: message.into(),
    }
}

impl RcAstroCli {
    /// Use the `rc-astro` found on `PATH`, if any.
    pub fn locate() -> Option<Self> {
        let path = std::env::var_os("PATH")?;
        for directory in std::env::split_paths(&path) {
            for name in ["rc-astro", "rc-astro.exe"] {
                let candidate = directory.join(name);
                if candidate.is_file() {
                    return Some(Self::with_executable(candidate));
                }
            }
        }
        None
    }

    /// Use a specific executable.
    pub fn with_executable(executable: impl Into<PathBuf>) -> Self {
        Self {
            executable: executable.into(),
            host: format!("seiza-stacking-{}", env!("CARGO_PKG_VERSION")),
        }
    }

    /// Identify the integrating application to RC-Astro support (their
    /// `--host` option). Include a version, e.g. "psf-guard-0.8.0".
    pub fn with_host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    /// The executable this handle runs.
    pub fn executable(&self) -> &Path {
        &self.executable
    }

    /// Read one tool's live contract: `rc-astro <tool> --json`.
    pub fn tool_schema(&self, tool: &str) -> Result<ExternalToolSchema> {
        let output = Command::new(&self.executable)
            .arg(tool)
            .arg("--json")
            .stdin(Stdio::null())
            .output()
            .map_err(|error| external_error(tool, format!("could not run rc-astro: {error}")))?;
        let document: serde_json::Value = serde_json::from_slice(&output.stdout)
            .map_err(|error| external_error(tool, format!("schema is not JSON: {error}")))?;
        parse_schema(tool, &document)
    }

    /// Run a tool on `input`, writing `output` (and possibly sidecars
    /// beside it). Progress arrives as a fraction in `0.0..=1.0`.
    pub fn run_on_file(
        &self,
        schema: &ExternalToolSchema,
        request: &ExternalToolRequest,
        input: &Path,
        output: &Path,
        cancel: Option<&CancelSignal>,
        progress: &mut dyn FnMut(f32),
    ) -> Result<ExternalToolRun> {
        if !schema.licensed {
            let detail = schema
                .license_message
                .as_deref()
                .unwrap_or("the schema reports no valid license");
            return Err(external_error(
                &request.tool,
                format!("not licensed: {detail}"),
            ));
        }
        // The child runs from the executable's directory (the layout some
        // builds expect for finding models), so both paths must be absolute.
        let input = std::path::absolute(input)
            .map_err(|error| external_error(&request.tool, format!("input path: {error}")))?;
        let output = std::path::absolute(output)
            .map_err(|error| external_error(&request.tool, format!("output path: {error}")))?;
        let arguments = build_arguments(schema, request, &self.host, &input, &output)?;

        let mut command = Command::new(&self.executable);
        command
            .args(&arguments)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(directory) = self.executable.parent()
            && !directory.as_os_str().is_empty()
        {
            command.current_dir(directory);
        }
        let mut child = command
            .spawn()
            .map_err(|error| external_error(&request.tool, format!("could not spawn: {error}")))?;

        let stdout = child.stdout.take().expect("stdout was piped");
        let mut events = RunEvents::default();
        let reader = std::io::BufReader::new(stdout);
        for line in reader.lines() {
            if let Some(cancel) = cancel
                && cancel.is_cancelled()
            {
                let _ = child.kill();
                let _ = child.wait();
                return Err(Error::Cancelled);
            }
            let Ok(line) = line else { break };
            absorb_event_line(&line, &mut events, progress);
        }

        let status = child
            .wait()
            .map_err(|error| external_error(&request.tool, format!("wait failed: {error}")))?;
        if let Some(cancel) = cancel
            && cancel.is_cancelled()
        {
            return Err(Error::Cancelled);
        }
        if !status.success() {
            // Exit 77 is the CLI's documented "unlicensed" code.
            let mut message = if status.code() == Some(77) {
                "the product is not licensed on this machine".to_string()
            } else {
                format!("exited with {status}")
            };
            if !events.errors.is_empty() {
                message = format!("{message}: {}", events.errors.join("; "));
            }
            return Err(external_error(&request.tool, message));
        }
        if !output.is_file() {
            return Err(external_error(
                &request.tool,
                format!("reported success but wrote no {}", output.display()),
            ));
        }

        let mut sidecars: Vec<PathBuf> = events
            .outputs
            .iter()
            .map(|reported| {
                // Event paths mirror what we passed for -o (absolute), but a
                // build that reports relative names resolves beside the output.
                if reported.is_absolute() {
                    reported.clone()
                } else {
                    output
                        .parent()
                        .map(|parent| parent.join(reported))
                        .unwrap_or_else(|| reported.clone())
                }
            })
            .filter(|path| path != &output && path.is_file())
            .collect();
        sidecars.dedup();
        Ok(ExternalToolRun {
            primary: output,
            sidecars,
            device: events.device,
            warnings: events.warnings,
        })
    }

    /// Round-trip a [`LinearImage`] through a tool: write it as 32-bit-float
    /// FITS, run, and read the result (and any stars sidecar) back. WCS and
    /// observation metadata from `reference_headers` ride along so the
    /// processed file keeps its provenance.
    ///
    /// The tools clamp float samples to `[0, 1]` (the PixInsight
    /// convention), so an image on a physical ADU scale is divided by its
    /// peak on the way out and multiplied back on the way in — starless and
    /// stars alike, so their sum still reconstructs the original.
    pub fn process_image(
        &self,
        schema: &ExternalToolSchema,
        request: &ExternalToolRequest,
        image: &LinearImage,
        reference_headers: &[(String, HeaderValue)],
        cancel: Option<&CancelSignal>,
        progress: &mut dyn FnMut(f32),
    ) -> Result<ProcessedStackImage> {
        let workdir = tempfile::tempdir()
            .map_err(|error| external_error(&request.tool, format!("tempdir: {error}")))?;
        let input = workdir.path().join("input.fits");
        let output = workdir.path().join("processed.fits");
        let peak = image.data.iter().copied().fold(0.0f32, f32::max);
        let scale = if peak > 1.0 { peak } else { 1.0 };
        if scale > 1.0 {
            let scaled = LinearImage::new(
                image.width,
                image.height,
                image.channels,
                image.data.iter().map(|sample| sample / scale).collect(),
            )?;
            write_processed_image_fits_f32(&input, &scaled, reference_headers, &[])?;
        } else {
            write_processed_image_fits_f32(&input, image, reference_headers, &[])?;
        }

        let run = self.run_on_file(schema, request, &input, &output, cancel, progress)?;

        let read_back = |path: &Path| -> Result<LinearImage> {
            let frame = FitsFrame::open(path)?;
            if frame.image.width != image.width
                || frame.image.height != image.height
                || frame.image.channels != image.channels
            {
                return Err(external_error(
                    &request.tool,
                    format!(
                        "output geometry {}x{}x{} does not match input {}x{}x{}",
                        frame.image.width,
                        frame.image.height,
                        frame.image.channels,
                        image.width,
                        image.height,
                        image.channels
                    ),
                ));
            }
            let mut restored = frame.image;
            if scale > 1.0 {
                for sample in &mut restored.data {
                    *sample *= scale;
                }
            }
            Ok(restored)
        };

        let processed = read_back(&run.primary)?;
        let stars = match run.sidecars.first() {
            Some(sidecar) => Some(read_back(sidecar)?),
            None => None,
        };
        Ok(ProcessedStackImage {
            image: processed,
            stars,
            device: run.device,
            warnings: run.warnings,
        })
    }
}

/// Read a schema document into [`ExternalToolSchema`], tolerating the
/// contract's known versions (v3 through v6) and unknown extra fields.
fn parse_schema(tool: &str, document: &serde_json::Value) -> Result<ExternalToolSchema> {
    let schema_version = document
        .get("schemaVersion")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| external_error(tool, "schema document has no schemaVersion"))?
        as u32;
    let key = document
        .get("key")
        .and_then(serde_json::Value::as_str)
        .unwrap_or(tool)
        .to_string();
    // v3 documents carry no license object; the run's exit code 77 remains
    // the backstop, so absence reads as licensed.
    let (licensed, license_message) = match document.get("license") {
        Some(license) => (
            license
                .get("valid")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            license
                .get("message")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
        ),
        None => (true, None),
    };
    let mut parameters = Vec::new();
    if let Some(list) = document
        .get("parameters")
        .and_then(serde_json::Value::as_array)
    {
        for entry in list {
            let Some(name) = entry.get("name").and_then(serde_json::Value::as_str) else {
                continue;
            };
            let kind = match entry.get("type").and_then(serde_json::Value::as_str) {
                Some("float") => ExternalParameterKind::Float {
                    default: number(entry, "default").unwrap_or(0.0),
                    min: number(entry, "min").unwrap_or(f64::NEG_INFINITY),
                    max: number(entry, "max").unwrap_or(f64::INFINITY),
                },
                Some("bool") => ExternalParameterKind::Bool {
                    default: entry
                        .get("default")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false),
                },
                Some("int") => ExternalParameterKind::Int {
                    default: integer(entry, "default").unwrap_or(0),
                    min: integer(entry, "min").unwrap_or(i64::MIN),
                    max: integer(entry, "max").unwrap_or(i64::MAX),
                },
                // A future parameter type: skip it rather than refuse the
                // whole tool; the tool's default applies.
                _ => continue,
            };
            parameters.push(ExternalToolParameter {
                name: name.to_string(),
                flag: entry
                    .get("flag")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string),
                label: entry
                    .get("label")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or(name)
                    .to_string(),
                description: entry
                    .get("description")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                kind,
            });
        }
    }
    Ok(ExternalToolSchema {
        schema_version,
        cli_version: document
            .get("cliVersion")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string(),
        key,
        name: document
            .get("name")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(tool)
            .to_string(),
        ml_version: document
            .get("mlVersion")
            .and_then(serde_json::Value::as_i64),
        licensed,
        license_message,
        parameters,
    })
}

fn number(entry: &serde_json::Value, field: &str) -> Option<f64> {
    entry.get(field).and_then(serde_json::Value::as_f64)
}

fn integer(entry: &serde_json::Value, field: &str) -> Option<i64> {
    entry.get(field).and_then(serde_json::Value::as_i64)
}

/// Build the argv (after the executable) for one run. Pure so tests can
/// check it without spawning anything.
fn build_arguments(
    schema: &ExternalToolSchema,
    request: &ExternalToolRequest,
    host: &str,
    input: &Path,
    output: &Path,
) -> Result<Vec<std::ffi::OsString>> {
    if request.tool != schema.key {
        return Err(external_error(
            &request.tool,
            format!("schema describes {:?}, not this tool", schema.key),
        ));
    }
    let mut arguments: Vec<std::ffi::OsString> = vec![request.tool.clone().into()];
    // --host exists from contract v6; an older CLI rejects it.
    if schema.schema_version >= 6 {
        arguments.push("--host".into());
        arguments.push(host.into());
    }
    arguments.push("-o".into());
    arguments.push(output.into());
    arguments.push("--overwrite".into());
    arguments.push("--depth".into());
    arguments.push(EXCHANGE_DEPTH.into());
    if let Some(device) = &request.device {
        // The compute selector was per-product --engine through contract v3
        // and became the global --device in v4.
        arguments.push(
            if schema.schema_version >= 4 {
                "--device"
            } else {
                "--engine"
            }
            .into(),
        );
        arguments.push(device.into());
    }
    arguments.push("--json".into());
    for (name, value) in &request.parameters {
        let parameter = schema
            .parameters
            .iter()
            .find(|parameter| &parameter.name == name)
            .ok_or_else(|| external_error(&request.tool, format!("unknown parameter {name:?}")))?;
        let Some(flag) = &parameter.flag else {
            return Err(external_error(
                &request.tool,
                format!("parameter {name:?} has no CLI flag"),
            ));
        };
        match (&parameter.kind, value) {
            (ExternalParameterKind::Bool { .. }, ExternalParameterValue::Bool(enabled)) => {
                if *enabled {
                    arguments.push(flag.into());
                }
            }
            (ExternalParameterKind::Float { min, max, .. }, ExternalParameterValue::Float(v)) => {
                if !v.is_finite() || v < min || v > max {
                    return Err(external_error(
                        &request.tool,
                        format!("{name} = {v} is outside [{min}, {max}]"),
                    ));
                }
                arguments.push(flag.into());
                arguments.push(format!("{v}").into());
            }
            (ExternalParameterKind::Int { min, max, .. }, ExternalParameterValue::Int(v)) => {
                if v < min || v > max {
                    return Err(external_error(
                        &request.tool,
                        format!("{name} = {v} is outside [{min}, {max}]"),
                    ));
                }
                arguments.push(flag.into());
                arguments.push(format!("{v}").into());
            }
            (expected, provided) => {
                return Err(external_error(
                    &request.tool,
                    format!("parameter {name:?} expects {expected:?}, got {provided:?}"),
                ));
            }
        }
    }
    arguments.push(input.into());
    Ok(arguments)
}

#[derive(Default)]
struct RunEvents {
    outputs: Vec<PathBuf>,
    device: Option<String>,
    warnings: Vec<String>,
    errors: Vec<String>,
}

/// Absorb one NDJSON event line. Unknown events and non-JSON lines are
/// ignored: the stream is additive across contract versions.
fn absorb_event_line(line: &str, events: &mut RunEvents, progress: &mut dyn FnMut(f32)) {
    let Ok(event) = serde_json::from_str::<serde_json::Value>(line) else {
        return;
    };
    match event.get("event").and_then(serde_json::Value::as_str) {
        Some("progress") => {
            if let Some(done) = event.get("done").and_then(serde_json::Value::as_f64) {
                progress((done as f32 / 100.0).clamp(0.0, 1.0));
            }
        }
        Some("status") => {
            if event.get("phase").and_then(serde_json::Value::as_str) == Some("complete")
                && let Some(output) = event.get("output").and_then(serde_json::Value::as_str)
            {
                events.outputs.push(PathBuf::from(output));
            }
        }
        // v3 reported the device as its own event; v4+ routes it through
        // info with topic "device". Same payload either way.
        Some("device") => {
            if let Some(device) = event.get("device").and_then(serde_json::Value::as_str) {
                events.device = Some(device.to_string());
            }
        }
        Some("info") => {
            if event.get("topic").and_then(serde_json::Value::as_str) == Some("device")
                && let Some(device) = event.get("device").and_then(serde_json::Value::as_str)
            {
                events.device = Some(device.to_string());
            }
        }
        Some("warning") => {
            if let Some(message) = event.get("message").and_then(serde_json::Value::as_str) {
                events.warnings.push(message.to_string());
            }
        }
        Some("error") => {
            if let Some(message) = event.get("message").and_then(serde_json::Value::as_str) {
                events.errors.push(message.to_string());
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trimmed from a real `rc-astro sxt --json` (CLI 2.6.6, contract v6).
    const SXT_SCHEMA: &str = r#"{
        "schemaVersion": 6,
        "cliVersion": "2.6.6",
        "key": "sxt",
        "name": "RC-Astro StarXTerminator",
        "mlVersion": 11,
        "license": {"status": "permanent", "valid": true, "message": "Permanently licensed"},
        "parameters": [
            {"label": "Tile Overlap", "name": "overlap", "flag": "--overlap",
             "description": "Fractional overlap", "type": "float",
             "precision": 2, "default": 0.2, "max": 0.5, "min": 0.0},
            {"label": "Generate Star Image", "name": "stars", "flag": "--stars",
             "description": "Also write a stars-only image", "type": "bool", "default": false},
            {"label": "Unscreen Stars", "name": "unscreen", "flag": "--unscreen",
             "description": "Unscreen stars", "type": "bool", "default": false},
            {"label": "Color Separation", "name": "csep", "type": "bool", "default": false}
        ]
    }"#;

    /// Real event stream from an `rc-astro sxt --stars` run (2.6.6).
    const SXT_EVENTS: &str = concat!(
        r#"{"event":"info","topic":"version","cliVersion":"2.6.6","schemaVersion":6}"#,
        "\n",
        r#"{"event":"status","phase":"initializing","message":"Initializing"}"#,
        "\n",
        r#"{"event":"info","topic":"device","device":"cpu","id":"cpu","name":"","provider":"CPU","runtime":"onnxruntime 1.23.2"}"#,
        "\n",
        r#"{"event":"progress","done":50.0,"mpPerSec":0.1,"eta":1.0}"#,
        "\n",
        r#"{"event":"progress","done":100.0,"mpPerSec":0.1,"eta":0.0}"#,
        "\n",
        r#"{"event":"status","phase":"saving","message":"Saving","output":"starless.fits"}"#,
        "\n",
        r#"{"event":"status","phase":"complete","message":"Done","output":"starless.fits"}"#,
        "\n",
        r#"{"event":"status","phase":"saving","message":"Saving","output":"starless-stars.fits"}"#,
        "\n",
        r#"{"event":"status","phase":"complete","message":"Done","output":"starless-stars.fits"}"#,
        "\n",
    );

    fn sxt_schema() -> ExternalToolSchema {
        parse_schema("sxt", &serde_json::from_str(SXT_SCHEMA).unwrap()).unwrap()
    }

    #[test]
    fn the_live_schema_parses_with_types_ranges_and_license() {
        let schema = sxt_schema();
        assert_eq!(schema.schema_version, 6);
        assert_eq!(schema.cli_version, "2.6.6");
        assert_eq!(schema.key, "sxt");
        assert_eq!(schema.ml_version, Some(11));
        assert!(schema.licensed);
        assert_eq!(schema.parameters.len(), 4);
        let overlap = &schema.parameters[0];
        assert_eq!(overlap.flag.as_deref(), Some("--overlap"));
        assert!(matches!(
            overlap.kind,
            ExternalParameterKind::Float { min, max, .. } if min == 0.0 && max == 0.5
        ));
        // csep carries no flag in this document: GUI-only.
        assert_eq!(schema.parameters[3].flag, None);
    }

    #[test]
    fn a_v3_document_without_a_license_object_reads_as_licensed() {
        let schema = parse_schema(
            "nxt",
            &serde_json::json!({"schemaVersion": 3, "key": "nxt", "parameters": []}),
        )
        .unwrap();
        assert!(schema.licensed);
        assert_eq!(schema.license_message, None);
    }

    fn request(parameters: Vec<(&str, ExternalParameterValue)>) -> ExternalToolRequest {
        ExternalToolRequest {
            tool: "sxt".into(),
            parameters: parameters
                .into_iter()
                .map(|(name, value)| (name.to_string(), value))
                .collect(),
            device: Some("cpu".into()),
        }
    }

    fn strings(arguments: &[std::ffi::OsString]) -> Vec<String> {
        arguments
            .iter()
            .map(|value| value.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn arguments_resolve_flags_from_the_schema() {
        let arguments = build_arguments(
            &sxt_schema(),
            &request(vec![
                ("stars", ExternalParameterValue::Bool(true)),
                ("overlap", ExternalParameterValue::Float(0.3)),
            ]),
            "psf-guard-test",
            Path::new("/work/in.fits"),
            Path::new("/work/out.fits"),
        )
        .unwrap();
        assert_eq!(
            strings(&arguments),
            [
                "sxt",
                "--host",
                "psf-guard-test",
                "-o",
                "/work/out.fits",
                "--overwrite",
                "--depth",
                "32F",
                "--device",
                "cpu",
                "--json",
                "--stars",
                "--overlap",
                "0.3",
                "/work/in.fits",
            ]
        );
    }

    #[test]
    fn a_false_switch_emits_no_flag_and_old_contracts_drop_host() {
        let mut schema = sxt_schema();
        schema.schema_version = 4;
        let arguments = build_arguments(
            &schema,
            &request(vec![("stars", ExternalParameterValue::Bool(false))]),
            "psf-guard-test",
            Path::new("/work/in.fits"),
            Path::new("/work/out.fits"),
        )
        .unwrap();
        let rendered = strings(&arguments);
        assert!(!rendered.contains(&"--stars".to_string()));
        assert!(!rendered.contains(&"--host".to_string()));
        assert!(rendered.contains(&"--device".to_string()));
    }

    #[test]
    fn a_v3_contract_uses_engine_instead_of_device() {
        let mut schema = sxt_schema();
        schema.schema_version = 3;
        let arguments = build_arguments(
            &schema,
            &request(vec![]),
            "h",
            Path::new("/i"),
            Path::new("/o"),
        )
        .unwrap();
        let rendered = strings(&arguments);
        assert!(rendered.contains(&"--engine".to_string()));
        assert!(!rendered.contains(&"--device".to_string()));
    }

    #[test]
    fn out_of_range_unknown_and_gui_only_parameters_are_refused() {
        let schema = sxt_schema();
        for parameters in [
            vec![("overlap", ExternalParameterValue::Float(0.9))],
            vec![("no-such", ExternalParameterValue::Bool(true))],
            vec![("csep", ExternalParameterValue::Bool(true))],
            vec![("stars", ExternalParameterValue::Float(1.0))],
        ] {
            let result = build_arguments(
                &schema,
                &request(parameters),
                "h",
                Path::new("/i"),
                Path::new("/o"),
            );
            assert!(result.is_err());
        }
    }

    #[test]
    fn the_event_stream_yields_outputs_device_and_progress() {
        let mut events = RunEvents::default();
        let mut seen = Vec::new();
        for line in SXT_EVENTS.lines() {
            absorb_event_line(line, &mut events, &mut |fraction| seen.push(fraction));
        }
        assert_eq!(seen, vec![0.5, 1.0]);
        assert_eq!(events.device.as_deref(), Some("cpu"));
        assert_eq!(
            events.outputs,
            vec![
                PathBuf::from("starless.fits"),
                PathBuf::from("starless-stars.fits")
            ]
        );
        assert!(events.errors.is_empty());
    }

    #[test]
    fn an_unlicensed_schema_is_refused_before_spawning() {
        let mut schema = sxt_schema();
        schema.licensed = false;
        let cli = RcAstroCli::with_executable("/nonexistent/rc-astro");
        let error = cli
            .run_on_file(
                &schema,
                &request(vec![]),
                Path::new("/i.fits"),
                Path::new("/o.fits"),
                None,
                &mut |_| {},
            )
            .unwrap_err();
        assert!(error.to_string().contains("not licensed"));
    }

    /// A stand-in for rc-astro: emits the real event stream, copies the
    /// input to the output, and writes a stars sidecar when asked.
    #[cfg(unix)]
    fn fake_rc_astro(directory: &Path) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = directory.join("rc-astro");
        let script = r#"#!/bin/sh
out=""
input=""
stars=0
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    --stars) stars=1; shift ;;
    --host|--depth|--device) shift 2 ;;
    --*|sxt|bxt|nxt) shift ;;
    *) input="$1"; shift ;;
  esac
done
cp "$input" "$out"
echo '{"event":"info","topic":"device","device":"cpu"}'
echo '{"event":"progress","done":100.0}'
echo "{\"event\":\"status\",\"phase\":\"complete\",\"output\":\"$out\"}"
if [ "$stars" = 1 ]; then
  sidecar="${out%.fits}-stars.fits"
  cp "$input" "$sidecar"
  echo "{\"event\":\"status\",\"phase\":\"complete\",\"output\":\"$sidecar\"}"
fi
"#;
        std::fs::write(&path, script).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    #[cfg(unix)]
    #[test]
    fn a_linear_image_round_trips_with_a_stars_sidecar() {
        let directory = tempfile::tempdir().unwrap();
        let cli = RcAstroCli::with_executable(fake_rc_astro(directory.path()));
        let image = LinearImage::new(3, 2, 1, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        let mut fractions = Vec::new();
        let processed = cli
            .process_image(
                &sxt_schema(),
                &request(vec![("stars", ExternalParameterValue::Bool(true))]),
                &image,
                &[],
                None,
                &mut |fraction| fractions.push(fraction),
            )
            .unwrap();
        assert_eq!(processed.image.data, image.data);
        assert_eq!(processed.stars.unwrap().data, image.data);
        assert_eq!(processed.device.as_deref(), Some("cpu"));
        assert_eq!(fractions, vec![1.0]);
    }

    #[cfg(unix)]
    #[test]
    fn a_physical_adu_scale_survives_the_round_trip() {
        // The tools clamp float samples to [0, 1]; an ADU-scale stack must
        // come back on its own scale, not flattened to white.
        let directory = tempfile::tempdir().unwrap();
        let cli = RcAstroCli::with_executable(fake_rc_astro(directory.path()));
        let image = LinearImage::new(2, 2, 1, vec![512.0, 65_535.0, 1_024.0, 300.5]).unwrap();
        let processed = cli
            .process_image(
                &sxt_schema(),
                &request(vec![("stars", ExternalParameterValue::Bool(true))]),
                &image,
                &[],
                None,
                &mut |_| {},
            )
            .unwrap();
        for (restored, original) in processed.image.data.iter().zip(&image.data) {
            assert!((restored - original).abs() < original * 1e-6 + 1e-3);
        }
        let stars = processed.stars.unwrap();
        assert!((stars.data[1] - 65_535.0).abs() < 0.1);
    }

    #[cfg(unix)]
    #[test]
    fn without_the_stars_switch_no_sidecar_comes_back() {
        let directory = tempfile::tempdir().unwrap();
        let cli = RcAstroCli::with_executable(fake_rc_astro(directory.path()));
        let image = LinearImage::new(2, 2, 1, vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        let processed = cli
            .process_image(
                &sxt_schema(),
                &request(vec![]),
                &image,
                &[],
                None,
                &mut |_| {},
            )
            .unwrap();
        assert!(processed.stars.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn a_cancelled_run_reports_cancelled() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().unwrap();
        // A tool that emits one event then stalls, so cancellation lands
        // while the stream is open.
        let path = directory.path().join("rc-astro");
        std::fs::write(
            &path,
            "#!/bin/sh\necho '{\"event\":\"progress\",\"done\":1.0}'\nsleep 30\n",
        )
        .unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        let cli = RcAstroCli::with_executable(path);
        let cancel = CancelSignal::new({
            let cancelled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let flag = cancelled.clone();
            move || {
                flag.store(true, std::sync::atomic::Ordering::Relaxed);
                true
            }
        });
        let error = cli
            .run_on_file(
                &sxt_schema(),
                &request(vec![]),
                Path::new("/dev/null"),
                &directory.path().join("out.fits"),
                Some(&cancel),
                &mut |_| {},
            )
            .unwrap_err();
        assert!(matches!(error, Error::Cancelled));
    }
}
