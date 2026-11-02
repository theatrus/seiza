//! Running PixInsight processes on images, starting with MLDenoise.
//!
//! PixInsight is a desktop program that runs scripts from its command line:
//! `-n` starts a new instance, `--automation-mode` runs a script without
//! stopping at dialogs, `-r=` names the script and its arguments, and
//! `--force-exit` closes the instance when the script ends. It needs a
//! display; on Linux without one, `xvfb-run` supplies a virtual one.
//!
//! MLDenoise is the neural-network noise reduction of PixInsight's
//! MachineLearning module. It reads its network from a model file
//! (`.xmlm`), which PixInsight distributes separately.
//! [`PixInsight::ml_denoise_image`] writes the image to a temporary XISF
//! file on a 0 to 1 scale, runs a script that applies MLDenoise and saves
//! the result, and reads the result back on the image's own scale. The
//! script writes its outcome to a JSON file; PixInsight's own output is
//! kept only for an error message.

use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use seiza_fits::F32ImageData;

use crate::CancelSignal;
use crate::fits::{FitsFrame, write_processed_image_fits_f32};
use crate::image::LinearImage;
use crate::{Error, Result};

const TOOL: &str = "PixInsight MLDenoise";

/// The longest XMLM header read. Real headers are under a kilobyte; the
/// limit stops a damaged length field from allocating gigabytes.
const MAX_XMLM_HEADER: usize = 1 << 20;

/// How often a run checks for cancellation and for PixInsight exiting.
const POLL: Duration = Duration::from_millis(250);

/// The script PixInsight runs. Its arguments are the request file and the
/// result file; the request names the images and MLDenoise's parameters.
const ML_DENOISE_SCRIPT: &str = r#"#feature-id Seiza > MLDenoise
var requestPath = jsArguments[0];
var resultPath = jsArguments[1];
var result = { ok: false };
try {
   result.pixinsightVersion = CoreApplication.versionMajor + "." +
      CoreApplication.versionMinor + "." + CoreApplication.versionRelease;
   if (typeof MLDenoise === "undefined")
      throw new Error("this PixInsight has no MLDenoise process; it needs the MachineLearning module");
   var request = JSON.parse(File.readTextFile(requestPath));
   var started = Date.now();
   var windows = ImageWindow.open(request.input);
   if (windows.length != 1)
      throw new Error("PixInsight opened " + windows.length + " images from " + request.input + ", not one");
   var window = windows[0];
   try {
      var P = new MLDenoise;
      P.modelPath = request.model;
      P.amount = request.amount;
      P.overlapPercent = request.overlapPercent;
      // A process's enumeration constants are properties of its instances.
      P.precision = request.precision == "high" ? P.Precision_High : P.Precision_Fast;
      P.noisePreview = false;
      P.maskPreview = false;
      P.useCache = false;
      P.mask = request.localSupport != null;
      if (request.localSupport != null) {
         P.maskClipLow = request.localSupport.clipLow;
         P.maskBackground = request.localSupport.background;
         P.maskSmoothness = request.localSupport.smoothness;
      }
      console.beginLog();
      var executed = false;
      try {
         executed = P.executeOn(window.mainView, false);
      } finally {
         result.log = console.endLog().toString();
      }
      if (!executed)
         throw new Error("MLDenoise did not run");
      if (!window.saveAs(request.output, false, false, false, false))
         throw new Error("PixInsight could not save " + request.output);
   } finally {
      window.forceClose();
   }
   result.seconds = (Date.now() - started) / 1000;
   result.ok = true;
} catch (error) {
   result.error = "" + error;
}
File.writeTextFile(resultPath, JSON.stringify(result));
"#;

/// How MLDenoise runs inference on a GPU. CPU inference is always in full
/// precision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlDenoisePrecision {
    /// Reduced GPU precision: TensorFloat-32 on NVIDIA, low-precision
    /// accumulation in Core ML. PixInsight's default.
    #[default]
    Fast,
    /// The full float32 precision of the model's weights, about a quarter
    /// slower on a GPU.
    High,
}

impl std::fmt::Display for MlDenoisePrecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Fast => "fast",
            Self::High => "high",
        })
    }
}

impl std::str::FromStr for MlDenoisePrecision {
    type Err = Error;
    fn from_str(text: &str) -> Result<Self> {
        match text {
            "fast" => Ok(Self::Fast),
            "high" => Ok(Self::High),
            _ => Err(external_error(format!(
                "precision must be \"fast\" or \"high\", not {text:?}"
            ))),
        }
    }
}

/// MLDenoise's local support: a mask made from the image's brightness that
/// protects bright structures. Dark mask pixels block noise reduction and
/// bright ones allow all of it.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct MlDenoiseLocalSupport {
    /// Where the mask clips, in robust sigma units from the median, from
    /// -20 to 20. Positive values clip most of the background, which
    /// reduces more noise there. Default 0.
    pub clip_low: f64,
    /// The mask's mean background level after it is delinearized, from 0 to
    /// 1. Higher values protect more of the highlights. Default 0.25.
    pub background: f64,
    /// Radius in pixels of the median filter that smooths the mask, from 0
    /// to 7; 0 leaves it unsmoothed. Default 2.
    pub smoothness: u32,
}

impl Default for MlDenoiseLocalSupport {
    fn default() -> Self {
        Self {
            clip_low: 0.0,
            background: 0.25,
            smoothness: 2,
        }
    }
}

/// What MLDenoise runs with. The ranges are the ones PixInsight 1.9.6
/// enforces; [`Self::validate`] checks them before PixInsight starts.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MlDenoiseOptions {
    /// The model file (`.xmlm`).
    pub model: PathBuf,
    /// How much of the denoised image each pixel takes, from 0 to 1: the
    /// result is `(1 - amount) * original + amount * denoised`. Default 0.9.
    #[serde(default = "default_amount")]
    pub amount: f64,
    /// How much neighboring inference tiles overlap, in percent, from 12 to
    /// 50. Default 25.
    #[serde(default = "default_overlap_percent")]
    pub overlap_percent: u32,
    /// GPU inference precision. Default [`MlDenoisePrecision::Fast`].
    #[serde(default)]
    pub precision: MlDenoisePrecision,
    /// A brightness mask that protects bright structures. Default none:
    /// the whole image is denoised at `amount`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_support: Option<MlDenoiseLocalSupport>,
}

fn default_amount() -> f64 {
    0.9
}

fn default_overlap_percent() -> u32 {
    25
}

impl MlDenoiseOptions {
    /// PixInsight's defaults with `model`.
    pub fn new(model: impl Into<PathBuf>) -> Self {
        Self {
            model: model.into(),
            amount: default_amount(),
            overlap_percent: default_overlap_percent(),
            precision: MlDenoisePrecision::default(),
            local_support: None,
        }
    }

    /// Refuse values PixInsight would refuse, naming the field.
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.amount) {
            return Err(external_error(format!(
                "amount must be from 0 to 1, not {}",
                self.amount
            )));
        }
        if !(12..=50).contains(&self.overlap_percent) {
            return Err(external_error(format!(
                "overlap_percent must be from 12 to 50, not {}",
                self.overlap_percent
            )));
        }
        if let Some(support) = &self.local_support {
            if !(-20.0..=20.0).contains(&support.clip_low) {
                return Err(external_error(format!(
                    "local_support.clip_low must be from -20 to 20, not {}",
                    support.clip_low
                )));
            }
            if !(0.0..=1.0).contains(&support.background) {
                return Err(external_error(format!(
                    "local_support.background must be from 0 to 1, not {}",
                    support.background
                )));
            }
            if support.smoothness > 7 {
                return Err(external_error(format!(
                    "local_support.smoothness must be from 0 to 7, not {}",
                    support.smoothness
                )));
            }
        }
        Ok(())
    }
}

/// What an MLDenoise model file declares in its header.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MlDenoiseModel {
    /// The PixInsight process the model is for: `MLDenoise`.
    pub process: String,
    /// The model format version, such as `V3`.
    pub format_version: String,
    /// When the model was created, as the header writes it.
    pub created: Option<String>,
    /// Whether the model can denoise a one-channel image.
    pub mono: bool,
    /// Whether the model can denoise an RGB image.
    pub rgb: bool,
    /// The SHA-256 of the model's weights. A host caching results keys on
    /// it with the PixInsight version and the options: another model gives
    /// another result for the same input.
    pub weights_sha256: Option<String>,
}

impl MlDenoiseModel {
    /// Read the header of an `.xmlm` file, without reading its weights.
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let refuse = |message: String| external_error(format!("{}: {message}", path.display()));
        let mut file = std::fs::File::open(path)
            .map_err(|error| refuse(format!("cannot open the model file: {error}")))?;
        let mut lead = [0u8; 16];
        file.read_exact(&mut lead)
            .map_err(|_| refuse("not a PixInsight model file (.xmlm)".into()))?;
        if &lead[..8] != b"XMLM0100" {
            return Err(refuse("not a PixInsight model file (.xmlm)".into()));
        }
        let length = u32::from_le_bytes(lead[8..12].try_into().expect("four bytes")) as usize;
        if length == 0 || length > MAX_XMLM_HEADER {
            return Err(refuse(format!(
                "the model header length {length} is not valid"
            )));
        }
        let mut header = vec![0u8; length];
        file.read_exact(&mut header)
            .map_err(|_| refuse("the model header is cut short".into()))?;
        let model = parse_xmlm_header(&header).map_err(refuse)?;
        if model.process != "MLDenoise" {
            return Err(refuse(format!(
                "the model is for {}, not MLDenoise",
                model.process
            )));
        }
        Ok(model)
    }

    /// Whether the model can denoise an image with `channels` channels.
    pub fn accepts_channels(&self, channels: usize) -> bool {
        match channels {
            1 => self.mono,
            3 => self.rgb,
            _ => false,
        }
    }
}

fn parse_xmlm_header(xml: &[u8]) -> std::result::Result<MlDenoiseModel, String> {
    use quick_xml::events::Event;
    let mut reader = quick_xml::Reader::from_reader(xml);
    let mut buffer = Vec::new();
    let mut open: Vec<String> = Vec::new();
    let mut process = None;
    let mut format_version = None;
    let mut created = None;
    let mut mono = false;
    let mut rgb = false;
    let mut weights_sha256 = None;
    let mut model_element = |start: &quick_xml::events::BytesStart<'_>| {
        let mut role = None;
        let mut sha256 = None;
        for attribute in start.attributes().flatten() {
            let value = attribute
                .normalized_value(quick_xml::XmlVersion::Implicit1_0)
                .map(|value| value.into_owned())
                .unwrap_or_default();
            match attribute.key.as_ref() {
                "role" => role = Some(value),
                "sha256" => sha256 = Some(value),
                _ => {}
            }
        }
        match role.as_deref() {
            Some("mono") => mono = true,
            Some("rgb") => rgb = true,
            Some("weights") => weights_sha256 = sha256,
            _ => {}
        }
    };
    loop {
        match reader
            .read_event_into(&mut buffer)
            .map_err(|error| format!("the model header is not valid XML: {error}"))?
        {
            Event::Start(start) => {
                let name = local_name(start.name().as_ref());
                if name == "Model" {
                    model_element(&start);
                }
                open.push(name);
            }
            Event::Empty(start) => {
                if local_name(start.name().as_ref()) == "Model" {
                    model_element(&start);
                }
            }
            Event::End(_) => {
                open.pop();
            }
            Event::Text(text) => {
                let text = text.into_inner().trim().to_string();
                match open.last().map(String::as_str) {
                    Some("Process") => process = Some(text),
                    Some("Version") => format_version = Some(text),
                    Some("CreationTime") => created = Some(text),
                    _ => {}
                }
            }
            Event::Eof => break,
            _ => {}
        }
        buffer.clear();
    }
    let process = process
        .filter(|process| !process.is_empty())
        .ok_or("the model header names no process")?;
    Ok(MlDenoiseModel {
        process,
        format_version: format_version.unwrap_or_default(),
        created,
        mono,
        rgb,
        weights_sha256,
    })
}

fn local_name(name: &str) -> String {
    name.rsplit(':').next().unwrap_or(name).to_string()
}

/// An image MLDenoise processed in memory.
#[derive(Clone, Debug)]
pub struct MlDenoised {
    /// The denoised image, on the input's scale and with its geometry.
    pub image: LinearImage,
    /// The PixInsight version that ran, such as "1.9.6". Part of a cache
    /// key: another version can give another result.
    pub pixinsight_version: String,
    /// The model's header.
    pub model: MlDenoiseModel,
    /// Seconds PixInsight spent opening, denoising and saving the image.
    pub seconds: f64,
}

/// A file MLDenoise processed.
#[derive(Clone, Debug)]
pub struct MlDenoiseRun {
    /// The file written.
    pub output: PathBuf,
    /// The PixInsight version that ran.
    pub pixinsight_version: String,
    /// The model's header.
    pub model: MlDenoiseModel,
    /// Seconds PixInsight spent opening, denoising and saving the image.
    pub seconds: f64,
}

/// A handle on an installed PixInsight.
#[derive(Debug, Clone)]
pub struct PixInsight {
    executable: PathBuf,
}

impl PixInsight {
    /// Use PixInsight from its standard install location for this platform,
    /// or the first `PixInsight` executable on `PATH`.
    pub fn locate() -> Option<Self> {
        let mut candidates: Vec<PathBuf> = Vec::new();
        if cfg!(target_os = "macos") {
            candidates
                .push("/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight".into());
        } else if cfg!(windows) {
            for root in ["ProgramFiles", "ProgramW6432"] {
                if let Some(root) = std::env::var_os(root) {
                    candidates.push(PathBuf::from(root).join("PixInsight/bin/PixInsight.exe"));
                }
            }
        } else {
            candidates.push("/opt/PixInsight/bin/PixInsight.sh".into());
            candidates.push("/usr/local/PixInsight/bin/PixInsight.sh".into());
            if let Some(home) = std::env::var_os("HOME") {
                candidates.push(PathBuf::from(home).join("PixInsight/bin/PixInsight.sh"));
            }
        }
        if let Some(path) = std::env::var_os("PATH") {
            for directory in std::env::split_paths(&path) {
                for name in ["PixInsight", "PixInsight.sh", "PixInsight.exe"] {
                    candidates.push(directory.join(name));
                }
            }
        }
        candidates
            .into_iter()
            .find(|candidate| candidate.is_file())
            .map(Self::with_executable)
    }

    /// Use a specific PixInsight. `executable` is the program, or a folder
    /// that holds it: the install folder, or `PixInsight.app` on macOS.
    pub fn with_executable(executable: impl Into<PathBuf>) -> Self {
        let executable = executable.into();
        let executable = std::path::absolute(&executable).unwrap_or(executable);
        let executable = if executable.is_dir() {
            [
                "Contents/MacOS/PixInsight",
                "PixInsight.app/Contents/MacOS/PixInsight",
                "bin/PixInsight.sh",
                "bin/PixInsight.exe",
                "bin/PixInsight",
            ]
            .iter()
            .map(|below| executable.join(below))
            .find(|candidate| candidate.is_file())
            .unwrap_or(executable)
        } else {
            executable
        };
        Self { executable }
    }

    /// The executable this handle runs.
    pub fn executable(&self) -> &Path {
        &self.executable
    }

    /// Denoise an image with MLDenoise. `image` is linear, on any scale: it
    /// is mapped onto 0 to 1 for PixInsight, offset by its most negative
    /// sample and scaled by its span, and the result is mapped back. A
    /// one-channel image needs a model with a mono route, and a
    /// three-channel image one with an RGB route. `cancel` stops the run
    /// within a quarter of a second and closes PixInsight.
    pub fn ml_denoise_image(
        &self,
        image: &LinearImage,
        options: &MlDenoiseOptions,
        cancel: Option<&CancelSignal>,
    ) -> Result<MlDenoised> {
        options.validate()?;
        let model_path = std::path::absolute(&options.model)
            .map_err(|error| external_error(format!("model path: {error}")))?;
        let model = MlDenoiseModel::read(&model_path)?;
        if !model.accepts_channels(image.channels) {
            return Err(external_error(format!(
                "the model {} cannot denoise a {}-channel image",
                model_path.display(),
                image.channels
            )));
        }
        let (offset, scale) = exchange_scale(image)?;

        let workdir = tempfile::tempdir()
            .map_err(|error| external_error(format!("temporary folder: {error}")))?;
        let input = workdir.path().join("input.xisf");
        let output = workdir.path().join("denoised.xisf");
        let normalized: Vec<f32> = image
            .data
            .iter()
            .map(|sample| (sample - offset) / scale)
            .collect();
        let pixels = if image.channels == 3 {
            F32ImageData::RgbInterleaved(&normalized)
        } else {
            F32ImageData::Mono(&normalized)
        };
        seiza_image_io::save_f32(
            &input,
            image.width,
            image.height,
            pixels,
            &seiza_image_io::SaveOptions {
                format: Some(seiza_image_io::Format::Xisf),
                headers: &[],
                xisf_bounds: Some((0.0, 1.0)),
                fits_top_down: false,
            },
        )
        .map_err(|error| external_error(format!("could not write the exchange image: {error}")))?;

        let request = ScriptRequest {
            input: &input,
            output: &output,
            model: &model_path,
            amount: options.amount,
            overlap_percent: options.overlap_percent,
            precision: options.precision,
            local_support: options.local_support.map(|support| ScriptLocalSupport {
                clip_low: support.clip_low,
                background: support.background,
                smoothness: support.smoothness,
            }),
        };
        let outcome = self.run_script(workdir.path(), &request, cancel)?;

        let frame = FitsFrame::open(&output)?;
        if frame.image.width != image.width
            || frame.image.height != image.height
            || frame.image.channels != image.channels
        {
            return Err(external_error(format!(
                "PixInsight returned a {}x{}x{} image for a {}x{}x{} one",
                frame.image.width,
                frame.image.height,
                frame.image.channels,
                image.width,
                image.height,
                image.channels
            )));
        }
        let mut denoised = frame.image;
        for sample in &mut denoised.data {
            *sample = *sample * scale + offset;
        }
        Ok(MlDenoised {
            image: denoised,
            pixinsight_version: outcome.pixinsight_version,
            model,
            seconds: outcome.seconds,
        })
    }

    /// Denoise an image file with MLDenoise and write the result to
    /// `output`, keeping the input's headers. Both are FITS or XISF, by
    /// extension, and the output holds 32-bit floats on the input's scale.
    /// See [`Self::ml_denoise_image`].
    pub fn ml_denoise_file(
        &self,
        input: impl AsRef<Path>,
        output: impl AsRef<Path>,
        options: &MlDenoiseOptions,
        cancel: Option<&CancelSignal>,
    ) -> Result<MlDenoiseRun> {
        let output = output.as_ref();
        // A bad option is reported before the input is read.
        options.validate()?;
        let frame = FitsFrame::open(input.as_ref())?;
        let denoised = self.ml_denoise_image(&frame.image, options, cancel)?;
        write_processed_image_fits_f32(output, &denoised.image, &frame.headers, &[])?;
        Ok(MlDenoiseRun {
            output: output.to_path_buf(),
            pixinsight_version: denoised.pixinsight_version,
            model: denoised.model,
            seconds: denoised.seconds,
        })
    }

    /// Write the script and its request below `workdir`, run PixInsight on
    /// them, and read the result the script wrote.
    fn run_script(
        &self,
        workdir: &Path,
        request: &ScriptRequest<'_>,
        cancel: Option<&CancelSignal>,
    ) -> Result<ScriptOutcome> {
        let script = workdir.join("seiza-mldenoise.js");
        let request_path = workdir.join("request.json");
        let result_path = workdir.join("result.json");
        let log_path = workdir.join("pixinsight.log");
        // PixInsight separates a script's arguments with commas.
        for path in [&script, &request_path, &result_path] {
            if path.to_string_lossy().contains(',') {
                return Err(external_error(format!(
                    "the temporary folder {} contains a comma, which PixInsight's command line cannot carry",
                    workdir.display()
                )));
            }
        }
        let write = |path: &Path, contents: &[u8]| {
            std::fs::write(path, contents).map_err(|error| {
                external_error(format!("could not write {}: {error}", path.display()))
            })
        };
        write(&script, ML_DENOISE_SCRIPT.as_bytes())?;
        write(
            &request_path,
            &serde_json::to_vec(request)
                .map_err(|error| external_error(format!("request: {error}")))?,
        )?;
        let log = std::fs::File::create(&log_path).map_err(|error| {
            external_error(format!("could not create {}: {error}", log_path.display()))
        })?;
        let log_err = log
            .try_clone()
            .map_err(|error| external_error(format!("log: {error}")))?;

        let mut command = self.command(&format!(
            "-r={},{},{}",
            script.display(),
            request_path.display(),
            result_path.display()
        ))?;
        command
            .current_dir(workdir)
            .stdin(Stdio::null())
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(log_err));
        // Its own process group, so closing it reaches PixInsight itself
        // and not only a launcher (`PixInsight.sh`, `xvfb-run`).
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt as _;
            command.process_group(0);
        }
        let child = command.spawn().map_err(|error| {
            external_error(format!(
                "could not start {}: {error}",
                self.executable.display()
            ))
        })?;
        // Closes PixInsight on cancellation and on every early return, so
        // no instance keeps computing in the background.
        let mut instance = Instance(Some(child));
        loop {
            if let Some(cancel) = cancel
                && cancel.is_cancelled()
            {
                return Err(Error::Cancelled);
            }
            let child = instance
                .0
                .as_mut()
                .expect("the instance runs until it exits");
            match child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) => std::thread::sleep(POLL),
                Err(error) => return Err(external_error(format!("wait failed: {error}"))),
            }
        }
        instance.0 = None;

        let tail = || log_tail(&log_path);
        let text = std::fs::read_to_string(&result_path).map_err(|_| {
            external_error(format!(
                "PixInsight ended without running the script; its output ends: {}",
                tail()
            ))
        })?;
        let result: ScriptResult = serde_json::from_str(&text).map_err(|error| {
            external_error(format!("the script's result is not valid: {error}"))
        })?;
        if !result.ok {
            let mut message = result
                .error
                .unwrap_or_else(|| "MLDenoise failed".to_string());
            if let Some(log) = result
                .log
                .as_deref()
                .map(str::trim)
                .filter(|log| !log.is_empty())
            {
                message = format!("{message}: {}", last_lines(log, 8));
            }
            return Err(external_error(message));
        }
        Ok(ScriptOutcome {
            pixinsight_version: result.pixinsight_version.unwrap_or_default(),
            seconds: result.seconds.unwrap_or_default(),
        })
    }

    /// The command that runs PixInsight headless with `run` (its `-r=`
    /// argument). On Linux without a display, `xvfb-run` supplies one.
    fn command(&self, run: &str) -> Result<Command> {
        let arguments = ["-n", "--automation-mode", run, "--force-exit"];
        if cfg!(target_os = "linux")
            && std::env::var_os("DISPLAY").is_none()
            && std::env::var_os("WAYLAND_DISPLAY").is_none()
        {
            let xvfb_run = std::env::var_os("PATH").and_then(|path| {
                std::env::split_paths(&path)
                    .map(|directory| directory.join("xvfb-run"))
                    .find(|candidate| candidate.is_file())
            });
            let Some(xvfb_run) = xvfb_run else {
                return Err(external_error(
                    "PixInsight needs a display and there is none; install xvfb (xvfb-run) to give it a virtual one",
                ));
            };
            let mut command = Command::new(xvfb_run);
            command
                .args(["-a", "--server-args=-screen 0 1920x1200x24"])
                .arg(&self.executable)
                .args(arguments);
            return Ok(command);
        }
        let mut command = Command::new(&self.executable);
        command.args(arguments);
        Ok(command)
    }
}

/// A running PixInsight, closed when dropped: on Unix its whole process
/// group, since the program started may be a launcher whose children a kill
/// of the launcher alone leaves running.
struct Instance(Option<std::process::Child>);

impl Drop for Instance {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            #[cfg(unix)]
            if let Ok(group) = libc::pid_t::try_from(child.id()) {
                // SAFETY: kill only sends a signal; the group is the one
                // this child leads.
                unsafe {
                    libc::kill(-group, libc::SIGKILL);
                }
            }
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// The affine map onto 0 to 1: subtract `offset` (the most negative sample,
/// or 0) and divide by `scale` (the span, or 1 when it already fits).
fn exchange_scale(image: &LinearImage) -> Result<(f32, f32)> {
    let mut peak = f32::NEG_INFINITY;
    let mut floor = f32::INFINITY;
    for sample in &image.data {
        if sample.is_finite() {
            peak = peak.max(*sample);
            floor = floor.min(*sample);
        }
    }
    if !peak.is_finite() || !floor.is_finite() {
        return Err(external_error("the image has no finite samples to denoise"));
    }
    let offset = floor.min(0.0);
    let scale = if peak - offset > 1.0 {
        peak - offset
    } else {
        1.0
    };
    Ok((offset, scale))
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ScriptRequest<'a> {
    input: &'a Path,
    output: &'a Path,
    model: &'a Path,
    amount: f64,
    overlap_percent: u32,
    precision: MlDenoisePrecision,
    local_support: Option<ScriptLocalSupport>,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ScriptLocalSupport {
    clip_low: f64,
    background: f64,
    smoothness: u32,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScriptResult {
    ok: bool,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    log: Option<String>,
    #[serde(default)]
    pixinsight_version: Option<String>,
    #[serde(default)]
    seconds: Option<f64>,
}

struct ScriptOutcome {
    pixinsight_version: String,
    seconds: f64,
}

fn external_error(message: impl Into<String>) -> Error {
    Error::ExternalTool {
        tool: TOOL.to_string(),
        message: message.into(),
    }
}

/// The last lines of PixInsight's output, for an error message.
fn log_tail(path: &Path) -> String {
    let text = std::fs::read(path).unwrap_or_default();
    let text = String::from_utf8_lossy(&text);
    let tail = last_lines(text.trim(), 8);
    if tail.is_empty() {
        "nothing".to_string()
    } else {
        tail
    }
}

fn last_lines(text: &str, count: usize) -> String {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    lines[lines.len().saturating_sub(count)..].join(" / ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The start of a real MLDenoise model's header (MLDenoise_v42.xmlm).
    const HEADER: &str = r#"<?xml version="1.0" encoding="UTF-8"?><!--
PixInsight Machine Learning Model Format - xmlm version 1.0
--><xmlm version="1.0" xmlns="http://www.pixinsight.com/xmlm"><CreationTime>2026-10-05T22:31:37.600Z</CreationTime><CreatorOS>macOS</CreatorOS><Process>MLDenoise</Process><Version>V3</Version><Model role="mono" size="217277" blockSize="217280" sha256="613e">mono.onnx</Model><Model role="rgb" size="554180" blockSize="554192" sha256="1606">rgb.onnx</Model><Model role="weights" size="594864928" blockSize="594864944" sha256="7ca0">weights.bin</Model></xmlm>"#;

    fn model_file(directory: &Path, header: &str) -> PathBuf {
        let path = directory.join("model.xmlm");
        let mut bytes = b"XMLM0100".to_vec();
        bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&[0; 4]);
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[7; 64]);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn a_model_header_gives_its_process_routes_and_weights() {
        let directory = tempfile::tempdir().unwrap();
        let model = MlDenoiseModel::read(model_file(directory.path(), HEADER)).unwrap();
        assert_eq!(
            model,
            MlDenoiseModel {
                process: "MLDenoise".into(),
                format_version: "V3".into(),
                created: Some("2026-10-05T22:31:37.600Z".into()),
                mono: true,
                rgb: true,
                weights_sha256: Some("7ca0".into()),
            }
        );
        assert!(model.accepts_channels(1) && model.accepts_channels(3));
        assert!(!model.accepts_channels(2));

        let mono_only = HEADER.replace(r#"<Model role="rgb""#, r#"<Model role="other""#);
        let model = MlDenoiseModel::read(model_file(directory.path(), &mono_only)).unwrap();
        assert!(model.accepts_channels(1) && !model.accepts_channels(3));
    }

    #[test]
    fn a_file_that_is_not_an_mldenoise_model_is_refused() {
        let directory = tempfile::tempdir().unwrap();
        let other = HEADER.replace("<Process>MLDenoise", "<Process>MLStarExtraction");
        let error = MlDenoiseModel::read(model_file(directory.path(), &other)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("MLStarExtraction, not MLDenoise"),
            "{error}"
        );

        let fits = directory.path().join("image.fits");
        std::fs::write(&fits, b"SIMPLE  =                    T").unwrap();
        let error = MlDenoiseModel::read(&fits).unwrap_err();
        assert!(
            error.to_string().contains("not a PixInsight model file"),
            "{error}"
        );

        let error = MlDenoiseModel::read(directory.path().join("missing.xmlm")).unwrap_err();
        assert!(error.to_string().contains("cannot open"), "{error}");
    }

    #[test]
    fn options_outside_pixinsights_ranges_are_refused() {
        let base = MlDenoiseOptions::new("model.xmlm");
        base.validate().unwrap();
        assert_eq!((base.amount, base.overlap_percent), (0.9, 25));
        let refused = |change: fn(&mut MlDenoiseOptions), field: &str| {
            let mut options = base.clone();
            change(&mut options);
            let error = options.validate().unwrap_err().to_string();
            assert!(error.contains(field), "{error}");
        };
        refused(|o| o.amount = 1.5, "amount");
        refused(|o| o.amount = f64::NAN, "amount");
        refused(|o| o.overlap_percent = 11, "overlap_percent");
        refused(|o| o.overlap_percent = 51, "overlap_percent");
        refused(
            |o| {
                o.local_support = Some(MlDenoiseLocalSupport {
                    clip_low: 21.0,
                    ..Default::default()
                })
            },
            "clip_low",
        );
        refused(
            |o| {
                o.local_support = Some(MlDenoiseLocalSupport {
                    background: -0.1,
                    ..Default::default()
                })
            },
            "background",
        );
        refused(
            |o| {
                o.local_support = Some(MlDenoiseLocalSupport {
                    smoothness: 8,
                    ..Default::default()
                })
            },
            "smoothness",
        );
        for precision in [MlDenoisePrecision::Fast, MlDenoisePrecision::High] {
            assert_eq!(
                precision.to_string().parse::<MlDenoisePrecision>().unwrap(),
                precision
            );
        }
        assert!("full".parse::<MlDenoisePrecision>().is_err());
        // Options read from JSON take PixInsight's defaults for what they omit.
        let read: MlDenoiseOptions =
            serde_json::from_str(r#"{"model": "m.xmlm", "local_support": {"smoothness": 4}}"#)
                .unwrap();
        assert_eq!(read.amount, 0.9);
        assert_eq!(read.local_support.unwrap().background, 0.25);
    }

    #[test]
    fn the_exchange_scale_maps_any_linear_image_onto_zero_to_one() {
        let image = |data: Vec<f32>| LinearImage::new(2, 2, 1, data).unwrap();
        assert_eq!(
            exchange_scale(&image(vec![0.0, 0.1, 0.5, 1.0])).unwrap(),
            (0.0, 1.0)
        );
        assert_eq!(
            exchange_scale(&image(vec![-100.0, 0.0, 900.0, f32::NAN])).unwrap(),
            (-100.0, 1000.0)
        );
        assert!(exchange_scale(&image(vec![f32::NAN; 4])).is_err());
    }

    /// A stand-in PixInsight: reads `-r=script,request,result` as the real
    /// one runs a script, then does what `body` says with the request.
    #[cfg(unix)]
    fn fake_pixinsight(directory: &Path, body: &str) -> PixInsight {
        use std::os::unix::fs::PermissionsExt;
        let path = directory.join("PixInsight");
        let script = format!(
            "#!/usr/bin/env python3\n\
             import json, shutil, sys, time\n\
             run = next(a for a in sys.argv if a.startswith('-r='))[3:].split(',')\n\
             assert '--automation-mode' in sys.argv and '--force-exit' in sys.argv\n\
             request = json.load(open(run[1]))\n\
             result_path = run[2]\n\
             def finish(result):\n    json.dump(result, open(result_path, 'w'))\n\
             {body}\n"
        );
        std::fs::write(&path, script).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        PixInsight::with_executable(path)
    }

    #[cfg(unix)]
    fn adu_image() -> LinearImage {
        // ADU with a negative background sample, as a background-subtracted
        // stack holds.
        LinearImage::new(3, 2, 1, vec![-12.0, 0.0, 250.5, 1000.0, 40_000.0, 65_000.0]).unwrap()
    }

    #[cfg(unix)]
    #[test]
    fn an_image_returns_on_its_own_scale_with_the_run_s_details() {
        let directory = tempfile::tempdir().unwrap();
        let model = model_file(directory.path(), HEADER);
        // Returning the exchange image unchanged must give back the input.
        let pixinsight = fake_pixinsight(
            directory.path(),
            "assert request['amount'] == 0.5 and request['overlapPercent'] == 30\n\
             assert request['precision'] == 'high'\n\
             assert request['localSupport'] == {'clipLow': -1.0, 'background': 0.3, 'smoothness': 3}\n\
             assert request['model'].endswith('model.xmlm')\n\
             shutil.copy(request['input'], request['output'])\n\
             finish({'ok': True, 'pixinsightVersion': '1.9.6', 'seconds': 2.5})",
        );
        let options = MlDenoiseOptions {
            amount: 0.5,
            overlap_percent: 30,
            precision: MlDenoisePrecision::High,
            local_support: Some(MlDenoiseLocalSupport {
                clip_low: -1.0,
                background: 0.3,
                smoothness: 3,
            }),
            ..MlDenoiseOptions::new(&model)
        };
        let image = adu_image();
        let denoised = pixinsight.ml_denoise_image(&image, &options, None).unwrap();
        assert_eq!(denoised.pixinsight_version, "1.9.6");
        assert_eq!(denoised.seconds, 2.5);
        assert_eq!(denoised.model.weights_sha256.as_deref(), Some("7ca0"));
        for (got, want) in denoised.image.data.iter().zip(&image.data) {
            assert!(
                (got - want).abs() <= want.abs() * 1e-6 + 1e-3,
                "{got} != {want}"
            );
        }

        // The file call keeps the input's headers on the output.
        let input = directory.path().join("light.fits");
        crate::write_processed_image_fits_f32(
            &input,
            &image,
            &[(
                "OBJECT".into(),
                seiza_fits::HeaderValue::String("M31".into()),
            )],
            &[],
        )
        .unwrap();
        let output = directory.path().join("denoised.fits");
        let run = pixinsight
            .ml_denoise_file(&input, &output, &options, None)
            .unwrap();
        assert_eq!(run.output, output);
        let written = FitsFrame::open(&output).unwrap();
        assert!(
            written
                .headers
                .iter()
                .any(|(key, value)| key == "OBJECT" && value.as_str() == Some("M31"))
        );
        assert_eq!(written.image.width, 3);
    }

    #[cfg(unix)]
    #[test]
    fn a_failed_run_explains_what_pixinsight_said() {
        let directory = tempfile::tempdir().unwrap();
        let model = model_file(directory.path(), HEADER);
        let failing = fake_pixinsight(
            directory.path(),
            "finish({'ok': False, 'error': 'Error: MLDenoise did not run', \
                     'log': 'Loading model\\n*** Failed to load the selected model.'})",
        );
        let error = failing
            .ml_denoise_image(&adu_image(), &MlDenoiseOptions::new(&model), None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("MLDenoise did not run"), "{error}");
        assert!(
            error.contains("Failed to load the selected model"),
            "{error}"
        );

        let silent = fake_pixinsight(directory.path(), "print('License has expired')");
        let error = silent
            .ml_denoise_image(&adu_image(), &MlDenoiseOptions::new(&model), None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("without running the script"), "{error}");
        assert!(error.contains("License has expired"), "{error}");

        // An RGB image needs an RGB route; this model has none.
        let mono_only = HEADER.replace(r#"<Model role="rgb""#, r#"<Model role="other""#);
        let model = model_file(directory.path(), &mono_only);
        let rgb = LinearImage::new(1, 1, 3, vec![0.1, 0.2, 0.3]).unwrap();
        let error = failing
            .ml_denoise_image(&rgb, &MlDenoiseOptions::new(&model), None)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("cannot denoise a 3-channel image"),
            "{error}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_cancelled_run_closes_pixinsight() {
        let directory = tempfile::tempdir().unwrap();
        let model = model_file(directory.path(), HEADER);
        let marker = directory.path().join("still-running");
        let hanging = fake_pixinsight(
            directory.path(),
            &format!(
                "time.sleep(2)\nopen({:?}, 'w').write('yes')",
                marker.display().to_string()
            ),
        );
        let started = std::time::Instant::now();
        let cancel = CancelSignal::new(move || started.elapsed() > Duration::from_millis(300));
        let error = hanging
            .ml_denoise_image(&adu_image(), &MlDenoiseOptions::new(&model), Some(&cancel))
            .unwrap_err();
        assert!(matches!(error, Error::Cancelled), "{error}");
        assert!(started.elapsed() < Duration::from_secs(2));
        std::thread::sleep(Duration::from_secs(3));
        assert!(!marker.exists(), "PixInsight kept running after the cancel");
    }

    /// Runs the installed PixInsight with a real model. Set
    /// `SEIZA_MLDENOISE_MODEL` to an `.xmlm` file to run it.
    #[test]
    #[ignore = "needs PixInsight and an MLDenoise model"]
    fn real_pixinsight_denoises_an_image() {
        let model = std::env::var_os("SEIZA_MLDENOISE_MODEL").expect("SEIZA_MLDENOISE_MODEL");
        let pixinsight = PixInsight::locate().expect("PixInsight is installed");
        // A noisy background with one star, in ADU.
        let (width, height) = (256, 256);
        let mut state = 0x2545_f491_u32;
        let mut noise = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32 - 0.5) * 200.0
        };
        let data: Vec<f32> = (0..width * height)
            .map(|index| {
                let (x, y) = ((index % width) as f32, (index / width) as f32);
                let star = 30_000.0 * (-((x - 128.0).powi(2) + (y - 128.0).powi(2)) / 8.0).exp();
                1_000.0 + star + noise()
            })
            .collect();
        let image = LinearImage::new(width, height, 1, data).unwrap();
        let options = MlDenoiseOptions {
            amount: 1.0,
            ..MlDenoiseOptions::new(PathBuf::from(model))
        };
        let denoised = pixinsight.ml_denoise_image(&image, &options, None).unwrap();
        let spread = |image: &LinearImage| {
            let corner: Vec<f32> = (0..64)
                .flat_map(|y| (0..64).map(move |x| y * width + x))
                .map(|index| image.data[index])
                .collect();
            let mean = corner.iter().sum::<f32>() / corner.len() as f32;
            (corner.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / corner.len() as f32).sqrt()
        };
        eprintln!(
            "PixInsight {} in {:.1} s: background spread {:.1} -> {:.1} ADU, star peak {:.0} -> {:.0}",
            denoised.pixinsight_version,
            denoised.seconds,
            spread(&image),
            spread(&denoised.image),
            image.data[128 * width + 128],
            denoised.image.data[128 * width + 128]
        );
        assert!(spread(&denoised.image) < spread(&image) * 0.8);
        assert!(
            (denoised.image.data[0] - 1_000.0).abs() < 200.0,
            "kept the image's scale"
        );
    }
}
