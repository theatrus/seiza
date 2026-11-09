//! Running PixInsight processes on images, starting with MLDenoise.
//!
//! PixInsight is a desktop program that runs scripts from its command line:
//! `-n` starts a new instance, `--automation-mode` runs a script without
//! stopping at most dialogs, `-r=` gives the script's path and its
//! arguments, and `--force-exit` closes the instance when the script ends.
//! It needs a display; on Linux without one, `xvfb-run` supplies a virtual
//! one.
//!
//! MLDenoise is the neural-network noise reduction of PixInsight's
//! MachineLearning module. It reads its network from a model file
//! (`.xmlm`), which PixInsight distributes separately.
//! [`PixInsight::ml_denoise_image`] writes the image to a temporary XISF
//! file on a 0 to 1 scale, runs a script that applies MLDenoise and saves
//! the result, and reads the result back on the image's own scale. The
//! script writes its outcome to a JSON file; PixInsight's own output is
//! kept only for an error message.
//!
//! A run that is cancelled, or that passes its time limit, closes
//! PixInsight. A host that exits during a run does not: PixInsight finishes
//! on its own and leaves its temporary folder behind.
//!
//! On macOS, the ONNX Runtime MLDenoise runs in keeps Core ML models in the
//! per-user temporary folder, and a PixInsight closed by `--force-exit` or
//! by a kill leaves them there. They are removed once each instance has
//! exited.

use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use seiza_fits::F32ImageData;
use sha2::{Digest, Sha256};

use crate::CancelSignal;
use crate::exchange::{CAMERA_FULL_SCALE, Exchange};
use crate::fits::{FitsFrame, write_processed_image_with_bounds};
use crate::image::LinearImage;
use crate::{Error, Result};

const TOOL: &str = "PixInsight MLDenoise";

/// The longest XMLM header read. Real headers are under a kilobyte; the
/// limit stops a damaged length field from allocating gigabytes.
const MAX_XMLM_HEADER: usize = 1 << 20;

/// How often a run checks for cancellation and for PixInsight exiting.
const POLL: Duration = Duration::from_millis(250);

/// How often PixInsight is started again when another instance took the
/// same instance slot at the same moment.
const START_ATTEMPTS: u32 = 3;

/// What PixInsight prints when another instance took the slot it chose.
const SLOT_TAKEN: &str = "Unable to create application instance";

/// Instances started by this process start one at a time: two PixInsight
/// instances starting at once can choose the same instance slot, and one of
/// them then fails. The lock is held until the script has started.
static STARTUP: Mutex<()> = Mutex::new(());

/// The script PixInsight runs. Its arguments are the request file, the
/// result file, and a file it writes as soon as it starts; the request
/// gives the image paths and MLDenoise's parameters.
const ML_DENOISE_SCRIPT: &str = r#"#feature-id Seiza > MLDenoise
var requestPath = jsArguments[0];
var resultPath = jsArguments[1];
File.writeTextFile(jsArguments[2], "started");
var result = { ok: false };
try {
   var core = CoreApplication;
   result.pixinsightVersion = core.versionMajor + "." + core.versionMinor + "." + core.versionRelease;
   if (typeof core.versionRevision == "number" && core.versionRevision > 0)
      result.pixinsightVersion += "-" + core.versionRevision;
   if (typeof core.versionBuild == "number")
      result.pixinsightVersion += " build " + core.versionBuild;
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
            _ => Err(refused(format!(
                "the precision must be \"fast\" or \"high\", not {text:?}"
            ))),
        }
    }
}

/// MLDenoise's local support: a mask made from the image's brightness that
/// protects bright structures. Dark mask pixels block noise reduction and
/// bright ones allow all of it.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
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
#[serde(rename_all = "camelCase")]
pub struct MlDenoiseOptions {
    /// The model file (`.xmlm`).
    pub model: PathBuf,
    /// The weight of the denoised image in each output pixel, from 0 to 1:
    /// the result is `(1 - amount) * original + amount * denoised`.
    /// Default 0.9.
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
    /// The longest a run may take, in seconds, before PixInsight is closed
    /// and the run fails. Default none: ten minutes plus 20 seconds per
    /// million samples, which is far longer than a run takes on a CPU.
    /// PixInsight shows license dialogs even in automation mode, and a run
    /// waiting on one would otherwise never end.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_seconds: Option<f64>,
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
            timeout_seconds: None,
        }
    }

    /// Refuse values PixInsight would refuse, as
    /// [`Error::ExternalToolRefused`].
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.amount) {
            return Err(refused(format!(
                "the amount must be from 0 to 1, not {}",
                self.amount
            )));
        }
        if !(12..=50).contains(&self.overlap_percent) {
            return Err(refused(format!(
                "the tile overlap must be from 12 to 50 percent, not {}",
                self.overlap_percent
            )));
        }
        if let Some(support) = &self.local_support {
            if !(-20.0..=20.0).contains(&support.clip_low) {
                return Err(refused(format!(
                    "the mask's clipping point must be from -20 to 20 sigma, not {}",
                    support.clip_low
                )));
            }
            if !(0.0..=1.0).contains(&support.background) {
                return Err(refused(format!(
                    "the mask's background level must be from 0 to 1, not {}",
                    support.background
                )));
            }
            if support.smoothness > 7 {
                return Err(refused(format!(
                    "the mask's smoothness must be from 0 to 7 pixels, not {}",
                    support.smoothness
                )));
            }
        }
        if let Some(timeout) = self.timeout_seconds
            && !(timeout.is_finite() && timeout > 0.0)
        {
            return Err(refused(format!(
                "the timeout must be a positive number of seconds, not {timeout}"
            )));
        }
        Ok(())
    }

    /// The time limit for a run on an image of `samples` samples.
    fn time_limit(&self, samples: usize) -> Duration {
        let seconds = self
            .timeout_seconds
            .unwrap_or(600.0 + 20.0 * samples as f64 / 1e6);
        Duration::from_secs_f64(seconds)
    }
}

/// What an MLDenoise model file declares in its header.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MlDenoiseModel {
    /// The PixInsight process the model is for: `MLDenoise`.
    pub process: String,
    /// The model format version, such as `V3`.
    pub format_version: String,
    /// When the model was created, as the header gives it.
    pub created: Option<String>,
    /// Whether the model can denoise a one-channel image.
    pub mono: bool,
    /// Whether the model can denoise an RGB image.
    pub rgb: bool,
    /// The SHA-256 of the model's weights.
    pub weights_sha256: Option<String>,
    /// The SHA-256 of the whole header, which records the SHA-256 of every
    /// payload: the weights and each route's network. It identifies the
    /// model. A host caching results keys on it with the PixInsight
    /// version and the options; two releases of a model can share their
    /// weights and still give different results.
    pub header_sha256: String,
}

impl MlDenoiseModel {
    /// Read the header of an `.xmlm` file, without reading its weights.
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let refuse = |message: String| refused(format!("{}: {message}", path.display()));
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
        .ok_or("the model header gives no process")?;
    Ok(MlDenoiseModel {
        process,
        format_version: format_version.unwrap_or_default(),
        created,
        mono,
        rgb,
        weights_sha256,
        header_sha256: hex(&Sha256::digest(xml)),
    })
}

fn local_name(name: &str) -> String {
    name.rsplit(':').next().unwrap_or(name).to_string()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// An image MLDenoise processed in memory.
#[derive(Clone, Debug)]
pub struct MlDenoised {
    /// The denoised image, on the input's scale and with its geometry.
    /// Samples that were not finite in the input are returned unchanged.
    pub image: LinearImage,
    /// The PixInsight version and build that ran, such as "1.9.6 build
    /// 1702". Part of a cache key: another build can give another result.
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
    /// The PixInsight version and build that ran.
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

    /// Denoise a linear image with MLDenoise.
    ///
    /// PixInsight works on 0 to 1. An image whose samples lie within 0 to 1
    /// keeps its scale. Any other image is offset by its most negative
    /// sample, if it has one, and divided by its span or by 65535,
    /// whichever is larger, which is how PixInsight opens a 16-bit camera
    /// frame; the result is mapped back. Samples that are not finite, such
    /// as the uncovered edges of a stack, are given the channel's median for
    /// PixInsight and returned unchanged, and so are samples more than 30
    /// robust sigmas below it, such as a stack's resampling ringing beside a
    /// saturated star, which take no part in the offset.
    ///
    /// A one-channel image needs a model with a mono route, and a
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
            .map_err(|error| refused(format!("model path: {error}")))?;
        let model = MlDenoiseModel::read(&model_path)?;
        if !model.accepts_channels(image.channels) {
            return Err(refused(format!(
                "the model {} cannot denoise a {}-channel image",
                model_path.display(),
                image.channels
            )));
        }
        let exchange = Exchange::of(image)
            .ok_or_else(|| refused("the image has no finite samples to denoise"))?;

        let workdir = tempfile::tempdir()
            .map_err(|error| external_error(format!("temporary folder: {error}")))?;
        let input = workdir.path().join("input.xisf");
        let output = workdir.path().join("denoised.xisf");
        let normalized = exchange.normalize(image);
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
        drop(normalized);

        let request = ScriptRequest {
            input: &input,
            output: &output,
            model: &model_path,
            amount: options.amount,
            overlap_percent: options.overlap_percent,
            precision: options.precision,
            local_support: options.local_support,
        };
        let time_limit = options.time_limit(image.data.len());
        let outcome = self.run_script(workdir.path(), &request, time_limit, cancel)?;

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
        // The map back assumes PixInsight saved on the 0 to 1 it was given.
        if let Some((low, high)) = frame.bounds
            && (low, high) != (0.0, 1.0)
        {
            return Err(external_error(format!(
                "PixInsight saved the result on {low} to {high}, not 0 to 1"
            )));
        }
        let mut denoised = frame.image;
        exchange.restore(&mut denoised.data, &image.data);
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
    /// An XISF output declares the range the input declared, or 0 to 65535
    /// when the input declared none. See [`Self::ml_denoise_image`].
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
        write_processed_image_with_bounds(
            output,
            &denoised.image,
            &frame.headers,
            &[],
            frame.bounds.unwrap_or((0.0, CAMERA_FULL_SCALE as f64)),
        )?;
        Ok(MlDenoiseRun {
            output: output.to_path_buf(),
            pixinsight_version: denoised.pixinsight_version,
            model: denoised.model,
            seconds: denoised.seconds,
        })
    }

    /// Write the script and its request below `workdir`, run PixInsight on
    /// them, and read the result the script wrote. When another instance
    /// took the same instance slot, PixInsight is started again.
    fn run_script(
        &self,
        workdir: &Path,
        request: &ScriptRequest<'_>,
        time_limit: Duration,
        cancel: Option<&CancelSignal>,
    ) -> Result<ScriptOutcome> {
        let paths = ScriptPaths::below(workdir);
        self.check_arguments(workdir, &paths)?;
        let write = |path: &Path, contents: &[u8]| {
            std::fs::write(path, contents).map_err(|error| {
                external_error(format!("could not write {}: {error}", path.display()))
            })
        };
        write(&paths.script, ML_DENOISE_SCRIPT.as_bytes())?;
        write(
            &paths.request,
            &serde_json::to_vec(request)
                .map_err(|error| external_error(format!("request: {error}")))?,
        )?;
        let deadline = Instant::now() + time_limit;
        let mut attempt = 1;
        loop {
            let status = self.run_once(&paths, deadline, time_limit, cancel)?;
            if let Ok(text) = std::fs::read_to_string(&paths.result) {
                return script_outcome(&text);
            }
            let log = std::fs::read(&paths.log).unwrap_or_default();
            let log = String::from_utf8_lossy(&log);
            if log.contains(SLOT_TAKEN) && attempt < START_ATTEMPTS {
                // Another instance took the slot at the same moment; it may
                // belong to another process, which the startup lock does not
                // reach. Wait a little, differently each time, and start
                // again.
                let jitter = u64::from(std::process::id() % 7) * 100;
                std::thread::sleep(Duration::from_millis(500 * u64::from(attempt) + jitter));
                attempt += 1;
                continue;
            }
            return Err(external_error(format!(
                "PixInsight {status} before the script wrote its result; its output ends: {}",
                last_lines(log.trim(), 8).if_empty("nothing")
            )));
        }
    }

    /// Run PixInsight once on the script, until it exits. Returns how it
    /// exited.
    fn run_once(
        &self,
        paths: &ScriptPaths,
        deadline: Instant,
        time_limit: Duration,
        cancel: Option<&CancelSignal>,
    ) -> Result<String> {
        let log = std::fs::File::create(&paths.log).map_err(|error| {
            external_error(format!("could not create {}: {error}", paths.log.display()))
        })?;
        let log_err = log
            .try_clone()
            .map_err(|error| external_error(format!("log: {error}")))?;
        let mut command = self.command(&format!(
            "-r={},{},{},{}",
            paths.script.display(),
            paths.request.display(),
            paths.result.display(),
            paths.started.display()
        ))?;
        command
            .current_dir(paths.script.parent().unwrap_or(Path::new(".")))
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
        let _ = std::fs::remove_file(&paths.started);
        let mut startup = Some(
            STARTUP
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        );
        let started_at = Instant::now();
        let child = command.spawn().map_err(|error| {
            external_error(format!(
                "could not start {}: {error}",
                self.executable.display()
            ))
        })?;
        // Closes PixInsight on cancellation, at the time limit, and on every
        // early return, so no instance keeps computing in the background.
        let mut instance = Instance::new(child);
        loop {
            // The next instance may start once this one runs the script, or
            // after a minute in which it has not.
            if startup.is_some()
                && (paths.started.exists() || started_at.elapsed() > Duration::from_secs(60))
            {
                startup = None;
            }
            if let Some(cancel) = cancel
                && cancel.is_cancelled()
            {
                return Err(Error::Cancelled);
            }
            if Instant::now() > deadline {
                return Err(external_error(format!(
                    "PixInsight did not finish within {:.0} seconds and was closed. \
                     PixInsight shows license dialogs even in automation mode, which \
                     nobody can answer on a virtual display; check its license by \
                     starting it on this computer. A larger time limit helps a very \
                     large image.",
                    time_limit.as_secs_f64()
                )));
            }
            let child = instance
                .child
                .as_mut()
                .expect("the instance runs until it exits");
            match child.try_wait() {
                Ok(Some(status)) => {
                    instance.child = None;
                    return Ok(describe_exit(status));
                }
                Ok(None) => std::thread::sleep(POLL),
                Err(error) => return Err(external_error(format!("wait failed: {error}"))),
            }
        }
    }

    /// Refuse a script folder PixInsight's command line cannot carry.
    fn check_arguments(&self, workdir: &Path, paths: &ScriptPaths) -> Result<()> {
        // PixInsight splits a script's arguments at commas.
        for path in paths.arguments() {
            if path.to_string_lossy().contains(',') {
                return Err(refused(format!(
                    "the temporary folder {} contains a comma, at which PixInsight \
                     splits script arguments; set TMPDIR to a folder without one",
                    workdir.display()
                )));
            }
        }
        // The Linux launcher passes its arguments through the shell again.
        let launcher = self
            .executable
            .extension()
            .is_some_and(|extension| extension == "sh");
        if launcher {
            let plain =
                |character: char| character.is_ascii_alphanumeric() || "/._-+".contains(character);
            for path in paths.arguments() {
                if !path.to_string_lossy().chars().all(plain) {
                    return Err(refused(format!(
                        "the temporary folder {} contains characters that PixInsight.sh \
                         would pass to the shell; set TMPDIR to a folder whose path has \
                         only letters, digits and / . _ - +",
                        workdir.display()
                    )));
                }
            }
        }
        Ok(())
    }

    /// The command that runs PixInsight headless with `run` (its `-r=`
    /// argument). On Linux without a display, `xvfb-run` supplies one.
    fn command(&self, run: &str) -> Result<Command> {
        let arguments = [
            "-n",
            "--automation-mode",
            "--no-focus-stealing",
            run,
            "--force-exit",
        ];
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

/// The files of one run, all in its temporary folder.
struct ScriptPaths {
    script: PathBuf,
    request: PathBuf,
    result: PathBuf,
    started: PathBuf,
    log: PathBuf,
}

impl ScriptPaths {
    fn below(workdir: &Path) -> Self {
        Self {
            script: workdir.join("seiza-mldenoise.js"),
            request: workdir.join("request.json"),
            result: workdir.join("result.json"),
            started: workdir.join("started"),
            log: workdir.join("pixinsight.log"),
        }
    }

    /// The paths PixInsight's command line carries.
    fn arguments(&self) -> [&Path; 4] {
        [&self.script, &self.request, &self.result, &self.started]
    }
}

fn describe_exit(status: ExitStatus) -> String {
    match status.code() {
        Some(code) => format!("exited with code {code}"),
        None => "was stopped by a signal".to_string(),
    }
}

/// A running PixInsight, closed when dropped: on Unix its whole process
/// group, since the program started may be a launcher whose children a kill
/// of the launcher alone leaves running. Dropping it also removes the Core
/// ML models it left, however it ended; see [`remove_core_ml_leftovers`].
struct Instance {
    /// The program started, until it has exited and been waited for.
    child: Option<std::process::Child>,
    /// The program's process id, which names its Core ML models.
    pid: u32,
}

impl Instance {
    fn new(child: std::process::Child) -> Self {
        Self {
            pid: child.id(),
            child: Some(child),
        }
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
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
        remove_core_ml_leftovers(self.pid);
    }
}

/// Remove the Core ML models that the PixInsight which ran as `pid`, and
/// has exited, left in the per-user temporary folder.
///
/// MLDenoise runs its network in ONNX Runtime, which on macOS converts the
/// model of each session to a Core ML package and compiles it, both in that
/// folder, and removes them when the session closes. `--force-exit` and a
/// kill end PixInsight with its sessions open, so each run left them: 1.6
/// GB, in 60 sessions, for one 512 by 512 image. The folder is the one
/// `confstr` gives: ONNX Runtime writes there whatever TMPDIR says, and
/// [`std::env::temp_dir`] follows TMPDIR. A removal that fails costs disk
/// space, not the run, and is passed over.
///
/// Only macOS has Core ML; on Linux, the program started may also be
/// `xvfb-run`, whose pid is not PixInsight's.
#[cfg(target_os = "macos")]
fn remove_core_ml_leftovers(pid: u32) {
    if let Some(folder) = darwin_user_temp_dir() {
        remove_core_ml_leftovers_in(&folder, pid);
    }
}

#[cfg(not(target_os = "macos"))]
fn remove_core_ml_leftovers(_pid: u32) {}

/// The per-user temporary folder, `confstr(_CS_DARWIN_USER_TEMP_DIR)`.
#[cfg(target_os = "macos")]
fn darwin_user_temp_dir() -> Option<PathBuf> {
    use std::ffi::{CStr, OsStr};
    use std::os::unix::ffi::OsStrExt as _;
    let mut buffer = vec![0u8; libc::PATH_MAX as usize];
    // SAFETY: confstr writes at most `buffer.len()` bytes, ending with a
    // NUL, and returns the length the whole value needs, or 0.
    let needed = unsafe {
        libc::confstr(
            libc::_CS_DARWIN_USER_TEMP_DIR,
            buffer.as_mut_ptr().cast(),
            buffer.len(),
        )
    };
    if needed == 0 || needed > buffer.len() {
        return None;
    }
    let folder = CStr::from_bytes_until_nul(&buffer).ok()?;
    Some(PathBuf::from(OsStr::from_bytes(folder.to_bytes())))
}

/// Remove the entries in `folder` that [`core_ml_leftover_pid`] gives `pid`
/// for: folders with what they hold, and anything else, a link included,
/// by itself.
#[cfg(any(target_os = "macos", test))]
fn remove_core_ml_leftovers_in(folder: &Path, pid: u32) {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return;
    };
    for entry in entries.flatten() {
        if entry.file_name().to_str().and_then(core_ml_leftover_pid) != Some(pid) {
            continue;
        }
        let path = entry.path();
        let _ = match entry.file_type() {
            Ok(kind) if kind.is_dir() => std::fs::remove_dir_all(&path),
            _ => std::fs::remove_file(&path),
        };
    }
}

/// The process id in the name of an entry ONNX Runtime's Core ML provider
/// made: `onnxruntime-` and the process's `globallyUniqueString`,
/// `<UUID>-<pid>-<16 hex digits>`, with `.mlmodelc` added for a compiled
/// model. `None` for any other name.
#[cfg(any(target_os = "macos", test))]
fn core_ml_leftover_pid(name: &str) -> Option<u32> {
    let unique = name.strip_prefix("onnxruntime-")?;
    let unique = unique.strip_suffix(".mlmodelc").unwrap_or(unique);
    let mut fields = unique.rsplitn(3, '-');
    let (serial, pid, uuid) = (fields.next()?, fields.next()?, fields.next()?);
    let hex = |field: &str| field.bytes().all(|byte| byte.is_ascii_hexdigit());
    let uuid_shaped =
        uuid.split('-').map(str::len).eq([8, 4, 4, 4, 12]) && uuid.split('-').all(hex);
    let serial_shaped = serial.len() == 16 && hex(serial);
    // Digits only: `parse` would also take a sign.
    let decimal = !pid.is_empty() && pid.bytes().all(|byte| byte.is_ascii_digit());
    if !(uuid_shaped && serial_shaped && decimal) {
        return None;
    }
    pid.parse().ok()
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
    local_support: Option<MlDenoiseLocalSupport>,
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

fn script_outcome(text: &str) -> Result<ScriptOutcome> {
    let result: ScriptResult = serde_json::from_str(text)
        .map_err(|error| external_error(format!("the script's result is not valid: {error}")))?;
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

fn external_error(message: impl Into<String>) -> Error {
    Error::ExternalTool {
        tool: TOOL.to_string(),
        message: message.into(),
    }
}

fn refused(message: impl Into<String>) -> Error {
    Error::ExternalToolRefused {
        tool: TOOL.to_string(),
        message: message.into(),
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

trait IfEmpty {
    fn if_empty(self, fallback: &str) -> String;
}

impl IfEmpty for String {
    fn if_empty(self, fallback: &str) -> String {
        if self.is_empty() {
            fallback.to_string()
        } else {
            self
        }
    }
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
    fn a_model_header_gives_its_process_routes_and_identity() {
        let directory = tempfile::tempdir().unwrap();
        let model = MlDenoiseModel::read(model_file(directory.path(), HEADER)).unwrap();
        assert_eq!(model.process, "MLDenoise");
        assert_eq!(model.format_version, "V3");
        assert_eq!(model.created.as_deref(), Some("2026-10-05T22:31:37.600Z"));
        assert!(model.mono && model.rgb);
        assert_eq!(model.weights_sha256.as_deref(), Some("7ca0"));
        assert_eq!(model.header_sha256.len(), 64);
        assert!(model.accepts_channels(1) && model.accepts_channels(3));
        assert!(!model.accepts_channels(2));

        // Two releases can share their weights and differ in a route; the
        // header's hash tells them apart.
        let other_route = HEADER.replace(r#"sha256="613e""#, r#"sha256="8bff""#);
        let other = MlDenoiseModel::read(model_file(directory.path(), &other_route)).unwrap();
        assert_eq!(other.weights_sha256, model.weights_sha256);
        assert_ne!(other.header_sha256, model.header_sha256);

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
            matches!(error, Error::ExternalToolRefused { .. }),
            "{error}"
        );
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
        let refused = |change: fn(&mut MlDenoiseOptions), words: &str| {
            let mut options = base.clone();
            change(&mut options);
            let error = options.validate().unwrap_err();
            assert!(
                matches!(error, Error::ExternalToolRefused { .. }),
                "{error}"
            );
            assert!(error.to_string().contains(words), "{error}");
        };
        refused(|o| o.amount = 1.5, "amount");
        refused(|o| o.amount = f64::NAN, "amount");
        refused(|o| o.overlap_percent = 11, "tile overlap");
        refused(|o| o.overlap_percent = 51, "tile overlap");
        refused(|o| o.timeout_seconds = Some(0.0), "timeout");
        refused(
            |o| {
                o.local_support = Some(MlDenoiseLocalSupport {
                    clip_low: 21.0,
                    ..Default::default()
                })
            },
            "clipping point",
        );
        refused(
            |o| {
                o.local_support = Some(MlDenoiseLocalSupport {
                    background: -0.1,
                    ..Default::default()
                })
            },
            "background level",
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
        let read: MlDenoiseOptions = serde_json::from_str(
            r#"{"model": "m.xmlm", "precision": "high", "localSupport": {"smoothness": 4}}"#,
        )
        .unwrap();
        assert_eq!(read.amount, 0.9);
        assert_eq!(read.precision, MlDenoisePrecision::High);
        assert_eq!(read.local_support.unwrap().background, 0.25);
        // The default time limit grows with the image.
        assert_eq!(base.time_limit(0), Duration::from_secs(600));
        assert_eq!(base.time_limit(30_000_000), Duration::from_secs(1200));
    }

    /// A stand-in PixInsight: reads `-r=script,request,result,started` as
    /// the real one runs a script, then does what `body` says.
    #[cfg(unix)]
    fn fake_pixinsight(directory: &Path, name: &str, body: &str) -> PixInsight {
        use std::os::unix::fs::PermissionsExt;
        let path = directory.join(name);
        let script = format!(
            "#!/usr/bin/env python3\n\
             import json, os, shutil, sys, time\n\
             assert '--automation-mode' in sys.argv and '--force-exit' in sys.argv\n\
             assert '--no-focus-stealing' in sys.argv\n\
             run = next(a for a in sys.argv if a.startswith('-r='))[3:].split(',')\n\
             request = json.load(open(run[1]))\n\
             result_path = run[2]\n\
             open(run[3], 'w').write('started')\n\
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
        // stack holds, and an uncovered pixel.
        LinearImage::new(
            3,
            2,
            1,
            vec![-12.0, 0.0, f32::NAN, 1000.0, 40_000.0, 65_000.0],
        )
        .unwrap()
    }

    #[cfg(unix)]
    #[test]
    fn an_image_returns_on_its_own_scale_with_the_runs_details() {
        let directory = tempfile::tempdir().unwrap();
        let model = model_file(directory.path(), HEADER);
        // Returning the exchange image unchanged must give back the input.
        let pixinsight = fake_pixinsight(
            directory.path(),
            "PixInsight",
            "assert request['amount'] == 0.5 and request['overlapPercent'] == 30\n\
             assert request['precision'] == 'high'\n\
             assert request['localSupport'] == {'clipLow': -1.0, 'background': 0.3, 'smoothness': 3}\n\
             assert request['model'].endswith('model.xmlm')\n\
             shutil.copy(request['input'], request['output'])\n\
             finish({'ok': True, 'pixinsightVersion': '1.9.6 build 1702', 'seconds': 2.5})",
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
        assert_eq!(denoised.pixinsight_version, "1.9.6 build 1702");
        assert_eq!(denoised.seconds, 2.5);
        assert_eq!(denoised.model.weights_sha256.as_deref(), Some("7ca0"));
        for (got, want) in denoised.image.data.iter().zip(&image.data) {
            if want.is_finite() {
                assert!(
                    (got - want).abs() <= want.abs() * 1e-6 + 1e-2,
                    "{got} != {want}"
                );
            } else {
                assert!(got.is_nan(), "the uncovered pixel stays uncovered");
            }
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
    fn an_xisf_output_declares_the_range_its_input_declared() {
        let directory = tempfile::tempdir().unwrap();
        let model = model_file(directory.path(), HEADER);
        let pixinsight = fake_pixinsight(
            directory.path(),
            "PixInsight",
            "shutil.copy(request['input'], request['output'])\n\
             finish({'ok': True, 'pixinsightVersion': '1.9.6', 'seconds': 1})",
        );
        // An image PixInsight normalized: 0 to 1, declared so.
        let input = directory.path().join("normalized.xisf");
        let data = [0.1_f32, 0.2, 0.3, 0.4];
        seiza_image_io::save_f32(
            &input,
            2,
            2,
            F32ImageData::Mono(&data),
            &seiza_image_io::SaveOptions {
                format: Some(seiza_image_io::Format::Xisf),
                headers: &[],
                xisf_bounds: Some((0.0, 1.0)),
                fits_top_down: false,
            },
        )
        .unwrap();
        let output = directory.path().join("denoised.xisf");
        pixinsight
            .ml_denoise_file(&input, &output, &MlDenoiseOptions::new(&model), None)
            .unwrap();
        let written = FitsFrame::open(&output).unwrap();
        assert_eq!(written.bounds, Some((0.0, 1.0)));
        assert!((written.image.data[3] - 0.4).abs() < 1e-6);
    }

    #[cfg(unix)]
    #[test]
    fn a_failed_run_explains_what_pixinsight_said() {
        let directory = tempfile::tempdir().unwrap();
        let model = model_file(directory.path(), HEADER);
        let failing = fake_pixinsight(
            directory.path(),
            "failing",
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

        let crashing = fake_pixinsight(
            directory.path(),
            "crashing",
            "print('Segmentation fault')\nsys.exit(139)",
        );
        let error = crashing
            .ml_denoise_image(&adu_image(), &MlDenoiseOptions::new(&model), None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("exited with code 139"), "{error}");
        assert!(error.contains("Segmentation fault"), "{error}");

        // An RGB image needs an RGB route; this model has none.
        let mono_only = HEADER.replace(r#"<Model role="rgb""#, r#"<Model role="other""#);
        let model = model_file(directory.path(), &mono_only);
        let rgb = LinearImage::new(1, 1, 3, vec![0.1, 0.2, 0.3]).unwrap();
        let error = failing
            .ml_denoise_image(&rgb, &MlDenoiseOptions::new(&model), None)
            .unwrap_err();
        assert!(
            matches!(error, Error::ExternalToolRefused { .. }),
            "{error}"
        );
        assert!(
            error
                .to_string()
                .contains("cannot denoise a 3-channel image"),
            "{error}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_slot_another_instance_took_is_tried_again() {
        let directory = tempfile::tempdir().unwrap();
        let model = model_file(directory.path(), HEADER);
        let attempts = directory.path().join("attempts");
        let racing = fake_pixinsight(
            directory.path(),
            "racing",
            &format!(
                "count = len(open({path:?}).read()) if os.path.exists({path:?}) else 0\n\
                 open({path:?}, 'a').write('x')\n\
                 if count == 0:\n    \
                     print('*** Fatal Error: Unable to create application instance: QSharedMemory::create: already exists')\n    \
                     sys.exit(1)\n\
                 shutil.copy(request['input'], request['output'])\n\
                 finish({{'ok': True, 'pixinsightVersion': '1.9.6', 'seconds': 1}})",
                path = attempts.display().to_string()
            ),
        );
        racing
            .ml_denoise_image(&adu_image(), &MlDenoiseOptions::new(&model), None)
            .unwrap();
        assert_eq!(std::fs::read_to_string(&attempts).unwrap(), "xx");
    }

    #[cfg(unix)]
    #[test]
    fn a_run_past_its_time_limit_or_cancelled_closes_pixinsight() {
        let directory = tempfile::tempdir().unwrap();
        let model = model_file(directory.path(), HEADER);
        for (name, cancel_after_start) in [("timed-out", false), ("cancelled", true)] {
            let pid_path = directory.path().join(format!("{name}.pid"));
            let finished = directory.path().join(format!("{name}.finished"));
            let hanging = fake_pixinsight(
                directory.path(),
                name,
                &format!(
                    "with open({pid:?}, 'w') as file: file.write(str(os.getpid()))\n\
                     time.sleep(60)\n\
                     open({finished:?}, 'w').write('finished')",
                    pid = pid_path.display().to_string(),
                    finished = finished.display().to_string(),
                ),
            );
            let ready = pid_path.clone();
            let cancel = CancelSignal::new(move || {
                // This callback runs after spawn. Wait for the actual child,
                // even if preprocessing or the shared startup lock took longer
                // than the timeout. The bound only guards a broken fixture.
                let deadline = Instant::now() + Duration::from_secs(30);
                loop {
                    if std::fs::read_to_string(&ready)
                        .is_ok_and(|text| text.parse::<libc::pid_t>().is_ok_and(|pid| pid > 0))
                    {
                        return cancel_after_start;
                    }
                    assert!(Instant::now() < deadline, "fake PixInsight did not start");
                    std::thread::sleep(Duration::from_millis(10));
                }
            });
            let options = MlDenoiseOptions {
                // For cancellation, leave ample time to observe readiness and
                // return Cancelled. For timeout, the callback never cancels.
                timeout_seconds: Some(if cancel_after_start { 30.0 } else { 0.3 }),
                ..MlDenoiseOptions::new(&model)
            };
            let error = hanging
                .ml_denoise_image(&adu_image(), &options, Some(&cancel))
                .unwrap_err();
            let pid: libc::pid_t = std::fs::read_to_string(&pid_path).unwrap().parse().unwrap();
            assert!(pid > 0);
            let mut status = 0;
            // SAFETY: this is the PID our fixture just wrote; status is writable.
            // A reaped child gives ECHILD, a running child gives zero, and an
            // exited but unreaped child gives its PID. Neither latter case passes.
            let waited = unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) };
            let wait_error = std::io::Error::last_os_error();
            if waited == 0 {
                // Clean up a child left alive by a regression before asserting.
                // SAFETY: waitpid above confirmed this PID is still our child.
                unsafe {
                    libc::kill(pid, libc::SIGKILL);
                    libc::waitpid(pid, &mut status, 0);
                }
            }
            assert_eq!(waited, -1, "{name}: PixInsight was not reaped");
            assert_eq!(wait_error.raw_os_error(), Some(libc::ECHILD));
            assert!(
                !finished.exists(),
                "{name}: PixInsight finished instead of being killed"
            );
            if cancel_after_start {
                assert!(matches!(error, Error::Cancelled), "{error}");
            } else {
                let error = error.to_string();
                assert!(error.contains("did not finish within"), "{error}");
                assert!(error.contains("license"), "{error}");
            }
        }
    }

    #[test]
    fn only_the_core_ml_models_of_the_exited_instance_are_removed() {
        let uuid = "43E2160F-B36C-4871-89AE-DF590E19297F";
        let leftover = |pid: &str| format!("onnxruntime-{uuid}-{pid}-00001ED4787A11A3");
        assert_eq!(core_ml_leftover_pid(&leftover("92078")), Some(92078));
        let compiled = format!("{}.mlmodelc", leftover("92078"));
        assert_eq!(core_ml_leftover_pid(&compiled), Some(92078));

        let folder = tempfile::tempdir().unwrap();
        // A package and its compiled model, both folders, and a file from
        // another session of the same instance.
        for name in [leftover("9203"), format!("{}.mlmodelc", leftover("9203"))] {
            let package = folder.path().join(name);
            std::fs::create_dir_all(package.join("Data")).unwrap();
            std::fs::write(package.join("Manifest.json"), "{}").unwrap();
        }
        let file = "onnxruntime-0B7357C7-74CC-49E6-91CD-6D213F707C64-9203-00001ED9BD0FAF90";
        std::fs::write(folder.path().join(file), "").unwrap();
        let mut kept = vec![
            // Other instances, whose pids hold the same digits.
            leftover("92031"),
            format!("{}.mlmodelc", leftover("92031")),
            leftover("19203"),
            leftover("920"),
            leftover("+9203"),
            // Names that are not ONNX Runtime's.
            format!("onnxruntime-{uuid}-9203"),
            format!("onnxruntime-{uuid}-9203-00001ED4787A11A3.mlpackage"),
            "onnxruntime-not-a-uuid-at-all-9203-00001ED4787A11A3".to_string(),
            format!("seiza-{uuid}-9203-00001ED4787A11A3"),
            "com.apple.metal".to_string(),
        ];
        for name in &kept {
            std::fs::create_dir(folder.path().join(name)).unwrap();
        }
        remove_core_ml_leftovers_in(folder.path(), 9203);
        let mut left: Vec<String> = std::fs::read_dir(folder.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        left.sort();
        kept.sort();
        assert_eq!(left, kept);
        // A folder that cannot be read removes nothing, and the run goes on.
        remove_core_ml_leftovers_in(&folder.path().join("missing"), 9203);
    }

    /// A stand-in PixInsight that leaves Core ML models, as the real one
    /// does when it is closed, has them removed however its run ends.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_core_ml_models_an_instance_leaves_are_removed_however_it_ends() {
        let directory = tempfile::tempdir().unwrap();
        let model = model_file(directory.path(), HEADER);
        let temp = darwin_user_temp_dir().expect("the per-user temporary folder");
        let pids = directory.path().join("pids");
        // A package and its compiled model, named as ONNX Runtime names
        // them, and the pid, so the test can look for them.
        let leave = format!(
            "import random, uuid\n\
             name = 'onnxruntime-%s-%d-%016X' % (str(uuid.uuid4()).upper(), os.getpid(), random.getrandbits(64))\n\
             for entry in (name, name + '.mlmodelc'):\n    os.makedirs(os.path.join({temp:?}, entry, 'Data'))\n\
             open({pids:?}, 'a').write('%d\\n' % os.getpid())",
            temp = temp.display().to_string(),
            pids = pids.display().to_string(),
        );
        let fake = |name: &str, then: &str| {
            fake_pixinsight(directory.path(), name, &format!("{leave}\n{then}"))
        };
        let options = MlDenoiseOptions::new(&model);
        fake(
            "finishing",
            "shutil.copy(request['input'], request['output'])\n\
             finish({'ok': True, 'pixinsightVersion': '1.9.6', 'seconds': 1})",
        )
        .ml_denoise_image(&adu_image(), &options, None)
        .unwrap();
        fake(
            "failing",
            "finish({'ok': False, 'error': 'MLDenoise did not run'})",
        )
        .ml_denoise_image(&adu_image(), &options, None)
        .unwrap_err();
        // Cancelled once its models are there: closed by a kill.
        let written = pids.clone();
        let cancel = CancelSignal::new(move || {
            std::fs::read_to_string(&written).is_ok_and(|pids| pids.lines().count() == 3)
        });
        let bounded = MlDenoiseOptions {
            timeout_seconds: Some(30.0),
            ..options.clone()
        };
        let error = fake("hanging", "time.sleep(60)")
            .ml_denoise_image(&adu_image(), &bounded, Some(&cancel))
            .unwrap_err();
        assert!(matches!(error, Error::Cancelled), "{error}");

        let pids: Vec<u32> = std::fs::read_to_string(&pids)
            .unwrap()
            .lines()
            .map(|pid| pid.parse().unwrap())
            .collect();
        assert_eq!(pids.len(), 3);
        let left: Vec<String> = std::fs::read_dir(&temp)
            .unwrap()
            .flatten()
            .filter_map(|entry| entry.file_name().into_string().ok())
            .filter(|name| core_ml_leftover_pid(name).is_some_and(|pid| pids.contains(&pid)))
            .collect();
        assert!(left.is_empty(), "left behind: {left:?}");
    }

    #[cfg(unix)]
    #[test]
    fn a_launcher_is_not_given_a_folder_the_shell_would_interpret() {
        let directory = tempfile::tempdir().unwrap();
        let launcher = PixInsight::with_executable(directory.path().join("PixInsight.sh"));
        let unsafe_folder = directory.path().join("a$(touch x)");
        let error = launcher
            .check_arguments(&unsafe_folder, &ScriptPaths::below(&unsafe_folder))
            .unwrap_err();
        assert!(
            matches!(error, Error::ExternalToolRefused { .. }),
            "{error}"
        );
        assert!(error.to_string().contains("TMPDIR"), "{error}");
        let plain = directory.path().join("plain");
        launcher
            .check_arguments(&plain, &ScriptPaths::below(&plain))
            .unwrap();
        // A program PixInsight runs directly takes such a folder.
        let program = PixInsight::with_executable(directory.path().join("PixInsight"));
        program
            .check_arguments(&unsafe_folder, &ScriptPaths::below(&unsafe_folder))
            .unwrap();
    }

    /// Runs the installed PixInsight with a real model. Set
    /// `SEIZA_MLDENOISE_MODEL` to an `.xmlm` file to run it.
    #[test]
    #[ignore = "needs PixInsight and an MLDenoise model"]
    fn real_pixinsight_denoises_an_image() {
        let model = std::env::var_os("SEIZA_MLDENOISE_MODEL").expect("SEIZA_MLDENOISE_MODEL");
        let pixinsight = PixInsight::locate().expect("PixInsight is installed");
        // A background with Gaussian noise and one star, in ADU, and an
        // uncovered strip. MLDenoise leaves uniform noise largely alone.
        let (width, height) = (256, 256);
        // splitmix64: consecutive xorshift32 pairs made patterned noise,
        // which MLDenoise keeps as structure.
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut uniform = || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let mut gaussian = || {
            let (u, v) = (uniform(), uniform());
            ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
        };
        let data: Vec<f32> = (0..width * height)
            .map(|index| {
                let (x, y) = ((index % width) as f32, (index / width) as f32);
                let noise = 57.0 * gaussian();
                if x < 16.0 {
                    return f32::NAN;
                }
                let star = 30_000.0 * (-((x - 128.0).powi(2) + (y - 128.0).powi(2)) / 8.0).exp();
                1_000.0 + star + noise
            })
            .collect();
        let image = LinearImage::new(width, height, 1, data).unwrap();
        let options = MlDenoiseOptions {
            amount: 1.0,
            ..MlDenoiseOptions::new(PathBuf::from(&model))
        };
        let denoised = pixinsight.ml_denoise_image(&image, &options, None).unwrap();
        let corner = |image: &LinearImage| -> (f32, f32) {
            let corner: Vec<f32> = (0..64)
                .flat_map(|y| (32..96).map(move |x| y * width + x))
                .map(|index| image.data[index])
                .collect();
            let mean = corner.iter().sum::<f32>() / corner.len() as f32;
            let spread = (corner.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
                / corner.len() as f32)
                .sqrt();
            (mean, spread)
        };
        let (before_mean, before_spread) = corner(&image);
        let (after_mean, after_spread) = corner(&denoised.image);
        eprintln!(
            "PixInsight {} in {:.1} s: background {before_mean:.1}±{before_spread:.1} -> \
             {after_mean:.1}±{after_spread:.1} ADU, star peak {:.0} -> {:.0}",
            denoised.pixinsight_version,
            denoised.seconds,
            image.data[128 * width + 128],
            denoised.image.data[128 * width + 128]
        );
        assert!(after_spread < before_spread * 0.2);
        assert!(
            (after_mean - before_mean).abs() < 5.0,
            "kept the background"
        );
        assert!(
            (0..height).all(|y| denoised.image.data[y * width].is_nan()),
            "the uncovered strip stays uncovered"
        );

        // RGB at amount 0 comes back as it went.
        let rgb_data: Vec<f32> = (0..64 * 64 * 3)
            .map(|index| 500.0 + (index % 251) as f32 * 3.0 - 100.0 * (index % 3) as f32)
            .collect();
        let rgb = LinearImage::new(64, 64, 3, rgb_data).unwrap();
        let unchanged = pixinsight
            .ml_denoise_image(
                &rgb,
                &MlDenoiseOptions {
                    amount: 0.0,
                    ..MlDenoiseOptions::new(PathBuf::from(&model))
                },
                None,
            )
            .unwrap();
        let worst = unchanged
            .image
            .data
            .iter()
            .zip(&rgb.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(worst < 0.05, "RGB at amount 0 changed by {worst} ADU");
    }
}
