use crate::bundle::ModelBundle;
use crate::error::ModelError;
use directories::ProjectDirs;
use ort::environment::Environment;
use ort::ep::{self, ExecutionProvider};
use serde::{Deserialize, Serialize};
use std::ffi::{CStr, c_char, c_void};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::OnceLock;
use std::{env, fmt};

/// Names ONNX Runtime's library, for hosts that do not keep it where Seiza
/// looks. A set variable that points at nothing is an error, never a
/// fall-through.
const LIBRARY_VARIABLE: &str = "SEIZA_ONNXRUNTIME";

/// Stops Microsoft's own ONNX Runtime builds reporting to Microsoft, which
/// they do unless told not to. Only this, read as the library starts, stops
/// the uploader and its device identifier being made at all.
const TELEMETRY_VARIABLE: &str = "ORT_DISABLE_TELEMETRY";

#[cfg(target_os = "macos")]
const LIBRARY_FILE: &str = "libonnxruntime.dylib";
#[cfg(target_os = "windows")]
const LIBRARY_FILE: &str = "onnxruntime.dll";
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
const LIBRARY_FILE: &str = "libonnxruntime.so";

#[cfg(target_os = "macos")]
const SYSTEM_DIRECTORIES: &[&str] = &["/opt/homebrew/lib", "/usr/local/lib"];
#[cfg(target_os = "windows")]
const SYSTEM_DIRECTORIES: &[&str] = &[];
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
const SYSTEM_DIRECTORIES: &[&str] = &[
    "/usr/lib",
    "/usr/lib64",
    "/usr/local/lib",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
];

/// Which of Core ML's compute units a model runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CoreMlUnits {
    /// The Neural Engine, which needs an f16 graph: an f32 graph on this
    /// setting runs on the CPU without saying so.
    NeuralEngine,
    /// The GPU.
    Gpu,
    /// Core ML's own CPU path.
    Cpu,
}

/// What runs a model's tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Core ML, on Apple hardware.
    CoreMl(CoreMlUnits),
    /// NVIDIA TensorRT.
    TensorRt,
    /// NVIDIA TensorRT for RTX.
    TensorRtRtx,
    /// NVIDIA CUDA.
    Cuda,
    /// WebGPU, through ONNX Runtime's experimental provider.
    WebGpu,
    /// ONNX Runtime's CPU provider, which every build has.
    Cpu,
}

/// A backend's spelling was not one of the known words.
#[derive(Debug, thiserror::Error)]
#[error("unknown backend {0:?}: expected one of {known}", known = Backend::ALL.map(|b| b.to_string()).join(", "))]
pub struct UnknownBackend(String);

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::CoreMl(CoreMlUnits::NeuralEngine) => "neural-engine",
            Self::CoreMl(CoreMlUnits::Gpu) => "coreml-gpu",
            Self::CoreMl(CoreMlUnits::Cpu) => "coreml-cpu",
            Self::TensorRt => "tensorrt",
            Self::TensorRtRtx => "tensorrt-rtx",
            Self::Cuda => "cuda",
            Self::WebGpu => "webgpu",
            Self::Cpu => "cpu",
        })
    }
}

impl FromStr for Backend {
    type Err = UnknownBackend;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|backend| backend.to_string() == text)
            .ok_or_else(|| UnknownBackend(text.to_string()))
    }
}

/// The precision of a model's graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    /// 32-bit floats.
    F32,
    /// 16-bit floats, which the Neural Engine needs.
    F16,
}

/// A precision's spelling was not `f32` or `f16`.
#[derive(Debug, thiserror::Error)]
#[error("unknown precision {0:?}: expected f32 or f16")]
pub struct UnknownPrecision(String);

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        })
    }
}

impl FromStr for Precision {
    type Err = UnknownPrecision;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            _ => Err(UnknownPrecision(text.to_string())),
        }
    }
}

/// What a host may set about how a backend is reached.
#[derive(Clone, Debug, Default)]
pub struct BackendOptions {
    /// ONNX Runtime's library, in place of searching for it. The library loads
    /// once per process, so only the first call that needs it is heard.
    pub library: Option<PathBuf>,
    /// Where compiled models are kept, in place of the platform's cache
    /// directory.
    pub cache_dir: Option<PathBuf>,
    /// A tile's side, in place of the backend's default.
    pub tile: Option<usize>,
    /// The graph's precision, in place of the backend's default.
    pub precision: Option<Precision>,
}

/// Which ONNX Runtime a process loaded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeInfo {
    /// The library's path.
    pub path: PathBuf,
    /// The library's version, such as `1.31.1`.
    pub version: String,
}

/// Whether a backend can run here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendStatus {
    /// The backend.
    pub backend: Backend,
    /// Why it cannot run here, or `None` when it can.
    pub unavailable: Option<String>,
    /// Whether [`Backend::detect`] considers it. A backend that runs but has
    /// not been measured on this hardware is used only when asked for.
    pub automatic: bool,
}

pub(crate) struct Runtime {
    pub(crate) environment: Environment,
    pub(crate) info: RuntimeInfo,
}

static RUNTIME: OnceLock<Result<Runtime, String>> = OnceLock::new();

/// ONNX Runtime, found and opened on first use and held for the process's
/// life, since it can be loaded only once.
pub(crate) fn runtime(options: &BackendOptions) -> Result<&'static Runtime, ModelError> {
    RUNTIME
        .get_or_init(|| load(options))
        .as_ref()
        .map_err(|message| ModelError::RuntimeNotFound(message.clone()))
}

fn load(options: &BackendOptions) -> Result<Runtime, String> {
    let pinned = env::var_os(LIBRARY_VARIABLE).map(PathBuf::from);
    let places = library_places();
    let path = find_library(options.library.as_deref(), pinned.as_deref(), &places)?;
    if env::var_os(TELEMETRY_VARIABLE).is_none() {
        // SAFETY: this runs once, under `RUNTIME`'s lock. Another thread
        // reading the environment through C at this moment is the one risk,
        // and it has to be taken here, since the library reads the variable
        // only as it starts.
        unsafe { env::set_var(TELEMETRY_VARIABLE, "1") };
    }
    let builder = ort::init_from(&path).map_err(|error| {
        if path.is_absolute() {
            return error.to_string();
        }
        let searched: Vec<String> = places
            .iter()
            .map(|place| place.display().to_string())
            .collect();
        format!(
            "{LIBRARY_FILE} is not in {} or on the system's search path: install ONNX Runtime, \
             or set {LIBRARY_VARIABLE} to its library",
            searched.join(", ")
        )
    })?;
    // For a process that had ONNX Runtime loaded before Seiza came to it.
    let environment = builder
        .with_name("seiza")
        .with_telemetry(false)
        .build()
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let version = library_version(&path).unwrap_or_else(|| "unknown".to_string());
    Ok(Runtime {
        environment,
        info: RuntimeInfo { path, version },
    })
}

/// The version the library reports. `ort` reads it to check compatibility but
/// keeps it to itself, and a build's own string may not hold it.
fn library_version(path: &Path) -> Option<String> {
    /// The head of ONNX Runtime's `OrtApiBase`, which its C API keeps stable.
    #[repr(C)]
    struct ApiBase {
        get_api: *const c_void,
        get_version_string: unsafe extern "C" fn() -> *const c_char,
    }

    // SAFETY: `path` is the library `ort` has just loaded and checked, so this
    // only takes another reference to it. `OrtGetApiBase` takes no arguments
    // and returns a pointer to a static table that begins as `ApiBase` does,
    // and `GetVersionString` returns a static string.
    unsafe {
        let library = libloading::Library::new(path).ok()?;
        let get_base = library
            .get::<unsafe extern "C" fn() -> *const ApiBase>(b"OrtGetApiBase")
            .ok()?;
        let base = get_base();
        if base.is_null() {
            return None;
        }
        let version = ((*base).get_version_string)();
        (!version.is_null()).then(|| CStr::from_ptr(version).to_string_lossy().into_owned())
    }
}

/// The directories searched for the library, best first: beside the
/// executable, where a package puts it, then the shared data directory, then
/// the system's.
fn library_places() -> Vec<PathBuf> {
    let mut places = Vec::new();
    if let Some(directory) = env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
    {
        if let Some(prefix) = directory.parent() {
            places.push(prefix.join("lib").join("seiza"));
        }
        places.insert(0, directory);
    }
    if let Some(dirs) = ProjectDirs::from("fyi", "Seiza", "seiza") {
        places.push(dirs.data_dir().to_path_buf());
    }
    places.extend(SYSTEM_DIRECTORIES.iter().map(PathBuf::from));
    places
}

/// The library to load: `explicit`, then `pinned`, then the first of `places`
/// that holds it, then the bare name, for the system's own search.
fn find_library(
    explicit: Option<&Path>,
    pinned: Option<&Path>,
    places: &[PathBuf],
) -> Result<PathBuf, String> {
    if let Some(path) = explicit {
        return usable(path, "the library given");
    }
    if let Some(path) = pinned {
        return usable(path, LIBRARY_VARIABLE);
    }
    Ok(places
        .iter()
        .map(|place| place.join(LIBRARY_FILE))
        .find(|path| path.is_file())
        .unwrap_or_else(|| PathBuf::from(LIBRARY_FILE)))
}

fn usable(path: &Path, source: &str) -> Result<PathBuf, String> {
    if path.is_file() {
        Ok(path.to_path_buf())
    } else {
        Err(format!(
            "{source} names {}, which is not a file",
            path.display()
        ))
    }
}

impl Backend {
    /// Every backend, best first. Apple's GPU comes before its Neural Engine:
    /// on an otherwise idle M4 Max it was faster, at f32 as well as f16.
    pub const ALL: [Backend; 8] = [
        Self::CoreMl(CoreMlUnits::Gpu),
        Self::CoreMl(CoreMlUnits::NeuralEngine),
        Self::CoreMl(CoreMlUnits::Cpu),
        Self::TensorRtRtx,
        Self::TensorRt,
        Self::Cuda,
        Self::WebGpu,
        Self::Cpu,
    ];

    /// The precision this backend runs unless asked otherwise. Only the Neural
    /// Engine needs f16: on Apple's GPU it saved 8% of the time, for about 800
    /// times the error on the sky.
    pub fn default_precision(self) -> Precision {
        match self {
            Self::CoreMl(CoreMlUnits::NeuralEngine) => Precision::F16,
            _ => Precision::F32,
        }
    }

    /// The tile this backend runs unless asked otherwise, or `None` for the
    /// model card's. A larger tile keeps more of what it computes.
    pub fn default_tile(self) -> Option<usize> {
        match self {
            Self::CoreMl(CoreMlUnits::Cpu) | Self::Cpu => None,
            _ => Some(1024),
        }
    }

    /// Whether [`Backend::detect`] may pick this backend on this machine.
    /// NVIDIA and Intel GPUs have not been measured, so they run only when
    /// asked for.
    fn automatic(self) -> bool {
        match self {
            Self::CoreMl(CoreMlUnits::NeuralEngine | CoreMlUnits::Gpu) | Self::Cpu => true,
            Self::CoreMl(CoreMlUnits::Cpu)
            | Self::TensorRt
            | Self::TensorRtRtx
            | Self::Cuda
            | Self::WebGpu => false,
        }
    }

    /// Why this backend cannot run here, once ONNX Runtime is loaded.
    fn unavailable(self) -> Option<String> {
        if self == Self::CoreMl(CoreMlUnits::NeuralEngine)
            && !cfg!(all(target_os = "macos", target_arch = "aarch64"))
        {
            return Some("needs Apple silicon".to_string());
        }
        let (built, provider) = match self {
            Self::CoreMl(_) => (ep::CoreML::is_available(), "Core ML"),
            Self::TensorRt => (ep::TensorRT::is_available(), "TensorRT"),
            Self::TensorRtRtx => (ep::NVRTX::is_available(), "TensorRT for RTX"),
            Self::Cuda => (ep::CUDA::is_available(), "CUDA"),
            Self::WebGpu => (ep::WebGPU::is_available(), "WebGPU"),
            Self::Cpu => return None,
        };
        match built {
            Ok(true) => None,
            Ok(false) => Some(format!("this ONNX Runtime was built without {provider}")),
            Err(error) => Some(error.to_string()),
        }
    }

    /// Reports every backend with why it cannot run, which is what a user
    /// needs when a GPU goes unused. With no ONNX Runtime, that is the reason
    /// for all of them.
    pub fn available(options: &BackendOptions) -> Vec<BackendStatus> {
        let loaded = runtime(options);
        Self::ALL
            .into_iter()
            .map(|backend| BackendStatus {
                backend,
                unavailable: match &loaded {
                    Ok(_) => backend.unavailable(),
                    Err(error) => Some(error.to_string()),
                },
                automatic: backend.automatic(),
            })
            .collect()
    }

    /// The best backend this machine can run and `bundle`'s card allows.
    pub fn detect(bundle: &ModelBundle, options: &BackendOptions) -> Result<Backend, ModelError> {
        runtime(options)?;
        Self::ALL
            .into_iter()
            .find(|&backend| {
                backend.automatic()
                    && backend.unavailable().is_none()
                    && bundle.card().allows(backend)
            })
            .ok_or_else(|| {
                ModelError::ModelContract(format!(
                    "no backend here is on the card of {}",
                    bundle.card().name
                ))
            })
    }
}

/// The ONNX Runtime a process has loaded, loading it if need be.
pub fn runtime_info(options: &BackendOptions) -> Result<RuntimeInfo, ModelError> {
    Ok(runtime(options)?.info.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_backend_s_word_reads_back_as_itself() {
        for backend in Backend::ALL {
            assert_eq!(backend.to_string().parse::<Backend>().unwrap(), backend);
        }
        assert!("metal".parse::<Backend>().is_err());
    }

    #[test]
    fn only_the_neural_engine_defaults_to_f16_and_both_take_a_large_tile() {
        let engine = Backend::CoreMl(CoreMlUnits::NeuralEngine);
        let gpu = Backend::CoreMl(CoreMlUnits::Gpu);
        assert_eq!(engine.default_precision(), Precision::F16);
        assert_eq!(gpu.default_precision(), Precision::F32);
        assert_eq!(engine.default_tile(), Some(1024));
        assert_eq!(gpu.default_tile(), Some(1024));
        assert_eq!(Backend::Cpu.default_precision(), Precision::F32);
        assert_eq!(Backend::Cpu.default_tile(), None);
    }

    #[test]
    fn apple_s_gpu_comes_before_its_neural_engine() {
        let place = |units| {
            Backend::ALL
                .iter()
                .position(|&backend| backend == Backend::CoreMl(units))
        };
        assert!(place(CoreMlUnits::Gpu) < place(CoreMlUnits::NeuralEngine));
    }

    #[test]
    fn an_unmeasured_gpu_runs_only_when_asked_for() {
        assert!(!Backend::Cuda.automatic());
        assert!(!Backend::WebGpu.automatic());
        assert!(Backend::Cpu.automatic());
    }

    #[test]
    fn the_library_search_keeps_its_order_and_a_pin_never_falls_through() {
        let temp = tempfile::tempdir().unwrap();
        let (first, second) = (temp.path().join("first"), temp.path().join("second"));
        std::fs::create_dir_all(&first).unwrap();
        std::fs::create_dir_all(&second).unwrap();
        let places = [first.clone(), second.clone()];

        assert_eq!(
            find_library(None, None, &places).unwrap(),
            PathBuf::from(LIBRARY_FILE)
        );
        let library = second.join(LIBRARY_FILE);
        std::fs::write(&library, b"").unwrap();
        assert_eq!(find_library(None, None, &places).unwrap(), library);

        let chosen = temp.path().join("chosen.lib");
        std::fs::write(&chosen, b"").unwrap();
        assert_eq!(find_library(None, Some(&chosen), &places).unwrap(), chosen);
        assert_eq!(
            find_library(Some(&chosen), Some(&library), &places).unwrap(),
            chosen
        );

        let missing = temp.path().join("missing.lib");
        assert!(find_library(None, Some(&missing), &places).is_err());
        assert!(find_library(Some(&missing), None, &places).is_err());
    }
}
