//! The C ABI's one-call file preparation adapter. Native hosts use
//! [`FileCalibration`] and its methods, never this request protocol.
use super::*;
use seiza_calibration::library::json::{RigSettings, Setting, rule_book};

/// What a C workflow call does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    /// [`FileCalibration::analyze`].
    Analyze,
    /// [`FileCalibration::prepare`] of `review_key`.
    Execute,
    /// [`verify_sources`] of `sources`.
    Verify,
    /// The native settings panel's fields.
    Schema,
}

/// C-only workflow input. Fields use Rust names here; the C binding checks and
/// translates camelCase. Database-backed hosts use native library stages.
#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub operation: Operation,
    #[serde(default)]
    pub lights: Vec<PathBuf>,
    #[serde(default)]
    pub library: Option<PathBuf>,
    #[serde(default)]
    pub provided: Vec<PathBuf>,
    #[serde(default)]
    pub cache: PathBuf,
    #[serde(default)]
    pub settings: Vec<Setting>,
    #[serde(default)]
    pub rigs: Vec<RigSettings>,
    #[serde(default)]
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub group_overrides: Vec<seiza_calibration::flat_response::groups::GroupOverrideRow>,
    #[serde(default)]
    pub light_choices: Vec<LightChoice>,
    #[serde(default)]
    pub incomplete: IncompleteLights,
    #[serde(default)]
    pub flat_star_masking: bool,
    #[serde(default)]
    pub flat_stability: policy::FlatRetryPolicy,
    #[serde(default)]
    pub flat_association: policy::Association,
    #[serde(default)]
    pub dark_association: policy::Association,
    #[serde(default)]
    pub review_key: Option<String>,
    #[serde(default)]
    pub sources: Vec<Source>,
}
/// The operation's native return value, before C property-name translation.
#[derive(Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum Answer {
    /// Review or executed file sessions.
    Report(Box<PreparationReport>),
    /// Requested source stamps still match.
    Verified { valid: bool },
    /// The native settings panel schema.
    Settings {
        settings: Vec<policy::rules::FieldSpec>,
    },
}
/// Translate a C workflow into native calls. No cache or matching decisions
/// live here; the same typed coordinator is available to Python and Rust.
pub fn run(
    request: Request,
    cancel: CancelSignal,
    progress: impl Fn(&PreparationProgress),
) -> std::result::Result<Answer, PreparationError> {
    let review_key = match request.operation {
        Operation::Schema => {
            return Ok(Answer::Settings {
                settings: policy::rules::fields().to_vec(),
            });
        }
        Operation::Verify => {
            verify_sources(&request.sources)?;
            return Ok(Answer::Verified { valid: true });
        }
        Operation::Analyze => None,
        Operation::Execute => Some(
            request
                .review_key
                .ok_or("execute needs the reviewKey analyze returned")?,
        ),
    };
    let mut files = FileCalibration::new(request.lights, request.cache);
    files.library = request.library.filter(|p| !p.as_os_str().is_empty());
    files.provided = request.provided;
    files.rules = rule_book(request.settings, request.rigs)
        .map_err(|e| PreparationError::Failed(e.to_string()))?;
    files.options = PreparationOptions {
        incomplete: request.incomplete,
        masters: policy::SessionOptions {
            flat_star_masking: request.flat_star_masking,
            flat_stability: request.flat_stability,
        },
        associations: policy::Associations {
            flat: request.flat_association,
            dark: request.dark_association,
        },
        choices: request.choices,
        group_overrides: request.group_overrides,
        light_choices: request.light_choices,
    };
    let report = match review_key {
        None => files.analyze(cancel, progress),
        Some(key) => files.prepare(&key, cancel, progress),
    }?;
    Ok(Answer::Report(Box::new(report)))
}
