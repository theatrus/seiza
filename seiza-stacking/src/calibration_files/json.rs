//! The C ABI's one-call file preparation adapter. Native hosts use
//! [`FileCalibration`] and its methods, never this request protocol.
use super::*;
use seiza_calibration::library::json::{RigSettings, Setting};

/// C-only workflow input. Fields use Rust names here; the C binding checks and
/// translates camelCase. Database-backed hosts use native library stages.
#[derive(Default, Deserialize, schemars::JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct Request {
    pub operation: String,
    pub lights: Vec<PathBuf>,
    pub library: Option<PathBuf>,
    pub provided: Vec<PathBuf>,
    pub cache: PathBuf,
    pub settings: Vec<Setting>,
    pub rigs: Vec<RigSettings>,
    pub choices: Vec<Choice>,
    pub group_overrides: Vec<seiza_calibration::flat_response::groups::GroupOverrideRow>,
    pub light_choices: Vec<LightChoice>,
    pub incomplete: IncompleteLights,
    pub flat_star_masking: bool,
    pub flat_stability: policy::FlatRetryPolicy,
    pub flat_association: policy::Association,
    pub dark_association: policy::Association,
    pub review_key: Option<String>,
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
    match request.operation.as_str() {
        "schema" => {
            return Ok(Answer::Settings {
                settings: policy::rules::fields().to_vec(),
            });
        }
        "verify" => {
            verify_sources(&request.sources)?;
            return Ok(Answer::Verified { valid: true });
        }
        "analyze" | "execute" => {}
        _ => {
            return Err(PreparationError::Failed(
                "Unknown calibration operation".into(),
            ));
        }
    }
    let values =
        |settings: Vec<Setting>| -> std::result::Result<BTreeMap<String, Value>, PreparationError> {
            let mut out = BTreeMap::new();
            for setting in settings {
                if out.insert(setting.key, setting.value).is_some() {
                    return Err(PreparationError::Failed(
                        "Duplicate calibration setting".into(),
                    ));
                }
            }
            Ok(out)
        };
    let mut rigs = BTreeMap::new();
    for rig in request.rigs {
        if rigs
            .insert(
                rig.rig_uuid,
                policy::rules::RigLayer {
                    name: rig.name,
                    values: values(rig.settings)?,
                },
            )
            .is_some()
        {
            return Err(PreparationError::Failed(
                "Duplicate calibration rig settings".into(),
            ));
        }
    }
    let rules = policy::rules::RuleBook::validated(values(request.settings)?, rigs)
        .map_err(|e| PreparationError::Failed(e.to_string()))?;
    let mut files = FileCalibration::new(request.lights, request.cache);
    files.library = request.library.filter(|p| !p.as_os_str().is_empty());
    files.provided = request.provided;
    files.rules = rules;
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
    let report = if request.operation == "analyze" {
        files.analyze(cancel, progress)
    } else {
        files.prepare(
            request
                .review_key
                .as_deref()
                .ok_or(PreparationError::ReviewChanged)?,
            cancel,
            progress,
        )
    }?;
    Ok(Answer::Report(Box::new(report)))
}
