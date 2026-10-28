//! The C ABI translates the shared library policy's immutable value snapshot.
use super::{
    clear_error, display_length, ffi_result, owned_json, required_f32_slice, required_str,
};
use seiza_calibration::library;
use serde_json::Value;
use std::ffi::c_char;
use std::ptr;
use std::sync::Arc;

/// Immutable measured grouping book. Safe for concurrent read-only planning.
pub struct SeizaFlatGroups {
    book: Arc<seiza_calibration::flat_response::groups::Book>,
}

/// Borrowed, row-major float image. RGB channels are interleaved. Samples use
/// the same units as the other images in this measurement. No pointer is retained.
#[repr(C)]
pub struct SeizaFlatStabilityImage {
    pub data: *const f32,
    pub length: usize,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
}

unsafe fn cached_image(
    image: &SeizaFlatStabilityImage,
) -> Result<seiza_stacking::LinearImage, String> {
    let length = display_length(image.width, image.height, image.channels)?;
    if length != image.length {
        return Err("cached image length must match width * height * channels".into());
    }
    let data = unsafe { required_f32_slice(image.data, length, "cached image") }?;
    seiza_stacking::LinearImage::new(image.width, image.height, image.channels, data.to_vec())
        .map_err(|e| e.to_string())
}

/// Measure cached master/source images without file I/O, using the same core
/// algorithm as seiza_master_flat_stability_json. Nullable options_json is
/// camelCase FlatStabilityImageOptions: inputs (aligned exposureSeconds, bayer,
/// skipped records), bayer {pattern,xOffset,yOffset}, darkExposureSeconds and
/// darkBiasSubtracted. Empty inputs metadata uses defaults for all sources.
/// Bias/dark are optional; pixels are already on the same scale. Buffers are
/// copied for the call and never retained. Free result/error with seiza_string_free.
/// # Safety
/// master is a valid descriptor; inputs points to input_count descriptors (or
/// is null when count is zero). Bias/dark are null or valid descriptors. Every
/// data pointer holds length aligned readable floats. Descriptors and buffers
/// remain immutable for this call. options_json is null or NUL-terminated UTF-8;
/// error_out is null or writable pointer storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_master_flat_stability_images_json(
    master: *const SeizaFlatStabilityImage,
    inputs: *const SeizaFlatStabilityImage,
    input_count: usize,
    bias: *const SeizaFlatStabilityImage,
    dark: *const SeizaFlatStabilityImage,
    options_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let master = unsafe { master.as_ref() }.ok_or("master image is required")?;
        if input_count > isize::MAX as usize / std::mem::size_of::<SeizaFlatStabilityImage>()
            || (input_count > 0 && inputs.is_null())
        {
            return Err("input image descriptors are missing or their count overflows".into());
        }
        let inputs = if input_count == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(inputs, input_count) }
        };
        let options = if options_json.is_null() {
            Default::default()
        } else {
            let value: Value =
                serde_json::from_str(&required_str(options_json, "flat image options")?)
                    .map_err(|e| e.to_string())?;
            if field_names(field_names(value.clone(), false), true) != value {
                return Err("flat image option field names must be camelCase".into());
            }
            serde_json::from_value(field_names(value, false)).map_err(|e| e.to_string())?
        };
        let master = unsafe { cached_image(master) }?;
        let inputs = inputs
            .iter()
            .map(|image| unsafe { cached_image(image) })
            .collect::<Result<Vec<_>, _>>()?;
        let bias = unsafe { bias.as_ref() }
            .map(|image| unsafe { cached_image(image) })
            .transpose()?;
        let dark = unsafe { dark.as_ref() }
            .map(|image| unsafe { cached_image(image) })
            .transpose()?;
        let result = seiza_stacking::measure_flat_stability_images(
            &master,
            &inputs,
            bias.as_ref(),
            dark.as_ref(),
            options,
        )
        .map_err(|e| e.to_string())?;
        owned_json(&field_names(
            serde_json::to_value(result).map_err(|e| e.to_string())?,
            true,
        ))
    })
    .unwrap_or(ptr::null_mut())
}

/// Prepare a reusable grouping book from camelCase flat-group request JSON.
/// No file I/O. Free the returned handle with seiza_flat_groups_free.
/// # Safety
/// The request is NUL-terminated UTF-8; error_out is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_flat_groups_prepare_json(
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut SeizaFlatGroups {
    clear_error(error_out);
    ffi_result(error_out, || {
        let json = required_str(request_json, "flat group request")?;
        let value: Value = serde_json::from_str(&json).map_err(|e| e.to_string())?;
        if field_names(field_names(value.clone(), false), true) != value {
            return Err("flat group request field names must be camelCase".into());
        }
        let request =
            serde_json::from_value(field_names(value, false)).map_err(|e| e.to_string())?;
        let book = seiza_calibration::flat_response::groups::prepare(request)
            .map_err(|e| e.to_string())?;
        Ok(Box::into_raw(Box::new(SeizaFlatGroups {
            book: Arc::new(book),
        })))
    })
    .unwrap_or(ptr::null_mut())
}

/// Release an immutable grouping handle; null is allowed.
/// # Safety
/// A non-null pointer must be a live handle returned by prepare and freed once,
/// with no concurrent call still borrowing it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_flat_groups_free(groups: *mut SeizaFlatGroups) {
    if !groups.is_null() {
        drop(unsafe { Box::from_raw(groups) });
    }
}

// Settings are keyed records. Their identifiers and JSON values are opaque;
// only schema fields, including those in measurements/grouping, change case.
pub(crate) fn field_names(value: Value, camel: bool) -> Value {
    match value {
        Value::Array(values) => {
            Value::Array(values.into_iter().map(|v| field_names(v, camel)).collect())
        }
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(key, value)| {
                    let opaque = key == "value";
                    let mut name = String::new();
                    let mut upper = false;
                    for ch in key.chars() {
                        if camel && ch == '_' {
                            upper = true;
                        } else if !camel && ch.is_ascii_uppercase() {
                            name.push('_');
                            name.push(ch.to_ascii_lowercase());
                        } else {
                            name.push(if upper { ch.to_ascii_uppercase() } else { ch });
                            upper = false;
                        }
                    }
                    (
                        name,
                        if opaque {
                            value
                        } else {
                            field_names(value, camel)
                        },
                    )
                })
                .collect(),
        ),
        value => value,
    }
}

/// The JSON Schema (draft 2020-12) of the calibration library's JSON calls:
/// `calls` maps each export to the schema of its request and of its answer,
/// and `$defs` holds the types they share. Property names are camelCase, as
/// the calls take and return them; enum values are as given. Rust hosts call
/// the native API instead, and Python binds it. Free the result or error
/// with seiza_string_free.
///
/// # Safety
/// When non-null, `error_out` must point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_calibration_library_schema_json(
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || owned_json(&library_schema())).unwrap_or(ptr::null_mut())
}

/// The document [`seiza_calibration_library_schema_json`] returns.
pub(crate) fn library_schema() -> Value {
    use library::json as calls;
    use seiza_calibration::flat_response::groups;
    let mut generator = schemars::generate::SchemaSettings::draft2020_12().into_generator();
    // A call that takes no request has none; one that returns a handle has
    // no answer.
    let call = |request: Option<schemars::Schema>, answer: Option<schemars::Schema>| serde_json::json!({ "request": request, "answer": answer });
    let mut documented = serde_json::Map::new();
    let mut add = |name: &str, entry: Value| {
        documented.insert(name.to_string(), entry);
    };
    add(
        "seiza_calibration_settings_schema_json",
        call(
            None,
            Some(generator.subschema_for::<Vec<library::rules::FieldSpec>>()),
        ),
    );
    add(
        "seiza_calibration_setting_check_json",
        call(
            Some(generator.subschema_for::<calls::Setting>()),
            Some(generator.subschema_for::<library::rules::SettingCheck>()),
        ),
    );
    for name in [
        "seiza_calibration_library_plan_json",
        "seiza_calibration_library_plan_with_groups_json",
    ] {
        add(
            name,
            call(
                Some(generator.subschema_for::<calls::Request>()),
                Some(generator.subschema_for::<calls::Snapshot>()),
            ),
        );
    }
    add(
        "seiza_calibration_flat_response_plan_json",
        call(
            Some(generator.subschema_for::<calls::Request>()),
            Some(generator.subschema_for::<calls::FlatResponsePlans>()),
        ),
    );
    add(
        "seiza_calibration_master_plan_json",
        call(
            Some(generator.subschema_for::<calls::MasterPlanRequest>()),
            Some(generator.subschema_for::<library::Session>()),
        ),
    );
    add(
        "seiza_calibration_master_cohort_json",
        call(
            Some(generator.subschema_for::<calls::MasterCohortRequest>()),
            Some(generator.subschema_for::<calls::MasterCohort>()),
        ),
    );
    add(
        "seiza_calibration_flat_retry_json",
        call(
            Some(generator.subschema_for::<calls::FlatRetryRequest>()),
            Some(generator.subschema_for::<library::FlatRetryDecision>()),
        ),
    );
    add(
        "seiza_calibration_flat_work_plan_json",
        call(
            Some(generator.subschema_for::<calls::FlatWorkRequest>()),
            Some(generator.subschema_for::<calls::FlatWorkPlans>()),
        ),
    );
    add(
        "seiza_calibration_flat_work_finish_json",
        call(
            Some(generator.subschema_for::<calls::FlatWorkAnswer>()),
            Some(generator.subschema_for::<calls::FlatWorkResult>()),
        ),
    );
    add(
        "seiza_flat_groups_prepare_json",
        call(Some(generator.subschema_for::<groups::Request>()), None),
    );
    add(
        "seiza_flat_group_plan_json",
        call(
            Some(generator.subschema_for::<groups::Request>()),
            Some(generator.subschema_for::<calls::FlatGroupPlan>()),
        ),
    );
    add(
        "seiza_master_flat_stability_json",
        call(
            Some(generator.subschema_for::<seiza_stacking::FlatStabilityFileRequest>()),
            Some(generator.subschema_for::<seiza_stacking::FlatSetStability>()),
        ),
    );
    add(
        "seiza_master_flat_stability_images_json",
        call(
            Some(generator.subschema_for::<seiza_stacking::FlatStabilityImageOptions>()),
            Some(generator.subschema_for::<seiza_stacking::FlatSetStability>()),
        ),
    );
    let definitions = generator.take_definitions(true);
    schema_names(serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Seiza calibration library C ABI",
        "calls": documented,
        "$defs": definitions,
    }))
}

/// A schema with its property names as the C ABI spells them: every key of
/// a `properties` object, and every name `required` lists, in camelCase.
fn schema_names(value: Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.into_iter().map(schema_names).collect()),
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(key, value)| {
                    let value = match (key.as_str(), value) {
                        ("properties", Value::Object(properties)) => Value::Object(
                            properties
                                .into_iter()
                                .map(|(name, schema)| (camel_case(&name), schema_names(schema)))
                                .collect(),
                        ),
                        ("required", Value::Array(names)) => Value::Array(
                            names
                                .into_iter()
                                .map(|name| match name {
                                    Value::String(name) => Value::String(camel_case(&name)),
                                    other => other,
                                })
                                .collect(),
                        ),
                        (_, value) => schema_names(value),
                    };
                    (key, value)
                })
                .collect(),
        ),
        value => value,
    }
}

fn camel_case(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut upper = false;
    for ch in name.chars() {
        if ch == '_' {
            upper = true;
        } else if upper {
            out.push(ch.to_ascii_uppercase());
            upper = false;
        } else {
            out.push(ch);
        }
    }
    out
}

/// The calibration settings a library request takes, in the order a settings
/// panel shows them: each key, group, label, help, effect, kind with its range
/// and default. camelCase JSON array. Free the result or error with
/// seiza_string_free.
///
/// # Safety
/// When non-null, `error_out` must point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_calibration_settings_schema_json(
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        owned_json(&field_names(
            serde_json::to_value(library::rules::fields()).map_err(|e| e.to_string())?,
            true,
        ))
    })
    .unwrap_or(ptr::null_mut())
}

/// Judge one setting value as a settings panel saves it. `request_json` is
/// camelCase `{key, value}`; the value stays as given. Returns camelCase
/// `{key, valid, error, text, default}`: whether it can be saved, why not, the
/// value in words, and whether it means the default. Free the result or
/// error with seiza_string_free.
///
/// # Safety
/// `request_json` must point to NUL-terminated UTF-8. When non-null,
/// `error_out` must point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_calibration_setting_check_json(
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let json = required_str(request_json, "setting")?;
        let setting: library::json::Setting =
            serde_json::from_str(&json).map_err(|e| format!("invalid setting: {e}"))?;
        let check = library::rules::check_setting(&setting.key, &setting.value);
        owned_json(&field_names(
            serde_json::to_value(check).map_err(|e| e.to_string())?,
            true,
        ))
    })
    .unwrap_or(ptr::null_mut())
}

/// Grade and select a calibration library using PSF Guard's shared policy.
/// `request_json` is a camelCase library::json::Request: frames, lights, quality,
/// globalSettings, rigs, grouping, availableResponses, nightShifts, regrade, masterFailures, pedestalEvidence,
/// forceFlatWithoutPedestal, flatStarMasking, flatStability, flatAssociation and darkAssociation.
/// Missing collections are empty. Settings are {key,value} records; identifiers,
/// enum values and arbitrary setting values are preserved. The returned snapshot
/// contains grades, selections, complete gates, sessions, assignments and master
/// dependency recipes. This pure call opens no files and builds no master pixels.
/// Hosts verify sources and execute recipes through the native master builders.
/// See docs/calibration-library.md for provenance, units and execution contracts.
/// Free the result or error with seiza_string_free.
///
/// # Safety
/// `request_json` must point to NUL-terminated UTF-8. When non-null, `error_out`
/// must point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_calibration_library_plan_json(
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    unsafe { seiza_calibration_library_plan_with_groups_json(request_json, ptr::null(), error_out) }
}

/// Plan using a reusable grouping handle, or null to build from request rows.
/// Non-null handles require empty grouping rows and matching grouping limits.
/// # Safety
/// The request/error pointers follow seiza_calibration_library_plan_json's
/// contract; a non-null groups pointer must remain valid throughout this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_calibration_library_plan_with_groups_json(
    request_json: *const c_char,
    groups: *const SeizaFlatGroups,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let json = required_str(request_json, "calibration library request")?;
        let value: Value = serde_json::from_str(&json).map_err(|e| e.to_string())?;
        if field_names(field_names(value.clone(), false), true) != value {
            return Err("calibration library request field names must be camelCase".into());
        }
        let request: library::json::Request = serde_json::from_value(field_names(value, false))
            .map_err(|e| format!("invalid calibration library request: {e}"))?;
        let result = match unsafe { groups.as_ref() } {
            Some(groups) => library::json::evaluate_with_book(request, Arc::clone(&groups.book)),
            None => library::json::evaluate(request),
        }
        .map_err(|e| e.to_string())?;
        owned_json(&field_names(
            serde_json::to_value(result).map_err(|e| e.to_string())?,
            true,
        ))
    })
    .unwrap_or(ptr::null_mut())
}

/// Resolve verified selected candidates, build options and session identity from camelCase JSON. No I/O.
/// Free the result/error with seiza_string_free.
/// # Safety
/// request_json is NUL-terminated UTF-8; error_out is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_calibration_master_plan_json(
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let json = required_str(request_json, "master policy request")?;
        let value: Value = serde_json::from_str(&json).map_err(|e| e.to_string())?;
        if field_names(field_names(value.clone(), false), true) != value {
            return Err("master policy field names must be camelCase".into());
        }
        let request: library::json::MasterPlanRequest =
            serde_json::from_value(field_names(value, false)).map_err(|e| e.to_string())?;
        let result = library::json::plan_masters(request).map_err(|e| e.to_string())?;
        owned_json(&field_names(
            serde_json::to_value(result).map_err(|e| e.to_string())?,
            true,
        ))
    })
    .unwrap_or(ptr::null_mut())
}

/// Light choices may use exact full paths; these take precedence over basename
/// entries, preserving existing catalog lookup for other lights.
/// Plan raw-flat response sessions and bounded samples from a camelCase
/// library::json::Request. Returns plans, resolved limits/night boundaries, grades
/// and unplanned frame IDs. Pure metadata/measurement policy; no pixel I/O.
/// flatAssociation/darkAssociation accept automatic (default) or one_set in
/// library and master requests. OneSet keeps all eligible compatible inputs;
/// incompatible supplied sets return an error instead of silently splitting.
/// Free the result/error with seiza_string_free.
/// # Safety
/// request_json is NUL-terminated UTF-8; error_out is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_calibration_flat_response_plan_json(
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let json = required_str(request_json, "flat response plan request")?;
        let value: Value = serde_json::from_str(&json).map_err(|e| e.to_string())?;
        if field_names(field_names(value.clone(), false), true) != value {
            return Err("flat response plan field names must be camelCase".into());
        }
        let request: library::json::Request =
            serde_json::from_value(field_names(value, false)).map_err(|e| e.to_string())?;
        let result = library::json::plan_flat_responses(request).map_err(|e| e.to_string())?;
        owned_json(&field_names(
            serde_json::to_value(result).map_err(|e| e.to_string())?,
            true,
        ))
    })
    .unwrap_or(ptr::null_mut())
}

fn work_json<T: serde::de::DeserializeOwned, R: serde::Serialize>(
    request: *const c_char,
    error_out: *mut *mut c_char,
    operation: impl FnOnce(T) -> seiza_calibration::Result<R>,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let value: Value = serde_json::from_str(&required_str(request, "flat workflow request")?)
            .map_err(|e| e.to_string())?;
        if field_names(field_names(value.clone(), false), true) != value {
            return Err("flat workflow field names must be camelCase".into());
        }
        let request =
            serde_json::from_value(field_names(value, false)).map_err(|e| e.to_string())?;
        let result = operation(request).map_err(|e| e.to_string())?;
        owned_json(&field_names(
            serde_json::to_value(result).map_err(|e| e.to_string())?,
            true,
        ))
    })
    .unwrap_or(ptr::null_mut())
}
/// Plan light evidence, correction inputs and manual choice resolution from
/// camelCase FlatWorkRequest {library,lightContext}. Pure cached-record policy;
/// no file/header/pixel I/O. Results identify bounded samples and response keys.
/// Free result/error with seiza_string_free.
/// # Safety
/// request_json is NUL-terminated UTF-8; error_out is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_calibration_flat_work_plan_json(
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    work_json(request_json, error_out, library::json::plan_flat_work)
}
/// Finalize native family-choice/change-fit feedback as a cached evidence row.
/// camelCase FlatWorkAnswer is kind=family {question,choice,lights} or
/// kind=change {question,fits}; the core resolves ties and aggregate verdicts.
/// # Safety
/// request_json is NUL-terminated UTF-8; error_out is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_calibration_flat_work_finish_json(
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    work_json(request_json, error_out, library::json::finish_flat_work)
}

/// Resolve a coherent master cohort from camelCase JSON. No I/O.
/// Free the result/error with seiza_string_free.
/// # Safety
/// request_json is NUL-terminated UTF-8; error_out is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_calibration_master_cohort_json(
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let json = required_str(request_json, "master policy request")?;
        let value: Value = serde_json::from_str(&json).map_err(|e| e.to_string())?;
        if field_names(field_names(value.clone(), false), true) != value {
            return Err("master policy field names must be camelCase".into());
        }
        let request: library::json::MasterCohortRequest =
            serde_json::from_value(field_names(value, false)).map_err(|e| e.to_string())?;
        let result = library::json::master_cohort(request).map_err(|e| e.to_string())?;
        owned_json(&field_names(
            serde_json::to_value(result).map_err(|e| e.to_string())?,
            true,
        ))
    })
    .unwrap_or(ptr::null_mut())
}

/// Decide flat trimming, alternate-session retry and publication from camelCase JSON. No I/O.
/// Free the result/error with seiza_string_free.
/// # Safety
/// request_json is NUL-terminated UTF-8; error_out is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_calibration_flat_retry_json(
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let json = required_str(request_json, "master policy request")?;
        let value: Value = serde_json::from_str(&json).map_err(|e| e.to_string())?;
        if field_names(field_names(value.clone(), false), true) != value {
            return Err("master policy field names must be camelCase".into());
        }
        let request: library::json::FlatRetryRequest =
            serde_json::from_value(field_names(value, false)).map_err(|e| e.to_string())?;
        let result = library::json::flat_retry(request).map_err(|e| e.to_string())?;
        owned_json(&field_names(
            serde_json::to_value(result).map_err(|e| e.to_string())?,
            true,
        ))
    })
    .unwrap_or(ptr::null_mut())
}

/// Measure master-flat feature stability from files, one source at a time.
/// camelCase JSON: master, inputs [{path,exposureSeconds,skipped}], optional
/// bias/dark/darkExposureSeconds and normalizedFullScale. Missing/unreadable or
/// incompatible inputs count as drifting; skipped inputs have zero disagreement.
/// Free the returned JSON/error with seiza_string_free.
/// # Safety
/// request_json is NUL-terminated UTF-8; error_out is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_master_flat_stability_json(
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let json = required_str(request_json, "flat stability request")?;
        let value: Value = serde_json::from_str(&json).map_err(|e| e.to_string())?;
        // Nested input fields follow the same camelCase contract as the outer request.
        if field_names(field_names(value.clone(), false), true) != value {
            return Err("flat stability field names must be camelCase".into());
        }
        let normalized = field_names(value, false);
        let request: seiza_stacking::FlatStabilityFileRequest =
            serde_json::from_value(normalized).map_err(|e| e.to_string())?;
        let result =
            seiza_stacking::measure_flat_stability_files(request).map_err(|e| e.to_string())?;
        owned_json(&field_names(
            serde_json::to_value(result).map_err(|e| e.to_string())?,
            true,
        ))
    })
    .unwrap_or(ptr::null_mut())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{CStr, CString};
    fn run(value: Value) -> Result<Value, String> {
        let json = CString::new(value.to_string()).unwrap();
        let mut error = ptr::null_mut();
        let output = unsafe { seiza_calibration_library_plan_json(json.as_ptr(), &mut error) };
        let (pointer, failed) = if output.is_null() {
            (error, true)
        } else {
            assert!(error.is_null());
            (output, false)
        };
        assert!(!pointer.is_null());
        let text = unsafe { CStr::from_ptr(pointer) }
            .to_str()
            .unwrap()
            .to_owned();
        unsafe { crate::seiza_string_free(pointer) };
        if failed {
            Err(text)
        } else {
            Ok(serde_json::from_str(&text).unwrap())
        }
    }
    #[test]
    fn psf_guard_reference_crosses_the_c_abi() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../seiza-calibration/tests/fixtures/library/psf-guard.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let result = run(field_names(case["request"].clone(), true)).unwrap();
            let expected = field_names(case["expected"].clone(), true);
            for key in [
                "bias",
                "dark",
                "darkFlat",
                "flat",
                "keptOut",
                "tolerances",
                "flatBasis",
                "flatFamily",
            ] {
                assert_eq!(
                    result["selections"][0][key], expected[key],
                    "{}: {key}",
                    case["name"]
                );
            }
            assert_eq!(result["assignments"], serde_json::json!([0, 0]));
        }
    }
    #[test]
    fn captured_grading_and_fitted_recipe_fields_cross_the_c_abi() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../seiza-calibration/tests/fixtures/library/grading.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let result = run(field_names(case["request"].clone(), true)).unwrap();
            for (grade, expected) in result["grades"]
                .as_array()
                .unwrap()
                .iter()
                .zip(case["grades"].as_array().unwrap())
            {
                assert_eq!(grade["judged"]["grade"], expected["grade"]);
                assert_eq!(grade["keptOut"], expected["kept_out"]);
            }
        }
        let fixture: Value = serde_json::from_str(include_str!(
            "../../seiza-calibration/tests/fixtures/library/psf-guard.json"
        ))
        .unwrap();
        let mut request = field_names(fixture["cases"][9]["request"].clone(), true);
        let key = run(request.clone()).unwrap()["sessions"][0]["key"].clone();
        request["pedestalEvidence"] = serde_json::json!([{"sessionKey":key,"estimates":[300.0]}]);
        let result = run(request.clone()).unwrap();
        assert_eq!(
            result["sessions"][0]["application"]["flatUse"],
            "fitted_pedestal"
        );
        assert_eq!(result["sessions"][0]["application"]["pedestal"], 300.0);
        request["masterFailures"] =
            serde_json::json!([{"sessionKey":key,"kind":"flat","reason":"read failed"}]);
        let result = run(request).unwrap();
        assert_eq!(
            result["sessions"][0]["application"]["flatUse"],
            "unavailable"
        );
        assert_eq!(result["sessions"][0]["recipes"][3]["action"], "failed");
    }

    #[test]
    fn resolved_recipe_regressions_cross_the_c_abi() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../seiza-calibration/tests/fixtures/library/recipes.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let result = run(field_names(case["request"].clone(), true)).unwrap();
            assert_eq!(
                result["assignments"], case["assignments"],
                "{}",
                case["name"]
            );
            assert_eq!(
                result["sessions"].as_array().unwrap().len(),
                case["recipes"].as_array().unwrap().len()
            );
            for (session, expected) in result["sessions"]
                .as_array()
                .unwrap()
                .iter()
                .zip(case["recipes"].as_array().unwrap())
            {
                let recipe = session["recipes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|r| r["kind"] == case["kind"])
                    .unwrap();
                let expected = field_names(expected.clone(), true);
                for key in ["action", "inputs", "exposureSeconds"] {
                    assert_eq!(recipe[key], expected[key], "{}: {key}", case["name"]);
                }
                assert_eq!(recipe["requiresSourceVerification"], false);
            }
        }
    }
    #[test]
    fn prepared_grouping_handle_matches_value_planning() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../seiza-calibration/tests/fixtures/library/psf-guard.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let mut request = field_names(case["request"].clone(), true);
            let expected = run(request.clone()).unwrap();
            let mut grouping = request
                .as_object_mut()
                .unwrap()
                .remove("grouping")
                .unwrap_or(serde_json::json!({}));
            let rigs: std::collections::BTreeSet<_> = grouping["responses"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|r| r["rigUuid"].as_str().unwrap())
                .collect();
            let limits: Vec<_> = rigs.into_iter().map(|rig| serde_json::json!({"rigUuid":rig,"limits":{"measured":true,"dust":0.01,"broad":0.02,"window":86400,"rotation":2.0}})).collect();
            grouping["limits"] = serde_json::json!(limits);
            let grouping = CString::new(grouping.to_string()).unwrap();
            let mut error = ptr::null_mut();
            let handle = unsafe { seiza_flat_groups_prepare_json(grouping.as_ptr(), &mut error) };
            assert!(!handle.is_null());
            assert!(error.is_null());
            let request = CString::new(request.to_string()).unwrap();
            let out = unsafe {
                seiza_calibration_library_plan_with_groups_json(
                    request.as_ptr(),
                    handle,
                    &mut error,
                )
            };
            assert!(!out.is_null());
            assert!(error.is_null());
            let actual: Value =
                serde_json::from_str(unsafe { CStr::from_ptr(out) }.to_str().unwrap()).unwrap();
            assert_eq!(actual, expected);
            unsafe {
                crate::seiza_string_free(out);
                seiza_flat_groups_free(handle);
            }
        }
        unsafe {
            seiza_flat_groups_free(ptr::null_mut());
        }
    }
    #[test]
    fn schema_is_strict_but_setting_values_are_opaque() {
        let value = serde_json::json!({"globalSettings":[{"key":"future_Setting","value":{"user_key":"Preserve_This"}}]});
        assert_eq!(field_names(field_names(value.clone(), false), true), value);
        assert!(
            run(value)
                .unwrap_err()
                .contains("unknown setting 'future_Setting'")
        );
        assert!(run(serde_json::json!({"global_settings":[]})).is_err());
        assert!(run(serde_json::json!({"unknown":1})).is_err());
    }
}

#[cfg(test)]
mod master_policy_tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn cached_pixels_cross_the_c_abi_and_bad_lengths_are_rejected() {
        let mut master =
            seiza_stacking::LinearImage::new(128, 128, 1, vec![1.; 128 * 128]).unwrap();
        for y in 56..72 {
            for x in 56..72 {
                master.data[y * 128 + x] = 0.5;
            }
        }
        let clean = seiza_stacking::LinearImage::new(128, 128, 1, vec![1.; 128 * 128]).unwrap();
        let expected = seiza_stacking::measure_flat_stability_images(
            &master,
            &[master.clone(), clean.clone()],
            None,
            None,
            Default::default(),
        )
        .unwrap();
        let descriptor = |image: &seiza_stacking::LinearImage| SeizaFlatStabilityImage {
            data: image.data.as_ptr(),
            length: image.data.len(),
            width: image.width,
            height: image.height,
            channels: image.channels,
        };
        let master_buffer = descriptor(&master);
        let inputs = [descriptor(&master), descriptor(&clean)];
        let mut error = ptr::null_mut();
        let result = unsafe {
            seiza_master_flat_stability_images_json(
                &master_buffer,
                inputs.as_ptr(),
                inputs.len(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut error,
            )
        };
        assert!(error.is_null());
        assert!(!result.is_null());
        let value: Value = serde_json::from_str(
            unsafe { std::ffi::CStr::from_ptr(result) }
                .to_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            field_names(value, false),
            serde_json::to_value(expected).unwrap()
        );
        unsafe {
            crate::seiza_string_free(result);
        }
        let bad_master = SeizaFlatStabilityImage {
            length: 1,
            ..master_buffer
        };
        let result = unsafe {
            seiza_master_flat_stability_images_json(
                &bad_master,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut error,
            )
        };
        assert!(result.is_null());
        assert!(
            unsafe { std::ffi::CStr::from_ptr(error) }
                .to_str()
                .unwrap()
                .contains("length must match")
        );
        unsafe {
            crate::seiza_string_free(error);
        }
    }

    #[test]
    fn session_policy_and_strict_settings_cross_the_c_abi() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../seiza-calibration/tests/fixtures/library/flat-retries.json"
        ))
        .unwrap();
        let retry = &fixture["cases"][0]["request"];
        let session = unsafe {
            call(
                serde_json::json!({
                    "frames": retry["candidates"], "lights": [{}],
                    "flat_stability": {"enabled": false}, "flat_held_back": retry["held_back"],
                }),
                seiza_calibration_master_plan_json,
            )
        };
        let result = unsafe {
            call(
                serde_json::json!({
                    "session":session, "candidates":retry["candidates"], "first":retry["first"],
                }),
                seiza_calibration_flat_retry_json,
            )
        };
        assert_eq!(result["action"], "publish_first");
        for (field, function) in [
            (
                "globalSettings",
                seiza_calibration_master_plan_json as unsafe extern "C" fn(_, _) -> _,
            ),
            ("settings", seiza_calibration_master_cohort_json),
            ("settings", seiza_calibration_flat_retry_json),
        ] {
            let mut request = serde_json::json!({"lights":[{}]});
            if field == "settings" {
                request.as_object_mut().unwrap().remove("lights");
            }
            request[field] = serde_json::json!([{"key":"use_darks", "value":"false"}]);
            let request = CString::new(request.to_string()).unwrap();
            let mut error = ptr::null_mut();
            let result = unsafe { function(request.as_ptr(), &mut error) };
            assert!(result.is_null());
            assert!(
                unsafe { std::ffi::CStr::from_ptr(error) }
                    .to_str()
                    .unwrap()
                    .contains("use_darks")
            );
            unsafe {
                crate::seiza_string_free(error);
            }
        }
    }
    unsafe fn call(
        request: Value,
        function: unsafe extern "C" fn(*const c_char, *mut *mut c_char) -> *mut c_char,
    ) -> Value {
        let json = CString::new(field_names(request, true).to_string()).unwrap();
        let mut error = ptr::null_mut();
        let output = unsafe { function(json.as_ptr(), &mut error) };
        assert!(error.is_null());
        assert!(!output.is_null());
        let result = unsafe { CString::from_raw(output) };
        field_names(serde_json::from_slice(result.as_bytes()).unwrap(), false)
    }
    #[test]
    fn settings_schema_and_checks_follow_the_shared_table() {
        let mut error = ptr::null_mut();
        let output = unsafe { seiza_calibration_settings_schema_json(&mut error) };
        assert!(error.is_null());
        let schema: Value =
            serde_json::from_slice(unsafe { CString::from_raw(output) }.as_bytes()).unwrap();
        assert_eq!(
            schema.as_array().unwrap().len(),
            library::rules::fields().len()
        );
        assert!(schema[0].get("key").is_some());
        let check = |key: &str, value: Value| unsafe {
            call(
                serde_json::json!({"key": key, "value": value}),
                seiza_calibration_setting_check_json,
            )
        };
        let ok = check("rotation_tolerance_deg", serde_json::json!(3.0));
        assert_eq!(ok["valid"], true);
        assert_eq!(ok["text"], "3°");
        assert_eq!(ok["default"], false);
        let wide = check("rotation_tolerance_deg", serde_json::json!(1000.0));
        assert_eq!(wide["valid"], false);
        assert!(wide["error"].as_str().unwrap().contains("Flat rotation"));
        let unknown = check("from_the_future", serde_json::json!(true));
        assert_eq!(unknown["valid"], false);
        assert!(
            unknown["error"]
                .as_str()
                .unwrap()
                .contains("unknown setting")
        );
    }
    #[test]
    fn full_path_light_choices_do_not_cross_same_name_files() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../seiza-calibration/tests/fixtures/library/psf-guard.json"
        ))
        .unwrap();
        let mut request = fixture["cases"][13]["request"].clone();
        request["lights"][0]["path"] = serde_json::json!("/first/shared.fits");
        request["lights"][1] = request["lights"][0].clone();
        request["lights"][1]["path"] = serde_json::json!("/second/shared.fits");
        let first = unsafe { call(request.clone(), seiza_calibration_library_plan_json) };
        let event = &first["selections"][0]["flat_basis"]["event"];
        request["grouping"]["light_choices"] = serde_json::json!([{"light_id":0,"file_name":"/first/shared.fits","filter":"Ha","at":request["lights"][0]["timestamp"],"row":{"light_guid":"first","change_key":event["key"],"rig_uuid":request["lights"][0]["rig_uuid"],"change_start":event["start"],"change_end":event["end"],"choice":"before","group_frame":null,"group_session_key":null,"label":"before","chosen_at":1}}]);
        let result = unsafe { call(request, seiza_calibration_library_plan_json) };
        assert!(
            !result["selections"][0]["flat_choices"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert!(
            result["selections"][1]["flat_choices"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }
    #[test]
    fn cached_record_workflow_and_feedback_cross_the_c_abi() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../seiza-calibration/tests/fixtures/library/psf-guard.json"
        ))
        .unwrap();
        let mut library = fixture["cases"][12]["request"].clone();
        library["grouping"]["lights"] = serde_json::json!([]);
        let light = library["lights"][0].clone();
        library["lights"] = serde_json::json!(
            (0..12)
                .map(|i| {
                    let mut l = light.clone();
                    l["timestamp"] = serde_json::json!(345600 + i * 60);
                    l
                })
                .collect::<Vec<_>>()
        );
        let request = serde_json::json!({"library":library,"light_context":[{"light_index":0,"id":100,"target":"M42","pedestal_adu":123.0}]});
        let expected =
            library::json::plan_flat_work(serde_json::from_value(request.clone()).unwrap())
                .unwrap();
        let actual = unsafe { call(request, seiza_calibration_flat_work_plan_json) };
        assert_eq!(actual, serde_json::to_value(expected).unwrap());
        let question = actual["changes"][0].clone();
        let feedback = serde_json::json!({"kind":"change","question":question,"fits":[]});
        let row = unsafe { call(feedback, seiza_calibration_flat_work_finish_json) };
        assert_eq!(row["kind"], "change");
        assert_eq!(row["row"]["verdict"], "undecided");
    }
    #[test]
    fn response_preparation_and_supplied_sets_cross_the_c_abi() {
        let frames: Vec<_> = (0..80).map(|i| serde_json::json!({
            "frame_uuid":format!("flat-{i}"), "rig_uuid":"r", "kind":"flat",
            "source_path":format!("/flat-{i}.fits"), "source_fingerprint":format!("stamp-{i}"),
            "source_verified":true,"captured_at":1_700_000_000_i64+(i/20)*259200,
            "width":512,"height":512,"channels":1,"gain":100,"offset":20,
            "filter":"Ha","rotation":0.0,"exposure_s":30.0
        })).collect();
        let request = serde_json::json!({"frames":frames});
        let core =
            library::json::plan_flat_responses(serde_json::from_value(request.clone()).unwrap())
                .unwrap();
        assert_eq!(
            unsafe { call(request.clone(), seiza_calibration_flat_response_plan_json) },
            serde_json::to_value(core).unwrap()
        );
        let mut request = request;
        request["flat_association"] = serde_json::json!("one_set");
        request["dark_association"] = serde_json::json!("one_set");
        request["lights"] = serde_json::json!([{"path":"/light.fits","rig_uuid":"r","width":512,"height":512,"channels":1,"gain":100,"offset":20,"filter":"Ha","rotator_position":0.0,"exposure_s":30.0,"timestamp":1_700_000_000}]);
        let result = unsafe { call(request, seiza_calibration_library_plan_json) };
        let flat = result["sessions"][0]["recipes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["kind"] == "flat")
            .unwrap();
        assert_eq!(flat["inputs"].as_array().unwrap().len(), 80);
        assert_eq!(flat["input_limit"], 80);
    }
    #[test]
    fn native_flat_measurement_crosses_the_c_abi() {
        let dir = tempfile::tempdir().unwrap();
        let master_path = dir.path().join("master.fits");
        let input = dir.path().join("different.fits");
        let mut master =
            seiza_stacking::LinearImage::new(128, 128, 1, vec![1.0; 128 * 128]).unwrap();
        for y in 56..72 {
            for x in 56..72 {
                master.data[y * 128 + x] = 0.5;
            }
        }
        seiza_stacking::write_linear_image_fits_f32(&master_path, &master, &[], &[]).unwrap();
        let clean = seiza_stacking::LinearImage::new(128, 128, 1, vec![1.0; 128 * 128]).unwrap();
        seiza_stacking::write_linear_image_fits_f32(&input, &clean, &[], &[]).unwrap();
        let inputs = vec![seiza_stacking::FlatStabilityInput {
            path: input,
            exposure_seconds: None,
            skipped: false,
        }];
        let expected = seiza_stacking::measure_flat_stability(&inputs, &master, None, None, None);
        let request = serde_json::json!({"master": master_path, "inputs":inputs});
        assert_eq!(
            unsafe { call(request, seiza_master_flat_stability_json) },
            serde_json::to_value(expected).unwrap()
        );
    }
    #[test]
    fn shared_master_resolution_and_feedback_cross_the_c_abi() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../seiza-calibration/tests/fixtures/library/flat-retries.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let expected = serde_json::to_value(
                library::json::flat_retry(serde_json::from_value(case["request"].clone()).unwrap())
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(
                unsafe { call(case["request"].clone(), seiza_calibration_flat_retry_json) },
                expected
            );
        }
        let fixture: Value = serde_json::from_str(include_str!(
            "../../seiza-calibration/tests/fixtures/library/recipes.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let request = serde_json::json!({"frames": case["request"]["frames"], "lights": case["request"]["lights"], "global_settings": case["request"].get("global_settings").cloned().unwrap_or_else(|| serde_json::json!([]))});
            let expected = serde_json::to_value(
                library::json::plan_masters(serde_json::from_value(request.clone()).unwrap())
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(
                unsafe { call(request, seiza_calibration_master_plan_json) },
                expected
            );
            let request =
                serde_json::json!({"kind":case["kind"], "frames":case["request"]["frames"]});
            let expected = serde_json::to_value(
                library::json::master_cohort(serde_json::from_value(request.clone()).unwrap())
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(
                unsafe { call(request, seiza_calibration_master_cohort_json) },
                expected
            );
        }
    }
}

#[cfg(test)]
mod schema_tests {
    use super::*;
    use std::ffi::{CStr, CString};

    const SCHEMA_FILE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/include/seiza_calibration_library.schema.json"
    );
    const SCHEMA_URL: &str = "https://seiza.invalid/calibration-library.schema.json";

    type Export = unsafe extern "C" fn(*const c_char, *mut *mut c_char) -> *mut c_char;

    fn call(export: Export, request: &Value) -> Result<Value, String> {
        let json = CString::new(request.to_string()).unwrap();
        let mut error = ptr::null_mut();
        let output = unsafe { export(json.as_ptr(), &mut error) };
        let pointer = if output.is_null() { error } else { output };
        let text = unsafe { CStr::from_ptr(pointer) }
            .to_str()
            .unwrap()
            .to_owned();
        unsafe { crate::seiza_string_free(pointer) };
        if output.is_null() {
            Err(text)
        } else {
            Ok(serde_json::from_str(&text).unwrap())
        }
    }

    #[test]
    fn the_checked_in_schema_is_the_one_the_calls_take() {
        let generated = serde_json::to_string_pretty(&library_schema()).unwrap() + "\n";
        if std::env::var_os("SEIZA_UPDATE_SCHEMA").is_some() {
            std::fs::write(SCHEMA_FILE, &generated).unwrap();
        }
        // A Windows checkout may give the file CRLF line endings.
        let kept = std::fs::read_to_string(SCHEMA_FILE)
            .unwrap_or_default()
            .replace("\r\n", "\n");
        assert!(
            kept == generated,
            "the calibration library's JSON changed: run `SEIZA_UPDATE_SCHEMA=1 cargo test -p \
             seiza-cabi schema` and commit include/seiza_calibration_library.schema.json"
        );
    }

    /// Every fixture request, as C sends it, against the schema of the call
    /// it goes to, and the call's answer against the answer's schema. A
    /// request the schema refuses must be one the call refuses too.
    #[test]
    fn fixture_requests_and_answers_follow_the_schema() {
        let mut compiler = boon::Compiler::new();
        compiler.add_resource(SCHEMA_URL, library_schema()).unwrap();
        let mut schemas = boon::Schemas::new();
        let mut compile = |call: &str, part: &str| {
            compiler
                .compile(&format!("{SCHEMA_URL}#/calls/{call}/{part}"), &mut schemas)
                .unwrap_or_else(|error| panic!("{call} {part}: {error}"))
        };
        let calls: [(&str, Export, &str); 5] = [
            (
                "seiza_calibration_library_plan_json",
                seiza_calibration_library_plan_json,
                "psf-guard.json",
            ),
            (
                "seiza_calibration_library_plan_json",
                seiza_calibration_library_plan_json,
                "grading.json",
            ),
            (
                "seiza_calibration_library_plan_json",
                seiza_calibration_library_plan_json,
                "recipes.json",
            ),
            (
                "seiza_calibration_flat_response_plan_json",
                seiza_calibration_flat_response_plan_json,
                "psf-guard.json",
            ),
            (
                "seiza_calibration_flat_retry_json",
                seiza_calibration_flat_retry_json,
                "flat-retries.json",
            ),
        ];
        let indices: Vec<_> = calls
            .iter()
            .map(|(name, _, _)| (compile(name, "request"), compile(name, "answer")))
            .collect();
        let plan_request = indices[0].0;
        let group_request = compile("seiza_flat_group_plan_json", "request");
        let group_answer = compile("seiza_flat_group_plan_json", "answer");
        let mut checked = 0;
        for ((name, export, fixture), (request_schema, answer_schema)) in calls.iter().zip(indices)
        {
            let path = format!(
                "{}/../seiza-calibration/tests/fixtures/library/{fixture}",
                env!("CARGO_MANIFEST_DIR")
            );
            let cases: Value =
                serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            for case in cases["cases"].as_array().unwrap() {
                let request = field_names(case["request"].clone(), true);
                let fits = schemas.validate(&request, request_schema);
                match (fits, call(*export, &request)) {
                    (_, Ok(answer)) => {
                        if let Err(error) = schemas.validate(&request, request_schema) {
                            panic!(
                                "{name} {fixture} {}: the call takes a request its schema refuses: {error}",
                                case["name"]
                            );
                        }
                        if let Err(error) = schemas.validate(&answer, answer_schema) {
                            panic!(
                                "{name} {fixture} {}: its answer does not follow the schema: {error}",
                                case["name"]
                            );
                        }
                        checked += 1;
                    }
                    (Ok(()), Err(_)) | (Err(_), Err(_)) => {}
                }
                if *name == "seiza_calibration_library_plan_json"
                    && !case["request"]["grouping"].is_null()
                {
                    let grouping = field_names(case["request"]["grouping"].clone(), true);
                    if let Ok(answer) =
                        call(crate::flat_groups::seiza_flat_group_plan_json, &grouping)
                    {
                        schemas.validate(&grouping, group_request).unwrap();
                        schemas.validate(&answer, group_answer).unwrap();
                    }
                }
            }
        }
        assert!(checked >= 30, "only {checked} answers were checked");
        // The schema holds C to its own spelling and to known fields.
        let refused = |request: Value| schemas.validate(&request, plan_request).is_err();
        assert!(refused(serde_json::json!({"global_settings": []})));
        assert!(refused(serde_json::json!({"frames": [], "bogus": 1})));
        assert!(!refused(
            serde_json::json!({"globalSettings": [], "frames": []})
        ));
    }

    #[test]
    fn settings_calls_follow_the_schema() {
        let mut compiler = boon::Compiler::new();
        compiler.add_resource(SCHEMA_URL, library_schema()).unwrap();
        let mut schemas = boon::Schemas::new();
        let mut compile = |part: &str, call: &str| {
            compiler
                .compile(&format!("{SCHEMA_URL}#/calls/{call}/{part}"), &mut schemas)
                .unwrap()
        };
        let table = compile("answer", "seiza_calibration_settings_schema_json");
        let request = compile("request", "seiza_calibration_setting_check_json");
        let answer = compile("answer", "seiza_calibration_setting_check_json");
        let mut error = ptr::null_mut();
        let output = unsafe { seiza_calibration_settings_schema_json(&mut error) };
        let fields: Value =
            serde_json::from_str(unsafe { CStr::from_ptr(output) }.to_str().unwrap()).unwrap();
        unsafe { crate::seiza_string_free(output) };
        schemas.validate(&fields, table).unwrap();
        for setting in [
            serde_json::json!({"key": "rotationToleranceDeg", "value": 3.0}),
            serde_json::json!({"key": "rotation_tolerance_deg", "value": 3.0}),
            serde_json::json!({"key": "from_the_future", "value": true}),
        ] {
            schemas.validate(&setting, request).unwrap();
            let checked = call(seiza_calibration_setting_check_json, &setting).unwrap();
            schemas.validate(&checked, answer).unwrap();
        }
    }
}
