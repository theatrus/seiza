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
fn field_names(value: Value, camel: bool) -> Value {
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

/// Grade and select a calibration library using PSF Guard's shared policy.
/// `request_json` is a camelCase library::Request: frames, lights, quality,
/// globalSettings, rigs, grouping, availableResponses, nightShifts, regrade, masterFailures, pedestalEvidence,
/// forceFlatWithoutPedestal, flatStarMasking and flatStability.
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
        let request: library::Request = serde_json::from_value(field_names(value, false))
            .map_err(|e| format!("invalid calibration library request: {e}"))?;
        let result = match unsafe { groups.as_ref() } {
            Some(groups) => library::evaluate_with_book(request, Arc::clone(&groups.book)),
            None => library::evaluate(request),
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
        let request: library::MasterPlanRequest =
            serde_json::from_value(field_names(value, false)).map_err(|e| e.to_string())?;
        let result = library::plan_masters(request).map_err(|e| e.to_string())?;
        owned_json(&field_names(
            serde_json::to_value(result).map_err(|e| e.to_string())?,
            true,
        ))
    })
    .unwrap_or(ptr::null_mut())
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
        let request: library::MasterCohortRequest =
            serde_json::from_value(field_names(value, false)).map_err(|e| e.to_string())?;
        let result = library::master_cohort(request).map_err(|e| e.to_string())?;
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
        let request: library::FlatRetryRequest =
            serde_json::from_value(field_names(value, false)).map_err(|e| e.to_string())?;
        let result = library::flat_retry(request).map_err(|e| e.to_string())?;
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
                library::flat_retry(serde_json::from_value(case["request"].clone()).unwrap())
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
                library::plan_masters(serde_json::from_value(request.clone()).unwrap()).unwrap(),
            )
            .unwrap();
            assert_eq!(
                unsafe { call(request, seiza_calibration_master_plan_json) },
                expected
            );
            let request =
                serde_json::json!({"kind":case["kind"], "frames":case["request"]["frames"]});
            let expected = serde_json::to_value(
                library::master_cohort(serde_json::from_value(request.clone()).unwrap()).unwrap(),
            )
            .unwrap();
            assert_eq!(
                unsafe { call(request, seiza_calibration_master_cohort_json) },
                expected
            );
        }
    }
}
