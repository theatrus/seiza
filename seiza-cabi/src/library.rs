//! The C ABI translates the shared library policy's immutable value snapshot.
use super::{clear_error, ffi_result, owned_json, required_str};
use seiza_calibration::library;
use serde_json::Value;
use std::ffi::c_char;
use std::ptr;

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
/// forceFlatWithoutPedestal and flatStarMasking.
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
    clear_error(error_out);
    ffi_result(error_out, || {
        let json = required_str(request_json, "calibration library request")?;
        let value: Value = serde_json::from_str(&json).map_err(|e| e.to_string())?;
        if field_names(field_names(value.clone(), false), true) != value {
            return Err("calibration library request field names must be camelCase".into());
        }
        let request: library::Request = serde_json::from_value(field_names(value, false))
            .map_err(|e| format!("invalid calibration library request: {e}"))?;
        let result = library::evaluate(request).map_err(|e| e.to_string())?;
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
    fn schema_is_strict_but_setting_values_are_opaque() {
        let value = serde_json::json!({"globalSettings":[{"key":"future_Setting","value":{"user_key":"Preserve_This"}}]});
        assert_eq!(field_names(field_names(value.clone(), false), true), value);
        assert!(run(value).is_ok());
        assert!(run(serde_json::json!({"global_settings":[]})).is_err());
        assert!(run(serde_json::json!({"unknown":1})).is_err());
    }
}
