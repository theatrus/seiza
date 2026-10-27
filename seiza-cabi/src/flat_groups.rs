//! Flat grouping uses the same batch evaluator as Rust and Python.
use super::{clear_error, ffi_result, owned_json, required_str};
use seiza_calibration::flat_response::groups;
use serde_json::Value;
use std::ffi::c_char;
use std::ptr;

// This wire schema contains only typed records and arrays. Keyed collections
// are arrays of pairs, so rig/frame/family identifiers are always values and
// are never renamed. Enum values likewise keep their native spelling.
pub(crate) fn field_names(value: Value, camel: bool) -> Value {
    match value {
        Value::Array(values) => {
            Value::Array(values.into_iter().map(|v| field_names(v, camel)).collect())
        }
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(key, value)| {
                    let mut name = String::new();
                    let mut uppercase = false;
                    for ch in key.chars() {
                        if camel && ch == '_' {
                            uppercase = true;
                        } else if !camel && ch.is_ascii_uppercase() {
                            name.push('_');
                            name.push(ch.to_ascii_lowercase());
                        } else {
                            name.push(if uppercase {
                                ch.to_ascii_uppercase()
                            } else {
                                ch
                            });
                            uppercase = false;
                        }
                    }
                    (name, field_names(value, camel))
                })
                .collect(),
        ),
        other => other,
    }
}

/// Evaluate measured flat families, groups, changes and light placements.
/// `request_json` is `flat_response::groups::Request` with camelCase field
/// names; omitted collections are empty. Responses, comparisons, evidence,
/// imaging sessions, rig limits, group overrides, night boundaries, epoch
/// samples, family evidence, light choices and light queries are arrays.
/// The returned `Snapshot` uses camelCase too. See the shared flat-group
/// schema documentation for records, units and index meanings.
///
/// This pure stage does not select darks, grade frames, or build masters.
/// Light queries must supply families that passed the host's hard matching
/// gates. Automatic choices and personal overrides are returned separately.
/// Returns null and sets `error_out` on failure. Free either returned string
/// with `seiza_string_free`.
///
/// # Safety
/// `request_json` must be a NUL-terminated UTF-8 string. When non-null,
/// `error_out` must point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_flat_group_plan_json(
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let json = required_str(request_json, "flat group request JSON")?;
        let value: Value = serde_json::from_str(&json).map_err(|error| error.to_string())?;
        if field_names(field_names(value.clone(), false), true) != value {
            return Err("flat group request field names must be camelCase".into());
        }
        let request: groups::Request = serde_json::from_value(field_names(value, false))
            .map_err(|error| format!("invalid flat group request: {error}"))?;
        let snapshot = groups::evaluate(request).map_err(|error| error.to_string())?;
        let value = serde_json::to_value(snapshot).map_err(|error| error.to_string())?;
        owned_json(&field_names(value, true))
    })
    .unwrap_or(ptr::null_mut())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{CStr, CString};

    fn run(value: Value) -> Result<Value, String> {
        let json = CString::new(serde_json::to_string(&value).unwrap()).unwrap();
        let mut error = ptr::null_mut();
        let out = unsafe { seiza_flat_group_plan_json(json.as_ptr(), &mut error) };
        if out.is_null() {
            let message = unsafe { CStr::from_ptr(error) }
                .to_str()
                .unwrap()
                .to_owned();
            unsafe {
                super::super::seiza_string_free(error);
            }
            return Err(message);
        }
        assert!(error.is_null());
        let value = serde_json::from_str(unsafe { CStr::from_ptr(out) }.to_str().unwrap()).unwrap();
        unsafe {
            super::super::seiza_string_free(out);
        }
        Ok(value)
    }

    #[test]
    fn psf_guard_snapshot_and_uncertainty_survive_the_c_boundary() {
        let input: Value = serde_json::from_str(include_str!(
            "../../seiza-calibration/tests/fixtures/flat-groups/request.json"
        ))
        .unwrap();
        let expected: Value = serde_json::from_str(include_str!(
            "../../seiza-calibration/tests/fixtures/flat-groups/expected.json"
        ))
        .unwrap();
        let actual = run(field_names(input, true)).unwrap();
        let actual = field_names(actual, false);
        for field in ["families", "boundaries", "events"] {
            assert_eq!(actual[field], expected[field]);
        }
        for (index, name) in ["before", "during", "after"].iter().enumerate() {
            assert_eq!(
                actual["lights"][index]["events"][0]["placement"],
                expected["placements"][0][name]
            );
        }
        assert!(
            run(serde_json::json!({"limitts": []}))
                .unwrap_err()
                .contains("unknown field")
        );
        assert!(run(serde_json::json!({"lights": [{"fileName":"x", "rigUuid":"rig", "filter":null, "at":9223372036854775807_i64, "angle":0, "compatibleFamilies":[], "rotationTolerance":2}]})).unwrap_err().contains("capture time"));
        let keys = serde_json::json!({"frame_uuid": "rig_with_underscores", "angle_terms": [["rig_with_underscores", {"pairs": 1}]]});
        assert_eq!(field_names(field_names(keys.clone(), true), false), keys);
    }
}
