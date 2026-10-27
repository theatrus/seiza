use super::*;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AnalysisRequest {
    inputs: Vec<PathBuf>,
    #[serde(default = "default_samples")]
    samples: usize,
    #[serde(default)]
    filter: Option<String>,
    #[serde(default)]
    max_frames: Option<usize>,
}
fn default_samples() -> usize {
    262_144
}

// Translate only JSON member names. Paths, filter values, warnings, and other
// strings are untouched. The scientific result is serialized by the core.
fn keys(value: serde_json::Value, camel: bool) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => serde_json::Value::Object(
            map.into_iter()
                .map(|(key, value)| {
                    let mut translated = String::new();
                    let mut upper = false;
                    for c in key.chars() {
                        if camel && c == '_' {
                            upper = true;
                        } else if camel && upper {
                            translated.extend(c.to_uppercase());
                            upper = false;
                        } else if !camel && c.is_ascii_uppercase() {
                            translated.push('_');
                            translated.push(c.to_ascii_lowercase());
                        } else {
                            translated.push(c);
                        }
                    }
                    (translated, keys(value, camel))
                })
                .collect(),
        ),
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.into_iter().map(|v| keys(v, camel)).collect())
        }
        other => other,
    }
}

/// Analyze mono FITS/XISF inputs without stacking. Request: inputs (paths,
/// directories or text lists), samples (8192..2097152, default 262144), filter
/// (optional), maxFrames (optional). Returns the core report with camelCase
/// keys. Individual frame failures remain in the report. Free with seiza_string_free.
///
/// # Safety
/// request_json must be NUL-terminated UTF-8. error_out must be null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_analyze_project_json(
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let request: AnalysisRequest = parse_request(request_json, "project analysis")?;
        let report = seiza_analysis::analyze(
            &request.inputs,
            &seiza_analysis::AnalysisOptions {
                samples: request.samples,
                filter: request.filter,
                max_frames: request.max_frames,
            },
        )
        .map_err(|e| e.to_string())?;
        owned_json(&keys(
            serde_json::to_value(report).map_err(|e| e.to_string())?,
            true,
        ))
    })
    .unwrap_or(ptr::null_mut())
}

/// Render the camelCase JSON report returned by seiza_analyze_project_json as
/// self-contained HTML. Returns owned UTF-8; free with seiza_string_free.
///
/// # Safety
/// report_json must be NUL-terminated UTF-8. error_out must be null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_analysis_html(
    report_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let value: serde_json::Value = parse_request(report_json, "analysis report")?;
        let report = serde_json::from_value(keys(value, false)).map_err(|e| e.to_string())?;
        let html = seiza_analysis::render_html(&report).map_err(|e| e.to_string())?;
        CString::new(html)
            .map(CString::into_raw)
            .map_err(|e| e.to_string())
    })
    .unwrap_or(ptr::null_mut())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn key_translation_roundtrips_without_touching_values() {
        let v = serde_json::json!({"schema_version":1,"frames":[{"capture_time":"A_B", "path":"my_file.FITS", "median_star_snr":12.0}]});
        assert_eq!(keys(v.clone(), true)["frames"][0]["medianStarSnr"], 12.0);
        assert_eq!(keys(keys(v.clone(), true), false), v);
    }
    #[test]
    fn entry_point_reports_invalid_options_and_missing_frame() {
        unsafe {
            let mut error = ptr::null_mut();
            let bad = CString::new(r#"{"inputs":[],"samples":0}"#).unwrap();
            assert!(seiza_analyze_project_json(bad.as_ptr(), &mut error).is_null());
            assert!(!error.is_null());
            seiza_string_free(error);
            let request = CString::new(r#"{"inputs":["does-not-exist.fits"]}"#).unwrap();
            let json = seiza_analyze_project_json(request.as_ptr(), &mut error);
            assert!(error.is_null());
            assert!(!json.is_null());
            let value: serde_json::Value =
                serde_json::from_str(CStr::from_ptr(json).to_str().unwrap()).unwrap();
            assert_eq!(value["schemaVersion"], 1);
            assert_eq!(value["frames"][0]["status"], "rejected");
            let html = seiza_analysis_html(json, &mut error);
            assert!(!html.is_null());
            assert!(error.is_null());
            seiza_string_free(html);
            seiza_string_free(json);
        }
    }
}
