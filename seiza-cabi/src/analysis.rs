use super::*;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AnalysisRequest {
    inputs: Vec<PathBuf>,
    #[serde(default = "default_samples")]
    samples: usize,
    #[serde(default)]
    metrics_only: bool,
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
/// (optional), maxFrames (optional), metricsOnly (default false). Returns the core report with camelCase
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
                metrics_only: request.metrics_only,
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

/// Capture admitted-input history and actual stack settings as camelCase JSON.
/// Free the returned string with seiza_string_free.
/// # Safety
/// stacker must be a live handle; error_out must be null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_live_stacker_report_context_json(
    stacker: *const SeizaLiveStacker,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let stacker = unsafe { required_live_stacker(stacker)? };
        owned_json(&keys(
            serde_json::to_value(stacker.stacker.report_context()).map_err(|e| e.to_string())?,
            true,
        ))
    })
    .unwrap_or(ptr::null_mut())
}
fn report_context_json(
    context: *const c_char,
) -> Result<Option<seiza_stacking::StackReportContext>, String> {
    if context.is_null() {
        return Ok(None);
    }
    let value: serde_json::Value = parse_request(context, "stack report context")?;
    serde_json::from_value(keys(value, false))
        .map(Some)
        .map_err(|e| e.to_string())
}
/// Summarize this immutable snapshot as camelCase JSON. Optional context must
/// have been captured with this result. Free with seiza_string_free.
/// # Safety
/// snapshot must be live; context must be null or NUL-terminated UTF-8 JSON;
/// error_out must be null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_stack_snapshot_report_json(
    snapshot: *const SeizaStackSnapshot,
    context: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let snapshot = unsafe { snapshot.as_ref() }.ok_or("snapshot is null")?;
        let context = report_context_json(context)?;
        let report = seiza_stacking::stack_report(
            &snapshot.snapshot,
            context.as_ref(),
            snapshot.final_samples.as_deref(),
        )
        .map_err(|e| e.to_string())?;
        owned_json(&keys(
            serde_json::to_value(report).map_err(|e| e.to_string())?,
            true,
        ))
    })
    .unwrap_or(ptr::null_mut())
}
/// Summarize drizzle support as drop area times weight, never frame count.
/// Optional context must be captured with this result. Free with seiza_string_free.
/// # Safety
/// result must be live; context must be null or NUL-terminated UTF-8 JSON;
/// error_out must be null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_drizzle_result_report_json(
    result: *const SeizaDrizzleResult,
    context: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let result = unsafe { result.as_ref() }.ok_or("drizzle result is null")?;
        let context = report_context_json(context)?;
        let report = seiza_stacking::drizzle_report(
            &result.result,
            result.accepted_frames,
            result.rejected_frames,
            context.as_ref(),
            result.final_samples.as_deref(),
        )
        .map_err(|e| e.to_string())?;
        owned_json(&keys(
            serde_json::to_value(report).map_err(|e| e.to_string())?,
            true,
        ))
    })
    .unwrap_or(ptr::null_mut())
}
/// Render camelCase post-stack JSON as self-contained HTML. Free with seiza_string_free.
/// # Safety
/// report_json must be NUL-terminated UTF-8; error_out must be null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_stack_report_html(
    report_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let value: serde_json::Value = parse_request(report_json, "post-stack report")?;
        let report = serde_json::from_value(keys(value, false)).map_err(|e| e.to_string())?;
        let html = seiza_analysis::render_stack_html(&report).map_err(|e| e.to_string())?;
        CString::new(html)
            .map(CString::into_raw)
            .map_err(|e| e.to_string())
    })
    .unwrap_or(ptr::null_mut())
}

/// Render a camelCase schema-1 collection with title and stacks [{label, report}].
/// Every nested report is validated. Free the result with seiza_string_free.
/// # Safety
/// report_json must be NUL-terminated UTF-8; error_out must be null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_stack_collection_html(
    report_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let value: serde_json::Value = parse_request(report_json, "stack collection")?;
        let report = serde_json::from_value(keys(value, false)).map_err(|e| e.to_string())?;
        let html =
            seiza_analysis::render_stack_collection_html(&report).map_err(|e| e.to_string())?;
        CString::new(html)
            .map(CString::into_raw)
            .map_err(|e| e.to_string())
    })
    .unwrap_or(ptr::null_mut())
}

#[cfg(test)]
mod stack_report_tests {
    use super::*;
    #[test]
    fn completed_report_camel_case_roundtrip_and_null_errors() {
        unsafe {
            let mut error = ptr::null_mut();
            assert!(
                seiza_stack_snapshot_report_json(ptr::null(), ptr::null(), &mut error).is_null()
            );
            assert!(!error.is_null());
            seiza_string_free(error);
            let image = seiza_stacking::LinearImage {
                width: 2,
                height: 1,
                channels: 1,
                data: vec![1.0; 2],
            };
            let snapshot = SeizaStackSnapshot {
                snapshot: RustStackSnapshot {
                    image: image.clone(),
                    variance: image,
                    coverage: vec![2, 1],
                    rejected_samples: vec![0, 1],
                    accepted_frames: 2,
                    rejected_frames: 0,
                },
                final_samples: Some(vec![
                    seiza_stacking::BatchFrameDiagnostics {
                        finite_samples: 2,
                        integrated_samples: 1
                    };
                    2
                ]),
                reference_headers: vec![],
                input_paths: vec![],
            };
            let json = seiza_stack_snapshot_report_json(&snapshot, ptr::null(), &mut error);
            assert!(error.is_null());
            assert!(!json.is_null());
            let report: serde_json::Value =
                serde_json::from_str(CStr::from_ptr(json).to_str().unwrap()).unwrap();
            assert_eq!(report["supportMap"]["quantity"], "accepted_observations");
            assert_eq!(report["finalSamples"][0]["integratedSamples"], 1);
            for level in ["none", "cheap", "rich"] {
                let request = CString::new(
                    serde_json::json!({"report":report,"level":level,"inputs":["missing.fits"]})
                        .to_string(),
                )
                .unwrap();
                let prepared = seiza_prepare_stack_report_json(request.as_ptr(), &mut error);
                assert!(error.is_null());
                assert!(!prepared.is_null());
                let value: serde_json::Value =
                    serde_json::from_str(CStr::from_ptr(prepared).to_str().unwrap()).unwrap();
                if level == "none" {
                    assert!(value.is_null());
                } else {
                    assert_eq!(value["supportMap"], report["supportMap"]);
                    assert_eq!(value["provenance"]["reporting"]["level"], level);
                    if level == "rich" {
                        assert_eq!(
                            value["provenance"]["reporting"]["analysis"]["frames"][0]["measurementStatus"],
                            "failed"
                        );
                    }
                }
                seiza_string_free(prepared);
            }

            let html = seiza_stack_report_html(json, &mut error);
            assert!(error.is_null());
            assert!(!html.is_null());
            assert!(
                CStr::from_ptr(html)
                    .to_str()
                    .unwrap()
                    .contains("Completed stack")
            );
            seiza_string_free(html);
            let collection = CString::new(serde_json::json!({"schemaVersion":1,"title":"SHO", "stacks":[{"label":"S","report":report}]}).to_string()).unwrap();
            let html = seiza_stack_collection_html(collection.as_ptr(), &mut error);
            assert!(error.is_null());
            assert!(!html.is_null());
            assert!(CStr::from_ptr(html).to_str().unwrap().contains("SHO"));
            seiza_string_free(html);
            assert!(seiza_stack_collection_html(ptr::null(), &mut error).is_null());
            assert!(!error.is_null());
            seiza_string_free(error);
            seiza_string_free(json);
        }
    }
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
            let request =
                CString::new(r#"{"inputs":["does-not-exist.fits"],"metricsOnly":true}"#).unwrap();
            let json = seiza_analyze_project_json(request.as_ptr(), &mut error);
            assert!(error.is_null());
            assert!(!json.is_null());
            let value: serde_json::Value =
                serde_json::from_str(CStr::from_ptr(json).to_str().unwrap()).unwrap();
            assert_eq!(value["schemaVersion"], 2);
            assert_eq!(value["frames"][0]["measurementStatus"], "failed");
            assert_eq!(value["frames"][0]["depthStatus"], "not_requested");
            assert_eq!(value["timings"]["registrationSeconds"], 0.0);
            let html = seiza_analysis_html(json, &mut error);
            assert!(!html.is_null());
            assert!(error.is_null());
            seiza_string_free(html);
            seiza_string_free(json);
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct StackReportRequest {
    #[serde(default)]
    report: Option<serde_json::Value>,
    level: String,
    #[serde(default)]
    inputs: Vec<PathBuf>,
}
/// Select none, cheap retained diagnostics, or rich source analysis. Request:
/// {report: optional camelCase stack report, level: string, inputs: source paths}.
/// Returns owned camelCase JSON (JSON null for none), freed with seiza_string_free.
/// Rich reads inputs again; it never changes the completed stack.
/// # Safety
/// request_json must be NUL-terminated UTF-8; error_out null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_prepare_stack_report_json(
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let request: StackReportRequest = parse_request(request_json, "stack report options")?;
        let report = request
            .report
            .map(|r| serde_json::from_value(keys(r, false)))
            .transpose()
            .map_err(|e| e.to_string())?;
        let result = seiza_analysis::prepare_stack_report(report, &request.level, &request.inputs)
            .map_err(|e| e.to_string())?;
        owned_json(&keys(
            serde_json::to_value(result).map_err(|e| e.to_string())?,
            true,
        ))
    })
    .unwrap_or(ptr::null_mut())
}
