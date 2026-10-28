//! A thin C translation of the shared file calibration coordinator.
use super::{SeizaCancelSignal, clear_error, ffi_result, owned_json, required_str};
use crate::library::field_names;
use std::ffi::{CString, c_char, c_void};
use std::sync::Arc;

/// Synchronous phase progress, called on the coordinating thread. `phase` is
/// borrowed UTF-8 for this callback only; total zero denotes discovery. Callback
/// and context are never retained. Do not unwind or reenter the same cache.
pub type SeizaCalibrationProgressCallback = Option<
    unsafe extern "C" fn(
        phase: *const c_char,
        completed: usize,
        total: usize,
        context: *mut c_void,
    ),
>;

/// Grade/group/match provided FITS/XISF files or a recursive library and prepare
/// cached masters through the native file coordinator. Run off the UI thread.
/// camelCase request: operation (analyze/execute/verify/schema), lights, library
/// (optional), provided, cache, settings/rigs, choices/groupOverrides/lightChoices,
/// incomplete (stop/keep/exclude), flatStarMasking/flatStability, flatAssociation/
/// darkAssociation. Analyze builds no masters; execute requires its reviewKey.
/// Verify takes sources; schema returns native settings. Reports retain original
/// order, per-light assignments, evidence and corrected master paths. Refuse
/// changed reviews, missing required corrections and unsafe aliases explicitly.
/// Cooperative cancel also interrupts cache-lock waits. Completed immutable cache
/// entries survive cancellation; no partial master is published. Free result or
/// error with seiza_string_free. Source grants and UI remain host responsibilities.
/// Request/report schemas: seiza_calibration_library_schema_json.
/// # Safety
/// request_json is NUL-terminated UTF-8. cancel is null or a live cancellation
/// handle until return. progress/context remain valid until return; progress may
/// be null. error_out is null or writable pointer storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_calibration_files_json(
    request_json: *const c_char,
    cancel: *const SeizaCancelSignal,
    progress: SeizaCalibrationProgressCallback,
    context: *mut c_void,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let raw: serde_json::Value =
            serde_json::from_str(&required_str(request_json, "calibration files")?)
                .map_err(|e| e.to_string())?;
        if field_names(field_names(raw.clone(), false), true) != raw {
            return Err("calibration file fields must be camelCase".into());
        }
        let request = serde_json::from_value(field_names(raw, false)).map_err(|e| e.to_string())?;
        let cancel = unsafe { cancel.as_ref() }
            .map(|s| seiza_stacking::CancelSignal::from(Arc::clone(&s.cancelled)))
            .unwrap_or_else(|| seiza_stacking::CancelSignal::new(|| false));
        let report = seiza_stacking::calibration_files::json::run(request, cancel, |event| {
            if let Some(progress) = progress {
                let phase = CString::new(event.phase.to_string()).expect("a phase has no NUL");
                unsafe { progress(phase.as_ptr(), event.completed, event.total, context) };
            }
        })
        .map_err(|e| e.to_string())?;
        owned_json(&field_names(
            serde_json::to_value(report).map_err(|e| e.to_string())?,
            true,
        ))
    })
    .unwrap_or(std::ptr::null_mut())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::ffi::CStr;
    use std::path::Path;
    fn write(path: &Path, kind: &str, gain: i64, level: f32) {
        use seiza_fits::{F32ImageData, HeaderValue as H, WriteHeaderCard as C};
        let cards = vec![
            C::new("IMAGETYP", H::String(kind.into())),
            C::new("INSTRUME", H::String("TestCam".into())),
            C::new("GAIN", H::Integer(gain)),
            C::new("OFFSET", H::Integer(50)),
            C::new(
                "EXPTIME",
                H::Float(if kind == "BIAS" { 0.01 } else { 30.0 }),
            ),
            C::new("DATE-OBS", H::String("2026-10-20T20:00:00Z".into())),
        ];
        let pixels = (0..4096)
            .map(|i| level + (i % 7) as f32 - 3.0)
            .collect::<Vec<_>>();
        seiza_image_io::save_f32(
            path,
            64,
            64,
            F32ImageData::Mono(&pixels),
            &seiza_image_io::SaveOptions {
                headers: &cards,
                ..Default::default()
            },
        )
        .unwrap();
    }
    unsafe extern "C" fn progress(
        phase: *const c_char,
        _done: usize,
        _total: usize,
        context: *mut c_void,
    ) {
        let phases = unsafe { &mut *context.cast::<Vec<String>>() };
        phases.push(unsafe { CStr::from_ptr(phase) }.to_str().unwrap().into());
    }
    fn call(
        request: &Value,
        cancel: *const SeizaCancelSignal,
        phases: &mut Vec<String>,
    ) -> Result<Value, String> {
        let request = CString::new(request.to_string()).unwrap();
        let mut error = std::ptr::null_mut();
        let result = unsafe {
            seiza_calibration_files_json(
                request.as_ptr(),
                cancel,
                Some(progress),
                std::ptr::from_mut(phases).cast(),
                &mut error,
            )
        };
        if result.is_null() {
            let message = unsafe { CStr::from_ptr(error) }
                .to_str()
                .unwrap()
                .to_owned();
            unsafe { crate::seiza_string_free(error) };
            Err(message)
        } else {
            let value =
                serde_json::from_slice(unsafe { CStr::from_ptr(result) }.to_bytes()).unwrap();
            unsafe { crate::seiza_string_free(result) };
            Ok(value)
        }
    }
    #[test]
    fn file_workflow_uses_native_review_sessions_cache_progress_and_schema() {
        let root = tempfile::tempdir().unwrap();
        let library = root.path().join("library");
        std::fs::create_dir(&library).unwrap();
        let mut lights = Vec::new();
        for (i, gain) in [100, 200, 100].into_iter().enumerate() {
            let path = root.path().join(format!("light-{i}.fits"));
            write(&path, "LIGHT", gain, 2000.0);
            lights.push(path);
        }
        for gain in [100, 200] {
            for i in 0..3 {
                write(
                    &library.join(format!("bias-{gain}-{i}.fits")),
                    "BIAS",
                    gain,
                    gain as f32 + 400.0,
                );
            }
        }
        let mut request = json!({"operation":"analyze","lights":lights,"library":library,"cache":root.path().join("cache")});
        let mut phases = Vec::new();
        let review = call(&request, std::ptr::null(), &mut phases).unwrap();
        assert_eq!(review["snapshot"]["assignments"], json!([0, 1, 0]));
        assert!(review["fileSessions"].as_array().unwrap().is_empty());
        assert!(phases.iter().any(|p| p == "evidence"));
        const URL: &str = "https://seiza.test/calibration-library.json";
        let mut compiler = boon::Compiler::new();
        compiler
            .add_resource(URL, crate::library::library_schema())
            .unwrap();
        let mut schemas = boon::Schemas::new();
        let req = compiler
            .compile(
                &format!("{URL}#/calls/seiza_calibration_files_json/request"),
                &mut schemas,
            )
            .unwrap();
        let answer = compiler
            .compile(
                &format!("{URL}#/calls/seiza_calibration_files_json/answer"),
                &mut schemas,
            )
            .unwrap();
        schemas.validate(&request, req).unwrap();
        schemas.validate(&review, answer).unwrap();
        request["operation"] = json!("execute");
        request["reviewKey"] = review["reviewKey"].clone();
        let built = call(&request, std::ptr::null(), &mut phases).unwrap();
        schemas.validate(&built, answer).unwrap();
        assert_eq!(built["fileSessions"].as_array().unwrap().len(), 2);
        assert!(phases.iter().any(|p| p == "masters"));
        call(
            &json!({"operation":"verify","sources":built["sources"]}),
            std::ptr::null(),
            &mut phases,
        )
        .unwrap();
        request["flat_star_masking"] = json!(true);
        assert!(
            call(&request, std::ptr::null(), &mut phases)
                .unwrap_err()
                .contains("camelCase")
        );
        let cancel = crate::seiza_cancel_signal_create();
        unsafe { crate::seiza_cancel_signal_cancel(cancel) };
        assert!(call(&json!({"operation":"analyze","lights":lights,"library":library,"cache":root.path().join("other-cache")}),cancel,&mut phases).unwrap_err().contains("cancel"));
        unsafe { crate::seiza_cancel_signal_free(cancel) };
        write(&lights[0], "LIGHT", 100, 3100.0);
        assert!(
            call(
                &json!({"operation":"verify","sources":built["sources"]}),
                std::ptr::null(),
                &mut phases
            )
            .is_err()
        );
    }
}
