//! Session-aware file planning; numerical preparation stays in the stacker.
use super::*;

/// Plan an empty expanded stack with per-light calibration sessions.
/// `paths_json` is an array of original file paths, `assignments_json` an
/// equally long array of session indices, and `sessions_json` an array of
/// `{bias, dark, flat, darkExposureSeconds, pedestal, normalizedFullScale,
/// tolerances}` records (paths may
/// be omitted/null). Tolerances use camelCase and include the lightRotation
/// basis. Omitted tolerances use the complete native default gate.
///
/// Assignments apply automatically when pushing these files, including after
/// context restore. Manual master/gate changes and anonymous pixel pushes
/// are refused. Pipelines process assigned inputs in order with one active
/// master set. Sources and historical masters remain protected from output
/// overwrite and are checked for changes before use/replay. The canvas limit,
/// cancellation, ownership and options follow the ordinary expanded opener.
///
/// # Safety
/// The three required JSON pointers must name NUL-terminated UTF-8 strings.
/// `options_json` may be null. `cancel` must be null or a live cancel signal.
/// When non-null, `error_out` must point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_live_stacker_open_fits_expanding_sessions(
    paths_json: *const c_char,
    sessions_json: *const c_char,
    assignments_json: *const c_char,
    options_json: *const c_char,
    max_canvas_megasamples: usize,
    cancel: *const SeizaCancelSignal,
    error_out: *mut *mut c_char,
) -> *mut SeizaLiveStacker {
    clear_error(error_out);
    ffi_result(error_out, || {
        let paths: Vec<PathBuf> = serde_json::from_str(&required_str(paths_json, "light paths")?)
            .map_err(|e| format!("invalid light paths: {e}"))?;
        let value: serde_json::Value =
            serde_json::from_str(&required_str(sessions_json, "calibration sessions")?)
                .map_err(|e| format!("invalid calibration sessions: {e}"))?;
        let sessions: Vec<seiza_stacking::FileCalibrationSession> =
            serde_json::from_value(crate::flat_groups::field_names(value, false))
                .map_err(|e| format!("invalid calibration sessions: {e}"))?;
        let assignments: Vec<usize> =
            serde_json::from_str(&required_str(assignments_json, "calibration assignments")?)
                .map_err(|e| format!("invalid calibration assignments: {e}"))?;
        let samples = (if max_canvas_megasamples == 0 {
            256
        } else {
            max_canvas_megasamples
        })
        .checked_mul(1_000_000)
        .ok_or("canvas limit is too large")?;
        let options = expanding_stack_options(options_json)?;
        let cancel = unsafe { cancel.as_ref() };
        let stacker = LiveStacker::open_fits_expanding_sessions(
            &paths,
            &sessions,
            &assignments,
            options,
            samples,
            |_, _, _, _| {
                if cancel.is_some_and(|s| s.cancelled.load(Ordering::Relaxed)) {
                    seiza_stacking::Continue::No
                } else {
                    seiza_stacking::Continue::Yes
                }
            },
        )
        .map_err(|e| e.to_string())?;
        Ok(Box::into_raw(Box::new(SeizaLiveStacker { stacker })))
    })
    .unwrap_or(ptr::null_mut())
}

#[cfg(test)]
mod tests {
    use super::*;
    use seiza_fits::HeaderValue;

    #[test]
    fn assigned_calibration_and_gate_survive_the_c_api_and_context() {
        assigned_recipe(false);
        assigned_recipe(true);
    }

    fn assigned_recipe(fitted: bool) {
        let root = tempfile::tempdir().unwrap();
        let paths = crate::tests::overlapping_panels(root.path());
        let mut sessions = Vec::new();
        for index in 0..2 {
            let level = if index == 0 { 500.0 } else { 1500.0 };
            let master = root.path().join(format!("bias-{index}.fits"));
            let shape = seiza_stacking::FitsFrame::open(&paths[0]).unwrap().image;
            let bias = LinearImage::new(
                shape.width,
                shape.height,
                1,
                vec![level; shape.width * shape.height],
            )
            .unwrap();
            let headers = vec![("GAIN".into(), HeaderValue::Integer(100 + index * 100))];
            seiza_stacking::write_processed_image_fits_f32(&master, &bias, &headers, &[]).unwrap();
            for path in paths
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 2 == index as usize)
                .map(|(_, p)| p)
            {
                let mut raw = seiza_stacking::FitsFrame::open(path).unwrap();
                for sample in &mut raw.image.data {
                    *sample += level;
                }
                seiza_stacking::write_processed_image_fits_f32(path, &raw.image, &headers, &[])
                    .unwrap();
            }
            let rotation = if index == 1 {
                serde_json::json!({"basis":"measured","degrees":90.0})
            } else {
                serde_json::json!({"basis":"recorded"})
            };
            let mut recipe = if fitted {
                let flat = root.path().join(format!("flat-{index}.fits"));
                let mut normalized = bias.clone();
                normalized.data.fill(1.0);
                seiza_stacking::write_processed_image_fits_f32(
                    &flat,
                    &normalized,
                    &headers,
                    &[seiza_fits::WriteHeaderCard::new(
                        "FLATNORM",
                        HeaderValue::Logical(true),
                    )],
                )
                .unwrap();
                serde_json::json!({"flat":flat,"pedestal":level,"normalizedFullScale":65535})
            } else {
                serde_json::json!({"bias":master})
            };
            recipe["tolerances"] = serde_json::json!({"lightRotation":rotation});
            sessions.push(recipe);
        }
        let paths_json = CString::new(serde_json::to_string(&paths).unwrap()).unwrap();
        let sessions_json = CString::new(serde_json::to_string(&sessions).unwrap()).unwrap();
        let options=c"{\"normalization\":{\"mode\":\"none\"},\"rejection\":{\"mode\":\"none\"},\"acceptance\":{\"minimum_overlap_fraction\":0.1}}";
        let mut error = ptr::null_mut();
        let mut handle = unsafe {
            seiza_live_stacker_open_fits_expanding_sessions(
                paths_json.as_ptr(),
                sessions_json.as_ptr(),
                c"[0,1,0]".as_ptr(),
                options.as_ptr(),
                256,
                ptr::null(),
                &mut error,
            )
        };
        assert!(
            !handle.is_null(),
            "{:?}",
            if error.is_null() {
                None
            } else {
                Some(unsafe { CStr::from_ptr(error) })
            }
        );
        assert!(error.is_null());
        assert_eq!(unsafe { seiza_live_stacker_accepted_frames(handle) }, 0);
        for (index, path) in paths.iter().enumerate() {
            let path = CString::new(path.as_str()).unwrap();
            let output =
                unsafe { seiza_live_stacker_push_fits_json(handle, path.as_ptr(), &mut error) };
            assert!(error.is_null());
            let result: serde_json::Value =
                serde_json::from_str(unsafe { CStr::from_ptr(output) }.to_str().unwrap()).unwrap();
            unsafe {
                seiza_string_free(output);
            }
            assert_eq!(result["accepted"], true, "{result}");
            let gate = unsafe { &*handle }.stacker.match_tolerances();
            assert_eq!(
                gate.light_rotation,
                if index == 1 {
                    seiza_calibration::LightRotation::Measured { degrees: 90.0 }
                } else {
                    seiza_calibration::LightRotation::Recorded
                }
            );
            if index == 1 {
                let context =
                    CString::new(root.path().join("resume.stack").to_str().unwrap()).unwrap();
                assert!(unsafe {
                    seiza_live_stacker_save_context(handle, context.as_ptr(), &mut error)
                });
                unsafe {
                    seiza_live_stacker_free(handle);
                }
                handle = unsafe { seiza_live_stacker_open_context(context.as_ptr(), &mut error) };
                assert!(!handle.is_null());
                assert!(error.is_null());
            }
        }
        assert_eq!(unsafe { seiza_live_stacker_accepted_frames(handle) }, 3);
        unsafe {
            seiza_live_stacker_free(handle);
        }
    }
}
