//! Distributed stacking's one-call workflows: a channel stacked as partial
//! stacks into a store, one session's step, a stack file's identity, and
//! stacks combined. Each runs `seiza_pipeline`'s native calls.
use super::*;
use seiza_pipeline::{
    ChannelPlan, ChannelReference, PartialStack, Session, SessionOptions, SessionOutcome,
    SessionStep, Store,
};

/// A request, its keys camelCase or snake_case, as `T`.
fn request<T: serde::de::DeserializeOwned>(text: *const c_char, name: &str) -> Result<T, String> {
    let value: Value = serde_json::from_str(&required_str(text, name)?)
        .map_err(|error| format!("invalid {name}: {error}"))?;
    serde_json::from_value(crate::flat_groups::field_names(value, false))
        .map_err(|error| format!("invalid {name}: {error}"))
}

/// An answer, its keys camelCase.
fn answer(value: Value) -> Result<*mut c_char, String> {
    owned_json(&crate::flat_groups::field_names(value, true))
}

fn session_answer(outcome: &SessionOutcome) -> Value {
    json!({
        "key": outcome.key,
        "path": outcome.path,
        "reused": outcome.reused,
        "extended_from": outcome.extended_from,
        "rejected": outcome
            .rejected
            .iter()
            .map(|(path, reason)| json!({"path": path, "reason": reason.to_string()}))
            .collect::<Vec<_>>(),
    })
}

fn identity_answer(stack: &PartialStack) -> Result<Value, String> {
    serde_json::to_value(&stack.identity).map_err(|error| error.to_string())
}

/// Stack a channel as partial stacks, one per session, into the store at
/// `store_path`, and combine them into the channel's tile stack, doing only
/// the steps whose outputs the store lacks (`seiza_pipeline::run_channel`).
///
/// `plan_json` is `{reference: {image, light, calibration}, sessions:
/// [{lights, calibration}], options: {stack, reintegration, contributor,
/// host, keepContext}}`: `calibration` a session record of
/// [`seiza_live_stacker_open_fits_expanding_sessions`], `stack` the options
/// of [`seiza_live_stacker_open_fits`], `reintegration` null or `{lowSigma,
/// highSigma, minimumSigma, scratchDirectory}`, `host` a 64-digit digest,
/// and `keepContext` true to keep each session's live-stack context in the
/// store, so a session that gains lights is extended from it rather than
/// stacked again. Keys may be camelCase or snake_case.
///
/// Returns owned JSON `{tile: {path, reused, identity}, sessions: [{key,
/// path, reused, extendedFrom, rejected: [{path, reason}]}]}`, where
/// `identity` is that of [`seiza_pipeline_read_identity_json`], a session
/// that accepted no light has a null `path`, and `extendedFrom` is the key
/// a session was extended from, or null. Free it with [`seiza_string_free`].
///
/// # Safety
/// `plan_json` and `store_path` must name NUL-terminated UTF-8 strings.
/// When non-null, `error_out` must point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_pipeline_run_channel_json(
    plan_json: *const c_char,
    store_path: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let plan: ChannelPlan = request(plan_json, "channel plan")?;
        let store = Store::new(required_path(store_path, "store path")?);
        let outcome =
            seiza_pipeline::run_channel(&plan, &store, |_| {}).map_err(|e| e.to_string())?;
        answer(json!({
            "tile": {
                "path": outcome.path,
                "reused": outcome.reused,
                "identity": identity_answer(&outcome.stack)?,
            },
            "sessions": outcome.sessions.iter().map(session_answer).collect::<Vec<_>>(),
        }))
    })
    .unwrap_or(ptr::null_mut())
}

/// Do one session's step against the store at `store_path`: plan it, and
/// return its partial stack there when the store holds one under its key,
/// or else stack the session and store it (`seiza_pipeline::SessionStep`).
/// `reference_json`, `session_json` and `options_json` are the `reference`,
/// one of the `sessions` and the `options` of
/// [`seiza_pipeline_run_channel_json`]'s plan; `options_json` may be null.
///
/// Returns owned JSON `{key, path, reused, extendedFrom, rejected: [{path,
/// reason}]}`, as each session of [`seiza_pipeline_run_channel_json`]'s
/// answer. Free it with [`seiza_string_free`].
///
/// # Safety
/// `reference_json`, `session_json` and `store_path` must name
/// NUL-terminated UTF-8 strings; `options_json` may be null. When non-null,
/// `error_out` must point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_pipeline_run_session_json(
    reference_json: *const c_char,
    session_json: *const c_char,
    options_json: *const c_char,
    store_path: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let reference: ChannelReference = request(reference_json, "channel reference")?;
        let session: Session = request(session_json, "session")?;
        let options: SessionOptions = if options_json.is_null() {
            SessionOptions::default()
        } else {
            request(options_json, "session options")?
        };
        let store = Store::new(required_path(store_path, "store path")?);
        let outcome = SessionStep::plan(&reference, &session, &options)
            .and_then(|step| step.run_into(&store))
            .map_err(|e| e.to_string())?;
        answer(session_answer(&outcome))
    })
    .unwrap_or(ptr::null_mut())
}

/// Read a stack file's identity from its header alone, without decoding
/// pixels: how a host learns whether a step is done.
///
/// Returns owned JSON `{kind, stackingVersion, pipelineVersion, stepKey,
/// inputs, inputCount, combined, calibration, options, host, estimator,
/// context, contributor, weightSource, grid}`, the cards `SEIZAPST` to
/// `GRIDSHA`: `kind` is `partial`, `tile` or `mosaic`, `estimator`
/// `additive` or `clipped`, `weightSource` `stack` or `measured`, and each
/// digest 64 hexadecimal digits or null. Free it with [`seiza_string_free`].
///
/// # Safety
/// `path` must name a NUL-terminated UTF-8 string. When non-null,
/// `error_out` must point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_pipeline_read_identity_json(
    path: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let identity = PartialStack::read_identity(&required_path(path, "stack path")?)
            .map_err(|e| e.to_string())?;
        answer(serde_json::to_value(&identity).map_err(|e| e.to_string())?)
    })
    .unwrap_or(ptr::null_mut())
}

/// Combine the stack files `paths_json` names, a JSON array of paths, on one
/// grid into a tile stack, weighing each by its weight and reading one at a
/// time (`seiza_pipeline::PartialStack::combine_files`), and write it to
/// `output_path`, FITS or XISF by its extension. Stacks on another grid or
/// made with other options are refused.
///
/// Returns owned JSON of the tile's identity, as
/// [`seiza_pipeline_read_identity_json`] gives it. Free it with
/// [`seiza_string_free`].
///
/// # Safety
/// `paths_json` and `output_path` must name NUL-terminated UTF-8 strings.
/// When non-null, `error_out` must point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_pipeline_combine_json(
    paths_json: *const c_char,
    output_path: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let paths: Vec<PathBuf> = serde_json::from_str(&required_str(paths_json, "stack paths")?)
            .map_err(|e| format!("invalid stack paths: {e}"))?;
        let output = required_path(output_path, "output path")?;
        let tile = PartialStack::combine_files(&paths).map_err(|e| e.to_string())?;
        tile.write(&output).map_err(|e| e.to_string())?;
        answer(identity_answer(&tile)?)
    })
    .unwrap_or(ptr::null_mut())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(json: *mut c_char, error: *mut c_char) -> Value {
        assert!(
            error.is_null(),
            "{:?}",
            unsafe { CStr::from_ptr(error) }.to_string_lossy()
        );
        assert!(!json.is_null());
        let value =
            serde_json::from_str(unsafe { CStr::from_ptr(json) }.to_str().unwrap()).unwrap();
        unsafe {
            seiza_string_free(json);
        }
        value
    }

    /// A channel of the panel fixture's first light, as two sessions with
    /// biases of their own, stacked into a store, resumed, read and
    /// combined through the C calls.
    #[test]
    fn a_channel_stacks_resumes_and_combines_through_the_c_api() {
        let root = tempfile::tempdir().unwrap();
        let panels = crate::tests::overlapping_panels(root.path());
        // Six dithers of the first panel, three to each session.
        let base = seiza_stacking::FitsFrame::open(&panels[0]).unwrap().image;
        let mut sessions = Vec::new();
        for session in 0..2 {
            let level = 500.0 * (session + 1) as f32;
            let bias = root.path().join(format!("bias-{session}.fits"));
            let flat =
                LinearImage::new(base.width, base.height, 1, vec![level; base.data.len()]).unwrap();
            seiza_stacking::write_processed_image_fits_f32(&bias, &flat, &[], &[]).unwrap();
            let mut lights = Vec::new();
            for index in 0..3 {
                let mut image = base.clone();
                for (sample, value) in image.data.iter_mut().enumerate() {
                    *value += level + ((sample * 31 + index * 7 + session * 3) % 11) as f32;
                }
                let light = root.path().join(format!("light-{session}-{index}.fits"));
                seiza_stacking::write_processed_image_fits_f32(&light, &image, &[], &[]).unwrap();
                lights.push(light);
            }
            sessions.push(json!({"lights": lights, "calibration": {"bias": bias}}));
        }
        let reference = json!({
            "image": sessions[0]["lights"][0],
            "light": sessions[0]["lights"][0],
            "calibration": sessions[0]["calibration"],
        });
        let options = json!({
            "stack": {"normalization": {"mode": "none"}, "rejection": {"mode": "none"}},
            "contributor": "tester",
        });
        let plan = CString::new(
            json!({"reference": reference, "sessions": sessions, "options": options}).to_string(),
        )
        .unwrap();
        let store = CString::new(root.path().join("store").to_str().unwrap()).unwrap();
        let mut error = ptr::null_mut();
        let first = call(
            unsafe { seiza_pipeline_run_channel_json(plan.as_ptr(), store.as_ptr(), &mut error) },
            error,
        );
        assert_eq!(first["tile"]["reused"], false);
        assert!(first["sessions"][0]["extendedFrom"].is_null());
        assert_eq!(first["tile"]["identity"]["kind"], "tile");
        assert_eq!(first["tile"]["identity"]["combined"], 6);
        assert_eq!(first["sessions"].as_array().unwrap().len(), 2);
        let again = call(
            unsafe { seiza_pipeline_run_channel_json(plan.as_ptr(), store.as_ptr(), &mut error) },
            error,
        );
        assert_eq!(again["tile"]["reused"], true);
        assert_eq!(again["tile"]["identity"], first["tile"]["identity"]);

        // One session's step finds its partial stack in the store.
        let reference = CString::new(reference.to_string()).unwrap();
        let session = CString::new(sessions[1].to_string()).unwrap();
        let options = CString::new(options.to_string()).unwrap();
        let step = call(
            unsafe {
                seiza_pipeline_run_session_json(
                    reference.as_ptr(),
                    session.as_ptr(),
                    options.as_ptr(),
                    store.as_ptr(),
                    &mut error,
                )
            },
            error,
        );
        assert_eq!(step["reused"], true);
        assert_eq!(step["key"], first["sessions"][1]["key"]);

        let partial = CString::new(step["path"].as_str().unwrap()).unwrap();
        let identity = call(
            unsafe { seiza_pipeline_read_identity_json(partial.as_ptr(), &mut error) },
            error,
        );
        assert_eq!(identity["kind"], "partial");
        assert_eq!(identity["stepKey"], step["key"]);
        assert_eq!(identity["contributor"], "tester");
        assert_eq!(identity["weightSource"], "stack");

        let paths = CString::new(
            json!([first["sessions"][0]["path"], first["sessions"][1]["path"]]).to_string(),
        )
        .unwrap();
        let output = root.path().join("tile.xisf");
        let output_path = CString::new(output.to_str().unwrap()).unwrap();
        let combined = call(
            unsafe {
                seiza_pipeline_combine_json(paths.as_ptr(), output_path.as_ptr(), &mut error)
            },
            error,
        );
        assert_eq!(combined, first["tile"]["identity"]);
        assert!(output.is_file());

        let missing = CString::new(root.path().join("missing.fits").to_str().unwrap()).unwrap();
        let none = unsafe { seiza_pipeline_read_identity_json(missing.as_ptr(), &mut error) };
        assert!(none.is_null() && !error.is_null());
        unsafe {
            seiza_string_free(error);
        }
    }
}
