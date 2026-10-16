//! Flat sessions and the dust they hold (`seiza_calibration::flat_response`
//! and `seiza_calibration::dust`), as Python's `flat_*` and `dust_*`
//! functions offer them.
//!
//! A flat set reduces to one **response**: its frames binned
//! ([`seiza_flat_bin_frame`]) and combined ([`seiza_flat_build_response_json`]).
//! Two responses of one light path compare by their ratio
//! ([`seiza_flat_compare_json`]); the change between them makes a map
//! ([`seiza_flat_change_map`]) that carries a flat across the change
//! ([`seiza_flat_apply_change_map`]). A few lights' own background
//! ([`seiza_flat_lights_background`]) says which side of a change they were
//! shot on ([`seiza_flat_fit_change_json`]), or which of several responses
//! they follow ([`seiza_flat_choose_response_json`]). The dust calls find
//! the dust in one plane or a whole session, check one flat against its
//! session, and say how two sessions' dust differs.
//!
//! Planes are row-major `float` samples, `width × height`, owned by the
//! caller; several planes of one size are one buffer of `count × width ×
//! height` samples, plane after plane. A call that makes a plane writes it
//! into the caller's buffer. Results are the core's own JSON, the form it
//! keeps in its caches, so their keys are snake_case; options are
//! `DustOptions` JSON, any field left out taking its default. Everything is
//! pure: the host reads the frames, keeps the responses and comparisons,
//! and decides what a verdict means for its library.

use super::{
    clear_error, ffi_result, owned_json, required_f32_slice, required_f32_slice_mut, required_str,
};
use seiza_calibration::dust::{self, DustFeature, DustMap, DustOptions};
use seiza_calibration::flat_response::{Plane, Side, Verdict, choice, evidence, metric};
use std::ffi::{CString, c_char};
use std::ptr;

/// The plane at `data`, `width × height` samples.
unsafe fn plane(
    data: *const f32,
    width: usize,
    height: usize,
    name: &str,
) -> Result<Plane, String> {
    let length = width.checked_mul(height).ok_or("the plane is too large")?;
    if length == 0 {
        return Err(format!("{name} needs a width and a height"));
    }
    let samples = unsafe { required_f32_slice(data, length, name) }?;
    Ok(Plane {
        width,
        height,
        data: samples.to_vec(),
    })
}

/// `count` planes of `width × height`, one after another at `data`.
unsafe fn planes(
    data: *const f32,
    count: usize,
    width: usize,
    height: usize,
    name: &str,
) -> Result<Vec<Plane>, String> {
    let each = width
        .checked_mul(height)
        .ok_or("the planes are too large")?;
    if each == 0 {
        return Err(format!("{name} need a width and a height"));
    }
    let length = each.checked_mul(count).ok_or("the planes are too large")?;
    let samples = unsafe { required_f32_slice(data, length, name) }?;
    Ok(samples
        .chunks_exact(each)
        .map(|chunk| Plane {
            width,
            height,
            data: chunk.to_vec(),
        })
        .collect())
}

/// Copy `plane` into the caller's `out`, which must hold exactly its samples.
unsafe fn write_plane(plane: &Plane, out: *mut f32, out_length: usize) -> Result<(), String> {
    if out_length != plane.data.len() {
        return Err(format!(
            "the output holds {out_length} samples; the plane has {} ({}x{})",
            plane.data.len(),
            plane.width,
            plane.height
        ));
    }
    let out = unsafe { required_f32_slice_mut(out, out_length, "out") }?;
    out.copy_from_slice(&plane.data);
    Ok(())
}

/// `DustOptions` from JSON, or the defaults for null.
fn dust_options(json: *const c_char) -> Result<DustOptions, String> {
    if json.is_null() {
        return Ok(DustOptions::default());
    }
    let json = required_str(json, "dust options JSON")?;
    serde_json::from_str(&json).map_err(|error| format!("invalid dust options JSON: {error}"))
}

/// The JSON text `null`, for a call whose answer is "none".
fn null_json() -> *mut c_char {
    CString::new("null")
        .map(CString::into_raw)
        .unwrap_or(ptr::null_mut())
}

/// The block factor that brings a `width × height` frame's long side within
/// the response size: each response pixel covers `factor × factor` sensor
/// pixels. With `cfa` true the blocks cover whole 2×2 colour cells.
#[unsafe(no_mangle)]
pub extern "C" fn seiza_flat_bin_factor(width: usize, height: usize, cfa: bool) -> usize {
    metric::bin_factor(width, height, cfa)
}

/// Bin a frame for a response: block means of `factor × factor` pixels,
/// its `channels` interleaved channels averaged. Partial blocks at the right
/// and bottom edges are left out, so `out` holds `(width / factor) × (height
/// / factor)` samples. Returns false with `error_out` set on failure.
///
/// # Safety
///
/// `data` must point to `width × height × channels` readable, aligned
/// floats and `out` to `out_length` writable ones. When non-null,
/// `error_out` must point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_flat_bin_frame(
    data: *const f32,
    width: usize,
    height: usize,
    channels: usize,
    factor: usize,
    out: *mut f32,
    out_length: usize,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        if width == 0 || height == 0 || channels == 0 || factor == 0 {
            return Err("the frame needs a width, a height, channels and a factor".into());
        }
        let length = width
            .checked_mul(height)
            .and_then(|area| area.checked_mul(channels))
            .ok_or("the frame is too large")?;
        let samples = unsafe { required_f32_slice(data, length, "data") }?;
        let binned = metric::bin_frame(samples, width, height, channels, factor);
        unsafe { write_plane(&binned, out, out_length) }
    })
    .is_some()
}

/// Combine `count` binned flats of one session (each `width × height`, raw
/// ADU with their pedestals) into one response, written into `out` (`width
/// × height` samples). `bin` is the factor they were binned by. Returns JSON
/// `{"bin", "level", "noise", "frames"}`, the text `null` when the flats are
/// too few or unusable, or null with `error_out` set on failure. Release it
/// with `seiza_string_free`.
///
/// # Safety
///
/// `planes` must point to `count × width × height` readable, aligned floats
/// and `out` to `out_length` writable ones. When non-null, `error_out` must
/// point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_flat_build_response_json(
    planes_data: *const f32,
    count: usize,
    width: usize,
    height: usize,
    bin: usize,
    out: *mut f32,
    out_length: usize,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let binned = unsafe { planes(planes_data, count, width, height, "planes") }?;
        let Some(response) = metric::build_response(&binned, bin) else {
            return Ok(null_json());
        };
        unsafe { write_plane(&response.plane, out, out_length) }?;
        owned_json(&serde_json::json!({
            "bin": response.bin,
            "level": response.level,
            "noise": response.noise,
            "frames": response.frames,
        }))
    })
    .unwrap_or(ptr::null_mut())
}

/// Compare two responses of one light path (`width × height`, pedestals
/// removed) binned by `bin`: the broad change, the dust, and the spots that
/// came, went or moved. `options_json` is `DustOptions` JSON, or null for
/// the defaults. Returns the comparison's JSON, the text `null` when the
/// responses hold too little to compare, or null with `error_out` set.
/// Release it with `seiza_string_free`.
///
/// # Safety
///
/// `earlier` and `later` must each point to `width × height` readable,
/// aligned floats. `options_json` must be null or a NUL-terminated UTF-8
/// string. When non-null, `error_out` must point to writable storage for
/// one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_flat_compare_json(
    earlier: *const f32,
    later: *const f32,
    width: usize,
    height: usize,
    bin: usize,
    options_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let earlier = unsafe { plane(earlier, width, height, "earlier") }?;
        let later = unsafe { plane(later, width, height, "later") }?;
        let options = dust_options(options_json)?;
        match metric::compare_with(&earlier, &later, bin, &options) {
            Some(comparison) => owned_json(&comparison),
            None => Ok(null_json()),
        }
    })
    .unwrap_or(ptr::null_mut())
}

/// The compact change between a session before an event and one after it
/// (`after / before`, each `width × height`), written into `out` (`width ×
/// height` samples). Returns 1 when written, 0 when the responses hold no
/// change map, and -1 with `error_out` set on failure.
///
/// # Safety
///
/// `before` and `after` must each point to `width × height` readable,
/// aligned floats and `out` to `out_length` writable ones. When non-null,
/// `error_out` must point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_flat_change_map(
    before: *const f32,
    after: *const f32,
    width: usize,
    height: usize,
    out: *mut f32,
    out_length: usize,
    error_out: *mut *mut c_char,
) -> i32 {
    clear_error(error_out);
    ffi_result(error_out, || {
        let before = unsafe { plane(before, width, height, "before") }?;
        let after = unsafe { plane(after, width, height, "after") }?;
        let Some(map) = metric::change_map(&before, &after) else {
            return Ok(0);
        };
        unsafe { write_plane(&map, out, out_length) }?;
        Ok(1)
    })
    .unwrap_or(-1)
}

/// Carry a flat across a change in place: divide `data` (`width × height ×
/// channels` interleaved samples, sensor-aligned) by the change map
/// (`map_width × map_height`, binned by `bin`) for a light shot before the
/// change (`divide` true), or multiply for one shot after it. Returns false
/// with `error_out` set on failure.
///
/// # Safety
///
/// `data` must point to `width × height × channels` writable, aligned
/// floats and `map` to `map_width × map_height` readable ones. When
/// non-null, `error_out` must point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_flat_apply_change_map(
    data: *mut f32,
    width: usize,
    height: usize,
    channels: usize,
    map: *const f32,
    map_width: usize,
    map_height: usize,
    bin: usize,
    divide: bool,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        if width == 0 || height == 0 || channels == 0 || bin == 0 {
            return Err("the image needs a width, a height, channels and a bin".into());
        }
        let length = width
            .checked_mul(height)
            .and_then(|area| area.checked_mul(channels))
            .ok_or("the image is too large")?;
        let map = unsafe { plane(map, map_width, map_height, "map") }?;
        let data = unsafe { required_f32_slice_mut(data, length, "data") }?;
        metric::apply_change_map(data, width, channels, &map, bin, divide);
        Ok(())
    })
    .is_some()
}

/// The sky background of a few lights, for weighing them against a change:
/// `count` binned lights (each `width × height`), each less `pedestal` and
/// divided by its own median, combined, and written into `out` (`width ×
/// height` samples). Returns 1 when written, 0 when the lights hold too
/// little, and -1 with `error_out` set on failure.
///
/// # Safety
///
/// `planes` must point to `count × width × height` readable, aligned floats
/// and `out` to `out_length` writable ones. When non-null, `error_out` must
/// point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_flat_lights_background(
    planes_data: *const f32,
    count: usize,
    width: usize,
    height: usize,
    pedestal: f32,
    bin: usize,
    out: *mut f32,
    out_length: usize,
    error_out: *mut *mut c_char,
) -> i32 {
    clear_error(error_out);
    ffi_result(error_out, || {
        let binned = unsafe { planes(planes_data, count, width, height, "planes") }?;
        let Some(background) = evidence::background(&binned, pedestal, bin) else {
            return Ok(0);
        };
        unsafe { write_plane(&background, out, out_length) }?;
        Ok(1)
    })
    .unwrap_or(-1)
}

/// Which side of a change a few lights were shot on: their `background`
/// weighed against the `reference` response, which lies on side
/// `reference_side` (`"before"` or `"after"`) of the change `map`, all
/// `width × height`. Returns the outcome's JSON (`{"outcome": "fit",
/// "carried", "spread", "verdict"}`, `{"outcome": "no_shadows"}` or
/// `{"outcome": "unfitted"}`), or null with `error_out` set. Release it with
/// `seiza_string_free`.
///
/// # Safety
///
/// `background`, `reference` and `map` must each point to `width × height`
/// readable, aligned floats. `reference_side` must be a NUL-terminated
/// UTF-8 string. When non-null, `error_out` must point to writable storage
/// for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_flat_fit_change_json(
    background: *const f32,
    reference: *const f32,
    map: *const f32,
    width: usize,
    height: usize,
    reference_side: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let background = unsafe { plane(background, width, height, "background") }?;
        let reference = unsafe { plane(reference, width, height, "reference") }?;
        let map = unsafe { plane(map, width, height, "map") }?;
        let side = required_str(reference_side, "reference_side")?;
        let side: Side = serde_json::from_value(serde_json::Value::String(side.clone()))
            .map_err(|_| format!("reference_side must be \"before\" or \"after\", not {side:?}"))?;
        owned_json(&evidence::fit_outcome(&background, &reference, side, &map))
    })
    .unwrap_or(ptr::null_mut())
}

/// One verdict for an imaging session from its filters' verdicts: a side
/// some fit decides and none contradicts. `verdicts_json` is a JSON array
/// of `"before"`, `"after"` and `"undecided"`. Returns the verdict as JSON
/// text (such as `"before"`), or null with `error_out` set. Release it with
/// `seiza_string_free`.
///
/// # Safety
///
/// `verdicts_json` must be a NUL-terminated UTF-8 string. When non-null,
/// `error_out` must point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_flat_combine_verdicts_json(
    verdicts_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let json = required_str(verdicts_json, "verdicts JSON")?;
        let verdicts: Vec<Verdict> = serde_json::from_str(&json)
            .map_err(|error| format!("invalid verdicts JSON: {error}"))?;
        owned_json(&evidence::combine(&verdicts))
    })
    .unwrap_or(ptr::null_mut())
}

/// Which of `count` candidate responses (each `width × height`, such as one
/// rig's flats at several rotator angles) the lights' `background` follows.
/// `alike_json` is null, or a JSON array of `[first, second]` index pairs
/// the host already knows to be alike, which are not weighed. Returns
/// `{"winners": [...], "pairs": [...]}`, or null with `error_out` set.
/// Release it with `seiza_string_free`.
///
/// # Safety
///
/// `background` must point to `width × height` readable, aligned floats and
/// `candidates` to `count × width × height`. `alike_json` must be null or a
/// NUL-terminated UTF-8 string. When non-null, `error_out` must point to
/// writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_flat_choose_response_json(
    background: *const f32,
    candidates: *const f32,
    count: usize,
    width: usize,
    height: usize,
    alike_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let background = unsafe { plane(background, width, height, "background") }?;
        let candidates = unsafe { planes(candidates, count, width, height, "candidates") }?;
        let alike: Vec<[usize; 2]> = if alike_json.is_null() {
            Vec::new()
        } else {
            let json = required_str(alike_json, "alike JSON")?;
            serde_json::from_str(&json).map_err(|error| format!("invalid alike JSON: {error}"))?
        };
        let choice = choice::choose_response(&background, &candidates, |first, second| {
            alike
                .iter()
                .any(|&[a, b]| (a, b) == (first, second) || (a, b) == (second, first))
        });
        owned_json(&choice)
    })
    .unwrap_or(ptr::null_mut())
}

/// Encode a plane (`width × height`) for a host's cache: an 8-byte size,
/// then 16-bit fixed point covering 0 to 2. Returns the bytes needed; when
/// `out` is non-null and holds that many, they are written. Returns 0 for a
/// null `plane` or an empty size.
///
/// # Safety
///
/// `plane` must be null or point to `width × height` readable, aligned
/// floats, and `out` null or point to `out_length` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_flat_encode_plane(
    plane_data: *const f32,
    width: usize,
    height: usize,
    out: *mut u8,
    out_length: usize,
) -> usize {
    let Ok(plane) = (unsafe { plane(plane_data, width, height, "plane") }) else {
        return 0;
    };
    let bytes = metric::encode_plane(&plane);
    if !out.is_null() && out_length >= bytes.len() {
        unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len()) };
    }
    bytes.len()
}

/// Read a plane [`seiza_flat_encode_plane`] wrote. Its size goes to
/// `width_out` and `height_out`; when `out` is non-null, its samples go
/// there, and `out_length` must equal `width × height`. Pass a null `out`
/// first to learn the size. Returns 1 when read, 0 when the bytes are not a
/// plane, and -1 with `error_out` set on failure.
///
/// # Safety
///
/// `bytes` must point to `length` readable bytes; `width_out` and
/// `height_out` to writable `size_t`s; `out` must be null or point to
/// `out_length` writable floats. When non-null, `error_out` must point to
/// writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_flat_decode_plane(
    bytes: *const u8,
    length: usize,
    out: *mut f32,
    out_length: usize,
    width_out: *mut usize,
    height_out: *mut usize,
    error_out: *mut *mut c_char,
) -> i32 {
    clear_error(error_out);
    ffi_result(error_out, || {
        if bytes.is_null() || width_out.is_null() || height_out.is_null() {
            return Err("bytes, width_out and height_out are required".into());
        }
        let bytes = unsafe { std::slice::from_raw_parts(bytes, length) };
        let Some(plane) = metric::decode_plane(bytes) else {
            return Ok(0);
        };
        unsafe {
            *width_out = plane.width;
            *height_out = plane.height;
        }
        if !out.is_null() {
            unsafe { write_plane(&plane, out, out_length) }?;
        }
        Ok(1)
    })
    .unwrap_or(-1)
}

/// Find the dust in one plane (`width × height`, binned by `bin`): a flat
/// session's response or one binned flat. `options_json` is `DustOptions`
/// JSON, or null for the defaults. Returns the dust map's JSON, the text
/// `null` when the plane holds too little, or null with `error_out` set.
/// Release it with `seiza_string_free`.
///
/// # Safety
///
/// `plane` must point to `width × height` readable, aligned floats.
/// `options_json` must be null or a NUL-terminated UTF-8 string. When
/// non-null, `error_out` must point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_dust_find_json(
    plane_data: *const f32,
    width: usize,
    height: usize,
    bin: usize,
    options_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let plane = unsafe { plane(plane_data, width, height, "plane") }?;
        let options = dust_options(options_json)?;
        match dust::find_dust(&plane, bin, &options) {
            Some(map) => owned_json(&map),
            None => Ok(null_json()),
        }
    })
    .unwrap_or(ptr::null_mut())
}

/// Find a whole flat session's dust: `count` binned flats (each `width ×
/// height`, raw ADU, binned by `bin`) stacked, so faint disks and donuts
/// clear the noise, and each feature counted across the flats that show it.
/// `pedestal` is the flats' pedestal. Returns the session's dust map JSON,
/// the text `null` with fewer than two usable flats, or null with
/// `error_out` set. Release it with `seiza_string_free`.
///
/// # Safety
///
/// `planes` must point to `count × width × height` readable, aligned
/// floats. `options_json` must be null or a NUL-terminated UTF-8 string.
/// When non-null, `error_out` must point to writable storage for one
/// pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_dust_find_session_json(
    planes_data: *const f32,
    count: usize,
    width: usize,
    height: usize,
    bin: usize,
    pedestal: f32,
    options_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let binned = unsafe { planes(planes_data, count, width, height, "planes") }?;
        let options = dust_options(options_json)?;
        match dust::find_session_dust(&binned, bin, pedestal, &options) {
            Some(map) => owned_json(&map),
            None => Ok(null_json()),
        }
    })
    .unwrap_or(ptr::null_mut())
}

/// One flat of a session against the session's dust: which of the
/// session's features it shows, and what it holds that the session does
/// not. `frame` is binned like the session's flats (`width × height`, raw
/// ADU); `session_json` is the dust map [`seiza_dust_find_session_json`]
/// returned. Returns `{"seen": [...], "transients": [...]}`, the text
/// `null` when the frame holds too little, or null with `error_out` set.
/// Release it with `seiza_string_free`.
///
/// # Safety
///
/// `frame` must point to `width × height` readable, aligned floats.
/// `session_json` must be a NUL-terminated UTF-8 string and `options_json`
/// null or one. When non-null, `error_out` must point to writable storage
/// for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_dust_frame_json(
    frame: *const f32,
    width: usize,
    height: usize,
    pedestal: f32,
    session_json: *const c_char,
    options_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let frame = unsafe { plane(frame, width, height, "frame") }?;
        let session = required_str(session_json, "session dust JSON")?;
        let session: DustMap = serde_json::from_str(&session)
            .map_err(|error| format!("invalid session dust JSON: {error}"))?;
        let options = dust_options(options_json)?;
        match dust::frame_dust(&frame, pedestal, &session, &options) {
            Some(found) => owned_json(&found),
            None => Ok(null_json()),
        }
    })
    .unwrap_or(ptr::null_mut())
}

/// How the dust of two sessions of one light path differs: `earlier` and
/// `later` are responses on one grid (`width × height`, binned by `bin`,
/// pedestals removed). Returns the changes' JSON (what came, went or moved,
/// with both sessions' dust maps), the text `null` when the planes hold too
/// little, or null with `error_out` set. Release it with
/// `seiza_string_free`.
///
/// # Safety
///
/// `earlier` and `later` must each point to `width × height` readable,
/// aligned floats. `options_json` must be null or a NUL-terminated UTF-8
/// string. When non-null, `error_out` must point to writable storage for
/// one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_dust_find_changes_json(
    earlier: *const f32,
    later: *const f32,
    width: usize,
    height: usize,
    bin: usize,
    options_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let earlier = unsafe { plane(earlier, width, height, "earlier") }?;
        let later = unsafe { plane(later, width, height, "later") }?;
        let options = dust_options(options_json)?;
        match dust::find_changes(&earlier, &later, bin, &options) {
            Some(changes) => owned_json(&changes),
            None => Ok(null_json()),
        }
    })
    .unwrap_or(ptr::null_mut())
}

/// Match two lists of dust features, as two dust maps' `features` hold
/// them: each pair whose centres lie within the options' tolerance for
/// their radius. `earlier_json` and `later_json` are JSON arrays of
/// features. Returns a JSON array of `{"earlier", "later", "offset",
/// "within_tolerance"}`, or null with `error_out` set. Release it with
/// `seiza_string_free`.
///
/// # Safety
///
/// `earlier_json` and `later_json` must be NUL-terminated UTF-8 strings and
/// `options_json` null or one. When non-null, `error_out` must point to
/// writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_dust_match_features_json(
    earlier_json: *const c_char,
    later_json: *const c_char,
    options_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let features = |json: *const c_char, name: &str| -> Result<Vec<DustFeature>, String> {
            let json = required_str(json, name)?;
            serde_json::from_str(&json).map_err(|error| format!("invalid {name}: {error}"))
        };
        let earlier = features(earlier_json, "earlier features JSON")?;
        let later = features(later_json, "later features JSON")?;
        let options = dust_options(options_json)?;
        owned_json(&dust::match_features(&earlier, &later, &options))
    })
    .unwrap_or(ptr::null_mut())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{CStr, CString};

    const W: usize = 512;
    const H: usize = 384;
    const BIN: usize = 7;
    const LEVEL: f32 = 30_000.0;
    const PEDESTAL: f32 = 300.0;

    /// A binned flat in raw ADU: a soft vignetting falloff, dust disks
    /// `(x, y, radius, depth)` in plane pixels, and seeded noise.
    fn flat(disks: &[(f32, f32, f32, f32)], noise: f32, seed: u64) -> Vec<f32> {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut uniform = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64).max(1e-12)
        };
        let (cx, cy) = (W as f32 / 2.0, H as f32 / 2.0);
        let reach = cx.hypot(cy);
        (0..W * H)
            .map(|index| {
                let (x, y) = ((index % W) as f32, (index / W) as f32);
                let mut value = 1.0 - 0.02 * ((x - cx).hypot(y - cy) / reach).powi(2);
                for &(dx, dy, radius, depth) in disks {
                    if (x - dx).hypot(y - dy) <= radius {
                        value *= 1.0 - depth;
                    }
                }
                let gauss = ((-2.0 * uniform().ln()).sqrt()
                    * (std::f64::consts::TAU * uniform()).cos()) as f32;
                (value * (1.0 + noise * gauss)) * LEVEL + PEDESTAL
            })
            .collect()
    }

    fn session(disks: &[(f32, f32, f32, f32)], seed: u64) -> Vec<f32> {
        (0..6)
            .flat_map(|frame| flat(disks, 0.0015, seed + frame))
            .collect()
    }

    fn text(pointer: *mut c_char, error: *mut c_char) -> String {
        assert!(!pointer.is_null(), "{}", take(error));
        let value = unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned();
        unsafe { crate::seiza_string_free(pointer) };
        value
    }

    fn take(error: *mut c_char) -> String {
        if error.is_null() {
            return String::new();
        }
        let message = unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned();
        unsafe { crate::seiza_string_free(error) };
        message
    }

    fn json(pointer: *mut c_char, error: *mut c_char) -> serde_json::Value {
        serde_json::from_str(&text(pointer, error)).unwrap()
    }

    /// A session's response, with its pedestal removed as the comparisons want.
    fn response(flats: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0_f32; W * H];
        let mut error = ptr::null_mut();
        let pedestal_removed: Vec<f32> = flats.iter().map(|value| value - PEDESTAL).collect();
        let summary = json(
            unsafe {
                seiza_flat_build_response_json(
                    pedestal_removed.as_ptr(),
                    6,
                    W,
                    H,
                    BIN,
                    out.as_mut_ptr(),
                    out.len(),
                    &mut error,
                )
            },
            error,
        );
        assert_eq!(summary["frames"], 6, "{summary}");
        assert_eq!(summary["bin"], BIN, "{summary}");
        out
    }

    #[test]
    fn a_session_finds_its_dust_and_each_flat_is_checked_against_it() {
        let flats = session(&[(200.0, 150.0, 40.0, 0.004)], 20);
        let mut error = ptr::null_mut();
        let session_json = text(
            unsafe {
                seiza_dust_find_session_json(
                    flats.as_ptr(),
                    6,
                    W,
                    H,
                    BIN,
                    PEDESTAL,
                    ptr::null(),
                    &mut error,
                )
            },
            error,
        );
        let map: serde_json::Value = serde_json::from_str(&session_json).unwrap();
        assert_eq!(map["frames"], 6, "{map}");
        let disks: Vec<_> = map["features"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|feature| feature["shape"] == "disk")
            .collect();
        assert_eq!(disks.len(), 1, "{map}");
        assert_eq!(disks[0]["seen_in"]["frames"], 6, "{map}");

        // One flat shows the session's disk.
        let session_c = CString::new(session_json).unwrap();
        let found = json(
            unsafe {
                seiza_dust_frame_json(
                    flats.as_ptr(),
                    W,
                    H,
                    PEDESTAL,
                    session_c.as_ptr(),
                    ptr::null(),
                    &mut error,
                )
            },
            error,
        );
        assert!(
            found["seen"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!(0)),
            "{found}"
        );

        // The session's features match themselves.
        let features = CString::new(map["features"].to_string()).unwrap();
        let matches = json(
            unsafe {
                seiza_dust_match_features_json(
                    features.as_ptr(),
                    features.as_ptr(),
                    ptr::null(),
                    &mut error,
                )
            },
            error,
        );
        assert!(
            matches
                .as_array()
                .unwrap()
                .iter()
                .all(|pair| pair["earlier"] == pair["later"]),
            "{matches}"
        );
    }

    #[test]
    fn two_sessions_compare_and_a_change_map_carries_a_flat_across() {
        let earlier = response(&session(&[(200.0, 150.0, 40.0, 0.004)], 40));
        let later = response(&session(
            &[(200.0, 150.0, 40.0, 0.004), (360.0, 250.0, 30.0, 0.006)],
            60,
        ));
        let mut error = ptr::null_mut();
        let comparison = json(
            unsafe {
                seiza_flat_compare_json(
                    earlier.as_ptr(),
                    later.as_ptr(),
                    W,
                    H,
                    BIN,
                    ptr::null(),
                    &mut error,
                )
            },
            error,
        );
        for key in ["broad", "dust", "sigma", "spots"] {
            assert!(comparison.get(key).is_some(), "{key}: {comparison}");
        }
        let changes = json(
            unsafe {
                seiza_dust_find_changes_json(
                    earlier.as_ptr(),
                    later.as_ptr(),
                    W,
                    H,
                    BIN,
                    ptr::null(),
                    &mut error,
                )
            },
            error,
        );
        assert!(
            changes["changes"]
                .as_array()
                .unwrap()
                .iter()
                .any(|change| change["kind"] == "new"),
            "{changes}"
        );

        // The change map, applied to a light shot before the change.
        let mut map = vec![0.0_f32; W * H];
        assert_eq!(
            unsafe {
                seiza_flat_change_map(
                    earlier.as_ptr(),
                    later.as_ptr(),
                    W,
                    H,
                    map.as_mut_ptr(),
                    map.len(),
                    &mut error,
                )
            },
            1,
            "{}",
            take(error)
        );
        let mut image = vec![1000.0_f32; 70 * 70];
        assert!(
            unsafe {
                seiza_flat_apply_change_map(
                    image.as_mut_ptr(),
                    70,
                    70,
                    1,
                    map.as_ptr(),
                    W,
                    H,
                    BIN,
                    true,
                    &mut error,
                )
            },
            "{}",
            take(error)
        );
        assert!(image.iter().all(|value| value.is_finite()));

        // Lights shot after the change, weighed against the earlier session.
        let fit = json(
            unsafe {
                seiza_flat_fit_change_json(
                    later.as_ptr(),
                    earlier.as_ptr(),
                    map.as_ptr(),
                    W,
                    H,
                    c"before".as_ptr(),
                    &mut error,
                )
            },
            error,
        );
        assert!(fit.get("outcome").is_some(), "{fit}");
        let candidates: Vec<f32> = earlier.iter().chain(&later).copied().collect();
        let choice = json(
            unsafe {
                seiza_flat_choose_response_json(
                    later.as_ptr(),
                    candidates.as_ptr(),
                    2,
                    W,
                    H,
                    ptr::null(),
                    &mut error,
                )
            },
            error,
        );
        assert!(
            choice.get("winners").is_some() && choice.get("pairs").is_some(),
            "{choice}"
        );
    }

    #[test]
    fn planes_bin_round_trip_through_the_cache_and_verdicts_combine() {
        assert!(seiza_flat_bin_factor(6000, 4000, false) >= 1);
        let frame = vec![2.0_f32; 30 * 20];
        let mut binned = vec![0.0_f32; 10 * 5];
        let mut error = ptr::null_mut();
        assert!(!unsafe {
            seiza_flat_bin_frame(
                frame.as_ptr(),
                30,
                20,
                1,
                3,
                binned.as_mut_ptr(),
                binned.len(),
                &mut error,
            )
        });
        assert!(take(error).contains("holds 50 samples"));
        let mut binned = vec![0.0_f32; 10 * 6];
        assert!(unsafe {
            seiza_flat_bin_frame(
                frame.as_ptr(),
                30,
                20,
                1,
                3,
                binned.as_mut_ptr(),
                60,
                &mut error,
            )
        });
        assert!(binned.iter().all(|&value| value == 2.0));

        // Zero is the format's mark for a missing sample, so start above it.
        let plane: Vec<f32> = (1..=12).map(|index| index as f32 / 8.0).collect();
        let needed = unsafe { seiza_flat_encode_plane(plane.as_ptr(), 4, 3, ptr::null_mut(), 0) };
        let mut bytes = vec![0_u8; needed];
        unsafe { seiza_flat_encode_plane(plane.as_ptr(), 4, 3, bytes.as_mut_ptr(), bytes.len()) };
        let (mut width, mut height) = (0_usize, 0_usize);
        assert_eq!(
            unsafe {
                seiza_flat_decode_plane(
                    bytes.as_ptr(),
                    bytes.len(),
                    ptr::null_mut(),
                    0,
                    &mut width,
                    &mut height,
                    &mut error,
                )
            },
            1
        );
        assert_eq!((width, height), (4, 3));
        let mut decoded = vec![0.0_f32; 12];
        let read = unsafe {
            seiza_flat_decode_plane(
                bytes.as_ptr(),
                bytes.len(),
                decoded.as_mut_ptr(),
                12,
                &mut width,
                &mut height,
                &mut error,
            )
        };
        assert!(
            plane
                .iter()
                .zip(&decoded)
                .all(|(a, b)| (a - b).abs() < 1e-4),
            "{read} {plane:?} {decoded:?}"
        );
        assert_eq!(
            unsafe {
                seiza_flat_decode_plane(
                    [1_u8, 2].as_ptr(),
                    2,
                    ptr::null_mut(),
                    0,
                    &mut width,
                    &mut height,
                    &mut error,
                )
            },
            0
        );

        let verdict = text(
            unsafe {
                seiza_flat_combine_verdicts_json(
                    c"[\"before\", \"undecided\"]".as_ptr(),
                    &mut error,
                )
            },
            error,
        );
        assert_eq!(verdict, "\"before\"");

        // A misspelt option is refused rather than ignored.
        let plane = vec![1.0_f32; 64 * 64];
        let refused = unsafe {
            seiza_dust_find_json(
                plane.as_ptr(),
                64,
                64,
                1,
                c"{\"edge_sigmaa\": 3}".as_ptr(),
                &mut error,
            )
        };
        assert!(refused.is_null());
        assert!(take(error).contains("unknown field"));
    }
}
