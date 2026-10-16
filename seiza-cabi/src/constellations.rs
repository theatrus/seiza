//! Constellation stick figures (`seiza::constellations`), as Python's
//! `constellation_figures` and `project_constellations` give them.
//!
//! The figures come from the Constellation Lines dataset by Marc van der
//! Sluys (CC BY 4.0). Anything that shows them must carry the `attribution`
//! both calls return.

use super::{clear_error, ffi_result, owned_json, required_str};
use crate::parallax::WcsRequest;
use seiza::constellations;
use std::ffi::c_char;
use std::ptr;

/// Every constellation figure and the stars its lines use, for a host that
/// draws figures its own way. Returns JSON, released with
/// `seiza_string_free`: `{"attribution", "epoch", "figures": [{"abbr",
/// "name", "segments": [[hr, hr], ...], "labels": [[ra, dec], ...]}],
/// "stars": [{"hr", "ra", "dec", "mag"}]}`, positions in ICRS degrees at
/// the Julian `epoch`. Null with `error_out` set on failure.
///
/// # Safety
///
/// When non-null, `error_out` must point to writable storage for one
/// pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_constellation_figures_json(
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || owned_json(&constellations::dataset())).unwrap_or(ptr::null_mut())
}

/// The constellation figures of a solved `width × height` image, as pixel
/// polylines: each segment follows its great circle, the parts behind the
/// tangent plane are dropped, and lines that leave the frame run to its
/// edge. `wcs_json` is the solution as `seiza_solve_image_json` returns it
/// (`pixelOrigin` 0 or 1). Returns JSON, released with `seiza_string_free`:
/// `{"attribution", "figures": [{"abbr", "name", "polylines": [[[x, y],
/// ...]], "label": [x, y] or null}]}`, pixels counted from 0. Null with
/// `error_out` set on failure.
///
/// # Safety
///
/// `wcs_json` must be a NUL-terminated UTF-8 string. When non-null,
/// `error_out` must point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_constellations_project_json(
    wcs_json: *const c_char,
    width: u32,
    height: u32,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        if width == 0 || height == 0 {
            return Err("the image needs a width and a height".into());
        }
        let wcs = serde_json::from_str::<WcsRequest>(&required_str(wcs_json, "WCS JSON")?)
            .map_err(|error| format!("invalid WCS JSON: {error}"))?
            .wcs()?;
        owned_json(&constellations::project(&wcs, (width, height)))
    })
    .unwrap_or(ptr::null_mut())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{CStr, CString};

    fn json(pointer: *mut c_char) -> serde_json::Value {
        assert!(!pointer.is_null());
        let value =
            serde_json::from_str(&unsafe { CStr::from_ptr(pointer) }.to_string_lossy()).unwrap();
        unsafe { crate::seiza_string_free(pointer) };
        value
    }

    #[test]
    fn the_figures_and_a_solved_images_projection_come_with_their_attribution() {
        let mut error = ptr::null_mut();
        let data = json(unsafe { seiza_constellation_figures_json(&mut error) });
        assert!(data["attribution"].as_str().unwrap().contains("CC BY 4.0"));
        assert_eq!(data["figures"].as_array().unwrap().len(), 88);
        let orion = data["figures"]
            .as_array()
            .unwrap()
            .iter()
            .find(|figure| figure["abbr"] == "Ori")
            .unwrap();
        assert_eq!(orion["name"], "Orion");
        assert!(!orion["segments"].as_array().unwrap().is_empty());
        assert_eq!(data["stars"].as_array().unwrap().len(), 697);

        // A wide field on Orion's belt shows Orion.
        let wcs =
            seiza::Wcs::from_center_scale_rotation((83.8, -1.2), (999.5, 749.5), 60.0, 0.0, false);
        let request = CString::new(
            serde_json::json!({"crval": [wcs.crval.0, wcs.crval.1], "crpix": [wcs.crpix.0, wcs.crpix.1], "cd": wcs.cd})
                .to_string(),
        )
        .unwrap();
        let projected = json(unsafe {
            seiza_constellations_project_json(request.as_ptr(), 2000, 1500, &mut error)
        });
        let orion = projected["figures"]
            .as_array()
            .unwrap()
            .iter()
            .find(|figure| figure["abbr"] == "Ori")
            .unwrap();
        assert!(
            !orion["polylines"].as_array().unwrap().is_empty(),
            "{orion}"
        );
        for point in orion["polylines"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|line| line.as_array().unwrap())
        {
            let (x, y) = (point[0].as_f64().unwrap(), point[1].as_f64().unwrap());
            assert!(
                (-0.5..2000.5).contains(&x) && (-0.5..1500.5).contains(&y),
                "{point}"
            );
        }

        let refused =
            unsafe { seiza_constellations_project_json(c"{}".as_ptr(), 2000, 1500, &mut error) };
        assert!(refused.is_null());
        let message = unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned();
        unsafe { crate::seiza_string_free(error) };
        assert!(message.contains("WCS JSON"), "{message}");
    }
}
