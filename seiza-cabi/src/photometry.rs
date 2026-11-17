//! Thin JSON and planar-buffer translation of forced aperture photometry.
use super::*;
use seiza_stacking::{ApertureMaps, ApertureMeasurement, AperturePhotometryOptions};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AperturePhotometryRequest {
    /// Zero-based [x, y] positions, pixel centres on whole numbers.
    positions: Vec<[f64; 2]>,
    options: AperturePhotometryOptions,
}

#[derive(Serialize)]
struct AperturePhotometryResponse {
    /// One per requested position, in request order.
    measurements: Vec<ApertureMeasurement>,
}

fn photometry_error(error: seiza_stacking::Error) -> String {
    match error {
        seiza_stacking::Error::Photometry(message) => message,
        other => other.to_string(),
    }
}

/// Measure a circular aperture at each supplied position on one linear mono
/// plane. Request JSON is {"positions": [[x, y], ...], "options": {"geometry":
/// {"apertureRadius", "annulusInner", "annulusOuter"}, "saturation"?, "gain"?}}.
/// Positions are zero-based with pixel centres on whole numbers and are
/// measured where given; nothing is detected or moved. mask, saturation_mask
/// and uncertainty are nullable width*height planes: mask is nonzero where a
/// pixel must not be used, saturation_mask nonzero where it clipped, and
/// uncertainty each pixel's total 1-sigma noise in the image's units.
///
/// Returns {"measurements": [...]}, one per position in order, each with x, y,
/// flux, fluxError, magnitude, magnitudeError, background, backgroundNoise,
/// apertureSum, apertureArea, annulusPixels, peak and flags {saturated,
/// missingCoverage, unusable}. Flux is signed; values that could not be
/// measured are null. The noise model is the one
/// seiza_stacking::aperture_photometry documents. Free the result with
/// seiza_string_free. No input buffer is modified.
///
/// # Safety
/// data points to width*height readable floats. mask and saturation_mask, if
/// non-null, point to width*height readable bytes, and uncertainty, if
/// non-null, to width*height readable floats. request_json is NUL-terminated
/// UTF-8. error_out, if non-null, points to writable pointer storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_aperture_photometry_json(
    data: *const f32,
    width: usize,
    height: usize,
    mask: *const u8,
    saturation_mask: *const u8,
    uncertainty: *const f32,
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let request: AperturePhotometryRequest =
            parse_request(request_json, "aperture photometry")?;
        let length = display_length(width, height, 1)?;
        let samples = unsafe { required_f32_slice(data, length, "aperture photometry image")? };
        let image =
            LinearImage::new(width, height, 1, samples.to_vec()).map_err(photometry_error)?;
        let bytes = |plane: *const u8| {
            (!plane.is_null()).then(|| unsafe { std::slice::from_raw_parts(plane, length) })
        };
        let maps = ApertureMaps {
            mask: bytes(mask),
            saturation: bytes(saturation_mask),
            uncertainty: if uncertainty.is_null() {
                None
            } else {
                Some(unsafe { required_f32_slice(uncertainty, length, "uncertainty map")? })
            },
        };
        let measurements = seiza_stacking::aperture_photometry(
            &image,
            &request.positions,
            &request.options,
            &maps,
        )
        .map_err(photometry_error)?;
        owned_json(&AperturePhotometryResponse { measurements })
    })
    .unwrap_or(ptr::null_mut())
}

#[cfg(test)]
mod tests {
    use super::*;
    use seiza_stacking::ApertureGeometry;

    #[test]
    fn json_measurements_match_the_core_with_and_without_maps() {
        let (width, height) = (64_usize, 48_usize);
        let data: Vec<f32> = (0..width * height)
            .map(|i| {
                let (x, y) = ((i % width) as f64, (i / width) as f64);
                let r2 = (x - 30.4).powi(2) + (y - 22.7).powi(2);
                (100.0 + 0.2 * x + 900.0 * (-r2 / 5.0).exp()) as f32
            })
            .collect();
        let mut mask = vec![0_u8; width * height];
        mask[23 * width + 31] = 1;
        let mut clipped = vec![0_u8; width * height];
        clipped[22 * width + 30] = 1;
        let uncertainty = vec![2.0_f32; width * height];
        let positions = vec![[30.4, 22.7], [10.0, 40.0], [-50.0, 3.0]];
        let mut options = AperturePhotometryOptions::new(ApertureGeometry::new(5.0, 8.0, 12.0));
        options.gain = Some(1.5);
        let image = LinearImage::new(width, height, 1, data.clone()).unwrap();
        let request = CString::new(
            serde_json::json!({"positions": positions, "options": options}).to_string(),
        )
        .unwrap();
        let call = |mask: *const u8, clipped: *const u8, uncertainty: *const f32| {
            let mut error = ptr::null_mut();
            let output = unsafe {
                seiza_aperture_photometry_json(
                    data.as_ptr(),
                    width,
                    height,
                    mask,
                    clipped,
                    uncertainty,
                    request.as_ptr(),
                    &mut error,
                )
            };
            assert!(error.is_null());
            let value: serde_json::Value =
                unsafe { serde_json::from_str(CStr::from_ptr(output).to_str().unwrap()).unwrap() };
            unsafe { seiza_string_free(output) };
            value
        };
        let expected = |maps: &ApertureMaps<'_>| {
            let measurements =
                seiza_stacking::aperture_photometry(&image, &positions, &options, maps).unwrap();
            serde_json::json!({ "measurements": measurements })
        };
        let plain = call(ptr::null(), ptr::null(), ptr::null());
        assert_eq!(plain, expected(&ApertureMaps::default()));
        assert_eq!(plain["measurements"][2]["flags"]["unusable"], true);
        assert!(plain["measurements"][2]["flux"].is_null());
        assert!(plain["measurements"][0]["fluxError"].as_f64().unwrap() > 0.0);
        let mapped = call(mask.as_ptr(), clipped.as_ptr(), uncertainty.as_ptr());
        assert_eq!(
            mapped,
            expected(&ApertureMaps {
                mask: Some(&mask),
                saturation: Some(&clipped),
                uncertainty: Some(&uncertainty),
            })
        );
        assert_eq!(mapped["measurements"][0]["flags"]["saturated"], true);
        assert_eq!(mapped["measurements"][0]["flags"]["missingCoverage"], true);
        assert_ne!(mapped, plain);
    }

    #[test]
    fn bad_requests_report_an_error_and_return_null() {
        let data = [1.0_f32; 16];
        for request in [
            r#"{"positions":[[2,2]],"options":{"geometry":{"apertureRadius":3,"annulusInner":2,"annulusOuter":4}}}"#,
            r#"{"positions":[[2,2]]}"#,
            r#"{"positions":[[2,2]],"options":{"geometry":{"apertureRadius":1,"annulusInner":2,"annulusOuter":4},"detect":true}}"#,
        ] {
            let request = CString::new(request).unwrap();
            let mut error = ptr::null_mut();
            let output = unsafe {
                seiza_aperture_photometry_json(
                    data.as_ptr(),
                    4,
                    4,
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    request.as_ptr(),
                    &mut error,
                )
            };
            assert!(output.is_null());
            assert!(!error.is_null());
            unsafe { seiza_string_free(error) };
        }
    }
}
