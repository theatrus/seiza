//! Buffer and JSON translations for emission and display structure blending.
use super::*;
use seiza_stacking::{EmissionBlendLine, EmissionBlendOptions};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EmissionBlendRequest {
    lines: Vec<EmissionBlendLine>,
    #[serde(default)]
    options: EmissionBlendOptions,
}

/// Add linear mono residuals to linear RGB using exact continuum plans and
/// empirical reference units. Request JSON contains lines and optional options.
/// Residuals are planar in lines order; broadband/output are interleaved RGB.
/// Errors leave output unchanged. Output may alias either input buffer.
///
/// # Safety
/// broadband/output each address width*height*3 floats; residuals addresses
/// width*height*line-count floats (nullable only for zero lines). request_json
/// is NUL-terminated UTF-8 and error_out, if non-null, is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_color_blend_emission_json(
    broadband: *const f32,
    residuals: *const f32,
    width: usize,
    height: usize,
    request_json: *const c_char,
    output: *mut f32,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let request: EmissionBlendRequest = parse_request(request_json, "emission blend")?;
        let length = display_length(width, height, 3)?;
        let broadband = LinearImage::new(
            width,
            height,
            3,
            unsafe { required_f32_slice(broadband, length, "broadband RGB")? }.to_vec(),
        )
        .map_err(color_request_error)?;
        let count = request.lines.len();
        let images = if count == 0 {
            Vec::new()
        } else {
            let length = display_length(width, height, count)?;
            unsafe { required_f32_slice(residuals, length, "emission residuals")? }
                .chunks_exact(width * height)
                .map(|data| {
                    LinearImage::new(width, height, 1, data.to_vec()).map_err(color_request_error)
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        let residuals: Vec<_> = images.iter().collect();
        let result = seiza_stacking::blend_emission_rgb(
            &broadband,
            &residuals,
            &request.lines,
            &request.options,
        )
        .map_err(color_request_error)?;
        unsafe { required_f32_slice_mut(output, result.data.len(), "emission RGB output")? }
            .copy_from_slice(&result.data);
        Ok(())
    })
    .is_some()
}

/// Blend aligned bounded display RGB structures. Amount zero keeps narrowband,
/// one keeps broadband; intermediate amounts use common coverage. Output may
/// alias inputs and is unchanged on an error. No star extraction is performed.
///
/// # Safety
/// Each image pointer addresses width*height*3 floats (output writable).
/// error_out, if non-null, points to writable pointer storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_color_blend_display_structure(
    narrowband: *const f32,
    broadband: *const f32,
    width: usize,
    height: usize,
    amount: f64,
    output: *mut f32,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let length = display_length(width, height, 3)?;
        let n = LinearImage::new(
            width,
            height,
            3,
            unsafe { required_f32_slice(narrowband, length, "narrowband display")? }.to_vec(),
        )
        .map_err(color_request_error)?;
        let b = LinearImage::new(
            width,
            height,
            3,
            unsafe { required_f32_slice(broadband, length, "broadband display")? }.to_vec(),
        )
        .map_err(color_request_error)?;
        let result =
            seiza_stacking::blend_display_structure(&n, &b, amount).map_err(color_request_error)?;
        unsafe { required_f32_slice_mut(output, length, "display structure output")? }
            .copy_from_slice(&result.data);
        Ok(())
    })
    .is_some()
}

/// Add a linear RGB contribution with multiplicative calibration gains.
/// Inputs may alias output; errors leave output unchanged.
///
/// # Safety
/// Image pointers address width*height*3 floats; gains addresses three floats.
/// Output and non-null error_out must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_color_add_rgb_contribution(
    base: *const f32,
    contribution: *const f32,
    width: usize,
    height: usize,
    gains: *const f32,
    output: *mut f32,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let length = display_length(width, height, 3)?;
        let base = LinearImage::new(
            width,
            height,
            3,
            unsafe { required_f32_slice(base, length, "base RGB")? }.to_vec(),
        )
        .map_err(color_request_error)?;
        let contribution = LinearImage::new(
            width,
            height,
            3,
            unsafe { required_f32_slice(contribution, length, "RGB contribution")? }.to_vec(),
        )
        .map_err(color_request_error)?;
        let gains = unsafe { required_f32_slice(gains, 3, "RGB gains")? };
        let result = seiza_stacking::add_rgb_contribution(
            &base,
            &contribution,
            [gains[0], gains[1], gains[2]],
        )
        .map_err(color_request_error)?;
        unsafe { required_f32_slice_mut(output, length, "RGB output")? }
            .copy_from_slice(&result.data);
        Ok(())
    })
    .is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{ffi::CString, ptr};
    #[test]
    fn blends_are_alias_safe_and_errors_leave_output_unchanged() {
        let source =
            |name: &str| serde_json::json!({"name":name,"revision":"v1","transfer":"linear"});
        let request=CString::new(serde_json::json!({"lines":[{"plan":{"schemaVersion":1,"sources":{"width":1,"height":1,"narrowband":source("ha"),"references":[source("red")]},"coefficients":[0.2],"origin":"manual","diagnostics":null},"referenceChannels":[0],"amount":0.25}]}).to_string()).unwrap();
        let mut data = [0.2_f32, 0.3, 0.4];
        let residual = [0.04];
        let mut error = ptr::null_mut();
        assert!(unsafe {
            seiza_color_blend_emission_json(
                data.as_ptr(),
                residual.as_ptr(),
                1,
                1,
                request.as_ptr(),
                data.as_mut_ptr(),
                &mut error,
            )
        });
        assert!(error.is_null());
        assert!((data[0] - 0.25).abs() < 1e-6);
        let before = data;
        let gains = [2.0_f32, 1.0, 0.5];
        assert!(unsafe {
            seiza_color_add_rgb_contribution(
                data.as_ptr(),
                data.as_ptr(),
                1,
                1,
                gains.as_ptr(),
                data.as_mut_ptr(),
                &mut error,
            )
        });
        for i in 0..3 {
            assert!((data[i] - before[i] * (1.0 + gains[i])).abs() < 1e-6);
        }
        let before = data;
        assert!(!unsafe {
            seiza_color_blend_display_structure(
                data.as_ptr(),
                data.as_ptr(),
                1,
                1,
                2.0,
                data.as_mut_ptr(),
                &mut error,
            )
        });
        assert_eq!(data, before);
        assert!(!error.is_null());
        unsafe { seiza_string_free(error) };
    }
}

/// Reduce excess display green in-place, preserving red/blue and coverage.
/// Invalid input or amount leaves all samples unchanged.
///
/// # Safety
/// data addresses width*height*3 writable floats; non-null error_out is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_color_reduce_green_excess(
    data: *mut f32,
    width: usize,
    height: usize,
    amount: f64,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let length = display_length(width, height, 3)?;
        let samples = unsafe { required_f32_slice_mut(data, length, "display RGB")? };
        let mut image =
            LinearImage::new(width, height, 3, samples.to_vec()).map_err(color_request_error)?;
        seiza_stacking::reduce_green_excess(&mut image, amount).map_err(color_request_error)?;
        samples.copy_from_slice(&image.data);
        Ok(())
    })
    .is_some()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EmissionFluxRequest {
    plans: Vec<seiza_stacking::ContinuumPlan>,
    calibration: seiza_stacking::EmissionFluxCalibration,
}

/// Invert calibrated line responses. Planar residuals/output are in calibration
/// line order. JSON contains plans and calibration; the returned owned JSON
/// report must be freed with seiza_string_free. Errors leave output untouched.
///
/// # Safety
/// residuals/output address width*height*lines floats (lines 1..3); output is
/// writable and may alias residuals. request_json is NUL-terminated UTF-8;
/// error_out, if non-null, is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_emission_flux_json(
    residuals: *const f32,
    width: usize,
    height: usize,
    lines: usize,
    request_json: *const c_char,
    output: *mut f32,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        if !(1..=3).contains(&lines) {
            return Err("Flux inversion needs one to three lines".into());
        }
        let request: EmissionFluxRequest = parse_request(request_json, "emission flux")?;
        let length = display_length(width, height, lines)?;
        let images = unsafe { required_f32_slice(residuals, length, "line residuals")? }
            .chunks_exact(width * height)
            .map(|data| {
                LinearImage::new(width, height, 1, data.to_vec()).map_err(color_request_error)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (images, report) = seiza_stacking::solve_emission_flux(
            &images.iter().collect::<Vec<_>>(),
            &request.plans,
            &request.calibration,
        )
        .map_err(color_request_error)?;
        let json = owned_json(&report)?;
        let destination =
            match unsafe { required_f32_slice_mut(output, length, "line flux output") } {
                Ok(destination) => destination,
                Err(error) => {
                    unsafe { seiza_string_free(json) };
                    return Err(error);
                }
            };
        for (image, plane) in images
            .iter()
            .zip(destination.chunks_exact_mut(width * height))
        {
            plane.copy_from_slice(&image.data);
        }
        Ok(json)
    })
    .unwrap_or(ptr::null_mut())
}

#[cfg(test)]
mod flux_and_green_tests {
    use super::*;
    use std::ffi::{CStr, CString};
    #[test]
    fn calibrated_flux_reports_units_and_is_alias_safe() {
        let source =
            |name: &str| serde_json::json!({"name":name,"revision":"exact-v1","transfer":"linear"});
        let sources = serde_json::json!({"width":2,"height":1,"narrowband":source("ha"),"references":[source("red")]});
        let request=CString::new(serde_json::json!({"plans":[{"schemaVersion":1,"sources":sources,"coefficients":[0.2],"origin":"manual","diagnostics":null}],"calibration":{"revision":"lab-v1","unit":"flux/pixel","lines":["ha"],"sources":[sources],"narrowbandResponse":[[1.0]],"referenceResponse":[[[1.0]]],"continuumCoefficients":[[0.2]],"residualZeroLevels":[0.0],"residualCovariance":[[4.0]]}}).to_string()).unwrap();
        let mut data = [4.0, -4.0];
        let mut error = ptr::null_mut();
        let result = unsafe {
            seiza_emission_flux_json(
                data.as_ptr(),
                2,
                1,
                1,
                request.as_ptr(),
                data.as_mut_ptr(),
                &mut error,
            )
        };
        assert!(!result.is_null());
        assert!(error.is_null());
        assert_eq!(data, [5.0, -5.0]);
        let report: serde_json::Value =
            serde_json::from_str(unsafe { CStr::from_ptr(result) }.to_str().unwrap()).unwrap();
        assert_eq!(report["unit"], "flux/pixel");
        assert_eq!(report["fluxCovariance"][0][0], 6.25);
        unsafe { seiza_string_free(result) };
        let mut image = [0.2, 0.8, 0.3];
        assert!(!unsafe {
            seiza_color_reduce_green_excess(image.as_mut_ptr(), 1, 1, 2.0, &mut error)
        });
        assert_eq!(image, [0.2, 0.8, 0.3]);
        unsafe { seiza_string_free(error) };
        error = ptr::null_mut();
        assert!(unsafe {
            seiza_color_reduce_green_excess(image.as_mut_ptr(), 1, 1, 0.5, &mut error)
        });
        assert!((image[1] - 0.55).abs() < 1e-6);
    }
}

#[cfg(test)]
mod star_reduction_tests {
    use super::*;
    #[test]
    fn reduced_stars_match_the_linked_reference_and_validate_before_writing() {
        let mut nebula = [0.1_f32, 0.2, 0.3].repeat(9);
        let starless = [0.2_f32; 27];
        let mut original = Vec::new();
        for level in [0.05_f32, 0.2, 0.05, 0.2, 0.8, 0.2, 0.05, 0.2, 0.05] {
            for color in [1.0, 0.5, 0.25] {
                original.push(1.0 - 0.8 * (1.0 - level * color));
            }
        }
        let before = nebula.clone();
        let mut error = ptr::null_mut();
        assert!(!unsafe {
            seiza_color_screen_rgb_stars_reduced(
                nebula.as_mut_ptr(),
                original.as_ptr(),
                starless.as_ptr(),
                3,
                3,
                2.0,
                &mut error,
            )
        });
        assert_eq!(nebula, before);
        unsafe { seiza_string_free(error) };
        error = ptr::null_mut();
        assert!(unsafe {
            seiza_color_screen_rgb_stars_reduced(
                nebula.as_mut_ptr(),
                original.as_ptr(),
                starless.as_ptr(),
                3,
                3,
                0.5,
                &mut error,
            )
        });
        for (actual, expected) in nebula[12..15].iter().zip([0.55, 0.4, 0.3875]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}
