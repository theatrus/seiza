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
