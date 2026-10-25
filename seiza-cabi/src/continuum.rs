//! Thin JSON and planar-buffer translations of the continuum core.
use super::*;
use seiza_stacking::{
    ContinuumFitOptions, ContinuumPhotometryOptions, ContinuumPlan, ContinuumSources, ContinuumStar,
};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ContinuumFitRequest {
    sources: ContinuumSources,
    stars: Vec<ContinuumStar>,
    #[serde(default)]
    options: ContinuumFitOptions,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ContinuumMeasureRequest {
    sources: ContinuumSources,
    #[serde(default)]
    positions: Vec<[f64; 2]>,
    #[serde(default, alias = "fit_options")]
    fit_options: ContinuumFitOptions,
    #[serde(default, alias = "photometry_options")]
    photometry_options: ContinuumPhotometryOptions,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ContinuumSubtractRequest {
    sources: ContinuumSources,
    plan: ContinuumPlan,
}

/// Fit premeasured common stellar fluxes. Request JSON contains sources, stars
/// and optional options, as defined by the continuum core. Returns a tagged
/// fitted/skipped JSON outcome; free it with seiza_string_free. A skip is not
/// an API error and never supplies a subtraction plan.
///
/// # Safety
/// request_json is a NUL-terminated UTF-8 object; error_out, if non-null,
/// points to writable pointer storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_continuum_fit_json(
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let request: ContinuumFitRequest = parse_request(request_json, "continuum fit")?;
        let outcome =
            seiza_stacking::fit_continuum(&request.stars, &request.sources, &request.options)
                .map_err(color_request_error)?;
        owned_json(&outcome)
    })
    .unwrap_or(ptr::null_mut())
}

unsafe fn decode_planes(
    data: *const f32,
    width: usize,
    height: usize,
    planes: usize,
    sources: &ContinuumSources,
) -> Result<Vec<LinearImage>, String> {
    if !(2..=4).contains(&planes)
        || sources.references.len() + 1 != planes
        || (sources.width, sources.height) != (width, height)
    {
        return Err("Continuum data needs NB followed by one to three planar references on its declared grid".into());
    }
    let length = display_length(width, height, planes)?;
    let samples = unsafe { required_f32_slice(data, length, "continuum planes")? };
    let size = length / planes;
    samples
        .chunks_exact(size)
        .map(|plane| {
            LinearImage::new(width, height, 1, plane.to_vec()).map_err(color_request_error)
        })
        .collect()
}

/// Measure and fit aligned linear planes. Data is planar: narrowband, then
/// references in sources order. Request JSON contains sources, optional
/// positions, fitOptions and photometryOptions. Empty positions asks Seiza to
/// detect stars. Sources declare exact prepared revisions, linear transfer and
/// physical saturation in each plane's units. Returns fitted/skipped JSON;
/// free it with seiza_string_free. The input buffer is never modified.
///
/// # Safety
/// data points to width*height*planes readable floats (planes 2..4).
/// request_json is NUL-terminated UTF-8. error_out, if non-null, is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_continuum_measure_fit_json(
    data: *const f32,
    width: usize,
    height: usize,
    planes: usize,
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let request: ContinuumMeasureRequest =
            parse_request(request_json, "continuum measurement")?;
        let images = unsafe { decode_planes(data, width, height, planes, &request.sources)? };
        let references: Vec<_> = images[1..].iter().collect();
        let outcome = seiza_stacking::measure_and_fit_continuum(
            &images[0],
            &references,
            &request.sources,
            &request.positions,
            &request.fit_options,
            &request.photometry_options,
        )
        .map_err(color_request_error)?;
        owned_json(&outcome)
    })
    .unwrap_or(ptr::null_mut())
}

/// Measure with producer-derived clipping masks. Like seiza_continuum_measure_fit_json,
/// with one planar byte mask per input (NB then references). Zero is usable,
/// one is clipped. Each source declares clippingRevision; saturation may be omitted.
/// Returns the same fitted/skipped JSON and never changes either input buffer.
///
/// # Safety
/// data points to width*height*planes floats and clipping_masks to the same
/// number of readable bytes. request_json is NUL-terminated UTF-8. error_out,
/// if non-null, points to writable pointer storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_continuum_measure_masked_fit_json(
    data: *const f32,
    width: usize,
    height: usize,
    planes: usize,
    clipping_masks: *const u8,
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let request: ContinuumMeasureRequest =
            parse_request(request_json, "masked continuum measurement")?;
        let images = unsafe { decode_planes(data, width, height, planes, &request.sources)? };
        let length = display_length(width, height, planes)?;
        if clipping_masks.is_null() {
            return Err("Continuum clipping masks are null".into());
        }
        let mask_data = unsafe { std::slice::from_raw_parts(clipping_masks, length) };
        let masks: Vec<_> = mask_data.chunks_exact(width * height).collect();
        let references: Vec<_> = images[1..].iter().collect();
        let outcome = seiza_stacking::measure_and_fit_continuum_masked(
            &images[0],
            &references,
            &request.sources,
            &masks,
            &request.positions,
            &request.fit_options,
            &request.photometry_options,
        )
        .map_err(color_request_error)?;
        owned_json(&outcome)
    })
    .unwrap_or(ptr::null_mut())
}

/// Apply a continuum plan, preserving signed linear residuals and coverage.
/// Input data is planar NB then references. Request JSON contains sources and
/// plan. Output is width*height mono floats and may alias an input plane.
/// Errors leave the output untouched. No normalization or clipping is applied.
///
/// # Safety
/// data points to width*height*planes readable floats (planes 2..4), output
/// to width*height writable floats. request_json is NUL-terminated UTF-8.
/// error_out, if non-null, points to writable pointer storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_continuum_subtract_json(
    data: *const f32,
    width: usize,
    height: usize,
    planes: usize,
    request_json: *const c_char,
    output: *mut f32,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let request: ContinuumSubtractRequest =
            parse_request(request_json, "continuum subtraction")?;
        let images = unsafe { decode_planes(data, width, height, planes, &request.sources)? };
        let references: Vec<_> = images[1..].iter().collect();
        let image = seiza_stacking::subtract_continuum(
            &images[0],
            &references,
            &request.sources,
            &request.plan,
        )
        .map_err(color_request_error)?;
        let out =
            unsafe { required_f32_slice_mut(output, image.data.len(), "continuum residual")? };
        out.copy_from_slice(&image.data);
        Ok(())
    })
    .is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn json_fit_matches_core_and_image_measurement_returns_a_structured_skip() {
        use seiza_stacking::{ContinuumFitOutcome, ContinuumPhotometry, ContinuumTransfer};
        let source = |name: &str| seiza_stacking::ContinuumSource {
            name: name.into(),
            revision: "v1".into(),
            transfer: ContinuumTransfer::Linear,
            saturation: Some(65535.0),
            clipping_revision: None,
        };
        let sources = ContinuumSources {
            width: 256,
            height: 256,
            narrowband: source("ha"),
            references: vec![source("red")],
        };
        let measurement = |flux: f64, i: usize| ContinuumPhotometry {
            x: 25.0 + (i % 7) as f64 * 30.0,
            y: 25.0 + (i / 7) as f64 * 20.0,
            flux,
            peak: 100.0 + flux / 20.0,
            fwhm: 3.0,
            snr: 100.0,
        };
        let stars: Vec<_> = (0..40)
            .map(|i| {
                let flux = 1000.0 + i as f64 * 100.0;
                ContinuumStar {
                    narrowband: measurement(flux * 0.35, i),
                    references: vec![measurement(flux, i)],
                }
            })
            .collect();
        let expected =
            seiza_stacking::fit_continuum(&stars, &sources, &Default::default()).unwrap();
        let request =
            CString::new(serde_json::json!({"sources":sources,"stars":stars}).to_string()).unwrap();
        let mut error = ptr::null_mut();
        let output = unsafe { seiza_continuum_fit_json(request.as_ptr(), &mut error) };
        assert!(error.is_null());
        assert!(!output.is_null());
        let actual: serde_json::Value =
            unsafe { serde_json::from_str(CStr::from_ptr(output).to_str().unwrap()).unwrap() };
        unsafe { seiza_string_free(output) };
        assert_eq!(actual, serde_json::to_value(expected).unwrap());
        let positions: Vec<_> = (0..16)
            .map(|i| [35.0 + (i % 4) as f64 * 60.0, 35.0 + (i / 4) as f64 * 60.0])
            .collect();
        let plane = |factor: f64, background: f64| {
            (0..256 * 256)
                .map(|i| {
                    let x = (i % 256) as f64;
                    let y = (i / 256) as f64;
                    (background + 0.1 * x - 0.05 * y
                        + positions
                            .iter()
                            .enumerate()
                            .map(|(i, p)| {
                                factor
                                    * (1000.0 + i as f64 * 50.0)
                                    * (-((x - p[0]).powi(2) + (y - p[1]).powi(2))
                                        / (2.0 * 1.4_f64.powi(2)))
                                    .exp()
                            })
                            .sum::<f64>()) as f32
                })
                .collect::<Vec<_>>()
        };
        let narrowband = LinearImage::new(256, 256, 1, plane(0.35, 100.0)).unwrap();
        let reference = LinearImage::new(256, 256, 1, plane(1.0, 300.0)).unwrap();
        let expected = seiza_stacking::measure_and_fit_continuum(
            &narrowband,
            &[&reference],
            &sources,
            &positions,
            &Default::default(),
            &Default::default(),
        )
        .unwrap();
        assert!(matches!(&expected, ContinuumFitOutcome::Fitted { .. }));
        let request=CString::new(serde_json::json!({"sources":sources,"positions":positions,"fitOptions":{"minimumStars":12},"photometryOptions":{"apertureRadius":8}}).to_string()).unwrap();
        let data: Vec<_> = narrowband
            .data
            .iter()
            .chain(&reference.data)
            .copied()
            .collect();
        let output = unsafe {
            seiza_continuum_measure_fit_json(
                data.as_ptr(),
                256,
                256,
                2,
                request.as_ptr(),
                &mut error,
            )
        };
        assert!(error.is_null());
        assert!(!output.is_null());
        let actual: serde_json::Value =
            unsafe { serde_json::from_str(CStr::from_ptr(output).to_str().unwrap()).unwrap() };
        unsafe { seiza_string_free(output) };
        assert_eq!(actual, serde_json::to_value(expected).unwrap());
        let mut masked_sources = sources.clone();
        for source in
            std::iter::once(&mut masked_sources.narrowband).chain(&mut masked_sources.references)
        {
            source.saturation = None;
            source.clipping_revision = Some("producer-v1".into());
        }
        let masks = vec![0_u8; data.len()];
        let request = CString::new(
            serde_json::json!({"sources":masked_sources,"positions":positions}).to_string(),
        )
        .unwrap();
        let expected = seiza_stacking::measure_and_fit_continuum_masked(
            &narrowband,
            &[&reference],
            &masked_sources,
            &[&masks[..256 * 256], &masks[256 * 256..]],
            &positions,
            &Default::default(),
            &Default::default(),
        )
        .unwrap();
        let output = unsafe {
            seiza_continuum_measure_masked_fit_json(
                data.as_ptr(),
                256,
                256,
                2,
                masks.as_ptr(),
                request.as_ptr(),
                &mut error,
            )
        };
        assert!(error.is_null());
        assert!(!output.is_null());
        let actual: serde_json::Value =
            unsafe { serde_json::from_str(CStr::from_ptr(output).to_str().unwrap()).unwrap() };
        unsafe { seiza_string_free(output) };
        assert_eq!(actual, serde_json::to_value(expected).unwrap());
        let request = CString::new(serde_json::json!({"sources":sources}).to_string()).unwrap();
        let image = vec![0.0; 256 * 256 * 2];
        let output = unsafe {
            seiza_continuum_measure_fit_json(
                image.as_ptr(),
                256,
                256,
                2,
                request.as_ptr(),
                &mut error,
            )
        };
        assert!(error.is_null());
        assert!(!output.is_null());
        let actual: ContinuumFitOutcome =
            unsafe { serde_json::from_str(CStr::from_ptr(output).to_str().unwrap()).unwrap() };
        unsafe { seiza_string_free(output) };
        assert!(matches!(
            actual,
            ContinuumFitOutcome::Skipped {
                reason: seiza_stacking::ContinuumSkipReason::NoStars,
                ..
            }
        ));
    }

    #[test]
    fn signed_subtraction_is_alias_safe_and_refuses_changed_revisions_transactionally() {
        let mut data = vec![0.2, 0.1, 0.5, 0.5, 0.3, 0.4];
        let source = |name: &str| serde_json::json!({"name":name,"revision":"v1","transfer":"linear","saturation":65535});
        let sources = serde_json::json!({"width":3,"height":1,"narrowband":source("ha"),"references":[source("red")]});
        let plan = serde_json::json!({"schemaVersion":1,"sources":sources,"coefficients":[0.35],"origin":"manual","diagnostics":null});
        let request =
            CString::new(serde_json::json!({"sources":sources,"plan":plan}).to_string()).unwrap();
        let mut error = ptr::null_mut();
        assert!(unsafe {
            seiza_continuum_subtract_json(
                data.as_ptr(),
                3,
                1,
                2,
                request.as_ptr(),
                data.as_mut_ptr(),
                &mut error,
            )
        });
        assert!(error.is_null());
        assert!((data[0] - 0.025).abs() < 1e-6);
        assert!(data[1] < 0.0);
        let mut bad: serde_json::Value = serde_json::from_str(request.to_str().unwrap()).unwrap();
        bad["sources"]["narrowband"]["revision"] = "changed".into();
        let bad = CString::new(bad.to_string()).unwrap();
        let before = data.clone();
        assert!(!unsafe {
            seiza_continuum_subtract_json(
                data.as_ptr(),
                3,
                1,
                2,
                bad.as_ptr(),
                data.as_mut_ptr(),
                &mut error,
            )
        });
        assert!(!error.is_null());
        unsafe { seiza_string_free(error) };
        assert_eq!(data, before);
    }
}
