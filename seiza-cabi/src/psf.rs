//! Empirical PSF fields (`seiza-stars`), as Python's `PsfField` measures
//! them.
//!
//! [`seiza_psf_field_measure`] measures a linear image's PSF from its
//! isolated stars: a mean stamp plus principal components whose
//! coefficients vary as polynomials of degree 1 to 3 across the field, with
//! the wings from saturated stars. The handle then gives the local PSF stamp
//! ([`seiza_psf_field_stamp_at`]) and its shape
//! ([`seiza_psf_field_shape_at`]) anywhere in the image, and a report of the
//! stars and the fit ([`seiza_psf_field_report_json`]).

use super::{clear_error, ffi_result, required_f32_slice, required_f32_slice_mut};
use seiza_stars::{
    CatalogueStar, PsfChannelFit, PsfField, PsfFieldOptions, PsfShape, PsfStar, PsfStarRole,
    SparseFieldPolicy,
};
use serde::{Deserialize, Serialize};
use std::ffi::{CStr, CString, c_char};
use std::ptr;

/// An opaque measured PSF field. Release it with [`seiza_psf_field_free`].
/// It is read-only once made, so several threads may sample it at once;
/// its report string is borrowed and remains valid until it is freed.
pub struct SeizaPsfField {
    pub(crate) field: PsfField,
    report_json: CString,
}

/// A catalogue star at its image position, for isolating the stars a PSF
/// field is built from. Pixels count from 0.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SeizaPsfCatalogueStar {
    pub x: f64,
    pub y: f64,
    /// A magnitude in any one band; only differences between stars count.
    pub magnitude: f64,
}

/// The shape of a PSF at one point, from an elliptical Moffat profile with
/// free β fitted to its stamp. Widths are in pixels; `angle` is the major
/// axis's direction in radians over `[0, π)`, from `+x` towards `+y`;
/// `e1` and `e2` are `χ·cos 2φ` and `χ·sin 2φ` with
/// `χ = (a² − b²)/(a² + b²)`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SeizaPsfShape {
    pub fwhm: f64,
    pub fwhm_major: f64,
    pub fwhm_minor: f64,
    pub e1: f64,
    pub e2: f64,
    pub angle: f64,
    pub beta: f64,
}

impl From<PsfShape> for SeizaPsfShape {
    fn from(shape: PsfShape) -> Self {
        Self {
            fwhm: shape.fwhm,
            fwhm_major: shape.fwhm_major,
            fwhm_minor: shape.fwhm_minor,
            e1: shape.e1,
            e2: shape.e2,
            angle: shape.angle,
            beta: shape.beta,
        }
    }
}

/// `PsfFieldOptions` as JSON; every field is optional and takes the core's
/// default when absent.
#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
struct PsfFieldOptionsRequest {
    stamp_size: Option<usize>,
    components: Option<usize>,
    minimum_stars: Option<usize>,
    maximum_stars: Option<usize>,
    minimum_snr: Option<f64>,
    detection_sigma: Option<f64>,
    saturation: Option<f32>,
    isolation_magnitudes: Option<f64>,
    sparse: Option<SparseFieldPolicy>,
    degree: Option<usize>,
}

impl PsfFieldOptionsRequest {
    fn options(self) -> PsfFieldOptions {
        let defaults = PsfFieldOptions::default();
        PsfFieldOptions {
            stamp_size: self.stamp_size,
            components: self.components.unwrap_or(defaults.components),
            minimum_stars: self.minimum_stars.unwrap_or(defaults.minimum_stars),
            maximum_stars: self.maximum_stars.unwrap_or(defaults.maximum_stars),
            minimum_snr: self.minimum_snr.unwrap_or(defaults.minimum_snr),
            detection_sigma: self.detection_sigma.unwrap_or(defaults.detection_sigma),
            saturation: self.saturation,
            isolation_magnitudes: self
                .isolation_magnitudes
                .unwrap_or(defaults.isolation_magnitudes),
            sparse: self.sparse.unwrap_or(defaults.sparse),
            degree: self.degree,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PsfFieldReportResponse {
    width: usize,
    height: usize,
    channels: usize,
    stamp_size: usize,
    component_count: usize,
    star_count: usize,
    is_global: bool,
    warnings: Vec<String>,
    channel_fits: Vec<PsfChannelFitResponse>,
    stars: Vec<PsfStarResponse>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PsfChannelFitResponse {
    reduced_chi_squared: f64,
    relative_residual: f64,
    degree: usize,
    wing_stars: usize,
    wing_overlap: Option<[f64; 2]>,
}

impl From<&PsfChannelFit> for PsfChannelFitResponse {
    fn from(fit: &PsfChannelFit) -> Self {
        Self {
            reduced_chi_squared: fit.reduced_chi_squared,
            relative_residual: fit.relative_residual,
            degree: fit.degree,
            wing_stars: fit.wing_stars,
            wing_overlap: fit.wing_overlap.map(|(inner, outer)| [inner, outer]),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PsfStarResponse {
    x: f64,
    y: f64,
    flux: f64,
    snr: f64,
    fwhm: f64,
    role: PsfStarRole,
}

impl From<&PsfStar> for PsfStarResponse {
    fn from(star: &PsfStar) -> Self {
        Self {
            x: star.x,
            y: star.y,
            flux: star.flux,
            snr: star.snr,
            fwhm: star.fwhm,
            role: star.role,
        }
    }
}

fn report(field: &PsfField) -> PsfFieldReportResponse {
    PsfFieldReportResponse {
        width: field.width(),
        height: field.height(),
        channels: field.channels(),
        stamp_size: field.stamp_size(),
        component_count: field.component_count(),
        star_count: field.star_count(),
        is_global: field.is_global(),
        warnings: field
            .warnings()
            .iter()
            .map(|warning| warning.to_string())
            .collect(),
        channel_fits: field.channel_fits().iter().map(Into::into).collect(),
        stars: field.stars().iter().map(Into::into).collect(),
    }
}

pub(crate) unsafe fn required_psf_field<'a>(
    field: *const SeizaPsfField,
) -> Result<&'a SeizaPsfField, String> {
    unsafe { field.as_ref() }.ok_or_else(|| "PSF field is required".into())
}

#[unsafe(no_mangle)]
/// Measures the PSF field of interleaved linear `float` samples.
///
/// `channels` must be one or three and `data_length` must equal
/// `width * height * channels`; RGB samples are pixel-interleaved, and each
/// channel gets its own field. Non-finite samples never enter a stamp.
/// `catalogue` holds `catalogue_length` catalogue stars at their image
/// positions, or is null with zero length: with stars, a star gives a stamp
/// only when it matches one, and catalogue neighbours within the isolation
/// magnitudes are masked out of its stamp, or disqualify it when they lie
/// close to its core. Pass null or empty `options_json` for
/// the defaults, or a JSON object with any of `stampSize`, `components`,
/// `minimumStars`, `maximumStars`, `minimumSnr`, `detectionSigma`,
/// `saturation`, `isolationMagnitudes`, `sparse` (`"refuse"`, the
/// default, or `"global"` for one PSF when too few stars allow a varying
/// field) and `degree` (1 to 3; absent chooses it by cross-validation). Returns null with an error when the image, the options or the
/// stars do not allow a field. The handle retains no pointer to the inputs.
///
/// # Safety
/// `data` must point to `data_length` readable floats. A non-null
/// `catalogue` must point to `catalogue_length` readable stars. A non-null
/// `options_json` must be NUL-terminated. When non-null, `error_out` must
/// point to writable storage for one pointer.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn seiza_psf_field_measure(
    data: *const f32,
    data_length: usize,
    width: usize,
    height: usize,
    channels: usize,
    catalogue: *const SeizaPsfCatalogueStar,
    catalogue_length: usize,
    options_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut SeizaPsfField {
    clear_error(error_out);
    ffi_result(error_out, || {
        let data = unsafe { required_f32_slice(data, data_length, "PSF field image")? };
        let catalogue: Vec<CatalogueStar> = if catalogue.is_null() {
            if catalogue_length != 0 {
                return Err("PSF field catalogue is null but its length is not zero".into());
            }
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(catalogue, catalogue_length) }
                .iter()
                .map(|star| CatalogueStar {
                    x: star.x,
                    y: star.y,
                    magnitude: star.magnitude,
                })
                .collect()
        };
        let options = psf_field_options(options_json)?;
        let field = if catalogue.is_empty() {
            PsfField::measure(data, width, height, channels, &options)
        } else {
            PsfField::measure_with_catalogue(data, width, height, channels, &catalogue, &options)
        }
        .map_err(|error| error.to_string())?;
        let report_json = CString::new(
            serde_json::to_string(&report(&field)).map_err(|error| error.to_string())?,
        )
        .map_err(|_| "PSF field report contains a NUL byte".to_string())?;
        Ok(SeizaPsfField { field, report_json })
    })
    .map_or(ptr::null_mut(), |field| Box::into_raw(Box::new(field)))
}

fn psf_field_options(options_json: *const c_char) -> Result<PsfFieldOptions, String> {
    if options_json.is_null() {
        return Ok(PsfFieldOptions::default());
    }
    let text = unsafe { CStr::from_ptr(options_json) }
        .to_str()
        .map_err(|_| "PSF field options are not valid UTF-8".to_string())?;
    if text.trim().is_empty() {
        return Ok(PsfFieldOptions::default());
    }
    let request: PsfFieldOptionsRequest = serde_json::from_str(text)
        .map_err(|error| format!("invalid PSF field options: {error}"))?;
    Ok(request.options())
}

#[unsafe(no_mangle)]
/// Returns the field's report as borrowed JSON: `width`, `height`,
/// `channels`, `stampSize`, `componentCount`, `starCount`, `isGlobal`,
/// `warnings` (messages), `channelFits` (each with `reducedChiSquared`,
/// `relativeResidual`, `degree`, `wingStars` and `wingOverlap`, the radii
/// where the core hands over to the wings, or null) and `stars` (each with `x`, `y`,
/// `flux`, `snr`, `fwhm` and `role`, `"core"` or `"wing"`). The string
/// remains valid until the field is freed.
///
/// # Safety
/// `field` must be null or a live pointer from [`seiza_psf_field_measure`].
pub unsafe extern "C" fn seiza_psf_field_report_json(field: *const SeizaPsfField) -> *const c_char {
    unsafe { field.as_ref() }.map_or(ptr::null(), |field| field.report_json.as_ptr())
}

#[unsafe(no_mangle)]
/// The side of the field's stamps in pixels, odd; 0 for a null field.
///
/// # Safety
/// `field` must be null or a live pointer from [`seiza_psf_field_measure`].
pub unsafe extern "C" fn seiza_psf_field_stamp_size(field: *const SeizaPsfField) -> usize {
    unsafe { field.as_ref() }.map_or(0, |field| field.field.stamp_size())
}

#[unsafe(no_mangle)]
/// The field's channels, one or three; 0 for a null field.
///
/// # Safety
/// `field` must be null or a live pointer from [`seiza_psf_field_measure`].
pub unsafe extern "C" fn seiza_psf_field_channels(field: *const SeizaPsfField) -> usize {
    unsafe { field.as_ref() }.map_or(0, |field| field.field.channels())
}

#[unsafe(no_mangle)]
/// Writes the local PSF at pixel (`x`, `y`) into `output`: `stamp_size *
/// stamp_size * channels` floats, row by row with channels interleaved,
/// each channel summing to one and centred on the middle pixel. Points
/// outside the image take the PSF at the nearest edge.
///
/// # Safety
/// `field` must be a live pointer from [`seiza_psf_field_measure`].
/// `output` must point to `output_length` writable floats. When non-null,
/// `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_psf_field_stamp_at(
    field: *const SeizaPsfField,
    x: f64,
    y: f64,
    output: *mut f32,
    output_length: usize,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let field = unsafe { required_psf_field(field)? };
        let stamp = field.field.stamp_at(x, y);
        if output_length != stamp.data.len() {
            return Err(format!(
                "PSF stamp output has {output_length} floats; expected {}",
                stamp.data.len()
            ));
        }
        let output = unsafe { required_f32_slice_mut(output, output_length, "PSF stamp output")? };
        output.copy_from_slice(&stamp.data);
        Ok(())
    })
    .is_some()
}

#[unsafe(no_mangle)]
/// Fits the shape of one channel's local PSF at pixel (`x`, `y`) and writes
/// it to `shape_out`. Returns false with an error when `channel` is out of
/// range or the profile cannot be fitted.
///
/// # Safety
/// `field` must be a live pointer from [`seiza_psf_field_measure`].
/// `shape_out` must point to writable storage for one shape. When non-null,
/// `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_psf_field_shape_at(
    field: *const SeizaPsfField,
    channel: usize,
    x: f64,
    y: f64,
    shape_out: *mut SeizaPsfShape,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let field = unsafe { required_psf_field(field)? };
        if shape_out.is_null() {
            return Err("PSF shape output is required".into());
        }
        if channel >= field.field.channels() {
            return Err(format!(
                "channel {channel} is out of range for a {}-channel PSF field",
                field.field.channels()
            ));
        }
        let shape = field
            .field
            .shape_at(channel, x, y)
            .ok_or_else(|| format!("the PSF at ({x}, {y}) could not be fitted"))?;
        unsafe { *shape_out = shape.into() };
        Ok(())
    })
    .is_some()
}

#[unsafe(no_mangle)]
/// Releases a PSF field. Null is ignored.
///
/// # Safety
/// `field` must be null or a pointer from [`seiza_psf_field_measure`] that
/// has not been freed.
pub unsafe extern "C" fn seiza_psf_field_free(field: *mut SeizaPsfField) {
    if !field.is_null() {
        drop(unsafe { Box::from_raw(field) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stars on a grid, Gaussian with a FWHM of 3 px, on a flat sky.
    fn star_field(width: usize, height: usize) -> Vec<f32> {
        let sigma = 3.0 / 2.354_82;
        let mut data = vec![100.0_f32; width * height];
        let mut state = 0x2545_f491_u64;
        for cy in (24..height - 24).step_by(32) {
            for cx in (24..width - 24).step_by(32) {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let flux = 20_000.0 + (state % 50_000) as f64;
                let (sx, sy) = (cx as f64 + 0.3, cy as f64 - 0.2);
                for y in cy - 10..=cy + 10 {
                    for x in cx - 10..=cx + 10 {
                        let r2 = (x as f64 - sx).powi(2) + (y as f64 - sy).powi(2);
                        let value = flux / (2.0 * std::f64::consts::PI * sigma * sigma)
                            * (-0.5 * r2 / (sigma * sigma)).exp();
                        data[y * width + x] += value as f32;
                    }
                }
            }
        }
        for (index, value) in data.iter_mut().enumerate() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *value +=
                ((state >> 40) as f32 / (1 << 24) as f32 - 0.5) * 4.0 + (index % 3) as f32 * 0.0;
        }
        data
    }

    #[test]
    fn measures_samples_and_reports_a_field() {
        let (width, height) = (420, 360);
        let data = star_field(width, height);
        let mut error = ptr::null_mut();
        let field = unsafe {
            seiza_psf_field_measure(
                data.as_ptr(),
                data.len(),
                width,
                height,
                1,
                ptr::null(),
                0,
                c"{\"components\": 2}".as_ptr(),
                &mut error,
            )
        };
        assert!(error.is_null());
        assert!(!field.is_null());
        let size = unsafe { seiza_psf_field_stamp_size(field) };
        assert_eq!(size, 33);
        assert_eq!(unsafe { seiza_psf_field_channels(field) }, 1);
        let mut stamp = vec![0.0_f32; size * size];
        assert!(unsafe {
            seiza_psf_field_stamp_at(
                field,
                200.0,
                180.0,
                stamp.as_mut_ptr(),
                stamp.len(),
                &mut error,
            )
        });
        assert!((stamp.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        let mut shape = SeizaPsfShape::default();
        assert!(unsafe {
            seiza_psf_field_shape_at(field, 0, 200.0, 180.0, &mut shape, &mut error)
        });
        assert!((shape.fwhm - 3.0).abs() < 0.2, "{shape:?}");
        assert!(!unsafe {
            seiza_psf_field_shape_at(field, 1, 200.0, 180.0, &mut shape, &mut error)
        });
        assert!(!error.is_null());
        unsafe { super::super::seiza_string_free(error) };
        let report: serde_json::Value = serde_json::from_str(
            unsafe { CStr::from_ptr(seiza_psf_field_report_json(field)) }
                .to_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(report["stampSize"], 33);
        assert_eq!(report["isGlobal"], false);
        assert!(report["starCount"].as_u64().unwrap() >= 30);
        assert_eq!(report["stars"][0]["role"], "core");
        unsafe { seiza_psf_field_free(field) };
    }

    #[test]
    fn refuses_a_sparse_field_unless_asked_for_a_global_psf() {
        let (width, height) = (200, 160);
        let data = star_field(width, height);
        let mut error = ptr::null_mut();
        let field = unsafe {
            seiza_psf_field_measure(
                data.as_ptr(),
                data.len(),
                width,
                height,
                1,
                ptr::null(),
                0,
                ptr::null(),
                &mut error,
            )
        };
        assert!(field.is_null());
        let message = unsafe { CStr::from_ptr(error) }
            .to_str()
            .unwrap()
            .to_owned();
        assert!(message.contains("too few"), "{message}");
        unsafe { super::super::seiza_string_free(error) };
        let field = unsafe {
            seiza_psf_field_measure(
                data.as_ptr(),
                data.len(),
                width,
                height,
                1,
                ptr::null(),
                0,
                c"{\"sparse\": \"global\"}".as_ptr(),
                &mut error,
            )
        };
        assert!(!field.is_null(), "{:?}", unsafe {
            error.as_ref().map(|e| CStr::from_ptr(e))
        });
        let report = unsafe { CStr::from_ptr(seiza_psf_field_report_json(field)) }
            .to_str()
            .unwrap()
            .to_owned();
        assert!(report.contains("\"isGlobal\":true"), "{report}");
        unsafe { seiza_psf_field_free(field) };
    }
}
