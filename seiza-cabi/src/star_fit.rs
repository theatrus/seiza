//! Catalogue-guided star photometry, removal and rendering (`seiza-stars`),
//! as Python's `StarFit` runs it.
//!
//! [`seiza_star_fit_new`] fits the flux of each catalogue star at its image
//! position with the local PSF of a measured field, overlapping stars
//! jointly. The handle then removes the stars from an image
//! ([`seiza_star_fit_subtract`]), renders them through a round Gaussian
//! ([`seiza_star_fit_render`]), or both at once, the classical star
//! sharpener ([`seiza_star_fit_sharpen`]).

use super::{clear_error, ffi_result, required_f32_slice, required_f32_slice_mut};
use crate::psf::{SeizaPsfCatalogueStar, SeizaPsfField, required_psf_field};
use seiza_stars::{CatalogueStar, FittedStar, StarFit, StarFitOptions, StarPositions};
use serde::{Deserialize, Serialize};
use std::ffi::{CStr, CString, c_char};
use std::ptr;

/// An opaque fit of catalogue stars. Release it with
/// [`seiza_star_fit_free`]. It is read-only once made and holds no pointer
/// to its inputs; its stars string is borrowed and remains valid until it is
/// freed.
pub struct SeizaStarFit {
    fit: StarFit,
    stars_json: CString,
}

/// `StarFitOptions` as JSON; absent fields take the defaults.
#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
struct StarFitOptionsRequest {
    fit_radius: Option<f64>,
    saturation: Option<f32>,
    positions: Option<StarPositions>,
    max_shift: Option<f64>,
}

impl StarFitOptionsRequest {
    fn options(self) -> StarFitOptions {
        let defaults = StarFitOptions::default();
        StarFitOptions {
            fit_radius: self.fit_radius,
            saturation: self.saturation,
            positions: self.positions.unwrap_or(defaults.positions),
            max_shift: self.max_shift.unwrap_or(defaults.max_shift),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StarFitReportResponse {
    width: usize,
    height: usize,
    channels: usize,
    stars: Vec<FittedStarResponse>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FittedStarResponse {
    x: f64,
    y: f64,
    magnitude: f64,
    flux: Vec<f64>,
    flux_error: Vec<f64>,
    background: Vec<f64>,
    reduced_chi_squared: f64,
    saturated: bool,
}

impl From<&FittedStar> for FittedStarResponse {
    fn from(star: &FittedStar) -> Self {
        Self {
            x: star.x,
            y: star.y,
            magnitude: star.magnitude,
            flux: star.flux.clone(),
            flux_error: star.flux_error.clone(),
            background: star.background.clone(),
            reduced_chi_squared: star.reduced_chi_squared,
            saturated: star.saturated,
        }
    }
}

unsafe fn required_fit<'a>(fit: *const SeizaStarFit) -> Result<&'a SeizaStarFit, String> {
    unsafe { fit.as_ref() }.ok_or_else(|| "star fit is required".into())
}

fn copy_out(values: &[f32], output: *mut f32, output_length: usize) -> Result<(), String> {
    if output_length != values.len() {
        return Err(format!(
            "star fit output has {output_length} floats; expected {}",
            values.len()
        ));
    }
    let output = unsafe { required_f32_slice_mut(output, output_length, "star fit output")? };
    output.copy_from_slice(values);
    Ok(())
}

#[unsafe(no_mangle)]
/// Fits the flux of each of `star_count` catalogue stars, at their image
/// positions, in interleaved linear `float` samples of the PSF field's size
/// and channels, with the field's local PSF: stars whose fit windows overlap
/// jointly, each group with its own background plane. `noise`, when not
/// null, holds the per-sample noise as a standard deviation in the image's
/// layout (`noise_length` must then equal `data_length`); a sample whose
/// noise is not finite and positive is left out. Pass null or empty
/// `options_json` for the defaults, or a JSON object with any of
/// `fitRadius` (pixels; three FWHM by default), `saturation` (samples at or
/// above it are left out), `positions` (`"fixed"`, the default, or
/// `"refined"`) and `maxShift` (how far a refined star may move, 1 px).
///
/// # Safety
/// `field` must be a live pointer from `seiza_psf_field_measure`. `data`
/// must point to `data_length` readable floats, `stars` to `star_count`
/// readable stars, and a non-null `noise` to `noise_length` readable floats.
/// A non-null `options_json` must be NUL-terminated. When non-null,
/// `error_out` must point to writable storage for one pointer.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn seiza_star_fit_new(
    field: *const SeizaPsfField,
    data: *const f32,
    data_length: usize,
    stars: *const SeizaPsfCatalogueStar,
    star_count: usize,
    noise: *const f32,
    noise_length: usize,
    options_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut SeizaStarFit {
    clear_error(error_out);
    ffi_result(error_out, || {
        let field = unsafe { required_psf_field(field)? };
        let data = unsafe { required_f32_slice(data, data_length, "star fit image")? };
        let stars: Vec<CatalogueStar> = if star_count == 0 {
            Vec::new()
        } else {
            if stars.is_null() {
                return Err("star fit stars are required".into());
            }
            unsafe { std::slice::from_raw_parts(stars, star_count) }
                .iter()
                .map(|star| CatalogueStar {
                    x: star.x,
                    y: star.y,
                    magnitude: star.magnitude,
                })
                .collect()
        };
        let noise = if noise.is_null() {
            None
        } else {
            Some(unsafe { required_f32_slice(noise, noise_length, "star fit noise map")? })
        };
        let options = if options_json.is_null() {
            StarFitOptions::default()
        } else {
            let text = unsafe { CStr::from_ptr(options_json) }
                .to_str()
                .map_err(|_| "star fit options are not valid UTF-8".to_string())?;
            if text.trim().is_empty() {
                StarFitOptions::default()
            } else {
                serde_json::from_str::<StarFitOptionsRequest>(text)
                    .map_err(|error| format!("invalid star fit options: {error}"))?
                    .options()
            }
        };
        let fit = StarFit::fit(data, &field.field, &stars, noise, &options)
            .map_err(|error| error.to_string())?;
        let report = StarFitReportResponse {
            width: fit.width(),
            height: fit.height(),
            channels: fit.channels(),
            stars: fit.stars().iter().map(Into::into).collect(),
        };
        let stars_json = CString::new(serde_json::to_string(&report).map_err(|e| e.to_string())?)
            .map_err(|_| "star fit report contains a NUL byte".to_string())?;
        Ok(SeizaStarFit { fit, stars_json })
    })
    .map_or(ptr::null_mut(), |fit| Box::into_raw(Box::new(fit)))
}

#[unsafe(no_mangle)]
/// Returns the fit as borrowed JSON: `width`, `height`, `channels` and
/// `stars`, in the order given, each with `x` and `y` (where it was
/// fitted), `magnitude`, and per channel `flux`, `fluxError` and
/// `background` (null where nothing constrained it), then
/// `reducedChiSquared` and `saturated`. The string remains valid until the
/// fit is freed.
///
/// # Safety
/// `fit` must be null or a live pointer from [`seiza_star_fit_new`].
pub unsafe extern "C" fn seiza_star_fit_stars_json(fit: *const SeizaStarFit) -> *const c_char {
    unsafe { fit.as_ref() }.map_or(ptr::null(), |fit| fit.stars_json.as_ptr())
}

#[unsafe(no_mangle)]
/// Writes the fitted stars through the measured PSF, on black, into
/// `output`: the star layer [`seiza_star_fit_subtract`] removes.
///
/// # Safety
/// `fit` must be a live pointer from [`seiza_star_fit_new`]. `output` must
/// point to `output_length` writable floats. When non-null, `error_out` must
/// point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_star_fit_model(
    fit: *const SeizaStarFit,
    output: *mut f32,
    output_length: usize,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let fit = unsafe { required_fit(fit)? };
        copy_out(&fit.fit.model(), output, output_length)
    })
    .is_some()
}

#[unsafe(no_mangle)]
/// Writes `data` with the fitted stars removed into `output`.
///
/// # Safety
/// `fit` must be a live pointer from [`seiza_star_fit_new`]. `data` must
/// point to `data_length` readable floats and `output` to `output_length`
/// writable floats. When non-null, `error_out` must point to writable
/// storage for one pointer.
pub unsafe extern "C" fn seiza_star_fit_subtract(
    fit: *const SeizaStarFit,
    data: *const f32,
    data_length: usize,
    output: *mut f32,
    output_length: usize,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let fit = unsafe { required_fit(fit)? };
        let data = unsafe { required_f32_slice(data, data_length, "star fit image")? };
        let removed = fit.fit.subtract(data).map_err(|error| error.to_string())?;
        copy_out(&removed, output, output_length)
    })
    .is_some()
}

#[unsafe(no_mangle)]
/// Writes the fitted stars rendered through a pixel-integrated round
/// Gaussian of `target_fwhm` pixels, on black, into `output`.
///
/// # Safety
/// `fit` must be a live pointer from [`seiza_star_fit_new`]. `output` must
/// point to `output_length` writable floats. When non-null, `error_out` must
/// point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_star_fit_render(
    fit: *const SeizaStarFit,
    target_fwhm: f64,
    output: *mut f32,
    output_length: usize,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let fit = unsafe { required_fit(fit)? };
        let rendered = fit
            .fit
            .render(target_fwhm)
            .map_err(|error| error.to_string())?;
        copy_out(&rendered, output, output_length)
    })
    .is_some()
}

#[unsafe(no_mangle)]
/// The classical star sharpener: writes `data` with the fitted stars
/// removed and rendered again through a round Gaussian of `target_fwhm`
/// pixels into `output`.
///
/// # Safety
/// `fit` must be a live pointer from [`seiza_star_fit_new`]. `data` must
/// point to `data_length` readable floats and `output` to `output_length`
/// writable floats. When non-null, `error_out` must point to writable
/// storage for one pointer.
pub unsafe extern "C" fn seiza_star_fit_sharpen(
    fit: *const SeizaStarFit,
    data: *const f32,
    data_length: usize,
    target_fwhm: f64,
    output: *mut f32,
    output_length: usize,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let fit = unsafe { required_fit(fit)? };
        let data = unsafe { required_f32_slice(data, data_length, "star fit image")? };
        let sharpened = fit
            .fit
            .sharpen(data, target_fwhm)
            .map_err(|error| error.to_string())?;
        copy_out(&sharpened, output, output_length)
    })
    .is_some()
}

#[unsafe(no_mangle)]
/// Releases a star fit. Null is ignored.
///
/// # Safety
/// `fit` must be null or a pointer from [`seiza_star_fit_new`] that has not
/// been freed.
pub unsafe extern "C" fn seiza_star_fit_free(fit: *mut SeizaStarFit) {
    if !fit.is_null() {
        drop(unsafe { Box::from_raw(fit) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::psf::{seiza_psf_field_free, seiza_psf_field_measure};

    /// Gaussian stars of FWHM 3 on a grid on a flat sky, and their list.
    fn star_field(width: usize, height: usize) -> (Vec<f32>, Vec<SeizaPsfCatalogueStar>) {
        let sigma = 3.0 / 2.354_82;
        let mut data = vec![100.0_f32; width * height];
        let mut stars = Vec::new();
        let mut state = 0x51_7cc1_b727_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f64 / (1u64 << 24) as f64
        };
        for cy in (24..height - 24).step_by(30) {
            for cx in (24..width - 24).step_by(30) {
                let (sx, sy, flux) = (
                    cx as f64 + next() - 0.5,
                    cy as f64 + next() - 0.5,
                    2.0e4 + 4.0e4 * next(),
                );
                stars.push(SeizaPsfCatalogueStar {
                    x: sx,
                    y: sy,
                    magnitude: -2.5 * flux.log10(),
                });
                for y in cy - 12..=cy + 12 {
                    for x in cx - 12..=cx + 12 {
                        let r2 = (x as f64 - sx).powi(2) + (y as f64 - sy).powi(2);
                        data[y * width + x] += (flux / (2.0 * std::f64::consts::PI * sigma * sigma)
                            * (-0.5 * r2 / (sigma * sigma)).exp())
                            as f32;
                    }
                }
            }
        }
        for value in &mut data {
            *value += ((next() - 0.5) * 4.0) as f32;
        }
        (data, stars)
    }

    #[test]
    fn fits_removes_and_renders_catalogue_stars() {
        let (width, height) = (300, 270);
        let (data, stars) = star_field(width, height);
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
        assert!(!field.is_null());
        let fit = unsafe {
            seiza_star_fit_new(
                field,
                data.as_ptr(),
                data.len(),
                stars.as_ptr(),
                stars.len(),
                ptr::null(),
                0,
                c"{\"positions\": \"refined\"}".as_ptr(),
                &mut error,
            )
        };
        assert!(!fit.is_null(), "{:?}", unsafe {
            error.as_ref().map(|e| CStr::from_ptr(e))
        });
        unsafe { seiza_psf_field_free(field) };
        let report: serde_json::Value = serde_json::from_str(
            unsafe { CStr::from_ptr(seiza_star_fit_stars_json(fit)) }
                .to_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(report["stars"].as_array().unwrap().len(), stars.len());
        let flux = report["stars"][0]["flux"][0].as_f64().unwrap();
        let truth = 10_f64.powf(-0.4 * stars[0].magnitude);
        assert!(
            (flux - truth).abs() < 0.03 * truth,
            "{flux} against {truth}"
        );

        let mut model = vec![0.0_f32; data.len()];
        let mut removed = vec![0.0_f32; data.len()];
        let mut rendered = vec![0.0_f32; data.len()];
        let mut sharpened = vec![0.0_f32; data.len()];
        assert!(unsafe { seiza_star_fit_model(fit, model.as_mut_ptr(), model.len(), &mut error) });
        assert!(unsafe {
            seiza_star_fit_subtract(
                fit,
                data.as_ptr(),
                data.len(),
                removed.as_mut_ptr(),
                removed.len(),
                &mut error,
            )
        });
        assert!(unsafe {
            seiza_star_fit_render(fit, 2.0, rendered.as_mut_ptr(), rendered.len(), &mut error)
        });
        assert!(unsafe {
            seiza_star_fit_sharpen(
                fit,
                data.as_ptr(),
                data.len(),
                2.0,
                sharpened.as_mut_ptr(),
                sharpened.len(),
                &mut error,
            )
        });
        for index in [0, 1000, 24 * width + 24] {
            assert!((removed[index] + model[index] - data[index]).abs() < 1e-2);
            assert!((sharpened[index] - removed[index] - rendered[index]).abs() < 1e-2);
        }
        assert!(!unsafe {
            seiza_star_fit_render(fit, 0.0, rendered.as_mut_ptr(), rendered.len(), &mut error)
        });
        unsafe { super::super::seiza_string_free(error) };
        unsafe { seiza_star_fit_free(fit) };
    }
}
