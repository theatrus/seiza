//! Deconvolution towards a round target PSF with a PSF that varies across
//! the image, from `seiza-deconvolution`, as Python's `RemainingKernel`,
//! `deconvolve_field` and `reblur_consistency` run it.
//!
//! [`seiza_remaining_kernel_new`] takes a measured PSF field and a target,
//! and holds the remaining kernel `R`, with `P = T ∗ R`, on a grid of nodes.
//! [`seiza_deconvolve_field_in_place`] runs damped Richardson–Lucy with it,
//! and [`seiza_reblur_consistency_json`] checks an estimate at the target
//! against its input.

use super::{clear_error, ffi_result, owned_json, required_f32_slice, required_f32_slice_mut};
use crate::psf::{SeizaPsfField, required_psf_field};
use seiza_deconvolution::{
    ChannelConsistency, DEFAULT_NODE_SPACING, FieldDeconvolutionConfig, RemainingKernel, TargetPsf,
    deconvolve_field, deconvolve_field_masked, reblur_consistency,
};
use serde::{Deserialize, Serialize};
use std::ffi::{CStr, CString, c_char};
use std::ptr;

/// An opaque remaining kernel between a PSF field and a round target.
/// Release it with [`seiza_remaining_kernel_free`]. It is read-only once
/// made, and holds no pointer to the field it was made from; its report
/// string is borrowed and remains valid until it is freed.
pub struct SeizaRemainingKernel {
    kernel: RemainingKernel,
    report_json: CString,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
struct RemainingKernelRequest {
    target: Option<TargetPsf>,
    node_spacing: Option<usize>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RemainingKernelReportResponse {
    width: usize,
    height: usize,
    channels: usize,
    kernel_size: usize,
    node_spacing: usize,
    node_count: usize,
    channel_ranges: Vec<KernelChannelResponse>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct KernelChannelResponse {
    psf_fwhm_range: [f32; 2],
    target_fwhm_range: [f32; 2],
}

/// `FieldDeconvolutionConfig` as JSON, with `masked` for registered images
/// whose missing samples are non-finite. Absent fields take the
/// conservative defaults.
#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
struct FieldDeconvolutionRequest {
    iterations: Option<usize>,
    amount: Option<f32>,
    noise_fraction: Option<f32>,
    max_correction: Option<f32>,
    masked: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ChannelConsistencyResponse {
    reduced_chi_squared: f64,
    samples: usize,
    band_power: Vec<f64>,
    spectral_flatness: Option<f64>,
}

impl From<&ChannelConsistency> for ChannelConsistencyResponse {
    fn from(channel: &ChannelConsistency) -> Self {
        Self {
            reduced_chi_squared: channel.reduced_chi_squared,
            samples: channel.samples,
            band_power: channel.band_power.clone(),
            spectral_flatness: channel.spectral_flatness,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ReblurConsistencyResponse {
    channels: Vec<ChannelConsistencyResponse>,
}

fn optional_json<T: serde::de::DeserializeOwned + Default>(
    text: *const c_char,
    name: &str,
) -> Result<T, String> {
    if text.is_null() {
        return Ok(T::default());
    }
    let text = unsafe { CStr::from_ptr(text) }
        .to_str()
        .map_err(|_| format!("{name} options are not valid UTF-8"))?;
    if text.trim().is_empty() {
        return Ok(T::default());
    }
    serde_json::from_str(text).map_err(|error| format!("invalid {name} options: {error}"))
}

unsafe fn required_kernel<'a>(
    kernel: *const SeizaRemainingKernel,
) -> Result<&'a SeizaRemainingKernel, String> {
    unsafe { kernel.as_ref() }.ok_or_else(|| "remaining kernel is required".into())
}

fn report(kernel: &RemainingKernel) -> RemainingKernelReportResponse {
    RemainingKernelReportResponse {
        width: kernel.width(),
        height: kernel.height(),
        channels: kernel.channels(),
        kernel_size: kernel.kernel_size(),
        node_spacing: kernel.node_spacing(),
        node_count: kernel.node_count(),
        channel_ranges: (0..kernel.channels())
            .map(|channel| {
                let psf = kernel.psf_fwhm_range(channel).unwrap_or_default();
                let target = kernel.target_fwhm_range(channel).unwrap_or_default();
                KernelChannelResponse {
                    psf_fwhm_range: [psf.0, psf.1],
                    target_fwhm_range: [target.0, target.1],
                }
            })
            .collect(),
    }
}

#[unsafe(no_mangle)]
/// Makes the remaining kernel between a measured PSF field and a round
/// Gaussian target. Pass null or empty `options_json` for the field's best
/// round PSF as the target and nodes at most 256 px apart, or a JSON object with
/// `target` (`{"fwhm": 2.0}` for one FWHM in pixels, at least 1.5;
/// `{"ratio": 0.7}` for a fraction of the local FWHM; or `"bestRound"`) and
/// `nodeSpacing` (the most pixels between the nodes the kernel is sampled at, at
/// least 16). Returns null with an error for a bad target or spacing.
///
/// # Safety
/// `field` must be a live pointer from `seiza_psf_field_measure`. A
/// non-null `options_json` must be NUL-terminated. When non-null,
/// `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_remaining_kernel_new(
    field: *const SeizaPsfField,
    options_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut SeizaRemainingKernel {
    clear_error(error_out);
    ffi_result(error_out, || {
        let field = unsafe { required_psf_field(field)? };
        let request: RemainingKernelRequest = optional_json(options_json, "remaining kernel")?;
        let kernel = RemainingKernel::new(
            &field.field,
            request.target.unwrap_or(TargetPsf::BestRound),
            request.node_spacing.unwrap_or(DEFAULT_NODE_SPACING),
        )
        .map_err(|error| error.to_string())?;
        let report_json = CString::new(
            serde_json::to_string(&report(&kernel)).map_err(|error| error.to_string())?,
        )
        .map_err(|_| "remaining kernel report contains a NUL byte".to_string())?;
        Ok(SeizaRemainingKernel {
            kernel,
            report_json,
        })
    })
    .map_or(ptr::null_mut(), |kernel| Box::into_raw(Box::new(kernel)))
}

#[unsafe(no_mangle)]
/// Returns the kernel's report as borrowed JSON: `width`, `height`,
/// `channels`, `kernelSize`, `nodeSpacing`, `nodeCount` and, per channel in
/// `channelRanges`, the narrowest and widest local PSF FWHM
/// (`psfFwhmRange`) and target FWHM (`targetFwhmRange`) over the nodes. The
/// string remains valid until the kernel is freed.
///
/// # Safety
/// `kernel` must be null or a live pointer from
/// [`seiza_remaining_kernel_new`].
pub unsafe extern "C" fn seiza_remaining_kernel_report_json(
    kernel: *const SeizaRemainingKernel,
) -> *const c_char {
    unsafe { kernel.as_ref() }.map_or(ptr::null(), |kernel| kernel.report_json.as_ptr())
}

#[unsafe(no_mangle)]
/// Writes the kernel at pixel (`x`, `y`) in one channel, blended from the
/// nodes around it: `kernelSize²` floats summing to one, centred on the
/// middle pixel.
///
/// # Safety
/// `kernel` must be a live pointer from [`seiza_remaining_kernel_new`].
/// `output` must point to `output_length` writable floats. When non-null,
/// `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_remaining_kernel_kernel_at(
    kernel: *const SeizaRemainingKernel,
    channel: usize,
    x: f64,
    y: f64,
    output: *mut f32,
    output_length: usize,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let kernel = unsafe { required_kernel(kernel)? };
        let values = kernel
            .kernel
            .kernel_at(channel, x, y)
            .ok_or_else(|| format!("channel {channel} is out of range"))?;
        if output_length != values.len() {
            return Err(format!(
                "kernel output has {output_length} floats; expected {}",
                values.len()
            ));
        }
        let output = unsafe { required_f32_slice_mut(output, output_length, "kernel output")? };
        output.copy_from_slice(&values);
        Ok(())
    })
    .is_some()
}

#[unsafe(no_mangle)]
/// Blurs interleaved linear `float` samples by the remaining kernel, each
/// source pixel by its own, into `output`: what the image at the target
/// would look like with the measured PSF. Both buffers hold `width *
/// height * channels` samples of the field's size. Non-finite samples stay
/// non-finite and are left out of their neighbours' blur.
///
/// # Safety
/// `kernel` must be a live pointer from [`seiza_remaining_kernel_new`].
/// `data` must point to `data_length` readable floats and `output` to
/// `output_length` writable floats. When non-null, `error_out` must point
/// to writable storage for one pointer.
pub unsafe extern "C" fn seiza_remaining_kernel_blur(
    kernel: *const SeizaRemainingKernel,
    data: *const f32,
    data_length: usize,
    output: *mut f32,
    output_length: usize,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let kernel = unsafe { required_kernel(kernel)? };
        let data = unsafe { required_f32_slice(data, data_length, "kernel input")? };
        let blurred = kernel
            .kernel
            .blur(data)
            .map_err(|error| error.to_string())?;
        if output_length != blurred.len() {
            return Err(format!(
                "kernel output has {output_length} floats; expected {}",
                blurred.len()
            ));
        }
        let output = unsafe { required_f32_slice_mut(output, output_length, "kernel output")? };
        output.copy_from_slice(&blurred);
        Ok(())
    })
    .is_some()
}

#[unsafe(no_mangle)]
/// Releases a remaining kernel. Null is ignored.
///
/// # Safety
/// `kernel` must be null or a pointer from [`seiza_remaining_kernel_new`]
/// that has not been freed.
pub unsafe extern "C" fn seiza_remaining_kernel_free(kernel: *mut SeizaRemainingKernel) {
    if !kernel.is_null() {
        drop(unsafe { Box::from_raw(kernel) });
    }
}

#[unsafe(no_mangle)]
/// Deconvolves interleaved linear `float` samples in place towards the
/// kernel's target, with damped Richardson–Lucy and the remaining kernel
/// as its forward model. Pass null or empty `options_json` for the
/// conservative defaults, or a JSON object with any of `iterations` (4),
/// `amount` (0.35), `noiseFraction` (0.001), `maxCorrection` (2.0) and
/// `masked` (false: non-finite samples are an error; true: they are a mask
/// that stays). The image must be the field's size. The input is left
/// unchanged when the call fails.
///
/// # Safety
/// `kernel` must be a live pointer from [`seiza_remaining_kernel_new`].
/// `data` must point to `data_length` writable floats. A non-null
/// `options_json` must be NUL-terminated. When non-null, `error_out` must
/// point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_deconvolve_field_in_place(
    kernel: *const SeizaRemainingKernel,
    data: *mut f32,
    data_length: usize,
    options_json: *const c_char,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    ffi_result(error_out, || {
        let kernel = unsafe { required_kernel(kernel)? };
        let request: FieldDeconvolutionRequest =
            optional_json(options_json, "field deconvolution")?;
        let defaults = FieldDeconvolutionConfig::conservative();
        let config = FieldDeconvolutionConfig {
            iterations: request.iterations.unwrap_or(defaults.iterations),
            amount: request.amount.unwrap_or(defaults.amount),
            noise_fraction: request.noise_fraction.unwrap_or(defaults.noise_fraction),
            max_correction: request.max_correction.unwrap_or(defaults.max_correction),
        };
        let data = unsafe { required_f32_slice_mut(data, data_length, "deconvolution input")? };
        let restored = if request.masked {
            deconvolve_field_masked(data, &kernel.kernel, &config)
        } else {
            deconvolve_field(data, &kernel.kernel, &config)
        }
        .map_err(|error| error.to_string())?;
        data.copy_from_slice(&restored.data);
        Ok(())
    })
    .is_some()
}

#[unsafe(no_mangle)]
/// Checks an estimate at the kernel's target against the image it came
/// from: `output`, `input` and `noise` (the input's per-sample noise as a
/// standard deviation; a sample whose noise is not finite and positive is
/// left out) each hold `length` interleaved samples of the field's size.
/// Returns owned JSON, released with `seiza_string_free`: `channels`, each
/// with `reducedChiSquared` of `(R ∗ output − input) / σ`, the `samples` it
/// covers, `bandPower` (the residual's power in eight radial frequency
/// bands, lowest first, relative to their mean) and `spectralFlatness`
/// (their geometric over arithmetic mean, 1 for a flat spectrum, or null).
///
/// # Safety
/// `kernel` must be a live pointer from [`seiza_remaining_kernel_new`].
/// `output`, `input` and `noise` must each point to `length` readable
/// floats. When non-null, `error_out` must point to writable storage for
/// one pointer.
pub unsafe extern "C" fn seiza_reblur_consistency_json(
    kernel: *const SeizaRemainingKernel,
    output: *const f32,
    input: *const f32,
    noise: *const f32,
    length: usize,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let kernel = unsafe { required_kernel(kernel)? };
        let output = unsafe { required_f32_slice(output, length, "estimate")? };
        let input = unsafe { required_f32_slice(input, length, "input")? };
        let noise = unsafe { required_f32_slice(noise, length, "noise map")? };
        let consistency = reblur_consistency(output, input, noise, &kernel.kernel)
            .map_err(|error| error.to_string())?;
        owned_json(&ReblurConsistencyResponse {
            channels: consistency.channels.iter().map(Into::into).collect(),
        })
    })
    .unwrap_or(ptr::null_mut())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::psf::{seiza_psf_field_free, seiza_psf_field_measure};

    /// Gaussian stars of FWHM 3.4 on a grid, on a flat sky with noise.
    fn star_field(width: usize, height: usize) -> Vec<f32> {
        let sigma = 3.4 / 2.354_82;
        let mut data = vec![100.0_f32; width * height];
        let mut state = 0x9e37_79b9_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1 << 24) as f32
        };
        for cy in (24..height - 24).step_by(28) {
            for cx in (24..width - 24).step_by(28) {
                let flux = 2.0e4 + 5.0e4 * f64::from(next());
                for y in cy - 12..=cy + 12 {
                    for x in cx - 12..=cx + 12 {
                        let r2 = (x as f64 - cx as f64 - 0.2).powi(2)
                            + (y as f64 - cy as f64 + 0.3).powi(2);
                        data[y * width + x] += (flux / (2.0 * std::f64::consts::PI * sigma * sigma)
                            * (-0.5 * r2 / (sigma * sigma)).exp())
                            as f32;
                    }
                }
            }
        }
        for value in &mut data {
            *value += (next() - 0.5) * 4.0;
        }
        data
    }

    #[test]
    fn makes_applies_and_deconvolves_with_a_remaining_kernel() {
        let (width, height) = (300, 260);
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
        assert!(!field.is_null());
        let kernel = unsafe {
            seiza_remaining_kernel_new(
                field,
                c"{\"target\": {\"fwhm\": 2.0}, \"nodeSpacing\": 64}".as_ptr(),
                &mut error,
            )
        };
        assert!(!kernel.is_null());
        unsafe { seiza_psf_field_free(field) };
        let report: serde_json::Value = serde_json::from_str(
            unsafe { CStr::from_ptr(seiza_remaining_kernel_report_json(kernel)) }
                .to_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(report["nodeSpacing"], 64);
        assert_eq!(report["channelRanges"][0]["targetFwhmRange"][0], 2.0);
        let size = report["kernelSize"].as_u64().unwrap() as usize;
        let mut values = vec![0.0_f32; size * size];
        assert!(unsafe {
            seiza_remaining_kernel_kernel_at(
                kernel,
                0,
                150.0,
                130.0,
                values.as_mut_ptr(),
                values.len(),
                &mut error,
            )
        });
        assert!((values.iter().sum::<f32>() - 1.0).abs() < 1e-4);

        let mut blurred = vec![0.0_f32; data.len()];
        assert!(unsafe {
            seiza_remaining_kernel_blur(
                kernel,
                data.as_ptr(),
                data.len(),
                blurred.as_mut_ptr(),
                blurred.len(),
                &mut error,
            )
        });
        let mut restored = data.clone();
        assert!(unsafe {
            seiza_deconvolve_field_in_place(
                kernel,
                restored.as_mut_ptr(),
                restored.len(),
                c"{\"iterations\": 10, \"amount\": 1.0}".as_ptr(),
                &mut error,
            )
        });
        let peak = |image: &[f32]| image[24 * width + 24];
        assert!(peak(&restored) > peak(&data));

        let noise = vec![2.0_f32; data.len()];
        let json = unsafe {
            seiza_reblur_consistency_json(
                kernel,
                restored.as_ptr(),
                data.as_ptr(),
                noise.as_ptr(),
                data.len(),
                &mut error,
            )
        };
        assert!(!json.is_null());
        let consistency: serde_json::Value =
            serde_json::from_str(unsafe { CStr::from_ptr(json) }.to_str().unwrap()).unwrap();
        unsafe { super::super::seiza_string_free(json) };
        assert_eq!(
            consistency["channels"][0]["bandPower"]
                .as_array()
                .unwrap()
                .len(),
            8
        );
        assert!(
            consistency["channels"][0]["reducedChiSquared"]
                .as_f64()
                .unwrap()
                > 0.0
        );

        let bad = unsafe { seiza_remaining_kernel_new(ptr::null(), ptr::null(), &mut error) };
        assert!(bad.is_null());
        unsafe { super::super::seiza_string_free(error) };
        unsafe { seiza_remaining_kernel_free(kernel) };
    }
}
