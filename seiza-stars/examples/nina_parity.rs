//! Measure frames the way PSF Guard's quality scan does, for comparison with
//! what N.I.N.A. recorded.
//!
//! Reads manifest lines on standard input, `rig|id|stars|hfr|fwhm|path`, with
//! N.I.N.A.'s recorded star count, HFR and FWHM, and writes one CSV line per
//! frame with those values beside Seiza's. The detector is the first
//! argument:
//!
//! - `nina31`, `nina` (3.2, the default) or `nina33`: N.I.N.A.'s built-in
//!   detector with that version's measurement, on the frame autostretched as
//!   N.I.N.A. stretches it before detecting stars.
//! - `hocus`: the HocusFocus port, with parameters for the frame's optics as
//!   PSF Guard picks them.
//!
//! N.I.N.A.'s profile settings come from the environment:
//! `NINA_SENSITIVITY` (`normal`, `high` (the default, as in N.I.N.A.),
//! `highest`), `NINA_NOISE_REDUCTION`
//! (`none`, `normal`, `high`, `highest`, `median`), `NINA_STRETCH` (the
//! autostretch factor, 0.2) and `NINA_CLIP` (black clipping, −2.8). High
//! sensitivity takes its image scale from the XPIXSZ and FOCALLEN headers.
//! For `hocus`, `HF_VERSION` picks the HocusFocus release (`3`, `4`, or
//! `auto` by the frame's date), `HF_CLASS` (`wide`, `standard`, `long`)
//! overrides the rig class, `HF_WALKER` takes candidates with the older
//! walk, `HF_INNER_CROP` and `HF_OUTER_CROP` set N.I.N.A.'s autofocus crop
//! ratios, `HF_NC` sets the noise clipping multiplier, `HF_BIN` detection
//! binning, and `HF_NOISE_RADIUS`, `HF_SENSITIVITY`, `HF_MIN_HFR`,
//! `HF_LAYERS` and `HF_STRUCTURE=atrous` override the preset.
//! `NINA_PARITY_DEBUG` prints each frame's HFR spread and the detector's
//! diagnostics on standard error.
//!
//! ```sh
//! cargo run --release -p seiza-stars \
//!     --example nina_parity -- nina31 < manifest.txt
//! ```

use std::io::BufRead;
use std::path::Path;

use rayon::prelude::*;
use seiza_fits::{FitsImage, StretchParams, statistics_u16, stretch_u16_to_u16};
use seiza_stars::hocus_focus_star_detection::{
    HocusFocusParams, StructureRemovalMethod, detect_stars_hocus_focus,
};
use seiza_stars::nina_star_detection::{
    MeasurementVersion, NoiseReduction, StarDetectionParams, StarSensitivity,
    detect_stars_with_original,
};

/// One frame's measurement: star count, mean HFR, HFR spread, mean FWHM and
/// mean eccentricity.
struct Measured {
    stars: usize,
    hfr: f64,
    hfr_sd: f64,
    fwhm: f64,
    eccentricity: f64,
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// Arcseconds per pixel from the XPIXSZ and FOCALLEN headers, as N.I.N.A.
/// computes it for High sensitivity.
fn image_scale(fits: &FitsImage) -> Option<f64> {
    let pixel = fits.header_f64("XPIXSZ")?;
    let focal = fits.header_f64("FOCALLEN")?;
    (focal > 0.0).then(|| pixel / focal * 206.265)
}

fn measure(detector: &str, path: &Path) -> Result<Measured, String> {
    let fits = FitsImage::open(path).map_err(|error| error.to_string())?;
    let (width, height) = (fits.width, fits.height);
    let data = fits.to_u16().into_owned();
    if detector == "hocus" {
        // Parameters for the frame's optics, as PSF Guard picks them, for
        // the HocusFocus release asked for: `HF_VERSION=4`, or `auto` for 4
        // from HocusFocus 4's release on 2026-07-02.
        let (params, class) = HocusFocusParams::for_frame_headers(
            fits.header_f64("FOCALLEN"),
            fits.header_f64("XPIXSZ"),
        );
        let v4 = match std::env::var("HF_VERSION").as_deref() {
            Ok("4") => true,
            Ok("auto") => fits
                .header_str("DATE-OBS")
                .is_some_and(|date| date >= "2026-07-02"),
            _ => false,
        };
        let class = match std::env::var("HF_CLASS").as_deref() {
            Ok("wide") => seiza_stars::TelescopeClass::WideField,
            Ok("standard") => seiza_stars::TelescopeClass::Standard,
            Ok("long") => seiza_stars::TelescopeClass::LongFocalLength,
            _ => class,
        };
        let params = if v4 {
            HocusFocusParams::v4_for_telescope_class(class)
        } else {
            params
        };
        let mut params = params;
        if let Ok(v) = std::env::var("HF_NC") {
            params.noise_clipping_multiplier = v.parse().unwrap();
        }
        if let Ok(v) = std::env::var("HF_BIN") {
            params.detection_binning = v.parse().unwrap();
        }
        if std::env::var("HF_MEASURE_BLURRED").is_ok() {
            params.measurement_noise_reduction = true;
        }
        if let Ok(v) = std::env::var("HF_SENSITIVITY_V4") {
            params.sensitivity = v.parse().unwrap();
        }
        if let Ok(v) = std::env::var("HF_INNER_CROP") {
            params.autofocus_inner_crop_ratio = v.parse().unwrap();
        }
        if let Ok(v) = std::env::var("HF_OUTER_CROP") {
            params.autofocus_outer_crop_ratio = v.parse().unwrap();
        }
        if std::env::var("HF_WALKER").is_ok() {
            params.connected_components = false;
        }
        if let Ok(v) = std::env::var("HF_NOISE_RADIUS") {
            params.noise_reduction_radius = v.parse().unwrap();
        }
        if let Ok(v) = std::env::var("HF_SENSITIVITY") {
            params.sensitivity = v.parse().unwrap();
        }
        if let Ok(v) = std::env::var("HF_MIN_HFR") {
            params.min_hfr = v.parse().unwrap();
        }
        if let Ok(v) = std::env::var("HF_LAYERS") {
            params.structure_layers = v.parse().unwrap();
        }
        if std::env::var("HF_STRUCTURE").as_deref() == Ok("atrous") {
            params.structure_removal = StructureRemovalMethod::Atrous;
        }
        let found = detect_stars_hocus_focus(&data, width, height, &params);
        let hfrs = found.stars.iter().map(|s| s.hfr).collect::<Vec<_>>();
        let mean = found.average_hfr;
        let sd = (hfrs.iter().map(|h| (h - mean).powi(2)).sum::<f64>()
            / hfrs.len().saturating_sub(1).max(1) as f64)
            .sqrt();
        return Ok(Measured {
            stars: found.stars.len(),
            hfr: mean,
            hfr_sd: sd,
            fwhm: found.average_fwhm,
            eccentricity: f64::NAN,
        });
    }
    let mut stats = statistics_u16(&data);
    stats.mad = stats.mad.max(1.0);
    let stretch = StretchParams {
        target_median: env_f64("NINA_STRETCH", 0.2),
        shadows_clip: env_f64("NINA_CLIP", -2.8),
    };
    let stretched = stretch_u16_to_u16(&data, &stats, &stretch);
    let params = StarDetectionParams {
        sensitivity: match std::env::var("NINA_SENSITIVITY").as_deref() {
            Ok("normal") => StarSensitivity::Normal,
            Ok("highest") => StarSensitivity::Highest,
            _ => StarSensitivity::High,
        },
        noise_reduction: match std::env::var("NINA_NOISE_REDUCTION").as_deref() {
            Ok("normal") => NoiseReduction::Normal,
            Ok("high") => NoiseReduction::High,
            Ok("highest") => NoiseReduction::Highest,
            Ok("median") => NoiseReduction::Median,
            _ => NoiseReduction::None,
        },
        image_scale_arcsec: image_scale(&fits),
        measurement: match detector {
            "nina31" => MeasurementVersion::Nina31,
            "nina33" => MeasurementVersion::Nina33,
            _ => MeasurementVersion::Nina32,
        },
        ..StarDetectionParams::default()
    };
    let result = detect_stars_with_original(&stretched, &data, width, height, &params);
    if std::env::var_os("NINA_PARITY_DEBUG").is_some() {
        let mut hfrs = result.star_list.iter().map(|s| s.hfr).collect::<Vec<_>>();
        hfrs.sort_by(f64::total_cmp);
        let q = |p: f64| {
            hfrs.get(((hfrs.len() as f64 - 1.0) * p) as usize)
                .copied()
                .unwrap_or(f64::NAN)
        };
        eprintln!(
            "{}: stars {} HFR mean {:.3} p10 {:.3} p50 {:.3} p90 {:.3}; median {} MAD {}",
            path.display(),
            hfrs.len(),
            result.average_hfr,
            q(0.1),
            q(0.5),
            q(0.9),
            stats.median,
            stats.mad
        );
    }
    Ok(Measured {
        stars: result.star_list.len(),
        hfr: result.average_hfr,
        hfr_sd: result.hfr_std_dev,
        fwhm: result.average_fwhm.unwrap_or(f64::NAN),
        eccentricity: result.average_eccentricity.unwrap_or(f64::NAN),
    })
}

fn main() {
    seiza_stars::debug::init_debug(std::env::var_os("NINA_PARITY_DEBUG").is_some());
    let detector = std::env::args().nth(1).unwrap_or_else(|| "nina".into());
    let lines = std::io::stdin()
        .lock()
        .lines()
        .map_while(Result::ok)
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .collect::<Vec<_>>();
    println!("rig,id,nina_stars,nina_hfr,nina_fwhm,stars,hfr,hfr_sd,fwhm,eccentricity,path");
    let rows = lines
        .par_iter()
        .with_max_len(1)
        .map(|line| {
            let fields = line.splitn(6, '|').collect::<Vec<_>>();
            let [rig, id, stars, hfr, fwhm, path] = fields[..] else {
                return format!("# bad line: {line}");
            };
            match measure(&detector, Path::new(path)) {
                Ok(m) => format!(
                    "{rig},{id},{stars},{hfr},{fwhm},{},{:.4},{:.4},{:.4},{:.4},\"{path}\"",
                    m.stars, m.hfr, m.hfr_sd, m.fwhm, m.eccentricity
                ),
                Err(error) => format!("# {id}: {error}"),
            }
        })
        .collect::<Vec<_>>();
    for row in rows {
        println!("{row}");
    }
}
