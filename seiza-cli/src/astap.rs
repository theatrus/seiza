//! ASTAP-compatible solver mode, so N.I.N.A. (and anything else that
//! shells out to ASTAP) can use seiza by pointing the "ASTAP path" at
//! this binary. See docs/design/astap-mode.md for the full contract.
//!
//! Success is judged solely by the `<image-basename>.ini` written next
//! to the input file: `PLTSOLVD=T` plus CRVAL/CRPIX/CD keys, or
//! `PLTSOLVD=F` with an `ERROR=` line.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

#[derive(Debug, Default)]
struct AstapArgs {
    /// `-f`: input image
    file: Option<PathBuf>,
    /// `-fov`: field of view of the image HEIGHT, degrees (0 = unknown)
    fov_deg: f64,
    /// `-r`: search radius, degrees (>= 180 means blind)
    radius_deg: f64,
    /// `-ra`: hint right ascension, HOURS
    ra_hours: Option<f64>,
    /// `-spd`: hint as south polar distance (dec + 90), degrees
    spd_deg: Option<f64>,
    /// `-s`: maximum stars
    max_stars: Option<usize>,
    /// `-o`: output base path override (ASTAP extension some tools use)
    output: Option<PathBuf>,
    /// Seiza extension: detector numeric representation.
    detection_backend: seiza::DetectBackend,
    /// Seiza extension: retry policy after an Auto/u8 solve miss.
    detection_fallback: crate::DetectionFallback,
    /// Seiza extension: primary blind budget before an available f32 retry.
    detection_fallback_hypotheses: usize,
}

/// True when the binary itself is named like ASTAP. The Windows installer
/// puts an `astap.exe` copy next to `seiza.exe` because N.I.N.A. asks for a
/// file of that name, so a command line carrying no recognizable ASTAP flag
/// still belongs here rather than in clap's hands: the caller reads the
/// `.ini` and never sees a usage error.
pub fn invoked_as_astap(program: &str) -> bool {
    crate::common::binary_name_matches(program, "astap")
}

/// True when the raw command line looks like an ASTAP invocation
/// (lets a copy of the binary named `astap.exe` work without the
/// explicit subcommand).
pub fn looks_like_astap(args: &[String]) -> bool {
    matches!(
        args.first().map(String::as_str),
        Some("-f" | "-ra" | "-spd" | "-fov" | "-r" | "-z" | "-s" | "-o")
    )
}

/// Run ASTAP mode over raw (post program-name) arguments.
pub fn run(raw: &[String]) -> Result<()> {
    let args = parse_args(raw);
    let Some(image) = args.file.clone() else {
        anyhow::bail!("ASTAP mode requires -f <image>");
    };
    let ini = args
        .output
        .clone()
        .unwrap_or_else(|| image.clone())
        .with_extension("ini");

    match solve(&args, &image) {
        Ok(lines) => write_ini(&ini, &lines),
        Err(e) => {
            // Failures still write the result file: the caller ignores
            // the exit code and reads only the ini
            let message = format!("{e:#}").replace(['\r', '\n'], " ");
            write_ini(
                &ini,
                &["PLTSOLVD=F".to_string(), format!("ERROR={message}")],
            )?;
            Err(e)
        }
    }
}

fn parse_args(raw: &[String]) -> AstapArgs {
    let mut args = AstapArgs {
        detection_fallback_hypotheses: 64,
        ..Default::default()
    };
    let mut iter = raw.iter().peekable();
    fn value(iter: &mut std::iter::Peekable<std::slice::Iter<String>>) -> Option<String> {
        iter.next().cloned()
    }
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "-f" => args.file = value(&mut iter).map(PathBuf::from),
            "-o" => args.output = value(&mut iter).map(PathBuf::from),
            "-fov" => args.fov_deg = value(&mut iter).and_then(|v| v.parse().ok()).unwrap_or(0.0),
            "-r" => {
                args.radius_deg = value(&mut iter)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(180.0)
            }
            "-ra" => args.ra_hours = value(&mut iter).and_then(|v| v.parse().ok()),
            "-spd" => args.spd_deg = value(&mut iter).and_then(|v| v.parse().ok()),
            "-s" => args.max_stars = value(&mut iter).and_then(|v| v.parse().ok()),
            "--detection-backend" => {
                args.detection_backend = match value(&mut iter).as_deref() {
                    Some("u8") => seiza::DetectBackend::U8,
                    Some("f32") => seiza::DetectBackend::F32,
                    _ => seiza::DetectBackend::Auto,
                }
            }
            "--detection-fallback" => {
                args.detection_fallback = match value(&mut iter).as_deref() {
                    Some("none") => crate::DetectionFallback::None,
                    _ => crate::DetectionFallback::F32,
                }
            }
            "--detection-fallback-hypotheses" => {
                args.detection_fallback_hypotheses =
                    value(&mut iter).and_then(|v| v.parse().ok()).unwrap_or(64)
            }
            // -z (downsample), -log, -wcs, and anything future: accept
            // and ignore, consuming a value only for flags known to take
            // one
            "-z" | "-t" | "-m" | "-d" | "-D" | "-sqm" | "-focus1" => {
                let takes_value = iter.peek().is_some_and(|v| !v.starts_with('-'));
                if takes_value {
                    iter.next();
                }
            }
            _ => {}
        }
    }
    args
}

/// Scale range for the ASTAP-compatible blind fallback.
///
/// `-fov` is useful for the fast hinted attempt, but it is still only a hint:
/// N.I.N.A. derives it from the configured focal length, which can be stale or
/// describe a different optical train than a simulator/file-camera frame. Once
/// the hinted solve misses, retaining a hard scale bracket can make the
/// supposedly blind fallback unable to find an otherwise straightforward
/// field. Keep the parameter here so that policy is explicit at the call site;
/// a future solver may use it to prioritize hypotheses, but it must not exclude
/// scales from the practical blind range.
fn blind_scale_search_range(_fov_scale_hint: Option<f64>) -> (f64, f64) {
    let defaults = seiza::blind::BlindParams::default();
    (defaults.min_scale_arcsec_px, defaults.max_scale_arcsec_px)
}

fn solve(args: &AstapArgs, image_path: &Path) -> Result<Vec<String>> {
    let star_data = resolve_star_data()?;
    let catalog = seiza::catalog::TileCatalog::open(&star_data)
        .with_context(|| format!("cannot open star catalog {}", star_data.display()))?;

    let image = crate::load_image(image_path, args.detection_backend)?;
    let dims = image.dimensions();
    let config = seiza::DetectConfig {
        backend: args.detection_backend,
        max_stars: args.max_stars.unwrap_or(0).clamp(0, 2000).max(200),
        ..Default::default()
    };
    let mut invocation =
        crate::SolveInvocation::new(image_path, &image, config, args.detection_fallback);

    // Pixel scale from the height FOV when provided
    let scale = if args.fov_deg > 0.0 {
        Some(args.fov_deg * 3600.0 / dims.1 as f64)
    } else {
        None
    };

    let hint = match (args.ra_hours, args.spd_deg) {
        (Some(ra), Some(spd)) if args.radius_deg < 180.0 => Some((ra * 15.0, spd - 90.0)),
        _ => None,
    };

    // The hinted solver works best with tight radii; N.I.N.A. commonly
    // passes wide ones (30 deg). Try near the hint first, then fall back
    // to a blind search, which covers any radius
    let hinted = match (hint, scale) {
        (Some(center), Some(scale)) => invocation
            .solve(|stars| {
                seiza::solve::solve(
                    stars,
                    &catalog,
                    &seiza::solve::SolveHint {
                        center,
                        radius_deg: args.radius_deg.clamp(0.5, 3.0),
                        scale_arcsec_px: scale,
                        scale_tolerance: 0.3,
                        sip_order: 0,
                    },
                    dims,
                )
            })
            .ok(),
        _ => None,
    };

    let solution = match hinted {
        Some(solution) => solution,
        None => {
            // A failed hinted solve makes both the pointing and scale hints
            // suspect. Search the practical scale range instead of turning
            // N.I.N.A.'s -fov estimate into another hard failure boundary.
            let (min_scale, max_scale) = blind_scale_search_range(scale);
            let mut params = seiza::blind::BlindParams {
                min_scale_arcsec_px: min_scale,
                max_scale_arcsec_px: max_scale,
                ..Default::default()
            };
            let index = if let Some(path) = resolve_blind_index()? {
                let index = seiza::blind::BlindIndex::open(&path)
                    .map_err(anyhow::Error::from)
                    .with_context(|| format!("cannot open blind index {}", path.display()))?;
                params.index_mag_limit = index.index_mag_limit();
                params.max_pattern_deg = index.max_pattern_deg();
                let built_from = index.source_star_count();
                let runtime = catalog.star_count();
                if built_from > 0 && built_from.max(runtime) > 2 * built_from.min(runtime) {
                    eprintln!(
                        "warning: blind index built from {built_from} stars, catalog has \
                         {runtime}; deep-tier hypotheses may never verify"
                    );
                }
                index
            } else {
                // Without a prebuilt index only the default bright tiers
                // (G<=12.7) build at startup: a deep whole-sky index over a
                // 154M-star catalog takes minutes and gigabytes, which
                // inside an imaging loop reads as a hang. Small fine-scale
                // fields need the hosted index (see resolve_blind_index).
                seiza::blind::BlindIndex::build(&catalog, &params)
            };
            let can_retry_f32 = args.detection_fallback == crate::DetectionFallback::F32
                && crate::auto_can_retry_f32(image_path, &image, args.detection_backend);
            invocation.solve_with_pass(|stars, pass| {
                let attempt_params = crate::blind_params_for_detection_pass(
                    &params,
                    can_retry_f32,
                    args.detection_fallback_hypotheses,
                    pass,
                );
                seiza::blind::solve_blind(stars, &catalog, &index, &attempt_params, dims)
            })?
        }
    };

    Ok(ini_lines(&solution.wcs, dims))
}

/// The solved WCS as ASTAP-style ini lines. seiza's TAN WCS uses the
/// standard FITS CD convention, so values pass through directly; CRPIX
/// converts from 0-based pixel centers to FITS 1-based.
fn ini_lines(wcs: &seiza::Wcs, dims: (u32, u32)) -> Vec<String> {
    let cd = wcs.cd;
    // Informational CDELT/CROTA the way ASTAP reports them
    let cdelt2 = (cd[0][1] * cd[0][1] + cd[1][1] * cd[1][1]).sqrt();
    let determinant = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0];
    let cdelt1 = cdelt2.copysign(determinant);
    let crota2 = (-cd[0][1]).atan2(cd[1][1]).to_degrees();

    vec![
        "PLTSOLVD=T".to_string(),
        format!("CRPIX1={:.8}", wcs.crpix.0 + 1.0),
        format!("CRPIX2={:.8}", wcs.crpix.1 + 1.0),
        format!("CRVAL1={:.8}", wcs.crval.0),
        format!("CRVAL2={:.8}", wcs.crval.1),
        format!("CDELT1={cdelt1:.10}"),
        format!("CDELT2={cdelt2:.10}"),
        format!("CROTA2={crota2:.6}"),
        format!("CD1_1={:.10}", cd[0][0]),
        format!("CD1_2={:.10}", cd[0][1]),
        format!("CD2_1={:.10}", cd[1][0]),
        format!("CD2_2={:.10}", cd[1][1]),
        format!("DIMENSIONS={}x{}", dims.0, dims.1),
        "COMMENT=solved by seiza (ASTAP-compatible mode)".to_string(),
    ]
}

fn write_ini(path: &Path, lines: &[String]) -> Result<()> {
    let mut content = lines.join("\r\n");
    content.push_str("\r\n");
    std::fs::write(path, content).with_context(|| format!("cannot write {}", path.display()))
}

/// Star catalog resolution shared with every compatibility mode; see
/// [`seiza::data_paths`].
pub(crate) fn resolve_star_data() -> Result<PathBuf> {
    Ok(seiza::data_paths::star_data(None)?)
}

/// Optional prebuilt blind index resolution; see [`seiza::data_paths`].
/// `Ok(None)` means "build the index in memory". A set `SEIZA_BLIND_INDEX`
/// that resolves to nothing propagates as an error here too: silently
/// falling back to a startup-built bright index would honor the pin
/// everywhere except the compatibility modes.
pub(crate) fn resolve_blind_index() -> Result<Option<PathBuf>> {
    Ok(seiza::data_paths::blind_index(None)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nina_style_arguments() {
        let raw: Vec<String> = [
            "-f",
            "C:\\temp\\img.fits",
            "-fov",
            "0.38",
            "-z",
            "0",
            "-s",
            "500",
            "-r",
            "30",
            "-ra",
            "19.7275",
            "-spd",
            "113.379",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let args = parse_args(&raw);
        assert_eq!(args.file.as_deref(), Some(Path::new("C:\\temp\\img.fits")));
        assert!((args.fov_deg - 0.38).abs() < 1e-9);
        assert_eq!(args.max_stars, Some(500));
        assert!((args.radius_deg - 30.0).abs() < 1e-9);
        assert!((args.ra_hours.unwrap() - 19.7275).abs() < 1e-9);
        assert!((args.spd_deg.unwrap() - 113.379).abs() < 1e-9);
        assert_eq!(args.detection_fallback_hypotheses, 64);
    }

    #[test]
    fn blind_invocation_has_no_hint() {
        let raw: Vec<String> = ["-f", "x.fits", "-fov", "1.0", "-r", "180"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let args = parse_args(&raw);
        assert!(args.ra_hours.is_none());
        assert!((args.radius_deg - 180.0).abs() < 1e-9);
    }

    #[test]
    fn blind_fallback_does_not_retain_a_stale_fov_scale_constraint() {
        // Regression: a 749 mm frame presented by a N.I.N.A. profile set to
        // 300 mm has an actual scale near 1.03"/px but a 2.59"/px hint. The
        // old half-to-double bracket started above the real solution.
        let stale_scale_hint = 2.585;
        let actual_scale = 1.029;
        assert!(actual_scale < stale_scale_hint / 2.0);

        let hinted_range = blind_scale_search_range(Some(stale_scale_hint));
        let unhinted_range = blind_scale_search_range(None);
        assert_eq!(hinted_range, unhinted_range);
        assert!(actual_scale >= hinted_range.0 && actual_scale <= hinted_range.1);
    }

    #[test]
    fn recognizes_the_installed_astap_copy_by_name() {
        assert!(invoked_as_astap("astap.exe"));
        assert!(invoked_as_astap("C:\\Program Files\\Seiza CLI\\astap.exe"));
        assert!(invoked_as_astap("/usr/local/bin/astap"));
        assert!(!invoked_as_astap("seiza.exe"));
        assert!(!invoked_as_astap("astap-cli.exe"));
    }

    #[test]
    fn parses_detection_backend_extension() {
        let raw: Vec<String> = [
            "-f",
            "x.jpg",
            "--detection-backend",
            "f32",
            "--detection-fallback",
            "none",
            "--detection-fallback-hypotheses",
            "32",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let args = parse_args(&raw);
        assert_eq!(args.detection_backend, seiza::DetectBackend::F32);
        assert_eq!(args.detection_fallback, crate::DetectionFallback::None);
        assert_eq!(args.detection_fallback_hypotheses, 32);
    }

    #[test]
    fn ini_encodes_scale_rotation_and_parity() {
        // A WCS at 2"/px, rotated 30 deg, standard parity (CDELT1 < 0)
        let s = 2.0 / 3600.0;
        let theta = 30.0f64.to_radians();
        let wcs = seiza::Wcs {
            crval: (123.456, -20.5),
            crpix: (1999.0, 1499.0),
            cd: [
                [-s * theta.cos(), s * theta.sin()],
                [s * theta.sin(), s * theta.cos()],
            ],
            sip: None,
        };
        let lines = ini_lines(&wcs, (4000, 3000));
        let get = |key: &str| -> f64 {
            lines
                .iter()
                .find_map(|l| l.strip_prefix(&format!("{key}=")))
                .unwrap()
                .parse()
                .unwrap()
        };
        assert_eq!(lines[0], "PLTSOLVD=T");
        assert!((get("CRVAL1") - 123.456).abs() < 1e-9);
        assert!((get("CRPIX1") - 2000.0).abs() < 1e-9);
        // Scale recovered the way N.I.N.A. does it
        let scale = (get("CD1_2").powi(2) + get("CD2_2").powi(2)).sqrt() * 3600.0;
        assert!((scale - 2.0).abs() < 1e-6, "scale {scale}");
        // Parity: negative determinant = standard sky orientation here
        assert!(get("CDELT1") < 0.0);
    }
}
