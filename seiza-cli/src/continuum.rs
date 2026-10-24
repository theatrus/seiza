//! CLI preparation and file provenance around Seiza's continuum core.
use crate::provenance::{file_identity, validate_path_roles, write_json_atomic};
use anyhow::{Context, Result, bail};
use clap::Args;
use seiza_fits::{HeaderValue, WriteHeaderCard};
use seiza_stacking::{
    ContinuumFitOptions, ContinuumFitOutcome, ContinuumOrigin, ContinuumPhotometryOptions,
    ContinuumPlan, ContinuumSource, ContinuumSources, ContinuumStar, ContinuumTransfer, FitsFrame,
};
use std::path::{Path, PathBuf};

#[derive(Args)]
pub(crate) struct ContinuumArgs {
    /// Aligned, linear mono narrowband image containing stars
    #[arg(long)]
    narrowband: PathBuf,
    /// Aligned, linear mono continuum references, in coefficient order (1..3)
    #[arg(long, required=true, num_args=1..=3, action=clap::ArgAction::Append)]
    reference: Vec<PathBuf>,
    /// Physical saturation level in narrowband sample units, not an image maximum
    #[arg(long)]
    saturation: f64,
    /// Reference saturation levels in their sample units; defaults to --saturation
    #[arg(long, value_delimiter=',', num_args=1..=3)]
    reference_saturation: Vec<f64>,
    /// Declare untagged inputs linear; DISPLAY tags are always refused
    #[arg(long)]
    assume_linear: bool,
    /// Signed mono output; omit for a fit-only report or plan
    #[arg(long)]
    output: Option<PathBuf>,
    /// Tagged fitted/skipped JSON outcome, including diagnostics
    #[arg(long)]
    report: Option<PathBuf>,
    /// Save the exact-source plan JSON for reuse
    #[arg(long)]
    save_plan: Option<PathBuf>,
    /// Reuse an exact-source plan, bypassing automatic fitting
    #[arg(long, conflicts_with_all=["coefficients", "stars", "positions", "fit_options", "photometry_options"])]
    plan: Option<PathBuf>,
    /// Nonnegative manual weights in reference order; bypasses fitting
    #[arg(long, value_delimiter=',', num_args=1..=3, allow_negative_numbers=true, conflicts_with_all=["stars", "positions", "fit_options", "photometry_options"])]
    coefficients: Vec<f64>,
    /// JSON array of premeasured ContinuumStar objects; otherwise measure images
    #[arg(long, conflicts_with_all=["positions", "photometry_options"])]
    stars: Option<PathBuf>,
    /// JSON array of [x,y] positions; omit to detect stars automatically
    #[arg(long)]
    positions: Option<PathBuf>,
    /// JSON ContinuumFitOptions overrides
    #[arg(long)]
    fit_options: Option<PathBuf>,
    /// JSON aperture/annulus radius overrides
    #[arg(long)]
    photometry_options: Option<PathBuf>,
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_slice(&std::fs::read(path)?)
        .with_context(|| format!("invalid JSON in {}", path.display()))
}
fn open_linear(path: &Path, assume_linear: bool) -> Result<FitsFrame> {
    let frame = FitsFrame::open(path)?;
    if frame.image.channels != 1 || frame.bayer.is_some() {
        bail!(
            "{} must be a prepared mono plane; demosaic Bayer inputs and select a physical reference channel first",
            path.display()
        );
    }
    let tag = frame
        .headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case("SEIZATRF"));
    match tag {
        Some((_, HeaderValue::String(value))) if value.trim().eq_ignore_ascii_case("LINEAR") => {}
        None if assume_linear => {}
        _ => bail!(
            "{} must contain linear samples with stars: require SEIZATRF=LINEAR, or --assume-linear for untagged inputs; non-linear/unknown tags are refused",
            path.display()
        ),
    }
    Ok(frame)
}
fn source(path: &Path, name: String, saturation: f64) -> Result<ContinuumSource> {
    let identity = file_identity(path)?;
    Ok(ContinuumSource {
        name,
        revision: format!(
            "{}:{}:{}",
            seiza_stacking::OUTPUT_VERSION,
            identity.bytes,
            identity.sha256
        ),
        transfer: ContinuumTransfer::Linear,
        saturation,
    })
}

pub(crate) fn run(args: ContinuumArgs) -> Result<()> {
    if args.output.is_none() && args.report.is_none() && args.save_plan.is_none() {
        bail!("pass --output, --report, or --save-plan");
    }
    if args.reference.is_empty() || args.reference.len() > 3 {
        bail!("continuum subtraction needs one to three references");
    }
    if !args.reference_saturation.is_empty()
        && args.reference_saturation.len() != args.reference.len()
    {
        bail!("--reference-saturation must supply one value per reference");
    }
    let mut roles = vec![("narrowband input".into(), args.narrowband.as_path())];
    for (i, path) in args.reference.iter().enumerate() {
        roles.push((format!("reference {}", i + 1), path.as_path()));
    }
    for (role, path) in [
        ("output", &args.output),
        ("report", &args.report),
        ("saved plan", &args.save_plan),
        ("input plan", &args.plan),
        ("stars", &args.stars),
        ("positions", &args.positions),
        ("fit options", &args.fit_options),
        ("photometry options", &args.photometry_options),
    ] {
        if let Some(path) = path {
            roles.push((role.into(), path.as_path()));
        }
    }
    validate_path_roles(roles)?;
    let narrowband = open_linear(&args.narrowband, args.assume_linear)?;
    let references = args
        .reference
        .iter()
        .map(|p| open_linear(p, args.assume_linear))
        .collect::<Result<Vec<_>>>()?;
    let sources = ContinuumSources {
        width: narrowband.image.width,
        height: narrowband.image.height,
        narrowband: source(&args.narrowband, "narrowband".into(), args.saturation)?,
        references: args
            .reference
            .iter()
            .enumerate()
            .map(|(i, path)| {
                source(
                    path,
                    format!("reference_{}", i + 1),
                    args.reference_saturation
                        .get(i)
                        .copied()
                        .unwrap_or(args.saturation),
                )
            })
            .collect::<Result<_>>()?,
    };
    let images: Vec<_> = references.iter().map(|f| &f.image).collect();
    let outcome = if let Some(path) = &args.plan {
        ContinuumFitOutcome::Fitted {
            plan: read_json(path)?,
        }
    } else if !args.coefficients.is_empty() {
        ContinuumFitOutcome::Fitted {
            plan: ContinuumPlan {
                schema_version: 1,
                sources: sources.clone(),
                coefficients: args.coefficients.clone(),
                origin: ContinuumOrigin::Manual,
                diagnostics: None,
            },
        }
    } else {
        let options: ContinuumFitOptions = args
            .fit_options
            .as_deref()
            .map(read_json)
            .transpose()?
            .unwrap_or_default();
        if let Some(path) = &args.stars {
            let stars: Vec<ContinuumStar> = read_json(path)?;
            seiza_stacking::fit_continuum(&stars, &sources, &options)?
        } else {
            let positions: Vec<[f64; 2]> = args
                .positions
                .as_deref()
                .map(read_json)
                .transpose()?
                .unwrap_or_default();
            let photometry: ContinuumPhotometryOptions = args
                .photometry_options
                .as_deref()
                .map(read_json)
                .transpose()?
                .unwrap_or_default();
            seiza_stacking::measure_and_fit_continuum(
                &narrowband.image,
                &images,
                &sources,
                &positions,
                &options,
                &photometry,
            )?
        }
    };
    let plan = match &outcome {
        ContinuumFitOutcome::Fitted { plan } => plan,
        ContinuumFitOutcome::Skipped { reason, detail, .. } => {
            if let Some(path) = &args.report {
                write_json_atomic(path, &outcome)?;
            }
            bail!("continuum subtraction skipped ({reason:?}): {detail}");
        }
    };
    // Validate even fit-only/manual/reused plans before writing any artifact.
    let image = seiza_stacking::subtract_continuum(&narrowband.image, &images, &sources, plan)?;
    if let Some(path) = &args.output {
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let suffix = format!(
            ".{}",
            path.extension().unwrap_or_default().to_string_lossy()
        );
        let temporary = tempfile::Builder::new()
            .prefix(".seiza-continuum-")
            .suffix(&suffix)
            .tempfile_in(parent)?;
        let coefficients = plan
            .coefficients
            .iter()
            .map(|a| format!("{a:.17e}"))
            .collect::<Vec<_>>()
            .join(",");
        seiza_stacking::write_processed_image_fits_f32(
            temporary.path(),
            &image,
            &narrowband.headers,
            &[
                WriteHeaderCard::new("SEIZATRF", HeaderValue::String("LINEAR".into())),
                WriteHeaderCard::new("SEIZACON", HeaderValue::String("EMISSION-RESIDUAL".into()))
                    .with_comment("signed residual; no line-transmission correction"),
                WriteHeaderCard::new(
                    "HISTORY",
                    HeaderValue::String(format!("Continuum weights: {coefficients}")),
                ),
            ],
        )?;
        temporary.as_file().sync_all()?;
        temporary
            .persist(path)
            .with_context(|| format!("failed to replace {}", path.display()))?;
    }
    if let Some(path) = &args.save_plan {
        write_json_atomic(path, plan)?;
    }
    if let Some(path) = &args.report {
        write_json_atomic(path, &outcome)?;
    }
    eprintln!(
        "continuum coefficients: {:?}; signed linear emission residual",
        plan.coefficients
    );
    Ok(())
}
