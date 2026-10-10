//! `seiza parallax-video`: fly toward a point of a stretched image, with its
//! stars at their Gaia distances.
//!
//! The image is split into a starless image and its stars (by
//! StarXTerminator, or given as two files), plate-solved, and its stars
//! matched to Gaia DR3 for their Bailer-Jones distances, with Hipparcos for
//! the brightest stars Gaia has no parallax for. [`seiza_parallax`] renders
//! the frames and an encoder writes them.

use anyhow::{Context, Result, bail};
use clap::{Args, ValueEnum};
use image::Rgb32FImage;
use seiza::{DetectConfig, Wcs};
use seiza_parallax::{
    CutOptions, Easing, Extent, FfmpegSink, FrameSink, LightImage, PngSequence, Scene, Shot,
    SmallStars, Star, Start, VideoSettings,
};
use seiza_sources::{GaiaDistance, HipparcosStar};
use seiza_stars::PeakStar;
use std::path::{Path, PathBuf};

#[derive(Args)]
pub(crate) struct ParallaxVideoArgs {
    /// The stretched image: PNG, JPEG or TIFF. Without --starless and
    /// --stars, StarXTerminator splits it into the two
    image: Option<PathBuf>,
    /// The image with its stars removed, stretched like the stars image
    #[arg(long, requires = "stars")]
    starless: Option<PathBuf>,
    /// The stars alone, made to be screen-blended back over --starless
    /// (StarXTerminator's "unscreen" stars image)
    #[arg(long, requires = "starless")]
    stars: Option<PathBuf>,
    /// The video to write (.mp4), or a directory of PNG frames with
    /// --encoder png
    #[arg(short, long)]
    output: PathBuf,
    /// Point to fly toward, as image pixels `x,y` (default: the image
    /// centre)
    #[arg(long, value_parser = parse_point)]
    focus: Option<(f64, f64)>,
    /// Distance to the nebula or galaxy behind the stars, in parsecs
    /// (default: the distance of the catalogued object at the focus point,
    /// else the median Gaia distance of the stars near it)
    #[arg(long)]
    distance: Option<f64>,
    /// Object catalog file or directory, for finding the target (default:
    /// standard catalog locations)
    #[arg(long)]
    objects: Option<PathBuf>,
    /// Object distance file or directory (default: beside the object
    /// catalog, then the standard catalog locations)
    #[arg(long)]
    distances: Option<PathBuf>,
    /// Distance for stars without one, in parsecs (default: the median of
    /// the matched stars' distances, and never nearer than the nebula)
    #[arg(long)]
    unmatched_distance: Option<f64>,
    /// Fraction of the way to the nebula the camera flies
    #[arg(long, default_value_t = 0.4)]
    dolly: f64,
    /// Sideways swing at the middle of the shot, as a fraction of the
    /// nebula's distance: the camera moves out and back without turning, so
    /// near stars slide across far ones (reduced if the far stars would
    /// slide off the image). The default flies straight in
    #[arg(long, default_value_t = 0.0)]
    truck: f64,
    /// Magnification added over the shot by lengthening the lens, which
    /// enlarges every depth alike
    #[arg(long, default_value_t = 1.0)]
    zoom_end: f64,
    /// How much of its way to the focus point the camera turns rather than
    /// moves, 0 to 1 (reduced if the far stars would slide off the image).
    /// Turning sweeps the distant star field along with the nebula: a little
    /// keeps the target nearer the centre early on, much of it looks like the
    /// sky spinning
    #[arg(long, default_value_t = 0.0)]
    pan: f64,
    /// Direction of the sideways travel, degrees anticlockwise from the
    /// image's rightward axis
    #[arg(long, default_value_t = 0.0)]
    truck_angle: f64,
    /// What the first frame shows: `focus`, the widest view centred on the
    /// focus point, or `whole`, the widest view of the whole image, the aim
    /// then moving to the focus point as the camera flies in
    #[arg(long, value_enum, default_value_t = StartArg::Focus)]
    start: StartArg,
    /// How far the first frame zooms in: 1 shows the widest view --start
    /// allows inside the image
    #[arg(long, default_value_t = 1.0)]
    zoom: f64,
    /// Length of the video, seconds
    #[arg(long, default_value_t = 8.0)]
    seconds: f64,
    /// Frames per second
    #[arg(long, default_value_t = 30)]
    fps: u32,
    /// Frame size, `WIDTHxHEIGHT`, even numbers
    #[arg(long, default_value = "1920x1080", value_parser = parse_size)]
    size: (usize, usize),
    /// How the camera speeds up and slows down
    #[arg(long, value_enum, default_value_t = EasingArg::InOut)]
    easing: EasingArg,
    /// A star the camera nears grows with it up to this many times its size
    #[arg(long, default_value_t = 4.0)]
    growth_limit: f64,
    /// A star past this growth fades out as the camera flies by it
    #[arg(long, default_value_t = 6.0)]
    fade_from: f64,
    /// How many stars, brightest first, fly at their own distances. A deep
    /// image holds so many faint stars that, each moving on its own, they
    /// crowd the view; see --small-stars for the rest (default: all fly)
    #[arg(long)]
    max_stars: Option<usize>,
    /// What becomes of the stars past --max-stars: `drop` removes them,
    /// and `field` keeps them on the distant star field's plane
    #[arg(long, value_enum, default_value_t = SmallStarsArg::Drop)]
    small_stars: SmallStarsArg,
    /// How the video is written: auto uses ffmpeg when it runs, then the
    /// built-in encoder when this build has one
    #[arg(long, value_enum, default_value_t = EncoderArg::Auto)]
    encoder: EncoderArg,
    /// The ffmpeg program to run
    #[arg(long, default_value = "ffmpeg")]
    ffmpeg: PathBuf,
    /// Faintest Gaia G magnitude to match. Fainter stars stay where the
    /// unmatched ones go; a wide field holds too many for one archive query
    #[arg(long, default_value_t = 16.0)]
    gaia_max_mag: f32,
    /// Where fetched Gaia fields are kept for reuse (default: Seiza's data
    /// directory)
    #[arg(long)]
    gaia_cache: Option<PathBuf>,
    /// Star distance file (`seiza setup --star-distances`) giving the
    /// field's Gaia and Hipparcos stars offline (default: the standard
    /// catalog locations; without one they are fetched online)
    #[arg(long)]
    star_distances: Option<PathBuf>,
    /// Star tile file or catalog directory for plate solving (default:
    /// standard catalog locations)
    #[arg(long)]
    data: Option<PathBuf>,
    /// Blind pattern index (default: found beside the star catalog)
    #[arg(long)]
    index: Option<PathBuf>,
    /// Pixel-scale search bounds, arcseconds per pixel
    #[arg(long)]
    min_scale: Option<f64>,
    #[arg(long)]
    max_scale: Option<f64>,
    /// Keep the starless and stars images StarXTerminator made, beside the
    /// output
    #[arg(long)]
    keep_split: bool,
    /// Write the scene's layers to this directory as PNG files: the
    /// background, the star light no star took, and the stars cut out,
    /// marked by their distance
    #[arg(long)]
    debug_layers: Option<PathBuf>,
    /// Leave galaxies in the starless image, on the nebula's plane, rather
    /// than lifting them onto the far field where they hold still
    #[arg(long)]
    keep_galaxies: bool,
    /// Leave what lies behind the nebula as bright as photographed, rather
    /// than dimming it as it slides behind thicker dust (mapped from how
    /// few stars show through)
    #[arg(long)]
    no_dust: bool,
    /// How dark the dust is for the stars it hides: the light it lets
    /// through is the share of stars seen to this power
    #[arg(long, default_value_t = 3.0)]
    dust_opacity: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum StartArg {
    Focus,
    Whole,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum SmallStarsArg {
    Drop,
    Field,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum EasingArg {
    Linear,
    InOut,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum EncoderArg {
    Auto,
    Ffmpeg,
    Openh264,
    Png,
}

fn parse_point(text: &str) -> std::result::Result<(f64, f64), String> {
    let (x, y) = text
        .split_once(',')
        .ok_or_else(|| format!("expected x,y; got {text}"))?;
    let number = |part: &str| {
        part.trim()
            .parse::<f64>()
            .map_err(|error| format!("{part}: {error}"))
    };
    Ok((number(x)?, number(y)?))
}

fn parse_size(text: &str) -> std::result::Result<(usize, usize), String> {
    let (width, height) = text
        .split_once(['x', 'X'])
        .ok_or_else(|| format!("expected WIDTHxHEIGHT; got {text}"))?;
    let number = |part: &str| {
        part.trim()
            .parse::<usize>()
            .map_err(|error| format!("{part}: {error}"))
    };
    let size = (number(width)?, number(height)?);
    if size.0 < 16 || size.1 < 16 || size.0 % 2 != 0 || size.1 % 2 != 0 {
        return Err(format!("{text}: sides must be even and at least 16"));
    }
    Ok(size)
}

/// Open a stretched raster with values 0 to 1, in its EXIF orientation.
fn open_display(path: &Path) -> Result<Rgb32FImage> {
    Ok(seiza::raster::open_oriented(path)
        .with_context(|| format!("failed to open {}", path.display()))?
        .pixels
        .to_rgb32f())
}

pub(crate) fn run(args: ParallaxVideoArgs) -> Result<()> {
    check_shot(&args)?;
    let started = std::time::Instant::now();
    let (starless, stars, solve_path, _split_dir) = split(&args)?;
    if starless.dimensions() != stars.dimensions() {
        bail!(
            "the starless image is {:?} but the stars image is {:?}",
            starless.dimensions(),
            stars.dimensions()
        );
    }
    let (width, height) = (stars.width() as usize, stars.height() as usize);

    let wcs = solve(&args, &solve_path)?;
    let scale = wcs.scale_arcsec_per_px();
    let focal_px = 206_264.806_247 / scale;

    let stars_light = LightImage::from_display(&stars);
    let detections =
        seiza_stars::fold_core_fragments(seiza_parallax::find_stars(&stars_light, 5.0));
    println!("{} stars found in the stars image", detections.len());
    let (gaia, hipparcos) = match offline_field(&args, &wcs, width, height)? {
        Some(field) => field,
        None => (
            gaia_field(&args, &wcs, width, height)?,
            hipparcos_field(&args, &wcs, width, height),
        ),
    };
    let (matched, gaia_matches) = match_distances(&detections, &gaia, &hipparcos, &wcs, scale);
    let with_distance = matched
        .iter()
        .filter(|star| star.distance_pc.is_some())
        .count();
    println!("{gaia_matches} matched to Gaia, {with_distance} with a distance");

    let focus = args
        .focus
        .unwrap_or(((width as f64 - 1.0) / 2.0, (height as f64 - 1.0) / 2.0));
    if focus.0 < 0.0 || focus.1 < 0.0 || focus.0 >= width as f64 || focus.1 >= height as f64 {
        bail!("the focus point {focus:?} is outside the {width}x{height} image");
    }
    let distance = match args.distance {
        Some(distance) => distance,
        None => match target_distance(&args, &wcs, (width, height), focus) {
            Ok(Some((name, distance, basis))) => {
                println!("background at {distance:.0} pc, the distance of {name} ({basis})");
                distance
            }
            found => {
                if let Err(error) = found {
                    eprintln!("warning: no object distance: {error:#}");
                }
                let distance = median_distance_near(&matched, focus, width, height).context(
                    "no catalogued object or Gaia distances near the focus point to place the \
                     background at; pass --distance",
                )?;
                println!(
                    "background at {distance:.0} pc, the median distance of the stars near the \
                     focus point (no catalogued object there; set it with --distance)"
                );
                distance
            }
        },
    };

    // Most stars too faint to match are field stars well beyond a nearby
    // target, so they go to the matched stars' median distance rather than
    // onto the nebula.
    let unmatched = args.unmatched_distance.unwrap_or_else(|| {
        median(matched.iter().filter_map(|star| star.distance_pc))
            .unwrap_or(distance)
            .max(distance)
    });
    println!("stars without a distance at {unmatched:.0} pc");
    let placed: Vec<Star> = matched
        .iter()
        .map(|star| Star {
            distance_pc: Some(star.distance_pc.unwrap_or(unmatched)),
            ..*star
        })
        .collect();

    let mut starless = LightImage::from_display(&starless);
    // Galaxies lie far beyond everything else, but the star remover leaves
    // them on the nebula's plane.
    let mut galaxies = Vec::new();
    if !args.keep_galaxies {
        match galaxies_in_image(&args, &wcs, (width, height), focus) {
            Ok(found) => {
                for (name, extent) in found {
                    if let Some(sprite) =
                        seiza_parallax::lift_object(&mut starless, &extent, GALAXY_DISTANCE_PC)
                    {
                        galaxies.push((name, sprite));
                    }
                }
            }
            Err(error) => eprintln!("warning: no galaxies lifted: {error:#}"),
        }
        if !galaxies.is_empty() {
            let names: Vec<&str> = galaxies
                .iter()
                .take(5)
                .map(|(name, _)| name.as_str())
                .collect();
            println!(
                "{} galaxies lifted onto the far field: {}{}",
                galaxies.len(),
                names.join(", "),
                if galaxies.len() > names.len() {
                    ", ..."
                } else {
                    ""
                }
            );
        }
    }
    let mut scene = Scene::new(
        &starless,
        &stars_light,
        &placed,
        distance,
        unmatched,
        focal_px,
        &CutOptions {
            max_stars: args.max_stars,
            small_stars: match args.small_stars {
                SmallStarsArg::Drop => SmallStars::Drop,
                SmallStarsArg::Field => SmallStars::Field,
            },
            ..CutOptions::default()
        },
    );
    scene
        .sprites
        .extend(galaxies.into_iter().map(|(_, sprite)| sprite));
    if !args.no_dust {
        // The stars seen through the dust: all but the matched ones in
        // front of it. The unmatched ones count however near the dust they
        // were placed, as faint stars are mostly far.
        let behind: Vec<(f64, f64)> = matched
            .iter()
            .filter(|star| star.distance_pc.is_none_or(|pc| pc > distance))
            .map(|star| (star.x, star.y))
            .collect();
        scene.dust =
            seiza_parallax::Dust::from_star_counts(&behind, width, height, args.dust_opacity)
                .map(|dust| dust.with_darkness(&starless, &behind));
        match &scene.dust {
            Some(dust) => {
                let (cells, _) = dust.cells();
                let mut sorted = cells.to_vec();
                sorted.sort_by(f32::total_cmp);
                println!(
                    "dust mapped from {} stars behind it and the nebula's dark places: median \
                     transmission {:.0}%, thickest {:.0}%",
                    behind.len(),
                    100.0 * sorted[sorted.len() / 2],
                    100.0 * sorted[0]
                );
            }
            None => println!("too few stars to map the dust; nothing dims behind it"),
        }
    }
    if args.max_stars.is_some() {
        println!(
            "the {} brightest stars fly; the rest {}",
            scene.sprites.len(),
            match args.small_stars {
                SmallStarsArg::Drop => "are dropped",
                SmallStarsArg::Field => "stay on the star field",
            }
        );
    }
    if let Some(directory) = &args.debug_layers {
        write_layers(directory, &scene)?;
    }
    let (sin, cos) = args.truck_angle.to_radians().sin_cos();
    let frames = ((args.seconds * args.fps as f64).round() as usize).max(2);
    let shot = Shot {
        focus,
        dolly: args.dolly,
        truck: (args.truck * cos, -args.truck * sin),
        start: match args.start {
            StartArg::Focus => Start::Focus,
            StartArg::Whole => Start::Whole,
        },
        pan: args.pan,
        zoom: args.zoom,
        zoom_end: args.zoom_end,
        width: args.size.0,
        height: args.size.1,
        frames,
        easing: match args.easing {
            EasingArg::Linear => Easing::Linear,
            EasingArg::InOut => Easing::InOut,
        },
        growth_limit: args.growth_limit,
        fade_from: args.fade_from,
        ..Shot::default()
    };

    let (shot, fitted) = shot.fitted(&scene);
    if shot.pan < args.pan {
        println!(
            "pan reduced to {:.2} so the far stars stay inside the image",
            shot.pan
        );
    }
    if shot.lead < 1.0 {
        println!(
            "sideways travel toward the focus point comes later (lead {:.2}) so the far stars \
             stay inside the image",
            shot.lead
        );
    }
    if fitted < 1.0 && args.truck != 0.0 {
        println!(
            "truck reduced to {:.4} of the distance so the far stars stay inside the image",
            args.truck * fitted
        );
    }

    let settings = VideoSettings {
        width: args.size.0 as u32,
        height: args.size.1 as u32,
        fps: args.fps,
        bitrate: (args.size.0 * args.size.1 * args.fps as usize / 2).min(u32::MAX as usize) as u32,
    };
    let mut sink = open_sink(&args, settings)?;
    let rendering = std::time::Instant::now();
    for frame in 0..frames {
        let image = shot.render(&scene, frame).to_display_rgb8();
        sink.push(&image)?;
        if (frame + 1) % (frames / 10).max(1) == 0 || frame + 1 == frames {
            println!("rendered {}/{frames} frames", frame + 1);
        }
    }
    sink.finish()?;
    println!(
        "wrote {} ({frames} frames in {:.1}s, {:.1}s in all)",
        args.output.display(),
        rendering.elapsed().as_secs_f64(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn check_shot(args: &ParallaxVideoArgs) -> Result<()> {
    if args.image.is_none() && args.starless.is_none() {
        bail!("give an image to split, or --starless and --stars");
    }
    if !(0.0..1.0).contains(&args.dolly) {
        bail!("--dolly must be at least 0 and below 1");
    }
    if !(0.0..=1.0).contains(&args.pan) {
        bail!("--pan must be from 0 to 1");
    }
    if args.dust_opacity.is_nan() || args.dust_opacity < 0.0 {
        bail!("--dust-opacity must be at least 0");
    }
    if !(1.0..).contains(&args.zoom_end) {
        bail!("--zoom-end must be at least 1");
    }
    if args.seconds.is_nan() || args.seconds <= 0.0 || args.fps == 0 {
        bail!("--seconds and --fps must be positive");
    }
    if let Some(distance) = args.distance
        && (distance.is_nan() || distance <= 0.0)
    {
        bail!("--distance must be positive");
    }
    Ok(())
}

/// The starless and stars images, the path to plate-solve, and the
/// directory holding a split this run made, if any.
fn split(
    args: &ParallaxVideoArgs,
) -> Result<(Rgb32FImage, Rgb32FImage, PathBuf, Option<tempfile::TempDir>)> {
    if let (Some(starless), Some(stars)) = (&args.starless, &args.stars) {
        let solve_path = args.image.clone().unwrap_or_else(|| stars.clone());
        return Ok((
            open_display(starless)?,
            open_display(stars)?,
            solve_path,
            None,
        ));
    }
    let image_path = args.image.as_ref().expect("checked in check_shot");
    let image = open_display(image_path)?;
    println!("splitting {} with StarXTerminator", image_path.display());
    let (starless, stars) = star_x_terminator(&image)?;
    if args.keep_split {
        let stem = args.output.file_stem().map_or_else(
            || "parallax".into(),
            |stem| stem.to_string_lossy().into_owned(),
        );
        let directory = args.output.parent().unwrap_or(Path::new("."));
        for (suffix, split) in [("starless", &starless), ("stars", &stars)] {
            let path = directory.join(format!("{stem}-{suffix}.png"));
            image::DynamicImage::ImageRgb32F(split.clone())
                .to_rgb16()
                .save(&path)
                .with_context(|| format!("failed to write {}", path.display()))?;
            println!("wrote {}", path.display());
        }
    }
    Ok((starless, stars, image_path.clone(), None))
}

/// Split a stretched image with StarXTerminator into its starless image
/// and an unscreened stars image.
fn star_x_terminator(image: &Rgb32FImage) -> Result<(Rgb32FImage, Rgb32FImage)> {
    use seiza_stacking::{ExternalParameterValue, ExternalToolRequest, LinearImage, RcAstroCli};
    let cli = match std::env::var_os("SEIZA_RC_ASTRO") {
        Some(path) if !path.is_empty() => RcAstroCli::with_executable(PathBuf::from(path)),
        _ => RcAstroCli::locate().context(
            "StarXTerminator's rc-astro CLI was not found on PATH (or set SEIZA_RC_ASTRO); \
             pass --starless and --stars instead",
        )?,
    };
    let schema = cli.tool_schema("sxt")?;
    if !schema.licensed {
        bail!(
            "StarXTerminator is not licensed on this machine ({}); activate it, or pass \
             --starless and --stars",
            schema
                .license_message
                .as_deref()
                .unwrap_or("no license message")
        );
    }
    let request = ExternalToolRequest {
        tool: "sxt".into(),
        parameters: vec![
            ("stars".into(), ExternalParameterValue::Bool(true)),
            ("unscreen".into(), ExternalParameterValue::Bool(true)),
        ],
        device: None,
    };
    let linear = LinearImage::new(
        image.width() as usize,
        image.height() as usize,
        3,
        image.as_raw().clone(),
    )?;
    let mut last = -1;
    let processed = cli.process_image(&schema, &request, &linear, &[], None, &mut |fraction| {
        let percent = (fraction * 100.0) as i32;
        if percent / 10 != last / 10 {
            println!("StarXTerminator {percent}%");
            last = percent;
        }
    })?;
    let stars = processed
        .stars
        .context("StarXTerminator wrote no stars image")?;
    let to_display = |image: LinearImage| {
        let data = image
            .data
            .iter()
            .map(|value| value.clamp(0.0, 1.0))
            .collect();
        Rgb32FImage::from_raw(image.width as u32, image.height as u32, data)
            .expect("an RGB image of its own size")
    };
    Ok((to_display(processed.image), to_display(stars)))
}

fn solve(args: &ParallaxVideoArgs, path: &Path) -> Result<Wcs> {
    let data = crate::with_data_flag_hint(seiza::data_paths::star_data(args.data.as_deref()))?;
    let index = seiza::data_paths::blind_index_beside(args.index.as_deref(), &data)?;
    let options = crate::SolveBlindOptions {
        index_path: index.as_deref(),
        min_scale: args.min_scale,
        max_scale: args.max_scale,
        index_mag_limit: 12.7,
        max_hypotheses: 400,
        max_coarse_hypotheses: 20_000,
        sip_order: 2,
        sigma: 4.0,
        ignore_border: 0,
        detection_backend: seiza::DetectBackend::Auto,
        detection_fallback: crate::DetectionFallback::F32,
        detection_fallback_hypotheses: 64,
        annotate: None,
        wcs_path: None,
        sky_map: None,
    };
    let image = crate::load_image(path, options.detection_backend)?;
    let dims = image.dimensions();
    let metadata = if crate::is_astronomy_image_path(path) {
        seiza::raster::PhotoMetadata::default()
    } else {
        seiza::raster::PhotoMetadata::read(path)
    };
    let search = seiza::raster::ScaleSearch::new(&metadata, dims, args.min_scale, args.max_scale)?;
    let config = DetectConfig {
        backend: options.detection_backend,
        sigma: options.sigma,
        max_stars: 600,
        ..Default::default()
    };
    let can_retry_f32 = crate::auto_can_retry_f32(path, &image, options.detection_backend);
    let mut invocation =
        crate::SolveInvocation::new(path, &image, config, options.detection_fallback);
    let catalog = seiza::catalog::TileCatalog::open(&data)
        .with_context(|| format!("failed to open {}", data.display()))?;
    let (solution, elapsed) = crate::blind_solve_invocation(
        &mut invocation,
        &catalog,
        &search,
        &options,
        can_retry_f32,
        dims,
    )
    .with_context(|| format!("could not plate-solve {}", path.display()))?;
    println!(
        "solved in {:.1}s: {:.3}\"/px, {} stars matched",
        elapsed.as_secs_f64(),
        solution.wcs.scale_arcsec_per_px(),
        solution.matched_stars
    );
    Ok(solution.wcs)
}

/// The sky circle around the image and its centre.
fn field(wcs: &Wcs, width: usize, height: usize) -> ((f64, f64), f64) {
    let centre = wcs.pixel_to_world(width as f64 / 2.0, height as f64 / 2.0);
    let radius = [
        (0.0, 0.0),
        (width as f64, 0.0),
        (0.0, height as f64),
        (width as f64, height as f64),
    ]
    .iter()
    .map(|&(x, y)| separation(centre, wcs.pixel_to_world(x, y)))
    .fold(0.0_f64, f64::max)
        * 1.02;
    (centre, radius)
}

fn separation(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (ra1, dec1) = (a.0.to_radians(), a.1.to_radians());
    let (ra2, dec2) = (b.0.to_radians(), b.1.to_radians());
    let haversine = ((dec2 - dec1) / 2.0).sin().powi(2)
        + dec1.cos() * dec2.cos() * ((ra2 - ra1) / 2.0).sin().powi(2);
    (2.0 * haversine.sqrt().asin()).to_degrees()
}

fn default_gaia_cache() -> PathBuf {
    let catalogs = seiza::data_paths::default_catalog_dir();
    catalogs
        .parent()
        .map_or_else(|| catalogs.clone(), Path::to_path_buf)
        .join("gaia-fields")
}

/// The field's Gaia and Hipparcos stars from the offline star distance
/// file, or `None` when there is none or it stops short of the magnitude
/// asked for.
fn offline_field(
    args: &ParallaxVideoArgs,
    wcs: &Wcs,
    width: usize,
    height: usize,
) -> Result<Option<(Vec<GaiaDistance>, Vec<HipparcosStar>)>> {
    let Some(path) = seiza::data_paths::star_distances(args.star_distances.as_deref())? else {
        return Ok(None);
    };
    let catalog = seiza::catalog::StarDistanceCatalog::open(&path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    if catalog.max_mag() < args.gaia_max_mag {
        println!(
            "{} holds Gaia stars to G {}, short of --gaia-max-mag {}; fetching online",
            path.display(),
            catalog.max_mag(),
            args.gaia_max_mag
        );
        return Ok(None);
    }
    let (centre, radius) = field(wcs, width, height);
    let (mut gaia, mut hipparcos) = (Vec::new(), Vec::new());
    // Hipparcos stars whatever --gaia-max-mag says, as online.
    for star in catalog.cone_search(centre.0, centre.1, radius, 99.0) {
        if star.hipparcos {
            hipparcos.push(HipparcosStar {
                hip: 0,
                ra: star.ra,
                dec: star.dec,
                // A parallax that gives back the distance, as precise as
                // the catalogue found it.
                parallax: star.distance_pc.map(|distance| 1000.0 / distance),
                parallax_error: Some(0.0),
                hp_mag: Some(star.mag),
                pmra: None,
                pmdec: None,
            });
        } else if star.mag <= args.gaia_max_mag {
            gaia.push(GaiaDistance {
                ra: star.ra,
                dec: star.dec,
                pmra: None,
                pmdec: None,
                g: star.mag,
                bp_rp: None,
                parallax: None,
                parallax_error: None,
                distance: star.distance_pc,
                distance_low: None,
                distance_high: None,
            });
        }
    }
    println!(
        "{} Gaia and {} Hipparcos stars from {}",
        gaia.len(),
        hipparcos.len(),
        path.display()
    );
    Ok(Some((gaia, hipparcos)))
}

fn gaia_field(
    args: &ParallaxVideoArgs,
    wcs: &Wcs,
    width: usize,
    height: usize,
) -> Result<Vec<GaiaDistance>> {
    let (centre, radius) = field(wcs, width, height);
    let cache = args.gaia_cache.clone().unwrap_or_else(default_gaia_cache);
    let path = cache.join(format!(
        "gaia-dr3-distances-{:.2}{:+.2}-r{:.2}-g{}.csv",
        centre.0, centre.1, radius, args.gaia_max_mag
    ));
    if let Ok(csv) = std::fs::read_to_string(&path)
        && let Ok(stars) = seiza_sources::parse_gaia_distances(&csv)
    {
        println!("{} Gaia stars from {}", stars.len(), path.display());
        return Ok(stars);
    }
    let cones = cones(wcs, width, height);
    println!(
        "fetching Gaia DR3 distances within {radius:.2} deg of ({:.4}, {:+.4}) in {} cone(s)",
        centre.0,
        centre.1,
        cones.len()
    );
    let downloader = seiza_sources::SourceDownloader::new()?;
    let max_mag = args.gaia_max_mag;
    // Each cone is kept as it arrives, so a run stopped part way, or one
    // whose last cone fails, does not fetch the others again.
    let cone_path = move |&(ra, dec, radius): &(f64, f64, f64)| {
        format!("gaia-dr3-distances-cone-{ra:.4}{dec:+.4}-r{radius:.4}-g{max_mag}.csv")
    };
    let _ = std::fs::create_dir_all(&cache);
    let mut bodies = Vec::new();
    let mut missing = Vec::new();
    for cone in cones {
        match std::fs::read_to_string(cache.join(cone_path(&cone))) {
            Ok(csv) if seiza_sources::parse_gaia_distances(&csv).is_ok() => bodies.push(csv),
            _ => missing.push(cone),
        }
    }
    let cone_cache = cache.clone();
    let fetched = runtime()?
        .block_on(async move {
            // A few archive queries at once.
            let mut pending = missing.into_iter();
            let mut running = tokio::task::JoinSet::new();
            let mut fetched = Vec::new();
            loop {
                while running.len() < 4
                    && let Some(cone) = pending.next()
                {
                    let downloader = downloader.clone();
                    running.spawn(async move {
                        let (ra, dec, radius) = cone;
                        let csv = downloader
                            .gaia_distance_cone_csv(ra, dec, radius, max_mag)
                            .await;
                        (cone, csv)
                    });
                }
                let Some(done) = running.join_next().await else {
                    break;
                };
                let (cone, csv) = done.context("a Gaia query task failed")?;
                let csv = csv?;
                let path = cone_cache.join(cone_path(&cone));
                let partial = path.with_extension("csv.partial");
                if std::fs::write(&partial, &csv).is_ok() {
                    let _ = std::fs::rename(&partial, &path);
                }
                fetched.push(csv);
                println!("  {} cone(s) fetched", fetched.len());
            }
            anyhow::Ok(fetched)
        })
        .context("Gaia archive query failed")?;
    bodies.extend(fetched);
    let csv = seiza_sources::merge_csv(&bodies);
    let stars = seiza_sources::parse_gaia_distances(&csv)?;
    if std::fs::create_dir_all(&cache).is_ok() {
        let partial = path.with_extension("csv.partial");
        if std::fs::write(&partial, &csv).is_ok() {
            let _ = std::fs::rename(&partial, &path);
        }
    }
    println!("{} Gaia stars", stars.len());
    Ok(stars)
}

/// Cones about 1.2 degrees in radius covering the image, on a grid of
/// image cells: `(ra, dec, radius)`.
fn cones(wcs: &Wcs, width: usize, height: usize) -> Vec<(f64, f64, f64)> {
    let scale_deg = wcs.scale_arcsec_per_px() / 3600.0;
    let cell = (1.6 / scale_deg).max(1.0);
    let columns = (width as f64 / cell).ceil().max(1.0) as usize;
    let rows = (height as f64 / cell).ceil().max(1.0) as usize;
    let (cell_w, cell_h) = (width as f64 / columns as f64, height as f64 / rows as f64);
    let mut cones = Vec::with_capacity(columns * rows);
    for row in 0..rows {
        for column in 0..columns {
            let (left, top) = (column as f64 * cell_w, row as f64 * cell_h);
            let centre = wcs.pixel_to_world(left + cell_w / 2.0, top + cell_h / 2.0);
            let radius = [
                (left, top),
                (left + cell_w, top),
                (left, top + cell_h),
                (left + cell_w, top + cell_h),
            ]
            .iter()
            .map(|&(x, y)| separation(centre, wcs.pixel_to_world(x, y)))
            .fold(0.0_f64, f64::max)
                * 1.02;
            cones.push((centre.0, centre.1, radius));
        }
    }
    cones
}

/// Hipparcos stars in the field, or none when VizieR does not answer: they
/// only fill in the brightest stars' distances.
fn hipparcos_field(
    args: &ParallaxVideoArgs,
    wcs: &Wcs,
    width: usize,
    height: usize,
) -> Vec<HipparcosStar> {
    let (centre, radius) = field(wcs, width, height);
    // Kept beside the Gaia fields, so a field is asked for once.
    let cache = args.gaia_cache.clone().unwrap_or_else(default_gaia_cache);
    let path = cache.join(format!(
        "hipparcos-{:.4}{:+.4}-r{:.4}.csv",
        centre.0, centre.1, radius
    ));
    if let Ok(csv) = std::fs::read_to_string(&path)
        && let Ok(stars) = seiza_sources::parse_hipparcos(&csv)
    {
        return stars;
    }
    let fetched = seiza_sources::SourceDownloader::new()
        .map_err(anyhow::Error::from)
        .and_then(|downloader| {
            runtime()?
                .block_on(downloader.hipparcos_cone_csv(centre.0, centre.1, radius))
                .map_err(anyhow::Error::from)
        })
        .and_then(|csv| {
            let stars = seiza_sources::parse_hipparcos(&csv)?;
            let partial = path.with_extension("csv.partial");
            if std::fs::create_dir_all(&cache).is_ok() && std::fs::write(&partial, &csv).is_ok() {
                let _ = std::fs::rename(&partial, &path);
            }
            Ok(stars)
        });
    match fetched {
        Ok(stars) => stars,
        Err(error) => {
            eprintln!("warning: no Hipparcos distances for the brightest stars: {error}");
            Vec::new()
        }
    }
}

fn runtime() -> Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?)
}

/// Each detection with the distance of the Gaia star at its place, nearest
/// first and each Gaia star used once, and how many found a Gaia star.
/// Gaia's brightest stars have no parallax; a Hipparcos star at the same
/// place gives theirs.
fn match_distances(
    detections: &[PeakStar],
    gaia: &[GaiaDistance],
    hipparcos: &[HipparcosStar],
    wcs: &Wcs,
    scale_arcsec: f64,
) -> (Vec<Star>, usize) {
    let projected: Vec<(f64, f64)> = gaia
        .iter()
        .map(|star| {
            wcs.world_to_pixel(star.ra, star.dec)
                .unwrap_or((f64::NAN, f64::NAN))
        })
        .collect();
    let grid = Grid::new(&projected, 16.0);
    let mut used = vec![false; gaia.len()];
    let mut stars = Vec::with_capacity(detections.len());
    let mut found = 0;
    for detection in detections {
        // A saturated star's centroid can sit a few pixels off its catalog
        // place; a faint one's should not.
        let footprint = (detection.area as f64 / std::f64::consts::PI).sqrt();
        let reach = (2.0_f64).max(2.0 / scale_arcsec) + footprint * 0.25;
        let candidates = grid
            .near(detection.x, detection.y, reach)
            .filter(|index| !used[*index])
            .map(|index| {
                let (x, y) = projected[index];
                (index, (x - detection.x).hypot(y - detection.y))
            })
            .filter(|(_, distance)| *distance <= reach);
        // A large, saturated star is the brightest catalog star under it; a
        // fainter neighbour may lie nearer its centroid.
        let chosen = if footprint >= 4.0 {
            candidates.min_by(|a, b| gaia[a.0].g.total_cmp(&gaia[b.0].g))
        } else {
            candidates.min_by(|a, b| a.1.total_cmp(&b.1))
        };
        let gaia_distance = chosen.and_then(|(index, _)| {
            used[index] = true;
            found += 1;
            let star = &gaia[index];
            star.best_distance().or_else(|| {
                hipparcos
                    .iter()
                    .filter(|hip| separation((hip.ra, hip.dec), (star.ra, star.dec)) * 3600.0 < 5.0)
                    .find_map(HipparcosStar::distance)
            })
        });
        // Gaia lists no parallax, or no position at all, for some of the
        // brightest stars; Hipparcos measured them.
        let distance_pc = gaia_distance.or_else(|| {
            if footprint < 4.0 {
                return None;
            }
            hipparcos
                .iter()
                .filter_map(|hip| {
                    let (x, y) = wcs.world_to_pixel(hip.ra, hip.dec)?;
                    let offset = (x - detection.x).hypot(y - detection.y);
                    (offset <= reach).then_some((offset, hip))
                })
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .and_then(|(_, hip)| hip.distance())
        });
        stars.push(Star {
            x: detection.x,
            y: detection.y,
            distance_pc,
        });
    }
    (stars, found)
}

/// Points binned into square cells for neighbour lookups.
struct Grid {
    cell: f64,
    cells: std::collections::HashMap<(i64, i64), Vec<usize>>,
}

impl Grid {
    fn new(points: &[(f64, f64)], cell: f64) -> Self {
        let mut cells: std::collections::HashMap<(i64, i64), Vec<usize>> =
            std::collections::HashMap::new();
        for (index, (x, y)) in points.iter().enumerate() {
            if x.is_finite() && y.is_finite() {
                cells
                    .entry(((x / cell).floor() as i64, (y / cell).floor() as i64))
                    .or_default()
                    .push(index);
            }
        }
        Self { cell, cells }
    }

    fn near(&self, x: f64, y: f64, reach: f64) -> impl Iterator<Item = usize> + '_ {
        let span = (reach / self.cell).ceil() as i64;
        let (cx, cy) = (
            (x / self.cell).floor() as i64,
            (y / self.cell).floor() as i64,
        );
        (cy - span..=cy + span)
            .flat_map(move |row| (cx - span..=cx + span).map(move |column| (column, row)))
            .filter_map(|key| self.cells.get(&key))
            .flatten()
            .copied()
    }
}

/// Write `scene`'s layers as PNG files, for checking how the stars were cut.
fn write_layers(directory: &Path, scene: &Scene) -> Result<()> {
    std::fs::create_dir_all(directory)
        .with_context(|| format!("failed to create {}", directory.display()))?;
    let save = |name: &str, image: &LightImage| -> Result<()> {
        let path = directory.join(name);
        image
            .to_display_rgb8()
            .save(&path)
            .with_context(|| format!("failed to write {}", path.display()))?;
        println!("wrote {}", path.display());
        Ok(())
    };
    save("background.png", scene.background.base())?;
    save("leftover.png", scene.leftover.base())?;
    // Every sprite at its place, tinted by distance: red nearer than the
    // background, green near it, blue beyond.
    let mut sprites = LightImage::new(scene.width(), scene.height());
    for sprite in &scene.sprites {
        let ratio = sprite.distance_pc / scene.background_distance_pc;
        let tint = if ratio < 0.8 {
            [1.0, 0.25, 0.25]
        } else if ratio <= 1.25 {
            [0.25, 1.0, 0.25]
        } else {
            [0.35, 0.5, 1.0]
        };
        for y in 0..sprite.image.height {
            for x in 0..sprite.image.width {
                let light = sprite.image.pixels[y * sprite.image.width + x];
                let level = light[0].max(light[1]).max(light[2]);
                let pixel = &mut sprites.pixels[(sprite.top + y) * sprites.width + sprite.left + x];
                for channel in 0..3 {
                    pixel[channel] += level * tint[channel];
                }
            }
        }
    }
    save("sprites.png", &sprites)?;
    // The dust's transmission, white where clear.
    if let Some(dust) = &scene.dust {
        let (cells, columns) = dust.cells();
        let rows = cells.len() / columns;
        let path = directory.join("dust.png");
        image::GrayImage::from_fn(columns as u32, rows as u32, |x, y| {
            image::Luma([(255.0 * cells[y as usize * columns + x as usize]).round() as u8])
        })
        .save(&path)
        .with_context(|| format!("failed to write {}", path.display()))?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

/// The catalogued object at the focus point, its distance and how that
/// distance was found, or `None` without an object catalog, a distance file
/// or an object there.
fn target_distance(
    args: &ParallaxVideoArgs,
    wcs: &Wcs,
    (width, height): (usize, usize),
    focus: (f64, f64),
) -> Result<Option<(String, f64, &'static str)>> {
    let Ok(objects_path) = seiza::data_paths::objects(args.objects.as_deref()) else {
        return Ok(None);
    };
    let Some(distances_path) =
        seiza::data_paths::object_distances_beside(args.distances.as_deref(), &objects_path)?
    else {
        return Ok(None);
    };
    let catalog = seiza::objects::ObjectCatalog::open(&objects_path)
        .with_context(|| format!("failed to open {}", objects_path.display()))?;
    let distances = seiza::catalog::ObjectDistances::open(&distances_path, &catalog)
        .with_context(|| format!("failed to open {}", distances_path.display()))?;
    // Search a twentieth of the image's diagonal past the nearest edge.
    let reach = (width as f64).hypot(height as f64) / 20.0;
    let Some(found) = distances
        .object_at_pixel(&catalog, wcs, (width as u32, height as u32), focus, reach)
        .map_err(|error| anyhow::anyhow!("{error}"))?
    else {
        return Ok(None);
    };
    let object = &found.placed.object;
    let name = if object.common_name.is_empty() {
        object.name.clone()
    } else {
        format!("{} ({})", object.name, object.common_name)
    };
    Ok(found
        .distance()
        .map(|distance| (name, distance.distance_pc, distance.basis.as_str())))
}

/// Where lifted galaxies fly: so far beyond the stars that they hold still.
const GALAXY_DISTANCE_PC: f64 = 1e8;

/// Catalogued galaxies in the image large enough to see, largest first,
/// leaving out one at the focus point, which is the target, and each only
/// once: catalogs list some galaxies twice, a little apart.
fn galaxies_in_image(
    args: &ParallaxVideoArgs,
    wcs: &Wcs,
    (width, height): (usize, usize),
    focus: (f64, f64),
) -> Result<Vec<(String, Extent)>> {
    let Ok(objects_path) = seiza::data_paths::objects(args.objects.as_deref()) else {
        return Ok(Vec::new());
    };
    let catalog = seiza::objects::ObjectCatalog::open(&objects_path)
        .with_context(|| format!("failed to open {}", objects_path.display()))?;
    let placed = catalog
        .objects_in_footprint(wcs, (width as u32, height as u32))
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    let mut kept: Vec<(String, Extent)> = Vec::new();
    for object in placed {
        if object.object.kind != seiza::objects::ObjectKind::Galaxy || object.semi_major_px < 4.0 {
            continue;
        }
        let (semi_major, semi_minor, angle) = crate::object_outline(&object);
        let extent = Extent {
            x: object.x,
            y: object.y,
            semi_major,
            semi_minor: semi_minor.max(1.0),
            angle: angle.to_radians(),
        };
        if extent.reach(focus.0, focus.1) <= 2.0
            || kept
                .iter()
                .any(|(_, other)| other.reach(extent.x, extent.y) <= 1.5)
        {
            continue;
        }
        kept.push((object.object.name.clone(), extent));
        if kept.len() == 200 {
            break;
        }
    }
    Ok(kept)
}

/// The median distance of the stars within a fifth of the image's
/// diagonal of the focus point.
fn median_distance_near(
    stars: &[Star],
    focus: (f64, f64),
    width: usize,
    height: usize,
) -> Option<f64> {
    let reach = (width as f64).hypot(height as f64) / 5.0;
    median(
        stars
            .iter()
            .filter(|star| (star.x - focus.0).hypot(star.y - focus.1) <= reach)
            .filter_map(|star| star.distance_pc),
    )
}

fn median(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut values: Vec<f64> = values.collect();
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[values.len() / 2])
}

fn open_sink(args: &ParallaxVideoArgs, settings: VideoSettings) -> Result<Box<dyn FrameSink>> {
    let ffmpeg = || -> Result<Box<dyn FrameSink>> {
        Ok(Box::new(FfmpegSink::start(
            &args.ffmpeg,
            &args.output,
            settings,
        )?))
    };
    match args.encoder {
        EncoderArg::Png => Ok(Box::new(PngSequence::new(&args.output, settings)?)),
        EncoderArg::Ffmpeg => ffmpeg(),
        EncoderArg::Openh264 => openh264(&args.output, settings),
        EncoderArg::Auto => {
            if FfmpegSink::available(&args.ffmpeg) {
                ffmpeg()
            } else {
                openh264(&args.output, settings)
                    .context("ffmpeg did not run; install it, pass --ffmpeg, or use --encoder png")
            }
        }
    }
}

#[cfg(feature = "openh264")]
fn openh264(output: &Path, settings: VideoSettings) -> Result<Box<dyn FrameSink>> {
    Ok(Box::new(seiza_parallax::OpenH264Sink::start(
        output, settings,
    )?))
}

#[cfg(not(feature = "openh264"))]
fn openh264(_output: &Path, _settings: VideoSettings) -> Result<Box<dyn FrameSink>> {
    bail!("this build of seiza has no built-in H.264 encoder (the openh264 feature)")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_and_points_parse() {
        assert_eq!(parse_size("1920x1080"), Ok((1920, 1080)));
        assert!(parse_size("1921x1080").is_err());
        assert!(parse_size("8x8").is_err());
        assert_eq!(parse_point("10.5, 20"), Ok((10.5, 20.0)));
        assert!(parse_point("10").is_err());
    }

    #[test]
    fn cones_cover_the_image_and_merged_rows_appear_once() {
        let wcs =
            Wcs::from_center_scale_rotation((56.75, 24.12), (3124.0, 2088.0), 4.47, 0.0, false);
        let cones = cones(&wcs, 6248, 4176);
        assert_eq!(cones.len(), 20);
        // Every corner of the image lies inside some cone.
        for (x, y) in [
            (0.0, 0.0),
            (6248.0, 0.0),
            (0.0, 4176.0),
            (6248.0, 4176.0),
            (3124.0, 2088.0),
        ] {
            let point = wcs.pixel_to_world(x, y);
            assert!(
                cones
                    .iter()
                    .any(|&(ra, dec, radius)| separation((ra, dec), point) <= radius)
            );
        }
        assert!(cones.iter().all(|cone| cone.2 < 1.3));
    }

    #[test]
    fn detections_take_the_nearest_unused_gaia_star() {
        let wcs = Wcs::from_center_scale_rotation((56.75, 24.12), (500.0, 500.0), 2.0, 0.0, false);
        let place = |dx: f64, dy: f64| wcs.pixel_to_world(500.0 + dx, 500.0 + dy);
        let gaia_star = |(ra, dec): (f64, f64), distance: Option<f64>| GaiaDistance {
            ra,
            dec,
            pmra: None,
            pmdec: None,
            g: 10.0,
            bp_rp: None,
            parallax: None,
            parallax_error: None,
            distance,
            distance_low: None,
            distance_high: None,
        };
        let gaia = [
            gaia_star(place(0.5, 0.0), Some(136.0)),
            gaia_star(place(40.0, 0.0), None),
        ];
        let hipparcos = [HipparcosStar {
            hip: 1,
            ra: place(40.0, 0.0).0,
            dec: place(40.0, 0.0).1,
            parallax: Some(10.0),
            parallax_error: Some(0.5),
            hp_mag: Some(3.0),
            pmra: None,
            pmdec: None,
        }];
        let detection = |x: f64, y: f64| PeakStar {
            x: 500.0 + x,
            y: 500.0 + y,
            flux: 100.0,
            area: 9,
        };
        let detections = [
            detection(0.0, 0.0),
            detection(40.3, 0.2),
            detection(200.0, 0.0),
        ];
        let (stars, found) = match_distances(&detections, &gaia, &hipparcos, &wcs, 2.0);
        assert_eq!(found, 2);
        assert_eq!(stars[0].distance_pc, Some(136.0));
        let hip = stars[1].distance_pc.unwrap();
        assert!((hip - 100.0).abs() < 1e-9, "{hip}");
        assert_eq!(stars[2].distance_pc, None, "no Gaia star there");
        assert_eq!(
            median_distance_near(&stars, (500.0, 500.0), 1000, 1000),
            Some(136.0)
        );
    }
}
