//! `seiza parallax-video`: fly toward a point of a stretched image, with its
//! stars at their Gaia distances.
//!
//! The image is split into a starless image and its stars (by
//! StarXTerminator, or given as two files), plate-solved, and its stars
//! matched to Gaia DR3 for their Bailer-Jones distances, with Hipparcos for
//! the brightest stars Gaia has no parallax for. [`seiza_parallax::Parallax`]
//! does that and renders the frames, and an encoder writes them.

use anyhow::{Context, Result, bail};
use clap::{Args, ValueEnum};
use image::{Rgb, Rgb32FImage};
use seiza::{DetectConfig, Wcs};
use seiza_parallax::{
    CustomLabel, Easing, Event, FfmpegSink, FrameSink, LightImage, Parallax, ParallaxOptions,
    PngSequence, Quality, Scene, SceneOptions, SmallStars, Start, VideoOptions, VideoSettings,
};
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
    #[arg(short, long, required_unless_present = "plan_tour")]
    output: Option<PathBuf>,
    /// Point to fly toward, as image pixels `x,y` (default: the image
    /// centre)
    #[arg(long, value_parser = parse_point)]
    focus: Option<(f64, f64)>,
    /// Distance to the nebula or galaxy behind the stars, in parsecs
    /// (default: the distance of the catalogued object at the focus point,
    /// else the median Gaia distance of the stars near it)
    #[arg(long)]
    distance: Option<f64>,
    /// The image point, `x,y` in pixels, whose catalogued object or nearby
    /// stars give the nebula's distance (default: --focus, a tour's first
    /// stop flown in toward, or an automatic tour's best target). It sets
    /// the scene's depth; the camera's destination does not
    #[arg(long, value_parser = parse_point)]
    distance_focus: Option<(f64, f64)>,
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
    /// The first frame's turn about its centre, degrees anticlockwise. The
    /// frame turns from this to --rotate-end as the camera moves, turning
    /// every depth alike; a turned frame needs more of the image, so the
    /// first frame zooms in as far as it must
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    rotate: f64,
    /// The last frame's turn about its centre, degrees anticlockwise
    /// (default: --rotate, a steady tilt)
    #[arg(long, allow_hyphen_values = true)]
    rotate_end: Option<f64>,
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
    /// A stop on a tour, given in order, two or more: `X,Y` or `whole`
    /// (the image's centre), then any of `dolly=`, `zoom=`, `rotate=`
    /// (degrees), `pan=`, `travel=` and `hold=` (seconds), `spin=`
    /// (degrees turned while holding), `push=` (the share of the remaining
    /// way flown in while holding) and `title="..."`, e.g. `--stop
    /// "2700,3400 dolly=0.85 rotate=-20 pan=0.25 travel=6 hold=1.5"`. The
    /// camera glides on slowly through the stops it holds at; the first
    /// is the opening view. A tour sets the video's length and replaces
    /// --focus's single move
    #[arg(long = "stop", value_parser = seiza_parallax::parse_stop, allow_hyphen_values = true)]
    stops: Vec<seiza_parallax::TourStop>,
    /// Tour the catalogued objects in the field: every one worth a visit
    /// (up to twelve), or the N most worth it, visited in a short round
    /// from the whole image and back, each framed to its size with a
    /// gentle turn and pan
    #[arg(
        long,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "0",
        conflicts_with = "stops"
    )]
    auto_tour: Option<usize>,
    /// Plan a tour as --auto-tour would (taking its count, --tour-hold and
    /// --tour-motion), write it to this file for editing, and stop. Each
    /// line is a stop as --stop takes it, named in a comment; drop, move or
    /// change lines, then pass the file with --tour-file
    #[arg(
        long,
        conflicts_with_all = ["stops", "tour_file", "output", "tour_loop", "tour_titles", "debug_layers"]
    )]
    plan_tour: Option<PathBuf>,
    /// Take the tour's stops from this file, one a line as --stop takes
    /// them, with `#` starting a comment, and a `focus X,Y` line, if any,
    /// saying where the nebula's distance is taken (as --distance-focus
    /// does)
    #[arg(long, conflicts_with_all = ["stops", "auto_tour"])]
    tour_file: Option<PathBuf>,
    /// Seconds an automatic tour stays at each target; the three most
    /// prominent stay half as long again
    #[arg(long, default_value_t = 1.5)]
    tour_hold: f64,
    /// How much an automatic tour turns and pans: 0 for none, 1 for the
    /// gentle default, 2 for twice that
    #[arg(long, default_value_t = 1.0)]
    tour_motion: f64,
    /// How fast a tour moves on through a stop it holds at, as a share of
    /// its pace between stops, so it never quite stops; 0 comes to rest
    #[arg(long, default_value_t = 0.2)]
    tour_glide: f64,
    /// Show each tour stop's title, low in the frame, while the camera
    /// drifts through the stop, fading in and out: a planned tour titles
    /// each target with its name, and a stop's title= sets or changes it
    #[arg(long)]
    tour_titles: bool,
    /// End the tour where it began so the video loops: the last frame leads
    /// into the first. A last stop at the first's view is added unless the
    /// tour already ends there
    #[arg(long = "loop")]
    tour_loop: bool,
    /// Frames per second
    #[arg(long, default_value_t = 30)]
    fps: u32,
    /// Frame size: `720p`, `1080p`, `1440p` or `4k`, any of them with
    /// `-portrait` for vertical video (`1080p-portrait` is 1080x1920), or
    /// `WIDTHxHEIGHT` in even numbers
    #[arg(long, default_value = "1080p", value_parser = seiza_parallax::parse_frame_size)]
    size: (usize, usize),
    /// How the camera speeds up and slows down
    #[arg(long, value_enum, default_value_t = EasingArg::InOut)]
    easing: EasingArg,
    /// How carefully frames are drawn: `high` draws each at twice the size
    /// and averages it down, blending levels of detail so fine detail
    /// neither shimmers nor steps in sharpness, for about four times the
    /// rendering time
    #[arg(long, value_enum, default_value_t = QualityArg::Standard)]
    quality: QualityArg,
    /// The most a star's growth counts as it swells: growth is how many
    /// times nearer the camera has come, and a star swells as its square
    /// root, so the default 4 at most doubles it (at least 1)
    #[arg(long, default_value_t = 4.0)]
    growth_limit: f64,
    /// A star past this growth fades out as the camera flies by it, gone at
    /// twice it
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
    /// The codec ffmpeg writes: h264 (libx264, else libopenh264) or hevc
    /// (libx265, about a fifth smaller for the same look, and slower)
    #[arg(long, default_value = "h264", value_parser = seiza_parallax::Codec::parse)]
    codec: seiza_parallax::Codec,
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
    #[arg(long, conflicts_with = "starless")]
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
    /// Label the catalogued objects in the field as Seiza's image overlays
    /// do. Each label moves with the layer that shows its object and fades
    /// as it leaves the view; once the camera is inside an object, its name
    /// goes to a "Field within" caption
    #[arg(long)]
    overlay: bool,
    /// The share of the objects in view that --overlay labels, most
    /// prominent first, 0 to 1
    #[arg(long, default_value_t = seiza_parallax::overlay::DEFAULT_DENSITY)]
    overlay_density: f64,
    /// A label of your own, `X,Y:TEXT` in image pixels on the nebula's
    /// plane, or `X,Y,RADIUS:TEXT` to circle that many pixels about the
    /// point. Repeat for more
    #[arg(long = "label", value_parser = seiza_parallax::parse_label)]
    labels: Vec<CustomLabel>,
    /// Colour of your labels, `#RRGGBB`
    #[arg(long, default_value = "#f0f4f8", value_parser = seiza_parallax::parse_color)]
    label_color: Rgb<u8>,
    /// Write a line of text in the bottom-right corner of every frame
    /// (default text: "Rendered with seiza.fyi")
    #[arg(
        long,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = seiza_parallax::DEFAULT_WATERMARK
    )]
    watermark: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum QualityArg {
    Standard,
    High,
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

/// Open a stretched raster with values 0 to 1, in its EXIF orientation.
fn open_display(path: &Path) -> Result<Rgb32FImage> {
    Ok(seiza::raster::open_oriented(path)
        .with_context(|| format!("failed to open {}", path.display()))?
        .pixels
        .to_rgb32f())
}

pub(crate) fn run(args: ParallaxVideoArgs) -> Result<()> {
    if args.image.is_none() && args.starless.is_none() {
        bail!("give an image to split, or --starless and --stars");
    }
    if let Some(plan) = &args.plan_tour {
        return write_plan(&args, plan);
    }
    let file = match &args.tour_file {
        Some(path) => Some(read_tour(path)?),
        None => None,
    };
    let options = options(&args, file);
    options.check()?;
    // Everything that would refuse the video at the end is asked first: the
    // split, the solve and the stars' distances take minutes.
    check_output(&args)?;
    crate::interrupt::install();
    let stop = crate::interrupt::interrupted;
    let started = std::time::Instant::now();
    let (starless, stars, solve_path) = split(&args)?;
    let (wcs, solved) = solve(&args, &solve_path)?;
    if solved != starless.dimensions() {
        bail!(
            "{} is {}x{} but the split images are {}x{}: the plate solution must come from the \
             same pixels",
            solve_path.display(),
            solved.0,
            solved.1,
            starless.width(),
            starless.height()
        );
    }
    let parallax = Parallax::prepare(&starless, &stars, &wcs, &options, &mut print_event, &stop)?;
    if let Some(directory) = &args.debug_layers {
        write_layers(directory, parallax.scene())?;
    }
    let sink = open_sink(&args, parallax.video_settings())?;
    let rendering = std::time::Instant::now();
    parallax.render(sink, &mut print_event, &stop)?;
    println!(
        "wrote {} ({} frames in {:.1}s, {:.1}s in all)",
        output(&args).display(),
        parallax.frames(),
        rendering.elapsed().as_secs_f64(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Whether the video can be written where `-o` says, with the encoder
/// asked for, and does not name an input.
fn check_output(args: &ParallaxVideoArgs) -> Result<()> {
    let output = output(args);
    let same = |input: &Path| {
        let canonical = |path: &Path| std::fs::canonicalize(path).ok();
        canonical(input).is_some_and(|input| Some(input) == canonical(output))
    };
    for input in [&args.image, &args.starless, &args.stars]
        .into_iter()
        .flatten()
    {
        if same(input) {
            bail!(
                "-o {} is an input; the video would overwrite it",
                output.display()
            );
        }
    }
    let hevc = args.codec == seiza_parallax::Codec::Hevc;
    match args.encoder {
        EncoderArg::Png => {
            if output.is_file() {
                bail!("{} is a file; PNG frames go in a folder", output.display());
            }
        }
        EncoderArg::Ffmpeg => FfmpegSink::check(&args.ffmpeg, output, args.codec)?,
        EncoderArg::Openh264 if hevc => {
            bail!("the built-in encoder writes H.264 only; use ffmpeg for HEVC")
        }
        EncoderArg::Openh264 => {}
        EncoderArg::Auto if FfmpegSink::available(&args.ffmpeg) => {
            FfmpegSink::check(&args.ffmpeg, output, args.codec)?
        }
        EncoderArg::Auto if hevc => {
            bail!("ffmpeg did not run, and HEVC needs it; install it or pass --ffmpeg")
        }
        EncoderArg::Auto if !cfg!(feature = "openh264") => {
            bail!("ffmpeg did not run; install it, pass --ffmpeg, or use --encoder png")
        }
        EncoderArg::Auto => {}
    }
    Ok(())
}

/// Tell the user what the video's preparation and rendering are doing:
/// notes as they come, and every tenth of the frames.
fn print_event(event: Event) {
    match event {
        Event::Note(note) => println!("{note}"),
        Event::Warning(warning) => eprintln!("warning: {warning}"),
        Event::Frame { done, total } => {
            if done % (total / 10).max(1) == 0 || done == total {
                println!("rendered {done}/{total} frames");
            }
        }
    }
}

/// The video's options from the command line.
/// The automatic tour the flags ask for.
fn auto_tour(args: &ParallaxVideoArgs) -> seiza_parallax::AutoTour {
    seiza_parallax::AutoTour {
        targets: args.auto_tour.filter(|&targets| targets > 0),
        hold: args.tour_hold,
        motion: args.tour_motion,
    }
}

/// Plan a tour of the catalogued objects in the image and write it to
/// `path` for editing.
fn write_plan(args: &ParallaxVideoArgs, path: &Path) -> Result<()> {
    let solve_path = args
        .image
        .clone()
        .or_else(|| args.stars.clone())
        .expect("checked in run");
    let (wcs, dimensions) = solve(args, &solve_path)?;
    let plan = seiza_parallax::plan_tour(
        &wcs,
        dimensions,
        args.objects.as_deref(),
        args.size,
        &auto_tour(args),
    )?;
    let mut text = format!(
        "# A tour of {} planned by seiza parallax-video. Each line is a stop as\n\
         # --stop takes it; drop, move or change lines, then pass this file with\n\
         # --tour-file. A stop's title shows with --tour-titles. The nebula's\n\
         # distance is taken at the focus line.\n\
         focus {:.0},{:.0}  # {}\n",
        solve_path.display(),
        plan.focus.0,
        plan.focus.1,
        plan.focus_name
    );
    for planned in &plan.stops {
        let line = seiza_parallax::format_stop(&planned.stop);
        match &planned.name {
            Some(name) if planned.stop.title.as_deref() != Some(name) => {
                text.push_str(&format!("{line}  # {name}\n"))
            }
            Some(_) => text.push_str(&format!("{line}\n")),
            None if planned.stop.focus.is_some() => {
                text.push_str(&format!("{line}  # pulling back on the way\n"))
            }
            None => text.push_str(&format!("{line}\n")),
        }
    }
    std::fs::write(path, &text).with_context(|| format!("failed to write {}", path.display()))?;
    print!("{text}");
    println!("wrote {}", path.display());
    Ok(())
}

/// `line` up to a `#` outside double quotes.
fn uncommented(line: &str) -> &str {
    let mut quoted = false;
    let mut escaped = false;
    for (at, c) in line.char_indices() {
        match c {
            '\\' if quoted && !escaped => {
                escaped = true;
                continue;
            }
            '"' if !escaped => quoted = !quoted,
            '#' if !quoted => return &line[..at],
            _ => {}
        }
        escaped = false;
    }
    line
}

/// A tour file's stops, and its focus line if any.
type TourFile = (Vec<seiza_parallax::TourStop>, Option<(f64, f64)>);

/// The stops and focus of the tour file at `path`.
fn read_tour(path: &Path) -> Result<TourFile> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    parse_tour(&text).with_context(|| format!("in {}", path.display()))
}

/// A tour file's stops and focus line: a stop a line as --stop takes it,
/// `focus X,Y` at most once, `#` comments, and a byte-order mark, as some
/// Windows editors write, ignored.
fn parse_tour(text: &str) -> Result<TourFile> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let (mut stops, mut focus) = (Vec::new(), None);
    for (number, line) in text.lines().enumerate() {
        let line = uncommented(line).trim();
        if line.is_empty() {
            continue;
        }
        let where_ = || format!("line {}", number + 1);
        let mut words = line.splitn(2, char::is_whitespace);
        if words.next() == Some("focus") {
            if focus.is_some() {
                bail!("{}: a second focus line; keep one", where_());
            }
            let place = words.next().unwrap_or_default();
            focus = Some(
                parse_point(place.trim())
                    .map_err(anyhow::Error::msg)
                    .with_context(where_)?,
            );
        } else {
            stops.push(
                seiza_parallax::parse_stop(line)
                    .map_err(anyhow::Error::msg)
                    .with_context(where_)?,
            );
        }
    }
    if stops.len() < 2 {
        bail!(
            "a tour file needs two stops or more; it has {}",
            stops.len()
        );
    }
    Ok((stops, focus))
}

/// The video's options from the command line, with the tour file's stops
/// and focus if one was given.
fn options(args: &ParallaxVideoArgs, file: Option<TourFile>) -> ParallaxOptions {
    let (file_stops, file_focus) = file.unwrap_or_default();
    ParallaxOptions {
        scene: SceneOptions {
            distance_pc: args.distance,
            distance_focus: args.distance_focus.or(file_focus),
            unmatched_distance_pc: args.unmatched_distance,
            objects: args.objects.clone(),
            object_distances: args.distances.clone(),
            star_distances: args.star_distances.clone(),
            gaia_max_mag: args.gaia_max_mag,
            gaia_cache: args.gaia_cache.clone(),
            online: true,
            max_stars: args.max_stars,
            small_stars: match args.small_stars {
                SmallStarsArg::Drop => SmallStars::Drop,
                SmallStarsArg::Field => SmallStars::Field,
            },
            keep_galaxies: args.keep_galaxies,
            dust: !args.no_dust,
            dust_opacity: args.dust_opacity,
        },
        video: VideoOptions {
            focus: args.focus,
            start: match args.start {
                StartArg::Focus => Start::Focus,
                StartArg::Whole => Start::Whole,
            },
            dolly: args.dolly,
            truck: args.truck,
            truck_angle_deg: args.truck_angle,
            pan: args.pan,
            zoom: args.zoom,
            zoom_end: args.zoom_end,
            rotate_deg: (args.rotate, args.rotate_end.unwrap_or(args.rotate)),
            easing: match args.easing {
                EasingArg::Linear => Easing::Linear,
                EasingArg::InOut => Easing::InOut,
            },
            quality: match args.quality {
                QualityArg::Standard => Quality::Standard,
                QualityArg::High => Quality::High,
            },
            growth_limit: args.growth_limit,
            fade_from: args.fade_from,
            tour: if file_stops.is_empty() {
                args.stops.clone()
            } else {
                file_stops
            },
            auto_tour: args.auto_tour.map(|_| auto_tour(args)),
            tour_glide: args.tour_glide,
            tour_titles: args.tour_titles,
            tour_loop: args.tour_loop,
            size: args.size,
            seconds: args.seconds,
            fps: args.fps,
            overlay: args.overlay,
            overlay_density: args.overlay_density,
            labels: args.labels.clone(),
            label_color: args.label_color.0,
            watermark: args.watermark.clone(),
        },
    }
}

/// The starless and stars images, the path to plate-solve, and the
/// directory holding a split this run made, if any.
fn split(args: &ParallaxVideoArgs) -> Result<(Rgb32FImage, Rgb32FImage, PathBuf)> {
    if let (Some(starless), Some(stars)) = (&args.starless, &args.stars) {
        let solve_path = args.image.clone().unwrap_or_else(|| stars.clone());
        return Ok((open_display(starless)?, open_display(stars)?, solve_path));
    }
    let image_path = args.image.as_ref().expect("checked in run");
    let image = open_display(image_path)?;
    println!("splitting {} with StarXTerminator", image_path.display());
    let (starless, stars) = star_x_terminator(&image)?;
    if args.keep_split {
        let stem = output(args).file_stem().map_or_else(
            || "parallax".into(),
            |stem| stem.to_string_lossy().into_owned(),
        );
        let directory = output(args).parent().unwrap_or(Path::new("."));
        for (suffix, split) in [("starless", &starless), ("stars", &stars)] {
            let path = directory.join(format!("{stem}-{suffix}.png"));
            image::DynamicImage::ImageRgb32F(split.clone())
                .to_rgb16()
                .save(&path)
                .with_context(|| format!("failed to write {}", path.display()))?;
            println!("wrote {}", path.display());
        }
    }
    Ok((starless, stars, image_path.clone()))
}

/// Split a stretched image with StarXTerminator into its starless image
/// and an unscreened stars image.
fn star_x_terminator(image: &Rgb32FImage) -> Result<(Rgb32FImage, Rgb32FImage)> {
    use seiza_stacking::{LinearImage, RcAstroCli};
    let cli = match std::env::var_os("SEIZA_RC_ASTRO") {
        Some(path) if !path.is_empty() => RcAstroCli::with_executable(PathBuf::from(path)),
        _ => RcAstroCli::locate().context(
            "StarXTerminator's rc-astro CLI was not found on PATH (or set SEIZA_RC_ASTRO); \
             pass --starless and --stars instead",
        )?,
    };
    let linear = LinearImage::new(
        image.width() as usize,
        image.height() as usize,
        3,
        image.as_raw().clone(),
    )?;
    let mut last = -1;
    // Ctrl-C stops the run and lets its scratch files go, rather than
    // leaving hundreds of megabytes of FITS behind.
    let cancel = crate::interrupt::cancel_signal();
    let (starless, stars) = cli
        .split_stars(&linear, Some(&cancel), &mut |fraction| {
            let percent = (fraction * 100.0) as i32;
            if percent / 10 != last / 10 {
                println!("StarXTerminator {percent}%");
                last = percent;
            }
        })
        .context(
            "StarXTerminator could not split the image; pass --starless and --stars instead",
        )?;
    let to_display = |image: LinearImage| {
        let data = image
            .data
            .iter()
            .map(|value| value.clamp(0.0, 1.0))
            .collect();
        Rgb32FImage::from_raw(image.width as u32, image.height as u32, data)
            .expect("an RGB image of its own size")
    };
    Ok((to_display(starless), to_display(stars)))
}

/// The plate solution of the image at `path`, and its size.
fn solve(args: &ParallaxVideoArgs, path: &Path) -> Result<(Wcs, (u32, u32))> {
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
    Ok((solution.wcs, dims))
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

/// Where the video goes; clap asks for it unless only planning a tour.
fn output(args: &ParallaxVideoArgs) -> &Path {
    args.output
        .as_deref()
        .expect("required unless planning a tour")
}

fn open_sink(args: &ParallaxVideoArgs, settings: VideoSettings) -> Result<Box<dyn FrameSink>> {
    let ffmpeg = || -> Result<Box<dyn FrameSink>> {
        Ok(Box::new(FfmpegSink::start(
            &args.ffmpeg,
            output(args),
            settings,
            args.codec,
        )?))
    };
    match args.encoder {
        EncoderArg::Png => Ok(Box::new(PngSequence::new(output(args), settings)?)),
        EncoderArg::Ffmpeg => ffmpeg(),
        EncoderArg::Openh264 if args.codec == seiza_parallax::Codec::Hevc => {
            bail!("the built-in encoder writes H.264 only; use ffmpeg for HEVC")
        }
        EncoderArg::Openh264 => openh264(output(args), settings),
        EncoderArg::Auto => {
            if FfmpegSink::available(&args.ffmpeg) {
                ffmpeg()
            } else {
                openh264(output(args), settings)
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
    fn points_parse() {
        assert_eq!(parse_point("10.5, 20"), Ok((10.5, 20.0)));
        assert!(parse_point("10").is_err());
    }

    #[test]
    fn tour_files_read_as_editors_write_them() {
        // A byte-order mark, a tab after `focus`, and Windows line ends.
        let (stops, focus) =
            parse_tour("\u{feff}focus\t10,20\r\nwhole hold=1\r\n100,100 dolly=0.5\r\n").unwrap();
        assert_eq!((stops.len(), focus), (2, Some((10.0, 20.0))));
        // Two focus lines, or no stops, are refused rather than half read.
        assert!(parse_tour("focus 1,2\nfocus 3,4\nwhole\nwhole travel=1\n").is_err());
        assert!(parse_tour("focus 1,2\n# only a comment\n").is_err());
    }

    #[test]
    fn a_tour_line_ends_at_a_hash_outside_quotes() {
        assert_eq!(uncommented("whole hold=1  # the opening"), "whole hold=1  ");
        let quoted = r#"10,20 title="Sh2-170 #2 \"east\"" hold=1 # note"#;
        assert_eq!(
            uncommented(quoted),
            r#"10,20 title="Sh2-170 #2 \"east\"" hold=1 "#
        );
        let stop = seiza_parallax::parse_stop(uncommented(quoted)).unwrap();
        assert_eq!(stop.title.as_deref(), Some(r#"Sh2-170 #2 "east""#));
    }
}
