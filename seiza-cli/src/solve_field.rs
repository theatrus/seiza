//! Astrometry.net `solve-field` compatibility mode, so Siril (and anything
//! else that shells out to a local astrometry.net) can use seiza by pointing
//! its solver path at a copy of this binary named `solve-field`.
//!
//! The contract, as exercised by Siril's `local_asnet_platesolve`:
//! `solve-field --version` must print a single line; a solve invocation
//! passes detected stars as a FITS binary table (`.xyls` with float32
//! X/Y/FLUX/BACKGROUND columns and IMAGEW/IMAGEH keywords), an optional
//! scale window (`-u arcsecperpix -L low -H high`), an optional position
//! hint (`--ra --dec --radius`), and a SIP order (`-t N`, or `-T` for
//! linear). Success is the stdout line prefixed `Field center: (RA,Dec)`
//! plus a header-only FITS `.wcs` file next to the input; failure is a line
//! prefixed `Did not solve`. The caller reads only stdout and the `.wcs`
//! file — exit codes are ignored.

use anyhow::{Context, Result};
use seiza::FitsCardValue;
use std::path::{Path, PathBuf};

/// Single-line version answer. Siril treats multi-line output as a legacy
/// pre-0.88 astrometry.net; the string itself is only echoed into logs and
/// the `PLTSOLVD` comment.
const COMPAT_VERSION: &str = concat!("0.94-seiza-", env!("CARGO_PKG_VERSION"));

#[derive(Debug, Default)]
struct SolveFieldArgs {
    /// `-C`: abort when this file appears
    stop_file: Option<PathBuf>,
    /// `-L` / `-H`: pixel-scale window, arcseconds per pixel
    scale_low: Option<f64>,
    scale_high: Option<f64>,
    /// `--ra` / `--dec` / `--radius`: search hint, degrees
    ra: Option<f64>,
    dec: Option<f64>,
    radius_deg: Option<f64>,
    /// `-t`: SIP polynomial order; `-T` resets to linear
    sip_order: u8,
    /// Positional argument: the star table
    table: Option<PathBuf>,
    version: bool,
}

/// True when the binary is named like a shell. Windows Siril launches
/// astrometry.net through `<asnet_dir>/bin/bash -l -c ...`, so a copy of
/// seiza installed as `bin/bash.exe` receives those invocations and
/// interprets the two commands Siril issues — no cygwin required.
pub fn invoked_as_bash(program: &str) -> bool {
    binary_name_matches(program, "bash")
}

/// Compare the final path component (without `.exe`) case-insensitively,
/// accepting both separator styles regardless of host platform.
fn binary_name_matches(program: &str, expected: &str) -> bool {
    let name = program.rsplit(['/', '\\']).next().unwrap_or(program);
    let stem = name
        .strip_suffix(".exe")
        .or_else(|| name.strip_suffix(".EXE"))
        .unwrap_or(name);
    stem.eq_ignore_ascii_case(expected)
}

/// Handle a Siril-style `bash -l -c <command>` invocation: either the
/// version handshake (`solve-field --version`) or a generated `asnet.sh`
/// script containing `p="<table>"`, `c="<stopfile>"`, and one solve-field
/// command line.
pub fn run_as_bash(raw: &[String]) -> Result<()> {
    let mut command = None;
    let mut iter = raw.iter();
    while let Some(argument) = iter.next() {
        if argument == "-c" {
            command = iter.next().cloned();
        }
    }
    let Some(command) = command else {
        anyhow::bail!("bash compatibility mode understands only -l -c <command>");
    };
    let command = command.trim();
    if command.contains("solve-field") && command.contains("--version") {
        println!("{COMPAT_VERSION}");
        return Ok(());
    }
    let script = resolve_shell_path(command)?;
    let content = std::fs::read_to_string(&script)
        .with_context(|| format!("cannot read {}", script.display()))?;
    let argv = parse_asnet_script(&content)?;
    run(&argv)
}

/// Resolve a cygwin-style absolute path such as `/tmp/asnet.sh` against
/// the shell root: this binary is installed at `<root>/bin/bash`, so
/// `/tmp/asnet.sh` maps to `<root>/tmp/asnet.sh`. A literal path is a
/// fallback only, so host files cannot shadow the layout's script.
fn resolve_shell_path(path: &str) -> Result<PathBuf> {
    // The layout-root mapping comes first: in the shell Siril emulates,
    // `/tmp/...` always means `<root>/tmp/...`, and a stray host file at
    // the literal path must not shadow the layout's script.
    if let Ok(exe) = std::env::current_exe()
        && let Some(root) = exe.parent().and_then(Path::parent)
    {
        let mapped = root.join(path.trim_start_matches(['/', '\\']));
        if mapped.exists() {
            return Ok(mapped);
        }
    }
    let direct = PathBuf::from(path);
    if direct.exists() {
        return Ok(direct);
    }
    anyhow::bail!("cannot resolve shell path {path}")
}

/// Parse the deterministic script Siril writes: shell variable assignments
/// (`name="value"`) followed by one solve-field command line whose words
/// may reference the variables as `"$name"`.
fn parse_asnet_script(content: &str) -> Result<Vec<String>> {
    let mut variables: Vec<(String, String)> = Vec::new();
    let mut command = None;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((name, rest)) = line.split_once("=\"")
            && !name.is_empty()
            && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            && let Some(value) = rest.strip_suffix('"')
        {
            variables.push((name.to_string(), value.to_string()));
        } else if line.starts_with("solve-field") {
            command = Some(line.to_string());
        }
    }
    let command = command.context("no solve-field command line in script")?;
    let words = split_command_words(&command, &variables);
    if words.first().map(String::as_str) != Some("solve-field") {
        anyhow::bail!("unexpected script command {:?}", words.first());
    }
    Ok(words[1..].to_vec())
}

/// Split a command line into words, honoring double quotes and expanding
/// `$name` variable references. Backslashes are literal — the values are
/// Windows paths, not shell escapes.
fn split_command_words(command: &str, variables: &[(String, String)]) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut in_word = false;
    let mut in_quotes = false;
    let mut chars = command.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' => {
                in_quotes = !in_quotes;
                in_word = true;
            }
            c if c.is_whitespace() && !in_quotes => {
                if in_word {
                    words.push(std::mem::take(&mut current));
                    in_word = false;
                }
            }
            '$' => {
                in_word = true;
                let mut name = String::new();
                while chars
                    .peek()
                    .is_some_and(|c| c.is_ascii_alphanumeric() || *c == '_')
                {
                    name.push(chars.next().unwrap());
                }
                if let Some((_, value)) = variables.iter().find(|(n, _)| *n == name) {
                    current.push_str(value);
                }
            }
            c => {
                in_word = true;
                current.push(c);
            }
        }
    }
    if in_word {
        words.push(current);
    }
    words
}

/// True when the binary itself is named like astrometry.net's solver.
pub fn invoked_as_solve_field(program: &str) -> bool {
    binary_name_matches(program, "solve-field")
}

/// True when the raw arguments carry solve-field-specific markers. Siril
/// always passes `--crpix-center`; `--temp-axy` is equally distinctive.
pub fn looks_like_solve_field(args: &[String]) -> bool {
    args.iter()
        .any(|arg| arg == "--crpix-center" || arg == "--temp-axy")
}

/// Install the drop-in layout Siril expects: `<dir>/solve-field` for
/// Linux/macOS, and `<dir>/bin/bash` plus `<dir>/tmp/` for the Windows
/// launch path. Both are copies of the running binary, so one layout works
/// on every platform and survives either invocation style.
pub fn install_layout(dir: &Path) -> Result<()> {
    let exe = std::env::current_exe().context("cannot locate the running binary")?;
    let extension = if cfg!(windows) { ".exe" } else { "" };
    std::fs::create_dir_all(dir.join("bin"))
        .with_context(|| format!("cannot create {}", dir.join("bin").display()))?;
    std::fs::create_dir_all(dir.join("tmp"))
        .with_context(|| format!("cannot create {}", dir.join("tmp").display()))?;
    for target in [
        dir.join(format!("solve-field{extension}")),
        dir.join("bin").join(format!("bash{extension}")),
    ] {
        std::fs::copy(&exe, &target)
            .with_context(|| format!("cannot install {}", target.display()))?;
        println!("installed {}", target.display());
    }
    println!(
        "\nPoint Siril's astrometry.net directory preference at:\n  {}\n\
         Give seiza a star catalog (any one of):\n  \
         - run: seiza setup\n  \
         - set SEIZA_STAR_DATA to a stars-*.bin\n  \
         - drop a stars-*.bin next to the installed solve-field",
        dir.display()
    );
    Ok(())
}

/// Run solve-field mode over raw (post program-name) arguments.
pub fn run(raw: &[String]) -> Result<()> {
    let args = parse_args(raw);
    if args.version {
        println!("{COMPAT_VERSION}");
        return Ok(());
    }
    let Some(table) = args.table.clone() else {
        anyhow::bail!("solve-field mode requires an input star table (.xyls)");
    };

    // The caller judges success purely by stdout and the .wcs file, and a
    // real solve-field exits zero on an unsolved field: report failures on
    // stdout with the expected prefix instead of propagating them.
    match solve(&args, &table) {
        Ok((solution, dimensions)) => report_solution(&args, &table, &solution, dimensions),
        Err(error) => {
            let message = format!("{error:#}").replace(['\r', '\n'], " ");
            println!("Did not solve (seiza: {message}).");
            Ok(())
        }
    }
}

fn parse_args(raw: &[String]) -> SolveFieldArgs {
    let mut args = SolveFieldArgs::default();
    let mut iter = raw.iter().peekable();
    let parse = |value: Option<&String>| value.and_then(|value| value.parse::<f64>().ok());
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--version" => args.version = true,
            "-C" | "--cancel" => args.stop_file = iter.next().map(PathBuf::from),
            "-L" | "--scale-low" => args.scale_low = parse(iter.next()),
            "-H" | "--scale-high" => args.scale_high = parse(iter.next()),
            "--ra" => args.ra = parse(iter.next()),
            "--dec" => args.dec = parse(iter.next()),
            "--radius" => args.radius_deg = parse(iter.next()),
            "-t" | "--tweak-order" => {
                args.sip_order = iter
                    .next()
                    .and_then(|value| value.parse::<u8>().ok())
                    .map(|order| order.clamp(2, 5))
                    .unwrap_or(0)
            }
            "-T" | "--no-tweak" => args.sip_order = 0,
            // Flags taking a value that seiza accepts and ignores: output
            // suppression (Siril passes "none" for all of them), sort
            // column (always FLUX), units, time limit, depth, and config.
            "-N" | "-R" | "-M" | "-B" | "-U" | "-S" | "-s" | "-u" | "-l" | "-d" | "-o" | "-D"
            | "-b" | "--config" | "--depth" | "-E" | "-P" | "-i" | "-w" | "-e" => {
                iter.next();
            }
            // Valueless flags: plots, overwrite, crpix conventions, verbosity
            "--temp-axy" | "-p" | "--no-plots" | "-O" | "--overwrite" | "--crpix-center" | "-v"
            | "--verbose" | "--continue" | "-z" | "-2" => {}
            other => {
                if !other.starts_with('-') {
                    args.table = Some(PathBuf::from(other));
                }
            }
        }
    }
    args
}

fn stop_requested(args: &SolveFieldArgs) -> bool {
    args.stop_file
        .as_deref()
        .is_some_and(|stop_file| stop_file.exists())
}

fn solve(args: &SolveFieldArgs, table: &Path) -> Result<(seiza::solve::Solution, (u32, u32))> {
    let (table_stars, dimensions) = read_xyls(table)?;
    if table_stars.len() < 4 {
        anyhow::bail!("only {} stars in {}", table_stars.len(), table.display());
    }
    if stop_requested(args) {
        anyhow::bail!("cancelled before solving");
    }
    // The table's flux column may not rank stars photometrically (Siril
    // writes PSF amplitudes, which drift far from photometric order on
    // stretched data — see docs/design/rank-robust-matching.md). When the
    // source image sits next to the table, re-detect with seiza's own
    // photometric flux; the table's excellent positions calibrate the
    // orientation and frame offset.
    let stars = match redetect_from_source_image(table, &table_stars, dimensions) {
        Some(stars) => {
            println!(
                "seiza: re-measured {} stars from the source image (table flux may not be photometric)",
                stars.len()
            );
            stars
        }
        None => table_stars,
    };

    let star_data = crate::astap::resolve_star_data()?;
    let catalog = seiza::catalog::TileCatalog::open(&star_data)
        .with_context(|| format!("cannot open star catalog {}", star_data.display()))?;

    let scale = match (args.scale_low, args.scale_high) {
        (Some(low), Some(high)) if low > 0.0 && high >= low => Some((low, high)),
        _ => None,
    };

    // A position hint plus a scale window runs the hinted solver; anything
    // else runs the blind solver over the requested (or practical) scales.
    if let (Some(ra), Some(dec), Some((low, high))) = (args.ra, args.dec, scale) {
        let center_scale = (low + high) / 2.0;
        let hint = seiza::solve::SolveHint {
            center: (ra, dec),
            radius_deg: args.radius_deg.unwrap_or(1.0).clamp(0.25, 3.0),
            scale_arcsec_px: center_scale,
            scale_tolerance: ((high - low) / (high + low)).max(0.05),
            sip_order: args.sip_order,
        };
        if let Ok(solution) = seiza::solve::solve(&stars, &catalog, &hint, dimensions) {
            return Ok((solution, dimensions));
        }
        if stop_requested(args) {
            anyhow::bail!("cancelled");
        }
        // Stale hints fall back to a scale-bracketed blind search, exactly
        // like the ASTAP-compatible mode.
    }

    let (min_scale, max_scale) = scale.unwrap_or((0.1, 20.0));
    let mut params = seiza::blind::BlindParams {
        min_scale_arcsec_px: min_scale,
        max_scale_arcsec_px: max_scale,
        sip_order: args.sip_order,
        ..Default::default()
    };
    let index = if let Some(path) = crate::astap::resolve_blind_index()? {
        let index = seiza::blind::BlindIndex::open(&path)
            .map_err(anyhow::Error::from)
            .with_context(|| format!("cannot open blind index {}", path.display()))?;
        params.index_mag_limit = index.index_mag_limit();
        params.max_pattern_deg = index.max_pattern_deg();
        index
    } else {
        seiza::blind::BlindIndex::build(&catalog, &params)
    };
    if stop_requested(args) {
        anyhow::bail!("cancelled");
    }
    seiza::blind::solve_blind(&stars, &catalog, &index, &params, dimensions)
        .map(|solution| (solution, dimensions))
        .map_err(anyhow::Error::from)
}

fn report_solution(
    args: &SolveFieldArgs,
    table: &Path,
    solution: &seiza::solve::Solution,
    dimensions: (u32, u32),
) -> Result<()> {
    let wcs_path = table.with_extension("wcs");
    write_wcs_file(&wcs_path, &solution.wcs)
        .with_context(|| format!("cannot write {}", wcs_path.display()))?;

    let (width, height) = (dimensions.0 as f64, dimensions.1 as f64);
    let (ra, dec) = solution
        .wcs
        .pixel_to_world((width - 1.0) / 2.0, (height - 1.0) / 2.0);
    // Only the "Field center: (RA,Dec)" prefix is contractual; the other
    // lines mirror solve-field for humans reading the log.
    println!("Field: {}", table.display());
    println!("Field center: (RA,Dec) = ({ra:.6}, {dec:.6}) deg.");
    let scale = solution.wcs.scale_arcsec_per_px();
    println!(
        "Field size: {:.4} x {:.4} degrees",
        width * scale / 3600.0,
        height * scale / 3600.0
    );
    println!(
        "Field rotation angle: up is {:.3} degrees E of N",
        solution.wcs.cd[0][1]
            .atan2(-solution.wcs.cd[1][1])
            .to_degrees()
    );
    println!(
        "seiza: {} stars matched, RMS {:.3} arcsec{}",
        solution.matched_stars,
        solution.rms_arcsec,
        match (&solution.wcs.sip, args.sip_order) {
            (Some(sip), _) => format!(", SIP order {}", sip.order),
            (None, order) if order >= 2 => ", SIP not fitted (kept linear)".to_string(),
            _ => String::new(),
        }
    );
    Ok(())
}

/// Re-detect stars from the source image sitting next to the star table,
/// restoring photometric flux ordering. The table's positions calibrate
/// the coordinate frame: detection happens in loader orientation, so the
/// cross-match selects the orientation (Siril tables are bottom-up) and
/// measures the constant sub-pixel offset between the frames, and the
/// returned stars live in the exact frame the table — and therefore the
/// `.wcs` consumer — uses. Returns `None` when no matching image is found
/// or the cross-match cannot confirm the frames describe the same pixels.
fn redetect_from_source_image(
    table: &Path,
    table_stars: &[seiza::DetectedStar],
    dimensions: (u32, u32),
) -> Option<Vec<seiza::DetectedStar>> {
    let image = ["fit", "fits", "fts", "tif", "tiff", "png", "jpg", "jpeg"]
        .iter()
        .map(|extension| table.with_extension(extension))
        .find_map(|path| {
            path.exists()
                .then(|| crate::load_image(&path, seiza::DetectBackend::Auto).ok())
                .flatten()
        })?;
    if image.dimensions() != dimensions {
        return None; // cropped or downsampled selection; trust the table
    }
    let detected = image.detect_stars(&seiza::DetectConfig {
        max_stars: 600,
        ..Default::default()
    });
    if detected.len() < 20 {
        return None;
    }

    // Try both orientations; require a decisive majority of close matches.
    let height = dimensions.1 as f64;
    let candidates = [false, true].map(|flip| {
        let mapped: Vec<(f64, f64)> = detected
            .iter()
            .map(|star| (star.x, if flip { height - 1.0 - star.y } else { star.y }))
            .collect();
        let mut offsets = Vec::new();
        for &(x, y) in mapped.iter().take(200) {
            let nearest = table_stars
                .iter()
                .map(|table_star| {
                    let (dx, dy) = (table_star.x - x, table_star.y - y);
                    (dx.hypot(dy), dx, dy)
                })
                .min_by(|a, b| a.0.total_cmp(&b.0));
            if let Some((distance, dx, dy)) = nearest
                && distance < 2.0
            {
                offsets.push((dx, dy));
            }
        }
        (mapped, offsets)
    });
    let (mapped, offsets) = candidates
        .into_iter()
        .max_by_key(|(_, offsets)| offsets.len())?;
    // Majority of the *achievable* matches: the table may hold fewer stars
    // than we checked, and detection depth differences are not disagreement.
    let achievable = detected.len().min(200).min(table_stars.len());
    if offsets.len() * 2 < achievable {
        return None; // frames disagree; wrong file or transformed image
    }
    let median = |mut values: Vec<f64>| {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    };
    let dx = median(offsets.iter().map(|offset| offset.0).collect());
    let dy = median(offsets.iter().map(|offset| offset.1).collect());

    Some(
        detected
            .iter()
            .zip(mapped)
            .map(|(star, (x, y))| seiza::DetectedStar {
                x: x + dx,
                y: y + dy,
                ..*star
            })
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Minimal FITS I/O for the star-table input and header-only WCS output. The
// table is written by the caller with fixed float32 columns; this reader
// still locates columns by TTYPE and tolerates extras.

const FITS_BLOCK: usize = 2880;
const FITS_CARD: usize = 80;

fn read_xyls(path: &Path) -> Result<(Vec<seiza::DetectedStar>, (u32, u32))> {
    let bytes = std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    let mut offset = 0usize;

    // Primary HDU: header plus any (unexpected) data, both block-aligned.
    let primary = parse_header(&bytes, &mut offset)?;
    offset += data_bytes(&primary)?.next_multiple_of(FITS_BLOCK);

    // Table HDU.
    let header = parse_header(&bytes, &mut offset)?;
    let xtension = header_string(&header, "XTENSION").unwrap_or_default();
    if xtension != "BINTABLE" {
        anyhow::bail!("expected a BINTABLE extension, found {xtension:?}");
    }
    let row_bytes = header_integer(&header, "NAXIS1")? as usize;
    let rows = header_integer(&header, "NAXIS2")? as usize;
    let fields = header_integer(&header, "TFIELDS")? as usize;

    // Column offsets from the TFORM sequence; only scalar E/D columns occur.
    let mut columns = Vec::new();
    let mut column_offset = 0usize;
    for field in 1..=fields {
        let name = header_string(&header, &format!("TTYPE{field}")).unwrap_or_default();
        let form = header_string(&header, &format!("TFORM{field}"))
            .with_context(|| format!("missing TFORM{field}"))?;
        let (repeat, code) = split_tform(&form)?;
        let element = match code {
            'E' => 4,
            'D' => 8,
            'J' => 4,
            'K' => 8,
            'I' => 2,
            other => anyhow::bail!("unsupported column type {other:?} in {form}"),
        };
        columns.push((name, column_offset, code));
        column_offset += repeat * element;
    }
    if column_offset > row_bytes {
        anyhow::bail!("table columns exceed NAXIS1");
    }
    let data_start = offset;
    let data_len = row_bytes.checked_mul(rows).context("table size overflow")?;
    if data_start + data_len > bytes.len() {
        anyhow::bail!("truncated table data");
    }

    let column = |name: &str| {
        columns
            .iter()
            .find(|(candidate, _, _)| candidate == name)
            .map(|&(_, offset, code)| (offset, code))
            .with_context(|| format!("missing column {name}"))
    };
    let (x_offset, x_code) = column("X")?;
    let (y_offset, y_code) = column("Y")?;
    let flux = column("FLUX").ok();

    let read_value = |row_start: usize, offset: usize, code: char| -> f64 {
        let start = row_start + offset;
        match code {
            'E' => f32::from_be_bytes(bytes[start..start + 4].try_into().unwrap()) as f64,
            'D' => f64::from_be_bytes(bytes[start..start + 8].try_into().unwrap()),
            'I' => i16::from_be_bytes(bytes[start..start + 2].try_into().unwrap()) as f64,
            'J' => i32::from_be_bytes(bytes[start..start + 4].try_into().unwrap()) as f64,
            'K' => i64::from_be_bytes(bytes[start..start + 8].try_into().unwrap()) as f64,
            _ => 0.0,
        }
    };

    let mut stars = Vec::with_capacity(rows);
    for row in 0..rows {
        let row_start = data_start + row * row_bytes;
        // FITS convention: the center of the first pixel is (1, 1); seiza
        // is 0-indexed. Whatever half-pixel convention the caller used,
        // the emitted CRPIX describes the same frame the stars came in.
        let x = read_value(row_start, x_offset, x_code) - 1.0;
        let y = read_value(row_start, y_offset, y_code) - 1.0;
        let flux = flux
            .map(|(offset, code)| read_value(row_start, offset, code))
            .unwrap_or(1.0)
            .max(f64::MIN_POSITIVE);
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        stars.push(seiza::DetectedStar {
            x,
            y,
            flux,
            peak: flux as f32,
            area: 1,
        });
    }
    stars.sort_by(|a, b| b.flux.total_cmp(&a.flux));

    // IMAGEW/IMAGEH may live in either header; fall back to the star extent.
    let dimension = |key: &str| {
        header_integer(&header, key)
            .or_else(|_| header_integer(&primary, key))
            .ok()
            .filter(|&value| value > 0)
            .map(|value| value as u32)
    };
    let width = dimension("IMAGEW");
    let height = dimension("IMAGEH");
    let (width, height) = match (width, height) {
        (Some(width), Some(height)) => (width, height),
        _ => {
            let max_x = stars.iter().fold(0.0f64, |acc, s| acc.max(s.x));
            let max_y = stars.iter().fold(0.0f64, |acc, s| acc.max(s.y));
            ((max_x.ceil() as u32).max(2), (max_y.ceil() as u32).max(2))
        }
    };
    Ok((stars, (width, height)))
}

fn split_tform(form: &str) -> Result<(usize, char)> {
    let split = form.find(|c: char| c.is_ascii_alphabetic()).unwrap_or(0);
    let repeat = if split == 0 {
        1
    } else {
        form[..split].parse::<usize>().unwrap_or(1)
    };
    let code = form[split..]
        .chars()
        .next()
        .with_context(|| format!("empty TFORM {form:?}"))?;
    Ok((repeat, code))
}

/// Parse one FITS header (through its END card), advancing `offset` past the
/// block-aligned header. Returns the raw cards.
fn parse_header(bytes: &[u8], offset: &mut usize) -> Result<Vec<(String, String)>> {
    let mut cards = Vec::new();
    loop {
        let block = bytes
            .get(*offset..*offset + FITS_BLOCK)
            .context("truncated FITS header")?;
        *offset += FITS_BLOCK;
        for card in block.chunks_exact(FITS_CARD) {
            let card = std::str::from_utf8(card).context("non-ASCII FITS header")?;
            let keyword = card[..8].trim().to_string();
            if keyword == "END" {
                return Ok(cards);
            }
            if card.as_bytes().get(8) == Some(&b'=') {
                let value = card[9..]
                    .split('/')
                    .next()
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                cards.push((keyword, value));
            }
        }
    }
}

fn header_lookup<'a>(cards: &'a [(String, String)], keyword: &str) -> Option<&'a str> {
    cards
        .iter()
        .find(|(candidate, _)| candidate == keyword)
        .map(|(_, value)| value.as_str())
}

fn header_string(cards: &[(String, String)], keyword: &str) -> Option<String> {
    header_lookup(cards, keyword).map(|value| value.trim().trim_matches('\'').trim().to_string())
}

fn header_integer(cards: &[(String, String)], keyword: &str) -> Result<i64> {
    header_lookup(cards, keyword)
        .and_then(|value| value.trim().parse::<i64>().ok())
        .with_context(|| format!("missing or invalid {keyword}"))
}

fn data_bytes(cards: &[(String, String)]) -> Result<usize> {
    let naxis = header_integer(cards, "NAXIS").unwrap_or(0);
    if naxis == 0 {
        return Ok(0);
    }
    let mut total = header_integer(cards, "BITPIX")?.unsigned_abs() as usize / 8;
    for axis in 1..=naxis {
        total = total
            .checked_mul(header_integer(cards, &format!("NAXIS{axis}"))? as usize)
            .context("FITS data size overflow")?;
    }
    Ok(total)
}

/// Write the header-only FITS file astrometry.net calls `.wcs`: the WCS
/// keywords of the solution and no data unit. wcslib consumers (Siril,
/// astropy) read the solution, including SIP, straight from the header.
fn write_wcs_file(path: &Path, wcs: &seiza::Wcs) -> Result<()> {
    let mut header = String::new();
    let mut push = |keyword: &str, value: String| {
        header.push_str(&format!("{keyword:<8}= {value:>20}"));
        header.push_str(&" ".repeat(FITS_CARD - 30));
    };
    push("SIMPLE", "T".into());
    push("BITPIX", "8".into());
    push("NAXIS", "0".into());
    for (keyword, value) in wcs.fits_header_cards() {
        let formatted = match value {
            FitsCardValue::Text(text) => format!("'{text}'"),
            FitsCardValue::Integer(value) => value.to_string(),
            FitsCardValue::Number(value) => format!("{value:.13E}"),
        };
        push(&keyword, formatted);
    }
    header.push_str(&format!("{:<80}", "END"));
    let padded = header.len().next_multiple_of(FITS_BLOCK);
    header.push_str(&" ".repeat(padded - header.len()));
    std::fs::write(path, header.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(text: &str) -> Vec<u8> {
        let mut bytes = text.as_bytes().to_vec();
        bytes.resize(FITS_CARD, b' ');
        bytes
    }

    fn synthetic_xyls(stars: &[(f32, f32, f32)], width: i32, height: i32) -> Vec<u8> {
        let mut bytes = Vec::new();
        for text in [
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
        ] {
            bytes.extend(card(text));
        }
        bytes.extend(card("END"));
        bytes.resize(bytes.len().next_multiple_of(FITS_BLOCK), b' ');

        let header = [
            "XTENSION= 'BINTABLE'".to_string(),
            "BITPIX  =                    8".to_string(),
            "NAXIS   =                    2".to_string(),
            "NAXIS1  =                   16".to_string(),
            format!("NAXIS2  = {:>20}", stars.len()),
            "PCOUNT  =                    0".to_string(),
            "GCOUNT  =                    1".to_string(),
            "TFIELDS =                    4".to_string(),
            "TTYPE1  = 'X'".to_string(),
            "TFORM1  = '1E'".to_string(),
            "TTYPE2  = 'Y'".to_string(),
            "TFORM2  = '1E'".to_string(),
            "TTYPE3  = 'FLUX'".to_string(),
            "TFORM3  = '1E'".to_string(),
            "TTYPE4  = 'BACKGROUND'".to_string(),
            "TFORM4  = '1E'".to_string(),
            format!("IMAGEW  = {width:>20}"),
            format!("IMAGEH  = {height:>20}"),
        ];
        for text in &header {
            bytes.extend(card(text));
        }
        bytes.extend(card("END"));
        bytes.resize(bytes.len().next_multiple_of(FITS_BLOCK), b' ');

        for &(x, y, flux) in stars {
            bytes.extend(x.to_be_bytes());
            bytes.extend(y.to_be_bytes());
            bytes.extend(flux.to_be_bytes());
            bytes.extend(0.0f32.to_be_bytes());
        }
        bytes.resize(bytes.len().next_multiple_of(FITS_BLOCK), 0);
        bytes
    }

    #[test]
    fn parses_a_siril_style_xyls_table() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image.xyls");
        std::fs::write(
            &path,
            synthetic_xyls(&[(101.5, 201.5, 50.0), (301.25, 41.75, 500.0)], 4000, 3000),
        )
        .unwrap();
        let (stars, dimensions) = read_xyls(&path).unwrap();
        assert_eq!(dimensions, (4000, 3000));
        assert_eq!(stars.len(), 2);
        // Brightest first, FITS 1-based converted to 0-based
        assert!((stars[0].x - 300.25).abs() < 1e-6);
        assert!((stars[0].y - 40.75).abs() < 1e-6);
        assert!((stars[1].x - 100.5).abs() < 1e-6);
        assert!(stars[0].flux > stars[1].flux);
    }

    #[test]
    fn parses_siril_argument_shape() {
        let raw: Vec<String> = [
            "-C",
            "stop",
            "--temp-axy",
            "-p",
            "-O",
            "-N",
            "none",
            "-R",
            "none",
            "-M",
            "none",
            "-B",
            "none",
            "-U",
            "none",
            "-S",
            "none",
            "--crpix-center",
            "-s",
            "FLUX",
            "-u",
            "arcsecperpix",
            "-L",
            "1.827",
            "-H",
            "2.233",
            "-l",
            "60",
            "-t",
            "3",
            "--ra",
            "150.5",
            "--dec",
            "35.25",
            "--radius",
            "10.0",
            "/tmp/image.xyls",
        ]
        .iter()
        .map(ToString::to_string)
        .collect();
        assert!(looks_like_solve_field(&raw));
        let args = parse_args(&raw);
        assert_eq!(args.stop_file.as_deref(), Some(Path::new("stop")));
        assert_eq!(args.scale_low, Some(1.827));
        assert_eq!(args.scale_high, Some(2.233));
        assert_eq!(args.ra, Some(150.5));
        assert_eq!(args.dec, Some(35.25));
        assert_eq!(args.radius_deg, Some(10.0));
        assert_eq!(args.sip_order, 3);
        assert_eq!(args.table.as_deref(), Some(Path::new("/tmp/image.xyls")));

        let linear = parse_args(&["-T".to_string(), "x.xyls".to_string()]);
        assert_eq!(linear.sip_order, 0);
        assert!(invoked_as_solve_field("/opt/astrometry/bin/solve-field"));
        assert!(invoked_as_solve_field("solve-field.exe"));
        assert!(!invoked_as_solve_field("seiza"));
    }

    #[test]
    fn parses_sirils_windows_asnet_script() {
        let script = "p=\"C:\\Users\\astro\\NINA\\image.xyls\"\n\
c=\"C:\\Users\\astro\\NINA\\stop\"\n\
solve-field -C \"$c\" -p -O -N none -R none -M none -B none -U none -S none --crpix-center -s FLUX -u arcsecperpix -L 1.8 -H 2.2 -t 3 --ra 150.5 --dec 35.25 --radius 10.0 \"$p\"\n";
        let argv = parse_asnet_script(script).unwrap();
        let args = parse_args(&argv);
        assert_eq!(
            args.stop_file.as_deref(),
            Some(Path::new("C:\\Users\\astro\\NINA\\stop"))
        );
        assert_eq!(
            args.table.as_deref(),
            Some(Path::new("C:\\Users\\astro\\NINA\\image.xyls"))
        );
        assert_eq!(args.sip_order, 3);
        assert_eq!(args.scale_low, Some(1.8));
        assert!(invoked_as_bash("C:\\solver\\bin\\bash.exe"));
        assert!(invoked_as_bash("/opt/solver/bin/bash"));
        assert!(!invoked_as_bash("seiza"));
    }

    #[test]
    fn redetection_recovers_orientation_and_offset_from_a_flipped_table() {
        // A synthetic image with gaussian stars; the "table" carries the
        // same stars in Siril's bottom-up frame with a half-pixel offset
        // and garbage flux ordering. Redetection must pick the flip, absorb
        // the offset, and return photometrically ranked stars.
        let dir = tempfile::tempdir().unwrap();
        let (width, height) = (400u32, 300u32);
        let mut image = image::GrayImage::from_pixel(width, height, image::Luma([20u8]));
        // A jittered 6x4 grid of stars with strictly decreasing brightness,
        // enough detections to clear the redetection floor.
        let positions: Vec<(f64, f64, f64)> = (0..24)
            .map(|index| {
                let column = (index % 6) as f64;
                let row = (index / 6) as f64;
                (
                    45.0 + column * 60.0 + (index % 5) as f64 * 3.0,
                    40.0 + row * 70.0 + (index % 3) as f64 * 4.0,
                    220.0 - index as f64 * 8.0,
                )
            })
            .collect();
        for pixel_y in 0..height {
            for pixel_x in 0..width {
                let mut value = 20.0f64;
                for &(x, y, amplitude) in positions.iter() {
                    let d2 = (pixel_x as f64 - x).powi(2) + (pixel_y as f64 - y).powi(2);
                    value += amplitude * (-d2 / 3.0).exp();
                }
                image.put_pixel(pixel_x, pixel_y, image::Luma([value.min(255.0) as u8]));
            }
        }
        let image_path = dir.path().join("field.png");
        image.save(&image_path).unwrap();

        // Bottom-up table with a constant +0.4 px offset and inverted flux.
        let table_stars: Vec<seiza::DetectedStar> = positions
            .iter()
            .enumerate()
            .map(|(index, &(x, y, _))| seiza::DetectedStar {
                x: x + 0.4,
                y: (height as f64 - 1.0 - y) + 0.4,
                flux: index as f64 + 1.0, // garbage: faintest first
                peak: 1.0,
                area: 1,
            })
            .collect();

        let table = dir.path().join("field.xyls");
        let stars =
            redetect_from_source_image(&table, &table_stars, (width, height)).expect("redetect");
        assert!(stars.len() >= positions.len());
        // Brightest-first by measured flux, mapped into the table's frame.
        let (bx, by, _) = positions[0];
        assert!((stars[0].x - (bx + 0.4)).abs() < 0.5, "{}", stars[0].x);
        assert!(
            (stars[0].y - (height as f64 - 1.0 - by + 0.4)).abs() < 0.5,
            "{}",
            stars[0].y
        );
        assert!(stars[0].flux > stars[1].flux);
    }

    #[test]
    fn bash_mode_answers_the_version_handshake() {
        // Siril's Windows version probe is `bash -l -c "solve-field --version"`.
        let raw: Vec<String> = ["-l", "-c", "solve-field --version"]
            .iter()
            .map(ToString::to_string)
            .collect();
        run_as_bash(&raw).unwrap();
    }

    #[test]
    fn wcs_file_is_block_aligned_with_parseable_cards() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image.wcs");
        let mut wcs = seiza::Wcs::from_center_scale_rotation(
            (150.0, 35.0),
            (2000.0, 1500.0),
            2.0,
            15.0,
            false,
        );
        wcs.sip = Some(seiza::Sip {
            order: 2,
            a: vec![1e-7, 2e-7, 3e-7],
            b: vec![4e-7, 5e-7, 6e-7],
            ap: vec![0.0; 6],
            bp: vec![0.0; 6],
        });
        write_wcs_file(&path, &wcs).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len() % FITS_BLOCK, 0);

        // Our own parser must read back what we wrote.
        let mut offset = 0;
        let cards = parse_header(&bytes, &mut offset).unwrap();
        assert_eq!(
            header_string(&cards, "CTYPE1").as_deref(),
            Some("RA---TAN-SIP")
        );
        assert_eq!(header_integer(&cards, "A_ORDER").unwrap(), 2);
        let crpix1: f64 = header_lookup(&cards, "CRPIX1").unwrap().parse().unwrap();
        assert!((crpix1 - 2001.0).abs() < 1e-9, "{crpix1}");
        assert!(header_lookup(&cards, "A_2_0").is_some());
        assert!(header_lookup(&cards, "BP_0_0").is_some());
    }
}
