//! Measure and screen one set of real calibration frames.
//!
//! Usage: `cargo run --release -p seiza-stacking --example calibration_frame_quality --
//! <flat|bias|dark> [--cfa] [--full-scale N] [--inject <master> <first> <last> <frames>]
//! <frame>...`
//!
//! Give it the frames of one set: one filter and session for flats, one
//! exposure and temperature for darks. Frames are read one at a time and only
//! their measurements are kept, so memory stays near one frame. A raw colour
//! frame is treated as CFA when its header names a Bayer pattern, or always
//! with `--cfa`. Prints one line per frame, then each verdict.
//!
//! `--inject <master> <first> <last> <frames>` copies the compact defects of
//! a normalized master flat into the flats before they are measured, to test
//! the screen on real noise and vignetting. The defects enter at strength
//! `first` in frame 0 and change evenly to `last` by frame `frames - 1`;
//! later frames get none. Strength 1 is the master's own depth.

use seiza_calibration::{
    BiasScreening, DarkMeasureOptions, DarkScreening, FlatMeasureOptions, FlatScreening,
    FrameVerdict, LinearImageRef, measure_dark, measure_flat, screen_bias, screen_darks,
    screen_flats,
};
use seiza_stacking::FitsFrame;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    let mut arguments = std::env::args().skip(1);
    let kind = arguments.next().expect(
        "usage: calibration_frame_quality <flat|bias|dark> [--cfa] [--full-scale N] <frame>...",
    );
    let mut force_cfa = false;
    let mut flat_options = FlatMeasureOptions::default();
    let mut injection = None;
    let mut paths = Vec::new();
    let number = |arguments: &mut dyn Iterator<Item = String>, flag: &str| -> f32 {
        arguments
            .next()
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(|| panic!("{flag} takes a number"))
    };
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--cfa" => force_cfa = true,
            "--full-scale" => flat_options.full_scale = number(&mut arguments, "--full-scale"),
            "--inject" => {
                let master = PathBuf::from(arguments.next().expect("--inject takes a master"));
                let first = number(&mut arguments, "--inject");
                let last = number(&mut arguments, "--inject");
                let frames = number(&mut arguments, "--inject") as usize;
                injection = Some(Injection::new(&master, first, last, frames));
            }
            _ => paths.push(PathBuf::from(argument)),
        }
    }
    match kind.as_str() {
        "flat" => flats(&paths, force_cfa, &flat_options, injection.as_ref()),
        "bias" | "dark" => darks(&paths, kind == "bias"),
        _ => panic!("the first argument must be flat, bias, or dark"),
    }
}

fn name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn verdict(verdict: &FrameVerdict) -> String {
    let findings: Vec<String> = verdict
        .findings
        .iter()
        .map(|finding| {
            format!(
                "{}={:.4} ({} at {})",
                finding.code.as_str(),
                finding.value,
                finding.grade.as_str(),
                finding.limit
            )
        })
        .collect();
    format!("{:<7} {}", verdict.grade.as_str(), findings.join(", "))
}

fn flats(
    paths: &[PathBuf],
    force_cfa: bool,
    options: &FlatMeasureOptions,
    injection: Option<&Injection>,
) {
    let mut measurements = Vec::with_capacity(paths.len());
    println!(
        "frame  level  max-ch/fs  min-ch/fs  clipped  noise  depth  features  deepest  seconds"
    );
    for (index, path) in paths.iter().enumerate() {
        let mut frame = FitsFrame::open(path).expect("open frame");
        if let Some(injection) = injection {
            injection.apply(&mut frame.image.data, index);
        }
        let image = &frame.image;
        let view = LinearImageRef::new(&image.data, image.width, image.height, image.channels)
            .expect("image view");
        let started = Instant::now();
        let measurement =
            measure_flat(view, force_cfa || frame.bayer.is_some(), options).expect("measure flat");
        let seconds = started.elapsed().as_secs_f64();
        let full_scale = measurement.full_scale;
        let brightest = measurement
            .channel_levels
            .iter()
            .copied()
            .fold(0.0, f32::max);
        let dimmest = measurement
            .channel_levels
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        println!(
            "{}  {:.0}  {:.3}  {:.3}  {:.2e}  {:.4}  {:.4}  {}  {:.3}  {:.2}",
            name(path),
            measurement.level,
            brightest / full_scale,
            dimmest / full_scale,
            measurement.clipped_fraction,
            measurement.noise,
            measurement.detection_depth,
            measurement.feature_count,
            measurement
                .features
                .first()
                .map_or(0.0, |feature| feature.depth),
            seconds,
        );
        measurements.push(measurement);
    }
    let report = screen_flats(&measurements, &FlatScreening::default()).expect("screen flats");
    println!();
    for (path, frame) in paths.iter().zip(&report.frames) {
        println!("{}  {}", name(path), verdict(frame));
    }
    println!(
        "set: steady {} changing {} carried {}; master {}",
        report.steady_features,
        report.changing_features,
        report.carried_features,
        verdict(&report.master)
    );
}

fn darks(paths: &[PathBuf], bias: bool) {
    let options = DarkMeasureOptions::default();
    let mut measurements = Vec::with_capacity(paths.len());
    println!("frame  level  sigma  noise  rows  columns  hot  seconds");
    for path in paths {
        let frame = FitsFrame::open(path).expect("open frame");
        let image = &frame.image;
        let view = LinearImageRef::new(&image.data, image.width, image.height, image.channels)
            .expect("image view");
        let started = Instant::now();
        let measurement = measure_dark(view, &options).expect("measure dark");
        println!(
            "{}  {:.1}  {:.2}  {:.2}  {:.3}  {:.3}  {}  {:.2}",
            name(path),
            measurement.level,
            measurement.sigma,
            measurement.noise,
            measurement.row_banding,
            measurement.column_banding,
            measurement.hot_pixels,
            started.elapsed().as_secs_f64(),
        );
        measurements.push(measurement);
    }
    let verdicts = if bias {
        screen_bias(&measurements, &BiasScreening::default())
    } else {
        screen_darks(&measurements, &DarkScreening::default())
    };
    println!();
    for (path, frame) in paths.iter().zip(&verdicts) {
        println!("{}  {}", name(path), verdict(frame));
    }
}

/// Compact defects lifted from a normalized master flat, as sample index and
/// fractional change.
struct Injection {
    width: usize,
    height: usize,
    defects: Vec<(usize, f32)>,
    first: f32,
    last: f32,
    frames: usize,
}

impl Injection {
    /// Find where the master departs from its smooth response by more than
    /// 3%, and keep the master's own shape within 4 pixels of each spot.
    fn new(master: &Path, first: f32, last: f32, frames: usize) -> Self {
        const BIN: usize = 64;
        const REACH: usize = 4;
        let master = FitsFrame::open(master).expect("open master");
        let image = &master.image;
        assert_eq!(image.channels, 1, "the master must have one channel");
        let (width, height, data) = (image.width, image.height, &image.data);
        let (columns, rows) = (width.div_ceil(BIN), height.div_ceil(BIN));
        let mut bins = vec![0.0_f32; columns * rows];
        for (index, bin) in bins.iter_mut().enumerate() {
            let (x0, y0) = ((index % columns) * BIN, (index / columns) * BIN);
            let mut values: Vec<f32> = (y0..(y0 + BIN).min(height))
                .flat_map(|y| data[y * width + x0..y * width + (x0 + BIN).min(width)].iter())
                .copied()
                .filter(|value| value.is_finite())
                .collect();
            values.sort_by(f32::total_cmp);
            *bin = values.get(values.len() / 2).copied().unwrap_or(1.0);
        }
        let smooth = |x: usize, y: usize| -> f32 {
            let position = |p: usize, count: usize| {
                let p = (p as f32 - (BIN as f32 - 1.0) / 2.0) / BIN as f32;
                let p = p.clamp(0.0, (count - 1) as f32);
                let low = (p.floor() as usize).min(count.saturating_sub(2));
                (low, (low + 1).min(count - 1), p - low as f32)
            };
            let (x0, x1, tx) = position(x, columns);
            let (y0, y1, ty) = position(y, rows);
            let at = |column: usize, row: usize| bins[row * columns + column];
            let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * tx;
            let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * tx;
            top + (bottom - top) * ty
        };
        let change = |index: usize| data[index] / smooth(index % width, index / width) - 1.0;
        let mut keep = vec![false; width * height];
        for y in REACH..height - REACH {
            for x in REACH..width - REACH {
                if change(y * width + x).abs() > 0.03 {
                    for ky in y - REACH..=y + REACH {
                        keep[ky * width + x - REACH..=ky * width + x + REACH].fill(true);
                    }
                }
            }
        }
        let defects: Vec<(usize, f32)> = keep
            .iter()
            .enumerate()
            .filter(|(_, kept)| **kept)
            .map(|(index, _)| (index, change(index)))
            .collect();
        eprintln!(
            "injecting {} pixels of defects, strength {first} to {last} over {frames} frames",
            defects.len()
        );
        Self {
            width,
            height,
            defects,
            first,
            last,
            frames,
        }
    }

    fn apply(&self, data: &mut [f32], frame: usize) {
        assert_eq!(
            data.len(),
            self.width * self.height,
            "frame and master differ in size"
        );
        if frame >= self.frames {
            return;
        }
        let progress = if self.frames > 1 {
            frame as f32 / (self.frames - 1) as f32
        } else {
            0.0
        };
        let strength = self.first + (self.last - self.first) * progress;
        for &(index, change) in &self.defects {
            data[index] *= 1.0 + strength * change;
        }
    }
}
