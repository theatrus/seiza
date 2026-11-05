//! Exact temporal median/MAD rejection without a full-set scratch file.
//!
//! When the entire set fits safely, validated frames stay in memory. Otherwise
//! a window from each admitted input supplies a bounded frame-major tile.
//! Only the temporal samples at one detector coordinate are compared: steady
//! hot pixels and broad glow remain in the master, unlike spatial repair.
use super::{
    InputSignature, MasterBuildOptions, MasterBuildStage, MasterInputStatistics, check_cancelled,
    flat_rejection, report_progress,
};
use crate::{Error, FitsFrame, LinearImage, Result};
use rayon::prelude::*;
use seiza_image_io::ImageWindow;
use std::path::{Path, PathBuf};

/// Tile payload, excluding the output image and one decoded input/window.
const TILE_BYTES: usize = 256 * 1024 * 1024;

// Retain the validated, already calibrated frames only when the entire set
// fits in half the available physical memory. Leave the other half for the
// host, decode/read-ahead buffers, running moments, output and Rayon workers.
// No swap capacity is counted. On unsupported systems the window path stays.
pub(super) struct MemoryFrames {
    entries: Vec<(PathBuf, Vec<f32>)>,
}

impl MemoryFrames {
    pub(super) fn for_inputs(inputs: usize, samples: usize) -> Option<Self> {
        let mut system = sysinfo::System::new();
        system.refresh_memory();
        let available = system
            .cgroup_limits()
            .map_or(system.available_memory(), |limit| {
                system.available_memory().min(limit.free_memory)
            });
        Self::with_available(inputs, samples, available)
    }

    fn with_available(inputs: usize, samples: usize, available: u64) -> Option<Self> {
        let needed = inputs.checked_mul(samples)?.checked_mul(4)?;
        if needed as u64 > available / 2 {
            return None;
        }
        Some(Self {
            entries: Vec::new(),
        })
    }

    pub(super) fn push(&mut self, path: &Path, samples: Vec<f32>) {
        self.entries.push((path.to_path_buf(), samples));
    }

    pub(super) fn integrate(
        &self,
        paths: &[&PathBuf],
        options: &MasterBuildOptions,
    ) -> Result<IntegratedDark> {
        let frames = paths
            .iter()
            .map(|path| {
                self.entries
                    .iter()
                    .find(|(stored, _)| stored == *path)
                    .map(|(_, samples)| samples.as_slice())
                    .expect("admitted frame was cached")
            })
            .collect::<Vec<_>>();
        check_cancelled(options)?;
        report_progress(options, MasterBuildStage::Combine, 0, 1);
        let mut samples = vec![0.0; frames[0].len()];
        let tally = flat_rejection::combine_unmasked_frames(&frames, &mut samples, options)?;
        report_progress(options, MasterBuildStage::Combine, 1, 1);
        Ok(IntegratedDark {
            accepted_samples: (samples.len() * frames.len()) as u64 - tally.rejected_samples,
            samples,
            input_statistics: tally.input_statistics,
            rejected_samples: tally.rejected_samples,
            fallback_pixels: tally.fallback_pixels,
        })
    }
}

pub(super) fn keep_sources_open(
    paths: &[PathBuf],
    options: &MasterBuildOptions,
) -> Result<Vec<std::fs::File>> {
    // Keep source leases across window opens. Some network filesystems
    // invalidate cached pages when the last handle closes. No pixels are
    // read here: all decoding stays in seiza-image-io.
    check_cancelled(options)?;
    let Some(first) = paths.first() else {
        return Ok(Vec::new());
    };
    let Ok(file) = std::fs::File::open(first) else {
        return Ok(Vec::new());
    };
    // Reserve descriptor headroom while acquiring the optional leases, then
    // release it for decoding and the host. If even that reserve cannot be
    // acquired, omit this hint rather than causing a later open to fail.
    let mut reserve = vec![file];
    for _ in 1..32 {
        check_cancelled(options)?;
        let Ok(file) = reserve[0].try_clone() else {
            return Ok(Vec::new());
        };
        reserve.push(file);
    }
    let mut leases = Vec::new();
    for path in paths.iter().take(256) {
        check_cancelled(options)?;
        let Ok(file) = std::fs::File::open(path) else {
            break;
        };
        leases.push(file);
    }
    drop(reserve);
    Ok(leases)
}

pub(super) struct IntegratedDark {
    pub samples: Vec<f32>,
    pub input_statistics: Vec<MasterInputStatistics>,
    pub accepted_samples: u64,
    pub rejected_samples: u64,
    pub fallback_pixels: u64,
}

#[derive(Clone, Copy)]
struct Tile {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

pub(super) fn integrate(
    paths: &[&PathBuf],
    shape: &InputSignature,
    options: &MasterBuildOptions,
) -> Result<IntegratedDark> {
    integrate_with_budget(paths, shape, options, TILE_BYTES)
}

fn integrate_with_budget(
    paths: &[&PathBuf],
    shape: &InputSignature,
    options: &MasterBuildOptions,
    budget: usize,
) -> Result<IntegratedDark> {
    let frames = paths.len();
    let bytes_per_pixel = frames
        .checked_mul(shape.channels)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(size_error)?;
    let pixels = budget / bytes_per_pixel;
    if pixels == 0 {
        return Err(size_error());
    }
    let tile_width = shape.width.min(seiza_image_io::MAX_WINDOW_EDGE).min(pixels);
    let tile_height = (pixels / tile_width)
        .min(seiza_image_io::MAX_WINDOW_EDGE)
        .min(shape.height);
    let tiles = shape.width.div_ceil(tile_width) * shape.height.div_ceil(tile_height);
    let mut result = IntegratedDark {
        samples: vec![0.0; shape.width * shape.height * shape.channels],
        input_statistics: vec![
            MasterInputStatistics {
                accepted_samples: 0,
                rejected_samples: 0,
                masked_samples: 0,
            };
            frames
        ],
        accepted_samples: 0,
        rejected_samples: 0,
        fallback_pixels: 0,
    };
    let mut tile_index = 0;
    for y in (0..shape.height).step_by(tile_height) {
        for x in (0..shape.width).step_by(tile_width) {
            check_cancelled(options)?;
            report_progress(options, MasterBuildStage::Combine, tile_index, tiles);
            let tile = Tile {
                x,
                y,
                width: tile_width.min(shape.width - x),
                height: tile_height.min(shape.height - y),
            };
            let samples = tile.width * tile.height * shape.channels;
            let mut values = vec![0.0; samples * frames];
            let frame_samples = samples;
            for (paths, values) in paths.chunks(8).zip(values.chunks_mut(frame_samples * 8)) {
                // Host cancellation callbacks run only on the calling thread,
                // before scheduling each input and after the bounded batch.
                for _ in paths {
                    check_cancelled(options)?;
                }
                // Finish the parallel window reads before a full decode.
                // Compressed XISF may use Rayon internally: holding a lock in
                // a worker around that decode can deadlock nested Rayon work.
                let windows = paths
                    .par_iter()
                    .map(|path| {
                        seiza_image_io::read_window(
                            path,
                            tile.x,
                            tile.y,
                            tile.width,
                            tile.height,
                            None,
                        )
                        .ok()
                    })
                    .collect::<Vec<_>>();
                for ((path, window), values) in paths
                    .iter()
                    .zip(windows)
                    .zip(values.chunks_exact_mut(frame_samples))
                {
                    check_cancelled(options)?;
                    let window = finish_window(path, shape, tile, window)?;
                    prepare_window(&window, shape, tile, options.bias.as_ref(), values)?;
                }
                check_cancelled(options)?;
            }
            let mut output = vec![0.0; samples];
            let frame_values = values.chunks_exact(samples).collect::<Vec<_>>();
            let tally =
                flat_rejection::combine_unmasked_frames(&frame_values, &mut output, options)?;
            for (row, values) in output.chunks_exact(tile.width * shape.channels).enumerate() {
                let first = ((tile.y + row) * shape.width + tile.x) * shape.channels;
                result.samples[first..first + values.len()].copy_from_slice(values);
            }
            for (total, counts) in result
                .input_statistics
                .iter_mut()
                .zip(tally.input_statistics)
            {
                total.accepted_samples += counts.accepted_samples;
                total.rejected_samples += counts.rejected_samples;
            }
            result.accepted_samples += (samples * frames) as u64 - tally.rejected_samples;
            result.rejected_samples += tally.rejected_samples;
            result.fallback_pixels += tally.fallback_pixels;
            tile_index += 1;
        }
    }
    report_progress(options, MasterBuildStage::Combine, tiles, tiles);
    Ok(result)
}

fn finish_window(
    path: &Path,
    shape: &InputSignature,
    tile: Tile,
    window: Option<ImageWindow>,
) -> Result<ImageWindow> {
    let window = match window {
        Some(window) => window,
        // Compressed/embedded XISF has no window reader. Decode only one
        // full frame at a time, on the caller, after the window-read batch.
        None => read_full_window(path, shape, tile)?,
    };
    if (
        window.image_width,
        window.image_height,
        window.planes,
        window.x,
        window.y,
        window.width,
        window.height,
        window.values.len(),
    ) != (
        shape.width,
        shape.height,
        shape.channels,
        tile.x,
        tile.y,
        tile.width,
        tile.height,
        tile.width * tile.height * shape.channels,
    ) {
        return Err(Error::Calibration(format!(
            "{} changed geometry during dark integration",
            path.display()
        )));
    }
    Ok(window)
}

fn prepare_window(
    window: &ImageWindow,
    shape: &InputSignature,
    tile: Tile,
    bias: Option<&LinearImage>,
    output: &mut [f32],
) -> Result<()> {
    let pixels = tile.width * tile.height;
    if shape.channels == 1 && bias.is_none() {
        if window.values.iter().any(|value| !value.is_finite()) {
            return Err(Error::Calibration(
                "non-finite sample during dark integration".into(),
            ));
        }
        output.copy_from_slice(&window.values);
        return Ok(());
    }
    for (pixel, output) in output.chunks_exact_mut(shape.channels).enumerate() {
        let row = pixel / tile.width;
        let column = pixel % tile.width;
        let global = ((tile.y + row) * shape.width + tile.x + column) * shape.channels;
        for (channel, output) in output.iter_mut().enumerate() {
            let mut value = window.values[channel * pixels + pixel];
            if let Some(bias) = bias {
                value -= bias.data[global + channel];
            }
            if !value.is_finite() {
                return Err(Error::Calibration(
                    "non-finite sample during dark integration".into(),
                ));
            }
            *output = value;
        }
    }
    Ok(())
}

fn size_error() -> Error {
    Error::Calibration("dark temporal tile cannot hold one pixel across all inputs".into())
}

fn read_full_window(path: &Path, shape: &InputSignature, tile: Tile) -> Result<ImageWindow> {
    let frame = FitsFrame::open(path)?;
    if (frame.image.width, frame.image.height, frame.image.channels)
        != (shape.width, shape.height, shape.channels)
    {
        return Err(Error::Calibration(format!(
            "{} changed geometry during dark integration",
            path.display()
        )));
    }
    let mut values = Vec::with_capacity(tile.width * tile.height * shape.channels);
    for channel in 0..shape.channels {
        for y in tile.y..tile.y + tile.height {
            for x in tile.x..tile.x + tile.width {
                values.push(frame.image.data[(y * shape.width + x) * shape.channels + channel]);
            }
        }
    }
    Ok(ImageWindow {
        image_width: shape.width,
        image_height: shape.height,
        planes: shape.channels,
        x: tile.x,
        y: tile.y,
        width: tile.width,
        height: tile.height,
        values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CalibrationMasters, MasterFrameKind, MasterProgress, write_processed_image_fits_f32,
    };
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    #[test]
    fn source_leases_honor_cancellation() {
        let options = MasterBuildOptions {
            cancel: Some(crate::CancelSignal::new(|| true)),
            ..Default::default()
        };
        assert!(matches!(
            keep_sources_open(&[], &options),
            Err(Error::Cancelled)
        ));
    }

    #[test]
    #[cfg(unix)]
    fn source_leases_leave_descriptor_headroom() {
        const CHILD: &str = "SEIZA_TEST_LOW_FD_LEASES";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "master::dark_rejection::tests::source_leases_leave_descriptor_headroom",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        // Only this test runs in the child; never lower the test runner's
        // descriptor limit while other tests are using it.
        let limit = libc::rlimit {
            rlim_cur: 64,
            rlim_max: 64,
        };
        assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &limit) }, 0);
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("dark.fits");
        write_processed_image_fits_f32(
            &path,
            &LinearImage::new(2, 2, 1, vec![100.0; 4]).unwrap(),
            &[],
            &[],
        )
        .unwrap();
        let paths = vec![path.clone(); 200];
        let leases = keep_sources_open(&paths, &MasterBuildOptions::default()).unwrap();
        assert!(!leases.is_empty());
        assert!(leases.len() < 64);
        assert_eq!(FitsFrame::open(&path).unwrap().image.data, vec![100.0; 4]);
    }

    #[test]
    fn rgb_bias_calibration_and_counts_do_not_depend_on_tile_boundaries() {
        for extension in ["fits", "xisf"] {
            let directory = tempfile::tempdir().unwrap();
            let bias =
                LinearImage::new(5, 3, 3, (0..45).map(|i| 100.0 + i as f32).collect()).unwrap();
            let expected = (0..45)
                .map(|i| 1_000.0 + i as f32 * 10.0)
                .collect::<Vec<_>>();
            let paths = (0..9)
                .map(|frame| {
                    let mut data = expected
                        .iter()
                        .zip(&bias.data)
                        .map(|(v, b)| v + b + [-1.0, 0.0, 1.0][frame % 3])
                        .collect::<Vec<_>>();
                    if frame >= 6 {
                        data[17] += 2_000.0;
                        data[22] -= 800.0;
                    }
                    let path = directory.path().join(format!("dark-{frame}.{extension}"));
                    write_processed_image_fits_f32(
                        &path,
                        &LinearImage::new(5, 3, 3, data).unwrap(),
                        &[],
                        &[],
                    )
                    .unwrap();
                    path
                })
                .collect::<Vec<_>>();
            let options = MasterBuildOptions {
                bias: Some(bias),
                ..Default::default()
            };
            let calibration = CalibrationMasters::new(options.bias.clone(), None, None).unwrap();
            let first = super::super::prepare_input(
                &paths[0],
                FitsFrame::open(&paths[0]),
                MasterFrameKind::Dark,
                &options,
                &calibration,
                None,
                None,
            )
            .unwrap();
            let shape = InputSignature::from_frame(&first, MasterFrameKind::Dark);
            let mut mono_shape = shape.clone();
            mono_shape.channels = 1;
            let mono_tile = Tile {
                x: 0,
                y: 0,
                width: 5,
                height: 3,
            };
            let mut mono = ImageWindow {
                image_width: 5,
                image_height: 3,
                planes: 1,
                x: 0,
                y: 0,
                width: 5,
                height: 3,
                values: (0..15).map(|i| i as f32 / 8.0).collect(),
            };
            mono.values[0] = -0.0;
            mono.values[1] = f32::from_bits(1);
            mono.values[2] = f32::MAX;
            let mut output = vec![0.0; 15];
            prepare_window(&mono, &mono_shape, mono_tile, None, &mut output).unwrap();
            assert_eq!(
                output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                mono.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                mono.values[3] = invalid;
                assert!(prepare_window(&mono, &mono_shape, mono_tile, None, &mut output).is_err());
            }
            let paths = paths.iter().collect::<Vec<_>>();
            let budget = paths.len() * shape.channels * 4;
            let small = integrate_with_budget(&paths, &shape, &options, budget).unwrap();
            let large = integrate_with_budget(&paths, &shape, &options, TILE_BYTES).unwrap();
            let available = (paths.len() * expected.len() * 8) as u64;
            assert!(
                MemoryFrames::with_available(paths.len(), expected.len(), available - 1).is_none()
            );
            assert!(MemoryFrames::with_available(usize::MAX, usize::MAX, u64::MAX).is_none());
            let mut memory =
                MemoryFrames::with_available(paths.len(), expected.len(), available).unwrap();
            for path in &paths {
                let prepared = super::super::prepare_input(
                    path,
                    FitsFrame::open(path),
                    MasterFrameKind::Dark,
                    &options,
                    &calibration,
                    Some(&shape),
                    None,
                )
                .unwrap();
                memory.push(path, prepared.image.data);
            }
            let cached = memory.integrate(&paths, &options).unwrap();
            assert_eq!(cached.samples, small.samples);
            assert_eq!(cached.rejected_samples, small.rejected_samples);
            assert_eq!(cached.accepted_samples, small.accepted_samples);
            assert_eq!(cached.fallback_pixels, small.fallback_pixels);
            for (a, b) in cached.input_statistics.iter().zip(&small.input_statistics) {
                assert_eq!(a.accepted_samples, b.accepted_samples);
                assert_eq!(a.rejected_samples, b.rejected_samples);
            }
            // Level screening can remove admitted frames after caching.
            let subset = paths.iter().step_by(2).copied().collect::<Vec<_>>();
            let cached = memory.integrate(&subset, &options).unwrap();
            let windowed = integrate_with_budget(&subset, &shape, &options, budget).unwrap();
            assert_eq!(cached.samples, windowed.samples);
            assert_eq!(cached.rejected_samples, windowed.rejected_samples);
            assert_eq!(small.samples, expected);
            assert_eq!(large.samples, expected);
            assert_eq!(small.rejected_samples, 6);
            assert_eq!(small.accepted_samples, 9 * 45 - 6);
            assert_eq!(small.fallback_pixels, 0);
            assert_eq!(
                small
                    .input_statistics
                    .iter()
                    .map(|s| s.rejected_samples)
                    .collect::<Vec<_>>(),
                vec![0, 0, 0, 0, 0, 0, 2, 2, 2]
            );
            assert!(integrate_with_budget(&paths, &shape, &options, budget - 1).is_err());
            let tile = Tile {
                x: 1,
                y: 1,
                width: 3,
                height: 2,
            };
            assert_eq!(
                seiza_image_io::read_window(
                    paths[0],
                    tile.x,
                    tile.y,
                    tile.width,
                    tile.height,
                    None
                )
                .unwrap(),
                finish_window(paths[0], &shape, tile, None).unwrap()
            );
            let stopped = Arc::new(AtomicBool::new(false));
            let cancel = stopped.clone();
            let mut cancelled = options.clone();
            cancelled.cancel = Some(crate::CancelSignal::new(move || {
                cancel.load(Ordering::Relaxed)
            }));
            cancelled.progress = Some(MasterProgress::new(move |p| {
                if p.stage == MasterBuildStage::Combine && p.done == 1 {
                    stopped.store(true, Ordering::Relaxed);
                }
            }));
            assert!(matches!(
                integrate_with_budget(&paths, &shape, &cancelled, budget),
                Err(Error::Cancelled)
            ));
        }
    }
}
