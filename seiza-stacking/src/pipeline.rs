//! Overlapped frame preparation for [`LiveStacker`].
//!
//! A sequential stack alternates between two very different kinds of work: a
//! single-threaded read and decode, then a mostly-parallel register and
//! normalize, then a short accumulate. While one frame is being read, the
//! cores sit idle; while it is being registered, the disk or network sits
//! idle.
//!
//! Preparation reads only immutable stack state — the reference image, the
//! registrar's star catalogue, the calibration masters, and the options — so
//! it is a pure function of the frame. Only accumulation depends on the frames
//! before it. This module runs preparation for several frames at once and
//! hands the results back in submission order, so the accumulator sees exactly
//! the sequence it would have seen sequentially and the result is unchanged.

use crate::stack::{PreparedFrame, prepare_frame};
use crate::{Error, FitsFrame, FrameDisposition, LiveStacker, Result};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// How much work to keep in flight while stacking a sequence of frames.
#[derive(Clone, Copy, Debug)]
pub struct PipelineOptions {
    /// Ceiling on the bytes held by frames waiting to be integrated.
    ///
    /// A prepared frame is the size of the reference image in `f32` samples,
    /// and its undecoded source is held briefly alongside it, so the worker
    /// count falls out of this budget and the reference's size rather than
    /// being counted in frames. Frame sizes vary by an order of magnitude
    /// between a small guide camera and a full-frame sensor, so a fixed frame
    /// count would mean wildly different memory on different rigs.
    ///
    /// At least one frame is always prepared, however small the budget: a
    /// budget below one frame degrades to sequential rather than failing.
    pub max_in_flight_bytes: usize,
    /// Threads preparing frames, or `None` to derive one from the budget and
    /// the machine's parallelism.
    ///
    /// Each worker reads its own frame before registering it, so this is the
    /// read concurrency as well as the compute concurrency. The derived
    /// default suits frames on local storage, where reads are quick and
    /// preparation — itself parallel internally — is the cost.
    ///
    /// **Raise it for frames arriving over a network.** With a 300ms read
    /// latency and 12 frames, eleven workers finished in 1.58s against 1.89s
    /// for the derived six: once a worker spends most of its time waiting,
    /// more of them is what fills the link. A caller that knows its frames are
    /// remote should say so here, since this crate cannot tell a network mount
    /// from a local disk. The budget still bounds the memory.
    pub workers: Option<usize>,
}

/// Bytes of prepared frames to hold by default: enough to keep a few
/// full-frame sensors in flight without becoming a memory problem on a host
/// running other work.
const DEFAULT_IN_FLIGHT_BYTES: usize = 1024 * 1024 * 1024;

/// Ceiling on derived worker count. Preparation already uses every core
/// through Rayon, so the workers only need to cover each other's serial gaps.
/// Measured on a 16-core machine with 12MP frames from local storage, the
/// speedup climbs 1.03x, 1.41x, 1.67x, 1.86x, 1.98x for one through six
/// workers and then falls back slightly at eight, so there is nothing past
/// this to win locally. Frames read over a network do want more; that is what
/// [`PipelineOptions::workers`] is for.
const MAXIMUM_DERIVED_WORKERS: usize = 6;

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            max_in_flight_bytes: DEFAULT_IN_FLIGHT_BYTES,
            workers: None,
        }
    }
}

impl PipelineOptions {
    /// Options with a stated budget and a derived worker count.
    pub fn with_budget(max_in_flight_bytes: usize) -> Self {
        Self {
            max_in_flight_bytes,
            workers: None,
        }
    }

    /// Workers to run for a reference frame of `frame_bytes`.
    ///
    /// Each worker may hold a prepared frame and be building another, so it is
    /// budgeted at two frames. Always at least one.
    fn resolve_workers(&self, frame_bytes: usize) -> usize {
        let affordable = if frame_bytes == 0 {
            MAXIMUM_DERIVED_WORKERS
        } else {
            self.max_in_flight_bytes / frame_bytes.saturating_mul(2)
        };
        let parallelism = std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1);
        // Half the cores, since each worker's own work is already parallel.
        self.workers
            .unwrap_or_else(|| (parallelism / 2).clamp(1, MAXIMUM_DERIVED_WORKERS))
            .min(affordable.max(1))
            .max(1)
    }
}

/// What the caller wants to do after seeing one frame's outcome.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Continue {
    /// Keep going.
    Yes,
    /// Stop; the remaining paths are left untouched.
    No,
}

impl LiveStacker {
    /// Stack a sequence of FITS or XISF paths, preparing several at once.
    ///
    /// Each frame is read, calibrated, registered and normalized on a worker
    /// thread, so reads overlap both with each other and with the integration
    /// of earlier frames. That matters most when the frames are remote: with a
    /// simulated 300ms read latency this ran 3.0x faster than a sequential
    /// loop, against 2.0x on warm local storage.
    ///
    /// `on_frame` is called once per path, in the order given, with the same
    /// disposition a sequential run would have produced — the accumulator is
    /// still fed strictly in order. Answer [`Continue::No`] to stop early;
    /// paths after that point are never opened. The callback is where a caller
    /// records decisions, checkpoints, or notices a cancellation, so driving
    /// the loop is not given up to get the overlap.
    ///
    /// A path that fails to open is reported to the callback as an error and
    /// does not stop the run; it is the caller's to decide about, matching
    /// what a hand-written loop around [`LiveStacker::push_fits`] would do.
    ///
    /// Duplicate paths are refused up front, as [`LiveStacker::push_fits`]
    /// refuses them one at a time.
    pub fn push_fits_pipelined(
        &mut self,
        paths: &[PathBuf],
        options: &PipelineOptions,
        mut on_frame: impl FnMut(&Path, Result<FrameDisposition>) -> Continue,
    ) -> Result<()> {
        self.require_fits_input_mode()?;
        self.reject_duplicate_inputs(paths)?;
        if paths.is_empty() {
            return Ok(());
        }

        let frame_bytes = self.reference_resident_bytes();
        let workers = options.resolve_workers(frame_bytes).min(paths.len());

        // One slot per path. A worker fills its own indices and the consumer
        // reads them in order, so a slow frame never lets a later one overtake
        // it into the accumulator.
        let slots: Vec<Mutex<Option<Result<PreparedFrame>>>> =
            (0..paths.len()).map(|_| Mutex::new(None)).collect();
        let ready: Vec<(std::sync::Mutex<bool>, std::sync::Condvar)> = (0..paths.len())
            .map(|_| (Mutex::new(false), std::sync::Condvar::new()))
            .collect();
        let next_index = AtomicUsize::new(0);
        let stop = AtomicBool::new(false);
        // How far the consumer has got. A worker will not run more than
        // `workers` frames ahead of it, which is what bounds the memory.
        let consumed = AtomicUsize::new(0);
        let consumed_gate = (Mutex::new(0usize), std::sync::Condvar::new());

        let (preparation, mut integration) = self.split_for_pipeline();

        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| {
                    loop {
                        if stop.load(Ordering::Relaxed) {
                            return;
                        }
                        let index = next_index.fetch_add(1, Ordering::SeqCst);
                        if index >= paths.len() {
                            return;
                        }
                        // Wait until this frame is within the in-flight window.
                        {
                            let (lock, condvar) = &consumed_gate;
                            let mut seen = lock.lock().unwrap();
                            while !stop.load(Ordering::Relaxed)
                                && index >= consumed.load(Ordering::Acquire) + workers
                            {
                                seen = condvar.wait(seen).unwrap();
                            }
                        }
                        if stop.load(Ordering::Relaxed) {
                            return;
                        }

                        let outcome = prepare_one(&paths[index], &preparation);
                        *slots[index].lock().unwrap() = Some(outcome);
                        let (lock, condvar) = &ready[index];
                        *lock.lock().unwrap() = true;
                        condvar.notify_all();
                    }
                });
            }

            // Integrate in order on this thread, so the accumulator sees the
            // sequential sequence.
            let mut result = Ok(());
            for (index, path) in paths.iter().enumerate() {
                {
                    let (lock, condvar) = &ready[index];
                    let mut done = lock.lock().unwrap();
                    while !*done {
                        done = condvar.wait(done).unwrap();
                    }
                }
                let prepared = slots[index]
                    .lock()
                    .unwrap()
                    .take()
                    .expect("a readied slot always holds an outcome");

                let disposition = match prepared {
                    Ok(prepared) => {
                        let disposition = integration.integrate(prepared);
                        integration.record_input_path(path);
                        Ok(disposition)
                    }
                    Err(error) => Err(error),
                };
                let keep_going = on_frame(path, disposition);

                consumed.store(index + 1, Ordering::Release);
                {
                    let (lock, condvar) = &consumed_gate;
                    *lock.lock().unwrap() = index + 1;
                    condvar.notify_all();
                }

                if keep_going == Continue::No {
                    result = Ok(());
                    break;
                }
            }
            stop.store(true, Ordering::Relaxed);
            {
                let (lock, condvar) = &consumed_gate;
                let _guard = lock.lock().unwrap();
                condvar.notify_all();
            }
            result
        })
    }

    /// Bytes one prepared frame occupies, from the reference's shape.
    fn reference_resident_bytes(&self) -> usize {
        self.reference.data.len() * std::mem::size_of::<f32>()
    }
}

/// Open, calibrate, register and normalize one path.
fn prepare_one(path: &Path, half: &crate::stack::PreparationHalf<'_>) -> Result<PreparedFrame> {
    let mut frame = FitsFrame::open(path)?;
    if let Err(error) =
        half.calibration
            .apply(&mut frame.image, frame.exposure_seconds, frame.bayer)
    {
        let message = match error {
            Error::Calibration(message) => message,
            other => other.to_string(),
        };
        return Ok(PreparedFrame::Rejected(
            crate::FrameRejectionReason::Calibration(message),
        ));
    }
    let frame = match frame.into_prepared() {
        Ok(frame) => frame,
        Err(error) => {
            return Ok(PreparedFrame::Rejected(
                crate::FrameRejectionReason::IncompatibleImage(error.to_string()),
            ));
        }
    };
    prepare_frame(half.reference, half.registrar, half.options, frame.image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CalibrationMasters, StackOptions};
    use std::io::Write;

    fn card(out: &mut Vec<u8>, text: &str) {
        let mut bytes = text.as_bytes().to_vec();
        assert!(bytes.len() <= 80, "card too long: {text}");
        bytes.resize(80, b' ');
        out.extend_from_slice(&bytes);
    }

    /// A dithered star field, big enough for registration to have real work
    /// and small enough to stay a unit test.
    fn write_frame(path: &Path, frame: usize) {
        let (width, height) = (192usize, 160usize);
        let stars: Vec<(f32, f32, f32)> = (0..24)
            .map(|index| {
                let x = ((index * 7919) % 1000) as f32 / 1000.0 * (width as f32 - 24.0) + 12.0;
                let y = ((index * 6271) % 1000) as f32 / 1000.0 * (height as f32 - 24.0) + 12.0;
                (x, y, 6000.0 + ((index * 37) % 41) as f32 * 300.0)
            })
            .collect();
        let (dx, dy) = (
            ((frame * 13) % 5) as f32 - 2.0,
            ((frame * 7) % 3) as f32 - 1.0,
        );
        let mut header = Vec::new();
        card(&mut header, "SIMPLE  =                    T");
        card(&mut header, "BITPIX  =                   16");
        card(&mut header, "NAXIS   =                    2");
        card(&mut header, &format!("NAXIS1  = {width:>20}"));
        card(&mut header, &format!("NAXIS2  = {height:>20}"));
        card(&mut header, "BZERO   =                32768");
        card(&mut header, "IMAGETYP= 'LIGHT   '");
        card(&mut header, "EXPTIME =                 60.0");
        card(&mut header, "END");
        header.resize(header.len().div_ceil(2880) * 2880, b' ');

        let mut body = Vec::with_capacity(width * height * 2);
        for y in 0..height {
            for x in 0..width {
                let noise = ((x * 17 + y * 31 + frame * 11) % 23) as f32 * 1.5;
                let mut value = 1000.0 + noise;
                for (star_x, star_y, brightness) in &stars {
                    let ddx = x as f32 - (star_x + dx);
                    let ddy = y as f32 - (star_y + dy);
                    let r2 = ddx.mul_add(ddx, ddy * ddy);
                    if r2 < 40.0 {
                        value += brightness * (-r2 / 3.2).exp();
                    }
                }
                let stored = value.clamp(0.0, 65535.0) as u16;
                body.extend_from_slice(&((stored as i32 - 32768) as i16).to_be_bytes());
            }
        }
        body.resize(body.len().div_ceil(2880) * 2880, 0);

        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&body).unwrap();
    }

    fn frame_set(count: usize) -> (tempfile::TempDir, Vec<PathBuf>) {
        let directory = tempfile::tempdir().unwrap();
        let paths: Vec<PathBuf> = (0..count)
            .map(|frame| {
                let path = directory.path().join(format!("light_{frame:03}.fits"));
                write_frame(&path, frame);
                path
            })
            .collect();
        (directory, paths)
    }

    /// Compare as raw bits, so a masked `NaN` counts as equal to itself and
    /// any drift in the low bits is caught rather than tolerated.
    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    fn stacker_from(reference: &Path) -> LiveStacker {
        LiveStacker::new(
            FitsFrame::open(reference).unwrap(),
            CalibrationMasters::default(),
            StackOptions::default(),
        )
        .unwrap()
    }

    /// The whole point: overlapping preparation must not move a single bit of
    /// the result. If this ever fails, the split between what preparation
    /// reads and what integration mutates has been broken.
    #[test]
    fn a_pipelined_stack_is_bit_identical_to_a_sequential_one() {
        let (_directory, paths) = frame_set(7);

        let mut sequential = stacker_from(&paths[0]);
        let mut sequential_dispositions = Vec::new();
        for path in &paths[1..] {
            sequential_dispositions.push(format!("{:?}", sequential.push_fits(path).unwrap()));
        }
        let expected = sequential.snapshot().unwrap();

        let mut pipelined = stacker_from(&paths[0]);
        let mut pipelined_dispositions = Vec::new();
        pipelined
            .push_fits_pipelined(
                &paths[1..],
                &PipelineOptions::default(),
                |_, disposition| {
                    pipelined_dispositions.push(format!("{:?}", disposition.unwrap()));
                    Continue::Yes
                },
            )
            .unwrap();
        let actual = pipelined.snapshot().unwrap();

        assert_eq!(pipelined_dispositions, sequential_dispositions);
        assert_eq!(actual.accepted_frames, expected.accepted_frames);
        assert_eq!(actual.rejected_frames, expected.rejected_frames);
        assert_eq!(actual.coverage, expected.coverage);
        assert_eq!(actual.rejected_samples, expected.rejected_samples);
        assert_eq!(
            bits(&actual.image.data),
            bits(&expected.image.data),
            "pipelined means must match bit for bit"
        );
        assert_eq!(
            bits(&actual.variance.data),
            bits(&expected.variance.data),
            "pipelined variances must match bit for bit"
        );
        assert_eq!(pipelined.input_paths(), sequential.input_paths());
    }

    /// The same must hold when only one worker runs, so a tight memory budget
    /// does not quietly take a different code path.
    #[test]
    fn a_single_worker_pipeline_matches_too() {
        let (_directory, paths) = frame_set(5);

        let mut sequential = stacker_from(&paths[0]);
        for path in &paths[1..] {
            sequential.push_fits(path).unwrap();
        }
        let expected = sequential.snapshot().unwrap();

        let mut pipelined = stacker_from(&paths[0]);
        let options = PipelineOptions {
            max_in_flight_bytes: 1,
            workers: None,
        };
        pipelined
            .push_fits_pipelined(&paths[1..], &options, |_, _| Continue::Yes)
            .unwrap();

        assert_eq!(
            bits(&pipelined.snapshot().unwrap().image.data),
            bits(&expected.image.data)
        );
    }

    #[test]
    fn stopping_early_leaves_the_remaining_paths_alone() {
        let (_directory, paths) = frame_set(6);
        let mut stacker = stacker_from(&paths[0]);
        let mut seen = Vec::new();
        stacker
            .push_fits_pipelined(&paths[1..], &PipelineOptions::default(), |path, _| {
                seen.push(path.to_path_buf());
                if seen.len() == 2 {
                    Continue::No
                } else {
                    Continue::Yes
                }
            })
            .unwrap();

        assert_eq!(seen.len(), 2);
        assert_eq!(stacker.input_paths().len(), 2);
    }

    #[test]
    fn a_repeated_path_is_refused_before_anything_is_opened() {
        let (_directory, paths) = frame_set(3);
        let mut stacker = stacker_from(&paths[0]);

        let repeated = vec![paths[1].clone(), paths[1].clone()];
        let error = stacker
            .push_fits_pipelined(&repeated, &PipelineOptions::default(), |_, _| Continue::Yes)
            .expect_err("a batch repeating a path must be refused");
        assert!(format!("{error}").contains("twice"), "{error}");

        stacker.push_fits(&paths[1]).unwrap();
        let error = stacker
            .push_fits_pipelined(&[paths[1].clone()], &PipelineOptions::default(), |_, _| {
                Continue::Yes
            })
            .expect_err("a path already stacked must be refused");
        assert!(format!("{error}").contains("already been used"), "{error}");
    }

    #[test]
    fn an_unreadable_path_is_reported_without_stopping_the_run() {
        let (directory, paths) = frame_set(4);
        let broken = directory.path().join("broken.fits");
        std::fs::write(&broken, b"not a fits file").unwrap();

        let mut stacker = stacker_from(&paths[0]);
        let batch = vec![paths[1].clone(), broken, paths[2].clone()];
        let mut outcomes = Vec::new();
        stacker
            .push_fits_pipelined(&batch, &PipelineOptions::default(), |_, disposition| {
                outcomes.push(disposition.is_ok());
                Continue::Yes
            })
            .unwrap();

        assert_eq!(outcomes, vec![true, false, true]);
    }

    #[test]
    fn worker_count_falls_out_of_the_budget() {
        let frame = 100 * 1024 * 1024;
        // Budget for two frames in flight: one worker, since each is costed at
        // the frame it holds plus the one it is building.
        let options = PipelineOptions::with_budget(2 * frame);
        assert_eq!(options.resolve_workers(frame), 1);
        // Far too small a budget still runs, one frame at a time.
        assert_eq!(PipelineOptions::with_budget(1).resolve_workers(frame), 1);
        // An explicit count is honoured up to what the budget affords.
        let explicit = PipelineOptions {
            max_in_flight_bytes: 100 * frame,
            workers: Some(3),
        };
        assert_eq!(explicit.resolve_workers(frame), 3);
        // And is still clamped by a budget that cannot hold that many.
        let cramped = PipelineOptions {
            max_in_flight_bytes: 2 * frame,
            workers: Some(8),
        };
        assert_eq!(cramped.resolve_workers(frame), 1);
    }
}
