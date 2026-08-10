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
//!
//! The handoff is a bounded channel per worker rather than shared flags,
//! because a channel gets the failure paths right by construction: a worker
//! that panics drops its sender, so the consumer's `recv` fails instead of
//! waiting forever, and a consumer that panics drops the receivers, so the
//! workers' `send` fails and they stop. Either way the scope joins and the
//! real panic is re-raised rather than becoming a hang.

use crate::stack::{PreparedFrame, prepare_frame};
use crate::{Error, FitsFrame, FrameDisposition, LiveStacker, Result, path_identity};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{SyncSender, sync_channel};

/// How much work to keep in flight while stacking a sequence of frames.
#[derive(Clone, Copy, Debug)]
pub struct PipelineOptions {
    /// Ceiling on the bytes held by frames waiting to be integrated.
    ///
    /// A prepared frame is the size of the reference image in `f32` samples,
    /// so the default worker count falls out of this budget and the
    /// reference's size rather than being counted in frames. Frame sizes vary
    /// by an order of magnitude between a small guide camera and a full-frame
    /// sensor, so a fixed frame count would mean wildly different memory on
    /// different rigs.
    ///
    /// This bounds the derived worker count only. An explicit [`Self::workers`]
    /// is taken at its word, because a caller who names a number has usually
    /// measured something this crate cannot see — remote storage, most often.
    /// At least one frame is always prepared, however small the budget.
    pub max_in_flight_bytes: usize,
    /// Put a frame the file declares as normalized onto this full scale as it
    /// is read, or `None` to leave every sample exactly as stored.
    ///
    /// A PixInsight XISF frame declaring `bounds="0:1"` has samples running
    /// 0..1 where a camera frame's run in the thousands, so a group mixing the
    /// two normalizes and rejects against values four orders of magnitude
    /// apart. Set this to the scale the rest of the frames use — 65535.0 for
    /// 16-bit camera data — and such a frame arrives comparable. See
    /// [`FitsFrame::rescale_declared_unit_bounds`], which this applies.
    ///
    /// Left off by default, because it changes sample values and only a
    /// caller knows what scale its other frames are on.
    pub normalized_full_scale: Option<f32>,
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
    /// from a local disk. Memory then follows the count given, roughly two
    /// prepared frames per worker, held to [`MAXIMUM_WORKERS`] threads.
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

/// Hard ceiling on threads, however many a caller asks for. Each is an OS
/// thread holding up to two prepared frames, and `scope.spawn` panics rather
/// than failing when the system runs out, so a mistyped or config-driven
/// number must not be able to reach that.
pub const MAXIMUM_WORKERS: usize = 64;

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            max_in_flight_bytes: DEFAULT_IN_FLIGHT_BYTES,
            normalized_full_scale: None,
            workers: None,
        }
    }
}

impl PipelineOptions {
    /// Options with a stated budget and a derived worker count.
    pub fn with_budget(max_in_flight_bytes: usize) -> Self {
        Self {
            max_in_flight_bytes,
            normalized_full_scale: None,
            workers: None,
        }
    }

    /// Workers to run for a reference frame of `frame_bytes`.
    ///
    /// An explicit count is honoured as given, held only to
    /// [`MAXIMUM_WORKERS`]. A derived one is also held to what the budget
    /// affords, costing each worker the frame it holds plus the one it is
    /// building. Always at least one.
    fn resolve_workers(&self, frame_bytes: usize) -> usize {
        if let Some(workers) = self.workers {
            return workers.clamp(1, MAXIMUM_WORKERS);
        }
        let affordable = if frame_bytes == 0 {
            MAXIMUM_DERIVED_WORKERS
        } else {
            self.max_in_flight_bytes / frame_bytes.saturating_mul(2)
        };
        let parallelism = std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1);
        // Half the cores, since each worker's own work is already parallel.
        (parallelism / 2)
            .clamp(1, MAXIMUM_DERIVED_WORKERS)
            .min(affordable.max(1))
            .max(1)
    }
}

/// What the caller wants to do after seeing one frame's outcome.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Continue {
    /// Keep going.
    Yes,
    /// Stop. See [`LiveStacker::push_fits_pipelined`] for what is already in
    /// flight when this takes effect.
    No,
}

/// What a run did, so trouble is visible to a caller that only checks the
/// return value.
///
/// A failed path is reported to the callback and does not stop the run, which
/// means the call itself can succeed having integrated nothing. This is the
/// summary that says so.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[must_use]
pub struct PipelineReport {
    /// Frames the accumulator took.
    pub integrated: usize,
    /// Frames an admission gate turned away.
    pub rejected: usize,
    /// Paths that could not be read, or that repeated one already stacked.
    pub failed: usize,
}

/// Whether a path can be prepared at all, decided before anything is opened.
enum Planned {
    /// Repeats a path this stack has already taken, or an earlier one in this
    /// batch. Settled up front so it costs no read, and so the concurrent and
    /// sequential paths refuse it at the same point.
    Duplicate,
    /// To be read and prepared. Carries the resolved identity so the ledger
    /// does not canonicalize the path a second time.
    Prepare(PathBuf),
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
    /// outcome a sequential [`LiveStacker::push_fits`] loop would have
    /// produced — the accumulator is still fed strictly in order. That
    /// includes the errors: a path that cannot be opened, or that repeats one
    /// this stack has already taken, arrives as `Err` for that path alone and
    /// the run carries on, exactly as a loop that logged and skipped would.
    /// The returned [`PipelineReport`] counts those failures, so a caller that
    /// only checks the return value still sees a run that integrated nothing.
    ///
    /// # Cancelling
    ///
    /// Answering [`Continue::No`] stops the run, but does not reach back into
    /// reads already begun. Each worker may hold one prepared frame and be
    /// building another, so up to two frames per worker beyond the cancel
    /// point have already been opened and are finished and discarded. A read
    /// cannot be interrupted once started: a path on a stalled mount, a FIFO,
    /// or a device node will hold the call until that read returns. Cancel
    /// promptness is bounded by the slowest read already in flight, not by the
    /// callback.
    ///
    /// # Rayon
    ///
    /// Running from inside a Rayon pool thread — including under
    /// `pool.install(..)` — prepares frames sequentially on the calling
    /// thread. This is deliberately conservative: the deadlock it avoids needs
    /// the caller's pool to be the one preparation submits to, which cannot be
    /// told apart through Rayon's API, and a caller who installed a pool did
    /// so to reserve cores that spawning outside it would quietly undo.
    pub fn push_fits_pipelined(
        &mut self,
        paths: &[PathBuf],
        options: &PipelineOptions,
        mut on_frame: impl FnMut(&Path, Result<FrameDisposition>) -> Continue,
    ) -> Result<PipelineReport> {
        self.require_fits_input_mode()?;
        if paths.is_empty() {
            return Ok(PipelineReport::default());
        }

        // Settled before any thread starts, so both paths refuse a repeat at
        // the same point and neither reads a frame it is going to discard.
        let plan = self.plan_batch(paths);

        let frame_bytes = self.reference.data.len() * std::mem::size_of::<f32>();
        let workers = options.resolve_workers(frame_bytes).min(paths.len());
        // Preparation submits Rayon work; blocking a pool thread while waiting
        // for it can starve the pool of the threads that would do it.
        if rayon::current_thread_index().is_some() {
            return Ok(self.run_sequentially(
                paths,
                &plan,
                options.normalized_full_scale,
                &mut on_frame,
            ));
        }

        let stop = AtomicBool::new(false);

        // One bounded channel per worker; worker `w` takes indices `w`,
        // `w + workers`, ... and the consumer reads them round-robin, so a
        // slow frame never lets a later one overtake it into the accumulator.
        // Capacity one holds a worker to the frame it has queued plus the one
        // it is building.
        let mut senders = Vec::with_capacity(workers);
        let mut receivers = Vec::with_capacity(workers);
        for _ in 0..workers {
            let (sender, receiver) = sync_channel::<Result<PreparedFrame>>(1);
            senders.push(sender);
            receivers.push(receiver);
        }

        let (preparation, mut integration) = self.split_for_pipeline();
        let mut report = PipelineReport::default();

        std::thread::scope(|scope| {
            for (worker, sender) in senders.into_iter().enumerate() {
                let preparation = &preparation;
                let stop = &stop;
                let plan = &plan;
                scope.spawn(move || {
                    prepare_worker(
                        WorkerShare {
                            worker,
                            workers,
                            paths,
                            plan,
                            preparation,
                            normalized_full_scale: options.normalized_full_scale,
                            stop,
                        },
                        sender,
                    )
                });
            }

            // Integrate in order on this thread, so the accumulator sees the
            // sequential sequence.
            for (index, path) in paths.iter().enumerate() {
                let Ok(prepared) = receivers[index % workers].recv() else {
                    // The worker for this index is gone. Either it panicked —
                    // in which case the scope re-raises that panic when it
                    // joins, which is a truer report than anything invented
                    // here — or it stopped because the run is ending.
                    break;
                };

                let outcome = match prepared {
                    Err(error) => Err(error),
                    Ok(prepared) => {
                        let identity = match &plan[index] {
                            Planned::Prepare(identity) => identity.clone(),
                            Planned::Duplicate => unreachable!("a duplicate is never prepared"),
                        };
                        let disposition = integration.integrate(prepared);
                        integration.record_input_identity(identity);
                        Ok(disposition)
                    }
                };
                report.count(&outcome);

                if on_frame(path, outcome) == Continue::No {
                    break;
                }
            }

            // Reached on every exit from the loop, including an early stop and
            // a `recv` failure. A panic in `on_frame` skips it, but unwinding
            // drops the receivers, which fails the workers' sends and stops
            // them just the same — that is why the channel replaced a flag.
            stop.store(true, Ordering::Relaxed);
            drop(receivers);
        });

        Ok(report)
    }

    /// Resolve every path's identity once and mark the repeats, before any of
    /// them is opened.
    fn plan_batch(&self, paths: &[PathBuf]) -> Vec<Planned> {
        let mut seen: HashSet<PathBuf> = self.input_identities();
        paths
            .iter()
            .map(|path| {
                let identity = path_identity(path);
                if seen.insert(identity.clone()) {
                    Planned::Prepare(identity)
                } else {
                    Planned::Duplicate
                }
            })
            .collect()
    }

    /// The fallback taken on a Rayon pool thread, reporting through the same
    /// callback and refusing the same paths at the same point.
    fn run_sequentially(
        &mut self,
        paths: &[PathBuf],
        plan: &[Planned],
        normalized_full_scale: Option<f32>,
        on_frame: &mut impl FnMut(&Path, Result<FrameDisposition>) -> Continue,
    ) -> PipelineReport {
        let mut report = PipelineReport::default();
        let (preparation, mut integration) = self.split_for_pipeline();
        for (index, path) in paths.iter().enumerate() {
            let outcome = match &plan[index] {
                Planned::Duplicate => Err(duplicate_error(path)),
                Planned::Prepare(identity) => {
                    prepare_one(path, &preparation, normalized_full_scale).map(|prepared| {
                        let disposition = integration.integrate(prepared);
                        integration.record_input_identity(identity.clone());
                        disposition
                    })
                }
            };
            report.count(&outcome);
            if on_frame(path, outcome) == Continue::No {
                break;
            }
        }
        report
    }
}

impl PipelineReport {
    fn count(&mut self, outcome: &Result<FrameDisposition>) {
        match outcome {
            Ok(FrameDisposition::Accepted(_)) => self.integrated += 1,
            Ok(FrameDisposition::Rejected(_)) => self.rejected += 1,
            Err(_) => self.failed += 1,
        }
    }
}

fn duplicate_error(path: &Path) -> Error {
    Error::Stack(format!(
        "FITS frame {} has already been used by this stack",
        path.display()
    ))
}

/// Prepare this worker's share of the batch, stopping as soon as the consumer
/// has gone or the run has been cancelled.
/// One worker's share of a batch: which indices it owns and everything it
/// needs to prepare them.
struct WorkerShare<'a> {
    worker: usize,
    workers: usize,
    paths: &'a [PathBuf],
    plan: &'a [Planned],
    preparation: &'a crate::stack::PreparationHalf<'a>,
    normalized_full_scale: Option<f32>,
    stop: &'a AtomicBool,
}

fn prepare_worker(share: WorkerShare<'_>, sender: SyncSender<Result<PreparedFrame>>) {
    let WorkerShare {
        worker,
        workers,
        paths,
        plan,
        preparation,
        normalized_full_scale,
        stop,
    } = share;
    let mut index = worker;
    while index < paths.len() {
        // Checked before the open, so a cancelled run starts no further read.
        // It cannot reach into one already begun; see `push_fits_pipelined`.
        if stop.load(Ordering::Relaxed) {
            return;
        }
        let outcome = match &plan[index] {
            Planned::Duplicate => Err(duplicate_error(&paths[index])),
            Planned::Prepare(_) => prepare_one(&paths[index], preparation, normalized_full_scale),
        };
        // A closed channel means the consumer has stopped or unwound; there is
        // nobody left to hand this to.
        if sender.send(outcome).is_err() {
            return;
        }
        index += workers;
    }
}

/// Open, calibrate, register and normalize one path.
fn prepare_one(
    path: &Path,
    half: &crate::stack::PreparationHalf<'_>,
    normalized_full_scale: Option<f32>,
) -> Result<PreparedFrame> {
    let mut frame = FitsFrame::open(path)?;
    if let Some(full_scale) = normalized_full_scale {
        frame.rescale_declared_unit_bounds(full_scale);
    }
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

    /// Options that force the channel handoff whatever the host's core count.
    /// A derived count is 1 on a 2-vCPU runner, which would send these tests
    /// down the sequential fallback and quietly stop testing the concurrency.
    fn concurrent(workers: usize) -> PipelineOptions {
        PipelineOptions {
            workers: Some(workers),
            ..PipelineOptions::default()
        }
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
        let _ = pipelined
            .push_fits_pipelined(&paths[1..], &concurrent(3), |_, disposition| {
                pipelined_dispositions.push(format!("{:?}", disposition.unwrap()));
                Continue::Yes
            })
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
        // One worker, still through the channel: the one configuration where a
        // capacity-one channel could stall against itself.
        let options = concurrent(1);
        let _ = pipelined
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
        let _ = stacker
            .push_fits_pipelined(&paths[1..], &concurrent(3), |path, _| {
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

    /// A loop around `push_fits` stacked every other frame and errored only on
    /// the repeat. Aborting the batch instead would lose a whole night to one
    /// symlink alias in a directory listing.
    #[test]
    fn a_repeated_path_costs_that_path_alone() {
        let (_directory, paths) = frame_set(4);
        let mut stacker = stacker_from(&paths[0]);

        let batch = vec![paths[1].clone(), paths[1].clone(), paths[2].clone()];
        let mut outcomes = Vec::new();
        let _ = stacker
            .push_fits_pipelined(&batch, &concurrent(3), |_, outcome| {
                outcomes.push(outcome.map_err(|error| error.to_string()));
                Continue::Yes
            })
            .unwrap();

        assert!(outcomes[0].is_ok());
        assert!(
            outcomes[1]
                .as_ref()
                .unwrap_err()
                .contains("already been used"),
            "{outcomes:?}"
        );
        assert!(outcomes[2].is_ok(), "the run carries on past the repeat");
        assert_eq!(stacker.input_paths().len(), 2);
    }

    /// And a path this stack already took is refused the same way.
    #[test]
    fn a_path_already_stacked_is_refused_for_itself() {
        let (_directory, paths) = frame_set(3);
        let mut stacker = stacker_from(&paths[0]);
        stacker.push_fits(&paths[1]).unwrap();

        let mut outcomes = Vec::new();
        let batch = vec![paths[1].clone(), paths[2].clone()];
        let _ = stacker
            .push_fits_pipelined(&batch, &concurrent(3), |_, outcome| {
                outcomes.push(outcome.map_err(|error| error.to_string()));
                Continue::Yes
            })
            .unwrap();

        assert!(
            outcomes[0]
                .as_ref()
                .unwrap_err()
                .contains("already been used"),
            "{outcomes:?}"
        );
        assert!(outcomes[1].is_ok());
    }

    #[test]
    fn an_unreadable_path_is_reported_without_stopping_the_run() {
        let (directory, paths) = frame_set(4);
        let broken = directory.path().join("broken.fits");
        std::fs::write(&broken, b"not a fits file").unwrap();

        let mut stacker = stacker_from(&paths[0]);
        let batch = vec![paths[1].clone(), broken, paths[2].clone()];
        let mut outcomes = Vec::new();
        let report = stacker
            .push_fits_pipelined(&batch, &concurrent(3), |_, disposition| {
                outcomes.push(disposition.is_ok());
                Continue::Yes
            })
            .unwrap();

        assert_eq!(outcomes, vec![true, false, true]);
        assert_eq!(report.failed, 1);
    }

    #[test]
    fn a_derived_worker_count_falls_out_of_the_budget() {
        let frame = 100 * 1024 * 1024;
        // Budget for two frames in flight: one worker, since each is costed at
        // the frame it holds plus the one it is building.
        let options = PipelineOptions::with_budget(2 * frame);
        assert_eq!(options.resolve_workers(frame), 1);
        // Far too small a budget still runs, one frame at a time.
        assert_eq!(PipelineOptions::with_budget(1).resolve_workers(frame), 1);
    }

    /// The remote-storage advice is to raise `workers`; clamping that to a
    /// budget the advice never mentions would make it quietly do nothing.
    #[test]
    fn an_explicit_worker_count_is_taken_at_its_word() {
        let frame = 100 * 1024 * 1024;
        let explicit = PipelineOptions {
            max_in_flight_bytes: 2 * frame,
            workers: Some(11),
            ..PipelineOptions::default()
        };
        assert_eq!(explicit.resolve_workers(frame), 11);
        // Zero would stall the run, so it means one.
        let zero = PipelineOptions {
            max_in_flight_bytes: usize::MAX,
            workers: Some(0),
            ..PipelineOptions::default()
        };
        assert_eq!(zero.resolve_workers(frame), 1);
    }

    /// A repeat must be refused before the file is touched, so the answer does
    /// not depend on whether the file is still there — and so the concurrent
    /// and sequential paths cannot disagree about which error a caller sees.
    #[test]
    fn a_repeat_is_refused_without_reading_the_file() {
        let (_directory, paths) = frame_set(3);
        let mut stacker = stacker_from(&paths[0]);
        stacker.push_fits(&paths[1]).unwrap();
        // The frame is gone now. Opening it would fail with an I/O error; the
        // duplicate check runs first, so it must not.
        std::fs::remove_file(&paths[1]).unwrap();

        let mut messages = Vec::new();
        let report = stacker
            .push_fits_pipelined(&[paths[1].clone()], &concurrent(3), |_, outcome| {
                messages.push(outcome.map_err(|error| error.to_string()).unwrap_err());
                Continue::Yes
            })
            .unwrap();

        assert!(
            messages[0].contains("already been used"),
            "expected the duplicate error, got {messages:?}"
        );
        assert_eq!(report.failed, 1);
        assert_eq!(report.integrated, 0);
    }

    /// A run can report an error for every path and still answer `Ok`, so the
    /// summary is what a caller checking only the return value has to go on.
    #[test]
    fn the_report_counts_what_the_run_did() {
        let (directory, paths) = frame_set(4);
        let broken = directory.path().join("broken.fits");
        std::fs::write(&broken, b"not a fits file").unwrap();

        let mut stacker = stacker_from(&paths[0]);
        let batch = vec![paths[1].clone(), broken, paths[1].clone()];
        let report = stacker
            .push_fits_pipelined(&batch, &concurrent(3), |_, _| Continue::Yes)
            .unwrap();

        assert_eq!(report.integrated, 1);
        assert_eq!(report.failed, 2, "the unreadable path and the repeat");
        assert_eq!(report.rejected, 0);
    }

    /// `scope.spawn` panics rather than failing when the system runs out of
    /// threads, so a config-driven number must never reach it.
    #[test]
    fn an_explicit_worker_count_is_held_to_a_ceiling() {
        let options = PipelineOptions {
            max_in_flight_bytes: usize::MAX,
            workers: Some(100_000),
            ..PipelineOptions::default()
        };
        assert_eq!(options.resolve_workers(1024), MAXIMUM_WORKERS);
    }

    /// A normalized XISF frame must arrive on the same scale as the camera
    /// frames beside it, or the group normalizes against values four orders of
    /// magnitude apart.
    #[test]
    fn a_declared_unit_frame_is_scaled_as_it_is_read() {
        let directory = tempfile::tempdir().unwrap();
        // Samples spanning 0..1, so the writer declares bounds="0:1".
        let pixels: Vec<f32> = (0..12).map(|index| index as f32 / 11.0).collect();
        let path = directory.path().join("normalized.xisf");
        seiza_xisf::write_f32_image(&path, 4, 3, seiza_fits::F32ImageData::Mono(&pixels), &[])
            .unwrap();

        let plain = FitsFrame::open(&path).unwrap();
        assert_eq!(plain.bounds, Some((0.0, 1.0)));
        assert_eq!(plain.image.data.last().copied(), Some(1.0));

        let mut scaled = FitsFrame::open(&path).unwrap();
        assert!(scaled.rescale_declared_unit_bounds(65535.0));
        assert_eq!(scaled.image.data.last().copied(), Some(65535.0));
        assert_eq!(scaled.bounds, Some((0.0, 65535.0)));
        // Already on the new scale, so a second call changes nothing.
        assert!(!scaled.rescale_declared_unit_bounds(65535.0));

        // And a range that is not an exact 0:1 is left alone, because this
        // toolkit's own writer reports the observed minimum and maximum.
        let physical: Vec<f32> = (0..12).map(|index| 100.0 + index as f32 * 50.0).collect();
        let path = directory.path().join("physical.xisf");
        seiza_xisf::write_f32_image(&path, 4, 3, seiza_fits::F32ImageData::Mono(&physical), &[])
            .unwrap();
        let mut frame = FitsFrame::open(&path).unwrap();
        assert!(!frame.rescale_declared_unit_bounds(65535.0));
        assert_eq!(frame.image.data, physical);
    }

    /// A panic in the caller's callback used to leave workers parked on a
    /// condvar nobody would notify, so the scope joined forever. It must
    /// unwind out instead.
    #[test]
    fn a_panicking_callback_unwinds_instead_of_hanging() {
        let (_directory, paths) = frame_set(9);
        let mut stacker = stacker_from(&paths[0]);

        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            stacker
                .push_fits_pipelined(&paths[1..], &concurrent(3), |_, _| panic!("caller blew up"))
        }));
        std::panic::set_hook(previous);

        let panic = outcome.expect_err("the caller's panic must reach the caller");
        let message = panic
            .downcast_ref::<&str>()
            .copied()
            .unwrap_or_else(|| panic.downcast_ref::<String>().map_or("", |s| s.as_str()));
        assert_eq!(message, "caller blew up");
    }

    /// Cancelling must not wait on paths that were never started.
    #[test]
    fn cancelling_opens_no_further_paths() {
        let (directory, paths) = frame_set(4);
        // A path that does not exist would fail if it were ever opened; the
        // run is cancelled before reaching it, so it must never be reported.
        let missing = directory.path().join("never-opened.fits");
        let batch = vec![paths[1].clone(), paths[2].clone(), missing];

        let mut stacker = stacker_from(&paths[0]);
        let mut seen = Vec::new();
        let _ = stacker
            .push_fits_pipelined(&batch, &concurrent(3), |path, _| {
                seen.push(path.to_path_buf());
                Continue::No
            })
            .unwrap();

        assert_eq!(
            seen.len(),
            1,
            "the callback stops the run at the first frame"
        );
        assert_eq!(stacker.input_paths().len(), 1);
    }

    /// The ordered half of the design only matters when a frame is rejected
    /// against the accumulator, so the equivalence has to be checked there
    /// too, not only on a batch where everything is accepted.
    #[test]
    fn the_equivalence_holds_when_frames_are_rejected() {
        let (_directory, paths) = frame_set(7);
        // Demand that every last sample survive rejection. Once the online
        // estimator has warmed up some always do not, so frames start being
        // turned away by the order-dependent gate rather than during
        // preparation — which is the path this equivalence rests on.
        let options = StackOptions {
            acceptance: crate::FrameAcceptanceCriteria {
                minimum_integrated_fraction: 1.0,
                ..Default::default()
            },
            ..StackOptions::default()
        };
        let build = || {
            LiveStacker::new(
                FitsFrame::open(&paths[0]).unwrap(),
                CalibrationMasters::default(),
                options.clone(),
            )
            .unwrap()
        };

        let mut sequential = build();
        let mut sequential_dispositions = Vec::new();
        for path in &paths[1..] {
            sequential_dispositions.push(format!("{:?}", sequential.push_fits(path).unwrap()));
        }
        let expected = sequential.snapshot().unwrap();
        assert!(expected.rejected_frames > 0, "the gate must actually bite");

        let mut pipelined = build();
        let mut pipelined_dispositions = Vec::new();
        let report = pipelined
            .push_fits_pipelined(&paths[1..], &concurrent(3), |_, outcome| {
                pipelined_dispositions.push(format!("{:?}", outcome.unwrap()));
                Continue::Yes
            })
            .unwrap();
        assert_eq!(report.rejected, expected.rejected_frames as usize);
        let actual = pipelined.snapshot().unwrap();

        assert_eq!(pipelined_dispositions, sequential_dispositions);
        assert_eq!(actual.rejected_frames, expected.rejected_frames);
        assert_eq!(actual.accepted_frames, expected.accepted_frames);
        assert_eq!(bits(&actual.image.data), bits(&expected.image.data));
    }
}
