//! The three passes run band by band: every frame's samples for one band of
//! rows are read once, and all three passes run on them before the next
//! band is read.
//!
//! Taken a whole frame at a time, each pass streams every pixel's running
//! statistics, about 60 bytes per sample, through memory once per frame, and
//! reads every frame again; the passes are then bound by memory, not
//! arithmetic. Here each task takes a tile of one band: its statistics stay
//! in the core's caches while it runs pass 1 over every frame, then pass 2,
//! then pass 3, as PixInsight's ImageIntegration works through stripes of
//! its frames. Each sample sees the same arithmetic, frame by frame in the
//! same order, as in [`super::integrate_registered_frames`], so the result
//! is the same to the bit.

use super::{
    BatchFrameDiagnostics, BatchStackOptions, BatchStackResult, ChannelWeights, FirstEstimate,
    Moments, OutputChunk, Outputs, PackedFates, Rejection, SampleFate, check_cancelled,
    estimate_chunk, integrate_chunk, refine_chunk, validate_options,
};
use crate::{CancelSignal, Error, ReferenceRegion, Result};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Frames that can be read a band of rows at a time.
pub(crate) trait BandSource: Sync {
    /// Fill `band`, whole rows of interleaved samples, with frame `index`'s
    /// registered and normalized samples from row `top` on.
    fn read(&self, index: usize, top: usize, band: &mut [f32]) -> Result<()>;

    /// Stream a band into its destination in bounded runs. Cache sources
    /// override this to decode and normalize straight into GPU staging,
    /// without holding another complete input band.
    #[cfg(feature = "gpu")]
    fn read_into(
        &self,
        index: usize,
        top: usize,
        samples: usize,
        write: &mut dyn FnMut(usize, &[f32]),
    ) -> Result<()> {
        let mut band = vec![0.0; samples];
        self.read(index, top, &mut band)?;
        write(0, &band);
        Ok(())
    }

    /// The region of the grid frame `index` covers, every sample beyond
    /// which is `NaN`, when it is not the whole grid: a frame on part of an
    /// expanded canvas. A band it does not reach is not read.
    fn region(&self, _index: usize) -> Option<ReferenceRegion> {
        None
    }
}

/// Whether `region`, or the whole grid when there is none, holds any of
/// rows `top..top + rows`.
fn reaches_rows(region: Option<ReferenceRegion>, top: usize, rows: usize) -> bool {
    region.is_none_or(|region| top < region.y + region.height && region.y < top + rows)
}

/// Bytes of every frame's samples one tile should span: small enough for a
/// core's level 2 cache, where the second and third passes find what the
/// first read.
const TILE_BYTES: usize = 256 * 1024;

/// Integrate `frame_count` frames of `shape` (width, height, channels) from
/// `source` band by band, with the rejection and weights of `options`.
/// `progress` hears the band index and the band count before each band is
/// integrated. When `record_fates` is set, the result carries every frame's
/// sample fates for a drizzle.
///
/// Memory: the outputs, 16 bytes per sample (20 when weighted); two bands
/// of every frame, held within [`BatchStackOptions::band_memory_bytes`];
/// and, when recording fates, a quarter of a byte per sample per frame.
pub(crate) fn integrate_bands(
    shape: (usize, usize, usize),
    frame_count: usize,
    options: &BatchStackOptions,
    source: &impl BandSource,
    record_fates: bool,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<(BatchStackResult, Option<Vec<PackedFates>>)> {
    validate_options(frame_count, options)?;
    let (width, height, channels) = shape;
    if width == 0 || height == 0 || !matches!(channels, 1 | 3) {
        return Err(Error::Stack(
            "batch frames must share a valid registered image shape".into(),
        ));
    }
    let run = Run {
        shape,
        frame_count,
        options,
        record_fates,
        #[cfg(all(test, feature = "gpu"))]
        fail_gpu_band: None,
    };
    match (&options.frame_weights, channels) {
        (None, _) => {
            let weights = vec![ChannelWeights::<1>::unit(); frame_count];
            run.integrate::<1, false>(&weights, source, progress)
        }
        (Some(frame_weights), 1) => {
            let weights = ChannelWeights::<1>::for_frames(frame_weights)?;
            run.integrate::<1, true>(&weights, source, progress)
        }
        (Some(frame_weights), _) => {
            let weights = ChannelWeights::<3>::for_frames(frame_weights)?;
            run.integrate::<3, true>(&weights, source, progress)
        }
    }
}

/// Bytes of one set of bands to aim for: long reads, yet a first set,
/// which nothing overlaps, that is read quickly.
const BAND_SET_BYTES: usize = 128 * 1024 * 1024;

/// Rows per band for `frame_count` frames of `shape`: two sets of bands
/// within `memory`, each no more than [`BAND_SET_BYTES`], and one row at
/// least.
pub(crate) fn band_rows(shape: (usize, usize, usize), frame_count: usize, memory: usize) -> usize {
    let (width, height, channels) = shape;
    let row_bytes = width * channels * std::mem::size_of::<f32>();
    ((memory / 2).min(BAND_SET_BYTES) / frame_count.saturating_mul(row_bytes).max(1))
        .clamp(1, height)
}

/// Read every frame of `shape` from `source` a band of `rows` rows at a
/// time, handing each band's samples, one slice per frame, to `each` with
/// the band's index and first row, in order. A frame whose
/// [`BandSource::region`] holds none of the band's rows is not read, and its
/// slice is empty: every sample of it is `NaN`. The next band is read while
/// `each` works on this one, so two bands of every frame are held: on a
/// thread of its own, or from a pool thread as a task in that pool; see
/// [`crate::tasks`].
pub(crate) fn for_each_band(
    shape: (usize, usize, usize),
    frame_count: usize,
    rows: usize,
    source: &impl BandSource,
    cancel: Option<&CancelSignal>,
    mut each: impl FnMut(usize, usize, &[&[f32]]) -> Result<()>,
) -> Result<()> {
    let (width, height, channels) = shape;
    let row_samples = width * channels;
    let bands = height.div_ceil(rows);
    let band_samples = |band: usize| rows.min(height - band * rows) * row_samples;
    let regions = (0..frame_count)
        .map(|index| source.region(index))
        .collect::<Vec<_>>();
    let reaches = |index: usize, band: usize| {
        reaches_rows(
            regions[index],
            band * rows,
            band_samples(band) / row_samples,
        )
    };
    let cancelled = || cancel.is_some_and(CancelSignal::is_cancelled);
    let read = |band: usize, mut frames: Vec<Vec<f32>>| {
        if cancelled() {
            return Err(Error::Cancelled);
        }
        frames
            .par_iter_mut()
            .enumerate()
            .filter(|(index, _)| reaches(*index, band))
            .try_for_each(|(index, buffer)| {
                source.read(index, band * rows, &mut buffer[..band_samples(band)])
            })?;
        Ok(frames)
    };
    let buffers = || vec![vec![0.0_f32; rows * row_samples]; frame_count];
    crate::tasks::ahead(|ahead| {
        std::thread::scope(|scope| {
            let mut reading = Some(ahead.spawn(scope, Vec::new, move || read(0, buffers())));
            let mut spare = None;
            for band in 0..bands {
                let frames = ahead.join(
                    reading
                        .take()
                        .expect("each band is read before it is needed"),
                )?;
                if band + 1 < bands {
                    let next = spare.take().unwrap_or_else(buffers);
                    reading = Some(ahead.spawn(scope, Vec::new, move || read(band + 1, next)));
                }
                if cancelled() {
                    return Err(Error::Cancelled);
                }
                let samples = frames
                    .iter()
                    .enumerate()
                    .map(|(index, frame)| {
                        let len = if reaches(index, band) {
                            band_samples(band)
                        } else {
                            0
                        };
                        &frame[..len]
                    })
                    .collect::<Vec<_>>();
                each(band, band * rows, &samples)?;
                drop(samples);
                spare = Some(frames);
            }
            Ok(())
        })
    })
}

struct Run<'a> {
    shape: (usize, usize, usize),
    frame_count: usize,
    options: &'a BatchStackOptions,
    record_fates: bool,
    /// Exercise rollback after work has been submitted in both buffer sets.
    #[cfg(all(test, feature = "gpu"))]
    fail_gpu_band: Option<usize>,
}

/// Each frame's finite and integrated sample counts.
#[derive(Default)]
struct FrameCounts {
    finite: AtomicUsize,
    integrated: AtomicUsize,
}

/// What every tile of a band shares.
struct Band<'a, const C: usize> {
    /// Every frame's samples of the band; empty for a frame the band does
    /// not reach.
    frames: &'a [&'a [f32]],
    /// The region of the grid each frame covers, when it is not the whole
    /// grid.
    regions: &'a [Option<ReferenceRegion>],
    /// The grid's width in pixels and its channels: `C` is 1 for colour
    /// frames of equal weight.
    width: usize,
    channels: usize,
    weights: &'a [ChannelWeights<C>],
    rejection: &'a Rejection,
    cancel: Option<&'a CancelSignal>,
    counts: &'a [FrameCounts],
    fates: Option<&'a [Vec<AtomicU64>]>,
    /// The first sample of the band in the whole image.
    start: usize,
}

/// A task's working space for one tile.
struct TileScratch {
    /// The frames that reach the tile, in order.
    reaching: Vec<usize>,
    first: Vec<FirstEstimate>,
    kept: Vec<Moments>,
    /// One bit per sample per frame: what pass 2 kept.
    keep: Vec<u64>,
    /// One frame's fates for the tile, a byte each as they are decided.
    codes: Vec<u8>,
    /// The same fates packed, before they join the frame's.
    fates: Vec<u64>,
    counts: Vec<(usize, usize)>,
}

impl Run<'_> {
    fn integrate<const C: usize, const WEIGHTED: bool>(
        &self,
        weights: &[ChannelWeights<C>],
        source: &impl BandSource,
        progress: &mut dyn FnMut(usize, usize),
    ) -> Result<(BatchStackResult, Option<Vec<PackedFates>>)> {
        let (width, height, channels) = self.shape;
        let frame_count = self.frame_count;
        let options = self.options;
        let rejection = Rejection::new(frame_count, options);
        let row_samples = width * channels;
        let samples = row_samples * height;
        let rows = band_rows(self.shape, frame_count, options.band_memory_bytes);
        // A whole number of 64-pixel runs, so keep bits start on a word.
        let tile_pixels = (TILE_BYTES / (frame_count * channels * 4)).clamp(64, 1024) / 64 * 64;
        let tile = tile_pixels * channels;

        let mut outputs = Outputs::new(samples, WEIGHTED);
        let regions = (0..frame_count)
            .map(|index| source.region(index))
            .collect::<Vec<_>>();
        let counts = (0..frame_count)
            .map(|_| FrameCounts::default())
            .collect::<Vec<_>>();
        let fates = self.record_fates.then(|| {
            (0..frame_count)
                .map(|_| {
                    (0..samples.div_ceil(SampleFate::PER_WORD))
                        .map(|_| AtomicU64::new(0))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        });
        // The GPU integrates whole bands when there is one; a drizzle's
        // fates are recorded on the CPU.
        #[cfg(feature = "gpu")]
        let gpu = (!self.record_fates)
            .then(|| {
                let rows = crate::gpu::BandSession::rows(
                    self.shape,
                    frame_count,
                    options.band_memory_bytes,
                    &options.gpu,
                )?;
                let flat: Vec<f32> = weights
                    .iter()
                    .flat_map(|weights| weights.weight.map(|weight| weight as f32))
                    .collect();
                let session = || {
                    crate::gpu::BandSession::new(
                        frame_count,
                        channels,
                        rows * row_samples,
                        WEIGHTED.then_some((flat.as_slice(), C)),
                        rejection.gpu_tables(),
                        &options.gpu,
                    )
                };
                Some((rows, [session()?, session()?]))
            })
            .flatten();
        #[cfg(feature = "gpu")]
        let (rows, mut gpu) = match gpu {
            Some((rows, sessions)) => (rows, Some(sessions)),
            None => (rows, None),
        };
        let bands = height.div_ceil(rows);
        #[cfg(feature = "gpu")]
        let mut pending: Option<(usize, std::ops::Range<usize>)> = None;
        #[cfg(feature = "gpu")]
        let mut gpu_failure = None;
        #[cfg(all(test, feature = "gpu"))]
        let collecting = std::cell::Cell::new(0);
        #[cfg(feature = "gpu")]
        let collect = |session: &mut crate::gpu::BandSession,
                       outputs: &mut Outputs,
                       range: std::ops::Range<usize>| {
            #[cfg(test)]
            {
                let band = collecting.get();
                collecting.set(band + 1);
                if self.fail_gpu_band == Some(band) {
                    return Err("injected band collection failure".into());
                }
            }
            let mut band_counts = vec![(0, 0); frame_count];
            session.collect(outputs.band(range), &mut band_counts)?;
            for (counts, (finite, integrated)) in counts.iter().zip(band_counts) {
                counts.finite.fetch_add(finite, Ordering::Relaxed);
                counts.integrated.fetch_add(integrated, Ordering::Relaxed);
            }
            Ok::<_, String>(())
        };
        let integrate_cpu = |outputs: &mut Outputs, progress: &mut dyn FnMut(usize, usize)| {
            for_each_band(
                self.shape,
                frame_count,
                rows,
                source,
                options.cancel.as_ref(),
                |band, top, frames| {
                    progress(band, bands);
                    check_cancelled(options)?;
                    // A frame the band does not reach was not read, and its
                    // samples are empty.
                    let band_samples = rows.min(height - top) * row_samples;
                    let context = Band {
                        frames,
                        regions: &regions,
                        width,
                        channels,
                        weights,
                        rejection: &rejection,
                        cancel: options.cancel.as_ref(),
                        counts: &counts,
                        fates: fates.as_deref(),
                        start: top * row_samples,
                    };
                    outputs
                        .chunks_mut(context.start..context.start + band_samples, tile)
                        .enumerate()
                        .for_each_init(
                            || TileScratch::new(tile, frame_count),
                            |scratch, (index, mut output)| {
                                context.tile::<WEIGHTED>(scratch, index * tile, &mut output);
                            },
                        );
                    check_cancelled(options)
                },
            )
        };
        #[cfg(not(feature = "gpu"))]
        let integrated = integrate_cpu(&mut outputs, progress);
        #[cfg(feature = "gpu")]
        let integrated = if let Some(sessions) = &mut gpu {
            (0..bands).try_for_each(|band| {
                progress(band, bands);
                check_cancelled(options)?;
                let top = band * rows;
                let band_rows = rows.min(height - top);
                let band_samples = band_rows * row_samples;
                let reaching = (0..frame_count)
                    .filter(|&index| reaches_rows(regions[index], top, band_rows))
                    .collect();
                let slot = band % sessions.len();
                match sessions[slot].submit_from(source, top, band_samples, reaching) {
                    Ok(()) => {}
                    Err(crate::gpu::BandStageError::Source(error)) => return Err(error),
                    Err(crate::gpu::BandStageError::Gpu(reason)) => {
                        gpu_failure = Some(reason.clone());
                        return Err(Error::Stack(reason));
                    }
                }
                // Read and submit this band before collecting its predecessor:
                // the GPU overlaps its copy and compute with bounded cache reads.
                if let Some((previous, range)) = pending.take()
                    && let Err(reason) = collect(&mut sessions[previous], &mut outputs, range)
                {
                    gpu_failure = Some(reason.clone());
                    return Err(Error::Stack(reason));
                }
                pending = Some((slot, top * row_samples..top * row_samples + band_samples));
                check_cancelled(options)
            })
        } else {
            integrate_cpu(&mut outputs, progress)
        };
        #[cfg(feature = "gpu")]
        let integrated = integrated.and_then(|()| {
            check_cancelled(options)?;
            if let Some((slot, range)) = pending.take()
                && let Err(reason) = collect(&mut gpu.as_mut().unwrap()[slot], &mut outputs, range)
            {
                gpu_failure = Some(reason.clone());
                return Err(Error::Stack(reason));
            }
            Ok(())
        });
        #[cfg(feature = "gpu")]
        if let Some(reason) = gpu_failure {
            check_cancelled(options)?;
            eprintln!("seiza-stacking: retrying integration on the CPU: {reason}");
            // An earlier band may still be in flight, and inputs have
            // already been reused. Start afresh so no output or frame count
            // from a failed attempt survives. Sources support repeated reads.
            drop(gpu);
            drop(outputs);
            drop(fates);
            let mut options = options.clone();
            options.gpu.integration = false;
            return Run {
                shape: self.shape,
                frame_count,
                options: &options,
                record_fates: self.record_fates,
                #[cfg(all(test, feature = "gpu"))]
                fail_gpu_band: None,
            }
            .integrate::<C, WEIGHTED>(weights, source, progress);
        }
        integrated?;

        let frames = counts
            .into_iter()
            .map(|counts| BatchFrameDiagnostics {
                finite_samples: counts.finite.into_inner(),
                integrated_samples: counts.integrated.into_inner(),
            })
            .collect();
        let fates = fates.map(|fates| {
            fates
                .into_iter()
                .map(|words| PackedFates {
                    words: words.into_iter().map(AtomicU64::into_inner).collect(),
                    len: samples,
                })
                .collect()
        });
        Ok((
            outputs.finish(width, height, channels, frame_count, frames)?,
            fates,
        ))
    }
}

impl TileScratch {
    fn new(tile: usize, frame_count: usize) -> Self {
        Self {
            reaching: Vec::with_capacity(frame_count),
            first: vec![FirstEstimate::default(); tile],
            kept: vec![Moments::default(); tile],
            keep: vec![0; frame_count * tile.div_ceil(64)],
            codes: vec![0; tile],
            fates: vec![0; tile.div_ceil(SampleFate::PER_WORD) + 1],
            counts: vec![(0, 0); frame_count],
        }
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use crate::{GpuOptions, LinearImage};
    use std::sync::{Arc, atomic::AtomicBool};

    // Odd row and band sizes also exercise four-byte frame boundaries
    // within GPU mappings, whose view offsets must stay eight-byte aligned.
    const SHAPE: (usize, usize, usize) = (385, 513, 3);
    const DEPTH: usize = 12;
    const ROWS: usize = 81;

    struct Source {
        frames: Vec<LinearImage>,
        regions: Vec<ReferenceRegion>,
        fail_at: AtomicUsize,
    }

    impl Source {
        fn new() -> Self {
            let (width, height, channels) = SHAPE;
            let mut frames =
                super::super::tests::synthetic_frames(width, height, channels, DEPTH, 7);
            let regions = (0..DEPTH)
                .map(|index| ReferenceRegion {
                    x: index,
                    y: if index < DEPTH / 2 { 160 } else { 240 },
                    width: width - 2 * index,
                    height: 160,
                })
                .collect::<Vec<_>>();
            for (frame, region) in frames.iter_mut().zip(&regions) {
                for (index, pixel) in frame.data.chunks_exact_mut(channels).enumerate() {
                    let (x, y) = (index % width, index / width);
                    if !(region.x..region.x + region.width).contains(&x)
                        || !(region.y..region.y + region.height).contains(&y)
                    {
                        pixel.fill(f32::NAN);
                    }
                }
            }
            Self {
                frames,
                regions,
                fail_at: AtomicUsize::new(usize::MAX),
            }
        }
    }

    impl BandSource for Source {
        fn read(&self, index: usize, top: usize, band: &mut [f32]) -> Result<()> {
            if index == 0 && top >= self.fail_at.load(Ordering::Relaxed) {
                return Err(Error::Stack("scratch file vanished".into()));
            }
            let start = top * SHAPE.0 * SHAPE.2;
            band.copy_from_slice(&self.frames[index].data[start..start + band.len()]);
            Ok(())
        }

        fn region(&self, index: usize) -> Option<ReferenceRegion> {
            Some(self.regions[index])
        }
    }

    fn options() -> BatchStackOptions {
        BatchStackOptions {
            band_memory_bytes: 2 * ROWS * SHAPE.0 * SHAPE.2 * DEPTH * 4,
            ..BatchStackOptions::default()
        }
    }

    #[test]
    fn gpu_pipeline_reuses_buffers_across_empty_and_partial_bands() {
        if crate::gpu::adapter_description().is_none() {
            return;
        }
        let source = Source::new();
        let (expected, _) = integrate_bands(
            SHAPE,
            DEPTH,
            &BatchStackOptions::default(),
            &source,
            false,
            &mut |_, _| {},
        )
        .unwrap();
        let (actual, _) =
            integrate_bands(SHAPE, DEPTH, &options(), &source, false, &mut |_, _| {}).unwrap();
        super::super::tests::assert_same_result(&actual, &[], &expected, &[]);

        // A source error or cancellation with GPU work pending returns no
        // partial result, and does not turn into a CPU retry.
        let flag = Arc::new(AtomicBool::new(false));
        let cancelled = integrate_bands(
            SHAPE,
            DEPTH,
            &BatchStackOptions {
                cancel: Some(flag.clone().into()),
                ..options()
            },
            &source,
            false,
            &mut |band, _| {
                if band == 3 {
                    flag.store(true, Ordering::Relaxed);
                }
            },
        );
        assert!(matches!(cancelled, Err(Error::Cancelled)));
        source.fail_at.store(240, Ordering::Relaxed);
        let failed =
            integrate_bands(SHAPE, DEPTH, &options(), &source, false, &mut |_, _| {}).unwrap_err();
        assert!(failed.to_string().contains("scratch file vanished"));
    }

    #[test]
    fn a_failed_gpu_pipeline_restarts_without_partial_outputs_or_counts() {
        if crate::gpu::adapter_description().is_none() {
            return;
        }
        let source = Source::new();
        let (expected, _) = integrate_bands(
            SHAPE,
            DEPTH,
            &BatchStackOptions {
                gpu: GpuOptions::cpu(),
                ..options()
            },
            &source,
            false,
            &mut |_, _| {},
        )
        .unwrap();
        let weights = vec![ChannelWeights::<1>::unit(); DEPTH];
        // Fail after partial results have been collected, and while
        // draining the final band after the source reader has finished.
        for fail_gpu_band in [4, SHAPE.1.div_ceil(ROWS) - 1] {
            let mut visited = Vec::new();
            let (actual, _) = Run {
                shape: SHAPE,
                frame_count: DEPTH,
                options: &options(),
                record_fates: false,
                fail_gpu_band: Some(fail_gpu_band),
            }
            .integrate::<1, false>(&weights, &source, &mut |band, _| visited.push(band))
            .unwrap();
            assert_eq!(visited.iter().filter(|&&band| band == 0).count(), 2);
            super::super::tests::assert_same_result(&actual, &[], &expected, &[]);
        }
    }
}

/// Pack fate codes, one byte each, into `words`, the first of them
/// `offset` fates into its word.
fn pack_codes(words: &mut [u64], offset: usize, codes: &[u8]) {
    let head = ((SampleFate::PER_WORD - offset) % SampleFate::PER_WORD).min(codes.len());
    let (head_codes, rest) = codes.split_at(head);
    for (at, &code) in head_codes.iter().enumerate() {
        words[0] |= u64::from(code) << ((offset + at) * SampleFate::BITS);
    }
    // The rest start on a word of their own.
    let first = usize::from(offset > 0);
    for (word, run) in words[first..]
        .iter_mut()
        .zip(rest.chunks(SampleFate::PER_WORD))
    {
        *word |= run.iter().enumerate().fold(0, |bits, (at, &code)| {
            bits | u64::from(code) << (at * SampleFate::BITS)
        });
    }
}

impl<const C: usize> Band<'_, C> {
    /// Whether frame `index` covers any of the `len` samples from sample
    /// `absolute` of the whole image on. Every sample of a frame beyond its
    /// region is `NaN`, which no pass counts, so leaving out a frame that
    /// covers none of a tile changes nothing.
    fn reaches(&self, index: usize, absolute: usize, len: usize) -> bool {
        let Some(region) = self.regions[index] else {
            return true;
        };
        let (first, last) = (
            absolute / self.channels,
            (absolute + len - 1) / self.channels,
        );
        let (top, bottom) = (first / self.width, last / self.width);
        let rows = region.y..region.y + region.height;
        let columns = |from: usize, to: usize| from < region.x + region.width && region.x <= to;
        if top == bottom {
            return rows.contains(&top) && columns(first % self.width, last % self.width);
        }
        (rows.contains(&top) && columns(first % self.width, self.width - 1))
            || (rows.contains(&bottom) && columns(0, last % self.width))
            || (top + 1 < bottom && top + 1 < rows.end && rows.start < bottom)
    }

    /// All three passes over the tile of the band that starts `start`
    /// samples in and whose outputs are `output`.
    fn tile<const WEIGHTED: bool>(
        &self,
        scratch: &mut TileScratch,
        start: usize,
        output: &mut OutputChunk<'_>,
    ) {
        if self.cancel.is_some_and(CancelSignal::is_cancelled) {
            return;
        }
        let len = output.mean.len();
        let rejection = self.rejection;
        // Fates are packed from sample 0 of the whole image; the tile's
        // first and last words can be shared with the neighbouring tiles.
        let absolute = self.start + start;
        scratch.reaching.clear();
        scratch.reaching.extend(
            (0..self.frames.len())
                .filter(|&index| !self.frames[index].is_empty())
                .filter(|&index| self.reaches(index, absolute, len)),
        );
        let reaching = &scratch.reaching;

        let first = &mut scratch.first[..len];
        first.fill(FirstEstimate::default());
        for &index in reaching {
            estimate_chunk(
                first,
                &self.frames[index][start..start + len],
                &self.weights[index],
            );
        }
        for first in first.iter_mut() {
            first.finish(rejection);
        }

        let kept = &mut scratch.kept[..len];
        kept.fill(Moments::default());
        let words = len.div_ceil(64);
        // One run of keep bits for each frame that reaches the tile.
        let keep = &mut scratch.keep[..reaching.len() * words];
        keep.fill(0);
        for (&index, keep) in reaching.iter().zip(keep.chunks_mut(words)) {
            refine_chunk(
                kept,
                first,
                &self.frames[index][start..start + len],
                &self.weights[index],
                rejection,
                keep,
            );
        }

        let (first_word, offset) = (
            absolute / SampleFate::PER_WORD,
            absolute % SampleFate::PER_WORD,
        );
        let fate_words = (offset + len).div_ceil(SampleFate::PER_WORD);
        for (&index, keep) in reaching.iter().zip(keep.chunks(words)) {
            let (frame, weights) = (self.frames[index], &self.weights[index]);
            scratch.counts[index] = match self.fates {
                Some(fates) => {
                    // A byte per fate keeps the stores independent; they
                    // are packed in a run of their own.
                    let codes = &mut scratch.codes[..len];
                    let counts = integrate_chunk::<C, WEIGHTED>(
                        output,
                        kept,
                        &frame[start..start + len],
                        keep,
                        weights,
                        rejection,
                        |at, fate| codes[at] = fate.code() as u8,
                    );
                    let local = &mut scratch.fates[..fate_words];
                    local.fill(0);
                    pack_codes(local, offset, codes);
                    for (word, &bits) in fates[index][first_word..first_word + fate_words]
                        .iter()
                        .zip(local.iter())
                    {
                        if bits != 0 {
                            word.fetch_or(bits, Ordering::Relaxed);
                        }
                    }
                    counts
                }
                None => integrate_chunk::<C, WEIGHTED>(
                    output,
                    kept,
                    &frame[start..start + len],
                    keep,
                    weights,
                    rejection,
                    |_, _| {},
                ),
            };
        }
        for &frame in reaching {
            let (finite, integrated) = scratch.counts[frame];
            self.counts[frame]
                .finite
                .fetch_add(finite, Ordering::Relaxed);
            self.counts[frame]
                .integrated
                .fetch_add(integrated, Ordering::Relaxed);
        }
    }
}
