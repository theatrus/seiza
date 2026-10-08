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
use crate::{CancelSignal, Error, Result};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Frames that can be read a band of rows at a time.
pub(crate) trait BandSource: Sync {
    /// Fill `band`, whole rows of interleaved samples, with frame `index`'s
    /// registered and normalized samples from row `top` on.
    fn read(&self, index: usize, top: usize, band: &mut [f32]) -> Result<()>;
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
/// the band's index and first row, in order. A thread reads the next band
/// while `each` works on this one, so two bands of every frame are held.
/// From a pool thread the reads run in that pool, and the thread only waits
/// for them; see [`crate::tasks`].
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
    let cancelled = || cancel.is_some_and(CancelSignal::is_cancelled);
    crate::tasks::ahead(|ahead| {
        std::thread::scope(|scope| {
            let (requests, reader_requests) = std::sync::mpsc::channel::<(usize, Vec<Vec<f32>>)>();
            let (filled_sender, filled) =
                std::sync::mpsc::channel::<Result<(usize, Vec<Vec<f32>>)>>();
            scope.spawn(move || {
                for (band, mut frames) in reader_requests {
                    let read = if cancelled() {
                        Err(Error::Cancelled)
                    } else {
                        let read;
                        (read, frames) = ahead.run(move || {
                            let read = frames.par_iter_mut().enumerate().try_for_each(
                                |(index, buffer)| {
                                    source.read(
                                        index,
                                        band * rows,
                                        &mut buffer[..band_samples(band)],
                                    )
                                },
                            );
                            (read, frames)
                        });
                        read
                    };
                    if filled_sender.send(read.map(|()| (band, frames))).is_err() {
                        break;
                    }
                }
            });
            for band in 0..bands.min(2) {
                let _ = requests.send((band, vec![vec![0.0_f32; rows * row_samples]; frame_count]));
            }
            for band in 0..bands {
                let (read_band, frames) = ahead
                    .recv(&filled)
                    .map_err(|_| Error::Stack("the band reader stopped early".into()))??;
                debug_assert_eq!(read_band, band);
                if cancelled() {
                    return Err(Error::Cancelled);
                }
                let samples = frames
                    .iter()
                    .map(|frame| &frame[..band_samples(band)])
                    .collect::<Vec<_>>();
                each(band, band * rows, &samples)?;
                drop(samples);
                if band + 2 < bands {
                    let _ = requests.send((band + 2, frames));
                }
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
}

/// Each frame's finite and integrated sample counts.
#[derive(Default)]
struct FrameCounts {
    finite: AtomicUsize,
    integrated: AtomicUsize,
}

/// What every tile of a band shares.
struct Band<'a, const C: usize> {
    frames: &'a [&'a [f32]],
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
        let bands = height.div_ceil(rows);
        // A whole number of 64-pixel runs, so keep bits start on a word.
        let tile_pixels = (TILE_BYTES / (frame_count * channels * 4)).clamp(64, 1024) / 64 * 64;
        let tile = tile_pixels * channels;

        let mut outputs = Outputs::new(samples, WEIGHTED);
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
        for_each_band(
            self.shape,
            frame_count,
            rows,
            source,
            options.cancel.as_ref(),
            |band, top, frames| {
                progress(band, bands);
                let band_samples = frames[0].len();
                let context = Band {
                    frames,
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
        )?;

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
            first: vec![FirstEstimate::default(); tile],
            kept: vec![Moments::default(); tile],
            keep: vec![0; frame_count * tile.div_ceil(64)],
            codes: vec![0; tile],
            fates: vec![0; tile.div_ceil(SampleFate::PER_WORD) + 1],
            counts: vec![(0, 0); frame_count],
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

        let first = &mut scratch.first[..len];
        first.fill(FirstEstimate::default());
        for (frame, weights) in self.frames.iter().zip(self.weights) {
            estimate_chunk(first, &frame[start..start + len], weights);
        }
        for first in first.iter_mut() {
            first.finish(rejection);
        }

        let kept = &mut scratch.kept[..len];
        kept.fill(Moments::default());
        let words = len.div_ceil(64);
        let keep = &mut scratch.keep[..self.frames.len() * words];
        keep.fill(0);
        for ((frame, weights), keep) in self
            .frames
            .iter()
            .zip(self.weights)
            .zip(keep.chunks_mut(words))
        {
            refine_chunk(
                kept,
                first,
                &frame[start..start + len],
                weights,
                rejection,
                keep,
            );
        }

        // Fates are packed from sample 0 of the whole image; the tile's
        // first and last words can be shared with the neighbouring tiles.
        let absolute = self.start + start;
        let (first_word, offset) = (
            absolute / SampleFate::PER_WORD,
            absolute % SampleFate::PER_WORD,
        );
        let fate_words = (offset + len).div_ceil(SampleFate::PER_WORD);
        for (index, ((frame, weights), keep)) in self
            .frames
            .iter()
            .zip(self.weights)
            .zip(keep.chunks(words))
            .enumerate()
        {
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
        for (frame, &(finite, integrated)) in scratch.counts.iter().enumerate() {
            self.counts[frame]
                .finite
                .fetch_add(finite, Ordering::Relaxed);
            self.counts[frame]
                .integrated
                .fetch_add(integrated, Ordering::Relaxed);
        }
    }
}
