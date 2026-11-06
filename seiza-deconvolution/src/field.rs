//! Deconvolution towards a round target PSF, with a PSF that varies across
//! the field.
//!
//! A [`PsfField`] gives the PSF `P` anywhere in the image. The target `T` is
//! a round Gaussian, never narrower than [`MINIMUM_TARGET_FWHM`]. Between
//! them is the remaining kernel `R`, with `P = T ∗ R`: the blur that taking
//! the image from `T` to `P` would add. [`RemainingKernel`] holds `R` on a
//! grid of nodes, deconvolving each node's stamp by `T` with Richardson–Lucy
//! and FFT convolutions, and applies it across the image with each source
//! pixel blurred by its own interpolated kernel.
//!
//! [`deconvolve_field`] runs damped Richardson–Lucy with `R` as the forward
//! model, so its estimate is the image at `T`. [`reblur_consistency`] checks
//! any such estimate against the input: blurred by `R`, it must give the
//! input back to within the input's noise.

use rayon::prelude::*;
use rustfft::num_complex::Complex32;
use seiza_stars::{PsfField, PsfStamp};
use serde::{Deserialize, Serialize};

use crate::fft::{Fft2, smooth_length};
use crate::{
    Blur, DeconvolutionError, DeconvolutionResult, RestoreSettings, Result, reflect,
    restore_channel,
};

/// The narrowest target FWHM in pixels. A narrower target asks for detail
/// finer than the sampling holds.
pub const MINIMUM_TARGET_FWHM: f32 = 1.5;

/// The node spacing [`RemainingKernel::new`] suits most images with, in
/// pixels.
pub const DEFAULT_NODE_SPACING: usize = 256;

/// The round Gaussian PSF a field-varying deconvolution aims at.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TargetPsf {
    /// One FWHM in pixels everywhere, at least [`MINIMUM_TARGET_FWHM`].
    Fwhm(f32),
    /// A fraction, in `(0, 1]`, of the local PSF's FWHM, never below
    /// [`MINIMUM_TARGET_FWHM`]: the same relative sharpening everywhere.
    Ratio(f32),
    /// The field's narrowest local FWHM, everywhere. This corrects tilt,
    /// coma and elongation across the field without sharpening its best
    /// part.
    BestRound,
}

impl TargetPsf {
    /// Check the target's number, if it has one.
    pub fn validate(self) -> Result<()> {
        match self {
            Self::Fwhm(fwhm) if !fwhm.is_finite() || fwhm < MINIMUM_TARGET_FWHM => {
                Err(DeconvolutionError::Invalid(format!(
                    "target FWHM must be finite and at least {MINIMUM_TARGET_FWHM} pixels"
                )))
            }
            Self::Ratio(ratio) if !ratio.is_finite() || ratio <= 0.0 || ratio > 1.0 => Err(
                DeconvolutionError::Invalid("target ratio must be in (0, 1]".into()),
            ),
            _ => Ok(()),
        }
    }
}

/// The remaining kernel `R`, with `P = T ∗ R`, on a grid of nodes across
/// the image, for each channel.
#[derive(Clone, Debug)]
pub struct RemainingKernel {
    width: usize,
    height: usize,
    channels: usize,
    size: usize,
    spacing: usize,
    /// The nodes' columns and rows.
    xs: Axis,
    ys: Axis,
    /// Node-major, then channel: `size × size` samples summing to one.
    kernels: Vec<Vec<f32>>,
    /// The local PSF's FWHM and the target's, node-major then channel.
    psf_fwhm: Vec<f32>,
    target_fwhm: Vec<f32>,
}

impl RemainingKernel {
    /// The remaining kernel between `field`'s PSF and `target`, sampled at
    /// nodes evenly spaced at most `node_spacing` pixels apart (at least 16;
    /// less when the image is smaller), the first and last on the image's
    /// edges. Where the target is as wide as the PSF, or wider, `R` is
    /// close to a single pixel and nothing is sharpened.
    pub fn new(field: &PsfField, target: TargetPsf, node_spacing: usize) -> Result<Self> {
        target.validate()?;
        if node_spacing < 16 {
            return Err(DeconvolutionError::Invalid(
                "node spacing must be at least 16 pixels".into(),
            ));
        }
        let (width, height, channels) = (field.width(), field.height(), field.channels());
        let size = field.stamp_size();
        let spacing = node_spacing.min(width.max(height).saturating_sub(1)).max(1);
        let (xs, ys) = (Axis::new(width, spacing), Axis::new(height, spacing));
        let nodes: Vec<(f64, f64)> = ys
            .positions
            .iter()
            .flat_map(|&y| xs.positions.iter().map(move |&x| (x as f64, y as f64)))
            .collect();
        // Each node's PSF, per channel, and its FWHM.
        let stamps: Vec<(Vec<Vec<f32>>, Vec<f32>)> = nodes
            .par_iter()
            .map(|&(x, y)| {
                let stamp = field.stamp_at(x, y);
                let planes: Vec<Vec<f32>> = (0..channels).map(|c| stamp.plane(c)).collect();
                let fwhm: Vec<f32> = (0..channels)
                    .map(|c| {
                        stamp.shape(c).map_or_else(
                            || moment_fwhm(&planes[c], size),
                            |shape| shape.fwhm as f32,
                        )
                    })
                    .collect();
                (planes, fwhm)
            })
            .collect();
        let psf_fwhm: Vec<f32> = stamps.iter().flat_map(|(_, fwhm)| fwhm.clone()).collect();
        let best: Vec<f32> = (0..channels)
            .map(|c| {
                stamps
                    .iter()
                    .map(|(_, fwhm)| fwhm[c])
                    .fold(f32::INFINITY, f32::min)
            })
            .collect();
        let target_fwhm: Vec<f32> = stamps
            .iter()
            .flat_map(|(_, fwhm)| {
                (0..channels).map(|c| {
                    match target {
                        TargetPsf::Fwhm(value) => value,
                        TargetPsf::Ratio(ratio) => ratio * fwhm[c],
                        TargetPsf::BestRound => best[c],
                    }
                    .max(MINIMUM_TARGET_FWHM)
                })
            })
            .collect();
        let divider = Divider::new(size);
        let kernels: Vec<Vec<f32>> = stamps
            .par_iter()
            .enumerate()
            .flat_map_iter(|(node, (planes, _))| {
                let divider = &divider;
                let (psf_fwhm, target_fwhm) = (&psf_fwhm, &target_fwhm);
                planes.iter().enumerate().map(move |(c, plane)| {
                    let index = node * channels + c;
                    divider.remaining(plane, psf_fwhm[index], target_fwhm[index])
                })
            })
            .collect();
        Ok(Self {
            width,
            height,
            channels,
            size,
            spacing,
            xs,
            ys,
            kernels,
            psf_fwhm,
            target_fwhm,
        })
    }

    /// The image width the kernel covers.
    pub fn width(&self) -> usize {
        self.width
    }

    /// The image height the kernel covers.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Channels, one or three, each with its own kernel.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// The side of each node's kernel, in pixels; the PSF field's stamp
    /// size.
    pub fn kernel_size(&self) -> usize {
        self.size
    }

    /// The most the nodes are apart, in pixels. They are evenly spaced, with
    /// the first and last columns and rows on the image's edges.
    pub fn node_spacing(&self) -> usize {
        self.spacing
    }

    /// The number of nodes.
    pub fn node_count(&self) -> usize {
        self.xs.len() * self.ys.len()
    }

    /// The narrowest and widest local PSF FWHM over the nodes of one
    /// channel, or `None` for a channel out of range.
    pub fn psf_fwhm_range(&self, channel: usize) -> Option<(f32, f32)> {
        self.range(&self.psf_fwhm, channel)
    }

    /// The narrowest and widest target FWHM over the nodes of one channel,
    /// or `None` for a channel out of range.
    pub fn target_fwhm_range(&self, channel: usize) -> Option<(f32, f32)> {
        self.range(&self.target_fwhm, channel)
    }

    fn range(&self, values: &[f32], channel: usize) -> Option<(f32, f32)> {
        if channel >= self.channels {
            return None;
        }
        Some(
            values
                .iter()
                .skip(channel)
                .step_by(self.channels)
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), &v| {
                    (low.min(v), high.max(v))
                }),
        )
    }

    /// The kernel at pixel `(x, y)` in one channel, blended from the nodes
    /// around it as the convolution blends them: `kernel_size²` samples
    /// summing to one, centred on the middle pixel. `None` for a channel
    /// out of range.
    pub fn kernel_at(&self, channel: usize, x: f64, y: f64) -> Option<Vec<f32>> {
        if channel >= self.channels {
            return None;
        }
        let mut kernel = vec![0.0; self.size * self.size];
        for row in 0..self.ys.len() {
            let wy = self.ys.weight(row, y);
            if wy == 0.0 {
                continue;
            }
            for column in 0..self.xs.len() {
                let w = wy * self.xs.weight(column, x);
                if w == 0.0 {
                    continue;
                }
                let node = &self.kernels[(row * self.xs.len() + column) * self.channels + channel];
                for (sum, value) in kernel.iter_mut().zip(node) {
                    *sum += w as f32 * value;
                }
            }
        }
        Some(kernel)
    }

    fn check_length(&self, length: usize, name: &str) -> Result<()> {
        let expected = self.width * self.height * self.channels;
        if length == expected {
            Ok(())
        } else {
            Err(DeconvolutionError::Invalid(format!(
                "{name} has {length} samples; expected {expected}"
            )))
        }
    }

    /// `R ∗ image`: the image blurred by the remaining kernel, each source
    /// pixel by its own. Non-finite samples are left out and stay
    /// non-finite, and the blur near them is renormalised over the samples
    /// there are.
    pub fn blur(&self, data: &[f32]) -> Result<Vec<f32>> {
        self.check_length(data.len(), "image")?;
        let mut output = vec![0.0; data.len()];
        for channel in 0..self.channels {
            let blur = VaryingBlur::new(self, channel);
            let plane = extract(data, self.channels, channel);
            let finite = plane.iter().all(|v| v.is_finite());
            let blurred = if finite {
                let mut out = vec![0.0; plane.len()];
                blur.forward_into(&plane, &mut out);
                out
            } else {
                let mask: Vec<f32> = plane
                    .iter()
                    .map(|v| if v.is_finite() { 1.0 } else { 0.0 })
                    .collect();
                let zeroed: Vec<f32> = plane
                    .iter()
                    .map(|v| if v.is_finite() { *v } else { 0.0 })
                    .collect();
                let (mut out, mut support) = (vec![0.0; plane.len()], vec![0.0; plane.len()]);
                blur.forward_into(&zeroed, &mut out);
                blur.forward_into(&mask, &mut support);
                out.iter_mut().zip(&support).zip(&plane).for_each(
                    |((value, &support), original)| {
                        *value = if original.is_finite() {
                            *value / support.max(f32::MIN_POSITIVE)
                        } else {
                            f32::NAN
                        };
                    },
                );
                out
            };
            insert(&mut output, &blurred, self.channels, channel);
        }
        Ok(output)
    }
}

/// The nodes along one axis, evenly spaced and at most `spacing` apart,
/// the first on the first pixel and the last on the last: each node is
/// weighted where its PSF was measured, and no interval is much shorter
/// than the others, which would make the weights change steeply there.
#[derive(Clone, Debug)]
struct Axis {
    positions: Vec<usize>,
}

impl Axis {
    fn new(length: usize, spacing: usize) -> Self {
        let last = length.saturating_sub(1);
        let intervals = last.div_ceil(spacing.max(1));
        let positions = if intervals == 0 {
            vec![0]
        } else {
            (0..=intervals)
                .map(|index| ((index * last) as f64 / intervals as f64).round() as usize)
                .collect()
        };
        Self { positions }
    }

    fn len(&self) -> usize {
        self.positions.len()
    }

    /// Node `index`'s linear interpolation weight at `position`, taken
    /// inside the axis: the weights of all nodes sum to one everywhere, and
    /// the sky beyond an edge belongs to the edge's node.
    fn weight(&self, index: usize, position: f64) -> f64 {
        let last = *self.positions.last().unwrap_or(&0) as f64;
        let position = position.clamp(0.0, last);
        let at = self.positions[index] as f64;
        if position == at {
            return 1.0;
        }
        if position > at {
            match self.positions.get(index + 1) {
                Some(&next) if position < next as f64 => {
                    (next as f64 - position) / (next as f64 - at)
                }
                _ => 0.0,
            }
        } else if index > 0 {
            let previous = self.positions[index - 1] as f64;
            if position > previous {
                (position - previous) / (at - previous)
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
}

fn extract(data: &[f32], channels: usize, channel: usize) -> Vec<f32> {
    if channels == 1 {
        return data.to_vec();
    }
    data.iter()
        .skip(channel)
        .step_by(channels)
        .copied()
        .collect()
}

fn insert(data: &mut [f32], plane: &[f32], channels: usize, channel: usize) {
    for (index, value) in plane.iter().enumerate() {
        data[index * channels + channel] = *value;
    }
}

/// A Gaussian-equivalent FWHM from a stamp's second moments, for a stamp
/// whose profile a Moffat cannot fit.
fn moment_fwhm(stamp: &[f32], size: usize) -> f32 {
    let centre = (size / 2) as f64;
    let (mut sum, mut second) = (0.0, 0.0);
    for (index, &value) in stamp.iter().enumerate() {
        let value = f64::from(value.max(0.0));
        let (dx, dy) = (
            (index % size) as f64 - centre,
            (index / size) as f64 - centre,
        );
        sum += value;
        second += value * (dx * dx + dy * dy);
    }
    if sum <= 0.0 {
        return MINIMUM_TARGET_FWHM;
    }
    (2.354_820_045 * (second / sum / 2.0).sqrt()) as f32
}

/// A pixel-integrated round Gaussian of `fwhm` pixels, `size × size`,
/// centred on the middle pixel and summing to one.
pub(crate) fn gaussian_stamp(fwhm: f32, size: usize) -> Vec<f32> {
    PsfStamp::gaussian(f64::from(fwhm), size).data
}

/// Regularised deconvolution of a PSF stamp by a round Gaussian.
///
/// A Tikhonov division in Fourier space rings when the target is nearly as
/// wide as the PSF: the remaining kernel is then nearly a point, whose
/// spectrum the division cuts off sharply. Richardson–Lucy on the stamp
/// itself, with FFT convolutions, keeps the kernel positive and has no such
/// edge; stopping after a fixed number of iterations regularises it.
struct Divider {
    size: usize,
    fft: Fft2,
}

/// Richardson–Lucy iterations of the division.
const DIVISION_ITERATIONS: usize = 120;

impl Divider {
    fn new(size: usize) -> Self {
        let length = smooth_length(2 * size);
        Self {
            size,
            fft: Fft2::new(length, length),
        }
    }

    /// `R` with `P = T ∗ R` for the stamp `psf`, of FWHM `psf_fwhm`, and a
    /// target of `target_fwhm`: positive, summing to one, from a Gaussian
    /// start of the variance the two Gaussians would leave.
    fn remaining(&self, psf: &[f32], psf_fwhm: f32, target_fwhm: f32) -> Vec<f32> {
        let (n, length) = (self.size, self.fft.len());
        let scale = 1.0 / length as f32;
        let mut scratch = vec![Complex32::default(); length];
        let mut target = self.centred(&gaussian_stamp(target_fwhm, n));
        self.fft.forward(&mut target, &mut scratch);
        let convolve = |values: &[f32], adjoint: bool, scratch: &mut [Complex32]| -> Vec<f32> {
            let mut tile = self.centred(values);
            self.fft.forward(&mut tile, scratch);
            for (value, t) in tile.iter_mut().zip(&target) {
                *value *= if adjoint { t.conj() } else { *t };
            }
            self.fft.inverse(&mut tile, scratch);
            self.window(&tile).into_iter().map(|v| v * scale).collect()
        };
        let mut observed: Vec<f32> = psf.iter().map(|v| v.max(0.0)).collect();
        let total: f32 = observed.iter().sum();
        if total <= 0.0 {
            return point(n);
        }
        observed.iter_mut().for_each(|v| *v /= total);
        // The stamp's window, seen through the target: what each kernel
        // sample can reach of it.
        let support = convolve(&vec![1.0; n * n], true, &mut scratch);
        const FWHM_PER_SIGMA: f32 = 2.354_82;
        let variance = |fwhm: f32| (fwhm / FWHM_PER_SIGMA).powi(2);
        let start = (variance(psf_fwhm) - variance(target_fwhm))
            .max(0.04)
            .sqrt()
            * FWHM_PER_SIGMA;
        let mut kernel = gaussian_stamp(start, n);
        for _ in 0..DIVISION_ITERATIONS {
            let model = convolve(&kernel, false, &mut scratch);
            let ratio: Vec<f32> = observed
                .iter()
                .zip(&model)
                .map(|(&p, &m)| if m > 1e-20 { p / m } else { 0.0 })
                .collect();
            let correction = convolve(&ratio, true, &mut scratch);
            for ((value, c), s) in kernel.iter_mut().zip(&correction).zip(&support) {
                *value = (*value * c / s.max(1e-12)).max(0.0);
            }
        }
        let total: f32 = kernel.iter().sum();
        if total > 0.0 && total.is_finite() {
            kernel.iter_mut().for_each(|v| *v /= total);
            kernel
        } else {
            point(n)
        }
    }

    /// The `size × size` window around the origin of an FFT tile, as real
    /// samples centred on the middle pixel.
    fn window(&self, tile: &[Complex32]) -> Vec<f32> {
        let (n, half) = (self.size as isize, (self.size / 2) as isize);
        let (width, height) = (self.fft.width as isize, self.fft.height as isize);
        (0..n * n)
            .map(|index| {
                let (dx, dy) = (index % n - half, index / n - half);
                tile[(dy.rem_euclid(height) * width + dx.rem_euclid(width)) as usize].re
            })
            .collect()
    }

    /// `stamp` in an FFT tile with its centre at the origin, wrapped.
    fn centred(&self, stamp: &[f32]) -> Vec<Complex32> {
        let mut tile = vec![Complex32::default(); self.fft.len()];
        place_centred(&mut tile, stamp, self.size, self.fft.width, self.fft.height);
        tile
    }
}

/// A `size × size` kernel that is a single pixel.
fn point(size: usize) -> Vec<f32> {
    let mut kernel = vec![0.0; size * size];
    kernel[(size / 2) * size + size / 2] = 1.0;
    kernel
}

/// Write a `size × size` kernel centred on its middle pixel into a
/// `width × height` tile with that centre at the origin.
fn place_centred(tile: &mut [Complex32], kernel: &[f32], size: usize, width: usize, height: usize) {
    let half = (size / 2) as isize;
    for (index, &value) in kernel.iter().enumerate() {
        let dx = (index % size) as isize - half;
        let dy = (index / size) as isize - half;
        let at = dy.rem_euclid(height as isize) as usize * width
            + dx.rem_euclid(width as isize) as usize;
        tile[at] = Complex32::new(value, 0.0);
    }
}

/// The most memory a channel's node spectra are kept in, in bytes. A
/// larger image transforms each node's kernel again for each blur instead:
/// a 9576 × 6388 stack at the default spacing has 1014 nodes, whose
/// spectra would take 2.7 GB.
const SPECTRA_BYTES: usize = 256 << 20;

/// One channel of a remaining kernel, ready to convolve: each node's kernel
/// transformed at the size of the tile it is applied over, kept when they
/// fit [`SPECTRA_BYTES`].
struct VaryingBlur<'a> {
    kernel: &'a RemainingKernel,
    channel: usize,
    margin: isize,
    fft: Fft2,
    spectra: Option<Vec<Vec<Complex32>>>,
}

impl<'a> VaryingBlur<'a> {
    fn new(kernel: &'a RemainingKernel, channel: usize) -> Self {
        let margin = (kernel.size / 2) as isize;
        let span = 2 * kernel.spacing + 1 + 2 * margin as usize;
        let fft = Fft2::new(smooth_length(span), smooth_length(span));
        let mut blur = Self {
            kernel,
            channel,
            margin,
            fft,
            spectra: None,
        };
        let bytes = kernel.node_count() * blur.fft.len() * std::mem::size_of::<Complex32>();
        if bytes <= SPECTRA_BYTES {
            let spectra = (0..kernel.node_count())
                .into_par_iter()
                .map(|node| {
                    let mut spectrum = vec![Complex32::default(); blur.fft.len()];
                    let mut scratch = vec![Complex32::default(); blur.fft.len()];
                    blur.transform_kernel(node, &mut spectrum, &mut scratch);
                    spectrum
                })
                .collect();
            blur.spectra = Some(spectra);
        }
        blur
    }

    /// Node `node`'s kernel, transformed at the tile size, into `spectrum`.
    fn transform_kernel(&self, node: usize, spectrum: &mut [Complex32], scratch: &mut [Complex32]) {
        spectrum.fill(Complex32::default());
        let kernel = self.kernel;
        place_centred(
            spectrum,
            &kernel.kernels[node * kernel.channels + self.channel],
            kernel.size,
            self.fft.width,
            self.fft.height,
        );
        // Only the rows the kernel wraps onto hold samples.
        let half = kernel.size / 2;
        let rows: Vec<usize> = (0..=half)
            .chain(self.fft.height - half..self.fft.height)
            .collect();
        self.fft.forward_sparse_rows(spectrum, scratch, &rows);
    }

    /// The tile origin of node `(column, row)` in image coordinates.
    fn origin(&self, column: usize, row: usize) -> (isize, isize) {
        let spacing = self.kernel.spacing as isize;
        (
            self.kernel.xs.positions[column] as isize - spacing - self.margin,
            self.kernel.ys.positions[row] as isize - spacing - self.margin,
        )
    }

    /// Run `node_tile` for every node, nine interleaved passes so that the
    /// tiles of one pass never overlap, and add each pass's tiles into
    /// `output` with `place`. `node_tile` is handed the node's kernel
    /// spectrum. A pass holds only a few tiles per thread at a time.
    fn run(
        &self,
        output: &mut [f32],
        node_tile: impl Fn(usize, usize, &[Complex32], &mut [Complex32], &mut [Complex32]) + Sync,
        place: impl Fn(usize, usize, &[Complex32], &mut [f32]),
    ) {
        output.fill(0.0);
        let (columns, rows) = (self.kernel.xs.len(), self.kernel.ys.len());
        let batch = 2 * rayon::current_num_threads().max(1);
        for pass in 0..9 {
            let nodes: Vec<(usize, usize)> = (0..rows)
                .filter(|row| row % 3 == pass / 3)
                .flat_map(|row| {
                    (0..columns)
                        .filter(move |column| column % 3 == pass % 3)
                        .map(move |column| (column, row))
                })
                .collect();
            for chunk in nodes.chunks(batch) {
                // Each thread keeps its scratch and spectrum buffers.
                let tiles: Vec<(usize, usize, Vec<Complex32>)> = chunk
                    .par_iter()
                    .map_init(
                        || {
                            (
                                vec![Complex32::default(); self.fft.len()],
                                Vec::<Complex32>::new(),
                            )
                        },
                        |(scratch, spectrum), &(column, row)| {
                            let node = row * columns + column;
                            let mut tile = vec![Complex32::default(); self.fft.len()];
                            match &self.spectra {
                                Some(spectra) => {
                                    node_tile(column, row, &spectra[node], &mut tile, scratch)
                                }
                                None => {
                                    spectrum.resize(self.fft.len(), Complex32::default());
                                    self.transform_kernel(node, spectrum, scratch);
                                    node_tile(column, row, spectrum, &mut tile, scratch);
                                }
                            }
                            (column, row, tile)
                        },
                    )
                    .collect();
                for (column, row, tile) in tiles {
                    place(column, row, &tile, output);
                }
            }
        }
    }

    /// `output = Σ_k R_k ∗ (w_k · input)`: each source pixel blurred by its
    /// own interpolated kernel. Reflection supplies the sky beyond the
    /// edges.
    fn forward_into(&self, input: &[f32], output: &mut [f32]) {
        let (width, height) = (self.kernel.width as isize, self.kernel.height as isize);
        let (fw, fh) = (self.fft.width, self.fft.height);
        let (xs, ys) = (&self.kernel.xs, &self.kernel.ys);
        let scale = 1.0 / (fw * fh) as f32;
        let margin = self.margin;
        self.run(
            output,
            |column, row, spectrum, tile, scratch| {
                let (ox, oy) = self.origin(column, row);
                // The weights are taken inside the image, so the reflected
                // sky beyond an edge belongs to the edge's nodes and still
                // sums to one.
                for ty in 0..fh as isize {
                    let y = oy + ty;
                    if y < -margin || y > height - 1 + margin {
                        continue;
                    }
                    let wy = ys.weight(row, y as f64) as f32;
                    if wy == 0.0 {
                        continue;
                    }
                    let source_row = reflect(y, height as usize) * width as usize;
                    for tx in 0..fw as isize {
                        let x = ox + tx;
                        if x < -margin || x > width - 1 + margin {
                            continue;
                        }
                        let w = wy * xs.weight(column, x as f64) as f32;
                        if w != 0.0 {
                            let value = input[source_row + reflect(x, width as usize)];
                            tile[ty as usize * fw + tx as usize] = Complex32::new(w * value, 0.0);
                        }
                    }
                }
                self.fft.forward(tile, scratch);
                for (value, k) in tile.iter_mut().zip(spectrum) {
                    *value *= k;
                }
                self.fft.inverse(tile, scratch);
            },
            |column, row, tile, output| {
                let (ox, oy) = self.origin(column, row);
                for ty in 0..fh as isize {
                    let y = oy + ty;
                    if y < 0 || y >= height {
                        continue;
                    }
                    for tx in 0..fw as isize {
                        let x = ox + tx;
                        if x < 0 || x >= width {
                            continue;
                        }
                        output[(y * width + x) as usize] +=
                            tile[ty as usize * fw + tx as usize].re * scale;
                    }
                }
            },
        );
    }

    /// `output = Σ_k w_k · (R̃_k ∗ input)`, the adjoint of
    /// [`Self::forward_into`] away from the edges.
    fn adjoint_into(&self, input: &[f32], output: &mut [f32]) {
        let (width, height) = (self.kernel.width as isize, self.kernel.height as isize);
        let (fw, fh) = (self.fft.width, self.fft.height);
        let (xs, ys) = (&self.kernel.xs, &self.kernel.ys);
        let scale = 1.0 / (fw * fh) as f32;
        let margin = self.margin;
        self.run(
            output,
            |column, row, spectrum, tile, scratch| {
                let (ox, oy) = self.origin(column, row);
                for ty in 0..fh as isize {
                    let y = oy + ty;
                    if y < -margin || y > height - 1 + margin {
                        continue;
                    }
                    let source_row = reflect(y, height as usize) * width as usize;
                    for tx in 0..fw as isize {
                        let x = ox + tx;
                        if x < -margin || x > width - 1 + margin {
                            continue;
                        }
                        tile[ty as usize * fw + tx as usize] =
                            Complex32::new(input[source_row + reflect(x, width as usize)], 0.0);
                    }
                }
                self.fft.forward(tile, scratch);
                for (value, k) in tile.iter_mut().zip(spectrum) {
                    *value *= k.conj();
                }
                self.fft.inverse(tile, scratch);
            },
            |column, row, tile, output| {
                let (ox, oy) = self.origin(column, row);
                for ty in 0..fh as isize {
                    let y = oy + ty;
                    if y < 0 || y >= height {
                        continue;
                    }
                    let wy = ys.weight(row, y as f64) as f32;
                    if wy == 0.0 {
                        continue;
                    }
                    for tx in 0..fw as isize {
                        let x = ox + tx;
                        if x < 0 || x >= width {
                            continue;
                        }
                        let w = wy * xs.weight(column, x as f64) as f32;
                        output[(y * width + x) as usize] +=
                            w * tile[ty as usize * fw + tx as usize].re * scale;
                    }
                }
            },
        );
    }
}

impl Blur for VaryingBlur<'_> {
    fn forward(&self, input: &[f32], output: &mut [f32], _scratch: &mut [f32]) {
        self.forward_into(input, output);
    }

    fn adjoint(&self, input: &[f32], output: &mut [f32], _scratch: &mut [f32]) {
        self.adjoint_into(input, output);
    }

    fn symmetric(&self) -> bool {
        false
    }
}

/// Settings for [`deconvolve_field`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldDeconvolutionConfig {
    /// Richardson–Lucy update count. Small values are intentionally
    /// preferred.
    pub iterations: usize,
    /// Blend of the restored estimate into the original, in `[0, 1]`.
    pub amount: f32,
    /// Residual damping floor as a fraction of each channel's sample range.
    pub noise_fraction: f32,
    /// Per-iteration multiplicative correction limit, greater than `1`.
    pub max_correction: f32,
}

impl FieldDeconvolutionConfig {
    /// The conservative settings of the fixed-PSF prototype: four
    /// iterations, a 35% blend, a 0.1% noise floor and corrections within
    /// a factor of two.
    pub fn conservative() -> Self {
        Self {
            iterations: 4,
            amount: 0.35,
            noise_fraction: 0.001,
            max_correction: 2.0,
        }
    }

    /// Check that every setting lies in its supported range.
    pub fn validate(self) -> Result<()> {
        RestoreSettings::from(self).validate()
    }
}

impl From<FieldDeconvolutionConfig> for RestoreSettings {
    fn from(config: FieldDeconvolutionConfig) -> Self {
        Self {
            iterations: config.iterations,
            amount: config.amount,
            noise_fraction: config.noise_fraction,
            max_correction: config.max_correction,
        }
    }
}

/// Deconvolve a linear mono or interleaved RGB image towards the target of
/// `kernel`, with damped Richardson–Lucy and `kernel` as the forward model:
/// the estimate is the image as it would be with the target PSF. The
/// fixed-PSF prototype's guardrails all apply: the correction is clamped,
/// updates near the noise floor are damped, each channel keeps its flux,
/// and only `amount` of the estimate is blended in. The image must be the
/// size `kernel` was made for, and its samples finite; use
/// [`deconvolve_field_masked`] for registered images with missing borders.
pub fn deconvolve_field(
    data: &[f32],
    kernel: &RemainingKernel,
    config: &FieldDeconvolutionConfig,
) -> Result<DeconvolutionResult> {
    deconvolve_field_impl(data, kernel, config, false)
}

/// [`deconvolve_field`], treating non-finite samples as a missing-data
/// mask that stays in the output.
pub fn deconvolve_field_masked(
    data: &[f32],
    kernel: &RemainingKernel,
    config: &FieldDeconvolutionConfig,
) -> Result<DeconvolutionResult> {
    deconvolve_field_impl(data, kernel, config, true)
}

fn deconvolve_field_impl(
    data: &[f32],
    kernel: &RemainingKernel,
    config: &FieldDeconvolutionConfig,
    allow_masked: bool,
) -> Result<DeconvolutionResult> {
    let settings = RestoreSettings::from(*config);
    settings.validate()?;
    kernel.check_length(data.len(), "image")?;
    let channels = kernel.channels;
    let mut output = vec![0.0; data.len()];
    let mut diagnostics = Vec::with_capacity(channels);
    for channel in 0..channels {
        let blur = VaryingBlur::new(kernel, channel);
        let plane = extract(data, channels, channel);
        let restored = restore_channel(&plane, &blur, &settings, allow_masked)?;
        insert(&mut output, &restored.data, channels, channel);
        diagnostics.push(restored.diagnostics);
    }
    Ok(DeconvolutionResult {
        data: output,
        channels: diagnostics,
    })
}

/// Bands the residual's power spectrum is reported in, from the lowest
/// frequency to Nyquist.
pub const SPECTRUM_BANDS: usize = 8;

/// Side of the tiles the residual's spectrum is averaged over.
const SPECTRUM_TILE: usize = 128;

/// How well one channel of an estimate, blurred by the remaining kernel,
/// gives back its input.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChannelConsistency {
    /// The mean of `((R ∗ output − input) / σ)²` over the samples with a
    /// finite, positive noise: 1 when the residual is the input's noise.
    pub reduced_chi_squared: f64,
    /// The samples the χ² was measured over.
    pub samples: usize,
    /// The normalised residual's power in [`SPECTRUM_BANDS`] radial bands of
    /// frequency, lowest first, each relative to their mean: all 1 for a
    /// flat spectrum. Empty when the image holds no whole tile of measured
    /// samples.
    pub band_power: Vec<f64>,
    /// The geometric mean of the band powers over their arithmetic mean:
    /// 1 for a flat spectrum, lower as power gathers in some bands. `None`
    /// when `band_power` is empty.
    pub spectral_flatness: Option<f64>,
}

/// The reblur-consistency check of an estimate at the target PSF.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReblurConsistency {
    /// One entry per channel.
    pub channels: Vec<ChannelConsistency>,
}

/// Check an estimate at `kernel`'s target against the image it came from.
/// `noise` holds the input's per-sample noise as a standard deviation, in
/// the image's layout; a sample whose noise is not finite and positive is
/// left out, which is how saturated cores are masked. Blurred by the
/// remaining kernel, the estimate must give back the input to within that
/// noise: a reduced χ² near 1 and a flat residual spectrum. Structure in
/// the residual is detail the estimate added or lost that the data do not
/// support. It needs no truth.
pub fn reblur_consistency(
    output: &[f32],
    input: &[f32],
    noise: &[f32],
    kernel: &RemainingKernel,
) -> Result<ReblurConsistency> {
    kernel.check_length(output.len(), "estimate")?;
    kernel.check_length(input.len(), "input")?;
    kernel.check_length(noise.len(), "noise map")?;
    let reblurred = kernel.blur(output)?;
    let (width, height, channels) = (kernel.width, kernel.height, kernel.channels);
    let consistency = (0..channels)
        .map(|channel| {
            // The residual in noise units; NaN where it cannot be measured.
            let normalised: Vec<f32> = (0..width * height)
                .map(|pixel| {
                    let index = pixel * channels + channel;
                    let (model, actual, sigma) = (reblurred[index], input[index], noise[index]);
                    if model.is_finite() && actual.is_finite() && sigma.is_finite() && sigma > 0.0 {
                        (model - actual) / sigma
                    } else {
                        f32::NAN
                    }
                })
                .collect();
            let (sum, samples) = normalised
                .iter()
                .filter(|v| v.is_finite())
                .fold((0.0_f64, 0usize), |(sum, count), &v| {
                    (sum + f64::from(v) * f64::from(v), count + 1)
                });
            let band_power = residual_spectrum(&normalised, width, height);
            let spectral_flatness = if band_power.is_empty() {
                None
            } else {
                let arithmetic = band_power.iter().sum::<f64>() / band_power.len() as f64;
                let geometric = (band_power.iter().map(|p| p.max(1e-300).ln()).sum::<f64>()
                    / band_power.len() as f64)
                    .exp();
                Some(geometric / arithmetic)
            };
            ChannelConsistency {
                reduced_chi_squared: if samples > 0 {
                    sum / samples as f64
                } else {
                    f64::NAN
                },
                samples,
                band_power,
                spectral_flatness,
            }
        })
        .collect();
    Ok(ReblurConsistency {
        channels: consistency,
    })
}

/// The radially banded power spectrum of `residual`, averaged over the
/// Hann-windowed tiles that hold no NaN, each band relative to the bands'
/// mean.
fn residual_spectrum(residual: &[f32], width: usize, height: usize) -> Vec<f64> {
    let tile = SPECTRUM_TILE.min(width).min(height);
    if tile < 16 {
        return Vec::new();
    }
    let fft = Fft2::new(tile, tile);
    let window: Vec<f32> = (0..tile)
        .map(|i| {
            (std::f32::consts::PI * (i as f32 + 0.5) / tile as f32)
                .sin()
                .powi(2)
        })
        .collect();
    let origins: Vec<(usize, usize)> = (0..=(height - tile))
        .step_by(tile)
        .flat_map(|y| (0..=(width - tile)).step_by(tile).map(move |x| (x, y)))
        .collect();
    let spectra: Vec<Vec<f64>> = origins
        .par_iter()
        .filter_map(|&(ox, oy)| {
            let mut data = Vec::with_capacity(tile * tile);
            for y in 0..tile {
                for x in 0..tile {
                    let value = residual[(oy + y) * width + ox + x];
                    if !value.is_finite() {
                        return None;
                    }
                    data.push(Complex32::new(value * window[x] * window[y], 0.0));
                }
            }
            let mut scratch = vec![Complex32::default(); fft.len()];
            fft.forward(&mut data, &mut scratch);
            let (mut power, mut count) = (vec![0.0; SPECTRUM_BANDS], vec![0usize; SPECTRUM_BANDS]);
            for (index, value) in data.iter().enumerate() {
                let (fx, fy) = (index % tile, index / tile);
                let fx = fx.min(tile - fx) as f64 / tile as f64;
                let fy = fy.min(tile - fy) as f64 / tile as f64;
                let radius = fx.hypot(fy);
                if radius == 0.0 || radius > 0.5 {
                    continue;
                }
                let band =
                    ((radius / 0.5 * SPECTRUM_BANDS as f64) as usize).min(SPECTRUM_BANDS - 1);
                power[band] += f64::from(value.norm_sqr());
                count[band] += 1;
            }
            Some(
                power
                    .iter()
                    .zip(&count)
                    .map(|(p, &c)| if c > 0 { p / c as f64 } else { 0.0 })
                    .collect(),
            )
        })
        .collect();
    if spectra.is_empty() {
        return Vec::new();
    }
    let mut bands = [0.0; SPECTRUM_BANDS];
    for spectrum in &spectra {
        for (band, value) in bands.iter_mut().zip(spectrum) {
            *band += value;
        }
    }
    let mean = bands.iter().sum::<f64>() / SPECTRUM_BANDS as f64;
    if mean <= 0.0 {
        return Vec::new();
    }
    bands.iter().map(|band| band / mean).collect()
}

#[cfg(test)]
mod tests;
