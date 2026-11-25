//! Resample an image from one plate solution onto another: putting finished
//! masters of different scales, rotations and distortion onto one grid.

use crate::registration::{
    Interpolation, ReferenceRegion, Sampling, lanczos_clamp_spans, resample_region_on_cpu,
};
use crate::{Error, LinearImage, Result};
use rayon::prelude::*;
use seiza::Wcs;

/// How [`reproject`] resamples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReprojectOptions {
    /// The kernel each sample is read with, as registration uses it.
    pub interpolation: Interpolation,
    /// Average several samples per output pixel when the output's pixels
    /// are coarser than the source's, so a downsampled image is not
    /// aliased: `ceil(output scale / source scale)` samples a side, up to
    /// [`MAX_SUPERSAMPLING`]. Without it each output pixel reads one sample
    /// at its centre.
    pub antialias: bool,
}

impl Default for ReprojectOptions {
    fn default() -> Self {
        Self {
            interpolation: Interpolation::Bilinear,
            antialias: true,
        }
    }
}

/// The most samples a side [`ReprojectOptions::antialias`] averages per
/// output pixel.
pub const MAX_SUPERSAMPLING: usize = 16;

/// An image resampled onto another plate solution by [`reproject`].
#[derive(Clone, Debug, PartialEq)]
pub struct Reprojection {
    /// The image on the output grid, NaN where nothing of the source lands.
    pub image: LinearImage,
    /// Per output pixel, row by row, the share of its samples that landed
    /// on finite source samples: 1 inside the source, 0 outside it, and
    /// between at its edges.
    pub coverage: Vec<f32>,
    /// Samples averaged a side per output pixel: 1 without antialiasing.
    pub supersampling: usize,
}

/// Resample `source`, solved by `from`, onto a `width` × `height` grid
/// solved by `to`, such as one from [`Wcs::from_center_scale_rotation`]
/// with rotation 0 for a north-up grid. Each output pixel's sky position
/// comes from `to`, and its source pixel from `from`, SIP distortion
/// included on both sides, so the two may differ in scale, rotation,
/// parity and distortion.
///
/// Pixels count from 0 with centres on whole numbers, in both images'
/// stored row order. Every channel is resampled alike. A non-finite source
/// sample counts as missing, as registration treats it.
///
/// Refuses an empty output, or a `from` or `to` whose scale is not finite
/// and positive.
pub fn reproject(
    source: &LinearImage,
    from: &Wcs,
    to: &Wcs,
    width: usize,
    height: usize,
    options: &ReprojectOptions,
) -> Result<Reprojection> {
    if width == 0 || height == 0 {
        return Err(Error::InvalidImage(
            "a reprojection needs an output of at least one pixel".into(),
        ));
    }
    let (from_scale, to_scale) = (from.scale_arcsec_per_px(), to.scale_arcsec_per_px());
    if !(from_scale.is_finite() && from_scale > 0.0 && to_scale.is_finite() && to_scale > 0.0) {
        return Err(Error::InvalidImage(
            "a reprojection needs plate solutions with a finite, positive scale".into(),
        ));
    }
    let supersampling = if options.antialias {
        ((to_scale / from_scale).ceil() as usize).clamp(1, MAX_SUPERSAMPLING)
    } else {
        1
    };

    // The source pixel under each output pixel's centre, exactly. Samples
    // between centres interpolate it: the mapping bends far too slowly
    // across one pixel for that to matter.
    let centres = SourcePositions::map(from, to, width, height);
    let sampling = match options.interpolation {
        Interpolation::Bilinear => Sampling::Bilinear,
        Interpolation::Lanczos3 => Sampling::Lanczos3,
    };
    let clamp_spans = match options.interpolation {
        Interpolation::Lanczos3 => lanczos_clamp_spans(source),
        Interpolation::Bilinear => vec![0.0; source.channels],
    };
    let region = ReferenceRegion {
        x: 0,
        y: 0,
        width,
        height,
    };
    let channels = source.channels;
    let mut sums = vec![0.0_f64; width * height * channels];
    let mut counts = vec![0_u32; width * height * channels];
    let mut landed = vec![0_u32; width * height];
    for row in 0..supersampling {
        for column in 0..supersampling {
            let offset = |index: usize| (index as f64 + 0.5) / supersampling as f64 - 0.5;
            let (dx, dy) = (offset(column), offset(row));
            let pass = resample_region_on_cpu(
                source,
                region,
                &|x, y| centres.at(x + dx, y + dy),
                sampling,
                &clamp_spans,
            )?;
            sums.par_chunks_mut(channels)
                .zip(counts.par_chunks_mut(channels))
                .zip(landed.par_iter_mut())
                .zip(pass.data.par_chunks(channels))
                .for_each(|(((sum, count), landed), samples)| {
                    let mut any = false;
                    for ((sum, count), &sample) in sum.iter_mut().zip(count).zip(samples) {
                        if sample.is_finite() {
                            *sum += f64::from(sample);
                            *count += 1;
                            any = true;
                        }
                    }
                    *landed += u32::from(any);
                });
        }
    }
    let data = sums
        .par_iter()
        .zip(&counts)
        .map(|(&sum, &count)| {
            if count == 0 {
                f32::NAN
            } else {
                (sum / f64::from(count)) as f32
            }
        })
        .collect();
    let per_pixel = (supersampling * supersampling) as f32;
    let coverage = landed
        .par_iter()
        .map(|&landed| landed as f32 / per_pixel)
        .collect();
    Ok(Reprojection {
        image: LinearImage::new(width, height, channels, data)?,
        coverage,
        supersampling,
    })
}

/// A source position far off any image, for an output pixel whose sky has
/// no place in the source, so resampling leaves it empty.
const NOWHERE: (f64, f64) = (-1.0e9, -1.0e9);

/// The source pixel under the centre of every output pixel.
struct SourcePositions {
    width: usize,
    height: usize,
    /// Row by row; NaN where the sky is behind the source's horizon.
    positions: Vec<(f64, f64)>,
}

impl SourcePositions {
    fn map(from: &Wcs, to: &Wcs, width: usize, height: usize) -> Self {
        let positions = (0..width * height)
            .into_par_iter()
            .map(|index| {
                let (x, y) = ((index % width) as f64, (index / width) as f64);
                let (ra, dec) = to.pixel_to_world(x, y);
                from.world_to_pixel(ra, dec).unwrap_or((f64::NAN, f64::NAN))
            })
            .collect();
        Self {
            width,
            height,
            positions,
        }
    }

    /// The source position under output position `(x, y)`, bilinearly
    /// between the four nearest centres, or linearly past the outermost
    /// ones.
    fn at(&self, x: f64, y: f64) -> (f64, f64) {
        let axis = |value: f64, length: usize| {
            if length == 1 {
                return (0, 0, 0.0);
            }
            let low = (value.floor().max(0.0) as usize).min(length - 2);
            (low, low + 1, value - low as f64)
        };
        let (x0, x1, tx) = axis(x, self.width);
        let (y0, y1, ty) = axis(y, self.height);
        let at = |x: usize, y: usize| self.positions[y * self.width + x];
        let corners = [at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1)];
        if corners
            .iter()
            .any(|(x, y)| !x.is_finite() || !y.is_finite())
        {
            return NOWHERE;
        }
        let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
        let blend = |pick: fn(&(f64, f64)) -> f64| {
            lerp(
                lerp(pick(&corners[0]), pick(&corners[1]), tx),
                lerp(pick(&corners[2]), pick(&corners[3]), tx),
                ty,
            )
        };
        (blend(|p| p.0), blend(|p| p.1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smooth sky: brightness as a function of position, so any grid can
    /// be checked against it.
    fn sky(ra: f64, dec: f64) -> f32 {
        (1000.0 + 300.0 * ((ra - 150.0) * 20.0).sin() * ((dec - 30.0) * 15.0).cos()) as f32
    }

    fn render(wcs: &Wcs, width: usize, height: usize, channels: usize) -> LinearImage {
        let data = (0..width * height)
            .flat_map(|index| {
                let (ra, dec) = wcs.pixel_to_world((index % width) as f64, (index / width) as f64);
                let value = sky(ra, dec);
                (0..channels).map(move |channel| value + channel as f32 * 10.0)
            })
            .collect();
        LinearImage::new(width, height, channels, data).unwrap()
    }

    fn distorted(mut wcs: Wcs) -> Wcs {
        let terms = seiza::Sip::forward_terms(3);
        let mut a = vec![0.0; terms.len()];
        let mut b = vec![0.0; terms.len()];
        for (index, term) in terms.iter().enumerate() {
            match term {
                (3, 0) | (1, 2) => a[index] = 2e-7,
                (0, 3) | (2, 1) => b[index] = 2e-7,
                _ => {}
            }
        }
        // An inverse good to a small fraction of a pixel over this frame.
        let inverse_terms = seiza::Sip::inverse_terms(3);
        let mut ap = vec![0.0; inverse_terms.len()];
        let mut bp = vec![0.0; inverse_terms.len()];
        for (index, term) in inverse_terms.iter().enumerate() {
            match term {
                (3, 0) | (1, 2) => ap[index] = -2e-7,
                (0, 3) | (2, 1) => bp[index] = -2e-7,
                _ => {}
            }
        }
        wcs.sip = Some(seiza::Sip {
            order: 3,
            a,
            b,
            ap,
            bp,
        });
        wcs
    }

    #[test]
    fn a_rotated_distorted_image_lands_on_a_north_up_grid() {
        let source_wcs = distorted(Wcs::from_center_scale_rotation(
            (150.0, 30.0),
            (199.5, 149.5),
            2.0,
            37.0,
            true,
        ));
        let source = render(&source_wcs, 400, 300, 3);
        let grid = Wcs::from_center_scale_rotation((150.0, 30.0), (99.5, 99.5), 3.0, 0.0, false);
        for interpolation in [Interpolation::Bilinear, Interpolation::Lanczos3] {
            let options = ReprojectOptions {
                interpolation,
                antialias: false,
            };
            let result = reproject(&source, &source_wcs, &grid, 200, 200, &options).unwrap();
            assert_eq!(result.supersampling, 1);
            let mut checked = 0;
            for y in 0..200 {
                for x in 0..200 {
                    let index = y * 200 + x;
                    if result.coverage[index] < 1.0 {
                        assert!(
                            result.image.data[index * 3].is_nan() || result.coverage[index] > 0.0
                        );
                        continue;
                    }
                    let (ra, dec) = grid.pixel_to_world(x as f64, y as f64);
                    let expected = sky(ra, dec);
                    for channel in 0..3 {
                        let value = result.image.data[index * 3 + channel];
                        let truth = expected + channel as f32 * 10.0;
                        assert!(
                            (value - truth).abs() < 2.0,
                            "{interpolation} ({x}, {y}) channel {channel}: {value} vs {truth}"
                        );
                    }
                    checked += 1;
                }
            }
            // The rotated source covers most of the smaller grid.
            assert!(checked > 25_000, "{checked}");
            // Corners of the grid lie off the rotated source.
            assert_eq!(result.coverage[0], 0.0);
            assert!(result.image.data[0].is_nan());
        }
    }

    #[test]
    fn a_coarser_grid_averages_rather_than_aliases() {
        let source_wcs =
            Wcs::from_center_scale_rotation((150.0, 30.0), (255.5, 255.5), 1.0, 0.0, false);
        // A checkerboard at the source's own pixel scale averages to its
        // mean on any coarser grid, but aliases when read at points.
        let data = (0..512 * 512)
            .map(|index| {
                if (index % 512 + index / 512) % 2 == 0 {
                    0.0
                } else {
                    2.0
                }
            })
            .collect();
        let source = LinearImage::new(512, 512, 1, data).unwrap();
        let grid = Wcs::from_center_scale_rotation((150.0, 30.0), (31.5, 31.5), 7.3, 0.0, false);
        let smooth = reproject(
            &source,
            &source_wcs,
            &grid,
            64,
            64,
            &ReprojectOptions::default(),
        )
        .unwrap();
        assert_eq!(smooth.supersampling, 8);
        let spread = |image: &LinearImage| {
            let inner: Vec<f32> = (8..56)
                .flat_map(|y| (8..56).map(move |x| (x, y)))
                .map(|(x, y)| image.data[y * 64 + x])
                .collect();
            let max = inner.iter().copied().fold(f32::MIN, f32::max);
            let min = inner.iter().copied().fold(f32::MAX, f32::min);
            (min, max)
        };
        let (min, max) = spread(&smooth.image);
        assert!(min > 0.8 && max < 1.2, "{min}..{max}");
        let point = reproject(
            &source,
            &source_wcs,
            &grid,
            64,
            64,
            &ReprojectOptions {
                antialias: false,
                ..ReprojectOptions::default()
            },
        )
        .unwrap();
        let (min, max) = spread(&point.image);
        assert!(max - min > 1.0, "point sampling aliases: {min}..{max}");
    }

    #[test]
    fn a_grid_off_the_source_is_empty_and_bad_requests_are_refused() {
        let source_wcs =
            Wcs::from_center_scale_rotation((150.0, 30.0), (49.5, 49.5), 2.0, 0.0, false);
        let source = render(&source_wcs, 100, 100, 1);
        // The far side of the sky: behind the source's tangent plane.
        let away = Wcs::from_center_scale_rotation((330.0, -30.0), (9.5, 9.5), 2.0, 0.0, false);
        let result = reproject(
            &source,
            &source_wcs,
            &away,
            20,
            20,
            &ReprojectOptions::default(),
        )
        .unwrap();
        assert!(result.image.data.iter().all(|value| value.is_nan()));
        assert!(result.coverage.iter().all(|&share| share == 0.0));

        assert!(
            reproject(
                &source,
                &source_wcs,
                &away,
                0,
                20,
                &ReprojectOptions::default()
            )
            .is_err()
        );
        let mut flat = source_wcs.clone();
        flat.cd = [[0.0, 0.0], [0.0, 0.0]];
        assert!(reproject(&source, &flat, &away, 20, 20, &ReprojectOptions::default()).is_err());
    }

    #[test]
    fn the_same_grid_gives_the_source_back() {
        let wcs = Wcs::from_center_scale_rotation((150.0, 30.0), (31.5, 23.5), 2.0, 12.0, false);
        let source = render(&wcs, 64, 48, 1);
        let result = reproject(&source, &wcs, &wcs, 64, 48, &ReprojectOptions::default()).unwrap();
        assert_eq!(result.supersampling, 1);
        for (value, truth) in result.image.data.iter().zip(&source.data) {
            assert!((value - truth).abs() < 1e-3, "{value} vs {truth}");
        }
        assert!(result.coverage.iter().all(|&share| share == 1.0));
    }
}
