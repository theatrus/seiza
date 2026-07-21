use crate::{Error, LinearImage, Result};
use rayon::prelude::*;
use seiza_stretch::{ResolvedCurve, StretchConfig, StretchParams};
use std::str::FromStr;

/// How mono input channels are mapped into a common working range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColorNormalization {
    /// Preserve finite input samples exactly; a non-finite required input masks
    /// the complete output pixel. Dynamic Foraxx palettes require finite
    /// samples to already lie in `[0, 1]` when this mode is selected.
    None,
    /// Affinely map robust black and white percentiles to `[0, 1]`.
    Percentile {
        /// Fraction of samples mapped to the black point.
        black_percentile: f32,
        /// Fraction of samples mapped to the white point.
        white_percentile: f32,
        /// Cap on samples drawn when estimating the percentiles.
        max_samples: usize,
    },
}

impl Default for ColorNormalization {
    fn default() -> Self {
        Self::Percentile {
            black_percentile: 0.001,
            white_percentile: 0.995,
            max_samples: 1_000_000,
        }
    }
}

/// Shared preparation options for color composition from stacked mono frames.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ColorOptions {
    /// How input channels are mapped into the working range.
    pub normalization: ColorNormalization,
    /// Transfer already applied to the input channels.
    ///
    /// Display-referred inputs must use [`ColorNormalization::None`]. This is
    /// intended for callers that independently stretch each registered mono
    /// channel before composition.
    pub input_transfer: ColorTransfer,
}

/// Display preparation applied before a dynamic Foraxx composition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ForaxxOptions {
    /// Display level assigned to the channel median before Foraxx composition.
    pub target_median: f32,
    /// Normal-equivalent MADs below the median used as the Foraxx black point.
    pub shadows_clip: f32,
}

impl Default for ForaxxOptions {
    fn default() -> Self {
        Self {
            target_median: 0.2,
            shadows_clip: -2.8,
        }
    }
}

/// Whether output samples remain linear-light or have a display transfer applied.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ColorTransfer {
    /// Samples are linear-light.
    #[default]
    LinearLight,
    /// Samples carry a display transfer.
    DisplayReferred,
}

impl ColorTransfer {
    /// Short label written to the `SEIZATRF` FITS card.
    pub fn fits_name(self) -> &'static str {
        match self {
            Self::LinearLight => "LINEAR",
            Self::DisplayReferred => "DISPLAY",
        }
    }
}

/// Result of a color composition, including the transfer semantics of its samples.
#[derive(Clone, Debug, PartialEq)]
pub struct ColorComposition {
    /// The composed RGB image.
    pub image: LinearImage,
    /// Transfer semantics of the composed samples.
    pub transfer: ColorTransfer,
}

/// Coefficients for one output channel, ordered by physical narrowband filter.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NarrowbandMix {
    /// Weight applied to the SII channel.
    pub sii: f32,
    /// Weight applied to the H-alpha channel.
    pub ha: f32,
    /// Weight applied to the OIII channel.
    pub oiii: f32,
}

/// A static linear mapping from SII, H-alpha, and OIII to RGB.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NarrowbandMatrix {
    /// Filter weights for the red output channel.
    pub red: NarrowbandMix,
    /// Filter weights for the green output channel.
    pub green: NarrowbandMix,
    /// Filter weights for the blue output channel.
    pub blue: NarrowbandMix,
}

/// Common false-color assignments. A three-letter name lists the source used
/// for red, green, and blue respectively.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NarrowbandPalette {
    /// SII, H-alpha, OIII to red, green, blue (the classic Hubble palette).
    Sho,
    /// SII, OIII, H-alpha to red, green, blue.
    Soh,
    /// H-alpha, SII, OIII to red, green, blue.
    Hso,
    /// H-alpha, OIII, SII to red, green, blue.
    Hos,
    /// OIII, SII, H-alpha to red, green, blue.
    Osh,
    /// OIII, H-alpha, SII to red, green, blue.
    Ohs,
    /// H-alpha to red, OIII to both green and blue.
    Hoo,
    /// Dynamic Foraxx SHO blend, display-referred.
    ForaxxSho,
    /// Dynamic Foraxx HOO blend, display-referred.
    ForaxxHoo,
}

impl NarrowbandPalette {
    /// Uppercase name, for example `SHO` or `FORAXX-HOO`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Sho => "SHO",
            Self::Soh => "SOH",
            Self::Hso => "HSO",
            Self::Hos => "HOS",
            Self::Osh => "OSH",
            Self::Ohs => "OHS",
            Self::Hoo => "HOO",
            Self::ForaxxSho => "FORAXX-SHO",
            Self::ForaxxHoo => "FORAXX-HOO",
        }
    }

    /// Whether the palette needs an SII channel supplied.
    pub fn requires_sii(self) -> bool {
        !matches!(self, Self::Hoo | Self::ForaxxHoo)
    }

    /// Transfer semantics of the palette's output.
    pub fn transfer(self) -> ColorTransfer {
        if matches!(self, Self::ForaxxSho | Self::ForaxxHoo) {
            ColorTransfer::DisplayReferred
        } else {
            ColorTransfer::LinearLight
        }
    }

    fn matrix(self) -> Option<NarrowbandMatrix> {
        let s = NarrowbandMix {
            sii: 1.0,
            ..NarrowbandMix::default()
        };
        let h = NarrowbandMix {
            ha: 1.0,
            ..NarrowbandMix::default()
        };
        let o = NarrowbandMix {
            oiii: 1.0,
            ..NarrowbandMix::default()
        };
        let channels = match self {
            Self::Sho => [s, h, o],
            Self::Soh => [s, o, h],
            Self::Hso => [h, s, o],
            Self::Hos => [h, o, s],
            Self::Osh => [o, s, h],
            Self::Ohs => [o, h, s],
            Self::Hoo => [h, o, o],
            Self::ForaxxSho | Self::ForaxxHoo => return None,
        };
        Some(NarrowbandMatrix {
            red: channels[0],
            green: channels[1],
            blue: channels[2],
        })
    }
}

impl FromStr for NarrowbandPalette {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().replace('_', "-").as_str() {
            "sho" => Ok(Self::Sho),
            "soh" => Ok(Self::Soh),
            "hso" => Ok(Self::Hso),
            "hos" => Ok(Self::Hos),
            "osh" => Ok(Self::Osh),
            "ohs" => Ok(Self::Ohs),
            "hoo" => Ok(Self::Hoo),
            "foraxx-sho" | "foraxxsho" => Ok(Self::ForaxxSho),
            "foraxx-hoo" | "foraxxhoo" => Ok(Self::ForaxxHoo),
            _ => Err(Error::Color(format!("unknown narrowband palette {value}"))),
        }
    }
}

/// Combine three aligned mono stacks into a linear-light RGB image.
pub fn combine_rgb(
    red: &LinearImage,
    green: &LinearImage,
    blue: &LinearImage,
    options: &ColorOptions,
) -> Result<ColorComposition> {
    validate_options(options)?;
    validate_mono_set(&[("red", red), ("green", green), ("blue", blue)])?;
    let channels = [
        prepare_channel(red, options.normalization)?,
        prepare_channel(green, options.normalization)?,
        prepare_channel(blue, options.normalization)?,
    ];
    let mut data = vec![0.0; red.pixel_count() * 3];
    data.par_chunks_mut(3)
        .enumerate()
        .for_each(|(index, output)| {
            if channels.iter().any(|channel| !channel.valid(index)) {
                output.fill(f32::NAN);
                return;
            }
            output.copy_from_slice(&[
                channels[0].sample(index),
                channels[1].sample(index),
                channels[2].sample(index),
            ]);
        });
    Ok(ColorComposition {
        image: LinearImage::new(red.width, red.height, 3, data)?,
        transfer: options.input_transfer,
    })
}

/// Combine aligned L, R, G, and B stacks in linear-light RGB.
///
/// The source luminance is percentile-matched in the same way as the RGB
/// channels. `luminance_weight=1` fully replaces RGB luminance, while zero is
/// an RGB no-op. Chromaticity is retained by scaling the linear RGB triplet.
pub fn combine_lrgb(
    luminance: &LinearImage,
    red: &LinearImage,
    green: &LinearImage,
    blue: &LinearImage,
    luminance_weight: f32,
    options: &ColorOptions,
) -> Result<ColorComposition> {
    combine_lrgb_with_target(
        luminance,
        red,
        green,
        blue,
        LrgbTarget::Replace { luminance_weight },
        options,
    )
}

/// Combine aligned L, R, G, and B stacks using additive super-luminance.
///
/// After applying the selected channel normalization, the target luminance is
/// `L + R + G + B`. The RGB triplet is scaled to that luminance so its linear
/// chromaticity is retained. The linear `f32` result may exceed one.
pub fn combine_super_lrgb(
    luminance: &LinearImage,
    red: &LinearImage,
    green: &LinearImage,
    blue: &LinearImage,
    options: &ColorOptions,
) -> Result<ColorComposition> {
    combine_lrgb_with_target(luminance, red, green, blue, LrgbTarget::Super, options)
}

#[derive(Clone, Copy)]
enum LrgbTarget {
    Replace { luminance_weight: f32 },
    Super,
}

fn combine_lrgb_with_target(
    luminance: &LinearImage,
    red: &LinearImage,
    green: &LinearImage,
    blue: &LinearImage,
    target: LrgbTarget,
    options: &ColorOptions,
) -> Result<ColorComposition> {
    if let LrgbTarget::Replace { luminance_weight } = target
        && (!luminance_weight.is_finite() || !(0.0..=1.0).contains(&luminance_weight))
    {
        return Err(Error::Color(
            "luminance weight must be finite and between 0 and 1".into(),
        ));
    }
    validate_options(options)?;
    validate_mono_set(&[
        ("luminance", luminance),
        ("red", red),
        ("green", green),
        ("blue", blue),
    ])?;
    if matches!(
        target,
        LrgbTarget::Replace {
            luminance_weight: 0.0
        }
    ) {
        return combine_rgb(red, green, blue, options);
    }
    let l = prepare_channel(luminance, options.normalization)?;
    let rgb = [
        prepare_channel(red, options.normalization)?,
        prepare_channel(green, options.normalization)?,
        prepare_channel(blue, options.normalization)?,
    ];
    let mut data = vec![0.0; luminance.pixel_count() * 3];
    data.par_chunks_mut(3)
        .enumerate()
        .for_each(|(index, output)| {
            if !l.valid(index) || rgb.iter().any(|channel| !channel.valid(index)) {
                output.fill(f32::NAN);
                return;
            }
            let r = rgb[0].sample(index);
            let g = rgb[1].sample(index);
            let b = rgb[2].sample(index);
            let rgb_luminance = crate::image::rec709_luma(r, g, b);
            let target_luminance = match target {
                LrgbTarget::Replace { luminance_weight } => (1.0 - luminance_weight)
                    .mul_add(rgb_luminance, luminance_weight * l.sample(index)),
                LrgbTarget::Super => l.sample(index) + r + g + b,
            };
            if rgb_luminance > 1.0e-8 && target_luminance.is_finite() {
                let scale = target_luminance / rgb_luminance;
                output.copy_from_slice(&[r * scale, g * scale, b * scale]);
            } else {
                output.copy_from_slice(&[target_luminance; 3]);
            }
        });
    Ok(ColorComposition {
        image: LinearImage::new(luminance.width, luminance.height, 3, data)?,
        transfer: options.input_transfer,
    })
}

/// Compose a common narrowband palette from aligned mono stacks.
pub fn combine_narrowband(
    ha: &LinearImage,
    oiii: &LinearImage,
    sii: Option<&LinearImage>,
    palette: NarrowbandPalette,
    options: &ColorOptions,
    foraxx: &ForaxxOptions,
) -> Result<ColorComposition> {
    validate_options(options)?;
    let sii = if palette.requires_sii() {
        Some(
            sii.ok_or_else(|| Error::Color(format!("{} requires an SII channel", palette.name())))?,
        )
    } else {
        None
    };
    let mut inputs = vec![("H-alpha", ha), ("OIII", oiii)];
    if let Some(sii) = sii {
        inputs.push(("SII", sii));
    }
    validate_mono_set(&inputs)?;
    let h = prepare_channel(ha, options.normalization)?;
    let o = prepare_channel(oiii, options.normalization)?;
    let s = sii
        .map(|image| prepare_channel(image, options.normalization))
        .transpose()?;

    if let Some(matrix) = palette.matrix() {
        return combine_prepared_matrix(
            ha.width,
            ha.height,
            h,
            o,
            s,
            matrix,
            options.input_transfer,
        );
    }

    let (h_display, o_display, s_display) = match options.input_transfer {
        ColorTransfer::LinearLight => {
            validate_foraxx_options(foraxx)?;
            let maximum_samples = normalization_sample_limit(options.normalization);
            (
                display_transform(h, foraxx, maximum_samples)?,
                display_transform(o, foraxx, maximum_samples)?,
                s.map(|channel| display_transform(channel, foraxx, maximum_samples))
                    .transpose()?,
            )
        }
        ColorTransfer::DisplayReferred => (
            ResolvedCurve::Identity,
            ResolvedCurve::Identity,
            s.map(|_| ResolvedCurve::Identity),
        ),
    };
    let mut data = vec![0.0; ha.pixel_count() * 3];
    data.par_chunks_mut(3)
        .enumerate()
        .for_each(|(index, output)| {
            if !h.valid(index) || !o.valid(index) || s.is_some_and(|channel| !channel.valid(index))
            {
                output.fill(f32::NAN);
                return;
            }
            let h = h_display.map(h.sample(index));
            let o = o_display.map(o.sample(index));
            let product = (o * h).clamp(0.0, 1.0);
            let green_factor = product.powf(1.0 - product);
            let green = green_factor.mul_add(h, (1.0 - green_factor) * o);
            let red = match palette {
                NarrowbandPalette::ForaxxHoo => h,
                NarrowbandPalette::ForaxxSho => {
                    let o_factor = o.powf(1.0 - o);
                    o_factor.mul_add(
                        s_display
                            .expect("SII was validated")
                            .map(s.expect("SII was validated").sample(index)),
                        (1.0 - o_factor) * h,
                    )
                }
                _ => unreachable!("static palettes returned above"),
            };
            output.copy_from_slice(&[red, green, o]);
        });
    Ok(ColorComposition {
        image: LinearImage::new(ha.width, ha.height, 3, data)?,
        transfer: palette.transfer(),
    })
}

/// Apply a custom linear narrowband mixing matrix.
pub fn combine_narrowband_matrix(
    ha: &LinearImage,
    oiii: &LinearImage,
    sii: Option<&LinearImage>,
    matrix: NarrowbandMatrix,
    options: &ColorOptions,
) -> Result<ColorComposition> {
    validate_options(options)?;
    validate_matrix(matrix)?;
    let uses_sii = matrix_uses_sii(matrix);
    let sii = if uses_sii {
        Some(sii.ok_or_else(|| {
            Error::Color("the custom matrix references SII but no SII channel was supplied".into())
        })?)
    } else {
        None
    };
    let mut inputs = vec![("H-alpha", ha), ("OIII", oiii)];
    if let Some(sii) = sii {
        inputs.push(("SII", sii));
    }
    validate_mono_set(&inputs)?;
    let h = prepare_channel(ha, options.normalization)?;
    let o = prepare_channel(oiii, options.normalization)?;
    let s = sii
        .map(|image| prepare_channel(image, options.normalization))
        .transpose()?;
    combine_prepared_matrix(ha.width, ha.height, h, o, s, matrix, options.input_transfer)
}

fn combine_prepared_matrix(
    width: usize,
    height: usize,
    ha: PreparedChannel<'_>,
    oiii: PreparedChannel<'_>,
    sii: Option<PreparedChannel<'_>>,
    matrix: NarrowbandMatrix,
    transfer: ColorTransfer,
) -> Result<ColorComposition> {
    let mut data = vec![0.0; width * height * 3];
    data.par_chunks_mut(3)
        .enumerate()
        .for_each(|(index, output)| {
            if !ha.valid(index)
                || !oiii.valid(index)
                || sii.is_some_and(|channel| !channel.valid(index))
            {
                output.fill(f32::NAN);
                return;
            }
            let h = ha.sample(index);
            let o = oiii.sample(index);
            let s = sii.map_or(0.0, |channel| channel.sample(index));
            for (target, mix) in output
                .iter_mut()
                .zip([matrix.red, matrix.green, matrix.blue])
            {
                *target = mix.sii.mul_add(s, mix.ha.mul_add(h, mix.oiii * o));
            }
        });
    Ok(ColorComposition {
        image: LinearImage::new(width, height, 3, data)?,
        transfer,
    })
}

#[derive(Clone, Copy)]
struct PreparedChannel<'a> {
    values: &'a [f32],
    transform: ChannelTransform,
}

impl PreparedChannel<'_> {
    fn valid(self, index: usize) -> bool {
        self.values[index].is_finite()
    }

    fn sample(self, index: usize) -> f32 {
        let value = self.values[index];
        if !value.is_finite() {
            return 0.0;
        }
        match self.transform {
            ChannelTransform::Identity => value,
            ChannelTransform::Percentile {
                black,
                reciprocal_range,
            } => ((value - black) * reciprocal_range).clamp(0.0, 1.0),
        }
    }
}

#[derive(Clone, Copy)]
enum ChannelTransform {
    Identity,
    Percentile { black: f32, reciprocal_range: f32 },
}

fn prepare_channel(
    image: &LinearImage,
    normalization: ColorNormalization,
) -> Result<PreparedChannel<'_>> {
    let transform = match normalization {
        ColorNormalization::None => ChannelTransform::Identity,
        ColorNormalization::Percentile {
            black_percentile,
            white_percentile,
            max_samples,
        } => {
            let (black, white) =
                percentile_levels(&image.data, black_percentile, white_percentile, max_samples)?;
            ChannelTransform::Percentile {
                black,
                reciprocal_range: 1.0 / (white - black),
            }
        }
    };
    Ok(PreparedChannel {
        values: &image.data,
        transform,
    })
}

fn percentile_levels(
    values: &[f32],
    black_percentile: f32,
    white_percentile: f32,
    max_samples: usize,
) -> Result<(f32, f32)> {
    let stride = values.len().div_ceil(max_samples.max(1)).max(1);
    let mut sample = values
        .iter()
        .step_by(stride)
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if sample.is_empty() {
        return Err(Error::Color("channel contains no finite samples".into()));
    }
    sample.sort_unstable_by(f32::total_cmp);
    let index = |percentile: f32| ((sample.len() - 1) as f32 * percentile).round() as usize;
    let black = sample[index(black_percentile)];
    let white = sample[index(white_percentile)];
    if white <= black {
        return Err(Error::Color(
            "channel has no usable range between normalization percentiles".into(),
        ));
    }
    Ok((black, white))
}

fn display_transform(
    channel: PreparedChannel<'_>,
    options: &ForaxxOptions,
    max_samples: usize,
) -> Result<ResolvedCurve> {
    if matches!(channel.transform, ChannelTransform::Identity)
        && channel
            .values
            .par_iter()
            .copied()
            .filter(|value| value.is_finite())
            .any(|value| !(0.0..=1.0).contains(&value))
    {
        return Err(Error::Color(
            "Foraxx with normalization disabled requires finite samples in [0, 1]; use percentile normalization for sensor-unit inputs"
                .into(),
        ));
    }
    let stride = channel.values.len().div_ceil(max_samples.max(1)).max(1);
    let sample = (0..channel.values.len())
        .step_by(stride)
        .filter(|index| channel.values[*index].is_finite())
        .map(|index| channel.sample(index))
        .collect::<Vec<_>>();
    if sample.is_empty() {
        return Err(Error::Color(
            "channel contains no samples to stretch".into(),
        ));
    }
    let plan = StretchConfig::auto_mtf(
        StretchParams {
            target_median: f64::from(options.target_median),
            shadows_clip: f64::from(options.shadows_clip),
        },
        sample.len(),
    )
    .resolve_for(&sample, 1)
    .map_err(|error| Error::Color(error.to_string()))?;
    Ok(plan.curves()[0])
}

fn normalization_sample_limit(normalization: ColorNormalization) -> usize {
    match normalization {
        ColorNormalization::None => 1_000_000,
        ColorNormalization::Percentile { max_samples, .. } => max_samples,
    }
}

fn validate_foraxx_options(options: &ForaxxOptions) -> Result<()> {
    if !options.target_median.is_finite() || !(0.0..1.0).contains(&options.target_median) {
        return Err(Error::Color(
            "Foraxx target median must be finite and between zero and one".into(),
        ));
    }
    if !options.shadows_clip.is_finite() || options.shadows_clip > 0.0 {
        return Err(Error::Color(
            "Foraxx shadows clip must be finite and zero or negative".into(),
        ));
    }
    Ok(())
}

fn validate_normalization(normalization: ColorNormalization) -> Result<()> {
    if let ColorNormalization::Percentile {
        black_percentile,
        white_percentile,
        max_samples,
    } = normalization
        && (!black_percentile.is_finite()
            || !white_percentile.is_finite()
            || black_percentile < 0.0
            || white_percentile > 1.0
            || black_percentile >= white_percentile
            || max_samples == 0)
    {
        return Err(Error::Color(
            "normalization percentiles must satisfy 0 <= black < white <= 1 and max_samples > 0"
                .into(),
        ));
    }
    Ok(())
}

fn validate_options(options: &ColorOptions) -> Result<()> {
    validate_normalization(options.normalization)?;
    if options.input_transfer == ColorTransfer::DisplayReferred
        && options.normalization != ColorNormalization::None
    {
        return Err(Error::Color(
            "display-referred inputs must disable color normalization".into(),
        ));
    }
    Ok(())
}

fn validate_matrix(matrix: NarrowbandMatrix) -> Result<()> {
    if [matrix.red, matrix.green, matrix.blue]
        .into_iter()
        .flat_map(|mix| [mix.sii, mix.ha, mix.oiii])
        .any(|coefficient| !coefficient.is_finite())
    {
        return Err(Error::Color(
            "narrowband matrix coefficients must be finite".into(),
        ));
    }
    Ok(())
}

fn matrix_uses_sii(matrix: NarrowbandMatrix) -> bool {
    [matrix.red, matrix.green, matrix.blue]
        .iter()
        .any(|mix| mix.sii != 0.0)
}

fn validate_mono_set(images: &[(&str, &LinearImage)]) -> Result<()> {
    let (_, reference) = images
        .first()
        .ok_or_else(|| Error::Color("at least one input channel is required".into()))?;
    for (name, image) in images {
        if image.channels != 1 {
            return Err(Error::Color(format!("{name} input must be one-channel")));
        }
        if image.width != reference.width || image.height != reference.height {
            return Err(Error::Color(format!(
                "{name} dimensions {}x{} do not match {}x{}",
                image.width, image.height, reference.width, reference.height
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mono(values: &[f32]) -> LinearImage {
        LinearImage::new(values.len(), 1, 1, values.to_vec()).unwrap()
    }

    fn raw_options() -> ColorOptions {
        ColorOptions {
            normalization: ColorNormalization::None,
            input_transfer: ColorTransfer::LinearLight,
        }
    }

    fn display_options() -> ColorOptions {
        ColorOptions {
            normalization: ColorNormalization::None,
            input_transfer: ColorTransfer::DisplayReferred,
        }
    }

    #[derive(Clone, Copy)]
    enum PreviousDisplayTransform {
        Identity,
        Mtf {
            shadows: f32,
            midtone: f64,
            highlights: f32,
        },
    }

    impl PreviousDisplayTransform {
        fn map(self, value: f32) -> f32 {
            match self {
                Self::Identity => value.clamp(0.0, 1.0),
                Self::Mtf {
                    shadows,
                    midtone,
                    highlights,
                } => {
                    let input = 1.0 - f64::from(highlights) + f64::from(value - shadows);
                    seiza_fits::midtones_transfer_function(midtone, input) as f32
                }
            }
        }
    }

    fn previous_display_transform(
        channel: PreparedChannel<'_>,
        options: &ForaxxOptions,
        max_samples: usize,
    ) -> PreviousDisplayTransform {
        let stride = channel.values.len().div_ceil(max_samples.max(1)).max(1);
        let mut sample = (0..channel.values.len())
            .step_by(stride)
            .filter(|index| channel.values[*index].is_finite())
            .map(|index| channel.sample(index))
            .collect::<Vec<_>>();
        sample.sort_unstable_by(f32::total_cmp);
        let median = sample[sample.len() / 2];
        for value in &mut sample {
            *value = (*value - median).abs();
        }
        sample.sort_unstable_by(f32::total_cmp);
        let normal_mad = sample[sample.len() / 2] * 1.4826;
        if normal_mad <= f32::EPSILON {
            return PreviousDisplayTransform::Identity;
        }
        let (shadows, midtone, highlights) = if median > 0.5 {
            let highlights = median - options.shadows_clip * normal_mad;
            let midtone = seiza_fits::midtones_transfer_function(
                f64::from(options.target_median),
                f64::from(1.0 - (highlights - median)),
            );
            (0.0, midtone, highlights)
        } else {
            let shadows = (median + options.shadows_clip * normal_mad).max(0.0);
            let midtone = seiza_fits::midtones_transfer_function(
                f64::from(options.target_median),
                f64::from(median - shadows),
            );
            (shadows, midtone, 1.0)
        };
        PreviousDisplayTransform::Mtf {
            shadows,
            midtone,
            highlights,
        }
    }

    #[test]
    fn rgb_interleaves_linear_channels() {
        let result = combine_rgb(
            &mono(&[0.1, 0.2]),
            &mono(&[0.3, 0.4]),
            &mono(&[0.5, 0.6]),
            &raw_options(),
        )
        .unwrap();
        assert_eq!(result.transfer, ColorTransfer::LinearLight);
        assert_eq!(result.image.data, [0.1, 0.3, 0.5, 0.2, 0.4, 0.6]);
    }

    #[test]
    fn rgb_preserves_display_prepared_channels_and_transfer() {
        let result = combine_rgb(
            &mono(&[0.1, 0.2]),
            &mono(&[0.3, 0.4]),
            &mono(&[0.5, 0.6]),
            &display_options(),
        )
        .unwrap();
        assert_eq!(result.transfer, ColorTransfer::DisplayReferred);
        assert_eq!(result.image.data, [0.1, 0.3, 0.5, 0.2, 0.4, 0.6]);
    }

    #[test]
    fn lrgb_replaces_luminance_without_changing_rgb_ratios() {
        let result = combine_lrgb(
            &mono(&[0.8]),
            &mono(&[0.2]),
            &mono(&[0.4]),
            &mono(&[0.1]),
            1.0,
            &raw_options(),
        )
        .unwrap();
        let pixel = &result.image.data;
        let y = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
        assert!((y - 0.8).abs() < 1.0e-6);
        assert!((pixel[0] / pixel[1] - 0.5).abs() < 1.0e-6);
        assert!((pixel[2] / pixel[1] - 0.25).abs() < 1.0e-6);
    }

    #[test]
    fn super_lrgb_adds_all_four_channels_and_preserves_chromaticity() {
        let l = 0.8;
        let r = 0.2;
        let g = 0.4;
        let b = 0.1;
        let result = combine_super_lrgb(
            &mono(&[l]),
            &mono(&[r]),
            &mono(&[g]),
            &mono(&[b]),
            &raw_options(),
        )
        .unwrap();
        let pixel = &result.image.data;
        let output_luminance =
            0.2126_f32.mul_add(pixel[0], 0.7152_f32.mul_add(pixel[1], 0.0722 * pixel[2]));
        assert!((output_luminance - (l + r + g + b)).abs() < 1.0e-6);
        assert!((pixel[0] / pixel[1] - r / g).abs() < 1.0e-6);
        assert!((pixel[2] / pixel[1] - b / g).abs() < 1.0e-6);
        assert_eq!(result.transfer, ColorTransfer::LinearLight);
    }

    #[test]
    fn direct_palettes_map_physical_filters() {
        let h = mono(&[0.2]);
        let o = mono(&[0.3]);
        let s = mono(&[0.4]);
        let sho = combine_narrowband(
            &h,
            &o,
            Some(&s),
            NarrowbandPalette::Sho,
            &raw_options(),
            &ForaxxOptions::default(),
        )
        .unwrap();
        assert_eq!(sho.image.data, [0.4, 0.2, 0.3]);
        let hoo = combine_narrowband(
            &h,
            &o,
            None,
            NarrowbandPalette::Hoo,
            &raw_options(),
            &ForaxxOptions::default(),
        )
        .unwrap();
        assert_eq!(hoo.image.data, [0.2, 0.3, 0.3]);
    }

    #[test]
    fn foraxx_hoo_matches_the_published_dynamic_formula() {
        let options = raw_options();
        let h = 0.6_f32;
        let o = 0.4_f32;
        let result = combine_narrowband(
            &mono(&[h]),
            &mono(&[o]),
            None,
            NarrowbandPalette::ForaxxHoo,
            &options,
            &ForaxxOptions::default(),
        )
        .unwrap();
        let product = h * o;
        let factor = product.powf(1.0 - product);
        let expected_green = factor * h + (1.0 - factor) * o;
        assert!((result.image.data[0] - h).abs() < 1.0e-5);
        assert!((result.image.data[1] - expected_green).abs() < 1.0e-5);
        assert!((result.image.data[2] - o).abs() < 1.0e-5);
        assert_eq!(result.transfer, ColorTransfer::DisplayReferred);
    }

    #[test]
    fn foraxx_uses_display_prepared_inputs_without_a_second_stretch() {
        let h = 0.6_f32;
        let o = 0.4_f32;
        let result = combine_narrowband(
            &mono(&[h]),
            &mono(&[o]),
            None,
            NarrowbandPalette::ForaxxHoo,
            &display_options(),
            &ForaxxOptions {
                target_median: f32::NAN,
                shadows_clip: 1.0,
            },
        )
        .unwrap();
        let product = h * o;
        let factor = product.powf(1.0 - product);
        let expected_green = factor.mul_add(h, (1.0 - factor) * o);
        assert_eq!(result.image.data[0], h);
        assert!((result.image.data[1] - expected_green).abs() < 1.0e-6);
        assert_eq!(result.image.data[2], o);
        assert_eq!(result.transfer, ColorTransfer::DisplayReferred);
    }

    #[test]
    fn foraxx_working_stretch_places_the_median_at_its_target() {
        let image = mono(&[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.90]);
        let channel = prepare_channel(&image, ColorNormalization::None).unwrap();
        let options = ForaxxOptions::default();
        let transform = display_transform(channel, &options, 100).unwrap();
        assert!((transform.map(0.30) - options.target_median).abs() < 1.0e-5);
    }

    #[test]
    fn foraxx_working_stretch_is_bit_identical_to_the_previous_path() {
        let options = ForaxxOptions::default();
        for (offset, scale) in [(0.02_f32, 0.45_f32), (0.55, 0.40)] {
            let values = (0..4_097)
                .map(|index| offset + scale * ((index * 7_919 % 4_099) as f32 / 4_098.0))
                .collect::<Vec<_>>();
            let image = mono(&values);
            let channel = prepare_channel(&image, ColorNormalization::None).unwrap();
            let previous = previous_display_transform(channel, &options, 2_000);
            let current = display_transform(channel, &options, 2_000).unwrap();
            for value in values {
                assert_eq!(current.map(value).to_bits(), previous.map(value).to_bits());
            }
        }
    }

    #[test]
    fn percentile_normalization_maps_ends_to_unit_range() {
        let options = ColorOptions {
            normalization: ColorNormalization::Percentile {
                black_percentile: 0.0,
                white_percentile: 1.0,
                max_samples: 100,
            },
            input_transfer: ColorTransfer::LinearLight,
        };
        let result = combine_rgb(
            &mono(&[10.0, 20.0]),
            &mono(&[100.0, 200.0]),
            &mono(&[-5.0, 5.0]),
            &options,
        )
        .unwrap();
        assert_eq!(result.image.data, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn display_prepared_inputs_reject_percentile_normalization() {
        let options = ColorOptions {
            input_transfer: ColorTransfer::DisplayReferred,
            ..ColorOptions::default()
        };
        let error = combine_rgb(
            &mono(&[0.1, 0.2]),
            &mono(&[0.3, 0.4]),
            &mono(&[0.5, 0.6]),
            &options,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("display-referred inputs must disable color normalization")
        );
    }

    #[test]
    fn missing_registered_samples_mask_the_whole_output_pixel() {
        let result = combine_rgb(
            &mono(&[f32::NAN]),
            &mono(&[0.4]),
            &mono(&[0.6]),
            &raw_options(),
        )
        .unwrap();
        assert!(result.image.data.iter().all(|sample| sample.is_nan()));
    }

    #[test]
    fn sho_requires_sii() {
        let error = combine_narrowband(
            &mono(&[0.2]),
            &mono(&[0.3]),
            None,
            NarrowbandPalette::Sho,
            &raw_options(),
            &ForaxxOptions::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("requires an SII"));
    }

    #[test]
    fn direct_palette_does_not_validate_unused_foraxx_options() {
        let invalid_foraxx = ForaxxOptions {
            target_median: f32::NAN,
            shadows_clip: 1.0,
        };
        let direct = combine_narrowband(
            &mono(&[0.2]),
            &mono(&[0.3]),
            None,
            NarrowbandPalette::Hoo,
            &raw_options(),
            &invalid_foraxx,
        )
        .unwrap();
        assert_eq!(direct.transfer, ColorTransfer::LinearLight);

        let error = combine_narrowband(
            &mono(&[0.2]),
            &mono(&[0.3]),
            None,
            NarrowbandPalette::ForaxxHoo,
            &raw_options(),
            &invalid_foraxx,
        )
        .unwrap_err();
        assert!(error.to_string().contains("Foraxx target median"));
    }

    #[test]
    fn foraxx_without_normalization_requires_unit_scaled_inputs() {
        let h = mono(&[1_000.0, 1_100.0, 1_200.0]);
        let o = mono(&[900.0, 1_000.0, 1_100.0]);
        let error = combine_narrowband(
            &h,
            &o,
            None,
            NarrowbandPalette::ForaxxHoo,
            &raw_options(),
            &ForaxxOptions::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("finite samples in [0, 1]"));
    }

    #[test]
    fn hoo_ignores_an_unused_sii_input() {
        let unused_sii = mono(&[f32::NAN, f32::NAN]);
        let result = combine_narrowband(
            &mono(&[0.2]),
            &mono(&[0.3]),
            Some(&unused_sii),
            NarrowbandPalette::Hoo,
            &raw_options(),
            &ForaxxOptions::default(),
        )
        .unwrap();
        assert_eq!(result.image.data, [0.2, 0.3, 0.3]);
    }

    #[test]
    fn custom_matrix_ignores_zero_weight_sii_input() {
        let matrix = NarrowbandMatrix {
            red: NarrowbandMix {
                ha: 1.0,
                ..NarrowbandMix::default()
            },
            green: NarrowbandMix {
                oiii: 1.0,
                ..NarrowbandMix::default()
            },
            blue: NarrowbandMix {
                ha: 0.5,
                oiii: 0.5,
                ..NarrowbandMix::default()
            },
        };
        let unused_sii = mono(&[f32::NAN, f32::NAN]);
        let result = combine_narrowband_matrix(
            &mono(&[0.2]),
            &mono(&[0.4]),
            Some(&unused_sii),
            matrix,
            &raw_options(),
        )
        .unwrap();
        assert_eq!(result.image.data, [0.2, 0.4, 0.3]);
    }
}
