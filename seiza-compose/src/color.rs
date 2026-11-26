use crate::crop::{ChannelSamples, ColorCrop, CropReport, crop_report};
use crate::picture::{
    ComputedMask, InputRead, Layer, LayerKind, LuminanceTarget, Picture, Transfer, TransferStage,
};
use crate::render::{ImageSource, RenderOptions, render_image};
use crate::{Error, LinearImage, ReferenceRegion, Result};
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
    /// How the output is trimmed to the area every channel covers.
    ///
    /// Cropping also restricts normalization statistics to the kept region, so
    /// every channel is scaled from the same patch of sky.
    pub crop: ColorCrop,
}

/// Display preparation applied before a dynamic Foraxx composition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ForaxxOptions {
    /// Display level assigned to the channel median before Foraxx composition.
    pub target_median: f32,
    /// Normal-equivalent MADs below the median used as the Foraxx black point.
    pub shadows_clip: f32,
    /// Standard deviation, in pixels, of a Gaussian blur applied to each
    /// Foraxx blend factor after it is computed and before it mixes the
    /// channels.
    ///
    /// The factors come from single pixels, so OIII noise otherwise shows as
    /// speckle in the mixed channels. `0.0`, the default, applies no blur and
    /// leaves the output as it was before this option existed. The largest
    /// accepted value is [`ForaxxOptions::MAX_MASK_SMOOTHING`]. Pixels that
    /// any channel does not cover carry no weight in the blur and stay
    /// non-finite in the output. Unlike the other fields, this one also
    /// applies to display-referred inputs.
    ///
    /// The blur runs on the composed region and mirrors it at the region's
    /// edges. With a [`ColorCrop`], the edge is the crop's, so a cropped
    /// composition differs slightly near its edge from the same region of an
    /// uncropped one.
    ///
    /// Time grows with the value, because the kernel spans four standard
    /// deviations each way. On a 26-megapixel image a few pixels add about a
    /// tenth of a second and the largest value adds several seconds, and the
    /// pass holds several extra planes of the image's size while it runs.
    pub mask_smoothing: f32,
}

impl ForaxxOptions {
    /// Largest accepted [`ForaxxOptions::mask_smoothing`], in pixels.
    pub const MAX_MASK_SMOOTHING: f32 = 64.0;
}

impl Default for ForaxxOptions {
    fn default() -> Self {
        Self {
            target_median: 0.2,
            shadows_clip: -2.8,
            mask_smoothing: 0.0,
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
    /// Where the composed image sits on the input channels' pixel grid.
    ///
    /// Without a crop this is the whole grid. A cropped composition needs its
    /// reference `CRPIX` shifted by the region origin, which
    /// [`write_color_fits_f32`](crate::write_color_fits_f32) does.
    pub region: ReferenceRegion,
    /// What each channel covered and how much the crop kept, when
    /// [`ColorOptions::crop`] asked for one. `None` for an uncropped
    /// composition, which measures no coverage.
    pub crop: Option<CropReport>,
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

impl NarrowbandMatrix {
    /// The label [`write_color_fits_f32`](crate::write_color_fits_f32) is
    /// given for a composition through a custom matrix.
    pub const LABEL: &'static str = "NARROWBAND-MATRIX";
}

/// How an RGB composition forms its luminance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RgbLuminance {
    /// Keep the composed channels as they are ([`combine_rgb`]).
    #[default]
    Native,
    /// Scale the triplet to a synthetic luminance of `R + G + B`
    /// ([`combine_super_rgb`]).
    Super,
}

impl RgbLuminance {
    /// The label [`write_color_fits_f32`](crate::write_color_fits_f32) is
    /// given for the composition: `RGB` or `SUPER-RGB`.
    pub fn label(self) -> &'static str {
        match self {
            Self::Native => "RGB",
            Self::Super => "SUPER-RGB",
        }
    }
}

/// How an LRGB composition forms its luminance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LrgbLuminance {
    /// Swap in L, weighted from 0 (RGB alone) to 1 (L alone)
    /// ([`combine_lrgb`]).
    Replace {
        /// The weight of L in the output luminance.
        luminance_weight: f32,
    },
    /// Target the additive `L + R + G + B` ([`combine_super_lrgb`]).
    Super,
}

impl LrgbLuminance {
    /// The mode a host asked for: super-luminance or replacement, with the
    /// luminance weight it gave, if any (1 when it gave none). A weight
    /// only applies to replacement, so super-luminance refuses any weight
    /// but 1.
    pub fn new(super_luminance: bool, luminance_weight: Option<f32>) -> Result<Self> {
        match (super_luminance, luminance_weight) {
            (true, Some(weight)) if weight != 1.0 => Err(Error::Color(
                "the luminance weight only applies when the luminance mode is replace".into(),
            )),
            (true, _) => Ok(Self::Super),
            (false, weight) => Ok(Self::Replace {
                luminance_weight: weight.unwrap_or(1.0),
            }),
        }
    }

    /// The label [`write_color_fits_f32`](crate::write_color_fits_f32) is
    /// given for the composition: `LRGB` or `SUPER-LRGB`.
    pub fn label(self) -> &'static str {
        match self {
            Self::Replace { .. } => "LRGB",
            Self::Super => "SUPER-LRGB",
        }
    }
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

    /// Check whether an SII channel was `supplied` as the palette needs:
    /// the three-filter palettes need one, and HOO and Foraxx-HOO use none.
    pub fn check_sii(self, supplied: bool) -> Result<()> {
        match (self.requires_sii(), supplied) {
            (true, false) => Err(Error::Color(format!(
                "{} requires an SII channel",
                self.name()
            ))),
            (false, true) => Err(Error::Color(format!(
                "{} does not use an SII channel",
                self.name()
            ))),
            _ => Ok(()),
        }
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
    let inputs = [("red", red), ("green", green), ("blue", blue)];
    validate_mono_set(&inputs)?;
    let grid = CompositionGrid::of(&inputs, options.crop)?;
    let window = grid.window();
    let channels = [
        prepare_channel(red, options.normalization, window)?,
        prepare_channel(green, options.normalization, window)?,
        prepare_channel(blue, options.normalization, window)?,
    ];
    let picture = Picture {
        layers: primaries(&channels),
        ..Picture::default()
    };
    let data = render_recipe(&picture, &[red, green, blue], window)?;
    grid.compose(data, options.input_transfer)
}

/// Red, green and blue layers of three prepared channels, inputs 0 to 2,
/// added.
fn primaries(channels: &[PreparedChannel<'_>; 3]) -> Vec<Layer> {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        .into_iter()
        .zip(channels)
        .enumerate()
        .map(|(input, (color, channel))| {
            Layer::added(InputRead::new(input, channel.transfer()), color)
        })
        .collect()
}

/// Renders a recipe's picture of `images` over the window: the one
/// compositor every composition draws through.
fn render_recipe(
    picture: &Picture,
    images: &[&LinearImage],
    window: ChannelWindow,
) -> Result<Vec<f32>> {
    let source = ImageSource::new(images.to_vec())?;
    Ok(render_image(picture, &source, window.region, &RenderOptions::default())?.data)
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
        Some(luminance),
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
    combine_lrgb_with_target(
        Some(luminance),
        red,
        green,
        blue,
        LrgbTarget::Super,
        options,
    )
}

/// Combine aligned R, G, and B stacks with a synthetic super-luminance.
///
/// No luminance stack is supplied; after channel normalization the target
/// luminance is `R + G + B`, as if a luminance exposure had collected every
/// channel's light. The RGB triplet is scaled to that luminance so its linear
/// chromaticity is retained. The linear `f32` result may exceed one.
pub fn combine_super_rgb(
    red: &LinearImage,
    green: &LinearImage,
    blue: &LinearImage,
    options: &ColorOptions,
) -> Result<ColorComposition> {
    combine_lrgb_with_target(None, red, green, blue, LrgbTarget::SyntheticSum, options)
}

#[derive(Clone, Copy)]
enum LrgbTarget {
    Replace { luminance_weight: f32 },
    Super,
    SyntheticSum,
}

fn combine_lrgb_with_target(
    luminance: Option<&LinearImage>,
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
    let mut inputs = Vec::with_capacity(4);
    if let Some(luminance) = luminance {
        inputs.push(("luminance", luminance));
    }
    inputs.extend([("red", red), ("green", green), ("blue", blue)]);
    validate_mono_set(&inputs)?;
    if matches!(
        target,
        LrgbTarget::Replace {
            luminance_weight: 0.0
        }
    ) {
        return combine_rgb(red, green, blue, options);
    }
    let grid = CompositionGrid::of(&inputs, options.crop)?;
    let window = grid.window();
    let l = luminance
        .map(|luminance| prepare_channel(luminance, options.normalization, window))
        .transpose()?;
    let rgb = [
        prepare_channel(red, options.normalization, window)?,
        prepare_channel(green, options.normalization, window)?,
        prepare_channel(blue, options.normalization, window)?,
    ];
    let mut layers = primaries(&rgb);
    let (target, weight) = match target {
        LrgbTarget::Replace { luminance_weight } => (LuminanceTarget::Replace, luminance_weight),
        LrgbTarget::Super | LrgbTarget::SyntheticSum => (LuminanceTarget::Super, 1.0),
    };
    layers.push(Layer {
        opacity: weight,
        ..Layer::new(LayerKind::Luminance {
            read: l.as_ref().map(|l| InputRead::new(3, l.transfer())),
            target,
        })
    });
    let picture = Picture {
        layers,
        ..Picture::default()
    };
    let mut images = vec![red, green, blue];
    images.extend(luminance);
    let data = render_recipe(&picture, &images, window)?;
    grid.compose(data, options.input_transfer)
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
    // An SII channel a palette does not use is set aside here rather than
    // refused; hosts refuse it first with check_sii.
    let sii = if palette.requires_sii() {
        palette.check_sii(sii.is_some())?;
        sii
    } else {
        None
    };
    let mut inputs = vec![("H-alpha", ha), ("OIII", oiii)];
    if let Some(sii) = sii {
        inputs.push(("SII", sii));
    }
    validate_mono_set(&inputs)?;
    let grid = CompositionGrid::of(&inputs, options.crop)?;
    let window = grid.window();
    let h = prepare_channel(ha, options.normalization, window)?;
    let o = prepare_channel(oiii, options.normalization, window)?;
    let s = sii
        .map(|image| prepare_channel(image, options.normalization, window))
        .transpose()?;

    if let Some(matrix) = palette.matrix() {
        return combine_prepared_matrix(
            grid,
            [(ha, h), (oiii, o)],
            sii.zip(s),
            matrix,
            options.input_transfer,
        );
    }

    validate_foraxx_mask_smoothing(foraxx)?;
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
    // Foraxx as a picture: each term of each channel is a display-prepared
    // input added under its blend factor, a computed mask of those inputs.
    let display = |input: usize, channel: &PreparedChannel<'_>, curve: ResolvedCurve| {
        let mut transfer = channel.transfer();
        transfer.stages.push(TransferStage::Curve(curve));
        InputRead::new(input, transfer)
    };
    let h_read = display(0, &h, h_display);
    let o_read = display(1, &o, o_display);
    let smoothing = foraxx.mask_smoothing;
    let mut computed = vec![ComputedMask::foraxx(
        vec![h_read.clone(), o_read.clone()],
        smoothing,
    )];
    let masked = |read: &InputRead, color: [f32; 3], mask: usize| Layer {
        computed: Some(mask),
        ..Layer::added(read.clone(), color)
    };
    // The green factor's mask, and its inverse for OIII's share.
    computed.push(ComputedMask {
        inverted: true,
        ..computed[0].clone()
    });
    // Each channel's terms in the order Foraxx sums them: the share of the
    // second input, then the factor's share of the first, fused.
    let mut layers = vec![
        masked(&o_read, [0.0, 1.0, 0.0], 1),
        masked(&h_read, [0.0, 1.0, 0.0], 0),
        Layer::added(o_read.clone(), [0.0, 0.0, 1.0]),
    ];
    let mut images = vec![ha, oiii];
    match palette {
        NarrowbandPalette::ForaxxHoo => layers.push(Layer::added(h_read.clone(), [1.0, 0.0, 0.0])),
        NarrowbandPalette::ForaxxSho => {
            let (sii_image, sii) = sii.zip(s.as_ref()).expect("SII was validated");
            let s_read = display(2, sii, s_display.expect("SII was validated"));
            // The red factor is OIII's alone.
            computed.push(ComputedMask::foraxx(vec![o_read.clone()], smoothing));
            computed.push(ComputedMask {
                inverted: true,
                ..computed[2].clone()
            });
            layers.push(masked(&h_read, [1.0, 0.0, 0.0], 3));
            layers.push(masked(&s_read, [1.0, 0.0, 0.0], 2));
            images.push(sii_image);
        }
        _ => unreachable!("static palettes returned above"),
    }
    let picture = Picture {
        layers,
        computed,
        ..Picture::default()
    };
    let data = render_recipe(&picture, &images, window)?;
    grid.compose(data, palette.transfer())
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
    let grid = CompositionGrid::of(&inputs, options.crop)?;
    let window = grid.window();
    let h = prepare_channel(ha, options.normalization, window)?;
    let o = prepare_channel(oiii, options.normalization, window)?;
    let s = sii
        .map(|image| prepare_channel(image, options.normalization, window))
        .transpose()?;
    combine_prepared_matrix(
        grid,
        [(ha, h), (oiii, o)],
        sii.zip(s),
        matrix,
        options.input_transfer,
    )
}

fn combine_prepared_matrix(
    grid: CompositionGrid,
    channels: [(&LinearImage, PreparedChannel<'_>); 2],
    sii: Option<(&LinearImage, PreparedChannel<'_>)>,
    matrix: NarrowbandMatrix,
    transfer: ColorTransfer,
) -> Result<ColorComposition> {
    let window = grid.window();
    let [(ha_image, ha), (oiii_image, oiii)] = channels;
    let column = |weight: fn(&NarrowbandMix) -> f32| {
        [matrix.red, matrix.green, matrix.blue].map(|mix| weight(&mix))
    };
    // OIII, then H-alpha, then SII, added: each a weighted column of the
    // matrix.
    let mut layers = vec![
        Layer::added(InputRead::new(1, oiii.transfer()), column(|mix| mix.oiii)),
        Layer::added(InputRead::new(0, ha.transfer()), column(|mix| mix.ha)),
    ];
    let mut images = vec![ha_image, oiii_image];
    if let Some((sii_image, sii)) = &sii {
        layers.push(Layer::added(
            InputRead::new(2, sii.transfer()),
            column(|mix| mix.sii),
        ));
        images.push(sii_image);
    }
    let picture = Picture {
        layers,
        ..Picture::default()
    };
    let data = render_recipe(&picture, &images, window)?;
    grid.compose(data, transfer)
}

/// The part of the shared input grid a composition reads and writes.
///
/// Composition indexes output pixels; this maps each of them back to its
/// sample in the full-size input channels, so a crop needs no copy of the
/// inputs.
#[derive(Clone, Copy)]
struct ChannelWindow {
    source_width: usize,
    region: ReferenceRegion,
    whole_grid: bool,
}

impl ChannelWindow {
    fn pixel_count(self) -> usize {
        self.region.width * self.region.height
    }

    fn source_index(self, index: usize) -> usize {
        if self.whole_grid {
            return index;
        }
        let row = index / self.region.width;
        let column = index - row * self.region.width;
        (self.region.y + row) * self.source_width + self.region.x + column
    }
}

/// The output grid of one composition, with the crop diagnostics that chose it.
struct CompositionGrid {
    window: ChannelWindow,
    report: Option<CropReport>,
}

impl CompositionGrid {
    fn of(inputs: &[(&str, &LinearImage)], crop: ColorCrop) -> Result<Self> {
        let source = inputs
            .first()
            .ok_or_else(|| Error::Color("at least one input channel is required".into()))?
            .1;
        let report = match crop {
            ColorCrop::None => None,
            _ => {
                let samples = inputs
                    .iter()
                    .map(|(name, image)| ChannelSamples::new(name, image))
                    .collect::<Vec<_>>();
                Some(crop_report(&samples, crop)?)
            }
        };
        let region = match &report {
            Some(report) => report.region,
            None => ReferenceRegion {
                x: 0,
                y: 0,
                width: source.width,
                height: source.height,
            },
        };
        Ok(Self {
            window: ChannelWindow {
                source_width: source.width,
                region,
                whole_grid: region.x == 0
                    && region.y == 0
                    && region.width == source.width
                    && region.height == source.height,
            },
            report,
        })
    }

    fn window(&self) -> ChannelWindow {
        self.window
    }

    fn compose(self, data: Vec<f32>, transfer: ColorTransfer) -> Result<ColorComposition> {
        let region = self.window.region;
        Ok(ColorComposition {
            image: LinearImage::new(region.width, region.height, 3, data)?,
            transfer,
            region,
            crop: self.report,
        })
    }
}

#[derive(Clone, Copy)]
struct PreparedChannel<'a> {
    values: &'a [f32],
    transform: ChannelTransform,
    window: ChannelWindow,
}

impl PreparedChannel<'_> {
    /// Whether this channel covers the pixel at a source index.
    ///
    /// Every channel of one composition shares a window, so the compose loops
    /// map an output pixel to its source index once and hand it to each
    /// channel rather than repeating that division per channel.
    fn valid(self, source: usize) -> bool {
        self.values[source].is_finite()
    }

    fn sample(self, source: usize) -> f32 {
        let value = self.values[source];
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

    /// Every `stride`-th sample of the window, transform applied, skipping
    /// pixels this channel does not cover.
    fn samples(self, max_samples: usize) -> Vec<f32> {
        let count = self.window.pixel_count();
        let stride = count.div_ceil(max_samples.max(1)).max(1);
        (0..count)
            .step_by(stride)
            .map(|index| self.window.source_index(index))
            .filter(|source| self.valid(*source))
            .map(|source| self.sample(source))
            .collect()
    }
}

impl PreparedChannel<'_> {
    /// The channel's normalization as a transfer of its raw samples.
    fn transfer(&self) -> Transfer {
        match self.transform {
            ChannelTransform::Identity => Transfer::identity(),
            ChannelTransform::Percentile {
                black,
                reciprocal_range,
            } => Transfer::of([TransferStage::Normalize {
                black,
                reciprocal_range,
            }]),
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
    window: ChannelWindow,
) -> Result<PreparedChannel<'_>> {
    let raw = PreparedChannel {
        values: &image.data,
        transform: ChannelTransform::Identity,
        window,
    };
    let transform = match normalization {
        ColorNormalization::None => ChannelTransform::Identity,
        ColorNormalization::Percentile {
            black_percentile,
            white_percentile,
            max_samples,
        } => {
            let (black, white) =
                percentile_levels(raw.samples(max_samples), black_percentile, white_percentile)?;
            ChannelTransform::Percentile {
                black,
                reciprocal_range: 1.0 / (white - black),
            }
        }
    };
    Ok(PreparedChannel { transform, ..raw })
}

fn percentile_levels(
    mut sample: Vec<f32>,
    black_percentile: f32,
    white_percentile: f32,
) -> Result<(f32, f32)> {
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
        && (0..channel.window.pixel_count())
            .into_par_iter()
            .map(|index| channel.window.source_index(index))
            .filter(|source| channel.valid(*source))
            .any(|source| !(0.0..=1.0).contains(&channel.sample(source)))
    {
        return Err(Error::Color(
            "Foraxx with normalization disabled requires finite samples in [0, 1]; use percentile normalization for sensor-unit inputs"
                .into(),
        ));
    }
    let sample = channel.samples(max_samples);
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

/// Checked apart from [`validate_foraxx_options`] because the smoothing also
/// applies to display-referred inputs, which skip the stretch options.
fn validate_foraxx_mask_smoothing(options: &ForaxxOptions) -> Result<()> {
    if !(0.0..=ForaxxOptions::MAX_MASK_SMOOTHING).contains(&options.mask_smoothing) {
        return Err(Error::Color(format!(
            "Foraxx mask smoothing must be finite and between zero and {} pixels",
            ForaxxOptions::MAX_MASK_SMOOTHING
        )));
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

    fn whole_grid(image: &LinearImage) -> ChannelWindow {
        CompositionGrid::of(&[("channel", image)], ColorCrop::None)
            .unwrap()
            .window()
    }

    fn raw_options() -> ColorOptions {
        ColorOptions {
            normalization: ColorNormalization::None,
            input_transfer: ColorTransfer::LinearLight,
            crop: ColorCrop::None,
        }
    }

    fn display_options() -> ColorOptions {
        ColorOptions {
            normalization: ColorNormalization::None,
            input_transfer: ColorTransfer::DisplayReferred,
            crop: ColorCrop::None,
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
        let mut sample = channel.samples(max_samples);
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
    fn super_rgb_targets_channel_sum_and_preserves_chromaticity() {
        let r = 0.2;
        let g = 0.4;
        let b = 0.1;
        let result =
            combine_super_rgb(&mono(&[r]), &mono(&[g]), &mono(&[b]), &raw_options()).unwrap();
        let pixel = &result.image.data;
        let output_luminance = crate::render::luma([pixel[0], pixel[1], pixel[2]]);
        assert!((output_luminance - (r + g + b)).abs() < 1.0e-6);
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
                mask_smoothing: 0.0,
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

    fn plane(width: usize, height: usize, value: impl Fn(usize) -> f32) -> LinearImage {
        let data = (0..width * height).map(value).collect();
        LinearImage::new(width, height, 1, data).unwrap()
    }

    /// Uniform values in `[0, 1)` that depend only on the index and the seed.
    fn uniform(index: usize, seed: u64) -> f32 {
        let mut state = (index as u64 ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .wrapping_mul(0xBF58_476D_1CE4_E5B9);
        state ^= state >> 29;
        state = state.wrapping_mul(0x94D0_49BB_1331_11EB);
        state ^= state >> 32;
        (state >> 40) as f32 / (1_u64 << 24) as f32
    }

    fn smoothed(mask_smoothing: f32) -> ForaxxOptions {
        ForaxxOptions {
            mask_smoothing,
            ..ForaxxOptions::default()
        }
    }

    fn variance(values: impl Iterator<Item = f32> + Clone) -> f64 {
        let count = values.clone().count() as f64;
        let mean = values.clone().map(f64::from).sum::<f64>() / count;
        values
            .map(|value| (f64::from(value) - mean).powi(2))
            .sum::<f64>()
            / count
    }

    #[test]
    fn foraxx_without_mask_smoothing_is_bit_identical_to_the_per_pixel_formula() {
        let (width, height) = (37, 23);
        let h = plane(width, height, |index| 0.2 + 0.6 * uniform(index, 1));
        let mut o = plane(width, height, |index| 0.1 + 0.7 * uniform(index, 2));
        let s = plane(width, height, |index| 0.1 + 0.5 * uniform(index, 3));
        o.data[40] = f32::NAN;
        for palette in [NarrowbandPalette::ForaxxSho, NarrowbandPalette::ForaxxHoo] {
            let sii = palette.requires_sii().then_some(&s);
            let result =
                combine_narrowband(&h, &o, sii, palette, &display_options(), &smoothed(0.0))
                    .unwrap();
            assert_eq!(ForaxxOptions::default().mask_smoothing, 0.0);
            for (index, pixel) in result.image.data.as_chunks::<3>().0.iter().enumerate() {
                if index == 40 {
                    assert!(pixel.iter().all(|value| value.is_nan()));
                    continue;
                }
                // The expressions this function used before mask smoothing
                // existed, written out again.
                let (h, o, s) = (h.data[index], o.data[index], s.data[index]);
                let product = (o * h).clamp(0.0, 1.0);
                let green_factor = product.powf(1.0 - product);
                let green = green_factor.mul_add(h, (1.0 - green_factor) * o);
                let red = if palette == NarrowbandPalette::ForaxxSho {
                    let o_factor = o.powf(1.0 - o);
                    o_factor.mul_add(s, (1.0 - o_factor) * h)
                } else {
                    h
                };
                assert_eq!(
                    [pixel[0].to_bits(), pixel[1].to_bits(), pixel[2].to_bits()],
                    [red.to_bits(), green.to_bits(), o.to_bits()]
                );
            }
        }
    }

    #[test]
    fn foraxx_mask_smoothing_reduces_the_speckle_from_oiii_noise() {
        // 160 by 120 is above the size at which the blur runs in parallel.
        let (width, height) = (160, 120);
        let h = plane(width, height, |_| 0.7);
        let s = plane(width, height, |_| 0.2);
        let o = plane(width, height, |index| 0.2 + 0.3 * uniform(index, 7));
        let compose = |mask_smoothing| {
            combine_narrowband(
                &h,
                &o,
                Some(&s),
                NarrowbandPalette::ForaxxSho,
                &display_options(),
                &smoothed(mask_smoothing),
            )
            .unwrap()
            .image
            .data
        };
        let plain = compose(0.0);
        let smooth = compose(3.0);
        // H-alpha and SII are flat, so red varies only through the OIII
        // factor.
        let red = |data: &[f32]| variance(data.as_chunks::<3>().0.iter().map(|pixel| pixel[0]));
        assert!(red(&plain) > 1.0e-4);
        assert!(red(&smooth) < red(&plain) / 20.0);
        // Green still carries OIII itself. What smoothing removes is the
        // factor's share: the factor implied by each pixel varies less.
        let green_factor = |data: &[f32]| {
            variance(
                data.as_chunks::<3>()
                    .0
                    .iter()
                    .map(|pixel| (pixel[1] - pixel[2]) / (0.7 - pixel[2])),
            )
        };
        assert!(green_factor(&smooth) < green_factor(&plain) / 20.0);
        // Blue is OIII, which smoothing does not touch.
        for (plain, smooth) in plain
            .as_chunks::<3>()
            .0
            .iter()
            .zip(smooth.as_chunks::<3>().0.iter())
        {
            assert_eq!(plain[2].to_bits(), smooth[2].to_bits());
        }
    }

    #[test]
    fn foraxx_mask_smoothing_leaves_a_constant_image_unchanged() {
        let (width, height) = (160, 120);
        let h = plane(width, height, |_| 0.6);
        let o = plane(width, height, |_| 0.4);
        let s = plane(width, height, |_| 0.3);
        // One with every pixel covered, one with a hole, which takes the
        // weighted path.
        for hole in [None, Some(width * 60 + 80)] {
            let mut o = o.clone();
            if let Some(hole) = hole {
                o.data[hole] = f32::NAN;
            }
            let compose = |mask_smoothing| {
                combine_narrowband(
                    &h,
                    &o,
                    Some(&s),
                    NarrowbandPalette::ForaxxSho,
                    &display_options(),
                    &smoothed(mask_smoothing),
                )
                .unwrap()
                .image
                .data
            };
            for (index, (plain, smooth)) in compose(0.0).iter().zip(compose(5.0)).enumerate() {
                if Some(index / 3) == hole {
                    assert!(plain.is_nan() && smooth.is_nan());
                } else {
                    assert!((plain - smooth).abs() < 1.0e-5, "sample {index}");
                }
            }
        }
    }

    #[test]
    fn foraxx_mask_smoothing_ignores_uncovered_pixels_and_keeps_them_uncovered() {
        // OIII is bright on the left half and faint on the right, and each
        // channel in turn does not cover a block inside the left half.
        let (width, height) = (64, 48);
        let h = plane(width, height, |_| 0.7);
        let s = plane(width, height, |_| 0.2);
        let o = plane(
            width,
            height,
            |index| if index % width < 32 { 0.8 } else { 0.1 },
        );
        let uncovered = |index: usize| {
            let (x, y) = (index % width, index / width);
            (8..20).contains(&x) && (10..30).contains(&y)
        };
        let punch = |image: &LinearImage| {
            let mut image = image.clone();
            for (index, value) in image.data.iter_mut().enumerate() {
                if uncovered(index) {
                    *value = f32::NAN;
                }
            }
            image
        };
        let compose = |h: &LinearImage, o: &LinearImage, s: &LinearImage| {
            combine_narrowband(
                h,
                o,
                Some(s),
                NarrowbandPalette::ForaxxSho,
                &display_options(),
                &smoothed(2.0),
            )
            .unwrap()
            .image
            .data
        };
        let whole = compose(&h, &o, &s);
        for holed in [
            compose(&punch(&h), &o, &s),
            compose(&h, &punch(&o), &s),
            compose(&h, &o, &punch(&s)),
        ] {
            for (index, (pixel, whole)) in holed
                .as_chunks::<3>()
                .0
                .iter()
                .zip(whole.as_chunks::<3>().0.iter())
                .enumerate()
            {
                if uncovered(index) {
                    assert!(pixel.iter().all(|value| value.is_nan()), "pixel {index}");
                    continue;
                }
                assert!(pixel.iter().all(|value| value.is_finite()), "pixel {index}");
                // Left of column 24 the blur reaches only bright OIII, whose
                // factors are all equal, so a hole that leaked zeros, or
                // that took weight, would move these pixels.
                if index % width < 24 {
                    for (value, whole) in pixel.iter().zip(whole) {
                        assert!((value - whole).abs() < 1.0e-5, "pixel {index}");
                    }
                }
            }
        }
    }

    #[test]
    fn foraxx_mask_smoothing_of_a_crop_equals_smoothing_the_cropped_inputs() {
        // SII lacks a border three pixels wide on the left and two on the
        // top, which an inscribed crop removes.
        let (width, height) = (40, 30);
        let h = plane(width, height, |index| 0.2 + 0.6 * uniform(index, 11));
        let o = plane(width, height, |index| 0.1 + 0.7 * uniform(index, 12));
        let s = plane(width, height, |index| {
            if index % width < 3 || index / width < 2 {
                f32::NAN
            } else {
                0.1 + 0.5 * uniform(index, 13)
            }
        });
        let compose = |h: &LinearImage, o: &LinearImage, s: &LinearImage, crop| {
            combine_narrowband(
                h,
                o,
                Some(s),
                NarrowbandPalette::ForaxxSho,
                &ColorOptions {
                    crop,
                    ..display_options()
                },
                &smoothed(2.0),
            )
            .unwrap()
        };
        let cropped = compose(&h, &o, &s, ColorCrop::Inscribed);
        let region = cropped.region;
        assert_eq!(
            region,
            ReferenceRegion {
                x: 3,
                y: 2,
                width: width - 3,
                height: height - 2
            }
        );
        let cut = |image: &LinearImage| {
            plane(region.width, region.height, |index| {
                let (x, y) = (index % region.width, index / region.width);
                image.data[(region.y + y) * width + region.x + x]
            })
        };
        let from_cut = compose(&cut(&h), &cut(&o), &cut(&s), ColorCrop::None);
        assert!(cropped.image.data.iter().all(|value| value.is_finite()));
        assert_eq!(
            cropped
                .image
                .data
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            from_cut
                .image
                .data
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn foraxx_mask_smoothing_must_be_finite_and_within_its_range() {
        let compose = |options: &ColorOptions, mask_smoothing| {
            combine_narrowband(
                &mono(&[0.6, 0.5]),
                &mono(&[0.4, 0.3]),
                None,
                NarrowbandPalette::ForaxxHoo,
                options,
                &smoothed(mask_smoothing),
            )
        };
        for options in [raw_options(), display_options()] {
            for invalid in [
                -0.5,
                f32::NAN,
                f32::INFINITY,
                ForaxxOptions::MAX_MASK_SMOOTHING + 0.5,
            ] {
                let error = compose(&options, invalid).unwrap_err();
                assert!(
                    error.to_string().contains("Foraxx mask smoothing"),
                    "{invalid}: {error}"
                );
            }
            for valid in [0.0, 0.5, ForaxxOptions::MAX_MASK_SMOOTHING] {
                compose(&options, valid).unwrap();
            }
        }
        // A static palette does not read the option.
        combine_narrowband(
            &mono(&[0.6]),
            &mono(&[0.4]),
            None,
            NarrowbandPalette::Hoo,
            &raw_options(),
            &smoothed(f32::NAN),
        )
        .unwrap();
    }

    #[test]
    fn foraxx_working_stretch_places_the_median_at_its_target() {
        let image = mono(&[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.90]);
        let channel =
            prepare_channel(&image, ColorNormalization::None, whole_grid(&image)).unwrap();
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
            let channel =
                prepare_channel(&image, ColorNormalization::None, whole_grid(&image)).unwrap();
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
            ..ColorOptions::default()
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

    fn grid(width: usize, height: usize, values: &[f32]) -> LinearImage {
        LinearImage::new(width, height, 1, values.to_vec()).unwrap()
    }

    #[test]
    fn an_uncropped_composition_covers_the_whole_grid() {
        let result = combine_rgb(
            &mono(&[0.1, 0.2]),
            &mono(&[0.3, 0.4]),
            &mono(&[0.5, 0.6]),
            &raw_options(),
        )
        .unwrap();
        assert_eq!(
            result.region,
            ReferenceRegion {
                x: 0,
                y: 0,
                width: 2,
                height: 1
            }
        );
    }

    #[test]
    fn cropping_trims_the_registration_border_from_every_channel() {
        let nan = f32::NAN;
        let red = grid(3, 3, &[0.1; 9]);
        // Green is offset one pixel right, blue one pixel down.
        let green = grid(3, 3, &[nan, 0.2, 0.2, nan, 0.2, 0.2, nan, 0.2, 0.2]);
        let blue = grid(3, 3, &[nan, nan, nan, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]);
        let options = ColorOptions {
            crop: ColorCrop::Inscribed,
            ..raw_options()
        };
        let result = combine_rgb(&red, &green, &blue, &options).unwrap();
        assert_eq!(
            result.region,
            ReferenceRegion {
                x: 1,
                y: 1,
                width: 2,
                height: 2
            }
        );
        assert_eq!(result.image.width, 2);
        assert_eq!(result.image.height, 2);
        assert!(result.image.data.iter().all(|sample| sample.is_finite()));
        assert_eq!(result.image.data[..3], [0.1, 0.2, 0.3]);
    }

    #[test]
    fn cropping_normalizes_from_the_kept_region_only() {
        let nan = f32::NAN;
        // The bright pixel sits in the border that the crop removes, so it
        // must not set the white point of the kept region.
        let red = grid(3, 1, &[100.0, 0.0, 1.0]);
        let green = grid(3, 1, &[nan, 0.0, 1.0]);
        let blue = grid(3, 1, &[nan, 0.0, 1.0]);
        let options = ColorOptions {
            normalization: ColorNormalization::Percentile {
                black_percentile: 0.0,
                white_percentile: 1.0,
                max_samples: 100,
            },
            crop: ColorCrop::Bounds,
            ..ColorOptions::default()
        };
        let result = combine_rgb(&red, &green, &blue, &options).unwrap();
        assert_eq!(
            result.region,
            ReferenceRegion {
                x: 1,
                y: 0,
                width: 2,
                height: 1
            }
        );
        assert_eq!(result.image.data, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn narrowband_cropping_ignores_a_channel_the_palette_does_not_use() {
        let nan = f32::NAN;
        let ha = grid(2, 1, &[0.2, 0.2]);
        let oiii = grid(2, 1, &[0.3, 0.3]);
        let unused_sii = grid(2, 1, &[nan, nan]);
        let options = ColorOptions {
            crop: ColorCrop::Inscribed,
            ..raw_options()
        };
        let result = combine_narrowband(
            &ha,
            &oiii,
            Some(&unused_sii),
            NarrowbandPalette::Hoo,
            &options,
            &ForaxxOptions::default(),
        )
        .unwrap();
        assert_eq!(result.region.width, 2);
        assert_eq!(result.image.data, [0.2, 0.3, 0.3, 0.2, 0.3, 0.3]);
    }

    #[test]
    fn a_composition_carries_its_crop_diagnostics() {
        let covered = |shift: usize| {
            let mut values = vec![1.0_f32; 256 * 256];
            for row in 0..shift {
                for column in 0..256 {
                    values[row * 256 + column] = f32::NAN;
                }
            }
            LinearImage::new(256, 256, 1, values).unwrap()
        };
        let options = ColorOptions {
            crop: ColorCrop::Inscribed,
            ..raw_options()
        };
        let result = combine_rgb(&covered(0), &covered(1), &covered(90), &options).unwrap();
        let report = result.crop.expect("a cropped composition reports coverage");
        assert_eq!(report.region, result.region);
        assert_eq!(report.region.y, 90);
        assert!(report.retained_fraction() > 0.6);
        let flagged = report.off_center().collect::<Vec<_>>();
        assert_eq!(flagged.len(), 1);
        assert_eq!(flagged[0].name, "blue");
        assert!(flagged[0].center_offset_pixels() > 40.0);
    }

    #[test]
    fn an_uncropped_composition_reports_no_coverage() {
        let result = combine_rgb(
            &mono(&[0.1, 0.2]),
            &mono(&[0.3, 0.4]),
            &mono(&[0.5, 0.6]),
            &raw_options(),
        )
        .unwrap();
        assert!(result.crop.is_none());
    }

    #[test]
    fn cropping_rejects_channels_that_share_no_covered_pixel() {
        let nan = f32::NAN;
        let options = ColorOptions {
            crop: ColorCrop::Inscribed,
            ..raw_options()
        };
        let error = combine_rgb(
            &grid(2, 1, &[0.1, nan]),
            &grid(2, 1, &[nan, 0.2]),
            &grid(2, 1, &[0.3, 0.3]),
            &options,
        )
        .unwrap_err();
        assert!(error.to_string().contains("no pixel is covered"));
    }

    #[test]
    fn modes_carry_their_labels_and_refuse_a_weight_super_luminance_ignores() {
        assert_eq!(RgbLuminance::Native.label(), "RGB");
        assert_eq!(RgbLuminance::Super.label(), "SUPER-RGB");
        assert_eq!(
            LrgbLuminance::new(false, None).unwrap(),
            LrgbLuminance::Replace {
                luminance_weight: 1.0
            }
        );
        assert_eq!(
            LrgbLuminance::new(false, Some(0.4)).unwrap(),
            LrgbLuminance::Replace {
                luminance_weight: 0.4
            }
        );
        assert_eq!(
            LrgbLuminance::new(true, Some(1.0)).unwrap(),
            LrgbLuminance::Super
        );
        assert_eq!(
            LrgbLuminance::new(true, None).unwrap().label(),
            "SUPER-LRGB"
        );
        assert_eq!(LrgbLuminance::new(false, None).unwrap().label(), "LRGB");
        let error = LrgbLuminance::new(true, Some(0.5)).unwrap_err();
        assert!(error.to_string().contains("only applies"), "{error}");
    }

    #[test]
    fn palettes_check_the_sii_channel() {
        assert!(NarrowbandPalette::Sho.check_sii(true).is_ok());
        assert!(NarrowbandPalette::Hoo.check_sii(false).is_ok());
        let missing = NarrowbandPalette::ForaxxSho.check_sii(false).unwrap_err();
        assert!(
            missing
                .to_string()
                .contains("FORAXX-SHO requires an SII channel")
        );
        let unused = NarrowbandPalette::Hoo.check_sii(true).unwrap_err();
        assert!(
            unused
                .to_string()
                .contains("HOO does not use an SII channel")
        );
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
            mask_smoothing: 0.0,
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
