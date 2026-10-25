//! Explicit emission projection and display structure blending.
use crate::{ContinuumPlan, Error, LinearImage, Result};
use serde::{Deserialize, Serialize};

/// Signed residuals remain unchanged. This policy controls only their blend.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmissionNegativePolicy {
    #[default]
    Preserve,
    PositiveContribution,
}

/// One residual's exact continuum plan and its broadband destinations.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EmissionBlendLine {
    pub plan: ContinuumPlan,
    /// RGB channel index for each reference in the plan, in the same order.
    #[serde(alias = "reference_channels")]
    pub reference_channels: Vec<usize>,
    pub amount: f64,
    /// Explicit display baseline in residual units; the signed residual is untouched.
    #[serde(default, alias = "background_offset")]
    pub background_offset: f64,
}

/// Rendering policy, separate from scientific signed residual artifacts.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct EmissionBlendOptions {
    pub negative: EmissionNegativePolicy,
}

fn error(message: &str) -> Error {
    Error::Color(message.into())
}

fn reference_scales(plan: &ContinuumPlan) -> Result<Vec<f64>> {
    crate::continuum::validate_plan(plan)?;
    let n = plan.sources.references.len();
    if plan.schema_version != 1
        || n == 0
        || n > 3
        || plan.coefficients.len() != n
        || plan.coefficients.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(error("Emission blending requires a valid continuum plan"));
    }
    let ratios = if plan.reference_flux_ratios.is_empty() && n == 1 {
        vec![1.0]
    } else {
        plan.reference_flux_ratios.clone()
    };
    if ratios.len() != n || ratios.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(error(
            "Multi-reference emission projection needs measured reference flux ratios or explicit manual ratios",
        ));
    }
    let scale = plan
        .coefficients
        .iter()
        .zip(&ratios)
        .map(|(a, r)| a * r)
        .sum::<f64>();
    let scales: Vec<_> = ratios.iter().map(|r| scale / r).collect();
    if scales.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(error(
            "Emission reference scales must be finite and positive",
        ));
    }
    Ok(scales)
}

/// Add signed narrowband residuals in each reference channel's actual units.
/// For stellar ratios r_j=F_Bj/F_B0, k_j=sum(a_i*r_i)/r_j and the
/// contribution is amount*E/k_j. This commutes with independent input gains.
/// No throughput correction, hidden pedestal or implicit image normalization
/// is applied. Nonfinite contributing coverage masks the complete RGB pixel.
/// PositiveContribution clamps only the added residual, never its stored image.
pub fn blend_emission_rgb(
    broadband: &LinearImage,
    residuals: &[&LinearImage],
    lines: &[EmissionBlendLine],
    options: &EmissionBlendOptions,
) -> Result<LinearImage> {
    if broadband.channels != 3
        || broadband.width == 0
        || broadband.height == 0
        || broadband
            .width
            .checked_mul(broadband.height)
            .and_then(|n| n.checked_mul(3))
            != Some(broadband.data.len())
        || residuals.len() != lines.len()
    {
        return Err(error(
            "Emission blending requires RGB broadband and one mono residual per line",
        ));
    }
    let mut prepared = Vec::new();
    for (image, line) in residuals.iter().zip(lines) {
        if !line.amount.is_finite()
            || !(0.0..=5.0).contains(&line.amount)
            || !line.background_offset.is_finite()
        {
            return Err(error("Emission amounts must be between zero and five"));
        }
        if line.amount == 0.0 {
            continue;
        }
        if image.channels != 1
            || (image.width, image.height) != (broadband.width, broadband.height)
            || image.data.len() != broadband.width * broadband.height
            || (line.plan.sources.width, line.plan.sources.height) != (image.width, image.height)
            || line.reference_channels.len() != line.plan.coefficients.len()
            || line.reference_channels.iter().any(|channel| *channel > 2)
            || line
                .reference_channels
                .iter()
                .enumerate()
                .any(|(i, c)| line.reference_channels[..i].contains(c))
        {
            return Err(error(
                "Emission residuals, reference channels and plan grids must match the broadband grid",
            ));
        }
        prepared.push((*image, line, reference_scales(&line.plan)?));
    }
    let mut data = Vec::with_capacity(broadband.data.len());
    for (pixel, input) in broadband.data.as_chunks::<3>().0.iter().enumerate() {
        if input.iter().any(|v| !v.is_finite())
            || prepared
                .iter()
                .any(|(image, _, _)| !image.data[pixel].is_finite())
        {
            data.extend([f32::NAN; 3]);
            continue;
        }
        let mut out = [
            f64::from(input[0]),
            f64::from(input[1]),
            f64::from(input[2]),
        ];
        for (image, line, scales) in &prepared {
            let residual = f64::from(image.data[pixel]) - line.background_offset;
            let residual = match options.negative {
                EmissionNegativePolicy::Preserve => residual,
                EmissionNegativePolicy::PositiveContribution => residual.max(0.0),
            };
            for (channel, scale) in line.reference_channels.iter().zip(scales) {
                out[*channel] += line.amount * residual / scale;
            }
        }
        if out
            .iter()
            .any(|v| !v.is_finite() || v.abs() > f64::from(f32::MAX))
        {
            return Err(error("Emission blend overflows the output sample range"));
        }
        data.extend(out.map(|v| v as f32));
    }
    LinearImage::new(broadband.width, broadband.height, 3, data)
}

/// Blend prepared starless display RGB structures, before restoring stars.
/// Amount zero preserves the narrowband branch and ignores broadband gaps;
/// amount one does the reverse. Intermediate amounts require common coverage.
/// Inputs must already share a grid and bounded [0,1] display transfer.
pub fn blend_display_structure(
    narrowband: &LinearImage,
    broadband: &LinearImage,
    amount: f64,
) -> Result<LinearImage> {
    if !amount.is_finite()
        || !(0.0..=1.0).contains(&amount)
        || narrowband.channels != 3
        || broadband.channels != 3
        || (narrowband.width, narrowband.height) != (broadband.width, broadband.height)
        || narrowband.data.len() != broadband.data.len()
        || narrowband
            .width
            .checked_mul(narrowband.height)
            .and_then(|n| n.checked_mul(3))
            != Some(narrowband.data.len())
    {
        return Err(error(
            "Display structure blending needs matching RGB grids and an amount from zero to one",
        ));
    }
    for image in [narrowband, broadband] {
        if image
            .data
            .iter()
            .any(|v| v.is_finite() && !(0.0..=1.0).contains(v))
        {
            return Err(error("Display structures must be bounded in [0,1]"));
        }
    }
    let mut data = Vec::with_capacity(narrowband.data.len());
    for (n, b) in narrowband
        .data
        .as_chunks::<3>()
        .0
        .iter()
        .zip(broadband.data.as_chunks::<3>().0.iter())
    {
        if (amount < 1.0 && n.iter().any(|v| !v.is_finite()))
            || (amount > 0.0 && b.iter().any(|v| !v.is_finite()))
        {
            data.extend([f32::NAN; 3]);
            continue;
        }
        data.extend((0..3).map(|i| {
            if amount == 0.0 {
                n[i]
            } else if amount == 1.0 {
                b[i]
            } else {
                ((1.0 - amount) * f64::from(n[i]) + amount * f64::from(b[i])) as f32
            }
        }));
    }
    LinearImage::new(narrowband.width, narrowband.height, 3, data)
}

/// Add an RGB contribution using gains measured from the broadband stars.
/// Offsets belong to the calibrated base; only multiplicative gains affect
/// the contribution. Both images share a grid and interleaved linear RGB.
pub fn add_rgb_contribution(
    base: &LinearImage,
    contribution: &LinearImage,
    gains: [f32; 3],
) -> Result<LinearImage> {
    if base.channels != 3
        || contribution.channels != 3
        || (base.width, base.height) != (contribution.width, contribution.height)
        || base.data.len() != contribution.data.len()
        || base
            .width
            .checked_mul(base.height)
            .and_then(|n| n.checked_mul(3))
            != Some(base.data.len())
        || gains.iter().any(|g| !g.is_finite() || *g <= 0.0)
    {
        return Err(error(
            "RGB contribution needs matching RGB grids and finite positive gains",
        ));
    }
    let mut data = Vec::with_capacity(base.data.len());
    for (b, c) in base
        .data
        .as_chunks::<3>()
        .0
        .iter()
        .zip(contribution.data.as_chunks::<3>().0.iter())
    {
        if b.iter().chain(c).any(|v| !v.is_finite()) {
            data.extend([f32::NAN; 3]);
            continue;
        }
        for i in 0..3 {
            let v = f64::from(b[i]) + f64::from(c[i]) * f64::from(gains[i]);
            if !v.is_finite() || v.abs() > f64::from(f32::MAX) {
                return Err(error("RGB contribution overflows the output sample range"));
            }
            data.push(v as f32);
        }
    }
    LinearImage::new(base.width, base.height, 3, data)
}

/// Reduce only green above both red and blue on bounded display RGB.
/// Amount is in [0,1]. This aesthetic operation preserves red, blue, neutral
/// pixels and common coverage. Apply before adding independently prepared stars.
pub fn reduce_green_excess(image: &mut LinearImage, amount: f64) -> Result<()> {
    if image.width == 0
        || image.height == 0
        || image
            .width
            .checked_mul(image.height)
            .and_then(|n| n.checked_mul(3))
            != Some(image.data.len())
        || image.channels != 3
        || !amount.is_finite()
        || !(0.0..=1.0).contains(&amount)
        || image
            .data
            .iter()
            .any(|v| v.is_finite() && !(0.0..=1.0).contains(v))
    {
        return Err(Error::Color(
            "Green reduction requires display RGB in [0,1] and an amount from zero to one".into(),
        ));
    }
    if amount == 0.0 {
        return Ok(());
    }
    for pixel in image.data.as_chunks_mut::<3>().0 {
        if pixel.iter().all(|v| v.is_finite()) {
            pixel[1] -= amount as f32 * (pixel[1] - pixel[0].max(pixel[2])).max(0.0);
        } else {
            pixel.fill(f32::NAN);
        }
    }
    Ok(())
}
