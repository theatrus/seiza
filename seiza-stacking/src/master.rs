use crate::calibration::normalize_flat_response;
use crate::{
    BayerLayout, CalibrationMasters, Error, FitsFrame, LinearImage, MasterDark, Result,
    paths_refer_to_same_file,
};
use seiza_fits::HeaderValue;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MasterFrameKind {
    Bias,
    Dark,
    Flat,
}

impl MasterFrameKind {
    pub(crate) fn fits_name(self) -> &'static str {
        match self {
            Self::Bias => "BIAS",
            Self::Dark => "DARK",
            Self::Flat => "FLAT",
        }
    }

    /// Stable lowercase name used by user-facing APIs and reports.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bias => "bias",
            Self::Dark => "dark",
            Self::Flat => "flat",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MasterRejectionOptions {
    pub low_sigma: f32,
    pub high_sigma: f32,
}

impl Default for MasterRejectionOptions {
    fn default() -> Self {
        Self {
            low_sigma: 3.0,
            high_sigma: 3.0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MasterBuildOptions {
    pub rejection: MasterRejectionOptions,
    /// Assert an exposure when input headers omit or misreport it.
    pub exposure_seconds: Option<f64>,
    /// Bias used to calibrate dark or flat inputs before integration.
    pub bias: Option<LinearImage>,
    /// Dark or dark-flat used to calibrate flat inputs before integration.
    pub dark: Option<MasterDark>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MasterInputStatistics {
    pub accepted_samples: u64,
    pub rejected_samples: u64,
}

#[derive(Clone, Debug)]
pub struct MasterFrame {
    pub kind: MasterFrameKind,
    pub image: LinearImage,
    pub exposure_seconds: Option<f64>,
    pub bayer: Option<BayerLayout>,
    pub input_frames: usize,
    pub accepted_samples: u64,
    pub rejected_samples: u64,
    pub input_statistics: Vec<MasterInputStatistics>,
    pub bias_subtracted: bool,
    pub dark_subtracted: bool,
    pub normalized: bool,
    pub rejection: MasterRejectionOptions,
    pub reference_headers: Vec<(String, HeaderValue)>,
}

impl MasterFrame {
    pub fn into_dark(self) -> Result<MasterDark> {
        if self.kind != MasterFrameKind::Dark {
            return Err(Error::Calibration(
                "only a dark master can be converted to MasterDark".into(),
            ));
        }
        Ok(MasterDark {
            image: self.image,
            exposure_seconds: self.exposure_seconds,
            bias_subtracted: self.bias_subtracted,
            bayer: self.bayer,
        })
    }
}

/// Build a calibration master with a two-pass, leave-one-out sigma-clipped mean.
///
/// Inputs are reread on the second pass, so memory scales with one image rather
/// than with the number of calibration frames.
pub fn build_master_from_fits(
    paths: &[PathBuf],
    kind: MasterFrameKind,
    options: &MasterBuildOptions,
) -> Result<MasterFrame> {
    validate_options(paths, kind, options)?;
    let calibration = match kind {
        MasterFrameKind::Bias => CalibrationMasters::default(),
        MasterFrameKind::Dark => CalibrationMasters::new(options.bias.clone(), None, None)?,
        MasterFrameKind::Flat => {
            CalibrationMasters::new(options.bias.clone(), options.dark.clone(), None)?
        }
    };

    let mut reference_signature = None;
    let mut reference_headers = Vec::new();
    let mut reference_bayer = None;
    let mut dark_exposure = None;
    let mut mean = Vec::<f32>::new();
    let mut m2 = Vec::<f32>::new();

    for (index, path) in paths.iter().enumerate() {
        let prepared = prepare_input(
            path,
            kind,
            options,
            &calibration,
            reference_signature.as_ref(),
            dark_exposure,
        )?;
        if index == 0 {
            reference_headers = prepared.headers.clone();
            reference_bayer = prepared.bayer;
            reference_signature = Some(FrameSignature::from_frame(&prepared, kind));
            if kind == MasterFrameKind::Dark {
                dark_exposure = prepared.effective_exposure;
            }
            mean.resize(prepared.image.sample_count(), 0.0);
            m2.resize(prepared.image.sample_count(), 0.0);
        }
        let count = (index + 1) as f32;
        for ((mean, m2), value) in mean.iter_mut().zip(&mut m2).zip(prepared.image.data) {
            let delta = value - *mean;
            *mean += delta / count;
            *m2 += delta * (value - *mean);
        }
    }

    let mut integrated = vec![0.0_f32; mean.len()];
    let mut accepted_counts = vec![0_u32; mean.len()];
    let mut input_statistics = Vec::with_capacity(paths.len());
    let mut accepted_samples = 0_u64;
    let mut rejected_samples = 0_u64;
    let count = paths.len();

    for path in paths {
        let prepared = prepare_input(
            path,
            kind,
            options,
            &calibration,
            reference_signature.as_ref(),
            dark_exposure,
        )?;
        let mut frame_accepted = 0_u64;
        let mut frame_rejected = 0_u64;
        for index in 0..integrated.len() {
            let value = prepared.image.data[index];
            if rejects_sample(value, mean[index], m2[index], count, options.rejection) {
                frame_rejected += 1;
                continue;
            }
            let sample_count = accepted_counts[index]
                .checked_add(1)
                .ok_or_else(|| Error::Calibration("too many calibration frames".into()))?;
            accepted_counts[index] = sample_count;
            integrated[index] += (value - integrated[index]) / sample_count as f32;
            frame_accepted += 1;
        }
        accepted_samples = accepted_samples.saturating_add(frame_accepted);
        rejected_samples = rejected_samples.saturating_add(frame_rejected);
        input_statistics.push(MasterInputStatistics {
            accepted_samples: frame_accepted,
            rejected_samples: frame_rejected,
        });
    }

    for (value, count) in integrated.iter_mut().zip(accepted_counts) {
        if count == 0 {
            *value = f32::NAN;
        }
    }
    let signature = reference_signature.expect("at least two paths were validated");
    let image = LinearImage::new(
        signature.width,
        signature.height,
        signature.channels,
        integrated,
    )?;
    let dark_subtracted = kind == MasterFrameKind::Flat && options.dark.is_some();
    let bias_subtracted = match kind {
        MasterFrameKind::Bias => false,
        MasterFrameKind::Dark => options.bias.is_some(),
        MasterFrameKind::Flat => {
            options.bias.is_some()
                || options
                    .dark
                    .as_ref()
                    .is_some_and(|dark| !dark.bias_subtracted)
        }
    };

    Ok(MasterFrame {
        kind,
        image,
        exposure_seconds: (kind == MasterFrameKind::Dark)
            .then_some(dark_exposure)
            .flatten(),
        bayer: reference_bayer,
        input_frames: paths.len(),
        accepted_samples,
        rejected_samples,
        input_statistics,
        bias_subtracted,
        dark_subtracted,
        normalized: kind == MasterFrameKind::Flat,
        rejection: options.rejection,
        reference_headers,
    })
}

fn validate_options(
    paths: &[PathBuf],
    kind: MasterFrameKind,
    options: &MasterBuildOptions,
) -> Result<()> {
    if paths.len() < 2 {
        return Err(Error::Calibration(
            "at least two calibration frames are required".into(),
        ));
    }
    for (index, path) in paths.iter().enumerate() {
        if paths[..index]
            .iter()
            .any(|previous| paths_refer_to_same_file(path, previous))
        {
            return Err(Error::Calibration(format!(
                "duplicate calibration input {}",
                path.display()
            )));
        }
    }
    if !options.rejection.low_sigma.is_finite()
        || options.rejection.low_sigma <= 0.0
        || !options.rejection.high_sigma.is_finite()
        || options.rejection.high_sigma <= 0.0
    {
        return Err(Error::Calibration(
            "master rejection sigmas must be positive finite numbers".into(),
        ));
    }
    if options
        .exposure_seconds
        .is_some_and(|seconds| !seconds.is_finite() || seconds <= 0.0)
    {
        return Err(Error::Calibration(
            "master exposure must be a positive finite number".into(),
        ));
    }
    match kind {
        MasterFrameKind::Bias if options.bias.is_some() || options.dark.is_some() => {
            return Err(Error::Calibration(
                "bias masters cannot use bias or dark calibration inputs".into(),
            ));
        }
        MasterFrameKind::Dark if options.dark.is_some() => {
            return Err(Error::Calibration(
                "dark masters cannot use a dark calibration input".into(),
            ));
        }
        MasterFrameKind::Flat
            if options
                .dark
                .as_ref()
                .is_some_and(|dark| dark.bias_subtracted)
                && options.bias.is_none() =>
        {
            return Err(Error::Calibration(
                "a bias-subtracted dark-flat also requires a master bias".into(),
            ));
        }
        _ => {}
    }
    Ok(())
}

struct PreparedInput {
    image: LinearImage,
    headers: Vec<(String, HeaderValue)>,
    bayer: Option<BayerLayout>,
    effective_exposure: Option<f64>,
}

fn prepare_input(
    path: &Path,
    kind: MasterFrameKind,
    options: &MasterBuildOptions,
    calibration: &CalibrationMasters,
    reference: Option<&FrameSignature>,
    dark_exposure: Option<f64>,
) -> Result<PreparedInput> {
    let mut frame = FitsFrame::open(path)?;
    if let Some(reference) = reference {
        reference.validate(&frame, path)?;
    }
    let effective_exposure = options.exposure_seconds.or(frame.exposure_seconds);
    if kind == MasterFrameKind::Dark
        && let (Some(reference), Some(current)) = (dark_exposure, effective_exposure)
        && !close_exposure(reference, current)
    {
        return Err(Error::Calibration(format!(
            "{} has exposure {current:.6}s but the dark set uses {reference:.6}s",
            path.display()
        )));
    }
    if kind == MasterFrameKind::Dark
        && dark_exposure.is_some() != effective_exposure.is_some()
        && reference.is_some()
    {
        return Err(Error::Calibration(format!(
            "{} is missing exposure metadata present on the other dark frames; use an exposure override",
            path.display()
        )));
    }
    if kind == MasterFrameKind::Flat
        && options.bias.is_none()
        && let Some(dark) = &options.dark
        && !dark.bias_subtracted
        && let (Some(flat_exposure), Some(dark_exposure)) =
            (effective_exposure, dark.exposure_seconds)
        && !close_exposure(flat_exposure, dark_exposure)
    {
        return Err(Error::Calibration(format!(
            "{} uses a {flat_exposure:.6}s flat with a {dark_exposure:.6}s dark-flat that still contains bias; matching exposures or a master bias are required",
            path.display()
        )));
    }
    calibration.apply(&mut frame.image, effective_exposure, frame.bayer)?;
    if kind == MasterFrameKind::Flat {
        normalize_flat_response(&mut frame.image)?;
    }
    if frame.image.data.iter().any(|value| !value.is_finite()) {
        return Err(Error::Calibration(format!(
            "{} contains non-finite samples after calibration",
            path.display()
        )));
    }
    Ok(PreparedInput {
        image: frame.image,
        headers: frame.headers,
        bayer: frame.bayer,
        effective_exposure,
    })
}

fn close_exposure(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1.0e-3_f64.max(left.abs().max(right.abs()) * 1.0e-3)
}

#[derive(Clone, Debug)]
struct FrameSignature {
    width: usize,
    height: usize,
    channels: usize,
    bayer: Option<BayerLayout>,
    metadata: Vec<(&'static str, ComparableHeader)>,
}

impl FrameSignature {
    fn from_frame(frame: &PreparedInput, kind: MasterFrameKind) -> Self {
        let mut keys = vec![
            "INSTRUME", "CAMERA", "XBINNING", "YBINNING", "CCDXBIN", "CCDYBIN", "XPIXSZ", "YPIXSZ",
            "GAIN", "EGAIN", "OFFSET", "CCD-TEMP", "SET-TEMP", "READOUTM",
        ];
        if kind == MasterFrameKind::Flat {
            keys.push("FILTER");
        }
        let metadata = keys
            .into_iter()
            .filter_map(|key| header(&frame.headers, key).map(|value| (key, value.into())))
            .collect();
        Self {
            width: frame.image.width,
            height: frame.image.height,
            channels: frame.image.channels,
            bayer: frame.bayer,
            metadata,
        }
    }

    fn validate(&self, frame: &FitsFrame, path: &Path) -> Result<()> {
        if self.width != frame.image.width
            || self.height != frame.image.height
            || self.channels != frame.image.channels
        {
            return Err(Error::Calibration(format!(
                "{} is {}x{}x{} but the calibration set is {}x{}x{}",
                path.display(),
                frame.image.width,
                frame.image.height,
                frame.image.channels,
                self.width,
                self.height,
                self.channels
            )));
        }
        if self.bayer != frame.bayer {
            return Err(Error::Calibration(format!(
                "{} has a different Bayer layout from the calibration set",
                path.display()
            )));
        }
        for (key, expected) in &self.metadata {
            let Some(actual) = header(&frame.headers, key) else {
                continue;
            };
            let actual = ComparableHeader::from(actual);
            if !expected.compatible_with(&actual, key) {
                return Err(Error::Calibration(format!(
                    "{} has incompatible {key} metadata ({actual} instead of {expected})",
                    path.display()
                )));
            }
        }
        Ok(())
    }
}

fn header<'a>(headers: &'a [(String, HeaderValue)], key: &str) -> Option<&'a HeaderValue> {
    headers
        .iter()
        .find(|(candidate, _)| candidate == key)
        .map(|(_, value)| value)
}

#[derive(Clone, Debug)]
enum ComparableHeader {
    Number(f64),
    Text(String),
    Logical(bool),
}

impl ComparableHeader {
    fn compatible_with(&self, other: &Self, key: &str) -> bool {
        match (self, other) {
            (Self::Number(left), Self::Number(right)) => {
                let tolerance = if matches!(key, "CCD-TEMP" | "SET-TEMP") {
                    1.0
                } else {
                    1.0e-6_f64.max(left.abs().max(right.abs()) * 1.0e-6)
                };
                (left - right).abs() <= tolerance
            }
            (Self::Text(left), Self::Text(right)) => left.eq_ignore_ascii_case(right),
            (Self::Logical(left), Self::Logical(right)) => left == right,
            _ => false,
        }
    }
}

impl From<&HeaderValue> for ComparableHeader {
    fn from(value: &HeaderValue) -> Self {
        if let Some(number) = value.as_f64() {
            Self::Number(number)
        } else if let Some(value) = value.as_bool() {
            Self::Logical(value)
        } else if let Some(value) = value.as_str() {
            Self::Text(value.trim().to_string())
        } else {
            Self::Text(format!("{value:?}"))
        }
    }
}

impl std::fmt::Display for ComparableHeader {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Number(value) => write!(formatter, "{value}"),
            Self::Text(value) => formatter.write_str(value),
            Self::Logical(value) => write!(formatter, "{value}"),
        }
    }
}

fn rejects_sample(
    value: f32,
    mean: f32,
    m2: f32,
    count: usize,
    options: MasterRejectionOptions,
) -> bool {
    if count < 3 {
        return false;
    }
    let count = count as f64;
    let value = f64::from(value);
    let mean = f64::from(mean);
    let other_count = count - 1.0;
    let other_mean = (count * mean - value) / other_count;
    let other_m2 = (f64::from(m2) - (value - mean) * (value - other_mean)).max(0.0);
    let sigma = (other_m2 / (other_count - 1.0)).sqrt();
    let residual = value - other_mean;
    if sigma <= f64::EPSILON {
        let tolerance = f64::from(f32::EPSILON) * other_mean.abs().max(1.0) * 8.0;
        return residual.abs() > tolerance;
    }
    residual < -f64::from(options.low_sigma) * sigma
        || residual > f64::from(options.high_sigma) * sigma
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{StackSnapshot, write_fits_f32};

    fn write_image(path: &std::path::Path, values: &[f32]) {
        let image = LinearImage::new(2, 2, 1, values.to_vec()).unwrap();
        let snapshot = StackSnapshot {
            variance: LinearImage::new(2, 2, 1, vec![0.0; 4]).unwrap(),
            coverage: vec![1; 4],
            rejected_samples: vec![0; 4],
            image,
            accepted_frames: 1,
            rejected_frames: 0,
        };
        write_fits_f32(path, &snapshot, &[]).unwrap();
    }

    #[test]
    fn leave_one_out_clipping_rejects_a_single_outlier_in_small_sets() {
        let values = [10.0_f32, 10.0, 1000.0];
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let m2 = values
            .iter()
            .map(|value| (*value - mean).powi(2))
            .sum::<f32>();
        assert!(!rejects_sample(
            values[0],
            mean,
            m2,
            values.len(),
            MasterRejectionOptions::default()
        ));
        assert!(rejects_sample(
            values[2],
            mean,
            m2,
            values.len(),
            MasterRejectionOptions::default()
        ));
    }

    #[test]
    fn two_frame_sets_are_averaged_without_clipping() {
        assert!(!rejects_sample(
            100.0,
            50.5,
            4900.5,
            2,
            MasterRejectionOptions::default()
        ));
    }

    #[test]
    fn rejects_duplicate_source_paths() {
        let path = PathBuf::from("same.fits");
        let error = build_master_from_fits(
            &[path.clone(), path],
            MasterFrameKind::Bias,
            &MasterBuildOptions::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("duplicate calibration input"));
    }

    #[test]
    fn builds_a_sigma_clipped_master_without_retaining_inputs() {
        let directory = tempfile::tempdir().unwrap();
        let paths = (0..3)
            .map(|index| directory.path().join(format!("bias-{index}.fits")))
            .collect::<Vec<_>>();
        write_image(&paths[0], &[10.0, 20.0, 30.0, 40.0]);
        write_image(&paths[1], &[10.0, 20.0, 30.0, 40.0]);
        write_image(&paths[2], &[1000.0, 20.0, 30.0, 40.0]);

        let master = build_master_from_fits(
            &paths,
            MasterFrameKind::Bias,
            &MasterBuildOptions::default(),
        )
        .unwrap();
        assert_eq!(master.image.data, [10.0, 20.0, 30.0, 40.0]);
        assert_eq!(master.rejected_samples, 1);
        assert_eq!(master.input_statistics[2].rejected_samples, 1);
    }

    #[test]
    fn calibrates_and_normalizes_each_flat_before_integration() {
        let directory = tempfile::tempdir().unwrap();
        let paths = (0..2)
            .map(|index| directory.path().join(format!("flat-{index}.fits")))
            .collect::<Vec<_>>();
        write_image(&paths[0], &[110.0, 210.0, 110.0, 210.0]);
        write_image(&paths[1], &[210.0, 410.0, 210.0, 410.0]);
        let options = MasterBuildOptions {
            bias: Some(LinearImage::new(2, 2, 1, vec![10.0; 4]).unwrap()),
            ..MasterBuildOptions::default()
        };

        let master = build_master_from_fits(&paths, MasterFrameKind::Flat, &options).unwrap();
        assert!(master.bias_subtracted);
        assert!(master.normalized);
        for (actual, expected) in
            master
                .image
                .data
                .iter()
                .zip([2.0 / 3.0, 4.0 / 3.0, 2.0 / 3.0, 4.0 / 3.0])
        {
            assert!((actual - expected).abs() < 1.0e-5);
        }
    }
}
