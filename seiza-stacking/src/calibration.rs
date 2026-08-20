use crate::{BayerLayout, Error, FitsFrame, FrameSourceRole, LinearImage, Result};
use rayon::prelude::*;
use seiza_calibration::{
    FrameSignature, MatchTolerances, exposure_matches, optics_match, sensor_consistent,
    sensor_matches, temperature_matches,
};
use std::path::Path;

/// An integrated master dark, with the metadata needed to scale and apply it.
#[derive(Clone, Debug)]
pub struct MasterDark {
    /// Integrated dark image.
    pub image: LinearImage,
    /// Exposure of the dark, used when scaling it to a light frame.
    pub exposure_seconds: Option<f64>,
    /// Whether the bias pedestal has already been removed from this master.
    pub bias_subtracted: bool,
    /// Raw CFA sampling retained from the master FITS, when present.
    pub bayer: Option<BayerLayout>,
}

impl MasterDark {
    /// Decode a master dark, including Seiza's calibration-state headers when present.
    pub fn from_fits_frame(frame: FitsFrame, exposure_seconds: Option<f64>) -> Result<Self> {
        frame.validate_master_kind("DARK")?;
        Ok(Self {
            exposure_seconds: exposure_seconds.or(frame.exposure_seconds),
            bias_subtracted: header_bool(&frame, "BIASSUB").unwrap_or(false),
            bayer: frame.bayer,
            image: frame.image,
        })
    }
}

/// An integrated master flat used to correct the optical response.
#[derive(Clone, Debug)]
pub struct MasterFlat {
    /// Integrated flat image.
    pub image: LinearImage,
    /// True when the flat has already been calibrated and/or normalized.
    pub calibrated: bool,
    /// Raw CFA sampling retained from the master FITS, when present.
    pub bayer: Option<BayerLayout>,
}

impl MasterFlat {
    /// Wrap an uncalibrated flat with no CFA metadata.
    pub fn raw(image: LinearImage) -> Self {
        Self {
            image,
            calibrated: false,
            bayer: None,
        }
    }

    /// Construct a raw flat while retaining its CFA sampling.
    pub fn raw_with_bayer(image: LinearImage, bayer: BayerLayout) -> Self {
        Self {
            image,
            calibrated: false,
            bayer: Some(bayer),
        }
    }

    /// Decode a master flat, including Seiza's calibration-state headers when present.
    pub fn from_fits_frame(frame: FitsFrame) -> Result<Self> {
        frame.validate_master_kind("FLAT")?;
        let calibrated = ["BIASSUB", "DARKSUB", "FLATNORM"]
            .into_iter()
            .any(|key| header_bool(&frame, key).unwrap_or(false));
        Ok(Self {
            image: frame.image,
            calibrated,
            bayer: frame.bayer,
        })
    }
}

/// Precomputed master calibration data in the raw light frame's sampling.
#[derive(Clone, Debug, Default)]
pub struct CalibrationMasters {
    pub(crate) bias: Option<LinearImage>,
    pub(crate) bias_signature: Option<FrameSignature>,
    pub(crate) dark_signal: Option<LinearImage>,
    pub(crate) dark_exposure_seconds: Option<f64>,
    pub(crate) dark_scaling_safe: bool,
    pub(crate) dark_signature: Option<FrameSignature>,
    pub(crate) dark_bayer: Option<BayerLayout>,
    pub(crate) flat_response: Option<LinearImage>,
    pub(crate) flat_signature: Option<FrameSignature>,
    pub(crate) flat_bayer: Option<BayerLayout>,
}

impl CalibrationMasters {
    /// Load optional integrated calibration masters from FITS paths.
    ///
    /// A supplied dark exposure overrides its FITS metadata. Paths are read
    /// completely during this call and are not retained by the result.
    pub fn from_fits_paths(
        bias: Option<&Path>,
        dark: Option<&Path>,
        flat: Option<&Path>,
        dark_exposure_seconds: Option<f64>,
    ) -> Result<Self> {
        let bias = bias
            .map(FitsFrame::open)
            .transpose()?
            .map(|frame| {
                frame.validate_master_kind("BIAS")?;
                let signature = frame.metadata().signature;
                Ok::<_, Error>((frame.image, signature))
            })
            .transpose()?;
        let dark = dark
            .map(FitsFrame::open)
            .transpose()?
            .map(|frame| {
                let mut signature = frame.metadata().signature;
                signature.exposure_seconds = dark_exposure_seconds.or(frame.exposure_seconds);
                Ok::<_, Error>((
                    MasterDark::from_fits_frame(frame, dark_exposure_seconds)?,
                    signature,
                ))
            })
            .transpose()?;
        let flat = flat
            .map(FitsFrame::open)
            .transpose()?
            .map(|frame| {
                let signature = frame.metadata().signature;
                Ok::<_, Error>((MasterFlat::from_fits_frame(frame)?, signature))
            })
            .transpose()?;
        let (bias, bias_signature) = bias.unzip();
        let (dark, dark_signature) = dark.unzip();
        let (flat, flat_signature) = flat.unzip();
        let mut masters = Self::new(bias, dark, flat)?;
        masters.bias_signature = bias_signature;
        masters.dark_signature = dark_signature;
        masters.flat_signature = flat_signature;
        masters.validate_master_set_signatures()?;
        Ok(masters)
    }

    /// Prepare masters for use: validate metadata, check matching dimensions,
    /// isolate dark current, and normalize the flat response.
    pub fn new(
        bias: Option<LinearImage>,
        dark: Option<MasterDark>,
        flat: Option<MasterFlat>,
    ) -> Result<Self> {
        if dark.as_ref().is_some_and(|dark| {
            dark.exposure_seconds
                .is_some_and(|seconds| !seconds.is_finite() || seconds <= 0.0)
        }) {
            return Err(Error::Calibration(
                "master-dark exposure must be a positive finite number".into(),
            ));
        }
        // A bias-subtracted dark removes only dark current, so without a
        // master bias the light frame would keep its bias pedestal and
        // distort flat division.
        if bias.is_none() && dark.as_ref().is_some_and(|dark| dark.bias_subtracted) {
            return Err(Error::Calibration(
                "a bias-subtracted master dark requires a master bias to remove the light frame's bias pedestal".into(),
            ));
        }
        let reference = bias
            .as_ref()
            .or_else(|| dark.as_ref().map(|value| &value.image))
            .or_else(|| flat.as_ref().map(|value| &value.image));
        if let Some(reference) = reference {
            for image in bias
                .iter()
                .chain(dark.iter().map(|value| &value.image))
                .chain(flat.iter().map(|value| &value.image))
            {
                if !reference.dimensions_match(image) {
                    return Err(Error::Calibration(
                        "bias, dark, and flat masters must have matching dimensions and channels"
                            .into(),
                    ));
                }
            }
        }

        // An ordinary master dark includes a bias pedestal. Exposure scaling
        // is valid only when a supplied master bias lets us isolate the dark
        // current signal first.
        let bias_signature = bias
            .as_ref()
            .map(|image| image_signature(image, None, None));
        let dark_signature = dark
            .as_ref()
            .map(|dark| image_signature(&dark.image, dark.exposure_seconds, dark.bayer));
        let flat_signature = flat
            .as_ref()
            .map(|flat| image_signature(&flat.image, None, flat.bayer));
        let dark_scaling_safe = dark.as_ref().is_some_and(|dark| {
            (dark.bias_subtracted || bias.is_some()) && dark.exposure_seconds.is_some()
        });
        let dark_exposure_seconds = dark_scaling_safe
            .then(|| dark.as_ref().and_then(|dark| dark.exposure_seconds))
            .flatten();
        let dark_bayer = dark.as_ref().and_then(|dark| dark.bayer);
        let dark_signal = dark.map(|mut dark| {
            if !dark.bias_subtracted
                && let Some(bias) = &bias
            {
                dark.image
                    .data
                    .par_iter_mut()
                    .zip(bias.data.par_iter())
                    .for_each(|(value, bias_value)| {
                        *value -= *bias_value;
                    });
            }
            dark.image
        });

        let flat_bayer = flat.as_ref().and_then(|flat| flat.bayer);
        if let (Some(dark_bayer), Some(flat_bayer)) = (dark_bayer, flat_bayer)
            && dark_bayer != flat_bayer
        {
            return Err(Error::Calibration(
                "master dark and flat have different Bayer layouts".into(),
            ));
        }
        let flat_response = flat
            .map(|mut flat| {
                if !flat.calibrated
                    && let Some(bias) = &bias
                {
                    flat.image
                        .data
                        .par_iter_mut()
                        .zip(bias.data.par_iter())
                        .for_each(|(value, bias_value)| *value -= *bias_value);
                }
                normalize_flat_response(&mut flat.image)?;
                Ok(flat.image)
            })
            .transpose()?;

        Ok(Self {
            bias,
            bias_signature,
            dark_signal,
            dark_exposure_seconds,
            dark_scaling_safe,
            dark_signature,
            dark_bayer,
            flat_response,
            flat_signature,
            flat_bayer,
        })
    }

    /// Whether no master is present, so calibration would be a no-op.
    pub fn is_empty(&self) -> bool {
        self.bias.is_none() && self.dark_signal.is_none() && self.flat_response.is_none()
    }

    /// Ensure independently selected masters do not contradict one another.
    ///
    /// Matching against a light is intentionally asymmetric: a light with an
    /// unknown gain cannot rule out a calibration frame whose gain is known.
    /// That does not make two masters with different known gains safe to mix,
    /// though. Requiring sensor compatibility in both directions catches both
    /// explicit disagreement and a known-vs-unknown setting before pixels are
    /// combined. Missing signatures are retained only for legacy v1 contexts;
    /// their per-light admission path fails closed until masters are reloaded.
    pub(crate) fn validate_master_set_signatures(&self) -> Result<()> {
        let signatures = [
            ("bias", self.bias.is_some(), self.bias_signature.as_ref()),
            (
                "dark",
                self.dark_signal.is_some(),
                self.dark_signature.as_ref(),
            ),
            (
                "flat",
                self.flat_response.is_some(),
                self.flat_signature.as_ref(),
            ),
        ];
        for left in 0..signatures.len() {
            let (left_kind, left_active, Some(left_signature)) = signatures[left] else {
                continue;
            };
            if !left_active {
                continue;
            }
            for &(right_kind, right_active, right_signature) in &signatures[left + 1..] {
                let Some(right_signature) = right_signature else {
                    continue;
                };
                if right_active && !sensor_consistent(left_signature, right_signature) {
                    return Err(Error::Calibration(format!(
                        "master {left_kind} and master {right_kind} have incompatible sensor or readout metadata"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Validate one decoded light before any master mutates its pixels.
    ///
    /// With no active masters this is intentionally a no-op, so callers may
    /// stack already-processed lights. With calibration enabled, masters and
    /// already-calibrated inputs are refused and every master must match the
    /// light under the shared sensor/dark/optics policy.
    pub fn validate_light_frame(&self, frame: &FitsFrame) -> Result<()> {
        if self.is_empty() {
            return Ok(());
        }
        let metadata = frame.metadata();
        if metadata.is_master {
            return Err(Error::Calibration(
                "a calibration master cannot be used as a light while masters are active".into(),
            ));
        }
        if !matches!(
            metadata.role,
            FrameSourceRole::Light | FrameSourceRole::Unknown
        ) {
            return Err(Error::Calibration(format!(
                "a {} frame cannot be used as a light while masters are active",
                metadata.role.as_str()
            )));
        }
        if metadata.calibration_state.is_calibrated() {
            let state = metadata.calibration_state;
            let mut applied = Vec::new();
            if state.bias_subtracted {
                applied.push("bias subtraction");
            }
            if state.dark_subtracted {
                applied.push("dark subtraction");
            }
            if state.flat_normalized {
                applied.push("flat normalization");
            }
            return Err(Error::Calibration(format!(
                "light frame already declares {}; applying masters again would double-calibrate it",
                applied.join(", ")
            )));
        }
        self.validate_light_signature(&metadata.signature)
    }

    /// Validate acquisition metadata against every active master.
    pub fn validate_light_signature(&self, light: &FrameSignature) -> Result<()> {
        let tolerances = MatchTolerances::default();
        for (kind, active, signature) in [
            ("bias", self.bias.is_some(), self.bias_signature.as_ref()),
            (
                "dark",
                self.dark_signal.is_some(),
                self.dark_signature.as_ref(),
            ),
            (
                "flat",
                self.flat_response.is_some(),
                self.flat_signature.as_ref(),
            ),
        ] {
            if !active {
                continue;
            }
            let Some(signature) = signature else {
                return Err(Error::Calibration(format!(
                    "master {kind} has no compatibility metadata; reload calibration masters before pushing more lights"
                )));
            };
            if !sensor_matches(light, signature) {
                return Err(Error::Calibration(format!(
                    "master {kind} does not match the light frame's sensor or readout mode"
                )));
            }
        }
        if let Some(dark) = &self.dark_signature {
            if !temperature_matches(light, dark, &tolerances) {
                return Err(Error::Calibration(
                    "master dark temperature does not match the light frame".into(),
                ));
            }
            if !self.dark_scaling_safe {
                let known_positive = |value: Option<f64>| {
                    value.is_some_and(|value| value.is_finite() && value > 0.0)
                };
                if !known_positive(light.exposure_seconds) || !known_positive(dark.exposure_seconds)
                {
                    return Err(Error::Calibration(
                        "unscaled master dark requires known positive light and master exposures"
                            .into(),
                    ));
                }
                if !exposure_matches(light, dark, &tolerances) {
                    return Err(Error::Calibration(
                        "unscaled master dark exposure does not match the light frame".into(),
                    ));
                }
            }
        }
        if let Some(flat) = &self.flat_signature
            && !optics_match(light, flat, &tolerances)
        {
            return Err(Error::Calibration(
                "master flat does not match the light frame's optical configuration".into(),
            ));
        }
        Ok(())
    }

    /// Calibrate one linear frame in its current sampling.
    ///
    /// `bayer` describes the image's raw CFA layout. A known master layout must
    /// match it before any pixels are mutated.
    pub fn apply(
        &self,
        image: &mut LinearImage,
        exposure_seconds: Option<f64>,
        bayer: Option<BayerLayout>,
    ) -> Result<()> {
        if exposure_seconds.is_some_and(|seconds| !seconds.is_finite() || seconds <= 0.0) {
            return Err(Error::Calibration(
                "light exposure must be a positive finite number".into(),
            ));
        }
        for master in self
            .bias
            .iter()
            .chain(self.dark_signal.iter())
            .chain(self.flat_response.iter())
        {
            if !master.dimensions_match(image) {
                return Err(Error::Calibration(format!(
                    "light frame is {}x{}x{} but calibration master is {}x{}x{}",
                    image.width,
                    image.height,
                    image.channels,
                    master.width,
                    master.height,
                    master.channels
                )));
            }
        }
        for (kind, master, master_bayer) in [
            ("dark", self.dark_signal.as_ref(), self.dark_bayer),
            ("flat", self.flat_response.as_ref(), self.flat_bayer),
        ] {
            if master.is_some()
                && let Some(master_bayer) = master_bayer
                && Some(master_bayer) != bayer
            {
                return Err(Error::Calibration(format!(
                    "master {kind} Bayer layout does not match the light frame"
                )));
            }
        }

        if let Some(bias) = &self.bias {
            image
                .data
                .par_iter_mut()
                .zip(bias.data.par_iter())
                .for_each(|(value, bias_value)| *value -= *bias_value);
        }
        if let Some(dark) = &self.dark_signal {
            let scale = match (exposure_seconds, self.dark_exposure_seconds) {
                (Some(light), Some(master)) if master > 0.0 => (light / master) as f32,
                (None, Some(_)) => {
                    return Err(Error::Calibration(
                        "light exposure is required when scaling a master dark".into(),
                    ));
                }
                _ => 1.0,
            };
            image
                .data
                .par_iter_mut()
                .zip(dark.data.par_iter())
                .for_each(|(value, dark_value)| *value -= scale * *dark_value);
        }
        if let Some(flat) = &self.flat_response {
            image
                .data
                .par_iter_mut()
                .zip(flat.data.par_iter())
                .for_each(|(value, response)| {
                    *value = if response.is_finite() && *response > 1.0e-6 {
                        *value / *response
                    } else {
                        f32::NAN
                    };
                });
        }
        Ok(())
    }
}

fn image_signature(
    image: &LinearImage,
    exposure_seconds: Option<f64>,
    bayer: Option<BayerLayout>,
) -> FrameSignature {
    let mut signature = FrameSignature::default();
    signature.width = i64::try_from(image.width).ok();
    signature.height = i64::try_from(image.height).ok();
    signature.channels = i64::try_from(image.channels).ok();
    signature.exposure_seconds = exposure_seconds;
    signature.bayer_pattern = bayer.map(|layout| layout.pattern.as_str().to_ascii_uppercase());
    signature
}

pub(crate) fn normalize_flat_response(flat: &mut LinearImage) -> Result<()> {
    for channel in 0..flat.channels {
        let normal = robust_positive_median(
            flat.data
                .iter()
                .skip(channel)
                .step_by(flat.channels)
                .copied(),
        )
        .ok_or_else(|| {
            Error::Calibration(format!(
                "flat master channel {channel} has no positive finite response"
            ))
        })?;
        for pixel in flat.data.chunks_exact_mut(flat.channels) {
            pixel[channel] /= normal;
        }
    }
    Ok(())
}

fn header_bool(frame: &FitsFrame, key: &str) -> Option<bool> {
    frame
        .headers
        .iter()
        .find(|(candidate, _)| candidate == key)
        .and_then(|(_, value)| value.as_bool())
}

fn robust_positive_median(data: impl ExactSizeIterator<Item = f32>) -> Option<f32> {
    let stride = (data.len() / 200_000).max(1);
    let mut values = data
        .step_by(stride)
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    values.sort_unstable_by(f32::total_cmp);
    seiza_stats::median_of_sorted(&values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use seiza_fits::{HeaderValue, WriteHeaderCard};

    fn mono(values: &[f32]) -> LinearImage {
        LinearImage::new(2, 2, 1, values.to_vec()).unwrap()
    }

    fn light(exposure_seconds: Option<f64>, headers: Vec<(String, HeaderValue)>) -> FitsFrame {
        FitsFrame {
            image: mono(&[100.0; 4]),
            headers,
            exposure_seconds,
            bayer: None,
            source: None,
            bounds: None,
        }
    }

    #[test]
    fn applies_bias_scaled_dark_and_normalized_flat() {
        let calibration = CalibrationMasters::new(
            Some(mono(&[10.0; 4])),
            Some(MasterDark {
                image: mono(&[14.0; 4]),
                exposure_seconds: Some(20.0),
                bias_subtracted: false,
                bayer: None,
            }),
            Some(MasterFlat::raw(mono(&[12.0, 12.0, 14.0, 14.0]))),
        )
        .unwrap();
        let mut light = mono(&[110.0; 4]);
        calibration.apply(&mut light, Some(10.0), None).unwrap();
        assert!((light.data[0] - 147.0).abs() < 1.0e-4);
        assert!((light.data[2] - 73.5).abs() < 1.0e-4);
    }

    #[test]
    fn does_not_scale_a_dark_bias_pedestal_without_master_bias() {
        let calibration = CalibrationMasters::new(
            None,
            Some(MasterDark {
                image: mono(&[14.0; 4]),
                exposure_seconds: Some(20.0),
                bias_subtracted: false,
                bayer: None,
            }),
            None,
        )
        .unwrap();
        let mut light = mono(&[110.0; 4]);
        calibration.apply(&mut light, Some(10.0), None).unwrap();
        assert_eq!(light.data, [96.0; 4]);
    }

    #[test]
    fn active_masters_refuse_master_or_already_calibrated_lights() {
        let calibration = CalibrationMasters::new(Some(mono(&[10.0; 4])), None, None).unwrap();
        let master = light(
            Some(60.0),
            vec![("SEIZAMST".into(), HeaderValue::String("DARK".into()))],
        );
        let calibrated = light(
            Some(60.0),
            vec![("BIASSUB".into(), HeaderValue::Logical(true))],
        );
        let raw_dark = light(
            Some(60.0),
            vec![("IMAGETYP".into(), HeaderValue::String("DARK".into()))],
        );
        assert!(
            calibration
                .validate_light_frame(&master)
                .unwrap_err()
                .to_string()
                .contains("master cannot be used as a light")
        );
        assert!(
            calibration
                .validate_light_frame(&calibrated)
                .unwrap_err()
                .to_string()
                .contains("double-calibrate")
        );
        assert!(
            calibration
                .validate_light_frame(&raw_dark)
                .unwrap_err()
                .to_string()
                .contains("dark frame cannot be used as a light")
        );
        assert!(
            CalibrationMasters::default()
                .validate_light_frame(&calibrated)
                .is_ok()
        );
    }

    #[test]
    fn per_light_matching_distinguishes_scalable_and_pedestal_darks() {
        let raw_dark = || MasterDark {
            image: mono(&[14.0; 4]),
            exposure_seconds: Some(20.0),
            bias_subtracted: false,
            bayer: None,
        };
        let unscaled = CalibrationMasters::new(None, Some(raw_dark()), None).unwrap();
        assert!(
            unscaled
                .validate_light_frame(&light(Some(20.0), vec![]))
                .is_ok()
        );
        assert!(
            unscaled
                .validate_light_frame(&light(Some(10.0), vec![]))
                .unwrap_err()
                .to_string()
                .contains("unscaled master dark exposure")
        );
        assert!(
            unscaled
                .validate_light_frame(&light(None, vec![]))
                .unwrap_err()
                .to_string()
                .contains("known positive")
        );

        let scalable =
            CalibrationMasters::new(Some(mono(&[10.0; 4])), Some(raw_dark()), None).unwrap();
        assert!(scalable.dark_scaling_safe);
        assert!(
            scalable
                .validate_light_frame(&light(Some(10.0), vec![]))
                .is_ok()
        );
    }

    #[test]
    fn loaded_master_set_rejects_conflicting_sensor_metadata() {
        let directory = tempfile::tempdir().unwrap();
        let bias_path = directory.path().join("master-bias.fits");
        let dark_path = directory.path().join("master-dark.fits");
        let image = mono(&[10.0; 4]);
        crate::write_processed_image_fits_f32(
            &bias_path,
            &image,
            &[],
            &[
                WriteHeaderCard::new("SEIZAMST", HeaderValue::String("BIAS".into())),
                WriteHeaderCard::new("GAIN", HeaderValue::Integer(100)),
            ],
        )
        .unwrap();
        crate::write_processed_image_fits_f32(
            &dark_path,
            &image,
            &[],
            &[
                WriteHeaderCard::new("SEIZAMST", HeaderValue::String("DARK".into())),
                WriteHeaderCard::new("GAIN", HeaderValue::Integer(200)),
                WriteHeaderCard::new("EXPTIME", HeaderValue::Float(60.0)),
            ],
        )
        .unwrap();

        let error =
            CalibrationMasters::from_fits_paths(Some(&bias_path), Some(&dark_path), None, None)
                .unwrap_err()
                .to_string();
        assert!(error.contains("incompatible sensor or readout"), "{error}");
    }

    #[test]
    fn manual_master_loading_rejects_a_known_wrong_frame_role() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("not-a-bias.fits");
        crate::write_processed_image_fits_f32(
            &path,
            &mono(&[10.0; 4]),
            &[],
            &[WriteHeaderCard::new(
                "IMAGETYP",
                HeaderValue::String("LIGHT".into()),
            )],
        )
        .unwrap();

        let error = CalibrationMasters::from_fits_paths(Some(&path), None, None, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("frame metadata declares light"), "{error}");
    }

    #[test]
    fn every_light_must_match_master_sensor_temperature_and_optics() {
        let mut bias = CalibrationMasters::new(Some(mono(&[10.0; 4])), None, None).unwrap();
        bias.bias_signature.as_mut().unwrap().gain = Some(100);
        let wrong_gain = light(Some(60.0), vec![("GAIN".into(), HeaderValue::Integer(200))]);
        assert!(
            bias.validate_light_frame(&wrong_gain)
                .unwrap_err()
                .to_string()
                .contains("sensor or readout mode")
        );

        let mut dark = CalibrationMasters::new(
            Some(mono(&[10.0; 4])),
            Some(MasterDark {
                image: mono(&[14.0; 4]),
                exposure_seconds: Some(60.0),
                bias_subtracted: false,
                bayer: None,
            }),
            None,
        )
        .unwrap();
        dark.dark_signature.as_mut().unwrap().camera_temp_c = Some(-10.0);
        let warm = light(
            Some(60.0),
            vec![("CCD-TEMP".into(), HeaderValue::Float(-2.0))],
        );
        assert!(
            dark.validate_light_frame(&warm)
                .unwrap_err()
                .to_string()
                .contains("temperature")
        );

        let mut flat = CalibrationMasters::new(
            None,
            None,
            Some(MasterFlat::raw(mono(&[1.0, 1.1, 0.9, 1.0]))),
        )
        .unwrap();
        flat.flat_signature.as_mut().unwrap().filter = Some("Ha".into());
        let wrong_filter = light(
            Some(60.0),
            vec![("FILTER".into(), HeaderValue::String("OIII".into()))],
        );
        assert!(
            flat.validate_light_frame(&wrong_filter)
                .unwrap_err()
                .to_string()
                .contains("optical configuration")
        );
    }

    #[test]
    fn rejects_invalid_exposure_metadata() {
        assert!(
            CalibrationMasters::new(
                None,
                Some(MasterDark {
                    image: mono(&[14.0; 4]),
                    exposure_seconds: Some(0.0),
                    bias_subtracted: false,
                    bayer: None,
                }),
                None,
            )
            .is_err()
        );
        let calibration = CalibrationMasters::default();
        assert!(
            calibration
                .apply(&mut mono(&[1.0; 4]), Some(f64::NAN), None)
                .is_err()
        );
    }

    #[test]
    fn requires_light_exposure_when_scaling_a_dark() {
        let calibration = CalibrationMasters::new(
            Some(mono(&[10.0; 4])),
            Some(MasterDark {
                image: mono(&[14.0; 4]),
                exposure_seconds: Some(20.0),
                bias_subtracted: false,
                bayer: None,
            }),
            None,
        )
        .unwrap();
        assert!(
            calibration
                .apply(&mut mono(&[110.0; 4]), None, None)
                .is_err()
        );
    }

    #[test]
    fn normalizes_planar_rgb_flat_channels_independently() {
        let rgb = |values| LinearImage::new(2, 2, 3, values).unwrap();
        let calibration = CalibrationMasters::new(
            None,
            None,
            Some(MasterFlat::raw(rgb(vec![
                100.0, 200.0, 400.0, 100.0, 200.0, 400.0, 100.0, 200.0, 400.0, 200.0, 400.0, 800.0,
            ]))),
        )
        .unwrap();
        let mut light = rgb(vec![1000.0; 12]);
        calibration.apply(&mut light, None, None).unwrap();
        assert_eq!(&light.data[..9], &[1000.0; 9]);
        assert_eq!(&light.data[9..], &[500.0; 3]);
    }

    #[test]
    fn does_not_subtract_bias_twice_from_prepared_masters() {
        let calibration = CalibrationMasters::new(
            Some(mono(&[10.0; 4])),
            Some(MasterDark {
                image: mono(&[4.0; 4]),
                exposure_seconds: Some(20.0),
                bias_subtracted: true,
                bayer: None,
            }),
            Some(MasterFlat {
                image: mono(&[1.0, 1.0, 2.0, 2.0]),
                calibrated: true,
                bayer: None,
            }),
        )
        .unwrap();
        let mut light = mono(&[110.0; 4]);
        calibration.apply(&mut light, Some(10.0), None).unwrap();
        assert_eq!(light.data, [147.0, 147.0, 73.5, 73.5]);
    }

    #[test]
    fn rejects_a_flat_with_a_different_bayer_layout() {
        let rggb = BayerLayout {
            pattern: seiza_fits::BayerPattern::Rggb,
            x_offset: 0,
            y_offset: 0,
        };
        let bggr = BayerLayout {
            pattern: seiza_fits::BayerPattern::Bggr,
            x_offset: 0,
            y_offset: 0,
        };
        let calibration = CalibrationMasters::new(
            None,
            None,
            Some(MasterFlat::raw_with_bayer(mono(&[1.0; 4]), rggb)),
        )
        .unwrap();
        assert!(
            calibration
                .apply(&mut mono(&[100.0; 4]), None, Some(bggr))
                .is_err()
        );
    }
}
