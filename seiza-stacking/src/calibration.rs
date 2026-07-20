use crate::{Error, FitsFrame, LinearImage, Result};

#[derive(Clone, Debug)]
pub struct MasterDark {
    pub image: LinearImage,
    pub exposure_seconds: Option<f64>,
    /// Whether the bias pedestal has already been removed from this master.
    pub bias_subtracted: bool,
}

impl MasterDark {
    /// Decode a master dark, including Seiza's calibration-state headers when present.
    pub fn from_fits_frame(frame: FitsFrame, exposure_seconds: Option<f64>) -> Result<Self> {
        frame.validate_master_kind("DARK")?;
        Ok(Self {
            exposure_seconds: exposure_seconds.or(frame.exposure_seconds),
            bias_subtracted: header_bool(&frame, "BIASSUB").unwrap_or(false),
            image: frame.image,
        })
    }
}

#[derive(Clone, Debug)]
pub struct MasterFlat {
    pub image: LinearImage,
    /// True when the flat has already been calibrated and/or normalized.
    pub calibrated: bool,
}

impl MasterFlat {
    pub fn raw(image: LinearImage) -> Self {
        Self {
            image,
            calibrated: false,
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
        })
    }
}

/// Precomputed master calibration data in the raw light frame's sampling.
#[derive(Clone, Debug, Default)]
pub struct CalibrationMasters {
    bias: Option<LinearImage>,
    dark_signal: Option<LinearImage>,
    dark_exposure_seconds: Option<f64>,
    flat_response: Option<LinearImage>,
}

impl CalibrationMasters {
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
        let dark_exposure_seconds = dark.as_ref().and_then(|dark| {
            (dark.bias_subtracted || bias.is_some())
                .then_some(dark.exposure_seconds)
                .flatten()
        });
        let dark_signal = dark.map(|mut dark| {
            if !dark.bias_subtracted
                && let Some(bias) = &bias
            {
                for (value, bias_value) in dark.image.data.iter_mut().zip(&bias.data) {
                    *value -= *bias_value;
                }
            }
            dark.image
        });

        let flat_response = flat
            .map(|mut flat| {
                if !flat.calibrated
                    && let Some(bias) = &bias
                {
                    for (value, bias_value) in flat.image.data.iter_mut().zip(&bias.data) {
                        *value -= *bias_value;
                    }
                }
                normalize_flat_response(&mut flat.image)?;
                Ok(flat.image)
            })
            .transpose()?;

        Ok(Self {
            bias,
            dark_signal,
            dark_exposure_seconds,
            flat_response,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.bias.is_none() && self.dark_signal.is_none() && self.flat_response.is_none()
    }

    pub fn apply(&self, image: &mut LinearImage, exposure_seconds: Option<f64>) -> Result<()> {
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

        if let Some(bias) = &self.bias {
            for (value, bias_value) in image.data.iter_mut().zip(&bias.data) {
                *value -= *bias_value;
            }
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
            for (value, dark_value) in image.data.iter_mut().zip(&dark.data) {
                *value -= scale * *dark_value;
            }
        }
        if let Some(flat) = &self.flat_response {
            for (value, response) in image.data.iter_mut().zip(&flat.data) {
                *value = if response.is_finite() && *response > 1.0e-6 {
                    *value / *response
                } else {
                    f32::NAN
                };
            }
        }
        Ok(())
    }
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
    if values.is_empty() {
        return None;
    }
    values.sort_unstable_by(f32::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mono(values: &[f32]) -> LinearImage {
        LinearImage::new(2, 2, 1, values.to_vec()).unwrap()
    }

    #[test]
    fn applies_bias_scaled_dark_and_normalized_flat() {
        let calibration = CalibrationMasters::new(
            Some(mono(&[10.0; 4])),
            Some(MasterDark {
                image: mono(&[14.0; 4]),
                exposure_seconds: Some(20.0),
                bias_subtracted: false,
            }),
            Some(MasterFlat::raw(mono(&[12.0, 12.0, 14.0, 14.0]))),
        )
        .unwrap();
        let mut light = mono(&[110.0; 4]);
        calibration.apply(&mut light, Some(10.0)).unwrap();
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
            }),
            None,
        )
        .unwrap();
        let mut light = mono(&[110.0; 4]);
        calibration.apply(&mut light, Some(10.0)).unwrap();
        assert_eq!(light.data, [96.0; 4]);
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
                }),
                None,
            )
            .is_err()
        );
        let calibration = CalibrationMasters::default();
        assert!(
            calibration
                .apply(&mut mono(&[1.0; 4]), Some(f64::NAN))
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
            }),
            None,
        )
        .unwrap();
        assert!(calibration.apply(&mut mono(&[110.0; 4]), None).is_err());
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
        calibration.apply(&mut light, None).unwrap();
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
            }),
            Some(MasterFlat {
                image: mono(&[1.0, 1.0, 2.0, 2.0]),
                calibrated: true,
            }),
        )
        .unwrap();
        let mut light = mono(&[110.0; 4]);
        calibration.apply(&mut light, Some(10.0)).unwrap();
        assert_eq!(light.data, [147.0, 147.0, 73.5, 73.5]);
    }
}
