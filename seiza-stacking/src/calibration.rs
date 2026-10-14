use crate::{BayerLayout, Error, FitsFrame, FrameSourceRole, LinearImage, Result};
use rayon::prelude::*;
use seiza_calibration::{
    FrameSignature, MatchTolerances, exposure_matches, exposure_tolerance, optics_consistent,
    sensor_consistent, temperature_matches,
};
use serde::{Deserialize, Serialize};
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
    /// Bytes occupied by the bias, dark signal and flat response sample buffers.
    ///
    /// Excludes metadata and allocator overhead. Cloning these masters creates
    /// independent image buffers, so hosts retaining session masters and an
    /// active stacker should reserve both copies before budgeting frame workers.
    pub fn image_buffer_bytes(&self) -> usize {
        [&self.bias, &self.dark_signal, &self.flat_response]
            .into_iter()
            .flatten()
            .fold(0_usize, |bytes, image| {
                bytes.saturating_add(image.data.len().saturating_mul(std::mem::size_of::<f32>()))
            })
    }

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
        Self::from_fits_frames(
            bias.map(FitsFrame::open).transpose()?,
            dark.map(FitsFrame::open).transpose()?,
            flat.map(FitsFrame::open).transpose()?,
            dark_exposure_seconds,
        )
    }

    /// Prepare integrated calibration masters already decoded from files.
    ///
    /// Unlike [`Self::new`], which sees only pixels, this keeps each master's
    /// header signature — sensor temperature, gain, offset, filter and the
    /// rest — so every light is checked against what the master recorded.
    /// A supplied dark exposure overrides its FITS metadata.
    pub fn from_fits_frames(
        bias: Option<FitsFrame>,
        dark: Option<FitsFrame>,
        flat: Option<FitsFrame>,
        dark_exposure_seconds: Option<f64>,
    ) -> Result<Self> {
        let bias = bias
            .map(|frame| {
                frame.validate_master_kind("BIAS")?;
                let signature = frame.metadata().signature;
                Ok::<_, Error>((frame.image, signature))
            })
            .transpose()?;
        let dark = dark
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
    ///
    /// Pixels carry no acquisition metadata, so each master's signature holds
    /// only its geometry, the dark's exposure, and any CFA layout. A master
    /// read from a file belongs in [`Self::from_fits_frames`] or
    /// [`Self::from_fits_paths`], which keep its temperature, gain, offset,
    /// and optics for checking against every light.
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
    /// True when a bias master is loaded.
    pub fn has_bias(&self) -> bool {
        self.bias.is_some()
    }

    /// True when a dark master is loaded.
    pub fn has_dark(&self) -> bool {
        self.dark_signal.is_some()
    }

    /// True when a flat master is loaded.
    pub fn has_flat(&self) -> bool {
        self.flat_response.is_some()
    }

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

    /// Validate one decoded light before any master mutates its pixels,
    /// under [`MatchTolerances::default`].
    ///
    /// With no active masters this is intentionally a no-op, so callers may
    /// stack already-processed lights. With calibration enabled, masters and
    /// already-calibrated inputs are refused and every master must match the
    /// light under the shared sensor/dark/optics policy.
    pub fn validate_light_frame(&self, frame: &FitsFrame) -> Result<()> {
        self.validate_light_frame_with(frame, &MatchTolerances::default())
    }

    /// [`Self::validate_light_frame`] under a caller's own tolerances: a
    /// rig whose rotator scatters more than two degrees, or a light that
    /// has to be judged differently from the rest, such as one given a flat
    /// from the nearest other rotator angle on purpose. A live stack judges
    /// its lights this way under [`crate::StackOptions::match_tolerances`].
    ///
    /// Refuses tolerances [`MatchTolerances::validate`] refuses, before
    /// looking at the light.
    pub fn validate_light_frame_with(
        &self,
        frame: &FitsFrame,
        tolerances: &MatchTolerances,
    ) -> Result<()> {
        check_tolerances(tolerances)?;
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
        self.validate_light_signature_with(&metadata.signature, tolerances)
    }

    /// The subset of these masters this light can actually accept, with a
    /// reason for everything set aside, under [`MatchTolerances::default`].
    ///
    /// [`Self::validate_light_signature`] answers yes or no, which is the
    /// right shape for a caller that must not proceed. A caller in an
    /// automatic mode has a different contract: a mismatched master is
    /// something to warn about and work around, never a reason to abandon a
    /// stack half-integrated. This asks the same questions with the same
    /// tolerances, drops what fails, and names each drop — field and both
    /// readings — so the warning says what a reader would otherwise have to
    /// go and measure.
    ///
    /// Each master is judged only against the light, exactly as validation
    /// does, and by the same checks: validation passes exactly when nothing
    /// is dropped here, and the kept set always validates against this
    /// light under the same tolerances.
    pub fn compatible_for_light(&self, light: &FrameSignature) -> (Self, Vec<String>) {
        self.compatible_under(light, &MatchTolerances::default())
    }

    /// [`Self::compatible_for_light`] under a caller's own tolerances — the
    /// form the C and Python surfaces bind, since a host that lets its user
    /// widen the rotation gate has to ask this question with that same gate.
    ///
    /// Refuses tolerances [`MatchTolerances::validate`] refuses.
    pub fn compatible_for_light_with(
        &self,
        light: &FrameSignature,
        tolerances: &MatchTolerances,
    ) -> Result<(Self, Vec<String>)> {
        check_tolerances(tolerances)?;
        Ok(self.compatible_under(light, tolerances))
    }

    /// [`Self::compatible_for_light_with`] once the tolerances are known to
    /// be usable.
    fn compatible_under(
        &self,
        light: &FrameSignature,
        tolerances: &MatchTolerances,
    ) -> (Self, Vec<String>) {
        let mut kept = self.clone();
        let mut dropped = Vec::new();
        for refusal in self.refusals(light, tolerances) {
            match refusal.kind {
                MasterKind::Bias => {
                    kept.bias = None;
                    kept.bias_signature = None;
                }
                MasterKind::Dark => {
                    kept.dark_signal = None;
                    kept.dark_signature = None;
                    kept.dark_exposure_seconds = None;
                    kept.dark_scaling_safe = false;
                    kept.dark_bayer = None;
                }
                MasterKind::Flat => {
                    kept.flat_response = None;
                    kept.flat_signature = None;
                    kept.flat_bayer = None;
                }
            }
            dropped.push(format!(
                "{} set aside: {}",
                refusal.kind.as_str(),
                refusal.detail
            ));
        }
        (kept, dropped)
    }

    /// Validate acquisition metadata against every active master, under
    /// [`MatchTolerances::default`].
    ///
    /// The question here is consistency, not candidacy: two RECORDED readings
    /// must agree, and an unrecorded one proves nothing. Candidate-selection
    /// semantics — where a candidate missing a reading is passed over,
    /// because a library usually holds a better-documented alternative — are
    /// `seiza-calibration`'s business at selection time. By the time a master
    /// is loaded here it has already been chosen, and there is no
    /// alternative to prefer; judging its silence as disagreement rejected
    /// every light against masters written before the writer preserved
    /// optics metadata, which is how two filters of a real archive stopped
    /// stacking while the other five worked.
    ///
    /// The dark's sensor temperature follows the same rule: a master dark
    /// that recorded none is kept. The exposure of a dark that still holds
    /// its bias pedestal is the exception: it must be known on both sides,
    /// because such a dark cannot be scaled.
    pub fn validate_light_signature(&self, light: &FrameSignature) -> Result<()> {
        self.validate_light_signature_with(light, &MatchTolerances::default())
    }

    /// [`Self::validate_light_signature`] under a caller's own tolerances,
    /// for a host that runs its own pass; as with
    /// [`Self::validate_light_frame_with`], a live stack's pushes are judged
    /// under the tolerances the masters carry instead.
    ///
    /// Asks exactly what [`Self::compatible_for_light_with`] asks, so it
    /// succeeds exactly when that drops nothing. A refusal names the reading
    /// that failed and, for a measured gap, the tolerance in force.
    /// Tolerances [`MatchTolerances::validate`] refuses are refused first.
    pub fn validate_light_signature_with(
        &self,
        light: &FrameSignature,
        tolerances: &MatchTolerances,
    ) -> Result<()> {
        check_tolerances(tolerances)?;
        let refusals = self.refusals(light, tolerances);
        // A sensor that does not match outranks every other reason, as it
        // did when each kind of check ran over all three masters in turn.
        let Some(refusal) = refusals
            .iter()
            .find(|refusal| matches!(refusal.problem, Problem::NoMetadata | Problem::Sensor))
            .or_else(|| refusals.first())
        else {
            return Ok(());
        };
        let kind = refusal.kind.as_str();
        let detail = &refusal.detail;
        Err(Error::Calibration(match refusal.problem {
            Problem::NoMetadata => format!(
                "master {kind} has no compatibility metadata; reload calibration masters before pushing more lights"
            ),
            Problem::Sensor => format!(
                "master {kind} does not match the light frame's sensor or readout mode: {detail}"
            ),
            Problem::Temperature => {
                format!("master dark temperature does not match the light frame: {detail}")
            }
            Problem::UnknownExposure => {
                "unscaled master dark requires known positive light and master exposures".into()
            }
            Problem::Exposure => {
                format!("unscaled master dark exposure does not match the light frame: {detail}")
            }
            Problem::Optics => format!(
                "master flat does not match the light frame's optical configuration: {detail}"
            ),
        }))
    }

    /// Every active master this light cannot accept, and why: the one
    /// answer both [`Self::validate_light_signature_with`] and
    /// [`Self::compatible_for_light_with`] give, so they never disagree.
    ///
    /// Each master yields at most one refusal, its first failing check.
    fn refusals(&self, light: &FrameSignature, tolerances: &MatchTolerances) -> Vec<Refusal> {
        let mut refusals = Vec::new();
        for (kind, active, signature) in [
            (
                MasterKind::Bias,
                self.bias.is_some(),
                self.bias_signature.as_ref(),
            ),
            (
                MasterKind::Dark,
                self.dark_signal.is_some(),
                self.dark_signature.as_ref(),
            ),
            (
                MasterKind::Flat,
                self.flat_response.is_some(),
                self.flat_signature.as_ref(),
            ),
        ] {
            if !active {
                continue;
            }
            let refusal = |problem, detail| Refusal {
                kind,
                problem,
                detail,
            };
            // Kept only by legacy v1 contexts; fail closed until reloaded.
            let Some(signature) = signature else {
                refusals.push(refusal(
                    Problem::NoMetadata,
                    "no compatibility metadata; reload calibration masters".into(),
                ));
                continue;
            };
            if !sensor_consistent(light, signature) {
                refusals.push(refusal(
                    Problem::Sensor,
                    seiza_calibration::describe_sensor_mismatch(light, signature),
                ));
                continue;
            }
            let problem = match kind {
                MasterKind::Bias => None,
                MasterKind::Dark => self.dark_refusal(light, signature, tolerances),
                MasterKind::Flat => (!optics_consistent(light, signature, tolerances)).then(|| {
                    (
                        Problem::Optics,
                        seiza_calibration::describe_optics_mismatch(light, signature, tolerances),
                    )
                }),
            };
            if let Some((problem, detail)) = problem {
                refusals.push(refusal(problem, detail));
            }
        }
        refusals
    }

    fn dark_refusal(
        &self,
        light: &FrameSignature,
        dark: &FrameSignature,
        tolerances: &MatchTolerances,
    ) -> Option<(Problem, String)> {
        if !dark_temperature_consistent(light, dark, tolerances) {
            let mut detail = seiza_calibration::describe_value(
                "temperature",
                light.camera_temp_c,
                dark.camera_temp_c,
                "C",
            );
            if let (Some(light), Some(dark)) = (light.camera_temp_c, dark.camera_temp_c) {
                detail.push_str(&format!(
                    " ({:.2} C apart, tolerance {:.2})",
                    (light - dark).abs(),
                    tolerances.dark_temperature_c
                ));
            }
            return Some((Problem::Temperature, detail));
        }
        if self.dark_scaling_safe {
            return None;
        }
        let exposure = || {
            seiza_calibration::describe_value(
                "exposure",
                light.exposure_seconds,
                dark.exposure_seconds,
                "s",
            )
        };
        let known_positive =
            |value: Option<f64>| value.is_some_and(|value| value.is_finite() && value > 0.0);
        if !known_positive(light.exposure_seconds) || !known_positive(dark.exposure_seconds) {
            return Some((Problem::UnknownExposure, exposure()));
        }
        if !exposure_matches(light, dark, tolerances) {
            let mut detail = exposure();
            if let (Some(light), Some(dark)) = (light.exposure_seconds, dark.exposure_seconds) {
                detail.push_str(&format!(
                    " ({:.3} s apart, tolerance {:.3})",
                    (light - dark).abs(),
                    exposure_tolerance(Some(light), Some(dark), tolerances)
                ));
            }
            return Some((Problem::Exposure, detail));
        }
        None
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

/// Which master a [`Refusal`] concerns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MasterKind {
    Bias,
    Dark,
    Flat,
}

impl MasterKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Bias => "bias",
            Self::Dark => "dark",
            Self::Flat => "flat",
        }
    }
}

/// Which check a master failed against a light.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Problem {
    NoMetadata,
    Sensor,
    Temperature,
    UnknownExposure,
    Exposure,
    Optics,
}

/// One active master a light cannot accept: which, what failed, and the
/// readings that failed it.
#[derive(Clone, Debug)]
struct Refusal {
    kind: MasterKind,
    problem: Problem,
    detail: String,
}

/// Refuse tolerances no comparison can use, as a calibration error naming
/// the field. Every API that is handed tolerances calls this, so a stack
/// never holds tolerances that would refuse every light or fail to save.
pub(crate) fn check_tolerances(tolerances: &MatchTolerances) -> Result<()> {
    tolerances
        .validate()
        .map_err(|error| Error::Calibration(error.to_string()))
}

/// The tolerances a light is admitted under, as stack options, a saved
/// context and the configuration fingerprint record them.
///
/// Only the fields admission reads. `master_temperature_c` and
/// `flat_session_seconds` decide which frames a master is built from, never
/// whether a light accepts it, so a host that passes its master-building
/// settings through does not change a stack's identity or context format.
///
/// A missing field reads as its default, so a context saved before a gate
/// was added still opens. An unknown field is refused: a reader that
/// ignored a newer gate would admit lights under the wrong one.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct LightGate {
    exposure_seconds: f64,
    exposure_fraction: f64,
    dark_temperature_c: f64,
    rotation_deg: f64,
    focal_length_mm: f64,
}

impl LightGate {
    pub(crate) fn of(tolerances: &MatchTolerances) -> Self {
        // `+ 0.0` turns a negative zero, which validation accepts, into the
        // zero it equals, so the two write and hash alike.
        Self {
            exposure_seconds: tolerances.exposure_seconds + 0.0,
            exposure_fraction: tolerances.exposure_fraction + 0.0,
            dark_temperature_c: tolerances.dark_temperature_c + 0.0,
            rotation_deg: tolerances.rotation_deg + 0.0,
            focal_length_mm: tolerances.focal_length_mm + 0.0,
        }
    }

    /// Whether `tolerances` admit lights as the defaults do, so options, a
    /// context and a fingerprint leave them unwritten.
    pub(crate) fn admits_as_defaults(tolerances: &MatchTolerances) -> bool {
        Self::of(tolerances) == Self::default()
    }

    /// Full tolerances admitting lights under this gate, refused when no
    /// comparison could use them.
    pub(crate) fn tolerances(self) -> Result<MatchTolerances> {
        let tolerances = MatchTolerances {
            exposure_seconds: self.exposure_seconds,
            exposure_fraction: self.exposure_fraction,
            dark_temperature_c: self.dark_temperature_c,
            rotation_deg: self.rotation_deg,
            focal_length_mm: self.focal_length_mm,
            ..MatchTolerances::default()
        };
        check_tolerances(&tolerances)?;
        Ok(tolerances)
    }
}

impl Default for LightGate {
    fn default() -> Self {
        Self::of(&MatchTolerances::default())
    }
}

/// Serde for [`crate::StackOptions::match_tolerances`]: written as its
/// [`LightGate`], and read back with the tolerances that only build masters
/// at their defaults. A gate no comparison can use is refused when read.
pub(crate) mod light_gate {
    use super::{LightGate, MatchTolerances};
    use serde::{Deserialize, Serialize};

    pub(crate) fn serialize<S: serde::Serializer>(
        tolerances: &MatchTolerances,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        LightGate::of(tolerances).serialize(serializer)
    }

    pub(crate) fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<MatchTolerances, D::Error> {
        LightGate::deserialize(deserializer)?
            .tolerances()
            .map_err(serde::de::Error::custom)
    }
}

/// Whether a loaded master dark's sensor temperature agrees with a light's.
///
/// Consistency, not candidacy, for the reason
/// [`CalibrationMasters::validate_light_signature`] gives: two recorded
/// temperatures must agree within the dark tolerance, and a master that never
/// recorded one proves nothing either way. `temperature_matches` alone is the
/// selection rule, which passes over a silent candidate; applied to a master
/// the user already chose, it refused every light that recorded CCD-TEMP.
fn dark_temperature_consistent(
    light: &FrameSignature,
    dark: &FrameSignature,
    tolerances: &MatchTolerances,
) -> bool {
    dark.camera_temp_c.is_none_or(|value| !value.is_finite())
        || temperature_matches(light, dark, tolerances)
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
        let channels = flat.channels;
        flat.data
            .par_chunks_mut(channels * 16 * 1024)
            .for_each(|pixels| {
                for pixel in pixels.chunks_exact_mut(channels) {
                    pixel[channel] /= normal;
                }
            });
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
    // Selection finds the same middle values a sort would, bit for bit.
    seiza_stats::median_in_place(&mut values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_buffer_bytes_counts_all_master_sample_buffers() {
        assert_eq!(CalibrationMasters::default().image_buffer_bytes(), 0);
        let image = || LinearImage::new(4, 3, 3, vec![1.0; 36]).unwrap();
        let masters = CalibrationMasters {
            bias: Some(image()),
            dark_signal: Some(image()),
            flat_response: Some(image()),
            ..Default::default()
        };
        assert_eq!(
            masters.image_buffer_bytes(),
            3 * 36 * std::mem::size_of::<f32>()
        );
        assert_eq!(
            masters.clone().image_buffer_bytes(),
            masters.image_buffer_bytes()
        );
    }
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
    fn a_master_that_never_recorded_its_optics_still_validates() {
        // Taken from a live cache: a master flat written on 2026-08-14, before
        // the writer preserved optics metadata, carries FILTER and the sensor
        // cards but no TELESCOP and no FOCALLEN. The lights record both. Two
        // recorded readings must agree; a reading nobody wrote down is not a
        // disagreement — the old rule read it as one, and every OIII and SII
        // stack died at frame zero while five other filters worked.
        let mut calibration = CalibrationMasters::new(
            None,
            None,
            Some(MasterFlat::raw(mono(&[30.0, 30.0, 30.0, 30.0]))),
        )
        .unwrap();
        let mut master = FrameSignature::default();
        master.camera = Some("ZWO ASI2600MM Pro".into());
        master.filter = Some("OIII".into());
        // telescope, focal_length_mm, rotation_deg: never recorded.
        calibration.flat_signature = Some(master);

        let mut light = FrameSignature::default();
        light.camera = Some("ZWO ASI2600MM Pro".into());
        light.filter = Some("OIII".into());
        light.telescope = Some("C925".into());
        light.focal_length_mm = Some(2350.0);
        light.rotation_deg = Some(103.94);

        calibration
            .validate_light_signature(&light)
            .expect("an unrecorded reading is not a disagreement");
        let (kept, dropped) = calibration.compatible_for_light(&light);
        assert!(kept.has_flat(), "auto mode keeps it too");
        assert!(dropped.is_empty(), "{dropped:?}");

        // Two recorded readings that disagree still refuse: this is a
        // loosening of what silence means, not of what a reading means.
        let mut wrong = calibration.clone();
        wrong.flat_signature.as_mut().unwrap().filter = Some("SII".into());
        assert!(wrong.validate_light_signature(&light).is_err());
    }

    #[test]
    fn auto_mode_sets_a_mismatched_flat_aside_instead_of_refusing_the_light() {
        // The case that motivated this: a stack died at frame 45 of 100
        // because that light's rotator angle sat 2.31 degrees from the flat
        // master's. An automatic mode wants the flat set aside with a warning
        // that names both angles, and the light integrated with what remains.
        let mut calibration = CalibrationMasters::new(
            Some(mono(&[10.0; 4])),
            None,
            Some(MasterFlat::raw(mono(&[30.0, 30.0, 30.0, 30.0]))),
        )
        .unwrap();
        let mut bias_signature = FrameSignature::default();
        bias_signature.camera = Some("ZWO ASI2600MM Pro".into());
        let mut flat_signature = bias_signature.clone();
        flat_signature.filter = Some("OIII".into());
        flat_signature.rotation_deg = Some(104.24);
        calibration.bias_signature = Some(bias_signature.clone());
        calibration.flat_signature = Some(flat_signature);

        let mut light = bias_signature;
        light.filter = Some("OIII".into());
        light.rotation_deg = Some(101.93);

        assert!(calibration.validate_light_signature(&light).is_err());
        let (kept, dropped) = calibration.compatible_for_light(&light);
        assert!(kept.flat_response.is_none(), "the mismatched flat is gone");
        assert!(kept.bias.is_some(), "the agreeing bias survives");
        assert_eq!(dropped.len(), 1);
        assert!(dropped[0].contains("flat set aside"), "{}", dropped[0]);
        assert!(dropped[0].contains("101.93"), "{}", dropped[0]);
        assert!(dropped[0].contains("104.24"), "{}", dropped[0]);
        kept.validate_light_signature(&light)
            .expect("what remains must validate, or the stack dies anyway");

        // A light the masters fully match drops nothing.
        let mut agreeing = light.clone();
        agreeing.rotation_deg = Some(104.3);
        let (kept, dropped) = calibration.compatible_for_light(&agreeing);
        assert!(dropped.is_empty());
        assert!(kept.flat_response.is_some());
    }

    /// A camera's bias, an unscaled dark (no bias to isolate its signal) and
    /// a flat, each recording what tolerances act on.
    fn recorded_masters() -> CalibrationMasters {
        let mut masters = CalibrationMasters::new(
            None,
            Some(MasterDark {
                image: mono(&[5.0; 4]),
                exposure_seconds: Some(300.0),
                bias_subtracted: false,
                bayer: None,
            }),
            Some(MasterFlat::raw(mono(&[30.0; 4]))),
        )
        .unwrap();
        let mut sensor = FrameSignature::default();
        sensor.camera = Some("ZWO ASI2600MM Pro".into());
        sensor.gain = Some(100);
        let mut dark = sensor.clone();
        dark.camera_temp_c = Some(-10.0);
        dark.exposure_seconds = Some(300.0);
        let mut flat = sensor;
        flat.filter = Some("Ha".into());
        flat.focal_length_mm = Some(530.0);
        flat.rotation_deg = Some(3.2);
        masters.dark_signature = Some(dark);
        masters.flat_signature = Some(flat);
        masters
    }

    /// One field a test moves: its name, the change to the light, the change
    /// to the tolerances, and what the refusal must say.
    type ToleranceCase = (
        &'static str,
        fn(&mut FrameSignature),
        fn(&mut MatchTolerances),
        &'static str,
    );

    /// A light every master in [`recorded_masters`] accepts by default.
    fn matching_light() -> FrameSignature {
        let mut light = FrameSignature::default();
        light.camera = Some("ZWO ASI2600MM Pro".into());
        light.gain = Some(100);
        light.camera_temp_c = Some(-10.0);
        light.exposure_seconds = Some(300.0);
        light.filter = Some("Ha".into());
        light.focal_length_mm = Some(530.0);
        light.rotation_deg = Some(3.2);
        light
    }

    #[test]
    fn rotation_is_measured_the_short_way_round() {
        // 359 degrees is 4.2 from a flat at 3.2, not 355.8.
        let masters = recorded_masters();
        let mut light = matching_light();
        light.rotation_deg = Some(359.0);
        let error = masters
            .validate_light_signature(&light)
            .unwrap_err()
            .to_string();
        assert!(error.contains("4.20 deg apart, tolerance 2.00"), "{error}");
        let five = MatchTolerances {
            rotation_deg: 5.0,
            ..MatchTolerances::default()
        };
        masters
            .validate_light_signature_with(&light, &five)
            .unwrap();
    }

    #[test]
    fn caller_tolerances_admit_what_the_defaults_refuse() {
        // The flat-grouping fallback: a light at 113.8 degrees calibrated
        // with before-change flats from 3.2 degrees, on a rig whose measured
        // angle dependence is small, accepted for that light alone.
        let masters = recorded_masters();
        let defaults = MatchTolerances::default();
        let cases: [ToleranceCase; 3] = [
            (
                "rotation",
                |light| light.rotation_deg = Some(113.8),
                |tolerances| tolerances.rotation_deg = 180.0,
                "110.60 deg apart, tolerance 2.00",
            ),
            (
                "focal length",
                |light| light.focal_length_mm = Some(540.0),
                |tolerances| tolerances.focal_length_mm = 15.0,
                "10.00 mm apart, tolerance 1.00",
            ),
            (
                "temperature",
                |light| light.camera_temp_c = Some(-5.0),
                |tolerances| tolerances.dark_temperature_c = 6.0,
                "5.00 C apart, tolerance 3.00",
            ),
        ];
        for (field, change_light, widen, refusal) in cases {
            let mut light = matching_light();
            change_light(&mut light);
            let mut wide = defaults;
            widen(&mut wide);

            let error = masters.validate_light_signature(&light).unwrap_err();
            assert!(error.to_string().contains(refusal), "{field}: {error}");
            let error = masters
                .validate_light_signature_with(&light, &defaults)
                .unwrap_err();
            assert!(error.to_string().contains(refusal), "{field}: {error}");

            masters
                .validate_light_signature_with(&light, &wide)
                .unwrap_or_else(|error| panic!("{field}: {error}"));
            let (kept, dropped) = masters.compatible_for_light_with(&light, &wide).unwrap();
            assert!(dropped.is_empty(), "{field}: {dropped:?}");
            assert!(kept.has_dark() && kept.has_flat(), "{field}");
            kept.validate_light_signature_with(&light, &wide)
                .unwrap_or_else(|error| panic!("{field}: {error}"));
        }
    }

    #[test]
    fn caller_tolerances_reach_a_decoded_light_frame() {
        let masters = recorded_masters();
        let frame = light(
            Some(300.0),
            vec![
                ("IMAGETYP".into(), HeaderValue::String("LIGHT".into())),
                (
                    "INSTRUME".into(),
                    HeaderValue::String("ZWO ASI2600MM Pro".into()),
                ),
                ("GAIN".into(), HeaderValue::Integer(100)),
                ("CCD-TEMP".into(), HeaderValue::Float(-10.0)),
                ("FILTER".into(), HeaderValue::String("Ha".into())),
                ("FOCALLEN".into(), HeaderValue::Float(530.0)),
                ("ROTATANG".into(), HeaderValue::Float(113.8)),
            ],
        );
        let error = masters.validate_light_frame(&frame).unwrap_err();
        assert!(error.to_string().contains("rotation"), "{error}");
        let wide = MatchTolerances {
            rotation_deg: 180.0,
            ..MatchTolerances::default()
        };
        masters.validate_light_frame_with(&frame, &wide).unwrap();
        // A caller's tolerances loosen matching, never the frame checks.
        let mut calibrated = frame.clone();
        calibrated
            .headers
            .push(("FLATNORM".into(), HeaderValue::Logical(true)));
        assert!(
            masters
                .validate_light_frame_with(&calibrated, &wide)
                .is_err()
        );
    }

    #[test]
    fn a_tighter_tolerance_refuses_what_the_defaults_accept() {
        let masters = recorded_masters();
        let defaults = MatchTolerances::default();
        let cases: [ToleranceCase; 4] = [
            (
                "rotation",
                |light| light.rotation_deg = Some(4.7),
                |tolerances| tolerances.rotation_deg = 1.0,
                "1.50 deg apart, tolerance 1.00",
            ),
            (
                "focal length",
                |light| light.focal_length_mm = Some(530.8),
                |tolerances| tolerances.focal_length_mm = 0.5,
                "0.80 mm apart, tolerance 0.50",
            ),
            (
                "temperature",
                |light| light.camera_temp_c = Some(-8.0),
                |tolerances| tolerances.dark_temperature_c = 1.0,
                "2.00 C apart, tolerance 1.00",
            ),
            (
                "exposure",
                |light| light.exposure_seconds = Some(300.2),
                |tolerances| {
                    tolerances.exposure_seconds = 0.01;
                    tolerances.exposure_fraction = 0.0;
                },
                "0.200 s apart, tolerance 0.010",
            ),
        ];
        for (field, change_light, tighten, refusal) in cases {
            let mut light = matching_light();
            change_light(&mut light);
            let mut tight = defaults;
            tighten(&mut tight);

            masters
                .validate_light_signature(&light)
                .unwrap_or_else(|error| panic!("{field}: {error}"));
            let error = masters
                .validate_light_signature_with(&light, &tight)
                .unwrap_err();
            assert!(error.to_string().contains(refusal), "{field}: {error}");
            let (_, dropped) = masters.compatible_for_light_with(&light, &tight).unwrap();
            assert_eq!(dropped.len(), 1, "{field}: {dropped:?}");
            assert!(dropped[0].contains(refusal), "{field}: {dropped:?}");
        }
    }

    #[test]
    fn the_defaults_are_unchanged() {
        let masters = recorded_masters();
        let mut light = matching_light();
        light.rotation_deg = Some(5.21);
        let error = masters.validate_light_signature(&light).unwrap_err();
        assert_eq!(
            error.to_string(),
            masters
                .validate_light_signature_with(&light, &MatchTolerances::default())
                .unwrap_err()
                .to_string()
        );
        assert!(error.to_string().contains("tolerance 2.00"), "{error}");
        let (kept, dropped) = masters.compatible_for_light(&light);
        let (kept_with, dropped_with) = masters
            .compatible_for_light_with(&light, &MatchTolerances::default())
            .unwrap();
        assert_eq!(dropped, dropped_with);
        assert_eq!(kept.has_flat(), kept_with.has_flat());
        light.rotation_deg = Some(5.19);
        masters.validate_light_signature(&light).unwrap();
    }

    #[test]
    fn validation_passes_exactly_when_matching_drops_nothing() {
        // One answer to one question: validation and matching share their
        // checks, so no light can be kept by one and refused by the other —
        // under any tolerances, and whichever master is at fault.
        let lights: Vec<(&str, FrameSignature)> = {
            let base = matching_light();
            let vary = |change: fn(&mut FrameSignature)| {
                let mut light = base.clone();
                change(&mut light);
                light
            };
            vec![
                ("matching", base.clone()),
                ("rotated", vary(|light| light.rotation_deg = Some(113.8))),
                ("wrapped", vary(|light| light.rotation_deg = Some(359.0))),
                ("unrotated", vary(|light| light.rotation_deg = None)),
                (
                    "refocused",
                    vary(|light| light.focal_length_mm = Some(545.0)),
                ),
                ("warm", vary(|light| light.camera_temp_c = Some(-4.0))),
                ("no temperature", vary(|light| light.camera_temp_c = None)),
                ("longer", vary(|light| light.exposure_seconds = Some(301.0))),
                ("no exposure", vary(|light| light.exposure_seconds = None)),
                (
                    "other filter",
                    vary(|light| light.filter = Some("OIII".into())),
                ),
                ("other gain", vary(|light| light.gain = Some(200))),
            ]
        };
        let tolerances = {
            let mut all = Vec::new();
            for rotation_deg in [0.5, 2.0, 180.0] {
                for focal_length_mm in [0.5, 20.0] {
                    for dark_temperature_c in [1.0, 3.0, 10.0] {
                        for exposure_seconds in [0.01, 5.0] {
                            all.push(MatchTolerances {
                                rotation_deg,
                                focal_length_mm,
                                dark_temperature_c,
                                exposure_seconds,
                                ..MatchTolerances::default()
                            });
                        }
                    }
                }
            }
            all
        };
        let master_sets = {
            let recorded = recorded_masters();
            let mut scalable = CalibrationMasters::new(
                Some(mono(&[1.0; 4])),
                Some(MasterDark {
                    image: mono(&[5.0; 4]),
                    exposure_seconds: Some(300.0),
                    bias_subtracted: false,
                    bayer: None,
                }),
                Some(MasterFlat::raw(mono(&[30.0; 4]))),
            )
            .unwrap();
            scalable.bias_signature = recorded.dark_signature.clone();
            scalable.dark_signature = recorded.dark_signature.clone();
            scalable.flat_signature = recorded.flat_signature.clone();
            let mut legacy = recorded.clone();
            legacy.dark_signature = None;
            vec![
                ("recorded", recorded),
                ("scalable", scalable),
                ("legacy", legacy),
            ]
        };
        let mut refused = 0;
        for (set, masters) in &master_sets {
            for (name, light) in &lights {
                for tolerances in &tolerances {
                    let valid = masters.validate_light_signature_with(light, tolerances);
                    let (kept, dropped) = masters
                        .compatible_for_light_with(light, tolerances)
                        .unwrap();
                    assert_eq!(
                        valid.is_ok(),
                        dropped.is_empty(),
                        "{set} / {name} / {tolerances:?}: {valid:?} vs {dropped:?}"
                    );
                    kept.validate_light_signature_with(light, tolerances)
                        .unwrap_or_else(|error| {
                            panic!("{set} / {name} / {tolerances:?}: kept set refused: {error}")
                        });
                    refused += usize::from(valid.is_err());
                }
            }
        }
        assert!(refused > 0, "the grid must exercise refusals");
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
    fn masters_decoded_from_files_keep_their_header_signature() {
        // `seiza stack --dark` built its masters from pixels alone, so a
        // master dark that recorded CCD-TEMP = -10 refused every light at
        // -10C with "master did not record one". Decoded frames must keep
        // what the master's headers say.
        let directory = tempfile::tempdir().unwrap();
        let dark_path = directory.path().join("master-dark.fits");
        crate::write_processed_image_fits_f32(
            &dark_path,
            &mono(&[14.0; 4]),
            &[],
            &[
                WriteHeaderCard::new("SEIZAMST", HeaderValue::String("DARK".into())),
                WriteHeaderCard::new("EXPTIME", HeaderValue::Float(60.0)),
                WriteHeaderCard::new("CCD-TEMP", HeaderValue::Float(-10.0)),
                WriteHeaderCard::new("GAIN", HeaderValue::Integer(100)),
            ],
        )
        .unwrap();
        let calibration = CalibrationMasters::from_fits_frames(
            None,
            Some(FitsFrame::open(&dark_path).unwrap()),
            None,
            None,
        )
        .unwrap();
        let signature = calibration.dark_signature.as_ref().unwrap();
        assert_eq!(signature.camera_temp_c, Some(-10.0));
        assert_eq!(signature.gain, Some(100));
        assert_eq!(signature.exposure_seconds, Some(60.0));

        let cold = |temperature: f64, gain: i64| {
            light(
                Some(60.0),
                vec![
                    ("CCD-TEMP".into(), HeaderValue::Float(temperature)),
                    ("SET-TEMP".into(), HeaderValue::Float(temperature)),
                    ("GAIN".into(), HeaderValue::Integer(gain)),
                ],
            )
        };
        calibration
            .validate_light_frame(&cold(-10.0, 100))
            .expect("a light at the master's temperature is accepted");
        let warm = calibration
            .validate_light_frame(&cold(0.0, 100))
            .unwrap_err()
            .to_string();
        assert!(warm.contains("temperature light=0C"), "{warm}");
        let wrong_gain = calibration
            .validate_light_frame(&cold(-10.0, 200))
            .unwrap_err()
            .to_string();
        assert!(wrong_gain.contains("sensor or readout"), "{wrong_gain}");

        // The path loader is the same rule over the same frames.
        let from_path =
            CalibrationMasters::from_fits_paths(None, Some(&dark_path), None, None).unwrap();
        assert_eq!(from_path.dark_signature, calibration.dark_signature);
    }

    #[test]
    fn a_master_dark_that_recorded_no_temperature_still_calibrates_a_cooled_light() {
        // Consistency, not candidacy: a dark master the user chose that never
        // wrote down its temperature cannot disagree with the light's.
        let calibration = CalibrationMasters::new(
            None,
            Some(MasterDark {
                image: mono(&[14.0; 4]),
                exposure_seconds: Some(60.0),
                bias_subtracted: false,
                bayer: None,
            }),
            None,
        )
        .unwrap();
        assert_eq!(
            calibration.dark_signature.as_ref().unwrap().camera_temp_c,
            None
        );
        let cooled = light(
            Some(60.0),
            vec![("CCD-TEMP".into(), HeaderValue::Float(-10.0))],
        );
        calibration
            .validate_light_frame(&cooled)
            .expect("an unrecorded temperature is not a disagreement");
        let (kept, dropped) = calibration.compatible_for_light(&cooled.metadata().signature);
        assert!(kept.has_dark(), "auto mode keeps it too");
        assert!(dropped.is_empty(), "{dropped:?}");

        // A non-finite reading is no reading.
        let mut nan = calibration.clone();
        nan.dark_signature.as_mut().unwrap().camera_temp_c = Some(f64::NAN);
        nan.validate_light_frame(&cooled).unwrap();

        // Two recorded temperatures that disagree still refuse, in both modes.
        let mut warm = calibration.clone();
        warm.dark_signature.as_mut().unwrap().camera_temp_c = Some(0.0);
        assert!(warm.validate_light_frame(&cooled).is_err());
        let (kept, dropped) = warm.compatible_for_light(&cooled.metadata().signature);
        assert!(!kept.has_dark());
        assert!(dropped[0].contains("dark set aside"), "{dropped:?}");
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

    #[test]
    fn parallel_flat_normalization_matches_the_serial_sort_bit_for_bit() {
        // The loop as it ran before: a full sort for the median, then one
        // thread dividing every pixel.
        fn serial(flat: &mut LinearImage) {
            for channel in 0..flat.channels {
                let stride = (flat.data.len() / flat.channels / 200_000).max(1);
                let mut values = flat
                    .data
                    .iter()
                    .skip(channel)
                    .step_by(flat.channels)
                    .step_by(stride)
                    .copied()
                    .filter(|value| value.is_finite() && *value > 0.0)
                    .collect::<Vec<_>>();
                values.sort_unstable_by(f32::total_cmp);
                let normal = seiza_stats::median_of_sorted(&values).unwrap();
                for pixel in flat.data.chunks_exact_mut(flat.channels) {
                    pixel[channel] /= normal;
                }
            }
        }
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        for (width, channels) in [(301, 1), (300, 3), (1301, 1)] {
            let data = (0..width * 311 * channels)
                .map(|index| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    match index % 97 {
                        0 => f32::NAN,
                        1 => -1.0,
                        _ => 1000.0 + (state >> 40) as f32 / 4096.0,
                    }
                })
                .collect::<Vec<_>>();
            let mut actual = LinearImage::new(width, 311, channels, data).unwrap();
            let mut expected = actual.clone();
            normalize_flat_response(&mut actual).unwrap();
            serial(&mut expected);
            let bits = |image: &LinearImage| {
                image
                    .data
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            };
            assert_eq!(bits(&actual), bits(&expected));
        }
    }
}
