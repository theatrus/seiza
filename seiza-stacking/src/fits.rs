use crate::{
    BayerLayout, ColorComposition, Error, LinearImage, MasterFrame, Result, StackExportSnapshot,
    StackSnapshot,
};
use seiza_calibration::FrameSignature;
use seiza_fits::{BayerPattern, F32ImageData, FitsImage, HeaderValue, RowOrder, WriteHeaderCard};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const CAMERA_HEADER_KEYS: &[&str] = &["INSTRUME", "CAMERA"];
const TELESCOPE_HEADER_KEYS: &[&str] = &["TELESCOP", "TELESCOPE"];
const BINNING_X_HEADER_KEYS: &[&str] = &["XBINNING", "CCDXBIN"];
const BINNING_Y_HEADER_KEYS: &[&str] = &["YBINNING", "CCDYBIN"];
const READOUT_HEADER_KEYS: &[&str] = &["READOUTM", "READMODE", "READOUT"];
const EXPOSURE_HEADER_KEYS: &[&str] = &["XPOSURE", "EXPTIME", "EXPOSURE"];
const TEMPERATURE_HEADER_KEYS: &[&str] = &["CCD-TEMP", "SET-TEMP"];
const FOCAL_LENGTH_HEADER_KEYS: &[&str] = &["FOCALLEN", "FOCALLENGTH", "FOCAL"];
const ROTATION_HEADER_KEYS: &[&str] = &["ROTATANG", "ROTATOR", "ROTPOS"];
const CAPTURE_TIME_HEADER_KEYS: &[&str] = &["DATE-OBS", "DATE-BEG", "DATE-AVG"];

/// Calibration markers carried by a decoded frame's headers.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct FrameCalibrationState {
    /// A bias pedestal has already been removed.
    pub bias_subtracted: bool,
    /// Dark current has already been removed.
    pub dark_subtracted: bool,
    /// Flat response has already been normalized/applied.
    pub flat_normalized: bool,
}

/// Normalized acquisition role declared by a frame's metadata.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FrameSourceRole {
    /// A raw or integrated bias/offset frame.
    Bias,
    /// A raw or integrated dark frame.
    Dark,
    /// A dark captured for flat calibration.
    DarkFlat,
    /// A raw or integrated flat frame.
    Flat,
    /// A science/light/object frame.
    Light,
    /// No recognized role was declared.
    #[default]
    Unknown,
}

impl FrameSourceRole {
    /// Stable kebab-case spelling used by JSON and diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bias => "bias",
            Self::Dark => "dark",
            Self::DarkFlat => "dark-flat",
            Self::Flat => "flat",
            Self::Light => "light",
            Self::Unknown => "unknown",
        }
    }
}

impl std::str::FromStr for FrameSourceRole {
    type Err = Error;

    /// Read the spelling [`FrameSourceRole::as_str`] writes.
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "bias" => Ok(Self::Bias),
            "dark" => Ok(Self::Dark),
            "dark-flat" => Ok(Self::DarkFlat),
            "flat" => Ok(Self::Flat),
            "light" => Ok(Self::Light),
            "unknown" => Ok(Self::Unknown),
            other => Err(Error::Calibration(format!(
                "frame role must be bias, dark, dark-flat, flat, light or unknown, not {other:?}"
            ))),
        }
    }
}

impl FrameCalibrationState {
    /// Whether any calibration stage is already declared complete.
    pub fn is_calibrated(self) -> bool {
        self.bias_subtracted || self.dark_subtracted || self.flat_normalized
    }
}

/// Normalized metadata used to decide whether calibration is safe.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FrameMetadata {
    /// Normalized source role.
    #[serde(default)]
    pub role: FrameSourceRole,
    /// Acquisition settings used by shared calibration matching policy.
    pub signature: FrameSignature,
    /// Whether the source declares itself to be an integrated master.
    pub is_master: bool,
    /// Calibration steps the source says have already been applied.
    pub calibration_state: FrameCalibrationState,
}

impl FrameMetadata {
    /// Normalize calibration metadata from FITS-style header cards.
    ///
    /// Structural dimensions come from `NAXIS*` when present. A decoded frame
    /// should use [`FitsFrame::metadata`] so its verified geometry is used as
    /// the fallback.
    pub fn from_headers(headers: &[(String, HeaderValue)]) -> Self {
        metadata_from_headers(headers, None, None, None)
    }

    /// Normalize headers while using a decoded image as structural fallback.
    pub fn from_image(image: &LinearImage, headers: &[(String, HeaderValue)]) -> Self {
        metadata_from_headers(
            headers,
            Some((image.width, image.height, image.channels)),
            None,
            None,
        )
    }
}

/// A FITS or XISF frame decoded into linear, un-stretched `f32` samples.
#[derive(Clone, Debug)]
pub struct FitsFrame {
    /// Decoded linear image samples.
    pub image: LinearImage,
    /// Raw FITS header cards, preserved for metadata and WCS copying.
    pub headers: Vec<(String, HeaderValue)>,
    /// Exposure time in seconds, when the headers report a positive value.
    pub exposure_seconds: Option<f64>,
    /// Raw CFA sampling of a one-channel frame, when present.
    pub bayer: Option<BayerLayout>,
    /// Path the frame was read from, when opened from disk.
    pub source: Option<PathBuf>,
    /// The sample range an XISF source declared, when it declared a usable
    /// one. `None` for FITS, and for XISF that said nothing.
    ///
    /// Samples are never rescaled by it — see
    /// [`seiza_xisf::XisfImageInfo::bounds`] for why the attribute is a hint
    /// rather than a fact. It rides along so a caller mixing a normalized
    /// PixInsight frame with camera frames can notice that it is about to
    /// stack samples four orders of magnitude apart, and decide for itself.
    pub bounds: Option<(f64, f64)>,
}

impl FitsFrame {
    /// Read and decode a FITS or XISF file into a linear frame.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        // A camera's frame decodes straight to f32 into a buffer a finished
        // frame gave back, whose pages are mapped.
        let read = seiza_image_io::read_astronomy_with(path, crate::recycle::take)
            .map_err(|error| image_io_error(error, false))?;
        let mut image = read.image;
        // The metadata carries the astrometric solution, which a PixInsight
        // file keeps as properties rather than WCS cards.
        if let Some(metadata) = &read.xisf_metadata {
            seiza_image_io::add_xisf_solution_wcs(&mut image.headers, metadata);
        }
        let bounds = read.xisf_info.and_then(|info| info.bounds);
        let mut frame = Self::from_fits(image, Some(path.to_path_buf()))?;
        frame.bounds = bounds;
        Ok(frame)
    }

    /// Put a frame the file declares as normalized onto `full_scale`, and
    /// report whether it did.
    ///
    /// PixInsight writes floating-point images normalized to `bounds="0:1"`,
    /// so such a frame's samples run 0..1 where a camera frame's run in the
    /// thousands. Stacking the two together compares values four orders of
    /// magnitude apart, which wrecks normalization and rejection for the whole
    /// group.
    ///
    /// Only an exact `0:1` is converted, because that is the one spelling
    /// whose meaning is settled — this toolkit's own writer reports the
    /// observed sample minimum and maximum, so converting from any other
    /// declared range would as easily stretch an already-physical frame.
    /// `bounds` is updated to the new range, so a second call does nothing.
    pub fn rescale_declared_unit_bounds(&mut self, full_scale: f32) -> bool {
        if self.bounds != Some((0.0, 1.0)) || !full_scale.is_finite() || full_scale <= 0.0 {
            return false;
        }
        for sample in &mut self.image.data {
            *sample *= full_scale;
        }
        self.bounds = Some((0.0, f64::from(full_scale)));
        true
    }

    /// Convert an already-decoded [`FitsImage`] into a linear frame,
    /// interleaving color planes and reading exposure and CFA metadata.
    pub fn from_fits(fits: FitsImage, source: Option<PathBuf>) -> Result<Self> {
        let bayer_pattern = fits.bayer_pattern_in_storage_order();
        let x_offset = fits.header_f64("XBAYROFF").unwrap_or(0.0).max(0.0) as usize;
        let y_offset = fits.header_f64("YBAYROFF").unwrap_or(0.0).max(0.0) as usize;
        let exposure_seconds = EXPOSURE_HEADER_KEYS
            .iter()
            .find_map(|key| fits.header_f64(key))
            .filter(|value| value.is_finite() && *value > 0.0);
        let (width, height, planes) = (fits.width, fits.height, fits.planes);
        let headers = fits.headers.clone();
        let physical = fits.into_physical_f32_with(crate::recycle::take);

        let channels = if planes == 3 { 3 } else { 1 };
        let data = if channels == 3 {
            let interleaved = planar_to_interleaved(&physical, width * height);
            crate::recycle::give(physical);
            interleaved
        } else {
            physical
        };
        let bayer = if channels == 1 {
            bayer_pattern.map(|pattern| BayerLayout {
                pattern,
                x_offset,
                y_offset,
            })
        } else {
            None
        };

        Ok(Self {
            image: LinearImage::new(width, height, channels, data)?,
            headers,
            exposure_seconds,
            bayer,
            source,
            // Only `open` sees the container, so only `open` can fill this in.
            bounds: None,
        })
    }

    /// Reject a master whose declared kind does not match its use.
    ///
    /// Seiza masters declare `SEIZAMST`. External masters commonly declare
    /// only `IMAGETYP`/`OBSTYPE`; a recognized, contradictory role there is
    /// just as unsafe and is rejected too. A source with no recognizable role
    /// retains the legacy behavior and is accepted.
    pub fn validate_master_kind(&self, expected: &str) -> Result<()> {
        if let Some(actual) = self
            .headers
            .iter()
            .find(|(key, _)| key == "SEIZAMST")
            .and_then(|(_, value)| value.as_str())
            && !actual.eq_ignore_ascii_case(expected)
        {
            return Err(Error::Calibration(format!(
                "expected a {expected} master but FITS declares {actual}"
            )));
        }
        let role = self.metadata().role;
        let role_matches = if expected.eq_ignore_ascii_case("BIAS") {
            role == FrameSourceRole::Bias
        } else if expected.eq_ignore_ascii_case("DARK") {
            matches!(role, FrameSourceRole::Dark | FrameSourceRole::DarkFlat)
        } else if expected.eq_ignore_ascii_case("FLAT") {
            role == FrameSourceRole::Flat
        } else {
            false
        };
        if role != FrameSourceRole::Unknown && !role_matches {
            return Err(Error::Calibration(format!(
                "expected a {expected} master but frame metadata declares {}",
                role.as_str()
            )));
        }
        Ok(())
    }

    /// Normalize the acquisition and calibration metadata of this decoded
    /// frame. Decoded dimensions, channel count, exposure, and CFA layout are
    /// authoritative when a header omitted them.
    pub fn metadata(&self) -> FrameMetadata {
        metadata_from_headers(
            &self.headers,
            Some((self.image.width, self.image.height, self.image.channels)),
            self.exposure_seconds,
            self.bayer,
        )
    }

    /// Convert raw CFA sampling to the prepared RGB grid used by registration
    /// and stacking. Planar RGB and mono frames pass through unchanged. VNG
    /// runs on the GPU where the default [`crate::GpuOptions`] allow it; see
    /// [`Self::into_prepared_with_gpu`].
    pub fn into_prepared(self) -> Result<Self> {
        self.into_prepared_with_gpu(&crate::GpuOptions::default())
    }

    /// [`Self::into_prepared`], with VNG on the GPU only where `gpu` allows
    /// it. [`crate::GpuOptions::cpu`] keeps the debayering on the CPU.
    pub fn into_prepared_with_gpu(self, gpu: &crate::GpuOptions) -> Result<Self> {
        gpu.validate()?;
        Ok(self
            .into_prepared_with_layout(crate::Demosaic::default(), gpu)?
            .0)
    }

    /// This frame binned by `factor`, as [`crate::StackOptions::bin_factor`]
    /// bins every light once it is calibrated, repaired and debayered: its
    /// image by [`LinearImage::binned`], and its headers carried to the
    /// coarser grid. `XBINNING` and `YBINNING` (or `CCDXBIN` and `CCDYBIN`)
    /// and `XPIXSZ` and `YPIXSZ` grow by the factor, as SBFITSEXT defines
    /// them after binning, so a focal length or plate scale read from them
    /// stays right; `SEIZABIN` records the software bin, so a reader can tell
    /// it from the camera's; `NAXIS1` and `NAXIS2` give the binned size; and
    /// the WCS moves to the binned grid by [`rescaled_reference_headers`].
    /// A raw CFA frame is refused, since binning follows debayering. A
    /// factor of 1 returns the frame as it is.
    pub fn binned(self, factor: u32) -> Result<Self> {
        if factor == 1 {
            return Ok(self);
        }
        if self.bayer.is_some() {
            return Err(Error::InvalidImage(
                "a raw CFA frame is binned only once it is debayered".into(),
            ));
        }
        let mut frame = Self {
            image: self.image.binned(factor)?,
            ..self
        };
        frame.rebin_headers(factor);
        Ok(frame)
    }

    /// Carry the headers to this frame's image, just binned by `factor`.
    fn rebin_headers(&mut self, factor: u32) {
        self.headers = rebinned_headers(&self.headers, 1, factor);
        for (key, value) in &mut self.headers {
            let size = match key.as_str() {
                "NAXIS1" => self.image.width,
                "NAXIS2" => self.image.height,
                _ => continue,
            };
            *value = HeaderValue::Integer(size as i64);
        }
    }

    /// [`Self::into_prepared`], also returning the Bayer layout the frame was
    /// debayered from, for Bayer drizzle. The prepared frame itself carries
    /// none. VNG runs on the GPU where `gpu` allows it.
    pub(crate) fn into_prepared_with_layout(
        mut self,
        demosaic: crate::Demosaic,
        gpu: &crate::GpuOptions,
    ) -> Result<(Self, Option<BayerLayout>)> {
        let layout = self.bayer.take();
        if let Some(layout) = layout {
            self.image = self.image.debayer_with(layout, demosaic, gpu)?;
            // A superpixel is a bin of 2 as well as a demosaic.
            if demosaic == crate::Demosaic::Superpixel {
                self.rebin_headers(2);
            }
        }
        Ok((self, layout))
    }
}

fn metadata_from_headers(
    headers: &[(String, HeaderValue)],
    fallback_dimensions: Option<(usize, usize, usize)>,
    fallback_exposure: Option<f64>,
    fallback_bayer: Option<BayerLayout>,
) -> FrameMetadata {
    let fallback_width = fallback_dimensions.and_then(|value| i64::try_from(value.0).ok());
    let fallback_height = fallback_dimensions.and_then(|value| i64::try_from(value.1).ok());
    let fallback_channels = fallback_dimensions.and_then(|value| i64::try_from(value.2).ok());
    let width = header_i64(headers, &["NAXIS1"])
        .filter(|value| *value > 0)
        .or(fallback_width);
    let height = header_i64(headers, &["NAXIS2"])
        .filter(|value| *value > 0)
        .or(fallback_height);
    let bayer_pattern = header_text(headers, &["BAYERPAT"])
        .map(|value| {
            let pattern = BayerPattern::parse(&value);
            let row_order =
                header_text(headers, &["ROWORDER"]).and_then(|value| RowOrder::parse(&value));
            match (pattern, row_order, height) {
                (Some(pattern), Some(order), Some(height)) => pattern
                    .in_row_order(order, height as usize)
                    .as_str()
                    .to_owned(),
                _ => value.to_ascii_uppercase(),
            }
        })
        .or_else(|| fallback_bayer.map(|layout| layout.pattern.as_str().to_ascii_uppercase()));
    let channels = if width.is_some() && height.is_some() {
        Some(
            header_i64(headers, &["NAXIS3"])
                .filter(|value| *value > 0)
                .or_else(|| bayer_pattern.as_ref().map(|_| 1))
                .or(fallback_channels)
                .unwrap_or(1),
        )
    } else {
        fallback_channels
    };
    let exposure_seconds = header_f64(headers, EXPOSURE_HEADER_KEYS)
        .filter(|value| *value > 0.0)
        .or(fallback_exposure);
    let declared_master = header_text(headers, &["SEIZAMST"]);
    let raw_image_type = header_text(headers, &["IMAGETYP", "OBSTYPE", "FRAME"]);
    let role_source = declared_master.as_deref().or(raw_image_type.as_deref());
    let normalized_type = role_source.map(normalize_role_text).unwrap_or_default();

    let mut signature = FrameSignature::default();
    signature.camera = header_text(headers, CAMERA_HEADER_KEYS);
    signature.telescope = header_text(headers, TELESCOPE_HEADER_KEYS);
    signature.width = width;
    signature.height = height;
    signature.channels = channels;
    signature.binning_x = header_i64(headers, BINNING_X_HEADER_KEYS);
    signature.binning_y = header_i64(headers, BINNING_Y_HEADER_KEYS);
    signature.gain = header_i64(headers, &["GAIN"]);
    signature.offset = header_i64(headers, &["OFFSET"]);
    signature.readout_mode = header_i64(headers, READOUT_HEADER_KEYS);
    signature.bayer_pattern = bayer_pattern;
    signature.filter = header_text(headers, &["FILTER"]);
    signature.focal_length_mm =
        header_f64(headers, FOCAL_LENGTH_HEADER_KEYS).filter(|value| *value > 0.0);
    signature.rotation_deg = header_f64(headers, ROTATION_HEADER_KEYS);
    signature.exposure_seconds = exposure_seconds;
    signature.camera_temp_c = header_f64(headers, TEMPERATURE_HEADER_KEYS);
    signature.captured_at_unix = header_text(headers, CAPTURE_TIME_HEADER_KEYS)
        .as_deref()
        .and_then(parse_iso_unix);

    FrameMetadata {
        role: normalize_source_role(&normalized_type),
        signature,
        is_master: declared_master.is_some() || normalized_type.contains("master"),
        calibration_state: FrameCalibrationState {
            bias_subtracted: header_bool(headers, "BIASSUB").unwrap_or(false),
            dark_subtracted: header_bool(headers, "DARKSUB").unwrap_or(false),
            flat_normalized: header_bool(headers, "FLATNORM").unwrap_or(false),
        },
    }
}

fn normalize_source_role(normalized: &str) -> FrameSourceRole {
    if normalized.contains("darkflat") || normalized.contains("flatdark") {
        FrameSourceRole::DarkFlat
    } else if normalized.contains("bias") || normalized.contains("offset") {
        FrameSourceRole::Bias
    } else if normalized.contains("dark") {
        FrameSourceRole::Dark
    } else if normalized.contains("flat") {
        FrameSourceRole::Flat
    } else if normalized.contains("light") || normalized.contains("object") {
        FrameSourceRole::Light
    } else {
        FrameSourceRole::Unknown
    }
}

fn normalize_role_text(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn header_value<'a>(
    headers: &'a [(String, HeaderValue)],
    keys: &[&str],
) -> Option<&'a HeaderValue> {
    keys.iter().find_map(|key| {
        headers
            .iter()
            .find(|(candidate, _)| candidate.eq_ignore_ascii_case(key))
            .map(|(_, value)| value)
    })
}

fn header_text(headers: &[(String, HeaderValue)], keys: &[&str]) -> Option<String> {
    let value = header_value(headers, keys)?;
    let value = match value {
        HeaderValue::String(value) | HeaderValue::Raw(value) => value.trim(),
        _ => return None,
    };
    (!value.is_empty()).then(|| value.to_owned())
}

fn header_f64(headers: &[(String, HeaderValue)], keys: &[&str]) -> Option<f64> {
    header_value(headers, keys)?
        .as_f64()
        .filter(|value| value.is_finite())
}

fn header_i64(headers: &[(String, HeaderValue)], keys: &[&str]) -> Option<i64> {
    let value = header_f64(headers, keys)?;
    (value >= i64::MIN as f64 && value <= i64::MAX as f64).then_some(value as i64)
}

fn header_bool(headers: &[(String, HeaderValue)], key: &str) -> Option<bool> {
    let value = header_value(headers, &[key])?;
    match value {
        HeaderValue::Logical(value) => Some(*value),
        HeaderValue::Integer(value) => Some(*value != 0),
        HeaderValue::Float(value) if value.is_finite() => Some(*value != 0.0),
        HeaderValue::String(value) | HeaderValue::Raw(value) => {
            match value.trim().to_ascii_lowercase().as_str() {
                "true" | "t" | "yes" | "y" | "1" => Some(true),
                "false" | "f" | "no" | "n" | "0" => Some(false),
                _ => None,
            }
        }
        _ => None,
    }
}

fn parse_iso_unix(value: &str) -> Option<i64> {
    let value = value.trim().trim_end_matches('Z');
    let (date, clock) = value.split_once('T').unwrap_or((value, "0:0:0"));
    let mut date_parts = date.split('-');
    let year: i32 = date_parts.next()?.parse().ok()?;
    let month: u32 = date_parts.next()?.parse().ok()?;
    let day: u32 = date_parts.next()?.parse().ok()?;
    let mut clock_parts = clock.split(':');
    let hour: f64 = clock_parts.next()?.parse().ok()?;
    let minute: f64 = clock_parts.next().unwrap_or("0").parse().ok()?;
    let second: f64 = clock_parts.next().unwrap_or("0").parse().ok()?;
    let day_fraction = day as f64 + (hour + minute / 60.0 + second / 3_600.0) / 24.0;
    let julian = seiza::minor_bodies::julian_date(year, month, day_fraction);
    let seconds = (julian - 2_440_587.5) * 86_400.0;
    (seconds.is_finite() && seconds >= i64::MIN as f64 && seconds <= i64::MAX as f64)
        .then_some(seconds.round() as i64)
}

fn planar_to_interleaved(planar: &[f32], pixel_count: usize) -> Vec<f32> {
    let mut output = vec![0.0; pixel_count * 3];
    for index in 0..pixel_count {
        output[index * 3] = planar[index];
        output[index * 3 + 1] = planar[pixel_count + index];
        output[index * 3 + 2] = planar[pixel_count * 2 + index];
    }
    output
}

/// Write an unstretched stack as a primary-HDU 32-bit floating-point FITS,
/// or as monolithic XISF when the path ends in `.xisf`.
pub fn write_fits_f32(
    path: impl AsRef<Path>,
    snapshot: &StackSnapshot,
    reference_headers: &[(String, HeaderValue)],
) -> Result<()> {
    write_stack_image_fits_f32(
        path,
        &snapshot.image,
        snapshot.accepted_frames,
        snapshot.rejected_frames,
        reference_headers,
    )
}

/// Write a drizzled stack as unstretched 32-bit floating point FITS, or as
/// monolithic XISF when the path ends in `.xisf`, with the reference WCS
/// carried to the drizzle grid by [`drizzle_reference_headers`].
pub fn write_drizzle_fits_f32(
    path: impl AsRef<Path>,
    drizzle: &crate::DrizzleResult,
    accepted_frames: u32,
    rejected_frames: u32,
    reference_headers: &[(String, HeaderValue)],
) -> Result<()> {
    let headers = drizzle_reference_headers(reference_headers, drizzle.scale);
    let cards = [
        integer_card("STACKCNT", accepted_frames as i64, "accepted input frames"),
        integer_card("STACKREJ", rejected_frames as i64, "rejected input frames"),
        integer_card(
            "DRIZSCL",
            i64::from(drizzle.scale),
            "drizzle output pixels per reference pixel",
        ),
    ];
    write_linear_image_fits_f32(path, &drizzle.image, &headers, &cards)
}

/// Write a compact immutable live-stack export as unstretched 32-bit floating
/// point FITS, or as monolithic XISF when the path ends in `.xisf`.
pub fn write_stack_export_fits_f32(
    path: impl AsRef<Path>,
    snapshot: &StackExportSnapshot,
    reference_headers: &[(String, HeaderValue)],
) -> Result<()> {
    write_stack_image_fits_f32(
        path,
        &snapshot.image,
        snapshot.accepted_frames,
        snapshot.rejected_frames,
        reference_headers,
    )
}

fn write_stack_image_fits_f32(
    path: impl AsRef<Path>,
    image: &LinearImage,
    accepted_frames: u32,
    rejected_frames: u32,
    reference_headers: &[(String, HeaderValue)],
) -> Result<()> {
    let mut cards = vec![integer_card(
        "STACKCNT",
        accepted_frames as i64,
        "accepted input frames",
    )];
    cards.push(integer_card(
        "STACKREJ",
        rejected_frames as i64,
        "rejected input frames",
    ));
    // A stack of binned lights says so, as its reference does.
    if let Some(bin) = software_bin(reference_headers) {
        cards.push(integer_card(SOFTWARE_BIN_KEY, bin, SOFTWARE_BIN_COMMENT));
    }
    write_linear_image_fits_f32(path, image, reference_headers, &cards)
}

/// The card recording a software bin after calibration, which
/// [`FitsFrame::binned`] writes.
const SOFTWARE_BIN_KEY: &str = "SEIZABIN";
const SOFTWARE_BIN_COMMENT: &str = "software bin after calibration";

/// The software bin `headers` record, when above 1.
fn software_bin(headers: &[(String, HeaderValue)]) -> Option<i64> {
    headers
        .iter()
        .find(|(key, _)| key == SOFTWARE_BIN_KEY)
        .and_then(|(_, value)| value.as_f64())
        .filter(|bin| bin.is_finite() && *bin > 1.0)
        .map(|bin| bin.round() as i64)
}

/// The headers of an image binned by `from` (1 for the sensor's own
/// sampling), carried to the same image binned by `to`: the WCS by
/// [`rescaled_reference_headers`], the binning and pixel-size cards by the
/// ratio, and `SEIZABIN` to the software bin that results, dropped at 1.
pub(crate) fn rebinned_headers(
    headers: &[(String, HeaderValue)],
    from: u32,
    to: u32,
) -> Vec<(String, HeaderValue)> {
    let ratio = f64::from(to) / f64::from(from);
    let software = software_bin(headers).unwrap_or(1) as f64 * ratio;
    let mut headers = rescaled_reference_headers(headers, 1.0 / ratio);
    headers.retain(|(key, _)| key != SOFTWARE_BIN_KEY);
    for (key, value) in &mut headers {
        if !matches!(
            key.as_str(),
            "XBINNING" | "YBINNING" | "CCDXBIN" | "CCDYBIN" | "XPIXSZ" | "YPIXSZ"
        ) {
            continue;
        }
        let rewritten = match value {
            HeaderValue::Integer(number) if (*number as f64 * ratio).fract() == 0.0 => {
                HeaderValue::Integer((*number as f64 * ratio) as i64)
            }
            HeaderValue::Integer(number) => HeaderValue::Float(*number as f64 * ratio),
            HeaderValue::Float(number) if number.is_finite() => HeaderValue::Float(*number * ratio),
            _ => continue,
        };
        *value = rewritten;
    }
    if software.round() > 1.0 {
        headers.push((
            SOFTWARE_BIN_KEY.into(),
            HeaderValue::Integer(software.round() as i64),
        ));
    }
    headers
}

/// Write a composed RGB image as primary-HDU 32-bit floating-point FITS.
///
/// `label` identifies the composition (for example `LRGB`, `SHO`, or
/// `FORAXX-SHO`). WCS cards are copied from the chosen aligned reference. A
/// cropped composition moves `CRPIX` by its region origin, so the written
/// image keeps the reference solution, and records that origin.
pub fn write_color_fits_f32(
    path: impl AsRef<Path>,
    composition: &ColorComposition,
    reference_headers: &[(String, HeaderValue)],
    label: &str,
) -> Result<()> {
    if composition.image.channels != 3 {
        return Err(Error::Color(
            "color FITS output must have three channels".into(),
        ));
    }
    let mut cards = vec![
        string_card("COLORSPC", "RGB", "RGB color planes"),
        string_card("SEIZACLR", label, "Seiza color composition"),
        string_card(
            "SEIZATRF",
            composition.transfer.fits_name(),
            "sample transfer semantics",
        ),
    ];
    let region = composition.region;
    if region.x == 0 && region.y == 0 {
        return write_linear_image_fits_f32(path, &composition.image, reference_headers, &cards);
    }
    cards.push(integer_card(
        "SEIZACRX",
        region.x as i64,
        "crop origin column on the reference grid",
    ));
    cards.push(integer_card(
        "SEIZACRY",
        region.y as i64,
        "crop origin row on the reference grid",
    ));
    let headers = shift_reference_origin(reference_headers, region.x, region.y);
    write_linear_image_fits_f32(path, &composition.image, &headers, &cards)
}

/// A reference frame's headers for an image drizzled `scale` times finer,
/// such as [`crate::DrizzleResult::image`]: the WCS keeps the same sky. See
/// [`rescaled_reference_headers`].
pub fn drizzle_reference_headers(
    headers: &[(String, HeaderValue)],
    scale: u32,
) -> Vec<(String, HeaderValue)> {
    rescaled_reference_headers(headers, f64::from(scale))
}

/// A reference frame's headers for the same sky on a grid `scale` times as
/// fine: above 1 for a drizzle, `1 / b` for the reference binned by `b`.
///
/// `CRPIX` moves to the new grid, whose pixel edges fall at the
/// reference's: with pixel centres on whole numbers and FITS counting from
/// 1, `CRPIX' = (CRPIX − 0.5) · scale + 0.5`. `CD` and `CDELT` shrink by
/// the scale, `A_DMAX` and `B_DMAX` grow by it, and each SIP coefficient of
/// order `p + q` scales by `scale^(1 − p − q)`, since SIP maps pixel
/// offsets to pixel offsets. Binning by `b` and drizzling the result by `b`
/// gives the reference's WCS back.
pub fn rescaled_reference_headers(
    headers: &[(String, HeaderValue)],
    scale: f64,
) -> Vec<(String, HeaderValue)> {
    let sip_order = |key: &str| {
        let rest = ["AP_", "BP_", "A_", "B_"]
            .iter()
            .find_map(|prefix| key.strip_prefix(prefix))?;
        let (p, q) = rest.split_once('_')?;
        Some(p.parse::<i32>().ok()? + q.parse::<i32>().ok()?)
    };
    headers
        .iter()
        .map(|(key, value)| {
            let factor = match key.as_str() {
                "CRPIX1" | "CRPIX2" => {
                    return match value.as_f64() {
                        Some(pixel) if pixel.is_finite() => {
                            (key.clone(), HeaderValue::Float((pixel - 0.5) * scale + 0.5))
                        }
                        _ => (key.clone(), value.clone()),
                    };
                }
                "CDELT1" | "CDELT2" => 1.0 / scale,
                // The largest SIP correction, in pixels.
                "A_DMAX" | "B_DMAX" => scale,
                key if key.starts_with("CD1_") || key.starts_with("CD2_") => 1.0 / scale,
                key => match sip_order(key) {
                    Some(order) => scale.powi(1 - order),
                    None => return (key.to_owned(), value.clone()),
                },
            };
            match value.as_f64() {
                Some(number) if number.is_finite() => {
                    (key.clone(), HeaderValue::Float(number * factor))
                }
                _ => (key.clone(), value.clone()),
            }
        })
        .collect()
}

/// Move a reference frame's `CRPIX` to a crop's own pixel coordinates.
///
/// `CRPIX` is the one and only WCS card that a translation changes: the
/// rotation matrix, the reference world coordinate, and SIP distortion terms
/// are all expressed relative to it.
fn shift_reference_origin(
    headers: &[(String, HeaderValue)],
    x: usize,
    y: usize,
) -> Vec<(String, HeaderValue)> {
    headers
        .iter()
        .map(|(key, value)| {
            let shift = match key.as_str() {
                "CRPIX1" => x,
                "CRPIX2" => y,
                _ => return (key.clone(), value.clone()),
            };
            match value.as_f64() {
                Some(reference) if reference.is_finite() => {
                    (key.clone(), HeaderValue::Float(reference - shift as f64))
                }
                _ => (key.clone(), value.clone()),
            }
        })
        .collect()
}

/// Write an integrated calibration master with explicit calibration-state
/// headers and the normalized acquisition metadata used to match it later.
/// Header aliases collapse to one canonical FITS card per semantic field.
pub fn write_master_fits_f32(path: impl AsRef<Path>, master: &MasterFrame) -> Result<()> {
    let mut cards = vec![
        string_card(
            "SEIZAMST",
            master.kind.fits_name(),
            "Seiza master frame kind",
        ),
        integer_card("SEIZAVR", 1, "Seiza master header schema"),
        integer_card(
            "NCOMBINE",
            master.input_frames as i64,
            "integrated calibration frames",
        ),
        logical_card(
            "BIASSUB",
            master.bias_subtracted,
            "bias pedestal already removed",
        ),
        logical_card(
            "DARKSUB",
            master.dark_subtracted,
            "dark or dark-flat already removed",
        ),
        logical_card(
            "FLATNORM",
            master.normalized,
            "flat response normalized before combine",
        ),
        string_card(
            "REJMETH",
            master.rejection_method.as_str(),
            "temporal sample rejection method",
        ),
        float_card(
            "CLIPLOW",
            f64::from(master.rejection.low_sigma),
            "low sigma threshold",
        ),
        float_card(
            "CLIPHIGH",
            f64::from(master.rejection.high_sigma),
            "high sigma threshold",
        ),
        integer_card(
            "CLIPREJ",
            i64::try_from(master.rejected_samples).unwrap_or(i64::MAX),
            "rejected input samples",
        ),
        integer_card(
            "CLIPACC",
            i64::try_from(master.accepted_samples).unwrap_or(i64::MAX),
            "accepted input samples",
        ),
        integer_card(
            "CLIPFBK",
            i64::try_from(master.fallback_pixels).unwrap_or(i64::MAX),
            "pixels using the fallback center",
        ),
        logical_card(
            "STARMASK",
            master.flat_star_masking.is_some(),
            "native per-input flat star masks",
        ),
        integer_card(
            "MASKSAMP",
            i64::try_from(master.masked_samples).unwrap_or(i64::MAX),
            "input samples excluded by masks",
        ),
    ];
    if let Some(masking) = &master.flat_star_masking {
        cards.extend([
            integer_card(
                "MASKPIX",
                i64::try_from(masking.masked_pixels).unwrap_or(i64::MAX),
                "output pixels touched by input masks",
            ),
            integer_card(
                "COVMIN",
                i64::try_from(masking.minimum_clean_samples).unwrap_or(i64::MAX),
                "minimum retained input coverage",
            ),
            integer_card(
                "COVMAX",
                i64::try_from(masking.maximum_clean_samples).unwrap_or(i64::MAX),
                "maximum retained input coverage",
            ),
            integer_card(
                "COVLOW",
                i64::try_from(masking.low_coverage_samples).unwrap_or(i64::MAX),
                "output samples with fewer than 3 inputs",
            ),
            integer_card(
                "COVREQ",
                i64::try_from(masking.options.minimum_clean_samples).unwrap_or(i64::MAX),
                "required retained input coverage",
            ),
            integer_card(
                "SATUNMSK",
                i64::try_from(masking.unmasked_saturation_samples).unwrap_or(i64::MAX),
                "unmasked isolated saturation samples",
            ),
            integer_card(
                "SATUNKN",
                i64::try_from(masking.unknown_saturation_inputs).unwrap_or(i64::MAX),
                "inputs without known saturation ceiling",
            ),
            float_card(
                "MSKSIG",
                f64::from(masking.options.detection_sigma),
                "star detection sigma threshold",
            ),
            float_card(
                "MSKRAD",
                f64::from(masking.options.halo_radius_factor),
                "star halo radius multiplier",
            ),
            float_card(
                "MSKMINR",
                f64::from(masking.options.minimum_radius_pixels),
                "minimum mask radius in raw pixels",
            ),
        ]);
        if let Some(level) = masking.options.saturation_level {
            cards.push(float_card(
                "MSKSAT",
                f64::from(level),
                "explicit raw saturation ceiling",
            ));
        }
    }
    if let Some(exposure_seconds) = master.exposure_seconds {
        cards.push(float_card(
            "EXPTIME",
            exposure_seconds,
            "master dark exposure seconds",
        ));
    }
    if let Some(bayer) = master.bayer {
        // The in-memory layout is in storage order. Convert it back to the
        // sensor convention of the retained ROWORDER, avoiding a second flip
        // when this master is read again.
        let pattern = header_text(&master.reference_headers, &["ROWORDER"])
            .and_then(|value| RowOrder::parse(&value))
            .map_or(bayer.pattern, |order| {
                bayer.pattern.in_row_order(order, master.image.height)
            });
        cards.push(string_card(
            "BAYERPAT",
            pattern.as_str(),
            "raw color-filter-array layout",
        ));
        cards.push(integer_card(
            "XBAYROFF",
            bayer.x_offset as i64,
            "CFA horizontal origin offset",
        ));
        cards.push(integer_card(
            "YBAYROFF",
            bayer.y_offset as i64,
            "CFA vertical origin offset",
        ));
    }
    for &(output_key, aliases) in MASTER_METADATA_HEADER_GROUPS {
        if cards.iter().any(|card| card.keyword() == output_key) {
            continue;
        }
        let Some(value) = aliases.iter().find_map(|alias| {
            master
                .reference_headers
                .iter()
                .find(|(candidate, _)| candidate.eq_ignore_ascii_case(alias))
                .map(|(_, value)| value)
        }) else {
            continue;
        };
        if let Some(card) = value_card(output_key, value) {
            cards.push(card);
        }
    }
    write_linear_fits_f32(path.as_ref(), &master.image, cards)
}

/// Write a linear image while preserving a valid WCS from a reference frame.
///
/// `extra_cards` describes the processing operation. Structural FITS cards
/// are generated by the writer, and duplicate WCS cards in `extra_cards` take
/// precedence over the reference. A `.xisf` output path writes monolithic
/// XISF instead of FITS.
pub fn write_linear_image_fits_f32(
    path: impl AsRef<Path>,
    image: &LinearImage,
    reference_headers: &[(String, HeaderValue)],
    extra_cards: &[WriteHeaderCard],
) -> Result<()> {
    let mut cards = extra_cards.to_vec();
    append_reference_wcs(&mut cards, reference_headers);
    write_linear_fits_f32(path.as_ref(), image, cards)
}

/// Write a processed version of one source image, preserving its valid WCS
/// and observation/instrument metadata while replacing structural pixel
/// cards. A `.xisf` output path writes monolithic XISF instead of FITS.
pub fn write_processed_image_fits_f32(
    path: impl AsRef<Path>,
    image: &LinearImage,
    reference_headers: &[(String, HeaderValue)],
    extra_cards: &[WriteHeaderCard],
) -> Result<()> {
    let mut cards = extra_cards.to_vec();
    append_reference_wcs(&mut cards, reference_headers);
    for (key, value) in reference_headers {
        if preserve_processed_key(key)
            && (image.channels == 1 || !is_bayer_key(key))
            && !cards.iter().any(|card| card.keyword() == key)
            && let Some(card) = value_card(key, value)
        {
            cards.push(card);
        }
    }
    write_linear_fits_f32(path.as_ref(), image, cards)
}

fn is_bayer_key(key: &str) -> bool {
    matches!(key, "BAYERPAT" | "XBAYROFF" | "YBAYROFF")
}

fn write_linear_fits_f32(
    path: &Path,
    image: &LinearImage,
    extra_cards: Vec<WriteHeaderCard>,
) -> Result<()> {
    let pixels = if image.channels == 3 {
        F32ImageData::RgbInterleaved(&image.data)
    } else {
        F32ImageData::Mono(&image.data)
    };
    // XISF by its extension (a `.xish` header is refused), FITS otherwise.
    let format = match seiza_image_io::Format::from_path(path) {
        Some(seiza_image_io::Format::Xisf) => seiza_image_io::Format::Xisf,
        _ => seiza_image_io::Format::Fits,
    };
    let options = seiza_image_io::SaveOptions {
        format: Some(format),
        headers: &extra_cards,
        // Stacking works in 16-bit camera counts. Declaring that range lets
        // PixInsight load the samples on that scale rather than stretching
        // their own minimum and maximum across the display.
        xisf_bounds: Some((0.0, 65535.0)),
        // The reference frame's own ROWORDER, if any, travels in the cards.
        fits_top_down: false,
    };
    seiza_image_io::save_f32(path, image.width, image.height, pixels, &options)
        .map_err(|error| image_io_error(error, true))
}

/// Stacking's error for an image the shared reader or writer refused: the
/// FITS or XISF error it met, as a read or a write.
fn image_io_error(error: seiza_image_io::Error, writing: bool) -> Error {
    use seiza_image_io::Cause;
    let (path, cause) = error.into_parts();
    match (cause, writing) {
        (Cause::Xisf(source), false) => Error::XisfRead { path, source },
        (Cause::Xisf(source), true) => Error::XisfWrite { path, source },
        (Cause::Fits(source), false) => Error::FitsRead { path, source },
        (Cause::Fits(source), true) => Error::FitsWrite { path, source },
        (other, false) => Error::FitsRead {
            path,
            source: seiza_fits::FitsError::Unsupported(other.to_string()),
        },
        (other, true) => Error::FitsWrite {
            path,
            source: seiza_fits::FitsError::Unsupported(other.to_string()),
        },
    }
}

fn logical_card(key: &str, value: bool, comment: &str) -> WriteHeaderCard {
    WriteHeaderCard::new(key, HeaderValue::Logical(value)).with_comment(comment)
}

fn integer_card(key: &str, value: i64, comment: &str) -> WriteHeaderCard {
    WriteHeaderCard::new(key, HeaderValue::Integer(value)).with_comment(comment)
}

fn float_card(key: &str, value: f64, comment: &str) -> WriteHeaderCard {
    WriteHeaderCard::new(key, HeaderValue::Float(value)).with_comment(comment)
}

fn string_card(key: &str, value: &str, comment: &str) -> WriteHeaderCard {
    WriteHeaderCard::new(key, HeaderValue::String(value.into())).with_comment(comment)
}

fn value_card(key: &str, value: &HeaderValue) -> Option<WriteHeaderCard> {
    match value {
        HeaderValue::Float(value) if !value.is_finite() => None,
        HeaderValue::Raw(value) if value.is_empty() => None,
        _ => Some(
            WriteHeaderCard::new(key, value.clone()).with_comment("copied from reference frame"),
        ),
    }
}

fn append_reference_wcs(
    cards: &mut Vec<WriteHeaderCard>,
    reference_headers: &[(String, HeaderValue)],
) {
    if has_output_wcs(cards) {
        return;
    }
    if !has_reference_wcs(reference_headers) {
        return;
    }
    for (key, value) in reference_headers {
        if preserve_wcs_key(key)
            && !cards.iter().any(|card| card.keyword() == key)
            && let Some(card) = value_card(key, value)
        {
            cards.push(card);
        }
    }
}

fn has_output_wcs(cards: &[WriteHeaderCard]) -> bool {
    ["CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CTYPE1", "CTYPE2"]
        .iter()
        .all(|required| cards.iter().any(|card| card.keyword() == *required))
}

/// Whether a reference card belongs to its WCS. WCS keywords are standard
/// eight-column ones, so a longer `HIERARCH` keyword that shares a prefix
/// such as `A_` is not one.
pub(crate) fn preserve_wcs_key(key: &str) -> bool {
    key.len() <= 8 && is_wcs_keyword(key)
}

fn is_wcs_keyword(key: &str) -> bool {
    matches!(
        key,
        "CRPIX1"
            | "CRPIX2"
            | "CRVAL1"
            | "CRVAL2"
            | "CTYPE1"
            | "CTYPE2"
            | "CUNIT1"
            | "CUNIT2"
            | "CDELT1"
            | "CDELT2"
            | "CROTA1"
            | "CROTA2"
            | "WCSAXES"
            | "RADESYS"
            | "EQUINOX"
            | "LONPOLE"
            | "LATPOLE"
            | "SKYORIEN"
    ) || key.starts_with("CD1_")
        || key.starts_with("CD2_")
        || key.starts_with("PC1_")
        || key.starts_with("PC2_")
        || key.starts_with("PV1_")
        || key.starts_with("PV2_")
        || key.starts_with("A_")
        || key.starts_with("B_")
        || key.starts_with("AP_")
        || key.starts_with("BP_")
}

fn has_reference_wcs(headers: &[(String, HeaderValue)]) -> bool {
    ["CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CTYPE1", "CTYPE2"]
        .iter()
        .all(|required| headers.iter().any(|(key, _)| key == required))
}

/// One canonical FITS key and its priority-ordered aliases per normalized
/// acquisition field. Long XISF aliases such as `TELESCOPE` and
/// `FOCALLENGTH` become legal eight-character FITS cards, and contradictory
/// aliases collapse to the same value `metadata_from_headers` selected.
const MASTER_METADATA_HEADER_GROUPS: &[(&str, &[&str])] = &[
    ("INSTRUME", CAMERA_HEADER_KEYS),
    ("TELESCOP", TELESCOPE_HEADER_KEYS),
    ("XBINNING", BINNING_X_HEADER_KEYS),
    ("YBINNING", BINNING_Y_HEADER_KEYS),
    ("XPIXSZ", &["XPIXSZ"]),
    ("YPIXSZ", &["YPIXSZ"]),
    ("GAIN", &["GAIN"]),
    ("EGAIN", &["EGAIN"]),
    ("OFFSET", &["OFFSET"]),
    ("CCD-TEMP", TEMPERATURE_HEADER_KEYS),
    ("READOUTM", READOUT_HEADER_KEYS),
    ("FILTER", &["FILTER"]),
    ("FOCALLEN", FOCAL_LENGTH_HEADER_KEYS),
    ("ROTATANG", ROTATION_HEADER_KEYS),
    ("EXPTIME", EXPOSURE_HEADER_KEYS),
    ("DATE-OBS", CAPTURE_TIME_HEADER_KEYS),
    ("BAYERPAT", &["BAYERPAT"]),
    ("XBAYROFF", &["XBAYROFF"]),
    ("YBAYROFF", &["YBAYROFF"]),
    ("ROWORDER", &["ROWORDER"]),
];

fn preserve_master_key(key: &str) -> bool {
    MASTER_METADATA_HEADER_GROUPS
        .iter()
        .any(|(_, aliases)| aliases.contains(&key))
}

fn preserve_processed_key(key: &str) -> bool {
    (key.len() <= 8 && preserve_master_key(key))
        || matches!(
            key,
            "OBJECT"
                | "OBSERVER"
                | "TELESCOP"
                | "DATE-OBS"
                | "DATE-BEG"
                | "DATE-END"
                | "DATE-AVG"
                | "MJD-OBS"
                | "TIMESYS"
                | "EXPTIME"
                | "XPOSURE"
                | "EXPOSURE"
                | "BUNIT"
                | "OBSGEO-X"
                | "OBSGEO-Y"
                | "OBSGEO-Z"
                | "OBSGEO-L"
                | "OBSGEO-B"
                | "OBSGEO-H"
                | "SITELAT"
                | "SITELONG"
                | "LAT-OBS"
                | "LONG-OBS"
                | "ALT-OBS"
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cards_of(cards: &[(&str, HeaderValue)]) -> Vec<(String, HeaderValue)> {
        cards
            .iter()
            .map(|(key, value)| (key.to_string(), value.clone()))
            .collect()
    }

    fn number(headers: &[(String, HeaderValue)], key: &str) -> Option<f64> {
        headers
            .iter()
            .find(|(name, _)| name == key)
            .and_then(|(_, value)| value.as_f64())
    }

    /// Binning a frame carries its WCS, SIP included, to the binned grid,
    /// and drizzling the result by the same factor carries it back: a
    /// round trip that leaves every card as it was.
    #[test]
    fn binned_wcs_drizzles_back_to_the_sensor_grid() {
        let solved = cards_of(&[
            ("CTYPE1", HeaderValue::String("RA---TAN-SIP".into())),
            ("CTYPE2", HeaderValue::String("DEC--TAN-SIP".into())),
            ("CRVAL1", HeaderValue::Float(10.68)),
            ("CRVAL2", HeaderValue::Float(41.27)),
            ("CRPIX1", HeaderValue::Float(3124.5)),
            ("CRPIX2", HeaderValue::Float(2088.25)),
            ("CD1_1", HeaderValue::Float(-1.95e-4)),
            ("CD1_2", HeaderValue::Float(2.0e-6)),
            ("CD2_1", HeaderValue::Float(-2.0e-6)),
            ("CD2_2", HeaderValue::Float(1.95e-4)),
            ("A_ORDER", HeaderValue::Integer(2)),
            ("A_2_0", HeaderValue::Float(3.1e-7)),
            ("A_0_2", HeaderValue::Float(-1.2e-7)),
            ("B_1_1", HeaderValue::Float(2.2e-7)),
            ("A_DMAX", HeaderValue::Float(1.8)),
        ]);
        for bin in [2_u32, 3, 4] {
            let scale = 1.0 / f64::from(bin);
            let binned = rescaled_reference_headers(&solved, scale);
            // CRPIX counts from 1, so a pixel centre X maps to (X + 0.5) / b
            // − 0.5, and the card to (CRPIX − 0.5) / b + 0.5.
            let crpix = number(&binned, "CRPIX1").unwrap();
            assert!((crpix - ((3124.5 - 0.5) / f64::from(bin) + 0.5)).abs() < 1e-12);
            assert!((number(&binned, "CD1_1").unwrap() + 1.95e-4 * f64::from(bin)).abs() < 1e-18);
            // An order-2 SIP term grows by the factor; the largest correction
            // shrinks by it.
            assert!((number(&binned, "A_2_0").unwrap() - 3.1e-7 * f64::from(bin)).abs() < 1e-20);
            assert!((number(&binned, "A_DMAX").unwrap() - 1.8 / f64::from(bin)).abs() < 1e-15);
            assert_eq!(number(&binned, "A_ORDER"), Some(2.0));
            let back = drizzle_reference_headers(&binned, bin);
            for ((key, original), (_, round)) in solved.iter().zip(&back) {
                match (original.as_f64(), round.as_f64()) {
                    (Some(original), Some(round)) => assert!(
                        (round - original).abs() <= 1e-12 * original.abs().max(1.0),
                        "{key}: {round} against {original}"
                    ),
                    _ => assert_eq!(original, round, "{key}"),
                }
            }
        }
    }

    #[test]
    fn a_binned_frame_says_so_in_its_headers() {
        let frame = FitsFrame {
            image: LinearImage::new(6, 4, 1, (0..24).map(|value| value as f32).collect()).unwrap(),
            headers: cards_of(&[
                ("NAXIS1", HeaderValue::Integer(6)),
                ("NAXIS2", HeaderValue::Integer(4)),
                ("CCDXBIN", HeaderValue::Integer(1)),
                ("YBINNING", HeaderValue::Integer(2)),
                ("XPIXSZ", HeaderValue::Float(3.76)),
                ("YPIXSZ", HeaderValue::Integer(4)),
                ("FOCALLEN", HeaderValue::Float(530.0)),
            ]),
            exposure_seconds: Some(60.0),
            bayer: None,
            source: None,
            bounds: None,
        };
        let binned = frame.clone().binned(2).unwrap();
        assert_eq!((binned.image.width, binned.image.height), (3, 2));
        for (key, expected) in [
            ("NAXIS1", 3.0),
            ("NAXIS2", 2.0),
            ("CCDXBIN", 2.0),
            ("YBINNING", 4.0),
            ("XPIXSZ", 7.52),
            ("YPIXSZ", 8.0),
            ("FOCALLEN", 530.0),
            ("SEIZABIN", 2.0),
        ] {
            assert_eq!(number(&binned.headers, key), Some(expected), "{key}");
        }
        assert_eq!(binned.metadata().signature.binning_x, Some(2));
        // Carried back to the sensor, the cards are the sensor's again.
        let native = rebinned_headers(&binned.headers, 2, 1);
        assert_eq!(number(&native, "CCDXBIN"), Some(1.0));
        assert_eq!(number(&native, "XPIXSZ"), Some(3.76));
        assert_eq!(number(&native, "SEIZABIN"), None);
        // Binning a binned stack's frame again records both.
        let twice = rebinned_headers(&binned.headers, 1, 2);
        assert_eq!(number(&twice, "SEIZABIN"), Some(4.0));
        // A factor of 1 leaves the frame alone, and a raw mosaic is refused.
        let same = frame.clone().binned(1).unwrap();
        assert_eq!(same.headers, frame.headers);
        let mosaic = FitsFrame {
            bayer: Some(BayerLayout {
                pattern: seiza_fits::BayerPattern::Rggb,
                x_offset: 0,
                y_offset: 0,
            }),
            ..frame
        };
        assert!(mosaic.binned(2).is_err());
    }

    /// A PixInsight-style XISF keeps its solution only as properties;
    /// opening it yields the equivalent FITS WCS cards.
    #[test]
    fn an_xisf_solution_without_wcs_cards_becomes_wcs_cards() {
        let hex = |values: &[f64]| {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        };
        let property = |id: &str, kind: &str, dimensions: &str, values: &[f64]| {
            format!(
                "<Property id=\"AstrometricSolution:{id}\" type=\"{kind}\" {dimensions} location=\"inline:hex\">{}</Property>",
                hex(values)
            )
        };
        let pixels = [1.0_f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let header = |offset: usize| {
            format!(
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
                 <xisf version=\"1.0\" xmlns=\"http://www.pixinsight.com/xisf\">\
                 <Image geometry=\"2:2:1\" sampleFormat=\"Float32\" bounds=\"0:1\" colorSpace=\"Gray\" location=\"attachment:{offset}:16\">\
                 <FITSKeyword name=\"A_ORDER\" value=\"2\" comment=\"\"/>\
                 <Property id=\"AstrometricSolution:Version\" type=\"String\">1.0</Property>\
                 <Property id=\"AstrometricSolution:ProjectionSystem\" type=\"String\">Gnomonic</Property>\
                 {}{}{}\
                 </Image></xisf>",
                property(
                    "ReferenceCelestialCoordinates",
                    "F64Vector",
                    "length=\"2\"",
                    &[56.7, 24.1]
                ),
                property(
                    "ReferenceImageCoordinates",
                    "F64Vector",
                    "length=\"2\"",
                    &[1.0, 0.75]
                ),
                property(
                    "LinearTransformationMatrix",
                    "F64Matrix",
                    "rows=\"2\" columns=\"2\"",
                    &[3.8e-5, 1.24e-3, -1.24e-3, 3.8e-5]
                ),
            )
        };
        // The data offset is part of the header, so settle it first.
        let mut offset = 4096;
        let mut xml = header(offset);
        while 16 + xml.len() > offset {
            offset *= 2;
            xml = header(offset);
        }
        let mut bytes = b"XISF0100".to_vec();
        bytes.extend((xml.len() as u32).to_le_bytes());
        bytes.extend([0_u8; 4]);
        bytes.extend(xml.as_bytes());
        bytes.resize(offset, 0);
        bytes.extend(&pixels);
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("solved.xisf");
        std::fs::write(&path, bytes).unwrap();

        let frame = FitsFrame::open(&path).unwrap();
        let value = |key: &str| {
            frame
                .headers
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.clone())
        };
        assert_eq!(
            value("CTYPE1").and_then(|v| v.as_str().map(str::to_owned)),
            Some("RA---TAN".to_owned())
        );
        assert_eq!(value("CRPIX1").and_then(|v| v.as_f64()), Some(1.5));
        assert_eq!(value("CRPIX2").and_then(|v| v.as_f64()), Some(1.25));
        assert_eq!(value("CRVAL1").and_then(|v| v.as_f64()), Some(56.7));
        assert_eq!(value("CD1_2").and_then(|v| v.as_f64()), Some(1.24e-3));
        assert_eq!(value("CD2_1").and_then(|v| v.as_f64()), Some(-1.24e-3));
        // Rows stay in stored order, which the solution describes.
        assert_eq!(frame.image.data, vec![1.0, 2.0, 3.0, 4.0]);
        let wcs = seiza::Wcs::from_fits_values(
            |key| value(key).and_then(|v| v.as_f64()),
            |key| value(key).and_then(|v| v.as_str().map(str::to_owned)),
        )
        .unwrap();
        // The reference image point (1.0, 0.75) is pixel (0.5, 0.25) in
        // zero-based pixel-centre coordinates.
        let (ra, dec) = wcs.pixel_to_world(0.5, 0.25);
        assert!((ra - 56.7).abs() < 1e-9 && (dec - 24.1).abs() < 1e-9);
    }

    #[test]
    fn drizzled_headers_describe_the_same_sky_on_the_finer_grid() {
        let headers = vec![
            ("CRPIX1".to_owned(), HeaderValue::Float(100.5)),
            ("CRPIX2".to_owned(), HeaderValue::Float(20.0)),
            ("CD1_1".to_owned(), HeaderValue::Float(-2.0e-4)),
            ("CDELT2".to_owned(), HeaderValue::Float(1.0e-4)),
            ("A_ORDER".to_owned(), HeaderValue::Integer(2)),
            ("A_DMAX".to_owned(), HeaderValue::Float(1.5)),
            ("A_2_0".to_owned(), HeaderValue::Float(4.0e-6)),
            ("BP_0_1".to_owned(), HeaderValue::Float(1.0e-3)),
            (
                "CTYPE1".to_owned(),
                HeaderValue::String("RA---TAN-SIP".into()),
            ),
        ];
        let scaled = drizzle_reference_headers(&headers, 2);
        let value = |key: &str| {
            scaled
                .iter()
                .find(|(name, _)| name == key)
                .and_then(|(_, value)| value.as_f64())
        };
        // FITS pixel edges fall at half-integers: the reference's edge at
        // 100.5 is the finer grid's edge at 200.5.
        assert_eq!(value("CRPIX1"), Some(200.5));
        assert_eq!(value("CRPIX2"), Some(39.5));
        assert_eq!(value("CD1_1"), Some(-1.0e-4));
        assert_eq!(value("CDELT2"), Some(5.0e-5));
        assert_eq!(value("A_2_0"), Some(2.0e-6));
        assert_eq!(value("BP_0_1"), Some(1.0e-3));
        assert_eq!(value("A_DMAX"), Some(3.0));
        assert_eq!(scaled[4], headers[4]);
        assert_eq!(scaled[8], headers[8]);
    }
    use seiza_fits::Pixels;

    fn headers(bitpix: i64) -> Vec<(String, HeaderValue)> {
        vec![("BITPIX".into(), HeaderValue::Integer(bitpix))]
    }

    fn tiny_xisf(values: &[f32]) -> Vec<u8> {
        let raw = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let template = format!(
            "<?xml version=\"1.0\"?><xisf version=\"1.0\"><Image geometry=\"2:2:1\" sampleFormat=\"Float32\" colorSpace=\"Gray\" location=\"attachment:0000000000:{}\"><FITSKeyword name=\"EXPTIME\" value=\"30\"/></Image></xisf>",
            raw.len()
        );
        let offset = 16 + template.len();
        let header = template.replacen("0000000000", &format!("{offset:010}"), 1);
        let mut bytes = Vec::with_capacity(offset + raw.len());
        bytes.extend_from_slice(b"XISF0100");
        bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&[0; 4]);
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&raw);
        bytes
    }

    #[test]
    fn planar_color_becomes_interleaved() {
        assert_eq!(
            planar_to_interleaved(&[1.0, 2.0, 10.0, 20.0, 100.0, 200.0], 2),
            [1.0, 10.0, 100.0, 2.0, 20.0, 200.0]
        );
    }

    #[test]
    fn bottom_up_cfa_is_resolved_before_preparation_and_survives_processed_output() {
        let directory = tempfile::tempdir().unwrap();
        for height in [3, 4] {
            // Sensor RGGB with both offsets set: BG/GR from the top.
            let data = (0..height)
                .flat_map(|stored_y| {
                    let top_y = height - 1 - stored_y;
                    if top_y % 2 == 0 {
                        [3000, 2000, 3000, 2000]
                    } else {
                        [2000, 1000, 2000, 1000]
                    }
                })
                .collect::<Vec<_>>();
            let fits = FitsImage {
                width: 4,
                height,
                planes: 1,
                pixels: Pixels::U16(data.clone()),
                headers: vec![
                    ("NAXIS1".into(), HeaderValue::Integer(4)),
                    ("NAXIS2".into(), HeaderValue::Integer(height as i64)),
                    ("BAYERPAT".into(), HeaderValue::String("RGGB".into())),
                    ("ROWORDER".into(), HeaderValue::String("BOTTOM-UP".into())),
                    ("XBAYROFF".into(), HeaderValue::Integer(1)),
                    ("YBAYROFF".into(), HeaderValue::Integer(1)),
                ],
            };
            assert!(
                fits.debayer()
                    .unwrap()
                    .data
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .all(|pixel| *pixel == [1000, 2000, 3000])
            );
            let frame = FitsFrame::from_fits(fits, None).unwrap();
            let expected = if height == 4 {
                BayerPattern::Gbrg
            } else {
                BayerPattern::Rggb
            };
            assert_eq!(frame.bayer.unwrap().pattern, expected);
            assert_eq!(
                frame.metadata().signature.bayer_pattern.as_deref(),
                Some(expected.as_str())
            );
            assert_eq!(
                FrameMetadata::from_headers(&frame.headers).signature,
                frame.metadata().signature
            );
            assert_eq!(
                frame.image.data,
                data.iter().map(|&v| f32::from(v)).collect::<Vec<_>>()
            );
            for extension in ["fits", "xisf"] {
                let path = directory
                    .path()
                    .join(format!("processed-{height}.{extension}"));
                write_processed_image_fits_f32(&path, &frame.image, &frame.headers, &[]).unwrap();
                let restored = FitsFrame::open(&path).unwrap();
                assert_eq!(restored.bayer, frame.bayer);
                assert_eq!(restored.image.data, frame.image.data);
                let prepared = restored.into_prepared().unwrap();
                assert!(
                    prepared
                        .image
                        .data
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .all(|pixel| *pixel == [1000.0, 2000.0, 3000.0])
                );
            }
        }
    }

    #[test]
    fn raw_bayer_headers_keep_one_channel_when_a_restored_image_is_prepared_rgb() {
        let prepared = LinearImage::new(2, 2, 3, vec![0.0; 12]).unwrap();
        let headers = vec![
            ("NAXIS1".into(), HeaderValue::Integer(2)),
            ("NAXIS2".into(), HeaderValue::Integer(2)),
            ("BAYERPAT".into(), HeaderValue::String("RGGB".into())),
            ("IMAGETYP".into(), HeaderValue::String("LIGHT".into())),
        ];
        let metadata = FrameMetadata::from_image(&prepared, &headers);
        assert_eq!(metadata.signature.channels, Some(1));
        assert_eq!(metadata.signature.bayer_pattern.as_deref(), Some("RGGB"));
        assert_eq!(metadata.role, FrameSourceRole::Light);
    }

    #[test]
    fn converts_native_pixel_types_without_display_normalization() {
        let cases = [
            (Pixels::U8(vec![2]), 8, 2.0),
            (Pixels::U16(vec![200]), 16, 200.0),
            (Pixels::I32(vec![-3]), 32, -3.0),
            (Pixels::F32(vec![0.125]), -32, 0.125),
            (Pixels::F64(vec![4.5]), -64, 4.5),
        ];
        for (pixels, bitpix, expected) in cases {
            let frame = FitsFrame::from_fits(
                FitsImage {
                    width: 1,
                    height: 1,
                    planes: 1,
                    pixels,
                    headers: headers(bitpix),
                },
                None,
            )
            .unwrap();
            assert_eq!(frame.image.data, [expected]);
        }
    }

    #[test]
    fn path_loader_accepts_xisf_as_linear_input() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("light.xisf");
        std::fs::write(&path, tiny_xisf(&[0.0, 0.25, 0.5, 1.0])).unwrap();

        let frame = FitsFrame::open(&path).unwrap();
        assert_eq!((frame.image.width, frame.image.height), (2, 2));
        assert_eq!(frame.image.data, [0.0, 0.25, 0.5, 1.0]);
        assert_eq!(frame.exposure_seconds, Some(30.0));
        // Carried through, never applied: the samples are what the file held.
        assert_eq!(frame.bounds, None);
    }

    /// A frame this crate wrote declares its observed range, and reading it
    /// back must not change a sample.
    #[test]
    fn a_declared_range_rides_along_without_touching_the_samples() {
        let physical = [100.0_f32, 200.0, 5000.0, 30000.0];
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("written.xisf");
        write_processed_image_fits_f32(
            &path,
            &LinearImage::new(2, 2, 1, physical.to_vec()).unwrap(),
            &[],
            &[],
        )
        .unwrap();

        // Stacking declares the 16-bit camera range, and the samples are
        // stored as they are.
        let frame = FitsFrame::open(&path).unwrap();
        assert_eq!(frame.bounds, Some((0.0, 65535.0)));
        assert_eq!(frame.image.data, physical);

        let header_path = directory.path().join("written.xish");
        assert!(
            write_processed_image_fits_f32(
                &header_path,
                &LinearImage::new(2, 2, 1, physical.to_vec()).unwrap(),
                &[],
                &[],
            )
            .is_err()
        );
    }

    #[test]
    fn applies_nonstandard_fits_scaling_once() {
        let mut scaled_headers = headers(8);
        scaled_headers.push(("BZERO".into(), HeaderValue::Float(10.0)));
        scaled_headers.push(("BSCALE".into(), HeaderValue::Float(2.0)));
        let frame = FitsFrame::from_fits(
            FitsImage {
                width: 1,
                height: 1,
                planes: 1,
                pixels: Pixels::U8(vec![3]),
                headers: scaled_headers,
            },
            None,
        )
        .unwrap();
        assert_eq!(frame.image.data, [16.0]);

        // seiza-fits produces F32 for unusual BITPIX=16 scaling after it
        // has already applied BSCALE/BZERO.
        let mut decoded_headers = headers(16);
        decoded_headers.push(("BZERO".into(), HeaderValue::Float(10.0)));
        decoded_headers.push(("BSCALE".into(), HeaderValue::Float(2.0)));
        let frame = FitsFrame::from_fits(
            FitsImage {
                width: 1,
                height: 1,
                planes: 1,
                pixels: Pixels::F32(vec![16.0]),
                headers: decoded_headers,
            },
            None,
        )
        .unwrap();
        assert_eq!(frame.image.data, [16.0]);
    }

    #[test]
    fn float_writer_round_trips_linear_samples_and_stack_counts() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("stack.fits");
        std::fs::write(&path, b"previous complete output").unwrap();
        let image = LinearImage::new(2, 2, 1, vec![-2.5, 0.25, 100.0, f32::NAN]).unwrap();
        let snapshot = StackSnapshot {
            variance: LinearImage::new(2, 2, 1, vec![0.0; 4]).unwrap(),
            coverage: vec![3; 4],
            rejected_samples: vec![0; 4],
            image,
            accepted_frames: 3,
            rejected_frames: 1,
        };
        write_fits_f32(&path, &snapshot, &[]).unwrap();
        let decoded = FitsImage::open(&path).unwrap();
        let Pixels::F32(ref values) = decoded.pixels else {
            panic!("writer must emit BITPIX=-32");
        };
        assert_eq!(values[..3], [-2.5, 0.25, 100.0]);
        assert!(values[3].is_nan());
        assert_eq!(decoded.header_f64("STACKCNT"), Some(3.0));
        assert_eq!(decoded.header_f64("STACKREJ"), Some(1.0));
    }

    #[test]
    fn xisf_output_path_round_trips_rgb_and_metadata() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("processed.xisf");
        let interleaved = vec![1.0, 10.0, 100.0, 2.0, 20.0, 200.0];
        let image = LinearImage::new(2, 1, 3, interleaved.clone()).unwrap();
        let reference_headers = vec![
            ("OBJECT".into(), HeaderValue::String("Sh2-132".into())),
            ("EXPTIME".into(), HeaderValue::Float(30.0)),
        ];
        write_processed_image_fits_f32(&path, &image, &reference_headers, &[]).unwrap();

        assert!(seiza_xisf::inspect(&path).is_ok(), "output must be XISF");
        let frame = FitsFrame::open(&path).unwrap();
        assert_eq!((frame.image.width, frame.image.height), (2, 1));
        assert_eq!(frame.image.channels, 3);
        assert_eq!(frame.image.data, interleaved);
        assert_eq!(frame.exposure_seconds, Some(30.0));
    }

    #[test]
    fn processed_writer_preserves_source_metadata_without_structural_cards() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("processed.fits");
        let image = LinearImage::new(2, 2, 1, vec![1.0; 4]).unwrap();
        let reference_headers = vec![
            ("BITPIX".into(), HeaderValue::Integer(16)),
            (
                "DATE-OBS".into(),
                HeaderValue::String("2026-01-02T03:04:05Z".into()),
            ),
            ("FILTER".into(), HeaderValue::String("H-alpha".into())),
            ("OBJECT".into(), HeaderValue::String("Sh2-132".into())),
            ("CRPIX1".into(), HeaderValue::Float(1.5)),
            ("CRPIX2".into(), HeaderValue::Float(1.5)),
            ("CRVAL1".into(), HeaderValue::Float(120.0)),
            ("CRVAL2".into(), HeaderValue::Float(30.0)),
            ("CTYPE1".into(), HeaderValue::String("RA---TAN".into())),
            ("CTYPE2".into(), HeaderValue::String("DEC--TAN".into())),
            ("SKYORIEN".into(), HeaderValue::String("N-UP E-LEFT".into())),
        ];
        write_processed_image_fits_f32(&path, &image, &reference_headers, &[]).unwrap();
        let decoded = FitsImage::open(&path).unwrap();
        assert_eq!(decoded.header_f64("BITPIX"), Some(-32.0));
        assert_eq!(decoded.header_str("DATE-OBS"), Some("2026-01-02T03:04:05Z"));
        assert_eq!(decoded.header_str("FILTER"), Some("H-alpha"));
        assert_eq!(decoded.header_str("OBJECT"), Some("Sh2-132"));
        assert_eq!(decoded.header_str("SKYORIEN"), Some("N-UP E-LEFT"));
    }

    #[test]
    fn cropped_color_output_moves_crpix_to_its_own_grid() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("cropped.fits");
        let composition = ColorComposition {
            image: LinearImage::new(2, 2, 3, vec![0.5; 12]).unwrap(),
            transfer: crate::ColorTransfer::LinearLight,
            region: crate::ReferenceRegion {
                x: 3,
                y: 5,
                width: 2,
                height: 2,
            },
            crop: None,
        };
        let reference_headers = vec![
            ("CRPIX1".into(), HeaderValue::Float(20.5)),
            ("CRPIX2".into(), HeaderValue::Float(30.5)),
            ("CRVAL1".into(), HeaderValue::Float(120.0)),
            ("CRVAL2".into(), HeaderValue::Float(30.0)),
            ("CTYPE1".into(), HeaderValue::String("RA---TAN".into())),
            ("CTYPE2".into(), HeaderValue::String("DEC--TAN".into())),
        ];
        write_color_fits_f32(&path, &composition, &reference_headers, "SHO").unwrap();
        let decoded = FitsImage::open(&path).unwrap();
        assert_eq!(decoded.header_f64("CRPIX1"), Some(17.5));
        assert_eq!(decoded.header_f64("CRPIX2"), Some(25.5));
        assert_eq!(decoded.header_f64("CRVAL1"), Some(120.0));
        assert_eq!(decoded.header_f64("SEIZACRX"), Some(3.0));
        assert_eq!(decoded.header_f64("SEIZACRY"), Some(5.0));
    }

    #[test]
    fn uncropped_color_output_keeps_the_reference_crpix() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("full.fits");
        let composition = ColorComposition {
            image: LinearImage::new(2, 2, 3, vec![0.5; 12]).unwrap(),
            transfer: crate::ColorTransfer::LinearLight,
            region: crate::ReferenceRegion {
                x: 0,
                y: 0,
                width: 2,
                height: 2,
            },
            crop: None,
        };
        let reference_headers = vec![
            ("CRPIX1".into(), HeaderValue::Float(20.5)),
            ("CRPIX2".into(), HeaderValue::Float(30.5)),
            ("CRVAL1".into(), HeaderValue::Float(120.0)),
            ("CRVAL2".into(), HeaderValue::Float(30.0)),
            ("CTYPE1".into(), HeaderValue::String("RA---TAN".into())),
            ("CTYPE2".into(), HeaderValue::String("DEC--TAN".into())),
        ];
        write_color_fits_f32(&path, &composition, &reference_headers, "SHO").unwrap();
        let decoded = FitsImage::open(&path).unwrap();
        assert_eq!(decoded.header_f64("CRPIX1"), Some(20.5));
        assert_eq!(decoded.header_f64("CRPIX2"), Some(30.5));
        assert!(decoded.header("SEIZACRX").is_none());
    }

    #[test]
    fn replacement_wcs_drops_conflicting_reference_matrix_forms() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("oriented.fits");
        let image = LinearImage::new(4, 3, 1, vec![1.0; 12]).unwrap();
        let reference_headers = vec![
            ("CRPIX1".into(), HeaderValue::Float(2.5)),
            ("CRPIX2".into(), HeaderValue::Float(2.0)),
            ("CRVAL1".into(), HeaderValue::Float(120.0)),
            ("CRVAL2".into(), HeaderValue::Float(30.0)),
            ("CTYPE1".into(), HeaderValue::String("RA---TAN".into())),
            ("CTYPE2".into(), HeaderValue::String("DEC--TAN".into())),
            ("CDELT1".into(), HeaderValue::Float(-1.0 / 3600.0)),
            ("CDELT2".into(), HeaderValue::Float(1.0 / 3600.0)),
            ("CROTA2".into(), HeaderValue::Float(37.0)),
        ];
        let wcs =
            seiza::Wcs::from_center_scale_rotation((120.0, 30.0), (1.5, 1.0), 1.0, 37.0, false);
        let plan = crate::SkyOrientationPlan::new(4, 3, &wcs).unwrap();
        write_processed_image_fits_f32(
            &path,
            &plan.apply(&image).unwrap(),
            &reference_headers,
            &plan.fits_header_cards(),
        )
        .unwrap();
        let decoded = FitsImage::open(&path).unwrap();

        assert_eq!(decoded.header_str("SKYORIEN"), Some("N-UP E-LEFT"));
        assert!(decoded.header("CD1_1").is_some());
        assert!(decoded.header("CD2_2").is_some());
        assert!(decoded.header("CDELT1").is_none());
        assert!(decoded.header("CDELT2").is_none());
        assert!(decoded.header("CROTA2").is_none());
    }

    #[test]
    fn hierarch_reference_cards_are_not_mistaken_for_wcs() {
        let directory = tempfile::tempdir().unwrap();
        let image = LinearImage::new(2, 2, 1, vec![1.0; 4]).unwrap();
        let reference_headers = vec![
            ("CRPIX1".into(), HeaderValue::Float(1.0)),
            ("CRPIX2".into(), HeaderValue::Float(1.0)),
            ("CRVAL1".into(), HeaderValue::Float(120.0)),
            ("CRVAL2".into(), HeaderValue::Float(30.0)),
            ("CTYPE1".into(), HeaderValue::String("RA---TAN".into())),
            ("CTYPE2".into(), HeaderValue::String("DEC--TAN".into())),
            ("A_ORDER".into(), HeaderValue::Integer(2)),
            ("A_ESO DET CHIP".into(), HeaderValue::Integer(3)),
        ];
        // XISF keeps FITS keywords to eight characters, so a copied
        // HIERARCH card would fail the whole write.
        for name in ["processed.fits", "processed.xisf"] {
            let path = directory.path().join(name);
            write_processed_image_fits_f32(&path, &image, &reference_headers, &[]).unwrap();
        }
        let decoded = FitsImage::open(&directory.path().join("processed.fits")).unwrap();
        assert_eq!(decoded.header("A_ORDER"), Some(&HeaderValue::Integer(2)));
        assert!(decoded.header("A_ESO DET CHIP").is_none());
    }

    #[test]
    fn processed_rgb_writer_drops_stale_bayer_metadata() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("processed-rgb.fits");
        let image = LinearImage::new(2, 2, 3, vec![1.0; 12]).unwrap();
        let reference_headers = vec![
            ("BAYERPAT".into(), HeaderValue::String("RGGB".into())),
            ("XBAYROFF".into(), HeaderValue::Integer(0)),
            ("YBAYROFF".into(), HeaderValue::Integer(0)),
            ("ROWORDER".into(), HeaderValue::String("BOTTOM-UP".into())),
        ];
        write_processed_image_fits_f32(&path, &image, &reference_headers, &[]).unwrap();
        let decoded = FitsImage::open(&path).unwrap();
        assert_eq!(decoded.planes, 3);
        assert!(decoded.header("BAYERPAT").is_none());
        assert!(decoded.header("XBAYROFF").is_none());
        assert!(decoded.header("YBAYROFF").is_none());
        // Row direction also describes RGB/mono display orientation.
        assert_eq!(decoded.row_order(), Some(RowOrder::BottomUp));
    }

    #[test]
    fn master_writer_round_trips_state_and_canonicalizes_metadata_aliases() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("master-dark.fits");
        let master = MasterFrame {
            kind: crate::MasterFrameKind::Dark,
            image: LinearImage::new(2, 2, 1, vec![4.0; 4]).unwrap(),
            exposure_seconds: Some(30.0),
            bayer: Some(BayerLayout {
                pattern: seiza_fits::BayerPattern::Rggb,
                x_offset: 1,
                y_offset: 0,
            }),
            input_frames: 12,
            accepted_samples: 47,
            rejected_samples: 1,
            masked_samples: 0,
            flat_star_masking: None,
            fallback_pixels: 0,
            defect_pixels_replaced: 0,
            input_statistics: Vec::new(),
            bias_subtracted: true,
            dark_subtracted: false,
            normalized: false,
            rejection: crate::MasterRejectionOptions::default(),
            rejection_method: crate::MasterRejectionMethod::LeaveOneOut,
            reference_headers: vec![
                ("ROWORDER".into(), HeaderValue::String("BOTTOM-UP".into())),
                ("BAYERPAT".into(), HeaderValue::String("GBRG".into())),
                ("INSTRUME".into(), HeaderValue::String("Test Camera".into())),
                ("CAMERA".into(), HeaderValue::String("Ignored Alias".into())),
                ("TELESCOPE".into(), HeaderValue::String("Test Scope".into())),
                ("FOCALLENGTH".into(), HeaderValue::Float(400.0)),
                ("ROTPOS".into(), HeaderValue::Float(45.0)),
                (
                    "DATE-BEG".into(),
                    HeaderValue::String("2026-08-20T04:00:00Z".into()),
                ),
                ("EXPOSURE".into(), HeaderValue::Float(999.0)),
            ],
            skipped_inputs: Vec::new(),
        };
        write_master_fits_f32(&path, &master).unwrap();
        let decoded = FitsImage::open(&path).unwrap();
        assert_eq!(decoded.header_str("SEIZAMST"), Some("DARK"));
        assert_eq!(decoded.row_order(), Some(RowOrder::BottomUp));
        assert_eq!(decoded.bayer_pattern(), Some(BayerPattern::Gbrg));
        assert_eq!(
            decoded.bayer_pattern_in_storage_order(),
            Some(BayerPattern::Rggb)
        );
        assert_eq!(decoded.header_str("REJMETH"), Some("LEAVE_ONE_OUT"));
        assert_eq!(decoded.header_f64("NCOMBINE"), Some(12.0));
        assert_eq!(decoded.header_f64("EXPTIME"), Some(30.0));
        assert_eq!(
            decoded.header("BIASSUB").and_then(HeaderValue::as_bool),
            Some(true)
        );
        assert_eq!(decoded.header_str("INSTRUME"), Some("Test Camera"));
        assert_eq!(decoded.header_str("TELESCOP"), Some("Test Scope"));
        assert_eq!(decoded.header_f64("FOCALLEN"), Some(400.0));
        assert_eq!(decoded.header_f64("ROTATANG"), Some(45.0));
        assert_eq!(decoded.header_str("DATE-OBS"), Some("2026-08-20T04:00:00Z"));
        for alias in [
            "CAMERA",
            "TELESCOPE",
            "FOCALLENGTH",
            "ROTPOS",
            "DATE-BEG",
            "EXPOSURE",
        ] {
            assert!(
                decoded.header(alias).is_none(),
                "the master must not duplicate the canonical field as {alias}"
            );
        }
        let frame = FitsFrame::open(&path).unwrap();
        assert_eq!(frame.bayer, master.bayer);
        frame.validate_master_kind("DARK").unwrap();
        assert!(frame.validate_master_kind("BIAS").is_err());
    }
}
