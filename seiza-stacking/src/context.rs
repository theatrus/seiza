use crate::{BayerLayout, CalibrationMasters, Error, LinearImage, Result, StackOptions};
use seiza_fits::{BayerPattern, HeaderValue};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"SEIZASTK";
const FORMAT_VERSION: u32 = 1;
const MAXIMUM_METADATA_BYTES: u64 = 8 * 1024 * 1024;
const IO_BUFFER_VALUES: usize = 16 * 1024;

pub(crate) struct ContextWriteState<'a> {
    pub options: &'a StackOptions,
    pub calibration: &'a CalibrationMasters,
    pub reference: &'a LinearImage,
    pub reference_headers: &'a [(String, HeaderValue)],
    pub mean: &'a [f32],
    pub m2: &'a [f32],
    pub count: &'a [u32],
    pub rejected: &'a [u32],
    pub accepted_frames: u32,
    pub rejected_frames: u32,
    pub input_paths: &'a [PathBuf],
}

pub(crate) struct RestoredContext {
    pub options: StackOptions,
    pub calibration: CalibrationMasters,
    pub reference: LinearImage,
    pub reference_headers: Vec<(String, HeaderValue)>,
    pub mean: Vec<f32>,
    pub m2: Vec<f32>,
    pub count: Vec<u32>,
    pub rejected: Vec<u32>,
    pub accepted_frames: u32,
    pub rejected_frames: u32,
    pub input_paths: Vec<PathBuf>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ContextMetadata {
    schema_version: u32,
    options: StackOptions,
    calibration: CalibrationMetadata,
    reference: ImageMetadata,
    reference_headers: Vec<HeaderCardMetadata>,
    accepted_frames: u32,
    rejected_frames: u32,
    input_paths: Vec<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationMetadata {
    bias: Option<ImageMetadata>,
    dark_signal: Option<ImageMetadata>,
    dark_exposure_seconds: Option<f64>,
    dark_bayer: Option<BayerMetadata>,
    flat_response: Option<ImageMetadata>,
    flat_bayer: Option<BayerMetadata>,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ImageMetadata {
    width: u64,
    height: u64,
    channels: u8,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BayerMetadata {
    pattern: String,
    x_offset: u64,
    y_offset: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HeaderCardMetadata {
    name: String,
    value: HeaderValueMetadata,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum HeaderValueMetadata {
    Logical(bool),
    Integer(i64),
    FloatBits(u64),
    String(String),
    Raw(String),
}

pub(crate) fn write(path: &Path, state: ContextWriteState<'_>) -> Result<()> {
    validate_live_arrays(
        state.reference.sample_count(),
        state.mean,
        state.m2,
        state.count,
        state.rejected,
        state.accepted_frames,
    )
    .map_err(|message| context_write_error(path, message))?;
    state
        .options
        .validate()
        .map_err(|error| context_write_error(path, error.to_string()))?;
    validate_calibration(state.reference, state.calibration)
        .map_err(|message| context_write_error(path, message))?;

    let metadata = ContextMetadata::from_state(&state)
        .map_err(|message| context_write_error(path, message))?;
    let metadata = serde_json::to_vec(&metadata)
        .map_err(|error| context_write_error(path, error.to_string()))?;
    if metadata.len() as u64 > MAXIMUM_METADATA_BYTES {
        return Err(context_write_error(path, "context metadata is too large"));
    }

    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let prefix = format!(
        ".{}.",
        path.file_name().unwrap_or_default().to_string_lossy()
    );
    let mut builder = tempfile::Builder::new();
    builder.prefix(&prefix);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let permissions = std::fs::metadata(path)
            .map(|metadata| metadata.permissions())
            .unwrap_or_else(|_| std::fs::Permissions::from_mode(0o600));
        builder.permissions(permissions);
    }
    let mut temporary = builder
        .tempfile_in(parent)
        .map_err(|error| context_write_error(path, error.to_string()))?;

    {
        let mut writer = BufWriter::new(temporary.as_file_mut());
        writer
            .write_all(MAGIC)
            .and_then(|()| writer.write_all(&FORMAT_VERSION.to_le_bytes()))
            .map_err(|error| context_write_error(path, error.to_string()))?;
        let mut encoder = zstd::stream::write::Encoder::new(writer, 1)
            .map_err(|error| context_write_error(path, error.to_string()))?;
        encoder
            .include_checksum(true)
            .map_err(|error| context_write_error(path, error.to_string()))?;
        encoder
            .write_all(&(metadata.len() as u64).to_le_bytes())
            .and_then(|()| encoder.write_all(&metadata))
            .map_err(|error| context_write_error(path, error.to_string()))?;

        write_f32_values(&mut encoder, &state.reference.data)
            .and_then(|()| write_f32_values(&mut encoder, state.mean))
            .and_then(|()| write_f32_values(&mut encoder, state.m2))
            .and_then(|()| write_u32_values(&mut encoder, state.count))
            .and_then(|()| write_u32_values(&mut encoder, state.rejected))
            .and_then(|()| write_optional_image(&mut encoder, state.calibration.bias.as_ref()))
            .and_then(|()| {
                write_optional_image(&mut encoder, state.calibration.dark_signal.as_ref())
            })
            .and_then(|()| {
                write_optional_image(&mut encoder, state.calibration.flat_response.as_ref())
            })
            .map_err(|error| context_write_error(path, error.to_string()))?;

        let mut writer = encoder
            .finish()
            .map_err(|error| context_write_error(path, error.to_string()))?;
        writer
            .flush()
            .map_err(|error| context_write_error(path, error.to_string()))?;
    }
    temporary
        .as_file()
        .sync_all()
        .map_err(|error| context_write_error(path, error.to_string()))?;
    temporary
        .persist(path)
        .map_err(|error| context_write_error(path, error.error.to_string()))?;
    Ok(())
}

pub(crate) fn read(path: &Path) -> Result<RestoredContext> {
    let file = File::open(path).map_err(|error| context_read_error(path, error.to_string()))?;
    let mut reader = BufReader::new(file);
    let mut magic = [0_u8; 8];
    reader
        .read_exact(&mut magic)
        .map_err(|error| context_read_error(path, error.to_string()))?;
    if &magic != MAGIC {
        return Err(context_read_error(path, "not a Seiza live-stack context"));
    }
    let version = read_u32(&mut reader).map_err(|error| context_read_error(path, error))?;
    if version != FORMAT_VERSION {
        return Err(context_read_error(
            path,
            format!("unsupported context format version {version}"),
        ));
    }

    let mut decoder = zstd::stream::read::Decoder::new(reader)
        .map_err(|error| context_read_error(path, error.to_string()))?;
    let metadata_length =
        read_u64(&mut decoder).map_err(|error| context_read_error(path, error))?;
    if metadata_length > MAXIMUM_METADATA_BYTES {
        return Err(context_read_error(path, "context metadata is too large"));
    }
    let metadata_length = usize::try_from(metadata_length)
        .map_err(|_| context_read_error(path, "context metadata length overflows this platform"))?;
    let mut metadata_bytes = vec![0_u8; metadata_length];
    decoder
        .read_exact(&mut metadata_bytes)
        .map_err(|error| context_read_error(path, error.to_string()))?;
    let metadata: ContextMetadata = serde_json::from_slice(&metadata_bytes)
        .map_err(|error| context_read_error(path, error.to_string()))?;
    metadata
        .validate()
        .map_err(|message| context_read_error(path, message))?;

    let reference = read_image(&mut decoder, metadata.reference)
        .map_err(|error| context_read_error(path, error))?;
    let sample_count = reference.sample_count();
    let mean = read_f32_values(&mut decoder, sample_count)
        .map_err(|error| context_read_error(path, error))?;
    let m2 = read_f32_values(&mut decoder, sample_count)
        .map_err(|error| context_read_error(path, error))?;
    let count = read_u32_values(&mut decoder, sample_count)
        .map_err(|error| context_read_error(path, error))?;
    let rejected = read_u32_values(&mut decoder, sample_count)
        .map_err(|error| context_read_error(path, error))?;
    let bias = read_optional_image(&mut decoder, metadata.calibration.bias)
        .map_err(|error| context_read_error(path, error))?;
    let dark_signal = read_optional_image(&mut decoder, metadata.calibration.dark_signal)
        .map_err(|error| context_read_error(path, error))?;
    let flat_response = read_optional_image(&mut decoder, metadata.calibration.flat_response)
        .map_err(|error| context_read_error(path, error))?;

    let mut trailing = [0_u8; 1];
    if decoder
        .read(&mut trailing)
        .map_err(|error| context_read_error(path, error.to_string()))?
        != 0
    {
        return Err(context_read_error(
            path,
            "context has trailing payload data",
        ));
    }
    validate_live_arrays(
        sample_count,
        &mean,
        &m2,
        &count,
        &rejected,
        metadata.accepted_frames,
    )
    .map_err(|message| context_read_error(path, message))?;

    let calibration = CalibrationMasters {
        bias,
        dark_signal,
        dark_exposure_seconds: metadata.calibration.dark_exposure_seconds,
        dark_bayer: metadata
            .calibration
            .dark_bayer
            .map(BayerLayout::try_from)
            .transpose()
            .map_err(|message| context_read_error(path, message))?,
        flat_response,
        flat_bayer: metadata
            .calibration
            .flat_bayer
            .map(BayerLayout::try_from)
            .transpose()
            .map_err(|message| context_read_error(path, message))?,
    };
    validate_calibration(&reference, &calibration)
        .map_err(|message| context_read_error(path, message))?;
    let reference_headers = metadata
        .reference_headers
        .into_iter()
        .map(|card| (card.name, HeaderValue::from(card.value)))
        .collect();
    let input_paths = metadata
        .input_paths
        .into_iter()
        .map(PathBuf::from)
        .collect();

    Ok(RestoredContext {
        options: metadata.options,
        calibration,
        reference,
        reference_headers,
        mean,
        m2,
        count,
        rejected,
        accepted_frames: metadata.accepted_frames,
        rejected_frames: metadata.rejected_frames,
        input_paths,
    })
}

impl ContextMetadata {
    fn from_state(state: &ContextWriteState<'_>) -> std::result::Result<Self, String> {
        let input_paths = state
            .input_paths
            .iter()
            .map(|path| {
                path.to_str()
                    .map(str::to_owned)
                    .ok_or_else(|| format!("input path {} is not valid UTF-8", path.display()))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Self {
            schema_version: FORMAT_VERSION,
            options: state.options.clone(),
            calibration: CalibrationMetadata::from(state.calibration),
            reference: ImageMetadata::from(state.reference),
            reference_headers: state
                .reference_headers
                .iter()
                .map(|(name, value)| HeaderCardMetadata {
                    name: name.clone(),
                    value: HeaderValueMetadata::from(value),
                })
                .collect(),
            accepted_frames: state.accepted_frames,
            rejected_frames: state.rejected_frames,
            input_paths,
        })
    }

    fn validate(&self) -> std::result::Result<(), String> {
        if self.schema_version != FORMAT_VERSION {
            return Err(format!(
                "metadata schema version {} does not match container version {FORMAT_VERSION}",
                self.schema_version
            ));
        }
        self.options.validate().map_err(|error| error.to_string())?;
        self.reference.sample_count()?;
        for image in [
            self.calibration.bias,
            self.calibration.dark_signal,
            self.calibration.flat_response,
        ]
        .into_iter()
        .flatten()
        {
            image.sample_count()?;
        }
        if self.accepted_frames == 0 {
            return Err("context must contain at least its reference frame".into());
        }
        if self
            .calibration
            .dark_exposure_seconds
            .is_some_and(|seconds| !seconds.is_finite() || seconds <= 0.0)
        {
            return Err("context has an invalid master-dark exposure".into());
        }
        Ok(())
    }
}

impl From<&CalibrationMasters> for CalibrationMetadata {
    fn from(calibration: &CalibrationMasters) -> Self {
        Self {
            bias: calibration.bias.as_ref().map(ImageMetadata::from),
            dark_signal: calibration.dark_signal.as_ref().map(ImageMetadata::from),
            dark_exposure_seconds: calibration.dark_exposure_seconds,
            dark_bayer: calibration.dark_bayer.map(BayerMetadata::from),
            flat_response: calibration.flat_response.as_ref().map(ImageMetadata::from),
            flat_bayer: calibration.flat_bayer.map(BayerMetadata::from),
        }
    }
}

impl From<&LinearImage> for ImageMetadata {
    fn from(image: &LinearImage) -> Self {
        Self {
            width: image.width as u64,
            height: image.height as u64,
            channels: image.channels as u8,
        }
    }
}

impl ImageMetadata {
    fn dimensions(self) -> std::result::Result<(usize, usize, usize), String> {
        let width = usize::try_from(self.width)
            .map_err(|_| "image width overflows this platform".to_string())?;
        let height = usize::try_from(self.height)
            .map_err(|_| "image height overflows this platform".to_string())?;
        let channels = usize::from(self.channels);
        if width == 0 || height == 0 || !matches!(channels, 1 | 3) {
            return Err("context has invalid image dimensions or channels".into());
        }
        Ok((width, height, channels))
    }

    fn sample_count(self) -> std::result::Result<usize, String> {
        let (width, height, channels) = self.dimensions()?;
        width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(channels))
            .ok_or_else(|| "context image dimensions overflow".into())
    }
}

impl From<BayerLayout> for BayerMetadata {
    fn from(layout: BayerLayout) -> Self {
        Self {
            pattern: layout.pattern.as_str().into(),
            x_offset: layout.x_offset as u64,
            y_offset: layout.y_offset as u64,
        }
    }
}

impl TryFrom<BayerMetadata> for BayerLayout {
    type Error = String;

    fn try_from(value: BayerMetadata) -> std::result::Result<Self, Self::Error> {
        Ok(Self {
            pattern: BayerPattern::parse(&value.pattern)
                .ok_or_else(|| format!("invalid Bayer pattern {:?}", value.pattern))?,
            x_offset: usize::try_from(value.x_offset)
                .map_err(|_| "Bayer x offset overflows this platform".to_string())?,
            y_offset: usize::try_from(value.y_offset)
                .map_err(|_| "Bayer y offset overflows this platform".to_string())?,
        })
    }
}

impl From<&HeaderValue> for HeaderValueMetadata {
    fn from(value: &HeaderValue) -> Self {
        match value {
            HeaderValue::Logical(value) => Self::Logical(*value),
            HeaderValue::Integer(value) => Self::Integer(*value),
            HeaderValue::Float(value) => Self::FloatBits(value.to_bits()),
            HeaderValue::String(value) => Self::String(value.clone()),
            HeaderValue::Raw(value) => Self::Raw(value.clone()),
        }
    }
}

impl From<HeaderValueMetadata> for HeaderValue {
    fn from(value: HeaderValueMetadata) -> Self {
        match value {
            HeaderValueMetadata::Logical(value) => Self::Logical(value),
            HeaderValueMetadata::Integer(value) => Self::Integer(value),
            HeaderValueMetadata::FloatBits(value) => Self::Float(f64::from_bits(value)),
            HeaderValueMetadata::String(value) => Self::String(value),
            HeaderValueMetadata::Raw(value) => Self::Raw(value),
        }
    }
}

fn write_optional_image(
    writer: &mut impl Write,
    image: Option<&LinearImage>,
) -> std::io::Result<()> {
    if let Some(image) = image {
        write_f32_values(writer, &image.data)?;
    }
    Ok(())
}

fn read_optional_image(
    reader: &mut impl Read,
    metadata: Option<ImageMetadata>,
) -> std::result::Result<Option<LinearImage>, String> {
    metadata
        .map(|metadata| read_image(reader, metadata))
        .transpose()
}

fn read_image(
    reader: &mut impl Read,
    metadata: ImageMetadata,
) -> std::result::Result<LinearImage, String> {
    let (width, height, channels) = metadata.dimensions()?;
    let data = read_f32_values(reader, metadata.sample_count()?)?;
    LinearImage::new(width, height, channels, data).map_err(|error| error.to_string())
}

fn write_f32_values(writer: &mut impl Write, values: &[f32]) -> std::io::Result<()> {
    let mut bytes = vec![0_u8; IO_BUFFER_VALUES * 4];
    for chunk in values.chunks(IO_BUFFER_VALUES) {
        for (value, output) in chunk.iter().zip(bytes.chunks_exact_mut(4)) {
            output.copy_from_slice(&value.to_bits().to_le_bytes());
        }
        writer.write_all(&bytes[..chunk.len() * 4])?;
    }
    Ok(())
}

fn write_u32_values(writer: &mut impl Write, values: &[u32]) -> std::io::Result<()> {
    let mut bytes = vec![0_u8; IO_BUFFER_VALUES * 4];
    for chunk in values.chunks(IO_BUFFER_VALUES) {
        for (value, output) in chunk.iter().zip(bytes.chunks_exact_mut(4)) {
            output.copy_from_slice(&value.to_le_bytes());
        }
        writer.write_all(&bytes[..chunk.len() * 4])?;
    }
    Ok(())
}

fn read_f32_values(reader: &mut impl Read, length: usize) -> std::result::Result<Vec<f32>, String> {
    let mut values = Vec::with_capacity(length);
    let mut bytes = vec![0_u8; IO_BUFFER_VALUES * 4];
    let mut remaining = length;
    while remaining > 0 {
        let values_to_read = remaining.min(IO_BUFFER_VALUES);
        let bytes_to_read = values_to_read * 4;
        reader
            .read_exact(&mut bytes[..bytes_to_read])
            .map_err(|error| error.to_string())?;
        values.extend(bytes[..bytes_to_read].chunks_exact(4).map(|bytes| {
            f32::from_bits(u32::from_le_bytes(
                bytes.try_into().expect("four-byte chunk"),
            ))
        }));
        remaining -= values_to_read;
    }
    Ok(values)
}

fn read_u32_values(reader: &mut impl Read, length: usize) -> std::result::Result<Vec<u32>, String> {
    let mut values = Vec::with_capacity(length);
    let mut bytes = vec![0_u8; IO_BUFFER_VALUES * 4];
    let mut remaining = length;
    while remaining > 0 {
        let values_to_read = remaining.min(IO_BUFFER_VALUES);
        let bytes_to_read = values_to_read * 4;
        reader
            .read_exact(&mut bytes[..bytes_to_read])
            .map_err(|error| error.to_string())?;
        values.extend(
            bytes[..bytes_to_read]
                .chunks_exact(4)
                .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four-byte chunk"))),
        );
        remaining -= values_to_read;
    }
    Ok(values)
}

fn read_u32(reader: &mut impl Read) -> std::result::Result<u32, String> {
    let mut bytes = [0_u8; 4];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| error.to_string())?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> std::result::Result<u64, String> {
    let mut bytes = [0_u8; 8];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| error.to_string())?;
    Ok(u64::from_le_bytes(bytes))
}

fn validate_live_arrays(
    sample_count: usize,
    mean: &[f32],
    m2: &[f32],
    count: &[u32],
    rejected: &[u32],
    accepted_frames: u32,
) -> std::result::Result<(), String> {
    for (name, actual) in [
        ("mean", mean.len()),
        ("second moment", m2.len()),
        ("coverage", count.len()),
        ("rejection", rejected.len()),
    ] {
        if actual != sample_count {
            return Err(format!(
                "context {name} buffer has {actual} samples; expected {sample_count}"
            ));
        }
    }
    if accepted_frames == 0 {
        return Err("context must contain at least its reference frame".into());
    }
    for index in 0..sample_count {
        if !mean[index].is_finite() || !m2[index].is_finite() {
            return Err(format!(
                "context has a non-finite accumulator at sample {index}"
            ));
        }
        if count[index] > accepted_frames {
            return Err(format!(
                "context coverage {} exceeds accepted frame count {accepted_frames} at sample {index}",
                count[index]
            ));
        }
    }
    Ok(())
}

fn validate_calibration(
    reference: &LinearImage,
    calibration: &CalibrationMasters,
) -> std::result::Result<(), String> {
    let images = [
        ("bias", calibration.bias.as_ref()),
        ("dark signal", calibration.dark_signal.as_ref()),
        ("flat response", calibration.flat_response.as_ref()),
    ];
    let mut calibration_dimensions = None;
    for (name, image) in images
        .into_iter()
        .filter_map(|(name, image)| image.map(|image| (name, image)))
    {
        if image.width != reference.width || image.height != reference.height {
            return Err(format!(
                "context {name} dimensions do not match the registration reference"
            ));
        }
        if !matches!((reference.channels, image.channels), (1, 1) | (3, 1 | 3)) {
            return Err(format!(
                "context {name} channels are incompatible with the registration reference"
            ));
        }
        let dimensions = (image.width, image.height, image.channels);
        if calibration_dimensions.is_some_and(|expected| expected != dimensions) {
            return Err("context calibration buffers have inconsistent dimensions".into());
        }
        calibration_dimensions = Some(dimensions);
    }
    if let (Some(dark), Some(flat)) = (calibration.dark_bayer, calibration.flat_bayer)
        && dark != flat
    {
        return Err("context master dark and flat have different Bayer layouts".into());
    }
    for (name, layout, image) in [
        (
            "dark signal",
            calibration.dark_bayer,
            calibration.dark_signal.as_ref(),
        ),
        (
            "flat response",
            calibration.flat_bayer,
            calibration.flat_response.as_ref(),
        ),
    ] {
        if layout.is_some() && image.is_none_or(|image| image.channels != 1) {
            return Err(format!(
                "context {name} Bayer metadata requires a one-channel buffer"
            ));
        }
    }
    Ok(())
}

fn context_read_error(path: &Path, message: impl Into<String>) -> Error {
    Error::StackContextRead {
        path: path.to_path_buf(),
        message: message.into(),
    }
}

fn context_write_error(path: &Path, message: impl Into<String>) -> Error {
    Error::StackContextWrite {
        path: path.to_path_buf(),
        message: message.into(),
    }
}
