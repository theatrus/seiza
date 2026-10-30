//! Read a rectangle of an XISF image's samples without decoding the rest.
//!
//! An uncompressed image's samples sit at known places in its data block,
//! so a window is one short read per row (per row of each channel, for
//! planar storage). Nothing the size of the image is read or allocated.

use crate::{
    BlockLocation, ByteOrder, PixelStorage, Rescale, SampleFormat, XisfError, XisfImageInfo,
    complex_needs_its_own_reader, declares_unit_bounds, header_dir, parse_file,
};
use seiza_fits::ImageWindow;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// A window of an XISF image's samples, with the image's description.
///
/// The samples are as [`open`](crate::open) decodes them, as `f32`:
/// unsigned integers at their stored values, floats as stored. Read
/// [`XisfImageInfo::bounds`] and [`XisfImageInfo::sample_format`] to tell
/// whether they are normalized, or let [`Self::rescale_normalized_to`] do
/// what [`XisfImage::rescale_normalized_to`](crate::XisfImage::rescale_normalized_to)
/// does for a whole image.
#[derive(Clone, Debug)]
pub struct XisfWindow {
    pub window: ImageWindow,
    pub info: XisfImageInfo,
}

impl XisfWindow {
    /// Put a window of an image the file declares as normalized onto
    /// `0..=full_scale`, by the rules of
    /// [`XisfImage::rescale_normalized_to`](crate::XisfImage::rescale_normalized_to):
    /// only a declared range of exactly `0:1`, only floating-point samples,
    /// and only a finite, positive `full_scale`. Answers whether it ran; on
    /// success `info.bounds` becomes `0:full_scale`.
    pub fn rescale_normalized_to(&mut self, full_scale: f32) -> bool {
        if !declares_unit_bounds(&self.info) {
            return false;
        }
        self.rescale_from((0.0, 1.0), full_scale)
    }

    /// Map floating-point samples from `source` onto `0..=full_scale`, as
    /// [`XisfImage::rescale_from`](crate::XisfImage::rescale_from) maps a
    /// whole image: linearly, without clamping, refused for integer samples
    /// or a range that is not finite with `low < high`. Answers whether it
    /// ran; on success `info.bounds` becomes `0:full_scale`.
    ///
    /// A `Float64` image's window holds its samples as `f32` already, so
    /// they are rounded once before the map rather than once after it.
    pub fn rescale_from(&mut self, source: (f64, f64), full_scale: f32) -> bool {
        let Some(rescale) = Rescale::new(self.info.sample_format, source, full_scale) else {
            return false;
        };
        for sample in &mut self.window.values {
            *sample = rescale.f32(*sample);
        }
        self.info.bounds = Some(rescale.bounds());
        true
    }
}

/// Read the samples of the rectangle at `(x, y)`, `width` × `height`
/// pixels, of every channel of the first image in an XISF file (or a
/// distributed unit's `.xish` header), with the image's description.
///
/// Rows count from the first stored row, as [`open`](crate::open) holds
/// them. The rectangle is clipped to the image; one wholly outside it gives
/// an empty window. A side longer than [`seiza_fits::MAX_WINDOW_EDGE`] is
/// refused with [`XisfError::Unsupported`], and so is an image whose
/// samples cannot be reached without decoding it all: a compressed one, a
/// CIELab one (converted to RGB as a whole) and a complex one. The block's
/// checksum is not verified, as that would read the whole block.
pub fn read_window(
    path: &Path,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> Result<XisfWindow, XisfError> {
    let mut file = std::fs::File::open(path)?;
    let file_bytes = file.metadata()?.len();
    let mut parsed = parse_file(&mut file, file_bytes, header_dir(path).as_deref())?;
    let image = parsed.take(0)?;
    let info = &image.info;
    if info.sample_format.is_complex() {
        return Err(complex_needs_its_own_reader());
    }
    if info.color_space == "CIELab" {
        return Err(XisfError::Unsupported(
            "a window of a CIELab image; open the image, which converts it to RGB".into(),
        ));
    }
    if image.block.compression.is_some() {
        return Err(XisfError::Unsupported(
            "a window of a compressed image; open the image to decompress it".into(),
        ));
    }
    let mut window =
        ImageWindow::for_request(info.width, info.height, info.planes, x, y, width, height)
            .map_err(XisfError::Unsupported)?;
    if window.width == 0 {
        return Ok(XisfWindow {
            window,
            info: image.info,
        });
    }

    let mut external;
    let mut block = match (&image.block.location, &image.block.inline_data) {
        (&BlockLocation::Attachment { offset, .. }, _) => Block::File(&mut file, offset),
        (BlockLocation::External { path, offset, .. }, _) => {
            external = std::fs::File::open(path)?;
            Block::File(&mut external, *offset)
        }
        (_, Some(data)) => Block::Bytes(data),
        (_, None) => return Err(XisfError::Malformed("inline data block is missing".into())),
    };
    let format = info.sample_format;
    let order = info.byte_order;
    let sample_bytes = format.bytes_per_sample();
    let (image_width, image_height, planes) = (info.width, info.height, info.planes);
    let mut values = vec![0.0_f32; planes * window.width * window.height];
    let normal = info.pixel_storage == PixelStorage::Normal && planes > 1;
    if normal {
        // Each pixel's channels are stored together: one read a row.
        let mut row = vec![0; window.width * planes * sample_bytes];
        for line in 0..window.height {
            let index = ((window.y + line) * image_width + window.x) * planes;
            block.read_at(index * sample_bytes, &mut row)?;
            for (column, pixel) in row.chunks_exact(planes * sample_bytes).enumerate() {
                for (plane, sample) in pixel.chunks_exact(sample_bytes).enumerate() {
                    values[(plane * window.height + line) * window.width + column] =
                        decode(sample, format, order);
                }
            }
        }
    } else {
        let mut row = vec![0; window.width * sample_bytes];
        for plane in 0..planes {
            for line in 0..window.height {
                let index = (plane * image_height + window.y + line) * image_width + window.x;
                block.read_at(index * sample_bytes, &mut row)?;
                let start = (plane * window.height + line) * window.width;
                for (value, sample) in values[start..start + window.width]
                    .iter_mut()
                    .zip(row.chunks_exact(sample_bytes))
                {
                    *value = decode(sample, format, order);
                }
            }
        }
    }
    window.values = values;
    Ok(XisfWindow {
        window,
        info: image.info,
    })
}

/// Where an image's uncompressed data block is: in a file from an offset,
/// or decoded from inline or embedded text.
enum Block<'a> {
    File(&'a mut std::fs::File, u64),
    Bytes(&'a [u8]),
}

impl Block<'_> {
    /// Fill `buffer` from the block's bytes at `offset`. The block's size
    /// was checked against the image's when the header was read.
    fn read_at(&mut self, offset: usize, buffer: &mut [u8]) -> Result<(), XisfError> {
        let truncated = || XisfError::Malformed("image attachment is truncated".into());
        match self {
            Self::File(file, start) => {
                file.seek(SeekFrom::Start(*start + offset as u64))?;
                file.read_exact(buffer).map_err(|error| {
                    if error.kind() == std::io::ErrorKind::UnexpectedEof {
                        truncated()
                    } else {
                        XisfError::Io(error)
                    }
                })
            }
            Self::Bytes(data) => {
                let bytes = data
                    .get(offset..offset + buffer.len())
                    .ok_or_else(truncated)?;
                buffer.copy_from_slice(bytes);
                Ok(())
            }
        }
    }
}

/// One stored sample as `f32`, as the decoded image holds it.
fn decode(sample: &[u8], format: SampleFormat, order: ByteOrder) -> f32 {
    macro_rules! read {
        ($type:ty) => {{
            let bytes = sample.try_into().unwrap();
            match order {
                ByteOrder::Little => <$type>::from_le_bytes(bytes),
                ByteOrder::Big => <$type>::from_be_bytes(bytes),
            }
        }};
    }
    match format {
        SampleFormat::UInt8 => f32::from(sample[0]),
        SampleFormat::UInt16 => f32::from(read!(u16)),
        // Decoded images hold these as f64.
        SampleFormat::UInt32 => f64::from(read!(u32)) as f32,
        SampleFormat::UInt64 => read!(u64) as f64 as f32,
        SampleFormat::Float32 => read!(f32),
        SampleFormat::Float64 => read!(f64) as f32,
        SampleFormat::Complex32 | SampleFormat::Complex64 => {
            unreachable!("complex images are refused before reading")
        }
    }
}
