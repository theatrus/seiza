//! Seekable SER sequences, decoded one frame at a time without image processing.
//!
//! [`SerReader`] implements [`ImageSequence`], a small indexed interface for a
//! future stacker input. Samples retain their stored values and channel order;
//! no scaling, demosaicing, flipping or conversion to floating point occurs.
//! Siril's endian convention is the default; see [`EndianConvention`].
//!
//! This leaf crate is not yet connected to Seiza's file dispatch or bindings.
//! It reuses [`Pixels`] and [`BayerPattern`] from `seiza-fits`.
//!
//! ```no_run
//! use seiza_image_ser::{Error, SerReader};
//!
//! let mut sequence = SerReader::open("jupiter.ser")?;
//! let info = sequence.info().clone();
//! for index in 0..info.frame_count {
//!     let frame = sequence.read_frame(index)?;
//!     // Measure or stack frame.pixels using info.color and info.bit_depth.
//!     // Dropping each frame keeps memory independent of sequence length.
//! }
//! # Ok::<(), Error>(())
//! ```
#![deny(missing_docs)]

use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

pub use seiza_fits::{BayerPattern, Pixels};

const HEADER_SIZE: u64 = 178;

/// Why a SER sequence or frame could not be read.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An underlying read, seek or file-open failed.
    #[error(transparent)]
    Io(#[from] io::Error),
    /// The file does not carry the SER signature.
    #[error("not a SER file (missing LUCAM-RECORDER signature)")]
    NotSer,
    /// A header field has an unsupported or invalid value.
    #[error("invalid or unsupported SER {field}: {value}")]
    InvalidHeader {
        /// SER field name.
        field: &'static str,
        /// Value found in the header.
        value: u32,
    },
    /// Dimension, byte-count or offset arithmetic overflowed.
    #[error("SER dimensions or offsets overflow")]
    SizeOverflow,
    /// A frame exceeds the caller's allocation limit.
    #[error("SER frame is {bytes} bytes, exceeding limit {limit}")]
    FrameTooLarge {
        /// Encoded bytes in one frame (also the decoded sample storage size).
        bytes: u64,
        /// Configured maximum.
        limit: u64,
    },
    /// The advertised header or frames extend beyond the end of the input.
    #[error("truncated SER: need {expected} bytes, found {actual}")]
    Truncated {
        /// Minimum file length needed.
        expected: u64,
        /// Actual file length.
        actual: u64,
    },
    /// Trailing bytes are neither absent nor one timestamp per frame.
    #[error("invalid SER trailer: expected zero or {expected} bytes, found {actual}")]
    InvalidTrailer {
        /// Size of a complete timestamp trailer.
        expected: u64,
        /// Actual bytes after the frames.
        actual: u64,
    },
    /// A zero-based frame index is outside the sequence.
    #[error("frame {index} is outside sequence of {frame_count} frames")]
    FrameOutOfRange {
        /// Requested index.
        index: u64,
        /// Number of frames available.
        frame_count: u64,
    },
    /// Memory for one frame could not be reserved.
    #[error("could not allocate SER frame")]
    Allocation(#[from] std::collections::TryReserveError),
}

/// Interpretation of the SER header's historically ambiguous endian flag.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EndianConvention {
    /// Siril and SER Player: flag 0 means little endian, 1 means big endian.
    #[default]
    Siril,
    /// Published SER specification: flag 1 means little endian, 0 means big endian.
    Specification,
}

/// Reader compatibility and memory limits.
#[derive(Debug, Clone, Copy)]
pub struct ReadOptions {
    /// How to interpret the endian flag; defaults to Siril's convention.
    pub endian_convention: EndianConvention,
    /// Maximum bytes of sample storage per frame, checked before allocation.
    /// Defaults to 512 MiB. Decoding uses at most another 8 KiB of scratch.
    pub max_frame_bytes: u64,
}

impl Default for ReadOptions {
    fn default() -> Self {
        Self {
            endian_convention: EndianConvention::Siril,
            max_frame_bytes: 512 * 1024 * 1024,
        }
    }
}

/// Complementary-color CFA layouts supported by the SER container.
/// Letters give the top-left 2×2 tile in row order; no demosaicing is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplementaryPattern {
    /// Cyan, yellow / yellow, magenta.
    Cyym,
    /// Yellow, cyan / magenta, yellow.
    Ycmy,
    /// Yellow, magenta / cyan, yellow.
    Ymcy,
    /// Magenta, yellow / yellow, cyan.
    Myyc,
}

/// Stored channel layout. All rows run left to right, from the top down.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorLayout {
    /// One intensity sample per pixel.
    Mono,
    /// One raw sample per pixel, with an RGB CFA anchored at pixel (0, 0).
    Bayer(BayerPattern),
    /// One raw sample per pixel, with a complementary-color CFA at (0, 0).
    Complementary(ComplementaryPattern),
    /// Three interleaved samples per pixel, in red, green, blue order.
    Rgb,
    /// Three interleaved samples per pixel, in blue, green, red order.
    Bgr,
}

impl ColorLayout {
    /// Number of stored samples per pixel; CFA data has one, not three.
    pub fn channels(self) -> usize {
        match self {
            Self::Rgb | Self::Bgr => 3,
            _ => 1,
        }
    }
}

/// Fixed geometry and sample interpretation shared by a sequence's frames.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceInfo {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Number of frames; frame indices start at zero.
    pub frame_count: u64,
    /// Significant bits reported by the file, from 1 through 16 for SER.
    /// Samples are not shifted or normalized: SER specifies MSB alignment
    /// for depths below 8 and LSB alignment for depths from 9 through 15.
    pub bit_depth: u32,
    /// Channel ordering or CFA pattern at the top-left pixel.
    pub color: ColorLayout,
}

/// A SER date in 100 ns ticks since 0001-01-01, retaining the exact file value.
/// A zero file value is represented by `None` in metadata and frames.
/// Header local time and UTC time are kept separately; no timezone is guessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SerTimestamp(pub u64);

/// SER-specific observation metadata, separate from frame geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SerMetadata {
    /// Legacy Lumenera camera identifier.
    pub camera_id: u32,
    /// Observer text, terminated at NUL and trimmed of trailing spaces.
    pub observer: String,
    /// Instrument text; invalid UTF-8 is replaced with U+FFFD.
    pub instrument: String,
    /// Telescope text, decoded like the observer and instrument.
    pub telescope: String,
    /// Local start time as recorded; not interpreted as UTC.
    pub start_local: Option<SerTimestamp>,
    /// UTC start time as recorded.
    pub start_utc: Option<SerTimestamp>,
    /// Whether the file contains a complete per-frame UTC timestamp trailer.
    /// Determined from file length, even if the local header time is zero.
    pub has_timestamps: bool,
}

/// One owned frame, safe to retain while another is read.
#[derive(Debug)]
pub struct Frame {
    /// Zero-based index within its sequence.
    pub index: u64,
    /// Per-frame UTC timestamp, if supplied and nonzero. Never synthesized
    /// from the header start time or an assumed frame rate.
    pub timestamp_utc: Option<SerTimestamp>,
    /// Stored samples, top row first, with the layout in [`SequenceInfo::color`].
    /// SER produces only `Pixels::U8` and `Pixels::U16`. RGB/BGR is interleaved,
    /// unlike planar FITS data. Sub-16-bit samples retain their original values.
    pub pixels: Pixels,
}

/// Indexed access to raw frames for selection, repeated passes and stacking.
///
/// A consumer can process `0..source.info().frame_count` or revisit selected
/// indices without materializing the whole sequence. Mutable access owns the
/// seek cursor; parallel workers should open independent readers. A future AVI
/// reader can implement this interface after decoding its container and layout.
/// Frame dimensions and layout remain fixed for the lifetime of the source.
pub trait ImageSequence {
    /// Reader-specific errors, preserving actionable failure variants.
    type Error: std::error::Error;

    /// Geometry and layout common to all frames.
    fn info(&self) -> &SequenceInfo;

    /// Read an arbitrary zero-based frame, returning owned, unscaled samples.
    fn read_frame(&mut self, index: u64) -> Result<Frame, Self::Error>;
}

/// A validated SER header and seekable backing stream; no frames are preloaded.
///
/// The stream is interpreted from offset zero and must remain unchanged while
/// in use. Construction validates the full advertised length, dimensions,
/// format and allocation limit. A zero-frame sequence is allowed.
pub struct SerReader<R> {
    reader: R,
    info: SequenceInfo,
    metadata: SerMetadata,
    frame_bytes: u64,
    data_end: u64,
    little_endian: bool,
}

impl SerReader<BufReader<File>> {
    /// Open a SER file using Siril's byte order convention and default limits.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        Self::new(BufReader::new(File::open(path)?))
    }

    /// Open a SER file with explicit compatibility and allocation options.
    pub fn open_with_options(path: impl AsRef<Path>, options: ReadOptions) -> Result<Self, Error> {
        Self::with_options(BufReader::new(File::open(path)?), options)
    }
}

impl<R: Read + Seek> SerReader<R> {
    /// Read a SER header from a seekable input using the default options.
    pub fn new(reader: R) -> Result<Self, Error> {
        Self::with_options(reader, ReadOptions::default())
    }

    /// Validate a SER input, interpreting the entire stream from byte zero.
    pub fn with_options(mut reader: R, options: ReadOptions) -> Result<Self, Error> {
        let length = reader.seek(SeekFrom::End(0))?;
        if length < HEADER_SIZE {
            return Err(Error::Truncated {
                expected: HEADER_SIZE,
                actual: length,
            });
        }
        reader.seek(SeekFrom::Start(0))?;
        let mut header = [0_u8; HEADER_SIZE as usize];
        reader.read_exact(&mut header)?;
        if &header[..14] != b"LUCAM-RECORDER" {
            return Err(Error::NotSer);
        }
        let field = |offset| u32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
        let invalid = |field, value| Error::InvalidHeader { field, value };
        let color = match field(18) {
            0 => ColorLayout::Mono,
            8 => ColorLayout::Bayer(BayerPattern::Rggb),
            9 => ColorLayout::Bayer(BayerPattern::Grbg),
            10 => ColorLayout::Bayer(BayerPattern::Gbrg),
            11 => ColorLayout::Bayer(BayerPattern::Bggr),
            16 => ColorLayout::Complementary(ComplementaryPattern::Cyym),
            17 => ColorLayout::Complementary(ComplementaryPattern::Ycmy),
            18 => ColorLayout::Complementary(ComplementaryPattern::Ymcy),
            19 => ColorLayout::Complementary(ComplementaryPattern::Myyc),
            100 => ColorLayout::Rgb,
            101 => ColorLayout::Bgr,
            value => return Err(invalid("ColorID", value)),
        };
        let endian_flag = field(22);
        if endian_flag > 1 {
            return Err(invalid("LittleEndian", endian_flag));
        }
        let little_endian = match options.endian_convention {
            EndianConvention::Siril => endian_flag == 0,
            EndianConvention::Specification => endian_flag == 1,
        };
        let width = field(26);
        let height = field(30);
        let bit_depth = field(34);
        if width == 0 {
            return Err(invalid("ImageWidth", width));
        }
        if height == 0 {
            return Err(invalid("ImageHeight", height));
        }
        if !(1..=16).contains(&bit_depth) {
            return Err(invalid("PixelDepthPerPlane", bit_depth));
        }
        let frame_count = u64::from(field(38));
        let sample_bytes = if bit_depth <= 8 { 1 } else { 2 };
        let frame_bytes = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|n| n.checked_mul(color.channels() as u64))
            .and_then(|n| n.checked_mul(sample_bytes))
            .ok_or(Error::SizeOverflow)?;
        if frame_bytes > options.max_frame_bytes {
            return Err(Error::FrameTooLarge {
                bytes: frame_bytes,
                limit: options.max_frame_bytes,
            });
        }
        // Vec cannot hold more than isize::MAX bytes, even on a 32-bit host.
        if frame_bytes > isize::MAX as u64 {
            return Err(Error::SizeOverflow);
        }
        let data_end = frame_bytes
            .checked_mul(frame_count)
            .and_then(|n| n.checked_add(HEADER_SIZE))
            .ok_or(Error::SizeOverflow)?;
        if length < data_end {
            return Err(Error::Truncated {
                expected: data_end,
                actual: length,
            });
        }
        let trailer_bytes = frame_count * 8;
        let remaining = length - data_end;
        if remaining != 0 && remaining != trailer_bytes {
            return Err(Error::InvalidTrailer {
                expected: trailer_bytes,
                actual: remaining,
            });
        }
        let metadata = SerMetadata {
            camera_id: field(14),
            observer: header_text(&header[42..82]),
            instrument: header_text(&header[82..122]),
            telescope: header_text(&header[122..162]),
            start_local: timestamp(&header[162..170]),
            start_utc: timestamp(&header[170..178]),
            has_timestamps: remaining != 0,
        };
        Ok(Self {
            reader,
            info: SequenceInfo {
                width,
                height,
                frame_count,
                bit_depth,
                color,
            },
            metadata,
            frame_bytes,
            data_end,
            little_endian,
        })
    }

    /// Geometry and sample layout shared by every frame.
    pub fn info(&self) -> &SequenceInfo {
        &self.info
    }

    /// Observation strings, header dates and timestamp-trailer availability.
    pub fn metadata(&self) -> &SerMetadata {
        &self.metadata
    }

    /// Read one frame by zero-based index, without reading other frames.
    /// Out-of-range requests leave the reader usable for subsequent requests.
    pub fn read_frame(&mut self, index: u64) -> Result<Frame, Error> {
        if index >= self.info.frame_count {
            return Err(Error::FrameOutOfRange {
                index,
                frame_count: self.info.frame_count,
            });
        }
        // All offsets were bounded by the file length during construction.
        let timestamp_utc = if self.metadata.has_timestamps {
            self.reader
                .seek(SeekFrom::Start(self.data_end + index * 8))?;
            let mut bytes = [0; 8];
            self.reader.read_exact(&mut bytes)?;
            timestamp(&bytes)
        } else {
            None
        };
        self.reader
            .seek(SeekFrom::Start(HEADER_SIZE + index * self.frame_bytes))?;
        let pixels = if self.info.bit_depth <= 8 {
            let mut bytes = Vec::new();
            bytes.try_reserve_exact(self.frame_bytes as usize)?;
            bytes.resize(self.frame_bytes as usize, 0);
            self.reader.read_exact(&mut bytes)?;
            Pixels::U8(bytes)
        } else {
            let count = self.frame_bytes as usize / 2;
            let mut samples = Vec::new();
            samples.try_reserve_exact(count)?;
            let mut buffer = [0; 8192];
            while samples.len() < count {
                let bytes = ((count - samples.len()) * 2).min(buffer.len());
                self.reader.read_exact(&mut buffer[..bytes])?;
                samples.extend(buffer[..bytes].as_chunks::<2>().0.iter().map(|&pair| {
                    if self.little_endian {
                        u16::from_le_bytes(pair)
                    } else {
                        u16::from_be_bytes(pair)
                    }
                }));
            }
            Pixels::U16(samples)
        };
        Ok(Frame {
            index,
            timestamp_utc,
            pixels,
        })
    }
}

impl<R: Read + Seek> ImageSequence for SerReader<R> {
    type Error = Error;
    fn info(&self) -> &SequenceInfo {
        self.info()
    }
    fn read_frame(&mut self, index: u64) -> Result<Frame, Error> {
        self.read_frame(index)
    }
}

fn header_text(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).trim_end().to_owned()
}

fn timestamp(bytes: &[u8]) -> Option<SerTimestamp> {
    let ticks = u64::from_le_bytes(bytes.try_into().unwrap());
    (ticks != 0).then_some(SerTimestamp(ticks))
}
