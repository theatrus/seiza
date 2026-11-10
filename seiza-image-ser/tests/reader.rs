use seiza_image_ser::{
    BayerPattern, ColorLayout, ComplementaryPattern, EndianConvention, Error, ImageSequence,
    Pixels, ReadOptions, SerReader, SerTimestamp,
};
use std::cell::Cell;
use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
use std::rc::Rc;

// Byte-level fixtures are independent of the reader; header fields are always
// little endian, even when the pixel flag denotes big endian.
fn header(width: u32, height: u32, depth: u32, color: u32, frames: u32) -> Vec<u8> {
    let mut bytes = vec![0; 178];
    bytes[..14].copy_from_slice(b"LUCAM-RECORDER");
    for (offset, value) in [
        (26, width),
        (30, height),
        (34, depth),
        (18, color),
        (38, frames),
    ] {
        set_field(&mut bytes, offset, value);
    }
    bytes
}

fn set_field(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn samples(frame: seiza_image_ser::Frame) -> Vec<u16> {
    match frame.pixels {
        Pixels::U8(values) => values.into_iter().map(u16::from).collect(),
        Pixels::U16(values) => values,
        _ => panic!("SER must retain unsigned integer samples"),
    }
}

#[test]
fn indexed_frames_are_owned_and_trait_is_object_safe() {
    let mut bytes = header(2, 2, 8, 0, 3);
    bytes.extend(0..12);
    let mut reader = SerReader::new(Cursor::new(bytes)).unwrap();
    let source: &mut dyn ImageSequence<Error = Error> = &mut reader;
    assert_eq!(source.info().frame_count, 3);
    let last = source.read_frame(2).unwrap();
    assert_eq!(last.index, 2);
    assert_eq!(last.timestamp_utc, None);
    assert_eq!(samples(source.read_frame(0).unwrap()), [0, 1, 2, 3]);
    assert_eq!(samples(last), [8, 9, 10, 11]);
    assert!(matches!(
        source.read_frame(u64::MAX),
        Err(Error::FrameOutOfRange { .. })
    ));
    assert_eq!(samples(source.read_frame(1).unwrap()), [4, 5, 6, 7]);
}

#[test]
fn all_depths_preserve_stored_values_and_use_the_correct_storage_width() {
    for depth in 1..=16 {
        let mut bytes = header(2, 1, depth, 0, 1);
        let expected = if depth <= 8 {
            bytes.extend([0x80, 0xc0]);
            vec![0x80, 0xc0] // Do not shift sub-eight-bit data down.
        } else {
            bytes.extend([0x01, 0x01, 0xfe, 0x01]);
            vec![0x101, 0x1fe] // Do not promote 9..15-bit values to full range.
        };
        let mut reader = SerReader::new(Cursor::new(bytes)).unwrap();
        assert_eq!(reader.info().bit_depth, depth);
        let frame = reader.read_frame(0).unwrap();
        assert_eq!(matches!(frame.pixels, Pixels::U8(_)), depth <= 8);
        assert_eq!(samples(frame), expected);
    }
}

#[test]
fn siril_and_specification_endian_conventions_are_explicit() {
    for convention in [EndianConvention::Siril, EndianConvention::Specification] {
        for flag in [0, 1] {
            let mut bytes = header(2, 1, 16, 0, 1);
            set_field(&mut bytes, 22, flag);
            bytes.extend([0x12, 0x34, 0xcd, 0xab]);
            let mut reader = SerReader::with_options(
                Cursor::new(bytes),
                ReadOptions {
                    endian_convention: convention,
                    ..ReadOptions::default()
                },
            )
            .unwrap();
            let little = (convention == EndianConvention::Siril) == (flag == 0);
            assert_eq!(
                samples(reader.read_frame(0).unwrap()),
                if little {
                    vec![0x3412, 0xabcd]
                } else {
                    vec![0x1234, 0xcdab]
                }
            );
        }
    }
}

#[test]
fn every_ser_color_id_retains_channel_order_and_top_left_cfa() {
    let layouts = [
        (0, ColorLayout::Mono),
        (8, ColorLayout::Bayer(BayerPattern::Rggb)),
        (9, ColorLayout::Bayer(BayerPattern::Grbg)),
        (10, ColorLayout::Bayer(BayerPattern::Gbrg)),
        (11, ColorLayout::Bayer(BayerPattern::Bggr)),
        (16, ColorLayout::Complementary(ComplementaryPattern::Cyym)),
        (17, ColorLayout::Complementary(ComplementaryPattern::Ycmy)),
        (18, ColorLayout::Complementary(ComplementaryPattern::Ymcy)),
        (19, ColorLayout::Complementary(ComplementaryPattern::Myyc)),
        (100, ColorLayout::Rgb),
        (101, ColorLayout::Bgr),
    ];
    for (id, layout) in layouts {
        for depth in [8, 16] {
            let mut bytes = header(2, 2, depth, id, 1);
            let values: Vec<u16> = (1..=4 * layout.channels() as u16).collect();
            for &value in &values {
                if depth == 8 {
                    bytes.push(value as u8);
                } else {
                    bytes.extend(value.to_le_bytes());
                }
            }
            let mut reader = SerReader::new(Cursor::new(bytes)).unwrap();
            assert_eq!(reader.info().color, layout);
            assert_eq!(samples(reader.read_frame(0).unwrap()), values);
        }
    }
}

#[test]
fn observation_metadata_and_utc_trailer_are_separate() {
    let mut bytes = header(1, 1, 8, 0, 3);
    set_field(&mut bytes, 14, 42);
    bytes[42..48].copy_from_slice(b"Ada   ");
    bytes[82..88].copy_from_slice(b"Cam\0xx");
    bytes[122..126].copy_from_slice(b"T\xff  ");
    let local = 638_000_000_123_456_789_u64;
    let utc = local - 36_000_000_000;
    bytes[162..170].copy_from_slice(&local.to_le_bytes());
    bytes[170..178].copy_from_slice(&utc.to_le_bytes());
    bytes.extend([11, 22, 33]);
    for ticks in [utc, 0, utc + 123] {
        bytes.extend(ticks.to_le_bytes());
    }
    let mut reader = SerReader::new(Cursor::new(bytes)).unwrap();
    let metadata = reader.metadata();
    assert_eq!(metadata.camera_id, 42);
    assert_eq!(metadata.observer, "Ada");
    assert_eq!(metadata.instrument, "Cam");
    assert_eq!(metadata.telescope, "T\u{fffd}");
    assert_eq!(metadata.start_local, Some(SerTimestamp(local)));
    assert_eq!(metadata.start_utc, Some(SerTimestamp(utc)));
    assert!(metadata.has_timestamps);
    for (index, expected) in [
        (2, Some(SerTimestamp(utc + 123))),
        (0, Some(SerTimestamp(utc))),
        (1, None),
    ] {
        let frame = reader.read_frame(index).unwrap();
        assert_eq!(frame.timestamp_utc, expected);
        assert_eq!(samples(frame), [(index as u16 + 1) * 11]);
    }
}

#[test]
fn trailer_presence_follows_length_not_header_dates() {
    for local in [0_u64, 10] {
        for trailer in [false, true] {
            let mut bytes = header(1, 1, 8, 0, 1);
            bytes[162..170].copy_from_slice(&local.to_le_bytes());
            bytes.push(7);
            if trailer {
                bytes.extend(25_u64.to_le_bytes());
            }
            let mut reader = SerReader::new(Cursor::new(bytes)).unwrap();
            assert_eq!(reader.metadata().has_timestamps, trailer);
            assert_eq!(
                reader.read_frame(0).unwrap().timestamp_utc,
                trailer.then_some(SerTimestamp(25))
            );
        }
    }
}

#[test]
fn rejects_invalid_headers_before_reading_frames() {
    for (offset, value, field) in [
        (18, 7, "ColorID"),
        (22, 2, "LittleEndian"),
        (26, 0, "ImageWidth"),
        (30, 0, "ImageHeight"),
        (34, 0, "PixelDepthPerPlane"),
        (34, 17, "PixelDepthPerPlane"),
    ] {
        let mut bytes = header(1, 1, 8, 0, 0);
        set_field(&mut bytes, offset, value);
        assert!(matches!(SerReader::new(Cursor::new(bytes)),
            Err(Error::InvalidHeader { field: found, value: actual }) if found == field && actual == value));
    }
    let mut bytes = header(1, 1, 8, 0, 0);
    bytes[0] = 0;
    assert!(matches!(
        SerReader::new(Cursor::new(bytes)),
        Err(Error::NotSer)
    ));
}

#[test]
fn validates_header_frames_and_complete_trailers() {
    for length in 0..178 {
        assert!(matches!(
            SerReader::new(Cursor::new(vec![0; length])),
            Err(Error::Truncated { .. })
        ));
    }
    let mut bytes = header(2, 1, 8, 0, 2);
    bytes.extend([1, 2, 3]);
    assert!(matches!(
        SerReader::new(Cursor::new(bytes.clone())),
        Err(Error::Truncated {
            expected: 182,
            actual: 181
        })
    ));
    bytes.push(4);
    for length in 1..=20 {
        let mut with_trailer = bytes.clone();
        with_trailer.extend(vec![0; length]);
        let result = SerReader::new(Cursor::new(with_trailer));
        if length == 16 {
            assert!(result.is_ok());
        } else {
            assert!(matches!(result, Err(Error::InvalidTrailer { .. })));
        }
    }
}

#[test]
fn zero_frame_sequences_and_allocation_limits() {
    let mut reader = SerReader::new(Cursor::new(header(1, 1, 8, 0, 0))).unwrap();
    assert_eq!(reader.info().frame_count, 0);
    assert!(!reader.metadata().has_timestamps);
    assert!(matches!(
        reader.read_frame(0),
        Err(Error::FrameOutOfRange { frame_count: 0, .. })
    ));
    let options = ReadOptions {
        max_frame_bytes: 11,
        ..ReadOptions::default()
    };
    assert!(matches!(
        SerReader::with_options(Cursor::new(header(2, 1, 16, 100, 0)), options),
        Err(Error::FrameTooLarge {
            bytes: 12,
            limit: 11
        })
    ));
    assert!(matches!(
        SerReader::new(Cursor::new(header(u32::MAX, u32::MAX, 16, 100, 1))),
        Err(Error::SizeOverflow)
    ));
    // Individually permitted frame size, but frame_count * frame_bytes overflows.
    let options = ReadOptions {
        max_frame_bytes: u64::MAX,
        ..ReadOptions::default()
    };
    assert!(matches!(
        SerReader::with_options(Cursor::new(header(u32::MAX, 1, 16, 0, u32::MAX)), options),
        Err(Error::SizeOverflow)
    ));
}

struct CountingReader {
    cursor: Cursor<Vec<u8>>,
    bytes_read: Rc<Cell<usize>>,
}
impl Read for CountingReader {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        let count = self.cursor.read(bytes)?;
        self.bytes_read.set(self.bytes_read.get() + count);
        Ok(count)
    }
}
impl Seek for CountingReader {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.cursor.seek(position)
    }
}

#[test]
fn opening_and_selecting_do_not_decode_unselected_frames() {
    let mut bytes = header(10, 1, 16, 0, 100);
    bytes.extend(vec![0; 2000]);
    let count = Rc::new(Cell::new(0));
    let mut reader = SerReader::new(CountingReader {
        cursor: Cursor::new(bytes),
        bytes_read: count.clone(),
    })
    .unwrap();
    assert_eq!(count.get(), 178);
    reader.read_frame(99).unwrap();
    assert_eq!(count.get(), 198);
}

#[test]
fn sixteen_bit_decoding_crosses_scratch_buffer_boundaries() {
    let mut bytes = header(5001, 1, 16, 0, 1);
    let values: Vec<u16> = (0..5001).map(|i| i * 13).collect();
    for &value in &values {
        bytes.extend(value.to_le_bytes());
    }
    let mut reader = SerReader::new(Cursor::new(bytes)).unwrap();
    assert_eq!(samples(reader.read_frame(0).unwrap()), values);
}

// A virtual sparse stream keeps the >4 GiB test cheap on platforms where
// extending an ordinary file would allocate its entire length on disk.
struct LargeSer {
    header: Vec<u8>,
    position: u64,
    last_offset: u64,
    length: u64,
}

impl Read for LargeSer {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        let count = (self.length.saturating_sub(self.position)).min(bytes.len() as u64) as usize;
        for byte in &mut bytes[..count] {
            *byte = if self.position < 178 {
                self.header[self.position as usize]
            } else if self.position == self.last_offset {
                123
            } else {
                0
            };
            self.position += 1;
        }
        Ok(count)
    }
}

impl Seek for LargeSer {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        self.position = match from {
            SeekFrom::Start(value) => Some(value),
            SeekFrom::Current(delta) => self.position.checked_add_signed(delta),
            SeekFrom::End(delta) => self.length.checked_add_signed(delta),
        }
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid seek"))?;
        Ok(self.position)
    }
}

#[test]
fn seeks_beyond_four_gibibytes() {
    let frame_count = 65_537_u32;
    let frame_bytes = 65_536_u64;
    let last_offset = 178 + frame_bytes * u64::from(frame_count - 1);
    let mut reader = SerReader::new(LargeSer {
        header: header(65_536, 1, 8, 0, frame_count),
        position: 0,
        last_offset,
        length: last_offset + frame_bytes,
    })
    .unwrap();
    let pixels = samples(reader.read_frame(u64::from(frame_count - 1)).unwrap());
    assert_eq!(pixels.len(), 65_536);
    assert_eq!(pixels[0], 123);
    assert!(pixels[1..].iter().all(|&v| v == 0));
    assert!(
        samples(reader.read_frame(0).unwrap())
            .iter()
            .all(|&v| v == 0)
    );
}

#[test]
fn opens_files_with_default_and_explicit_options() {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&header(2, 1, 16, 0, 1)).unwrap();
    file.write_all(&[0x12, 0x34, 0xcd, 0xab]).unwrap();
    file.flush().unwrap();
    let mut reader = SerReader::open(file.path()).unwrap();
    assert_eq!(samples(reader.read_frame(0).unwrap()), [0x3412, 0xabcd]);
    let options = ReadOptions {
        endian_convention: EndianConvention::Specification,
        ..ReadOptions::default()
    };
    let mut reader = SerReader::open_with_options(file.path(), options).unwrap();
    assert_eq!(samples(reader.read_frame(0).unwrap()), [0x1234, 0xcdab]);
}

#[test]
fn reports_io_errors_instead_of_returning_partial_frames() {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&header(16_384, 1, 8, 0, 1)).unwrap();
    file.write_all(&vec![1; 16_384]).unwrap();
    file.flush().unwrap();
    let mut reader = SerReader::open(file.path()).unwrap();
    file.as_file().set_len(178).unwrap();
    assert!(
        matches!(reader.read_frame(0), Err(Error::Io(error)) if error.kind() == io::ErrorKind::UnexpectedEof)
    );
}
