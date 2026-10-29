//! Read a rectangle of an image's samples without decoding the rest.
//!
//! The header gives each sample's place in the data unit, so a window is a
//! few short reads: one per row of each plane, from a memory map of the
//! file. Only the pages those rows cover are read, and nothing the size of
//! the image is allocated. A tile-compressed image decodes only the tiles
//! that cross the window.

use crate::compressed::{self, Region, Shape};
use crate::{FitsError, ImageHdu, ImageSpec, locate_image, read_payload_exact};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// The longest side, in pixels, of a window [`read_window`] reads in one
/// call: 1024 × 1024 samples a plane at most, 12 MiB of `f32` for an RGB
/// image. Read a larger area in several windows, or open the image whole.
pub const MAX_WINDOW_EDGE: usize = 1024;

/// A rectangle of an image's samples in physical units, read without
/// decoding the rest of the data unit.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageWindow {
    /// The full image's size and plane count, from the header.
    pub image_width: usize,
    pub image_height: usize,
    pub planes: usize,
    /// The part of the request that lies on the image, clipped to it. Rows
    /// count in the order [`FitsImage`](crate::FitsImage) holds them
    /// (storage order), so `(x, y)` here is the pixel
    /// [`FitsImage::open`](crate::FitsImage::open) puts at that index.
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    /// The samples, plane by plane, each row by row:
    /// `values[(plane * height + row) * width + column]`. Empty when the
    /// request misses the image.
    pub values: Vec<f32>,
}

impl ImageWindow {
    /// The sample of `plane` at image pixel `(x, y)`, when the window holds
    /// it.
    pub fn get(&self, plane: usize, x: usize, y: usize) -> Option<f32> {
        if plane >= self.planes
            || !(self.x..self.x + self.width).contains(&x)
            || !(self.y..self.y + self.height).contains(&y)
        {
            return None;
        }
        let row = y - self.y;
        self.values
            .get((plane * self.height + row) * self.width + x - self.x)
            .copied()
    }

    /// The window of an image of the given size that a request for `x`,
    /// `y`, `width` and `height` covers, with no samples yet: the request
    /// clipped to the image, or an empty window at its nearest edge when it
    /// misses. Fails, saying why, when a side of the request is longer than
    /// [`MAX_WINDOW_EDGE`].
    pub fn for_request(
        image_width: usize,
        image_height: usize,
        planes: usize,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> Result<Self, String> {
        if width > MAX_WINDOW_EDGE || height > MAX_WINDOW_EDGE {
            return Err(format!(
                "a {width} × {height} window is larger than the {MAX_WINDOW_EDGE} × \
                 {MAX_WINDOW_EDGE} read at once; read it in parts, or open the image"
            ));
        }
        let left = x.min(image_width);
        let top = y.min(image_height);
        let right = x.saturating_add(width).min(image_width);
        let bottom = y.saturating_add(height).min(image_height);
        let (width, height) = if right > left && bottom > top {
            (right - left, bottom - top)
        } else {
            (0, 0)
        };
        Ok(Self {
            image_width,
            image_height,
            planes,
            x: left,
            y: top,
            width,
            height,
            values: Vec::new(),
        })
    }
}

/// Read the samples of the rectangle at `(x, y)`, `width` × `height`
/// pixels, of every plane of a FITS image, in physical units:
/// `BZERO + BSCALE × stored` as `f32`, an integer sample equal to `BLANK`
/// as NaN, exactly as [`FitsImage::open`](crate::FitsImage::open) and
/// [`FitsImage::into_physical_f32`](crate::FitsImage::into_physical_f32)
/// give the same pixels. A float sample keeps its stored value, NaN and
/// infinity included.
///
/// The image is the HDU [`FitsImage::open`](crate::FitsImage::open) reads.
/// The rectangle is clipped to the image; one wholly outside it gives an
/// empty window. A side longer than [`MAX_WINDOW_EDGE`] is refused with
/// [`FitsError::Unsupported`].
///
/// The file is mapped, and only the rows of the window are read from it,
/// so the time and memory a read takes grow with the window, not the
/// image. A tile-compressed image decodes only the tiles that cross the
/// window; the table that indexes them is read from the map too.
pub fn read_window(
    path: &Path,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> Result<ImageWindow, FitsError> {
    let mut file = std::fs::File::open(path)?;
    // Safety: the file is opened read-only, as for FitsImage::open;
    // concurrent truncation would fault, which is acceptable for
    // locally-managed image files.
    match unsafe { memmap2::Mmap::map(&file) } {
        Ok(map) => {
            let mut reader = std::io::Cursor::new(&map[..]);
            window_from(&mut reader, Some(&map), x, y, width, height)
        }
        Err(_) => window_from(&mut file, None, x, y, width, height),
    }
}

/// [`read_window`] on `reader`, with the whole file in `mapped` when it is
/// mapped.
pub(crate) fn window_from<R: Read + Seek>(
    reader: &mut R,
    mapped: Option<&[u8]>,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> Result<ImageWindow, FitsError> {
    let ImageHdu {
        header,
        data_start,
        table,
        ..
    } = locate_image(reader, true)?;
    let spec = ImageSpec::from_headers(&header.cards)?;
    let mut window =
        ImageWindow::for_request(spec.width, spec.height, spec.planes, x, y, width, height)
            .map_err(FitsError::Unsupported)?;
    if window.width == 0 {
        return Ok(window);
    }
    let region = Region {
        x: window.x,
        y: window.y,
        width: window.width,
        height: window.height,
    };
    let sample_bytes = spec.sample_bytes();
    let samples = spec.planes * region.width * region.height;
    let mut values = Vec::with_capacity(samples);
    match table {
        Some(cards) => {
            let shape = Shape {
                width: spec.width,
                height: spec.height,
                planes: spec.planes,
            };
            let table_bytes = match mapped {
                Some(data) => Some(
                    usize::try_from(data_start)
                        .ok()
                        .and_then(|start| data.get(start..))
                        .ok_or_else(|| FitsError::Malformed("data runs past EOF".into()))?,
                ),
                None => {
                    reader.seek(SeekFrom::Start(data_start))?;
                    None
                }
            };
            let raw = compressed::decompress_window(
                table_bytes,
                reader,
                &cards,
                &shape,
                region,
                spec.blank,
            )?;
            values.extend(
                raw.chunks_exact(sample_bytes)
                    .map(|sample| spec.physical(sample)),
            );
        }
        None => {
            let mut row = vec![0; region.width * sample_bytes];
            for plane in 0..spec.planes {
                for line in region.y..region.y + region.height {
                    let index = (plane * spec.height + line) * spec.width + region.x;
                    let offset = (index as u64)
                        .checked_mul(sample_bytes as u64)
                        .and_then(|offset| offset.checked_add(data_start))
                        .ok_or_else(|| FitsError::Malformed("data runs past EOF".into()))?;
                    read_at(reader, mapped, offset, &mut row)?;
                    values.extend(
                        row.chunks_exact(sample_bytes)
                            .map(|sample| spec.physical(sample)),
                    );
                }
            }
        }
    }
    window.values = values;
    Ok(window)
}

/// Fill `buffer` from the file's bytes at `offset`.
fn read_at(
    reader: &mut (impl Read + Seek),
    mapped: Option<&[u8]>,
    offset: u64,
    buffer: &mut [u8],
) -> Result<(), FitsError> {
    let past_eof = || FitsError::Malformed("data runs past EOF".into());
    match mapped {
        Some(data) => {
            let start = usize::try_from(offset).map_err(|_| past_eof())?;
            let bytes = start
                .checked_add(buffer.len())
                .and_then(|end| data.get(start..end))
                .ok_or_else(past_eof)?;
            buffer.copy_from_slice(bytes);
            Ok(())
        }
        None => {
            reader.seek(SeekFrom::Start(offset))?;
            read_payload_exact(reader, buffer)
        }
    }
}

impl ImageSpec {
    /// The bytes of one stored sample.
    pub(crate) fn sample_bytes(self) -> usize {
        (self.bitpix.unsigned_abs() / 8) as usize
    }

    /// One big-endian stored sample in physical units, as
    /// [`FitsImage::into_physical_f32`](crate::FitsImage::into_physical_f32)
    /// converts the decoded image: `BZERO + BSCALE × stored`, computed in
    /// `f64`, an integer sample equal to `BLANK` as NaN, and 64-bit data
    /// with `BZERO` 2^63 as the unsigned integer it encodes.
    fn physical(self, sample: &[u8]) -> f32 {
        let integer = |stored: i64, value: f64| {
            if self.blank == Some(stored) {
                f32::NAN
            } else {
                (self.bzero + self.bscale * value) as f32
            }
        };
        match self.bitpix {
            8 => integer(i64::from(sample[0]), f64::from(sample[0])),
            16 => {
                let stored = i16::from_be_bytes([sample[0], sample[1]]);
                integer(i64::from(stored), f64::from(stored))
            }
            32 => {
                let stored = i32::from_be_bytes(sample.try_into().unwrap());
                integer(i64::from(stored), f64::from(stored))
            }
            64 => {
                let stored = i64::from_be_bytes(sample.try_into().unwrap());
                if self.blank == Some(stored) {
                    f32::NAN
                } else if self.bscale == 1.0 && self.bzero == 9_223_372_036_854_775_808.0 {
                    ((stored as u64) ^ (1 << 63)) as f64 as f32
                } else {
                    (self.bzero + self.bscale * stored as f64) as f32
                }
            }
            -32 => {
                let value = f32::from_be_bytes(sample.try_into().unwrap());
                (self.bzero + self.bscale * f64::from(value)) as f32
            }
            -64 => {
                let value = f64::from_be_bytes(sample.try_into().unwrap());
                (self.bzero + self.bscale * value) as f32
            }
            _ => unreachable!("ImageSpec validates BITPIX"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FitsImage;

    fn card(keyword: &str, value: &str) -> String {
        format!("{keyword:<8}= {value:>20}")
    }

    /// One HDU: the cards and END, then the data, each padded to a block.
    fn unit(cards: &[String], data: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for text in cards.iter().map(String::as_str).chain(["END"]) {
            bytes.extend_from_slice(format!("{text:<80}").as_bytes());
        }
        bytes.resize(bytes.len().next_multiple_of(2880), b' ');
        bytes.extend_from_slice(data);
        bytes.resize(bytes.len().next_multiple_of(2880), 0);
        bytes
    }

    fn image_cards(primary: bool, bitpix: i64, axes: &[usize], extra: &[String]) -> Vec<String> {
        let mut cards = vec![if primary {
            card("SIMPLE", "T")
        } else {
            "XTENSION= 'IMAGE   '".to_string()
        }];
        cards.push(card("BITPIX", &bitpix.to_string()));
        cards.push(card("NAXIS", &axes.len().to_string()));
        for (index, length) in axes.iter().enumerate() {
            cards.push(card(&format!("NAXIS{}", index + 1), &length.to_string()));
        }
        if !primary {
            cards.extend([card("PCOUNT", "0"), card("GCOUNT", "1")]);
        }
        cards.extend_from_slice(extra);
        cards
    }

    /// Big-endian samples of `bitpix`, from a sample index.
    fn samples(bitpix: i64, count: usize, value: impl Fn(usize) -> f64) -> Vec<u8> {
        (0..count)
            .flat_map(|index| {
                let value = value(index);
                match bitpix {
                    8 => vec![value as u8],
                    16 => (value as i16).to_be_bytes().to_vec(),
                    32 => (value as i32).to_be_bytes().to_vec(),
                    64 => (value as i64).to_be_bytes().to_vec(),
                    -32 => (value as f32).to_be_bytes().to_vec(),
                    _ => value.to_be_bytes().to_vec(),
                }
            })
            .collect()
    }

    fn written(bytes: &[u8]) -> tempfile::NamedTempFile {
        let file = tempfile::Builder::new().suffix(".fits").tempfile().unwrap();
        std::fs::write(file.path(), bytes).unwrap();
        file
    }

    /// Every sample of `window` against the whole image, opened and
    /// converted to physical units, bit for bit.
    fn assert_matches_open(path: &Path, window: &ImageWindow) {
        let image = FitsImage::open(path).unwrap();
        let (width, height, planes) = (image.width, image.height, image.planes);
        assert_eq!(
            (window.image_width, window.image_height, window.planes),
            (width, height, planes)
        );
        let physical = image.into_physical_f32();
        let mut expected = Vec::new();
        for plane in 0..planes {
            for y in window.y..window.y + window.height {
                for x in window.x..window.x + window.width {
                    expected.push(physical[(plane * height + y) * width + x].to_bits());
                }
            }
        }
        let actual: Vec<u32> = window.values.iter().map(|value| value.to_bits()).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn every_sample_type_reads_as_the_whole_image_converts_it() {
        let (width, height) = (7, 5);
        let value = |index: usize| (index * 37 % 200) as f64 - 60.0;
        type Case<'a> = (i64, Vec<String>, Box<dyn Fn(usize) -> f64 + 'a>);
        let cases: Vec<Case> = vec![
            (8, vec![], Box::new(|index| (index * 37 % 250) as f64)),
            (
                8,
                vec![card("BZERO", "-128"), card("BLANK", "17")],
                Box::new(|index| (index * 17 % 250) as f64),
            ),
            (
                16,
                vec![card("BZERO", "32768"), card("BSCALE", "1")],
                Box::new(value),
            ),
            (
                16,
                vec![card("BZERO", "32768"), card("BLANK", "-23")],
                Box::new(value),
            ),
            (16, vec![], Box::new(|index| (index * 101) as f64)),
            (16, vec![], Box::new(value)),
            (
                16,
                vec![card("BZERO", "100.5"), card("BSCALE", "0.25")],
                Box::new(value),
            ),
            (
                32,
                vec![
                    card("BZERO", "7"),
                    card("BSCALE", "3"),
                    card("BLANK", "-60"),
                ],
                Box::new(value),
            ),
            (
                64,
                vec![card("BZERO", "9223372036854775808")],
                Box::new(value),
            ),
            (
                64,
                vec![card("BSCALE", "2"), card("BLANK", "-60")],
                Box::new(value),
            ),
            (
                -32,
                vec![card("BZERO", "10"), card("BSCALE", "2")],
                Box::new(|index| {
                    if index == 9 {
                        f64::NAN
                    } else {
                        value(index) / 7.0
                    }
                }),
            ),
            (-64, vec![], Box::new(|index| value(index) / 3.0)),
        ];
        for (bitpix, extra, value) in cases {
            let file = written(&unit(
                &image_cards(true, bitpix, &[width, height], &extra),
                &samples(bitpix, width * height, value),
            ));
            let window = read_window(file.path(), 2, 1, 4, 3).unwrap();
            assert_eq!(
                (window.x, window.y, window.width, window.height),
                (2, 1, 4, 3),
                "BITPIX {bitpix} {extra:?}"
            );
            assert_matches_open(file.path(), &window);
        }
    }

    #[test]
    fn a_blank_sample_is_nan() {
        let file = written(&unit(
            &image_cards(
                true,
                16,
                &[3, 2],
                &[card("BZERO", "32768"), card("BLANK", "-32768")],
            ),
            &samples(
                16,
                6,
                |index| if index == 4 { -32768.0 } else { index as f64 },
            ),
        ));
        let window = read_window(file.path(), 0, 0, 3, 2).unwrap();
        assert!(window.get(0, 1, 1).unwrap().is_nan());
        assert_eq!(window.get(0, 2, 1), Some(32773.0));
    }

    #[test]
    fn an_rgb_cube_gives_each_plane() {
        let file = written(&unit(
            &image_cards(true, -32, &[6, 4, 3], &[]),
            &samples(-32, 6 * 4 * 3, |index| index as f64 * 0.5),
        ));
        let window = read_window(file.path(), 1, 2, 3, 2).unwrap();
        assert_eq!(window.planes, 3);
        assert_eq!(window.values.len(), 3 * 3 * 2);
        assert_matches_open(file.path(), &window);
        // Plane 2, pixel (3, 3): the sample 2 × 24 + 3 × 6 + 3 in.
        assert_eq!(window.get(2, 3, 3), Some(69.0 * 0.5));
        assert_eq!(window.get(2, 0, 3), None);
    }

    #[test]
    fn a_window_is_clipped_to_the_image_and_one_outside_is_empty() {
        let file = written(&unit(
            &image_cards(true, 16, &[7, 5], &[card("BZERO", "32768")]),
            &samples(16, 35, |index| index as f64),
        ));
        let clipped = read_window(file.path(), 5, 3, 10, 10).unwrap();
        assert_eq!(
            (clipped.x, clipped.y, clipped.width, clipped.height),
            (5, 3, 2, 2)
        );
        assert_matches_open(file.path(), &clipped);

        for (x, y, width, height) in [(7, 0, 4, 4), (100, 100, 4, 4), (2, 2, 0, 3)] {
            let outside = read_window(file.path(), x, y, width, height).unwrap();
            assert_eq!((outside.width, outside.height), (0, 0));
            assert!(outside.values.is_empty());
            assert_eq!((outside.image_width, outside.image_height), (7, 5));
        }
    }

    #[test]
    fn an_image_in_an_extension_is_read_like_open_reads_it() {
        let primary = unit(
            &[card("SIMPLE", "T"), card("BITPIX", "8"), card("NAXIS", "0")],
            &[],
        );
        let extension = unit(
            &image_cards(false, 16, &[8, 6], &[card("BZERO", "32768")]),
            &samples(16, 48, |index| (index * 911 % 30000) as f64),
        );
        let file = written(&[primary, extension].concat());
        let window = read_window(file.path(), 3, 2, 4, 4).unwrap();
        assert_matches_open(file.path(), &window);
    }

    #[test]
    fn a_window_too_large_is_refused_and_a_truncated_file_is_malformed() {
        let file = written(&unit(
            &image_cards(true, -32, &[4, 4], &[]),
            &samples(-32, 16, |i| i as f64),
        ));
        assert!(matches!(
            read_window(file.path(), 0, 0, MAX_WINDOW_EDGE + 1, 1),
            Err(FitsError::Unsupported(_))
        ));
        assert!(read_window(file.path(), 0, 0, MAX_WINDOW_EDGE, MAX_WINDOW_EDGE).is_ok());

        let mut bytes = unit(&image_cards(true, -32, &[400, 400], &[]), &[]);
        bytes.extend(samples(-32, 1000, |i| i as f64));
        let truncated = written(&bytes);
        assert!(read_window(truncated.path(), 0, 0, 4, 2).is_ok());
        assert!(matches!(
            read_window(truncated.path(), 0, 300, 4, 2),
            Err(FitsError::Malformed(_))
        ));
    }
}
