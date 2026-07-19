use crate::{BayerLayout, Error, LinearImage, Result, StackSnapshot};
use seiza_fits::{FitsImage, HeaderValue, Pixels};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// A FITS frame decoded into linear, un-stretched `f32` samples.
#[derive(Clone, Debug)]
pub struct FitsFrame {
    pub image: LinearImage,
    pub headers: Vec<(String, HeaderValue)>,
    pub exposure_seconds: Option<f64>,
    pub bayer: Option<BayerLayout>,
    pub source: Option<PathBuf>,
}

impl FitsFrame {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let fits = FitsImage::open(path).map_err(|source| Error::FitsRead {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_fits(fits, Some(path.to_path_buf()))
    }

    pub fn from_fits(fits: FitsImage, source: Option<PathBuf>) -> Result<Self> {
        let bitpix = fits
            .header("BITPIX")
            .and_then(HeaderValue::as_i64)
            .unwrap_or(0);
        let bzero = fits.header_f64("BZERO").unwrap_or(0.0);
        let bscale = fits.header_f64("BSCALE").unwrap_or(1.0);
        let bayer_pattern = fits.bayer_pattern();
        let x_offset = fits.header_f64("XBAYROFF").unwrap_or(0.0).max(0.0) as usize;
        let y_offset = fits.header_f64("YBAYROFF").unwrap_or(0.0).max(0.0) as usize;
        let exposure_seconds = ["XPOSURE", "EXPTIME", "EXPOSURE"]
            .iter()
            .find_map(|key| fits.header_f64(key))
            .filter(|value| value.is_finite() && *value > 0.0);
        let physical = match fits.pixels {
            Pixels::U8(values) => values
                .into_iter()
                .map(|value| (bzero + bscale * f64::from(value)) as f32)
                .collect(),
            // seiza-fits has already applied the standard unsigned-camera BZERO.
            Pixels::U16(values) => values.into_iter().map(f32::from).collect(),
            Pixels::I32(values) => values
                .into_iter()
                .map(|value| (bzero + bscale * f64::from(value)) as f32)
                .collect(),
            // A BITPIX=16 image with unusual scaling is decoded to F32 by
            // seiza-fits and is already in physical units.
            Pixels::F32(values) if bitpix == 16 => values,
            Pixels::F32(values) => values
                .into_iter()
                .map(|value| (bzero + bscale * f64::from(value)) as f32)
                .collect(),
            Pixels::F64(values) => values
                .into_iter()
                .map(|value| (bzero + bscale * value) as f32)
                .collect(),
        };

        let channels = if fits.planes == 3 { 3 } else { 1 };
        let data = if channels == 3 {
            planar_to_interleaved(&physical, fits.width * fits.height)
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
            image: LinearImage::new(fits.width, fits.height, channels, data)?,
            headers: fits.headers,
            exposure_seconds,
            bayer,
            source,
        })
    }

    pub(crate) fn into_prepared(mut self) -> Result<Self> {
        if let Some(layout) = self.bayer.take() {
            self.image = self.image.debayer(layout)?;
        }
        Ok(self)
    }
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

/// Write an unstretched stack as a primary-HDU 32-bit floating-point FITS.
pub fn write_fits_f32(
    path: impl AsRef<Path>,
    snapshot: &StackSnapshot,
    reference_headers: &[(String, HeaderValue)],
) -> Result<()> {
    let path = path.as_ref();
    let file = std::fs::File::create(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut writer = BufWriter::new(file);
    let image = &snapshot.image;
    let mut cards = vec![
        logical_card("SIMPLE", true, "conforms to FITS standard"),
        integer_card("BITPIX", -32, "32-bit IEEE floating point"),
        integer_card("NAXIS", if image.channels == 3 { 3 } else { 2 }, ""),
        integer_card("NAXIS1", image.width as i64, ""),
        integer_card("NAXIS2", image.height as i64, ""),
    ];
    if image.channels == 3 {
        cards.push(integer_card("NAXIS3", 3, "RGB planes"));
    }
    cards.push(logical_card("EXTEND", true, "extensions may be present"));
    cards.push(integer_card(
        "STACKCNT",
        snapshot.accepted_frames as i64,
        "accepted input frames",
    ));
    cards.push(integer_card(
        "STACKREJ",
        snapshot.rejected_frames as i64,
        "rejected input frames",
    ));
    if has_reference_wcs(reference_headers) {
        for (key, value) in reference_headers {
            if preserve_wcs_key(key)
                && !cards
                    .iter()
                    .any(|card| card.starts_with(&format!("{key:<8}")))
                && let Some(card) = value_card(key, value)
            {
                cards.push(card);
            }
        }
    }
    cards.push(format!("{:<80}", "END"));
    write_block_padded(&mut writer, cards.concat().as_bytes(), b' ').map_err(|source| {
        Error::Io {
            path: path.to_path_buf(),
            source,
        }
    })?;

    let mut byte_buffer = Vec::with_capacity(1024 * 1024);
    if image.channels == 3 {
        for channel in 0..3 {
            write_float_values(
                &mut writer,
                (0..image.pixel_count()).map(|index| image.data[index * 3 + channel]),
                &mut byte_buffer,
            )
            .map_err(|source| Error::Io {
                path: path.to_path_buf(),
                source,
            })?;
        }
    } else {
        write_float_values(&mut writer, image.data.iter().copied(), &mut byte_buffer).map_err(
            |source| Error::Io {
                path: path.to_path_buf(),
                source,
            },
        )?;
    }
    let byte_len = image.sample_count() * 4;
    let padding = (2880 - byte_len % 2880) % 2880;
    writer
        .write_all(&vec![0; padding])
        .map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
    writer.flush().map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn write_float_values(
    writer: &mut impl Write,
    values: impl Iterator<Item = f32>,
    buffer: &mut Vec<u8>,
) -> std::io::Result<()> {
    buffer.clear();
    for value in values {
        buffer.extend_from_slice(&value.to_be_bytes());
        if buffer.len() == buffer.capacity() {
            writer.write_all(buffer)?;
            buffer.clear();
        }
    }
    writer.write_all(buffer)?;
    buffer.clear();
    Ok(())
}

fn write_block_padded(writer: &mut impl Write, bytes: &[u8], padding: u8) -> std::io::Result<()> {
    writer.write_all(bytes)?;
    let count = (2880 - bytes.len() % 2880) % 2880;
    writer.write_all(&vec![padding; count])
}

fn card(key: &str, value: &str, comment: &str) -> String {
    let text = if comment.is_empty() {
        format!("{key:<8}= {value:>20}")
    } else {
        format!("{key:<8}= {value:>20} / {comment}")
    };
    format!("{:<80}", text.chars().take(80).collect::<String>())
}

fn logical_card(key: &str, value: bool, comment: &str) -> String {
    card(key, if value { "T" } else { "F" }, comment)
}

fn integer_card(key: &str, value: i64, comment: &str) -> String {
    card(key, &value.to_string(), comment)
}

fn value_card(key: &str, value: &HeaderValue) -> Option<String> {
    let value = match value {
        HeaderValue::Logical(value) => {
            if *value {
                "T".into()
            } else {
                "F".into()
            }
        }
        HeaderValue::Integer(value) => value.to_string(),
        HeaderValue::Float(value) if value.is_finite() => format!("{value:.12E}"),
        HeaderValue::String(value) => format!("'{}'", value.replace('\'', "''")),
        HeaderValue::Raw(value) if !value.is_empty() => value.clone(),
        _ => return None,
    };
    Some(card(key, &value, "copied from reference frame"))
}

fn preserve_wcs_key(key: &str) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(bitpix: i64) -> Vec<(String, HeaderValue)> {
        vec![("BITPIX".into(), HeaderValue::Integer(bitpix))]
    }

    #[test]
    fn planar_color_becomes_interleaved() {
        assert_eq!(
            planar_to_interleaved(&[1.0, 2.0, 10.0, 20.0, 100.0, 200.0], 2),
            [1.0, 10.0, 100.0, 2.0, 20.0, 200.0]
        );
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
}
