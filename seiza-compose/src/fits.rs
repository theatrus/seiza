//! Writing a composition as FITS.

use crate::{ColorComposition, Error, Result};
use seiza_fits::{HeaderValue, WriteHeaderCard};
use seiza_stacking::write_linear_image_fits_f32;
use std::path::Path;

fn text_card(key: &str, value: &str, comment: &str) -> WriteHeaderCard {
    WriteHeaderCard::new(key, HeaderValue::String(value.into())).with_comment(comment)
}

fn integer_card(key: &str, value: i64, comment: &str) -> WriteHeaderCard {
    WriteHeaderCard::new(key, HeaderValue::Integer(value)).with_comment(comment)
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
        text_card("COLORSPC", "RGB", "RGB color planes"),
        text_card("SEIZACLR", label, "Seiza color composition"),
        text_card(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ColorTransfer, LinearImage, ReferenceRegion};
    use seiza_fits::FitsImage;

    #[test]
    fn cropped_color_output_moves_crpix_to_its_own_grid() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("cropped.fits");
        let composition = ColorComposition {
            image: LinearImage::new(2, 2, 3, vec![0.5; 12]).unwrap(),
            transfer: ColorTransfer::LinearLight,
            region: ReferenceRegion {
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
            transfer: ColorTransfer::LinearLight,
            region: ReferenceRegion {
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
}
