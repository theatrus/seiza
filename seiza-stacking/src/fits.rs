use crate::{
    BayerLayout, ColorComposition, Error, LinearImage, MasterFrame, Result, StackSnapshot,
};
use seiza_fits::{F32ImageData, FitsImage, HeaderValue, Pixels, WriteHeaderCard};
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

    /// Reject a Seiza master whose declared kind does not match its use.
    /// External masters without `SEIZAMST` retain the legacy inferred behavior.
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
        Ok(())
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
    let mut cards = vec![integer_card(
        "STACKCNT",
        snapshot.accepted_frames as i64,
        "accepted input frames",
    )];
    cards.push(integer_card(
        "STACKREJ",
        snapshot.rejected_frames as i64,
        "rejected input frames",
    ));
    write_linear_image_fits_f32(path, &snapshot.image, reference_headers, &cards)
}

/// Write a composed RGB image as primary-HDU 32-bit floating-point FITS.
///
/// `label` identifies the composition (for example `LRGB`, `SHO`, or
/// `FORAXX-SHO`). WCS cards are copied from the chosen aligned reference.
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
    let cards = vec![
        string_card("COLORSPC", "RGB", "RGB color planes"),
        string_card("SEIZACLR", label, "Seiza color composition"),
        string_card(
            "SEIZATRF",
            composition.transfer.fits_name(),
            "sample transfer semantics",
        ),
    ];
    write_linear_image_fits_f32(path, &composition.image, reference_headers, &cards)
}

/// Write an integrated calibration master with explicit calibration-state headers.
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
        float_card(
            "CLIPLOW",
            f64::from(master.rejection.low_sigma),
            "low leave-one-out sigma threshold",
        ),
        float_card(
            "CLIPHIGH",
            f64::from(master.rejection.high_sigma),
            "high leave-one-out sigma threshold",
        ),
        integer_card(
            "CLIPREJ",
            i64::try_from(master.rejected_samples).unwrap_or(i64::MAX),
            "rejected input samples",
        ),
    ];
    if let Some(exposure_seconds) = master.exposure_seconds {
        cards.push(float_card(
            "EXPTIME",
            exposure_seconds,
            "master dark exposure seconds",
        ));
    }
    if let Some(bayer) = master.bayer {
        cards.push(string_card(
            "BAYERPAT",
            bayer.pattern.as_str(),
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
    for (key, value) in &master.reference_headers {
        if preserve_master_key(key)
            && !cards.iter().any(|card| card.keyword() == key)
            && let Some(card) = value_card(key, value)
        {
            cards.push(card);
        }
    }
    write_linear_fits_f32(path.as_ref(), &master.image, cards)
}

/// Write a linear image while preserving a valid WCS from a reference frame.
///
/// `extra_cards` describes the processing operation. Structural FITS cards
/// are generated by the writer, and duplicate WCS cards in `extra_cards` take
/// precedence over the reference.
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
/// and observation/instrument metadata while replacing structural pixel cards.
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
    seiza_fits::write_f32_image(path, image.width, image.height, pixels, &extra_cards).map_err(
        |source| Error::FitsWrite {
            path: path.to_path_buf(),
            source,
        },
    )?;
    Ok(())
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

fn preserve_master_key(key: &str) -> bool {
    matches!(
        key,
        "INSTRUME"
            | "CAMERA"
            | "XBINNING"
            | "YBINNING"
            | "CCDXBIN"
            | "CCDYBIN"
            | "XPIXSZ"
            | "YPIXSZ"
            | "GAIN"
            | "EGAIN"
            | "OFFSET"
            | "CCD-TEMP"
            | "SET-TEMP"
            | "READOUTM"
            | "FILTER"
            | "BAYERPAT"
            | "XBAYROFF"
            | "YBAYROFF"
    )
}

fn preserve_processed_key(key: &str) -> bool {
    preserve_master_key(key)
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
        ];
        write_processed_image_fits_f32(&path, &image, &reference_headers, &[]).unwrap();
        let decoded = FitsImage::open(&path).unwrap();
        assert_eq!(decoded.header_f64("BITPIX"), Some(-32.0));
        assert_eq!(decoded.header_str("DATE-OBS"), Some("2026-01-02T03:04:05Z"));
        assert_eq!(decoded.header_str("FILTER"), Some("H-alpha"));
        assert_eq!(decoded.header_str("OBJECT"), Some("Sh2-132"));
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
        ];
        write_processed_image_fits_f32(&path, &image, &reference_headers, &[]).unwrap();
        let decoded = FitsImage::open(&path).unwrap();
        assert_eq!(decoded.planes, 3);
        assert!(decoded.header("BAYERPAT").is_none());
        assert!(decoded.header("XBAYROFF").is_none());
        assert!(decoded.header("YBAYROFF").is_none());
    }

    #[test]
    fn master_writer_round_trips_calibration_state() {
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
            input_statistics: Vec::new(),
            bias_subtracted: true,
            dark_subtracted: false,
            normalized: false,
            rejection: crate::MasterRejectionOptions::default(),
            reference_headers: vec![("INSTRUME".into(), HeaderValue::String("Test Camera".into()))],
        };
        write_master_fits_f32(&path, &master).unwrap();
        let decoded = FitsImage::open(&path).unwrap();
        assert_eq!(decoded.header_str("SEIZAMST"), Some("DARK"));
        assert_eq!(decoded.header_f64("NCOMBINE"), Some(12.0));
        assert_eq!(decoded.header_f64("EXPTIME"), Some(30.0));
        assert_eq!(
            decoded.header("BIASSUB").and_then(HeaderValue::as_bool),
            Some(true)
        );
        assert_eq!(decoded.header_str("INSTRUME"), Some("Test Camera"));
        let frame = FitsFrame::open(&path).unwrap();
        assert_eq!(frame.bayer, master.bayer);
        frame.validate_master_kind("DARK").unwrap();
        assert!(frame.validate_master_kind("BIAS").is_err());
    }
}
