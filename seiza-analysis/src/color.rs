use seiza_fits::{FitsImage, Pixels};
use serde::{Deserialize, Serialize};

/// Mapping from a measured component grid to original stored image pixels.
/// No interpolation, channel averaging or white balance is applied here.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamplingInfo {
    /// mono, cfa, or rgb. RGB does not imply raw or statistically independent pixels.
    pub image_kind: String,
    /// Mono, R, G, B, G_red (green on red row), or G_blue (green on blue row).
    pub component: String,
    pub source_width: usize,
    pub source_height: usize,
    /// Original-image pixels between adjacent component samples (2 for CFA).
    pub spacing_pixels: u32,
    /// Original stored pixel occupied by component sample (0, 0).
    pub offset_pixels: [usize; 2],
    /// Declared BAYERPAT, retained separately from row-order conversion.
    pub bayer_pattern: Option<String>,
    /// Pattern in stored row order, before origin offsets.
    pub storage_bayer_pattern: Option<String>,
    /// Declared offsets; parity is used when selecting the CFA lattice.
    pub bayer_offsets: Option<[usize; 2]>,
    pub row_order: Option<String>,
}

pub(crate) fn split(image: FitsImage) -> Result<Vec<(FitsImage, SamplingInfo)>, String> {
    let (w, h) = (image.width, image.height);
    let declared = image.header_str("BAYERPAT").map(str::to_owned);
    let pattern = image.bayer_pattern_in_storage_order();
    let row_order = image.header_str("ROWORDER").map(str::to_owned);
    if !matches!(image.planes, 1 | 3) {
        return Err("analysis supports mono/CFA or three-plane RGB images".into());
    }
    if image.planes == 1 && image.header("BAYERPAT").is_some() && pattern.is_none() {
        return Err(
            "unrecognized BAYERPAT: refusing to analyze a possible CFA mosaic as mono".into(),
        );
    }
    let mut layout = SamplingInfo {
        image_kind: "mono".into(),
        component: "Mono".into(),
        source_width: w,
        source_height: h,
        spacing_pixels: 1,
        offset_pixels: [0, 0],
        bayer_pattern: declared,
        storage_bayer_pattern: pattern.map(|p| p.as_str().into()),
        bayer_offsets: None,
        row_order,
    };
    if image.planes == 1 && pattern.is_none() {
        return Ok(vec![(image, layout)]);
    }
    let rgb = image.planes == 3;
    let offsets = if rgb {
        [0, 0]
    } else {
        let mut offsets = [0; 2];
        for (i, key) in ["XBAYROFF", "YBAYROFF"].iter().enumerate() {
            let value = match image.header_f64(key) {
                Some(v) => v,
                None if image.header(key).is_some() => {
                    return Err(format!(
                        "invalid {key}: expected a numeric CFA origin offset"
                    ));
                }
                None => 0.0,
            };
            if !value.is_finite()
                || value < 0.0
                || value.fract() != 0.0
                || value >= usize::MAX as f64
            {
                return Err(format!(
                    "invalid {key}: expected a nonnegative integer CFA origin offset"
                ));
            }
            offsets[i] = value as usize;
        }
        offsets
    };
    layout.image_kind = if rgb { "rgb" } else { "cfa" }.into();
    layout.spacing_pixels = if rgb { 1 } else { 2 };
    if !rgb {
        layout.bayer_offsets = Some(offsets);
    }
    let headers = image
        .headers
        .iter()
        .filter(|(k, _)| {
            !matches!(
                k.as_str(),
                "BAYERPAT" | "XBAYROFF" | "YBAYROFF" | "BZERO" | "BSCALE" | "BLANK"
            )
        })
        .cloned()
        .collect::<Vec<_>>();
    let data = image.into_physical_f32();
    let mut out = Vec::new();
    if rgb {
        for (c, name) in ["R", "G", "B"].iter().enumerate() {
            let mut info = layout.clone();
            info.component = (*name).into();
            out.push((
                FitsImage {
                    width: w,
                    height: h,
                    planes: 1,
                    pixels: Pixels::F32(data[c * w * h..(c + 1) * w * h].to_vec()),
                    headers: headers.clone(),
                },
                info,
            ));
        }
    } else {
        let pattern = pattern.unwrap().as_str().as_bytes();
        for name in ["R", "G_red", "G_blue", "B"] {
            let (x, y) = (0..2)
                .flat_map(|y| (0..2).map(move |x| (x, y)))
                .find(|&(x, y)| {
                    let yy = (y + (offsets[1] & 1)) & 1;
                    let xx = (x + (offsets[0] & 1)) & 1;
                    let color = pattern[yy * 2 + xx];
                    match name {
                        "R" => color == b'R',
                        "B" => color == b'B',
                        "G_red" => color == b'G' && pattern[yy * 2 + (1 - xx)] == b'R',
                        _ => color == b'G' && pattern[yy * 2 + (1 - xx)] == b'B',
                    }
                })
                .unwrap();
            let (cw, ch) = ((w + 1 - x) / 2, (h + 1 - y) / 2);
            let mut samples = Vec::with_capacity(cw * ch);
            for cy in 0..ch {
                for cx in 0..cw {
                    samples.push(data[(2 * cy + y) * w + 2 * cx + x]);
                }
            }
            let mut info = layout.clone();
            info.component = name.into();
            info.offset_pixels = [x, y];
            out.push((
                FitsImage {
                    width: cw,
                    height: ch,
                    planes: 1,
                    pixels: Pixels::F32(samples),
                    headers: headers.clone(),
                },
                info,
            ));
        }
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    use seiza_fits::{BayerPattern, HeaderValue};
    #[test]
    fn cfa_lattices_preserve_every_native_sample_and_metadata() {
        for pattern in ["RGGB", "BGGR", "GRBG", "GBRG"] {
            for height in [8, 9] {
                for bottom in [false, true] {
                    for xoff in [0, 1] {
                        for yoff in [0, 1] {
                            let width = 9;
                            let mut headers = vec![
                                ("BAYERPAT".into(), HeaderValue::String(pattern.into())),
                                ("XBAYROFF".into(), HeaderValue::Integer(xoff)),
                                ("YBAYROFF".into(), HeaderValue::Integer(yoff)),
                            ];
                            if bottom {
                                headers.push((
                                    "ROWORDER".into(),
                                    HeaderValue::String("BOTTOM-UP".into()),
                                ));
                            }
                            let image = FitsImage {
                                width,
                                height,
                                planes: 1,
                                pixels: Pixels::F32(
                                    (0..width * height).map(|i| i as f32).collect(),
                                ),
                                headers,
                            };
                            let storage = image.bayer_pattern_in_storage_order().unwrap();
                            let planes = split(image).unwrap();
                            assert_eq!(planes.len(), 4);
                            let mut visited = std::collections::HashSet::new();
                            for (plane, info) in planes {
                                assert_eq!(info.spacing_pixels, 2);
                                assert_eq!(info.bayer_pattern.as_deref(), Some(pattern));
                                assert_eq!(
                                    info.storage_bayer_pattern.as_deref(),
                                    Some(storage.as_str())
                                );
                                assert_eq!(
                                    info.bayer_offsets,
                                    Some([xoff as usize, yoff as usize])
                                );
                                let [x, y] = info.offset_pixels;
                                let bytes = storage.as_str().as_bytes();
                                let color =
                                    bytes[((y + yoff as usize) % 2) * 2 + (x + xoff as usize) % 2];
                                assert_eq!(color, info.component.as_bytes()[0]);
                                let Pixels::F32(values) = plane.pixels else {
                                    panic!()
                                };
                                for (i, v) in values.iter().enumerate() {
                                    let source = (2 * (i / plane.width) + y) * width
                                        + 2 * (i % plane.width)
                                        + x;
                                    assert_eq!(*v, source as f32);
                                    assert!(visited.insert(source));
                                }
                            }
                            assert_eq!(visited.len(), width * height);
                        }
                    }
                }
            }
            assert!(BayerPattern::parse(pattern).is_some());
        }
    }
    #[test]
    fn rgb_is_planar_and_never_averaged_or_debayered() {
        let source = FitsImage {
            width: 2,
            height: 2,
            planes: 3,
            pixels: Pixels::F32((0..12).map(|i| i as f32).collect()),
            headers: vec![("BAYERPAT".into(), HeaderValue::String("RGGB".into()))],
        };
        for (c, (image, info)) in split(source).unwrap().into_iter().enumerate() {
            assert_eq!(info.image_kind, "rgb");
            assert_eq!(info.spacing_pixels, 1);
            let Pixels::F32(values) = image.pixels else {
                panic!()
            };
            assert_eq!(
                values,
                (c * 4..c * 4 + 4).map(|v| v as f32).collect::<Vec<_>>()
            );
        }
    }
    #[test]
    fn unknown_pattern_and_invalid_offsets_are_not_silently_mono() {
        let image = |key: &str, value| FitsImage {
            width: 4,
            height: 4,
            planes: 1,
            pixels: Pixels::F32(vec![1.; 16]),
            headers: vec![
                ("BAYERPAT".into(), HeaderValue::String("RGGB".into())),
                (key.into(), value),
            ],
        };
        assert!(split(image("XBAYROFF", HeaderValue::Float(-1.))).is_err());
        assert!(split(image("YBAYROFF", HeaderValue::Float(0.5))).is_err());
        let mut bad = image("XBAYROFF", HeaderValue::Integer(0));
        bad.headers[0].1 = HeaderValue::String("XTRANS".into());
        assert!(split(bad).is_err());
    }
}
