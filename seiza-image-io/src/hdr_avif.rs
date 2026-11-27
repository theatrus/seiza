//! HDR AVIF: a picture above SDR white kept as HDR, for displays that show
//! it.
//!
//! [`save_hdr_avif`] writes linear Rec. 2020 samples as a 10-bit AVIF with
//! the PQ transfer (SMPTE ST 2084): what HDR10 stills and Apple's,
//! Google's and Microsoft's HDR viewers read. A sample of 1 is SDR
//! reference white, 203 cd/m² (ITU-R BT.2408), so an SDR picture's white is
//! the same white in the HDR file, and a sample of 4 is two stops above it.

use crate::Error;
use std::path::Path;

/// SDR reference white in cd/m², as ITU-R BT.2408 places it in PQ.
pub const SDR_WHITE_NITS: f32 = 203.0;

/// How [`save_hdr_avif`] encodes.
#[derive(Clone, Copy, Debug)]
pub struct HdrAvifOptions {
    /// From 1 (smallest file) to 100 (closest to the samples).
    pub quality: f32,
    /// rav1e's speed, from 0 (slowest, smallest) to 10 (fastest).
    pub speed: u8,
}

impl Default for HdrAvifOptions {
    fn default() -> Self {
        Self {
            quality: 90.0,
            speed: 6,
        }
    }
}

/// The PQ (SMPTE ST 2084) signal of a luminance in cd/m², from 0 to 1.
pub fn pq_encode(nits: f32) -> f32 {
    const M1: f64 = 2610.0 / 16384.0;
    const M2: f64 = 2523.0 / 4096.0 * 128.0;
    const C1: f64 = 3424.0 / 4096.0;
    const C2: f64 = 2413.0 / 4096.0 * 32.0;
    const C3: f64 = 2392.0 / 4096.0 * 32.0;
    let y = (f64::from(nits) / 10_000.0).clamp(0.0, 1.0).powf(M1);
    (((C1 + C2 * y) / (1.0 + C3 * y)).powf(M2)) as f32
}

/// Writes `rgb`, `width` by `height` pixels of linear Rec. 2020 red, green
/// and blue interleaved, where 1 is SDR reference white, as a 10-bit HDR
/// AVIF: PQ transfer, Rec. 2020 primaries and matrix, full range, chroma at
/// full resolution, with the brightest pixel and the mean recorded as its
/// content light levels. Written atomically.
pub fn save_hdr_avif(
    path: &Path,
    width: usize,
    height: usize,
    rgb: &[f32],
    options: &HdrAvifOptions,
) -> Result<(), Error> {
    if width == 0 || height == 0 || rgb.len() != width * height * 3 {
        return Err(Error::save(path, "its samples do not fill its size"));
    }
    let maximum = f32::from(1023_u16);
    let (mut brightest, mut total) = (0.0_f32, 0.0_f64);
    let mut planes = [
        vec![0_u16; width * height],
        vec![0_u16; width * height],
        vec![0_u16; width * height],
    ];
    for (at, pixel) in rgb.as_chunks::<3>().0.iter().enumerate() {
        let linear = pixel.map(|value| {
            if value.is_finite() {
                value.max(0.0)
            } else {
                0.0
            }
        });
        let nits = linear.map(|value| value * SDR_WHITE_NITS);
        let light = nits[0].max(nits[1]).max(nits[2]);
        brightest = brightest.max(light);
        total += f64::from(light);
        let [r, g, b] = nits.map(pq_encode);
        // Rec. 2020 non-constant luminance.
        let luma = 0.2627 * r + 0.6780 * g + 0.0593 * b;
        let blue = (b - luma) / 1.8814;
        let red = (r - luma) / 1.4746;
        planes[0][at] = (luma * maximum).round().clamp(0.0, maximum) as u16;
        planes[1][at] = ((blue + 0.5) * maximum).round().clamp(0.0, maximum) as u16;
        planes[2][at] = ((red + 0.5) * maximum).round().clamp(0.0, maximum) as u16;
    }
    let average = (total / (width * height) as f64) as f32;
    let light_levels = (
        brightest.round().clamp(0.0, 10_000.0) as u16,
        average.round().clamp(0.0, 10_000.0) as u16,
    );
    let av1 = encode_av1(width, height, &planes, light_levels, options)
        .map_err(|message| Error::save(path, message))?;
    let file = avif_serialize::Aviffy::new()
        .set_color_primaries(avif_serialize::constants::ColorPrimaries::Bt2020)
        .set_transfer_characteristics(avif_serialize::constants::TransferCharacteristics::Smpte2084)
        .set_matrix_coefficients(avif_serialize::constants::MatrixCoefficients::Bt2020Ncl)
        .set_full_color_range(true)
        .set_chroma_subsampling((false, false))
        .set_content_light_level(light_levels.0, light_levels.1)
        .to_vec(&av1, None, width as u32, height as u32, 10);
    crate::write_bytes_atomic(path, &file)
}

/// One 10-bit, 4:4:4 AV1 still from Y, Cb and Cr planes.
fn encode_av1(
    width: usize,
    height: usize,
    planes: &[Vec<u16>; 3],
    (max_light, average_light): (u16, u16),
    options: &HdrAvifOptions,
) -> Result<Vec<u8>, String> {
    use rav1e::prelude::*;
    let quality = options.quality.clamp(1.0, 100.0);
    // rav1e's quantizer runs from 0 (best) to 255.
    let quantizer = ((1.0 - quality / 100.0) * 255.0).round() as usize;
    let mut encoder = EncoderConfig::with_speed_preset(options.speed.min(10));
    encoder.width = width;
    encoder.height = height;
    encoder.bit_depth = 10;
    encoder.chroma_sampling = ChromaSampling::Cs444;
    encoder.pixel_range = PixelRange::Full;
    encoder.still_picture = true;
    encoder.quantizer = quantizer;
    encoder.min_quantizer = quantizer.saturating_sub(10) as u8;
    encoder.color_description = Some(ColorDescription {
        color_primaries: ColorPrimaries::BT2020,
        transfer_characteristics: TransferCharacteristics::SMPTE2084,
        matrix_coefficients: MatrixCoefficients::BT2020NCL,
    });
    encoder.content_light = Some(ContentLight {
        max_content_light_level: max_light,
        max_frame_average_light_level: average_light,
    });
    let config = Config::new().with_encoder_config(encoder);
    let mut context: Context<u16> = config
        .new_context()
        .map_err(|error| format!("the AV1 encoder refused its settings: {error}"))?;
    let mut frame = context.new_frame();
    for (plane, samples) in frame.planes.iter_mut().zip(planes) {
        let bytes: Vec<u8> = samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect();
        plane.copy_from_raw_u8(&bytes, width * 2, 2);
    }
    context
        .send_frame(frame)
        .map_err(|error| format!("the AV1 encoder did not take the picture: {error}"))?;
    context.flush();
    let mut data = Vec::new();
    loop {
        match context.receive_packet() {
            Ok(packet) => data.extend_from_slice(&packet.data),
            Err(EncoderStatus::Encoded) | Err(EncoderStatus::NeedMoreData) => {}
            Err(EncoderStatus::LimitReached) => break,
            Err(error) => return Err(format!("the AV1 encoder failed: {error}")),
        }
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pq_places_sdr_white_and_its_ends() {
        assert!(pq_encode(0.0) < 1e-6);
        assert!((pq_encode(10_000.0) - 1.0).abs() < 1e-6);
        // 203 cd/m² is about 58% of the PQ signal.
        assert!((pq_encode(SDR_WHITE_NITS) - 0.5807).abs() < 0.002);
    }

    #[test]
    fn an_hdr_avif_says_it_is_pq_in_rec_2020() {
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("picture.avif");
        let (width, height) = (32, 24);
        // A gray ramp from black to two stops above SDR white.
        let rgb: Vec<f32> = (0..width * height)
            .flat_map(|at| [4.0 * (at % width) as f32 / (width - 1) as f32; 3])
            .collect();
        save_hdr_avif(&path, width, height, &rgb, &HdrAvifOptions::default()).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[4..8], b"ftyp");
        // The colr box: nclx, primaries 9 (BT.2020), transfer 16 (PQ),
        // matrix 9 (BT.2020 NCL), full range.
        let colr = bytes
            .windows(4)
            .position(|window| window == b"colr")
            .unwrap();
        assert_eq!(&bytes[colr + 4..colr + 8], b"nclx");
        assert_eq!(&bytes[colr + 8..colr + 15], &[0, 9, 0, 16, 0, 9, 0x80]);
        // The content light level: the brightest is four times SDR white.
        let clli = bytes
            .windows(4)
            .position(|window| window == b"clli")
            .unwrap();
        let max_light = u16::from_be_bytes([bytes[clli + 4], bytes[clli + 5]]);
        assert_eq!(max_light, (4.0 * SDR_WHITE_NITS).round() as u16);
    }
}
