//! Lights binned by [`StackOptions::bin_factor`], and their reintegration at
//! the sensor's full resolution.

use super::*;
use crate::BatchStackOptions;
use seiza_fits::{F32ImageData, write_f32_image};

const WIDTH: usize = 384;
const HEIGHT: usize = 288;

/// Frame `index`'s view of the sky: a few pixels of dither and a small turn
/// about the frame's centre, and a half turn more after a meridian flip.
fn pose(index: usize, flipped: bool) -> SimilarityTransform {
    let turn = 0.004 * index as f64 + if flipped { std::f64::consts::PI } else { 0.0 };
    let (centre_x, centre_y) = ((WIDTH - 1) as f64 / 2.0, (HEIGHT - 1) as f64 / 2.0);
    let (dither_x, dither_y) = (
        ((index * 13) % 7) as f64 - 3.0 + 0.37 * index as f64,
        ((index * 7) % 5) as f64 - 2.0 - 0.21 * index as f64,
    );
    let (cosine, sine) = (turn.cos(), turn.sin());
    SimilarityTransform {
        scale: 1.0,
        rotation_radians: turn,
        translation_x: centre_x + dither_x - (cosine * centre_x - sine * centre_y),
        translation_y: centre_y + dither_y - (sine * centre_x + cosine * centre_y),
    }
}

/// `count` frames of one field of stars 3.5 pixels across at half maximum,
/// on a sky of 500 with noise, posed by [`pose`]; frame `flipped` after a
/// meridian flip. Returns each frame's path and pose.
fn field(
    directory: &Path,
    count: usize,
    flipped: Option<usize>,
) -> Vec<(PathBuf, SimilarityTransform)> {
    let mut seed = 0x0b1e_u64;
    let mut random = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f64 / (1u64 << 31) as f64
    };
    let margin = 60.0;
    let stars = (0..WIDTH * HEIGHT / 220)
        .map(|_| {
            (
                -margin + random() * (WIDTH as f64 + 2.0 * margin),
                -margin + random() * (HEIGHT as f64 + 2.0 * margin),
                150.0 + 9000.0 * random().powi(3),
            )
        })
        .collect::<Vec<_>>();
    let sigma = 3.5 / 2.354_820_045;
    (0..count)
        .map(|index| {
            let pose = pose(index, flipped == Some(index));
            let mut data = vec![500.0_f32; WIDTH * HEIGHT];
            for value in &mut data {
                let normal = (0..4).map(|_| random()).sum::<f64>() - 2.0;
                *value += (normal * 10.0) as f32;
            }
            for &(sky_x, sky_y, peak) in &stars {
                let (x, y) = pose.apply(sky_x, sky_y);
                let (left, top) = (x.floor() as isize - 6, y.floor() as isize - 6);
                for row in top.max(0)..(top + 14).min(HEIGHT as isize) {
                    for column in left.max(0)..(left + 14).min(WIDTH as isize) {
                        let r2 = (column as f64 - x).powi(2) + (row as f64 - y).powi(2);
                        data[row as usize * WIDTH + column as usize] +=
                            (peak * (-r2 / (2.0 * sigma * sigma)).exp()) as f32;
                    }
                }
            }
            let path = directory.join(format!("light-{index}.fits"));
            write_f32_image(&path, WIDTH, HEIGHT, F32ImageData::Mono(&data), &[]).unwrap();
            (path, pose)
        })
        .collect()
}

/// A stack of `frames`, every one admitted, with each frame's diagnostics.
fn stack(
    frames: &[(PathBuf, SimilarityTransform)],
    options: StackOptions,
) -> (LiveStacker, Vec<FrameDiagnostics>) {
    let mut stacker =
        LiveStacker::open_fits(&frames[0].0, None, None, None, None, options).unwrap();
    let diagnostics = frames[1..]
        .iter()
        .map(|(path, _)| match stacker.push_fits(path).unwrap() {
            FrameDisposition::Accepted(diagnostics) => diagnostics,
            FrameDisposition::Rejected(reason) => panic!("{}: {reason}", path.display()),
        })
        .collect();
    (stacker, diagnostics)
}

fn binned(bin_factor: u32) -> StackOptions {
    StackOptions {
        bin_factor,
        ..StackOptions::default()
    }
}

/// The largest distance between where `found` and `truth` carry the corners
/// of a frame of the sensor.
fn corner_error(found: SimilarityTransform, truth: SimilarityTransform) -> f64 {
    let (right, bottom) = ((WIDTH - 1) as f64, (HEIGHT - 1) as f64);
    [(0.0, 0.0), (right, 0.0), (0.0, bottom), (right, bottom)]
        .into_iter()
        .map(|(x, y)| {
            let (a, b) = found.apply(x, y);
            let (c, d) = truth.apply(x, y);
            (a - c).hypot(b - d)
        })
        .fold(0.0, f64::max)
}

/// The source-to-reference transform the truth gives frame `pose`.
fn truth(reference: SimilarityTransform, pose: SimilarityTransform) -> SimilarityTransform {
    let inverse = |t: SimilarityTransform| {
        let (x, y) = t.inverse_apply(0.0, 0.0);
        SimilarityTransform {
            scale: 1.0 / t.scale,
            rotation_radians: -t.rotation_radians,
            translation_x: x,
            translation_y: y,
        }
    };
    inverse(pose).then(reference)
}

/// Every transform registration finds on binned frames carries exactly to
/// the sensor's pixels, the frame after a meridian flip included, where
/// scaling the translation alone misses by the bin factor less one.
#[test]
fn binned_transforms_carry_to_the_sensor_through_a_half_turn() {
    let directory = tempfile::tempdir().unwrap();
    let frames = field(directory.path(), 5, Some(3));
    for bin_factor in [2, 4] {
        let (stacker, diagnostics) = stack(&frames, binned(bin_factor));
        assert_eq!(stacker.bin_factor(), bin_factor);
        let factor = bin_factor as usize;
        assert_eq!(
            (stacker.view().width, stacker.view().height),
            (WIDTH / factor, HEIGHT / factor)
        );
        for (frame, diagnostics) in frames[1..].iter().zip(&diagnostics) {
            assert_eq!(diagnostics.mapping.bin_factor(), bin_factor);
            let truth = truth(frames[0].1, frame.1);
            let native = diagnostics.mapping.rebinned(1).unwrap();
            assert_eq!(native.bin_factor(), 1);
            assert_eq!(
                (native.reference_width(), native.reference_height()),
                (WIDTH, HEIGHT)
            );
            let error = corner_error(native.transform(), truth);
            assert!(
                error < 0.15 * f64::from(bin_factor),
                "bin {bin_factor}, {}: {error:.3}px",
                frame.0.display()
            );
            // Registration RMS is reported in the sensor's pixels: the
            // native stack's frames fit at 0.2 to 0.3.
            assert!(diagnostics.registration_rms_pixels < 0.3 * f64::from(bin_factor));
            let flipped = diagnostics.transform.rotation_radians.cos() < 0.0;
            assert_eq!(flipped, frame.0.ends_with("light-3.fits"));
            if flipped {
                let mut naive = diagnostics.transform;
                naive.translation_x *= f64::from(bin_factor);
                naive.translation_y *= f64::from(bin_factor);
                assert!(
                    corner_error(naive, truth) > 0.6 * f64::from(bin_factor - 1),
                    "scaling the translation alone lands within {:.3}px",
                    corner_error(naive, truth)
                );
            }
        }
        // The drift bound is reported in the sensor's pixels.
        let native = LiveStacker::open_fits(&frames[0].0, None, None, None, None, binned(1))
            .unwrap()
            .effective_maximum_drift_pixels();
        assert!((stacker.effective_maximum_drift_pixels() - native).abs() <= f64::from(bin_factor));
    }
}

/// A binned stack reintegrated at full resolution matches the stack of the
/// same frames taken at full resolution throughout, refined or not, with
/// gains from stars or none.
///
/// [`NormalizationMode::Global`] is left out: its gain is the ratio of the
/// reference's dispersion to a resampled frame's, and on a sky this sparse
/// that is the noise's, which bilinear resampling smooths by an amount the
/// sub-pixel shift sets. The same frames took gains of 1.47 at full
/// resolution and 1.36 binned by 2, and the full-resolution pass keeps the
/// binned pass's normalization.
#[test]
fn a_binned_stack_reintegrates_at_full_resolution_as_a_native_stack() {
    let directory = tempfile::tempdir().unwrap();
    let frames = field(directory.path(), 6, Some(4));
    let options = BatchStackOptions::default();
    for normalization in [
        NormalizationMode::None,
        NormalizationMode::LocalBackground { tile_size: 64 },
    ] {
        let with = |bin_factor| StackOptions {
            normalization,
            ..binned(bin_factor)
        };
        let (native, _) = stack(&frames, with(1));
        let native = native.reintegrate(&options, |_, _, _| {}).unwrap().snapshot;
        for bin_factor in [2, 4] {
            let (stacker, _) = stack(&frames, with(bin_factor));
            for refine in [true, false] {
                let mut heard = Vec::new();
                let full = stacker
                    .reintegrate_full_resolution(&options, refine, |pass, index, count| {
                        heard.push((pass, index, count));
                    })
                    .unwrap();
                let what = format!("{normalization:?}, bin {bin_factor}, refine {refine}");
                let image = &full.result.snapshot.image;
                assert_eq!((image.width, image.height), (WIDTH, HEIGHT), "{what}");
                assert_eq!(full.result.snapshot.accepted_frames, 6, "{what}");
                assert_eq!(full.refined_frames, if refine { 5 } else { 0 }, "{what}");
                let rms = full.registration_rms_pixels.unwrap();
                assert!(rms < 0.3 * f64::from(bin_factor), "{what}: {rms}");
                assert!(
                    heard.contains(&(crate::BatchStackPass::Integrate, 5, 6)),
                    "{what}"
                );
                // Where every frame reaches, the two stacks agree to within a
                // fraction of the sky noise of one frame, and of the stars.
                let mut differences = image
                    .data
                    .iter()
                    .zip(&native.image.data)
                    .zip(full.result.snapshot.coverage.iter().zip(&native.coverage))
                    .filter(|(_, (a, b))| **a == 6 && **b == 6)
                    .map(|((a, b), _)| (a - b).abs() / b.abs().max(500.0) * 500.0)
                    .collect::<Vec<_>>();
                assert!(differences.len() > WIDTH * HEIGHT / 2, "{what}");
                differences.sort_by(f32::total_cmp);
                let median = differences[differences.len() / 2];
                let high = differences[differences.len() * 99 / 100];
                // One frame's sky noise is about 6. Refined, the frames land
                // where full-resolution registration puts them, and without
                // normalization the stacks are the same; a background fitted
                // on binned frames moves the stack by a fraction of the noise.
                // Carried unrefined, a star's flanks move by up to a few
                // percent at 4.
                let (median_limit, high_limit) = match (normalization, refine) {
                    (NormalizationMode::None, true) => (0.0, 0.0),
                    (_, true) => (1.5, 3.0),
                    (_, false) => (1.5, 15.0),
                };
                assert!(
                    median <= median_limit && high <= high_limit,
                    "{what}: median {median}, 99th percentile {high}"
                );
            }
        }
    }
}

/// A binned stack's options, mappings and full-resolution reintegration
/// survive a saved context, and default options still write no factor.
#[test]
fn a_binned_context_keeps_its_bin_factor() {
    let directory = tempfile::tempdir().unwrap();
    let frames = field(directory.path(), 4, None);
    let (stacker, _) = stack(&frames[..3], binned(2));
    let path = directory.path().join("binned.seiza-stack");
    stacker.save_context(&path).unwrap();
    let mut restored = LiveStacker::open_context(&path).unwrap();
    assert_eq!(restored.bin_factor(), 2);
    // A depth reading says what the stack was binned by.
    assert_eq!(restored.measure_depth().unwrap().bin_factor, 2);
    assert_eq!(crate::measure_depth(restored.view()).unwrap().bin_factor, 1);
    assert_eq!(
        restored.configuration_fingerprint(),
        stacker.configuration_fingerprint()
    );
    assert_eq!(
        (restored.view().width, restored.view().height),
        (WIDTH / 2, HEIGHT / 2)
    );
    let FrameDisposition::Accepted(diagnostics) = restored.push_fits(&frames[3].0).unwrap() else {
        panic!("the last frame joins the restored stack");
    };
    assert_eq!(diagnostics.mapping.bin_factor(), 2);
    let full = restored
        .reintegrate_full_resolution(&BatchStackOptions::default(), true, |_, _, _| {})
        .unwrap();
    assert_eq!(full.refined_frames, 3);
    assert_eq!(
        (
            full.result.snapshot.image.width,
            full.result.snapshot.image.height
        ),
        (WIDTH, HEIGHT)
    );

    let json = serde_json::to_string(&binned(2)).unwrap();
    assert!(json.contains("\"bin_factor\":2"), "{json}");
    let read: StackOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(read.bin_factor, 2);
    let default = serde_json::to_string(&StackOptions::default()).unwrap();
    assert!(!default.contains("bin_factor"), "{default}");
    let read: StackOptions = serde_json::from_str("{}").unwrap();
    assert_eq!(read.bin_factor, 1);
}

/// The reference's WCS and binning cards follow it onto the binned grid, and
/// back again for a full-resolution reintegration.
#[test]
fn reference_headers_follow_the_binned_grid_and_back() {
    let directory = tempfile::tempdir().unwrap();
    let mut frames = field(directory.path(), 3, None);
    let headers = [
        ("CTYPE1", HeaderValue::String("RA---TAN".into())),
        ("CTYPE2", HeaderValue::String("DEC--TAN".into())),
        ("CRVAL1", HeaderValue::Float(83.8)),
        ("CRVAL2", HeaderValue::Float(-5.4)),
        ("CRPIX1", HeaderValue::Float(192.5)),
        ("CRPIX2", HeaderValue::Float(144.5)),
        ("CD1_1", HeaderValue::Float(-3.0e-4)),
        ("CD1_2", HeaderValue::Float(1.0e-6)),
        ("CD2_1", HeaderValue::Float(1.0e-6)),
        ("CD2_2", HeaderValue::Float(3.0e-4)),
        ("XBINNING", HeaderValue::Integer(1)),
        ("YBINNING", HeaderValue::Integer(1)),
        ("XPIXSZ", HeaderValue::Float(3.76)),
        ("YPIXSZ", HeaderValue::Float(3.76)),
    ];
    let mut reference = FitsFrame::open(&frames[0].0).unwrap();
    reference.headers.extend(
        headers
            .iter()
            .map(|(key, value)| (key.to_string(), value.clone())),
    );
    let cards = reference
        .headers
        .iter()
        .filter(|(key, _)| headers.iter().any(|(wanted, _)| key == wanted))
        .map(|(key, value)| seiza_fits::WriteHeaderCard::new(key, value.clone()))
        .collect::<Vec<_>>();
    let path = directory.path().join("solved.fits");
    write_f32_image(
        &path,
        WIDTH,
        HEIGHT,
        F32ImageData::Mono(&reference.image.data),
        &cards,
    )
    .unwrap();
    frames[0].0 = path;
    let (stacker, _) = stack(&frames, binned(2));
    let value = |headers: &[(String, HeaderValue)], key: &str| {
        headers
            .iter()
            .find(|(name, _)| name == key)
            .and_then(|(_, value)| value.as_f64())
    };
    let binned = stacker.reference_headers();
    assert_eq!(value(binned, "CRPIX1"), Some(96.5));
    assert_eq!(value(binned, "CRPIX2"), Some(72.5));
    assert_eq!(value(binned, "CD1_1"), Some(-6.0e-4));
    assert_eq!(value(binned, "XBINNING"), Some(2.0));
    assert_eq!(value(binned, "XPIXSZ"), Some(7.52));
    assert_eq!(value(binned, "SEIZABIN"), Some(2.0));
    let full = stacker
        .reintegrate_full_resolution(&BatchStackOptions::default(), false, |_, _, _| {})
        .unwrap();
    for (key, expected) in &headers {
        let found = value(&full.reference_headers, key);
        match expected.as_f64() {
            Some(expected) => {
                let found = found.unwrap();
                assert!(
                    (found - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                    "{key}: {found} against {expected}"
                );
            }
            None => assert!(full.reference_headers.iter().any(|(name, _)| name == key)),
        }
    }
    assert_eq!(value(&full.reference_headers, "SEIZABIN"), None);
}

#[test]
fn bin_factors_outside_one_to_four_and_binned_bayer_drizzle_are_refused() {
    for bin_factor in [0, 5] {
        assert!(binned(bin_factor).validate().is_err(), "{bin_factor}");
    }
    for bin_factor in 1..=4 {
        binned(bin_factor).validate().unwrap();
    }
    let drizzle = StackOptions {
        cfa_integration: CfaIntegration::BayerDrizzle,
        ..binned(2)
    };
    assert!(drizzle.validate().is_err());
    StackOptions {
        bin_factor: 1,
        ..drizzle
    }
    .validate()
    .unwrap();
}

/// The stars [`field`] draws, as the sky holds them.
fn sky_stars() -> Vec<(f64, f64, f64)> {
    let mut seed = 0x0b1e_u64;
    let mut random = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f64 / (1u64 << 31) as f64
    };
    let margin = 60.0;
    (0..WIDTH * HEIGHT / 220)
        .map(|_| {
            (
                -margin + random() * (WIDTH as f64 + 2.0 * margin),
                -margin + random() * (HEIGHT as f64 + 2.0 * margin),
                150.0 + 9000.0 * random().powi(3),
            )
        })
        .collect()
}

/// The mean distance, over the brightest stars well apart from the rest,
/// between where `image`, a mono stack on frame `reference`'s full-resolution
/// grid, puts them and where the sky does.
fn star_misplacement(image: &LinearImage, reference: SimilarityTransform) -> f64 {
    let mut sky = image
        .data
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    sky.sort_by(f32::total_cmp);
    let sky = sky[sky.len() / 2];
    let placed = sky_stars()
        .into_iter()
        .map(|(x, y, peak)| {
            let (x, y) = reference.apply(x, y);
            (x, y, peak)
        })
        .collect::<Vec<_>>();
    let (mut total, mut count) = (0.0, 0);
    for &(x, y, peak) in &placed {
        let isolated = placed
            .iter()
            .filter(|&&(u, v, _)| (u - x).hypot(v - y) < 12.0)
            .count()
            == 1;
        let inside =
            x > 10.0 && y > 10.0 && x < image.width as f64 - 11.0 && y < image.height as f64 - 11.0;
        if peak < 2000.0 || !isolated || !inside {
            continue;
        }
        let (mut sum, mut sx, mut sy) = (0.0, 0.0, 0.0);
        for row in (y.round() as usize - 5)..=(y.round() as usize + 5) {
            for column in (x.round() as usize - 5)..=(x.round() as usize + 5) {
                let weight = f64::from(image.data[row * image.width + column] - sky).max(0.0);
                sum += weight;
                sx += weight * column as f64;
                sy += weight * row as f64;
            }
        }
        total += (sx / sum - x).hypot(sy / sum - y);
        count += 1;
    }
    assert!(count >= 5, "{count} stars measured");
    total / count as f64
}

/// A binned stack drizzles each frame's sensor pixels, a binned pixel's
/// part across, so a drizzle by the bin factor lands the stars where the
/// sensor's full resolution puts them.
#[test]
fn a_binned_stack_drizzles_to_the_sensor_s_resolution() {
    let directory = tempfile::tempdir().unwrap();
    let frames = field(directory.path(), 5, Some(3));
    let options = BatchStackOptions::default();
    let (native, _) = stack(&frames, binned(1));
    let native = native.reintegrate(&options, |_, _, _| {}).unwrap().snapshot;
    let native_error = star_misplacement(&native.image, frames[0].1);
    for bin_factor in [2, 4] {
        let (stacker, _) = stack(&frames, binned(bin_factor));
        let (_, drizzled) = stacker
            .reintegrate_drizzled(
                &options,
                &crate::DrizzleOptions {
                    scale: bin_factor,
                    drop_shrink: None,
                },
                |_, _, _| {},
            )
            .unwrap();
        assert_eq!(
            (drizzled.image.width, drizzled.image.height),
            (WIDTH, HEIGHT)
        );
        let error = star_misplacement(&drizzled.image, frames[0].1);
        assert!(
            error < native_error + 0.05 * f64::from(bin_factor),
            "bin {bin_factor}: stars {error:.3}px off, against {native_error:.3}px at full resolution"
        );
    }
}

/// Prepared pixels are binned too, so a host pushes what its sensor took.
#[test]
fn prepared_pixels_are_binned_as_files_are() {
    let directory = tempfile::tempdir().unwrap();
    let frames = field(directory.path(), 3, None);
    let images = frames
        .iter()
        .map(|(path, _)| FitsFrame::open(path).unwrap().image)
        .collect::<Vec<_>>();
    let mut from_pixels = LiveStacker::from_linear(images[0].clone(), binned(2)).unwrap();
    let (from_files, file_diagnostics) = stack(&frames, binned(2));
    for (image, from_file) in images[1..].iter().zip(&file_diagnostics) {
        let FrameDisposition::Accepted(diagnostics) =
            from_pixels.push_linear(image.clone()).unwrap()
        else {
            panic!("prepared pixels are admitted");
        };
        assert_eq!(diagnostics.transform, from_file.transform);
    }
    let bits = |stacker: &LiveStacker| {
        stacker
            .snapshot()
            .unwrap()
            .image
            .data
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    assert!(bits(&from_pixels) == bits(&from_files));
}

/// Raw RGGB lights of [`field`]'s stars, coloured, on a coloured sky.
fn bayer_field(
    directory: &Path,
    count: usize,
    flipped: Option<usize>,
) -> Vec<(PathBuf, SimilarityTransform)> {
    const SKY: [f32; 3] = [400.0, 600.0, 300.0];
    const COLOUR: [f32; 3] = [1.0, 0.8, 0.6];
    field(directory, count, flipped)
        .into_iter()
        .map(|(path, pose)| {
            let frame = FitsFrame::open(&path).unwrap();
            let data = frame
                .image
                .data
                .iter()
                .enumerate()
                .map(|(index, &value)| {
                    let (x, y) = (index % WIDTH, index / WIDTH);
                    let channel = [[0, 1], [1, 2]][y & 1][x & 1];
                    // `field`'s sky is 500; its stars and noise ride on it.
                    SKY[channel] + COLOUR[channel] * (value - 500.0)
                })
                .collect::<Vec<_>>();
            let cards = [seiza_fits::WriteHeaderCard::new(
                "BAYERPAT",
                HeaderValue::String("RGGB".into()),
            )];
            write_f32_image(&path, WIDTH, HEIGHT, F32ImageData::Mono(&data), &cards).unwrap();
            (path, pose)
        })
        .collect()
}

/// The mean offset, over the brightest isolated stars, of the red
/// centroid from the blue in `image`, a stack on frame `reference`'s grid
/// binned by `bin_factor`.
fn red_from_blue(
    image: &LinearImage,
    reference: SimilarityTransform,
    bin_factor: u32,
) -> (f64, f64) {
    let factor = f64::from(bin_factor);
    let offset = (factor - 1.0) / 2.0;
    let median = |channel: usize| {
        let mut values = image
            .data
            .iter()
            .skip(channel)
            .step_by(3)
            .copied()
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        values.sort_by(f32::total_cmp);
        values[values.len() / 2]
    };
    let sky = [median(0), median(1), median(2)];
    let centroid = |x: f64, y: f64, channel: usize| {
        let (mut sum, mut sx, mut sy) = (0.0, 0.0, 0.0);
        for row in (y.round() as isize - 3)..=(y.round() as isize + 3) {
            for column in (x.round() as isize - 3)..=(x.round() as isize + 3) {
                let value =
                    image.data[(row as usize * image.width + column as usize) * 3 + channel];
                let weight = f64::from(value - sky[channel]).max(0.0);
                sum += weight;
                sx += weight * column as f64;
                sy += weight * row as f64;
            }
        }
        (sx / sum, sy / sum)
    };
    // The same stars `field` drew, where the reference frame put them.
    let mut seed = 0x0b1e_u64;
    let mut random = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f64 / (1u64 << 31) as f64
    };
    let margin = 60.0;
    let stars = (0..WIDTH * HEIGHT / 220)
        .map(|_| {
            (
                -margin + random() * (WIDTH as f64 + 2.0 * margin),
                -margin + random() * (HEIGHT as f64 + 2.0 * margin),
                150.0 + 9000.0 * random().powi(3),
            )
        })
        .collect::<Vec<_>>();
    let placed = stars
        .iter()
        .map(|&(x, y, peak)| {
            let (x, y) = reference.apply(x, y);
            ((x - offset) / factor, (y - offset) / factor, peak)
        })
        .collect::<Vec<_>>();
    let (mut total_x, mut total_y, mut count) = (0.0, 0.0, 0);
    for &(x, y, peak) in &placed {
        let isolated = placed
            .iter()
            .filter(|&&(u, v, _)| (u - x).hypot(v - y) < 3.0 + 4.0 / factor)
            .count()
            == 1;
        let inside =
            x > 8.0 && y > 8.0 && x < image.width as f64 - 9.0 && y < image.height as f64 - 9.0;
        if peak > 2000.0 && isolated && inside {
            let (red, blue) = (centroid(x, y, 0), centroid(x, y, 2));
            total_x += red.0 - blue.0;
            total_y += red.1 - blue.1;
            count += 1;
        }
    }
    assert!(count >= 5, "{count} stars measured");
    (total_x / count as f64, total_y / count as f64)
}

/// A superpixel stack bins by 2 as it debayers, and resamples each colour
/// from where its photosite lies, so its red and blue line up where the
/// cells hold them half a sensor pixel apart, a meridian flip included; it
/// reintegrates at full resolution through VNG.
#[test]
fn a_superpixel_stack_lines_its_colours_up() {
    let directory = tempfile::tempdir().unwrap();
    let frames = bayer_field(directory.path(), 5, Some(3));
    for bin_factor in [2, 4] {
        let options = StackOptions {
            demosaic: crate::Demosaic::Superpixel,
            ..binned(bin_factor)
        };
        let (stacker, _) = stack(&frames, options);
        let factor = bin_factor as usize;
        let snapshot = stacker.snapshot().unwrap();
        assert_eq!(
            (
                snapshot.image.width,
                snapshot.image.height,
                snapshot.image.channels
            ),
            (WIDTH / factor, HEIGHT / factor, 3)
        );
        let bin = stacker
            .reference_headers()
            .iter()
            .find(|(key, _)| key == "SEIZABIN")
            .and_then(|(_, value)| value.as_f64());
        assert_eq!(bin, Some(f64::from(bin_factor)));
        let (dx, dy) = red_from_blue(&snapshot.image, frames[0].1, bin_factor);
        // Read from the pixel centres, red sat 0.29 binned pixels right of
        // and below blue at 2: a sensor pixel, less the flipped frame's.
        let limit = 0.15 / f64::from(bin_factor);
        assert!(
            dx.abs() < limit && dy.abs() < limit,
            "bin {bin_factor}: red is ({dx:.3}, {dy:.3}) from blue"
        );
        let full = stacker
            .reintegrate_full_resolution(&BatchStackOptions::default(), true, |_, _, _| {})
            .unwrap();
        assert_eq!(full.refined_frames, 4);
        assert_eq!(
            (
                full.result.snapshot.image.width,
                full.result.snapshot.image.height
            ),
            (WIDTH, HEIGHT)
        );
    }
    for bin_factor in [1, 3] {
        let options = StackOptions {
            demosaic: crate::Demosaic::Superpixel,
            ..binned(bin_factor)
        };
        assert!(options.validate().is_err(), "{bin_factor}");
    }
}

#[test]
fn superpixel_mapping_crops_match_live_reference_after_json_and_context() {
    let directory = tempfile::tempdir().unwrap();
    let frames = bayer_field(directory.path(), 2, None);
    for bin_factor in [2, 4] {
        let options = StackOptions {
            demosaic: crate::Demosaic::Superpixel,
            normalization: NormalizationMode::None,
            ..binned(bin_factor)
        };
        let mut stacker =
            LiveStacker::open_fits(&frames[0].0, None, None, None, None, options.clone()).unwrap();
        let (prepared, _) =
            develop(FitsFrame::open(&frames[0].0).unwrap(), &options, bin_factor).unwrap();
        let region = ReferenceRegion {
            x: 8,
            y: 8,
            width: 32,
            height: 24,
        };
        let expected = stacker.snapshot().unwrap().image.crop(region).unwrap();
        let context = directory.path().join(format!("mapping-{bin_factor}.seiza"));
        let FrameDisposition::Accepted(accepted) = stacker.push_fits(&frames[1].0).unwrap() else {
            panic!("second frame rejected");
        };
        assert_eq!(
            accepted.mapping.channel_sites(),
            stacker.reference_mapping().channel_sites()
        );
        stacker.save_context(&context).unwrap();
        let reopened = LiveStacker::open_context(&context).unwrap();
        assert_eq!(
            reopened.ledger.mapping(1).channel_sites(),
            accepted.mapping.channel_sites()
        );
        for mapping in [stacker.reference_mapping(), reopened.reference_mapping()] {
            let json = serde_json::to_string(&mapping).unwrap();
            let mapping: RegisteredFrameMapping = serde_json::from_str(&json).unwrap();
            let actual = mapping
                .extract_region_with(&prepared.image, region, options.interpolation)
                .unwrap();
            assert_eq!(
                actual.data, expected.data,
                "bin {bin_factor}: saved mapping crop"
            );
        }
    }
}

/// A binned stack's full-resolution reintegration refits each frame's
/// background on the sensor's grid, and leaves out its samples about the
/// reference's stars where they lie there: at twice their binned places,
/// and with discs twice as wide, not at the binned places on a grid twice
/// as large, with discs as wide as a reference with no stars to measure
/// gives.
#[test]
fn a_full_resolution_reintegration_masks_the_stars_where_they_lie() {
    let directory = tempfile::tempdir().unwrap();
    let frames = field(directory.path(), 3, None);
    let options = StackOptions {
        normalization: NormalizationMode::LocalBackground { tile_size: 64 },
        ..binned(2)
    };
    let (stacker, _) = stack(&frames, options);
    let binned = stacker.background_mask().expect("a binned mask");
    let binned_covered = (0..stacker.reference.width * stacker.reference.height)
        .filter(|&pixel| binned.covers(pixel))
        .count();
    for refine in [false, true] {
        let (full, _) = stacker
            .full_resolution(&BatchStackOptions::default(), refine, &mut |_, _, _| {})
            .unwrap();
        let mask = full.background_mask().expect("a full-resolution mask");
        let width = full.reference.width;
        for &(x, y) in &stacker.registrar.reference_star_positions() {
            let (column, row) = (
                (2.0 * x + 0.5).round() as usize,
                (2.0 * y + 0.5).round() as usize,
            );
            assert!(
                mask.covers(row * width + column),
                "refine {refine}: ({x}, {y})"
            );
        }
        // Four pixels for each binned one, give or take the discs' edges.
        let covered = (0..width * full.reference.height)
            .filter(|&pixel| mask.covers(pixel))
            .count();
        let ratio = covered as f64 / (4 * binned_covered) as f64;
        assert!((0.9..1.1).contains(&ratio), "refine {refine}: {ratio}");
    }
}
