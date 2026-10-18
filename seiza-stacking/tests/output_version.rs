//! Holds [`seiza_stacking::OUTPUT_VERSION`] to the output it promises.
//!
//! Each case stacks or combines fixed synthetic frames and compares a summary
//! of the result with the one recorded here. A change that moves a summary
//! changes what callers' cached stacks and masters would hold: set
//! `OUTPUT_VERSION` to the release that ships the change, and record the new
//! summary, which the failure prints.

use seiza_fits::{HeaderValue, WriteHeaderCard};
use seiza_stacking::{
    FrameDisposition, FrameWeighting, Interpolation, LinearImage, LiveStacker, MasterBuildOptions,
    MasterFrameKind, NormalizationMode, RegistrationModel, StackOptions, build_master_from_fits,
    write_processed_image_fits_f32,
};
use std::path::PathBuf;

const WIDTH: usize = 192;
const HEIGHT: usize = 160;

/// Relative to the larger of the recorded value and 1. Platforms differ in
/// the last bits of `exp` and friends; an algorithm change moves far more.
const TOLERANCE: f64 = 1e-5;

#[rustfmt::skip]
const MONO_DEFAULT: &[f64] = &[
    1.4835510e2, 2.8592231e2, 3.9204758e3, 9.7156273e1, 9.9581932e1, 1.0100323e2,
    1.1871532e2, 1.0544968e2, 9.7902931e1, 1.0035307e2, 1.0213988e2, 1.0341476e2,
    1.0482466e2, 9.8180191e1, 1.0143280e2, 3.7782477e2, 1.0442571e2, 1.0646262e2,
    1.0008590e2, 1.0090702e2, 1.0369869e2, 1.0641956e2, 1.0768224e2, 1.0096002e2,
    1.7523038e2, 1.0491113e2, 1.0702999e2, 1.1143148e2,
];
#[rustfmt::skip]
const MONO_CONFIGURED: &[f64] = &[
    1.4699745e2, 2.8121569e2, 3.9688047e3, 9.7713203e1, 1.0011301e2, 1.0122166e2,
    1.1481973e2, 1.0544106e2, 9.8354599e1, 1.0061407e2, 1.0243215e2, 1.0345580e2,
    1.0467356e2, 9.8428558e1, 1.0228342e2, 4.2148636e2, 1.0456822e2, 1.0636676e2,
    1.0037716e2, 1.0089593e2, 1.0360520e2, 1.0664339e2, 1.0773047e2, 1.0119784e2,
    1.7354935e2, 1.0513698e2, 1.0699198e2, 1.1135787e2,
];
#[rustfmt::skip]
const BAYER_DEFAULT: &[f64] = &[
    4.5997909e2, 5.6700967e2, 5.5486519e3, 3.4617752e2, 4.2780089e2, 4.1672310e3,
    2.3022666e2, 2.8343051e2, 2.7395906e3, 4.0003683e2, 3.0023581e2, 2.0065260e2,
    4.0085056e2, 3.0008279e2, 2.0111510e2, 4.0071625e2, 3.0122803e2, 1.9936935e2,
    4.2822186e2, 3.2929889e2, 2.1589828e2, 4.0070880e2, 3.0015631e2, 2.0029289e2,
    3.9967078e2, 2.9972723e2, 2.0029242e2, 4.0132016e2, 3.0027393e2, 1.9899487e2,
    4.0048090e2, 3.0053088e2, 2.0024478e2, 3.9874377e2, 2.9950262e2, 2.0064078e2,
    3.9835126e2, 2.9881433e2, 1.9862039e2, 3.9781042e2, 2.9859549e2, 2.0001193e2,
    4.0229999e2, 3.0093875e2, 2.0069858e2, 4.0114334e2, 2.9982175e2, 2.0000229e2,
    3.9900464e2, 2.9965771e2, 1.9908920e2, 3.9859332e2, 2.9932312e2, 1.9946545e2,
    4.0095612e2, 3.0028061e2, 2.0119962e2, 3.9827237e2, 2.9866693e2, 1.9951340e2,
    3.9935376e2, 3.0045358e2, 1.9897159e2, 4.0024432e2, 3.0033688e2, 2.0081078e2,
    4.0007721e2, 2.9931595e2, 1.9919868e2, 3.9992365e2, 3.0025327e2, 1.9911864e2,
    5.0519611e2, 3.8917441e2, 2.5463911e2, 3.9861279e2, 3.0019730e2, 1.9942378e2,
    4.0142902e2, 3.0100272e2, 1.9984351e2, 4.0275519e2, 3.0402594e2, 2.0407870e2,
];
#[rustfmt::skip]
const DARK_MASTER: &[f64] = &[
    5.0018033e2, 5.0018273e2, 5.0560712e2, 4.9912210e2, 5.0097733e2, 5.0010815e2,
    5.0059848e2, 5.0013586e2, 5.0037161e2, 5.0233649e2, 5.0203445e2, 4.9982404e2,
    4.9820868e2, 5.0218640e2, 4.9793271e2, 5.0225122e2, 4.9774274e2, 5.0003473e2,
    5.0157529e2, 5.0029086e2, 5.0083975e2, 5.0305823e2, 4.9923734e2, 5.0075592e2,
    4.9867371e2, 4.9978439e2, 5.0103198e2, 4.9647897e2,
];
#[rustfmt::skip]
const FLAT_MASTER: &[f64] = &[
    9.9502314e-1, 9.9747827e-1, 1.1070186e0, 8.8392133e-1, 9.7098047e-1, 9.9945378e-1,
    9.7076976e-1, 8.8421541e-1, 9.6401429e-1, 1.0505759e0, 1.0799839e0, 1.0501052e0,
    9.6203160e-1, 9.9122244e-1, 1.0760109e0, 1.1063745e0, 1.0758289e0, 9.9034131e-1,
    9.6445268e-1, 1.0503318e0, 1.0797215e0, 1.0510001e0, 9.6345681e-1, 8.8453329e-1,
    9.7012061e-1, 9.9929720e-1, 9.7092038e-1, 8.8284981e-1,
];

#[test]
fn output_version_is_a_released_version_no_newer_than_this_one() {
    let parse = |version: &str| {
        version
            .split('.')
            .map(|part| part.parse::<u64>().unwrap())
            .collect::<Vec<_>>()
    };
    let output = parse(seiza_stacking::OUTPUT_VERSION);
    assert_eq!(output.len(), 3, "{}", seiza_stacking::OUTPUT_VERSION);
    assert!(output <= parse(seiza_stacking::VERSION));
}

#[test]
fn a_mono_stack_with_default_options_keeps_its_output() {
    check(
        "MONO_DEFAULT",
        &mono_stack(StackOptions::default()),
        MONO_DEFAULT,
    );
}

#[test]
fn a_mono_stack_with_every_stage_configured_keeps_its_output() {
    let mut options = StackOptions {
        normalization: NormalizationMode::LocalBackground { tile_size: 32 },
        weighting: FrameWeighting::InverseNoiseVariance {
            minimum_weight: 0.05,
            maximum_weight: 20.0,
        },
        interpolation: Interpolation::Lanczos3,
        ..StackOptions::default()
    };
    options.registration.model = RegistrationModel::Quadratic;
    check("MONO_CONFIGURED", &mono_stack(options), MONO_CONFIGURED);
}

#[test]
fn a_debayered_stack_keeps_its_output() {
    let directory = tempfile::tempdir().unwrap();
    let bayer = [("BAYERPAT".to_string(), HeaderValue::String("RGGB".into()))];
    let paths = SHIFTS[..6]
        .iter()
        .enumerate()
        .map(|(frame, &(dx, dy))| {
            let image = bayer_field(dx, dy, frame);
            let path = directory.path().join(format!("light-{frame}.fits"));
            write_processed_image_fits_f32(&path, &image, &bayer, &[]).unwrap();
            path
        })
        .collect::<Vec<_>>();
    let mut stacker =
        LiveStacker::open_fits(&paths[0], None, None, None, None, StackOptions::default()).unwrap();
    for path in &paths[1..] {
        admit(stacker.push_fits(path).unwrap());
    }
    let snapshot = stacker.snapshot().unwrap();
    assert_eq!(snapshot.accepted_frames, 6);
    check("BAYER_DEFAULT", &summary(&snapshot.image), BAYER_DEFAULT);
}

#[test]
fn a_dark_master_keeps_its_output() {
    let directory = tempfile::tempdir().unwrap();
    let paths = (0..7)
        .map(|frame| {
            let mut data = (0..WIDTH * HEIGHT)
                .map(|index| 500.0 + 0.002 * (index % WIDTH) as f32 + noise(index, frame, 6.0))
                .collect::<Vec<_>>();
            // A cosmic-ray hit in one frame, which rejection removes.
            if frame == 3 {
                data[70 * WIDTH + 90] += 4_000.0;
            }
            write_raw(&directory, "dark", frame, data, "DARK")
        })
        .collect::<Vec<_>>();
    let master = build_master_from_fits(
        &paths,
        MasterFrameKind::Dark,
        &MasterBuildOptions::default(),
    )
    .unwrap();
    check("DARK_MASTER", &summary(&master.image), DARK_MASTER);
}

#[test]
fn a_flat_master_keeps_its_output() {
    let directory = tempfile::tempdir().unwrap();
    let paths = (0..7)
        .map(|frame| {
            let level = 20_000.0 * (0.9 + 0.03 * frame as f32);
            let data = (0..WIDTH * HEIGHT)
                .map(|index| level * vignetting(index) + noise(index, frame, 40.0))
                .collect::<Vec<_>>();
            write_raw(&directory, "flat", frame, data, "FLAT")
        })
        .collect::<Vec<_>>();
    let master = build_master_from_fits(
        &paths,
        MasterFrameKind::Flat,
        &MasterBuildOptions::default(),
    )
    .unwrap();
    check("FLAT_MASTER", &summary(&master.image), FLAT_MASTER);
}

/// Sub-pixel dithers, including one large enough to uncover an edge.
const SHIFTS: [(f32, f32); 8] = [
    (0.0, 0.0),
    (2.3, -1.6),
    (-3.1, 2.7),
    (4.6, 1.2),
    (-1.8, -3.9),
    (0.7, 4.4),
    (-4.2, -0.5),
    (3.4, 3.1),
];

fn mono_stack(options: StackOptions) -> Vec<f64> {
    let gain = |frame: usize| 0.96 + 0.01 * frame as f32;
    let mut stacker = LiveStacker::from_linear(mono_field(0.0, 0.0, gain(0), 0), options).unwrap();
    for (frame, &(dx, dy)) in SHIFTS.iter().enumerate().skip(1) {
        let mut image = mono_field(dx, dy, gain(frame), frame);
        // A satellite trail in one frame, which rejection removes.
        if frame == 4 {
            for x in 0..WIDTH {
                let y = 20 + x * 3 / 5;
                image.data[y * WIDTH + x] += 3_000.0;
            }
        }
        admit(stacker.push_linear(image).unwrap());
    }
    let snapshot = stacker.snapshot().unwrap();
    assert_eq!(snapshot.accepted_frames, SHIFTS.len() as u32);
    summary(&snapshot.image)
}

fn admit(disposition: FrameDisposition) {
    assert!(
        matches!(disposition, FrameDisposition::Accepted(_)),
        "{disposition:?}"
    );
}

/// Forty stars at fixed places, drawn from a small integer generator so the
/// field is the same on every platform.
fn stars() -> Vec<(f32, f32, f32)> {
    let mut state = 0x2545_f491_u32;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1 << 24) as f32
    };
    (0..40)
        .map(|_| {
            let x = 8.0 + next() * (WIDTH as f32 - 16.0);
            let y = 8.0 + next() * (HEIGHT as f32 - 16.0);
            (x, y, 400.0 + next() * 4_000.0)
        })
        .collect()
}

/// Uniform noise in `[-amplitude, amplitude)` from a hash of the sample and
/// the frame.
fn noise(index: usize, frame: usize, amplitude: f32) -> f32 {
    let mut hash = (index as u64)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add((frame as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9));
    hash ^= hash >> 31;
    hash = hash.wrapping_mul(0x94d0_49bb_1331_11eb);
    hash ^= hash >> 29;
    ((hash >> 40) as f32 / (1 << 24) as f32 * 2.0 - 1.0) * amplitude
}

fn star_light(stars: &[(f32, f32, f32)], x: f32, y: f32) -> f32 {
    stars
        .iter()
        .map(|&(star_x, star_y, flux)| {
            let (dx, dy) = (x - star_x, y - star_y);
            flux * (-(dx * dx + dy * dy) / 4.5).exp()
        })
        .sum()
}

fn mono_field(dx: f32, dy: f32, gain: f32, frame: usize) -> LinearImage {
    let stars = stars();
    let data = (0..WIDTH * HEIGHT)
        .map(|index| {
            let (x, y) = ((index % WIDTH) as f32, (index / WIDTH) as f32);
            // A gentle gradient, as light pollution leaves.
            let sky = 100.0 + 0.05 * x + 0.03 * y;
            gain * (sky + star_light(&stars, x - dx, y - dy)) + noise(index, frame, 3.0)
        })
        .collect();
    LinearImage::new(WIDTH, HEIGHT, 1, data).unwrap()
}

fn bayer_field(dx: f32, dy: f32, frame: usize) -> LinearImage {
    const SKY: [f32; 3] = [400.0, 300.0, 200.0];
    const COLOR: [f32; 3] = [1.0, 0.8, 0.6];
    let stars = stars();
    let data = (0..WIDTH * HEIGHT)
        .map(|index| {
            let (x, y) = (index % WIDTH, index / WIDTH);
            // RGGB: red, green / green, blue.
            let channel = [[0, 1], [1, 2]][y & 1][x & 1];
            SKY[channel]
                + COLOR[channel] * star_light(&stars, x as f32 - dx, y as f32 - dy)
                + noise(index, frame, 4.0)
        })
        .collect();
    LinearImage::new(WIDTH, HEIGHT, 1, data).unwrap()
}

fn vignetting(index: usize) -> f32 {
    let x = (index % WIDTH) as f32 / WIDTH as f32 - 0.5;
    let y = (index / WIDTH) as f32 / HEIGHT as f32 - 0.5;
    1.0 - 0.6 * (x * x + y * y)
}

fn write_raw(
    directory: &tempfile::TempDir,
    stem: &str,
    frame: usize,
    data: Vec<f32>,
    role: &str,
) -> PathBuf {
    let path = directory.path().join(format!("{stem}-{frame}.fits"));
    let image = LinearImage::new(WIDTH, HEIGHT, 1, data).unwrap();
    let cards = [
        WriteHeaderCard::new("IMAGETYP", HeaderValue::String(role.into())),
        WriteHeaderCard::new("EXPTIME", HeaderValue::Float(60.0)),
    ];
    write_processed_image_fits_f32(&path, &image, &[], &cards).unwrap();
    path
}

/// Each channel's mean, root mean square and maximum over finite samples,
/// then the samples on a 5 by 5 grid of interior points.
fn summary(image: &LinearImage) -> Vec<f64> {
    let channels = image.channels;
    let mut values = Vec::new();
    for channel in 0..channels {
        let (mut sum, mut squares, mut maximum, mut count) = (0.0_f64, 0.0_f64, f64::MIN, 0);
        for &sample in image.data.iter().skip(channel).step_by(channels) {
            if sample.is_finite() {
                let sample = f64::from(sample);
                sum += sample;
                squares += sample * sample;
                maximum = maximum.max(sample);
                count += 1;
            }
        }
        values.extend([sum / count as f64, (squares / count as f64).sqrt(), maximum]);
    }
    for row in 0..5 {
        for column in 0..5 {
            let x = 16 + column * (image.width - 32) / 4;
            let y = 16 + row * (image.height - 32) / 4;
            for channel in 0..channels {
                values.push(f64::from(
                    image.data[(y * image.width + x) * channels + channel],
                ));
            }
        }
    }
    values
}

fn check(case: &str, actual: &[f64], expected: &[f64]) {
    let same = actual.len() == expected.len()
        && actual.iter().zip(expected).all(|(actual, expected)| {
            (actual - expected).abs() <= TOLERANCE * expected.abs().max(1.0)
        });
    let recorded = actual
        .iter()
        .map(|value| format!("{value:.7e}"))
        .collect::<Vec<_>>()
        .join(", ");
    assert!(
        same,
        "{case}: the output changed. If that is intended, set OUTPUT_VERSION in \
         seiza-stacking/src/lib.rs to the release that will ship this change, \
         and record the new output:\nconst {case}: &[f64] = &[{recorded}];"
    );
}
