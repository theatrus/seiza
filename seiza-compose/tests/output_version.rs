//! Holds [`seiza_compose::OUTPUT_VERSION`] to the output it promises.
//!
//! Each case records a summary of a composition on fixed inputs. When a
//! change alters one on purpose, set `OUTPUT_VERSION` in
//! `seiza-compose/src/lib.rs` to the release that ships the change, and
//! record the new output the failure prints. Never re-record without moving
//! the version, and never loosen `TOLERANCE`.

use seiza_compose::{
    ColorComposition, ColorCrop, ColorNormalization, ColorOptions, ColorTransfer, ForaxxOptions,
    NarrowbandMatrix, NarrowbandMix, NarrowbandPalette, combine_lrgb, combine_narrowband,
    combine_narrowband_matrix, combine_rgb, combine_super_lrgb, combine_super_rgb,
};
use seiza_stacking::LinearImage;

const TOLERANCE: f64 = 1e-5;

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
         seiza-compose/src/lib.rs to the release that will ship this change, \
         and record the new output:\nconst {case}: &[f64] = &[{recorded}];"
    );
}

#[test]
fn the_output_version_is_a_release() {
    let parts: Vec<u32> = seiza_compose::OUTPUT_VERSION
        .split('.')
        .map(|part| part.parse().expect("a number"))
        .collect();
    assert_eq!(parts.len(), 3, "{}", seiza_compose::OUTPUT_VERSION);
}

/// A narrowband and a broadband field of 32 stars on a flat sky, as the
/// continuum fits in `seiza-stacking` measure them.
fn star_fields() -> (LinearImage, LinearImage) {
    let (width, height) = (400, 240);
    let mut n = vec![10.0_f32; width * height];
    let mut b = n.clone();
    for i in 0..32 {
        let p = [30.0 + (i % 8) as f64 * 45.0, 30.0 + (i / 8) as f64 * 50.0];
        for y in (p[1] as usize - 12)..=(p[1] as usize + 12) {
            for x in (p[0] as usize - 12)..=(p[0] as usize + 12) {
                let r2 = (x as f64 - p[0]).powi(2) + (y as f64 - p[1]).powi(2);
                let flux = 1500.0 + 50.0 * i as f64;
                n[y * width + x] += (flux / (2.0 * std::f64::consts::PI * 1.6_f64.powi(2))
                    * (-r2 / (2.0 * 1.6_f64.powi(2))).exp())
                    as f32;
                b[y * width + x] += (flux / (0.35 * 2.0 * std::f64::consts::PI * 1.4_f64.powi(2))
                    * (-r2 / (2.0 * 1.4_f64.powi(2))).exp())
                    as f32;
            }
        }
    }
    (
        LinearImage::new(width, height, 1, n).unwrap(),
        LinearImage::new(width, height, 1, b).unwrap(),
    )
}

/// A mono channel of a faint sky with structure and a few bright stars, the
/// same for every run. `gap` leaves one pixel uncovered.
fn channel(seed: u32, level: f32, gap: bool) -> LinearImage {
    let (width, height) = (24, 18);
    let mut data: Vec<f32> = (0..width * height)
        .map(|index| {
            let (x, y) = ((index % width) as f32, (index / width) as f32);
            let mixed = (index as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed);
            let noise = (mixed % 1000) as f32 / 1000.0 - 0.5;
            // Each channel's nebula sits and spreads differently, so a
            // blend of them shows which weights it used.
            let (cx, cy) = (6.0 + (seed % 5) as f32 * 3.0, 4.0 + (seed % 3) as f32 * 4.0);
            let spread = 20.0 + (seed % 4) as f32 * 15.0;
            let nebula = (-((x - cx).powi(2) + (y - cy).powi(2)) / spread).exp();
            let ridge = (0.5 + 0.5 * (x * 0.4 + seed as f32).sin()) * 0.15;
            level * (0.05 + 0.4 * nebula + ridge + 0.01 * noise)
        })
        .collect();
    for star in [(4, 3), (19, 14), (11, 9)] {
        data[star.1 * width + star.0] = level * 0.95;
    }
    if gap {
        data[width + 1] = f32::NAN;
    }
    LinearImage::new(width, height, 1, data).unwrap()
}

/// A composition's channel sums over covered pixels, its uncovered pixels,
/// its size, and three of its pixels.
fn summary(composition: &ColorComposition) -> Vec<f64> {
    let image = &composition.image;
    let mut sums = [0.0_f64; 3];
    let mut uncovered = 0.0;
    for pixel in image.data.chunks(3) {
        if pixel.iter().any(|value| !value.is_finite()) {
            uncovered += 1.0;
            continue;
        }
        for (sum, value) in sums.iter_mut().zip(pixel) {
            *sum += f64::from(*value);
        }
    }
    let mut out = vec![
        sums[0],
        sums[1],
        sums[2],
        uncovered,
        image.width as f64,
        image.height as f64,
    ];
    for (x, y) in [(12, 9), (4, 3), (20, 2)] {
        if x < image.width && y < image.height {
            let at = (y * image.width + x) * 3;
            out.extend(image.data[at..at + 3].iter().map(|value| f64::from(*value)));
        }
    }
    out
}

const COMBINE_RGB: &[f64] = &[
    8.2225182e1,
    9.4104822e1,
    9.5219168e1,
    1.0000000e0,
    2.4000000e1,
    1.8000000e1,
    3.6044863e-1,
    4.9007085e-1,
    4.2288914e-1,
    1.0000000e0,
    1.0000000e0,
    1.0000000e0,
    1.0187767e-1,
    5.1039554e-2,
    2.7958962e-1,
];
const COMBINE_RGB_UNNORMALIZED: &[f64] = &[
    1.0211423e2,
    8.5419818e1,
    6.5073843e1,
    1.0000000e0,
    2.4000000e1,
    1.8000000e1,
    3.8643578e-1,
    3.9349329e-1,
    2.5960705e-1,
    9.4999999e-1,
    7.5999999e-1,
    5.6999999e-1,
    1.5858643e-1,
    7.7943780e-2,
    1.8253492e-1,
];
const COMBINE_RGB_INSCRIBED: &[f64] = &[
    7.6109627e1,
    8.8235906e1,
    9.2210770e1,
    0.0000000e0,
    2.2000000e1,
    1.8000000e1,
    3.1190151e-1,
    5.0025308e-1,
    4.3464661e-1,
    2.0988511e-1,
    4.1301601e-2,
    1.3750444e-1,
    3.4356676e-2,
    8.4832283e-3,
    2.1956989e-1,
];
const COMBINE_LRGB: &[f64] = &[
    6.1778548e1,
    6.9698174e1,
    1.0530075e2,
    1.0000000e0,
    2.4000000e1,
    1.8000000e1,
    2.1889748e-1,
    2.9761598e-1,
    2.5681710e-1,
    1.0000000e0,
    1.0000000e0,
    1.0000000e0,
    1.2243648e-1,
    6.1339285e-2,
    3.3601055e-1,
];
const COMBINE_SUPER_RGB: &[f64] = &[
    2.5757073e2,
    2.6767649e2,
    3.5107214e2,
    1.0000000e0,
    2.4000000e1,
    1.8000000e1,
    1.0029187e0,
    1.3635819e0,
    1.1766542e0,
    3.0000000e0,
    3.0000000e0,
    3.0000000e0,
    5.6239092e-1,
    2.8175145e-1,
    1.5434066e0,
];
const COMBINE_SHO: &[f64] = &[
    1.0127385e2,
    8.1200960e1,
    9.5930370e1,
    1.0000000e0,
    2.4000000e1,
    1.8000000e1,
    4.2318171e-1,
    1.7048647e-1,
    2.1263912e-1,
    1.0000000e0,
    9.9999994e-1,
    1.0000000e0,
    1.8086252e-1,
    1.1599398e-1,
    1.9074835e-1,
];
const COMBINE_HOO: &[f64] = &[
    8.1227248e1,
    9.6117965e1,
    9.6117965e1,
    0.0000000e0,
    2.4000000e1,
    1.8000000e1,
    1.7048647e-1,
    2.1263912e-1,
    2.1263912e-1,
    9.9999994e-1,
    1.0000000e0,
    1.0000000e0,
    1.1599398e-1,
    1.9074835e-1,
    1.9074835e-1,
];
const COMBINE_FORAXX_SHO: &[f64] = &[
    9.9562471e1,
    1.0299738e2,
    1.0164553e2,
    1.0000000e0,
    2.4000000e1,
    1.8000000e1,
    2.9154080e-1,
    2.2722107e-1,
    2.2716345e-1,
    1.0000000e0,
    1.0000000e0,
    1.0000000e0,
    1.6079244e-1,
    2.0267820e-1,
    2.0416515e-1,
];
const COMBINE_RGB_BOUNDS: &[f64] = &[
    8.2225182e1,
    9.4104822e1,
    9.5219168e1,
    1.0000000e0,
    2.4000000e1,
    1.8000000e1,
    3.6044863e-1,
    4.9007085e-1,
    4.2288914e-1,
    1.0000000e0,
    1.0000000e0,
    1.0000000e0,
    1.0187767e-1,
    5.1039554e-2,
    2.7958962e-1,
];
const COMBINE_RGB_DISPLAY: &[f64] = &[
    1.0211423e2,
    8.5419818e1,
    6.5073843e1,
    1.0000000e0,
    2.4000000e1,
    1.8000000e1,
    3.8643578e-1,
    3.9349329e-1,
    2.5960705e-1,
    9.4999999e-1,
    7.5999999e-1,
    5.6999999e-1,
    1.5858643e-1,
    7.7943780e-2,
    1.8253492e-1,
];
const COMBINE_SUPER_LRGB: &[f64] = &[
    3.1058643e2,
    3.2691467e2,
    4.6069357e2,
    1.0000000e0,
    2.4000000e1,
    1.8000000e1,
    1.1611513e0,
    1.5787170e0,
    1.3622975e0,
    4.0000000e0,
    4.0000000e0,
    4.0000000e0,
    6.9363832e-1,
    3.4750491e-1,
    1.9035975e0,
];
const COMBINE_MATRIX: &[f64] = &[
    8.9230120e1,
    9.1511547e1,
    9.6464717e1,
    1.0000000e0,
    2.4000000e1,
    1.8000000e1,
    2.7156457e-1,
    1.9999333e-1,
    2.3369338e-1,
    1.0000000e0,
    1.0000000e0,
    1.0000000e0,
    1.4194140e-1,
    1.6832203e-1,
    1.8975976e-1,
];
const COMBINE_FORAXX_SHO_DISPLAY: &[f64] = &[
    8.6211399e1,
    4.7538344e1,
    4.4591856e1,
    1.0000000e0,
    2.4000000e1,
    1.8000000e1,
    1.9564444e-1,
    1.0221745e-1,
    9.9926628e-2,
    5.8500451e-1,
    6.7725003e-1,
    3.8000000e-1,
    1.4591879e-1,
    9.3089581e-2,
    9.2139825e-2,
];
const COMBINE_FORAXX_HOO: &[f64] = &[
    1.0246217e2,
    1.0357173e2,
    1.0184638e2,
    0.0000000e0,
    2.4000000e1,
    1.8000000e1,
    2.2762674e-1,
    2.2719137e-1,
    2.2716345e-1,
    9.9999994e-1,
    9.9999994e-1,
    1.0000000e0,
    1.5835716e-1,
    2.0251034e-1,
    2.0416515e-1,
];

#[test]
fn rgb_compositions_keep_their_output() {
    let (r, g, b) = (
        channel(1, 1.0, true),
        channel(2, 0.8, false),
        channel(3, 0.6, false),
    );
    let options = ColorOptions::default();
    check(
        "COMBINE_RGB",
        &summary(&combine_rgb(&r, &g, &b, &options).unwrap()),
        COMBINE_RGB,
    );
    let none = ColorOptions {
        normalization: ColorNormalization::None,
        ..ColorOptions::default()
    };
    check(
        "COMBINE_RGB_UNNORMALIZED",
        &summary(&combine_rgb(&r, &g, &b, &none).unwrap()),
        COMBINE_RGB_UNNORMALIZED,
    );
    let inscribed = ColorOptions {
        crop: ColorCrop::Inscribed,
        ..ColorOptions::default()
    };
    check(
        "COMBINE_RGB_INSCRIBED",
        &summary(&combine_rgb(&r, &g, &b, &inscribed).unwrap()),
        COMBINE_RGB_INSCRIBED,
    );
    let l = channel(4, 1.2, false);
    check(
        "COMBINE_LRGB",
        &summary(&combine_lrgb(&l, &r, &g, &b, 0.7, &options).unwrap()),
        COMBINE_LRGB,
    );
    check(
        "COMBINE_SUPER_RGB",
        &summary(&combine_super_rgb(&r, &g, &b, &options).unwrap()),
        COMBINE_SUPER_RGB,
    );
    check(
        "COMBINE_SUPER_LRGB",
        &summary(&combine_super_lrgb(&l, &r, &g, &b, &options).unwrap()),
        COMBINE_SUPER_LRGB,
    );
    let bounds = ColorOptions {
        crop: ColorCrop::Bounds,
        ..ColorOptions::default()
    };
    check(
        "COMBINE_RGB_BOUNDS",
        &summary(&combine_rgb(&r, &g, &b, &bounds).unwrap()),
        COMBINE_RGB_BOUNDS,
    );
    let display = ColorOptions {
        normalization: ColorNormalization::None,
        input_transfer: ColorTransfer::DisplayReferred,
        ..ColorOptions::default()
    };
    check(
        "COMBINE_RGB_DISPLAY",
        &summary(&combine_rgb(&r, &g, &b, &display).unwrap()),
        COMBINE_RGB_DISPLAY,
    );
}

#[test]
fn custom_and_display_narrowband_compositions_keep_their_output() {
    let (ha, oiii, sii) = (
        channel(5, 1.0, false),
        channel(6, 0.4, false),
        channel(7, 0.3, true),
    );
    let mix = |ha: f32, oiii: f32, sii: f32| NarrowbandMix { ha, oiii, sii };
    let matrix = NarrowbandMatrix {
        red: mix(0.6, 0.0, 0.4),
        green: mix(0.3, 0.7, 0.0),
        blue: mix(0.0, 0.9, 0.1),
    };
    check(
        "COMBINE_MATRIX",
        &summary(
            &combine_narrowband_matrix(&ha, &oiii, Some(&sii), matrix, &ColorOptions::default())
                .unwrap(),
        ),
        COMBINE_MATRIX,
    );
    let display = ColorOptions {
        normalization: ColorNormalization::None,
        input_transfer: ColorTransfer::DisplayReferred,
        ..ColorOptions::default()
    };
    check(
        "COMBINE_FORAXX_SHO_DISPLAY",
        &summary(
            &combine_narrowband(
                &ha,
                &oiii,
                Some(&sii),
                NarrowbandPalette::ForaxxSho,
                &display,
                &ForaxxOptions::default(),
            )
            .unwrap(),
        ),
        COMBINE_FORAXX_SHO_DISPLAY,
    );
}

#[test]
fn narrowband_compositions_keep_their_output() {
    let (ha, oiii, sii) = (
        channel(5, 1.0, false),
        channel(6, 0.4, false),
        channel(7, 0.3, true),
    );
    let options = ColorOptions::default();
    let plain = ForaxxOptions::default();
    let smoothed = ForaxxOptions {
        mask_smoothing: 1.5,
        ..ForaxxOptions::default()
    };
    let run = |sii: Option<&LinearImage>, palette, foraxx: &ForaxxOptions| {
        summary(&combine_narrowband(&ha, &oiii, sii, palette, &options, foraxx).unwrap())
    };
    check(
        "COMBINE_SHO",
        &run(Some(&sii), NarrowbandPalette::Sho, &plain),
        COMBINE_SHO,
    );
    check(
        "COMBINE_HOO",
        &run(None, NarrowbandPalette::Hoo, &plain),
        COMBINE_HOO,
    );
    check(
        "COMBINE_FORAXX_SHO",
        &run(Some(&sii), NarrowbandPalette::ForaxxSho, &smoothed),
        COMBINE_FORAXX_SHO,
    );
    check(
        "COMBINE_FORAXX_HOO",
        &run(None, NarrowbandPalette::ForaxxHoo, &plain),
        COMBINE_FORAXX_HOO,
    );
}

#[test]
fn narrowband_display_modes_keep_their_output() {
    let mut physical = LinearImage::new(
        1,
        4,
        3,
        vec![
            0.1, 0.05, 0.025, 0.1, 0.05, 0.025, 0.2, 0.1, 0.05, 0.5, 0.2, 0.1,
        ],
    )
    .unwrap();
    seiza_compose::balance_narrowband(&mut physical, &Default::default()).unwrap();
    let ha = LinearImage::new(1, 4, 1, physical.data.iter().step_by(3).copied().collect()).unwrap();
    let oiii = LinearImage::new(
        1,
        4,
        1,
        physical.data.iter().skip(1).step_by(3).copied().collect(),
    )
    .unwrap();
    let sii = LinearImage::new(
        1,
        4,
        1,
        physical.data.iter().skip(2).step_by(3).copied().collect(),
    )
    .unwrap();
    let sums: Vec<f64> = [&ha, &oiii, &sii]
        .iter()
        .map(|image| image.data.iter().map(|value| f64::from(*value)).sum())
        .collect();
    check(
        "NARROWBAND_BALANCE",
        &sums,
        &[9.0000001e-1, 6.5793314e-1, 5.4459879e-1],
    );
    let mut nebula = LinearImage::new(1, 1, 3, vec![0.1, 0.2, 0.3]).unwrap();
    let original = LinearImage::new(1, 1, 3, vec![0.4, 0.5, 0.6]).unwrap();
    let starless = nebula.clone();
    seiza_compose::screen_rgb_stars(&mut nebula, &original, &starless).unwrap();
    check(
        "RGB_STARS",
        &nebula
            .data
            .iter()
            .map(|v| f64::from(*v))
            .collect::<Vec<_>>(),
        &[4.0000004e-1, 5.0000000e-1, 6.0000002e-1],
    );
}

#[test]
fn star_excluded_balance_statistics_keep_their_output() {
    use seiza_compose::{NarrowbandBalanceOptions, balance_narrowband};
    let (n, b) = star_fields();
    let mut data = Vec::new();
    for (n, b) in n.data.iter().zip(&b.data) {
        data.extend_from_slice(&[*n / 1000.0, *b / 2500.0, *n / 2000.0]);
    }
    let mut image = LinearImage::new(n.width, n.height, 3, data).unwrap();
    let report = balance_narrowband(
        &mut image,
        &NarrowbandBalanceOptions {
            star_mask_radius: Some(8.0),
            ..Default::default()
        },
    )
    .unwrap();
    check(
        "STAR_EXCLUDED_BALANCE",
        &[
            report.masked_pixels as f64,
            report.detected_stars as f64,
            report.channels[0].samples as f64,
            report.channels[1].positive_deviation.unwrap(),
        ],
        &[6.3040000e3, 9.6000000e1, 8.9696000e4, 6.5510077e-9],
    );
}

fn rgb_star_reduction_fixture(center_gain: f32) -> (LinearImage, LinearImage, LinearImage) {
    let nebula = LinearImage::new(3, 3, 3, [0.1, 0.2, 0.3].repeat(9)).unwrap();
    let starless = LinearImage::new(3, 3, 3, vec![0.2; 27]).unwrap();
    let mut original = Vec::new();
    for (i, level) in [0.05, 0.2, 0.05, 0.2, 0.8, 0.2, 0.05, 0.2, 0.05]
        .iter()
        .enumerate()
    {
        let level = level * if i == 4 { center_gain } else { 1.0 };
        for color in [1.0, 0.5, 0.25] {
            original.push(1.0 - (1.0 - 0.2) * (1.0 - level * color));
        }
    }
    (
        nebula,
        LinearImage::new(3, 3, 3, original).unwrap(),
        starless,
    )
}

#[test]
fn reduced_rgb_stars_keep_their_output() {
    // Reference: half blend of centre .8 and cross-neighbour minimum .2
    // gives .5, or a common RGB star-layer gain of .625.
    let (mut nebula, original, starless) = rgb_star_reduction_fixture(1.0);
    seiza_compose::screen_rgb_stars_reduced(&mut nebula, &original, &starless, 0.5).unwrap();
    check(
        "RGB_STAR_REDUCTION",
        &nebula.data[12..15]
            .iter()
            .map(|v| f64::from(*v))
            .collect::<Vec<_>>(),
        &[0.55, 0.4, 0.3875],
    );
}

#[test]
fn independently_stretched_rgb_stars_keep_their_output() {
    // Baseline recorded on the unchanged MTF implementation at m=1/28;
    // a planted identity curve made this case fail before adding this API.
    let original = LinearImage::new(2, 1, 3, vec![0.001, 0.002, 0.004, 0.01, 0.02, 0.04]).unwrap();
    let zero = LinearImage::new(2, 1, 3, vec![0.0; 6]).unwrap();
    let stars = seiza_compose::stretch_rgb_stars(&original, &zero, 3.0).unwrap();
    let mut nebula = LinearImage::new(2, 1, 3, vec![0.2; 6]).unwrap();
    seiza_compose::screen_rgb_star_layer(&mut nebula, &stars, 0.0).unwrap();
    check(
        "INDEPENDENT_RGB_STAR_STRETCH",
        &nebula
            .data
            .iter()
            .map(|x| f64::from(*x))
            .collect::<Vec<_>>(),
        &[
            0.22105263, 0.24106464, 0.27826087, 0.37142857, 0.48421053, 0.62352941,
        ],
    );
}
