//! Holds [`seiza_stacking::OUTPUT_VERSION`] to the output it promises.
//!
//! Each case stacks or combines fixed synthetic frames and compares a summary
//! of the result with the one recorded here. A change that moves a summary
//! changes what callers' cached stacks and masters would hold: set
//! `OUTPUT_VERSION` to the release that ships the change, and record the new
//! summary, which the failure prints.

use seiza_fits::{HeaderValue, WriteHeaderCard};
use seiza_stacking::{
    BatchStackOptions, CalibrationMasters, CfaIntegration, Continue, DefectDetectionOptions,
    Demosaic, DrizzleOptions, FrameAcceptanceCriteria, FrameDisposition, FrameWeighting,
    GpuOptions, ImpulseFilterOptions, Interpolation, LinearImage, LiveStacker, MasterBuildOptions,
    MasterFrameKind, NormalizationMode, RegistrationModel, RejectionMode, StackOptions,
    StackSnapshot, build_master_from_fits, detect_defects, write_master_fits_f32,
    write_processed_image_fits_f32,
};
use std::path::PathBuf;

const WIDTH: usize = 192;
const HEIGHT: usize = 160;

/// Relative to the larger of the recorded value and 1. Platforms differ in
/// the last bits of `exp` and friends; an algorithm change moves far more.
const TOLERANCE: f64 = 1e-5;

#[test]
fn joint_canvas_preserves_filter_only_coverage_and_photometry() {
    use seiza_stacking::CanvasChannel;
    let directory = tempfile::tempdir().unwrap();
    let paths = mosaic_panels(&directory);
    let mut faint = Vec::new();
    for (i, path) in paths[2..].iter().enumerate() {
        let mut frame = seiza_stacking::FitsFrame::open(path).unwrap();
        for pixel in &mut frame.image.data {
            *pixel *= 0.01;
        }
        let path = directory.path().join(format!("faint-{i}.fits"));
        write_processed_image_fits_f32(&path, &frame.image, &frame.headers, &[]).unwrap();
        faint.push(path);
    }
    let groups = vec![
        CanvasChannel {
            paths: paths[..2].to_vec(),
            bias: None,
            dark: None,
            flat: None,
            dark_exposure_seconds: None,
        },
        CanvasChannel {
            paths: faint,
            bias: None,
            dark: None,
            flat: None,
            dark_exposure_seconds: None,
        },
    ];
    let mut options = StackOptions {
        normalization: NormalizationMode::Global,
        gpu: GpuOptions {
            enabled: false,
            ..Default::default()
        },
        ..Default::default()
    };
    options.acceptance.minimum_overlap_fraction = 0.1;
    let mut first = LiveStacker::open_fits_joint_canvas(
        &groups,
        0,
        options.clone(),
        usize::MAX,
        |_, _, _, _| Continue::Yes,
    )
    .unwrap();
    assert_eq!(first.view().accepted_frames, 0);
    assert!(first.view().width > MOSAIC_PANEL.0 * 3 / 2);
    let mut second = first
        .open_fits_on_canvas(
            &groups[1].paths,
            None,
            None,
            None,
            None,
            options.clone(),
            |_, _, _, _| Continue::Yes,
        )
        .unwrap();
    for path in &groups[0].paths {
        admit(first.push_fits(path).unwrap());
    }
    for path in &groups[1].paths {
        admit(second.push_fits(path).unwrap());
    }
    let a = first.snapshot().unwrap();
    let b = second.snapshot().unwrap();
    assert_eq!(
        (a.image.width, a.image.height),
        (b.image.width, b.image.height)
    );
    let a_only = a
        .image
        .data
        .iter()
        .zip(&b.image.data)
        .filter(|(a, b)| a.is_finite() && !b.is_finite())
        .count();
    let b_only = a
        .image
        .data
        .iter()
        .zip(&b.image.data)
        .filter(|(a, b)| !a.is_finite() && b.is_finite())
        .count();
    assert!(
        a_only > a.image.data.len() / 4 && b_only > b.image.data.len() / 4,
        "{a_only}, {b_only}"
    );
    let paired: Vec<_> = a
        .image
        .data
        .iter()
        .zip(&b.image.data)
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .collect();
    assert!(paired.len() > 1000);
    let ratio = paired.iter().map(|(_, b)| f64::from(**b)).sum::<f64>()
        / paired.iter().map(|(a, _)| f64::from(**a)).sum::<f64>();
    assert!((ratio - 0.01).abs() < 0.001, "{ratio}");
    assert!(
        LiveStacker::open_fits_joint_canvas(
            &groups,
            0,
            options.clone(),
            usize::MAX,
            |_, _, _, _| Continue::No
        )
        .is_err()
    );
    assert!(
        LiveStacker::open_fits_joint_canvas(
            &groups,
            2,
            options.clone(),
            usize::MAX,
            |_, _, _, _| Continue::Yes
        )
        .is_err()
    );
    assert!(
        LiveStacker::open_fits_joint_canvas(&groups, 0, options, 10, |_, _, _, _| Continue::Yes)
            .is_err()
    );
}

#[test]
fn shared_mosaic_canvas_keeps_geometry_and_channel_brightness_separate() {
    let directory = tempfile::tempdir().unwrap();
    let paths = mosaic_panels(&directory);
    let channel: Vec<_> = paths
        .iter()
        .enumerate()
        .map(|(index, path)| {
            let mut frame = seiza_stacking::FitsFrame::open(path).unwrap();
            for pixel in &mut frame.image.data {
                *pixel *= 0.25;
            }
            let path = directory.path().join(format!("oiii-{index}.fits"));
            write_processed_image_fits_f32(&path, &frame.image, &frame.headers, &[]).unwrap();
            path
        })
        .collect();
    let mut options = StackOptions {
        normalization: NormalizationMode::Global,
        gpu: GpuOptions {
            enabled: false,
            ..GpuOptions::default()
        },
        ..StackOptions::default()
    };
    options.acceptance.minimum_overlap_fraction = 0.1;
    let mut source = LiveStacker::open_fits_expanding(
        &paths,
        None,
        None,
        None,
        None,
        options.clone(),
        usize::MAX,
        |_, _, _, _| Continue::Yes,
    )
    .unwrap();
    let context = directory.path().join("canvas.seiza-stack");
    for path in &paths {
        admit(source.push_fits(path).unwrap());
    }
    source.save_context(&context).unwrap();
    let restored = LiveStacker::open_context(&context).unwrap();
    let mut first = source
        .open_fits_on_canvas(
            &paths,
            None,
            None,
            None,
            None,
            options.clone(),
            |_, _, _, _| Continue::Yes,
        )
        .unwrap();
    let mut second = restored
        .open_fits_on_canvas(
            &channel,
            None,
            None,
            None,
            None,
            options.clone(),
            |_, _, _, _| Continue::Yes,
        )
        .unwrap();
    assert!(!second.reference_is_integrated());
    assert_eq!(second.view().accepted_frames, 0);
    assert_eq!(source.view().accepted_frames as usize, paths.len());
    assert_eq!(first.reference_headers(), second.reference_headers());
    for path in &paths {
        admit(first.push_fits(path).unwrap());
    }
    for path in &channel {
        admit(second.push_fits(path).unwrap());
    }
    let a = first.snapshot().unwrap();
    let b = second.snapshot().unwrap();
    assert_eq!(
        (a.image.width, a.image.height),
        (b.image.width, b.image.height)
    );
    let mut paired = 0;
    for (&a, &b) in a.image.data.iter().zip(&b.image.data) {
        if a.is_finite() && b.is_finite() {
            assert!(
                (a * 0.25 - b).abs() < 0.005 * a.abs().max(1.0),
                "{a} -> {b}"
            );
            paired += 1;
        }
    }
    assert!(paired > a.image.data.len() / 2);
    assert!(
        source
            .open_fits_on_canvas(
                &channel,
                None,
                None,
                None,
                None,
                options.clone(),
                |_, _, _, _| Continue::No,
            )
            .is_err()
    );
    options.bin_factor = 2;
    assert!(
        source
            .open_fits_on_canvas(&channel, None, None, None, None, options, |_, _, _, _| {
                Continue::Yes
            },)
            .err()
            .unwrap()
            .to_string()
            .contains("bin factor")
    );
}

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
    1.4695953e2, 2.8119425e2, 3.9687632e3, 9.7643250e1, 1.0004932e2, 1.0116178e2,
    1.1477510e2, 1.0543606e2, 9.8268410e1, 1.0055153e2, 1.0238376e2, 1.0341871e2,
    1.0466166e2, 9.8374100e1, 1.0225138e2, 4.2145331e2, 1.0452397e2, 1.0633051e2,
    1.0034851e2, 1.0087305e2, 1.0359173e2, 1.0660538e2, 1.0766888e2, 1.0121416e2,
    1.7354532e2, 1.0512951e2, 1.0694588e2, 1.1127830e2,
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
const BINNED: &[f64] = &[
    1.4311715e2, 2.4653842e2, 2.9428250e3, 9.8566231e1, 1.0354869e2, 1.0185313e2,
    1.0289874e2, 1.0433101e2, 9.8669312e1, 1.0080196e2, 1.0262335e2, 1.5487537e3,
    1.0499112e2, 3.2784250e2, 1.0115610e2, 1.0298831e2, 1.0412970e2, 1.0742923e2,
    1.0033507e2, 2.2662872e2, 1.0329276e2, 1.0484836e2, 5.1373145e2, 1.0109541e2,
    1.1836398e2, 1.0411025e2, 1.0556331e2, 1.0695666e2,
];
#[rustfmt::skip]
const FULL_RESOLUTION: &[f64] = &[
    1.4310797e2, 2.6825418e2, 3.6725220e3, 9.7422989e1, 9.9715958e1, 1.0107971e2,
    1.1757957e2, 1.0521172e2, 9.8150475e1, 1.0046536e2, 1.0211687e2, 1.0329865e2,
    1.0460469e2, 9.8414558e1, 1.0147755e2, 1.0316610e2, 1.0423871e2, 1.0615415e2,
    1.0020127e2, 1.0096130e2, 1.0359206e2, 1.0612683e2, 1.0727988e2, 1.0100352e2,
    1.7025096e2, 1.0470015e2, 1.0667790e2, 1.1075444e2,
];
#[rustfmt::skip]
const SUPERPIXEL: &[f64] = &[
    4.5056642e2, 5.1461645e2, 3.7941848e3, 3.4991561e2, 4.2811706e2, 3.7417290e3,
    2.2169566e2, 2.4952920e2, 1.9389568e3, 4.0112073e2, 2.9977878e2, 2.0000548e2,
    4.0481573e2, 3.0254874e2, 2.0079019e2, 4.0097467e2, 2.9991870e2, 2.0088794e2,
    4.0043704e2, 3.0047525e2, 1.9980557e2, 3.9878918e2, 3.0012012e2, 2.0062187e2,
    3.9961615e2, 2.9866693e2, 2.0009059e2, 4.0098495e2, 3.0070251e2, 1.9977524e2,
    3.9968878e2, 3.0110107e2, 2.0002367e2, 2.2513330e3, 2.0821128e3, 9.0497809e2,
    3.9951294e2, 2.9996356e2, 1.9996620e2, 7.0606750e2, 6.3206653e2, 3.0315860e2,
    4.0049579e2, 3.0004858e2, 1.9898215e2, 4.0048694e2, 2.9974792e2, 1.9960785e2,
    4.0029657e2, 2.9866653e2, 1.9957559e2, 4.0279974e2, 3.0297729e2, 2.0015086e2,
    4.0091287e2, 2.9898071e2, 1.9929868e2, 5.5209436e2, 4.5315872e2, 2.6305588e2,
    3.9998114e2, 2.9879013e2, 2.0003249e2, 3.9933450e2, 3.0003784e2, 1.9961035e2,
    9.3254974e2, 8.2864453e2, 4.1087515e2, 3.9988791e2, 3.0035577e2, 1.9985046e2,
    4.1872589e2, 3.2848203e2, 2.0588953e2, 4.0029883e2, 2.9965512e2, 2.0006384e2,
    3.9799783e2, 2.9996805e2, 1.9934007e2, 3.9979184e2, 2.9921347e2, 2.0013539e2,
];
#[rustfmt::skip]
const BINNED_DRIZZLED: &[f64] = &[
    1.4310613e2, 2.6877686e2, 3.6995620e3, 9.7408691e1, 9.9783958e1, 1.0105090e2,
    1.1783337e2, 1.0524158e2, 9.8157951e1, 1.0044919e2, 1.0214176e2, 1.0327659e2,
    1.0460580e2, 9.8405037e1, 1.0147527e2, 1.0319716e2, 1.0421772e2, 1.0610207e2,
    1.0018525e2, 1.0092257e2, 1.0354213e2, 1.0615263e2, 1.0729715e2, 1.0100514e2,
    1.7140848e2, 1.0446319e2, 1.0666625e2, 1.1061572e2,
];
#[rustfmt::skip]
const MOSAIC_FULL_RESOLUTION: &[f64] = &[
    2.2342357e2, 4.1493165e2, 5.0366611e3, 1.0009515e2, 1.0359967e2, 1.0392278e2,
    1.4567566e2, 1.9966444e2, 1.0127506e2, 2.2257149e2, 9.9761011e2, 1.3664514e2,
    1.5093257e2, 1.0202713e2, 1.0343097e2, 1.6650745e2, 1.7741840e2, 1.0598458e2,
    1.0179967e2, 1.0169817e2, 1.0327754e2, 1.0577757e2, 1.0626158e2, 2.4992326e2,
    1.8955243e2, 1.0545649e2, 7.9738617e2, 1.0734964e2,
];
#[rustfmt::skip]
const LENS_MOSAIC_SIBLINGS: &[f64] = &[1.1248227e2, 1.3248900e2, 4.2809961e3, 9.9621635e1, 1.0401961e2, 1.0109291e2, 1.1935750e2, 9.7427887e1, 9.9319954e1, 1.0011208e2, 1.0266560e2, 1.6070786e2, 1.0302679e2, 1.0011884e2, 1.0015338e2, 1.0428073e2, 1.0138506e2, 1.0049202e2, 9.9340057e1, 9.8439827e1, 1.0166298e2, 1.0112801e2, 1.0463013e2, 1.1425014e2, 1.0245366e2, 1.0088453e2, 1.0183040e2, 9.8820175e1, 6.0000000e0, 6.1900000e2, 1.0160000e3];
#[rustfmt::skip]
const MOSAIC_CHANGING_GRADIENTS: &[f64] = &[2.5618486e2, 5.0731940e2, 5.8631538e3, 9.9515686e1, 1.0304424e2, 1.0594209e2, 3.6188834e2, 1.1663352e2, 9.9784561e1, 1.5832272e2, 2.3292229e3, 1.4172540e2, 1.1663142e2, 1.0048741e2, 1.0461007e2, 3.6352579e2, 3.1940692e2, 1.1519485e2, 9.7367989e1, 1.0164168e2, 1.0775472e2, 1.1233539e2, 1.4158466e2, 6.1375110e2, 1.1946843e2, 1.2367524e2, 4.9480072e2, 2.2657356e2];
const MILD_LENS_MOSAIC: &[f64] = &[
    1.2961286e2,
    1.6208965e2,
    4.5229419e3,
    1.1021458e2,
    9.6230270e1,
    9.4510620e1,
    9.8815392e1,
    8.9425873e1,
    9.1626221e1,
    9.0163712e1,
    1.0253673e2,
    1.0871326e2,
    1.3086557e2,
    1.0036616e2,
    1.0005003e2,
    1.4361838e2,
    1.1466187e2,
    1.0847977e2,
    1.1012643e2,
    9.3992554e1,
    1.2721870e2,
    9.7976517e1,
    5.2863477e2,
    1.1205984e2,
    1.1021914e2,
    9.6290451e1,
    1.0917767e2,
    9.0285599e1,
    1.0000000e0,
    8.0000000e0,
    9.0300000e2,
    7.0300000e2,
];
const CLIPPED_BAYER_MAP: &[f64] = &[
    1.3575466e-2,
    1.0238972e-1,
    1.0000000e0,
    0.0000000e0,
    0.0000000e0,
    0.0000000e0,
    0.0000000e0,
    0.0000000e0,
    0.0000000e0,
    0.0000000e0,
    0.0000000e0,
    0.0000000e0,
    0.0000000e0,
    0.0000000e0,
    0.0000000e0,
    0.0000000e0,
    0.0000000e0,
    0.0000000e0,
    0.0000000e0,
    0.0000000e0,
    0.0000000e0,
    0.0000000e0,
    0.0000000e0,
    0.0000000e0,
    0.0000000e0,
    0.0000000e0,
    0.0000000e0,
    0.0000000e0,
    4.1703831e2,
    3.9800000e2,
];
const MOSAIC_BACKGROUND_REFERENCE: &[f64] = &[
    2.5444598e2,
    5.0681991e2,
    5.8689082e3,
    9.8751671e1,
    1.0244911e2,
    1.0171320e2,
    3.1581039e2,
    1.1710111e2,
    9.9339058e1,
    1.5773672e2,
    2.3780955e3,
    1.2924583e2,
    1.1780296e2,
    1.0039929e2,
    1.0475031e2,
    3.5805563e2,
    3.2299777e2,
    1.1788648e2,
    9.7663734e1,
    1.0216030e2,
    1.0442368e2,
    1.0878123e2,
    1.5059869e2,
    6.1411371e2,
    1.2025177e2,
    1.2162159e2,
    4.8166986e2,
    2.4971623e2,
];
#[rustfmt::skip]
const SHARED_NEBULA_BACKGROUND: &[f64] = &[
    1.0662690e0, -2.7621416e1, 2.3148637e2, 2.6217258e2, 5.2560767e2, 1.3647876e2,
    3.5316037e2, 3.3801196e2, 1.8140225e2, 1.9095937e2, 1.2601199e2, 2.6254865e2,
    4.7366315e2, 2.6814981e2, 2.3112440e2, 7.4395386e1, 1.3285132e2, 5.1343073e2,
    3.0254535e2, 1.4823747e2, 7.5180016e1, 9.7638176e1, 4.0146280e2, 2.9319531e2,
    1.0922065e2, 1.1703941e2, 1.4173183e2, 2.4023395e2, 3.1646753e2, 1.5533604e2,
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

#[rustfmt::skip]
const LANCZOS_FAINT_SKY: &[f64] = &[
    8.4681377e1, 3.2267679e2, 4.9992329e3, 3.3310356e1, 3.0939747e1, 3.5212967e1,
    6.0685131e1, 2.6831116e1, 3.0434317e1, 3.4744232e1, 3.2350834e1, 2.8062109e1,
    1.3460752e1, 2.1576740e1, 3.8814690e1, 3.4215744e1, 2.7423660e1, 2.4253744e1,
    3.6647362e1, 2.5796043e1, 2.5571917e1, 3.0842657e1, 2.8100803e1, 3.3470314e1,
    1.2654603e2, 2.6511717e1, 2.6398441e1, 4.7701859e1,
];

#[rustfmt::skip]
const NO_REJECTION: &[f64] = &[
    1.4835552e2, 2.8592203e2, 3.9204758e3, 9.7156273e1, 9.9581932e1, 1.0100323e2,
    1.1871532e2, 1.0544968e2, 9.7902931e1, 1.0035307e2, 1.0213988e2, 1.0341476e2,
    1.0482466e2, 9.8180191e1, 1.0143280e2, 3.7782477e2, 1.0442571e2, 1.0646262e2,
    1.0008590e2, 1.0090702e2, 1.0369869e2, 1.0641956e2, 1.0768224e2, 1.0096002e2,
    1.7523038e2, 1.0491113e2, 1.0702999e2, 1.1143148e2,
];

#[rustfmt::skip]
const NO_NORMALIZATION: &[f64] = &[
    1.5167379e2, 2.8597475e2, 3.8939426e3, 1.0108342e2, 1.0350527e2, 1.0490619e2,
    1.2246421e2, 1.0930960e2, 1.0183619e2, 1.0425922e2, 1.0476214e2, 1.0602847e2,
    1.0869564e2, 1.0134892e2, 1.0532454e2, 3.8471097e2, 1.0830507e2, 1.1031990e2,
    1.0398833e2, 1.0323552e2, 1.0593650e2, 1.1028071e2, 1.1153486e2, 1.0413821e2,
    1.7847832e2, 1.0878827e2, 1.1088239e2, 1.1523884e2,
];

#[rustfmt::skip]
const LOCAL_NORMALIZATION: &[f64] = &[
    1.7070597e2, 4.0338009e2, 5.9989072e3, 9.7872574e1, 1.0013402e2, 1.0110046e2,
    1.2720847e2, 1.0561763e2, 9.8493698e1, 1.0039824e2, 1.0240100e2, 1.0306364e2,
    1.0442137e2, 9.8295601e1, 1.0182735e2, 4.3685776e2, 1.0421967e2, 1.0612961e2,
    1.0049208e2, 1.0049834e2, 1.0341342e2, 1.0668638e2, 1.0757182e2, 1.0133639e2,
    2.1256087e2, 1.0495863e2, 1.0714788e2, 1.1260919e2,
];

#[rustfmt::skip]
const AFFINE: &[f64] = &[
    1.4835515e2, 2.8592223e2, 3.9203108e3, 9.7157402e1, 9.9582314e1, 1.0100320e2,
    1.1871250e2, 1.0544969e2, 9.7903038e1, 1.0035374e2, 1.0213974e2, 1.0341547e2,
    1.0482413e2, 9.8179245e1, 1.0143272e2, 3.7778775e2, 1.0442602e2, 1.0646307e2,
    1.0008549e2, 1.0090685e2, 1.0369928e2, 1.0642014e2, 1.0768219e2, 1.0096152e2,
    1.7520320e2, 1.0491122e2, 1.0702918e2, 1.1143520e2,
];

#[rustfmt::skip]
const COSMETIC: &[f64] = &[
    1.4616230e2, 2.8399422e2, 3.9276370e3, 9.7141769e1, 9.9593956e1, 1.0101359e2,
    1.1868999e2, 1.0546960e2, 9.7891510e1, 1.0035527e2, 1.0215276e2, 1.0342080e2,
    1.0483676e2, 9.8179405e1, 1.0143327e2, 1.0328603e2, 1.0444096e2, 1.0648162e2,
    1.0008395e2, 1.0091266e2, 1.0369848e2, 1.0642391e2, 1.0771923e2, 1.0098036e2,
    1.7590361e2, 1.0492743e2, 1.0704359e2, 1.1148314e2,
];

#[rustfmt::skip]
const DEFECTS: &[f64] = &[
    1.4624760e2, 2.8432367e2, 3.9301108e3, 9.7182053e1, 9.9597610e1, 1.0102315e2,
    1.1859837e2, 1.0546263e2, 9.7914978e1, 1.0031285e2, 1.0215680e2, 1.0341294e2,
    1.0480682e2, 9.8183441e1, 1.0145301e2, 1.0331266e2, 1.0442049e2, 1.0647482e2,
    1.0005818e2, 1.0092070e2, 1.0373038e2, 1.0635432e2, 1.0780189e2, 1.0091165e2,
    1.7984712e2, 1.0493559e2, 1.0705829e2, 1.1151440e2,
];

#[rustfmt::skip]
const BAYER_MHC: &[f64] = &[
    4.5674225e2, 5.6631180e2, 5.8373555e3, 3.4586085e2, 4.3695066e2, 4.6274951e3,
    2.2923085e2, 2.8381343e2, 2.8202678e3, 4.0082318e2, 3.0061011e2, 2.0076942e2,
    4.0168649e2, 3.0026694e2, 2.0142854e2, 3.9993750e2, 3.0143237e2, 1.9926010e2,
    4.1516977e2, 3.1114905e2, 2.1042918e2, 4.0130972e2, 3.0096793e2, 2.0039337e2,
    3.9918958e2, 2.9931934e2, 1.9999316e2, 4.0147205e2, 3.0045212e2, 1.9934431e2,
    4.0030157e2, 3.0056192e2, 2.0042407e2, 3.9908899e2, 2.9932553e2, 2.0045662e2,
    3.9813293e2, 2.9877103e2, 1.9830243e2, 3.9736792e2, 2.9860568e2, 1.9951256e2,
    4.0181107e2, 3.0142288e2, 2.0094212e2, 4.0031870e2, 2.9983911e2, 1.9966464e2,
    3.9877933e2, 2.9995380e2, 1.9970662e2, 3.9832327e2, 2.9977090e2, 1.9961600e2,
    4.0170343e2, 3.0076880e2, 2.0151447e2, 3.9818176e2, 2.9855777e2, 1.9941484e2,
    3.9889536e2, 3.0004382e2, 1.9915315e2, 4.0125354e2, 3.0041254e2, 2.0091064e2,
    3.9999753e2, 2.9934375e2, 1.9913477e2, 4.0034247e2, 3.0018558e2, 1.9928577e2,
    4.7586914e2, 3.6243701e2, 2.4689253e2, 3.9967886e2, 3.0090262e2, 1.9925745e2,
    4.0154788e2, 3.0144910e2, 2.0008621e2, 3.9854239e2, 2.9568179e2, 1.9733656e2,
];

#[rustfmt::skip]
const BAYER_BILINEAR: &[f64] = &[
    4.4532135e2, 5.0249783e2, 3.7279351e3, 3.3961835e2, 4.0067415e2, 3.5733315e3,
    2.2729018e2, 2.6711427e2, 2.3114009e3, 3.9958936e2, 3.0058551e2, 2.0092488e2,
    4.0055634e2, 2.9951251e2, 2.0120500e2, 4.0039825e2, 3.0130328e2, 1.9874049e2,
    4.3049490e2, 3.1959821e2, 2.1807420e2, 4.0045050e2, 3.0028745e2, 2.0023831e2,
    4.0000522e2, 2.9969037e2, 2.0008347e2, 4.0132108e2, 3.0008411e2, 1.9914508e2,
    4.0010211e2, 3.0023087e2, 2.0011893e2, 3.9914197e2, 2.9954105e2, 2.0062022e2,
    3.9895993e2, 2.9949704e2, 1.9900305e2, 3.9916818e2, 2.9955185e2, 1.9981927e2,
    4.0173218e2, 3.0050037e2, 2.0027942e2, 4.0112958e2, 3.0003711e2, 1.9961928e2,
    3.9919644e2, 3.0028970e2, 1.9992502e2, 3.9940820e2, 3.0018787e2, 1.9964224e2,
    4.0064685e2, 2.9991864e2, 2.0123351e2, 3.9855615e2, 2.9905096e2, 1.9965834e2,
    4.0002087e2, 3.0026233e2, 1.9891631e2, 4.0032013e2, 2.9961661e2, 2.0085553e2,
    4.0015137e2, 2.9959717e2, 1.9911827e2, 3.9965424e2, 2.9981110e2, 1.9964763e2,
    4.8951233e2, 3.8260168e2, 2.6975504e2, 3.9890515e2, 3.0054367e2, 1.9936063e2,
    4.0086646e2, 3.0089194e2, 1.9962515e2, 4.0250031e2, 3.0532333e2, 2.0750577e2,
];

#[rustfmt::skip]
const BAYER_DRIZZLE: &[f64] = &[
    4.5951580e2, 5.6755295e2, 6.0160151e3, 3.4746441e2, 4.4323055e2, 5.2259707e3,
    2.3047388e2, 2.8685669e2, 3.2797820e3, 3.9923260e2, 3.0393765e2, 2.0186697e2,
    4.0011829e2, 2.9658450e2, 2.0409583e2, 4.0099524e2, 3.0534592e2, 1.9661295e2,
    4.1587717e2, 3.3729636e2, 2.1399542e2, 4.0080438e2, 3.0006308e2, 2.0210049e2,
    4.0072708e2, 3.0219907e2, 1.9806848e2, 4.0350177e2, 3.0019653e2, 1.9883069e2,
    4.0032388e2, 3.0478967e2, 1.9957800e2, 3.9659116e2, 2.9667596e2, 1.9931903e2,
    3.9679935e2, 3.0547321e2, 1.9937373e2, 3.9883789e2, 2.9959451e2, 2.0000089e2,
    4.0429779e2, 2.9850314e2, 1.9934682e2, 4.0338095e2, 2.9771069e2, 1.9971652e2,
    3.9805411e2, 3.0309778e2, 1.9996103e2, 3.9771613e2, 2.9857166e2, 1.9869106e2,
    4.0010281e2, 2.9901306e2, 2.0293388e2, 3.9816422e2, 2.9785291e2, 1.9756851e2,
    4.0024710e2, 3.0012387e2, 1.9560718e2, 4.0186978e2, 2.9884662e2, 2.0124152e2,
    4.0112332e2, 2.9998157e2, 1.9951701e2, 3.9954153e2, 2.9632028e2, 2.0322150e2,
    4.9452203e2, 4.6048315e2, 2.5332556e2, 3.9893463e2, 3.0487146e2, 2.0192215e2,
    4.0237207e2, 3.0388763e2, 1.9660590e2, 4.0044229e2, 3.0016763e2, 2.0506046e2,
];

#[rustfmt::skip]
const HOST_INTEGRATED: &[f64] = &[
    1.5026973e2, 2.9195168e2, 4.1777139e3, 1.0074651e2, 1.0366454e2, 1.0547443e2,
    1.1940752e2, 1.0942122e2, 1.0227286e2, 1.0470843e2, 1.0725134e2, 1.0821719e2,
    1.0901280e2, 1.0415617e2, 1.0412110e2, 1.0803335e2, 1.0816547e2, 1.1112226e2,
    1.0462069e2, 1.0627757e2, 1.0888474e2, 1.1139577e2, 1.1152663e2, 1.0536192e2,
    1.6954681e2, 1.0875781e2, 1.1115813e2, 1.1303532e2,
];

#[rustfmt::skip]
const LARGE_LIVE: &[f64] = &[
    1.3779775e2, 4.2943252e2, 1.8050789e4, 9.7606865e1, 1.0074303e2, 1.0332143e2,
    1.0640565e2, 1.0970185e2, 1.0199506e2, 1.0470235e2, 1.0888734e2, 1.1169503e2,
    1.1423000e2, 1.0608360e2, 1.0971593e2, 1.1210521e2, 1.1694767e2, 1.1877633e2,
    1.1073438e2, 1.1466586e2, 1.1635000e2, 1.1988004e2, 1.2327108e2, 1.1514570e2,
    1.1829337e2, 1.2178257e2, 1.2421250e2, 1.2779749e2,
];

#[rustfmt::skip]
const LARGE_BAYER: &[f64] = &[
    4.3709554e2, 7.3621421e2, 2.6103078e4, 3.2695387e2, 5.3528112e2, 1.8510148e4,
    2.1808718e2, 3.5899090e2, 1.2386176e4, 4.0050531e2, 3.0018240e2, 2.0045538e2,
    3.9962393e2, 2.9976782e2, 1.9975221e2, 3.9953403e2, 3.0012613e2, 1.9953230e2,
    3.9951831e2, 2.9954590e2, 1.9981696e2, 4.0076828e2, 2.9883420e2, 1.9905986e2,
    3.9881314e2, 2.9911264e2, 2.0006700e2, 3.9864709e2, 3.0036853e2, 1.9870906e2,
    4.0169772e2, 3.0038361e2, 2.0111650e2, 4.0058640e2, 3.0095898e2, 2.0120891e2,
    4.0081522e2, 3.0000491e2, 2.0044273e2, 4.0016983e2, 2.9984689e2, 1.9977567e2,
    3.9921402e2, 2.9975696e2, 1.9996878e2, 3.9955930e2, 2.9950284e2, 1.9945724e2,
    4.0122397e2, 3.0141617e2, 2.0101453e2, 3.9955362e2, 3.0063062e2, 2.0016751e2,
    3.9889505e2, 3.0023480e2, 2.0014856e2, 4.0034567e2, 3.0036606e2, 2.0079869e2,
    3.9922552e2, 2.9927734e2, 1.9887303e2, 3.9959268e2, 2.9980222e2, 1.9983990e2,
    3.9948978e2, 2.9909891e2, 1.9988362e2, 3.9961865e2, 2.9905649e2, 1.9947746e2,
    3.9930807e2, 2.9961392e2, 2.0005275e2, 3.9946231e2, 3.0006506e2, 2.0044939e2,
    4.0022821e2, 3.0036121e2, 2.0021049e2, 4.0079575e2, 2.9973444e2, 1.9990431e2,
];

#[rustfmt::skip]
const LARGE_REINTEGRATED: &[f64] = &[
    1.3750326e2, 4.2837601e2, 1.7889000e4, 9.7606865e1, 1.0074303e2, 1.0332143e2,
    1.0640565e2, 1.0970185e2, 1.0199506e2, 1.0470235e2, 1.0888734e2, 1.1142345e2,
    1.1423000e2, 1.0608360e2, 1.0971593e2, 1.1210521e2, 1.1694767e2, 1.1877633e2,
    1.1073438e2, 1.1466586e2, 1.1635000e2, 1.1988004e2, 1.2327108e2, 1.1537320e2,
    1.1829337e2, 1.2178257e2, 1.2421250e2, 1.2779749e2,
];

#[rustfmt::skip]
const REINTEGRATED: &[f64] = &[
    1.4613894e2, 2.8383271e2, 3.9288564e3, 9.7148277e1, 9.9584862e1, 1.0101023e2,
    1.1875857e2, 1.0546034e2, 9.7901772e1, 1.0035715e2, 1.0214648e2, 1.0342279e2,
    1.0483615e2, 9.8175110e1, 1.0144085e2, 1.0327808e2, 1.0444199e2, 1.0647711e2,
    1.0008474e2, 1.0090938e2, 1.0370780e2, 1.0643605e2, 1.0770409e2, 1.0096039e2,
    1.7538022e2, 1.0492811e2, 1.0705235e2, 1.1145596e2,
];

#[rustfmt::skip]
const DRIZZLED: &[f64] = &[
    1.4613605e2, 2.8698624e2, 4.1971372e3, 9.6619240e1, 9.8821266e1, 1.0033887e2,
    1.0672266e2, 1.0450388e2, 9.7934311e1, 9.9977234e1, 1.0149843e2, 1.0543295e2,
    1.0556454e2, 9.8304504e1, 9.9548264e1, 1.0380981e2, 1.0412138e2, 1.0599422e2,
    9.8567917e1, 1.0117728e2, 1.0439243e2, 1.0632543e2, 1.0853813e2, 1.0066372e2,
    1.0316244e2, 1.0475816e2, 1.0750745e2, 1.0973293e2,
];

#[rustfmt::skip]
const MOSAIC: &[f64] = &[
    2.6282714e2, 5.3448414e2, 6.2964482e3, 1.0066447e2, 1.0196080e2, 1.0381591e2,
    2.3579346e2, 2.8704211e2, 1.0150757e2, 2.2715549e2, 1.4126470e3, 1.5307355e2,
    2.1004921e2, 1.0194611e2, 1.0400060e2, 1.9063486e2, 1.5825851e2, 1.0700441e2,
    1.0222562e2, 1.0330536e2, 1.0399973e2, 1.0613483e2, 1.0971373e2, 1.8902301e2,
    1.6568292e2, 1.0487677e2, 6.2708032e2, 1.0713017e2,
];

#[rustfmt::skip]
const NARROW_MOSAIC: &[f64] = &[
    2.6718343e2, 5.4215986e2, 9.4800508e3, 9.9738777e1, 1.3014086e3, 1.7974307e2,
    9.9623672e1, 9.7871735e1, 1.0279266e2, 1.0594302e2, 1.0311459e2, 1.0515709e2,
    1.0424060e2, 1.0895921e2, 1.1358248e2, 1.1000770e2, 1.3116101e3, 1.1037816e2,
    1.1672225e2, 1.1417844e2, 1.2913323e3, 1.0445881e3, 1.8293590e2, 1.1889386e2,
    1.1991624e2, 1.2088194e2, 1.1871208e2, 1.3691556e2,
];

#[rustfmt::skip]
const MOSAIC_QUADRATIC: &[f64] = &[
    2.6184846e2, 5.3270642e2, 6.4287822e3, 1.0098178e2, 1.0197623e2, 1.0388168e2,
    2.4814735e2, 2.4940155e2, 1.0158381e2, 2.2572668e2, 1.4497437e3, 1.5627357e2,
    2.0429832e2, 1.0203711e2, 1.0398335e2, 1.9151581e2, 1.5916586e2, 1.0734766e2,
    1.0271570e2, 1.0331667e2, 1.0390027e2, 1.0628079e2, 1.0931861e2, 1.8415189e2,
    1.6650204e2, 1.0509101e2, 6.8839563e2, 1.0732013e2,
];

#[rustfmt::skip]
const FLAT_MASTER_BIASED: &[f64] = &[
    9.9507321e-1, 9.9752802e-1, 1.1058390e0, 8.8442200e-1, 9.7103345e-1, 9.9948442e-1,
    9.7075170e-1, 8.8438344e-1, 9.6391153e-1, 1.0503253e0, 1.0790998e0, 1.0502412e0,
    9.6402162e-1, 9.9071038e-1, 1.0769354e0, 1.1056914e0, 1.0769274e0, 9.9073726e-1,
    9.6417946e-1, 1.0502132e0, 1.0793134e0, 1.0504037e0, 9.6409714e-1, 8.8441980e-1,
    9.7073954e-1, 9.9950612e-1, 9.7100949e-1, 8.8434505e-1,
];

#[rustfmt::skip]
const CALIBRATED_LANCZOS: &[f64] = &[
    7.6405936e1, 2.9063132e2, 4.5017817e3, 2.9910845e1, 3.0009436e1, 3.1982927e1,
    5.5067642e1, 2.3747440e1, 2.7948290e1, 3.1852222e1, 2.8892441e1, 2.5063713e1,
    9.6558657e0, 1.9564846e1, 3.4706512e1, 3.1363016e1, 2.4861242e1, 2.1682476e1,
    3.2296391e1, 2.3311283e1, 2.4124556e1, 2.7315498e1, 2.5832870e1, 3.1812588e1,
    1.1398070e2, 2.3346869e1, 2.5175215e1, 4.4047523e1,
];

#[rustfmt::skip]
const BIAS_MASTER: &[f64] = &[
    3.0007062e2, 3.0007339e2, 3.0441794e2, 2.9925772e2, 3.0073709e2, 2.9994614e2,
    3.0028802e2, 2.9983591e2, 3.0033102e2, 3.0190173e2, 3.0158337e2, 2.9967471e2,
    2.9826190e2, 3.0187531e2, 2.9826392e2, 3.0179602e2, 2.9797226e2, 2.9981561e2,
    3.0139807e2, 3.0026102e2, 3.0065182e2, 3.0243387e2, 2.9918311e2, 3.0074722e2,
    2.9894547e2, 2.9980435e2, 3.0077734e2, 2.9691653e2,
];

#[test]
fn output_version_is_no_newer_than_the_next_minor_version() {
    let parse = |version: &str| {
        version
            .split('.')
            .map(|part| part.parse::<u64>().unwrap())
            .collect::<Vec<_>>()
    };
    let output = parse(seiza_stacking::OUTPUT_VERSION);
    assert_eq!(output.len(), 3, "{}", seiza_stacking::OUTPUT_VERSION);
    // A PR that changes output sets the crate's next minor version, and the
    // release that ships the change bumps the crate to it.
    let version = parse(seiza_stacking::VERSION);
    let next_minor = vec![version[0], version[1] + 1, 0];
    assert!(
        output <= next_minor,
        "{} is past {}'s next minor",
        seiza_stacking::OUTPUT_VERSION,
        seiza_stacking::VERSION
    );
}

// Reference coefficients and signed samples were recorded from the planted
// linear model before implementing the continuum operations.
#[test]
fn continuum_linear_modes_keep_their_output() {
    use seiza_stacking::{
        ContinuumFitOutcome, ContinuumOrigin, ContinuumPhotometry, ContinuumPlan, ContinuumSource,
        ContinuumSources, ContinuumStar, ContinuumTransfer, fit_continuum, subtract_continuum,
    };
    let source = |name: &str| ContinuumSource {
        name: name.into(),
        revision: "v1".into(),
        transfer: ContinuumTransfer::Linear,
        saturation: Some(65535.0),
        clipping_revision: None,
    };
    let mut sources = ContinuumSources {
        width: 256,
        height: 256,
        narrowband: source("ha"),
        references: vec![source("red"), source("green")],
    };
    let measurement = |flux: f64, i: usize| ContinuumPhotometry {
        x: 25.0 + (i % 7) as f64 * 30.0,
        y: 25.0 + (i / 7) as f64 * 20.0,
        flux,
        peak: 100.0 + flux / 20.0,
        fwhm: 3.0,
        snr: 100.0,
    };
    let stars: Vec<_> = (0..40)
        .map(|i| {
            let r = 1000.0 + ((i * 17) % 157) as f64 * 50.0;
            let g = 1000.0 + ((i * 28 + 83) % 157) as f64 * 50.0;
            ContinuumStar {
                narrowband: measurement(0.4 * r + 0.2 * g, i),
                references: vec![measurement(r, i), measurement(g, i)],
            }
        })
        .collect();
    let ContinuumFitOutcome::Fitted { plan } =
        fit_continuum(&stars, &sources, &Default::default()).unwrap()
    else {
        panic!("planted continuum fit was skipped")
    };
    check("CONTINUUM_COEFFICIENTS", &plan.coefficients, &[0.4, 0.2]);
    sources.width = 3;
    sources.height = 1;
    sources.references.truncate(1);
    let plan = ContinuumPlan {
        psf_matching: None,
        schema_version: 1,
        sources: sources.clone(),
        coefficients: vec![0.35],
        reference_flux_ratios: Vec::new(),
        origin: ContinuumOrigin::Manual,
        diagnostics: None,
    };
    let broad = LinearImage::new(3, 1, 1, vec![0.5, 0.3, 0.4]).unwrap();
    let narrow = LinearImage::new(3, 1, 1, vec![0.2, 0.1, 0.5]).unwrap();
    let output = subtract_continuum(&narrow, &[&broad], &sources, &plan).unwrap();
    let signed: Vec<_> = output.data.iter().map(|v| f64::from(*v)).collect();
    check(
        "CONTINUUM_SIGNED",
        &signed,
        &[
            0.02500000223517418,
            -0.005000002682209015,
            0.36000001430511475,
        ],
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
    seiza_stacking::balance_narrowband(&mut physical, &Default::default()).unwrap();
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
    seiza_stacking::screen_rgb_stars(&mut nebula, &original, &starless).unwrap();
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
    let snapshot = bayer_stack(StackOptions::default());
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

/// Lanczos-3 on a calibrated sky a few sigma above zero, the case #295
/// fixed: its clamp fired on noise there, by sub-pixel phase, and the
/// rotated frames of a real target stacked into cross-hatched bands. The
/// bright sky of the other cases never trips the clamp, so only this case
/// sees how it treats noise.
#[test]
fn lanczos_on_a_faint_rotated_sky_keeps_its_output() {
    let options = StackOptions {
        interpolation: Interpolation::Lanczos3,
        ..StackOptions::default()
    };
    let mut stacker = LiveStacker::from_linear(faint_field(0), options).unwrap();
    for frame in 1..SHIFTS.len() {
        admit(stacker.push_linear(faint_field(frame)).unwrap());
    }
    let snapshot = stacker.snapshot().unwrap();
    check(
        "LANCZOS_FAINT_SKY",
        &summary(&snapshot.image),
        LANCZOS_FAINT_SKY,
    );
}

#[test]
fn a_stack_without_rejection_keeps_its_output() {
    let options = StackOptions {
        rejection: RejectionMode::None,
        ..StackOptions::default()
    };
    check("NO_REJECTION", &mono_stack(options), NO_REJECTION);
}

#[test]
fn a_stack_without_normalization_keeps_its_output() {
    let options = StackOptions {
        normalization: NormalizationMode::None,
        ..StackOptions::default()
    };
    check("NO_NORMALIZATION", &mono_stack(options), NO_NORMALIZATION);
}

#[test]
fn a_locally_normalized_stack_keeps_its_output() {
    let options = StackOptions {
        normalization: NormalizationMode::Local { tile_size: 32 },
        ..StackOptions::default()
    };
    check(
        "LOCAL_NORMALIZATION",
        &mono_stack(options),
        LOCAL_NORMALIZATION,
    );
}

#[test]
fn an_affine_registered_stack_keeps_its_output() {
    let mut options = StackOptions::default();
    options.registration.model = RegistrationModel::Affine;
    check("AFFINE", &mono_stack(options), AFFINE);
}

#[test]
fn a_stack_with_the_cosmetic_filter_keeps_its_output() {
    let directory = tempfile::tempdir().unwrap();
    let paths = defective_lights(&directory);
    let options = StackOptions {
        cosmetic: Some(ImpulseFilterOptions::default()),
        ..StackOptions::default()
    };
    let snapshot = fits_stack(&paths, options);
    check("COSMETIC", &summary(&snapshot.image), COSMETIC);
}

#[test]
fn a_stack_repairing_learned_defects_keeps_its_output() {
    let directory = tempfile::tempdir().unwrap();
    let paths = defective_lights(&directory);
    let defects = detect_defects(
        &paths,
        &CalibrationMasters::default(),
        &DefectDetectionOptions::default(),
    )
    .unwrap();
    assert!(!defects.is_empty());
    let options = StackOptions {
        defects: Some(defects),
        ..StackOptions::default()
    };
    let snapshot = fits_stack(&paths, options);
    check("DEFECTS", &summary(&snapshot.image), DEFECTS);
}

#[test]
fn every_demosaic_and_bayer_drizzle_keep_their_output() {
    for (case, expected, options) in [
        (
            "BAYER_MHC",
            BAYER_MHC,
            StackOptions {
                demosaic: Demosaic::Mhc,
                ..StackOptions::default()
            },
        ),
        (
            "BAYER_BILINEAR",
            BAYER_BILINEAR,
            StackOptions {
                demosaic: Demosaic::Bilinear,
                ..StackOptions::default()
            },
        ),
        (
            "BAYER_DRIZZLE",
            BAYER_DRIZZLE,
            StackOptions {
                cfa_integration: CfaIntegration::BayerDrizzle,
                ..StackOptions::default()
            },
        ),
    ] {
        check(case, &summary(&bayer_stack(options).image), expected);
    }
}

/// The second pass hosts run over the frames a stack kept, and the drizzle
/// that rides on it.
#[test]
fn a_reintegration_and_its_drizzle_keep_their_output() {
    let directory = tempfile::tempdir().unwrap();
    let paths = SHIFTS
        .iter()
        .enumerate()
        .map(|(frame, &(dx, dy))| {
            let image = mono_field(dx, dy, 0.96 + 0.01 * frame as f32, frame);
            write_raw(&directory, "light", frame, image.data, "LIGHT")
        })
        .collect::<Vec<_>>();
    let mut stacker =
        LiveStacker::open_fits(&paths[0], None, None, None, None, StackOptions::default()).unwrap();
    stacker.retain_frames_for_reintegration(None).unwrap();
    for path in &paths[1..] {
        admit(stacker.push_fits(path).unwrap());
    }
    let options = BatchStackOptions::default();
    let reintegrated = stacker.reintegrate(&options, |_, _, _| {}).unwrap();
    check(
        "REINTEGRATED",
        &summary(&reintegrated.snapshot.image),
        REINTEGRATED,
    );
    let drizzle = DrizzleOptions {
        scale: 2,
        ..DrizzleOptions::default()
    };
    let (_, drizzled) = stacker
        .reintegrate_drizzled(&options, &drizzle, |_, _, _| {})
        .unwrap();
    check("DRIZZLED", &summary(&drizzled.image), DRIZZLED);
}

/// Frames a host registered and normalized itself, integrated with frame
/// weights by the three passes PSF Guard's final pass runs, a trail across
/// the first: loaded on every pass, and once into scratch files.
#[test]
fn registered_frames_a_host_integrates_keep_their_output() {
    let frames = (0..8)
        .map(|frame| {
            let mut image = mono_field(0.0, 0.0, 1.0, frame);
            if frame == 0 {
                for x in 0..WIDTH {
                    image.data[(HEIGHT / 2 + x / 12) * WIDTH + x] += 400.0;
                }
            }
            image
        })
        .collect::<Vec<_>>();
    let options = BatchStackOptions {
        frame_weights: Some(
            (0..frames.len())
                .map(|frame| vec![1.0 / (1.0 + 0.1 * frame as f32)])
                .collect(),
        ),
        ..BatchStackOptions::default()
    };
    let whole = seiza_stacking::integrate_registered_frames(frames.len(), &options, |_, index| {
        Ok(frames[index].clone())
    })
    .unwrap();
    check(
        "HOST_INTEGRATED",
        &summary(&whole.snapshot.image),
        HOST_INTEGRATED,
    );
    let scratch = tempfile::tempdir().unwrap();
    let options = BatchStackOptions {
        scratch_directory: Some(scratch.path().to_path_buf()),
        ..options
    };
    let banded = seiza_stacking::integrate_registered_frames_banded(
        frames.len(),
        &options,
        |index| Ok(frames[index].clone()),
        |_, _, _| {},
    )
    .unwrap();
    check(
        "HOST_INTEGRATED",
        &summary(&banded.snapshot.image),
        HOST_INTEGRATED,
    );
}

/// Four registered panels with independent skies, shared extended emission,
/// and two observations per panel, through the host's final integration path.
#[test]
fn registered_mosaic_backgrounds_keep_their_output() {
    let (width, height) = (256, 240);
    let frames = (0..8)
        .map(|index| {
            let panel = index / 2;
            let (left, top) = ((panel % 2) * 96, (panel / 2) * 88);
            let data = (0..width * height)
                .map(|pixel| {
                    let (x, y) = (pixel % width, pixel / width);
                    if x < left || x >= left + 160 || y < top || y >= top + 152 {
                        return f32::NAN;
                    }
                    let (xx, yy) = (x as f32 / width as f32, y as f32 / height as f32);
                    let nebula =
                        110.0 * (-((xx - 0.43).powi(2) + (yy - 0.55).powi(2)) / 0.025).exp();
                    let sky = [0.0, 17.0, -12.0, 8.0][panel]
                        + [0.0, 22.0, -7.0, 13.0][panel] * xx
                        + [0.0, -15.0, 19.0, -9.0][panel] * yy;
                    100.0 + nebula + sky + noise(pixel, index, 0.4)
                })
                .collect();
            LinearImage::new(width, height, 1, data).unwrap()
        })
        .collect::<Vec<_>>();
    let options = BatchStackOptions {
        mosaic_panels: Some((0..8).map(|i| i / 2).collect()),
        ..Default::default()
    };
    let result = seiza_stacking::integrate_registered_frames(frames.len(), &options, |_, index| {
        Ok(frames[index].clone())
    })
    .unwrap();
    check(
        "REGISTERED_MOSAIC_BACKGROUNDS",
        &summary(&result.snapshot.image),
        REGISTERED_MOSAIC_BACKGROUNDS,
    );
}

const REGISTERED_MOSAIC_BACKGROUNDS: &[f64] = &[
    1.1776097e2,
    1.1963589e2,
    2.1656921e2,
    1.0051737e2,
    1.0758487e2,
    1.1445993e2,
    1.2108497e2,
    1.2822806e2,
    9.8188171e1,
    1.0771823e2,
    1.1705459e2,
    1.1880308e2,
    1.2547261e2,
    9.6052299e1,
    1.4362491e2,
    1.9097940e2,
    1.1951297e2,
    1.2271482e2,
    9.3255600e1,
    1.1458409e2,
    1.3630460e2,
    1.1491711e2,
    1.2002249e2,
    9.0404106e1,
    9.7089279e1,
    1.0401866e2,
    1.1092870e2,
    1.1739417e2,
];

/// Extended emission fills the overlap: only a relative constant is
/// measurable safely when there are too few dim background cells.
#[test]
fn a_signal_dominated_registered_mosaic_keeps_its_output() {
    let (width, height) = (160, 96);
    let images = (0..4)
        .map(|frame| {
            let panel = frame / 2;
            let data = (0..width * height)
                .map(|pixel| {
                    let (x, y) = (pixel % width, pixel / width);
                    if x < panel * 48 || x >= panel * 48 + 112 {
                        return f32::NAN;
                    }
                    let nebula = 100.0
                        * (-((x as f32 - 78.0).powi(2) + (y as f32 - 48.0).powi(2)) / 180.0).exp();
                    20.0 + nebula + panel as f32 * 18.0
                })
                .collect();
            LinearImage::new(width, height, 1, data).unwrap()
        })
        .collect::<Vec<_>>();
    let options = BatchStackOptions {
        mosaic_panels: Some(vec![0, 0, 1, 1]),
        ..Default::default()
    };
    let result = seiza_stacking::integrate_registered_frames(4, &options, |_, index| {
        Ok(images[index].clone())
    })
    .unwrap();
    check(
        "REGISTERED_SIGNAL_DOMINATED_MOSAIC",
        &summary(&result.snapshot.image),
        REGISTERED_SIGNAL_DOMINATED_MOSAIC,
    );
}
const REGISTERED_SIGNAL_DOMINATED_MOSAIC: &[f64] = &[
    3.2681552e1,
    3.5193859e1,
    1.2900000e2,
    2.8999996e1,
    2.9002277e1,
    2.9330900e1,
    2.9000553e1,
    2.9000004e1,
    2.8999996e1,
    2.9162500e1,
    5.2587708e1,
    2.9039196e1,
    2.9000004e1,
    2.8999996e1,
    2.9673790e1,
    1.2680228e2,
    2.9162510e1,
    2.9000004e1,
    2.8999996e1,
    2.9162502e1,
    5.2587708e1,
    2.9039196e1,
    2.9000004e1,
    2.8999996e1,
    2.9002277e1,
    2.9330900e1,
    2.9000553e1,
    2.9000004e1,
];

/// Paired samples: stellar structure under a steep sky,
/// a partial-cell overlap, and independently missing RGB samples.
#[test]
fn registered_mosaic_paired_samples_keep_their_output() {
    let mut fingerprint = Vec::new();
    for mode in 0..3 {
        let (width, height, channels, span, shift) = match mode {
            0 => (256, 192, 1, 160, 96),
            1 => (248, 128, 1, 128, 120),
            _ => (160, 96, 3, 112, 48),
        };
        let frames = (0..2)
            .map(|panel| {
                let data = (0..width * height * channels)
                    .map(|sample| {
                        let (pixel, c) = (sample / channels, sample % channels);
                        let (x, y) = (pixel % width, pixel / width);
                        if x < panel * shift
                            || x >= panel * shift + span
                            || (mode == 2 && ((c == 0 && x % 2 == 1) || (c == 1 && x % 2 == 0)))
                        {
                            return f32::NAN;
                        }
                        if mode == 0 {
                            100.0
                                + panel as f32 * 1000.0 * x as f32 / 256.0
                                + 1000.0
                                    * (-((x as f32 - 103.0).powi(2) + (y as f32 - 82.0).powi(2))
                                        / 8.0)
                                        .exp()
                        } else {
                            100.0 + panel as f32 * 18.0
                        }
                    })
                    .collect();
                LinearImage::new(width, height, channels, data).unwrap()
            })
            .collect::<Vec<_>>();
        let options = BatchStackOptions {
            mosaic_panels: Some(vec![0, 1]),
            gpu: GpuOptions {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let result =
            seiza_stacking::integrate_registered_frames(2, &options, |_, i| Ok(frames[i].clone()))
                .unwrap();
        // Summaries operate on finite scalar samples; preserve each channel's
        // support separately rather than hiding NaNs by replacing them.
        for c in 0..channels {
            let finite = result
                .snapshot
                .image
                .data
                .iter()
                .skip(c)
                .step_by(channels)
                .copied()
                .filter(|v| v.is_finite())
                .collect::<Vec<_>>();
            fingerprint.extend([
                finite.iter().map(|&v| f64::from(v)).sum::<f64>() / finite.len() as f64,
                (finite.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / finite.len() as f64)
                    .sqrt(),
                f64::from(finite.iter().copied().fold(f32::NEG_INFINITY, f32::max)),
                f64::from(finite.iter().copied().fold(f32::INFINITY, f32::min)),
            ]);
            fingerprint.extend((0..16).map(|i| f64::from(finite[i * (finite.len() - 1) / 15])));
        }
    }
    check(
        "REGISTERED_MOSAIC_PAIRED_SAMPLES",
        &fingerprint,
        REGISTERED_MOSAIC_PAIRED_SAMPLES,
    );
}
const REGISTERED_MOSAIC_PAIRED_SAMPLES: &[f64] = &[
    4.4337287e2,
    5.6152321e2,
    1.3284091e3,
    -1.5085341e2,
    -1.5084932e2,
    7.9943890e2,
    5.6141522e2,
    3.2374130e2,
    8.1807182e1,
    1.0373828e3,
    7.9944031e2,
    5.6146613e2,
    3.1909070e2,
    8.1805809e1,
    1.0373842e3,
    7.9944165e2,
    5.5678729e2,
    3.1906598e2,
    8.1804436e1,
    1.0373856e3,
    1.0900000e2,
    1.0900000e2,
    1.0900001e2,
    1.0899999e2,
    1.0899999e2,
    1.0900000e2,
    1.0899999e2,
    1.0900001e2,
    1.0899999e2,
    1.0900001e2,
    1.0899999e2,
    1.0900001e2,
    1.0899999e2,
    1.0900001e2,
    1.0899999e2,
    1.0900001e2,
    1.0899999e2,
    1.0900001e2,
    1.0900000e2,
    1.0900001e2,
    1.0893251e2,
    1.0911080e2,
    1.1959697e2,
    9.8268074e1,
    9.8268074e1,
    1.0663826e2,
    1.1527836e2,
    1.0231677e2,
    1.1095766e2,
    1.1959697e2,
    1.0663651e2,
    1.1527836e2,
    1.0231677e2,
    1.1095838e2,
    1.1959697e2,
    1.0663716e2,
    1.1527836e2,
    1.0231677e2,
    1.1095687e2,
    1.1959697e2,
    1.0906749e2,
    1.0924557e2,
    1.1973193e2,
    9.8403030e1,
    9.8403030e1,
    1.0677322e2,
    1.1541331e2,
    1.0245173e2,
    1.1109272e2,
    1.1973193e2,
    1.0677155e2,
    1.1541331e2,
    1.0245173e2,
    1.1109341e2,
    1.1973193e2,
    1.0677222e2,
    1.1541331e2,
    1.0245173e2,
    1.1109183e2,
    1.1973193e2,
    1.0900000e2,
    1.0917821e2,
    1.1973193e2,
    9.8268074e1,
    9.8268074e1,
    1.0677322e2,
    1.1541331e2,
    1.0245173e2,
    1.1109272e2,
    1.1973193e2,
    1.0677155e2,
    1.1541331e2,
    1.0245173e2,
    1.1109341e2,
    1.1973193e2,
    1.0677222e2,
    1.1541331e2,
    1.0245173e2,
    1.1109183e2,
    1.1973193e2,
];

/// A transient hit on a sampled position is rejected in the full-resolution
/// pass; its background estimate must not carry it into neighboring cells.
#[test]
fn registered_mosaic_sampled_outliers_keep_their_output() {
    let frames = (0..6)
        .map(|index| {
            let panel = index / 3;
            let data = (0..256 * 192)
                .map(|pixel| {
                    let (x, y) = (pixel % 256, pixel / 256);
                    if x < panel * 96 || x >= panel * 96 + 160 {
                        return f32::NAN;
                    }
                    100.0
                        + panel as f32 * 18.0
                        + if index == 0 && (x, y) == (97, 85) {
                            3000.0
                        } else {
                            0.0
                        }
                })
                .collect();
            LinearImage::new(256, 192, 1, data).unwrap()
        })
        .collect::<Vec<_>>();
    let options = BatchStackOptions {
        mosaic_panels: Some(vec![0, 0, 0, 1, 1, 1]),
        gpu: GpuOptions {
            enabled: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let result =
        seiza_stacking::integrate_registered_frames(6, &options, |_, i| Ok(frames[i].clone()))
            .unwrap();
    check(
        "REGISTERED_MOSAIC_SAMPLED_OUTLIERS",
        &summary(&result.snapshot.image),
        REGISTERED_MOSAIC_SAMPLED_OUTLIERS,
    );
}
const REGISTERED_MOSAIC_SAMPLED_OUTLIERS: &[f64] = &[
    1.0899995e2,
    1.0923943e2,
    1.2147171e2,
    9.8092636e1,
    1.0356977e2,
    1.0904907e2,
    1.1452798e2,
    1.2000472e2,
    9.8092636e1,
    1.0356978e2,
    1.0904911e2,
    1.1452797e2,
    1.2000471e2,
    9.8092644e1,
    1.0356978e2,
    1.0904911e2,
    1.1452797e2,
    1.2000471e2,
    9.8092644e1,
    1.0356979e2,
    1.0904911e2,
    1.1452796e2,
    1.2000471e2,
    9.8092651e1,
    1.0356979e2,
    1.0904907e2,
    1.1452795e2,
    1.2000471e2,
];

/// Frames large enough that, with the `gpu` feature and a GPU present,
/// they resample there, and a reintegration whose bands integrate there:
/// stacked on the CPU, which recorded the summaries, and where the GPU may,
/// whose single-precision work must stay within this file's tolerance of
/// the CPU's.
#[test]
fn a_stack_large_enough_for_the_gpu_keeps_its_output() {
    large_stack(
        false,
        true,
        NormalizationMode::Global,
        LARGE_LIVE,
        LARGE_REINTEGRATED,
    );
}

/// Preparation runs ahead of admission, including the first batch before
/// the lazy accumulator would previously have been offered to the GPU.
#[test]
fn a_pipelined_stack_large_enough_for_the_gpu_keeps_its_output() {
    large_stack(
        true,
        true,
        NormalizationMode::Global,
        LARGE_LIVE,
        LARGE_REINTEGRATED,
    );
}

#[test]
fn a_locally_normalized_pipelined_stack_large_enough_for_the_gpu_keeps_its_output() {
    large_stack(
        true,
        true,
        NormalizationMode::LocalBackground { tile_size: 256 },
        LARGE_LOCAL_LIVE,
        LARGE_LOCAL_REINTEGRATED,
    );
}

#[test]
fn a_live_only_pipelined_stack_large_enough_for_the_gpu_keeps_its_output() {
    large_stack(
        true,
        false,
        NormalizationMode::Global,
        LARGE_LIVE,
        LARGE_REINTEGRATED,
    );
}

#[test]
fn a_live_only_locally_normalized_pipelined_stack_large_enough_for_the_gpu_keeps_its_output() {
    large_stack(
        true,
        false,
        NormalizationMode::LocalBackground { tile_size: 256 },
        LARGE_LOCAL_LIVE,
        LARGE_LOCAL_REINTEGRATED,
    );
}

// First recorded with lazy accumulator startup, before preparation began
// with resident moments. Local background fits exercise a grid of offsets.
#[rustfmt::skip]
const LARGE_LOCAL_LIVE: &[f64] = &[
    1.3754521e2, 4.2678341e2, 1.7932547e4, 9.7603149e1, 1.0072701e2, 1.0332149e2, 1.0637009e2,
    1.0963932e2, 1.0196701e2, 1.0466370e2, 1.0884956e2, 1.1161555e2, 1.1412208e2, 1.0603079e2,
    1.0965089e2, 1.1204286e2, 1.1683111e2, 1.1863527e2, 1.1065800e2, 1.1457457e2, 1.1625290e2,
    1.1974577e2, 1.2310165e2, 1.1479590e2, 1.1817994e2, 1.2165116e2, 1.2404044e2, 1.2759380e2,
];
#[rustfmt::skip]
const LARGE_LOCAL_REINTEGRATED: &[f64] = &[
    1.3728293e2, 4.2654226e2, 1.7932539e4, 9.7631035e1, 1.0074255e2, 1.0330618e2, 1.0636744e2,
    1.0964255e2, 1.0198824e2, 1.0467348e2, 1.0883335e2, 1.1135132e2, 1.1413447e2, 1.0604317e2,
    1.0965260e2, 1.1202531e2, 1.1683836e2, 1.1865505e2, 1.1066511e2, 1.1457175e2, 1.1623668e2,
    1.1975120e2, 1.2311982e2, 1.1527016e2, 1.1817233e2, 1.2163953e2, 1.2405253e2, 1.2762001e2,
];

fn large_stack(
    pipelined: bool,
    retain: bool,
    normalization: NormalizationMode,
    live_expected: &[f64],
    reintegrated_expected: &[f64],
) {
    let stars = large_stars();
    let directory = tempfile::tempdir().unwrap();
    let paths = SHIFTS
        .iter()
        .enumerate()
        .map(|(frame, &(dx, dy))| {
            let gain = 0.97 + 0.01 * frame as f32;
            let mut data: Vec<f32> = (0..LARGE_WIDTH * LARGE_HEIGHT)
                .map(|index| {
                    let (x, y) = ((index % LARGE_WIDTH) as f32, (index / LARGE_WIDTH) as f32);
                    let sky = 100.0 + 0.01 * x + 0.02 * y;
                    let light = large_star_light(&stars, (x, y), (dx, dy));
                    gain * (sky + light) + noise(index, frame, 3.0)
                })
                .collect();
            // A satellite trail in the first frame, which only the
            // reintegration can take out.
            if frame == 0 {
                for x in 0..LARGE_WIDTH {
                    data[(100 + x / 2) * LARGE_WIDTH + x] += 2_000.0;
                }
            }
            let path = directory.path().join(format!("large-{frame}.fits"));
            let image = LinearImage::new(LARGE_WIDTH, LARGE_HEIGHT, 1, data).unwrap();
            let cards = [
                WriteHeaderCard::new("IMAGETYP", HeaderValue::String("LIGHT".into())),
                WriteHeaderCard::new("EXPTIME", HeaderValue::Float(60.0)),
            ];
            write_processed_image_fits_f32(&path, &image, &[], &cards).unwrap();
            path
        })
        .collect::<Vec<_>>();
    for (on, gpu) in [
        ("CPU", GpuOptions::cpu()),
        ("GPU", GpuOptions::default()),
        (
            "traced GPU",
            GpuOptions {
                trace: true,
                ..GpuOptions::default()
            },
        ),
    ] {
        eprintln!("on the {on}");
        let options = StackOptions {
            gpu,
            normalization,
            ..StackOptions::default()
        };
        let mut stacker =
            LiveStacker::open_fits(&paths[0], None, None, None, None, options.clone()).unwrap();
        if retain {
            stacker.retain_frames_for_reintegration(None).unwrap();
        }
        if pipelined {
            let report = stacker
                .push_fits_pipelined(
                    &paths[1..],
                    &seiza_stacking::PipelineOptions {
                        workers: Some(2),
                        ..Default::default()
                    },
                    |_, result| {
                        admit(result.unwrap());
                        Continue::Yes
                    },
                )
                .unwrap();
            assert_eq!(report.integrated, paths.len() - 1);
        } else {
            for path in &paths[1..] {
                admit(stacker.push_fits(path).unwrap());
            }
        }
        check(
            if matches!(normalization, NormalizationMode::LocalBackground { .. }) {
                "LARGE_LOCAL_LIVE"
            } else {
                "LARGE_LIVE"
            },
            &summary(&stacker.snapshot().unwrap().image),
            live_expected,
        );
        // Exercise both one band and repeated buffer reuse, including a
        // shorter final band. Each full band still reaches the GPU's
        // minimum work size. Changing band boundaries changes no arithmetic.
        if !retain {
            continue;
        }
        for band_memory_bytes in [
            BatchStackOptions::default().band_memory_bytes,
            2 * 173 * LARGE_WIDTH * paths.len() * std::mem::size_of::<f32>(),
        ] {
            let reintegrated = stacker
                .reintegrate(
                    &BatchStackOptions {
                        band_memory_bytes,
                        gpu: options.gpu,
                        ..BatchStackOptions::default()
                    },
                    |_, _, _| {},
                )
                .unwrap();
            check(
                if matches!(normalization, NormalizationMode::LocalBackground { .. }) {
                    "LARGE_LOCAL_REINTEGRATED"
                } else {
                    "LARGE_REINTEGRATED"
                },
                &summary(&reintegrated.snapshot.image),
                reintegrated_expected,
            );
        }
    }
}

/// RGGB lights as large as [`a_stack_large_enough_for_the_gpu_keeps_its_output`]'s,
/// which with a GPU present demosaic there (VNG) as well as resample: a
/// colour stack, on the CPU, which recorded the summary, and where the GPU
/// may, within this file's tolerance of the CPU's.
#[test]
fn a_debayered_stack_large_enough_for_the_gpu_keeps_its_output() {
    const SKY: [f32; 3] = [400.0, 300.0, 200.0];
    const COLOR: [f32; 3] = [1.0, 0.8, 0.6];
    let stars = large_stars();
    let directory = tempfile::tempdir().unwrap();
    let bayer = [("BAYERPAT".to_string(), HeaderValue::String("RGGB".into()))];
    let paths = SHIFTS[..6]
        .iter()
        .enumerate()
        .map(|(frame, &(dx, dy))| {
            let data: Vec<f32> = (0..LARGE_WIDTH * LARGE_HEIGHT)
                .map(|index| {
                    let (x, y) = (index % LARGE_WIDTH, index / LARGE_WIDTH);
                    let channel = [[0, 1], [1, 2]][y & 1][x & 1];
                    let light = large_star_light(&stars, (x as f32, y as f32), (dx, dy));
                    SKY[channel] + COLOR[channel] * light + noise(index, frame, 4.0)
                })
                .collect();
            let image = LinearImage::new(LARGE_WIDTH, LARGE_HEIGHT, 1, data).unwrap();
            let path = directory.path().join(format!("large-bayer-{frame}.fits"));
            write_processed_image_fits_f32(&path, &image, &bayer, &[]).unwrap();
            path
        })
        .collect::<Vec<_>>();
    for (on, gpu) in [("CPU", GpuOptions::cpu()), ("GPU", GpuOptions::default())] {
        eprintln!("on the {on}");
        let options = StackOptions {
            gpu,
            ..StackOptions::default()
        };
        let snapshot = fits_stack(&paths, options);
        check("LARGE_BAYER", &summary(&snapshot.image), LARGE_BAYER);
    }
}

const LARGE_WIDTH: usize = 1280;
const LARGE_HEIGHT: usize = 960;

/// Full-size raw lights carry clipping through GPU resampling as well as
/// the stack's mean. Isolated saturated pixels give the mask actual work.
#[test]
fn a_large_stack_keeps_its_clipped_fraction() {
    let stars = large_stars();
    let frames = SHIFTS[..4]
        .iter()
        .enumerate()
        .map(|(frame, &(dx, dy))| {
            let mut data = (0..LARGE_WIDTH * LARGE_HEIGHT)
                .map(|index| {
                    let point = ((index % LARGE_WIDTH) as f32, (index / LARGE_WIDTH) as f32);
                    100.0 + large_star_light(&stars, point, (dx, dy)) + noise(index, frame, 3.0)
                })
                .collect::<Vec<_>>();
            if frame > 0 {
                for y in 410..416 {
                    for x in 610..616 {
                        data[y * LARGE_WIDTH + x] = 65535.0;
                    }
                }
            }
            seiza_stacking::FitsFrame {
                image: LinearImage::new(LARGE_WIDTH, LARGE_HEIGHT, 1, data).unwrap(),
                headers: vec![
                    ("BITPIX".into(), HeaderValue::Integer(16)),
                    ("BZERO".into(), HeaderValue::Float(32768.0)),
                ],
                exposure_seconds: None,
                bayer: None,
                source: None,
                bounds: None,
            }
        })
        .collect::<Vec<_>>();
    for gpu in [GpuOptions::cpu(), GpuOptions::default()] {
        let mut stacker = LiveStacker::new(
            frames[0].clone(),
            CalibrationMasters::default(),
            StackOptions {
                gpu,
                normalization: NormalizationMode::None,
                rejection: RejectionMode::None,
                ..StackOptions::default()
            },
        )
        .unwrap();
        assert!(stacker.clipped_fraction().is_none());
        for frame in &frames[1..] {
            admit(stacker.push(frame.clone()).unwrap());
        }
        let clipped = stacker.clipped_fraction().unwrap();
        const LARGE_CLIPPED: &[f64] = &[
            2.1972651e-5,
            2.4751522e-3,
            5.0000000e-1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        check("LARGE_CLIPPED", &summary(&clipped), LARGE_CLIPPED);
    }
}

/// Two hundred stars over a large frame, from a small integer generator.
fn large_stars() -> Vec<(f32, f32, f32)> {
    let mut state = 0x6a09_e667_u32;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1 << 24) as f32
    };
    (0..200)
        .map(|_| {
            (
                8.0 + next() * (LARGE_WIDTH as f32 - 16.0),
                8.0 + next() * (LARGE_HEIGHT as f32 - 16.0),
                400.0 + next() * 20_000.0,
            )
        })
        .collect()
}

/// The light of [`large_stars`] at `(x, y)` of a frame shifted by
/// `(dx, dy)`.
fn large_star_light(stars: &[(f32, f32, f32)], (x, y): (f32, f32), (dx, dy): (f32, f32)) -> f32 {
    stars
        .iter()
        .filter(|star| (star.0 - x + dx).abs() < 8.0 && (star.1 - y + dy).abs() < 8.0)
        .map(|&(star_x, star_y, flux)| {
            let (sx, sy) = (x - dx - star_x, y - dy - star_y);
            flux * (-(sx * sx + sy * sy) / 4.5).exp()
        })
        .sum()
}

/// Lights binned by 2 after calibration and before registration, and the
/// same lights reintegrated at the sensor's full resolution from the binned
/// stack, each frame's transform refined from its binned one.
#[test]
fn a_binned_stack_and_its_full_resolution_reintegration_keep_their_output() {
    let directory = tempfile::tempdir().unwrap();
    let paths = SHIFTS
        .iter()
        .enumerate()
        .map(|(frame, &(dx, dy))| {
            let image = mono_field(dx, dy, 0.96 + 0.01 * frame as f32, frame);
            write_raw(&directory, "light", frame, image.data, "LIGHT")
        })
        .collect::<Vec<_>>();
    let options = StackOptions {
        bin_factor: 2,
        ..StackOptions::default()
    };
    let mut stacker = LiveStacker::open_fits(&paths[0], None, None, None, None, options).unwrap();
    for path in &paths[1..] {
        admit(stacker.push_fits(path).unwrap());
    }
    check(
        "BINNED",
        &summary(&stacker.snapshot().unwrap().image),
        BINNED,
    );
    let full = stacker
        .reintegrate_full_resolution(&BatchStackOptions::default(), true, |_, _, _| {})
        .unwrap();
    assert_eq!(full.refined_frames, paths.len() - 1);
    check(
        "FULL_RESOLUTION",
        &summary(&full.result.snapshot.image),
        FULL_RESOLUTION,
    );
}

/// Raw RGGB lights made into one pixel per 2x2 cell, their colours resampled
/// from their photosites' places, and binned no further.
#[test]
fn a_superpixel_stack_keeps_its_output() {
    let options = StackOptions {
        demosaic: Demosaic::Superpixel,
        bin_factor: 2,
        ..StackOptions::default()
    };
    check(
        "SUPERPIXEL",
        &summary(&bayer_stack(options).image),
        SUPERPIXEL,
    );
}

/// Lights binned by 2, drizzled from their sensor pixels back onto the
/// sensor's grid.
#[test]
fn a_binned_stack_s_drizzle_keeps_its_output() {
    let directory = tempfile::tempdir().unwrap();
    let paths = SHIFTS
        .iter()
        .enumerate()
        .map(|(frame, &(dx, dy))| {
            let image = mono_field(dx, dy, 0.96 + 0.01 * frame as f32, frame);
            write_raw(&directory, "light", frame, image.data, "LIGHT")
        })
        .collect::<Vec<_>>();
    let options = StackOptions {
        bin_factor: 2,
        ..StackOptions::default()
    };
    let mut stacker = LiveStacker::open_fits(&paths[0], None, None, None, None, options).unwrap();
    for path in &paths[1..] {
        admit(stacker.push_fits(path).unwrap());
    }
    let drizzle = DrizzleOptions {
        scale: 2,
        ..DrizzleOptions::default()
    };
    let (_, drizzled) = stacker
        .reintegrate_drizzled(&BatchStackOptions::default(), &drizzle, |_, _, _| {})
        .unwrap();
    check(
        "BINNED_DRIZZLED",
        &summary(&drizzled.image),
        BINNED_DRIZZLED,
    );
}

/// A mosaic of lights binned by 2, reintegrated at the sensor's full
/// resolution, each frame refit against the canvas the others build there.
#[test]
fn a_binned_mosaic_s_full_resolution_reintegration_keeps_its_output() {
    let directory = tempfile::tempdir().unwrap();
    let paths = mosaic_panels(&directory);
    let mut options = StackOptions {
        bin_factor: 2,
        ..StackOptions::default()
    };
    options.acceptance.minimum_overlap_fraction = 0.1;
    let mut stacker = LiveStacker::with_expanding_reference(
        &paths,
        CalibrationMasters::default(),
        options,
        usize::MAX,
        |_, _, _, _| Continue::Yes,
    )
    .unwrap();
    for path in &paths {
        admit(stacker.push_fits(path).unwrap());
    }
    let full = stacker
        .reintegrate_full_resolution(&BatchStackOptions::default(), true, |_, _, _| {})
        .unwrap();
    assert!(full.refined_frames > 0);
    check(
        "MOSAIC_FULL_RESOLUTION",
        &summary(&full.result.snapshot.image),
        MOSAIC_FULL_RESOLUTION,
    );
}

/// Two panels sharing 8% of their footprints, as panels planned for 10%
/// overlap land, which join at the expanded canvas's suggested minimum
/// overlap and not at 0.10: a change in how planning measures a panel's
/// overlap, or admits a panel by it, moves this canvas.
#[test]
fn a_mosaic_overlapping_just_above_the_expanded_minimum_keeps_its_output() {
    let directory = tempfile::tempdir().unwrap();
    let paths = narrow_mosaic_panels(&directory);
    let mut options = StackOptions::default();
    options.acceptance.minimum_overlap_fraction =
        FrameAcceptanceCriteria::EXPANDED_MINIMUM_OVERLAP_FRACTION;
    let mut stacker = LiveStacker::with_expanding_reference(
        &paths,
        CalibrationMasters::default(),
        options,
        usize::MAX,
        |_, _, _, _| Continue::Yes,
    )
    .unwrap();
    for path in &paths {
        admit(stacker.push_fits(path).unwrap());
    }
    let snapshot = stacker.snapshot().unwrap();
    assert!(snapshot.image.height > 2 * NARROW_MOSAIC_STEP);
    check("NARROW_MOSAIC", &summary(&snapshot.image), NARROW_MOSAIC);
}

/// Two overlapping panels on a canvas that grows to hold them both, with
/// frames related by a similarity and by a quadratic warp.
#[test]
fn a_mosaic_keeps_its_output() {
    for (case, expected, model) in [
        ("MOSAIC", MOSAIC, RegistrationModel::Similarity),
        (
            "MOSAIC_QUADRATIC",
            MOSAIC_QUADRATIC,
            RegistrationModel::Quadratic,
        ),
    ] {
        check(case, &summary(&mosaic_stack(model).image), expected);
    }
}

/// The same mosaic with its frames prepared several at once, as many as a
/// small budget affords: how many fit sets only how the work overlaps, never
/// the canvas it builds.
#[test]
fn a_mosaic_prepared_several_frames_at_once_keeps_its_output() {
    let directory = tempfile::tempdir().unwrap();
    let paths = mosaic_panels(&directory);
    let mut options = StackOptions::default();
    options.acceptance.minimum_overlap_fraction = 0.1;
    let mut stacker = LiveStacker::with_expanding_reference(
        &paths,
        CalibrationMasters::default(),
        options,
        usize::MAX,
        |_, _, _, _| Continue::Yes,
    )
    .unwrap();
    let report = stacker
        .push_fits_pipelined(
            &paths,
            &seiza_stacking::PipelineOptions::with_budget(1_200_000),
            |_, result| {
                admit(result.unwrap());
                Continue::Yes
            },
        )
        .unwrap();
    assert_eq!(report.integrated, paths.len());
    check(
        "MOSAIC",
        &summary(&stacker.snapshot().unwrap().image),
        MOSAIC,
    );
}

/// A calibrated lens changes the canvas geometry even with the default
/// similarity model. Exposures following each panel's first frame must keep
/// that nonlinear projection when refining their dithered star positions.
#[test]
fn a_lens_mosaic_with_sibling_exposures_keeps_its_output() {
    let directory = tempfile::tempdir().unwrap();
    let paths = projected_lens_panels(&directory);
    let mut options = StackOptions {
        normalization: NormalizationMode::None,
        ..StackOptions::default()
    };
    options.acceptance.minimum_overlap_fraction = 0.1;
    // The far panel's tangent-plane projection grows by a little over 4%.
    options.acceptance.maximum_scale_deviation = 0.1;
    let mut stacker = LiveStacker::with_expanding_reference(
        &paths,
        CalibrationMasters::default(),
        options,
        usize::MAX,
        |_, _, _, _| Continue::Yes,
    )
    .unwrap();
    let lens = stacker
        .expanded_lens()
        .expect("the panels constrain the lens");
    assert!(lens.focal_length_pixels.is_some());
    // A saved canvas carries no per-file planned registrations. Its
    // continuation must recover the same geometry from the panel metadata.
    // Contexts require one admitted frame, so save just after the anchor.
    let context_path = directory.path().join("lens-mosaic.seiza-stack");
    let mut planned = Vec::new();
    for (index, path) in paths.iter().enumerate() {
        let FrameDisposition::Accepted(diagnostics) = stacker.push_fits(path).unwrap() else {
            panic!("lens mosaic frame {index} was not accepted");
        };
        let order = diagnostics.mapping.warp().map(|warp| warp.order());
        if index % 2 == 1 {
            assert!(diagnostics.registration_rms_pixels < 0.3, "{diagnostics:?}");
        }
        if index >= 2 {
            assert!(order.is_some_and(|order| order >= 2), "{diagnostics:?}");
        }
        planned.push(diagnostics);
        if index == 0 {
            stacker.save_context(&context_path).unwrap();
        }
    }
    let mut continued = LiveStacker::open_context(&context_path).unwrap();
    let mut snapshot = stacker.snapshot().unwrap();
    let mut maximum_mapping_difference = 0.0_f64;
    for (index, (path, expected)) in paths.iter().zip(&planned).enumerate().skip(1) {
        let FrameDisposition::Accepted(actual) = continued.push_fits(path).unwrap() else {
            panic!("continued lens mosaic frame {index} was not accepted");
        };
        let actual_warp = actual.mapping.warp().expect("the lens needs a warp");
        assert!(actual.registration_rms_pixels < 0.3, "{actual:?}");
        if index >= 2 {
            assert!(actual_warp.order() >= 2, "{actual:?}");
        }
        let expected_warp = expected.mapping.warp().unwrap();
        for y in [50.0, 200.0, 350.0] {
            for x in [50.0, 300.0, 550.0] {
                let (cx, cy) = expected.transform.apply(x, y);
                let (ex, ey) = expected_warp.apply(cx, cy);
                let (ax, ay) = actual_warp.apply(cx, cy);
                maximum_mapping_difference =
                    maximum_mapping_difference.max((ex - ax).hypot(ey - ay));
            }
        }
    }
    let restored = continued.snapshot().unwrap();
    assert_eq!(restored.accepted_frames, snapshot.accepted_frames);
    assert_eq!(
        (restored.image.width, restored.image.height),
        (snapshot.image.width, snapshot.image.height)
    );
    let (mut difference_power, mut signal_power) = (0.0_f64, 0.0_f64);
    let (mut planned_coverage, mut common_coverage) = (0, 0);
    for (&expected, &actual) in snapshot.image.data.iter().zip(&restored.image.data) {
        planned_coverage += usize::from(expected.is_finite());
        if expected.is_finite() && actual.is_finite() {
            common_coverage += 1;
            difference_power += f64::from(expected - actual).powi(2);
            signal_power += (f64::from(expected) - 100.0).powi(2);
        }
    }
    let relative_signal_rms = (difference_power / signal_power).sqrt();
    // Continuation refits later panels' first exposures too, whereas
    // planning places those by the camera poses. Require a tenth-pixel
    // agreement and less than 1% RMS difference in the stars above sky.
    assert!(
        maximum_mapping_difference < 0.1,
        "{maximum_mapping_difference}"
    );
    assert!(relative_signal_rms < 0.01, "{relative_signal_rms}");
    assert!(common_coverage as f64 >= planned_coverage as f64 * 0.995);
    // The projected footprint is not rectangular; include its uncovered
    // pixels as zero so their extent also participates in the fingerprint.
    for sample in &mut snapshot.image.data {
        if !sample.is_finite() {
            *sample = 0.0;
        }
    }
    let mut fingerprint = summary(&snapshot.image);
    fingerprint.extend([
        f64::from(snapshot.accepted_frames),
        snapshot.image.width as f64,
        snapshot.image.height as f64,
    ]);
    check("LENS_MOSAIC_SIBLINGS", &fingerprint, LENS_MOSAIC_SIBLINGS);
}

/// A globally ranked background reference must not leave a dimmer mosaic
/// panel without local samples. Its cloudier final exposure has a different
/// gradient from the twenty better exposures covering that same sky.
#[test]
fn a_mosaic_background_reference_keeps_its_output() {
    let directory = tempfile::tempdir().unwrap();
    let paths = uneven_mosaic_panels(&directory);
    let mut options = StackOptions {
        normalization: NormalizationMode::LocalBackground { tile_size: 32 },
        rejection: RejectionMode::None,
        ..StackOptions::default()
    };
    options.acceptance.minimum_overlap_fraction = 0.1;
    let mut stacker = LiveStacker::with_expanding_reference(
        &paths,
        CalibrationMasters::default(),
        options,
        usize::MAX,
        |_, _, _, _| Continue::Yes,
    )
    .unwrap();
    stacker
        .retain_frames_for_reintegration(Some(directory.path()))
        .unwrap();
    for (index, path) in paths.iter().enumerate() {
        let FrameDisposition::Accepted(diagnostics) = stacker.push_fits(path).unwrap() else {
            panic!("background mosaic frame {index} was not accepted");
        };
        if index < 21 {
            assert!(diagnostics.normalization_mean_gain < 1.2, "{diagnostics:?}");
        } else {
            assert!(diagnostics.normalization_mean_gain > 1.6, "{diagnostics:?}");
        }
    }
    let options = BatchStackOptions {
        rejection: seiza_stacking::MasterRejectionOptions {
            low_sigma: 100.0,
            high_sigma: 100.0,
        },
        ..BatchStackOptions::default()
    };
    let result = stacker.reintegrate(&options, |_, _, _| {}).unwrap();
    assert_eq!(result.snapshot.accepted_frames, 43);
    check(
        "MOSAIC_BACKGROUND_REFERENCE",
        &summary(&result.snapshot.image),
        MOSAIC_BACKGROUND_REFERENCE,
    );
}

/// Two interleaved mosaic panels whose every exposure has its own sky
/// gradient, as light pollution moves through a night. The live pass must
/// let those gradients average away rather than copy the skies of the
/// canvas's first frames into the stack.
#[test]
fn a_mosaic_with_changing_gradients_keeps_its_live_output() {
    let directory = tempfile::tempdir().unwrap();
    let paths = changing_gradient_panels(&directory);
    let mut options = StackOptions {
        normalization: NormalizationMode::LocalBackground { tile_size: 32 },
        rejection: RejectionMode::None,
        ..StackOptions::default()
    };
    options.acceptance.minimum_overlap_fraction = 0.1;
    let mut stacker = LiveStacker::with_expanding_reference(
        &paths,
        CalibrationMasters::default(),
        options,
        usize::MAX,
        |_, _, _, _| Continue::Yes,
    )
    .unwrap();
    for (index, path) in paths.iter().enumerate() {
        assert!(
            matches!(
                stacker.push_fits(path).unwrap(),
                FrameDisposition::Accepted(_)
            ),
            "changing-gradient frame {index} was not accepted"
        );
    }
    check(
        "MOSAIC_CHANGING_GRADIENTS",
        &summary(&stacker.snapshot().unwrap().image),
        MOSAIC_CHANGING_GRADIENTS,
    );
}

/// Four panels in a square through a telephoto lens with mild distortion,
/// its stars as faint as a short exposure's. The distortion takes less than a
/// fifth off the residual the panels' similarities leave, which star
/// centroids keep near 0.3 px: real, and enough to soften a mosaic's corners.
#[test]
fn a_mildly_distorted_lens_mosaic_keeps_its_output() {
    let directory = tempfile::tempdir().unwrap();
    let paths = lens_panels(&directory, (0.003, 0.0), 40.0, 8000, 1, (2, 2), 20.0);
    let mut options = StackOptions {
        normalization: NormalizationMode::None,
        ..StackOptions::default()
    };
    options.acceptance.minimum_overlap_fraction = 0.1;
    let mut stacker = LiveStacker::with_expanding_reference(
        &paths,
        CalibrationMasters::default(),
        options,
        usize::MAX,
        |_, _, _, _| Continue::Yes,
    )
    .unwrap();
    let lens = stacker.expanded_lens().is_some();
    for (index, path) in paths.iter().enumerate() {
        assert!(
            matches!(
                stacker.push_fits(path).unwrap(),
                FrameDisposition::Accepted(_)
            ),
            "mild lens mosaic frame {index} was not accepted"
        );
    }
    let mut snapshot = stacker.snapshot().unwrap();
    for sample in &mut snapshot.image.data {
        if !sample.is_finite() {
            *sample = 0.0;
        }
    }
    let mut fingerprint = summary(&snapshot.image);
    fingerprint.extend([
        f64::from(u8::from(lens)),
        f64::from(snapshot.accepted_frames),
        snapshot.image.width as f64,
        snapshot.image.height as f64,
    ]);
    check("MILD_LENS_MOSAIC", &fingerprint, MILD_LENS_MOSAIC);
}

/// Dithered Bayer frames whose bright stars clip at the level their
/// headers declare: the stack's map of where its frames clipped, carried
/// through the demosaic and registration as each frame was.
#[test]
fn a_clipped_bayer_stack_keeps_its_clipped_map() {
    let directory = tempfile::tempdir().unwrap();
    let bayer = [("BAYERPAT".to_string(), HeaderValue::String("RGGB".into()))];
    let saturate = [WriteHeaderCard::new("SATURATE", HeaderValue::Float(3000.0))];
    let paths = SHIFTS[..6]
        .iter()
        .enumerate()
        .map(|(frame, &(dx, dy))| {
            let mut image = bayer_field(dx, dy, frame);
            for value in &mut image.data {
                *value = value.min(3000.0);
            }
            let path = directory.path().join(format!("clipped-{frame}.fits"));
            write_processed_image_fits_f32(&path, &image, &bayer, &saturate).unwrap();
            path
        })
        .collect::<Vec<_>>();
    let mut stacker =
        LiveStacker::open_fits(&paths[0], None, None, None, None, StackOptions::default()).unwrap();
    for path in &paths[1..] {
        admit(stacker.push_fits(path).unwrap());
    }
    let map = stacker.clipped_fraction().expect("the bright stars clip");
    let mut fingerprint = summary(&map);
    fingerprint.extend([
        map.data.iter().map(|&value| f64::from(value)).sum::<f64>(),
        map.data.iter().filter(|&&value| value >= 0.5).count() as f64,
    ]);
    check("CLIPPED_BAYER_MAP", &fingerprint, CLIPPED_BAYER_MAP);
}

/// The same broad nebular emission is present in both images, while the
/// sky gradient differs. Local offsets must compare corresponding samples
/// so the nebula itself does not become a background correction.
#[test]
fn a_shared_nebula_local_background_keeps_its_output() {
    let mut reference = Vec::new();
    let mut source = Vec::new();
    for index in 0..WIDTH * HEIGHT {
        let (x, y) = ((index % WIDTH) as f32, (index / WIDTH) as f32);
        let filament = (-(x - 70.0 - 0.35 * y).powi(2) / 800.0).exp();
        let cloud = (-((x - 135.0).powi(2) / 3000.0 + (y - 50.0).powi(2) / 1900.0)).exp();
        let nebula = 350.0 * filament + 180.0 * cloud + 60.0 * (x * 0.04).sin() * (y * 0.05).sin();
        let signal = 100.0 + nebula + noise(index, 0, 0.4);
        reference.push(signal);
        source.push(0.8 * signal + 40.0 + 0.18 * x - 0.13 * y + 0.0015 * x * y);
    }
    let reference = LinearImage::new(WIDTH, HEIGHT, 1, reference).unwrap();
    let mut source = LinearImage::new(WIDTH, HEIGHT, 1, source).unwrap();
    let map = seiza_stacking::NormalizationMap::estimate(
        &reference,
        &source,
        NormalizationMode::LocalBackground { tile_size: 32 },
    )
    .unwrap();
    map.apply(&mut source).unwrap();
    let mut fingerprint = vec![f64::from(map.mean_gain()), f64::from(map.mean_offset())];
    fingerprint.extend(summary(&source));
    check(
        "SHARED_NEBULA_BACKGROUND",
        &fingerprint,
        SHARED_NEBULA_BACKGROUND,
    );
}

fn mosaic_stack(model: RegistrationModel) -> StackSnapshot {
    let directory = tempfile::tempdir().unwrap();
    let paths = mosaic_panels(&directory);
    let mut options = StackOptions::default();
    options.registration.model = model;
    options.acceptance.minimum_overlap_fraction = 0.1;
    let mut stacker = LiveStacker::with_expanding_reference(
        &paths,
        CalibrationMasters::default(),
        options,
        usize::MAX,
        |_, _, _, _| Continue::Yes,
    )
    .unwrap();
    for path in &paths {
        admit(stacker.push_fits(path).unwrap());
    }
    let snapshot = stacker.snapshot().unwrap();
    assert!(snapshot.image.width > MOSAIC_PANEL.0);
    snapshot
}

/// Raw lights on a faint sky, calibrated by a dark master and by a flat
/// master that a bias master calibrated, then resampled with Lanczos-3: the
/// path PSF Guard's banded stacks took before #295.
#[test]
fn calibrated_faint_lights_and_a_bias_calibrated_flat_keep_their_output() {
    calibrated_faint_lights(false, false);
}

#[test]
fn calibrated_faint_lights_and_a_dark_calibrated_flat_keep_their_output() {
    calibrated_faint_lights(true, false);
}

#[test]
fn different_gain_dark_calibrated_flats_keep_the_same_pixel_output() {
    calibrated_faint_lights(true, true);
}

fn calibrated_faint_lights(dark_flat: bool, different_gain: bool) {
    let directory = tempfile::tempdir().unwrap();
    // A fixed pattern the dark carries and subtracts.
    let pattern = |index: usize| 2.0 * (index.wrapping_mul(2_654_435_761) % 7) as f32;
    let raw = |stem: &str, kind: &str, seed: usize, sample: &dyn Fn(usize) -> f32| {
        (0..7)
            .map(|frame| {
                let data = (0..WIDTH * HEIGHT)
                    .map(|index| sample(index) + noise(index, seed + frame, 6.0))
                    .collect();
                write_raw(&directory, stem, frame, data, kind)
            })
            .collect::<Vec<_>>()
    };
    let bias = raw("bias", "BIAS", 100, &|_| 300.0);
    let darks = raw("dark", "DARK", 200, &|index| 325.0 + pattern(index));
    let flats = raw("flat", "FLAT", 300, &|index| {
        300.0 + 20_000.0 * vignetting(index)
    });
    let bias = build_master_from_fits(&bias, MasterFrameKind::Bias, &MasterBuildOptions::default())
        .unwrap();
    let mut flat = build_master_from_fits(
        &flats,
        MasterFrameKind::Flat,
        &MasterBuildOptions {
            bias: (!dark_flat).then(|| bias.image.clone()),
            dark: dark_flat.then_some(seiza_stacking::MasterDark {
                image: bias.image,
                exposure_seconds: Some(60.0),
                bias_subtracted: false,
                bayer: None,
            }),
            ..MasterBuildOptions::default()
        },
    )
    .unwrap();
    check(
        "FLAT_MASTER_BIASED",
        &summary(&flat.image),
        FLAT_MASTER_BIASED,
    );
    let dark = build_master_from_fits(
        &darks,
        MasterFrameKind::Dark,
        &MasterBuildOptions::default(),
    )
    .unwrap();
    let (dark_path, flat_path) = (
        directory.path().join("master-dark.fits"),
        directory.path().join("master-flat.fits"),
    );
    if different_gain {
        flat.reference_headers
            .push(("GAIN".into(), HeaderValue::Integer(0)));
        flat.reference_headers
            .push(("OFFSET".into(), HeaderValue::Integer(10)));
    }
    write_master_fits_f32(&dark_path, &dark).unwrap();
    write_master_fits_f32(&flat_path, &flat).unwrap();
    let lights = (0..SHIFTS.len())
        .map(|frame| {
            let sky = faint_field(frame);
            let data = sky
                .data
                .iter()
                .enumerate()
                .map(|(index, &value)| 325.0 + pattern(index) + vignetting(index) * value)
                .collect();
            let path = write_raw(&directory, "light", frame, data, "LIGHT");
            if different_gain {
                let frame = seiza_stacking::FitsFrame::open(&path).unwrap();
                write_processed_image_fits_f32(
                    &path,
                    &frame.image,
                    &frame.headers,
                    &[
                        WriteHeaderCard::new("GAIN", HeaderValue::Integer(100)),
                        WriteHeaderCard::new("OFFSET", HeaderValue::Integer(50)),
                    ],
                )
                .unwrap();
            }
            path
        })
        .collect::<Vec<_>>();
    let options = StackOptions {
        interpolation: Interpolation::Lanczos3,
        match_tolerances: seiza_calibration::MatchTolerances {
            allow_flat_gain_offset_mismatch: different_gain,
            ..Default::default()
        },
        ..StackOptions::default()
    };
    let mut stacker = LiveStacker::open_fits(
        &lights[0],
        None,
        Some(&dark_path),
        Some(&flat_path),
        None,
        options,
    )
    .unwrap();
    for path in &lights[1..] {
        admit(stacker.push_fits(path).unwrap());
    }
    let snapshot = stacker.snapshot().unwrap();
    check(
        "CALIBRATED_LANCZOS",
        &summary(&snapshot.image),
        CALIBRATED_LANCZOS,
    );
}

#[test]
fn a_bias_master_keeps_its_output() {
    let directory = tempfile::tempdir().unwrap();
    let paths = (0..7)
        .map(|frame| {
            let data = (0..WIDTH * HEIGHT)
                .map(|index| 300.0 + 0.001 * (index / WIDTH) as f32 + noise(index, frame, 5.0))
                .collect::<Vec<_>>();
            write_raw(&directory, "bias", frame, data, "BIAS")
        })
        .collect::<Vec<_>>();
    let master = build_master_from_fits(
        &paths,
        MasterFrameKind::Bias,
        &MasterBuildOptions::default(),
    )
    .unwrap();
    check("BIAS_MASTER", &summary(&master.image), BIAS_MASTER);
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

/// Push every light after the first onto a stack opened from it.
fn fits_stack(paths: &[PathBuf], options: StackOptions) -> StackSnapshot {
    let mut stacker = LiveStacker::open_fits(&paths[0], None, None, None, None, options).unwrap();
    for path in &paths[1..] {
        admit(stacker.push_fits(path).unwrap());
    }
    stacker.snapshot().unwrap()
}

fn bayer_stack(options: StackOptions) -> StackSnapshot {
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
    fits_stack(&paths, options)
}

/// Dithered raw lights whose sensor has hot, warm and dead pixels that stay
/// put while the sky moves.
fn defective_lights(directory: &tempfile::TempDir) -> Vec<PathBuf> {
    let mut state = 0x9e37_79b9_u32;
    let mut next = |limit: usize| {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as usize % limit
    };
    let defects = (0..60)
        .map(|defect| {
            let index = (8 + next(HEIGHT - 16)) * WIDTH + 8 + next(WIDTH - 16);
            let level = match defect % 3 {
                0 => 3_000.0,
                1 => 400.0,
                _ => -1.0,
            };
            (index, level)
        })
        .collect::<Vec<_>>();
    SHIFTS
        .iter()
        .enumerate()
        .map(|(frame, &(dx, dy))| {
            let mut image = mono_field(dx, dy, 0.96 + 0.01 * frame as f32, frame);
            for &(index, level) in &defects {
                if level < 0.0 {
                    image.data[index] = 0.0;
                } else {
                    image.data[index] += level;
                }
            }
            write_raw(directory, "light", frame, image.data, "LIGHT")
        })
        .collect()
}

/// A faint sky, 30 above zero with noise of sigma 11.5, as calibrated narrowband
/// lights leave it, with each frame rotated a little against the first so
/// the sub-pixel phase sweeps across it.
fn faint_field(frame: usize) -> LinearImage {
    let stars = stars();
    let (dx, dy) = SHIFTS[frame];
    let angle = (frame as f32 - 3.5) * 0.15_f32.to_radians();
    let (sin, cos) = angle.sin_cos();
    let (cx, cy) = ((WIDTH - 1) as f32 / 2.0, (HEIGHT - 1) as f32 / 2.0);
    let data = (0..WIDTH * HEIGHT)
        .map(|index| {
            let (x, y) = ((index % WIDTH) as f32 - cx, (index / WIDTH) as f32 - cy);
            let sky_x = cos * x + sin * y + cx - dx;
            let sky_y = -sin * x + cos * y + cy - dy;
            30.0 + star_light(&stars, sky_x, sky_y) + noise(index, frame, 20.0)
        })
        .collect();
    LinearImage::new(WIDTH, HEIGHT, 1, data).unwrap()
}

/// Each mosaic panel's size; the second panel starts 70% of a width along.
const MOSAIC_PANEL: (usize, usize) = (224, 160);

/// Two panels of a crowded field, two dithered frames each.
fn mosaic_panels(directory: &tempfile::TempDir) -> Vec<PathBuf> {
    let (width, height) = MOSAIC_PANEL;
    let step = width * 7 / 10;
    let sky_width = width + step + 8;
    let mut state = 0x5eed_u32;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1 << 24) as f32
    };
    let stars = (0..sky_width * (height + 8) / 160)
        .map(|_| {
            let x = 4.0 + next() * (sky_width - 8) as f32;
            let y = 4.0 + next() * height as f32;
            (x, y, 200.0 + 6_000.0 * next() * next())
        })
        .collect::<Vec<_>>();
    let mut paths = Vec::new();
    for panel in 0..2 {
        for dither in 0..2 {
            let (ox, oy) = ((panel * step + dither * 3) as f32, (dither * 2) as f32);
            let frame = panel * 2 + dither;
            let data = (0..width * height)
                .map(|index| {
                    let (x, y) = ((index % width) as f32 + ox, (index / width) as f32 + oy);
                    100.0 + 0.02 * x + star_light(&stars, x, y) + noise(index, frame, 3.0)
                })
                .collect::<Vec<_>>();
            let path = directory
                .path()
                .join(format!("panel-{panel}-{dither}.fits"));
            let image = LinearImage::new(width, height, 1, data).unwrap();
            write_processed_image_fits_f32(&path, &image, &[], &[]).unwrap();
            paths.push(path);
        }
    }
    paths
}

/// Three overlapping camera pointings through an off-centre radial lens,
/// two dithered exposures per pointing. Sky stars are projected from rays,
/// rather than translated on a flat plane, so panel-edge registration has
/// a measurable nonlinear component after the lens rebuilds the canvas.
fn projected_lens_panels(directory: &tempfile::TempDir) -> Vec<PathBuf> {
    lens_panels(directory, (0.02, 0.004), 5.0, 4000, 1, (3, 1), 1.0)
}

/// [`projected_lens_panels`] through a lens whose radial terms, in units of
/// the squared half diagonal, are `(k1, k2)`, with `stars` stars over sky
/// noise of `sigma`, every length `scale` times as long.
fn lens_panels(
    directory: &tempfile::TempDir,
    (k1, k2): (f64, f64),
    sigma: f32,
    stars: usize,
    scale: usize,
    (rows, columns): (usize, usize),
    telephoto: f64,
) -> Vec<PathBuf> {
    let size = scale as f64;
    let (width, height) = (600 * scale, 400 * scale);
    let (cx, cy) = (323.5 * size, 181.5 * size);
    let focal = 2400.0 * size * telephoto;
    let radius_squared = (width * width + height * height) as f64 * 0.25;
    let mut state = 7_u32;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f64 / (1 << 24) as f64
    };
    let field = (0..stars)
        .map(|_| {
            (
                (-10.0 + next() * (600 + (columns - 1) * 300 + 20) as f64) * size,
                (-10.0 + next() * (400 + (rows - 1) * 300 + 50) as f64) * size,
                200.0 + 5.0 / (0.01 + next()).powf(1.5),
            )
        })
        .collect::<Vec<_>>();
    let mut paths = Vec::new();
    for panel in 0..rows * columns {
        let (row, column) = (panel / columns, panel % columns);
        // A second column turns the camera about the frame's vertical axis.
        let (sin_x, cos_x) = ((column as f64 * 300.0 * size) / focal).atan().sin_cos();
        for dither in 0..2 {
            let frame = panel * 2 + dither;
            let angle = ((row as f64 * 300.0 * size + dither as f64 * 2.3) / focal).atan();
            let (sin, cos) = angle.sin_cos();
            let mut data = (0..width * height)
                .map(|index| 100.0 + noise(index, frame, sigma))
                .collect::<Vec<_>>();
            for &(sky_x, sky_y, peak) in &field {
                let ray_x = sky_x - cx - dither as f64 * 3.1;
                let ray_y = sky_y - cy;
                let mut depth = sin * ray_y + cos * focal;
                let mut across = ray_x;
                if column > 0 {
                    (across, depth) =
                        (cos_x * ray_x - sin_x * depth, sin_x * ray_x + cos_x * depth);
                }
                let u = focal * across / depth;
                let v = focal * (cos * ray_y - sin * focal) / depth;
                let q = (u * u + v * v) / radius_squared;
                let factor = 1.0 + k1 * q + k2 * q * q;
                let (sx, sy) = (cx + u * factor, cy + v * factor);
                if sx < -5.0 || sy < -5.0 || sx > width as f64 + 5.0 || sy > height as f64 + 5.0 {
                    continue;
                }
                let (x0, y0) = (sx.floor() as isize - 4, sy.floor() as isize - 4);
                for y in y0.max(0)..(y0 + 10).min(height as isize) {
                    for x in x0.max(0)..(x0 + 10).min(width as isize) {
                        let distance = (x as f64 - sx).powi(2) + (y as f64 - sy).powi(2);
                        data[y as usize * width + x as usize] +=
                            (peak * (-distance / 2.9).exp()) as f32;
                    }
                }
            }
            let path = directory.path().join(format!("lens-{panel}-{dither}.fits"));
            let image = LinearImage::new(width, height, 1, data).unwrap();
            let headers = [
                ("FOCALLEN".into(), HeaderValue::Float(12.0 * telephoto)),
                ("XPIXSZ".into(), HeaderValue::Float(5.0 / size)),
            ];
            write_processed_image_fits_f32(&path, &image, &headers, &[]).unwrap();
            paths.push(path);
        }
    }
    paths
}

/// How far below the first [`narrow_mosaic_panels`] panel the second one
/// starts: 92% of a panel's 512 rows.
const NARROW_MOSAIC_STEP: usize = 471;

/// Two 768 by 512 panels of a crowded field, one above the other and
/// sharing 8% of their footprints, two dithered frames each. A narrow
/// overlap holds few stars, so the field is wider than [`mosaic_panels`]'s.
fn narrow_mosaic_panels(directory: &tempfile::TempDir) -> Vec<PathBuf> {
    let (width, height) = (768, 512);
    let (sky_width, sky_height) = (width + 4, height + NARROW_MOSAIC_STEP + 4);
    let mut sky = vec![0.0_f32; sky_width * sky_height];
    let mut state = 0x0be1_u32;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1 << 24) as f32
    };
    for _ in 0..sky_width * sky_height / 160 {
        let (x, y) = (
            4.0 + next() * (sky_width - 8) as f32,
            4.0 + next() * (sky_height - 8) as f32,
        );
        let flux = 200.0 + 6_000.0 * next() * next();
        for py in y as usize - 4..=y as usize + 4 {
            for px in x as usize - 4..=x as usize + 4 {
                let (dx, dy) = (px as f32 - x, py as f32 - y);
                sky[py * sky_width + px] += flux * (-(dx * dx + dy * dy) / 4.5).exp();
            }
        }
    }
    let mut paths = Vec::new();
    for panel in 0..2 {
        for dither in 0..2 {
            let (ox, oy) = (dither * 3, panel * NARROW_MOSAIC_STEP + dither);
            let frame = panel * 2 + dither;
            let data = (0..width * height)
                .map(|index| {
                    let (x, y) = (index % width + ox, index / width + oy);
                    100.0 + 0.02 * y as f32 + sky[y * sky_width + x] + noise(index, frame, 3.0)
                })
                .collect::<Vec<_>>();
            let path = directory
                .path()
                .join(format!("narrow-{panel}-{dither}.fits"));
            let image = LinearImage::new(width, height, 1, data).unwrap();
            write_processed_image_fits_f32(&path, &image, &[], &[]).unwrap();
            paths.push(path);
        }
    }
    paths
}

/// Twelve exposures of each of [`mosaic_panels`]' two panels, interleaved,
/// each with a linear sky gradient of its own direction.
fn changing_gradient_panels(directory: &tempfile::TempDir) -> Vec<PathBuf> {
    let originals = mosaic_panels(directory);
    let panels = [0, 2].map(|index| {
        seiza_image_io::read_astronomy(&originals[index])
            .unwrap()
            .image
            .into_physical_f32()
    });
    let (width, height) = MOSAIC_PANEL;
    (0..24)
        .map(|frame| {
            // Successive golden angles spread the directions evenly.
            let angle = frame as f32 * 2.399_963;
            let (gx, gy) = (30.0 * angle.cos(), 30.0 * angle.sin());
            let data = panels[frame % 2]
                .iter()
                .enumerate()
                .map(|(index, &sample)| {
                    let x = (index % width) as f32 / width as f32 - 0.5;
                    let y = (index / width) as f32 / height as f32 - 0.5;
                    sample + gx * x + gy * y + noise(index, frame, 0.2)
                })
                .collect::<Vec<_>>();
            let path = directory.path().join(format!("gradient-{frame}.fits"));
            let image = LinearImage::new(width, height, 1, data).unwrap();
            write_processed_image_fits_f32(&path, &image, &[], &[]).unwrap();
            path
        })
        .collect()
}

fn uneven_mosaic_panels(directory: &tempfile::TempDir) -> Vec<PathBuf> {
    let originals = mosaic_panels(directory);
    let panels = [0, 2].map(|index| {
        seiza_image_io::read_astronomy(&originals[index])
            .unwrap()
            .image
            .into_physical_f32()
    });
    let (width, height) = MOSAIC_PANEL;
    (0..43)
        .map(|frame| {
            let panel = usize::from(frame >= 21);
            let transparency = if panel == 0 {
                1.0
            } else if frame == 42 {
                1.0 / 3.0
            } else {
                0.5
            };
            let data = panels[panel]
                .iter()
                .enumerate()
                .map(|(index, &sample)| {
                    let x = (index % width) as f32 / width as f32;
                    let y = (index / width) as f32 / height as f32;
                    let gradient = if panel == 0 {
                        0.0
                    } else if frame == 42 {
                        120.0 * x * x + 40.0 * y * y
                    } else {
                        15.0 * x * x + 10.0 * y * y
                    };
                    transparency * (sample + gradient) + noise(index, frame, 0.2)
                })
                .collect::<Vec<_>>();
            let path = directory.path().join(format!("background-{frame}.fits"));
            let image = LinearImage::new(width, height, 1, data).unwrap();
            write_processed_image_fits_f32(&path, &image, &[], &[]).unwrap();
            path
        })
        .collect()
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

/// Persisted superpixel provenance must reproduce registered crops.
#[test]
fn a_superpixel_mapping_crop_keeps_its_output() {
    let directory = tempfile::tempdir().unwrap();
    let raw = bayer_field(0.0, 0.0, 0);
    let path = directory.path().join("mapping.fits");
    let headers = [("BAYERPAT".to_string(), HeaderValue::String("RGGB".into()))];
    write_processed_image_fits_f32(&path, &raw, &headers, &[]).unwrap();
    let options = StackOptions {
        demosaic: Demosaic::Superpixel,
        bin_factor: 2,
        normalization: NormalizationMode::None,
        ..StackOptions::default()
    };
    let stacker = LiveStacker::open_fits(&path, None, None, None, None, options).unwrap();
    let mut rgb = Vec::new();
    for y in (0..HEIGHT).step_by(2) {
        for x in (0..WIDTH).step_by(2) {
            let i = y * WIDTH + x;
            rgb.extend([
                raw.data[i],
                (raw.data[i + 1] + raw.data[i + WIDTH]) * 0.5,
                raw.data[i + WIDTH + 1],
            ]);
        }
    }
    let source = LinearImage::new(WIDTH / 2, HEIGHT / 2, 3, rgb).unwrap();
    let mapping: seiza_stacking::RegisteredFrameMapping =
        serde_json::from_str(&serde_json::to_string(&stacker.reference_mapping()).unwrap())
            .unwrap();
    let crop = mapping
        .extract_region(
            &source,
            seiza_stacking::ReferenceRegion {
                x: 8,
                y: 8,
                width: 64,
                height: 48,
            },
        )
        .unwrap();
    check("SUPERPIXEL_MAPPING", &summary(&crop), SUPERPIXEL_MAPPING);
}
const SUPERPIXEL_MAPPING: &[f64] = &[
    4.6129709e2,
    5.3272965e2,
    3.4753945e3,
    3.4899965e2,
    4.1961527e2,
    3.2700420e3,
    2.3674625e2,
    2.8542620e2,
    2.0870115e3,
    3.9798587e2,
    2.9957031e2,
    1.9816434e2,
    3.9873520e2,
    2.9978717e2,
    1.9965741e2,
    4.0193481e2,
    2.9670645e2,
    1.9881876e2,
    4.0075842e2,
    2.9681415e2,
    2.0030560e2,
    3.9871417e2,
    3.0025806e2,
    2.0054459e2,
    4.0067859e2,
    3.0272089e2,
    2.0001205e2,
    4.0058521e2,
    3.0220166e2,
    1.9886996e2,
    7.0804315e2,
    5.7922803e2,
    3.8427917e2,
    3.9960498e2,
    3.0205103e2,
    1.9988741e2,
    4.8768317e2,
    3.3742047e2,
    2.3734398e2,
    3.9926544e2,
    3.0087814e2,
    2.0023781e2,
    3.9913580e2,
    3.0262817e2,
    2.0076703e2,
    4.0098938e2,
    3.0328943e2,
    2.0037813e2,
    4.0247852e2,
    3.0105304e2,
    2.0066864e2,
    4.8872424e2,
    3.6936542e2,
    2.8197803e2,
    4.0155557e2,
    3.0108139e2,
    2.0062837e2,
    4.0217484e2,
    3.0054605e2,
    1.9856046e2,
    4.0086761e2,
    2.9843954e2,
    2.0179404e2,
    4.0032507e2,
    3.0067365e2,
    1.9861986e2,
    4.0054169e2,
    2.9966174e2,
    1.9960168e2,
    4.0074237e2,
    3.0304419e2,
    1.9828320e2,
    3.9814484e2,
    2.9712723e2,
    1.9894531e2,
    3.9910040e2,
    2.9690900e2,
    2.0062030e2,
    3.9986386e2,
    2.9974023e2,
    1.9847993e2,
    4.0010007e2,
    3.0147623e2,
    2.0163464e2,
];

#[path = "common/continuum.rs"]
mod continuum_measurement_fixture;

#[test]
fn continuum_saturated_candidate_sampling_keeps_its_output() {
    check(
        "CONTINUUM_CANDIDATE_SAMPLING",
        &continuum_measurement_fixture::measurement_summary(),
        &[1.0, 0.35, 12.0, 4.0, 16.0],
    );
}

fn emission_fixture() -> seiza_stacking::EmissionBlendLine {
    use seiza_stacking::*;
    let source = |name: &str| ContinuumSource {
        name: name.into(),
        revision: "fixture-v1".into(),
        transfer: ContinuumTransfer::Linear,
        saturation: Some(1000.0),
        clipping_revision: None,
    };
    EmissionBlendLine {
        plan: ContinuumPlan {
            psf_matching: None,
            schema_version: 1,
            sources: ContinuumSources {
                width: 1,
                height: 1,
                narrowband: source("ha"),
                references: vec![source("red")],
            },
            coefficients: vec![0.2],
            reference_flux_ratios: vec![1.0],
            origin: ContinuumOrigin::Manual,
            diagnostics: None,
        },
        reference_channels: vec![0],
        amount: 0.25,
        background_offset: 0.0,
    }
}
#[test]
fn emission_blending_keeps_its_output() {
    use seiza_stacking::*;
    let base = LinearImage::new(1, 1, 3, vec![0.2, 0.3, 0.4]).unwrap();
    let residual = LinearImage::new(1, 1, 1, vec![0.04]).unwrap();
    let projected = blend_emission_rgb(
        &base,
        &[&residual],
        &[emission_fixture()],
        &Default::default(),
    )
    .unwrap();
    check(
        "EMISSION_RGB",
        &projected
            .data
            .iter()
            .map(|v| f64::from(*v))
            .collect::<Vec<_>>(),
        &[0.25, 0.3, 0.4],
    );
    let nebula = LinearImage::new(1, 1, 3, vec![0.1, 0.2, 0.3]).unwrap();
    let broadband = LinearImage::new(1, 1, 3, vec![0.5, 0.6, 0.7]).unwrap();
    let mixed = blend_display_structure(&nebula, &broadband, 0.25).unwrap();
    check(
        "EMISSION_STRUCTURE",
        &mixed.data.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
        &[0.2, 0.3, 0.4],
    );
}
#[test]
fn calibrated_emission_addition_keeps_its_output() {
    use seiza_stacking::*;
    let base = LinearImage::new(1, 1, 3, vec![0.2, 0.3, 0.4]).unwrap();
    let delta = LinearImage::new(1, 1, 3, vec![0.1, 0.2, 0.3]).unwrap();
    let result = add_rgb_contribution(&base, &delta, [2.0, 1.0, 0.5]).unwrap();
    check(
        "CALIBRATED_EMISSION",
        &result
            .data
            .iter()
            .map(|v| f64::from(*v))
            .collect::<Vec<_>>(),
        &[0.4, 0.5, 0.55],
    );
}
#[test]
fn continuum_clipping_map_keeps_its_output() {
    use seiza_stacking::*;
    let (n, b, positions, mut sources, options) = continuum_measurement_fixture::saturated_field();
    let masks: Vec<Vec<u8>> = [&n, &b]
        .iter()
        .map(|image| image.data.iter().map(|v| u8::from(*v >= 1000.0)).collect())
        .collect();
    for source in std::iter::once(&mut sources.narrowband).chain(&mut sources.references) {
        source.saturation = None;
        source.clipping_revision = Some("producer-v1".into());
    }
    let outcome = measure_and_fit_continuum_masked(
        &n,
        &[&b],
        &sources,
        &masks.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        &positions,
        &options,
        &Default::default(),
    )
    .unwrap();
    let ContinuumFitOutcome::Fitted { plan } = outcome else {
        panic!("masked fit skipped: {outcome:?}")
    };
    let d = plan.diagnostics.unwrap();
    check(
        "CONTINUUM_CLIPPING",
        &[
            plan.coefficients[0],
            d.used_stars as f64,
            d.validation_stars as f64,
            d.rejected[&ContinuumSkipReason::Saturated] as f64,
        ],
        &[0.35, 12.0, 4.0, 16.0],
    );
}

#[test]
fn continuum_deeper_candidate_search_keeps_its_output() {
    check(
        "CONTINUUM_DEEPER_CANDIDATES",
        &continuum_measurement_fixture::deeper_candidate_summary(),
        &[1.0, 0.35, 12.0, 4.0, 100.0],
    );
}

#[test]
fn continuum_profile_validation_keeps_its_output() {
    check(
        "CONTINUUM_PROFILE",
        &continuum_measurement_fixture::profile_summary(),
        &[0.0, 0.0],
    );
}

#[test]
fn continuum_psf_matching_keeps_its_output() {
    use seiza_stacking::*;
    let (n, b, positions, sources, options) = continuum_measurement_fixture::profile_field();
    let photo = ContinuumPhotometryOptions {
        psf_matching: Some(ContinuumPsfOptions {
            target_fwhm: Some(5.0),
        }),
        ..Default::default()
    };
    let ContinuumFitOutcome::Fitted { plan } =
        measure_and_fit_continuum(&n, &[&b], &sources, &positions, &options, &photo).unwrap()
    else {
        panic!("Matching should pass");
    };
    let residual = subtract_continuum(&n, &[&b], &sources, &plan).unwrap();
    let report = plan.psf_matching.as_ref().unwrap();
    check(
        "CONTINUUM_PSF",
        &[
            plan.coefficients[0],
            report.kernel_sigma[0],
            report.kernel_sigma[1],
            residual.data[30 * n.width + 30] as f64,
            plan.diagnostics
                .as_ref()
                .unwrap()
                .pixel_validation
                .as_ref()
                .unwrap()
                .p90_absolute_fraction,
        ],
        &[
            3.4987700e-1,
            1.5519567e0,
            1.6884675e0,
            4.9436164e0,
            2.3122674e-2,
        ],
    );
}

#[test]
fn green_reduction_and_response_calibration_keep_their_output() {
    use seiza_stacking::*;
    let mut display = LinearImage::new(1, 1, 3, vec![0.2, 0.8, 0.3]).unwrap();
    reduce_green_excess(&mut display, 0.5).unwrap();
    check(
        "GREEN_REDUCTION",
        &display
            .data
            .iter()
            .map(|v| f64::from(*v))
            .collect::<Vec<_>>(),
        &[0.2, 0.55, 0.3],
    );
    let plan = emission_fixture().plan;
    let cal = EmissionFluxCalibration {
        revision: "fixture-v1".into(),
        unit: "flux/pixel".into(),
        lines: vec!["ha".into()],
        sources: vec![plan.sources.clone()],
        narrowband_response: vec![vec![1.0]],
        continuum_coefficients: vec![vec![0.2]],
        residual_zero_levels: vec![0.0],
        reference_response: vec![vec![vec![1.0]]],
        residual_covariance: Some(vec![vec![4.0]]),
    };
    let residual = LinearImage::new(1, 1, 1, vec![4.0]).unwrap();
    let (flux, report) = solve_emission_flux(&[&residual], &[plan], &cal).unwrap();
    check(
        "CALIBRATED_FLUX",
        &[
            f64::from(flux[0].data[0]),
            report.effective_response[0][0],
            report.flux_covariance.unwrap()[0][0],
        ],
        &[5.0, 0.8, 6.25],
    );
}

#[test]
fn star_excluded_balance_statistics_keep_their_output() {
    use seiza_stacking::*;
    let (n, b, _, _, _) = continuum_measurement_fixture::profile_field();
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
    seiza_stacking::screen_rgb_stars_reduced(&mut nebula, &original, &starless, 0.5).unwrap();
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
    let stars = seiza_stacking::stretch_rgb_stars(&original, &zero, 3.0).unwrap();
    let mut nebula = LinearImage::new(2, 1, 3, vec![0.2; 6]).unwrap();
    seiza_stacking::screen_rgb_star_layer(&mut nebula, &stars, 0.0).unwrap();
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
