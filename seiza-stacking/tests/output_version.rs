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
    Demosaic, DrizzleOptions, FrameDisposition, FrameWeighting, ImpulseFilterOptions,
    Interpolation, LinearImage, LiveStacker, MasterBuildOptions, MasterFrameKind,
    NormalizationMode, RegistrationModel, RejectionMode, StackOptions, StackSnapshot,
    build_master_from_fits, detect_defects, write_processed_image_fits_f32,
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
const BIAS_MASTER: &[f64] = &[
    3.0007062e2, 3.0007339e2, 3.0441794e2, 2.9925772e2, 3.0073709e2, 2.9994614e2,
    3.0028802e2, 2.9983591e2, 3.0033102e2, 3.0190173e2, 3.0158337e2, 2.9967471e2,
    2.9826190e2, 3.0187531e2, 2.9826392e2, 3.0179602e2, 2.9797226e2, 2.9981561e2,
    3.0139807e2, 3.0026102e2, 3.0065182e2, 3.0243387e2, 2.9918311e2, 3.0074722e2,
    2.9894547e2, 2.9980435e2, 3.0077734e2, 2.9691653e2,
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

/// Two overlapping panels on a canvas that grows to hold them both.
#[test]
fn a_mosaic_keeps_its_output() {
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
    for path in &paths {
        admit(stacker.push_fits(path).unwrap());
    }
    let snapshot = stacker.snapshot().unwrap();
    assert!(snapshot.image.width > MOSAIC_PANEL.0);
    check("MOSAIC", &summary(&snapshot.image), MOSAIC);
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
