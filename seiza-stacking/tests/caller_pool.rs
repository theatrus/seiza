//! Stacking runs in the Rayon pool its caller installs.
//!
//! Every entry point here runs inside a one-thread and a two-thread pool,
//! and must give the same bits in both. The global pool must never start:
//! Rayon work that left the caller's pool, from a thread of the crate's own
//! or spawned outside the pool, would start it. That can be checked once
//! per process, and each file under `tests/` runs as a process of its own,
//! so this is one test.

use rayon::ThreadPoolBuilder;
use seiza_fits::HeaderValue;
use seiza_stacking::{
    BatchStackOptions, CalibrationMasters, CfaIntegration, ColorOptions, Continue,
    DarkLevelScreening, DrizzleOptions, FitsFrame, FrameWeighting, ImpulseFilterOptions,
    Interpolation, LinearImage, LiveStacker, MasterBuildOptions, MasterFrameKind,
    NormalizationMode, PipelineOptions, RegistrationModel, RejectionMode, SimilarityTransform,
    StackOptions, StackSnapshot, build_master_from_fits_with_scratch, combine_rgb, frame_noise,
    integrate_registered_frames, measure_depth, reference_score, resample_to_reference,
    suppress_impulses,
};
use std::path::{Path, PathBuf};

const WIDTH: usize = 192;
const HEIGHT: usize = 160;
const FRAMES: usize = 6;

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn bits64(value: f64) -> [u32; 2] {
    let bits = value.to_bits();
    [bits as u32, (bits >> 32) as u32]
}

fn snapshot_bits(snapshot: &StackSnapshot) -> Vec<u32> {
    let mut out = bits(&snapshot.image.data);
    out.extend(bits(&snapshot.variance.data));
    out.extend(&snapshot.coverage);
    out.extend(&snapshot.rejected_samples);
    out.push(snapshot.accepted_frames);
    out
}

/// A dithered star field with a little fixed-pattern noise.
fn star_field(frame: usize) -> LinearImage {
    let stars: Vec<(f32, f32, f32)> = (0..24)
        .map(|index| {
            let x = ((index * 7919) % 1000) as f32 / 1000.0 * (WIDTH as f32 - 24.0) + 12.0;
            let y = ((index * 6271) % 1000) as f32 / 1000.0 * (HEIGHT as f32 - 24.0) + 12.0;
            (x, y, 6000.0 + ((index * 37) % 41) as f32 * 300.0)
        })
        .collect();
    let (dx, dy) = (
        ((frame * 13) % 5) as f32 - 2.0,
        ((frame * 7) % 3) as f32 - 1.0,
    );
    let data = (0..WIDTH * HEIGHT)
        .map(|index| {
            let (x, y) = (index % WIDTH, index / WIDTH);
            let mut value = 1000.0 + ((x * 17 + y * 31 + frame * 11) % 23) as f32 * 1.5;
            for (star_x, star_y, brightness) in &stars {
                let ddx = x as f32 - (star_x + dx);
                let ddy = y as f32 - (star_y + dy);
                let r2 = ddx.mul_add(ddx, ddy * ddy);
                if r2 < 40.0 {
                    value += brightness * (-r2 / 3.2).exp();
                }
            }
            value.round()
        })
        .collect();
    LinearImage::new(WIDTH, HEIGHT, 1, data).unwrap()
}

/// A flat field, or a dark at `level` with a little noise.
fn calibration_frame(index: usize, flat: bool) -> LinearImage {
    let data = (0..WIDTH * HEIGHT)
        .map(|sample| {
            let noise = ((sample * 37 + index * 11) % 17) as f32 * 0.6;
            if flat {
                let (x, y) = ((sample % WIDTH) as f32, (sample / WIDTH) as f32);
                let r2 = (x - 96.0).powi(2) + (y - 80.0).powi(2);
                20_000.0 - r2 * 0.05 + noise
            } else {
                500.0 + noise + if index == 4 { 400.0 } else { 0.0 }
            }
        })
        .collect();
    LinearImage::new(WIDTH, HEIGHT, 1, data).unwrap()
}

struct Inputs {
    directory: tempfile::TempDir,
    lights: Vec<PathBuf>,
    bayer: Vec<PathBuf>,
    darks: Vec<PathBuf>,
    flats: Vec<PathBuf>,
}

fn write(
    directory: &Path,
    name: String,
    image: &LinearImage,
    headers: &[(String, HeaderValue)],
) -> PathBuf {
    let path = directory.join(name);
    seiza_stacking::write_processed_image_fits_f32(&path, image, headers, &[]).unwrap();
    path
}

fn inputs() -> Inputs {
    let directory = tempfile::tempdir().unwrap();
    let bayer = [("BAYERPAT".to_string(), HeaderValue::String("RGGB".into()))];
    let path = directory.path();
    Inputs {
        lights: (0..FRAMES)
            .map(|frame| write(path, format!("light-{frame}.fits"), &star_field(frame), &[]))
            .collect(),
        bayer: (0..FRAMES)
            .map(|frame| {
                write(
                    path,
                    format!("bayer-{frame}.fits"),
                    &star_field(frame),
                    &bayer,
                )
            })
            .collect(),
        darks: (0..5)
            .map(|index| {
                write(
                    path,
                    format!("dark-{index}.fits"),
                    &calibration_frame(index, false),
                    &[],
                )
            })
            .collect(),
        flats: (0..4)
            .map(|index| {
                write(
                    path,
                    format!("flat-{index}.fits"),
                    &calibration_frame(index, true),
                    &[],
                )
            })
            .collect(),
        directory,
    }
}

fn open(reference: &Path, options: StackOptions) -> LiveStacker {
    LiveStacker::open_fits(reference, None, None, None, None, options).unwrap()
}

fn options_for(normalization: NormalizationMode, cfa: CfaIntegration) -> StackOptions {
    StackOptions {
        normalization,
        cfa_integration: cfa,
        ..StackOptions::default()
    }
}

/// Everything run inside `pool`, as the bits of each result.
fn run(pool: &rayon::ThreadPool, inputs: &Inputs) -> Vec<(&'static str, Vec<u32>)> {
    let mut results = Vec::new();
    let lights = &inputs.lights;
    let scratch = inputs.directory.path();
    let batch = BatchStackOptions {
        scratch_directory: Some(scratch.to_path_buf()),
        ..BatchStackOptions::default()
    };
    // Scratch that cannot be created, so reintegration reads every frame
    // whole for each pass instead of in bands.
    let whole_frames = BatchStackOptions {
        scratch_directory: Some(lights[0].join("not-a-directory")),
        ..BatchStackOptions::default()
    };
    let drizzle = DrizzleOptions {
        scale: 2,
        drop_shrink: Some(0.8),
    };

    // Frames pushed one at a time, then measured and replayed.
    for (name, normalization) in [
        ("global", NormalizationMode::Global),
        (
            "local background",
            NormalizationMode::LocalBackground { tile_size: 32 },
        ),
    ] {
        let (live, banded, whole, drizzled) = pool.install(|| {
            let mut stacker = open(
                &lights[0],
                options_for(normalization, CfaIntegration::Demosaic),
            );
            for path in &lights[1..] {
                stacker.push_fits(path).unwrap();
            }
            let depth = measure_depth(stacker.view()).unwrap();
            let live = stacker.snapshot().unwrap();
            let banded = stacker.reintegrate(&batch, |_, _, _| {}).unwrap();
            let whole = stacker.reintegrate(&whole_frames, |_, _, _| {}).unwrap();
            let (_, drizzled) = stacker
                .reintegrate_drizzled(&batch, &drizzle, |_, _, _| {})
                .unwrap();
            let (_, whole_drizzled) = stacker
                .reintegrate_drizzled(&whole_frames, &drizzle, |_, _, _| {})
                .unwrap();
            assert_eq!(bits(&drizzled.image.data), bits(&whole_drizzled.image.data));
            let mut live_bits = snapshot_bits(&live);
            live_bits.extend(bits64(depth.noise));
            (
                live_bits,
                snapshot_bits(&banded.snapshot),
                snapshot_bits(&whole.snapshot),
                bits(&drizzled.image.data),
            )
        });
        assert_eq!(
            banded, whole,
            "{name}: banded and whole-frame replays differ"
        );
        results.push((name, live));
        results.push((name, banded));
        results.push((name, drizzled));
    }

    // Two calibration sets loaded from paths, and the options the command
    // line uses. Replay loads the first set again while it prepares frames
    // ahead; one stack keeps its frames as they are stacked, the other reads
    // every frame whole for each pass.
    let sessions = pool.install(|| {
        let mut options = StackOptions {
            normalization: NormalizationMode::LocalBackground { tile_size: 32 },
            interpolation: Interpolation::Lanczos3,
            weighting: FrameWeighting::inverse_noise_variance(),
            ..StackOptions::default()
        };
        options.registration.model = RegistrationModel::Quadratic;
        let stack = |retain: bool| {
            let mut stacker = LiveStacker::open_fits(
                &lights[0],
                Some(&inputs.darks[0]),
                None,
                None,
                None,
                options.clone(),
            )
            .unwrap();
            if retain {
                stacker
                    .retain_frames_for_reintegration(Some(scratch))
                    .unwrap();
            }
            for path in &lights[1..3] {
                stacker.push_fits(path).unwrap();
            }
            stacker
                .set_calibration_from_fits_paths(Some(&inputs.darks[1]), None, None, None)
                .unwrap();
            for path in &lights[3..] {
                stacker.push_fits(path).unwrap();
            }
            stacker
        };
        let (kept, read) = (stack(true), stack(false));
        let banded = kept.reintegrate(&batch, |_, _, _| {}).unwrap();
        let whole = read.reintegrate(&whole_frames, |_, _, _| {}).unwrap();
        assert_eq!(
            snapshot_bits(&banded.snapshot),
            snapshot_bits(&whole.snapshot)
        );
        let (_, banded_drizzle) = kept
            .reintegrate_drizzled(&batch, &drizzle, |_, _, _| {})
            .unwrap();
        let (_, whole_drizzle) = read
            .reintegrate_drizzled(&whole_frames, &drizzle, |_, _, _| {})
            .unwrap();
        assert_eq!(
            bits(&banded_drizzle.image.data),
            bits(&whole_drizzle.image.data)
        );
        let mut out = snapshot_bits(&banded.snapshot);
        out.extend(bits(&banded_drizzle.image.data));
        out
    });
    results.push(("two sessions", sessions));

    // Bayer frames drizzled live, then replayed.
    let (bayer_live, bayer_replayed) = pool.install(|| {
        let options = StackOptions {
            normalization: NormalizationMode::None,
            rejection: RejectionMode::None,
            ..options_for(NormalizationMode::None, CfaIntegration::BayerDrizzle)
        };
        let mut stacker = open(&inputs.bayer[0], options);
        for path in &inputs.bayer[1..] {
            stacker.push_fits(path).unwrap();
        }
        let replayed = stacker.reintegrate(&batch, |_, _, _| {}).unwrap();
        (
            snapshot_bits(&stacker.snapshot().unwrap()),
            snapshot_bits(&replayed.snapshot),
        )
    });
    results.push(("bayer drizzle", bayer_live));
    results.push(("bayer drizzle replay", bayer_replayed));

    // Batches: pipelined and one at a time in a named pool, called from
    // outside it, and the older pipelined call under `install`.
    let named = || pool.install(|| open(&lights[0], StackOptions::default()));
    let mut pipelined = named();
    let _ = pipelined
        .push_fits_pipelined_with_pool(
            &lights[1..],
            &PipelineOptions::default(),
            pool,
            |_, outcome| {
                outcome.unwrap();
                Continue::Yes
            },
        )
        .unwrap();
    let mut sequential = named();
    let _ = sequential
        .push_fits_sequential_with_pool(&lights[1..], None, pool, |_, outcome| {
            outcome.unwrap();
            Continue::Yes
        })
        .unwrap();
    let installed = pool.install(|| {
        let mut stacker = open(&lights[0], StackOptions::default());
        let _ = stacker
            .push_fits_pipelined(&lights[1..], &PipelineOptions::default(), |_, outcome| {
                outcome.unwrap();
                Continue::Yes
            })
            .unwrap();
        stacker.snapshot().unwrap()
    });
    let pipelined = pool.install(|| pipelined.snapshot().unwrap());
    let sequential = pool.install(|| sequential.snapshot().unwrap());
    assert_eq!(snapshot_bits(&pipelined), snapshot_bits(&sequential));
    assert_eq!(snapshot_bits(&pipelined), snapshot_bits(&installed));
    results.push(("pipelined", snapshot_bits(&pipelined)));

    // Masters, calibration and single-frame measures.
    let masters = pool.install(|| {
        let screened = MasterBuildOptions {
            dark_level_screening: Some(DarkLevelScreening::default()),
            ..MasterBuildOptions::default()
        };
        let dark = build_master_from_fits_with_scratch(
            &inputs.darks,
            MasterFrameKind::Dark,
            &screened,
            scratch,
        )
        .unwrap();
        let flat = build_master_from_fits_with_scratch(
            &inputs.flats,
            MasterFrameKind::Flat,
            &MasterBuildOptions::default(),
            scratch,
        )
        .unwrap();
        let mut out = bits(&dark.image.data);
        out.extend(bits(&flat.image.data));
        out.push(dark.input_frames as u32);
        out
    });
    results.push(("masters", masters));
    let calibrated = pool.install(|| {
        let mut stacker = open(&lights[0], StackOptions::default());
        let bias = FitsFrame::open(&inputs.darks[0]).unwrap();
        let masters = CalibrationMasters::new(Some(bias.image), None, None).unwrap();
        stacker.set_calibration(masters).unwrap();
        stacker.push_fits(&lights[1]).unwrap();
        snapshot_bits(&stacker.snapshot().unwrap())
    });
    results.push(("calibrated", calibrated));
    let single = pool.install(|| {
        let frame = FitsFrame::open(&lights[2]).unwrap();
        let score = reference_score(&frame).unwrap();
        let mut image = frame.image.clone();
        let replaced =
            suppress_impulses(&mut image, None, &ImpulseFilterOptions::default()).unwrap();
        let mut out = bits(&frame_noise(&frame.image).unwrap());
        out.extend(bits(&image.data));
        out.push(replaced as u32);
        out.push(score.stars as u32);
        out.push(score.background.to_bits());
        out
    });
    results.push(("single frame", single));

    // Registered frames integrated as a batch, and composed into colour.
    let (batch_bits, colour) = pool.install(|| {
        let frames = lights
            .iter()
            .map(|path| FitsFrame::open(path).unwrap().image)
            .collect::<Vec<_>>();
        let integrated =
            integrate_registered_frames(frames.len(), &BatchStackOptions::default(), |_, index| {
                Ok(frames[index].clone())
            })
            .unwrap();
        let shifted =
            resample_to_reference(&frames[1], WIDTH, HEIGHT, SimilarityTransform::IDENTITY)
                .unwrap();
        let colour =
            combine_rgb(&frames[0], &shifted, &frames[2], &ColorOptions::default()).unwrap();
        (
            snapshot_bits(&integrated.snapshot),
            bits(&colour.image.data),
        )
    });
    results.push(("batch", batch_bits));
    results.push(("colour", colour));
    results
}

#[test]
fn stacking_stays_in_the_callers_pool() {
    let pools = [1, 2].map(|threads| {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(move |index| format!("caller-{threads}-{index}"))
            .build()
            .unwrap()
    });
    let inputs = pools[0].install(inputs);
    let one = run(&pools[0], &inputs);
    let two = run(&pools[1], &inputs);
    assert_eq!(one.len(), two.len());
    for ((name, one), (_, two)) in one.iter().zip(&two) {
        assert!(one == two, "{name} differs between one and two threads");
    }
    ThreadPoolBuilder::new()
        .build_global()
        .expect("work escaped the caller's pool and started the global pool");
}
