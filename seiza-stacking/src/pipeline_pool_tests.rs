use super::tests::{bits, concurrent, constant_bias, frame_set, stacker_from};
use super::*;
use crate::{CalibrationMasters, LinearImage, StackOptions};
use rayon::prelude::*;
use std::sync::{Arc, Barrier, Mutex};

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|index| format!("stack-compute-{index}"))
        .build()
        .unwrap()
}

fn assert_same_stack(actual: &LiveStacker, expected: &LiveStacker) {
    let actual_snapshot = actual.snapshot().unwrap();
    let expected_snapshot = expected.snapshot().unwrap();
    assert_eq!(
        bits(&actual_snapshot.image.data),
        bits(&expected_snapshot.image.data)
    );
    assert_eq!(
        bits(&actual_snapshot.variance.data),
        bits(&expected_snapshot.variance.data)
    );
    assert_eq!(actual_snapshot.coverage, expected_snapshot.coverage);
    assert_eq!(
        actual_snapshot.rejected_samples,
        expected_snapshot.rejected_samples
    );
    assert_eq!(
        actual_snapshot.accepted_frames,
        expected_snapshot.accepted_frames
    );
    assert_eq!(
        actual_snapshot.rejected_frames,
        expected_snapshot.rejected_frames
    );
    assert_eq!(actual.input_paths(), expected.input_paths());
}

#[test]
fn supplied_pools_preserve_exact_order_results_and_read_failures() {
    let (directory, paths) = frame_set(9);
    let missing = directory.path().join("missing.fits");
    let batch = [&paths[1..4], &[missing, paths[2].clone()], &paths[4..]].concat();
    for threads in [1, 3] {
        let pool = pool(threads);
        let options = StackOptions {
            acceptance: crate::FrameAcceptanceCriteria {
                minimum_integrated_fraction: 1.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let build = || {
            pool.install(|| {
                LiveStacker::new(
                    FitsFrame::open(&paths[0]).unwrap(),
                    CalibrationMasters::default(),
                    options.clone(),
                )
                .unwrap()
            })
        };
        let mut expected = build();
        let mut expected_outcomes = Vec::new();
        for path in &batch {
            expected_outcomes.push(format!("{:?}", pool.install(|| expected.push_fits(path))));
        }
        let mut actual = build();
        let mut actual_outcomes = Vec::new();
        let coordinator = std::thread::current().id();
        let report = actual
            .push_fits_pipelined_with_pool(&batch, &concurrent(3), &pool, |path, outcome| {
                assert_eq!(std::thread::current().id(), coordinator);
                assert_eq!(path, batch[actual_outcomes.len()]);
                actual_outcomes.push(format!("{outcome:?}"));
                Continue::Yes
            })
            .unwrap();
        assert_eq!(actual_outcomes, expected_outcomes);
        assert_same_stack(&actual, &expected);
        assert!(report.frames.rejected > 0);
        assert_eq!(report.frames.failed, 2);
        assert_eq!(report.workers, 3);
        assert_eq!(report.execution, PipelineExecution::Overlapped);
        assert!(report.timings.read_decode > Duration::ZERO);
        assert!(report.timings.preparation > Duration::ZERO);
        assert!(report.timings.integration > Duration::ZERO);
        assert!(report.timings.elapsed > Duration::ZERO);
    }
}

#[test]
fn supplied_pool_session_master_changes_match_sequential_batches() {
    let (_directory, paths) = frame_set(7);
    let pool = pool(2);
    let mut expected = stacker_from(&paths[0]);
    let mut actual = stacker_from(&paths[0]);
    for (batch, bias) in [(&paths[1..4], 100.0), (&paths[4..], 250.0)] {
        expected.set_calibration(constant_bias(bias)).unwrap();
        actual.set_calibration(constant_bias(bias)).unwrap();
        for path in batch {
            pool.install(|| expected.push_fits(path)).unwrap();
        }
        let _ = actual
            .push_fits_pipelined_with_pool(batch, &concurrent(2), &pool, |_, _| Continue::Yes)
            .unwrap();
    }
    assert_same_stack(&actual, &expected);
}

#[test]
fn rgb_and_cfa_frames_match_sequential_preparation() {
    let (directory, mono_paths) = frame_set(4);
    let pool = pool(2);
    for cfa in [false, true] {
        let paths: Vec<_> = mono_paths
            .iter()
            .enumerate()
            .map(|(index, path)| {
                let mut frame = FitsFrame::open(path).unwrap();
                if cfa {
                    frame.headers.push((
                        "BAYERPAT".into(),
                        seiza_fits::HeaderValue::String("RGGB".into()),
                    ));
                } else {
                    frame.image = LinearImage::new(
                        frame.image.width,
                        frame.image.height,
                        3,
                        frame
                            .image
                            .data
                            .iter()
                            .flat_map(|&value| [value, value * 0.8, value * 1.2])
                            .collect(),
                    )
                    .unwrap();
                }
                let output = directory.path().join(format!("color-{cfa}-{index}.fits"));
                crate::write_processed_image_fits_f32(&output, &frame.image, &frame.headers, &[])
                    .unwrap();
                if cfa {
                    // The processed-image writer intentionally drops CFA metadata;
                    // supply it explicitly for this raw-frame fixture.
                    seiza_fits::write_f32_image(
                        &output,
                        frame.image.width,
                        frame.image.height,
                        seiza_fits::F32ImageData::Mono(&frame.image.data),
                        &[seiza_fits::WriteHeaderCard::new(
                            "BAYERPAT",
                            seiza_fits::HeaderValue::String("RGGB".into()),
                        )],
                    )
                    .unwrap();
                }
                output
            })
            .collect();
        let mut expected = pool.install(|| stacker_from(&paths[0]));
        let mut actual = pool.install(|| stacker_from(&paths[0]));
        for path in &paths[1..] {
            pool.install(|| expected.push_fits(path)).unwrap();
        }
        let report = actual
            .push_fits_pipelined_with_pool(&paths[1..], &concurrent(2), &pool, |_, _| Continue::Yes)
            .unwrap();
        assert_same_stack(&actual, &expected);
        assert_eq!(actual.snapshot().unwrap().image.channels, 3);
        assert_eq!(report.frames.failed, 0);
        assert_eq!(report.frames.integrated, paths.len() - 1);
    }
}

#[test]
fn nested_same_or_other_single_thread_pool_falls_back_without_deadlock() {
    let (_directory, paths) = frame_set(4);
    let supplied = pool(1);
    let other = pool(1);
    for caller in [&supplied, &other] {
        let mut actual = stacker_from(&paths[0]);
        let mut expected = stacker_from(&paths[0]);
        for path in &paths[1..] {
            supplied.install(|| expected.push_fits(path)).unwrap();
        }
        let report = caller
            .install(|| {
                actual.push_fits_pipelined_with_pool(
                    &paths[1..],
                    &concurrent(3),
                    &supplied,
                    |_, _| {
                        assert!(caller.current_thread_index().is_some());
                        Continue::Yes
                    },
                )
            })
            .unwrap();
        assert_eq!(report.execution, PipelineExecution::SequentialRayonFallback);
        assert_eq!(report.workers, 1);
        assert_same_stack(&actual, &expected);
    }
}

#[test]
fn normalized_xisf_inputs_match_the_legacy_pool_fallback() {
    let (directory, paths) = frame_set(3);
    let mut frame = FitsFrame::open(&paths[2]).unwrap();
    for sample in &mut frame.image.data {
        *sample /= 65535.0;
    }
    frame.image.data[0] = 0.0;
    frame.image.data[1] = 1.0;
    let xisf = directory.path().join("normalized.xisf");
    seiza_xisf::write_f32_image(
        &xisf,
        frame.image.width,
        frame.image.height,
        seiza_fits::F32ImageData::Mono(&frame.image.data),
        &[],
    )
    .unwrap();
    assert_eq!(FitsFrame::open(&xisf).unwrap().bounds, Some((0.0, 1.0)));
    let batch = vec![paths[1].clone(), xisf];
    let pool = pool(2);
    let options = PipelineOptions {
        normalized_full_scale: Some(65535.0),
        ..concurrent(2)
    };
    let mut expected = stacker_from(&paths[0]);
    let mut actual = stacker_from(&paths[0]);
    let expected_report = pool
        .install(|| expected.push_fits_pipelined(&batch, &options, |_, _| Continue::Yes))
        .unwrap();
    let actual_report = actual
        .push_fits_pipelined_with_pool(&batch, &options, &pool, |_, _| Continue::Yes)
        .unwrap();
    assert_eq!(actual_report.frames, expected_report);
    assert_eq!(actual_report.frames.integrated, 2);
    assert_same_stack(&actual, &expected);
}

#[test]
fn reads_and_preparation_overlap_on_the_intended_threads() {
    let (_directory, paths) = frame_set(3);
    let pool = pool(2);
    let read_barrier = Barrier::new(2);
    let prep_barrier = Barrier::new(2);
    let reads = Mutex::new(Vec::new());
    let preparations = Mutex::new(Vec::new());
    let coordinator = std::thread::current().id();
    let mut actual = stacker_from(&paths[0]);
    let report = actual
        .run_pipeline(
            &paths[1..],
            &concurrent(2),
            ComputePool(Some(&pool)),
            &|path| {
                assert!(
                    rayon::current_thread_index().is_none(),
                    "reads must not occupy a Rayon worker"
                );
                assert_ne!(std::thread::current().id(), coordinator);
                reads.lock().unwrap().push(std::thread::current().id());
                read_barrier.wait();
                FitsFrame::open(path)
            },
            &|frame, half, scale| {
                assert!(pool.current_thread_index().is_some());
                preparations
                    .lock()
                    .unwrap()
                    .push(std::thread::current().id());
                prep_barrier.wait();
                (0..32)
                    .into_par_iter()
                    .for_each(|_| assert!(pool.current_thread_index().is_some()));
                prepare_decoded(frame, half, scale)
            },
            |_, outcome| {
                assert_eq!(std::thread::current().id(), coordinator);
                outcome.unwrap();
                Continue::Yes
            },
        )
        .unwrap();
    assert_eq!(report.frames.integrated, 2);
    let reads = reads.lock().unwrap();
    let preparations = preparations.lock().unwrap();
    assert_ne!(reads[0], reads[1]);
    assert_ne!(preparations[0], preparations[1]);
}

#[test]
fn budget_caps_explicit_workers_and_includes_rgb_intermediates() {
    let pool = pool(8);
    let mono = LinearImage::new(100, 100, 1, vec![0.0; 10_000]).unwrap();
    let rgb = LinearImage::new(100, 100, 3, vec![0.0; 30_000]).unwrap();
    let mono_memory = PoolPipelineMemory::for_reference(mono.pixel_count(), mono.sample_count());
    let rgb_memory = PoolPipelineMemory::for_reference(rgb.pixel_count(), rgb.sample_count());
    assert_eq!(mono_memory.worker_bytes, 800_000);
    assert_eq!(rgb_memory.worker_bytes, 1_120_000);
    assert_eq!(rgb_memory.integration_bytes, 120_000);
    assert_eq!(rgb_memory.in_flight_bytes(0), 0);
    assert_eq!(
        PoolPipelineMemory::for_reference(usize::MAX, usize::MAX).in_flight_bytes(2),
        usize::MAX
    );
    let mut options = concurrent(11);
    options.max_in_flight_bytes = rgb_memory.in_flight_bytes(2);
    assert_eq!(resolve_pool_workers(&options, &rgb, &pool).unwrap(), 2);
    assert_eq!(
        options.resolve_workers(120_000),
        11,
        "legacy explicit-worker contract stays unchanged"
    );
    options.max_in_flight_bytes = 1;
    assert!(resolve_pool_workers(&options, &rgb, &pool).is_err());
    options.max_in_flight_bytes = usize::MAX;
    options.workers = Some(usize::MAX);
    assert_eq!(
        resolve_pool_workers(&options, &rgb, &pool).unwrap(),
        MAXIMUM_WORKERS
    );
}

#[test]
fn insufficient_budget_fails_before_any_source_is_read() {
    let (_directory, paths) = frame_set(2);
    let mut stacker = stacker_from(&paths[0]);
    let pool = pool(1);
    let result = stacker.run_pipeline(
        &paths[1..],
        &PipelineOptions::with_budget(1),
        ComputePool(Some(&pool)),
        &|_| panic!("budget must be checked before reading"),
        &prepare_decoded,
        |_, _| panic!("no frame should be delivered"),
    );
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("cannot fit one worker")
    );
    assert!(stacker.input_paths().is_empty());
}

#[test]
fn integration_is_available_while_another_reader_waits() {
    let (_directory, paths) = frame_set(3);
    let pool = pool(1);
    let (integrated, integration) = std::sync::mpsc::channel();
    let integration = Mutex::new(integration);
    let mut stacker = stacker_from(&paths[0]);
    let report = stacker
        .run_pipeline(
            &paths[1..],
            &concurrent(2),
            ComputePool(Some(&pool)),
            &|path| {
                if path == paths[2] {
                    integration
                        .lock()
                        .unwrap()
                        .recv_timeout(Duration::from_secs(10))
                        .expect("integration must not wait for the slow reader");
                }
                FitsFrame::open(path)
            },
            &prepare_decoded,
            |path, outcome| {
                outcome.unwrap();
                if path == paths[1] {
                    integrated.send(()).unwrap();
                }
                Continue::Yes
            },
        )
        .unwrap();
    assert_eq!(report.frames.integrated, 2);
}

#[test]
fn cancellation_and_callback_panic_join_bounded_in_flight_work() {
    let (_directory, paths) = frame_set(12);
    let pool = pool(1);
    for panic in [false, true] {
        let opened = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut actual = stacker_from(&paths[0]);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            actual.run_pipeline(
                &paths[1..],
                &concurrent(2),
                ComputePool(Some(&pool)),
                &|path| {
                    opened.fetch_add(1, Ordering::Relaxed);
                    FitsFrame::open(path)
                },
                &prepare_decoded,
                |_, _| {
                    assert!(!panic, "callback panic");
                    Continue::No
                },
            )
        }));
        if panic {
            assert!(result.is_err());
        } else {
            assert_eq!(result.unwrap().unwrap().frames.integrated, 1);
        }
        assert_eq!(actual.input_paths().len(), 1);
        assert!(
            opened.load(Ordering::Relaxed) <= 5,
            "two buffered/building frames per worker plus consumed frame"
        );
        let mut expected = stacker_from(&paths[0]);
        for path in &paths[1..] {
            pool.install(|| expected.push_fits(path)).unwrap();
        }
        let _ = actual
            .push_fits_pipelined_with_pool(&paths[2..], &concurrent(2), &pool, |_, _| Continue::Yes)
            .unwrap();
        assert_same_stack(&actual, &expected);
    }
}

#[test]
fn read_and_preparation_panics_propagate_and_do_not_hang() {
    let (_directory, paths) = frame_set(7);
    let pool = pool(1);
    for in_read in [false, true] {
        let mut actual = stacker_from(&paths[0]);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            actual.run_pipeline(
                &paths[1..],
                &concurrent(3),
                ComputePool(Some(&pool)),
                &|path| {
                    assert!(!in_read || path != paths[1], "read panic");
                    FitsFrame::open(path)
                },
                &|frame, half, scale| {
                    assert!(
                        in_read || frame.source.as_ref() != Some(&paths[1]),
                        "preparation panic"
                    );
                    prepare_decoded(frame, half, scale)
                },
                |_, _| Continue::Yes,
            )
        }));
        assert!(result.is_err());
        assert!(actual.input_paths().is_empty());
    }
}

#[test]
fn empty_batch_has_no_workers_or_artificial_timings() {
    let (_directory, paths) = frame_set(1);
    let pool = pool(1);
    let mut stacker = stacker_from(&paths[0]);
    assert_eq!(
        stacker
            .push_fits_pipelined_with_pool(&[], &concurrent(3), &pool, |_, _| unreachable!())
            .unwrap(),
        PoolPipelineReport::default()
    );
}

#[test]
fn callback_can_hold_non_send_coordinator_state() {
    let (_directory, paths) = frame_set(3);
    let pool = pool(1);
    let mut stacker = stacker_from(&paths[0]);
    let count = std::rc::Rc::new(std::cell::Cell::new(0));
    let report = stacker
        .push_fits_pipelined_with_pool(&paths[1..], &concurrent(2), &pool, |_, _| {
            count.set(count.get() + 1);
            Continue::Yes
        })
        .unwrap();
    assert_eq!(report.frames.integrated, count.get());
    assert_eq!(count.get(), 2);
}
