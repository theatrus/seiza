//! Regressions from the two most discrepant pixels in the 200 ASI294MM
//! 600 s darks evaluated against APP; these are measured input histories,
//! not an algorithm-generated expectation. APP's dominant-state means are
//! independently reproduced by a median/MAD calculation in the evaluation.
use seiza_stacking::{
    LinearImage, MasterBuildOptions, MasterFrameKind, MasterRejectionMethod,
    build_master_from_fits, write_processed_image_fits_f32,
};

#[test]
fn measured_variable_pixels_match_app_while_steady_hot_signal_survives() {
    let histories: serde_json::Value =
        serde_json::from_str(include_str!("data/dark_pixel_histories.json")).unwrap();
    let histories = histories.as_array().unwrap();
    let directory = tempfile::tempdir().unwrap();
    let paths = (0..200)
        .map(|index| {
            let path = directory.path().join(format!("dark-{index}.fits"));
            let mut data = vec![1_000.0; 16];
            for (pixel, history) in histories.iter().enumerate() {
                data[pixel] = history["samples"][index].as_f64().unwrap() as f32;
            }
            // Do not spatially remove actual stable hot pixels or glow.
            data[2] = 50_000.0;
            data[3] = 2_000.0;
            write_processed_image_fits_f32(
                &path,
                &LinearImage::new(4, 4, 1, data).unwrap(),
                &[],
                &[],
            )
            .unwrap();
            path
        })
        .collect::<Vec<_>>();
    let master = build_master_from_fits(
        &paths,
        MasterFrameKind::Dark,
        &MasterBuildOptions::default(),
    )
    .unwrap();
    assert_eq!(master.rejection_method, MasterRejectionMethod::MedianMad);
    assert_eq!(master.input_frames, 200);
    assert_eq!(
        master.image.data[0],
        histories[0]["app_adu"].as_f64().unwrap() as f32
    );
    assert!(
        (master.image.data[1] - histories[1]["app_adu"].as_f64().unwrap() as f32).abs() < 0.001
    );
    assert_eq!(master.image.data[2], 50_000.0);
    assert_eq!(master.image.data[3], 2_000.0);
    assert_eq!(master.fallback_pixels, 0);
    assert_eq!(master.rejected_samples, 68 + 66);
    assert_eq!(master.accepted_samples + master.rejected_samples, 16 * 200);
    assert_eq!(
        master
            .input_statistics
            .iter()
            .map(|s| s.rejected_samples)
            .sum::<u64>(),
        134
    );
}

#[test]
fn aggressive_thresholds_fall_back_to_the_temporal_median() {
    let directory = tempfile::tempdir().unwrap();
    let paths = [100.0, 100.0, 102.0, 102.0]
        .into_iter()
        .enumerate()
        .map(|(i, value)| {
            let path = directory.path().join(format!("dark-{i}.fits"));
            write_processed_image_fits_f32(
                &path,
                &LinearImage::new(2, 2, 1, vec![value; 4]).unwrap(),
                &[],
                &[],
            )
            .unwrap();
            path
        })
        .collect::<Vec<_>>();
    let options = MasterBuildOptions {
        rejection: seiza_stacking::MasterRejectionOptions {
            low_sigma: 0.01,
            high_sigma: 0.01,
        },
        ..Default::default()
    };
    let master = build_master_from_fits(&paths, MasterFrameKind::Dark, &options).unwrap();
    assert_eq!(master.image.data, vec![101.0; 4]);
    assert_eq!(master.fallback_pixels, 4);
    assert_eq!(master.accepted_samples, 0);
    assert_eq!(master.rejected_samples, 16);
}
