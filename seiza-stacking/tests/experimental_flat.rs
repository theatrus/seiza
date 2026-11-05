use seiza_stacking::{
    BayerLayout, CancelSignal, ExperimentalFlatPreprocessingOptions, LinearImage,
    preprocess_flat_spatial,
};

#[test]
fn same_plane_mask_and_negative_response_are_preserved() {
    let width = 32;
    let mut data = (0..width * width)
        .map(|i| if i % 2 == 0 { 1000.0 } else { 100.0 })
        .collect::<Vec<_>>();
    let hot = 16 * width + 16;
    let low = 8 * width + 8;
    data[hot] = 50000.0;
    data[low] = 500.0;
    let mut image = LinearImage::new(width, width, 1, data).unwrap();
    let bayer = BayerLayout {
        pattern: seiza_fits::BayerPattern::Rggb,
        x_offset: 0,
        y_offset: 0,
    };
    let mut mask = vec![0; width * width];
    let count = preprocess_flat_spatial(
        &mut image,
        Some(bayer),
        &ExperimentalFlatPreprocessingOptions::default(),
        Some(&mut mask),
        None,
    )
    .unwrap();
    assert_eq!(count, 1);
    assert_eq!(mask.iter().map(|&v| u64::from(v)).sum::<u64>(), count);
    assert_eq!(mask[hot], 1);
    assert_eq!(image.data[hot], 1000.0);
    assert_eq!(image.data[low], 500.0);
}

#[test]
fn cancellation_and_invalid_options_do_not_publish_partial_corrections() {
    let mut image = LinearImage::new(16, 16, 1, vec![1000.0; 256]).unwrap();
    image.data[8 * 16 + 8] = 50000.0;
    let before = image.data.clone();
    let cancel = CancelSignal::new(|| true);
    let mut mask = vec![7; 256];
    assert!(
        preprocess_flat_spatial(
            &mut image,
            None,
            &ExperimentalFlatPreprocessingOptions::default(),
            Some(&mut mask),
            Some(&cancel)
        )
        .is_err()
    );
    assert_eq!(image.data, before);
    assert!(mask.iter().all(|&v| v == 7));
    for sigma in [0.0, f32::NAN, f32::INFINITY] {
        let options = ExperimentalFlatPreprocessingOptions {
            sigma,
            ..Default::default()
        };
        assert!(preprocess_flat_spatial(&mut image, None, &options, None, None).is_err());
        assert_eq!(image.data, before);
    }
}
