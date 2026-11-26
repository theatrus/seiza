//! Screening stars back over a starless picture.

use seiza_compose::{screen_rgb_stars, screen_rgb_stars_reduced};
use seiza_stacking::LinearImage;

#[test]
fn linked_star_reduction_preserves_color_coverage_and_the_zero_path() {
    let (width, height) = (7, 7);
    let base = [0.1_f32, 0.2, 0.3];
    let nebula = LinearImage::new(width, height, 3, base.repeat(width * height)).unwrap();
    let starless = LinearImage::new(width, height, 3, vec![0.2; width * height * 3]).unwrap();
    let mut data = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let value = 0.8 * (-((x as f32 - 3.0).powi(2) + (y as f32 - 3.0).powi(2)) / 4.0).exp();
            for color in [1.0, 0.5, 0.25] {
                data.push(1.0 - 0.8 * (1.0 - value * color));
            }
        }
    }
    let original = LinearImage::new(width, height, 3, data).unwrap();
    let mut legacy = nebula.clone();
    screen_rgb_stars(&mut legacy, &original, &starless).unwrap();
    let mut zero = nebula.clone();
    screen_rgb_stars_reduced(&mut zero, &original, &starless, 0.0).unwrap();
    assert_eq!(legacy.data, zero.data);
    let mut reduced = nebula.clone();
    screen_rgb_stars_reduced(&mut reduced, &original, &starless, 0.5).unwrap();
    for (i, (old, new)) in legacy.data.iter().zip(&reduced.data).enumerate() {
        assert!(*new <= *old + 1e-6 && *new >= base[i % 3] - 1e-6);
    }
    let center = (3 * width + 3) * 3;
    assert!(reduced.data[center] < legacy.data[center]);
    let layer: Vec<_> = (0..3)
        .map(|c| (reduced.data[center + c] - base[c]) / (1.0 - base[c]))
        .collect();
    assert!((layer[1] / layer[0] - 0.5).abs() < 1e-6);
    assert!((layer[2] / layer[0] - 0.25).abs() < 1e-6);
    assert!(screen_rgb_stars_reduced(&mut reduced, &original, &starless, 1.1).is_err());
    let mut original = original;
    original.data[(3 * width + 2) * 3] = f32::NAN;
    let mut masked = nebula;
    screen_rgb_stars_reduced(&mut masked, &original, &starless, 1.0).unwrap();
    assert!(
        masked.data[(3 * width + 2) * 3..(3 * width + 2) * 3 + 3]
            .iter()
            .all(|v| v.is_nan())
    );
    assert_eq!(
        &masked.data[center..center + 3],
        &legacy.data[center..center + 3]
    );
}
