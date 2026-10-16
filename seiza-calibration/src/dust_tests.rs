//! Tests for [`crate::dust`] on synthetic responses: each shape found where
//! and as large as it is, the false positives each guard is for, matching
//! across sessions and persistence within one.

use super::*;
use crate::flat_response::metric::tests::synthetic;

const W: usize = 512;
const H: usize = 384;
const BIN: usize = 7;

/// A response with the given disks `(x, y, radius, depth)` and noise.
fn plane(disks: &[(f32, f32, f32, f32)], noise_level: f32, seed: u64) -> Plane {
    synthetic(W, H, disks, 0.02, &[], noise_level, seed)
}

/// Light the middle of a shadow again, as a central obstruction does.
fn hole(plane: &mut Plane, x: f32, y: f32, radius: f32, depth: f32) {
    for (index, value) in plane.data.iter_mut().enumerate() {
        let (px, py) = ((index % W) as f32, (index / W) as f32);
        if (px - x).hypot(py - y) <= radius {
            *value /= 1.0 - depth;
        }
    }
}

fn large(map: &DustMap) -> Vec<DustFeature> {
    map.features
        .iter()
        .copied()
        .filter(|f| f.shape.is_large())
        .collect()
}

#[test]
fn a_faint_wide_disk_is_found_at_its_size() {
    // A refractor's filter dust: 0.3% deep, 80 plane pixels across.
    let map = find_dust(
        &plane(&[(200.0, 150.0, 40.0, 0.003)], 0.0005, 1),
        BIN,
        &DustOptions::default(),
    )
    .unwrap();
    let disks = large(&map);
    assert_eq!(disks.len(), 1, "{:?}", map.features);
    let disk = disks[0];
    assert_eq!(disk.shape, DustShape::Disk);
    assert!((disk.x / BIN as f32 - 200.0).abs() < 3.0, "{disk:?}");
    assert!((disk.radius / BIN as f32 - 40.0).abs() < 4.0, "{disk:?}");
    assert!((disk.depth - 0.003).abs() < 0.001, "{disk:?}");
    assert!(
        map.features.iter().all(|f| f.shape.is_large()),
        "{:?}",
        map.features
    );
}

#[test]
fn noise_and_a_gradient_alone_hold_no_dust() {
    for seed in 0..3 {
        let map = find_dust(&plane(&[], 0.001, 10 + seed), BIN, &DustOptions::default()).unwrap();
        assert!(map.features.is_empty(), "{:?}", map.features);
        assert!(map.vignetting.is_empty(), "{:?}", map.vignetting);
    }
}

#[test]
fn a_donut_is_one_feature_with_its_hole() {
    let mut p = plane(&[(250.0, 190.0, 45.0, 0.006)], 0.0005, 2);
    hole(&mut p, 250.0, 190.0, 16.0, 0.006);
    let map = find_dust(&p, BIN, &DustOptions::default()).unwrap();
    let found = large(&map);
    assert_eq!(found.len(), 1, "{:?}", map.features);
    assert_eq!(found[0].shape, DustShape::Donut, "{:?}", found[0]);
    let ratio = found[0].hole.unwrap();
    assert!((0.2..0.55).contains(&ratio), "{ratio}");
    // The bright centre is neither a spot of its own nor a cancelled depth.
    assert!(found[0].depth > 0.004, "{:?}", found[0]);
    assert_eq!(map.features.len(), 1);
}

#[test]
fn optics_give_a_distance_and_rule_out_donuts() {
    let optics = Optics {
        focal_ratio: 4.9,
        pixel_size_um: 3.76,
        obstruction: 0.0,
    };
    let mut p = plane(&[(250.0, 190.0, 42.0, 0.004)], 0.0005, 3);
    hole(&mut p, 250.0, 190.0, 15.0, 0.004);
    let options = DustOptions {
        optics: Some(optics),
        ..DustOptions::default()
    };
    let map = find_dust(&p, BIN, &options).unwrap();
    let found = large(&map);
    assert_eq!(found.len(), 1, "{:?}", map.features);
    assert_eq!(found[0].shape, DustShape::Disk);
    // 84 plane pixels of 7 sensor pixels at f/4.9 and 3.76 µm: 10.8 mm.
    let distance = found[0].distance_mm.unwrap();
    assert!((distance - 10.8).abs() < 1.5, "{distance}");
    assert!((optics.shadow_diameter_px(distance) - 2.0 * found[0].radius).abs() < 1.0);
}

#[test]
fn overlapping_disks_stay_disks_and_no_circle_wraps_them() {
    let disks = [
        (180.0, 160.0, 35.0, 0.004),
        (230.0, 180.0, 35.0, 0.004),
        (330.0, 230.0, 35.0, 0.003),
    ];
    let map = find_dust(&plane(&disks, 0.0005, 4), BIN, &DustOptions::default()).unwrap();
    let found = large(&map);
    assert!(found.len() >= 2, "{:?}", map.features);
    for f in &found {
        assert!(
            f.radius / (BIN as f32) < 50.0,
            "a circle round several: {f:?}"
        );
    }
}

#[test]
fn a_compact_dip_is_found_and_a_cold_pixel_is_not() {
    let mut p = plane(&[(120.0, 100.0, 4.0, 0.03)], 0.002, 5);
    p.data[300 * W + 400] *= 0.97;
    let map = find_dust(&p, BIN, &DustOptions::default()).unwrap();
    let compact: Vec<_> = map
        .features
        .iter()
        .filter(|f| !f.shape.is_large())
        .collect();
    assert_eq!(compact.len(), 1, "{:?}", map.features);
    assert!((compact[0].x / BIN as f32 - 120.0).abs() < 3.0);
}

#[test]
fn a_shadow_in_one_corner_is_vignetting_not_dust() {
    let mut p = plane(&[], 0.0005, 6);
    for (index, value) in p.data.iter_mut().enumerate() {
        let (x, y) = ((index % W) as f32, (index / W) as f32);
        let d = x.hypot(H as f32 - y);
        if d < 90.0 {
            *value *= 1.0 - 0.03 * (1.0 - d / 90.0);
        }
    }
    let map = find_dust(&p, BIN, &DustOptions::default()).unwrap();
    assert_eq!(map.vignetting.len(), 1, "{:?}", map.vignetting);
    assert_eq!(map.vignetting[0].edge, Edge::BottomLeft);
    assert!(map.features.is_empty(), "{:?}", map.features);
}

#[test]
fn features_pair_within_a_tolerance_that_grows_with_size() {
    let disk = |x: f32, radius: f32| DustFeature {
        x,
        y: 1000.0,
        left: 0,
        top: 0,
        right: 0,
        bottom: 0,
        pixels: 0,
        depth: 0.003,
        shape: DustShape::Disk,
        radius,
        hole: None,
        distance_mm: None,
        seen_in: None,
    };
    let options = DustOptions::default();
    // A donut of radius 400 moving 56 pixels: within 15% of its radius. A
    // small one moving 30: beyond 25 pixels, so moved.
    let matches = match_features(
        &[disk(1000.0, 400.0), disk(3000.0, 60.0)],
        &[disk(1056.0, 400.0), disk(3030.0, 60.0)],
        &options,
    );
    let pair = |i: usize| matches.iter().find(|m| m.earlier == Some(i)).unwrap();
    assert!(pair(0).within_tolerance && pair(0).later == Some(0));
    assert!(!pair(1).within_tolerance && pair(1).later == Some(1));
    assert_eq!(pair(1).offset, Some(30.0));
    // Unlike sizes are not the same dust.
    let matches = match_features(&[disk(1000.0, 400.0)], &[disk(1000.0, 200.0)], &options);
    assert_eq!(matches.len(), 2);
}

#[test]
fn dust_moved_by_focus_is_within_tolerance_and_new_dust_is_new() {
    let earlier = plane(&[(200.0, 150.0, 40.0, 0.004)], 0.0005, 7);
    // Two plane pixels (14 sensor pixels) of focus shift, and a new disk.
    let later = plane(
        &[(202.0, 151.0, 40.0, 0.004), (380.0, 250.0, 30.0, 0.003)],
        0.0005,
        8,
    );
    let changes = find_changes(&earlier, &later, BIN, &DustOptions::default()).unwrap();
    let counted: Vec<_> = changes
        .changes
        .iter()
        .filter(|c| !c.within_tolerance)
        .collect();
    assert_eq!(counted.len(), 1, "{:?}", changes.changes);
    assert_eq!(counted[0].kind, ChangeKind::New);
    assert!((counted[0].feature.x / BIN as f32 - 380.0).abs() < 4.0);
    assert!(changes.vignetting.is_empty());
}

#[test]
fn a_star_one_sky_flat_kept_is_not_dust() {
    let earlier = synthetic(W, H, &[], 0.02, &[(150.0, 120.0, 0.004)], 0.0005, 9);
    let later = synthetic(W, H, &[], 0.02, &[], 0.0005, 10);
    let changes = find_changes(&earlier, &later, BIN, &DustOptions::default()).unwrap();
    assert!(changes.changes.is_empty(), "{:?}", changes.changes);
}

#[test]
fn a_session_counts_each_feature_in_its_flats() {
    let level = 30_000.0_f32;
    let flats: Vec<Plane> = (0..6)
        .map(|frame| {
            let mut p = plane(&[(200.0, 150.0, 40.0, 0.004)], 0.0015, 20 + frame);
            // A drop of dew in the first flat only.
            if frame == 0 {
                for (index, value) in p.data.iter_mut().enumerate() {
                    let (x, y) = ((index % W) as f32, (index / W) as f32);
                    if (x - 380.0).hypot(y - 260.0) <= 4.0 {
                        *value *= 0.9;
                    }
                }
            }
            for value in &mut p.data {
                *value = *value * level + 300.0;
            }
            p
        })
        .collect();
    let options = DustOptions::default();
    let map = find_session_dust(&flats, BIN, 300.0, &options).unwrap();
    assert_eq!(map.frames, 6);
    let disks = large(&map);
    assert_eq!(disks.len(), 1, "{:?}", map.features);
    assert_eq!(disks[0].seen_in, Some(SeenIn { frames: 6, of: 6 }));
    assert!(
        map.features.iter().all(|f| f.shape.is_large()),
        "{:?}",
        map.features
    );
    let first = frame_dust(&flats[0], 300.0, &map, &options).unwrap();
    assert_eq!(first.seen, vec![0]);
    assert_eq!(first.transients.len(), 1, "{:?}", first.transients);
    let second = frame_dust(&flats[1], 300.0, &map, &options).unwrap();
    assert!(second.transients.is_empty(), "{:?}", second.transients);
}

#[test]
fn lone_dips_need_more_than_five_sigma_on_a_large_frame() {
    let floor = lone_dip_floor(26_000_000, 1.0, 0.05);
    assert!((6.4..6.7).contains(&floor), "{floor}");
    assert!(lone_dip_floor(500_000, 1.0, 0.05) < floor);
    assert!((upper_tail_sigma(0.001_349_9) - 3.0).abs() < 0.01);
}
