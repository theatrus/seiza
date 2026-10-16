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

#[test]
fn new_window_dust_in_front_of_a_held_disk_is_new() {
    // A filter disk both sessions hold, and new compact dust inside it.
    let earlier = plane(&[(200.0, 150.0, 40.0, 0.004)], 0.0005, 7);
    let later = plane(
        &[(200.0, 150.0, 40.0, 0.004), (215.0, 160.0, 4.0, 0.03)],
        0.0005,
        8,
    );
    // The later session's own map holds it too, beside the disk.
    let later_map = find_dust(&later, BIN, &DustOptions::default()).unwrap();
    assert!(
        later_map.features.iter().any(|f| !f.shape.is_large()
            && (f.x / BIN as f32 - 215.0).hypot(f.y / BIN as f32 - 160.0) < 4.0),
        "{:?}",
        later_map.features
    );
    let changes = find_changes(&earlier, &later, BIN, &DustOptions::default()).unwrap();
    let counted: Vec<_> = changes
        .changes
        .iter()
        .filter(|c| !c.within_tolerance)
        .collect();
    assert_eq!(counted.len(), 1, "{:?}", changes.changes);
    assert_eq!(counted[0].kind, ChangeKind::New);
}

#[test]
fn new_window_dust_beside_held_window_dust_is_new() {
    // Compact dust both sessions hold, and new compact dust 25 plane px
    // (175 sensor px) away.
    let earlier = plane(&[(200.0, 150.0, 4.0, 0.03)], 0.0005, 7);
    let later = plane(
        &[(200.0, 150.0, 4.0, 0.03), (200.0, 175.0, 4.0, 0.03)],
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
}

fn disk_at(x: f32, y: f32, radius: f32, shape: DustShape) -> DustFeature {
    DustFeature {
        x,
        y,
        left: (x - radius).max(0.0) as u32,
        top: (y - radius).max(0.0) as u32,
        right: (x + radius) as u32,
        bottom: (y + radius) as u32,
        pixels: (std::f32::consts::PI * radius * radius) as u32,
        depth: 0.003,
        shape,
        radius,
        hole: None,
        distance_mm: None,
        seen_in: None,
    }
}

#[test]
fn matching_takes_the_most_pairs_not_the_closest_first() {
    let disk = |x: f32| disk_at(x, 1000.0, 100.0, DustShape::Disk);
    // E0 at 980 and E1 at 1010; L0 at 1000 and L1 at 1034. E1-L0 is closest
    // (10), but taking it leaves E0 (only near L0) and L1 (only near E1)
    // unpaired: one gone and one new where two pairs exist.
    let matches = match_features(
        &[disk(980.0), disk(1010.0)],
        &[disk(1000.0), disk(1034.0)],
        &DustOptions::default(),
    );
    let pair = |i: usize| matches.iter().find(|m| m.earlier == Some(i)).unwrap();
    assert_eq!(pair(0).later, Some(0), "{matches:?}");
    assert_eq!(pair(1).later, Some(1), "{matches:?}");
    assert_eq!(matches.len(), 2, "{matches:?}");
}

#[test]
fn a_patch_pairs_with_the_disk_another_session_saw() {
    // One session resolved a disk; the other merged it with its neighbours
    // into a patch. A dip far from both pairs with neither.
    let options = DustOptions::default();
    let matches = match_features(
        &[disk_at(1000.0, 1000.0, 200.0, DustShape::Disk)],
        &[
            disk_at(1080.0, 1010.0, 300.0, DustShape::Patch),
            disk_at(3000.0, 1000.0, 10.0, DustShape::Compact),
        ],
        &options,
    );
    let pair = matches.iter().find(|m| m.earlier == Some(0)).unwrap();
    assert_eq!(pair.later, Some(0), "{matches:?}");
    assert!(pair.within_tolerance, "{matches:?}");
    assert!(
        matches
            .iter()
            .any(|m| m.later == Some(1) && m.earlier.is_none())
    );
}

#[test]
fn alike_disks_gone_and_new_nearby_are_not_called_moved() {
    // Two filter disks of one size go and two arrive between them: which
    // went where cannot be told, so none is called moved. One alone is.
    // Each gone disk is 2.4 radii from each new one: no overlap, within the
    // three radii a lone moved shadow may travel.
    let options = DustOptions::default();
    let large = |earlier: &[(f32, f32, f32, f32)], later: &[(f32, f32, f32, f32)]| {
        let changes = find_changes(
            &plane(earlier, 0.0005, 50),
            &plane(later, 0.0005, 51),
            BIN,
            &options,
        )
        .unwrap();
        changes
            .changes
            .into_iter()
            .filter(|c| c.feature.shape.is_large())
            .collect::<Vec<_>>()
    };
    let gone = [(180.0, 150.0, 30.0, 0.004), (300.0, 150.0, 30.0, 0.004)];
    let new = [(240.0, 110.0, 30.0, 0.004), (240.0, 190.0, 30.0, 0.004)];
    let both = large(&gone, &new);
    assert_eq!(both.len(), 4, "{both:?}");
    assert!(both.iter().all(|c| c.kind != ChangeKind::Moved), "{both:?}");
    let one = large(&gone[..1], &new[..1]);
    assert_eq!(one.len(), 2, "{one:?}");
    assert!(one.iter().all(|c| c.kind == ChangeKind::Moved), "{one:?}");
}

#[test]
fn changes_do_not_depend_on_the_thread_count() {
    let earlier = plane(
        &[(200.0, 150.0, 40.0, 0.002), (330.0, 230.0, 30.0, 0.0015)],
        0.001,
        31,
    );
    let later = plane(
        &[(203.0, 150.0, 40.0, 0.002), (100.0, 300.0, 25.0, 0.002)],
        0.001,
        32,
    );
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| find_changes(&earlier, &later, BIN, &DustOptions::default()).unwrap())
    };
    assert_eq!(run(1), run(8));
}

#[test]
fn non_finite_and_tiny_planes_do_not_panic() {
    let options = DustOptions::default();
    let mut p = plane(&[(200.0, 150.0, 40.0, 0.004)], 0.0005, 40);
    for i in 0..2000 {
        p.data[i * 37 % (W * H)] = f32::NAN;
    }
    for i in 0..500 {
        p.data[(i * 101 + 7) % (W * H)] = f32::INFINITY;
    }
    for i in 0..500 {
        p.data[(i * 53 + 3) % (W * H)] = f32::NEG_INFINITY;
    }
    let _ = find_dust(&p, BIN, &options);
    let clean = plane(&[], 0.0005, 41);
    let _ = find_changes(&clean, &p, BIN, &options);
    // An all-NaN half.
    let mut half = clean.clone();
    for v in half.data.iter_mut().take(W * H / 2) {
        *v = f32::NAN;
    }
    let _ = find_dust(&half, BIN, &options);
    let _ = find_changes(&clean, &half, BIN, &options);
    // Tiny and degenerate planes.
    for (w, h) in [(32, 32), (33, 400), (400, 33), (31, 31), (1, 1)] {
        let tiny = synthetic(w, h, &[], 0.0, &[], 0.001, 3);
        let _ = find_dust(&tiny, 1, &options);
        let _ = find_changes(&tiny, &tiny, 1, &options);
        let _ = metric::change_map(&tiny, &tiny);
    }
    let constant = Plane {
        width: 64,
        height: 64,
        data: vec![1.0; 64 * 64],
    };
    let _ = find_dust(&constant, 1, &options);
    let zero = Plane {
        width: 64,
        height: 64,
        data: vec![0.0; 64 * 64],
    };
    assert!(find_dust(&zero, 1, &options).is_none());
    // bin 0 and a huge bin.
    let _ = find_dust(&clean, 0, &options);
    let _ = find_dust(&clean, 1 << 20, &options);
}

#[test]
fn a_frame_smaller_than_its_session_does_not_panic() {
    let session = DustMap {
        algorithm_version: DUST_ALGORITHM_VERSION,
        bin: BIN,
        features: vec![DustFeature::compact(
            3000.0,
            2000.0,
            (2990, 1990, 3010, 2010),
            400,
            0.02,
            DustShape::Compact,
        )],
        vignetting: Vec::new(),
        sigma: 0.0,
        star_fraction: 0.0,
        frames: 6,
    };
    let mut small = synthetic(256, 192, &[], 0.0, &[], 0.001, 3);
    for v in &mut small.data {
        *v = *v * 30_000.0 + 300.0;
    }
    let found = frame_dust(&small, 300.0, &session, &DustOptions::default()).unwrap();
    assert!(found.seen.is_empty(), "{found:?}");
}

#[test]
fn faint_disks_without_optics_are_not_donuts() {
    // Faint plain disks (no hole) at about one session's noise: without
    // optics nothing rules a donut out, so its hole must stand clear of the
    // noise.
    let (mut disks, mut donuts) = (0, 0);
    for seed in 0..12 {
        let map = find_dust(
            &plane(&[(200.0, 150.0, 45.0, 0.0012)], 0.0012, 100 + seed),
            BIN,
            &DustOptions::default(),
        )
        .unwrap();
        for f in large(&map) {
            match f.shape {
                DustShape::Disk => disks += 1,
                DustShape::Donut => donuts += 1,
                _ => {}
            }
        }
    }
    assert_eq!(donuts, 0, "{disks} disks, {donuts} donuts");
    assert!(disks >= 6, "{disks} disks");
}

#[test]
fn clear_disks_of_one_size_do_not_hide_a_faint_one() {
    // A filter's disks all share one radius. The null rim spread is read on
    // a 7 x 7 grid of circles of that radius, many of which cross the clear
    // disks, so the null would grow with the dust it is meant to judge.
    let faint = (400.0, 300.0, 30.0, 0.0006);
    let clear: Vec<(f32, f32, f32, f32)> = [
        (70.0, 70.0),
        (180.0, 60.0),
        (300.0, 80.0),
        (430.0, 70.0),
        (80.0, 200.0),
        (200.0, 190.0),
        (320.0, 200.0),
        (90.0, 320.0),
        (220.0, 310.0),
    ]
    .iter()
    .map(|(x, y)| (*x, *y, 30.0, 0.004))
    .collect();
    let found = |disks: &[(f32, f32, f32, f32)], seed: u64| {
        let map = find_dust(&plane(disks, 0.0012, seed), BIN, &DustOptions::default()).unwrap();
        map.features.iter().any(|f| {
            f.shape.is_large() && (f.x / BIN as f32 - 400.0).hypot(f.y / BIN as f32 - 300.0) < 6.0
        })
    };
    let (mut alone, mut crowd) = (0, 0);
    for seed in 0..8 {
        alone += usize::from(found(&[faint], 200 + seed));
        let mut all = clear.clone();
        all.push(faint);
        crowd += usize::from(found(&all, 200 + seed));
    }
    assert_eq!(alone, crowd, "alone {alone}/8, among clear ones {crowd}/8");
}

#[test]
fn wide_faint_new_dust_is_not_taken_for_a_star() {
    // New dust 16 plane pixels across and 0.25% deep: wider than the 9-pixel
    // median the star veto reads, so the darker session shows no local dip
    // there and the brighter session's noise peak could pass for a star.
    for seed in 0..8 {
        let earlier = plane(&[], 0.0005, 300 + seed);
        let later = plane(&[(250.0, 200.0, 8.0, 0.0025)], 0.0005, 400 + seed);
        let changes = find_changes(&earlier, &later, BIN, &DustOptions::default()).unwrap();
        assert!(
            changes.changes.iter().any(|c| {
                (c.feature.x / BIN as f32 - 250.0).hypot(c.feature.y / BIN as f32 - 200.0) < 6.0
            }),
            "seed {seed}: {:?}",
            changes.changes
        );
    }
}

#[test]
fn a_faint_disk_survives_the_candidate_cap() {
    // A noisy 1,024 x 680 plane yields thousands of rim candidates; only the
    // best 800 are measured, and a faint disk's circle must be among them.
    let (w, h) = (1024, 680);
    let p = synthetic(w, h, &[(700.0, 300.0, 60.0, 0.0015)], 0.02, &[], 0.0012, 77);
    let (residual, _) = own_residual(&p).unwrap();
    let usable: Vec<bool> = residual.iter().map(|v| v.is_finite()).collect();
    let circles = rim_circles(&residual, &usable, w, h, &DustOptions::default());
    let rank = circles
        .iter()
        .position(|c| (c.x - 700.0).hypot(c.y - 300.0) < 6.0 && (c.radius - 60.0).abs() < 8.0);
    assert!(
        rank.is_some_and(|rank| rank < MAX_CANDIDATES),
        "rank {rank:?} of {}",
        circles.len()
    );
    let bin = 6;
    let map = find_dust(&p, bin, &DustOptions::default()).unwrap();
    assert!(
        map.features.iter().any(|f| f.shape.is_large()
            && (f.x / bin as f32 - 700.0).hypot(f.y / bin as f32 - 300.0) < 6.0),
        "{:?}",
        map.features
    );
}
