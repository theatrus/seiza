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

/// Counted (not within tolerance) new changes near (x, y) plane pixels.
fn counted_new_near(changes: &DustChanges, x: f32, y: f32) -> usize {
    changes
        .changes
        .iter()
        .filter(|c| !c.within_tolerance && c.kind == ChangeKind::New)
        .filter(|c| (c.feature.x / BIN as f32 - x).hypot(c.feature.y / BIN as f32 - y) < 6.0)
        .count()
}

#[test]
fn new_window_dust_on_a_held_disks_rim_is_new() {
    // A filter disk both sessions hold, unmoved, and new compact window dust
    // 1.2% deep on its rim. The disk did not move, so it leaves no crescent;
    // the dip is new dust. Control: the same dip away from the disk.
    let earlier = plane(&[(200.0, 150.0, 40.0, 0.004)], 0.0005, 7);
    let later = plane(
        &[
            (200.0, 150.0, 40.0, 0.004),
            (240.0, 150.0, 4.0, 0.012),
            (400.0, 280.0, 4.0, 0.012),
        ],
        0.0005,
        8,
    );
    let changes = find_changes(&earlier, &later, BIN, &DustOptions::default()).unwrap();
    assert_eq!(
        counted_new_near(&changes, 400.0, 280.0),
        1,
        "{:?}",
        changes.changes
    );
    assert_eq!(
        counted_new_near(&changes, 240.0, 150.0),
        1,
        "{:?}",
        changes.changes
    );
}

#[test]
fn faint_new_window_dust_inside_a_held_disk_is_new() {
    // New compact dust 0.35% deep inside a held 0.4% disk. Control: the same
    // dip away from the disk.
    let earlier = plane(&[(200.0, 150.0, 40.0, 0.004)], 0.0005, 7);
    let later = plane(
        &[
            (200.0, 150.0, 40.0, 0.004),
            (210.0, 150.0, 4.0, 0.0035),
            (400.0, 280.0, 4.0, 0.0035),
        ],
        0.0005,
        8,
    );
    let changes = find_changes(&earlier, &later, BIN, &DustOptions::default()).unwrap();
    assert_eq!(
        counted_new_near(&changes, 400.0, 280.0),
        1,
        "{:?}",
        changes.changes
    );
    assert_eq!(
        counted_new_near(&changes, 210.0, 150.0),
        1,
        "{:?}",
        changes.changes
    );
}

#[test]
fn optics_from_parts_refuse_what_they_would_ignore() {
    assert_eq!(Optics::from_parts(None, None, None).unwrap(), None);
    let optics = Optics::from_parts(Some(4.9), Some(3.76), None)
        .unwrap()
        .unwrap();
    assert_eq!(optics.obstruction, 0.0);
    assert!(Optics::from_parts(None, None, Some(0.35)).is_err());
    assert!(Optics::from_parts(Some(10.0), None, Some(0.35)).is_err());
    assert!(Optics::from_parts(None, Some(3.76), None).is_err());
}

/// Six flats of one session, raw ADU over a 300 ADU pedestal: a filter disk
/// in each, window dust in each, and a drop of dew in the first only.
fn session_flats() -> Vec<Plane> {
    let level = 30_000.0_f32;
    (0..6)
        .map(|frame| {
            let mut p = plane(
                &[(200.0, 150.0, 40.0, 0.004), (400.0, 90.0, 4.0, 0.03)],
                0.0015,
                20 + frame,
            );
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
        .collect()
}

#[test]
fn session_dust_from_lists_keeps_what_the_flats_show() {
    let options = DustOptions::default();
    let flats = session_flats();
    let stack = metric::build_response(&flats, BIN)
        .unwrap()
        .without_pedestal(300.0);
    // Each flat's own dust, as a host stores it.
    let lists: Vec<Vec<DustFeature>> = flats
        .iter()
        .map(|flat| {
            let own = Plane {
                width: flat.width,
                height: flat.height,
                data: flat.data.iter().map(|v| v - 300.0).collect(),
            };
            find_dust(&own, BIN, &options).unwrap().features
        })
        .collect();
    let views: Vec<&[DustFeature]> = lists.iter().map(Vec::as_slice).collect();
    let (map, frames) = session_dust_from_lists(&stack, BIN, &views, &options).unwrap();
    assert_eq!(map.frames, 6);
    assert_eq!(frames.len(), 6);
    // The disk and the window dust, each in every flat; the dew is no dust
    // of the session's.
    assert_eq!(map.features.len(), 2, "{:?}", map.features);
    for feature in &map.features {
        assert_eq!(
            feature.seen_in,
            Some(SeenIn { frames: 6, of: 6 }),
            "{feature:?}"
        );
    }
    assert!(
        frames.iter().all(|frame| frame.seen.len() == 2),
        "{frames:?}"
    );
    assert_eq!(frames[0].transients.len(), 1, "{:?}", frames[0].transients);
    assert!(frames[1..].iter().all(|frame| frame.transients.is_empty()));
    // The same session read flat by flat keeps the same dust.
    let pixels = find_session_dust(&flats, BIN, 300.0, &options).unwrap();
    assert_eq!(
        pixels.features.len(),
        map.features.len(),
        "{:?}",
        pixels.features
    );
}

#[test]
fn a_feature_too_few_flats_show_is_dropped_and_a_steady_one_is_no_transient() {
    let options = DustOptions::default();
    let flats = session_flats();
    let stack = metric::build_response(&flats, BIN)
        .unwrap()
        .without_pedestal(300.0);
    let stack_map = find_dust(&stack, BIN, &options).unwrap();
    let window = *stack_map
        .features
        .iter()
        .find(|f| !f.shape.is_large())
        .expect("the window dust");
    let disk = *stack_map
        .features
        .iter()
        .find(|f| f.shape.is_large())
        .expect("the disk");
    // The window dust found in two flats of six (under 60%); a speck the
    // stack does not hold, in five of six (steady, not passing); the disk
    // in two (over 30%).
    let speck = {
        let mut f = window;
        f.x += 700.0;
        f.left += 700;
        f.right += 700;
        f
    };
    let lists: Vec<Vec<DustFeature>> = (0..6)
        .map(|frame| {
            let mut list = Vec::new();
            if frame < 2 {
                list.push(window);
                list.push(disk);
            }
            if frame < 5 {
                list.push(speck);
            }
            list
        })
        .collect();
    let views: Vec<&[DustFeature]> = lists.iter().map(Vec::as_slice).collect();
    let (map, frames) = session_dust_from_lists(&stack, BIN, &views, &options).unwrap();
    assert_eq!(map.features.len(), 1, "{:?}", map.features);
    assert!(map.features[0].shape.is_large());
    assert_eq!(map.features[0].seen_in, Some(SeenIn { frames: 2, of: 6 }));
    assert_eq!(frames[0].seen, vec![0]);
    assert!(frames[5].seen.is_empty());
    assert!(
        frames.iter().all(|frame| frame.transients.is_empty()),
        "{frames:?}"
    );
}

/// `session_dust_from_lists` as first written, checking every feature of
/// every other flat for recurrence: the reference the indexed one must
/// equal.
fn session_dust_from_lists_by_scan(
    stack: &Plane,
    bin: usize,
    frames: &[&[DustFeature]],
    options: &DustOptions,
) -> Option<(DustMap, Vec<FrameDust>)> {
    let mut map = find_dust(stack, bin, options)?;
    map.frames = frames.len();
    let t = options.move_tolerance.max(0.0);
    let overlap = |a: &DustFeature, b: &DustFeature| {
        a.left as f32 - t <= b.right as f32
            && b.left as f32 <= a.right as f32 + t
            && a.top as f32 - t <= b.bottom as f32
            && b.top as f32 <= a.bottom as f32 + t
    };
    let mut seen_by: Vec<Vec<usize>> = Vec::new();
    let mut transients: Vec<Vec<DustFeature>> = Vec::new();
    for (index, features) in frames.iter().enumerate() {
        let matches = match_features(&map.features, features, options);
        seen_by.push(
            matches
                .iter()
                .filter(|m| m.later.is_some())
                .filter_map(|m| m.earlier)
                .collect(),
        );
        let others = frames.len().saturating_sub(1) as f32;
        transients.push(
            matches
                .iter()
                .filter(|m| m.earlier.is_none())
                .filter_map(|m| m.later.map(|j| features[j]))
                .filter(|f| {
                    !map.features
                        .iter()
                        .any(|own| (own.x - f.x).hypot(own.y - f.y) < own.radius.max(f.radius))
                })
                .filter(|f| {
                    let recurs = frames
                        .iter()
                        .enumerate()
                        .filter(|(other, list)| {
                            *other != index && list.iter().any(|g| overlap(f, g))
                        })
                        .count();
                    (recurs as f32) < STEADY_SHARE * others
                })
                .collect(),
        );
    }
    let renumber = keep_persistent(&mut map, &seen_by);
    let views = seen_by
        .into_iter()
        .zip(transients)
        .map(|(seen, transients)| FrameDust {
            seen: seen.iter().filter_map(|index| renumber[*index]).collect(),
            transients,
        })
        .collect();
    Some((map, views))
}

/// A small deterministic generator for large synthetic lists.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    fn below(&mut self, n: u32) -> u32 {
        self.next() % n.max(1)
    }
}

/// A compact dip with its box `size` pixels across at `(left, top)`.
fn speck(left: u32, top: u32, size: u32) -> DustFeature {
    let half = size as f32 / 2.0;
    DustFeature {
        x: left as f32 + half,
        y: top as f32 + half,
        left,
        top,
        right: left + size,
        bottom: top + size,
        pixels: size * size,
        depth: 0.01,
        shape: DustShape::Compact,
        radius: half,
        ..Default::default()
    }
}

/// `frames` lists of about `per_frame` features each over a sensor as large
/// as the synthetic stack's: specks that recur in most flats, a little
/// displaced, among specks of noise, a wide patch, and the session's
/// own dust in every flat.
fn large_lists(
    session: &[DustFeature],
    frames: usize,
    per_frame: usize,
    seed: u64,
) -> Vec<Vec<DustFeature>> {
    let (width, height) = ((W * BIN) as u32, (H * BIN) as u32);
    let mut rng = Lcg(seed);
    let steady: Vec<DustFeature> = (0..per_frame / 10)
        .map(|_| {
            speck(
                rng.below(width - 20),
                rng.below(height - 20),
                3 + rng.below(8),
            )
        })
        .collect();
    (0..frames)
        .map(|_| {
            let mut list = session.to_vec();
            for s in &steady {
                if rng.below(10) < 7 {
                    let (dx, dy) = (rng.below(5), rng.below(5));
                    list.push(speck(
                        (s.left + dx).saturating_sub(2),
                        (s.top + dy).saturating_sub(2),
                        s.right - s.left,
                    ));
                }
            }
            while list.len() < per_frame - 1 {
                list.push(speck(
                    rng.below(width - 20),
                    rng.below(height - 20),
                    1 + rng.below(12),
                ));
            }
            let size = 50 + rng.below(250);
            list.push(speck(
                rng.below(width - size),
                rng.below(height - size),
                size,
            ));
            list
        })
        .collect()
}

/// The stack of [`session_flats`], pedestal out.
fn session_stack() -> Plane {
    metric::build_response(&session_flats(), BIN)
        .unwrap()
        .without_pedestal(300.0)
}

#[test]
fn session_dust_from_lists_needs_two_flats() {
    let options = DustOptions::default();
    let stack = session_stack();
    let session = find_dust(&stack, BIN, &options).unwrap().features;
    assert!(session_dust_from_lists(&stack, BIN, &[], &options).is_none());
    assert!(session_dust_from_lists(&stack, BIN, &[&session], &options).is_none());
    assert!(session_dust_from_lists(&stack, BIN, &[&[]], &options).is_none());
    assert!(session_dust_from_lists(&stack, BIN, &[&session, &session], &options).is_some());
}

#[test]
fn flats_that_found_nothing_count_as_flats_without_the_dust() {
    let options = DustOptions::default();
    let stack = session_stack();
    // Two flats that looked and found nothing: too few to say for the
    // disk, which stays; the window dust needs both.
    let (map, frames) = session_dust_from_lists(&stack, BIN, &[&[], &[]], &options).unwrap();
    assert_eq!(map.frames, 2);
    assert_eq!(map.features.len(), 1, "{:?}", map.features);
    assert!(map.features[0].shape.is_large());
    assert_eq!(map.features[0].seen_in, Some(SeenIn { frames: 0, of: 2 }));
    assert!(
        frames
            .iter()
            .all(|frame| frame.seen.is_empty() && frame.transients.is_empty()),
        "{frames:?}"
    );
    // Six such flats are enough to say neither is there.
    let empty: Vec<&[DustFeature]> = vec![&[]; 6];
    let (map, frames) = session_dust_from_lists(&stack, BIN, &empty, &options).unwrap();
    assert_eq!(map.frames, 6);
    assert!(map.features.is_empty(), "{:?}", map.features);
    assert_eq!(frames.len(), 6);
    assert!(
        frames
            .iter()
            .all(|frame| frame.seen.is_empty() && frame.transients.is_empty()),
        "{frames:?}"
    );
}

#[test]
fn indexed_recurrence_equals_a_full_scan_on_large_lists() {
    let stack = session_stack();
    let session = find_dust(&stack, BIN, &DustOptions::default())
        .unwrap()
        .features;
    for (seed, frames, per_frame, tolerance) in [
        (1, 8, 2_000, None),
        (2, 10, 1_000, Some(0.0)),
        (3, 5, 1_500, Some(40.0)),
    ] {
        let mut options = DustOptions::default();
        if let Some(t) = tolerance {
            options.move_tolerance = t;
        }
        let lists = large_lists(&session, frames, per_frame, seed);
        let views: Vec<&[DustFeature]> = lists.iter().map(Vec::as_slice).collect();
        let indexed = session_dust_from_lists(&stack, BIN, &views, &options).unwrap();
        let scanned = session_dust_from_lists_by_scan(&stack, BIN, &views, &options).unwrap();
        assert_eq!(indexed, scanned, "seed {seed}");
        // Both kinds of leftover occur: specks of noise that pass as
        // transients, and steady specks held back because they recur.
        let transients: usize = indexed.1.iter().map(|f| f.transients.len()).sum();
        let steady = frames * (per_frame / 10) / 2;
        eprintln!("seed {seed}: {transients} transients");
        assert!(transients > 0, "seed {seed}");
        assert!(transients < frames * per_frame - steady, "seed {seed}");
    }
}

#[test]
fn the_box_index_finds_what_a_full_scan_finds() {
    let mut rng = Lcg(7);
    let mut features: Vec<DustFeature> = (0..3_000)
        .map(|_| speck(rng.below(4_000), rng.below(3_000), rng.below(30)))
        .collect();
    // Boxes at the origin, wider than the grid takes, and reversed.
    features.push(speck(0, 0, 0));
    features.push(speck(10, 10, 3_000));
    let mut reversed = speck(500, 500, 10);
    std::mem::swap(&mut reversed.left, &mut reversed.right);
    features.push(reversed);
    let mut tall = speck(2_000, 0, 4);
    tall.bottom = u32::MAX;
    features.push(tall);
    let index = BoxIndex::new(&features);
    let mut queries: Vec<DustFeature> = (0..3_000)
        .map(|_| speck(rng.below(4_200), rng.below(3_200), rng.below(60)))
        .collect();
    queries.extend(features.iter().copied());
    queries.push(speck(u32::MAX - 10, u32::MAX - 10, 5));
    for t in [0.0, 0.5, 2.0, 64.0, 1e6, f32::INFINITY] {
        for query in &queries {
            let scanned = features.iter().any(|g| boxes_touch(query, g, t));
            assert_eq!(index.any_touching(query, t), scanned, "{query:?} t={t}");
        }
    }
}

/// The recurrence check on a long session's lists; run with
/// `cargo test --release -p seiza-calibration -- --ignored --nocapture`.
/// With the default 25 px tolerance most specks of a dense list recur; with
/// none, most do not, and each is checked against every other flat.
#[test]
#[ignore = "a benchmark"]
fn bench_session_dust_from_lists() {
    let stack = session_stack();
    let session = find_dust(&stack, BIN, &DustOptions::default())
        .unwrap()
        .features;
    for (frames, per_frame, tolerance) in [
        (30, 3_000, None),
        (30, 3_000, Some(0.0)),
        (30, 20_000, None),
        (30, 20_000, Some(0.0)),
        (30, 100_000, Some(0.0)),
    ] {
        let mut options = DustOptions::default();
        if let Some(t) = tolerance {
            options.move_tolerance = t;
        }
        let lists = large_lists(&session, frames, per_frame, 11);
        let start = std::time::Instant::now();
        let (_, views) = session_dust_from_lists(&stack, BIN, &views_of(&lists), &options).unwrap();
        let transients: usize = views.iter().map(|f| f.transients.len()).sum();
        println!(
            "{frames} flats x {per_frame} features, tolerance {} px: {:.3} s, {transients} transients",
            options.move_tolerance,
            start.elapsed().as_secs_f64()
        );
        if per_frame <= 3_000 {
            let start = std::time::Instant::now();
            session_dust_from_lists_by_scan(&stack, BIN, &views_of(&lists), &options).unwrap();
            println!("  a full scan: {:.3} s", start.elapsed().as_secs_f64());
        }
    }
}

fn views_of(lists: &[Vec<DustFeature>]) -> Vec<&[DustFeature]> {
    lists.iter().map(Vec::as_slice).collect()
}

#[test]
fn find_session_dust_drops_what_too_few_flats_show() {
    // A filter disk in every flat; window dust in four flats of six, which
    // is enough, and other window dust in two, which is not.
    let level = 30_000.0_f32;
    let flats: Vec<Plane> = (0..6)
        .map(|frame| {
            let mut disks = vec![(200.0, 150.0, 40.0, 0.004)];
            if frame < 4 {
                disks.push((400.0, 90.0, 4.0, 0.03));
            }
            if frame < 2 {
                disks.push((100.0, 300.0, 4.0, 0.03));
            }
            let mut p = plane(&disks, 0.0015, 40 + frame);
            for value in &mut p.data {
                *value = *value * level + 300.0;
            }
            p
        })
        .collect();
    let options = DustOptions::default();
    let stack = metric::build_response(&flats, BIN)
        .unwrap()
        .without_pedestal(300.0);
    // The stack holds all three.
    let held = find_dust(&stack, BIN, &options).unwrap().features;
    assert_eq!(held.len(), 3, "{held:?}");
    assert!(held.iter().any(|f| (f.x / BIN as f32 - 100.0).abs() < 3.0));
    // The rare window dust is dropped, and each kept feature counts its
    // flats exactly.
    let map = find_session_dust(&flats, BIN, 300.0, &options).unwrap();
    let kept: Vec<(DustShape, f32, f32, Option<SeenIn>)> = map
        .features
        .iter()
        .map(|f| {
            let at = |v: f32| (v / BIN as f32).round();
            (f.shape, at(f.x), at(f.y), f.seen_in)
        })
        .collect();
    assert_eq!(
        kept,
        vec![
            (
                DustShape::Disk,
                201.0,
                151.0,
                Some(SeenIn { frames: 6, of: 6 })
            ),
            (
                DustShape::Compact,
                401.0,
                90.0,
                Some(SeenIn { frames: 4, of: 6 })
            ),
        ],
        "{:?}",
        map.features
    );
    // Read from each flat's own features, the same dust is kept.
    let lists: Vec<Vec<DustFeature>> = flats
        .iter()
        .map(|flat| {
            let own = Plane {
                width: flat.width,
                height: flat.height,
                data: flat.data.iter().map(|v| v - 300.0).collect(),
            };
            find_dust(&own, BIN, &options).unwrap().features
        })
        .collect();
    let (from_lists, _) =
        session_dust_from_lists(&stack, BIN, &views_of(&lists), &options).unwrap();
    assert_eq!(from_lists.features, map.features);
}
