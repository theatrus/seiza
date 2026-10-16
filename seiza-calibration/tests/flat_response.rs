//! `flat_response` and a measured light angle through the public API:
//! degenerate input, a background that cannot be weighed, the order of the
//! candidates, the noise a response reports, and what a measured angle
//! leaves alone. `timing_at_1024` is `#[ignore]`: run it with
//! `--release -- --ignored --nocapture`.

use seiza_calibration::flat_response::{choose_response, evidence, metric, metric::Plane};
use seiza_calibration::{
    FrameSignature, LightRotation, MatchTolerances, describe_optics_mismatch, optics_match,
};

fn noise(x: usize, y: usize, seed: u64) -> f32 {
    let mut h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ seed.wrapping_mul(0x1656_67B1_9E37_79F9);
    let mut sum = 0f32;
    for _ in 0..4 {
        h ^= h >> 33;
        h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        h ^= h >> 33;
        sum += (h >> 40) as f32 / (1u64 << 24) as f32;
    }
    (sum - 2.0) * 1.732
}

fn synthetic(
    width: usize,
    height: usize,
    dust: &[(f32, f32, f32, f32)],
    noise_level: f32,
    seed: u64,
) -> Plane {
    let (cx, cy) = (width as f32 / 2.0, height as f32 / 2.0);
    let r_max = (cx * cx + cy * cy).sqrt();
    let mut data = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            let r = (dx * dx + dy * dy).sqrt() / r_max;
            let mut value = 1.0 - 0.3 * r * r;
            for (sx, sy, radius, depth) in dust {
                let d = ((x as f32 - sx).powi(2) + (y as f32 - sy).powi(2)).sqrt();
                let edge = ((radius - d) / 2.0).clamp(0.0, 1.0);
                value *= 1.0 - depth * edge;
            }
            value *= 1.0 + noise_level * noise(x, y, seed);
            data.push(value);
        }
    }
    Plane {
        width,
        height,
        data,
    }
}

fn lights(flat: &Plane, seed: u64) -> Vec<Plane> {
    (0..5)
        .map(|frame| Plane {
            width: flat.width,
            height: flat.height,
            data: flat
                .data
                .iter()
                .enumerate()
                .map(|(index, f)| {
                    let (x, y) = (index % flat.width, index / flat.width);
                    1000.0 * f * (1.0 + 0.002 * noise(x, y, seed + frame))
                })
                .collect(),
        })
        .collect()
}

fn signature() -> FrameSignature {
    let mut s = FrameSignature::default();
    s.camera = Some("ASI2600MM".into());
    s.telescope = Some("SCT".into());
    s.width = Some(6248);
    s.height = Some(4176);
    s.channels = Some(1);
    s.binning_x = Some(1);
    s.binning_y = Some(1);
    s.gain = Some(100);
    s.filter = Some("OIII".into());
    s.focal_length_mm = Some(2350.0);
    s.rotation_deg = Some(72.6);
    s.exposure_seconds = Some(300.0);
    s.camera_temp_c = Some(-10.0);
    s
}

/// A measured or unknown light angle changes the angle only.
#[test]
fn measured_or_unknown_angle_still_refuses_other_optics() {
    let light = signature();
    for basis in [
        LightRotation::Unknown,
        LightRotation::Measured { degrees: 72.6 },
    ] {
        let tolerances = MatchTolerances {
            light_rotation: basis,
            ..MatchTolerances::default()
        };
        let mut flat = signature();
        flat.filter = Some("Ha".into());
        assert!(!optics_match(&light, &flat, &tolerances));
        assert!(describe_optics_mismatch(&light, &flat, &tolerances).contains("filter"));
        let mut flat = signature();
        flat.focal_length_mm = Some(1645.0);
        assert!(!optics_match(&light, &flat, &tolerances));
        let mut flat = signature();
        flat.telescope = Some("Askar".into());
        assert!(!optics_match(&light, &flat, &tolerances));
    }
}

fn empirical_sigma(plane: &Plane) -> f32 {
    let mut v: Vec<f32> = plane.data.clone();
    v.sort_by(f32::total_cmp);
    let m = v[v.len() / 2];
    let mut d: Vec<f32> = v.iter().map(|x| (x - m).abs()).collect();
    d.sort_by(f32::total_cmp);
    1.4826 * d[d.len() / 2]
}

/// `FlatResponse::noise` is the response's own per-pixel noise, for two
/// flats as for more: a per-pixel spread of two or three samples is no
/// estimate of it.
#[test]
fn two_flat_noise_matches_the_response() {
    let mut ratios = Vec::new();
    for count in [2usize, 3, 4, 5, 8] {
        let frames: Vec<Plane> = (0..count as u64)
            .map(|seed| {
                let mut p = synthetic(400, 300, &[], 0.01, seed + 1);
                // Flatten the vignette so the plane spread is pure noise.
                let flat = synthetic(400, 300, &[], 0.0, 0);
                p.data
                    .iter_mut()
                    .zip(&flat.data)
                    .for_each(|(v, f)| *v = *v / f * 20_000.0);
                p
            })
            .collect();
        let response = metric::build_response(&frames, 1).unwrap();
        let real = empirical_sigma(&response.plane);
        let ratio = response.noise / real;
        eprintln!(
            "n={count} reported={:.5} real={real:.5} ratio={ratio:.2}",
            response.noise
        );
        ratios.push((count, ratio));
    }
    let two = ratios[0].1;
    assert!(
        (0.8..1.25).contains(&two),
        "reported/real noise for 2 flats {two:.2}; all: {ratios:?}"
    );
}

fn two_angles() -> (Plane, Plane) {
    (
        synthetic(240, 160, &[(60.0, 90.0, 14.0, 0.016)], 0.0003, 1),
        synthetic(240, 160, &[(170.0, 40.0, 14.0, 0.016)], 0.0003, 2),
    )
}

/// A background that cannot be fitted is not "alike": the two candidates
/// carry dust in different places.
#[test]
fn an_unmeasurable_background_is_not_alike() {
    let (first, second) = two_angles();
    let background = Plane::filled(240, 160, f32::NAN);
    let choice = choose_response(&background, &[first, second], |_, _| false);
    assert!(!choice.all_alike(), "{choice:?}");
}

/// The same, locally: the lights' background is missing (a masked region,
/// a bright galaxy core clipped out, an edge) exactly where the two
/// candidates differ.
#[test]
fn a_background_blind_where_candidates_differ_is_not_alike() {
    let (first, second) = two_angles();
    let mut background = evidence::background(&lights(&first, 7), 0.0, 10).unwrap();
    let w = background.width;
    for (index, v) in background.data.iter_mut().enumerate() {
        let (x, y) = ((index % w) as f32, (index / w) as f32);
        let near = |cx: f32, cy: f32| (x - cx).hypot(y - cy) < 22.0;
        if near(60.0, 90.0) || near(170.0, 40.0) {
            *v = f32::NAN;
        }
    }
    let choice = choose_response(&background, &[first, second], |_, _| false);
    let pair = choice.pair(0, 1).unwrap();
    assert!(!pair.alike(), "{choice:?}");
}

/// Planes too small to compare (`change_map` needs 16 px) are not alike.
#[test]
fn candidates_too_small_to_compare_are_not_alike() {
    let first = synthetic(12, 12, &[(3.0, 3.0, 2.0, 0.2)], 0.0, 1);
    let second = synthetic(12, 12, &[(9.0, 9.0, 2.0, 0.2)], 0.0, 2);
    let background = first.clone();
    let choice = choose_response(&background, &[first, second], |_, _| false);
    assert!(!choice.all_alike(), "{choice:?}");
}

/// The same candidates reversed choose the same.
#[test]
fn candidate_order_does_not_change_the_choice() {
    let a = synthetic(240, 160, &[(60.0, 90.0, 14.0, 0.016)], 0.0003, 1);
    let b = synthetic(240, 160, &[(170.0, 40.0, 14.0, 0.016)], 0.0003, 2);
    let c = synthetic(240, 160, &[(120.0, 130.0, 12.0, 0.02)], 0.0003, 3);
    let mut flips = Vec::new();
    for (truth, label) in [(&a, 0usize), (&b, 1), (&c, 2)] {
        for seed in 0..6u64 {
            let mut ls = lights(truth, 50 + seed);
            // Mild sky structure, so margins are tested.
            for l in &mut ls {
                let w = l.width;
                for (i, v) in l.data.iter_mut().enumerate() {
                    *v *= 1.0 + 0.03 * noise(i % w / 6, i / w / 6, 900 + seed);
                }
            }
            let bg = evidence::background(&ls, 0.0, 10).unwrap();
            let forward = choose_response(&bg, &[a.clone(), b.clone(), c.clone()], |_, _| false);
            let backward = choose_response(&bg, &[c.clone(), b.clone(), a.clone()], |_, _| false);
            let mut back: Vec<usize> = backward.winners.iter().map(|w| 2 - w).collect();
            back.sort();
            if forward.winners != back {
                flips.push((label, seed, forward.winners.clone(), back));
            }
        }
    }
    assert!(flips.is_empty(), "order changed the choice: {flips:?}");
}

/// Map helpers on degenerate input do not panic.
#[test]
fn map_at_on_an_empty_map_does_not_panic() {
    let map = Plane {
        width: 0,
        height: 0,
        data: vec![],
    };
    let _ = metric::map_at(&map, 4, 1.0, 1.0);
}

#[test]
fn apply_change_map_with_zero_width_does_not_panic() {
    let map = Plane::filled(4, 4, 1.0);
    let mut data: Vec<f32> = vec![];
    metric::apply_change_map(&mut data, 0, 1, &map, 4, true);
}

/// Timing at the real response size (run with --release --ignored).
#[test]
#[ignore]
fn timing_at_1024() {
    let (w, h) = (1024, 683);
    let cands: Vec<Plane> = (0..4)
        .map(|i| {
            synthetic(
                w,
                h,
                &[(150.0 + 200.0 * i as f32, 300.0, 40.0, 0.016)],
                0.0003,
                i as u64,
            )
        })
        .collect();
    let t = std::time::Instant::now();
    let frames: Vec<Plane> = (0..3)
        .map(|s| synthetic(w, h, &[], 0.001, 10 + s))
        .collect();
    let _ = metric::build_response(&frames, 6).unwrap();
    eprintln!("build_response x3: {:?}", t.elapsed());
    let t = std::time::Instant::now();
    let _ = metric::compare(&cands[0], &cands[1], 6).unwrap();
    eprintln!("compare: {:?}", t.elapsed());
    let bg = evidence::background(&lights(&cands[0], 5), 0.0, 6).unwrap();
    let t = std::time::Instant::now();
    let choice = choose_response(&bg, &cands, |_, _| false);
    eprintln!(
        "choose_response 4 candidates (6 pairs): {:?} -> {:?}",
        t.elapsed(),
        choice.winners
    );
}

fn blind_near(background: &mut Plane, spots: &[(f32, f32)], radius: f32) {
    let w = background.width;
    for (index, v) in background.data.iter_mut().enumerate() {
        let (x, y) = ((index % w) as f32, (index / w) as f32);
        if spots.iter().any(|(cx, cy)| (x - cx).hypot(y - cy) < radius) {
            *v = f32::NAN;
        }
    }
}

/// A host may take several winners as alike and pick among them. Three
/// angles; lights shot through `truth`; the background is blind (masked
/// target, NaN) over truth's and wrong's shadows only. truth against wrong
/// cannot be fitted, so `wrong` must not win by beating `other` on the one
/// shadow it can see.
#[test]
fn an_unfitted_pair_does_not_make_a_co_winner() {
    let truth = synthetic(240, 160, &[(60.0, 90.0, 14.0, 0.016)], 0.0003, 1);
    let wrong = synthetic(240, 160, &[(170.0, 40.0, 14.0, 0.016)], 0.0003, 2);
    let other = synthetic(240, 160, &[(120.0, 130.0, 12.0, 0.02)], 0.0003, 3);
    let mut background = evidence::background(&lights(&truth, 7), 0.0, 10).unwrap();
    blind_near(&mut background, &[(60.0, 90.0), (170.0, 40.0)], 22.0);
    let choice = choose_response(&background, &[truth, wrong, other], |_, _| false);
    assert!(
        !choice.winners.contains(&1),
        "a candidate never weighed against the truth wins: {choice:?}"
    );
}

/// A dark globule in the sky exactly where one wrong candidate's shadow
/// sits, while the truth and the wrong candidate each carry their own
/// shadow: the Birge inflation across two shadows keeps this undecided
/// rather than choosing `wrong`.
#[test]
fn a_dark_globule_on_a_wrong_shadow_does_not_choose_it() {
    let truth = synthetic(240, 160, &[(60.0, 90.0, 14.0, 0.016)], 0.0003, 1);
    let wrong = synthetic(240, 160, &[(170.0, 40.0, 14.0, 0.016)], 0.0003, 2);
    let other = synthetic(240, 160, &[(120.0, 130.0, 12.0, 0.02)], 0.0003, 3);
    let mut bad = Vec::new();
    for depth in [0.008f32, 0.016, 0.03] {
        // The globule darkens the sky where `wrong` would cast its shadow.
        let globule = synthetic(240, 160, &[(170.0, 40.0, 14.0, depth)], 0.0, 0);
        let vignette = synthetic(240, 160, &[], 0.0, 0);
        let mut ls = lights(&truth, 21);
        for l in &mut ls {
            for ((v, g), f) in l.data.iter_mut().zip(&globule.data).zip(&vignette.data) {
                *v *= g / f;
            }
        }
        let background = evidence::background(&ls, 0.0, 10).unwrap();
        for candidates in [
            vec![truth.clone(), wrong.clone()],
            vec![truth.clone(), wrong.clone(), other.clone()],
        ] {
            let choice = choose_response(&background, &candidates, |_, _| false);
            if choice.winners.contains(&1) {
                bad.push((depth, candidates.len(), choice.winners.clone()));
            }
        }
    }
    assert!(bad.is_empty(), "a globule chose the wrong flats: {bad:?}");
}

/// Nebula-like structure at several scales and strengths: the choice is
/// the truth or nothing, never another candidate.
#[test]
fn structured_skies_choose_the_truth_or_nothing() {
    let cands = [
        synthetic(240, 160, &[(60.0, 90.0, 14.0, 0.016)], 0.0003, 1),
        synthetic(240, 160, &[(170.0, 40.0, 14.0, 0.016)], 0.0003, 2),
        synthetic(240, 160, &[(120.0, 130.0, 12.0, 0.02)], 0.0003, 3),
    ];
    let mut wrong = Vec::new();
    let mut decided = 0;
    for (truth, flat) in cands.iter().enumerate() {
        for (scale, amp) in [(3usize, 0.02f32), (6, 0.05), (12, 0.1), (20, 0.2)] {
            for seed in 0..3u64 {
                let mut ls = lights(flat, 300 + seed);
                for l in &mut ls {
                    let w = l.width;
                    for (i, v) in l.data.iter_mut().enumerate() {
                        *v *= 1.0 + amp * noise(i % w / scale, i / w / scale, 700 + seed);
                    }
                }
                let bg = evidence::background(&ls, 0.0, 10).unwrap();
                let choice = choose_response(&bg, &cands, |_, _| false);
                if !choice.winners.is_empty() {
                    decided += 1;
                }
                if !(choice.winners.is_empty() || choice.winners == vec![truth]) {
                    wrong.push((truth, scale, amp, seed, choice.winners.clone()));
                }
            }
        }
    }
    eprintln!("decided {decided} of 36");
    assert!(wrong.is_empty(), "wrong choices: {wrong:?}");
}

/// Public entry points on degenerate input (empty, tiny, all-NaN,
/// infinite) do not panic.
#[test]
fn degenerate_planes_do_not_panic() {
    let shapes = [
        (0usize, 0usize),
        (1, 1),
        (3, 2),
        (15, 15),
        (16, 16),
        (40, 17),
    ];
    for (w, h) in shapes {
        for fill in [f32::NAN, f32::INFINITY, 0.0, 1.0, -1.0] {
            let p = Plane::filled(w, h, fill);
            let frames = vec![p.clone(), p.clone(), p.clone()];
            let _ = metric::build_response(&frames, 1);
            let _ = metric::compare(&p, &p, 1);
            let _ = metric::change_map(&p, &p);
            let _ = metric::star_mask(&p);
            let _ = evidence::background(&frames, 0.0, 1);
            let _ = choose_response(&p, &[p.clone(), p.clone()], |_, _| false);
            let _ = metric::decode_plane(&metric::encode_plane(&p));
            let _ = metric::bin_frame(&p.data, w, h, 1, 4);
        }
    }
}

/// Same inputs, 1 thread vs many: bitwise equal responses, comparisons and
/// choices.
#[test]
fn results_do_not_depend_on_the_thread_count() {
    let run = || {
        let frames: Vec<Plane> = (0..3)
            .map(|s| {
                let mut p = synthetic(300, 200, &[(80.0, 60.0, 12.0, 0.02)], 0.002, 40 + s);
                p.data.iter_mut().for_each(|v| *v *= 20_000.0);
                p
            })
            .collect();
        let r = metric::build_response(&frames, 4).unwrap();
        let (a, b) = two_angles();
        let c = metric::compare(&a, &b, 10).unwrap();
        let bg = evidence::background(&lights(&a, 3), 0.0, 10).unwrap();
        let choice = choose_response(&bg, &[a, b], |_, _| false);
        (
            r.plane.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            r.noise.to_bits(),
            format!("{c:?}"),
            format!("{choice:?}"),
        )
    };
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(run);
    let many = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap()
        .install(run);
    assert!(one == many, "thread count changed a result");
}
