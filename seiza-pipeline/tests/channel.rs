//! A channel stacked as partial stacks, one per session, and combined, is
//! the stack of every light; run again, it does only what is missing.

use seiza_fits::{F32ImageData, write_f32_image};
use seiza_pipeline::{
    ChannelPlan, ChannelProgress, ChannelReference, Estimator, PartialStack, Reintegration,
    Session, SessionOptions, StackKind, Store, run_channel,
};
use seiza_stacking::{
    FileCalibrationSession, FrameDisposition, FrameWeighting, LiveStacker, NormalizationMode,
    RejectionMode, StackOptions,
};
use std::path::PathBuf;

const WIDTH: usize = 160;
const HEIGHT: usize = 128;

/// Lights of one star field, a few pixels apart, `per_session` to each of
/// `sessions` sessions with a bias of their own, which their lights carry.
/// Each light has noise of its own, so frame weights differ.
struct Night {
    _directory: tempfile::TempDir,
    root: PathBuf,
    sessions: Vec<Session>,
}

impl Night {
    fn new(sessions: usize, per_session: usize) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().to_path_buf();
        let mut seed = 0x5eed_u64;
        let mut random = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as f64 / (1u64 << 31) as f64
        };
        let margin = 12;
        let (sky_width, sky_height) = (WIDTH + 2 * margin, HEIGHT + 2 * margin);
        let mut sky = vec![100.0_f32; sky_width * sky_height];
        for _ in 0..90 {
            let x = 6.0 + random() * (sky_width - 12) as f64;
            let y = 6.0 + random() * (sky_height - 12) as f64;
            let peak = 200.0 + 3000.0 * random().powi(3);
            for dy in -5_isize..=5 {
                for dx in -5_isize..=5 {
                    let px = x.floor() as isize + dx;
                    let py = y.floor() as isize + dy;
                    let r2 = (px as f64 - x).powi(2) + (py as f64 - y).powi(2);
                    sky[py as usize * sky_width + px as usize] += (peak * (-r2 / 3.2).exp()) as f32;
                }
            }
        }
        let mut plan = Vec::new();
        for session in 0..sessions {
            let bias_level = 50.0 * (session + 1) as f32;
            let bias = root.join(format!("bias-{session}.fits"));
            write_f32_image(
                &bias,
                WIDTH,
                HEIGHT,
                F32ImageData::Mono(&vec![bias_level; WIDTH * HEIGHT]),
                &[],
            )
            .unwrap();
            let mut lights = Vec::new();
            for index in 0..per_session {
                let frame = session * per_session + index;
                let (ox, oy) = (margin - 4 + frame % 9, margin - 3 + (frame * 5) % 7);
                let sigma = 3.0 + (frame % 3) as f64 * 2.0;
                let mut data = Vec::with_capacity(WIDTH * HEIGHT);
                for y in 0..HEIGHT {
                    for x in 0..WIDTH {
                        let normal = (0..4).map(|_| random()).sum::<f64>() - 2.0;
                        data.push(
                            sky[(y + oy) * sky_width + x + ox]
                                + bias_level
                                + (normal * sigma * 1.7) as f32,
                        );
                    }
                }
                let path = root.join(format!("light-{session}-{index}.fits"));
                write_f32_image(&path, WIDTH, HEIGHT, F32ImageData::Mono(&data), &[]).unwrap();
                lights.push(path);
            }
            plan.push(Session {
                lights,
                calibration: FileCalibrationSession {
                    bias: Some(bias),
                    ..Default::default()
                },
            });
        }
        Self {
            _directory: directory,
            root,
            sessions: plan,
        }
    }

    fn reference(&self) -> ChannelReference {
        ChannelReference {
            image: self.sessions[0].lights[0].clone(),
            light: self.sessions[0].lights[0].clone(),
            calibration: self.sessions[0].calibration.clone(),
        }
    }

    fn plan(&self, sessions: &[Session], options: SessionOptions) -> ChannelPlan {
        ChannelPlan {
            reference: self.reference(),
            sessions: sessions.to_vec(),
            options,
        }
    }

    /// One stack of every light of `sessions`, on the same reference.
    fn whole(&self, sessions: &[Session], options: &StackOptions) -> seiza_stacking::StackSnapshot {
        let paths = sessions
            .iter()
            .flat_map(|session| session.lights.iter().cloned())
            .collect::<Vec<_>>();
        let recipes = sessions
            .iter()
            .map(|session| session.calibration.clone())
            .collect::<Vec<_>>();
        let assignments = sessions
            .iter()
            .enumerate()
            .flat_map(|(index, session)| session.lights.iter().map(move |_| index))
            .collect::<Vec<_>>();
        let mut stack = LiveStacker::open_fits_sessions_with_reference_image(
            &paths[0],
            &paths,
            &recipes,
            &assignments,
            0,
            options.clone(),
        )
        .unwrap();
        for path in &paths {
            assert!(matches!(
                stack.push_fits(path).unwrap(),
                FrameDisposition::Accepted(_)
            ));
        }
        stack.into_snapshot().unwrap()
    }
}

fn options() -> SessionOptions {
    SessionOptions {
        stack: StackOptions {
            normalization: NormalizationMode::None,
            rejection: RejectionMode::None,
            weighting: FrameWeighting::inverse_noise_variance(),
            ..StackOptions::default()
        },
        contributor: Some("tester".into()),
        ..SessionOptions::default()
    }
}

fn run(
    plan: &ChannelPlan,
    store: &Store,
) -> (seiza_pipeline::ChannelOutcome, Vec<ChannelProgress>) {
    let mut events = Vec::new();
    let outcome = run_channel(plan, store, |event| events.push(event)).unwrap();
    (outcome, events)
}

fn stacked(events: &[ChannelProgress]) -> Vec<usize> {
    events
        .iter()
        .filter_map(|event| match event {
            ChannelProgress::Stacking { session, .. } => Some(*session),
            _ => None,
        })
        .collect()
}

fn combined(events: &[ChannelProgress]) -> bool {
    events
        .iter()
        .any(|event| matches!(event, ChannelProgress::Combining { .. }))
}

/// The partition invariant: sessions stacked apart and combined by their
/// weights, without rejection, match one weighted stack of every light on
/// the same grid, to `f32` rounding.
#[test]
fn partial_stacks_combine_into_the_stack_of_every_light() {
    let night = Night::new(3, 3);
    let store = Store::new(night.root.join("store"));
    let plan = night.plan(&night.sessions, options());
    let (outcome, events) = run(&plan, &store);
    assert_eq!(stacked(&events), [0, 1, 2]);
    assert!(!outcome.reused);
    let tile = &outcome.stack;
    assert_eq!(tile.identity.kind, StackKind::Tile);
    assert_eq!(tile.identity.input_count, 3);
    assert_eq!(tile.identity.combined, 9);
    assert_eq!(tile.identity.estimator, Estimator::Additive);

    let whole = night.whole(&night.sessions, &plan.options.stack);
    assert_eq!(tile.coverage, whole.coverage);
    let whole_weight = whole.weight_sum.as_ref().unwrap();
    let mut compared = 0;
    for (index, (&expected, &weight)) in whole.image.data.iter().zip(whole_weight).enumerate() {
        if whole.coverage[index] == 0 {
            assert!(tile.mean.data[index].is_nan());
            continue;
        }
        let (expected, actual) = (f64::from(expected), f64::from(tile.mean.data[index]));
        assert!(
            (actual - expected).abs() <= 2e-6 * expected.abs().max(1.0),
            "sample {index}: {actual} vs {expected}"
        );
        let weight = f64::from(weight);
        assert!((f64::from(tile.weight[index]) - weight).abs() <= 1e-5 * weight);
        compared += 1;
    }
    assert!(compared > whole.image.data.len() * 3 / 4);

    // Each session's partial stack is a file that describes itself.
    for (session, part) in outcome.sessions.iter().enumerate() {
        let identity = PartialStack::read_identity(part.path.as_deref().unwrap()).unwrap();
        assert_eq!(identity.step_key, part.key);
        assert_eq!(identity.kind, StackKind::Partial);
        assert_eq!(identity.input_count, 3);
        assert_eq!(identity.contributor.as_deref(), Some("tester"));
        assert!(part.rejected.is_empty(), "session {session}");
    }
    // The tile in the store is the tile returned.
    let stored = PartialStack::open(&outcome.path).unwrap();
    assert_eq!(stored.identity, tile.identity);
}

/// Run again, nothing is stacked or combined; a session added is the only
/// one stacked; the order of the sessions changes nothing.
#[test]
fn a_channel_run_again_does_only_what_is_missing() {
    let night = Night::new(3, 2);
    let store = Store::new(night.root.join("store"));
    let first = night.plan(&night.sessions[..2], options());
    let (two, events) = run(&first, &store);
    assert_eq!(stacked(&events), [0, 1]);
    let bytes = std::fs::read(&two.path).unwrap();

    let (again, events) = run(&first, &store);
    assert!(stacked(&events).is_empty() && !combined(&events));
    assert!(again.reused && again.sessions.iter().all(|session| session.reused));
    assert_eq!(again.path, two.path);
    assert_eq!(std::fs::read(&again.path).unwrap(), bytes);

    let all = night.plan(&night.sessions, options());
    let (three, events) = run(&all, &store);
    assert_eq!(stacked(&events), [2]);
    assert!(combined(&events));
    assert_ne!(three.stack.identity.step_key, two.stack.identity.step_key);
    assert_eq!(three.stack.identity.combined, 6);

    let mut reordered = night.sessions.clone();
    reordered.reverse();
    let (same, events) = run(&night.plan(&reordered, options()), &store);
    assert!(stacked(&events).is_empty() && !combined(&events));
    assert_eq!(same.stack.identity.step_key, three.stack.identity.step_key);

    // Other options are other steps.
    let clipped = SessionOptions {
        reintegration: Some(Reintegration::default()),
        ..options()
    };
    let (other, events) = run(&night.plan(&night.sessions, clipped), &store);
    assert_eq!(stacked(&events), [0, 1, 2]);
    assert_eq!(other.stack.identity.estimator, Estimator::Clipped);
    assert_ne!(other.stack.identity.step_key, three.stack.identity.step_key);
}

/// Reintegrated within its session, a light's satellite trail is rejected
/// before the session's partial stack leaves the machine.
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// A session that gains lights is extended from its stored context, only
/// the new lights pushed, and its partial stack is, to the bit, the one
/// stacking every light gives; reintegrated, it reintegrates every light.
/// Without a context, it is stacked again.
#[test]
fn a_session_that_gains_lights_is_extended_from_its_context() {
    let night = Night::new(2, 4);
    let mut early = night.sessions.clone();
    early[1].lights.truncate(2);
    for reintegration in [None, Some(Reintegration::default())] {
        let clipped = reintegration.is_some();
        let kept = SessionOptions {
            keep_context: true,
            reintegration,
            ..options()
        };
        let store = Store::new(night.root.join(format!("store-{clipped}")));
        let (first, _) = run(&night.plan(&early, kept.clone()), &store);
        assert!(
            first
                .sessions
                .iter()
                .all(|session| session.extended_from.is_none())
        );
        let (later, events) = run(&night.plan(&night.sessions, kept.clone()), &store);
        assert_eq!(stacked(&events), [1]);
        assert!(later.sessions[0].reused);
        assert_eq!(later.sessions[1].extended_from, Some(first.sessions[1].key));

        let fresh = Store::new(night.root.join(format!("fresh-{clipped}")));
        let (fresh, _) = run(&night.plan(&night.sessions, kept), &fresh);
        assert!(fresh.sessions[1].extended_from.is_none());
        assert_eq!(later.stack.identity.step_key, fresh.stack.identity.step_key);
        let extended = PartialStack::open(later.sessions[1].path.as_deref().unwrap()).unwrap();
        let whole = PartialStack::open(fresh.sessions[1].path.as_deref().unwrap()).unwrap();
        assert_eq!(extended.identity.step_key, whole.identity.step_key);
        assert_eq!(extended.identity.combined, 4);
        assert!(extended.identity.context.is_some());
        assert_eq!(
            bits(&extended.mean.data),
            bits(&whole.mean.data),
            "{clipped}"
        );
        assert_eq!(bits(&extended.weight), bits(&whole.weight));
        assert_eq!(extended.coverage, whole.coverage);
        assert_eq!(extended.rejected, whole.rejected);
        assert_eq!(
            extended.variance.as_deref().map(bits),
            whole.variance.as_deref().map(bits)
        );
    }

    let store = Store::new(night.root.join("plain"));
    run(&night.plan(&early, options()), &store);
    let (later, events) = run(&night.plan(&night.sessions, options()), &store);
    assert_eq!(stacked(&events), [1]);
    assert!(later.sessions[1].extended_from.is_none());
}

/// A stored context that cannot be extended here, its masters moved since
/// it was saved, as on another machine, is passed over: the session is
/// stacked afresh from the masters where they are now, to the same stack.
#[test]
fn a_context_that_cannot_be_extended_is_stacked_afresh() {
    let night = Night::new(2, 4);
    let kept = SessionOptions {
        keep_context: true,
        ..options()
    };
    let mut early = night.sessions.clone();
    early[1].lights.truncate(2);
    let store = Store::new(night.root.join("store"));
    run(&night.plan(&early, kept.clone()), &store);

    let mut moved = night.sessions.clone();
    let old = moved[1].calibration.bias.clone().unwrap();
    let new = night.root.join("moved-bias.fits");
    std::fs::rename(&old, &new).unwrap();
    moved[1].calibration.bias = Some(new);
    let (later, events) = run(&night.plan(&moved, kept.clone()), &store);
    assert_eq!(stacked(&events), [1]);
    assert!(later.sessions[1].extended_from.is_none());

    let fresh = Store::new(night.root.join("fresh"));
    let (whole, _) = run(&night.plan(&moved, kept), &fresh);
    assert_eq!(later.sessions[1].key, whole.sessions[1].key);
    let a = PartialStack::open(later.sessions[1].path.as_deref().unwrap()).unwrap();
    let b = PartialStack::open(whole.sessions[1].path.as_deref().unwrap()).unwrap();
    assert_eq!(bits(&a.mean.data), bits(&b.mean.data));
}

/// What cannot be combined is refused: stacks made against another
/// reference light, a plan that drizzles Bayer frames, and a contributor
/// handle a FITS card cannot hold, before any frame is stacked.
#[test]
fn what_cannot_be_combined_is_refused() {
    let night = Night::new(2, 3);
    let store = Store::new(night.root.join("store"));
    let (first, _) = run(&night.plan(&night.sessions[..1], options()), &store);
    let mut other = night.plan(&night.sessions[1..], options());
    other.reference.light = night.sessions[0].lights[1].clone();
    let (second, _) = run(&other, &store);
    assert!(matches!(
        PartialStack::combine_files(&[&first.path, &second.path]),
        Err(seiza_pipeline::Error::Incompatible(_))
    ));

    // Another weighting gives weights on another scale; an impulse filter
    // only changes which samples are kept.
    let mut equal = options();
    equal.stack.weighting = FrameWeighting::Equal;
    let (reweighted, _) = run(&night.plan(&night.sessions[1..], equal), &store);
    assert!(matches!(
        PartialStack::combine_files(&[&first.path, &reweighted.path]),
        Err(seiza_pipeline::Error::Incompatible(_))
    ));
    let mut filtered = options();
    filtered.stack.cosmetic = Some(Default::default());
    let (cleaned, _) = run(&night.plan(&night.sessions[1..], filtered), &store);
    let options_of = |outcome: &seiza_pipeline::ChannelOutcome| {
        PartialStack::read_identity(outcome.sessions[0].path.as_deref().unwrap())
            .unwrap()
            .options
    };
    assert_ne!(options_of(&cleaned), options_of(&first));
    assert!(PartialStack::combine_files(&[&first.path, &cleaned.path]).is_ok());

    let mut drizzled = options();
    drizzled.stack.cfa_integration = seiza_stacking::CfaIntegration::BayerDrizzle;
    let mut handle = options();
    handle.contributor = Some("Zoë".into());
    for options in [drizzled, handle] {
        let error = run_channel(&night.plan(&night.sessions, options), &store, |_| {}).unwrap_err();
        assert!(
            matches!(error, seiza_pipeline::Error::Invalid(_)),
            "{error}"
        );
    }
}

#[test]
fn a_transient_is_rejected_within_its_session() {
    let night = Night::new(2, 5);
    let trail = &night.sessions[1].lights[2];
    let mut frame = seiza_stacking::FitsFrame::open(trail).unwrap();
    let row = HEIGHT / 2;
    for x in 0..WIDTH {
        frame.image.data[row * WIDTH + x] += 5000.0;
    }
    write_f32_image(
        trail,
        WIDTH,
        HEIGHT,
        F32ImageData::Mono(&frame.image.data),
        &[],
    )
    .unwrap();
    let options = SessionOptions {
        reintegration: Some(Reintegration::default()),
        ..options()
    };
    let store = Store::new(night.root.join("store"));
    let (outcome, _) = run(&night.plan(&night.sessions, options), &store);
    let part = PartialStack::open(outcome.sessions[1].path.as_deref().unwrap()).unwrap();
    let rejected = part.rejected.as_ref().unwrap();
    let rejected_on_row = (0..WIDTH)
        .filter(|&x| {
            // The trail lands within a row or two of where it was drawn.
            (row.saturating_sub(4)..row + 4).any(|y| rejected[y * WIDTH + x] > 0)
        })
        .count();
    assert!(rejected_on_row > WIDTH / 2, "{rejected_on_row}");
    let tile = &outcome.stack;
    let sky = tile
        .mean
        .data
        .iter()
        .filter(|value| value.is_finite())
        .map(|&value| f64::from(value))
        .fold(f64::INFINITY, f64::min);
    let brightest_on_row = (row.saturating_sub(4)..row + 4)
        .flat_map(|y| (8..WIDTH - 8).map(move |x| y * WIDTH + x))
        .map(|index| f64::from(tile.mean.data[index]))
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let median = {
        let mut values = brightest_on_row.clone();
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    };
    assert!(
        median < sky + 500.0,
        "trail left in the tile: {median} vs sky {sky}"
    );
}
