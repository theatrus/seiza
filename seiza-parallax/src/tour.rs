//! Tours planned from the catalogued objects in an image: the most
//! prominent targets, visited in a short round from the whole image and
//! back, each framed to its size with a gentle turn and pan.

use crate::overlay::{numbered, sharpless, word};
use crate::pipeline::TourStop;
use seiza::objects::{ObjectKind, PlacedObject, SkyObject};

/// The least worth (see [`targets`]) a target needs when the tour takes
/// every one worth a visit: a well-known catalogue's object, or a broad
/// region that is large, bright or named.
pub const WORTH_A_VISIT: f64 = 0.5;

/// How to plan a tour of the image's catalogued objects.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutoTour {
    /// How many targets to visit, those most worth a visit first; `None`
    /// for every one worth a visit, up to twelve.
    pub targets: Option<usize>,
    /// Seconds to stay at each target; the three most prominent stay half
    /// as long again.
    pub hold: f64,
    /// How much the camera turns and pans on the way, 0 for none, 1 for
    /// the gentle default, 2 for twice that.
    pub motion: f64,
}

impl Default for AutoTour {
    fn default() -> Self {
        Self {
            targets: None,
            hold: 1.5,
            motion: 1.0,
        }
    }
}

/// A place worth visiting.
#[derive(Clone, Debug, PartialEq)]
pub struct Target {
    pub name: String,
    /// Image pixels.
    pub x: f64,
    pub y: f64,
    /// Its catalogued extent's semi-major axis, image pixels.
    pub radius: f64,
    pub prominence: f64,
}

/// How much an object is worth a visit, about 0 to 1.5. Each stop is
/// framed to its target, so size counts for little: the well-known
/// catalogues count most (Messier, NGC, then IC, Sharpless, van den Bergh
/// and Cederblad), the broad regions of LBN and the dark nebula catalogues
/// least, and brightness and a common name add to it.
fn worth(object: &SkyObject) -> f64 {
    let name = object.name.trim();
    let catalogue = if numbered(name, "M") {
        1.0
    } else if numbered(name, "NGC") {
        0.8
    } else if numbered(name, "IC") || sharpless(name) || word(name, "vdB") || word(name, "Ced") {
        0.7
    } else if word(name, "UGC") || word(name, "Abell") {
        0.5
    } else if word(name, "LBN") {
        0.4
    } else if word(name, "LDN") || numbered(name, "B") || object.kind == ObjectKind::DarkNebula {
        0.3
    } else {
        0.4
    };
    let brightness = object
        .mag
        .map_or(0.0, |mag| ((16.0 - mag as f64) / 16.0).clamp(0.0, 1.0));
    let named = if object.common_name.is_empty() {
        0.0
    } else {
        1.0
    };
    let size = object.major_arcmin.map_or(0.0, |major| {
        ((major as f64).max(1.0).log10() / 2.0).min(1.0)
    });
    catalogue + 0.25 * brightness + 0.2 * named + 0.15 * size
}

/// The `count` catalogued objects of a `width` × `height` image most worth
/// flying to, or with no count every one worth a visit up to twelve:
/// inside the image, not so large that the whole view already shows them,
/// and not sharing a frame with one more worth a visit.
pub fn targets(
    placed: &[PlacedObject],
    (width, height): (usize, usize),
    count: Option<usize>,
) -> Vec<Target> {
    let (count, least) = match count {
        Some(count) => (count, f64::NEG_INFINITY),
        None => (12, WORTH_A_VISIT),
    };
    let (width, height) = (width as f64, height as f64);
    let short = width.min(height);
    let diagonal = width.hypot(height);
    let mut candidates: Vec<Target> = placed
        .iter()
        .filter(|placed| {
            !matches!(
                placed.object.kind,
                ObjectKind::Transient | ObjectKind::Star | ObjectKind::DoubleStar
            )
        })
        .filter(|placed| {
            let margin = 0.03 * short;
            (margin..width - margin).contains(&placed.x)
                && (margin..height - margin).contains(&placed.y)
        })
        .filter(|placed| placed.semi_major_px <= 0.35 * short)
        .map(|placed| Target {
            name: if placed.object.common_name.is_empty() {
                placed.object.name.clone()
            } else {
                format!("{} ({})", placed.object.name, placed.object.common_name)
            },
            x: placed.x,
            y: placed.y,
            radius: placed.semi_major_px,
            prominence: worth(&placed.object),
        })
        .filter(|target| target.prominence >= least)
        .collect();
    candidates.sort_by(|a, b| b.prominence.total_cmp(&a.prominence));
    let mut chosen: Vec<Target> = Vec::new();
    for candidate in candidates {
        if chosen.len() == count {
            break;
        }
        let crowded = chosen
            .iter()
            .any(|kept| (kept.x - candidate.x).hypot(kept.y - candidate.y) < 0.08 * diagonal);
        if !crowded {
            chosen.push(candidate);
        }
    }
    chosen
}

/// Visit `targets` in an order that keeps the round from the image's
/// centre and back short: nearest first, then untangled.
pub fn route(targets: &[Target], centre: (f64, f64)) -> Vec<usize> {
    let place = |index: usize| (targets[index].x, targets[index].y);
    let apart = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).hypot(a.1 - b.1);
    let mut order = Vec::with_capacity(targets.len());
    let mut left: Vec<usize> = (0..targets.len()).collect();
    let mut here = centre;
    while !left.is_empty() {
        let (at, _) = left
            .iter()
            .enumerate()
            .min_by(|a, b| apart(here, place(*a.1)).total_cmp(&apart(here, place(*b.1))))
            .expect("some left");
        let next = left.remove(at);
        here = place(next);
        order.push(next);
    }
    // Untangle: reverse any stretch whose ends swapped make the round
    // shorter, until none does.
    let point = |order: &[usize], at: isize| {
        if at < 0 || at as usize >= order.len() {
            centre
        } else {
            place(order[at as usize])
        }
    };
    let mut improved = true;
    while improved {
        improved = false;
        for i in 0..order.len() {
            for j in i + 1..order.len() {
                let (a, b) = (point(&order, i as isize - 1), point(&order, i as isize));
                let (c, d) = (point(&order, j as isize), point(&order, j as isize + 1));
                if apart(a, c) + apart(b, d) < apart(a, b) + apart(c, d) - 1e-9 {
                    order[i..=j].reverse();
                    improved = true;
                }
            }
        }
    }
    order
}

/// A stop a planner chose, with the name of the target it visits, if
/// any, so a caller can see what each stop is for before editing the list.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannedStop {
    pub stop: TourStop,
    pub name: Option<String>,
}

/// A tour of `targets` in a `width` × `height` image filmed at
/// `frame` (output pixels): from the whole image, to each target in a
/// short round, and back to the whole image.
pub fn plan(
    targets: &[Target],
    (width, height): (usize, usize),
    frame: (usize, usize),
    auto: &AutoTour,
) -> Vec<PlannedStop> {
    let (width, height) = (width as f64, height as f64);
    let centre = ((width - 1.0) / 2.0, (height - 1.0) / 2.0);
    let diagonal = width.hypot(height);
    // Image pixels per output pixel in the whole view, and the frame's
    // half sides in those pixels.
    let whole = (width / frame.0 as f64).min(height / frame.1 as f64);
    let half = (whole * frame.0 as f64 / 2.0, whole * frame.1 as f64 / 2.0);
    // Where to aim to see `place` from `near` of the way: on it, or moved
    // in from the image's edge until the frame, with room to turn a
    // little, stays inside.
    let aim = |place: (f64, f64), near: f64| {
        let axis = |at: f64, half: f64, size: f64| {
            let margin = half * near * 1.25;
            if 2.0 * margin >= size - 1.0 {
                (size - 1.0) / 2.0
            } else {
                at.clamp(margin, size - 1.0 - margin)
            }
        };
        (axis(place.0, half.0, width), axis(place.1, half.1, height))
    };
    let motion = auto.motion.max(0.0);
    let order = route(targets, centre);
    let mut prominent: Vec<usize> = (0..targets.len()).collect();
    prominent.sort_by(|&a, &b| targets[b].prominence.total_cmp(&targets[a].prominence));
    let starring = |index: usize| prominent.iter().take(3).any(|&top| top == index);
    let whole_view = |travel: f64| PlannedStop {
        stop: TourStop {
            focus: None,
            travel,
            hold: auto.hold,
            ..TourStop::default()
        },
        name: None,
    };

    let mut stops = vec![whole_view(0.0)];
    let mut here = (centre, 0.0);
    for (step, &index) in order.iter().enumerate() {
        let target = &targets[index];
        // Near enough that the target fills about three fifths of the
        // frame's shorter side, flying 0.3 to 0.85 of the way.
        let fill = (target.radius.max(0.01 * diagonal) / 0.6) / half.0.min(half.1);
        let near = fill.clamp(0.15, 0.7);
        let dolly = 1.0 - near;
        let place = aim((target.x, target.y), near);
        // Turns alternate in direction so they do not add up, and vary in
        // size; for a target near the image's edge, where room is short,
        // they and the pan are smaller.
        let room = (target.x.min(width - 1.0 - target.x) / half.0)
            .min(target.y.min(height - 1.0 - target.y) / half.1)
            / (1.25 * near);
        let edge = (room - 0.5).clamp(0.0, 1.0);
        let sign = if step % 2 == 0 { 1.0 } else { -1.0 };
        let size = 6.0 + 8.0 * ((step * 7 + 3) % 5) as f64 / 4.0;
        let rotate_deg = sign * size * motion * (0.4 + 0.6 * edge);
        let pan = (0.3 * motion * edge).min(1.0);
        let distance = (place.0 - here.0.0).hypot(place.1 - here.0.1);
        let frames = distance / (2.0 * half.1);
        let travel = (3.5 + 3.0 * frames + 1.5 * (dolly - here.1).abs()).clamp(4.0, 9.0);
        let hold = if starring(index) {
            auto.hold * 1.5
        } else {
            auto.hold
        };
        let visit = |travel: f64| PlannedStop {
            stop: TourStop {
                focus: Some(place),
                dolly,
                rotate_deg,
                pan,
                travel,
                hold,
                ..TourStop::default()
            },
            name: Some(target.name.clone()),
        };
        // A long way between two near views pulls back on the way, so the
        // viewer keeps their bearings.
        if distance > 0.35 * diagonal && here.1 > 0.5 && dolly > 0.5 {
            let back = (here.1.min(dolly) * 0.4).min(0.4);
            stops.push(PlannedStop {
                stop: TourStop {
                    focus: Some(aim(
                        ((here.0.0 + place.0) / 2.0, (here.0.1 + place.1) / 2.0),
                        1.0 - back,
                    )),
                    dolly: back,
                    travel: travel / 2.0,
                    ..TourStop::default()
                },
                name: None,
            });
            stops.push(visit(travel / 2.0 + 1.0));
        } else {
            stops.push(visit(travel));
        }
        here = (place, dolly);
    }
    let home = (here.0.0 - centre.0).hypot(here.0.1 - centre.1) / (2.0 * half.1);
    stops.push(whole_view((4.0 + 2.0 * home).clamp(4.0, 8.0)));
    stops
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(name: &str, x: f64, y: f64, radius: f64, prominence: f64) -> Target {
        Target {
            name: name.into(),
            x,
            y,
            radius,
            prominence,
        }
    }

    #[test]
    fn a_tour_goes_round_its_targets_from_the_whole_image_and_back() {
        let targets = [
            target("a", 800.0, 200.0, 60.0, 0.9),
            target("b", 200.0, 600.0, 30.0, 0.5),
            target("c", 820.0, 600.0, 120.0, 0.7),
            target("d", 250.0, 180.0, 40.0, 0.6),
        ];
        let planned = plan(&targets, (1000, 800), (1920, 1080), &AutoTour::default());
        let stops: Vec<TourStop> = planned.iter().map(|planned| planned.stop).collect();
        assert!(stops.first().unwrap().focus.is_none() && stops.last().unwrap().focus.is_none());
        // Every target once, in a round that does not cross itself: the
        // corners are taken in turn, not a then b then c then d.
        let order: Vec<&str> = planned
            .iter()
            .filter_map(|planned| planned.name.as_deref())
            .collect();
        for target in &targets {
            assert_eq!(order.iter().filter(|&&name| name == target.name).count(), 1);
        }
        let crossing = order == ["a", "b", "c", "d"] || order == ["d", "c", "b", "a"];
        assert!(!crossing, "{order:?}");
        for stop in &stops {
            assert!((0.0..1.0).contains(&stop.dolly) && (0.0..=1.0).contains(&stop.pan));
            assert!(stop.rotate_deg.abs() <= 14.0 + 1e-9);
            assert!(stop.travel >= 0.0 && stop.hold >= 0.0);
        }
        // A bigger target is framed from farther off, and a target near
        // the image's edge is aimed in from it.
        let at = |name: &str| {
            planned
                .iter()
                .find(|planned| planned.name.as_deref() == Some(name))
                .unwrap()
                .stop
        };
        assert!(at("c").dolly < at("b").dolly);
        let (x, y) = at("d").focus.unwrap();
        assert!(x >= 250.0 && y >= 180.0, "({x}, {y})");
        // Turns alternate.
        let turns: Vec<f64> = stops
            .iter()
            .filter(|stop| stop.hold > 0.0 && stop.focus.is_some())
            .map(|stop| stop.rotate_deg)
            .collect();
        for pair in turns.windows(2) {
            assert!(pair[0] * pair[1] < 0.0, "{turns:?}");
        }
    }

    fn placed(name: &str, kind: ObjectKind, x: f64, y: f64, radius: f64) -> PlacedObject {
        PlacedObject {
            object: SkyObject {
                kind,
                ra: 0.0,
                dec: 0.0,
                mag: None,
                major_arcmin: Some((radius / 15.0) as f32),
                minor_arcmin: None,
                position_angle_deg: None,
                name: name.into(),
                common_name: String::new(),
                metadata: Default::default(),
            },
            x,
            y,
            semi_major_px: radius,
            semi_minor_px: radius,
            angle_deg: None,
        }
    }

    #[test]
    fn targets_are_the_objects_worth_a_visit_one_to_a_frame() {
        let placed = [
            placed("NGC 7822", ObjectKind::HiiRegion, 500.0, 500.0, 100.0),
            // Shares NGC 7822's frame, and is worth less.
            placed("Ced 214", ObjectKind::Nebula, 540.0, 520.0, 40.0),
            placed("NGC 7762", ObjectKind::OpenCluster, 1500.0, 300.0, 50.0),
            // A small dark cloud: not worth a visit unless asked for.
            placed("LDN 1267", ObjectKind::DarkNebula, 1200.0, 1200.0, 60.0),
            // So large the whole view already shows it.
            placed("Sh2-171", ObjectKind::HiiRegion, 1000.0, 800.0, 900.0),
            // Off the image.
            placed("vdB 1", ObjectKind::Nebula, 1990.0, 900.0, 20.0),
        ];
        let names = |targets: Vec<Target>| -> Vec<String> {
            targets.into_iter().map(|target| target.name).collect()
        };
        assert_eq!(
            names(targets(&placed, (2000, 1600), None)),
            ["NGC 7822", "NGC 7762"]
        );
        assert_eq!(
            names(targets(&placed, (2000, 1600), Some(3))),
            ["NGC 7822", "NGC 7762", "LDN 1267"]
        );
    }

    #[test]
    fn no_motion_means_no_turn_or_pan() {
        let targets = [
            target("a", 300.0, 300.0, 50.0, 0.9),
            target("b", 700.0, 500.0, 50.0, 0.5),
        ];
        let still = AutoTour {
            motion: 0.0,
            ..AutoTour::default()
        };
        for planned in plan(&targets, (1000, 800), (1920, 1080), &still) {
            assert_eq!((planned.stop.rotate_deg, planned.stop.pan), (0.0, 0.0));
        }
    }
}
