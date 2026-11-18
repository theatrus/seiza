//! A sharpening run's reading of an image against the Python harness a model
//! was trained with, on the answers
//! `tests/fixtures/make_sharpen_fixtures.py` recorded. The harness measures
//! the PSF field and detects the stars through this workspace's own code,
//! so those agree to rounding; the rest is the harness's arithmetic, ported.

use crate::bundle::{ModelBundle, SharpenCard};
use crate::noise::image_map;
use crate::psf::{Field, Point, detections, field_point_map};
use crate::sharpen::rolloff_step;
use crate::sky::scales;
use serde::Deserialize;

#[derive(Deserialize)]
struct Expected {
    width: usize,
    height: usize,
    stride: usize,
    strengths: (f64, f64),
    background: f32,
    noise: f32,
    stars: usize,
    ys: Vec<f64>,
    xs: Vec<f64>,
    widths: Vec<Option<f64>>,
    stamp_side: usize,
    stamp_sums: Vec<f64>,
    stamp_peaks: Vec<f32>,
    points: Vec<[Option<f32>; 4]>,
    ratio_stars: Vec<Option<f32>>,
    ratio_rest: Vec<Option<f32>>,
    point_map: Vec<Option<f32>>,
    vectors: Vec<Vec<f32>>,
    corner: (usize, usize),
    vector_map: Vec<f32>,
    rolloff: f32,
    edged_rolled: Vec<Option<f32>>,
}

struct Fixture {
    expected: Expected,
    image: Vec<f32>,
    card: SharpenCard,
    bundle: ModelBundle,
}

fn fixture() -> Fixture {
    let directory = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/sharpen");
    let text = std::fs::read_to_string(format!("{directory}/expected.json")).unwrap();
    let bytes = std::fs::read(format!("{directory}/image.f32")).unwrap();
    let image = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    let bundle = ModelBundle::open(format!("{directory}/tiny")).unwrap();
    Fixture {
        expected: serde_json::from_str(&text).unwrap(),
        image,
        card: bundle.card().sharpen.clone().unwrap(),
        bundle,
    }
}

impl Fixture {
    fn field(&self) -> Field {
        let Expected { width, height, .. } = self.expected;
        Field::measure(
            &self.image,
            width,
            height,
            &self.card.psf,
            self.card.encoder.side,
        )
        .unwrap()
    }

    /// Hold `plane` to the harness's samples, each within `tolerance` of the
    /// larger of one and the sample.
    fn check(&self, name: &str, plane: &[f32], samples: &[Option<f32>], tolerance: f32) {
        let Expected {
            width,
            height,
            stride,
            ..
        } = self.expected;
        let mut samples = samples.iter();
        for y in (0..height).step_by(stride) {
            for x in (0..width).step_by(stride) {
                let got = plane[y * width + x];
                match samples.next().unwrap() {
                    None => assert!(got.is_nan(), "{name} at ({x}, {y}): {got} for no value"),
                    Some(want) => assert!(
                        (got - want).abs() <= tolerance * want.abs().max(1.0),
                        "{name} at ({x}, {y}): {got} against {want}"
                    ),
                }
            }
        }
        assert!(samples.next().is_none());
    }
}

#[test]
fn the_psf_field_s_nodes_stamps_and_widths_match_the_harness() {
    let fixture = fixture();
    let want = &fixture.expected;
    let field = fixture.field();
    assert_eq!(field.ys, want.ys);
    assert_eq!(field.xs, want.xs);
    assert_eq!(field.side, want.stamp_side);
    assert_eq!(field.stars, want.stars);
    assert!(!field.global, "a field that varies");
    assert_eq!(field.widths.len(), want.widths.len());
    for (node, (got, want)) in field.widths.iter().zip(&want.widths).enumerate() {
        let want = want.expect("every node has a width");
        assert!(
            (got / want - 1.0).abs() < 1e-9,
            "node {node}: {got} against {want}"
        );
    }
    for (node, stamp) in field.stamps.iter().enumerate() {
        let sum: f64 = stamp.iter().map(|&value| f64::from(value)).sum();
        let peak = stamp.iter().copied().fold(0.0, f32::max);
        assert!(
            (sum - want.stamp_sums[node]).abs() < 1e-9,
            "node {node}: {sum}"
        );
        assert_eq!(peak, want.stamp_peaks[node], "node {node}");
    }
}

#[test]
fn the_point_sources_match_the_harness() {
    let fixture = fixture();
    let want = &fixture.expected;
    let found = detections(
        &fixture.image,
        want.width,
        want.height,
        &fixture.card.points,
    );
    assert_eq!(found.len(), want.points.len());
    assert!(
        found.iter().any(|point| point.flux.is_nan()),
        "the fixture holds clipped cores the detector turns away"
    );
    for (index, (got, want)) in found.iter().zip(&want.points).enumerate() {
        let [x, y, flux, peak] = *want;
        assert_eq!(Some(got.x), x, "point {index}");
        assert_eq!(Some(got.y), y, "point {index}");
        assert_eq!(Some(got.peak), peak, "point {index}");
        match flux {
            None => assert!(got.flux.is_nan(), "point {index}"),
            Some(flux) => assert!(
                (got.flux / flux - 1.0).abs() < 1e-6,
                "point {index}: {} against {flux}",
                got.flux
            ),
        }
    }
}

#[test]
fn the_scales_and_the_ratio_maps_match_the_harness() {
    let fixture = fixture();
    let want = &fixture.expected;
    let reading = &fixture.bundle.card().reading;
    let sigma = image_map(&fixture.image, want.width, want.height, reading);
    let (background, noise) = scales(&fixture.image, &sigma).unwrap();
    assert_eq!(background, want.background);
    assert!((noise / want.noise - 1.0).abs() < 1e-5, "{noise}");

    let field = fixture.field();
    let floor = fixture.card.floor_px;
    let stars = field
        .ratio_map(want.width, want.height, want.strengths.0, floor)
        .unwrap();
    let rest = field
        .ratio_map(want.width, want.height, want.strengths.1, floor)
        .unwrap();
    fixture.check("the stars' ratio", &stars, &want.ratio_stars, 1e-6);
    fixture.check("the rest's ratio", &rest, &want.ratio_rest, 1e-6);
    assert!(
        stars.iter().any(|&ratio| ratio > 0.47) && stars.iter().any(|&ratio| ratio < 0.4601),
        "the floor holds the narrow stars on the left, and the wide ones take the share"
    );
}

#[test]
fn the_point_source_map_matches_the_harness() {
    let fixture = fixture();
    let want = &fixture.expected;
    let field = fixture.field();
    let points: Vec<Point> = want
        .points
        .iter()
        .map(|&[x, y, flux, peak]| Point {
            x: x.unwrap(),
            y: y.unwrap(),
            flux: flux.unwrap_or(f32::NAN),
            peak: peak.unwrap(),
        })
        .collect();
    let map = field_point_map(&points, &field, want.width, want.height, want.background);
    // The harness's Fourier transform leaves a floor of rounding under each
    // star, a few parts in 10⁷ of its peak.
    let highest = map.iter().copied().fold(0.0, f32::max);
    let Expected {
        width,
        height,
        stride,
        ..
    } = *want;
    let mut samples = want.point_map.iter();
    for y in (0..height).step_by(stride) {
        for x in (0..width).step_by(stride) {
            let (got, want) = (map[y * width + x], samples.next().unwrap().unwrap());
            assert!(
                (got - want).abs() <= 1e-5 * want.abs() + 1e-6 * highest,
                "({x}, {y}): {got} against {want}"
            );
        }
    }
}

#[test]
fn a_tile_s_vector_map_matches_the_harness() {
    let fixture = fixture();
    let want = &fixture.expected;
    let field = fixture.field();
    let features = fixture.card.encoder.features;
    let vectors: Vec<f32> = want.vectors.iter().flatten().copied().collect();
    assert_eq!(vectors.len(), field.stamps.len() * features);
    let tiling = fixture.bundle.card().tiling.tiling(None);
    let cells = tiling.tile / tiling.multiple;
    let mut map = vec![0.0_f32; features * cells * cells];
    let (top, left) = want.corner;
    field.sampled_features(
        &vectors,
        features,
        (left, top),
        tiling.tile,
        tiling.multiple,
        &mut map,
    );
    assert_eq!(map.len(), want.vector_map.len());
    for (index, (got, want)) in map.iter().zip(&want.vector_map).enumerate() {
        assert!(
            (got - want).abs() <= 1e-6 * want.abs().max(1.0),
            "{index}: {got} against {want}"
        );
    }
}

#[test]
fn the_roll_off_matches_the_harness() {
    let fixture = fixture();
    let want = &fixture.expected;
    let (width, height) = (want.width, want.height);
    let reading = &fixture.bundle.card().reading;
    let sigma = image_map(&fixture.image, width, height, reading);
    // The fixture's change: each pixel moved from its left neighbour by half
    // as much again.
    let mut edged = fixture.image.clone();
    for (line, source) in edged.chunks_mut(width).zip(fixture.image.chunks(width)) {
        for x in 1..width {
            line[x] += 0.5 * (source[x] - source[x - 1]);
        }
    }
    let rolled = rolloff_step(&edged, &fixture.image, width, height, want.rolloff, &sigma);
    fixture.check("the roll-off", &rolled, &want.edged_rolled, 2e-6);
    let moved = rolled
        .iter()
        .zip(&edged)
        .filter(|(rolled, edged)| (*rolled - *edged).abs() > 0.5)
        .count();
    assert!(
        moved > width * height / 4,
        "the fixture's change is held back: {moved}"
    );
}
