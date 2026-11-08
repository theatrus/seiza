//! This crate against the Python harness a model was trained with, on the
//! answers `tests/fixtures/make_fixtures.py` recorded. A model is trained on
//! exactly what the harness makes, so a difference here is a difference in
//! what a shipped model reads.

use crate::bands::NoiseBands;
use crate::bundle::{HoleGuard, InputPlane, NoiseInput, NoisePlane, Reading, SkyInput};
use crate::error::ModelError;
use crate::holes::{guarded, guarded_residual, sky_floor};
use crate::noise::{input_sigma, noise_correlation};
use crate::sky::{network_input, scales, seen, smooth_sky};
use crate::tiles::{Tiling, infer, origins, window};
use serde::Deserialize;

/// Sampled values; `null` where the plane is NaN.
type Samples = Vec<Option<f64>>;

#[derive(Deserialize)]
struct Expected {
    width: usize,
    height: usize,
    stride: usize,
    flat_block: usize,
    image_map_scale: f64,
    image_sigma: Samples,
    constant: f64,
    smooth_sky: Samples,
    background: f64,
    noise: f64,
    plane_image: Samples,
    plane_noise: Samples,
    origins: Vec<usize>,
    window: Vec<f64>,
    blended: Samples,
    blend_whole: Samples,
    blend_kept: Samples,
}

struct Fixture {
    expected: Expected,
    image: Vec<f32>,
}

fn fixture() -> Fixture {
    let directory = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let text = std::fs::read_to_string(format!("{directory}/expected.json")).unwrap();
    let expected: Expected = serde_json::from_str(&text).unwrap();
    let bytes = std::fs::read(format!("{directory}/image.f32")).unwrap();
    let image = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    Fixture { expected, image }
}

impl Fixture {
    fn reading(&self, noise_map: NoiseInput) -> Reading {
        Reading {
            noise_map,
            noise_plane: NoisePlane::Asinh,
            sky: SkyInput::Flat,
            sky_block: 32,
            flat_block: self.expected.flat_block,
            image_map_scale: self.expected.image_map_scale,
            noise_correlation: None,
            holes: None,
        }
    }

    /// Hold `plane` to `expected`'s samples, within `tolerance` of the larger
    /// of the value and one.
    fn check(&self, name: &str, plane: &[f32], expected: &Samples, tolerance: f64) {
        let (width, height, stride) = (
            self.expected.width,
            self.expected.height,
            self.expected.stride,
        );
        let mut samples = expected.iter();
        for y in (0..height).step_by(stride) {
            for x in (0..width).step_by(stride) {
                let got = f64::from(plane[y * width + x]);
                match samples.next().unwrap() {
                    None => assert!(got.is_nan(), "{name} at ({x}, {y}): {got} is not NaN"),
                    Some(want) => {
                        let scale = want.abs().max(1.0);
                        assert!(
                            (got - want).abs() <= tolerance * scale,
                            "{name} at ({x}, {y}): {got} against {want}"
                        );
                    }
                }
            }
        }
        assert!(samples.next().is_none(), "{name}: samples left over");
    }
}

#[test]
fn the_noise_map_matches_the_harness() {
    let fixture = fixture();
    let Expected { width, height, .. } = fixture.expected;
    let reading = fixture.reading(NoiseInput::Image);
    let mapped = input_sigma(&fixture.image, width, height, &reading);
    fixture.check("image_sigma", &mapped, &fixture.expected.image_sigma, 1e-5);
}

#[test]
fn the_constant_map_s_level_matches_the_harness() {
    let fixture = fixture();
    let Expected {
        width,
        height,
        constant,
        ..
    } = fixture.expected;
    let reading = fixture.reading(NoiseInput::Constant);
    let mapped = input_sigma(&fixture.image, width, height, &reading);
    let level = mapped
        .iter()
        .copied()
        .find(|value| value.is_finite())
        .unwrap();
    assert!(
        (f64::from(level) / constant - 1.0).abs() < 1e-5,
        "{level} against {constant}"
    );
}

#[test]
fn the_smooth_sky_matches_the_harness() {
    let fixture = fixture();
    let Expected {
        width,
        height,
        flat_block,
        ..
    } = fixture.expected;
    let sky = smooth_sky(&fixture.image, width, height, flat_block);
    fixture.check("smooth_sky", &sky, &fixture.expected.smooth_sky, 1e-5);
}

/// The whole path up to the network: the planes it reads and their units.
fn planes(fixture: &Fixture) -> (Vec<f32>, f32, f32) {
    let Expected { width, height, .. } = fixture.expected;
    let reading = fixture.reading(NoiseInput::Constant);
    let sigma = input_sigma(&fixture.image, width, height, &reading);
    let looked = seen(&fixture.image, width, height, &reading);
    let (background, noise) = scales(&looked, &sigma).unwrap();
    let planes = network_input(
        &looked,
        &sigma,
        background,
        noise,
        NoisePlane::Asinh,
        &[InputPlane::Image, InputPlane::Noise],
    );
    (planes, background, noise)
}

#[test]
fn the_planes_a_network_reads_match_the_harness() {
    let fixture = fixture();
    let (planes, background, noise) = planes(&fixture);
    let want = &fixture.expected;
    assert!(
        (f64::from(background) / want.background - 1.0).abs() < 1e-5,
        "{background}"
    );
    assert!(
        (f64::from(noise) / want.noise - 1.0).abs() < 1e-5,
        "{noise}"
    );
    let (first, second) = planes.split_at(want.width * want.height);
    fixture.check("image plane", first, &want.plane_image, 2e-5);
    fixture.check("noise plane", second, &want.plane_noise, 1e-5);
}

#[test]
fn tile_starts_and_windows_match_the_harness() {
    let want = fixture().expected;
    assert_eq!(origins(200, 64, 48), want.origins);
    let got = window(64, 48, 8, false, false);
    assert_eq!(got.len(), want.window.len());
    for (index, (got, want)) in got.iter().zip(&want.window).enumerate() {
        assert!(
            (f64::from(*got) - want).abs() < 1e-6,
            "{index}: {got} against {want}"
        );
    }
}

/// The harness's stand-in for a network, which looks past a pixel so that a
/// tile's edge shows unless the margins keep it out.
fn local_filter(planes: &[f32], side: usize, residual: &mut [f32]) -> Result<(), ModelError> {
    let (first, second) = planes.split_at(side * side);
    let at = |x: isize, y: isize| {
        let inside = (0..side as isize).contains(&x) && (0..side as isize).contains(&y);
        if inside {
            first[y as usize * side + x as usize]
        } else {
            0.0
        }
    };
    for y in 0..side as isize {
        for x in 0..side as isize {
            // In the harness's order, so that the sums round alike.
            let around = at(x, y - 1) + at(x, y + 1) + at(x - 1, y) + at(x + 1, y);
            let index = y as usize * side + x as usize;
            residual[index] = first[index] - 0.25 * around + 0.1 * second[index];
        }
    }
    Ok(())
}

/// The harness's image map on an image whose noise is spread over neighbours,
/// as `tests/fixtures/make_spread_fixtures.py` recorded it.
#[derive(Deserialize)]
struct SpreadExpected {
    width: usize,
    height: usize,
    stride: usize,
    sky_block: usize,
    image_map_scale: f64,
    noise_correlation: f64,
    correlation: f64,
    image_sigma: Samples,
}

#[test]
fn the_map_s_raise_for_spread_noise_matches_the_harness() {
    let directory = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let text = std::fs::read_to_string(format!("{directory}/spread.json")).unwrap();
    let want: SpreadExpected = serde_json::from_str(&text).unwrap();
    let image = read_f32(&format!("{directory}/spread.f32"));
    let (width, height) = (want.width, want.height);
    let correlation = noise_correlation(&image, width, height);
    assert!(
        (correlation / want.correlation - 1.0).abs() < 1e-9,
        "{correlation} against {}",
        want.correlation
    );
    assert!(correlation > 2.0, "the spread noise is seen as spread");

    let reading = |noise_correlation| Reading {
        noise_map: NoiseInput::Image,
        noise_plane: NoisePlane::Asinh,
        sky: SkyInput::Flat,
        sky_block: want.sky_block,
        flat_block: 256,
        image_map_scale: want.image_map_scale,
        noise_correlation,
        holes: None,
    };
    let raised = input_sigma(
        &image,
        width,
        height,
        &reading(Some(want.noise_correlation)),
    );
    let mut expected = want.image_sigma.iter();
    for y in (0..height).step_by(want.stride) {
        for x in (0..width).step_by(want.stride) {
            let got = f64::from(raised[y * width + x]);
            match expected.next().unwrap() {
                None => assert!(got.is_nan(), "({x}, {y}): {got} is not NaN"),
                Some(value) => assert!(
                    (got / value - 1.0).abs() < 1e-5,
                    "({x}, {y}): {got} against {value}"
                ),
            }
        }
    }
    assert!(expected.next().is_none());

    let measured = input_sigma(&image, width, height, &reading(None));
    let index = 100 * width + 120;
    let ratio = raised[index] / measured[index];
    let expected_ratio = (want.correlation / want.noise_correlation) as f32;
    assert!(
        (ratio - expected_ratio).abs() < 1e-5,
        "a card without it leaves the map as measured: {ratio}"
    );
}

/// Little-endian floats from a fixture file.
fn read_f32(path: &str) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// The harness's hole guard on an image holding holes, as
/// `tests/fixtures/make_hole_fixtures.py` recorded it.
#[derive(Deserialize)]
struct HoleExpected {
    width: usize,
    height: usize,
    stride: usize,
    sky_block: usize,
    image_map_scale: f64,
    noise_correlation: Option<f64>,
    guard: HoleGuard,
    level: f32,
    floor: Vec<f64>,
    mask: Vec<usize>,
    read: Vec<f64>,
    residual: Vec<f64>,
}

#[test]
fn the_hole_guard_matches_the_harness() {
    let directory = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let text = std::fs::read_to_string(format!("{directory}/holes.json")).unwrap();
    let want: HoleExpected = serde_json::from_str(&text).unwrap();
    let image = read_f32(&format!("{directory}/holes.f32"));
    let (width, height) = (want.width, want.height);
    let close = |name: &str, got: f32, want: f64, tolerance: f64| {
        let scale = want.abs().max(1.0);
        assert!(
            (f64::from(got) - want).abs() <= tolerance * scale,
            "{name}: {got} against {want}"
        );
    };

    let floor = sky_floor(&image, width, height, want.sky_block);
    let mut expected = want.floor.iter();
    for y in (0..height).step_by(want.stride) {
        for x in (0..width).step_by(want.stride) {
            close(
                "floor",
                floor[y * width + x],
                *expected.next().unwrap(),
                1e-6,
            );
        }
    }

    let reading = Reading {
        noise_map: NoiseInput::Constant,
        noise_plane: NoisePlane::Asinh,
        sky: SkyInput::Flat,
        sky_block: want.sky_block,
        flat_block: 256,
        image_map_scale: want.image_map_scale,
        noise_correlation: want.noise_correlation,
        holes: Some(want.guard),
    };
    let guarded = guarded(&image, width, height, &want.guard, &reading).unwrap();
    let masked: Vec<usize> = (0..image.len()).filter(|&i| guarded.mask[i]).collect();
    assert_eq!(
        masked, want.mask,
        "the same pixels are kept from the network"
    );
    for (&index, &value) in masked.iter().zip(&want.read) {
        close("read", guarded.read[index], value, 1e-6);
    }
    let found: Vec<f32> = guarded
        .read
        .iter()
        .map(|value| want.level - value)
        .collect();
    let residual = guarded_residual(&image, &guarded, &found, &want.guard);
    for (&index, &value) in masked.iter().zip(&want.residual) {
        close("residual", residual[index], value, 1e-5);
    }
}

#[test]
fn blending_tiles_and_bands_match_the_harness() {
    let fixture = fixture();
    let want = &fixture.expected;
    let (width, height) = (want.width, want.height);
    let (planes, _, noise) = planes(&fixture);
    let tiling = Tiling {
        tile: 64,
        overlap: 36,
        margin: 8,
        multiple: 16,
    };
    let mut runner = local_filter;
    let run = |scale: f32, runner: &mut dyn crate::tiles::TileRunner| {
        infer(
            runner,
            &planes,
            2,
            width,
            height,
            &tiling,
            scale,
            &|| false,
            &mut |_, _| {},
        )
        .unwrap()
    };
    let blended = run(1.0, &mut runner);
    fixture.check("blended", &blended, &want.blended, 5e-5);

    let residual: Vec<f32> = blended.iter().map(|value| value * noise).collect();
    let whole = NoiseBands::default().apply(&fixture.image, &residual, width, height);
    fixture.check("whole", &whole, &want.blend_whole, 5e-5);
    let bands = NoiseBands {
        high: 0.7,
        low: 0.9,
        scale_px: 5.0,
    };
    let kept = bands.apply(&fixture.image, &residual, width, height);
    fixture.check("kept", &kept, &want.blend_kept, 5e-5);
}
