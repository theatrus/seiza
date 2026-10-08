//! Detection and solving run in the Rayon pool their caller installs.
//!
//! Every entry point here runs inside a one-thread and a two-thread pool.
//! The global pool must never start: Rayon work that left the caller's
//! pool would start it. That can be checked once per process, and each file
//! under `tests/` runs as a process of its own, so this is one test.

use rayon::ThreadPoolBuilder;
use seiza::blind::{BlindIndex, BlindParams, solve_blind};
use seiza::catalog::{CatalogStar, TileCatalog, TileSetBuilder, angular_separation_deg};
use seiza::solve::{Solution, SolveHint, solve};
use seiza::{DetectBackend, DetectConfig, DetectedStar, Wcs, detect_stars, detect_stars_luma_f32};
use std::path::Path;

const DIMENSIONS: (u32, u32) = (4000, 3000);
const CENTER: (f64, f64) = (150.5, 33.2);

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn truth() -> Wcs {
    Wcs::from_center_scale_rotation(CENTER, (2000.0, 1500.0), 2.0, 47.3, false)
}

/// A patch of sky around the field, and what a camera at [`truth`] sees of
/// it, brightest first.
fn scene() -> (Vec<CatalogStar>, Vec<DetectedStar>) {
    let truth = truth();
    let mut rng = Lcg(7);
    let mut catalog = Vec::new();
    let mut detected = Vec::new();
    for _ in 0..3000 {
        let ra = CENTER.0 + (rng.next() - 0.5) * 12.0;
        let dec = CENTER.1 + (rng.next() - 0.5) * 12.0;
        let mag = (4.0 + rng.next() * 8.0) as f32;
        catalog.push(CatalogStar { ra, dec, mag });
        if let Some((x, y)) = truth.world_to_pixel(ra, dec)
            && x > 0.0
            && y > 0.0
            && x < DIMENSIONS.0 as f64
            && y < DIMENSIONS.1 as f64
            && rng.next() >= 0.3
        {
            detected.push(DetectedStar {
                x: x + (rng.next() - 0.5) * 0.6,
                y: y + (rng.next() - 0.5) * 0.6,
                flux: 10f64.powf(-0.4 * mag as f64) * 1e6,
                peak: 0.5,
                area: 20,
            });
        }
    }
    detected.sort_by(|a, b| b.flux.total_cmp(&a.flux));
    (catalog, detected)
}

/// Gaussian stars on a noisy sky, as linear samples.
fn star_image(width: u32, height: u32) -> Vec<f32> {
    let mut rng = Lcg(11);
    let stars = (0..60)
        .map(|_| {
            (
                8.0 + rng.next() * (width as f64 - 16.0),
                8.0 + rng.next() * (height as f64 - 16.0),
                2000.0 + rng.next() * 20000.0,
                1.2 + rng.next() * 1.2,
            )
        })
        .collect::<Vec<_>>();
    (0..width * height)
        .map(|index| {
            let (x, y) = ((index % width) as f64, (index / width) as f64);
            let mut value = 1000.0 + ((index as u64 * 2654435761) % 97) as f64;
            for &(cx, cy, amplitude, sigma) in &stars {
                let d2 = (x - cx).powi(2) + (y - cy).powi(2);
                if d2 < 36.0 * sigma * sigma {
                    value += amplitude * (-d2 / (2.0 * sigma * sigma)).exp();
                }
            }
            value.min(65535.0) as f32
        })
        .collect()
}

fn star_bits(stars: &[DetectedStar]) -> Vec<u64> {
    stars
        .iter()
        .flat_map(|star| {
            [
                star.x.to_bits(),
                star.y.to_bits(),
                star.flux.to_bits(),
                u64::from(star.peak.to_bits()),
                u64::from(star.area),
            ]
        })
        .collect()
}

fn solution_bits(solution: &Solution) -> Vec<u64> {
    let wcs = &solution.wcs;
    let mut out = vec![
        wcs.crval.0.to_bits(),
        wcs.crval.1.to_bits(),
        wcs.crpix.0.to_bits(),
        wcs.crpix.1.to_bits(),
    ];
    out.extend(wcs.cd.iter().flatten().map(|value| value.to_bits()));
    out.push(solution.matched_stars as u64);
    out.push(solution.rms_arcsec.to_bits());
    out
}

/// Where a solution puts the image centre, and how far that is from the
/// truth, in arcseconds.
fn centre_error_arcsec(solution: &Solution) -> f64 {
    let (x, y) = (DIMENSIONS.0 as f64 / 2.0, DIMENSIONS.1 as f64 / 2.0);
    let (ra, dec) = solution.wcs.pixel_to_world(x, y);
    let (true_ra, true_dec) = truth().pixel_to_world(x, y);
    angular_separation_deg(ra, dec, true_ra, true_dec) * 3600.0
}

struct Run {
    /// Results that must not depend on the pool's size.
    exact: Vec<(&'static str, Vec<u64>)>,
    /// The blind solution, which may come from any of the hypotheses
    /// verified together, so it is checked against the truth instead.
    blind: Solution,
}

fn run(pool: &rayon::ThreadPool, directory: &Path, threads: usize) -> Run {
    pool.install(|| {
        let (width, height) = (640, 480);
        let pixels = star_image(width, height);
        let config = DetectConfig {
            max_stars: 300,
            ..DetectConfig::default()
        };
        let linear = detect_stars_luma_f32(&pixels, width, height, &config);
        let wide = image::DynamicImage::ImageLuma16(
            image::ImageBuffer::from_raw(width, height, pixels.iter().map(|&v| v as u16).collect())
                .unwrap(),
        );
        let float = detect_stars(
            &wide,
            &DetectConfig {
                backend: DetectBackend::F32,
                ..config.clone()
            },
        );
        let narrow = image::DynamicImage::ImageLuma8(
            image::ImageBuffer::from_raw(
                width,
                height,
                pixels
                    .iter()
                    .map(|&v| ((v - 1000.0) / 40.0).clamp(0.0, 255.0) as u8)
                    .collect(),
            )
            .unwrap(),
        );
        let compact = detect_stars(
            &narrow,
            &DetectConfig {
                backend: DetectBackend::U8,
                ..config
            },
        );
        assert!(
            linear.len() > 20 && compact.len() > 20,
            "too few stars detected"
        );

        // A star catalog and blind index written and opened as psf-guard
        // opens them.
        let (stars, detected) = scene();
        let catalog_path = directory.join(format!("stars-{threads}.bin"));
        let mut builder = TileSetBuilder::new(45, 2016.0, "synthetic");
        for star in &stars {
            builder.add(star.ra, star.dec, star.mag);
        }
        builder.write_to(&catalog_path).unwrap();
        let catalog = TileCatalog::open(&catalog_path).unwrap();
        assert_eq!(catalog.star_count(), 3000);

        let hinted = solve(
            &detected,
            &catalog,
            &SolveHint {
                center: (150.8, 33.0),
                radius_deg: 2.0,
                scale_arcsec_px: 2.2,
                scale_tolerance: 0.25,
                sip_order: 0,
            },
            DIMENSIONS,
        )
        .unwrap();
        assert!(centre_error_arcsec(&hinted) < 3.0);

        let params = BlindParams {
            min_scale_arcsec_px: 1.0,
            max_scale_arcsec_px: 4.0,
            ..BlindParams::default()
        };
        let index_path = directory.join(format!("blind-{threads}.idx"));
        BlindIndex::build(&catalog, &params)
            .write_to(&index_path)
            .unwrap();
        let resolved = seiza::data_paths::blind_index(Some(&index_path))
            .unwrap()
            .unwrap();
        let index = BlindIndex::open(&resolved).unwrap();
        index.validate().unwrap();
        let params = BlindParams {
            index_mag_limit: index.index_mag_limit(),
            max_pattern_deg: index.max_pattern_deg(),
            ..params
        };
        let blind = solve_blind(&detected, &catalog, &index, &params, DIMENSIONS).unwrap();

        let index_bytes = std::fs::read(&index_path).unwrap();
        Run {
            exact: vec![
                ("linear detection", star_bits(&linear)),
                ("16-bit detection", star_bits(&float)),
                ("8-bit detection", star_bits(&compact)),
                ("hinted solve", solution_bits(&hinted)),
                (
                    "blind index",
                    index_bytes.iter().map(|&byte| u64::from(byte)).collect(),
                ),
            ],
            blind,
        }
    })
}

#[test]
fn detection_and_solving_stay_in_the_callers_pool() {
    let directory = tempfile::tempdir().unwrap();
    let pools = [1, 2].map(|threads| {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(move |index| format!("caller-{threads}-{index}"))
            .build()
            .unwrap()
    });
    let one = run(&pools[0], directory.path(), 1);
    let two = run(&pools[1], directory.path(), 2);
    for ((name, one), (_, two)) in one.exact.iter().zip(&two.exact) {
        assert!(one == two, "{name} differs between one and two threads");
    }
    for blind in [&one.blind, &two.blind] {
        assert!(blind.matched_stars >= 12, "{blind:?}");
        assert!(centre_error_arcsec(blind) < 3.0, "{blind:?}");
    }
    ThreadPoolBuilder::new()
        .build_global()
        .expect("work escaped the caller's pool and started the global pool");
}
