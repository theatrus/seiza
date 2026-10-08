//! Star detection and measurement run in the Rayon pool their caller
//! installs.
//!
//! Every entry point here runs inside a one-thread and a two-thread pool,
//! and must give the same bits in both. The global pool must never start:
//! Rayon work that left the caller's pool would start it. That can be
//! checked once per process, and each file under `tests/` runs as a process
//! of its own, so this is one test.

use rayon::ThreadPoolBuilder;
use seiza_stars::accord_imaging::{BlobCounter, DetectionUtility};
use seiza_stars::hocus_focus_star_detection::{
    HocusFocusDetectionResult, HocusFocusParams, StructureRemovalMethod, detect_stars_hocus_focus,
    detect_stars_hocus_focus_adaptive,
};
use seiza_stars::nina_star_detection::{
    NoiseReduction, StarDetectionParams, StarSensitivity, detect_stars_with_original,
};
use seiza_stars::psf_fitting::{PSFFitter, PSFModel, PSFType};
use seiza_stars::star_contours::StarBlobDetector;
use seiza_stars::tilt::{TiltStar, analyze_cells, tilt_summary};

const WIDTH: usize = 384;
const HEIGHT: usize = 288;

/// Gaussian stars, a little elongated towards one corner, on a noisy sky.
fn star_field() -> Vec<u16> {
    let mut seed = 5u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let stars = (0..40)
        .map(|_| {
            let (x, y) = (
                12.0 + next() * (WIDTH as f64 - 24.0),
                12.0 + next() * (HEIGHT as f64 - 24.0),
            );
            let stretch = 1.0 + 0.4 * x / WIDTH as f64;
            (x, y, 8000.0 + next() * 30000.0, 1.6 + next() * 0.8, stretch)
        })
        .collect::<Vec<_>>();
    (0..WIDTH * HEIGHT)
        .map(|index| {
            let (x, y) = ((index % WIDTH) as f64, (index / WIDTH) as f64);
            let mut value = 1000.0 + ((index as u64 * 2654435761) % 61) as f64;
            for &(cx, cy, amplitude, sigma, stretch) in &stars {
                let (dx, dy) = ((x - cx) / stretch, y - cy);
                let d2 = dx * dx + dy * dy;
                if d2 < 36.0 * sigma * sigma {
                    value += amplitude * (-d2 / (2.0 * sigma * sigma)).exp();
                }
            }
            value.min(65535.0) as u16
        })
        .collect()
}

fn model_bits(model: &PSFModel) -> Vec<u64> {
    [
        model.amplitude,
        model.background,
        model.x0,
        model.y0,
        model.sigma_x,
        model.sigma_y,
        model.theta,
        model.r_squared,
        model.rmse,
        model.fwhm,
        model.eccentricity,
    ]
    .iter()
    .map(|value| value.to_bits())
    .collect()
}

fn hocus_bits(result: &HocusFocusDetectionResult) -> Vec<u64> {
    let mut out = vec![
        result.stars.len() as u64,
        result.average_hfr.to_bits(),
        result.average_fwhm.to_bits(),
        result.noise_sigma.to_bits(),
        result.background_mean.to_bits(),
    ];
    for star in &result.stars {
        out.extend(
            [
                star.position.0,
                star.position.1,
                star.hfr,
                star.fwhm,
                star.brightness,
                star.background,
                star.snr,
                star.flux,
            ]
            .iter()
            .map(|value| value.to_bits()),
        );
        out.push(star.pixel_count as u64);
        out.push(u64::from(star.saturated));
        out.extend(star.psf_model.iter().flat_map(model_bits));
    }
    out
}

fn optional(value: Option<f64>) -> u64 {
    value.map_or(u64::MAX, f64::to_bits)
}

fn run(pool: &rayon::ThreadPool) -> Vec<(&'static str, Vec<u64>)> {
    pool.install(|| {
        let mut results = Vec::new();
        let data = star_field();

        // HocusFocus, with a PSF fit, each structure removal and binning.
        let mut fitted = None;
        for (name, params) in [
            ("hocus focus", HocusFocusParams::default()),
            (
                "hocus focus gaussian",
                HocusFocusParams {
                    psf_type: PSFType::Gaussian,
                    ..HocusFocusParams::default()
                },
            ),
            (
                "hocus focus atrous",
                HocusFocusParams {
                    structure_removal: StructureRemovalMethod::Atrous,
                    noise_reduction_radius: 0,
                    ..HocusFocusParams::default()
                },
            ),
            (
                "hocus focus binned",
                HocusFocusParams {
                    detection_binning: 2,
                    ..HocusFocusParams::default()
                },
            ),
        ] {
            let result = detect_stars_hocus_focus(&data, WIDTH, HEIGHT, &params);
            assert!(!result.stars.is_empty(), "{name}: no stars");
            if params.psf_type == PSFType::Gaussian {
                fitted = Some(result.clone());
            }
            results.push((name, hocus_bits(&result)));
        }
        let adaptive = detect_stars_hocus_focus_adaptive(
            &data,
            WIDTH,
            HEIGHT,
            &HocusFocusParams {
                sensitivity: 40.0,
                ..HocusFocusParams::default()
            },
            1000,
        );
        results.push(("hocus focus adaptive", hocus_bits(&adaptive)));

        // A star fitted on its own, and its residuals.
        let fitted = fitted.unwrap();
        let star = &fitted.stars[0];
        let fitter = PSFFitter::new(PSFType::Moffat4);
        let model = fitter
            .fit_star(
                &data,
                WIDTH,
                HEIGHT,
                star.position.0,
                star.position.1,
                12.0,
                12.0,
                star.background,
                star.brightness,
            )
            .unwrap();
        let (observed, model_map, residuals) = fitter
            .generate_residuals(
                &data,
                WIDTH,
                HEIGHT,
                star.position.0,
                star.position.1,
                &model,
            )
            .unwrap();
        let mut psf = model_bits(&model);
        for map in [observed, model_map, residuals] {
            psf.extend(map.iter().flatten().map(|value| value.to_bits()));
        }
        results.push(("psf fit", psf));

        // Tilt from the fitted stars.
        let tilt_stars = fitted
            .stars
            .iter()
            .map(|star| TiltStar {
                x: star.position.0,
                y: star.position.1,
                hfr: star.hfr,
                eccentricity: star.psf_model.as_ref().map_or(0.0, |m| m.eccentricity),
                theta: star.psf_model.as_ref().map(PSFModel::major_axis_theta),
            })
            .collect::<Vec<_>>();
        let cells = analyze_cells(&tilt_stars, WIDTH, HEIGHT);
        let summary = tilt_summary(&cells);
        let mut tilt = cells
            .iter()
            .flat_map(|cell| {
                [
                    cell.star_count as u64,
                    optional(cell.median_hfr),
                    optional(cell.median_eccentricity),
                    optional(cell.mean_theta),
                    cell.theta_coherence.to_bits(),
                ]
            })
            .collect::<Vec<_>>();
        tilt.extend([
            optional(summary.center_hfr),
            optional(summary.mean_hfr),
            optional(summary.tilt_percent),
            optional(summary.curvature_percent),
        ]);
        results.push(("tilt", tilt));

        // N.I.N.A.'s detector, at each noise reduction.
        for (name, noise_reduction) in [
            ("nina", NoiseReduction::None),
            ("nina normal", NoiseReduction::Normal),
            ("nina median", NoiseReduction::Median),
        ] {
            let params = StarDetectionParams {
                sensitivity: StarSensitivity::High,
                noise_reduction,
                ..StarDetectionParams::default()
            };
            let result = detect_stars_with_original(&data, &data, WIDTH, HEIGHT, &params);
            let mut out = vec![
                result.star_list.len() as u64,
                result.average_hfr.to_bits(),
                result.hfr_std_dev.to_bits(),
            ];
            for star in &result.star_list {
                out.extend(
                    [
                        star.hfr,
                        star.position.0,
                        star.position.1,
                        star.average_brightness,
                        star.max_brightness,
                        star.background,
                        star.flux,
                    ]
                    .iter()
                    .map(|value| value.to_bits()),
                );
            }
            results.push((name, out));
        }

        // Blobs and contours of a thresholded frame, and a resize for
        // detection.
        let binary = data
            .iter()
            .map(|&value| if value > 4000 { 255 } else { 0 })
            .collect::<Vec<u8>>();
        let mut counter = BlobCounter::new();
        counter.process_image(&binary, WIDTH, HEIGHT);
        // The counter lists blobs in hash map order, which changes from run
        // to run whatever the pool, so they are compared sorted.
        let mut rectangles = counter
            .get_objects_information()
            .iter()
            .map(|blob| {
                let r = blob.rectangle;
                [r.x, r.y, r.width, r.height]
            })
            .collect::<Vec<_>>();
        rectangles.sort_unstable();
        let mut blobs = rectangles
            .iter()
            .flatten()
            .map(|&value| value as u64)
            .collect::<Vec<_>>();
        let contours = StarBlobDetector::default().analyze_star_contours(&binary, WIDTH, HEIGHT);
        for contour in &contours {
            blobs.extend(
                [
                    contour.area,
                    contour.perimeter,
                    contour.circularity,
                    contour.convexity,
                    contour.centroid.0,
                    contour.centroid.1,
                ]
                .iter()
                .map(|value| value.to_bits()),
            );
        }
        let (resized, width, height) =
            DetectionUtility::resize_for_detection(&binary, WIDTH, HEIGHT, 256, 0.5);
        blobs.extend([width as u64, height as u64]);
        blobs.extend(resized.iter().map(|&value| u64::from(value)));
        results.push(("blobs and contours", blobs));
        results
    })
}

#[test]
fn star_measurement_stays_in_the_callers_pool() {
    seiza_stars::debug::init_debug(false);
    let pools = [1, 2].map(|threads| {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(move |index| format!("caller-{threads}-{index}"))
            .build()
            .unwrap()
    });
    let one = run(&pools[0]);
    let two = run(&pools[1]);
    assert_eq!(one.len(), two.len());
    for ((name, one), (_, two)) in one.iter().zip(&two) {
        assert!(one == two, "{name} differs between one and two threads");
    }
    ThreadPoolBuilder::new()
        .build_global()
        .expect("work escaped the caller's pool and started the global pool");
}
