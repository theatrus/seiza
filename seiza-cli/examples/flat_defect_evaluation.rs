//! Experimental, uncalibrated flat comparison; never part of production integration.
//! cargo run --release -p seiza-cli --example flat_defect_evaluation --
//!   OUTPUT_DIRECTORY INPUT_FLAT... [--quick]
//! Quick evaluates all six configurations at 30 frames only. Default also
//! evaluates 5/10/20 frames with three deterministic subset selections.
use anyhow::{Result, ensure};
use clap::Parser;
use seiza_fits::{F32ImageData, WriteHeaderCard};
use seiza_stacking::{
    FitsFrame, ImpulseFilterOptions, LinearImage, MasterBuildOptions, MasterFrame, MasterFrameKind,
    build_master_from_fits, suppress_impulses, write_master_fits_f32,
    write_processed_image_fits_f32,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

fn median(values: &mut [f32]) -> f32 {
    values.sort_unstable_by(f32::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) * 0.5
    }
}

fn spatial(frame: &mut FitsFrame, sigma: f32, relative: f32, floor: f32) -> Vec<f32> {
    let mut mask = vec![0; frame.image.width * frame.image.height];
    seiza_stacking::preprocess_flat_spatial(
        &mut frame.image,
        frame.bayer,
        &seiza_stacking::ExperimentalFlatPreprocessingOptions {
            sigma,
            minimum_relative_excess: relative,
            noise_floor_adu: floor,
        },
        Some(&mut mask),
        None,
    )
    .unwrap();
    mask.into_iter().map(f32::from).collect()
}

fn mask_file(
    path: &Path,
    image: &LinearImage,
    data: Vec<f32>,
    headers: &[(String, seiza_fits::HeaderValue)],
) -> Result<()> {
    let row_order = headers
        .iter()
        .filter(|(key, _)| key.eq_ignore_ascii_case("ROWORDER"))
        .map(|(k, v)| WriteHeaderCard::new(k, v.clone()))
        .collect::<Vec<_>>();
    seiza_image_io::save_f32(
        path,
        image.width,
        image.height,
        F32ImageData::Mono(&data),
        &seiza_image_io::SaveOptions {
            headers: &row_order,
            fits_top_down: false,
            ..Default::default()
        },
    )?;
    Ok(())
}

fn build(
    paths: &[PathBuf],
    stars: bool,
    defects: bool,
    mask: Option<&Path>,
    saturation: f32,
) -> Result<MasterFrame> {
    let mut master = build_master_from_fits(
        paths,
        MasterFrameKind::Flat,
        &MasterBuildOptions {
            flat_star_masking: stars.then_some(seiza_stacking::FlatStarMaskingOptions {
                saturation_level: Some(saturation),
                ..Default::default()
            }),
            ..Default::default()
        },
    )?;
    if defects {
        let before = mask.map(|_| master.image.data.clone());
        master.defect_pixels_replaced = suppress_impulses(
            &mut master.image,
            master.bayer,
            &ImpulseFilterOptions::default(),
        )? as u64;
        if let (Some(path), Some(before)) = (mask, before) {
            let data = before
                .chunks(master.image.channels)
                .zip(master.image.data.chunks(master.image.channels))
                .map(|(a, b)| f32::from(a.iter().zip(b).any(|(a, b)| a.to_bits() != b.to_bits())))
                .collect();
            mask_file(path, &master.image, data, &master.reference_headers)?;
        }
    }
    Ok(master)
}

fn residual(a: &LinearImage, b: &LinearImage) -> Value {
    let mut sum = 0.0f64;
    let mut squares = 0.0f64;
    let mut changed = 0usize;
    let mut above_percent = 0usize;
    for (&a, &b) in a.data.iter().zip(&b.data) {
        let d = f64::from(a) - f64::from(b);
        sum += d;
        squares += d * d;
        changed += usize::from(a.to_bits() != b.to_bits());
        above_percent += usize::from(d.abs() > 0.01);
    }
    let n = a.data.len() as f64;
    json!({"mean_response_difference": sum/n, "rms_response_difference": (squares/n).sqrt(),
        "changed_samples": changed, "samples_above_0_01_response": above_percent})
}

fn identity(path: &Path) -> Result<Value> {
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(
        json!({"path": path.canonicalize()?, "bytes": fs::metadata(path)?.len(), "sha256": hash.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>()}),
    )
}

fn selections(paths: &[PathBuf], quick: bool) -> Vec<(usize, usize, Vec<usize>)> {
    let mut result = Vec::new();
    for n in [5, 10, 20, 30] {
        if n > paths.len() || (quick && n != 30) {
            continue;
        }
        for seed in 0..if n == paths.len() { 1 } else { 3 } {
            let mut indices = (0..paths.len()).collect::<Vec<_>>();
            // Stable LCG Fisher-Yates, independent of rand/library versions.
            let mut state = 20260912u64 + seed as u64;
            for i in (1..indices.len()).rev() {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                indices.swap(i, (state % (i + 1) as u64) as usize);
            }
            indices.truncate(n);
            indices.sort_unstable();
            result.push((n, seed, indices));
        }
    }
    result
}

#[derive(Parser)]
struct Arguments {
    output: PathBuf,
    #[arg(required = true)]
    images: Vec<PathBuf>,
    #[arg(long)]
    quick: bool,
    #[arg(long, value_delimiter = ',', default_value = "6,10")]
    spatial_sigmas: Vec<f32>,
    #[arg(long, default_value_t = 0.01)]
    minimum_relative_excess: f32,
    #[arg(long, default_value_t = 1.0)]
    noise_floor_adu: f32,
    /// Explicit raw-ADU ceiling, kept identical for integer and processed floats
    #[arg(long, default_value_t = 65535.0)]
    saturation_level: f32,
}

fn main() -> Result<()> {
    let args = Arguments::parse();
    let output = args.output;
    let mut paths = args.images;
    let quick = args.quick;
    ensure!(
        args.noise_floor_adu.is_finite() && args.noise_floor_adu > 0.0,
        "noise floor must be positive and finite"
    );
    ensure!(
        args.minimum_relative_excess.is_finite() && args.minimum_relative_excess >= 0.0,
        "relative excess must be nonnegative and finite"
    );
    ensure!(
        args.saturation_level.is_finite() && args.saturation_level > 0.0,
        "saturation ceiling must be positive and finite"
    );
    ensure!(
        !args.spatial_sigmas.is_empty()
            && args
                .spatial_sigmas
                .iter()
                .all(|s| s.is_finite() && *s > 0.0),
        "sigmas must be positive and finite"
    );
    paths.sort();
    ensure!(paths.len() >= 30, "need at least 30 real flats");
    ensure!(
        !output.exists(),
        "output must be new; never overwrite experiment results"
    );
    fs::create_dir_all(&output)?;
    let start = Instant::now();
    let mut dataset = Vec::new();
    for path in &paths {
        let frame = FitsFrame::open(path)?;
        ensure!(
            frame.image.channels == 1,
            "this evaluation expects mono/CFA flats"
        );
        let mut sampled = frame
            .image
            .data
            .iter()
            .step_by(97)
            .copied()
            .collect::<Vec<_>>();
        let mut item = identity(path)?;
        item["width"] = json!(frame.image.width);
        item["height"] = json!(frame.image.height);
        item["headers"] = json!(
            frame
                .headers
                .iter()
                .map(|(k, v)| (k, format!("{v:?}")))
                .collect::<Vec<_>>()
        );
        item["sampled_median_adu"] = json!(median(&mut sampled));
        item["nonfinite_samples"] =
            json!(frame.image.data.iter().filter(|v| !v.is_finite()).count());
        item["samples_at_or_above_65535"] =
            json!(frame.image.data.iter().filter(|&&v| v >= 65535.0).count());
        dataset.push(item);
        eprintln!("audited {}", path.display());
    }
    fs::write(
        output.join("dataset.json"),
        serde_json::to_vec_pretty(&dataset)?,
    )?;
    let subsets = selections(&paths, quick);
    fs::write(
        output.join("subsets.json"),
        serde_json::to_vec_pretty(&subsets)?,
    )?;
    let mut log = fs::File::create(output.join("results.jsonl"))?;
    // Each sigma's cache is removed before the next to bound scratch space.
    for sigma in std::iter::once(None).chain(args.spatial_sigmas.iter().copied().map(Some)) {
        let cache = tempfile::tempdir()?;
        let preprocess_start = Instant::now();
        let mut corrected = Vec::new();
        let mut per_frame_seconds = Vec::new();
        let source = if let Some(sigma) = sigma {
            let mut processed = Vec::new();
            for (i, path) in paths.iter().enumerate() {
                let frame_start = Instant::now();
                let mut frame = FitsFrame::open(path)?;
                let mask = spatial(
                    &mut frame,
                    sigma,
                    args.minimum_relative_excess,
                    args.noise_floor_adu,
                );
                corrected.push(mask.iter().filter(|&&v| v == 1.0).count());
                let indices = mask
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| **v == 1.0)
                    .map(|(i, _)| i)
                    .collect::<Vec<_>>();
                fs::write(
                    output.join(format!("spatial-{sigma}-{i}-mask.json")),
                    serde_json::to_vec(
                        &json!({"width":frame.image.width,"height":frame.image.height,
                    "storage_order_indices":indices,"default":0,"marked_value":1}),
                    )?,
                )?;
                let path = cache.path().join(format!("{i}.fits"));
                write_processed_image_fits_f32(&path, &frame.image, &frame.headers, &[])?;
                processed.push(path);
                per_frame_seconds.push(frame_start.elapsed().as_secs_f64());
                eprintln!("preprocessed sigma={sigma} frame={i}");
            }
            processed
        } else {
            paths.clone()
        };
        let preprocessing_seconds = preprocess_start.elapsed().as_secs_f64();
        for (n, seed, indices) in &subsets {
            let inputs = indices
                .iter()
                .map(|&i| source[i].clone())
                .collect::<Vec<_>>();
            let held_out = paths
                .iter()
                .enumerate()
                .filter(|(i, _)| !indices.contains(i))
                .map(|(_, p)| p.clone())
                .collect::<Vec<_>>();
            let reference = if held_out.len() >= 3 {
                Some(build(&held_out, true, true, None, args.saturation_level)?)
            } else {
                None
            };
            let configurations: &[(&str, bool, bool)] = if sigma.is_none() {
                &[
                    ("A", false, false),
                    ("B", true, false),
                    ("C", false, true),
                    ("D", true, true),
                ]
            } else {
                &[("E", false, false), ("F", true, true)]
            };
            for &(id, stars, defects) in configurations {
                let label = format!("{id}-sigma{}-n{n}-seed{seed}", sigma.unwrap_or(0.0));
                let timer = Instant::now();
                let mask_path = output.join(format!("{label}-mask.fits"));
                let keep = *n == 30 || (*n == 5 && *seed == 0);
                let master = build(
                    &inputs,
                    stars,
                    defects,
                    (defects && keep).then_some(mask_path.as_path()),
                    args.saturation_level,
                )?;
                let integration_seconds = timer.elapsed().as_secs_f64();
                // Retain 30-frame masters and one 5-frame comparison; others
                // are evaluated in memory to bound disk consumption.
                if *n == 30 || (*n == 5 && *seed == 0) {
                    write_master_fits_f32(output.join(format!("{label}.fits")), &master)?;
                }
                let mut row = json!({"configuration":id,"spatial_sigma":sigma,"count":n,"seed":seed,
                    "indices":indices,"integration_and_mask_seconds":integration_seconds,
                    "preprocessing_all_frames_seconds":preprocessing_seconds,"per_frame_corrections":corrected,
                    "selected_preprocessing_seconds":indices.iter().map(|&i| per_frame_seconds.get(i).copied().unwrap_or(0.0)).sum::<f64>(),
                    "defect_pixels_replaced":master.defect_pixels_replaced,"rejected_samples":master.rejected_samples,
                    "masked_samples":master.masked_samples,"fallback_pixels":master.fallback_pixels,
                    "input_statistics":master.input_statistics.iter().map(|s| json!({"accepted_samples":s.accepted_samples,"rejected_samples":s.rejected_samples,"masked_samples":s.masked_samples})).collect::<Vec<_>>(),"star_masking":master.flat_star_masking,
                    "reference_count":held_out.len(),"bias_subtracted":false,"dark_subtracted":false});
                if let Some(reference) = &reference {
                    row["held_out_comparison"] = residual(&master.image, &reference.image);
                }
                writeln!(log, "{row}")?;
                log.flush()?;
                eprintln!(
                    "completed {label}: {integration_seconds:.2}s, {} corrections",
                    master.defect_pixels_replaced
                );
            }
        }
    }
    fs::write(
        output.join("run.json"),
        serde_json::to_vec_pretty(&json!({
            "elapsed_seconds":start.elapsed().as_secs_f64(),
        "seiza_version":env!("CARGO_PKG_VERSION"),
        "rayon_threads":rayon::current_num_threads(),
        "git_head":std::process::Command::new("git").args(["rev-parse","HEAD"]).output().ok().map(|o|String::from_utf8_lossy(&o.stdout).trim().to_owned()),
        "harness_sha256":Sha256::digest(include_bytes!("flat_defect_evaluation.rs")).iter().map(|b|format!("{b:02x}")).collect::<String>(),"command":std::env::args().collect::<Vec<_>>(),
            "minimum_relative_excess":args.minimum_relative_excess,"noise_floor_adu":args.noise_floor_adu,"sigma_thresholds":args.spatial_sigmas,
        "saturation_level":args.saturation_level,
            "peak_memory_linux_status":fs::read_to_string("/proc/self/status").ok().and_then(|s|s.lines().find(|l|l.starts_with("VmHWM:")).map(str::to_owned)),
            "limitations":["No bias or dark subtraction", "Held-out references are small same-session comparisons, not ground truth",
                "30-frame candidates have no independent reference", "No downstream light validation", "Rejection counts only; no per-coordinate rejection API"]
        }))?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prototype_preserves_negative_and_subthreshold_response() {
        let mut image = LinearImage::new(16, 16, 1, vec![1000.0; 256]).unwrap();
        image.data[5 * 16 + 5] = 50000.0;
        image.data[10 * 16 + 10] = 0.0;
        image.data[5 * 16 + 10] = 1005.0;
        let fits = seiza_fits::FitsImage {
            width: 16,
            height: 16,
            planes: 1,
            pixels: seiza_fits::Pixels::F32(image.data),
            headers: vec![],
        };
        let mut frame = FitsFrame::from_fits(fits, None).unwrap();
        let mask = spatial(&mut frame, 6.0, 0.01, 1.0);
        assert_eq!(mask.iter().sum::<f32>(), 1.0);
        assert_eq!(frame.image.data[5 * 16 + 5], 1000.0);
        assert_eq!(frame.image.data[10 * 16 + 10], 0.0);
        assert_eq!(frame.image.data[5 * 16 + 10], 1005.0);
    }
    #[test]
    fn injected_positive_impulses_and_persistent_response_are_indistinguishable() {
        // Ground truth deliberately includes real high-response photosites.
        // Spatial shape alone cannot tell them from sensor artifacts.
        for sigma in [6.0, 10.0] {
            let width = 128;
            let mut data = (0..width * width)
                .map(|i| 10000.0 + ((i * 2654435761usize) % 97) as f32 - 48.0)
                .collect::<Vec<_>>();
            let positive = [
                20 * width + 20,
                20 * width + 40,
                20 * width + 60,
                40 * width + 20,
                40 * width + 40,
                40 * width + 60,
            ];
            let negative = [60 * width + 20, 60 * width + 40];
            let genuine_response = [
                80 * width + 20,
                80 * width + 40,
                80 * width + 60,
                80 * width + 80,
            ];
            for i in positive {
                data[i] += 20000.0;
            }
            for i in negative {
                data[i] -= 8000.0;
            }
            for i in genuine_response {
                data[i] += 1000.0;
            }
            let fits = seiza_fits::FitsImage {
                width,
                height: width,
                planes: 1,
                pixels: seiza_fits::Pixels::F32(data),
                headers: vec![],
            };
            let mut frame = FitsFrame::from_fits(fits, None).unwrap();
            let mask = spatial(&mut frame, sigma, 0.01, 1.0);
            let tp = positive.iter().filter(|&&i| mask[i] == 1.0).count();
            let fp = mask.iter().filter(|&&v| v == 1.0).count() - tp;
            let negative_detected = negative.iter().filter(|&&i| mask[i] == 1.0).count();
            assert_eq!(tp, positive.len());
            assert_eq!(fp, genuine_response.len());
            assert_eq!(negative_detected, 0);
            println!(
                "injection_metrics: {}",
                json!({"sigma":sigma,"true_positive":tp,"false_positive":fp,
                "positive_impulse_recall":tp as f64/positive.len() as f64,
                "positive_impulse_precision":tp as f64/(tp+fp) as f64,
                "negative_impulse_recall":0.0,"genuine_high_response_removed":fp,
                "scope":"synthetic isolated probes; not a real-data accuracy estimate"})
            );
        }
    }
}
