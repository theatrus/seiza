//! Flat-only downstream comparisons through the existing calibration API.
//! No bias/dark subtraction and no claim of calibrated-light ground truth.
use anyhow::{Context, Result, ensure};
use seiza_stacking::{CalibrationMasters, FitsFrame, LinearImage};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, io::Read, path::PathBuf};

fn median(data: &mut [f32]) -> f32 {
    data.sort_unstable_by(f32::total_cmp);
    let n = data.len();
    if n % 2 == 1 {
        data[n / 2]
    } else {
        (data[n / 2 - 1] + data[n / 2]) * 0.5
    }
}

fn noise(image: &LinearImage) -> f32 {
    let mut differences = Vec::new();
    for y in (0..image.height).step_by(8) {
        for x in (0..image.width.saturating_sub(1)).step_by(8) {
            let i = y * image.width + x;
            differences.push(image.data[i + 1] - image.data[i]);
        }
    }
    let center = median(&mut differences);
    for v in &mut differences {
        *v = (*v - center).abs();
    }
    1.4826 * median(&mut differences) / std::f32::consts::SQRT_2
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let directory = PathBuf::from(
        args.next()
            .context("supply experiment directory and matching raw light paths")?,
    );
    let lights = args.map(PathBuf::from).collect::<Vec<_>>();
    ensure!(!lights.is_empty(), "supply at least one raw light");
    let mut masters = fs::read_dir(&directory)?
        .map(|p| p.map(|p| p.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    masters.retain(|p| {
        p.file_name()
            .unwrap()
            .to_string_lossy()
            .ends_with("-n30-seed0.fits")
    });
    masters.sort();
    let baseline_path = directory.join("D-sigma0-n30-seed0.fits");
    let baseline_masters =
        CalibrationMasters::from_fits_paths(None, None, Some(&baseline_path), None)?;
    let mut results = Vec::new();
    for path in lights {
        let source = FitsFrame::open(&path)?;
        ensure!(source.image.channels == 1, "mono light required");
        baseline_masters.validate_light_frame(&source)?;
        let mut baseline = source.image.clone();
        baseline_masters.apply(&mut baseline, source.exposure_seconds, source.bayer)?;
        let baseline_noise = noise(&baseline);
        let mut file = fs::File::open(&path)?;
        let mut hash = Sha256::new();
        let mut buffer = vec![0; 1024 * 1024];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hash.update(&buffer[..n]);
        }
        let identity = json!({"path":path,"sha256":hash.finalize().iter().map(|b|format!("{b:02x}")).collect::<String>(),
            "exposure_seconds":source.exposure_seconds,"headers":source.headers.iter().map(|(k,v)|(k,format!("{v:?}"))).collect::<Vec<_>>()});
        for master_path in &masters {
            let prepared =
                CalibrationMasters::from_fits_paths(None, None, Some(master_path), None)?;
            prepared.validate_light_frame(&source)?;
            let mut calibrated = source.image.clone();
            prepared.apply(&mut calibrated, source.exposure_seconds, source.bayer)?;
            let mut squares = 0.0f64;
            let mut sum = 0.0f64;
            let mut changed = 0usize;
            let mut maximum = (0, 0.0f32);
            for (i, (&a, &b)) in calibrated.data.iter().zip(&baseline.data).enumerate() {
                let d = a - b;
                sum += f64::from(d);
                squares += f64::from(d).powi(2);
                changed += usize::from(a.to_bits() != b.to_bits());
                if d.abs() > maximum.1 {
                    maximum = (i, d.abs());
                }
            }
            let name = master_path.file_stem().unwrap().to_string_lossy();
            results.push(json!({"light":identity,"master":name,"bias_subtracted":false,"dark_subtracted":false,
                "rms_difference_from_D_adu":(squares/calibrated.data.len() as f64).sqrt(),
                "mean_difference_from_D_adu":sum/calibrated.data.len() as f64,
                "maximum_absolute_difference_adu":maximum.1,"changed_samples":changed,
                "sampled_adjacent_difference_mad_sigma_adu":noise(&calibrated),
                "D_sampled_adjacent_difference_mad_sigma_adu":baseline_noise}));
            if name.starts_with('F') {
                let width = calibrated.width;
                let height = calibrated.height;
                let x = (maximum.0 % width)
                    .saturating_sub(32)
                    .min(width.saturating_sub(64));
                let y = (maximum.0 / width)
                    .saturating_sub(32)
                    .min(height.saturating_sub(64));
                let w = 64.min(width);
                let h = 64.min(height);
                let crop = |data: &[f32]| {
                    (y..y + h)
                        .flat_map(|yy| (x..x + w).map(move |xx| data[yy * width + xx]))
                        .collect::<Vec<_>>()
                };
                let light_name = path.file_stem().unwrap().to_string_lossy();
                let crop_path = directory.join(format!("light-{light_name}-{name}-crop.json"));
                fs::write(
                    crop_path,
                    serde_json::to_vec(
                        &json!({"name":format!("{light_name}: {name}"),"x":x,"y":y,"width":w,"height":h,
                    "units":"Flat-only light (ADU)","difference_units":"ADU difference",
                    "candidate":crop(&calibrated.data),"D":crop(&baseline.data)}),
                    )?,
                )?;
            }
            eprintln!("checked {} with {name}", path.display());
        }
    }
    let report: Value = json!({"status":"complete","flat_only":true,"results":results,
        "limitations":["No dark/bias subtraction", "No independent calibrated-light truth", "Noise statistic includes all sampled scene regions; not a verified star-free background"]});
    fs::write(
        directory.join("light-comparisons.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(())
}
