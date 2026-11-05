//! Read evaluation masters through the shared image reader and export numeric
//! differences/crops for plotting. No stretch feeds back into measurements.
use anyhow::{Context, Result};
use seiza_stacking::FitsFrame;
use serde_json::json;
use std::{fs, path::PathBuf};

fn main() -> Result<()> {
    let directory = PathBuf::from(
        std::env::args()
            .nth(1)
            .context("supply evaluation directory")?,
    );
    let mut paths = fs::read_dir(&directory)?
        .map(|p| p.map(|p| p.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.sort();
    let mut comparisons = Vec::new();
    for path in paths {
        let name = path.file_stem().unwrap().to_string_lossy();
        if path.extension().is_none_or(|ext| ext != "fits") || name.ends_with("-mask") {
            continue;
        }
        let (_, suffix) = name
            .split_once("-n")
            .context("unexpected master filename")?;
        let baseline_path = directory.join(format!("D-sigma0-n{suffix}.fits"));
        let baseline = FitsFrame::open(baseline_path)?;
        let candidate = FitsFrame::open(&path)?;
        let mut squares = 0.0f64;
        let mut changed = 0usize;
        let mut extreme = (0, 0.0f32);
        for (i, (&a, &b)) in candidate
            .image
            .data
            .iter()
            .zip(&baseline.image.data)
            .enumerate()
        {
            let d = a - b;
            squares += f64::from(d).powi(2);
            changed += usize::from(a.to_bits() != b.to_bits());
            if d.abs() > extreme.1 {
                extreme = (i, d.abs());
            }
        }
        // Use a fixed 64x64 crop around the largest departure, sharing the
        // same sensor coordinate for candidate and D. Export linear samples.
        let width = candidate.image.width;
        let height = candidate.image.height;
        let x = (extreme.0 % width)
            .saturating_sub(32)
            .min(width.saturating_sub(64));
        let y = (extreme.0 / width)
            .saturating_sub(32)
            .min(height.saturating_sub(64));
        let w = 64.min(width);
        let h = 64.min(height);
        let crop = |data: &[f32]| {
            (y..y + h)
                .flat_map(|yy| (x..x + w).map(move |xx| data[yy * width + xx]))
                .collect::<Vec<_>>()
        };
        let mut row = json!({"name":name,"rms_difference_from_D":(squares/candidate.image.data.len() as f64).sqrt(),
            "changed_samples_from_D":changed,"maximum_absolute_difference_from_D":extreme.1,
            "largest_difference_coordinate":[extreme.0%width,extreme.0/width]});
        if !name.starts_with('D') {
            let crop_path = directory.join(format!("{name}-crop.json"));
            fs::write(
                crop_path,
                serde_json::to_vec(&json!({"name":name,"x":x,"y":y,"width":w,"height":h,
                "candidate":crop(&candidate.image.data),"D":crop(&baseline.image.data)}))?,
            )?;
        }
        let mask_path = directory.join(format!("{name}-mask.fits"));
        if mask_path.exists() {
            let mask = FitsFrame::open(mask_path)?;
            row["mask_ones"] = json!(mask.image.data.iter().filter(|&&v| v == 1.0).count());
        }
        comparisons.push(row);
    }
    fs::write(
        directory.join("master-comparisons.json"),
        serde_json::to_vec_pretty(&comparisons)?,
    )?;
    let dataset: Vec<serde_json::Value> =
        serde_json::from_slice(&fs::read(directory.join("dataset.json"))?)?;
    let mut saturation = Vec::new();
    for row in dataset {
        let path = row["path"].as_str().context("dataset path missing")?;
        let frame = FitsFrame::open(path)?;
        let positions = frame
            .image
            .data
            .iter()
            .enumerate()
            .filter(|(_, v)| **v >= 65535.0)
            .map(|(i, _)| vec![i % frame.image.width, i / frame.image.width])
            .collect::<Vec<_>>();
        saturation.push(json!({"path":path,"coordinates":positions}));
    }
    fs::write(
        directory.join("saturated-coordinates.json"),
        serde_json::to_vec_pretty(&saturation)?,
    )?;
    Ok(())
}
