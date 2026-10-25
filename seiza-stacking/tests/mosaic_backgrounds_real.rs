//! Opt-in comparison on existing linear panel integrations. No catalog access.
use seiza_stacking::{
    BatchStackOptions, CalibrationMasters, Continue, FitsFrame, FrameDisposition, FrameWeighting,
    GpuOptions, Interpolation, LinearImage, LiveStacker, NormalizationMode, ReferenceRegion,
    StackOptions, integrate_registered_frames_banded, write_processed_image_fits_f32,
};
use std::path::{Path, PathBuf};

fn preview(
    path: &Path,
    image: &LinearImage,
    domain: seiza_stretch::ResolvedSampleDomain,
    plan: &seiza_stretch::StretchPlan,
) {
    let normalized = domain.apply(&image.data, 1).unwrap();
    let display = plan.apply_f32(&normalized, 1).unwrap();
    seiza_image_io::save_f32(
        path,
        image.width,
        image.height,
        seiza_fits::F32ImageData::Mono(&display),
        &Default::default(),
    )
    .unwrap();
}

// At a changing panel footprint, subtract the local sky trend measured by
// inputs covering both sides. This separates a coverage step from real
// structure crossing the same boundary. Medians limit stars and residual
// registration differences; report each panel as well as the pooled result.
fn seam_errors(image: &LinearImage, inputs: &[LinearImage]) -> serde_json::Value {
    let mut panels = Vec::new();
    let mut pooled = Vec::new();
    for frame in inputs {
        let mut errors = Vec::new();
        for y in 1..image.height - 1 {
            for x in 1..image.width - 1 {
                let p = y * image.width + x;
                for q in [p + 1, p + image.width] {
                    if frame.data[p].is_finite() == frame.data[q].is_finite()
                        || !image.data[p].is_finite()
                        || !image.data[q].is_finite()
                    {
                        continue;
                    }
                    let mut trends = inputs
                        .iter()
                        .filter(|f| f.data[p].is_finite() && f.data[q].is_finite())
                        .map(|f| f.data[q] - f.data[p])
                        .collect::<Vec<_>>();
                    if trends.is_empty() {
                        continue;
                    }
                    trends.sort_by(f32::total_cmp);
                    errors.push((image.data[q] - image.data[p] - trends[trends.len() / 2]).abs());
                }
            }
        }
        errors.sort_by(f32::total_cmp);
        panels.push(describe(&errors));
        pooled.extend(errors);
    }
    pooled.sort_by(f32::total_cmp);
    serde_json::json!({"panels":panels,"pooled":describe(&pooled)})
}

fn describe(sorted: &[f32]) -> serde_json::Value {
    if sorted.is_empty() {
        return serde_json::json!({"samples":0});
    }
    serde_json::json!({"samples":sorted.len(),"median_adu":sorted[sorted.len()/2],"p90_adu":sorted[sorted.len()*9/10]})
}

#[test]
#[ignore = "requires SEIZA_MOSAIC_PANELS: JSON array of four linear mono panel paths"]
fn real_mosaic_backgrounds_match_across_panel_edges() {
    let paths: Vec<PathBuf> =
        serde_json::from_str(&std::env::var("SEIZA_MOSAIC_PANELS").unwrap()).unwrap();
    assert_eq!(paths.len(), 4);
    let bin = std::env::var("SEIZA_MOSAIC_BIN")
        .unwrap_or_else(|_| "4".into())
        .parse::<u32>()
        .unwrap();
    let temporary = tempfile::tempdir().unwrap();
    let mut references = Vec::new();
    for (index, path) in paths.iter().enumerate() {
        let frame = FitsFrame::open(path).unwrap().binned(bin).unwrap();
        assert_eq!(frame.image.channels, 1);
        let path = temporary.path().join(format!("input-{index}.fits"));
        write_processed_image_fits_f32(&path, &frame.image, &frame.headers, &[]).unwrap();
        references.push(path);
    }
    let cpu = GpuOptions {
        enabled: false,
        ..Default::default()
    };
    let mut options = StackOptions {
        normalization: NormalizationMode::LocalBackground { tile_size: 256 },
        weighting: FrameWeighting::Equal,
        interpolation: Interpolation::Lanczos3,
        gpu: cpu,
        ..Default::default()
    };
    options.registration.model = seiza_stacking::RegistrationModel::Quadratic;
    options.acceptance.minimum_overlap_fraction = 0.05;
    let mut stacker = LiveStacker::with_expanding_reference(
        &references,
        CalibrationMasters::default(),
        options,
        150_000_000,
        |_, _, _, _| Continue::Yes,
    )
    .unwrap();
    let mut inputs = Vec::new();
    for path in &references {
        let FrameDisposition::Accepted(diagnostics) = stacker.push_fits(path).unwrap() else {
            panic!("panel rejected: {}", path.display());
        };
        let frame = FitsFrame::open(path).unwrap();
        let mapping = diagnostics.mapping;
        let image = mapping
            .extract_region_with_gpu(
                &frame.image,
                ReferenceRegion {
                    x: 0,
                    y: 0,
                    width: mapping.reference_width(),
                    height: mapping.reference_height(),
                },
                Interpolation::Lanczos3,
                &cpu,
            )
            .unwrap();
        inputs.push(image);
    }
    drop(stacker);
    let base = BatchStackOptions {
        scratch_directory: Some(temporary.path().to_path_buf()),
        gpu: cpu,
        ..Default::default()
    };
    let before =
        integrate_registered_frames_banded(4, &base, |i| Ok(inputs[i].clone()), |_, _, _| {})
            .unwrap();
    let corrected = BatchStackOptions {
        mosaic_panels: Some(vec![0, 1, 2, 3]),
        ..base
    };
    let started = std::time::Instant::now();
    let after =
        integrate_registered_frames_banded(4, &corrected, |i| Ok(inputs[i].clone()), |_, _, _| {})
            .unwrap();
    let elapsed = started.elapsed();
    assert!(
        before
            .snapshot
            .coverage
            .iter()
            .zip(&after.snapshot.coverage)
            .all(|(&a, &b)| (a > 0) == (b > 0)),
        "the covered footprint changed"
    );
    let before_report = seam_errors(&before.snapshot.image, &inputs);
    let after_report = seam_errors(&after.snapshot.image, &inputs);
    assert!(
        after_report["pooled"]["median_adu"].as_f64().unwrap()
            < before_report["pooled"]["median_adu"].as_f64().unwrap(),
        "{before_report} -> {after_report}"
    );
    if let Ok(output) = std::env::var("SEIZA_MOSAIC_OUTPUT") {
        let output = PathBuf::from(output);
        std::fs::create_dir_all(&output).unwrap();
        let domain = seiza_stretch::SampleDomain::PhysicalLinear {
            normalization: Default::default(),
        }
        .resolve(&before.snapshot.image.data, 1)
        .unwrap();
        let normalized = domain.apply(&before.snapshot.image.data, 1).unwrap();
        let display = seiza_stretch::StretchConfig::auto_mtf(Default::default(), 200_000)
            .resolve_for(&normalized, 1)
            .unwrap();
        for (name, image) in [
            ("before", &before.snapshot.image),
            ("after", &after.snapshot.image),
        ] {
            write_processed_image_fits_f32(output.join(format!("{name}.fits")), image, &[], &[])
                .unwrap();
            preview(&output.join(format!("{name}.png")), image, domain, &display);
        }
        let report = serde_json::json!({"bin":bin,"registration_model":"quadratic","width":after.snapshot.image.width,"height":after.snapshot.image.height,"before":before_report,"after":after_report,"corrected_integration_seconds":elapsed.as_secs_f64(),"stacking_output_version":seiza_stacking::OUTPUT_VERSION,"display_domain":domain,"display_plan":display});
        std::fs::write(
            output.join("report.json"),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
    }
    eprintln!(
        "{} x {}, {elapsed:?}: {before_report} -> {after_report}",
        after.snapshot.image.width, after.snapshot.image.height
    );
}
