//! Bounded, read-only summaries of completed integrations.
use crate::{BatchFrameDiagnostics, DrizzleResult, Error, LiveStacker, Result, StackSnapshot};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// One admitted input, in integration order. Missing history stays unavailable.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StackReportFrame {
    pub source: Option<PathBuf>,
    pub exposure_seconds: Option<f64>,
    pub calibration_index: u32,
    /// Admission-pass registration RMS in sensor pixels, not a refined result.
    pub registration_rms_pixels: Option<f64>,
    pub weight: Vec<f32>,
    pub noise: Vec<f32>,
    pub mapping: serde_json::Value,
}

/// Capture this at the same point as the snapshot being reported. It does not
/// reread inputs or retain a second live history. Contexts restored from disk
/// can lack exposure and registration measurements.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StackReportContext {
    pub accepted_frames: u32,
    pub rejected_frames: u32,
    pub reference_integrated: bool,
    pub reference_image: Option<PathBuf>,
    pub options: serde_json::Value,
    pub calibrations: Vec<serde_json::Value>,
    pub frames: Vec<StackReportFrame>,
    pub planning_notes: Vec<String>,
}

/// A block-mean map, at most 128 by 128 cells per channel. Unsupported pixels
/// contribute zero to the block mean; no display stretch is applied to the numbers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StackReportMap {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    /// Row-major, interleaved channels: (y * width + x) * channels + channel.
    pub values: Vec<f64>,
    pub quantity: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StackReportChannel {
    pub supported_pixels: usize,
    pub total_pixels: usize,
    pub support_minimum: f64,
    pub support_maximum: f64,
    pub support_mean: f64,
    pub rejected_samples: Option<u64>,
}

/// Final sample decisions in integration order, never live admission decisions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StackReportSamples {
    pub finite_samples: usize,
    pub integrated_samples: usize,
}

/// Portable post-stack report. `provenance` optionally carries a host's
/// admission/provenance document; it is not used to calculate output statistics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PostStackReport {
    pub schema_version: u32,
    pub stacking_version: String,
    pub output_version: String,
    pub stage: String,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub accepted_frames: u32,
    pub rejected_frames: u32,
    pub support: Vec<StackReportChannel>,
    pub support_map: StackReportMap,
    pub rejection_map: Option<StackReportMap>,
    pub final_samples: Option<Vec<StackReportSamples>>,
    pub context: Option<StackReportContext>,
    pub provenance: Option<serde_json::Value>,
    pub warnings: Vec<String>,
}

impl LiveStacker {
    /// Capture the existing ledger and actual options without reading images.
    /// This describes admitted inputs, not a log of failed admission attempts.
    pub fn report_context(&self) -> StackReportContext {
        let (frames, calibrations) = self.ledger.report_records();
        let reference_image = self.reference_image_path().map(PathBuf::from).or_else(|| {
            if self.reference_is_integrated() {
                frames.first().and_then(|frame| frame.source.clone())
            } else {
                None
            }
        });
        StackReportContext {
            accepted_frames: self.accepted_frames,
            rejected_frames: self.rejected_frames,
            reference_integrated: self.reference_is_integrated(),
            reference_image,
            options: serde_json::to_value(&self.options).expect("stack options serialize"),
            frames,
            calibrations,
            planning_notes: self.expansion_planning_notes().to_vec(),
        }
    }
}

fn dimensions(width: usize, height: usize, channels: usize, length: usize) -> Result<usize> {
    let size = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(channels));
    if width == 0 || height == 0 || channels == 0 || size != Some(length) {
        return Err(Error::Stack("invalid report image dimensions".into()));
    }
    Ok(width * height)
}

fn summarize(
    width: usize,
    height: usize,
    channels: usize,
    support: impl Fn(usize) -> f64,
    rejected: Option<&[u32]>,
    quantity: &str,
) -> (
    Vec<StackReportChannel>,
    StackReportMap,
    Option<StackReportMap>,
) {
    let mw = width.min(128);
    let mh = height.min(128);
    let mut values = vec![0.0; mw * mh * channels];
    let mut rejects = rejected.map(|_| vec![0.0; values.len()]);
    let mut counts = vec![0_usize; mw * mh];
    let mut stats = (0..channels)
        .map(|_| StackReportChannel {
            supported_pixels: 0,
            total_pixels: width * height,
            support_minimum: f64::INFINITY,
            support_maximum: 0.0,
            support_mean: 0.0,
            rejected_samples: rejected.map(|_| 0),
        })
        .collect::<Vec<_>>();
    for y in 0..height {
        for x in 0..width {
            let cell = (y * mh / height) * mw + x * mw / width;
            counts[cell] += 1;
            for (c, stat) in stats.iter_mut().enumerate() {
                let i = (y * width + x) * channels + c;
                let v = support(i);
                stat.supported_pixels += usize::from(v > 0.0);
                stat.support_minimum = stat.support_minimum.min(v);
                stat.support_maximum = stat.support_maximum.max(v);
                stat.support_mean += v;
                values[cell * channels + c] += v;
                if let (Some(raw), Some(map)) = (rejected, rejects.as_mut()) {
                    *stat.rejected_samples.as_mut().unwrap() += u64::from(raw[i]);
                    map[cell * channels + c] += f64::from(raw[i]);
                }
            }
        }
    }
    for stat in &mut stats {
        stat.support_mean /= (width * height) as f64;
    }
    for (cell, &count) in counts.iter().enumerate() {
        for c in 0..channels {
            values[cell * channels + c] /= count as f64;
            if let Some(map) = rejects.as_mut() {
                map[cell * channels + c] /= count as f64;
            }
        }
    }
    let map = StackReportMap {
        width: mw,
        height: mh,
        channels,
        values,
        quantity: quantity.into(),
    };
    let rejection = rejects.map(|values| StackReportMap {
        width: mw,
        height: mh,
        channels,
        values,
        quantity: "rejected_observations".into(),
    });
    (stats, map, rejection)
}

fn assemble(
    shape: (usize, usize, usize),
    counts: (u32, u32),
    stage: &str,
    context: Option<&StackReportContext>,
    final_samples: Option<&[BatchFrameDiagnostics]>,
    maps: (
        Vec<StackReportChannel>,
        StackReportMap,
        Option<StackReportMap>,
    ),
) -> Result<PostStackReport> {
    let (width, height, channels) = shape;
    let (accepted, rejected) = counts;
    if let Some(context) = context {
        if context.accepted_frames != accepted || context.rejected_frames != rejected {
            return Err(Error::Stack(
                "report context counts do not match this result; capture context with the result"
                    .into(),
            ));
        }
        if !context.frames.is_empty() && context.frames.len() != accepted as usize {
            return Err(Error::Stack(
                "report context has an incomplete admitted-frame ledger".into(),
            ));
        }
    }
    if let Some(frames) = final_samples
        && (frames.len() != accepted as usize
            || frames
                .iter()
                .any(|f| f.integrated_samples > f.finite_samples))
    {
        return Err(Error::Stack(
            "final sample diagnostics do not match this result".into(),
        ));
    }
    Ok(PostStackReport {
        schema_version: 1,
        stacking_version: crate::VERSION.into(),
        output_version: crate::OUTPUT_VERSION.into(),
        stage: stage.into(),
        width,
        height,
        channels,
        accepted_frames: accepted,
        rejected_frames: rejected,
        support: maps.0,
        support_map: maps.1,
        rejection_map: maps.2,
        final_samples: final_samples.map(|rows| rows.iter().map(|f| StackReportSamples {
            finite_samples: f.finite_samples,
            integrated_samples: f.integrated_samples,
        }).collect()),
        context: context.cloned(),
        provenance: None,
        warnings: vec![
            "Coverage is not a calibrated uncertainty or SNR measurement.".into(),
            "The admitted-frame ledger does not retain individual failed admission attempts; host provenance may provide those reasons.".into(),
            "Registration RMS and mappings describe the admission pass, not any subsequent full-resolution refinement.".into(),
        ],
    })
}

/// Summarize the supplied immutable result. Pass batch diagnostics only for
/// the completed batch integration which produced this snapshot. No input files are reread.
pub fn stack_report(
    snapshot: &StackSnapshot,
    context: Option<&StackReportContext>,
    final_samples: Option<&[BatchFrameDiagnostics]>,
) -> Result<PostStackReport> {
    let image = &snapshot.image;
    dimensions(image.width, image.height, image.channels, image.data.len())?;
    if snapshot.coverage.len() != image.data.len()
        || snapshot.rejected_samples.len() != image.data.len()
    {
        return Err(Error::Stack(
            "report count maps do not match the output image".into(),
        ));
    }
    let maps = summarize(
        image.width,
        image.height,
        image.channels,
        |i| f64::from(snapshot.coverage[i]),
        Some(&snapshot.rejected_samples),
        "accepted_observations",
    );
    assemble(
        (image.width, image.height, image.channels),
        (snapshot.accepted_frames, snapshot.rejected_frames),
        if final_samples.is_some() {
            "batch"
        } else {
            "online"
        },
        context,
        final_samples,
        maps,
    )
}

/// Summarize the drizzle output grid. Its support is drop area times frame
/// weight, not frame count. Final rejection statistics refer to the input grid.
pub fn drizzle_report(
    result: &DrizzleResult,
    accepted_frames: u32,
    rejected_frames: u32,
    context: Option<&StackReportContext>,
    final_samples: Option<&[BatchFrameDiagnostics]>,
) -> Result<PostStackReport> {
    let image = &result.image;
    dimensions(image.width, image.height, image.channels, image.data.len())?;
    if result.weight.width != image.width
        || result.weight.height != image.height
        || result.weight.channels != image.channels
        || result.weight.data.len() != image.data.len()
        || result
            .weight
            .data
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(Error::Stack("invalid drizzle support map".into()));
    }
    let maps = summarize(
        image.width,
        image.height,
        image.channels,
        |i| f64::from(result.weight.data[i]),
        None,
        "drop_area_times_frame_weight",
    );
    let mut report = assemble(
        (image.width, image.height, image.channels),
        (accepted_frames, rejected_frames),
        "drizzled",
        context,
        final_samples,
        maps,
    )?;
    report.warnings.push("Drizzle support is accumulated drop area multiplied by frame weight. Per-frame rejection counts describe the pre-drizzle integration grid.".into());
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LinearImage;
    fn snapshot() -> StackSnapshot {
        StackSnapshot {
            image: LinearImage {
                width: 2,
                height: 1,
                channels: 2,
                data: vec![1.0; 4],
            },
            variance: LinearImage {
                width: 2,
                height: 1,
                channels: 2,
                data: vec![0.0; 4],
            },
            coverage: vec![0, 2, 4, 6],
            weight_sum: None,
            rejected_samples: vec![1, 0, 2, 3],
            accepted_frames: 6,
            rejected_frames: 1,
        }
    }
    #[test]
    fn channel_support_and_rejection_are_measured_without_changing_pixels() {
        let snapshot = snapshot();
        let report = stack_report(&snapshot, None, None).unwrap();
        assert_eq!(report.stage, "online");
        assert_eq!(report.support[0].supported_pixels, 1);
        assert_eq!(report.support[1].support_mean, 4.0);
        assert_eq!(report.support[0].rejected_samples, Some(3));
        assert_eq!(report.support_map.values, vec![0.0, 2.0, 4.0, 6.0]);
        assert_eq!(snapshot.image.data, vec![1.0; 4]);
        assert_eq!(snapshot.coverage, vec![0, 2, 4, 6]);
    }
    #[test]
    fn map_is_bounded_block_mean_including_missing_pixels() {
        let (_, map, _) = summarize(
            256,
            256,
            1,
            |i| if i % 2 == 0 { 0.0 } else { 4.0 },
            None,
            "test",
        );
        assert_eq!((map.width, map.height), (128, 128));
        assert!(map.values.iter().all(|&v| v == 2.0));
    }
    #[test]
    fn final_diagnostics_are_explicit_and_validated() {
        let snapshot = snapshot();
        assert!(stack_report(&snapshot, None, Some(&[])).is_err());
        let samples = vec![
            BatchFrameDiagnostics {
                finite_samples: 4,
                integrated_samples: 3
            };
            6
        ];
        let report = stack_report(&snapshot, None, Some(&samples)).unwrap();
        assert_eq!(report.stage, "batch");
        assert_eq!(report.final_samples.unwrap()[0].integrated_samples, 3);
    }
    #[test]
    fn drizzle_reports_weight_not_frame_count() {
        let result = DrizzleResult {
            image: snapshot().image,
            weight: LinearImage {
                width: 2,
                height: 1,
                channels: 2,
                data: vec![0.0, 0.25, 1.5, 2.0],
            },
            scale: 2,
        };
        let report = drizzle_report(&result, 6, 0, None, None).unwrap();
        assert_eq!(report.support_map.quantity, "drop_area_times_frame_weight");
        assert_eq!(report.support[0].support_mean, 0.75);
        assert!(report.rejection_map.is_none());
    }
}
