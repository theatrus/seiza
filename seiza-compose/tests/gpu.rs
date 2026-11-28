//! The GPU compositor draws what the CPU compositor draws, within the
//! precision of its half-float inputs and tables. The CPU render is the
//! reference. Without a GPU wgpu can use, these tests pass without checking.

#![cfg(feature = "gpu")]

mod common;

use common::{everything, plane, whole};
use seiza_compose::gpu::{GpuContext, render_gpu};
use seiza_compose::picture::Picture;
use seiza_compose::render::{ImageSink, ImageSource, RenderOptions, render_image};
use seiza_stacking::LinearImage;

/// How far apart two renders are over pixels both cover: the mean, the
/// 99th percentile and the largest difference of any channel.
fn gap(cpu: &[f32], gpu: &[f32]) -> (f64, f32, f32) {
    let mut differences: Vec<f32> = cpu
        .iter()
        .zip(gpu)
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| (a - b).abs())
        .collect();
    differences.sort_by(f32::total_cmp);
    let mean = differences
        .iter()
        .map(|value| f64::from(*value))
        .sum::<f64>()
        / differences.len().max(1) as f64;
    let at = |share: f64| differences[((differences.len() - 1) as f64 * share) as usize];
    (mean, at(0.99), at(1.0))
}

/// The CPU and GPU renders of `picture`, or `None` without a GPU.
fn both(picture: &Picture, images: &[&LinearImage]) -> Option<(Vec<f32>, Vec<f32>)> {
    let context = GpuContext::new()?;
    let source = ImageSource::new(images.to_vec()).unwrap();
    let region = whole(images[0]);
    let options = RenderOptions {
        tile: 256,
        ..RenderOptions::default()
    };
    let cpu = render_image(picture, &source, region, &options)
        .unwrap()
        .data;
    let sink = ImageSink::new(region);
    render_gpu(picture, &source, region, &sink, &context, &options).unwrap();
    let gpu = sink.into_image().unwrap().data;
    for (a, b) in cpu.iter().zip(&gpu) {
        assert_eq!(
            a.is_nan(),
            b.is_nan(),
            "a pixel is covered in one render only"
        );
    }
    Some((cpu, gpu))
}

fn inputs() -> [LinearImage; 3] {
    let (width, height) = (520, 390);
    [
        plane(width, height, 1, true),
        plane(width, height, 2, false),
        plane(width, height, 3, false),
    ]
}

#[test]
fn layers_masks_heals_and_finishing_match_the_cpu() {
    let inputs = inputs();
    let images: Vec<&LinearImage> = inputs.iter().collect();
    let mut picture = everything();
    // Smoothing is the GPU's own approximation; the next test covers it.
    picture.computed[0].smoothing = 0.0;
    let Some((cpu, gpu)) = both(&picture, &images) else {
        eprintln!("no GPU: the GPU compositor was not checked");
        return;
    };
    let (mean, p99, largest) = gap(&cpu, &gpu);
    eprintln!("unsmoothed: mean {mean:e}, p99 {p99:e}, max {largest:e}");
    assert!(
        mean < 1e-3 && p99 < 5e-3,
        "mean {mean:e}, p99 {p99:e}, max {largest:e}"
    );
}

#[test]
fn smoothed_masks_stay_close_to_the_cpu() {
    let inputs = inputs();
    let images: Vec<&LinearImage> = inputs.iter().collect();
    let Some((cpu, gpu)) = both(&everything(), &images) else {
        eprintln!("no GPU: the GPU compositor was not checked");
        return;
    };
    let (mean, p99, largest) = gap(&cpu, &gpu);
    eprintln!("smoothed: mean {mean:e}, p99 {p99:e}, max {largest:e}");
    assert!(
        mean < 5e-3 && p99 < 3e-2,
        "mean {mean:e}, p99 {p99:e}, max {largest:e}"
    );
}

/// Reads through one transfer of one input share a stretch-table row, so
/// a mask that reads what its layers show takes no row of its own. A read
/// that finds no row left is named, not dropped without a word.
#[test]
fn reads_share_stretch_rows_and_one_without_a_row_is_named() {
    use seiza_compose::gpu::{GpuInput, GpuScene, LAYER_ROWS, MAX_LAYERS};
    use seiza_compose::picture::{ComputedMask, InputRead, Layer, Transfer, TransferStage};
    let read = |row: usize| InputRead::new(0, Transfer::of([TransferStage::Scale(row as f32)]));
    let place = [GpuInput {
        layer: 0,
        scale: 1.0,
    }];
    let mut picture = Picture {
        layers: (0..MAX_LAYERS)
            .map(|row| Layer::added(read(row), [1.0; 3]))
            .collect(),
        ..Picture::default()
    };
    // Every row is taken by a layer; a mask that reads two of them fits.
    picture
        .computed
        .push(ComputedMask::foraxx(vec![read(0), read(5)], 0.0));
    picture.layers[0].computed = Some(0);
    assert_eq!(
        GpuScene::new(&picture, &place).left_out,
        Vec::<String>::new()
    );
    // A mask through a transfer no layer uses needs a row of its own: the
    // rows left after the layers' take masks until none is left.
    let spare = LAYER_ROWS - MAX_LAYERS;
    for mask in 0..=spare / 2 {
        let first = MAX_LAYERS + 2 * mask;
        picture.computed.push(ComputedMask::foraxx(
            vec![read(first), read(first + 1)],
            0.0,
        ));
    }
    let left_out = GpuScene::new(&picture, &place).left_out;
    let named = format!("computed mask {}", spare / 2 + 1);
    assert!(
        left_out.iter().any(|line| line.contains(&named)),
        "{left_out:?}"
    );
}

/// Two Tone steps that read neighborhoods, each over the picture the one
/// before left: the GPU measures each in its own layer, as the CPU does.
#[test]
fn two_local_tone_steps_match_the_cpu() {
    use seiza_compose::picture::{FinishKind, FinishStep};
    use seiza_compose::tone::Tone;
    let inputs = inputs();
    let images: Vec<&LinearImage> = inputs.iter().collect();
    let mut picture = everything();
    picture.computed[0].smoothing = 0.0;
    picture.finish.push(FinishStep {
        kind: FinishKind::Tone(Tone {
            clarity: -0.6,
            highlights: 0.5,
            dehaze: -0.4,
            ..Tone::default()
        }),
        opacity: 0.8,
        mask: None,
        range: None,
    });
    let place: Vec<seiza_compose::gpu::GpuInput> = (0..3)
        .map(|layer| seiza_compose::gpu::GpuInput { layer, scale: 1.0 })
        .collect();
    assert_eq!(
        seiza_compose::gpu::GpuScene::new(&picture, &place).left_out,
        Vec::<String>::new()
    );
    let Some((cpu, gpu)) = both(&picture, &images) else {
        eprintln!("no GPU: the GPU compositor was not checked");
        return;
    };
    let (mean, p99, largest) = gap(&cpu, &gpu);
    eprintln!("two local steps: mean {mean:e}, p99 {p99:e}, max {largest:e}");
    assert!(
        mean < 1e-3 && p99 < 5e-3,
        "mean {mean:e}, p99 {p99:e}, max {largest:e}"
    );
}

/// A picture with headroom of its own makes its Tone steps HDR ones, on the
/// GPU as on the CPU.
#[test]
fn a_picture_s_headroom_makes_its_tone_hdr() {
    use seiza_compose::picture::FinishKind;
    let inputs = inputs();
    let images: Vec<&LinearImage> = inputs.iter().collect();
    let mut picture = everything();
    picture.computed[0].smoothing = 0.0;
    picture.headroom = 2.0;
    for step in &mut picture.finish {
        if let FinishKind::Tone(tone) = &mut step.kind {
            tone.hdr = false;
            tone.exposure = 1.5;
        }
    }
    let Some((cpu, gpu)) = both(&picture, &images) else {
        eprintln!("no GPU: the GPU compositor was not checked");
        return;
    };
    let above = cpu.iter().filter(|value| **value > 1.0).count();
    assert!(above > 100, "the picture reaches above SDR white: {above}");
    let (mean, p99, largest) = gap(&cpu, &gpu);
    eprintln!("picture headroom: mean {mean:e}, p99 {p99:e}, max {largest:e}");
    assert!(
        mean < 2e-3 && p99 < 1e-2,
        "mean {mean:e}, p99 {p99:e}, max {largest:e}"
    );
}

/// An HDR Tone step's highlights above SDR white are drawn on the GPU as
/// the CPU renders them.
#[test]
fn an_hdr_tone_step_matches_the_cpu() {
    use seiza_compose::picture::FinishKind;
    let inputs = inputs();
    let images: Vec<&LinearImage> = inputs.iter().collect();
    let mut picture = everything();
    picture.computed[0].smoothing = 0.0;
    for step in &mut picture.finish {
        if let FinishKind::Tone(tone) = &mut step.kind {
            tone.hdr = true;
            tone.headroom = 2.0;
            tone.exposure = 1.5;
        }
    }
    let Some((cpu, gpu)) = both(&picture, &images) else {
        eprintln!("no GPU: the GPU compositor was not checked");
        return;
    };
    let above = cpu.iter().filter(|value| **value > 1.0).count();
    // The picture keeps within its peak, on both; NaN marks a pixel the
    // picture does not cover.
    let peak = seiza_compose::tone::extended_srgb_encode(4.0);
    assert!(
        cpu.iter()
            .chain(&gpu)
            .filter(|value| !value.is_nan())
            .all(|value| *value <= peak + 1e-3)
    );
    assert!(above > 100, "the picture reaches above SDR white: {above}");
    let (mean, p99, largest) = gap(&cpu, &gpu);
    eprintln!("hdr: mean {mean:e}, p99 {p99:e}, max {largest:e}");
    assert!(
        mean < 2e-3 && p99 < 1e-2,
        "mean {mean:e}, p99 {p99:e}, max {largest:e}"
    );
}

/// A layer whose transfer is an HDR layered stretch, with its core
/// compression and contrast recovery reading maps of the whole picture, is
/// drawn on the GPU as the CPU renders it.
#[test]
fn a_layered_stretch_matches_the_cpu() {
    use seiza_compose::picture::{
        InputRead, Layer, LayerKind, LayeredStage, Transfer, TransferStage,
    };
    use seiza_stretch::{
        ColorStrategy, LayeredHdr, LayeredHdrParams, LayeredParams, ResolvedCurve, StretchConfig,
        StretchModel, reduce_luminance,
    };
    use std::sync::Arc;
    let inputs = inputs();
    let images: Vec<&LinearImage> = inputs.iter().collect();
    let image = &inputs[0];
    let config = StretchConfig {
        model: StretchModel::LayeredHdr(LayeredHdrParams {
            layered: LayeredParams {
                contrast_recovery: 0.6,
                scale_separation: 64,
                ..LayeredParams::default()
            },
            hdr: LayeredHdr {
                headroom: 2.0,
                core_threshold: 60.0,
                core_compression: 0.3,
                core_scale: 8.0,
                ..LayeredHdr::default()
            },
        }),
        color_strategy: ColorStrategy::Linked,
        max_analysis_samples: 200_000,
    };
    let finite: Vec<f32> = image
        .data
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    let plan = config.resolve_for(&finite, 1).unwrap();
    let ResolvedCurve::Layered(layered) = plan.curves()[0] else {
        panic!("not a layered curve");
    };
    let (reduced, width, height) = reduce_luminance(&image.data, image.width, image.height);
    let maps = Arc::new(layered.measure(&reduced, width, height));
    assert!(!maps.core.is_empty() && !maps.large.is_empty());
    let stage = LayeredStage::new(layered, maps);
    let picture = Picture {
        layers: vec![Layer::new(LayerKind::Input {
            read: InputRead::new(0, Transfer::of([TransferStage::Layered(stage)])),
            color: [1.0, 0.8, 0.6],
            curve: 0.0,
        })],
        headroom: 2.0,
        ..Picture::default()
    };
    let place: Vec<seiza_compose::gpu::GpuInput> = (0..3)
        .map(|layer| seiza_compose::gpu::GpuInput { layer, scale: 1.0 })
        .collect();
    assert_eq!(
        seiza_compose::gpu::GpuScene::new(&picture, &place).left_out,
        Vec::<String>::new()
    );
    let Some((cpu, gpu)) = both(&picture, &images) else {
        eprintln!("no GPU: the GPU compositor was not checked");
        return;
    };
    let above = cpu.iter().filter(|value| **value > 1.0).count();
    assert!(above > 50, "the stretch reaches above SDR white: {above}");
    let (mean, p99, largest) = gap(&cpu, &gpu);
    eprintln!("layered: mean {mean:e}, p99 {p99:e}, max {largest:e}");
    assert!(
        mean < 2e-3 && p99 < 1e-2,
        "mean {mean:e}, p99 {p99:e}, max {largest:e}"
    );
}

/// A stretch after a layered HDR one shapes the range up to SDR white and
/// keeps the headroom, on the GPU as on the CPU.
#[test]
fn a_stretch_after_a_layered_hdr_one_keeps_the_headroom_on_the_gpu() {
    use seiza_compose::picture::{
        InputRead, Layer, LayerKind, LayeredStage, Transfer, TransferStage,
    };
    use seiza_stretch::{
        ColorStrategy, LayeredHdr, LayeredHdrParams, LayeredParams, ResolvedCurve, StretchConfig,
        StretchModel, reduce_luminance,
    };
    use std::sync::Arc;
    let inputs = inputs();
    let images: Vec<&LinearImage> = inputs.iter().collect();
    let image = &inputs[0];
    let config = StretchConfig {
        model: StretchModel::LayeredHdr(LayeredHdrParams {
            layered: LayeredParams {
                contrast_recovery: 0.6,
                scale_separation: 64,
                ..LayeredParams::default()
            },
            hdr: LayeredHdr {
                headroom: 2.0,
                core_threshold: 60.0,
                core_compression: 0.3,
                core_scale: 8.0,
                ..LayeredHdr::default()
            },
        }),
        color_strategy: ColorStrategy::Linked,
        max_analysis_samples: 200_000,
    };
    let finite: Vec<f32> = image
        .data
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    let plan = config.resolve_for(&finite, 1).unwrap();
    let ResolvedCurve::Layered(layered) = plan.curves()[0] else {
        panic!("not a layered curve");
    };
    let (reduced, width, height) = reduce_luminance(&image.data, image.width, image.height);
    let maps = Arc::new(layered.measure(&reduced, width, height));
    assert!(!maps.core.is_empty() && !maps.large.is_empty());
    let stage = LayeredStage::new(layered, maps);
    let picture = Picture {
        layers: vec![Layer::new(LayerKind::Input {
            read: InputRead::new(
                0,
                Transfer::of([
                    TransferStage::Layered(stage),
                    // A midtones stretch after it, up to SDR white only.
                    TransferStage::BelowWhite(ResolvedCurve::Mtf {
                        shadows: 0.0,
                        midtone: 0.35,
                        highlights: 1.0,
                    }),
                ]),
            ),
            color: [1.0, 0.8, 0.6],
            curve: 0.0,
        })],
        headroom: 2.0,
        ..Picture::default()
    };
    let place: Vec<seiza_compose::gpu::GpuInput> = (0..3)
        .map(|layer| seiza_compose::gpu::GpuInput { layer, scale: 1.0 })
        .collect();
    assert_eq!(
        seiza_compose::gpu::GpuScene::new(&picture, &place).left_out,
        Vec::<String>::new()
    );
    let Some((cpu, gpu)) = both(&picture, &images) else {
        eprintln!("no GPU: the GPU compositor was not checked");
        return;
    };
    let above = cpu.iter().filter(|value| **value > 1.0).count();
    assert!(above > 50, "the stretch reaches above SDR white: {above}");
    let (mean, p99, largest) = gap(&cpu, &gpu);
    eprintln!("after layered: mean {mean:e}, p99 {p99:e}, max {largest:e}");
    assert!(
        mean < 2e-3 && p99 < 1e-2,
        "mean {mean:e}, p99 {p99:e}, max {largest:e}"
    );
}

#[test]
fn a_frame_draws_more_than_four_layered_stretches() {
    use seiza_compose::picture::{
        InputRead, Layer, LayerKind, LayeredStage, Transfer, TransferStage,
    };
    use seiza_stretch::{
        ColorStrategy, LayeredHdr, LayeredHdrParams, LayeredParams, ResolvedCurve, StretchConfig,
        StretchModel, reduce_luminance,
    };
    use std::sync::Arc;
    let inputs = inputs();
    let images: Vec<&LinearImage> = inputs.iter().collect();
    let image = &inputs[0];
    let config = StretchConfig {
        model: StretchModel::LayeredHdr(LayeredHdrParams {
            layered: LayeredParams {
                contrast_recovery: 0.6,
                scale_separation: 64,
                ..LayeredParams::default()
            },
            hdr: LayeredHdr {
                headroom: 2.0,
                core_threshold: 60.0,
                core_compression: 0.3,
                core_scale: 8.0,
                ..LayeredHdr::default()
            },
        }),
        color_strategy: ColorStrategy::Linked,
        max_analysis_samples: 200_000,
    };
    let finite: Vec<f32> = image
        .data
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    let plan = config.resolve_for(&finite, 1).unwrap();
    let ResolvedCurve::Layered(layered) = plan.curves()[0] else {
        panic!("not a layered curve");
    };
    let (reduced, width, height) = reduce_luminance(&image.data, image.width, image.height);
    let maps = Arc::new(layered.measure(&reduced, width, height));
    assert!(!maps.core.is_empty() && !maps.large.is_empty());
    let stage = LayeredStage::new(layered, maps);
    // Six bands, each with its layered stretch, as a narrowband picture
    // with RGB stars has.
    let picture = Picture {
        layers: (0..6)
            .map(|_| {
                Layer::new(LayerKind::Input {
                    read: InputRead::new(0, Transfer::of([TransferStage::Layered(stage.clone())])),
                    color: [1.0, 1.0, 1.0],
                    curve: 0.0,
                })
            })
            .collect(),
        headroom: 2.0,
        ..Picture::default()
    };
    let place: Vec<seiza_compose::gpu::GpuInput> = (0..3)
        .map(|layer| seiza_compose::gpu::GpuInput { layer, scale: 1.0 })
        .collect();
    assert_eq!(
        seiza_compose::gpu::GpuScene::new(&picture, &place).left_out,
        Vec::<String>::new()
    );
    let _ = images;
}

/// The same stretch with a sky floor: the sky goes to black on the GPU as
/// on the CPU, and the range above fills to the peak again.
#[test]
fn a_layered_stretch_with_a_sky_floor_matches_the_cpu() {
    use seiza_compose::picture::{
        InputRead, Layer, LayerKind, LayeredStage, Transfer, TransferStage,
    };
    use seiza_stretch::{
        ColorStrategy, LayeredHdr, LayeredHdrParams, LayeredParams, ResolvedCurve, StretchConfig,
        StretchModel, reduce_luminance,
    };
    use std::sync::Arc;
    let inputs = inputs();
    let images: Vec<&LinearImage> = inputs.iter().collect();
    let image = &inputs[0];
    let config = StretchConfig {
        model: StretchModel::LayeredHdr(LayeredHdrParams {
            layered: LayeredParams {
                contrast_recovery: 0.6,
                scale_separation: 64,
                sky_floor: 0.15,
                ..LayeredParams::default()
            },
            hdr: LayeredHdr {
                headroom: 2.0,
                core_threshold: 60.0,
                core_compression: 0.3,
                core_scale: 8.0,
                ..LayeredHdr::default()
            },
        }),
        color_strategy: ColorStrategy::Linked,
        max_analysis_samples: 200_000,
    };
    let finite: Vec<f32> = image
        .data
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    let plan = config.resolve_for(&finite, 1).unwrap();
    let ResolvedCurve::Layered(layered) = plan.curves()[0] else {
        panic!("not a layered curve");
    };
    let (reduced, width, height) = reduce_luminance(&image.data, image.width, image.height);
    let maps = Arc::new(layered.measure(&reduced, width, height));
    assert!(!maps.core.is_empty() && !maps.large.is_empty());
    let stage = LayeredStage::new(layered, maps);
    let picture = Picture {
        layers: vec![Layer::new(LayerKind::Input {
            read: InputRead::new(0, Transfer::of([TransferStage::Layered(stage)])),
            color: [1.0, 0.8, 0.6],
            curve: 0.0,
        })],
        headroom: 2.0,
        ..Picture::default()
    };
    let place: Vec<seiza_compose::gpu::GpuInput> = (0..3)
        .map(|layer| seiza_compose::gpu::GpuInput { layer, scale: 1.0 })
        .collect();
    assert_eq!(
        seiza_compose::gpu::GpuScene::new(&picture, &place).left_out,
        Vec::<String>::new()
    );
    let Some((cpu, gpu)) = both(&picture, &images) else {
        eprintln!("no GPU: the GPU compositor was not checked");
        return;
    };
    let above = cpu.iter().filter(|value| **value > 1.0).count();
    assert!(above > 50, "the stretch reaches above SDR white: {above}");
    // The sky goes to black under the floor.
    let black = cpu.iter().filter(|value| **value == 0.0).count();
    assert!(
        black > cpu.len() / 10,
        "the floor takes the sky to black: {black}"
    );
    let (mean, p99, largest) = gap(&cpu, &gpu);
    eprintln!("layered with a sky floor: mean {mean:e}, p99 {p99:e}, max {largest:e}");
    assert!(
        mean < 2e-3 && p99 < 1e-2,
        "mean {mean:e}, p99 {p99:e}, max {largest:e}"
    );
}

/// A layered stretch that follows another input's luminance and compresses
/// star cores against that luminance's fine blur draws on the GPU as the
/// CPU renders it.
#[test]
fn a_layered_stretch_with_guides_matches_the_cpu() {
    use seiza_compose::picture::{
        InputRead, Layer, LayerKind, LayeredStage, Transfer, TransferStage,
    };
    use seiza_stretch::{
        ColorStrategy, LayeredParams, ResolvedCurve, StretchConfig, StretchModel, fine_luminance,
        reduce_luminance,
    };
    use std::sync::Arc;
    let [first, luminance, third] = inputs();
    let fine = LinearImage::new(
        luminance.width,
        luminance.height,
        1,
        fine_luminance(&luminance.data, luminance.width, luminance.height),
    )
    .unwrap();
    let all = [first, luminance, third, fine];
    let images: Vec<&LinearImage> = all.iter().collect();
    let guide = &all[1];
    let config = StretchConfig {
        model: StretchModel::Layered(LayeredParams {
            dynamic_range_compression: 0.8,
            contrast_recovery: 0.4,
            scale_separation: 64,
            ..LayeredParams::default()
        }),
        color_strategy: ColorStrategy::Linked,
        max_analysis_samples: 200_000,
    };
    let finite: Vec<f32> = guide
        .data
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    let plan = config.resolve_for(&finite, 1).unwrap();
    let ResolvedCurve::Layered(layered) = plan.curves()[0] else {
        panic!("not a layered curve");
    };
    assert!(layered.compresses_star_cores());
    let (reduced, width, height) = reduce_luminance(&guide.data, guide.width, guide.height);
    let maps = Arc::new(layered.measure(&reduced, width, height));
    let stage = LayeredStage::new(layered, maps).with_guides(Some(1), Some(3));
    let picture = Picture {
        layers: vec![Layer::new(LayerKind::Input {
            read: InputRead::new(0, Transfer::of([TransferStage::Layered(stage)])),
            color: [1.0, 0.0, 0.0],
            curve: 0.0,
        })],
        ..Picture::default()
    };
    assert_eq!(picture.inputs(), vec![0, 1, 3]);
    let Some((cpu, gpu)) = both(&picture, &images) else {
        eprintln!("no GPU: the GPU compositor was not checked");
        return;
    };
    let (mean, p99, largest) = gap(&cpu, &gpu);
    eprintln!("guided: mean {mean:e}, p99 {p99:e}, max {largest:e}");
    // No speck where the luminance is at the black point: a channel that
    // follows it divides by at least one noise sigma.
    assert!(largest < 0.1, "max {largest:e}");
    assert!(
        mean < 2e-3 && p99 < 1e-2,
        "mean {mean:e}, p99 {p99:e}, max {largest:e}"
    );
}

/// A screened layer over a picture brighter than SDR white keeps what is
/// beneath in order, up to the picture's peak, on the GPU as on the CPU:
/// stars screened over an HDR core leave the core's detail.
#[test]
fn screen_keeps_values_above_sdr_white_in_order() {
    use seiza_compose::picture::{Blend, InputRead, Layer, LayerKind, Transfer, TransferStage};
    use seiza_stretch::ResolvedCurve;
    let inputs = inputs();
    let images: Vec<&LinearImage> = inputs.iter().collect();
    let curve = ResolvedCurve::Mtf {
        shadows: 0.0,
        midtone: 0.3,
        highlights: 1.0,
    };
    let read = |input: usize| InputRead::new(input, Transfer::of([TransferStage::Curve(curve)]));
    // A bright layer, three times SDR white at most, then a dim one screened
    // over it.
    let mut bright = Layer::new(LayerKind::Input {
        read: read(0),
        color: [3.0, 3.0, 3.0],
        curve: 0.0,
    });
    bright.blend = Blend::Add;
    let mut stars = Layer::new(LayerKind::Input {
        read: read(1),
        color: [0.5, 0.5, 0.5],
        curve: 0.0,
    });
    stars.blend = Blend::Screen;
    let mut picture = Picture {
        layers: vec![bright.clone()],
        headroom: 2.0,
        ..Picture::default()
    };
    let peak = picture.peak();
    let source = seiza_compose::render::ImageSource::new(images.clone()).unwrap();
    let region = whole(images[0]);
    let alone = render_image(&picture, &source, region, &RenderOptions::default())
        .unwrap()
        .data;
    picture.layers.push(stars);
    let Some((cpu, gpu)) = both(&picture, &images) else {
        eprintln!("no GPU: the GPU compositor was not checked");
        return;
    };
    // Where the layer beneath is above SDR white and below the peak, the
    // screened picture is above it too, and never below what was beneath.
    let mut above = 0;
    for (before, after) in alone.iter().zip(&cpu) {
        if before.is_nan() {
            continue;
        }
        let before = before.min(peak);
        assert!(
            *after >= before - 1e-4,
            "screen darkened {before} to {after}"
        );
        if before > 1.05 && before < peak - 0.05 {
            above += 1;
            assert!(*after > 1.0, "{before} above SDR white became {after}");
        }
    }
    assert!(above > 100, "the picture reaches above SDR white: {above}");
    let (mean, p99, largest) = gap(&cpu, &gpu);
    eprintln!("hdr screen: mean {mean:e}, p99 {p99:e}, max {largest:e}");
    assert!(
        mean < 2e-3 && p99 < 1e-2,
        "mean {mean:e}, p99 {p99:e}, max {largest:e}"
    );
}

/// Steps in the stack that map each pixel alone are drawn by the layers
/// pass, as the CPU draws them.
#[test]
fn steps_in_the_stack_match_the_cpu() {
    use seiza_compose::picture::{
        FinishKind, InputRead, Layer, LayerKind, Transfer, TransferStage,
    };
    use seiza_compose::tone::Tone;
    use seiza_stretch::ResolvedCurve;
    let inputs = inputs();
    let images: Vec<&LinearImage> = inputs.iter().collect();
    let curve = ResolvedCurve::Mtf {
        shadows: 0.0,
        midtone: 0.3,
        highlights: 1.0,
    };
    let read = |input: usize| InputRead::new(input, Transfer::of([TransferStage::Curve(curve)]));
    let step = |kind| Layer::new(LayerKind::Step(kind));
    let picture = Picture {
        layers: vec![
            Layer::added(read(0), [1.0, 0.2, 0.1]),
            step(FinishKind::Curves(0.5)),
            Layer::added(read(1), [0.1, 0.9, 0.3]),
            Layer {
                opacity: 0.6,
                ..step(FinishKind::Saturation(0.2))
            },
            Layer {
                opacity: 0.8,
                ..Layer::new(LayerKind::Adjustment(
                    seiza_compose::picture::Adjustment::GreenReduction(0.7),
                ))
            },
            Layer::new(LayerKind::GroupBegin),
            Layer::added(read(2), [0.2, 0.1, 0.8]),
            step(FinishKind::Tone(Tone {
                exposure: 0.4,
                contrast: 0.3,
                vibrance: 0.2,
                ..Tone::default()
            })),
            Layer {
                blend: seiza_compose::picture::Blend::Screen,
                ..Layer::new(LayerKind::GroupEnd)
            },
        ],
        ..Picture::default()
    };
    let Some((cpu, gpu)) = both(&picture, &images) else {
        eprintln!("no GPU: the GPU compositor was not checked");
        return;
    };
    let (mean, p99, largest) = gap(&cpu, &gpu);
    eprintln!("steps in the stack: mean {mean:e}, p99 {p99:e}, max {largest:e}");
    assert!(
        mean < 1e-3 && p99 < 5e-3,
        "mean {mean:e}, p99 {p99:e}, max {largest:e}"
    );
}

/// Steps in the stack that read neighbors, heals and Tone steps with local
/// controls, are drawn in stages: the layers beneath each first, the step on
/// what they made, and the layers above from there; inside a group, on what
/// its members made, with the picture beneath the group kept for its end.
/// The GPU then matches the CPU, which works the same way.
#[test]
fn steps_that_read_neighbors_match_the_cpu_in_the_stack() {
    use seiza_compose::gpu::{GpuInput, GpuScene};
    use seiza_compose::picture::{
        FinishKind, InputRead, Layer, LayerKind, Transfer, TransferStage,
    };
    use seiza_compose::tone::Tone;
    use seiza_stretch::ResolvedCurve;
    let inputs = inputs();
    let images: Vec<&LinearImage> = inputs.iter().collect();
    let curve = ResolvedCurve::Mtf {
        shadows: 0.0,
        midtone: 0.3,
        highlights: 1.0,
    };
    let read = |input: usize| InputRead::new(input, Transfer::of([TransferStage::Curve(curve)]));
    let step = |kind| Layer::new(LayerKind::Step(kind));
    let picture = Picture {
        layers: vec![
            Layer::added(read(0), [1.0, 0.2, 0.1]),
            step(FinishKind::Tone(Tone {
                clarity: 0.7,
                texture: 0.4,
                dehaze: 0.3,
                ..Tone::default()
            })),
            Layer::added(read(1), [0.1, 0.9, 0.3]),
            Layer {
                mask: Some(0),
                ..step(FinishKind::Heal {
                    offset: [21, -13],
                    shift: [0.0; 3],
                })
            },
            Layer::added(read(2), [0.2, 0.1, 0.8]),
            Layer {
                opacity: 0.7,
                ..step(FinishKind::Tone(Tone {
                    clarity: -0.4,
                    highlights: 0.3,
                    ..Tone::default()
                }))
            },
            step(FinishKind::Curves(0.3)),
            // Inside a group the steps work on its members alone, and the
            // group is then screened over what is beneath it.
            Layer::new(LayerKind::GroupBegin),
            Layer::added(read(1), [0.6, 0.6, 0.6]),
            Layer {
                mask: Some(1),
                ..step(FinishKind::Heal {
                    offset: [-17, 24],
                    shift: [0.0; 3],
                })
            },
            // A group in the group, with a heal limited by a clip and a
            // computed mask, and a local Tone step at half strength.
            Layer::new(LayerKind::GroupBegin),
            Layer::added(read(2), [0.3, 0.5, 0.9]),
            Layer {
                mask: Some(2),
                clip: Some(1),
                computed: Some(0),
                ..step(FinishKind::Heal {
                    offset: [9, -14],
                    shift: [0.02, 0.0, -0.01],
                })
            },
            Layer {
                opacity: 0.5,
                computed: Some(0),
                ..step(FinishKind::Tone(Tone {
                    clarity: 0.6,
                    ..Tone::default()
                }))
            },
            Layer {
                blend: seiza_compose::picture::Blend::Add,
                ..Layer::new(LayerKind::GroupEnd)
            },
            step(FinishKind::Tone(Tone {
                clarity: 0.5,
                ..Tone::default()
            })),
            Layer::added(read(0), [0.2, 0.2, 0.2]),
            Layer {
                blend: seiza_compose::picture::Blend::Screen,
                opacity: 0.8,
                ..Layer::new(LayerKind::GroupEnd)
            },
        ],
        computed: vec![seiza_compose::picture::ComputedMask::foraxx(
            vec![read(0), read(2)],
            0.0,
        )],
        masks: vec![
            common::stroke([60.0, 50.0], [150.0, 120.0], 18.0),
            common::stroke([300.0, 200.0], [380.0, 260.0], 22.0),
            common::stroke([200.0, 150.0], [320.0, 230.0], 30.0),
        ],
        ..Picture::default()
    };
    let places: Vec<GpuInput> = (0..3).map(|layer| GpuInput { layer, scale: 1.0 }).collect();
    assert_eq!(
        GpuScene::new(&picture, &places).left_out,
        Vec::<String>::new()
    );
    let Some((cpu, gpu)) = both(&picture, &images) else {
        eprintln!("no GPU: the GPU compositor was not checked");
        return;
    };
    let (mean, p99, largest) = gap(&cpu, &gpu);
    eprintln!("neighbors in the stack: mean {mean:e}, p99 {p99:e}, max {largest:e}");
    assert!(
        mean < 1e-3 && p99 < 5e-3,
        "mean {mean:e}, p99 {p99:e}, max {largest:e}"
    );
}

/// Range masks by brightness, on an input layer (its value), an adjustment
/// and a step in the stack (the picture beneath), a group (what its members
/// made), a step that reads neighbors, and a Finish step: the GPU weighs
/// each as the CPU does.
#[test]
fn range_masks_match_the_cpu() {
    use seiza_compose::picture::{
        Adjustment, FinishKind, FinishStep, InputRead, Layer, LayerKind, RangeMask, Transfer,
        TransferStage,
    };
    use seiza_compose::tone::Tone;
    use seiza_stretch::ResolvedCurve;
    let inputs = inputs();
    let images: Vec<&LinearImage> = inputs.iter().collect();
    let curve = ResolvedCurve::Mtf {
        shadows: 0.0,
        midtone: 0.3,
        highlights: 1.0,
    };
    let read = |input: usize| InputRead::new(input, Transfer::of([TransferStage::Curve(curve)]));
    let range = |low, high, feather| Some(RangeMask { low, high, feather });
    let picture = Picture {
        layers: vec![
            Layer::added(read(0), [1.0, 0.2, 0.1]),
            Layer {
                range: range(0.3, 0.8, 0.1),
                ..Layer::added(read(1), [0.1, 0.9, 0.3])
            },
            Layer {
                range: range(0.0, 0.35, 0.15),
                ..Layer::new(LayerKind::Adjustment(Adjustment::Exposure(0.8)))
            },
            Layer {
                range: range(0.4, 2.0, 0.0),
                ..Layer::new(LayerKind::Step(FinishKind::Curves(0.6)))
            },
            Layer::new(LayerKind::GroupBegin),
            Layer::added(read(2), [0.2, 0.1, 0.8]),
            Layer {
                blend: seiza_compose::picture::Blend::Screen,
                range: range(0.1, 0.5, 0.2),
                ..Layer::new(LayerKind::GroupEnd)
            },
            Layer {
                range: range(0.2, 0.6, 0.1),
                ..Layer::new(LayerKind::Step(FinishKind::Tone(Tone {
                    clarity: 0.5,
                    ..Tone::default()
                })))
            },
        ],
        finish: vec![FinishStep {
            kind: FinishKind::Saturation(1.8),
            opacity: 1.0,
            mask: None,
            range: range(0.5, 1.0, 0.25),
        }],
        ..Picture::default()
    };
    let Some((cpu, gpu)) = both(&picture, &images) else {
        eprintln!("no GPU: the GPU compositor was not checked");
        return;
    };
    let (mean, p99, largest) = gap(&cpu, &gpu);
    eprintln!("range masks: mean {mean:e}, p99 {p99:e}, max {largest:e}");
    assert!(
        mean < 1e-3 && p99 < 5e-3,
        "mean {mean:e}, p99 {p99:e}, max {largest:e}"
    );
}

/// Core compression before an ordinary curve: the GPU compresses with the
/// map as the CPU does, then reads the curve from the row after it.
#[test]
fn core_compression_before_any_curve_matches_the_cpu() {
    use seiza_compose::picture::{CoreStage, InputRead, Layer, LayerKind, Transfer, TransferStage};
    use seiza_stretch::{
        CoreCompression, ResolvedCore, ResolvedCurve, StretchAnalysis, reduce_luminance,
    };
    use std::sync::Arc;
    let inputs = inputs();
    let images: Vec<&LinearImage> = inputs.iter().collect();
    let image = &inputs[0];
    let finite: Vec<f32> = image
        .data
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    let analysis = StretchAnalysis::analyze(&finite, 1, 200_000).unwrap();
    let core = ResolvedCore::from_analysis(
        CoreCompression {
            amount: 0.6,
            threshold: 20.0,
            scale: 8.0,
        },
        &analysis,
    )
    .unwrap();
    let (reduced, width, height) = reduce_luminance(&image.data, image.width, image.height);
    let maps = Arc::new(core.measure(&reduced, width, height));
    assert!(!maps.core.is_empty());
    let curve = ResolvedCurve::Mtf {
        shadows: 0.0,
        midtone: 0.2,
        highlights: 1.0,
    };
    let compressed = Transfer::of([
        TransferStage::Core(CoreStage::new(core, maps)),
        TransferStage::Curve(curve),
    ]);
    let picture = Picture {
        layers: vec![Layer::new(LayerKind::Input {
            read: InputRead::new(0, compressed),
            color: [1.0, 0.8, 0.6],
            curve: 0.0,
        })],
        ..Picture::default()
    };
    let place: Vec<seiza_compose::gpu::GpuInput> = (0..3)
        .map(|layer| seiza_compose::gpu::GpuInput { layer, scale: 1.0 })
        .collect();
    assert_eq!(
        seiza_compose::gpu::GpuScene::new(&picture, &place).left_out,
        Vec::<String>::new()
    );
    // The compression changes the picture: the plain curve gives more.
    let plain = Picture {
        layers: vec![Layer::new(LayerKind::Input {
            read: InputRead::new(0, Transfer::of([TransferStage::Curve(curve)])),
            color: [1.0, 0.8, 0.6],
            curve: 0.0,
        })],
        ..Picture::default()
    };
    let source = ImageSource::new(images.clone()).unwrap();
    let options = RenderOptions::default();
    let with = render_image(&picture, &source, whole(image), &options)
        .unwrap()
        .data;
    let without = render_image(&plain, &source, whole(image), &options)
        .unwrap()
        .data;
    let lowered = with
        .iter()
        .zip(&without)
        .filter(|(a, b)| a.is_finite() && b.is_finite() && **b - **a > 1e-3)
        .count();
    assert!(lowered > 50, "core compression lowered {lowered} samples");
    let Some((cpu, gpu)) = both(&picture, &images) else {
        eprintln!("no GPU: the GPU compositor was not checked");
        return;
    };
    let (mean, p99, largest) = gap(&cpu, &gpu);
    eprintln!("core compression: mean {mean:e}, p99 {p99:e}, max {largest:e}");
    assert!(
        mean < 2e-3 && p99 < 1e-2,
        "mean {mean:e}, p99 {p99:e}, max {largest:e}"
    );
}
