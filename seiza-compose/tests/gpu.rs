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
    use seiza_compose::gpu::{GpuInput, GpuScene, MAX_LAYERS};
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
    // A mask through a transfer no layer uses needs a row of its own.
    picture
        .computed
        .push(ComputedMask::foraxx(vec![read(MAX_LAYERS)], 0.0));
    let left_out = GpuScene::new(&picture, &place).left_out;
    assert_eq!(left_out.len(), 1, "{left_out:?}");
    assert!(left_out[0].contains("computed mask 1"), "{left_out:?}");
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
