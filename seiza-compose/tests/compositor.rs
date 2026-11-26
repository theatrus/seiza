//! The compositor renders a picture the same in any tiling, and draws what
//! its layers, masks, heals and finishing steps describe.

use seiza_compose::picture::{
    Adjustment, Blend, ComputedMask, FinishKind, FinishStep, InputRead, Layer, LayerHeal,
    LayerKind, LuminanceTarget, Picture, Segment, Transfer, TransferStage,
};
use seiza_compose::render::{ImageSource, RenderOptions, render_image};
use seiza_compose::tone::Tone;
use seiza_stacking::{LinearImage, ReferenceRegion};
use seiza_stretch::{ResolvedCurve, StretchConfig, StretchModel};

/// A plane of a faint sky with structure, stars and, with `gap`, one
/// uncovered pixel.
fn plane(width: usize, height: usize, seed: u32, gap: bool) -> LinearImage {
    let mut data: Vec<f32> = (0..width * height)
        .map(|index| {
            let (x, y) = ((index % width) as f32, (index / width) as f32);
            let mixed = (index as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed);
            let noise = (mixed % 1000) as f32 / 1000.0 - 0.5;
            let nebula = (-((x - width as f32 * 0.4).powi(2) + (y - height as f32 * 0.6).powi(2))
                / (3000.0 + seed as f32 * 500.0))
                .exp();
            0.05 + 0.5 * nebula + 0.1 * (0.5 + 0.5 * (x * 0.05 + seed as f32).sin()) + 0.01 * noise
        })
        .collect();
    for star in 0..40 {
        let at = ((star * 7919 + seed as usize * 131) % (width * height)).max(1);
        data[at] = 0.98;
    }
    if gap {
        data[width * 3 + 5] = f32::NAN;
    }
    LinearImage::new(width, height, 1, data).unwrap()
}

fn whole(image: &LinearImage) -> ReferenceRegion {
    ReferenceRegion {
        x: 0,
        y: 0,
        width: image.width,
        height: image.height,
    }
}

fn stroke(from: [f32; 2], to: [f32; 2], radius: f32) -> Vec<Segment> {
    vec![Segment {
        from,
        to,
        radius,
        hardness: 0.4,
        value: 1.0,
    }]
}

/// A picture with every kind of entry the compositor draws.
fn everything() -> Picture {
    let curve = ResolvedCurve::Mtf {
        shadows: 0.0,
        midtone: 0.3,
        highlights: 1.0,
    };
    let stretched =
        |input: usize| InputRead::new(input, Transfer::of([TransferStage::Curve(curve)]));
    let plan = StretchConfig {
        model: StretchModel::Mtf {
            shadows: 0.02,
            midtone: 0.4,
            highlights: 1.0,
        },
        color_strategy: Default::default(),
        max_analysis_samples: 10_000,
    }
    .resolve_explicit(3)
    .expect("an explicit stretch resolves");
    let mut finish = vec![FinishStep {
        kind: FinishKind::Heal {
            offset: [17, -9],
            shift: [0.01, 0.0, -0.01],
        },
        opacity: 1.0,
        mask: Some(1),
    }];
    finish.push(FinishStep {
        kind: FinishKind::Stretch(plan),
        opacity: 0.5,
        mask: None,
    });
    finish.push(FinishStep {
        kind: FinishKind::Tone(Tone {
            contrast: 0.3,
            shadows: 0.4,
            vibrance: 0.3,
            ..Tone::default()
        }),
        opacity: 1.0,
        mask: None,
    });
    Picture {
        layers: vec![
            Layer::added(stretched(0), [1.0, 0.2, 0.1]),
            Layer {
                computed: Some(0),
                ..Layer::added(stretched(1), [0.1, 0.8, 0.9])
            },
            Layer::new(LayerKind::GroupBegin),
            Layer {
                blend: Blend::Screen,
                ..Layer::new(LayerKind::Input {
                    read: stretched(2),
                    color: [0.9, 0.3, 0.2],
                    curve: 0.2,
                })
            },
            Layer {
                own: true,
                ..Layer::new(LayerKind::Adjustment(Adjustment::Exposure(0.3)))
            },
            Layer {
                opacity: 0.7,
                blend: Blend::Lighten,
                mask: Some(0),
                ..Layer::new(LayerKind::GroupEnd)
            },
            Layer::new(LayerKind::Adjustment(Adjustment::Saturation(1.2))),
            Layer {
                opacity: 0.4,
                ..Layer::new(LayerKind::Luminance {
                    read: Some(stretched(0)),
                    target: LuminanceTarget::Replace,
                })
            },
        ],
        computed: vec![ComputedMask::foraxx(vec![stretched(0), stretched(1)], 4.0)],
        masks: vec![
            stroke([200.0, 150.0], [420.0, 260.0], 60.0),
            stroke([300.0, 300.0], [330.0, 310.0], 25.0),
            stroke([90.0, 90.0], [140.0, 95.0], 20.0),
        ],
        heals: vec![
            LayerHeal {
                layer: 0,
                before_transfer: true,
                offset: [-23, 11],
                shift: 0.0,
                mask: 2,
            },
            LayerHeal {
                layer: 3,
                before_transfer: false,
                offset: [14, 5],
                shift: 0.02,
                mask: 0,
            },
        ],
        finish,
    }
}

fn render(
    picture: &Picture,
    images: &[&LinearImage],
    region: ReferenceRegion,
    tile: usize,
) -> Vec<f32> {
    let source = ImageSource::new(images.to_vec()).unwrap();
    render_image(
        picture,
        &source,
        region,
        &RenderOptions {
            tile,
            ..RenderOptions::default()
        },
    )
    .unwrap()
    .data
}

#[test]
fn any_tiling_gives_the_same_pixels() {
    let (width, height) = (520, 390);
    let inputs = [
        plane(width, height, 1, true),
        plane(width, height, 2, false),
        plane(width, height, 3, false),
    ];
    let images: Vec<&LinearImage> = inputs.iter().collect();
    let picture = everything();
    let one = render(&picture, &images, whole(&inputs[0]), 4096);
    for tile in [48, 128, 300] {
        let tiled = render(&picture, &images, whole(&inputs[0]), tile);
        let same = one
            .iter()
            .zip(&tiled)
            .all(|(a, b)| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()));
        assert!(same, "tiles of {tile} differ from one tile");
    }
    // A region inside the inputs reflects at its own edge, in any tiling.
    let inner = ReferenceRegion {
        x: 31,
        y: 17,
        width: 400,
        height: 300,
    };
    let one = render(&picture, &images, inner, 4096);
    let tiled = render(&picture, &images, inner, 64);
    assert!(
        one.iter()
            .zip(&tiled)
            .all(|(a, b)| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()))
    );
}

#[test]
fn an_uncovered_input_leaves_its_pixel_uncovered() {
    let (width, height) = (40, 30);
    let inputs = [
        plane(width, height, 1, true),
        plane(width, height, 2, false),
    ];
    let picture = Picture {
        layers: vec![
            Layer::added(InputRead::new(0, Transfer::identity()), [1.0, 0.0, 0.0]),
            Layer::added(InputRead::new(1, Transfer::identity()), [0.0, 1.0, 0.0]),
        ],
        ..Picture::default()
    };
    let out = render(&picture, &[&inputs[0], &inputs[1]], whole(&inputs[0]), 512);
    let gap = (3 * width + 5) * 3;
    assert!(out[gap..gap + 3].iter().all(|value| value.is_nan()));
    assert!(out[..gap].iter().all(|value| value.is_finite()));
}

#[test]
fn a_painted_mask_limits_its_layer() {
    let (width, height) = (100, 100);
    let inputs = [
        plane(width, height, 1, false),
        plane(width, height, 2, false),
    ];
    let picture = Picture {
        layers: vec![
            Layer::new(LayerKind::Input {
                read: InputRead::new(0, Transfer::identity()),
                color: [1.0, 0.0, 0.0],
                curve: 0.0,
            }),
            Layer {
                mask: Some(0),
                ..Layer::new(LayerKind::Input {
                    read: InputRead::new(1, Transfer::identity()),
                    color: [0.0, 1.0, 0.0],
                    curve: 0.0,
                })
            },
        ],
        masks: vec![vec![Segment {
            from: [50.0, 50.0],
            to: [50.0, 50.0],
            radius: 10.0,
            hardness: 1.0,
            value: 1.0,
        }]],
        ..Picture::default()
    };
    let out = render(&picture, &[&inputs[0], &inputs[1]], whole(&inputs[0]), 512);
    let at = |x: usize, y: usize| &out[(y * width + x) * 3..(y * width + x) * 3 + 3];
    assert_eq!(at(50, 50)[0], 0.0);
    assert!(at(50, 50)[1] > 0.0);
    assert!(at(10, 10)[0] > 0.0);
    assert_eq!(at(10, 10)[1], 0.0);
}

#[test]
fn a_stop_ends_the_render() {
    let image = plane(300, 200, 1, false);
    let picture = Picture {
        layers: vec![Layer::added(
            InputRead::new(0, Transfer::identity()),
            [1.0; 3],
        )],
        ..Picture::default()
    };
    let source = ImageSource::new(vec![&image]).unwrap();
    let stopped = render_image(
        &picture,
        &source,
        whole(&image),
        &RenderOptions {
            tile: 64,
            stop: Some(&|| true),
            progress: None,
        },
    );
    assert!(stopped.is_err());
}

#[test]
fn a_region_past_the_inputs_is_refused() {
    let image = plane(30, 20, 1, false);
    let picture = Picture::default();
    let source = ImageSource::new(vec![&image]).unwrap();
    let region = ReferenceRegion {
        x: 10,
        y: 0,
        width: 30,
        height: 20,
    };
    assert!(render_image(&picture, &source, region, &RenderOptions::default()).is_err());
}
