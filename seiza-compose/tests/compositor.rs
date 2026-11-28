//! The compositor renders a picture the same in any tiling, and draws what
//! its layers, masks, heals and finishing steps describe.

mod common;

use common::{everything, plane, whole};
use seiza_compose::picture::{InputRead, Layer, LayerKind, Picture, Segment, Transfer};
use seiza_compose::render::{ImageSource, RenderOptions, render_image};
use seiza_stacking::{LinearImage, ReferenceRegion};

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

/// A Tone step that reads neighborhoods gives the same pixels in any
/// tiling: each tile reads far enough around it. Nothing else in this
/// picture reads beyond a pixel, so its margin is the Tone step's alone.
#[test]
fn a_tone_step_s_neighborhoods_do_not_depend_on_tiling() {
    use seiza_compose::picture::{FinishKind, FinishStep};
    use seiza_compose::tone::Tone;
    let (width, height) = (300, 220);
    let input = plane(width, height, 4, false);
    let picture = Picture {
        layers: vec![Layer::added(
            InputRead::new(
                0,
                Transfer::of([seiza_compose::picture::TransferStage::Scale(20.0)]),
            ),
            [1.0, 0.9, 0.8],
        )],
        finish: vec![FinishStep {
            kind: FinishKind::Tone(Tone {
                shadows: 0.6,
                clarity: 0.8,
                texture: 0.5,
                dehaze: 0.4,
                ..Tone::default()
            }),
            opacity: 1.0,
            mask: None,
            range: None,
        }],
        ..Picture::default()
    };
    let mut twice = picture.clone();
    // A second such step reads around a picture the first read around.
    twice.finish.push(FinishStep {
        kind: FinishKind::Tone(Tone {
            clarity: -0.5,
            highlights: 0.4,
            ..Tone::default()
        }),
        opacity: 1.0,
        mask: None,
        range: None,
    });
    for picture in [&picture, &twice] {
        let one = render(picture, &[&input], whole(&input), 4096);
        for tile in [32, 100] {
            let tiled = render(picture, &[&input], whole(&input), tile);
            assert!(
                one.iter()
                    .zip(&tiled)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "tiles of {tile} differ from one tile with {} steps",
                picture.finish.len()
            );
        }
    }
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

/// A mask painted on its own is the mask a layer is limited by: where a
/// masked layer shows, the painted mask is 1, and it is 0 where it does not.
#[test]
fn a_painted_mask_on_its_own_is_the_layer_s_mask() {
    use seiza_compose::render::paint_mask;
    let stroke = [Segment {
        from: [30.0, 40.0],
        to: [70.0, 40.0],
        radius: 8.0,
        hardness: 1.0,
        value: 1.0,
    }];
    let region = ReferenceRegion {
        x: 20,
        y: 30,
        width: 60,
        height: 20,
    };
    let mask = paint_mask(&stroke, region);
    assert_eq!(mask.len(), 60 * 20);
    // On the stroke's line, inside it: covered. Its region's corner, more
    // than a radius away: not.
    assert!(mask[10 * 60 + 30] > 0.99, "{}", mask[10 * 60 + 30]);
    assert_eq!(mask[0], 0.0);
    // The same pixels as the compositor's: a picture of the mask alone.
    let input = plane(100, 100, 1, false);
    let picture = Picture {
        layers: vec![Layer {
            mask: Some(0),
            ..Layer::new(LayerKind::Solid { color: [1.0; 3] })
        }],
        masks: vec![stroke.to_vec()],
        ..Picture::default()
    };
    let rendered = render(&picture, &[&input], region, 4096);
    for (at, cover) in mask.iter().enumerate() {
        assert!((rendered[at * 3] - cover).abs() < 1e-6, "pixel {at}");
    }
}

/// Three planes and a read of each through a midtones curve.
fn stacked_inputs() -> ([LinearImage; 3], impl Fn(usize) -> InputRead) {
    use seiza_compose::picture::TransferStage;
    use seiza_stretch::ResolvedCurve;
    let (width, height) = (240, 180);
    let curve = ResolvedCurve::Mtf {
        shadows: 0.0,
        midtone: 0.3,
        highlights: 1.0,
    };
    (
        [
            plane(width, height, 1, false),
            plane(width, height, 2, false),
            plane(width, height, 3, false),
        ],
        move |input: usize| InputRead::new(input, Transfer::of([TransferStage::Curve(curve)])),
    )
}

fn close(a: &[f32], b: &[f32], tolerance: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| (a - b).abs() <= tolerance)
}

fn step(kind: seiza_compose::picture::FinishKind) -> Layer {
    Layer::new(LayerKind::Step(kind))
}

/// Steps at the top of the stack do what the same steps in the Finish stage
/// do, sample for sample: the stack is the one place to adjust.
#[test]
fn steps_on_top_of_the_stack_are_finish_steps() {
    use seiza_compose::picture::{FinishKind, FinishStep};
    use seiza_compose::tone::Tone;
    let (inputs, read) = stacked_inputs();
    let images: Vec<&LinearImage> = inputs.iter().collect();
    let kinds = [
        FinishKind::Curves(0.4),
        FinishKind::Tone(Tone {
            shadows: 0.5,
            clarity: 0.6,
            texture: 0.3,
            dehaze: 0.2,
            ..Tone::default()
        }),
        FinishKind::Saturation(1.3),
    ];
    let layers = vec![
        Layer::added(read(0), [1.0, 0.2, 0.1]),
        Layer::added(read(1), [0.1, 0.9, 0.3]),
    ];
    let finished = Picture {
        layers: layers.clone(),
        finish: kinds
            .iter()
            .map(|kind| FinishStep {
                kind: kind.clone(),
                opacity: 1.0,
                mask: None,
                range: None,
            })
            .collect(),
        ..Picture::default()
    };
    let stacked = Picture {
        layers: layers
            .into_iter()
            .chain(kinds.iter().cloned().map(step))
            .collect(),
        ..Picture::default()
    };
    let region = whole(&inputs[0]);
    let a = render(&finished, &images, region, 4096);
    let b = render(&stacked, &images, region, 4096);
    assert!(a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()));
}

/// A step in the stack changes the layers beneath it, and the layers above
/// it are blended onto its result; inside a group it changes the group's
/// members beneath it only.
#[test]
fn a_step_changes_only_what_is_beneath_it() {
    use seiza_compose::picture::FinishKind;
    let (inputs, read) = stacked_inputs();
    let images: Vec<&LinearImage> = inputs.iter().collect();
    let region = whole(&inputs[0]);
    let gray = || step(FinishKind::Saturation(0.0));
    let red = || Layer::added(read(0), [1.0, 0.2, 0.1]);
    let green = || Layer::added(read(1), [0.1, 0.9, 0.3]);
    let picture = |layers: Vec<Layer>| Picture {
        layers,
        ..Picture::default()
    };
    let both = render(
        &picture(vec![red(), gray(), green()]),
        &images,
        region,
        4096,
    );
    let beneath = render(&picture(vec![red(), gray()]), &images, region, 4096);
    let above = render(&picture(vec![green()]), &images, region, 4096);
    let sum: Vec<f32> = beneath.iter().zip(&above).map(|(a, b)| a + b).collect();
    assert!(close(&both, &sum, 1e-6));
    // The red layer is gray beneath the step, and the green one keeps its
    // color above it.
    assert!((beneath[0] - beneath[1]).abs() < 1e-6 && (beneath[1] - beneath[2]).abs() < 1e-6);

    // In a group, the layers beneath the group keep their color.
    let grouped = render(
        &picture(vec![
            green(),
            Layer::new(LayerKind::GroupBegin),
            red(),
            gray(),
            Layer {
                blend: seiza_compose::picture::Blend::Add,
                ..Layer::new(LayerKind::GroupEnd)
            },
        ]),
        &images,
        region,
        4096,
    );
    let sum: Vec<f32> = above.iter().zip(&beneath).map(|(a, b)| a + b).collect();
    assert!(close(&grouped, &sum, 1e-6));
}

/// Steps in the stack that read neighboring pixels, a local Tone step and
/// a heal, give the same pixels in any tiling: a tile reads far enough
/// around it for each.
#[test]
fn steps_in_the_stack_do_not_depend_on_tiling() {
    use seiza_compose::picture::FinishKind;
    use seiza_compose::tone::Tone;
    let (inputs, read) = stacked_inputs();
    let images: Vec<&LinearImage> = inputs.iter().collect();
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
            step(FinishKind::Tone(Tone {
                clarity: -0.4,
                highlights: 0.3,
                ..Tone::default()
            })),
        ],
        masks: vec![common::stroke([60.0, 50.0], [150.0, 120.0], 18.0)],
        ..Picture::default()
    };
    let one = render(&picture, &images, whole(&inputs[0]), 4096);
    for tile in [32, 90] {
        let tiled = render(&picture, &images, whole(&inputs[0]), tile);
        assert!(
            one.iter()
                .zip(&tiled)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "tiles of {tile} differ from one tile"
        );
    }
}

/// A heal inside a group copies the group's members from beside it, before
/// the group is blended over what is beneath: a small region of a large
/// picture, so the copy comes from the margin around it.
#[test]
fn a_heal_inside_a_group_copies_its_members() {
    use seiza_compose::picture::{Blend, FinishKind};
    let inputs = [plane(200, 120, 1, true), plane(200, 120, 2, false)];
    let images: Vec<&LinearImage> = inputs.iter().collect();
    let read = |input: usize| InputRead::new(input, Transfer::identity());
    let picture = |healed: bool| Picture {
        layers: [
            vec![
                Layer::added(read(0), [0.3, 0.3, 0.3]),
                Layer::new(LayerKind::GroupBegin),
                Layer::added(read(1), [1.0, 1.0, 1.0]),
            ],
            healed
                .then(|| Layer {
                    mask: Some(0),
                    ..Layer::new(LayerKind::Step(FinishKind::Heal {
                        offset: [20, 0],
                        shift: [0.0; 3],
                    }))
                })
                .into_iter()
                .collect(),
            vec![Layer {
                blend: Blend::Screen,
                ..Layer::new(LayerKind::GroupEnd)
            }],
        ]
        .concat(),
        masks: vec![common::stroke([100.0, 60.0], [100.0, 60.0], 9.0)],
        ..Picture::default()
    };
    let region = ReferenceRegion {
        x: 96,
        y: 56,
        width: 8,
        height: 8,
    };
    let plain = render(&picture(false), &images, region, 256);
    let healed = render(&picture(true), &images, region, 256);
    let gap = plain
        .iter()
        .zip(&healed)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(gap > 1e-3, "the heal changed nothing");
}

/// Groups nest: added groups in added groups make the same sum as the
/// layers added flat, and so does a group nested past the deepest kept,
/// whose members blend into the group around it.
#[test]
fn nested_groups_blend_as_their_members() {
    use seiza_compose::picture::{Blend, MAX_GROUP_DEPTH};
    let inputs = [plane(60, 40, 1, true), plane(60, 40, 2, false)];
    let images: Vec<&LinearImage> = inputs.iter().collect();
    let read = |input: usize| InputRead::new(input, Transfer::identity());
    let added = |input: usize, color: [f32; 3]| Layer::added(read(input), color);
    let end = || Layer {
        blend: Blend::Add,
        ..Layer::new(LayerKind::GroupEnd)
    };
    let flat = Picture {
        layers: vec![added(0, [0.4, 0.2, 0.1]), added(1, [0.1, 0.3, 0.5])],
        ..Picture::default()
    };
    for depth in [1, 2, MAX_GROUP_DEPTH + 2] {
        let mut layers = vec![added(0, [0.4, 0.2, 0.1])];
        layers.extend((0..depth).map(|_| Layer::new(LayerKind::GroupBegin)));
        layers.push(added(1, [0.1, 0.3, 0.5]));
        layers.extend((0..depth).map(|_| end()));
        let nested = Picture {
            layers,
            ..Picture::default()
        };
        let a = render(&flat, &images, whole(&inputs[0]), 256);
        let b = render(&nested, &images, whole(&inputs[0]), 256);
        let gap = a
            .iter()
            .zip(&b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(gap < 1e-6, "{depth} groups deep: {gap}");
    }
}

/// The balance stage gives a channel the curve Seiza's narrowband balance
/// resolves for it: applied to OIII, it matches the balanced image.
#[test]
fn the_balance_stage_matches_the_narrowband_balance() {
    use seiza_compose::picture::TransferStage;
    use seiza_compose::{NarrowbandBalanceOptions, balance_narrowband};
    let (width, height) = (160, 120);
    // H-alpha bright, OIII faint, SII between, each with structure.
    let data: Vec<f32> = (0..width * height)
        .flat_map(|at| {
            let (x, y) = ((at % width) as f32, (at / width) as f32);
            let wave = ((x * 0.07).sin() * (y * 0.05).cos()).max(0.0);
            let grain = ((at * 7919) % 101) as f32 / 101.0 * 0.01;
            [
                0.15 + 0.6 * wave + grain,
                0.10 + 0.15 * wave + grain,
                0.12 + 0.3 * wave + grain,
            ]
        })
        .collect();
    let original = LinearImage::new(width, height, 3, data).unwrap();
    let mut balanced = original.clone();
    let report = balance_narrowband(&mut balanced, &NarrowbandBalanceOptions::default()).unwrap();
    let oiii = &report.channels[1];
    let (Some(background), Some(midtone)) = (oiii.background, oiii.midtone) else {
        panic!("OIII was not balanced: {report:?}");
    };
    let transfer = Transfer::of([TransferStage::Balance {
        background: background as f32,
        midtone: midtone as f32,
    }]);
    for (pixel, after) in original
        .data
        .as_chunks::<3>()
        .0
        .iter()
        .zip(balanced.data.as_chunks::<3>().0)
    {
        assert!((transfer.map(pixel[1]) - after[1]).abs() < 1e-5);
    }
}
