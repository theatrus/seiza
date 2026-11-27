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
