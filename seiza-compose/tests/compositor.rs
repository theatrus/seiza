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
