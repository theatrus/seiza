//! Inputs and a picture the compositor's tests share.

#![allow(dead_code)]

use seiza_compose::picture::{
    Adjustment, Blend, ComputedMask, FinishKind, FinishStep, InputRead, Layer, LayerHeal,
    LayerKind, LuminanceTarget, Picture, Segment, Transfer, TransferStage,
};
use seiza_compose::tone::Tone;
use seiza_stacking::{LinearImage, ReferenceRegion};
use seiza_stretch::{ResolvedCurve, StretchConfig, StretchModel};

/// A plane of a faint sky with structure, stars and, with `gap`, one
/// uncovered pixel.
pub fn plane(width: usize, height: usize, seed: u32, gap: bool) -> LinearImage {
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

pub fn whole(image: &LinearImage) -> ReferenceRegion {
    ReferenceRegion {
        x: 0,
        y: 0,
        width: image.width,
        height: image.height,
    }
}

pub fn stroke(from: [f32; 2], to: [f32; 2], radius: f32) -> Vec<Segment> {
    vec![Segment {
        from,
        to,
        radius,
        hardness: 0.4,
        value: 1.0,
    }]
}

/// A picture with every kind of entry the compositor draws.
pub fn everything() -> Picture {
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
            highlights: -0.3,
            texture: 0.5,
            clarity: 0.4,
            dehaze: 0.3,
            vibrance: 0.3,
            // A measured noise, so the GPU's Texture edge is the measured
            // one, not the fixed floor.
            noise: 0.02,
            ..Tone::default()
        }),
        opacity: 1.0,
        mask: None,
    });
    // A heal after the Tone step copies the toned picture, each source
    // pixel toned with its own neighborhood.
    finish.push(FinishStep {
        kind: FinishKind::Heal {
            offset: [-23, 11],
            shift: [0.0; 3],
        },
        opacity: 1.0,
        mask: Some(1),
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
        // A wide space, so the GPU is checked with weights that are not
        // sRGB's.
        space: seiza_compose::space::WorkingSpace::Rec2020,
        headroom: 0.0,
    }
}
