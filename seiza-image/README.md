# seiza-image

What Seiza's image steps share, so that a host will be able to run any of
them on part of an image and combine their results. This crate holds the
shared types. Each step adopts them in its own crate; none has yet.

- `ImageShape`: the width, height and channel count of a caller's samples.
- `Region`: a rectangle of an image, with the copying and padding a step
  needs to work on it.
- `Domain`: whether samples are linear light or display-referred. `Accepts`
  gives the refusal a step returns when it is handed the wrong kind.
- `Mask`: one plane of values from 0 to 1 that limits where a step takes
  effect. Masks come from a range of sample values or from a host's own
  values, and combine by union, intersection and inversion.
- `RegionOperation`: the contract for a step that is resolved once from a
  whole image and then applied to any region of it.

This crate adds no image type. A step takes row-major, channel-interleaved
`f32` samples with their shape, so a host keeps its own storage.

## The region contract

An editor shows a full-resolution crop while a slider moves, works through a
mosaic in tiles, and caches results by region. All three need one guarantee:
the result for a region matches the same pixels of the result for the whole
image. A step whose reach is short matches exactly. A step whose influence
falls off with distance may declare less padding than its full reach, and
then states and tests the tolerance it keeps.

A step keeps it by resolving first and applying second. Resolving measures
what the step needs from the whole image, such as a sample range, a noise
level or a fitted model, and stores it in the operation. Applying uses only
those stored values and the block it is given, and states how many pixels of
context it needs:

```rust
use seiza_image::{Block, ImageShape, Region, RegionOperation, apply_to_region};

struct Scale {
    // Measured once, from the whole image.
    low: f32,
    high: f32,
}

impl RegionOperation for Scale {
    type Error = std::convert::Infallible;

    fn padding(&self) -> usize {
        0
    }

    fn apply(&self, block: Block<'_>) -> Result<Vec<f32>, Self::Error> {
        Ok(block
            .samples
            .iter()
            .map(|value| (value - self.low) / (self.high - self.low))
            .collect())
    }
}

let shape = ImageShape::new(640, 480, 1);
let image = vec![0.25_f32; 640 * 480];
let scale = Scale { low: 0.0, high: 1.0 };
let crop = apply_to_region(&scale, &image, shape, Region::new(100, 80, 256, 256)).unwrap();
assert_eq!(crop.len(), 256 * 256);
```

`region_mismatch` measures the guarantee for one region. Test every
implementation with it on regions inside the image, near its edges, on its
edges and in its corners.

The trait covers applying only. Resolving, diagnostics, progress and
cancellation stay on each step's own functions. An operation returns samples
of the block's shape, so a step that changes an image's size is not one.
