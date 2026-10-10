# seiza-draw

Anti-aliased lines, ellipses and text for marking images, as Seiza draws
them on sky maps (`seiza solve --sky-map`) and over parallax videos
(`seiza parallax-video --overlay`).

Shapes and text go into a `Mask` of coverage, one per colour, which is laid
over an `image::RgbImage` once, so overlapping strokes never double-blend.
`Mask::dilated` spreads a mask into a soft halo to lay under it in a dark
colour, which keeps labels readable over a busy photograph.

```rust
use image::{Rgb, RgbImage};
use seiza_draw::{Fonts, Mask, draw_text};

let fonts = Fonts::load().unwrap();
let mut canvas = RgbImage::new(400, 200);
let mut mask = Mask::new(400, 200);
mask.ellipse((200.0, 100.0), 80.0, 40.0, 30.0, 1.5);
draw_text(&mut mask, &fonts.regular, 18.0, 0.0, (150.0, 30.0), "NGC 7023");
mask.dilated(2.0).composite(&mut canvas, Rgb([0, 0, 0]), 0.75);
mask.composite(&mut canvas, Rgb([85, 207, 255]), 1.0);
```

## Fonts

Text uses Inter 4.1 Regular and SemiBold, embedded in the crate and subset
to Latin, Greek and common punctuation (see [`fonts/README.md`](fonts/README.md)).
Inter is under the SIL Open Font License 1.1, whose full text is in
[`fonts/LICENSE-Inter.txt`](fonts/LICENSE-Inter.txt); anything that ships
this crate's code, a binary or a library, must ship that file too.

## License

Apache-2.0 for the code; OFL-1.1 for the embedded font.
