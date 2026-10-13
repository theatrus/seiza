# Images over parallax videos

`seiza parallax-video` could already write text over its frames. This
note adds two kinds of image: a watermark image, such as a logo, and title
cards, made elsewhere with transparency and shown over the video.

Status: built in `seiza-parallax` (`picture.rs`, `overlay.rs`), the CLI,
the C ABI and Python, as below. The apps' fields are still to come.

## What there was

- `--watermark[=TEXT]` writes one line in the bottom-right corner of every
  frame, at 80% opacity over a dark halo ("Rendered with seiza.fyi" without
  text). The C ABI's `watermark` takes `true` or the text; Python's takes
  `True` or the text.
- `--tour-titles` shows each tour stop's `title="..."` low on the left under
  a short rule. A title fades in a little before the camera reaches its stop
  and out a little after it leaves (`title_shares`); a looped tour's first
  and last stops share one title across the join.
- `--overlay` and `--label` mark objects and places, each moving with its
  depth.

`Parallax::frame` renders a frame, turns it into an 8-bit RGB image at the
output size, and only then draws the overlay on it (`Overlay::draw`). With
`--quality high` the frame is drawn at twice the size and halved before
that, so overlays are always drawn once, at the output size. Text is drawn
as a coverage mask from `seiza-draw` and blended over the frame in sRGB,
not in linear light. Images go in the same place, in the same way.

## Reading the images

Every image is read through `seiza_image_io::open_raster`, which keeps an
image's alpha and turns it upright by its EXIF orientation. PNG and TIFF
carry alpha. An image without alpha, such as a JPEG, is drawn opaque,
and the video says so, since a card without transparency hides the whole
frame. A FITS or XISF file is refused: it is data, not artwork.

Each image is converted once, when the video is set up, to 8-bit RGBA with
its colour multiplied by its alpha (premultiplied), then scaled to the size
it will be drawn at. Scaling premultiplied pixels keeps dark or light
fringes out of soft edges. A shrunk image uses a Lanczos filter; one drawn
at its own size is copied as it is. Nothing is scaled while frames are
drawn.

Drawing a premultiplied pixel `(c, a)` at opacity `o` over a frame pixel
`f` gives `c · o + f · (1 − a · o)`, in sRGB like the text. A logo or card
then looks as it does in the program that made it.

A missing or unreadable file fails when the video is set up (the C ABI's
prepare or reconfigure), never part way through a render.

## Watermark image

`--watermark-image FILE` draws the image in a corner of every frame.

- **Size.** Its height is a share of the frame's shorter side,
  `--watermark-size` (default 0.06), and its width follows its shape, at
  most a third of the frame's width. The text watermark's letters are
  about 2% of the shorter side tall, so a logo of 6% sits beside it
  without crowding it.
- **Place.** `--watermark-corner`: `bottom-right` (the default),
  `bottom-left`, `top-right` or `top-left`, inset by the margin the text
  uses (the label font size). The corner moves the text too. With both an
  image and text, the text keeps its line along the edge and the image
  sits just inside it, above the text at the bottom and below it at the
  top, aligned to the same side.
- **Opacity.** `--watermark-opacity`, default 0.8 as the text always had,
  sets both the text's and the image's; the image's own alpha multiplies
  it.
- **C ABI.** `watermark` also takes an object:
  `{"text": ..., "image": PATH, "corner": ..., "size": ..., "opacity": ...}`,
  each optional; `true` and a string keep their meanings. The corner is
  spelt `bottomRight` and so on, as the C ABI's other choices are. Python's
  `watermark` takes the same as a dict, with `bottom_right` and so on.
- **Apps.** The Export step gets a file picker beside the watermark switch,
  with the corner and size.

## Title cards

A title card is a full image, made with transparency in Photoshop, Figma,
After Effects or the like, laid over the video. It lets a user title a
video in any typeface and layout, which the built-in titles cannot.

**Stop cards.** A tour stop's `card=FILE` shows the card at that stop, in
the window and with the fades a text title has (`title_shares`), and in
place of its text title. A card shows whether or not `--tour-titles` is
given: naming a file is asking for it. In a tour file, a relative path is
taken from the tour file's folder, so a folder of tour and cards can move
as one. An automatic tour plans no cards; add them to the planned file.

**Opening and closing cards.** `--intro-card FILE` shows over the start of
the video: whole from the first frame, then fading out.
`--outro-card FILE` fades in over the end and stays to the last frame.
`--card-seconds` sets how long each shows, default 3, of which the last
(or first) 0.8 s is the fade. Both work for the single move as well as
for a tour.

**Looped videos.** A loop must join without a jump. With `--loop`, the
opening and closing cards show across the join instead: a card fades in
over the last seconds and out over the first, as a looped tour's title
does, so the frame after the last is still the first. With both cards
given, the closing card fades out as the opening card fades in.

**Size and place.** A card is fitted inside the frame and centred,
keeping its shape, so a smaller card is enlarged. A card made at the
output size (1920 × 1080, or 1080 × 1920 for `-portrait`) is used pixel
for pixel, which is the way to get sharp lettering. When a card's shape differs from the frame's by more than
2%, the video warns, since a landscape card on a portrait video shrinks to
a band across the middle: make a card for each shape.

**Order.** Labels and "Field within" captions first, then text titles,
then cards, then the watermark, so the watermark stays on top.

**C ABI and Python.** `introCard`, `outroCard` and `cardSeconds` in the
video options, and `card` on a tour stop, all paths; Python takes
`intro_card=`, `outro_card=`, `card_seconds=` and a stop's `card=`. A
planned tour has no cards; a stop sent back with one keeps it. The apps'
tour list gets a card field for each stop, and the Export step the
opening and closing cards.

## Memory and speed

Images are kept at the size they are drawn at, 8 bits a channel: a 4K
card is 33 MB, and the watermark a few hundred kilobytes. Drawing a card
touches only the pixels where its alpha is above zero, so a card of
lettering on a clear background costs little; a full-frame opaque card
costs one blend a pixel, small beside rendering the frame.

## Tests

- A card with alpha 0 everywhere leaves every frame unchanged, and one
  with alpha 1 everywhere replaces it.
- The opening card's first frame is the card over the photograph's first
  frame, and a looped video's frame after the last equals its first, with
  cards on.
- A stop's card shows over the same frames its text title would, and
  replaces it.
- A card of another shape is scaled to fit and centred, with a warning; a
  card at the output size is copied pixel for pixel.
- The watermark image sits in each corner as asked, inside the margin, and
  above the text when both are given.
- A missing file fails at setup, naming the file.

## Later

- **Moving cards.** A numbered PNG sequence played over the opening or a
  stop, for animated titles. Video with alpha (ProRes 4444, WebM VP9) would
  need `ffmpeg` to decode it, which only some builds carry.
- **A card on a depth.** Drawn at a depth rather than flat on the screen,
  so the parallax moves it with the scene, as labels move now.
