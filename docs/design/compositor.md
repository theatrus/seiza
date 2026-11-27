# One compositor for every color picture

> Status: built. `seiza_compose::picture` and `seiza_compose::render` are the
> compositor and its CPU render, `combine_*` render through them, and
> `seiza_compose::gpu` renders the same pictures with wgpu.

## Why

A color picture is made in two places today. PSF Guard's color previews call
`combine_rgb`, `combine_lrgb` and `combine_narrowband`, which are fixed
recipes over whole channels. PSF Studio, its color editor, blends stretched
inputs as layers, with masks, groups, heals and finishing steps, on the GPU,
and has its own CPU copy of those passes for exports. With two renderers,
the same setup gives two different pictures, and every fix is made twice.

So Seiza gets one authoritative renderer: a compositor that takes a
description of a picture and renders it on the CPU, in 32-bit floats, in
tiles. The `combine_*` functions become builders of that description, so
their callers keep their calls and their output. PSF Studio's GPU canvas
draws the same description as a fast approximation, and a parity check holds
it to this renderer.

## The picture

A `Picture` lists:

- **Inputs:** the mono planes a picture reads, by name, each with its
  transfer: identity, a clamped affine map (percentile normalization), or a
  chain of `seiza_stretch` curves after a resolved sample domain. A transfer
  holds Seiza's resolved curves, never a lookup table; a GPU derives its
  tables from them.
- **Layers,** bottom first: an input in a color, with an opacity and a blend;
  a solid color; the start and end of a group; or an adjustment of what is
  beneath. A layer can be limited by a painted mask, by the mask of the layer
  it is clipped to, and by a computed mask.
- **Blends:** normal, screen, add, multiply, lighten, darken, luminance, and
  two that LRGB needs: *luminance ratio*, which scales the colors beneath so
  their Rec. 709 luminance becomes the layer's (opacity is LRGB's weight), and
  *super luminance*, whose target is the layer plus `R + G + B`.
- **Computed masks:** the product of one or two transferred inputs, shaped
  `p^(1−p)` and raised to a reach, optionally smoothed by Seiza's
  coverage-weighted Gaussian blur, with painted strokes added or removed.
  Foraxx's blend factors are such masks.
- **Painted masks and heals:** brush strokes in document pixels, and heals
  of an input, a layer or the whole picture.
- **Finish steps** on the combined picture: output stretches (Seiza stretch
  plans), Tone and color, curves and saturation, heals.

## Tone and color

Tone and color has Camera Raw's Basic controls. Exposure, Whites and Blacks
set the luminance's range; Contrast, Vibrance and Saturation follow. The
rest read each pixel's neighborhood, so they change what is around a pixel
rather than every pixel of one brightness alike:

| Control | What it does |
|---|---|
| Highlights, Shadows | Move the tones of the local base, a pixel's surroundings, and keep the detail above it. A star in a lifted field stays as far above the field. |
| Clarity | Raises or lowers the detail around the local base, most in the midtones. |
| Texture | Raises or lowers the detail around the mean of a few pixels, as far as the window holds more than noise. A host measures the picture's noise with `tone::luminance_noise` and gives it to the step; unmeasured, a fixed level stands in. |
| Dehaze | Takes away a veil estimated from the darkest channel nearby, so a gray sky glow goes and a colored nebula stays. Negative adds a veil. |
| HDR, headroom | With HDR on, the brightest areas rise above SDR white, 1, instead of clipping at it: values pass as they are through the first 80 % of the headroom, then a shoulder that starts with slope 1 flattens toward the peak, so a core an HDR stretch already placed in the headroom keeps its contrast. Values above 1 continue on the extended sRGB curve that macOS's extended color spaces use, so the peak is the encoded value of 2 to the headroom (up to four stops) times SDR white's light: about 1.82 for two stops. Exposure and Whites push areas there; Vibrance and Saturation never push a channel past the peak. Below SDR white nothing changes. |

A neighborhood is the means of the luminance, its square, the darkest
channel and their product over a square window: 97 pixels across for the
local base, 7 for the fine one. A base is the window's mean drawn toward
the pixel where the window's variance is large, as a guided filter does, so
an edge, such as a star's, is its own base and gets no halo. The veil is the
darkest channel's base, guided by the luminance. A pixel's neighborhood is
of the picture as the steps before Tone left it, heals left out, the
picture a heal copies from. The veil is never brighter than the window's
darkest channel on average, so a large bright area, such as a galaxy's
core, is not taken for one. Each Tone step that reads neighborhoods adds
its window to a tile's margin, since it reads around a picture an earlier
one already read around. With none of these controls set, the step is a
curve of each pixel's luminance, as before.

A pixel any layer's input does not cover is not covered in the output: it
is `NaN`, as `combine_*` return today.

## Working space

A picture's `space` sets the primaries its colors are in: sRGB (the
default, so the recipes render as before), Display P3 or Rec. 2020. Every
space has D65 white and the sRGB transfer curve, extended past 1 for HDR,
as Apple's `ITUR_2020_sRGBGamma` pairs Rec. 2020 with that curve. The
space sets the luminance weights that Tone, Saturation, the Luminance
blend and LRGB use: BT.709's for sRGB, BT.2020's for Rec. 2020, and the
XYZ row for Display P3. The GPU reads them from the globals.

A host shows the picture on a display through
`WorkingSpace::map_to_display`, or its GPU twin `DISPLAY_SHADER` with a
`DisplayMapping`: decode, convert the light with a matrix, then bring a
color the display cannot show inside its gamut by desaturating it toward
the gray of its own luminance until no channel is negative. That keeps the
color's luminance and hue. Values above 1, HDR's highlights, are kept.

## HDR pictures

A picture's `headroom`, in stops, makes it HDR all the way through: the
combined picture and every Finish step may reach `Picture::peak`, the value
on the extended sRGB curve of 2 to the headroom, and every Tone step is an
HDR one with that headroom. A host shows every view of the picture, before
or after any step, up to that peak. Without it, an HDR Tone step alone
lifts the picture above SDR white from that step on.

## HDR stretches

A stretch curve flattens toward 1, so the brightest part of an image, such
as a planetary nebula's core, loses its structure before Tone can lift it.
`TransferStage::TailedCurve` continues a resolved curve above a knee
(encoded 0.75 by default) into the headroom: the display light rises as a
power of the input above the curve's black, starting with the curve's own
slope and relaxing toward the power that reaches 2 to the headroom times
SDR white's light at the white. The curve and its slope match at the knee,
so the sky and faint halo are as the curve made them. `seiza_stretch::
HdrTail` resolves and maps it; a host sets the headroom to its HDR Tone
step's. On the Cat's Eye (NGC 6543) the tail alone keeps about a quarter of
the core's fine contrast at 3 stops, where the plain curve and Tone keep 1 %.

## Layered stretches

Seiza's layered stretch, plain or HDR, reads more than each pixel: the
HDR core compression blurs the log of luminance, and contrast recovery
blurs luminance into a large-scale layer. Both are smooth, so
`ResolvedLayered::measure` measures them once from the whole picture at a
quarter of its resolution (`reduce_luminance`, `MAP_REDUCTION`), along
with the HDR tail's white, the brightest large-scale luminance after the
compression. `seiza_stretch` itself stretches a whole image the same way.
`TransferStage::Layered` carries the stretch and its maps; the CPU render
reads the maps at each sample's place, and the GPU reads them from a
texture array, interpolating between map pixels as the CPU does, and works
out the curve, core gain, HDR tail and detail gain in the shader. Up to
four layers per draw take a layered stretch; another is drawn with its
curve alone and named in `left_out`, as is a layered stretch a computed
mask or a luminance layer reads. A host measures the maps again only when
a setting that moves them changes (the core compression's scale, or the
stretch's background statistics), so a slider redraws at once. The plain
layered stretch's star-core compression needs full resolution and is not
drawn by the compositor. When the core compression leaves the white below
the knee, the knee comes down to 80 % of the white's curve value, so the
brightest structure still reaches the top of the headroom.

## The recipes as pictures

| Recipe | Picture |
|---|---|
| `combine_rgb` | R, G and B layers in red, green and blue, added. |
| Static palettes and `combine_narrowband_matrix` | One layer per matrix term, the weight in its color, added. |
| `combine_lrgb` | The RGB picture, then L in luminance ratio at opacity `luminance_weight`. |
| `combine_super_lrgb`, `combine_super_rgb` | The RGB picture, then a super luminance layer (L, or none). |
| Foraxx SHO and HOO | Each term (`f·H`, `(1−f)·O`, ...) as an added layer under a computed mask of the display-prepared inputs. |

Normalization becomes each input's transfer, measured once on the kept
region, as now. `tests/output_version.rs` holds every recipe to the output it
gave before the compositor existed.

## Tiles

The compositor never holds a whole picture. It renders the output in square
tiles. Each tile reads its inputs over the tile and a margin wide enough for
everything that reads beyond a pixel: the largest mask smoothing (its blur
reaches four standard deviations, as Foraxx's does), plus the largest heal
offsets, plus the local window of a Tone step that reads neighborhoods. Inputs come from a `Source`, which reads a rectangle of the named
planes; outputs go to a `Sink`, which takes finished tiles in any order.

What needs the whole picture is measured before rendering, once, from
samples: percentile levels, stretch analyses, and the region every input
covers. A tile then needs nothing but its own pixels and margin, so tiles
render in parallel in the caller's Rayon pool, and peak memory follows the
tile size rather than the picture's.

`combine_*` read their `LinearImage`s through an in-memory source and write
to an in-memory sink, so they keep returning a `ColorComposition`. A host
with a picture larger than memory gives a source over its files and a sink
that writes a FITS, XISF or TIFF tile by tile.

## The GPU

Seiza owns the shaders as well. `seiza_compose::gpu` (the default `gpu`
feature) draws a `Picture` with wgpu:

- `GpuScene::new` packs a picture for the shaders: each transfer a row of a
  stretch table indexed by the square root of a sample, each finishing
  stretch its curves. Tone and color is computed in the shader.
- `GpuCompositor::draw` runs the passes (painted masks, healed layers,
  computed masks and their blur, layers, a Tone step's neighborhoods,
  finishing steps) into targets the
  size of the area drawn, from input tiles the host binds, and leaves the
  finished picture in the working space. A canvas applies its display
  transform after it.
- `render_gpu` renders a picture from a `Source` into a `Sink` in tiles,
  as `render` does, for a host without a canvas.

The CPU render is the reference. Where the GPU differs by design, it is
an approximation: inputs and stretch tables are half floats, a computed
mask's smoothing is a 33-tap Gaussian over three standard deviations,
clamped at the drawn area's edge, and a canvas draws an uncovered pixel
black. A Tone step's windows are cut at the edge of the picture drawn,
within the scissor, and are measured in drawn pixels: zoomed out, a window
covers the same part of the picture at the canvas's resolution; zoomed in
past about five times, a window is capped at 256 drawn pixels either side.
Up to four Tone steps per draw read neighborhoods, each measured into its
own layer of the local textures after the steps before it; a fifth is
drawn with each pixel alone, without Dehaze, and named in `left_out`.
`tests/gpu.rs` renders the same picture both ways and bounds the gap; on
Apple silicon the mean difference is about 1e-4 and the 99th percentile
under 1e-3, smoothing included. A planted shader change fails
it. Without a GPU the tests pass without checking.

PSF Studio keeps only what belongs to an interactive canvas: streaming tiles
from its pyramid, panes and the wipe divider, the brush overlay's strokes,
the transform to the display, and turning its document into a `Picture`.
PSF Guard's color previews render through the same crate.

## As built

- An added input layer is one fused multiply-add per channel
  (`tint·value + below`, or `(tint·value)·cover + below` under a mask), in
  the order each recipe always summed its terms, so `combine_rgb`, the
  matrix palettes, LRGB and unsmoothed Foraxx give the same bits as before.
- A smoothed computed mask is the coverage-weighted mean of its
  neighbours everywhere. Before, Foraxx divided by the blurred coverage only
  when some pixel of the region was uncovered; a tile cannot know that, and
  where every pixel is covered the blurred coverage is 1 within a few float
  steps. The recorded cases pass unchanged.
- A pixel any input of the picture does not cover is `NaN`, and has no
  weight in a computed mask's smoothing.
