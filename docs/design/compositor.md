# One compositor for every color picture

> Status: proposed. `seiza-compose` holds Seiza's color routines (moved out of
> `seiza-stacking` unchanged); this note describes the compositor that the
> next change adds to it, and that `combine_*` will be built on.

## Why

A color picture is made in two places today. PSF Guard's color previews call
`combine_rgb`, `combine_lrgb` and `combine_narrowband`, which are fixed
recipes over whole channels. PSF Studio, its color editor, blends stretched
inputs as layers, with masks, groups, heals and finishing steps, on the GPU,
and has its own CPU copy of those passes for exports. Two renderers drift:
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

A pixel any layer's input does not cover is not covered in the output: it
is `NaN`, as `combine_*` give today.

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
offsets. Inputs come from a `Source`, which reads a rectangle of the named
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

PSF Studio's canvas draws a `Picture` with shaders that follow this renderer.
Where a GPU cannot match it exactly (half-float textures, 8-bit masks, a
blur sampled in fewer taps), the canvas is the approximation and this
renderer the reference; `psf-studio --parity-report` measures the gap on a
real picture.
