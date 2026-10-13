# seiza-parallax

`seiza-parallax` renders fly-through videos of stretched astrophotographs.
An image is split into its starless image and its stars, for example with
StarXTerminator's "unscreen" star image. The starless image becomes a plane at
the target's distance, regions of it may be lifted onto planes at their own,
each star is cut out and placed at its own distance, and
a camera flies toward a point of the target. Nearer stars spread out and grow
as it approaches, while the distant star field barely moves.

The crate takes the images and the stars' distances as inputs; it does not
solve images or query catalogs. `seiza parallax-video` in `seiza-cli` does
both, matching stars to Gaia DR3 for their Bailer-Jones distances.

- **Finding stars:** `find_stars` runs `seiza-stars`' peak detector
  (`find_peak_stars`) on a star image's light, taking each local peak that
  rises clearly above its own surroundings. Crowded stars stay apart, a
  saturated core is one star, and a ripple on a bright star's halo is not a
  star.
- **Cutting them out:** `Scene::new` gives each star a soft footprint that
  grows until it reaches the star image's background. Overlapping light is
  shared in proportion to each star's modelled light, so a bright star keeps
  its halo. Star light no star takes forms a plane of its own at the star
  field's distance. `CutOptions::max_stars` lets only the brightest stars
  fly; the rest are dropped or stay on that plane (`SmallStars`).
- **Galaxies:** `lift_object` takes an extended object, such as a galaxy a
  star remover left in the starless image, out onto a sprite of its own at
  its distance, and fills the nebula in behind it from around it.
- **Depth planes:** `lift_region` lifts a region of the starless image onto
  a `Plane` at its own distance: a cloud in front of the target or behind
  it. A `PlaneMask`, an image white where the region is, becomes a `Region`
  at the image's size. What lay behind the region is filled in by
  push-pull: the light around the hole is averaged into coarser and coarser
  levels and blended back down, so the fill is smooth deep in the hole and
  close to the light at its edge. `Scene::with_planes` adds the planes,
  smallest first; a star at a plane's distance, inside it, joins it, and
  `Scene::depth_at` gives the depth under a point. Each plane is drawn in
  perspective over the part of the frame it reaches and fades as the
  camera flies through it, and `Shot::fitted` keeps the edges where the
  image cuts a plane off out of view. Where a tour's frame would show much
  of the smooth fill behind a plane (`Shot::fill_shown`), the fit flies
  that stop less far and zooms its lens to keep its target's size. Where a region in front of the
  target is darker than the fill, a dark cloud blocks the light behind: the
  plane keeps a `Transmission`, each colour's, the light behind is
  restored, and everything behind the cloud dims by it at the cloud's own
  depth.
- **Dust:** `Dust::from_star_counts` maps the dust's transmission from how
  few stars show through it, and `Dust::with_darkness` adds globules too
  small for the counts: darker than their surroundings and short of stars.
  With `Scene::dust` set, far stars, the far star field and lifted galaxies
  dim as they slide behind thicker dust. `Parallax` splits the counted dust
  into clouds, each at one distance: a dark nebula moves with the cloud it
  lies in, and a cloud the catalogue does not place lies where the Gaia
  stars seen through it run short.
- **Blending:** layers combine as a screen blend, done as addition of "light"
  `−ln(1 − v)` taken from each pixel's brightest channel, so stars keep their
  hue as they brighten.
- **The camera:** `Shot` dollies toward the background plane, optionally
  trucks sideways (reduced by `Shot::fitted` so no layer slides off the image),
  can lengthen its lens, and can roll the frame about its centre from one
  angle to another (`Shot::fitted` zooms the first frame in to keep the turned
  frame inside the image). Stars stay points: they keep their first-frame
  size and only swell and brighten as the camera nears them, fading out as it
  passes.
- **Placing marks:** `Shot::view` gives the camera at any frame, and
  `View::project` puts a point of any depth in it, so a caller can draw
  labels that move with what they mark (`seiza parallax-video --overlay`
  does). `Shot::star_fade` says how far a star the camera passes has faded.
- **Output:** frames go to `ffmpeg` (H.264 with libx264, or libopenh264 when
  that is the build's H.264 encoder, or HEVC with libx265), in MP4, MOV or
  Matroska by the file's extension, with BT.709 colour; to numbered PNG
  files; or, with the `openh264` feature, to an in-process OpenH264 encoder
  and MP4 muxer. A video stopped or failed part way leaves no file.

## License

Apache-2.0.
