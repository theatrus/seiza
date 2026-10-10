# seiza-parallax

`seiza-parallax` renders fly-through videos of stretched astrophotographs.
An image is split into its starless image and its stars, for example with
StarXTerminator's "unscreen" star image. The starless image becomes a plane at
the target's distance, each star is cut out and placed at its own distance, and
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
- **Blending:** layers combine as a screen blend, done as addition of "light"
  `−ln(1 − v)` taken from each pixel's brightest channel, so stars keep their
  hue as they brighten.
- **The camera:** `Shot` dollies toward the background plane, optionally
  trucks sideways (reduced by `Shot::fitted` so no layer slides off the image),
  and can lengthen its lens. Stars stay points: they keep their first-frame
  size and only swell and brighten as the camera nears them, fading out as it
  passes.
- **Output:** frames go to `ffmpeg` (libx264, or libopenh264 when that is the
  build's H.264 encoder), to numbered PNG files, or, with the `openh264`
  feature, to an in-process OpenH264 encoder and MP4 muxer.

## License

Apache-2.0.
