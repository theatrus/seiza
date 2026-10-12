# Depth planes for parallax videos

A parallax video puts the whole starless image on one plane at the target's
distance. Stars fly at their Gaia distances, and galaxies are lifted onto the
far field, but every cloud of gas and dust moves as if it lay at the target's
distance. Many fields hold clouds at other depths: a bright nebula in front of
the target, a dark lane between us and it, or a faint shell far behind. This
note describes how the scene holds several such depths.

## The scene before depth planes

- `Scene::background` is the starless image as one plane at
  `background_distance_pc`. `Scene::leftover` holds star light that no star
  took, on a plane at the star field's distance.
- Stars are sprites at their own distances. A sprite keeps its first-frame
  size as the camera nears it, swelling only a little, and fades as the
  camera passes it.
- `lift_object` cuts a galaxy out of the starless image onto a sprite at
  1e8 pc. It fills the hole from a ring of light around the galaxy, and the
  sprite plus the filled image add back to the starless image. That rule is
  what keeps the first frame equal to the photograph.
- Layers add light (a screen blend done as addition), so for glowing gas the
  order of layers does not matter. Only dust, which blocks light, depends on
  order.
- The dust sits on the background plane alone: `Scene::flies_at` assumes
  every opaque cloud lies at the target's distance.

## The model

`Scene::planes` holds regions lifted off the starless image, smallest first.
A `Plane` is a cropped image of light with its place in the image, its
distance and the region it came from. The background keeps the light no plane
took, and the fill behind each plane. The rule stays: at frame 0 the
background, the planes, the leftover and the sprites add back to the
photograph.

## Splitting the light

A `PlaneMask` is an image, white where the image belongs to the plane and
black elsewhere, at any size of the image's shape. `Region::from_mask` reads
it at the image's size, each pixel taking the mask's share at its centre, and
crops it to the pixels it covers plus a dark border.

`lift_region` then fills in what lies behind the region and takes, in the
region's share of each pixel, the light above the fill. The ring fill that
works for a small galaxy smears across a large cloud, so planes use push-pull
(Gortler et al., 1996):

1. Weight each pixel by how much of it lies outside the region.
2. Average the weighted light into coarser and coarser levels, each
   coarse pixel known as far as the pixels under it hold a whole pixel's
   worth.
3. Blend back down: each level takes the share of a pixel it does not know
   from the level above, drawn up to its size.

Deep in a hole the fill is smooth; at its edge it stays close to the light
just outside, so the plane's light falls to nothing at the mask's edge even
when the mask is hard.

A plane takes only light above the fill. A dark cloud blocks light rather
than giving it, so it gives its plane nothing; dark clouds need dust per
plane (below).

Where regions nest, the smaller is lifted first. It takes its own light above
its local fill, and the larger then takes the rest, the fill under the
smaller included. Each depth keeps its own light.

## Drawing

Planes are drawn as the background is, in perspective, so they grow as the
camera nears them. Drawing them as sprites would be wrong: a sprite keeps a
star's size. Three more rules:

- A plane fades as the camera flies through it, by the same growth limits
  that fade a star, and is gone once behind the camera.
- A plane is drawn only over the part of the frame its box reaches, so a
  small cloud costs little. A full-frame plane costs about one more
  background draw a frame.
- A plane that the image's edge cuts off would show that edge when it slides.
  `Shot::fitted` guards those edges, as it guards the far star field's, and
  only those: the mask fades a plane out before its other edges.

A star within `CutOptions::embedded` of a plane's distance, where the plane
covers at least half its pixel, joins the plane, as a star at the
background's distance joins the background. This is most often the star
lighting a reflection nebula.

`Scene::depth_at(x, y)` gives the depth under a point: the smallest plane
covering it, else the background. Labels use it, so a label on a cloud moves
with the cloud.

## Where the distances come from

**Given by the user**

- `seiza parallax-video --plane MASK=PARSECS`, repeatable, and `planes` in
  the C ABI's request, each `{mask, distanceParsecs, name}`.
- Later: `--export-planes DIR` writes the planes in use, automatic ones
  included, as masks with their distances, to edit and give back, as
  `--export-dust-map` does for the dust. `--object-distance NAME=PARSECS`
  overrides a catalogue distance.
- In the apps, the Source step could list suggested planes, each with an on
  and off switch and a distance field, and import a mask. Painting a region
  could come later.

**Automatic (later)**

- **Catalogued objects.** Take objects in the field that have a distance in
  the object distance data and an outline (an OpenNGC contour, else the
  catalogue ellipse). Fit the outline to the image's light, as
  `lift_object` does. Split an object off only when its distance differs
  from the plane around it by about 15% or more. The data records how each
  distance was found (`DistanceBasis`): use measured distances, those of a
  containing object and those of a nearby cloud, never `KindDefault`, a
  typical distance for the object's kind.
- **Dark clouds, from the Gaia stars already matched.** Inside a region the
  dust map marks opaque, compare star counts by distance with the counts
  outside it. The distance where the inside counts fall short is the cloud's
  distance; published cloud distances are measured the same way. This gives
  the counted dust a real depth.
- **Galaxies** still go to the far field.
- **Depth images.** Accept a continuous depth image and cut it into a few
  planes, sharing each pixel between its two nearest planes so no steps show.

## Dust per plane (later)

Each plane could carry its own dust. What lies behind a plane would dim as it
slides behind thicker dust on that plane, the dimming from several planes
multiplying. `Scene::flies_at` would then put a star seen through opaque dust
in front of that cloud's own distance, not the target's. Until then, planes
beyond the background dim by the background's dust, as the leftover plane
does.

## Tours (later)

A tour stop's `dolly` is a share of the way to the background plane, and the
camera aims at the background point at the stop's place. Flying straight in,
every depth along that line of sight stays centred, so a stop on a near cloud
works. With sideways travel, the cloud drifts a little from the centre. A
later change could aim and dolly a stop toward `Scene::depth_at` its place.

## Risks

- **Cost.** Each plane costs a draw over the share of the frame it covers.
  There is no cap on the number of planes; the notes say how much of the
  image each covers.
- **Fill quality.** A large near plane on a long sideways move shows much of
  the fill behind it, which is smooth and can look smeared. `Shot::fitted`
  could later limit the sideways travel by how much fill would show.
- **Wrong distances look wrong.** A bad catalogue distance gives strong,
  wrong motion. Every plane is reported with its distance, and automatic
  ones will say how the distance was found, so the user can see it and
  override it.
- **Flat planes.** Real clouds have depth of their own; planes are flat
  cut-outs. That is fine for the effect.

## Order of work

1. The library: planes, `lift_region`, drawing, fading, guarded edges,
   stars joining planes, labels at a plane's depth, and tests that frame 0
   is the photograph and that nearer planes move more. `--plane` and the C
   ABI's `planes`. (Done.)
2. `--export-planes` and `--object-distance`.
3. Automatic planes from catalogued objects.
4. Dark-cloud distances from Gaia star counts, and dust per plane.
5. Tour stops at a plane's depth, and the apps' interface.
