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
- `--export-planes DIR` writes the planes in use, automatic ones included,
  as masks with their distances in `planes.txt`, to edit and give back with
  `--plane-file`, as `--export-dust-map` does for the dust.
  `--object-distance NAME=PARSECS` (the C ABI's `distanceOverrides`)
  overrides a catalogue distance, for the nebula's distance and the
  automatic planes.
- In the apps, the Source step could list suggested planes, each with an on
  and off switch and a distance field, and import a mask. Painting a region
  could come later.

**Automatic**

- **Catalogued objects** (`--auto-planes`, the C ABI's `autoPlanes`). Take
  the nebulae in the field that have a distance in the object distance
  data. The data records how each distance was found (`DistanceBasis`): use
  measured distances and those borrowed from an object around them, never
  `KindDefault`, a typical distance for the object's kind. Split an object
  off only when its distance differs from the nebula's by 15% or more
  (`AUTO_PLANE_DEPTH`), and not when its range of distances takes in the
  nebula's. Leave out an object whose ellipse holds the point the nebula's
  distance was taken at: it is the nebula, or holds it, and would carry it
  off. The region is the outermost brightness level of its OpenNGC outline,
  softened, or else its catalogue ellipse, moved onto the light and grown
  ring by ring until the light comes near its surroundings or levels off,
  at most three times the ellipse (`Region::from_extent`). On the Big Q
  field this finds Sh2-170 at 2790 pc, behind NGC 7822 at 900 pc, and
  vdB 2 at 717 pc, in front.
- **Dark clouds, from the Gaia stars already matched (later).** Inside a region the
  dust map marks opaque, compare star counts by distance with the counts
  outside it. The distance where the inside counts fall short is the cloud's
  distance; published cloud distances are measured the same way. This gives
  the counted dust a real depth.
- **Galaxies** still go to the far field.
- **Depth images (later).** Accept a continuous depth image and cut it into a few
  planes, sharing each pixel between its two nearest planes so no steps show.

## Dust per plane (later)

Each plane could carry its own dust. What lies behind a plane would dim as it
slides behind thicker dust on that plane, the dimming from several planes
multiplying. `Scene::flies_at` would then put a star seen through opaque dust
in front of that cloud's own distance, not the target's. Until then, planes
beyond the background dim by the background's dust, as the leftover plane
does.

## Tours

A tour stop's `dolly` is a share of the way to the background plane. A stop
on a depth plane centres its place on that plane: the stop's `depth`
(`depth=PARSECS` in a stop, else the plane under its place) says which. A
camera that has moved sideways sees a nearer or farther plane shifted
against the background, so each stop aims at the background point that
puts its place at the frame's centre (`aim_through`), and the camera glides
between those background points as before, so it keeps to the image on the
way.

The camera cannot centre a plane far behind the background by moving
sideways: it would have to stand far off to the side. Turning `near` of its
way (the share left to the background) puts it on the line of sight from
Earth through the stop's place, where every depth along that line is
centred, at any depth. The planner gives each stop on a plane that pan, and
the fit keeps it rather than scaling it down with the others. To and from
such a stop, the pan is blended as a share of the camera's nearness, so the
camera keeps to that line rather than turning past it as it nears.

The planner frames a target on a plane at its own depth (`framing`). A
nearer plane grows sooner, so the camera flies less far. The camera never
flies past `MOST_DOLLY`, so a target beyond the background takes the rest of
its growth from the lens. Along the line of sight the frame shows more of a
farther plane, the far star field most, and that must stay inside the
image: near the image's edge, the camera flies less far and zooms more,
until the target lies in the middle four fifths of its frame.

A plane's cut edges are judged by its light, not its region: an edge counts
as cut only where the plane holds more than a twentieth of its body's mean
light along it.

## Risks

- **Cost.** Each plane costs a draw over the share of the frame it covers.
  There is no cap on the number of planes; the notes say how much of the
  image each covers.
- **Fill quality.** A large near plane on a long sideways move shows much of
  the fill behind it, which is smooth and can look smeared. `Shot::fitted`
  could later limit the sideways travel by how much fill would show.
- **Wrong distances look wrong.** A bad catalogue distance gives strong,
  wrong motion. Every plane is reported with its distance and how it was
  found, so the user can see it and override it.
- **Flat planes.** Real clouds have depth of their own; planes are flat
  cut-outs. That is fine for the effect.

## Order of work

1. The library: planes, `lift_region`, drawing, fading, guarded edges,
   stars joining planes, labels at a plane's depth, and tests that frame 0
   is the photograph and that nearer planes move more. `--plane` and the C
   ABI's `planes`. (Done.)
2. `--export-planes`, `--plane-file` and `--object-distance`. (Done.)
3. Automatic planes from catalogued objects, and tour stops at a plane's
   depth. (Done.)
4. Dark-cloud distances from Gaia star counts, and dust per plane.
5. The apps' interface.
