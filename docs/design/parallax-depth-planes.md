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

Where the region is brighter than the fill, the plane takes the light
above it; where darker, a cloud blocks light (dust per plane, below). Light
within a twentieth of the fill either way is the background's own
variation, and the plane takes nothing there. A glowing nebula found in the
catalogue takes only its light: the darkness about it is dust, which stays
with its cloud.

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
  automatic planes. A nebula the catalogue places at the distance of one
  around it (vdB 2 and Ced 1 in the Big Q) takes a distance given for that
  one, so the two stay one plane.
- In the apps, the Source step has a Depth planes section: a switch for
  automatic planes, the planes found at the last preparation, and a
  distance field for each plane a catalogued object places (the summary's
  `object`). Switching planes off one by one, importing a mask and painting
  a region could come later.

**Automatic**

- **Catalogued objects** (`--auto-planes`, the C ABI's `autoPlanes`). Take
  the nebulae in the field that have a distance in the object distance data.
  The data records how each distance was found (`DistanceBasis`): use
  measured distances and those borrowed from an object around them, never
  `KindDefault`, a typical distance for the object's kind. Split an object
  off only when its distance differs from the nebula's by 15% or more
  (`AUTO_PLANE_DEPTH`), and not when its range of distances takes in the
  nebula's. Leave out an object whose ellipse holds the point the nebula's
  distance was taken at: it is the nebula, or holds it, and would carry it
  off. Leave out too one so large it spans most of the image, one that
  cannot be cut out, and one whose region would cover more than a third of
  the image (`AUTO_PLANE_COVERAGE`) unless its distance is given: such a
  nebula is the scene's backdrop, and lifting it would move most of the
  picture off the nebula's depth. An object inside one left out, whose
  distance the catalogue borrows from it, stays with it, so one structure is
  not torn in two. The region is the outermost brightness level of its
  OpenNGC outline, softened, or else its catalogue ellipse, moved onto the
  light and grown ring by ring until the light comes near its surroundings
  or levels off, at most three times the ellipse (`Region::from_extent`). On
  the Big Q field this finds Sh2-170 at 2790 pc, behind NGC 7822 at 900 pc,
  and vdB 2 at 717 pc, in front. On the North America Nebula, given at 800
  pc, LBN 371 at a kinematic 1500 pc would have covered 85% of the image. On
  a wide field about WR 134 in Cygnus, given at 1850 pc, Sh2-109 and LBN 205
  span the image and LBN 162 runs off its edge, yet a dozen nebulae inside
  them took their distances, 1500, 9300 and 12700 pc, and between them
  covered most of the image; left with them, three planes remain: Sh2-104 in
  front, and IC 1310 and SNR G073.9+00.9 behind.
- **Dust clouds**, dark nebulae among them, by the clouds the star counts
  find and the Gaia stars' distances (below).
- **Galaxies** still go to the far field.
- **Depth images (later).** Accept a continuous depth image and cut it into a few
  planes, sharing each pixel between its two nearest planes so no steps show.

## Dust per plane

A plane can block light as well as give it. Where its region is darker than
the fill, below 95% of it, `lift_region` gives the plane a `Transmission`:
each colour's share of the fill the image shows there, so a reddish cloud
lets through more red than blue. The starless image under the cloud is
restored to the light behind it, 95% of the fill, so the restored light
times the transmission is the image again. The cloud's texture, noise and
all, lies in its transmission and moves with it.

Drawing, every layer behind a dark cloud is dimmed by it at the cloud's
own depth, the dimming of several clouds multiplying:

- a layer whose light was restored, the background and the planes lifted
  after the cloud, by the light the cloud lets through where it lies now;
- the stars, the far star field and the planes lifted before the cloud,
  photographed through it, by the change in that light from where they were
  photographed through it, brightening at most half again, as the dust on
  the background's plane does.

At the first frame every factor gives back the image, but where the frame
samples between pixels' centres: there the cloud and the light behind,
sampled apart, multiply to a little more or less than the image's own
light at a hard edge, at most a step of an 8-bit frame. As the camera
passes through a cloud, it fades, and once behind the camera it blocks
nothing.

A star behind a dark cloud, seen where the cloud lets through 5% or less,
must lie in front of it: `Scene::flies_at` puts it a tenth short of the
cloud. The dust counted from the star counts on the background's plane
clears as far as a dark cloud's plane in front of it blocks the light
itself, so the cloud is not counted twice.

Only a plane in front of the background blocks light: a cloud behind the
nebula cannot darken it, and restoring the light behind it would brighten
the first frame. Its darkness stays in the background. A plane lifted
after a dark cloud, and lying in front of it, takes its light from the
restored image, so it takes that light dimmed by the cloud, as it was
photographed: the cloud never dims a plane in front of it.

## Dust clouds

A dark nebula is the thickest part of a cloud of dust that reaches out
around it, glowing faintly where starlight falls on it and hiding the stars
behind it. Cut out alone, a dark nebula slides across its own dust. So the
dust moves by clouds:

1. `Clouds::find` joins the cells of the star counts' dust map that let
   through 70% or less, faint dust included, into clouds. In a field such
   as the Iris, the dust joins into one web.
2. A catalogued dark nebula belongs to the cloud a quarter or more of its
   region lies in.
3. The cloud under the point the nebula's distance was taken at, or within
   two cells of it, is the nebula's own, and stays on the background with
   every dark nebula in it. The notes name them.
4. Any other cloud at least 15% in front of the nebula moves as one plane,
   its region the cloud's cells fading out over a cell, with its dark
   nebulae's regions added: at the distance of its largest dark nebula
   that has one, else where the Gaia stars seen through it run short.
   Clouds behind the nebula stay, as a plane there blocks nothing, and so
   does a cloud that would cover more than a third of the image, unless a
   distance is given for its dark nebula.
5. A dark nebula in no cloud stands alone, on a plane of its own as before
   when it lies at least 15% in front of the nebula, by its catalogued
   distance or the Gaia stars seen through it. Its region is its
   catalogue ellipse grown to where the image brightens to its
   surroundings (`Region::from_dark_extent`), as the catalogue does not
   outline dark nebulae.

### Distances from star counts

Gaia counts the stars to its magnitude limit however brightly the image
glows, and gives their distances. In front of a cloud there are as many
stars at each distance as over clear sky; behind it, fewer. Published
distances to dark clouds are measured the same way.

`counted_distance` takes the distances of the Gaia stars over a cloud's
core, the cells letting through 30% or less (else its thickest half), and
of those over clear sky, the cells letting through 90% or more. The thin
dust about a cloud hides few stars, and counted in, would hide where the
cloud lies. For each split, after the k-th nearest star of the cloud, it
compares the cloud's stars with the clear sky's in front and behind, each
as a share, so the sky thinning across the image does not read as a
cloud. The likeliest split must be much likelier than none (a natural log
of 12 or more), leave at most 70% of the stars showing behind, have five
or more stars in front, and those as many for their area as at least half
the clear sky's, or the cloud lies in front of them too. The cloud lies
past its last star in front, by as far as the next would most likely have
come. Its range takes in the splits within a natural log of 4 of the best,
widened by a tenth each way, as Gaia's distances are no closer. On test
skies of 1,500 stars, a cloud hiding four fifths of the stars behind it
comes within a tenth of its distance; with fewer stars, or thinner dust,
fewer clouds are placed, and less closely. No cloud was found in 250
skies without one.

On the Iris, given at 355 pc, all of the field's dust joins the nebula's
own cloud, and LDN 1167, 1168, 1170 and 1171 stay with it; the counts put
that dust at 357 pc (308 to 676). On the Big Q, the nebula's cloud covers
most of the image and holds its dark nebulae; the counts put it at 1160 pc
(1040 to 1280), behind the nebula's catalogued 900 pc.

The counted dust on the background's plane keeps the stars' dimming for a
cloud moved onto a plane of its own: the plane carries only the darkness
the starless image shows. As the dimming is coarse and smooth, its slower
motion at the background's depth hardly shows.

## Tours

A tour stop's `dolly` is a share of the way to the background plane. A stop
on a depth plane centres its place on that plane: the stop's `depth`
(`depth=PARSECS` in a stop, else the plane under its place) says which. A
tour file keeps the depths it was planned with, so after the planes change
a stop can name a depth the image no longer shows at its place, which
frames its place off centre; a warning names each such stop, more than 15%
from the depth under it. A
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
4. Dust per plane, and catalogued dark clouds as planes. (Done.)
5. Dust clouds: dark nebulae moving with the dust they lie in, and
   distances from Gaia star counts for clouds the catalogue does not place.
   (Done.)
6. The apps' interface: the Depth planes section in seiza-mac (#64) and
   seiza-win (#53). (Done.)
