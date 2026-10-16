//! Dust in flats: one definition of what dust looks like, how it is found,
//! and how it is matched between two sessions.
//!
//! Dust blocks part of the light cone that reaches the sensor, so its shadow
//! is the shape of that cone where it meets the dust: the aperture, out of
//! focus. How large the shadow is follows from how far the dust sits in front
//! of the sensor, `diameter ≈ distance / focal ratio`, so one feature's size
//! also says where to clean ([`Optics::distance_mm`]). That gives the shapes
//! this module reports ([`DustShape`]):
//!
//! - **Compact** and **irregular** dips: dust on or near the sensor window,
//!   a few pixels to a few dozen across, sharp, often not round. Reported
//!   with their extent and matched by overlap.
//! - **Disks**: dust far up the train of an unobstructed telescope, such as
//!   a refractor's filter, casts a large filled disk, often well under 1%
//!   deep. Every disk from one surface has nearly the same diameter.
//! - **Donuts**: the same through a centrally obstructed telescope (SCT,
//!   RASA, Newtonian): a dark ring around the secondary's bright shadow. The
//!   bright centre belongs to the donut; it is neither a separate spot nor
//!   allowed to cancel the ring's depth.
//! - **Patches**: large shadows that are not one circle, such as several
//!   overlapping disks the rim vote could not tell apart.
//!
//! **Vignetting** ([`Vignetting`]) is reported apart from dust: a large,
//! smooth shadow anchored to an edge or corner, such as an off-axis guider's
//! prism. When it changes between sessions it changes the flat, but it is
//! never "new dust".
//!
//! # How the shapes are found
//!
//! Every measure runs on a **residual**: the plane divided by a smooth
//! surface ([`super::flat_response::metric`]'s robust degree-4 fit), minus
//! one, with stars and unusable pixels left out and the sensor's row and
//! column pattern taken out. Large-scale mottle of a few tenths of a percent
//! remains in a single session; it is what defeats a plain threshold, and
//! the measures below are built to see past it.
//!
//! - Disks and donuts are found by their **rim**. A shadow's rim is sharp: a
//!   0.3% disk on a 1,024-pixel plane climbs over about six pixels, some
//!   thirty times the gradient of the mottle. Rim pixels vote for a centre
//!   one radius against their gradient, each vote marking which sector of
//!   the rim it came from ([`DustOptions::rim_sectors`] of 16 are needed:
//!   "darker inside, most of the way round"). A straight edge, a gradient or
//!   a star's ring fills few sectors. A candidate is then measured sector by
//!   sector, ring inside against ring outside; one whose sectors disagree,
//!   such as a circle drawn around several disks, is refused.
//! - Compact dips are connected regions of the smoothed residual below
//!   [`DustOptions::compact_sigma`] times its noise, after a wide local
//!   median. Small ones must clear a threshold raised for the number of
//!   pixels tried ([`DustOptions::false_dips`]), since a plane of a million
//!   pixels holds a handful of 5σ noise dips.
//! - Vignetting is a heavily smoothed residual below
//!   [`DustOptions::vignetting_depth`] that touches the plane's border.
//!
//! # Changes between sessions
//!
//! [`find_changes`] looks at the ratio of two sessions, where what both
//! share cancels, with the same measures on its darker side (new dust) and
//! its brighter side (dust gone). Each session's own map is matched too
//! ([`match_features`]): a shadow both hold within
//! [`DustOptions::move_tolerance`] — the few pixels focus travel moves an
//! SCT's donuts — is the same dust, and the crescent it leaves in the ratio
//! is reported as moved within tolerance rather than as new. A compact spot
//! where the brighter session holds a point bump is a star a tracked sky
//! flat kept, and is dropped.

use crate::flat_response::metric::{self, Plane};
use rayon::prelude::*;
use seiza_imgproc::components::{Connectivity, connected_components};
use serde::{Deserialize, Serialize};

/// Bump when equal planes may give different dust.
pub const DUST_ALGORITHM_VERSION: u32 = 1;

/// Sectors a rim is split into.
const SECTORS: usize = 16;
/// White noise left after a Gaussian of one pixel, as a share of the
/// pixel noise: `1 / (2 √π σ)`.
const SMOOTHED_NOISE: f32 = 0.282;
/// A corner of an image circle is reported as vignetting only when it is
/// this many times as deep as the corners' median.
const IMAGE_CIRCLE_OUTLIER: f32 = 2.0;
/// Widest ring, in plane pixels, either side of a rim when its sharpness is
/// read: about the width of a dust shadow's rim on a 1,024-pixel plane.
const RIM_BAND: f32 = 6.0;
/// Most rim candidates measured, best first.
const MAX_CANDIDATES: usize = 800;
/// Sectors a candidate is measured in, ring inside against ring outside.
const DEPTH_SECTORS: usize = 8;
/// Each tried radius is this factor above the last.
const RADIUS_STEP: f32 = 1.05;
/// The smallest rim tried, in plane pixels; smaller is a compact dip.
const MIN_RIM_RADIUS: f32 = 3.0;
/// Gradients for radii below [`COARSE_FROM_RADIUS`] are read after this
/// much smoothing, in plane pixels...
const FINE_SIGMA: f64 = 1.5;
/// ...and for larger radii after this much, which keeps faint wide rims.
const COARSE_SIGMA: f64 = 3.0;
const COARSE_FROM_RADIUS: f32 = 12.0;
/// Width of the box whose mean the row and column pattern is read against.
const PATTERN_WINDOW: usize = 31;
/// Edge band, as a fraction of each side, where a compact dip is not
/// counted (it may be the border's own fall-off) and where a vignetting
/// shadow must reach.
const BORDER_FRACTION: f32 = 0.02;
/// A sector's depth counts once it holds this many samples on each ring.
const MIN_RING_SAMPLES: usize = 3;
/// Most dust features one map keeps, largest and deepest first.
const MAX_FEATURES: usize = 400;
/// A compact dip inside a disk or donut, once the shadow's profile is taken
/// out, counts only when it is this many times as deep as the shadow.
const INSIDE_SHADOW_DEPTH: f32 = 1.5;

/// The light cone at the sensor, to size dust shadows and say where the dust
/// sits.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Optics {
    /// Focal ratio at the sensor, with any reducer or extender.
    pub focal_ratio: f32,
    /// Sensor pixel size in micrometres (before binning).
    pub pixel_size_um: f32,
    /// Central obstruction as a fraction of the aperture's diameter; 0 for
    /// a refractor.
    #[serde(default)]
    pub obstruction: f32,
}

impl Optics {
    /// Whether the numbers can size a shadow.
    pub fn is_usable(&self) -> bool {
        self.focal_ratio.is_finite()
            && self.focal_ratio > 0.0
            && self.pixel_size_um.is_finite()
            && self.pixel_size_um > 0.0
            && (0.0..1.0).contains(&self.obstruction)
    }

    /// Diameter in sensor pixels of the shadow cast by dust `distance_mm` in
    /// front of the sensor.
    pub fn shadow_diameter_px(&self, distance_mm: f32) -> f32 {
        distance_mm * 1000.0 / (self.focal_ratio * self.pixel_size_um)
    }

    /// How far in front of the sensor dust sits whose shadow is
    /// `diameter_px` sensor pixels across, in millimetres.
    pub fn distance_mm(&self, diameter_px: f32) -> f32 {
        diameter_px * self.focal_ratio * self.pixel_size_um / 1000.0
    }

    /// The light cone from parts a host may each leave out: none gives
    /// `None`; a focal ratio and a pixel size give the cone, unobstructed
    /// when `obstruction` is left out. Any other mix, such as an obstruction
    /// alone, is refused rather than quietly ignored.
    pub fn from_parts(
        focal_ratio: Option<f32>,
        pixel_size_um: Option<f32>,
        obstruction: Option<f32>,
    ) -> crate::Result<Option<Self>> {
        match (focal_ratio, pixel_size_um) {
            (None, None) if obstruction.is_none() => Ok(None),
            (Some(focal_ratio), Some(pixel_size_um)) => Ok(Some(Self {
                focal_ratio,
                pixel_size_um,
                obstruction: obstruction.unwrap_or(0.0),
            })),
            _ => Err(crate::Error::FrameQuality(
                "optics need both a focal ratio and a pixel size, and an obstruction needs both"
                    .into(),
            )),
        }
    }
}

/// What a dust feature looks like.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum DustShape {
    /// A small dip, a few pixels to a few dozen across.
    #[default]
    Compact,
    /// A small dip that is not round: elongated or ragged, as dust on the
    /// window often is.
    Irregular,
    /// A filled out-of-focus disk.
    Disk,
    /// An out-of-focus ring around a central obstruction's bright shadow.
    Donut,
    /// A large shadow that is not one circle: overlapping out-of-focus
    /// shadows, each with a rim, that do not separate. Its radius is that of
    /// a disc of the same area.
    Patch,
}

impl DustShape {
    /// A disk, donut or patch: the out-of-focus shadow of dust up the train,
    /// found on the binned plane, rather than a dip of window dust.
    pub fn is_large(self) -> bool {
        matches!(self, Self::Disk | Self::Donut | Self::Patch)
    }

    /// A disk or a donut: one shadow with a measured circular rim.
    pub fn is_round(self) -> bool {
        matches!(self, Self::Disk | Self::Donut)
    }
}

/// One dust shadow. Positions and sizes are in sensor pixels.
///
/// Construct with [`Default::default`] and set what is known; more fields may
/// be added without a major version.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DustFeature {
    /// Centre column: the centroid of a compact dip, the rim's centre of a
    /// disk or donut.
    pub x: f32,
    /// Centre row.
    pub y: f32,
    /// First column the shadow covers.
    pub left: u32,
    /// First row the shadow covers.
    pub top: u32,
    /// Last column the shadow covers.
    pub right: u32,
    /// Last row the shadow covers.
    pub bottom: u32,
    /// Sensor pixels the shadow covers.
    pub pixels: u32,
    /// How much light it takes, as a fraction of the response around it:
    /// 0.1 means a tenth less light. A compact dip's deepest point; a disk's
    /// or donut's ring, never its bright centre.
    pub depth: f32,
    /// What it looks like.
    #[serde(default)]
    pub shape: DustShape,
    /// Radius: a disk's or donut's rim; for a compact dip, that of a disc of
    /// the same area. 0 in measurements made before shapes were reported.
    #[serde(default)]
    pub radius: f32,
    /// A donut's hole: the radius of its bright centre as a fraction of
    /// [`Self::radius`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hole: Option<f32>,
    /// How far in front of the sensor the dust sits, in millimetres, from
    /// the shadow's diameter and the [`Optics`]; only for disks and donuts
    /// measured with optics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distance_mm: Option<f32>,
    /// In how many of a session's flats it shows, where it was found on the
    /// session's stack ([`find_session_dust`]). Dust holds still across
    /// flats shot a minute apart; noise does not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seen_in: Option<SeenIn>,
}

/// In how many flats of a session a feature shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeenIn {
    /// Flats it shows in.
    pub frames: u32,
    /// Flats looked at.
    pub of: u32,
}

impl SeenIn {
    /// Whether that many flats can say much: with fewer than
    /// [`MIN_PERSISTENCE_FRAMES`], a feature's presence in all of them is
    /// weak evidence.
    pub fn is_reliable(&self) -> bool {
        self.of as usize >= MIN_PERSISTENCE_FRAMES
    }
}

/// Fewest flats whose agreement makes persistence a strong test.
pub const MIN_PERSISTENCE_FRAMES: usize = 4;

impl DustFeature {
    fn compact(
        x: f32,
        y: f32,
        bounds: (u32, u32, u32, u32),
        pixels: u32,
        depth: f32,
        shape: DustShape,
    ) -> Self {
        Self {
            x,
            y,
            left: bounds.0,
            top: bounds.1,
            right: bounds.2,
            bottom: bounds.3,
            pixels,
            depth,
            shape,
            radius: (pixels as f32 / std::f32::consts::PI).sqrt(),
            hole: None,
            distance_mm: None,
            seen_in: None,
        }
    }
}

/// Which part of the border a vignetting shadow hangs from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Edge {
    Top,
    Bottom,
    Left,
    Right,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// A large, smooth shadow anchored to the border: an off-axis guider's
/// prism, a filter holder, a dew shield. Not dust.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Vignetting {
    /// The edge or corner it hangs from.
    pub edge: Edge,
    /// Centroid column, sensor pixels.
    pub x: f32,
    /// Centroid row.
    pub y: f32,
    /// Sensor pixels it covers below the vignetting depth.
    pub area: f32,
    /// Its deepest point after smoothing, as a fraction of the response.
    pub depth: f32,
}

/// Settings for [`find_dust`] and [`find_changes`]. One set serves every
/// caller, so a flat's spots and a comparison's agree.
///
/// Deliberately not `#[non_exhaustive]`, like
/// [`crate::MatchTolerances`]: a new setting changes what is found, and
/// `DustOptions { optics, ..Default::default() }` is how a host writes it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DustOptions {
    /// The light cone, when known: gives each disk and donut a distance and
    /// a donut its hole.
    pub optics: Option<Optics>,
    /// Sectors of 16 in which rim pixels must vote for a centre before the
    /// circle is measured. Where the border cuts a rim, the same share of
    /// the sectors on the plane, and at least half the rim must be on it.
    pub rim_sectors: usize,
    /// A rim pixel's gradient is at least this many robust sigmas of the
    /// plane's gradient noise.
    pub edge_sigma: f32,
    /// Shallowest dust counted, as a fraction of the response.
    pub min_depth: f32,
    /// A disk's or donut's rim, read on thin rings either side of it, must
    /// carry at least this share of its depth, darker inside in three
    /// sectors of four. A circle drawn around several shadows, or around a
    /// patch of mottle, has no sharp rim of its own.
    pub rim_contrast: f32,
    /// That rim contrast must be at least this many times its own noise,
    /// from the plane's pixel noise and the pixels the rings hold. A session
    /// response of ten flats has a third of one flat's noise, which is what
    /// lets faint disks through.
    pub rim_significance: f32,
    /// A compact dip reaches this many robust sigmas of the smoothed
    /// residual.
    pub compact_sigma: f32,
    /// How many noise dips a plane may report. Dips of up to four plane
    /// pixels, or twice [`Self::min_compact_area`] if that is more, must
    /// clear the threshold this implies for the number of pixels tried.
    pub false_dips: f64,
    /// Smallest compact dip, in sensor pixels.
    pub min_compact_area: f32,
    /// Vignetting: a border-anchored shadow at least this deep after heavy
    /// smoothing.
    pub vignetting_depth: f32,
    /// Two sessions hold the same dust when its centre moved no more than
    /// this many sensor pixels, or [`Self::move_tolerance_share`] of its
    /// radius if that is more: focus travel moves an SCT's donuts, and the
    /// larger ones, cast by dust further up the train, the most.
    pub move_tolerance: f32,
    /// The share of a disk's or donut's radius it may move and still be the
    /// same dust.
    pub move_tolerance_share: f32,
}

impl DustOptions {
    /// How far dust of this radius (sensor pixels) may move between two
    /// sessions and still be the same dust.
    pub fn tolerance_for(&self, radius: f32) -> f32 {
        self.move_tolerance.max(self.move_tolerance_share * radius)
    }
}

impl Default for DustOptions {
    fn default() -> Self {
        Self {
            optics: None,
            rim_sectors: 8,
            edge_sigma: 2.5,
            min_depth: 0.0005,
            rim_contrast: 0.5,
            rim_significance: 5.0,
            compact_sigma: 5.0,
            false_dips: 0.05,
            min_compact_area: 200.0,
            vignetting_depth: 0.004,
            move_tolerance: 25.0,
            move_tolerance_share: 0.15,
        }
    }
}

/// The dust in one plane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DustMap {
    /// [`DUST_ALGORITHM_VERSION`] when it was found.
    pub algorithm_version: u32,
    /// Sensor pixels per plane pixel.
    pub bin: usize,
    /// Disks and donuts first, then compact and irregular dips; deepest
    /// first within each.
    pub features: Vec<DustFeature>,
    /// Border-anchored shadows.
    pub vignetting: Vec<Vignetting>,
    /// Robust noise of the residual after one pixel of smoothing.
    pub sigma: f32,
    /// Share of the plane masked as stars.
    pub star_fraction: f32,
    /// Flats the plane combines: 1 for one flat, the session's count for a
    /// stack from [`find_session_dust`], 0 when not known.
    #[serde(default)]
    pub frames: usize,
}

/// One flat of a session against the session's dust
/// ([`frame_dust`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FrameDust {
    /// Indices into the session's features of those this flat shows.
    pub seen: Vec<usize>,
    /// Features this flat holds that the session's stack does not: a drop
    /// of dew drying, a passing particle, a satellite in a sky flat. They
    /// are not dust of the light path.
    pub transients: Vec<DustFeature>,
}

/// What became of one shadow between two sessions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ChangeKind {
    /// The later session is darker here: dust that arrived.
    New,
    /// The later session is brighter here: dust that left.
    Gone,
    /// Dust both sessions hold, in a different place.
    Moved,
}

/// One difference between two sessions.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DustChange {
    /// New, gone or moved.
    pub kind: ChangeKind,
    /// Whether the later session is darker here.
    pub darker: bool,
    /// Where and what, in the ratio: its depth is how much the ratio departs
    /// there, as a fraction of the background.
    pub feature: DustFeature,
    /// Dust both sessions hold within [`DustOptions::move_tolerance`]: the
    /// crescent a focus shift leaves, not a change of dust.
    pub within_tolerance: bool,
    /// How far the matched dust moved, in sensor pixels.
    pub offset: Option<f32>,
}

/// A vignetting shadow that changed between two sessions.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct VignettingChange {
    /// Whether the later session is darker there.
    pub darker: bool,
    /// Where, and how deep the difference is.
    pub vignetting: Vignetting,
}

/// How the dust of two sessions differs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DustChanges {
    /// [`DUST_ALGORITHM_VERSION`].
    pub algorithm_version: u32,
    /// Changes of dust, deepest first; moved-within-tolerance ones included
    /// and marked.
    pub changes: Vec<DustChange>,
    /// Changes of vignetting.
    pub vignetting: Vec<VignettingChange>,
    /// Robust noise of the ratio's residual after one pixel of smoothing.
    pub sigma: f32,
    /// Share of the plane masked as stars in either session.
    pub star_fraction: f32,
    /// The earlier session's own dust.
    pub earlier: DustMap,
    /// The later session's own dust.
    pub later: DustMap,
    /// Plane pixels each change covers, in [`Self::changes`] order then
    /// [`Self::vignetting`] order: what a change map carries.
    #[serde(skip)]
    pub(crate) footprints: Vec<Vec<usize>>,
}

/// How a feature of one session pairs with one of another.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeatureMatch {
    /// Index into the earlier features; `None` for dust only the later holds.
    pub earlier: Option<usize>,
    /// Index into the later features; `None` for dust only the earlier holds.
    pub later: Option<usize>,
    /// How far the centre moved, in sensor pixels, when both hold it.
    pub offset: Option<f32>,
    /// Both hold it and it moved no more than the tolerance.
    pub within_tolerance: bool,
}

/// Pair the features of two sessions.
///
/// Disks and donuts pair with disks and donuts of a radius within 30%,
/// whose centres are within [`DustOptions::tolerance_for`] their radius
/// (same dust, within tolerance) or within half the radius (moved). Compact
/// and irregular dips pair when their boxes, widened by
/// [`DustOptions::move_tolerance`], overlap: window dust is not round, and
/// its centre wanders with noise. A patch, or a disk one session saw as a
/// dip, pairs with whatever its box overlaps of a radius within a factor of
/// two and a centre within the larger radius: the same shadow, measured
/// another way.
///
/// The pairing is the one with the most pairs and, among those, the least
/// total distance, so one close pair never leaves two others unpaired. What
/// is left is new or gone.
pub fn match_features(
    earlier: &[DustFeature],
    later: &[DustFeature],
    options: &DustOptions,
) -> Vec<FeatureMatch> {
    let mut pairs: Vec<(usize, usize, f32, bool)> = Vec::new();
    for (i, a) in earlier.iter().enumerate() {
        for (j, b) in later.iter().enumerate() {
            if let Some((distance, within)) = pairing(a, b, options) {
                pairs.push((i, j, distance, within));
            }
        }
    }
    let assigned = optimal_pairs(earlier.len(), later.len(), &pairs);
    let mut used_earlier = vec![false; earlier.len()];
    let mut used_later = vec![false; later.len()];
    let mut out = Vec::new();
    for index in assigned {
        let (i, j, distance, within) = pairs[index];
        used_earlier[i] = true;
        used_later[j] = true;
        out.push(FeatureMatch {
            earlier: Some(i),
            later: Some(j),
            offset: Some(distance),
            within_tolerance: within,
        });
    }
    for (i, used) in used_earlier.iter().enumerate() {
        if !used {
            out.push(FeatureMatch {
                earlier: Some(i),
                later: None,
                offset: None,
                within_tolerance: false,
            });
        }
    }
    for (j, used) in used_later.iter().enumerate() {
        if !used {
            out.push(FeatureMatch {
                earlier: None,
                later: Some(j),
                offset: None,
                within_tolerance: false,
            });
        }
    }
    out
}

/// Whether two features may be the same dust: how far apart their centres
/// are, and whether that is within tolerance.
fn pairing(a: &DustFeature, b: &DustFeature, options: &DustOptions) -> Option<(f32, bool)> {
    let distance = (a.x - b.x).hypot(a.y - b.y);
    let larger = a.radius.max(b.radius);
    let size_ratio = larger / a.radius.min(b.radius).max(1e-3);
    let t = options.move_tolerance.max(0.0);
    let boxes_overlap = a.left as f32 - t <= b.right as f32
        && b.left as f32 <= a.right as f32 + t
        && a.top as f32 - t <= b.bottom as f32
        && b.top as f32 <= a.bottom as f32 + t;
    if a.shape.is_round() && b.shape.is_round() {
        if size_ratio > 1.3 {
            None
        } else if distance <= options.tolerance_for(larger) {
            Some((distance, true))
        } else if distance <= 0.5 * larger {
            Some((distance, false))
        } else {
            None
        }
    } else if !a.shape.is_large() && !b.shape.is_large() {
        boxes_overlap.then_some((distance, true))
    } else {
        // A patch's centre is its circles' mean, and a dip's that of the part
        // that cleared the threshold: neither is a rim's centre, so a pair
        // is within tolerance up to half the smaller radius.
        let smaller = a.radius.min(b.radius);
        (boxes_overlap && size_ratio <= 2.0 && distance <= larger).then_some((
            distance,
            distance <= options.tolerance_for(larger).max(0.5 * smaller),
        ))
    }
}

/// The candidate pairs `(earlier, later, distance, _)` to keep: as many as
/// can be taken without using a feature twice, and among those the least
/// total distance. Indices into `pairs`, by earlier feature.
fn optimal_pairs(earlier: usize, later: usize, pairs: &[(usize, usize, f32, bool)]) -> Vec<usize> {
    // Features that share no candidate cannot affect each other's pairing:
    // each connected group is solved on its own, and most are one pair.
    let mut group: Vec<usize> = (0..earlier + later).collect();
    fn root(group: &mut [usize], mut i: usize) -> usize {
        while group[i] != i {
            group[i] = group[group[i]];
            i = group[i];
        }
        i
    }
    for (i, j, _, _) in pairs {
        let (a, b) = (root(&mut group, *i), root(&mut group, earlier + j));
        group[a.max(b)] = a.min(b);
    }
    let mut members: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for (index, (i, _, _, _)) in pairs.iter().enumerate() {
        let r = root(&mut group, *i);
        members.entry(r).or_default().push(index);
    }
    let mut chosen = Vec::new();
    for list in members.values() {
        if list.len() == 1 {
            chosen.push(list[0]);
            continue;
        }
        let mut rows: Vec<usize> = list.iter().map(|k| pairs[*k].0).collect();
        let mut cols: Vec<usize> = list.iter().map(|k| pairs[*k].1).collect();
        rows.sort_unstable();
        rows.dedup();
        cols.sort_unstable();
        cols.dedup();
        // Each pair costs its distance less a sum larger than all of them,
        // so one more pair always beats a shorter total; a row may also take
        // one of its own spare columns, at no cost, to stay unpaired.
        let big = 1.0 + list.iter().map(|k| f64::from(pairs[*k].2)).sum::<f64>();
        let width = cols.len() + rows.len();
        let mut cost = vec![vec![0.0_f64; width]; rows.len()];
        let mut which = vec![vec![None; width]; rows.len()];
        for k in list {
            let (i, j, distance, _) = pairs[*k];
            let r = rows.binary_search(&i).unwrap_or(0);
            let c = cols.binary_search(&j).unwrap_or(0);
            cost[r][c] = f64::from(distance) - big;
            which[r][c] = Some(*k);
        }
        for (r, column) in assignment(&cost).into_iter().enumerate() {
            if let Some(k) = which[r][column] {
                chosen.push(k);
            }
        }
    }
    chosen.sort_by_key(|k| (pairs[*k].0, pairs[*k].1));
    chosen
}

/// The least-cost assignment of each row of `cost` to its own column, with
/// at least as many columns as rows (the Hungarian method with potentials).
/// The column each row takes.
fn assignment(cost: &[Vec<f64>]) -> Vec<usize> {
    let rows = cost.len();
    let cols = cost.first().map_or(0, Vec::len);
    // One-based, with row and column 0 as the method's sentinel.
    let mut u = vec![0.0_f64; rows + 1];
    let mut v = vec![0.0_f64; cols + 1];
    let mut owner = vec![0usize; cols + 1];
    let mut way = vec![0usize; cols + 1];
    for row in 1..=rows {
        owner[0] = row;
        let mut column = 0usize;
        let mut least = vec![f64::INFINITY; cols + 1];
        let mut used = vec![false; cols + 1];
        loop {
            used[column] = true;
            let current = owner[column];
            let mut delta = f64::INFINITY;
            let mut next = 0usize;
            for j in 1..=cols {
                if used[j] {
                    continue;
                }
                let reduced = cost[current - 1][j - 1] - u[current] - v[j];
                if reduced < least[j] {
                    least[j] = reduced;
                    way[j] = column;
                }
                if least[j] < delta {
                    delta = least[j];
                    next = j;
                }
            }
            for j in 0..=cols {
                if used[j] {
                    u[owner[j]] += delta;
                    v[j] -= delta;
                } else {
                    least[j] -= delta;
                }
            }
            column = next;
            if owner[column] == 0 {
                break;
            }
        }
        loop {
            let previous = way[column];
            owner[column] = owner[previous];
            column = previous;
            if column == 0 {
                break;
            }
        }
    }
    let mut taken = vec![0usize; rows];
    for j in 1..=cols {
        if owner[j] != 0 {
            taken[owner[j] - 1] = j - 1;
        }
    }
    taken
}

/// Find the dust in one plane: a flat session's response or one flat
/// binned ([`metric::bin_frame`]), with or without its pedestal. `bin` is
/// sensor pixels per plane pixel. `None` when the plane is too small or too
/// empty to fit a surface to.
pub fn find_dust(plane: &Plane, bin: usize, options: &DustOptions) -> Option<DustMap> {
    let (width, height) = (plane.width, plane.height);
    let (residual, stars) = own_residual(plane)?;
    let mut found = detect(&residual, width, height, bin.max(1), options);
    // Shadows in three or four corners are the telescope's own image
    // circle, which every flat of the rig holds: not worth a report. A
    // corner much deeper than the others still is: something else, such as
    // a guider's prism, shades it as well.
    let corners: std::collections::HashSet<u8> = found
        .vignetting
        .iter()
        .filter_map(|(v, _)| match v.edge {
            Edge::TopLeft => Some(0),
            Edge::TopRight => Some(1),
            Edge::BottomLeft => Some(2),
            Edge::BottomRight => Some(3),
            _ => None,
        })
        .collect();
    if corners.len() >= 3 {
        let mut depths: Vec<f32> = found.vignetting.iter().map(|(v, _)| v.depth).collect();
        let typical = metric::median_in_place(&mut depths);
        found
            .vignetting
            .retain(|(v, _)| v.depth > IMAGE_CIRCLE_OUTLIER * typical);
    }
    let star_fraction = stars.iter().filter(|star| **star).count() as f32 / (width * height) as f32;
    Some(DustMap {
        algorithm_version: DUST_ALGORITHM_VERSION,
        bin: bin.max(1),
        features: found
            .features
            .into_iter()
            .map(|(feature, _)| feature)
            .collect(),
        vignetting: found.vignetting.into_iter().map(|(v, _)| v).collect(),
        sigma: found.sigma,
        star_fraction,
        frames: 0,
    })
}

/// Find the dust of one flat session on its stack, and say in how many of
/// its flats each feature shows.
///
/// `binned` holds the session's flats binned ([`metric::bin_frame`]), raw
/// ADU; `pedestal` is the ADU the camera adds to every pixel, taken out
/// before depths are read. The flats are combined as
/// [`metric::build_response`] does, so a stack of ten has about a third of
/// one flat's noise: faint disks and donuts clear the floor there that they
/// cannot in one flat.
///
/// Each feature is then measured in every flat on its own, without being
/// searched for again: a disk's rim contrast, a compact dip's depth against
/// its surroundings. Dust shows in nearly every flat of a session shot
/// minutes apart; a noise dip in at most one. A compact dip must show in at
/// least 60% of the flats (all of them with fewer than
/// [`MIN_PERSISTENCE_FRAMES`]). A disk or donut, which the stack may show
/// when no single flat can, needs 30%; its count says how sure it is.
/// `None` with fewer than two flats or flats of different sizes.
pub fn find_session_dust(
    binned: &[Plane],
    bin: usize,
    pedestal: f32,
    options: &DustOptions,
) -> Option<DustMap> {
    let response = metric::build_response(binned, bin)?;
    let stack = response.without_pedestal(pedestal);
    let mut map = find_dust(&stack, bin, options)?;
    map.frames = binned.len();
    let residuals: Vec<Option<(Vec<f32>, f32)>> = binned
        .par_iter()
        .map(|frame| frame_residual(frame, pedestal))
        .collect();
    let (width, height) = (stack.width, stack.height);
    let seen_by: Vec<Vec<usize>> = residuals
        .iter()
        .flatten()
        .map(|(residual, noise)| {
            (0..map.features.len())
                .filter(|index| {
                    shows_in(residual, *noise, width, height, bin, &map.features[*index])
                })
                .collect()
        })
        .collect();
    keep_persistent(&mut map, &seen_by);
    Some(map)
}

/// Find the dust of one flat session from its stack and each flat's own
/// features, reading no flat again: for a host that keeps a session's
/// stack (a [`metric::FlatResponse`], pedestal out) and each flat's
/// measured features ([`crate::measure_flat`]'s, or [`find_dust`]'s on the
/// flat binned), but not the flats.
///
/// The stack's dust is found as [`find_dust`] finds it. A flat shows a
/// feature when one of its own features pairs with it ([`match_features`]);
/// what is kept is what [`find_session_dust`] keeps, by the same rule. Each
/// flat's view says which of the kept features it shows, and its
/// transients: features of its own that pair with none of the stack's, lie
/// on none of them, and do not recur in most of the other flats (dust the
/// stack is too coarse to hold is not a passing thing). One list per flat,
/// in the order given. `None` when the stack is too small or too empty to
/// fit a surface to.
///
/// Persistence read this way counts only the flats that find a feature on
/// their own, where [`find_session_dust`] measures each feature in each
/// flat at its known place: a faint disk the stack shows may be dropped
/// here that it keeps.
pub fn session_dust_from_lists(
    stack: &Plane,
    bin: usize,
    frames: &[&[DustFeature]],
    options: &DustOptions,
) -> Option<(DustMap, Vec<FrameDust>)> {
    let mut map = find_dust(stack, bin, options)?;
    map.frames = frames.len();
    let t = options.move_tolerance.max(0.0);
    let overlap = |a: &DustFeature, b: &DustFeature| {
        a.left as f32 - t <= b.right as f32
            && b.left as f32 <= a.right as f32 + t
            && a.top as f32 - t <= b.bottom as f32
            && b.top as f32 <= a.bottom as f32 + t
    };
    let mut seen_by: Vec<Vec<usize>> = Vec::with_capacity(frames.len());
    let mut transients: Vec<Vec<DustFeature>> = Vec::with_capacity(frames.len());
    for (index, features) in frames.iter().enumerate() {
        let matches = match_features(&map.features, features, options);
        seen_by.push(
            matches
                .iter()
                .filter(|m| m.later.is_some())
                .filter_map(|m| m.earlier)
                .collect(),
        );
        let others = frames.len().saturating_sub(1) as f32;
        transients.push(
            matches
                .iter()
                .filter(|m| m.earlier.is_none())
                .filter_map(|m| m.later.map(|j| features[j]))
                .filter(|f| {
                    !map.features
                        .iter()
                        .any(|own| (own.x - f.x).hypot(own.y - f.y) < own.radius.max(f.radius))
                })
                .filter(|f| {
                    let recurs = frames
                        .iter()
                        .enumerate()
                        .filter(|(other, list)| {
                            *other != index && list.iter().any(|g| overlap(f, g))
                        })
                        .count();
                    (recurs as f32) < STEADY_SHARE * others
                })
                .collect(),
        );
    }
    let renumber = keep_persistent(&mut map, &seen_by);
    let views = seen_by
        .into_iter()
        .zip(transients)
        .map(|(seen, transients)| FrameDust {
            seen: seen.iter().filter_map(|index| renumber[*index]).collect(),
            transients,
        })
        .collect();
    Some((map, views))
}

/// A compact dip must show in this share of a session's flats to be dust
/// (all of them with fewer than [`MIN_PERSISTENCE_FRAMES`]); a flat's own
/// feature that recurs in this share of the other flats is no transient.
const STEADY_SHARE: f32 = 0.6;
/// A disk, donut or patch, which the stack may show when no single flat
/// can, needs this share of the flats.
const LARGE_SHARE: f32 = 0.3;

/// Count in how many flats each feature of `map` shows (`seen_by`: for each
/// flat looked at, the indices of the features it shows), and keep those
/// that persist: a compact dip in [`STEADY_SHARE`] of the flats (all of
/// them with fewer than [`MIN_PERSISTENCE_FRAMES`]), a large shadow in
/// [`LARGE_SHARE`] (any, with too few flats to say). Returns each old
/// index's new one.
fn keep_persistent(map: &mut DustMap, seen_by: &[Vec<usize>]) -> Vec<Option<usize>> {
    let of = seen_by.len() as u32;
    let mut renumber = Vec::with_capacity(map.features.len());
    let mut kept = Vec::with_capacity(map.features.len());
    for (index, feature) in map.features.iter().enumerate() {
        let frames = seen_by.iter().filter(|seen| seen.contains(&index)).count() as u32;
        let seen = SeenIn { frames, of };
        let persists = if feature.shape.is_large() {
            // A shadow most flats do not show on their own is mottle the
            // stack happened to line up, unless there are too few flats to
            // say.
            !seen.is_reliable() || frames as f32 >= LARGE_SHARE * of as f32
        } else if !seen.is_reliable() {
            frames >= of
        } else {
            frames >= ((STEADY_SHARE * of as f32).ceil() as u32).max(2)
        };
        if persists {
            let mut feature = *feature;
            feature.seen_in = Some(seen);
            renumber.push(Some(kept.len()));
            kept.push(feature);
        } else {
            renumber.push(None);
        }
    }
    map.features = kept;
    renumber
}

/// One flat of a session against the session's dust: which of the
/// session's features it shows, and what it holds that the session's stack
/// does not. `frame` is binned like the session's flats, raw ADU.
pub fn frame_dust(
    frame: &Plane,
    pedestal: f32,
    session: &DustMap,
    options: &DustOptions,
) -> Option<FrameDust> {
    let (residual, noise) = frame_residual(frame, pedestal)?;
    let bin = session.bin.max(1);
    let seen = session
        .features
        .iter()
        .enumerate()
        .filter(|(_, feature)| shows_in(&residual, noise, frame.width, frame.height, bin, feature))
        .map(|(index, _)| index)
        .collect();
    let level = metric::sampled_median(&frame.data);
    let own = Plane {
        width: frame.width,
        height: frame.height,
        data: frame
            .data
            .iter()
            .map(|value| value - pedestal.min(0.5 * level))
            .collect(),
    };
    let found = find_dust(&own, bin, options)?;
    let matches = match_features(&session.features, &found.features, options);
    let transients = matches
        .iter()
        .filter(|m| m.earlier.is_none())
        .filter_map(|m| m.later.map(|j| found.features[j]))
        .filter(|feature| {
            // A large transient must also not sit on a session feature it
            // failed to pair with, such as a disk measured a little larger.
            !session.features.iter().any(|own| {
                (own.x - feature.x).hypot(own.y - feature.y) < own.radius.max(feature.radius)
            })
        })
        .collect();
    Some(FrameDust { seen, transients })
}

/// One flat's residual, its pedestal taken out, and its pixel noise.
fn frame_residual(frame: &Plane, pedestal: f32) -> Option<(Vec<f32>, f32)> {
    let level = metric::sampled_median(&frame.data);
    if !(level.is_finite() && level > 0.0) {
        return None;
    }
    let pedestal = pedestal.clamp(0.0, 0.5 * level);
    let plane = Plane {
        width: frame.width,
        height: frame.height,
        data: frame.data.iter().map(|value| value - pedestal).collect(),
    };
    let (residual, _) = own_residual(&plane)?;
    let usable: Vec<bool> = residual.iter().map(|value| value.is_finite()).collect();
    let wider = fill_smooth(&residual, &usable, frame.width, frame.height, 2.0);
    let high: Vec<f32> = residual
        .iter()
        .zip(&wider)
        .step_by(((frame.width * frame.height) / (1 << 20)).max(1))
        .filter(|(value, _)| value.is_finite())
        .map(|(value, smooth)| value - smooth)
        .collect();
    Some((residual, (1.1 * metric::robust_sigma(&high)).max(1e-7)))
}

/// Whether one flat's residual shows a feature found on the stack: a disk's
/// or donut's rim at 40% of its depth or more, darker inside most of the way
/// round, clear of the flat's noise; a compact dip's box a quarter of its
/// depth below the pixels around it, clear of the noise.
fn shows_in(
    residual: &[f32],
    noise: f32,
    width: usize,
    height: usize,
    bin: usize,
    feature: &DustFeature,
) -> bool {
    let scale = bin as f32;
    if feature.shape.is_round() {
        let (contrast, share, samples) = rim_sharpness(
            residual,
            width,
            height,
            (feature.x + 0.5) / scale - 0.5,
            (feature.y + 0.5) / scale - 0.5,
            feature.radius / scale,
        );
        return share >= 0.6
            && contrast >= 0.4 * feature.depth
            && contrast >= 3.0 * rim_noise(noise, samples);
    }
    let (left, top) = (
        (feature.left / bin as u32) as usize,
        (feature.top / bin as u32) as usize,
    );
    let (right, bottom) = (
        ((feature.right / bin as u32) as usize).min(width.saturating_sub(1)),
        ((feature.bottom / bin as u32) as usize).min(height.saturating_sub(1)),
    );
    // A feature beyond this frame, such as one of a larger session's.
    if left > right || top > bottom {
        return false;
    }
    let margin = ((right - left).max(bottom - top) / 2).max(2);
    let (outer_left, outer_top) = (left.saturating_sub(margin), top.saturating_sub(margin));
    let (outer_right, outer_bottom) = (
        (right + margin).min(width.saturating_sub(1)),
        (bottom + margin).min(height.saturating_sub(1)),
    );
    let (mut inside, mut around) = (Vec::new(), Vec::new());
    for y in outer_top..=outer_bottom {
        for x in outer_left..=outer_right {
            let value = residual[y * width + x];
            if !value.is_finite() {
                continue;
            }
            if (left..=right).contains(&x) && (top..=bottom).contains(&y) {
                inside.push(value);
            } else {
                around.push(value);
            }
        }
    }
    if inside.is_empty() || around.len() < 3 {
        return false;
    }
    let mean = inside.iter().sum::<f32>() / inside.len() as f32;
    let level = metric::median_in_place(&mut around);
    let dip = level - mean;
    // The box mean sits shallower than the deepest smoothed pixel the
    // feature was found by.
    dip >= 0.25 * feature.depth && dip >= 3.0 * noise / (inside.len() as f32).sqrt()
}

/// A plane against its own smooth surface, less one, with its row and column
/// pattern taken out: NaN at stars and unusable pixels. With the star mask.
fn own_residual(plane: &Plane) -> Option<(Vec<f32>, Vec<bool>)> {
    let (width, height) = (plane.width, plane.height);
    if width < 32 || height < 32 || plane.data.len() != width * height {
        return None;
    }
    let level = metric::sampled_median(&plane.data);
    if !(level.is_finite() && level > 0.0) {
        return None;
    }
    let stars = metric::star_mask(plane);
    let masked = Plane {
        width,
        height,
        data: plane
            .data
            .iter()
            .zip(&stars)
            .map(|(value, star)| {
                if *star || !value.is_finite() || *value <= 0.05 * level {
                    f32::NAN
                } else {
                    value / level
                }
            })
            .collect(),
    };
    let (surface, _) = metric::smooth_surface(&masked)?;
    let mut residual: Vec<f32> = masked
        .data
        .iter()
        .zip(&surface)
        .map(|(value, smooth)| {
            if value.is_finite() && *smooth > 0.0 {
                value / smooth - 1.0
            } else {
                f32::NAN
            }
        })
        .collect();
    remove_pattern(&mut residual, width, height);
    Some((residual, stars))
}

/// How the dust of two sessions of one light path differs. Both planes are
/// responses on one grid, their pedestals already removed; `bin` is sensor
/// pixels per plane pixel. `None` when the planes differ in size or hold too
/// little to fit.
pub fn find_changes(
    earlier: &Plane,
    later: &Plane,
    bin: usize,
    options: &DustOptions,
) -> Option<DustChanges> {
    if earlier.width != later.width
        || earlier.height != later.height
        || earlier.width < 32
        || earlier.height < 32
    {
        return None;
    }
    let bin = bin.max(1);
    let (width, height) = (earlier.width, earlier.height);
    let earlier_map = find_dust(earlier, bin, options)?;
    let later_map = find_dust(later, bin, options)?;
    let (earlier_own, _) = own_residual(earlier)?;
    let (later_own, _) = own_residual(later)?;
    let stars: Vec<bool> = metric::star_mask(earlier)
        .into_iter()
        .zip(metric::star_mask(later))
        .map(|(a, b)| a || b)
        .collect();
    let raw: Vec<f32> = earlier
        .data
        .iter()
        .zip(&later.data)
        .map(|(a, b)| {
            if a.is_finite() && b.is_finite() && *a > 0.05 && *b > 0.05 {
                b / a
            } else {
                f32::NAN
            }
        })
        .collect();
    let middle = metric::sampled_median(&raw);
    if !(middle.is_finite() && middle > 0.0) {
        return None;
    }
    let masked = Plane {
        width,
        height,
        data: raw
            .iter()
            .zip(&stars)
            .map(|(value, star)| if *star { f32::NAN } else { value / middle })
            .collect(),
    };
    let (surface, _) = metric::smooth_surface(&masked)?;
    let residual: Vec<f32> = masked
        .data
        .iter()
        .zip(&surface)
        .map(|(value, smooth)| {
            if value.is_finite() && *smooth > 0.0 {
                value / smooth - 1.0
            } else {
                f32::NAN
            }
        })
        .collect();
    let darker = detect(&residual, width, height, bin, options);
    let brighter_residual: Vec<f32> = residual.iter().map(|value| -value).collect();
    let brighter = detect(&brighter_residual, width, height, bin, options);

    // Each session's own pixels against a 9-pixel median: a star is a point
    // bump there, dust is not.
    let local = |plane: &Plane| -> Vec<f32> {
        let level = metric::sampled_median(&plane.data);
        let around = metric::median_filter(plane, 9);
        plane
            .data
            .iter()
            .zip(&around.data)
            .map(|(value, around)| {
                if value.is_finite() && around.is_finite() && level > 0.0 {
                    (value - around) / level
                } else {
                    0.0
                }
            })
            .collect()
    };
    let earlier_local = local(earlier);
    let later_local = local(later);

    let matches = match_features(&earlier_map.features, &later_map.features, options);
    // Dust both sessions hold.
    let held: Vec<Held> = matches
        .iter()
        .filter_map(|m| {
            let (a, b) = (
                &earlier_map.features[m.earlier?],
                &later_map.features[m.later?],
            );
            Some(Held {
                x: 0.5 * (a.x + b.x),
                y: 0.5 * (a.y + b.y),
                reach: reach(a).max(reach(b)),
                offset: m.offset.unwrap_or(0.0),
                within_tolerance: m.within_tolerance,
                depth: a.depth.max(b.depth),
                hole: a.hole.or(b.hole),
                depth_change: (a.depth - b.depth).abs(),
            })
        })
        .collect();

    let mut changes: Vec<(DustChange, Vec<usize>)> = Vec::new();
    for (side_darker, found) in [(true, darker.features), (false, brighter.features)] {
        let (bright_local, dark_local) = if side_darker {
            (&earlier_local, &later_local)
        } else {
            (&later_local, &earlier_local)
        };
        for (feature, footprint) in found {
            if !feature.shape.is_large() {
                // A star a tracked sky flat kept: a point bump in the
                // brighter session, at least half as tall as the spot is
                // deep, and taller than any dip the darker one holds.
                let around = metric::grow(
                    &footprint_mask(&footprint, width * height),
                    width,
                    height,
                    1,
                );
                let (mut bump, mut dip) = (f32::NEG_INFINITY, f32::INFINITY);
                for (index, inside) in around.iter().enumerate() {
                    if *inside {
                        bump = bump.max(bright_local[index]);
                        dip = dip.min(dark_local[index]);
                    }
                }
                if bump > 0.5 * feature.depth && bump > -dip {
                    continue;
                }
            }
            let near = held
                .iter()
                .find(|held| held.explains(&feature, options.move_tolerance));
            // A disk or donut in the ratio is new or gone dust only when the
            // darker session shows its rim on its own, darker inside, if more
            // faintly than its map can count: two sessions' mottle can differ
            // by a rim-like edge too.
            if near.is_none() && feature.shape.is_round() {
                let own = if side_darker {
                    &later_own
                } else {
                    &earlier_own
                };
                let scale = bin as f32;
                let (contrast, share, _) = rim_sharpness(
                    own,
                    width,
                    height,
                    (feature.x + 0.5) / scale - 0.5,
                    (feature.y + 0.5) / scale - 0.5,
                    feature.radius / scale,
                );
                if contrast < 0.5 * feature.depth || share < 0.6 {
                    continue;
                }
            }
            let (kind, within_tolerance, offset) = match near {
                Some(held) => (ChangeKind::Moved, held.within_tolerance, Some(held.offset)),
                None if side_darker => (ChangeKind::New, false, None),
                None => (ChangeKind::Gone, false, None),
            };
            changes.push((
                DustChange {
                    kind,
                    darker: side_darker,
                    feature,
                    within_tolerance,
                    offset,
                },
                footprint,
            ));
        }
    }

    // Large dust one session holds and the other lacks, which the ratio's
    // rim test missed (overlapping shadows hide rims): kept when the ratio
    // carries at least half its depth there.
    for m in &matches {
        let (feature, side_darker) = match (m.earlier, m.later) {
            (None, Some(j)) => (later_map.features[j], true),
            (Some(i), None) => (earlier_map.features[i], false),
            _ => continue,
        };
        if !feature.shape.is_large() {
            continue;
        }
        if changes.iter().any(|(change, _)| {
            (change.feature.x - feature.x).hypot(change.feature.y - feature.y)
                < 0.5 * feature.radius.max(change.feature.radius)
        }) {
            continue;
        }
        let signed = if side_darker {
            &residual
        } else {
            &brighter_residual
        };
        let (cx, cy, r) = (
            (feature.x + 0.5) / bin as f32 - 0.5,
            (feature.y + 0.5) / bin as f32 - 0.5,
            feature.radius / bin as f32,
        );
        let Some(measured) = measure_circle(signed, width, height, cx, cy, r) else {
            continue;
        };
        if measured.depth < 0.5 * feature.depth || measured.depth < options.min_depth {
            continue;
        }
        let mut change_feature = feature;
        change_feature.depth = measured.depth;
        let footprint = disc_footprint(width, height, cx, cy, r);
        changes.push((
            DustChange {
                kind: if side_darker {
                    ChangeKind::New
                } else {
                    ChangeKind::Gone
                },
                darker: side_darker,
                feature: change_feature,
                within_tolerance: false,
                offset: None,
            },
            footprint,
        ));
    }

    pair_moved(&mut changes);
    changes.sort_by(|a, b| {
        b.0.feature
            .depth
            .total_cmp(&a.0.feature.depth)
            .then(a.0.feature.x.total_cmp(&b.0.feature.x))
    });
    let mut vignetting: Vec<(VignettingChange, Vec<usize>)> = darker
        .vignetting
        .into_iter()
        .map(|(v, footprint)| {
            (
                VignettingChange {
                    darker: true,
                    vignetting: v,
                },
                footprint,
            )
        })
        .chain(brighter.vignetting.into_iter().map(|(v, footprint)| {
            (
                VignettingChange {
                    darker: false,
                    vignetting: v,
                },
                footprint,
            )
        }))
        .collect();
    vignetting.sort_by(|a, b| b.0.vignetting.depth.total_cmp(&a.0.vignetting.depth));
    let star_fraction = stars.iter().filter(|star| **star).count() as f32 / (width * height) as f32;
    let mut footprints: Vec<Vec<usize>> = Vec::new();
    let changes = changes
        .into_iter()
        .map(|(change, footprint)| {
            footprints.push(footprint);
            change
        })
        .collect();
    let vignetting = vignetting
        .into_iter()
        .map(|(change, footprint)| {
            footprints.push(footprint);
            change
        })
        .collect();
    Some(DustChanges {
        algorithm_version: DUST_ALGORITHM_VERSION,
        changes,
        vignetting,
        sigma: darker.sigma,
        star_fraction,
        earlier: earlier_map,
        later: later_map,
        footprints,
    })
}

/// Dust both sessions of a comparison hold: where it is, how far it reaches,
/// how far it moved and whether that is within tolerance, how deep it is,
/// a donut's hole as a share of its radius, and how much its depth changed
/// between the two sessions' maps.
struct Held {
    x: f32,
    y: f32,
    reach: f32,
    offset: f32,
    within_tolerance: bool,
    depth: f32,
    hole: Option<f32>,
    depth_change: f32,
}

/// A crescent a moved donut leaves can be this many times as deep as the
/// donut's depth, which is a median over its ring: a donut is darkest at its
/// hole's edge. A disk's crescents are no deeper than
/// [`INSIDE_SHADOW_DEPTH`] times the disk.
const DONUT_CRESCENT_DEPTH: f32 = 4.0;

impl Held {
    /// Whether a feature of the ratio is this dust's doing rather than dust
    /// of its own. Where held dust moved it leaves a crescent on its rim (or
    /// a donut's hole's rim), within the move (plus `slack`) of it and about
    /// as deep as the dust; where it changed depth, the same shadow on nearly
    /// the same centre, or pieces of it inside no deeper than that change.
    /// Anything else, such as window dust in front of a filter's disk or on
    /// its rim, is new or gone.
    fn explains(&self, feature: &DustFeature, slack: f32) -> bool {
        let distance = (feature.x - self.x).hypot(feature.y - self.y);
        let extent = reach(feature);
        if feature.shape.is_large() {
            let size = extent.max(self.reach) / extent.min(self.reach).max(1e-3);
            return size <= 1.5 && distance <= 0.5 * extent.max(self.reach) + self.offset;
        }
        let band = self.offset + slack + extent;
        let on_rim = |rim: f32| (distance - rim).abs() <= band;
        let deepest = if self.hole.is_some() {
            DONUT_CRESCENT_DEPTH
        } else {
            INSIDE_SHADOW_DEPTH
        };
        let crescent = (on_rim(self.reach)
            || self.hole.is_some_and(|hole| on_rim(hole * self.reach)))
            && feature.depth <= deepest * self.depth;
        let inside = distance <= self.reach
            && feature.depth <= self.depth.min(INSIDE_SHADOW_DEPTH * self.depth_change);
        crescent || inside
    }
}

/// How far a feature reaches from its centre: its radius, or half its box's
/// longer side if that is more.
fn reach(feature: &DustFeature) -> f32 {
    let wide = feature.right.saturating_sub(feature.left);
    let tall = feature.bottom.saturating_sub(feature.top);
    feature.radius.max(0.5 * wide.max(tall) as f32)
}

/// A new and a gone shadow of like size and depth within three radii are one
/// shadow that moved further than the maps could pair. The disks of one
/// surface share a size and often a depth, so a disk or donut pairs only when
/// each is the other's sole candidate: with two alike nearby there is no
/// telling which went where, and they stay new and gone.
fn pair_moved(found: &mut [(DustChange, Vec<usize>)]) {
    let like = |a: &DustFeature, b: &DustFeature| {
        let distance = (a.x - b.x).hypot(a.y - b.y);
        let ratio = |p: f32, q: f32| p.max(q) / p.min(q).max(1e-9);
        let size = if a.shape.is_large() { 1.3 } else { 1.75 };
        a.shape.is_large() == b.shape.is_large()
            && distance <= 3.0 * a.radius.max(b.radius)
            && ratio(a.depth, b.depth) <= 2.0
            && ratio(a.radius, b.radius) <= size
    };
    let mut taken = vec![false; found.len()];
    for i in 0..found.len() {
        if taken[i] || found[i].0.kind != ChangeKind::New {
            continue;
        }
        let candidates = |kind: ChangeKind, of: usize, taken: &[bool]| -> Vec<usize> {
            (0..found.len())
                .filter(|j| !taken[*j] && found[*j].0.kind == kind)
                .filter(|j| like(&found[of].0.feature, &found[*j].0.feature))
                .collect()
        };
        let gone = candidates(ChangeKind::Gone, i, &taken);
        let distance = |j: usize| {
            let (p, q) = (&found[i].0.feature, &found[j].0.feature);
            (p.x - q.x).hypot(p.y - q.y)
        };
        let Some(j) = gone
            .iter()
            .copied()
            .min_by(|a, b| distance(*a).total_cmp(&distance(*b)))
        else {
            continue;
        };
        if found[i].0.feature.shape.is_large()
            && (gone.len() > 1 || candidates(ChangeKind::New, j, &taken).len() > 1)
        {
            continue;
        }
        let offset = distance(j);
        for k in [i, j] {
            taken[k] = true;
            found[k].0.kind = ChangeKind::Moved;
            found[k].0.offset = Some(offset);
        }
    }
}

fn footprint_mask(footprint: &[usize], len: usize) -> Vec<bool> {
    let mut mask = vec![false; len];
    for index in footprint {
        if *index < len {
            mask[*index] = true;
        }
    }
    mask
}

fn disc_footprint(width: usize, height: usize, cx: f32, cy: f32, r: f32) -> Vec<usize> {
    let mut out = Vec::new();
    let (y0, y1) = (
        (cy - r).floor().max(0.0) as usize,
        ((cy + r).ceil().max(0.0) as usize).min(height.saturating_sub(1)),
    );
    let (x0, x1) = (
        (cx - r).floor().max(0.0) as usize,
        ((cx + r).ceil().max(0.0) as usize).min(width.saturating_sub(1)),
    );
    for y in y0..=y1 {
        for x in x0..=x1 {
            if (x as f32 - cx).hypot(y as f32 - cy) <= r {
                out.push(y * width + x);
            }
        }
    }
    out
}

/// Features with the plane pixels each covers, vignetting likewise, and the
/// residual's noise.
struct Detected {
    features: Vec<(DustFeature, Vec<usize>)>,
    vignetting: Vec<(Vignetting, Vec<usize>)>,
    sigma: f32,
}

/// Find dust in a residual whose negative values are missing light; NaN
/// where there is no usable data.
fn detect(
    residual: &[f32],
    width: usize,
    height: usize,
    bin: usize,
    options: &DustOptions,
) -> Detected {
    let valid: Vec<bool> = residual.iter().map(|value| value.is_finite()).collect();
    let border_x = ((width as f32 * BORDER_FRACTION) as usize).max(2);
    let border_y = ((height as f32 * BORDER_FRACTION) as usize).max(2);
    let in_band = |index: usize| {
        let (x, y) = (index % width, index / width);
        x < border_x || x >= width - border_x || y < border_y || y >= height - border_y
    };

    // Vignetting first: what it covers is no place to look for dust.
    let vignetting = find_vignetting(residual, &valid, width, height, bin, options);
    let mut usable = valid.clone();
    // Compact dips keep further away: the band beyond a shadow's rim, where
    // the surface cannot follow the fall, breaks into small dips.
    let mut compact_usable = valid.clone();
    if !vignetting.is_empty() {
        let mut shadow = vec![false; width * height];
        for (_, footprint) in &vignetting {
            for index in footprint {
                shadow[*index] = true;
            }
        }
        let long = width.max(height);
        let near = metric::grow(&shadow, width, height, (long / 50).max(2));
        let band = metric::grow(&shadow, width, height, (long / 20).max(2));
        for index in 0..width * height {
            usable[index] &= !near[index];
            compact_usable[index] &= !band[index];
        }
    }

    let filled = fill_smooth(residual, &usable, width, height, 1.0);
    let interior: Vec<f32> = (0..width * height)
        .filter(|index| usable[*index] && !in_band(*index))
        .map(|index| filled[index])
        .collect();
    let sigma = metric::robust_sigma(&interior).max(1e-6);
    // The pixel noise, read against a 2-pixel smoothing so structure does
    // not count, and what is left of it after the 1-pixel smoothing.
    let wider = fill_smooth(residual, &usable, width, height, 2.0);
    let high: Vec<f32> = (0..width * height)
        .filter(|index| usable[*index] && !in_band(*index))
        .step_by(((width * height) / (1 << 20)).max(1))
        .map(|index| residual[index] - wider[index])
        .collect();
    let pixel_noise = (1.1 * metric::robust_sigma(&high)).max(1e-7);

    let mut circles = rim_circles(residual, &usable, width, height, options);
    circles.truncate(MAX_CANDIDATES);
    // How much rim the plane's own mottle shows, for each size tried.
    let mut sizes: Vec<u32> = circles
        .iter()
        .map(|circle| (circle.radius * 4.0).round() as u32)
        .collect();
    sizes.sort_unstable();
    sizes.dedup();
    let null_rims_of = |plane: &[f32], sizes: &[u32]| -> std::collections::HashMap<u32, f32> {
        sizes
            .par_iter()
            .map(|size| {
                (
                    *size,
                    null_rim_spread(plane, width, height, *size as f32 / 4.0),
                )
            })
            .collect()
    };
    let null_rims = null_rims_of(residual, &sizes);
    // Each candidate measured on its own, in parallel; the best are then
    // kept one by one. A shadow's rim is sharp at any size; the edge of a
    // patch of mottle is soft. The contrast a few pixels either side of the
    // rim must stand clear of the white noise and of what circles of this
    // size drawn anywhere on the plane read; that is judged once all else
    // has passed.
    let measured: Vec<Option<Candidate>> = circles
        .par_iter()
        .map(|circle| {
            let measured =
                measure_circle(residual, width, height, circle.x, circle.y, circle.radius)?;
            if measured.depth < options.min_depth
                || measured.rim < options.rim_contrast * measured.depth
                || measured.rim_share < 0.75
                || measured.even_share < 0.75
            {
                return None;
            }
            let radius = refine_radius(residual, width, height, circle.x, circle.y, circle.radius);
            let (sharp, sharp_share, samples) =
                rim_sharpness(residual, width, height, circle.x, circle.y, radius);
            let (fill, spill) = shape_fit(
                &filled,
                &usable,
                width,
                height,
                circle.x,
                circle.y,
                radius,
                measured.ring,
                measured.depth,
            );
            let shaped = sharp >= options.rim_contrast * measured.depth
                && sharp_share >= 0.75
                && fill >= 0.6
                && spill <= 0.55;
            shaped.then_some(Candidate {
                radius,
                measured,
                sharp,
                samples,
                size: (circle.radius * 4.0).round() as u32,
            })
        })
        .collect();
    let significant = |candidate: &Candidate, nulls: &std::collections::HashMap<u32, f32>| {
        let null = nulls.get(&candidate.size).copied().unwrap_or(0.0);
        candidate.sharp / rim_noise(pixel_noise, candidate.samples).max(null)
            >= options.rim_significance
    };
    let mut accepted: Vec<bool> = measured
        .iter()
        .map(|candidate| {
            candidate
                .as_ref()
                .is_some_and(|candidate| significant(candidate, &null_rims))
        })
        .collect();
    // Clear shadows of one size lie under many of the circles the null is
    // read on, and inflate it for a faint shadow of that size: read it again
    // with them taken out for the candidates it held back, those that are not
    // the clear shadows over again.
    let clear: Vec<(f32, f32, f32)> = circles
        .iter()
        .zip(&measured)
        .zip(&accepted)
        .filter_map(|((circle, candidate), ok)| {
            candidate
                .as_ref()
                .filter(|_| *ok)
                .map(|candidate| (circle.x, circle.y, candidate.radius))
        })
        .collect();
    let pending: Vec<usize> = (0..circles.len())
        .filter(|k| {
            measured[*k].is_some()
                && !accepted[*k]
                && clear
                    .iter()
                    .all(|(x, y, r)| (circles[*k].x - x).hypot(circles[*k].y - y) > *r)
        })
        .collect();
    if !pending.is_empty() && !clear.is_empty() {
        let mut masked = residual.to_vec();
        for (x, y, r) in &clear {
            for index in disc_footprint(width, height, *x, *y, 1.3 * r) {
                masked[index] = f32::NAN;
            }
        }
        let mut sizes: Vec<u32> = pending
            .iter()
            .filter_map(|k| measured[*k].as_ref().map(|candidate| candidate.size))
            .collect();
        sizes.sort_unstable();
        sizes.dedup();
        let nulls = null_rims_of(&masked, &sizes);
        for k in pending {
            if let Some(candidate) = &measured[k] {
                accepted[k] = significant(candidate, &nulls);
            }
        }
    }
    let mut features: Vec<(DustFeature, Vec<usize>)> = Vec::new();
    let mut kept: Vec<(f32, f32, f32)> = Vec::new();
    let mut kept_depth: Vec<f32> = Vec::new();
    for ((circle, candidate), ok) in circles.iter().zip(measured).zip(accepted) {
        let Some(Candidate {
            radius, measured, ..
        }) = candidate.filter(|_| ok)
        else {
            continue;
        };
        // The same shadow again, or a circle inside one already found.
        if kept.iter().any(|(x, y, r)| {
            let distance = (circle.x - x).hypot(circle.y - y);
            (distance < 0.5 * r.max(circle.radius)
                && circle.radius.max(*r) / circle.radius.min(*r) < 1.6)
                || distance + circle.radius <= 1.15 * r
        }) {
            continue;
        }
        kept.push((circle.x, circle.y, radius));
        kept_depth.push(measured.depth);
        // A donut's hole is lit nearly as well as the background outside:
        // the central obstruction's shadow of the shadow. Off axis it sits
        // off centre, so it is looked for anywhere inside the ring. Optics
        // without an obstruction cast no donut.
        let unobstructed = options
            .optics
            .is_some_and(|optics| optics.is_usable() && optics.obstruction <= 0.0);
        let hole = if unobstructed {
            None
        } else {
            bright_core(
                &filled,
                &usable,
                width,
                height,
                circle.x,
                circle.y,
                radius,
                measured.ring,
                measured.depth,
                pixel_noise,
            )
            .map(|ratio| match options.optics {
                Some(optics) if optics.is_usable() && optics.obstruction > 0.0 => {
                    optics.obstruction
                }
                _ => ratio,
            })
        };
        let donut = hole.is_some();
        let footprint = disc_footprint(width, height, circle.x, circle.y, radius);
        let scale = bin as f32;
        let sensor = |plane: f32| (plane + 0.5) * scale - 0.5;
        let clamp_u32 =
            |value: f32, limit: usize| value.clamp(0.0, (limit * bin - 1) as f32) as u32;
        let r_sensor = radius * scale;
        let (cx, cy) = (sensor(circle.x), sensor(circle.y));
        features.push((
            DustFeature {
                x: cx,
                y: cy,
                left: clamp_u32(cx - r_sensor, width),
                top: clamp_u32(cy - r_sensor, height),
                right: clamp_u32(cx + r_sensor, width),
                bottom: clamp_u32(cy + r_sensor, height),
                pixels: u32::try_from(footprint.len() * bin * bin).unwrap_or(u32::MAX),
                depth: measured.depth,
                shape: if donut {
                    DustShape::Donut
                } else {
                    DustShape::Disk
                },
                radius: r_sensor,
                hole,
                distance_mm: options
                    .optics
                    .filter(|optics| optics.is_usable())
                    .map(|optics| optics.distance_mm(2.0 * r_sensor)),
                seen_in: None,
            },
            footprint,
        ));
    }
    // A circle found before a larger one that holds it is a piece of that
    // one's ring or interior.
    let circles_found: Vec<(f32, f32, f32)> =
        features.iter().map(|(f, _)| (f.x, f.y, f.radius)).collect();
    features.retain(|(f, _)| !inside_larger(&circles_found, f.x, f.y, f.radius));
    merge_clusters(&mut features);

    // Compact dips: what a wide local median leaves once each disk and
    // donut found is taken out by its own radial profile, so window dust in
    // front of one is still seen.
    let mut local = residual
        .iter()
        .zip(&usable)
        .map(|(value, ok)| if *ok { *value } else { 0.0 })
        .collect::<Vec<f32>>();
    for (x, y, r) in &kept {
        subtract_profile(&mut local, &usable, width, height, *x, *y, *r);
    }
    metric::subtract_local_median(&mut local, width, height);
    let smoothed = fill_smooth(&local, &usable, width, height, 1.0);
    let local_interior: Vec<f32> = (0..width * height)
        .filter(|index| usable[*index] && !in_band(*index))
        .map(|index| smoothed[index])
        .collect();
    let local_sigma = metric::robust_sigma(&local_interior).max(1e-6);
    let threshold = options.min_depth.max(options.compact_sigma * local_sigma);
    let small_threshold = threshold.max(lone_dip_floor(
        width * height,
        local_sigma,
        options.false_dips,
    ));
    let min_area = (options.min_compact_area / (bin * bin) as f32).max(1.0);
    // Dips this small are what noise leaves: up to four plane pixels, or
    // twice the smallest counted, whichever is more.
    let small_area = (2.0 * min_area).max(4.0);
    let mask: Vec<u8> = (0..width * height)
        .map(|index| u8::from(compact_usable[index] && smoothed[index] < -threshold))
        .collect();
    let mut compact: Vec<(DustFeature, Vec<usize>)> = Vec::new();
    for component in connected_components(&mask, width, height, Connectivity::Eight) {
        let area = component.pixels.len() as f32;
        if area < min_area {
            continue;
        }
        if component.pixels.iter().any(|index| in_band(*index)) {
            continue;
        }
        let deepest = component
            .pixels
            .iter()
            .map(|index| -smoothed[*index])
            .fold(0.0_f32, f32::max);
        if area <= small_area && deepest < small_threshold {
            continue;
        }
        // Inside a disk or donut already found, what its profile leaves (a
        // blurred rim, an off-centre hole's other side, mottle) is no deeper
        // than the shadow itself; window dust in front of it is.
        if kept.iter().zip(&kept_depth).any(|((x, y, r), depth)| {
            (component.centroid_x - x).hypot(component.centroid_y - y) <= 1.15 * r
                && deepest < INSIDE_SHADOW_DEPTH * depth
        }) {
            continue;
        }
        let shape = if component.fill_fraction() < 0.5 || component.elongation > 2.5 {
            DustShape::Irregular
        } else {
            DustShape::Compact
        };
        let scale = bin as f32;
        let first = |plane: usize| u32::try_from(plane * bin).unwrap_or(u32::MAX);
        let last = |plane: usize| u32::try_from(plane * bin + bin - 1).unwrap_or(u32::MAX);
        compact.push((
            DustFeature::compact(
                (component.centroid_x + 0.5) * scale - 0.5,
                (component.centroid_y + 0.5) * scale - 0.5,
                (
                    first(component.min_x),
                    first(component.min_y),
                    last(component.max_x),
                    last(component.max_y),
                ),
                u32::try_from(component.pixels.len() * bin * bin).unwrap_or(u32::MAX),
                deepest,
                shape,
            ),
            component.pixels,
        ));
    }
    features.sort_by(|a, b| b.0.depth.total_cmp(&a.0.depth));
    compact.sort_by(|a, b| b.0.depth.total_cmp(&a.0.depth));
    features.extend(compact);
    features.truncate(MAX_FEATURES);
    Detected {
        features,
        vignetting,
        sigma,
    }
}

/// The plane's row and column pattern, read from the residual against a
/// box mean and taken out: a sensor's columns are not dust.
fn remove_pattern(residual: &mut [f32], width: usize, height: usize) {
    for axis in [0, 1] {
        let high = high_pass(residual, width, height);
        let lines = if axis == 0 { width } else { height };
        let pattern: Vec<f32> = (0..lines)
            .into_par_iter()
            .map(|line| {
                let mut values: Vec<f32> = if axis == 0 {
                    (0..height).map(|y| high[y * width + line]).collect()
                } else {
                    high[line * width..(line + 1) * width].to_vec()
                };
                values.retain(|value| value.is_finite());
                if values.len() < 8 {
                    0.0
                } else {
                    metric::median_in_place(&mut values)
                }
            })
            .collect();
        for (index, value) in residual.iter_mut().enumerate() {
            let line = if axis == 0 {
                index % width
            } else {
                index / width
            };
            if value.is_finite() {
                *value -= pattern[line];
            }
        }
    }
}

/// The residual less its box mean over [`PATTERN_WINDOW`], NaNs left out.
fn high_pass(residual: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mean = box_mean(residual, width, height, PATTERN_WINDOW / 2);
    residual
        .iter()
        .zip(&mean)
        .map(|(value, mean)| {
            if value.is_finite() && mean.is_finite() {
                value - mean
            } else {
                f32::NAN
            }
        })
        .collect()
}

/// Mean over a `(2 half + 1)²` box of the finite values, by summed-area
/// tables.
fn box_mean(values: &[f32], width: usize, height: usize, half: usize) -> Vec<f32> {
    let stride = width + 1;
    let mut sum = vec![0f64; stride * (height + 1)];
    let mut count = vec![0u32; stride * (height + 1)];
    for y in 0..height {
        for x in 0..width {
            let value = values[y * width + x];
            let (s, c) = if value.is_finite() {
                (f64::from(value), 1)
            } else {
                (0.0, 0)
            };
            let at = (y + 1) * stride + x + 1;
            sum[at] = s + sum[at - 1] + sum[at - stride] - sum[at - stride - 1];
            count[at] = c + count[at - 1] + count[at - stride] - count[at - stride - 1];
        }
    }
    (0..width * height)
        .into_par_iter()
        .map(|index| {
            let (x, y) = (index % width, index / width);
            let (x0, y0) = (x.saturating_sub(half), y.saturating_sub(half));
            let (x1, y1) = ((x + half + 1).min(width), (y + half + 1).min(height));
            let pick = |table: &[f64]| {
                table[y1 * stride + x1] - table[y0 * stride + x1] - table[y1 * stride + x0]
                    + table[y0 * stride + x0]
            };
            let n = count[y1 * stride + x1] + count[y0 * stride + x0]
                - count[y0 * stride + x1]
                - count[y1 * stride + x0];
            if n == 0 {
                f32::NAN
            } else {
                (pick(&sum) / f64::from(n)) as f32
            }
        })
        .collect()
}

/// A Gaussian of the usable values, normalised by the usable weight, so a
/// hole does not pull its surroundings to zero.
fn fill_smooth(
    values: &[f32],
    usable: &[bool],
    width: usize,
    height: usize,
    sigma: f64,
) -> Vec<f32> {
    let numerator: Vec<f32> = values
        .iter()
        .zip(usable)
        .map(|(value, ok)| {
            if *ok && value.is_finite() {
                *value
            } else {
                0.0
            }
        })
        .collect();
    let weight: Vec<f32> = usable.iter().map(|ok| f32::from(u8::from(*ok))).collect();
    let numerator = metric::gaussian(&numerator, width, height, sigma);
    let weight = metric::gaussian(&weight, width, height, sigma);
    numerator
        .iter()
        .zip(&weight)
        .map(|(n, w)| if *w > 1e-3 { n / w } else { 0.0 })
        .collect()
}

/// A circle the rim vote found, in plane pixels.
struct Circle {
    x: f32,
    y: f32,
    radius: f32,
}

/// One rim pixel: where, which way the light rises, and how steeply.
struct RimPixel {
    x: f32,
    y: f32,
    ux: f32,
    uy: f32,
    weight: f32,
    sector: u16,
}

fn rim_pixels(
    residual: &[f32],
    usable: &[bool],
    inner: &[bool],
    width: usize,
    height: usize,
    sigma: f64,
    edge_sigma: f32,
) -> Vec<RimPixel> {
    let smooth = fill_smooth(residual, usable, width, height, sigma);
    let mut gx = vec![0f32; width * height];
    let mut gy = vec![0f32; width * height];
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let index = y * width + x;
            gx[index] = 0.5 * (smooth[index + 1] - smooth[index - 1]);
            gy[index] = 0.5 * (smooth[index + width] - smooth[index - width]);
        }
    }
    let sample: Vec<f32> = (0..width * height)
        .filter(|index| inner[*index])
        .step_by(((width * height) / (1 << 20)).max(1))
        .map(|index| gx[index])
        .collect();
    let noise = metric::robust_sigma(&sample).max(1e-9);
    let threshold = edge_sigma * noise;
    (0..width * height)
        .filter_map(|index| {
            if !inner[index] {
                return None;
            }
            let (dx, dy) = (gx[index], gy[index]);
            let g = dx.hypot(dy);
            if g <= threshold {
                return None;
            }
            let angle = dy.atan2(dx);
            let sector = (((angle + std::f32::consts::PI) / std::f32::consts::TAU * SECTORS as f32)
                as usize)
                .min(SECTORS - 1);
            Some(RimPixel {
                x: (index % width) as f32,
                y: (index / width) as f32,
                ux: dx / g,
                uy: dy / g,
                weight: g / noise,
                sector: sector as u16,
            })
        })
        .collect()
}

/// Circles whose rim is darker inside in most sectors, best first.
fn rim_circles(
    residual: &[f32],
    usable: &[bool],
    width: usize,
    height: usize,
    options: &DustOptions,
) -> Vec<Circle> {
    // Rim pixels must sit clear of stars and the border, whose edges are
    // not dust's.
    let unusable: Vec<bool> = usable.iter().map(|ok| !ok).collect();
    let near_unusable = metric::grow(&unusable, width, height, 3);
    let inner: Vec<bool> = (0..width * height)
        .map(|index| {
            let (x, y) = (index % width, index / width);
            !near_unusable[index] && x >= 2 && y >= 2 && x + 2 < width && y + 2 < height
        })
        .collect();
    let fine = rim_pixels(
        residual,
        usable,
        &inner,
        width,
        height,
        FINE_SIGMA,
        options.edge_sigma,
    );
    let coarse = rim_pixels(
        residual,
        usable,
        &inner,
        width,
        height,
        COARSE_SIGMA,
        options.edge_sigma,
    );
    let largest = (width.min(height) as f32 / 4.0).max(MIN_RIM_RADIUS);
    let mut radii = Vec::new();
    let mut radius = MIN_RIM_RADIUS;
    while radius <= largest {
        radii.push(radius);
        radius *= RADIUS_STEP;
    }
    let share = options.rim_sectors.clamp(1, SECTORS) as f32 / SECTORS as f32;
    let mut candidates: Vec<(Circle, f32)> = radii
        .par_iter()
        .flat_map_iter(|&radius| {
            let pixels = if radius < COARSE_FROM_RADIUS {
                &fine
            } else {
                &coarse
            };
            vote(pixels, &inner, width, height, radius, share)
        })
        .collect();
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
    candidates.into_iter().map(|(circle, _)| circle).collect()
}

/// The centres one radius votes for, with a score: covered share of the
/// visible rim, then rim strength.
fn vote(
    pixels: &[RimPixel],
    inner: &[bool],
    width: usize,
    height: usize,
    radius: f32,
    share: f32,
) -> Vec<(Circle, f32)> {
    let cell = (0.06 * radius).round().max(1.0);
    let offset = radius.ceil();
    let grid_w = ((width as f32 + 2.0 * offset) / cell) as usize + 1;
    let grid_h = ((height as f32 + 2.0 * offset) / cell) as usize + 1;
    let mut bits = vec![0u16; grid_w * grid_h];
    let mut strength = vec![0f32; grid_w * grid_h];
    for pixel in pixels {
        let cx = ((pixel.x - radius * pixel.ux + offset) / cell).round();
        let cy = ((pixel.y - radius * pixel.uy + offset) / cell).round();
        if cx < 0.0 || cy < 0.0 {
            continue;
        }
        let (cx, cy) = (cx as usize, cy as usize);
        if cx >= grid_w || cy >= grid_h {
            continue;
        }
        strength[cy * grid_w + cx] += pixel.weight;
        for y in cy.saturating_sub(1)..=(cy + 1).min(grid_h - 1) {
            for x in cx.saturating_sub(1)..=(cx + 1).min(grid_w - 1) {
                bits[y * grid_w + x] |= 1 << pixel.sector;
            }
        }
    }
    let rim_length = std::f32::consts::TAU * radius;
    let mut out = Vec::new();
    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let cover = bits[gy * grid_w + gx].count_ones() as usize;
            if cover < 6 {
                continue;
            }
            let score = |x: usize, y: usize| {
                bits[y * grid_w + x].count_ones() as f32 + strength[y * grid_w + x] / 1e6
            };
            let here = score(gx, gy);
            let mut best = true;
            'look: for y in gy.saturating_sub(1)..=(gy + 1).min(grid_h - 1) {
                for x in gx.saturating_sub(1)..=(gx + 1).min(grid_w - 1) {
                    if (x, y) != (gx, gy) && score(x, y) > here {
                        best = false;
                        break 'look;
                    }
                }
            }
            if !best {
                continue;
            }
            let centre_x = (gx as f32 + 0.5) * cell - 0.5 - offset;
            let centre_y = (gy as f32 + 0.5) * cell - 0.5 - offset;
            // Sectors whose rim lies on usable pixels.
            let mut visible = 0usize;
            let mut seen = 0usize;
            let set = bits[gy * grid_w + gx];
            for sector in 0..SECTORS {
                let angle = (sector as f32 + 0.5) / SECTORS as f32 * std::f32::consts::TAU
                    - std::f32::consts::PI;
                let (x, y) = (
                    (centre_x + radius * angle.cos()).round(),
                    (centre_y + radius * angle.sin()).round(),
                );
                if x < 0.0 || y < 0.0 || x >= width as f32 || y >= height as f32 {
                    continue;
                }
                if inner[y as usize * width + x as usize] {
                    visible += 1;
                    if set & (1 << sector) != 0 {
                        seen += 1;
                    }
                }
            }
            let needed = ((share * visible as f32).ceil() as usize).max(6);
            if visible * 2 < SECTORS {
                continue;
            }
            if seen < needed {
                continue;
            }
            let rim = strength[gy * grid_w + gx] / rim_length;
            out.push((
                Circle {
                    x: centre_x,
                    y: centre_y,
                    radius,
                },
                seen as f32 / visible as f32 + rim.min(100.0) / 1e4,
            ));
        }
    }
    out
}

/// A rim candidate that passed every test but its rim's significance.
struct Candidate {
    /// Refined radius, plane pixels.
    radius: f32,
    measured: CircleDepth,
    /// Rim contrast on thin rings, and the samples each side of a sector.
    sharp: f32,
    samples: f32,
    /// The tried radius in quarter pixels, which keys the null rim spread.
    size: u32,
}

/// What a candidate circle holds, ring inside against ring outside.
struct CircleDepth {
    /// Median of the sector depths: the broad rings, 0.6–0.88 against
    /// 1.12–1.4 radii.
    depth: f32,
    /// Median of the sector rim contrasts: the thin rings either side of
    /// the rim, 0.75–0.95 against 1.05–1.25 radii. A real rim carries most
    /// of the depth; a circle drawn around several shadows carries little.
    rim: f32,
    /// Share of measured sectors whose rim is darker inside.
    rim_share: f32,
    /// Share of sectors whose depth lies within 0.4 to 2.5 times the median.
    even_share: f32,
    /// Median residual of the inner ring.
    ring: f32,
}

fn measure_circle(
    residual: &[f32],
    width: usize,
    height: usize,
    x: f32,
    y: f32,
    radius: f32,
) -> Option<CircleDepth> {
    let reach = 1.4 * radius;
    let (x0, x1) = (
        (x - reach).floor().max(0.0) as usize,
        ((x + reach).ceil().max(0.0) as usize).min(width.saturating_sub(1)),
    );
    let (y0, y1) = (
        (y - reach).floor().max(0.0) as usize,
        ((y + reach).ceil().max(0.0) as usize).min(height.saturating_sub(1)),
    );
    let ring = || -> Vec<Vec<f32>> { vec![Vec::new(); DEPTH_SECTORS] };
    let (mut inside, mut outside, mut rim_in, mut rim_out) = (ring(), ring(), ring(), ring());
    for py in y0..=y1 {
        for px in x0..=x1 {
            let value = residual[py * width + px];
            if !value.is_finite() {
                continue;
            }
            let (dx, dy) = (px as f32 - x, py as f32 - y);
            let d = dx.hypot(dy) / radius.max(1e-3);
            let sector = (((dy.atan2(dx) + std::f32::consts::PI) / std::f32::consts::TAU
                * DEPTH_SECTORS as f32) as usize)
                .min(DEPTH_SECTORS - 1);
            if (0.6..=0.88).contains(&d) {
                inside[sector].push(value);
            }
            if (1.12..=1.4).contains(&d) {
                outside[sector].push(value);
            }
            if (0.75..=0.95).contains(&d) {
                rim_in[sector].push(value);
            }
            if (1.05..=1.25).contains(&d) {
                rim_out[sector].push(value);
            }
        }
    }
    let contrasts = |inner: Vec<Vec<f32>>, outer: Vec<Vec<f32>>| -> (Vec<f32>, Vec<f32>) {
        let mut contrast = Vec::new();
        let mut levels = Vec::new();
        for (mut a, mut b) in inner.into_iter().zip(outer) {
            if a.len() < MIN_RING_SAMPLES || b.len() < MIN_RING_SAMPLES {
                continue;
            }
            let level = metric::median_in_place(&mut a);
            contrast.push(metric::median_in_place(&mut b) - level);
            levels.push(level);
        }
        (contrast, levels)
    };
    let (mut depths, mut rings) = contrasts(inside, outside);
    let (mut rims, _) = contrasts(rim_in, rim_out);
    if depths.len() < DEPTH_SECTORS / 2 || rims.len() < DEPTH_SECTORS / 2 {
        return None;
    }
    let rim_share = rims.iter().filter(|d| **d > 0.0).count() as f32 / rims.len() as f32;
    let all_depths = depths.clone();
    let depth = metric::median_in_place(&mut depths);
    // A shadow of the aperture is evenly deep all the way round; a circle
    // laid over an irregular shadow is deep on one side and shallow on the
    // other.
    let even_share = all_depths
        .iter()
        .filter(|d| **d >= 0.4 * depth && **d <= 2.5 * depth)
        .count() as f32
        / all_depths.len() as f32;
    let rim = metric::median_in_place(&mut rims);
    let ring = metric::median_in_place(&mut rings);
    Some(CircleDepth {
        depth,
        even_share,
        rim,
        rim_share,
        ring,
    })
}

/// A donut's hole: a bright patch inside the ring (within 0.6 radii of the
/// centre, not necessarily on it), at least three quarters of the way from
/// the ring's level back to the background outside, whose mean clears the
/// noise of that many pixels (`pixel_noise` is the residual's, unsmoothed)
/// and whose size is a plausible obstruction. Returns its radius as a
/// fraction of the ring's.
#[allow(clippy::too_many_arguments)]
fn bright_core(
    smoothed: &[f32],
    usable: &[bool],
    width: usize,
    height: usize,
    x: f32,
    y: f32,
    radius: f32,
    ring: f32,
    depth: f32,
    pixel_noise: f32,
) -> Option<f32> {
    let reach = 0.6 * radius;
    let (x0, x1) = (
        (x - reach).floor().max(0.0) as usize,
        ((x + reach).ceil().max(0.0) as usize).min(width.saturating_sub(1)),
    );
    let (y0, y1) = (
        (y - reach).floor().max(0.0) as usize,
        ((y + reach).ceil().max(0.0) as usize).min(height.saturating_sub(1)),
    );
    let inside = |px: usize, py: usize| (px as f32 - x).hypot(py as f32 - y) <= reach;
    let mut peak = (f32::NEG_INFINITY, 0usize);
    for py in y0..=y1 {
        for px in x0..=x1 {
            let index = py * width + px;
            if usable[index] && inside(px, py) && smoothed[index] > peak.0 {
                peak = (smoothed[index], index);
            }
        }
    }
    let lift = peak.0 - ring;
    if !(lift >= 0.75 * depth && lift > 4.0 * SMOOTHED_NOISE * pixel_noise) {
        return None;
    }
    // The patch around the peak above halfway between ring and peak.
    let level = ring + 0.5 * lift;
    let mut seen = vec![false; (x1 - x0 + 1) * (y1 - y0 + 1)];
    let local = |index: usize| (index / width - y0) * (x1 - x0 + 1) + (index % width - x0);
    let mut stack = vec![peak.1];
    seen[local(peak.1)] = true;
    let (mut area, mut sum) = (0usize, 0.0_f32);
    while let Some(index) = stack.pop() {
        area += 1;
        sum += smoothed[index];
        let (px, py) = (index % width, index / width);
        for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
            let (nx, ny) = (px as i64 + dx, py as i64 + dy);
            if nx < x0 as i64 || ny < y0 as i64 || nx > x1 as i64 || ny > y1 as i64 {
                continue;
            }
            let (nx, ny) = (nx as usize, ny as usize);
            let next = ny * width + nx;
            if seen[local(next)] || !inside(nx, ny) || !usable[next] || smoothed[next] < level {
                continue;
            }
            seen[local(next)] = true;
            stack.push(next);
        }
    }
    let ratio = (area as f32 / std::f32::consts::PI).sqrt() / radius;
    // Near dust casts a small donut whose hole the rim's blur narrows; such
    // a hole must be lit fully to count.
    // A bright patch of a pixel or two inside a large disk is what is left
    // of a star.
    let core_radius = ratio * radius;
    let plausible = core_radius >= 2.0
        && ((0.12..=0.6).contains(&ratio)
            || ((0.06..0.12).contains(&ratio) && radius < 30.0 && lift >= depth));
    // The patch as a whole must stand clear of the noise, not only its
    // brightest pixel: the peak of hundreds of pixels is the noise's tail.
    let clear = sum / area as f32 - ring > 4.0 * pixel_noise / (area as f32).sqrt();
    (plausible && clear).then_some(ratio)
}

/// The rim's contrast on rings a few pixels wide either side of it, by
/// sector: the median, and the share of sectors darker inside.
fn rim_sharpness(
    residual: &[f32],
    width: usize,
    height: usize,
    x: f32,
    y: f32,
    radius: f32,
) -> (f32, f32, f32) {
    let band = (0.25 * radius).clamp(1.5, RIM_BAND);
    let reach = radius + band + 1.0;
    let (x0, x1) = (
        (x - reach).floor().max(0.0) as usize,
        ((x + reach).ceil().max(0.0) as usize).min(width.saturating_sub(1)),
    );
    let (y0, y1) = (
        (y - reach).floor().max(0.0) as usize,
        ((y + reach).ceil().max(0.0) as usize).min(height.saturating_sub(1)),
    );
    let mut inner: Vec<Vec<f32>> = vec![Vec::new(); DEPTH_SECTORS];
    let mut outer: Vec<Vec<f32>> = vec![Vec::new(); DEPTH_SECTORS];
    for py in y0..=y1 {
        for px in x0..=x1 {
            let value = residual[py * width + px];
            if !value.is_finite() {
                continue;
            }
            let (dx, dy) = (px as f32 - x, py as f32 - y);
            let d = dx.hypot(dy) - radius;
            let sector = (((dy.atan2(dx) + std::f32::consts::PI) / std::f32::consts::TAU
                * DEPTH_SECTORS as f32) as usize)
                .min(DEPTH_SECTORS - 1);
            if (-band..=-0.5).contains(&d) {
                inner[sector].push(value);
            } else if (0.5..=band).contains(&d) {
                outer[sector].push(value);
            }
        }
    }
    let mut contrasts = Vec::new();
    let mut samples = Vec::new();
    for (mut a, mut b) in inner.into_iter().zip(outer) {
        if a.len() < MIN_RING_SAMPLES || b.len() < MIN_RING_SAMPLES {
            continue;
        }
        samples.push(a.len().min(b.len()) as f32);
        contrasts.push(metric::median_in_place(&mut b) - metric::median_in_place(&mut a));
    }
    if contrasts.len() < DEPTH_SECTORS / 2 {
        return (0.0, 0.0, 0.0);
    }
    let share = contrasts.iter().filter(|c| **c > 0.0).count() as f32 / contrasts.len() as f32;
    let typical = metric::median_in_place(&mut samples);
    (metric::median_in_place(&mut contrasts), share, typical)
}

/// Three or more circles that all overlap closely are one irregular shadow
/// the rim vote tiled with circles, not three shadows of the aperture: they
/// become one [`DustShape::Patch`] covering them all.
fn merge_clusters(features: &mut Vec<(DustFeature, Vec<usize>)>) {
    let count = features.len();
    let close = |a: &DustFeature, b: &DustFeature| {
        (a.x - b.x).hypot(a.y - b.y) < 0.9 * a.radius.max(b.radius)
    };
    let mut group: Vec<usize> = (0..count).collect();
    fn root(group: &mut [usize], mut i: usize) -> usize {
        while group[i] != i {
            group[i] = group[group[i]];
            i = group[i];
        }
        i
    }
    for i in 0..count {
        for j in i + 1..count {
            if features[i].0.shape.is_large()
                && features[j].0.shape.is_large()
                && close(&features[i].0, &features[j].0)
            {
                let (a, b) = (root(&mut group, i), root(&mut group, j));
                group[a.max(b)] = a.min(b);
            }
        }
    }
    let mut members: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for i in 0..count {
        let r = root(&mut group, i);
        members.entry(r).or_default().push(i);
    }
    let mut drop = vec![false; count];
    let mut merged = Vec::new();
    for list in members.values() {
        if list.len() < 3 {
            continue;
        }
        let mut footprint: Vec<usize> = list
            .iter()
            .flat_map(|i| features[*i].1.iter().copied())
            .collect();
        footprint.sort_unstable();
        footprint.dedup();
        let first = features[list[0]].0;
        let mut feature = first;
        for i in list {
            let f = features[*i].0;
            feature.left = feature.left.min(f.left);
            feature.top = feature.top.min(f.top);
            feature.right = feature.right.max(f.right);
            feature.bottom = feature.bottom.max(f.bottom);
            feature.depth = feature.depth.max(f.depth);
            drop[*i] = true;
        }
        let total: f32 = list.iter().map(|i| features[*i].0.radius.powi(2)).sum();
        feature.x = list
            .iter()
            .map(|i| features[*i].0.x * features[*i].0.radius.powi(2))
            .sum::<f32>()
            / total;
        feature.y = list
            .iter()
            .map(|i| features[*i].0.y * features[*i].0.radius.powi(2))
            .sum::<f32>()
            / total;
        // Sensor pixels per plane pixel, from the first member.
        feature.pixels = (footprint.len() as f32 * first.pixels as f32
            / features[list[0]].1.len().max(1) as f32) as u32;
        feature.radius = (feature.pixels as f32 / std::f32::consts::PI).sqrt();
        feature.shape = DustShape::Patch;
        feature.hole = None;
        feature.distance_mm = None;
        merged.push((feature, footprint));
    }
    if merged.is_empty() {
        return;
    }
    let mut index = 0;
    features.retain(|_| {
        let keep = !drop[index];
        index += 1;
        keep
    });
    features.extend(merged);
}

/// Whether a larger circle of `circles` (sensor pixels) holds this one.
fn inside_larger(circles: &[(f32, f32, f32)], x: f32, y: f32, r: f32) -> bool {
    circles
        .iter()
        .any(|(ox, oy, or)| *or > r && (x - ox).hypot(y - oy) + r <= 1.15 * or)
}

/// How well a shadow fills its circle and stays inside it: the share of
/// the band 0.5–0.85 radii that is dark (below halfway from the ring's
/// level to the background's), and the share of the band 1.15–1.4 radii that
/// is dark as well. A shadow of the aperture fills its circle and stops at
/// the rim; an irregular shadow under a circle does neither.
#[allow(clippy::too_many_arguments)]
fn shape_fit(
    smoothed: &[f32],
    usable: &[bool],
    width: usize,
    height: usize,
    x: f32,
    y: f32,
    radius: f32,
    ring: f32,
    depth: f32,
) -> (f32, f32) {
    let level = ring + 0.5 * depth;
    let reach = 1.4 * radius;
    let (x0, x1) = (
        (x - reach).floor().max(0.0) as usize,
        ((x + reach).ceil().max(0.0) as usize).min(width.saturating_sub(1)),
    );
    let (y0, y1) = (
        (y - reach).floor().max(0.0) as usize,
        ((y + reach).ceil().max(0.0) as usize).min(height.saturating_sub(1)),
    );
    let (mut inside, mut inside_dark, mut outside, mut outside_dark) = (0u32, 0u32, 0u32, 0u32);
    for py in y0..=y1 {
        for px in x0..=x1 {
            let index = py * width + px;
            if !usable[index] {
                continue;
            }
            let d = (px as f32 - x).hypot(py as f32 - y) / radius.max(1e-3);
            let dark = smoothed[index] < level;
            if (0.5..=0.85).contains(&d) {
                inside += 1;
                inside_dark += u32::from(dark);
            } else if (1.15..=1.4).contains(&d) {
                outside += 1;
                outside_dark += u32::from(dark);
            }
        }
    }
    (
        inside_dark as f32 / inside.max(1) as f32,
        outside_dark as f32 / outside.max(1) as f32,
    )
}

/// The robust spread of [`rim_sharpness`] over circles of `radius` centred
/// on a 7 × 7 grid across the plane: how much rim the plane's own mottle
/// shows at that size.
fn null_rim_spread(residual: &[f32], width: usize, height: usize, radius: f32) -> f32 {
    let contrasts: Vec<f32> = (0..49)
        .into_par_iter()
        .filter_map(|k| {
            let (i, j) = (k % 7, k / 7);
            let x = (i as f32 + 0.5) / 7.0 * width as f32;
            let y = (j as f32 + 0.5) / 7.0 * height as f32;
            let (contrast, _, samples) = rim_sharpness(residual, width, height, x, y, radius);
            (samples > 0.0).then_some(contrast)
        })
        .collect();
    if contrasts.len() < 8 {
        return 0.0;
    }
    metric::robust_sigma(&contrasts)
}

/// The noise of [`rim_sharpness`]'s median contrast for a pixel noise of
/// `noise`, with `samples` on each side of each sector: a median's error
/// is 1.25 times a mean's, for each side and again across the sectors.
fn rim_noise(noise: f32, samples: f32) -> f32 {
    1.25 * noise * (2.0 / samples.max(1.0)).sqrt() * 1.25 / (DEPTH_SECTORS as f32).sqrt()
}

/// The residual's median at each whole radius from `x, y` out to `reach`.
fn radial_profile(
    residual: &[f32],
    width: usize,
    height: usize,
    x: f32,
    y: f32,
    reach: f32,
) -> Vec<f32> {
    let bins = reach.ceil() as usize + 1;
    let mut rings: Vec<Vec<f32>> = vec![Vec::new(); bins];
    let (x0, x1) = (
        (x - reach).floor().max(0.0) as usize,
        ((x + reach).ceil().max(0.0) as usize).min(width.saturating_sub(1)),
    );
    let (y0, y1) = (
        (y - reach).floor().max(0.0) as usize,
        ((y + reach).ceil().max(0.0) as usize).min(height.saturating_sub(1)),
    );
    for py in y0..=y1 {
        for px in x0..=x1 {
            let value = residual[py * width + px];
            let d = (px as f32 - x).hypot(py as f32 - y);
            if value.is_finite() && d <= reach {
                rings[d as usize].push(value);
            }
        }
    }
    rings
        .into_iter()
        .map(|mut ring| {
            if ring.len() < 3 {
                f32::NAN
            } else {
                metric::median_in_place(&mut ring)
            }
        })
        .collect()
}

/// Take a disk's or donut's radial profile out of `values` within 1.3 radii
/// of its centre, against the level just outside it: the shadow goes, and
/// what lies in front of it stays.
fn subtract_profile(
    values: &mut [f32],
    usable: &[bool],
    width: usize,
    height: usize,
    x: f32,
    y: f32,
    radius: f32,
) {
    let reach = 1.4 * radius;
    let (x0, x1) = (
        (x - reach).floor().max(0.0) as usize,
        ((x + reach).ceil().max(0.0) as usize).min(width.saturating_sub(1)),
    );
    let (y0, y1) = (
        (y - reach).floor().max(0.0) as usize,
        ((y + reach).ceil().max(0.0) as usize).min(height.saturating_sub(1)),
    );
    let mut rings: Vec<Vec<f32>> = vec![Vec::new(); reach.ceil() as usize + 1];
    for py in y0..=y1 {
        for px in x0..=x1 {
            let index = py * width + px;
            let d = (px as f32 - x).hypot(py as f32 - y);
            if usable[index] && values[index].is_finite() && d <= reach {
                rings[d as usize].push(values[index]);
            }
        }
    }
    let profile: Vec<f32> = rings
        .into_iter()
        .map(|mut ring| {
            if ring.len() < MIN_RING_SAMPLES {
                f32::NAN
            } else {
                metric::median_in_place(&mut ring)
            }
        })
        .collect();
    let mut outside: Vec<f32> = profile
        .iter()
        .enumerate()
        .filter(|(r, v)| v.is_finite() && *r as f32 >= 1.15 * radius)
        .map(|(_, v)| *v)
        .collect();
    if outside.is_empty() {
        return;
    }
    let level = metric::median_in_place(&mut outside);
    let limit = 1.3 * radius;
    for py in y0..=y1 {
        for px in x0..=x1 {
            let index = py * width + px;
            let d = (px as f32 - x).hypot(py as f32 - y);
            if usable[index] && d <= limit {
                let shadow = profile[d as usize];
                if shadow.is_finite() {
                    values[index] -= shadow - level;
                }
            }
        }
    }
}

/// Where between 0.7 and 1.3 radii the profile crosses halfway from the
/// ring inside to the ring outside: the rim, finer than the vote's steps.
fn refine_radius(
    residual: &[f32],
    width: usize,
    height: usize,
    x: f32,
    y: f32,
    radius: f32,
) -> f32 {
    let profile = radial_profile(residual, width, height, x, y, 1.45 * radius);
    let pick = |from: f32, to: f32| {
        let mut values: Vec<f32> = profile
            .iter()
            .enumerate()
            .filter(|(r, v)| v.is_finite() && *r as f32 >= from && *r as f32 <= to)
            .map(|(_, v)| *v)
            .collect();
        if values.is_empty() {
            f32::NAN
        } else {
            metric::median_in_place(&mut values)
        }
    };
    let inside = pick(0.6 * radius, 0.85 * radius);
    let outside = pick(1.15 * radius, 1.4 * radius);
    if !(inside.is_finite() && outside.is_finite()) || outside <= inside {
        return radius;
    }
    let half = 0.5 * (inside + outside);
    let start = (0.7 * radius).floor() as usize;
    let end = ((1.3 * radius).ceil() as usize).min(profile.len().saturating_sub(1));
    for r in start..end {
        let (a, b) = (profile[r], profile[r + 1]);
        if a.is_finite() && b.is_finite() && a <= half && b > half {
            return r as f32 + (half - a) / (b - a);
        }
    }
    radius
}

/// Border-anchored shadows of the heavily smoothed residual.
fn find_vignetting(
    residual: &[f32],
    valid: &[bool],
    width: usize,
    height: usize,
    bin: usize,
    options: &DustOptions,
) -> Vec<(Vignetting, Vec<usize>)> {
    let long = width.max(height) as f64;
    let smooth = fill_smooth(residual, valid, width, height, (long / 100.0).max(3.0));
    let mask: Vec<u8> = (0..width * height)
        .map(|index| u8::from(valid[index] && smooth[index] < -options.vignetting_depth))
        .collect();
    let band_x = ((width as f32 * BORDER_FRACTION) as usize).max(2);
    let band_y = ((height as f32 * BORDER_FRACTION) as usize).max(2);
    let min_area = (0.03 * long).powi(2) as usize;
    let mut out = Vec::new();
    for component in connected_components(&mask, width, height, Connectivity::Eight) {
        if component.pixels.len() < min_area {
            continue;
        }
        let left = component.min_x < band_x;
        let right = component.max_x + band_x >= width;
        let top = component.min_y < band_y;
        let bottom = component.max_y + band_y >= height;
        if !(left || right || top || bottom) {
            continue;
        }
        let (fx, fy) = (
            component.centroid_x / width as f32,
            component.centroid_y / height as f32,
        );
        let horizontal = match (left, right) {
            (true, false) => Some(false),
            (false, true) => Some(true),
            (true, true) => Some(fx > 0.5),
            _ => None,
        };
        let vertical = match (top, bottom) {
            (true, false) => Some(false),
            (false, true) => Some(true),
            (true, true) => Some(fy > 0.5),
            _ => None,
        };
        let edge = match (horizontal, vertical) {
            (Some(false), Some(false)) => Edge::TopLeft,
            (Some(true), Some(false)) => Edge::TopRight,
            (Some(false), Some(true)) => Edge::BottomLeft,
            (Some(true), Some(true)) => Edge::BottomRight,
            (Some(false), None) => Edge::Left,
            (Some(true), None) => Edge::Right,
            (None, Some(false)) => Edge::Top,
            _ => Edge::Bottom,
        };
        let depth = component
            .pixels
            .iter()
            .map(|index| -smooth[*index])
            .fold(0.0_f32, f32::max);
        let scale = bin as f32;
        out.push((
            Vignetting {
                edge,
                x: (component.centroid_x + 0.5) * scale - 0.5,
                y: (component.centroid_y + 0.5) * scale - 0.5,
                area: component.pixels.len() as f32 * scale * scale,
                depth,
            },
            component.pixels,
        ));
    }
    out
}

/// The depth a dip of four pixels or fewer must reach so that a plane of
/// `pixels` pixels with smoothed noise `noise` reports about `false_dips`
/// noise dips: the one-sided normal threshold for that many tries, widened
/// by [`LONE_TAIL`] because smoothed noise has heavier tails than a normal.
pub(crate) fn lone_dip_floor(pixels: usize, noise: f32, false_dips: f64) -> f32 {
    upper_tail_sigma(false_dips.max(1e-9) / (pixels.max(1) as f64)) as f32 * noise * LONE_TAIL
}

/// How much heavier than a normal's the tail of smoothed noise runs. On
/// 70 flats of three telescopes (26 and 61 megapixels), the deepest noise
/// dips of a 3×3 median reach 6.4σ where a normal's would stop at 5.9σ.
const LONE_TAIL: f32 = 1.1;

/// The z for which a standard normal exceeds it with probability `p`, by
/// bisection on the complementary error function.
fn upper_tail_sigma(p: f64) -> f64 {
    let p = p.clamp(1e-300, 0.5);
    let (mut lo, mut hi) = (0.0_f64, 40.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if 0.5 * erfc(mid / std::f64::consts::SQRT_2) > p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Complementary error function (Numerical Recipes' Chebyshev fit, about
/// 1e-7 relative).
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

#[cfg(test)]
#[path = "dust_tests.rs"]
mod tests;
