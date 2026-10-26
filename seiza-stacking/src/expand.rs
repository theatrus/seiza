//! Planning and registration for a canvas grown to hold a connected set of
//! overlapping frames.
//!
//! Each frame registers against the footprints of the panels it may overlap,
//! never against the whole canvas: the brightest stars of a large canvas
//! crowd into a few panels, leaving a narrow overlap elsewhere with too few
//! of them to match. A similarity places every frame; a frame lying mostly
//! within one panel also gets the polynomial warp the registration model asks
//! for, fitted against that panel.
//!
//! The similarities placing each panel's first frame leave the lens's
//! distortion in the seams between panels. Once every frame has joined, one
//! radial distortion is fitted where those first frames overlap (see
//! [`crate::lens`]); when it explains what the similarities leave, the canvas
//! is planned again on undistorted grids, and every frame's warp leads
//! through the lens onto its own pixels.

use crate::binning::Scale;
use crate::lens::{LensDistortion, LensFit, LensFrame};
use crate::{
    CalibrationMasters, Continue, Error, FitsFrame, FrameMetadata, LinearImage, NormalizationMap,
    NormalizationMode, PolynomialWarp, ReferenceRegion, Registrar, RegistrationOptions,
    RegistrationResult, Result, SimilarityTransform, StackOptions, path_identity,
};
use rayon::prelude::*;
use seiza::DetectedStar;
use seiza_fits::HeaderValue;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};

/// A frame that covers at least this fraction of new sky starts a panel of
/// its own; dithered frames of a panel only extend its edges.
const NEW_PANEL_FRACTION: f64 = 0.25;

/// The chance a panel match may be a coincidence. Registration tries every
/// translation within its search radius, so the best of them pairs stars by
/// luck far above what any one transform would: on a real mosaic, a frame
/// sharing no sky with a panel paired 6 of its 800 brightest stars where one
/// transform expects 0.16, and in a crowded synthetic field 12 where it
/// expects 4. A true overlap between frames taken months apart paired 14 to 29.
const FALSE_MATCH_PROBABILITY: f64 = 1.0e-3;

/// The fraction of the stars a match claims the frame and panel share that
/// must actually pair. A pattern fixed to the sensor pairs far above chance
/// but only a sliver of them.
const MINIMUM_SHARED_FRACTION: f64 = 0.05;

/// A frame that neither starts a panel nor lies mostly within one keeps a
/// similarity, which cannot follow lens distortion across it, so where it
/// overlaps the frames warped onto a panel it would leave copies of each star.
/// Such a frame fits far worse than the warped frames around it: similarities
/// across this distortion fit at 1.1 to 1.4px RMS on a real mosaic, where its
/// warped frames fit at 0.04 to 0.20. One fitting more than this many times
/// the median of the frames that do not start a panel is rejected. On that
/// mosaic it rejected one S-II frame (0.43px), which cloud had already ruled
/// out, and no H-alpha or O-III frame; it guards against what the warp cannot
/// reach.
const RESIDUAL_OUTLIER_FACTOR: f64 = 5.0;

/// Below this RMS a non-starting frame is never rejected as an outlier,
/// however small the mosaic's median.
const RESIDUAL_OUTLIER_FLOOR_PIXELS: f64 = 0.3;

/// The non-starting frames a median of their residuals needs.
const RESIDUAL_OUTLIER_SAMPLE: usize = 5;

/// How many times the matching budget of stars planning pairs. An overlap
/// holds only a fraction of each frame's brightest stars, and frames taken on
/// different nights rank them differently; on a three-panel mosaic shot months
/// apart, 200 stars paired only 8 across one overlap and put the rotation half
/// a degree out, while 800 paired 29 and found it.
const PLANNING_STARS_FACTOR: usize = 4;

/// The brightest stars whose triangles seed a panel match. A narrow overlap
/// holds few of a frame's brightest stars, and fewer still in a filter as
/// faint as S-II: across one such overlap, triangles from 24 stars locked
/// onto a cluster at -0.78 degrees, pairing 22 stars at 1.86px RMS, while 40
/// or more agreed on -0.53 degrees, pairing twice as many at 1.27px, for
/// about a tenth of a second more per match.
const PLANNING_TRIANGLE_STARS: usize = 40;

/// The fraction of a frame's stars that must lie within the panel it matched
/// for a polynomial warp to be fitted against that panel. Lens distortion
/// reverses across a meridian flip and is largest at the frame's edges, where
/// panels overlap: on a real mosaic a similarity left 1.16px RMS between the
/// two sides of the pier, and a quadratic 0.06px. Fitted through a narrow
/// overlap, though, a quadratic would extrapolate across the rest of the
/// frame, so a frame starting a panel keeps its similarity, until a lens
/// measured across all the panels places it.
const WARP_COVERAGE: f64 = 0.5;

/// How close a frame's own star must land to a panel star for photometry to
/// measure that star. Lens distortion bends opposite edges of two panels
/// opposite ways: on a real mosaic a similarity matched the middle of an
/// overlap and missed its ends by 17px, where apertures read empty sky, and
/// the panel's gain came out 3.2 against a true 0.97.
const PHOTOMETRY_PAIRING_PIXELS: f64 = 1.5;

/// Cleanly measured stars a panel's gain needs. An overlap holds few stars,
/// and fewer still pair up across distortion: on a real S-II mosaic one join
/// had 10. Their median beats the alternative, the ratio of the two panels'
/// noise, which there read 3.0 where the stars said about 1.1.
const PANEL_PHOTOMETRY_STARS: usize = 5;

/// Within this many degrees of a half turn from the anchor, a frame is taken
/// to come from the other side of the pier. Sessions on one rig differ by a
/// few degrees when the camera comes off between them, by 10.8 on a real
/// mosaic; a frame at a quarter turn is neither side, and a panel may start
/// from it.
const PIER_FLIP_TOLERANCE_DEGREES: f64 = 20.0;

/// Stars detected in the anchor and in each frame starting a panel to measure
/// the lens by, against the 2000 a frame registers with: an overlap holds a
/// sixth of them or less.
const LENS_STARS: usize = 8_000;

/// How far a panel start's polynomial may stray from the camera rotation it
/// stands for, over the frame, before the lens model is abandoned.
const MAXIMUM_GEOMETRY_ERROR_PIXELS: f64 = 0.02;

/// How far the polynomial seeding a joined frame's registration through the
/// lens may stray from its fitted pose. The seed only predicts where each
/// star lies, for pairing within the registrar's 2 px, and the stars then
/// fit the warp, so it need not place pixels as a panel start does; a seed
/// it refused cost the frame a second blind match. On 129 Seestar frames,
/// the poses' polynomials missed the start's bound by up to 0.05 px on 70.
const SEED_GEOMETRY_ERROR_PIXELS: f64 = 0.2;

/// Frames planning reads, prepares and finds the stars of ahead of the one
/// registering: enough to keep the planning thread fed, few enough that the
/// prepared frames waiting stay a few hundred megabytes.
const PLAN_READ_AHEAD: usize = 3;

/// Columns and rows of the grid of a panel start's pixels its polynomial is
/// fitted through.
const GEOMETRY_GRID: (usize, usize) = (24, 16);

/// Where a prepared frame's colours sample the sky; see
/// [`crate::StackOptions::channel_sites`].
type Sites = Option<crate::registration::ChannelSites>;

/// A frame prepared again by its path, for a canvas with this many
/// channels.
type Prepare<'a> = dyn Fn(&Path, usize) -> Result<(LinearImage, Sites)> + 'a;

/// Where planning put a frame, or why it could not.
#[derive(Clone)]
pub(crate) struct PlannedRegistration {
    pub result: std::result::Result<RegistrationResult, String>,
    stamp: Option<crate::replay::SourceStamp>,
    /// Whether the frame anchored or started a panel, and so was placed
    /// without being warped onto one.
    starts_panel: bool,
}

/// The planned registration of the frame read from `path`: its placement,
/// the reason it could not join the canvas, or `None` when it was not
/// planned.
pub(crate) fn registration_for<'a>(
    plans: &'a HashMap<PathBuf, PlannedRegistration>,
    path: Option<&PathBuf>,
) -> Result<Option<std::result::Result<&'a RegistrationResult, &'a str>>> {
    let Some(path) = path else { return Ok(None) };
    let Some(plan) = plans.get(&path_identity(path)) else {
        return Ok(None);
    };
    if plan.stamp != crate::replay::SourceStamp::of(path) {
        return Err(Error::Stack(format!(
            "{} changed after expansion planning",
            path.display()
        )));
    }
    Ok(Some(plan.result.as_ref().map_err(String::as_str)))
}

/// Whether a frame at `rotation_radians` to the anchor lies on the other side
/// of the pier.
fn flipped(rotation_radians: f64) -> bool {
    (rotation_radians.to_degrees().rem_euclid(360.0) - 180.0).abs() <= PIER_FLIP_TOLERANCE_DEGREES
}

/// The canvas-to-creator geometry, before the shared lens is attached.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PanelProjection {
    pub width: usize,
    pub height: usize,
    pub warp: PolynomialWarp,
}

impl PanelProjection {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 || self.width.checked_mul(self.height).is_none() {
            return Err(Error::Registration(
                "panel projection has invalid sensor dimensions".into(),
            ));
        }
        self.warp
            .source_transformed(SimilarityTransform::IDENTITY)?;
        Ok(())
    }
}

/// One panel's footprint on the canvas, with the stars inside it.
#[derive(Clone, Debug)]
struct Window {
    region: ReferenceRegion,
    registrar: Registrar,
    /// The registrar's stars, on the panel's own grid.
    stars: Vec<(f64, f64)>,
    maximum_stars: usize,
    match_tolerance_pixels: f64,
    /// Roughly independent transforms registration tries: translations
    /// within the search radius one tolerance apart, and triangle candidates.
    hypotheses: f64,
    projection: Option<PanelProjection>,
    creator: Option<Registrar>,
}

impl Window {
    /// The panel at `region` of `canvas`, binned at `scale`, matching with
    /// `options` in the sensor's pixels, its stars detected where `gpu`
    /// allows.
    fn new(
        canvas: &LinearImage,
        region: ReferenceRegion,
        options: &RegistrationOptions,
        scale: Scale,
        gpu: &crate::GpuOptions,
    ) -> Result<Self> {
        Self::from_pixels(&crop(canvas, region, 0, 0)?, region, options, scale, gpu)
    }

    /// A panel at `region` whose stars are detected in `pixels`, an image of
    /// the region's size.
    fn from_pixels(
        pixels: &LinearImage,
        region: ReferenceRegion,
        options: &RegistrationOptions,
        scale: Scale,
        gpu: &crate::GpuOptions,
    ) -> Result<Self> {
        let options = Self::options(region, options, scale);
        let stars = crate::registration::detect_stars(
            pixels,
            &options,
            options.dense_star_count(),
            scale,
            gpu,
        );
        Self::from_stars(region, stars, options, scale, gpu)
    }

    /// The registration options, in working pixels, a panel at `region`
    /// matches with, from `options` in the sensor's pixels.
    fn options(
        region: ReferenceRegion,
        options: &RegistrationOptions,
        scale: Scale,
    ) -> RegistrationOptions {
        let mut options = scale.registration(options);
        // A frame overlapping this panel anywhere lies within about two of
        // its diagonals of the panel's center.
        options.maximum_drift_pixels = 2.0 * (region.width as f64).hypot(region.height as f64);
        options.maximum_stars *= PLANNING_STARS_FACTOR;
        options.triangle_stars = options.triangle_stars.max(PLANNING_TRIANGLE_STARS);
        options
    }

    /// A panel at `region` matching `stars`, on the region's grid, with
    /// [`Self::options`]. The stars of the frames it registers are detected
    /// where `gpu` allows.
    fn from_stars(
        region: ReferenceRegion,
        stars: Vec<DetectedStar>,
        options: RegistrationOptions,
        scale: Scale,
        gpu: &crate::GpuOptions,
    ) -> Result<Self> {
        let maximum_stars = options.maximum_stars;
        let match_tolerance_pixels = options.match_tolerance_pixels;
        let radius = options.effective_maximum_drift_pixels(region.width, region.height);
        let hypotheses =
            (radius / match_tolerance_pixels).powi(2) + options.maximum_candidates as f64;
        let registrar =
            Registrar::from_stars(region.width, region.height, stars, options, scale, gpu)?;
        Ok(Self {
            region,
            stars: registrar.reference_star_positions(),
            registrar,
            maximum_stars,
            match_tolerance_pixels,
            hypotheses,
            projection: None,
            creator: None,
        })
    }

    fn with_projection(mut self, projection: Option<PanelProjection>) -> Result<Self> {
        self.creator = match &projection {
            Some(projection) => {
                projection.validate()?;
                Some(self.registrar.remapped_stars(
                    projection.width,
                    projection.height,
                    |x, y| {
                        projection
                            .warp
                            .apply(x + self.region.x as f64, y + self.region.y as f64)
                    },
                )?)
            }
            None => None,
        };
        self.projection = projection;
        Ok(self)
    }

    /// Find the exposure on its panel's sensor plane, where a similarity
    /// describes dithers and rotation; retain the canvas projection exactly.
    fn register_projected(
        &self,
        stars: &[DetectedStar],
        width: usize,
        height: usize,
    ) -> Result<(RegistrationResult, f64)> {
        let projection = self
            .projection
            .as_ref()
            .ok_or_else(|| Error::Registration("panel has no projection".into()))?;
        let creator = self
            .creator
            .as_ref()
            .ok_or_else(|| Error::Registration("panel has no creator registrar".into()))?;
        if (width, height) != (projection.width, projection.height) {
            return Err(Error::Registration(
                "frame dimensions differ from the projected panel's sensor".into(),
            ));
        }
        let found = creator.register_similarity_stars(stars)?;
        let coverage = self.validate_match(
            &found,
            stars,
            (width, height),
            &creator.reference_star_positions(),
            (projection.width, projection.height),
        )?;
        let matched = found.transform;
        let (translation_x, translation_y) = matched.inverse_apply(0.0, 0.0);
        let inverse = SimilarityTransform {
            scale: 1.0 / matched.scale,
            rotation_radians: -matched.rotation_radians,
            translation_x,
            translation_y,
        };
        let warp = projection.warp.source_transformed(inverse)?;
        if coverage < WARP_COVERAGE {
            return Err(Error::Registration(
                "too little of the frame overlaps a projected panel to refine its geometry".into(),
            ));
        }
        let mut source = Vec::new();
        let mut canvas = Vec::new();
        for row in 0..=8 {
            for column in 0..=8 {
                let x = self.region.x as f64 + (self.region.width - 1) as f64 * column as f64 / 8.0;
                let y = self.region.y as f64 + (self.region.height - 1) as f64 * row as f64 / 8.0;
                let (u, v) = warp.apply(x, y);
                if u >= 0.0 && v >= 0.0 && u < width as f64 && v < height as f64 {
                    source.push((u, v));
                    canvas.push((x, y));
                }
            }
        }
        let transform = crate::lens::fit_similarity(&source, &canvas)
            .ok_or_else(|| Error::Registration("projected panel has no usable footprint".into()))?;
        let seed = RegistrationResult {
            transform,
            warp: Some(warp),
            ..found
        };
        let result = self.refine_warp(seed, stars).ok_or_else(|| Error::Registration(
            "projected registration has too few reliable stars to refine; frame was not integrated".into()
        ))?;
        Ok((result, coverage))
    }

    /// Register a `width` by `height` source's stars to this panel by
    /// similarity, returning a canvas transform and the fraction of the
    /// source's stars that land within the panel.
    fn register(
        &self,
        stars: &[DetectedStar],
        width: usize,
        height: usize,
    ) -> Result<(RegistrationResult, f64)> {
        if self.projection.is_some() {
            return self.register_projected(stars, width, height);
        }
        let mut result = self.registrar.register_similarity_stars(stars)?;
        let coverage = self.validate_match(
            &result,
            stars,
            (width, height),
            &self.stars,
            (self.region.width, self.region.height),
        )?;
        result.transform.translation_x += self.region.x as f64;
        result.transform.translation_y += self.region.y as f64;
        Ok((result, coverage))
    }

    /// Reject chance alignments and sparse sensor-fixed matches in the same
    /// coordinate plane in which the similarity was measured.
    fn validate_match(
        &self,
        result: &RegistrationResult,
        stars: &[DetectedStar],
        source_size: (usize, usize),
        reference_positions: &[(f64, f64)],
        reference_size: (usize, usize),
    ) -> Result<f64> {
        let (width, height) = source_size;
        let (reference_width, reference_height) = reference_size;
        let transform = result.transform;
        let inside = |(x, y): (f64, f64), width: usize, height: usize| {
            x >= 0.0 && y >= 0.0 && x < width as f64 && y < height as f64
        };
        let source_stars = &stars[..stars.len().min(self.maximum_stars)];
        let source_shared = source_stars
            .iter()
            .filter(|star| {
                inside(
                    transform.apply(star.x, star.y),
                    reference_width,
                    reference_height,
                )
            })
            .count();
        let panel_shared = reference_positions
            .iter()
            .filter(|&&(x, y)| inside(transform.inverse_apply(x, y), width, height))
            .count();
        // Pairs one transform finds by chance: each star it places on the
        // other image finds one of that image's stars within the tolerance as
        // often as their density allows.
        let disc = std::f64::consts::PI * self.match_tolerance_pixels.powi(2);
        let panel_density =
            reference_positions.len() as f64 / (reference_width * reference_height) as f64;
        let source_density = source_stars.len() as f64 / (width * height) as f64;
        let chance =
            (source_shared as f64 * panel_density).max(panel_shared as f64 * source_density) * disc;
        let required = chance_ceiling(chance, self.hypotheses);
        if result.matched_stars < required {
            return Err(Error::Registration(format!(
                "{} stars matched; the best of {:.0} transforms pairs up to {} by chance",
                result.matched_stars,
                self.hypotheses,
                required - 1
            )));
        }
        // Features fixed to the sensor pair at the same pixels in every frame,
        // so a frame of another target can match at no shift or turn at all:
        // two uncalibrated lights of M81 among a Rosette mosaic paired 9 to
        // 11 of the 800 stars they claimed to share, at exactly the anchor's
        // placement. Real overlaps paired a tenth or more.
        let shared = source_shared.min(panel_shared);
        if (result.matched_stars as f64) < MINIMUM_SHARED_FRACTION * shared as f64 {
            return Err(Error::Registration(format!(
                "only {} of the {shared} stars the overlap should share matched",
                result.matched_stars
            )));
        }
        Ok(source_shared as f64 / source_stars.len().max(1) as f64)
    }

    /// Fit the registration model's polynomial warp against this panel around
    /// a canvas `registration` of the source whose `stars` these are. Leaves
    /// a similarity model's registration, or one too few stars support,
    /// unchanged.
    fn warp(&self, registration: &mut RegistrationResult, stars: &[DetectedStar]) {
        let mut local = registration.transform;
        local.translation_x -= self.region.x as f64;
        local.translation_y -= self.region.y as f64;
        if let Some((warp, pairs, rms)) = self.registrar.warp_stars(local, stars) {
            registration.warp = Some(warp.translated(self.region.x as f64, self.region.y as f64));
            registration.matched_stars = pairs;
            registration.rms_error_pixels = rms;
        }
    }

    fn refine_warp(
        &self,
        mut seed: RegistrationResult,
        stars: &[DetectedStar],
    ) -> Option<RegistrationResult> {
        let local = seed
            .warp
            .as_ref()?
            .translated(-(self.region.x as f64), -(self.region.y as f64));
        let (warp, pairs, rms) = self.registrar.warp_stars_seeded(&local, stars)?;
        seed.warp = Some(warp.translated(self.region.x as f64, self.region.y as f64));
        seed.matched_stars = pairs;
        seed.rms_error_pixels = rms;
        Some(seed)
    }
}

/// Registers frames to an expanded canvas through its panels' footprints.
#[derive(Clone, Debug)]
pub(crate) struct CanvasRegistrar {
    width: usize,
    height: usize,
    windows: Vec<Window>,
    stars: Vec<(f64, f64)>,
    /// The canvas's [`crate::normalization::photometric_ceilings`], which
    /// photometry would otherwise read the whole canvas for, frame by frame.
    ceilings: Vec<f32>,
    /// The lens the canvas is undistorted from, when planning measured one.
    lens: Option<LensDistortion>,
    summary: Option<crate::ExpandedLens>,
    planning_notes: Vec<String>,
    /// The widest and tallest of the frames planning read, binned as the
    /// canvas is; `None` for a reopened context, which does not plan again.
    largest_frame: Option<(usize, usize)>,
}

impl CanvasRegistrar {
    /// Panels at `regions` of `canvas`, matching the stars planning saved
    /// for them, or, for a context saved before panels kept their stars,
    /// the stars of the canvas beneath them. `canvas`, its panels and `lens`
    /// are binned at `scale`, and `options` are in the sensor's pixels. The
    /// stars of the frames it registers are detected where `gpu` allows.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        canvas: &LinearImage,
        regions: &[ReferenceRegion],
        stars: Option<Vec<Vec<DetectedStar>>>,
        lens: Option<LensDistortion>,
        options: &RegistrationOptions,
        scale: Scale,
        gpu: &crate::GpuOptions,
    ) -> Result<Self> {
        if regions.is_empty() {
            return Err(Error::Stack("an expanded canvas needs a panel".into()));
        }
        if stars
            .as_ref()
            .is_some_and(|stars| stars.len() != regions.len())
        {
            return Err(Error::Stack(
                "an expanded canvas saves stars for every panel".into(),
            ));
        }
        let mut stars = stars.map(Vec::into_iter);
        let windows = regions
            .iter()
            .map(|region| {
                if region.width == 0
                    || region.height == 0
                    || region
                        .x
                        .checked_add(region.width)
                        .is_none_or(|x| x > canvas.width)
                    || region
                        .y
                        .checked_add(region.height)
                        .is_none_or(|y| y > canvas.height)
                {
                    return Err(Error::Stack(format!(
                        "panel {region:?} lies outside the {}x{} canvas",
                        canvas.width, canvas.height
                    )));
                }
                match stars.as_mut().and_then(Iterator::next) {
                    Some(stars) => Window::from_stars(
                        *region,
                        stars,
                        Window::options(*region, options, scale),
                        scale,
                        gpu,
                    ),
                    None => Window::new(canvas, *region, options, scale, gpu),
                }
            })
            .collect::<Result<Vec<_>>>()?;
        if let Some(lens) = &lens {
            lens.validate()?;
        }
        // Reported in the sensor's pixels, as planning measured it.
        let summary =
            lens.map(|lens| crate::ExpandedLens::new(&lens.rescaled(f64::from(scale.bin())), None));
        Ok(Self::from_windows(canvas, windows, lens).with_summary(summary))
    }

    pub(crate) fn planning_notes(&self) -> &[String] {
        &self.planning_notes
    }

    fn with_planning_notes(mut self, notes: Vec<String>) -> Self {
        self.planning_notes = notes;
        self
    }

    fn with_summary(self, summary: Option<crate::ExpandedLens>) -> Self {
        Self { summary, ..self }
    }

    fn with_largest_frame(self, largest_frame: (usize, usize)) -> Self {
        Self {
            largest_frame: Some(largest_frame),
            ..self
        }
    }

    /// The widest and tallest of the frames planning read, binned as the
    /// canvas is, or for a reopened context those of its panels' footprints,
    /// each of which holds the frame that started it.
    pub(crate) fn largest_frame(&self) -> (usize, usize) {
        self.largest_frame.unwrap_or_else(|| {
            self.windows.iter().fold((0, 0), |(width, height), window| {
                (
                    width.max(window.region.width),
                    height.max(window.region.height),
                )
            })
        })
    }

    /// What planning measured of the lens the canvas is undistorted from.
    pub(crate) fn lens_summary(&self) -> Option<crate::ExpandedLens> {
        self.summary
    }

    fn from_windows(
        canvas: &LinearImage,
        windows: Vec<Window>,
        lens: Option<LensDistortion>,
    ) -> Self {
        // Overlapping panels share stars; photometry should count each once.
        let mut seen = HashSet::new();
        let stars = windows
            .iter()
            .flat_map(|window| {
                window
                    .stars
                    .iter()
                    .map(|(x, y)| (x + window.region.x as f64, y + window.region.y as f64))
            })
            .filter(|(x, y)| seen.insert((x.round() as i64, y.round() as i64)))
            .collect();
        Self {
            width: canvas.width,
            height: canvas.height,
            windows,
            stars,
            ceilings: crate::normalization::photometric_ceilings(canvas),
            lens,
            summary: None,
            planning_notes: Vec::new(),
            largest_frame: None,
        }
    }

    /// Every panel's stars, on its own grid, as [`Self::new`] takes them.
    pub(crate) fn panel_stars(&self) -> Vec<Vec<DetectedStar>> {
        self.windows
            .iter()
            .map(|window| window.registrar.detected_stars().to_vec())
            .collect()
    }

    pub(crate) fn panel_projections(&self) -> Vec<Option<PanelProjection>> {
        if self
            .windows
            .iter()
            .all(|window| window.projection.is_none())
        {
            Vec::new()
        } else {
            self.windows
                .iter()
                .map(|window| window.projection.clone())
                .collect()
        }
    }

    pub(crate) fn with_projections(
        mut self,
        projections: Vec<Option<PanelProjection>>,
    ) -> Result<Self> {
        if projections.is_empty() {
            return Ok(self);
        }
        if projections.len() != self.windows.len() {
            return Err(Error::Stack(
                "an expanded canvas saves a projection for every panel".into(),
            ));
        }
        self.windows = self
            .windows
            .into_iter()
            .zip(projections)
            .map(|(window, projection)| window.with_projection(projection))
            .collect::<Result<_>>()?;
        Ok(self)
    }

    /// The lens the canvas is undistorted from, if any.
    pub(crate) fn lens(&self) -> Option<LensDistortion> {
        self.lens
    }

    /// The anchor panel's registrar, on the anchor's own grid. An expanded
    /// stack registers through [`Self::register`]; this stands in where a
    /// single registrar is required, without detecting stars canvas-wide.
    pub(crate) fn anchor_registrar(&self) -> Registrar {
        self.windows[0].registrar.clone()
    }

    pub(crate) fn regions(&self) -> Vec<ReferenceRegion> {
        self.windows.iter().map(|window| window.region).collect()
    }

    /// Every panel's brightest stars, on the canvas grid.
    pub(crate) fn star_positions(&self) -> Vec<(f64, f64)> {
        self.stars.clone()
    }

    /// The canvas's [`crate::normalization::photometric_ceilings`].
    pub(crate) fn photometric_ceilings(&self) -> &[f32] {
        &self.ceilings
    }

    /// The best-matched registration among the panels, warped against that
    /// panel when the frame lies mostly within it.
    pub(crate) fn register(&self, source: &LinearImage) -> Result<RegistrationResult> {
        if self.lens.is_some()
            && self
                .windows
                .iter()
                .all(|window| window.projection.is_none())
        {
            return Err(Error::Registration(
                "this saved mosaic has no panel projections; rebuild the expanded reference from the original frames before adding exposures".into(),
            ));
        }
        let mut stars = self.windows[0].registrar.detect_source_stars(source);
        if let Some(lens) = &self.lens {
            stars = lens.undistort_stars(&stars);
        }
        let mut best: Option<(RegistrationResult, f64, &Window)> = None;
        let mut last_error = None;
        for window in &self.windows {
            if self.lens.is_some() && window.projection.is_none() {
                continue;
            }
            match window.register(&stars, source.width, source.height) {
                Ok((result, coverage)) => {
                    if best
                        .as_ref()
                        .is_none_or(|(best, _, _)| better(&result, best))
                    {
                        best = Some((result, coverage, window));
                    }
                }
                Err(error) => last_error = Some(error),
            }
        }
        let (mut best, coverage, window) = best.ok_or_else(|| {
            last_error.unwrap_or_else(|| Error::Registration("no panel matched".into()))
        })?;
        if best.warp.is_none() && coverage >= WARP_COVERAGE {
            window.warp(&mut best, &stars);
        }
        let mut best = through_lens(best, self.lens);
        best.drift_pixels = best
            .transform
            .displacement_at(self.width as f64 * 0.5, self.height as f64 * 0.5);
        Ok(best)
    }
}

/// Reject the frames that neither start a panel nor fit like the frames
/// warped onto one; see [`RESIDUAL_OUTLIER_FACTOR`]. Residuals are in working
/// pixels, binned at `scale`.
fn reject_residual_outliers(plans: &mut HashMap<PathBuf, PlannedRegistration>, scale: Scale) {
    let mut residuals = plans
        .values()
        .filter(|plan| !plan.starts_panel)
        .filter_map(|plan| plan.result.as_ref().ok())
        .map(|result| result.rms_error_pixels)
        .collect::<Vec<_>>();
    if residuals.len() < RESIDUAL_OUTLIER_SAMPLE {
        return;
    }
    residuals.sort_by(f64::total_cmp);
    let median = residuals[residuals.len() / 2];
    let limit = (RESIDUAL_OUTLIER_FACTOR * median).max(scale.length(RESIDUAL_OUTLIER_FLOOR_PIXELS));
    for plan in plans.values_mut().filter(|plan| !plan.starts_panel) {
        if let Ok(result) = &plan.result
            && result.rms_error_pixels > limit
        {
            plan.result = Err(format!(
                "registered at {:.2}px RMS where this mosaic's frames median {median:.2}px, \
                 too loosely to stack without leaving copies of the stars it overlaps",
                result.rms_error_pixels
            ));
        }
    }
}

/// The `panel` stars within `tolerance` of a `landed` frame star.
fn paired_stars(panel: &[(f64, f64)], landed: &[(f64, f64)], tolerance: f64) -> Vec<(f64, f64)> {
    let cell = |x: f64, y: f64| {
        (
            (x / tolerance).floor() as i64,
            (y / tolerance).floor() as i64,
        )
    };
    let mut cells = HashMap::<(i64, i64), Vec<(f64, f64)>>::new();
    for &(x, y) in landed {
        cells.entry(cell(x, y)).or_default().push((x, y));
    }
    panel
        .iter()
        .copied()
        .filter(|&(x, y)| {
            let (cx, cy) = cell(x, y);
            (cx - 1..=cx + 1).any(|nx| {
                (cy - 1..=cy + 1).any(|ny| {
                    cells.get(&(nx, ny)).is_some_and(|near| {
                        near.iter()
                            .any(|&(sx, sy)| (sx - x).hypot(sy - y) <= tolerance)
                    })
                })
            })
        })
        .collect()
}

/// The fewest pairs that `hypotheses` transforms, each pairing a Poisson
/// count of mean `chance` stars by coincidence, reach with probability at
/// most [`FALSE_MATCH_PROBABILITY`].
fn chance_ceiling(chance: f64, hypotheses: f64) -> usize {
    let limit = (FALSE_MATCH_PROBABILITY / hypotheses.max(1.0)).ln();
    let chance = chance.max(1.0e-12);
    // log P(X = k) for k up to well past the mean, then tails from the top.
    let last = (chance + 20.0 * chance.sqrt() + 60.0).ceil() as usize;
    let mut log_pmf = Vec::with_capacity(last + 1);
    let mut log_factorial = 0.0;
    for k in 0..=last {
        if k > 0 {
            log_factorial += (k as f64).ln();
        }
        log_pmf.push(k as f64 * chance.ln() - chance - log_factorial);
    }
    let mut log_tail = f64::NEG_INFINITY;
    let mut ceiling = last + 1;
    for k in (0..=last).rev() {
        let (high, low) = if log_pmf[k] > log_tail {
            (log_pmf[k], log_tail)
        } else {
            (log_tail, log_pmf[k])
        };
        log_tail = high + (low - high).exp().ln_1p();
        if log_tail > limit {
            break;
        }
        ceiling = k;
    }
    ceiling
}

/// A registration found on undistorted stars, given the warp that carries
/// it through `lens` onto the source's pixels.
fn through_lens(
    mut registration: RegistrationResult,
    lens: Option<LensDistortion>,
) -> RegistrationResult {
    if let Some(lens) = lens {
        let warp = registration
            .warp
            .take()
            .unwrap_or_else(|| PolynomialWarp::from_similarity(registration.transform));
        registration.warp = Some(warp.with_distortion(lens));
    }
    registration
}

fn better(candidate: &RegistrationResult, incumbent: &RegistrationResult) -> bool {
    candidate.matched_stars > incumbent.matched_stars
        || (candidate.matched_stars == incumbent.matched_stars
            && candidate.rms_error_pixels < incumbent.rms_error_pixels)
}

#[derive(Clone)]
pub(crate) struct Expansion {
    pub image: LinearImage,
    pub headers: Vec<(String, HeaderValue)>,
    pub metadata: FrameMetadata,
    pub noise: Vec<f32>,
    pub registrations: HashMap<PathBuf, PlannedRegistration>,
    pub canvas: CanvasRegistrar,
}

/// The canvas as planning grows it.
struct Plan {
    image: LinearImage,
    headers: Vec<(String, HeaderValue)>,
    /// Finite samples on the canvas.
    finite: usize,
    windows: Vec<Window>,
    /// The panel the previous frame joined, which its dithered neighbors
    /// most likely overlap too.
    last_window: usize,
    registrations: HashMap<PathBuf, PlannedRegistration>,
    /// The lens every frame's stars are undistorted by, once measured.
    lens: Option<LensDistortion>,
    /// Where the anchor's undistorted grid has its origin on the canvas.
    origin: (f64, f64),
    record: Record,
}

/// Only acquisition settings relevant to sharing a lens. Filter, date and
/// photometric/calibration settings deliberately do not participate.
struct OpticalMetadata {
    camera: Option<String>,
    values: [Option<f64>; 7],
}

impl OpticalMetadata {
    fn from_frame(frame: &FitsFrame) -> Self {
        let signature = frame.metadata().signature;
        let pixel = |key: &str| {
            frame
                .headers
                .iter()
                .find(|(k, _)| k == key)
                .and_then(|(_, value)| value.as_f64())
        };
        let positive = |value: Option<f64>| value.filter(|v| v.is_finite() && *v > 0.0);
        Self {
            camera: signature
                .camera
                .map(|s| {
                    s.split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" ")
                        .to_ascii_lowercase()
                })
                .filter(|s| !s.is_empty()),
            values: [
                Some(frame.image.width as f64),
                Some(frame.image.height as f64),
                positive(signature.binning_x.map(|v| v as f64)),
                positive(signature.binning_y.map(|v| v as f64)),
                positive(pixel("XPIXSZ")),
                positive(pixel("YPIXSZ")),
                positive(signature.focal_length_mm),
            ],
        }
    }
}

/// Compare all known values, even when the anchor omitted a field. Missing
/// values cannot prove compatibility, but do not veto an empirical fit.
fn optical_compatibility(frames: &[(PathBuf, OpticalMetadata)]) -> (bool, Vec<String>) {
    let labels = [
        "width",
        "height",
        "X binning",
        "Y binning",
        "X pixel size",
        "Y pixel size",
        "focal length",
    ];
    let mut missing = Vec::new();
    let mut notes = Vec::new();
    let mut camera: Option<(&Path, &str)> = None;
    for (path, frame) in frames {
        if let Some(value) = frame.camera.as_deref() {
            if let Some((first, expected)) = camera {
                if value != expected {
                    notes.push(format!("shared lens fit skipped: camera differs between {} ({expected}) and {} ({value})", first.display(), path.display()));
                    break;
                }
            } else {
                camera = Some((path, value));
            }
        } else if !missing.contains(&"camera") {
            missing.push("camera");
        }
    }
    for (index, label) in labels.iter().enumerate() {
        let mut known: Option<(&Path, f64)> = None;
        for (path, frame) in frames {
            if let Some(value) = frame.values[index] {
                if let Some((first, expected)) = known {
                    // Dimensions/binning are exact; physical measurements
                    // may be rounded differently by acquisition software.
                    let tolerance = if index >= 4 {
                        expected.abs().max(value.abs()) * 0.01
                    } else {
                        0.0
                    };
                    if (value - expected).abs() > tolerance {
                        notes.push(format!("shared lens fit skipped: {label} differs between {} ({expected}) and {} ({value})", first.display(), path.display()));
                        break;
                    }
                } else {
                    known = Some((path, value));
                }
            } else if !missing.contains(label) {
                missing.push(label);
            }
        }
    }
    let compatible = notes.is_empty();
    if compatible && !missing.is_empty() {
        notes.push(format!("shared lens optical compatibility could not be verified: missing {} on one or more joined frames; lens fitting remains allowed", missing.join(", ")));
    }
    (compatible, notes)
}

/// What planning learns of each frame, to plan again through a lens.
#[derive(Default)]
struct Record {
    /// The anchor's size and headers, before the canvas moved its reference
    /// pixel.
    anchor_size: (usize, usize),
    anchor_headers: Vec<(String, HeaderValue)>,
    optics: Vec<(PathBuf, OpticalMetadata)>,
    /// The anchor first, then every frame that started a panel, in order.
    starts: Vec<Start>,
    /// Every frame that joined, in order, the anchor aside.
    joined: Vec<Joined>,
}

struct Start {
    path: PathBuf,
    /// Up to [`LENS_STARS`] of its brightest stars, on its own sensor.
    stars: Vec<DetectedStar>,
    /// The panel it registered to, and whether it started one of its own.
    window: usize,
    creates_window: bool,
}

struct Joined {
    path: PathBuf,
    stars: Vec<DetectedStar>,
    size: (usize, usize),
    window: usize,
    starts_panel: bool,
}

/// Each frame joins the growing canvas through actual common stars and pixels.
/// Unconnected frames are retried after growth, so input order need not follow
/// adjacency. `maximum_samples` bounds the canvas's pixels times channels.
pub(crate) fn plan(
    paths: &[PathBuf],
    calibration: &CalibrationMasters,
    options: &StackOptions,
    maximum_samples: usize,
    mut progress: impl FnMut(&Path, usize, usize, usize) -> Continue,
) -> Result<Expansion> {
    let first = paths
        .first()
        .ok_or_else(|| Error::Stack("expansion needs light frames".into()))?;
    let scale = options.scale();
    // Keep the original identity across deferred matches and the lens pass.
    // A later read must never combine new pixels with cached star detections.
    let stamps: HashMap<_, _> = paths
        .iter()
        .map(|path| (path.clone(), crate::replay::SourceStamp::of(path)))
        .collect();
    let check_source = |path: &Path| -> Result<()> {
        if stamps[path] != crate::replay::SourceStamp::of(path) {
            return Err(Error::Stack(format!(
                "{} changed during expansion planning",
                path.display()
            )));
        }
        Ok(())
    };
    check_source(first)?;
    let frame = FitsFrame::open(first)?;
    let metadata = frame.metadata();
    let anchor_optics = OpticalMetadata::from_frame(&frame);
    let (mut frame, anchor_cfa) =
        crate::stack::prepare_light(frame, calibration, &options.match_tolerances, options)
            .map_err(crate::stack::LightFailure::into_error)?;
    check_source(first)?;
    if frame.image.sample_count() > maximum_samples {
        return Err(Error::Stack(format!(
            "the anchor frame's {} samples exceed the expanded-canvas limit of {maximum_samples}",
            frame.image.sample_count()
        )));
    }
    let noise = if options.weighting.is_equal() {
        Vec::new()
    } else {
        crate::stack::measure_reference_noise(&frame.image)?
    };
    // The canvas starts as the anchor's pixels, its colours resampled onto
    // their centres as every frame placed on it is.
    frame.image = crate::stack::co_site(
        frame.image,
        options.channel_sites(anchor_cfa),
        options.interpolation,
        &options.gpu,
    )?;
    let anchor = ReferenceRegion {
        x: 0,
        y: 0,
        width: frame.image.width,
        height: frame.image.height,
    };
    // For the memory a stack plans each frame it prepares with.
    let mut largest_frame = (anchor.width, anchor.height);
    let anchor = Window::new(
        &frame.image,
        anchor,
        &options.registration,
        scale,
        &options.gpu,
    )?;
    let record = Record {
        anchor_size: (frame.image.width, frame.image.height),
        anchor_headers: frame.headers.clone(),
        optics: vec![(first.clone(), anchor_optics)],
        starts: vec![Start {
            path: first.clone(),
            stars: crate::registration::detect_stars(
                &frame.image,
                &options.registration,
                LENS_STARS,
                scale,
                &options.gpu,
            ),
            window: 0,
            creates_window: true,
        }],
        joined: Vec::new(),
    };
    let mut plan = Plan {
        finite: frame.image.data.iter().filter(|v| v.is_finite()).count(),
        registrations: HashMap::from([(
            path_identity(first),
            PlannedRegistration {
                stamp: stamps[first],
                starts_panel: true,
                result: Ok(RegistrationResult {
                    transform: SimilarityTransform::IDENTITY,
                    // The anchor matches each of its own stars exactly.
                    matched_stars: anchor.stars.len(),
                    rms_error_pixels: 0.0,
                    drift_pixels: 0.0,
                    warp: None,
                }),
            },
        )]),
        image: frame.image,
        headers: frame.headers,
        windows: vec![anchor],
        last_window: 0,
        lens: None,
        origin: (0.0, 0.0),
        record,
    };
    // A frame as the stack prepares it, binned by `bin_factor`: the
    // stack's own, or 1 for the lens's stars.
    let prepare_at = |path: &Path,
                      channels: usize,
                      bin_factor: u32|
     -> Result<(LinearImage, OpticalMetadata, Sites)> {
        check_source(path)?;
        let frame = FitsFrame::open(path)?;
        let optics = OpticalMetadata::from_frame(&frame);
        let (frame, cfa) = crate::stack::prepare_light_at(
            frame,
            calibration,
            &options.match_tolerances,
            options,
            bin_factor,
        )
        .map_err(crate::stack::LightFailure::into_error)?;
        check_source(path)?;
        if frame.image.channels != channels {
            return Err(Error::Stack(format!(
                "{} has a different channel count",
                path.display()
            )));
        }
        let sites = options.channel_sites_at(cfa, bin_factor);
        Ok((frame.image, optics, sites))
    };
    let prepare = |path: &Path, channels: usize| prepare_at(path, channels, options.bin_factor);
    // A frame read for the first time, prepared, with the stars the anchor's
    // panel detects in it: the detector depends only on the options, so it
    // runs off the planning thread.
    let detector = plan.windows[0].registrar.clone();
    let channels = plan.image.channels;
    let measure =
        |path: &Path| -> Result<(LinearImage, OpticalMetadata, Sites, Vec<DetectedStar>)> {
            let (prepared, optics, sites) = prepare(path, channels)?;
            let stars = detector.detect_source_stars(&prepared);
            Ok((prepared, optics, sites, stars))
        };
    let mut pending = paths[1..]
        .iter()
        .map(|path| Pending {
            path,
            stars: Vec::new(),
            size: None,
            optics: None,
            tried: 0,
            error: None,
            overlap: None,
        })
        .collect::<Vec<_>>();
    // Whether a frame from the other side of the pier may start a panel: only
    // once a pass finds nothing else to join.
    let mut flipped_start = false;
    let mut unjoined = None;
    while !pending.is_empty() {
        let before = pending.len();
        let mut deferred = Vec::new();
        let mut first_error = None;
        let mut waiting_for_pier_side = false;
        // Every frame not yet read, in this pass's order, is read, prepared
        // and has its stars found a few frames ahead, while earlier frames
        // register and join; each joins as before, in order, with the same
        // stars, and a frame's error surfaces when the pass reaches it.
        let unread = pending
            .iter()
            .filter(|item| item.size.is_none())
            .map(|item| item.path)
            .collect::<Vec<_>>();
        crate::tasks::ahead(|ahead| {
            std::thread::scope(|scope| -> Result<()> {
                let mut unread = unread.into_iter();
                let mut reading = std::collections::VecDeque::new();
                for mut item in pending {
                    while reading.len() < PLAN_READ_AHEAD
                        && let Some(path) = unread.next()
                    {
                        reading.push_back(ahead.spawn(
                            scope,
                            || vec![path.to_path_buf()],
                            move || measure(path),
                        ));
                    }
                    let path = item.path;
                    if progress(
                        path,
                        plan.registrations.len(),
                        plan.image.width,
                        plan.image.height,
                    ) == Continue::No
                    {
                        return Err(Error::Cancelled);
                    }
                    check_source(path)?;
                    // A frame that matched no panel keeps its stars, and is read
                    // again only once a panel added since then matches it.
                    let mut image = None;
                    if item.size.is_none() {
                        let measured = reading
                            .pop_front()
                            .expect("every unread frame is read ahead");
                        let (prepared, optics, sites, stars) = ahead.join(measured)?;
                        item.optics = Some(optics);
                        item.stars = stars;
                        item.size = Some((prepared.width, prepared.height));
                        largest_frame = (
                            largest_frame.0.max(prepared.width),
                            largest_frame.1.max(prepared.height),
                        );
                        image = Some((prepared, sites));
                    }
                    let (width, height) = item.size.expect("the frame was measured");
                    let panels = plan.windows.len();
                    let joined =
                        match plan.register(&item.stars, width, height, item.tried, options) {
                            Err(error) => {
                                item.tried = panels;
                                Err(error)
                            }
                            Ok((registration, window, coverage)) => {
                                let (image, sites) = match image {
                                    Some(image) => image,
                                    None => {
                                        let (image, _, sites) = prepare(path, plan.image.channels)?;
                                        (image, sites)
                                    }
                                };
                                let offer = plan.join(
                                    path,
                                    (&image, sites),
                                    &item.stars,
                                    (registration, window, coverage),
                                    flipped_start,
                                    options,
                                    maximum_samples,
                                );
                                // The canvas may yet grow enough for any panel to admit it.
                                item.tried = 0;
                                offer.map(|offer| (offer, window))
                            }
                        };
                    match joined {
                        Ok((Offer::Joined(registration, starts_panel), window)) => {
                            plan.registrations.insert(
                                path_identity(path),
                                PlannedRegistration {
                                    result: Ok(registration),
                                    stamp: stamps[path],
                                    starts_panel,
                                },
                            );
                            plan.record.optics.push((
                                path.clone(),
                                item.optics.take().expect("the frame was measured"),
                            ));
                            plan.record.joined.push(Joined {
                                path: path.clone(),
                                stars: std::mem::take(&mut item.stars),
                                size: (width, height),
                                window,
                                starts_panel,
                            });
                        }
                        Ok((Offer::Flipped, _)) => {
                            waiting_for_pier_side = true;
                            deferred.push(item);
                        }
                        Ok((Offer::Short(overlap), _)) => {
                            item.overlap =
                                Some(item.overlap.map_or(overlap, |best| best.max(overlap)));
                            let error = Error::Registration(format!(
                                "panel overlap {overlap:.3} is below {}",
                                options.acceptance.minimum_overlap_fraction
                            ));
                            first_error
                                .get_or_insert_with(|| format!("{}: {error}", path.display()));
                            item.error = Some(error.to_string());
                            deferred.push(item);
                        }
                        Err(error) => {
                            if !retryable(&error) {
                                return Err(error);
                            }
                            first_error
                                .get_or_insert_with(|| format!("{}: {error}", path.display()));
                            item.error = Some(error.to_string());
                            deferred.push(item);
                        }
                    }
                }
                Ok(())
            })
        })?;
        if deferred.len() == before {
            if waiting_for_pier_side && !flipped_start {
                flipped_start = true;
                pending = deferred;
                continue;
            }
            // Nothing joined the anchor: there is no mosaic to build.
            if plan.registrations.len() == 1 {
                return Err(Error::Stack(format!(
                    "cannot expand: {} frame(s) could not join the overlapping mosaic; {}",
                    deferred.len(),
                    first_error.unwrap_or_default()
                )));
            }
            // A frame of another target, or cloud, would otherwise cost the
            // whole mosaic; it is rejected when pushed, with the reason. So
            // would a panel overlapping a little less than the minimum: that
            // is said now, rather than only frame by frame as each is pushed.
            unjoined = unjoined_note(
                &deferred,
                paths.len(),
                plan.windows.len(),
                options.acceptance.minimum_overlap_fraction,
            );
            for item in deferred {
                plan.registrations.insert(
                    path_identity(item.path),
                    PlannedRegistration {
                        result: Err(item.error.unwrap_or_else(|| "no panel matched".into())),
                        stamp: stamps[item.path],
                        starts_panel: false,
                    },
                );
            }
            break;
        }
        flipped_start = false;
        pending = deferred;
    }
    let mut summary = None;
    let (compatible, optical_notes) = optical_compatibility(&plan.record.optics);
    let mut planning_notes = unjoined
        .into_iter()
        .chain(optical_notes)
        .collect::<Vec<_>>();
    // The lens bends neighboring panels' edges opposite ways, which the
    // similarities placing the panels' first frames cannot follow; measured
    // where those frames overlap, it is undone and the canvas planned again.
    let estimate = if compatible {
        plan.estimate_lens(
            &|path, channels| prepare_at(path, channels, 1).map(|(image, _, _)| image),
            options,
        )?
    } else {
        None
    };
    if let Some((fit, measured)) = estimate {
        match plan.rebuild(
            &fit,
            first,
            &|path, channels| prepare(path, channels).map(|(image, _, sites)| (image, sites)),
            options,
            maximum_samples,
            &mut progress,
        ) {
            Ok(rebuilt) => {
                summary = Some(measured);
                plan = rebuilt;
            }
            // Only a geometric or photometric failure can keep the first
            // plan. Resource, input, and cancellation errors remain fatal.
            Err(error) if retryable(&error) => planning_notes.push(format!(
                "lens-corrected canvas rebuild failed; retaining the original plan: {error}"
            )),
            Err(error) => return Err(error),
        }
    }
    reject_residual_outliers(&mut plan.registrations, scale);
    let (width, height) = (plan.image.width, plan.image.height);
    for planned in plan
        .registrations
        .values_mut()
        .filter_map(|planned| planned.result.as_mut().ok())
    {
        planned.drift_pixels = planned
            .transform
            .displacement_at(width as f64 * 0.5, height as f64 * 0.5);
    }
    if progress(first, plan.registrations.len(), width, height) == Continue::No {
        return Err(Error::Cancelled);
    }
    for path in paths {
        check_source(path)?;
    }
    Ok(Expansion {
        canvas: CanvasRegistrar::from_windows(&plan.image, plan.windows, plan.lens)
            .with_summary(summary)
            .with_planning_notes(planning_notes)
            .with_largest_frame(largest_frame),
        image: plan.image,
        headers: plan.headers,
        metadata,
        noise,
        registrations: plan.registrations,
    })
}

/// A frame resampled onto the canvas it would grow, before committing.
struct Placement {
    width: usize,
    height: usize,
    dx: usize,
    dy: usize,
    transform: SimilarityTransform,
    region: ReferenceRegion,
    /// The frame resampled onto `region`, when counting its coverage needed
    /// it; see [`crate::registration::resampled_coverage`].
    incoming: Option<LinearImage>,
    /// Samples both define, and samples only the frame defines.
    common: usize,
    uncovered: usize,
}

/// A frame waiting to join the canvas.
struct Pending<'a> {
    path: &'a PathBuf,
    stars: Vec<DetectedStar>,
    /// Width and height, once read.
    size: Option<(usize, usize)>,
    optics: Option<OpticalMetadata>,
    /// Panels it has already failed to match.
    tried: usize,
    /// Why it last failed to join.
    error: Option<String>,
    /// The most of its footprint it has shared with the canvas when it
    /// matched a panel but overlapped too little to join.
    overlap: Option<f64>,
}

/// What became of a frame offered to the canvas.
enum Offer {
    /// It joined, starting a panel or not.
    Joined(RegistrationResult, bool),
    /// It would start a panel from the other side of the pier than the
    /// anchor, which waits until a pass finds nothing else to join.
    Flipped,
    /// It matched a panel, but shared only this fraction of the smaller
    /// footprint with the canvas, below the minimum overlap.
    Short(f64),
}

/// What planning tells of the `unjoined` lights, of `total`, that it left
/// off a canvas of `panels` panels, or `None` when every light joined. Each
/// is rejected, with its reason, when pushed. A mosaic planned for 10%
/// overlap lands nearer 9%, so a minimum just above that leaves every other
/// panel out without failing planning; the note says so at once, and how
/// close they came.
fn unjoined_note(
    unjoined: &[Pending<'_>],
    total: usize,
    panels: usize,
    minimum_overlap: f32,
) -> Option<String> {
    if unjoined.is_empty() {
        return None;
    }
    let short = unjoined
        .iter()
        .filter_map(|item| item.overlap)
        .collect::<Vec<_>>();
    let mut note = format!(
        "{} of {total} lights did not join the canvas, which holds {panels} panel{}, \
         and are rejected when pushed",
        unjoined.len(),
        if panels == 1 { "" } else { "s" }
    );
    if let Some(best) = short.iter().copied().reduce(f64::max) {
        let some = if short.len() == unjoined.len() {
            "they".to_string()
        } else {
            format!("{} of them", short.len())
        };
        note.push_str(&format!(
            "; {some} matched a panel but overlapped the canvas by at most {best:.3}, below \
             the minimum overlap of {minimum_overlap}: if they are panels of this mosaic, \
             a lower minimum overlap admits them"
        ));
    }
    Some(note)
}

impl Plan {
    /// Grow the canvas to the footprint of `source`, registered to `window`
    /// with `coverage` of its stars inside it, and fill the sky it newly
    /// covers. Leaves the plan untouched unless the frame joins, and does not
    /// join one that would start a panel from the other side of the pier
    /// than the anchor, unless `flipped_start` allows it.
    #[allow(clippy::too_many_arguments)]
    fn join(
        &mut self,
        path: &Path,
        (source, sites): (&LinearImage, Sites),
        stars: &[DetectedStar],
        (registration, window, coverage): (RegistrationResult, usize, f64),
        flipped_start: bool,
        options: &StackOptions,
        maximum_samples: usize,
    ) -> Result<Offer> {
        let criteria = options.acceptance;
        let finite_source = source.data.iter().filter(|v| v.is_finite()).count() as f64;
        let mut registration = registration;
        let mut refined = false;
        let placed = loop {
            let placed = self.place((source, sites), &registration, options, maximum_samples)?;
            // A frame starting a panel sets the geometry its neighbors follow,
            // so it is refit; frames within known sky keep the panel's fit.
            let source_area = finite_source * placed.transform.scale.powi(2);
            if refined || (placed.uncovered as f64) < NEW_PANEL_FRACTION * source_area {
                break placed;
            }
            refined = true;
            match self.refine(&registration, source, stars, options) {
                Some(better) => registration = better,
                None => break placed,
            }
        };
        // A panel's first frame keeps its similarity, so its edges carry its
        // lens distortion into the seams with its neighbors. That distortion
        // reverses across a meridian flip; a panel started from the anchor's
        // side of the pier meets its neighbors on equal terms.
        if refined && !flipped_start && flipped(registration.transform.rotation_radians) {
            return Ok(Offer::Flipped);
        }
        let Placement {
            width,
            height,
            dx,
            dy,
            transform,
            region,
            incoming,
            common,
            uncovered,
        } = placed;
        let source_area = finite_source * transform.scale.powi(2);
        let overlap = common as f64 / (self.finite as f64).min(source_area).max(1.0);
        if common == 0 || overlap < f64::from(criteria.minimum_overlap_fraction) {
            return Ok(Offer::Short(overlap));
        }

        // Only a panel's first frame fills the canvas. Its siblings would add
        // slivers at most, resampled through a similarity that across a
        // meridian flip misses the stars that photometry measures; the
        // integration normalizes every frame on its own.
        let starts_panel = refined;
        let mut incoming = match incoming {
            Some(incoming) => Some(incoming),
            None if starts_panel => Some(crate::registration::resample_region_sited(
                source,
                width,
                height,
                region,
                crate::registration::FrameGeometry::Similarity(transform),
                options.interpolation.into(),
                sites,
                &options.gpu,
            )?),
            None => None,
        };
        if let Some(incoming) = incoming.as_mut().filter(|_| starts_panel && uncovered > 0) {
            let landed = stars
                .iter()
                .map(|star| {
                    let (x, y) = transform.apply(star.x, star.y);
                    (x - region.x as f64, y - region.y as f64)
                })
                .collect::<Vec<_>>();
            let existing = crop(&self.image, region, dx, dy)?;
            self.match_panel(
                incoming,
                &existing,
                region,
                (dx, dy),
                &landed,
                window,
                options,
            )?;
        }

        // Every check has passed: commit.
        let filled = self.commit(
            (width, height, dx, dy),
            region,
            incoming.as_ref().filter(|_| starts_panel),
        )?;
        let mut joined = RegistrationResult {
            transform,
            ..registration
        };
        if !starts_panel && coverage >= WARP_COVERAGE {
            self.windows[window].warp(&mut joined, stars);
        }
        // The panel's stars come from its first frame alone. The canvas under
        // its footprint also holds its neighbours' edges, placed by their own
        // similarities and bent by distortion the other way; warps fitted to
        // such a mixture compromise between the two. On a real mosaic, frames
        // warped onto a panel built from the canvas fitted at 0.48 to 0.72px
        // RMS and left copies of each star; built from its first frame, the
        // same frames fit at 0.05px.
        let mut creates_window = false;
        if let Some(incoming) = incoming.as_ref().filter(|_| starts_panel)
            && filled as f64 >= NEW_PANEL_FRACTION * source_area
            && let Ok(panel) = Window::from_pixels(
                incoming,
                region,
                &options.registration,
                options.scale(),
                &options.gpu,
            )
        {
            self.windows.push(panel);
            self.last_window = self.windows.len() - 1;
            creates_window = true;
        }
        if starts_panel {
            self.record.starts.push(Start {
                path: path.to_path_buf(),
                stars: crate::registration::detect_stars(
                    source,
                    &options.registration,
                    LENS_STARS,
                    options.scale(),
                    &options.gpu,
                ),
                window,
                creates_window,
            });
        }
        Ok(Offer::Joined(joined, starts_panel))
    }

    /// Match a panel-starting frame's pixels, resampled onto `region` of the
    /// canvas about to move `dx`, `dy` from its origin, to the canvas through
    /// their overlap, by the stars `landed` on `region`'s grid that pair with
    /// `window`'s.
    #[allow(clippy::too_many_arguments)]
    fn match_panel(
        &self,
        incoming: &mut LinearImage,
        existing: &LinearImage,
        region: ReferenceRegion,
        (dx, dy): (usize, usize),
        landed: &[(f64, f64)],
        window: usize,
        options: &StackOptions,
    ) -> Result<()> {
        let criteria = options.acceptance;
        // Match the new sky to the canvas through the overlap. The canvas
        // is a registration and photometric template, not an integrated
        // observation, so covered pixels keep their first value.
        let panel_stars = self.windows[window]
            .stars
            .iter()
            .map(|&(x, y)| {
                (
                    x + (self.windows[window].region.x + dx) as f64 - region.x as f64,
                    y + (self.windows[window].region.y + dy) as f64 - region.y as f64,
                )
            })
            .filter(|&(x, y)| {
                x >= 0.0 && y >= 0.0 && x < region.width as f64 && y < region.height as f64
            })
            .collect::<Vec<_>>();
        // Panels are shot on different nights, under different skies, so
        // their noise says nothing of their transparency: a gain comes
        // from stars or not at all, leaving only the background matched.
        let gains = matches!(
            options.normalization,
            NormalizationMode::LocalBackground { .. }
        )
        .then(|| {
            let scale = options.scale();
            [
                scale.floored_length(PHOTOMETRY_PAIRING_PIXELS, 0.75),
                scale.floored_length(options.registration.match_tolerance_pixels, 1.0),
            ]
            .into_iter()
            .find_map(|tolerance| {
                crate::normalization::photometric_gains_from(
                    existing,
                    crate::image::Placed::whole(incoming),
                    &paired_stars(&panel_stars, landed, tolerance),
                    PANEL_PHOTOMETRY_STARS,
                )
            })
            .unwrap_or_else(|| vec![1.0; incoming.channels])
        });
        let normalization = NormalizationMap::estimate_with_gains(
            existing,
            crate::image::Placed::whole(incoming),
            options.scale().normalization(options.normalization),
            gains.as_deref(),
        )?;
        let (low, high) = normalization.gain_range();
        if low < criteria.minimum_normalization_gain || high > criteria.maximum_normalization_gain {
            return Err(Error::Normalization(
                "panel normalization gain exceeds admission limits".into(),
            ));
        }
        normalization.apply(incoming)
    }

    /// Grow the canvas to `width` by `height`, the present one `dx`, `dy`
    /// from its origin, and fill the samples of `region` only the frame
    /// resampled onto it, `fill`, defines. Returns the samples filled.
    fn commit(
        &mut self,
        (width, height, dx, dy): (usize, usize, usize, usize),
        region: ReferenceRegion,
        fill: Option<&LinearImage>,
    ) -> Result<usize> {
        if (width, height, dx, dy) != (self.image.width, self.image.height, 0, 0) {
            self.grow(width, height, dx, dy)?;
        }
        let channels = self.image.channels;
        let mut filled = 0;
        let Some(incoming) = fill else {
            return Ok(0);
        };
        for row in 0..region.height {
            let start = ((region.y + row) * width + region.x) * channels;
            let target = &mut self.image.data[start..start + region.width * channels];
            let from = row * region.width * channels;
            for (target, value) in target
                .iter_mut()
                .zip(&incoming.data[from..from + region.width * channels])
            {
                if !target.is_finite() && value.is_finite() {
                    *target = *value;
                    filled += 1;
                }
            }
        }
        self.finite += filled;
        Ok(filled)
    }

    /// Register a `width` by `height` frame's stars to the panel the previous
    /// frame joined when it lies mostly within it, or else to the panel from
    /// `first` on that matches the most stars. Returns the registration, the panel, and the fraction of
    /// the frame's stars within it.
    fn register(
        &mut self,
        stars: &[DetectedStar],
        width: usize,
        height: usize,
        first: usize,
        options: &StackOptions,
    ) -> Result<(RegistrationResult, usize, f64)> {
        let mut best: Option<(RegistrationResult, usize, f64)> = None;
        let mut last_error = None;
        let last = Some(self.last_window).filter(|&last| last >= first);
        let others = (first..self.windows.len())
            .rev()
            .filter(|&i| Some(i) != last);
        for index in last.into_iter().chain(others) {
            let result = self.windows[index]
                .register(stars, width, height)
                .and_then(|(result, coverage)| Ok((admissible(result, options)?, coverage)));
            match result {
                // A dither of the previous frame lies within its panel, and
                // that panel settles it. A frame only touching the panel's
                // edge goes on to the others: on a real mosaic, frames of
                // one panel taken after a neighbour's matched that neighbour
                // through a 14% strip by 12 stars, missing their own panel's
                // hundreds, and were placed far off.
                Ok((result, coverage)) if Some(index) == last && coverage >= WARP_COVERAGE => {
                    best = Some((result, index, coverage));
                    break;
                }
                Ok((result, coverage)) => {
                    if best
                        .as_ref()
                        .is_none_or(|(best, _, _)| better(&result, best))
                    {
                        best = Some((result, index, coverage));
                    }
                }
                Err(error) => last_error = Some(error),
            }
        }
        let best = best.ok_or_else(|| {
            last_error.unwrap_or_else(|| Error::Registration("no panel matched".into()))
        })?;
        self.last_window = best.1;
        Ok(best)
    }

    /// Where `source`, its colours sampling the sky at `sites`, lands under
    /// `registration`: the canvas that holds it, its resampled footprint,
    /// and the canvas beneath.
    fn place(
        &self,
        (source, sites): (&LinearImage, Sites),
        registration: &RegistrationResult,
        options: &StackOptions,
        maximum_samples: usize,
    ) -> Result<Placement> {
        let (width, height, dx, dy) =
            union_bounds(&self.image, source, registration.transform, maximum_samples)?;
        let mut transform = registration.transform;
        transform.translation_x += dx as f64;
        transform.translation_y += dy as f64;
        let region = footprint(source, transform, width, height)
            .ok_or_else(|| Error::Registration("panel footprint is empty".into()))?;
        // Most frames join within sky the canvas holds, and their pixels are
        // never wanted: only which samples they cover. Resample only when
        // that cannot be told from the geometry.
        let geometry = crate::registration::FrameGeometry::Similarity(transform);
        let sampling = options.interpolation.into();
        let (incoming, covered) = match crate::registration::resampled_coverage(
            source, region, geometry, sampling, sites,
        )? {
            Some(covered) => (None, covered),
            None => {
                let incoming = crate::registration::resample_region_sited(
                    source,
                    width,
                    height,
                    region,
                    geometry,
                    sampling,
                    sites,
                    &options.gpu,
                )?;
                let covered = incoming.data.iter().map(|v| v.is_finite()).collect();
                (Some(incoming), covered)
            }
        };
        let (common, uncovered) = overlap(&self.image, region, (dx, dy), &covered);
        Ok(Placement {
            width,
            height,
            dx,
            dy,
            transform,
            region,
            incoming,
            common,
            uncovered,
        })
    }

    /// A panel's overlap with a neighbor can be a narrow strip, whose few
    /// stars leave rotation and scale loose; the frame's far edge would then
    /// miss its other neighbors. Refit against every canvas star the frame
    /// covers. `None` unless more stars pair up than with the panel alone.
    fn refine(
        &self,
        registration: &RegistrationResult,
        source: &LinearImage,
        stars: &[DetectedStar],
        options: &StackOptions,
    ) -> Option<RegistrationResult> {
        let region = footprint(
            source,
            registration.transform,
            self.image.width,
            self.image.height,
        )?;
        let covered = crop(&self.image, region, 0, 0).ok()?;
        let mut initial = registration.transform;
        initial.translation_x -= region.x as f64;
        initial.translation_y -= region.y as f64;
        let scale = options.scale();
        let mut refined = crate::registration::refine_similarity(
            &covered,
            source,
            stars,
            initial,
            &scale.registration(&options.registration),
            options.registration.maximum_stars * PLANNING_STARS_FACTOR,
            scale,
            &options.gpu,
        )
        .filter(|refined| refined.matched_stars > registration.matched_stars)?;
        refined = admissible(refined, options).ok()?;
        refined.transform.translation_x += region.x as f64;
        refined.transform.translation_y += region.y as f64;
        Some(refined)
    }

    fn placement(&self, path: &Path) -> Option<SimilarityTransform> {
        Some(
            self.registrations
                .get(&path_identity(path))?
                .result
                .as_ref()
                .ok()?
                .transform,
        )
    }

    /// Measure the lens where the anchor and the frames that started panels
    /// overlap. `None` unless every frame has the same dimensions and the lens
    /// explains what their similarities leave. Returns the fit on the
    /// working grid, and what it measured, in the sensor's pixels.
    ///
    /// On binned frames the lens is measured on the sensor's own pixels: the
    /// anchor and each panel start are prepared again unbinned by `native`,
    /// and their stars detected there, so the lens's pairing radii and the
    /// improvement it must show keep their meaning rather than falling into
    /// the centroid noise of a few binned pixels. The fit then carries back
    /// to the binned grid exactly.
    fn estimate_lens(
        &self,
        native: &dyn Fn(&Path, usize) -> Result<LinearImage>,
        options: &StackOptions,
    ) -> Result<Option<(LensFit, crate::ExpandedLens)>> {
        let (width, height) = self.record.anchor_size;
        if self.record.starts.len() < 2
            || self
                .record
                .joined
                .iter()
                .any(|joined| joined.size != (width, height))
        {
            return Ok(None);
        }
        let Some(anchor) = self.placement(&self.record.starts[0].path) else {
            return Ok(None);
        };
        let to_anchor = SimilarityTransform {
            translation_x: -anchor.translation_x,
            translation_y: -anchor.translation_y,
            ..SimilarityTransform::IDENTITY
        };
        let Some(placements) = self
            .record
            .starts
            .iter()
            .map(|start| Some(self.placement(&start.path)?.then(to_anchor)))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        let focal = focal_length_pixels(&self.record.anchor_headers);
        let bin = options.bin_factor;
        if bin == 1 {
            let frames = self
                .record
                .starts
                .iter()
                .zip(placements)
                .map(|(start, placement)| LensFrame {
                    stars: &start.stars,
                    placement,
                })
                .collect::<Vec<_>>();
            return Ok(
                crate::lens::estimate(&frames, width, height, focal).map(|fit| {
                    let measured = crate::ExpandedLens::new(&fit.distortion, Some(&fit));
                    (fit, measured)
                }),
            );
        }
        // The binned grid spans `bin` sensor pixels per pixel, its focal
        // length and placements with it.
        let factor = f64::from(bin);
        let stars = self
            .record
            .starts
            .iter()
            .map(|start| {
                let image = native(&start.path, self.image.channels)?;
                Ok(crate::registration::detect_stars(
                    &image,
                    &options.registration,
                    LENS_STARS,
                    Scale::NATIVE,
                    &options.gpu,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let frames = stars
            .iter()
            .zip(placements)
            .map(|(stars, placement)| LensFrame {
                stars,
                placement: placement.rescaled(factor),
            })
            .collect::<Vec<_>>();
        let (native_width, native_height) = (width * bin as usize, height * bin as usize);
        Ok(crate::lens::estimate(
            &frames,
            native_width,
            native_height,
            focal.map(|focal| focal * factor),
        )
        .map(|fit| {
            let measured = crate::ExpandedLens::new(&fit.distortion, Some(&fit));
            (fit.rescaled(1.0 / factor), measured)
        }))
    }

    /// Plan the canvas again on the undistorted grids `fit` measured: the
    /// anchor and each panel's first frame placed by their fitted poses, and
    /// every other frame registered again by its undistorted stars. Only the
    /// anchor and the panel starts are read again.
    fn rebuild(
        &self,
        fit: &LensFit,
        first: &Path,
        prepare: &Prepare<'_>,
        options: &StackOptions,
        maximum_samples: usize,
        progress: &mut dyn FnMut(&Path, usize, usize, usize) -> Continue,
    ) -> Result<Plan> {
        let lens = fit.distortion;
        let channels = self.image.channels;
        if progress(first, 0, self.image.width, self.image.height) == Continue::No {
            return Err(Error::Cancelled);
        }
        let (source, sites) = prepare(first, channels)?;
        // The anchor's undistorted footprint sets the canvas's first bounds.
        let edge = boundary(source.width, source.height)
            .into_iter()
            .map(|(x, y)| lens.undistort(x, y))
            .collect::<Vec<_>>();
        let (left, top, right, bottom) = point_bounds(&edge);
        let (width, height) = ((right - left) as usize, (bottom - top) as usize);
        if width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(channels))
            .is_none_or(|n| n > maximum_samples)
        {
            return Err(Error::Stack(format!(
                "expanded canvas {width}x{height} exceeds the limit of {maximum_samples} samples"
            )));
        }
        let transform = SimilarityTransform {
            translation_x: -left,
            translation_y: -top,
            ..SimilarityTransform::IDENTITY
        };
        let projection = PanelProjection {
            width: source.width,
            height: source.height,
            warp: PolynomialWarp::from_similarity(transform),
        };
        let warp = projection.warp.clone().with_distortion(lens);
        let region = ReferenceRegion {
            x: 0,
            y: 0,
            width,
            height,
        };
        let image = crate::registration::resample_region_sited(
            &source,
            width,
            height,
            region,
            crate::registration::FrameGeometry::Warp(&warp),
            options.interpolation.into(),
            sites,
            &options.gpu,
        )?;
        drop(source);
        let anchor = Window::new(
            &image,
            region,
            &options.registration,
            options.scale(),
            &options.gpu,
        )?
        .with_projection(Some(projection))?;
        let mut headers = self.record.anchor_headers.clone();
        // These pixels are undistorted. Translating CRPIX cannot compose the
        // raw frame's WCS (including SIP) with this nonlinear mapping. Leave
        // astrometry unset until the mosaic is plate-solved on its new grid.
        headers.retain(|(key, _)| !crate::fits::preserve_wcs_key(key));
        let stamp = |path: &Path| {
            self.registrations
                .get(&path_identity(path))
                .and_then(|planned| planned.stamp)
        };
        let mut plan = Plan {
            finite: image.data.iter().filter(|v| v.is_finite()).count(),
            registrations: HashMap::from([(
                path_identity(first),
                PlannedRegistration {
                    stamp: stamp(first),
                    starts_panel: true,
                    result: Ok(RegistrationResult {
                        transform,
                        matched_stars: anchor.stars.len(),
                        rms_error_pixels: 0.0,
                        drift_pixels: 0.0,
                        warp: Some(warp),
                    }),
                },
            )]),
            image,
            headers,
            windows: vec![anchor],
            last_window: 0,
            lens: Some(lens),
            origin: (-left, -top),
            record: Record::default(),
        };
        for (index, start) in self.record.starts.iter().enumerate().skip(1) {
            let (width, height) = (plan.image.width, plan.image.height);
            if progress(&start.path, plan.registrations.len(), width, height) == Continue::No {
                return Err(Error::Cancelled);
            }
            let (source, sites) = prepare(&start.path, channels)?;
            plan.place_start(
                (&source, sites),
                start,
                stamp(&start.path),
                fit,
                index,
                options,
                maximum_samples,
            )?;
        }
        // Each panel the starts created, by the start that created it.
        let creators = self
            .record
            .starts
            .iter()
            .enumerate()
            .filter(|(_, start)| start.creates_window)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        for joined in self
            .record
            .joined
            .iter()
            .filter(|joined| !joined.starts_panel)
        {
            let (width, height) = (plan.image.width, plan.image.height);
            if progress(&joined.path, plan.registrations.len(), width, height) == Continue::No {
                return Err(Error::Cancelled);
            }
            let seed = creators
                .get(joined.window)
                .and_then(|&start| self.seed(joined, start, fit, options.scale()));
            let result = match plan.register_through_lens(joined, seed, options, maximum_samples) {
                Ok(result) => Ok(result),
                Err(error) if retryable(&error) => Err(error.to_string()),
                Err(error) => return Err(error),
            };
            plan.registrations.insert(
                path_identity(&joined.path),
                PlannedRegistration {
                    result,
                    stamp: stamp(&joined.path),
                    starts_panel: false,
                },
            );
        }
        // Frames that joined nothing keep their reasons.
        for (path, planned) in &self.registrations {
            if planned.result.is_err() && !plan.registrations.contains_key(path) {
                plan.registrations.insert(
                    path.clone(),
                    PlannedRegistration {
                        result: planned.result.clone(),
                        stamp: planned.stamp,
                        starts_panel: false,
                    },
                );
            }
        }
        Ok(plan)
    }

    /// Where a frame that joined the panel `start` began lands on the
    /// anchor's undistorted grid: through the similarities planning placed
    /// it and that panel's first frame by, onto that frame's fitted pose.
    /// Within a panel the two frames' distortion all but cancels, so this
    /// lies within a pixel or so, for registration to refine without
    /// searching.
    fn seed(
        &self,
        joined: &Joined,
        start: usize,
        fit: &LensFit,
        scale: Scale,
    ) -> Option<RegistrationResult> {
        let lens = fit.distortion;
        let pose = fit.poses[start];
        let placed = self.placement(&joined.path)?;
        let first = self.placement(&self.record.starts[start].path)?;
        let (width, height) = joined.size;
        let mut from = Vec::new();
        let mut to = Vec::new();
        for row in 0..=2 * GEOMETRY_GRID.1 {
            for column in 0..=2 * GEOMETRY_GRID.0 {
                let x = (width - 1) as f64 * column as f64 / (2 * GEOMETRY_GRID.0) as f64;
                let y = (height - 1) as f64 * row as f64 / (2 * GEOMETRY_GRID.1) as f64;
                let (canvas_x, canvas_y) = placed.apply(x, y);
                let (start_x, start_y) = first.inverse_apply(canvas_x, canvas_y);
                let (u, v) = lens.undistort(start_x, start_y);
                to.push(pose.apply(fit.projection, u, v)?);
                from.push(lens.undistort(x, y));
            }
        }
        let transform = crate::lens::fit_similarity(&from, &to)?;
        let (left, top, right, bottom) = point_bounds(&to);
        let mut grid = Vec::new();
        let mut validation = Vec::new();
        for (index, (&source, &(x, y))) in from.iter().zip(&to).enumerate() {
            let pair = ((x - left, y - top), source);
            let row = index / (2 * GEOMETRY_GRID.0 + 1);
            let col = index % (2 * GEOMETRY_GRID.0 + 1);
            if row.is_multiple_of(2) && col.is_multiple_of(2) {
                grid.push(pair);
            } else {
                validation.push(pair);
            }
        }
        let warp = fit_panel_geometry(
            (right - left) as usize,
            (bottom - top) as usize,
            &grid,
            &validation,
            scale,
            SEED_GEOMETRY_ERROR_PIXELS,
        )
        .ok()?;
        Some(RegistrationResult {
            transform,
            warp: Some(warp.translated(left, top)),
            matched_stars: 0,
            rms_error_pixels: 0.0,
            drift_pixels: 0.0,
        })
    }

    /// Place a frame that started a panel by its fitted pose: grow the canvas
    /// to its undistorted footprint, match and fill the sky it newly covers,
    /// and start its panel where it started one before.
    #[allow(clippy::too_many_arguments)]
    fn place_start(
        &mut self,
        (source, sites): (&LinearImage, Sites),
        start: &Start,
        stamp: Option<crate::replay::SourceStamp>,
        fit: &LensFit,
        index: usize,
        options: &StackOptions,
        maximum_samples: usize,
    ) -> Result<()> {
        let lens = fit.distortion;
        let pose = fit.poses[index];
        let lost = || Error::Registration("a panel start turns away from the anchor's view".into());
        let to_anchor = |x: f64, y: f64| {
            let (u, v) = lens.undistort(x, y);
            pose.apply(fit.projection, u, v).ok_or_else(lost)
        };
        let edge = boundary(source.width, source.height)
            .into_iter()
            .map(|(x, y)| {
                let (a, b) = to_anchor(x, y)?;
                Ok((a + self.origin.0, b + self.origin.1))
            })
            .collect::<Result<Vec<_>>>()?;
        let (left, top, right, bottom) = point_bounds(&edge);
        let (width, height, dx, dy) =
            grown_bounds(&self.image, (left, top, right, bottom), maximum_samples)?;
        let (dx_f, dy_f) = (dx as f64, dy as f64);
        let origin = (self.origin.0 + dx_f, self.origin.1 + dy_f);
        let region = region_within(
            (left + dx_f, top + dy_f, right + dx_f, bottom + dy_f),
            width,
            height,
        )
        .ok_or_else(|| Error::Registration("panel footprint is empty".into()))?;
        let (region_x, region_y) = (region.x as f64, region.y as f64);
        // Approximate the exact sky projection on a grid, checking the
        // interleaved grid independently. Wide mosaics may need a cubic;
        // the ordinary star-fitted registration model remains unchanged.
        let mut grid = Vec::new();
        let mut validation = Vec::new();
        for row in 0..=2 * GEOMETRY_GRID.1 {
            for column in 0..=2 * GEOMETRY_GRID.0 {
                let x = (source.width - 1) as f64 * column as f64 / (2 * GEOMETRY_GRID.0) as f64;
                let y = (source.height - 1) as f64 * row as f64 / (2 * GEOMETRY_GRID.1) as f64;
                let (u, v) = lens.undistort(x, y);
                let (a, b) = pose.apply(fit.projection, u, v).ok_or_else(lost)?;
                let pair = ((a + origin.0 - region_x, b + origin.1 - region_y), (u, v));
                if row % 2 == 0 && column % 2 == 0 {
                    grid.push(pair);
                } else {
                    validation.push(pair);
                }
            }
        }
        let polynomial = fit_panel_geometry(
            region.width,
            region.height,
            &grid,
            &validation,
            options.scale(),
            MAXIMUM_GEOMETRY_ERROR_PIXELS,
        )?;
        let warp = polynomial
            .translated(region_x, region_y)
            .with_distortion(lens);
        let (ideal, placed): (Vec<_>, Vec<_>) = grid
            .iter()
            .map(|&((x, y), ideal)| (ideal, (x + region_x, y + region_y)))
            .unzip();
        let transform = crate::lens::fit_similarity(&ideal, &placed).ok_or_else(lost)?;
        let mut incoming = crate::registration::resample_region_sited(
            source,
            width,
            height,
            region,
            crate::registration::FrameGeometry::Warp(&warp),
            options.interpolation.into(),
            sites,
            &options.gpu,
        )?;
        let existing = crop(&self.image, region, dx, dy)?;
        let (mut common, mut uncovered) = (0, 0);
        for (a, b) in existing.data.iter().zip(&incoming.data) {
            if b.is_finite() {
                if a.is_finite() {
                    common += 1;
                } else {
                    uncovered += 1;
                }
            }
        }
        if common == 0 {
            return Err(Error::Registration(
                "a panel start no longer overlaps the canvas".into(),
            ));
        }
        if uncovered > 0 {
            let landed = lens
                .undistort_stars(&start.stars)
                .iter()
                .filter_map(|star| pose.apply(fit.projection, star.x, star.y))
                .map(|(a, b)| (a + origin.0 - region_x, b + origin.1 - region_y))
                .collect::<Vec<_>>();
            self.match_panel(
                &mut incoming,
                &existing,
                region,
                (dx, dy),
                &landed,
                start.window,
                options,
            )?;
        }
        self.commit((width, height, dx, dy), region, Some(&incoming))?;
        if start.creates_window {
            self.windows.push(
                Window::from_pixels(
                    &incoming,
                    region,
                    &options.registration,
                    options.scale(),
                    &options.gpu,
                )?
                .with_projection(Some(PanelProjection {
                    width: source.width,
                    height: source.height,
                    warp: polynomial.translated(region_x, region_y),
                }))?,
            );
        }
        self.registrations.insert(
            path_identity(&start.path),
            PlannedRegistration {
                stamp,
                starts_panel: true,
                result: Ok(RegistrationResult {
                    transform,
                    matched_stars: fit.pairs[index],
                    rms_error_pixels: fit.rms[index],
                    drift_pixels: 0.0,
                    warp: Some(warp),
                }),
            },
        );
        Ok(())
    }

    /// Register a frame that joined a panel before by its undistorted stars,
    /// refining `seed`, its expected place on the anchor's undistorted grid,
    /// against that panel, or else searching every panel; then grow the
    /// canvas to it.
    fn register_through_lens(
        &mut self,
        joined: &Joined,
        seed: Option<RegistrationResult>,
        options: &StackOptions,
        maximum_samples: usize,
    ) -> Result<RegistrationResult> {
        let lens = self
            .lens
            .ok_or_else(|| Error::Stack("no lens was measured".into()))?;
        let stars = lens.undistort_stars(&joined.stars);
        let (width, height) = joined.size;
        let window = joined.window.min(self.windows.len() - 1);
        let origin = SimilarityTransform {
            translation_x: self.origin.0,
            translation_y: self.origin.1,
            ..SimilarityTransform::IDENTITY
        };
        let refined = if let Some(mut seed) = seed {
            seed.transform = seed.transform.then(origin);
            seed.warp = seed
                .warp
                .map(|warp| warp.translated(self.origin.0, self.origin.1));
            let (result, window) =
                self.register_seeded(seed, &stars, (width, height), &lens).ok_or_else(|| {
                    Error::Registration("projected registration has too few reliable stars to refine; frame was not integrated".into())
                })?;
            Some((admissible(result, options)?, window, 1.0))
        } else {
            None
        };
        let (mut registration, window, coverage) = match refined {
            Some(refined) => refined,
            None => {
                self.last_window = window;
                self.register(&stars, width, height, 0, options)?
            }
        };
        let edge = boundary(width, height)
            .into_iter()
            .map(|(x, y)| {
                let (u, v) = lens.undistort(x, y);
                registration.transform.apply(u, v)
            })
            .collect::<Vec<_>>();
        let (canvas_width, canvas_height, dx, dy) =
            grown_bounds(&self.image, point_bounds(&edge), maximum_samples)?;
        if (canvas_width, canvas_height, dx, dy) != (self.image.width, self.image.height, 0, 0) {
            self.grow(canvas_width, canvas_height, dx, dy)?;
        }
        registration.transform.translation_x += dx as f64;
        registration.transform.translation_y += dy as f64;
        registration.warp = registration
            .warp
            .map(|warp| warp.translated(dx as f64, dy as f64));
        if registration.warp.is_none() && coverage >= WARP_COVERAGE {
            self.windows[window].warp(&mut registration, &stars);
        }
        Ok(through_lens(registration, Some(lens)))
    }

    /// Register a joined frame's undistorted `stars` from the `seed` its
    /// first placement and its panel start's pose predict, in place of a
    /// second blind match. The seed pairs stars on every panel, and the
    /// panel pairing the most takes the frame: a frame's own panel may share
    /// only a strip of it with the frame. The similarity then follows that
    /// panel's star-fitted warp across the `size` frame, where the seed's
    /// came from a pose fitted before the lens and can sit pixels off, and the
    /// warp is fitted through it as an unseeded registration's is.
    fn register_seeded(
        &self,
        seed: RegistrationResult,
        stars: &[DetectedStar],
        (width, height): (usize, usize),
        lens: &LensDistortion,
    ) -> Option<(RegistrationResult, usize)> {
        let (mut result, window) = self
            .windows
            .iter()
            .enumerate()
            .filter_map(|(index, window)| Some((window.refine_warp(seed.clone(), stars)?, index)))
            .max_by_key(|(result, _)| result.matched_stars)?;
        if let Some(warp) = &result.warp {
            let (from, to): (Vec<_>, Vec<_>) = (0..=GEOMETRY_GRID.1)
                .flat_map(|row| (0..=GEOMETRY_GRID.0).map(move |column| (row, column)))
                .map(|(row, column)| {
                    let (u, v) = lens.undistort(
                        (width - 1) as f64 * column as f64 / GEOMETRY_GRID.0 as f64,
                        (height - 1) as f64 * row as f64 / GEOMETRY_GRID.1 as f64,
                    );
                    let (x, y) = result.transform.apply(u, v);
                    (warp.apply(x, y), (x, y))
                })
                .unzip();
            if let Some(transform) = crate::lens::fit_similarity(&from, &to) {
                result.transform = transform;
            }
        }
        let mut refitted = RegistrationResult {
            warp: None,
            ..result.clone()
        };
        self.windows[window].warp(&mut refitted, stars);
        Some((
            if refitted.warp.is_some() {
                refitted
            } else {
                result
            },
            window,
        ))
    }

    /// Move the canvas into a larger one, `dx` and `dy` from its origin.
    fn grow(&mut self, width: usize, height: usize, dx: usize, dy: usize) -> Result<()> {
        let channels = self.image.channels;
        let mut padded = blank(width, height, channels)?;
        let row = self.image.width * channels;
        for y in 0..self.image.height {
            let to = ((y + dy) * width + dx) * channels;
            padded.data[to..to + row].copy_from_slice(&self.image.data[y * row..(y + 1) * row]);
        }
        self.image = padded;
        for window in &mut self.windows {
            window.region.x += dx;
            window.region.y += dy;
            if let Some(projection) = &mut window.projection {
                projection.warp = projection.warp.translated(dx as f64, dy as f64);
            }
        }
        self.origin = (self.origin.0 + dx as f64, self.origin.1 + dy as f64);
        for registered in self
            .registrations
            .values_mut()
            .filter_map(|planned| planned.result.as_mut().ok())
        {
            registered.transform.translation_x += dx as f64;
            registered.transform.translation_y += dy as f64;
            if let Some(warp) = &mut registered.warp {
                *warp = warp.translated(dx as f64, dy as f64);
            }
        }
        shift_reference_pixel(&mut self.headers, dx as f64, dy as f64);
        Ok(())
    }
}

/// Move a WCS reference pixel by `dx`, `dy`, as the canvas moves under it.
fn shift_reference_pixel(headers: &mut [(String, HeaderValue)], dx: f64, dy: f64) {
    for (key, value) in headers {
        let delta = match key.as_str() {
            "CRPIX1" => dx,
            "CRPIX2" => dy,
            _ => continue,
        };
        if let Some(pixel) = value.as_f64().filter(|v| v.is_finite()) {
            *value = HeaderValue::Float(pixel + delta);
        }
    }
}

/// The focal length in pixels a frame's headers give: FOCALLEN in
/// millimetres over XPIXSZ in micrometres, which capture programs record
/// with any binning applied.
fn focal_length_pixels(headers: &[(String, HeaderValue)]) -> Option<f64> {
    let value = |name: &str| {
        headers
            .iter()
            .find(|(key, _)| key == name)
            .and_then(|(_, value)| value.as_f64())
            .filter(|value| value.is_finite() && *value > 0.0)
    };
    Some(value("FOCALLEN")? * 1000.0 / value("XPIXSZ")?)
}

/// Points around the edge of a `width` by `height` frame, at most 64
/// pixels apart, corners included.
fn boundary(width: usize, height: usize) -> Vec<(f64, f64)> {
    let (right, bottom) = (
        width.saturating_sub(1) as f64,
        height.saturating_sub(1) as f64,
    );
    let steps = |length: f64| ((length / 64.0).ceil() as usize).max(1);
    let mut points = Vec::new();
    let across = steps(right);
    for step in 0..=across {
        let x = right * step as f64 / across as f64;
        points.extend([(x, 0.0), (x, bottom)]);
    }
    let down = steps(bottom);
    for step in 1..down {
        let y = bottom * step as f64 / down as f64;
        points.extend([(0.0, y), (right, y)]);
    }
    points
}

/// The pixel bounds (left, top, right, bottom) `points` touch, as
/// [`corner_bounds`] gives a frame's.
fn point_bounds(points: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    let (mut left, mut top, mut right, mut bottom) = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for &(x, y) in points {
        left = left.min(x);
        top = top.min(y);
        right = right.max(x);
        bottom = bottom.max(y);
    }
    (
        left.floor(),
        top.floor(),
        right.ceil() + 1.0,
        bottom.ceil() + 1.0,
    )
}

/// Keep the simplest approximation meeting the same accuracy bound on both
/// the fit nodes and independent nodes between them: in working pixels,
/// binned at `scale`, `limit` of the sensor's.
fn fit_panel_geometry(
    width: usize,
    height: usize,
    grid: &[crate::registration::PointPair],
    validation: &[crate::registration::PointPair],
    scale: Scale,
    limit: f64,
) -> Result<PolynomialWarp> {
    let limit = scale.length(limit);
    let mut worst = f64::INFINITY;
    for order in [2, 3] {
        let Some(warp) = PolynomialWarp::fit(order, width, height, grid) else {
            continue;
        };
        worst = grid
            .iter()
            .chain(validation)
            .fold(0.0_f64, |worst, &((x, y), (u, v))| {
                let (a, b) = warp.apply(x, y);
                let error = (a - u).hypot(b - v);
                if error.is_finite() {
                    worst.max(error)
                } else {
                    f64::INFINITY
                }
            });
        if worst <= limit {
            return Ok(warp);
        }
    }
    Err(Error::Registration(format!(
        "a panel start's sky projection cannot be approximated within {limit}px (worst {worst:.3}px)"
    )))
}

/// Only matching/admission failures may defer a frame or abandon a lens fit.
fn retryable(error: &Error) -> bool {
    matches!(error, Error::Registration(_) | Error::Normalization(_))
}

fn admissible(
    registration: RegistrationResult,
    options: &StackOptions,
) -> Result<RegistrationResult> {
    let criteria = options.acceptance;
    if options.scale().native_length(registration.rms_error_pixels)
        > criteria.maximum_registration_rms_pixels
        || (registration.transform.scale - 1.0).abs() > criteria.maximum_scale_deviation
        || crate::stack::rotation_deviation_degrees(registration.transform.rotation_radians)
            > criteria.maximum_rotation_degrees
    {
        return Err(Error::Registration(
            "panel transform exceeds registration admission limits".into(),
        ));
    }
    Ok(registration)
}

/// Corners of `source` under `transform`, as (left, top, right, bottom)
/// bounds of the pixels they touch.
fn corner_bounds(source: &LinearImage, transform: SimilarityTransform) -> (f64, f64, f64, f64) {
    let snap = |v: f64| {
        if (v - v.round()).abs() < 1.0e-6 {
            v.round()
        } else {
            v
        }
    };
    let mut bounds = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for (x, y) in [
        (0.0, 0.0),
        ((source.width - 1) as f64, 0.0),
        (0.0, (source.height - 1) as f64),
        ((source.width - 1) as f64, (source.height - 1) as f64),
    ] {
        let (x, y) = transform.apply(x, y);
        bounds.0 = bounds.0.min(snap(x).floor());
        bounds.1 = bounds.1.min(snap(y).floor());
        bounds.2 = bounds.2.max(snap(x).ceil() + 1.0);
        bounds.3 = bounds.3.max(snap(y).ceil() + 1.0);
    }
    bounds
}

/// The canvas size holding both `reference` and `source`, and where the
/// reference's origin lands on it.
fn union_bounds(
    reference: &LinearImage,
    source: &LinearImage,
    transform: SimilarityTransform,
    maximum_samples: usize,
) -> Result<(usize, usize, usize, usize)> {
    grown_bounds(reference, corner_bounds(source, transform), maximum_samples)
}

/// The canvas size holding both `reference` and pixel `bounds`, and where
/// the reference's origin lands on it.
fn grown_bounds(
    reference: &LinearImage,
    (left, top, right, bottom): (f64, f64, f64, f64),
    maximum_samples: usize,
) -> Result<(usize, usize, usize, usize)> {
    let (left, top) = (left.min(0.0), top.min(0.0));
    let (right, bottom) = (
        right.max(reference.width as f64),
        bottom.max(reference.height as f64),
    );
    let (w, h) = (right - left, bottom - top);
    if !w.is_finite()
        || !h.is_finite()
        || w <= 0.0
        || h <= 0.0
        || w * h * reference.channels as f64 > maximum_samples as f64
    {
        return Err(Error::Stack(format!(
            "expanded canvas {w}x{h} exceeds the limit of {maximum_samples} samples"
        )));
    }
    Ok((w as usize, h as usize, (-left) as usize, (-top) as usize))
}

/// The canvas pixels `source` may cover under `transform`.
fn footprint(
    source: &LinearImage,
    transform: SimilarityTransform,
    width: usize,
    height: usize,
) -> Option<ReferenceRegion> {
    region_within(corner_bounds(source, transform), width, height)
}

/// The part of a `width` by `height` canvas within pixel `bounds`.
fn region_within(
    (left, top, right, bottom): (f64, f64, f64, f64),
    width: usize,
    height: usize,
) -> Option<ReferenceRegion> {
    let x = left.max(0.0) as usize;
    let y = top.max(0.0) as usize;
    let right = (right.max(0.0) as usize).min(width);
    let bottom = (bottom.max(0.0) as usize).min(height);
    (right > x && bottom > y).then_some(ReferenceRegion {
        x,
        y,
        width: right - x,
        height: bottom - y,
    })
}

/// The samples `covered` marks on `region` of a canvas about to move `dx`
/// and `dy` from its origin, as [`crop`] would hold them: those the canvas
/// also defines, and those it does not.
fn overlap(
    canvas: &LinearImage,
    region: ReferenceRegion,
    (dx, dy): (usize, usize),
    covered: &[bool],
) -> (usize, usize) {
    let channels = canvas.channels;
    let row_samples = region.width * channels;
    covered
        .par_chunks(row_samples)
        .enumerate()
        .map(|(row, covered)| {
            let defined = |x: usize| {
                let column = region.x + x / channels;
                (region.y + row)
                    .checked_sub(dy)
                    .filter(|&y| y < canvas.height)
                    .zip(column.checked_sub(dx).filter(|&x| x < canvas.width))
                    .is_some_and(|(y, column)| {
                        canvas.data[(y * canvas.width + column) * channels + x % channels]
                            .is_finite()
                    })
            };
            let common = covered
                .iter()
                .enumerate()
                .filter(|&(x, &covered)| covered && defined(x))
                .count();
            (
                common,
                covered.iter().filter(|&&covered| covered).count() - common,
            )
        })
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
}

/// `region` of a canvas about to move `dx` and `dy` from its origin: the
/// region is on the moved grid, and samples off the present canvas are `NaN`.
fn crop(
    canvas: &LinearImage,
    region: ReferenceRegion,
    dx: usize,
    dy: usize,
) -> Result<LinearImage> {
    let channels = canvas.channels;
    let mut output = blank(region.width, region.height, channels)?;
    for row in 0..region.height {
        let Some(y) = (region.y + row)
            .checked_sub(dy)
            .filter(|y| *y < canvas.height)
        else {
            continue;
        };
        let start = region.x.max(dx);
        let end = (region.x + region.width).min(canvas.width + dx);
        if end <= start {
            continue;
        }
        let from = (y * canvas.width + start - dx) * channels;
        let to = (row * region.width + start - region.x) * channels;
        let count = (end - start) * channels;
        output.data[to..to + count].copy_from_slice(&canvas.data[from..from + count]);
    }
    Ok(output)
}

fn blank(width: usize, height: usize, channels: usize) -> Result<LinearImage> {
    let count = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(channels))
        .ok_or_else(|| Error::Stack("expanded canvas dimensions overflow".into()))?;
    let mut data = Vec::new();
    data.try_reserve_exact(count)
        .map_err(|e| Error::Stack(format!("cannot allocate expanded canvas: {e}")))?;
    data.resize(count, f32::NAN);
    LinearImage::new(width, height, channels, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn projected_window(count: usize) -> (Window, Vec<DetectedStar>, SimilarityTransform) {
        let mut state = 191_u64;
        let mut random = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as f64 / (1_u64 << 31) as f64
        };
        let stars = (0..count)
            .map(|_| DetectedStar {
                x: 40.0 + 880.0 * random(),
                y: 40.0 + 880.0 * random(),
                flux: 1.0,
                peak: 1.0,
                area: 3,
            })
            .collect::<Vec<_>>();
        let geometry = |x: f64, y: f64| (x + 0.00003 * x * y + 1e-8 * x * x * y, y + 0.01 * x);
        let pairs = (0..9)
            .flat_map(|row| {
                (0..9).map(move |column| {
                    let (x, y) = (column as f64 * 120.0, row as f64 * 120.0);
                    ((x, y), geometry(x, y))
                })
            })
            .collect::<Vec<_>>();
        let region = ReferenceRegion {
            x: 1500,
            y: 1800,
            width: 1000,
            height: 1000,
        };
        let projection = PanelProjection {
            width: 1000,
            height: 1000,
            warp: PolynomialWarp::fit(3, 1000, 1000, &pairs)
                .unwrap()
                .translated(region.x as f64, region.y as f64),
        };
        let dither = SimilarityTransform {
            scale: 1.001,
            rotation_radians: 0.004,
            translation_x: 3.2,
            translation_y: -2.1,
        };
        let source = stars
            .iter()
            .map(|star| {
                let (x, y) = geometry(star.x, star.y);
                let (x, y) = dither.apply(x, y);
                DetectedStar { x, y, ..*star }
            })
            .collect();
        let window = Window::from_stars(
            region,
            stars,
            Window::options(region, &RegistrationOptions::default(), Scale::NATIVE),
            Scale::NATIVE,
            &crate::GpuOptions::default(),
        )
        .unwrap()
        .with_projection(Some(projection))
        .unwrap();
        (window, source, dither)
    }

    #[test]
    fn projected_default_registration_retains_geometry_for_unplanned_exposures() {
        let (window, source, dither) = projected_window(600);
        let (registered, coverage) = window.register(&source, 1000, 1000).unwrap();
        assert!(coverage > 0.95);
        assert!(registered.matched_stars > 550);
        assert!(registered.rms_error_pixels < 1e-8);
        let fitted = registered.warp.unwrap();
        assert_eq!(fitted.order(), 3);
        let projection = window.projection.as_ref().unwrap();
        for (x, y) in [(1501.2, 1803.4), (2438.1, 2737.3), (2017.4, 2312.8)] {
            let (u, v) = projection.warp.apply(x, y);
            let expected = dither.apply(u, v);
            let actual = fitted.apply(x, y);
            assert!((actual.0 - expected.0).hypot(actual.1 - expected.1) < 1e-8);
        }
    }

    #[test]
    fn projected_default_registration_retains_geometry_across_a_meridian_flip() {
        let (window, source, dither) = projected_window(600);
        // Pixel centres span 0..999, so a half-turn about the sensor centre
        // sends (x, y) to (999 - x, 999 - y) in its undistorted plane.
        let flip = SimilarityTransform {
            rotation_radians: std::f64::consts::PI,
            translation_x: 999.0,
            translation_y: 999.0,
            ..SimilarityTransform::IDENTITY
        };
        let flipped = source
            .iter()
            .map(|star| {
                let (x, y) = flip.apply(star.x, star.y);
                DetectedStar { x, y, ..*star }
            })
            .collect::<Vec<_>>();
        let (registered, coverage) = window.register(&flipped, 1000, 1000).unwrap();
        assert!(coverage > 0.95);
        assert!(registered.matched_stars > 550);
        assert!(registered.rms_error_pixels < 1e-8);
        let fitted = registered.warp.unwrap();
        assert_eq!(fitted.order(), 3);
        let projection = window.projection.as_ref().unwrap();
        let source_transform = dither.then(flip);
        for (x, y) in [(1501.2, 1803.4), (2438.1, 2737.3), (2017.4, 2312.8)] {
            let (u, v) = projection.warp.apply(x, y);
            let expected = source_transform.apply(u, v);
            let actual = fitted.apply(x, y);
            assert!((actual.0 - expected.0).hypot(actual.1 - expected.1) < 1e-8);
        }
    }

    #[test]
    fn projected_registration_never_downgrades_when_too_few_stars_constrain_the_warp() {
        let (window, source, _) = projected_window(20);
        assert!(
            window
                .creator
                .as_ref()
                .unwrap()
                .register_similarity_stars(&source)
                .is_ok()
        );
        let error = window
            .register(&source, 1000, 1000)
            .unwrap_err()
            .to_string();
        assert!(error.contains("too few reliable stars"), "{error}");
        let mut plan = Plan {
            image: blank(2500, 2800, 1).unwrap(),
            finite: 0,
            registrations: HashMap::new(),
            headers: Vec::new(),
            windows: vec![window.clone()],
            last_window: 0,
            lens: Some(LensDistortion::none(1000, 1000)),
            origin: (0.0, 0.0),
            record: Record::default(),
        };
        let joined = Joined {
            path: "sparse.fits".into(),
            stars: source,
            size: (1000, 1000),
            window: 0,
            starts_panel: false,
        };
        let seed = RegistrationResult {
            transform: SimilarityTransform::IDENTITY,
            warp: Some(window.projection.unwrap().warp),
            matched_stars: 20,
            rms_error_pixels: 0.0,
            drift_pixels: 0.0,
        };
        let error = plan
            .register_through_lens(&joined, Some(seed), &StackOptions::default(), usize::MAX)
            .unwrap_err()
            .to_string();
        assert!(error.contains("too few reliable stars"), "{error}");
    }

    #[test]
    fn projected_registration_rejects_sparse_sensor_fixed_matches() {
        let (width, height) = (4000, 3000);
        let mut state = 1517_u64;
        let mut random = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as f64 / (1_u64 << 31) as f64
        };
        let mut reference = Vec::<DetectedStar>::new();
        while reference.len() < 800 {
            let (x, y) = (40.0 + 3920.0 * random(), 40.0 + 2920.0 * random());
            if reference
                .iter()
                .all(|star| (star.x - x).hypot(star.y - y) >= 8.0)
            {
                reference.push(DetectedStar {
                    x,
                    y,
                    flux: if reference.len() < 20 { 2.0 } else { 1.0 },
                    peak: 1.0,
                    area: 3,
                });
            }
        }
        // Twenty bright detections are fixed to the sensor, not the sky.
        // Every other source star belongs to an unrelated field, with no
        // accidental correspondence near the identity transform.
        let mut source = reference[..20].to_vec();
        while source.len() < 800 {
            let (x, y) = (40.0 + 3920.0 * random(), 40.0 + 2920.0 * random());
            if reference
                .iter()
                .chain(&source)
                .all(|star| (star.x - x).hypot(star.y - y) >= 8.0)
            {
                source.push(DetectedStar {
                    x,
                    y,
                    flux: 1.0,
                    peak: 1.0,
                    area: 3,
                });
            }
        }
        let region = ReferenceRegion {
            x: 0,
            y: 0,
            width,
            height,
        };
        let window = Window::from_stars(
            region,
            reference,
            Window::options(region, &RegistrationOptions::default(), Scale::NATIVE),
            Scale::NATIVE,
            &crate::GpuOptions::default(),
        )
        .unwrap()
        .with_projection(Some(PanelProjection {
            width,
            height,
            warp: PolynomialWarp::from_similarity(SimilarityTransform::IDENTITY),
        }))
        .unwrap();
        let found = window
            .creator
            .as_ref()
            .unwrap()
            .register_similarity_stars(&source)
            .unwrap();
        assert_eq!(found.matched_stars, 20);
        let seed = RegistrationResult {
            warp: Some(PolynomialWarp::from_similarity(found.transform)),
            ..found
        };
        // An affine warp requires only fifteen pairs; a tiny fitted RMS
        // therefore does not establish that these images show the same sky.
        let fitted = window.refine_warp(seed, &source).unwrap();
        assert_eq!(fitted.matched_stars, 20);
        assert!(fitted.rms_error_pixels < 1e-8);
        let error = window
            .register(&source, width, height)
            .unwrap_err()
            .to_string();
        assert!(error.contains("only 20 of the 800 stars"), "{error}");
    }

    #[test]
    fn legacy_lens_canvas_requires_replanning_before_adding_frames() {
        let (mut window, _, _) = projected_window(100);
        window.projection = None;
        window.creator = None;
        let image = blank(2500, 2800, 1).unwrap();
        let canvas = CanvasRegistrar::from_windows(
            &image,
            vec![window],
            Some(LensDistortion::none(1000, 1000)),
        );
        assert!(canvas.panel_projections().is_empty());
        let error = canvas
            .register(&blank(1000, 1000, 1).unwrap())
            .unwrap_err()
            .to_string();
        assert!(error.contains("rebuild the expanded reference"), "{error}");
    }

    #[test]
    fn wide_panel_projection_uses_a_validated_cubic() {
        // ASI2600 / 250 mm geometry, with a panel well away from the
        // anchor's tangent point. A quadratic exceeds the 0.02 px budget.
        let (width, height) = (6248, 4176);
        let focal = 250_000.0 / 3.76;
        let angle = (15_000.0_f64 / focal).atan();
        let (s, c) = angle.sin_cos();
        let projection = crate::lens::Projection {
            focal,
            center_x: width as f64 / 2.0,
            center_y: height as f64 / 2.0,
        };
        let pose = crate::lens::Pose::Rotation {
            rotation: [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
            focal,
        };
        let mut grid = Vec::new();
        let mut validation = Vec::new();
        for row in 0..=32 {
            for col in 0..=48 {
                let source = (
                    (width - 1) as f64 * col as f64 / 48.0,
                    (height - 1) as f64 * row as f64 / 32.0,
                );
                let (x, y) = pose.apply(Some(projection), source.0, source.1).unwrap();
                let pair = ((x - 15_000.0, y), source);
                if row % 2 == 0 && col % 2 == 0 {
                    grid.push(pair);
                } else {
                    validation.push(pair);
                }
            }
        }
        let quadratic = PolynomialWarp::fit(2, width, height, &grid).unwrap();
        assert!(validation.iter().any(|&((x, y), (u, v))| {
            let (a, b) = quadratic.apply(x, y);
            (a - u).hypot(b - v) > MAXIMUM_GEOMETRY_ERROR_PIXELS
        }));
        let warp = fit_panel_geometry(
            width,
            height,
            &grid,
            &validation,
            Scale::NATIVE,
            MAXIMUM_GEOMETRY_ERROR_PIXELS,
        )
        .unwrap();
        assert_eq!(warp.order(), 3);
        warp.validate().unwrap();
        // Saved warps, including the compact context payload, must replay
        // the same map rather than silently losing the higher-order terms.
        let json: PolynomialWarp =
            serde_json::from_str(&serde_json::to_string(&warp).unwrap()).unwrap();
        let compact: PolynomialWarp =
            postcard::from_bytes(&postcard::to_stdvec(&warp).unwrap()).unwrap();
        assert_eq!(json, warp);
        assert_eq!(compact, warp);
        let moved = warp.translated(1300.0, 2700.0);
        for &((x, y), (u, v)) in &validation {
            let (a, b) = moved.apply(x + 1300.0, y + 2700.0);
            assert!((a - u).hypot(b - v) < MAXIMUM_GEOMETRY_ERROR_PIXELS);
        }
        // Independent checks must also be allowed to veto the fit.
        let mut inconsistent = validation.clone();
        inconsistent[0].1.0 += 1.0;
        assert!(
            fit_panel_geometry(
                width,
                height,
                &grid,
                &inconsistent,
                Scale::NATIVE,
                MAXIMUM_GEOMETRY_ERROR_PIXELS,
            )
            .is_err()
        );
    }

    fn optical_frame(camera: &str, filter: &str, date: &str) -> FitsFrame {
        FitsFrame {
            image: blank(100, 80, 1).unwrap(),
            headers: vec![
                ("INSTRUME".into(), HeaderValue::String(camera.into())),
                ("XBINNING".into(), HeaderValue::Integer(1)),
                ("YBINNING".into(), HeaderValue::Integer(1)),
                ("XPIXSZ".into(), HeaderValue::Float(3.76)),
                ("YPIXSZ".into(), HeaderValue::Float(3.76)),
                ("FOCALLEN".into(), HeaderValue::Float(528.0)),
                ("FILTER".into(), HeaderValue::String(filter.into())),
                ("DATE-OBS".into(), HeaderValue::String(date.into())),
            ],
            exposure_seconds: None,
            bayer: None,
            source: None,
            bounds: None,
        }
    }

    #[test]
    fn optical_check_allows_filters_dates_and_rounded_measurements() {
        let a = optical_frame(" Camera A ", "S", "2025-01-01");
        let mut b = optical_frame("camera a", "O", "2025-02-01");
        for (key, value) in &mut b.headers {
            if key == "FOCALLEN" {
                *value = HeaderValue::Float(530.0);
            }
            if key == "XPIXSZ" {
                *value = HeaderValue::Float(3.77);
            }
        }
        let result = optical_compatibility(&[
            ("a".into(), OpticalMetadata::from_frame(&a)),
            ("b".into(), OpticalMetadata::from_frame(&b)),
        ]);
        assert_eq!(result, (true, Vec::new()));
    }

    #[test]
    fn optical_check_finds_conflicts_even_with_missing_anchor_metadata() {
        let a = optical_frame("Camera A", "S", "2025-01-01");
        let mut missing = a.clone();
        missing.headers.clear();
        let allowed =
            optical_compatibility(&[("missing".into(), OpticalMetadata::from_frame(&missing))]);
        assert!(allowed.0);
        assert!(allowed.1[0].contains("could not be verified"));
        for key in [
            "INSTRUME", "XBINNING", "YBINNING", "XPIXSZ", "YPIXSZ", "FOCALLEN",
        ] {
            let mut b = a.clone();
            let value = &mut b.headers.iter_mut().find(|(k, _)| k == key).unwrap().1;
            *value = if key == "INSTRUME" {
                HeaderValue::String("Camera B".into())
            } else if key.ends_with("BINNING") {
                HeaderValue::Integer(2)
            } else {
                HeaderValue::Float(10.0)
            };
            let (compatible, notes) = optical_compatibility(&[
                ("missing".into(), OpticalMetadata::from_frame(&missing)),
                ("a".into(), OpticalMetadata::from_frame(&a)),
                ("b".into(), OpticalMetadata::from_frame(&b)),
            ]);
            assert!(!compatible, "{key}");
            assert!(notes[0].contains("shared lens fit skipped"));
            assert!(notes[0].contains("a (") && notes[0].contains("b ("));
        }
    }

    #[test]
    fn restored_panels_reject_invalid_bounds_and_stars() {
        let canvas = blank(100, 80, 1).unwrap();
        let region = ReferenceRegion {
            x: 0,
            y: 0,
            width: 100,
            height: 80,
        };
        let options = RegistrationOptions {
            minimum_matches: 3,
            ..RegistrationOptions::default()
        };
        let stars = [(10.0, 12.0), (50.0, 19.0), (65.0, 60.0), (20.0, 55.0)]
            .into_iter()
            .map(|(x, y)| DetectedStar {
                x,
                y,
                flux: 1.0,
                peak: 1.0,
                area: 1,
            })
            .collect::<Vec<_>>();
        let restore = |region, stars| {
            CanvasRegistrar::new(
                &canvas,
                &[region],
                Some(vec![stars]),
                None,
                &options,
                Scale::NATIVE,
                &crate::GpuOptions::default(),
            )
        };
        assert!(restore(region, stars.clone()).is_ok());
        for coordinate in [f64::NAN, f64::INFINITY, -1.0, 100.0] {
            let mut invalid = stars.clone();
            invalid[0].x = coordinate;
            assert!(restore(region, invalid).is_err());
        }
        let mut invalid = stars.clone();
        invalid[0].y = 80.0;
        assert!(restore(region, invalid).is_err());
        let mut invalid = stars.clone();
        invalid[0].flux = f64::NAN;
        assert!(restore(region, invalid).is_err());
        let mut invalid = stars.clone();
        invalid[0].peak = f32::INFINITY;
        assert!(restore(region, invalid).is_err());
        for invalid in [
            ReferenceRegion {
                x: usize::MAX,
                ..region
            },
            ReferenceRegion {
                y: usize::MAX,
                ..region
            },
        ] {
            assert!(restore(invalid, stars.clone()).is_err());
        }
    }

    #[test]
    fn bounds_include_rotated_corners_and_negative_origins() {
        let reference = blank(40, 20, 1).unwrap();
        let source = blank(30, 10, 1).unwrap();
        for rotation in [0.0, 0.2, std::f64::consts::PI] {
            let transform = SimilarityTransform {
                scale: 1.02,
                rotation_radians: rotation,
                translation_x: -12.3,
                translation_y: -8.7,
            };
            let (w, h, dx, dy) = union_bounds(&reference, &source, transform, 10_000).unwrap();
            assert!(w >= reference.width + dx && h >= reference.height + dy);
            let mut shifted = transform;
            shifted.translation_x += dx as f64;
            shifted.translation_y += dy as f64;
            let region = footprint(&source, shifted, w, h).unwrap();
            for (x, y) in [(0.0, 0.0), (29.0, 0.0), (0.0, 9.0), (29.0, 9.0)] {
                let (x, y) = shifted.apply(x, y);
                assert!(x >= 0.0 && x < w as f64 && y >= 0.0 && y < h as f64);
                assert!(x >= region.x as f64 && x < (region.x + region.width) as f64);
                assert!(y >= region.y as f64 && y < (region.y + region.height) as f64);
            }
        }
    }

    #[test]
    fn sample_limit_counts_channels() {
        let reference = blank(10, 10, 3).unwrap();
        let source = blank(10, 10, 3).unwrap();
        let shifted = SimilarityTransform {
            translation_x: 5.0,
            ..SimilarityTransform::IDENTITY
        };
        // 15x10 pixels, three channels.
        assert!(union_bounds(&reference, &source, shifted, 15 * 10 * 3).is_ok());
        assert!(union_bounds(&reference, &source, shifted, 15 * 10 * 3 - 1).is_err());
    }

    #[test]
    fn chance_ceiling_corrects_for_the_transforms_searched() {
        // One transform expecting 0.16 pairs rarely reaches 3, but the best
        // of 85 million reaches 6; 8 is out of reach.
        assert_eq!(chance_ceiling(0.16, 1.0), 3);
        assert_eq!(chance_ceiling(0.16, 8.5e7), 8);
        // A crowded field expecting 4 pairs per transform.
        let crowded = chance_ceiling(4.0, 2.6e5);
        assert!((18..=24).contains(&crowded), "{crowded}");
        assert!(chance_ceiling(64.0, 2.6e5) > 100);
    }

    #[test]
    fn paired_stars_keep_only_panel_stars_with_a_frame_star_nearby() {
        let panel = [(10.0, 10.0), (50.0, 50.0), (90.0, 12.0)];
        let landed = [(10.8, 10.9), (58.0, 50.0), (89.0, 13.0), (0.0, 0.0)];
        assert_eq!(
            paired_stars(&panel, &landed, 1.5),
            vec![(10.0, 10.0), (90.0, 12.0)]
        );
        assert_eq!(paired_stars(&panel, &landed, 10.0).len(), 3);
    }

    #[test]
    fn residual_outliers_exclude_panel_starts_and_need_a_median() {
        let planned = |rms: f64, starts_panel: bool| PlannedRegistration {
            result: Ok(RegistrationResult {
                transform: SimilarityTransform::IDENTITY,
                matched_stars: 100,
                rms_error_pixels: rms,
                drift_pixels: 0.0,
                warp: None,
            }),
            stamp: None,
            starts_panel,
        };
        let mut plans = HashMap::new();
        for (index, rms) in [0.05, 0.06, 0.07, 0.07, 0.08, 0.09].into_iter().enumerate() {
            plans.insert(PathBuf::from(format!("{index}")), planned(rms, false));
        }
        plans.insert("start".into(), planned(1.3, true));
        plans.insert("partial".into(), planned(0.54, false));
        plans.insert("seeing".into(), planned(0.25, false));
        reject_residual_outliers(&mut plans, Scale::NATIVE);
        assert!(plans[Path::new("start")].result.is_ok());
        assert!(plans[Path::new("seeing")].result.is_ok(), "under the floor");
        assert!(
            plans[Path::new("partial")]
                .result
                .as_ref()
                .unwrap_err()
                .contains("0.54px RMS")
        );
        // Too few frames for a median: nothing is judged.
        let mut few = HashMap::from([
            ("a".into(), planned(0.05, false)),
            ("b".into(), planned(0.9, false)),
        ]);
        reject_residual_outliers(&mut few, Scale::NATIVE);
        assert!(few.values().all(|plan| plan.result.is_ok()));
    }

    #[test]
    fn crop_reads_a_moving_canvas() {
        let canvas = LinearImage::new(3, 2, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let region = ReferenceRegion {
            x: 0,
            y: 0,
            width: 3,
            height: 2,
        };
        let moved = crop(&canvas, region, 1, 1).unwrap();
        assert!(moved.data[..4].iter().all(|v| v.is_nan()));
        assert_eq!(&moved.data[4..], &[1.0, 2.0]);
        assert_eq!(crop(&canvas, region, 0, 0).unwrap().data, canvas.data);
    }

    /// The overlap counted on the canvas is the one its crop gives, for a
    /// canvas with gaps, about to move, under a region reaching past it.
    #[test]
    fn overlap_counts_what_the_crop_holds() {
        let (width, height, channels) = (9, 7, 3);
        let data = (0..width * height * channels)
            .map(|index| {
                if index % 7 == 3 {
                    f32::NAN
                } else {
                    index as f32
                }
            })
            .collect();
        let canvas = LinearImage::new(width, height, channels, data).unwrap();
        let region = ReferenceRegion {
            x: 1,
            y: 2,
            width: 10,
            height: 8,
        };
        let covered = (0..region.width * region.height * channels)
            .map(|index| index % 5 != 1)
            .collect::<Vec<_>>();
        for shift in [(0, 0), (2, 1), (4, 6)] {
            let crop = crop(&canvas, region, shift.0, shift.1).unwrap();
            let common = crop
                .data
                .iter()
                .zip(&covered)
                .filter(|(value, covered)| **covered && value.is_finite())
                .count();
            let all = covered.iter().filter(|covered| **covered).count();
            assert!(common > 0 && common < all);
            assert_eq!(
                overlap(&canvas, region, shift, &covered),
                (common, all - common)
            );
        }
    }
}
