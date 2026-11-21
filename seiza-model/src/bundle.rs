use crate::backend::{Backend, Precision};
use crate::error::ModelError;
use crate::tiles::Tiling;
use seiza_stars::hocus_focus_star_detection::StructureRemovalMethod;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};

/// The card format this build reads.
const CARD_FORMAT: u32 = 1;
/// The widest margin a card's hole guard may ask for, in pixels.
const MAX_HOLE_MARGIN: usize = 32;
/// The widest box a card's hole guard may fill from, in pixels each way.
const MAX_FILL_RADIUS: usize = 64;
/// A card's file name in its bundle.
pub(crate) const CARD_FILE: &str = "card.json";

/// What a model does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Task {
    /// Takes out noise.
    Denoise,
    /// Brings an image's PSF to a narrower one.
    Sharpen,
}

impl fmt::Display for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Denoise => "denoise",
            Self::Sharpen => "sharpen",
        })
    }
}

impl std::str::FromStr for Task {
    type Err = ModelError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "denoise" => Ok(Self::Denoise),
            "sharpen" => Ok(Self::Sharpen),
            other => Err(ModelError::InvalidOption(format!(
                "a model's task is denoise or sharpen, not {other:?}"
            ))),
        }
    }
}

/// What fills a network's noise plane. Both are made from the image alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NoiseInput {
    /// The image map's median everywhere: its noise level, and nothing of
    /// where that changes.
    Constant,
    /// The image map: second differences in blocks, plus a signal term.
    Image,
}

impl fmt::Display for NoiseInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Constant => "constant",
            Self::Image => "image",
        })
    }
}

/// What one plane of a network's input holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InputPlane {
    /// The image, as `asinh((x − b) / σ̄)`.
    Image,
    /// The noise input, read as [`NoisePlane`] says.
    Noise,
    /// The width a sharpening model is asked to bring each star to, as a
    /// share of its own: FWHM(target) / FWHM(image), at every pixel.
    Ratio,
    /// The same share, asked of everything that is not a star.
    NonstellarRatio,
    /// The image's point sources, rendered with the local PSF, as
    /// `asinh(p / σ̄)`.
    Points,
}

impl fmt::Display for InputPlane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Image => "image",
            Self::Noise => "noise",
            Self::Ratio => "ratio",
            Self::NonstellarRatio => "nonstellar-ratio",
            Self::Points => "points",
        })
    }
}

/// One of a graph's input tensors.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphInput {
    /// The tensor's name in the graph.
    pub tensor: String,
    /// The planes it holds, in order.
    pub planes: Vec<InputPlane>,
}

/// How a network reads its noise plane. A model given the reading it was not
/// trained on fails badly, so the card names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NoisePlane {
    /// `asinh(σ / σ̄)`.
    Asinh,
    /// `σ / σ̄`.
    Linear,
}

/// How a network reads the sky.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SkyInput {
    /// Less the one background of the whole image.
    Global,
    /// Less the image's smooth sky first.
    Flat,
}

/// How a model reads an image: everything that must match its training. A
/// setting this build does not know is refused, never ignored, so a card that
/// asks for more than a build does cannot run without it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reading {
    /// What fills the noise plane.
    pub noise_map: NoiseInput,
    /// How the noise plane is read.
    pub noise_plane: NoisePlane,
    /// How the sky is read.
    pub sky: SkyInput,
    /// The side of the blocks the noise map takes the sky's variance over.
    pub sky_block: usize,
    /// The side of the blocks whose medians make the smooth sky.
    pub flat_block: usize,
    /// The image map's level.
    pub image_map_scale: f64,
    /// How much neighbouring pixels shared their noise in the stacks the model
    /// was trained on: the second-difference noise of an image binned 2 × 2,
    /// times 2, over its own, which white noise gives as 1. The image map is
    /// raised by as much as an image's exceeds it, as a drizzled stack's does.
    /// None leaves the map as measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise_correlation: Option<f64>,
    /// How the image's holes are kept from the network. None runs the network
    /// on the image as it is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub holes: Option<HoleGuard>,
}

/// How the image's holes are kept from a network: pixels far below the sky
/// that every frame held, such as hot pixels a dark took too much of. A network
/// that meets one fills it and overshoots into a bright spike ringed by
/// speckle, so they are filled from their surroundings before it runs, and put
/// back after.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HoleGuard {
    /// How far below the sky's floor a pixel is a hole, in the image map's
    /// noise.
    pub depth: f64,
    /// How far the mask reaches past a hole, in pixels, for the lobes
    /// registration spreads round it.
    pub margin: usize,
    /// A masked pixel is filled with the median of the unmasked pixels this
    /// far round it.
    pub fill_radius: usize,
    /// Within the mask, the input is put back where it differs from the
    /// network's output by more than this many of the image map's noise.
    pub restore: f64,
}

/// How a model tiles an image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TilingCard {
    /// The tile a backend with no size of its own runs.
    pub tile: usize,
    /// How far in from an inner edge a tile takes no weight: the network's
    /// reach.
    pub margin: usize,
    /// How far past the margins the blend's cosine runs.
    pub ramp: usize,
    /// Every tile starts on a multiple of this.
    pub multiple: usize,
}

impl TilingCard {
    /// The tiling for a tile of `tile` pixels, or this card's own. An overlap
    /// too large for `usize` saturates, and [`Tiling::tiles`] refuses it.
    pub fn tiling(&self, tile: Option<usize>) -> Tiling {
        Tiling {
            tile: tile.unwrap_or(self.tile),
            overlap: self.margin.saturating_mul(2).saturating_add(self.ramp),
            margin: self.margin,
            multiple: self.multiple,
        }
    }
}

/// What one plane of a PSF stamp holds, as a sharpening model's encoder reads
/// it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StampPlane {
    /// The stamp over its peak.
    Peak,
    /// `asinh(s / (peak · wings)) / asinh(1 / wings)`, which shows the wings.
    Wings,
}

/// How a sharpening model's residual meets the image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ResidualMode {
    /// Added to the asinh of the image in σ̄, and taken back through sinh.
    Asinh,
    /// Added to the image in σ̄.
    Linear,
}

/// The graph that reduces a PSF stamp to the vector a sharpening model's body
/// reads. It is single precision and runs on ONNX Runtime's CPU provider
/// whatever runs the body, once for each node of the PSF field, so the
/// vectors are the same on every backend.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StampEncoder {
    /// The file's name in the bundle.
    pub file: String,
    /// The file's SHA-256, in hexadecimal.
    pub sha256: String,
    /// The graph's input tensor: the stamp's planes.
    pub input: String,
    /// The graph's output tensor: the vector.
    pub output: String,
    /// The planes the stamp is read as, in order.
    pub planes: Vec<StampPlane>,
    /// The stamp's side in pixels, odd. A PSF field's stamp is cut or padded
    /// to it about its middle.
    pub side: usize,
    /// The share of the peak the [`StampPlane::Wings`] plane is scaled by.
    pub wings: f64,
    /// The vector's length.
    pub features: usize,
}

/// The body's tensor that holds the encoder's vectors as a map: one vector
/// for each pixel of the network's coarsest grid, a tile's side over the
/// tiling's multiple each way.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorInput {
    /// The tensor's name in the graph.
    pub tensor: String,
    /// The graph's free dimensions for the map's height and width.
    pub dimensions: Vec<String>,
}

/// The shares of their width a sharpening brings things to: 1 leaves them as
/// they are, and less is sharper.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Strengths {
    /// The share of its width every star is brought to.
    pub stars: f64,
    /// The share asked of everything that is not a star. A model with no
    /// [`InputPlane::NonstellarRatio`] plane does not read it.
    pub rest: f64,
}

/// How the image's PSF field is measured: [`seiza_stars::PsfFieldOptions`]'
/// settings of the same names, and how the field is sampled.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PsfReading {
    /// Principal components beyond the mean PSF.
    pub components: usize,
    /// The fewest stars a varying field is fitted from. With fewer, one PSF
    /// is fitted for the whole image.
    pub minimum_stars: usize,
    /// The most stars the core is fitted from.
    pub maximum_stars: usize,
    /// The lowest signal-to-noise ratio of a star that gives the core.
    pub minimum_snr: f64,
    /// The detection threshold, in noise σ.
    pub detection_sigma: f64,
    /// How much fainter than a star a near neighbour may be.
    pub isolation_magnitudes: f64,
    /// The most the field's nodes are apart, in pixels. Both edges of each
    /// axis hold a node.
    pub grid_px: usize,
    /// A stamp's width is a round Gaussian's, fitted to the pixels at this
    /// share of its peak or more.
    pub width_core: f64,
}

/// How the point-source detector takes large-scale structure (nebulosity,
/// gradients) away before it looks for stars.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StructureRemoval {
    /// Gaussian and domain-transform layers subtracted: the detector's
    /// default. It takes bright stars on bright, structured nebula away
    /// with the nebula.
    Filtered,
    /// The à trous wavelet residual, which keeps those stars.
    Atrous,
}

impl StructureRemoval {
    /// The detector's own name for this removal.
    pub(crate) fn method(self) -> StructureRemovalMethod {
        match self {
            Self::Filtered => StructureRemovalMethod::Filtered,
            Self::Atrous => StructureRemovalMethod::Atrous,
        }
    }
}

impl fmt::Display for StructureRemoval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Filtered => "filtered",
            Self::Atrous => "atrous",
        })
    }
}

impl std::str::FromStr for StructureRemoval {
    type Err = ModelError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "filtered" => Ok(Self::Filtered),
            "atrous" => Ok(Self::Atrous),
            other => Err(ModelError::InvalidOption(format!(
                "a structure removal is filtered or atrous, not {other:?}"
            ))),
        }
    }
}

/// What a card that names no structure removals detects with: the filtered
/// removal alone, as every card did before they were named.
fn filtered_alone() -> Vec<StructureRemoval> {
    vec![StructureRemoval::Filtered]
}

/// How the image's point sources are found.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PointsReading {
    /// The detector's noise reduction radius; 0 detects on the image
    /// unsmoothed.
    pub noise_reduction_radius: usize,
    /// The detector runs once with each of these, in turn. The clipped
    /// cores are found after the first run, against its stars alone, so
    /// its list and cores are as with that removal alone. A star a later
    /// run finds within `core_radius` of a point already listed, a core
    /// among them, is the same star; the others are added after them.
    #[serde(default = "filtered_alone")]
    pub structure_removals: Vec<StructureRemoval>,
    /// The clip level is read from this many of the brightest objects' peaks.
    pub clip_peaks: usize,
    /// An object's peak is the largest value of a block of this side that
    /// is the largest of the 3 × 3 blocks about it.
    pub peak_block: usize,
    /// A clipped core, or a star a later detector run finds, within this
    /// many pixels of a brighter core or of a star already listed is the
    /// same star.
    pub core_radius: f64,
}

/// What a sharpening model reads beyond a noise model's card. Everything it
/// names is made from the image alone.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SharpenCard {
    /// The stamp encoder.
    pub encoder: StampEncoder,
    /// The body's vector map.
    pub vector: VectorInput,
    /// How the residual meets the image.
    pub residual: ResidualMode,
    /// The narrowest a star is asked to be, in pixels of its stamp's width.
    pub floor_px: f64,
    /// The sharpest share the model was trained to be asked for. A run may
    /// ask for less, and is then outside what the model has seen.
    pub sharpest: f64,
    /// The shares a run asks for when its caller names none.
    pub strengths: Strengths,
    /// How the PSF field is measured.
    pub psf: PsfReading,
    /// How the point sources are found.
    pub points: PointsReading,
}

/// One of a bundle's graphs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphFile {
    /// The graph's precision.
    pub precision: Precision,
    /// The file's name in the bundle.
    pub file: String,
    /// The file's SHA-256, in hexadecimal.
    pub sha256: String,
}

/// What a model is and how it must be run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelCard {
    /// The card's format.
    pub format: u32,
    /// The model's name, which is also its directory when installed.
    pub name: String,
    /// What the model does.
    pub task: Task,
    /// The graphs, one per precision.
    pub graphs: Vec<GraphFile>,
    /// The graph's input tensors. A model with no noise plane lists only the
    /// image, and the order of the planes is the graph's.
    pub inputs: Vec<GraphInput>,
    /// The graph's output tensor: one plane, the same size as a tile.
    pub output: String,
    /// The graph's free dimensions, each set to the tile's side. None means
    /// a graph fixed at `tiling.tile`, which is then the only tile it runs.
    pub dimensions: Vec<String>,
    /// How the model reads an image.
    pub reading: Reading,
    /// How the model tiles an image.
    pub tiling: TilingCard,
    /// The backends the model has passed parity on. Empty allows every one.
    #[serde(default)]
    pub parity: Vec<String>,
    /// What a sharpening model reads besides: present exactly when `task`
    /// is [`Task::Sharpen`]. Its `graphs` are then the body's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sharpen: Option<SharpenCard>,
}

impl ModelCard {
    /// Every plane the graph reads, in the order of its inputs.
    pub fn planes(&self) -> Vec<InputPlane> {
        self.inputs
            .iter()
            .flat_map(|input| input.planes.iter().copied())
            .collect()
    }

    /// Whether the card lets `backend` run this model unasked.
    pub fn allows(&self, backend: Backend) -> bool {
        self.parity.is_empty() || self.parity.iter().any(|word| *word == backend.to_string())
    }

    fn check(&self) -> Result<(), ModelError> {
        let refuse = |reason: String| Err(ModelError::ModelContract(reason));
        if self.format != CARD_FORMAT {
            return refuse(format!(
                "card format {} is not the {CARD_FORMAT} this build reads",
                self.format
            ));
        }
        self.tiling
            .tiling(None)
            .tiles(self.tiling.tile, self.tiling.tile)
            .map_err(|error| {
                ModelError::ModelContract(format!("{}'s tiling: {error}", self.name))
            })?;
        if self.inputs.iter().any(|input| input.planes.is_empty()) {
            return refuse(format!("{} names an input with no planes", self.name));
        }
        for (index, input) in self.inputs.iter().enumerate() {
            if self.inputs[..index]
                .iter()
                .any(|i| i.tensor == input.tensor)
            {
                return refuse(format!("{} names {} twice", self.name, input.tensor));
            }
        }
        let planes = self.planes();
        let count = |plane| planes.iter().filter(|&&each| each == plane).count();
        if count(InputPlane::Image) != 1 {
            return refuse(format!("{} must read the image once", self.name));
        }
        if count(InputPlane::Noise) > 1 {
            return refuse(format!("{} reads the noise input twice", self.name));
        }
        let asked = [
            InputPlane::Ratio,
            InputPlane::NonstellarRatio,
            InputPlane::Points,
        ];
        match (self.task, &self.sharpen) {
            (Task::Denoise, None) => {
                if let Some(plane) = asked.into_iter().find(|&plane| count(plane) > 0) {
                    return refuse(format!(
                        "{} is a noise model, which has no {plane} plane",
                        self.name
                    ));
                }
            }
            (Task::Sharpen, Some(sharpen)) => {
                if count(InputPlane::Ratio) != 1 {
                    return refuse(format!("{} must read the ratio once", self.name));
                }
                if let Some(plane) = asked.into_iter().find(|&plane| count(plane) > 1) {
                    return refuse(format!("{} reads the {plane} plane twice", self.name));
                }
                if self.inputs.len() != 1 {
                    return refuse(format!("{} must read its planes as one tensor", self.name));
                }
                if self.reading.holes.is_some() {
                    return refuse(format!(
                        "{} is a sharpening model, which has no hole guard",
                        self.name
                    ));
                }
                sharpen.check(self)?;
            }
            (Task::Denoise, Some(_)) => {
                return refuse(format!(
                    "{} is a noise model with a sharpening section",
                    self.name
                ));
            }
            (Task::Sharpen, None) => {
                return refuse(format!(
                    "{} is a sharpening model with no sharpening section",
                    self.name
                ));
            }
        }
        if self.graphs.is_empty() {
            return refuse(format!("{} names no graph", self.name));
        }
        if let Some(trained) = self.reading.noise_correlation
            && !(trained > 0.0 && trained.is_finite())
        {
            return refuse(format!(
                "{}'s noise correlation must be positive, not {trained}",
                self.name
            ));
        }
        if let Some(guard) = self.reading.holes {
            let positive = |value: f64| value > 0.0 && value.is_finite();
            if !(positive(guard.depth) && positive(guard.restore)) {
                return refuse(format!(
                    "{}'s holes need a positive depth and restore, not {} and {}",
                    self.name, guard.depth, guard.restore
                ));
            }
            if guard.margin > MAX_HOLE_MARGIN || guard.fill_radius > MAX_FILL_RADIUS {
                return refuse(format!(
                    "{}'s holes reach at most {MAX_HOLE_MARGIN} pixels past a hole and fill from \
                     at most {MAX_FILL_RADIUS}, not {} and {}",
                    self.name, guard.margin, guard.fill_radius
                ));
            }
        }
        let name = Path::new(&self.name);
        if self.name.is_empty() || name.file_name() != Some(name.as_os_str()) {
            return refuse(format!("{:?} cannot name a model's directory", self.name));
        }
        for name in self.files() {
            let file = Path::new(name);
            if file.file_name() != Some(file.as_os_str()) {
                return refuse(format!("{name:?} is not a file in the bundle"));
            }
        }
        Ok(())
    }

    /// Every file the card names besides itself: its graphs, and a
    /// sharpening model's encoder.
    pub(crate) fn files(&self) -> impl Iterator<Item = &str> {
        self.graphs.iter().map(|graph| graph.file.as_str()).chain(
            self.sharpen
                .iter()
                .map(|sharpen| sharpen.encoder.file.as_str()),
        )
    }
}

impl SharpenCard {
    /// Refuse a section a run could not follow.
    fn check(&self, card: &ModelCard) -> Result<(), ModelError> {
        let name = &card.name;
        let refuse = |reason: String| Err(ModelError::ModelContract(format!("{name} {reason}")));
        let positive = |value: f64| value > 0.0 && value.is_finite();
        let share = |value: f64| value > 0.0 && value <= 1.0;
        let encoder = &self.encoder;
        if encoder.side.is_multiple_of(2) || !(9..=1025).contains(&encoder.side) {
            return refuse(format!(
                "reads a stamp of {} pixels, which must be odd and 9 to 1025",
                encoder.side
            ));
        }
        if encoder.planes.is_empty() || encoder.features == 0 || encoder.features > 4096 {
            return refuse("names an encoder with no planes or no vector".to_string());
        }
        if !(positive(encoder.wings) && encoder.wings < 1.0) {
            return refuse(format!(
                "scales a stamp's wings by {}, which must lie between 0 and 1",
                encoder.wings
            ));
        }
        if self.vector.dimensions.len() != 2
            || card.dimensions.len() != 2
            || self
                .vector
                .dimensions
                .iter()
                .any(|dimension| card.dimensions.contains(dimension))
        {
            return refuse(
                "must leave its body's height and width free, and its vector map's, under four \
                 names"
                    .to_string(),
            );
        }
        if card
            .inputs
            .iter()
            .any(|input| input.tensor == self.vector.tensor)
        {
            return refuse(format!("names {} twice", self.vector.tensor));
        }
        if !(positive(self.floor_px) && share(self.sharpest)) {
            return refuse(format!(
                "needs a positive floor and a sharpest share up to 1, not {} and {}",
                self.floor_px, self.sharpest
            ));
        }
        if !(share(self.strengths.stars) && share(self.strengths.rest)) {
            return refuse(format!(
                "asks for shares of {} and {}, which must be above 0 and at most 1",
                self.strengths.stars, self.strengths.rest
            ));
        }
        if self.psf.grid_px == 0 || !(positive(self.psf.width_core) && self.psf.width_core < 1.0) {
            return refuse("names a PSF grid of no pixels or a core of no share".to_string());
        }
        let options = self.psf.options(seiza_stars::SparseFieldPolicy::Refuse);
        if let Err(error) = options.validate() {
            return refuse(format!(
                "measures its PSF field with settings refused: {error}"
            ));
        }
        if self.points.peak_block == 0 || self.points.clip_peaks == 0 {
            return refuse(
                "finds clipped cores in blocks of no pixels or from no peaks".to_string(),
            );
        }
        let removals = &self.points.structure_removals;
        if removals.is_empty()
            || removals
                .iter()
                .enumerate()
                .any(|(index, removal)| removals[..index].contains(removal))
        {
            return refuse(format!(
                "detects point sources with the structure removals {removals:?}, which must be \
                 at least one, each once"
            ));
        }
        if !(self.points.core_radius >= 0.0 && self.points.core_radius.is_finite()) {
            return refuse(format!(
                "merges clipped cores within {} pixels, which must be finite and not negative",
                self.points.core_radius
            ));
        }
        Ok(())
    }
}

impl PsfReading {
    /// The settings [`seiza_stars::PsfField::measure`] takes, with `sparse`
    /// saying what too few stars lead to.
    pub fn options(&self, sparse: seiza_stars::SparseFieldPolicy) -> seiza_stars::PsfFieldOptions {
        seiza_stars::PsfFieldOptions {
            components: self.components,
            minimum_stars: self.minimum_stars,
            maximum_stars: self.maximum_stars,
            minimum_snr: self.minimum_snr,
            detection_sigma: self.detection_sigma,
            isolation_magnitudes: self.isolation_magnitudes,
            sparse,
            ..Default::default()
        }
    }
}

/// A model on disk: its card, and the graphs the card names.
#[derive(Clone, Debug)]
pub struct ModelBundle {
    root: PathBuf,
    card: ModelCard,
    card_sha256: String,
}

impl ModelBundle {
    /// Read a bundle's card. Its graphs are not hashed until one is run, or
    /// [`verify`](Self::verify) is called.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, ModelError> {
        let root = root.as_ref().to_path_buf();
        let path = root.join(CARD_FILE);
        let text = std::fs::read_to_string(&path).map_err(ModelError::io(&path))?;
        let card: ModelCard = serde_json::from_str(&text)
            .map_err(|error| ModelError::ModelContract(format!("{}: {error}", path.display())))?;
        card.check()?;
        let card_sha256 = hex(Sha256::digest(text.as_bytes()));
        Ok(Self {
            root,
            card,
            card_sha256,
        })
    }

    /// The bundle's directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The bundle's card.
    pub fn card(&self) -> &ModelCard {
        &self.card
    }

    /// The SHA-256 of the bundle's card file, in hexadecimal. The card names
    /// each graph by its own hash, and everything a run reads the image by,
    /// so this changes whenever a run's output could: a host keys cached
    /// results on it, with the backend, the precision and
    /// [`OUTPUT_VERSION`](crate::OUTPUT_VERSION). Installing copies the card
    /// as it is, so an installed model's is its bundle's.
    pub fn card_sha256(&self) -> &str {
        &self.card_sha256
    }

    /// The graph of `precision`, when the bundle has one.
    pub fn graph(&self, precision: Precision) -> Option<&GraphFile> {
        self.card
            .graphs
            .iter()
            .find(|graph| graph.precision == precision)
    }

    /// The path of a graph in this bundle.
    pub(crate) fn path_of(&self, graph: &GraphFile) -> PathBuf {
        self.root.join(&graph.file)
    }

    /// Refuse a graph whose file does not match the card.
    pub(crate) fn verify_graph(&self, graph: &GraphFile) -> Result<(), ModelError> {
        let file = self.path_of(graph);
        let found = sha256_of(&file)?;
        if found != graph.sha256 {
            return Err(ModelError::HashMismatch {
                file,
                expected: graph.sha256.clone(),
                found,
            });
        }
        Ok(())
    }

    /// Refuse a bundle any of whose graphs does not match its card.
    pub fn verify(&self) -> Result<(), ModelError> {
        self.card
            .graphs
            .iter()
            .try_for_each(|graph| self.verify_graph(graph))?;
        self.verify_encoder()
    }

    /// Refuse a sharpening model's encoder whose file does not match the
    /// card. A noise model has none.
    pub(crate) fn verify_encoder(&self) -> Result<(), ModelError> {
        let Some(sharpen) = &self.card.sharpen else {
            return Ok(());
        };
        let file = self.root.join(&sharpen.encoder.file);
        let found = sha256_of(&file)?;
        if found != sharpen.encoder.sha256 {
            return Err(ModelError::HashMismatch {
                file,
                expected: sharpen.encoder.sha256.clone(),
                found,
            });
        }
        Ok(())
    }
}

fn sha256_of(path: &Path) -> Result<String, ModelError> {
    let mut file = std::fs::File::open(path).map_err(ModelError::io(path))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 20];
    loop {
        let count = file.read(&mut buffer).map_err(ModelError::io(path))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hex(hasher.finalize()))
}

fn hex(digest: impl AsRef<[u8]>) -> String {
    digest
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A bundle of one small file, its card holding `noise_map`.
    pub(crate) fn write_bundle(root: &Path, name: &str, noise_map: &str) {
        std::fs::create_dir_all(root).unwrap();
        std::fs::write(root.join("model-f32.onnx"), b"graph").unwrap();
        let sha256 = sha256_of(&root.join("model-f32.onnx")).unwrap();
        let card = format!(
            r#"{{
              "format": 1, "name": "{name}", "task": "denoise",
              "graphs": [{{"precision": "f32", "file": "model-f32.onnx", "sha256": "{sha256}"}}],
              "inputs": [{{"tensor": "planes", "planes": ["image", "noise"]}}],
              "output": "residual",
              "dimensions": ["height", "width"],
              "reading": {{"noise_map": "{noise_map}", "noise_plane": "asinh", "sky": "flat",
                           "sky_block": 32, "flat_block": 256, "image_map_scale": 1.13}},
              "tiling": {{"tile": 512, "margin": 122, "ramp": 32, "multiple": 16}}
            }}"#
        );
        std::fs::write(root.join(CARD_FILE), card).unwrap();
    }

    /// A sharpening bundle of two small files and its card.
    pub(crate) fn write_sharpen_bundle(root: &Path, name: &str) {
        std::fs::create_dir_all(root).unwrap();
        std::fs::write(root.join("body-f32.onnx"), b"body").unwrap();
        std::fs::write(root.join("encoder-f32.onnx"), b"encoder").unwrap();
        let body = sha256_of(&root.join("body-f32.onnx")).unwrap();
        let encoder = sha256_of(&root.join("encoder-f32.onnx")).unwrap();
        let card = format!(
            r#"{{
              "format": 1, "name": "{name}", "task": "sharpen",
              "graphs": [{{"precision": "f32", "file": "body-f32.onnx", "sha256": "{body}"}}],
              "inputs": [{{"tensor": "planes",
                           "planes": ["image", "noise", "ratio", "nonstellar-ratio", "points"]}}],
              "output": "residual",
              "dimensions": ["height", "width"],
              "reading": {{"noise_map": "image", "noise_plane": "asinh", "sky": "global",
                           "sky_block": 32, "flat_block": 256, "image_map_scale": 1.13,
                           "noise_correlation": 1.24}},
              "tiling": {{"tile": 512, "margin": 122, "ramp": 32, "multiple": 16}},
              "sharpen": {{
                "encoder": {{"file": "encoder-f32.onnx", "sha256": "{encoder}", "input": "stamp",
                             "output": "vector", "planes": ["peak", "wings"], "side": 129,
                             "wings": 0.001, "features": 64}},
                "vector": {{"tensor": "vector", "dimensions": ["map_height", "map_width"]}},
                "residual": "asinh", "floor_px": 1.5, "sharpest": 0.25,
                "strengths": {{"stars": 0.46, "rest": 0.5}},
                "psf": {{"components": 3, "minimum_stars": 30, "maximum_stars": 400,
                         "minimum_snr": 30.0, "detection_sigma": 5.0,
                         "isolation_magnitudes": 5.0, "grid_px": 256, "width_core": 0.05}},
                "points": {{"noise_reduction_radius": 0, "structure_removals": ["filtered", "atrous"],
                            "clip_peaks": 300, "peak_block": 16, "core_radius": 24.0}}
              }}
            }}"#
        );
        std::fs::write(root.join(CARD_FILE), card).unwrap();
    }

    #[test]
    fn a_sharpening_card_reads_back_and_refuses_what_a_run_could_not_follow() {
        let temp = tempfile::tempdir().unwrap();
        write_sharpen_bundle(temp.path(), "sharpen-test");
        let bundle = ModelBundle::open(temp.path()).unwrap();
        let card = bundle.card();
        assert_eq!(card.task, Task::Sharpen);
        assert_eq!(card.planes().len(), 5);
        let sharpen = card.sharpen.as_ref().unwrap();
        assert_eq!(sharpen.residual, ResidualMode::Asinh);
        assert_eq!(
            sharpen.encoder.planes,
            [StampPlane::Peak, StampPlane::Wings]
        );
        assert_eq!(sharpen.psf.grid_px, 256);
        assert_eq!(
            sharpen.points.structure_removals,
            [StructureRemoval::Filtered, StructureRemoval::Atrous]
        );
        bundle.verify().unwrap();

        let path = temp.path().join(CARD_FILE);
        let good: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let json = |text: &str| serde_json::from_str::<serde_json::Value>(text).unwrap();
        for (pointer, value) in [
            ("/task", json(r#""denoise""#)),
            ("/sharpen", serde_json::Value::Null),
            ("/sharpen/encoder/side", json("128")),
            ("/sharpen/encoder/file", json(r#""../encoder.onnx""#)),
            ("/sharpen/encoder/wings", json("0")),
            ("/sharpen/vector/tensor", json(r#""planes""#)),
            ("/sharpen/vector/dimensions", json(r#"["height", "width"]"#)),
            ("/sharpen/floor_px", json("0")),
            ("/sharpen/strengths/stars", json("1.5")),
            ("/sharpen/psf/grid_px", json("0")),
            ("/sharpen/psf/minimum_stars", json("1")),
            ("/sharpen/points/peak_block", json("0")),
            ("/sharpen/points/structure_removals", json("[]")),
            (
                "/sharpen/points/structure_removals",
                json(r#"["atrous", "atrous"]"#),
            ),
            ("/sharpen/points/structure_removals", json(r#"["median"]"#)),
            ("/sharpen/residual", json(r#""log""#)),
            ("/sharpen/stars", json("true")),
            ("/inputs/0/planes", json(r#"["image", "noise", "points"]"#)),
            (
                "/reading/holes",
                json(r#"{"depth": 30.0, "margin": 3, "fill_radius": 12, "restore": 5.0}"#),
            ),
        ] {
            let mut card = good.clone();
            match card.pointer_mut(pointer) {
                Some(slot) => *slot = value,
                None => {
                    let (parent, key) = pointer.rsplit_once('/').unwrap();
                    card.pointer_mut(parent).unwrap()[key] = value;
                }
            }
            std::fs::write(&path, serde_json::to_vec(&card).unwrap()).unwrap();
            assert!(
                matches!(
                    ModelBundle::open(temp.path()),
                    Err(ModelError::ModelContract(_))
                ),
                "{pointer}"
            );
        }
    }

    #[test]
    fn a_card_that_names_no_structure_removal_detects_as_cards_did_before() {
        let temp = tempfile::tempdir().unwrap();
        write_sharpen_bundle(temp.path(), "sharpen-test");
        let path = temp.path().join(CARD_FILE);
        let mut card: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        card.pointer_mut("/sharpen/points")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .remove("structure_removals");
        std::fs::write(&path, serde_json::to_vec(&card).unwrap()).unwrap();
        let bundle = ModelBundle::open(temp.path()).unwrap();
        let points = &bundle.card().sharpen.as_ref().unwrap().points;
        assert_eq!(points.structure_removals, [StructureRemoval::Filtered]);
    }

    #[test]
    fn a_structure_removal_reads_and_writes_as_a_word() {
        for removal in [StructureRemoval::Filtered, StructureRemoval::Atrous] {
            assert_eq!(
                removal.to_string().parse::<StructureRemoval>().unwrap(),
                removal
            );
            assert_eq!(
                serde_json::to_string(&removal).unwrap(),
                format!("\"{removal}\"")
            );
        }
        assert!("median".parse::<StructureRemoval>().is_err());
    }

    #[test]
    fn a_noise_card_has_no_sharpening_planes() {
        let temp = tempfile::tempdir().unwrap();
        write_bundle(temp.path(), "noise-test", "constant");
        let path = temp.path().join(CARD_FILE);
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::write(&path, text.replace(r#""noise"]"#, r#""noise", "ratio"]"#)).unwrap();
        let error = ModelBundle::open(temp.path()).unwrap_err();
        assert!(error.to_string().contains("ratio"), "{error}");
    }

    #[test]
    fn a_card_refuses_invalid_tile_geometry_before_loading_a_graph() {
        let temp = tempfile::tempdir().unwrap();
        write_bundle(temp.path(), "noise-test", "constant");
        let path = temp.path().join(CARD_FILE);
        let good: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        for (field, value) in [
            ("margin", usize::MAX),
            ("margin", usize::MAX / 2 + 1),
            ("ramp", usize::MAX),
            ("multiple", 0),
            ("tile", 0),
            ("tile", 511),
            ("tile", 256),
        ] {
            let mut card = good.clone();
            card["tiling"][field] = serde_json::json!(value);
            std::fs::write(&path, serde_json::to_vec(&card).unwrap()).unwrap();
            assert!(
                matches!(
                    ModelBundle::open(temp.path()),
                    Err(ModelError::ModelContract(_))
                ),
                "{field}={value}"
            );
        }
    }

    #[test]
    fn a_card_s_hash_follows_what_it_holds() {
        let temp = tempfile::tempdir().unwrap();
        write_bundle(temp.path(), "noise-test", "constant");
        let first = ModelBundle::open(temp.path()).unwrap();
        assert_eq!(first.card_sha256().len(), 64);
        assert_eq!(
            ModelBundle::open(temp.path()).unwrap().card_sha256(),
            first.card_sha256(),
            "the same card, the same hash"
        );
        let path = temp.path().join(CARD_FILE);
        let mut card: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        card["reading"]["holes"] =
            serde_json::json!({"depth": 30.0, "margin": 3, "fill_radius": 12, "restore": 5.0});
        std::fs::write(&path, serde_json::to_vec(&card).unwrap()).unwrap();
        assert_ne!(
            ModelBundle::open(temp.path()).unwrap().card_sha256(),
            first.card_sha256(),
            "a reading that changes the output changes the hash, though the graph does not"
        );
    }

    #[test]
    fn a_card_reads_back_and_its_graph_verifies() {
        let temp = tempfile::tempdir().unwrap();
        write_bundle(temp.path(), "noise-test", "constant");
        let bundle = ModelBundle::open(temp.path()).unwrap();
        assert_eq!(bundle.card().reading.noise_map, NoiseInput::Constant);
        assert!(bundle.graph(Precision::F32).is_some());
        assert!(bundle.graph(Precision::F16).is_none());
        assert_eq!(bundle.card().tiling.tiling(None).overlap, 276);
        assert_eq!(bundle.card().tiling.tiling(Some(1024)).tile, 1024);
        bundle.verify().unwrap();
    }

    #[test]
    fn a_card_lists_its_planes_in_the_graph_s_order() {
        let temp = tempfile::tempdir().unwrap();
        write_bundle(temp.path(), "noise-test", "constant");
        let mut card = ModelBundle::open(temp.path()).unwrap().card.clone();
        assert_eq!(card.planes(), [InputPlane::Image, InputPlane::Noise]);
        card.inputs = vec![
            GraphInput {
                tensor: "level".to_string(),
                planes: vec![InputPlane::Noise],
            },
            GraphInput {
                tensor: "picture".to_string(),
                planes: vec![InputPlane::Image],
            },
        ];
        assert_eq!(card.planes(), [InputPlane::Noise, InputPlane::Image]);
        assert!(card.check().is_ok());
        card.inputs.remove(0);
        assert!(card.check().is_ok(), "a model may read the image alone");
    }

    #[test]
    fn a_card_must_read_the_image_once_and_name_each_tensor_once() {
        let temp = tempfile::tempdir().unwrap();
        write_bundle(temp.path(), "noise-test", "constant");
        let good = ModelBundle::open(temp.path()).unwrap().card.clone();
        let input = |tensor: &str, planes: &[InputPlane]| GraphInput {
            tensor: tensor.to_string(),
            planes: planes.to_vec(),
        };
        let (image, noise) = (InputPlane::Image, InputPlane::Noise);
        for inputs in [
            vec![],
            vec![input("a", &[])],
            vec![input("a", &[noise])],
            vec![input("a", &[image, image])],
            vec![input("a", &[image, noise, noise])],
            vec![input("a", &[image]), input("a", &[noise])],
        ] {
            let card = ModelCard {
                inputs: inputs.clone(),
                ..good.clone()
            };
            assert!(
                matches!(card.check(), Err(ModelError::ModelContract(_))),
                "{inputs:?}"
            );
        }
    }

    #[test]
    fn a_plane_this_build_does_not_know_is_refused_by_name() {
        let temp = tempfile::tempdir().unwrap();
        write_bundle(temp.path(), "noise-test", "constant");
        let path = temp.path().join(CARD_FILE);
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::write(&path, text.replace(r#""noise"]"#, r#""psf"]"#)).unwrap();
        let error = ModelBundle::open(temp.path()).unwrap_err();
        assert!(error.to_string().contains("psf"), "{error}");
    }

    #[test]
    fn a_changed_graph_is_refused() {
        let temp = tempfile::tempdir().unwrap();
        write_bundle(temp.path(), "noise-test", "image");
        std::fs::write(temp.path().join("model-f32.onnx"), b"other").unwrap();
        let bundle = ModelBundle::open(temp.path()).unwrap();
        assert!(matches!(
            bundle.verify(),
            Err(ModelError::HashMismatch { .. })
        ));
    }

    #[test]
    fn a_card_needing_the_stack_s_records_is_refused() {
        let temp = tempfile::tempdir().unwrap();
        write_bundle(temp.path(), "noise-test", "model");
        assert!(matches!(
            ModelBundle::open(temp.path()),
            Err(ModelError::ModelContract(_))
        ));
    }

    #[test]
    fn a_name_that_leaves_the_store_is_refused() {
        let temp = tempfile::tempdir().unwrap();
        write_bundle(temp.path(), "../outside", "constant");
        assert!(matches!(
            ModelBundle::open(temp.path()),
            Err(ModelError::ModelContract(_))
        ));
    }

    #[test]
    fn a_card_turns_the_hole_guard_on_with_its_settings() {
        let temp = tempfile::tempdir().unwrap();
        write_bundle(temp.path(), "noise-test", "constant");
        assert_eq!(
            ModelBundle::open(temp.path()).unwrap().card().reading.holes,
            None,
            "a card without it runs the network on the image as it is"
        );
        let path = temp.path().join(CARD_FILE);
        let mut card: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        card["reading"]["holes"] =
            serde_json::json!({"depth": 30.0, "margin": 3, "fill_radius": 12, "restore": 5.0});
        std::fs::write(&path, serde_json::to_vec(&card).unwrap()).unwrap();
        let guard = ModelBundle::open(temp.path()).unwrap().card().reading.holes;
        assert_eq!(
            guard,
            Some(HoleGuard {
                depth: 30.0,
                margin: 3,
                fill_radius: 12,
                restore: 5.0
            })
        );
        for (field, value) in [
            ("restore", serde_json::json!(0.0)),
            ("margin", serde_json::json!(usize::MAX)),
            ("fill_radius", serde_json::json!(100_000)),
            ("kind", serde_json::json!("median")),
        ] {
            let mut bad = card.clone();
            bad["reading"]["holes"][field] = value;
            std::fs::write(&path, serde_json::to_vec(&bad).unwrap()).unwrap();
            assert!(
                matches!(
                    ModelBundle::open(temp.path()),
                    Err(ModelError::ModelContract(_))
                ),
                "{field}"
            );
        }
    }

    #[test]
    fn a_reading_this_build_does_not_know_is_refused() {
        let temp = tempfile::tempdir().unwrap();
        write_bundle(temp.path(), "noise-test", "constant");
        let path = temp.path().join(CARD_FILE);
        let mut card: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        card["reading"]["stars"] = serde_json::json!({"keep": true});
        std::fs::write(&path, serde_json::to_vec(&card).unwrap()).unwrap();
        let error = ModelBundle::open(temp.path()).unwrap_err();
        assert!(error.to_string().contains("stars"), "{error}");
    }

    #[test]
    fn a_parity_list_limits_what_runs_unasked() {
        let temp = tempfile::tempdir().unwrap();
        write_bundle(temp.path(), "noise-test", "constant");
        let mut card = ModelBundle::open(temp.path()).unwrap().card.clone();
        assert!(card.allows(Backend::Cuda), "an empty list allows every one");
        card.parity = vec!["cpu".to_string()];
        assert!(card.allows(Backend::Cpu));
        assert!(!card.allows(Backend::Cuda));
    }
}
