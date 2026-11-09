use crate::backend::{Backend, Precision};
use crate::error::ModelError;
use crate::tiles::Tiling;
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
}

impl fmt::Display for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Denoise => "denoise",
        })
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
}

impl fmt::Display for InputPlane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Image => "image",
            Self::Noise => "noise",
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
        for graph in &self.graphs {
            let file = Path::new(&graph.file);
            if file.file_name() != Some(file.as_os_str()) {
                return refuse(format!("{:?} is not a file in the bundle", graph.file));
            }
        }
        Ok(())
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
            .try_for_each(|graph| self.verify_graph(graph))
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
