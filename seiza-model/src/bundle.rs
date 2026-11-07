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

/// How a model reads an image: everything that must match its training.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
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
    /// The tiling for a tile of `tile` pixels, or this card's own.
    pub fn tiling(&self, tile: Option<usize>) -> Tiling {
        Tiling {
            tile: tile.unwrap_or(self.tile),
            overlap: 2 * self.margin + self.ramp,
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
    /// The graph's input tensor.
    pub input: String,
    /// The planes that tensor holds.
    pub input_channels: usize,
    /// The graph's output tensor.
    pub output: String,
    /// The graph's free dimensions, each set to the tile's side.
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
        if self.input_channels != 2 {
            return refuse(format!(
                "a denoise graph reads 2 planes, not {}",
                self.input_channels
            ));
        }
        if self.graphs.is_empty() {
            return refuse(format!("{} names no graph", self.name));
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
        Ok(Self { root, card })
    }

    /// The bundle's directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The bundle's card.
    pub fn card(&self) -> &ModelCard {
        &self.card
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
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
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
              "input": "planes", "input_channels": 2, "output": "residual",
              "dimensions": ["height", "width"],
              "reading": {{"noise_map": "{noise_map}", "noise_plane": "asinh", "sky": "flat",
                           "sky_block": 32, "flat_block": 256, "image_map_scale": 1.13}},
              "tiling": {{"tile": 512, "margin": 122, "ramp": 32, "multiple": 16}}
            }}"#
        );
        std::fs::write(root.join(CARD_FILE), card).unwrap();
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
