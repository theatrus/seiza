//! XISF content carried from a read to a later write.

use crate::{Unit, XisfElement, XisfError, parse_block, with_block};
use std::collections::BTreeMap;
use std::io::{Read, Seek};

/// Image attributes that describe the stored pixels. A writer derives them
/// from the pixels it writes, so they are not carried.
pub(crate) const PIXEL_ATTRIBUTES: [&str; 10] = [
    "geometry",
    "sampleFormat",
    "bounds",
    "colorSpace",
    "pixelStorage",
    "location",
    "compression",
    "subblocks",
    "checksum",
    "byteOrder",
];

/// Metadata properties that describe how a unit was encoded. A writer
/// states its own.
pub(crate) const ENCODING_PROPERTIES: [&str; 10] = [
    "XISF:CreationTime",
    "XISF:CreatorApplication",
    "XISF:CreatorModule",
    "XISF:CreatorOS",
    "XISF:BlockAlignmentSize",
    "XISF:ChecksumAlgorithms",
    "XISF:CompressionCodecs",
    "XISF:CompressionLevel",
    "XISF:MaxInlineBlockSize",
    "XISF:OutputHints",
];

/// Root attributes that every writer sets for itself.
const ROOT_ATTRIBUTES: [&str; 5] = ["version", "xmlns", "xmlns:xsi", "xsi:schemaLocation", "id"];

/// Everything an XISF file says about one image and its unit beyond the
/// pixels, kept so a later write can carry it all, whether or not this crate
/// understands it.
///
/// Pass it to the writer through [`WriteOptions::metadata`]. The writer
/// states the pixel format, geometry, storage and encoding itself, and drops
/// what the new pixels would make false: FITS scaling keywords, the
/// astrometric solution when the width or height changed, and the color
/// filter array when the channel count changed. Anything else that no
/// longer holds after processing, such as an astrometric solution after a
/// registration that kept the size, is for the caller to remove.
///
/// [`WriteOptions::metadata`]: crate::WriteOptions::metadata
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct XisfMetadata {
    /// Attributes of the `Image` element other than those that describe the
    /// stored pixels: `id`, `uuid`, `imageType`, `offset`, `orientation` and
    /// any other.
    pub image_attributes: BTreeMap<String, String>,
    /// Every child element of the `Image` element, in order, with each
    /// `Reference` replaced by the element it names and attached or external
    /// blocks loaded into [`XisfElement::block`].
    pub image_elements: Vec<XisfElement>,
    /// The `Property` elements of the unit's `Metadata` element.
    pub unit_properties: Vec<XisfElement>,
    /// Other children of the root element, such as extension elements in
    /// other namespaces.
    pub root_elements: Vec<XisfElement>,
    /// Attributes of the root element other than the version and the
    /// standard namespaces, such as declarations of extension namespaces.
    pub root_attributes: BTreeMap<String, String>,
    /// Width, height and channel count of the image the metadata came from.
    pub source_geometry: (usize, usize, usize),
    /// Elements left out because their data blocks could not be read, with
    /// the reason.
    pub dropped: Vec<String>,
}

impl XisfMetadata {
    /// The image property with a given identifier.
    pub fn property(&self, id: &str) -> Option<&XisfElement> {
        self.image_elements.iter().find(|element| {
            element.local_name() == "Property" && element.attribute("id") == Some(id)
        })
    }

    /// Remove every image property whose identifier starts with `prefix`,
    /// such as `"AstrometricSolution:"` after a geometric transformation.
    pub fn remove_properties(&mut self, prefix: &str) {
        self.image_elements.retain(|element| {
            element.local_name() != "Property"
                || !element
                    .attribute("id")
                    .is_some_and(|id| id.starts_with(prefix))
        });
    }
}

/// The unit-level part of [`XisfMetadata`], shared by every image of a file.
#[derive(Clone, Debug, Default)]
pub(crate) struct UnitMetadata {
    properties: Vec<XisfElement>,
    root_elements: Vec<XisfElement>,
    root_attributes: BTreeMap<String, String>,
}

impl UnitMetadata {
    pub(crate) fn from_root(root: &XisfElement) -> Self {
        let properties = root
            .children
            .iter()
            .filter(|child| child.local_name() == "Metadata")
            .flat_map(|metadata| &metadata.children)
            .filter(|child| child.local_name() == "Property")
            .cloned()
            .collect();
        // Images and the Metadata element are written anew, and shared
        // elements travel inside the images that reference them.
        let root_elements = root
            .children
            .iter()
            .filter(|child| {
                !matches!(child.local_name(), "Image" | "Metadata")
                    && child.attribute("uid").is_none()
            })
            .cloned()
            .collect();
        let root_attributes = root
            .attributes
            .iter()
            .filter(|(name, _)| !ROOT_ATTRIBUTES.contains(&name.as_str()))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect();
        Self {
            properties,
            root_elements,
            root_attributes,
        }
    }
}

/// Build the metadata of an image, loading every attached or external block
/// its preserved elements locate. An element whose block cannot be read is
/// left out and named in [`XisfMetadata::dropped`].
pub(crate) fn collect(
    reader: &mut (impl Read + Seek),
    unit: &Unit,
    unit_metadata: &UnitMetadata,
    image_attributes: &BTreeMap<String, String>,
    image_elements: &[XisfElement],
    source_geometry: (usize, usize, usize),
) -> XisfMetadata {
    let mut metadata = XisfMetadata {
        image_attributes: image_attributes
            .iter()
            .filter(|(name, _)| !PIXEL_ATTRIBUTES.contains(&name.as_str()))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect(),
        root_attributes: unit_metadata.root_attributes.clone(),
        source_geometry,
        ..XisfMetadata::default()
    };
    let mut load = |elements: Vec<XisfElement>, dropped: &mut Vec<String>| {
        elements
            .into_iter()
            .filter_map(
                |mut element| match load_blocks(reader, unit, &mut element) {
                    Ok(()) => Some(element),
                    Err(error) => {
                        let what = element
                            .attribute("id")
                            .map_or_else(|| element.name.clone(), str::to_string);
                        dropped.push(format!("{what}: {error}"));
                        None
                    }
                },
            )
            .collect::<Vec<_>>()
    };
    // The image's own Data child holds its pixels, which are written anew.
    let image_elements = image_elements
        .iter()
        .filter(|element| element.local_name() != "Data")
        .cloned()
        .collect();
    metadata.image_elements = load(image_elements, &mut metadata.dropped);
    metadata.unit_properties = load(unit_metadata.properties.clone(), &mut metadata.dropped);
    metadata.root_elements = load(unit_metadata.root_elements.clone(), &mut metadata.dropped);
    metadata
}

/// Load the stored bytes of every attached or external block in an element
/// and its descendants.
fn load_blocks(
    reader: &mut (impl Read + Seek),
    unit: &Unit,
    element: &mut XisfElement,
) -> Result<(), XisfError> {
    let mut result = Ok(());
    element.visit_mut(&mut |element| {
        if result.is_err() || !element.has_outside_block() {
            return;
        }
        result = parse_block(&element.attributes, "", None, unit, None)
            .and_then(|block| {
                with_block(reader, &block, |stored| {
                    let mut bytes = Vec::new();
                    stored.read_to_end(&mut bytes)?;
                    Ok(bytes)
                })
            })
            .map(|bytes| element.block = Some(bytes));
    });
    result
}
