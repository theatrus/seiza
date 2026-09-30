//! Astrometric solutions stored in the `AstrometricSolution` property
//! namespace of XISF 1.0, Revision 1.
//!
//! This module reads a solution into typed layers and applies the rules for
//! which layers are usable. It does not evaluate the transformation.

use crate::{ByteOrder, XisfElement, XisfError, XisfMetadata, element_block_bytes};
use std::collections::BTreeMap;

const PREFIX: &str = "AstrometricSolution:";

/// The projection systems of the projection vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionSystem {
    Gnomonic,
    Stereographic,
    ZenithalEqualArea,
    Orthographic,
    PlateCarree,
    Mercator,
    HammerAitoff,
}

impl ProjectionSystem {
    fn parse(identifier: &str) -> Option<Self> {
        Some(match identifier {
            "Gnomonic" => Self::Gnomonic,
            "Stereographic" => Self::Stereographic,
            "ZenithalEqualArea" => Self::ZenithalEqualArea,
            "Orthographic" => Self::Orthographic,
            "PlateCarree" => Self::PlateCarree,
            "Mercator" => Self::Mercator,
            "HammerAitoff" => Self::HammerAitoff,
            _ => return None,
        })
    }

    pub fn is_zenithal(self) -> bool {
        matches!(
            self,
            Self::Gnomonic | Self::Stereographic | Self::ZenithalEqualArea | Self::Orthographic
        )
    }
}

/// Layer 1: the projection and the linear solution. Always present in a
/// usable solution.
#[derive(Clone, Debug, PartialEq)]
pub struct Projection {
    pub system: ProjectionSystem,
    /// (α₀, δ₀) of the projection reference point, in degrees.
    pub reference_celestial: [f64; 2],
    /// (x₀, y₀), the image coordinates of the projection plane origin.
    pub reference_image: [f64; 2],
    /// Degrees per pixel, row-major: (u, v) = M (x − x₀, y − y₀).
    pub linear: [[f64; 2]; 2],
    /// (φ₀, θ₀) of the reference point, in degrees, with the WCS default
    /// filled in when the file gives none.
    pub reference_native: [f64; 2],
    /// (φₚ, θₚ) of the celestial pole, when the file gives it. `None` means
    /// the WCS default.
    pub celestial_pole_native: Option<[f64; 2]>,
    /// `ICRS` unless the file says otherwise. An unrecognized identifier is
    /// kept; it only prevents converting to another reference system.
    pub celestial_reference_system: String,
}

/// Layer 2: projective transformations in both directions, as 3×3
/// row-major matrices on homogeneous coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectiveTransformation {
    pub image_to_projection: [[f64; 3]; 3],
    pub projection_to_image: [[f64; 3]; 3],
}

/// A radial basis function of the basis function vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BasisFunction {
    ThinPlateSpline,
    VariableOrder,
    Gaussian,
    Multiquadric,
    InverseMultiquadric,
    InverseQuadratic,
}

impl BasisFunction {
    fn parse(identifier: &str) -> Option<Self> {
        Some(match identifier {
            "ThinPlateSpline" => Self::ThinPlateSpline,
            "VariableOrder" => Self::VariableOrder,
            "Gaussian" => Self::Gaussian,
            "Multiquadric" => Self::Multiquadric,
            "InverseMultiquadric" => Self::InverseMultiquadric,
            "InverseQuadratic" => Self::InverseQuadratic,
            _ => return None,
        })
    }

    pub fn has_shape_parameter(self) -> bool {
        !matches!(self, Self::ThinPlateSpline | Self::VariableOrder)
    }
}

/// One scalar surface spline.
#[derive(Clone, Debug, PartialEq)]
pub struct Spline {
    /// (x₀, y₀, r₀): the evaluation point is q = r₀ (p − p₀).
    pub normalization: [f64; 3],
    /// Nodes, already normalized.
    pub nodes: Vec<[f64; 2]>,
    /// Radial coefficients in node order, then polynomial coefficients.
    pub coefficients: Vec<f64>,
    pub shape_parameter: Option<f64>,
}

/// The X and Y component splines of a term, with node sharing already
/// applied: a Y component that shares the X nodes carries copies of them.
#[derive(Clone, Debug, PartialEq)]
pub struct SplinePair {
    pub x: Spline,
    pub y: Spline,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LocalTerm {
    /// Center of the support disc, in source coordinates.
    pub center: [f64; 2],
    /// Radius of the support disc, in source units.
    pub radius: f64,
    pub splines: SplinePair,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FallbackTerm {
    /// The coverage threshold t₀.
    pub threshold: f64,
    pub splines: SplinePair,
}

/// The distortion model of one direction of the image-plane step.
#[derive(Clone, Debug, PartialEq)]
pub struct DistortionModel {
    pub basis_function: BasisFunction,
    /// The order m: the polynomial degree plus one.
    pub order: u32,
    pub polynomial: bool,
    pub global: Option<SplinePair>,
    pub local: Vec<LocalTerm>,
    pub fallback: Option<FallbackTerm>,
}

/// Layer 3: distortion models in both directions.
#[derive(Clone, Debug, PartialEq)]
pub struct Distortion {
    pub image_to_projection: DistortionModel,
    pub projection_to_image: DistortionModel,
}

/// Layer 4: what a solver needs to re-solve or audit the solution. Every
/// part is optional.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Provenance {
    pub control_points_celestial: Option<Vec<[f64; 2]>>,
    pub control_points_image: Option<Vec<[f64; 2]>>,
    pub weights: Option<Vec<f64>>,
    pub rejected: Option<Vec<i64>>,
    pub catalog: Option<String>,
    pub creation_time: Option<String>,
    pub creator_application: Option<String>,
    pub creator_module: Option<String>,
    pub creator_os: Option<String>,
}

/// An astrometric solution, read into the layers this crate could use.
#[derive(Clone, Debug, PartialEq)]
pub struct AstrometricSolution {
    /// The `major.minor` revision of the namespace the solution follows.
    pub version: (u32, u32),
    pub projection: Projection,
    /// `None` when absent or unusable; see [`Self::unavailable`].
    pub projective: Option<ProjectiveTransformation>,
    /// `None` when absent or unusable, and always `None` without layer 2.
    pub distortion: Option<Distortion>,
    pub provenance: Provenance,
    /// Why a layer present in the file was not usable.
    pub unavailable: Vec<String>,
}

impl XisfMetadata {
    /// The image's astrometric solution, read into typed layers.
    ///
    /// `Ok(None)` means the image has no `AstrometricSolution` properties.
    /// A solution whose first layer is unusable, or whose major revision
    /// this crate does not know, is an error: the specification forbids
    /// interpreting any part of it. Higher layers that are unusable fall
    /// back as the specification requires, with the reason in
    /// [`AstrometricSolution::unavailable`].
    ///
    /// Solutions written by PixInsight before Revision 1 use the
    /// `PCL:AstrometricSolution` namespace, which is not read here.
    pub fn astrometric_solution(&self) -> Result<Option<AstrometricSolution>, XisfError> {
        let properties = Properties::new(self);
        if properties.elements.is_empty() {
            return Ok(None);
        }
        let version = properties
            .text("Version")
            .map_err(XisfError::Malformed)?
            .ok_or_else(|| XisfError::Malformed("astrometric solution has no Version".into()))?;
        let version = parse_version(&version).ok_or_else(|| {
            XisfError::Malformed(format!("invalid astrometric solution version {version:?}"))
        })?;
        if version.0 != 1 {
            return Err(XisfError::Unsupported(format!(
                "astrometric solution version {}.{}",
                version.0, version.1
            )));
        }
        let projection = projection(&properties).map_err(|reason| {
            XisfError::Malformed(format!("astrometric solution layer 1: {reason}"))
        })?;
        let mut unavailable = Vec::new();
        let projective = match projective(&properties) {
            Ok(layer) => layer,
            Err(reason) => {
                unavailable.push(format!("layer 2: {reason}"));
                None
            }
        };
        let distortion = match (&projective, distortion(&properties)) {
            (_, Ok(None)) => None,
            (None, Ok(Some(_))) => {
                unavailable.push("layer 3: layer 2 is unavailable".into());
                None
            }
            (Some(_), Ok(Some(layer))) => Some(layer),
            (_, Err(reason)) => {
                unavailable.push(format!("layer 3: {reason}"));
                None
            }
        };
        let provenance = provenance(&properties, &mut unavailable);
        Ok(Some(AstrometricSolution {
            version,
            projection,
            projective,
            distortion,
            provenance,
            unavailable,
        }))
    }
}

fn parse_version(version: &str) -> Option<(u32, u32)> {
    let (major, minor) = version.trim().split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}

type Layer<T> = Result<T, String>;

/// The `AstrometricSolution` properties of an image, by identifier without
/// the namespace prefix.
struct Properties<'a> {
    elements: BTreeMap<&'a str, &'a XisfElement>,
}

/// A decoded vector or matrix property.
struct Array {
    values: Vec<f64>,
    rows: usize,
    columns: usize,
}

impl<'a> Properties<'a> {
    fn new(metadata: &'a XisfMetadata) -> Self {
        let elements = metadata
            .image_elements
            .iter()
            .filter(|element| element.local_name() == "Property")
            .filter_map(|element| {
                let id = element.attribute("id")?.trim().strip_prefix(PREFIX)?;
                Some((id, element))
            })
            .collect();
        Self { elements }
    }

    fn has(&self, id: &str) -> bool {
        self.elements.contains_key(id)
    }

    fn text(&self, id: &str) -> Layer<Option<String>> {
        let Some(element) = self.elements.get(id) else {
            return Ok(None);
        };
        if element.attribute("location").is_some() {
            let (bytes, _) =
                element_block_bytes(element, None).map_err(|error| format!("{id}: {error}"))?;
            return String::from_utf8(bytes)
                .map(Some)
                .map_err(|_| format!("{id} is not UTF-8"));
        }
        Ok(Some(
            element
                .attribute("value")
                .map_or_else(|| element.text.clone(), str::to_string),
        ))
    }

    fn scalar(&self, id: &str) -> Layer<Option<f64>> {
        let Some(element) = self.elements.get(id) else {
            return Ok(None);
        };
        let value = element.attribute("value").unwrap_or(&element.text).trim();
        let parsed = match element.attribute("type").map(str::trim) {
            Some("Boolean") => match value {
                "true" | "1" => Some(1.0),
                "false" | "0" => Some(0.0),
                _ => None,
            },
            _ => parse_number(value),
        };
        parsed
            .map(Some)
            .ok_or_else(|| format!("{id} has an invalid value {value:?}"))
    }

    fn array(&self, id: &str) -> Layer<Option<Array>> {
        let Some(element) = self.elements.get(id) else {
            return Ok(None);
        };
        let type_name = element.attribute("type").unwrap_or("").trim();
        let (item, shape) = array_type(type_name)
            .ok_or_else(|| format!("{id} has type {type_name:?}, not a real vector or matrix"))?;
        let dimension = |name: &str| {
            element
                .attribute(name)
                .and_then(|value| value.trim().parse::<usize>().ok())
                .ok_or_else(|| format!("{id} has no valid {name} attribute"))
        };
        let (rows, columns) = match shape {
            Shape::Vector => (dimension("length")?, 1),
            Shape::Matrix => (dimension("rows")?, dimension("columns")?),
        };
        let count = rows
            .checked_mul(columns)
            .ok_or_else(|| format!("{id} is too large"))?;
        let expected = count
            .checked_mul(item.bytes())
            .ok_or_else(|| format!("{id} is too large"))?;
        let (bytes, order) = element_block_bytes(element, Some(expected))
            .map_err(|error| format!("{id}: {error}"))?;
        let values = bytes
            .chunks_exact(item.bytes())
            .map(|bytes| item.decode(bytes, order))
            .collect();
        Ok(Some(Array {
            values,
            rows,
            columns,
        }))
    }

    fn vector(&self, id: &str, length: Option<usize>) -> Layer<Option<Vec<f64>>> {
        let Some(array) = self.array(id)? else {
            return Ok(None);
        };
        if array.columns != 1 || length.is_some_and(|length| length != array.rows) {
            return Err(format!("{id} has an inconsistent length"));
        }
        Ok(Some(array.values))
    }

    fn matrix(&self, id: &str, columns: usize, rows: Option<usize>) -> Layer<Option<Array>> {
        let Some(array) = self.array(id)? else {
            return Ok(None);
        };
        if array.columns != columns || rows.is_some_and(|rows| rows != array.rows) {
            return Err(format!("{id} has inconsistent dimensions"));
        }
        Ok(Some(array))
    }

    fn required<T>(&self, id: &str, value: Layer<Option<T>>) -> Layer<T> {
        value?.ok_or_else(|| format!("{id} is missing"))
    }
}

fn parse_number(value: &str) -> Option<f64> {
    let (negative, digits) = match value.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, value.strip_prefix('+').unwrap_or(value)),
    };
    let radix = |prefix: &str, radix: u32| {
        digits
            .strip_prefix(prefix)
            .and_then(|digits| u128::from_str_radix(digits, radix).ok())
            .map(|magnitude| {
                if negative {
                    -(magnitude as f64)
                } else {
                    magnitude as f64
                }
            })
    };
    radix("0x", 16)
        .or_else(|| radix("0b", 2))
        .or_else(|| radix("0o", 8))
        .or_else(|| value.parse::<f64>().ok())
}

enum Shape {
    Vector,
    Matrix,
}

#[derive(Clone, Copy)]
enum Item {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
}

impl Item {
    fn bytes(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }

    fn decode(self, bytes: &[u8], order: ByteOrder) -> f64 {
        macro_rules! read {
            ($type:ty) => {{
                let bytes = bytes.try_into().unwrap();
                match order {
                    ByteOrder::Little => <$type>::from_le_bytes(bytes),
                    ByteOrder::Big => <$type>::from_be_bytes(bytes),
                }
            }};
        }
        match self {
            Self::I8 => f64::from(bytes[0] as i8),
            Self::U8 => f64::from(bytes[0]),
            Self::I16 => f64::from(read!(i16)),
            Self::U16 => f64::from(read!(u16)),
            Self::I32 => f64::from(read!(i32)),
            Self::U32 => f64::from(read!(u32)),
            Self::I64 => read!(i64) as f64,
            Self::U64 => read!(u64) as f64,
            Self::F32 => f64::from(read!(f32)),
            Self::F64 => read!(f64),
        }
    }
}

/// The item type and shape of a real vector or matrix property type.
fn array_type(type_name: &str) -> Option<(Item, Shape)> {
    let type_name = match type_name {
        "ByteArray" => "UI8Vector",
        "IVector" => "I32Vector",
        "UIVector" => "UI32Vector",
        "Vector" => "F64Vector",
        "ByteMatrix" => "UI8Matrix",
        "IMatrix" => "I32Matrix",
        "UIMatrix" => "UI32Matrix",
        "Matrix" => "F64Matrix",
        other => other,
    };
    let (item, shape) = if let Some(item) = type_name.strip_suffix("Vector") {
        (item, Shape::Vector)
    } else {
        (type_name.strip_suffix("Matrix")?, Shape::Matrix)
    };
    let item = match item {
        "I8" => Item::I8,
        "UI8" => Item::U8,
        "I16" => Item::I16,
        "UI16" => Item::U16,
        "I32" => Item::I32,
        "UI32" => Item::U32,
        "I64" => Item::I64,
        "UI64" => Item::U64,
        "F32" => Item::F32,
        "F64" => Item::F64,
        _ => return None,
    };
    Some((item, shape))
}

fn pair(values: &[f64]) -> [f64; 2] {
    [values[0], values[1]]
}

fn pairs(array: &Array) -> Vec<[f64; 2]> {
    array.values.chunks_exact(2).map(pair).collect()
}

fn projection(properties: &Properties<'_>) -> Layer<Projection> {
    let system_name =
        properties.required("ProjectionSystem", properties.text("ProjectionSystem"))?;
    let system = ProjectionSystem::parse(system_name.trim())
        .ok_or_else(|| format!("unrecognized projection system {system_name:?}"))?;
    let vector2 = |id: &str| -> Layer<Option<[f64; 2]>> {
        Ok(properties.vector(id, Some(2))?.map(|values| pair(&values)))
    };
    let linear = properties.required(
        "LinearTransformationMatrix",
        properties.matrix("LinearTransformationMatrix", 2, Some(2)),
    )?;
    let default_native = if system.is_zenithal() {
        [0.0, 90.0]
    } else {
        [0.0, 0.0]
    };
    Ok(Projection {
        system,
        reference_celestial: properties.required(
            "ReferenceCelestialCoordinates",
            vector2("ReferenceCelestialCoordinates"),
        )?,
        reference_image: properties.required(
            "ReferenceImageCoordinates",
            vector2("ReferenceImageCoordinates"),
        )?,
        linear: [pair(&linear.values[..2]), pair(&linear.values[2..])],
        reference_native: vector2("ReferenceNativeCoordinates")?.unwrap_or(default_native),
        celestial_pole_native: vector2("CelestialPoleNativeCoordinates")?,
        celestial_reference_system: properties
            .text("CelestialReferenceSystem")?
            .map_or_else(|| "ICRS".into(), |system| system.trim().to_string()),
    })
}

fn matrix3(array: &Array) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| std::array::from_fn(|column| array.values[row * 3 + column]))
}

fn projective(properties: &Properties<'_>) -> Layer<Option<ProjectiveTransformation>> {
    const FORWARD: &str = "ProjectiveTransformation:ImageToProjection";
    const INVERSE: &str = "ProjectiveTransformation:ProjectionToImage";
    match (
        properties.matrix(FORWARD, 3, Some(3))?,
        properties.matrix(INVERSE, 3, Some(3))?,
    ) {
        (None, None) => Ok(None),
        (Some(forward), Some(inverse)) => Ok(Some(ProjectiveTransformation {
            image_to_projection: matrix3(&forward),
            projection_to_image: matrix3(&inverse),
        })),
        _ => Err("only one direction of the projective transformation is present".into()),
    }
}

fn distortion(properties: &Properties<'_>) -> Layer<Option<Distortion>> {
    const FORWARD: &str = "DistortionModel:ImageToProjection:";
    const INVERSE: &str = "DistortionModel:ProjectionToImage:";
    let present = |prefix: &str| properties.elements.keys().any(|id| id.starts_with(prefix));
    match (present(FORWARD), present(INVERSE)) {
        (false, false) => Ok(None),
        (true, true) => Ok(Some(Distortion {
            image_to_projection: distortion_model(properties, FORWARD)?,
            projection_to_image: distortion_model(properties, INVERSE)?,
        })),
        _ => Err("only one direction of the distortion model is present".into()),
    }
}

fn distortion_model(properties: &Properties<'_>, prefix: &str) -> Layer<DistortionModel> {
    let id = |name: &str| format!("{prefix}{name}");
    let basis_name =
        properties.required(&id("BasisFunction"), properties.text(&id("BasisFunction")))?;
    let basis_function = BasisFunction::parse(basis_name.trim())
        .ok_or_else(|| format!("unrecognized basis function {basis_name:?}"))?;
    let order = properties.required(&id("Order"), properties.scalar(&id("Order")))?;
    let minimum = if basis_function == BasisFunction::VariableOrder {
        3.0
    } else {
        2.0
    };
    if order.fract() != 0.0 || order < minimum || order > 1000.0 {
        return Err(format!("invalid order {order} for {basis_function:?}"));
    }
    let order = order as u32;
    let polynomial = properties
        .scalar(&id("Polynomial"))?
        .is_none_or(|value| value != 0.0);
    if !polynomial && !basis_function.has_shape_parameter() {
        return Err(format!("{basis_function:?} requires a polynomial part"));
    }
    let polynomial_terms = if polynomial {
        (order * (order + 1) / 2) as usize
    } else {
        0
    };
    let terms = properties.required(&id("Terms"), properties.text(&id("Terms")))?;
    let mut kinds = Vec::new();
    for kind in terms
        .split('\n')
        .map(str::trim)
        .filter(|kind| !kind.is_empty())
    {
        if !matches!(kind, "Global" | "Local" | "Fallback") {
            return Err(format!("unrecognized term kind {kind:?}"));
        }
        if kinds.contains(&kind) {
            return Err(format!("term kind {kind} is listed twice"));
        }
        kinds.push(kind);
    }
    if kinds.contains(&"Fallback") && !kinds.contains(&"Local") {
        return Err("a Fallback term requires Local terms".into());
    }
    let record = SplineRecord {
        properties,
        basis_function,
        polynomial_terms,
    };
    let global = if kinds.contains(&"Global") {
        Some(record.pair(&id("Global:"))?)
    } else {
        None
    };
    let local = if kinds.contains(&"Local") {
        record.local(&id("Local:"))?
    } else {
        Vec::new()
    };
    let fallback = if kinds.contains(&"Fallback") {
        let threshold = properties.required(
            &id("Fallback:Threshold"),
            properties.scalar(&id("Fallback:Threshold")),
        )?;
        if threshold.is_nan() || threshold <= 0.0 {
            return Err(format!("invalid Fallback threshold {threshold}"));
        }
        Some(FallbackTerm {
            threshold,
            splines: record.pair(&id("Fallback:"))?,
        })
    } else {
        None
    };
    Ok(DistortionModel {
        basis_function,
        order,
        polynomial,
        global,
        local,
        fallback,
    })
}

/// Reads the spline records of one direction.
struct SplineRecord<'p, 'a> {
    properties: &'p Properties<'a>,
    basis_function: BasisFunction,
    polynomial_terms: usize,
}

impl SplineRecord<'_, '_> {
    fn shape_parameter(&self, value: Option<f64>, id: &str) -> Layer<Option<f64>> {
        match (self.basis_function.has_shape_parameter(), value) {
            (true, Some(shape)) if shape > 0.0 => Ok(Some(shape)),
            (true, _) => Err(format!("{id} is missing or not positive")),
            (false, None) => Ok(None),
            (false, Some(_)) => Err(format!("{id} is given for {:?}", self.basis_function)),
        }
    }

    /// The spline record of a Global or Fallback term.
    fn pair(&self, prefix: &str) -> Layer<SplinePair> {
        let p = self.properties;
        let id = |name: &str| format!("{prefix}{name}");
        let component = |axis: &str, shared: Option<&Spline>| -> Layer<Spline> {
            let coefficients_id = id(&format!("{axis}:Coefficients"));
            let coefficients = p.required(&coefficients_id, p.vector(&coefficients_id, None))?;
            let nodes_id = id(&format!("{axis}:Nodes"));
            let (normalization, nodes, shape) = match (p.matrix(&nodes_id, 2, None)?, shared) {
                (None, Some(shared)) => {
                    for name in ["Normalization", "ShapeParameter"] {
                        if p.has(&id(&format!("{axis}:{name}"))) {
                            return Err(format!("{axis}:{name} is given without {axis}:Nodes"));
                        }
                    }
                    (
                        shared.normalization,
                        shared.nodes.clone(),
                        shared.shape_parameter,
                    )
                }
                (None, None) => return Err(format!("{nodes_id} is missing")),
                (Some(nodes), _) => {
                    let normalization_id = id(&format!("{axis}:Normalization"));
                    let normalization =
                        p.required(&normalization_id, p.vector(&normalization_id, Some(3)))?;
                    let shape_id = id(&format!("{axis}:ShapeParameter"));
                    let shape = self.shape_parameter(p.scalar(&shape_id)?, &shape_id)?;
                    (
                        [normalization[0], normalization[1], normalization[2]],
                        pairs(&nodes),
                        shape,
                    )
                }
            };
            if coefficients.len() != nodes.len() + self.polynomial_terms {
                return Err(format!("{coefficients_id} has an inconsistent length"));
            }
            Ok(Spline {
                normalization,
                nodes,
                coefficients,
                shape_parameter: shape,
            })
        };
        let x = component("X", None)?;
        let y = component("Y", Some(&x))?;
        Ok(SplinePair { x, y })
    }

    /// The Local terms of a direction.
    fn local(&self, prefix: &str) -> Layer<Vec<LocalTerm>> {
        let p = self.properties;
        let id = |name: &str| format!("{prefix}{name}");
        let centers = p.required(&id("Center"), p.matrix(&id("Center"), 2, None))?;
        let count = centers.rows;
        let radii = p.required(&id("Radius"), p.vector(&id("Radius"), Some(count)))?;
        let x = self.packed(prefix, "X", count, None)?;
        let y = self.packed(prefix, "Y", count, Some(&x))?;
        Ok(x.into_iter()
            .zip(y)
            .enumerate()
            .map(|(term, (x, y))| LocalTerm {
                center: pair(&centers.values[term * 2..]),
                radius: radii[term],
                splines: SplinePair { x, y },
            })
            .collect())
    }

    /// The splines of one component of all Local terms, unpacked.
    fn packed(
        &self,
        prefix: &str,
        axis: &str,
        count: usize,
        shared: Option<&[Spline]>,
    ) -> Layer<Vec<Spline>> {
        let p = self.properties;
        let id = |name: &str| format!("{prefix}{axis}:{name}");
        let q = self.polynomial_terms;
        let coefficients = p.required(&id("Coefficients"), p.vector(&id("Coefficients"), None))?;
        let layout = match (p.matrix(&id("Nodes"), 2, None)?, shared) {
            (None, Some(shared)) => {
                for name in ["Normalization", "NodeOffsets", "ShapeParameter"] {
                    if p.has(&id(name)) {
                        return Err(format!("{} is given without {}", id(name), id("Nodes")));
                    }
                }
                shared
                    .iter()
                    .map(|spline| {
                        (
                            spline.normalization,
                            spline.nodes.clone(),
                            spline.shape_parameter,
                        )
                    })
                    .collect::<Vec<_>>()
            }
            (None, None) => return Err(format!("{} is missing", id("Nodes"))),
            (Some(nodes), _) => {
                let normalization = p.required(
                    &id("Normalization"),
                    p.matrix(&id("Normalization"), 3, Some(count)),
                )?;
                let offsets = p.required(
                    &id("NodeOffsets"),
                    p.vector(&id("NodeOffsets"), Some(count + 1)),
                )?;
                let offsets = offsets
                    .iter()
                    .map(|&offset| {
                        (offset >= 0.0 && offset.fract() == 0.0).then_some(offset as usize)
                    })
                    .collect::<Option<Vec<_>>>()
                    .filter(|offsets| {
                        offsets[0] == 0
                            && offsets.windows(2).all(|pair| pair[0] <= pair[1])
                            && offsets[count] == nodes.rows
                    })
                    .ok_or_else(|| format!("{} are inconsistent", id("NodeOffsets")))?;
                let shapes = match (
                    self.basis_function.has_shape_parameter(),
                    p.vector(&id("ShapeParameter"), Some(count))?,
                ) {
                    (true, Some(shapes)) if shapes.iter().all(|&shape| shape > 0.0) => {
                        shapes.into_iter().map(Some).collect()
                    }
                    (false, None) => vec![None; count],
                    _ => {
                        return Err(format!(
                            "{} does not suit {:?}",
                            id("ShapeParameter"),
                            self.basis_function
                        ));
                    }
                };
                let nodes = pairs(&nodes);
                (0..count)
                    .map(|term| {
                        let row = &normalization.values[term * 3..term * 3 + 3];
                        (
                            [row[0], row[1], row[2]],
                            nodes[offsets[term]..offsets[term + 1]].to_vec(),
                            shapes[term],
                        )
                    })
                    .collect()
            }
        };
        let total = layout
            .iter()
            .map(|(_, nodes, _)| nodes.len())
            .sum::<usize>()
            + count * q;
        if coefficients.len() != total {
            return Err(format!("{} has an inconsistent length", id("Coefficients")));
        }
        let mut start = 0;
        Ok(layout
            .into_iter()
            .map(|(normalization, nodes, shape_parameter)| {
                let end = start + nodes.len() + q;
                let spline = Spline {
                    normalization,
                    coefficients: coefficients[start..end].to_vec(),
                    nodes,
                    shape_parameter,
                };
                start = end;
                spline
            })
            .collect())
    }
}

fn provenance(properties: &Properties<'_>, unavailable: &mut Vec<String>) -> Provenance {
    let mut note = |reason: String| unavailable.push(format!("layer 4: {reason}"));
    let mut points = |id: &str| match properties.matrix(id, 2, None) {
        Ok(points) => points.map(|points| pairs(&points)),
        Err(reason) => {
            note(reason);
            None
        }
    };
    let (mut celestial, mut image) = (
        points("ControlPoints:Celestial"),
        points("ControlPoints:Image"),
    );
    let mut provenance = Provenance::default();
    let mut note = |reason: String| unavailable.push(format!("layer 4: {reason}"));
    if let (Some(a), Some(b)) = (&celestial, &image)
        && a.len() != b.len()
    {
        note("control point lists differ in length".into());
        (celestial, image) = (None, None);
    }
    provenance.control_points_celestial = celestial;
    provenance.control_points_image = image;
    provenance.weights = properties.vector("Weights", None).unwrap_or_else(|reason| {
        note(reason);
        None
    });
    provenance.rejected = properties
        .vector("ControlPoints:Rejected", None)
        .map(|rejected| {
            rejected.map(|values| values.into_iter().map(|value| value as i64).collect())
        })
        .unwrap_or_else(|reason| {
            note(reason);
            None
        });
    let mut text = |id: &str| {
        properties.text(id).unwrap_or_else(|reason| {
            note(reason);
            None
        })
    };
    provenance.catalog = text("Catalog");
    provenance.creation_time = text("CreationTime");
    provenance.creator_application = text("CreatorApplication");
    provenance.creator_module = text("CreatorModule");
    provenance.creator_os = text("CreatorOS");
    provenance
}
