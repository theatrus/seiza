//! Deep-sky object and named-star catalogs, and their projection into a
//! solved image for overlays.
//!
//! Binary format `SEIZAOB2` (little-endian): magic, u32 count, then a u32
//! byte length and payload for each object. Payloads begin with the complete
//! `SEIZAOB1` object record, followed by a stable ID, source, aliases, and
//! parent IDs. Length-delimited records let newer writers append fields while
//! older v2 readers safely skip the unknown tail. [`ObjectCatalog::open`]
//! remains backward-compatible with `SEIZAOB1` files.

use crate::wcs::Wcs;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, Write};
use std::path::Path;

const MAGIC_V1: &[u8; 8] = b"SEIZAOB1";
const MAGIC_V2: &[u8; 8] = b"SEIZAOB2";
const MAX_RECORD_BYTES: u32 = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ObjectKind {
    Galaxy = 0,
    OpenCluster = 1,
    GlobularCluster = 2,
    Nebula = 3,
    PlanetaryNebula = 4,
    HiiRegion = 5,
    SupernovaRemnant = 6,
    DarkNebula = 7,
    ClusterWithNebula = 8,
    Star = 9,
    DoubleStar = 10,
    Association = 11,
    Other = 12,
    /// A transient event: supernova, nova, bright AT designation
    Transient = 13,
}

impl ObjectKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Galaxy => "galaxy",
            Self::OpenCluster => "open-cluster",
            Self::GlobularCluster => "globular-cluster",
            Self::Nebula => "nebula",
            Self::PlanetaryNebula => "planetary-nebula",
            Self::HiiRegion => "hii-region",
            Self::SupernovaRemnant => "supernova-remnant",
            Self::DarkNebula => "dark-nebula",
            Self::ClusterWithNebula => "cluster-nebula",
            Self::Star => "star",
            Self::DoubleStar => "double-star",
            Self::Association => "association",
            Self::Other => "other",
            Self::Transient => "transient",
        }
    }

    fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Galaxy,
            1 => Self::OpenCluster,
            2 => Self::GlobularCluster,
            3 => Self::Nebula,
            4 => Self::PlanetaryNebula,
            5 => Self::HiiRegion,
            6 => Self::SupernovaRemnant,
            7 => Self::DarkNebula,
            8 => Self::ClusterWithNebula,
            9 => Self::Star,
            10 => Self::DoubleStar,
            11 => Self::Association,
            13 => Self::Transient,
            _ => Self::Other,
        }
    }
}

/// Identity, hierarchy, and provenance carried by a catalog object.
///
/// IDs are opaque to seiza, but producers should make them stable and
/// source-qualified (for example `openngc:NGC224`). Parent IDs use the same
/// namespace and may point to containing nebulae, galaxies, or clusters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ObjectMetadata {
    /// Stable, source-qualified record ID. Empty for legacy v1 records.
    pub id: String,
    /// Dataset or table that supplied this record.
    pub source: String,
    /// Alternate catalog designations, excluding the primary display name.
    pub aliases: Vec<String>,
    /// Stable IDs of containing catalog objects.
    pub parent_ids: Vec<String>,
    /// Stable IDs assigned to the same object by other source catalogs.
    pub alternate_ids: Vec<String>,
    /// Additional catalogs that contributed aliases or measurements.
    pub alternate_sources: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SkyObject {
    pub kind: ObjectKind,
    /// ICRS degrees
    pub ra: f64,
    pub dec: f64,
    /// Visual magnitude when known
    pub mag: Option<f32>,
    /// Major axis in arcminutes when known
    pub major_arcmin: Option<f32>,
    pub minor_arcmin: Option<f32>,
    /// Position angle of the major axis, degrees east of north
    pub position_angle_deg: Option<f32>,
    /// Catalog designation, e.g. "NGC 7331", "Sh2-101", "M 31"
    pub name: String,
    /// Popular name when one exists, e.g. "Andromeda Galaxy"
    pub common_name: String,
    /// Identity, hierarchy, and provenance. Empty when read from v1.
    pub metadata: ObjectMetadata,
}

/// An object projected into a solved image.
#[derive(Debug, Clone)]
pub struct PlacedObject {
    pub object: SkyObject,
    /// Pixel position of the object center
    pub x: f64,
    pub y: f64,
    /// Ellipse semi-axes in pixels (0 when the size is unknown)
    pub semi_major_px: f64,
    pub semi_minor_px: f64,
    /// Rotation of the major axis in image coordinates, degrees
    /// counter-clockwise from the +x axis
    pub angle_deg: f64,
}

/// A sky region that can be queried without plate-solving an image.
///
/// Polygon vertices must describe a convex region in boundary order. Image
/// footprints returned by [`Wcs::footprint`] satisfy that requirement.
#[derive(Debug, Clone, PartialEq)]
pub enum SkyRegion {
    /// A circular region on the sky.
    Cone {
        /// ICRS degrees.
        center: (f64, f64),
        radius_deg: f64,
    },
    /// A convex spherical polygon, with ICRS vertices in boundary order.
    Polygon { vertices: Vec<(f64, f64)> },
}

/// Ordering applied to results from [`ObjectCatalog::query_region`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ObjectSort {
    /// Heuristic likelihood that an object is a major feature of the field.
    #[default]
    Prominence,
    Size,
    Magnitude,
    Distance,
    Name,
}

/// Filters and output controls for a coordinate-only object query.
#[derive(Debug, Clone)]
pub struct ObjectQuery {
    /// Empty means every object kind.
    pub kinds: Vec<ObjectKind>,
    /// Objects with unknown magnitude are excluded when this is set.
    pub max_mag: Option<f32>,
    /// Objects with unknown angular size are excluded when this is set.
    pub min_major_arcmin: Option<f32>,
    /// Require a populated common-name field, rather than only a designation.
    pub common_name_only: bool,
    /// Include large objects whose catalog extent reaches into the region even
    /// when their center lies outside it.
    pub include_extent_overlaps: bool,
    pub limit: Option<usize>,
    pub sort: ObjectSort,
}

impl Default for ObjectQuery {
    fn default() -> Self {
        Self {
            kinds: Vec::new(),
            max_mag: None,
            min_major_arcmin: None,
            common_name_only: false,
            include_extent_overlaps: true,
            limit: None,
            sort: ObjectSort::Prominence,
        }
    }
}

/// A catalog object associated with a coordinate-only sky region.
#[derive(Debug, Clone, Copy)]
pub struct ObjectHit<'a> {
    pub object: &'a SkyObject,
    /// True when the catalog position itself is inside the region.
    pub center_inside: bool,
    /// True when only the object's catalog extent reaches into the region.
    /// Extents are conservatively approximated by the major-axis radius.
    pub extent_only: bool,
    pub distance_from_center_deg: f64,
    /// A 0..1 heuristic based on angular size relative to the field,
    /// integrated magnitude, common-name availability, and center placement.
    /// It predicts likely prominence; it does not prove pixel visibility.
    pub predicted_prominence: f64,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ObjectQueryError {
    #[error("invalid ICRS coordinate ({ra}, {dec})")]
    InvalidCoordinate { ra: f64, dec: f64 },
    #[error("cone radius must be finite and in the range (0, 180], got {0}")]
    InvalidRadius(f64),
    #[error("a polygon needs at least three vertices, got {0}")]
    TooFewVertices(usize),
    #[error("polygon vertices must form a non-degenerate convex region in boundary order")]
    InvalidPolygon,
}

/// An in-memory object catalog.
#[derive(Debug, Default)]
pub struct ObjectCatalog {
    objects: Vec<SkyObject>,
    trailing_bytes: u64,
}

impl ObjectCatalog {
    pub fn new(objects: Vec<SkyObject>) -> Self {
        Self {
            objects,
            trailing_bytes: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    pub fn objects(&self) -> &[SkyObject] {
        &self.objects
    }

    /// Write the current, extensible `SEIZAOB2` format.
    pub fn write_to(&self, path: &Path) -> io::Result<()> {
        let mut out = BufWriter::new(File::create(path)?);
        write_header(&mut out, MAGIC_V2, self.objects.len())?;
        for o in &self.objects {
            let record_len = v2_record_len(o)?;
            let mut record = Vec::with_capacity(record_len as usize);
            write_v1_object(&mut record, o)?;
            write_string(&mut record, &o.metadata.id)?;
            write_string(&mut record, &o.metadata.source)?;
            write_string_list(&mut record, &o.metadata.aliases)?;
            write_string_list(&mut record, &o.metadata.parent_ids)?;
            write_string_list(&mut record, &o.metadata.alternate_ids)?;
            write_string_list(&mut record, &o.metadata.alternate_sources)?;
            debug_assert_eq!(record.len(), record_len as usize);
            out.write_all(&record_len.to_le_bytes())?;
            out.write_all(&record)?;
        }
        out.flush()
    }

    /// Write the legacy `SEIZAOB1` format.
    ///
    /// Identity, hierarchy, and provenance metadata cannot be represented and
    /// is omitted. This is intended only for controlled compatibility exports.
    pub fn write_v1_to(&self, path: &Path) -> io::Result<()> {
        let mut out = BufWriter::new(File::create(path)?);
        write_header(&mut out, MAGIC_V1, self.objects.len())?;
        for object in &self.objects {
            write_v1_object(&mut out, object)?;
        }
        out.flush()
    }

    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut input = BufReader::new(file);
        let mut magic = [0u8; 8];
        input.read_exact(&mut magic)?;
        let is_v2 = if &magic == MAGIC_V2 {
            true
        } else if &magic == MAGIC_V1 {
            false
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a seiza object catalog",
            ));
        };
        let count = read_u32(&mut input)?;
        let minimum_record_bytes = if is_v2 { 53 } else { 37 };
        if count as u64 > file_len.saturating_sub(12) / minimum_record_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "object count cannot fit in the catalog file",
            ));
        }
        let mut objects = Vec::with_capacity(count as usize);
        if is_v2 {
            for _ in 0..count {
                let record_len = read_u32(&mut input)?;
                if record_len > MAX_RECORD_BYTES {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "object record exceeds the 16 MiB safety limit",
                    ));
                }
                let mut record = vec![0u8; record_len as usize];
                input.read_exact(&mut record)?;
                let mut record = io::Cursor::new(record);
                let mut object = read_v1_object(&mut record)?;
                object.metadata.id = read_string(&mut record)?;
                object.metadata.source = read_string(&mut record)?;
                object.metadata.aliases = read_string_list(&mut record)?;
                object.metadata.parent_ids = read_string_list(&mut record)?;
                object.metadata.alternate_ids = read_string_list(&mut record)?;
                object.metadata.alternate_sources = read_string_list(&mut record)?;
                // Any remaining bytes belong to a future v2 extension.
                objects.push(object);
            }
        } else {
            for _ in 0..count {
                objects.push(read_v1_object(&mut input)?);
            }
        }
        let trailing_bytes = file_len.saturating_sub(input.stream_position()?);
        Ok(Self {
            objects,
            trailing_bytes,
        })
    }

    /// Validate catalog-wide semantic invariants. Opening only performs the
    /// structural decoding needed to construct the in-memory catalog; this
    /// explicit pass checks coordinates, measurements, and stable-ID
    /// uniqueness.
    pub fn validate(&self) -> io::Result<()> {
        if self.trailing_bytes != 0 {
            return Err(invalid_object_data("object catalog has trailing bytes"));
        }
        let mut ids = std::collections::HashSet::new();
        for object in &self.objects {
            if object.name.trim().is_empty() {
                return Err(invalid_object_data("object has an empty name"));
            }
            if !object.ra.is_finite()
                || !object.dec.is_finite()
                || !(-90.0..=90.0).contains(&object.dec)
            {
                return Err(invalid_object_data("object has invalid coordinates"));
            }
            if object.mag.is_some_and(|value| !value.is_finite())
                || object
                    .major_arcmin
                    .is_some_and(|value| !value.is_finite() || value < 0.0)
                || object
                    .minor_arcmin
                    .is_some_and(|value| !value.is_finite() || value < 0.0)
                || object
                    .position_angle_deg
                    .is_some_and(|value| !value.is_finite())
            {
                return Err(invalid_object_data("object has invalid measurements"));
            }
            if !object.metadata.id.is_empty() && !ids.insert(object.metadata.id.as_str()) {
                return Err(invalid_object_data(
                    "object catalog has duplicate stable IDs",
                ));
            }
        }
        Ok(())
    }

    /// Query catalog objects using known sky coordinates, without detecting
    /// stars or plate-solving an image.
    ///
    /// An object's angular extent is treated as a conservative circle using
    /// its major-axis radius for boundary intersection. The original ellipse
    /// metadata remains available on [`ObjectHit::object`].
    pub fn query_region<'a>(
        &'a self,
        region: &SkyRegion,
        query: &ObjectQuery,
    ) -> Result<Vec<ObjectHit<'a>>, ObjectQueryError> {
        let region = PreparedRegion::new(region)?;
        let mut hits: Vec<_> = self
            .objects
            .iter()
            .filter(|object| {
                (query.kinds.is_empty() || query.kinds.contains(&object.kind))
                    && query
                        .max_mag
                        .is_none_or(|max| object.mag.is_some_and(|mag| mag <= max))
                    && query
                        .min_major_arcmin
                        .is_none_or(|min| object.major_arcmin.is_some_and(|major| major >= min))
                    && (!query.common_name_only || !object.common_name.is_empty())
            })
            .filter_map(|object| {
                let point = UnitVector::from_radec(object.ra, object.dec)?;
                let extent_radius_deg = object.major_arcmin.unwrap_or(0.0) as f64 / 120.0;
                let distance_from_center_deg = region.distance_from_center_deg(point);
                let query_reach = if query.include_extent_overlaps {
                    extent_radius_deg
                } else {
                    0.0
                };
                // Every supported region lies inside this cap. Rejecting
                // distant objects here avoids polygon edge calculations for
                // nearly the entire all-sky catalog.
                if distance_from_center_deg
                    > region.characteristic_radius_deg() + query_reach + 1e-10
                {
                    return None;
                }
                let center_inside = region.contains(point);
                let intersects = center_inside
                    || (query.include_extent_overlaps
                        && extent_radius_deg > 0.0
                        && region.distance_to_boundary_deg(point) <= extent_radius_deg);
                if !intersects {
                    return None;
                }
                Some(ObjectHit {
                    object,
                    center_inside,
                    extent_only: !center_inside,
                    distance_from_center_deg,
                    predicted_prominence: predicted_prominence(
                        object,
                        center_inside,
                        region.characteristic_radius_deg(),
                    ),
                })
            })
            .collect();

        sort_hits(&mut hits, query.sort);
        if let Some(limit) = query.limit {
            hits.truncate(limit);
        }
        Ok(hits)
    }

    /// Objects whose extent intersects a solved image, with pixel geometry.
    /// Sorted large-to-small so prominent objects come first.
    pub fn objects_in_footprint(&self, wcs: &Wcs, dimensions: (u32, u32)) -> Vec<PlacedObject> {
        let (width, height) = (dimensions.0 as f64, dimensions.1 as f64);
        let scale = wcs.scale_arcsec_per_px();
        let mut placed: Vec<PlacedObject> = self
            .objects
            .iter()
            .filter_map(|o| {
                let (x, y) = wcs.world_to_pixel(o.ra, o.dec)?;
                let semi_major_px = o.major_arcmin.unwrap_or(0.0) as f64 * 60.0 / 2.0 / scale;
                let semi_minor_px = match o.minor_arcmin {
                    Some(minor) => minor as f64 * 60.0 / 2.0 / scale,
                    None => semi_major_px,
                };
                let margin = semi_major_px.max(1.0);
                if x < -margin || y < -margin || x >= width + margin || y >= height + margin {
                    return None;
                }
                let angle_deg = major_axis_image_angle(wcs, o, x, y);
                Some(PlacedObject {
                    object: o.clone(),
                    x,
                    y,
                    semi_major_px,
                    semi_minor_px,
                    angle_deg,
                })
            })
            .collect();
        placed.sort_by(|a, b| b.semi_major_px.total_cmp(&a.semi_major_px));
        placed
    }
}

fn sort_hits(hits: &mut [ObjectHit<'_>], sort: ObjectSort) {
    hits.sort_by(|a, b| {
        let primary = match sort {
            ObjectSort::Prominence => b.predicted_prominence.total_cmp(&a.predicted_prominence),
            ObjectSort::Size => b
                .object
                .major_arcmin
                .unwrap_or(0.0)
                .total_cmp(&a.object.major_arcmin.unwrap_or(0.0)),
            ObjectSort::Magnitude => a
                .object
                .mag
                .unwrap_or(f32::INFINITY)
                .total_cmp(&b.object.mag.unwrap_or(f32::INFINITY)),
            ObjectSort::Distance => a
                .distance_from_center_deg
                .total_cmp(&b.distance_from_center_deg),
            ObjectSort::Name => a.object.name.cmp(&b.object.name),
        };
        primary.then_with(|| {
            b.object
                .major_arcmin
                .unwrap_or(0.0)
                .total_cmp(&a.object.major_arcmin.unwrap_or(0.0))
        })
    });
}

fn predicted_prominence(object: &SkyObject, center_inside: bool, region_radius_deg: f64) -> f64 {
    let frame_diameter_arcmin = (region_radius_deg * 120.0).max(1e-9);
    let size_score = object
        .major_arcmin
        .map(|major| 1.0 - (-2.0 * major as f64 / frame_diameter_arcmin).exp())
        .unwrap_or(0.0);
    let magnitude_score = object
        .mag
        .map(|mag| ((16.0 - mag as f64) / 16.0).clamp(0.0, 1.0))
        .unwrap_or(0.0);
    let common_name_score = if object.common_name.is_empty() {
        0.0
    } else {
        1.0
    };
    let placement = if center_inside { 1.0 } else { 0.7 };
    ((0.65 * size_score + 0.30 * magnitude_score + 0.05 * common_name_score) * placement)
        .clamp(0.0, 1.0)
}

#[derive(Debug, Clone)]
enum PreparedRegion {
    Cone {
        center: UnitVector,
        radius_deg: f64,
    },
    Polygon {
        center: UnitVector,
        characteristic_radius_deg: f64,
        vertices: Vec<UnitVector>,
    },
}

impl PreparedRegion {
    fn new(region: &SkyRegion) -> Result<Self, ObjectQueryError> {
        match region {
            SkyRegion::Cone { center, radius_deg } => {
                let center = UnitVector::from_radec(center.0, center.1).ok_or(
                    ObjectQueryError::InvalidCoordinate {
                        ra: center.0,
                        dec: center.1,
                    },
                )?;
                if !radius_deg.is_finite() || *radius_deg <= 0.0 || *radius_deg > 180.0 {
                    return Err(ObjectQueryError::InvalidRadius(*radius_deg));
                }
                Ok(Self::Cone {
                    center,
                    radius_deg: *radius_deg,
                })
            }
            SkyRegion::Polygon { vertices } => {
                if vertices.len() < 3 {
                    return Err(ObjectQueryError::TooFewVertices(vertices.len()));
                }
                let vertices: Vec<_> = vertices
                    .iter()
                    .map(|&(ra, dec)| {
                        UnitVector::from_radec(ra, dec)
                            .ok_or(ObjectQueryError::InvalidCoordinate { ra, dec })
                    })
                    .collect::<Result<_, _>>()?;
                let center = vertices
                    .iter()
                    .copied()
                    .fold(UnitVector::ZERO, UnitVector::add)
                    .normalized()
                    .ok_or(ObjectQueryError::InvalidPolygon)?;
                let characteristic_radius_deg = vertices
                    .iter()
                    .map(|&vertex| center.angle_deg(vertex))
                    .fold(0.0, f64::max);
                if !characteristic_radius_deg.is_finite()
                    || characteristic_radius_deg <= 0.0
                    || characteristic_radius_deg >= 90.0
                {
                    return Err(ObjectQueryError::InvalidPolygon);
                }
                // A convex polygon in boundary order keeps the interior point
                // on the same side of every great-circle edge.
                for index in 0..vertices.len() {
                    let edge = vertices[index].cross(vertices[(index + 1) % vertices.len()]);
                    let center_side = edge.dot(center);
                    if edge.norm() <= 1e-12 || center_side.abs() <= 1e-12 {
                        return Err(ObjectQueryError::InvalidPolygon);
                    }
                    for (vertex_index, &vertex) in vertices.iter().enumerate() {
                        if vertex_index != index
                            && vertex_index != (index + 1) % vertices.len()
                            && edge.dot(vertex) * center_side < -1e-12
                        {
                            return Err(ObjectQueryError::InvalidPolygon);
                        }
                    }
                }
                Ok(Self::Polygon {
                    center,
                    characteristic_radius_deg,
                    vertices,
                })
            }
        }
    }

    fn center(&self) -> UnitVector {
        match self {
            Self::Cone { center, .. } | Self::Polygon { center, .. } => *center,
        }
    }

    fn characteristic_radius_deg(&self) -> f64 {
        match self {
            Self::Cone { radius_deg, .. } => *radius_deg,
            Self::Polygon {
                characteristic_radius_deg,
                ..
            } => *characteristic_radius_deg,
        }
    }

    fn distance_from_center_deg(&self, point: UnitVector) -> f64 {
        self.center().angle_deg(point)
    }

    fn contains(&self, point: UnitVector) -> bool {
        match self {
            Self::Cone { center, radius_deg } => center.angle_deg(point) <= *radius_deg + 1e-10,
            Self::Polygon {
                center, vertices, ..
            } => (0..vertices.len()).all(|index| {
                let edge = vertices[index].cross(vertices[(index + 1) % vertices.len()]);
                edge.dot(point) * edge.dot(*center) >= -1e-12
            }),
        }
    }

    fn distance_to_boundary_deg(&self, point: UnitVector) -> f64 {
        match self {
            Self::Cone { center, radius_deg } => (center.angle_deg(point) - radius_deg).abs(),
            Self::Polygon { vertices, .. } => (0..vertices.len())
                .map(|index| {
                    distance_to_arc_deg(
                        point,
                        vertices[index],
                        vertices[(index + 1) % vertices.len()],
                    )
                })
                .fold(f64::INFINITY, f64::min),
        }
    }
}

/// Minimum angular distance from a point to the minor great-circle arc AB.
fn distance_to_arc_deg(point: UnitVector, a: UnitVector, b: UnitVector) -> f64 {
    let mut best = point.angle_deg(a).min(point.angle_deg(b));
    let arc_angle = a.angle_deg(b);
    let Some(normal) = a.cross(b).normalized() else {
        return best;
    };
    let projected = point.add(normal.scale(-point.dot(normal)));
    let Some(projected) = projected.normalized() else {
        return best;
    };
    for candidate in [projected, projected.scale(-1.0)] {
        let along = a.angle_deg(candidate) + candidate.angle_deg(b);
        if (along - arc_angle).abs() <= 1e-7 {
            best = best.min(point.angle_deg(candidate));
        }
    }
    best
}

#[derive(Debug, Clone, Copy)]
struct UnitVector(f64, f64, f64);

impl UnitVector {
    const ZERO: Self = Self(0.0, 0.0, 0.0);

    fn from_radec(ra: f64, dec: f64) -> Option<Self> {
        if !ra.is_finite() || !dec.is_finite() || !(-90.0..=90.0).contains(&dec) {
            return None;
        }
        let (sin_ra, cos_ra) = ra.rem_euclid(360.0).to_radians().sin_cos();
        let (sin_dec, cos_dec) = dec.to_radians().sin_cos();
        Some(Self(cos_dec * cos_ra, cos_dec * sin_ra, sin_dec))
    }

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0, self.1 + other.1, self.2 + other.2)
    }

    fn scale(self, factor: f64) -> Self {
        Self(self.0 * factor, self.1 * factor, self.2 * factor)
    }

    fn dot(self, other: Self) -> f64 {
        self.0 * other.0 + self.1 * other.1 + self.2 * other.2
    }

    fn cross(self, other: Self) -> Self {
        Self(
            self.1 * other.2 - self.2 * other.1,
            self.2 * other.0 - self.0 * other.2,
            self.0 * other.1 - self.1 * other.0,
        )
    }

    fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    fn normalized(self) -> Option<Self> {
        let norm = self.norm();
        (norm > 1e-15).then(|| self.scale(1.0 / norm))
    }

    fn angle_deg(self, other: Self) -> f64 {
        self.dot(other).clamp(-1.0, 1.0).acos().to_degrees()
    }
}

/// Image-frame rotation of an object's major axis: its position angle is
/// degrees east of north on the sky, so build the north and east directions
/// at the object by finite differences and rotate.
fn major_axis_image_angle(wcs: &Wcs, object: &SkyObject, x: f64, y: f64) -> f64 {
    let pa = object.position_angle_deg.unwrap_or(0.0) as f64;
    let eps = 1.0 / 60.0; // 1 arcmin
    let cos_dec = object.dec.to_radians().cos().max(1e-6);
    let north = wcs.world_to_pixel(object.ra, (object.dec + eps).min(90.0));
    let east = wcs.world_to_pixel(object.ra + eps / cos_dec, object.dec);
    let (Some(north), Some(east)) = (north, east) else {
        return 0.0;
    };
    let n = normalize((north.0 - x, north.1 - y));
    let e = normalize((east.0 - x, east.1 - y));
    let (sin_pa, cos_pa) = pa.to_radians().sin_cos();
    let dir = (n.0 * cos_pa + e.0 * sin_pa, n.1 * cos_pa + e.1 * sin_pa);
    // The major axis direction; express as CCW angle from +x in image coords
    dir.1.atan2(dir.0).to_degrees()
}

fn normalize(v: (f64, f64)) -> (f64, f64) {
    let len = v.0.hypot(v.1).max(1e-12);
    (v.0 / len, v.1 / len)
}

fn write_header(out: &mut impl Write, magic: &[u8; 8], count: usize) -> io::Result<()> {
    let count = u32::try_from(count)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "too many catalog objects"))?;
    out.write_all(magic)?;
    out.write_all(&count.to_le_bytes())
}

fn write_v1_object(out: &mut impl Write, object: &SkyObject) -> io::Result<()> {
    out.write_all(&[object.kind as u8])?;
    out.write_all(&object.ra.to_le_bytes())?;
    out.write_all(&object.dec.to_le_bytes())?;
    out.write_all(&object.mag.unwrap_or(f32::NAN).to_le_bytes())?;
    out.write_all(&object.major_arcmin.unwrap_or(0.0).to_le_bytes())?;
    out.write_all(&object.minor_arcmin.unwrap_or(0.0).to_le_bytes())?;
    out.write_all(&object.position_angle_deg.unwrap_or(f32::NAN).to_le_bytes())?;
    write_string(out, &object.name)?;
    write_string(out, &object.common_name)
}

fn invalid_object_data(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn read_v1_object(input: &mut impl Read) -> io::Result<SkyObject> {
    let mut kind = [0u8; 1];
    input.read_exact(&mut kind)?;
    let ra = read_f64(input)?;
    let dec = read_f64(input)?;
    let mag = read_f32(input)?;
    let major = read_f32(input)?;
    let minor = read_f32(input)?;
    let pa = read_f32(input)?;
    Ok(SkyObject {
        kind: ObjectKind::from_u8(kind[0]),
        ra,
        dec,
        mag: if mag.is_nan() { None } else { Some(mag) },
        major_arcmin: if major > 0.0 { Some(major) } else { None },
        minor_arcmin: if minor > 0.0 { Some(minor) } else { None },
        position_angle_deg: if pa.is_nan() { None } else { Some(pa) },
        name: read_string(input)?,
        common_name: read_string(input)?,
        metadata: ObjectMetadata::default(),
    })
}

fn v2_record_len(object: &SkyObject) -> io::Result<u32> {
    // Fixed numeric prefix plus four scalar strings and four string-list counts.
    let mut len = 33usize;
    for value in [
        &object.name,
        &object.common_name,
        &object.metadata.id,
        &object.metadata.source,
    ] {
        len = len
            .checked_add(encoded_string_len(value)?)
            .ok_or_else(record_too_large)?;
    }
    for values in [
        &object.metadata.aliases,
        &object.metadata.parent_ids,
        &object.metadata.alternate_ids,
        &object.metadata.alternate_sources,
    ] {
        u16::try_from(values.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "object catalog string list exceeds 65535 entries",
            )
        })?;
        len = len.checked_add(2).ok_or_else(record_too_large)?;
        for value in values {
            len = len
                .checked_add(encoded_string_len(value)?)
                .ok_or_else(record_too_large)?;
        }
    }
    let len = u32::try_from(len).map_err(|_| record_too_large())?;
    if len > MAX_RECORD_BYTES {
        return Err(record_too_large());
    }
    Ok(len)
}

fn encoded_string_len(value: &str) -> io::Result<usize> {
    u16::try_from(value.len()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "object catalog string exceeds 65535 bytes",
        )
    })?;
    Ok(2 + value.len())
}

fn record_too_large() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "object record exceeds the 16 MiB safety limit",
    )
}

fn write_string(out: &mut impl Write, s: &str) -> io::Result<()> {
    let bytes = s.as_bytes();
    let len = u16::try_from(bytes.len()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "object catalog string exceeds 65535 bytes",
        )
    })?;
    out.write_all(&len.to_le_bytes())?;
    out.write_all(bytes)
}

fn read_string(input: &mut impl Read) -> io::Result<String> {
    let mut len = [0u8; 2];
    input.read_exact(&mut len)?;
    let mut buf = vec![0u8; u16::from_le_bytes(len) as usize];
    input.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn write_string_list(out: &mut impl Write, values: &[String]) -> io::Result<()> {
    let count = u16::try_from(values.len()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "object catalog string list exceeds 65535 entries",
        )
    })?;
    out.write_all(&count.to_le_bytes())?;
    for value in values {
        write_string(out, value)?;
    }
    Ok(())
}

fn read_string_list(input: &mut impl Read) -> io::Result<Vec<String>> {
    let mut count = [0u8; 2];
    input.read_exact(&mut count)?;
    let count = u16::from_le_bytes(count) as usize;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(read_string(input)?);
    }
    Ok(values)
}

fn read_u32(input: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    input.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_f32(input: &mut impl Read) -> io::Result<f32> {
    let mut b = [0u8; 4];
    input.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn read_f64(input: &mut impl Read) -> io::Result<f64> {
    let mut b = [0u8; 8];
    input.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m31() -> SkyObject {
        SkyObject {
            kind: ObjectKind::Galaxy,
            ra: 10.684793,
            dec: 41.269065,
            mag: Some(3.44),
            major_arcmin: Some(177.83),
            minor_arcmin: Some(69.66),
            position_angle_deg: Some(35.0),
            name: "NGC 224".to_string(),
            common_name: "Andromeda Galaxy".to_string(),
            metadata: ObjectMetadata {
                id: "openngc:NGC224".to_string(),
                source: "OpenNGC".to_string(),
                aliases: vec!["M 31".to_string(), "PGC 2557".to_string()],
                parent_ids: vec!["curated:local-group".to_string()],
                alternate_ids: vec!["messier:M31".to_string()],
                alternate_sources: vec!["Messier catalog".to_string()],
            },
        }
    }

    fn test_object(name: &str, ra: f64, dec: f64) -> SkyObject {
        SkyObject {
            kind: ObjectKind::Nebula,
            ra,
            dec,
            mag: None,
            major_arcmin: None,
            minor_arcmin: None,
            position_angle_deg: None,
            name: name.to_string(),
            common_name: String::new(),
            metadata: ObjectMetadata::default(),
        }
    }

    #[test]
    fn cone_query_distinguishes_centers_from_extent_only_hits() {
        let inside = test_object("Inside", 0.0, 0.0);
        let mut overlapping = test_object("Overlapping", 1.2, 0.0);
        // 30' major axis => 0.25-degree radius, reaching into a 1-degree cone.
        overlapping.major_arcmin = Some(30.0);
        let outside = test_object("Outside", 3.0, 0.0);
        let catalog = ObjectCatalog::new(vec![inside, overlapping, outside]);
        let region = SkyRegion::Cone {
            center: (0.0, 0.0),
            radius_deg: 1.0,
        };

        let hits = catalog
            .query_region(&region, &ObjectQuery::default())
            .unwrap();
        assert_eq!(hits.len(), 2);
        let center = hits.iter().find(|hit| hit.object.name == "Inside").unwrap();
        assert!(center.center_inside);
        assert!(!center.extent_only);
        let extent = hits
            .iter()
            .find(|hit| hit.object.name == "Overlapping")
            .unwrap();
        assert!(!extent.center_inside);
        assert!(extent.extent_only);

        let centers_only = ObjectQuery {
            include_extent_overlaps: false,
            ..Default::default()
        };
        let hits = catalog.query_region(&region, &centers_only).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].object.name, "Inside");
    }

    #[test]
    fn polygon_query_handles_ra_wrap() {
        let inside = test_object("At zero", 0.0, 0.0);
        let outside = test_object("Far away", 10.0, 0.0);
        let catalog = ObjectCatalog::new(vec![inside, outside]);
        let region = SkyRegion::Polygon {
            vertices: vec![(359.0, -1.0), (1.0, -1.0), (1.0, 1.0), (359.0, 1.0)],
        };

        let hits = catalog
            .query_region(&region, &ObjectQuery::default())
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].object.name, "At zero");
        assert!(hits[0].center_inside);
        assert!(hits[0].distance_from_center_deg < 1e-6);
    }

    #[test]
    fn polygon_query_handles_polar_fields_and_extent_overlap() {
        let pole = test_object("Pole", 0.0, 90.0);
        let catalog = ObjectCatalog::new(vec![pole]);
        let polar_region = SkyRegion::Polygon {
            vertices: vec![(0.0, 89.0), (90.0, 89.0), (180.0, 89.0), (270.0, 89.0)],
        };
        let hits = catalog
            .query_region(&polar_region, &ObjectQuery::default())
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].center_inside);

        let mut overlap = test_object("Overlap", 1.2, 0.0);
        overlap.major_arcmin = Some(30.0);
        let catalog = ObjectCatalog::new(vec![overlap]);
        let box_region = SkyRegion::Polygon {
            vertices: vec![(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
        };
        let hits = catalog
            .query_region(&box_region, &ObjectQuery::default())
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].extent_only);
    }

    #[test]
    fn query_filters_and_prominence_sort_are_explicit() {
        let mut large = test_object("Large", 0.0, 0.0);
        large.major_arcmin = Some(120.0);
        let mut small_named = test_object("Small", 0.1, 0.0);
        small_named.major_arcmin = Some(1.0);
        small_named.mag = Some(10.0);
        small_named.common_name = "Named feature".to_string();
        let catalog = ObjectCatalog::new(vec![small_named, large]);
        let region = SkyRegion::Cone {
            center: (0.0, 0.0),
            radius_deg: 5.0,
        };

        let hits = catalog
            .query_region(&region, &ObjectQuery::default())
            .unwrap();
        assert_eq!(hits[0].object.name, "Large");
        assert!(hits[0].predicted_prominence > hits[1].predicted_prominence);

        let named_and_measured = ObjectQuery {
            max_mag: Some(12.0),
            common_name_only: true,
            ..Default::default()
        };
        let hits = catalog.query_region(&region, &named_and_measured).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].object.name, "Small");
    }

    #[test]
    fn query_rejects_invalid_regions() {
        let catalog = ObjectCatalog::default();
        let polygon = SkyRegion::Polygon {
            vertices: vec![(0.0, 0.0), (1.0, 0.0)],
        };
        assert!(matches!(
            catalog.query_region(&polygon, &ObjectQuery::default()),
            Err(ObjectQueryError::TooFewVertices(2))
        ));
        let cone = SkyRegion::Cone {
            center: (0.0, 91.0),
            radius_deg: 1.0,
        };
        assert!(matches!(
            catalog.query_region(&cone, &ObjectQuery::default()),
            Err(ObjectQueryError::InvalidCoordinate { .. })
        ));
        let concave = SkyRegion::Polygon {
            vertices: vec![
                (-2.0, -2.0),
                (2.0, -2.0),
                (0.0, 0.0),
                (2.0, 2.0),
                (-2.0, 2.0),
            ],
        };
        assert!(matches!(
            catalog.query_region(&concave, &ObjectQuery::default()),
            Err(ObjectQueryError::InvalidPolygon)
        ));
    }

    #[test]
    fn round_trips_through_the_file_format() {
        let dir = std::env::temp_dir().join(format!("seiza-obj-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("objects.bin");

        let sizeless = SkyObject {
            kind: ObjectKind::Star,
            ra: 101.28,
            dec: -16.72,
            mag: None,
            major_arcmin: None,
            minor_arcmin: None,
            position_angle_deg: None,
            name: "Sirius".to_string(),
            common_name: String::new(),
            metadata: ObjectMetadata::default(),
        };
        ObjectCatalog::new(vec![m31(), sizeless])
            .write_to(&path)
            .unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..8], MAGIC_V2);
        let catalog = ObjectCatalog::open(&path).unwrap();
        catalog.validate().unwrap();
        assert_eq!(catalog.len(), 2);
        let a = &catalog.objects()[0];
        assert_eq!(a.kind, ObjectKind::Galaxy);
        assert_eq!(a.common_name, "Andromeda Galaxy");
        assert_eq!(a.mag, Some(3.44));
        assert_eq!(a.position_angle_deg, Some(35.0));
        assert_eq!(a.metadata.id, "openngc:NGC224");
        assert_eq!(a.metadata.source, "OpenNGC");
        assert_eq!(a.metadata.aliases, ["M 31", "PGC 2557"]);
        assert_eq!(a.metadata.parent_ids, ["curated:local-group"]);
        assert_eq!(a.metadata.alternate_ids, ["messier:M31"]);
        assert_eq!(a.metadata.alternate_sources, ["Messier catalog"]);
        let b = &catalog.objects()[1];
        assert_eq!(b.mag, None);
        assert_eq!(b.major_arcmin, None);
        assert_eq!(b.position_angle_deg, None);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn open_defers_object_semantic_validation() {
        let dir = std::env::temp_dir().join(format!("seiza-obj-lazy-open-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("objects.bin");
        let mut invalid = test_object("Invalid", 10.0, 100.0);
        invalid.metadata.id = "test:invalid".to_string();
        ObjectCatalog::new(vec![invalid]).write_to(&path).unwrap();

        let catalog = ObjectCatalog::open(&path).unwrap();
        assert!(catalog.validate().is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn opens_v1_catalogs_with_empty_metadata() {
        let dir = std::env::temp_dir().join(format!("seiza-obj-v1-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("objects-v1.bin");
        ObjectCatalog::new(vec![m31()]).write_v1_to(&path).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..8], MAGIC_V1);
        let catalog = ObjectCatalog::open(&path).unwrap();
        catalog.validate().unwrap();
        let object = &catalog.objects()[0];
        assert_eq!(object.name, "NGC 224");
        assert_eq!(object.metadata, ObjectMetadata::default());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn v2_reader_ignores_unknown_record_tail() {
        let dir = std::env::temp_dir().join(format!("seiza-obj-tail-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("objects-v2.bin");
        ObjectCatalog::new(vec![m31()]).write_to(&path).unwrap();

        let mut bytes = std::fs::read(&path).unwrap();
        let record_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        bytes[12..16].copy_from_slice(&(record_len + 4).to_le_bytes());
        bytes.extend_from_slice(&[1, 2, 3, 4]);
        std::fs::write(&path, bytes).unwrap();

        let catalog = ObjectCatalog::open(&path).unwrap();
        assert_eq!(catalog.objects()[0].metadata.id, "openngc:NGC224");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn writer_rejects_oversized_metadata_without_truncation() {
        let dir = std::env::temp_dir().join(format!("seiza-obj-large-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("objects-v2.bin");
        let mut object = m31();
        object.metadata.id = "x".repeat(u16::MAX as usize + 1);

        let error = ObjectCatalog::new(vec![object])
            .write_to(&path)
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn footprint_projection_places_and_sizes_objects() {
        // 2"/px camera pointing at M31, north up
        let wcs = Wcs::from_center_scale_rotation(
            (10.684793, 41.269065),
            (2000.0, 1500.0),
            2.0,
            0.0,
            false,
        );
        let far = SkyObject {
            ra: 30.0,
            dec: 20.0,
            ..m31()
        };
        let catalog = ObjectCatalog::new(vec![m31(), far]);
        let placed = catalog.objects_in_footprint(&wcs, (4000, 3000));
        assert_eq!(placed.len(), 1);

        let p = &placed[0];
        assert!((p.x - 2000.0).abs() < 0.01);
        assert!((p.y - 1500.0).abs() < 0.01);
        // 177.83' major axis at 2"/px => semi-major = 177.83*60/2/2 px
        assert!((p.semi_major_px - 177.83 * 15.0).abs() < 1.0);
        assert!((p.semi_minor_px - 69.66 * 15.0).abs() < 1.0);
        // North up, east left: PA 35° E of N maps to 35° past "up" toward
        // "left" => angle from +x axis is -(90 + 35)... measured CCW in
        // y-down image coords: up is -y (angle -90°), east is -x (180°).
        // dir = n*cos35 + e*sin35 => atan2 in image coords:
        let expected = (-(35.0f64.to_radians().cos()))
            .atan2(-(35.0f64.to_radians().sin()))
            .to_degrees();
        let diff = ((p.angle_deg - expected).abs() + 180.0) % 360.0 - 180.0;
        assert!(diff.abs() < 0.5, "{} vs {expected}", p.angle_deg);
    }

    #[test]
    fn open_rejects_garbage() {
        let dir = std::env::temp_dir().join(format!("seiza-obj-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("garbage.bin");
        std::fs::write(&path, b"not an object catalog").unwrap();
        assert!(ObjectCatalog::open(&path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
