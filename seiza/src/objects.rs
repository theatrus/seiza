//! Deep-sky object and named-star catalogs, and their projection into a
//! solved image for overlays.
//!
//! Binary format `SEIZAOB1` (little-endian): magic, u32 count, then per
//! object: kind u8, ra f64, dec f64, mag f32 (NaN = unknown), major axis
//! f32 arcmin (0 = unknown), minor axis f32 arcmin, position angle f32
//! degrees E of N (NaN = unknown), then two length-prefixed (u16) UTF-8
//! strings: designation and common name.

use crate::wcs::Wcs;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"SEIZAOB1";

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

/// An in-memory object catalog.
#[derive(Debug, Default)]
pub struct ObjectCatalog {
    objects: Vec<SkyObject>,
}

impl ObjectCatalog {
    pub fn new(objects: Vec<SkyObject>) -> Self {
        Self { objects }
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

    pub fn write_to(&self, path: &Path) -> io::Result<()> {
        let mut out = BufWriter::new(File::create(path)?);
        out.write_all(MAGIC)?;
        out.write_all(&(self.objects.len() as u32).to_le_bytes())?;
        for o in &self.objects {
            out.write_all(&[o.kind as u8])?;
            out.write_all(&o.ra.to_le_bytes())?;
            out.write_all(&o.dec.to_le_bytes())?;
            out.write_all(&o.mag.unwrap_or(f32::NAN).to_le_bytes())?;
            out.write_all(&o.major_arcmin.unwrap_or(0.0).to_le_bytes())?;
            out.write_all(&o.minor_arcmin.unwrap_or(0.0).to_le_bytes())?;
            out.write_all(&o.position_angle_deg.unwrap_or(f32::NAN).to_le_bytes())?;
            write_string(&mut out, &o.name)?;
            write_string(&mut out, &o.common_name)?;
        }
        out.flush()
    }

    pub fn open(path: &Path) -> io::Result<Self> {
        let mut input = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 8];
        input.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a seiza object catalog",
            ));
        }
        let count = read_u32(&mut input)?;
        let mut objects = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let mut kind = [0u8; 1];
            input.read_exact(&mut kind)?;
            let ra = read_f64(&mut input)?;
            let dec = read_f64(&mut input)?;
            let mag = read_f32(&mut input)?;
            let major = read_f32(&mut input)?;
            let minor = read_f32(&mut input)?;
            let pa = read_f32(&mut input)?;
            let name = read_string(&mut input)?;
            let common_name = read_string(&mut input)?;
            objects.push(SkyObject {
                kind: ObjectKind::from_u8(kind[0]),
                ra,
                dec,
                mag: if mag.is_nan() { None } else { Some(mag) },
                major_arcmin: if major > 0.0 { Some(major) } else { None },
                minor_arcmin: if minor > 0.0 { Some(minor) } else { None },
                position_angle_deg: if pa.is_nan() { None } else { Some(pa) },
                name,
                common_name,
            });
        }
        Ok(Self { objects })
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

fn write_string(out: &mut impl Write, s: &str) -> io::Result<()> {
    let bytes = s.as_bytes();
    let len = bytes.len().min(u16::MAX as usize);
    out.write_all(&(len as u16).to_le_bytes())?;
    out.write_all(&bytes[..len])
}

fn read_string(input: &mut impl Read) -> io::Result<String> {
    let mut len = [0u8; 2];
    input.read_exact(&mut len)?;
    let mut buf = vec![0u8; u16::from_le_bytes(len) as usize];
    input.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
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
        }
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
        };
        ObjectCatalog::new(vec![m31(), sizeless])
            .write_to(&path)
            .unwrap();

        let catalog = ObjectCatalog::open(&path).unwrap();
        assert_eq!(catalog.len(), 2);
        let a = &catalog.objects()[0];
        assert_eq!(a.kind, ObjectKind::Galaxy);
        assert_eq!(a.common_name, "Andromeda Galaxy");
        assert_eq!(a.mag, Some(3.44));
        assert_eq!(a.position_angle_deg, Some(35.0));
        let b = &catalog.objects()[1];
        assert_eq!(b.mag, None);
        assert_eq!(b.major_arcmin, None);
        assert_eq!(b.position_angle_deg, None);

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
