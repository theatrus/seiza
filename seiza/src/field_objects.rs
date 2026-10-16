//! The catalogued objects in a solved image: deep-sky objects and named
//! stars, transients, and comets and asteroids, placed in its pixels with
//! the outlines the catalog holds for them.
//!
//! [`field_objects`] reads every catalog a catalog directory holds and says
//! why any it could not read gave nothing. The other functions work on one
//! open catalog each.

use crate::minor_bodies::{MinorBodyCatalog, MinorBodyKind};
use crate::objects::{
    GeometryData, GeometryQuality, GeometryRole, ObjectCatalog, ObjectGeometry, ObjectKind,
    ObjectQuery, SkyRegion,
};
use crate::wcs::Wcs;
use std::collections::HashMap;
use std::path::Path;

/// The faintest comet or asteroid listed, in visual magnitudes.
pub const MINOR_BODY_LIMIT_MAG: f64 = 18.0;

/// Why [`field_objects`] lists no comets or asteroids for an image with no
/// capture time.
pub const NO_CAPTURE_TIME: &str = "Solar-system positions require a FITS DATE-OBS acquisition time";

/// A catalogued object placed in a solved image.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldObject {
    /// The catalog's identifier, such as `openngc:NGC1976`; `None` for
    /// comets and asteroids.
    pub stable_id: Option<String>,
    pub name: String,
    /// A common name, or for a transient its catalog details, or for a
    /// comet or asteroid its magnitude and distance.
    pub common_name: String,
    /// The object's kind ([`ObjectKind::as_str`]), `"transient"` for
    /// anything a transient catalog lists, or `"comet"` or `"asteroid"`.
    pub kind: &'static str,
    /// The catalog it came from: `"deep_sky"` (deep-sky objects and named
    /// stars), `"transient"` or `"minor_body"`.
    pub source: &'static str,
    /// The source catalog the record cites, when it cites one.
    pub catalog_source: Option<String>,
    /// Position in pixels, counted from 0.
    pub x: f64,
    pub y: f64,
    /// The extent's half axes in pixels; 0 for a point.
    pub semi_major_px: f64,
    pub semi_minor_px: f64,
    /// The major axis's angle in the image, degrees.
    pub angle_deg: Option<f64>,
    /// How prominent the catalog expects the object to look in this field.
    pub prominence: Option<f64>,
    pub ra_deg: f64,
    pub dec_deg: f64,
    /// A transient's discovery date, `YYYY-MM-DD`.
    pub discovered: Option<String>,
    /// Whether a transient was found within a year before the capture (or
    /// 30 days after); true when either date is unknown. Always true for
    /// comets and asteroids, and `None` for everything else.
    pub near_capture: Option<bool>,
    /// A comet's or asteroid's distance from Earth, AU.
    pub distance_au: Option<f64>,
    /// A comet's or asteroid's speed against the stars, arcseconds per hour.
    pub motion_arcsec_per_hour: Option<f64>,
    /// A comet's tail or an asteroid's trail: its position angle on the sky,
    /// degrees east of north, and its angle in the image, degrees from the
    /// +x axis towards +y.
    pub direction_position_angle_deg: Option<f64>,
    pub direction_image_angle_deg: Option<f64>,
    /// The object's outlines in pixels.
    pub outlines: Vec<FieldOutline>,
}

impl FieldObject {
    /// The overlay layer the object belongs to: `"named_stars"`,
    /// `"transients"`, `"minor_bodies"` or `"deep_sky"`.
    pub fn layer(&self) -> &'static str {
        match self.kind {
            "star" | "double-star" => "named_stars",
            "transient" => "transients",
            "comet" | "asteroid" => "minor_bodies",
            _ => "deep_sky",
        }
    }
}

/// One of an object's outlines from the catalog, projected into pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldOutline {
    pub geometry_id: String,
    pub source_record_id: String,
    pub role: GeometryRole,
    pub quality: GeometryQuality,
    /// How the outline was made.
    pub method: String,
    /// What the outline rests on.
    pub evidence: String,
    /// The brightness level it traces, when it traces one.
    pub level: Option<String>,
    pub contours: Vec<FieldContour>,
}

/// One closed or open line of an outline, in pixels.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FieldContour {
    pub closed: bool,
    pub points: Vec<(f64, f64)>,
}

/// What [`field_objects`] found, and why any catalog gave nothing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FieldObjects {
    pub objects: Vec<FieldObject>,
    /// Why the deep-sky and named-star catalog (`objects.bin`) could not
    /// be read.
    pub object_catalog_error: Option<String>,
    /// Why the transient catalog (`transients.bin`) could not be read.
    pub transient_catalog_error: Option<String>,
    /// Why no comets or asteroids were placed: no minor-body catalog
    /// (`minor-bodies.bin`), or no capture time ([`NO_CAPTURE_TIME`]).
    pub minor_body_error: Option<String>,
}

/// The catalogued objects in an image `dimensions` pixels wide and high
/// that `wcs` solves, from the catalogs in `catalog_directory` (or the
/// standard places, as [`crate::data_paths`] finds them). `capture_time`,
/// ISO 8601 in UTC, places comets and asteroids and dates transients; with
/// none, no comets or asteroids are placed.
///
/// A catalog that is missing or cannot be read gives no objects and an
/// error in its field of the result; the others still give theirs. Deep-sky
/// objects and named stars come first, then transients, then comets and
/// asteroids.
pub fn field_objects(
    catalog_directory: Option<&Path>,
    wcs: &Wcs,
    dimensions: (u32, u32),
    capture_time: Option<&str>,
) -> FieldObjects {
    let capture_jd = capture_time.and_then(capture_julian_date);
    let mut found = FieldObjects::default();

    let objects = crate::data_paths::objects(catalog_directory)
        .map_err(|error| error.to_string())
        .and_then(|path| {
            ObjectCatalog::open(&path)
                .map_err(|error| format!("failed to open {}: {error}", path.display()))
        })
        .and_then(|catalog| catalog_objects(&catalog, wcs, dimensions, capture_jd, false));
    match objects {
        Ok(objects) => found.objects.extend(objects),
        Err(error) => found.object_catalog_error = Some(error),
    }

    let transients = crate::data_paths::transients(catalog_directory)
        .map_err(|error| error.to_string())
        .and_then(|path| {
            ObjectCatalog::open(&path).map_err(|error| {
                format!(
                    "failed to open transient catalog {}: {error}",
                    path.display()
                )
            })
        })
        .and_then(|catalog| catalog_objects(&catalog, wcs, dimensions, capture_jd, true));
    match transients {
        Ok(objects) => found.objects.extend(objects),
        Err(error) => found.transient_catalog_error = Some(error),
    }

    let minor_bodies = crate::data_paths::minor_bodies(catalog_directory)
        .map_err(|error| error.to_string())
        .and_then(|path| {
            MinorBodyCatalog::open(&path).map_err(|error| {
                format!(
                    "failed to open minor-body catalog {}: {error}",
                    path.display()
                )
            })
        })
        .and_then(|catalog| {
            let jd = capture_jd.ok_or_else(|| NO_CAPTURE_TIME.to_string())?;
            Ok(minor_body_objects(&catalog, wcs, dimensions, jd))
        });
    match minor_bodies {
        Ok(objects) => found.objects.extend(objects),
        Err(error) => found.minor_body_error = Some(error),
    }
    found
}

/// The objects of one object catalog inside the image, largest first, with
/// their outlines. `as_transients` marks every one a transient, as for a
/// transient catalog, and dates each against `capture_jd` (a Julian date);
/// an object whose kind is transient is dated either way.
pub fn catalog_objects(
    catalog: &ObjectCatalog,
    wcs: &Wcs,
    dimensions: (u32, u32),
    capture_jd: Option<f64>,
    as_transients: bool,
) -> Result<Vec<FieldObject>, String> {
    let prominence_by_id: HashMap<String, f64> = catalog
        .query_region(
            &SkyRegion::Polygon {
                vertices: wcs.footprint(dimensions.0, dimensions.1).to_vec(),
            },
            &ObjectQuery::default(),
        )
        .map_err(|error| error.to_string())?
        .into_iter()
        .map(|hit| (hit.object.metadata.id, hit.predicted_prominence))
        .collect();
    let placed = catalog
        .objects_in_footprint(wcs, dimensions)
        .map_err(|error| error.to_string())?;
    Ok(placed
        .into_iter()
        .map(|placed| {
            let transient = as_transients || placed.object.kind == ObjectKind::Transient;
            let stable_id =
                (!placed.object.metadata.id.is_empty()).then(|| placed.object.metadata.id.clone());
            let prominence = stable_id
                .as_ref()
                .and_then(|id| prominence_by_id.get(id))
                .copied();
            let outlines = stable_id
                .as_deref()
                .and_then(|id| catalog.geometries(id).ok())
                .map(|geometries| project_outlines(geometries, wcs))
                .unwrap_or_default();
            let discovered = transient
                .then(|| transient_discovery_date(&placed.object.common_name))
                .flatten();
            let near_capture =
                transient.then(|| transient_near_capture(discovered.as_deref(), capture_jd));
            let catalog_source = (!placed.object.metadata.source.is_empty())
                .then(|| placed.object.metadata.source.clone());
            FieldObject {
                stable_id,
                name: placed.object.name,
                common_name: placed.object.common_name,
                kind: if as_transients {
                    "transient"
                } else {
                    placed.object.kind.as_str()
                },
                source: if transient { "transient" } else { "deep_sky" },
                catalog_source,
                x: placed.x,
                y: placed.y,
                semi_major_px: placed.semi_major_px,
                semi_minor_px: placed.semi_minor_px,
                angle_deg: placed.angle_deg,
                prominence,
                ra_deg: placed.object.ra,
                dec_deg: placed.object.dec,
                discovered,
                near_capture,
                distance_au: None,
                motion_arcsec_per_hour: None,
                direction_position_angle_deg: None,
                direction_image_angle_deg: None,
                outlines,
            }
        })
        .collect())
}

/// The comets and asteroids of a minor-body catalog inside the image at
/// Julian date `jd`, down to [`MINOR_BODY_LIMIT_MAG`].
pub fn minor_body_objects(
    catalog: &MinorBodyCatalog,
    wcs: &Wcs,
    dimensions: (u32, u32),
    jd: f64,
) -> Vec<FieldObject> {
    catalog
        .objects_in_footprint(wcs, dimensions, jd, MINOR_BODY_LIMIT_MAG)
        .into_iter()
        .map(|placed| FieldObject {
            stable_id: None,
            name: placed.body.name,
            common_name: format!("V~{:.1}, {:.2} AU", placed.mag, placed.delta_au),
            kind: match placed.body.kind {
                MinorBodyKind::Comet => "comet",
                MinorBodyKind::Asteroid => "asteroid",
            },
            source: "minor_body",
            catalog_source: None,
            x: placed.x,
            y: placed.y,
            semi_major_px: 0.0,
            semi_minor_px: 0.0,
            angle_deg: Some(0.0),
            prominence: None,
            ra_deg: placed.ra,
            dec_deg: placed.dec,
            discovered: None,
            near_capture: Some(true),
            distance_au: Some(placed.delta_au),
            motion_arcsec_per_hour: placed.motion_arcsec_per_hour,
            direction_position_angle_deg: placed.direction_pa_deg,
            direction_image_angle_deg: placed
                .direction_pa_deg
                .and_then(|angle| image_angle(wcs, placed.ra, placed.dec, angle)),
            outlines: Vec::new(),
        })
        .collect()
}

/// The outline sets among `geometries`, projected into pixels by `wcs`.
/// A contour with a vertex `wcs` cannot place, or with too few vertices (3
/// closed, 2 open), is left out, and so is an outline left with none.
pub fn project_outlines(geometries: Vec<ObjectGeometry>, wcs: &Wcs) -> Vec<FieldOutline> {
    geometries
        .into_iter()
        .filter_map(|geometry| {
            let GeometryData::OutlineSet { level, contours } = geometry.data else {
                return None;
            };
            let contours = contours
                .into_iter()
                .filter_map(|contour| {
                    let points = contour
                        .vertices
                        .into_iter()
                        .map(|(ra, dec)| wcs.world_to_pixel(ra, dec))
                        .collect::<Option<Vec<_>>>()?;
                    let minimum_points = if contour.closed { 3 } else { 2 };
                    (points.len() >= minimum_points).then_some(FieldContour {
                        closed: contour.closed,
                        points,
                    })
                })
                .collect::<Vec<_>>();
            (!contours.is_empty()).then_some(FieldOutline {
                geometry_id: geometry.id,
                source_record_id: geometry.source_record_id,
                role: geometry.role,
                quality: geometry.quality,
                method: geometry.method,
                evidence: geometry.evidence,
                level,
                contours,
            })
        })
        .collect()
}

/// The Julian date of an ISO 8601 time as FITS writes one:
/// `YYYY-MM-DD`, optionally followed by `Thh:mm:ss.sss` and a `Z`.
pub fn capture_julian_date(value: &str) -> Option<f64> {
    let value = value.trim().trim_end_matches('Z');
    let (date, clock) = value.split_once('T').unwrap_or((value, "0:0:0"));
    let mut date_parts = date.split('-');
    let year: i32 = date_parts.next()?.parse().ok()?;
    let month: u32 = date_parts.next()?.parse().ok()?;
    let day: u32 = date_parts.next()?.parse().ok()?;
    let mut clock_parts = clock.split(':');
    let hour: f64 = clock_parts.next()?.parse().ok()?;
    let minute: f64 = clock_parts.next().unwrap_or("0").parse().ok()?;
    let second: f64 = clock_parts.next().unwrap_or("0").parse().ok()?;
    let day_fraction = day as f64 + (hour + minute / 60.0 + second / 3_600.0) / 24.0;
    Some(crate::minor_bodies::julian_date(year, month, day_fraction))
}

/// A transient's discovery date from its catalog details (`..., disc.
/// 2020/01/01, ...`), as `YYYY-MM-DD`.
fn transient_discovery_date(details: &str) -> Option<String> {
    let value = details
        .split(", ")
        .find_map(|part| part.strip_prefix("disc. "))?;
    let mut parts = value.split('/');
    let year: i32 = parts.next()?.trim().parse().ok()?;
    let month: u32 = parts.next()?.trim().parse().ok()?;
    let day: u32 = parts.next()?.trim().parse().ok()?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    Some(format!("{year:04}-{month:02}-{day:02}"))
}

fn transient_near_capture(discovered: Option<&str>, capture_jd: Option<f64>) -> bool {
    let (Some(discovered), Some(capture_jd)) = (discovered, capture_jd) else {
        return true;
    };
    let Some(discovered_jd) = capture_julian_date(discovered) else {
        return true;
    };
    discovered_jd >= capture_jd - 365.0 && discovered_jd <= capture_jd + 30.0
}

/// A sky position angle at `(ra, dec)`, degrees east of north, as an angle
/// in the image, degrees from the +x axis towards +y.
fn image_angle(wcs: &Wcs, ra: f64, dec: f64, pa_deg: f64) -> Option<f64> {
    let (x, y) = wcs.world_to_pixel(ra, dec)?;
    let epsilon = 1.0 / 60.0;
    let north = wcs.world_to_pixel(ra, (dec + epsilon).min(90.0))?;
    let east = wcs.world_to_pixel(ra + epsilon / dec.to_radians().cos().abs().max(1e-6), dec)?;
    let normalize = |point: (f64, f64)| {
        let vector = (point.0 - x, point.1 - y);
        let length = vector.0.hypot(vector.1).max(1e-12);
        (vector.0 / length, vector.1 / length)
    };
    let north = normalize(north);
    let east = normalize(east);
    let (sin, cos) = pa_deg.to_radians().sin_cos();
    Some(
        (north.1 * cos + east.1 * sin)
            .atan2(north.0 * cos + east.0 * sin)
            .to_degrees(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::{ObjectContour, ObjectMetadata, SkyObject};

    #[test]
    fn outlines_are_projected_into_image_pixels() {
        let wcs = Wcs::from_center_scale_rotation((10.0, 20.0), (100.0, 100.0), 3.6, 0.0, false);
        let expected = [(30.0, 40.0), (70.0, 40.0), (50.0, 80.0)];
        let vertices = expected
            .iter()
            .map(|&(x, y)| wcs.pixel_to_world(x, y))
            .collect();
        let outlines = project_outlines(
            vec![ObjectGeometry {
                id: "openngc:NGC1#outline-1".into(),
                source_record_id: "openngc:NGC1".into(),
                role: GeometryRole::BrightnessLevel,
                quality: GeometryQuality::Catalog,
                method: "OpenNGC outline".into(),
                evidence: "survey isophote".into(),
                data: GeometryData::OutlineSet {
                    level: Some("1".into()),
                    contours: vec![ObjectContour {
                        closed: true,
                        vertices,
                    }],
                },
            }],
            &wcs,
        );

        assert_eq!(outlines.len(), 1);
        assert_eq!(outlines[0].role, GeometryRole::BrightnessLevel);
        assert_eq!(outlines[0].quality, GeometryQuality::Catalog);
        assert_eq!(outlines[0].method, "OpenNGC outline");
        assert_eq!(outlines[0].evidence, "survey isophote");
        assert_eq!(outlines[0].level.as_deref(), Some("1"));
        assert!(outlines[0].contours[0].closed);
        for (actual, expected) in outlines[0].contours[0].points.iter().zip(expected) {
            assert!((actual.0 - expected.0).abs() < 1e-6);
            assert!((actual.1 - expected.1).abs() < 1e-6);
        }
    }

    #[test]
    fn capture_times_read_as_julian_dates() {
        assert!(capture_julian_date("2025-07-20T12:34:56.5Z").is_some());
        assert!((capture_julian_date("2000-01-01T12:00:00").unwrap() - 2_451_545.0).abs() < 1e-9);
        assert!(capture_julian_date("not-a-date").is_none());
    }

    fn object(name: &str, common_name: &str, kind: ObjectKind) -> SkyObject {
        SkyObject {
            kind,
            ra: 10.0,
            dec: 20.0,
            mag: Some(4.0),
            major_arcmin: None,
            minor_arcmin: None,
            position_angle_deg: None,
            name: name.into(),
            common_name: common_name.into(),
            metadata: ObjectMetadata {
                id: format!("test:{name}"),
                source: "test-catalog".into(),
                aliases: Vec::new(),
                parent_ids: Vec::new(),
                alternate_ids: Vec::new(),
                alternate_sources: Vec::new(),
            },
        }
    }

    #[test]
    fn named_stars_stay_and_transients_are_dated() {
        let wcs = Wcs::from_center_scale_rotation((10.0, 20.0), (50.0, 50.0), 3.6, 0.0, false);
        let catalog = ObjectCatalog::new(vec![
            object("Sirius", "Dog Star", ObjectKind::Star),
            object("NGC 1", "Test Galaxy", ObjectKind::Galaxy),
        ]);
        let objects = catalog_objects(&catalog, &wcs, (100, 100), None, false).unwrap();
        assert_eq!(objects.len(), 2);
        let star = objects.iter().find(|object| object.kind == "star").unwrap();
        assert_eq!(star.layer(), "named_stars");
        assert_eq!(star.source, "deep_sky");
        assert_eq!(star.near_capture, None);
        let galaxy = objects
            .iter()
            .find(|object| object.kind == "galaxy")
            .unwrap();
        assert_eq!(galaxy.layer(), "deep_sky");
        assert_eq!(galaxy.catalog_source.as_deref(), Some("test-catalog"));

        let transients = ObjectCatalog::new(vec![object(
            "SN 2020abc",
            "disc. 2020/01/01",
            ObjectKind::Galaxy,
        )]);
        let capture = capture_julian_date("2025-07-20T12:00:00Z");
        let objects = catalog_objects(&transients, &wcs, (100, 100), capture, true).unwrap();
        assert_eq!(objects[0].kind, "transient");
        assert_eq!(objects[0].layer(), "transients");
        assert_eq!(objects[0].discovered.as_deref(), Some("2020-01-01"));
        assert_eq!(objects[0].near_capture, Some(false));
    }

    #[test]
    fn missing_catalogs_say_why() {
        let directory = tempfile::tempdir().unwrap();
        let wcs = Wcs::from_center_scale_rotation((10.0, 20.0), (50.0, 50.0), 3.6, 0.0, false);
        ObjectCatalog::new(vec![object("NGC 1", "Test Galaxy", ObjectKind::Galaxy)])
            .write_to(&directory.path().join("objects.bin"))
            .unwrap();

        let found = field_objects(Some(directory.path()), &wcs, (100, 100), None);
        assert_eq!(found.objects.len(), 1);
        assert_eq!(found.objects[0].name, "NGC 1");
        assert_eq!(found.object_catalog_error, None);
        assert!(found.transient_catalog_error.is_some());
        assert!(found.minor_body_error.is_some());
    }
}
