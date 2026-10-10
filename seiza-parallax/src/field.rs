//! The stars of a solved image with their distances: Gaia DR3 and
//! Hipparcos stars from the offline star distance file or the archives,
//! matched to the stars found in the image.

use crate::pipeline::{Error, Event};
use crate::scene::Star;
use seiza::Wcs;
use seiza_sources::{GaiaDistance, HipparcosStar};
use seiza_stars::PeakStar;
use std::path::{Path, PathBuf};

/// A field's Gaia stars and Hipparcos stars.
type CatalogueStars = (Vec<GaiaDistance>, Vec<HipparcosStar>);

/// Where the field's catalogue stars come from.
pub(crate) struct FieldSource<'a> {
    /// The offline star distance file, or `None` for the standard places.
    pub star_distances: Option<&'a Path>,
    /// The faintest Gaia G magnitude to match.
    pub gaia_max_mag: f32,
    /// Where fetched fields are kept, or `None` for Seiza's data directory.
    pub cache: Option<&'a Path>,
    /// Whether the archives may be asked when there is no offline file.
    pub online: bool,
}

/// The field's Gaia and Hipparcos stars, offline if the star distance file
/// is installed and goes deep enough, else from the archives.
pub(crate) fn catalogue_stars(
    source: &FieldSource,
    wcs: &Wcs,
    (width, height): (usize, usize),
    report: &mut dyn FnMut(Event),
) -> Result<CatalogueStars, Error> {
    if let Some(field) = offline_field(source, wcs, width, height, report)? {
        return Ok(field);
    }
    if !source.online {
        report(Event::Warning(
            "no star distance file deep enough for the field, and fetching online is off: \
             every star sits at one distance",
        ));
        return Ok((Vec::new(), Vec::new()));
    }
    Ok((
        gaia_field(source, wcs, width, height, report)?,
        hipparcos_field(source, wcs, width, height, report),
    ))
}

/// The sky circle around the image and its centre.
fn field(wcs: &Wcs, width: usize, height: usize) -> ((f64, f64), f64) {
    let centre = wcs.pixel_to_world(width as f64 / 2.0, height as f64 / 2.0);
    let radius = [
        (0.0, 0.0),
        (width as f64, 0.0),
        (0.0, height as f64),
        (width as f64, height as f64),
    ]
    .iter()
    .map(|&(x, y)| separation(centre, wcs.pixel_to_world(x, y)))
    .fold(0.0_f64, f64::max)
        * 1.02;
    (centre, radius)
}

fn separation(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (ra1, dec1) = (a.0.to_radians(), a.1.to_radians());
    let (ra2, dec2) = (b.0.to_radians(), b.1.to_radians());
    let haversine = ((dec2 - dec1) / 2.0).sin().powi(2)
        + dec1.cos() * dec2.cos() * ((ra2 - ra1) / 2.0).sin().powi(2);
    (2.0 * haversine.sqrt().asin()).to_degrees()
}

/// Where fetched Gaia and Hipparcos fields are kept by default, beside
/// Seiza's catalogs.
pub fn default_gaia_cache() -> PathBuf {
    let catalogs = seiza::data_paths::default_catalog_dir();
    catalogs
        .parent()
        .map_or_else(|| catalogs.clone(), Path::to_path_buf)
        .join("gaia-fields")
}

/// The field's Gaia and Hipparcos stars from the offline star distance
/// file, or `None` when there is none or it stops short of the magnitude
/// asked for.
fn offline_field(
    source: &FieldSource,
    wcs: &Wcs,
    width: usize,
    height: usize,
    report: &mut dyn FnMut(Event),
) -> Result<Option<CatalogueStars>, Error> {
    let Some(path) = seiza::data_paths::star_distances(source.star_distances)
        .map_err(|error| Error::Invalid(error.to_string()))?
    else {
        return Ok(None);
    };
    let catalog =
        seiza::catalog::StarDistanceCatalog::open(&path).map_err(|error| Error::Catalog {
            path: path.clone(),
            message: error.to_string(),
        })?;
    if catalog.max_mag() < source.gaia_max_mag {
        report(Event::Note(&format!(
            "{} holds Gaia stars to G {}, short of G {}; fetching online",
            path.display(),
            catalog.max_mag(),
            source.gaia_max_mag
        )));
        return Ok(None);
    }
    let (centre, radius) = field(wcs, width, height);
    let (mut gaia, mut hipparcos) = (Vec::new(), Vec::new());
    // Hipparcos stars whatever the Gaia magnitude limit, as online.
    for star in catalog.cone_search(centre.0, centre.1, radius, 99.0) {
        if star.hipparcos {
            hipparcos.push(HipparcosStar {
                hip: 0,
                ra: star.ra,
                dec: star.dec,
                // A parallax that gives back the distance, as precise as
                // the catalogue found it.
                parallax: star.distance_pc.map(|distance| 1000.0 / distance),
                parallax_error: Some(0.0),
                hp_mag: Some(star.mag),
                pmra: None,
                pmdec: None,
            });
        } else if star.mag <= source.gaia_max_mag {
            gaia.push(GaiaDistance {
                ra: star.ra,
                dec: star.dec,
                pmra: None,
                pmdec: None,
                g: star.mag,
                bp_rp: None,
                parallax: None,
                parallax_error: None,
                distance: star.distance_pc,
                distance_low: None,
                distance_high: None,
            });
        }
    }
    report(Event::Note(&format!(
        "{} Gaia and {} Hipparcos stars from {}",
        gaia.len(),
        hipparcos.len(),
        path.display()
    )));
    Ok(Some((gaia, hipparcos)))
}

fn gaia_field(
    source: &FieldSource,
    wcs: &Wcs,
    width: usize,
    height: usize,
    report: &mut dyn FnMut(Event),
) -> Result<Vec<GaiaDistance>, Error> {
    let gaia_error = |error: &dyn std::fmt::Display| Error::Gaia(error.to_string());
    let (centre, radius) = field(wcs, width, height);
    let cache = source
        .cache
        .map_or_else(default_gaia_cache, Path::to_path_buf);
    let max_mag = source.gaia_max_mag;
    let path = cache.join(format!(
        "gaia-dr3-distances-{:.2}{:+.2}-r{:.2}-g{}.csv",
        centre.0, centre.1, radius, max_mag
    ));
    if let Ok(csv) = std::fs::read_to_string(&path)
        && let Ok(stars) = seiza_sources::parse_gaia_distances(&csv)
    {
        report(Event::Note(&format!(
            "{} Gaia stars from {}",
            stars.len(),
            path.display()
        )));
        return Ok(stars);
    }
    let cones = cones(wcs, width, height);
    report(Event::Note(&format!(
        "fetching Gaia DR3 distances within {radius:.2} deg of ({:.4}, {:+.4}) in {} cone(s)",
        centre.0,
        centre.1,
        cones.len()
    )));
    let downloader = seiza_sources::SourceDownloader::new().map_err(|error| gaia_error(&error))?;
    // Each cone is kept as it arrives, so a run stopped part way, or one
    // whose last cone fails, does not fetch the others again.
    let cone_path = move |&(ra, dec, radius): &(f64, f64, f64)| {
        format!("gaia-dr3-distances-cone-{ra:.4}{dec:+.4}-r{radius:.4}-g{max_mag}.csv")
    };
    let _ = std::fs::create_dir_all(&cache);
    let mut bodies = Vec::new();
    let mut missing = Vec::new();
    for cone in cones {
        match std::fs::read_to_string(cache.join(cone_path(&cone))) {
            Ok(csv) if seiza_sources::parse_gaia_distances(&csv).is_ok() => bodies.push(csv),
            _ => missing.push(cone),
        }
    }
    let cone_cache = cache.clone();
    // The outer future runs on this thread, so it may report as it goes.
    let fetched = runtime()?.block_on(async {
        // A few archive queries at once.
        let mut pending = missing.into_iter();
        let mut running = tokio::task::JoinSet::new();
        let mut fetched = Vec::new();
        loop {
            while running.len() < 4
                && let Some(cone) = pending.next()
            {
                let downloader = downloader.clone();
                running.spawn(async move {
                    let (ra, dec, radius) = cone;
                    let csv = downloader
                        .gaia_distance_cone_csv(ra, dec, radius, max_mag)
                        .await;
                    (cone, csv)
                });
            }
            let Some(done) = running.join_next().await else {
                break;
            };
            let (cone, csv) = done.map_err(|error| gaia_error(&error))?;
            let csv = csv.map_err(|error| gaia_error(&error))?;
            let path = cone_cache.join(cone_path(&cone));
            let partial = path.with_extension("csv.partial");
            if std::fs::write(&partial, &csv).is_ok() {
                let _ = std::fs::rename(&partial, &path);
            }
            fetched.push(csv);
            report(Event::Note(&format!("  {} cone(s) fetched", fetched.len())));
        }
        Ok::<_, Error>(fetched)
    })?;
    bodies.extend(fetched);
    let csv = seiza_sources::merge_csv(&bodies);
    let stars = seiza_sources::parse_gaia_distances(&csv).map_err(|error| gaia_error(&error))?;
    if std::fs::create_dir_all(&cache).is_ok() {
        let partial = path.with_extension("csv.partial");
        if std::fs::write(&partial, &csv).is_ok() {
            let _ = std::fs::rename(&partial, &path);
        }
    }
    report(Event::Note(&format!("{} Gaia stars", stars.len())));
    Ok(stars)
}

/// Cones about 1.2 degrees in radius covering the image, on a grid of
/// image cells: `(ra, dec, radius)`.
pub(crate) fn cones(wcs: &Wcs, width: usize, height: usize) -> Vec<(f64, f64, f64)> {
    let scale_deg = wcs.scale_arcsec_per_px() / 3600.0;
    let cell = (1.6 / scale_deg).max(1.0);
    let columns = (width as f64 / cell).ceil().max(1.0) as usize;
    let rows = (height as f64 / cell).ceil().max(1.0) as usize;
    let (cell_w, cell_h) = (width as f64 / columns as f64, height as f64 / rows as f64);
    let mut cones = Vec::with_capacity(columns * rows);
    for row in 0..rows {
        for column in 0..columns {
            let (left, top) = (column as f64 * cell_w, row as f64 * cell_h);
            let centre = wcs.pixel_to_world(left + cell_w / 2.0, top + cell_h / 2.0);
            let radius = [
                (left, top),
                (left + cell_w, top),
                (left, top + cell_h),
                (left + cell_w, top + cell_h),
            ]
            .iter()
            .map(|&(x, y)| separation(centre, wcs.pixel_to_world(x, y)))
            .fold(0.0_f64, f64::max)
                * 1.02;
            cones.push((centre.0, centre.1, radius));
        }
    }
    cones
}

/// Hipparcos stars in the field, or none when VizieR does not answer: they
/// only fill in the brightest stars' distances.
fn hipparcos_field(
    source: &FieldSource,
    wcs: &Wcs,
    width: usize,
    height: usize,
    report: &mut dyn FnMut(Event),
) -> Vec<HipparcosStar> {
    let (centre, radius) = field(wcs, width, height);
    // Kept beside the Gaia fields, so a field is asked for once.
    let cache = source
        .cache
        .map_or_else(default_gaia_cache, Path::to_path_buf);
    let path = cache.join(format!(
        "hipparcos-{:.4}{:+.4}-r{:.4}.csv",
        centre.0, centre.1, radius
    ));
    if let Ok(csv) = std::fs::read_to_string(&path)
        && let Ok(stars) = seiza_sources::parse_hipparcos(&csv)
    {
        return stars;
    }
    let fetched = seiza_sources::SourceDownloader::new()
        .map_err(|error| error.to_string())
        .and_then(|downloader| {
            runtime()
                .map_err(|error| error.to_string())?
                .block_on(downloader.hipparcos_cone_csv(centre.0, centre.1, radius))
                .map_err(|error| error.to_string())
        })
        .and_then(|csv| {
            let stars = seiza_sources::parse_hipparcos(&csv).map_err(|error| error.to_string())?;
            let partial = path.with_extension("csv.partial");
            if std::fs::create_dir_all(&cache).is_ok() && std::fs::write(&partial, &csv).is_ok() {
                let _ = std::fs::rename(&partial, &path);
            }
            Ok(stars)
        });
    match fetched {
        Ok(stars) => stars,
        Err(error) => {
            report(Event::Warning(&format!(
                "no Hipparcos distances for the brightest stars: {error}"
            )));
            Vec::new()
        }
    }
}

fn runtime() -> Result<tokio::runtime::Runtime, Error> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| Error::Gaia(format!("could not start the network runtime: {error}")))
}

/// Each detection with the distance of the Gaia star at its place, nearest
/// first and each Gaia star used once, and how many found a Gaia star.
/// Gaia's brightest stars have no parallax; a Hipparcos star at the same
/// place gives theirs.
pub(crate) fn match_distances(
    detections: &[PeakStar],
    gaia: &[GaiaDistance],
    hipparcos: &[HipparcosStar],
    wcs: &Wcs,
    scale_arcsec: f64,
) -> (Vec<Star>, usize) {
    let projected: Vec<(f64, f64)> = gaia
        .iter()
        .map(|star| {
            wcs.world_to_pixel(star.ra, star.dec)
                .unwrap_or((f64::NAN, f64::NAN))
        })
        .collect();
    let grid = Grid::new(&projected, 16.0);
    let mut used = vec![false; gaia.len()];
    let mut stars = Vec::with_capacity(detections.len());
    let mut found = 0;
    for detection in detections {
        // A saturated star's centroid can sit a few pixels off its catalog
        // place; a faint one's should not.
        let footprint = (detection.area as f64 / std::f64::consts::PI).sqrt();
        let reach = (2.0_f64).max(2.0 / scale_arcsec) + footprint * 0.25;
        let candidates = grid
            .near(detection.x, detection.y, reach)
            .filter(|index| !used[*index])
            .map(|index| {
                let (x, y) = projected[index];
                (index, (x - detection.x).hypot(y - detection.y))
            })
            .filter(|(_, distance)| *distance <= reach);
        // A large, saturated star is the brightest catalog star under it; a
        // fainter neighbour may lie nearer its centroid.
        let chosen = if footprint >= 4.0 {
            candidates.min_by(|a, b| gaia[a.0].g.total_cmp(&gaia[b.0].g))
        } else {
            candidates.min_by(|a, b| a.1.total_cmp(&b.1))
        };
        let gaia_distance = chosen.and_then(|(index, _)| {
            used[index] = true;
            found += 1;
            let star = &gaia[index];
            star.best_distance().or_else(|| {
                hipparcos
                    .iter()
                    .filter(|hip| separation((hip.ra, hip.dec), (star.ra, star.dec)) * 3600.0 < 5.0)
                    .find_map(HipparcosStar::distance)
            })
        });
        // Gaia lists no parallax, or no position at all, for some of the
        // brightest stars; Hipparcos measured them.
        let distance_pc = gaia_distance.or_else(|| {
            if footprint < 4.0 {
                return None;
            }
            hipparcos
                .iter()
                .filter_map(|hip| {
                    let (x, y) = wcs.world_to_pixel(hip.ra, hip.dec)?;
                    let offset = (x - detection.x).hypot(y - detection.y);
                    (offset <= reach).then_some((offset, hip))
                })
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .and_then(|(_, hip)| hip.distance())
        });
        stars.push(Star {
            x: detection.x,
            y: detection.y,
            distance_pc,
        });
    }
    (stars, found)
}

/// Points binned into square cells for neighbour lookups.
struct Grid {
    cell: f64,
    cells: std::collections::HashMap<(i64, i64), Vec<usize>>,
}

impl Grid {
    fn new(points: &[(f64, f64)], cell: f64) -> Self {
        let mut cells: std::collections::HashMap<(i64, i64), Vec<usize>> =
            std::collections::HashMap::new();
        for (index, (x, y)) in points.iter().enumerate() {
            if x.is_finite() && y.is_finite() {
                cells
                    .entry(((x / cell).floor() as i64, (y / cell).floor() as i64))
                    .or_default()
                    .push(index);
            }
        }
        Self { cell, cells }
    }

    fn near(&self, x: f64, y: f64, reach: f64) -> impl Iterator<Item = usize> + '_ {
        let span = (reach / self.cell).ceil() as i64;
        let (cx, cy) = (
            (x / self.cell).floor() as i64,
            (y / self.cell).floor() as i64,
        );
        (cy - span..=cy + span)
            .flat_map(move |row| (cx - span..=cx + span).map(move |column| (column, row)))
            .filter_map(|key| self.cells.get(&key))
            .flatten()
            .copied()
    }
}

/// The median distance of the stars within a fifth of the image's
/// diagonal of the focus point.
pub(crate) fn median_distance_near(
    stars: &[Star],
    focus: (f64, f64),
    width: usize,
    height: usize,
) -> Option<f64> {
    let reach = (width as f64).hypot(height as f64) / 5.0;
    median(
        stars
            .iter()
            .filter(|star| (star.x - focus.0).hypot(star.y - focus.1) <= reach)
            .filter_map(|star| star.distance_pc),
    )
}

pub(crate) fn median(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut values: Vec<f64> = values.collect();
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[values.len() / 2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cones_cover_the_image_and_merged_rows_appear_once() {
        let wcs =
            Wcs::from_center_scale_rotation((56.75, 24.12), (3124.0, 2088.0), 4.47, 0.0, false);
        let cones = cones(&wcs, 6248, 4176);
        assert_eq!(cones.len(), 20);
        // Every corner of the image lies inside some cone.
        for (x, y) in [
            (0.0, 0.0),
            (6248.0, 0.0),
            (0.0, 4176.0),
            (6248.0, 4176.0),
            (3124.0, 2088.0),
        ] {
            let point = wcs.pixel_to_world(x, y);
            assert!(
                cones
                    .iter()
                    .any(|&(ra, dec, radius)| separation((ra, dec), point) <= radius)
            );
        }
        assert!(cones.iter().all(|cone| cone.2 < 1.3));
    }

    #[test]
    fn detections_take_the_nearest_unused_gaia_star() {
        let wcs = Wcs::from_center_scale_rotation((56.75, 24.12), (500.0, 500.0), 2.0, 0.0, false);
        let place = |dx: f64, dy: f64| wcs.pixel_to_world(500.0 + dx, 500.0 + dy);
        let gaia_star = |(ra, dec): (f64, f64), distance: Option<f64>| GaiaDistance {
            ra,
            dec,
            pmra: None,
            pmdec: None,
            g: 10.0,
            bp_rp: None,
            parallax: None,
            parallax_error: None,
            distance,
            distance_low: None,
            distance_high: None,
        };
        let gaia = [
            gaia_star(place(0.5, 0.0), Some(136.0)),
            gaia_star(place(40.0, 0.0), None),
        ];
        let hipparcos = [HipparcosStar {
            hip: 1,
            ra: place(40.0, 0.0).0,
            dec: place(40.0, 0.0).1,
            parallax: Some(10.0),
            parallax_error: Some(0.5),
            hp_mag: Some(3.0),
            pmra: None,
            pmdec: None,
        }];
        let detection = |x: f64, y: f64| PeakStar {
            x: 500.0 + x,
            y: 500.0 + y,
            flux: 100.0,
            area: 9,
        };
        let detections = [
            detection(0.0, 0.0),
            detection(40.3, 0.2),
            detection(200.0, 0.0),
        ];
        let (stars, found) = match_distances(&detections, &gaia, &hipparcos, &wcs, 2.0);
        assert_eq!(found, 2);
        assert_eq!(stars[0].distance_pc, Some(136.0));
        let hip = stars[1].distance_pc.unwrap();
        assert!((hip - 100.0).abs() < 1e-9, "{hip}");
        assert_eq!(stars[2].distance_pc, None, "no Gaia star there");
        assert_eq!(
            median_distance_near(&stars, (500.0, 500.0), 1000, 1000),
            Some(136.0)
        );
    }
}
